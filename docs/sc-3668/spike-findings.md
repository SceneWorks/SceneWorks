# sc-3668 — Port-or-drop AuraSR upscaler engine on Mac: path-selection spike

**Recommendation: DROP on Mac.** UI-gate the `engine=aura-sr` option out of the
Mac upscale UI (sc-3486 mechanism) and keep Real-ESRGAN x4 as the only Mac
upscaler. Keep AuraSR fully available on the Windows/Linux torch path. The two
engines are substitutable 4×-only upscalers; the measured quality delta is
marginal-and-subjective while the port cost is disproportionate.

Confidence: **~80%** (DROP). The residual uncertainty is purely subjective taste
for AuraSR's GAN-texture character on some real-world photos — which does not
move the cost/benefit, and which Win/Linux users still get.

> Context update (2026-06-11): the org has decided to **retire the ONNX/ort +
> CoreML path** and re-port Real-ESRGAN itself to native **MLX** (mlx-rs). So the
> real question here is **not** "can AuraSR ride Real-ESRGAN's CoreML rail" — it's
> **"is AuraSR worth an MLX port of its own, on top of the Real-ESRGAN MLX port
> that is happening anyway."** The ONNX/CoreML probe below is retained because its
> architecture findings (what makes AuraSR hard) transfer directly to the MLX
> port-cost estimate.

## What AuraSR is (the thing being considered for a port)

`fal/AuraSR-v2` (`apps/worker/scene_worker/upscalers.py:AuraSRUpscaler`,
`image_adapters.py:AuraSrUpscaler`) is a **GigaGAN** `UnetUpsampler` — a
reproduction of the GigaGAN paper based on lucidrains/gigagan-pytorch. It is a
fundamentally different and far heavier animal than Real-ESRGAN's RRDBNet:

| | Real-ESRGAN (ported) | AuraSR |
|---|---|---|
| arch | RRDBNet: plain conv stack | GigaGAN UNet: StyleGAN2 modulated convs + attention |
| params | ~17M | **617.6M** (fp32, ~2.4 GB, 268 tensors) |
| scale | 2× and 4× | 4× only |
| tiling | 512 px tiles (768² → **4** tiles) | 64 px tiles (768² → **144** tiles; ×2 for the overlapped 2-pass = 288) |
| determinism | deterministic | **stochastic** — random style vector `randn(b,128)` + `noise_aug = randn_like(x)·1e-3` per call |
| core op | `nn.Conv2d` (static weights) | `AdaptiveConv2DMod`: conv weights **computed at runtime** from a style vector, run via the batch-as-groups trick (`F.conv2d(..., groups=b)`) |

The defining op, `AdaptiveConv2DMod` (upscalers' `aura_sr.py:25-126`), does
StyleGAN2 weight modulation/demodulation, optional adaptive kernel selection
(softmax over `num_conv_kernels=4`), then a grouped conv whose **weight tensor is
a runtime activation, not a parameter.** Every `Block` uses one; the model is
built almost entirely out of them, interleaved with full + linear attention
transformers and a style MLP.

## The spike (reproducible artifacts under `scripts/spikes/`)

- `sc3668_aura_export_probe.py` — loads the real `fal/AuraSR-v2` weights, exports
  the `UnetUpsampler` to ONNX (batch 1, one 64² tile), and statically analyses the
  graph: how many `Conv` nodes have a **static (initializer)** weight vs a
  **dynamic (runtime activation)** weight, plus op histogram, faithfulness and
  latency. Mirrors the sc-3489 export probe.
- `sc3668_quality_ab.py` / `sc3668_quality_crops.py` — A/B AuraSR vs the shipped
  Real-ESRGAN x4 (the repo's own pure-torch RRDBNet from `upscalers.py`) on torch/
  MPS, on a real photographic image (the rusty-robot/candle scene), with
  texture-rich crops upscaled 4× for a visual side-by-side. **No CoreML, no MLX** —
  both run the torch path the models use today. Outputs in `/tmp/sc3668/`.

### Findings — why a cheap Rust path does not exist

The legacy TorchScript exporter **chokes on the stock model** (dynamic
shape-derived padding inside the adaptive conv) — it needs a patch just to trace.
Once traced (batch 1, one tile):

- **102 of 170 Conv nodes (60%) have dynamic, runtime-computed weights** — the
  adaptive modulated convs. A conv whose weight is an activation cannot be folded
  by CoreML (or most accelerators) into a standard `Conv`; it is the structural
  opposite of RRDBNet, which is 100% static-weight convs (and is precisely why
  sc-3489's CoreML path worked). This same property is what makes the **MLX** port
  hard: `mlx_rs::ops::conv2d` is groups=1 only, so the batch-as-groups adaptive
  conv needs a hand-rolled per-sample formulation.
- The traced graph is **4,945 nodes** of fragmented dynamic ops (692 Reshape, 142
  Transpose, 112 Softmax, 102× each of ConstantOfShape/Equal/Where/Sqrt/Sigmoid
  for the modulate–demodulate), with several **144 MB `Expand` constant tensors**
  (the adaptive-kernel `repeat(...)` materialised at trace time).
- At 617M params (>2 GB) the export spills to ONNX **external data** (183 external
  initializers across ~150 files), which the ORT CoreML EP refuses to initialise
  (`model_path must not be empty`) — and the traced graph isn't even faithful on
  CPU (`max|Δ|=0.75` on a [0,1] image, vs RRDBNet's PSNR ≈101 dB out of the box).

So even before the org decided to retire CoreML, the cheap ONNX/ort path was a
non-starter for AuraSR. **The only real port is a from-scratch MLX reimplementation.**

### MLX port cost (path b) — feasible but disproportionate

A native mlx-rs port would have to reimplement, beyond what the Real-ESRGAN MLX
port already needs (conv stack + pixel_unshuffle + nearest upsample): the
`AdaptiveConv2DMod` (StyleGAN2 modulate/demodulate + adaptive-kernel softmax +
**batch-as-groups conv via a hand-rolled per-sample path**, since mlx conv2d is
groups=1 only), the `StyleGanNetwork` MLP, full + linear attention transformers,
the UNet skip-connection plumbing, the stochastic style/noise inputs, a 2.4 GB /
268-tensor weight convert, and the 64 px tiling + overlapped 2-pass checkerboard
blend. ~10× the architectural surface of the RRDBNet port.

It also **lacks the clean parity story** every other MLX port in this codebase
relied on: AuraSR is stochastic (MLX RNG ≠ torch RNG), so there is no bit-parity
reference — validation degrades to visual/metric similarity. This is
roughly an InstantID/SAM2-scale engine effort, for a *substitutable secondary*
upscaler.

### Findings — quality A/B (the decision criterion)

Decision criterion (from the story): *is AuraSR's quality enough better than
Real-ESRGAN x4 to justify the port cost?*

Objective reference test (HR → bicubic /4 → SR ×4 → vs HR) is a poor discriminator
on clean renders (the perception–distortion tradeoff — plain bicubic "wins" PSNR),
but the relative numbers are telling: AuraSR SSIM **0.788** vs Real-ESRGAN
**0.951** — i.e. AuraSR *deviates more* from ground truth (it hallucinates), and
it was **~30× slower** (4.0 s vs 0.13 s) even at 192²→768².

Real-use texture crops (rusty robot head/eyes, rivets+"13" label, candle scene),
4× upscale, eyeball at zoom (`/tmp/sc3668/quality_crops.png`):

- **No night-and-day gap.** AuraSR adds marginally more micro-grain in the rust —
  arguably "more detail," arguably "more noise." Real-ESRGAN is a touch cleaner.
- Defocused background (candle/curtain) is **identical** — neither invents detail
  that isn't there.
- **AuraSR is ~35–50× slower** per crop (≈5.0 s vs ≈0.1 s for a 200 px crop) on
  the same MPS torch path — and that gap is *intrinsic* (617M params + attention +
  144 tiles/image + 2-pass overlap), so it would persist even after an MLX port: a
  1024²→4096² image is ~512 GigaGAN tile-forwards.

## Decision

Both engines are substitutable 4× upscalers; Real-ESRGAN x4 is the **default** and
is being ported to MLX regardless. AuraSR's quality edge is marginal and
subjective, its runtime cost is intrinsically ~order-of-magnitude higher, and its
MLX port is an InstantID/SAM2-scale, parity-unfriendly engine project. The
cost/benefit does not justify it on the Mac-interim.

→ **DROP on Mac** (UI-gate `aura-sr` via sc-3486; flip `mac_capabilities` so users
never reach the `mlx_unsupported` error), **keep on Win/Linux torch**. This is the
sanctioned "drop is a spike outcome" path under epic 3482's no-silent-drops policy.

If a port is nonetheless wanted, it should be scoped as its own mlx-gen engine epic
(GigaGAN `UnetUpsampler` in mlx-rs), not folded into this story.

## Implementation outline (DROP path)

- `apps/web/src/macGating.js`: add an engine-level gate helper (e.g.
  `macUpscaleEngineBlock`) keyed off the same `mac_capabilities` payload.
- `apps/web/src/screens/ImageEditor.jsx` (`UPSCALE_ENGINES`, ~:199) and
  `ImageStudio.jsx` (~:117): filter `aura-sr` out of the engine list when Mac
  gating is active so it is never selectable; ensure factor falls back to a
  real-esrgan-valid value.
- `crates/sceneworks-core/src/jobs_store.rs`: `upscale_job_is_mlx_eligible` /
  `worker_supports_job` already refuse aura-sr (keep). Keep `mac_capabilities`
  `imageUpscale.supported = true` (the tool works via real-esrgan); the gate is at
  the engine option, not the tool.
- `docs/mac-rust-gaps.md:127`: move the AuraSR clause to a "UI-gated on Mac / Win+
  Linux only" note, referencing sc-3668.
- Tests: `crates/sceneworks-core/tests/jobs_store.rs` (oracle still refuses aura-sr),
  a web test asserting `aura-sr` is hidden under Mac gating, keep
  `apps/rust-api/src/tests.rs` imageUpscale=supported.
