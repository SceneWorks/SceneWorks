# Video-Model Feasibility Matrix (sc-1174)

> **Story:** [sc-1174 — Validate video model feasibility matrix](https://app.shortcut.com/trefry/story/1174)
> **Epic:** [1093 — SceneWorks: Research Tracks](https://app.shortcut.com/trefry/epic/1093)
> **Last updated:** 2026-07-27 (Krea Realtime 14B empirical entry — sc-8446 / epic 8431)
> **Status:** Validated — chosen model spiked empirically; comparison set web-verified (June 2026). Two claims in the story were tested and corrected.

**Provenance:** ⚙️ = empirically run on this machine (Apple M5 Max) · 🌐 = web-verified June 2026 · 📄 = SceneWorks code/manifest.

## Recommendation (v1 ordering)

1. **`ltx_2_3` — primary.** ~5.7× faster than Wan2.2 A14B on 24 GB-class HW (distilled 8-step); the
   **only** option with native **synchronized audio** (a real differentiator for a media app);
   richest native conditioning (first/last/keyframe-bridge + region-regen); best Apple/MLX story;
   longer practical clips (~15–20 s). ⚙️ Ran natively here. **License caveat below.**
2. **`wan_2_2` TI2V-5B — safe default fallback.** Dense 5B, ~10 GB, 24 fps native, **Apache-2.0 /
   ungated**, fits 24 GB trivially (even ~8 GB). Removes all license risk; the workhorse for
   low-VRAM and license-sensitive users.
3. **`wan_2_2` A14B — quality tier / multi-GPU path.** 27B MoE (14B active); higher fidelity but
   needs fp8/GGUF + expert-swap on 24 GB (bf16 wants 80 GB) and is ~5–6× slower. First-class
   multi-GPU (FSDP + Ulysses) if SceneWorks ever scales out.

> **Update (sc-8446, July 2026):** a fourth family now runs natively on Mac —
> **`krea_realtime_14b`** (Krea Realtime 14B, Wan-2.1-T2V-14B distilled via Self-Forcing into an
> autoregressive chunk generator, Apache-2.0). It is the only SceneWorks video engine that is *not*
> full-clip block diffusion, and the only Wan-2.1-14B-class engine that renders a full 81-frame
> 832×480 clip inside 28 GiB of MLX-active memory. Measured numbers below.

**Do not** make MLX video a v1 *gating requirement* on Mac — treat it as best-effort; NVIDIA is the
supported path. (Apple still runs LTX-2.3 natively — see empirical result — it's just memory-bound.)

## ⚠️ Two story claims, tested and corrected 🌐

The story description asserts two specifics. Validation says:

- **"LTX-2.3 best at ~15 s or less" → PARTIALLY TRUE — it's a quality sweet-spot, not the ceiling.**
  Hard model max is **~20 s**; ~15 s is a real quality knee (subject-drift beyond 12–15 s on portrait
  clips; "Standard" tier recommended "for temporal consistency across 15 s"). **Encode 15 s as a
  recommended sweet-spot, not a hard limit.**
- **"Wan2.2 may loop cleanly around ~7 s" → UNVERIFIED / LIKELY INCORRECT.** Native clips are **~5 s**
  (81 frames @16 fps A14B; 121 @24 fps TI2V-5B). Seamless looping is an **open feature request**, only
  achievable via community first=last-frame / last-frame chaining. **No source pins anything to 7 s.
  Do not encode a "7 s native loop" capability.**

## Empirical result — LTX-2.3 ⚙️

`ltx_real_weights_with_audio` on Apple M5 Max (LTX-2.3 q4 + Gemma-3-12B TE, 256², 9 frames, 24 fps,
synchronized audio, native MLX, in-process):

| Metric | Value |
|---|---|
| Gen time | 14.49 s (in-process) / 15.65 s wall |
| **Peak unified memory** | **53.4 GB** |
| Host RSS | 23.0 GB |
| Manifest `mlx.minMemoryGb` | 31 (`builtin.models.jsonc:2199`) |

> ⚠️ Measured peak (**53.4 GB**) is **~1.7× the 31 GB manifest estimate** — the Gemma-3-12B text
> encoder loads dequantized (~24 GB) alongside the q4 DiT + VAEs. Even a *minimal* 256²/9-frame clip
> uses 53 of 64 GB, so **64 GB Macs have little headroom for real-size LTX clips**. Feeds sc-1177.

## Empirical result — Krea Realtime 14B ⚙️ (sc-8446, 2026-07-27)

`mlx-gen-krea-realtime` `generate_smoke` on Apple M5 Max / 128 GB, **Q4 tier** of
`SceneWorks/krea-realtime-14b-mlx` @ `e68e9a3d`, 832×480, 81 frames @ 24 fps, the shipped
Self-Forcing schedule (5 steps/chunk × 7 autoregressive chunks = 35 forwards), native MLX,
in-process:

| Metric | Value |
|---|---|
| Whole-clip wall | **256 s** (3.4 s of video → ~75× realtime) |
| Load + UMT5 encode | 4.8–5.9 s |
| AR denoise | 211 s (**6.2 s/step**, **31 s/chunk** mean; 20 s first chunk → 36 s once the KV window is full) |
| VAE decode (84→81 frames) | 38–39 s |
| **MLX peak, active** | **27.9 GiB** |
| MLX peak, active + allocator cache | 90.4 GiB |
| KV-cache residency (measured) | **7.14 GiB**, flat from chunk 2 on |
| Manifest `mlx.minMemoryGb` | 64 |

**The peak is set by the VAE decode, and it is now measured rather than hedged.** Cumulative active
peak by phase: 9.6 GiB after load+encode → 17.4 GiB after the AR loop → **27.9 GiB** after the decode.
The AR loop's 17.4 GiB closes against the manifest's accounted terms (Q4 DiT 7.8 + KV 7.1 + VAE 0.5 ≈
15.4 GiB); the decode adds ~10.5 GiB on top. Two things follow:

- **`mlx.minMemoryGb: 64` is correct, but for a different reason than it was written for.** It is not
  a VAE-decode hedge — that term is +10.5 GiB, not the ~2× that was feared. A single per-model value
  has to admit the heaviest *installable* tier, and the **bf16** tier swaps a 7.8 GiB DiT for a
  26.6 GiB one (exact hosted byte counts), putting its active peak at **~47 GiB**. 64 covers that with
  OS/app headroom. On the default **Q4** tier the true requirement is ~28 GiB active (a ~40 GB
  machine); a per-tier gate would read **Q4 40 / Q8 48 / bf16 64**.
- **The ceiling does not grow with clip length.** 33-frame and 81-frame clips both peak at 27.6–27.9
  GiB active, because the decode is temporally tiled (8-frame windows, 2-frame overlap at 832×480).
- ⚠️ `mlx::get_peak_memory` is the **active** high-water mark only. The allocator's buffer cache
  (reclaimable, but real resident memory) took the same run to 90.4 GiB on a 128 GB host. Any
  `minMemoryGb` derived from `get_peak_memory` alone under-reports the machine's working set.

**KV cache.** 40 layers × 9360 tokens (a 6-latent-frame window × 1560 tokens/frame) × 40 heads × 128
dims × {k,v} × bf16 = **7.14 GiB**, confirming the S1 architecture estimate exactly. It holds
*activations*, so it is **bf16 on every weight tier** — quantizing the DiT to Q4 does not shrink it, and
it is the single largest term after the weights. Quantizing the KV cache itself remains an available
lever if 720p (3600 tokens/frame → ~16.5 GiB) is ever targeted on a 64 GB machine.

**Coherence.** The 81-frame clip is genuinely watchable — a fox trotting through snowy pines, stable
subject identity and plausible motion end to end. It is **not** drift-free: the render progressively
stylizes (frame 0 photographic → frame 80 noticeably more saturated and contrasty, background
softening), the expected Self-Forcing bounded-window behaviour with `sink_size = 0` and no first-frame
re-anchor (tracked as sc-15127). A second artifact is visible in the per-frame delta trace: a
brightness/detail discontinuity **every 8 output frames**, exactly the decode tile stride — the tiled
decode costs ~27/255 mean absolute error versus a single-pass decode of the same latents.

## Capability & spec matrix 🌐

| Dimension | **LTX-2.3 (22B)** | **Wan2.2 A14B (27B MoE/14B active)** | **Wan2.2 TI2V-5B (dense)** | **Krea Realtime 14B (dense AR)** ⚙️ |
|---|---|---|---|---|
| Text-to-video | Yes | Yes | Yes (unified) | Yes |
| Image-to-video | Yes | Yes | Yes | Yes (a still warms the AR KV cache) |
| First-frame cond. | Yes (native) | Yes (I2V) | Yes | Yes (that IS the i2v path) |
| Last-frame / first+last bridge | **Yes, native** (keyframe interp) | **Not in core Wan2.2** — via node / Fun-InP (FLF was Wan2.1) | via node, same caveat |
| Video extend / region-regen | **Yes** (retake/region) | No native continue-clip | No native | v2v restyle only (strength-controlled AR init); no extend/region |
| LoRA (infer / train) | Yes / Yes (`ltx-trainer`) | Yes / Yes (high+low-noise pair) | Yes / Yes | **Infer: yes — any Wan-2.1-14B-T2V LoRA** ⚙️ / no trainer |
| **Native FPS** | 24/25/48/50; **24–25 rec.** | **16 fps** | **24 fps** | **24 fps** |
| Practical duration | **~20 s max; ~12–15 s sweet-spot** | **~5 s** (81f@16) | **~5 s** (121f@24) | AR loop is unbounded in principle; visible stylization drift accumulates past ~3 s (sc-15127) |
| **Audio** | **Yes — native synced A/V (24 kHz)** | No native audio | No | No |
| Frame constraint | `(F-1) % 8 == 0`, dims ÷32 | `(F-1) % 4 == 0` | same | `(F-1) % 4 == 0`, dims ÷16; latent frames in 3-frame AR chunks |
| **License** | **Custom "LTX-2 Community License"** — free commercial **only under $10M ARR**; anti-compete clause; **NOT Apache** | **Apache-2.0** | **Apache-2.0** | **Apache-2.0** |
| Gating | effectively ungated (verify acceptance click-through) | ungated | ungated | ungated |
| CFG | Yes | Yes | Yes | **No** — Self-Forcing distilled guidance out; one batch-1 forward/step, no negative prompt |
| Mac (measured) | 53.4 GB peak, 256²/9f | no measured 14B Mac time | — | **27.9 GiB active peak, 256 s for 832×480/81f @ Q4** ⚙️ |

## 24 GB NVIDIA VRAM & runtime 🌐

| | **LTX-2.3 22B** | **Wan2.2 A14B** | **Wan2.2 TI2V-5B** |
|---|---|---|---|
| bf16 on 24 GB | DiT fits; **Gemma 3-12B encoder is the squeeze (~24–27 GB)** | **No** (min 80 GB; 28.6 GB/expert ×2) | **Yes** (min 24 GB → ~8 GB offloaded) |
| GGUF Q4_K_M | distilled **17.8 GB** single file | **9.65 GB/expert** (~18 GB, loaded 1 at a time) | **3.43 GB** |
| 24 GB unlock | CPU-offload or **fp4 encoder ≈ 8.8 GB**; FFN chunking | MoE expert-swap + `--t5_cpu` + offload → ~6–8 GB @480p | fits natively |
| Head-to-head (RTX 5090, Q4, 832×480, 81f I2V) | **22.1 s warm / 48.5 s cold** | 125 s warm / 143.9 s cold | — |
| → speed | **~5.7× faster** (distilled 8-step) | baseline | 5 s/720p in <9 min |
| Multi-GPU | **none native** (community sharding only) | **first-class** (FSDP + Ulysses) | same framework |

LTX's 5.7× lead reflects the **distilled 8-step** checkpoint at 480p/Q4; it narrows with the full
non-distilled pipeline (~9 min multimodal) and at 4K.

## Apple / MPS / MLX feasibility 🌐⚙️
- **LTX-2 / 2.3 MLX is the strongest video MLX story** (multiple real ports; q4≈12 GB/16 GB-Mac,
  q8≈21 GB/32 GB-Mac, bf16≈42 GB/64 GB-Mac). ⚙️ SceneWorks' own `ltx_2_3` MLX engine ran here at
  53.4 GB peak (q4 + Gemma-12B), 15.6 s for a minimal clip.
- **Wan2.2 MLX is experimental** — only via `mlx-video`; on M2 Max/32 GB Wan2.2-14B "uses almost all
  32 GB" while LTX-2.3-22B ≈ 19.4 GB; only Wan2.1-1.3B runs comfortably. **No measured 14B/22B Mac
  generation times exist** (assume minutes).
- ComfyUI on MPS breaks on fp8 (silent CPU fallback); image-only MLX stacks (mflux/DiffusionKit) have
  no Wan/LTX video.

## Rust backend target 📄

The repo **already implements** this contract; below are validated deltas, not greenfield design.

- **Manifest:** `ltx_2_3` (adapter `ltx_video`) and `wan_2_2` / `wan_2_2_t2v_14b` / `wan_2_2_i2v_14b`
  already exist as `ModelKind::Video` (`builtin.models.jsonc:2394,2501,2608`). **Multi-file resources
  already supported** via typed per-platform `downloads` *and* the untyped `resources` named-slot map
  (`checkpoint`/`spatialUpscaler`/`distilledLora`/`gemma`); LTX-2.3 already uses it (`:2137`).
  - **Gaps:** (a) no per-file `sha256`/url/size — only HF `repo`+`file`+`estimatedSizeBytes`; consider
    per-file hashes for supply-chain integrity (more pointed given the LTX license obligation). (b)
    VRAM is untyped (`mlx.minMemoryGb` only, and the LTX run shows it undercounts ~1.7×) — add a
    **precision-keyed VRAM block** (`bf16`/`fp8`/`gguf_q4` + `text_encoder_vram`) given the LTX
    24-vs-32 GB encoder cliff and Wan A14B's 80 GB-bf16 / 24 GB-fp8 split.
- **Capability flags** (`ModelCapability`, `contracts.rs:542-552`): has `text_to_video`,
  `image_to_video`, `video_extend`, `video_bridge`. **Skew to fix:** manifest video entries use mode
  strings not in the enum — `first_last_frame`, `extend_clip`, `replace_person` — and `video_extend`
  (enum) ≠ `extend_clip` (manifest), a latent naming collision. Reconcile (add `first_last_frame` to
  the enum or document that `capabilities` are `ContractMode` strings).
- **Job payload:** `VideoGenerate/VideoExtend/VideoBridge/VideoUpscale` job types + a typed
  `VideoRequest` already carry `mode, prompt, duration, fps, width/height, seed, loras[],
  source_asset_id, last_frame_asset_id, source_clip_asset_id, bridge_right_clip_asset_id,
  model_manifest_entry` — **the full LTX/Wan conditioning surface**. Frame-snapping helpers exist
  (`ltx_frame_count` → `8k+1`, `wan_frame_count` → `4k+1`), matching the verified constraints.
  **Add:** surface the LTX **audio** track on the job result/asset (the muxed MP4 carries it, but
  audio-track metadata isn't separately modeled).
- **Scheduling constraints:** routing is capability-based, not VRAM-based. Given the LTX 24-vs-32 GB
  cliff and Wan A14B offload-dependence, add a **precision-aware admission check** (does
  model+precision fit the assigned GPU/unified free memory?) using the typed VRAM block — don't
  discover OOM at runtime. `JobSnapshot` already records post-hoc peak GPU mem.
- **Asset outputs:** `AssetFile` already video-capable (`path, mime_type, width, height, duration,
  fps`) — no new fields needed except optional audio-track metadata.

## Caveats / could-not-verify 🌐 (research limits, not story gaps)
"Wan ~7 s loop" unsupported by any source; "LTX ≤15 s" true as sweet-spot only; Wan A14B fp8 on
exactly 24 GB @720p unverified (480p is the safe claim); no measured 14B/22B Mac generation times;
**"LTX-2.3 is Apache-2.0" is a widespread but incorrect claim** (authoritative LICENSE is the custom
Community License — high-confidence refutation); LTX HF acceptance click-through not ruled out; Wan
Fun-InP (FLF path) license not separately fetched. Full source list retained in research notes.

## Sources
Empirical: `ltx_real_weights_with_audio` on M5 Max; `mlx-gen-krea-realtime`'s
`tests/generate_smoke.rs` + `tests/style_lora_real_weights.rs` on M5 Max (sc-8446, inference repo). Web (June 2026): github.com/Lightricks/LTX-2
(+LICENSE), HF Lightricks/LTX-2.3, ltx.io/model/license, github.com/Wan-Video/Wan2.2, HF
Wan-AI/Wan2.2-{I2V-A14B,TI2V-5B}, QuantStack/Wan2.2-*-GGUF, comfy.org Wan2.2 docs, RTX-5090
LTX-vs-Wan benchmark (zenn.dev), github.com/Blaizzy/mlx-video, github.com/dgrauet/ltx-2-mlx. Code:
`crates/sceneworks-core/src/contracts.rs`, `config/manifests/builtin.models.jsonc`. Prior:
`documents/VIDEO_MODEL_RESEARCH.md`, `documents/EPIC_NATIVE_LTX23_VIDEO_ADAPTER.md`.
