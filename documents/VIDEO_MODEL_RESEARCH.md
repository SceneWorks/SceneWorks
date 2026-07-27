# SceneWorks Video Model Research

Use `ltx_2_3` as the first SceneWorks video target, with Wan2.2 present as the next fallback family once the adapter boundary can host multiple runtimes.

## Recommendation

- First adapter: LTX-2.3.
- Runtime path: start with the official Lightricks PyTorch/ComfyUI-compatible path and keep the worker contract isolated behind a SceneWorks adapter.
- First shipped implementation in this repo: `procedural_video`, which produces deterministic preview clips while exercising the real Video Studio, job, recipe, lineage, and Library asset contracts.
- Fallback family: Wan2.2, exposed in the manifest so UI and payload settings can already express Wan-aware limits.

## Why LTX-2.3 First

- Lightricks documents LTX-2.3 as an open-weights DiT audio-video model with multimodal inputs including text, image, video, audio, depth, and LoRA-based customization.
- The Hugging Face model card provides the practical local entry point, model IDs, checkpoint variants, and PyTorch repository requirements.
- The official usage guides support both image-to-video and text-to-video, with resolution and frame count guidance that maps well to SceneWorks controls.
- It is a better first SceneWorks target than Wan2.2 for the current product slice because it supports a single family for I2V, T2V, first/last-frame style conditioning, and future audio-aware workflows.

## Encoded Product Limits

- Keep SceneWorks oriented around short shots assembled later in the editor.
- Simple UI should recommend 4-8 seconds for fast iteration and keep 10 seconds as the normal LTX-2.3 ceiling.
- The broader product assumption from the plan says LTX2.3 is best at 15 seconds or less. Current official guides list 257 frames, roughly 10 seconds at 25fps, for the common I2V/T2V workflows, so the UI should favor 10 seconds for now and reserve longer durations for future adapter-specific support.
- Resolution dimensions must be divisible by 32 for LTX-2.3. Favor 768x512, 640x640, 1280x720, and 720x1280 presets.
- FPS controls should default to 25fps for LTX-2.3, with 24fps and 30fps available in advanced mode.
- Quality should map to raw adapter settings:
  - Fast: fewer frames/steps for iteration.
  - Balanced: default distilled settings.
  - Best: higher step budget and future multiscale/upscale path.

## Wan2.2 Notes

- Wan2.2 has official T2V, I2V, TI2V, and S2V model entries with 480P/720P support.
- Wan2.2 is valuable as a fallback and later adapter because it has broad video modes and Diffusers/ComfyUI ecosystem support.
- Keep Wan-aware UI guidance conservative: shorter clips around 5-7 seconds are recommended until local looping behavior is validated against the exact runtime.
- **Quantized A14B inference (sc-1982).** A14B is GPU-heavy at bf16 (~56GB of transformers), so the manifest declares quantization variants and Video Studio exposes a **Quantization** selector (Advanced panel). Two paths:
  - **GGUF (torch adapter, cross-platform):** the two experts load via `WanTransformer3DModel.from_single_file(..., quantization_config=GGUFQuantizationConfig(...))` (high-noise → `transformer`, low-noise → `transformer_2`) from `QuantStack/Wan2.2-{T2V,I2V}-A14B-GGUF`. The 5B (TI2V) has a single-transformer GGUF too. Defaults are per-platform: **Q8_0 on MPS** (trivial dequant, ~3× slower vs ~13× for k-quants — and the GGUF path runs fp32 on MPS because Wan's Conv3d has no bf16 Metal kernel), **Q4_K_M on CUDA** (smallest, fused kernel). `auto` follows the default; `none` forces the unquantized base.
  - **MLX-Q4 (preferred on Mac):** the `model_convert` job accepts `quantizeBits`/`quantizeGroupSize` (and a `--quantize-only` pass for turnkey bf16 MLX repos), and the MLX adapter prefers a locally-converted/quantized dir over the turnkey download. ~3.7× faster + ~2.6× less memory than GGUF-on-MPS in the sc-1950 spike (84s / 41GB peak), fitting a 64GB Mac.
  - Quantized experts still accept trained per-expert LoRAs (validated for GGUF in sc-1950; MLX via `loras_high`/`loras_low`). Weights are Apache-2.0.

## Krea Realtime 14B Notes (sc-8446 / epic 8431, measured 2026-07-27)

`krea/krea-realtime-video` is **Wan 2.1 T2V 14B, weight-for-weight**, distilled via Self-Forcing into
an **autoregressive** generator: frame-chunks left to right over a persistent causal KV cache, ~5
denoising forwards per 3-latent-frame chunk, 24 fps. SceneWorks runs it as the native MLX engine
`krea_realtime_14b` (crate `mlx-gen-krea-realtime`), from the rehost
`SceneWorks/krea-realtime-14b-mlx` (Q4 / Q8 / bf16 tiers, Apache-2.0). It is architecturally distinct
from every other SceneWorks video engine, all of which are full-clip block diffusion. Do not confuse it
with the unrelated `krea_2_*` **image** lane.

- **It renders a real, watchable clip on Mac.** 832×480, 81 frames @ 24 fps on the Q4 tier: 256 s wall
  (31 s/AR chunk, 6.2 s/step, 38 s VAE decode), 27.9 GiB MLX-active peak. Subject identity and motion
  hold across the whole clip.
- **Known quality caveat — AR stylization drift.** The render progressively saturates/stylizes over the
  clip. This is the Self-Forcing bounded-window behaviour with `sink_size = 0` and no first-frame VAE
  re-anchor (the reference's `release_server.py` keeps one; the batch path does not). Tracked as
  sc-15127. Keep UI guidance to **short shots (~2–4 s)** until an anchor lands, consistent with the
  rest of this document's short-shot posture.
- **Known artifact — decode tile seams.** The VAE decode is temporally tiled (8-frame windows,
  2-frame overlap at 832×480), and the tiling costs ~27/255 mean absolute error versus a single-pass
  decode of the same latents, visible as a discontinuity every 8 output frames. It is what bounds the
  memory peak, so it is not simply removable.
- **CFG is off.** The distillation baked guidance out: no negative prompt, no guidance scale. Video
  Studio hides both for this model (`video.supportsGuidance/supportsNegativePrompt: false`).
- **LoRA: any Wan-2.1-14B **T2V** LoRA works.** Verified on real published files — a plain style LoRA
  (`shauray/Origami_WanLora`, 400 per-block targets) installs and moves the render substantially, and a
  step-distill LoRA (lightx2v Wan2.1-T2V-14B cfg-step-distill v2, which additionally targets the
  whole-model patch/text/time/head projections) installs 406 targets. Wan-**I2V** LoRAs are correctly
  rejected: they name `cross_attn.k_img`/`v_img`, modules a T2V backbone does not have.
- **Memory guidance.** `mlx.minMemoryGb: 64` admits the heaviest installable tier (bf16, ~47 GiB active
  peak, derived from the exact hosted DiT byte counts). The default **Q4** tier needs ~28 GiB active
  (~40 GB machine). The AR KV cache is a fixed **7.14 GiB** at 832×480 and does **not** shrink with the
  weight tier — it holds bf16 activations — so 720p (3600 tokens/frame, ~16.5 GiB of KV) is the
  memory-relevant resolution jump, not the weight tier.

## Sources

- LTX open source overview: https://docs.ltx.video/open-source-model/getting-started/overview
- LTX-2.3 Hugging Face model card: https://huggingface.co/Lightricks/LTX-2.3
- LTX image-to-video guide: https://docs.ltx.video/open-source-model/usage-guides/image-to-video
- LTX text-to-video guide: https://docs.ltx.video/open-source-model/usage-guides/text-to-video
- Wan2.2 Hugging Face model card: https://huggingface.co/Wan-AI/Wan2.2-S2V-14B
- Krea Realtime 14B model card: https://huggingface.co/krea/krea-realtime-video
- SceneWorks MLX rehost: https://huggingface.co/SceneWorks/krea-realtime-14b-mlx
- Measurements: `mlx-gen-krea-realtime/tests/generate_smoke.rs` (inference repo, sc-8446)
