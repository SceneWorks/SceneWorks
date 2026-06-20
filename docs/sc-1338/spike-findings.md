# sc-1338 — Validate LTX video adapter on MPS (Phase 4 video gate)

**Result: PROVEN on both paths.** LTX-2.3 produces a coherent, prompt-matching
MP4 on Apple Silicon (MPS), with performance and memory characteristics
documented below. The story's gate — "an LTX video job produces a valid MP4 asset
on Apple Silicon; performance and memory characteristics documented" — is **met**.

**Decision: HYBRID** (recorded on epic 1330). MLX is the primary Mac path for
t2v / i2v / first-last / LoRA / audio (LTX + Wan); the native PyTorch/MPS path is
retained for replace-person and advanced masked conditioning, which MLX cannot do
(no spatial-mask / control-video). The proven MPS recipe from this spike is now
folded into the shipping native adapter — see "Where the recipe lives" below.

Confidence: **high** on the gate (coherent MP4 reproduced on real M-series
hardware, 137 GB unified memory). The one open variable from the original probe
(ltx-core pin vs torch version on the high-level two-stage pipeline) is noted under
"Open / follow-up" and is tracked as separate implementation stories, not part of
this spike.

## What was validated

Native Lightricks LTX-2.3 distilled, driven directly on `device=mps`, generated a
coherent 49-frame 768×512 24fps H.264 + AAC clip (golden retriever on a beach,
real frame-to-frame motion) on an M-series Mac.

ltx-core hard-targets CUDA and never selects MPS on its own; off-CUDA it is wrong
in five specific ways, each of which the recipe corrects:

1. **Device selection** — `get_device()` only ever returns `cuda` or `cpu`. Pass
   `device=torch.device("mps")` explicitly, or the pipeline runs on CPU.
2. **fp8 quantization is CUDA-only** — `fp8_cast` is gated on
   `device_supports_fp8 == cuda`. Use `dtype=bfloat16`, `quantization=None`.
3. **Block-streaming offload rides CUDA streams/events** — force offload `none`;
   stream components sequentially (text encoder → free → transformer → free → VAE)
   to bound peak memory instead.
4. **Unguarded `torch.cuda.synchronize()` / `empty_cache()`** in `cleanup_memory()`
   and the `gpu_model` teardown raise `Torch not compiled with CUDA enabled` and
   abort the run on a non-CUDA macOS build. Neutralize both when CUDA is truly absent.
5. **Audio vocoder dtype** — `VocoderWithBWE.forward` wraps itself in
   `autocast(float32)`, but MPS autocast only supports bf16/fp16, so the context
   silently disables and the bf16 vocoder weights then meet the fp32 mel input
   (`Input type (float) and bias type (BFloat16)`). Run the audio VAE decode +
   vocoder in fp32 on MPS.

## Performance & memory (the documented gate output)

Head-to-head on M-series, identical model / schedule / resolution / frames
(LTX-2.3 distilled, 8+3 steps, 768×512, 49 frames):

| Path | Time | Peak memory | Coherent? | Notes |
|---|---|---|---|---|
| **MLX** (`mlx-video-with-audio`, Q4) | **37.5 s** | **~31 GB** | yes | fits 32 GB Macs; also covers Wan natively |
| **PyTorch / MPS** (native recipe) | **83.5 s** | **~56 GB** | yes | text 5 s · 8-step denoise 48 s · decode+encode 30 s |

MLX is ≈ 2.2× faster and uses ~45% less peak memory, and is the only path that
fits a 32 GB Mac — hence MLX-primary in the hybrid decision. The native MPS path's
higher peak is the resident (offload `none`) component cost; it is retained for the
conditioning features MLX lacks, not for throughput.

## Where the recipe lives now

The five corrections are no longer throwaway-harness knowledge — they are device-aware
guards inside `apps/worker/scene_worker/video_adapters.py`, behind a CUDA check so
the production NVIDIA path is byte-for-byte unchanged:

- `ltx_mps_gating(cuda_available, device_str)` — returns the off-CUDA overrides
  (`device=mps`, `disable_fp8`, `force_offload_none`, `fp32_audio`,
  `guard_cuda_sync`); empty/no-op when CUDA is present.
- `_neutralize_cuda_sync_for_mps()` — no-ops `torch.cuda.synchronize` /
  `empty_cache` only when CUDA is truly absent (correction #4). Idempotent.
- `_Fp32AudioDecoder` — forces the LTX-2.3 audio decode path to fp32 on MPS
  (correction #5).
- `_ensure_pytorch_mps_fallback()` / `_ltx_inference_mode()` — set
  `PYTORCH_ENABLE_MPS_FALLBACK` and disable autograd around the pipeline call
  (ltx-core only decorates its CLI `main()`, so direct callers must, or the
  per-step activation graph is retained and OOMs).

Because this is a **shared codebase** — one `requirements-ltx.txt` / worker across
Windows desktop + Mac desktop + Docker — every off-CUDA patch is drift-guarded
(`VendorPatchDriftError` via `_require_patch_target`, sc-1647): a dependency bump
that moves a patched symbol fails loudly and points at the pin to re-validate,
rather than silently reinstating the CUDA-only behaviour.

## Open / follow-up (not part of this gate)

- **Pin/torch isolation.** The coherent result used LTX Desktop's pin (`00dc53d`,
  v1.0.0) + manual `DistilledNativePipeline` on torch 2.12. SceneWorks' own pin
  (`1799988`, v1.1.3) via the high-level two-stage `DistilledPipeline` on torch 2.8
  executes but yielded a degenerate flat field in the original probe. Two variables
  differ (ltx pin AND torch) and were not isolated here. Any pin/torch bump must
  re-validate the Windows CUDA/fp8 path because of the shared worker.
- **Implementation stories** (out of scope for this spike): MLX Mac adapter;
  `LtxPipelinesVideoAdapter` device-awareness for the replace-person path; adapter
  routing; MLX model management; ffmpeg audio muxing.

## Reproduction

The original probe ran in a throwaway diagnostic env with shipped pins untouched
(LTX Desktop pin `00dc53d` + torch 2.12 + transformers 4.57.6) using two harnesses
(`gen_desktop.py`, working native recipe; `gen.py`, SceneWorks-pin degenerate).
Those `/tmp` artifacts are gone; the recipe they proved now lives in the adapter
guards listed above, which are the maintained reproduction surface. Weights:
LTX-2.3 distilled (~46 GB) + gemma-3-12b (~24 GB) from the Hugging Face cache.
