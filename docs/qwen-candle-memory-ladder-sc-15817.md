# Qwen-Image Candle/CUDA memory ladder (SC-15817)

This record separates implementation/loadability evidence from calibration. It does not promote any
Qwen Candle cell to Verified; the exact request-peak cells remain owned by SC-15865, SC-15868, and
SC-15871.

## Implemented production surface

- Routes: `qwen_image` and `qwen_image_edit`; the `qwen_image_edit_2511_lightning` entry shares the
  edit provider but its built-in LoRA makes bounded transformer residency explicitly Missing.
- Tiers: BF16, packed Q4, and packed Q8.
- Ladder: resident; staged residency; head-once/tail-tiled decode; bounded attention; streamed
  60-block DiT residency.
- Decode candidates: output-pixel edges 768, 640, 512, 448, 384, 320, and 256 with overlap 64.
- Attention candidate: a 64 MiB shared score budget (`67108864` bytes).
- Transformer candidates: DiT windows 1, 2, 4, 8, 15, and 30, including ragged final windows.
- Exclusions: PiD, control, ComfyUI single-file transformer streaming, and adapter-bearing rung 4.

The provider contract fingerprint is
`qwen-image-cuda-staged-tiled-decode-bounded-attention-device-format-blocks-v1`. Optimized requests
require staged residency and an exact current calibration identity. Missing or stale evidence falls
back to the existing resident/sequential admission gate.

## Real-weight overlap decision

The SC-15817 ignored CUDA harness decoded the same deterministic 1024x1024 latent in three fresh
processes on GPU 0. It used the Qwen VAE shipped in `SceneWorks/krea-2-turbo-mlx` q4 at snapshot
`d009674080cc1bccf2b629d834c34bf5eccdb723`; the VAE safetensors SHA-256 is
`ab1b61103959913d6c7e628cf793dbb2ca4726a40a3b3ae206c52b8e75bf6f08`.

| 384px tail tile arm | Observed peak | Rounded-RGB MAE vs monolithic | RMSE | Maximum error |
| --- | ---: | ---: | ---: | ---: |
| Monolithic reference | 13.6 GB | 0 | 0 | 0 |
| Overlap 64 | 2.7 GB | 0.1247 | 0.6036 | 27 |
| Overlap 96 | 2.7 GB | 0.1323 | 0.4771 | 20 |

Both overlaps produced the same one-decimal observed peak. Overlap 96 changed more rounded pixels
and had slightly worse mean absolute error, while only lowering the tail of the error distribution;
it also evaluates more redundant pixels. There is no material fidelity or memory win that justifies
the extra work, so overlap 64 remains the sole shipped candidate below 512px.

## Real-weight five-rung CUDA exercise

The shared harness exercised every implemented rung in one model-load batch on GPU 0 using the
packed Q4 snapshot `SceneWorks/qwen-image-mlx@8080a4171f1c8b7fca6c30491eafbe6ffab754bf`.
The schema-valid capture is published separately at
`docs/generated/qwen-candle-five-rung-sc-15817.json`, outside the runtime admission bundle.
The capture is bound to SceneWorks `523e0c6351b49209084a9ab762a667c85c00261b` and inference
`378f6c7dadd559bba7d32e8c87165e0dd7900710`, with both worktrees clean. The device was an NVIDIA
RTX PRO 6000 Blackwell Max-Q Workstation Edition (compute capability 12.0, 102,641,958,912 bytes,
CUDA 12.9, driver 596.36).

| Rung | Parameters | Observed request peak |
| --- | --- | ---: |
| Resident | none | 55,638,491,136 bytes |
| Staged residency | none | 42,048,946,176 bytes |
| Bounded decode | tile 512, overlap 64 | 29,700,915,200 bytes |
| Bounded attention | tile 512, overlap 64, score budget 67,108,864 | 29,700,915,200 bytes |
| Bounded transformer residency | prior parameters, block window 1 | 29,700,915,200 bytes |

Every record passed exact artifact loadability and reported the selected contract fingerprint,
load shape, engaged-rung identity, parameters, phase peaks, and clean repository revisions. These
records remain `candidate`/`gated`: they prove implementation and representative real-weight
execution, but deliberately do not substitute for each sibling story's promotion-quality scenario,
negative-mutation, lifecycle, tier, mode, and overlay calibration.

## Validation boundary

The host has CUDA 12.9 and two RTX PRO 6000 Blackwell GPUs. The full base provider, text encoders,
transformer, and VAE load from the exact packed snapshot on CUDA. Contract/conformance, candidate
coverage, exact request scope, stale fingerprint, cleanup, cancellation, and
resident-versus-streamed ragged-window parity are exercised by tests. The checked-in measurements
are representative implementation evidence only; the three blocked model stories still own their
complete request-specific calibration and Full signoff.
