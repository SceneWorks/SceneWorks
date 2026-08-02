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

## Validation boundary

The host has CUDA 12.9 and two RTX PRO 6000 Blackwell GPUs. The full provider and the real Qwen VAE
load on CUDA, while contract/conformance, candidate coverage, exact request scope, stale fingerprint,
cleanup, cancellation, and resident-versus-streamed ragged-window parity are exercised by tests.
Exact current Qwen base/edit transformer and text-encoder artifacts were not installed locally, so
this story records those complete-route cells as loadability-unverified rather than borrowing the
older resident/sequential measurements or fabricating calibration evidence.
