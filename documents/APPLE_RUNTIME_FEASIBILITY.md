# Apple Runtime Feasibility (sc-1176)

> **Story:** [sc-1176 — Validate Apple runtime feasibility](https://app.shortcut.com/trefry/story/1176)
> **Epic:** [1093 — SceneWorks: Research Tracks](https://app.shortcut.com/trefry/epic/1093)
> **Last updated:** 2026-06-18
> **Status:** Validated empirically on Apple M5 Max / macOS 26.5.1.

**Provenance:** ⚙️ = empirically spiked on this machine · 📄 = grounded in shipped code/docs.

## Bottom line

Apple is a **first-class v1 runtime**, not a stretch goal. The native, in-process **Rust + MLX**
worker builds from source, passes the NAX correctness guard, and **runs both v1 flagship models
end-to-end on this machine** — no Python, no venv, no subprocess, no Docker GPU. The decisive
constraint is **memory** (video is heavy) and a **macOS 26.2 floor** for correct 16-bit kernels,
not whether Apple can run the stack at all.

## Test machine ⚙️

| | |
|---|---|
| Chip | Apple **M5 Max** (40-core GPU, 18-core CPU) |
| Unified memory | **64 GB** |
| macOS | **26.5.1** (build 25F80) — above the 26.2 NAX floor |
| Toolchain | Xcode 26.5, Metal toolchain v17.6.42, Rust 1.96.0 |

## Empirical results

### 1. Native MLX worker builds from source ⚙️
`cargo build -p sceneworks-worker --release` compiled Apple MLX + all `mlx-gen` provider crates
(`z-image, flux2, qwen-image, wan, ltx, sdxl, sam2/3, chroma, ideogram, kolors, pulid, instantid,
seedvr2, scail2, bernini, joycaption, sensenova, …`) in **2m 22s, 0 errors, 0 warnings**, at the
`MACOSX_DEPLOYMENT_TARGET = "26.2"` pin from the workspace `.cargo/config.toml`. The full
native-MLX engine registry is present — every v1-relevant model family has a native Apple engine.

### 2. NAX fast-path kernels present **and numerically correct** ⚙️📄
`cargo test -p sceneworks-worker --release --test nax_guard` → **`nax_16bit_sdpa_is_correct ... ok`**
(0.12 s). This test runs a 16-bit fused scaled-dot-product-attention against an f32 reference across
6 shape/layout combos and asserts worst-case relative error < 0.05 (`nax_guard.rs:47-109`). It is the
tripwire for the macOS 26.0 mis-compile where 16-bit (bf16/f16) tensor ops "compile but produce
garbage" (`nax_guard.rs:3-11`). **Passing means the Apple matrix-unit kernels are both compiled in
and producing correct fp16/bf16 results** on this box.

### 3. Flagship image model runs natively ⚙️
`zimage_real_weights_generates_one_image` (Z-Image-Turbo, 512², 8-step, Q8) — generated one image
in-process, no server:

| Metric | Value |
|---|---|
| In-process gen | 6.16 s |
| Wall | 7.12 s |
| **Peak unified memory** | **28.6 GB** |
| Host RSS | 3.34 GB |
| Manifest `mlx.minMemoryGb` | 40 (`builtin.models.jsonc:66`) — measured peak is **under** the declared floor at 512² |

### 4. Flagship video model runs natively ⚙️
`ltx_real_weights_with_audio` (LTX-2.3 q4 + Gemma-3-12B text encoder, 256², 9 frames, 24 fps,
synchronized audio) — generated a real clip in-process:

| Metric | Value |
|---|---|
| In-process gen | 14.49 s |
| Wall | 15.65 s |
| **Peak unified memory** | **53.4 GB** |
| Host RSS | 23.0 GB |
| Manifest `mlx.minMemoryGb` | 31 (`builtin.models.jsonc:2199`) |

> ⚠️ **Memory finding (feeds [risk register](#) sc-1177):** measured peak (**53.4 GB**) is **~1.7× the
> manifest's 31 GB estimate**. The gap is the **Gemma-3-12B text encoder loading dequantized
> (~24 GB)** alongside the q4 DiT and VAEs. At a *minimal* 256²/9-frame clip this already consumes
> 53 GB of 64 GB — so **64 GB Macs have little headroom for real-size LTX clips**, and the manifest
> memory estimate undercounts the true simultaneous footprint. The scheduler's admission check
> should key on measured peak, not the optimistic DiT-only estimate.

## Acceptance-criteria findings

### Apple feasibility matrix (selected v1 adapters/models)
| Model | Native Apple engine | Ran here ⚙️ | Peak | Notes |
|---|---|---|---|---|
| Z-Image-Turbo (image) | `z_image_turbo` (MLX) 📄 | ✅ | 28.6 GB | 512²/8-step/Q8; responsive |
| LTX-2.3 (video) | `ltx_2_3` (MLX) 📄 | ✅ | 53.4 GB | q4 + Gemma-12B TE; memory-bound |
| FLUX.2 (image) | `flux2_dev`, `flux2_klein` (MLX) 📄 | engine compiled, not run | — | Mac-only/Q4 tiers exist; large |
| Wan / SDXL / Qwen-Image / SAM2-3 | native engines compiled 📄 | not run | — | present in `mlx-gen` registry |

### Unsupported ops / FP8/FP16 issues 📄⚙️
- **16-bit correctness is macOS-version-gated.** Below 26.2 the NAX 16-bit GEMM/SDPA kernels
  miscompile to garbage (right scale, uncorrelated); at 26.0 they compile but are wrong; **26.2+
  makes them correct and fast** (`.cargo/config.toml`, `nax_guard.rs:3-11`). Verified correct here.
- **Quantization is the Apple precision story, not fp8.** Models ship Q4/Q8 (`z_image` `quantize: 8`,
  LTX `q4`); the native path quantizes weights rather than relying on CUDA-style fp8 GEMM.
- **CoreML execution provider is unusable for the detector path** — it "hangs indefinitely in
  `commit_from_file`" on the Ultralytics YOLO11 ONNX export, so the whole Mac stack is **MLX
  (mlx-rs), not `ort`+CoreML** (`docs/sc-3633-mlx-port.md:5-8`).
- **A few models remain torch-only on Mac** (each with a tracked porting epic — bare drops are
  disallowed): `pulid_flux_dev` (epic 3069), Kolors variants (epic 3090), Qwen strict-pose + base
  reference/edit (epic 3401), LoKr-on-Wan training (epic 3039), Wan/LTX `model_convert` (sc-3491)
  (`docs/mac-rust-gaps.md:53-133`). **AuraSR is dropped on Mac only** (617M torch-only GigaGAN, no
  viable Rust path, ~35–50× slower than Real-ESRGAN) and UI-gated out, kept on Win/Linux
  (`docs/mac-rust-gaps.md:130`).

### Docker acceleration limits 📄
Docker Desktop on macOS cannot pass through the Metal GPU, so the Linux/Docker server path
(`--features backend-candle` / CUDA) **cannot accelerate generation on a Mac**. This is *why* Apple
uses the native in-process MLX worker rather than the containerized Linux path — the runtime split
is deliberate, not incidental. Windows/Linux keep the Python-torch / candle path for unported models
(`docs/mac-rust-gaps.md:50`).

### Native runtime requirements 📄
- **macOS ≥ 26.2** at runtime for the NAX fast path (pinned at cutover sc-3032).
- **Full Xcode + Metal toolchain** at build (`xcode-select -p` → Xcode, `xcrun --find metal`
  resolves) — verified here.
- `MACOSX_DEPLOYMENT_TARGET = "26.2"` **must** live in the workspace `.cargo/config.toml` (Cargo does
  not inherit a dependency's config); missing it floors at macOS 14 and compiles out NAX (~2.5×
  regression). `nax_guard` is the CI tripwire and requires a **self-hosted 26.2+ runner** (hosted
  GitHub macOS images can't exercise NAX) (`docs/rust-mlx-build.md`).

### Architecture boundaries that avoid baking in CUDA assumptions 📄
The seams already exist and should be treated as the contract for keeping Apple a peer runtime:
- **Engine indirection.** On macOS the worker maps a SceneWorks model id → a `gen_core` MLX engine id
  via `MODEL_TABLE` (`crates/sceneworks-worker/src/engines.rs:37`); providers self-register at link
  time via `inventory`. The manifest `adapter` field (`z_image_diffusers`, `ltx_video`) is the
  *torch* label — it does **not** dictate the Mac path. So "which runtime" is a target/registry
  decision, never hard-wired to CUDA.
- **Target-gated build seam.** `mlx-gen`/`mlx-rs` are `cfg(target_os = "macos")` deps, git-pinned by
  SHA; Linux/Windows resolve but never compile them (`docs/rust-mlx-build.md:8-11`).
- **Routing oracle.** `mac_rust_supported(job)` (`crates/sceneworks-core/src/jobs_store.rs:2046`) is
  the single source of truth for whether a job can run on the Apple path, backed by
  `MLX_ROUTED_MODELS` / `VIDEO_MLX_ROUTED_MODELS` (`jobs_store.rs:3099`, `:2264`, `:2361`).
- **Capability surface for clients.** `model_mac_support(model_id, model_type)`
  (`jobs_store.rs:2251`) returns `ModelMacSupport { supported, reason, features }` /
  `ModelMacFeatures { pose, reference, edit, lycoris, video_modes }` (`jobs_store.rs:2207-2233`),
  surfaced on `GET /api/v1/models` and gated client-side in `apps/web/src/macGating.js`
  (`docs/mac-rust-gaps.md:35-41`).
- **Warn-only rollout.** `SCENEWORKS_MLX_REQUIRED=1` logs unsupported surfaces without breaking; each
  surface flips to enforce only once ported or UI-gated (`docs/mac-rust-gaps.md:27-31`).

## Rust backend target (per AC)

- **Device-capability abstraction:** present via the engine registry + `mac_rust_supported` oracle +
  `ModelMacSupport`. **Recommended add (from the video/image feasibility work):** a *precision-aware
  VRAM/peak-memory* field on the manifest — the current `mlx.minMemoryGb` is a single optimistic
  number that the LTX run shows undercounts the real peak by ~1.7×. Admission should compare
  *measured peak per precision* against the device's free unified memory.
- **Runtime selection:** keep it a function of `cfg(target_os)` + the routing oracle, never of a
  CUDA-named adapter. No change needed.
- **Scheduling constraints:** Apple is memory-bound, not VRAM-count-bound; video jobs should carry a
  realistic peak so the scheduler doesn't admit an LTX clip that OOMs a 36/48 GB Mac. The 600 s
  forward-progress watchdog (`SCENEWORKS_VIDEO_STALL_SECS`) is adequate for cold loads observed here.
- **Adapter registration:** the link-time `inventory` registration + `MODEL_TABLE` is the right seam;
  new Apple engines register without touching the API/contract layer.
- **No new versioned Rust contract change is required** to call Apple a v1 runtime — the seams ship
  today. The one additive change worth making is the typed peak-memory/precision field above.

## Sources

Empirical: build + `nax_guard` + `zimage_real_weights_generates_one_image` +
`ltx_real_weights_with_audio` run on this M5 Max / macOS 26.5.1 (see numbers above).
Code/docs: `crates/sceneworks-worker/tests/nax_guard.rs`, `.cargo/config.toml`,
`docs/rust-mlx-build.md`, `docs/mac-rust-gaps.md`, `docs/sc-3633-mlx-port.md`,
`crates/sceneworks-worker/src/engines.rs`, `crates/sceneworks-core/src/jobs_store.rs`,
`config/manifests/builtin.models.jsonc`.
