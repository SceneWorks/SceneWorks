# syntax=docker/dockerfile:1.7
# Shared multi-target build for the Rust API and Rust worker images. The builder
# stage (base image, workspace COPYs, cargo build) was previously copy-pasted
# between docker/rust-api.Dockerfile and docker/rust-worker.Dockerfile, differing
# only in the `-p` target and the runtime apt packages (sc-4284 / F-INFRA-7).
#
# Build a specific image with `--target` + `--build-arg BIN=…`; docker-compose
# sets both per service:
#   docker build -f docker/rust.Dockerfile --target rust-api   --build-arg BIN=sceneworks-rust-api   .
#   docker build -f docker/rust.Dockerfile --target rust-api-embed --build-arg BIN=sceneworks-rust-api .
#   docker build -f docker/rust.Dockerfile --target rust-worker --build-arg BIN=sceneworks-rust-worker .

# Production SPA bundle for the opt-in rust-api-embed target. The plain rust-api
# target has no dependency on this stage, so it neither installs Node packages nor
# embeds web assets. The license corpus is imported directly by the production UI.
FROM node:22-bookworm-slim AS web-builder
WORKDIR /app
COPY apps/web/package.json apps/web/package-lock.json ./apps/web/
RUN --mount=type=cache,target=/root/.npm npm ci --prefix apps/web
COPY apps/web ./apps/web
COPY apps/desktop/licenses ./apps/desktop/licenses
# Explicit empty (not unset): apps/web/src/api.js maps this to window.location.origin.
ENV VITE_API_BASE_URL=""
# Guard the built artifact, not just the Dockerfile environment declaration:
# an unset value bakes the local-Vite fallback into the production bundle.
RUN npm run build --prefix apps/web \
    && ! grep -R -q "http://localhost:8000" apps/web/dist

FROM rust:1-bookworm AS builder
# Which workspace binary to build (sceneworks-rust-api | sceneworks-rust-worker).
ARG BIN=sceneworks-rust-api
WORKDIR /app

COPY Cargo.toml Cargo.lock rust-toolchain.toml rustfmt.toml ./
COPY .cargo/config.toml ./.cargo/config.toml
COPY crates/sceneworks-core/Cargo.toml ./crates/sceneworks-core/Cargo.toml
COPY crates/sceneworks-worker/Cargo.toml ./crates/sceneworks-worker/Cargo.toml
COPY crates/sceneworks-image-quality/Cargo.toml ./crates/sceneworks-image-quality/Cargo.toml
COPY crates/sceneworks-memory-adapter/Cargo.toml ./crates/sceneworks-memory-adapter/Cargo.toml
COPY crates/sceneworks-mcp/Cargo.toml ./crates/sceneworks-mcp/Cargo.toml
COPY apps/rust-api/Cargo.toml ./apps/rust-api/Cargo.toml
COPY apps/rust-worker/Cargo.toml ./apps/rust-worker/Cargo.toml
COPY apps/desktop/Cargo.toml ./apps/desktop/Cargo.toml

RUN mkdir -p \
      apps/desktop/src \
      apps/rust-api/src \
      apps/rust-worker/src \
      crates/sceneworks-core/src \
      crates/sceneworks-worker/src \
      crates/sceneworks-image-quality/src \
      crates/sceneworks-memory-adapter/src/bin \
      crates/sceneworks-mcp/src \
    && printf 'fn main() {}\n' > apps/desktop/src/main.rs \
    && printf 'fn main() {}\n' > apps/rust-api/src/main.rs \
    && printf 'fn main() {}\n' > apps/rust-worker/src/main.rs \
    && touch crates/sceneworks-core/src/lib.rs crates/sceneworks-worker/src/lib.rs crates/sceneworks-image-quality/src/lib.rs crates/sceneworks-mcp/src/lib.rs \
      crates/sceneworks-memory-adapter/src/lib.rs \
      crates/sceneworks-memory-adapter/src/bin/candle.rs \
      crates/sceneworks-memory-adapter/src/bin/mlx.rs

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo fetch --locked

COPY crates ./crates
COPY apps/rust-api ./apps/rust-api
COPY apps/rust-worker ./apps/rust-worker
# Copied purely to satisfy workspace membership (the desktop crate is in the
# workspace but is not built into either image).
COPY apps/desktop/Cargo.toml ./apps/desktop/Cargo.toml
COPY apps/desktop/build.rs ./apps/desktop/build.rs
# The builtin catalog: `sceneworks-core` embeds these manifests via `include_str!`
# so the API can seed an empty config dir, which means they must exist in the
# build context (not just the runtime bind mount) or the compile can't read them.
COPY config ./config
# Same constraint, second source (sc-16080): `sceneworks-core::memory_calibration`
# embeds the generated calibration evidence via `include_str!`, and that embed is NOT
# test-gated, so a release build of the API cannot compile without it.
#
# Deliberately the single embedded FILE rather than `docs/generated`: that directory
# also holds `memory-matrix.json`, which is regenerated and re-hashed by any change to
# the selector's source, so copying the directory would invalidate this layer and
# rebuild the whole Rust graph on edits the image does not depend on.
#
# Every `include_str!`/`include_bytes!` that reaches outside its crate needs a line
# here, or the Docker build breaks while `cargo build` on a checkout stays green — the
# two see different trees. Embeds inside `mod tests` are exempt: this stage builds
# `--release` without tests.
COPY docs/generated/memory-calibration-evidence.json ./docs/generated/

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --offline -p "${BIN}" --release \
    && mkdir -p /out \
    && cp "target/release/${BIN}" "/out/${BIN}"

# --- Rust API runtime ---------------------------------------------------------
FROM debian:bookworm-slim AS rust-api

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /out/sceneworks-rust-api /usr/local/bin/sceneworks-rust-api

CMD ["sceneworks-rust-api"]

# --- Rust API + embedded production web runtime ------------------------------
# Start from the unchanged API builder, add only the Node-built dist tree that
# rust-embed consumes, and rebuild the API with the opt-in feature. Deriving the
# runtime from rust-api keeps its packages and command identical to the plain image.
FROM builder AS embed-builder

COPY --from=web-builder /app/apps/web/dist ./apps/web/dist

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --offline -p sceneworks-rust-api --release --features embed-web \
    && mkdir -p /out \
    && cp target/release/sceneworks-rust-api /out/sceneworks-rust-api

FROM rust-api AS rust-api-embed

COPY --from=embed-builder /out/sceneworks-rust-api /usr/local/bin/sceneworks-rust-api

# --- Rust worker runtime ------------------------------------------------------
FROM debian:bookworm-slim AS rust-worker

# ffmpeg: candle video lanes encode mp4. No Python here: model downloads are the
# native in-process `ensure_hf_files_cached` path (sc-12227 / sc-12232), so the
# retired `huggingface_hub[cli]` venv is gone — keeping this image Python-free
# (epic 3482). Don't re-add the hf CLI; it would bypass the native download watchdog.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /out/sceneworks-rust-worker /usr/local/bin/sceneworks-rust-worker

CMD ["sceneworks-rust-worker"]

# --- Candle GPU worker build (CUDA; compute_80 PTX → sm_120) ------------------
# Separate builder: the candle backend needs the CUDA toolkit (nvcc) to compile
# candle-kernels, which the stock rust:bookworm builder above lacks. CUDA 12.9 is the
# toolchain the candle lane builds + validates against (server-candle-linux.yml + the
# dev box). CUDA_COMPUTE_CAP=80 emits compute_80 PTX the driver JITs forward to sm_120
# (RTX PRO 6000) — one binary covers Ampere→Blackwell, matching the Windows desktop
# bundle (build-sidecar.mjs) and the Linux candle CI lane. NB: that forward-JIT story
# holds for the DENSE (PTX) kernels only; the GGUF quant/moe kernels are a static SASS
# libmoe.a whose multi-arch coverage (sm_80+sm_90+sm_120 + compute_120 PTX) comes from
# the root Cargo.toml [patch] onto the inference repo's vendored candle-kernels
# (sc-7544 / sc-13510 — guarded by candle_kernels_patch_guard). The backend-candle
# feature lives on the sceneworks-worker library crate, enabled through the thin binary
# (epic 5483 Phase 7 / sc-5503 — the Docker torch→candle cutover).
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04 AS candle-builder
ENV DEBIAN_FRONTEND=noninteractive
# build-essential + pkg-config for the CUDA/native build scripts; libssl-dev because
# native-tls (pulled transitively by the worker's deps) links system OpenSSL on Linux
# — the Windows host build uses schannel instead, so this only surfaces here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*
# Rust toolchain. The default rustup profile ships rustfmt+clippy; the COPY'd
# rust-toolchain.toml (a pinned concrete version + those components) is what any
# in-repo cargo actually resolves, auto-installed by rustup on first use.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:/usr/local/cuda/bin:${PATH}"
ENV CUDA_PATH=/usr/local/cuda
ARG CUDA_COMPUTE_CAP=80
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}
WORKDIR /app

# Dependency-graph layer (mirrors the builder above): COPY the manifests + stub
# entrypoints, then `cargo fetch` so the candle dependency tree (candle-gen +
# candle/cudarc, all public git deps) caches independently of source edits.
COPY Cargo.toml Cargo.lock rust-toolchain.toml rustfmt.toml ./
COPY .cargo/config.toml ./.cargo/config.toml
COPY crates/sceneworks-core/Cargo.toml ./crates/sceneworks-core/Cargo.toml
COPY crates/sceneworks-worker/Cargo.toml ./crates/sceneworks-worker/Cargo.toml
COPY crates/sceneworks-image-quality/Cargo.toml ./crates/sceneworks-image-quality/Cargo.toml
COPY crates/sceneworks-memory-adapter/Cargo.toml ./crates/sceneworks-memory-adapter/Cargo.toml
COPY crates/sceneworks-mcp/Cargo.toml ./crates/sceneworks-mcp/Cargo.toml
COPY apps/rust-api/Cargo.toml ./apps/rust-api/Cargo.toml
COPY apps/rust-worker/Cargo.toml ./apps/rust-worker/Cargo.toml
COPY apps/desktop/Cargo.toml ./apps/desktop/Cargo.toml
RUN mkdir -p \
      apps/desktop/src apps/rust-api/src apps/rust-worker/src \
      crates/sceneworks-core/src crates/sceneworks-worker/src crates/sceneworks-image-quality/src \
      crates/sceneworks-memory-adapter/src/bin crates/sceneworks-mcp/src \
    && printf 'fn main() {}\n' > apps/desktop/src/main.rs \
    && printf 'fn main() {}\n' > apps/rust-api/src/main.rs \
    && printf 'fn main() {}\n' > apps/rust-worker/src/main.rs \
    && touch crates/sceneworks-core/src/lib.rs crates/sceneworks-worker/src/lib.rs crates/sceneworks-image-quality/src/lib.rs crates/sceneworks-mcp/src/lib.rs \
      crates/sceneworks-memory-adapter/src/lib.rs \
      crates/sceneworks-memory-adapter/src/bin/candle.rs \
      crates/sceneworks-memory-adapter/src/bin/mlx.rs
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    cargo fetch --locked

COPY crates ./crates
COPY apps/rust-api ./apps/rust-api
COPY apps/rust-worker ./apps/rust-worker
# Workspace-membership only (not built into this image).
COPY apps/desktop/Cargo.toml ./apps/desktop/Cargo.toml
COPY apps/desktop/build.rs ./apps/desktop/build.rs
# The builtin catalog, embedded via include_str! by sceneworks-core (see above).
COPY config ./config

# nvcc compiles every candle provider's CUDA kernels here (compiling needs no GPU).
# The general Candle kernels retain compute_80 PTX, but the GGUF/MoE kernels in
# libmoe.a need explicit cubins. Inspect the exact executable copied to /out
# (rather than a Cargo build-directory candidate) so a stale cache entry cannot
# satisfy the check when the linked artifact is wrong. This prevents a
# dropped/changed vendored patch from silently narrowing the documented RunPod
# matrix (sc-10369; the same guard protects desktop packaging in
# apps/desktop/scripts/build-sidecar.mjs).
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --offline -p sceneworks-rust-worker --release \
        --features sceneworks-worker/backend-candle \
    && mkdir -p /out \
    && cp target/release/sceneworks-rust-worker /out/sceneworks-rust-worker \
    && cuobjdump --list-elf /out/sceneworks-rust-worker > /tmp/worker-elf.txt \
    && cuobjdump --list-ptx /out/sceneworks-rust-worker > /tmp/worker-ptx.txt \
    && grep -q 'sm_80\.cubin' /tmp/worker-elf.txt \
    && grep -q 'sm_90\.cubin' /tmp/worker-elf.txt \
    && grep -q 'sm_120\.cubin' /tmp/worker-elf.txt \
    && grep -q 'sm_120\.ptx' /tmp/worker-ptx.txt

# --- Candle GPU worker runtime (CUDA-12) -------------------------------------
# The off-Mac torch replacement: Docker GPU inference runs on the native candle/CUDA
# worker, not the Python torch worker. The CUDA-runtime base provides cudart/cublas/
# cublasLt for candle; the `ort` CV-aux lanes (DWPose/YOLO/Real-ESRGAN, sc-6209) get a
# version-matched onnxruntime-gpu + its cuDNN-9 / cuFFT / nvJitLink / nvRTC deps from
# PyPI, dlopened via ORT_DYLIB_PATH (the `ort` crate links load-dynamic).
#
# ubuntu24.04 (Python 3.12) on purpose: onnxruntime-gpu >= 1.26 (the `ort` API floor —
# see below) ships no cp310 Linux wheel, so the 22.04 base's Python 3.10 caps at
# 1.23.2 and can't satisfy it; cp312 has 1.26.0. The builder stays on 22.04 (it needs
# no Python) — its older-glibc binary runs fine on 24.04 (glibc is backward-compatible).
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04 AS rust-worker-candle
ENV DEBIAN_FRONTEND=noninteractive
# ffmpeg: candle video lanes encode mp4. python3/venv: stage onnxruntime-gpu (model
# downloads are the native in-process path, no hf CLI — sc-12227 / sc-12232).
# libgomp1: onnxruntime's OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl ffmpeg python3 python3-venv libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# onnxruntime-gpu + the CUDA-12 deps its providers_cuda needs that the CUDA-runtime
# base doesn't ship — cuDNN-9 (incl. lazily-loaded sub-engines), cuFFT, nvJitLink,
# nvRTC. onnxruntime-gpu does NOT declare these as hard deps (they sit behind its
# `cuda`/`cudnn` extras), so request them explicitly (sc-6209). 1.26.0 matches the ORT
# API version the `ort` crate requests (its `api-26` feature, pinned in the workspace
# Cargo.toml); validated on RTX PRO 6000 with cuDNN-cu12 9.23. The two are bound at
# COMPILE time by `PROVISIONED_ONNXRUNTIME_MINOR` in crates/sceneworks-worker/src/
# pose_jobs.rs — the crate's floor is set by that api-N FEATURE, not by its rc version,
# so re-read it after any `ort` bump rather than assuming a version maps to an API.
#
# Do not "just bump" this to 1.27+: that release moved the CUDA execution provider to
# CUDA 13 (`nvidia-*-cu13`), which is incompatible with the CUDA 12.9 base and the
# cu12 pins below.
#
# The four nvidia-*-cu12 versions below are PINNED to exactly match the desktop CUDA
# provisioner (apps/desktop/src/cuda_provision.rs COMPONENTS table, REDIST_VERSION
# `cuda12.9-ort1.26.0-cudnn9.23-1`) so both surfaces stage the identical CUDA set that
# onnxruntime-gpu 1.26.0 was validated against — reproducible server builds, no silent
# cuDNN 10.x / floating-range drift (sc-13611). Each also satisfies onnxruntime-gpu
# 1.26.0's own declared ranges (cudnn~=9.0, cufft~=11.0, cuda-nvrtc~=12.0) and the
# CUDA 12.9 runtime base. Pip --require-hashes hardening is tracked in sc-14078.
ENV ORT_PY_SITE=/opt/ort/lib/python3.12/site-packages
ARG ONNXRUNTIME_GPU_VERSION=1.26.0
# Mirror apps/desktop/src/cuda_provision.rs — keep in lockstep if that manifest bumps.
ARG NVIDIA_CUDNN_CU12_VERSION=9.23.0.39
ARG NVIDIA_CUFFT_CU12_VERSION=11.4.1.4
ARG NVIDIA_NVJITLINK_CU12_VERSION=12.9.86
ARG NVIDIA_CUDA_NVRTC_CU12_VERSION=12.9.86
RUN python3 -m venv /opt/ort \
    && /opt/ort/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/ort/bin/pip install --no-cache-dir \
        "onnxruntime-gpu==${ONNXRUNTIME_GPU_VERSION}" \
        "nvidia-cudnn-cu12==${NVIDIA_CUDNN_CU12_VERSION}" \
        "nvidia-cufft-cu12==${NVIDIA_CUFFT_CU12_VERSION}" \
        "nvidia-nvjitlink-cu12==${NVIDIA_NVJITLINK_CU12_VERSION}" \
        "nvidia-cuda-nvrtc-cu12==${NVIDIA_CUDA_NVRTC_CU12_VERSION}" \
    && ORT_SO="$(ls ${ORT_PY_SITE}/onnxruntime/capi/libonnxruntime.so* | head -1)" \
    && test -n "${ORT_SO}" \
    && ln -sf "${ORT_SO}" "${ORT_PY_SITE}/onnxruntime/capi/libonnxruntime.so"

# Point the `ort` crate (load-dynamic) at the staged onnxruntime, and tell ort_cuda
# where the CUDA-12 runtime (base) + cuDNN-9 (pip wheel) live. LD_LIBRARY_PATH is the
# Linux analogue of the Windows PATH-prepend in ort_cuda::preload_cuda_dylibs — the
# dynamic linker resolves the providers' CUDA/cuDNN deps + cuDNN's lazy sub-engines.
ENV ORT_DYLIB_PATH=${ORT_PY_SITE}/onnxruntime/capi/libonnxruntime.so
ENV SCENEWORKS_ORT_CUDA_DIR=/usr/local/cuda/lib64
ENV SCENEWORKS_ORT_CUDNN_DIR=${ORT_PY_SITE}/nvidia/cudnn/lib
ENV LD_LIBRARY_PATH=${ORT_PY_SITE}/onnxruntime/capi:${ORT_PY_SITE}/nvidia/cudnn/lib:${ORT_PY_SITE}/nvidia/cufft/lib:${ORT_PY_SITE}/nvidia/nvjitlink/lib:${ORT_PY_SITE}/nvidia/cuda_nvrtc/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH="/opt/ort/bin:${PATH}"

COPY --from=candle-builder /out/sceneworks-rust-worker /usr/local/bin/sceneworks-rust-worker

CMD ["sceneworks-rust-worker"]

# --- Combined RunPod GPU runtime ---------------------------------------------
# RunPod starts one image per pod, so this target packages the embedded-web API
# and the candle/CUDA worker together. The PID-1 entrypoint starts the API, waits
# for its loopback health endpoint, then starts the worker in `auto` mode. The
# worker's existing supervisor discovers the visible NVIDIA GPU(s) and starts a
# dedicated CPU utility child alongside the GPU child; those children advertise
# disjoint capabilities, so generation cannot leak onto the utility lane and
# downloads/imports/exports cannot occupy the GPU lane.
#
# Keep this runtime based on rust-worker-candle: it is the validated CUDA 12.9.1
# + cuDNN 9 + onnxruntime-gpu image and already contains ffmpeg. Model Manager
# downloads run through the worker's native in-process downloader; intentionally
# do not install the retired Hugging Face CLI (sc-12227 / sc-12232).
FROM rust-worker-candle AS runpod

COPY --from=embed-builder /out/sceneworks-rust-api /usr/local/bin/sceneworks-rust-api
COPY docker/runpod-entrypoint.sh /usr/local/bin/sceneworks-runpod-entrypoint
RUN chmod 0755 /usr/local/bin/sceneworks-runpod-entrypoint

ENV SCENEWORKS_API_HOST=0.0.0.0 \
    SCENEWORKS_API_PORT=8010 \
    SCENEWORKS_API_URL=http://127.0.0.1:8010 \
    SCENEWORKS_VOLUME=/workspace \
    SCENEWORKS_JOBS_DB_PATH=/tmp/sceneworks/cache/jobs.db \
    SCENEWORKS_CANDLE_REQUIRED=1 \
    SCENEWORKS_CANDLE_UNSUPPORTED_MODE=enforce \
    SCENEWORKS_DISABLE_MODEL_SIZE_ESTIMATE=1 \
    SCENEWORKS_WORKER_ID=runpod-worker-0

EXPOSE 8010
VOLUME ["/workspace"]
HEALTHCHECK --interval=20s --timeout=5s --retries=3 --start-period=10s \
    CMD curl -fsS "http://127.0.0.1:${SCENEWORKS_API_PORT:-8010}/api/v1/health" >/dev/null || exit 1

ENTRYPOINT ["/usr/local/bin/sceneworks-runpod-entrypoint"]
