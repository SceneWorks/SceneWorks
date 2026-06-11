# Shared multi-target build for the Rust API and Rust worker images. The builder
# stage (base image, workspace COPYs, cargo build) was previously copy-pasted
# between docker/rust-api.Dockerfile and docker/rust-worker.Dockerfile, differing
# only in the `-p` target and the runtime apt packages (sc-4284 / F-INFRA-7).
#
# Build a specific image with `--target` + `--build-arg BIN=…`; docker-compose
# sets both per service:
#   docker build -f docker/rust.Dockerfile --target rust-api   --build-arg BIN=sceneworks-rust-api   .
#   docker build -f docker/rust.Dockerfile --target rust-worker --build-arg BIN=sceneworks-rust-worker .

FROM rust:1-bookworm AS builder
# Which workspace binary to build (sceneworks-rust-api | sceneworks-rust-worker).
ARG BIN
WORKDIR /app

COPY Cargo.toml Cargo.lock rust-toolchain.toml rustfmt.toml ./
COPY crates ./crates
COPY apps/rust-api ./apps/rust-api
COPY apps/rust-worker ./apps/rust-worker
# Copied purely to satisfy workspace membership (the desktop crate is in the
# workspace but is not built into either image).
COPY apps/desktop ./apps/desktop

RUN cargo build -p "${BIN}" --release

# --- Rust API runtime ---------------------------------------------------------
FROM debian:bookworm-slim AS rust-api

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/sceneworks-rust-api /usr/local/bin/sceneworks-rust-api

CMD ["sceneworks-rust-api"]

# --- Rust worker runtime ------------------------------------------------------
FROM debian:bookworm-slim AS rust-worker

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/hf-cli \
    && /opt/hf-cli/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/hf-cli/bin/pip install --no-cache-dir "huggingface_hub[cli]>=0.36,<1"

ENV PATH="/opt/hf-cli/bin:${PATH}"

COPY --from=builder /app/target/release/sceneworks-rust-worker /usr/local/bin/sceneworks-rust-worker

CMD ["sceneworks-rust-worker"]
