# SceneWorks CodeGraph Notice

> [!WARNING]
> **Deprecated generated snapshot.** The former summary in this file described
> SceneWorks as a Python/FastAPI service after that runtime had been removed.
> Do not use historical CodeGraph summaries as architecture guidance. Use the
> source-owned references below until a replacement passes the regeneration
> contract in this document.

## Current ownership

- [`Cargo.toml`](Cargo.toml) defines the Rust workspace. The Axum HTTP, REST,
  SSE, and MCP surface is owned by
  [`apps/rust-api`](apps/rust-api/Cargo.toml).
- Native job execution and platform routing are owned by
  [`crates/sceneworks-worker`](crates/sceneworks-worker/Cargo.toml), with
  backend-neutral contracts and the MLX/candle runtime bundles pinned from the
  `SceneWorks/inference` repository.
- The product UI is React/Vite and is owned by
  [`apps/web`](apps/web/package.json). The desktop shell is Tauri and is owned
  by [`apps/desktop`](apps/desktop/Cargo.toml).
- [`docker-compose.yml`](docker-compose.yml) is the optional server deployment
  for the Rust API, native candle/CUDA worker, Rust utility workers, and web UI.
  Python files that remain under packaging or test tooling are not an
  application server or inference runtime.

The maintained product overview is [`README.md`](README.md), and contributor
architecture and validation guidance is in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Provenance

`CODEGRAPH.md` was written by external automation, not by a generator in this
repository. Git history records repeated commits authored by
`CodeGraph <codegraph@localhost>`; the last was
`7996b777734a42c07d9016be776b0ebdaf362b9a` on 2026-06-17. There is no tracked
CodeGraph configuration, generation script, package command, or GitHub Actions
workflow to reproduce the file.

At source commit `e4ef405d8bafc0b4d6644a19c64af04ae3860dc8`, the available external
CodeGraph report still indexed deleted Python worker paths and repeated the
obsolete FastAPI classification. It therefore cannot provide an exact,
source-correct regeneration for this revision.

## Regeneration contract

Any generated replacement for this notice must be validated against the exact
repository revision it names and must:

1. identify the Rust workspace, Axum API, React/Vite UI, Tauri desktop shell,
   native MLX/candle workers, and optional Docker deployment;
2. identify the pinned `SceneWorks/inference` dependency rather than claiming
   that this repository is self-contained;
3. never classify SceneWorks as a Python API or FastAPI service, or cite the
   deleted `apps/api` and `apps/worker/scene_worker` trees as current code; and
4. retain this warning instead of publishing an unverified inferred summary.
