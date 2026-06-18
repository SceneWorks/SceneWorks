# Worker & routing observability (epic 3447)

This is the operator's guide to "what happened this session?" — especially **why a
job ran on the MLX worker vs the Python torch/MPS worker**. It documents where logs
live, the structured-event vocabulary, and the in-app Logs screen.

## Where logs live

The desktop app (Tauri wrapper) captures each sidecar's stdout/stderr and appends it
to a per-process file under the platform log dir (`apps/desktop/src/setup.rs::logs_dir`):

| File | Process | Content |
| --- | --- | --- |
| `api.log` | `sceneworks-api` | API events incl. `mlx_route_decision`, plus axum/startup output |
| `worker.log` | Python torch worker | `emit_worker_event` JSON (load/lora/inference + `backend`) |
| `mlx-worker.log` | Rust MLX GPU worker | `claim_lock_contention`, `image_inference_*`, supervisor events |

- **macOS:** `~/Library/Logs/SceneWorks/`
- **Windows:** `%LOCALAPPDATA%\SceneWorks\logs\`
- **Linux:** `$XDG_STATE_HOME/sceneworks/logs` (or `~/.local/state/sceneworks/logs`)

You rarely need to open these directly — see the in-app Logs screen below.

## In-app Logs screen

**System → Logs** (`apps/web/src/screens/LogsScreen.jsx`). Read-only, live-tailing,
filter by source (api / worker / mlx-worker) and level (info / warn / error), free-text
search. Routing (`mlx_route_decision`) and contention (`claim_lock_contention`) events
are visually highlighted. Click a row to expand its raw structured event.

Data source:
- **Desktop:** `get_session_logs` Tauri command reads a process-global ring buffer fed
  by the same stdout capture that writes the three files (`apps/desktop/src/setup.rs`,
  sc-3451). "Current session" = the desktop process's lifetime.
- **Web / Docker:** `GET /api/v1/logs` returns the API process's own event buffer
  (`apps/rust-api/src/logs.rs`, sc-3453). The shared `LogEntry` shape
  (`sceneworks_core::session_log`) makes the screen source-agnostic.

## Event vocabulary

All structured events are one JSON object per line: `{ event, reportedAt, ...payload }`
(matches the Python worker's `emit_worker_event`). `LogEntry` infers a `level`
(`info`/`warn`/`error`) and a compact `message` summary from each line.

### Routing — `mlx_route_decision` (API, sc-3449)

Emitted at claim time whenever a claim is routing-relevant. **This is the line that
answers "why did this run on torch instead of MLX?"**

| field | meaning |
| --- | --- |
| `decision` | `deferred_to_mlx` \| `claimed_by_mlx` \| `fell_back_to_torch` \| `explicit_gpu` |
| `reason` | `idle_mlx_available` \| `mlx_worker` \| `no_idle_mlx_worker` \| `explicit_gpu` |
| `model`, `jobType`, `requestedGpu`, `workerId`, `gpuId` | context |

- `deferred_to_mlx` / `idle_mlx_available` — a torch worker yielded the job to an idle MLX worker.
- `claimed_by_mlx` / `mlx_worker` — the MLX worker took an MLX-eligible job (the happy path).
- `fell_back_to_torch` / `no_idle_mlx_worker` — **an MLX-eligible job ran on torch because no idle MLX worker was available** (restart churn, busy, or down).
- `explicit_gpu` — the user pinned a specific GPU, so MLX routing was intentionally bypassed.

### Claim contention — `claim_lock_contention` (Rust worker, sc-3448)

Emitted when a worker's claim poll hits `database is locked` (warn level): `workerId`,
`gpuId`, `consecutiveFailures`, `retryInSeconds`. Should be **absent** under normal load
after the `busy_timeout` + `BEGIN IMMEDIATE` hardening — recurring contention means a
regression there.

### Generation — `image_inference_start` / `image_inference_complete` (Rust MLX worker, sc-3450)

Per-image lifecycle on the MLX path (parity with the Python worker), emitted from the
shared streaming consumer (`image_jobs::consume_gen_events`): `jobId`, `imageIndex`,
`imageCount`, `backend` (`mlx`). Confirms an MLX job is actually progressing image-by-image.

### Pipeline load — `image_pipeline_load_start` / `image_pipeline_load_complete` (Rust MLX worker, sc-3450)

Brackets the engine load (`mlx_gen::load`) inside each per-family blocking generation
closure (all five MLX image families): `jobId`, `engine` (the mlx-gen engine id, e.g.
`qwen_image_edit`, `sdxl`, `z_image_turbo_control`), `adapterCount`. A `start` with **no
matching `complete`** means the load failed (the job then errors). A long gap between the
two is the signature of a cold-weight load — the prime suspect when an MLX job looks
"stuck" before its first `image_inference_start`.

> **No separate `image_distill_lora_fuse_*` / `image_lora_apply_*` events on the MLX path**
> (the torch worker emits these as distinct phases). On MLX, `mlx_gen::load` is a single
> atomic call that *also* fuses any distill LoRA and applies user LoRAs (`spec.with_adapters`),
> so there is no separable fuse/apply step to bracket. The adapter total (distill + user) is
> reported as `adapterCount` on the `image_pipeline_load_*` events instead — same diagnostic
> information, accurate to the Rust engine's architecture, rather than fabricated
> zero-duration sub-phase events.

### MLX generator cache idle eviction — `mlx_generator_cache_idle_evicted`

Emitted by the Rust MLX worker when the shared generator cache drops its
resident generator after the idle timeout: `engine`, `idleSeconds`. This is
expected after the worker has been idle for
`SCENEWORKS_GENERATOR_CACHE_IDLE_SECONDS` seconds (default 300). It should
correlate with the worker releasing cached Metal/MLX allocations before the
next generation cold-loads weights again.

## Diagnosing "MLX-eligible job ran on torch/MPS"

1. Open **System → Logs**, filter source = `api`, search `mlx_route_decision`.
2. Find the decision for the job's model. `fell_back_to_torch` + `no_idle_mlx_worker`
   means the MLX worker wasn't idle/claimable at claim time — check `mlx-worker.log`
   (filter source = `mlx-worker`) for restarts or `claim_lock_contention`.
3. `claimed_by_mlx` plus `image_inference_*` on `mlx-worker` confirms a true MLX run.
4. The asset's recorded `backend` (`mlx` vs `mps`/`cuda`) is the ground truth for where it ran.
