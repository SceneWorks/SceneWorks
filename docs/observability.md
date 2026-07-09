# Worker & routing observability (epic 3447)

This is the operator's guide to "what happened this session?" — especially **why a
job ran on the MLX worker vs the Python torch/MPS worker**. It documents where logs
live, the structured-event vocabulary, and the in-app Logs screen.

## Where logs live

The desktop app (Tauri wrapper) captures each sidecar's stdout/stderr and appends it
to a per-process file under the platform log dir (`apps/desktop/src/setup.rs::logs_dir`):

| File | Process | Content |
| --- | --- | --- |
| `api.log` | `sceneworks-api` | API events incl. `gpu_route_decision`, plus axum/startup output |
| `worker.log` | Python torch worker | `emit_worker_event` JSON (load/lora/inference + `backend`) |
| `mlx-worker.log` | Rust MLX GPU worker | `claim_lock_contention`, `image_inference_*`, supervisor events |

- **macOS:** `~/Library/Logs/SceneWorks/`
- **Windows:** `%LOCALAPPDATA%\SceneWorks\logs\`
- **Linux:** `$XDG_STATE_HOME/sceneworks/logs` (or `~/.local/state/sceneworks/logs`)

You rarely need to open these directly — see the in-app Logs screen below.

## Logging backbone

All Rust crates log through [`tracing`]. A single init function —
`sceneworks_core::observability::init_logging()` (and the API's buffer-aware
`init_logging_with_buffer()`) — installs the subscriber from each binary's `main`
(`apps/rust-api`, `apps/rust-worker`, and the desktop shell). It is idempotent, so
the embedded-worker and standalone-worker paths can both call it safely.

**Format-adaptive output (`SCENEWORKS_LOG_FORMAT = json | pretty | auto`, default
`auto`).** In `auto` the process emits **pretty**, human-readable colored lines when
`stdout` is an interactive TTY (a developer running `cargo run`), and **JSON** — one
object per line — otherwise. "Otherwise" is every deployment that matters here: a
Tauri sidecar whose stdout the desktop captures, a Docker container, or any pipe.
So desktop sidecars and headless servers both emit JSON (what the ring buffer and
log ingestion want), while a terminal stays readable. Force either with
`SCENEWORKS_LOG_FORMAT=json` / `=pretty`.

All of SceneWorks' own operational output — startup/lifecycle lines, the structured
event vocabulary below, and the error/warn paths — goes through `tracing`, so in JSON
mode every SceneWorks line is one JSON object. (Plain `println!`/`eprintln!` remain
only in `#[test]`/E2E code, which never runs in a server/Docker process.) Two
non-structured lines can still appear and are deliberately tolerated rather than
forced into JSON: a linked backend / third-party library writing its own line to
stdout/stderr, and the Rust runtime's terminal `Error: …` line when a process exits
with a fatal error (kept as-is so the non-zero exit code still drives the
supervisor's crash attribution). The ring-buffer ingestion handles both — a line
that isn't a JSON object is kept verbatim and level-inferred — so a stray line is
surfaced, not dropped.

**Filtering (`RUST_LOG`).** Honored via `EnvFilter`; the default when unset is
`info,sceneworks=debug`.

**Levels are declared, not inferred.** Each event's severity is the `tracing` level
chosen at the call site (`error!` / `warn!` / `info!` / `debug!`), carried as an
explicit `level` field in the JSON envelope. `session_log` trusts that declared
level verbatim and only falls back to its text/name heuristic for legacy or plain
lines that lack one (e.g. the Python worker's `emit_worker_event`). This is what
makes the Logs-screen `level` filter trustworthy — filtering by `level=error` no
longer silently drops a real error, and a routine 4xx logged at `debug` is not
falsely promoted to error by its `_error`-suffixed name.

Secret redaction is unchanged: `session_log::redact_secrets` still scrubs
tokens / api-keys / bearer / authorization on ingestion before anything is
persisted or surfaced.

[`tracing`]: https://docs.rs/tracing

## In-app Logs screen

**System → Logs** (`apps/web/src/screens/LogsScreen.jsx`). Read-only, live-tailing,
filter by source (api / worker / mlx-worker) and level (info / warn / error), free-text
search. Routing (`gpu_route_decision`) and contention (`claim_lock_contention`) events
are visually highlighted. Click a row to expand its raw structured event.

Data source:
- **Desktop:** `get_session_logs` Tauri command reads a process-global ring buffer fed
  by the same stdout capture that writes the three files (`apps/desktop/src/setup.rs`,
  sc-3451). "Current session" = the desktop process's lifetime.
- **Web / Docker:** `GET /api/v1/logs` returns the API process's own event buffer
  (`apps/rust-api/src/logs.rs`, sc-3453). The shared `LogEntry` shape
  (`sceneworks_core::session_log`) makes the screen source-agnostic.

## Event vocabulary

All structured events are one JSON object per line:
`{ event, level, reportedAt, ...payload }` (the Rust crates emit this via the
`tracing` backbone above; the Python worker's `emit_worker_event` emits the same
shape minus the declared `level`). `LogEntry` reads the declared `level` when present
(`error`/`warn`/`info`/`debug`), falling back to a heuristic otherwise, and derives a
compact `message` summary from each line.

### Routing — `gpu_route_decision` (API, sc-3449)

Emitted at claim time whenever a claim is routing-relevant. **This is the line that
answers "which backend ran this job?"** Every label is named after the backend that
actually claimed the job — never as a deficiency, so the line never reads as if a worker
is missing on a platform that never had one (e.g. there is no MLX worker off-Mac).

| field | meaning |
| --- | --- |
| `decision` | `deferred_to_mlx` \| `claimed_by_mlx` \| `claimed_by_candle` \| `claimed_by_gpu` \| `explicit_gpu` |
| `reason` | `idle_mlx_available` \| `mlx_worker` \| `candle_worker` \| `gpu_worker` \| `explicit_gpu` |
| `model`, `jobType`, `requestedGpu`, `workerId`, `gpuId` | context |

- `claimed_by_mlx` / `mlx_worker` — the MLX worker took the job (the Mac GPU path).
- `claimed_by_candle` / `candle_worker` — the candle (Windows/Linux CUDA) worker took the job. **The expected off-Mac happy path**, not a fallback. Candle is identified by the `candle` capability marker (it runs on a real GPU index, so `gpuId` alone can't distinguish it).
- `claimed_by_gpu` / `gpu_worker` — a defensive catch-all: a GPU worker that is neither MLX nor candle took the job. With the Python torch worker retired from every surface, this should not appear in practice; it names no specific backend.
- `deferred_to_mlx` / `idle_mlx_available` — a non-mlx worker yielded the job to an idle MLX worker (Mac only).
- `explicit_gpu` — the user pinned a specific GPU, so backend auto-routing was intentionally bypassed.

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

### Generator cache idle eviction — `generator_cache_idle_evicted`

Emitted by the Rust worker (MLX on Mac, candle/CUDA off-Mac — the generator cache is
shared code) when it drops its resident generator after the idle timeout: `engine`,
`idleSeconds`. This is **expected** after the worker has been idle for
`SCENEWORKS_GENERATOR_CACHE_IDLE_SECONDS` seconds (default 300) and is logged at info
level. It correlates with the worker releasing cached GPU allocations (Metal/MLX or
CUDA) before the next generation cold-loads weights again.

### API errors — `api_error` (API)

Emitted from `ApiError`'s `IntoResponse` so no failure leaves the server without a
trace. Fields: `status` (HTTP code), `detail` (the message returned to the client).
**5xx responses log at `error`** (an untyped internal failure that an operator must
see); routine typed **4xx responses log at `debug`** so expected validation/not-found
churn doesn't drown the error level. Filtering the Logs screen by `level=error` and
searching `api_error` surfaces exactly the server-side failures.

### Auth rejections — `auth_rejected` (API)

Emitted by `auth::access_control` when a request to a protected route is rejected for
a missing/invalid access token (warn level). Fields: `path` (the request path, no
query string), `reason` (`missing_or_invalid_token`), `status` (401). The token /
secret is deliberately **never** logged. Previously these rejections returned 401
with no server-side trace.

## Diagnosing "which backend ran this job?"

1. Open **System → Logs**, filter source = `api`, search `gpu_route_decision`.
2. Find the decision for the job's model. The `decision`/`reason` names the backend
   that claimed it: `claimed_by_candle` (Windows/Linux CUDA), `claimed_by_mlx` (Mac),
   or `deferred_to_mlx` (Mac, yielded to the MLX worker). None of these means anything
   is missing. (`claimed_by_gpu` is a generic catch-all that should not appear on a
   shipped surface.)
3. **On Mac, to confirm a true MLX run:** `claimed_by_mlx` plus `image_inference_*` on
   `mlx-worker`. If you instead see `claimed_by_gpu` for an MLX-eligible model, the MLX
   worker wasn't idle/claimable at claim time — check `mlx-worker.log` for restarts or
   `claim_lock_contention`.
4. The asset's recorded `backend` (`mlx` / `mps` / `cuda`) is the ground truth for where
   it ran.

## Generation metrics & the Stats screen (epic 10402)

Distinct from the session logs above, SceneWorks records **structured per-run
metrics for every job** the worker runs and surfaces them for comparison.

### What's captured

On completion the worker POSTs a `GenerationMetrics` block to
`POST /api/v1/jobs/:id/metrics`, persisted 1:1 by job id in the
`generation_metrics` companion table in `jobs.db` (kept out of the hot `jobs`
row). Three partial blocks coalesce-merge into one row:

- **Hardware** — a probe wrapping the dispatch (`crates/sceneworks-worker/src/job_metrics.rs`):
  `peakMemoryBytes` / `peakMemoryPct` (MLX `get_peak_memory` on macOS, sampled
  `nvidia-smi memory.used` on candle), best-effort `peakGpuLoadPct` (sampled at the
  heartbeat cadence), `totalMs`, `backend`. Restores + broadens the per-job peak
  capture that the Python→Rust cutover dropped (sc-2086).
  > `peakGpuLoadPct` is **best-effort**: a background task samples GPU load at a
  > ~1s cadence during the job (the same unprivileged `ioreg` "Device Utilization %"
  > / `nvidia-smi utilization.gpu` probe as the Queue-screen GPU meter) and keeps the
  > running max. GPU load is bursty, so a very fast (roughly sub-second) generation
  > can miss the active window and omit it; a normal-length run captures it. Peak
  > **memory** is captured exactly regardless of run length.
- **Phase timing** — the shared stream consumers: `loadMs` / `sampleMs` /
  `decodeMs`, derived from the `Step` → `Decoding` → item-done event boundaries.
- **Effective settings** (image lane): the *resolved* `model`, `quantLabel` /
  `quantBits`, `sampler`, `scheduler`, `steps`, `guidanceScale`, `guidanceMethod`,
  `usePid` / `pidTarget`, dims, seed, loras — the value the run actually used, not
  the sparse `advanced` payload, so a default-settings run is fully populated.

Metrics fire for **every** job type; jobs with no sample/decode phase record
timing + memory only.

### Reading it

- `GET /api/v1/jobs/:id/metrics` — one job's block (`null` for pre-feature jobs).
- `GET /api/v1/metrics?type=&model=&quant=&limit=` — the aggregate feed (metrics
  joined to job identity), newest first.

### In-app Stats screen

**System → Generation Stats** (`apps/web/src/screens/StatsScreen.jsx`): a
filterable/sortable table of every run with its captured metrics + a per-run
detail panel, plus comparison charts (recharts) — median load/sample/decode by a
selectable dimension (quant / model / scheduler / sampler / cfg / …) and a
steps-vs-time scatter by quant. The charts cover generation jobs; the list covers
every job type.
