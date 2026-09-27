# Epic 22988 (YuE2) terminal evidence runbook

The terminal story [sc-23002](https://app.shortcut.com/trefry/story/23002) proves the integrated
YuE2 product on macOS Metal and Windows CUDA with two harnesses. Run each **once**, at the frozen,
pin-matched feature head, after the epic's pin bump. Neither runs in CI on its own, and neither
gates a build.

| Harness | What it proves | Where it runs |
|---|---|---|
| `scripts/yue2-acceptance.mjs` | The epic acceptance matrix (AT1–AT5) through the real app: the source-built API, a worker from the same binary and the registered `yue2` provider on the production device. It covers install, offline use, create/plan/restore, supplied ABC, EN/ZH, covers, score edits, cached decodes, cancellation, kill/resume, artifact integrity, admission, batches, V1/V2 isolation and the licence gate. | Metal: the dev Mac, in the owner's GPU window. CUDA: `windows-candle.yml`, by dispatch. |
| `scripts/yue2-memory-profile.mjs run` | The one measurement campaign (config/yue2-memory-profile-plan.json). It also covers AT1's long request that exercises chunk and context boundaries (`yue2:q4:<backend>:long-context`). | Same hosts, after the acceptance run, reusing its app data dir and HF home. |

Each acceptance case writes one record to `<out>/evidence/records/<case>.json`, plus
`summary.json` and `summary.md`. A record holds:
- the exact request and the response ids;
- the run identity and the output SHA-256, duration and RMS;
- the truncation flags;
- per-stage timings, from job progress and from the engine's `result.json`;
- peak memory: `phys_footprint` of the API + worker on Metal, `nvidia-smi` on CUDA;
- device, dtype and tier, taken from the run's own engine records (`config.json`, `result.json`),
  never from the worker's echo of the request;
- the pass/fail reason.

Audio is CC BY-NC 4.0: it stays in the app data dir on the host for the listening review and is
never copied into the evidence or uploaded.

## Shipped configuration

The driver starts the API and the worker with the desktop's own spawn environment
(`apps/desktop/src/setup.rs`). It removes every `SCENEWORKS_*`, `HF_*`, `TRANSFORMERS_*` and
`CUDA_VISIBLE_DEVICES` variable inherited from the shell first, so the shell you start from cannot
change what is measured. The summary records the effective environment (`serviceEnv`, never a
secret) and every deliberate deviation (`deviations`):
- The desktop's bundled `SCENEWORKS_FFMPEG` is not set; YuE2 jobs never use ffmpeg.
- The Metal worker gets no `SCENEWORKS_PARENT_PID`: its parent-death exit would drop a render
  mid-command-buffer.
- On CUDA the worker is the per-GPU child the desktop's `auto` supervisor would spawn, started
  directly with the supervisor's child environment.

The worker keeps the shipped 10 s heartbeat. Cancellation rides that heartbeat, so the cancel cases
make their target stage longer than it: 3 000 forced semantic tokens for AR, 400 ODE steps for NAR,
and the smallest decode tile for decode.

## Cold install: a fresh, per-run Hugging Face home

`--hf-home` (default `<out>/hf-home`) must be a **fresh** directory for every run. `install-cold`
**fails** unless the hub holds no cached repository and the app reports nothing installed. A
preseeded hub is recorded as `mode: preseeded` in the summary and never counts as AT4's cold-install
evidence. The cold install downloads about 11 GB.

Never point `--hf-home` at a shared cache such as `/Volumes/Models/huggingface`. Installing into a hub
writes `refs/`, re-creates the snapshot symlinks and deletes any blob whose size does not match the
Hub (`remove_incomplete_download`). The run would also fail, because the hub is preseeded.

## Metal (the dev Mac, inside the agreed GPU window)

Build first, outside the window. Compiling is not a render.

```sh
cargo build --release --locked -p sceneworks-rust-api
```

Every run gets its own state directory on the models volume (app data dir and HF home, ~40 GB with
the derived tiers). The evidence goes to `/private/tmp`:

```sh
RUN=$(date -u +%Y%m%dT%H%M%SZ)
STATE=/Volumes/Models/sc-23002/$RUN; OUT=/private/tmp/sc-23002-yue2-metal-$RUN
mkdir -p "$STATE" "$OUT"
```

**Step 1: the acceptance matrix.** It runs under the footprint watchdog. The watchdog guards
exactly one process group. The driver keeps the API and the worker in its own group (it never
detaches them), so this guard covers both. The ceilings are the ones
`measure-memory-catalog.mjs` derives for this 128 GiB host. The runtime cap is this step's budget.

```sh
python3 scripts/memory-calibration-watchdog.py \
  --max-footprint-bytes 94822600832 \
  --host-memory-bytes 137438953472 \
  --min-memory-free-bytes 2147483648 \
  --max-runtime-seconds 21600 \
  --sample-interval 2 --telemetry-timeout 10 --term-grace 1 \
  --event-file "$OUT/watchdog-acceptance.jsonl" -- \
  node scripts/yue2-acceptance.mjs --platform metal \
    --out "$OUT/acceptance" --data-dir "$STATE/app-data" --hf-home "$STATE/hf-home" \
    --api-bin target/release/sceneworks-rust-api
```

**Metal safety.** The driver never sends a signal to a Metal worker that is (or may be) mid-render.
Before any restart or teardown it cancels the active jobs through the API and waits for the worker
to go idle. If the worker stays busy, or its state cannot be read, the driver leaves the API and the
worker running, prints their pids and fails the run. Stop them yourself once the render ends, or
let the watchdog do it.

`worker-kill-resume` sends SIGKILL to the worker in the middle of a render. A signal-kill in the
middle of a command buffer can wedge the host's GPU client until a reboot. So on `metal` the driver
records that case as `skipped`, with the reason, unless you pass `--allow-metal-worker-kill`. Add
the flag only on the owner's explicit say-so for this window. CUDA always runs the case.

**Step 2: the memory profile.** The profile harness wraps each capture in the same watchdog. It
derives the same three ceilings (`--max-footprint-bytes 94822600832`, `--host-memory-bytes
137438953472`, `--min-memory-free-bytes 2147483648`), and `--budget-minutes` is the per-capture
`--max-runtime-seconds`. Do not wrap it in a second watchdog: each capture's guard starts its own
process group, which an outer guard could not see. It resolves the weights the acceptance run
installed, so point it at the same app data dir and HF home, with no other hub variable set:

```sh
env -u HF_HUB_CACHE -u HUGGINGFACE_HUB_CACHE HF_HOME="$STATE/hf-home" \
node scripts/yue2-memory-profile.mjs run --backend metal \
  --inference-repo <checkout of SceneWorks/inference at the Cargo pin> \
  --data-dir "$STATE/app-data" --out "$OUT/profile" --budget-minutes 120
```

The profile refuses a dirty checkout. Current (v2) records must state the run's truncation, stage
times and identities; only legacy v1 records may lack them. Ingest the records with
`node scripts/yue2-memory-profile.mjs ingest "$OUT"/profile/*/record.json`.

## CUDA (Windows runner, by dispatch)

```sh
gh workflow run windows-candle.yml --ref feature/sc-22988-yue2 \
  -f run_yue2_terminal_cuda=true -f inference_revision=<the Cargo.toml inference pin>
```

The `yue2-terminal-cuda` job does the following:
- runs on `[self-hosted, Windows, X64, cuda, real-weights]` in `windows-candle-gpu-real-weights`;
- refuses to share the dispatch with another measurement flag, and refuses an `inference_revision`
  that is not the Cargo pin;
- clears `RUSTC_WRAPPER`;
- builds the release API with `backend-candle` under vcvars64 + `NVCC_CCBIN`;
- runs the acceptance matrix with a fresh per-run app data dir and HF home under
  `E:\sceneworks-terminal\sc-23002-yue2\<run>` (a cold ~11 GB download; the shared
  `E:\huggingface\hub` is never used);
- runs the CUDA profile campaign on that data dir and HF home;
- uploads the evidence and receipts (never audio) before it enforces the verdict.

The ordinary `candle-worker` lane stands down for this dispatch, because every `cuda` listener
shares the measured GPU. After the listening review, remove the run's `E:` state directory.

## Reading the verdict

- `pass` (exit 0): every case passed.
- `pass-with-blockers` (exit 2): nothing failed or was skipped, but a path is **blocked**. Today
  that is AT2's recording → transcription path (SheetSage2 + MERT-v2-FullSong): `summary.md` lists
  each blocker with its reason and unblock condition. A blocked path is **surfaced for the owner's
  decision, not counted as a pass**. The CUDA job goes red on it.
- `fail` (exit 1): a case failed, or the run stopped early. The record names the failed assertion.
- `incomplete` (exit 1): nothing failed, but a case was skipped (for example the Metal worker kill
  without the flag, or a case whose dependency did not pass).

Only `pass` is a Done claim for sc-23002 on that platform.

`--dry-run` starts the API alone, with no worker. It runs only the catalog, isolation, licence-gate
and transcription cases, and never loads weights. Use it to check a host before the GPU window; it
exits 0 when none of those failed.
