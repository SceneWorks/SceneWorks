# Epic 22988 (YuE2) terminal evidence runbook

The terminal story [sc-23002](https://app.shortcut.com/trefry/story/23002) proves the integrated
YuE2 product on macOS Metal and Windows CUDA with two harnesses. Run each **once**, at the frozen,
pin-matched feature head, after the epic's pin bump. Neither runs in CI on its own, and neither
gates a build.

| Harness | What it proves | Where it runs |
|---|---|---|
| `scripts/yue2-acceptance.mjs` | The epic acceptance matrix (AT1–AT5) through the real app: the source-built API, a worker from the same binary and the registered `yue2` provider on the production device. It covers install, offline use, create/plan/restore, supplied ABC, EN/ZH, covers, score edits, cached decodes, cancellation, kill/resume, artifact integrity, admission, batches, V1/V2 isolation and the licence gate. | Metal: the dev Mac, in the owner's GPU window. CUDA: `windows-candle.yml`, by dispatch. |
| `scripts/yue2-memory-profile.mjs run` | The one measurement campaign (config/yue2-memory-profile-plan.json). It also covers AT1's long request that exercises chunk and context boundaries (`yue2:q4:<backend>:long-context`). | Same hosts, after the acceptance run, reusing its app data dir. |

Each acceptance case writes one record to `<out>/evidence/records/<case>.json`, plus
`summary.json` and `summary.md`. A record holds:
- the exact request and the response ids;
- the run identity and the output SHA-256, duration and RMS;
- the truncation flags;
- per-stage timings, from job progress and from the engine's `result.json`;
- peak memory: `phys_footprint` of the API + worker on Metal, `nvidia-smi` on CUDA;
- device and dtype, taken from the run's own `config.json`;
- the pass/fail reason.

A skipped case always states why. The verdict is `pass` only when no case failed or was skipped.
Audio is CC BY-NC 4.0: it stays in the app data dir on the host for the listening review and is
never copied into the evidence or uploaded.

## Metal (the dev Mac, inside the agreed GPU window)

Build first, outside the window. Compiling is not a render.

```sh
cargo build --release --locked -p sceneworks-rust-api
```

**Step 1: the acceptance matrix.** It runs under the footprint watchdog. The watchdog guards
exactly one process group. The driver keeps the API and the worker in its own group (it never
detaches them), so this guard covers both. The ceilings are the ones
`measure-memory-catalog.mjs` derives for this 128 GiB host. The runtime cap is this step's budget.

```sh
OUT=/private/tmp/sc-23002-yue2-metal; mkdir -p "$OUT"
python3 scripts/memory-calibration-watchdog.py \
  --max-footprint-bytes 94822600832 \
  --host-memory-bytes 137438953472 \
  --min-memory-free-bytes 2147483648 \
  --max-runtime-seconds 21600 \
  --sample-interval 2 --telemetry-timeout 10 --term-grace 1 \
  --event-file "$OUT/watchdog-acceptance.jsonl" -- \
  node scripts/yue2-acceptance.mjs --platform metal \
    --out "$OUT/acceptance" --data-dir "$OUT/state/app-data" \
    --hf-hub /Volumes/Models/huggingface/hub \
    --api-bin target/release/sceneworks-rust-api
```

`--hf-hub` names the hub the app installs into. The install flow re-verifies every blob against
the Hub and re-links the pinned snapshot there, so a shared hub is touched only to restore links to
the same bytes. Omit `--hf-hub` for a fully cold install into `<out>/hf-hub`, which downloads about
11 GB.

`worker-kill-resume` sends SIGKILL to the worker in the middle of a render. On Metal a signal-kill
in the middle of a command buffer can wedge the host's GPU client until a reboot. So on `metal`
the driver records that case as `skipped`, with the reason, unless you pass
`--allow-metal-worker-kill`. Add the flag only on the owner's explicit say-so for this window.
CUDA always runs the case.

**Step 2: the memory profile.** The profile harness wraps each capture in the same watchdog. It
derives the same three ceilings (`--max-footprint-bytes 94822600832`, `--host-memory-bytes
137438953472`, `--min-memory-free-bytes 2147483648`), and `--budget-minutes` is the per-capture
`--max-runtime-seconds`. Do not wrap it in a second watchdog: each capture's guard starts its own
process group, which an outer guard could not see.

```sh
HF_HUB_CACHE=/Volumes/Models/huggingface/hub \
node scripts/yue2-memory-profile.mjs run --backend metal \
  --inference-repo <checkout of SceneWorks/inference at the Cargo pin> \
  --data-dir "$OUT/state/app-data" --out "$OUT/profile" --budget-minutes 120
```

The profile refuses a dirty checkout. Ingest the records with
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
- runs the acceptance matrix with a fresh app data dir under
  `E:\sceneworks-terminal\sc-23002-yue2\<run>` and `HF_HUB_CACHE=E:\huggingface\hub`;
- runs the CUDA profile campaign on that data dir;
- uploads the evidence and receipts (never audio) before it enforces the verdict.

The ordinary `candle-worker` lane stands down for this dispatch, because every `cuda` listener
shares the measured GPU. After the listening review, remove the run's `E:` state directory.

## Reading the verdict

- `pass`: every case passed. `transcription-blocked` counts as a pass when it is `blocked`, which
  is its designed refusal.
- `fail`: a case failed. Its record names the failed assertion.
- `incomplete`: nothing failed, but a case was skipped (for example the Metal worker kill without
  the flag, or a case whose dependency did not pass), or the run was `--dry-run`.

An incomplete platform is not a Done claim for sc-23002.

`--dry-run` starts the API alone, with no worker. It runs only the catalog, isolation, licence-gate
and transcription cases, and never loads weights. Use it to check a host before the GPU window.
