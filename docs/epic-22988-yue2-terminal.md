# Epic 22988 (YuE2) terminal evidence runbook

The terminal story [sc-23002](https://app.shortcut.com/trefry/story/23002) proves the integrated
YuE2 product on macOS Metal and Windows CUDA with two harnesses. Run each **once**, at the frozen,
pin-matched feature head, after the epic's pin bump. Neither runs in CI on its own, and neither
gates a build.

| Harness | What it proves | Where it runs |
|---|---|---|
| `scripts/yue2-acceptance.mjs` | The epic acceptance matrix (AT1–AT5) through the real app: the source-built API, a worker from the same binary and the registered `yue2` provider on the production device. It covers install, offline use, create/plan/restore, supplied ABC, EN/ZH, recording transcription, reviewed melody and full covers, score edits, cached decodes, cancellation, kill/resume, artifact integrity, admission, batches, V1/V2 isolation and the licence gate. | Metal: the dev Mac, in the owner's GPU window. CUDA: `windows-candle.yml`, by dispatch. |
| `scripts/yue2-memory-profile.mjs run` | The one measurement campaign (config/yue2-memory-profile-plan.json). It also covers AT1's long request that exercises chunk and context boundaries (`yue2:q4:<backend>:long-context`). | Same hosts, after the acceptance run, reusing its app data dir and HF home. |

Each acceptance case writes one record to `<out>/evidence/records/<case>.json`, plus
`summary.json` and `summary.md`. A record holds:
- the exact request and the response ids;
- the run identity and the output SHA-256, duration and RMS;
- the truncation flags;
- per-stage timings, from job progress and from the engine's `result.json`;
- peak memory: `phys_footprint` of the API + worker on Metal, `nvidia-smi` on CUDA;
- device, dtype and tier, taken from the run's own engine records (`config.json`, `result.json`,
  or `provenance.json` for a plan-only run),
  never from the worker's echo of the request;
- for transcription, the generated source recording's SHA-256 and RMS, a re-hash of the review
  manifest and every listed artifact, replay and unload evidence, warnings and readiness; covers
  also verify the reviewed score version, source recording, transcriber residency and output asset;
- the absolute ffmpeg binary path and its probed version; the same path is given to the API and
  worker through `SCENEWORKS_FFMPEG` for recording transcription;
- the pass/fail reason.

These experimental runs use CC BY-NC 4.0 model weights and retain their model/license provenance.
The ordinary acceptance/profile campaign retains audio on its host and uploads evidence without
audio. Separately owner-authorized bounded precision runs retain WAVs in internal noncommercial
listening artifacts with 30-day retention; their source identities and digests are recorded below.
This does not determine rights in user lyrics, source recordings or generated outputs.

## Shipped configuration

The driver starts the API and the worker with the desktop's own spawn environment
(`apps/desktop/src/setup.rs`). It removes every `SCENEWORKS_*`, `HF_*`, `TRANSFORMERS_*` and
`CUDA_VISIBLE_DEVICES` variable inherited from the shell first, so the shell you start from cannot
change what is measured. The summary records the effective environment (`serviceEnv`, never a
secret) and every deliberate deviation (`deviations`):
- The source-built harness resolves and probes a host ffmpeg (or takes `--ffmpeg-bin`), then sets
  `SCENEWORKS_FFMPEG` for both services. The packaged desktop uses its bundled ffmpeg. SheetSage2
  transcription uses the shared decoder for WAV, WebM/Opus and MP4/AAC recordings; a missing or
  broken ffmpeg fails before the acceptance cases run.
- The Metal worker gets no `SCENEWORKS_PARENT_PID`: its parent-death exit would drop a render
  mid-command-buffer.
- On CUDA the worker is the per-GPU child the desktop's `auto` supervisor would spawn, started
  directly with the supervisor's child environment.

The worker keeps the shipped 10 s heartbeat. Cancellation rides that heartbeat, so the cancel cases
make their target stage longer than it: 3 000 forced semantic tokens for AR, 400 ODE steps for NAR,
and the smallest decode tile for decode.

## Weight tier and compute precision

The Song Lab has separate **Weight tier** and **Compute precision** controls. `bf16` is the released
weight tier; `q8` and `q4` are locally derived compressed matmul weights. A Q8/Q4 tier does not
mean every operation performs integer Q8/Q4 arithmetic. A fresh create, plan, saved-plan render,
cover, score-version render or cached decode API request must include `computePolicy`:

| `computePolicy` | MoT model stage | VAE decoder stage |
|---|---|---|
| `auto` | BF16 on a supported GPU; FP32 on CPU | FP32 |
| `bf16` | BF16 on a supported GPU | BF16 on that GPU; CPU is refused before weights load |
| `fp32` | FP32 | FP32 |

For example, a fresh request can use
`{"kind":"create","lyrics":"[verse] hello","tier":"q8","computePolicy":"bf16"}`.
Omitting `computePolicy` returns a typed `yue2_missing_field` error; sending it together with the
old `precision` field returns `yue2_invalid_combination`. The transcription-only request is a
separate CPU model and does not accept this control. New experimental FP8 AR submissions require
the released `bf16` weight tier, an explicitly selected supported CUDA GPU, and
`computePolicy:"auto"`; its FP8 AR projections are a separate opt-in from floating compute policy.

Explicit BF16 or FP32 applies to both stages' floating weights and activations; Q8/Q4 matmul
weights retain their separately selected compressed representation. At each Q8/Q4 GGML projection,
the operator casts its input activation to a transient FP32 operand, produces an FP32 matmul result,
then casts that result back to the selected stage dtype; admission reserves this transient pair in
addition to the compressed weights and interlayer residency. Standard FP32 kernel
accumulators/reductions and final audio serialization do not change that stage selection. Auto
deliberately mixes BF16 MoT with FP32 VAE on GPU and discloses both stage dtypes in the engine's
`config.json` (`compute_policy`, `model_dtype`, `vae_dtype`). The worker's `effectiveSettings`
reports `computePolicy` separately from the old `precision` field and includes that engine config
after publication. A cached decode may reuse a verified compatible source latent and runs its new
VAE under the newly requested policy; its new run identity and source provenance remain distinct.

Historical queued/retried jobs keep their recorded Legacy load behavior. Earlier completed
receipts and profile records keep the actual MoT/FP32-VAE policy they measured; neither a saved
BF16 weight tier nor an old unspecified/default precision becomes explicit strict BF16 or Auto.
Old saved Song Lab settings and presets with default precision require a visible compute-policy
choice before another submission. A previously explicit `fp32` setting maps to new FP32 because
both model stages already ran FP32. Existing API clients must send `computePolicy` on fresh
generation/decode requests; they cannot rely on the historical default.

The ten baseline native profile records below remain historical default/Legacy measurements,
not strict-BF16 VAE proof. The bounded explicit-policy receipts below identify their own app,
controller and unchanged M6 runtime revisions. Off-plan cases use
`capture --case-file <outside-repo.json>` through the same guarded native capture helper; they do
not change the checked-in plan or ingest into its corpus. Their hardware checks passed for the
recorded requests; the owner accepted all 15 new renders after listening on 2026-10-05.

For a fresh CUDA or Metal profile host, `yue2-acceptance.mjs --platform <cuda|metal>
--profile-install-only` runs only the existing cold app install flow (BF16 plus both decoders and
locally derived Q8/Q4). It leaves every acceptance case skipped and its verdict incomplete; a
successful install preparation is not terminal acceptance.

## Cold install: a fresh, per-run Hugging Face home

`--hf-home` (default `<out>/hf-home`) must be a **fresh** directory for every run. `install-cold`
**fails** unless the hub holds no cached repository and the app reports nothing installed. A
preseeded hub is recorded as `mode: preseeded` in the summary and never counts as AT4's cold-install
evidence. The cold YuE2 install downloads about 11 GB. `transcription-closure` later installs the
conditional cover closure (SheetSage2 and MERT-v2-FullSong) through the app's own route, downloading
about 2.76 GB more. It does not change YuE2's installed tier state. Keep room for both and the
derived tiers in the per-run state directory.

Never point `--hf-home` at a shared cache such as `/Volumes/Models/huggingface`. Installing into a hub
writes `refs/`, re-creates the snapshot symlinks and deletes any blob whose size does not match the
Hub (`remove_incomplete_download`). The run would also fail, because the hub is preseeded.

## AT2 recording and review path

The harness synthesizes a deterministic 21.5 s mono PCM16 recording of the public-domain Ode to
Joy theme in `scripts/lib/yue2-test-recording.mjs` and uploads it as an ordinary project audio
asset. It first proves that transcription refuses the missing closure, a direct cover from the
recording refuses the unreviewed source, and an unacknowledged closure download queues no jobs.
It then installs both pinned components through the conditional-components route.

`transcribe-cover` runs SheetSage2 on the production transcription device (CPU on both Metal and
CUDA), verifies the review artifact and its replay, and checks that the transcriber unloads. The
record must import both a melody and full score version. The harness downloads one MIDI and one
LAB export and hashes them, edits the melody version for review, then renders a melody cover from
the edit and a full cover from the imported full version. Each cover must retain the transcription
and source-recording provenance, noncommercial policy, and a zero live-transcriber count when YuE2
loads. `transcribe-silence` uploads digital silence and requires a visible `empty_melody` refusal
with no imported score version. The WAVs and exports remain in the app data dir; receipts contain
their hashes and metadata only.

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
FFMPEG_BIN=${FFMPEG_BIN:-$(command -v ffmpeg)}
"$FFMPEG_BIN" -version | head -1
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
  --require-completion-handshake \
  --event-file "$OUT/watchdog-acceptance.jsonl" -- \
  node scripts/yue2-acceptance.mjs --platform metal \
    --out "$OUT/acceptance" --data-dir "$STATE/app-data" --hf-home "$STATE/hf-home" \
    --api-bin target/release/sceneworks-rust-api --ffmpeg-bin "$FFMPEG_BIN"
```

The completion handshake is separate from preallocation child attestation: it does not apply the
2× initial-free-memory policy. After service teardown and durable evidence writes, the driver
sends nonce-bound DONE and remains alive until the guard commits a successful final group/host
sample and acknowledges it. Descendants remain monitored until sentinel cleanup finishes; the
sentinel's actual status is preserved (including status 1 for the owner-skipped case). Missing
or invalid DONE, unacknowledged root telemetry loss, and failed final sampling remain red. This
protocol does not change or reconstruct the verdict of any previously captured run.

**Metal safety.** The driver never sends a signal to a Metal worker that is (or may be) mid-render.
Before any restart or teardown it cancels the active jobs through the API and waits for the worker
to go idle. If the worker stays busy, or its state cannot be read, the driver leaves the API and the
worker running, prints their pids and fails the run. Stop them yourself once the render ends, or
let the watchdog do it.

`worker-kill-resume` sends SIGKILL to the worker in the middle of a render. A signal-kill in the
middle of a command buffer can wedge the host's GPU client until a reboot. The owner explicitly
skips this case on Metal for this run. The driver records it as `skipped`, with the reason; do not
pass `--allow-metal-worker-kill`. Report the skipped case and the resulting incomplete Metal verdict
without counting that one case as accepted. CUDA runs it.

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

For an acceptance-only rerun after a valid profile has already been captured, explicitly add
`-f yue2_acceptance_only=true` to that dispatch. It requires `run_yue2_terminal_cuda=true`;
the default remains the complete acceptance-plus-profile campaign. The rerun skips the profile
and its inference-source checkout, names its artifact `acceptance-only`, and records in the job
summary that it has no new profile verdict. Retain the earlier profile artifact and assess its
five captures separately; for the current CUDA campaign, that is run
[36357082332](https://github.com/SceneWorks/SceneWorks/actions/runs/36357082332).

For the new experimental FP8 AR route, dispatch the same pinned branch with
`-f yue2_fp8_profile_only=true` (and the required `run_yue2_terminal_cuda=true` and exact
`inference_revision`). This mutually excludes `yue2_acceptance_only`. It installs YuE2 into a
fresh per-run state through the app's `install-cold` case, whose acceptance summary remains
**incomplete by design**; it does not rerun the acceptance matrix or the five native CUDA
profiles. It then captures only `yue2:bf16:cuda:experimental-fp8-ar`, requiring a completed,
current record with requested `experimentalFp8`, published engine `config.json` quantization
`fp8`, and admission pricing of at least 2 GiB of retained BF16 AR originals in host RAM.
Receipts upload before the targeted verdict. A passing dispatch proves one integrated native
profile through the product load/admission/generation path; it is not a new HTTP job acceptance
run. Retain the earlier native-profile and app-acceptance artifacts separately.

The `yue2-terminal-cuda` job does the following:
- runs on `[self-hosted, Windows, X64, cuda, real-weights]` in `windows-candle-gpu-real-weights`;
- refuses to share the dispatch with another measurement flag, and refuses an `inference_revision`
  that is not the Cargo pin;
- clears `RUSTC_WRAPPER`;
- builds the release API with `backend-candle` under vcvars64 + `NVCC_CCBIN`;
- runs the acceptance matrix, or the explicit FP8-only install preparation, with a fresh per-run app data dir and HF home under
  `E:\sceneworks-terminal\sc-23002-yue2\<run>` (a cold ~11 GB YuE2 install plus the ~2.76 GB
  conditional cover closure; the shared
  `E:\huggingface\hub` is never used);
- runs the full CUDA profile campaign by default, skips it for `yue2_acceptance_only`, or captures
  only the FP8 case for `yue2_fp8_profile_only`;
- uploads evidence and receipts without audio before enforcing this ordinary terminal workflow's
  verdict; separately authorized bounded precision listening artifacts are described below.

The ordinary `candle-worker` lane stands down for this dispatch, because every `cuda` listener
shares the measured GPU. After the listening review, remove the run's `E:` state directory.

## Reading the verdict

- `pass` (exit 0): every case passed.
- `pass-with-blockers` (exit 2): nothing failed or was skipped, but a path is **blocked**.
  `summary.md` lists each blocker with its reason and unblock condition. A blocked path is surfaced
  for the owner's decision and is not counted as a pass. The CUDA job goes red on it. The current
  AT2 recording path is executable and has no expected blocker.
- `fail` (exit 1): a case failed, or the run stopped early. The record names the failed assertion.
- `incomplete` (exit 1): nothing failed, but a case was skipped (the owner-directed Metal worker
  kill skip, or a case whose dependency did not pass).

Only `pass` is a Done claim for sc-23002 on that platform.

`--dry-run` starts the API alone, with no worker. It runs the catalog, isolation, licence gate and
the transcription-closure refusal checks, including upload of the generated recording. It does
not install the closure, transcribe, or load weights. Use it to check a host before the GPU window;
it exits 0 when none of those checks failed.

## Captured evidence (2026-09-29)

The baseline inference source was M2 `99a60541a73706fb67b4e78f44458774423ffb8e`. The ten
source-owned records in `docs/calibration/yue2/` were **current completed** under that closure:
five Metal and five CUDA cases, including both
long-context captures. Their default/Legacy GPU behavior used BF16 MoT with FP32 VAE; labels such
as `bf16` in a case ID name the weight tier, not a strict all-stage BF16 run. They retain the SceneWorks revision and admission estimate of their own
capture. The new FP8 route changes the source closure, so the checker may mark these records stale
under a later pin; their measured bytes and historical validity are unchanged, and currency is
advisory under `FEATURE_DEVELOPMENT.md`. The four original Metal records contain eleven `UNDER-PRICED` stage observations; the
later admission correction does not rewrite those measurements. The resumed Metal q4-default
record has real external samples in all five stages (load 6, plan 6, semantic 16, acoustic 39,
decode 4), with a 6,864,981,976-byte peak.

| Evidence | Result and qualification |
|---|---|
| [CUDA app acceptance-only run 36367954494](https://github.com/SceneWorks/SceneWorks/actions/runs/36367954494) | 27 passed, zero failed or skipped at SceneWorks `b934d5dc` / M2. It intentionally did not rerun the already valid five-case CUDA profile from [run 36357082332](https://github.com/SceneWorks/SceneWorks/actions/runs/36357082332); that earlier run's app acceptance failed and remains red. |
| [Metal app run 36565987342](https://github.com/SceneWorks/inference/actions/runs/36565987342) | At SceneWorks `44dc4119` / M2, 26 cases passed; the owner-approved `worker-kill-resume` safety skip leaves acceptance **incomplete**, exit 1. Its 754-event watchdog chain includes a durable final sample and completion acknowledgement, then returned the child status after the owned process group emptied. The before-profile preflight stopped the profile; the overall workflow is red. An earlier local Metal run also remains red after watchdog exit 97, despite writing the same 26/1 case counts. |
| [Metal q4-default profile resume 36570798539](https://github.com/SceneWorks/inference/actions/runs/36570798539) | Successful targeted capture in the original app state at SceneWorks `44dc4119` / M2. It rehashed the original acceptance summary and watchdog file before measuring. Its 75-event guard chain verifies; the guarded test exited 0 after process-group cleanup. The original acceptance verdict remains incomplete. |
| [CUDA engine real-weight run 36361386863](https://github.com/SceneWorks/inference/actions/runs/36361386863) | Exact M2 registered render and fidelity checks passed for the declared CUDA configurations. |
| [Metal quality runtime 36559069894](https://github.com/SceneWorks/inference/actions/runs/36559069894) and [artifact-only recovery 36560050786](https://github.com/SceneWorks/inference/actions/runs/36560050786) | Exact M2 f32dev, bf16, q8 and q4 Metal fidelity checks passed against the regenerated CPU reference. The runtime workflow stayed red because its artifact upload path was rejected; the separate upload-only run retained the unchanged receipts without rerunning inference. These metrics use previously characterized bounds and are not a listening verdict. |
| [Registered FP8 engine run 36641569264](https://github.com/SceneWorks/inference/actions/runs/36641569264) | One selected real-weight registered-provider test passed at inference `c1e8f8e023bf4e1fe94a61c4c39e08f881fdd8e6`, producing audio with effective FP8 AR / BF16 compute and mode-bound artifact identity. This short test is not broad fidelity, long-context or app memory evidence. |
| [FP8 app profile run 36661480677](https://github.com/SceneWorks/SceneWorks/actions/runs/36661480677) | Completed at SceneWorks `91a9a582c9531b7b648b76b53d9aee2cb116e630` / inference `c1e8f8e023bf4e1fe94a61c4c39e08f881fdd8e6` on CUDA. The explicit `experimentalFp8` request was admitted, engine output reported `fp8`, and admission retained 2,818,572,288 bytes of BF16 originals in host RAM. All five stages have external samples and measured peaks within estimates. Audio duration was 84.88 seconds, with no ABC or semantic truncation. This native loader/admission/generation profile is not a new HTTP acceptance run or broad FP8 fidelity proof. |
| [Earlier FP8 app attempts 36652399981](https://github.com/SceneWorks/SceneWorks/actions/runs/36652399981) | Attempt 1 was cancelled before capture to avoid competing CUDA work. Attempt 2 refused before load because the standalone profile skipped capability discovery. Neither produced an FP8 measurement; both remain visible. PR2966 repaired selected-GPU capability initialization before the successful capture above. |

The owner [accepted the retained local listening playlist](https://app.shortcut.com/trefry/story/23002/yue2-integrated-metalcuda-readiness-and-v1v2-license-separation#activity-24386):
“They all sound really good.” That review covered the original recording and 22 retained renders;
it does not extend to remote-only FP8 output. Hashes, RMS, stage peaks and fidelity metrics do not
replace a listening review.

The owner corrected the future precision contract above; the earlier FP32 standard/legacy VAEs
and BF16 embedding/latent-position tables plus norms/biases in q8/q4 remain historical facts of
these records, not evidence of strict BF16 or all-integer quantized arithmetic. Listening
acceptance does not resolve them. Weight rehosting remains gated. Preserve the Metal safety skip,
the earlier red runs and separate run identities in final evidence; this table alone does not
claim sc-23002 Done or feature-to-main delivery.

## Bounded explicit precision receipts (2026-10-05)

Production runtime M6 is `25bd55cdb6a56c78b07584a12150c9f5d46be439`. The bounded evidence
combines independently retained CUDA8, Metal BF16 two-case and Metal five-case captures. Their
app and controller revisions differ; each receipt retains its actual source and admission
estimate. This is not one common app head or one seven-case Metal workflow.

| Capture | App / controller (runtime M6 throughout) | Hardware result and scope |
|---|---|---|
| [CUDA8 37334669358](https://github.com/SceneWorks/inference/actions/runs/37334669358) | `f63d173089f81a8cc591d9e905ec952eaa5cda83` / `2c820231926094566d1ed719c085ba7a7a2284a5` | Eight bounded cases passed: BF16 standard/legacy, FP32 standard, Q8/Q4 BF16/FP32 standard and FP8-auto. Case-bound owned stage samples, actual dtypes and WAV identities were independently checked. Known owned release was verified. |
| [Metal3 37345072908](https://github.com/SceneWorks/inference/actions/runs/37345072908) | `f63d173089f81a8cc591d9e905ec952eaa5cda83` / `708462b0db6dde4acace41804100aacb88986b4c` | Workflow remains failed. BF16 standard and legacy completed with stage coverage; these two passes are retained. FP32 standard rendered but failed load/semantic/acoustic pricing and remains excluded from the retained candidate set; decode was covered. Awaited native/guard release is retained, without an independent postcase census claim. |
| [Metal five continuation 37358074867](https://github.com/SceneWorks/inference/actions/runs/37358074867) | `7a8aac63efe1610c33cce02250ed50ee433c4339` / `e5bdedda9a7602689716fff90fb664298be05392` | Five cases passed: FP32 standard; Q8 BF16 standard; Q4 BF16 standard; Q8 FP32 standard; Q4 FP32 standard. Independent audit verified source/closure, raw stage coverage, actual model/VAE/latent dtypes, five WAVs and known owned release through guard chains and after-case known PID absence. Scope is `metal-five-continuation`, with `full_backend_profile:false`. |

Original authenticated artifacts (IDs and original ZIP SHA-256):

- CUDA8 metrics `11358745707`: `40491a1eed3a35f289f85b770e990b721d1e0e7e1da7cbc63d7379339a658429`;
  listening `11358745714`: `82151bb9cfc6758cd206042d98a42fda90fd2e1c33457af77818b82418b3cd66`.
- Metal3 metrics `11360986164`: `44e9ff1d1f61c8cb50b31c2df4bd9e8535e1674b58729ba90f650e943cb8d215`;
  listening `11360409172`: `3c80fb46da20e5b4bb324967cb8e724ef72dc664691c3843c27b32c6ca8bc622`.
- Metal five metrics `11366117888`: `3b4e37b00ad3637191fd06c804f1ca7842be78e4c253caa3c641ad468464169b`
  (122 files); listening `11366187638`: `2b6faa1e02676881f7c1dec52b2770d87c902eb19cd661f470c128ab68ca4228`
  (five WAVs).

App `7a8aac63` adds a 3 GiB empirical allowance to explicit FP32 Metal MoT accounting, based on
the saved undercoverage. The pinned Metal allocator rounds FP32 conversion buffers beyond logical
tensor bytes, but the complete physical-footprint residual was not isolated. The new five-case
samples are covered by their actual `7a8aac63` estimates; the failed `f63d1730` measurements remain
unchanged. CUDA, BF16, decoder formulas and the recorded ample-budget production controls are
unchanged. Sample coverage does not establish a universal memory bound.

**Human listening accepted, 2026-10-05:** the owner reported “All 15 sound good” after reviewing
the source-qualified final playlist. It binds CUDA8,
Metal BF16 two-case and Metal five-case WAVs to their actual records and hashes. The failed original
Metal FP32 WAV is explicitly excluded. The historical 22-render listening acceptance does not
extend to these new outputs; this is a separate verdict for the 15 retained renders.

The original ten profile records, red attempts, Metal watchdog exit 97 and owner-directed
worker-kill safety skip remain disclosed above. Successful historical M2 long-context receipts
retain their original identities; any budget-exceeded attempt remains a distinct historical
failure with its exact command. These short precision captures do not repeat or replace that
campaign. Known captured-process release does not prove absence of unknown or escaped services,
or admit later workloads; Metal native PID birth times and outer watchdog PIDs were not retained.
These bounded receipts do not by themselves close sc-23002 or the epic, prove common-head HTTP
acceptance or establish current long-context evidence.
