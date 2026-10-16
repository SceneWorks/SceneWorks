SC-26322 uses the existing `windows-candle.yml` workflow at the reviewed story ref. Its `off` mode keeps the ordinary worker jobs; `telemetry` keeps the separate read-only job. `prepare`, `freeze`, `calibration`, and `scored` use the dedicated execution job. Scored execution requires explicit coordinator acceptance of the measured calibration. Nothing is dispatched by the scripts themselves.

Preparation downloads approximately the selected artifact bytes reported by the pinned Hub metadata, plus binary Python wheels, and builds the native baseline API. It requires that sum plus 48GiB free build/wheel/output headroom. All six snapshots, the Python environment, wheel locks, native source/build, protocol and receipts persist under `E:\sceneworks-evaluation\moss\sc-26322-<campaign>`. Cargo builds use four concurrent jobs. Cargo dependencies/target, native data/config/cache and Hugging Face snapshots are isolated from production caches. Preparation does not load a model or initialize CUDA. Its Rust build compiles CUDA kernels; compiler availability is a separate prerequisite from GPU ownership.

The fixed Python3.12 primary requirements are torch2.9.1+cu128, diffusers0.38.0, transformers5.3.0, accelerate1.10.1, safetensors0.8.0, Pillow12.1.1, psutil7.2.2 and huggingface-hub1.5.0. The checked-in `dependency-resolution.json` records the official-metadata Windows CPython3.12 resolution: all43 packages, exact wheel URLs and SHA256s,43 metadata requests and zero binary wheel requests. Diffusers0.38.0 requires Safetensors>=0.8.0rc0; the former0.6.2 pin was unsatisfiable. Preparation downloads exactly the43 reviewed wheel hashes without dependency re-resolution, writes the full hash lock, installs offline with `--require-hashes`, and runs actual imports including Qwen3_5Model and AutoencoderKLFlux2. These versions are a source-based compatible candidate until that host import and real-weight load succeed; CPU protocol tests do not prove runtime compatibility. Subsequent stages reuse the exact installed environment, full package inventory and wheel lock. The public native inference dependency is fetched through the source's `git-fetch-with-cli` configuration; a fetch/authentication failure is a preparation failure, never GPU or model unavailability evidence.

MOSS Base fetches its transformer only, SFT supplies its own transformer plus the shared tokenizer/encoder/VAE/scheduler and reviewed pipeline, and DMD fetches the full FP32 transformer, dataspace and split adapter/replacement tensors. `generator.safetensors` is excluded. All three native baselines fetch q4 only. Every downloaded file is checked against immutable revision metadata (LFS SHA256, or Git blob SHA1 for small files), then receives its own SHA256 receipt. No external Python is executed during acquisition. The only custom pipeline accepted by execution has SHA256 `1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31`. Preserve downloaded README/license notices; this internal evaluation is not a license-cleared redistribution conclusion.

Run commands from an authenticated coordinator checkout. Supply a reviewed story ref in `MOSS_STORY_REF`. Each control JSON must have a new UTC deadline no more than six hours ahead. Dispatch inputs travel as JSON/environment, never executable shell fragments. Ordinary run/provision flags must all remain false; the workflow rejects combinations before checkout/setup.

```sh
# Read-only physical telemetry first; inspect its artifact and reconcile every shared-host owner.
gh workflow run windows-candle.yml --ref "$MOSS_STORY_REF" -f moss_image_comparison_mode=telemetry

# Create prepare-control.json locally with ONLY these fields:
# {"campaign":"sc-26322-reference-v1","deadline_utc":"<UTC ISO8601, within six hours>"}
python3 - <<'PY' | gh workflow run windows-candle.yml --ref "$MOSS_STORY_REF" --json
import json
print(json.dumps({'moss_image_comparison_mode':'prepare','moss_control_json':open('prepare-control.json').read()}))
PY
```

After `prepare` completes, inspect `moss-control-<run>-<attempt>` and its `prepared.json`, `requirements.lock`, `wheels.json`, and `job-*.json/.log`. A failure includes an absolute persistent receipt/log path and next action; do not infer that missing credentials, tools or hardware were tested successfully. An interrupted preparation may resume the same campaign and verifies all retained files; a changed source/lock requires an explicitly reviewed new campaign. It never automatically retries.

When only reviewed harness/corpus code changes before generation, prepare a new
campaign without changing the previous campaign's harness. After the coordinator
has confirmed the old preparation job is terminal and completed, use this
explicit reuse control with `prepare`:

```json
{
  "campaign": "sc-26322-reference-v2",
  "deadline_utc": "<fresh UTC deadline within six hours>",
  "reuse": {
    "campaign": "sc-26322-reference-v1",
    "prepared_sha256": "<old prepared.json file SHA256>",
    "job": "38970962483-1",
    "job_receipt_sha256": "<old job-38970962483-1-prepare.json file SHA256>"
  }
}
```

The old job receipt must report completed CPU preparation at the exact unchanged
SceneWorks/inference pins; a failed/running/missing receipt is rejected. Both
campaigns are fixed siblings under the evaluation root; arbitrary source paths
and production caches are rejected. The control does not cancel, monitor or
retry the old workflow. Review its terminal GitHub status before dispatching.
Use the same prepare workflow invocation above, with these new control bytes.

Reuse rehashes the old prepared closure and the same43 reviewed wheel hashes and
URL-derived basenames (including Torch's `+cu128` local version). It seeds only
selected model files, metadata and locked wheels by verified hardlink or copy;
new refs, venv and harness remain independent. It installs the exact closure
offline into the new venv and reruns actual imports. No second model download is
needed. Copy fallback requires enough free bytes; another48GiB remains required
for preparation/output headroom. `reuse.json` records every seed's bytes, hash
and link/copy method, with its own SHA256 bound by new `prepared.json`.

For native reuse, preparation rechecks the old clean exact source and inference
lock, then runs the same `cargo build --release --locked -p sceneworks-rust-api
--features backend-candle --bin sceneworks-rust-api` against the old owned
`CARGO_HOME` and `CARGO_TARGET_DIR`, with `CARGO_NET_OFFLINE=true`. The resulting
exe must have the old prepared SHA256. It is copied independently into the new
campaign, so later Cargo writes cannot mutate the new binary through a hardlink.
If adjacent DLLs are present, reuse stops for inspection or a reviewed fresh
locked build; it never silently copies unrecorded dependency files.

Both build paths capture actual `rustc -Vv`, `cargo -V`, `nvcc --version`,
`CUDA_PATH`, PE import names from `dumpbin /DEPENDENTS`, and hashes/versions of
safely resolved DLL candidates. Unknown/dynamically loaded dependencies remain
explicit. This read-only identity is not a loader/GPU compatibility proof and
Torch's CUDA12.8 build is not the native compiler/runtime version. This source
starts the API for unrecognized arguments, so `--help` is not used as a CPU probe.
The new host/driver/runtime freeze and measured conditions still govern cost
comparisons. The old harness, refs and prepared receipt are never rewritten.

Create `freeze-control.json` with exactly these fields, using the actual host GPU UUID from telemetry and a coordinator-issued lease identity:

```json
{
  "campaign": "sc-26322-reference-v1",
  "deadline_utc": "<fresh UTC deadline>",
  "gpu_uuid": "<actual GPU-... UUID>",
  "ownership": {
    "responsible_chat": "01a1466f-0404-7a91-a33d-dc6cea34e7dc",
    "lease": "<coordinator-issued lease ID>",
    "job": "sc-26322-epic-26321-reference-v1",
    "timeout_seconds": 1800,
    "resume": "Inspect this persistent campaign; no calibration retry without coordinator review",
    "next_action": "Review frozen protocol then authorize at most six calibration outputs"
  }
}
```

```sh
python3 - <<'PY' | gh workflow run windows-candle.yml --ref "$MOSS_STORY_REF" --json
import json
print(json.dumps({'moss_image_comparison_mode':'freeze','moss_control_json':open('freeze-control.json').read()}))
PY
```

Freeze performs no generation. It verifies the preparation, captures actual host/GPU1/driver identity, builds the six configuration manifest and hashes the complete local input closure. Review its `protocol.lock.json` and use that document's **`sha256` field**, the canonical protocol digest, in `calibration-control.json`:

```json
{
  "campaign": "sc-26322-reference-v1",
  "deadline_utc": "<fresh UTC deadline within six hours>",
  "protocol_sha256": "<reviewed protocol sha256 field>",
  "lease": {
    "id": "<same lease ID as frozen ownership>",
    "owner": "01a1466f-0404-7a91-a33d-dc6cea34e7dc",
    "host": "<actual telemetry hostname>",
    "gpu_uuid": "<same actual GPU UUID>",
    "expires_utc": "<UTC deadline covering the entire calibration>",
    "external_owners_reconciled": true,
    "desktop_baseline": [],
    "baseline_max_memory_mib": 2048,
    "baseline_max_utilization_percent": 2
  }
}
```

`external_owners_reconciled` is a coordinator attestation after reviewing the actual shared host and its other jobs. An idle GitHub listener does not establish this condition. This runner is restricted to physical GPU1, UUID `GPU-e4b79931-7be6-f216-460a-f5405cfafffe`; physical GPU0 is outside this run. Calibration verifies that exact index/UUID pair, holds an exclusive UUID-named runtime lock, constrains all workers to that UUID (logical CUDA0), and aborts its own process tree if an unrelated GPU PID appears or the deadline expires. The initial device owners must exactly equal the explicitly reviewed desktop baseline. Each baseline entry must have `pid`, `name`, and `created_epoch` (UTC Unix seconds from the captured process creation time); unresolved identities are rejected. An empty list is valid only when no GPU processes exist. The example ceilings are reviewed control inputs, constrained to at most4096MiB and5percent, and checked before allocation and between configurations. Every baseline process identity remains checked during generation; new foreign processes always abort. Existing desktop/system processes are never terminated or changed. Their presence is recorded in `run/<phase>-background-<job>.json` as a timing/device-wide-memory limitation, so this is an exclusive model-compute lease, not a claim that the physical device is empty. A stale lock is never stolen or removed automatically. The run CLI verifies the held lock belongs to a live ancestor process before it can spawn a runtime.

```sh
python3 - <<'PY' | gh workflow run windows-candle.yml --ref "$MOSS_STORY_REF" --json
import json
print(json.dumps({'moss_image_comparison_mode':'calibration','moss_control_json':open('calibration-control.json').read()}))
PY
```

Calibration attempts the first existing scored fixture at its actual1024x1024 canvas once per configuration, serially, at most six images total. Completed validated attempts resume without rendering again; terminal runtime failures/timeouts remain failed and are never retried. Independent configurations continue after such a guarded runtime failure. A lease, foreign-owner, cache/input, or evidence-validation error still invalidates and stops execution. Orphaned outputs block retry. Actual receipts include per-cell measurements and logs; `calibration-estimate.json` reports that actual canvas and one image/config, estimates only matching-canvas scored rows, and leaves the whole campaign estimate unknown because other shapes/prompt variance are unmeasured. DMD parity uses separate oracle/candidate prediction processes; candidate-only generation timings exclude that oracle work. Hash validation warms the filesystem cache, so cold measurements mean a fresh model/worker from staged cache, not OS-cold disk. The coordinator must review those measurements before scored authorization. Native MOSS feasibility remains a later comparison-dependent decision. Failed cells, missing evidence, DMD parity uncertainty and unavailable configurations must remain visible in the evaluation report.

Direct host equivalents, when access is separately available, are `python scripts/evaluation/moss/bootstrap.py prepare|freeze|calibration|scored` with the same reviewed `MOSS_CONTROL_JSON`. Freeze/calibration/scored relaunch themselves using the persistent campaign's locked Python and verified copied harness. Job receipts remain outside git and include run/attempt identity, absolute log path, deadline, resume command and next action.


After reviewing all six terminal calibration attempts, the estimate and its shape uncertainty, create `scored-control.json` by extending the calibration control with these fields:

```json
{
  "calibration_receipts_sha256": "<printed canonical terminal six-attempt bundle digest>",
  "calibration_estimate_sha256": "<printed calibration-estimate.json file SHA256>",
  "authorization": {
    "id": "<coordinator-issued scored authorization ID>",
    "coordinator": "01a1466f-0404-7a91-a33d-dc6cea34e7dc",
    "accepted_calibration": true,
    "max_scored_outputs": 288,
    "acknowledge_shape_uncertainty": true
  }
}
```

Normally all six calibration attempts succeed and the control above is unchanged.
If an actual owned runtime attempt failed or timed out, inspect and fix any code,
acquisition, or environment defect in place before making an availability decision.
A failed attempt is not automatically model unavailability. Only after reviewing
genuine configuration unavailability may the coordinator add this optional field
at the top level of the same scored control (structure example, **not evidence
that MOSS DMD or any configuration has failed**):

```json
{
  "unavailable_configs": {
    "moss_dmd": {
      "calibration_receipt_sha256": "<actual failed calibration JSON file SHA256>",
      "calibration_log_sha256": "<log SHA256 bound by that failed receipt>",
      "reason": "<reviewed genuine unavailability reason>",
      "reproduce": "<exact reproducible failed request/environment and evidence>",
      "code_and_acquisition_defects_excluded": true
    }
  }
}
```

Use only actual failed configuration IDs. The mapping must cover every failed
calibration and cannot contain a successful one. Missing attempts, unguarded
failures, changed receipts/logs, and guard/resource/cache/input invalidations
cannot be approved. The coordinator identity and accepted six-attempt bundle
remain bound to the original frozen protocol. No generation setting changes,
new freeze, calibration rerenders, or receipt rewriting are involved. The
estimate preserves failed conditions with null cost estimates and records actual
successful outputs separately from the six-attempt/six-image cap.

The immutable `scored-authorization.json` and its private run mirror
`run/availability-authorization.json` must be established before any scored
receipt or image. Existing calibration-only DMD prediction proofs are not scored
images. Every scored receipt in a mixed-availability campaign binds the same
original authorization; later quality-conditioned exclusions or decision changes
are rejected. The approved failed configuration launches no model and emits all
48 unavailable scored records with the explicit reproduction and calibration
receipt/log evidence. The other configurations follow the original settings.
All288 cells and432 comparisons remain present, with unavailable arms counted
against availability; missing quality/cost measurements remain missing. Retain
all six original calibration receipts/logs and the authorization mirror when
copying private evidence for review. Machine review never relabels these as
successful renders or positive evidence.

Retain `campaign`, a fresh bounded `deadline_utc`, the reviewed `protocol_sha256`, and a currently valid GPU1 `lease` with its exact reviewed desktop baseline. The authorization's coordinator must equal that lease owner. No new user approval is needed for the already authorized288 outputs; this control records the coordinator's post-calibration execution decision. A larger budget is rejected.

```sh
python3 - <<'PYRUN' | gh workflow run windows-candle.yml --ref "$MOSS_STORY_REF" --json
import json
print(json.dumps({'moss_image_comparison_mode':'scored','moss_control_json':open('scored-control.json').read()}))
PYRUN
```

Both phases use the same persistent `run/receipts` and `run/outputs`. A bounded job resumes validated immutable successful cells and approved unavailable scored records. Calibration resumes terminal runtime-failure records without retrying them. It refuses unapproved scored failures, changed receipts, invalidated inputs, and output files without valid receipts. It never duplicates an already completed image and never expands the frozen288-cell plan. Renew only the lease/deadline for a subsequent reviewed dispatch; `scored-authorization.json` remains identical. Actual renders are guarded by the same `ownership.ExecutionGuard` in the workflow and direct `run.py` CLI, including GPU1 index/UUID, live coordinator ancestor, expiry, desktop PID/name/creation identity, foreign GPU owners, and exact isolated cache refs/repository/snapshot closure. Cheap input stat fingerprints are monitored during each cell; full selected-config hashes are checked at configuration boundaries and mutations invalidate the attempt's evidence.
