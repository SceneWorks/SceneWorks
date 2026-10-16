SC-26322 uses the existing `windows-candle.yml` workflow at the reviewed story ref. Its `off` mode keeps the ordinary worker jobs; `telemetry` keeps the separate read-only job. `prepare`, `freeze`, and `calibration` use the dedicated execution job. No scored mode exists. Nothing is dispatched by the scripts themselves.

Preparation downloads approximately the selected artifact bytes reported by the pinned Hub metadata, plus binary Python wheels, and builds the native baseline API. It requires that sum plus 48GiB free build/wheel/output headroom. All six snapshots, the Python environment, wheel locks, native source/build, protocol and receipts persist under `E:\sceneworks-evaluation\moss\sc-26322-<campaign>`. Cargo builds use four concurrent jobs. Cargo dependencies/target, native data/config/cache and Hugging Face snapshots are isolated from production caches. Preparation does not load a model or initialize CUDA. Its Rust build compiles CUDA kernels; compiler availability is a separate prerequisite from GPU ownership.

The fixed Python3.12 primary requirements are torch2.9.1+cu128, diffusers0.38.0, transformers5.3.0, accelerate1.10.1, safetensors0.6.2, Pillow12.1.1, psutil7.2.2 and huggingface-hub1.5.0. Preparation resolves binary wheels for the actual Windows Python, freezes every transitive wheel/version/SHA256, installs offline with `--require-hashes`, and runs actual imports including Qwen3_5Model and AutoencoderKLFlux2. These versions are a source-based compatible candidate until that host import and real-weight load succeed; CPU protocol tests do not prove runtime compatibility. Subsequent stages reuse the exact installed environment, full package inventory and wheel lock. The public native inference dependency is fetched through the source's `git-fetch-with-cli` configuration; a fetch/authentication failure is a preparation failure, never GPU or model unavailability evidence.

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

`external_owners_reconciled` is a coordinator attestation after reviewing the actual shared host and its other jobs. An idle GitHub listener does not establish this condition. This runner is restricted to physical GPU1, UUID `GPU-e4b79931-7be6-f216-460a-f5405cfafffe`; physical GPU0 is outside this run. Calibration verifies that exact index/UUID pair, holds an exclusive UUID-named runtime lock, constrains all workers to that UUID (logical CUDA0), and aborts its own process tree if an unrelated GPU PID appears or the deadline expires. The initial device owners must exactly equal the explicitly reviewed desktop baseline. Each baseline entry must have `pid`, `name`, and `created_epoch` (UTC Unix seconds from the captured process creation time); unresolved identities are rejected. An empty list is valid only when no GPU processes exist. The example ceilings are reviewed control inputs, constrained to at most4096MiB and5percent, and checked before allocation and between configurations. Every baseline process identity remains checked during generation; new foreign processes always abort. Existing desktop/system processes are never terminated or changed. Their presence is recorded in `background-device.json` as a timing/device-wide-memory limitation, so this is an exclusive model-compute lease, not a claim that the physical device is empty. A stale lock is never stolen or removed automatically. The run CLI verifies the held lock belongs to a live ancestor process before it can spawn a runtime.

```sh
python3 - <<'PY' | gh workflow run windows-candle.yml --ref "$MOSS_STORY_REF" --json
import json
print(json.dumps({'moss_image_comparison_mode':'calibration','moss_control_json':open('calibration-control.json').read()}))
PY
```

Calibration executes one 512x512 output per configuration, serially, at most six total. It stops on the first failed/timeout cell and never silently retries an existing calibration directory. Actual receipts include per-cell measurements and logs; `calibration-estimate.json` contains the measured 512x512 generation cost times48 per configuration, explicitly excluding unknown resolution scaling. The coordinator must review that measurement before a separate scored authorization; this transport cannot start the 288 scored cells. Native MOSS feasibility remains a later comparison-dependent decision. Failed cells, missing evidence, DMD parity uncertainty and unavailable configurations must remain visible in the evaluation report.

Direct host equivalents, when access is separately available, are `python scripts/evaluation/moss/bootstrap.py prepare|freeze|calibration` with the same reviewed `MOSS_CONTROL_JSON`. Freeze/calibration relaunch themselves using the persistent campaign's locked Python and verified copied harness. Job receipts remain outside git and include run/attempt identity, absolute log path, deadline, resume command and next action.
