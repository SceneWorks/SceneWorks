# Cross-backend image-memory calibration harness

`scripts/image-memory-calibration-harness.mjs` is the executable, versioned evidence runner for
MLX/Metal and Candle/CUDA. It owns plan expansion, repository resolution, provider execution,
validation, resume, atomic writes, and generated-matrix ingestion. It does not change runtime memory
selection policy. Runtime ingestion evaluates the bundled JSON Schema recursively before semantic
checks, so `check`, `run`, `ingest`, and matrix generation reject the same closed record structures
as the Draft 2020-12 validation suite.

## Truth status

The committed v2 evidence bundle is empty. Existing Krea manifest rows do not contain the complete
phase allocator/device/wired metrics, lifecycle fault-injection results, executed parameter sweep, or
an exact SceneWorks Git SHA required by this schema. They are therefore not restated as authoritative
v2 records.

The Qwen identical-latent ladder captured at inference `1deefff` remains useful historical tolerance
input, but lacks required phase memory and lifecycle results and is not the current inference pin.
It is not emitted as a complete record. No MLX or Candle record may become current/Verified until the
authoritative commands below run successfully on clean, exact-final-SHA repositories.

## Identities and resume

Logical plan IDs contain provider intent only: evidence scope, backend, route, tier/mode/overlay,
geometry, exact parameters, fingerprint, fixture, and negative-case status. Placeholders never enter
identity. Resolved record IDs additionally contain both probed Git SHAs and dirty states, the
generated matrix source-tree digest, exact hardware, artifact repository/resolved revision, and
harness version. The source-tree digest is the matrix comparison domain; the Git SHA remains
independent provenance. `evidenceScope` is present in both IDs, so fixture results cannot suppress
authoritative runs.

Only schema-valid `complete` records suppress their executed **passed** positive cases; a failed
case embedded in a positive sweep never consumes its negative plan entry. A measured
`negative_complete` record suppresses only its expected-failure case and cannot become current
matrix evidence. Merge is commutative and
stable-sorted. Different content with the same resolved identity is rejected instead of using
arrival order or timestamp. Writes use a same-directory temporary file and rename.

## Provider protocol

`run` executes the provider command, supplied as a JSON argv array. It sends one JSON request on
stdin and expects one JSON response on stdout:

1. `{ "action": "probe" }` → `{ "hardware": ... }`
2. `{ "action": "run", "planned": ..., "repositories": ..., "hardware": ... }` → record fragment

The runner resolves both repositories using `git rev-parse HEAD` and `git status --porcelain`, reads
the generated matrix source-tree identity, and probes them again after hardware probing and after
every provider case. Any mid-run HEAD, dirty-state, or source-identity change rejects the capture.
The provider probe must return:

- Candle: selected CUDA device ID/name, compute capability, driver, runtime, total bytes, and probe
  description.
- MLX: exact Mac model/chip/OS/Metal device, physical bytes, MLX memory limit, wired ceiling, and
  probe description.

A provider record is complete only if it reports synchronized conditioning/denoise/decode/overall
active, allocator, device, wired, and reclaimable values; overall covers each phase; all scenarios
pass (overlay may be justified structurally N/A); exact-fit values exist and are equal; cancel/error
restore baseline and pass a warm follow-up; identical-latent quality passes; a measured negative
mutation breaches a threshold; executed passed cases exactly derive every claimed range; and the
resolved artifact loads.

Warm-repeat is one loaded generator running A→B→A, not three fresh processes. Cancellation and error
are injected during every phase, followed by synchronization, baseline comparison, and a successful
warm request.

## Commands

Plan without hardware placeholders:

```text
node scripts/image-memory-calibration-harness.mjs plan \
  --config config/image-memory-calibration-plan.json \
  --output .tmp/image-memory-plan.json \
  --resume docs/generated/image-memory-calibration-evidence.json
```

Run an authoritative provider adapter:

```text
node scripts/image-memory-calibration-harness.mjs run \
  --config config/image-memory-calibration-plan.json \
  --provider-command '["/absolute/path/to/image-memory-provider-adapter"]' \
  --sceneworks-repo /absolute/path/to/SceneWorks \
  --inference-repo /absolute/path/to/inference \
  --resume docs/generated/image-memory-calibration-evidence.json \
  --output .tmp/authoritative-image-memory-evidence.json
```

Validate or merge captured output:

```text
node scripts/image-memory-calibration-harness.mjs check \
  --input .tmp/authoritative-image-memory-evidence.json

node scripts/image-memory-calibration-harness.mjs ingest \
  --input .tmp/authoritative-image-memory-evidence.json \
  --resume docs/generated/image-memory-calibration-evidence.json \
  --output .tmp/merged-image-memory-evidence.json
```

The provider adapter can wrap the existing real-weight inference seams:

```text
# Apple hardware, from the exact final inference checkout
cargo test -p mlx-gen-qwen-image --release \
  --test vae_tiling_real_weights -- --ignored --nocapture

# CUDA hardware: use candle-gen::testkit::VramProbe and the Krea phase harness
cargo test -p candle-gen-krea --release --features cuda \
  -- --ignored --nocapture
```

Those raw tests must be parameterized/adapted to the JSON protocol so phase synchronization,
hardware probing, lifecycle injection, and record emission occur in the provider process. A raw
green test log is not ingestible evidence.

The Qwen plan has seven passing overlap-64 cases (`768, 640, 512, 448, 384, 320, 256`) and a separate
expected-failure `256/32` mutation. The Krea plan enumerates candidate tile, overlap, attention-chunk,
and transformer-window combinations explicitly; a record may claim only cases actually returned by
the provider.

## Matrix promotion

Matrix ingestion binds a record to one cell and independently checks complete status, quality,
executed range, calibration fingerprint, exact runtime strategy parameters, width/height/batch/frame
geometry envelope, artifact revision/variant, and resolved loadability. A gated record remains
`gated` even when its SHAs match. A complete authoritative record is `current` only when the clean
inference SHA and SceneWorks matrix source-tree digest exactly match. Exact records never promote an
aggregate geometry or parameter envelope.
