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
  --backend mlx \
  --provider mlx-qwen-vae-decode \
  --provider-command '["/absolute/path/to/image-memory-provider-adapter"]' \
  --sceneworks-repo /absolute/path/to/SceneWorks \
  --inference-repo /absolute/path/to/inference \
  --resume docs/generated/image-memory-calibration-evidence.json \
  --output .tmp/authoritative-image-memory-evidence.json
```

One provider process probes one backend-specific hardware shape, so `--backend mlx|candle` is
required when the config contains both backends. Omitting it from a mixed plan fails before starting
the adapter. `--provider <plan-provider-name>` optionally selects one named provider block; use it
to run the current Krea v1 production point separately from non-promotable v2 candidates.

Validate or merge captured output:

```text
node scripts/image-memory-calibration-harness.mjs check \
  --input .tmp/authoritative-image-memory-evidence.json

node scripts/image-memory-calibration-harness.mjs ingest \
  --input .tmp/authoritative-image-memory-evidence.json \
  --resume docs/generated/image-memory-calibration-evidence.json \
  --output .tmp/merged-image-memory-evidence.json
```

SceneWorks now contains two real provider-protocol executables. They compile against the same exact
inference revision as the worker and never convert partial measurements into complete evidence:

```text
# Apple hardware
cargo build --release -p sceneworks-image-memory-adapter \
  --features mlx --bin image-memory-mlx-adapter

# Windows/Linux CUDA hardware (run in the supported CUDA + host-compiler environment)
cargo build --release -p sceneworks-image-memory-adapter \
  --features candle --bin image-memory-candle-adapter
```

The MLX adapter requires:

```text
SCENEWORKS_QWEN_IMAGE_ROOT=/absolute/path/to/Qwen-Image-snapshot
SCENEWORKS_QWEN_IMAGE_REPOSITORY=Qwen/Qwen-Image
SCENEWORKS_QWEN_IMAGE_REVISION=<resolved immutable artifact revision>
# Only when the host sysctl does not expose a nonzero current wired ceiling:
SCENEWORKS_MLX_WIRED_LIMIT_BYTES=<current host wired ceiling>
```

After this workflow change is present on the repository default branch, the self-hosted macOS ARM64
NAX runner can execute the same adapter through a guarded manual dispatch:

```text
gh workflow run macos-mlx.yml --ref main \
  -f run_image_memory_calibration=true \
  -f inference_revision=<exact-adapter-inference-pin> \
  -f qwen_repository=SceneWorks/qwen-image-mlx \
  -f qwen_revision=<exact-artifact-revision>
```

The runner resolves only the fixed `SceneWorks/qwen-image-mlx` artifact. By default it derives
`$HOME/.cache/huggingface/hub/models--SceneWorks--qwen-image-mlx/snapshots/<exact-revision>/bf16`;
the optional `SCENEWORKS_QWEN_IMAGE_ROOT` repository secret can override that one path when the
runner cache lives elsewhere. The override is canonicalized and must still end in the fixed
repository/exact-revision `/models--SceneWorks--qwen-image-mlx/snapshots/<exact-revision>/bf16`
suffix. The dispatch validates but never prints the resolved path, checks out the exact inference
revision, builds the release adapter, runs the authoritative provider through the harness,
schema-checks the raw bundle, and uploads it as a workflow artifact. The workflow
cannot prove in advance that the private runner has the requested snapshot; an absent exact
directory fails before model load. GitHub only accepts `workflow_dispatch` input schemas from a
workflow that exists on the default branch, so the first calibration run is intentionally
post-merge.

It loads the real pinned `QwenVae`, deterministically encodes a 1024-square gradient fixture, runs
untiled and requested tiled decode on the identical latent, and reports the actual MLX active/cache
measurements plus maximum/mean error. The `256/32` case becomes `negative_complete` only when its
measured error breaches the declared threshold. Positive records remain `gated` because the pinned
Qwen VAE seam does not expose synchronized full-pipeline conditioning/denoise
device/wired/reclaimable phase telemetry or all required lifecycle injections.

The Candle adapter requires:

```text
SCENEWORKS_KREA_ROOT=/absolute/path/to/krea-2-turbo-q4-snapshot
SCENEWORKS_KREA_REPOSITORY=SceneWorks/krea-2-turbo-candle
SCENEWORKS_KREA_REVISION=<resolved immutable artifact revision>
```

It first reads the actual `krea_2_turbo` image-memory contract from the pinned CUDA runtime catalog.
A plan/provider calibration fingerprint or parameter mismatch returns a schema-valid
`gated_before_execution` diagnostic without loading weights. A compatible tuple loads the real q4
provider with sequential residency, opens its image-memory request scope, applies the exact requested
decode/attention/window tuple, renders seed 42 through the public generator API, and samples the
selected physical GPU through trusted-path `nvidia-smi` plus
`candle_gen::testkit::VramProbe`. One loaded generator measures conditioning, denoise, decode, and
overall device peaks; injects cancellation and error at each phase; verifies post-fault device
growth against a fixed tolerance; and follows every fault with a successful warm request. It also
runs Residentâ†’BoundedTransformerResidencyâ†’Resident, requires the resident repeat to be pixel exact,
and reports the bounded output's maximum and mean pixel error without inventing a passing tolerance.
The record remains `gated` until predicted phase curves, an approved bounded-output tolerance,
exact-fit/stale/unknown worker selection, and a measured negative mutation execute.

At the current SceneWorks pin (`d36390da51bf6a1a67f8e00a8c7d7d8a385d2f20`), the authoritative plan
matches the provider's `krea-turbo-cuda-phase-curves-v1` fingerprint and singleton production domain
`512/128/134217728/window=1`. The unsupported `384/...` and `640/.../window=2` v2 experiments live in
a separate `candidate` evidence scope. Candidate records can be measured or gated, but can never
become current matrix evidence or impersonate the production v1 cell.

The inference repository's raw real-weight tests remain useful mechanism checks:

```text
# Apple hardware, from the exact inference checkout
cargo test -p mlx-gen-qwen-image --release \
  --test vae_tiling_real_weights -- --ignored --nocapture

# CUDA hardware
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
