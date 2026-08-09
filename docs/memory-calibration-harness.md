# Cross-backend memory-strategy calibration harness

> **Measuring a lane? Start with [calibration-runbook.md](calibration-runbook.md)** (sc-18103) — the
> parameterized, copy-paste procedure from "I want `<backend>:<provider>` measured" to "PR open with
> the record ingested". This page is the *reference* it draws on: identities, schema semantics,
> resume, provider protocol, truth status.

`scripts/memory-calibration-harness.mjs` is the executable, versioned evidence runner for
MLX/Metal and Candle/CUDA. It owns plan expansion, repository resolution, provider execution,
validation, resume, atomic writes, and generated-matrix ingestion. It does not change runtime memory
selection policy. Runtime ingestion evaluates the bundled JSON Schema recursively before semantic
checks, so `check`, `run`, `ingest`, and matrix generation reject the same closed record structures
as the Draft 2020-12 validation suite.

## Truth status

The committed v3 evidence bundle is empty. Existing Krea manifest rows do not contain the complete
phase allocator/device/wired metrics, lifecycle fault-injection results, executed parameter sweep, or
an exact SceneWorks Git SHA required by this schema. They are therefore not restated as authoritative
v3 records. Schema v3 / harness v4 makes `strategy.engagedRungs` part of every record and both logical
and resolved identity. Pre-composition v2 records are deliberately invalid rather than treated as
composition-agnostic evidence.
Each generated matrix cell also carries the current expected `engagedRungs`. `calibrationBinding`
requires an exact ordered match with the adapter-attested record, so a historical row measured under
a different prerequisite graph remains visible as provenance but cannot verify the current cell.

The Qwen identical-latent ladder captured at inference `1deefff` remains useful historical tolerance
input, but lacks required phase memory and lifecycle results. It is not emitted as a complete record.
No MLX or Candle record may become current/Verified until the authoritative commands below run
successfully on clean, exact-final-SHA repositories.

### What makes a record `current` (sc-17774)

Not the inference pin. A complete record is `current` when the **compile-closure digest of the
provider it measured** still matches the live one for that `<backend>:<provider>` lane in
`config/inference-provider-closures.json`:

```js
record.repositories.inference.closureDigest === revisions.inferenceClosureDigests[lane]
```

`repositories.inference.revision` is capture provenance and is never compared — comparing it is what
made every inference commit, including a commit to an unrelated model, demote every calibrated
provider at once. The runner stamps the digest at capture time from the inference checkout it is
already given; `evidenceSemantics` fails loudly rather than falling back if a record lacks one, since
a silent fallback would restore the old policy invisibly.

The runner derives **one digest per authoritative lane it is about to capture**, keyed on each
selected plan entry's own `<backend>:<target.provider>` — never on `--provider`, which names a plan
entry (`candle-krea-q4-fresh-reference-resident`), not a lane. A run that selects several lanes
(`--backend candle` with no `--fixture`) stamps each record with its own lane's digest. A selection
that is entirely `fixture`/`candidate` scope derives nothing: those records can never be `current`,
so they need neither a declarations entry nor an inference crate layout. Derivation happens **before
the first capture invocation**, so an undeclared lane fails in seconds rather than after a multi-hour
GPU sweep. Every authoritative lane in `config/memory-calibration-plan.json` must therefore have an
entry in `config/inference-provider-closures.json`; `memory-calibration-harness.test.mjs` asserts
that, and that every `--fixture` a checked-in capture workflow names still exists in the plan.

Full rule, what the digest covers, what it deliberately does not see, and the pin-bump procedure:
[calibration-invalidation-sc-17774.md](calibration-invalidation-sc-17774.md).

## Identities and resume

Logical plan IDs contain provider intent only: evidence scope, backend, route, tier/mode/overlay,
geometry, selected rung, the provider-declared ordered set of actually engaged rungs, exact
parameters, fingerprint, fixture, and negative-case status. Placeholders never enter identity.
Resolved record IDs additionally contain both probed Git SHAs and dirty states, the
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

Provider resume also tracks attempted work, independently of evidence completion. A gated or
candidate record suppresses a repeated operational attempt only when its logical case, harness
version, repository receipts (including source-tree and dirty state), and hardware probe exactly
match the new run. Stale or foreign-provenance records remain scheduled. This lets a checkpointed
GPU sweep continue after failure without treating its non-authoritative records as complete or
eligible for runtime promotion.

## Provider protocol

`run` executes the provider command, supplied as a JSON argv array. It sends one JSON request on
stdin and expects one JSON response on stdout:

1. `{ "action": "probe" }` → `{ "hardware": ... }`
2. `{ "action": "run", "planned": ..., "repositories": ..., "hardware": ... }` → record fragment
3. `{ "action": "run_batch", "planned": [...], ... }` → `{ "modelLoads": 1, "fragments": [...] }`
4. `{ "action": "assess_batch", "planned": [...] }` → an explicit reuse-eligibility verdict

`modelLoadPolicy: "batch_rungs"` plus a shared `modelLoadGroup` schedules pending cases for one
target, backend, and fixture in canonical rung order. When multiple parameter points are pending for
one rung, the runner selects one for the batch; a returned complete sweep can then retire the others.
The adapter must attest exactly one model load, and every fragment preserves its existing logical and
resolved identity. After every response the runner recomputes completed logical cases before choosing
another provider invocation. If candidate or gated evidence leaves only part of the original rung
cohort pending, the runner executes those remaining parameter points individually instead of sending
an invalid partial `run_batch`. The CLI atomically checkpoints the accumulated schema-valid bundle
after every successful provider response, so a later failure does not discard earlier captures.

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
interrupt the active selected rung, restore baseline, and pass a warm follow-up; quality compares
either identical latents or final outputs from identical inputs and passes; a measured negative
mutation breaches a threshold; executed passed cases exactly derive every claimed range; and the
resolved artifact loads.

Warm-repeat is one loaded generator running A→B→A, not three fresh processes. Cancellation and error
are injected at the active physical boundary for the rung being certified, followed by synchronization,
baseline comparison, and a successful warm request.

## Commands

`.tmp/` is gitignored on purpose. The runner stamps every record with the repositories' `dirty`
state from `git status --porcelain`, which counts UNTRACKED paths, and `complete` evidence from a
dirty repository is rejected. Writing captures anywhere else inside the repo therefore poisons the
NEXT run — the first one looks fine because git does not report an empty directory and the output is
only written at the end. Commit before capturing, and change nothing in either repository while a
capture is running: the runner re-probes after every provider case and aborts if HEAD or dirty state
moved.

The `loadShape` on a record is the ADAPTER's attestation of what its run actually loaded under. The
plan declares the shape a rung is expected to select and the runner cross-checks the two, but the
planned value is never written onto a fragment: a receipt may only testify to its own run (sc-16482).
An adapter that omits the field, or attests a shape the plan did not declare, fails the capture.

Plan without hardware placeholders:

```text
node scripts/memory-calibration-harness.mjs plan \
  --config config/memory-calibration-plan.json \
  --output .tmp/memory-plan.json \
  --resume docs/generated/memory-calibration-evidence.json
```

Run an authoritative provider adapter:

```text
node scripts/memory-calibration-harness.mjs run \
  --config config/memory-calibration-plan.json \
  --backend mlx \
  --fixture qwen-image-bf16-seed15511-step2 \
  --fresh-per-case \
  --provider-command '["/absolute/path/to/memory-provider-adapter"]' \
  --sceneworks-repo /absolute/path/to/SceneWorks \
  --inference-repo /absolute/path/to/inference \
  --resume docs/generated/memory-calibration-evidence.json \
  --output .tmp/authoritative-memory-evidence.json
```

One provider process probes one backend-specific hardware shape, so `--backend mlx|candle` is
required when the config contains both backends. Omitting it from a mixed plan fails before starting
the adapter. `--provider <plan-provider-name>` optionally selects one named provider block; use it
to run the current Krea v1 production point separately from non-promotable v2 candidates.
`--fixture <fixture-name>` selects every provider block sharing that fixture, which is the intended
way to execute a multi-rung reference ladder as one reproducible capture.
`--fresh-per-case` overrides scheduling for an oracle capture; `--batch-rungs` forces one target's
rungs into an experimental batch. Compare the two bundles with the committed larger-of tolerance
(256 MiB absolute or 5% relative for every phase/metric):

```text
node scripts/memory-calibration-harness.mjs compare-reuse \
  --fresh .tmp/fresh.json --reused .tmp/reused.json \
  --output .tmp/reuse-comparison.json
```

When a backend cannot truthfully execute a batch, record that before measurement:

```text
node scripts/memory-calibration-harness.mjs assess-reuse \
  --config config/memory-calibration-plan.json \
  --backend mlx --fixture <five-rung-fixture> \
  --provider-command '["/absolute/path/to/memory-provider-adapter"]' \
  --output .tmp/reuse-assessment.json
```

Validate or merge captured output:

```text
node scripts/memory-calibration-harness.mjs check \
  --input .tmp/authoritative-memory-evidence.json

node scripts/memory-calibration-harness.mjs ingest \
  --input .tmp/authoritative-memory-evidence.json \
  --resume docs/generated/memory-calibration-evidence.json \
  --output .tmp/merged-memory-evidence.json
```

SceneWorks now contains two real provider-protocol executables. They compile against the same exact
inference revision as the worker and never convert partial measurements into complete evidence:

```text
# Apple hardware
cargo build --release -p sceneworks-memory-adapter \
  --features mlx --bin memory-mlx-adapter

# Windows/Linux CUDA hardware (run in the supported CUDA + host-compiler environment)
cargo build --release -p sceneworks-memory-adapter \
  --features candle --bin memory-candle-adapter
```

The MLX adapter requires:

```text
SCENEWORKS_QWEN_IMAGE_ROOT=/absolute/path/to/Qwen-Image-tier-snapshot
SCENEWORKS_QWEN_IMAGE_REPOSITORY=SceneWorks/qwen-image-mlx
SCENEWORKS_QWEN_IMAGE_REVISION=<resolved immutable artifact revision>
SCENEWORKS_MEMORY_MODEL_BYTES=<bytes from hash-artifact-inventory.mjs>
SCENEWORKS_MEMORY_MODEL_INVENTORY_SHA256=<inventory SHA-256 from that script>
SCENEWORKS_MEMORY_CAPTURE_DIR=/absolute/path/outside/the/checkout/raw-receipts
SCENEWORKS_MEMORY_SOURCE_PATH_PREFIX=docs/calibration/<story-or-campaign>
# Optional explicit byte override. Otherwise the adapter uses the configured
# iogpu/kernel policy, then derives recommendedMaxWorkingSetSize from MLX's
# untouched default memory limit using the worker's real-hardware-validated rule.
SCENEWORKS_MLX_WIRED_LIMIT_BYTES=<current host wired ceiling>
```

The adapter derives `bf16`, `q4`, or `q8` from the selected plan target, canonicalizes the root, and
requires the matching fixed
`/models--SceneWorks--qwen-image-mlx/snapshots/<exact-revision>/<tier>` suffix before loading. The
fixture name must carry that same tier and the exact numeric seed, preventing a packed-tier record
from being emitted against another tier's weights.
It resolves the wired ceiling from the explicit override first, then the host's configured
`iogpu.wired_limit_mb`, then the legacy `kern.memorystatus_wired_mem_limit` byte sysctl, and finally
the untouched MLX default memory limit. MLX documents that default as 1.5 times Metal's recommended
working-set size, so the final fallback uses `get_memory_limit() / 3 * 2`, matching the production
worker's current-host derivation and rounding down to remain at or below the ceiling. The selected
source is recorded in `hardware.probe` and must resolve to a nonzero value. A present explicit
override must parse as a positive byte count; malformed or zero values fail closed.

After this workflow change is present on the repository default branch, the self-hosted macOS ARM64
NAX runner can execute the same adapter through a guarded manual dispatch:

```text
gh workflow run macos-mlx.yml --ref main \
  -f run_memory_calibration=true \
  -f provision_qwen_snapshot=false \
  -f qwen_tier=bf16 \
  -f inference_revision=<exact-adapter-inference-pin> \
  -f qwen_repository=SceneWorks/qwen-image-mlx \
  -f qwen_revision=<exact-artifact-revision> \
  -f qwen_source_path_prefix=docs/calibration/<story-or-campaign>
```

The runner resolves only the fixed `SceneWorks/qwen-image-mlx` artifact. Without an override it
checks exactly two canonical locations, in order:
`$HOME/.cache/huggingface/hub/models--SceneWorks--qwen-image-mlx/snapshots/<exact-revision>/<tier>`,
then
`$HOME/Library/Application Support/SceneWorks/data/cache/huggingface/hub/models--SceneWorks--qwen-image-mlx/snapshots/<exact-revision>/<tier>`.
It does not scan other directories. The optional `SCENEWORKS_QWEN_IMAGE_ROOT` repository secret can
override those locations when the runner cache lives elsewhere. The override is canonicalized and
used only when it ends in the selected tier's fixed
repository/exact-revision `/models--SceneWorks--qwen-image-mlx/snapshots/<exact-revision>/<tier>`
suffix; a stale override for another tier is ignored in favor of the canonical locations. The
dispatch validates but never prints the resolved path, checks out the exact inference
revision using `SCENEWORKS_INFERENCE_READ_TOKEN` when configured and the workflow's scoped token
otherwise, fingerprints the exact tier inventory, builds the release adapter, runs the
authoritative provider through the harness, hashes every artifact byte, schema-checks the bundle, and
uploads the evidence JSON together with a repository-relative receipt tree containing the exact
request, raw provider response, and selected/reference RGB outputs. Every newly planned q4/bf16
record carries `sourceProvenance: physical_mlx_v1`; JS and Rust bundle consumers require its claims
to share one `physical_mlx` session bound to the record's exact inventory. Validate the downloaded
artifact with `check --source-root <unpacked-raw-root>`, copy its `docs/calibration/...` tree into the
checkout, and then ingest. Missing or altered receipt files fail both check and ingest. The workflow
cannot prove in advance that the private runner has the requested snapshot; an absent exact
directory fails before model load. GitHub only accepts `workflow_dispatch` input schemas from a
workflow that exists on the default branch, so the first calibration run is intentionally
post-merge.

When both canonical roots are absent, explicitly set `provision_qwen_snapshot=true` on the same
calibration dispatch. Provisioning is rejected unless `run_memory_calibration=true`. That
opt-in lane validates the self-hosted runner's Python 3.12 prerequisite, creates an isolated
job-local virtual environment, and pins `huggingface_hub==0.36.0` to resumably and idempotently
download only `<tier>/**` from the fixed public `SceneWorks/qwen-image-mlx` repository at the exact
`qwen_revision`. It uses no token and writes to the canonical SceneWorks application-data Hugging
Face cache unless the fixed repository already exists in the standard Hugging Face cache; in that
case it resumes there and reuses existing blobs. The pinned client always verifies and resumes the
requested `<tier>/**` files rather than treating a progressively created snapshot directory as
complete. Progress and paths are not logged. Because the largest snapshot is approximately 57 GiB, only a
provisioning dispatch receives the extended four-hour job timeout; ordinary MLX CI and
non-provisioning calibration retain the 45-minute ceiling. A provisioning or calibration failure
cannot upload a schema-checked evidence artifact.

It loads the real pinned `QwenVae`, deterministically encodes a 1024-square gradient fixture, runs
untiled and requested tiled decode on the identical latent, and reports the actual MLX active/cache
measurements plus maximum/mean error. Production cases never alter either comparison output. The
expected-failure `256/32` case first requires that same unmodified identical-latent comparison to
pass, retains those real measurements in `quality`, then applies the planned
`comparisonOutputBias=0.05` away from the baseline at every comparison element. Only the resulting
measured perturbed metrics enter `negativeMutation`, and the case becomes `negative_complete` only
when they breach the unchanged production thresholds. Positive records remain `gated` because the pinned
Qwen VAE seam does not expose synchronized full-pipeline conditioning/denoise
device/wired/reclaimable phase telemetry or all required lifecycle injections.

The Candle adapter requires:

```text
SCENEWORKS_KREA_ROOT=/absolute/path/to/krea-2-turbo-q4-snapshot
SCENEWORKS_KREA_REPOSITORY=SceneWorks/krea-2-turbo-mlx
SCENEWORKS_KREA_REVISION=<resolved immutable artifact revision>
```

The adapter canonicalizes the root and requires the fixed
`/models--SceneWorks--krea-2-turbo-mlx/snapshots/<exact-revision>/q4` suffix before loading.

It first reads the actual `krea_2_turbo` memory-strategy contract from the pinned CUDA runtime catalog.
A plan/provider calibration fingerprint or parameter mismatch returns a schema-valid
`gated_before_execution` diagnostic without loading weights. A compatible tuple loads the real q4
provider with sequential residency, opens its memory-strategy request scope, applies the exact requested
decode/attention/window tuple, renders seed 42 through the public generator API, and samples the
selected physical GPU through trusted-path `nvidia-smi` plus
`candle_gen::testkit::VramProbe`. One loaded generator measures conditioning, denoise, decode, and
overall device peaks; injects cancellation and error at each phase; verifies post-fault device
growth against a fixed tolerance; and follows every fault with a successful warm request. It also
runs Residentâ†’BoundedTransformerResidencyâ†’Resident, requires the resident repeat to be pixel exact,
and reports the bounded output's maximum and mean pixel error without inventing a passing tolerance.
The record remains `gated` until predicted phase curves, an approved bounded-output tolerance,
exact-fit/stale/unknown worker selection, and a measured negative mutation execute.

The authoritative plan and manifest name the provider's
`krea-turbo-cuda-phase-curves-v1` fingerprint and singleton production domain
`512/128/134217728/window=1`. Each historical Krea evidence record now also names the exact measured
composition for each manifest rung. The three-stage and streamed-block rows still match the current
provider contract and remain usable. The q4/q8/bf16 bounded-decode and bounded-attention rows were
captured with staged residency active, while those selections no longer engage staging, so the
selector returns the dedicated `CompositionMismatch` verdict for those six rows until SC-15913
recaptures them. The unsupported `384/...` and `640/.../window=2` experiments remain in a separate
`candidate` evidence scope. Candidate records
can be measured or gated, but can never become current matrix evidence or impersonate the production
v1 cell.

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

The Qwen plan has fifteen authoritative records. BF16 retains the complete eleven-record ladder:
resident, staged residency, seven overlap-64 decode edges (`768, 640, 512, 448, 384, 320, 256`),
bounded attention, and window-1 transformer residency. Q4 and Q8 each add the exact whole-request
bounded-attention versus window-1 transformer-residency pair required by SC-16353. BF16 uses seed
15511; the two packed tiers use seed 16353 so their rung-3/rung-4 comparison shares the existing
fully-resident and window-domain attribution fixture. Every record executes its own deterministic
broad-bias mutation and may claim only its exact returned strategy tuple. The Krea plan likewise
enumerates candidate tile, overlap, attention-chunk, and transformer-window combinations explicitly.

## Matrix promotion

Matrix ingestion binds a record to one cell and independently checks complete status, quality,
executed range, calibration fingerprint, exact runtime strategy parameters, width/height/batch/frame
geometry envelope, artifact revision/variant, and resolved loadability. A gated record remains
`gated` even when its SHAs match. A complete authoritative record is `current` only when its clean
inference SHA matches the exact workspace pin; SceneWorks invalidation is owned by the provider ABI
fingerprint checked by `calibrationBinding`. The captured SceneWorks revision and matrix source-tree
digest remain provenance, not a second invalidation gate. Exact records never promote an aggregate
geometry or parameter envelope.
