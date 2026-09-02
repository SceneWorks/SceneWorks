# Cross-backend memory-strategy calibration harness

> **Measuring a lane? Start with [calibration-runbook.md](calibration-runbook.md)** (sc-18103) — the
> parameterized, copy-paste procedure from "I want `<backend>:<provider>` measured" to "PR open with
> the record ingested". This page is the *reference* it draws on: identities, schema semantics,
> the anchor plan format, provider protocol, truth status.

`scripts/memory-calibration-harness.mjs` is the executable, versioned evidence runner for
MLX/Metal and Candle/CUDA. It owns repository resolution, provider execution, validation and atomic
writes. It does not change runtime memory selection policy. Runtime ingestion evaluates the bundled
JSON Schema recursively before semantic checks, so `check`, `capture`, `ingest`, and matrix
generation reject the same closed record structures as the Draft 2020-12 validation suite.

**One anchor per (model, quant tier, backend lane) is the whole measurement obligation** (epic
22505, sc-22514). `config/memory-calibration-plan.json` is an ANCHOR plan: an object keyed
`<modelId>:<tier>:<backend>`, one entry each, validated against
`packages/schemas/memory-anchor-plan.schema.json`. A second measurement of one cell, a geometry or
parameter sweep, a rung grid, a negative case and a batch group are not merely rejected — the format
cannot express them. Everything the grid used to measure is now DERIVED from the anchor by
`crates/sceneworks-core/src/memory_anchor.rs`.

Capturing is a two-step, and the second step is cheap and CPU-only:

1. `capture` — one command, one render, one record, written as a `{records: [...]}` bundle under
   `docs/calibration/<story>/`.
2. `node scripts/extract-memory-anchors.mjs` then `node scripts/anchor-loader-closure.mjs
   --stamp-anchors` — re-derive `config/memory-anchors.json` from the committed corpora (the store is
   a pure function of them) and stamp each anchor's loader-closure currency key.

There is no campaign, no resume, no reuse assessment and no currency ceremony in between.

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

The runner derives the digest for the ONE lane it is capturing, keyed on the anchor's own
`<backend>:<provider>`. A non-authoritative (`fixture`/`candidate`) anchor derives nothing: such a
record can never be `current`, so it needs neither a declarations entry nor an inference crate
layout. Derivation happens **before the provider render**, so a declaration problem surfaces in
seconds rather than after a multi-hour capture. An UNDECLARED lane is still captured (sc-22512); its
record simply carries no currency term and therefore reads `historical`.
`memory-calibration-harness.test.mjs` asserts that every `--anchor` a checked-in capture workflow
names is still declared by the plan.

Full rule, what the digest covers, what it deliberately does not see, and the pin-bump procedure:
[calibration-invalidation-sc-17774.md](calibration-invalidation-sc-17774.md).

## Identities

Logical case IDs contain anchor intent only: evidence scope, backend, route, tier/mode/overlay,
geometry, the anchor composition, fingerprint and fixture. Placeholders never enter identity.
Resolved record IDs additionally contain both probed Git SHAs and dirty states, the generated matrix
source-tree digest, exact hardware, artifact repository/resolved revision, and harness version. The
source-tree digest is the matrix comparison domain; the Git SHA remains independent provenance.
`evidenceScope` is present in both IDs, so fixture results cannot suppress authoritative ones.

The ANCHOR COMPOSITION is not planned — it is fixed per backend lane by the harness
(`ANCHOR_STRATEGY`), mirroring `isDerivable` in `scripts/extract-memory-anchors.mjs`: MLX anchors the
`resident` composition, candle anchors the shallow optimized one (`staged_residency` engaged and
nothing deeper), which is the only composition the candle image law can price. Planning the rung
would let a capture spend hours producing a render the extractor then refuses.

Each capture writes its own bundle. Two captures are two files, and the extractor walks every
retained corpus, so there is nothing to merge and no arrival order to resolve. Writes use a
same-directory temporary file and rename.

## Provider protocol

`capture` executes the provider command, supplied as a JSON argv array. It sends one JSON request on
stdin and expects one JSON response on stdout:

1. `{ "action": "probe" }` → `{ "hardware": ... }`
2. `{ "action": "run", "planned": ..., "repositories": ..., "hardware": ... }` → record fragment

That is the whole protocol for an anchor capture: exactly two invocations, in that order. The
`planned` payload still carries `expectedResult: "passed"`, `negative: false`,
`modelLoadPolicy: "fresh_per_case"` and `modelLoadGroup: null` — the adapters read them — but they
are CONSTANTS the harness attaches, not plan fields, so no plan can ask for anything else.

The runner resolves both repositories using `git rev-parse HEAD` and `git status --porcelain`, reads
the generated matrix source-tree identity, and probes them again after hardware probing and after
the provider render. Any mid-run HEAD, dirty-state, or source-identity change rejects the capture.
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

List the anchor obligation (cheap, CPU-only — it reads the plan and computes identities):

```text
node scripts/memory-calibration-harness.mjs plan \
  --plan config/memory-calibration-plan.json \
  --output .tmp/memory-anchors-planned.json
```

`.tmp/` is gitignored on purpose (see the dirty-state warning above).

Capture ONE anchor:

```text
node scripts/memory-calibration-harness.mjs capture \
  --plan config/memory-calibration-plan.json \
  --anchor qwen_image:bf16:mlx \
  --provider-command '["/absolute/path/to/memory-provider-adapter"]' \
  --sceneworks-repo /absolute/path/to/SceneWorks \
  --inference-repo /absolute/path/to/inference \
  --output /absolute/path/OUTSIDE/the/repo/qwen-image-bf16-mlx-anchor.json
```

`--anchor` names one `<modelId>:<tier>:<backend>` key the plan declares; an unknown or malformed key
fails in milliseconds, before the adapter starts. The command probes hardware once, renders once, and
writes a bundle with EXACTLY one record. To capture two anchors, run it twice — there is no resume,
because there is no accumulating campaign state to resume into.

Physical MLX provenance adds the raw-log pair (both flags or neither), which must point OUTSIDE both
checkouts:

```text
  --raw-log-dir /absolute/path/OUTSIDE/the/repo/raw \
  --source-path-prefix docs/calibration/sc-XXXXX \
```

An `ltx_2_5` anchor additionally requires `--ltx25-snapshot-root`, the canonical public
repository/revision snapshot path. The harness checks the snapshot suffix, the shared enhancer, the
dev refinement adapter and the anchor's own `<transformerVariant>/<tier>` layout, hashes each once,
and re-hashes them around every provider invocation so a mutation during the render is caught. It
then injects the tier and shared-component inventory variables the MLX adapter requires, preserving
the capture-directory and raw-provenance environment.

Validate a captured bundle, and normalize an externally captured session file. Pass `--source-root`
alongside `--input` whenever the capture wrote raw logs, or the physical source-session derivation
cannot be validated:

```text
node scripts/memory-calibration-harness.mjs check \
  --input /absolute/path/OUTSIDE/the/repo/qwen-image-bf16-mlx-anchor.json \
  --source-root /absolute/path/OUTSIDE/the/repo/raw

node scripts/memory-calibration-harness.mjs ingest \
  --input /absolute/path/OUTSIDE/the/repo/qwen-image-bf16-mlx-anchor.json \
  --source-root /absolute/path/OUTSIDE/the/repo/raw \
  --output /absolute/path/OUTSIDE/the/repo/validated-anchor.json
```

Then commit the bundle (and any raw receipts) under `docs/calibration/<story>/` and run the
two-step's second half:

```text
node scripts/extract-memory-anchors.mjs
node scripts/anchor-loader-closure.mjs --stamp-anchors
```

`extract-memory-anchors.mjs` only ANCHORS from a corpus named in `PACKAGED_MEMORY_ANCHOR_SOURCES`
(`crates/sceneworks-core/src/memory_anchor.rs`); an unpackaged corpus contributes envelope evidence
to an analytic-only row instead. Add the new file there when the lane's derivation law is fitted to
it.


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
request, raw provider response, and selected/reference RGB outputs. The three session outputs are a
closed typed set (`request`, `selected_rgb`, `reference_rgb`) with unique paths; RGB filenames include
their logical case, role, dimensions, and content SHA-256, and receipt creation never overwrites
different existing bytes. The adapter independently emits each RGB digest and byte count; capture
fails if the post-provider bytes, the attestation, and the content-addressed filename disagree. The
request must be canonical JSON matching the one evidence record bound to the session. Every newly planned q4/bf16
record carries `sourceProvenance: physical_mlx_v1`. Receipt validation reconstructs the complete
adapter-owned record from the provider response and exact request, compares the source inputs and
claims, requires the session repository/time/hardware/target identity to match, and re-derives the
deterministic session ID, so editing measurements and restamping only the
log digest is rejected. JS and Rust bundle consumers require its claims
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

The plain MLX Krea adapter uses a separate reference-free lane from Krea pose control and requires:

```text
SCENEWORKS_KREA_ROOT=/absolute/path/to/krea-2-turbo-snapshot/<tier>
SCENEWORKS_KREA_REPOSITORY=SceneWorks/krea-2-turbo-mlx
SCENEWORKS_KREA_REVISION=<resolved immutable artifact revision>
```

The selected plan target derives `bf16`, `q4`, or `q8`; the adapter canonicalizes the root and
requires the matching fixed
`/models--SceneWorks--krea-2-turbo-mlx/snapshots/<exact-revision>/<tier>` suffix. It refuses a
reference, control, edit, adapter, overlay, or PiD surface before loading weights and performs no
network fetch. The checked-in plan covers all five implemented rungs across the three shipped tiers
at the 768 and 1024 production geometries, using the production deferred-materialization route and
the exact native 512/64 decode, 67108864 attention, and DiT window-1 domains. The arm validates the
pinned registry contract, runs synchronized phase peaks, parity and negative-mutation checks, and
proves cancellation/error cleanup with warm recovery. SC-18377 deliberately publishes no Krea
evidence record or manifest binding; only a later real Apple-Silicon capture may do that.

The recommended plain MLX SDXL lane requires:

```text
SCENEWORKS_SDXL_ROOT=/absolute/path/to/sdxl-base-snapshot/<tier>
SCENEWORKS_SDXL_REPOSITORY=SceneWorks/sdxl-base-mlx
SCENEWORKS_SDXL_REVISION=<resolved immutable artifact revision>
```

The adapter validates the exact immutable snapshot, pinned registry contract and
`sdxl-mlx-unet-shared-ladder-v3` fingerprint before loading. It accepts only reference-free,
overlay-free `text_to_image`; q4, q8 and bf16 are planned at 768 and 1024 with the production
deferred-materialization shape. Each tier includes only Resident, Staged, and bounded-transformer
residency across cadence `1, 2, 5, 10` (`Dit`): SDXL's bounded decode and bounded attention were measured `Missing`
and are not invented here. The arm records synchronized conditioning/denoise/decode peaks, exact-fit,
unknown-budget and stale-evidence behavior, selected-versus-unselected parity, and a required failing
output mutation. The pinned SDXL provider does not read the calibration error hook, and its plain
untiled VAE decode does not consult the cancellation flag, so the adapter truthfully emits
`runtime_complete`: formal warm/cancel/error lifecycle scenarios remain `not_run`, the mutation is
kept in diagnostics, and neither is promoted into full lifecycle coverage. SC-18379 publishes
apparatus only—no physical capture, evidence record, or manifest
binding—so production remains estimate-backed until a real Apple-Silicon capture is reviewed.

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

sc-22514 retired the per-tier rung ladders those adapters used to be driven through. The plan now
declares ONE anchor per (model, tier, lane) — `qwen_image:bf16:mlx`, `qwen_image:q4:mlx`,
`qwen_image:q8:mlx` and so on — and the ladder positions the ladder used to measure are derived from
the anchor instead. Each record still executes its own deterministic broad-bias mutation and may
claim only its exact returned strategy tuple.

## Matrix promotion

Matrix ingestion binds a record to one cell and independently checks complete status, quality,
executed range, calibration fingerprint, exact runtime strategy parameters, width/height/batch/frame
geometry envelope, artifact revision/variant, and resolved loadability. A gated record remains
`gated` even when its SHAs match. Applicability is owned by the provider ABI fingerprint and
provider-specific compile-closure digest checked by `calibrationBinding`; the captured inference,
SceneWorks, and matrix source revisions remain provenance, not invalidation gates. Exact records
never promote an aggregate geometry or parameter envelope.
