# Epic 20738 terminal CUDA evidence profile

SC-20945 adds one deliberately opt-in terminal evidence profile to
`.github/workflows/windows-candle.yml`. It is source preparation, not permission to run a campaign.
The profile stays off on pushes, pull requests, and ordinary manual runs. Run it once only after the
epic's SceneWorks and inference heads are final, clean, reviewed, and pin-matched.

SC-21306's reviewed recovery is deliberately sparse. It accepts only exact GitHub artifact
`9492288293` from run `32628540694`, validates that artifact under its frozen legacy profile, and
rehashes all receipt evidence. It imports only PASS cells whose individual semantic tuples are
unchanged: ordinals 1-13 and 15-17. The legacy LTX failure at ordinal 14 and the two Illustrious
failures at ordinals 18-19 are quarantined, never promoted. Only ordinals `[14, 18, 19]` execute
under the corrected profile. The summary retains the complete prior lineage and the explicit sparse
execution list; it never represents the segments as one run.

## Frozen scope

`config/terminal-evidence/epic-20738-cuda.json` is the review authority. The controller rejects any
change to the canonical digest of all 19 ordered semantic tuples, including model/engine/kind,
artifact IDs, requested tier, capability, and every request key/value:

1. Chroma 1 base, flash, and HD at q4 and q8 (six cells).
2. FLUX.1 dev and schnell at q4 and q8 (four cells).
3. SCAIL2 at q4, the ordered six Reference/Mask-pair multiReference boundary at q4, and q8 (three
   cells).
4. LTX-2.3 at q8 (one cell).
5. SDXL, RealVisXL, RealVisXL Lightning, Illustrious XL v1, and Illustrious XL v2 with the one approved
   SDXL OpenPose ControlNet at q4 (five cells).

Anima, SANA, VACE, FLUX.2 TrueV2, and historical Eros surfaces are rejected by source validation.
The four deliberately dismantled pin-keyed gates are not used or changed. This profile is terminal
evidence, so it must not be added to routine measurement, calibration, canary, or memory-matrix runs.
LTX-2.3 distilled uses its fixed eight-step schedule; any four-, seven-, or nine-step profile
mutation fails closed. The corrected ordered cell digest is
`2fcd20e4909f0bd0ba6c78c6a85247267c354735f77f4ed4912d47941a8512c1`; the reviewed 23-authority
digest after the truthful LTX parent change is
`5b9ef60c18ab15caeca7ff0411b199618f0aa22cc051a70607aa7a0f7c6cd932`.

## Dispatch contract

Select **Windows Candle worker** in GitHub Actions and set:

- `run_epic_20738_terminal_cuda`: `true`
- `inference_revision`: the exact 40-character inference SHA pinned by that SceneWorks head

The SceneWorks revision is always the checked-out workflow SHA and is not a caller-controlled input.
Leave `run_five_rung_reference`, `provision_snapshot`, and `run_ltx_eros_acceptance` false. The
workflow rejects a combined dispatch. Its fixed terminal concurrency key admits only one campaign at
a time across the Windows CUDA pool.

The terminal dispatch arm gives this one strictly serial job a 720-minute hard timeout. That is a
job-level budget below GitHub's five-day self-hosted execution limit, not a step timeout; no step is
assigned a timeout above 360 minutes. The unrelated LTX Eros, provisioning, five-rung, and ordinary
job budgets remain 360, 240, 120, and 45 minutes respectively.

The job checks out exact source revisions and requires both checkouts to be clean. It installs the
lockfile-pinned in-process Node Draft 2020-12 validator and creates a fresh `RUNNER_TEMP` Python
virtual environment with `huggingface_hub==0.36.0`, then verifies and passes that exact interpreter
to the controller. The trusted source cache is the fixed
`E:\huggingface\hub`; it is not a dispatch input and is never used as writable runtime storage.
Before any GPU cell, the controller freezes and hashes the target bytes in a census of every distinct remaining exact
repository/revision/allow-pattern authority. The exact required filenames come from the checked-in
immutable download-pattern evidence; every listed file is required, and only those files are copied.
Unreviewed extras such as `.incomplete` blobs and model-adjacent `.candle-device-format-v1`
derivatives are excluded. Hugging Face snapshot file links are valid only when
their resolved blobs remain inside that trusted root; broken, empty, or escaping links fail closed.

The sparse recovery census is exactly eight logical authorities and 70 files: the LTX q8/Gemma pair,
the two Illustrious primaries, and the four shared SDXL helpers. File sizes and hashes follow valid
Hugging Face links to their trusted blob targets; link-entry length is never used as model-byte
accounting. The exact followed-target total is 66,821,159,278 bytes.

Authorities are copied just in time into campaign-owned staging under `RUNNER_TEMP`, immediately
before their first selected consumer, and are retained only through their exact last selected
consumer. The LTX q8/Gemma pair is staged together for cell 14. The four shared SDXL helpers live
across sparse cells 18-19 while the two Illustrious primaries rotate after one cell. Existing valid
files are never overwritten or downloaded. Because Schnell is outside this sparse scope, any missing
file fails preflight; the campaign performs zero network downloads and forces offline mode before
the first selected cell. LTX q8 and Gemma use the complete production-approved cached parent revision
`254989c3ca7ee691187647f350b112c0c448789d`, not the absent current revision. Network mode stays
offline for every JIT stage and cell.

The disk admission model binds exact source bytes, the conservative streamed-FLUX sidecar ceilings
(494 files; q8 12,573,868,032 bytes and q4 7,396,392,960 bytes, each with at most 16 KiB per-file
reserve), and a
40-GiB non-model reserve for Cargo target, output, the pinned venv, logs, and filesystem fluctuation.
The largest model-plus-sidecar live set is therefore the LTX pair itself at
56,156,615,634 bytes, so the controller requires at least 99,106,288,594 free bytes. It checks that
floor after the missing-file fill, before every authority stage, and before every GPU process. The
former all-at-once plan is deterministically refused.

Output and scratch must be fresh, distinct, non-nested descendants of the resolved `RUNNER_TEMP`,
outside both repositories. Every recursive removal rechecks that confinement and rejects
symlink/reparse replacement.

The Node controller awaits one fresh `sceneworks-worker` process for each selected cell with
`--test-threads=1`; it executes exactly ordinals 14, 18, and 19, never starts cells in parallel, and
exposes no per-cell dispatch input. The Rust
entrypoint constructs the production load spec and deterministic request in reviewable code. Packed
tier markers must agree with the requested q4/q8 directory, and runtime results must say requested
tier equals resolved tier with `denseFallback: false`. The same runtime result carries a closed
request-memory record: current FLUX cells report `default-resident`, `requestMemoryPresent: false`,
`stageResidency: false`, and `streamTransformerBlocks: false`; non-FLUX cells report exact
`not-applicable`.

All writable cache and temporary environment variables are redirected to the current cell's scratch,
while model roots point only at the active JIT staging roots. Every selected root and nested
top-level component root carries an ordinary-file `.candle-device-format-v1` obstruction, forcing
Candle away from model-adjacent writes. `SCENEWORKS_CANDLE_DEVICE_CACHE_DIR` points to a separate
campaign-owned derived root under `RUNNER_TEMP`. The terminal request leaves `GenerationRequest.memory`
at its production default: on an ample device, a successful resident/non-streamed FLUX request must
leave the canonical derived root exactly empty. If the provider instead exercises its bounded
transformer path, exactly one b646 component-path-derived namespace shares the authority's lifetime
and must contain the reviewed bounded 494-file sidecar set. Partial, stray, multiple, wrong, or
non-regular derived entries fail closed in either case; every non-FLUX root must stay exactly empty.
An early provider failure may retain only its separately labeled exact-empty namespace evidence. The
controller validates the runtime request-memory record before accepting a successful derived
disposition: default/explicit/staged resident strategies pair only with exact empty evidence, while
`bounded-transformer` requires the exact present/staged/streamed tuple (`true/true/true`) and pairs
only with the canonical 494-file namespace. Streamed-without-staged residency is invalid. A provider
failure has no runtime-result strategy claim. The controller rehashes
each selected staged authority file and every obstruction immediately before and after every cell,
then proves exact stage/namespace absence at release and proves final staging, derived cache, and
missing-file store emptiness. Any stage, hash, capacity, or cleanup drift quarantines later cells; the
source cache remains untouched.

## Evidence and failure behavior

Every cell directory contains a receipt, controller/runtime logs, the generated input summary and
input witnesses where applicable, the runtime result, and output witnesses. SHA-256 entries bind all
persisted inputs, outputs, and logs. Receipts also bind:

- exact clean SceneWorks and inference SHAs;
- artifact repository, revision, role, selected subdirectory, allow-list, byte/file counts, and full
  inventory SHA-256;
- requested and resolved tier plus the explicit no-dense-fallback verdict;
- workflow run, head, attempt, runner OS/architecture/name, system-memory identity, GPU index/PCI
  identity/UUID/compute capability/driver/total memory, and raw VRAM samples.

The controller accepts exactly one candidate for artifact `9492288293`, size 15,452,320 bytes,
GitHub digest `sha256:dbae4c7d67d824bb8568909231614c6bcc268868087eb19974ce013bfc557724`,
SceneWorks head `43c718b7e9a852bd5029448d18841fed0f508c3a`, and inference pin
`b646a6f89ba9f6b07efe53dd583d8a42e21e9871`. Its source cell digest is the legacy
`dc0e529b40e898727eb9401562a928345b958b4c94677d0206ccc70471f6f879`. The importer validates the
closed campaign summary, full prior 1-9 lineage, both cache phases, cleanup, exact provenance, schema,
and rehashed input/output/log evidence for all 19 receipts. It then compares each PASS cell's
canonical tuple independently with the corrected profile. A global digest match is never used to
waive a per-cell mismatch. Ordinals 14, 18, and 19 must remain failures and are recorded as
quarantined; any attempt to promote one invalidates the candidate.

The closed Draft 2020-12 `cache-preflight.json` binds the download-evidence SHA-256, authoritative
filename census, followed target bytes/hashes, hit/download partition, first/last-use plan, persistent
store, exact disk plan/free floor, and the no-GPU-before-validation verdict. Its own byte count and
SHA-256 are bound from the campaign summary. Each continuation receipt binds that cell's live-set
transition, free-space probes, pre/post immutable verification, the exact derived disposition
(`resident-empty`, `bounded-transformer-sidecars`, `provider-failed-empty`, or `not-applicable`),
its closed request-memory strategy, derived inventory, and exact releases;
the final summary binds the complete lifecycle and empty terminal state.

A cache preflight directory, write, stat, hash, schema/semantic validation, census, staging, or final
offline-validation failure starts no sparse GPU cell. An independent emergency writer retains the
failed preflight evidence, and the controller emits durable failed outcomes for ordinals 14, 18, and
19 while retaining the 16 compatible imported PASS receipts. A cell
failure anywhere in setup, provisioning, execution, hashing, schema validation, atomic
receipt publication, or cleanup is recorded and the controller proceeds through all reviewed cells.
If a per-cell scratch removal fails or its absence cannot be proven, the controller quarantines the
campaign: it skips setup, provisioning, and execution for every later cell while still emitting
bound failure receipts for all 19 outcomes. It reports the aggregate failure only after iterating all
19. The workflow uploads the run-scoped evidence with
`always()` before enforcing that aggregate verdict, so a later load, generation, comparison, or
schema failure cannot hide earlier receipts. A provisioning failure records the exact attempted
authority and the reason its inventory is incomplete. If primary log, evidence, validation, or
atomic receipt finalization fails, an independent writer publishes a fresh schema-valid failure
receipt under `_emergency/`. A failed primary campaign-summary write likewise uses an independent
`_emergency/campaign-summary-fallback.json`, preserving all 19 outcomes before the verdict.

## Source-only checks

These checks do not dispatch hardware or download weights:

```text
npm ci --ignore-scripts
node scripts/epic-20738-terminal-cuda-harness.mjs check
node --test scripts/epic-20738-terminal-cuda-harness.test.mjs scripts/hash-artifact-inventory.test.mjs
python -m unittest scripts/provision_epic_20738_terminal_artifact_test.py
cargo fmt --all -- --check
```

The hardware test is `#[ignore]`d, test-only, Candle-only, and additionally refuses to run unless the
workflow-only `SCENEWORKS_ENABLE_EPIC_20738_TERMINAL_CUDA=1` opt-in is present.
