# Epic 20738 terminal CUDA evidence profile

SC-20945 adds one deliberately opt-in terminal evidence profile to
`.github/workflows/windows-candle.yml`. It is source preparation, not permission to run a campaign.
The profile stays off on pushes, pull requests, and ordinary manual runs. Run it once only after the
epic's SceneWorks and inference heads are final, clean, reviewed, and pin-matched.

SC-21306's current recovery accepts only exact GitHub artifact `9498929065` from run `32655428377`.
It revalidates and imports the authentic PASS ordinals 1-13 and 15-19, retains their complete prior
lineage, and quarantines the failed LTX receipt at ordinal 14. Only ordinal `[14]` executes with the
corrected inference pin. The earlier `9492288293` 16-PASS route remains a separately frozen legacy
compatibility path; no receipt or provenance is rewritten across either boundary.

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
`2fcd20e4909f0bd0ba6c78c6a85247267c354735f77f4ed4912d47941a8512c1`; the current reviewed
23-authority digest is
`1e98392f71b1ad3d10d4bf18a6f23a497f5ffe588127ac59c54e53d392e6e255`.

The current Illustrious primaries select published immutable revisions
`778c3f02b7703b0c2755d0c0447592897193c6b5` (v1) and
`672e9851ede4dc856fa945649b6691975c9d74a3` (v2). Their q4 text encoder, second text encoder, and
UNet configs must each carry the exact `bits: 4`, `group_size: 64` marker represented by the closed
profile marker. The current download inventory is independently digest-bound as
`1fa06ef39a0e2c321a4fa15fa1128c0157ba8cf22fd868ac54c6cefaec13a5ee`. The republished
authorities change configuration metadata only; their packed q4 component configs are bound to
SHA-256 `f87b89e4249e027632236caba75d1140e14fd4c2ce4b4e554f2912b234e72cf9`,
`3b96bc14843360d24e864f7d1ac6d83e95cad8f68209e7e503cefa9a4f65b18b`, and
`aeb34c12f61f1edd9f7e17d8332f91197bacad70754bfaa450836137c40c8c4d` for the first text
encoder, second text encoder, and UNet respectively. The q8 component config hashes are
`c74289384c56ae6fdac29b39f01c081aa7dbd20161b371c5dc3b486fa94bf8fb`,
`2891a3c519b99c3b5983aeee973095974f2e08df5d88e7d59c88635638fa8d6a`, and
`379fb37b8b7a893113925439117acebc03b205615b281bcc139b7c418b26ce7f` in the same order. The
VAE weight identities and measured decode baselines are unchanged.

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
accounting. The exact followed-target total is 66,821,159,668 bytes; the 390-byte increase is the six
reviewed q4 config markers across the two current Illustrious authorities.

Authorities are copied just in time into campaign-owned staging under `RUNNER_TEMP`, immediately
before their first selected consumer, and are retained only through their exact last selected
consumer. The LTX q8/Gemma pair is staged together for cell 14. The four shared SDXL helpers live
across sparse cells 18-19 while the two Illustrious primaries rotate after one cell. Existing valid
files are never overwritten or downloaded. The exact current Illustrious v1/v2 q4 snapshots may be
fully or partially absent: their frozen 19-file inventories are the only sparse authorities approved
for hydration. Every present file must match its reviewed size and SHA-256, and every missing file is
downloaded by exact repository, revision, path, size, and content SHA (plus LFS SHA where applicable)
into the isolated campaign store.
Unexpected entries, old revisions, file-list/evidence drift, corrupt bytes, and unsafe links fail the
source census before transfer. LTX q8 and Gemma use the complete production-approved cached parent
revision `254989c3ca7ee691187647f350b112c0c448789d`, not the absent current revision. Network mode
stays offline after the reviewed fill, for every JIT stage and cell; the combined staged authority is
audited again without weakening the post-download byte checks.

The disk admission model binds exact source bytes, the conservative streamed-FLUX sidecar ceilings
(494 files; q8 12,573,868,032 bytes and q4 7,396,392,960 bytes, each with at most 16 KiB per-file
reserve), and a
40-GiB non-model reserve for Cargo target, output, the pinned venv, logs, and filesystem fluctuation.
Before hydration, the largest model-plus-sidecar live set is the LTX pair at 56,156,615,634 bytes.
A fully absent pair of current Illustrious q4 snapshots adds 3,911,656,986 and 3,911,656,662
persistent bytes before ordinal 14, producing the reviewed 63,979,929,282-byte physical peak. With
the 40-GiB reserve, the controller therefore requires at least 106,929,602,242 free bytes. It checks
that floor after the missing-file fill, before every authority stage, and before every GPU process.
Each downloaded authority store becomes that authority's stage root and is released at its exact
ordinal-18/19 lifetime boundary. The former all-at-once plan is deterministically refused.

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

The current controller accepts exactly one candidate for artifact `9498929065`, size 16,071,005
bytes, GitHub digest
`sha256:fae791001dd4e2015ce0567290b9b0a1d67de9e503712d2b9a60a0f9af07ec9c`, SceneWorks head
`655414ef3e4dec1fe9142901caea538e73ac1490`, run `32655428377`, attempt 1, and historical source
inference pin `b646a6f89ba9f6b07efe53dd583d8a42e21e9871`. Its cell and artifact semantic digests are
`2fcd20e4909f0bd0ba6c78c6a85247267c354735f77f4ed4912d47941a8512c1` and
`1e98392f71b1ad3d10d4bf18a6f23a497f5ffe588127ac59c54e53d392e6e255`. The importer rehashes all
18 PASS receipts and the failed cell-14 receipt, validates both current cache phases and exact prior
lineage, and refuses any attempt to promote failed cell 14. Its successor executes only cell 14 with
the two LTX authorities (28 files, 56,156,615,634 source bytes) and the reviewed
106,929,602,242-byte disk floor.

The preserved legacy controller accepts exactly one candidate for artifact `9492288293`, size
15,452,320 bytes,
GitHub digest `sha256:dbae4c7d67d824bb8568909231614c6bcc268868087eb19974ce013bfc557724`,
SceneWorks head `43c718b7e9a852bd5029448d18841fed0f508c3a`, and inference pin
`b646a6f89ba9f6b07efe53dd583d8a42e21e9871`. Its source cell digest is the legacy
`dc0e529b40e898727eb9401562a928345b958b4c94677d0206ccc70471f6f879`. The importer validates the
closed campaign summary, full prior 1-9 lineage, both cache phases, cleanup, exact provenance, schema,
and rehashed input/output/log evidence for all 19 receipts. It then compares each PASS cell's
canonical tuple independently with the corrected profile. A global digest match is never used to
waive a per-cell mismatch. Ordinals 14, 18, and 19 must remain failures and are recorded as
quarantined; any attempt to promote one invalidates the candidate.

Artifact `9492288293` remains bound to the frozen pre-marker Illustrious revisions
`c5a92a902dd4e6ee99c2a57981ecf66209905dd1` (v1) and
`7c5c8b2bb75a8f38a7365e70bdf84d38d6204473` (v2), artifact-authority digest
`5b9ef60c18ab15caeca7ff0411b199618f0aa22cc051a70607aa7a0f7c6cd932`, and download-evidence
digest `9eda09eeacb9386167ca4a080b4805b9c7dd3cd5134ca037ce342ad434b17e0b`. The current inventory
retains those exact legacy rows only for importer selection; it never re-labels old receipts with
the new revisions or digest.

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

## Prior independently audited promotion outcome

The promotion authority is GitHub Actions run `32628540694`, attempt 1, exact SceneWorks head
`43c718b7e9a852bd5029448d18841fed0f508c3a`, artifact
`sc-20945-epic-20738-43c718b7e9a852bd5029448d18841fed0f508c3a-32628540694-1`
(artifact id `9492288293`, 15,452,320 bytes, GitHub SHA-256
`dbae4c7d67d824bb8568909231614c6bcc268868087eb19974ce013bfc557724`). The independent audit
rehashes the original cells 1-7, recovery cells 8-9, and current cells 10-19 rather than trusting the
campaign summary. All selected inputs, outputs, controller/runtime logs, repository heads, pins,
runtime results, and receipts match their imported lineage and frozen profile/schema/semantic
digests. Thirty-one output PNGs are non-degenerate, including SCAIL-2's exact six-pair
counterfactual boundary.

Exactly cells 1-13 and 15-17 are product-promotable. Receipt peaks are physical-target GPU MiB;
manifest GiB values are MiB / 1024 rounded to three decimals, and a new default-tier blanket adds
the worker-owned 2 GiB reserve and rounds up. FLUX keeps its older, larger 1024x1024 admission rows
because replacing them with lower 512x512 observations would weaken an independently established
guard.

| Cells | Product cells | Receipt peaks (MiB) | Receipt-backed memory evidence |
| --- | --- | --- | --- |
| 1-2 | Chroma1 Base q4 / q8 | 16,676 / 25,094 | q4 16.285 / 19 GiB floor; q8 24.506 / 27 GiB floor |
| 3-4 | Chroma1 Flash q4 / q8 | 16,676 / 23,364 | q4 16.285 / 19 GiB floor; q8 22.816 / 25 GiB floor |
| 5-6 | Chroma1 HD q4 / q8 | 16,708 / 23,364 | q4 16.316 / 19 GiB floor; q8 22.816 / 25 GiB floor |
| 7-8 | FLUX.1 dev q4 / q8 | 11,556 / 23,752 | terminal 11.285 / 23.195; stricter existing 24 / 34 GiB floors retained |
| 9-10 | FLUX.1 Schnell q4 / q8 | 14,733 / 22,268 | terminal 14.388 / 21.746; stricter existing 24 / 34 GiB floors retained |
| 11-13 | SCAIL-2 q4, q4 six-reference, q8 | 58,890 / 62,730 / 66,486 | q4 61.260 / 64 GiB floor; q8 64.928 / 67 GiB floor |
| 15 | SDXL q4 + OpenPose | 7,171 | q4 7.003 / 10 GiB floor |
| 16 | RealVisXL q4 + OpenPose | 9,649 | q4 9.423 / 12 GiB floor |
| 17 | RealVisXL Lightning q4 + OpenPose | 6,915 | q4 6.753 / 9 GiB floor |

Promotion deletes only the twelve proven precision exception paths: six Chroma, four FLUX.1, and
two SCAIL-2 q4/q8 cells. The three accepted SDXL OpenPose compositions had no exception paths to
delete; their catalog, routing, install companion, and q4-only worker validation are narrowed to
those exact models and tier.

The Chroma and OpenPose rows are deliberately not written as partial generic `candle` blocks. That
block is both the base-route VRAM gate and the complete advertised tier axis. Chroma's previously
established bf16 route has no terminal cell here, and the shared SDXL base routes retain independent
q8/bf16 authority. Partial q4/q8 or q4-only ladders would make those tiers borrow a smaller default
floor and erase them from the generated memory matrix. Chroma therefore retains its pre-existing
base-route admission behavior, while the bespoke OpenPose route validates q4 before download/load
and performs its existing load-exact base-plus-overlay admission. The table records the measured
composition rows without turning them into dense or cross-tier authority.

Cells 14, 18, and 19 are not promotion authority:

- Cell 14 requested four LTX-2.3 q8 steps, but the immutable package has a baked fixed eight-step
  recipe. LTX q8 returns to the epic-9083 precision exception and remains fail-closed; q4 is
  unchanged.
- Cells 18-19 selected Illustrious XL v1/v2 q4 authorities without the packed-quantization marker.
  They do not authorize on-the-fly or dense fallback. Both models remain absent from the shared
  OpenPose install/UI surface and the exact worker/router allow-list.

The audit also verifies GPU index/PCI identity/UUID/driver/compute capability and confinement to the
97,887 MiB target, cache-preflight census and capacity, JIT authority lifecycles, derived-cache
dispositions, per-cell scratch cleanup, and final emptiness of staging, derived, and persistent-miss
stores. No rerun is needed for cells 15-17: their complete receipts and independently hashed outputs
stand on their own. The safest recovery is one later sparse terminal campaign containing only LTX q8
with its fixed eight-step request and the two Illustrious q4 cells after republished packed-marker
authorities; it must import these sixteen PASS cells rather than rerun them.

This promotion closes only SC-20741, SC-20744, and SC-20748. SC-20742 remains blocked on the
separate FLUX.2 TrueV2 authority and is not implied complete by this terminal campaign.
