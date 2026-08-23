# Epic 20738 terminal CUDA evidence profile

SC-20945 adds one deliberately opt-in terminal evidence profile to
`.github/workflows/windows-candle.yml`. It is source preparation, not permission to run a campaign.
The profile stays off on pushes, pull requests, and ordinary manual runs. Run it once only after the
epic's SceneWorks and inference heads are final, clean, reviewed, and pin-matched.

SC-20974's reviewed continuation is deliberately segmented. It imports and rehashes the contiguous
PASS prefix from cells 1-7 of the old `8886a9e69f26beec05688c81b414859bd102f6d0` run, quarantines
that run's non-executed cell-8 setup skeleton, and executes cells 8-19 under the corrected head. The
summary records both run identities; it never represents the two segments as one run.

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
The ordered cell digest remains
`dc0e529b40e898727eb9401562a928345b958b4c94677d0206ccc70471f6f879`; the reviewed 23-authority
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

The continuation census is exactly 16 logical authorities and 199 files. File sizes and hashes follow
valid Hugging Face links to their trusted blob targets; link-entry length is never used as model-byte
accounting. The reviewed followed-target total is 179,028,698,264 bytes (the earlier link-length
observation undercounted 18 Schnell-q8 targets by exactly 5,361,654,035 bytes).

Authorities are copied just in time into campaign-owned staging under `RUNNER_TEMP`, immediately
before their first consumer, and are retained only through their exact last consumer. SCAIL q4 lives
across cells 11-12; the four shared SDXL helpers live across cells 15-19 while each backbone rotates
after one cell. The LTX q8/Gemma pair is staged together for cell 14. Existing valid files are never
overwritten or downloaded.
The sole reviewed network exception is
`SceneWorks/flux1-schnell-mlx@bba3ae01dfd94089f173c05edd4e1a4c551f2599` file
`q8/transformer/model.safetensors`. If and only if the frozen census reports exactly that miss, the
pinned client fetches that exact filename into an isolated campaign-owned persistent store and proves the returned commit,
metadata size, LFS SHA-256, and actual file SHA-256. It never calls a snapshot/glob/ref-main route;
existing `.incomplete` files are untrusted and are neither renamed nor resumed. LTX q8 and Gemma use
the complete production-approved cached parent revision
`254989c3ca7ee691187647f350b112c0c448789d`, not the absent current revision. After the one possible
fill, network mode is forced offline for every JIT stage and cell.

The disk admission model binds exact source bytes, the conservative streamed-FLUX sidecar ceilings
(494 files; q8 12,573,868,032 bytes and q4 7,396,392,960 bytes, each with at most 16 KiB per-file
reserve), and a
40-GiB non-model reserve for Cargo target, output, the pinned venv, logs, and filesystem fluctuation.
The persistent Schnell-q8 miss is retained only through cell 10 and removed with that authority.
The largest model-plus-sidecar live set is therefore the LTX pair itself at
56,156,615,634 bytes, so the controller requires at least 99,106,288,594 free bytes. It checks that
floor after the missing-file fill, before every authority stage, and before every GPU process. The
former all-at-once plan is deterministically refused.

Output and scratch must be fresh, distinct, non-nested descendants of the resolved `RUNNER_TEMP`,
outside both repositories. Every recursive removal rechecks that confinement and rejects
symlink/reparse replacement.

The Node controller awaits one fresh `sceneworks-worker` process for each cell with
`--test-threads=1`; it never starts cells in parallel and exposes no per-cell dispatch input. The Rust
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

The controller discovers artifacts by the fixed old-head terminal prefix and accepts exactly one
candidate whose first seven primary receipts are contiguous PASS outcomes at inference pin
`b646a6f89ba9f6b07efe53dd583d8a42e21e9871`, old cell digest
`dc0e529b40e898727eb9401562a928345b958b4c94677d0206ccc70471f6f879`, and old artifact digest
`f2bb7a77b83ce11cc32c3a1f9639534a67a149bc464a9730fb5c0988b4a03f9e`. It rehashes every imported
input, output, and log and retains the GitHub artifact SHA-256 in lineage. Cell 8 may exist only as
the initial controller log plus failed pre-execution
receipt: no selected/model bytes, input, runtime result, output, or cleanup attempt. That residue is
copied under `_imported-boundary-residue/` and excluded from the seven-PASS lineage. Any substantive
cell-8 file or any later cell invalidates the candidate.

The closed Draft 2020-12 `cache-preflight.json` binds the download-evidence SHA-256, authoritative
filename census, followed target bytes/hashes, hit/download partition, first/last-use plan, persistent
store, exact disk plan/free floor, and the no-GPU-before-validation verdict. Its own byte count and
SHA-256 are bound from the campaign summary. Each continuation receipt binds that cell's live-set
transition, free-space probes, pre/post immutable verification, the exact derived disposition
(`resident-empty`, `bounded-transformer-sidecars`, `provider-failed-empty`, or `not-applicable`),
its closed request-memory strategy, derived inventory, and exact releases;
the final summary binds the complete lifecycle and empty terminal state.

A cache preflight directory, write, stat, hash, schema/semantic validation, census, staging, or final
offline-validation failure starts no continuation GPU cell. An independent emergency writer retains
the failed preflight evidence, and the controller emits durable failed outcomes for cells 8-19 while
retaining the seven imported PASS receipts. A cell
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

## Independently audited promotion outcome

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
