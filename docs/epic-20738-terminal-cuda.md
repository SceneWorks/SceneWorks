# Epic 20738 terminal CUDA evidence profile

SC-20945 adds one deliberately opt-in terminal evidence profile to
`.github/workflows/windows-candle.yml`. It is source preparation, not permission to run a campaign.
The profile stays off on pushes, pull requests, and ordinary manual runs. Run it once only after the
epic's SceneWorks and inference heads are final, clean, reviewed, and pin-matched.

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

## Dispatch contract

Select **Windows Candle worker** in GitHub Actions and set:

- `run_epic_20738_terminal_cuda`: `true`
- `sceneworks_revision`: the exact 40-character SHA selected for the workflow run
- `inference_revision`: the exact 40-character inference SHA pinned by that SceneWorks head

Leave `run_five_rung_reference`, `provision_krea_snapshot`, and `run_ltx_eros_acceptance` false. The
workflow rejects a combined dispatch. Its fixed terminal concurrency key admits only one campaign at
a time across the Windows CUDA pool.

The terminal dispatch arm gives this one strictly serial job a 720-minute hard timeout. That is a
job-level budget below GitHub's five-day self-hosted execution limit, not a step timeout; no step is
assigned a timeout above 360 minutes. The unrelated LTX Eros, provisioning, five-rung, and ordinary
job budgets remain 360, 240, 120, and 45 minutes respectively.

The job checks out exact source revisions and requires both checkouts to be clean. It installs the
lockfile-pinned in-process Node Draft 2020-12 validator and creates a fresh Python environment only
for the pinned anonymous artifact client, then downloads
each public Hugging Face artifact at its exact commit and allow-listed paths into isolated per-cell
scratch. Runner caches, weight-root secrets, and pre-existing snapshots are not accepted as
authorities. Output and scratch must be fresh, distinct, non-nested descendants of the resolved
`RUNNER_TEMP`, outside both repositories. Every recursive removal rechecks that confinement and
rejects symlink/reparse replacement.

The Node controller awaits one fresh `sceneworks-worker` process for each cell with
`--test-threads=1`; it never starts cells in parallel and exposes no per-cell dispatch input. The Rust
entrypoint constructs the production load spec and deterministic request in reviewable code. Packed
tier markers must agree with the requested q4/q8 directory, and runtime results must say requested
tier equals resolved tier with `denseFallback: false`.

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

A cell failure anywhere in setup, provisioning, execution, hashing, schema validation, atomic
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
cargo fmt --all -- --check
```

The hardware test is `#[ignore]`d, test-only, Candle-only, and additionally refuses to run unless the
workflow-only `SCENEWORKS_ENABLE_EPIC_20738_TERMINAL_CUDA=1` opt-in is present.
