# Shared memory-strategy contract

SC-15449 defines one worker-owned selection path for every image provider. The generated
[`memory-matrix.json`](generated/memory-matrix.json) is the authoritative inventory and
evidence schema; `sceneworks_worker::memory_strategy::select_strategy` is the authoritative runtime
selector. Providers may reject a selected strategy defensively, but must not contain a second
least-cost selector.

## Ownership

| Owner | Responsibilities |
| --- | --- |
| Provider | Capabilities, cache/load/drop lifecycle, estimate formula, backend realization, calibration ABI fingerprint, and defense-in-depth validation |
| Manifest/generated evidence | Coefficients, envelopes, provenance, backend/tier/mode/overlay/geometry/parameter coverage, conformance state, and structural N/A |
| Worker | Live budget and reclaimable memory, ordered least-cost selection, precision invariance, rejection/fallback, telemetry, and user advice |

The five ordered strategies are `Resident`, `StagedResidency`, `BoundedDecode`,
`BoundedAttention`, and `BoundedTransformerResidency`. Candle/CUDA budgets discrete live VRAM and
may credit the evicted single-slot cache. MLX/Metal budgets total unified memory and reserves the OS
share as explicit headroom. In both cases effective capacity is
`min(total - reserve, available + reclaimable - reserve)`, saturating at zero. Those backend
semantics are not interchangeable.

The selector accepts only `Verified` evidence with all six generated evidence dimensions, matching
provider/calibration fingerprints, current SceneWorks and inference revisions, an exact tier/mode/
overlay match, and an in-envelope geometry. `Implemented/unverified`, unknown, stale, fingerprint
mismatch, structural N/A, route unavailable, or out-of-envelope evidence returns `Unverified`; it
never selects an optimized rung. Equality fits (`needed_gb <= available_gb`). A calibration that
needs tolerance must include it in the provider estimate and golden evidence.

Strategy changes never change precision. A lower precision tier is a separate candidate evaluated
by the existing tier chooser before the memory selector.

## Tier integrity

**No component is resident above the user's selected quant tier unless a declared, measured exception
says otherwise.** (SC-15799.)

Choosing q8 or q4 *is* a memory decision. If a user picks q4 because bf16 does not fit, carrying any
component at bf16 defeats the choice they made. This is a rule about the catalog, not a per-entry
observation: the **rule** binds every entry, on every hosted tier, on both backends. What the rule
binds is not the same as what the catalog observes — see point 4: the *residency* a declared tier
produces is a per-backend fact, so "it holds on both backends" is a statement about the obligation, not
a claim that the two lanes hold the same component at the same precision.

Four things follow, and they are not negotiable individually:

1. **The packing tier is not a memory lever.** A component's precision is decided by the tier the user
   selected, before any budget is read, and it is identical on a 16 GB card and a 96 GB one. Packing
   something to the tier the user already asked for is not a concession to a tight budget, so it must
   never sit in an escalation ladder. `gen_core::tier_integrity` is the executable rule;
   `gen_core::mempolicy` no longer has a `BranchQuant` lever.
2. **Every above-tier residency is declared.** `config/tier-integrity.jsonc` is the ledger: per entry,
   per component, the precision it is actually resident at, the selected tiers on which that is above
   tier, why, and what it costs. `scripts/check-tier-integrity.mjs` validates it in `npm run check`
   (and with it the `parity` lane) and generates the audit table at
   [`docs/generated/tier-integrity.md`](generated/tier-integrity.md). A declaration is not optional
   paperwork — an exception the shared decision cannot see is the defect, not the residency itself.
   The ledger is complete **against a stated threshold**, which its own header spells out (named
   components at any size, in; per-channel norm/modulation vectors, out). It does not claim
   completeness in the abstract: the first revision did, and a review then found three omissions whose
   costs were already in the tree. If you find an above-tier component on the declare side of that
   threshold with no row, that is an omission — file it.
3. **An undeclared or unmeasured exception is a defect.** A declared one carries the measurement that
   justifies it. Where the catalog has above-tier residency whose isolated cost is not yet measured, the
   row says so and names the story that owes it; the checker permits that **only** for the exact
   `(model, component)` pairs grandfathered when the ledger was created, and pins the set's size against
   a committed constant so any change is a two-place edit a reviewer has to see. Keying the amnesty per
   *model* was not a ratchet: it handed every amnestied entry a free slot for every component it had not
   yet declared. A grandfathered pair with no matching row is also an error, so promoting a row to
   `measured` must delete its amnesty line rather than leave the slot open.
4. **The same declared tier can yield different residency on different backends, and that is a
   tier-integrity fact in its own right — not a footnote to one.** The two ports make dtype decisions
   independently, so "this entry is q4" does not fix what any given component is resident at until the
   lane is named. SenseNova-U1's vision conv kernels and `fm_head` are bf16 under `mlx-gen-sensenova` but
   **f32** under `candle-gen-sensenova`, which widens every dense leaf it multiplies against an f32
   activation; the `mage_flow*` `transformerHead` q8 floor exists only in `mlx-gen-mage`, so a q4 candle
   render there really is uniformly q4. It follows that `residentTier` is a per-lane quantity, that the
   ledger's row identity is **(model, component, backend lane)** rather than (model, component), and that
   a component whose lanes disagree is declared as one row per lane — declaring a single value would
   either over-declare one lane or under-declare the other, and under-declaring is the direction that
   hides an above-tier residency. The lane is validated, cross-checked against the lanes the catalog
   entry actually has, and published as a column of the generated audit; a per-backend claim that the
   audit does not show is a claim nobody can review.

The exceptions divide by *cause*, and the causes have different fixes. A **packing exception** is a
deliberate quality decision (a precision-sensitive decoder, a control residual that drifts) and needs a
measurement. A **backend capability** gap means the backend cannot serve the on-disk format and upcasts;
packing cannot fix it, so it needs a compute path or a different published dtype and is tracked
separately. A **structural** case has nothing quantizable at all. Reading them as one bucket is how a
capability gap gets mistaken for a packing choice and "fixed" by repackaging that cannot help.

Two mechanisms are load-bearing today. `mlx.denseTextEncoderTier` is the only way an entry obtains a
dense text encoder on a packed tier, and setting it requires a matching ledger row — the hardcoded
worker registry it used to mirror is deleted, because ids living in Rust while the catalog declared
nothing is exactly the invisible carve-out this rule exists to remove. `candle.control.branchTierByBaseTier`
declares the Krea pose-control branch's tier per base tier: q8 follows the tier, bf16 is already at
tier, and **q4 floors its branch at q8** — the one declared, measured exception, because a q4 control
residual measures "pose-locked; non-pose details drift" and the residual is the thing the user asked
for. That floor is the rule working, not a hole in it.

### What tier integrity gives up

The rule costs something, and a rule whose cost is unstated is a rule nobody can weigh. Removing the
Krea branch-quant rung removed two configurations that used to fit:

- **A q4 base could pack its branch to q4** — *at* the selected tier, so tier integrity never forbade it.
  It is refused on QUALITY grounds instead (the measured drift), which is the declared exception above and
  is recorded with its cost in the ledger.
- **A q8 base could pack its branch to q4** — one tier **below** the selection. Tier integrity permits
  below-tier residency outright; only *above*-tier needs an exception. That rung went because it was a
  *rung*, not because the invariant required it. On the shipped rows it admitted a q8 control job down to
  **~28.97 GB** free where the tier-integral configuration needs **~30.77 GB**. That ~1.8 GB band is a
  real capability traded away so that no render silently substitutes precision the user did not select.
  (Both figures come from the retracted `candle.control.branchPackSaveGb`; on the corrected weight-side
  accounting — the branch is 6.6 GB bf16, ~3.3 GB at q8, ~1.7 GB at q4 — a q4 branch buys ~1.6 GB, so the
  true band is slightly narrower. SC-16013's re-measure pins it.)

Neither is a *reject* today: while `candle.control.measured` is `false` the control fit ladder's floor is
a best-effort admit, because a reject computed from a superseded upper bound can refuse a job that would
actually run — and refusing a job that fits is the failure an admission gate is least allowed to have.

**Repacking invalidates measurements.** A component's resident precision is an input to every peak
measured against it, so changing it makes those peaks stale even though every provenance field still
looks valid. Evidence captured before a repack must be renamed or fingerprint-bumped so it cannot read
as green — SC-15799 renamed the Krea control-lane rows to `bf16Branch*` and set
`candle.control.measured: false` for exactly this reason. **Tier integrity therefore precedes
calibration**: calibrating first means paying for those measurements twice.

**And renaming is not enough on its own — the reader has to honour the flag.** A superseded row may be
read as an **upper bound** and nothing more. It may not be "corrected" into the shipping configuration by
subtracting some other number, because a correction is a new measurement wearing a stale one's clothes;
SC-15799's first revision subtracted `branchPackSaveGb` (8.4 GB, against a control branch that is 6.6 GB
in total) and under-predicted a live CUDA admission path by ~5 GB one-directionally toward an OOM. It may
not produce a hard reject, since an upper bound cannot rule a job out. And it may not be recorded as a
reclaimable high-water, since over-stating a pool lets the next gate over-admit. `measured: false` is a
field the code reads, not a note to a future human.

**Reading an upper bound verbatim is the safe direction, not the free one — so quantify what it costs.**
"Over-predicting adapts a card early" is a disclosure only if *how* early is written down. For the Krea
control lane the answer is: **zero reject bands** (the only entry with a `candle.control` block has
superseded evidence, so a hard reject is unreachable) and a bounded set of **premature rung engagement**
bands, which cost render *speed* and nothing else. On a q4 base the gate stages instead of staying
resident across `[34.9, 38.2)` GB free, tiles its VAE decode unnecessarily across `[31.2, 34.5)`
(+26% decode), chunks attention unnecessarily across `[24.5, 27.8)` (~+6% render), and returns
`BestEffort`-with-warn where a clean `Fits` held across `[22.07, 25.37)`; on a q8 base (which has no
tiling rung) the bands are `[44.1, 47.4)`, `[39.17, 41.6)` and `[35.87, 39.17)`; on a bf16 base the
over-prediction is zero and so is every band, because the branch is already at tier. None of it is
user-visible — the `BestEffort` warning is a `tracing` event that reaches no response, record or UI. The
derivation and the per-rung arithmetic live in `crates/sceneworks-worker/src/krea_control_fit.rs`
("The remaining cost of reading the rows verbatim"); SC-16013's re-measure collapses the bands to zero.

## Lifecycle and telemetry

A provider capability declaration is selectable only when cancel and error transitions are safe:
each transition must either leave the warm cache reusable or invalidate it atomically. Warm cache
hits retain their selected realization; cold loads are re-estimated. The worker records the selected
strategy, estimate, effective live budget, rejection/unverified reason, and whether reclaimable
memory affected the outcome. Advice may name only a lower geometry or strategy that has verified
evidence.

## Reconciliation

- Krea 2 Turbo CUDA text-to-image keeps its measured boundaries. Resident admission now runs
  through the shared selector first; `threeStage`, `tiledVae`, `chunkedAttention`, and
  `streamedBlocks` map respectively to staged residency, bounded decode, bounded attention, and
  bounded transformer residency. Its manifest phase curves and maximum pixel envelope now carry
  explicit revision/fingerprint provenance and exact cumulative tile, overlap,
  attention-budget, and transformer-window parameters before flowing through the shared selector.
- Generic MLX cold loads keep the existing recursive safetensors disk sum, provider footprint
  component split, architecture headroom, OS reserve, and weights-fit safety floor. Resident and
  staged choices now flow through the shared selector; the provider capability registry remains the
  defense-in-depth check.
- Mage must not copy either selector. Its provider estimator must add request geometry, mode, overlay,
  and parameter terms to the provider-owned formula, publish matching evidence cells, and submit
  ordinary strategy candidates to the worker selector. Until that calibration is verified,
  request-aware optimized selection remains `Unverified` and the existing MLX weights safety floor
  is the only permissible fallback.

The provider ABI types and compatibility defaults live in the pinned SceneWorks inference
repository. A provider that does not implement the additive contract exposes resident-only,
unverified defaults, preserving its current behavior without claiming optimization.

## Adding a backend to a catalog entry

Matrix ownership is keyed per (entry, backend) and is not inferred (SC-15812): a cell must name the
Shortcut story that can actually close it, and an MLX story cannot be closed from CUDA hardware. So
adding a `candle` block to an entry in `config/manifests/builtin.models.jsonc` — or adding a new
entry — fails the `parity` lane until that backend's owning model story and family story are recorded
in `MODEL_STORIES`/`FAMILY_STORIES` in `scripts/generate-memory-matrix.mjs`. The failure is deliberate
and its message names the missing twin, but it does mean a catalog change is blocked on filing a
Shortcut story first. File the backend twin, add the id, then regenerate.
