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
provider/calibration fingerprints, the current inference revision, an exact tier/mode/overlay match,
and an in-envelope geometry. `Implemented/unverified`, unknown, stale, fingerprint
mismatch, structural N/A, route unavailable, or out-of-envelope evidence returns `Unverified`; it
never selects an optimized rung. Equality fits (`needed_gb <= available_gb`). A calibration that
needs tolerance must include it in the provider estimate and golden evidence.

SceneWorks invalidation is owned by the provider calibration ABI fingerprint, not by the exact source
tree. The fallback fingerprint hashes the provider's explicit ABI version, pinned inference revision,
model/provider/backend/tier/mode/overlay/rung identity, and runtime strategy parameters. Bump a
provider's entry in `CALIBRATION_ABI_VERSIONS` when its quantization floors, tensor layout, or
execution structure invalidate measurements; bump the default only for an ecosystem-wide calibration
contract change. `matrixSourceRevision`, `generatedFrom.sceneWorksRevision`, and cell
`evidenceRevision.sceneWorks` retain exact source-tree provenance, but comments, formatting, docs, and
unrelated source edits do not make otherwise matching evidence stale.

Strategy changes never change precision. A lower precision tier is a separate candidate evaluated
by the existing tier chooser before the memory selector.

## The cost order is ordinal, and it stays shared across backends

**One ladder, one order, on every backend — even where the same rung costs an order of magnitude more
on one of them.** (SC-16090.)

SC-15791 measured rung 4 on Candle/CUDA at ~8.0 s per denoise step against MLX's ~0.309 s for the same
rung, same model, same snapshot — ~26×, and ~4–5× more again on Candle-q8 than Candle-q4 — and asked
whether the ladder's cost order must therefore become per-backend and per-tier, or be replaced by a
selector reading measured per-backend cost. **Neither. Nothing about the order changes.** Three
independent reasons, each sufficient on its own:

1. **The ladder never trades cost against cost.** `select_strategy` walks the five rungs in order and
   returns the *first* whose measured peak fits the live budget — the cheapest sufficient rung. Rung 4
   is therefore reached only when no cheaper rung is **selectable**, so it never wins over a cheaper
   rung that fits, and a multiplier on it cannot flip that comparison in either direction. Its
   alternative is no render: usually a `Reject`, or an `Unverified` verdict where a cheaper rung's peak
   would have fit but its evidence was stale, out-of-envelope or fingerprint-mismatched (that path is
   real — excluding a cheaper rung does not stop the walk — and it yields a cheaper *non-answer*, not a
   cheaper render). `Candidate` carries no cost field.
   `the_cheapest_selectable_rung_wins_and_rung_four_never_beats_one_that_fits` pins all four cases,
   including that caveat.
2. **A cost *order* is ordinal; 26× is a magnitude, and no consumer of the order reads one.** The
   order's whole job is "try the cheaper mechanism first", which it discharges on any backend where
   rung 4 is still the most expensive rung. Per-backend *magnitudes* are already carried where they
   belong: per cell, as calibration evidence, keyed by backend. (SC-15791 timed no Candle rung other
   than 4, so "rung 4 is still last on Candle" remains the epic's premise rather than a measurement —
   it is not evidence this decision leans on.)
3. **The 26× is not a backend cost.** It decomposes entirely into per-window work that is not per-window
   work. Per 97.1 MiB block, medians on a host whose run-to-run spread reaches ±38%: **~204 ms** for the
   leg the spike labels *host read + repack* (the mapped read is inside that number, and on a cold page
   cache it dominates, since Candle re-reads the tier every step); **62.5 ms** attributable to a
   device→host round trip, as a residual rather than an independently timed leg; and the PCIe transfer
   itself **unresolvable against a ±34.9 ms noise floor**. The conversion is a pure deterministic
   function of bytes that never change, recomputed once per block per **forward** — 240 times per
   8-step render, 480 under true CFG — for one answer. Forking the shared order would write a fixable
   implementation defect into the contract every family story then has to honour.

**What was added instead is the invariant that makes the shared order true.**
`gen_core::memory_strategy::MemoryWindowMaterialization` is the obligation, asked backend-neutrally and
answered per realization: a rung-4 window must be a transfer of bytes **already in the form the
accelerator consumes** — a mapped read plus, where memory is not unified, a host-to-device copy — and
never a per-window format conversion. MLX satisfies it structurally, because there the mmap *is* the
residency. A realization that does not may still ship, but only while saying what it converts and naming
the story that removes it; `conformance_errors` refuses an undeclared one, and refuses an *unstated*
realization with the fix it can actually perform. `candle-gen-krea` declares `HostFormatConversion` today
with `owner_story: sc-16096`, which is the story that flips it to `DeviceFormatTransfer`.

It is declared on the backend realization and read through
`MemoryProviderContract::window_materialization`, mirroring how `engages` layers over
`MemoryStrategy::engages`. The bare `MemoryStrategy` enum is deliberately untouched: it is contract-owned
so no provider can opt out of arithmetic, and a rung-4 field there would change the rung for MLX too.
The field is **required** rather than defaulted — which is why every construction site had to change —
because a default is an opt-out that reads as a declaration. What the contract cannot do is *verify* the
answer: a provider that types `DeviceFormatTransfer` over a converting loader has made a false
declaration and nothing catches it. The compile-forced choice buys a deliberate statement a reviewer can
check against the loader, not a proof.

Declaring is deliberately **not** vetoing: within the ladder rung 4 at 8.0 s/step still beats the
rejected render that is its only alternative, so disabling it would trade a slow render for no render.
What the rule refuses is an **undeclared** realization, never a slow one — cost is not what it reads.
Note the severity that follows: like every rule in `conformance_errors`, an error is *contract*-level and
`select_strategy` bails on any non-empty result, so a misdeclaration costs rungs 0–4 rather than rung 4.
That is right for "this contract cannot be interpreted" and would be wrong for a cost verdict, which is
a second reason the rule does not attempt one.

### Where the magnitude reaches a tier decision — and why that is allowed

Reason 1 is a statement about the **ladder**, and must not be over-read into "the cost never matters".
The Krea capability-downtier consumes the ladder's verdict as a boolean: `KreaTurboFit::Resident` and
`KreaTurboFit::Fits` both collapse to `TierFit::Fits`
(`crates/sceneworks-worker/src/image_jobs/base.rs`), and `choose_downtier` keeps the highest-fidelity
tier that fits. So under **Auto** a tier whose *only* fit is rung 4 outranks a lower tier that fits
resident, and the cost of that rung is what the higher tier is bought with — extrapolating z-image's
measured q8 figure (~1466 ms/block), tens of seconds per step rather than a q4 resident render's
low single digits.

**That is intended, and is the same rule the rest of this epic applies.** Fidelity the user asked for is
not a memory lever: tier integrity refuses to substitute precision, SC-15807 refuses to substitute
geometry, and refusing to substitute *speed-for-fidelity* is the same refusal. A slow render at the
selected tier beats a fast render of something the user did not choose. So the downtier's ordering
stands as written, on both the explicit and the Auto path.

Two things this does **not** license:

- **An explicit pick never reaches the downtier at all** and never had to — `explicit_pick`
  (`mlxQuantizeExplicit`) skips it outright, as does an NVFP4 selection, which is unrankable on purpose.
  The rule above is about Auto only.
- **Accepting the cost is not the same as hiding it.** Nothing today tells a caller that its render is
  slow *because* it is streaming blocks to hold the tier, and a render 1-2 orders of magnitude slower
  than the same job at a lower tier is operationally indistinguishable from a hang. Disclosure is
  SC-16104's remaining scope; the policy question it was filed to settle is settled here.

One consequence to carry forward: this **declines** SC-15791's recommendation *"do not enable rung 4 on
q8 until the repack is hoisted"*. Deliberately — gating q8 would be exactly the fidelity substitution the
rule forbids. It does mean the per-window conversion is paid on the tier where it is worst, which raises
the value of SC-16096 rather than lowering it.

### A rung declares no non-VRAM resource cost

SC-15791 also asked how a rung declares a **host RAM** cost, because its recommended Candle fix — cache
the repacked bytes host-side — holds ~3.16 GiB of host memory for the whole request, on exactly the
low-RAM hosts rung 4 exists for, while MLX pays nothing host-side. (That 3.16 GiB is derived arithmetic
for a *proposed* fix, not a measurement; nothing in the spike watched host RSS.) A rung whose resource
cost changes *kind* between backends cannot be expressed as one cost order or one saving figure.

**No such axis is added, because the cost belongs to that one candidate fix rather than to the rung.** A
realization that materializes windows from device-format bytes on disk holds its host copy as the mapped
file — reclaimable page cache, the same *kind* of resource MLX's mmap is — and makes no anonymous
per-request allocation at all. Adding a resource-kind axis for an implementation nobody has to write is
the same mistake as (3) above, one layer down.

**What that leaves undeclared, stated rather than glossed.** The `HostFormatConversion` realization that
ships today *does* have a host cost, and this decision does not surface its size. Its per-window
conversion allocates anonymous host memory per projection and frees it again: the block's unpacked codes
for q4, and for q8 a **full dense f32 grid** (`repack::dequant_mlx_q8_gs`), which for a 3840 × 15360
projection is a few hundred MiB. Those are transients rather than request-held, and they are **not
measured** — SC-15791 says plainly that q8 host memory is unverified. So on a low-RAM host a rung-4
Candle selection today carries an unquantified host transient no gate sees. It is recorded here so it is
not mistaken for zero; SC-16096 both removes the conversion and owes the measurement.

**Revisit trigger — deliberately broad enough to catch that case.** The axis is owed if a realization is
measured to need host memory proportional to the model that a fit gate would have to account for: held
for the request or transient, reclaimable or not, device-format alternative or none. SC-16096 must raise
it rather than absorb it. What is *not* owed is an axis built ahead of that measurement, on a figure
derived for a fix nobody has written.

### What this decision does not settle

The *prerequisite* axis was checked and needs nothing: SC-15791 confirmed rung 4's single edge (requires
rung 1 engaged in the same request) holds on Candle for the same arithmetic reason, so
`MemoryProviderContract::requires` stays unbuilt. Two things the spike left genuinely open are **not**
answered here and are not preconditions for this decision: whether the ~26× survives the fix (SC-16096
measures it), and whether the small-card behaviour rung 4 exists for can be validated at all on the dev
box — SC-15791 over-committed 3.41 GiB into 1.93 GiB of driver-visible free and completed at 1.07× wall
time, so the ceiling is not enforced there and neither completion nor wall time detects a spill
(SC-16091).

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
