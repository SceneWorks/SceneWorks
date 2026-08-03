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
an in-envelope geometry, and an exact ordered engaged composition. The provider contract derives that
composition from its shared prerequisites, capabilities, and provider-specific prerequisite edges;
the selected rung alone is not evidence identity. A missing or malformed composition is `Invalid`,
while a valid measured set that differs from the current contract returns the dedicated
`CompositionMismatch` verdict. Neither can authorize a guessed fit. `Implemented/unverified`,
unknown, stale, fingerprint mismatch, structural N/A, route unavailable, or out-of-envelope evidence
also returns `Unverified`; it never selects an optimized rung. Equality fits
(`needed_gb <= available_gb`). A calibration that needs tolerance must include it in the provider
estimate and golden evidence.

SceneWorks invalidation is owned by the provider calibration ABI fingerprint, not by the exact source
tree. The fallback fingerprint hashes the provider's explicit ABI version, pinned inference revision,
model/provider/backend/tier/mode/overlay/rung identity, runtime strategy parameters, and the
calibration-relevant manifest values for that backend and tier. Captured evidence additionally binds
the exact composition, so a prerequisite change invalidates only rows whose execution set changed.
Bump a provider's entry in
`CALIBRATION_ABI_VERSIONS` when its quantization floors, tensor layout, or execution structure
invalidate measurements; bump the default only for an ecosystem-wide calibration contract change.
`matrixSourceRevision` and `generatedFrom.sceneWorksRevision` carry semantic generated-source
provenance: every hashed source is normalised before hashing, so semantically inert edits produce no
generated artifact churn. The JSONC manifest and the JSON plan and evidence bundles are hashed after
parsing, which makes comments AND formatting inert for them. The Rust, TOML and generator-script
sources are hashed with their blank lines and whole-line comments removed — and only those: a
re-indent, a re-wrapped line or a block comment still rotates the fingerprint, because stripping
them safely needs a tokenizer and noisy-but-safe is the correct side to err on. A whole-line comment
containing `"` also stays hashed, since the generators' parsers read double-quoted literals (see
`scripts/lib/source-revision.mjs`). Exact raw-source history remains available in version control
and must not be used as evidence staleness.

**Provenance is stamped once, not per row (sc-16268).** The revision pair belongs to the document,
so it lives in `generatedFrom` and nowhere else. Cells carry no `evidenceRevision`: it held the same
constant in all ~7,360 rows, which turned every fingerprint rotation into a ~14,700-line rewrite of a
file that can only be regenerated, never hand-merged — so two concurrent PRs touching any
fingerprinted source were guaranteed to conflict. `additionalProperties: false` on a cell keeps it
gone. A generated artifact must not duplicate document-scoped provenance into its rows.

Strategy changes never change precision. A lower precision tier is a separate candidate evaluated
by the existing tier chooser before the memory selector.

## Registration-time behavioural conformance

`MemoryRegistration` exposes both halves of a provider's weights-free admission surface: its
`contract(&LoadSpec)` builder and its `safety_check(&LoadSpec, &MemoryProviderContract,
&MemoryRunContext)` callback. The load specification is passed through without opening weights so a
tier- or load-shape-sensitive provider can execute the same safety logic as its loaded `Generator`.
A registration must point to that production check (directly or through a thin adapter), not a
test-only approximation.

A PiD-capable contract declares `pid_decode_routes`, with separate native and PiD tile domains. The
domains must be non-empty, non-zero, internally unique, mutually disjoint, and their union must equal
the provider's published `BoundedDecode` edges and overlaps. PiD eligibility is therefore explicit in
the backend-neutral contract; `gen-core` does not infer it from an MLX Cargo dependency.

The complete MLX and Candle catalog tests run a shared registry walk over
`ProviderRegistry::memory_strategy_registrations()`. Every registration receives static contract
conformance. For every contract that declares PiD routes, the walk also makes four weights-free,
device-free admission calls: native geometry on the native route and PiD geometry on the PiD route
must be accepted, while native geometry with `use_pid: true` and PiD geometry with `use_pid: false`
must be rejected. The matching-route controls make the proof non-vacuous; always rejecting does not
conform. Cross-route geometry must never be substituted or re-planned.

This behavioural walk complements inference's `check_pid_decode_route_adoption` workspace gate. The
static gate fires on provider-source changes and requires production constructor and admission-call
evidence while excluding comments, literals, and `#[cfg(test)]`-only markers. It cannot prove that a
textually present `DecodeRoutes::validate` call is reached by admission. The registry walk proves that
observable rejection through the real callback; both checks are intentionally load-bearing.

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
realization with the fix it can actually perform. `candle-gen-krea` now declares
`DeviceFormatTransfer`: content-addressed sidecars hold the GGML bytes the accelerator consumes, so
each rung-4 window maps and transfers those bytes without repeating the packed-weight conversion.

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
resident, and the cost of that rung is what the higher tier is bought with. After SC-16096 hoisted the
packed repack, Krea's real packed q8 block-materialization median on an RTX PRO 6000 Blackwell is
101.7 ms. Krea Turbo is CFG-free and walks its fixed 30-block transformer once per denoise step, so
the measured materialization component is about 3.05 seconds per step. That is still additive
streaming work, but no longer the pre-fix ~1.3–1.5 seconds per block.

**That is intended, and is the same rule the rest of this epic applies.** Fidelity the user asked for is
not a memory lever: tier integrity refuses to substitute precision, SC-15807 refuses to substitute
geometry, and refusing to substitute *speed-for-fidelity* is the same refusal. A slow render at the
selected tier beats a fast render of something the user did not choose. So the downtier's ordering
stands as written, on both the explicit and the Auto path.

Two things this does **not** license:

- **An explicit pick never reaches the downtier at all** and never had to — `explicit_pick`
  (`mlxQuantizeExplicit`) skips it outright, as does an NVFP4 selection, which is unrankable on purpose.
  The rule above is about Auto only.
- **Accepting the cost is not the same as hiding it.** SC-16104 surfaces a compact progress note for
  Auto jobs that engage rung 4, naming that transformer blocks are streaming to hold the selected
  tier. The worker also emits a structured `image_memory_strategy_selected` event and a tracing event
  with the cause and SC-16096 measurement. This deliberately uses the existing inline progress/log
  surfaces — no dialog, toast, or large UI warning — while leaving the fidelity-first selection and
  explicit-pick bypass unchanged.

One historical consequence: this **declined** SC-15791's recommendation *"do not enable rung 4 on q8
until the repack is hoisted"*. Deliberately — gating q8 would have been exactly the fidelity substitution
the rule forbids. SC-16096 has now removed that per-window conversion, so the temporary cost that made
the recommendation attractive is no longer paid.

### A rung declares no non-VRAM resource cost

SC-15791 also asked how a rung declares a **host RAM** cost, because one candidate Candle fix — retain
all repacked bytes in anonymous host memory — would have held model-scale memory for the whole request,
on exactly the low-RAM hosts rung 4 exists for, while MLX pays nothing host-side. A rung whose resource
cost changes *kind* between backends cannot be expressed as one cost order or one saving figure.

**No such axis is added, because the cost belongs to that one candidate fix rather than to the rung.** A
realization that materializes windows from device-format bytes on disk holds its host copy as the mapped
file — reclaimable page cache, the same *kind* of resource MLX's mmap is — and requires no
model-proportional anonymous host copy. Adding a resource-kind axis for an implementation nobody has to
write is the same mistake as (3) above, one layer down.

**The implemented and measured result.** SC-16096 stores content-addressed q4/q8 device-format sidecars
on disk and maps one projection at a time. The complete sidecar sets are 3,668.0 MiB for q4 and
6,235.6 MiB for q8, but they are file-backed cache, not anonymous request-held memory; the largest
per-projection mappings measured 23.4 MiB and 39.8 MiB. In steady-state rung-4 windows, host
working-set/private-commit peak deltas were 23.1/128.3 MiB for q4 and 39.4/192.5 MiB for q8.

First cache creation is intentionally separate from the window path. It streams the source mapping and
builds one projection at a time; measured working-set/private-commit deltas were 3,439.9/155.5 MiB for
q4 and 6,507.0/368.0 MiB for q8. The large working-set leg is file-backed and reclaimable, while the
q8 dense-f32 private transient is bounded by one projection and is never repeated per window.

**Revisit trigger.** A host-memory axis is still owed if a future realization needs model-proportional
host memory that a fit gate must reserve or account for: held for the request or transient, reclaimable
or not, device-format alternative or none. SC-16096 does not meet that trigger in steady state: its
model-scale bytes are file-backed and reclaimable, and its anonymous transient is projection-bounded
and hoisted out of rung-4 windows.

### What this decision does not settle

The *prerequisite* axis was checked and needs nothing: SC-15791 confirmed rung 4's single edge (requires
rung 1 engaged in the same request) holds on Candle for the same arithmetic reason, so
`MemoryProviderContract::requires` stays unbuilt. SC-16096 has now answered the performance question:
q4 block materialization fell from a 247.8 ms median to 56.2 ms, and q8 from 1,322.6 ms to 101.7 ms,
with bit-exact device bytes and CUDA outputs across all eight sampled packed projections. The remaining
open question is whether the small-card behaviour rung 4 exists for can be validated at all on the dev
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
   `gen_core::mempolicy` and the former provider-local branch-quantization step were deleted rather
   than retained as a second policy.
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
   activation. Mage-Flow is the convergence case: both providers now advertise and apply the same q8
   text-encoder-layer and transformer-head floors on q4. It follows that `residentTier` is a per-lane quantity, that the
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

### Lens MXFP4 resolution

SC-16014 chose the **re-hosted dtype** option for the Lens and Lens-Turbo q4/q8 turnkeys. The upstream
gpt-oss encoder identifies its expert matrices as MXFP4, which MLX cannot multiply directly, but the
shipped `SceneWorks/lens[-turbo]-mlx` q4 and q8 artifacts already contain offline-converted MLX affine
packs. Their provider footprint is therefore disk-derived and their experts run through the packed
`gather_qmm` path; they are not a bf16 residency exception. The existing encoder parity gates bound the
conversion's quality cost: Q8 must remain above 0.995 cosine to the bf16 golden and Q4 above 0.95.

The other two options remain deliberately unselected. A native MXFP4 compute path would retain the
upstream artifact and its source precision without a conversion/re-hosting pipeline, but adds a new MLX
kernel, dispatch surface, and maintenance burden after the shipped tiers have already removed the
production gap. Rung-4 block materialization would reduce simultaneous weight residency, but repeatedly
materializes blocks and adds transfer/dispatch latency; more importantly, it would still run an
above-tier encoder and cannot make that configuration tier-integral.

The bf16 turnkey is a separate fit-gate case. Its source still carries MXFP4 experts and materializes
them as bf16, so the gate retains the measured 17.24 GiB weight expansion and 29.88 GiB activation
transient. A genuinely bf16-on-disk Lens encoder receives the same architecture-specific activation
headroom but no invented weight expansion. The provider distinguishes packed affine, MXFP4, explicit
bf16, and unknown storage; unknown remains conservative. The shared
`sc-16014-resolution: rehosted-q4-q8` marker binds this decision to the ledger checker, which rejects
both a stale Lens text-encoder exception and one-sided edits to the fit-gate contract.

Three mechanisms are load-bearing today. `mlx.denseTextEncoderTier` is the only way an entry obtains a
dense text encoder on a packed tier, and setting it requires a matching ledger row — the hardcoded
worker registry it used to mirror is deleted, because ids living in Rust while the catalog declared
nothing is exactly the invisible carve-out this rule exists to remove. `candle.control.branchTierByBaseTier`
declares the Krea pose-control branch's tier per base tier: q8 follows the tier, bf16 is already at
tier, and **q4 floors its branch at q8** — the one declared, measured exception, because a q4 control
residual measures "pose-locked; non-pose details drift" and the residual is the thing the user asked
for. `precisionFloors` is the backend-neutral component map for provider-local substitutions such as
Mage-Flow's q4 → q8 LM layers and transformer head. The active provider descriptor must match the
manifest map; the worker uses it in effective asset labels and `MemoryNumericTier`, so a mixed q4/q8
run cannot share a plain-q4 evidence identity. The checker separately requires matching measured
ledger rows for every hosted backend. Those floors are the rule working, not holes in it.

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
  true band is slightly narrower. The direct SC-16013 measurements intentionally do not restore that
  below-tier branch choice.)

**Repacking invalidates measurements.** A component's resident precision is an input to every peak
measured against it, so changing it makes those peaks stale even though every provenance field still
looks valid. Evidence captured before a repack must be renamed or fingerprint-bumped so it cannot read
as green. SC-15799 withdrew the old Krea control-lane rows and set
`candle.control.measured: false` for exactly this reason; SC-16013 then replaced them with direct
measurements against every packed branch configuration that ships and restored `measured: true`.
**Tier integrity therefore precedes calibration**: calibrating first means paying twice.

**And renaming is not enough on its own — the reader has to honour the flag.** A superseded row may be
read as an **upper bound** and nothing more. It may not be "corrected" into the shipping configuration by
subtracting some other number, because a correction is a new measurement wearing a stale one's clothes;
SC-15799's first revision subtracted `branchPackSaveGb` (8.4 GB, against a control branch that is 6.6 GB
in total) and under-predicted a live CUDA admission path by ~5 GB one-directionally toward an OOM. It may
not produce a hard reject, since an upper bound cannot rule a job out. And it may not be recorded as a
reclaimable high-water, since over-stating a pool lets the next gate over-admit. `measured: false` is a
field the code reads, not a note to a future human.

SC-16013 collapses those stale-evidence bands to zero. On an RTX PRO 6000 Blackwell at 1024², 8 steps,
and control scale 0.6, the current direct resident/staged peaks are q4 33.5/29.6 GB, q8 41.9/36.1 GB,
bf16 65.8/50.1 GB, and INT8-ConvRot 50.7/35.5 GB. The q4 staged decode-tiling saving is 7.2 GB and the
additional attention-chunking saving is 2.8 GB. The worker reads those rows verbatim, adds only standard
admission headroom, and can hard-reject below the deepest measured rung.

## Overlay coverage: declared unmeasured, not quietly absent (SC-16069)

An **overlay** is a second network held resident beside the base for one render — a strict-pose
ControlNet branch, an IP-Adapter, a face-identity encoder. Krea strict-pose is the measured exception;
the remaining unmeasured cells are recorded decisions rather than gaps you have to notice.

The state of the evidence, stated plainly:

- Of the memory-matrix cells with `overlay != "none"`, **zero** carry any historical, current-environment
  or strategy-parameter verification.
- The `overlay` harness scenario is one of only two the calibration validator lets off `passed` (the other
  being `not_run` on a gated record), which makes it the one place a stock excuse can silently read as
  coverage. `bin/candle.rs` therefore **derives** its verdict from `planned.target.overlay` and **refuses**
  a non-`none` target outright, instead of emitting the fixed "ordinary Krea Turbo text-to-image
  calibration has no overlay" it used to emit on every run.
- No adapter can execute an overlay render: doing so needs the control-branch load path and a control-map
  fixture. The decision, and the exact thing that would unblock it, live in
  `config/memory-calibration-plan.json` → `overlayCoverage`.

**What this costs, stated plainly.** The candle conditioning admission gate
(`crates/sceneworks-worker/src/conditioning_fit.rs`) has no measured overlay peak to gate on, so it gates
on an on-disk weights **FLOOR**: base + overlay bytes + headroom, compared against live free VRAM. That
direction is the safe one for admission — it cannot refuse a host that can hold the weights, which is the
never-block-without-evidence posture this contract requires — but it prices no activations, no denoise
steady state and no decode spike, so a **marginal** conditioning render still reaches the reactive CUDA
OOM. Inflating the floor with an invented activation factor would manufacture rejections no measurement
supports, so the floor stays a floor and says so in its own rejection message ("at least ~N GB … that is
the weights alone"). The principled gate — the overlay's own bytes and transients inside the predicted
peak — needs both contract vocabulary for a second resident network and at least one measured overlay
cell.

An unpriced **tier** on an otherwise-measured overlay lane is a distinct hole and is no longer silent: the
Krea control ladder returns an explicit, logged `Unverified` for it (stages residency, never rejects)
rather than collapsing into the no-signal `Unknown` that took the zero-adaptation path. SC-16013 now
prices every hosted Krea tier, including INT8-ConvRot, so this verdict is a fail-safe for malformed
catalog entries and future tiers rather than a known shipping hole.

## Five-rung load reuse (SC-16059, oracle from SC-16402)

`config/memory-calibration-plan.json` contains two same-target reference ladders used for the
fresh/reused decision:

- MLX/Metal: `z_image_turbo`, q4, 768×768 text-to-image, no overlay, seed 16402, two steps. All five
  rungs use the provider's shape-independent content fingerprint. Rungs 0–3 require the typed
  `eager_materialization` load shape and rung 4 requires `deferred_materialization`; the load-shape
  receipt, not a suffix on the fingerprint, keeps those calibration populations distinct.
- Candle/CUDA: `krea_2_turbo`, q4, 1024×1024 text-to-image, no overlay, seed 16402, two steps. The
  provider-owned compositions for rungs 2–4 include staged residency because those controls execute
  through Krea's physical three-stage loader; the plain staged rung uses its one-boundary residency
  path.

The harness preserves a forced fresh-per-case oracle for both backends. Candle can execute its five
cases as one diagnostic `run_batch` invocation; the adapter loads Krea once, returns five ordinary
record fragments, and attests `modelLoads: 1`. A comparison passes only when every phase's active,
allocator, device, wired, and reclaimable metric is within the larger of 256 MiB and 5% of its fresh
value. The authoritative CUDA comparison returned `unable_to_amortize`, so the shipped plan keeps
Candle fresh-per-case rather than changing the recorded peaks. MLX also remains fresh-per-case:
rungs 0–3 require an eager load while rung 4 requires a deferred load, so one loaded Z-Image
generator cannot preserve all five typed load-shape identities even though their content fingerprint
is the same. `assess-reuse` reads both identities from the pinned provider contracts for the exact
eager and deferred load specs and rejects any plan mismatch; it does not trust the plan's strings as
capability evidence. The structured verdict records this backend as
`unable_to_amortize`; it is not treated as a failed measurement or silently averaged.

Two denoise steps are load-bearing for phase measurement: providers with an explicit loading boundary
close conditioning there, while resident providers use the first Step callback as a conservative
conditioning envelope and measure the second step as a denoise-only interval. Decoding always starts a
separate measured interval; no phase value is synthesized from another run or a static estimate.
Run only from clean checkouts at the exact revisions being recorded:

```bash
# Apple Silicon / Metal
SCENEWORKS_Z_IMAGE_REPOSITORY=SceneWorks/z-image-turbo-mlx \
SCENEWORKS_Z_IMAGE_REVISION=bb2bc9893b3c49ae96c813350775f791a2e8bc80 \
SCENEWORKS_Z_IMAGE_ROOT=/absolute/path/to/models--SceneWorks--z-image-turbo-mlx/snapshots/bb2bc9893b3c49ae96c813350775f791a2e8bc80/q4 \
cargo build --release --locked -p sceneworks-memory-adapter --features mlx --bin memory-mlx-adapter
node scripts/memory-calibration-harness.mjs run \
  --config config/memory-calibration-plan.json --backend mlx \
  --fixture fresh-five-rung-z-image-q4-768-seed16402-step2 \
  --fresh-per-case \
  --provider-command '["target/release/memory-mlx-adapter"]' \
  --sceneworks-repo "$PWD" --inference-repo /absolute/path/to/inference-pin \
  --output /tmp/sc-16059-mlx-fresh.json
node scripts/memory-calibration-harness.mjs assess-reuse \
  --config config/memory-calibration-plan.json --backend mlx \
  --fixture fresh-five-rung-z-image-q4-768-seed16402-step2 \
  --provider-command '["target/release/memory-mlx-adapter"]' \
  --output /tmp/sc-16059-mlx-reuse-assessment.json

# Windows / NVIDIA CUDA (PowerShell; use target\release\memory-candle-adapter.exe)
$env:SCENEWORKS_KREA_REPOSITORY='SceneWorks/krea-2-turbo-mlx'
$env:SCENEWORKS_KREA_REVISION='d009674080cc1bccf2b629d834c34bf5eccdb723'
$env:SCENEWORKS_KREA_ROOT='C:\absolute\path\to\models--SceneWorks--krea-2-turbo-mlx\snapshots\d009674080cc1bccf2b629d834c34bf5eccdb723\q4'
cargo build --release --locked -p sceneworks-memory-adapter --features candle --bin memory-candle-adapter
node scripts/memory-calibration-harness.mjs run --config config/memory-calibration-plan.json --backend candle `
  --fixture fresh-five-rung-krea-q4-1024-seed16402-step2 `
  --fresh-per-case `
  --provider-command '["target/release/memory-candle-adapter.exe"]' `
  --sceneworks-repo $PWD --inference-repo C:\absolute\path\to\inference-pin `
  --output $env:TEMP\sc-16059-candle-fresh.json
node scripts/memory-calibration-harness.mjs run --config config/memory-calibration-plan.json --backend candle `
  --fixture fresh-five-rung-krea-q4-1024-seed16402-step2 --batch-rungs `
  --provider-command '["target/release/memory-candle-adapter.exe"]' `
  --sceneworks-repo $PWD --inference-repo C:\absolute\path\to\inference-pin `
  --output $env:TEMP\sc-16059-candle-reused.json
node scripts/memory-calibration-harness.mjs compare-reuse `
  --fresh $env:TEMP\sc-16059-candle-fresh.json `
  --reused $env:TEMP\sc-16059-candle-reused.json `
  --output $env:TEMP\sc-16059-candle-reuse-comparison.json
```

Validate either capture with `node scripts/memory-calibration-harness.mjs check --input <file>`.
These authoritative records deliberately remain `gated`: they contain exact strategy identity and
observed conditioning/denoise/decode/overall memory for the reuse oracle, but do not pretend that a
single reference render also completed the promotion-quality sweep, negative mutation, or lifecycle
fault suite. They therefore cannot become current calibration evidence or update the cost model.

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
