# Review log

This log records the disposition of every review finding against the Shared Generation Architecture documents. Dispositioned findings are marked **Accepted** (the docs were changed), **Modified** (accepted in a different form), or **Rejected** (with the reason). New findings are **Open** until the author records a disposition; a review does not approve or amend the requirements.

## Round 0: internal adversarial reviews of draft 0.1 → draft 0.2 (2026-11-03)

Two independent reviewers read draft 0.1 and checked its claims against the code. Their full reports are in [evidence/reviews/](evidence/reviews/).

The dispositions below record the draft-0.2 changes as they happened. Later rounds supersede them where noted; in particular R2-06 replaces ENG-17's unchanged-admission rationale, and R2-08 splits the original ENG-4 roadmap feature into ENG-4a/4b.

- **Engine side:** 26 findings: 5 BLOCKER, 16 MAJOR, 5 MINOR. ([round0-engine.md](evidence/reviews/round0-engine.md))
- **SceneWorks and process side:** 43 findings: 7 BLOCKER, 28 MAJOR, 8 MINOR. ([round0-app-process.md](evidence/reviews/round0-app-process.md))

All 12 blockers were accepted. No finding was rejected outright. Where a finding is marked Modified, the reason is given.

### Engine findings

| # | Sev | Finding | Disposition |
|---|---|---|---|
| ENG-1 | BLOCKER | sc-20414 is not a phase-runner base: fresh-per-pass only, always re-noises, no geometry, host-owned combine, refuses Krea phases, adopted only for `denoisePasses`. The port is near a rewrite. | **Accepted.** REQ §1.3 corrected. DEC-4 now says "salvage invariants, testkit and pass-schedule math; do not port the wire contract". The salvage review is in PLAN §2 and ENG-1 item 0. Krea moved out of ENG-1. |
| ENG-2 | BLOCKER | "One solver per phase" breaks Wan MoE, which carries history and picks the expert per σ evaluation. | **Accepted.** PIPE-3 adds a `Carry`/`Reset` history policy and σ-keyed model-slot rules. The runner invariant is now "history never crosses a geometry change or re-noise". Tier 0 adds a Wan trace-equality test. |
| ENG-3 | BLOCKER | Unified seeds change every count>1 image and the persisted SceneWorks seed relation. | **Accepted.** PIPE-11 makes the `SeedScheme` route-declared, keeps legacy schemes bit-exact, uses splitmix only for new streams, and exports the image-index → seed relation. AP12 is extended. |
| ENG-4 | BLOCKER | A per-branch forward with a runner-owned combine cannot reproduce batched CFG or the family formulas. | **Accepted.** `predict(BranchSet)` lets families batch. Bit-exact `CombineVariant`s are added. Each route declares its row order. Branches can carry model perturbations (STG). Unifying them is an optional class (c) decision. |
| ENG-5 | BLOCKER | ScaleContext arrives after Krea, LTX and Wan, which need it. | **Accepted.** ENG-4 (Linear spine + request-scoped scales) now precedes all per-phase adapter re-expressions. Krea is removed from ENG-1. |
| ENG-6 | MAJOR | The stage model is single-stream; AV needs bundles, per-stream schedules and stream-targeted transitions. | **Accepted.** PIPE-5 adds `LatentBundle`/streams. `StepInfo` gets per-stream σ. `TimeInput` gets per-token, row and flow-map variants. Transitions get stream sets. |
| ENG-7 | MAJOR | Two schedule modes don't cover LTX, and the shared re-noise isn't bit-exact with LTX's. | **Accepted.** A third span kind, `Explicit` table, is added, along with route-supplied bit-exact transitions. DEC-3 now lists three span kinds. |
| ENG-8 | MAJOR | The tolerance policy is under-specified: one ULP becomes a visible change, tiny weights miss quantized kernels, there is no Tier 1 reference, and "allocation order" can't be tested. | **Accepted.** New AP13 defines classes (a) bit-exact, (b) component-tolerance and (c) accepted re-baseline with sign-off. The Tier 1 protocol is now same-runner base vs head on shape-faithful weights. AP3 says "peak memory" instead of "allocation order". Expected classes per feature are in PLAN §7.3. |
| ENG-9 | MAJOR | "Candle defaults to residual" is a self-inflicted regression. | **Accepted.** The install policy is per route and today's policy stays the default. A route switches to residual only for a Feature that needs runtime scales (WGT-3, PLAN §3.8). |
| ENG-10 | MAJOR | ScaleContext must be request-scoped, not state on a shared module. | **Accepted.** `ForwardArgs.adapter_scales`. New PIPE-14 states the concurrency rule, with a Tier 0 test. |
| ENG-11 | MAJOR | The Feature trait isn't object-safe, and tensor hooks can't live in gen-core. | **Accepted.** The trait is split into `FeatureSpec` (dyn, tensor-free) and per-backend `FeatureHooks<B>`. G1 is qualified. |
| ENG-12 | MAJOR | Memory relocation overclaims "families supply nothing", omits iOS, and must drive `MemoryRequestScope` until relocation. | **Accepted.** MEM-1 now lists the family declarations (lifetimes, tiling law, block layout, quanta). New MEM-5 makes the runner drive `MemoryRequestScope` until relocation. New MEM-6 covers iOS admission. MEM-3 adds reference pixel geometry. |
| ENG-13 | MAJOR | MEM-4 "admission never changes output" is false (tiled decode). | **Accepted.** The Equivalent class now means "within a declared, recorded quality bound". Admission never selects a recipe Feature and never changes tier. |
| ENG-14 | MAJOR | Hires.fix needs a pixel round trip. | **Accepted.** `Transition::PixelRoundTrip` is added, with VAE residency priced across phases. Hires.fix is kept in HOOK-2. |
| ENG-15 | MAJOR | Escape-hatch and family lists were incomplete: Krea-realtime, MOSS-Image, Supra2, Iris, SeedVR2, PiD, InstantID and PuLID. | **Accepted.** New PIPE-18 adds a route inventory generated from the catalogs. The PIPE-7 list is extended. All families are placed in ENG-1…6. |
| ENG-16 | MAJOR | Low-res coverage on SDXL and Wan native depends on unverified `DriverSupport`. | **Accepted.** PIPE-7: drivers are generic over a forward seam and conformance-tested, and must honour `SliceInput`. Refusals need approval (DEC-13). FEAT-1 lists the expected routes. |
| ENG-17 | MAJOR | ENG-1 was too big, contradicted the all-routes rule, and had an admission adapter that crossed features. | **Accepted** *(superseded by R2-06 and C4-01: admission is no longer unchanged, and ENG-1 is now split into ENG-1a and ENG-1b)*. ENG-1 is now the runner plus all FLUX.1 routes plus low-res start. Admission is unchanged, because the maximum phase is the final geometry. The cross-feature adapter is removed. |
| ENG-18 | MAJOR | ENG-2/ENG-4 overlapped and exceeded the 6-week window. | **Accepted.** ENG-2…6 are split by mechanism. Kit work is sequenced after migration waves. File-ownership tables are required per feature (PLAN §7.1). |
| ENG-19 | MAJOR | Engine requirements were missing: count/batch, concurrency, load cancellation, error taxonomy, progress units, streaming, telemetry, per-component tiers, a cross-backend non-goal. | **Accepted.** New PIPE-13…17, WGT-5, WGT-1 per-component tiers, DEC-19, and a non-goal on cross-backend identity. |
| ENG-20 | MAJOR | COMP-1 ignored the existing llm tap API and named the wrong blockers. | **Accepted.** COMP-1 is rewritten on the existing API, with named blockers (causal masks, all-layers return, quant codec). gpt-oss is dropped. Encoder migrations are class (b)/(c). |
| ENG-21 | MAJOR | "Derived, never declared" overclaims. | **Accepted.** AP5 and G3 now read "one primary declaration per fact, everything else derived". Behavioural declarations get conformance checks. |
| ENG-22 | MINOR | Monomorphization and dispatch shape were unspecified. | **Accepted.** PLAN §3.3 uses dyn at family granularity and erases the step context. The iOS size baseline is in ENG-1 item 0. |
| ENG-23 | MINOR | Step-context cache eviction and transition memory were unspecified. | **Accepted.** PIPE-5 adds an eviction rule. Memory effects include transition residency. |
| ENG-24 | MINOR | `Learned(UpsamplerId)` hides a component. | **Accepted.** It is now `Learned(ComponentId)` with a declared normalization. |
| ENG-25 | MINOR | Factual errors: `MemoryRequestScope` path, guidance users, sc-20414 adoption, the vendored-copy count, 139 fixtures, [E §3.6]'s cross-backend parity claim. | **Accepted.** All corrected. The cross-backend claim is contradicted by the new non-goal. |
| ENG-26 | MINOR | A generated hook map conflicts with AP11. | **Accepted.** The hook enum is the source; any listing is a typed dump with no checked-in matrix. |

### SceneWorks and process findings

| # | Sev | Finding | Disposition |
|---|---|---|---|
| APP-1 | BLOCKER | "The engine track starts now" isn't engine-only; it collides with SW-F327 PRs. | **Accepted.** Every item is tagged `[INF]` or `[SW]`. DEC-5 gates every `[SW]` item on the named SW-F327 PRs merging. |
| APP-2 | BLOCKER | SW-F327 already delivers parts of SGA. | **Accepted.** Overlap ledger added (PLAN §7.5) and MIG-7 added. #3268's catalog deletion is removed from SGA scope. Baselines are taken after merge. |
| APP-3 | MAJOR | The plan's own bridges violated MIG-2/AP9. | **Accepted.** AP9 is rewritten: the producer migrates its consumers, and there are no interim adapters. Every listed bridge is removed or re-sequenced. A bridge register was rejected, per the standing rule that such registers are laundered deferments. |
| APP-4 | BLOCKER | Hidden APP→ENG dependencies; the APP calendar was not credible. | **Accepted.** PLAN §9.3/§9.4 is redrawn with real edges (APP-4 after ENG-6, APP-5 after ENG-7, APP-7 after ENG-9). The calendar is recomputed. |
| APP-5 | MAJOR | APP features over 6 weeks were not split. | **Accepted.** Panels are now their own feature (APP-3). The lanes are split into APP-5 and APP-6. |
| APP-6 | MAJOR | Pin choreography between concurrent features was missing. | **Accepted.** PLAN §7.4 and MIG-7. |
| APP-7 | BLOCKER | The "18 paths" admission inventory is incomplete. | **Accepted.** ADM-1 acceptance is now a grep over probe and refusal patterns. The scope lists ≥25 paths. APP-4 item 0 re-inventories. |
| APP-8 | MAJOR | MLX "RefuseIfUnpriced" would regress routes that admit unpriced today. | **Accepted.** Failure posture is per (backend × evidence basis), per route (ADM-2). |
| APP-9 | MAJOR | Admission behaviour changes had no acceptance bound. | **Accepted.** New ADM-7: zero unapproved newly refused cells, with measured consequences. |
| APP-10 | MINOR | The cudarc probe moves driver-init failures to admission time. | **Accepted.** New ADM-8: probe failure is `BudgetUnavailable`, not a refusal, and its cost is stated. |
| APP-11 | BLOCKER | The registry can't hold runtime data (scans, imports, overlays, projections). | **Accepted.** APPREG-1 adds a static layer plus a runtime layer, with fixtures for imported, external and trained models. |
| APP-12 | MAJOR | The job carrier breaks for non-builtins and multi-worker deployments. | **Modified.** Jobs carry a resolved typed `ModelSpec` snapshot (APPREG-6) rather than a revision reference. That removes the skew class entirely. |
| APP-13 | MAJOR | Export vs live engine: two authorities. | **Accepted.** REG-3: the worker's linked engine is authoritative at runtime and refuses with a typed reason. The export drives UI availability. A Tier 0 export-equality check is added. |
| APP-14 | MAJOR | Capability dumps are campaign artifacts produced off-Mac. | **Accepted.** REG-3 requires a hardware-free build product. ENG-7 item 0 verifies the hosted candle build. Risk R13 is the fallback. |
| APP-15 | MAJOR | "6–8 hand tables" understates the problem. | **Accepted.** APPREG-2 uses an enumerated checked-in inventory classified derive/keep/delete. The baseline is now about 20 named plus about 230 const tables. |
| APP-16 | MAJOR | The manifest-memory move had no consumer migration. | **Accepted.** New APPREG-5: manifest v2, user jsonc upgrader, seeding, and consumers migrated in the same feature. |
| APP-17 | MAJOR | Evidence keys would be re-keyed. | **Accepted.** APPREG-4 keeps identity keys verbatim; no recapture. |
| APP-18 | MAJOR | The `ADVANCED_KEY_RULES` seed is incomplete. | **Accepted.** OPT-1 seeds from the union of all sources. Acceptance: every `advanced.*` read has an `OptionSpec`. |
| APP-19 | MAJOR | Structured renderers, Context intent and side effects were missing. | **Accepted.** OPT-1/OPT-5 now enumerate renderers and add `IntentClass::Context` and `effects`. |
| APP-20 | MAJOR | DEC-9 ignored RunPod latency. | **Modified.** The recommendation is now a wasm-compiled core validator (one implementation, instant, offline), with the hybrid approach as fallback. |
| APP-21 | MAJOR | OPT-6 missed persisted and external shapes and forward compatibility. | **Accepted.** OPT-6 now covers film plans, training plans, HTTP/MCP DTOs and SceneWorks-Art, plus forward and downgrade rules. The corpus comes from git history. |
| APP-22 | MINOR | OPT-7 conflicts with film compile. | **Accepted.** Film is explicitly in scope. |
| APP-23 | MAJOR | `JobType` classification was missing. | **Accepted.** LANE-1 adds a checked-in `JobType` table (APP-5 item 0). |
| APP-24 | MAJOR | Registering bespoke providers had no owner. | **Accepted.** New REG-6, in ENG-7. |
| APP-25 | MAJOR | API→worker coupling had no end state. | **Accepted.** New LANE-6, coordinated with SW-6824. |
| APP-26 | BLOCKER | The backend-capability matrix gate was omitted. | **Accepted.** TOOL-1 names it. DEC-16 deletes it in APP-1 item 0. |
| APP-27 | MAJOR | Memory-adapter scope and proof were understated. | **Accepted.** TOOL-2 lists every binary and its workflow consumers. DEC-18 covers the proof run. |
| APP-28 | MINOR | "The memory matrix, if still wanted" hid a decision. | **Accepted.** It is now DEC-17. |
| APP-29 | MAJOR | DEC-6 evidence was incomplete. | **Accepted.** DEC-6 now cites the 08-15 ban, the 10-11 scoping to memory and measurement gates, SW-F327 E3 and #3270, and the 10-23 structural-check request. It reuses `architecture_tests.rs`, with a grep fallback if guards are rejected. |
| APP-30 | MAJOR | The Tier 1 workflow mismatch and per-story matrix conflict. | **Accepted.** MIG-3 and PLAN §8.2 add a workflow change (route `nax-macos-2` via an `rw-*` label) and a same-runner protocol. DEC-14 is an explicit exception, with its cost. |
| APP-31 | MAJOR | Candle tests can't run in feature CI. | **Accepted.** The admission and routing core is cfg-free in core. One `windows-candle.yml` dispatch is budgeted per batch (§9.7). |
| APP-32 | MAJOR | Tier 2 for refactor-only features vs SW-F327 E7; PERF compared against a stale baseline. | **Accepted.** New DEC-15. PERF-1/2 are A/B on the same runner in the same window. |
| APP-33 | MINOR | Runner facts were stated as fixed. | **Accepted.** §9.5 now says to discover the pool at planning; a snapshot is included. |
| APP-34 | BLOCKER | PAT-1 contradicts ENG-1. | **Accepted.** PAT-1 is redefined as a new Feature added after APP-2 (PLAN §6.4). FEAT-1 has its own acceptance. |
| APP-35 | MAJOR | The PAT-4 method was undefined. | **Accepted.** PLAN §10.2 gives a 26-fact list, defines an encoding, and lists exceptions. |
| APP-36 | MAJOR | The PAT-8 method was undefined. | **Accepted.** PLAN §10.3 specifies the control-inventory tool and its behaviour fields. |
| APP-37 | MAJOR | PAT-10 became a gate, and the SceneWorks target was unsupported. | **Accepted.** PAT-10 now just records the census. The SceneWorks target is re-derived bottom-up as −15% to −25%. |
| APP-38 | MAJOR | PAT-3 omitted non-code inputs. | **Accepted.** PAT-3 lists measured anchors as data. Other inputs must be derived, or listed by decision. |
| APP-39 | MINOR | Inconsistent model-id rules. | **Accepted.** One rule applies (conditionals only), with an enumerated exemption list (PAT-5). |
| APP-40 | MAJOR | External consumers were absent. | **Accepted.** New XL-4 covers ChatWorks, SceneWorks-mobile and SceneWorks-Art. |
| APP-41 | MINOR | Refusal UX, telemetry and the stamp key. | **Accepted.** New OPT-8; ADM-5 keeps the stamp key name. |
| APP-42 | MINOR | Accessibility and the Simple studios. | **Accepted.** OPT-5 scope and acceptance cover both. |
| APP-43 | MINOR | Internal inconsistencies (tiers, counts, DEC-5 trigger, numbering, authority, Hires.fix owner, labels, pin sites). | **Accepted.** Fixes: tiers named in REQ; grep patterns instead of frozen counts; DEC-5 names its PRs; DECs renumbered in order; REG-3 settles authority; Hires.fix moved to HOOK-2 (engine work in ENG-2's Hires.fix route via PixelRoundTrip); §8 labels corrected; [H] noted as superseding [G] on pin sites. |

## Round 1: Michael (pending)

## Round 2: Codex architecture review of draft 0.2 (2026-11-03)

Reviewed design commit `3bc17e3a0587b0c2aff069f6bab384eca9eef047`. Full findings, source evidence, proposed resolutions and closure checks: [round2-codex.md](evidence/reviews/round2-codex.md).

**Verdict: sound direction, but revise before implementation approval. Confidence: high.** Shared loop ownership, tensor-free policy, backend execution hooks, structural applicability and incremental route adoption address the stated problem. The remaining gaps concern the contracts those shared features need, rather than the desirability of sharing the pipeline. The backend-generic model rollout remains appropriately conditional on pilots; confidence in that end state is medium.

**3 BLOCKER, 6 MAJOR findings at review; all nine Accepted in draft 0.3.** Michael requested the design corrections and a subsequent re-review. Acceptance here means the documents now contain the resolution, not that native behavior has been implemented or validated, or that open product/process decisions are approved. These severities describe design approval risk, not defects in shipped code. Round-0 changes were considered; R2-03 and R2-07 identified remaining gaps in its history and Hires.fix resolutions.

| # | Sev | Finding | Disposition |
|---|---|---|---|
| R2-01 | BLOCKER | Low-res start does not specify whether the transition consumes noisy latents or an x0 estimate, how that estimate is obtained, or which noise-scaling law applies. | **Accepted.** PIPE-4/FEAT-1 and PLAN §3.1 define typed boundary state, a read-only endpoint evaluation with provenance, explicit casts/resample, flow versus VE/EDM laws, and accounting for the extra evaluation. §6.5 is the design trace; ENG-1 owns implementation/closure checks. |
| R2-02 | BLOCKER | Merging boundaries has no coordinate or ownership semantics across sliced schedules, fresh passes and explicit multi-stream tables. | **Accepted.** PIPE-1/2/4, HOOK-3 and PLAN §3.1 give stable pass domains, executed-step spans, stream coupling, transition/schedule/history ownership, full-active-set cycle detection and validation after rewrites. §6.6/6.8 cover composition. |
| R2-03 | MAJOR | `Carry` lacks an execution-state lifecycle, solver compatibility rules and stochastic draw-index continuity. | **Accepted.** PIPE-3/11 and PLAN §3.3 give request-owned solver/driver state, previous intervals, compatibility checks and RNG coordinates separate from history/causal caches. §6.7/6.8 and Tier 0 split conformance cover closure. |
| R2-04 | MAJOR | The context key and prepare API omit adapter scales and guidance branch structure; Krea explicitly prepares adapter-dependent conditioning today. | **Accepted.** PIPE-5 and PLAN §3.3 add preparation arguments and declared dependency keys for effective weights, scales, conditioning and branch layout, with request isolation and priced retention. Krea base/adapter/base and CFG-off/on cases are in §6.7; ENG-4b owns adoption. |
| R2-05 | BLOCKER | Maximum-over-phases-and-streams pricing undercounts simultaneously live AV buffers and joint workspaces. | **Accepted.** MEM-3 and PLAN §3.9 define per-budget-domain maximum simultaneous distinct-allocation totals, with shared/aliased buffers deduplicated and joint/transition workspace included. §6.8 gives explicit synthetic arithmetic. |
| R2-06 | MAJOR | ENG-1 assumes unchanged admission while adding transition allocations and new phase identities; plan-aware admission arrives only after ENG-6. | **Accepted.** MEM-5/ADM-3, PLAN §3.9/6.1 and ENG-1/2 require conservative plan inputs, transition and actual phase identities when each route first ships. APP-4 later unifies those production consumers; no “negligible transient” assumption remains. |
| R2-07 | MAJOR | A pixel round trip alone does not preserve Hires.fix: the second pass changes prompt enhancement, encoding and conditioning outside the phase loop. | **Accepted.** PIPE-1/HOOK-2 and PLAN §3.1/3.3/6.6 model pass-local prompt, encode, reference and intermediate-decode stages. ENG-2 preserves enhancement/report/event/seed/output semantics and updates admission. PIPE-16/ENG-6 retain new Krea decoded streaming in scope while correcting the baseline's batch-only status. |
| R2-08 | MAJOR | Putting the shared Linear spine in media runtime crates while LLM crates consume it creates an MLX crate dependency cycle. | **Accepted.** BACK-1 and PLAN §1.2/9.2 place backend primitives below media and LLM. ENG-4a extracts the owner and migrates media/LLM projections, preserving all existing variants/platform partitions; ENG-4b then adds phase scales. Locked dependency checks are explicit implementation acceptance. |
| R2-09 | MAJOR | FEAT-1 acceptance can pass with unusable images; step-domain, axis, rounding and schedule-shift choices are unresolved. | **Accepted.** FEAT-1, DEC-10 and PLAN §6.1 specify proposed recipe-v1 K/scale/grid/pass/shift semantics, versioned replay, targeted-pass mode restrictions and bounded quality/content acceptance alongside full-cost performance. These concrete product choices remain proposed under DEC-10. |

The four requested weights-free design traces are in PLAN §6.5–6.8. They define the expected contract and future checks; no model or conformance implementation was executed. The full roadmap remains, with prerequisite ownership corrected and the stale aggregate calendar withdrawn pending consumer-based feature sizing. No implementation or tracker items were created by this revision.

## Round 3: re-review of draft 0.3 (2026-11-03)

Reviewed the published revision `71d278386e`, then verified the requirements/plan corrections at `1f5fd6508`. Full report: [round3-codex.md](evidence/reviews/round3-codex.md). This was a subsequent **self-review by the revision author**, not an independent review.

**Verdict: all nine round-2 findings addressed at the design level; no unresolved BLOCKER or MAJOR finding in the re-reviewed scope.** Confidence is high in the contract coherence, medium in unmeasured performance/quality and backend-generic feasibility. Open §10 decisions remain open; the design walkthroughs and document checks are not native implementation evidence.

| # | Sev | Finding | Disposition |
|---|---|---|---|
| R3-01 | MAJOR | A read-only endpoint evaluation must also protect APG/Feature mutable state, and geometry changes must invalidate incompatible state separately from solver history. | **Accepted.** PIPE-3 and PLAN §3.3/6.5/7.2 define `EvaluationKind::BoundaryEstimate`, snapshot nonmutation, dependency-based invalidation, priced scratch and future fake-operator closure checks. Source: inference `gen-core/src/guidance.rs:168-208`. |
| R3-02 | MINOR | PLAN §6.1 still summarized the preparation key as geometry-only. | **Accepted.** The worked-example row now uses the complete dependency contract and state rules. |
| R3-03 | MINOR | The UI fraction still said total steps rather than executed steps in the selected pass. | **Accepted.** FEAT-1 now states K/M for that pass's executed steps. |

Document links/fences, whitespace, nine dispositions, illustrative memory arithmetic and the 23-feature dependency DAG were checked. No implementation, model execution, builds/CI dispatches or tracker items. Design decisions require their existing approval process; no runtime or performance acceptance is claimed.

## Round 4: independent review of draft 0.3 by Claude (2026-11-03)

Reviewed `3bc17e3a0..1a5500bdb`. Full report: [round4-claude.md](evidence/reviews/round4-claude.md). This review is independent of the round-2/3 author.

**Verdict: all nine round-2 findings and their resolutions accepted** (three source claims spot-checked and confirmed). No blocker. **3 MAJOR and 5 MINOR findings**, mostly scope, altitude and fit with standing direction. All eight were applied in draft 0.4 at Michael's direction (2026-11-03).

| # | Sev | Finding | Disposition |
|---|---|---|---|
| C4-01 | MAJOR | ENG-1 has grown far beyond a ≤6-week walking skeleton. M1's dependency on SW-F327 #3264 is unstated. Proposal: split into ENG-1a (runner core + FLUX.1 migration) and ENG-1b (low-res start). | **Accepted.** ENG-1 split into ENG-1a (runner core + FLUX.1 migration, legacy deleted) and ENG-1b (low-res start, gated on #3264). Sequencing, M1 and the §3.9 first-consumer text updated. |
| C4-02 | MAJOR | MEM-3's per-allocation live graph risks rebuilding the dismantled memory apparatus, against the standing rules "two anchors plus derivation" and "no new memory gates". Proposal: keep sum-of-concurrent semantics at component/phase granularity on existing evidence, plus shape-computed transition deltas. | **Accepted.** MEM-3 and PLAN §3.9 now price at component/phase granularity: per-component weights, per-phase evidence, sum-of-concurrent streams, shape-computed transition deltas, alias rules only where needed. The R2-05 fixtures are kept. |
| C4-03 | MAJOR | REQUIREMENTS now carries implementation mechanics (FEAT-1 rounding and interpolation, PIPE-16 process text), and several new passages are single 150–300-word blocks. Proposal: move mechanics to PLAN §6.1/DEC-10 and the ENG-6 row; rewrite the dense passages as bullets. | **Accepted.** FEAT-1 and PIPE-16 rewritten as outcomes; mechanics stay in PLAN §6.1/§3.1 and DEC-10. PLAN §3.1 coordinates and boundary conversion, §3.3 state, preparation and stage notes, and §3.9 pricing are now bullets. |
| C4-04 | MINOR | Add a declared `EstimateSource::PreviousEvaluation`, which uses no extra evaluation, as a DEC-10 option judged by the bounded quality comparison. The endpoint evaluation stays the default. | **Accepted.** `EstimateSource::PreviousEvaluation` added as a DEC-10 candidate (PLAN §3.1). `EndpointEvaluation` stays the default. |
| C4-05 | MINOR | ENG-1 should not build an upgrader. `lowResStart` is a new key: persist a versioned block through the existing path and let APP-2 adopt it. | **Accepted.** ENG-1b persists a versioned `lowResStart` block through the existing path, with no upgrader; APP-2 adopts it. |
| C4-06 | MINOR | The withdrawn calendar leaves DEC-12 unenforceable. Proposal: restore per-feature ranges and an aggregate range, labelled low-confidence `[L]`, with assumptions. | **Accepted.** Per-feature ranges and an aggregate range restored as `[L]` (PLAN §9.2 and §9.4), with assumptions and the main uncertainty named. |
| C4-07 | MINOR | Streaming product delivery belongs to its existing epic. SGA owns the runner chunk contract and migration of the existing path. List the epic in the MIG-7 ledger. | **Accepted.** PIPE-16 and the ENG-6 row give SGA the runner chunk contract and the existing path. Streaming delivery stays with its epic, now listed in the MIG-7 ledger. |
| C4-08 | MINOR | README evidence row is stale. ENG-17's round-0 row needs an inline "superseded by R2-06" marker. | **Accepted.** README evidence row updated; ENG-17 row marked as superseded. |

## Round 5: Codex review of draft 0.4 (2026-11-03)

Reviewed Claude's design revision `2576499e03`. Full report: [round5-codex.md](evidence/reviews/round5-codex.md). This review checks the round-4 delta rather than reopening the original architectural direction.

**Keep the split, simpler pricing, clearer requirements and separate streaming product ownership. Confidence: high.** At review, two major clarifications and two minor findings were Open. All four are Accepted in draft 0.5 under Michael’s design-only revision request; this records document corrections, not implementation or decision approval. The default endpoint estimator remains the recommendation.

| # | Sev | Finding | Disposition |
|---|---|---|---|
| R5-01 | MAJOR | Transition pricing lists resident weights and shape deltas but does not explicitly include executing component/evaluation workspace or define the basis of existing peak evidence. | **Accepted.** MEM-3 and PLAN §3.9 price operation workspace from aggregate evidence, normalize total versus activation-only coverage, add only disjoint concurrent extras, and maximize over phases/operations. Joint totals are not counted twice; unknown basis follows existing unpriced posture. Worked fixtures cover both bases and learned-transition workspace. No per-allocation graph, re-keying or new campaign. |
| R5-02 | MAJOR | `PreviousEvaluation` skips an integration without defining multi-evaluation solver, state/draw, post-step and K/progress semantics; the percentage saving is unmeasured. | **Accepted.** FEAT-1/DEC-10 and PLAN §3.1/6.1 define completed-K retained-primary semantics, including normal correctors/midpoints, draws, hooks and state commits. Capture provenance/estimator identity is persisted, retention is priced and unsupported candidates refuse without shrinking default coverage. Endpoint stays default; no fixed saving remains. |
| R5-03 | MINOR | Mock-only chunk acceptance conflicts with HOOK-1's production-user requirement; the existing streaming owner is still a placeholder. | **Accepted.** PIPE-16 and PLAN §6.9/ENG-6 use the shared `emit` primitive in production for a single terminal batch chunk, with unchanged public outputs/events. SW-F148 (`sc-8432`) remains the separate streaming owner. Before/after adoption and NO-GO are explicit; no mock-only stream hook or HOOK-1/AP9 exception is needed. |
| R5-04 | MINOR | Stale live-set terminology remains, and the restored calendar counts 17 pre-pilot engine features instead of 16. | **Accepted.** Positive pricing references now use `PricingPlan` component/phase/operation accounting. The roadmap contains 16 pre-pilot engine features and 24 nodes including pilots and APP-1…7. Estimates remain low-confidence and do not follow mechanically from the count. |

Design-only. No requirements/plan changes, implementation, builds, model execution, CI dispatches or tracker items in round 5.

The subsequent draft-0.5 corrections above address round 5; its original report remains historical evidence. Round 6 below follows publication of the revision. No implementation, builds/model execution or program tracker items were created by the correction work.

## Round 6: self-review of draft 0.5 (2026-11-03)

Reviewed published revision `62e247b1fb`, then the final requirements/plan clarification at `b2b8d01720`. Full report: [round6-codex.md](evidence/reviews/round6-codex.md). This is the revision author's **self-review**, not an independent review.

**All four round-5 findings are addressed at the design-document level. No unresolved BLOCKER or MAJOR finding in the re-reviewed scope.** Confidence: high in the contract coherence; medium in unmeasured quality/performance and native migration feasibility. No §10 approval or runtime acceptance is implied.

Closure checks cover transition workspace and evidence inclusion basis, completed-K estimator provenance/state/draw semantics, production terminal-chunk adoption with SW-F148 before/after/NO-GO ownership, and `PricingPlan` terminology/counts. Editorial clarification labels endpoint estimation as the default and requires payload transfer without an extra full-clip copy or surviving legacy finalization path.

Verified links/fences/whitespace, four Accepted R5 dispositions, pricing and draw-coordinate arithmetic, and the acyclic 24-node roadmap with 16 pre-pilot engine features. These are document checks and walkthroughs; the owning features still implement their conformance/runtime proofs. No new unresolved findings, code implementation, learned execution, builds/CI dispatches or program tracker items.

## Section 10 decision review (2026-11-03)

**DEC-1 approved by Michael:** option (b), shared host math and identical layer APIs, is the baseline. Option (c), generic model graphs over a layer-level trait, is conditional on both pilots passing: a small representative model, then FLUX.3 end to end including LoRA training. Preserve capabilities, per-backend fidelity, speed, memory and binary-size bounds. If either fails, retain (b). Backend kernels and execution policy remain specialized; shared pipeline and low-res work proceed independently of generic rollout.

Recorded in REQUIREMENTS (G1/G5, BACK-5 and §10), PLAN (§3.7 and ENG-15) and README. Confidence: high in the approved choice and baseline; medium in unproven generic feasibility. DEC-2…19 remain open. This individual decision does not approve the overall design or authorize pilots, implementation or program tracker items. Next discussion: DEC-2, lane scope.

**DEC-2 approved by Michael:** include audio, LLM-based encoders and training as program deliverables. Design shared interfaces from the start and stage adoption around the image/video core. Share weights, layers, adapters, encoders and lifecycle contracts. Diffusion/flow audio uses the runner; autoregressive audio retains its token-generation driver. Training reuses differentiable model modules while keeping optimizer/loss/data orchestration. Chat-product features remain outside scope.

Recorded in REQUIREMENTS (§4, WGT-4, XL-1 and §10), PLAN (§1, with adoption owned by ENG-4a/8/11/14) and README. Confidence: medium-high; training and audio migration carry more uncertainty than encoder/primitive reuse. DEC-3…19 remain open. Design-only: no implementation, pilots, builds, model execution or program tracker items. Next discussion: DEC-3, schedule sources.

**DEC-3 approved by Michael:** support global-sliced, fresh-per-pass and explicit-table sources, resolved into one immutable schedule representation before execution. Each pass owns its schedule and step numbering; phase spans select executed outer-step ranges within that domain. Source choice does not implicitly select transition, re-noise or solver-history behavior; stream coupling remains explicit.

Recorded in REQUIREMENTS (PIPE-4 and §10), PLAN (§3.1 and ENG-1b dependency status) and README. Confidence: high; each source serves an existing route or recipe requirement, while execution consumes one resolved form. DEC-4…19 remain open. Design-only: no implementation, builds, model execution or program tracker items. Next discussion: DEC-4, prior-art salvage.

**DEC-4 reuse strategy approved by Michael:** selectively reuse scheduling helpers, backend-neutral conformance tests and execution safeguards from sc-20414. Build the shared runner around the approved pass/phase architecture. Check extracted pieces against current contracts; adapt cleanup to the current error policy. Use the old request schema for behavior/compatibility analysis, with any resulting schema owned by the shared option system.

Recorded in REQUIREMENTS (§10), PLAN (§2 and §3.3) and README. Confidence: high in selective reuse; medium in how much code transfers unchanged. The user-configurable chained-pass Feature remains open, as do DEC-5…19. This approval does not select that product feature or authorize implementation, builds, model execution or program tracker items. Next discussion: DEC-4's chained-pass product scope.

**DEC-4 chained-pass product scope approved by Michael:** include advanced, opt-in user-configurable chained denoise passes after the shared option foundations. Compile the recipe into the common `RenderPlan`, with explicit transitions/seeds/history, whole-plan applicability and pricing, and versioned per-backend replay. No general quality claim. This completes DEC-4's approval.

Added FEAT-2 and the cross-repository ENG-16 delivery slice (ENG-7 and APP-2/3 prerequisites), including controls, compatibility inventory, route coverage, composition and bounded output evidence. Updated current roadmap to 17 non-pilot engine features and 25 nodes; ENG-16's 4–6 week estimate is low-confidence, and keeping the aggregate range assumes parallel delivery alongside later component waves. Historical review counts remain at their reviewed revisions. Design-only: no program tracker items, implementation, builds or model execution. Next discussion: DEC-5, SW-F327 coordination.

Document checks: links/fences/whitespace and the 25-node dependency graph pass; ENG-16's prerequisites introduce no cycle. Refreshed DEC-5's factual overlap snapshot through CodeGraph, GitHub PR metadata and fetched main ancestry: #3258/#3264 are on the remediation feature branch, not main; #3260/#3268/#3270 remain open. DEC-5 itself remains unapproved.

**DEC-5 approved by Michael:** independent engine work may proceed once implementation is authorized, preserving worker-facing APIs and inference-pin compatibility. Overlapping SceneWorks changes wait for the owning remediation to reach `main`, then re-baseline. Reuse delivered remediation; in-review fixes retain their current owner. Refresh file/contract overlap at each feature's planning and verify main ancestry rather than treating a feature-branch merge as delivery. Unrelated work does not wait for all SW-F327.

Recorded in REQUIREMENTS (MIG-7, constraints and §10), PLAN (§3.9, §7.5 and roadmap policy) and README. Confidence: high. The prior PR snapshot is historical and must be refreshed at implementation planning. DEC-6…19 remain open. Design-only: no implementation, program tracker items, builds, model execution or changes to the remediation feature. Next discussion: DEC-6, structural architecture checks.

**DEC-6 approved by Michael:** narrow automated structural checks protect agreed ownership and dependency rules alongside each consolidation. Prefer language visibility/types and Cargo boundaries; focused syntax checks cover remaining gaps. Document essential backend/native variants and actionable failures; prove known violations fail and legitimate variants pass. Extend existing architecture tests/CI lanes without model execution. No new memory/measurement gates, source-hash snapshots, evidence-currency or formatting-sensitive checks. Numerical/runtime evidence remains separate.

Recorded in REQUIREMENTS (TOOL-4 and §10), PLAN (§1.1, overlap ledger and risk treatment) and README. Confidence: high in narrow checks' value; medium in precise check boundaries until migration. DEC-7…19 remain open. Design-only: no implementation, CI changes, program tracker items, builds or model execution. Next discussion: DEC-7, numerical fidelity classes.

**DEC-7 direction from Michael, class-policy approval still open:** bit-exact is preferred, but do not run endless GPU cycles over small differences (the user's example was “<1%”). Revised AP13/PIPE-10 and PLAN §7.3 to require a finite proof/investigation budget, reuse evidence, inspect source/math/state first, and stop GPU retries solely for validated sub-tolerance drift. Further learned runs answer a concrete unresolved question or validate relevant changed numerics. At the budget boundary, present measured evidence and a concrete acceptance/fix/fallback choice.

No universal 1% tolerance or pilot-gate waiver is inferred. Metrics, normalization, outlier bounds, output checks and run budgets are component/feature-specific, set before comparison. Material output/replay/capability changes remain explicit. Confidence: high in the bounded-effort direction, medium in unmeasured numerical limits. Updated README and BACK-5's pilot stop rule. Design-only; no implementation, program tracker items or execution. DEC-7 remains the active discussion.

**Revised DEC-7 approved by Michael:** bit-exact is preferred, with finite proof/investigation budgets and reuse of valid evidence. Supported class (b) differences within predefined component-specific bounds and acceptable output checks do not trigger GPU retries solely to eliminate rounding drift. Further runs answer a concrete unresolved question or validate relevant numerical changes. Material/semantic changes remain class (c), requiring per-route sign-off; no universal 1% threshold or pilot exception is approved.

Recorded in REQUIREMENTS (AP13, PIPE-10 and §10), PLAN (§7.3) and README. Confidence: high in policy; medium in unmeasured component limits, to be specified at feature planning. DEC-8…19 remain open. Design-only: no implementation, program tracker items, builds or execution. Next discussion: DEC-8, feature classes and admission authority.

**DEC-8 approved by Michael:** classify Features as recipe, equivalent or diagnostic. Recipes are explicitly user-chosen or part of the selected recipe. Admission may select only equivalent Features with declared, evidenced quality bounds; it cannot silently add low-res start/passes, change quantization tier or enable diagnostics to fit memory. Diagnostics are explicitly configured observation. Insufficient permitted adjustments produce a typed refusal. Feature classes govern runtime activation; AP13 classes govern migration fidelity. Quality evidence follows DEC-7's bounded-validation policy.

Recorded in REQUIREMENTS (HOOK-2, MEM-4 and §10), PLAN (§3.4) and README. Confidence: high. DEC-9…19 remain open: eleven decisions. Design-only: no implementation, program tracker items, builds or execution. Next discussion: DEC-9, web validation mechanism.

## Decision review complete — draft 0.6 (2026-11-03)

Michael accepted the remaining recommendations together after approving DEC-1…8 individually. **All DEC-1…19 are now approved design decisions.** This selects the design direction and its stated conditions; it does not authorize implementation, tracker creation or learned execution during this design-only review. Per-route output changes, specific scope exceptions, pilot outcomes and component limits retain their explicit review/evidence requirements.

| Decision | Approved direction |
|---|---|
| DEC-9 | Pure Rust validator in WebAssembly; authoritative server/worker checks; hybrid fallback for disproportionate portability/bundle cost. APP-2 sets and verifies UX/build budgets. |
| DEC-10 | Low-res recipe v1's declared scope, K/scale/grid/schedule rules and replay. Endpoint evaluation remains default; completed-K previous estimate remains conditional on solver support and bounded comparison. |
| DEC-11 | CodeGraph-key policy first; verify/reuse a delivered fix or land only the missing prerequisite. |
| DEC-12 | ≤6-week feature branch window; split/re-plan without dropping scope. |
| DEC-13 | Typed refusals for structurally unsupported operations; list concrete route exceptions for review at planning. Native generation remains available. |
| DEC-14 | Separately dispatched Tier 1 per migration batch, with evidence reuse and DEC-7 budgets; no per-PR/Cartesian campaign. Discover live runner access before use. |
| DEC-15 | At feature end, affected proof-set A/B on the same runner/window, combining fidelity/time/memory observations and reusing evidence; no refactor-only calibration campaign. |
| DEC-16 | APP-1 deletes the capability source-digest matrix and migrates its consumers. |
| DEC-17 | APP-7 deletes the memory generator/mirror after consumer migration and required evidence/provenance preservation. |
| DEC-18 | One bounded paired old-vs-new memory-adapter proof on each designated backend, duration stated beforehand and combined with applicable feature-end evidence. No run now. |
| DEC-19 | User-visible outer steps; evaluation accounting internal. |

Synchronized REQUIREMENTS (status, §10 and affected requirements) and PLAN (validation, recipe, migration, roadmap dependencies and proof policy); README now summarizes the approved design. Duration, quality and performance estimates remain estimates, not measured acceptance. Confidence: high in recorded choices and contract consistency; medium in unmeasured native feasibility/quality/performance. This is a document consistency check by the revision author, not an independent architecture review or runtime validation.

Verified all 19 unique decision rows are Approved, with no stale open-decision markers in the current requirements/plan/README. Local document links, code fences and whitespace pass. The 25-node roadmap remains acyclic with 17 non-pilot engine features; decision approval changed no dependency edge. Only the four design documents changed. No product code, tracker items, build/test model execution, CI dispatch or shared resource use occurred.

## Rounds 7 and 8 · Pre-planning revision (2026-11-07)

The [round 7 adversarial review](evidence/reviews/round7-codex.md) found five blockers and twelve major findings in draft 0.6. Draft 0.7 revises the requirements, runner contracts, proof cohorts and producer/consumer exits. The [round 8 closure review](evidence/reviews/round8-codex.md) checks every finding separately and records which observations are from current product code, which are design walkthroughs, and which must be proved during implementation.

| Findings | Accepted change in draft 0.7 |
|---|---|
| B1, B2, B3 | Use independent route coverage, bounded conditional stages and executor-owned native state. |
| B4, B5 | Refuse unpriced recipe increments, retain ordinary route posture and assign Hires.fix deletion to its last consumer. |
| M1, M2, M3 | Separate model identity from worker-local binding, define exact replay/share compatibility and specify reduced-grid img2img initialization. |
| M4, M5, M6 | Qualify trainer backward/resume, retain the H3 CUDA operating envelope and capture the pre-change product UI population. |
| M7, M8, M9 | Name API-only and desktop composition roots, isolate semantic descriptor export from measurement currency and preserve physical iOS admission behavior. |
| M10, M11, M12 | Bound total composed work, make terminal output publication attempt-owned, and distinguish inert-feature, migration and one-sided performance comparisons. |

DEC-1…19 retain Michael's November 3 approval. The new detailed contracts are draft resolutions. Round 8 is the revision author's design review. Native training parity, fit, speed and quality remain acceptance obligations of the named features.
