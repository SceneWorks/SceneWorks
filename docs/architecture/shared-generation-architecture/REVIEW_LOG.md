# Review log

This log records the disposition of every review finding against the Shared Generation Architecture documents. Each finding is marked **Accepted** (the docs were changed), **Modified** (accepted in a different form), or **Rejected** (with the reason).

## Round 0: internal adversarial reviews of draft 0.1 → draft 0.2 (2026-11-03)

Two independent reviewers read draft 0.1 and checked its claims against the code. Their full reports are in [evidence/reviews/](evidence/reviews/).

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
| ENG-17 | MAJOR | ENG-1 was too big, contradicted the all-routes rule, and had an admission adapter that crossed features. | **Accepted.** ENG-1 is now the runner plus all FLUX.1 routes plus low-res start. Admission is unchanged, because the maximum phase is the final geometry. The cross-feature adapter is removed. |
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
