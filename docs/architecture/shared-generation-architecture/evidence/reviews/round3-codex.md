# Round 3: re-review of Shared Generation Architecture draft 0.3

Date: 2026-11-03. Reviewer: Codex, the same agent that authored the draft-0.3 corrections; **this is a subsequent self-review, not an independent review**. Published revision first reviewed: `71d278386ea7e5ff4b7e5a511a137559d5875e13`. Final requirements/implementation-plan revision reviewed after the corrections below: `1f5fd650898f03c98bdd316738ead743444be8d5`.

Source checks retain the design's baseline revisions: inference `a43724b82fbae35c9227cef4bcdabba6dd1da12c` and SceneWorks `a1d95e2c43eef4a34a3fe5bf420bb2cc2abe181f`. The review does not claim to audit all subsequent changes on either repository's main branch. Live implementation planning must re-baseline through MIG-7.

## Verdict

**The nine round-2 findings are addressed at the design-document level. No unresolved BLOCKER or MAJOR architecture finding remains in this re-review's scope. Confidence: high in the coherence of the revised contracts; medium in performance, quality and backend-generic feasibility until implemented and measured.**

Shared planning and orchestration are now concrete enough to describe the first multi-resolution consumer without pushing its logic back into every family. A pass owns its schedules and stage bindings; phases own pass-local spans; typed transitions own state conversion; request state owns numerical history, randomness and mutable Feature state. Backend kernels and essential driver algorithms retain their declared numerical variants. These boundaries support the requested sharing and extension hooks.

This verdict is not approval of the open decisions in REQUIREMENTS §10 or runtime acceptance. DEC-10 now contains a specific proposed low-res recipe instead of leaving implementation-dependent semantics. Its product choices, the numerical/process decisions and any route exceptions still require the existing approval process. The design-only request did not authorize starting implementation or creating tracker items.

## Round-2 closure review

| Finding | Re-review result | Concrete contract and consumer ownership |
|---|---|---|
| R2-01 | Addressed | PIPE-4/FEAT-1, PLAN §3.1/6.5 distinguish noisy state and endpoint x0 with provenance, use a read-only evaluation and destination flow/VE noise coefficients, and count its cost. ENG-1 owns implementation and equation/trace checks. |
| R2-02 | Addressed | PIPE-1/2, HOOK-3 and PLAN §3.1/6.6/6.8 establish stable pass ids, actual executed-step coordinates, stream coupling, override/boundary ownership and whole-plan validation after composition. Fresh/explicit schedules are no longer competing phase coordinate systems. |
| R2-03 | Addressed | PIPE-3/11 and PLAN §3.3/6.7/6.8 separate request-owned solver/driver state, interval history, RNG coordinates and cache state. Carry requires compatibility; an inert split cannot repeat stochastic draws or add numerical work. |
| R2-04 | Addressed | PIPE-5 and PLAN §3.3/6.7 give preparation request arguments and declared model/conditioning/adapter/branch dependencies, with priced retention. Krea base/adapter/base and CFG batch changes have explicit closure cases. |
| R2-05 | Addressed | MEM-3 and PLAN §3.9/6.8 sum distinct concurrent allocations per budget domain and maximize over execution points. Joint AV workspaces, transition scratch and aliases are explicit. The illustrative 190/230-unit arithmetic is consistent and labeled as non-measurement. |
| R2-06 | Addressed | MEM-5/ADM-3 and ENG-1/2 deliver conservative live-set inputs and actual phase/transition scope identities with the first adopted routes. APP-4 later unifies those consumers; safe activation does not depend on that later feature. Count/rung/budget-boundary checks are assigned in §7.2. |
| R2-07 | Addressed | PIPE-1, PLAN §3.3/6.6 and ENG-2 preserve base versus refinement prompt preparation, intermediate RGB decode/pixel-fit, single-reference replacement, re-encoding, seed/event/output behavior and residency. Krea batch-versus-new-streaming scope is explicit and remains owned by ENG-6. |
| R2-08 | Addressed | BACK-1, PLAN §1.2 and ENG-4a/4b put primitives below both media and LLM, migrate media/LLM projections during extraction, preserve existing formats/platform partitions and leave no media↔LLM cycle. Program-wide audio adoption remains tracked in ENG-14. |
| R2-09 | Addressed as a concrete proposal | FEAT-1/DEC-10 and PLAN §6.1 define targeted pass, effective M/K, spatial-only scale, legal-grid refusal, fixed schedule/mu, endpoint arithmetic and versioned replay. Quality/content acceptance accompanies measured full-cost speed. Proposed product values remain open for approval; native proof remains future work. |

The four design walkthroughs in PLAN §6.5–6.8 cover the intended contract interactions. They are explanatory traces, not implemented conformance suites. No runtime claim is inferred from their existence.

## Additional findings from this re-review

### R3-01 — MAJOR, Accepted and corrected: read-only endpoint evaluation must include guidance and Feature state

At `71d278386e`, read-only behavior explicitly protected KV and RNG, but left guidance/Feature-state update semantics implicit. APG has a persistent running tensor: `MomentumBuffer::update` writes it and `normalize_diff` calls that update ([guidance.rs:168–208](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/contracts/gen-core/src/guidance.rs#L168)). An extra endpoint combine could advance momentum; carrying its old-grid tensor into the full-size phase could also cause incompatible-shape arithmetic. Solver-history Reset alone does not reset this state.

**Correction reviewed at `1f5fd6508`:** PIPE-3 and PLAN §3.3 add an explicit `EvaluationKind::BoundaryEstimate` propagated through prediction, guidance and Feature hooks. It evaluates a snapshot without committing momentum/cache/counter updates. Shape/configuration-dependent state has declared invalidation and is reset where incompatible after a geometry change; unchanged native trajectories keep their cadence. Snapshot scratch is priced. Unsupported read-only combinations refuse during composed-plan validation. §7.2 assigns fake-operator nonmutation/invalidation checks to the existing Tier 0 validation surface; these checks have not been implemented or run in this design-only change.

**Confidence: high.** This resolves the remaining state-contract ambiguity without adding family-specific low-res wiring.

### R3-02 — MINOR, Accepted and corrected: remove the geometry-only cache summary

PLAN §6.1's family row still described `prepare_step_context` as keyed on geometry alone, contradicting the complete dependency contract. It now explicitly uses all declared dependencies and points to the read-only/invalidation rules. **Confidence: high.**

### R3-03 — MINOR, Accepted and corrected: align the UI fraction with executed pass steps

FEAT-1's UI bullet still said “fraction of total steps,” despite the new definition of M after strength truncation in the selected pass. It now says K/M for that pass's executed steps. **Confidence: high.**

## Verification and limits

- Re-read the revised pass/transition/state contracts, Feature composition, memory pricing, first-consumer adoption, four traces, roadmap and round-2 closure criteria. Checked the guidance-state correction against exact baseline source.
- Documentation changes only. Checked Markdown links/fences and whitespace, nine explicit round-2 dispositions, illustrative memory arithmetic, and all 23 roadmap feature dependency nodes for undefined edges/cycles. Historical review reports remain unchanged.
- No learned forward, encode/decode, training, runtime conformance suite, build, CI dispatch, memory campaign or GPU use occurred. No implementation, release, PR or program tracker item was created. Concrete closure tests are specified for the future owning features.

The remaining risks are empirical and delivery-related: numerical equivalence on each migrated route, latency/peak memory of the extra endpoint forward, active-recipe quality, driver support, iOS partitions and complete consumer inventories. The existing validation tiers and corrected first-consumer ownership address those risks at implementation time. The earlier aggregate calendar is withdrawn rather than treated as evidence that the expanded prerequisite work fits.

The next design step is decision review against the revised §10, especially DEC-10. The full capability scope remains documented; no unimplemented scope has been declared complete.
