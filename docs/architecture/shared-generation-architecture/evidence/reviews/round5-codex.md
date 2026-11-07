# Round 5: review of Claude's draft-0.4 revision

Date: 2026-11-03. Reviewer: Codex. Reviewed design commit `2576499e036ad5ee4c477b35efd7fced0ef92445`, specifically its delta from `1a5500bdba`. This is a review of the round-4 changes, not another full-repository audit. Source checks retain inference baseline `a43724b82fbae35c9227cef4bcdabba6dd1da12c`. No implementation, model execution, build, dispatch or tracker-item creation.

## Assessment

**Keep the ENG-1a/1b split, component/phase pricing, clearer requirements and separate streaming product owner. Confidence: high.** Those changes make the plan more proportionate while preserving the corrected shared-pipeline direction. My draft-0.3 per-allocation requirement was broader than necessary; aggregate component/stage evidence plus explicit transition buffers is sufficient if its accounting basis is clear. This does not require rebuilding a per-model allocation graph or adding a measurement gate.

No additional broad architectural blocker was found. There are **two MAJOR clarifications and two MINOR consistency/coordination findings**, all **Open**. The default endpoint estimator remains sound as specified. The optional cheaper estimator is a different recipe choice, and its solver semantics need to be settled before it is selected. Design/process decisions in §10 remain open.

## R5-01 — MAJOR: Include transition component/evaluation workspace in the aggregate pricing rule

**Confidence: high.** PLAN §3.9, lines 539–547, gives a phase peak as resident weights plus activation evidence, but a transition peak as the departing resident set plus a shape-derived delta. That delta lists latent buffers and component residency, such as an upsampler or codec. It does not explicitly include the **activation/workspace peak** of a component invoked during the transition, or explain how endpoint-evaluation evidence is applied there.

Weights and latent buffers alone cannot bound a learned upsampler, codec invocation or model evaluation. The incoming component set may also differ from the departing set. The requirement says transitions are priced above both neighboring phases, but the plan's departing-residency formula does not state how that is achieved. Depending on scheduling, phase evidence may already cover an evaluation's ordinary work; that overlap must be explicit rather than assumed or charged twice.

**Proposed resolution, still at component/phase granularity:**

- Treat each executing transition operation as a small stage with its existing component/evaluation peak evidence, its required resident component set, and only the additional buffers/contexts that coexist with it.
- Maximize those operation peaks and the neighboring steady-phase peaks. Use the actual declared resident components at each operation; do not universally add both complete phase peaks together.
- State whether an evidence value includes weights, ordinary latents and shared workspace, or is activation-only. Convert to one accounting basis before adding deltas. A measured joint AV total must not also be added to its already included stream totals.

**Closure:** a pure fixture with a learned transition component whose workspace exceeds the latent-only delta, plus a joint-total evidence fixture that must not double-count streams. Reuse existing evidence and policy; no new calibration campaign or per-allocation model is needed.

## R5-02 — MAJOR: Define the cheaper estimator's solver and step semantics

**Confidence: high on the ambiguity; medium on quality or speed until measured.** PLAN §3.1, line 245, adds `PreviousEvaluation`: use x0 at sigma[K−1], skip the K-th low-res integration, and begin full resolution at sigma[K]. Keeping this as an explicit DEC-10 candidate is reasonable, but it does not yet define a generic execution contract.

For Euler, the primary x0 is unambiguous. For Heun, the last integration has both a primary estimate and a corrector at the next sigma ([solvers.rs:138–183](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/contracts/gen-core/src/sampling/solvers.rs#L138)). DPM++ SDE has a midpoint estimate and stochastic draws ([solvers.rs:327–403](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/contracts/gen-core/src/sampling/solvers.rs#L327)). Stopping after the primary evaluation omits sub-evaluations, draws, post-step hooks and state updates, not merely an arithmetic integration. Alternatively, completing all K steps and retaining a named earlier estimate preserves the step protocol, but differs from the text's instruction to skip the integration.

**Proposed resolution:** choose one explicit contract:

- Complete the normal K low-res steps and retain a precisely identified primary estimate from the last step, then use that earlier clean estimate in the transition; or
- Define a distinct terminal-step replacement, including which evaluations, stochastic coordinates, guidance/Feature updates and post-step hooks occur, plus how the logical K/progress count is reported.

Declare applicability per solver/driver. The candidate can initially be limited to a fully specified Euler path while the default endpoint estimator continues to support other eligible routes. That limits an optional estimator experiment, not the multi-resolution capability itself. Persist estimator identity/provenance so quality selection cannot silently change replay under the same recipe semantics.

The stated **1–3%** saving is not an established bound. The fractional saving is the removed work's elapsed cost divided by the endpoint recipe's full elapsed cost; it varies with M, K, scale, solver evaluation count and encode/decode overhead. Label it as an illustrative hypothesis or omit it until measured.

**Closure:** weights-free Euler and multi-evaluation traces specifying K, committed state and draw positions, followed by the existing bounded quality/time comparison if the candidate is pursued. Quality comparison complements a defined algorithm; it does not define that algorithm.

## R5-03 — MINOR: Reconcile mock-only chunk acceptance with the production-user rule

**Confidence: high.** Keeping decoded streaming delivery in its existing epic is appropriate. However, PIPE-16/ENG-6 now allow a mocked chunk producer to complete SGA's chunk contract while the current production route remains whole-clip batch. HOOK-1 still requires every hook point to have a production user or be removed. The new ledger row names only “Krea-realtime streaming epic (owner confirmed at ENG-6 planning).” It does not yet identify a feature key or a sequencing/adoption boundary.

The existing owner is discoverable now: CodeGraph resolves `sc-8432` to **SW-F148, “SceneWorks: Krea Realtime — interactive realtime streaming (spike-gated)”**, open with seven unfinished items at this review. Its imported requirements explicitly allow a NO-GO outcome from the feasibility spike. That is tracker evidence of ownership and intended gating, not proof of present runtime behavior or performance.

**Proposed resolution:** replace the placeholder with SW-F148 (`sc-8432`) and make the production adoption boundary explicit when ENG-6 is planned. Keep streaming product delivery with that owner. Either sequence the chunk implementation/adoption with it, or explicitly resolve the tension with HOOK-1/AP9 as a design decision, including the NO-GO case. A mock is appropriate contract evidence, but should not be reported as production adoption. Do not duplicate the streaming feature inside SGA.

## R5-04 — MINOR: Finish the terminology and estimate-count update

**Confidence: high.** The plan still says “derive the allocation live set” at §3.1 line 236 and “live-set inputs” in ENG-2/APP-4, after replacing the program-wide live-set model with `PricingPlan`. Align these passages with component/phase evidence and transition pricing so the removed obligation is not reintroduced during implementation.

The dependency table has **16 engine features before ENG-15 pilots**, not 17: ENG-1…14 contributes 14, splitting ENG-1 adds one and splitting ENG-4 adds one. There are 24 total roadmap nodes including pilots and APP-1…7; the graph is acyclic. Correct the count without treating the broad calendar estimate as validated by this arithmetic. Low-confidence ranges are useful for planning, but the actual consumer inventory still determines whether a slice fits DEC-12.

## Other round-4 changes

- A new `lowResStart` key does not require a bespoke legacy upgrader. Keep registration in the existing persistence/share path, explicit recipe-version handling and old-reader/replay checks. Those are compatibility obligations, not a second upgrader.
- The first-consumer admission rule and the #3264 dependency are retained in ENG-1b. That is the correct place for the new transition-pricing inputs.
- Moving detailed arithmetic out of requirements and breaking up long passages improves reviewability without changing the core contract.
- The existing typed boundary states, read-only guidance snapshots, preparation dependencies, independent RNG/history state and backend-layer direction remain intact.

Only this review report and REVIEW_LOG are changed by round 5. No proposed resolution above was silently applied to the requirements or implementation plan.
