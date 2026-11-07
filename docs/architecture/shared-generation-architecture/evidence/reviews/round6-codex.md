# Round 6: self-review of SGA draft 0.5

Date: 2026-11-03. Reviewer: Codex, also the revision author; **a subsequent self-review, not an independent review**. Published revision first reviewed: `62e247b1fb8af34b87b6d2eddc3822494d9e4859`. Final requirements/plan revision after editorial clarification: `b2b8d01720dab62f5917a78c3b1ff1962a953dc7`.

Scope: the four round-5 findings and their effects on the existing boundary/state, pricing, output-adoption and migration contracts. This is not a fresh audit of every model or of main-branch changes since the stated source baselines. No implementation, learned execution, builds, CI dispatches or program tracker items.

## Verdict

**All four round-5 findings are addressed at the document level. No unresolved BLOCKER or MAJOR architecture finding remains in this re-reviewed scope. Confidence: high in the contract coherence; medium in unmeasured estimator quality/performance and native migration feasibility.**

The revision retains the shared runner direction, ENG-1a/1b separation, component-level accounting and separate streaming product owner. It does not restore a per-allocation memory graph or invent a new measurement campaign. DEC-10's estimator candidate remains a proposed recipe choice; this review approves no open §10 decision and establishes no runtime acceptance.

## Closure review

| Finding | Result | Contract checked |
|---|---|---|
| R5-01 | Addressed | MEM-3 and PLAN §3.9 include the workspace of each executing transition component/evaluation. Total versus activation-only evidence declares its inclusion basis; only missing resident components/coexisting buffers are added. The request maximum includes neighboring phases and each transition operation. Joint totals/aliases are not counted twice; unknown basis preserves existing unpriced posture. |
| R5-02 | Addressed | FEAT-1/DEC-10 and PLAN §3.1/6.1 choose completed-K semantics. The candidate retains the last outer step's named guided primary x0, then completes all normal corrector/midpoint evaluations, draws, state commits, materialization and hooks before transitioning. No integration is skipped; no endpoint forward is added. Estimator/source identity and provenance are persisted, capture retention is priced, and solver support is explicit. |
| R5-03 | Addressed | PIPE-16, PLAN §6.9, ENG-6 and the MIG-7 ledger identify SW-F148 (`sc-8432`). SGA's output primitive has a real batch consumer that emits one terminal chunk with unchanged public results/events. Repeated output uses the same primitive; no mock-only stream hook survives. Both before/after streaming adoption orders and SW-F148's NO-GO case are defined. |
| R5-04 | Addressed | Compilation, admission consumers, validation descriptions and roadmap now use `PricingPlan` component/phase/operation terminology. Counts are 16 pre-pilot engine features and 24 total roadmap nodes, with an acyclic dependency graph. The count does not validate the low-confidence calendar. |

### Pricing walkthrough check

The examples correctly distinguish an operation's ordinary workspace from new transition buffers:

- Total peak 180 plus disjoint extras 96 = 276. Activation-only peak 80 plus required resident weights 100 plus the same extras 96 also = 276. A budget of 250 refuses this priced candidate despite neighboring phases of 160 and 200 fitting.
- Joint AV total 190 already includes weights 100, video 40, audio 30 and joint workspace 20. It stays 190 rather than becoming 280 through duplicate additions. A new disjoint 40-unit buffer makes 230 while it coexists; an alias adds zero.
- Unknown evidence basis remains unpriced and uses the existing route posture; neither a smaller invented value nor a new refusal policy is inferred.

These are verified arithmetic examples, not measured device memory or executable pricing tests. The owning features still implement the pure fixtures and existing dispatched proofs.

### Estimator protocol check

For K completed steps, the captured primary estimate belongs to outer step K−1. Heun's later corrector and DPM++ SDE's midpoint cannot replace it merely because they are newer. The ordinary step still completes, so its state commits, stochastic coordinates and progress are unchanged before the geometry transition. For interior two-step SDE examples, draw addresses 0/1 then 2/3 complete, followed by 4/5 on the next drawing step. Terminal/no-noise branches keep their actual conditional behavior.

This is compatible with the existing primary/corrector/midpoint distinction at the source baseline ([solvers.rs:138–183](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/contracts/gen-core/src/sampling/solvers.rs#L138), [327–403](https://github.com/SceneWorks/inference/blob/a43724b82fbae35c9227cef4bcdabba6dd1da12c/crates/contracts/gen-core/src/sampling/solvers.rs#L327)). It does not prove that each native driver already exposes the capture seam; those support claims require the planned conformance evidence. The default endpoint evaluation still uses the read-only guidance/Feature snapshot contract.

The estimator choice is fixed in resolved recipe identity rather than chosen by admission or silently changed on replay. Extra retention can cost memory, so the candidate is judged on quality, full elapsed time and peak memory without assuming a fixed saving. Candidate refusal does not reduce the default recipe's eligible-route coverage.

### Output adoption check

Batch use is an actual migration of the common output/finalization path, not a dummy call added solely to satisfy HOOK-1. PLAN §6.9 requires payload transfer/reference without a second full-clip copy, removal of the replaced batch finalization path in the same feature, and unchanged public outputs/events/persistence. The permanent mapping to existing public types is a boundary conversion, not a temporary cross-feature bridge.

SW-F148 remains responsible for interactive streaming and its spike. CodeGraph was read during this revision and still identifies that feature as open and spike-gated with NO-GO permitted. That is ownership/requirements evidence, not proof of a current streaming implementation or current performance. ENG-6's production terminal-chunk consumer remains valid if the spike is NO-GO. Multi-chunk mocks prove contract behavior; native streaming completion still needs its own production evidence.

## Verification and limits

Re-read the revised requirements, estimator protocol, pricing rules, output adoption boundary, roadmap and dispositions. Checked document links, code-fence balance and whitespace; all four R5 rows are Accepted. Parsed all 24 roadmap nodes to check references/cycles and the 16-feature pre-pilot engine count. Verified the illustrative pricing arithmetic and completed-K draw-coordinate arithmetic without executing a model or a runtime conformance suite.

Two editorial details were clarified during re-review: the §6.1 summary explicitly labels the endpoint estimator as the **default**, and the output adoption text explicitly preserves payload ownership/copy behavior and removes the old finalization path. No additional unresolved finding resulted.

The required implementation proofs remain in their owning features. Document checks do not establish native numerical fidelity, a memory-fit guarantee, active-recipe quality, a speedup or streaming delivery. The revised §10 proposals remain ready for decision review; no implementation work was started.
