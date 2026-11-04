# Shared Generation Architecture (SGA)

This is a design-only program. Its aim is to slim SceneWorks and its inference engine by sharing code across models and backends, and by exposing hooks at every stage of generation. With those in place, a cross-cutting capability such as "low-res start", a guidance trick or a memory rung is built once. It then works for every model whose structure supports it.

| Document | Purpose |
|---|---|
| [REQUIREMENTS.md](REQUIREMENTS.md) | Problem, goals, principles (AP1–AP13), requirements, acceptance, metrics, and **decision status** (§10: DEC-1…6 approved; DEC-7…19 open). |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Target architecture, hook map, design traces (§6.5–6.8), production chunk adoption boundary (§6.9), migration, validation, roadmap (ENG-1a/1b…16 with ENG-4a/4b, APP-1…7), and risks. |
| [REVIEW_LOG.md](REVIEW_LOG.md) | The disposition of every review finding, round by round. |
| [evidence/](evidence/) | Eight research reports and review reports (round 0 and rounds 2–6). Source-behavior citations retain inference `a43724b82` and SceneWorks `a1d95e2c4`; CodeGraph identifies the current streaming owner. Design walkthroughs are not executed proof. Current dispositions live in REVIEW_LOG. |

## Evidence reports

| Report | Topic |
|---|---|
| [A](evidence/A-gen-core-contracts.md) | gen-core contract layer: Generator, request, capabilities, memory contract, sampling, weights, bundles; pain points P1–P18 |
| [B](evidence/B-family-anatomy.md) | Per-family pipeline anatomy across 14 families on both backends; stage model; variance catalog; feature mapping |
| [C](evidence/C-component-inventory.md) | Component inventory (encoders, VAEs, blocks, kits, quantized Linear, conditioning modules) and a dedupe ranking |
| [D](evidence/D-backend-options.md) | Backend options and pilots |
| [E](evidence/E-engine-subsystems.md) | Weights, quantization, adapters, capabilities, training, audio and LLM, iOS, test infrastructure |
| [F](evidence/F-worker.md) | SceneWorks worker: lifecycle, engine boundary, lanes, admission paths, cache, cfg split |
| [G](evidence/G-app-layers.md) | SceneWorks manifest, hand tables, request contracts, validation, web gating, persistence, tooling, LOC census |
| [H](evidence/H-constraints.md) | Process rules, prior attempts, code-review themes, in-flight work, hazards, hardware |
| [reviews/](evidence/reviews/) | Round-0 adversarial reviews, round-2 architecture review, [round-3 self-review](evidence/reviews/round3-codex.md), [round-4 independent review](evidence/reviews/round4-claude.md), [round-5 review](evidence/reviews/round5-codex.md), and [round-6 self-review](evidence/reviews/round6-codex.md). |

## Status

**Draft 0.5.** It addresses all four round-5 findings while retaining the ENG-1a/1b split, earlier numerical/state contracts, existing persistence path and low-confidence planning estimates:

- **Pricing** reuses aggregate phase/operation evidence with a declared inclusion basis and adds only extra coexisting component/buffer costs. Workspace is covered without a program-wide allocation model or new measurement gate.
- **Estimator candidate:** `PreviousEvaluation` completes K steps normally and retains their final step's named guided primary x0; it skips no integration. Endpoint evaluation stays default. Choice/provenance is persisted; no fixed saving is assumed.
- **Chunk adoption:** the current Krea batch path uses one terminal chunk through the shared production output primitive. Repeated decoded streaming stays with **SW-F148 (`sc-8432`)**, including its NO-GO outcome; no dormant streaming hook is justified by a mock alone.
- **Roadmap:** 17 non-pilot engine features; 25 nodes including pilots and application features. ENG-16 adds the approved configurable chained-pass recipe after ENG-7 and APP-2/3. The calendar remains a low-confidence estimate, with its parallel-delivery assumption stated in the plan.

The round-6 self-review found no unresolved blocker or major finding in the corrected scope. It is a document-level assessment, not native implementation or performance evidence. Michael explicitly approved DEC-1…6 on 2026-11-03: (b) is the backend baseline, with (c) conditional on both pilots passing; audio, LLM-based encoders and training are included, with interfaces designed from the start and staged adoption around the image/video core; all three schedule sources resolve into one immutable per-pass representation. DEC-4 includes configurable chained passes. DEC-5 permits future independent engine progress while overlapping SceneWorks changes wait for remediation on `main` and re-baseline. DEC-6 permits narrow structural checks in existing tests/CI, with no model execution or new memory/measurement gates. DEC-7…19 remain open. The discussion remains design-only; neither pilots nor implementation are authorized by individual decision approvals.

No CodeGraph features or items will be created until the design is approved. After approval, each roadmap feature is planned individually with `codegraph-plan`.
