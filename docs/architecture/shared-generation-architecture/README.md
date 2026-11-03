# Shared Generation Architecture (SGA)

This is a design-only program. Its aim is to slim SceneWorks and its inference engine by sharing code across models and backends, and by exposing hooks at every stage of generation. With those in place, a cross-cutting capability such as "low-res start", a guidance trick or a memory rung is built once. It then works for every model whose structure supports it.

| Document | Purpose |
|---|---|
| [REQUIREMENTS.md](REQUIREMENTS.md) | Problem, goals, principles (AP1–AP13), numbered requirements, program acceptance tests, metrics, and **decisions required** (§10, DEC-1…19). |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Target architecture, hook map, design traces (§6.5–6.8), production chunk adoption boundary (§6.9), migration, validation, roadmap (ENG-1a/1b…15 with ENG-4a/4b, APP-1…7), and risks. |
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
- **Roadmap:** 16 engine features before pilots; 24 nodes including pilots and application features. The calendar remains a low-confidence estimate, not a consequence of those counts.

The round-6 self-review found no unresolved blocker or major finding in the corrected scope. It is a document-level assessment, not native implementation or performance evidence. No product or process decision in REQUIREMENTS §10 is implicitly approved by these edits.

No CodeGraph features or items will be created until the design is approved. After approval, each roadmap feature is planned individually with `codegraph-plan`.
