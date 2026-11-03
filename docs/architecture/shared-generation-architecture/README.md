# Shared Generation Architecture (SGA)

This is a design-only program. Its aim is to slim SceneWorks and its inference engine by sharing code across models and backends, and by exposing hooks at every stage of generation. With those in place, a cross-cutting capability such as "low-res start", a guidance trick or a memory rung is built once. It then works for every model whose structure supports it.

| Document | Purpose |
|---|---|
| [REQUIREMENTS.md](REQUIREMENTS.md) | Problem, goals, principles (AP1–AP13), numbered requirements, program acceptance tests, metrics, and **decisions required** (§10, DEC-1…19). |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Target architecture, engine and SceneWorks designs, hook map, worked examples, migration and coordination (including the SW-F327 overlap ledger), validation tiers, roadmap (ENG-1…15, APP-1…7), and risks. |
| [REVIEW_LOG.md](REVIEW_LOG.md) | The disposition of every review finding, round by round. |
| [evidence/](evidence/) | Eight read-only research reports and the round-0 internal reviews. Every factual claim is cited to these or to code, at inference `origin/main` `a43724b82` and SceneWorks `main` `a1d95e2c4`. |

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
| [reviews/](evidence/reviews/) | Round-0 adversarial reviews (engine; SceneWorks and process) |

## Status

**Draft 0.2**, ready for review round 1. Round 0 consisted of two internal adversarial reviews: 69 findings, all 12 blockers accepted.

No CodeGraph features or items will be created until the design is approved. After approval, each roadmap feature is planned individually with `codegraph-plan`.
