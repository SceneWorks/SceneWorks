# Shared Generation Architecture (SGA)

This is a design-only program. Its aim is to slim SceneWorks and its inference engine by sharing code across models and backends, and by exposing hooks at every stage of generation. With those in place, a cross-cutting capability such as "low-res start", a guidance trick or a memory rung is built once. It then works for every model whose structure supports it.

| Document | Purpose |
|---|---|
| [REQUIREMENTS.md](REQUIREMENTS.md) | Problem, goals, principles (AP1–AP13), numbered requirements, program acceptance tests, metrics, and **decisions required** (§10, DEC-1…19). |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Target architecture, engine and SceneWorks designs, hook map, four boundary/state/memory design traces (§6.5–6.8), migration and coordination, validation tiers, roadmap (ENG-1a/1b…15 with ENG-4a/4b, APP-1…7), and risks. |
| [REVIEW_LOG.md](REVIEW_LOG.md) | The disposition of every review finding, round by round. |
| [evidence/](evidence/) | Eight read-only research reports and the review reports for rounds 0, 2, 3 and 4. Every factual claim is cited to these or to code, at inference `origin/main` `a43724b82` and SceneWorks `main` `a1d95e2c4`. |

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
| [reviews/](evidence/reviews/) | Round-0 adversarial reviews, the round-2 architecture review, the [round-3 re-review](evidence/reviews/round3-codex.md) (self-review), and the independent [round-4 review](evidence/reviews/round4-claude.md). |

## Status

**Draft 0.4.** It addresses the round-4 findings (C4-01…08) on top of draft 0.3, which resolved all nine round-2 findings:

- **ENG-1 is split.** ENG-1a migrates the runner core and FLUX.1; ENG-1b adds low-res start, the first Feature.
- **Memory pricing** works at component and phase granularity, using existing evidence plus shape-computed transition deltas.
- **Requirements state outcomes.** Mechanics moved to the plan, and dense passages became bullets.
- **DEC-10** now lists a zero-extra-evaluation estimator as a candidate.
- **No new upgrader in ENG-1b.**
- **Estimates are restored** at low confidence.
- **Streaming delivery** stays with its own epic.

No product or process decision in REQUIREMENTS §10 is implicitly approved by these edits.

No CodeGraph features or items will be created until the design is approved. After approval, each roadmap feature is planned individually with `codegraph-plan`.
