# Shared Generation Architecture (SGA)

This is a design-only program. Its aim is to slim SceneWorks and its inference engine by sharing code across models and backends, and by exposing hooks at every stage of generation. With those in place, a cross-cutting capability such as "low-res start", a guidance trick or a memory rung is built once. It then works for every model whose structure supports it.

| Document | Purpose |
|---|---|
| [REQUIREMENTS.md](REQUIREMENTS.md) | Problem, goals, principles (AP1–AP13), requirements, acceptance, metrics, and **decision status** (§10: all DEC-1…19 approved). |
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

**Draft 0.6 — all 19 design decisions approved by Michael on 2026-11-03.** DEC-1…8 were settled individually; Michael accepted the remaining recommendations together. The requirements and plan record the selected policies and their conditions.

- **Shared architecture:** shared host math and layer APIs are the baseline; generic model graphs remain conditional on both pilots. Audio, LLM-based encoders and training are included with staged adoption.
- **Runner and recipes:** all three schedule sources; selective prior-art reuse; configurable chained passes; approved low-res recipe v1 with endpoint estimation as default. The previous-estimate candidate still needs its bounded comparison and persisted identity.
- **Control and fidelity:** preserve user choice through recipe/equivalent/diagnostic classes. Admission selects only equivalent Features and never changes quantization tier. Bit-exact output is preferred, with predefined component tolerances and bounded investigation; no endless GPU retries or universal 1% threshold. Material output changes and pilot exceptions remain explicit.
- **App validation:** pure Rust rules compiled to WebAssembly, authoritative server revalidation and local-state checks; hybrid fallback if portability/bundle cost is disproportionate. APP-2 verifies measurable UX/build budgets.
- **Delivery and proof:** CodeGraph naming policy first, ≤6-week feature branches with re-planning, typed structural refusals, Tier 1 per migration batch and affected Tier 2 proof subsets at feature end. Evidence is reused and learned execution stays separately dispatched. SW-F327 overlap waits for main delivery; narrow structural checks use existing tests/CI.
- **Simplification:** delete the source-digest capability matrix and generated memory matrix through their owning features, preserving required evidence and migrating consumers. One bounded paired memory-adapter proof per designated backend; user-visible progress stays in outer steps.
- **Pricing and output:** aggregate phase/operation evidence covers workspace and concurrent extras without a program-wide allocation model. Shared production terminal-chunk adoption remains in SGA; repeated decoded streaming stays with SW-F148 (`sc-8432`).
- **Roadmap:** 17 non-pilot engine features; 25 nodes including pilots and application features. ENG-16 delivers configurable chained passes after ENG-7 and APP-2/3. Calendar estimates and parallel-delivery assumptions remain low-confidence.

The round-6 self-review remains historical document-level evidence. The post-decision consistency check is recorded in REVIEW_LOG; no native quality, performance or implementation acceptance is implied. Confidence is high in the documented contracts and medium in unmeasured native feasibility/performance.

This remains design-only. No CodeGraph program features/items, implementation, pilots, builds, CI dispatches or model runs were started. When implementation is separately requested, plan each roadmap feature with `codegraph-plan`, verify live prerequisites and execute through its tracked items.
