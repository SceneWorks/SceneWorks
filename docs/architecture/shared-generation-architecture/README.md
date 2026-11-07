# Shared Generation Architecture (SGA)

This is a design-only program. Its aim is to slim SceneWorks and its inference engine by sharing code across models and backends, and by exposing hooks at every stage of generation. With those in place, a cross-cutting capability such as "low-res start", a guidance trick or a memory rung is built once. It then works for every model whose structure supports it.

| Document | Purpose |
|---|---|
| [CodeGraph plan and session handoff](CODEGRAPH_PLAN.md) | Feature and item keys, direct blockers and starting work. Open SW-F331 in CodeGraph for the live program index. |
| [REQUIREMENTS.md](REQUIREMENTS.md) | Problem, goals, principles (AP1–AP13), requirements, acceptance, metrics, and **decision status** (§10: all DEC-1…19 approved). |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Target architecture, worked traces (§6), migration, proof cohorts and the engine/app roadmap (§9). |
| [REVIEW_LOG.md](REVIEW_LOG.md) | The disposition of every review finding, round by round. |
| [evidence/](evidence/) | Eight research reports plus the review history. A–H retain their original source revisions. Round 7 and its closure review use the November 7 main/pin pair. |

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
| [reviews/](evidence/reviews/) | Historical reviews through round 6, the [round 7 adversarial review](evidence/reviews/round7-codex.md) and the [round 8 closure review](evidence/reviews/round8-codex.md). |

## Status

**Draft 0.7 · November 7, 2026.** Michael approved DEC-1…19 on November 3. This revision addresses the five blockers and twelve major findings raised against draft 0.6. It proposes concrete runner, replay, pricing and cutover contracts under those decisions.

- The runner admits fixed and bounded conditional stages, preserves native owning-thread execution, and records the actual path taken. The independent route inventory includes callable product, platform and bespoke surfaces.
- Hires.fix migrates by provider and mode. ENG-7 owns final removal after the last imported and generic caller has moved. ENG-6 closes its image/video cohort, ENG-14 closes audio, and app features retain their public lane obligations.
- Low-res recipe v1 uses a versioned outer share for active behavior and prices every added operation. The existing FLUX.1 evidence bounds ordinary render observations. Qualification of the new transition remains an ENG-1b product obligation.
- Shared host math and layer APIs remain the baseline, with backend-generic graphs conditional on both pilots. Preserve bounded AP13 proof, the H3 CUDA envelope, trainer backward/resume behavior, and iOS headroom and size.
- The design baseline is SceneWorks `e1ffa54df6` with inference `2b09938e33`. SW-F327's main-delivery condition and CodeGraph branch naming are satisfied. The 25-node roadmap retains its separate per-feature acceptance boundaries.

The [closure review](evidence/reviews/round8-codex.md) records exactly what was verified before planning. Native feasibility, quality and performance are owned by the affected implementation features.
