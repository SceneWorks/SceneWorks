# Round 4: independent review of draft 0.3 (Codex rounds 2–3)

Date: 2026-11-03. Reviewer: Claude (the draft 0.1/0.2 author; independent of the round-2/3 changes). Reviewed: `3bc17e3a0..1a5500bdb`. That covers four commits, which changed REQUIREMENTS, IMPLEMENTATION_PLAN, README and REVIEW_LOG and added the round-2/3 reports. Baselines are unchanged: inference `a43724b82`, SceneWorks `a1d95e2c4`. Read-only: no builds, model execution or tracker writes.

## Verdict

**I accept all nine round-2 findings and their direction. Confidence: high.** They are real gaps in my draft 0.2. I spot-checked three of the source claims the resolutions rely on, and all three hold:

- The Hires base pass deliberately skips prompt enhancement. Enhancement "belongs to the final persisted pass" (`worker/src/image_jobs/base.rs` ~8708).
- `mlx-gen` already depends on `mlx-llm` for `text_sample` (`mlx-gen/Cargo.toml:159-161`). That edge is what turns "llm consumes the media Linear" into a Cargo cycle.
- DPM++ SDE draws noise at `(seed, 2*i)` and `(seed, 2*i+1)` (`gen-core/src/sampling/solvers.rs:380`). Restarting the index per slice would therefore repeat draws.

Round 3 is a **self-review by the author of the round-2 corrections**. The round-3 report says so itself. Its verdict ("no unresolved BLOCKER/MAJOR") therefore needs an independent check, and this round provides one. I found no blocker. I found **three MAJOR and five MINOR issues**. Most of them concern scope, altitude and fit with standing direction, not correctness of the new contracts.

## Findings

### C4-01 — MAJOR: ENG-1 is no longer a walking skeleton; split it

ENG-1 now contains:

- the pass/phase `RenderPlan`;
- typed boundary states;
- request-owned solver, RNG and guidance state;
- dependency-keyed preparation;
- whole-plan validation;
- live-set pricing and `MemoryRequestScope` identities;
- the error and count contracts;
- **all** FLUX.1 routes on both backends;
- the Feature registry;
- low-res recipe v1 with a quality acceptance;
- `[SW]` admission-consumer updates, persistence and upgrader, and the UI.

That is not a 6-week feature. DEC-12's window cannot be met or enforced without an estimate, and the estimate was withdrawn (C4-06).

M1 also now depends on SW-F327 #3264. The `[SW]` admission inputs touch `candle_memory_strategy.rs` and `mlx_fit_gate.rs`, and DEC-5 gates those files on #3264. The plan doesn't say so.

**Proposed fix.** Split ENG-1 at a complete consumer boundary:

- **ENG-1a: runner core plus FLUX.1 migration.** This covers the plan, passes and phases, state contracts, preparation keys, the scope driving and the error/count contracts. All FLUX.1 routes move onto the runner as class (a) and their legacy orchestration is deleted. It is a complete refactor with no user-visible change, which satisfies AP9 because the consumers are migrated and the old path is deleted.
- **ENG-1b: low-res start as the first Feature.** This covers the Feature registry and export field, recipe v1, the endpoint estimator, the transition delta in pricing and scope identities, and the `[SW]` option, persistence and UI. It is gated on #3264.

State M1's dependency on #3264 explicitly.

### C4-02 — MAJOR: MEM-3's per-allocation live graph risks rebuilding the memory apparatus you dismantled

R2-05 is right that concurrent streams must be summed, not maxed. The resolution goes much further. The new MEM-3 and PLAN §3.9 require a live-allocation graph in which every allocation has:

- a stable physical identity;
- a size bound;
- a budget domain;
- first/last use;
- alias deduplication;
- a per-point maximum.

Producing per-allocation size bounds for every route means modelling every family's activation workspaces. That is the job the calibration and measurement apparatus did. Epic 18093 diagnosed that apparatus as "armor against agent failure modes", and you tore it down. The new graph also sits uneasily with two standing directions:

- memory evidence = **two anchors plus derivation**, never a measured grid (memory note `ios_evidence_is_two_anchors_plus_derivation_not_a_grid`);
- **no new memory/measurement gates**.

**Proposed fix.** Keep the principle, but express it at component and phase granularity on existing evidence:

- **Weights:** per-component weight bytes, known from the weight plan.
- **Steady phases:** per-stage/phase activation peaks from today's measured or derived evidence (anchors plus derivation).
- **Concurrent streams:** each phase's peak is a **sum**. An AV joint phase sums its streams plus its joint workspace from evidence.
- **Transitions:** an explicit transition delta computed from latent shapes and dtypes (source plus destination latents, endpoint x0, fresh noise, f32 scratch), plus declared component residency such as an upsampler or codec.
- **Phase rule:** take the maximum over phases and transitions only, with the declared alias rules for latents.
- **Scope:** reserve per-allocation identity for the few places that need it, chiefly latent aliasing at transitions. Do not make it a program-wide modelling obligation.

The R2-05 closure fixtures (two streams that each fit but don't fit together; a transition peak; alias deduplication) still pass under this formulation.

### C4-03 — MAJOR: requirements altitude and readability

**Altitude.** REQUIREMENTS now carries implementation specification:

- FEAT-1 specifies floor-to-quantum rounding, interpolation semantics and the resolved replay fields;
- PIPE-16 contains process directives ("Reconcile the existing streaming epic…", "Do not report…");
- PIPE-3 and PIPE-5 describe internal state mechanics.

Requirements should state observable behaviour. The mechanics belong in the plan.

**Readability.** Several new passages are single blocks of 150–300 words: PLAN §3.1 *Coordinates and composition* and *Boundary conversion*, §3.3 *Request execution state* and *Guidance/Feature state*, §3.9 *Pricing*, and the DEC-10 cell. These documents are going through multiple human review rounds, and in this form they are hard to review and easy to mis-read.

**Proposed fix.**
- Move the recipe-v1 mechanics (rounding, interpolation, endpoint arithmetic, persisted fields) into PLAN §6.1 and DEC-10. FEAT-1 then states the outcomes: the targeted pass, the K/scale domains, refusal cases, deterministic replay, quality acceptance and the speed measurement.
- Move PIPE-16's process text into the ENG-6 row.
- Break the dense passages into short bullets: one rule per bullet, with the evidence citation at the end.

### C4-04 — MINOR: add a zero-extra-evaluation estimator as a DEC-10 candidate

The default endpoint estimator costs one extra low-res forward at σ[K]. That is correct and safe, and its cost is small: a low-res forward is roughly ¼ to ⅛ of a full-res step.

There is a cheaper standard alternative. Take x0 from the trajectory evaluation at σ[K−1] and jump straight to σ[K] at full resolution with fresh noise, skipping the K-th low-res integration. For Euler, that x0 is exactly what the step integrates toward. Round 2 correctly said the most recent x0 must not be *silently* treated as the endpoint estimate. A declared `EstimateSource::PreviousEvaluation`, recorded with its provenance, is a different matter.

**Proposed fix.** List both estimators as DEC-10 options. Keep the endpoint evaluation as the default. Decide with the bounded quality comparison FEAT-1 already requires. Expected benefit is small (about 1–3% of total time), so the priority is low.

### C4-05 — MINOR: ENG-1 should not build its own upgrader

The plan has ENG-1 upgrade "existing persistence carriers … ahead of the general APP-2 consolidation". `lowResStart` is a **new** key: no legacy shape exists, so there is nothing to upgrade. Building upgrade machinery in ENG-1 creates a second mechanism, which APP-2 must then absorb.

**Proposed fix.** ENG-1 persists a versioned `lowResStart` block (`{version, k, scale}` plus the resolved fields) through the existing `ADVANCED_KEY_RULES`/share path. It defines the forward-compatibility behaviour (an old build refuses or ignores the block, whichever the share-envelope rules already specify). APP-2's upgrader adopts the block as one of its shapes.

### C4-06 — MINOR: restore an explicit, low-confidence estimate rather than none

The aggregate calendar was withdrawn because prerequisite work grew. That is honest. But it leaves:

- DEC-12 (the ≤6-week window) unenforceable;
- the program with no size at all for the go/no-go decision.

The standing instruction is to state estimates with explicit confidence, not to omit them.

**Proposed fix.**
- Give each feature a low-confidence range, with its assumptions named. Include ENG-1a/1b after C4-01's split.
- Restate the aggregate as a range labelled `[L]`, noting that it is re-estimated at each feature's planning.

### C4-07 — MINOR: Krea-realtime streaming belongs to its own epic; SGA owns the runner contract

PIPE-16 and ENG-6 now make SGA responsible for "deliver/adopt decoded chunk streaming, reconciling the existing streaming epic". The correction of the baseline (Krea-realtime decodes once today) is right.

Taking delivery of another epic's product scope is a different matter. It enlarges SGA and risks two owners for one capability.

**Proposed fix.**
- **SGA's scope:** the runner's chunk-stage contract, plus migration of whatever Krea-realtime path exists when ENG-6 runs.
- **Streaming product delivery:** stays with its epic, listed in the MIG-7 overlap ledger as "conflicting → sequenced" or "delivered elsewhere".

This is neither a narrowing nor a deferment of SGA scope. The capability is tracked in its existing owner.

### C4-08 — MINOR: two small consistency fixes

1. **The README evidence row still says** "Eight read-only research reports and the round-0 internal reviews". It should cover rounds 0–4.
2. **The REVIEW_LOG header** says round-0 dispositions are "superseded where noted". ENG-17's row still reads "Admission is unchanged, because the maximum phase is the final geometry", which R2-06 overturned. Add an inline "(superseded by R2-06)" marker on that row so a reader doesn't act on it.

## Things that are right and should not change

- **Typed boundary state.** Separating noisy state from a clean estimate with provenance, and using `ModelSampling::noise_scaling_coeffs` instead of a second family rule, closes a real double-noising bug class.
- **Per-pass state.** A per-pass schedule domain with executed-step coordinates, and Hires as a separate pass that low-res start targets once, are both correct.
- **Independent state lifetimes.** Request-owned solver state with RNG coordinates independent of history reset is correct. So is `EvaluationKind::BoundaryEstimate` with snapshot non-mutation; the round-3 APG catch was real.
- **Preparation dependencies** include adapter scales and branch layout. This is required for Krea and for Wan's CFG batch.
- **Crate layout.** `*-layers` sits below both media and LLM, with the direction stated as the contract (ENG-4a/4b).
- **First-consumer admission rule.** The feature that first adds allocations or phases updates admission for its routes, rather than waiting for APP-4. This is consistent with AP9.
