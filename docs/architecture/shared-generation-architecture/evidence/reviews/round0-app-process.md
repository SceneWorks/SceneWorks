# Adversarial review: SGA REQUIREMENTS / IMPLEMENTATION_PLAN, SceneWorks and process side

Reviewer scope: APPREG, OPT, ADM, LANE, TOOL, MIG, §7 PATs, §8 metrics, §9 constraints, §10 decisions; plan §4, §7, §8, §9 (APP-1..5, §9.1), §10, §11.
Revisions inspected: SceneWorks `a1d95e2c4` (this worktree), inference `origin/main`, live GitHub state (PRs, branches, runners) on 2026-11-03. Read-only; nothing executed beyond grep/gh.

## Summary verdict

**Not approvable as written for the SceneWorks track. Confidence in this verdict: high.**

The direction is sound. Specifically, the typed `ModelSpec`, the `OptionSpec` seeded from `ADVANCED_KEY_RULES`, an admission contract with a `BackendBudget` hook, and the lane harness modelled on `generate_video_using` all follow from the evidence. The problems are on the app side and in how the work is sequenced:

1. **DEC-5 says the engine track can start now. That premise is false.** ENG-1 already contains SceneWorks worker, admission, core and web edits. Every later ENG feature has a SceneWorks consumer half, and that half lands in the files SW-F327 has open PRs against today.
2. **The plan breaks its own MIG-2/AP9 rule.** It has at least four cross-feature interim adapters or dual paths. Several APP features also depend on late ENG features (ENG-2b, ENG-5, ENG-6) that §9.1 does not show.
3. **Three of the four "single source" inventories are incomplete:**
   - "18 admission paths" misses at least 7 more.
   - The `ADVANCED_KEY_RULES` seed misses live keys.
   - The "6–8 hand tables" count misses more than a dozen.
4. **Several PATs cannot be tested as defined:**
   - PAT-1 is contradicted by ENG-1's own scope.
   - PAT-4 and PAT-8 have no method.
   - PAT-10 turns "goals, not gates" into a gate.
5. **Several process claims don't match the repository:**
   - The Tier 1 workflow does not run MLX on `nax-macos-2`.
   - Candle-gated tests never run in feature-PR CI.
   - Per-batch Tier 1 conflicts with FEATURE_DEVELOPMENT.md's "no per-story validation matrix".
   - The 61-source backend-capability-matrix gate fires on almost every APP PR and is not mentioned.

Severity counts: 7 BLOCKER (findings 1, 2, 4, 7, 11, 26, 34), 28 MAJOR, 8 MINOR.

---

## Findings

### A. Coordination, sequencing and migration state

**1. BLOCKER: DEC-5 / REQ §9.8 / PLAN §7.5, §9 — "start the engine track now" is not engine-only work, and it collides with live SW-F327 PRs.**

- **Problem.** The SceneWorks track is gated on SW-F327, but the engine track is not, even though ENG features have SceneWorks halves in the same files.
  - ENG-1 item 1 deletes Krea's bespoke multiphase drivers. The worker lane `crates/sceneworks-worker/src/image_jobs/krea_multiphase.rs` consumes them.
  - ENG-1 item 5 edits "today's admission paths", i.e. `mlx_fit_gate.rs` and `candle_memory_strategy.rs`.
  - ENG-1 item 6 edits core validation, `ADVANCED_KEY_RULES` and the web.
  - ENG-3 removes `Generator` family methods (which the worker reads), changes the request the worker builds at about 45 sites [F §2.3], and moves SceneWorks to bundles (REG-5), which is a Cargo graph change.
- **Evidence.**
  - Open SW-F327 PR #3264 (18 cards, "unify memory contracts and bounded device helpers") touches `image_jobs/krea_multiphase.rs`, `candle_memory_strategy.rs`, `generator_cache.rs`, `base_admission.rs`, `conditioning_gate.rs`, `moss.rs` and `instantid.rs` (`gh pr view 3264 --json files`).
  - #3258 touches the same lane files.
  - #3270 de-duplicates the worker's Cargo dependencies and adds a dependency-direction guard (cards SW-6812, SW-6817, SW-6824).
  - #3260 pins SceneWorks to the inference SW-F327 head.
- **Fix.**
  - Rewrite DEC-5's options as "engine-only items now (gen-core/backends behind unchanged worker-facing APIs); every ENG feature's SceneWorks consumer half is gated identically to APP".
  - In §9, mark each ENG item as `[INF]` or `[SW]`.
  - Move ENG-1 items 5–6 and the Krea worker-lane change behind the same SW-F327 gate, or give an explicit per-file absorption list for Michael to sign off.

**2. BLOCKER: DEC-5 understates the overlap. SW-F327 is already delivering parts of APPREG, ADM, LANE and TOOL, so baselines and items will be stale.**

- **Evidence (open PRs targeting `feature/sw-f327-code-review-remediation`).**
  - **#3268** deletes `apps/web/src/modelCatalog.js` (confirmed with `gh pr diff 3268`). That is TOOL-3's and APPREG-2's "web seed" deletion.
  - **#3264** covers:
    - SW-6659 (30 `apply_vram_cap(nvidia_vram_budget_gb…)` sites);
    - SW-6662 ("Four YuE/YuE2 admission modules duplicate tier enums, budget probes and evict credit");
    - SW-6753 (Mochi/SVD fit-gate twins);
    - SW-6754 (`hw.memsize` probes);
    - SW-6640 (MLX vs Candle floor semantics).
    These are ADM-2's "one budget vocabulary, probes" and parts of A13/A15/A16.
  - **#3270** delivers the cfg alias (SW-6677/6679 → LANE-5's "1,959 cfg predicates" baseline) and AST-backed architecture checks (SW-6686 → the subject of DEC-6).
  - **#3258** (SW-6642) builds a shared stream/heartbeat/cancel supervisor, which is LANE-4 territory.
- **Fix.**
  - Add a "SW-F327 overlap ledger" table to §7.5 mapping each of these cards to SGA requirement IDs: delivered by SW-F327 / absorbed / conflicting.
  - Capture §10 baselines *after* SW-F327 merges (see finding 30).
  - State that DEC-5's "absorb cards" option is moot for cards already in review. Those land first, and SGA rebaselines.

**3. MAJOR: MIG-2 and AP9 are violated by the plan's own cross-feature interim states.**

- **Problem.** MIG-2 says "Dual paths exist only inside a single feature, never across features". AP9 forbids "mechanism now, activation later". The plan nonetheless contains:
  - ENG-1 item 5: a "thin adapter, replaced in APP-3";
  - ENG-1 item 6: `lowResStart` through the legacy `ADVANCED_KEY_RULES` path, to be redone by APP-2's generated panel;
  - ENG-1 item 4: "capability export v2, first fields only", completed in ENG-3;
  - APP-1: "Ingest the engine export (v2 when ENG-3 lands; today's dumps until then)", i.e. two ingest paths across features;
  - APP-3: must ship before MEM-2 derived contracts (ENG-6), so it consumes hand contracts and is re-plumbed later (finding 4).
- **Fix.** Either:
  - state an explicit, decided exception class ("bridging adapters with a named removing feature, listed in a register"); be aware that memory note `owner_signature_registers_are_laundered_deferments` treats such registers as deferments; or
  - re-sequence so each consumer starts only after its producer is complete.

  In both cases, list every bridge in a table with the feature that deletes it.

**4. BLOCKER: §9.1 hides APP dependencies on late engine features. As drawn, the APP track cannot meet its own requirements.**

| APP feature | Requirement it must meet | Engine feature it actually needs | Shown in §9.1? |
|---|---|---|---|
| APP-3 | ADM-3 needs "the route's derived memory contract" (MEM-2) and a `RenderPlan` for all routes | MEM-2 lands in ENG-6; `RenderPlan` for all routes lands in ENG-2b | No; only ENG-3 is shown |
| APP-5 | TOOL-2: "does not re-implement per-family loads" | Family loaders disappear only in ENG-5 (WGT-1) | No |
| APP-2 | OPT-2: availability from the Feature export for every route | ENG-2b + ENG-3 | No |
| APP-4 | LANE-1 covers audio | Audio pipeline adoption is ENG-8 (last) | No; only ENG-3 is shown |

- **Consequence.** "App track 19–30 weeks, in parallel" (§9.1) is not credible. The realistic critical path runs to ENG-6 (month 7–10+) for APP-3 and APP-5.
- **Fix.**
  - Redraw §9.1 with these edges.
  - Either split APP-3 into "pipeline/types/stamp/refusal over today's contracts" and "derived-contract input" (ending the second half at ENG-6), or move APP-3 after ENG-6.
  - Recompute the calendar.

**5. MAJOR: DEC-12 and checklist item 8 — APP features over the ≤6-week window are not split.**

- **Problem.**
  - APP-4 is estimated at 5–8 weeks with no split.
  - APP-2 at 4–6 weeks is implausible. It has to:
    - migrate every option across ImageStudio (5,061 lines, 79 `useState`), VideoStudio (3,367), AudioStudio, TrainingStudio, the Simple studios and Character [G §5.1, §3.1];
    - add the validate endpoint and the upgrader corpus;
    - do all of this with a zero-removal control diff.
- **Fix.**
  - Split APP-2 into (a) `OptionSpec`, `validate()` and the upgrader, and (b) generated panels by studio.
  - Split APP-4 into (a) image plus event consumer and (b) video/audio plus routing table plus backend module.
  - Each split part ends on `main`.

**6. MAJOR: The pin and lockfile choreography between concurrent cross-repo features is unspecified.**

- **Problem.** SW-F327's SceneWorks branch pins the inference SW-F327 head (#3260). An SGA engine feature needs its own interim feature pins at the same sites plus `Cargo.lock`. FEATURE_DEVELOPMENT.md invariant 5 requires `main` to pin a commit reachable from inference `main`.
- **Fix.** Add to §7.5 an explicit ordering rule:
  - whichever inference feature lands first, the other merges inference `main` before its final pin;
  - `cargo metadata --locked` after every lock merge (memory note `merged_cargo_lock_must_pass_locked_metadata`);
  - name who resolves the pin conflict.

### B. Admission (ADM)

**7. BLOCKER: ADM-1 / PAT-9 / §8 "18 paths [H]" — the inventory is incomplete, so "each of the 18 inventoried paths is migrated, then deleted" can pass while admission paths remain.**

- **Evidence.** Production admission or budget code outside A1–A18 [F §4.1]:

| Location | What it does | Job kind |
|---|---|---|
| `image_jobs/moss.rs:8,34,45` | `moss_live_budget`, `moss_admit`, `moss_load_admitted`; fresh `nvidia_vram_budget_gb_fresh_blocking` probe | image |
| `sensenova_jobs.rs:190,242,515` | `sensenova_direct_admission`, both backends | VQA/interleave |
| `image_jobs/instantid.rs:290,307,394,405` | `instantid_memory_budget`, `instantid_validate_admission` | image |
| `iris_training.rs:92,203,287` | `iris_run_memory_refusal`, `iris_memory_refusal`, `iris_plan_memory_refusal` | training |
| `iris_restore.rs:502` | `admission_error` | upscale/restore |
| `depth_jobs.rs:617` | `admission_decision` | depth |
| API (`apps/rust-api`) | Calls worker admission code: `sceneworks_worker::iris_plan_memory_error`, `full_finetune_memory_error`, `iris_training_install_refusal` | API-side training admission |

  - F itself marks training admission "trainer-specific [M]" and Iris restore "[M]" (§1.2).
  - F §9 says the audio, vector, Iris restore/depth and SenseNova budgets/probes were "not read".
  - ADM-1 names training and upscale, but the inventory contains no training or upscale path.
- **Fix.**
  - Re-inventory by grep for budget probes (`nvidia_vram_budget_gb*`, `total_unified_memory_gb`, `host_available_bytes`, `mlx_tier_budget`, `live_request_budget`): 40 files today, 34+ production.
  - Change ADM-1/PAT-9 acceptance to "no budget probe or admission refusal outside the pipeline and `BackendBudget` impls (grep)", not "the 18 inventoried paths".
  - Downgrade §8's `[H]` to `[M]`.

**8. MAJOR: §4.3 "unpriced → backend `failure_posture` (MLX: RefuseIfUnpriced)" changes MLX behaviour on routes that admit unpriced today, which is a G7/PAT-8 regression.**

- **Evidence.**
  - `sensenova_jobs.rs` (macOS): if the manifest has no peak or the total is unknown, it admits (`if let (Some(predicted_peak_gb), Some(total_gb))` … else `Ok(Some(…))`).
  - `instantid.rs:285-288` documents "A host with NO readable budget yields a budget that admits exactly this request — the worker's standing 'no evidence never blocks' convention".
  - So posture is per route today, not per backend.
- **Fix.**
  - Make posture a property of (backend × evidence basis), declared per route or derived, not of the backend.
  - Add an ADM acceptance: the decision-diff shows zero newly refused cells relative to legacy unless each one is listed and approved.

**9. MAJOR: The "behaviour changes" in §4.3 are acknowledged generically, with no acceptance bound and no user-facing consequence analysis.**

- **Problem.**
  - "Ladder first, then floor, everywhere": Krea control is floor-first today (`krea_control_candle.rs:786` [F §4.7]). Ladder-first makes cheap impossible-load refusals pay the full ladder cost, and pre-load admission already took 95 s on Krea (comment cited [F §1.3 step 11]).
  - "Video admits in Plan and Run": video today admits only in-cache, after a cold load, with live measurements. A Plan-phase admission priced from packaged curves can refuse requests that are served today.
- **Fix.** For each row of the divergence table, add the measurable consequence (latency of refusal, refusal delta) and a requirement: "No previously admitted (route, tier, geometry, budget) cell is refused, except in a listed and approved set".

**10. MINOR: §4.3 "One probe on candle (cudarc), with nvidia-smi as a fallback only" has a subtle consequence. Confidence: low.**

- **Problem.** `cudarc` `mem_get_info` needs a driver context (cuInit). Today's off-Mac preflight is NVML-only (memory note `cuda_preflight_is_nvml_only_not_cuinit`), so the probe change moves driver-init failures to admission time. A probe failure is not a refusal and must not be classified as one.
- **Fix.** State the context-creation cost and error-classification rule for the probe.

### C. Registry (APPREG)

**11. BLOCKER: APPREG-1's three inputs (export, slim manifest, evidence store) cannot represent runtime-scanned, host-dependent and user-written model data.**

- **Evidence: [G §1.1, §1.3]**
  - ComfyUI external bases are synthesized by a disk scan (`apps/rust-api/src/external_base_models.rs`) with their own hand tables (`ANCHOR_SUBDIRS` `:66`, `required_text_encoders(family)` `:519`, `family_quant_runnable` `:508`).
  - Iris trained checkpoints are overlaid on every read (`overlay_trained_checkpoint_surface`).
  - User hides and imports live in `user.models.jsonc`.
  - OS projections: `retain_models_for_os`, `retain_downloads_for_os`.
  - Install-state projections: `installedChoices`, `variants`, `hasVariantMatrix`, decoder availability.
  - Credential-gated rows: `gated`, `credentialHost`.
  - Converted-model `modelPath` injection.
  - The plan's build step is "a Rust build-time generator plus a checked-in snapshot" (§4.1). That cannot hold any of these.
  - APPREG-3 only asserts "same registry path".
- **Fix.** Define `ModelSpec = static layer (build-time: export ⊕ manifest ⊕ evidence) ⊕ runtime layer (imports, external scans, trained overlays, user overrides, host/OS/install projections)`, with:
  - one merge rule;
  - where each runtime source lives (API vs worker);
  - an acceptance that covers an imported model, a ComfyUI external base and an Iris trained checkpoint.

**12. MAJOR: The §4.1 job carrier `{model_id, spec_revision, user_override_delta}` breaks for non-builtin models and introduces API/worker skew.**

- **Problem.**
  - Imported, external and trained entries have no builtin to apply a delta to.
  - The worker today reads limits, defaults and manifest memory facts from the `modelManifestEntry` carrier (e.g. `sensenova_manifest_peak_gb(&request.model_manifest_entry, …)`; `image_request.rs:239-251`).
  - Workers are separate processes that register over HTTP: RunPod runs one CUDA worker per GPU plus a CPU utility worker (`docs/deploy-runpod.md:11-12`).
  - Persisted job rows carry the carrier [G §6], and retry re-resolves it.
- **Fix.**
  - Specify how a worker obtains runtime-layer specs (a resolved spec in the claim response, or an API fetch by revision).
  - Specify behaviour on a `spec_revision` mismatch (refuse with a typed reason, never silently use a different spec).
  - Specify how old job rows with full carriers are upgraded.

**13. MAJOR: "SceneWorks reads only this export for engine facts" (REG-3, APPREG-1, §3.10) conflicts with the worker using the live linked engine (§4.4 `engine.compile`, `routing_table`). Two sources can disagree.**

- **Problem.**
  - Capability dumps are deliberately allowed to go stale relative to the pin. Commit `e14171984`; memory note `pin_bump_needs_an_off_mac_candle_capability_dump`.
  - The MLX and candle dumps are already at different revisions [G §7.2].
  - So the UI's availability (from the export) can disagree with what the worker's linked registry compiles or refuses.
- **Fix.** State which source is authoritative at runtime and how disagreement surfaces, for example: the worker refuses with a typed reason, and the UI treats the export as advisory. Note that this is a new drift class the old tables did not have.

**14. MAJOR: REG-3 / §4.1 — "export v2 checked in at each pin" makes the capability dump the sole availability source, but dumps are campaign artifacts with an off-Mac producer.**

- **Problem.**
  - FEATURE_DEVELOPMENT.md:152-154 puts capability dumps in "Measurement campaigns … run once, at the end of an epic".
  - The candle media dump can only be produced off-Mac on the Windows/CUDA box (memory note above).
  - ENG-1 item 6 ("web control gated by the export") and every Feature-applicability change therefore need a mid-feature re-dump on the CUDA host.
  - The SceneWorks producer is `sceneworks-worker --bin dump-engine-capabilities` (`memory-catalog-campaign.yml:911`), not "emitted from bundles" (§1.2, §3.10).
- **Fix.**
  - Make export v2 a build-time, hardware-free product: a pure function of the linked registry that runs on hosted CI for both backends' feature sets. Require this as part of REG-3, along with "no GPU or off-Mac step".
  - If that is impossible for candle on macOS, say so and add the CUDA host to every feature's critical path.

**15. MAJOR: APPREG-2 / C9 / §8 — "6–8 hand tables, target 0" understates the scope. The acceptance grep will fail on tables the doc never lists.**

- **Evidence.** [G §1.4, §2.1] lists tables APPREG-2 omits:
  - `STRICT_CONTROL_ENGINES`, `SDXL_CONTROL_MODELS`, `CANDLE_VIDEO_LORA_MODELS`, `CANDLE_IMAGE_CHECKS`;
  - `SENSENOVA_MODEL_IDS`, `QWEN_IMAGE_2_1_MODEL_IDS`, `FLUX3_TIERED_MODEL_IDS`;
  - `VIDEO_UI_MODES` and the web `VIDEO_MODES`;
  - `VIDEO_ENGINE_IDS`, `MacCapabilities`;
  - the ComfyUI assembly tables;
  - `lora_family.rs` `SIGNATURES` (`:1557`, a 7,960-line file);
  - web `WAN_MOE_PAIRED_LORA_MODEL_IDS`;
  - 91 + 50 routing predicate functions with 149 id-like literals;
  - `EXPECTED_*` test mirrors.
  
  G counts 109 core, 103 worker and 18 API `const` slice tables [M].
- **Fix.**
  - Replace "6–8" with a checked-in enumerated list (path:line), classified as "derive", "keep as product data" or "delete".
  - Make APPREG-2's acceptance that list, not an unbounded grep.

**16. MAJOR: APPREG-4 moves memory data out of the manifest, but the doc has no migration for consumers or for user-written manifests.**

- **Evidence.**
  - Scripts reading manifest memory keys: `report-staleness.mjs:66` (`memoryStrategyContract.implementations`), `stale-lane-report.mjs`, `extract-memory-anchors.mjs`, `check-tier-integrity.mjs`, `backfill-closure-digests.mjs`, `epic-20738-terminal-cuda-harness.mjs`.
  - Rust readers of manifest memory keys: `memory-adapter/bin/{mlx,candle,candle_wan_scail2}.rs`, `fit_gate.rs`, `mlx_fit_gate.rs`, `vram_gate.rs`, `conditioning_fit.rs`, `krea_control_fit.rs`, `admission_geometry.rs`, `memory_anchor.rs` (grep for `memoryStrategyContract|vramGbByTier|minMemoryGbByTier|turboFit|admissionGeometry|sequentialPeakGb|calibrations`).
  - `user.models.jsonc` imported entries carry the old schema (`schemaVersion: 1`).
  - Desktop seeding (`seed_builtin_manifests` Overwrite / SyncFromEmbedded) must ship the evidence store too.
- **Fix.** Add APPREG-5: "manifest schema v2 with an upgrader for `user.*.jsonc`; seeding ships the evidence store; consumer list migrated in the same feature".

**17. MAJOR: Evidence-store identity "keyed by (model, backend, tier)" contradicts the calibration identity that is actually used.**

- **Evidence.** Memory note `calibration_identity_must_be_keyed_per_provider_and_artifact_tier`: identity is (provider route, artifact-proven tier) per load shape. The anchor store keys on (modelId, backend, tier).
- **Problem.** Re-keying orphans existing evidence. Recapture is a campaign, and SW-F327 E7 says "No campaign solely for refactoring" [H §6.6].
- **Fix.** State that the evidence store preserves the existing identity keys verbatim (a format move only, no re-keying), and make this an explicit §7.4 rule-1 item for APP-1.

### D. Options, validation and versioning (OPT)

**18. MAJOR: OPT-1 — the `ADVANCED_KEY_RULES` seed is incomplete. Live `advanced.*` keys exist outside it.**

- **Evidence.** These are read via `advanced.get("…")` in production but absent from `workflow_share.rs`:

| Key | Read in |
|---|---|
| `enhanceTemperature`, `enhanceMaxTokens` | `apps/rust-api/src/lib.rs` |
| `useUncensoredEnhancer` | `video_jobs/ltx.rs` |
| `referenceStrength` | `routing/candle.rs` |
| `referenceImageShortEdge` | `video_request.rs:89` |
| `timestepShift` | `image_jobs/base.rs`, `sensenova.rs` |

  - Film plans carry `model.advanced.*` (`film_compile.rs:768,776`; `config/film-harness/courier-workshop/plan.*.jsonc`).
  - G's executive summary line "classifies every `advanced.*` key" is wrong.
- **Fix.** OPT-1 seeds from the union of `ADVANCED_KEY_RULES`, `TECHNIQUE_KEYS`, all production `advanced.get` literals and film-plan keys. Its acceptance includes "no `advanced.*` key read anywhere lacks an `OptionSpec`".

**19. MAJOR: OPT-5 and plan §4.2 cover only "phase editor, pose picker, layout editors" as structured renderers. Many structured or contextual shapes are missing.**

- **Evidence.** Missing shapes include `AdvancedShape::{StructuredPrompt, H3VedaSettings, MossImageOptions/Report/EditOptions/EditReport/Advanced, PromptBatch, SampleIndex, NumberPair, …}` (`workflow_share.rs:1297-1327`), the YuE2 lab (about 3.4k lines), the ideogram structured caption (774 lines) and the Character builder.
- **Problem.**
  - Editor/timeline context keys (`timelineAction`, `timelineContext`, `selectedPersonTrack`, `recipePresetId`, `presetMissingLoras`) are neither "what to make" nor "machine budget". `IntentClass` needs a third class (provenance/context).
  - Cross-option side effects are not expressible in `Predicate` availability: today the web clears PiD when upscale or Hires.fix is on (`ImageStudio.jsx:809-815`).
- **Fix.**
  - Enumerate every `AdvancedShape` with its renderer.
  - Add `IntentClass::Context`.
  - Add declared side effects or exclusions to `OptionSpec`.

**20. MAJOR: DEC-9 (server-evaluated validation) is justified only by the desktop topology.**

- **Problem.** RunPod serves the web and API through a public proxy (`docs/deploy-runpod.md:16-17`), and the doc reports proxy `524` timeouts (`:292`). Debounced per-change round trips over the internet will lag the Generate button and the disabled reasons.
  - Payloads can be large (poses, layout boxes, prompt batches).
  - Cold start: SW-F327 #3268 removes the offline seed, so validation has no client fallback.
- **Fix.**
  - Re-evaluate DEC-9 with RunPod latency as an input.
  - Consider a hybrid: server-evaluated availability and ranges, delivered once per (model, backend, mode) and cached; only cross-field rules round-trip.
  - State the offline/cold-start behaviour.

**21. MAJOR: OPT-6 versioning misses several persisted or external shapes and says nothing about forward compatibility.**

- **Missing shapes:**
  - film plans (`model.advanced`, `PLAN_SCHEMA_VERSION`);
  - training plans (`TRAINING_PLAN_VERSION`, techniques under `advanced`) [G §6];
  - the public HTTP DTOs (`ImageJobRequest` etc.) used by authenticated external callers and by MCP clients (`sceneworks-mcp` builds its own `advanced`, `server.rs:1461-1479`);
  - the external SceneWorks-Art "recipe-sharing platform" repo, which consumes share envelopes and recipes.
- **Forward compatibility.** `workflow_share.rs:3515` rejects newer envelope versions, so old desktop builds cannot read new shares. Downgrade or rollback (desktop updater, pinned RunPod tags) meets job rows with an unknown `requestSchemaVersion`.
- **Corpus.** "A corpus of real persisted payloads" implies real user data (prompts, paths).
- **Fix.**
  - Extend OPT-6 to film plans, training plans, the HTTP/MCP API and external consumers.
  - Decide writer behaviour: keep emitting v1-compatible shares through aliases, or accept older-client rejection.
  - Define downgrade behaviour.
  - Build the corpus from historical builder outputs (git history), not user data.

**22. MINOR: OPT-7 "zero `advanced.get` reads outside the upgrader" conflicts with film_compile/film_planner reading `advanced.steps` (`film_planner.rs:1376-1378`) unless film plans are in scope.**

- **Fix.** Scope film plans explicitly. REQ §4 says film is in scope "only where they consume … option facts", and this is such a place.

### E. Lane harness (LANE)

**23. MAJOR: LANE-1 and §4.4 classify only a handful of the 33 job kinds. The "one generation lane" boundary is undefined and internally inconsistent.**

- **Evidence.** F §1.2 lists 33 dispatched `JobType` arms.
- **Problems.**
  - PIPE-12 makes "interleaved text and image" (SenseNova) an engine output, but §4.4 puts VQA under `LaneHooks` (non-generation).
  - VectorGenerate (StarVector) is generation but is listed as non-generation.
  - ImageDetail, VideoExtend/Bridge, PersonReplace, VideoUpscale (SeedVR2 via `generate_video`), ImageUpscale (Generator/Transform), Iris restore, depth, segment and PromptRefine (`refine_model_cache`, evict-before-load) are unclassified.
  - Audio is in LANE-1 but its pipeline adoption is ENG-8 (finding 4).
- **Fix.**
  - Add a table of every `JobType` → generation lane / `LaneHooks` / utility (no lane), with admission and cache ownership for each.
  - Make LANE-1's acceptance "every `JobType` is in the table".

**24. MAJOR: Ownership of "register today's bespoke candle providers" is unassigned, yet APP-4 depends on it.**

- **Evidence.**
  - LANE-3 includes the bespoke candle providers.
  - APP-4 "Depends on: ENG-3 (all providers registered)".
  - ENG-3's slice list does not mention bespoke worker-side providers. Example: Kolors IP imports `runtime_cuda::providers::*` directly (`kolors_ipadapter.rs:311` [F §4.6]).
- **Fix.** Add an explicit ENG-3 (or ENG-2b) slice that registers every bespoke provider in the inference registry, with an acceptance criterion that no worker file imports `runtime_cuda::providers` or `runtime_macos::providers` directly.

**25. MAJOR: Hidden coupling — the API embeds and calls the worker. The doc's mitigation (R10 "APP-4 decouples lanes") has no requirement or end state.**

- **Evidence.**
  - `apps/rust-api/Cargo.toml:31` depends on `sceneworks-worker`.
  - The API calls `sceneworks_worker::{run_worker_loop, iris_plan_memory_error, full_finetune_memory_error, iris_training_install_refusal, prompt_enhancement_max_tokens, iris_restore, flux3_receipted_snapshot, …}`.
  - `backend-candle` passes through to the worker (`apps/rust-api/Cargo.toml:66`).
  - SW-F327 SW-6824 flags exactly this.
- **Fix.** Add a requirement on the API→worker dependency end state:
  - validation and admission math live in core;
  - the API links no lane code;
  - the in-process worker mode is preserved for desktop.

  Coordinate it with SW-6824.

### F. Tooling and gates (TOOL)

**26. BLOCKER: TOOL-1 omits the live backend-capability-matrix gate, which fires on nearly every APP PR.**

- **Evidence.**
  - `crates/sceneworks-core/src/jobs_store/routing/matrix.rs` `include_str!`s 61 sources and byte-compares `config/backend-capabilities/matrix.json` (`checked_in_matrix_matches_live`, `:3882-3912`).
  - The 61 sources include worker `lib.rs`, `image_jobs/base.rs`, every `video_jobs/*`, `audio_jobs.rs`, `upscale_jobs.rs`, `training_jobs.rs`, API `jobs.rs`/`generation.rs`/`lib.rs`/`dto.rs`, web `imageJobRequest.js`, `imageJobAdvanced.js`, `macGating.js`, `VideoStudio.jsx`, and the engine capability files.
  - It is accompanied by `exceptions.json` ("approved asymmetries"; memory note `owner_signature_registers_are_laundered_deferments`).
  - APP-1..4 rewrite every one of these files.
- **Fix.** APP-1 item 0: decide the fate of `routing/matrix.rs`, `matrix.json` and `exceptions.json` (delete, or regenerate from `ModelSpec` with no source digests) before any other APP item. Name this in TOOL-1.

**27. MAJOR: TOOL-2 / APP-5 — the memory-adapter scope and validation are understated, and its campaign coupling is ignored.**

- **Evidence.**
  - Beyond `bin/mlx.rs` (34,023 lines) and `bin/candle.rs` (13,140 lines), the adapter has: `mlx_ltx25.rs` 2,040, `flux3_common.rs` 2,061, `mlx_wan_scail2.rs` 1,673, `candle_wan_scail2.rs` 1,507, `mlx_iris.rs` 1,048, `candle_iris.rs` 894, `flux3_resource.rs` 864, `mlx_supra2.rs` 583, `candle_supra2.rs` 495, and the flux3 self-test bins.
  - `memory-catalog-campaign.yml:626,880` builds `memory-mlx-adapter` and `memory-candle-adapter` by name.
  - A rewritten harness can only be validated by running it, which is a campaign: real model execution, Mac GPU, Michael's loan rule.
- **Fix.**
  - List every bin and workflow consumer.
  - Add an APP-5 terminal item: one dispatched equivalence run (old harness vs new) on `nax-macos-2` and the CUDA host, with the duration stated up front. Or decide explicitly that the harness is rewritten without measurement proof.

**28. MINOR: §4.5 "the memory matrix, if still wanted" is an undecided scope item hidden in prose.**

- **Problem.**
  - AP11 says "No new generated matrices".
  - `generate-memory-matrix.test.mjs` and `matrix-removed-sources.test.mjs` run in `npm run check` (`package.json` `scripts.check`).
  - Memory notes `generate_memory_matrix_check_is_not_a_pure_check` and `memory_matrix_is_fingerprinted_on_eight_sources`.
- **Fix.** Make it a DEC (delete vs regenerate from typed dumps) with an owner feature.

**29. MAJOR: DEC-6's evidence is incomplete, and the "No new gates" memory is mis-scoped in the recommendation.**

- **Problem.**
  - The memory note's ban list includes "a static-analysis check · a lint rule · a workspace invariant · an anti-drift guard". The note also says "Do not re-argue this in a later session".
  - The more direct precedent is SW-F327 E3 / #3270 SW-6686 ("compiler/AST-backed checks cover dispatch exhaustiveness, capability parity, dependency direction, cfg rules and new helper copies"), which Michael approved for SW-F327 [H §1.4]. DEC-6 does not cite it.
  - Several acceptance criteria depend on DEC-6 without saying what happens if it is rejected: PIPE-8, BACK-2 and §1.1's "three new rules".
- **Fix.**
  - Cite the SW-F327 E3 precedent.
  - Propose reusing `architecture_tests.rs` (no new lane).
  - Give a fallback for each affected acceptance criterion if DEC-6 is "forbid": a one-time grep recorded in the terminal item.

### G. Process and validation infrastructure

**30. MAJOR: DEC-14 / MIG-3 / plan §8.2 — the named Tier 1 workflow does not do what the plan says. The per-batch cadence also conflicts with FEATURE_DEVELOPMENT.md.**

- **Workflow mismatch.**
  - Inference `model-inference-proofs.yml` maps `mlx` → GitHub-hosted `macos-15` and `candle-cuda` → `[self-hosted, windows, cuda, real-weights]` (`scripts/ci/model_inference_proofs.py:42-47`), with `max-parallel: 1`.
  - It has no `nax-macos-2` MLX route. The `runner="nax"` option matches the `nax` label, which per memory note `use_nax_macos_2_via_workflows_not_the_dev_mac` is **the dev Mac** (`nax-macos`).
- **Runner inventory (live, 2026-11-03).** All six runners are in org group `self-hosted-gpu` (selected visibility, SceneWorks + inference + ChatWorks); no repo-scoped runners.

| Runners | Count | `real-weights` label | Idle at query time |
|---|---|---|---|
| `cuda-windows`, `cuda-windows-2`, `cuda-windows-3`, `cuda-windows-4` | 4 CUDA listeners (likely one host, per H) | `cuda-windows`, `cuda-windows-2` only | 3 of 4 |
| `nax-macos`, `nax-macos-2` | 2 macOS | both | both |

- **Cadence conflict.** FEATURE_DEVELOPMENT.md:193 says "Do not define a per-story validation matrix", and the project CLAUDE.md marks per-story validation as obsolete. DEC-14 per-batch dispatched runs are such a matrix; the doc does not acknowledge the conflict.
- **Goldens.** Bit-identity "vs pre-migration" needs either goldens (memory note `never_commit_a_machine_dependent_value_in_a_golden`) or both SHAs run on one runner in one job; neither is specified.
- **Fix.**
  - Specify the workflow change: pin MLX Tier 1 to an `rw-*` label on `nax-macos-2`.
  - Specify the old/new same-runner protocol.
  - Restate DEC-14 as an explicit exception to FEATURE_DEVELOPMENT.md, with its cost in runner-hours.

**31. MAJOR: MIG-3 "every consolidation PR shows its Tier 0 checks green" is impossible for candle-gated code in feature CI.**

- **Evidence.**
  - The SceneWorks `check.yml` `candle` job only typechecks: "It cannot RUN candle tests" (`check.yml:470-473`).
  - `windows-candle.yml` runs only on PRs to `main` (FEATURE_DEVELOPMENT.md:375).
  - Admission paths A5–A14 and most bespoke lanes are candle-gated.
  - REQ §9.7 requires "a candle-cfg test execution", but names no runner. The Docker route risks memory note `agent_docker_builds_can_take_down_michaels_containers`.
- **Fix.**
  - Require the pure admission and routing core to be cfg-free in `sceneworks-core`, so decision-diffs run in `parity` on Linux.
  - Budget one `windows-candle.yml` dispatch per APP batch, and state that this is allowed.

**32. MAJOR: Tier 2 per feature for refactor-only features conflicts with SW-F327 E7 ("No campaign solely for refactoring", [H §6.6]).**

- **Problem.** PERF-1's ±2% s/step "measured in the feature-end proof runs" against a baseline captured once at program start (§10) is not comparable across 7–10 months of driver, OS and runner changes.
- **Fix.**
  - Add a DEC on Tier 2 for refactor-only features.
  - If kept, measure A/B (pre-feature `main` vs feature head) on the same runner in the same window. Do not compare against the program-start baseline.

**33. MINOR: REQ §9.5 states "One CUDA host and two Macs" as a fixed fact.** The user's standing instruction is to discover the live pool (both scopes, groups, labels) at planning time. The `real-weights` label exists on only 2 of the 4 CUDA listeners. **Fix:** replace it with "discover at feature planning; record registered, label-compatible and idle counts separately".

### H. Metrics and PATs

**34. BLOCKER: PAT-1 contradicts ENG-1, and §10 counts files from "the diff of the feature that introduces it".**

- **Problem.** ENG-1 introduces low-res start and by design edits today's admission paths (item 5), core validation, `ADVANCED_KEY_RULES` and the web (item 6), plus `krea_multiphase.rs`. Every one of those builder files is also digested by the matrix gate (finding 26). PAT-1 ("zero SceneWorks lane, route or admission files") can never pass on that diff.
- **Fix.** Redefine PAT-1 as a *second* recipe Feature added after APP-1..4 (for example a new guidance trick, §6.4), measured on its own diff. Keep low-res start as FEAT-1 with its own acceptance criteria.

**35. MAJOR: PAT-4's method is undefined.**

- **Problem.**
  - §10 says "about 25 facts" but names 12.
  - "Grep recipe that counts encodings" cannot distinguish a declaration from a consumer or a derived copy.
  - A product menu legitimately intersects an engine limit (§4.1 "engine hard ∩ product menus"), so some facts have two legitimate homes.
- **Fix.**
  - Check in the full fact list now.
  - Define "encoding" as "a source location where the value or id-set is written as a literal".
  - List the allowed exceptions (product menus, upgrader alias tables).

**36. MAJOR: PAT-8's control inventory has no method.**

- **Problem.**
  - Controls today are hand JSX gated by about 10 sources [G §5.2].
  - A "before" inventory needs an instrumented render of every (model, backend, mode) combination, which no tool does today.
  - "Zero removed controls" does not cover behaviour: defaults, clamp, persistence key, emit-when-default rule, tooltips, mutual-exclusion side effects.
- **Fix.** Specify the inventory tool and its output schema in §10, and extend the diff to per-control behaviour fields.

**37. MAJOR: PAT-10 turns §8 into a gate, which contradicts "goals, not gates" (§8, §10, AP11). The SceneWorks target is also not supported by G.**

- **Arithmetic.** The −30% to −45% target on about 950k lines means deleting 285k–427k lines. G's itemized slimming adds up to roughly 80–90k:

| Area | Estimated deletion |
|---|---|
| routing (−40–60% of 23k) | 9–14k |
| `models.rs` (−30–40% of 18.8k) | 6–7.5k |
| drift tooling | 10–15k |
| memory-adapter | about 47k (plan §4.5) |

  Even deleting 60% of the worker's lane and admission files (about 197k) brings the total to about 200k, roughly 21%. Characterization-first (AP10) also adds tests.
- **Fix.**
  - Drop PAT-10, or make it "census recorded".
  - Re-derive the SceneWorks target bottom-up per area, using G §8's per-area table and its `[L]` labels.

**38. MAJOR: PAT-3 ("new model is data + graph") omits mandatory non-code inputs.**

- **Omitted inputs.**
  - Measured evidence: on MLX, unpriced requests refuse (§4.3). Memory note `ios_evidence_is_two_anchors_plus_derivation_not_a_grid` makes this two anchors, which is a measurement run.
  - LoRA family signatures (`lora_family.rs` `SIGNATURES`).
  - Converter registry entries (`NATIVE_CONVERTERS`).
  - `ui.promptGuide` for picker entries (`check-scaffold.mjs:377`).
  - Web label tables for new samplers.
- **Fix.** List them as allowed inputs, or make each one derived.

**39. MINOR: PAT-5, APPREG-2 and §8 disagree on the model-id literal rule.**

- PAT-5 says "outside manifest, route records and tests … in conditionals".
- APPREG-2 says "only the manifest and tests", with any literal line counting.
- §8 says "about 0 outside manifest".
- None exempts upgrader alias tables, film-harness configs, MCP defaults, or family-specific product modules (`moss_image.rs`, `supra2.rs`, `h3_veda.rs`, the YuE2 lab).

**Fix:** pick one rule (conditionals only), with an enumerated exemption list.

### I. Missing app-side requirements

**40. MAJOR: Consumers outside the two repositories are absent.**

| Consumer | Dependency on SGA work |
|---|---|
| `SceneWorks/SceneWorks-mobile` (iOS app) | "real on-device engine (Rust/MLX behind an FFI) arrives later", i.e. a future `runtime-ios` consumer |
| `SceneWorks/ChatWorks` | Depends on inference by tag `runtime-2026.10.0` (`Cargo.toml:23-24`, candle crates). Affected by ENG-4/BACK-1 and XL-2 |
| `SceneWorks/SceneWorks-Art` | "recipe-sharing platform", which consumes share envelopes and recipes (OPT-6) |

**Fix:** add a consumer-contract section (runtime tag compatibility, envelope compatibility) and make it part of each affected feature's acceptance.

**41. MINOR: No requirement for refusal and error UX.**

- **Problem.**
  - ADM-4 and HOOK-4 produce typed refusals, but nothing requires the web or MCP to render them. Today, raw engine strings reach `QueueScreen.jsx:75` (memory note `cuda_preflight_is_nvml_only_not_cuinit`).
  - No requirement keeps telemetry and observability consumers working (`docs/observability.md`, `post_generation_metrics`) when ADM-5 changes stamp fields.
  - `regenerateImage.js:33` deletes `advanced.memoryStrategy` by name, so the stamp key name is a replay contract.
- **Fix.** Add OPT/ADM requirements covering refusal rendering, telemetry field stability and preservation of the stamp key.

**42. MINOR: Accessibility and Simple UI.**

- **Problem.**
  - Generated panels (OPT-5) need an a11y floor at least equal to today's (labels, keyboard, titles).
  - The Simple studios (`simple/`, 11.6k lines) and the Character builder are not mentioned in OPT-5 or APP-2.
- **Fix.** Add these to OPT-5's scope and acceptance.

### J. Internal inconsistencies (REQUIREMENTS vs IMPLEMENTATION_PLAN)

**43. MINOR: Inconsistencies, with the fix for each.**

1. **Tier names.** REQ MIG-3 defines Tier 0 and Tier 1 only. The plan's §7.2 and §8.2 add Tier 2, and REQ MIG-4 describes it without naming it. **Fix:** name Tier 2 in REQ.
2. **Function-family counts.** LANE-1's acceptance uses 56/64/53/43/49. F §0 reports 62/68/47/70, and a live grep gives `*_available` 64, `raw_settings` 68, `ensure_*` 48, `generate_*stream` 70. **Fix:** use grep-defined patterns, not frozen counts (memory note `pinned_population_assertions_are_frozen_corpus_gates`).
3. **DEC-5 trigger.** DEC-5's options cite "E5/E6", the recommendation cites only the splits, and REQ §9.8 says "structural splits". SW-F327 E5 (twin consolidation) is the overlapping one [H §4.1]. **Fix:** state the trigger as specific SW-F327 items or PRs merged to `main`.
4. **DEC numbering.** DEC-14 precedes DEC-13 in REQ §10, and R9 says "DEC-1…14". **Fix:** renumber.
5. **Export authority.** "Export is the only engine-fact input" (REQ REG-3, PLAN §3.10) contradicts PLAN §4.4's worker using the live engine (finding 13).
6. **Hires.fix ownership.** HOOK-2's acceptance requires Hires.fix as a Feature, but no ENG or APP feature in §9 owns moving the worker-side `HiresFixPlan` (`image_jobs/base.rs:8390-8420`) into the engine. **Fix:** assign it, likely to ENG-2a.
7. **Confidence label.** §8's "Admission paths 18 [H]" contradicts F's own "[H existence / M internals]" and F §9 (finding 7).
8. **Pin sites.** REQ §9.3 says "≥8 sites" (from H), while G §7.2 says "five sites" in the same evidence set. **Fix:** note that H supersedes G.

---

## Things that are right and should not change

- **Characterization tiers.** No model execution in PR CI (Tier 0) and dispatched Tier 1/2 matches memory note `real_model_execution_is_not_normal_ci`. Nothing listed in Tier 0 executes a model forward.
- **A typed `ModelSpec` with one merge rule.** This fixes the verified shallow-vs-six-key merge split (`manifest.rs:324-361` vs `models.rs:12402-12436`).
- **`ADVANCED_KEY_RULES` as the `OptionSpec` seed**, extended (not replaced), including the WhatToMake/MachineBudget split that the share registry already uses.
- **The admission contract.** Adopting F §4.6 (two phases, `BackendBudget`, typed refusal, one stamp) is right. Stamping is genuinely hand-written in 12 files today (verified).
- **Lane template.** Using `generate_video_using` plus `CandleStrictControl` as the template, and keeping `run_utility_job_inner` as the outer harness.
- **Generate → assert equal → switch → delete** as the cutover recipe for each hand table.
- **DEC-11 (the CodeGraph-key policy PR first) is correct.** Inference `main`'s `feature_epic_policy.py:29-37` accepts only `sc-` names, and the fix exists only on `feature/sw-f327-code-review-remediation`.
- **Process rules restated correctly:** merge-commit, `strict:false`, no merge queue, ≤2-round whole-branch review, 8–10 subagent cap, cleaning finished `target/` directories, and the dev Mac as a loan.
