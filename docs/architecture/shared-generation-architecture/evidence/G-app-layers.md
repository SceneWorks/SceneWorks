# G: The app layers above the worker. Registry, validation and the advanced-option pipeline

**Baseline:** SceneWorks `origin/main` at `a1d95e2c4`, read-only worktree
`/Users/michael/Repos/SceneWorks/.claude/worktrees/multi-resolution-inference-848ec4`.
All paths in this document are relative to that root. Nothing was built or run. Counts come from
`git ls-files`, `grep` and `python3` over the checked-in sources.

**Confidence tags:** **[H]** means I read the code at the cited lines. **[M]** means the claim rests on
grep or structure, or on reading part of a function. **[L]** means inference or extrapolation.

**Count discrepancies against the prior audit facts:**
- `IMAGE_MODEL_CAPS`: I count **63** `ModelCaps::new(` rows in `crates/sceneworks-core/src/jobs_store/routing/catalog.rs:837-1188`. The audit had 77. The audit probably counted comment or test mirrors. [M]
- `CANDLE_IMAGE_ROUTES`: I count **34** `CandleImageRoute {` literals in `routing/candle.rs:127-329`. The audit had 36. [M]
- `MODEL_TABLE`: I count **61** `ModelRow {` rows in `crates/sceneworks-worker/src/engines.rs:37-845`. The audit had 62. [M]
- These match the audit: `VIDEO_MODEL_CAPS` (14), `CANDLE_POSE_MODELS` (12), `RULES` (85), manifest models (143), and the 79 non-test `sceneworks_worker::` refs in the API. [H]

---

## 0. Executive summary

1. **Model knowledge has no typed home.** Every layer handles a model as `serde_json::Value`. The
   typed `ModelManifestEntry` (`crates/sceneworks-core/src/contracts.rs:1838-1866`) is used only by
   a fixture round-trip test (`crates/sceneworks-core/tests/contract_roundtrip.rs:128-150`). Its
   `ModelCapability` vocabulary (`contracts.rs:690-701`: `image_edit`, `person_replace`,
   `video_extend`) has already drifted from the data, which uses `edit_image`, `replace_person` and
   `extend_clip`. An `Unknown(String)` fallback in the `string_enum!` macro (`contracts.rs:77-100`)
   hides the drift. [H]
2. **The manifest is about 58% generated data posing as hand-authored data.** `mlx.memoryStrategyContract`
   and `candle.memoryStrategyContract` make up 57.8% of the manifest's JSON bytes. Both are projections of
   the engine dumps, written into marked regions by `scripts/generate-manifest-memory-declarations.mjs:1-36`.
   Comments add another 28% (593 KB of the 2.12 MB file). Hand-authored product facts are a minority of
   `config/manifests/builtin.models.jsonc`. [H]
3. **There are two merge semantics for one model entry.** The catalog uses shallow `merge_entries_by_id`
   (`apps/rust-api/src/manifest.rs:324-361`). The worker carrier uses `merge_model_manifest_entry`, which
   merges six nested keys one level deep (`apps/rust-api/src/models.rs:12402-12436`). A user override of
   `ui.*` or `limits.*` therefore means different things in `/models` and in the job. [H]
4. **A derivation precedent already exists and works.** Imported-model routes come from the checked-in
   engine facts (`routing/catalog.rs:1378-1440`). So do live-preview support
   (`crates/sceneworks-core/src/preview_support.rs`, `config/manifests/builtin.preview-support.jsonc`)
   and decoder options (`crates/sceneworks-core/src/decoder_support.rs`). Builtin routes are still
   hand-tabled. The engine descriptors in `config/engine-capabilities/runtime/capabilities.*.json` already
   carry `conditioning`, `supports_lora`, `supports_lokr`, `supported_quants`, `supports_preview`,
   `supports_prompt_enhancement`, `samplers`, `schedulers` and `supported_guidance_methods`. The manifest
   and the code tables restate all of these. [H]
5. **A nascent option registry also exists.** `ADVANCED_KEY_RULES` (`crates/sceneworks-core/src/workflow_share.rs:1411-1833`,
   76 entries) classifies every `advanced.*` key for sharing. A lint walks 17 registered web builders
   (`workflow_share.rs:655-934`). It knows key, shape, source builder and share disposition. It does not
   know type or range, applicability, default or UI metadata. Those live in 4–9 hand-written places per
   option (see §3). [H]
6. **Validation is layered and repeated.** Hires.fix applicability appears in roughly 9 places across
   web, core, API, routing and worker (§3.3). The prompt-enhancement temperature cap `2.0` is defined
   twice (`apps/rust-api/src/lib.rs:5160`, `crates/sceneworks-worker/src/image_jobs.rs:260`). The
   enhance-prompt model allowlist is hard-coded in the API (`lib.rs:5336`), while the manifest
   (`ui.promptEnhance`) and the engine descriptor (`supports_prompt_enhancement`) each say the same
   thing separately. The API's retry and duplicate path (`apps/rust-api/src/jobs.rs:924-1265`) re-runs
   only part of the create-path validation (§4.4). [H for the facts, M for the gap]
7. **Job payloads are unversioned.** There is no payload or request schema version anywhere (§6). Persisted
   recipes (`Recipe.raw_adapter_settings`, `contracts.rs:1568-1579`) and the workflow-share envelope (v1,
   `workflow_share.rs:121`) are the replay surfaces that constrain any change to request shape. [H]

---

## 1. The manifest

### 1.1 Files, loading and embedding

| File | Lines | Role |
|---|---:|---|
| `config/manifests/builtin.models.jsonc` | 31,519 | 143 models, `schemaVersion: 1` |
| `config/manifests/builtin.loras.jsonc` | 339 | builtin LoRA catalog |
| `config/manifests/builtin.recipe-presets.jsonc` | 5 | **empty** `presets: []` |
| `config/manifests/builtin.styles.jsonc` | 1,454 | generated from `documents/style.txt` (`npm run gen:styles`) |
| `config/manifests/builtin.control_overlays.jsonc` | 57 | control overlays |
| `config/manifests/builtin.preview-support.jsonc` | 327 | **generated** by `apps/web/scripts/generate-preview-support.mjs` from engine facts |
| `packages/schemas/model-manifest.schema.json` | 2,428 | authoring schema with 33 `$defs` |

Model types in the data: image 61, utility 52, audio 15, video 13, vector 2. [H] The typed
`ModelKind` enum (`contracts.rs:681-688`) has no `vector`. The API's `ALLOWED_MODEL_TYPES`
(`apps/rust-api/src/models.rs:14`) is `image|video|audio|utility`. [H]

**Embedding and seeding.**
- `crates/sceneworks-core/src/builtin_manifests.rs:23-47`: `BUILTIN_MANIFESTS` `include_str!`s five manifests. `preview-support` is not among them; core embeds it separately. [H]
- `seed_builtin_manifests` (`builtin_manifests.rs:284-302`) writes them to `config_dir/manifests` under `SeedMode::{Overwrite, SyncFromEmbedded, IfMissing}` (`:218-258`).
- Desktop uses `Overwrite`. The API uses `SyncFromEmbedded`, rewriting when the bytes drift. User customisations go in `user.*.jsonc` (`:220-247`). [H]
- `builtin_model_entry(id)` (`:51-70`) is a separate lazily parsed lookup into the embedded copy. [H]
- `builtin.models.jsonc` is `include_str!`-embedded at about 46 sites in 21 files. Most are `#[cfg(test)]` drift guards in the worker, e.g. 16 in `crates/sceneworks-worker/src/mlx_fit_gate.rs` and 4 in `memory_route_registry.rs`. Runtime embeds are `apps/rust-api/src/models.rs:12355` (`embedded_builtin_catalog_entry`), `builtin_manifests.rs`, `routing/matrix.rs:24`, and `video_request.rs` (one site). [M; test-only status checked for 8 worker files by locating the preceding `#[cfg(test)]`]

**Loading in the API.**
- `load_manifest_entries` (`apps/rust-api/src/manifest.rs:265-313`) reads one manifest field. It strips JSONC with `sceneworks_core::jsonc::strip_jsonc_comments` and caches by `(path, field, mtime_ns, size)` in `state.manifest_cache`. Parsing is untyped `Vec<Value>`. [H]
- `merged_model_manifest_entries` (`models.rs:12097-12110`) and `load_model_catalog_inputs` (`models.rs:12112-12159`) merge `builtin.models.jsonc` with `user.models.jsonc` through `merge_entries_by_id` (`manifest.rs:324-347`).
  - That merge is shallow: `merge_object` replaces top-level keys wholesale (`manifest.rs:349-361`). [H]
  - It also applies `iris_training::overlay_trained_checkpoint_surface`, `retain_models_for_os` (`models.rs:12456-12495`; withdraws video products per OS), `inherit_provider_decoder_downloads` (`models.rs:11563`) and `retain_downloads_for_os`. [H]
- `build_model_catalog_snapshot` (`models.rs:12168-12256`) runs `apply_model_catalog_entry` (`models.rs:11344`) per model inside a bounded blocking sweep. It merges ComfyUI external base models scanned by `external_base_models::scan_external_base_models`, applies size fields, and sorts by `(type, name)`. Results are cached through `model_catalog_snapshot` (`models.rs:4977`). [H]
- `resolve_model_manifest_entry` (`models.rs:12266-12347`) builds the worker-facing entry. It:
  - forwards synthesized `external_base_*` rows (`:12278-12296`);
  - uses `merge_model_manifest_entry` (`:12402-12436`), which does a one-level-deep merge of `paths/resources/defaults/limits/loraCompatibility/ui`. The comment at `:12396-12401` says this is deliberately deeper than the catalog merge;
  - falls back to the compiled-in manifest (`embedded_builtin_catalog_entry`, `:12349-12366`);
  - stamps `catalogScope` (`:12333-12339`);
  - injects `modelPath` for converted models (`inject_converted_model_path`, `:12374-12400`);
  - applies `decoder_support::apply_to_model_entry` and decoder availability (`:12341-12345`). [H]
- The resolved entry is injected into the job payload as `modelManifestEntry`. Audio voice clones also get `baseModelManifestEntry`, and some paths get `modelManifestEntries`. Raw `/jobs` callers have these keys stripped as "private carriers" (`apps/rust-api/src/jobs.rs:101-109`, `:962`, `:1246`). The worker reads limits and defaults from that carrier; see `ImageRequest::from_payload` (`crates/sceneworks-core/src/image_request.rs:239-251`). [H]

**Catalog projection: fields the API synthesises onto each `/models` row.** Each one is a further source of UI truth.
- `gated` and `credentialHost` (`apply_gating_fields`, `models.rs:9977-10004`).
- `minQualityTier`, `macSupport` and `candleSupport` (`apply_mac_and_mlx_fields`, `models.rs:11013-11175`). `macSupport` comes from `routing::catalog::model_mac_support` (`catalog.rs:108-125`), which probes the scheduler predicates with synthetic payloads (`image_model_mac_support`, `catalog.rs:127-202`; `probe_payload`, `:88-96`). [H]
- `variants` and `hasVariantMatrix` (`apply_variant_fields`, `models.rs:10006`; `model_variant_states`, `:9729`).
- Imported-provider surfaces (`apply_imported_provider_surface`, `:10451`; `apply_imported_lora_advertisement`, `:10278`; `project_imported_operation_surface`, `:10839`).
- Decoder availability (`apply_decoder_availability`, `:11652`), `installedChoices` and `preview.byBackend`. [M]

### 1.2 Schema structure: every per-model key group, classified

The schema root is `{$schema, schemaVersion, models[]}` with `additionalProperties:false`. A model
item has 45 properties and requires `id, name, family, type`. [H; python dump of the schema]

The table below groups the keys by who actually owns the fact. "Engine" means the inference
descriptor or registry publishes it today, in `config/engine-capabilities/runtime/capabilities.*.json`
under `snapshot.generator_capabilities[]`. Those fields are `id, family, backend, modality,
conditioning, supports_lora, supports_lokr, supported_quants, supports_preview,
supports_prompt_enhancement, samplers, schedulers, supported_guidance_methods`. [H]

| Key group | Sub-keys (schema) | # models carrying it | Owner today | Should be owned by |
|---|---|---:|---|---|
| identity | `id,name,type,family,adapter` | 143 | manifest | product registry (`adapter` is also `ModelRow.adapter_label`, `engines.rs:33`) |
| `capabilities[]` | untyped strings, 21 distinct tokens | 128 | manifest | **derive** from descriptor `conditioning` + route mappings (`runtime/*.json.modelMappings`, `videoModelMappings`); keep a product allowlist |
| `downloads[]` (+`paths`, `conditionalComponents`, `resources`, `quantization`, `precisionFloors`) | repo, revision, files, platforms, coRequisite, componentId, sha/size, variant footprints | 143 / 134 / 3 / 8 / 3 / 6 | manifest | product registry (distribution facts); `precisionFloors` is drift-checked against the provider (`engines.rs:1645`) |
| licensing | `license, licenseUrl, licenseNotice, requiresLicenseAcknowledgment, inheritedLicenseAcknowledgments, modelProvenance, nonCommercial, commercialUse, gated, credentialHost` | 79 / 10 / 7 / 14 / 51 … | manifest | product registry |
| product flags | `recommended, autoDownload, macOnly, componentOnly, experimental, breaking, promptless, structuredPrompt, supportsAlphaOutput, captionStyle` | 11 / 50 / 47 / 27 / … | manifest | product; `macOnly` (47) duplicates route facts for image models |
| `defaults` | `count,duration,fps,guidanceScale,quality,resolution,steps,sampler,scheduler,schedulerShift` | 100 | manifest; **also** `ModelRow.default_steps/default_guidance` (`engines.rs:29-31`) | product defaults; engine defaults as fallback |
| `limits` | `resolutions,count,durations,fps,hardMax/MinDuration,steps,hardMin/MaxSteps,maxPixels,requiresDimensionsMultipleOf,min/maxDimension,maxReferenceAssets,recommendedMaxDuration,maxSourceClipAssets,maxReferenceAudioAssets,maxCombinedReferenceAssets,samplers,schedulers,guidanceMethods` | 100 | manifest (+ `mlx.limits`, `candle.limits` overrides) | **split.** Product menus (resolutions, durations) stay. Engine envelopes (min/max dimension, stride, max references, steps hard bounds, samplers/schedulers/guidance) come from descriptors |
| `ui` (28 sub-keys) | presentation: `label, attribution, description, durationHint, promptHint, defaultPrompt, defaultNegativePrompt, recommendedFor, promptGuide` | 143 | manifest | product copy |
| `ui` capability flags | `img2img, multiReference, sourceWithMultiReference, scail2MultiReference, controlModes, viewAngles, poseLibrary, poseControlScale, controlOverlay, precisionToggle, promptEnhance, hideReferenceStrength, controlScale, identityStructure, img2imgStrength, textStyleGain, variationStrength, editReferences, pid, referenceStrengthDefault` | 1–17 each | manifest | **option registry** (applicability). Most restate engine conditioning or descriptor flags, e.g. `ui.promptEnhance` (3 models) = `supports_prompt_enhancement` |
| `image` | `supportsGuidance, supportsNegativePrompt, negativePromptRequiresGuidance, supportsSchedulerShift, supportsCfgInterval, captionOverflowPolicies, patchGridVisibleBelow, maxPromptBatchImages, maxPromptTokens, nativeCountBatch` | 29 | manifest | engine descriptor (`engines.rs:11-13` already reads `supports_guidance/negative_prompt` from the descriptor and dropped the duplicated row flags) |
| `video` | `supportsGuidance, supportsNegativePrompt` | 5 | manifest | engine |
| `audio` | `voices, languages, sampleRates, maxDurationSecs, editModes, …, conditioning` (17) | 15 | manifest | engine (audio descriptors exist: `config/engine-capabilities/audio/`) |
| `vector` | `acceptsTextGuidance, maxNewTokens, maxSvgBytes, maxWallTimeMs, deviceAdmission, providers` | 2 | manifest | engine + product |
| `loraCompatibility` | `families, types, baseModel, baseRevision, adapterKinds, irisTask` | 88 | manifest | product (family) + engine (`supports_lora/lokr`) |
| **memory / calibration** | `minMemoryGb`; `mlx.{minMemoryGb, minMemoryGbByTier, measuredSiblingActiveDiskFraction, memoryStrategyCapabilities, memoryStrategyStructuralExemptions, calibrations, supportsSequentialOffload, memoryDeclarationWithhold, memoryStrategyContract, standardTierLayout, denseTextEncoderTier}`; `candle.{minMemoryGb, minMemoryGbByTier, vramGbByTier, vramMeasuredPixels, sequentialPeakGb, imageActivationGbPerMegapixel, supportsSequentialOffload, memoryStrategyCapabilities, memoryStrategyStructuralExemptions, calibrations, memoryDeclarationWithhold, memoryStrategyContract, turboFit, measured, control}`; `admissionGeometry`; `depth.admission` | mlx 75 / candle 86 | manifest (partly generated) | **separate measured-evidence store** keyed by `(model, backend, tier)` |
| MLX conversion | `mlx.{quantize, minQualityTier, converter, convertSourceFile, convertBaseRepo, convertBaseSubdir, convertSourceRepo, autoDistillLora, repo, textEncoderRepo, requiresConversion}` | ≤75 | manifest; **also** `NATIVE_CONVERTERS` (`routing/gaps.rs:1335-1373`) | product distribution + engine converter registry |

**Byte share of `builtin.models.jsonc`** (comments stripped, JSON re-serialised) [H]:

| Key | Share of model JSON bytes |
|---|---:|
| `mlx` | 40.4% (`mlx.memoryStrategyContract` alone is 37.5%) |
| `candle` | 26.1% (`candle.memoryStrategyContract` is 20.3%) |
| `downloads` | 13.6% |
| `ui` | 10.6% |
| `limits` | 1.9% |
| `defaults` | 0.7% |

Comments are 593,471 of 2,118,800 file bytes, about 28%. **[L] Estimate:** moving generated memory
contracts and calibration evidence out, and dropping restated engine facts, would cut the
hand-authored manifest to about 30–40% of its current size.

### 1.3 Imported and user models, presets, LoRAs

- **`user.models.jsonc`** is written by the API through `mutate_manifest_entries` and `save_manifest_entries` (`manifest.rs:131-215`) behind a file lock (`:32-88`). It is header-stamped as API-managed (`:90`). It holds model imports, hides and Iris trained checkpoints. `overlay_trained_checkpoint_surface` copies the engine entry's generation surface onto a trained checkpoint on every read (`models.rs:12138-12140`). [H]
- **ComfyUI external base models** come from `apps/rust-api/src/external_base_models.rs` (1,263 lines, header `:1-40`). Rows are scanned and synthesized with `catalogScope:"external"` and `external_base_*` ids (`:43-48`). Assembly uses its own hand tables: `ANCHOR_SUBDIRS` (`:66`), `required_text_encoders(family)` (`:519`), `model_type_for_family` (`:536`) and `family_quant_runnable` (`:508`). External LoRAs are in `external_loras.rs` (645). [H]
- **Imported checkpoints** use core `checkpoint_import.rs` (1,613), `checkpoint_plan_store.rs` (2,694), `checkpoint_inspector.rs` (3,487), `base_weights.rs` (3,389; header-only family/component/quant detection) and `lora_family.rs` (7,960; LoRA family signature tables such as `SIGNATURES` at `:1557`), plus the API's `checkpoint_library.rs` (701), `model_library.rs` (409) and `model_sources.rs` (772).
  - Imported routing goes by provider facts: `imported_provider_routes(backend, family)` (`catalog.rs:1426`) over `config/engine-capabilities/capabilities.{mlx,candle}.json` `imports[]`, with 7 entries in the MLX file.
  - The API overwrites an imported row's provisional `macSupport` from those facts (`catalog.rs:100-107` doc; `models.rs:10451-10530`). [H]
- **Recipe presets.** `RecipePresetManifestEntry` (`contracts.rs:1933+`) is typed. Its schema props (`packages/schemas/recipe-preset.schema.json`) are `id, name, kind, scope, archived, lastUsedAt, order, workflow, modes, model, defaults{count,resolution,aspect,negativePrompt}, prompt{prefix,suffix}, loras[], builtInLoras, appliedDefaults{notes}, ui{description}`.
  - **Presets carry no `advanced` keys.** That matters for §6.
  - API: `apps/rust-api/src/recipe_presets.rs` (1,204). Server folding: `apply_recipe_preset_to_image_payload` (`generation.rs:1962`), `merge_preset_loras_into_payload` (`:2059`), `apply_recipe_preset_to_video_payload` (`:2175`). Web: `presetUtils.js` (848). [H]
- **LoRAs.** `builtin.loras.jsonc` and `lora-manifest.schema.json`. The typed `LoraManifestEntry` (`contracts.rs:1906+`) has untyped `compatibility` and `source`. API: `loras.rs` (4,634), including `validate_job_lora_compatibility_with` (`:1515`) and `validate_lora_specs_for_model` (`:1564`). Job cap: `MAX_JOB_LORAS = 5` (`lora_family.rs:3360`). [H]

### 1.4 Keys duplicated in code tables

These are cases where code tables restate a manifest key: [H unless marked]

| Manifest key | Duplicated in |
|---|---|
| `downloads[0].repo` | `ModelRow.default_repo` (`engines.rs:27`); policed by `tests/test_builtin_manifest_audit.py:505` `test_every_model_table_default_repo_is_installed_by_a_builtin_model` |
| `defaults.steps`, `defaults.guidanceScale` | `ModelRow.default_steps/default_guidance` (`engines.rs:29-31`) |
| `adapter` | `ModelRow.adapter_label` (`engines.rs:33`) |
| `capabilities` + per-backend route | `IMAGE_MODEL_CAPS` / `VIDEO_MODEL_CAPS` (`catalog.rs:837`, `:1189`), `CANDLE_IMAGE_ROUTES` model lists (`candle.rs:127`), `CANDLE_POSE_MODELS` (`candle.rs:2141`), `SDXL_CONTROL_MODELS` (`candle.rs:1482`), `CANDLE_VIDEO_LORA_MODELS` (`candle.rs:35`), `WIRED_{MLX,CANDLE}_POSE_FAMILIES` (`crates/sceneworks-worker/src/image_jobs/base.rs:233`, `:811`; 7 + 7), `STRICT_CONTROL_ENGINES` (`image_jobs/strict_control.rs:54`; 8) |
| `limits.samplers/schedulers/guidanceMethods` | engine descriptors (runtime facts); web label tables `SAMPLER_LABELS/SCHEDULER_LABELS/GUIDANCE_METHOD_LABELS` (`apps/web/src/samplerOptions.js:9-60`); policed by worker test `engines.rs:1245-1262` ("manifest ⊆ engine drift guard") |
| `image.supportsGuidance/NegativePrompt` | descriptor; web seed `apps/web/src/modelCatalog.js` (46 seeded models); policed by `imageAxesParity.test.js` and pytest `test_cfg_free_image_models_declare_both_axes_absent` (`tests/test_builtin_manifest_audit.py:3596`) |
| `ui.controlModes` | web seed + `controlModesParity.test.js` (130 lines) |
| `ui.promptEnhance` | API allowlist `matches!(model, "flux2_dev" \| "flux3_klein" \| "flux3_klein_base")` (`apps/rust-api/src/lib.rs:5336`); worker route checks (`crates/sceneworks-worker/src/image_jobs.rs:386-470`); descriptor `supports_prompt_enhancement` |
| `mlx.converter` | `NATIVE_CONVERTERS` (8) and `CANDLE_NATIVE_CONVERTERS` (2) (`routing/gaps.rs:1335-1375`). Only 4 converter ids appear as `"converter"` values in the manifest; the others appear in `models.rs` and `model_jobs.rs`. [M] |
| `mlx.minQualityTier` | re-surfaced as top-level `minQualityTier` (`models.rs:11023-11036`) |
| per-model memory floors and curves | `apps/web/src/tierSuggestion.js` (861) constants; policed by `memoryFloorCatalogParity.test.js` (1,523 lines) |
| model memory route coverage | worker `RULES` (`memory_route_registry.rs:562`, 85 rows), used by `generate-manifest-memory-declarations.mjs` as an input (`:20-22`) |

---

## 2. Hand tables restating manifest or engine facts, and how drift is policed

### 2.1 Inventory of hand tables

| Table | Location | Rows | Restates | Notes |
|---|---|---:|---|---|
| `IMAGE_MODEL_CAPS` (`ModelCaps{id, mlx_routed, candle_routed, candle_quant, candle_lora, candle_quant_lora}`) | `crates/sceneworks-core/src/jobs_store/routing/catalog.rs:742-835` (struct), `:837-1188` (rows) | 63 | route existence per backend + quant/LoRA (descriptor `supported_quants`, `supports_lora`) | the const-fn `assert!` superset invariant (`:768-775`) is the only structure |
| `VIDEO_MODEL_CAPS` | `catalog.rs:796-835`, `:1189-1340` | 14 | per-backend video routing, i2v-only, VACE | |
| `EXPECTED_*` test mirrors | `catalog.rs:2586-2911`, `:3067-3068` | ~11 lists | the two tables above, restated for tests | a third copy |
| `MLX_ROUTED_FAMILIES` / `CANDLE_ROUTED_FAMILIES` | `catalog.rs:1350`, `:1369` | 3 / 4 | imported families | `#[cfg(test)]`; production uses derived provider facts (`:1354-1356`) |
| `MLX_/CANDLE_/MLX_ONLY_TRAINING_KERNELS` | `catalog.rs:2382`, `:2419`, `:2488` | — | trainer registry | worker `TRAINER_IDS` (`engines.rs:873`, 15) restates them too |
| `VIDEO_UI_MODES` | `catalog.rs:205` | ~10 | UI mode list | web `VIDEO_MODES` (`apps/web/src/modelEligibility.js:14`) is a separate copy |
| `CANDLE_IMAGE_ROUTES` (`{lane, models: Any(&[..])\|Family(fn), shape: fn}`) | `routing/candle.rs:106-329` | 34 | which models get which candle lane | shape predicates are genuine logic; model lists restate descriptors |
| `CANDLE_IMAGE_CHECKS` (refusal table) | `candle.rs:480-520` | ~15 | per-family refusals | |
| `CANDLE_POSE_MODELS` | `candle.rs:2141-2154` | 12 | pose lane per model | worker `WIRED_CANDLE_POSE_FAMILIES` (7) is a parallel table |
| `CANDLE_VIDEO_LORA_MODELS`, `SDXL_CONTROL_MODELS` | `candle.rs:35`, `:1482` | — | | |
| `SENSENOVA_MODEL_IDS`, `QWEN_IMAGE_2_1_MODEL_IDS`, `FLUX3_TIERED_MODEL_IDS` | `routing.rs:22`, `:287`; `image_request.rs:577` | | family membership | |
| `NATIVE_CONVERTERS` / `CANDLE_NATIVE_CONVERTERS` | `routing/gaps.rs:1335-1375` | 8 / 2 | converter registry | |
| routing predicate fns | `candle.rs` (91 fns matching `*candle*`), `mlx.rs` (50 matching `mlx/eligible`) | — | payload-shape × model | 149 distinct snake_case id-like string literals across `candle/mlx/catalog/gaps.rs` [M] |
| `MODEL_TABLE` (`ModelRow{sceneworks_id, engine_id, default_repo, default_steps, default_guidance, adapter_label}`) | `crates/sceneworks-worker/src/engines.rs:22-845` | 61 | id→engine join + manifest defaults | the engine join is also in `runtime/*.json.modelMappings` |
| `VIDEO_ENGINE_IDS`, `TRAINER_IDS` | `engines.rs:846`, `:873` | 9 / 15 | engine registry | |
| `RULES` (memory route registry) | `crates/sceneworks-worker/src/memory_route_registry.rs:562` | 85 | provider × backend × tier × mode × load-profile | dumped into `memoryRouteWitnesses` (783 MLX witnesses) |
| `STRICT_CONTROL_ENGINES`, `WIRED_*_POSE_FAMILIES` | `image_jobs/strict_control.rs:54`; `image_jobs/base.rs:233`, `:811` | 8; 7 + 7 | control wiring | |
| `MacCapabilities` / features | `catalog.rs:441-740` (`mac_capabilities`, `:509`) | ~11 fields + feature map | per-platform feature flags | |
| ComfyUI assembly tables | `apps/rust-api/src/external_base_models.rs:66`, `:508-541` | | family → required TEs, model type | |
| `ADVANCED_KEY_RULES` / `ADVANCED_BUILDERS` | `workflow_share.rs:1411-1833` / `:655-934` | 76 / 17 (+1 deferred, `:935`) | `advanced.*` key universe | the closest thing to an option registry |
| Web `modelCatalog.js` seed | `apps/web/src/modelCatalog.js` (1,233) | 46 | manifest `ui`, `image`, `capabilities` | test-only mirror (`:1-8`), consumed by 11 test files |
| Web vocab tables | `samplerOptions.js:9-60`; `modelEligibility.js:14-34` (`VIDEO_MODES`, `AUDIO_MODES`, `VECTOR_MODES`); `quantTier.js:33-69`; `jobTypes.js:8-47`; `constants.js:94` (`WAN_MOE_PAIRED_LORA_MODEL_IDS`) | | engine vocab, job taxonomy | the engine sampler list includes `er_sde, dpmpp_2m_sde, beta57`, which are absent from `SAMPLER_LABELS`; the UI falls back to the raw key [H] |
| Web model-id literals | `VideoStudio.jsx:140-144` (`ltxVideoModelId="ltx_2_3"`, `ltx25VideoModelId="ltx_2_5"`, `LTX25_TEMPORAL_UPSAMPLE_MAX_ROUNDS=2`); `ImageStudio.jsx:1611` (`"realvisxl_lightning"`, `["lightning","lcm","hyper"]`); `imageJobRequest.js:56` (`flux3_klein*`); `videoStudioValidation.js:15` (`SCAIL2_MODEL_ID`) | | | |

Approximate counts of non-test `const X: &[..]` slice tables (inline `#[cfg(test)]` modules included):
core 109, worker 103, API 18. [M]

### 2.2 How drift is policed today

| Mechanism | What it reconciles | Location | Wired? |
|---|---|---|---|
| Backend capability matrix | evaluates routing predicates for every manifest model × axis. Stores SHA-256 digests of 52 `include_str!`'d sources: API `jobs.rs/generation.rs/lib.rs/dto.rs`, worker dispatch files, and web `imageJobRequest.js`, `imageJobAdvanced.js`, `modelCatalog.js`, `macGating.js`, `previewSupport.js`, `simpleJobs.js`, `VideoStudio.jsx`, … (`routing/matrix.rs:24-135`). A change in any of those requires regenerating `config/backend-capabilities/matrix.json` (221 KB) | `crates/sceneworks-core/src/jobs_store/routing/matrix.rs` (6,790 lines), bin `src/bin/dump-backend-capability-matrix.rs`, `config/backend-capabilities/README.md` | Rust unit test byte-compares the rebuilt matrix (README) [H] |
| Exceptions register | approved MLX/Candle asymmetries with an approver/authority | `config/backend-capabilities/exceptions.json` (14 KB) | part of the matrix test |
| Engine capability facts | stage-1 dumps of linked registries | `config/engine-capabilities/capabilities.{mlx,candle}.json` (478 KB / 494 KB), `runtime/*.json` (102 KB / 89 KB), `audio/` | `scripts/compare-engine-capability-facts.mjs` (76; compares a fresh dump to the checked-in one modulo revision) is in `npm run check` |
| Manifest ⊆ engine sampler/scheduler | `limits.samplers/schedulers` vs the linked registry | `engines.rs:1245-1262` + following tests | `cargo test` on macOS / candle lanes |
| Manifest audit (pytest) | 141 tests: schema validity (`:922` `test_builtin_models_manifest_satisfies_authoring_schema`), MODEL_TABLE repo ↔ manifest (`:505`), pinned revisions, guidance axes, measured-peak tables, license gates, per-model pins | `tests/test_builtin_manifest_audit.py` (5,055) | `check.yml:194` (pytest) |
| Web parity tests | seed ↔ manifest (`imageAxesParity`, `controlModesParity`, `videoGeometryParity`, `jobTypesParity`, `memoryFloorCatalogParity` 1,523, `data/previewSupportCatalog.test.js` 775, `imageStylePreview.parity`) | `apps/web/src/*Parity*.test.js` | `npm --prefix apps/web run test` (`check.yml:55-56`) |
| Advanced-key lints | builders' emitted keys ↔ `ADVANCED_KEY_RULES`; web builder discovery | `crates/sceneworks-core/tests/workflow_share.rs`; doc `docs/workflow-share-envelope.md:653-702` | `cargo test` |
| Source-as-data tests | **240** `include_str!` of `.rs/.js/.jsx/.mjs` source files in Rust tests (e.g. `image_jobs/tests.rs` 90, `routing/matrix.rs` 52, `video_jobs/tests.rs` 44, `memory-adapter/bin/mlx.rs` 12) | | `cargo test` |
| `check-scaffold.mjs` (655) | required paths; manifest roots ↔ schemas (`:318-375`); `ui.promptGuide` on picker entries (`:377`, via `apps/web/src/promptGuideContract.js`); workspace versions | scripts | invoked by `npm run check`? **No.** Not in the `check` inventory. It runs in CI through `desktop-linux-check.yml` references [M] |
| Preview-support generation | engine facts → `builtin.preview-support.jsonc` + `apps/web/src/data/previewSupport.json` | `apps/web/scripts/generate-preview-support.mjs` (167), `previewSupportDerivation.js` (756) | vitest drift guard (preview_support.rs doc) |
| Manifest memory declarations | engine `memoryContracts` / `memoryRouteWitnesses` + `MODEL_TABLE` + `STRICT_CONTROL_ENGINES` + `WIRED_*_POSE_FAMILIES` + `RULES` → manifest regions | `scripts/generate-manifest-memory-declarations.mjs` (166), `scripts/lib/manifest-memory-declarations.mjs` | **explicitly not a gate**, no `--check` (`:27-32`). `report-memory-contract-reconciliation.mjs` always exits 0 |
| Memory matrix | regex-parses **19** Rust files (`scripts/generate-memory-matrix.mjs`; list verified) → `docs/generated/memory-matrix.json` (255,178 lines) + `.md` | `generate-memory-matrix.mjs` (3,412) | `generate-memory-matrix.test.mjs` and `matrix-removed-sources.test.mjs` in `npm run check`; `check:memory-matrix` exists as an npm script |
| Tier integrity | `config/tier-integrity.jsonc` → `docs/generated/tier-integrity.{json,md}` | `scripts/check-tier-integrity.mjs` (1,331) | `npm run rust:check` → `check:rust-derived-docs` → `check:tier-integrity` (`package.json`) |
| Memory anchors | `scripts/extract-memory-anchors.mjs` (2,079) → `config/memory-anchors.json` (571 KB), embedded at runtime (`crates/sceneworks-core/src/memory_anchor.rs:73`) | | `extract-memory-anchors.test.mjs` |
| Closure digests | `scripts/inference-closure-digest.mjs` (848), `anchor-loader-closure.mjs` (1,222), `backfill-closure-digests.mjs` → `config/inference-provider-closures.json` (169 KB; runtime-embedded at `memory_calibration.rs:56`), `config/anchor-loader-closures.json` (3.3 MB; `pub const` in `memory_anchor.rs:3675`, documented as never read at runtime, `:3672-3674`) | | per-lane closure verification step **DISABLED** (`check.yml:367`) |
| Download patterns | `scripts/check-download-patterns.mjs` ↔ `config/download-pattern-evidence.json` (294 KB) | | `check:download-patterns` (offline variant) |

**The `npm run check` inventory** (`package.json` `scripts.check`) runs about 115 `node --test`
files. Roughly 60 of them belong to terminal campaigns (starvector, flux3, h3, moss, yue2, qwen).
`scripts/scaffold-test-selection.mjs:7-12` splits out 4 "historical" suites. [H]

**Gates disabled in `.github/workflows/check.yml`:**
- gen-core skew (`:195`)
- NC weights (`:274`)
- license coverage (`:286`)
- per-lane closure digests (`:367`)

These match the FEATURE_DEVELOPMENT.md gate teardown (§7). [H]

---

## 3. Request contracts and the advanced bag

### 3.1 Contracts by layer

| Layer | Image | Video | Audio | Training |
|---|---|---|---|---|
| Web builders | `buildImageJobRequest` (`apps/web/src/imageJobRequest.js:70-372`) + `buildImageJobAdvanced` (`imageJobAdvanced.js:19-359`); `buildSimpleImageRequest` (`simple/simpleJobs.js:202`); `imageRecipeRequest` replay (`regenerateImage.js:7-60`); `buildEditJobBody/DetailJobBody/UpscaleJobBody` (`imageJobs.js`); Character `buildAdvanced` | `VideoStudio.jsx` `submit` (advanced built inline, `:2040-2080`); `buildSimpleVideoRequest` (`simpleJobs.js:311`); editor `buildBasePayload`, `extendSelectedClip`, `replaceSelectedItem`, `bridgeGap` | `buildSimpleAudioRequest` (`simpleJobs.js:373`); AudioStudio | `trainingConfigSnapshot` (`training/trainingConfig.js`, 1,789) |
| API DTO | `ImageJobRequest` (`apps/rust-api/src/dto.rs:761-897`, camelCase): typed top-level fields + `upscale: ImageUpscaleRequest`, `hires_fix: HiresFixRequest` (`:885-886`), `advanced: JsonObject` (`:888`), loose strength carriers kept as `Option<Value>` so FLUX.3 can refuse them (`:839-852`) | `VideoJobRequest` (`dto.rs:1075-1204`): 28 typed fields + `advanced: JsonObject` | `AudioJobRequest` (`dto.rs:1228+`) | API `training.rs` (3,815) |
| Core parse | `ImageRequest` (`crates/sceneworks-core/src/image_request.rs:140-200`) and `from_payload` (`:239-311`). Infallible clamp-and-default parse that carries `model_manifest_entry` and `advanced` through untyped | `VideoRequest` (`crates/sceneworks-core/src/video_request.rs:496-600`), `from_payload` (`:602`) | worker `audio_jobs.rs` | `TrainingConfig` (`crates/sceneworks-core/src/training.rs:758`) |
| Shared sub-contracts | `HiresFixRequest` (`contracts.rs:832-880`), `ImageUpscaleRequest`, `JobCreateRequest{payload: JsonObject}` (`contracts.rs:742-755`) | — | — | `TECHNIQUE_KEYS` [45] (`training.rs:103`) |

The job envelope is `JobCreateRequest{type, project_id, payload: JsonObject, requested_gpu}`.
Everything persisted per job is that untyped `payload` (`contracts.rs:742-755`). [H]

### 3.2 How a new advanced option flows today

The rule of thumb in the code: a model-specific knob goes under `advanced.*`. A cross-cutting
processing feature such as `upscale` or `hiresFix` becomes a typed top-level DTO field.

The steps for a new `advanced` scalar [H for each touchpoint; the list is the union observed in the two traces below]:

1. **Web state.** Add a `useState` in the studio, seeded from `saved.*`. ImageStudio has 79 `useState(` calls, VideoStudio 61.
2. **Web persistence.** Add the field to the `useStudioSettingsWriter` snapshot (`ImageStudio.jsx:2540`; `hooks/useStudioSettings.js`). ImageStudio persists about 47 fields (`useStudioSettings.js:17-20` comment).
3. **Web replay restore.** Map `rawSettings.*` back into state (VideoStudio `:1371`, `:1501`).
4. **Web control.** Hand-written JSX, gated by `ui.*` flags, model-id literals or family helpers (`isFlux3LayoutModel`, `isMossImage`, …).
5. **Web builder.** Add the key to `buildImageJobAdvanced`, or to the inline video `submit`, with an omit-when-default rule (`imageJobAdvanced.js:155-355`).
6. **Simple studio.** Add it to `simpleJobs.js` if it applies there.
7. **Share registry.** Add a row to `ADVANCED_KEY_RULES` (`workflow_share.rs:1411+`), or the lint fails (`docs/workflow-share-envelope.md:653-702`).
8. **Web share/embed tables.** Label in `workflowShare.js` (e.g. `:397`); lane in `workflowEmbed.js` (e.g. `:213`).
9. **Core validator.** Add a `requested_*` parser and an error fn in `video_request.rs`, `image_sampling_controls.rs` or a family module.
10. **API.** Call the validator from the create handler (`generation.rs`), usually inside a hand-written model gate. Also check the retry/duplicate path (`jobs.rs:924+`).
11. **Routing.** If a backend cannot serve it, add a refusal predicate in `routing/{mlx,candle}.rs`.
12. **Worker.** Parse it again from `request.advanced` in the family arm and map it to the engine request field.
13. **Docs.** Add a row in `docs/workflow-share-envelope.md`.
14. **Matrix.** Regenerate the backend matrix if any digested source changed (§2.2).

### 3.3 Trace A: Hires.fix (typed top-level contract)

| # | Layer | File:line | What it does |
|---|---|---|---|
| 1 | Web state | `apps/web/src/screens/ImageStudio.jsx:720-726` | 5 `useState`s seeded from `saved.hiresFix*` (defaults 0 / 0.7 / 2 / "") |
| 2 | Web mutual exclusion | `ImageStudio.jsx:809-815` | effect clears PiD when upscale or hires is on |
| 3 | Web applicability | `ImageStudio.jsx:1608-1625` | `hiresFixModelCapable = supportsImg2img \|\| family==="sdxl"`; `hiresFixAccelerationBlocked = model==="realvisxl_lightning" \|\| ["lightning","lcm","hyper"].includes(sampler)`; input blocks: img2img ref, poses, control, multi-phase; plus a disabled-reason string |
| 4 | Web UI | `ImageStudio.jsx` ~`:4860-4900` | hand JSX (`<label title=…>` at `:4878`) |
| 5 | Web builder | `apps/web/src/imageJobRequest.js:115-119`, `:270-283` | top-level `hiresFix{enabled,steps,denoisingStrength,upscaleBy,cfgScale?}`; skipped for MOSS edit |
| 6 | Web replay | `apps/web/src/regenerateImage.js:27`, `:56` | deletes `advanced.hiresFix`; reads `raw.hiresFix ?? settings.hiresFix` |
| 7 | Core contract | `crates/sceneworks-core/src/contracts.rs:832-880` | `HiresFixRequest` with serde defaults; `effective_steps` clamps 1..150 |
| 8 | API DTO | `apps/rust-api/src/dto.rs:880-886` | typed field. The comment notes it was previously **dropped by serde**, so the second pass never ran |
| 9 | Core rule set | `image_request.rs:343-430` `hires_fix_request_error` | t2i only; not with upscale; needs `family=="sdxl" \|\| ui.img2img`; no edit, ref, mask, character, poses or controlImage; not PiD; not phases; not lightning/lcm/hyper or `realvisxl_lightning`; target ≤4096 |
| 10 | API create | `apps/rust-api/src/generation.rs:268-283` | calls 9 with the family from the canonical entry |
| 11 | API retry/duplicate | `apps/rust-api/src/jobs.rs:1172-1184` | calls 9 again (shallow `payload_changes` merge) |
| 12 | Routing (candle imported) | `crates/sceneworks-core/src/jobs_store/routing/catalog.rs:2087-2146` | refuses hires + edit, poses or reference on a single-file lane |
| 13 | Routing (MLX) | `routing/mlx.rs:507-513`, `:599-605` | `iris_mlx_eligible` and a sibling refuse hires |
| 14 | Family refusals | `crates/sceneworks-core/src/moss_image.rs:289-294`; `crates/sceneworks-core/src/supra2.rs:257` | MOSS and Supra2 refuse |
| 15 | Worker gate | `crates/sceneworks-worker/src/image_jobs.rs:629`, `:3809-3810` | calls 9 again |
| 16 | Worker lanes | `image_jobs/checkpoint_plan.rs:250-316`, `:482`; `image_jobs/sdxl_control.rs:209`; `image_jobs/krea_imported.rs` (18 refs); `image_jobs/sdxl_imported.rs` (7) | lane-local refusals and handling |
| 17 | Worker execution | `crates/sceneworks-worker/src/image_jobs/base.rs:8390-8420` `HiresFixPlan`/`resolve_hires_fix_plan`; `:5925-5928` writes `hiresFix` into the recipe | |
| 18 | Memory | `crates/sceneworks-worker/src/candle_memory_strategy.rs:350-352`, test `:9305` | two-pass geometry scope |
| 19 | Tests | `image_jobs/tests.rs` (45), `apps/rust-api/src/tests/jobs.rs` (14), `imageJobRequest.test.js` (17), routing tests (14) | |

**Observation.** The applicability rule is written once in core (row 9) and again in the web (row 3),
with the same literals (`realvisxl_lightning`, `lightning|lcm|hyper`, `sdxl`). Six more places refuse
it on specific lanes or families (rows 12–16). None of them declares "Hires.fix requires an img2img
conditioning surface" as data. [H]

### 3.4 Trace B: `advanced.temporalUpsampleRounds` (LTX-2.5)

| # | Layer | File:line | What it does |
|---|---|---|---|
| 1 | Web constants | `apps/web/src/screens/VideoStudio.jsx:141`, `:144` | `ltx25VideoModelId = "ltx_2_5"`, `LTX25_TEMPORAL_UPSAMPLE_MAX_ROUNDS = 2` |
| 2 | Web clamp | `VideoStudio.jsx:207-211` | `clampLtx25TemporalUpsampleRounds` |
| 3 | Web state and persist | `VideoStudio.jsx:334-335`, `:1541`, `:1571` | `useState` seeded from `saved`; included in the settings snapshot |
| 4 | Web replay | `VideoStudio.jsx:1371`, `:1501` | restore from `rawSettings` |
| 5 | Web builder | `VideoStudio.jsx:2061-2069` | emitted only when `model === ltx25VideoModelId`, bundled with 5 sibling LTX-2.5 keys |
| 6 | Web UI | `VideoStudio.jsx:2871-2883` | hand `<select>` with options 0/1/2 written out literally |
| 7 | Web share label | `apps/web/src/workflowShare.js:397-401` | label and detail strings |
| 8 | Web embed lane | `apps/web/src/workflowEmbed.js:213` | `["advanced.temporalUpsampleRounds", L_VIDEO_TIMING]` |
| 9 | Core share rule | `crates/sceneworks-core/src/workflow_share.rs:1711-1716` | `allow_from(…, Scalar, VideoStudioBuilder, reason)` |
| 10 | Core validator | `crates/sceneworks-core/src/video_request.rs:50` (`LTX25_MAX_TEMPORAL_UPSAMPLE_ROUNDS = 2`), `:413-431` (`requested_temporal_upsample_rounds`) | range check, duplicating web constant 1 |
| 11 | API | `apps/rust-api/src/generation.rs:2234-2290` `normalize_ltx25_generation_controls` | hard-coded `KEYS[6]` plus `if model_id != "ltx_2_5"` refusal (`:2266-2272`), `DEAD_GUIDANCE_KEYS[4]` refusal, then calls 10; re-exported via `apps/rust-api/src/lib.rs:71` |
| 12 | Worker parse | `crates/sceneworks-worker/src/video_jobs/ltx.rs:139`, `:159`, `:188-189`, `:253-254`, `:1998` | calls 10 again and maps to the engine `Option<u32>` |
| 13 | Worker wiring | `video_jobs/wan.rs:1379`, `:1523`, `:1715-1716`, `:2225`; `video_jobs/candle.rs:1453-1454` | passed through shared video input structs (set `None` for Wan); added to the memory overlay label |
| 14 | Docs | `docs/workflow-share-envelope.md:437` | |

**Observation.** One integer knob with a range of 0..2 lives in **9 source files across 4 languages
or layers**. Its applicability ("LTX-2.5 only") is expressed three times: the web id literal, the API
id literal, and the worker family arm. Its range is defined twice (web and core). Nothing gives the UI
the label, option set or range as data. [H]

### 3.5 The advanced-key universe

- `ADVANCED_KEY_RULES` classifies **76** keys (`workflow_share.rs:1411-1833`). Parsed list: `flux3_layout, flux3_reference_ids, mossImageEdit, …, sampler, scheduler, schedulerShift, cfgInterval, captionOverflow, steps, guidanceScale, guidanceMethod, enhancePrompt, transparentBackground, usePid, decoder, pidTarget, ipAdapterScale, controlnetConditioningScale, trueCfgScale, strength, textStyleGain, viewAngle, poses, faceRestore, controlMode, controlScale, styleId, stylePrompt, cnScale, angleSet, systemMessage, imageGuidanceScale, keypointCollectionId, phases, flashAttn, mlxQuantize, mlxQuantizeExplicit, convRot, quantTier, controlImage, controlWeights, recipePresetId, presetMissingLoras, motion, ltxPipeline, distilledVariant, transformerVariant, vaeDecoder, autoDuration, autoDurationMin/MaxSeconds, temporalUpsampleRounds, textEncoderModel, lightning, videoCfgGuidanceScale, videoStgGuidanceScale, videoRescaleScale, videoConditioningStrength, bridgeRightVideoConditioningStrength, veda, timelineAction, durationHint, precision, quantization, selectedPersonTrack, replacementModeLabel, timelineContext`. [H]
- Training uses a separate universe of 45 `TECHNIQUE_KEYS` (`training.rs:103`) under `advanced`. [H]
- Rust reads, by `advanced.get("…")` literal grep, outside tests: worker 61 sites / 20 keys, core 82 / 28, API 15 / 13. The heaviest keys are `mlxQuantize` (25) and `poses` (18). This is a lower bound, since keys are also read through constants. [M]
- Some keys are hardware or budget keys rather than intent: `mlxQuantize`, `mlxQuantizeExplicit`, `convRot`, `quantTier`, `flashAttn`. The share registry already separates them with `deny`. That split, "what to make" vs "what this machine can afford", is the right axis for a typed request (§Design). [H]
- MCP builds its own `advanced` from a fixed subset (`crates/sceneworks-mcp/src/server.rs:1461-1479`: `mossImageAdvanced, mossImage, mossImageEdit, steps, guidanceScale, sampler, schedulerShift, cfgInterval, captionOverflow`). It serves one hand-written JSON Schema, `moss_image_advanced::schema()` (`crates/sceneworks-core/src/moss_image_advanced.rs:117-122`). That is precedent for schema-described options. [H]

---

## 4. Validation inventory by layer

### 4.1 Web

| Module | Lines | Rules |
|---|---:|---|
| `apps/web/src/imageStudioValidation.js` | 250 | `imageGenerateValidation` (`:63-176`) mostly **aggregates precomputed booleans and strings**: project, caption or prompt, composed-prompt budget, character, edit source, edit LoRA, preset/LoRA issues, multi-phase issues, `dimensionError`, `referenceOverflowError`, decoder capability pending, restore plan. Also `batchResolutionMessage` (`:177`) and `imageBatchValidation` (`:189`) |
| `apps/web/src/videoStudioValidation.js` | 142 | `videoGenerateValidation` (`:38-142`): prompt, budget, inputs, mode support, LTX IC-LoRA, replace readiness, SCAIL-2 >6 refs (**literal 6**; core has `MAX_SCAIL2_REFERENCE_CHARACTERS = 6` at `video_request.rs:43`), reference limits, audio-only reference set, Veda, execution tier |
| `apps/web/src/generationValidation.js` | 51 | preset dialog, preset LoRA issues |
| `apps/web/src/workflowReplayValidation.js` | 298 | replay compatibility |
| `apps/web/src/mossImageAdvancedValidation.js` | 35 | MOSS prompt rules |
| `apps/web/src/styleComposer.js:148` `promptBudget` | — | composed-prompt cap. Server cap `MAX_PROMPT_CHARS` comes from `sceneworks_core` (`apps/rust-api/src/lib.rs:460`) |
| `apps/web/src/samplerOptions.js:185-192` | — | `BASE_MAX_STEPS = 80` (core `BASE_MAX_IMAGE_STEPS = 80`, `image_request.rs:71`); `limits.hardMaxSteps` override |
| `apps/web/src/imageReferenceLimits.js`, `imageDimensions.js`, `pidEligibility.js`, `imageMultiPhase.js`, `quantTier.js`, `tierSuggestion.js` | — | computed inputs to the validators |
| In-component rules | — | `ImageStudio.jsx:1608-1625` (Hires.fix), `:809-815` (exclusions); `VideoStudio.jsx:207` (clamps) |

The web pattern is "the component computes, the validator lists". The real rules are spread across
the 5,061-line studio and its helper modules. [H]

### 4.2 API (`apps/rust-api`)

| Function | File:line | Notes |
|---|---|---|
| `validate_image_job` | `lib.rs:5391-5551` | project, MOSS carriers, prompt ≤`MAX_PROMPT_CHARS`, prompts batch, LoRA count ≤5, mode allowlist, `count` 1..8 (**literal**; core `MAX_COUNT=8`), dimension floor/ceiling, upscale engine and factor ∈{2,4}, Iris rules |
| `validate_image_pose_count` | `lib.rs:5552` | `MAX_JOB_POSES` (core) |
| `validate_prompt_extras` | `lib.rs:5142-5158` | negative prompt length, `MAX_ADVANCED_JSON_BYTES` |
| `validate_prompt_enhancement_fields/route/payload` | `lib.rs:5166-5374` | temperature 0..2 (`:5160`), max tokens, model allowlist (`:5336`), strict-control exclusion |
| `validate_video_job` | `lib.rs:5618` | |
| `validate_audio_job`, `validate_audio_song_fields`, `validate_audio_job_for_model`, `validate_audio_edit_fields` | `lib.rs:5853`, `:6010`, `:6133`, `:6261` | |
| `validate_video_reference_asset_ids_payload` | `lib.rs:6381` | |
| `validate_dimension(_in)` | `lib.rs:6491-6510` | |
| `create_image_job` pipeline | `generation.rs:121-550` | `validate_image_job`, MOSS, preset, style, prompt enhancement, `canonicalize_image_model_payload`, decoder (`validate_selected_decoder_for_manifest`, `:54`), FLUX.3 refs, default geometry/count, `hires_fix_request_error`, `image_sampling_knob_error` (`video_request.rs:1330`), `image_sampling_control_error` (`image_sampling_controls.rs:211`), `prompt_batch_error`, `image_reference_limit_error`, `image_dimension_error`, `admission_geometry::refuse_over_envelope`, `ordered_image_reference_error`, `validate_ordered_image_references`, LoRA compatibility, control overlay, `validate_imported_submission` (`:1739`) |
| `normalize_ltx25_generation_controls` | `generation.rs:2234` | per-model key gating (§3.4) |
| `create_video_job` | `generation.rs:2473` | |
| Vector / VQA / interleave | `generation.rs:551-766`, `:1840`, `:1884` | |
| Raw `/jobs` | `jobs.rs:77-150` | refuses typed generation types (`:91-99`), strips carriers, `validate_raw_job_payload` (`:1679`) |
| Retry / duplicate | `jobs.rs:427`, `:703` → `validate_and_canonicalize_merged_generation_payload` (`:924-1265`) | §4.4 |

### 4.3 Core and worker

- **Core** holds the reusable rules:
  - `image_request.rs`: `hires_fix_request_error`, `ordered_image_reference_error`, FLUX.3 tiers `:571-790`, `default_count/resolution`.
  - `video_request.rs`: `duration_limit_error` `:929`, `hard_min/max_steps` `:985/1000`, `image_max_reference_assets` `:1024`, `image_reference_limit_error` `:1044`, `image_dimension_envelope/error` `:1086/1120`, `allowed_steps`/`steps_limit_error` `:1210/1276`, `image_sampling_knob_error` `:1330`, `fps_limit_error` `:1597`, `reference_caps`/`reference_limit_error` `:1663/1751`, `video_admission*` `:2093-2534`.
  - `image_sampling_controls.rs` (scheduler shift, CFG interval, caption overflow, prompt batch).
  - `admission_geometry.rs`, `moss_image.rs`, `supra2.rs`, `h3_veda.rs`, `training.rs` (`technique_value`, `:78-80`).
  - **Placement smell:** image validators live in `video_request.rs`. [H]
- **Worker** re-applies the same rules before load:
  - `image_jobs.rs:600-680` calls MOSS `native_request`, `supra2::validate_request`, `validate_flux3_layout_request` (`:561`), `validate_hires_fix_request` (`:3809`), `validate_prompt_enhancement_request` (`:471`; its own `PROMPT_ENHANCE_MAX_TEMPERATURE = 2.0` at `:260`), `flux3_precision::validate_image_quant_tier_request`, and `validate_selected_decoder_request` (`image_jobs/base.rs:7320`).
  - Video: `video_jobs/mod.rs:1293` `validate_video_lora_compatibility`; per-family parsers re-run the core `requested_*` fns.
  - Memory and fit admission: `vram_gate.rs` (8,217), `mlx_fit_gate.rs` (23,311), `candle_memory_strategy.rs` (10,132), `fit_gate.rs` (`SVD_32GB_VALIDATED_MAX_STEPS = 25`, `:235`). [H for file and line; M for completeness]

### 4.4 Duplicates and gaps

| Rule | Web | API | Core | Worker | Routing |
|---|---|---|---|---|---|
| Hires.fix applicability | `ImageStudio.jsx:1608-1625` | via core (`generation.rs:283`, `jobs.rs:1184`) | `image_request.rs:343` | via core (`image_jobs.rs:3809`) + lane refusals | `catalog.rs:2087`, `mlx.rs:507/599` |
| Prompt enhancement | `ui.promptEnhance` gate (ImageStudio) | `lib.rs:5166-5374` (own constants, model allowlist) | — | `image_jobs.rs:259-560` (own constants); `video_jobs/ltx.rs:1922` | — |
| Max steps | `samplerOptions.js:185` | via core | `image_request.rs:71`, `video_request.rs:1000` | `image_jobs/base.rs:5414` clamp | — |
| Count 1..8 | studio | `lib.rs` literal `1..=8` | `MAX_COUNT=8` (`image_request.rs:66`) | clamp in `from_payload` | — |
| Upscale factor {2,4} | studio | `lib.rs` (`upscale.factor must be 2 or 4`) | — | — | — |
| SCAIL-2 ≤6 refs | `videoStudioValidation.js` literal | via core | `video_request.rs:43` | | |
| Temporal upsample 0..2 | `VideoStudio.jsx:144` | via core + id gate | `video_request.rs:50` | via core | |
| Reference caps | `imageReferenceLimits.js` | via core | `video_request.rs:1024-1070`, `:1663-1760` | | |
| Prompt length | `styleComposer.js:148` | `MAX_PROMPT_CHARS` (~15 sites in `lib.rs`/`generation.rs`) | `sceneworks_core::MAX_PROMPT_CHARS` | | |

**Retry and duplicate path gap [M].** `validate_and_canonicalize_merged_generation_payload`
(`jobs.rs:924-1265`) re-runs these:
- raw-payload checks, video reference ids, `VideoRequest::from_payload`, platform availability;
- pose count, control overlay, FLUX.3 refs, prompt enhancement, decoder, Hires.fix, LoRAs, imported submission and runtime model sources.

A grep of that range for `sampling|dimension|steps|reference_limit|prompt_batch|envelope` finds
**none** of these core checks, all of which the create path calls (`generation.rs`):
`image_sampling_knob_error`, `image_sampling_control_error`, `image_dimension_error`,
`image_reference_limit_error`, `prompt_batch_error`, `refuse_over_envelope`. The worker re-checks
some of them, so the effect may be "refused later" rather than "admitted". This is a structural
consequence of hand-composing validators per path. I did not run it, so it is not verified.

---

## 5. Web: studios, gating and control visibility

### 5.1 Sizes (non-test LOC)

| File | Lines |
|---|---:|
| `apps/web/src/screens/ImageStudio.jsx` | 5,061 (75 imports, 79 `useState`) |
| `apps/web/src/screens/ImageEditor.jsx` | 4,542 |
| `apps/web/src/App.jsx` | 4,326 |
| `apps/web/src/screens/VideoStudio.jsx` | 3,367 (61 `useState`) |
| `apps/web/src/screens/TrainingStudio.jsx` | 2,619 |
| `apps/web/src/screens/ModelManagerScreen.jsx` | 2,566 |
| `apps/web/src/screens/AudioStudio.jsx` | 1,751 |
| `apps/web/src/components/generationStudio.jsx` | 1,252 |
| `apps/web/src/simple/SimpleImageStudio.jsx` | 980; `simple/` total 11,643 incl. tests |
| `apps/web/src/styles.css` | 20,109 |

Web total: 253,739 lines, of which 118,585 are in `*.test.*` files and 135,154 are not. [H]

### 5.2 Gating modules

| Module | Lines | Reads |
|---|---:|---|
| `macGating.js` | 161 | `/capabilities/mac` → `macGatingActive`; `model.macSupport.{supported, reason, features.{pose,reference,edit,lycoris,videoModes}}`; `macFeatureBlock(caps,key)`, `macUpscaleEngineBlocked`, `macTrainingKernelBlocked` |
| `candleGating.js` | 89 | `candleGatingActive`; `model.candleSupport.{supported, features.videoModes}`; `training.candleSupportedKernels` |
| `modelEligibility.js` | 396 | `imageModelServesMode` (`:75-96`) accepts both `edit_image` and `image_edit` tokens; `videoModelServesMode` (`:130-136`); `supportedControlModes` (`:326`) from `ui.controlModes`; vocab lists `VIDEO_MODES/AUDIO_MODES/VECTOR_MODES` (`:14-34`) |
| `samplerOptions.js` | 237 | `effectiveLimits(model, backend)` (`:90`) merges `limits` with `mlx.limits`/`candle.limits`; sampler/scheduler/guidance menus, defaults, step menus, hints (`patchGridHint`, `promptTokenLimitHint` from `image.*`) |
| `tierSuggestion.js` | 861 | memory floors, headroom constants (`TRANSIENT_HEADROOM_BYTES` 14 GiB `:54`, `CANDLE_HEADROOM_GB` 2.0 `:480`, `MEMORY_HEADROOM_FRACTION` 0.9 `:68`), evidence vs. declared peaks |
| `quantTier.js` | 479 | tier vocab, `tierPickerOptions`, `shouldShowTierPicker`, `defaultTierSelection`, floor |
| `previewSupport.js` | 190 | `preview.byBackend` |
| `modelCapabilities.js` | — | `CAPABILITY_LABELS`, retired caps |

**Sources of "is this control or model available?"** observed in the studios [H]:

1. `capabilities[]`
2. `ui.*` flags. ImageStudio reads 27 distinct `ui.` keys; VideoStudio 4.
3. `image.*`, `video.*` and `audio.*` axes
4. `limits.*` / `defaults.*` and per-backend `mlx.limits` / `candle.limits`
5. `macSupport` / `candleSupport` plus the `/capabilities/mac` `features` map
6. `preview.byBackend`
7. `variants` / `hasVariantMatrix` / `minQualityTier`
8. decoder options (`decoder_support`)
9. model-id literals and family helpers (`isFlux3LayoutModel` ×8 and `isMossImage` ×4 in ImageStudio; `ltx_2_5`/`ltx_2_3` constants in VideoStudio)
10. worker advertisement (`workerAdvertises`, `simpleJobs.js:33`)

**Family-specific web modules** total about 9.7k lines. The largest:
- `ideogramCaption.js` (774), `imageMultiPhase.js` (281, Krea), `qwenAlpha.js` (169), `irisRestore.js` (145), `minimaxH3Turbo.js` (110)
- `mossImage*.js` (around 170), `Flux3LayoutEditor.jsx` (133), `MultiPhaseEditor.jsx` (233), `QwenRewritePromptControl.jsx` (227)
- the YuE2 lab: `yue2Lab.js` 1,123, `audioYue2*.jsx` about 3.4k. [H]

### 5.3 How controls are shown or hidden today

Controls are hand-written JSX blocks wrapped in `cond ? <label>…</label> : null`. Examples:
- `ImageStudio.jsx:4606-4660`: guidance method select from `guidanceMethodOptions`; flash-attention toggle `!moss`; `promptEnhance ? …`; `precisionToggle && !showTierPicker ? …`
- `VideoStudio.jsx:2871-2883`: hard-coded `<option>` list

Each control owns its label, tooltip copy, default, clamp, persistence key and emit rule. The only
data-driven pieces are menu contents: samplers and schedulers from `limits`, tiers from `variants`,
text encoders, decoders and control modes. [H]

### 5.4 What a capability-driven, schema-generated advanced panel needs

Derived from the gaps above. This is a design inference [L], grounded in the observed touchpoints.

1. **An option descriptor**, served per model and per backend: `key`, `type`
   (bool/int/float/enum/list/object), `default`, `range` or `enum`, `step`, `unit`, `group`
   (sampling/quality/conditioning/post/engine), `label`, `help`, `emitPolicy` (omit-when-default,
   always), `persist` (studio-sticky or not), `share` (allow/deny, which `ADVANCED_KEY_RULES` already
   has), `replay` (prefill policy, which `workflowShare.js` labels already partly have), and
   `availability{available, reason}` computed server-side for this model/backend/mode.
2. **Dependency and exclusion expressions** that are data, not JSX: "requires mode ∈ {…}", "conflicts
   with option X", "hidden unless X". Hires.fix today needs `img2img ∨ sdxl`, ¬upscale, ¬PiD, ¬phases,
   ¬accelerated sampler and ¬inputs. A small predicate AST, evaluated by core in Rust and by the web
   in JS, or evaluated only server-side through a `/models/:id/options?mode=…` endpoint.
3. **A generic state container** keyed by option id. This replaces the 79 + 61 `useState`s for
   advanced knobs and the 47-field snapshot. Persistence and replay become `Record<optionId, value>`.
4. **Custom-renderer escape hatches** for structured options (FLUX.3 layout boxes, MOSS tensors,
   multi-phase lists, poses, YuE2 lab). `AdvancedShape` already names these shapes
   (`workflow_share.rs` `AdvancedShape::{Flux3Layout, MossImageOptions, PromptBatch, …}`).
5. **A server-computed effective menu** for enum options (samplers, schedulers, guidance, decoders,
   tiers) from engine descriptors, so the label tables are the only web-side vocabulary.
6. **Removal of the pre-catalog seed** (`modelCatalog.js`) and its parity tests, once the catalog is
   the only source. The seed exists only for cold start (`modelCatalog.js:1-8`).

---

## 6. Persistence and replay: constraints on request shapes

| Store | Shape | Version | Constraint |
|---|---|---|---|
| Job rows (SQLite `jobs_store`) | `JobCreateRequest.payload: JsonObject` verbatim, including the server-injected `modelManifestEntry` carrier | **none**: no payload or request version (repo-wide grep for `payloadVersion`, `requestVersion` and `recipeVersion` finds only `STRUCTURED_PROMPT_RECIPE_VERSION = 1`, `ideogramCaption.js:750`) [H] | Retry and duplicate (`jobs.rs:427`, `:703`) shallow-merge `payload_changes` into the persisted payload and re-validate (`jobs.rs:924+`). Old payload shapes must stay parseable as long as job rows exist |
| Asset recipe sidecars | `Recipe{mode, model, adapter, prompt, negative_prompt, seed, loras, style_preset, normalized_settings: JsonObject, raw_adapter_settings: JsonObject, imported_model_ref, …}` (`contracts.rs:1568-1600`); written by the worker (e.g. `image_jobs/base.rs:5925`) | project DB `PROJECT_SCHEMA_VERSION = 8` (`project_store.rs:4903`) covers tables, not recipe content | `imageRecipeRequest` (`regenerateImage.js:7-60`) rebuilds a request from `rawAdapterSettings` and strips 12 server keys by name. **Every historical `advanced` key must stay accepted or be translated.** |
| Workflow share envelope (PNG/MP4 embedded) | `WorkflowShare` (`workflow_share.rs`); `ADVANCED_KEY_RULES` filters `advanced` | `WORKFLOW_SHARE_SCHEMA_VERSION = 1` (`:121`); a newer file is rejected (`:3515-3518`) | Shared images in the wild carry v1 keys. Renaming a key needs an alias table |
| Studio settings | per-studio, per-workspace snapshot in localStorage plus a durable server copy (128 KiB per workspace entry; `hooks/useStudioSettings.js:14-45`); about 47 ImageStudio fields | none (field-wise `saved.x ?? default`) | Field names are state names, not request keys. Two vocabularies for the same knob (`hiresFixEnabled` vs `hiresFix.enabled`) |
| Recipe presets | typed `RecipePresetManifestEntry`; `defaults{count,resolution,aspect,negativePrompt}`, prompt prefix/suffix, LoRAs | `schemaVersion: 1` | No `advanced` content, so a request-shape change does not break presets |
| Training plans | `TRAINING_PLAN_VERSION = 1`, `TRAINING_CONTRACT_SCHEMA_VERSION = 1` (`training.rs:36-40`) | versioned | The only versioned request-like contract |
| Film plans / reference packs | `PLAN_SCHEMA_VERSION`, `REFERENCE_PACK_SCHEMA_VERSION` (`project_store.rs:39-40`) | versioned | out of scope here |

**Implication.** Any typed request redesign must ship one of two things. Either a request-shape
version with an upgrader that maps legacy `advanced` keys and top-level fields into the new option
map (covering job rows, recipe sidecars and v1 share envelopes), or a permanent alias layer.
`ADVANCED_KEY_RULES` is the natural home for aliases, and the 12-key strip list in
`regenerateImage.js:21-35` shows the server-key versus authored-key split already exists ad hoc. [H/L]

---

## 7. Tooling and CI

### 7.1 Gate-teardown state (binding)

`FEATURE_DEVELOPMENT.md:135-165`:
- Measurement currency is advisory.
- Pin-keyed gates were dismantled (sc-19758 `68670a3ee`: four `check.yml` steps `if: false`; sc-19751: license coverage exits 0; `e14171984`: a pin bump no longer invalidates capability dumps).
- Disabled or unwired scripts are the designed state.
- Campaigns (capability dumps, memory-matrix regeneration, calibration, VRAM/canary) run once at epic end.
- Tests over measured corpora assert shape, not populations.

`CLAUDE.md` repeats that there is no merge queue and that `strict:false` applies. [H]

**Consequence for this redesign.** The registry design must not add new measurement gates. Deriving
facts at build time from checked-in dumps, which is the existing preview-support and decoder pattern,
is compatible with that policy. Requiring a fresh dump per PR is not. [H/L]

### 7.2 The pin bump (five sites)

`scripts/bump-inference.mjs` (1,856) rewrites the following, then regenerates `Cargo.lock` and runs
`check-gen-core-skew.sh` [H; `bump-inference.mjs:1-120`, `memory/inference_pin_has_five_sites_not_three.md`]:

1. root `Cargo.toml` `[patch]` (candle-core / candle-kernels / candle-metal-kernels)
2. `crates/sceneworks-worker/Cargo.toml`
3. `crates/sceneworks-memory-adapter/Cargo.toml` (manifests are discovered, `:84-95`)
4. `INFERENCE_RUNTIME_REVISION` in `crates/sceneworks-worker/src/catalog_semantic_jobs.rs`
5. `INFERENCE_PIN` in `crates/sceneworks-memory-adapter/src/lib.rs`

It also touches `config/inference-provider-closures.json` and `config/anchor-currency-attestations.json`
(`:113-116`). `--self-test` is deliberately not in `npm run check` (`:22-28`). Facts files carry
`generatedFrom.inferenceRevision`. For example, MLX facts are at `0b80ec51…` and Candle preview
facts at `354b41cc…` (`builtin.preview-support.jsonc:8-14`), so the two backends' dumps are at
**different** revisions. That is allowed under the teardown. [H]

### 7.3 Generated artifacts

| Artifact | Size | Producer | Runtime consumer? |
|---|---:|---|---|
| `docs/generated/memory-matrix.json` | 255,178 lines | `scripts/generate-memory-matrix.mjs` (regex over 19 Rust files) | no (docs, tests) |
| `docs/generated/memory-calibration-evidence.json` | 28,776 lines | calibration harness | **yes**: `memory_calibration.rs:87`, `memory_anchor.rs:1488` |
| `docs/generated/{krea-candle-five-rung, qwen-candle-five-rung, ltx-mlx-*, video-memory-curves}.json` | ~13k lines | campaigns | **yes**: `memory_anchor.rs:1470-1494`, `video_memory_curves.rs:25-31` |
| `config/backend-capabilities/matrix.json` | 221 KB | `dump-backend-capability-matrix` | yes, test byte-compare (`routing/matrix.rs`) |
| `config/engine-capabilities/*.json` | ~1.17 MB total | `cargo run -p sceneworks-worker --bin dump-engine-capabilities` | **yes**: `catalog.rs:1378-1381`; preview, decoder |
| `config/manifests/builtin.preview-support.jsonc` | 327 lines | `generate-preview-support.mjs` | yes |
| `config/memory-anchors.json` | 571 KB | `extract-memory-anchors.mjs` | yes (`memory_anchor.rs:73`) |
| `config/memory-calibration-plan.json` | 181 KB | hand + harness | worker (`inference_runtime.rs:588`, `memory_route_registry.rs:3762`) |
| `config/anchor-loader-closures.json` | 3.3 MB | `anchor-loader-closure.mjs` | **no**; documented test-only (`memory_anchor.rs:3672-3676`) |
| `config/inference-provider-closures.json` | 169 KB | `inference-closure-digest.mjs` | yes (`memory_calibration.rs:56`) |
| `config/tier-integrity.jsonc` → `docs/generated/tier-integrity.*` | 256 KB | `check-tier-integrity.mjs` | no |
| manifest `memoryStrategyContract` regions | about 58% of manifest bytes | `generate-manifest-memory-declarations.mjs` | yes (inside the manifest) |
| `config/rung4-*`, `download-pattern-evidence.json`, `sdxl-cuda-memory-endpoint-evidence/` | ~0.8 MB | various | partially (`sdxl-cuda-memory-endpoints.json` embedded) |

### 7.4 What could be deleted if facts were derived (**[L] estimates**)

| Tool or code | Lines | Why it exists | Fate under a derived registry |
|---|---:|---|---|
| `routing/matrix.rs` source-digest machinery + `config/backend-capabilities/matrix.json` | 6,790 + 9,674 (json lines) | proves predicate answers and catches drift across 52 digested files | shrink to a generated per-model/axis report from the registry (maybe 1–2k). Source digests go entirely |
| `EXPECTED_*` mirrors in `catalog.rs` tests | ~400 | third copy of the route tables | delete |
| `IMAGE_MODEL_CAPS`, `VIDEO_MODEL_CAPS`, model lists in `CANDLE_IMAGE_ROUTES`, `CANDLE_POSE_MODELS`, `NATIVE_CONVERTERS`, `MODEL_TABLE` defaults columns, `VIDEO_ENGINE_IDS`, `TRAINER_IDS`, `WIRED_*_POSE_FAMILIES` | ~1.5–2k rows + comments | hand routing facts | generated registry |
| `apps/web/src/modelCatalog.js` + parity tests (`imageAxesParity`, `controlModesParity`, `videoGeometryParity`, much of `memoryFloorCatalogParity`, `previewSupportCatalog.test`) | 1,233 + ~2.7k | cold-start seed mirror | delete; render a skeleton until the catalog arrives |
| `generate-manifest-memory-declarations.mjs` + lib + `report-memory-contract-reconciliation.mjs` | ~1–2k | engine → manifest projection | becomes the registry generator's memory sidecar; no in-place manifest rewrite |
| `generate-memory-matrix.mjs` (regex over Rust) + tests + `docs/generated/memory-matrix.json` | 3,412 + ~1.4k tests + 255k json | memory coverage census | regenerate from the typed registry and `RULES` dump. **Never regex Rust** |
| Python manifest audit tests that restate tables (`test_every_model_table_default_repo…`, guidance-axis tests, sampler tables) | subset of 5,055 | | move to schema + registry invariants |
| Source-as-data `include_str!` tests (240) | — | cross-file drift detection | most become registry assertions |
| Web vocab tables (`SAMPLER_LABELS` etc.) | small | | keep labels, derive membership |

`scripts/` is 193,392 lines over 399 files. Category greps (they overlap) give: memory, calibration,
anchor and closure about 68k; terminal campaigns (starvector 13.8k, flux3 11.6k, h3/minimax 17.4k,
moss 12.4k, yue2 5.5k, qwen and epic harnesses 17.8k) about 78k; CI and relevance checks about 26k.
The campaign harnesses are not registry duplicates, and slimming them is a separate decision. [M]

---

## 8. LOC census (SceneWorks, `git ls-files | xargs wc -l`)

| Area | Files | Lines | Of which tests* | Notes |
|---|---:|---:|---:|---|
| `apps/web/src` | 716 | 253,739 | 118,585 (`*.test.*`) | `styles.css` 20,109 |
| `apps/rust-api` | 114 | 178,233 | ~90,177 (`src/tests/` + `*tests*`) | `models.rs` 18,793; `lib.rs` 6,603; `generation.rs` 3,493; `jobs.rs` 3,183; `dto.rs` 1,644 |
| `crates/sceneworks-core` | 169 | 262,370 | ≥63,350 (test dirs/files; inline `mod tests` not separated) | routing 23,235 (incl. `matrix.rs` 6,790); `project_store.rs` 16,589; `memory_anchor.rs` 10,096 |
| `crates/sceneworks-worker` | 326 | 437,542 | ≥93,736 | `image_jobs/` 99,410 (`base.rs` 20,329, `tests.rs` 30,261); `video_jobs/` 46,022; `mlx_fit_gate.rs` 23,311; `memory_route_registry.rs` 10,513; `candle_memory_strategy.rs` 10,132; `vram_gate.rs` 8,217 |
| `crates/sceneworks-memory-adapter` | 20 | 71,824 | — | `bin/mlx.rs` 34,023; `bin/candle.rs` 13,140; `lib.rs` 3,930 |
| `crates/sceneworks-mcp` | 11 | 7,449 | — | |
| `crates/sceneworks-image-quality` | 5 | 13,775 | — | |
| `apps/desktop` | 217 | 34,304 | — | only 12,492 lines are `.rs` (113 `.txt` files) |
| `apps/rust-worker` | 2 | 24 | — | thin bin |
| `scripts/` | 399 | 193,392 | ~115 `node --test` files in `npm run check` | §7.4 |
| `config/` | 111 | 224,402 | — | manifests 33,701; engine-capabilities 45,013; anchor-loader-closures 53,716; film-harness 23,840; backend-capabilities 9,674; memory-anchors 14,423 |
| `docs/generated/` | 11 | 298,182 | — | memory-matrix.json 255,178 |
| `docs/` (other) | — | ~34k `.md` | — | plus large raw `.rgb` evidence under `docs/calibration/` (line counts meaningless) |
| `packages/schemas` | 14 | 5,182 | — | model schema 2,428 |
| `tests/` (pytest) | 57 | 30,458 | — | manifest audit 5,055; contract snapshots 1,848 + fixtures |
| `.github/` | 43 | 11,789 | — | 32 workflows, `check.yml` 512 |

\*Rust test splits are lower bounds: inline `#[cfg(test)] mod tests` inside source files are not separated.

**Slimming targets** (**[L]**: these depend on other researchers' findings on inference and the worker):

| Area | Target |
|---|---|
| Routing (`jobs_store/routing/*`, 23k) | −40–60%. Tables and probes become generated; payload-shape predicates stay but key on declared conditioning |
| `apps/rust-api/src/models.rs` (18.8k) | −30–40%. One typed catalog projection replaces ad-hoc `apply_*` field synthesis |
| Web studios (`ImageStudio` + `VideoStudio` + Simple, ~9.4k) | −35–50% for advanced/gating code with a generated panel |
| `apps/web/src` gating/vocab helpers (~2.5k) | −50% |
| Hand-authored manifest | −60–70% once generated memory contracts move out and restated engine facts are dropped |
| Drift tooling (`matrix.rs`, parity tests, `EXPECTED_*`, memory-matrix regex generator, manifest projection script) | about 10–15k lines of code removable, plus about 255k lines of generated JSON |

---

## Design implications

### D1. One registry, three inputs, one typed output

```
inference registry ──(stage-1 dump, checked in; exists today)──► config/engine-capabilities/{runtime,imports,audio}
product manifest (slim, hand-authored: identity, downloads, licensing, copy, product defaults & menus)
measured evidence store (calibrations, floors, curves, memory contracts — generated, keyed (model,backend,tier))
                         └──────────── build.rs / generator ────────────┘
                                         ▼
                 crates/sceneworks-core::registry::ModelSpec (typed, serde)
        ├─ routes:  per backend → {engine_id, modes, conditioning, quants, lora/lokr, preview, decoders}
        ├─ options: resolved OptionSpec set (D2) with availability per backend/mode
        ├─ limits:  merged envelope (engine hard limits ∩ product menus)
        └─ memory:  ladder facts (consumed by admission; not the UI)
```

- **Typed, not `Value`.** Replace `ModelManifestEntry`'s `JsonObject` fields
  (`contracts.rs:1838-1866`) with real structs. Make the capability vocabulary match the data. Today's
  `Unknown(String)` fallback hides `edit_image` vs `image_edit`. Keep `#[serde(flatten)] extra` only for
  forward compatibility.
- **One merge rule** for builtin plus user overrides, replacing `manifest.rs:324` (shallow) and
  `models.rs:12402` (six nested keys). The catalog and the worker carrier must be the same projection.
  Today the API injects `modelManifestEntry` into every job. With a typed spec, the job carries
  `{model_id, spec_revision}` plus only user-override deltas. The worker resolves the same registry it
  links. Note that the API embeds the worker crate (79 non-test refs), and server topology runs an API
  without engines (`preview_support.rs` doc), which is why derivation must go through checked-in facts.
- **Generate route tables.** Generate `IMAGE_MODEL_CAPS`, `VIDEO_MODEL_CAPS`, `CANDLE_*` model lists,
  `MODEL_TABLE` (id→engine join + defaults), `VIDEO_ENGINE_IDS`, `TRAINER_IDS`, `WIRED_*_POSE_FAMILIES`
  and `NATIVE_CONVERTERS` from `runtime/*.json.modelMappings`, `videoModelMappings` and
  `trainerMappings` plus descriptors. The imported-model path (`catalog.rs:1378-1440`) is the template:
  builtins and imports go through **one** provider-facts lookup.
- **Move memory data out.** Take the memory contracts (58% of manifest bytes) out of the hand-authored
  manifest into a generated sidecar. Delete the in-place rewrite projection script. Calibrations and
  floors join the measured-evidence store already loaded by `memory_anchor.rs` and `memory_calibration.rs`.
- **Optional runtime refinement.** Workers can publish their descriptor-derived `ModelSpec` deltas at
  registration (`WorkerRegisterRequest.extra`, `contracts.rs:972-981`), so a server can detect a
  worker/facts mismatch. Under the teardown this is advisory and never a gate.

### D2. One option registry: extend `ADVANCED_KEY_RULES` into `OptionSpec`

`ADVANCED_KEY_RULES` (76 keys) and `TECHNIQUE_KEYS` (45 training keys) become a single core
registry. Each `OptionSpec` carries:

- `id` (wire key), `aliases` (legacy keys), `location` (`advanced.*` vs typed top-level such as `hiresFix`, `upscale`)
- `value` type/shape (reuse `AdvancedShape`), `default`, `range/enum/step`. Ranges can be **bound to a declared limit** (e.g. `max = model.limits.hardMaxSteps ?? BASE_MAX_IMAGE_STEPS`)
- `applies_when`: a small predicate AST over `ModelSpec` and request context. Examples: `requires: conditioning ⊇ {img2img} ∨ family = sdxl`, `modes: [text_to_image]`, `excludes: [upscale, usePid, phases, poses, controlImage]`, `sampler ∉ accel_set`. The accelerated-sampler set comes from the engine sampler descriptor, so it is not a web literal.
- `intent_class`: `what_to_make` (shared, replayed) vs `machine_budget` (denied). This is the existing allow/deny axis.
- `ui`: `group, label, help, control, order, advanced:boolean`, `persist: sticky|session|none`.
- `engine_binding`: the gen_core request field or hook stage it feeds (coordinate with the inference-side researcher).

The training pipeline already follows this pattern: descriptor `techniques.*` → target `limits.supports*` →
web toggle → shared `technique_value` accessor at API and worker (`training.rs:60-80`). D2 generalises it.

### D3. One validator, invoked identically at every boundary

- `core::validate(spec: &ModelSpec, req: &TypedRequest, ctx) -> Vec<Issue{option, code, message, severity}>`.
- Call it from the API create path, the retry/duplicate merged path (closing the §4.4 gap), the MCP tools and the worker preflight. The worker keeps only checks that need local state, such as installed files and memory admission.
- Remove the duplicated constants: `PROMPT_ENHANCE_MAX_TEMPERATURE`, hard-coded `1..=8`, `{2,4}`, the SCAIL-2 `6`, temporal upsample `2`, and the enhance-prompt and LTX-2.5 model-id allowlists.
- Web validation becomes rendering of `Issue`s. Either serve resolved option availability and issues from `POST /models/:id/validate` (debounced), or compile the AST to a JS interpreter from the same JSON. **Tradeoff:** server-only evaluation means one implementation and a round-trip latency. A shared AST means instant feedback and two interpreters, held to agreement by a fixture corpus.
- Routing predicates (`routing/{mlx,candle}.rs`) stop encoding option refusals per model. A lane declares the options and conditioning it serves, and refusal comes from the same applicability evaluation that gates the UI. `macSupport`/`candleSupport` already probe routing to drive the UI (`catalog.rs:46-56`); D3 inverts that, so both derive from declarations.

### D4. Generic advanced-option pipeline (UI → API → worker → engine)

1. **UI.** A generated panel renders `OptionSpec`s with `availability.available=true`, grouped and ordered. Disabled options show `availability.reason`. State is a single `Record<optionId, value>`, persisted and replayed as-is. Custom renderers cover the structured shapes.
2. **Builder.** `buildRequest(spec, state)` emits only non-default values by `emitPolicy`. This replaces `imageJobAdvanced.js`'s 40 bespoke spreads and the 17 registered builders' ad-hoc maps.
3. **API.** Deserialize into `TypedRequest{common, options: OptionMap}` with alias upgrade, validate (D3), and persist with `requestSchemaVersion`.
4. **Worker.** Receives `ResolvedOptions`, already typed and defaulted. Family arms read typed fields; nobody re-parses `advanced`.
5. **Engine.** Options bind to gen_core request fields or stage hooks (pre-text-encode, per-step, post-decode, post-process). A cross-cutting option such as Hires.fix or a new guidance trick is implemented once, behind a capability flag the descriptor publishes. It then shows up for every model whose descriptor says yes, with no table edits.

### D5. Persistence and versioning

- Add `requestSchemaVersion` to job payloads and recipe `rawAdapterSettings`. Write a core upgrader that maps legacy shapes, both `advanced` keys and top-level `hiresFix`/`upscale`, into the option map.
- Keep `WORKFLOW_SHARE_SCHEMA_VERSION = 1` readable forever through the alias table. Bump it only when the envelope structure changes.
- Studio settings switch from 47 named fields to option-map snapshots, keyed by option id and model id. A size budget still applies (`useStudioSettings.js`).
- Presets need no change, because they carry no `advanced` keys.

### D6. Tooling

- Every derived artifact is produced by a generator from checked-in inputs, and checked by a byte-compare test only where it is cheap and pin-independent. This follows the teardown: no new measurement gates and no per-PR dumps.
- Replace regex-over-Rust generators with typed dumps (`memory-matrix`), and source-digest drift (`matrix.rs`) with registry invariants.
- Delete the web cold-start seed and its parity tests.
- **[L] Sequencing suggestion:**
  1. typed `ModelSpec` loader with one merge rule, behind today's JSON;
  2. generate route tables from facts and assert equality with the hand tables, then delete the hand tables;
  3. `OptionSpec` registry seeded from `ADVANCED_KEY_RULES`, porting Hires.fix and `temporalUpsampleRounds` as walking-skeleton options end to end;
  4. shared validator at all boundaries;
  5. generated web panel;
  6. move memory data to the sidecar;
  7. request versioning and upgrader;
  8. tooling deletion.

---

## Uncertain / not verified

- **Row counts.** I counted 63, 34 and 61 for `IMAGE_MODEL_CAPS`, `CANDLE_IMAGE_ROUTES` and `MODEL_TABLE`. The prior audit had 77, 36 and 62. Mine come from grepping constructor literals in the const ranges. A row built by a helper or macro would be missed.
- **Retry/duplicate validation gap (§4.4).** Inferred from grep over `jobs.rs:924-1265`. Not traced through every helper or run. Some checks may sit inside `validate_raw_job_payload` or the worker.
- **Runtime vs test embedding.** I classified 8 worker files as test-only by locating a preceding `#[cfg(test)]`. I did not do that for all 21 `builtin.models.jsonc` embed sites.
- **`check-scaffold.mjs` invocation.** I saw it referenced from `desktop-linux-check.yml`. I did not find it in `npm run check` or `check.yml` steps, so I don't know exactly where it runs.
- **Manifest byte shares.** Computed on comment-stripped, re-serialised JSON, not on file bytes.
- **Engine descriptors as a full replacement.** I only verified that the facts carry the listed fields (conditioning, lora/lokr, quants, preview, prompt enhancement, samplers, schedulers, guidance). Whether `conditioning` is fine-grained enough to replace `ui.img2img`, `ui.multiReference`, `ui.controlModes` and `ui.viewAngles`, and to drive `capabilities[]`, needs the inference-side researcher. The `image.supportsGuidance`/`supportsNegativePrompt` duplication rests on the `engines.rs:11-13` comment, not on inspecting descriptor fields in the facts file.
- **`NATIVE_CONVERTERS` vs manifest converters (8 vs 4).** The extra ids appear in `models.rs` and `model_jobs.rs`. I did not establish whether they are live or stale.
- **`advanced.get` counts** are a lower bound, because keys are also read through constants.
- **Scripts category LOC** comes from overlapping filename greps and is approximate.
- **Slimming percentages and the sequencing** are estimates [L]. They depend on the inference and worker findings from other researchers.
- **Not examined in depth:** AudioStudio and YuE2 request flows, vector/VQA/interleave contracts, Character Studio and the Editor builders, film harness, desktop seeding beyond `SeedMode`.
