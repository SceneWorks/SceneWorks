# F — Worker map for a lane execution harness and a single admission pipeline

Repo: `SceneWorks` worktree at `a1d95e2c4` (origin/main). Read-only survey; nothing was built or run.
Path shorthand: `W/` = `crates/sceneworks-worker/src/`, `C/` = `crates/sceneworks-core/src/`.
Inference pin seen in `crates/sceneworks-worker/Cargo.toml:43-60`: `a43724b82fbae35c9227cef4bcdabba6dd1da12c`.

Confidence legend used below: **[H]** read the code at the cited line; **[M]** inferred from
signatures/comments plus partial reads; **[L]** extrapolated, not verified.

---

## 0. Size and shape at a glance

| Fact | Value | Source |
|---|---|---|
| Worker `src` total | 432,919 lines (`.rs`, incl. tests) | `wc -l` over `W/` |
| Largest files | `image_jobs/tests.rs` 30,261; `mlx_fit_gate.rs` 23,311; `video_jobs/tests.rs` 21,964; `image_jobs/base.rs` 20,329; `training_jobs.rs` 11,870; `model_jobs.rs` 10,826; `memory_route_registry.rs` 10,513; `candle_memory_strategy.rs` 10,132; `vram_gate.rs` 8,217 | same |
| `JobType` variants dispatched | 33 explicit arms + fallback | `W/lib.rs:2533-2789` |
| Image route enums | `ImageRoute` 27 (MLX), `CandleImageRoute` 39 | `W/image_jobs/base.rs:128`, `:676` |
| Video route enums | `VideoRoute` 14 (incl. `NoPlanLane`, `Stub`), `CandleVideoRoute` 11 | `W/video_jobs/mod.rs:220`, `:393` |
| `fn *_available(` | 62 | grep |
| `fn *raw_settings*(` | 68 | grep |
| `async fn ensure_*(` | 47 | grep |
| `fn generate_*stream*` | 70 | grep |
| `consume_gen_events*` call sites | 47 | grep |
| `start_gen_stream` (uncached) call sites | 15 | grep |
| `start_cached_gen_stream*` call sites | 48 | grep |
| `nvidia_vram_budget_gb(` (non-test) call sites | 34 | grep |
| `"memoryStrategy"` stamped by hand in raw_settings | 12 files (all candle bespoke lanes) | grep |
| `#[cfg(...)]`/`#[cfg_attr(...)]` attributes | 4,304 (all, incl. test gates); `#[cfg(target_os="macos")]` 1,210; `#[cfg(all(not(macos), feature="backend-candle"))]` 676; `not(any(` "neither" stubs 99; `cfg!(target_os="macos")` 45 | grep |
| Per-file cfg density | `image_jobs/base.rs` 330, `image_jobs.rs` 165, `lib.rs` 136, `mlx_fit_gate.rs` 104, `generator_cache.rs` 63, `video_jobs/mod.rs` 59 | grep |

The prior audit's "1,959 cfg predicates" counted a narrower predicate set; my 4,304 counts every
`#[cfg`/`#[cfg_attr` attribute including `test` gates. Both describe the same problem. [H]

---

## 1. Job lifecycle end to end

### 1.1 Process and loop

1. `run_worker_loop` `W/lib.rs:1924`
   - logs gen-core contract version (`:1929`), applies the MLX GPU memory ceiling once via
     `initialize_worker_gpu_with(.., generator_cache::apply_gpu_memory_limit)` (`:1939-1943`),
     persists weight-seal digests (`:1946-1948`), spawns MLX telemetry (`:1952-1954`).
   - CUDA startup probe `cuda_startup_health` (`:1602`) → `discover_gpu` → `register_worker_with_retry`
     (`:2257`); capabilities come from `engines::registry_capabilities` (`W/engines.rs:1042`) and
     the GPU probe (ARCHITECTURE.md:35-55).
   - Starts one resolved-cache maintenance slot (retention checkpoint / promotion drain, epic 19703;
     `:1978`, `:2088-2121`).
   - Loop: MLX poisoned-worker recycle check (`mlx_worker_recovery`, `:2007-2049`), periodic GPU
     re-probe (`:2050-2053`), `poll_once` raced against shutdown (`:2066-2074`).
2. `poll_once` `W/lib.rs:2293` — live MLX limit sync (`:2302-2304`), idle/unhealthy heartbeat,
   `POST /api/v1/jobs/claim` (`:2330-2338`), returns `JobSnapshot` (`C/contracts.rs:1066`).
3. `run_job_with_shutdown` `W/lib.rs:2209` — runs the job future un-dropped; on SIGTERM trips a
   shared `CancelFlag`, waits `shutdown_timeout_seconds`, posts `mark_job_canceled`
   (`W/progress.rs:220`).
4. `run_utility_job` `W/lib.rs:2468` → wraps everything in `load_progress::with_job`
   (`W/load_progress.rs:466`, request-scoped latest-wins load progress).
5. `run_utility_job_inner` `W/lib.rs:2484`:
   - `JobMetricsProbe::start` (`:2501`; `W/job_metrics.rs:76`).
   - `with_shutdown_flag` (`W/progress.rs:649`).
   - **Model-source guard for every job kind**: `external_library_runtime::RuntimeSourceGuard::begin(job_type, payload, settings)`
     on a blocking thread with `heartbeat_while_blocking` (`:2510-2532`;
     `W/external_library_runtime.rs:73`; `W/progress.rs:488`). This is already a generic
     "plan the model closure before the lane" stage (sc-19708).
   - Big `match job.job_type` (`:2533-2789`) → per-kind entrypoint; each mapped to a
     `(&'static str user message, WorkerError)` pair.
   - On success `source_guard.finish_success()` (records resolved-cache promotion hint); on failure
     `source_guard.classify_failure` (`:2793-2798`).
   - Failure handling (`:2806-2903`): `Canceled` is silent; else MLX poison detection
     (`mlx_worker_recovery::global().observe`), CUDA driver annotation (`annotate_cuda_driver_failure`),
     `fail_job` (`W/progress.rs:188`) with retry-until-persisted for poisoned MLX workers.
   - Always: `post_generation_metrics` (`W/job_metrics.rs:283`) and an `Idle` heartbeat (`:2905-2909`).

**Harness note [H]:** steps 4-5 are already a uniform job harness for *every* job kind. The
lane-level duplication starts inside the per-kind entrypoint.

### 1.2 Job kinds and entrypoints

`JobType` is declared in `C/contracts.rs:221` (string enum). Dispatch arms in `W/lib.rs`:

| JobType | Entrypoint (file:line) | Engine surface used | Model residency | Own admission |
|---|---|---|---|---|
| Placeholder | `run_placeholder_job` `W/lib.rs:2915` | none | — | — |
| ImageGenerate, ImageEdit | `run_image_generate_job` `W/image_jobs.rs:587` → `_unexplained` `:596` | `Generator` via registry, or bespoke provider structs | generator_cache (most) / uncached `start_gen_stream` (15 lanes) | yes, many (§4) |
| VectorGenerate | `run_vector_job` `W/vector_jobs.rs:1096` | vector provider | — [M] | `vector_admission` (`W/vector_admission.rs`) |
| ImageDetail | `run_image_detail_job` `W/image_jobs/detail.rs:480` (stub twin `W/image_jobs.rs:4304`) | Generator | generator_cache | `admit_sdxl_bespoke_memory` [M] |
| ImageVqa / ImageInterleave | `run_vqa_job` / `run_interleave_job` `W/sensenova_jobs.rs:624`, `:1038` (stub twins `:949`, `:1492`) | SenseNova | [M] | — [M] |
| VideoGenerate / VideoExtend / VideoBridge / PersonReplace | `run_video_generate_job` `W/video_jobs/mod.rs:666` | Generator (video) | generator_cache | `video_admission` + per-family `vram_gate::*_fit_error` |
| AudioGenerate | `run_audio_generate_job` `W/audio_jobs.rs:980` → `_using` `:997` | `inference_runtime::load_audio` (audio registry) | **uncached**, loaded per job inside `spawn_blocking` (`W/audio_jobs.rs:~1672-1683`) | `yue_admission::check` (`:1012-1027`), `yue2_admission` |
| LoraTrain | `run_lora_train_job` `W/training_jobs.rs:215` | `Trainer` (`inference_runtime::load_trainer` `W/inference_runtime.rs:511`) | evicts generator + transform caches first (`W/training_jobs.rs:3677-3678`) | trainer-specific [M] |
| ControlTraining | `control_training_jobs::run_control_training_job` `W/control_training_jobs.rs:640` | Trainer | — | — |
| TrainingCaption | `run_training_caption_job` `W/caption_jobs.rs:187` (stub twin `:682`) | `Captioner` (`W/inference_runtime.rs:519`) | — [M] | — |
| DatasetParquetImport / DatasetAnalysis / CatalogAnalysis / DatasetFaceAnalysis / DatasetSubjectMask / FaceLikenessCompare | `W/dataset_parquet_jobs.rs:46`, `W/dataset_analysis_jobs.rs:70`, `W/catalog_semantic_jobs.rs:1315`, `W/face_analysis_jobs.rs:328`, `W/subject_mask_jobs.rs:186`, `W/face_likeness_compare_jobs.rs:186` (each with a `not(any(...))` twin) | embedders/detectors | analysis-common batching `W/analysis_jobs_common.rs:158` | — |
| PromptRefine | `run_prompt_refine_job` `W/prompt_refine_jobs.rs:1099` (twin `:1803`) | `TextLlm` | `refine_model_cache` (`cache_thread`, evict-before-load=true) | — [M] |
| ModelDownload / LoraImport / LoraDownload / ModelImport / ModelConvert | `W/model_jobs.rs:63`, `:5327`, `:367`, `:6830`, `:1602` | none (CPU utility) | — | — |
| FrameExtract / TimelineExport / PersonDetect / PersonTrack | `W/media_jobs.rs:90`, `:2046`, `:440`, `:1740` | ffmpeg / onnx | — | — |
| PoseDetect / KpsExtract (macOS or candle only) | `W/pose_jobs.rs:1138`, `W/kps_jobs.rs:385` | ort CoreML/CUDA, SCRFD | — | — |
| ImageUpscale / DatasetUpscale | `W/upscale_jobs.rs:965`, `:1730` (Iris restore `:1448`) | Generator/Transform | `with_cached_generator` (`W/upscale_jobs.rs:795`), `transform_cache` | `iris_restore` admission [M] |
| ImageSegment / ImageDepth | `W/segment_jobs.rs:125`, `W/depth_jobs.rs:83` | SAM3 / Iris depth | [M] | — |
| VideoUpscale | `video_jobs::seedvr2::run_video_upscale_job` `W/video_jobs/seedvr2.rs:1721` | Generator via `generate_video` (`:1564`) | generator_cache | video funnel |

Every GPU kind that has a "neither" build has a cfg-twin entrypoint (e.g. `W/caption_jobs.rs:187`
is `cfg(any(macos, candle))`, `:682` is `cfg(not(any(...)))`). [H]

### 1.3 Image job lifecycle (`run_image_generate_job_unexplained`, `W/image_jobs.rs:596-1000`)

Step-by-step (all [H]):

1. **Payload validation (family-specific, hard-coded at the top):** MOSS native request
   (`:601-609`), Supra2 interface (`:613-621`).
2. `ImageRequest::from_payload` (`:622`; struct `C/image_request.rs:140`).
3. Generic validators: projectId (`:623`), FLUX.3 layout (`:628`), hires-fix (`:629`), prompt
   enhancement (`:630`), FLUX.3 precision tier (`:633-637`), **geometry envelope**
   `admission_geometry::refuse_over_envelope` (`:644-653`; `C/admission_geometry.rs`), decoder
   selection (`:654-681`).
4. Project lookup and `assets/images` dir creation (`:682-685`).
5. Upscale multiplier (`:692-700`).
6. **Route resolution** (cfg-split):
   - macOS: `prepare_image_route_with_progress(.., prepare_image_route)` (`:706-713`;
     `W/image_jobs/stream.rs:161`; `W/image_jobs/base.rs:511`).
   - candle: `prepare_candle_image_route` (`:741-748`; `W/image_jobs/base.rs:1417`).
   - neither: no route (`:759-764`).
7. `ImagePlan::with_count_and_adapter` using `route.kind().image_count()` and `.adapter_label()`
   (`:721-730`, `:749-758`; plan struct `W/image_jobs.rs:1966`).
8. Workflow share hash (`:767-770`).
9. LoRA family-compat check (`:777-783`, `C/lora_family`), then route-allowlist LoRA guard
   (`applies_request_loras`, `:789-830`).
10. Busy heartbeat + `Preparing` progress (`:834-847`).
11. **Pre-load heartbeat pump** for the whole dispatch region (`HeartbeatPump::start`, `:860`;
    `W/progress.rs:541`) — added because pre-load admission took 95 s on Krea (comment `:849-859`).
12. **Dispatch** `match route.kind()` → one `generate_*_stream` per arm (MLX arms `:866-1258`; candle
    arms `:1272-1775`). Every arm takes the same 7 args `(api, settings, job, &plan, &project_path, backend, &mut asset_writes)`
    except prepared routes that unwrap `PreparedImageRoute::*` payloads first (e.g. `:1037-1060`).
13. If unhandled: `mlx_weights_gap` (`W/image_jobs/stream.rs:950`) / `candle_weights_gap`
    (`W/image_jobs/base.rs:2043`) fail loudly; else `dispatch_legacy_fallback` →
    `generate_stub_stream` (`W/image_jobs.rs:513`, `:1901`).
14. Inline upscale post-pass `apply_inline_upscale` (`:1840-1851`; fn `W/image_jobs.rs:3269`).
15. `drop(preload_heartbeat)` (`:1864`).
16. Warning collection (`lane_controls::job_warning_messages`) and terminal `Completed` update with
    `streaming_result(&plan, &asset_writes)` (`:1870-1898`; `W/image_jobs.rs:3683`).
17. Wrapper `converted_install::explain_job_result` rewrites errors for convert-at-install models
    (`W/image_jobs.rs:593`; `W/converted_install.rs:216`).

### 1.4 Video job lifecycle (`run_video_generate_job`, `W/video_jobs/mod.rs:666-1293`)

1. `VideoRequest` parse, `video_preflight` (`W/video_jobs/mod.rs:582`), reference-audio conditioning
   (`resolve_reference_audio_conditioning`), `VideoPlan::new` (`:1411`).
2. Busy heartbeat + `Preparing`, `HeartbeatPump::start` (`~:700`).
3. **Route**: macOS `resolve_video_route` (`:270`) → `resolve_video_route_with` (`:291`); candle
   `resolve_candle_video_route` (`:473`) + `reject_unsupported_candle_video_route` (`:427`).
4. **Dispatch** (`:728-1238`): each arm calls a family handler that returns a *different tuple
   shape*: `(decoded, status, lightning)` for SCAIL replace, `(decoded, raw_settings)` for H3 and
   Krea realtime, plain `decoded` for Wan; the dispatcher then normalizes to
   `(decoded, adapter, raw_settings, replacement_status)` (`:761`, `:1040`, `:1216`), computing
   raw_settings itself for some lanes (e.g. `wan_vace_raw_settings` `:1101`).
5. `record_veda_outcome` (`:1242`; fn `:1648`), `encode_media` (`:1252`), `video_asset_fact`
   (`:1270`; fn `:1666`), `streaming_result` (`:1271`; fn `:1777`), drop pump, `Completed` (`:1273-1290`).

Video already has a *de facto harness*: every family handler builds a `VideoGenInput`
(`W/video_jobs/wan.rs:1355`) and calls `generate_video` / `generate_video_using`
(`W/video_jobs/wan.rs:2770`, `:2806`) — 22 call sites across `bernini.rs`, `ltx.rs`, `candle.rs`,
`minimax_h3.rs`, `mochi.rs`, `svd.rs`, `seedvr2.rs`, `scail2.rs`, `vace.rs`, `wan.rs`,
`krea_realtime.rs`. [H] The funnel owns: Veda predictor resolution, admission (inside the cache
callback), cache acquisition, request build (`run_loaded_video_generation` `:2207`), the event loop
with stall watchdog, cancel and metrics. Image has no equivalent single funnel.

### 1.5 Event consumption, progress, preview, heartbeat, cancel

**Image stream shape** (`W/image_jobs/stream.rs`):
- `GenEvent` enum (`:1-85`): `AssetWritten`, `Step`, `Decoding`, `Upsampling`, `Loading`,
  `Preview`, `PromptEnhancement`, `CheckpointWeightFacts`, `ImageFacts`, `Image`.
- Producers: `start_gen_stream` (uncached, `:605`), `start_cached_gen_stream*` (`:644`, `:704`,
  `:775`, `:816`, owned core `:866`) which wraps
  `generator_cache::with_cached_generator_for_request_using_cache` and emits
  `image_pipeline_load_start/complete` (`:897-918`).
- Per-item drivers: `drive_gen_items` (`:244`), `_reported` (`:285`), `_scored` (`:337`),
  `_scored_reported` (`:369`), `drive_native_prompt_batch` (`:416`); progress mapping
  `send_gen_progress` (`:171`); `preview_sink_for` (`:102`).
- Consumer: `consume_gen_events` → `consume_gen_events_with_disclosure`
  (`W/image_jobs/base.rs:16107`, `:16142`):
  - `CancelJoinGuard::new(cancel, blocking)` (`:16205`); `interval(progress_report_interval)`.
  - `select!` loop (`:16241`): `Step` → `update_job` + cancel peek every 2 s
    (`cancel_requested_peek` `W/progress.rs:609`; `begin_image_cancel` `W/image_jobs.rs:3879`,
    called at `:16268`); `Loading` (`:16336`); `Preview` → data-URL (`encode_preview_data_url`
    `:16062`, slim result `:16090`) (`:16357`); `AssetWritten` (`:16386`); `Image` → **`write_image_asset`**
    (`:16418`, `:16483`); interval tick → Busy heartbeat + cancel peek (`:16523-16533`).
  - After loop: cancel terminal, metrics (`build_image_metrics` `:16019/16040`,
    `PhaseTimer` load/sample/decode ms) and `post_generation_metrics`.
- 47 call sites; every image lane (registry or bespoke) funnels its pixels through it. [H]

**Video loop**: separate implementation in `generate_video_using` (`W/video_jobs/wan.rs:3312-3500`):
own `CancelJoinGuard` (`:3312`), `select!` (`:3346`), stall watchdog with pre-step and step
windows (`video_stall_window` `:2359`, `VIDEO_STALL_TIMEOUT` `:2333`,
`VIDEO_PRE_STEP_STALL_TIMEOUT` `:2349`), `begin_video_cancel` (`:2587`). Not shared with image.

**Audio**: its own progress/cancel (`check_cancel`, `audio_progress`; `W/audio_jobs.rs:1028-1138`).

**Heartbeat layers** (four, overlapping by design): idle loop heartbeat (`W/lib.rs:2307-2328`),
`heartbeat_while_blocking` (`W/progress.rs:488`), `HeartbeatPump` region (`W/progress.rs:541`),
stream interval tick (`W/image_jobs/base.rs:16523`), plus `with_job_heartbeat` (`W/progress.rs:586`).

**Load progress**: process-wide, request-scoped session (`W/load_progress.rs:1-5`, `:227`,
`generator_progress` `:253`, `spawn`/`spawn_blocking` `:444`/`:456` capture the session).

### 1.6 Output writing and receipts

- **Single image writer**: `write_image_asset` (`W/image_jobs.rs:2990`). Takes
  `(plan, index, seed, w, h, pixels, adapter, raw_settings, project_path)`; stamps checkpoint
  weight facts if the lane did not (`:3003-3020`, `checkpoint_weight_facts_host`), writes PNG +
  sidecar + generation set. Every image lane reaches it through `consume_gen_events`. [H]
- **raw_settings builders** (68 functions). The generic one is `mlx_raw_settings_for_bundle`
  (`W/image_jobs/base.rs:5869`): copies `request.advanced`, scrubs worker-owned keys
  (memory-strategy disclosure, prefix-KV, MOSS reports, Supra2 revisions; `:5884-5893`), adds
  `realModelInference`, `repo`, `numInferenceSteps`, `guidanceScale`, `mlxQuantize`, `hiresFix`
  (`:5911-5930`). It is reused by the candle generic lane (`:14373`). Bespoke lanes each have their
  own (e.g. `kolors_ipadapter_raw_settings` `W/image_jobs/kolors_ipadapter.rs:189`,
  `qwen_edit_candle_raw_settings` `W/image_jobs/qwen_edit_candle.rs:603`,
  `krea_control_candle_raw_settings` `W/image_jobs/krea_control_candle.rs:421`,
  `ltx_raw_settings` `W/video_jobs/ltx.rs:1885`, `minimax_h3_raw_settings`
  `W/video_jobs/minimax_h3.rs:718`, `wan_raw_settings` `W/video_jobs/wan.rs:90`).
- **Memory-decision stamping** is not uniform:
  - candle generic: worker-owned disclosure key inserted after final admission (`W/image_jobs/base.rs:15310`), telemetry via `CandleMemoryEvaluation::selection_telemetry` (`W/candle_memory_strategy.rs:739-797`).
  - candle bespoke lanes: hand-inserted `"memoryStrategy"`, `"memoryEvidenceRevision"`, `"memoryReceipt"` (e.g. `W/image_jobs/kolors_ipadapter.rs:400-414`) — 12 files, each a copy.
  - qwen-edit: six keys inserted (`W/image_jobs/qwen_edit_candle.rs:1004-1024`).
  - MLX: the evaluation is applied inside the drive closure; disclosure via `ImageFacts`/telemetry [M].
  - video: `VideoAdmissionOutcome` written into `VideoGenInput.memory/memory_context` by
    `apply_video_admission_outcome` (`W/video_jobs/wan.rs:2188`); Veda outcome stamped by
    `record_veda_outcome` (`W/video_jobs/mod.rs:1648`).
- **Metrics**: `image_settings_metrics` / `build_image_metrics` (`W/image_jobs/base.rs:15812`,
  `:16019`), `build_video_metrics` (`W/video_jobs/wan.rs:2725`), `JobMetricsProbe`
  (`W/job_metrics.rs:76`).

### 1.7 Error mapping

- `WorkerError` (`W/error.rs:4-22`): `Http | Io | Json | ProjectStore | Api{status,detail,code} |
  InvalidPayload(String) | ExternalLibraryUnavailable(String) | Engine(String) | Canceled(String)`.
- `classify_engine_error` (`W/error.rs:118`): `gen_core::Error::Canceled → Canceled`,
  `Unsupported → InvalidPayload`, else `Engine` + CUDA driver guidance.
- **Admission refusals are `InvalidPayload(String)`** everywhere (e.g. `W/mlx_fit_gate.rs:4984-5010`,
  `W/image_jobs/kolors_ipadapter.rs:394-399`, `W/image_jobs/qwen_edit_candle.rs:984`). There is no
  typed refusal (binding phase, needed/available, suggestions) at the worker boundary; video core has
  one (`VideoAdmission::Refused{message,geometry,needed_gb,available_gb}`, `C/video_request.rs:2402-2420`)
  that is flattened to a string by `VideoAdmissionOutcome.refusal: Option<String>`
  (`W/video_admission.rs:2425-2429`).
- Job-level message prefix per kind is in the dispatch table (`W/lib.rs:2533-2789`).

---

## 2. The engine boundary

### 2.1 Facade: `W/inference_runtime.rs` (947 lines)

- Backend alias: `use runtime_cuda as platform_runtime` (candle, `:20-21`) / `use runtime_macos as platform_runtime` (`:22-23`).
- `catalog()` → `platform_runtime::catalog()` `OnceLock` (`:32-41`); "neither" build returns an empty
  `ProviderRegistry` (`:73-97`).
- Entry points: `media()` `:73`, `load(id, &LoadSpec) -> Box<dyn Generator>` `:426`,
  `load_transform` `:433`, `load_audio` `:153`, `load_audio_transform` `:263`,
  `load_voice_embedder` `:284`, `text()` `:299`, `load_trainer` `:511`, `load_captioner` `:519`,
  `load_image_embedder` `:527`, `load_text_embedder` `:538`, `media_descriptor` `:339`,
  `media_encoder_contract` `:408`, `imported_model_descriptor` `:462`, `checkpoint_adapter` `:486`,
  `CHECKPOINT_BACKEND` `:502-505`, `capability_snapshot_json` `:55`,
  `memory_contract_surface_registry` `:99`.
- Memory surfaces queried by the worker through the registry: `media().memory_strategy_contract(engine_id, spec)`
  (`W/mlx_fit_gate.rs:5022-5027`) and `media().footprint(engine_id, spec)` (`W/mlx_fit_gate.rs:6437-6450`).

This facade is already backend-neutral at the type level (`gen_core::Generator`, `LoadSpec`,
`ProviderRegistry`). The cfg split is confined to which `platform_runtime` is linked. [H]

### 2.2 Two ways the worker calls an engine

1. **Registry generator** (`Box<dyn Generator>`): `inference_runtime::load` injected as the
   `load_generator` closure into `start_cached_gen_stream_with_request_state_using`
   (`W/image_jobs/stream.rs:816`) or `generate_video_using` (`W/video_jobs/wan.rs:2806`). The
   request goes through `memory_strategy::generate_with_scope[_report]`
   (`W/memory_strategy.rs:232`, `:251`) which binds the admitted `MemoryRunContext`.
2. **Bespoke provider by name** (candle only): concrete types loaded from private `*Paths`
   structs, e.g. `IpAdapterKolors::load_with_memory_context(&paths, ctx)` +
   `generate_with_memory_context(&ctx, &req, &reference, ..)`
   (`W/image_jobs/kolors_ipadapter.rs:459-499`), `QwenEdit::load_with_memory_context` (`W/image_jobs/qwen_edit_candle.rs:1083`, `:1140`),
   strict-control providers via the `CandleStrictControl` trait (`W/image_jobs/candle_strict_control.rs:45-130`).
   These bypass `LoadSpec`, the generator cache, `apply_residency_policy` and the generic preview
   sink (`W/conditioning_fit.rs` module docs lines 1-30 explain why). They use the uncached
   `start_gen_stream` (`W/image_jobs/stream.rs:605`).

### 2.3 Where the worker builds `GenerationRequest`

Non-test construction sites (`GenerationRequest {` grep): `image_jobs/moss.rs` 10,
`mlx_fit_gate.rs` 7, `audio_jobs.rs` 5, `sensenova_jobs.rs` 4, `image_jobs/base.rs` 3,
`generator_cache.rs` 3, and 1-2 each in `zimage.rs`, `qwen.rs`, `mage_finetuned.rs`,
`krea_imported.rs`, `flux2.rs`, `yue2_jobs.rs`, `memory_strategy.rs`, `vram_gate.rs`,
`video_jobs/wan.rs`, `upscale_jobs.rs`, `training_jobs.rs`, `zimage_comfyui_candle.rs`,
`sensenova.rs`, `sdxl_imported.rs`, `sdxl_control.rs`, `sdxl.rs`, `qwen_comfyui_candle.rs`,
`pulid.rs`, `prompt_upsampling.rs`, `krea_turbo_raw.rs`, `krea_multiphase.rs`,
`krea_edit_candle.rs`, `krea_edit.rs`, `krea_control.rs`, `kolors.rs`, `flux3_references.rs`,
`flux2_comfyui_candle.rs`, `flux1_control.rs`, `detail.rs`, `checkpoint_plan.rs`, `bernini.rs`,
`execution_planner.rs` (plus smoke/test files). ~45 production sites. [H]

The two canonical builders:

| Builder | Fields set | Where |
|---|---|---|
| Image: `render_on_surface` | `prompt`, `prompt_batch`, `negative_prompt`, `width`, `height`, `count`, `seed`, `steps`, `guidance`, `true_cfg`, `sampler`, `scheduler`, `scheduler_shift`, `guidance_method`, `use_pid`, `text_style_gain`, `memory`, `conditioning` (built by `build_lane_conditioning_with_alpha` `:8448`), `flux3_reference_output`, `output_channels`, `cfg_interval`, `caption_overflow`, `preview`, `cancel`; then `PromptEnhance::apply`, FLUX.3 `layout.apply` | `W/image_jobs/base.rs:8259-8337`; executed via `memory_strategy::generate_with_scope_report` (`:8350`) or `flux3_references::render_reference_request` (`:8343`) |
| Video: `run_loaded_video_generation` | `veda`, `prompt`, `negative_prompt`, `width`, `height`, `frames` (unless auto-duration), `fps`, `auto_duration`, `temporal_upsample_rounds`, `reference_image_short_edge`, `reference_image_resize_mode`, `steps`, `guidance`, `scheduler_shift`, `sampler`, `scheduler`, `seed`, `conditioning`, `control_scale`, `video_mode`, `enhance_prompt`, `use_uncensored_enhancer`, `enhance_max_tokens`, `enhance_temperature`, `motion_bucket_id`, `noise_aug_strength`, `decode_chunk_size`, `conditioning_fps`, `softness`, `memory`, `cancel` | `W/video_jobs/wan.rs:2207-2246` |

Per-family fields therefore live in two flat bags: `VideoGenInput` (`W/video_jobs/wan.rs:1355-1500`, comments name the family per field: "LTX-only", "SVD-only", "SeedVR2", "MiniMax-H3") and the
`render_on_surface` argument list (26 parameters plus `LaneRgbaSurface` `W/image_jobs/base.rs:8162`).
Bespoke candle lanes build engine-specific request structs instead (`IpAdapterKolorsRequest`
`W/image_jobs/kolors_ipadapter.rs:483-496`, etc.).

### 2.4 `LoadSpec` construction pipeline (generic MLX image lane)

1. `load_spec(weights_dir, quant, adapters, ip_adapter_dir)` (`W/image_jobs/base.rs:5934`).
2. `mlx_load_quant_for_resolved_artifact` (`:5959`), PiD `with_pid` (`:10897-10907` region),
   FLUX.3 plan (`:10900+`).
3. `attach_manifest_text_encoder` → `ManifestTextEncoderAttachedSpec` (`:5997-6036`).
4. `attach_required_components`, `attach_selected_decoder` (`:7264`, `:7396`).
5. `prepare_mlx_load_policy` (`:6542`) — reads `memory_route_registry` (`MemoryRouteMode::from_mlx_request`
   at `:10958`), applies declared Sequential/deferred shape; `declared_sequential_resident_alternative` (`:6616`).
6. `mlx_fit_gate::bind_decode_quality_policies_from_manifest` + `attach_decode_quality_binding`
   (`W/mlx_fit_gate.rs:389`, `:455`).
7. Candle equivalents: `candle_certified_load_spec` (`W/image_jobs/base.rs:4032`), seals
   (`seal_sdxl_endpoint_load_spec` `:4140`, `seal_recursive_receipt_load_spec` `:4489`,
   `seal_ideogram_load_spec` `:4595`), `apply_declared_candle_image_load_shape` (`:6097`),
   `apply_candle_qwen_load_shape` (`:6138`), `apply_request_scoped_candle_residency` (`:11843`).
8. Video: `video_load_spec(&VideoGenInput)` (`W/video_jobs/wan.rs:1560`).

### 2.5 Catalog lookups

- `engines::MODEL_TABLE` (`W/engines.rs:37`, 62 rows of `ModelRow{sceneworks_id, engine_id,
  default_repo, default_steps, default_guidance, adapter_label}` `:22-35`); `mlx_model` (`:990`) returns
  `ResolvedModel` (`:911`) combining the row with the linked descriptor; `engine_id_for_model`
  (`:1024`), `default_repo_for` (`:1015`). Despite the name `mlx_model`, it is consumed by the candle
  lane too [M].
- `VIDEO_ENGINE_IDS` (`W/engines.rs:846`), `TRAINER_IDS` (`:873`).
- Manifest entry (`request.model_manifest_entry`, a JSON object forwarded in the payload) is the
  source of `candle.vramGbByTier` (85 reads), `minMemoryGb` (45), `turboFit` (37),
  `sequentialPeakGb` (25), `calibrations` (21), `loadShape` (16), `calibrationFingerprint[ByTier]`
  (19), `memoryEvidenceRevision` (8), `footprint` (6), `peakGbByTier` (4), `admissionGeometry` (3),
  `decodeTileSaveGb`/`chunkAttnSaveGb` (2 each). [H, grep counts]
- ~1,015 literal model ids in worker (prior audit) — confirmed qualitatively: ladders compare
  `request.model == "kolors"` etc. (e.g. `W/image_jobs/base.rs:13449`, `:1233`).

---

## 3. Representative lanes, step by step

Legend: **H** = identical (or near-identical) across lanes → harness candidate; **K** = lane-specific
→ hook candidate; **X** = accidental divergence (same intent, different code) → unify.

### 3.1 Generic MLX t2i — `ImageRoute::Mlx` → `generate_stream` → `generate_stream_using`

`W/image_jobs/base.rs:10585` (wrapper), `:10613-11581` (~970 lines, macOS only).

| # | Step | Line | Class |
|---|---|---|---|
| 1 | MOSS divert | `:10624-10635` | X (family branch inside generic lane) |
| 2 | FLUX.3 owned references + layout (normalize canvas) | `:10636-10652` | K (family pre-plan hook) |
| 3 | `mlx_model(&model)` → engine_id | `:10653-10657` | H |
| 4 | `ensure_boogu_tier_present`, `ensure_ideogram_tier_present` (on-demand tier download) | `:10662-10663` | K → H (generic "ensure tier present" by manifest) |
| 5 | `resolve_weights_dir` | `:10665` | H |
| 6 | FLUX.3 precision plan overrides weights dir | `:10670-10678` | K |
| 7 | `resolve_adapters`, adapter identities | `:10679-10683` | H |
| 8 | Steps/guidance/sampler/scheduler/guidance-method normalization vs descriptor (`normalize_sampling_knob`) | `:10690-10760` | H (+ K: `realvisxl_lightning` sampler override `:10700-10704`) |
| 9 | true-CFG, negative prompt, `lane_controls::resolve_lane_request_controls` | `:10760-10790` | H |
| 10 | Seeds | `:10790-10795` | H |
| 11 | Reference conditioning resolution (`resolve_generic_lane_inputs` `:9762`, zimage/ideogram/qwen-2.1 edits) | `~:10760-10780` | K (family conditioning hook) |
| 12 | RGBA channels, hires plan, ideogram caption, PromptEnhance, PiD, native prompt batch, text-style gain | `~:10774-10820` | mix: hires/PiD/batch = H (cross-cutting), ideogram caption = K |
| 13 | Live budget `generator_cache::mlx_tier_budget` | `:10862` | H (backend probe) |
| 14 | `prepare` closure: LoadSpec + TE attach + route declaration + `prepare_mlx_load_policy` + decode-quality binding + `MlxRequestPlan` + `MlxRequestInputs` | `:10863-11110` | H (all generic, parameterized by request facts) |
| 15 | Tier candidates (`tier_key_for_resolved_dir`, installed tiers) | `~:11082-11115` | H |
| 16 | **`choose_mlx_request_tier`** loop: provenance, `prepare`, `mlx_fit_gate::preflight_request`, resident-alternative retry, `preflight_load_rejection` | `:11117-11160` | H (admission phase 1 = tier probe) |
| 17 | Quant reconcile | `~:11165-11250` | H |
| 18 | `mlx_raw_settings_for_bundle` | `:11251` | H |
| 19 | Likeness staging | `~:11254-11262` | H (identity scoring is cross-lane) |
| 20 | `start_cached_gen_stream_with_request_state_using` → cache acquire | `:11284` | H |
| 21 | In drive: scorer build, `mlx_fit_gate::evaluate_request` (admission phase 2, cache-state aware) and `apply_request_gpu_memory_limit` | `:11355-11367`, `:11437-11452` | H |
| 22 | Render: `render_on_surface` (native batch) or `generate_one_with_hires_on_surface` per seed; ideogram placeholder recovery; face likeness; image facts; prefix-KV record; FLUX.3 facts | `:11370-11545` | H (hires/batch/likeness) + K (ideogram, FLUX.3) |
| 23 | `consume_gen_events` | `:11557` | H |

### 3.2 Generic candle t2i — `CandleImageRoute::CandleTxt2Img` (and 6 siblings) → `generate_candle_stream` → `generate_candle_registry_stream_using`

`W/image_jobs/base.rs:13498`, `:13530`, `:13581-15791` (~2,210 lines, candle only).

| # | Step | Line | Class |
|---|---|---|---|
| 1 | MOSS divert (`generate_moss_stream_with_runtime`) | `:13561-13574` | X |
| 2 | FLUX.3 references | `:13597-13604` | K |
| 3 | engine id, `is_sd35`, `is_receipt_priced` | `:13643-13648` | K → data (provider facts) |
| 4 | `resolve_weights_dir` (+ ConvRot/NVFP4 re-pointing) | `:13690` | H |
| 5 | `resolve_adapters` | `:13754` | H |
| 6 | Conditioning: `resolve_candle_lane_inputs` | fn `:13397-13470` | K (twin of MLX `resolve_generic_lane_inputs`; X: separately coded) |
| 7 | Evict resident Iris transform before probing | `:13929` | H |
| 8 | Budget: `vram_gate::apply_vram_cap(gpu::nvidia_vram_budget_gb, cuda_vram_cap_gb)` | `:13936-13939` | H (backend probe) |
| 9 | Reclaimable credit `reclaimable_pool_gb` + `with_reclaimable` | `:13948-13950` | H |
| 10 | Ladder reserve from RAW probe `ladder_reserve_gb` | `:13955` | H |
| 11 | `engine_supports_sequential` (from `mlx_fit_gate`!) | `:13991` | H (naming X) |
| 12 | Downtier loop: per installed tier `candle_tier_fit` / `candle_certified_load_spec` / `evaluate_shared_image_priced` / Krea turbo `krea_turbo_fit_with_runtime` → `choose_downtier` | `:14056-14330` | H for the loop, K for the Krea-turbo branch |
| 13 | `mlx_raw_settings_for_bundle` | `:14373` | H |
| 14 | FLUX.3 source activation allowance; `predicted_peak_gb_with_adapter_bytes` | `:14391-14425` | K / H |
| 15 | Final admission for chosen tier: `evaluate_shared_image_priced_with_resources` (FLUX.3 refs) / `evaluate_shared_image_priced` / `evaluate_shared_image` | `:14589`, `:14631`, `:14682` | H (X: three entry variants) |
| 16 | Krea turbo resident/fits/reject/unverified handling; `FitDecision::Offload`/`resolve_offload`; sequential overflow; smaller-geometry suggestions | `:14794-15200` | K (Krea) + H (offload fallback) |
| 17 | Record admitted peak for reclaim | `:15202-15215` | H |
| 18 | `apply_request_scoped_candle_residency`, `apply_declared_candle_image_load_shape` | `:15288`, `:15293` | H |
| 19 | Disclosure into raw_settings | `:15310` | H |
| 20 | `start_cached_gen_stream_with_request_state_using` | `:15357` | H |
| 21 | Drive: `generate_one_with_hires_on_surface`, adapter reports, facts | `:15620-15760` | H |
| 22 | `consume_gen_events_with_disclosure` | `:15765` | H |

### 3.3 Qwen edit candle — `CandleImageRoute::QwenEdit` → `generate_candle_qwen_edit_stream`

`W/image_jobs/qwen_edit_candle.rs:739-1185`.

| # | Step | Line | Class |
|---|---|---|---|
| 1 | Resolve base (`resolve_qwen_edit_candle_base` `:475`) | `:749` | H (weights resolve) |
| 2 | Unsupported-axis refusal (`qwen_edit_candle_unsupported_request_axis` `:306`) | `:752-760` | K (capability declaration — should be descriptor data) |
| 3 | Lightning LoRA fetch (`ensure_qwen_lightning_lora_cached` `:657`) | `:778` | K (ensure hook) |
| 4 | User adapters | `:790` | H |
| 5 | raw_settings | `:804-812` | K (keys) / H (builder) |
| 6 | Budget probe `apply_vram_cap(nvidia_vram_budget_gb)` | `:897-899` | H (dup of 3.2 #8) |
| 7 | `evaluate_shared_image` closure | `:926-958` | H |
| 8 | `gate_with_evict_reclaim` (evict generator cache if reclaim would improve plan) | `:959` (fn `W/image_jobs/base.rs:13349`) | H |
| 9 | Reject message with installed-smaller tail | `:974-995` | H |
| 10 | Stamp six memory keys into raw_settings | `:1004-1024` | X |
| 11 | `note_loaded_peak` per LoadPlan | `:1034-1070` | H |
| 12 | `start_gen_stream` (UNcached) → `QwenEdit::load_with_memory_context` | `:1078-1083` | X (should be cached through one seam) |
| 13 | Per-item `generate_with_memory_context` | `:1140` | K (engine call) |
| 14 | `consume_gen_events` | `:1161` | H |

### 3.4 Krea control candle — `CandleImageRoute::KreaControl` → `generate_candle_krea_control_stream`

`W/image_jobs/krea_control_candle.rs:646-~1000` and the shared driver
`run_candle_strict_control` (`W/image_jobs/candle_strict_control.rs:132`).

| # | Step | Line | Class |
|---|---|---|---|
| 1 | ConvRot base ensure / resolve base | `:657-662` | K |
| 2 | Control weights ensure | `:669` | K (ensure hook) |
| 3 | Adapters | `:676` | H |
| 4 | raw_settings | `:705` | K |
| 5 | Branch tier `control_branch_tier_for_key` | `:741` | K (provider fact) |
| 6 | Conditioning floor `admit_conditioning_paths` x2 (weights floor) | `:786`, `:804` | H |
| 7 | Budget probe | `:830` | H (dup) |
| 8 | `gate_with_evict_reclaim` + `krea_control_fit::fit_ladder_for_entry_with_runtime` → `select_strategy` (`W/krea_control_fit.rs:721`, `:913`) | `:837-880` | H (selector) + K (control-lane candidate builder) |
| 9 | `note_loaded_peak` / Unverified logging | `:881-940` | H |
| 10 | Shared driver: `ConditioningAdmission::GatedInPreamble` skip (`candle_strict_control.rs:156-166`), control-kind validation, preprocess (pose/canny/depth), likeness, `start_gen_stream` (uncached), `provider.generate_one`, `consume_gen_events` | `candle_strict_control.rs:132-330` | H (already a harness for 5 lanes) |

**The `CandleStrictControl` trait is the closest existing thing to the proposed lane hook**:
required methods `engine_id`, `engine_label`, `stream_tag`, `out_width/height`,
`conditioning_admission` (no default, deliberately), `load`, `generate_one`
(`W/image_jobs/candle_strict_control.rs:45-130`). [H]

### 3.5 Kolors IP-Adapter candle — `CandleImageRoute::KolorsIpAdapter`

`W/image_jobs/kolors_ipadapter.rs:213-563`.

| # | Step | Line | Class |
|---|---|---|---|
| 1 | Resolve base | `:222-227` | H |
| 2 | Reference asset required + load | `:228-243` | H (reference loading is generic: `load_reference_image` `W/image_jobs/base.rs:8836`) |
| 3 | Ensure IP weights | `:244` | K (ensure) |
| 4 | Likeness staging | `:246-256` | H |
| 5 | Steps/guidance/ip_scale; sampler/scheduler normalized against a **curated menu**, not the descriptor | `:258-285` | X (generic lane uses descriptor) |
| 6 | Repo + raw_settings | `:286-297` | H/K |
| 7 | Work list, negative prompt, adapters | `:299-305` | H |
| 8 | Contract via `runtime_cuda::providers::kolors::memory_strategy::provider_contract_for_ip` (direct provider import, not via registry) | `:311-316` | X (bypasses `inference_runtime`) |
| 9 | Overlay receipt identity | `:317-325` | K |
| 10 | Budget probe | `:329-332` | H (dup) |
| 11 | Certification checks (`candle_certified_hf_artifact_path` x2) | `:334-352` | H |
| 12 | `evaluate_shared_bespoke_image(..., "character_image", Some("identity"), geometry, ...)` | `:353-399` | H |
| 13 | Stamp `memoryStrategy`, `memoryEvidenceRevision`, `memoryReceipt` + tracing + `emit_event` | `:400-439` | X (12 copies) |
| 14 | `admit_conditioning_paths` (weights floor) **after** the ladder | `:445-452` | X (ordering differs from krea-control) |
| 15 | `start_gen_stream` (uncached) → `IpAdapterKolors::load_with_memory_context` | `:454-474` | X |
| 16 | per-seed `generate_with_memory_context` + likeness | `:476-528` | K + H |
| 17 | `consume_gen_events` | `:533` | H |

### 3.6 Video: Wan and LTX (MLX) → `generate_video_using`

- Wan handler `generate_wan` (`W/video_jobs/wan.rs:35-89`): sampling (`wan_sampling` `:1221`),
  conditioning (`resolve_wan_conditioning` `:847` / clip `:984`), `ensure_wan_tier_present` (`:551`),
  `ensure_wan_lightning_present` (`:592`), tier dir+quant (`:509`), adapters (`:710`), build
  `VideoGenInput`, `generate_video`.
- LTX handler `generate_ltx` (`W/video_jobs/ltx.rs:1899-2018`): text-encoder selection
  (`resolve_ltx_text_encoder_for_model` `:862`), conditioning by mode (clip `:1628`, replace
  `:1816`, i2v `:1252`), `ensure_ltx_q8_present`/`bf16`/`gemma` (`:895`, `:931`, `:964`), model dir,
  gemma dir, `VideoGenInput{..}`, `generate_video`.
- Shared funnel `generate_video_using` (`W/video_jobs/wan.rs:2806-3541`):

| # | Step | Line | Class |
|---|---|---|---|
| 1 | Veda predictor resolve (blocking) | `~:2819-2850` | K (H3/Veda feature hook) |
| 2 | Admission identity from payload (model id, manifest, family, mode) | `~:2849-2858` | H |
| 3 | Stall policy (`video_stall_timeout` `:2418`) | `:2894-2906` | H |
| 4 | Spawn: imported-H3 preparation supervision | `~:2905-2970` | K |
| 5 | Cache callback: spec validation, `resolved_video_numeric_tier`, `spec_headroom_bytes`, reference count/shape/overlay, `VideoAdmissionInputs`, `live_video_runtime_state`, Veda host workspace, **`admit_video_generation_with_veda`** | `:3035-3207` | H |
| 6 | `apply_video_admission_outcome` | `:3213` | H |
| 7 | `run_loaded_video_generation` (builds `GenerationRequest`) | `:3223` | H |
| 8 | Cache acquisition variants: `with_cached_generator_for_request_using_cold_admission` / `_using` | `:3267`, `:3279`, `:3293` | H |
| 9 | Event loop + stall watchdog + cancel | `:3312-3500` | H |
| 10 | Metrics | after `:3500` | H |

Then back in `run_video_generate_job`: encode, fact, complete (§1.4). LTX and Wan differ only in
the handler (ensure + conditioning + input fields). [H]

### 3.7 MiniMax-H3 — `VideoRoute::MiniMaxH3` → `generate_minimax_h3` → `generate_minimax_h3_using`

`W/video_jobs/minimax_h3.rs:1511`, `:1543-1685`.

| # | Step | Line | Class |
|---|---|---|---|
| 1 | Imported-checkpoint guard + registry presence | `:1555-1566` | K |
| 2 | `prepare_imported_minimax_h3_dit` (async preparation) | `:1567` | K (import-plan hook; generic for imported checkpoints) |
| 3 | Partition request, validate partition | `:1568-1573` | K |
| 4 | Sampling (`minimax_h3_sampling` `:650`), Veda (`requested_h3_veda` `:592`), frame count | `:1574-1576` | K |
| 5 | Conditioning (`resolve_minimax_h3_conditioning` `:1455`) | `:1577-1579` | K |
| 6 | Load resolution (imported or tier) | `:1580-1587` | H-ish (weights resolve) |
| 7 | Task, reference short edge/resize mode, adapters | `:1588-1593` | K/H |
| 8 | raw_settings + imported identity | `:1594-1605` | K |
| 9 | `VideoGenInput{ dit_component_dir, text_encoder_component_dir, imported_h3_* ...}` | `:1606-1637` | H (bag) |
| 10 | `generate_video_using` | `:1638` | H |

Candle twin `generate_candle_minimax_h3_using` (`:1810`) repeats the same shape with candle tier
resolution (`candle_minimax_h3_tier` `:1687`, `resolve_candle_minimax_h3_load` `:1738`). Prior to
admission inside the funnel, candle video also calls per-family legacy weights gates
(`vram_gate::minimax_h3_fit_error` `W/vram_gate.rs:2775`, `wan_video_fit_error` `:2723`, `mochi_fit_error`
`:2367`, `svd_fit_error` `:2443`, `scail2_video_fit_error` `:2858`, `video_weights_fit_error` `:2673`). [H]

### 3.8 Cross-lane synthesis

Steps that are the same intent in all 7 lanes (harness candidates):

1. Validate request against descriptor/capabilities (sampler/scheduler/guidance normalization,
   unsupported-axis refusal, LoRA allowlist).
2. Resolve weights (tier dir, installed tiers, certified/provenance check) — `resolve_weights_dir`,
   `resolve_qwen_edit_candle_base`, `resolve_kolors_ipadapter_base`, `resolve_wan_tier_dir_and_quant`,
   `resolve_minimax_h3_load`.
3. Ensure on-demand assets (47 `ensure_*`).
4. Resolve adapters (`resolve_adapters`, `resolve_wan_adapters`, `resolve_ltx_adapters`, …).
5. Probe live budget (backend-specific) and reclaimable credit.
6. Price candidates per tier (downtier) and per rung (selector).
7. Shape load (residency/load-shape declaration), seal spec.
8. Stamp decision into raw_settings + telemetry event.
9. Acquire generator (cache or uncached), with optional evict-to-reclaim.
10. Per-item drive (seeds / poses / prompts / batch) with preview, cancel, progress.
11. Optional cross-cutting passes: hires-fix two-pass, PiD decoder, prompt enhancement, likeness
    scoring, inline upscale, prefix-KV cache.
12. Consume events → write assets → metrics → terminal.

Truly lane-specific (hook candidates):

- **Conditioning assembly** (references, masks, control maps, clips, keyframes, audio refs).
- **Engine request field mapping** (bespoke request structs; family-only `VideoGenInput` fields).
- **Provider-specific candidate builders** where evidence shape differs (Krea turbo phase curves,
  Krea control ladder, SDXL endpoints, Kolors/Ideogram/SD3.5 receipts).
- **Work-list expansion** (image_count: poses, angle sets, edit grouping —
  `ImageRoute::image_count` `W/image_jobs/base.rs:610-648`).
- **Import/plan preparation** (H3 imported DiT, checkpoint plans, ComfyUI sources —
  `PreparedImageRoute` `W/image_jobs/base.rs:484`).
- **Family raw_settings keys** (additive to a generic core).

---

## 4. Admission

### 4.1 Inventory of admission paths

| # | Path | Entry | Backend | Budget probe / type | Evidence inputs | Output | Stamped where |
|---|---|---|---|---|---|---|---|
| A1 | Request geometry envelope | `C/admission_geometry.rs` `refuse_over_envelope` called `W/image_jobs.rs:644` | both | none | manifest `admissionGeometry` | `Option<String>` refusal | not stamped |
| A2 | MLX image tier probe (pre-load) | `mlx_fit_gate::preflight_request` `W/mlx_fit_gate.rs:5012`; called from `choose_mlx_request_tier` `W/image_jobs/base.rs:10427`, `:11127` | MLX | `generator_cache::mlx_tier_budget` (`W/generator_cache.rs:1906`) → `gen_core::MemoryBudget` | provider `MemoryProviderContract`, packaged calibration (`MlxCalibrationBinding` `:855`), memory anchors (`mlx_image_anchor_store` `:2247`), decode-quality policies, prefix-KV | `MlxRequestAdmission::{Admitted(MlxRequestEvaluation), Rejected(WorkerError)}` (`:3819`) | tier choice → LoadSpec |
| A3 | MLX image generation (post-load, cache-aware) | `mlx_fit_gate::evaluate_request` `W/mlx_fit_gate.rs:5091` / `_with_reference_spec` `:5124`; 35 call sites in 13 files (`krea_*`, `flux1_control`, `flux2`, `pulid`, `checkpoint_plan`, `mage_finetuned`, `stream.rs`, `base.rs`) | MLX | `live_request_budget` (`:4936`: `mlx_rs::memory` active/cache + sysctl total + cap + legacy reserve; Mage uses provider safe budget) | same as A2 + `MemoryCacheState`, loaded `OffloadPolicy`, `WarmPolicyProposal`, external committed bytes | `MlxRequestEvaluation{memory, context, decode_quality_decisions, process_limit_bytes, prefix_kv_cache}` (`:1370`) | `GenerationRequest.memory` + `MemoryRunContext`; `apply_request_gpu_memory_limit` |
| A4 | Cache cold-load residency | `mlx_fit_gate::apply_residency_policy` `W/mlx_fit_gate.rs:6389` (budget at `decide_residency_for_spec` `:6767`), called inside generator_cache loader (`W/generator_cache.rs:~1617`) on **all** backends | effectively MLX (sysctl probe has a non-mac twin, `:5873-5885`; likely no-op on candle) [M] | sysctl total + cap | `media().footprint()` component bytes, `engine_supports_sequential` | spec (maybe Sequential) or `InvalidPayload` | `ExecutionPolicy` in cache entry |
| A5 | Candle generic image | inline in `generate_candle_registry_stream_using` (`W/image_jobs/base.rs:13936-15300`) → `candle_memory_strategy::evaluate_shared_image*` (`W/candle_memory_strategy.rs:2034`, `:2083`, `:2137`) | candle | `gpu::nvidia_vram_budget_gb` (nvidia-smi, `W/gpu.rs:753`) → `vram_gate::VramBudget{free_gb,total_gb}` + reclaimable + ladder reserve | manifest `vramGbByTier`/`sequentialPeakGb`, packaged evidence, candle anchors (`candle_image_anchor` `:1934`), certification, overlay receipts | `CandleMemoryEvaluation{memory, context, predicted_peak_gb, basis, phase_peaks, admitted, prefix_kv_cache, warm_staged}` (`:691`) + `FitDecision` | raw_settings disclosure `:15310`; `note_loaded_peak` |
| A6 | Candle Krea turbo | `vram_gate::krea_turbo_fit_with_runtime` `W/vram_gate.rs:1265` (5 calls in base.rs) | candle | VramBudget | manifest `turboFit` phase curves, `KreaRuntimeEvidenceContext` (`:669`), anchors | `KreaTurboFit::{Resident,Fits,Reject,Unverified}` (`:546`) | raw_settings |
| A7 | Candle bespoke shared selector | `evaluate_shared_bespoke_image` `W/candle_memory_strategy.rs:2368` (kolors_ipadapter, kolors_control, pulid_candle, base.rs) and `evaluate_shared_image` from 9 bespoke files (flux2_control_candle, krea_multiphase, flux2_edit_candle, flux_ipadapter, flux1_control_candle, qwen_edit_candle, bernini, zimage_control, krea_edit_candle) | candle | VramBudget (each lane probes itself) | per-lane provider contract (sometimes imported directly from `runtime_cuda::providers::*`) | `CandleMemoryEvaluation` | hand-stamped raw_settings (12 files) |
| A8 | Candle conditioning-overlay floor | `conditioning_fit::decide/admit` `W/conditioning_fit.rs:272`, `:360`; via `admit_conditioning_overlay/paths` (22 call sites, 10 files) | candle | VramBudget | on-disk safetensors sums + `HEADROOM_GB` | `ConditioningFit::{NoWeightEvidence,NoBudget,TooBig,..}` → error | none |
| A9 | Candle bespoke base floor | `W/image_jobs/base_admission.rs` (717 lines), `admit_sdxl_bespoke_memory` `W/image_jobs/base.rs:12811` (sdxl_edit_candle, sdxl_ipadapter, detail) | candle | VramBudget | `vramGbByTier` or on-disk floor for imported/ComfyUI | Ok/refusal | — |
| A10 | Candle Krea control ladder | `krea_control_fit::fit_ladder_for_entry_with_runtime` `W/krea_control_fit.rs:913` → `select_strategy` `:721` | candle | VramBudget via `gate_with_evict_reclaim` | manifest `candle.control.peakGbByTier`, `decodeTileSaveGb`, `chunkAttnSaveGb` | `KreaControlFit::{Unknown,Fits,BestEffort,TooBig,Unverified}` (`:119`) | raw_settings [M] |
| A11 | Generic evict-to-reclaim wrapper | `gate_with_evict_reclaim` `W/image_jobs/base.rs:13349` | candle | VramBudget + `reclaimable_pool_gb` | any gate closure | `(D, budget)` + eviction | — |
| A12 | Video (both backends) | `video_admission::admit_video_generation[_with_veda]` `W/video_admission.rs:2456` (called inside the cache callback, `W/video_jobs/wan.rs:3207`) implementing core `VideoStrategySelector` (`C/video_request.rs:2395`) | both (`LANE` const `W/video_admission.rs:3682-3684`) | `live_video_runtime_state` → `gen_core::MemoryBudget` (MLX: `live_request_budget`; candle: **cudarc `mem_get_info`**, `:3623-3650`) | packaged video memory curves (`C/video_memory_curves`), decode/encode profiles, anchors, exceeded-bound store, Veda native rows | `VideoAdmissionOutcome{memory, context, refusal: Option<String>}` (`:2425`) | `VideoGenInput.memory/memory_context` |
| A13 | Candle video legacy weights gates | `vram_gate::{wan_video_fit_error, minimax_h3_fit_error, mochi_fit_error, svd_fit_error, scail2_video_fit_error, video_weights_fit_error}` (13 calls; `minimax_h3.rs`, `mochi.rs`, `candle.rs`) | candle | VramBudget | weights bytes + fixed activation constants (`W/fit_gate.rs` mochi/SVD constants) | `Option<WorkerError>` | — |
| A14 | Candle cold-load admission closure | `GeneratorColdLoadAdmission` `W/generator_cache.rs:1024` (SCAIL `W/video_jobs/candle.rs:2384`, `:2565`) | candle | fresh nvidia-smi `nvidia_vram_budget_gb_fresh_blocking` (`W/gpu.rs:775`) | resident reclaimable bytes | Ok/Err inside cache transaction | — |
| A15 | Audio YuE | `yue_admission::check` `W/audio_jobs.rs:1014`, `:1084` | candle audio | own probe [M] | lyric-section counting, per-stage pricing | refusal | — |
| A16 | Audio YuE2 | `W/yue2_admission.rs` (2,868 lines; also used by `video_jobs/minimax_h3.rs` and `wan.rs` for `host_available_bytes`, `W/video_jobs/wan.rs:~3204`) | candle audio | `host_available_bytes` | per-stage pricing, residency leases | chosen memory controls or refusal | — |
| A17 | Vector StarVector | `W/vector_admission.rs` | both | static device | safetensor sums (impossible-load floor) | refusal | — |
| A18 | FLUX.2-dev edit multi-reference safety | `W/image_jobs/flux2.rs` (provider-owned policy, referenced `W/mlx_fit_gate.rs` docs) | MLX | — | provider policy | refusal [M] | — |

So: ≥11 image-memory paths (A2-A11, A18) and ≥4 video paths (A4, A12, A13, A14), 2 audio, 1 vector,
1 geometry. [H for existence and entry lines; M for some internals]

### 4.2 The shared core that already exists

`W/memory_strategy.rs` (3,663 lines) is a backend-neutral selector:

- `RequestScope{resolved_route, backend, tier: MemoryNumericTier, mode, overlay, geometry: MemoryGeometry}` (`:383-395`).
- `Candidate{selection: MemorySelection, evidence: &MemoryEvidence, basis: CandidateBasis, unmodeled_activation_bytes}` (`:473-480`); `CandidateBasis::{Measured, EstimateFittedCurve, EstimateAnchorDerived{lane}, EstimateMeasuredEndpoint, EstimateFloor}` (`:406-412`).
- `Budget{available_gb, reclaimable_gb, total_gb, reserved_headroom_gb}` (`:572-577`), `ReserveCharge` (`:631`).
- `Selection::{Selected{selection, needed_gb, available_gb}, Reject{needed_gb, available_gb}, Unverified{reason}}` (`:653-666`).
- `select_strategy` / `_charging` / `_with_resident_credit` (`:777`, `:794`, `:807`).
- Callers: `mlx_fit_gate.rs:4614`, `:4655`, `:6296`, `:17061`; `video_admission.rs:2335`;
  `krea_control_fit.rs:721`; `vram_gate.rs:2091`; `candle_memory_strategy.rs:3080`, `:3118`,
  `:3247`, `:3474-3519`, `:7643`, `:7729`. [H]
- Margins: `W/ladder_margin_policy.rs` (406 lines; MLX vs candle recapture spreads).
- Shared reserve kinds: `W/fit_gate.rs` (`MemoryReserveKind::{UnifiedForeignResident, DedicatedVramAllocator}`, `FitDecision::{Unknown,Fits,Offload,TooBig}`, `resolve_offload`).

**Grading and first-fit are already unified; candidate construction, budget acquisition, tier
iteration, fallbacks, refusals and stamping are not.** That is the precise gap. [H]

### 4.3 Decision flow per path (condensed)

**A2+A3 MLX image (two-phase):**
1. Budget `mlx_tier_budget(engine_id)` (`W/generator_cache.rs:1906`).
2. For each installed tier (default first): provenance check → build spec via `prepare(dir, prefer_resident)` (`W/image_jobs/base.rs:10863`) → `preflight_request(spec, plan, inputs, budget)`.
3. `preflight_request` (`W/mlx_fit_gate.rs:5012`): reference workspace bytes → `media().memory_strategy_contract` → `select_request_with_budget_using_bundle(TierProbe, ...)` (`:3834`, ~1,100 lines):
   - `packaged_admission_route` / `evidence_admission_route` (`:1894`, `:1914`) → `AdmissionPath::{Evidence, Legacy}` with `LegacyAdmissionReason::{PackagedEmpty,NoBinding,NoRecord,OutOfEnvelope,FingerprintMismatch,IdentityMismatch,NoProvenance}` (`:1163-1180`).
   - Estimate bases (`collect_estimate_bases` `:2174`), anchor-derived peaks (`:2273`), synthesized ladder (`:3134`, `:3162`), floor (`estimate_floor_parameters` `:2793`).
   - `select_strategy_with_resident_credit` then a floored retry (`:4614`, `:4655`); lower verified geometry alternative (`:3549`, `:3633`) for refusals.
4. If Admitted and resident alternative exists, retry released-resident (`W/image_jobs/base.rs:11132-11150`); then `preflight_load_rejection` (`W/mlx_fit_gate.rs:4984`).
5. Tier chosen by `choose_mlx_request_tier` (`W/image_jobs/base.rs:10427`) → `MlxTierFit::{Fits, TooBig}` (`:10419`).
6. In cache callback: `evaluate_request(generator, plan, inputs, cache_state, load_policy, warm_policy, external_bytes)` (`W/mlx_fit_gate.rs:5091`) re-runs with `live_request_budget` (`:4936`) at `Generation` phase → `MlxRequestEvaluation`; `apply_request_gpu_memory_limit` sets soft MLX limit (`W/generator_cache.rs:873`).

**A5 candle generic:** budget raw → +reclaimable → reserve from raw (`W/image_jobs/base.rs:13936-13955`) →
downtier loop over `downtier_candidate_tiers` (`:5290`) with `candle_tier_fit` (`:11762`) /
`evaluate_shared_image_priced` / Krea turbo → `choose_downtier` (`:5212`) → final
`evaluate_shared_image*` for chosen tier (`:14589-14702`) → Krea branches / `fit_decision` +
`resolve_offload` (`:14926`) → sequential overflow check → `note_loaded_peak` (`:15211`) →
residency + load-shape (`:15288`, `:15293`) → disclosure (`:15310`).

**A7 candle bespoke:** probe → (sometimes `gate_with_evict_reclaim`) → `evaluate_shared_image` or
`evaluate_shared_bespoke_image` with a lane-built contract → refuse on `None` → stamp → (sometimes)
conditioning floor → `note_loaded_peak` → uncached load.

**A12 video:** in cache callback: tier from spec (`resolved_video_numeric_tier` `W/mlx_fit_gate.rs:5394`),
headroom (`spec_headroom_bytes` `:5484`), reference shape/count/overlay (`W/video_jobs/wan.rs:1657`,
`:1984`, `:2105`), `VideoAdmissionInputs` (`W/video_admission.rs:3517`), runtime only if packaged
evidence covers the request (`packaged_video_evidence_covers_request` `:2518`) → `live_video_runtime_state`
→ Veda host workspace → `admit_video_generation_with_veda` → core `VideoAdmission::{NotRouted,
Undecidable, Admitted{rung,geometry,..}, Refused{message,..}}` → `apply_video_admission_outcome`.

### 4.4 Input taxonomy (what every path consumes)

| Input class | MLX image | Candle image | Video (both) | Backend-specific? |
|---|---|---|---|---|
| Route identity (provider/engine id, resolved route) | `MlxRequestPlan.engine_id` | engine_id + model id | `route` + model id + family | no |
| Tier | `MemoryNumericTier` from spec/dir | tier key string (`"q4"`, `"nvfp4"`, `"int8-convrot"`) | `MemoryNumericTier` | key vocabulary differs (X) |
| Mode / overlay / reference count & shape | `MlxRequestInputs.mode/overlay/reference_count/reference_resources` (`W/mlx_fit_gate.rs:1276`) | `request_mode_value`, `overlay`, `MemoryGeometry.reference_count` | `mode`, `reference_shape`, `reference_count`, `overlay` | no |
| Geometry | width/height/count/native_batch | `MemoryGeometry{w,h,batch,frames=1,refs}` | w/h/frames/fps/decode_chunk | no |
| Request axes | `use_pid`, `has_phases`, `conditioning_windows`, `negative_branch`, `prefix_kv_cache_eligible`, adapter_count | `use_pid`, `has_phases`, `has_reference` | Veda/native-quant request, transformer variant, decoder | no |
| Load facts | spec (offload policy, load shape, components), footprint | spec + certification + receipts | spec + headroom | no |
| Provider contract | `media().memory_strategy_contract` | same, or direct `runtime_cuda::providers::*` call | loaded `generator`'s contract | no (should be registry-only) |
| Evidence | packaged calibrations, anchors, decode-quality fixtures | packaged evidence, manifest rows, anchors | packaged curves, profiles, anchors, exceeded-bound store | no (data) |
| Live budget | unified: sysctl total, MLX active/cache, cap env, OS reserve | VRAM: nvidia-smi free/total (image) or cudarc mem_get_info (video); cap env; reclaimable pool estimate; ladder reserve | via `live_video_runtime_state` | **yes** |
| Margins/reserve | `HEADROOM_GB`, `OS_APP_RESERVE_GB` (`W/mlx_fit_gate.rs:5292-5307`), MLX estimate margin | `DEDICATED_VRAM_ALLOCATOR_SLACK_GB`, `LADDER_RESERVE_MARGIN_GB`, `MEASURED_PRELOAD_RESIDENCY_*` (`W/vram_gate.rs:44-86`), candle recapture spread | backend reserve | **yes** |
| Cache state | `MemoryCacheState`, warm policy, external committed bytes, provider resident bytes | `Cold` at pre-load; warm-staged eval | cache_state, load_policy, provider_resident_bytes | no (concept), yes (measurement) |
| Failure posture | MLX OOM = SIGKILL → must refuse pre-load; Metal watchdog poisons process | CUDA OOM recoverable `Err` → looser margin; evict-to-reclaim | per lane | **yes** |

### 4.5 Outputs and stamping

Common output already exists in gen-core: `GenerationMemory` (into `GenerationRequest.memory`) and
`MemoryRunContext` (selection, geometry, mode, overlay, authority, cache_state, predicted peak,
evidence revision). Every path ends up producing (or failing to produce) that pair, plus one of:
tier choice, `OffloadPolicy`/`LoadShape`, soft process limit, reclaimable peak note, prefix-KV plan,
decode-quality decisions. The stamp is ad hoc per path (§1.6).

### 4.6 Proposed minimal typed contract

Designed so every path in §4.1 is an instance; fields map 1:1 onto what the code already computes.

```rust
/// Phase the decision is made in. MLX image does both; video and candle bespoke do one.
pub enum AdmissionPhase { Plan /* pre-load, may pick tier */, Run /* post-acquire, cache-aware */ }

pub struct AdmissionRequest<'a> {
    pub route: RouteKey<'a>,              // provider/engine id + resolved route + model id + family
    pub phase: AdmissionPhase,
    pub mode: MemoryMode, pub overlay: Option<&'a str>,
    pub geometry: MemoryGeometry,         // w, h, batch, frames, reference_count (+ fps for video)
    pub axes: RequestAxes,                // use_pid, has_phases, negative_branch, cond windows,
                                          // prefix_kv_eligible, adapter bytes, veda/native-quant
    pub tiers: TierCandidates<'a>,        // ordered installed tiers, default first; explicit pick flag
    pub load: &'a LoadSpec,               // already shaped by declarations for the tier under test
    pub contract: ContractSource<'a>,     // Registry(engine_id) | Loaded(&dyn Generator)
    pub evidence: EvidenceSet<'a>,        // packaged records, anchors, curves, manifest rows
    pub cache: CacheFacts,                // Cold | Warm{loaded_policy, provider_resident, external}
}

pub trait BackendBudget {                 // the ONLY backend-specific piece of grading
    fn probe(&self, route: &RouteKey) -> Result<MemoryBudget, BudgetUnavailable>;
    fn reclaimable(&self) -> u64;         // MLX cache bytes | candle pool high-water
    fn margins(&self) -> &MarginPolicy;   // reserve kind, estimate margin, recapture spread
    fn failure_posture(&self) -> FailurePosture; // RefuseIfUnpriced (MLX) | AdmitIfUnpriced (candle floors)
    fn evict_to_reclaim(&self) -> Option<&dyn Fn() -> WorkerResult<bool>>;
}

pub enum AdmissionDecision {
    Admit {
        tier: TierKey, offload: OffloadPolicy, load_shape: LoadShape,
        memory: Option<GenerationMemory>, context: Option<MemoryRunContext>,
        basis: CandidateBasis, needed_bytes: u64, available_bytes: u64, reserve_bytes: u64,
        process_limit_bytes: Option<u64>, prefix_kv: Option<PrefixKvCachePlan>,
        evict_first: bool, record_peak_bytes: Option<u64>,
        notes: Vec<AdmissionNote>,        // legacy-fallback reason, unverified, decode-quality, ...
    },
    Refuse(AdmissionRefusal),
    NotRouted,                            // video core's NotRouted / Undecidable collapse here
}

pub struct AdmissionRefusal {
    pub binding_phase: Option<BindingPhase>, // conditioning | denoise | decode | weights
    pub needed_bytes: u64, pub available_bytes: u64,
    pub tried_tiers: Vec<(TierKey, Option<u64>)>,
    pub suggestions: Vec<Suggestion>,        // smaller installed tier, lower geometry, fewer refs
    pub message: String,                     // house-style text, generated once
}

/// One stamping function replaces 12+ hand-written inserts.
pub fn stamp(decision: &AdmissionDecision, raw: &mut JsonObject, telemetry: &mut TelemetrySink);
```

Mapping check: A2/A3 = `Plan`+`Run`, `Registry`/`Loaded`; A5/A7/A10 = `Plan`, `Registry` (bespoke
must stop importing `runtime_cuda::providers::*` directly — 1 seen at
`W/image_jobs/kolors_ipadapter.rs:311`); A8/A9/A13/A17 = evidence set containing only a weights
floor, `CandidateBasis::EstimateFloor`; A4 = a `Plan` decision applied at cold load (keep as
`BackendBudget`-driven guard in cache); A12 = `Run`, `Loaded`; A14 = `evict_first` + fresh probe;
A15/A16 = audio stage pricing would need `BindingPhase` per stage (fits `binding_phase` + `notes`). [M]

**Backend-specific inputs (must stay behind `BackendBudget`)**: probe source (sysctl + `mlx_rs::memory`
vs nvidia-smi vs cudarc), caps (`SCENEWORKS_MLX_MEMORY_CAP_GB` `W/mlx_fit_gate.rs:5236`,
`SCENEWORKS_CUDA_VRAM_CAP_GB` `W/vram_gate.rs:36`), reserves and margins, reclaimable model
(MLX cache vs cudarc pool high-water `note_loaded_peak`), failure posture (SIGKILL vs `Err`),
Mage's provider-owned safe budget (`W/mlx_fit_gate.rs:4940-4943`), soft process limit (MLX only).

### 4.7 Divergences the unified pipeline must decide explicitly

1. Two probes on candle: image uses nvidia-smi (`W/gpu.rs:753`, 1 s cached); video and SCAIL cold
   load use cudarc / fresh nvidia-smi (`W/video_admission.rs:3623`, `W/gpu.rs:775`). [H]
2. Four budget structs: `VramBudget{free_gb,total_gb}` (`W/vram_gate.rs:125`), `MlxMemoryBudget`
   (`W/mlx_fit_gate.rs:5370`), `gen_core::MemoryBudget` (bytes), `memory_strategy::Budget` (GiB f64).
3. Tier vocabulary: `MemoryNumericTier` vs string tier keys vs `MemoryRouteTier` (`W/memory_route_registry.rs:29`).
4. Order of floor vs ladder differs (Krea control: floor first `krea_control_candle.rs:786`; Kolors IP: ladder first then floor `kolors_ipadapter.rs:445`).
5. MLX refuses pre-load on unpriced; candle floors admit on no evidence (`W/conditioning_fit.rs:276-281`).
6. Video admission runs inside the cache transaction (after load may have happened on a cold miss) while image runs before acquisition; warm-path semantics differ.

---

## 5. generator_cache and residency

### 5.1 Generator cache (`W/generator_cache.rs`, 5,231 lines)

- **Single-resident** `CacheThread<LoadIdentity, CachedGenerator>` on a dedicated OS thread
  (`:20-24`, `:44-50`; scaffolding `W/cache_thread.rs`). Idle eviction 300 s default,
  `SCENEWORKS_GENERATOR_CACHE_IDLE_SECONDS` (`:55-56`). `GENERATOR_EVICT_BEFORE_LOAD = false`
  (`:63`): cold miss clears the entry then loads; the refine cache uses `true` (`W/cache_thread.rs:12-30`).
- `CachedGenerator{generator, loaded_policy, external_committed_bytes, reclaimable_weight_bytes, provider_resident_bytes}` (`:26-43`).
- **Key** `LoadIdentity` (`:181-226`): engine id, resolved route, decode-geometry policies,
  decode-quality identity, weights source (with file fingerprint), quantize, precision, control,
  extra controls, ip-adapter, adapters, PiD pair, identity weights, text encoder, components,
  prepared file pins, imported-H3 closure, `released_resident`, SDXL CUDA offload policy.
  Residency/materialization policy is deliberately **excluded** (`ExecutionPolicy` `:233-237`), so
  a policy change does not reload, except the two listed exceptions.
- **Warm policy switching**: `propose_warm_policy` (`:525`) + `execution_planner` (`W/execution_planner.rs`, `WarmPolicyProposal`, `WarmPolicyOnce`).
- **Core transaction** `with_cached_generator_for_request_after_cold_admission_using_on_with_h3`
  (`:1585-1716`): evict resident Iris transform → derive key → cold-admission closure receives the
  resident's reclaimable bytes → loader runs `validate_prepared_file_pins` →
  `mlx_fit_gate::apply_residency_policy` → bind CUDA context lease (`bind_backend_load_context`
  `:118-141`) → capture committed bytes before/after load → `run(generator, cache_state,
  loaded_policy, warm_policy, external_bytes, provider_resident_bytes)`.
- Variants (each a thin wrapper): `with_cached_generator` `:1057`, `_for_request` (test) `:1104`,
  `_after_cold_admission` `:1140`, `_using_cold_admission` `:1223`, `_for_request_using_cold_admission`
  `:1293`, `_for_request_using` `:1426`, `_for_request_using_cache` `:1458`; uncached
  `with_uncached_generator` (candle) `:1718`. Eviction: `evict_cached_generator[_blocking]`
  (`:1814`, `:1826`), resident bytes `:1848`. [H]
- MLX memory limit plumbing lives here too: `apply_gpu_memory_limit` (`:848`),
  `apply_request_gpu_memory_limit` (`:873`, RAII guard `:858`), `sync_gpu_memory_limit` (`:909`),
  telemetry (`:928-962`), `mlx_tier_budget` (`:1906`).

### 5.2 Other resident caches sharing the device

- `transform_cache` (Iris restorer) — "third thin wrapper over cache_thread, sharing one resident
  engine with the generator cache" (`W/lib.rs:282-288`); evicted before candle generation probes
  (`W/image_jobs/base.rs:13929`) and before every cached load (`W/generator_cache.rs:1611`).
- `refine_model_cache` (prompt refine LLM, evict-before-load=true), `mlx_decode_cache`,
  `llm_kv_cache` (`W/lib.rs:315-342`).
- Uncached paths: 15 `start_gen_stream` lanes (all bespoke candle + some MLX), audio
  (`W/audio_jobs.rs`), training (evicts caches first, `W/training_jobs.rs:3677`).
- Candle pool accounting: because cudarc does not return freed memory, bespoke uncached lanes record
  `vram_gate::note_loaded_peak` (15 call sites) and use `gate_with_evict_reclaim` to decide whether
  evicting the cached generator would improve admission (`W/image_jobs/base.rs:13349`).

### 5.3 Resolved model hot cache (epic 19703)

This is a **disk** tier, not a residency cache: `RuntimeSourceGuard::finish_success` records a
`PendingPromotion` (`W/external_library_runtime.rs:214`, `:344`); the idle loop drains it
(`W/resolved_cache_promotion.rs` module docs `:1-30`; `W/lib.rs:2088-2121`); retention checkpoint
`spawn_retention_checkpoint` (`W/lib.rs:1753`). Local tier selection at job time:
`ResolvedCacheStore::valid_local_artifacts` (`W/external_library_runtime.rs:598-602`). Opt-in:
`ResolvedCachePolicy::default().enabled = false` (`C/resolved_cache.rs:95-102`). [H]

Implication: the harness's "plan" stage should consume the source guard's resolved closure (local
tier vs source library), not re-resolve weights per lane. Today lanes call `resolve_weights_dir` etc.
independently after the guard ran. [M]

---

## 6. cfg / backend split

### 6.1 Build configurations

| Config | Predicate | Linked runtime | What compiles |
|---|---|---|---|
| macOS | `target_os = "macos"` | `runtime-macos` (`Cargo.toml:123`), `mlx-rs` fork, `mlx-gen-ltx`, `ort` CoreML | MLX image routes, MLX video routes, `mlx_fit_gate` full, MLX telemetry |
| Candle | `all(not(macos), feature = "backend-candle")` | `runtime-cuda`, `candle-gen`, `candle-gen-ltx`, `ort` CUDA (`Cargo.toml:78`, `:131-151`) | `vram_gate`, `candle_memory_strategy`, `krea_control_fit`, `conditioning_fit` (`W/lib.rs:368-483`), all `*_candle.rs` image lanes (`W/image_jobs.rs:4133-4225`), candle video handlers |
| Neither | `not(macos)`, no feature | none (empty registry, `W/inference_runtime.rs:90-97`) | CPU utilities, stub generation, `not(any(..))` twins (99) |
| test overlay | `test` in many predicates | testkit | e.g. `generate_candle_stream_using` compiled on macOS tests (`W/image_jobs/base.rs:13525-13528`) |

Notable cross-overs: `mlx_fit_gate` is compiled everywhere with `allow(dead_code)` off-mac
(`W/lib.rs:386-389`) and the candle lane uses its `engine_supports_sequential`
(`W/image_jobs/base.rs:13991`); `generator_cache` compiles everywhere (`W/lib.rs:275-281`);
`engines.rs` is all-targets pure data (`W/engines.rs:1-16`).

### 6.2 Where the split forces duplication

- Route types and ladders: `ImageRoute`/`CandleImageRoute`, `VideoRoute`/`CandleVideoRoute` with
  different predicate **orders** (MLX: control lanes first, `W/image_jobs/base.rs:284-341`; candle:
  MOSS edit, InstantID, SDXL control first, `:1170-1210`). Any unified route table must encode
  order as data and prove equivalence per backend.
- Dispatch match duplicated per backend (`W/image_jobs.rs:866-1258` vs `:1272-1775`), with arms that
  call the *same* function on both (InstantId, SdxlControl, KreaMultiPhase, KreaTurboOnRaw,
  KreaImported, CheckpointPlan, MageFinetuned, SdxlImported).
- Generic lanes duplicated (`generate_stream_using` ~970 lines MLX vs
  `generate_candle_registry_stream_using` ~2,210 lines candle), sharing helpers (`render_on_surface`,
  `generate_one_with_hires_on_surface`, `mlx_raw_settings_for_bundle`, `consume_gen_events`).
- Admission modules split by backend (`mlx_fit_gate` vs `vram_gate`+`candle_memory_strategy`),
  sharing only `memory_strategy` and `fit_gate`.
- Job entrypoint twins for "neither" builds (§1.2).

### 6.3 A backend-neutral worker structure (sketch)

```
worker/
  job/           run loop, claim, run_utility_job (exists, keep), JobKind registry (table, not match)
  lane/          LaneHarness<L: Lane>: plan → admit → acquire → drive → write  (backend-neutral)
                 Lane trait (hooks): validate, expand_work, resolve_conditioning, build_request,
                                     raw_settings_extra, candidates (optional provider-specific)
  engine/        EngineHandle = Registry(dyn Generator) | Bespoke(Box<dyn BespokeModel>)
                 (bespoke candle providers wrapped once so they are cacheable and preview-wired)
  admission/     AdmissionRequest/Decision, Pipeline (candidate builders → memory_strategy::select)
  backend/       mod mlx; mod cuda; mod none   — implement BackendBudget, residency guard,
                 memory limit, device context; the ONLY cfg-gated modules
  routing/       data table: (backend, model predicate, mode predicate, priority) → LaneId
  sinks/         events (GenEvent + video loop merged), asset writers (image/video/audio), stamping
```

Target cfg surface: `backend/{mlx,cuda,none}` + linker selection in `inference_runtime`. Lanes that
exist on one backend only are expressed as routing-table rows absent on the other, not as cfg on
the lane code. [L — design proposal]

---

## 7. Tests and a migration safety net

### 7.1 How lanes are tested today

- Volume: `image_jobs/tests.rs` 419 tests; `video_jobs/tests.rs` 362; `mlx_fit_gate.rs` 172;
  `image_jobs/base.rs` 143; `video_admission/tests.rs` 106; `vram_gate.rs` 84; `memory_route_registry.rs` 59;
  `candle_memory_strategy.rs` 51; `generator_cache.rs` 50; `architecture_tests.rs` 9;
  `candle_preview_wiring_tests.rs` 6 (counts of `#[test]`/`#[tokio::test]`). [H]
- **Engine injection seams (`_using`)**: `generate_stream_using` (`W/image_jobs/base.rs:10613`),
  `generate_candle_stream_using` (`:13530`, also compiled for macOS tests), `generate_video_using`
  (`W/video_jobs/wan.rs:2806`), `generate_minimax_h3_using` (`W/video_jobs/minimax_h3.rs:1543`),
  `run_audio_generate_job_using` (`W/audio_jobs.rs:997`), `run_image_depth_job_using`
  (`W/depth_jobs.rs:93`). Tests pass a closure returning a fake `Box<dyn Generator>`
  (e.g. `W/image_jobs/supra2.rs:2441-2452`, `W/image_jobs/iris.rs:1014-1025`,
  `W/image_jobs/moss_dispatch_tests.rs:488`, `W/video_jobs/tests.rs:18292`).
- **Fake generators**: `StubGenerator`/`StreamingStubGenerator` (`W/generator_cache.rs` tests
  `~:3286-3358`), `VideoLifecycleGenerator` (`W/video_jobs/tests.rs:1072`), `ProbeGenerator`
  (`:4568`), `OutcomeGenerator` (`:21748`), `FixtureGenerator`/`ShapeGatedGenerator`
  (`W/video_admission/tests.rs:650`, `:699`), `StrictControlEngine` (`W/image_jobs/strict_control.rs:28`),
  `ContractOnlyGenerator`, `MockGenerator`, `PrefixCacheRecordingGenerator`, `HiresProbeGenerator`.
- **Fake API servers**: axum stubs for heartbeat/cancel/finishing (`W/tests/cancel_and_heartbeat.rs:413`,
  `:746`, `:1138`, `:1288`), HF stubs (`W/tests/support_stubs.rs`), overlay/tree stubs
  (`W/tests/lora_and_downloads.rs:757`, `:1249`).
- **Pure admission tests**: admission functions take explicit budgets (`preflight_request_with_contract`
  `W/mlx_fit_gate.rs:5046`; `evaluate_request_with_budget*` `:3652-3710`; `candle_budget_from_total_free`
  `W/video_admission.rs:3596`; synthetic CUDA budget for tests `:3671`).
- **Source-text drift guards**: `architecture_tests.rs` parses `contracts.rs`/`lib.rs`/`ARCHITECTURE.md`
  (`W/architecture_tests.rs:9-30`); `candle_preview_wiring_tests.rs` scans bespoke lanes for inert
  `preview: Default::default()`.
- **Weights-free capability facts**: `engine_capability_facts.rs` (2,308 lines),
  `inference_runtime::capability_snapshot_json` (`:55`), `memory_contract_surface_registry` (`:99`),
  `scripts/compare-engine-capability-facts.mjs`.
- **GPU/real-weight smokes**: ~40 `*_gpu_smoke.rs` / `*_mlx_smoke.rs` modules, `cfg(test)` +
  backend-gated (`W/lib.rs:686-942`); not normal CI (memory rule: real model execution is not
  ordinary PR CI).
- Integration tests dir: `crates/sceneworks-worker/tests/` (guards: `candle_kernels_patch_guard.rs`,
  `image_workspace_dep_guard.rs`, `nax_guard.rs`, `model_artifact_inventory.rs`, ...).

### 7.2 Safety net proposal for the migration

1. **Admission decision-diff harness** (the sc-19049 idea; I found no `sc-19049` reference in the
   repo, so this is a fresh proposal). Enumerate a corpus of `(model, mode, tier, geometry, refs,
   adapters, cache state, budget)` cells from `memory_route_registry::RULES` (`W/memory_route_registry.rs:562`)
   × `MODEL_TABLE` × a fixed budget grid (8/16/24/32/48/96/128 GB). For each cell, call the **old**
   path (via the existing pure seams: `preflight_request_with_contract`, `evaluate_shared_image*`,
   `krea_control_fit::fit_ladder`, `admit_video_generation_with_curves_and_profiles`
   `W/video_admission.rs:2772`) and the **new** pipeline; serialize both to a canonical
   `AdmissionDecision` JSON (tier, rung, offload, load shape, needed/available bytes rounded,
   basis, refusal class) and diff. Run on both backends' test builds (candle seams compile on macOS
   tests in places, `W/image_jobs/base.rs:13525`). Pure, weights-free, uses packaged evidence.
2. **Golden raw_settings / receipt snapshots**: for each lane, drive the `_using` seam with a
   recording fake generator against a stub API and capture `(GenerationRequest fields,
   LoadSpec/LoadIdentity, raw_settings, asset sidecar, progress event sequence)`. Assert shape (keys
   and value types), not machine-dependent numbers (memory rules: no machine goldens; assert shape
   for pinned corpora). Strip `createdAt`, ids, paths.
3. **Route-table equivalence**: generate a payload corpus (model ids from `MODEL_TABLE` + manifest
   modes) and assert old `resolve_image_route_with_imported_availability` /
   `resolve_candle_image_route_with_prepared_availability` / `resolve_video_route_with` /
   `resolve_candle_video_route` == new table lookup, per backend.
4. **Engine-request equivalence**: a `RecordingGenerator` that stores the `GenerationRequest` it
   receives; diff old lane vs harness lane per corpus cell (fields listed in §2.3).
5. **Existing lifecycle tests stay authoritative**: cancel/heartbeat/stall tests
   (`W/tests/cancel_and_heartbeat.rs`, video stall tests) run unchanged against the harness.

---

## 8. Design implications

1. **The job harness exists; the lane harness does not.** `run_utility_job_inner` already provides
   source guard, metrics, error mapping, shutdown, recovery for all kinds (`W/lib.rs:2484-2910`).
   The new layer is between dispatch and `consume_gen_events`.
2. **Video's funnel is the template.** `VideoGenInput` + `generate_video_using` already centralizes
   admission, cache, request build, event loop and stall policy for ~13 handlers. Image should get
   the same, but with a *typed* per-lane hook instead of a flat field bag (the bag is where family
   knobs accumulate: `VideoGenInput` has LTX-only/SVD-only/SeedVR2-only/H3-only fields).
3. **`CandleStrictControl` is the right hook shape** (required `conditioning_admission`, `load`,
   `generate_one`; shared driver does everything else). Generalize it to all bespoke lanes and to MLX.
   Adding `fn admission_candidates()` (optional, default = generic) and `fn raw_settings_extra()`
   covers Kolors/Qwen/Krea.
4. **Wrap bespoke candle providers once** behind an engine handle that takes the admitted
   `MemoryRunContext`; then they become cacheable (removing `note_loaded_peak` + `gate_with_evict_reclaim`
   pool bookkeeping, 15 call sites) and preview-wired by construction (removing the source-scan test).
   Better still, register them in the inference registry so `dyn Generator` covers them (cross-repo).
5. **One admission pipeline = candidate builders + existing `select_strategy` + one `BackendBudget`.**
   Selection is already shared; consolidate budget acquisition (4 budget structs, 3 probes), tier
   iteration (`choose_mlx_request_tier` vs `choose_downtier`), fallbacks (floors, legacy), refusal
   construction and stamping. Keep two phases (Plan pre-load; Run in-cache) because MLX needs warm/cold
   awareness and video already admits in-cache.
6. **Typed refusal** replaces `InvalidPayload(String)` for admission; carries binding phase,
   needed/available, tried tiers and suggestions, so UI and cross-cutting features (low-res first
   steps, multi-phase) can react programmatically.
7. **Cross-cutting features attach at harness stages, once**: hires-fix and low-res-first-steps
   (geometry/phase plan before admission; admission must price each phase — `has_phases` already an
   axis), PiD, prompt enhancement, likeness scoring, inline upscale, prefix-KV, memory rungs
   (admission), multi-phase (Krea multiphase is a lane today, `W/image_jobs/krea_multiphase.rs`).
   Today hires is wired into both generic lanes but not bespoke ones; low-res-first-steps would need
   the same N-way wiring without a harness.
8. **Routing as data**: the four ladders are priority-ordered predicate chains with different orders
   per backend; encode them as a table keyed by `(backend, predicate, priority)` and keep the
   `*_available` functions as predicates initially (62 of them), then fold weights-availability into
   the source guard's resolved closure.
9. **cfg collapses to a backend module** if lanes are backend-neutral and backend-only lanes are
   just missing routing rows; target is `backend/{mlx,cuda,none}` + `inference_runtime` linkage.
10. **Unify image and video event loops** (`consume_gen_events` vs video `select!` + stall watchdog):
    one consumer parameterized by an output sink (image writer per item; video encoder at end) and a
    watchdog policy (images currently have none).
11. **Single stamp function** for memory decisions, checkpoint facts and receipts; today 12 bespoke
    files hand-write `memoryStrategy` keys and the generic lanes use different mechanisms.
12. **Migration order with lowest risk**: (a) admission decision type + stamp + budget trait behind
    existing call sites (decision-diff harness green), (b) image lane harness for the two generic
    lanes, (c) port strict-control/bespoke candle lanes as hooks, (d) MLX bespoke lanes, (e) video
    onto the same harness, (f) route table, (g) cfg collapse.

---

## 9. Uncertain / not verified

- Exact internals of `select_request_with_budget_using_bundle` (`W/mlx_fit_gate.rs:3834-4935`) were
  summarized from function names and the few lines read; the legacy-vs-evidence branching order is [M].
- `apply_residency_policy` on candle: `probe_total_unified_memory_bytes` uses a
  `cfg(not(macos))` `sysctl_total_unified_memory_bytes` twin (`W/mlx_fit_gate.rs:5868-5885`) that
  I believe returns `None`, making the budget `None` and the policy a no-op unless
  `SCENEWORKS_MLX_MEMORY_CAP_GB` is set (then `resolve_budget` may still produce a budget). I did not
  read `resolve_budget`'s cap-only branch. [M]
- Where the MLX generic lane stamps the admitted strategy into raw_settings/asset facts (vs only
  `GenerationRequest.memory` + telemetry) was not traced to a specific line. [M]
- Audio (`yue_admission`, `yue2_admission`), vector, Iris restore/depth and SenseNova VQA lifecycles
  were mapped at entrypoint level only; their own budgets/probes not read. [M]
- Counts of `GenerationRequest {` sites include a few cfg(test)-only helpers inside production files
  (e.g. in `mlx_fit_gate.rs`); the ~45 production figure is approximate. [M]
- cfg counts are raw attribute counts; multi-line `#[cfg(any(` predicates were counted once per
  attribute, and the "any(macos, candle)" breakdown in §0 was not computed precisely. [M]
- Bespoke candle provider-contract import directly from `runtime_cuda::providers::*` was seen in
  Kolors IP (`W/image_jobs/kolors_ipadapter.rs:311`); I did not enumerate all such sites. [M]
- I did not verify whether every MLX bespoke lane (e.g. `krea_control.rs`, `flux2.rs`, `qwen.rs`)
  runs both admission phases (A2 and A3) or only A3. [L]
- sc-19049 "decision-diff harness" is not referenced anywhere in the repo; its original scope is
  unknown to me — §7.2(1) is my own proposal.
- Line numbers inside very long functions (e.g. "~:10760-10780") are approximate where marked `~`.
