# Full Codebase Review — SceneWorks — 2026-07-12

## Executive summary

- **Repository at a glance:** local-first, desktop-native AI image/video generation studio, well past the midpoint of a Python→Rust migration. Rust workspace (~200k LOC across `crates/sceneworks-core` ~48k, `crates/sceneworks-worker` ~109k, `apps/rust-api` ~33k, `crates/sceneworks-mcp` ~3.6k, `crates/sceneworks-image-quality` ~1.5k, `apps/desktop` Tauri ~4.7k), a React 18 + Vite frontend (~61k LOC JS/JSX in `apps/web`), a live pytest parity/e2e suite that spins up the Rust binaries, JSONC model manifests, and SQLite (`project.db`, `jobs.db`) for job/asset state. MLX on macOS (`mlx-gen`), Candle/CUDA on Windows/Linux. Reviewed at commit `29c811d9`.
- **Coverage:** the entire live tree was read across ten parallel subsystem passes plus a dedicated prior-findings verification pass — worker generation jobs (image/video/media), worker training/model/analysis jobs, worker infrastructure (dispatch/supervisor/downloads/GPU/caches/credentials), worker CV (person/pose/segment/track/upscale), `sceneworks-core`, `apps/rust-api`, the web shell (`App.jsx`/hooks/context/state), web screens + components, the platform layer (Tauri desktop, CI workflows, release/signing, docker, scripts, manifests), and the aux crates + pytest suite + docs. Test files were surveyed structurally rather than line-audited (noted per subsystem); `styles.css` (~10k lines, design-only), binary assets, `Cargo.lock`, and individual manifest rows were excluded except where sampled for schema/data errors.
- **Headline:** the codebase is in **good** shape and has **materially improved since the 2026-07-01 review**. All **nine** High findings from that review are fixed — verified against current code, not just "file changed" — via a coordinated sc-8804…sc-8877 remediation campaign whose fixes carry in-code F-number citations and dedicated regression tests; of 15 sampled Mediums, only one (`training_store` missing `busy_timeout`) remains open. There are **no Critical findings** and no remotely-exploitable defect under the default single-user loopback posture. The five High findings are all either new code that missed an established house pattern or a defect that surfaces only under specific input: a **path-traversal-on-write** via an unsanitized `request.model` in asset filenames (the one genuine new security gap, and the API does not validate the model id at enqueue), a **UTF-8 panic** in job-title truncation that can brick a project's queue on one non-ASCII prompt, two **heartbeat-silent blocking phases** that get falsely swept `interrupted`, a **complete absence of HTTP timeouts** on outbound clients, and a **stale-closure** in Dataset Doctor that can silently revert a user's unsaved edits. The dominant *maintainability* risk is unchanged and structural: **MLX↔Candle twin drift** (fixes and pins landing on one backend but not the other) plus large **god modules** (`ImageEditor.jsx` now 4.1k, `ImageStudio.jsx` 3.2k, `App.jsx` 2.4k, worker `image_jobs` `include!`-composed with a 6.3k-line `base.rs`, `jobs_store.rs`, two 13k+ test monoliths).
- **Counts:** Critical: 0 | High: 5 | Medium: 46 | Low: ~40 (grouped into 9 cluster findings) | Info: ~15 (grouped into 2 findings). Findings are numbered `F-001`… in the order laid out below (severity-ranked, then by subsystem); same-shape occurrences across sibling files are grouped under one F-number listing all locations.

---

## Critical findings

*None.* No exploitable-by-default, data-loss, or production-blocking defect was found. The path-traversal and blocking-silence findings below are High rather than Critical because the shipped default posture is a single-user loopback desktop app; the path-traversal (F-003) rises toward Critical as the token-gated LAN remote-access mode (epic 4484) widens the trust boundary.

---

## High findings

#### [F-001] Set HTTP timeouts on every outbound reqwest client
- **Category:** security
- **Severity:** High
- **Location:** `crates/sceneworks-worker/src/api_client.rs:14-21`, `crates/sceneworks-worker/src/downloads.rs:767-769,1172-1181`, `crates/sceneworks-worker/src/lib.rs:700`; `crates/sceneworks-mcp/src/api_client.rs:84-95`; `apps/desktop/src/cuda_provision.rs:515-517`
- **Finding:** No reqwest client in the worker, the MCP crate, or the desktop first-run provisioner sets `timeout`/`connect_timeout`/`read_timeout`; reqwest's default is *no* timeout. A server that accepts the TCP connection and never responds hangs the caller at `send().await` indefinitely. In the worker this precedes any progress/heartbeat interval (the `HEAD` in `lora_source_content_length`/`remote_content_length` runs before a cancel checkpoint exists), so the process never heartbeats and never claims again.
- **Impact:** A user-supplied (LAN-reachable in remote mode) `sourceUrl` that stalls pre-headers permanently wedges a utility worker; four such jobs wedge the whole default utility pool, and the supervisor never restarts a process that hasn't exited. In MCP the same gap defeats the tool's own "a stuck job can never hang the call forever" guarantee; on desktop first-run it combines with F-026 to freeze the setup screen unrecoverably.
- **Suggested fix:** Build each client with `connect_timeout` (~5–10s) and a chunk-level `read_timeout` (~60s so multi-GB streams still work); use a total `timeout` (~30s) on the non-streaming API/MCP clients.
- **Confidence:** High (mechanics verified across all three crates); Medium on real-world frequency for the loopback-only default.

#### [F-002] Wrap the two remaining heartbeat-silent blocking phases so the stale sweep can't false-interrupt them
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `crates/sceneworks-worker/src/control_training_jobs.rs:185-196`, `crates/sceneworks-worker/src/model_jobs.rs:1317-1398`
- **Finding:** Two multi-minute blocking phases are awaited as bare `spawn_blocking` with no heartbeat loop, no cancel poll, and (for prep) a discarded progress callback: the A2 control-dataset prep (per-image decode + DWPose/YOLO inference + PNG encode) and `run_model_convert_job`'s native converters (FLUX.2-dev/SD3/LTX/Anima prequant). The worker's `last_seen` is refreshed only by in-handler heartbeats (posting job progress does not refresh it — see `training_jobs.rs:1157-1161`), and the API stale-sweep marks silent jobs `interrupted` after ~90s. This is the exact class sc-8804/sc-8390 fixed everywhere else; these two sites were missed.
- **Impact:** A moderately sized control dataset or a large model conversion exceeds 90s, gets flipped to `interrupted` mid-work, and the terminal `Completed` post then 409s — a confusing "failed install" even though the output dir was promoted; user cancels are ignored for the whole phase.
- **Suggested fix:** Route both through the existing `run_blocking_with_heartbeat` (as `kps_jobs.rs:424-444` does); for the prep, thread a `gen_core::CancelFlag` into `prepare_control_dataset`'s per-item loop and stream its `on_progress` callback instead of discarding it. Convert may keep `cancel = None` (non-interruptible) — the keepalive is what's missing.
- **Confidence:** High (code + documented sweep semantics); confirm end-to-end by timing a >90s convert against the job row.

#### [F-003] Sanitize `request.model` before embedding it in asset file paths
- **Category:** security
- **Severity:** High
- **Location:** `crates/sceneworks-worker/src/image_jobs.rs:1246-1264,1461-1479`, `crates/sceneworks-worker/src/video_jobs.rs:805-810`; enqueue gap at `apps/rust-api/src/generation.rs:35` + `apps/rust-api/src/models.rs:1574-1609` (`resolve_model_manifest_entry` returns `{}` for unknown ids) and `apps/rust-api/src/lib.rs:2476-2510` (`validate_image_job` never checks the model id)
- **Finding:** `write_image_asset` builds `filename = format!("{}_{}_{}_{:04}.png", date, request.model, plan.slug, …)` and `media_path = project_path.join(media_rel)`; `VideoPlan::new` does the same for `.mp4`. `request.model` comes verbatim from the job payload (`image_request.rs:78`, no sanitization) while only the prompt is `slugify`'d. I verified the API side: the create handler does **not** validate `payload.model` against the catalog — `resolve_model_manifest_entry` returns `{}` for an unknown id and the job is created anyway, and the worker's stub lane deliberately serves unknown model ids — so a crafted id reaches the `format!` on every platform.
- **Impact:** A model id containing `../` or `\` traverses out of the project directory; `create_dir_all(parent)` + the atomic rename then write an attacker-named PNG/MP4 anywhere the worker user can write. In remote-access mode any authorized job-submitter gets an arbitrary-write primitive (attacker-controlled location, constrained content); locally a corrupted preset silently writes outside the project and breaks the sidecar/indexing contract. This is an outlier — every other payload-derived path is confined (`safe_project_path`, `normalize_app_managed_lora_path`).
- **Suggested fix:** Run `request.model` through `slugify` (or a `safe_weight_filename`-style single-component validator) when building the filename in `write_image_asset`, `write_upscaled_asset`, and `VideoPlan::new`; independently, reject unknown model ids in `validate_image_job`/`validate_video_job` at enqueue.
- **Confidence:** High (worker mechanics + missing API validation both verified).

#### [F-004] Fix the UTF-8 byte-slice panic in job-title prompt truncation
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `crates/sceneworks-core/src/jobs_store.rs:2173-2184`
- **Finding:** `truncate_prompt` does `prompt[..max]` after a byte-length check (`prompt.len() <= max`), but `max` (80/60) is a byte index; if it lands mid-way through a multi-byte UTF-8 character the slice panics ("byte index is not a char boundary"). Prompts are arbitrary user text (CJK, emoji, accents), and this runs inside `derive_job_title`, which `row_to_job` calls on **every** job row read.
- **Impact:** A single job with a >80-byte non-ASCII prompt (e.g. 27 three-byte CJK chars) makes every `list_jobs`, `get_job`, `queue_summary`, claim, and sweep that touches the row panic — the queue/API surface for that project is effectively bricked until the row is deleted out-of-band.
- **Suggested fix:** Truncate on a char boundary (`prompt.chars().take(N).collect()`, or a `char_indices` cut ≤ max), keeping the word-boundary trim afterwards; add a regression test with a long CJK prompt.
- **Confidence:** High.

#### [F-005] Stop Dataset Doctor fix-actions from capturing stale editor state and reverting unsaved edits
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `apps/web/src/screens/TrainingStudio.jsx:1404-1417`
- **Finding:** The `datasetDoctor` bundle is memoized on `[readiness, readinessLoading]` with a comment asserting its six fix-action handlers are "stable hoisted fn decls." Function declarations inside a component are recreated every render, so the memo freezes the handler instances from the render at which readiness last resolved. Those handlers all call `persistDataset()`, which closes over `selectedAssetIds`, `captionDraftById`, `draftName`, and `associatedCharacterId`; readiness only refetches on dataset-version change, so edits made since the last save are invisible to the captured handlers.
- **Impact:** A user who adds images or edits captions, then clicks any one-tap Doctor fix ("Remove duplicates", "Upscale low-res", "Smart crop", "Strip metadata", "Analyze"), triggers a save built from the stale pre-edit snapshot; `setActiveDataset(saved)` then resets the UI to that stale membership — silently discarding the newer unsaved edits.
- **Suggested fix:** Drop the memo (the bundle is cheap) or route the handlers through a ref (`handlersRef.current = {…}` each render; memo closes over the ref), matching the ref-bridge pattern ImageEditor already uses for exactly this problem.
- **Confidence:** High on the mechanism; Medium on frequency (confirm by adding images, clicking a Doctor action, and diffing the persisted item set).

---

## Medium findings

### Worker — generation, training, CV, infrastructure

#### [F-006] Confine payload-supplied `controlWeights.path` in both Krea control lanes
- **Category:** security
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/image_jobs/krea_control.rs:144-157`, `crates/sceneworks-worker/src/image_jobs/krea_control_candle.rs:160-174`
- **Finding:** Both Krea strict-pose lanes accept a payload-supplied absolute `advanced.controlWeights.path` and load it directly (`PathBuf::from(path); if p.is_file() { return Ok(p) }`) with no confinement, while every sibling payload-path input is confined (LoRA via `normalize_app_managed_lora_path`, ComfyUI components via `normalize_app_managed_model_path`).
- **Impact:** A crafted job can point the control-overlay loader at any file on disk (arbitrary-file-as-weights; error messages probe file existence), breaking the WKA-002 "payload can never point outside a declared root" invariant.
- **Suggested fix:** Route the value through `normalize_app_managed_model_path` in both twins (the B4/sc-10165 registry already resolves legitimate values under app-managed roots).
- **Confidence:** High on the code; severity scoped by the remote-access boundary.

#### [F-007] Complete the sc-9879 HF revision-pinning rollout on the MLX lanes
- **Category:** security
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/image_jobs/pulid.rs:218-225`, `krea_control.rs:179`, `video_jobs.rs:3143-3164,4980-5024`, `image_jobs/kolors_ipadapter.rs:20`
- **Finding:** The candle lanes pin weight repos to exact commits, but their MLX twins fetch at mutable `main`: PuLID (`guozinan/PuLID`, `SceneWorks/pulid-flux-mlx`), the Krea control overlay (`SceneWorks/krea2-pose-controlnet-beta`), and `lightx2v/Wan2.2-Lightning` (both video lanes); `kolors_ipadapter` pins to `refs/pr/4` (a mutable PR ref), not a commit.
- **Impact:** An upstream re-push (or a compromised HF token on the first-party repos) can silently swap the exact weights the pinned candle twin protects — defense-in-depth is asymmetric between backends.
- **Suggested fix:** Add the two PuLID repos to `instantid_revision` matching the candle shas, pin the MLX Krea overlay to `KREA_CONTROL_OVERLAY_REVISION`'s commit, add a shared `WAN_LIGHTNING_REVISION` const, and replace `refs/pr/4` with its commit sha.
- **Confidence:** High.

#### [F-008] Reject strict-pose jobs when the candle control base snapshot is absent instead of rendering plain txt2img
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/image_jobs/base.rs:334-355` (PoseReject exclusion list) with `zimage_control.rs:88-124`, `kolors_control.rs:80-85`, `krea_control_candle.rs:63-107`
- **Finding:** `resolve_candle_image_route`'s `PoseReject` arm excludes the wired pose families on the assumption their control lanes claim the job, but each control lane is weight-gated on a *dense upstream* snapshot that is a different repo from the packed turnkey the txt2img lane resolves. With only the turnkey installed (the normal catalog install), a `z_image_turbo`+poses job fails the control gate, skips PoseReject (excluded), lands in `CandleTxt2Img`, and renders plain txt2img — the poses silently dropped (the sc-5968 failure mode the reject arm exists to prevent). The wired-family list is also hand-duplicated in three places.
- **Impact:** Users with a standard turnkey-only install get unconditioned images labeled as pose generations, with no error.
- **Suggested fix:** For the wired families, when `pose_entries` is non-empty but the control lane's base gate fails, return a "control base snapshot not installed" error instead of falling through; derive all three family lists from one shared const.
- **Confidence:** Medium — in-worker fall-through certain; confirm end-to-end reachability against `zimage_control_candle_eligible`/`image_job_candle_pose_reject` in jobs_store.

#### [F-009] Compute the candle plan `expectedCount` from the strict-pose set, not `request.count`
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/image_jobs.rs:412-424` vs `crates/sceneworks-worker/src/image_jobs/candle_strict_control.rs:122-124,215-231`
- **Finding:** On macOS the plan bakes the route's real total (`route.image_count(...)` → pose count). The candle plan only special-cases InstantID and otherwise reports `request.count`, while every candle strict-pose lane actually produces `pose_entries().len()` images (the `total` handed to `consume_gen_events`).
- **Impact:** For a candle pose job with pose-count ≠ requested count (default 4), the gallery streams against the wrong expected total — stuck placeholders or a never-completing set indicator, diverging from identical jobs on macOS.
- **Suggested fix:** Mirror the macOS shape: return `pose_entries(request).len()` when a strict-pose candle route is taken (or add an `image_count` method on `CandleImageRoute` and compute the plan after route resolution).
- **Confidence:** High on the mismatch; user-visible severity depends on how `persist_reported_assets` treats `expectedCount` vs actual writes.

#### [F-010] Validate the Real-ESRGAN ONNX output shape before indexing
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/upscale_jobs.rs:326-341`
- **Finding:** `Upscaler::upscale` reads `oshape[2]`/`oshape[3]` with no rank check and indexes `odata[…]` with no length/scale validation. Every sibling decode path was hardened under sc-8904/F-102, but the upscaler was missed — and the generic `SCENEWORKS_REALESRGAN_ONNX` pin serves both x2 and x4, so pinning an x2 export and running a 4x job makes the output dims half the assumed size → out-of-bounds slice → panic inside `spawn_blocking` (job dies as a join error, `UPSCALERS` lock poisoned).
- **Impact:** A wrong env pin or manifest override becomes an unexplained panic/poison-recovery instead of a typed error naming the mismatch.
- **Suggested fix:** After `try_extract_tensor`, check `oshape.len() == 4`, `och == ch*factor && ocw == cw*factor`, and `odata.len() >= 3*och*ocw`; return `WorkerError::Engine` with expected-vs-actual dims (mirror `person_jobs::decode`).
- **Confidence:** High.

#### [F-011] Make the person-detect/SAM weight env pins fail loudly when set-but-missing
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/person_jobs.rs:970-976`, `person_segment.rs:84-90`, `person_segment_sam3_common.rs:74-84`
- **Finding:** `SCENEWORKS_PERSON_DETECTOR_WEIGHTS`, `SCENEWORKS_SAM2_WEIGHTS`, and `SCENEWORKS_SAM3_WEIGHTS` silently fall through to cache/download when the pinned path doesn't exist. sc-8911 already classified this as a defect and fixed it for DWPose, Real-ESRGAN, and SeedVR2 (`resolve_env_file_pin`), but these three resolvers kept the old behavior.
- **Impact:** An operator typo silently loads different weights than intended (downloaded snapshot instead of the local pin) — the exact "masked typo" mode sc-8911 was raised to eliminate, most confusing during parity/debug runs.
- **Suggested fix:** Route all three through the existing `resolve_env_file_pin`-style helper (set+missing → `InvalidPayload` naming the var; unset → fall through).
- **Confidence:** High.

#### [F-012] Bound the thread-local smart-select model cache to one blocking thread
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/person_segment_sam3.rs:92-113` (used 386-416, 502-523)
- **Finding:** `BOX_SEGMENTER`/`POINT_TRACKER` cache a quantized SAM3 instance per blocking thread (forced by `Rc<Backbone>` being `!Send`). When other blocking jobs occupy the most-recent thread, successive smart-select clicks land on different threads, each building and retaining its own ~0.9 GB Q8 model until that thread's 10s idle reap.
- **Impact:** Under interleaved load a burst of interactive clicks holds several GB of duplicate model memory on a unified-memory Mac and re-pays the seconds-long build+quantize the cache (sc-8846) was meant to avoid.
- **Suggested fix:** Route smart-select jobs through a dedicated single-thread executor (a `std::thread`+channel owned by a `OnceLock`) so the `!Send` model lives on exactly one thread, or expose a `Send` handle from mlx-gen-sam3.
- **Confidence:** Medium — confirm by logging `thread::current().id()` across clicks while a video job runs.

#### [F-013] Don't let one spawn/try_wait error tear down the whole worker supervisor
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/supervisor.rs:143,197-204` (loop at 82-92)
- **Finding:** `restart_exited_children_at` propagates `try_wait()?` and `spawner(...)?` errors; `supervise_children` then `?`s them out of its loop and returns. One transient respawn failure (AV-locked exe during update, momentary resource exhaustion) terminates the supervisor while healthy siblings keep running — and with the parent-death cleanup being desktop-side only, orphans the still-running children (the historical orphan bug class, re-entered through a new door).
- **Impact:** Supervisor exits, orphaning per-GPU/CPU children and stopping all further restarts.
- **Suggested fix:** In the restart pass, treat a spawn failure like a crash (log, re-arm backoff, continue); log-and-skip on a `try_wait` error rather than propagating.
- **Confidence:** High.

#### [F-014] Give Windows supervised children a graceful shutdown signal
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/supervisor.rs:208-256`
- **Finding:** `terminate_child` sends SIGTERM on unix but falls straight to `start_kill()` (TerminateProcess) on Windows, so the subsequent grace-window wait is moot there — children die before the deadline loop starts. The entire sc-8845 graceful-cancel machinery in `lib.rs` (trip cancel flag → wind-down → terminal `Canceled`) is unreachable for supervised children on Windows.
- **Impact:** On the Windows candle box any supervisor-driven stop kills a child mid-GPU-write; the in-flight job dangles until the 90s stale sweep marks it the generic `interrupted` — the outcome sc-8845 was built to prevent, silently regressed per-platform.
- **Suggested fix:** Add a Windows graceful signal (named event / stdin-close / control pipe that `shutdown_signal()` also selects on) and only `start_kill()` after the grace deadline, mirroring the unix ladder.
- **Confidence:** High for in-crate behavior; Medium on net user impact (desktop-managed stops may differ).

#### [F-015] Clean up the leaked control-dataset work directory
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/control_training_jobs.rs:162-167`
- **Finding:** The rendered control dataset is written to `data/cache/control-datasets/<job.id>` with a comment claiming it is "cleaned with the cache," but no sweeper or `remove_dir_all` for that tree exists anywhere in the repo — it is never deleted on success, failure, or cancel.
- **Impact:** Every `control_training` run permanently leaks a full letterboxed copy of the training corpus (a `.target.png`+`.pose.png` pair per image — easily GBs), silently growing the app data dir.
- **Suggested fix:** `remove_dir_all(work_dir)` after `run_training_execution` (both arms, via a scope guard), or add a startup sweep of `cache/control-datasets/*` for inactive jobs.
- **Confidence:** High.

#### [F-016] Coalesce caption per-step progress posts (the sc-8840 fix never reached the caption twin)
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/caption_jobs.rs:201-217,255-271`
- **Finding:** Every `Progress::Step` from JoyCaption decode is forwarded on a bounded(64) channel, each firing a full `update_job` POST awaited inline — byte-for-byte the pattern sc-8840 (F-038) replaced in `prompt_refine_jobs.rs` with a latest-wins `watch` channel + 250 ms coalescing, precisely because per-token POSTs back-pressure GPU decode on API latency.
- **Impact:** For a dataset caption run (up to 256 tokens/image × N images) the worker emits hundreds of sequential POSTs per image and the `blocking_send` throttles token decode to API round-trip latency — worse over the epic-4484 LAN mode. Refine is fixed; caption is not.
- **Suggested fix:** Port the sc-8840 watch-channel + interval coalescing from `prompt_refine_jobs.rs` (keep the `Captioned` per-item posts).
- **Confidence:** Medium — depends on JoyCaption emitting `Step` per token (as its LLM siblings do).

#### [F-017] Defer the terminal `Canceled` until the blocking task actually stops (caption + refine)
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/caption_jobs.rs:311-315`, `crates/sceneworks-worker/src/prompt_refine_jobs.rs:1145-1149`
- **Finding:** Both tick arms use `check_cancel`, which posts the terminal `Canceled` (freeing the worker row) at acknowledgement time while the blocking GPU task is still running, then trip the engine flag and keep draining. `analysis_jobs_common.rs:146-151` and the training path replaced this with `cancel_requested_peek` + a deferred `mark_job_canceled` (sc-8917, F-115) for exactly this reason.
- **Impact:** The scheduler sees a "free" worker and can hand it the next job while the captioner/refiner is still on the GPU — transient double-residency/memory pressure. Bounded (engines poll per token/item) but non-zero, and an unreconciled divergence between siblings sharing this contract.
- **Suggested fix:** Switch both arms to `cancel_requested_peek` + a `canceled` latch and post `mark_job_canceled` only after `guard.into_handle().await` returns, mirroring `run_batched_analysis_job`.
- **Confidence:** High on the divergence; Medium on incident likelihood.

#### [F-018] Hoist the remaining verbatim-duplicated pure logic out of the SAM3 backend twins
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/person_segment_sam3.rs:317-333,618-656` vs `person_segment_sam3_candle.rs:247-263,357-394`; cancel constants at `person_segment_sam3_candle.rs:53-71` vs `person_segment.rs:36-51`
- **Finding:** After the sc-8847 extraction to `person_segment_sam3_common`, three tensor-independent blocks remain duplicated line-for-line across the MLX and candle twins: `frame_mask_for_object` (only the frame-output type differs — the `Sam3FrameOutput` trait already abstracts it), the left-to-right paint-order computation, and the per-frame `obj_ids×masks → mask_to_frame` mapping, plus the cancel message/progress alias/coarse-cancel trio.
- **Impact:** Exactly the drift class the common module was created to kill: a tie-break tweak landing in one twin silently changes SCAIL-2 palette assignment or replace-person mask selection on one platform only, with no compile error.
- **Suggested fix:** Add generic `frame_mask_for_object<F: Sam3FrameOutput>`, `paint_order<F>`, and `per_frame_masks<F>` to `person_segment_sam3_common` and move the cancel constants there.
- **Confidence:** High.

#### [F-019] Deduplicate the generator/refine cache-thread scaffolding
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/refine_model_cache.rs:41-255` vs `generator_cache.rs:20-432`
- **Finding:** `refine_model_cache` re-implements the generator cache nearly verbatim (the `Fingerprint` enum with identical doc comments, `panic_message`, the dedicated-thread worker loop, idle-timeout event enum, evict-on-panic, mtime fingerprinting, the oneshot-reply seam). The pair has already drifted: the refine cache evicts *before* loading a miss to bound peak memory, the generator cache does not (`refine_model_cache.rs:136-147` vs `generator_cache.rs:176-191`).
- **Impact:** ~250 duplicated lines whose invariants must be fixed twice; the evict-before-load divergence is undocumented.
- **Suggested fix:** Extract a generic single-resident `CacheThread<K, M>` parameterized over key/model/loader; keep two thin wrappers; document or unify the evict-before-load difference.
- **Confidence:** High.

#### [F-020] Convert the `include!`-composed `image_jobs` module into real submodules
- **Category:** readability
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/image_jobs.rs:1729-1914` (`include!` ladder), `image_jobs/base.rs` (6,273 lines)
- **Finding:** All ~30 image-job lane files are `include!`d into one module namespace with load-bearing include *ordering* and cross-file unqualified references (e.g. `zimage.rs` reaching helpers in `base.rs`). `base.rs` alone mixes geometry helpers, routing enums, five tier-resolver families, LoRA resolution, the generation harness, the VRAM gate, and metrics.
- **Impact:** Name-collision risk grows with every lane; per-file `cargo fmt`, rust-analyzer scoping, and blame all degrade. This is the substrate on which the twin-drift findings (F-007/F-008/F-009) keep occurring.
- **Suggested fix:** Incrementally convert to real `mod`s with explicit `use super::…`, starting with the leaf candle lanes (already `*_CANDLE_*`-prefixed) and splitting base.rs's tier-resolver block first.
- **Confidence:** High (cost assessment); migration priority is a judgment call.

#### [F-021] Move the 1,700-line pin-bump changelog out of the worker Cargo.toml
- **Category:** readability
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/Cargo.toml:1-1922`
- **Finding:** The manifest is 1,922 lines, ~1,710 of them accumulated per-bump narrative comments; the actual dependency list (~120 lines) is buried within.
- **Impact:** The file is effectively undiffable (~86K tokens), grows with each pin bump, and hides what actually changed in a dependency edit; comment-block merge conflicts are likely.
- **Suggested fix:** Keep only the current-pin rationale (last 1–2 entries) and move history to `PINS.md` or rely on git log (each entry is already a commit message).
- **Confidence:** High.

#### [F-022] Regenerate the drifted worker ARCHITECTURE.md dispatch matrix
- **Category:** readability
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/ARCHITECTURE.md:68-111`
- **Finding:** Every "Proof (file:line)" reference is stale (all rows shifted ~440 lines); the matrix omits dispatchable job kinds that exist in `run_utility_job` (`control_training`, `lora_download`, `dataset_analysis`, `dataset_face_analysis`, `face_likeness_compare`, `image_segment`, `dataset_upscale`); and the `video_upscale` row still says "Mac-only," contradicted by `lib.rs:1258-1267` (macOS **or** candle) and `gpu.rs:105-113` (candle advertises `VideoUpscale`, sc-5928).
- **Impact:** The document's "provably complete matrix" promise is broken; a developer trusting the Mac-only claim mis-routes the candle SeedVR2 lane.
- **Suggested fix:** Regenerate the matrix (add the seven rows, fix video_upscale) and replace brittle line numbers with function-name anchors.
- **Confidence:** High.

### Core (`sceneworks-core`)

#### [F-023] Bring builtin-manifest seeding up to the house atomic-write standard
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/builtin_manifests.rs:65-84`
- **Finding:** `seed_builtin_manifests` stages to a *deterministic* temp name (`{name}.tmp`) and renames with no fsync — the two defects `store_util::atomic_write` was hardened against (sc-1633 colliding writers, sc-8949 rename-before-durable). The doc comment claims a crash "can't leave a truncated manifest," which the implementation does not guarantee.
- **Impact:** A power loss right after rename can leave a zero-length `builtin.*.jsonc`; in `SeedMode::IfMissing` the truncated file *exists*, so later seedings skip it and the broken catalog persists. Concurrent seeding by two processes can collide on the shared temp name and fail startup.
- **Suggested fix:** Reuse `store_util::atomic_write` (unique temp suffix, `sync_all` before rename, best-effort parent-dir fsync).
- **Confidence:** High.

#### [F-024] Skip one corrupt registry entry instead of failing the whole project listing
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/project_store.rs:273-295` (with `read_project_summary` at 3475-3489)
- **Finding:** `list_projects` propagates `read_project_summary(...)?` per entry, so a single registered project whose `project.json` is missing/unreadable/short-a-field fails the entire listing instead of skipping that entry.
- **Impact:** One damaged project (interrupted copy, cloud-sync conflict file) makes every healthy project unreachable through the switcher until repaired by hand.
- **Suggested fix:** Match the store's fail-open convention elsewhere (`let Ok(x) = … else { continue }`, as `list_characters`/`list_assets`): skip and log a structured warning naming the path.
- **Confidence:** Medium — confirm the rust-api handler doesn't already catch this per-entry.

#### [F-025] Reduce SQLite connection churn on the hot jobs path
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/jobs_store.rs:1848-1884` (pattern repeated per method; also `project_store.rs:3519-3524`, `character_store.rs:1024-1030`)
- **Finding:** Every store operation opens a brand-new SQLite connection: `create_dir_all` on the DB parent, `Connection::open`, `busy_timeout`, a `journal_mode=wal` write-locking pragma round-trip, and `foreign_keys`. Claims, heartbeats, and progress updates each pay this repeatedly; the claim path is already the site of observed `claim_lock_contention`.
- **Impact:** Per-request FS syscalls plus repeated `journal_mode` pragma churn add fixed latency under the process-wide write mutex, amplifying contention under multi-worker load.
- **Suggested fix:** Keep one long-lived write connection (or a tiny pool) per store guarded by the existing mutex; the mutex already serializes writers so semantics are preserved.
- **Confidence:** Medium — correct today; cost matters under load, which the `claim_lock_contention` events suggest is real.

#### [F-026] Give `training_store` project.db connections the same `busy_timeout` as every other opener (carried open from 2026-07-01 F-013)
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/training_store.rs:1269,1374,1409,1483,1509`
- **Finding:** Five sites open project.db via raw `Connection::open(project_path.join("project.db"))` with zero `busy_timeout` matches in the file, and `resolve_asset_source` (1260) skips migrations. This is the **only** sampled finding from the prior review still open — every other project.db opener sets a busy_timeout.
- **Impact:** Under concurrent access (worker + API both touching project.db) these connections get an immediate `SQLITE_BUSY` instead of waiting, surfacing as spurious training-store errors — the `database is locked` class the timeout exists to absorb.
- **Suggested fix:** Route all five through the shared connection helper that sets `busy_timeout` (and runs migrations); add a test asserting the pragma is set.
- **Confidence:** High.

#### [F-027] Continue decomposing the `jobs_store` god module and its order-sensitive dispatch chain
- **Category:** readability
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/jobs_store.rs` (6,463 lines; production 1-3306), `jobs_store/routing/candle.rs:40-259`
- **Finding:** The sc-8816 routing split helped, but `jobs_store.rs` still mixes schema/migrations, job lifecycle, worker registry, four sweep families, claim/dispatch scoring, metrics, and title derivation, plus ~3,100 lines of in-file tests. `image_job_is_candle_eligible` is a 220-line sequential if-chain where **branch order is load-bearing** (bespoke lanes must divert before the generic txt2img gate) with no structural guard.
- **Impact:** New-family wiring requires editing an exact position in the chain; a misordered branch silently changes lane priority (a class of bug several sc-comments describe post-hoc).
- **Suggested fix:** Move in-file tests to `jobs_store/tests.rs`; split metrics and sweep methods into submodules; consider a `(guard_fn, lane)` table so candle dispatch order is data, not code position.
- **Confidence:** High.

### rust-api

#### [F-028] Validate ownership/terminal status before running job-completion side-effects
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/rust-api/src/jobs.rs:385-425`
- **Finding:** `update_job_progress` runs `register_completed_training_lora` (386), `register_completed_control_overlay` (392), and `persist_reported_assets` (399-401) whenever the *reported* status is `Completed` — before `store.update_job_progress` enforces `TerminalJobImmutable`/`NotJobOwner` (402-425). A worker report that loses the race with cancel/sweep/reclaim still gets its LoRA/overlay registered and its assets persisted, and only then receives the 409.
- **Impact:** A canceled `lora_train`/`ControlTraining` job can register a ghost catalog entry; a canceled generation can persist assets the user explicitly canceled. Any authenticated caller can also trigger these writes for a job it doesn't own (ownership check fires too late).
- **Suggested fix:** Read the job snapshot once, reject non-owned/terminal jobs first (or move the three hooks after a successful `store.update_job_progress`).
- **Confidence:** High.

#### [F-029] Collapse the drifting `prompt_batches` / `recipe_presets` structural clone
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/rust-api/src/prompt_batches.rs:21-457` vs `recipe_presets.rs:3-695`
- **Finding:** The entire CRUD surface (list/get/create/update/delete-as-archive/duplicate, write-scope, write-manifest-path, find/location helpers, scope sort) is copy-adapted between the two, ~400 mirrored lines differing only in field/scope names, and already drifted (each strips a different set of runtime fields on update; create-path strip is missing in both — see F-056).
- **Impact:** Every fix in one module must be re-discovered in the other; the next manifest-backed entity mints a third copy.
- **Suggested fix:** Extract a generic manifest-entity CRUD helper parameterized by (manifest field, filename, project path, validator).
- **Confidence:** High.

#### [F-030] Split the 13.2k-line rust-api `tests.rs` monolith
- **Category:** readability
- **Severity:** Medium
- **Location:** `apps/rust-api/src/tests.rs:1-13241`
- **Finding:** 183 tests (147 async end-to-end router tests) in one essentially flat file. Auth-critical coverage is actually strong (throttle, loopback trust, method-aware gating, event/media tickets, project-file traversal), but locating and extending it requires text search, any edit recompiles the whole module, and new route families (control-overlays has no dedicated route test) get appended ad hoc.
- **Impact:** Test-rot risk and invisible per-feature coverage gaps.
- **Suggested fix:** Split into per-domain modules (`tests/auth.rs`, `tests/media.rs`, `tests/catalog.rs`, …) sharing helpers via a `tests/support` module.
- **Confidence:** High.

### Web (shell + screens)

#### [F-031] Share one job-request builder between ImageStudio single-submit and batch (img2img + PiD silently dropped)
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/web/src/screens/ImageStudio.jsx:1883-1958` vs `1765-1870`
- **Finding:** `buildBatchJobRequest` is a ~75-line near-copy of `submit()`'s payload that has drifted: the batch `advanced` block omits `pidTarget` and the `supportsImg2img`/`img2imgReferenceAssetId`/`img2imgStrength` trio, and the top-level `referenceAssetId` drops submit's img2img branch.
- **Impact:** A batch run on an img2img-capable model (Krea 2 Turbo) with a reference image silently ignores it; a batch with PiD "2K" selected renders at the slower 4K default. Batch output differs from single Generate with identical visible settings.
- **Suggested fix:** Extract one shared `buildJobRequest(promptOverrides)` used by both paths (the structured-caption branch is the only real difference).
- **Confidence:** High on the drift; Medium that img2img-in-batch wasn't a deliberate omission.

#### [F-032] Thumbnail and virtualize the Library asset grid
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `apps/web/src/components/assetPanels.jsx:293-342` (AssetGrid), `apps/web/src/screens/LibraryScreen.jsx:188-196`
- **Finding:** `AssetGrid` maps every visible asset to `<AssetMedia>`: image tiles load full-resolution originals (no thumbnail endpoint used here) and video tiles mount a full `<video controls preload="metadata">` each (unlike `AssetThumbnail`, which uses poster JPEGs). No windowing/virtualization exists on the Library, Pose Library, or character-asset grids.
- **Impact:** A few-hundred-asset project decodes hundreds of full-res images and opens a metadata connection per video on every Library visit — memory/network blowup that grows linearly, with visible jank in WebView2/WKWebView.
- **Suggested fix:** Render `AssetThumbnail` (poster-based) in grid tiles, add a server thumbnail size param, and window the grid (`content-visibility: auto` as a cheap first step, or react-window).
- **Confidence:** High on mechanics; confirm impact by profiling a 500-asset project.

#### [F-033] Virtualize the Logs screen and back off its poll on failure
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `apps/web/src/screens/LogsScreen.jsx:30-31,101-124,153-156,232-256`
- **Finding:** The snapshot mirrors the entire 5,000-entry server buffer and renders all rows as DOM (re-rendered on every 2s poll append); the `setInterval(poll, 2000)` keeps firing at full rate when every request fails, with no backoff.
- **Impact:** A full buffer holds 5,000+ rows plus expanded JSON; each tick re-renders the list, and in remote-auth mode a dead API is hammered every 2s indefinitely.
- **Suggested fix:** Window the rendered tail (search already runs over the in-memory snapshot) and back off the interval after consecutive failures.
- **Confidence:** High.

#### [F-034] Break up the 4,100-line `ImageEditor.jsx` god module
- **Category:** readability
- **Severity:** Medium
- **Location:** `apps/web/src/screens/ImageEditor.jsx:1-4121` (esp. `renderToolPanel` 2325-2823, `renderLoraSection` 2825-2888, `renderLayers` 3350-3480)
- **Finding:** The per-tool logic was genuinely extracted (sc-9752 hooks), but the file is still 4,121 lines: exported helpers, eight inline `render*()` closures rebuilt per render, an inline SVG icon set, and the redesign shell all in one module. The edit-model `<select>` is duplicated between the Edit and Boxes panels, and `renderLoraSection` re-implements the slot/weight UI that `LoraPickerSection` already provides.
- **Impact:** The highest-churn screen remains the hardest to review; the duplicated LoRA/model controls will drift like the studios' builders already have.
- **Suggested fix:** Move tool panels into memoized components under `screens/imageEditor/`, extract the icon map and re-export block, and converge the editor LoRA section on `LoraPickerSection`.
- **Confidence:** High.

#### [F-035] Continue extracting feature hooks out of the ImageStudio/VideoStudio god components
- **Category:** readability
- **Severity:** Medium
- **Location:** `apps/web/src/screens/ImageStudio.jsx:272-1694`, `apps/web/src/screens/VideoStudio.jsx:82-711`
- **Finding:** ImageStudio is 3,244 lines with ~60 `useState` and ~25 effects in one component; VideoStudio repeats the same skeleton with parallel-but-diverging markup. Shared logic was factored into `useGenerationStudio`/`useSavePreset` (good), but per-feature knob clusters (batch, control, tier, PiD, img2img) keep accreting into the top-level component (14 `exhaustive-deps` disables in ImageStudio alone).
- **Impact:** Every new capability adds state+effect+payload wiring in a 3k-line closure; the F-031 submit/batch drift is a direct symptom.
- **Suggested fix:** Pull the batch subsystem and the strict-control/tier/PiD clusters into dedicated hooks colocated with their validation, following the proven `useGenerationStudio` pattern.
- **Confidence:** High.

#### [F-036] Group the ~45-prop bags TrainingStudio threads into its panels
- **Category:** readability
- **Severity:** Medium
- **Location:** `apps/web/src/screens/TrainingStudio.jsx:1453-1552`; `screens/training/DatasetEditorPanel.jsx:81-143`; `ConfigureJobPanel.jsx:40-83`
- **Finding:** All state/handlers live in the 1,567-line TrainingStudio and are threaded as ~45 individual props into `DatasetEditorPanel` and ~35 into `ConfigureJobPanel`. This is exactly the prop-drilling shape the F-005 stale-closure bug grew out of.
- **Impact:** Any dataset-editor feature touches three files and a fragile positional prop list; stale-closure/missed-prop bugs are structurally likely.
- **Suggested fix:** Group props into 3–4 cohesive objects (dataset session, caption session, doctor, config) or a small Training-scoped context.
- **Confidence:** High.

#### [F-037] Make the two `useJobEvents` inputs that violate its stability contract actually stable
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/web/src/hooks/useJobEvents.js:18-23,44-212` (eslint-disable at 211); `App.jsx:1564-1577` (`hasVisibleLocalFailure`); `hooks/useTimelines.js:270-299` (`enqueueTimelineGenerationApply`)
- **Finding:** The hook documents "App feeds them identity-stable," but `hasVisibleLocalFailure` and `enqueueTimelineGenerationApply` are plain function declarations recreated every render; the SSE effect (deps `[access.authRequired, ready, token]`) captures whichever closure existed at subscribe time and never refreshes it. Safe today only because both read live state via refs/args — the same latent stale-closure class as the prior F-009, with no lint or test to catch a future edit.
- **Impact:** Any future change that makes `applyCompletedTimelineGeneration` read `activeProject`/`timelines` directly silently freezes at the subscribe-time snapshot for the whole session.
- **Suggested fix:** Wrap both in `useCallback([])` bodies delegating through refs (the `stableRefreshData` pattern), or publish them via refs like the other nine callbacks; fix the doc comment.
- **Confidence:** High on the contract mismatch.

#### [F-038] Retry the access probe on failure so the login gate can't fail open with no recovery
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/web/src/hooks/useAccessGate.js:18,40-45,122-131`; gate render `App.jsx:2272`
- **Finding:** If mount-time `GET /api/v1/access` fails (transient blip, API mid-restart), `.finally(() => setAccessResolved(true))` releases the gate while `access` stays at its `{ authRequired: false }` default. `authenticated` then evaluates true with no token, so all data loads and the SSE connect fire unauthenticated, and — because the login band renders only when `access.authRequired` is true — the password prompt never appears. No probe retry exists.
- **Impact:** On an auth-required remote deployment, one failed probe produces a 401 storm and no login path; the only recovery is a manual reload — the failure mode the `accessResolved` gate was built to prevent, re-entered through the error path.
- **Suggested fix:** Retry the access probe with backoff (mirroring the media-ticket mint loop in the same file), or treat a failed probe as unresolved (keep holding data loads) while surfacing a "can't reach host" notice.
- **Confidence:** Medium — logic verified; end-user impact needs a remote-auth deployment with a dropped `/access`.

#### [F-039] Stop per-keystroke/per-navigation invalidation of the static app context
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `apps/web/src/App.jsx:1269-1293` (`createPlaceholderJob` deps), `1999-2000,2129` (`jobPrompt` in `appStaticValue`)
- **Finding:** `jobPrompt` (edited per keystroke in QueueScreen) and `activeView` sit in `createPlaceholderJob`'s dep array, and `jobPrompt` is a direct member of `appStaticValue`. Every keystroke in the Queue placeholder field and every sidebar navigation mints a new `appStaticValue` identity, re-rendering every `useAppStatic()` consumer.
- **Impact:** Directly undermines the sc-8855 static/live split; bounded today (only the mounted screen consumes it) but any always-mounted static consumer inherits per-keystroke re-renders.
- **Suggested fix:** Move `jobPrompt`/`setJobPrompt` into QueueScreen local state (consumed nowhere else — verified), and read `activeView` via the existing `activeViewRef` inside `createPlaceholderJob`.
- **Confidence:** High.

### Aux (MCP / tests / docs)

#### [F-040] Re-enable a Host/Origin (DNS-rebinding) defense on `/mcp`
- **Category:** security
- **Severity:** Medium
- **Location:** `crates/sceneworks-mcp/src/lib.rs:33-56`
- **Finding:** The transport is built with `StreamableHttpServerConfig::default().disable_allowed_hosts()`, justified only by "the real access control is the surrounding access_control middleware." In the shipped default desktop posture (loopback bind, no token, or `SCENEWORKS_TRUST_LOOPBACK=1`) that middleware performs no Host/Origin validation, so a malicious web page can use DNS rebinding to reach `/mcp` from the victim's browser and drive job submission / ticketed file reads.
- **Impact:** Drive-by browser content can escalate from "any local process is trusted" to "any remote website is trusted" on default desktop installs, via `/mcp` tools.
- **Suggested fix:** Configure `allowed_hosts` from the deployment (loopback by default; add the LAN host when remote access is on) instead of disabling the check, or add a Host-header validation layer in front of `/mcp`.
- **Confidence:** Medium — mechanism verified in this crate; confirming `auth.rs` has no Host/Origin check in the tokenless path would raise this to a systemic High.

#### [F-041] Cap the inline base64 payload of the MCP `generate_image` tool
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `crates/sceneworks-mcp/src/server.rs:432-468`
- **Finding:** The tool fetches every result asset with `get_bytes` and base64-inlines all of them with no per-image or total size cap; `count` goes to 8 and dimensions are caller-chosen, so a 2048²×8 job returns a multi-tens-of-MB JSON-RPC response (held twice in memory as raw bytes + base64).
- **Impact:** Large responses can exceed MCP client message limits or blow the model context, turning a completed render into a failed tool call after the GPU work is done.
- **Suggested fix:** Above a threshold (~2–4 MB/image or ~10 MB total) fall back to the `get_job_result` ticketed-link shape and say so in the summary.
- **Confidence:** High.

#### [F-042] Drain spawned-binary stdout/stderr pipes in the pytest harness
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `tests/test_rust_api_worker_smoke.py:252-269,522-555,582-692`, `tests/test_rust_api_contract_snapshots.py:202-241` (helpers in `tests/rust_api_harness.py:53-78`)
- **Finding:** Every API/worker spawn uses `Popen(stdout=PIPE, stderr=PIPE)` and reads the pipes only after `poll()` reports exit. Nothing drains them while tests run, so once a child writes more than the OS pipe buffer (~64 KB) of tracing output it blocks on write and the test hangs until its own deadline, mis-reported as "job did not reach status."
- **Impact:** Flaky-looking e2e/parity hangs whenever the Rust binaries get chattier (a `RUST_LOG` bump or a new warn-level log in a hot path is enough), costing full CI-timeout cycles to diagnose.
- **Suggested fix:** Send stdout to `DEVNULL` and stderr to a tempfile (read in failure messages), or add a background drain thread in `rust_api_harness`.
- **Confidence:** Medium — depends on CI log volume; reproduce with `RUST_LOG=debug` on a long smoke.

#### [F-043] Banner or archive the historical planning docs that contradict the shipped architecture
- **Category:** readability
- **Severity:** Medium
- **Location:** `documents/IMPLEMENTATION_PLAN.md:1-10`, `documents/SCENEWORKS_PLAN.md:1-15` (and vintage siblings `IMAGE_MODEL_RESEARCH.md`, `V1_RISK_REGISTER.md`)
- **Finding:** These plans describe "a local Docker-based app … FastAPI backend, Python workers" — the exact architecture the repo has since deleted (README: "There is no Python venv on any platform") — with no "historical" marker, sitting beside the current `TRAINING_QUICKSTART.md`, lending them apparent authority.
- **Impact:** New contributors can take the FastAPI/Python plan as current direction and waste effort or file wrong-premise issues.
- **Suggested fix:** Add a one-line "Historical — superseded, see README" banner to each stale plan or move them to `documents/archive/`.
- **Confidence:** High.

### Platform (desktop / CI / docker)

#### [F-044] Remove the stale hardened-runtime weakenings from the macOS entitlements
- **Category:** security
- **Severity:** Medium
- **Location:** `apps/desktop/Entitlements.plist:8-11`
- **Finding:** The signed, notarized app ships `com.apple.security.cs.disable-library-validation` and `com.apple.security.cs.allow-dyld-environment-variables`. The in-file justification (spawns uv/python loading unsigned dylibs; uses `DYLD_*`) is stale: the Python venv was retired, the bundled onnxruntime dylib is now codesigned with the same Developer ID (`scripts/build-sidecar.mjs:111-120`), and no `DYLD_*` var is set anywhere in the crate (only `ORT_DYLIB_PATH`/`PMETAL_METALLIB_PATH`).
- **Impact:** Both entitlements weaken hardened-runtime protection on every process: any same-team-unsigned dylib can be injected, and `DYLD_INSERT_LIBRARIES` attacks become possible — the class hardened runtime exists to stop.
- **Suggested fix:** Build a notarized test bundle with both keys removed; keep `disable-library-validation` only if the ort/coreml load actually fails, and drop `allow-dyld-environment-variables` outright.
- **Confidence:** Medium — stale rationale and same-team signing verified in-repo; a keys-removed notarized smoke would confirm removability.

#### [F-045] Narrow the remote IPC capability grant
- **Category:** security
- **Severity:** Medium
- **Location:** `apps/desktop/capabilities/default.json:5-7`
- **Finding:** The remote capability grants the full command set (including `set_credential`, `set_remote_access_password`, `set_remote_access`, `save_asset_as`) to `http://127.0.0.1:*` **and** `http://localhost:*`. The shell only ever navigates to `http://127.0.0.1:<port>` (`setup.rs:760`), so `localhost:*` is unused surface; and the port wildcard means any loopback-served page the window can be steered to receives keychain-write IPC. An XSS in the API-served React app could chain `set_remote_access_password` → `set_remote_access(true, …)` to enable the 0.0.0.0 LAN bind with an attacker-known token at next launch.
- **Impact:** Doubles the trusted-origin surface for no benefit and lets webview XSS escalate to "keychain write + network-exposure toggle."
- **Suggested fix:** Drop `http://localhost:*`; longer term, split the LAN-toggle/password commands into a capability granted only after a native confirmation dialog.
- **Confidence:** High for the unused grant; Medium for the XSS-chain exploitability.

#### [F-046] Pin the pip-installed NVIDIA runtime wheels in the candle worker image
- **Category:** security
- **Severity:** Medium
- **Location:** `docker/rust.Dockerfile:185-193` (also 80-82)
- **Finding:** The runtime stage pins `onnxruntime-gpu==1.26.0` but installs `nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-nvjitlink-cu12 nvidia-cuda-nvrtc-cu12` unpinned (and `huggingface_hub[cli]>=0.36,<1` as a range), no hashes — while the desktop provisioner pins exact URL + sha256 for the same DLL set (`cuda_provision.rs:81-158`).
- **Impact:** Server image builds are non-reproducible; a cuDNN 10.x release or a compromised `nvidia-*` PyPI publish flows silently into the next `docker compose up --build`, breaking or trojaning the validated cuDNN-9.23 ort provider.
- **Suggested fix:** Pin exact versions (`nvidia-cudnn-cu12==9.23.0.39`, etc.) mirroring the desktop manifest, ideally with `--require-hashes`.
- **Confidence:** High for the unpinned installs; Medium for exploit likelihood.

#### [F-047] Isolate signing material from npm lifecycle scripts on the release Mac
- **Category:** security
- **Severity:** Medium
- **Location:** `.github/workflows/release.yml:63-66`
- **Finding:** The release job runs `npm ci` for apps/web and apps/desktop on the self-hosted signing Mac whose login keychain holds the Developer ID cert and whose disk holds the notary `.p8`. Step-scoping the secret env vars (sc-8864, verified fixed) does not protect these: any compromised npm dependency's install script runs in the same session and can invoke `codesign` against the unlocked keychain or read the `.p8`.
- **Impact:** A supply-chain compromise of any transitive npm dependency yields the ability to sign and notarize arbitrary binaries as the developer — the exact trust the updater chain depends on.
- **Suggested fix:** Run `npm ci --ignore-scripts` on the release lane (verify Vite/Tauri builds don't need lifecycle scripts), move the cert into a dedicated keychain requiring per-use unlock, and store the `.p8` outside the runner user's readable path, injected only into notary steps.
- **Confidence:** Medium — exposure verified from the workflow; `--ignore-scripts` drop-in needs one test build.

#### [F-048] Align the Cargo workspace version with the product version
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `Cargo.toml:26` (`version = "0.2.0"`), `scripts/sync-version.mjs:34-51`, `scripts/check-scaffold.mjs:197-228`
- **Finding:** The product is at 0.7.3 (root/desktop/web package.json + tauri.conf.json, all guarded by `assertVersionsAligned`), but the Rust workspace version has sat at 0.2.0 — `sync-version.mjs` never touches Cargo.toml and the scaffold check doesn't include it. Every binary's `CARGO_PKG_VERSION` reports 0.2.0, including the `appVersion` stamped into project files (`settings.rs:775-784`).
- **Impact:** Logs, health payloads, and project-file `appVersion` claim 0.2.0 while users run 0.7.3 — misleading during support and for any future version-gated migration.
- **Suggested fix:** Have `sync-version.mjs` rewrite `[workspace.package] version` and add Cargo.toml to `assertVersionsAligned`.
- **Confidence:** High for the divergence.

#### [F-049] Add a timeout to the first-run GPU-runtime download and unblock Retry on a hang
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/desktop/src/cuda_provision.rs:515-517`, `apps/desktop/src/setup.rs:1766-1777`
- **Finding:** The 2.7 GB first-run provisioner builds `reqwest::Client::builder().build()` with no timeout (the F-001 class), and `start_setup` holds the `running` guard across the entire awaited `run_startup`. A stalled connection mid-download hangs forever, and because `running` never clears, the setup screen's Retry button silently no-ops until force-quit.
- **Impact:** A user on a flaky network gets an unrecoverable, unexplained frozen setup screen at the highest-churn moment of the product.
- **Suggested fix:** Set connect + per-chunk read timeouts on the client and surface a timeout error so `start_setup` returns and re-arms Retry.
- **Confidence:** High.

#### [F-050] Track per-component provisioning so a retry doesn't re-download the whole 2.7 GB
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `apps/desktop/src/cuda_provision.rs:519-559` (+ `apps/desktop/README.md:300-304`)
- **Finding:** A failure on any of the 8 components aborts the run, deletes the temp wheels, and writes no marker; the only retry skip is the *full* sentinel set, so a partially-complete stage (CUDA wheels done, onnxruntime failed) re-downloads all ~2.7 GB including the ~1.2 GB already extracted. The README's "The download resumes" claim is false.
- **Impact:** On flaky links first-run cost multiplies per failure; metered-connection users pay repeatedly for bytes already on disk.
- **Suggested fix:** Skip a component whose extracted DLLs already exist (per-component sentinel via `dir_has_dll`) or write per-component markers; fix the README wording.
- **Confidence:** High.

#### [F-051] Deduplicate the desktop GPU-worker supervision loops
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/desktop/src/setup.rs:893-1063` and `1081-1258`
- **Finding:** `supervise_mlx_worker` and `supervise_candle_worker` duplicate a ~170-line supervision skeleton (id minting, spawn-with-env, PID recording, stdout/stderr pumping, Terminated/Error handling, exponential backoff with 20s reset, shutdown checks); only the env block differs.
- **Impact:** Any fix to backoff/restart/logging must land twice; the loops have begun drifting — the shape from which the past orphan-process class recurs.
- **Suggested fix:** Extract `supervise_worker(app, log_name, id_prefix, env_builder, pid_recorder)`; keep the per-platform env closures at the call sites.
- **Confidence:** High.

---

## Low findings

*Grouped by subsystem; each bundles several same-severity occurrences. All locations verified.*

#### [F-052] Worker CV low-severity cluster
- **Category:** bad-pattern / redundant / dead-code
- **Severity:** Low
- **Location:** `pose_jobs.rs:310-319` (tautological `simcc_y` length guard — `wy` derived, can't fail), `segment_jobs.rs:43-67` == `upscale_jobs.rs:823-847` + near-copy `pose_jobs.rs:1042-1059` (`resolve_source` path-confinement helper triplicated), `pose_jobs.rs:695-701` (stale `#[allow(dead_code)]` on `detect_and_render_skeleton` — caller now exists at `control_preprocess.rs:229-237`), `person_segment.rs:138-157` (`mask_box_coverage` indexes without a `pixels.len()` guard — sibling `scail2_masks::paint` got the sc-8907 guard), `person_jobs.rs:244-263`/`pose_jobs.rs:170-196,240-266` (per-channel bilinear samplers refetch neighbors 3×; `sample_rgb`/`sample_bgr` near-duplicates).
- **Finding:** Consistency/cleanup items; the path-confinement triplication (security-relevant helper) and the missing mask length guard are the two worth prioritizing.
- **Suggested fix:** One shared `resolve_asset_media_path` helper; add the `pixels.len()` guard; delete the dead y-guard and stale allow; sample once per pixel.
- **Confidence:** High (mask-guard reachability Medium — needs an engine bug).

#### [F-053] Worker infrastructure low-severity cluster
- **Category:** bad-pattern / dead-code / efficiency
- **Severity:** Low
- **Location:** `manifest.rs:194-209` (async `write_json_value` skips the `sync_all` its blocking twin does), `util.rs:72-85` (`bounded_tail` mixes byte len with char count, keeps one extra char), `lib.rs:604-623` (signal listeners re-created per select → signals in the gaps dropped), `mlx_fit_gate.rs:211-237` (`sum_safetensors_bytes` follows dir symlinks with no cycle guard → stack overflow on a looped symlink), `lib.rs:163-169,420-430` (`face_likeness` ~62KB + `control_dataset_byo` ~23KB whole-module `#[allow(dead_code)]`, no production caller yet), `job_metrics.rs:91-127`+`gpu.rs:220-237` (subprocess GPU sampling ~1 process/sec/job), `bernini_tier_build.rs`/`wan_*_tier_build.rs` (dir_size/report_tier/QUANT_TIERS copy-pasted across four `#[ignore]` harnesses), `gpu.rs:56-194` (~130 lines of repetitive capability-push blocks, list restated in `mlx_gpu`).
- **Finding:** The symlink cycle guard and the async-writer fsync are the two with real (if narrow) durability/crash consequences; the rest are cleanup.
- **Suggested fix:** As noted per item; add `sync_all().await` to the async writer, skip symlinked dirs / add a depth cap, and consolidate the tier-build helpers into `smoke_support.rs`.
- **Confidence:** High.

#### [F-054] Worker training/analysis low-severity cluster
- **Category:** bad-pattern / efficiency / redundant
- **Severity:** Low
- **Location:** `kps_jobs.rs:190-211` (SCRFD `WEIGHTS` cache not path-keyed while its detector cache is → stale weights if the path changes mid-process), `control_dataset_byo.rs:334-335` (COCO keypoints JSON parsed twice), `prompt_refine_jobs.rs:835-841` (vision-ref decode+downscale on the async runtime thread, unlike the sc-8909-wrapped siblings), `model_jobs.rs:1714-1718` (`huggingface_snapshot_dir` called twice → re-runs the Windows hardlink walk), `training_jobs.rs:929` (`SamplePersister` rebuilds the config instead of receiving the finalized one), `caption_jobs.rs:349-350` (`projectId`/`datasetId` validated only after the full GPU run), `control_dataset_byo.rs` (module-level dead code + unreachable `IngestRoute::Normalize`, tracked by sc-10171), `fail_stranded_mlx/candle_jobs` + `fail_unsupported_*` structural twins in jobs_store.
- **Finding:** Mostly latent (the SCRFD path is deterministic today); the caption prereq validation wastes a full GPU run on a malformed payload and is the cheapest real win.
- **Suggested fix:** Hoist the caption `required_payload_string` checks above the model load; key `WEIGHTS` on `(PathBuf, Weights)`; parse COCO once; fold decode into the blocking closure.
- **Confidence:** High.

#### [F-055] Core low-severity cluster
- **Category:** redundant / efficiency / security / bad-pattern
- **Severity:** Low
- **Location:** `jobs_store/routing/candle.rs:656-1031` (16 byte-identical `*_edit`/`*_ipadapter`/pose-gate predicate bodies + ~12 repeated `has_nonempty_id` closures), `project_store.rs:2537-2544` (`find_project_path` re-reads+parses `recent-projects.json` on every project-scoped request, incl. every media GET), `project_store.rs:2825-2876` (`upscale_lineage_group` reads+parses every asset sidecar per asset move, quadratic), `session_log.rs:99-116,167-222` (redaction re-lowercases the whole line per marker under the buffer mutex), `session_log.rs:187-193` (redaction misses `token : value` / YAML-ish shapes and `secret`/`password`/`apikey` keys), `lora_url.rs:81-131` (SSRF blocklist relies on connect-time re-validation for domain hosts; missing 100.64/10, 192.0.0/24, 198.18/15, 192.88.99/24), `jobs_store.rs:1698-1706` vs `contracts.rs:862-866` (progress-report `(None, Some(owner))` rejection contradicts the documented legacy pass-through).
- **Finding:** The redaction shape gaps and SSRF reserved-range omissions are the security-relevant members (both bounded by the loopback default); the sidecar-scan-per-move is the sharpest efficiency edge on large projects.
- **Suggested fix:** Extract `edit_with_source_eligible`/`pure_reference_eligible`/`pose_conditioned_eligible` helpers; cache the parsed registry with an mtime check; resolve fold groups from the assets index; lowercase once per redaction call and extend the marker set + reserved ranges.
- **Confidence:** High (SSRF real-world exposure Medium — depends on consumer re-validation).

#### [F-056] rust-api low-severity cluster
- **Category:** bad-pattern / redundant / efficiency
- **Severity:** Low
- **Location:** `manifest.rs:315-336` (quadratic `merge_entries_by_id`, up to ~16M comparisons per submit with 4096-adapter external roots), `preferences.rs:60-66,121-149` (unserialized non-atomic `set_ui_preferences` write), `training.rs:1286-1296`/`loras.rs:848-886`/`models.rs:660-674` (blocking FS probes inline on the async runtime, unlike the sc-4202 catalog paths), `recipe_presets.rs:47-97`/`prompt_batches.rs:56-102` (create path never strips client `manifestPath`/`appliedDefaults`/`lastUsedAt`, unlike update), `lib.rs:2512-2524`/`generation.rs:106-125,149-174` (character-test/vqa/interleave validators hard-code `4000` and skip `validate_prompt_extras`), `lib.rs:2476-2510` (job `loras` array uncapped, each triggers a header read), `poses.rs:42-69`/`keypoints.rs:23-50` (staged multipart uploads leak on a mid-stream error, unlike `import_asset`), `jobs.rs:402-435` (non-transactional read-then-write `status_changed`), `assets.rs:255-259` (`move_asset_to_character` uses raw `Json` not `ApiJson`), `jobs.rs:510-656` vs `664-802` / `loras.rs:952-1093` vs `models.rs:732-844` (registration + import staging clones), `lib.rs:1413-1420` (`get_project_file` maps a vanished-file race to 500 not 404).
- **Finding:** The uncapped `loras` array and the create-path field-strip divergence are the two mild-security members; the staged-upload leak and blocking-FS-on-async are correctness/robustness cleanups the crate's own disciplines already prescribe.
- **Suggested fix:** HashMap-index the merge; route preferences through `write_manifest_atomic`+lock; `spawn_blocking` the FS probes; call the strip helper + `validate_prompt_extras` + `MAX_PROMPT_CHARS` in the three validators; cap attached LoRAs; clean staged temps on multipart error; map `NotFound`→404.
- **Confidence:** High (merge impact Medium — needs a 1–4k-adapter benchmark).

#### [F-057] Web-shell low-severity cluster
- **Category:** bad-pattern / redundant / efficiency
- **Severity:** Low
- **Location:** `hooks/usePresets.js:14-29`/`usePromptBatches.js:12-27`/`useTraining.js:19-48` (missing the sc-8858 stale-project guard the other five refreshers have), `hooks/useJobEvents.js:147-156` (`setWorkers(summary.workers.sort(...))` mutates the object just committed as `queueSummary`), `hooks/usePresets.js` ≈ `usePromptBatches.js` (~100-line structural clone), `App.jsx:1965-2150` (~130-entry hand-mirrored `appStaticValue` dep array guarded only by warn-level `exhaustive-deps`), `hooks/useGenerationMetrics.js:61`/`App.jsx:918-923` (uncancelled async setState), `App.jsx:2369-2406`+`assetPanels.jsx:708` (unmemoized `FullscreenPreview` re-renders on every SSE tick).
- **Finding:** The three unguarded refreshers are the one to prioritize — the moment any gains an SSE/unaborted call site, the sc-8858 project-clobber race returns; the warn-level lint is the systemic weakness behind F-037.
- **Suggested fix:** Add the `activeProjectRef.current.id !== projectId` guard to the three refreshers (and to `projectScopedRefreshStaleGuard.test.jsx`); copy-before-sort; extract a `useScopedCrudList` factory; promote `exhaustive-deps` to error or run CI with `--max-warnings 0`.
- **Confidence:** High.

#### [F-058] Web-screens low-severity cluster
- **Category:** bad-pattern / dead-code
- **Severity:** Low
- **Location:** `ImageStudio.jsx:1091-1104`/`ReferenceCaptionPicker.jsx:70-98` (aspect-probe `new Image()` with no cleanup/staleness guard, logic duplicated), `ImageEditor.jsx:2108-2153` (AI-op completion async IIFE with no unmount guard → object URL created after unmount never revoked), screen test gaps (`CharacterStudio`, `EditorScreen`, `QueueScreen`, `LibraryScreen`, `DocumentStudio`, `SetupWizard`, `ReplacePersonPanel` — the timeline undo/commit and person-track correction math are the untested algorithm-dense surfaces).
- **Finding:** All benign in React 18 except the editor object-URL-after-unmount leak (one URL per abandoned op) and the untested correction/timeline math.
- **Suggested fix:** Null `onload`/add a `cancelled` flag and share one `probeImageDimensions` helper; add an `aliveRef`/AbortController to the editor effect; add targeted tests for `EditorScreen` commit/undo and `ReplacePersonPanel` diffing.
- **Confidence:** High (editor leak Medium — needs the exact unmount-window repro).

#### [F-059] Aux (MCP / tests / docs / assets) low-severity cluster
- **Category:** bad-pattern / redundant / dead-code / efficiency
- **Severity:** Low
- **Location:** `mcp/src/server.rs:45-49,362-377` (`MAX_CONSECUTIVE_POLL_ERRORS=5` tolerates only ~5s of API downtime, not the "brief restart" the comment claims; count scales with interval), `mcp/tests/*.rs` (`spawn`/`snapshot`/`StubState` copied across three files with drift) + `server.rs:1008-1027,1121-1145` (overlapping MIME tables), `tests/rust_api_harness.py:32-36` (`free_port()` bind race), `tests/test_builtin_manifest_audit.py:93-128` (JSONC audits match whitespace-exact substrings + a brace walker that counts braces in strings; the parsed-dict path already exists), `test_builtin_manifest_audit.py:12-15` (docstring references deleted `worker_runtime_shared.py`), `conftest.py:55-63`+snapshot tests (double `terminalreporter` lookup, redundant `@pytest.mark.parity`, `atexit` writer, inconsistent workerId `startswith` vs `==`), `docs/sc-3734` + `docs/sc-4422-kps-experiment` (~10 MB frozen torch spike code + montage PNGs; a referenced `render_all11.py` was never committed), `poses/` (92 files byte-identical to `apps/web/public/poses/`, no committed consumer or `index.json` generator), `data/character_sheet_3x3b.webp` + `design/*.zip` (stray/opaque committed binaries), `image-quality/src/lib.rs:216-230,371-394,440-472` (Vec `contains` O(n²), per-pixel bounds-checked `get_pixel`, two exposure passes that could share one histogram), `scripts/spike_lokr_roundtrip.py:127-141` (runs work at import top-level, wrong directory).
- **Finding:** The MCP poll-failure window (aborts a live render on a routine API restart) and the manifest-audit fragility (format change → false CI fail) are the two functional members; the rest is repo-hygiene residue worth one cleanup pass.
- **Suggested fix:** Make the MCP tolerance time-based; run manifest audits against `_load_builtin_models_manifest()`; let the API bind port 0; extract a shared MCP test harness; banner/prune the docs spike artifacts and de-duplicate `poses/`.
- **Confidence:** High.

#### [F-060] Platform low-severity cluster
- **Category:** bad-pattern / redundant / dead-code / efficiency / security
- **Severity:** Low
- **Location:** `config/manifests/builtin.control_overlays.jsonc:1-63` (no `$schema`, no `packages/schemas/` counterpart, absent from check-scaffold — every sibling has all three), `config/manifests/builtin.models.jsonc` (0/64 model downloads pin an HF `revision` while the control-overlay manifest documents the pin "so a repo re-push can't swap the checkpoint"), `scripts/check-health.mjs:3` (default port 8000, stack default is 8010), `apps/desktop/src/settings.rs:655-662` vs `677-684` (`choose_folder`/`choose_data_dir` byte-identical, two commands + two ACL grants), `apps/desktop/src/setup.rs:250-269` (`append_log` re-opens the log file per captured line; the API pump holds one handle), `docker-compose.yml:136-143` (CPU utility `rust-worker` declares an NVIDIA device reservation), `packages/schemas/*.schema.json` (model/LoRA item schemas require nothing though every entry carries `id`/`name`/`family`).
- **Finding:** The missing control-overlay schema (a malformed edit ships past `npm run check`) and the unpinned model revisions (defense-in-depth the repo applies elsewhere) are the members worth scheduling.
- **Suggested fix:** Add `control-overlay-manifest.schema.json` + register the pair; pin `revision` on the SceneWorks-org turnkey entries; fix the health-check port; collapse the folder-picker commands; open the worker log once per iteration; drop the GPU reservation from `rust-worker`; add `required` to the manifest schemas.
- **Confidence:** High.

---

## Informational

#### [F-061] Positive confirmations — prior-review remediation and security fundamentals held
- **Category:** security
- **Severity:** Info
- **Location:** repo-wide (evidence per item below)
- **Finding:** All nine 2026-07-01 High findings are FIXED at HEAD, verified against current code: SHA-pinned CI actions + Dependabot (`.github/workflows/*`, `dependabot.yml`), confined import `sourcePath` (`paths.rs:434-462`), the `CancelJoinGuard` cancel-and-join across every streaming consumer (`progress.rs:79-175`), the image-detail heartbeat arm (`detail.rs:486-541`), no per-chunk cancel poll (`downloads.rs:1034-1057`), SAM2/SAM3 propagate cancel+progress (`person_segment_sam3*.rs`), the `passwordDraft` login gate (`useAccessGate.js`), the media-ticket system end-to-end (`tickets.rs`+`auth.rs:263-309`+`assetMedia.jsx`), and the stable `refreshData` context wrappers (`App.jsx:630-643`). Ongoing strengths verified this pass: parameterized SQL throughout, canonicalizing path confinement on file-serving routes, constant-time token compare, an SSRF guard covering the IPv4-mapped-IPv6 bypass, atomic temp+rename writes, secrets written 0600, zip-slip-guarded extraction, and pinned+sha256-verified download manifests for both build-time staging and first-run provisioning.
- **Impact:** The remediation campaign was thorough and test-backed; the media/auth trust boundary and the GPU-abandon/heartbeat class are genuinely closed.
- **Suggested fix:** None — retain the F-number-citation + regression-test discipline for this cycle's fixes.
- **Confidence:** High.

#### [F-062] Accepted, documented residual risks
- **Category:** security
- **Severity:** Info
- **Location:** `apps/web/src/accessToken.js:5-23` (token plaintext in localStorage), `apps/web/src/api.js:40-74` (SSE/media tickets in URL query params), `apps/rust-api/src/server.rs:307-380` (plain HTTP in LAN mode), `apps/rust-api/src/auth.rs:20-126` (shared-IP throttle can lock out all clients behind one proxy IP), `apps/rust-api/src/tickets.rs:76-92` (ticket lookup not constant-time), `apps/desktop/src/cred_ipc.rs:67-133` (non-constant-time socket token + predictable fallback token), `crates/sceneworks-worker/src/image_jobs.rs:1531-1552` (documented O(images²·steps) `streaming_result`, sc-8953).
- **Finding:** Each is a deliberate, documented tradeoff bounded by the single-user loopback/opt-in-LAN model; none is a defect under the shipped posture. Listed so a future reviewer inherits the context rather than re-flagging them.
- **Impact:** These widen as remote-access mode hardens (TLS, httpOnly-cookie exchange, per-scope tickets); the single-seam designs (`accessToken.js`, `tickets.rs`) keep those future changes contained.
- **Suggested fix:** Track under the epic-4484 hardening work; the cheap wins (constant-time ticket/socket compare, `getentropy` fallback token, scrub `ticket` from server request logs) can land opportunistically.
- **Confidence:** High.

---

## Themes and systemic observations

1. **MLX↔Candle twin drift is the dominant correctness risk.** Fixes and pins repeatedly land on one backend and miss the other: HF revision pins (F-007), plan `expectedCount` (F-009), the caption progress-coalescing fix (F-016), terminal-cancel timing (F-017), and residual SAM3 twin duplication (F-018). Shared modules (`person_segment_sam3_common`, `analysis_jobs_common`, the sensenova driver) are the right antidote but were applied incompletely, and new lane pairs keep accreting. Prefer generic-over-`Sam3FrameOutput`-style abstractions and shared consts over copy-paste twins.

2. **An established house pattern exists for almost every hazard, but is applied unevenly.** `run_blocking_with_heartbeat` (F-002), `resolve_env_file_pin` (F-011), atomic writes (F-023), the stale-project guard (F-057), `spawn_blocking` for FS on async (F-056), HTTP timeouts (F-001) — each has a canonical implementation the codebase already trusts, with 1–3 sibling call sites that never adopted it. The highest-leverage fix is a sweep that finds the stragglers, not new mechanisms.

3. **God modules concentrate the churn and breed the drift.** `ImageEditor.jsx` (4.1k, grew since last review), `ImageStudio.jsx` (3.2k, ~60 hooks), `App.jsx` (2.4k with a 130-entry hand-mirrored dep array), the `include!`-composed worker `image_jobs` with a 6.3k-line `base.rs`, `jobs_store.rs` (order-sensitive dispatch), and two 13k+ test monoliths. The batch/submit drift (F-031) and the Dataset Doctor stale-closure (F-005) are direct symptoms of state living in oversized closures with positional prop threading. Decomposition here pays down multiple finding classes at once.

4. **Robustness lives in the happy path; the unhappy paths are thinner.** No outbound HTTP timeouts anywhere (F-001/F-049), all-or-nothing first-run provisioning (F-050), a supervisor that aborts on one spawn error (F-013) and hard-kills on Windows (F-014), fail-open on a dropped access probe (F-038), and undrained subprocess pipes in the test harness (F-042). None bites under normal single-user desktop use; all bite on flaky networks, restarts, or LAN mode.

5. **Duplication has begun to drift, not just accumulate.** The prompt-batches/recipe-presets clone (F-029), the generator/refine caches (F-019), the desktop supervision loops (F-051), and the candle routing predicates (F-055) are all past the "identical copy" stage into "subtly divergent copy" — the point at which a fix in one silently fails to protect the other.

6. **Version/config/doc drift is low-stakes but pervasive.** Cargo workspace 0.2.0 vs product 0.7.3 (F-048), a stale ARCHITECTURE.md matrix (F-022), historical FastAPI plans with no banner (F-043), a 1,700-line changelog in Cargo.toml (F-021), and stale operational comments. Individually trivial; collectively they erode trust in the repo's own documentation.

## Coverage notes

- **Reviewed:** the entire live Rust workspace (all six crates + both apps), the full `apps/web` JS/JSX tree (shell, hooks, context, screens, components, module helpers), the platform layer (Tauri desktop source + config, all `.github/` workflows, `docker/`, root workspace/build files, `scripts/`, `config/manifests/`, `packages/schemas/`), the MCP and image-quality crates, and the `tests/` pytest suite (harness code read fully; test bodies surveyed structurally).
- **Surveyed structurally rather than line-audited (stated per subsystem):** the worker `*_smoke.rs` / `*_tier_build.rs` GPU harnesses and the two large test monoliths (`worker src/tests/*`, `rust-api tests.rs`, `apps/web *.test.jsx`), the ~1,200-line static `MODEL_TABLE`/capability tables in `engines.rs` and `jobs_store/routing/catalog.rs`, the ~1,900-line per-family literal tables in `training.rs`, and the `*.txt` prompt templates (treated as data).
- **Excluded:** `apps/web/src/styles.css` (~10k lines, design-only), binary assets and fonts, `Cargo.lock`/`package-lock.json`, individual manifest rows in `builtin.models.jsonc` beyond a programmatic consistency check (64 models, zero duplicate ids verified), `docs/sc-*` validation-record bodies (inventoried, flagged as historical), and `scripts/spikes/**` (frozen spike code, structural survey only).
- **Confirmed still load-bearing (not dead):** `apps/rust-worker` (the Docker/server worker entry point and the pytest harness's `SCENEWORKS_RUST_WORKER_BINARY`).
- **Confidence caveats:** findings dependent on runtime behavior are marked Medium/Low with the confirming action named (e.g. F-008 end-to-end reachability against jobs_store routing, F-012 thread-id logging, F-032 grid profiling, F-042 CI log volume). F-003's exploitability was verified this session (the API does not validate the model id at enqueue).
```
