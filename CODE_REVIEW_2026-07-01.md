# Full Codebase Review — SceneWorks — 2026-07-01

## Executive summary

- **Repository at a glance:** local-first AI image/video generation studio, mid-way through a Python→Rust migration. Rust workspace (~147k LOC across `apps/rust-api`, `apps/rust-worker`, `apps/desktop` Tauri, and `crates/sceneworks-core` / `-worker` / `-image-quality`), a React 18 + Vite frontend (~63k LOC JS/JSX in `apps/web`), a **retired** Python worker + shared package (~55k LOC Python, runtime-dead since 2026-06-27, tracked for deletion under epic 8283), and a live pytest parity/e2e suite. SQLite (`project.db`, `jobs.db`) for job/asset state, JSONC model manifests, MLX on macOS (via `mlx-gen`) and Candle on Linux/Windows for inference. Reviewed at commit `1c7521dd`.
- **Coverage:** the entire live tree was read across ten parallel subsystem passes — worker job modules (image/video/media/training/model/refine/sensenova/analysis), worker infrastructure (dispatch, downloads, GPU, caching, person/pose/segment/upscale/kps), `sceneworks-core`, `sceneworks-image-quality`, `apps/rust-api`, `apps/web` (screens + shell + `packages/`), `apps/desktop` + scripts + docker + CI, and the retired Python tree + live `tests/`. Test files were structurally surveyed rather than line-audited where noted; `styles.css` (~10k lines, design-only), binary assets, `Cargo.lock`, and the `data/` model manifests' individual rows were excluded. The retired `apps/worker/` Python got a liveness + secrets + inventory pass, not a line review (per its dead-code status).
- **Headline:** the codebase is in **good** shape — the security fundamentals are genuinely strong (parameterized SQL, canonicalizing path-confinement primitives with tests, constant-time token compare, SSRF guards covering the IPv4-mapped-IPv6 bypass, atomic writes, secrets written 0600-from-birth), and the three High findings from the 2026-06-15 review that I re-checked are all fixed in source. There are **no Critical findings** and no remotely-exploitable defect under the default loopback-desktop posture. The nine High findings cluster in three places: the **release/auto-update supply chain** (unpinned third-party GitHub Actions can reach the updater signing key), the **remote-access (epic 4484) auth seam** which is incomplete at every "headerless" boundary (browser `<img>`/`<video>`/download requests 401, and the login password is wired live into the API token so each keystroke floods the API), and a recurring **"streaming job abandons live GPU work / goes heartbeat-silent"** structural class in the worker. The dominant *maintainability* risk is unchanged from prior reviews: large, drift-prone **copy-paste duplication** — MLX-vs-Candle backend twins, per-model image/video lanes, per-hook poll loops, and four+ god modules (`jobs_store.rs`, `App.jsx`, `ImageEditor.jsx`, the worker `tests.rs`).
- **Counts:** Critical: 0 | High: 9 | Medium: 57 | Low: 78 | Info: 11 (155 findings total; many Low/Info entries group several same-shape occurrences across sibling files under one F-number).

---

## Critical findings

*None.* No exploitable-by-default, data-loss, or production-blocking defect was found. The path-confinement and auth gaps below are High rather than Critical because the shipped default posture is a single-user loopback desktop app; they rise in severity as epic 4484 (LAN remote access) widens the trust boundary.

---

## High findings

#### [F-001] Pin third-party GitHub Actions to commit SHAs in the release/signing path
- **Category:** security
- **Severity:** High
- **Location:** `.github/workflows/release.yml:59,162,166,172`, `.github/workflows/desktop-windows.yml:61,64,68,77`, `.github/workflows/check.yml:41,44`
- **Finding:** Third-party actions are referenced by mutable tag/branch: `dtolnay/rust-toolchain@stable` (a moving branch), `Swatinem/rust-cache@v2`, `ilammy/msvc-dev-cmd@v1`, `Jimver/cuda-toolkit@v0.2.35`. In `release.yml` these run in jobs whose env holds `TAURI_SIGNING_PRIVATE_KEY(_PASSWORD)` and the Apple notary credentials, and the macOS job runs on the self-hosted signing Mac. (Verified.)
- **Impact:** A compromised upstream tag (the tj-actions-style supply-chain pattern) can exfiltrate the minisign updater private key that signs the `latest.json` payloads the shipped app auto-installs — a full user-base compromise vector — and on the self-hosted runner gets code execution on the box holding the Developer ID cert.
- **Suggested fix:** Pin all non-`actions/*` actions to full commit SHAs (with a `# vX.Y.Z` comment); enable Dependabot `github-actions` updates to keep the SHAs fresh.
- **Confidence:** High

#### [F-002] Confine client-supplied `sourcePath` before importing LoRA/model files
- **Category:** security
- **Severity:** High
- **Location:** `crates/sceneworks-worker/src/model_jobs.rs:1640-1665,1981-1987`
- **Finding:** `run_lora_import_job` / `run_model_import_job` pass payload `sourcePath`/`secondarySourcePath` straight to `import_lora_source_path` / `import_lora_source_file_as`, which copy — or with `uploadedSourcePath:true`, **move** — the file with no confinement. Target dirs and every sibling *read* path are confined (`normalize_app_managed_lora_path`, sc-5723/WKA-002), but the import *source* is not. (Verified: `Path::new(source_path)` reaches the copy at L1665/L1982.)
- **Impact:** The worker is the stated trust boundary (unauthenticated local jobs API; LAN exposure via epic 4484). A crafted import job copies any host-readable file (e.g. `~/.ssh/id_rsa`) into `data/loras/<name>/` where the app makes it listable/fetchable — an arbitrary-file-read/exfiltration primitive; `prefer_move` additionally deletes the original.
- **Suggested fix:** Resolve both source paths through a confinement helper (allowed roots: the API staged-upload dir for uploads, plus `data_dir`); reject everything else with `InvalidPayload`.
- **Confidence:** High (missing confinement verified); Medium on real-world exploitability (depends who can post import jobs).

#### [F-003] Streaming job consumers abandon live GPU work when a progress/heartbeat POST fails
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `crates/sceneworks-worker/src/training_jobs.rs:835-1001` (esp. 849,914-946); same shape in `video_jobs.rs:2774-2848`, `caption_jobs.rs:225-285`, `prompt_refine_jobs.rs:871-909`, `media_jobs.rs:2522-2561` (ffmpeg), `model_jobs.rs:1294-1320` (`hf` CLI), `progress.rs:100-121`
- **Finding:** Every streaming consumer uses `select! { channel, interval } … update_job(...).await? / heartbeat(...).await?`, and on POST failure the `?` propagates out **without** tripping the engine `CancelFlag` or awaiting/aborting the blocking task; dropping a `JoinHandle` does not stop the task, and the trainer's `let _ = tx.blocking_send(...)` swallows the closed channel. (Verified in `training_jobs.rs`: the heartbeat/`update_job` `?` at L914-946 return with the spawned run still live.) Child processes (`run_ffmpeg`, `download_model_with_hf_cli`) are spawned without `kill_on_drop(true)` and leak on the same error paths.
- **Impact:** A transient API failure or 409 (job reclaimed by the stale sweep) leaves a multi-minute/multi-hour GPU denoise or training run burning unified memory while the worker returns and claims the next job — two concurrent GPU workloads on one Metal device, the documented SIGKILL/OOM class (sc-8390); orphaned ffmpeg/`hf` processes keep writing partial files.
- **Suggested fix:** One shared cancel-and-join guard (drop-guard that trips `CancelFlag` and awaits/aborts the task) used by all consumers; add `kill_on_drop(true)` to the ffmpeg and `hf` commands. Fixing instances individually will keep missing siblings.
- **Confidence:** High

#### [F-004] Add the heartbeat/cancel interval arm to the image-detail event loop
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `crates/sceneworks-worker/src/image_jobs/detail.rs:415-482`
- **Finding:** `run_image_detail_job` drives a bespoke `while let Some(..) = rx.recv().await` loop with no `tokio::select!` interval arm, so the cold SDXL load and each full 24-step tile refine emit nothing — no `Busy` heartbeat, cancel polled only at tile boundaries. This is the sc-4276 staleness bug that `consume_gen_events` (base.rs:2919-3051) was fixed for; the fix never propagated to this sibling.
- **Impact:** A cold-cache detail job can be falsely swept `interrupted` during model load (sc-8390: silent >90s ⇒ interrupted), and a user cancel takes up to a full tile (tens of seconds to minutes) to trip.
- **Suggested fix:** Mirror `consume_gen_events`: wrap the recv in `select!` with `interval.tick()` posting a heartbeat + the 2s `cancel_requested_peek`.
- **Confidence:** Medium (missing arm certain; the interrupted outcome depends on load duration vs the 90s window).

#### [F-005] Stop polling the cancel endpoint on every download chunk
- **Category:** efficiency
- **Severity:** High
- **Location:** `crates/sceneworks-worker/src/downloads.rs:790-810`
- **Finding:** `download_source_url`'s chunk loop calls `check_download_cancel` (a `GET /api/v1/jobs/{id}`) on **every** received HTTP chunk, on top of the interval-tick check in `report_download_progress`; the sibling `download_file_inner` checks only on the 5–15s tick. (Verified: the GET sits inside the `response.chunk()` arm at L797.)
- **Impact:** A multi-GB LoRA/model source download issues tens-to-hundreds of thousands of API GETs, serializes the transfer on API round-trips (chunk→GET→write), and hammers the SQLite-backed API — plausibly feeding the `claim_lock_contention` "database is locked" path.
- **Suggested fix:** Delete the per-chunk `check_download_cancel`; the interval arm already heartbeats + cancel-checks. Bonus (F-related): reuse the `JobSnapshot` returned by `update_job` for the cancel decision instead of a third GET per tick (`downloads.rs:1005-1018`).
- **Confidence:** High

#### [F-006] Wire cancel/progress into SAM2/SAM3 propagate — long propagates are heartbeat-silent and uncancellable
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `crates/sceneworks-worker/src/person_segment.rs:226-230`, `person_segment_sam3.rs:327-332,645-649`; call sites `media_jobs.rs:940,962,1120,2844,3040`
- **Finding:** `propagate` gained `cancel` + per-frame `progress` params (gen-core d8038beb) for exactly the video per-step cancel contract, but all MLX call sites pass `None, None`, and the segment step runs via plain `spawn_blocking` with no `run_blocking_with_heartbeat` wrapper — unlike the YOLO detect step (media_jobs.rs:600), which is wrapped citing sc-8390.
- **Impact:** A cold-start SAM3 segmentation (3.2 GB parse + quantize + ~24-frame 1008² propagate) can exceed 90s of heartbeat silence and be swept `interrupted` mid-work; user cancel is impossible mid-propagate.
- **Suggested fix:** Pass a progress closure that pings the heartbeat (or wrap the sites in `run_blocking_with_heartbeat`) and thread a real `CancelFlag`; the candle twin needs the candle-gen-sam3 API bump.
- **Confidence:** Medium (the `None,None` + missing wrap are verified; whether a real propagate exceeds 90s needs a cold-cache timing).

#### [F-007] Login password shares state with the live API token, flooding the API per keystroke
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `apps/web/src/App.jsx:759-764,985-990,1022-1162,2255-2272`
- **Finding:** The login gate's password `<input>` writes directly into `token` state (`onChange={... setToken}`) and `authenticated = accessResolved && (!authRequired || token.length > 0)`. Typing the first character flips `authenticated` true, so the `[authenticated, token]` effects fire `refreshData()` and tear down/rebuild the SSE connection (incl. `POST /jobs/events/ticket`) on **every keystroke**, each with a partial password that 401s. (Verified: input at L2262 binds `setToken`; `authenticated` memo at L762; effects at 990/1020/1162.)
- **Impact:** On an auth-required remote deployment (epic 4484's flagship scenario), typing the password floods the API with failing requests, fills the notices band with errors mid-typing, and churns EventSource connections with backoff timers.
- **Suggested fix:** Keep a separate `passwordDraft` state; only call `setToken` after `POST /api/v1/auth/verify` succeeds in `saveToken`. Derive gate visibility from `token` state, not a render-time `localStorage` read.
- **Confidence:** High

#### [F-008] Asset media URLs cannot authenticate — every image/video/document 401s in remote-auth mode
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `apps/web/src/components/assetMedia.jsx:4-18`, `assetPanels.jsx:95`, `assetActions.js:133-146`
- **Finding:** `assetUrl()` builds bare `GET /api/v1/projects/:id/files/*` URLs consumed by `<img src>`, `<video src>`, `DocumentReader`'s raw `fetch(url)`, and `browserDownload()`. That route is protected (not in `PUBLIC_PATHS`) and accepts tokens **only** via `X-SceneWorks-Token`/`Authorization` headers — no query-param or cookie path — but element-driven requests can't send headers, and `DocumentReader` doesn't pass the token. The SSE stream got a ticket mechanism for exactly this constraint; media never did. (Verified: `assetUrl` returns a bare `API_BASE_URL + …` string.)
- **Impact:** In the LAN remote-browser deployment (non-loopback peer + token set), every thumbnail, preview, video, pose preview, and browser download 401s — the UI works only where loopback trust applies.
- **Suggested fix:** Mirror the SSE ticket pattern for file URLs (short-lived signed ticket query param honored by `get_project_file`), or fetch media as blobs through `apiFetch` with object URLs; at minimum pass `token` into `DocumentReader`'s fetch.
- **Confidence:** High

#### [F-009] `appContextValue` memoization is silently defeated by unstable `refreshData` props
- **Category:** efficiency
- **Severity:** High
- **Location:** `apps/web/src/App.jsx:1164-1261,2124`, `apps/web/src/hooks/useModelsAndLoras.js:48-84`
- **Finding:** `refreshData`/`refreshDataWithLoraOverlay` are plain function declarations recreated every App render, and are `useCallback` deps of `deleteModel`/`deleteLora`, which sit in `appContextValue`'s dependency array — so the ~150-key context value (whose memoization is the whole point of sc-4194) is rebuilt with a fresh identity on **every** App render. The sc-4194 stability test covers usePresets/usePersonTracks/useCharacters but not useModelsAndLoras, so nothing catches it.
- **Impact:** Every `useAppContext` consumer re-renders on every App render — theme/accent toggles, preview open/close, notices, each password keystroke — so the whole-tree memoization the codebase explicitly engineered is dead weight.
- **Suggested fix:** Wrap `refreshData`/`refreshDataWithLoraOverlay` in `useCallback` (bodies already work through refs/stable setters), and add useModelsAndLoras/useTraining/useTimelines to `hookStability.test.jsx`.
- **Confidence:** High

---

## Medium findings

#### [F-010] Global 2 GiB body limit lets any JSON endpoint buffer 2 GiB in RAM
- **Category:** security
- **Severity:** Medium
- **Location:** `apps/rust-api/src/lib.rs:142,1023`
- **Finding:** The 2 GiB `DefaultBodyLimit` (sized for streaming multipart asset upload) is applied router-wide, so every JSON route (`POST /jobs`, `/image/jobs`, progress, presets) accepts up to 2 GiB, and `Json`/`ApiJson` buffers the whole body into memory before parsing. Only `retry_job` self-limits (1 MiB, jobs.rs:280).
- **Impact:** An authenticated/loopback-trusted caller (or a runaway client) can drive multi-GiB memory spikes per request; a few concurrent requests OOM the API. With LAN access enabled this is a one-request DoS lever for any password holder.
- **Suggested fix:** Drop the router-wide default to a small JSON cap (2–10 MiB) and attach the large limit per-route to the multipart upload endpoints only (as `/loras/import` and `/models/import` already do).
- **Confidence:** High

#### [F-011] Serialize credential-store read-modify-write to prevent lost updates
- **Category:** security
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/credentials.rs:83-107`
- **Finding:** `CredentialFileStore::set`/`delete` each do an unlocked load→mutate→save of `credentials.json`. Two concurrent calls interleave and one write silently overwrites the other — the one store in the crate that doesn't serialize RMW (every other uses `lock_project_files`/`JobsStore.lock`).
- **Impact:** A saved credential can silently vanish after a concurrent set/delete; the user re-enters a token that "didn't stick" and a worker download fails with an auth error that looks like a token problem.
- **Suggested fix:** Add a `Mutex<()>` (or reuse the striped-lock helper keyed on the file path) held across `load`+`save`.
- **Confidence:** High

#### [F-012] Don't silently treat a corrupt credentials file as empty
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/credentials.rs:62-67`
- **Finding:** `load()` maps both read errors and JSON parse errors to an empty map via `.ok()`; a subsequent `set`/`delete` then persists the empty map, permanently destroying every stored credential on the first write after corruption.
- **Impact:** One malformed byte in `credentials.json` (partial disk, manual edit) silently wipes all download credentials on the next save, with no log and no error.
- **Suggested fix:** Distinguish "file absent" (empty map) from "unreadable/unparseable" — return an error, or `warn!` and refuse destructive saves until it parses.
- **Confidence:** High

#### [F-013] Give `training_store` connections the same `busy_timeout` as every other project.db opener
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/training_store.rs:1278,1383,1418,1492,1518`
- **Finding:** Five sites open `project.db` via raw `Connection::open(...)` with rusqlite's default 0 ms busy timeout, bypassing the two `connect_project_db` helpers (both set `busy_timeout(5000)`). `resolve_asset_source` also skips migrations and queries `assets` directly.
- **Impact:** Under any cross-connection overlap (worker embedding writes racing an API dataset read on the same project.db), these paths fail immediately with `database is locked` instead of queueing — intermittent 500s on the training-dataset surface.
- **Suggested fix:** Route all five through a shared `connect_project_db`.
- **Confidence:** High

#### [F-014] Split the `jobs_store` god module: SQL store vs. routing policy vs. model catalog
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/jobs_store.rs:1-6001`
- **Finding:** One 8.8k-line file mixes the SQLite jobs/workers store, the MLX/candle routing-eligibility engine (~40 per-model predicates), five hard-coded parallel model-catalog lists (`MLX_ROUTED_MODELS`, `CANDLE_ROUTED_MODELS`, `CANDLE_QUANT_LORA_MODELS`, …), and the UI-gating surface. The comments themselves record two shipped routing bugs from missed-list-edits (chroma sc-5576, krea sc-7836).
- **Impact:** Every new model family/tier requires edits scattered across five lists plus a match arm; the "engine wired but router half missed" bug class is a direct product of the layout.
- **Suggested fix:** Extract `routing/` submodules (`catalog.rs` as one per-model capability table, `mlx.rs`/`candle.rs` predicates, `gaps.rs` error builders), leaving `jobs_store.rs` as the SQL store.
- **Confidence:** High

#### [F-015] Deduplicate the payload-parsing helpers between `image_request` and `video_request`
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/image_request.rs:117-218`, `video_request.rs:198-286`
- **Finding:** `string_or`, `optional_id`, `optional_i64`, `string_list`, `array_or_empty`, `object_or_empty`, and the int-clamp helper are copy-pasted — and already drifted: `image_request::string_or` returns the raw value while `video_request::string_or` trims and empty-filters (image_request needs a separate `nonempty_string_or` for the same semantics).
- **Impact:** Same-named helpers with different semantics is exactly the drift that produces subtle payload-parsing mismatches between the image and video lanes.
- **Suggested fix:** Hoist the helpers into a shared `payload_util` with one clearly named trimming variant; both parsers use it.
- **Confidence:** High

#### [F-016] Ideogram auto-caption blocks the image-job POST for up to ~9 minutes
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/rust-api/src/ideogram.rs:25-33,169-256` (called from `generation.rs:27`)
- **Finding:** `create_image_job` synchronously runs `rich_auto_caption_for_ideogram`, which enqueues a `prompt_refine` job and polls every 500 ms with a 180 s cap *per attempt* × 3 attempts — the HTTP request can hang ~9 minutes before the image job is even created.
- **Impact:** Headless/API clients posting plain-text Ideogram prompts hit client/proxy timeouts and perceive the API as dead; the connection is held the whole time and impatient retries stack more refine jobs.
- **Suggested fix:** Create the image job immediately in a `pending_caption` stage and let a background task/worker expand and rewrite the payload before dispatch; or cap total wall time to one attempt and document the latency.
- **Confidence:** High

#### [F-017] Model/LoRA catalogs fully re-assembled (with filesystem probing) several times per job-create request
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `apps/rust-api/src/generation.rs:21-31`, `models.rs:1154-1312`, `loras.rs:227-334`
- **Finding:** One `POST /image/jobs` with a preset triggers `recipe_preset_catalog`, `merge_preset_loras_into_payload`, `resolve_model_manifest_entry`, and `validate_job_lora_compatibility` — each re-running `model_catalog`/`lora_catalog`, whose per-model install-state probes (recursive HF-cache walks, `model_is_installed`, `mlx_catalog_status`) are uncached and run 2–3× per request over the whole catalog.
- **Impact:** Job-create latency scales with catalog size × HF-cache-tree size; adds hundreds of ms to seconds per submit on a large cache and hammers the blocking pool under batch submission.
- **Suggested fix:** Thread one `model_catalog`/`lora_catalog` snapshot through the request, and/or add a short-TTL (1–5 s) cache alongside the existing manifest cache.
- **Confidence:** High

#### [F-018] Silent quant-tier downgrade + wrong quant recorded in the recipe
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/image_jobs/base.rs:390-428` (and `ideogram_model_subdir`, `boogu_model_subdir`, `krea_model_subdir`)
- **Finding:** `standard_tier_subdir` falls through `q4→q8→bf16` when the requested tier isn't downloaded, with no warning/event, while the sidecar records the *requested* quant — a user who selected bf16 with only `q4/` present silently renders Q4 while the asset says dense.
- **Impact:** A/B quant comparisons (the epic 8506 workflow this exists for) can silently compare a tier against itself; asset telemetry lies about precision.
- **Suggested fix:** Error or `warn!` + emit an event when the preferred tier is absent, and derive `quant_bits` from the tier dir actually chosen.
- **Confidence:** High

#### [F-019] Confine payload-supplied `controlWeights.filename` before joining it into weight paths
- **Category:** security
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/image_jobs/zimage.rs:41-65` (also `qwen.rs`, `flux2.rs:676`, `flux1_control.rs:94`, `pid.rs:142` `pidCheckpoint.filename`)
- **Finding:** `resolve_control_weights_for` takes `advanced.controlWeights.filename` from the payload and does `snapshot.join(filename)` with only an existence check — a `../../…` filename escapes the HF snapshot and loads an arbitrary local file as control weights. LoRA/modelPath are confined; this key is the gap. (Verified at zimage.rs:62.)
- **Impact:** A crafted payload (reachable via the epic-4484 LAN API) points weight loading at any readable file, bypassing the confinement the same crate enforces for LoRA paths.
- **Suggested fix:** Reject filenames with path separators/`..` (or route through `resolve_app_managed_*`) in every `*_control_repo_file` and `resolve_pid_weights`.
- **Confidence:** High

#### [F-020] Base Z-Image strict-control lane dropped identity-likeness scoring
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/image_jobs/zimage.rs:475-603`
- **Finding:** Copy-paste drift: `generate_zimage_base_control_stream` (sc-8251) is documented as "differing only in engine id, checkpoint, and REAL CFG" from the Turbo stream but uses `drive_gen_items` and omits the entire sc-4410 likeness block that Turbo, flux2, qwen, flux1, and all four candle control lanes carry.
- **Impact:** A Character-Studio pose set on base `z_image` with an identity reference gets no `faceLikeness` blocks — a silent feature gap vs every sibling lane, invisible until someone compares sidecars.
- **Suggested fix:** Add the likeness-source/face-stack/scorer wiring and switch to `drive_gen_items_scored`, or fold both Z-Image streams into one parametrized stream.
- **Confidence:** High

#### [F-021] Candle Kolors control lane silently ignores `controlMode`
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/image_jobs/kolors_control.rs:195-374`
- **Finding:** The sc-8304 `CandleStrictControl` driver collapsed the qwen/zimage/flux2/flux1 candle lanes, but `generate_candle_kolors_control_stream` still carries the pre-driver scaffold and never calls `requested_control_kind`/`validate_control_kind` — a job with `advanced.controlMode:"canny"` renders a pose skeleton anyway instead of being rejected.
- **Impact:** Silent wrong-conditioning output on the Kolors candle lane, plus ~180 lines of duplicated scaffold that keeps drifting.
- **Suggested fix:** Route through `run_candle_strict_control` with a pose-only `supported_kinds` row so canny/depth are *rejected*, or at minimum add an explicit pose-only rejection.
- **Confidence:** High

#### [F-022] Fit/letterbox geometry reimplemented per candle edit lane, with `contain_box` drift
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/image_jobs/sdxl_edit_candle.rs:148-211`, `flux2_edit_candle.rs:182-227` (vs shared `base.rs:12-72`)
- **Finding:** `sdxl_edit_fit_*`/`flux2_edit_fit_*` duplicate the shared `fit_rgb`/`fit_engine_image` (already compiled on the candle lane, already used by `zimage_edit_candle.rs:156`) — and the copies differ: the twins use `gen_core::imageops::contain_box` for pad while base.rs `fit_rgb` uses a local `contain_box` with `.round()` semantics, so the kept-rect can differ by a pixel, and the outpaint mask helper relies on the two agreeing.
- **Impact:** One-pixel mask/canvas misalignment risk on outpaint edges, plus three implementations to patch for any fit bug.
- **Suggested fix:** Delete the per-lane twins, call shared `fit_engine_image`, and make base.rs `fit_rgb`'s pad arm use `gen_core::imageops::contain_box`.
- **Confidence:** Medium (duplication certain; the rounding divergence needs a quick check of `contain_box`).

#### [F-023] `steps`/`guidance` resolver cloned ~9 times across bespoke lanes with drifting clamp bounds
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/image_jobs/{sdxl_edit_candle.rs:108,sdxl_ipadapter.rs:86,kolors_ipadapter.rs:88,kolors_control.rs:74,qwen_control.rs:87,flux2_edit_candle.rs:117,instantid.rs:194,pulid.rs:96,zimage_edit_candle.rs:75}`
- **Finding:** "advanced.steps → manifest → per-lane default, clamped" and the guidance equivalent are re-implemented per file, each with its own inline parse closure; the clamp bounds already drift (1..=80 vs 1..=50 vs 1..=100) with no justifying comment.
- **Impact:** Steady drift in clamp ranges and parse behavior; every new lane adds two more copies.
- **Suggested fix:** Add `resolve_advanced_or_manifest_u32/_f32(request, key, default, range)` helpers next to `resolve_steps`.
- **Confidence:** High

#### [F-024] ~90-line grouping + likeness-gating block triplicated between FLUX.2, Qwen, and SenseNova edit streams
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/image_jobs/flux2.rs:382-472`, `qwen.rs:653-750`, `sensenova.rs:205-274`
- **Finding:** The `(seeds, prompts, pose_inputs)` grouping match, angleSet/poseLibrary raw-settings stamping, and the `character_set`/`plain_with_character`/`score_likeness`/face-stack block are near-verbatim triplicated (comments included).
- **Impact:** The highest-traffic scoring/gating logic in the layer; a fix to the plain-with-character gate (sc-4411) must be re-applied three times with three chances to drift.
- **Suggested fix:** Extract a `plan_edit_batch(request, grouping) -> EditBatch` builder plus a shared `stage_likeness(...)` helper (the warn-and-None staging block alone appears 10+ times).
- **Confidence:** High

#### [F-025] PuLID/LTX/Gemma weight seams mutate process-global env vars from async code
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/image_jobs/pulid.rs:204-234,348-353`; `video_jobs.rs:3036-3043,4853-4860`
- **Finding:** `set_pulid_weight_env` and `ensure_ltx_gemma_dir`/`ensure_bundled_ltx_gemma_dir` call `std::env::set_var`/`remove_var` on the multithreaded tokio runtime at job time. `set_var` is unsound in a multithreaded program (UB on POSIX if another thread reads env — reqwest/DNS, `HF_HOME` reads, linked C/ObjC), the RAII restore races the spawned stream task by design, and `set_var` is `unsafe` in Rust 2024.
- **Impact:** Latent crash/UB risk during PuLID/video jobs; also blocks an edition bump and makes engine load order-dependent on scheduling.
- **Suggested fix:** Pass the paths on the `LoadSpec`/engine config instead of env, or set the vars once at process startup before the runtime spawns threads.
- **Confidence:** Medium (unsoundness real; concurrent-reader-in-practice unconfirmed).

#### [F-026] `run_video_generate_job` / `run_image_generate_job` / `generate_stream` are cfg-laden dispatch god functions
- **Category:** readability
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/video_jobs.rs:152-589`, `image_jobs.rs:334-1001`, `image_jobs/base.rs:1918-2290`
- **Finding:** Each mixes validation, multi-arm cfg-gated engine-dispatch ladders (the macOS video ladder alone is ~215 lines of 4-tuple returns), and result assembly; `generate_stream`'s 5-armed `(identity_init, flux_ip_dir, flux_true_cfg)` match embeds per-family conditioning policy inline (already special-cased five ways).
- **Impact:** Adding a family means editing the middle of two/three god functions; the inline per-family conditioning is the classic place the next drift bug lands. A misindented block already exists at video_jobs.rs:2784-2812.
- **Suggested fix:** Table-ize dispatch (predicate → handler fn pointers, like the macOS `ImageRoute` enum) and extract `resolve_generic_lane_conditioning(...) -> LaneConditioning`.
- **Confidence:** High

#### [F-027] Bound memory in the SeedVR2 video-upscale path (whole clip decoded into RAM)
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/video_jobs.rs:1237-1295,1406-1445`
- **Finding:** `decode_seedvr2_source_frames` decodes every source frame (`-fps_mode passthrough`, no cap) into an in-memory `Vec<Image>`, and the up-to-4× output is also held fully in memory before encoding; unlike the generate path, `VideoUpscaleRequest` imposes no frame/duration bound.
- **Impact:** A few-minute 1080p clip allocates tens of GB of RGB8 (≈17 GB source + ~68 GB at 4×) → OOM-killed after minutes of GPU work.
- **Suggested fix:** Enforce a frame/duration cap before decode, or stream the upscale in temporal chunks (decode→upscale→append-encode).
- **Confidence:** Medium (worker-side absence of a cap confirmed; an API-side pre-validation could mitigate).

#### [F-028] Deduplicate the MLX/candle sibling adapter/conditioning/dispatch functions in `video_jobs`
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/video_jobs.rs:1931-1993,3326-3431,3439-3558 vs 4374-4487,3624-3724 vs 4557-4660,3870-3959 vs 6653-6756`
- **Finding:** `resolve_wan_vace_adapters`/`resolve_scail2_adapters` are byte-identical; `candle_resolve_lora_file` re-implements `resolve_lora_file` (inlining `first_safetensors_path`, which exists in core); the SCAIL-2 conditioning/extend/bridge functions each have ~100-line candle twins differing only in the segmenter module path; `resolve_bernini_quant == resolve_scail2_quant`.
- **Impact:** Any fix to LoRA confinement, lightning detection, or SAM3 mask painting must land in 2–3 places; the twins have already drifted.
- **Suggested fix:** One shared adapter resolver parameterized by the max-LoRA constant; one SCAIL-2 conditioning orchestrator taking a segmenter closure; a `non_empty_negative_prompt` helper.
- **Confidence:** High

#### [F-029] Ineffective upper clamp on person-detect `sourceTimestamp` (`duration.max(3600.0)`)
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/media_jobs.rs:376-381`
- **Finding:** `payload_f64(...,"sourceTimestamp",...).clamp(0.0, duration.max(3600.0))` uses `max`, so the upper bound is always ≥3600 and never the clip duration; `duration.min(3600.0)` was meant.
- **Impact:** A timestamp past the end of a short clip passes validation, `ffmpeg -ss` yields no frame, and the job fails with a generic "no frame output" instead of a payload-validation error.
- **Suggested fix:** `.clamp(0.0, duration.min(3600.0))`.
- **Confidence:** High

#### [F-030] `render_item_segment -t` truncates slow-motion segments, desyncing crossfades
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/media_jobs.rs:2266-2289`
- **Finding:** The video branch passes `-t {source_duration}` as an output option while `setpts={1/speed}*PTS` rescales timestamps; for `speed < 1.0` the stretched output is truncated at `source_duration`, shorter than the `duration` the function returns, and `crossfade_filter_complex` computes xfade offsets from the declared durations.
- **Impact:** One truncated slow-mo segment desynchronizes every subsequent transition in the exported MP4.
- **Suggested fix:** Use `-t {duration:.3}` (the timeline length) after the setpts filter, or trim input-side before `-i`.
- **Confidence:** Medium (arg ordering verified; a speed=0.5 run would confirm).

#### [F-031] macOS/candle `segment_assembly_frames` and `assemble_real_person_track` are ~120-line near-duplicates
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/media_jobs.rs:856-1016 vs 1049-1173`, `703-835 vs 1182-1314`
- **Finding:** The candle variants duplicate the macOS variants line-for-line (span computation, clip-path/anchor assembly, degraded/missing rollup, mask-PNG dispatch; sampling loop, observe/assemble, work-dir lifecycle), differing only in the segmenter/device seam.
- **Impact:** Fixes like the `frame_paths.len() <= last` guard, the `p > 127` threshold, or the temp-dir-leak fix must land twice; drift already visible.
- **Suggested fix:** One cfg-free orchestrator taking a segmenter-backend closure/trait.
- **Confidence:** High

#### [F-032] Caption job abandons the running captioner on progress/heartbeat errors
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/caption_jobs.rs:225-285`
- **Finding:** `update_job(...).await?`/`heartbeat(...).await?` exit without `cancel.cancel()` or awaiting `blocking`; the detached JoyCaption task keeps the model loaded and generating. (Instance of the F-003 class, called out separately as it's a distinct handler.)
- **Impact:** On a transient API error the worker may claim another GPU job while a caption generation still occupies unified memory (sc-8390 contention).
- **Suggested fix:** Cancel-and-join on every error exit via the shared guard.
- **Confidence:** High

#### [F-033] Trainer/analysis consumers abandon GPU work on API error (High-class instances)
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/dataset_analysis_jobs.rs:212-248`, `face_analysis_jobs.rs:286-322`
- **Finding:** Both analysis `select!` loops propagate `update_job`/`heartbeat` errors with `?` without tripping `cancel` or joining `blocking`; the blocking thread keeps embedding the whole dataset. (Grouped with F-003 but Medium here because analysis jobs are shorter-lived than training.)
- **Impact:** Abandoned analysis keeps the CLIP/face stack on the GPU while the worker claims the next job.
- **Suggested fix:** Same drop-guard, ideally via one shared consumer helper.
- **Confidence:** High

#### [F-034] Near-duplicate job scaffolding across the two analysis modules (and training)
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/dataset_analysis_jobs.rs:78-457 vs face_analysis_jobs.rs:219-525`
- **Finding:** Structural clones: item parsing (~80 identical lines each), byte-identical image loaders, three identical `*_progress` constructors, the `0.12 + 0.78 * …` loop, the sidecar POST fold, and the platform stub. Drift visible (a needless `items.clone()` in dataset_analysis where face_analysis moves; divergent cancel handling).
- **Impact:** The abandon-without-cancel bug must now be fixed in three places; each new analysis job re-stamps ~400 lines.
- **Suggested fix:** Extract a shared `run_batched_analysis_job` scaffold parameterized by endpoint/space/per-item embed fn.
- **Confidence:** High

#### [F-035] Kill the `hf` CLI / ffmpeg child on all error exits; fix `finalize_converted_dir` destroy-before-rename
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/model_jobs.rs:1294-1320,1219-1232`
- **Finding:** `download_model_with_hf_cli` spawns without `kill_on_drop(true)` and a `heartbeat(...).await?` failure returns while the child runs; and `finalize_converted_dir` `remove_dir_all(final_dir)` **before** renaming, so a rename failure (or crash between) loses the previously working model — contradicting its own "on error the final location is left untouched" doc.
- **Impact:** An API hiccup orphans a multi-GB `hf download`; re-converting an installed model can leave the user with no model at all after minutes of work.
- **Suggested fix:** `kill_on_drop(true)` (or kill on every error arm); rename stale aside → rename temp→final → delete stale, restoring on failure.
- **Confidence:** High

#### [F-036] Move interleave/sensenova PNG encoding + document writes off the async runtime
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/sensenova_jobs.rs:650-668,899-917,957-1094`
- **Finding:** `write_interleaved_document` (sync PNG encodes of up to 10 multi-megapixel images + fs writes) is called directly from the async handlers on a tokio worker thread.
- **Impact:** Multi-second async-runtime stalls — the class sc-8390 keepalive work exists to prevent.
- **Suggested fix:** Wrap the call in `spawn_blocking`.
- **Confidence:** High

#### [F-037] Deduplicate the MLX/candle sibling VQA and interleave handlers
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/sensenova_jobs.rs:88-246 vs 253-409; 435-684 vs 692-933`
- **Finding:** `run_vqa_job` and `run_interleave_job` are duplicated near line-for-line per backend (~150 and ~240 lines), differing only in load/preprocess/decode/cancel-arg shapes; drift already visible (`width, height` vs `width as usize`, cancel-arg shape).
- **Impact:** The purest instance of the engine-duplication theme.
- **Suggested fix:** Shared async driver + a small backend trait/closure for load/preprocess/generate/decode.
- **Confidence:** High

#### [F-038] Coalesce per-token progress posts in prompt refine; cache the refine model
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/prompt_refine_jobs.rs:755,857-899,760-869`
- **Finding:** Every generated token sends on a bounded(64) channel and triggers a full `update_job` POST — thousands of sequential POSTs per 4096-token caption, with `blocking_send` back-pressuring generation on API latency; and each refine/caption/describe job cold-loads the ~16 GB snapshot via `load_for_model_with` with no provider cache (unlike image/video lanes' `generator_cache`).
- **Impact:** Sustained progress spam against the local API, generation latency coupled to API latency (worse over epic-4484 LAN), and a multi-second cold load on every interactive refine click.
- **Suggested fix:** `try_send` from the token callback / post only the latest snapshot per tick; cache the loaded `TextLlm` keyed by weights dir with idle eviction.
- **Confidence:** High (spam); Medium (model cache may be a deliberate memory decision).

#### [F-039] Fingerprint adapter/weight content in the generator-cache key, not just paths
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/generator_cache.rs:29-84,120-143`
- **Finding:** `GeneratorCacheKey` identifies weights/adapters by path + scale only; a file replaced at the same path (re-imported LoRA, re-converted checkpoint) is served from cache with the old tensors until the 300 s idle timeout.
- **Impact:** Re-import a LoRA under the same name and regenerate within 5 minutes → silently the stale adapter ("my new LoRA does nothing" reports that self-heal before debugging).
- **Suggested fix:** Include file size + mtime of each adapter/weights file in the cache key, or expose an explicit evict hook the import/convert jobs call.
- **Confidence:** Medium (staleness window verified; overwrite frequency unverified).

#### [F-040] Confine dataset-upscale `imagePath` reads and sanitize `itemId`/`datasetId` write paths
- **Category:** security
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/upscale_jobs.rs:1082-1103,1174,1195-1219`
- **Finding:** `run_dataset_upscale_job` decodes `items[].imagePath` verbatim (no confinement, despite `resolve_dataset_item_path` existing) and writes `project_path.join("training/datasets/{dataset_id}/upscaled/{item_id}.png")` with both ids taken verbatim (only non-empty checked). (Verified at L1213.) An id with `/` or `..` traverses out; `save_with_format` overwrites.
- **Impact:** A crafted `dataset_upscale` payload gains an arbitrary-image **read**/exfil and an arbitrary-location file **write** (PNG bytes) under worker privileges — sharper once epic 4484 exposes the API.
- **Suggested fix:** Resolve reads via `resolve_dataset_item_path`; route the output through `safe_project_path` (rejects non-`Normal` components).
- **Confidence:** High

#### [F-041] Serialize concurrent manifest upserts across the utility-worker pool
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/manifest.rs:13-64` (callers `model_jobs.rs:1760,2092`)
- **Finding:** `upsert_manifest_entry` is an unlocked read-modify-write of `user.loras.jsonc`/`user.models.jsonc`, but the default utility pool is 4 separate processes, so two concurrent install jobs interleave and one entry's upsert overwrites the other's.
- **Impact:** Lost manifest entries after parallel installs — a freshly downloaded LoRA/model vanishes from the manifest and shows "not installed" despite the completed job.
- **Suggested fix:** Advisory lock (`fd-lock` on a `.lock` sibling) around read→merge→rename, or route manifest writes through the API.
- **Confidence:** Medium (race window verified in code; needs two simultaneous jobs to manifest).

#### [F-042] Abort/await the running task when the keepalive loop errors out
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/progress.rs:100-121` (same pattern `upscale_jobs.rs:696-728`)
- **Finding:** In `run_blocking_with_heartbeat`/`run_upscale_with_heartbeat`, a heartbeat/`update_job` error in the interval arm propagates via `?`, dropping the `JoinHandle` — which does not stop the task; the blocking GPU compute keeps running detached. (This is the shared helper underlying the F-003 class.)
- **Impact:** After a 409 the worker can run two GPU workloads concurrently — memory pressure/OOM on the unified-memory budget.
- **Suggested fix:** On the error path trip the `CancelFlag` and `task.abort()`/await with a timeout before returning; at minimum log that a detached compute is draining.
- **Confidence:** High

#### [F-043] Don't drop the in-flight job future on shutdown without a job-state write
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/lib.rs:480-521`
- **Finding:** `run_worker_loop`'s `select!` races the whole `poll_once` (claim + full job execution) against `shutdown_signal()`. On SIGTERM/Ctrl-C mid-job the future is cancelled at an arbitrary await point, an `Offline` heartbeat is posted, and the process exits — the claimed job sits `running` until the 90 s stale sweep marks it `interrupted`, and any `spawn_blocking` GPU work is killed mid-write.
- **Impact:** Every graceful desktop quit during a job produces a delayed, generic `interrupted` instead of a prompt terminal state; partial outputs can be left behind.
- **Suggested fix:** Observe shutdown between jobs (select only around claim/sleep); on shutdown-during-job trip cancel and post a terminal `Canceled` before returning.
- **Confidence:** High (mechanism); Medium (operational severity).

#### [F-044] Cache or skip the per-click SAM3 model rebuild + quantize in smart-select
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/person_segment_sam3.rs:410-416,539-545`
- **Finding:** `segment_box_blocking`/`segment_points_blocking` build a fresh `Sam3ImageSegmenter`/`Sam3Tracker` from the cached dense weights and re-run `quantize(8)` on every invocation; the fresh-per-call rationale for the *video* model (per-session tracking state) doesn't obviously apply to the single-image paths.
- **Impact:** Seconds of added latency per interactive smart-select click, plus transient memory spikes (dense 3.2 GB resident while a quantized copy is materialized per call).
- **Suggested fix:** Cache the quantized segmenter/tracker process-wide if stateless, else cache the quantized weight map.
- **Confidence:** Medium (needs mlx-gen-sam3 confirmation the image paths are stateless).

#### [F-045] Deduplicate the MLX/candle SAM3 modules' shared pure helpers
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/person_segment_sam3_candle.rs:86-248,354-490` (vs `person_segment_sam3.rs`)
- **Finding:** ~300 lines are duplicated verbatim (self-described "Shared verbatim with the MLX module") between the two cfg-exclusive files: `resolve_segmenter_weights`, `ensure_segmenter_weights`, `normalize_chw`, `mask_box_containment`, `select_object`, `mask_to_frame`, `mask_centroid_x`, `AllPersonMasks`, plus duplicated tests; `rollup_mask_state` is triplicated.
- **Impact:** Mask/association math is the correctness core of person-replace; fixes must be applied twice or the platforms silently diverge.
- **Suggested fix:** Extract a backend-neutral `person_segment_sam3_common` compiled on both cfg lanes; keep only the tensor/model seam per backend.
- **Confidence:** High

#### [F-046] Move skeleton rendering + preview PNG writes off the async thread and under heartbeat coverage
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/pose_jobs.rs:897-975`
- **Finding:** After `run_blocking_with_heartbeat` returns, the handler loops over every source × person doing CPU raster (`draw_wholebody` on a `max(w,h)²` canvas — ~108 MB RGB per person at 6000²) plus synchronous `skeleton.save()` PNG encodes, inline in the async fn with no heartbeat arm running.
- **Impact:** Blocks a tokio runtime thread for the whole render loop; large multi-person high-res batches grow the silent window toward the 90 s sweep — the failure class sc-8390 fixed for the inference half.
- **Suggested fix:** Fold the conversion/render/save loop into the `spawn_blocking` batch task (or a second keepalive call).
- **Confidence:** High that it blocks the runtime thread; Medium that realistic batches reach the sweep.

#### [F-047] Debounce the Logs search — it refetches a 1000-row snapshot per keystroke
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `apps/web/src/screens/LogsScreen.jsx:64-100,155-162`
- **Finding:** `search` feeds `loadSnapshot`'s `useCallback` deps and `useEffect(...,[loadSnapshot])` runs it, so every keystroke issues a fresh `limit:1000` fetch plus re-arms the 2s poll with the partial term.
- **Impact:** A 10-char filter fires ~10 full 1000-row fetches; on the remote-LAN path that's real network churn, and interleaved responses can briefly show stale-prefix results.
- **Suggested fix:** Debounce the input (~250 ms) before it reaches the fetch deps, and/or filter client-side over the already-held `entries`.
- **Confidence:** High

#### [F-048] Purge in-flight AI-op scratch/result assets when ImageEditor unmounts
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/web/src/screens/ImageEditor.jsx:2281-2342,2532-2575,2637-2653`
- **Finding:** `runAiOp` imports the working bitmap as a real project "scratch" asset and relies on a completion effect in this component to purge scratch + result, but the leave guard is registered only when `dirty` (starting an AI op doesn't set `dirty`), and App mounts screens only while active. Navigating away mid-job unmounts the editor; the effect never fires.
- **Impact:** The edit result is silently lost and the scratch upload + result asset permanently land in the Library as orphans; repeated occurrences accumulate junk assets and disk.
- **Suggested fix:** Register the leave guard while `aiOp` is non-null too, and move scratch/result purge to something that survives unmount (App-level watcher on a `scratch:true` marker, or a completed-job scratch sweep).
- **Confidence:** High

#### [F-049] Stop resetting the user's picked character reference on every characters refetch
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/web/src/screens/characterPanels.jsx:543-545`
- **Finding:** `useEffect(() => setReferenceAssetId(approvedReferences[0]?.assetId ?? ""), [characterId, approvedReferences])` unconditionally snaps back to reference #1; `approvedReferences` gets a new identity on every character mutation (useCharacters replaces the object), so the panel's own upload flow (which sets the new id first) is immediately undone.
- **Impact:** Uploading a reference in the Angle/Pose panels selects it then deselects it; picking thumbnail N is undone on refresh, so users generate against the wrong identity reference.
- **Suggested fix:** Mirror ImageStudio's guard (`ImageStudio.jsx:781-789`): keep the current id when it's still in `approvedReferences`, fall back to `[0]` only when invalid.
- **Confidence:** High

#### [F-050] Serialize ImageEditor undo/redo — rapid ⌘Z corrupts the redo stack
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/web/src/screens/ImageEditor.jsx:1130-1217`
- **Finding:** `undo()` pops the history ref synchronously but restores asynchronously (`await blobToImage` per layer); a second key-repeat undo before the first restore completes captures the stale `workingRef` as "present", pushing a duplicate onto `future` and racing the two `restoreSnapshot`→`setWorking` calls.
- **Impact:** Holding ⌘Z on large-layer sessions yields a redo chain replaying the wrong states — an intermediate edit becomes unreachable.
- **Suggested fix:** Gate undo/redo behind an `isRestoringRef` (ignore while a restore is in flight) or queue them.
- **Confidence:** Medium

#### [F-051] Unify the four job→result-asset resolvers and the two upscale-engine tables
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/web/src/screens/ImageStudio.jsx:227-247,160-172`, `VideoStudio.jsx:48-63`, `QueueScreen.jsx:307-323`, `characterPanels.jsx:34-43`, `imageJobs.js:74-86`
- **Finding:** Four near-identical "resolve a job's produced assets against the live catalog" implementations (only the image one has batch-index ordering), and the `UPSCALE_ENGINES` table exists twice with already-drifted shapes (`imageJobs.js` keys on `key`, ImageStudio on `id`) plus a copy-pasted stale-selection fallback effect.
- **Impact:** A worker result-contract change (batch-slot ordering already only in one copy) must be fixed in four places; adding/gating an upscale engine must be done twice.
- **Suggested fix:** One `resolveJobResultAssets(job, assets, {type})` in a shared module; export the single upscale table + `useUpscaleEngineFallback` hook from `imageJobs.js`.
- **Confidence:** High

#### [F-052] `App` / `ImageEditor` / `ImageStudio` god components
- **Category:** readability
- **Severity:** Medium
- **Location:** `apps/web/src/App.jsx:463-2392`, `screens/ImageEditor.jsx:752-3955`, `screens/ImageStudio.jsx:276-1600`
- **Finding:** `App()` owns ~40 hooks, the SSE client, auth, theme, preview nav, five poll loops, a dozen job callbacks, and a 150-entry context literal with a hand-mirrored dep array (the exact place the F-009 staleness bug hid). `ImageEditor` is a ~3,200-line component with ~45 hooks and eight tools plus a ref-mirror pattern (`editsRef`/`boxesRef`/…) that exists because the state is too entangled for normal closures. `ImageStudio`'s `submit()` is a 240-line function building `advanced` from 15 conditional spreads.
- **Impact:** High onboarding/review cost; the hand-maintained memo dep array is where silent mismatches land; the `advanced` payload is the app's highest-drift surface.
- **Suggested fix:** Continue the sc-1651/sc-4196 extraction: `useJobEvents`/`useAccessGate` hooks; per-tool ImageEditor modules (mask, boxes, color-grade); a pure `buildImageJobAdvanced(state)`; split AppContext into high-churn vs low-churn.
- **Confidence:** High

#### [F-053] Monolithic AppContext re-renders every consumer on any data tick
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `apps/web/src/context/AppContext.js:7`, `App.jsx:1963-2137`
- **Finding:** A single context carries jobs, workers, assets, models, presets, characters, training, timelines, and all actions; even with F-009 fixed, `jobs`/`workersById` change on every SSE tick, giving the context a new identity and re-rendering every consumer (incl. Settings/Licenses/pose pickers that read only static actions).
- **Impact:** Whole-tree re-render per worker/queue/job event; `React.memo` on screens is ineffective because the context read invalidates them.
- **Suggested fix:** Split into a high-churn context (jobs/workers/queue) and a low-churn one (actions/catalogs/project), or adopt `use-context-selector` for the hot fields.
- **Confidence:** High

#### [F-054] Five copy-pasted poll-to-completion loops in App.jsx
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/web/src/App.jsx:1421-1605` (refinePrompt/magicPrompt/imageCaption/imageDescribe/compareFaceLikeness)
- **Finding:** Five callbacks repeat the identical ~30-line "POST job, then `while (Date.now() < deadline)` of `abortableDelay(1000)` + `GET /jobs/:id`" pattern, varying only in endpoint/deadline/result field — despite the open SSE `job.updated` stream already carrying the same terminal transitions (the `interrupted` status was already added to each copy by hand).
- **Impact:** ~150 lines of drift-prone duplication; any fix (backoff, cancellation edge, new terminal status) must be applied five times.
- **Suggested fix:** Extract one `pollJobToCompletion({createPath, body, deadlineMs, resolveResult, signal})`, or resolve from the SSE stream keyed on jobId with a poll fallback.
- **Confidence:** High

#### [F-055] `assetMatchesCharacter` logic triplicated, already drifting
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/web/src/components/AssetPicker.jsx:156-178`, `DatasetAddDialog.jsx:10-21`, `assetPanels.jsx:212-224`
- **Finding:** The "does this asset belong to this character" predicate exists three times: AssetPicker checks `approvedReferences` **and** `references`; the other two check only `references`. The recipe/metadata checks are byte-identical.
- **Impact:** Already diverged — an asset only in `approvedReferences` matches in the Image-Edit source picker but not the dataset dialog or detail-panel linker; future membership-shape changes must be found in three files.
- **Suggested fix:** One exported predicate (superset semantics) imported in all three.
- **Confidence:** High

#### [F-056] SSE-triggered project-scoped refreshes lack the stale-response guard, racing project switches
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/web/src/App.jsx:1227-1241`, `hooks/useCharacters.js:16-31`, `useModelsAndLoras.js:33-46`, `usePersonTracks.js:14-29`
- **Finding:** The SSE `job.updated` handler calls `refreshAssetsRef.current?.(job.projectId)` (and characters/loras/person-tracks) with no post-await check that the project is still active; `refreshTimelines` guards exactly this (`activeProjectRef.current.id !== projectId → return`), proving the hazard is known, but the other four setters commit unconditionally.
- **Impact:** A job completing for project A while its refresh is in flight and the user switches to B can overwrite B's `assets`/`characters`/`personTracks` with A's data.
- **Suggested fix:** Add the `activeProjectRef.current?.id === projectId` check before each `setX(items)`.
- **Confidence:** High

#### [F-057] User-pose preview URLs skip `API_BASE_URL`, breaking on split-origin deployments
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/web/src/poseLibrary.js:55-69` (consumer `PoseLibraryPicker.jsx:800`)
- **Finding:** `poseAssetToRecord` sets `previewUrl: asset.url ?? (asset.file?.path ? `/${asset.file.path}` : undefined)`; `asset.url` is the API-relative path every other consumer prefixes with `API_BASE_URL`, and the `/${asset.file.path}` fallback omits the `/api/v1/projects/:id/files/` route prefix entirely.
- **Impact:** In Vite dev and any split-origin deployment (`VITE_API_BASE_URL` set), user pose thumbnails 404; the raw-path fallback is broken on every deployment. Only the embedded same-origin desktop build renders them.
- **Suggested fix:** Build the preview through `assetUrl(asset)`.
- **Confidence:** High

#### [F-058] Jobs state grows unbounded and is fully re-sorted on every SSE event
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `apps/web/src/App.jsx:130-142,1047`; also `hooks/useTraining.js:201`, `useModelsAndLoras.js:108`
- **Finding:** Every `job.updated` does `[job, ...items.filter(...)].sort(sortNewest)` over the entire array, and nothing prunes terminal jobs (`mergeFreshJobs` deliberately keeps client-side entries the server no longer returns).
- **Impact:** In a long session the array grows monotonically; each frequent progress tick costs an O(n log n) copy-sort plus a full App render — the app's steady-state render tax with F-009/F-053.
- **Suggested fix:** Cap retained terminal jobs (newest N terminal + all active) and replace sort-on-insert with `createdAt` insertion.
- **Confidence:** High

#### [F-059] Retired Python worker's e2e test client + embedded live manifest audits will break epic-8283 deletion
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `tests/test_rust_api_worker_smoke.py:15-29,135-422`; `tests/test_worker_image_adapters.py:943-1119,3677-3722`
- **Finding:** Three of the five e2e tests (the only coverage of the live Rust API's worker protocol / procedural pipeline / sc-2226 LoRA boundary) drive the API using `scene_worker.runtime.ApiClient`; and structural audits of the live `config/manifests/builtin.models.jsonc` live inside the retired-worker test file. Deleting `apps/worker` takes both down.
- **Impact:** The e2e gate for the live API's claim/heartbeat/cancel/asset-writing contract and live catalog-config gates are silently lost if the worker tests are deleted rather than reimplemented.
- **Suggested fix:** Before/with the 8283 deletion, re-express the three e2e tests as pure-HTTP clients and extract the manifest audits into a standalone `tests/test_builtin_manifest_audit.py`; file stories linked to epic 8283.
- **Confidence:** High

#### [F-060] `UPDATE_SNAPSHOTS=1` on a filtered run clobbers all untouched snapshots
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `tests/test_rust_api_contract_snapshots.py:24-66,400-414`
- **Finding:** `write_updated_snapshots` (atexit) rewrites `snapshots.json` wholesale from `UPDATED_SNAPSHOTS`, which only accumulates labels asserted during that run; `UPDATE_SNAPSHOTS=1 pytest -k character` silently drops every non-character snapshot.
- **Impact:** Regenerating one drifted snapshot with `-k` deletes the rest — surfaces as confusing "Missing snapshot" failures on the next full run, or gets committed.
- **Suggested fix:** Merge into the existing file: `merged = snapshots(); merged.update(UPDATED_SNAPSHOTS); write merged`.
- **Confidence:** High

#### [F-061] Nine training/lycoris tests permanently skip in CI; 84% of the CI Python suite tests retired code
- **Category:** dead-code
- **Severity:** Medium
- **Location:** `tests/test_worker_training.py:170,192,416,455,506,514,528`, `tests/test_lycoris_lokr.py:165-166,241-242`; broadly `tests/test_worker_*.py`, `worker_runtime_shared.py`
- **Finding:** Those tests `importorskip("torch"/"lycoris")` but the only pytest CI lane never installs torch/lycoris, so they skip every run; more broadly 18 of 21 test files import `scene_worker` and ~540 of ~558 tests validate the retired Python worker.
- **Impact:** Perpetually-skipped tests inflate the suite and imply coverage that doesn't run; the retired-worker tests are ~12.5k LOC of CI time gating code that can't regress in production.
- **Suggested fix:** Fold deletion into epic 8283 after extracting the live gates (F-059).
- **Confidence:** High

#### [F-062] Restrict release secrets from job-wide env; bind the Docker web dev server to loopback; verify staged wheel hashes
- **Category:** security
- **Severity:** Medium
- **Location:** `.github/workflows/release.yml:46-56,149-156`; `docker-compose.yml:151-152`; `apps/desktop/scripts/stage-ffmpeg.py:46-72`, `stage-onnxruntime.py:36-58`
- **Finding:** Three distinct release/deploy-surface gaps: (a) `TAURI_SIGNING_*`/`APPLE_*` secrets are set at job-level `env`, visible to `npm ci` (postinstall scripts) and every crate `build.rs`, not just the Tauri bundle step; (b) the `web` compose service publishes `"${SCENEWORKS_WEB_PORT:-5173}:5173"` with no host-IP prefix (0.0.0.0) while `api` correctly defaults to `127.0.0.1`, exposing an unauthenticated Vite dev server (with `/@fs/` file serving) to the LAN; (c) the ffmpeg/onnxruntime binaries that get codesigned+notarized are pip-downloaded by version pin only, no sha256 — unlike the Windows first-run downloader which pins URL+sha256.
- **Impact:** A compromised npm/crates dependency can read the updater signing key; a default `docker compose up` LAN-exposes a dev server; a PyPI/CDN compromise flows a malicious binary into a signed release.
- **Suggested fix:** Move signing secrets to the specific bundle steps; publish web as `127.0.0.1:${PORT}:5173`; pin wheel URL+sha256 (the `cuda_provision.rs` pattern).
- **Confidence:** High

#### [F-063] Fix the version drift between root, web, and desktop packages
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `package.json:4` / `apps/web/package.json:4` (0.4.0) vs `apps/desktop/tauri.conf.json:4` / `apps/desktop/package.json:4` (0.5.1); `scripts/sync-version.mjs:20-51`
- **Finding:** `sync-version.mjs`'s contract is that one root `npm version` bumps everything atomically, but the tree is already skewed (root/web 0.4.0, shipped desktop 0.5.1) and nothing in CI asserts the invariant.
- **Impact:** The next `npm version patch` at root produces 0.4.1 — a tag/version *below* what users run — so the auto-updater (comparing against `latest.json`'s version) serves no update and release bookkeeping diverges.
- **Suggested fix:** Re-align root+web to 0.5.1 now, and add a CI assert (in `check-scaffold.mjs`) that all four version strings match.
- **Confidence:** High

#### [F-064] Consolidate the copy-pasted smoke-harness helpers into a shared test-support module
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/{chroma1_base_q4,lens_base_q4,lens_turbo_q4,sdxl_base_q8}_mlx_smoke.rs`, `{krea_turbo,sd3_5,flux1_control}_mlx_smoke.rs`, `{flux2_dev,realvisxl_lightning,scail2}_gpu_smoke.rs`, `footprint_measure.rs:56-137`
- **Finding:** `env_or`, `image_std`/`image_mean`, `is_all_zero`, `save_png`, the progress-dedup closure, and `resolve_qX_dir`/`cached_turnkey_root` are copied verbatim across 10+ files; `footprint_measure.rs` already has the generalized `resolve_tier_dir`/`cached_tier_dir`.
- **Impact:** Every new tier smoke re-copies ~90 lines; improvements (e.g. the stronger coherence gates only in `sd3_5_mlx_smoke.rs:85-137`) don't propagate, and copies are already drifting (F-072).
- **Suggested fix:** A `#[cfg(test)] mod smoke_support` exporting the shared helpers; each smoke keeps only engine id, repo, sentinel, and request recipe.
- **Confidence:** High

#### [F-065] Split the 3,435-line worker `tests.rs` monolith into per-domain modules
- **Category:** readability
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/tests.rs:1-3435`
- **Finding:** One flat module mixes ≥8 domains (HF validation, family detection, snapshot downloads + five axum stub servers, tokenizer overlays, supervisor lifecycle, media planning, credentials, cancel, manifest gating) with shared helpers buried ~2,900 lines below first use.
- **Impact:** Hard to locate coverage; four separately hand-rolled stub servers already exist; merge conflicts concentrate here.
- **Suggested fix:** Break into `tests/` submodules with the stub servers and `test_settings` in a `support` module.
- **Confidence:** High

#### [F-066] Don't drive the CFG-free SD3.5 Large Turbo with guidance + a negative prompt in the LoRA smoke
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/sd3_5_mlx_smoke.rs:393-441`
- **Finding:** The LoRA smoke's engine table gives `sd3_5_large_turbo` `default_guidance` 3.5 and passes `Some(guidance)` unconditionally (also switching on a negative prompt at L192), while the descriptor advertises `supports_guidance=false` and the dedicated turbo smoke passes `None`.
- **Impact:** Running the LoRA-apply smoke against Turbo either fails spuriously or validates a request shape the shipped worker never sends.
- **Suggested fix:** Make the tuple carry `Option<f32>` and use `None` for the turbo entry.
- **Confidence:** Medium

---

## Low findings

#### [F-067] Unauthenticated PUT on `/api/v1/ui-preferences`
- **Category:** security · **Severity:** Low · **Location:** `apps/rust-api/src/lib.rs:119-126`, `auth.rs:72-74`, `preferences.rs:83-105`
- **Finding:** `requires_token` is path-only and `/api/v1/ui-preferences` is in `PUBLIC_PATHS` to allow the pre-auth theme *read*, but the PUT handler (writes `ui-preferences.json`) is equally exempt.
- **Impact:** On a token-configured LAN bind, any unauthenticated caller can overwrite theme/accent (whitelist-validated, so nuisance) — an unauthenticated disk write violating epic 4484's "every write authenticated" invariant.
- **Suggested fix:** Make `requires_token` method-aware, or split the route so only GET is public.
- **Confidence:** High

#### [F-068] No rate limiting on the public token-verification oracle
- **Category:** security · **Severity:** Low · **Location:** `apps/rust-api/src/lib.rs:1060-1064`, `auth.rs:91-99`
- **Finding:** `POST /api/v1/auth/verify` is public and returns `{ok}` for any candidate; no rate limit/lockout/attempt counter anywhere in the middleware (compare is constant-time, so only online brute force applies).
- **Impact:** In LAN mode the token is a user-chosen password; a LAN attacker can guess at wire speed. Weak passwords are brute-forceable.
- **Suggested fix:** Add a small in-memory failed-attempt throttle keyed by peer IP on `verify_access` and `auth_rejected`; log repeated failures at warn.
- **Confidence:** High

#### [F-069] Validate the timeline id charset before deriving its file path
- **Category:** security · **Severity:** Low · **Location:** `crates/sceneworks-core/src/project_store.rs:773-820,2979-2992`
- **Finding:** `save_timeline` never runs `is_safe_id` on the client-supplied `id`; `timeline_file_path` embeds the id's last 8 chars into the write path, and `relative_string`'s lexical strip lets an unnormalized `..`-bearing id pass. Asset/character/track/dataset ids are all charset-checked; timelines are the gap.
- **Impact:** Constrained write anywhere inside the project dir (filename always ends `.sceneworks.timeline.json`) plus an index row whose `file_path` contains `..`.
- **Suggested fix:** Reject `id`s failing `is_safe_id` in `save_timeline`.
- **Confidence:** Medium

#### [F-070] Stop trusting the upload filename extension for video uploads
- **Category:** security · **Severity:** Low · **Location:** `crates/sceneworks-core/src/project_store.rs:1277-1320,3831-3843`
- **Finding:** For non-image uploads `upload_extension` takes the extension verbatim from the client filename; an upload declared `video/mp4` but named `evil.html` is stored `…-xxxx.html`, and `project_file` derives the serve mime from the stored filename → `text/html`. Images are protected by sniffing; videos aren't.
- **Impact:** Content-type confusion on the file-serving endpoint → potential stored-XSS on the API origin, worse under epic 4484.
- **Suggested fix:** Derive the stored extension from the declared/sniffed mime for video, or allow-list stored extensions.
- **Confidence:** Medium (depends on rust-api response headers).

#### [F-071] Verify job ownership before accepting a heartbeat's `current_job_id`
- **Category:** security · **Severity:** Low · **Location:** `crates/sceneworks-core/src/jobs_store.rs:651-656`
- **Finding:** `heartbeat_worker` updates `last_heartbeat_at`/`updated_at` for whatever `current_job_id` the worker reports, without checking the job's `worker_id` matches (progress updates were hardened against this in sc-4172; heartbeat wasn't).
- **Impact:** A stale/buggy worker heartbeating an old job id keeps refreshing it, preventing the stale sweep from marking it `interrupted` — a stuck "running" job.
- **Suggested fix:** Add `and worker_id = ?<reporting worker>` to the heartbeat UPDATE.
- **Confidence:** Medium

#### [F-072] Fix the empty-value early return in secret redaction; scope over-broad Authorization redaction
- **Category:** security · **Severity:** Low · **Location:** `crates/sceneworks-core/src/session_log.rs:175-208`
- **Finding:** In `redact_marker_value`, an immediately-terminated marker (`token=&…`) returns the whole line instead of continuing, so a later real occurrence stays unredacted; JSON-shaped secrets (`"token":"abc"`) match no `key=value` marker; and `redact_authorization_header` replaces from `authorization:` to end-of-line, destroying subsequent JSON fields.
- **Impact:** Secrets can survive into the session ring buffer / `GET /api/v1/logs` (shown on the Logs screen, copied into bug reports); structured-log fidelity is lost for lines carrying an `authorization` key.
- **Suggested fix:** `search_from = value_end; continue;` instead of returning; add a JSON-marker pass; terminate Authorization redaction at the next delimiter.
- **Confidence:** High

#### [F-073] Confine relative pose source paths like absolute ones
- **Category:** security · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/pose_jobs.rs:754-772`
- **Finding:** Absolute payload paths are confined and asset-id resolution rejects non-`Normal` components, but the project-relative branch does a bare `proj.join(raw)` with no `..` filter and the final fallback passes a raw relative path to `decode_image_any` (resolved against cwd). `../../<anything>.png` escapes.
- **Impact:** Pose detection on any worker-readable image (keypoint/skeleton disclosure, not raw bytes); inconsistent with WKA-002.
- **Suggested fix:** Apply the `Component::Normal`-only filter to the `proj.join(raw)` branch and drop the raw fallback.
- **Confidence:** High (mechanism); impact rated low.

#### [F-074] Sidecar `mask` path escapes the project directory in person_replace
- **Category:** security · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/person_replace.rs:218-222`
- **Finding:** `project_path.join(rel)` uses the sidecar JSON's `mask` string verbatim; a `../../…`/absolute value resolves outside the project dir and is read as the replacement mask.
- **Impact:** Low (sidecars app-written, single-user worker), but a tampered project file can pull arbitrary readable images into output.
- **Suggested fix:** Reject absolute/`..` paths (or canonicalize + require under `project_path`) before reading.
- **Confidence:** High (traversal real; severity capped by threat model).

#### [F-075] Canonicalize before `ensure_path_under` in the lexical-only confinement helpers
- **Category:** security · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/paths.rs:68-98,286-342`
- **Finding:** `ensure_path_under` does a lexical `starts_with` (no symlink resolution) in `normalize_app_managed_path`, `normalize_app_managed_model_path`, `resolve_lora_import_target`, `resolve_model_import_target`, `resolve_model_convert_output` — while the LoRA-path/cache helpers in the same file also check the canonicalized form. A symlink under `data/models`/`data/loras` lets a confined-looking write land outside.
- **Impact:** Defense-in-depth gap; two confinement strengths coexist and the write-target helpers got the weaker one. Requires local write access, hence Low.
- **Suggested fix:** Route these five through the same dual (`normalize_absolute_path` + `normalize_existing_or_absolute`) check.
- **Confidence:** Medium

#### [F-076] Sanitize `plan.output.file_name` before the trainer joins it; canonicalize dataset/output confinement
- **Category:** security · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/training_jobs.rs:679,712-719,132-149`
- **Finding:** `output_dir` is confined but `file_name` reaches `TrainingRequest` verbatim and is joined under `output_dir` by the engine (only the preview stem is sanitized); a `../`/absolute `file_name` escapes. The training/dataset/output guards are also lexical-only (no canonicalize) unlike `normalize_app_managed_lora_path`.
- **Impact:** A forged payload gets an arbitrary-location adapter-file write — inconsistent with the trust posture of every other payload path in the same function.
- **Suggested fix:** Reject non-`Normal` `file_name` components in `validate_training_plan`; route dataset/output through the canonical+lexical dual check.
- **Confidence:** Medium (engine-side join not inspected).

#### [F-077] Verify or pin third-party runtime-weight downloads (DWPose, SeedVR2)
- **Category:** security · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/pose_jobs.rs:62-65,636-653`, `upscale_jobs.rs:445-541`
- **Finding:** DWPose ONNX zips are fetched from `download.openmmlab.com` with `expected_size:None` and no digest; the SeedVR2 checkpoint comes from third-party `numz/SeedVR2_comfyUI` at the mutable `main` revision (the HF `lfs.oid` check only proves consistency with current `main`).
- **Impact:** TLS is the only integrity control on model artifacts parsed by onnxruntime/the MLX loader; a compromised upstream or moved `main` silently changes what runs.
- **Suggested fix:** Pin known SHA-256 digests (DWPose) and a fixed HF revision (SeedVR2), verifying via `verify_file_sha256`.
- **Confidence:** High (behavior); Medium (risk weighting).

#### [F-078] Access password persisted in plaintext localStorage
- **Category:** security · **Severity:** Low · **Location:** `apps/web/src/App.jsx:1286`, `credentials.js:21-25`
- **Finding:** The remote-access password (= API token) is stored verbatim in `localStorage["sceneworks-token"]`; any XSS on the origin can exfiltrate it and it survives indefinitely.
- **Impact:** Contained by the app's strong XSS posture and LAN-scoped threat model, but the token is the *only* host credential.
- **Suggested fix:** Acceptable for the threat model; if hardening, prefer `sessionStorage` or an httpOnly-cookie exchange and document the tradeoff.
- **Confidence:** High

#### [F-079] Harden manifest-supplied license/homepage URLs before rendering as links
- **Category:** security · **Severity:** Low · **Location:** `apps/web/src/screens/ModelManagerScreen.jsx:250-259`, `LicensesScreen.jsx:81`
- **Finding:** `model.licenseUrl` and `component.homepage` are rendered directly into `<a href>`; `gatedRepoUrl()` builds its own `https://` URL but `licenseUrl` is used raw, so a `javascript:` value in a manifest executes on click.
- **Impact:** Low today (manifests local, import disabled per epic 7080); a future user-imported manifest turns it into a click vector.
- **Suggested fix:** Validate scheme (`http:`/`https:`) via a shared `safeExternalUrl()` before rendering.
- **Confidence:** Medium

#### [F-080] Startup directory creation and stale-upload sweeps silently swallow errors
- **Category:** bad-pattern · **Severity:** Low · **Location:** `apps/rust-api/src/lib.rs:670-679`
- **Finding:** `create_app_with_state` discards `create_dir_all` (×3) and all four sweep errors with bare `let _ =` and no logging (unlike the adjacent `ensure_global_poses_project` which logs).
- **Impact:** A permissions/disk problem is invisible; leaked multi-GB upload temps are never reclaimed; only downstream 500s show.
- **Suggested fix:** Replace `let _ =` with `if let Err(e) = … { warn!(…) }`.
- **Confidence:** High

#### [F-081] Duplicate `file` multipart field leaks the first staged asset temp file
- **Category:** bad-pattern · **Severity:** Low · **Location:** `apps/rust-api/src/assets.rs:87-92,121-126`
- **Finding:** In `import_asset`, a second `file` field overwrites `file = Some((…, temp_path))` without deleting the prior temp; error-path cleanup removes only the last. The LoRA/model imports reject a second file; this one doesn't.
- **Impact:** A client bug/crafted request leaves an orphaned multi-GB tmp per request until the 24 h sweep — a disk-exhaustion lever for an authenticated caller.
- **Suggested fix:** Reject a second `file` field with 400, or delete the prior temp before replacing.
- **Confidence:** High

#### [F-082] `negativePrompt` and `advanced` escape all length validation
- **Category:** bad-pattern · **Severity:** Low · **Location:** `apps/rust-api/src/lib.rs:1981-2013`, `dto.rs:586-640`
- **Finding:** `prompt` is capped at 4000 chars but `negative_prompt`, the free-form `advanced` object, and `loras[*]` strings have no size limit — anything up to the body limit is persisted into jobs.db and broadcast over SSE on every `job.updated`.
- **Impact:** One oversized field bloats the row and re-serializes to every SSE subscriber on every status change.
- **Suggested fix:** Cap `negative_prompt` (4000) and a serialized-size ceiling on `advanced` (e.g. 64 KiB) in `validate_image_job`/`validate_video_job`.
- **Confidence:** High

#### [F-083] Four copy-paste stale-upload sweepers with behavioral drift
- **Category:** redundant · **Severity:** Low · **Location:** `apps/rust-api/src/assets.rs:160-196`, `loras.rs:972-1008`, `poses.rs:74-101`, `keypoints.rs:106-133`
- **Finding:** Four near-identical `sweep_stale_*_uploads` differ only in subdir — except the LoRA variant skips non-directories while the others handle both, and only assets/loras have the testable `_before(cutoff)` split; all share the misnamed `STALE_LORA_UPLOAD_SECONDS`.
- **Impact:** The next staging area (or a fix to one sweeper) will be copied to some-but-not-all variants; drift already visible.
- **Suggested fix:** One `sweep_stale_uploads(data_dir, subdir, cutoff)`; rename the constant to `STALE_UPLOAD_SECONDS`.
- **Confidence:** High

#### [F-084] Three copy-paste multipart streaming writers
- **Category:** redundant · **Severity:** Low · **Location:** `apps/rust-api/src/assets.rs:202-244`, `loras.rs:691-736`, `models.rs:725-773`
- **Finding:** `write_upload_field_to_dir`, `write_lora_upload_field_to_staged_file`, `write_model_upload_field_to_staged_file` are the same stream-to-temp-with-cap loop differing in cap source/dir/message; cleanup helpers are also duplicated (`drop(file)` before cleanup only in the model variant).
- **Impact:** A fix to the chunk loop (fsync-before-rename, cancellation cleanup) must be applied three times.
- **Suggested fix:** One writer `(dir, filename, max_bytes, limit_msg)` + one shared cleanup.
- **Confidence:** High

#### [F-085] Dead "Phase 2" typed-contract helpers in recipe_presets.rs
- **Category:** dead-code · **Severity:** Low · **Location:** `apps/rust-api/src/recipe_presets.rs:1024-1107`
- **Finding:** `value_to_recipe_preset_entry`, `recipe_preset_entry_to_value`, `validate_typed_recipe_preset_entry` are `#[allow(dead_code)]` with zero callers; the last duplicates the live validators and has already drifted (its `_ => "unknown"` workflow arm would pass a bogus workflow).
- **Impact:** ~85 lines of unmaintained parallel validation that will diverge from the live path and mislead readers about which validator is authoritative.
- **Suggested fix:** Delete them, or file the Phase-2 conversion as a story and remove until it lands.
- **Confidence:** High

#### [F-086] Duplicate archive-character handlers; use the shared CSPRNG id generator in create_job
- **Category:** redundant · **Severity:** Low · **Location:** `apps/rust-api/src/characters.rs:69-91`; `crates/sceneworks-core/src/jobs_store.rs:1476-1481`
- **Finding:** `archive_character` (DELETE) and `archive_character_explicit` (POST) have byte-identical bodies; and `create_job_on_connection` still generates ids via `select lower(hex(randomblob(16)))` — the pattern sc-4209 removed from the project stores in favor of `store_util::random_hex`.
- **Impact:** A behavior change to archiving must touch two handlers; inconsistent id generation with the old SQLite-failure surface.
- **Suggested fix:** Route both methods to one handler; replace with `format!("job_{}", random_hex(16)?)`.
- **Confidence:** High

#### [F-087] Duplicate stale-worker sweep on every claim/mutation via publish_queue
- **Category:** efficiency · **Severity:** Low · **Location:** `apps/rust-api/src/jobs.rs:55-57,121-128`, `lib.rs:1303-1322`
- **Finding:** `claim_job` runs `mark_stale_workers_interrupted` in its own transaction, then `publish_queue → queue_summary_snapshot` runs it again in a second blocking round-trip; every job mutation pays a sweep inside `publish_queue` even when one just ran.
- **Impact:** Two blocking dispatches + duplicate SQLite sweep per worker claim poll — the API's hottest path doing double work.
- **Suggested fix:** Give `queue_summary_snapshot` a `skip_sweep` variant for callers that just swept.
- **Confidence:** High

#### [F-088] `lib.rs` is a 2,530-line grab-bag; `STALE_LORA_UPLOAD_SECONDS` misnamed
- **Category:** readability · **Severity:** Low · **Location:** `apps/rust-api/src/lib.rs:1-2530,146`
- **Finding:** The crate root mixes the router, settings, server lifecycle, worker supervision, byte-range parsing, per-domain validators, ~40 `default_*` fns, and the error type, with `use module::*` glob re-exports making symbol origin untraceable; and the 24 h cutoff constant is named for LoRA uploads but governs asset/pose/keypoint sweeps too.
- **Impact:** New handlers accrete helpers into lib.rs by default; glob imports make refactors risky; the constant misleads when tuning one flow.
- **Suggested fix:** Move validators/`default_*` next to their DTOs, extract `error.rs`/`server.rs`, replace glob imports; rename to `STALE_UPLOAD_SECONDS`.
- **Confidence:** High

#### [F-089] Infallible `EventTicketStore::issue` returns `Result`
- **Category:** dead-code · **Severity:** Low · **Location:** `apps/rust-api/src/events.rs:55-65`
- **Finding:** `issue()` has no failure path but returns `Result<EventTicket, ApiError>`, forcing a `?` for an error that can't occur.
- **Impact:** Cosmetic; obscures the actual failure surface.
- **Suggested fix:** Return `EventTicket` directly.
- **Confidence:** High

#### [F-090] Remove the duplicated trigger-word caption composer
- **Category:** redundant · **Severity:** Low · **Location:** `crates/sceneworks-core/src/training_store.rs:1024-1037`, `training.rs:2110-2123`
- **Finding:** `caption_text_with_trigger_words` and `caption_with_trigger_words` are line-for-line identical, feeding caption sidecars vs the training plan.
- **Impact:** If the dedup/ordering rule changes in one copy, the `.txt` sidecars a user inspects and the captions the trainer consumes silently disagree.
- **Suggested fix:** Keep one `pub(crate)` fn in `training.rs`, call from `training_store.rs`.
- **Confidence:** High

#### [F-091] Replace the hand-rolled character JSON pretty-printer
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-core/src/character_store.rs:1339-1510`
- **Finding:** Character sidecars are serialized by a ~170-line recursive JSON writer with heuristic key ordering (`ordered_character_keys` guesses object "type" from which keys are present); a new field silently changes which branch matches and reorders the file.
- **Impact:** Fragile untyped serialization for a format that round-trips through serde everywhere else; a misclassification writes strangely-ordered files and defeats the diff-stability goal it exists for.
- **Suggested fix:** Serialize `contracts::Character` with `to_string_pretty`, or a `#[serde(serialize_with)]` key-order shim.
- **Confidence:** Medium

#### [F-092] Cache asset-summary lookups when hydrating character references
- **Category:** efficiency · **Severity:** Low · **Location:** `crates/sceneworks-core/src/character_store.rs:1088-1151,1242-1249`
- **Finding:** `hydrate_character` calls `character_asset_summary` per reference, each opening a **new** SQLite connection (+`apply_project_migrations`) and re-reading the sidecar; `list_characters` does this for every reference of every character.
- **Impact:** 20 characters × 15 references ≈ 300 connections + 300 sidecar reads per Character Studio listing — O(C×R) connections where one would do.
- **Suggested fix:** Open one connection in `list_characters`/`get_character` and thread it + an asset memo through `hydrate_character`.
- **Confidence:** High

#### [F-093] Avoid cloning the whole ring buffer before applying the query limit
- **Category:** efficiency · **Severity:** Low · **Location:** `crates/sceneworks-core/src/session_log.rs:119-151`
- **Finding:** `SessionLog::query` clones every matching entry into a Vec then `split_off`s down to `limit`, holding the buffer mutex — up to 5000 clones to return 500 while blocking `push_line` from the stdout capture threads.
- **Impact:** Unnecessary allocation and lock-hold on a hot polling path; worst case stalls sidecar stdout ingestion.
- **Suggested fix:** Iterate `.rev()` collecting up to `limit`, then reverse; clone only what's returned.
- **Confidence:** High

#### [F-094] Fold the per-status loop in list_jobs_by_status into one `status in (…)` query
- **Category:** efficiency · **Severity:** Low · **Location:** `crates/sceneworks-core/src/jobs_store.rs:1508-1521`
- **Finding:** The helper prepares+executes `where status = ?1` once per status (5× for `ACTIVE_STATUSES`) although `active_statuses_sql()` exists.
- **Impact:** 5 statement preps + 5 scans per startup interrupt sweep; rows ordered by status-group not creation time.
- **Suggested fix:** `format!("… where status in ({})", active_statuses_sql())`.
- **Confidence:** High

#### [F-095] Move `parse_utc_seconds`/`days_from_civil` next to their inverse in time.rs
- **Category:** readability · **Severity:** Low · **Location:** `crates/sceneworks-core/src/jobs_store.rs:1935-1985` vs `time.rs:24-37`
- **Finding:** `time.rs` owns `civil_from_days` while its exact inverse `days_from_civil` and the hand-rolled `parse_utc_seconds` live buried in `jobs_store.rs` — two halves of one date algorithm 8k lines apart; the parser even special-cases a `.digitsZ` suffix `format_unix_seconds` never emits.
- **Impact:** Anyone adjusting the timestamp format must know to update both files.
- **Suggested fix:** Move both into `time.rs` beside `format_unix_seconds` with a round-trip test.
- **Confidence:** High

#### [F-096] Confine payload-path imports — the dead `exists()` after `canonicalize()` in LoRA import
- **Category:** redundant · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/imports.rs:13-19`
- **Finding:** `import_lora_source_path` calls `source.canonicalize()?` (fails NotFound if missing) then checks `!source.exists()` to build a NotFound error — the branch is unreachable except a rare TOCTOU, and the friendly message never fires for the common case.
- **Impact:** Dead branch; the intended friendly "LoRA source not found" is never what the user sees.
- **Suggested fix:** Match the canonicalize error, map NotFound to the friendly message; drop the `exists()` check.
- **Confidence:** High

#### [F-097] Don't let one child's restart backoff stall the whole supervisor tick
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/supervisor.rs:138-162`
- **Finding:** `restart_exited_children_with_spawner` sleeps each exited child's backoff delay inline and sequentially inside the supervision tick, so a 30 s backoff on one child delays restarting the others (and detecting further exits).
- **Impact:** On a multi-GPU/multi-utility host, one crash-looping child slows recovery of healthy siblings.
- **Suggested fix:** Track per-child `next_restart_at: Instant`, restart eligible children each 1 s tick instead of sleeping inline.
- **Confidence:** High

#### [F-098] Guard against interleaved writes when two utility workers download the same file
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/downloads.rs:16-59,318-437`
- **Finding:** `ensure_cached_file`/`download_file_inner` write shared cache targets with no cross-process locking; the default 4-process utility pool can have two jobs resume/append the same partial file concurrently.
- **Impact:** Interleaved appends can produce a corrupt file that passes the size check when no sha256 is available, surfacing later as an opaque load failure.
- **Suggested fix:** Write to a per-process temp name and rename, or advisory-lock `<target>.lock` for the transfer.
- **Confidence:** Medium (race verified; API-side dedup may make it rare).

#### [F-099] Log the swallowed accel-init errors before falling back to CPU (YOLO, DWPose, Upscaler)
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/person_jobs.rs:861-872`, `pose_jobs.rs:464-480`, `upscale_jobs.rs:258-268`
- **Finding:** `OrtYolo::load`, `Detector::load`, and `Upscaler::load` match `Err(_)` on the CUDA/CoreML session build and silently rebuild on CPU, discarding the error that explains why; DWPose also eagerly builds the second accel session even after the first failed.
- **Impact:** A misconfigured GPU box silently runs ~10× slower with no log; sc-6209-style dylib issues become guess-and-check.
- **Suggested fix:** Bind and `warn!` the error before the CPU fallback; short-circuit the second accel build.
- **Confidence:** High

#### [F-100] Degrade to box mask instead of failing the job on a corrupt stored mask
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/person_replace.rs:224-232`
- **Finding:** `load_track_masks` `?`-propagates a decode failure of a stored mask, failing the whole replace-person job, while a *missing* file already falls back to `box_mask` and the contracts say located-person tracks are never failed by the mask pass.
- **Impact:** One corrupt mask PNG aborts an otherwise-runnable video replacement.
- **Suggested fix:** On decode error, log and push `box_mask(...)` for that frame (keeping the degraded accounting), unless Python parity requires the raise.
- **Confidence:** Medium

#### [F-101] Replace frames/anchors `assert_eq!` with a WorkerError in the blocking entry points
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/person_segment.rs:165-169`, `person_segment_sam3.rs:268-272`, `person_segment_sam3_candle.rs:266-270`
- **Finding:** The three `*_blocking` entry points `assert_eq!` on `clip_frame_paths.len() == anchors.len()`; a mismatch panics inside `spawn_blocking` and the media_jobs callers absorb it into a silent "degraded" result.
- **Impact:** An internal invariant violation masquerades as normal quality degradation; the bug goes unnoticed.
- **Suggested fix:** Return `WorkerError::InvalidPayload`/`Engine` on mismatch.
- **Confidence:** High

#### [F-102] Guard ONNX output rank/length before indexing in `decode`
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/person_jobs.rs:238-256`; `pose_jobs.rs:172-190,236-298`
- **Finding:** `decode` indexes `shape[1]`/`shape[2]` and `data[score_ch*anchors+a]` without checking `shape.len()>=3` or `data.len()`; `yolox_decode`/`pose_decode`/`wholebody_to_openpose` similarly index unchecked. `SCENEWORKS_PERSON_DETECTOR_WEIGHTS`/`_DWPOSE_*` let a user pin an arbitrary ONNX with a different output shape, and `wholebody_to_openpose` runs on the async task.
- **Impact:** A wrong env-pinned/corrupt model panics (index OOB) instead of a clear "unexpected output shape" error — unwinding the async task in the DWPose case.
- **Suggested fix:** Early-`Err(Engine)` when the shape/length doesn't match.
- **Confidence:** High (panic path real; likelihood low).

#### [F-103] Bounds-check the predictor's returned frame index in pass assembly; `mask_to_frame` swallows malformed masks
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/person_segment.rs:232-238`, `person_segment_sam3.rs:229-244,339-347`
- **Finding:** `out[*frame_idx as usize] = …` trusts `propagate`'s indices (a negative i32 wraps via `as usize`), and the SAM3 path indexes `frame.masks[i]` assuming parallel-length vecs; `mask_to_frame` returns an empty vec if `GrayImage::from_raw` fails — the exact sentinel the orchestrator reads as "object absent → box fallback".
- **Impact:** A model-contract slip panics (absorbed as silent "degraded"), and a genuine mask-length bug silently degrades mask quality frame-by-frame.
- **Suggested fix:** `out.get_mut(idx)` with an `Engine` error; return `WorkerResult<Vec<u8>>` from `mask_to_frame` (or debug-assert the length).
- **Confidence:** High/Medium

#### [F-104] Merge the duplicated NHWC/NCHW letterbox preprocessors; classify YOLO forward failures as Engine
- **Category:** redundant · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/person_jobs.rs:193-227,786-820,710-719`
- **Finding:** `preprocess` (macOS NHWC) and `preprocess_nchw` (off-Mac) are byte-for-byte identical except the destination index; both cfg-gated so drift is invisible per platform. Separately, `detect_people` maps forward/reshape errors to `InvalidPayload` while siblings use `Engine`.
- **Impact:** A future letterbox fix desyncs the two backends; misleading error taxonomy misfiles engine faults as user errors.
- **Suggested fix:** One shared fn taking a layout enum/index closure; change the two `map_err`s to `Engine`.
- **Confidence:** High

#### [F-105] Bounds-check ONNX / error on mask-buffer mismatch in scail2_masks
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/scail2_masks.rs:64-73`
- **Finding:** `paint` bounds-checks each write, silently ignoring any mask longer than the `w*h` buffer and painting fewer pixels for a shorter one; a mismatch can only mean an upstream SAM3 dimension bug.
- **Impact:** A dimension bug corrupts the color-coded SCAIL-2 conditioning masks (wrong person regions) with no error or log.
- **Suggested fix:** Debug-assert or log/error when `mask.len() != px.len()/3`, keeping the per-pixel guard.
- **Confidence:** High (behavior); Medium (reachability).

#### [F-106] Wire the smart-select CancelFlag into the SAM3 compute or stop creating it
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/segment_jobs.rs:282-319`
- **Finding:** A `CancelFlag` is created and passed to `run_blocking_with_heartbeat` but neither `segment_points_blocking` nor `segment_box_blocking` accepts/reads it (verified at `person_segment_sam3.rs:378,496`), so tripping it does nothing.
- **Impact:** User cancel on smart-select is a no-op until completion; the flag reads as if the compute were cancelable.
- **Suggested fix:** Thread the flag into the SAM3 stage boundaries, or drop it with a comment (matching kps_jobs' note).
- **Confidence:** High

#### [F-107] Move source decode/encode off the async thread (kps, segment, upscale, likeness, training samples)
- **Category:** efficiency · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/kps_jobs.rs:268-275`, `segment_jobs.rs:238-241,338-341`, `upscale_jobs.rs:854-856,1195-1197`, `face_likeness_compare_jobs.rs:207`, `training_jobs.rs:888-898,1126-1170`, `image_jobs.rs:1104-1137`, `image_jobs/detail.rs:510-517`
- **Finding:** These handlers do synchronous `std::fs::read` + full image decode (potentially an AVIF/HEIC transcode subprocess) or PNG encodes directly on the async runtime thread, with only the model inference in `spawn_blocking`; `write_training_sample` also `image.pixels.clone()`s the full buffer.
- **Impact:** Runtime-thread stalls of hundreds of ms on large photos, delaying heartbeats — inconsistent with the care around the inference stage.
- **Suggested fix:** Fold decode/encode into the existing `spawn_blocking` closures; take `image` by value.
- **Confidence:** High

#### [F-108] Cache the SCRFD detector across kps jobs (or document the per-job reload)
- **Category:** efficiency · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/kps_jobs.rs:144-155,192-197`
- **Finding:** Every `kps_extract` job reloads the ~100+ MB SCRFD weights and rebuilds the model, unlike the sibling detectors (`pose_jobs::DETECTOR`, `person_segment_sam3::WEIGHTS`) which cache process-wide.
- **Impact:** Seconds of avoidable per-job latency for an interactive action.
- **Suggested fix:** Mirror the `OnceLock<Mutex<Option<…>>>` cache, or comment that the reload is deliberate for GPU headroom.
- **Confidence:** High (mechanism); Medium (that caching is net-right).

#### [F-109] Don't silently ignore a nonexistent env-pinned weight path (DWPose, upscaler, SeedVR2)
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/pose_jobs.rs:617-622`, `upscale_jobs.rs:376-386,490-495`
- **Finding:** If `SCENEWORKS_DWPOSE_DET`/`_POSE` (etc.) is set but the path doesn't exist, `ensure_one` silently falls through to cache/download.
- **Impact:** A typo'd pin silently validates the *downloaded* weights instead of the pinned ones.
- **Suggested fix:** Error (or `warn!`) when the env var is set but missing.
- **Confidence:** High (behavior); Medium (that erroring is the right call).

#### [F-110] Run temp-upload / work-dir cleanup on error paths too (pose, person-track)
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/pose_jobs.rs:945-948,979`; `media_jobs.rs:734-820,1213-1299`
- **Finding:** `cleanup_temp_sources` runs only on the pose success path; the person-track `work_dir` is removed only on not-found and success — every intervening `?` (cancel, render fail, detect error) leaks the staged uploads / 24 frame PNGs. The timeline export uses `tempfile::Builder` (RAII) and gets this right.
- **Impact:** Orphaned upload files / temp frame dirs accumulate after failed/canceled jobs.
- **Suggested fix:** Use `tempfile::Builder`, or run cleanup defer-style on both outcomes.
- **Confidence:** High

#### [F-111] Person-track work dir embeds raw job.id instead of `safe_download_dir`
- **Category:** security · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/media_jobs.rs:734,1213` (also `video_jobs.rs:5980,1243`)
- **Finding:** `temp_dir().join(format!("sw-person-track-{}", job.id))` embeds the API-supplied job id unsanitized, while the timeline export in the same file routes the same value through `safe_download_dir`.
- **Impact:** Job ids are trusted UUIDs today (not exploitable now), but the file's own convention is applied inconsistently and epic 4484 moves the API toward untrusted input.
- **Suggested fix:** `safe_download_dir(&job.id)` at all temp-dir sites.
- **Confidence:** High (inconsistency); Low (present-day exploitability).

#### [F-112] run_frame_extract/run_person_detect and person-track constants duplicate scaffolding
- **Category:** redundant · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/media_jobs.rs:99-286,348-567,1553,1998`
- **Finding:** Both frame handlers resolve project/asset/source, render, tmp-rename, build near-identical `"type":"frame"` asset JSON, write sidecar+recipe, and index (drift already visible in the F-029 clamp); and `PERSON_TRACK_SAMPLE_RATE_FPS`/`MAX_SAMPLES` value-duplicate `person_track::` constants (sidecar records one, sampler uses the other).
- **Impact:** Divergent behavior between the two frame paths; changing the real sample cadence would leave sidecar metadata lying to downstream consumers.
- **Suggested fix:** Extract `render_frame_asset(...)` with a recipe-builder closure; re-export one source of truth for the constants.
- **Confidence:** High

#### [F-113] One ffmpeg process spawned per sampled frame in person tracking
- **Category:** efficiency · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/media_jobs.rs:744-790,1223-1269`
- **Finding:** Each of up to 24 sampled frames spawns a fresh accurate-seek ffmpeg process that re-opens the container and decodes from the nearest keyframe.
- **Impact:** For long-GOP sources, sampling dominates pre-detection wall clock (bounded by MAX_SAMPLES=24).
- **Suggested fix:** Single ffmpeg invocation with a `select=`/`fps` filter to an image2 sequence.
- **Confidence:** Medium

#### [F-114] Caption/training backend labels and error strings hardcode "MLX"/"mlx"
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/caption_jobs.rs:100,143,170,205,340-344`; `training_jobs.rs:1217,640-645`
- **Finding:** The shared caption path emits "Loading JoyCaption MLX model." / "JoyCaption MLX load failed" even on the candle backend, and the unsupported message says "Windows candle backend" (candle also serves Linux); `training_result` hardcodes `"backend":"mlx"` and `run_training_execution` reports "No MLX trainer" on a backend-neutral path.
- **Impact:** Misleading triage ("why does my CUDA worker say MLX failed?") and wrong provenance recorded in off-Mac training results.
- **Suggested fix:** Interpolate the computed backend label everywhere.
- **Confidence:** High

#### [F-115] Analysis jobs post terminal Canceled while the blocking task still runs; redundant re-maps
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/dataset_analysis_jobs.rs:134,237-241`, `face_analysis_jobs.rs:311-315`, `training_jobs.rs:815,940-946`
- **Finding:** The analysis tick arms call `check_cancel` (posts terminal `Canceled` at ack time) then keep looping — the pattern training replaced with `cancel_requested_peek` + deferred terminal write; `dataset_analysis` also `items.clone()`s the whole vec used only for `len()`, `consume_training_events` re-runs `map_training_config` just for sample fields, and every training progress event posts `update_job`+`heartbeat` serially.
- **Impact:** The scheduler can hand the "free" worker a new job while the current embed finishes; wasted clones/round-trips; back-pressure throttles GPU stepping to API latency.
- **Suggested fix:** `cancel_requested_peek` + deferred terminal write; compute `total` and move `items`; pass the finalized config in; drop the per-event heartbeat.
- **Confidence:** High

#### [F-116] Collapse duplicated download completion-result blocks; share the refine reply-cleanup prelude
- **Category:** redundant · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/model_jobs.rs:66-96,159-188,197-321`; `prompt_refine_jobs.rs:186-212,572-587`
- **Finding:** The `{modelId, repo, path, storage, completedAt}` result + `Completed` update is built twice in `run_model_download_job` and re-copied in `run_lora_download_job`; and `clean_refine_output`/`clean_json_output` duplicate the strip-think/orphan-close/fence-unwrap steps verbatim.
- **Impact:** Parallel maintenance for identical logic.
- **Suggested fix:** `complete_hf_cache_download(...)` helper; `strip_reasoning_and_fence` helper.
- **Confidence:** High

#### [F-117] Remove/wire the unused `dtype` convert field; candle interleave's un-trippable cancel flag
- **Category:** dead-code · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/model_jobs.rs:772-775,1017`; `sensenova_jobs.rs:842-858`
- **Finding:** Convert `dtype` is parsed (default "bfloat16") but only interpolated into a progress message — no converter receives it, so `dtype:"float16"` claims a float16 conversion while the converters do their fixed thing; and the candle interleave handler constructs a fresh un-trippable `CancelFlag` inside the blocking closure (the macOS twin passes `None`).
- **Impact:** Misleading message vs actual behavior; reads as if candle mid-rollout cancel works when it doesn't.
- **Suggested fix:** Drop or thread `dtype`; thread the real per-job cancel flag on both backends (or match `None`).
- **Confidence:** High

#### [F-118] Clean up scratch dirs on the error path; reject unsupported upscale factors; other job hygiene
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/model_jobs.rs:689-703`; `video_jobs.rs:1330,417-427,2704`
- **Finding:** `ensure_ltx_upscaler_cached` removes its scratch dir only on success (the `?` leaks it); `video_upscale` coerces any factor other than 4 to 2 rather than rejecting; `scail2_raw_settings` re-resolves adapters with `.unwrap_or_default()` discarding errors; `generate_video` re-parses the whole payload just for sampler knobs.
- **Impact:** Junk dirs under `data/cache`; silently-different upscale output; duplicate disk work + a theoretical lightning-flag inconsistency; per-job re-parse allocation.
- **Suggested fix:** Bind-then-remove; `match req.factor { 2|4 => …, other => Err }`; return the resolved lightning bool from `generate_scail2`; pass `&request.advanced` into `generate_video`.
- **Confidence:** High

#### [F-119] Split the four-task `run_prompt_refine_job` boolean ladder; decompose the convert/training god functions
- **Category:** readability · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/prompt_refine_jobs.rs:600-958`; `model_jobs.rs:764-1118`; `training_jobs.rs:787-1001`; `media_jobs.rs:1391-1643`
- **Finding:** `run_prompt_refine_job` multiplexes four tasks via five booleans in six scattered ladders (incoherent combinations representable); `run_model_convert_job` is a 350-line multi-phase function with eight `fail_job` early returns; `consume_training_events` is a 9-arg ~215-line function behind `#[allow(too_many_arguments)]`; `run_person_track` returns an 8-tuple.
- **Impact:** These are the exact shapes that hid the F-003 cancel bug and invite drift.
- **Suggested fix:** `enum RefineTask`; pure `resolve_convert_plan(...)`; a `SamplePersister`/job-context struct; a `TrackKind` enum.
- **Confidence:** High

#### [F-120] Silent skip of unreadable character reference images in video replace
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/video_jobs.rs:6093-6109`
- **Finding:** `resolve_character_references` drops any reference whose `load_reference_image` fails (`if let Ok(image)`), erroring only when *all* fail.
- **Impact:** A corrupted approved reference silently reduces identity conditioning with no signal (matches torch parity, hence Low).
- **Suggested fix:** `warn!` per skipped reference and surface a `referenceCount` the UI can compare against the approved count.
- **Confidence:** High

#### [F-121] Report the Real-ESRGAN execution device instead of dead-coding it
- **Category:** dead-code · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/upscale_jobs.rs:215-219`
- **Finding:** `Upscaler.device` is `#[allow(dead_code)]` and never read, yet its doc says "Stored as `Upscaler.device` for observability" and the sibling pose job reports `detector.device`.
- **Impact:** No way to tell from an upscale result whether it ran CoreML/CUDA or fell back to CPU — the observability the field exists for.
- **Suggested fix:** Surface `device` in the result/`rawAdapterSettings` (drop the allow), or delete the field.
- **Confidence:** High

#### [F-122] Fix behavioral drift among the duplicated smoke helpers; reject unknown quant env values
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/{flux2_dev,realvisxl_lightning,scail2}_gpu_smoke.rs`, `flux1_control_mlx_smoke.rs:152`, `lens_turbo_q4_mlx_smoke.rs:118-133`, `sd3_5_mlx_smoke.rs:49-54`
- **Finding:** The copied smoke helpers diverged: the candle-side `env_or` doesn't filter set-but-empty values (panics in `.parse()`); default out dirs are cwd-relative on some smokes and `/tmp`-absolute elsewhere; degenerate-floor thresholds drift (std>5.0 vs >20.0); the two Lens smokes share env keys with inconsistently-named dir overrides; and `SD3_QUANT`/`FLUX2_DEV_QUANT` map any unrecognized value to a default tier.
- **Impact:** Confusing panics, PNGs scattered into arbitrary cwds, cross-test env bleed, and hand-run tier validations that silently PASS the wrong tier.
- **Suggested fix:** Unify on the empty-filtering `env_or`, absolute defaults, one documented threshold, per-test env prefixes; `panic!` on unrecognized quant values.
- **Confidence:** High

#### [F-123] Hoist the SDXL pipeline load out of `render()` in the RealVisXL smoke; enforce one-tier-per-process in footprint
- **Category:** efficiency · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/realvisxl_lightning_gpu_smoke.rs:62-74`, `footprint_measure.rs:23-27,162-231`
- **Finding:** `render()` calls `gen_core::load("sdxl", …)` every invocation, loading the multi-GB snapshot twice for the `RVXL_CONTRAST=1` run (the flux1 smoke loads once); and the footprint harness mandates "RUN ONE TIER PER PROCESS" (MLX counters are process-global) but nothing enforces it, so an unfiltered `--ignored` run corrupts every `[[FOOTPRINT]]` number.
- **Impact:** Doubled wall-clock/VRAM for the contrast run; silently bogus footprint numbers destined for the manifest + RAM→tier calibration (sc-8508/8509).
- **Suggested fix:** Load once and pass `&dyn Generator` into `render()`; `assert!(!FOOTPRINT_RAN.swap(true, …))` in `measure_footprint`.
- **Confidence:** High

#### [F-124] Warn on missing ORT CUDA/cuDNN dir override; extract shared manifest-gate test helpers; make SD3.5 LoRA self-skip visible
- **Category:** bad-pattern · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/ort_cuda.rs:63-97`, `tests.rs:779-1079`, `sd3_5_mlx_smoke.rs:375-385`
- **Finding:** `dir_from_env` returns `None` when `SCENEWORKS_ORT_CUDA_DIR`/`_CUDNN_DIR` points at a missing dir (typo indistinguishable from unset → wrong-version DLLs, deferred failure); three manifest-gating tests re-implement the same ~20-line lookup (SANA is a near-verbatim copy of SD3.5); and the SD3.5 LoRA smoke returns `Ok` when `SD3_LORA` is unset, reporting PASS having exercised nothing.
- **Impact:** Confusing mid-job cuDNN failures with no log; test boilerplate drift; a story-recorded validation run showing green without the adapter path executing.
- **Suggested fix:** `warn!` the missing env path; add `builtin_model_entry(id)` + accessors; `panic!` with the hint instead of returning.
- **Confidence:** High/Medium

#### [F-125] Delete the DWPose zip after extraction; avoid needless full-image clones (canny, SeedVR2)
- **Category:** efficiency · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/pose_jobs.rs:636-675`, `canny.rs:33-35`, `upscale_jobs.rs:891-899`
- **Finding:** `ensure_one` downloads the openmmlab `.zip`, extracts the `.onnx`, and never removes the zip (never reused); `to_gray` does `DynamicImage::ImageRgb8(img.clone()).to_luma8()` cloning ~48 MB for a 4096² source; the SeedVR2 branch `source_image.clone()`s the full image before a spawn although `source_image` is unused afterward.
- **Impact:** Roughly doubled DWPose cache footprint; avoidable multi-MB allocations per canny/SeedVR2 op.
- **Suggested fix:** `remove_file(zip_path)` after extraction; `image::imageops::grayscale(&RgbImage)`; move `source_image` into the future.
- **Confidence:** High

#### [F-126] Fold `run_upscale_with_heartbeat` into the shared helper; probe macOS GPU subprocesses concurrently
- **Category:** redundant · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/upscale_jobs.rs:678-729` (vs `progress.rs:85-121`); `gpu.rs:290-295`
- **Finding:** `run_upscale_with_heartbeat` re-implements `run_blocking_with_heartbeat` verbatim except one "Canceling image upscale." post (its doc says "Keep them in sync"); and `query_mlx_utilization` awaits `sysctl`/`vm_stat`/`ioreg` sequentially (worst-case 2+2+3 s of timeouts) on every heartbeat.
- **Impact:** The sc-8390-critical heartbeat loop exists in two copies (a fix must land twice); slow/hung probes push heartbeat latency toward the sweep budget.
- **Suggested fix:** Add an `on_cancel_acknowledged` closure param to the shared helper and delete the copy; `tokio::join!` the three probes and cache `hw.memsize`.
- **Confidence:** High

#### [F-127] Write settings.json atomically; suppress the console flash; harden port/exec parsing (desktop)
- **Category:** bad-pattern · **Severity:** Low · **Location:** `apps/desktop/src/settings.rs:203-210,481-487`, `setup.rs:275-290`, `scripts/build-sidecar.mjs:23-31`
- **Finding:** `save_settings` uses a bare `fs::write` (the pidfile writer already does temp+rename), so a torn write wipes data-dir/HF-home overrides + credential metadata; `run_capture` (nvidia-smi on Windows) omits the `CREATE_NO_WINDOW` flag `cuda_preflight` documents, flashing a console; `parse_listening_port` matches the *first* line with a loopback `addr:port` (an earlier diagnostic seeds the wrong port → 30 s dead-port poll); `build-sidecar.mjs run()` joins into an `execSync` shell string (breaks on a `PYTHON` path with spaces).
- **Impact:** Data-loss on crash mid-write; console flash on Windows; false "API did not start in time"; broken builds on spaced paths.
- **Suggested fix:** temp+rename in `save_settings`; `creation_flags(0x0800_0000)` under `#[cfg(windows)]`; anchor the port parse to the API's startup marker; switch `run()` to `execFileSync`.
- **Confidence:** High

#### [F-128] Extend desktop invariant-guard coverage; decide/supervise the API sidecar
- **Category:** bad-pattern · **Severity:** Low · **Location:** `apps/desktop/src/setup.rs:1664-1678,583-651`
- **Finding:** `desktop_never_sets_allow_open_bind` scans only 4 of 7 source files via `include_str!` (misses `update.rs`, `cred_ipc.rs`, `cuda_provision.rs`); and the API sidecar is spawn-once (a `Terminated` event is only logged, and the `is_some()` guard blocks respawn) while the GPU workers are supervised with backoff.
- **Impact:** The open-bind security invariant is enforced for only 4 files; if the API crashes mid-session the webview points at a dead origin with no recovery except quitting (Retry is inert).
- **Suggested fix:** Add the three `include_str!` entries; supervise the API like the workers, or clear `Managed.api` on `Terminated` and emit an error so Retry works.
- **Confidence:** High/Medium

#### [F-129] Stream the CUDA wheel downloads instead of buffering whole files
- **Category:** efficiency · **Severity:** Low · **Location:** `apps/desktop/src/cuda_provision.rs:210-254,282-288`
- **Finding:** Despite "Network IO is async (reqwest stream)", `response.bytes().await` buffers each wheel fully in RAM (cuDNN ≈660 MB, cuBLAS ≈530 MB), moves it into the hash task while also writing to disk, and `extract_dlls` reads each DLL fully into another Vec.
- **Impact:** First-run provisioning spikes ~1.2 GB+ transient RAM on exactly the memory-tight machines; no per-chunk progress for a 660 MB download.
- **Suggested fix:** Stream chunks to the temp file while feeding `Sha256` incrementally (free progress emits); `std::io::copy` from the zip entry to the output.
- **Confidence:** High

#### [F-130] Remove/retire orphaned desktop scripts and the scaffold gate for the retired Python runtime
- **Category:** dead-code · **Severity:** Low · **Location:** `apps/desktop/scripts/stage-onnxruntime-cuda.py:1-131`, `scripts/check-scaffold.mjs:15,18`
- **Finding:** `stage-onnxruntime-cuda.py` claims it's "invoked by build-sidecar.mjs on the Windows candle build" but nothing stages CUDA/onnxruntime on Windows anymore (replaced by `cuda_provision.rs`); and `check-scaffold.mjs` (run by CI `check.yml:56`) hard-requires the retired `apps/worker/scene_worker/runtime.py` in `requiredPaths`.
- **Impact:** A maintainer following the docstring stages ~1 GB nothing reads; the epic-8283 deletion PR will fail the scaffold gate until the entry is removed.
- **Suggested fix:** Delete the script (or mark it manual-only/superseded); remove the `runtime.py` entry from `requiredPaths` now.
- **Confidence:** High

#### [F-131] Align/pin CI action versions and Python test deps; rename shadowing PowerShell helpers
- **Category:** bad-pattern · **Severity:** Low · **Location:** `.github/workflows/release.yml:58-62,158-161`, `check.yml:45-48`, `scripts/smoke-lens.ps1:81-90`
- **Finding:** The release macOS job uses `actions/checkout@v5`+`setup-node@v5` while the windows job (same file) uses `@v4`; `pytest`/`httpx` are exact-pinned but `pillow`/`numpy`/`opencv-python-headless` are ranged (a new release changes the resolved set with no in-repo diff); and `smoke-lens.ps1` defines `Get-Job`/`Wait-Job` functions shadowing the built-in cmdlets.
- **Impact:** In-file version churn; a breaking pillow/numpy minor reds the parity lane on an unrelated PR; any code expecting the real job cmdlets silently gets the HTTP pollers.
- **Suggested fix:** Use one major (or SHA-pin per F-001) in both jobs; exact-pin all five Python deps; rename to `Get-SwJob`/`Wait-SwJob`.
- **Confidence:** High

#### [F-132] Corrupt PNG fixture + duplicated harness in the live API test files
- **Category:** bad-pattern · **Severity:** Low · **Location:** `tests/test_rust_api_contract_snapshots.py:33-72,166-180` (valid original `test_rust_api_worker_smoke.py:41-76`)
- **Finding:** `PNG_1X1` here has a stray `\x01` before the IHDR CRC (verified: stored CRC `01907753` vs correct `907753de`) — not a decodable PNG, yet every asset-upload parity test posts it; and `free_port`/`wait_for_health`/the PNG/safetensors builders are copy-pasted between the two live files and have drifted.
- **Impact:** The upload parity contract is pinned to an invalid image (if the API ever adds upload validation these fail confusingly or freeze accept-invalid behavior); drift already produced the broken fixture.
- **Suggested fix:** Replace with the valid bytes and regenerate the snapshot; move the harness helpers into `tests/rust_api_harness.py` imported by both.
- **Confidence:** High

#### [F-133] e2e/parity gates pass green if all tests skip; star-import shared harness
- **Category:** bad-pattern · **Severity:** Low · **Location:** `.github/workflows/check.yml:70-73` with `tests/test_rust_api_worker_smoke.py:99-100,426-427,485-487`; `tests/worker_runtime_shared.py:754`
- **Finding:** The `rust_api` fixture/smoke tests `pytest.skip` when `cargo`/`ffmpeg` are absent, and pytest exits 0 when all collected tests skip — the e2e step would go green having exercised nothing (latent: the toolchain is guaranteed by earlier steps today); and `worker_runtime_shared.py` exports every import via `__all__ = [... globals() ...]` consumed by seven `import *` files.
- **Impact:** A future workflow refactor dropping the toolchain turns the e2e gate into a silent no-op; no test file's dependencies are inspectable.
- **Suggested fix:** `pytest.fail` instead of skip when `CI` is set and cargo is missing; curate `__all__` + explicit imports (low priority — deletion candidates under epic 8283).
- **Confidence:** Medium/High

#### [F-134] Centralize the scattered LoRA-cap literals; fix stale "3 LoRA"/"MAX_EDIT_REFERENCES" comments
- **Category:** redundant · **Severity:** Low · **Location:** `apps/web/src/screens/generationStudio.jsx:259,340`, `PresetManagerScreen.jsx:213,809`, `components/LoraPickerField.jsx:15-17`; `crates/sceneworks-worker/src/image_jobs.rs:313-319`, `image_jobs/base.rs:836-842`, `flux2_edit_candle.rs:36-39`
- **Finding:** The per-job user-LoRA cap is the magic number `4` in two web spots and a hand-copied `MAX_JOB_LORAS = 5` mirror in `LoraPickerField`; the preset cap is a separate `5` with a stale `/3` badge; the worker constant is `5` but adjacent comments say "cap … at 3"; and `FLUX2_EDIT_CANDLE_MAX_REFERENCES = 5` claims "parity with MLX `MAX_EDIT_REFERENCES`" which is 4.
- **Impact:** Raising a cap (already done once, 3→4/5) means hunting scattered literals; the `/3` badge and the "3 LoRA"/parity comments are already wrong.
- **Suggested fix:** Serve the cap from the API (capabilities/catalog) or export single constants; reword the stale comments to reference the constant; align or explain the 4-vs-5 reference cap.
- **Confidence:** High

#### [F-135] Deduplicate the shared "Save as Preset"/asset-card-grid/DOM-test-helper blocks (web)
- **Category:** redundant · **Severity:** Low · **Location:** `apps/web/src/screens/ImageStudio.jsx:1281-1343` & `VideoStudio.jsx:410-476`; `components/AssetPicker.jsx:323-356,577-606,732-764`; `screens/ImageStudio.test.jsx:68-102` (+6 siblings)
- **Finding:** `handleSaveAsPreset` + the preset-defaults hydrate effect are duplicated near line-for-line across the two studios (`useGenerationStudio` absorbed the rest but left these); AssetPicker renders three byte-similar `role="listbox"` card grids inside one file; and `click`/`setInput`/`setSelect`/`setFileInput` + the `createRoot` scaffolding are re-declared in ≥7 test files.
- **Impact:** Validation/UX and card-layout changes drift; a test-technique fix is a seven-file edit.
- **Suggested fix:** `useSavePreset({buildDefaults})` in `generationStudio.jsx`; a `PickerCardGrid` component; a `src/testUtils/dom.js` harness.
- **Confidence:** High

#### [F-136] Remove dead web exports/scaffolds (VideoStudio shortcuts card, findReplacementModel, errorStatuses)
- **Category:** dead-code · **Severity:** Low · **Location:** `apps/web/src/screens/VideoStudio.jsx:1717-1741`, `ReplacePersonPanel.jsx:6-8`, `jobTypes.js:419`
- **Finding:** VideoStudio's "Keyboard" card advertises "Send to editor ⇧E"/"Loop preview L" but no keydown handler exists (verified — only ⌘↵ submit works); `findReplacementModel` is exported with zero references (the panel computes inline); `errorStatuses` is consumed only by its own test while `formatting.js:368` hard-codes the same terminal list.
- **Impact:** Users are told shortcuts exist that do nothing; dead exports imply APIs to "reuse".
- **Suggested fix:** Implement or delete the two shortcut rows; delete `findReplacementModel`; delete `errorStatuses` or use it where the literal list appears.
- **Confidence:** High

#### [F-137] Memoize per-render asset-catalog filters; don't mutate imported asset records
- **Category:** efficiency · **Severity:** Low · **Location:** `apps/web/src/screens/ImageEditor.jsx:980`, `VideoStudio.jsx:181-182,698-706`, `TrainingStudio.jsx:1055-1059`
- **Finding:** `imageAssets = (assets ?? []).filter(assetCanRenderAsImage)` runs on every render — including per-pointermove renders of a brush stroke/box drag — and VideoStudio repeats the pattern; `TrainingStudio.handleImport` mutates `asset.datasetOnly = true` directly on the API-returned object.
- **Impact:** Re-filtering the full catalog on every mouse-move during a stroke (jank on big projects); shared-object mutation that can leak the flag if the instance is held in context.
- **Suggested fix:** `useMemo(..., [assets])` for the filters; `{ ...asset, datasetOnly: true }`.
- **Confidence:** High

#### [F-138] Register CurveEditor window listeners once; move App refresh-ref mutation into an effect
- **Category:** bad-pattern · **Severity:** Low · **Location:** `apps/web/src/components/CurveEditor.jsx:1119-1143`, `App.jsx:1253-1261`
- **Finding:** The CurveEditor `pointermove`/`pointerup` effect has no dep array, so both window listeners are removed and re-added on every render (including every drag-tick `onChange`); and `refreshDataRef.current = refreshData` (×9) executes in the App render body, which React documents as unsafe (a discarded concurrent render can leave refs pointing at uncommitted closures).
- **Impact:** Listener churn per drag tick with a constantly re-created stale closure; latent ref-staleness under StrictMode/concurrent features.
- **Suggested fix:** Register listeners once (`[]` deps) reading points via a ref (or SVG pointer capture); move the ref assignments into a `useEffect` or make the functions `useCallback`-stable (also fixes F-009).
- **Confidence:** High/Medium

#### [F-139] `packages/shared` Python + flag-gated web scaffolds are dead-but-rotting
- **Category:** dead-code · **Severity:** Low · **Location:** `packages/shared/sceneworks_shared/`; `apps/web/src/screens/ModelManagerScreen.jsx:116,1134-1237`, `ImageEditor.jsx:119-122,3092-3102`
- **Finding:** `sceneworks_shared` (~430 lines, incl. a parallel `project.db` migration/indexing impl that can drift from the Rust one) is imported only by the retired Python worker + its tests; and ~130 lines of model-import form/handler/state sit behind the compile-time-false `MODEL_IMPORT_ENABLED` (epic 7080) while ImageEditor maps over a permanently-empty `UPCOMING_TOOLS`.
- **Impact:** Maintained code that never runs; documented placeholders that bit-rot untested and inflate two huge files.
- **Suggested fix:** Fold `packages/shared` removal into epic 8283; move the dead import form behind a flag-imported separate file (or accept per the epic-7080 note).
- **Confidence:** Medium/High

#### [F-140] Decompose the widest React prop lists and overloaded signatures
- **Category:** readability · **Severity:** Low · **Location:** `apps/web/src/screens/TrainingStudio.jsx:1503-1622`, `hooks/useTraining.js:84-103`
- **Finding:** `DatasetEditorPanel` (~55 props) and `ConfigureJobPanel` (~45 props) were extracted verbatim (sc-4199) leaving all state in the screen and threading `DatasetDoctorReadout` props twice; `setTrainingDatasetItemQualityAck`'s 4th param is `optionsOrProjectId` (string=projectId, object=options) — a type-switching signature unique in the codebase and easy to call wrong (silently uses the active project).
- **Impact:** Adding one dataset action touches four files; a wrong-typed arg fails silently.
- **Suggested fix:** Group cohesive prop bundles into objects (or a `useDatasetEditor` hook); take a single options object.
- **Confidence:** High

#### [F-141] Split the 10,724-line `main.test.jsx` monolith
- **Category:** readability · **Severity:** Low · **Location:** `apps/web/src/main.test.jsx:1-10724`
- **Finding:** A single test file covers App wiring, jobTypes enums, SSE, auth, and notices at ~10.7k lines — larger than App.jsx and all hooks combined; contributors bolt new assertions on because there's no per-domain home (the `errorStatuses` test lives here, not beside jobTypes.js).
- **Impact:** Slow to navigate/run selectively; failures hard to localize.
- **Suggested fix:** Split along the hook/screen seams (App.auth.test, App.sse.test, jobTypes.test, …); vitest needs no config change.
- **Confidence:** High

#### [F-142] Prune the unbounded App refresh-tracking map
- **Category:** efficiency · **Severity:** Low · **Location:** `apps/web/src/App.jsx:580,1040-1055`
- **Finding:** `generatedAssetRefreshesRef` gains an entry per asset-producing job and never evicts for the session lifetime.
- **Impact:** Slow unbounded growth (small objects keyed by job id) in multi-day sessions.
- **Suggested fix:** Delete the entry when the job reaches a terminal status in `handleJobUpdated`.
- **Confidence:** High

#### [F-143] Misleading canny error / control-repo error messages point at the wrong field
- **Category:** readability · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/image_jobs/strict_control.rs:365-369`, `zimage.rs:241-245,493-497`, `qwen.rs:153-157`
- **Finding:** The canny no-source error says "requires a source image (advanced.controlImage)" but the auto-derive source is `sourceAssetId`/`referenceAssetId` (the depth error gets this right); and the "weights not found (download …)" messages interpolate local repo constants that duplicate the `STRICT_CONTROL_ENGINES` table, so a table repoint leaves the message telling users to download the wrong repo.
- **Impact:** A user/agent acting on the error attaches the wrong field / downloads the wrong repo.
- **Suggested fix:** Match the depth wording; build the message from `strict_control_default_repo(ENGINE_ID)` and delete the local repo constants.
- **Confidence:** High

#### [F-144] Rename the misplaced `flux2_*` shared helpers and `zimage_identity`/`load_mask_asset_image` duplicates
- **Category:** redundant · **Severity:** Low · **Location:** `crates/sceneworks-worker/src/image_jobs/flux2.rs:1-27,83-112,771-813`, `zimage.rs:617-660`, `sdxl.rs:76-83`
- **Finding:** `Flux2Grouping`/`flux2_grouping`/`flux2_edit_reference_ids` are the grouping logic for the Qwen/SenseNova lanes too (misdirected navigation); `flux2_identity_strength`/`resolve_flux2_identity_init` are line-for-line copies of the Z-Image pair; `load_mask_asset_image` is a pass-through alias of `load_reference_image`.
- **Impact:** A reader auditing Qwen behavior must know to look in flux2.rs; identity-gate changes must be made twice.
- **Suggested fix:** Rename to `EditGrouping`/`edit_*` and move to base.rs; one shared `resolve_identity_init`; inline the mask alias.
- **Confidence:** High

---

## Informational

#### [F-145] SSE auth ticket travels in the query string
- **Category:** security · **Severity:** Info · **Location:** `apps/rust-api/src/events.rs:93-103`, `apps/web/src/api.js:40-46`
- **Finding:** `/api/v1/jobs/events` is public and gated by a single-use `?ticket=` with a 30 s TTL (EventSource can't set headers); query strings can persist in proxy/access logs. The server's own auth-rejection log deliberately logs only the path.
- **Impact:** A logged ticket is replayable for up to 30 s if unconsumed; low given single-use + TTL. Recorded so nobody extends the pattern to the long-lived token.
- **Suggested fix:** None urgent; keep tickets short-TTL/one-shot; optionally pin issuance to the peer IP.
- **Confidence:** High

#### [F-146] Loopback trust grants token bypass to every local OS user
- **Category:** security · **Severity:** Info · **Location:** `apps/rust-api/src/auth.rs:87-89`, `lib.rs:203-209`
- **Finding:** With `SCENEWORKS_TRUST_LOOPBACK` set (desktop default in LAN mode), any process on the machine bypasses the token via 127.0.0.1 — documented and deliberate (worker + embedded UI have no password; Docker stays fail-closed).
- **Impact:** On multi-user machines, local users get full API control. Accepted design tradeoff.
- **Suggested fix:** No change required; note the multi-user caveat in the LAN docs, or move worker auth to a locally-issued secret so loopback trust can retire later.
- **Confidence:** High

#### [F-147] `atomic_write`/`write_secret_file` never fsync before rename
- **Category:** bad-pattern · **Severity:** Info · **Location:** `crates/sceneworks-core/src/store_util.rs:30-43`, `credentials.rs:109-132`
- **Finding:** The temp-then-rename writers don't `sync_all()` the temp (or the dir) before `rename`; on power loss a filesystem may persist the rename before the data, leaving a zero-length "atomically written" manifest/sidecar/credentials file.
- **Impact:** A rare corruption window for every JSON sidecar and the credentials file after a crash — the failure the atomic pattern is meant to preclude. Desktop risk profile keeps this Info.
- **Suggested fix:** `write_all` + `sync_all` on the temp before `rename` (accept the latency, or gate to the credentials/manifest writers).
- **Confidence:** High

#### [F-148] Per-call SQLite connection + one global mutex serializes all jobs-store traffic
- **Category:** efficiency · **Severity:** Info · **Location:** `crates/sceneworks-core/src/jobs_store.rs:194-197,1422-1458`
- **Finding:** Every `JobsStore` method opens a fresh `Connection` (re-running the WAL/FK pragmas) and takes the single process-wide `Mutex<()>` — including pure reads (`list_jobs`, `get_job`, `queue_summary`). WAL exists to let readers run concurrently with a writer, but the mutex forbids it.
- **Impact:** Latency coupling between polling and claim/sweep writes; per-call open cost. Acceptable at desktop scale — noted because the WAL comment implies concurrency the lock removes.
- **Suggested fix:** Let read-only methods skip the mutex (single SELECTs; WAL handles isolation), or hold a pooled connection.
- **Confidence:** Medium

#### [F-149] Truncated slugify can leave a trailing dash; retire the permanently-None routing seam
- **Category:** readability · **Severity:** Info · **Location:** `crates/sceneworks-core/src/slug.rs:13-21`, `jobs_store.rs:3047-3049,2839`
- **Finding:** `slugify` trims trailing dashes *before* `truncate(max_length)`, so mid-boundary truncation can reintroduce a trailing `-` (`"my project"`/max 3 → `"my-"`); and `torch_only_image_model_epic` unconditionally returns `None` (every family ported) with `MacFeatureSupport::unsupported` `#[allow(dead_code)]` — consciously retained "vocabulary seams".
- **Impact:** Cosmetic filenames; slight reader overhead. Both documented decisions.
- **Suggested fix:** Re-run the trailing-dash trim after truncate; keep the seams (docs justify them) or inline the `None`.
- **Confidence:** High

#### [F-150] Miscellaneous worker readability/redundancy nits
- **Category:** readability · **Severity:** Info · **Location:** `crates/sceneworks-worker/src/scail2_masks.rs:78-99`, `person_jobs.rs:679-680`, `pose_jobs.rs:946-948`, `depth.rs:47-93`, `lib.rs:593-594,932`, `flux1_control_mlx_smoke.rs:23`, `footprint_measure.rs:162-231`, `ort_cuda.rs:104-120`
- **Finding:** `paint_driving_masks` re-searches `order` via `position()` for an index the loop already has (O(n²)); `run` clones C2PSA output twice (`block10`/`b10`); a preview-write IO failure is labeled `InvalidPayload`; the two `depth_control_image` bodies are identical apart from the backend crate; `run_utility_job` posts an Idle heartbeat then `poll_once` immediately sends a second; the flux1 smoke has a redundant inner `#![cfg]`; `measure_footprint`'s `(u64,u64)` return is discarded by all callers; and `preload_cuda_dylibs` mutates process-global `PATH` (becomes `unsafe` in edition 2024).
- **Impact:** Individually trivial; collectively the same "duplicate/mislabel/redundant-call" noise that accretes.
- **Suggested fix:** Enumerate-with-index; drop `b10`; map to `Io`/`Engine`; cfg-alias the depth backend; `mark_sent()` after a job; drop the inner cfg; return `()`; pre-write the edition-2024 safety comment.
- **Confidence:** High

#### [F-151] Worker efficiency notes accepted as-is (SAM2 pass-2 re-encode, depth per-call reload, image streaming result)
- **Category:** efficiency · **Severity:** Info · **Location:** `crates/sceneworks-worker/src/person_segment.rs:212-263`, `depth.rs:54-55`, `image_jobs/base.rs:2954-2966`, `image_jobs/instantid.rs:145-175`
- **Finding:** SAM2 pass-2 rebuilds tracking state over all frames and re-propagates on any drift (likely required — state exposes no reset); `DepthAnythingV2::from_dir` loads ~100 MB per depth-controlled generation (no cache); every `GenEvent::Step` reserializes the full `assetWrites` result (O(images²·steps) cloning); the InstantID angle-collection DB lookup runs twice per job.
- **Impact:** Bounded extra work (≤24 frames; one depth load; harmless at current image/step counts) — noted for when sizes grow.
- **Suggested fix:** Only if the engine crates later expose reuse; send the streaming result on `Image`/`Decoding` events only; resolve the angle collection once and thread it through the plan.
- **Confidence:** Medium

#### [F-152] Six near-identical `#[cfg(test)]` load wrappers; smoke-harness one-tier gaps folded into F-064/F-122/F-123
- **Category:** dead-code · **Severity:** Info · **Location:** `crates/sceneworks-worker/src/image_jobs/{base.rs:934,zimage.rs:126,flux2.rs:835,qwen.rs:48,flux1_control.rs:205}`
- **Finding:** Each control lane keeps a `#[cfg(all(target_os="macos", test))]` load wrapper used only by `#[ignore]`d real-weight smokes — copies of `spec + gen_core::load + map_err`; 70 of tests.rs's 145 image tests are `#[ignore]`d manual smoke harnesses living in the production crate.
- **Impact:** No runtime cost; the smoke bulk (roughly half of tests.rs) obscures the CI-run tests.
- **Suggested fix:** One generic `#[cfg(test)] fn load_control_engine(engine_id, spec)`; optionally split real-weight smokes into a `tests_smoke.rs` include.
- **Confidence:** High

#### [F-153] Duplicated e2e test scaffolding + `_tmp_path` dead param in media_jobs
- **Category:** dead-code · **Severity:** Info · **Location:** `crates/sceneworks-worker/src/media_jobs.rs:2331-2343,2688-2754,2943-3032`
- **Finding:** `mux_with_crossfades`'s `_tmp_path: &Path` is never used (the single-filter-graph rewrite removed the intermediate file), and the SAM2/SAM3 `#[ignore]` E2E tests each carry ~100 identical setup lines.
- **Impact:** Dead param + copy-paste test setup.
- **Suggested fix:** Drop the param and its pass-through; extract a test-local `detect_and_assemble` helper.
- **Confidence:** High

#### [F-154] Remote Google Fonts dependency; theme accent-id list duplicated in the pre-paint script
- **Category:** security · **Severity:** Info · **Location:** `apps/web/index.html:9-14`, `apps/web/public/theme-init.js:13-14` vs `src/accents.js:7-15`
- **Finding:** The shell loads Plus Jakarta Sans / JetBrains Mono from `fonts.googleapis.com`/`fonts.gstatic.com` at startup (leaks usage signal to a third party, fallback-renders offline — at odds with the local-first posture and the strict-CSP intent noted in theme-init.js); and `ACCENT_IDS` is hand-copied into the pre-paint script with a "keep in sync" comment.
- **Impact:** Third-party signal on every launch; a one-frame accent flash for palettes not mirrored into the pre-paint list.
- **Suggested fix:** Self-host the two OFL fonts in `public/` and drop the preconnects; generate `theme-init.js` from `accents.js` at build time.
- **Confidence:** High

#### [F-155] Retired Python worker tree — inventory, liveness, and secrets pass
- **Category:** dead-code · **Severity:** Info · **Location:** `apps/worker/` (34,074 py LOC), `packages/shared/`
- **Finding:** No runtime surface invokes this code (no `python -m scene_worker` spawn in any Rust/JS, no Docker COPY, no CI execution, no Tauri bundling — verified via a repo-wide grep of spawns/imports/COPYs/bundling); remaining references are comments, dev spike scripts, and the test suite. Largest modules: `image_adapters.py` 5,841, `training_adapters.py` 3,306, `video_adapters.py` 2,868. A full token scan (hf_/sk-/ghp_/api_key/Bearer) found no real secrets — the one `api_key="sk-ant-xxx"` is a vendored upstream docstring placeholder; hardcoded URLs are all public endpoints. (Git-history scanning not performed — current tree only.)
- **Impact:** 34k+ LOC of unmaintained code that reads as live (extensive Rust doc-comment cross-references) plus stale torch/diffusers pins that will trip vulnerability scanners.
- **Suggested fix:** Execute epic 8283 deletion (after extracting the live gates per F-059); update the Rust doc comments that cite `apps/worker/scene_worker/*.py` as the porting reference.
- **Confidence:** High

---

## Themes and systemic observations

1. **Backend-twin and per-lane copy-paste is the dominant maintainability risk, and drift is already measurable.** MLX-vs-Candle twins (SAM3 modules ~300 duplicated lines, `depth_control_image`, YOLO preprocessors, sensenova VQA/interleave ~55% verbatim, the video SCAIL-2 conditioning/extend/bridge), per-model image/video lanes (steps/guidance resolvers ×9, likeness-gating blocks ×3, edit-fit geometry ×3), the web layer (four job-result resolvers, five poll loops, three character-membership predicates, two upscale tables), and the API (upload writers ×3, sweeps ×4) all share the property that the copies are mutually cfg-exclusive or file-separated, so drift is invisible to any single build/CI lane — and several have already diverged behaviorally (kolors candle control skips `controlMode`, base Z-Image control lost likeness scoring, the corrupt-PNG fixture, the `UPSCALE_ENGINES` field split, `MAX_EDIT_REFERENCES` 4-vs-5). The proven cure is already in the codebase: `strict_control.rs`, `candle_strict_control.rs`, `useGenerationStudio`, `openpose_skeleton`/`scail2_masks`, and the extracted pure modules (ideogramCaption, colorGrade, tierSuggestion) are single-sourced and don't drift. Adopt a standing rule — pure/tensor-free logic lives in a shared always-compiled module; only the backend seam is cfg-gated — and extract the ~6 remaining shared pieces named above.

2. **"Streaming job abandons live GPU work / goes heartbeat-silent" is a structural defect class, not a set of local bugs.** Every streaming consumer (video, training, caption, prompt-refine, both analysis loops, and the shared `run_blocking_with_heartbeat`) propagates a failed `update_job`/`heartbeat` POST via `?` without tripping the engine `CancelFlag` or aborting the blocking task; child processes (`ffmpeg`, `hf`) leak on the same paths; and three fresh instances of the *silent* variant exist where long compute isn't routed through the keepalive helper at all (SAM2/SAM3 propagate, the pose post-render loop, image-detail). One shared cancel-and-join guard + `kill_on_drop(true)` closes the abandon class; a convention "no `spawn_blocking` in a job handler except via the keepalive helper" closes the silent class. Fixing instances individually will keep missing siblings — this class has already recurred across sc-8200/sc-8390.

3. **Path-confinement and hardening are genuinely strong but applied campaign-style, leaving stragglers at exactly the sites that hand-rolled resolution.** The confinement primitives (`safe_project_path`, `is_safe_id`, `normalize_app_managed_*`, `resolve_dataset_item_path`) and the SSRF/token-compare hardening are excellent and tested — but the same omission recurs wherever a handler joined a payload string itself instead of calling the helper: LoRA/model import `sourcePath` (F-002), dataset-upscale read+write (F-040), `controlWeights.filename`/`pidCheckpoint.filename` (F-019), timeline ids (F-069), pose relative sources (F-073), person_replace masks (F-074), trainer `file_name` (F-076), and the lexical-only variants of `ensure_path_under` (F-075). Likewise the sc-42xx hardening sweep left `create_job`'s `randomblob` id-gen, training_store's missing `busy_timeout`, the heartbeat ownership guard, and the credentials store's total lack of locking. A "payload string → filesystem/subprocess" audit checklist (and a grep/clippy CI check for bare `project_path.join(<dynamic>)`) would close the class before epic 4484 widens the trust boundary — this is the same theme flagged in the 2026-06-15 review, now with the four prior High instances fixed and a fresh crop at the newer surfaces.

4. **The remote-access (epic 4484) auth retrofit is incomplete at every "headerless" seam.** Everything that can send a header was converted (`apiFetch`, ticketed SSE), but every browser-native request path was left unauthenticated: `<img>`/`<video>`/anchor downloads/`DocumentReader`'s `fetch` all 401 in remote mode (F-008), the login password is wired live into the API token so each keystroke floods the API (F-007), the `ui-preferences` PUT is method-blind public (F-067), there's no verify-endpoint rate limit (F-068), and the compose web service LAN-exposes an unauthenticated Vite dev server (F-062). The root cause is uniform — the app was only ever exercised in loopback-trusted mode, where auth is invisible — so a single "remote-browser, token-set" test pass would surface the whole cluster.

5. **God modules concentrate risk and are where the subtle bugs hide.** `jobs_store.rs` (8.8k lines: SQL store + routing policy + five parallel model-catalog lists + UI gating — its own comments record two shipped routing bugs from missed-list edits), `App.jsx` (~2.4k: SSE + auth + a hand-mirrored 150-key context dep array where the F-009 memo bug lived), `ImageEditor.jsx` (~3.9k: eight tools + a ref-mirror pattern that exists *because* the state is too entangled), the worker `tests.rs` (3.4k) and web `main.test.jsx` (10.7k), plus the `run_*_job` dispatch god functions. Splitting these along the seams they already suggest (routing submodules with one capability table; `useJobEvents`/`useAccessGate`; per-tool editor modules; per-domain test files) removes both the review cost and the hand-synced-invariant class of bug.

6. **Invariants are commendably encoded as executable guards — but the guards enumerate frozen lists that silently go stale.** The codebase turns past incidents into checks (ACL drift tests, the gen-core skew gate, the open-bind scan, compose assertions, the hookStability test) better than most — yet nearly every guard failure found is the same shape: a hardcoded list that didn't grow with the code (the `include_str!` module list missing 3 files, `check-scaffold.mjs` requiring the retired `runtime.py`, sync-version's unenforced atomicity, the hookStability test omitting the one hook that broke the memo, the "keep in sync" constant/comment mirrors). Where a guard enumerates files/values, derive the list (glob the src dir, read all four version manifests) instead of hardcoding it.

7. **The Python retirement is real but its deletion has under-estimated blast radius.** The runtime Python is verifiably dead, but epic 8283's "delete `apps/worker`" will fail CI (the scaffold gate requires `runtime.py`) and silently shed live coverage unless the pre-work is sequenced: the e2e gate's protocol *client* is the retired worker, live `builtin.models.jsonc` audits are embedded in dead-worker test files, and `packages/shared` carries a parallel `project.db` migration impl. Extract the live gates and reimplement the e2e client as pure-HTTP *before* the deletion.

## Coverage notes

- **Read fully:** all live Rust (`apps/rust-api` 26 modules + lib/main, `apps/rust-worker`, `apps/desktop` 7 sources + Tauri config/capabilities/build.rs, `crates/sceneworks-core` 26 modules, `crates/sceneworks-image-quality`, and the entire `crates/sceneworks-worker` job + infrastructure tree incl. `video_jobs.rs` all 9,503 lines); all `apps/web/src` (screens read line-by-line, shell/hooks/components/context/training/data + config); `packages/schemas` + `packages/shared`; all six CI workflows, `docker-compose.yml` + both Dockerfiles, all `scripts/`, root manifests, `pytest.ini`, `rust-toolchain.toml`; the live `tests/` harness (`conftest`, both live API suites, `worker_runtime_shared`) in full.
- **Surveyed, not line-audited:** the worker `tests.rs` (3,435 lines) and image_jobs `tests.rs` (7,547 lines) — helper layout, stub servers, and gates checked but not every `#[ignore]`d smoke body; `apps/rust-api/tests.rs` (10,006 lines, 126 tests) — auth/ticket/credential protection tests confirmed present, no test asserting the `ui-preferences` PUT posture; web `main.test.jsx` (10,724 lines) — sampled for dead-export verification; the ~12.5k LOC of retired-worker Python tests — targeted assertion-free/skip-condition/swallowed-exception scans + spot reads, proportionate to their dead-code-under-test status.
- **Inventory + liveness + secrets pass only (not a line review):** `apps/worker/` (~34k retired Python) — verified runtime-dead via a repo-wide spawn/import/COPY/bundle grep; secrets scan of the current tree clean (git-history not scanned).
- **Excluded:** `styles.css` (~10k lines, design-only), binary assets (icons, PNGs, `screenshot.png`, the one `.safetensors`), `Cargo.lock`/`package-lock.json`, the ~20 archived `scripts/spikes/**` research scripts (grepped for shell=True/os.system/secrets — clean), and the individual model rows of `config/manifests/builtin.models.jsonc` (~5.7k lines — the validator was reviewed, not each row). Cross-crate security claims were verified by reading the actual `sceneworks-core` implementations the API/worker delegate to (`project_file`, `pose_preview_path`, `assert_allowed_lora_source`, `lora_url` SSRF guard) rather than assumed.
- **Prior-review status:** the three 2026-06-15 High findings I re-checked are **fixed** in source — the `mediaPath` traversal guard is present (`project_store.rs:1418`, sc-5721), `load_source_video_frames` now uses `safe_project_path` (`video_jobs.rs:5972`), and open binds now require an explicit `SCENEWORKS_ALLOW_OPEN_BIND=1`. The LoRA/import-path confinement gap (WKA-002) is partially addressed (read/target paths confined) but the import *source* path (F-002) remains open.
