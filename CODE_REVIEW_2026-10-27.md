# Full Codebase Review — SceneWorks — 2026-10-27

Branch `claude/sceneworks-codebase-review-j3yimb`, reviewed at the tip as of 2026-10-27. Prior reviews: `docs/code-review/CODE_REVIEW_2026-05-17.md`, `_2026-07-01.md`, `_2026-07-12.md`.

## Executive summary

- **Repository at a glance (Part 1, Rust backend):** Rust workspace (`crates/sceneworks-worker`, `-core`, `-memory-adapter`, `-mcp`, `-image-quality`; `apps/rust-api`, `apps/desktop`, `apps/rust-worker`), roughly 580 source files and ~900k lines including very large inline test modules. Axum API + native MLX/Candle job workers + Tauri shell; inference engines come from the pinned `SceneWorks/inference` repo.
- **Coverage:** two passes. **Part 1** — the Rust backend (11 parallel read-only reviewers, F-001…F-188). **Part 2** — the web app (`apps/web`, ~225k LOC incl. ~350 test files), `scripts/`, `tests/`, `docker/`, `.github/workflows`, `packages/schemas` and repo hygiene (7 reviewers, F-189…F-274). Excluded: `docs/` prose, `vendor/` (only noted whether still needed), generated files, binaries. Tests were surveyed structurally, not line-audited. Reviewers read and grepped; **nobody ran `cargo`, `npm`, ESLint or knip**. The Part 2 reviews were shallower than Part 1 on the largest files; see *Coverage notes* for per-area gaps.
- **Headline (Part 1):** no Critical findings. Your instinct about duplication is right and it is the dominant theme: the pattern is *"a helper is hardened in one place, then re-copied unhardened elsewhere"*, usually because the original is `pub(crate)` or the shared module (`smoke_support`, `analysis_jobs_common`, `store_util`, `payload`) was only partly adopted. Several copies have already drifted in behaviour (MLX vs Candle floors, ETag/path confinement, atomic-write durability, repo-id validation, env-flag parsing). Architectural drift shows up mostly as *rules that exist only as comments*: `architecture_tests.rs` does not enforce layering or cfg rules, and the web `fallbackModels` seed has already drifted from the manifest. The most important security item is a combined one: **loopback trust with no `Origin`/`Host` validation** on `/api/*`, and the `/mcp` rebinding defence being switched off in the desktop's LAN mode. Most items from the 2026-07-12 review are fixed (26 of 28 spot-checked); the two big structural ones are still open and worse (`image_jobs/base.rs` is now 19.6k lines, the 1,292-line image job function remains).
- **Counts (274 findings):** Critical: 0 | High: 7 | Medium: 118 | Low: 137 | Info: 12.  Part 1 (Rust): Critical: 0 | High: 7 | Medium: 86 | Low: 89 | Info: 6.  Part 2 (web/tooling): Critical: 0 | High: 0 | Medium: 32 | Low: 48 | Info: 6.
- **Findings by category:** redundant 98, architecture-drift 55, bad-pattern 38, readability 29, security 23, dead-code 21, efficiency 10.
- **How to read this report:** findings are severity-ranked and renumbered `F-001…`. Each keeps its **Source ID** (e.g. `W3-003`) because findings and the per-reviewer notes refer to each other by source ID. Prefixes: W = worker, C = core, A = rust-api, X = memory-adapter/mcp/desktop/vendor, D = cross-crate (Part 1); B = web app, S = scripts/CI/docker/hygiene (Part 2). Confidence is the reviewer's; the *Lead verification* line marks the four I re-checked against the code myself.
- **Headline (Part 2):** the web app has no High or Critical findings and a good security posture (no HTML sinks, `postMessage`/iframe surface or unsafe links found in the areas read; ticketed SSE/media; write-only credential inputs). The problems are structural: god components that keep growing (`App.jsx` 2.4k→4.3k, `ImageStudio` 3.2k→4.9k lines), helpers/hooks extracted once but never migrated to (the `api/` layer is bypassed at ~236 call sites), server-owned rules re-implemented client-side (license acks in `localStorage`, training validation that ignores the advertised limits, a test-only model catalog mirror), unguarded polling/import loops, and a parallel Simple-UI overlay stack without the shared modal's focus handling. `scripts/` (139k LOC) re-implements hashing/canonical-JSON helpers per harness; the dev web Docker image is missing build inputs; 192 MB of raw calibration frames are tracked.


# Part 1 — Rust backend


### Critical findings — Rust backend

*None.*


### High findings — Rust backend

#### [F-001] No Origin/Host validation on `/api/*`: CSRF and DNS-rebinding reach the loopback-trusted API
- **Source ID:** A2-001
- **Category:** security
- **Severity:** High
- **Location:** `apps/rust-api/src/auth.rs:79-166` (access_control), `apps/rust-api/src/lib.rs:1493-1509` (only `/mcp` gets a Host allow-list), `lib.rs:2265` (`POST /worker/restart`), `lib.rs:1585-1600` (catalog pause/resume/analyze), `lib.rs:1608,1669,1687,2067` (2 GiB multipart uploads), `jobs.rs:623-628`, `events.rs:54-68`
- **Finding:** The desktop sets `SCENEWORKS_TRUST_LOOPBACK`, so any request whose TCP peer is 127.0.0.1 skips the token (`loopback_trusted`). Nothing checks `Origin` or `Host` on `/api` (grep for ORIGIN/HOST finds only CORS config and the MCP allow-list; the MCP comment at lib.rs:1501-1505 admits `access_control` "performs NO Host/Origin validation"). CORS only blocks *reading* responses and only forces preflight for non-simple requests, and a body-less `POST` or a `multipart/form-data` `POST` is a simple request, so any web page open in the user's browser can fire `POST /api/v1/worker/restart`, pause/resume/analyze catalogs, and stream multi-GB multipart uploads into `data/cache`. `retry_job` and `create_event_ticket` read the body with `to_bytes`+`serde_json::from_slice` (not `Json<T>`), so they accept `text/plain` and skip the content-type guard the `ApiJson` routes get. DNS rebinding (attacker hostname resolving to 127.0.0.1) is not blocked on `/api` at all, which makes the full API readable and writable, including path-taking routes (attach catalog, LoRA/model import from local path).
- **Impact:** A malicious page can drive state-changing actions and fill the disk on any desktop install without any token. With rebinding it is full API takeover, which includes reads of project files and local-path import/attach routes.
- **Suggested fix:** Add one middleware ahead of `access_control` that (a) rejects any request with an `Origin` header not in `cors_origins`/the app's own origin, (b) for loopback-trusted peers validates `Host` against the same allow-list `mcp_allowed_hosts` already computes (lib.rs:1507). Require `Content-Type: application/json` in the two hand-parsed handlers (or parse through `ApiJson`). Add a test that a loopback request with a foreign `Origin`/`Host` gets 403.
- **Confidence:** Medium (depends on browser Private Network Access behavior and the desktop webview; the missing server-side checks themselves are certain)
- **Lead verification:** Verified by lead against `apps/rust-api/src/auth.rs:133-139,185-199`: loopback peers skip the token check and nothing in `apps/rust-api/src` inspects `Origin`/`Host`; CORS only limits response reads. Same root cause as X1-001. Exploit not attempted.

#### [F-002] MCP DNS-rebinding defense is OFF in the desktop's LAN mode, where loopback trust is also ON
- **Source ID:** X1-001
- **Category:** security
- **Severity:** High
- **Location:** `crates/sceneworks-mcp/src/lib.rs:67-92,160-180`; `apps/desktop/src/setup.rs:1296-1345,1386`; `apps/desktop/src/net.rs:63-83`; `docs/mcp-server.md:148-157`
- **Finding:** With remote access enabled, the desktop binds `0.0.0.0` (`decide_api_bind_env`) AND sets `SCENEWORKS_TRUST_LOOPBACK=true`, but never sets `SCENEWORKS_MCP_ALLOWED_HOSTS`. `mcp_allowed_hosts` therefore returns an empty vec for the wildcard bind, and the constructor calls `disable_allowed_hosts()`. The justification ("a browser doing DNS rebinding is a remote (non-loopback) peer that cannot present the token") is wrong: a page rebinding `attacker.example` to 127.0.0.1 reaches the API from the victim's own machine, so the peer IP IS loopback and `access_control` skips the token. The remote port is the fixed `DEFAULT_REMOTE_PORT`, so it is guessable.
- **Impact:** Exactly the F-040 attack (job submission, ticket minting, ticketed file reads via `/mcp`) is reopened for every desktop user who turns on LAN access. The same loopback-trust reasoning applies to the other `/api/v1` routes, so the rust-api owner should check them too.
- **Suggested fix:** For a wildcard bind, do not fall back to allow-all. Enumerate the interface IPs and hostname (the desktop already does this in `net::lan_addresses`) and always allow loopback plus those, or have the desktop export `SCENEWORKS_MCP_ALLOWED_HOSTS` from `lan_addresses()`. Also add an Origin check that rejects any request carrying a non-matching `Origin`, and fix the doc and comment claims.
- **Confidence:** Medium (mechanism verified in code; exploitability depends on browser rebinding behavior and the user having enabled LAN mode).
- **Lead verification:** Same root cause as A2-001 (loopback trust + no Host/Origin validation); desktop sets `SCENEWORKS_TRUST_LOOPBACK` at `apps/desktop/src/setup.rs:1386`. Fix both together.

#### [F-003] MLX and Candle `active_component_floors` have diverged semantics (`any` vs `all`)
- **Source ID:** W2-001
- **Category:** architecture-drift
- **Severity:** High
- **Location:** `crates/sceneworks-worker/src/mlx_fit_gate.rs:1138-1146`, `crates/sceneworks-worker/src/candle_memory_strategy.rs:842-855`
- **Finding:** Both gates decide which declared `ComponentPrecisionFloor`s attach to the selected tier. MLX returns the whole declared slice if ANY floor applies to the selected quant. Candle returns it only if the slice is non-empty and ALL floors apply. They sit beside identical `declared_component_floors` copies (candle 831-840, mlx 838-849). Neither cites the other, so there is no sign the difference is intentional.
- **Impact:** A provider with floors that only cover some quants gets a different floor set per backend. One backend can then price a tier with floors it does not need, or drop floors it does. That changes the predicted peak and the admit/refuse outcome on one backend.
- **Suggested fix:** Move `declared_component_floors`/`active_component_floors` into the shared `memory_strategy.rs` with one documented semantics and one test matrix. If the divergence is intended, say so in both doc comments.
- **Confidence:** Medium (the code difference is certain; whether any shipped provider declares partial-coverage floors was not checked).
- **Lead verification:** Verified by lead: `candle_memory_strategy.rs:842-855` uses `all(applies_to)`, `mlx_fit_gate.rs:1138-1146` uses `any(applies_to)`. Whether any shipped engine declares partially-applicable floors (so admission actually differs) was not checked.

#### [F-004] Authenticated source-URL download loop is a ~150-line twin of the HF file download loop
- **Source ID:** W3-001
- **Category:** redundant
- **Severity:** High
- **Location:** `crates/sceneworks-worker/src/downloads.rs:1111-1249` (`download_file_inner`) vs `crates/sceneworks-worker/src/downloads.rs:1539-1730` (`download_source_url`)
- **Finding:** Both implement the same state machine: Range resume, 206/200 append-or-truncate, `select!` over `response.chunk()` and a progress interval, stall watchdog via `DownloadResumeState`, post-transfer size check with resume, and delete-if-oversized. They have already drifted: only the source-URL copy has a byte cap, and only the HF copy goes through `DownloadLock`, verify and `.partial` promotion.
- **Impact:** Every fix to resume/stall/size handling has to be made twice. The two copies already differ on 416 handling and on what happens when the total size is unknown (see W3-006).
- **Suggested fix:** Extract one `stream_to_file(response_factory, dest, expected_size, max_bytes, progress, resume)` core. `download_file_inner` and `download_source_url` then supply only the request builder (redirect-following vs plain GET) and the cap.
- **Confidence:** High

#### [F-005] Stream/heartbeat/cancel supervision loop is hand-cloned in 4 more places despite the "one place" scaffold
- **Source ID:** W4-001
- **Category:** redundant
- **Severity:** High
- **Location:** `crates/sceneworks-worker/src/analysis_jobs_common.rs:14-18,295-345` (the shared copy); clones at `caption_jobs.rs:397-515`, `prompt_refine_jobs.rs:1574-1670`, `control_training_jobs.rs:258-330` (outside area), `training_jobs.rs:3716-3790` (outside area)
- **Finding:** `stream_batched_records` owns the `CancelJoinGuard` + `select!{rx / heartbeat tick}` + `cancel_requested_peek`/`shutdown_requested` latch + deferred `mark_job_canceled` after join + `cancel_and_join` on error. Caption, prompt-refine, control-prep and training each re-implement the same ~100-line skeleton, with near-identical sc-8804/sc-9618/sc-11189/F-017/F-115 comment blocks pasted in each. The module header claims the cancel handling "can only ever be fixed in one place", which is false today.
- **Impact:** Every cancel/shutdown/terminal-write fix (already needed at least four times: sc-8804, sc-8917, sc-9618, sc-11189) must be applied in five places; a miss silently reintroduces the "worker row freed while GPU still busy" bug. The header comment misleads the next maintainer.
- **Suggested fix:** Generalise `stream_batched_records` over the event type (`T`) with an `on_event: FnMut(T) -> Option<ProgressRequest>` plus an optional extra coalesced-progress arm (caption/prompt-refine's `progress_interval` + watch channel). Move caption, control-prep and the non-result-returning part of prompt-refine onto it; at minimum update the header to list the remaining clones.
- **Confidence:** High

#### [F-006] project.db write transactions are DEFERRED and read before they write, so busy_timeout cannot save a cross-process writer
- **Source ID:** C1-001
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `crates/sceneworks-core/src/project_store.rs:4232-4233, 5590-5592, 6210-6213, 7412-7414, 7440-7442, 7580-7582`; `crates/sceneworks-core/src/character_store.rs:201, 245, 269, 287, 321, 402, 439, 469, 520, 571, 603, 647, 696, 728, 858`; opener at `project_store.rs:6375-6381`
- **Finding:** Every project.db write path does `connect_project_db` then `connection.transaction()` (DEFERRED), and most then call `apply_project_migrations(&transaction)`, whose first statement is `pragma user_version` (a read), before the first write. `jobs_store.rs:2859-2866` documents exactly why this fails: SQLite does not run the busy handler on a read-to-write lock upgrade, so the 5s `busy_timeout` set at `project_store.rs:6379` is bypassed and one of two overlapping writers gets an immediate `SQLITE_BUSY`. project.db is also left in rollback-journal mode (no WAL pragma anywhere in the file), which makes the window wider than on jobs.db.
- **Impact:** The worker is a separate process that opens the same project.db through `ProjectStore` (`sceneworks-worker/src/asset_media.rs`, `depth_asset.rs`, `audio_jobs.rs`). A worker asset index write that overlaps an API write or reindex can fail with "database is locked" despite the busy_timeout, leaving the sidecar written and the index stale. The AssetIndexMutation dirty marker heals this on the next read, but the user sees a failed generation or upload. Confidence is Medium because the in-process `lock_project_files` hides it for same-process callers.
- **Suggested fix:** Add one `begin_project_write(&mut Connection)` helper that uses `TransactionBehavior::Immediate` (as jobs_store does) and migrates before opening the transaction. Use it in every site above, and in character_store through `connect_project_db_migrated`. Decide explicitly whether project.db should be WAL, and document the choice next to `connect_project_db`.
- **Confidence:** Medium

#### [F-007] The atomic-write helper is reimplemented about nine times with drifting durability
- **Source ID:** C1-002
- **Category:** redundant
- **Severity:** High
- **Location:** `crates/sceneworks-core/src/store_util.rs:37-72` (canonical: random tmp, fsync file, rename, fsync dir); `catalog_store.rs:4319-4337` plus its own `random_hex` at `4340-4352`; `credentials.rs:292-316, 351-357`; `resolved_cache.rs:3497-3520`; `sceneworks-worker/src/manifest.rs:225-262`; `sceneworks-worker/src/catalog_semantic_jobs.rs:607-620`; `sceneworks-worker/src/catalog_image_fetch.rs:1823-1905`, `1901-1907`; `sceneworks-worker/src/downloads.rs:704-722`
- **Finding:** The same stage, fsync, rename pattern exists in at least eight places, plus three `sync_parent_dir` copies. The copies differ on the points sc-1633 and sc-8949/F-147 were fixed for. `catalog_store::atomic_write` never fsyncs the directory. `manifest.rs::write_json_value_blocking` and `write_json_value` use a deterministic `<name>.json.tmp`, and the async one has an unlocked caller at `media_jobs.rs:2036`, which is the exact interleaving bug sc-1633 removed from `store_util`. `catalog_store::random_hex` is a byte-for-byte twin of `store_util::random_hex`, differing only in the error type.
- **Impact:** The durability and uniqueness guarantee applies only to callers that happen to use store_util. A catalog manifest can lose its rename on power loss. Two concurrent writers of the same worker manifest or track file can corrupt each other.
- **Suggested fix:** Move `atomic_write` and `random_hex` to a neutral error-agnostic module (io::Result) with a mode parameter or a `write_secret` variant for credentials. Have store_util, catalog_store, credentials, manifest.rs and media_jobs.rs call it. Keep resolved_cache's `create_new` variant only if its semantics truly differ, and say so in a comment.
- **Confidence:** High


### Medium findings — Rust backend

#### [F-008] Server-supplied ETag is joined into a path without confinement
- **Source ID:** W3-003
- **Category:** security
- **Severity:** Medium
- **Location:** `downloads.rs:1364-1371` (also `normalize_etag` at `1458-1464`)
- **Finding:** `blobs_dir.join(&etag)` uses `x-linked-etag` or `etag` from the HEAD response, stripped only of quotes and `W/`. A value like `../../x` or an absolute path escapes `blobs/`. Every other server- or payload-derived component in this function goes through `safe_join` (the comment cites sc-13583), so this one is an omission. The blob is then written and hash-verified there, and `link_blob` symlinks it.
- **Impact:** A malicious or compromised `huggingface_base_url` mirror, or any HF-compatible endpoint a user configures, can make the worker write a file outside the HF cache. The write is partly constrained by the sha256 check for LFS files, but not for non-LFS ones.
- **Suggested fix:** Validate the etag against `[A-Za-z0-9._-]{1,128}` (the same shape as `blob_fallback_name`) or fall back to `blob_fallback_name`. Use `safe_join` for the blob path.
- **Confidence:** High on the code path. Exploitability needs a hostile endpoint.
- **Lead verification:** Verified by lead: `normalize_etag` (`downloads.rs:1458-1463`) only trims whitespace/`W/`/quotes and the result is joined at `downloads.rs:1371` with no `safe_join`, unlike the neighbouring `revision` and per-file joins.

#### [F-009] Asset-media confinement is symlink-safe in `asset_media.rs` but lexical-only in most other resolvers
- **Source ID:** W4-005
- **Category:** security
- **Severity:** Medium
- **Location:** `asset_media.rs:8-34,36-52` vs `paths.rs:905-923` (`safe_project_path`), `media_jobs.rs:161-186`, `vector_jobs.rs:1153` -> `sceneworks-core/src/project_store.rs:2723-2741`
- **Finding:** `asset_media.rs` documents that "confinement must not drift between separate copies" and canonicalizes + `starts_with` the project root. Only segment, upscale and pose use it. Frame-extract/person-detect/track/export (`media_jobs.rs`, 5 call sites), image/depth/video jobs (8 more `safe_project_path` sites) and vector image-to-svg (`resolve_asset_media_path`) only reject `..`/absolute components, so a symlink inside a project dir pointing outside is followed.
- **Impact:** A project containing a symlinked media file (e.g. an imported or shared project) lets a job read an arbitrary local file into frames / ffmpeg input. Exploitability depends on whether projects can arrive with attacker-controlled symlinks.
- **Suggested fix:** Make `safe_project_path` (and the core resolver) canonicalize-and-confine, or route all asset-source resolution through `asset_media::resolve_asset_media`; keep one implementation.
- **Confidence:** Medium

#### [F-010] ffmpeg is spawned on every audio upload with no timeout, no protocol whitelist and no content sniff
- **Source ID:** C2-001
- **Category:** security
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/media_convert.rs:554-569` (audio), `:480-492` (image fallback), `:314-318` (HDR); caller `crates/sceneworks-core/src/project_store.rs:7131-7137`
- **Finding:** `normalize_audio_upload` sends EVERY audio upload to `transcode_to_wav_pcm16` (the module docs say there is deliberately no sniff gate). That function runs `ffmpeg -i <upload>` via blocking `.output()` with no `-protocol_whitelist`/`-f` pin and no timeout. ffmpeg auto-probes the demuxer from content, so a crafted playlist-style file (HLS/concat/SDP) can make it open `file:`/`http(s):` URIs named inside the upload (local-file read into the decoded stream, or SSRF from the API host), and a pathological file can hang the call forever. Images are protected by `sniff_image_kind` first; audio is not.
- **Impact:** A hostile or merely malformed upload can stall the upload request/blocking thread indefinitely, and on an exposed API can be turned into a file-read/SSRF primitive. The worker's own ffmpeg runner already has a bounded timeout (`media_jobs.rs` `SCENEWORKS_FFMPEG_TIMEOUT_SECONDS`), so core is the weaker twin.
- **Suggested fix:** Add `-nostdin -protocol_whitelist file,pipe` (and `-f wav|mp3|...` after a magic-byte sniff, or reject playlist/text containers), run through a timeout-wrapped spawn shared with the worker (see C2-012).
- **Confidence:** Medium (depends on how rust-api gates upload extensions before reaching here)

#### [F-011] Catalog create/attach and Parquet import accept arbitrary host paths from any token-holding LAN client
- **Source ID:** A1-001
- **Category:** security
- **Severity:** Medium
- **Location:** `apps/rust-api/src/catalogs.rs:536-589` (create_catalog, attach_catalog), `apps/rust-api/src/catalogs.rs:2895-2902` (require_absolute_catalog_path), `apps/rust-api/src/training.rs:716-740` (validate_parquet_import_request), `crates/sceneworks-core/src/catalog_store.rs:2395-2412` (prepare_new_root), `:2294` (remove_dir_all)
- **Finding:** `model_library.rs:179` and `checkpoint_library.rs:438` gate host-path routes on `require_local_peer`. Their stated reason is that typed refusals are a filesystem-existence oracle and a LAN client could re-point shared state. Catalog `create`/`attach` only check `is_absolute()`. `create_catalog` will `create_dir_all` and write a SQLite DB at any writable path. `attach` and the Parquet import `sourcePath` canonicalize and open any readable dir or file, with distinct error text for not-found, not-a-catalog and wrong-extension. `delete_catalog_on_disk` then `remove_dir_all`s the attached path.
- **Impact:** A LAN client holding the access token can probe the server's filesystem layout. It can create directories and DB files anywhere the API user can write. It can point a worker Parquet job at arbitrary host files. This is the same threat the two library modules explicitly close.
- **Suggested fix:** Extract one shared `require_local_peer` extractor or guard. Apply it to catalog `create`/`attach` and to any request field carrying an absolute host path (`sourceConfig.paths`, Parquet `sourcePath`). Alternatively confine these to approved roots, as `validate_lora_import_source_path` does.
- **Confidence:** Medium (assumes the LAN-exposed token model of epic 4484 is in scope for these routes; no route-level guard was found in `lib.rs:1545-1586`)

#### [F-012] Raw internal error text (paths, SQLite messages) returned to clients; inconsistent with the sanitized catalog errors
- **Source ID:** A2-005
- **Category:** security
- **Severity:** Medium
- **Location:** `error.rs:197` and `error.rs:245` (`other => Self::internal(other.to_string())`), `error.rs:140-148`, ~200 `ApiError::internal(error.to_string())` call sites (e.g. `lib.rs:2509,2515,2520`, `assets.rs`, `film_planning.rs`), `lib.rs:4486-4506`
- **Finding:** `CatalogError` is deliberately mapped to fixed messages (error.rs:253-307), but `JobsStoreError`/`ProjectStoreError` fall through to `internal(error.to_string())`, and handlers wrap raw `io::Error`/`tokio` errors the same way. The detail goes verbatim into the JSON body, so absolute data-dir paths and rusqlite text reach every token holder (LAN mode). `validate_lora_import_source_path` also echoes the caller's path and distinguishes "not found" from "must be inside", an existence oracle for arbitrary filesystem paths.
- **Impact:** Information disclosure to LAN clients; `health` goes out of its way to hide directories when a token is set (lib.rs:2312-2316), so this contradicts the stated policy.
- **Suggested fix:** Make `ApiError::internal` log the detail at error level (it already does, error.rs:322) and return a generic body with a correlation id; keep detailed text only for typed 4xx. Replace the two `other =>` arms with a generic message.
- **Confidence:** High

#### [F-013] Replace ~20 inline `mlxQuantize` parsers with the existing `payload::quant_int`, and one shared bits-to-tier map
- **Source ID:** W1-001
- **Category:** redundant
- **Severity:** Medium
- **Location:** `video_jobs/wan.rs:449,1129`; `video_jobs/mochi.rs:119`; `video_jobs/ltx.rs:400`; `video_jobs/mod.rs:2422`; `video_jobs/minimax_h3.rs:1569`; `video_jobs/candle.rs:244,313,2015`; `video_jobs/bernini.rs:218,616`; `image_jobs/qwen.rs:30`; `image_jobs/qwen_control.rs:154`; `image_jobs/tier_resolver.rs:444`; `image_jobs/krea_imported.rs:1235`; `image_jobs/base.rs:1687,2153,2335,2548,3126,3213,3270,3342,3750,3791`; `image_jobs/instantid.rs:673`; `image_jobs/sdxl_control.rs:70`. Helper that already exists: `payload.rs:17` (`quant_int`), also `vram_gate.rs:241`.
- **Finding:** The same `v.as_i64().or_else(|| v.as_str()?.trim().parse().ok())` expression and the same `<=0 -> bf16, <=4 -> q4, else q8` ladder are re-typed in about 20 places, with per-file wrappers (`wan_quant_bits`, `ltx_quant_bits`, `mochi_quant_bits`, `candle_wan_quant_bits`). The None-default differs per copy (q8 in image tiers and candle Wan, q4 in MLX Wan, MiniMax and Bernini). The scattered copies make those differences easy to miss.
- **Impact:** A change to what `mlxQuantize` accepts (a string like `"q8"`, a float, or a bool) has to be made in every copy. Copies that disagree on the default silently pick different tiers per family. `quant_int` also rejects booleans explicitly; the inline copies rely on `as_i64` returning None.
- **Suggested fix:** Add `ImageRequest::quant_bits()` / `VideoRequest::quant_bits()` (or a single `requested_quant_bits(&JsonObject)`) in `payload.rs` and a `tier_for_bits(bits, default)` helper. Make the defaults an explicit argument at each call site.
- **Confidence:** High

#### [F-014] MLX/Candle twin helpers in the Qwen and Z-Image control lanes are copy-pasted instead of shared
- **Source ID:** W1-002
- **Category:** redundant
- **Severity:** Medium
- **Location:** `image_jobs/qwen.rs:27-64` vs `image_jobs/qwen_control.rs:154-190` (`qwen_control_tier_subdir`, `qwen_control_installed_default_tier`, identical bodies); `qwen.rs:66` vs `qwen_control.rs:117` (`qwen_control_available`); `qwen.rs:219` vs `qwen_control.rs:304` (`qwen_control_raw_settings`); `zimage.rs:22,212` vs `zimage_control.rs:146,272` (`zimage_control_available`, `zimage_control_raw_settings`)
- **Finding:** Each pair carries a doc comment saying "Mirrors the MLX/candle ..." but is a separate copy. The 20-line q8/bf16/q4 clamp ladder is verbatim in both Qwen files. The Z-Image `*_raw_settings` pair has already diverged: MLX records `guidanceScale: null` and `mlxQuantize`, while Candle records guidance only for the base model plus a `controlEngine` key. The two `*_available` predicates also differ: the Candle ones add `mode != "edit_image"`, and the Z-Image MLX one checks `z_image_turbo` only.
- **Impact:** Fixing the tier clamp or the telemetry keys on one backend does not reach the other. The sc-10726 "never force a new fetch" rule exists in two places.
- **Suggested fix:** Move the pure functions (tier clamp, raw-settings builder with a backend discriminator) into a backend-neutral module (e.g. `tier_resolver.rs`). Keep only the cfg-specific availability predicate in each lane file.
- **Confidence:** High

#### [F-015] Image and video each implement LoRA file resolution; the copies have drifted and the video doc comments are stale
- **Source ID:** W1-003
- **Category:** redundant
- **Severity:** Medium
- **Location:** `image_jobs.rs:2005-2031` (`resolve_adapter_file`) vs `video_jobs/mod.rs:2327-2352` (`resolve_lora_file`); `image_jobs.rs:2062` (`lora_weight`, default 1.0) vs `video_jobs/mod.rs:2296-2310` (`lora_scale`); callers at `video_jobs/ltx.rs:1068`, `video_jobs/wan.rs:752`, `video_jobs/mod.rs:2375`, `image_jobs/base.rs:5748-5790`
- **Finding:** `resolve_lora_file` and `resolve_adapter_file` have the same body: confine, directory to `.safetensors`, exists check. The `lora_scale` doc says "default 0.8 -- matches the image path" but the code returns 1.0. The `resolve_lora_file` doc says it uses core's `first_safetensors_path` but the code calls `crate::resolve_adapter_in_dir`. The `MAX_JOB_LORAS` guard plus the resolve/classify/scale loop is repeated four times in video and twice in image. The image lane also has a pinned variant (`resolve_prepared_adapter_file`, `pin_app_managed_model_file`) that closes the check-then-load window. The video lanes have no equivalent.
- **Impact:** A confinement or pinning fix lands on the image side only; the video LoRA path stays on the older unpinned behaviour. Readers following the stale docs are misled about the default strength.
- **Suggested fix:** Have the video adapter builders call `image_jobs::resolve_adapter_file` / `lora_weight`, or move all three into `paths.rs`, and add the pinned variant for video. Correct the two doc comments.
- **Confidence:** High

#### [F-016] Dispatch god functions remain: `run_image_generate_job_unexplained` is 1,292 lines
- **Source ID:** W1-004
- **Category:** readability
- **Severity:** Medium
- **Location:** `image_jobs.rs:489-1780`; `video_jobs/mod.rs:635-1241` (607 lines); `video_jobs/candle.rs:1119-1487` (369 lines)
- **Finding:** Previously F-026 (2026-07-01), still present and larger (the old range was `image_jobs.rs:334-1001`). The image function contains 22 `#[cfg]` blocks, a macOS dispatch arm (about 247-649) and a Candle arm (about 655-1160) that mirror each other, and the stub fall-through guards at 1161-1190.
- **Impact:** Every new family edits the middle of one of these functions. The route selection already lives in `prepare_image_route` / `resolve_candle_image_route`, but execution is still an inline `match`.
- **Suggested fix:** Extract each arm into `run_mlx_route(...)` / `run_candle_route(...)` with a shared `RouteOutcome`. Move validation and plan building ahead of them and keep the heartbeat region wrapper in one place.
- **Confidence:** High

#### [F-017] `image_jobs/base.rs` is still a 19.6k-line `include!` module with inline test modules interleaved with production code
- **Source ID:** W1-005
- **Category:** readability
- **Severity:** Medium
- **Location:** `image_jobs.rs:3833-4190` (about 20 `include!` lines); `image_jobs/base.rs` (19,649 lines; 23 `mod *_tests` blocks between line 1503 and 19525, interleaved with production items such as `choose_downtier` at 5188 and `resolve_steps` at 5358); `image_jobs/tests.rs` (29,810); `video_jobs/tests.rs` (21,539); `video_admission/tests.rs` (5,667)
- **Finding:** Previously F-020 (2026-07-12), partly addressed. About 30 Candle lane files are now real `mod`s, but `base.rs` and about 20 families are still `include!`d into one namespace. `base.rs` mixes production helpers with thousands of test lines, and there are no `//!` docs saying what belongs where.
- **Impact:** Name collisions, no per-module privacy, and rust-analyzer and diff tools struggle at this size. Several `cfg_attr(..., allow(dead_code))` attributes (see W1-013) exist because of the shared-namespace model.
- **Suggested fix:** Move the inline test modules out of `base.rs` first (mechanical, no visibility changes). Then split production code by concern (geometry, tier resolution, adapters, generation harness, admission) into real modules.
- **Confidence:** High

#### [F-018] `GenerationMemory`-from-selection is hand-built on Candle, delegated to the contract on MLX
- **Source ID:** W2-002
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/candle_memory_strategy.rs:1999-2015`, `crates/sceneworks-worker/src/mlx_fit_gate.rs:1593-1599`
- **Finding:** MLX calls `contract.generation_memory(&selection).unwrap_or_default()`, the contract's own translation (sc-15805). Candle re-implements it field by field, returns `None` for `Resident`, and fills the rest with `..Default::default()`.
- **Impact:** Any new `GenerationMemory` field or rung that gen-core's `generation_memory` handles is silently dropped on Candle until someone edits this copy. The two lanes then run different request memory blocks for the same selection.
- **Suggested fix:** Use `contract.generation_memory` on Candle and keep the `Resident => None` mapping at the call site. Add a parity test over both.
- **Confidence:** Medium

#### [F-019] Estimate-candidate `MemoryEvidence` literal copy-pasted about 10 times with unsynchronised revision stamps
- **Source ID:** W2-003
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/mlx_fit_gate.rs:1615-1651,2429-2470,6031-6070`; `candle_memory_strategy.rs:1610-1650,2590-2640`; `vram_gate.rs:1652,1811,2001`; `krea_control_fit.rs:450-520,662-690`; `memory_strategy.rs:1076`
- **Finding:** The unverified-estimate `MemoryEvidence` is rebuilt from scratch at each site. It carries the same `ImplementedUnverified` conformance and the same `MemoryEvidenceDimensions` (Satisfied/Missing/Missing/Missing/Unverified/Satisfied). It also has the same `parity: Exact`, `NotRun` and `observed_peak_bytes: None` fields. The only parameters are backend, mode/overlay, revision strings and peak.
- **Impact:** A change to what "estimate evidence" means (a new dimension, or flipping a verdict) must be made in 10+ places. Missing one produces a candidate that ranks or demotes differently on one route.
- **Suggested fix:** Add `MemoryEvidence::unverified_estimate(key, peak, fingerprint, revisions)` in `memory_strategy.rs` and call it from every site.
- **Confidence:** High

#### [F-020] Frozen, mismatched inference-revision SHAs stamped into evidence; core doc claims they are gone
- **Source ID:** W2-004
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/mlx_fit_gate.rs:62-63,1646,2460,19134`; `crates/sceneworks-worker/src/catalog_semantic_jobs.rs:51`; `crates/sceneworks-core/src/memory_calibration.rs:48-56`; `crates/sceneworks-memory-adapter/src/lib.rs:13`
- **Finding:** `mlx_fit_gate::INFERENCE_CONTRACT_REVISION` is `1c4354b4…`, while the actual pin (Cargo.toml, `INFERENCE_PIN`, `INFERENCE_RUNTIME_REVISION`) is `dad3036a…`. Candle evidence uses `catalog_semantic_jobs::INFERENCE_RUNTIME_REVISION` (an unrelated embedding-job module), so MLX and Candle stamp different revisions. `memory_calibration.rs` says "All of them are gone; every lane now asks this table", which is false for these constants. A long orphaned comment ("Must remain identical to mlx-gen-mage's loaded provider contract…") sits directly under the revision const at 64-80 and belongs to a different constant.
- **Impact:** Evidence revisions are hand-edited at pin bumps, so they drift (already one stale value). A reader will trust the "gone" doc and not update them. A Candle gate imports its revision from a catalog-embedding job, which is the wrong dependency direction.
- **Suggested fix:** Have one pin constant in core (or `inference_runtime`) that all three crates read, with a test against the Cargo.toml rev. Fix the core doc and move the orphan comment to its const.
- **Confidence:** High

#### [F-021] Four copies of the "smallest declared value per engaged knob" closure and a near-duplicate params builder
- **Source ID:** W2-005
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/mlx_fit_gate.rs:2643-2676,2678-2735`; `candle_memory_strategy.rs:1269-1316`; `sensenova_jobs.rs:355-380`
- **Finding:** `mlx_fit_gate::estimate_floor_smallest_parameters` and `candle_memory_strategy::estimate_floor_parameters` build `MemoryStrategyParameters` with identical logic. They differ only in `transformer_window_component` (None on MLX). The `let smallest = |strategy, pick| …` closure is repeated verbatim in four places. `request_regime` is also duplicated: `candle_memory_strategy.rs:1801` returns `Option<RequestRegime>`, `mlx_fit_gate.rs:2254` takes the parameters by value and returns `RequestRegime`, with the same field logic.
- **Impact:** Selection of "most bounding parameters" must stay in lockstep with `validate_selection`. Fixing a range edge case in one copy leaves the others wrong.
- **Suggested fix:** Put `smallest_declared_parameters(contract, |s| engaged(s), window_component)` and one `request_regime` in `memory_strategy.rs`.
- **Confidence:** High

#### [F-022] About 30 call sites repeat `apply_vram_cap(nvidia_vram_budget_gb(..), cuda_vram_cap_gb())`
- **Source ID:** W2-006
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/vram_gate.rs:131-156`; sites in `image_jobs/base_admission.rs:215,286`, `video_jobs/candle.rs:1013`, `yue_admission.rs:970`, `yue2_admission.rs:2702`, `sensenova_jobs.rs`, `depth_jobs.rs`, `iris_restore.rs`, `iris_training.rs`, `image_jobs/{bernini,flux_ipadapter,kolors_*,krea_*,flux2_*,pulid_candle,qwen_edit_candle,flux1_control_candle,conditioning_gate}.rs`
- **Finding:** Every Candle admission site composes the same three calls by hand. The MLX side has one `resolve_budget`/`live` seam.
- **Impact:** A change to cap semantics, probe freshness or error handling needs about 30 edits. A site that forgets the cap silently bypasses the small-card emulation env.
- **Suggested fix:** Add `vram_gate::live_capped_budget(gpu_id) -> Option<VramBudget>` and migrate the sites. `architecture_tests.rs` could then forbid the raw pairing.
- **Confidence:** High

#### [F-023] `gpu.rs` bypasses `run_bounded_command`; four probes leak the child on timeout
- **Source ID:** W2-008
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/gpu.rs:496-503,774-778,807-813,886`; contrast `gpu.rs:448-465`
- **Finding:** `run_bounded_command` sets `kill_on_drop(true)` and its comment explains that Tokio otherwise leaves the child running when the future is dropped. `nvidia_compute_cap`, `sysctl_memsize_mb`, `ioreg_accelerator_stats` and `vm_stat` each wrap a bare `Command::…output()` in `tokio::time::timeout`. They do not set kill-on-drop, so a hung `nvidia-smi`, `ioreg` or `vm_stat` is orphaned on every timeout.
- **Impact:** The heartbeat runs these repeatedly. On a wedged driver or IOKit this accumulates stuck processes, the failure the helper was written to prevent.
- **Suggested fix:** Route all four through `run_bounded_command` (generalise its error type) or add `.kill_on_drop(true)`.
- **Confidence:** High

#### [F-024] Provider ids hard-coded in generic gates; parallel per-family lists must be kept in sync by hand
- **Source ID:** W2-012
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/candle_memory_strategy.rs:74-80,2320-2325,2465-2492,2196,2353,2704`; `mlx_fit_gate.rs:84-89,817,3802,4052,4132,4793,4819,6316`; `memory_route_registry.rs:RULES (562-1460)`; `engines.rs MODEL_TABLE`
- **Finding:** The evidence-revision table (`_ => DECLARATION_REQUEST_EVIDENCE_REVISION`), `is_ideogram`/`is_sana`/`is_chroma` and the SDXL checks are string matches on engine ids. `engine_id.starts_with("mage_flow")` appears six times in `mlx_fit_gate`, and the mage_flow id list is enumerated separately on Candle (2476-2481) and MLX (84-89). `memory_route_registry::RULES` holds a third per-family list.
- **Impact:** A new family has to be added to 3-4 lists. An omission falls through to the default arm and is admitted with the generic revision rather than failing closed.
- **Suggested fix:** Make the revision/family attributes fields of the `RULES` rows (or contract facts) and look them up. Make the `_` arm an error.
- **Confidence:** Medium

#### [F-025] Four `YuE`/`YuE2` admission modules duplicate tier enums, budget probes and evict credit
- **Source ID:** W2-013
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/yue_admission.rs:235-262,725-760,809-813,955-985`; `crates/sceneworks-worker/src/yue2_admission.rs:205-226,2685-2710,2636-2660`
- **Finding:** `YueTier` and `Yue2Tier` both carry the `Bf16/Q8/Q4` + `key()` + `from_key()` shape. Both define a `working_set_ceiling_bytes()` copy and a `live_budget` that repeats the `apply_vram_cap`+`nvidia_vram_budget_gb` pairing. The evict credit `(free + reclaimable.max(0)).min(total)` is inlined in yue_admission rather than calling `vram_gate::with_reclaimable` (`vram_gate.rs:168`), and both also redefine `gib`.
- **Impact:** Two budget semantics for one hardware fact. A fix to the reclaim clamp would miss the inlined copy.
- **Suggested fix:** Reuse `vram_gate::with_reclaimable`, share a `UnifiedWorkingSet`/ceiling helper in `fit_gate`, and use one tier type.
- **Confidence:** Medium

#### [F-026] Two SSRF-pinned client builders with different hardening (proxy bypass and unbounded DNS)
- **Source ID:** W3-002
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `downloads.rs:1845-1856` (`build_source_url_client`) vs `downloads.rs:2074-2090` (`build_public_source_url_client`); `downloads.rs:2502-2532` (`validate_lora_url_dns`) vs `downloads.rs:2009-2073`
- **Finding:** The dataset-image path calls `.no_proxy()` (comment: a system proxy would bypass the validated/pinned address set) and wraps DNS in `tokio::time::timeout`. The LoRA/model `sourceUrl` path (`build_source_url_client`) does neither. It uses `apply_download_timeouts` only, and `validate_lora_url_dns` calls `lookup_host` with no timeout. The validation and `resolve_to_addrs` logic is otherwise duplicated line for line.
- **Impact:** With `HTTPS_PROXY` set (common on corporate and Docker hosts), reqwest sends CONNECT to the proxy, so the pinned addresses are ignored and the resolution happens at the proxy. The IP validation becomes advisory for authenticated downloads, which carry bearer or query tokens. A stalled resolver can also hang a job pre-connect.
- **Suggested fix:** Have one pinned-client builder take options (`no_proxy`, timeouts). Route `validate_lora_url_dns` through the bounded resolver the dataset path uses.
- **Confidence:** Medium. Proxy behaviour depends on the deployment, but the code difference is certain.

#### [F-027] Non-atomic cache hit accepts any existing file when size is unknown, and HEAD failures are swallowed
- **Source ID:** W3-004
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `downloads.rs:336-345` (`ensure_cached_file_verified`), `downloads.rs:387-401` (`remote_content_length`)
- **Finding:** `remote_content_length` converts any HEAD error or non-2xx to `Ok(None)`. The cache check then treats `expected_size == None` as a hit (`.unwrap_or(true)`), so any file at `target` is returned. In this variant `target` is itself the resumable partial (only the `_atomically` variant uses `.partial`). A truncated file from an interrupted transfer is therefore reported complete if the HEAD blips on the next attempt and the caller supplied no size or digest.
- **Impact:** A silently truncated weight file gets loaded. It shows up as an opaque loader failure, or as wrong output if the file is a header-valid shard.
- **Suggested fix:** Treat an unknown expected size plus an existing target as "resume, don't trust". Require size or digest for a cache hit, as `ensure_cached_file_verified_atomically` already does by failing closed. Alternatively migrate remaining callers to the atomic variant.
- **Confidence:** Medium. It needs the caller to omit the size (the HF resolver usually supplies one).

#### [F-028] Source-URL downloads write straight to the final filename: no staging, no hash, and a 416 edge case when the total is unknown
- **Source ID:** W3-006
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `downloads.rs:1553-1560`, `1585-1600`, `1604-1630`, `1700-1730`
- **Finding:** Output goes to `target_dir/<file_name>` and is resumed in place. There is no `.partial` plus rename, so a cancelled or failed import leaves a plausible-looking truncated `.safetensors` in the final location. When HEAD yields no length (`total_bytes == None`) and a fully downloaded file already exists, the next attempt sends `Range: bytes=N-`. The 416 is then only accepted as complete if `total_bytes.or(range_total)` matches. Otherwise it falls through to `error_for_status()` and fails. No caller digest verification happens on this path at all.
- **Impact:** A retry of an already-complete URL download can fail permanently. A half-written LoRA dir can be scanned as an installed LoRA.
- **Suggested fix:** Stage into `<name>.partial` and promote on success, reusing `promote_partial_download`. On 416 with unknown total, treat `existing_bytes > 0` plus matching `Content-Range` as complete or restart.
- **Confidence:** Medium

#### [F-029] Desktop credential socket pull has no timeout and an unbounded read, and the "prompt once" claim races
- **Source ID:** W3-009
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `credentials_ipc.rs:136-150` (`fetch_via_ipc`), `189-202` (`request_credential`)
- **Finding:** `UnixStream::connect`, the write and `read_to_string` have no timeout and no size cap. The cache is checked and then filled with the lock released across the request, so concurrent downloads for the same host each hit the keychain socket. The doc comment says "prompts at most once".
- **Impact:** A wedged desktop server hangs every gated download, because `with_hf_auth` runs before each HF request. That holds the job until the stale-worker sweep. Concurrent first use can show duplicate keychain prompts.
- **Suggested fix:** Wrap the exchange in `tokio::time::timeout` (a few seconds if the keychain is already unlocked). Use `.take(64 KiB)` on the read. Use a per-host `OnceCell` or an in-flight map so one request serves all waiters.
- **Confidence:** Medium

#### [F-030] `advanced_*` readers silently coerce where `preflight_*` readers are strict (drift, and a truncating cast)
- **Source ID:** W3-010
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `training_jobs.rs:1411-1450` (`preflight_string/bool/u32`) vs `training_jobs.rs:2303-2370` (`advanced_str/f32/u32/bool`)
- **Finding:** Two parallel parsers for the same `advanced` JSON object. Preflight rejects a wrong type with a message. The mapper's versions fall back to the default on a wrong type, and `advanced_u32` does `as u32` on a `u64`, so 4294967297 becomes 1. Any key not routed through a `preflight_*` call is therefore unchecked.
- **Impact:** A malformed or hand-edited `advanced.<key>` yields a run with a silently different configuration. Examples: a typo'd string for a numeric key, or a wrapped step count. No error is shown to the user.
- **Suggested fix:** Have one typed accessor set return `Result`. Use `u32::try_from`. Make `map_training_config` use the strict readers, or run preflight over every key the mapper reads.
- **Confidence:** High on the code. Which keys escape preflight is not enumerated.

#### [F-031] Prepared-bundle copy plus SHA-256 of multi-GB files runs synchronously inside async job paths
- **Source ID:** W3-011
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `training_jobs.rs:620-710` (`copy_and_verify_prepared_bundle`), called through `preflight_training_run` from `training_jobs.rs:2061-2066` (`run_training_dry_run`) and `3008` (`prepare_training_run`)
- **Finding:** `preflight_training_run` is a sync function that does blocking `std::fs` open, copy (64 KiB chunks) and hash. It is invoked directly in async fns with no `spawn_blocking`, and no `heartbeat_while_blocking` (which `downloads::sha256_file` uses for exactly this reason, sc-13856).
- **Impact:** For LTX prepared bundles (multi-GB safetensors) the executor thread is blocked, so heartbeats and cancel polling stall past the 90 s stale-worker window. The job is then swept as interrupted. On a current-thread runtime the stall is total.
- **Suggested fix:** Run `preflight_training_run` under `spawn_blocking` with `heartbeat_while_blocking`, as `sha256_file` and the control-prep path do.
- **Confidence:** Medium. It depends on bundle size and runtime flavor.

#### [F-032] `control_dataset_byo.rs` is a 569-line module with zero production callers
- **Source ID:** W3-012
- **Category:** dead-code
- **Severity:** Medium
- **Location:** `lib.rs:896-897` (`#[allow(dead_code)] mod control_dataset_byo;`), `control_dataset_byo.rs` (all)
- **Finding:** No code outside the module references `ingest_prepared_pose_pairs`, `ingest_coco_annotated` or `CocoAnnotatedSource`. The lib.rs comment says wiring is "a scoped follow-up". It also lacks the cancel polling and progress callbacks that `prepare_control_dataset` has, and it joins the COCO JSON `file_name` onto `images_dir` unconfined (`control_dataset_byo.rs:~373`). Both would have to be fixed before wiring it in.
- **Impact:** It compiles and tests on every lane while carrying a latent path-traversal shape (a `../` in an untrusted COCO file) for whoever wires it up.
- **Suggested fix:** Delete it, or wire it up with `safe_join`, cancel and progress added. At minimum, note the traversal hazard at the declaration.
- **Confidence:** High on unwired. Medium on the traversal being reachable once wired.

#### [F-033] Streaming file-SHA-256 loop reimplemented 7+ times across worker, core and memory-adapter
- **Source ID:** W3-013
- **Category:** redundant
- **Severity:** Medium
- **Location:** `downloads.rs:1084`, `training_jobs.rs:649`, `model_jobs.rs:3003`, `iris_curricula.rs:109`, `sceneworks-core/src/checkpoint_ingest.rs:654`, `sceneworks-core/src/checkpoint_plan_store.rs:1193`, `sceneworks-memory-adapter/src/lib.rs:2109`
- **Finding:** Each is a hand-written read-loop over `Sha256` with a different buffer constant (64 KiB, 1 MiB, `COPY_BUFFER_BYTES`, `HASH_BUFFER_BYTES`, `io::copy`). `checkpoint_ingest` and `checkpoint_plan_store` have byte-identical `fn sha256_file`.
- **Impact:** Divergent buffer sizes and error text, and no single place to add progress or cancellation.
- **Suggested fix:** Add `sceneworks_core::hashing::sha256_file(path) -> io::Result<String>` (plus a `Write` adapter for copy-and-hash) and have callers use it. Keep the async `heartbeat_while_blocking` wrapper in the worker.
- **Confidence:** High

#### [F-034] Seven identical `*_progress` ProgressRequest constructors differ only by name
- **Source ID:** W4-002
- **Category:** redundant
- **Severity:** Medium
- **Location:** `audio_jobs.rs:2514-2536`, `caption_jobs.rs:698-720`, `prompt_refine_jobs.rs:1870-1892`, `analysis_jobs_common.rs:55-77`, `image_jobs.rs:3719-3741`, `video_jobs/mod.rs:2065-2087`, `training_jobs.rs:2141-2163`; canonical `progress.rs:667-693`
- **Finding:** All seven build the same `ProgressRequest` (error/eta/peaks `None`, `backend: Some(..)`, `worker_id: None`, empty `extra`) from `(status, stage, progress, message, result, backend)`. `yue2_jobs.rs:212-221` already shows the right shape (call `progress_payload`, override `backend`).
- **Impact:** Adding or defaulting a `ProgressRequest` field (contract change) needs seven edits; copies already carry slightly different comments (some drop the sc-4172 note), and the module-local `#[cfg(any(macos, candle))]` gates make a missed copy only fail on one platform.
- **Suggested fix:** Add `progress_payload_for_backend(status, stage, progress, message, result, backend)` (or a `.with_backend()` on `progress_payload`) in `progress.rs`; make the seven names thin aliases or delete them.
- **Confidence:** High

#### [F-035] Catalog semantic job hand-rolls two more cancel loops with different, weaker semantics
- **Source ID:** W4-003
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `catalog_semantic_jobs.rs:752-766` (vision), `catalog_semantic_jobs.rs:912-940` (image fetch)
- **Finding:** The vision loop uses a hard-coded 2 s sleep instead of `progress_report_interval(settings)` and on cancel trips the flag then returns `Canceled` immediately, dropping the `with_cached_refiner` future without waiting for the decode thread to stop. The fetch loop uses `check_cancel` (posts terminal `Canceled` at acknowledgement, the F-115 anti-pattern replaced everywhere else) and does not consult `shutdown_requested()` (sc-9618).
- **Impact:** The worker row can be freed (terminal `Canceled` / error return) while the VLM is still on the GPU, so the scheduler may hand the same worker a second GPU job; a process quit does not interrupt a long fetch until the next tick finishes. Behaviour differs from every sibling job.
- **Suggested fix:** Route both through the shared helpers (`run_blocking_with_heartbeat` / the generalised stream loop from W4-001), or at least use `cancel_requested_peek` + `shutdown_requested` and join before returning.
- **Confidence:** Medium

#### [F-036] Three dataset item parsers re-implement payload parsing instead of using `payload.rs` helpers
- **Source ID:** W4-004
- **Category:** redundant
- **Severity:** Medium
- **Location:** `dataset_analysis_jobs.rs:272-340`, `face_analysis_jobs.rs:460-530` (outside area), `caption_jobs.rs:765-830`; helpers at `payload.rs:34-66`
- **Finding:** Each re-writes the same `datasetRoot` extraction, `items` array check, and per-item `.get(k).and_then(as_str).map(str::trim).filter(!empty).ok_or_else(InvalidPayload(...))` chain for `itemId`/`contentHash`/`imagePath`, differing only in message prefix, then call `resolve_dataset_item_path`. The established `required_payload_string`/`required_value_str` are not used (and don't trim, so a trimmed-label variant is what is missing). 228 `map(str::trim)` sites crate-wide show the pattern is general.
- **Impact:** Validation/trim rules can diverge per job (one already trims where `required_payload_string` does not); a new dataset job will copy a fourth parser.
- **Suggested fix:** Add `payload::required_trimmed_str(obj, key, label)` and a `dataset_items(settings, payload, label, per_item_fn)` iterator in `dataset_analysis_jobs`/`analysis_jobs_common`; keep only the item-specific fields in each module.
- **Confidence:** High

#### [F-037] Three catalog image-path resolvers with different confinement rules
- **Source ID:** W4-006
- **Category:** redundant
- **Severity:** Medium
- **Location:** `catalog_analysis.rs:1280-1290`, `catalog_semantic_jobs.rs:543-576`, `apps/rust-api/src/catalogs.rs:962-985` (`verified_cached_source`)
- **Finding:** Same name `resolve_catalog_image` returns `io::Result` in one module and `WorkerResult` in the other. Semantic version prefers `metadata.acquisition.cachePath`, rejects non-Normal components and then canonicalizes; the analysis version only canonicalizes `image_path` (so a `..` path that stays inside the root is accepted, and a URL-valued `image_path` with a cachePath is unavailable to analysis but available to embeddings).
- **Impact:** A record can be "analysed" or "embedded" against different files, or be skipped by one stage and accepted by another; fixes to confinement land in one copy only.
- **Suggested fix:** Put one `Catalog::resolve_local_image(&record)` in `sceneworks-core::catalog_store` and have worker and API call it.
- **Confidence:** Medium

#### [F-038] Parquet row/image helpers and decode limits duplicated between dataset import and catalog scanner/fetch
- **Source ID:** W4-008
- **Category:** redundant
- **Severity:** Medium
- **Location:** `dataset_parquet_jobs.rs:464-505` vs `catalog_parquet_scanner.rs:1590-1650,1810-1824`; `dataset_parquet_jobs.rs:17-19,519-560` vs `catalog_image_fetch.rs:36-41,103-160`
- **Finding:** `row_u32`/`row_string`/`hex_prefix` are subsets of the scanner's `row_u32_alias`/`row_string_alias`/`hex_prefix` (identical field match arms). Dataset import also re-implements fetch -> format guess -> limits -> decode with its own constants (32 MiB / 16 384 edge) while `catalog_image_fetch::verify_catalog_image_bytes` does the same with 256 MiB / 32 768.
- **Impact:** Two SSRF-adjacent download+decode validators with different caps; a fix to one (e.g. decode alloc limits) will not reach the other.
- **Suggested fix:** Expose `row_string_alias`/`row_u32_alias`/`hex_prefix` `pub(crate)` from the scanner and delete the dataset copies; have dataset import call `verify_catalog_image_bytes` with its smaller limits as a parameter.
- **Confidence:** High

#### [F-039] Parquet metadata scan is not cancellable and scans unboundedly when filters reject rows
- **Source ID:** W4-009
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `dataset_parquet_jobs.rs:119-135,353-461`
- **Finding:** `read_candidates` runs under `heartbeat_while_blocking` (which explicitly has no cancel path) and only stops once `limit` candidates are accepted; `MAX_SCAN_CANDIDATES` bounds accepted candidates, not rows read. With restrictive include/exclude/min_edge filters or wrong columns across many shards it walks every row of every shard.
- **Impact:** A user cancel is ignored until the scan ends (potentially minutes on LAION-scale shards), while the job holds the worker; the first `check_cancel` is after the scan.
- **Suggested fix:** Pass a `CancelFlag` plus a row-count cap into `read_candidates` and run via `run_blocking_with_heartbeat`.
- **Confidence:** Medium

#### [F-040] 4-line backend cfg predicate repeated ~1,900 times in the worker crate
- **Source ID:** W4-010
- **Category:** readability
- **Severity:** Medium
- **Location:** crate-wide, e.g. `prompt_refine_jobs.rs` (63), `caption_jobs.rs` (31), `analysis_jobs_common.rs:3x`, `media_jobs.rs` (19); `build.rs:1-9`; inconsistent variant at `sensenova_jobs.rs` (`any(macos, test, feature = "backend-candle")`)
- **Finding:** `#[cfg(any(target_os = "macos", all(not(target_os = "macos"), feature = "backend-candle")))]` appears 1,886 times (`rg` count of the inner `all(...)`). `build.rs` already exists but emits no cfg alias. `sensenova_jobs.rs` uses a subtly different predicate (feature on macOS also counts).
- **Impact:** Noise that hides logic, easy to mistype on one site (platform-only compile failures), and the sensenova variant is already divergent.
- **Suggested fix:** Emit `cargo:rustc-cfg=native_lane` (plus `rustc-check-cfg`) from `build.rs` and replace with `#[cfg(native_lane)]`; normalise the sensenova gates.
- **Confidence:** High

#### [F-041] Oversized files and functions mix orchestration with a hand-written SVG sanitizer
- **Source ID:** W4-014
- **Category:** readability
- **Severity:** Medium
- **Location:** `vector_jobs.rs` (8,014 lines; `run_vector_job_with_provider` 1114-1529, `validate_static_svg_source` 3124-3527, `index_svg_resources` 1825-2097, `sanitize_inert_svg_bytes` 3651-3887); `prompt_refine_jobs.rs:1099-1793` (`run_prompt_refine_job`, 694 lines); `audio_jobs.rs:997-1284`; `catalog_semantic_jobs.rs:1314-1693`
- **Finding:** The SVG static-validation/canonicalisation stack (~5k lines) lives in the job handler file next to provider streaming and publication. `run_prompt_refine_job` interleaves payload parsing, model selection, the supervision loop and result building in one fn.
- **Impact:** Security-sensitive sanitizer is hard to review or reuse (API-side SVG serving cannot call it), and handler changes risk it.
- **Suggested fix:** Extract `svg_sanitize.rs` (with its tests) from `vector_jobs.rs`; split prompt-refine into parse / run / finalize.
- **Confidence:** High

#### [F-042] The native-lane cfg predicate is pasted ~1,900 times; a build-script cfg alias would remove the trap CONTRIBUTING documents
- **Source ID:** W5-001
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `lib.rs:60-1040` (module gates), `depth_jobs.rs:587-606`, `face_analysis_jobs.rs`, `subject_mask_jobs.rs`, `kps_jobs.rs`, `util.rs:90-100`; `../build.rs:1-9`; `CONTRIBUTING.md:100-135`
- **Finding:** `any(target_os = "macos", all(not(target_os = "macos"), feature = "backend-candle"))` appears on 1,886 lines across 100 files, with a negated twin for the stub arm. `build.rs` already exists and only exports `SW_HOST_TARGET`. CONTRIBUTING calls this predicate "the base.rs / candle cfg rule" and says it has bitten sc-10404 and sc-8390.
- **Impact:** Each new helper must remember a 4-line predicate. A forgotten or subtly different copy is only caught by the manual `rust:check:neither` command or the CI parity lane. The predicate buries the real signal in `lib.rs`, where the module list alone is ~1,000 lines.
- **Suggested fix:** Emit `cargo:rustc-cfg=native_lane` from `build.rs` (or use `cfg_aliases`), declared with `cargo:rustc-check-cfg`, and replace the predicate with `#[cfg(native_lane)]` / `#[cfg(not(native_lane))]`. A mechanical sed does it.
- **Confidence:** High

#### [F-043] `ort_err` and `build_session` (CoreML/CUDA EP setup) are copy-pasted into pose_jobs, upscale_jobs and person_jobs
- **Source ID:** W5-002
- **Category:** redundant
- **Severity:** Medium
- **Location:** `pose_jobs.rs:573-609`, `upscale_jobs.rs:240-276`, `person_jobs.rs:820-851`; shared helper already exists at `ort_cuda.rs`
- **Finding:** Three copies of `Session::builder()` plus CoreML/CUDA EP registration with `error_on_failure` and `preload_cuda_dylibs()`, plus three byte-identical `ort_err`. Each copy's doc comment says it "mirrors `pose_jobs::build_session`". The person_jobs copy is CUDA-only, so the three have already diverged.
- **Impact:** The CUDA honesty contract (sc-5496/5499/6209) and the cuDNN preload fix have to be applied three times. They already were once, and each time the fix was re-explained in a different comment.
- **Suggested fix:** Move `build_session(path, accel)` and `ort_err` into `ort_cuda.rs` (or a sibling `ort_session.rs`) with the cfg split inside it, and have the three callers import it.
- **Confidence:** High

#### [F-044] The *_smoke / *_gpu_smoke files re-implement the same scaffolding instead of using smoke_support.rs
- **Source ID:** W5-003
- **Category:** redundant
- **Severity:** Medium
- **Location:** `smoke_support.rs:26-440` (what it owns today). Duplicates:
  - `env_path` x9: `scail2_gpu_smoke.rs:28`, `sensenova_gpu_smoke.rs:78`, `realvisxl_lightning_gpu_smoke.rs:44`, `instantid_pid_gpu_smoke.rs:57`, `sdxl_edit_pid_gpu_smoke.rs:30`, `zimage_pid_gpu_smoke.rs:31`, `flux1_control_mlx_smoke.rs:33`, `conditioned_image_gpu_smoke.rs:16`, `flux2_dev_gpu_smoke.rs:52`; `required_env_path` x3 (`imported_nvfp4_gpu_smoke.rs:37`, `imported_comfy_quant_gpu_smoke.rs:48`, `ltx_eros_gpu_smoke.rs:48`).
  - `env_or`, `image_std`, `save_png` re-declared in `instantid_pid_gpu_smoke.rs:65/72/111`, `sdxl_edit_pid_gpu_smoke.rs:38/45/60`, `zimage_pid_gpu_smoke.rs:39/46/61`, although `smoke_support.rs:85/105/431` exports exactly those. Those three files, plus `cuda_preflight_gpu_smoke.rs` and `generator_cache_context_gpu_smoke.rs`, do not import `smoke_support` at all.
  - `cached_turnkey_root` x5 (`sensenova_fast_q8`, `sdxl_base_q8`, `krea_turbo`, `lens_base_q4`, `lens_turbo_q4`, `chroma1_base_q4` `_mlx_smoke.rs`) and `resolve_q4_dir` x3 (`lens_base`, `lens_turbo`, `chroma1`). They differ only in the HF repo name. `cached_tier_dir` is also copied in `resolution_sweep.rs:138`, `footprint_measure.rs:69` and `ladder_e2e_sc18101.rs:132`.
  - `load_rgb` x3, `mean_abs_delta` x3, `quant_from_env` x3.
  - The axum worker-API stub (`stub_job_snapshot`, `stub_job`, `stub_progress`, `stub_heartbeat`, `stub_metrics`, `spawn_worker_api_stub`, `completed_result`) is cloned in `iris_gpu_smoke.rs:99-158`, `iris_training_gpu_smoke.rs:48-107`, `imported_nvfp4_gpu_smoke.rs:44-171` and `imported_comfy_quant_gpu_smoke.rs:55-182`. The last two diff almost line-for-line. `tests/support_stubs.rs` has a fifth, generic version.
- **Finding:** `smoke_support.rs` was created to own this scaffolding but only some smokes adopted it. 37 smoke/gpu_smoke files exist, and the remaining copies are verbatim.
- **Impact:** About 600 duplicated lines. Fixing a smoke helper, such as changing the HF cache layout, means editing 5-9 files, and the copies silently diverge.
- **Suggested fix:** Add `env_path`/`required_env_path`, `hf_snapshot_with(repo, sentinel)`, `resolve_tier_dir`, `load_rgb`, `mean_abs_delta`, `quant_from_env` and a `WorkerApiStub` (parameterised by job snapshot JSON) to `smoke_support.rs`, and port the files above. Make the Cargo/CI guard fail on a new `fn env_path` outside that module.
- **Confidence:** High

#### [F-045] Dataset-item payload parsing is cloned in four job modules even though analysis_jobs_common exists to prevent exactly this
- **Source ID:** W5-004
- **Category:** redundant
- **Severity:** Medium
- **Location:** `face_analysis_jobs.rs:460-534` (`face_analysis_items`), `subject_mask_jobs.rs:597-657` (`subject_mask_items`), `dataset_analysis_jobs.rs:272-340` (`analysis_items`), `upscale_jobs.rs:1676-1703` (`parse_dataset_upscale_items`); `analysis_jobs_common.rs:1-60`
- **Finding:** Each re-reads `datasetRoot`, `items`, `itemId`, `contentHash` and `imagePath` with the same trim and non-empty chain, then calls `resolve_dataset_item_path`. Only the noun in the error message differs. The upscale variant is looser: it silently `filter_map`s away malformed items instead of rejecting them, which makes it a silent behavioural fork. The `analysis_jobs_common.rs` header says "Each module keeps its own item parsing (the item shape differs)", but the shapes are identical apart from `captionText` and `captionHash`.
- **Impact:** About 250 lines of copies. A validation fix (e.g. rejecting duplicate itemIds or enforcing the confinement error text) has to land four times, and the upscale job already skips what the others reject.
- **Suggested fix:** Add `parse_dataset_items(settings, payload, job_label) -> WorkerResult<(root, Vec<RawItem{item_id, content_hash, image_path}>)>` to `analysis_jobs_common.rs` and let dataset_analysis extend with caption fields. Make the upscale looseness an explicit flag, or remove it.
- **Confidence:** High

#### [F-046] The poison-recovery cache idiom is hand-copied 13 times, including twice within one file
- **Source ID:** W5-005
- **Category:** redundant
- **Severity:** Medium
- **Location:** `person_segment_sam3.rs:95-113` (`with_sam3_weights`) vs `person_segment_sam3.rs:399-418` (`with_cached_weights`, identical body); `pose_jobs.rs:738,783,861`; `person_jobs.rs:795,916`; `person_segment.rs:254,387`; `kps_jobs.rs:238`; `upscale_jobs.rs:437`; `person_segment_sam3_candle.rs:92`
- **Finding:** `cell.lock().unwrap_or_else(|p| { let mut g = p.into_inner(); *g = None; g })` followed by "reload if cache key mismatch" is pasted at each site, and the three `pose_jobs` blocks differ only in what follows. `person_segment.rs:387` even re-types the idiom inside a test instead of calling a helper. The two functions in `person_segment_sam3.rs` are the same function under two names.
- **Impact:** The sc-4277/F-MLXW-13 rule ("a panic while holding the lock must drop the cached model") is only a convention. One site that forgets it keeps serving a possibly corrupt model, and nothing would flag it.
- **Suggested fix:** Introduce `ProcessCache<T>` (or a `lock_resetting(&Mutex<Option<T>>)` helper) in `cache_thread.rs`/`util.rs` with `get_or_load(key, loader)`, use it at all sites, and delete the duplicated sam3 function.
- **Confidence:** High

#### [F-047] Basic helpers are triplicated across worker, rust-api and core
- **Source ID:** W5-006
- **Category:** redundant
- **Severity:** Medium
- **Location:**
  - `paths.rs:4-23` (`safe_download_dir`) = `apps/rust-api/src/lib.rs:4445` = `sceneworks-core/src/model_artifacts/artifact_selection.rs:26` (whose doc comment admits it is "Byte-identical to the API and the worker").
  - `util.rs:4-16` (`format_bytes`) = `apps/rust-api/src/lib.rs:4417`.
  - `util.rs:18-28` (`quote_path`) is also copied in rust-api (`lib.rs:~4400`).
  - `util.rs:30` (`now_rfc3339`) = `apps/rust-api/src/lib.rs:4584`.
  - `civil_from_days` exists in `sceneworks-core/src/time.rs:79` and `sceneworks-memory-adapter/src/lib.rs:2343`.
- **Finding:** Core already owns `safe_download_dir` and the time module, and the worker and API both depend on core. The "byte-identical" claim is maintained by hand. `safe_download_dir` is the on-disk directory name for downloads, so a drift would make the API and worker disagree about where models live.
- **Impact:** A silent drift in one copy breaks the model-cache layout contract between processes. Everything else is just maintenance weight.
- **Suggested fix:** Delete the worker and API copies and re-export the core versions (`pub use sceneworks_core::...`). Put `format_bytes` and `quote_path` in core as well. Make `memory-adapter` use `sceneworks_core::time`.
- **Confidence:** High

#### [F-048] Five worker implementations of "join a relative path under a root, rejecting non-Normal components", with different strictness
- **Source ID:** W5-007
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `pose_jobs.rs:1123-1132` (`join_project_relative`, returns `Option`, lexical only); `asset_media.rs:9-18` (`lexically_confined_media_path`, plus a canonicalize and `starts_with` check at `:20-26`); `paths.rs:24-37` (`safe_join`); `paths.rs:~905-922` (`safe_project_path`); `iris_task_training.rs:~233-240` (`safe_relative`). Same shape in rust-api (`lib.rs:4110-4125`, `training.rs:3619`).
- **Finding:** `asset_media` resolves a symlink-safe canonical path under the project, but `pose_jobs::resolve_source` takes the lexical `join_project_relative` result and only checks `exists()` (`pose_jobs.rs:1099-1112`). A symlink inside the project dir therefore resolves outside it for pose sources. The `paths.rs` helpers are the documented "worker trust boundary" module, but `pose_jobs` does not use them.
- **Impact:** The sc-8875 traversal fix is applied unevenly. Pose sources are weaker than asset-media sources (only exploitable by someone who can already place a symlink in the project tree, hence Medium/Low).
- **Suggested fix:** Add one `confine_under(root, relative) -> WorkerResult<PathBuf>` in `paths.rs` with an optional canonical (symlink-safe) mode, and route `pose_jobs`, `asset_media`, `safe_join` and `safe_project_path` through it.
- **Confidence:** Medium (symlink exploitability is runtime-dependent)

#### [F-049] architecture_tests.rs enforces a narrow slice of what the docs imply, and it enforces it by text-scanning
- **Source ID:** W5-008
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `architecture_tests.rs:293-470` (job-type matrix, dispatch arms, json import), `:472-1220` (family allow-list inventory, ~750 lines), `:1222-1255` (heartbeat keepalive wiring); `ARCHITECTURE.md:30-33`; `lib.rs:2247-2680`
- **Finding:** Only four things are enforced: (1) every `JobType` has a row in `ARCHITECTURE.md` and a textual `JobType::X =>` arm in `run_utility_job`; (2) no new checkpoint-family allow-list exists outside a recorded inventory; (3) image_jobs modules that invoke `json!` import it; (4) the source-guard await is wrapped in `heartbeat_while_blocking`.
  - **Not enforced.** The layering and cfg rules that CONTRIBUTING and AGENTS state are not tested here at all: the base.rs/candle cfg rule, the "neither build" rule, and the rule that handlers do not carry availability or cache policy (sc-19708).
  - **Dispatch is not compiler-enforced.** `run_utility_job` ends with a `_ =>` arm, so a missing `JobType` does not fail to compile. The test covers that by regex over source text. It counts a cfg-gated arm as present even on builds where it is compiled out.
  - **Scanner limits.** About 60% of the file is a bespoke Rust tokeniser (`slice_const_bodies`, `matching_bracket`, `string_literals`). Its own header concedes that a single-token `sdxl` gate is invisible to it.
  - **Name vs content.** The file is called `architecture_tests` but checks docs-vs-code table parity and one wiring pattern. The routing oracles in core (`mac_rust_supported`/`candle_supported`) are not cross-checked against worker dispatch or `gpu.rs` advertised capabilities.
- **Impact:** People read "architecture_tests passes" as "layering holds". It does not. The family-allow-list scanner is expensive to maintain relative to what it can prove.
- **Suggested fix:** Rename or split into `doc_drift_tests` plus a small set of real structural checks, for example a `cargo metadata` / `syn` test of the module graph or capability-vs-dispatch parity. Replace the `_ =>` arm with an explicit "unsupported" list so the compiler enforces exhaustiveness. Consider syn instead of the hand tokeniser.
- **Confidence:** Medium (the judgement is about intent; the facts above are verified)

#### [F-050] Three incompatible ways to build test `Settings`; one reads the real environment; the struct differs under `cfg(test)`
- **Source ID:** W5-009
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `settings.rs:11,64,108-121,127-155`; `test_env.rs:136-141`; hand-written literals (~28 files, e.g. `downloads.rs:2794,2956,3195`, `caption_jobs.rs:979`, `person_segment.rs:523`, `video_jobs/tests.rs` x5, `image_jobs/base.rs` x4); per-module helpers (`test_settings` in `caption_jobs.rs:970`, `dataset_analysis_jobs.rs:408`, `training_jobs.rs:4367`, `downloads.rs:2947`, `subject_mask_jobs/tests.rs:3`; `settings` in `image_jobs/base.rs:2757,4741,16498`, `image_jobs/tests.rs:28933`, `local_derivation.rs:194`, `external_library_runtime.rs:1155`)
- **Finding:** `Settings::for_test` (30 uses), `test_env::offline_settings()` (44 uses) and 12+ local builders coexist. `offline_settings()` is `..Settings::from_env()`, so it picks up the developer's real `HF_TOKEN`, `SCENEWORKS_ACCESS_TOKEN`, config-dir credentials and data dir, and only pins two URLs. The `resolved_cache` field is `#[cfg(not(test))]`, so test builds compile a different struct and `resolved_cache_policy()` takes a different code path (`settings.rs:108-121`).
- **Impact:** Tests are not hermetic on a machine that has real tokens or cache env vars set, and production-only field handling is never exercised by unit tests. Adding a `Settings` field means editing ~30 test sites.
- **Suggested fix:** Make `Settings` implement a test-only `Default` (or a builder) used everywhere, remove the `cfg(not(test))` on `resolved_cache`, and have `offline_settings` build from `for_test` instead of `from_env`.
- **Confidence:** High

#### [F-051] Fail-queued-job sweeps are copy-pasted five times in jobs_store
- **Source ID:** C1-003
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/jobs_store.rs:2480-2520, 2545-2580, 2640-2680, 2745-2770, 2800-2835` (also the active-job variant at `2389-2400`)
- **Finding:** `fail_stranded_mlx_jobs`, `fail_unsupported_mlx_jobs`, the candle stranded sweep, the platform-reachability sweep and `fail_unsupported_candle_jobs` each repeat the same shape: `select * from jobs where status='queued'`, `collect_jobs`, a filter predicate, then an identical `update jobs set status='failed', stage='failed', message=?, error=?, completed_at=?1, updated_at=?1, worker_id=null where id=? and status='queued'`, then `jobs_by_ids`. Only the predicate, the message and the error text differ. The two "unsupported" sweeps also repeat the `(job, reason)` zip-back dance.
- **Impact:** A change to failure semantics (for example clearing `stage` or adding an event) must be made in six places. The five copies are already ordered inconsistently: some use `format_unix_seconds(now)` and some use `now_unix_seconds()`.
- **Suggested fix:** Extract `fail_queued_jobs(&Transaction, now, candidates: impl Iterator<Item=(&JobSnapshot, message, error)>) -> Vec<String>` and have each sweep supply only its predicate and message.
- **Confidence:** High

#### [F-052] Eleven or more independent copies of the `[A-Za-z0-9_-]` id check, one inside the same file as the canonical helper
- **Source ID:** C1-004
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/store_util.rs:137-142` (canonical `is_safe_id`); `project_store.rs:7831-7836` (`is_safe_track_id`, byte-identical, used at `2966, 3003`); `training_store.rs:2684-2691` (`is_safe_file_stem`); `film_plan.rs:954-960` (`is_safe_plan_id`); `yue2_score/jobs.rs:655-662, 905, 936`; `sceneworks-mcp/src/server.rs:1973-1978`; `apps/rust-api/src/film_planner_connections.rs:405-412`; `apps/rust-api/src/film_planning.rs:1022-1030`
- **Finding:** The same predicate is rewritten with only the length cap differing. The mcp copy's doc comment says "Match the core store's safe-id contract", so it is a manual mirror of `is_safe_id`.
- **Impact:** Any tightening, for example rejecting overlong ids in `is_safe_id` (which today has no length cap, unlike most copies), has to be hand-propagated. `project_store`'s own track-id twin shows the helper is not discoverable.
- **Suggested fix:** Expose one `pub fn is_safe_id_with_max(value, max)` from core (for example in `slug.rs` or `contracts`) and have all callers use it. Delete `is_safe_track_id`.
- **Confidence:** High

#### [F-053] "Confined relative path" validation exists in at least eight variants despite the checkpoint seam's single-source rule
- **Source ID:** C1-005
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/checkpoint_plan_store.rs:1013-1075` (stated canonical `portable_relative_path_parts`); `store_util.rs:129-135` (`is_safe_relative_path`); `catalog_store.rs:2486-2512`; `download_receipt.rs:49-57` (inline); `safetensors.rs:121-133`; `yue2_score/transcriptions.rs:179-184`; `model_artifacts.rs:1150-1175, 1461-1475`; `resolved_cache/materialization.rs:645-655`
- **Finding:** `checkpoint_plan_store.rs:1030-1033` records that five copies were collapsed into one because they "drifted looser", but the store modules and others still carry their own. They disagree. `store_util::is_safe_relative_path` allows `:`, control bytes and `a//b` on Unix. `yue2 safe_artifact_rel` also allows a backslash on Unix. `catalog_store` handles drive prefixes but not control characters. `safe_relative_shard_path` rejects `:` but not control bytes.
- **Impact:** The same string is accepted by one validator and refused by another, depending on the module. `is_safe_relative_path` guards `project_file`, upload and asset media paths (`project_store.rs:732, 1591, 2735, 3333, 4904`), where a `:` or control character is fine on Linux but an alternate-data-stream or odd-name hazard on Windows/macOS.
- **Suggested fix:** Make `portable_relative_path_parts` the shared primitive (move it to `store_util` or a `paths` module) and implement `is_safe_relative_path`, `safe_relative_shard_path`, `safe_artifact_rel`, the catalog validator and the download_receipt inline check as thin wrappers. Keep a Windows-only check only if it adds something.
- **Confidence:** High

#### [F-054] The sniff-then-transcode-or-copy image ingestion block is repeated across core with the `needs_transcode ... .expect` idiom
- **Source ID:** C1-006
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/project_store.rs:3825-3845, 7090-7105`; `training_store.rs:1048-1066, 1125-1140, 2262-2290`
- **Finding:** The pattern `sniff_image_kind_at` then `!is_natively_supported()` then `source_kind.expect("needs_transcode implies a sniffed kind")` then `transcode_to_png(..).map_err(|e| BadRequest("Could not convert {kind.label()} image..."))` else `fs::copy` appears at three sites with the identical error message. Each also re-derives the output extension by hand, and the fallbacks differ (`png` in project_store, `png` or `bin` in training_store).
- **Impact:** The sc-6143 chokepoint is a convention, not a function. A new ingestion path can forget it. The `.expect` calls are only sound by local reasoning.
- **Suggested fix:** Add `media_convert::ingest_image(source, dest_stem) -> Result<(ext, mime)>` returning a typed result, with the extension fallback as a parameter, and replace the three sites.
- **Confidence:** High

#### [F-055] Three different migration strategies across the three SQLite stores, only one of them concurrency-safe
- **Source ID:** C1-007
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/project_store.rs:5296-5360` (version gate, no transaction, no re-check); `jobs_store.rs:612-935` (no version at all, a long `ensure_column` chain replayed at every `initialize`); `catalog_store.rs:3585-3850` (stepwise `BEGIN IMMEDIATE` with the version re-read inside each step); `store_util.rs:204-223`
- **Finding:** catalog_store's `migrate_*_vN` is the right shape. `apply_project_migrations` runs a multi-statement DDL batch plus a check-then-`ALTER` `ensure_column` outside any transaction, and `user_version` is stamped last. Two processes (API and worker open the same project.db) that both see an old version race, and the loser fails with "duplicate column name". The `ensure_column` check-then-act is itself not atomic outside a transaction.
- **Impact:** First start after an upgrade can hit a rare cross-process failure on project.db. Three approaches also mean three places to learn and to get wrong.
- **Suggested fix:** Wrap `apply_project_migrations`' slow path in `BEGIN IMMEDIATE`, re-read `user_version` inside it, and stamp inside the same transaction (the catalog pattern). Longer term, a shared tiny `run_migrations(conn, &[Step])` helper.
- **Confidence:** Medium

#### [F-056] Job status is a typed enum plus three parallel string-slice constants plus hundreds of SQL literals, and the constants are copied elsewhere
- **Source ID:** C1-008
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/contracts.rs:344-374` (`JobStatus`); `jobs_store.rs:72-113` (`ACTIVE_STATUSES`, `TERMINAL_STATUSES`, `PENDING_STATUSES`, `JOB_STATUSES`); `jobs_store.rs:163-212` (three identical `*_statuses_sql` fns); `jobs_store.rs:458` (`Ok("queued")`); `apps/rust-api/src/film_harness.rs:131-132`; `apps/rust-api/src/film_planning.rs:993`
- **Finding:** The `JobStatus` enum is the typed source of truth, but the sets are re-listed as string slices that nothing ties to the enum (no exhaustive test links `JOB_STATUSES` to the variants). `film_harness.rs` keeps a private `TERMINAL_STATUSES` copy whose doc comment says "Statuses the job store treats as terminal". `active_statuses_sql`, `terminal_statuses_sql` and `pending_statuses_sql` are three copies of the same `OnceLock` quoting function. SQL statements hard-code `'queued'`, `'failed'` and so on in about 60 places.
- **Impact:** Adding or renaming a status (the sc-9120 `pending_caption` and `pending_workflow` precedent) needs edits in at least five places that the compiler cannot check.
- **Suggested fix:** Derive the three slices from `JobStatus` methods (`is_terminal`, `is_active`, `is_pending`), generate the SQL lists with one `fn sql_list(&[&str])`, and import the core constant in `film_harness` and `film_planning`.
- **Confidence:** High

#### [F-057] ProjectStore is one 4,200-line impl with 111 public methods plus 8,000 lines of production code in one file
- **Source ID:** C1-011
- **Category:** readability
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/project_store.rs:796-5030` (impl), file prod region `1-8019`; also `catalog_store.rs` (4,300 prod lines) and `jobs_store.rs`
- **Finding:** One `impl ProjectStore` mixes projects registry, film drafts and runs, timelines, uploads and transcoding, assets, keypoint library, training dataset pass-throughs and character pass-throughs. About 30 of its methods are one-line delegates that do `lock_project` then `XStore::new(..).method(..)` (for example `project_store.rs:2758-2923`). Instrumentation (`record_asset_list_filesystem_operation`, 31 call sites across `project_store`, `asset_index`, `character_store`) is woven through the production code.
- **Impact:** Review and merge-conflict cost, and the cross-cutting rules (lock, then migrate, then transact) are re-decided per method; see C1-001.
- **Suggested fix:** Split by aggregate (`registry.rs`, `film_drafts.rs`, `timelines.rs`, `uploads.rs`, `keypoints.rs`), and have character/training/voice delegation go through one `with_locked_project(project_id, |path| ..)` helper.
- **Confidence:** High

#### [F-058] Image-lane enqueue gates live in video_request.rs; image_request depends on the video module
- **Source ID:** C2-002
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/video_request.rs:785` (`image_max_reference_assets`), `:805` (`image_reference_limit_error`), `:836-881` (`ImageDimensionEnvelope`, `image_dimension_envelope`, `image_dimension_error`), `:1091` (`image_sampling_knob_error`); consumer `crates/sceneworks-core/src/image_request.rs:247`
- **Finding:** Five image-only validators (references, dimension envelope/grid, sampler/scheduler menu) are defined in the 6.9k-line video module, while image-only rules sit in image_request.rs (`hires_fix_request_error`, `ordered_image_reference_error`) and image_sampling_controls.rs (cfg interval, scheduler shift, caption overflow, prompt batch). `ImageRequest::from_payload` reaches into `crate::video_request::image_dimension_envelope`. `default_resolution` is also defined twice (image_request.rs:132 and video_request.rs:1311) as one-line wrappers over `declared_resolution`.
- **Impact:** Image validation is spread over three files with the "image" half of one concern in the wrong crate module; a reader fixing an image gate will not look in video_request, and the two request modules are coupled in a direction (image -> video) that has no domain reason.
- **Suggested fix:** Move the image_* items to image_request.rs (or a new image_limits.rs next to image_sampling_controls.rs); share `steps/humanized_number_menu` helpers via a `limits` module used by both lanes.
- **Confidence:** High

#### [F-059] Safetensors header reading is reimplemented 6+ times with different size caps and different strictness
- **Source ID:** C2-003
- **Category:** redundant
- **Severity:** Medium
- **Location:** `lora_family.rs:20,57-95` (16 MiB, checks truncation), `safetensors.rs:11,29-60` (100,000,000, full structural), `checkpoint_inspector.rs:1472-1535` (`MAX_SAFETENSORS_HEADER_BYTES`), `training_store.rs:1351`, `dataset_quality.rs:1754-1766` (no cap at all, in-memory), `apps/rust-api/src/models.rs:11368-11385` (64 MiB, `header_len as usize`)
- **Finding:** Each site opens the file, reads the 8-byte LE length, bounds it with its own constant, reads and parses the JSON. `lora_family.rs` (a LoRA family detector) also owns the public `read_safetensors_header`/`SafetensorsHeaderError`, which unrelated modules (`base_weights.rs:51`) import from there, even though a dedicated `safetensors.rs` exists.
- **Impact:** The same file is "valid" under one gate and "invalid header" under another (16 MiB vs 64 MiB vs 100 MB); a fix such as the sc-6072 truncation check was only applied to one copy.
- **Suggested fix:** Make `safetensors.rs` the single owner (`read_header(path, max)` + one shared cap constant + the truncation check) and have lora_family, base_weights, inspector, rust-api models.rs call it.
- **Confidence:** High

#### [F-060] Builtin-manifest lookup/strip/parse is hand-rolled at each call site, and failures silently become "permissive"
- **Source ID:** C2-005
- **Category:** redundant
- **Severity:** Medium
- **Location:** `film_plan.rs:3173-3181` (`embedded_manifest`), `film_plan.rs:3237-3267` (`model_lora_families`), `minimax_h3_turbo.rs:111-120`, `video_memory_curves.rs:634-642`; test copies in `control_weights.rs:138`, `admission_geometry.rs:243`, `hf_revision_supersessions.rs:339`, `model_usage_policy_tests.rs:10`
- **Finding:** `builtin_manifests.rs` exposes only the raw `BUILTIN_MANIFESTS: &[(&str,&str)]` table, so every consumer does `.iter().find(name).strip_jsonc_comments().from_str()` itself and picks its own failure policy: film_plan returns `""`/empty (so `lora_family_matches_model` becomes permissive), video_memory_curves returns an error, tests `expect`.
- **Impact:** A malformed or renamed embedded manifest degrades film LoRA compatibility checks to "accept everything" instead of failing loudly; the catalog is parsed several separate times per process.
- **Suggested fix:** Add `builtin_manifests::parsed(name) -> &'static Result<Value,_>` (OnceLock per manifest) and migrate callers; keep one failure policy.
- **Confidence:** High

#### [F-061] Film plan LoRA/model compatibility bypasses lora_family's accepted-families rules
- **Source ID:** C2-006
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `film_plan.rs:3237-3280` vs `lora_family.rs:2827-2866` (`extra_compatible_lora_families`, `accepted_lora_families`, `canonical_lora_family`)
- **Finding:** `lora_family_matches_model` does exact `families.contains(&lora.family)` against the raw `loraCompatibility.families` strings. The runtime gate (`validate_lora_compatibility`) normalizes both sides and adds extra-compatible families (chroma<-flux, flux2-klein/dev<-flux2, krea-realtime<-wan-video, scail2<-wan-video).
- **Impact:** A catalog LoRA that the enqueue gate accepts on krea-realtime or scail2 is silently dropped from a film shot (`plan_loras_for_partition` filters it out) or refused by plan validation; spelling variants (`krea2`/`krea-2`/`krea_2`) also fail to match.
- **Suggested fix:** Implement `lora_family_matches_model` via `accepted_lora_families(model_family)` over `canonical_lora_family` values, ideally by calling the same predicate the enqueue gate uses.
- **Confidence:** High

#### [F-062] Production lib carries ~700 lines of test-only workflow-share builder registries naming web JS files
- **Source ID:** C2-007
- **Category:** dead-code
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/workflow_share.rs:655-1335` (`ADVANCED_BUILDERS`, `DEFERRED_ADVANCED_BUILDERS`, `WORKFLOW_WRITE_SEAMS`, `*_BUILDERS`, `WebBuilderRef` paths like `apps/web/src/screens/VideoStudio.jsx`)
- **Finding:** These `pub const` tables are referenced only by `crates/sceneworks-core/tests/workflow_share.rs` (grep finds no other consumer in apps/ or crates/). They describe JS function names in the web app and are compiled into every binary that links core.
- **Impact:** Dead weight in the shipping library, and a Rust library whose production surface embeds frontend file/function names; renaming a JS function breaks core's public constants rather than just a lint.
- **Suggested fix:** Move the registries into the integration test (or a `#[cfg(test)]`/`test-support` module) and keep only `ADVANCED_KEY_RULES` + sanitizer in the library.
- **Confidence:** High

#### [F-063] Training target limit flags are hand-copied per target while sibling flags are injected centrally
- **Source ID:** C2-008
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/training.rs:879-975` (`builtin_training_targets` loops), per-target literals e.g. `:2108-2140` (`supportsWeightNoise` x16, `supportsSubjectMaskLoss` x15, `supportsDepthAnchoring` x14, `"lrSchedulers"` x17, `"mixedPrecision"` x19); `:102-148` (`TECHNIQUE_KEYS: [&str; 45]`)
- **Finding:** Face-loss, resolution-bucket, latent-perceptual and x0-decoder flags are stamped by loops after the list, but weight-noise/gradient-noise/depth-anchoring/body-loss/subject-mask flags are copied as JSON literals into each of ~20 target constructors (each 100+ lines). `TECHNIQUE_KEYS` is a hand-counted array that mixes consts with bare strings (`"bodyProportionWeight"`, `"normalEvery"` ...) for the same technique family.
- **Impact:** Adding a trainer means editing ~10 keys in a new literal and remembering the array length; a missed flag silently under-advertises the target (a worker drift test is the only guard).
- **Suggested fix:** Table-drive the support flags (`TECHNIQUE_SUPPORT: &[(&str, &[&str])]`) and stamp them in the existing post-loop; derive `TECHNIQUE_KEYS` from per-module key slices (and make it a `&[&str]`).
- **Confidence:** Medium

#### [F-064] Hand-synced model-id and family tables in lora_family
- **Source ID:** C2-009
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `lora_family.rs:2937-2940` (`same_ltx_generation` hardcodes `ltx_2_3`, `ltx_2_3_eros`, `ltx_2_5`), `:2949` (`_14b` suffix convention), `:3026-3033`, `:3158-3163` (`model_id == "ltx_2_5"` literal), `:554-600` (`model_adapter_for_family`) vs `:606-741` (`model_capabilities_for_type_and_family`); also `image_request.rs:370,383` (`family != "sdxl"`, `model == "realvisxl_lightning"`), `h3_keyframes.rs:3` (exact `"minimax_h3"`) vs `video_request.rs:568` (`starts_with("minimax_h3")`)
- **Finding:** The LTX generation gate, Wan size/conditioning class, hires-fix model exceptions and H3 predicates key on literal ids or prefixes that must be edited by hand when the manifest adds a model (a new `ltx_2_6` is not `same_ltx_generation` of anything). `model_adapter_for_family` has no `flux2` (or `krea-realtime`, `minimax-h3`) arm although `model_capabilities_for_type_and_family` has `("image","flux2")` and IMPORT_SUPPORTED_FAMILIES lists flux2, so imported FLUX.2 entries get capabilities but no `adapter`. `extra_compatible_backbone_is_image_conditioned` has a redundant `"krea-realtime" => false, _ => false`. Two adjacent comment blocks restate the same Lens/Chroma note (`:628-641`).
- **Impact:** New model ids silently fall through to "not interchangeable" or "no adapter"; H3 id predicates disagree (exact vs prefix) between modules.
- **Suggested fix:** Read generation/size-class/adapter facts from manifest fields (`loraCompatibility.baseGeneration`, `adapter`) instead of string literals; unify the `is_*_model` predicates in one place.
- **Confidence:** Medium

#### [F-065] Four independent family-name vocabularies for base/checkpoint detection
- **Source ID:** C2-010
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `lora_family.rs:807-840` (`diffusers_class_name_to_family`) and `:1467+` (`SIGNATURES`), `base_weights.rs:1214-1330` (`detect_transformer_family`), `checkpoint_inspector.rs:2006-2035` (`normalize_family`), `base_weights.rs:335` (`IMPORT_SUPPORTED_FAMILIES`)
- **Finding:** Each detector emits family tokens independently with inconsistent spelling (`krea_2` vs `krea-2`, `mage-flow` vs `mage_flow`) and heuristics: `checkpoint_inspector::normalize_family` uses bare substring `contains` on descriptor strings (`"wan"`, `"flux"`, `"krea"`, `"ltx"`) and maps `cosmos` to `anima`; `normalize_model_family` (lora_family.rs:732) stores `krea-2` as the default `loraCompatibility.families` entry while `canonical_lora_family` says the stored token is `krea_2`.
- **Impact:** A new family has to be added in at least four tables; substring matching can misclassify names (e.g. any "…wan…"/"…ltx…" architecture string) and the checkpoint inspector and LoRA detector can disagree on the same family.
- **Suggested fix:** One `family` module with a typed enum + canonical string and alias list, used by all four detectors.
- **Confidence:** Medium

#### [F-066] Hugging Face repo-id validation is implemented four times with different rules
- **Source ID:** A1-002
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/rust-api/src/models.rs:235-251` (called at `:3725`), `crates/sceneworks-worker/src/model_jobs.rs:5067-5100` (`validate_hf_repo_id`), `crates/sceneworks-core/src/workflow_share.rs:2165-2182` (`hf_repo_id`), `crates/sceneworks-core/src/model_artifacts.rs:1117-1132` (`validate_repository`)
- **Finding:** The API accepts `a--b`, a leading `-`, and a bare `x.y`, and does not require an alphanumeric first character. The worker rejects `--`, `..` and leading/trailing `-`. core's `hf_repo_id` requires an alphanumeric first character. core's `validate_repository` allows leading `-` and `--`. The API validates `repo.trim()` but the caller keeps the untrimmed `payload.repo`.
- **Impact:** A repo the API accepts (201, job queued) can be rejected later by the worker, as the comment at `models.rs:3797-3807` warns for install ids. Path-safety rules for a value joined into `models--owner--name` directories drift between layers.
- **Suggested fix:** Add one `sceneworks_core::hf_repo::parse_repo_id() -> Result<RepoId, _>` using the strictest rule set. Make all four call sites use it, and store the trimmed value.
- **Confidence:** High

#### [F-067] Four byte-identical helpers are copy-pasted across rust-api, worker and core
- **Source ID:** A1-003
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/rust-api/src/lib.rs:4380-4386` (`json_size_to_u64`), `:4387-4403` (`allow_pattern_matches`, `pattern_matches`), `:4405-4415` (`quote_huggingface_repo`), `:4417-4430` (`format_bytes`). Twins: `crates/sceneworks-worker/src/imports.rs:268-283`, `crates/sceneworks-core/src/model_artifacts/artifact_selection.rs:609-625`, `crates/sceneworks-worker/src/util.rs:4-27` (`format_bytes`, `quote_path`), `crates/sceneworks-worker/src/payload.rs:4-8`.
- **Finding:** The glob matcher, including the `cfg!(windows)` lowercasing, exists three times. Percent-quoting and `format_bytes` exist twice. `json_size_to_u64` exists twice. `models.rs` consumes the rust-api copies (`:11494-11600`).
- **Impact:** The API's size estimate and the worker's allow-pattern filter must agree on which files a tier contains. A fix to one copy, such as a case-folding change, silently desynchronises install-state from what the worker downloads.
- **Suggested fix:** Make core's `artifact_selection` matcher `pub` and have rust-api and worker call it. Move `format_bytes` and the quote helper into core, or use the `percent-encoding` crate.
- **Confidence:** High

#### [F-068] ComfyUI external LoRA and external base-model scanners are a copy-paste pair
- **Source ID:** A1-004
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/rust-api/src/external_loras.rs:56-107,154-205,262-290` vs `apps/rust-api/src/external_base_models.rs:58-118,205-262,275-286,707-722`
- **Finding:** Both files carry the same size+mtime memo cache (`get`/`insert`/`retain_seen`), the same depth-bounded canonicalize-and-contain walk, and `relative_display_name`. They also carry `unique_id`, which differs only in prefix and fallback slug. The doc comment at `external_base_models.rs:~205` says "Mirrors `external_loras::collect_adapters`".
- **Impact:** The walk is the symlink-containment security boundary (`canonical.starts_with(root)`). A hardening fix applied to one copy, for example Windows case or the TOCTOU between canonicalize and read, will miss the other. The two files already differ in hidden-file handling: `collect_weight_files` calls `is_hidden_file`, and `collect_adapters` relies on `is_safetensors_file`.
- **Suggested fix:** Create a generic `ExternalScanCache<T>` and `walk_confined(root, depth, cap, predicate)` in a shared module (core `external_roots` is a natural home). Parameterise `unique_id(prefix, fallback)`.
- **Confidence:** High

#### [F-069] Install-health safetensors check is a weaker, drifted twin of core's validator
- **Source ID:** A1-005
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `apps/rust-api/src/models.rs:11368-11388` (`safetensors_header_is_valid`) vs `crates/sceneworks-core/src/safetensors.rs:11,29-110` (`file_is_structurally_valid`, `MAX_HEADER_BYTES = 100_000_000`)
- **Finding:** The API copy accepts any JSON object header of up to 64 MiB. It does not require a tensor, does not check dtype/shape/byte-range agreement, and does not check contiguous coverage. Core's version does all of that, with a 100 MB limit. `torn_shard_indexes_in` at `models.rs:11407` correctly uses core's `missing_indexed_shards`, so the same function mixes both standards.
- **Impact:** A `{}`-header placeholder or truncated body passes `diffusers_component_safetensors_are_valid`, so the diffusers snapshot is reported installed while the worker's stricter check later rejects it. Headers between 64 and 100 MiB are judged differently by the two.
- **Suggested fix:** Delete the local helper and call `sceneworks_core::safetensors::file_is_structurally_valid`.
- **Confidence:** Medium (the weaker check may have been a deliberate cheapness choice; both read only the header)

#### [F-070] "Is this relative path safe" predicate reimplemented at least eight times
- **Source ID:** A1-006
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/store_util.rs:129-135`, `crates/sceneworks-core/src/safetensors.rs:121-133`, `crates/sceneworks-core/src/checkpoint_inspector.rs:2293-2296`, `crates/sceneworks-worker/src/iris_task_training.rs:232-238`, `crates/sceneworks-worker/src/catalog_image_fetch.rs:1800-1812`, `apps/rust-api/src/iris_task_training.rs:48-62`, `apps/rust-api/src/training.rs:~880` (inline canonicalize variant), `crates/sceneworks-core/src/model_artifacts.rs` (`validate_relative_path`)
- **Finding:** Each is a variation of "all components are `Normal`", with different handling of `\`, `:`, NUL, whitespace and empty strings. The rust-api iris `item_image` does not validate at all. It pushes every component of `item.path`, including `RootDir` or `..`, onto the root.
- **Impact:** Traversal defences are inconsistent. A future caller picking the lenient variant gets a hole. `PathBuf::push` of an absolute component replaces the root.
- **Suggested fix:** Provide one `sceneworks_core::safe_path::relative_normal_components(&str) -> Option<PathBuf>` and use it everywhere.
- **Confidence:** Medium

#### [F-071] Model lookup by id: five copies of the same 404 block, each deep-cloning the whole catalog
- **Source ID:** A1-007
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/rust-api/src/models.rs:905-916,1206-1217,1832-1843,2603-2614` (and `:2086-2093`), `apps/rust-api/src/yue2_jobs.rs:165-176`, `apps/rust-api/src/generation.rs:866-870,1423-1427`, `apps/rust-api/src/model_sources.rs:353-358`, `apps/rust-api/src/models.rs:4135-4140` (`model_catalog`), `:4157` (`selection_catalog = models.clone()`)
- **Finding:** `model_catalog()` does `snapshot.as_ref().clone()`, a deep copy of every manifest `Value` (the builtin manifest is ~2 MB), and callers then `.into_iter().find(id == x)` to take one entry. `ApiError { status: NOT_FOUND, detail, context: None, code: None }` is hand-built 23 times (`rg "status: StatusCode::NOT_FOUND"`: models.rs ×5, loras.rs ×5, yue2_jobs.rs ×5, others). `ApiError` has no `not_found()` constructor, though every other status has one (`error.rs:31-130`).
- **Impact:** Every model download, convert, delete, import and utility-job admission pays a multi-MB deep clone to read one row. It is then repeated by `model_sources.rs:353` via `refresh_live_external_availability`. The boilerplate hides which sites differ.
- **Suggested fix:** Add `ApiError::not_found(detail)`. Add `async fn catalog_entry(state, id) -> Result<Value, ApiError>` on `model_catalog_snapshot` that clones only the matched row. Use it in all of the above sites. `yue2_jobs::catalog_entry` already does exactly this.
- **Confidence:** High

#### [F-072] models.rs is a 15.8k-line module, 46% of it inline tests, mixing at least nine concerns
- **Source ID:** A1-008
- **Category:** readability
- **Severity:** Medium
- **Location:** `apps/rust-api/src/models.rs` (production ~8.5k lines; test modules at `:1675,4907,5202,10041,10231,11859,11992,13757,14025,15135,15493,15788` total ~7.4k lines)
- **Finding:** One file holds the following, all behind a single `use super::*` glob into the 6.3k-line `lib.rs` root:
  - licence acknowledgement gating, `:509-900`
  - download jobs, `:900-1450`
  - convert jobs, `:1578-1950`
  - delete and tier removal, `:1950-3140`
  - import, `:3142-4130`
  - receipt repair and install state, `:4490-8250`
  - catalog snapshot building, `:9485-10560`
  - HF/diffusers cache-health probes, `:10874-11470`
  - size estimation with its single-flight cache, `:255-470,11482-11600`

  Tests run up to 2.5k lines in one module (`download_receipt_tests`, `:5202-7686`).
- **Impact:** Compile and review blast radius is the whole file. Navigation relies on grep. Three sibling modules (`model_sources`, `model_library`, `model_cache`) already show the intended `models/` split.
- **Suggested fix:** Convert to `models/{mod,license,download,convert,delete,import,catalog,install_state,hf_cache_health,size_estimate}.rs` with `models/tests/*.rs`. Do tests first (zero-risk, halves the file), then peel off `hf_cache_health` and `size_estimate`, which are leaf modules. Replace `use super::*` with explicit imports as each piece moves.
- **Confidence:** High

#### [F-073] God functions with test-only branches interleaved in production bodies
- **Source ID:** A1-009
- **Category:** readability
- **Severity:** Medium
- **Location:** `apps/rust-api/src/training.rs:1771-2383` (`create_training_job`, 612 lines), `apps/rust-api/src/catalogs.rs:2342-2696` (`schedule_catalog_scan`, 354 lines, ~15 `#[cfg(test)]` locals/branches), `apps/rust-api/src/models.rs:3652-3943` (`queue_model_import_job`, 291), `:900-1178` (`create_model_download_job`, 278), `apps/rust-api/src/catalogs.rs:670-946` (`materialize_catalog_results`, 276), `apps/rust-api/src/loras.rs:1073-1331` (`queue_lora_import_job`, 258)
- **Finding:** `create_training_job` sequentially does target resolution, preset metadata, dataset normalisation, id allocation, guardrails, Iris resource checks, plan build, manifest pre-registration, and job creation in one body. `schedule_catalog_scan` has a deliberate "production has no hook fields" design (`server.rs:~458`), but the hooks sit inline in the retry loop.
- **Impact:** These are the highest-risk paths (queueing real GPU jobs). They cannot be unit-tested in pieces, and a change to ordering is easy to get wrong. The model-vs-LoRA import pair also duplicates manifest-entry and payload construction.
- **Suggested fix:** Split `create_training_job` into `resolve_target`, `prepare_dataset`, `allocate_output`, `run_guardrails`, `build_and_enqueue` with a small context struct. Move catalog scan test hooks behind a `ScanHooks` trait object that is a ZST in non-test builds.
- **Confidence:** High

#### [F-074] Contract-bearing helpers copy-pasted byte-for-byte across rust-api, worker and core
- **Source ID:** A2-002
- **Category:** redundant
- **Severity:** Medium
- **Location:** `lib.rs:4445-4462` / `crates/sceneworks-worker/src/paths.rs:4-22` / `crates/sceneworks-core/src/model_artifacts/artifact_selection.rs:26-44` (`safe_download_dir`); `lib.rs:4387-4403` / `worker/src/imports.rs:268-284` / `artifact_selection.rs:609-625` (`allow_pattern_matches`+`pattern_matches`); `lib.rs:4417-4429` / `worker/src/util.rs:4-16` (`format_bytes`); `lib.rs:4405-4415` / `worker/src/util.rs:19-30` (`quote_huggingface_repo` = worker `quote_path`); `lib.rs:6336-6350` / `worker/src/settings.rs:215-230` (`env_string`, `env_path_or`)
- **Finding:** `safe_download_dir` decides the on-disk `models/<safe>` directory and the core copy's doc comment says "Byte-identical to the API ... and the worker", i.e. the duplication is known and held together by a comment. Three copies of the glob matcher decide which files a download keeps. The worker already depends on core and rust-api depends on both, so there is a legal home (core) for all of them.
- **Impact:** A fix to one copy (e.g. a Windows case-folding or Unicode change in `safe_download_dir`) silently desynchronizes where the API looks for a model from where the worker wrote it.
- **Suggested fix:** Keep one pub fn in `sceneworks-core` (`artifact_selection` already exports `safe_download_dir`), delete the other copies, and re-export from worker if its public API needs it. Do the same for the glob matcher and `format_bytes`.
- **Confidence:** High

#### [F-075] `lib.rs` (6,353 lines) is still a grab-bag and a `use super::*` prelude; decomposition plan
- **Source ID:** A2-003
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `apps/rust-api/src/lib.rs` (whole file); 44 of 61 sibling files start with `use super::*;`
- **Finding:** Previous splits (error.rs, server.rs, auth.rs, defaults.rs, dto.rs) moved the obvious pieces but lib.rs still holds, by line range: inline `web_assets` module with hand-rolled Accept-Encoding negotiation (2972-3343, ~370 lines); project-file serving incl. Range, ETag, thumbnails, workflow-strip (2376-2970, ~600); public-job redaction / path scrubbing (3420-3945, ~520); a 1,012-line router constructor (1293-2304, see A2-004); job payload validators for image/video/audio/prompt-enhancement/timeline/person (4866-6262, ~1,400 lines, with their MAX_*/AUDIO_* consts); LoRA family / HF-cache / trash helpers (4131-4835, ~700); in-process utility worker and parent-death watchdog (690-905); upload streaming and stale sweeps (926-1135); ~110 lines of `use` that every child module inherits through `use super::*`. 47 `#[cfg(test)]` items are interleaved with production code.
- **Impact:** Every module depends implicitly on everything in lib.rs, so a child module cannot be compiled, tested or moved on its own, and any edit to lib.rs recompiles the whole crate's dependency picture. It also hides real layering: nothing stops `film_planner.rs` from reaching `validate_video_job`.
- **Suggested fix:** Extract in this order (each is a pure move): `web_assets.rs`; `project_files.rs` (get_project_file + helpers + consts); `job_redaction.rs` (public_job_snapshot, redact_*); `job_validation/{image,video,audio,prompt}.rs`; `lora_helpers.rs`/`hf_cache.rs`; `inprocess_worker.rs`. Then replace `use super::*` with explicit imports in the leaf modules and keep lib.rs as `mod` list + re-exports (<500 lines).
- **Confidence:** High

#### [F-076] 1,000-line router constructor; auth coverage is implicit and untested as a table
- **Source ID:** A2-004
- **Category:** readability
- **Severity:** Medium
- **Location:** `apps/rust-api/src/lib.rs:1293-2304` (`create_app_with_state_mode`), `lib.rs:2276-2284`, `auth.rs:222-239`
- **Finding:** One function builds AppState (1293-1490), the MCP service and a ~740-line flat chain of `.route(...)`. Body-limit overrides are attached ad hoc per route (e.g. 1608, 1619, 1669, 1687, 1694, 1719, 1739, 1744, 2067, 2077, 2159, 2188), so the effective limit of a route is only discoverable by reading its neighbours. Separately, `requires_token` is allow-by-default: anything not under `/api/` or `/mcp` is ungated (auth.rs:222-239) because the SPA fallback needs it.
- **Impact:** A future route registered outside `/api/` (e.g. `/metrics`, `/ws`, `/debug`) would be public with no failing test. Reviewing "which routes lack auth" requires reading 740 lines.
- **Suggested fix:** Give each area module a `pub(crate) fn routes() -> Router<AppState>` and compose them; put upload limits in the owning module. Add a test that enumerates the router (or a const route list) and asserts every non-`PUBLIC_PATHS` route returns 401 for a non-loopback peer without a token, and that `app_fallback` is the only ungated non-`/api` handler (only when `embed-web`).
- **Confidence:** High

#### [F-077] Stringly-typed error dispatch and a catalog-specific constructor used as a generic 503
- **Source ID:** A2-006
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `error.rs:204-208`, `error.rs:76-83`, uses at `catalogs.rs:2205,2209,2260,2300`
- **Finding:** `ProjectStoreError::BadRequest(detail)` is promoted to 409 only if `detail.starts_with("Film draft revision conflict:")`, so a wording change in core turns a conflict into a 400. `ApiError::service_unavailable` hardcodes `code: Some("catalog_preflight_unavailable")`, which is also attached to "Parquet validation is shutting down" and scan-interrupted errors.
- **Impact:** Clients keyed on `code` get a wrong machine-readable reason; the 409 mapping is one core string edit from breaking silently.
- **Suggested fix:** Add a `ProjectStoreError::Conflict` variant in core and map it; make `service_unavailable` take a code argument.
- **Confidence:** High

#### [F-078] Two ad-hoc "atomic write" helpers diverge from core's durable `atomic_write`
- **Source ID:** A2-007
- **Category:** redundant
- **Severity:** Medium
- **Location:** `film_planning.rs:1067-1085` (`write_json_file`/`write_latest_operation`), `film_harness.rs:2133-2150` (`write_atomically`), vs `crates/sceneworks-core/src/store_util.rs:37-50` (`pub(crate) atomic_write`)
- **Finding:** `write_latest_operation` writes a temp then `rename`s with no `sync_all` (the exact crash window core closed under sc-8949/F-147). `write_atomically` fsyncs the file but uses a fixed temp name derived from the target (`latest.json` -> `latest.json.tmp`), so two concurrent writers collide on the same temp file, and it skips the parent-dir fsync. Core's version is `pub(crate)`, which is why rust-api reimplemented it.
- **Impact:** Planning-operation state (`latest.json`) can be left truncated/empty after a power loss; concurrent harness writers can corrupt each other's temp. Worker has 2 more copies (`catalog_semantic_jobs.rs:607`, `catalog_image_fetch.rs:1823`).
- **Suggested fix:** Make core's `atomic_write` `pub` (or add a `sceneworks_core::fsutil::write_atomic`) and use it in all of these.
- **Confidence:** High

#### [F-079] `load_settings` silently resets a corrupt `settings.json`, and the next save persists the reset; read-modify-write is unlocked
- **Source ID:** X1-002
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/desktop/src/settings.rs:320-336,363-368`; callers `:1005,1050,1070,1354-1397`
- **Finding:** `load_settings` does `read_to_string(...).ok().and_then(from_str(...).ok()).unwrap_or_default()`. A truncated, hand-edited or newer-schema file reads as default settings with no log. Every mutator (`set_credential`, `set_data_dir`, `set_gpu_memory_limit`, ...) then does load, mutate, `save_settings`, which atomically overwrites the file with defaults plus the one change. There is also no mutex around the cycle, so two concurrent commands (async commands run on the pool, sync ones on the main thread) can lose each other's update.
- **Impact:** One bad byte drops the data-dir and HF-home overrides, the credential metadata (the secrets stay in the keychain but are orphaned) and the remote-access flags. The app then starts against an empty default data directory, which looks like data loss.
- **Suggested fix:** Distinguish "missing" from "unparseable". On parse failure, log it, copy the file to `settings.json.corrupt-<ts>`, and refuse to save over it until the user acts. Serialize mutators with one process-wide `Mutex<()>` around load-modify-save.
- **Confidence:** High (code path), Medium (frequency).

#### [F-080] App-update download has no timeout, so a stalled connection wedges the update flow forever
- **Source ID:** X1-003
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/desktop/src/update.rs:177-204` (esp. `:195`); plugin hook `vendor/tauri-plugin-updater-2.11.0/src/updater.rs:698-700`
- **Finding:** `check_for_update` is wrapped in `tokio::time::timeout(30s)`, but `download_app_update` calls `update.download(|_, _| {}, || {}).await` with no timeout. The plugin applies a timeout only if `Update.timeout` is set, and this app never sets it (nor `updater_builder().timeout`). `session.downloading = true` is set before the await.
- **Impact:** On a flaky or captive network the download never returns. `downloading` stays true, so every later call returns "An update is already downloading." and the user must force-quit. This is the same class as F-049, which was fixed only for the GPU provisioner.
- **Suggested fix:** Set `update.timeout` (a generous total, or per-read if the plugin supports it) or wrap the future in `tokio::time::timeout`. In either case reset `downloading` on the error path, ideally with a drop guard.
- **Confidence:** High.

#### [F-081] `save_image_export` takes a `Vec<u8>` of up to 256 MB over JSON IPC, and the size check runs after the allocation
- **Source ID:** X1-004
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `apps/desktop/src/settings.rs:1241-1285` (`MAX_EXPORT_BYTES`, `save_image_export`)
- **Finding:** A Tauri command argument typed `Vec<u8>` arrives as a JSON array of numbers. A 256 MB export is roughly 1 GB of JSON text, which the shell parses into `Vec<u8>` before `image_bytes.len() > MAX_EXPORT_BYTES` is evaluated. The doc comment claims the ceiling "prevents an untrusted webview payload from causing an unbounded allocation", but the allocation happens first. The command is granted to any `http://127.0.0.1:*` origin.
- **Impact:** Large editor exports can OOM or stall the shell (and `set_remote_access_*` shares its process). A compromised page can force multi-GB allocations. The comment is misleading.
- **Suggested fix:** Take `tauri::ipc::Request` and read the raw body (`InvokeBody::Raw`) with the filename in a header. Alternatively, have the webview POST to the API and let the API write the file after the dialog. At minimum, correct the comment.
- **Confidence:** Medium (depends on the Tauri 2 IPC serialization of `Vec<u8>`).

#### [F-082] Windows and Linux GPU-runtime provisioners duplicate the download, verify, marker and extract machinery
- **Source ID:** X1-005
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/desktop/src/cuda_provision.rs:282-290,543-560,562-623,625-668,759-763` vs `apps/desktop/src/linux_cuda_provision.rs:168-190,208-243,528-593,646-649`; `apps/desktop/src/cuda_provision_check.rs`
- **Finding:** `top_marker_current`, `all_component_markers_current`, `write_marker`, `invalidate_markers`, the stream-to-disk plus incremental-SHA-256 plus mismatch error block of `fetch_component`, the `enclosed_name`/basename zip-extract loop, and the connect(10s)/read(60s) client builder are near copy-pastes. Only the filename filter (`.dll` vs `.so*`), the destination layout and the promotion step differ. `cuda_provision_check.rs` was meant to be the shared, cfg-free home, but it holds only `dir_has_dll`, `component_marker`, `write_component_marker` and `component_provisioned`.
- **Impact:** A security-relevant path (hash verify and archive confinement) is maintained twice, and a future fix to one can miss the other. The tests are duplicated too.
- **Suggested fix:** Move `download_verified(client, url, sha256, dest) -> Result<(), String>`, the marker helpers and `build_client()` into the shared module. Parameterize extraction with a `Fn(&str) -> bool` filename predicate. Keep the per-platform manifests and promotion separate.
- **Confidence:** High.

#### [F-083] MLX and Candle adapter twins: roughly 140 lines are byte-identical and many more differ only by a name or a platform noun
- **Source ID:** X1-006
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-memory-adapter/src/bin/mlx.rs` (33,825 lines) vs `candle.rs` (13,117 lines); e.g. `mlx.rs:9560` vs `candle.rs:6914` (`instantid_kps`), `mlx.rs:11063` vs `candle.rs:7413` (`qwen_edit_lightning_adapter`), `mlx.rs:11888` vs `candle.rs:3891` (`validate_mage_plan_identity`), `mlx.rs:3649` vs `candle.rs:1102` (`strategy_name`), `mlx.rs:22890` vs `candle.rs:10931` (`flux_one_snapshot_root`), `mlx.rs:22902` vs `candle.rs:10941` (`staged_pulid_bundle`)
- **Finding:** Measured by whitespace-exact function-body comparison, there are 15 identical functions (139 lines) between mlx.rs and candle.rs, 4 between `mlx_iris.rs` and `candle_iris.rs`, 3 between the supra2 files, and 6 between the wan_scail2 files (about 87 lines, including `target_geometry`, `plane`, `validate_mode`). Another group differs by 2-10 lines of substitution, for example `instantid_artifact_fingerprint` (the only difference is `Path` vs `std::path::Path`), `qwen_edit_snapshot_root`, `planned_memory_strategy` and `validate_planned_fingerprint_is_producible` (only `mlx_`/`candle_production_fingerprint`). The two files share 54 function names, 6 struct names (`Flux2Arm`, `MageArm`, `QwenEditArm`, `QwenEditArtifact`, `QwenEditArtifactSource`, `QwenEditLightningSource`) and 26 const names. The cross-lane protocol already lives in `lib.rs` (`protocol::*`), but these helpers were never lifted into it.
- **Impact:** Every fixture or validation rule must be changed in two 13k-34k line files, and the Cargo.toml comments cite the sc-22738 incident that this transcription pattern caused (a repair landed in one copy and not the other).
- **Suggested fix:** Move the backend-neutral pieces (snapshot-root and fingerprint helpers, `*Source` structs, mage/qwen-edit plan validators, fixture builders) into `lib.rs` or a `common` module that both binaries `#[path]`-include. Keep only the engine-specific load and run bodies per lane.
- **Confidence:** High.

#### [F-084] Drift: MLX re-spells `planned.target.*` extraction about 80 times while Candle added local helpers that the shared protocol lacks
- **Source ID:** X1-007
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `candle.rs:1123-1135` (`planned_provider`, `planned_mode`); `mlx.rs:3778,4469,5364,...` (12 occurrences of `"planned.target.mode must be a string"`), `mlx.rs:22561` (and 16 occurrences of `planned.target.provider must be a string` across the mlx files); `crates/sceneworks-memory-adapter/src/lib.rs` (`planned()` only)
- **Finding:** Candle extracted `planned_provider`/`planned_mode` helpers in its own file. MLX still does `protocol::planned(request)?.pointer("/target/…").and_then(Value::as_str).ok_or_else(|| "planned.target.… must be a string")` inline: 80 `protocol::planned(request)?` calls in mlx.rs vs 46 in candle.rs, 92 "must be a string" error literals in mlx.rs vs 48 in candle.rs. `validate_mage_mode` is a concrete example of the two lanes now having different bodies for the same check (`mlx.rs` vs `candle.rs:validate_mage_mode`).
- **Impact:** Error text and failure semantics for malformed plans are inconsistent across lanes. A change to the plan schema needs about 130 edits.
- **Suggested fix:** Add `protocol::planned_str(request, "/target/mode")` and typed accessors in `lib.rs`, and use them from every arm in both lanes.
- **Confidence:** High.

#### [F-085] Candle's "weights-free fingerprint" allow-list is hand-typed strings, while the MLX twin reads engine constants
- **Source ID:** X1-008
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `candle.rs:1945-1951` vs `mlx.rs:5593-5600`; contrast `crates/sceneworks-memory-adapter/Cargo.toml:41-45` (sc-22738 rationale)
- **Finding:** `mlx.rs` builds `is_turnkey_weights_free_fingerprint` from `runtime_macos::providers::lens::memory_strategy::STATIC_BEHAVIOR_FINGERPRINT` and `LEGACY_TEXT_ENCODER_FINGERPRINT`. The Candle version `starts_with`-matches four literal strings (`"kolors-candle-registry-behavior-v1"`, `"lens-candle-registry-behavior-v1"`, `"ideogram4-candle-request-scoped-staged-residency-static-v1"`, `"kolors-candle-staged-chatglm-unet-f32-vae-v1"`). Cargo.toml says the project deliberately reads route revisions from the engine "rather than transcribed into the adapter" after exactly this kind of drift broke CUDA run 34272596969.
- **Impact:** An engine-side rename of one of these fingerprints makes Candle treat a weights-free provider as weights-bearing (or the reverse) with no compile error.
- **Suggested fix:** Read the constants from the candle-gen crates, as the SDXL arm already does.
- **Confidence:** Medium (I did not diff the strings against the pinned engine constants).

#### [F-086] `mlx.rs` and `candle.rs` are god files, and the MLX `run_*` arm functions are 200-650 lines each
- **Source ID:** X1-009
- **Category:** readability
- **Severity:** Medium
- **Location:** `mlx.rs:17927` area `run_ltx_with_admission` (650 lines, `:16687`), `run_bernini` (531, `:19549`), `run_z_image_reference_loaded` (435, `:3921`), `run_mage_provider` (430, `:12251`), `run_minimax_h3` (427, `:18517`), `run_krea_control` (390, `:9895`), `run_flux2` (383, `:7444`); `candle.rs` `run` (658 lines), `run_five_rung_reference_loaded` (419), `load_five_rung_generator` (372)
- **Finding:** 33.8k and 13.1k lines each, with 28 `run_*` functions in mlx.rs over 200 lines and about 40 `#[cfg(test)]` blocks interleaved (272 `#[test]` in mlx.rs, 90 in candle.rs, so a large fraction of the file is tests). Only LTX-2.5, Iris, Supra2 and Wan/SCAIL-2 were split into `#[path]` submodules, and those splits are the only ones with matching Candle twins.
- **Impact:** Review, `rust-analyzer` and incremental builds on these crates are painful, and the twin-drift problems in X1-006/007/008 are hard to spot by eye.
- **Suggested fix:** Continue the existing `#[path = "mlx_<family>.rs"]` split per arm (flux_one, mage, qwen_edit, sdxl, ltx), and move test modules into sibling files. Do this together with the dedup in X1-006.
- **Confidence:** High.

#### [F-087] "Safe id" validator reimplemented in at least 10 places; core's copy is `pub(crate)`
- **Source ID:** X1-010
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-mcp/src/server.rs:1859-1869,1973-1979`; `crates/sceneworks-mcp/src/yue2.rs:215-228`; `crates/sceneworks-core/src/store_util.rs:137-142` (the origin, `pub(crate) fn is_safe_id`); `crates/sceneworks-core/src/training_store.rs:2690`; `crates/sceneworks-core/src/project_store.rs:7835`; `crates/sceneworks-core/src/yue2_score/jobs.rs:659,905,936`; `crates/sceneworks-worker/src/image_jobs/base.rs:2530`; `apps/rust-api/src/film_planner_connections.rs:406-411`; `apps/rust-api/src/film_planning.rs:1028`
- **Finding:** `valid_project_id`'s doc comment says "Match the core store's safe-id contract", but it re-spells `is_ascii_alphanumeric() || '_' | '-'` instead of calling it, because core's function is crate-private. MCP alone has three variants: `valid_job_id` (trim, `-_`), `record_id` (adds a 64-char cap) and `valid_project_id`; they disagree on length and trimming.
- **Impact:** A security-relevant path-component gate has several subtly different definitions. A tightening (for example rejecting a leading `-` or capping length) will be applied unevenly.
- **Suggested fix:** Export one `sceneworks_core::ids::is_safe_id` (and a length-bounded variant) and replace the copies.
- **Confidence:** High.

#### [F-088] Atomic temp+rename file write reimplemented in 8 places with different durability and collision guarantees
- **Source ID:** D1-001
- **Category:** redundant
- **Severity:** Medium
- **Location:** canonical `crates/sceneworks-core/src/store_util.rs:37-47` (pub(crate)); copies `apps/rust-api/src/manifest.rs:242-262` (no fsync), `crates/sceneworks-worker/src/manifest.rs:225-242` and `:244-261` (fixed `<ext>.tmp` name, no dir fsync), `apps/rust-api/src/film_harness.rs:2133-2149` (fixed temp name), `crates/sceneworks-worker/src/catalog_semantic_jobs.rs:607-618`, `crates/sceneworks-core/src/catalog_store.rs:4319-4337` (own `random_hex`, no dir fsync), `crates/sceneworks-core/src/resolved_cache.rs:3497-3515`, `apps/desktop/src/settings.rs:350-361` (fixed `json.tmp`, no fsync at all). Legit specializations (not counted): `catalog_image_fetch.rs:1823`, `credentials.rs:289-307`.
- **Finding:** `store_util::atomic_write` was hardened for two real bugs (sc-1633 colliding writers on a deterministic temp name; sc-8949 rename-before-data-durable). It is `pub(crate)`, so every other crate wrote its own variant, and several reintroduce exactly those two defects: worker `manifest.rs` and `film_harness.rs` use a deterministic temp path; rust-api `manifest.rs` and desktop `settings.rs` skip `sync_all`. The worker `write_json_value` is called on project timeline tracks (`media_jobs.rs:2036`), the same files the API writes through `store_util`.
- **Impact:** Two concurrent writers to one worker manifest/track path interleave bytes into one `.tmp`; a power loss after rename can leave a zero-length desktop `settings.json` (data-dir/HF-home overrides + credential metadata). Prior F-023 (2026-07-12) fixed only the `builtin_manifests` instance of this class.
- **Suggested fix:** Promote `store_util::atomic_write` to a `pub` core API (e.g. `sceneworks_core::fs_atomic::{write, write_json}`; error type generic or `io::Error`), keep a thin async wrapper (`spawn_blocking`) for tokio callers, and delete the six plain copies. Canonical = `store_util` version (unique token, `sync_all`, parent-dir fsync).
- **Confidence:** High

#### [F-089] Hex/SHA-256 helpers copied across core, worker, rust-api, desktop and the memory adapter (13 streaming file-hash loops, 10 hex encoders, 4 random-hex generators)
- **Source ID:** D1-002
- **Category:** redundant
- **Severity:** Medium
- **Location:** streaming file hash (identical `File::open` + 1 MiB `read` loop + `hasher.update`): `crates/sceneworks-core/src/checkpoint_plan_store.rs:1193`, `checkpoint_ingest.rs:654`, `media_convert.rs:203 (file_content_hash)`, `managed_file.rs:85`, `model_artifacts/artifact_selection.rs:181`, `project_store.rs:637`, `checkpoint_inspector.rs:2263`, `crates/sceneworks-worker/src/downloads.rs:1084`, `catalog_image_fetch.rs:2217`, `catalog_analysis.rs:1301`, `iris_curricula.rs:118`, `apps/desktop/src/linux_cuda_provision.rs:203`, `crates/sceneworks-memory-adapter/src/bin/mlx_ltx25.rs:965`. Byte-slice hex: `apps/rust-api/src/film_harness.rs:825`, `apps/rust-api/src/catalogs.rs:1556 (hex_sha256)`, `crates/sceneworks-worker/src/vector_jobs.rs:64`, `yue2_jobs.rs:741`, `qwen_prompt_rewrite.rs:269 (hex_digest)`, `core/src/license_acknowledgments.rs:90`, `core/src/yue2_score/abc.rs:661`, `apps/desktop/src/cuda_provision.rs:549`, `training_subject_masks.rs:340`, `image_jobs/base.rs:4228`. Random hex: `core/src/store_util.rs:167`, `catalog_store.rs:4340`, `resolved_cache.rs:3647`, `yue2_score/jobs.rs:1230 (fresh_hex_id)`.
- **Finding:** Core has no shared `hash`/`hex` module, so each call site re-derives `format!("{byte:02x}")`/`{:x}` and the read loop. `store_util::random_hex` and `catalog_store::random_hex` are line-for-line twins (different error types only).
- **Impact:** Pure maintenance cost today, but the hashes are identity/receipt keys (checkpoint plans, download receipts); a buffer-size or error-handling fix has to be found in 13 places. `yue2_score/jobs.rs` additionally carries a non-cryptographic time/address fallback that no sibling has.
- **Suggested fix:** Add `sceneworks_core::hashing::{hex(bytes), sha256_hex(bytes), sha256_file_hex(path)}` and `random_hex(n)`; worker's heartbeat-wrapped `sha256_file` keeps wrapping the core function in `spawn_blocking`. Canonical = core (every consumer already depends on it).
- **Confidence:** High

#### [F-090] `safe_download_dir` exists three times byte-for-byte; sibling path helpers (`safe_join`, `is_safe_media_path`, slugify) also duplicated
- **Source ID:** D1-003
- **Category:** redundant
- **Severity:** Medium
- **Location:** `crates/sceneworks-core/src/model_artifacts/artifact_selection.rs:26-43`, `crates/sceneworks-worker/src/paths.rs:4-22`, `apps/rust-api/src/lib.rs:4445-4462`. Also `crates/sceneworks-worker/src/paths.rs:24-37` vs `crates/sceneworks-worker/src/video_jobs/seedvr2.rs:429-438` (`safe_join`); `apps/rust-api/src/film_harness.rs:1950-1962` vs `apps/rust-api/src/film_harness/references.rs:1109-1121` (`is_safe_media_path`, identical); slugify family: `core/src/slug.rs:1` (dash), `apps/rust-api/src/lib.rs:4562 slugify_lora_id` (same algorithm, `_` separator), `worker/src/gpu.rs:1217 slugify_worker_id_part`; `format_bytes` at `worker/src/util.rs:4` and `apps/rust-api/src/lib.rs:4417`.
- **Finding:** The core copy's own doc comment says it is "Byte-identical to the API ... and the worker; receipts written by either are read here" — the duplication is acknowledged and held together by a comment. This is the managed `models/<safe>` directory key, a security- and data-layout-critical function (path confinement + receipt lookup).
- **Impact:** If one copy changes, receipts and install directories silently stop matching across API/worker; the path-confinement helpers are security-relevant and have already forked in behavior (`safe_join` returns `WorkerResult` vs `Option`).
- **Suggested fix:** Keep only the core `safe_download_dir` (worker and API already depend on core) and re-export; move `safe_join` and `is_safe_media_path` next to `store_util::is_safe_relative_path`; parameterize one `slugify_with(sep, allowed_extra)` in `core::slug`.
- **Confidence:** High

#### [F-091] SCENEWORKS_* boolean env toggles accept different truthy sets depending on which file parses them
- **Source ID:** D1-004
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `apps/rust-api/src/lib.rs:547-549` and `:591` (`1/true/TRUE/yes/YES`); `apps/rust-api/src/server.rs:259-274` (`SCENEWORKS_MLX_REQUIRED`, `CANDLE_REQUIRED`, `TRUST_LOOPBACK`, `RUN_UTILITY_INPROCESS`: `1/true/TRUE/True`, no `yes`); `apps/rust-api/src/models.rs:11485-11490` (`1/true/TRUE/yes/YES`); `crates/sceneworks-worker/src/settings.rs:241-251 env_bool` (case-insensitive `1/true/yes/on` and explicit false set); `apps/desktop/src/webkitgtk.rs:8-20` (case-insensitive incl. `on`); `crates/sceneworks-core/src/resolved_cache.rs:177-183` (`1/true/0/false` only, rejects `yes`). Also `env_string`/`env_path_or` duplicated at `apps/rust-api/src/lib.rs:6336-6350` and `worker/src/settings.rs:215-230`, `env_secs` at `rust-api/src/server.rs:357`.
- **Finding:** At least six parsers for the same convention. Concretely: `SCENEWORKS_TRUST_LOOPBACK=yes` is silently false while `SCENEWORKS_ALLOW_OPEN_BIND=yes` is true; `SCENEWORKS_MLX_REQUIRED=on` is false in the API but `SCENEWORKS_BACKEND_MLX_ENABLED=on` is true in the worker.
- **Impact:** An operator following one doc line gets a security-relevant toggle (`TRUST_LOOPBACK`, open-bind override) silently off; fail-safe direction here, but the inverse mismatch (`ALLOW_OPEN_BIND`) fails open for `YES` and closed for `Yes`.
- **Suggested fix:** Add `sceneworks_core::env::{flag, string, path_or, secs}` (case-insensitive `1/true/yes/on`, documented) and route all SCENEWORKS_* reads through it; canonical = worker `env_bool` semantics.
- **Confidence:** High

#### [F-092] Web `fallbackModels` seed has drifted from `builtin.models.jsonc` on fields no parity test covers
- **Source ID:** D1-005
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `apps/web/src/modelCatalog.js` (e.g. `instantid_realvisxl` samplers/schedulers ~L551-552); manifest `config/manifests/builtin.models.jsonc`; parity tests only cover `image` axes (`apps/web/src/imageAxesParity.test.js`), control modes, video geometry, job types, memory floors.
- **Finding:** Diffing the 46 seed entries against the manifest (script run locally) shows today: `instantid_realvisxl` seed offers samplers `euler_a, dpmpp, unipc` and schedulers `karras, exponential` (not valid keys in `packages/schemas/model-manifest.schema.json` `samplerKey`; manifest declares only `default/euler/heun/dpmpp_2m` and scheduler `default`); `chroma1_flash` seed menus are `default/euler/heun` + `default` vs the manifest's full menu; capabilities differ for `sdxl`, `realvisxl`, `illustrious_xl_v1/v2` (seed lacks `image_detail`, `image_inpaint`), `flux2_klein_9b` (lacks `edit_image`, `image_to_image`), and `z_image_edit` (lacks `image_to_image`). The comment at the Qwen 2.1 seed asserts an `imageLimitsParity` that asserts "both directions", but no such test exists in `apps/web/src` (only the files listed above).
- **Impact:** Until the live catalog loads (or if it fails), Image Studio offers sampler keys the API/engine reject and hides edit/inpaint/detail actions the model supports.
- **Suggested fix:** Generate the seed from the manifest at build time (as `previewSupport.json` already is) or add a parity test over `capabilities`, `limits.samplers/schedulers/resolutions` and `defaults` for every seeded id; correct the nonexistent test reference in the comment.
- **Confidence:** Medium (runtime exposure only on pre-catalog/fallback path)

#### [F-093] `image_jobs` is still an `include!` ladder and has tripled in size since prior F-020
- **Source ID:** D1-006
- **Category:** readability
- **Severity:** Medium
- **Location:** `crates/sceneworks-worker/src/image_jobs.rs:3833-3910...` (24 `include!` lines), `crates/sceneworks-worker/src/image_jobs/base.rs:19649` (nested `include!("moss.rs")`), `image_jobs/moss.rs:3201` (nested `include!("moss_dispatch_tests.rs")`).
- **Finding:** Previously F-020 (2026-07-12, still present, worse): `base.rs` was 6,273 lines, is now 19,649; `image_jobs.rs` 4,204; the module is 53 files / ~98.8k lines in one namespace with order-sensitive includes and cfg gating on `base.rs` that CONTRIBUTING.md calls a "trap" (sc-10404, sc-8390).
- **Impact:** The cfg trap documented in CONTRIBUTING exists *because* of the include substrate; every new lane adds name-collision and dead-code-under-`-D warnings` risk that only the Linux "neither" build reveals.
- **Suggested fix:** Convert leaf lanes to real `mod`s with explicit `use super::`, starting by splitting `base.rs`'s tier-resolver block and moving the cfg gate onto `mod` declarations.
- **Confidence:** High


### Low findings — Rust backend

#### [F-094] Bearer token survives a same-host scheme or port change on redirect
- **Source ID:** W3-007
- **Category:** security
- **Severity:** Low
- **Location:** `downloads.rs:1812-1818`
- **Finding:** The redirect check compares only `host_str()`. An `https://host/..` to `http://host/..` (or another port) redirect keeps `Authorization: Bearer`. `parse_lora_source_url_with_private` validates only the first URL's scheme, and the redirect branch accepts http or https.
- **Impact:** A hostile or misconfigured host that downgrades a redirect leaks the stored token in clear text. This is narrow: the host must be the one the user attached a credential to.
- **Suggested fix:** Compare `(scheme, host, port)` and drop the bearer on any difference, or refuse https to http downgrades when a credential is attached.
- **Confidence:** High

#### [F-095] HF tree pagination follows an unvalidated `Link: rel=next` URL with the HF token attached
- **Source ID:** W3-008
- **Category:** security
- **Severity:** Low
- **Location:** `downloads.rs:782-800` and `877-899` (`next_page_url`)
- **Finding:** The next-page URL is taken verbatim from the response header and fetched through `with_hf_auth`, with no check that it shares the origin of `huggingface_base_url`.
- **Impact:** A hostile configured mirror can redirect the bearer token to a third-party host. HF itself will not.
- **Suggested fix:** Reject a next URL whose origin differs from `base_url`.
- **Confidence:** Medium

#### [F-096] LoRA source copy follows symlinks inside the allowed root
- **Source ID:** C1-010
- **Category:** security
- **Severity:** Low
- **Location:** `crates/sceneworks-core/src/character_store.rs:1351-1391, 1413-1426`
- **Finding:** `assert_allowed_lora_source` canonicalises only the top-level source, then `copy_dir_recursive` uses `is_dir()` and `fs::copy`, both of which follow symlinks, for every entry beneath it. A symlink placed inside `data/loras/<x>/` or `project/loras/<x>/` makes the character LoRA import copy an arbitrary readable file or tree into the project (where `project_file` can then serve it).
- **Impact:** Needs write access to the managed LoRA folder (for example a hostile archive extraction), so low. It defeats the stated "must be inside the app-managed folders" guard.
- **Suggested fix:** Use `symlink_metadata` per entry and skip or reject symlinks, or re-canonicalise each entry against the allowed root. Reuse an existing confined-copy helper if one exists (see `resolved_cache` materialization).
- **Confidence:** Medium

#### [F-097] `project_file` serves any file under the project root, including project.db and internal markers
- **Source ID:** C1-016
- **Category:** security
- **Severity:** Low
- **Location:** `crates/sceneworks-core/src/project_store.rs:4899-4931`; route at `apps/rust-api/src/lib.rs:2417-2430`
- **Finding:** After the confinement check the store serves whatever is found, with the mime guessed from the extension, so `project.db`, `.asset-index-dirty`, sidecar JSON, `trash/` and `training/` content are all reachable by relative path. The only content gate is the `moss_output_publication::require_path` pre-check, which concerns one feature.
- **Impact:** In LAN or remote mode (ticketed media) a media ticket exposes more than media. Low because it needs a valid ticket or session and the data is the user's own.
- **Suggested fix:** Allow-list served top-level directories (`assets/`, `generated/`, `exports/`, and so on) or deny dot-files and `project.db`.
- **Confidence:** Medium

#### [F-098] `validate_lora_import_source_path` leaks path existence and is mis-named for model imports
- **Source ID:** A1-012
- **Category:** security
- **Severity:** Low
- **Location:** `apps/rust-api/src/lib.rs:4479-4506`, used by `apps/rust-api/src/models.rs:3827` and `apps/rust-api/src/loras.rs:1186`
- **Finding:** A nonexistent path returns "LoRA sourcePath not found: {path}". An existing path outside the roots returns "must be inside app-managed ...". The raw client path is echoed in the message. The same function validates model imports (so a model import returns "LoRA" errors).
- **Impact:** This is the existence oracle that `model_library.rs:97-99` and `checkpoint_library.rs:432-437` say they avoid. Impact is limited to authenticated clients.
- **Suggested fix:** Return one generic "sourcePath must be inside an approved folder" for both cases. Rename to `validate_import_source_path` and drop "LoRA" from the text.
- **Confidence:** High

#### [F-099] `resolve_project_confined_asset_path` is lexical-only, unlike `project_file`
- **Source ID:** A2-010
- **Category:** security
- **Severity:** Low
- **Location:** `lib.rs:4100-4129` vs core `project_store.rs:4909-4923`
- **Finding:** It rejects non-`Normal` components but never canonicalizes, so a symlink inside the project dir (e.g. from an imported/zipped project) resolves outside it. `get_project_file` does canonicalize+`starts_with`.
- **Impact:** Worker-side reads of an arbitrary file named by a crafted asset record; low because creating the symlink needs write access to the project dir.
- **Suggested fix:** Reuse `ProjectStore::project_file`-style canonical confinement.
- **Confidence:** Medium

#### [F-100] Media ticket is one shared, indefinitely-sliding token for all project files
- **Source ID:** A2-012
- **Category:** security
- **Severity:** Low
- **Location:** `tickets.rs:104-124` (`issue_sliding`), `auth.rs:279-291`
- **Finding:** `issue_sliding` returns the same `latest` ticket to every caller and re-arms its TTL on each refresh, and it unlocks `GET /projects/*/files/*` for every project. A URL leaked via history, Referer, screenshot or a log keeps working as long as any client refreshes, and rotation never happens.
- **Impact:** Leaked media URL becomes a long-lived bearer credential for the whole library (LAN mode). Bounded by being read-only media.
- **Suggested fix:** Cap absolute lifetime (rotate after N minutes regardless of refresh) and add `Referrer-Policy: no-referrer` on the media responses.
- **Confidence:** Medium

#### [F-101] `Settings` derives `Debug` with the access token; token also taken via argv in the CLI
- **Source ID:** A2-013
- **Category:** security
- **Severity:** Low
- **Location:** `server.rs:59-73`; `bin/film-harness.rs:130,516,1363,1487,1574`
- **Finding:** `#[derive(Debug, Clone)] pub struct Settings { pub access_token: String ... }` means any future `{:?}`/`?settings` in a log prints the LAN password. No current leak found (grepped). `film-harness --token TOKEN` puts the secret in process args, and the same `--token`/env fallback is re-implemented four times.
- **Impact:** Latent secret-in-logs risk and `ps` exposure.
- **Suggested fix:** Wrap the token in a redacting newtype (as `ApiClientConfig` avoids `Debug`), prefer env/stdin for the CLI and parse once.
- **Confidence:** High

#### [F-102] No request-level timeouts or concurrency limits on a listener that can bind `0.0.0.0`
- **Source ID:** A2-014
- **Category:** security
- **Severity:** Low
- **Location:** `lib.rs:2276-2304` (layer stack), `server.rs:829,1117` (`axum::serve`)
- **Finding:** Layers are body limit, auth, CORS, compression and timing only; no `TimeoutLayer`, no connection/concurrency cap, no hyper header-read timeout. Authenticated callers can also hold up to a 256 GiB model upload (`MAX_MODEL_UPLOAD_BYTES`, lib.rs:420) open indefinitely, and unauthenticated peers can hold connections and reach the `/auth/verify` and `/health` handlers.
- **Impact:** Slowloris-style resource exhaustion in LAN mode; low for loopback-only desktop.
- **Suggested fix:** Header-read timeout and a global `ConcurrencyLimitLayer`; per-route idle timeout on upload routes.
- **Confidence:** Medium

#### [F-103] Config-write IPC commands granted to the remote origin lack the native confirmation that the LAN toggles have
- **Source ID:** X1-011
- **Category:** security
- **Severity:** Low
- **Location:** `apps/desktop/capabilities/default.json:10-44`; `apps/desktop/src/settings.rs:1005,1050,1070,898,915` vs `:780-840`
- **Finding:** `set_remote_access` and `set_remote_access_password` require a native OS dialog (sc-13610), but `set_data_dir`, `save_storage_setup`, `set_model_library` (all persist paths that become spawn environment on the next launch) and `restart_app` are callable by any page on `http://127.0.0.1:*` with no confirmation. `set_data_dir` accepts any absolute path.
- **Impact:** A webview XSS can retarget the data or model directory (for example to a network share), and the app uses it on the next start, which gives data-exfiltration or poisoning leverage. This is lower than the LAN toggle but the same trust model.
- **Suggested fix:** Apply the same native-confirm gate (or at least a Tauri dialog showing the new path) to path-changing commands, and consider a separate capability for first-run-only commands.
- **Confidence:** Medium.

#### [F-104] `asset_media_path` lets `..` segments through; only the preview path is filtered
- **Source ID:** X1-013
- **Category:** security
- **Severity:** Low
- **Location:** `crates/sceneworks-mcp/src/server.rs:2001-2014`, used at `:640-660` (inline fetch) and `:1178-1180`; compare the filter at `:1245`
- **Finding:** Preview paths are rejected if they start with `/` or contain `..`, but the main asset path from `/file/path` or `/path` is only backslash-normalized and left-trimmed, then percent-encoded per segment (`..` is not encoded). The reqwest URL parser collapses `..` before sending, and the request carries `X-SceneWorks-Token`.
- **Impact:** A poisoned job result (the path originates in worker output / imported workflow metadata) could make the MCP fetch another API route with the token and return the body as "image" bytes. Low because the source is server-written.
- **Suggested fix:** Apply the same `..`/absolute filter inside `asset_media_path` and encode `.` / `..` segments.
- **Confidence:** Low.

#### [F-105] `safe_download_dir` is byte-identical in three crates
- **Source ID:** W1-006
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/sceneworks-core/src/model_artifacts/artifact_selection.rs:26-43`; `crates/sceneworks-worker/src/paths.rs:4-21`; `apps/rust-api/src/lib.rs:4445-4462`
- **Finding:** The core copy's doc comment says it is "byte-identical" to the API and worker copies because receipts written by either are read by the others. The worker and API both depend on core and could simply call it.
- **Impact:** If the three drift, a download directory written by one component is not found by another, and nothing guards against it.
- **Suggested fix:** Delete the worker and API copies and use `sceneworks_core::model_artifacts::artifact_selection::safe_download_dir` (keep `paths::safe_download_dir` as a re-export if the worker API needs it).
- **Confidence:** High

#### [F-106] Five `WeightsSource` to `&Path` helpers and a verbatim `path_weight_bytes` twin
- **Source ID:** W1-007
- **Category:** redundant
- **Severity:** Low
- **Location:** `conditioning_fit.rs:207` (`weights_source_path`) and `conditioning_fit.rs:232-238` (`path_weight_bytes`); `image_jobs/base_admission.rs:14-26` (`source_path`, `path_weight_bytes`, both identical); `text_encoder_selection.rs:81-92` (`source_path`, `source_kind`); `mlx_fit_gate.rs:5428`; `image_jobs/instantid.rs:121`; inline matches at `transform_cache.rs:72`, `audio_jobs.rs:5738`, `iris_training.rs:1411`
- **Finding:** The same two-arm match, and the file-or-directory byte sizer, are re-written in several modules. `base_admission.rs` already imports from `conditioning_fit`'s neighbours but re-declares its own copies.
- **Impact:** Small, but the on-disk sizer is part of a safety gate. A fix to hidden-file or symlink handling would need to be applied to every copy.
- **Suggested fix:** Add `WeightsSourceExt::path()` (or one `pub(crate) fn weights_source_path`) and make `conditioning_fit::path_weight_bytes` `pub(crate)`. Delete the other copies.
- **Confidence:** High

#### [F-107] Candle Wan tier-order doc claims to mirror the macOS order, but the default differs
- **Source ID:** W1-008
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `video_jobs/candle.rs:425-455` vs `video_jobs/wan.rs:466-473` (compare `wan.rs:455-465` rationale)
- **Finding:** The Candle doc says it "mirrors the macOS `wan_tier_order`", but with no explicit pick Candle tries `["q8","q4"]` and MLX tries `["q4","q8"]`. The MLX comment documents q4-first as a deliberate video carve-out (sc-10859: no Q8 UI lever, OOM risk). The Candle comment justifies q8 with the "app-wide default" and does not mention the carve-out. MiniMax-H3 on Candle (`minimax_h3.rs:1569`) defaults q4.
- **Impact:** Whether a default Wan job loads Q4 or Q8 depends on the backend, and the comments contradict each other. A reader cannot tell if the Candle behaviour is intended.
- **Suggested fix:** Decide the Candle default explicitly and say so in the Candle comment (VRAM headroom may justify the difference), or unify both through W1-001's shared helper.
- **Confidence:** Medium (the difference is real; whether it is intended is unclear)

#### [F-108] Three near-identical MLX model-dir resolvers (Bernini, SCAIL-2, Krea Realtime) and scattered env reads
- **Source ID:** W1-009
- **Category:** redundant
- **Severity:** Low
- **Location:** `video_jobs/bernini.rs:53-71`; `video_jobs/scail2.rs:108-126`; `video_jobs/krea_realtime.rs:415-437`; related older style at `video_jobs/svd.rs:71`, `video_jobs/vace.rs:189`; `video_admission.rs:1083` (`WAN_VAE_BUDGET_GIB`)
- **Finding:** Each resolver is "`local_mlx_dir(env, name)` then `huggingface_snapshot_dir(repo)` then an error that spells out `<data>/models/mlx/<name>`", differing only in three strings. SVD and VACE still hand-roll the env lookup. About 24 `std::env::var("SCENEWORKS_…")` reads sit in image and video job code outside `Settings`, plus the un-prefixed operator knobs `LTX_GEMMA_DIR` and `WAN_VAE_BUDGET_GIB`.
- **Impact:** Adding a family means copying a function. Env knobs are undiscoverable and cannot be seen in settings or tests without process-global mutation.
- **Suggested fix:** Add `resolve_mlx_family_dir(settings, env_key, name, repo)` and have the three call it. Collect the env knobs in `Settings` or one `env_knobs` module.
- **Confidence:** High

#### [F-109] `text_encoder_selection::allowed_roots` re-implements `paths::allowed_model_roots` with weaker normalisation
- **Source ID:** W1-010
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `text_encoder_selection.rs:236-254` (discovery) vs `paths.rs:105-119` (`allowed_model_roots`, used at job time in `prepare_selected_text_encoder`, text_encoder_selection.rs:~514)
- **Finding:** Discovery builds the allow-list from `std::path::absolute`/`canonicalize` and filters candidate sources with `starts_with`. `std::path::absolute` does not collapse `..` on Unix, so a catalog seed like `<data>/x/../../etc` passes `starts_with(data_dir)` lexically. The job-time check (`normalize_absolute_path` + `ensure_path_under`) re-validates, so this is not exploitable today. Discovery and execution still apply two different definitions of "inside an allowed root".
- **Impact:** An out-of-root path can be offered as a selectable option (path-free ids, but its existence and shape are probed). The job later fails with a less obvious error. Any change to the root list must be made twice.
- **Suggested fix:** Make `allowed_roots` take already-normalised input (reuse `paths::normalize_absolute_path`), or share one root-list builder between the API-side discovery and the worker.
- **Confidence:** Medium

#### [F-110] Cache wrappers still triplicate their idle-timeout env, default and seam boilerplate
- **Source ID:** W1-011
- **Category:** redundant
- **Severity:** Low
- **Location:** `generator_cache.rs:55-56`, `transform_cache.rs:44-45`, `refine_model_cache.rs:42-43` (same env name `SCENEWORKS_GENERATOR_CACHE_IDLE_SECONDS`, default 300, each with its own `*_idle_timeout(raw)` wrapper and tests); `generator_cache.rs:63-98` (four cfg variants of `capture_backend_committed_bytes`)
- **Finding:** F-019's `cache_thread` extraction left the three wrappers each declaring the same env constant, default and one-line wrapper. `transform_cache` even re-asserts the string literal in a test (line 520) to guard against divergence. The refine cache's const is named `REFINE_…` but reads the "GENERATOR" env var.
- **Impact:** Changing the default or env name needs three edits. The guard test shows divergence is a known risk.
- **Suggested fix:** Put `CACHE_IDLE_SECONDS_ENV`, `DEFAULT_CACHE_IDLE_SECONDS` and `idle_timeout_from_env()` in `cache_thread.rs`.
- **Confidence:** High

#### [F-111] Model-specific Bernini receipt and surface logic sits inside the generic `video_admission` module
- **Source ID:** W1-012
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `video_admission.rs:53-59,280-720` (Bernini receipts, shape math), `2744-3035` (`bernini_surface_is_exact`, about 220 lines, `bernini_rv2v_receipt_matches_surface`); `video_admission/` holds only `tests.rs`
- **Finding:** The module's own doc for the exceeded-bound lookup says "Nothing here names a model", yet roughly 800 of its 2,700 non-test lines are Bernini specific. `video_admission/` exists as a directory but contains no submodules. `fit_ladder_for_tier` in `krea_control_fit.rs:392-785` is the same shape (one 394-line, 12-parameter function).
- **Impact:** The generic admission funnel is hard to read. Adding another family-specific receipt (Krea V2V, SCAIL-2 already follow the same pattern) grows the same file.
- **Suggested fix:** Move Bernini and the other per-family receipt builders to `video_admission/{bernini,scail2,krea}.rs`. Split `fit_ladder_for_tier` into candidate building, evidence building (`make_evidence`) and verdict mapping. `MemoryEvidenceDimensions { .. }` literals appear in 12 places (`vram_gate.rs`, `mlx_fit_gate.rs`, `candle_memory_strategy.rs`, `krea_control_fit.rs`) and could share a builder.
- **Confidence:** Medium

#### [F-112] About 30 `cfg_attr(<inverse cfg>, allow(dead_code))` attributes paper over mis-scoped items
- **Source ID:** W1-013
- **Category:** dead-code
- **Severity:** Low
- **Location:** `video_jobs/seedvr2.rs:221,655,675,771,1001,1127,1277`; `video_admission.rs:801,816,827,858,894,913`; `image_jobs/stream.rs:44,58,255,307,340,386`; `image_jobs/strict_control.rs:136,450,516`; `image_jobs/base.rs:3098,5187,10014`; `image_decode.rs:31,60`; `video_jobs/seedvr2.rs:221` (`check_seedvr2_output_disk`, `#[allow(dead_code)]`, called only from tests)
- **Finding:** Items are compiled under `any(macos, candle)` and then given `allow(dead_code)` for the opposite cfg, rather than gated to where they are used. `check_seedvr2_output_disk` is unconditionally dead in production and kept "as a regression seam".
- **Impact:** The warning that would reveal a truly dead function on one backend is silenced everywhere. This is a recurring maintenance tax on every new helper.
- **Suggested fix:** Gate the items with the same `cfg` as their callers (or a `cfg_aliases` alias such as `native_backend`), then drop the `allow`s. Delete `check_seedvr2_output_disk` and keep the test on `check_seedvr2_scratch_disk`.
- **Confidence:** High

#### [F-113] SeedVR2 free-disk probe shells out to `df` (no timeout) while `fs2::available_space` is already a dependency
- **Source ID:** W1-014
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `video_jobs/seedvr2.rs:176-208`; compare `yue2_transcription.rs:545` and `apps/rust-api/src/training.rs:3546`, which use `fs2::available_space`
- **Finding:** The Unix branch spawns `df -k -P` synchronously and parses column 4. The Windows branch already uses `fs2::available_space`, which also works on Unix. A hung network mount would block the worker thread (the call is made from `check_seedvr2_scratch_disk`, reached on the async job path at seedvr2.rs:1466).
- **Impact:** Three disk-space probes with three behaviours. A stuck `df` stalls the job before any heartbeat region is established.
- **Suggested fix:** Use `fs2::available_space` on all platforms, preferably through one shared `disk_free_bytes()` helper in `util.rs`.
- **Confidence:** Medium

#### [F-114] Duplicate stub-render scaffolding between image and video; duplicated test helpers across lane files
- **Source ID:** W1-015
- **Category:** redundant
- **Severity:** Low
- **Location:** `image_jobs.rs:3794-3816` (`stub_rgb8`, `lerp`) vs `video_jobs/mod.rs:1465-1500` (`stub_video_rgb8`, `lerp`); `image_jobs/iris.rs:49`, `image_jobs/supra2.rs:37`, `image_jobs/sdxl_control.rs:753` (`isolate_hf_cache`, identical); `iris.rs:58-72` vs `supra2.rs:45-59` (shipped-catalog-entry loader); `image_jobs/iris.rs:655`, `supra2.rs:335` (`spawn_job_api`)
- **Finding:** The gradient generator and `lerp` are copied between the image and video stub paths. Three test modules each define `isolate_hf_cache`, and iris/supra2 each define a near-identical "load builtin catalog entry" helper and mock job API.
- **Impact:** Minor; every new lane test module copies the same fixtures again.
- **Suggested fix:** Put `isolate_hf_cache` and `shipped_catalog_entry(id)` in `test_env.rs`, and the gradient helper in `util.rs`.
- **Confidence:** High

#### [F-115] On-device harnesses repeat HF-cache discovery (and the `~/.cache/huggingface` path) instead of using `smoke_support` / `hf_home`
- **Source ID:** W1-016
- **Category:** redundant
- **Severity:** Low
- **Location:** `wan_t2v_14b_tier_build.rs:42-76`, `wan_i2v_14b_tier_build.rs:45-82`, `wan_ti2v_5b_tier_build.rs:64-100` (`resolve_tokenizer`, three copies differing in env name, repo and one extra lookup); `resolution_sweep.rs:138-169`, `footprint_measure.rs:69`, `pid_tier_mlx_smoke.rs:34`, `sana_mlx_smoke.rs:70`, and six `cached_turnkey_root()` copies (`lens_base_q4_mlx_smoke.rs:59`, `lens_turbo_q4`, `chroma1_base_q4`, `krea_turbo`, `sdxl_base_q8`, `sensenova_fast_q8`)
- **Finding:** Each harness hard-codes `$HOME/.cache/huggingface/hub` and scans `snapshots/*` itself, although `sceneworks_core::hf_home::huggingface_hub_cache_dir` exists and `smoke_support.rs` already hosts shared helpers (`report_tier`, `env_or`).
- **Impact:** All of these ignore `HF_HOME` / `HF_HUB_CACHE` and a relocated cache, so the ignored tests fail with a misleading "not cached" panic on such machines. There are about 12 copies to update.
- **Suggested fix:** Add `smoke_support::cached_snapshot(repo, sentinel)` built on `huggingface_hub_cache_dir`, and `resolve_wan_tokenizer(native, env_key, repo, extra_candidates)`.
- **Confidence:** High

#### [F-116] Mochi/SVD/generic fit gates exist twice with line-for-line twins
- **Source ID:** W2-007
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/sceneworks-worker/src/mlx_fit_gate.rs:5316-5332,5362-5372,5744-5800`; `vram_gate.rs:386-403,406-413,2353-2415`
- **Finding:** `fit_decision`, `sequential_overflow_gb` and `mochi_fit_error`/`mochi_too_big_error` are the same function, differing only in `budget.total_gb` (MLX) versus `budget.free_gb` (Candle) and the message wording. The Mochi arithmetic was already hoisted to `fit_gate`; the decision step and the error struct were not. `legacy_unified_reserve(_total_gb)` (`fit_gate.rs:66`) takes an argument it ignores.
- **Impact:** Low. The doc comments justify the different prose, but the compare-with-epsilon logic must be fixed twice.
- **Suggested fix:** Give `fit_gate` `fit_decision(needed, available_gb)` and `overflow(needed, available_gb)`, and have both gates pass in their own budget number. Drop the unused parameter.
- **Confidence:** High

#### [F-117] `hw.memsize` probed in two production places; the fit-gate one is a synchronous, un-timed subprocess
- **Source ID:** W2-009
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/sceneworks-worker/src/mlx_fit_gate.rs:5662-5688`, `crates/sceneworks-worker/src/gpu.rs:769-800`
- **Finding:** `probe_total_unified_memory_bytes` uses `std::process::Command` with no timeout and is called from async admission paths (`depth_jobs.rs:597`, `iris_restore.rs:546`, `yue2_admission.rs:2653`). `gpu.rs::sysctl_memsize_mb` does the same query, async, with a 2s timeout and its own cache.
- **Impact:** One worker holds two caches and two code paths for one fact. A stuck `sysctl` would block an executor thread once.
- **Suggested fix:** One cached probe (the gpu.rs one, with a sync accessor for the cache) used by both.
- **Confidence:** Medium

#### [F-118] `BYTES_PER_GIB` and GiB helpers re-declared in about 8 places despite the core constant
- **Source ID:** W2-010
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/sceneworks-core/src/memory_anchor.rs:4214`; re-declared in `memory_strategy.rs:401`, `candle_memory_strategy.rs:50`, `vector_admission.rs:41`, `iris_restore.rs:480`, `sensenova_jobs.rs:106`, `image_jobs/instantid.rs:108`, `core/video_memory_curves.rs:40`; `fit_gate.rs:133` correctly aliases it. Helpers: `yue_admission.rs:754 gib`, `mlx_fit_gate.rs:1366 gib_to_bytes` (thin wrapper), `mlx_fit_gate.rs:1370 decimal_gb_to_bytes` vs `memory-adapter/src/bin/candle.rs:643`
- **Finding:** The constant is defined once as `pub const` in core and re-declared locally elsewhere. The core doc for `gib_to_bytes` (4228-4232) still describes `mlx_fit_gate::gib_to_bytes` as "the worker's own conversion", but that function now only forwards to core.
- **Impact:** Low risk today, since the values are identical. It is clutter and a stale comment.
- **Suggested fix:** Import `fit_gate::BYTES_PER_GIB`, delete local copies and the forwarding wrapper, and fix the doc.
- **Confidence:** High

#### [F-119] Same 2.0 GB reserve declared by value in several modules instead of through `fit_gate`
- **Source ID:** W2-011
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `crates/sceneworks-worker/src/conditioning_fit.rs:76`, `crates/sceneworks-worker/src/iris_restore.rs:472-473`, `crates/sceneworks-core/src/training/iris/floors.rs:24,31`, `crates/sceneworks-worker/src/fit_gate.rs:38-48`, `crates/sceneworks-worker/src/krea_control_fit.rs:95`
- **Finding:** `fit_gate` made the reserve a typed policy so "the generic gate, control gate, Mochi, and the web mirror cannot assign independent meanings" (`vram_gate.rs:41-44`). `conditioning_fit::HEADROOM_GB = 2.0` says it "mirrors" vram_gate but is a bare literal. `iris_restore` and core `floors.rs` keep their own literal reserves.
- **Impact:** Tuning `DEDICATED_VRAM_ALLOCATOR_SLACK_GB` leaves these gates on the old value without any test failing.
- **Suggested fix:** Alias the typed constants, or add a test asserting equality.
- **Confidence:** High

#### [F-120] Doubled/stale doc comments and orphaned comment blocks
- **Source ID:** W2-014
- **Category:** readability
- **Severity:** Low
- **Location:** `crates/sceneworks-worker/src/mlx_fit_gate.rs:2630-2642` (the same doc paragraph twice, the first copy attached to nothing), `mlx_fit_gate.rs:64-80` (see W2-004), `vram_gate.rs:~2355-2362` ("Reserves HEADROOM_GB rather than the MLX gate's the MLX unified reserve"), `engines.rs:1-30` (still describes "mlx-gen registry" and "Python `MODEL_TARGETS`" parity for a backend-neutral table)
- **Finding:** Stale or garbled comments left after refactors.
- **Impact:** Reader confusion and a mildly misleading architecture description.
- **Suggested fix:** Delete or fix.
- **Confidence:** High

#### [F-121] Blanket `allow(dead_code)` on whole modules; production-dead items kept alive by tests
- **Source ID:** W2-015
- **Category:** dead-code
- **Severity:** Low
- **Location:** `crates/sceneworks-worker/src/lib.rs:231-234` (`mlx_fit_gate`, `mlx_phase_estimate` get `cfg_attr(not(macos), allow(dead_code))` for the entire 22k-line module); `yue2_admission.rs:1010,1020,1987,2126,2518` (`device_peak`, `host_peak`, `transcriber_bytes`, `transcription_shape`, `Yue2Lease::estimate` are `cfg_attr(not(test), allow(dead_code))`); `execution_planner.rs:951` (`allow(unused_variables)` on `prefix_kv_cache_surcharge_bytes`)
- **Finding:** The module-wide allow means no real dead item in `mlx_fit_gate` is ever reported on Candle/Linux builds. Several `yue2_admission` pub(crate) items have no production caller (the comment says the transcription job does not exist yet).
- **Impact:** Dead code accumulates silently. Speculative API is kept compiling only for tests.
- **Suggested fix:** Scope the allows per item, or split the cross-platform pieces out of `mlx_fit_gate`. Delete or `#[cfg(test)]` the unused yue2 items.
- **Confidence:** High

#### [F-122] God files: inline test modules of 5k-15k lines inside gate sources
- **Source ID:** W2-016
- **Category:** readability
- **Severity:** Low
- **Location:** `crates/sceneworks-worker/src/mlx_fit_gate.rs` (production ~6.8k lines, tests 6828-22560); `vram_gate.rs` (prod ~3k, tests 2987-8176); `candle_memory_strategy.rs` (prod ~3.2k, tests 3164-9791); `memory_route_registry.rs` (prod 2.6k incl. a 900-line `RULES` table, tests 3400-10486); `sceneworks-core/src/memory_anchor.rs` (prod 4.5k, tests 4487-9484)
- **Finding:** Test code is 1.5x-3x the size of the production code in the same file, and `mlx_fit_gate` mixes budget math, evidence assembly, ladder synthesis, Mochi/SVD gates and the Mage fingerprint table. `yue2_admission` already uses `#[cfg(test)] mod tests;` as a sibling file.
- **Impact:** Slow navigation and review, long incremental compiles, and helpers duplicated across test modules (`fixture_spec` at `mlx_fit_gate.rs:8956,11465`, `walk`, `reserve_for`, and the same-named test `fit_decision_rejects_only_a_genuine_overflow` in both gate files).
- **Suggested fix:** Move tests to `*/tests.rs` as yue2 does. Split the video/Mochi/SVD gates and the evidence builders out of `mlx_fit_gate` into submodules. Share test fixtures.
- **Confidence:** High

#### [F-123] Measured calibration constants embedded as literals in `mlx_phase_estimate`
- **Source ID:** W2-017
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `crates/sceneworks-worker/src/mlx_phase_estimate.rs:100-115` (e.g. `3_864_237_044 - 2_318_787_072`, `4_118_238_400 - 1_992_341_184`, `3_140_425_096`), plus the layerwise decode fit
- **Finding:** Coefficients derived from one captured run are inlined as arithmetic on byte counts, while the Candle lane keeps its retained corpus as packaged data (`memory_anchor.rs` `PACKAGED_MEMORY_ANCHOR_SOURCES`) with derivation tests.
- **Impact:** There is no machine check tying the numbers to their capture, so they cannot be audited or regenerated.
- **Suggested fix:** Move the per-family workspace coefficients into the packaged anchor sources and read them like the Candle lane does.
- **Confidence:** Medium

#### [F-124] `ensure_hf_cached_file` and `ensure_hf_cached_file_atomically` duplicate the snapshot-resolve block
- **Source ID:** W3-005
- **Category:** redundant
- **Severity:** Low
- **Location:** `downloads.rs:471-520` and `downloads.rs:523-565`
- **Finding:** The ~25 lines that resolve the snapshot, find the single file and raise "Hugging Face file {file} not found" are identical. Only the final call differs.
- **Impact:** Minor, but it is the third copy of the single-file resolve pattern (see also `huggingface_file_content_sha256` at `830`).
- **Suggested fix:** Extract `resolve_single_hf_file(context, repo, revision, file) -> SnapshotFile`.
- **Confidence:** High

#### [F-125] Full `Settings` struct literal copied into ~16 test modules, and `isolate_hf_cache` into 11
- **Source ID:** W3-014
- **Category:** redundant
- **Severity:** Low
- **Location:** `Settings { … external_model_roots: Vec::new() }` in `downloads.rs`, `model_jobs.rs`, `training_jobs.rs:4367`, `external_library_runtime.rs`, `person_segment.rs`, `caption_jobs.rs` and more (46 field-lines of `external_model_roots: Vec::new()`). `isolate_hf_cache` is defined in `downloads.rs:2777`, `model_jobs.rs:6921` and `7738` (twice in one file), `converted_install.rs:264`, `audio_jobs.rs:3579`, `depth_jobs/tests.rs:135`, `image_jobs/{tests,supra2,sdxl_control,iris}.rs` and `tests/lora_and_downloads.rs:1336`. A shared `test_env::offline_settings()` already exists (`test_env.rs:136`).
- **Finding:** Test support is copy-pasted instead of using the existing shared helper.
- **Impact:** Every new `Settings` field forces edits in ~16 files. The env var list in `isolate_hf_cache` can drift between copies.
- **Suggested fix:** Move `isolate_hf_cache` into `test_env.rs` and add `Settings::for_test(data_dir)` or extend `offline_settings(data_dir)`. Replace the literals.
- **Confidence:** High

#### [F-126] Iris GPU smoke files duplicate the entire API stub harness
- **Source ID:** W3-015
- **Category:** redundant
- **Severity:** Low
- **Location:** `iris_gpu_smoke.rs:99-179` vs `iris_training_gpu_smoke.rs:48-127`. In-file mock routes `job_route`, `progress_route` and `heartbeat_route` also exist in `iris_training.rs:1183-1204` and `training_jobs.rs:5101-5115`.
- **Finding:** `stub_job_snapshot`, `stub_job`, `stub_progress`, `stub_heartbeat`, `stub_metrics`, `spawn_worker_api_stub` and `shipped_iris_entry` are near-identical in both smoke files. The unit tests re-roll similar axum mocks again.
- **Impact:** Maintenance drift. A job-snapshot shape change has to be applied in four places.
- **Suggested fix:** Move the stub into `tests/support_stubs.rs` (which exists) and parameterise it on job type and project.
- **Confidence:** High

#### [F-127] Research-instrument modules (`lora_eval_harness`, `lora_train_driver`) live in `src/`
- **Source ID:** W3-016
- **Category:** dead-code
- **Severity:** Low
- **Location:** `lib.rs:480-486`; `lora_eval_harness.rs` (1008 lines), `lora_train_driver.rs` (566 lines)
- **Finding:** Both are `#[cfg(all(test, target_os = "macos"))]` and consist of `#[ignore]` tests that depend on a personal dataset layout (`~/Datasets/...`) and a MLX-only Z-Image study (sc-6541).
- **Impact:** About 1.5k lines of study tooling on the review and compile path of every macOS test build. The `src/` location suggests product code.
- **Suggested fix:** Move them to a dev-only `examples/` or `tests/` crate, or archive them with the study docs.
- **Confidence:** High

#### [F-128] Catalog analysis lock opens lock files without the symlink check that fetch and core use
- **Source ID:** W4-007
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `catalog_analysis.rs:481-510` vs `catalog_image_fetch.rs:618-651` and `sceneworks-core/src/catalog_store.rs:660`
- **Finding:** `CatalogFetchLease` and the core store `symlink_metadata`-check the lock path (regular file only) before `OpenOptions::create(true)`; `CatalogAnalysisLease` creates/opens both `FETCH_LOCK_FILE` and `ANALYSIS_LOCK_FILE` through any symlink. The open+`FileLock::try_exclusive`+EAGAIN->Busy sequence is also copy-pasted a third time in `yue2_jobs.rs:1279-1296` (`RunClaim`).
- **Impact:** A symlinked lock file in a catalog root makes analysis create/truncate-open an arbitrary file path; inconsistent hardening.
- **Suggested fix:** Add `FileLock::open_exclusive_regular(path) -> io::Result<..>` in `sceneworks-core::file_lock` doing the symlink check + contention mapping; use it in all four places.
- **Confidence:** Medium

#### [F-129] Small text/cancel helpers re-implemented with divergent behaviour
- **Source ID:** W4-011
- **Category:** redundant
- **Severity:** Low
- **Location:** `catalog_semantic_jobs.rs:230-246` vs `prompt_refine_jobs.rs:1075-1086` (`clean_json_output`); `sensenova_jobs.rs:1820-1840` vs `prompt_refine_jobs.rs:633-647` (think-block stripping); `catalog_analysis.rs:1416` / `catalog_semantic_jobs.rs:382` (`bounded_error`); `catalog_analysis.rs:702`, `catalog_semantic_jobs.rs:414`, `person_segment_sam3_common.rs:39` (`check_*_cancel`); `vector_jobs.rs:64`, `yue2_jobs.rs:741`, `catalog_image_fetch.rs:2232`, `qwen_prompt_rewrite.rs:269` (sha256 hex)
- **Finding:** The semantic `clean_json_output` does not strip `<think>` blocks or orphan `</think>` (the prompt-refine one does), though it feeds a VLM that may emit them; sensenova's `strip_reasoning` is case-sensitive and ignores orphan closers, unlike `strip_think_blocks`. Four sha256-to-hex helpers use two different formatting strategies.
- **Impact:** A reasoning-model reply that parses in prompt-refine fails in catalog semantic analysis; small fixes land in one copy.
- **Suggested fix:** Reuse `prompt_refine_jobs::clean_json_output`/`strip_think_blocks` (hoist to a `text_clean` module) and add `sceneworks_core::hash::sha256_hex`.
- **Confidence:** Medium

#### [F-130] `is_caption_json` duplicates `sceneworks_core::ideogram_caption::is_caption`
- **Source ID:** W4-012
- **Category:** redundant
- **Severity:** Low
- **Location:** `ideogram_caption.rs:61-70` vs `sceneworks-core/src/ideogram_caption.rs:39-44`
- **Finding:** Same `compositional_deconstruction`-is-object rule re-coded on a raw string; prompt-refine already imports the core one at `prompt_refine_jobs.rs:1714`.
- **Impact:** Two definitions of "is a caption"; the file header says the web/core rules are "not re-implemented here" yet this is a re-implementation.
- **Suggested fix:** Parse once and call `sceneworks_core::ideogram_caption::is_caption`.
- **Confidence:** High

#### [F-131] Dead/over-suppressed pipeline entry points in catalog modules
- **Source ID:** W4-013
- **Category:** dead-code
- **Severity:** Low
- **Location:** `catalog_analysis.rs:533-553` (`analyze_attached_catalog_under_processing_lease`), `catalog_analysis.rs:473`, `catalog_image_fetch.rs:610` (`#[allow(dead_code)]` on a used fn)
- **Finding:** `analyze_attached_catalog_under_processing_lease` has no callers anywhere in the workspace (only the `_with_control` sibling is used). The `#[allow(dead_code)]` on both `acquire_under_processing_lease` fns hides nothing, they are called at `catalog_analysis.rs:591` and `catalog_image_fetch.rs:726`.
- **Impact:** Misleading API surface; the allows mask future real dead code.
- **Suggested fix:** Delete the unused function; drop both allows.
- **Confidence:** High

#### [F-132] Local HTTP stub routes duplicated across test modules
- **Source ID:** W4-015
- **Category:** redundant
- **Severity:** Low
- **Location:** `audio_jobs.rs:3523-3545,5505-5516`, `caption_jobs.rs:1239-1250`, 45 `progress_route`/`heartbeat_route` definitions across 11 files vs shared `tests/support_stubs.rs:303-330`; `test_settings` in `caption_jobs.rs` and `dataset_analysis_jobs.rs`
- **Finding:** Unit-test modules in-file each spin their own axum stub with the same progress/heartbeat/job handlers rather than the shared stub support.
- **Impact:** Contract changes to these endpoints need many test edits.
- **Suggested fix:** Move to a `#[cfg(test)] pub(crate) mod test_api_stub` reused by in-file tests.
- **Confidence:** Medium

#### [F-133] Same "total minus unified-memory reserve" and "cap-then-budget" composition re-derived outside the fit gates
- **Source ID:** W5-010
- **Category:** redundant
- **Severity:** Low
- **Location:** `depth_jobs.rs:595-614`, `iris_restore.rs:545-565`, `iris_training.rs:381-400`, `yue_admission.rs:~950`, `generator_cache.rs:656`
- **Finding:** `resolve_budget(probe, mlx_memory_cap_gb()) - legacy_unified_reserve(total)` and `apply_vram_cap(nvidia_vram_budget_gb(..), cuda_vram_cap_gb()) - HEADROOM_GB` are rebuilt per job family. `apply_vram_cap(` is called from ~30 places outside `vram_gate.rs` (mostly other reviewers' files).
- **Impact:** Each lane's definition of "available memory" can drift between admission checks. Depth and Yue already compute the MLX figure in separate places.
- **Suggested fix:** Expose `fit_gate::lane_available_gb(settings)` and `lane_total_gb(settings)` and use them in depth, iris_restore and iris_training.
- **Confidence:** Medium

#### [F-134] depth_jobs can leave published files orphaned, and `depth_asset_fact` returns a `Result` it never fails
- **Source ID:** W5-011
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `depth_jobs.rs:~283-292` (`outputs.publish()` then `depth_asset_fact(...)?`), `depth_jobs.rs:719-748`
- **Finding:** After `publish()` renames the `.npy` and preview into place, the `?` on `depth_asset_fact` returns without `outputs.discard()`. Every later failure path does call `discard()`. `depth_asset_fact` is infallible today (always `Ok`), so the leak is latent, and the `WorkerResult` return type invites future fallible code to hit it. `DepthOutputPaths::publish` and `discard` also use blocking `std::fs` calls from an async fn (cheap renames, low risk).
- **Impact:** A future fallible change leaves unreferenced multi-MB depth files in the project.
- **Suggested fix:** Make `depth_asset_fact` return `Value`, or wrap the post-publish section so every error path discards.
- **Confidence:** High

#### [F-135] `optional_payload_string` filters blank strings but returns them untrimmed, so 227 call sites re-trim by hand
- **Source ID:** W5-012
- **Category:** readability
- **Severity:** Low
- **Location:** `payload.rs:34-48`; `.map(str::trim)` x227 in non-test code (e.g. `face_analysis_jobs.rs`, `kps_jobs.rs:318-345`, `face_likeness_compare_jobs.rs:90-95`, `subject_mask_jobs.rs`, `upscale_jobs.rs:1680`)
- **Finding:** The helper treats " " as absent but returns " abc " verbatim, so callers that want a trimmed value re-implement the `get/as_str/trim/filter` chain. `required_payload_string` is used 45 times in `model_jobs.rs` but almost nowhere in the CV modules. This is how the item-parsing copies in W5-004 arose.
- **Impact:** Inconsistent handling of whitespace in IDs and paths between job types.
- **Suggested fix:** Make the helper return the trimmed slice, and add `payload_trimmed(&Value, key)` for the `&Value` case.
- **Confidence:** High

#### [F-136] Stale, misplaced and dead items in lib.rs / canny.rs
- **Source ID:** W5-013
- **Category:** dead-code
- **Severity:** Low
- **Location:** `lib.rs:60-68` (comment about `advanced` sits above the unrelated `admission_geometry` re-export, so the doc comment is attached to the wrong item); `canny.rs:71-74` (`canny_control_image_default`, only used by its own tests and doc links); `person_jobs.rs:393-405` (five `YoloForward` block fields that exist "only for the parity test")
- **Finding:** Cosmetic drift. The misplaced comment will be read as describing `admission_geometry`. The `YoloForward` fields cost memory and attribute noise in production builds.
- **Impact:** Minor confusion, no behaviour change.
- **Suggested fix:** Move the comment down to `mod advanced;`. Mark `canny_control_image_default` `#[cfg(test)]` or delete it. Gate the oracle-only fields behind `#[cfg(test)]`.
- **Confidence:** High

#### [F-137] Giant functions: `run_utility_job` (~430 lines) and `run_worker_loop` (~265 lines)
- **Source ID:** W5-014
- **Category:** readability
- **Severity:** Low
- **Location:** `lib.rs:2247-2676`, `lib.rs:1693-1957`
- **Finding:** `run_utility_job` mixes the pre-loader source guard, a 40-arm dispatch (each arm carrying a multi-line historical comment and a repeated `.map_err(|e| ("X failed.", e))`), the MLX-poison terminal-failure retry loop, metrics posting and the idle heartbeat. The poison-retry closure is nested 6 levels deep. `run_worker_loop` inlines the startup, maintenance scheduling, poison quarantine and four error arms.
- **Impact:** This is the most safety-critical function in the worker (it decides job terminal state), and it is hard to review or unit-test. `architecture_tests` has to regex into it by text because it cannot call it in isolation.
- **Suggested fix:** Extract `dispatch(job_type) -> Result<(), (&'static str, WorkerError)>`, `handle_job_failure(...)` (including the poison path) and `maintenance_tick(...)`. A table of `(JobType, label)` would also remove the repeated message strings.
- **Confidence:** High

#### [F-138] A full YOLO11m network is hand-implemented in the worker, against the "engines live in the pinned inference repo" boundary
- **Source ID:** W5-015
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `person_jobs.rs:1-60` (rationale), `:380-740` (MLX Conv/C3k2/SPPF/C2PSA/Detect forward), `:820-930` (ort twin)
- **Finding:** SAM, SCRFD and ArcFace are consumed from the engine repo, but the person detector forward pass, the DWPose YOLOX/RTMW pre/post-processing (`pose_jobs.rs:238-570`), and the Real-ESRGAN tiling/session (`upscale_jobs.rs:110-480`) live in the app crate, each with an MLX variant and an ORT variant that have already diverged (the MLX path is NHWC, the ORT path NCHW, with different caching).
- **Impact:** Model-math changes need app releases rather than engine pin bumps, and the ort cache/EP code (W5-002) keeps being re-copied.
- **Suggested fix:** Longer term, move the detector forward and DWPose decode behind an engine `Detector` trait. At minimum, keep the shared decode/NMS in one `cv_common` module and put the ort glue in one place.
- **Confidence:** Medium

#### [F-139] Settings parsing silently swallows invalid numeric env values
- **Source ID:** W5-016
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `settings.rs:232-238` (`env_u64_any`), `settings.rs:241-251` (`env_bool`)
- **Finding:** An unparsable `SCENEWORKS_POLL_SECONDS=abc` or `SCENEWORKS_MAX_MODEL_URL_BYTES=256G` silently becomes the default. `ResolvedCachePolicy` goes the other way and fails closed. `env_bool` has two identical arms (`""` and `_` both return `default`).
- **Impact:** Operator typos on size limits (`max_*_url_bytes`) quietly widen to a 256 GiB cap.
- **Suggested fix:** Log a warning (`event=invalid_env`) on a parse failure, merge the redundant arm, and treat a bad limit as an error at startup.
- **Confidence:** High

#### [F-140] CatalogProcessingLease reimplements FileLock, and the flock open and contention boilerplate is copied across crates
- **Source ID:** C1-009
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/sceneworks-core/src/catalog_store.rs:625-700`; `file_lock.rs:1-6` (acknowledges the twin); `sceneworks-worker/src/downloads.rs:122-160`; `sceneworks-worker/src/catalog_analysis.rs:485-505`; `sceneworks-worker/src/catalog_image_fetch.rs:630-645`; `sceneworks-worker/src/model_jobs.rs:4850-4860`; `resolved_cache.rs:2554, 2575`
- **Finding:** `file_lock.rs` exists so every holder releases with `LOCK_UN`, yet the catalog lease keeps its own `Drop` and its own `inherited_descriptor`. Each site also repeats the open (`create/read/write/truncate(false)`, 23 occurrences), the `fs2::lock_contended_error().raw_os_error()` comparison and a spin-wait loop, instead of calling one shared `try_lock_path`.
- **Impact:** The invariant that took sc-22738 to find is duplicated; the next holder may forget it.
- **Suggested fix:** Add `FileLock::open_exclusive(path)` and `FileLock::is_contended(&io::Error)`, make CatalogProcessingLease wrap `FileLock`, and drop the dead `FileLock::try_shared`. Also remove the redundant `fs2` entry from `[dev-dependencies]` (`Cargo.toml:91` duplicates `:15`; its justifying comment is stale).
- **Confidence:** High

#### [F-141] Film reference and sound staging blocks in project_store are near-duplicates
- **Source ID:** C1-012
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/sceneworks-core/src/project_store.rs:727-755` vs `758-792`
- **Finding:** Both loops do the same unsafe-path check, canonicalize, `starts_with(canonical_assets)`, `is_file`, create parent and copy. They differ only in labels and the optional-source skip for generated dialogue.
- **Impact:** A confinement fix needs two edits in a security-sensitive block.
- **Suggested fix:** Extract `stage_draft_file(draft_assets, canonical_assets, dest_root, file, label)`.
- **Confidence:** High

#### [F-142] `required_str` and related JSON-field helpers are defined three times with different messages
- **Source ID:** C1-013
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/sceneworks-core/src/asset_index.rs:607-610`; `project_store.rs:7838-7841`; `character_store.rs:1467-1474`
- **Finding:** Two copies are identical ("Missing required field: {key}") and the character copy produces a different user-facing message ("{Title} is required"). `store_util` already hosts the sibling `optional_*` helpers.
- **Impact:** Inconsistent API error text for the same condition, and drift.
- **Suggested fix:** Put one `required_str` in `store_util` and keep the character message override as a thin wrapper if the text must differ.
- **Confidence:** High

#### [F-143] SQLite pragma policy is scattered, and the 5000 ms busy timeout is a bare literal in three places
- **Source ID:** C1-014
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `crates/sceneworks-core/src/project_store.rs:6379`; `jobs_store.rs:3481`; `catalog_store.rs:3550` (and `4596`, `training_store.rs:3696` pin the value in tests)
- **Finding:** Each opener hand-codes its own settings: project.db has only `busy_timeout` (rollback journal), jobs.db has WAL plus `foreign_keys` and default `synchronous`, and the catalog has WAL plus `synchronous=normal`, `temp_store`, `cache_size` and `wal_autocheckpoint`. Doc comments assert the 5s value is "shared", but there is no shared constant. jobs.db keeps `synchronous=FULL` under WAL, which makes every heartbeat and progress write an fsync.
- **Impact:** Tuning concurrency or durability means finding three sites. Nothing records why project.db and jobs.db differ.
- **Suggested fix:** Add `store_util::SQLITE_BUSY_TIMEOUT` and a `configure_connection(conn, Profile)` helper, then document the intended per-DB policy in one place. Consider `synchronous=NORMAL` for jobs.db under WAL.
- **Confidence:** Medium

#### [F-144] credentials.rs write path duplicates store_util's atomic write and its `sync_parent_dir`
- **Source ID:** C1-015
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/sceneworks-core/src/credentials.rs:292-316, 330-357`; `store_util.rs:37-72`
- **Finding:** `CredentialFileStore::save` repeats the tmp-name, `sync_all`, rename, `sync_parent_dir` sequence and carries a second copy of `sync_parent_dir`; the only real difference is the `0600` mode. (Counted in C1-002 but called out separately because it is the secrets path.)
- **Impact:** Any fix to atomic write (for example orphan tmp cleanup on error, which neither does) must be made twice. The tmp file is also left behind if the rename fails.
- **Suggested fix:** Give the shared writer a `mode` option and remove the credentials copy; clean up the tmp on rename failure.
- **Confidence:** High

#### [F-145] Streaming file SHA-256 loop copy-pasted across core
- **Source ID:** C2-004
- **Category:** redundant
- **Severity:** Low
- **Location:** `checkpoint_ingest.rs:654-666`, `checkpoint_plan_store.rs:1193-1206`, `managed_file.rs:78-90`, `model_artifacts.rs:1229-1244` (also `download_receipt.rs:168`, `resolved_cache.rs:2646,3442`, `training_store.rs:1500`)
- **Finding:** `sha256_file` is byte-identical in checkpoint_ingest.rs and checkpoint_plan_store.rs (only the buffer constant name differs); managed_file and model_artifacts inline the same read/update loop with 1 MiB and 64 KiB buffers respectively. Worker has its own async variant too.
- **Impact:** Four buffer sizes, no shared place to add cancellation or progress; drift is already visible.
- **Suggested fix:** One `store_util::sha256_file(path) -> io::Result<String>` (plus a `_prefixed` variant for `sha256:` callers).
- **Confidence:** High

#### [F-146] Fixture helpers duplicated across integration test files; no tests/common
- **Source ID:** C2-011
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/sceneworks-core/tests/`: `write_safetensors` in `checkpoint_ingest.rs:56`, `checkpoint_inspector.rs:28`, `checkpoint_plan_store.rs:80`; `write_krea_native_file` in `checkpoint_derivative.rs:36`, `checkpoint_ingest.rs:44`, `checkpoint_plan_store.rs:68`; `tree_snapshot` in `checkpoint_derivative.rs:126`, `checkpoint_plan_store.rs:1381`; `repo_root` in `workflow_png.rs:33`, `workflow_share.rs:27`, `workflow_share_doc.rs:58`; `fixture_path` in `character_store.rs:13`, `contract_roundtrip.rs:14`, `training_contracts.rs:23`
- **Finding:** There is no `tests/common/mod.rs`; each of the 16 integration files (37k lines; `jobs_store.rs` 9.6k, `workflow_share.rs` 6.1k, `training_contracts.rs` 4.5k) re-declares its fixture writers. `workflow_share.rs` also source-greps worker files (`worker_sources()`) to enforce architecture.
- **Impact:** Fixture drift between the three checkpoint test files; each is its own compile unit/binary which slows link time.
- **Suggested fix:** Add `tests/common/mod.rs` with safetensors/GGUF/tree-snapshot/repo-root helpers; consider merging small integration files into one binary.
- **Confidence:** High

#### [F-147] ffmpeg program resolution and error handling duplicated between core and worker
- **Source ID:** C2-012
- **Category:** redundant
- **Severity:** Low
- **Location:** `media_convert.rs:648-654` (`ffmpeg_program`), `:824,917` (test-time `env::var`), `crates/sceneworks-worker/src/media_jobs.rs:782-790,4083-4090`
- **Finding:** Three copies of the `SCENEWORKS_FFMPEG` lookup with a trim/non-empty filter in only some. The comment at `media_convert.rs:646` states it "mirrors" the worker's resolution rather than sharing it, and only the worker copy has a timeout.
- **Impact:** Divergent behavior (blank env var, timeout) already exists.
- **Suggested fix:** Move resolution + a timeout-bounded runner to core and use from both.
- **Confidence:** High

#### [F-148] HDR/EXR preview decodes the full float frame into memory with no dimension cap
- **Source ID:** C2-013
- **Category:** efficiency
- **Severity:** Low
- **Location:** `crates/sceneworks-core/src/media_convert.rs:301-336`
- **Finding:** `transcode_hdr_to_png` takes dimensions from the file header, runs `ffmpeg -f rawvideo -pix_fmt gbrpf32le -` and buffers `w*h*12` bytes in `output.stdout`, then allocates another `w*h*3` Vec. `w*h*3*4` is computed in `usize` with no `checked_mul` or pixel cap.
- **Impact:** A large (or lying-header) EXR can allocate multiple GB inside the thumbnail route.
- **Suggested fix:** Cap pixels (e.g. reject above N MP), use `checked_mul`, and stream the planes.
- **Confidence:** Medium

#### [F-149] Request-side validation asymmetries: image requests clamp silently, video requests reject, and upscale factor is validated in three places
- **Source ID:** C2-014
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `image_request.rs:236-300` (`count`/`width`/`height` clamped by `clamped_u32`), `contracts.rs:757-766` (`factor: u8`, no validation), `apps/rust-api/src/lib.rs:5311`, `crates/sceneworks-worker/src/image_jobs.rs:3173` (`if factor == 4 {4} else {2}`), `video_request.rs:1003-1010` (`requested_steps` `as u32`)
- **Finding:** `ImageRequest::from_payload` is infallible and clamps count to 1..=8 and dimensions to 256..=4096, so out-of-range input reaches the engine as a different value than requested (the image dimension doc itself says silent coercion is the failure class). Upscale `factor` has no validator in core; the API checks `2|4` and the worker coerces anything else to 2. `requested_steps` re-implements numeric parsing instead of `payload_util::parse_u32` and truncates with `as u32` (`4294967300` becomes `4` and passes the step gate).
- **Impact:** Silent request rewriting on image lane; validation rules scattered and, for steps, truncating.
- **Suggested fix:** Add a core `image_request_error(&ImageRequest)` mirroring video's gates (count, dims, upscale factor) called by both API and worker; use `u32::try_from` in `requested_steps`.
- **Confidence:** Medium

#### [F-150] Slug helpers duplicated with different separators
- **Source ID:** C2-015
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/sceneworks-core/src/slug.rs:1-27`, `apps/rust-api/src/lib.rs:4562-4580` (`slugify_lora_id`, `_`-separated), `apps/rust-api/src/recipe_presets.rs:1075` (wraps it), `crates/sceneworks-worker/src/gpu.rs:1217`
- **Finding:** `slugify_lora_id` is the core `slugify` loop with `'_'` for `'-'` and a fixed fallback.
- **Impact:** Two places to fix the trailing-separator class of bug (sc-8951 was fixed only in core's copy).
- **Suggested fix:** `slugify_with(value, fallback, max, sep)` in core.
- **Confidence:** High

#### [F-151] builtin_training_targets has a stale comment/loop structure and a runtime panic
- **Source ID:** C2-016
- **Category:** readability
- **Severity:** Low
- **Location:** `crates/sceneworks-core/src/training.rs:932-973`
- **Finding:** Two consecutive `for target in &mut registry.targets` loops; the comment above the second says "Epic 2123 resolution buckets" but the loop begins with face-loss flags, and bucket stamping is mid-body. The loop calls `.unwrap_or_else(|e| panic!(...))` on `expected_backend_local_trainer_id`, so a routing-table gap panics the API/worker at registry build time rather than failing a test only.
- **Impact:** Misleading comments; a catalog edit can crash startup.
- **Suggested fix:** Merge the loops, reorder comments to match, return `Result` or guard with a unit test (a test already exists per comment; make the function infallible by construction).
- **Confidence:** Medium

#### [F-152] Ten handlers use bare `axum::Json` instead of `ApiJson`, bypassing the API's error contract
- **Source ID:** A1-010
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `apps/rust-api/src/checkpoint_library.rs:271,294,387`, `apps/rust-api/src/model_cache.rs:353,375,397`, `apps/rust-api/src/model_library.rs:103`, `apps/rust-api/src/iris_curricula.rs:169` (compare `apps/rust-api/src/lib.rs:519-535`, `json_rejection_response` at `:599`)
- **Finding:** `ApiJson` exists specifically to turn decode failures into the `{detail}` 422 shape and to preserve 413. These handlers take `Json<T>` and get axum's default plain-text rejection. The 2026-07-12 review noted the same for `assets.rs:255-259` as part of a roll-up. This is a further instance in the model/library routes.
- **Impact:** The web client's error handling sees a non-JSON body on a malformed request to the model-library and cache routes.
- **Suggested fix:** Swap to `ApiJson`. Consider a clippy-style grep test that forbids `Json(` extractors in `apps/rust-api/src`.
- **Confidence:** High

#### [F-153] Blocking std::fs and CPU work on the async runtime in several handlers
- **Source ID:** A1-011
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `apps/rust-api/src/training.rs:863-889` (up to `MAX_DATASET_PARQUET_IMPORT_ITEMS` = 100,000 `std::fs::canonicalize` calls in a loop inside an async handler), `:142,168,1389` (`std::fs::remove_file`), `apps/rust-api/src/iris_task_training.rs:91,148` (`std::fs::rename`/`remove_file`), `apps/rust-api/src/models.rs:3023-3040` (`prune_empty_repo_cache` calls `std::fs::read_dir`, `huggingface_snapshot_dirs`, `snapshot_files` in an async fn), `apps/rust-api/src/credentials.rs:14-79` (`CredentialFileStore` list/set/delete take a process lock and do sync I/O inline), `apps/rust-api/src/iris_training.rs:326,534` (`is_file`, `newest_complete_checkpoint`)
- **Finding:** `preferences.rs:274-345` already does this correctly (sc-4202, `spawn_blocking`). These sites do not.
- **Impact:** A large Parquet finalize stalls a Tokio worker thread for the whole walk. Slow or network-mounted disks stall unrelated requests.
- **Suggested fix:** Wrap the finalize item loop and `prune_empty_repo_cache` in `spawn_blocking`. Use `tokio::fs` for the single-file removals. Route credentials through a `blocking()` helper like `checkpoint_library.rs` does.
- **Confidence:** High

#### [F-154] Dead feature flag and a `const _` hack keeping an unused import alive
- **Source ID:** A1-013
- **Category:** dead-code
- **Severity:** Low
- **Location:** `apps/rust-api/src/models.rs:3142-3149,3151-3152,3189-3194` (`model_import_enabled()` always returns `true`; `MODEL_IMPORT_DISABLED_DETAIL` says "temporarily disabled while native model support is being built"), `apps/rust-api/src/models.rs:3-5` (`const _: fn(&FsPath) -> PathBuf = huggingface_hub_cache_dir;`), `apps/rust-api/src/lib.rs:31`
- **Finding:** The kill switch is unreachable, and its user-facing text contradicts the shipped behaviour. `huggingface_hub_cache_dir` is imported in `lib.rs` and used only by tests in models.rs. The `const _` item exists only to silence the unused-import warning.
- **Impact:** Misleading code and docs. A reader may think imports can be disabled.
- **Suggested fix:** Delete `model_import_enabled` and the constant, with the guarded branch. Move the import to `#[cfg(test)]` in the test modules and delete the `const _`.
- **Confidence:** High

#### [F-155] prompts.rs comment contradicts the dependency graph, and its pinning test pins nothing
- **Source ID:** A1-014
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `apps/rust-api/src/prompts.rs:9-14,305-307`; `crates/sceneworks-worker/src/qwen_prompt_rewrite.rs:68`
- **Finding:** The comment says the constant is duplicated "because the two crates share no dependency in this direction". `apps/rust-api/Cargo.toml:31` depends on `sceneworks-worker`, and other rust-api modules call it (`iris_*`, `saved_voices.rs`). The test `the_rewrite_task_name_matches_the_worker` asserts the constant against a string literal, not against the worker's `REWRITE_TASK`.
- **Impact:** The drift guard cannot fire if the worker's value changes.
- **Suggested fix:** Re-export `REWRITE_TASK` from the worker (or move it to core) and use it directly. If kept separate, assert `QWEN_IMAGE_REWRITE_TASK == sceneworks_worker::REWRITE_TASK`.
- **Confidence:** High

#### [F-156] Small twins: `require_local_peer`, licence predicate, and slugifier
- **Source ID:** A1-015
- **Category:** redundant
- **Severity:** Low
- **Location:** `apps/rust-api/src/model_library.rs:179-192` vs `apps/rust-api/src/checkpoint_library.rs:438-451`; `apps/rust-api/src/models.rs:514-519` vs `apps/rust-api/src/yue2_jobs.rs:179-184`; `apps/rust-api/src/lib.rs:4562-4580` (`slugify_lora_id`) and `apps/rust-api/src/recipe_presets.rs:1075` vs `crates/sceneworks-core/src/slug.rs:1-27`
- **Finding:** The two `require_local_peer` functions differ only in message and error code. The licence-required check is the same JSON key read twice. `slugify_lora_id` is `core::slug::slugify` with `_` as separator and without the truncation fix from F-149, so it is exposed to the trailing-separator class of bug (although there is no `max_length` here).
- **Impact:** Minor drift risk. The peer check is also the one A1-001 wants to extend, so it should be single-sourced first.
- **Suggested fix:** A shared `local_only` guard in `error.rs` or a new `guards.rs`. Call `core::slug::slugify` with a separator parameter. Reuse `model_requires_license_acknowledgment` from `yue2_jobs`.
- **Confidence:** High

#### [F-157] Style and embedded-manifest lookups re-read and re-parse large JSONC per call, bypassing the cached loader
- **Source ID:** A1-016
- **Category:** efficiency
- **Severity:** Low
- **Location:** `apps/rust-api/src/styles.rs:16-41` (219 KB `builtin.styles.jsonc` read and parsed per `styleId` job and per `GET /styles`), `apps/rust-api/src/models.rs:10648-10667` (`embedded_builtin_catalog_entry` strips comments and parses the ~2 MB embedded manifest on every call; reached from `models.rs:10613` and `model_sources.rs:359` when a lookup misses)
- **Finding:** `manifest.rs:265-300` (`load_manifest_entries`) caches parsed manifests keyed on (path, mtime, size). `styles_catalog` bypasses it. The embedded fallback has no `OnceLock`.
- **Impact:** A 2 MB parse and comment-strip on every unknown-model lookup. This is cheap to avoid and sits in job admission.
- **Suggested fix:** Parse the embedded manifest once into `static OnceLock<Value>`. Route styles through `load_manifest_root` plus the cache.
- **Confidence:** Medium

#### [F-158] Staged multipart upload writer and parse loops are duplicated across models, LoRAs, poses, keypoints and training
- **Source ID:** A1-017
- **Category:** redundant
- **Severity:** Low
- **Location:** `apps/rust-api/src/models.rs:4044-4082` vs `apps/rust-api/src/loras.rs:1469-1495` (`write_*_upload_field_to_staged_file`, `cleanup_staged_*_upload`), `apps/rust-api/src/models.rs:3944-4042` vs `apps/rust-api/src/loras.rs:~1340-1450` (multipart parse loops), `apps/rust-api/src/training.rs:119-146,150-171` vs `apps/rust-api/src/poses.rs:44-69` vs `apps/rust-api/src/keypoints.rs:25-50`
- **Finding:** The two staging writers are identical except for the subdir name and the size-limit message. The pose and keypoint multi-file handlers are identical except for the cache-dir constant. This was previously noted in the 2026-07-12 roll-up (`loras.rs:952-1093 vs models.rs:732-844`) and is still present.
- **Impact:** Duplicate cleanup and size-limit behaviour must be kept in sync by hand.
- **Suggested fix:** One `stage_upload(state, field, subdir, filename, limit)` and one `cleanup_staged_upload(path)` in `assets.rs` next to `write_upload_field_to_dir`.
- **Confidence:** High

#### [F-159] "Raster image asset" predicate triplicated with drifted logic
- **Source ID:** A2-008
- **Category:** redundant
- **Severity:** Low
- **Location:** `generation.rs:763`, `generation.rs:826`, `depth_jobs.rs:81-83`
- **Finding:** generation.rs requires `type == "image" && mime starts_with image/ && mime != svg`; depth_jobs.rs accepts `(type == "image" || mime starts_with image/) && mime != svg && type != "vector"`. Same intent, different truth tables (an asset with a missing `type` but an image mime passes depth and fails generation).
- **Impact:** Inconsistent acceptance across sibling job routes; a new route will copy one of three variants.
- **Suggested fix:** One `fn is_raster_image_asset(&Value) -> bool` (ideally in core next to the asset type constants).
- **Confidence:** High

#### [F-160] Blocking filesystem calls inside async handlers and futures
- **Source ID:** A2-009
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `film_planning.rs:254,258,304,397,443,562`; `openai_planner.rs:315-345` (`multimodal_content` does `std::fs::metadata`+`std::fs::read` of up to 8 images / 20 MiB inside `complete()`); `generation.rs:768,826` (`media_path.is_file()`); `lib.rs:4125` (`resolved.exists()`); `assets.rs` (`std::fs::remove_file` in `inspect_err`)
- **Finding:** Short blocking calls run on tokio workers. The cancel sentinel check (`cancel_path.exists()`, film_planning.rs:443) is polled every 25 ms from the planner select loop (openai_planner.rs:188-214).
- **Impact:** Bounded but real stalls under a slow disk or network drive; the codebase otherwise uses `project_call`/`spawn_blocking` consistently.
- **Suggested fix:** Use `tokio::fs`/`spawn_blocking` for the reads; do the `is_file` checks inside the existing `project_call` closure.
- **Confidence:** High

#### [F-161] Shared `reqwest::Client::new()` has no default timeout, redirect policy or body cap
- **Source ID:** A2-011
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `lib.rs:1437`, `server.rs:426`; consumers `openai_planner.rs:144,281` (per-request timeout set), `film_planner_connections.rs:162`, `models.rs:11564` (wrapped in `timeout(8s)`)
- **Finding:** Every current caller remembers to add its own timeout, but the client itself would hang forever for the next one. It also follows up to 10 redirects, so a user-supplied planner base URL that passed `validate_base_url` (film_planner_connections.rs:419-460) can redirect to an internal or plain-http target, and `response.json()` buffers an unbounded body from that endpoint (openai_planner.rs:207, 291).
- **Impact:** Footgun plus a small SSRF/redirect gap on the user-configured planner endpoint; no current hang found.
- **Suggested fix:** Build the client with `.timeout(..)`, `.connect_timeout(..)` and `redirect(Policy::none())` (planner) and cap response size.
- **Confidence:** Medium

#### [F-162] `pid_alive` has three implementations, one already drifted
- **Source ID:** A2-015
- **Category:** redundant
- **Severity:** Low
- **Location:** `lib.rs:829-856`, `apps/desktop/src/setup.rs:2650-2652`
- **Finding:** The API's unix version treats `EPERM` as alive; the desktop's `kill(..).is_ok()` treats it as dead. `now_rfc3339` is likewise a one-line wrapper duplicated at `lib.rs:4584` and `worker/src/util.rs:30` over `core::time::utc_now` (also `film_harness.rs:825` `sha256_hex` vs four other `sha256_hex`/`hex_digest` helpers in worker/core).
- **Impact:** Behaviour differs per process for the same question.
- **Suggested fix:** Move liveness probing and the tiny hash/time wrappers into `sceneworks-core`.
- **Confidence:** High

#### [F-163] Test-support: project-creation boilerplate repeated ~190 times, plus three local copies of `project()`/`submit()`
- **Source ID:** A2-016
- **Category:** redundant
- **Severity:** Low
- **Location:** `tests/yue2_jobs.rs:22-48`, `tests/iris_training.rs:123-177`, `tests/iris_task_training.rs:127-222`; ~189 inline `POST /api/v1/projects` in `tests/jobs.rs` (74), `training.rs` (43), `moss_image.rs` (16), `films.rs` (15), `media.rs` (13); `tests/support.rs` has no project helper
- **Finding:** `support.rs` (1,341 lines) has request/claim/seed helpers but not `create_project`; each file reinvents it with different assertions.
- **Impact:** Maintenance noise when the projects contract changes.
- **Suggested fix:** Add `support::create_project(app, name) -> String` and replace.
- **Confidence:** High

#### [F-164] Test seams embedded in production structs/functions
- **Source ID:** A2-017
- **Category:** readability
- **Severity:** Low
- **Location:** `server.rs` (22 `#[cfg(test)]` items, AppState fields at lib.rs:1470-1488), `lib.rs` (47), `catalogs.rs` (44), `models.rs` (35), `catalog_scan_supervisor.rs` (13)
- **Finding:** AppState carries a dozen `#[cfg(test)]` hook fields (`progress_before_accept_once`, `sse_snapshot_before_subscribe_once`, `catalog_scan_injected_sqlite_busy_failures`...) and thread-locals live in lib.rs next to production consts.
- **Impact:** Production struct shape differs between test and release builds; hard to see what is real state.
- **Suggested fix:** Collect hooks in one `TestHooks` struct behind a single `#[cfg(test)]` field.
- **Confidence:** Medium

#### [F-165] Giant single-module film harness inside the API crate
- **Source ID:** A2-018
- **Category:** readability
- **Severity:** Low
- **Location:** `film_harness.rs` (9,088 lines; ~6,538 non-test), `film_harness/review.rs` (2,801), `bin/film-harness.rs` (2,017)
- **Finding:** A CLI/test-driver (`HttpTransport`, `ApiTransport`, plan/provenance/relayout logic, fixture sound writer at film_harness.rs:~790 lines) is compiled into the server library as `pub mod` and is the largest non-test file in the crate.
- **Impact:** Slow incremental builds and review; server binary links CLI-only code.
- **Suggested fix:** Split into `film_harness/{transport,plan,run,timeline,provenance}.rs`; consider a separate crate for the CLI + harness.
- **Confidence:** Medium

#### [F-166] Credential IPC: chmod after bind, non-constant-time token compare, serial accept loop, duplicated accept loops
- **Source ID:** X1-012
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `apps/desktop/src/cred_ipc.rs:132-158` (bind then `set_permissions` at `:140`), `:160-180` (`start_tcp`), `:196-200` (`presented != token`)
- **Finding:** (a) The Unix socket is created under the process umask and chmod'd to 0600 afterwards, and the parent dir comes from `create_dir_all` with default perms. (b) The token is compared with `!=`. (c) `listener.incoming()` handles each connection inline with a 2 s read timeout, so one idle local client blocks credential delivery for 2 s at a time. (d) `start_unix` and `start_tcp` are the same 15-line accept loop.
- **Impact:** Minor: local-only, and the 256-bit per-launch token still gates any secret. But (a) is a real TOCTOU window, and (c) lets any local process degrade API and worker credential pulls.
- **Suggested fix:** Create the socket's directory with mode 0700 (or set `umask` around `bind`), compare with `subtle::ConstantTimeEq` (already used in rust-api), spawn a thread per connection or use a short timeout, and share one `serve(listener)` generic helper.
- **Confidence:** Medium.

#### [F-167] Blocking work inside Tauri commands and an async fn
- **Source ID:** X1-014
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `apps/desktop/src/settings.rs:1332-1346` (`list_credentials`, sync, one keychain read per credential), `:1354` (`set_credential`, sync), `:1091-1105` (`reveal_in_os`, sync, `.status()` waits for the child), `:1317-1323` (`std::fs::copy` inside `async fn save_asset_as`)
- **Finding:** Sync `#[tauri::command]` functions run on the main thread. macOS keychain reads can prompt or stall, and `explorer`/`xdg-open` are waited on synchronously. `save_asset_as` copies potentially hundreds of MB with blocking `std::fs::copy` on the async runtime.
- **Impact:** UI freezes during credential listing (a keychain prompt blocks the window) and during large saves.
- **Suggested fix:** Make these commands `async` and wrap blocking bits in `spawn_blocking`.
- **Confidence:** Medium.

#### [F-168] `resolved_data_dir` duplicated within the desktop crate
- **Source ID:** X1-015
- **Category:** redundant
- **Severity:** Low
- **Location:** `apps/desktop/src/setup.rs:654-659`, `apps/desktop/src/settings.rs:1161-1167`; related `app_managed_roots` at `settings.rs:1108-1130`
- **Finding:** Two private identical functions compute override-or-default data dir; `app_managed_roots` re-derives the same override logic a third way.
- **Impact:** A change to the precedence rules can diverge between the sidecar launch and the reveal/save guard.
- **Suggested fix:** Make one `pub(crate)` in settings and call it from setup.
- **Confidence:** High.

#### [F-169] Hand-rolled loopback HTTP POST in `setup.rs` duplicates the worker supervisor's report call, and its justification is partly stale
- **Source ID:** X1-016
- **Category:** redundant
- **Severity:** Low
- **Location:** `apps/desktop/src/setup.rs:1940-2064` (`post_loopback_json`), `:2066+` (`report_worker_terminated`); `apps/desktop/Cargo.toml:67`
- **Finding:** The doc says the file mirrors `sceneworks_worker::supervisor::report_worker_terminated` and avoids a client crate because of a reqwest-0.12 vs 0.13 collision and a macOS "No provider set" panic. It is 40 lines of manual HTTP/1.1 with a header-injection guard written by hand.
- **Impact:** Another HTTP client surface to maintain, and a fixed 5 s timeout per read, not total. The reqwest 0.12/0.13 split is still in the lockfile (the vendored updater uses 0.13).
- **Suggested fix:** Reuse the worker supervisor's function or core's API client for this one POST, or at least note an owner for the workaround.
- **Confidence:** Medium.

#### [F-170] Stale cross-references and a misplaced F-041 comment
- **Source ID:** X1-017
- **Category:** readability
- **Severity:** Low
- **Location:** `crates/sceneworks-memory-adapter/src/bin/mlx.rs:22585` (cites `candle.rs:540-548`, actual `plain_execution_path` is at `candle.rs:1349`); `crates/sceneworks-mcp/src/server.rs:818-826` (F-041 comment sits after the bytes are already inlined; the cap is enforced earlier by `get_bytes_bounded`)
- **Finding:** Line-number citations rot, and the comment above `total_bytes += bytes.len()` describes a fallback that happens ~25 lines earlier. `generate_image` itself is a 230-line method (`:605-830`) combining submit, poll with outage tolerance, cancel, and inline assembly.
- **Impact:** Misleads future edits; the poll loop is not reusable for `submit_video_job`/yue2 callers.
- **Suggested fix:** Reference symbols, not line numbers. Move the comment next to the cap logic and extract `wait_for_job(...)`.
- **Confidence:** High.

#### [F-171] Memory-adapter tests hand-roll temp directories about 20 times; the crate has no `tempfile` dev-dependency
- **Source ID:** X1-018
- **Category:** redundant
- **Severity:** Low
- **Location:** `mlx.rs:1636,2151,2498-2503,13145,22882-22887,23747,24073,25067,25541,25747,26549,26634,29865,30419,31064,32199,32455`; `candle.rs:10104,10920-10931`; `lib.rs:2596,2612`; `mlx_ltx25.rs:1841,1938`; `candle_wan_scail2.rs:1053`
- **Finding:** `temp_dir().join(format!("sc-NNNNN-{label}-{pid}-{nonce}"))` is repeated with distinct sc-numbers, and the desktop crate already moved off exactly this pattern (sc-17707: panicking tests skip trailing cleanup and PID reuse inherits leftovers).
- **Impact:** Leaked directories on test failure; copy-paste maintenance.
- **Suggested fix:** Add `tempfile` as a dev-dependency (already in the workspace lock) and use `tempfile::TempDir`.
- **Confidence:** High.

#### [F-172] `image-quality::decode_transcoding` is a second copy of the worker's decode-then-transcode backstop
- **Source ID:** X1-019
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/sceneworks-image-quality/src/lib.rs:88-114` vs `crates/sceneworks-worker/src/image_decode.rs:33-108`
- **Finding:** The doc comment says it "mirrors the worker's `decode_image_any`". Both implement open, sniff, tempdir, `transcode_to_png`, re-decode, with slightly different failure semantics (image-quality surfaces the transcode error; the worker surfaces the original open error). Neither is shared from `sceneworks_core::media_convert`.
- **Impact:** Divergent error messages and the same ffmpeg/sips shell-out reachable from API request threads (the Dataset Doctor is synchronous) with no timeout in `transcode_to_png`.
- **Suggested fix:** Put one `decode_image_any` in core next to `media_convert` and use it in both crates.
- **Confidence:** Medium.

#### [F-173] Brittle source-scraping unit test in the MCP crate
- **Source ID:** X1-020
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `crates/sceneworks-mcp/src/server.rs:3217-3250` (reads `../../apps/rust-api/src/lib.rs` via `CARGO_MANIFEST_DIR` and slices on text)
- **Finding:** `video_job_body_reaches_every_mode_the_api_admits` parses the API's source text to extract `VIDEO_JOB_MODES`. A reformat or move of that constant breaks the MCP crate's test with no semantic change, and it couples a lower-level crate to an app crate's file layout.
- **Impact:** Spurious failures on refactors of rust-api; breaks if the crate is ever built outside the workspace tree.
- **Suggested fix:** Export the admitted-mode list from `sceneworks_core` (the API already depends on it) and assert against that.
- **Confidence:** Medium.

#### [F-174] Architecture rationale in `architecture_tests.rs` and `mlx.rs` contradicts the Cargo graph (core DOES depend on gen-core; memory-adapter DOES depend on core)
- **Source ID:** D1-007
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `crates/sceneworks-worker/src/architecture_tests.rs:626-634` (rationale for `IMPORT_SUPPORTED_FAMILIES`: "neither `sceneworks-core` nor `apps/rust-api` ... depends on `sceneworks-gen-core`"); same claim re-stated for `MLX_ROUTED_FAMILIES` ("a crate with no adapter-registry dependency"); `crates/sceneworks-core/Cargo.toml:13` (depends on `sceneworks-gen-core`; used in `contracts.rs`, `training/iris*.rs`, `supra2.rs`, `memory_calibration.rs`...); `crates/sceneworks-memory-adapter/src/bin/mlx.rs:28287` ("this crate deliberately does not depend on `sceneworks-core`") vs `crates/sceneworks-memory-adapter/Cargo.toml:18` (path dep on core, used at `bin/mlx.rs:15379`, `bin/mlx_wan_scail2.rs:647`); the doc at `bin/mlx.rs:519-524` correctly admits it.
- **Finding:** The "NOT DERIVABLE" justification for keeping core-owned family allow-lists is partly stale: core can see `gen_core`; the real remaining obstacle is that the `checkpoint_adapters()` registry is built in the worker (`inference_runtime.rs:492`) and cfg-gated.
- **Impact:** The recorded exceptions (E2 audit) are supposed to be re-validated by their reasons; readers will trust a false premise. `ltx_snapped_frame_count` (`mlx.rs:533`) is a transcribed copy of `ltx_frame_count` (`core/video_request.rs:573`) kept only because the latter is not `const fn`.
- **Suggested fix:** Reword the two rationales to the true blocker (registry lives in worker + cfg gate); make `ltx_frame_count` a `const fn` and delete the port and its second pin test.
- **Confidence:** High

#### [F-175] Worker `Cargo.toml` declares `candle-gen` twice, making the optional/`backend-candle` gating meaningless
- **Source ID:** D1-008
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `crates/sceneworks-worker/Cargo.toml:50` (unconditional), `:119` (target-gated, `optional = true`), `:78` (`backend-candle = [..."dep:candle-gen"...]`); consumer `crates/sceneworks-worker/src/training_jobs.rs:1368-1404` (`candle_gen::train::flow_match` used unconditionally).
- **Finding:** The comment at L116-118 says the dependency is a "weights-free contract dependency" kept direct and `optional`, but the plain entry at L50 makes it non-optional on all targets, so the target-gated entry and `dep:candle-gen` are dead declarations that read as if candle-gen were feature-gated.
- **Impact:** Misleads anyone reasoning about what the macOS / "neither" build links; pin bumps must touch both lines.
- **Suggested fix:** Delete the L119 entry and `dep:candle-gen` from the feature, and move the explanatory comment to L50.
- **Confidence:** High

#### [F-176] Memory adapter reimplements core's UTC formatter citing a dependency it already has; `now_rfc3339` wrappers duplicated
- **Source ID:** D1-009
- **Category:** redundant
- **Severity:** Low
- **Location:** `crates/sceneworks-memory-adapter/src/lib.rs:2326-2357` (`captured_at`, `civil_from_days`, comment "without adding a time-formatting dependency"); canonical `crates/sceneworks-core/src/time.rs:3-30,79`; wrappers `crates/sceneworks-worker/src/util.rs:30` and `apps/rust-api/src/lib.rs:4584`.
- **Finding:** `sceneworks-memory-adapter` depends on `sceneworks-core` (`Cargo.toml:18`), whose `utc_now()` produces the same format; the stated reason for the copy no longer holds. Two private `now_rfc3339()` shims also just call `format_unix_seconds(now_unix_seconds())`.
- **Impact:** Duplicate calendar math (Hinnant) to maintain.
- **Suggested fix:** Call `sceneworks_core::time::utc_now()` and delete the shims.
- **Confidence:** High

#### [F-177] Stale prose in CONTRIBUTING.md, core README and worker ARCHITECTURE.md
- **Source ID:** D1-010
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `CONTRIBUTING.md:139-141` ("on every PR *and* on the merge queue's speculative merge" — queue removed 2026-08-11 per CLAUDE.md/FEATURE_DEVELOPMENT.md; `check.yml:6-11` keeps `merge_group:` only as a dormant trigger); `CONTRIBUTING.md:92-101` (says `rust:check` leads with a derived-docs check covering `memory-matrix.*` + `tier-integrity.*`, but `package.json:20` `check:rust-derived-docs` runs only `check:tier-integrity`); `CONTRIBUTING.md:163-165` lane names table vs `check.yml` (`parity`, `candle` jobs; `windows-candle`/`nax-worker` exist in other workflows); `crates/sceneworks-core/README.md:3-4` ("kept dependency-light while the Rust backend scaffold takes shape" — core is ~100 modules, SQLite/image/png/inference gen-core); `crates/sceneworks-worker/ARCHITECTURE.md` routing note 4 lists four `NON_GPU_JOB_TYPES`, code (`core/src/jobs_store.rs:115-122`) has six (adds `lora_download`, `dataset_parquet_import`; the matrix table rows themselves are right).
- **Finding:** Each is a doc claim contradicted by the code/CI as of today. (The memory-matrix gate removal is probably the deliberate gate teardown; the prose should say so.)
- **Impact:** Contributors told a merge-queue backstop exists; routing readers miss two CPU-only job kinds.
- **Suggested fix:** Edit the five passages; for ARCHITECTURE.md, derive the list from `NON_GPU_JOB_TYPES` in the existing drift test.
- **Confidence:** High

#### [F-178] Same helper test scaffolding copied across test modules (`isolate_hf_cache` x12, smoke `env_path`/`hub_cache_root`)
- **Source ID:** D1-011
- **Category:** redundant
- **Severity:** Low
- **Location:** `isolate_hf_cache` at `apps/rust-api/src/tests/support.rs:69` (the shared one), `worker/src/converted_install.rs:264`, `downloads.rs:2777`, `audio_jobs.rs:3579`, `depth_jobs/tests.rs:135`, `image_jobs/iris.rs:49`, `image_jobs/sdxl_control.rs:753`, `image_jobs/supra2.rs:37`, `image_jobs/tests.rs:28689`, `model_jobs.rs:6921` and `:7738` (same module twice), `tests/lora_and_downloads.rs:1336` — all set `HF_HUB_CACHE/HUGGINGFACE_HUB_CACHE/HF_HOME` to `""`; `env_path` x10 across `*_smoke.rs`; `hub_cache_root` at `mage_flow_q8_mlx_smoke.rs:54` and `flux2_klein_9b_q4_mlx_smoke.rs:42` (identical).
- **Finding:** `worker/src/test_env.rs` already exists as the shared env-test module but does not export an isolate helper; `smoke_support.rs:85` exports `env_or` but several smokes still define their own.
- **Impact:** Adding another HF env var (e.g. `HF_HUB_CACHE` successor) means editing 11 places; one missed copy leaks the developer's real cache into a test.
- **Suggested fix:** Add `test_env::isolate_hf_cache()` and `smoke_support::{env_path, hub_cache_root}`.
- **Confidence:** High

#### [F-179] HTTP client construction and restart-backoff policy are scattered, with differing timeouts/policies
- **Source ID:** D1-012
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** clients: `crates/sceneworks-worker/src/api_client.rs:27`, `downloads.rs:36,1340,1850,2082,2353`, `crates/sceneworks-mcp/src/api_client.rs:101`, `apps/desktop/src/cuda_provision.rs:759`, `linux_cuda_provision.rs:646`, `apps/rust-api/src/lib.rs:1437` (`reqwest::Client::new()` — no default timeout — shared as `state.http_client` for `film_planning.rs:475`/`models.rs:11560`, which rely on per-call timeouts); backoff: `crates/sceneworks-worker/src/supervisor.rs:40-47` (1,2,4,8,8... table), `crates/sceneworks-worker/src/lib.rs:2779 retry_delay` (poll x2^n clamp 30), `apps/desktop/src/setup.rs:2152-2213,2395` (1→30s doubling, 20 s healthy reset).
- **Finding:** Prior F-001 (timeouts) is fixed where it was reported, but the pattern is "every site builds its own client"; the rust-api shared client has no baseline timeout, so a new call site that forgets `.timeout()` reintroduces F-001. Three independent backoff policies with different constants.
- **Impact:** Latent regression path for F-001; operators see different restart cadences for the API sidecar workers vs desktop-supervised workers.
- **Suggested fix:** Build `state.http_client` with `connect_timeout` + a generous total/read timeout; centralize the backoff schedule as one small core type used by desktop and worker supervisors.
- **Confidence:** Medium

#### [F-180] Per-crate dependency declarations not centralized; one runtime dep that is test-only
- **Source ID:** D1-013
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `Cargo.toml` `[workspace.dependencies]` (only `tracing`, `tracing-subscriber`, `trash`, `image`, `png`, `ort`); repeated per crate: `serde` (`"1"` in `apps/desktop/Cargo.toml` vs `"1.0"` elsewhere), `serde_json`, `uuid "1.11"` (3x), `tokio "1.40"` (5x with five different feature sets), `sha2 "0.10"` (6x, plus an `asm` variant `crates/sceneworks-core/Cargo.toml:80`), `reqwest "0.12"` (4x, three feature sets), `nix "0.29"` (3x), `tempfile "3.13"`, `rusqlite "0.32"` (core: `bundled,functions`; worker/rust-api dev: `bundled` only). `apps/rust-api/Cargo.toml` lists `tempfile` under `[dependencies]` but non-test code never uses it (only `#[cfg(test)]` in `openai_planner.rs:1051`, `generation.rs:3072`).
- **Finding:** The workspace already centralizes `image`/`png`/`ort` after real skew incidents (see the comment block in root `Cargo.toml`); the rest rely on convention. Lockfile duplicates are all transitive and acceptable: `reqwest 0.12/0.13` (0.13 via `rmcp`, `tauri`), `png 0.17/0.18` (`tauri`, `tiny-skia`), `nix 0.29/0.31` (`zip`), `sha2 0.10/0.11` (`rust-embed-utils`), `rand 0.8/0.9/0.10`, `toml` x3, `windows-*` many.
- **Impact:** Version/feature skew is silent under feature unification (the root `Cargo.toml` documents sc-15052/sc-15028 as the precedent). The test-only `tempfile` bloats the shipped API dependency list.
- **Suggested fix:** Hoist `serde`, `serde_json`, `tokio`, `reqwest`, `sha2`, `uuid`, `tempfile`, `nix` into `[workspace.dependencies]`; move rust-api's `tempfile` to `[dev-dependencies]`.
- **Confidence:** High

#### [F-181] Hard-coded author-machine paths and orphaned epic scripts committed
- **Source ID:** D1-014
- **Category:** dead-code
- **Severity:** Low
- **Location:** `scripts/lib/moss-edit-terminal-actions.mjs:20`, `scripts/ci/moss-upstream-reference-common.mjs:10-11,28`, `scripts/ci/moss-upstream-quality.mjs:10-17`, `scripts/lib/moss-edit-quality-amendment.test.mjs:61,114-126,212-222`, `scripts/moss-edit-terminal-actions.test.mjs:437,492,625-665` (all `/Users/michael/.codex/epic-runs/sc-26752/...`); `crates/sceneworks-worker/src/krea_control_mlx_smoke.rs:139` (`/Users/michael/Repos/SceneWorks/poses/dance_01.png`).
- **Finding:** Constants and tests bind to directories that exist only on one developer's Mac, in dense single-line style (`requireThat(...)` chains).
- **Impact:** These scripts/tests cannot run anywhere else; readers cannot tell live tooling from epic residue. (Not flagged: unwired gate scripts per the gate-teardown policy.)
- **Suggested fix:** Parameterize roots via env/CLI args, or move finished epic tooling to an `archive/` path excluded from CI discovery.
- **Confidence:** Medium

#### [F-182] Repo hygiene: bulk calibration evidence, unreferenced binaries, duplicated pose PNGs
- **Source ID:** D1-015
- **Category:** dead-code
- **Severity:** Low
- **Location:** `docs/calibration/` (668 tracked files, ~197 MB working tree: 61 raw `.rgb` 3 MiB frames, 164 `.log`, 382 `.json`; `.git` is 91 MB); `docs/generated/memory-matrix.json` (9.2 MB); `design/*.zip` (12 zips, 4.3 MB, referenced only from `docs/code-review/CODE_REVIEW_2026-07-12.md`); `logo-large.png` (0 references); root `poses/` PNGs duplicate `apps/web/public/poses/*.png` (46 identical files; root copy used only by test fixture path `crates/sceneworks-worker/src/image_jobs/tests.rs:18863`); 30 `docs/sc-*` entries (e.g. `docs/sc-15360`, `sc-16353`, `sc-6529`, `sc-3031-parity.md` with zero inbound references), Python spikes in `scripts/spikes/` and `docs/sc-*/*.py` in a repo whose README says Python was removed; `vendor/*/Cargo.toml.orig`. No `target/`, `node_modules/`, `.DS_Store` or `*.pyc` are tracked.
- **Finding:** The `.log`s and `.rgb` are cited by `docs/generated/memory-calibration-evidence.json` and the calibration runbook, so they are deliberate evidence, not accidents; the cost is repo weight (every clone) rather than correctness. Root screenshots (`screenshot*.png`, ~3.5 MB) are README-referenced and fine.
- **Impact:** Clone/CI checkout size; stale `docs/sc-*` look like current documentation.
- **Suggested fix:** Move raw `.rgb` frames (hash-addressed already) to release assets/LFS with a manifest of digests; delete `logo-large.png` and `design/*.zip` if unused (or move to the design repo); point `poses/` consumers at `apps/web/public/poses`; add an "archived experiment" banner or move unreferenced `docs/sc-*` under `docs/archive/`.
- **Confidence:** Medium


### Informational findings — Rust backend

#### [F-183] Misc cleanliness: stray module-level `#[test]`, a silent credential parse failure, and an odd `src/` file
- **Source ID:** W3-017
- **Category:** readability
- **Severity:** Info
- **Location:**
  - `model_jobs.rs:5527-5539` has a free-standing `#[cfg(test)] #[test] fn …` in the middle of production code, outside the `mod tests` blocks at 3150+ and 6600+. `model_jobs.rs` (9.8k lines) interleaves four separate test modules and two huge subsystems (HF snapshot resolution, receipts and conversion).
  - `credentials.rs:44` uses `serde_json::from_str(raw).unwrap_or_default()`, so a syntactically broken `SCENEWORKS_CREDENTIALS` or credentials file silently produces no credentials. Gated downloads then fail as 401/403 with no hint.
  - `model_artifact_inventory.rs` sits in `src/` but is not a module of the lib. It is only `include!`d by `tests/model_artifact_inventory.rs`. It uses `//` header comments rather than `//!`, so `rustfmt`/`rustdoc` treat it differently.
- **Finding:** Small structural and diagnostic nits.
- **Impact:** Harder navigation. Credential typos are undiagnosable.
- **Suggested fix:** Log a `warn!` on credential parse failure (without the body). Split `model_jobs.rs` by concern (`hf_snapshot`, `receipts`, `convert`, `import`). Move the inventory file next to its test.
- **Confidence:** High

#### [F-184] `huggingface_hub_cache_dir` repeats the trimmed non-empty env read three times
- **Source ID:** C1-017
- **Category:** readability
- **Severity:** Info
- **Location:** `crates/sceneworks-core/src/hf_home.rs:116-139`; also the set-check in `45-52`
- **Finding:** Three consecutive blocks do `std::env::var(K).ok().map(trim).filter(!empty)`. `default_huggingface_home` has a fourth variant of the same predicate.
- **Impact:** Cosmetic, but the precedence order is easier to get wrong than it needs to be.
- **Suggested fix:** A local `env_nonempty(key)` helper.
- **Confidence:** High

#### [F-185] Vendored crates: both are still needed, but the trees carry unneeded files and the updater pin is one minor behind
- **Source ID:** X1-021
- **Category:** dead-code
- **Severity:** Info
- **Location:** `vendor/glib-0.18.5/` (`VENDORED.md`), `vendor/tauri-plugin-updater-2.11.0/` (`VENDORED.md`), `Cargo.toml:182-188`
- **Finding:** (1) glib 0.18.5 carries the upstream two-line `VariantStrIter::impl_get` fix (gtk-rs-core #1343) because GTK3 (via Tauri/wry on Linux) pins `glib ^0.18` and no 0.18.x release contains it. Still needed until the whole GTK stack moves. Its VENDORED.md says advisory scanners will keep flagging 0.18.5. (2) tauri-plugin-updater 2.11.0 differs from the published crate only in `Cargo.toml` (`zip` 4 to 9, for RUSTSEC-2026-0367/0368/0369). VENDORED.md says 2.14.0 (2026-10-17) still depends on zip 4, so the patch is still needed, but it is now three minors behind upstream, and the updater is security-sensitive (signature verification), so re-check at each release. (3) Both vendored trees include `benches/`, `tests/`, `Gir*.toml`, `api-iife.js` and other non-build files.
- **Impact:** Maintenance burden only; I could not verify upstream status offline today (2026-10-27).
- **Suggested fix:** Re-check updater >= 2.15 for a zip-9 release at the next release cut and drop the patch then; prune non-build files from `vendor/`.
- **Confidence:** Medium.

#### [F-186] Info: `apps/rust-worker` is a 8-line wrapper; per-crate `reqwest` specs
- **Source ID:** X1-022
- **Category:** architecture-drift
- **Severity:** Info
- **Location:** `apps/rust-worker/src/main.rs:1-8`; `crates/sceneworks-mcp/Cargo.toml:30`, `crates/sceneworks-worker/Cargo.toml:45`, `apps/desktop/Cargo.toml:67`, `apps/rust-api/Cargo.toml:21`
- **Finding:** The binary is a clean thin wrapper (`iris_rank_entry`, then `run_on_worker_entry_thread`). The Cargo comment says the worker library is "also embedded in-process by the Rust API (sc-1342)", which I did not verify. `reqwest = "0.12"` with the same feature set is declared four times with small feature differences instead of `[workspace.dependencies]`, which `image` already uses.
- **Impact:** Version or feature drift is possible; none observed.
- **Suggested fix:** Hoist `reqwest` into `[workspace.dependencies]`.
- **Confidence:** Medium.

#### [F-187] No executable guard on crate dependency direction; rust-api statically embeds the whole worker (info)
- **Source ID:** D1-016
- **Category:** architecture-drift
- **Severity:** Info
- **Location:** `apps/rust-api/Cargo.toml:21` (depends on `sceneworks-worker`), `apps/rust-api/src/server.rs:923-949` (runs `sceneworks_worker::run()` in-process), `apps/rust-api/src/iris_curricula.rs:38-213`, `jobs.rs:606-614`, `models.rs:1826-4274`, `catalogs.rs:30,983,1032` (API calls worker internals); `crates/sceneworks-worker/src/architecture_tests.rs` (guards job types, dispatch, family lists, `json!` imports only).
- **Finding:** Observed direction: core ← {image-quality, mcp, memory-adapter, worker, desktop}; worker ← {rust-worker, rust-api}; rust-api ← mcp, image-quality. No cycles, core imports nothing from worker/rust-api (verified by Cargo.toml; only doc comments mention `sceneworks_worker::`), desktop depends only on core. rust-api -> worker is a real layering edge: the API links every inference engine and reaches worker modules for training/model-convert/catalog logic, but no test or doc states that this edge is intentional or what surface of `sceneworks_worker` the API may use.
- **Impact:** None today; this is the seam where an API-to-worker dependency would grow unchecked (the E2 audit explicitly counts API allow-lists but not API-to-worker reach-through).
- **Suggested fix:** Add an architecture test pinning allowed `use sceneworks_worker::` paths from rust-api (or move the shared pure helpers named above — `resolve_model_convert_output`, `catalog_*`, Iris planners — into core).
- **Confidence:** Medium

#### [F-188] Prior-review spot check: nearly all High/Medium items fixed; two remain open (info)
- **Source ID:** D1-017
- **Category:** architecture-drift
- **Severity:** Info
- **Location:** `docs/code-review/CODE_REVIEW_2026-07-12.md`, `_2026-07-01.md`, `_2026-05-17.md`
- **Finding:** Verified FIXED in current code (24): 07-12 F-001 (timeouts: `worker/api_client.rs:28-29`, mcp connect timeout), F-003 (`slugify(&request.model...)` at `image_jobs.rs:2949,3436`), F-013 (`supervisor.rs:293-297,370`), F-019 (`cache_thread.rs:538` shared), F-021 (worker Cargo.toml 1,922 -> 135 lines), F-023 (`builtin_manifests` uses `store_util::atomic_write`), F-024 (`project_store.rs:830-838` skips bad entry), F-025 (cached write connection `jobs_store.rs:3455`), F-026 (busy_timeout, `training_store.rs:2412`), F-028 (ownership race won before side effects, `rust-api/jobs.rs:1835-1871`), F-040 (`allowed_hosts` loopback default, `mcp/lib.rs`), F-041 (`MAX_INLINE_IMAGE_BYTES`, `mcp/server.rs:67`), F-043 (banners; 3 of 5 docs), F-048 (versions aligned at 0.10.3), F-051 (`desktop/setup.rs:2119 supervise_worker`), F-010, F-014; 07-01 F-001 (actions SHA-pinned), F-002 (`validate_lora_import_source_path`), F-005, F-006; 05-17 F-008, F-009. STILL OPEN: 07-12 F-020 (see D1-006, worse); 07-12 F-029 partially — CRUD extracted to `rust-api/src/manifest_entity.rs`, but ~90 diff-lines of mirrored scope/location/validation helpers remain between `prompt_batches.rs:211-395` and `recipe_presets.rs:282-657`.
- **Impact:** Remediation discipline is strong; the remaining risk is the structural findings, not the point fixes.
- **Suggested fix:** None beyond D1-006.
- **Confidence:** High


# Part 2 — Web app, scripts and CI


### Critical findings — web/tooling

*None.*


### High findings — web/tooling

*None.*


### Medium findings — web/tooling

#### [F-189] Split the 3,750-line `App()` component along its existing section boundaries
- **Source ID:** B1-001
- **Category:** readability
- **Severity:** Medium
- **Location:** `apps/web/src/App.jsx:574-4320` (the single `App` function); sections at 1167-1330 (job creators), 2008-2490 (refresh/hydration), 2524-2810 (job/prompt runners), 2822-3265 (send-to-studio and asset actions), 3460-3580 (job actions), 3637-3891 (context literals), 1797-1900 (WebView2 repaint pump)
- **Finding:** `App()` is one function with about 238 hook calls plus roughly 80 inline `async function`/`useCallback` bodies. `useJobEvents`, `useAccessGate`, `useCharacters` and the others were extracted since F-052, but the file went from 2.4k to 4.3k lines. The growth is new feature wiring (workflow embed, model library, Simple UI, training datasets, film), not shell logic.
- **Impact:** Every feature touches this file. Review and merge conflicts concentrate here, and the hand-maintained dependency arrays (B1-002) are the main place bugs hide.
- **Suggested fix:** Extract in this order, each as a `useX({deps}) -> actions` hook like the existing ones. (a) `usePromptRunners` for refinePrompt, qwenRewritePrompt, magicPrompt, imageCaption, imageDescribe and compareFaceLikeness (2594-2782); they are already pure `pollJobToCompletion` wrappers over `token`. (b) `useAssetActions` for 3094-3265 (status, tags, move, delete, purge, preview delete). (c) `useSendToStudio` for 2822-3086 (`sendAssetTo*`, `launchImageRecipe`, `sendPresetToStudio`, `sendCharacterTo*`). (d) `useJobActions` for 3460-3580. (e) `useShellHydration` for 2008-2490, including the ledger refs. (f) `useWebView2RepaintPump` for the 100-line effect at 1797-1900 and `useThemeAccent` for the theme and accent effects at 1611-1640. (g) Move the nav and keep-alive tables (258-372) and `ProjectSwitcher`/`FirstRunProjectGate` (373-573) into `components/shell/`. Together these remove roughly 1,800 lines with no behavior change.
- **Confidence:** High

#### [F-190] Hand-mirrored 230-line context literals and a 30-ref forwarding pattern are fragile
- **Source ID:** B1-002
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `apps/web/src/App.jsx:3655-3891` (`appStaticValue` object plus dependency array), `App.jsx:1034-1064` (about 28 `*Ref` handles), `App.jsx:1314-1322`
- **Finding:** `appStaticValue` lists about 300 names twice: once as the object and once as the `useMemo` dependency array. A comment says "each dependency array must mirror the corresponding object". Forwarding refs (`refreshAssetsRef`, `refreshCharactersRef`, `refreshModelsRef`, ...) plus `stableRefreshData*` wrappers exist because the `function refreshX()` declarations (2051-2201, 2473) are recreated every render. This was the root cause of F-009.
- **Impact:** A key added to the object but not the array (or the reverse) gives a stale or never-updating context value. The compiler and `react-hooks/exhaustive-deps` do not check a manually built literal, and no test pins the equality.
- **Suggested fix:** Convert the `refreshX` function declarations to `useCallback` in their hooks, then delete the forwarding refs. Build the context value from domain sub-objects (`{ training, characters, timelines, models, assets }`), each memoized in the hook that owns it. Failing that, add a test that parses the literal and the array and asserts equal key sets.
- **Confidence:** High

#### [F-191] Seven hooks hand-roll the same project-scoped refresh body; `useScopedCrudList` covers only two
- **Source ID:** B1-003
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/web/src/hooks/useCharacters.js:20-48`, `usePersonTracks.js:17-45`, `useSavedVoices.js:19-60`, `useModelsAndLoras.js` (refreshLoras), `useTimelines.js`, `useTraining.js`, `App.jsx` (`refreshAssets`, 2154-2200); the shared form is `hooks/useScopedCrudList.js:22-46`
- **Finding:** Each hook repeats the sequence: missing-project guard, stale pre-check, `apiFetch`, stale post-check, `setItems`, `setError("")`, `refreshSuccess`, and a catch with stale, abort and error branches. `rg 'refreshFailure\("stale"\)'` finds this in 8 files. `useScopedCrudList` already encapsulates it, but only `usePresets` and `usePromptBatches` use it.
- **Impact:** F-056 had to be fixed by hand in four copies. The next fix (a new status such as "interrupted", or a changed notice policy) must again be applied to seven or eight places. `useSavedVoices` already uses a slightly different helper (`isCurrentVoiceRequest`) than its siblings.
- **Suggested fix:** Extract `useProjectScopedList({ path, token, activeProject, activeProjectRef, setError, normalize })` (or add a `scope: "project-only"` mode to `useScopedCrudList`) returning `{ items, setItems, refresh }`. Rebuild characters, voices, person-tracks and loras on top of it.
- **Confidence:** High

#### [F-192] Film timeline refresh polls once per second for as long as a film timeline is selected
- **Source ID:** B1-004
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `apps/web/src/hooks/useTimelines.js:98-115`
- **Finding:** The comment says "bounded read refresh". The effect is keyed on `[activeProject.id, selectedTimelineId, isFilmTimeline, token]`, where `isFilmTimeline = Boolean(activeTimeline?.filmAssembly)`. That value stays true after delivery, so `setInterval(…, 1000)` runs forever. Each tick is a full `GET …/timelines/:id` that is parsed and rebased into editor state. The request takes no `AbortSignal`, and the hook lives in `App`, so it keeps polling when the Editor is not the active view, and in background tabs. An SSE stream already exists.
- **Impact:** Steady 1 req/s against the local API plus `rebaseTimeline` work for as long as the selection is a film timeline, including after delivery and while on other screens. In remote/LAN mode it keeps hitting the auth throttle path.
- **Suggested fix:** Stop polling once the assembly status is terminal (delivered, failed or cancelled). Poll only while the Editor is visible and `document.visibilityState` is visible. Better, publish a `timeline.updated` or `job.updated`-driven refresh through `useJobEvents`. Pass an `AbortController` signal and abort it in cleanup.
- **Confidence:** Medium (the terminal-state field was not traced, but no stop condition is visible in the effect)

#### [F-193] API layer is mostly bypassed; request paths are hand-written at 236 call sites with unencoded ids
- **Source ID:** B1-005
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `apps/web/src/api.js:1-265`, `api/` (only `films.js`, `filmReview.js`, `filmPlannerConnections.js`, `yue2.js`); `rg 'apiFetch\(' --glob '!*.test.*'` gives 236 sites in 45 files, including components and screens (`screens/ImageEditor.jsx`, `ImageStudio.jsx`, `SettingsScreen.jsx`, `components/Moss*Controls.jsx`, `ControlOverlayPicker.jsx`). 53 of 56 `apiFetch(\`/api/v1/projects/${id}…\`)` sites interpolate the id without `encodeURIComponent` (for example `hooks/useCharacters.js:25`, `useTimelines.js:107`)
- **Finding:** `api/` is meant to be the endpoint layer, but only four domains live there. `api.js` itself mixes the transport (`apiFetch`, `ApiError`, tickets) with domain endpoints (workflow inspect, Iris curricula, training data-walk, Iris sidecars at 187-265). Hooks, screens and components build paths and bodies inline. The film and yue2 modules encode ids; the hooks mostly do not.
- **Impact:** No single place to find, mock, or change an endpoint contract. Id encoding is inconsistent, which is harmless for UUIDs but breaks if an id ever contains `/`, `?` or `#`. Tests mock `fetch` per file (hence the 34 FakeEventSource/fetch fixtures) because there is no seam.
- **Suggested fix:** Split `api.js` into `api/client.js` (transport, `ApiError`, tickets) and per-domain modules (`api/training.js`, `api/workflows.js`, `api/projects.js`, ...). Migrate the hooks first, since they have the most repeats. Add an ESLint `no-restricted-imports` or custom rule so `apiFetch` can only be imported from `api/**`, and always `encodeURIComponent` ids in the helper.
- **Confidence:** High

#### [F-194] `modelCatalog.js` is a 1,233-line hand-maintained mirror consumed only by tests and covering a third of the manifest
- **Source ID:** B1-007
- **Category:** dead-code
- **Severity:** Medium
- **Location:** `apps/web/src/modelCatalog.js:1-1233`, `apps/web/src/data/previewSupport.json`, `apps/web/scripts/generate-preview-support.mjs:86-166`, `apps/web/vite-plugin-bundle-report.js:76`
- **Finding:** After sc-24217 moved the catalog out of the eager graph, nothing in production imports `modelCatalog.js` (`rg 'modelCatalog|fallbackModels'` hits only `*.test.*` files, plus comments in `constants.js` and `modelEligibility.js`). The file has 46 entries; `config/manifests/builtin.models.jsonc` has 140. Missing entries include ideogram_4, flux2_dev, krea_2_*, boogu_image*, iris_3b*, mage_flow*, anima_*, pid_* and sana_*. It is a hand-written third copy beside the manifest and the Rust `MODEL_TABLE`. `previewSupport.json` is generated (and kept in CI-guarded sync) only to feed this file, and the bundle report still lists "Offline model catalog" as a tracked chunk.
- **Impact:** Parity tests (`controlModesParity`, `imageAxesParity`, `memoryFloorCatalogParity`, `videoGeometryParity`) guard only the 46 seeded models. Newer models are unguarded, and each new model asks the author to hand-edit this file for tests only. The `previewSupport.json` regeneration step is overhead with no runtime consumer.
- **Suggested fix:** Either delete the mirror and have the parity tests read `builtin.models.jsonc` through a small JSONC loader (the pytest side already does), or generate `modelCatalog.js` from the manifest the way `previewSupport.json` is. If it stays, add a test asserting the seed ids are a subset of the manifest ids so it cannot rot silently. Drop the `previewSupport.json` output and its bundle-report entry if the mirror goes.
- **Confidence:** High (importer check done by grep; the "test-only" reading matches the file header, so this is partly intended)

#### [F-195] Batch run keeps posting jobs after ImageStudio unmounts (project switch) and cannot be stopped
- **Source ID:** B2-001
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/web/src/screens/ImageStudio.jsx:2807-2814, 2818-2984`; remount key at `apps/web/src/App.jsx:4190`
- **Finding:** `runBatch` is an async loop that awaits `waitForBatchSlot()` and `createImageJob()`. `batchAbortRef` is only set by `cancelBatchRun` (L2987); there is no unmount cleanup. App.jsx keys ImageStudio by `activeProject.id:studioRestoreEpoch`, so switching project (or a studio restore) unmounts the component mid-run.
- **Impact:** The orphaned loop keeps enqueuing jobs for the old project, with a closure over the old request state, and the user has no Stop button for it. Once `activeJobIds` stops updating, the loop parks forever on a promise held in `batchSlotWaitersRef`. Large Cartesian batches (up to the confirmed cap) are the worst case.
- **Suggested fix:** Add `useEffect(() => () => { batchAbortRef.current = true; releaseBatchSlotWaiters(); }, [])`, and check a mounted ref before each `createImageJob`. Consider moving the batch runner into a hook (see B2-002).
- **Confidence:** Medium (derived from reading; keep-alive does not apply across a project switch because of the key).

#### [F-196] ImageStudio is a single 4,500-line component; propose concrete split seams (F-035, still present and worse)
- **Source ID:** B2-002
- **Category:** readability
- **Severity:** Medium
- **Location:** `apps/web/src/screens/ImageStudio.jsx:332-4846` (component body); 74 `useState`, 27 `useEffect`, 83 hook calls, 6 `exhaustive-deps` disables (L1341, 1431, 1445, 1667, 1918, 2135)
- **Finding:** Logic is L332-3204 (about 2,870 lines) and one JSX return is L3204-4845 (about 1,640 lines). The `screens/imageStudio/` folder holds only 3 tiny files (`useImageControlState.js` is 20 lines and exports `useStrictControlState`/`usePidState`, so the filename doesn't match). Review said 3.2k lines in July; it is now 4.9k.
- **Impact:** Every feature (MOSS, Iris, Krea multi-phase, hires-fix, batch, structured captions) edits the same closure. Order-dependent TDZ hazards are already documented in comments (L1420-1422). The cross-field reset effects (L1310-1342 touches about 15 setters) are fragile.
- **Suggested fix:** Extract along existing comment-delimited seams, in this order:
  1. `useImageBatchRun` (L2797-3000: replaceBatchRun, waitForBatchSlot, runBatch, cancelBatchRun, plus `useBatchPromptState`).
  2. `useImageRecipeReplay` (L1920-2135, a 215-line effect). It should be a pure `recipeToStudioPatch(recipe)` plus a thin effect.
  3. `useMossImageState` (L535-555, 1648-1670, `MossOrdinaryControls`).
  4. `useStructuredPrompt` (`onMagicExpand`/`onImageCaption`/`onImageDescribe`, L2175-2275).
  5. JSX: `<ImageAdvancedPanel>` (about L4208-4760), `<ImageReviewPanel>` (L4791+), `<ImagePromptComposer>` and the generation-settings row.
  6. A shared `validateSubmitPreconditions()` for the guards duplicated between `submit` and `runBatch` (see B2-009).
- **Confidence:** High

#### [F-197] ImageEditor component is 3,300 lines; pure helpers and 190-key "scope bag" undercut the existing split
- **Source ID:** B2-003
- **Category:** readability
- **Severity:** Medium
- **Location:** `apps/web/src/screens/ImageEditor.jsx:1208-4546` (component), `:1-1207` (about 50 exported pure helpers), `:3496-3676` (`toolPanelScope`), `apps/web/src/screens/imageEditor/ImageEditorToolPanel.jsx:22-45`
- **Finding:** Tool hooks (mask/boxes/color) and the tool panel were extracted, but `ImageEditor.jsx` still holds the history/undo, layer ops, crop, AI-op runner, save/export and shell. Roughly 40 exported helpers (`editorDownloadPlan`, `buildSaveProvenance`, `historyCheckpoint`, `leaveGuardArming`, `blobToEditorImage`, `exportEditorFile`, …) are exported only for tests: zero non-test importers (checked by grep). The tool panels receive a ~190-key `toolPanelScope` object that is re-proxied by `useStableImageEditorToolPanelScope`. Each panel destructures 60+ keys on one line (L57-58), and `samePanelProps` does a keywise `Object.is` across all of them. That is a service locator, not a component interface.
- **Impact:** Any new editor control means touching 3 sites (state, scope literal, panel destructure). Memoisation is mostly defeated, since any changed key (`jobs`, `assets`, `working`) re-renders the active panel. Unit-testing the pure logic requires importing a 4.5k-line React module.
- **Suggested fix:** Move L1-1207 helpers into `screens/imageEditor/{history,export,provenance,leaveGuard,launchGuard}.js` (tests re-point imports). Extract `useEditorHistory` (L1882-2046), `useLayerOps` (L2597-2760), `useEditorAiOps` (L2903-3213) and `useEditorSave` (L3213-3430). Give each tool panel a narrow props object built by its own hook (e.g. `useEditTool()` returns `{state, actions}`) instead of one global scope.
- **Confidence:** High

#### [F-198] Sampler/scheduler/steps seed-snap-render logic is copy-pasted across four surfaces
- **Source ID:** B2-005
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/web/src/screens/ImageStudio.jsx:1510-1522, 1629, 1673-1683, 4262-4290`; `apps/web/src/screens/VideoStudio.jsx:1005-1010, 2962`; `apps/web/src/components/CharacterAdvancedOptions.jsx:65-103, 333`; `apps/web/src/components/editor/GenerationRail.jsx:254-266`; also `screens/PresetManagerScreen.jsx:1621-1634`
- **Finding:** All of these re-derive `samplerOptionsFromModel`/`schedulerOptionsFromModel`, gate the picker on `options.length > 1`, snap to the model default when the current value leaves the menu, and render near-identical `<select>`s. `CharacterAdvancedOptions.jsx:20-23` documents "mirrors the Image Studio advanced panel" as the consistency mechanism.
- **Impact:** Each new knob (sc-15299 negative-prompt axis, sc-25681 CFG interval, etc.) has to be added to each copy and has drifted before. `samplerOptions.js` shares only the pure parts.
- **Suggested fix:** Create a `useSamplerControls(model, backend, {saved})` hook plus `<SamplerSchedulerFields>` in `components/`, consumed by all four surfaces.
- **Confidence:** Medium (Image/Character/Rail confirmed; Video checked at hook level only).

#### [F-199] ImageStudio re-implements helpers that already exist in the Simple UI modules (and the Simple UI is the wrong home)
- **Source ID:** B2-006
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/web/src/screens/ImageStudio.jsx:282-293, 394-418`; `apps/web/src/simple/modelDefaults.js:25-46`; `apps/web/src/simple/simpleJobs.js:33-40`; `apps/web/src/screens/DocumentStudio.jsx:128-131` vs `ImageStudio.jsx:295-298`
- **Finding:** `preferredOption`/`preferredResolution` are defined in ImageStudio with subtly different semantics from the simple twins (the ImageStudio `preferredOption` falls back to `"default"` for an empty list, the simple one returns `undefined`). The `convRotEligible`/`nvfp4Eligible` memos duplicate `workerAdvertises()` from `simple/simpleJobs.js`, whose comment says it "mirrors the advanced studios' gates". `formatResolutionLabel` is copied verbatim into DocumentStudio and ImageStudio. Shared logic sits under `simple/`, so the advanced screens would have to import "downwards" from the simple package.
- **Impact:** Simple and advanced runs of the same model are documented as byte-identical ("starts on the same geometry"), but the guarantee rests on two code copies staying in sync. The `"default"` vs `undefined` empty-list difference is already a behavioural fork.
- **Suggested fix:** Move `workerAdvertises`, `preferredOption`, `preferredResolution` and `formatResolutionLabel` into a neutral module (e.g. `studioDefaults.js`). Have ImageStudio, DocumentStudio and `simple/` import it, and delete the local copies.
- **Confidence:** High

#### [F-200] Film render/export `while (run.controllerActive)` loops have no unmount, project-switch, or iteration guard
- **Source ID:** B3-001
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `components/editor/FilmWorkspace.jsx:503-509`, `components/editor/FilmWorkspace.jsx:546-550`
- **Finding:** `renderSelected` and `exportCurrentCut` poll the run every 500 ms inside the click handler until `controllerActive` clears. They have no AbortController, mounted flag, or project/draft check, and they capture `activeProject.id` and `token` from the render that started them. Any thrown fetch error ends the loop (`catch` then `setNotice`), and the loop never checks whether the component or project is still current.
- **Impact:** After navigating away or switching project, the loop keeps hitting the old project's endpoint and calls `setLastRun`/`setNotice` for a stale run, so another project's workspace can show the old project's run. If the controller never clears, the loop runs forever. A single transient network error stops progress reporting while the render continues.
- **Suggested fix:** Replace both loops with a cancellable poller (effect keyed on runId plus an abort signal, like `pollInstall` at lines 226-257). `FilmLifecycle` already refreshes runs every 2 s, so deriving progress from `runs` would make the loops unnecessary.
- **Confidence:** High

#### [F-201] Three overlapping film-run polls, one unconditional and unabortable
- **Source ID:** B3-002
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `components/editor/FilmLifecycle.jsx:44-55`, `hooks/useTimelines.js:100-118`, `components/editor/FilmWorkspace.jsx:503-509`
- **Finding:** `FilmLifecycle` fires two GETs (`/film-runs` and `/planning`) every 2 s for as long as it is mounted, even with no active run. `useTimelines` fetches the full timeline every 1 s for any film timeline regardless of delivery state. `FilmWorkspace` adds its own 500 ms run poll while a render is active. There are no abort signals, no `document.hidden` pause, and no SSE use, although the job SSE stream (`useJobEvents`) already exists.
- **Impact:** A constant background request load (about 3.5 req/s during a render, about 1.5 req/s idle on a film timeline). Each timeline poll runs `adoptStoredTimeline`, which can re-render the editor. Requests are not cancelled on unmount.
- **Suggested fix:** Poll only while a run is `controllerActive` or a delivery is pending. Pass `signal` to `apiFetch`, pause when hidden, and merge into one run feed. Ideally push run and timeline-revision events over SSE.
- **Confidence:** High

#### [F-202] Training config validation ignores the target's advertised numeric ranges and the integer requirement the API enforces
- **Source ID:** B4-001
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/web/src/training/trainingConfig.js:1018-1032` (generic `> 0` loop) vs `crates/sceneworks-core/src/training.rs:4119-4160` (`validate_advertised_numeric_limits`), `:662-677` (`TrainingConfig` u32/i64 fields), `:2018-2022` (limits `rank [4,128]`, `steps [200,6000]`, `batchSize [1,4]`)
- **Finding:** `configValidation` only checks `rank/alpha/steps/resolution/batchSize/gradientAccumulation/saveEvery` for "> 0". It never reads `limits.rank`, `limits.alpha`, `limits.steps` or `limits.batchSize` (only `limits.sampleCount`, `microStepFields`, `resolutions` via the select), and never checks `Number.isInteger` or the u32 ceiling. The API refuses all of these (advertised range check, `u32` deserialization of `100.5`/`1e10`). `seed` is not validated at all, and a blank/garbage seed becomes `null` in the payload for a required `i64`.
- **Impact:** The "Needs input" pill reads ready and Start training enables for `steps=10` on a `[200,6000]` target or `rank=3.5`; the user gets a server 400 with no field outline, which is exactly what epic 10644's validation design was built to prevent. This is the one place the web rule set is looser than the backend while the file header claims it mirrors the API.
- **Suggested fix:** Add one loop over `["rank","alpha","steps","batchSize"]` that reads `selectedTarget.limits[field]` as `[min,max]`, requires `Number.isInteger`, and emits the same message as the API (`invalid_advertised_limit` wording); add integer/range checks for `gradientAccumulation`, `saveEvery`, `seed`. Optionally set `min`/`max` on the `numericInput`s from the same limits (ConfigureJobPanel.jsx:606-614 passes none).
- **Confidence:** High (backend side verified; web side is a straight read of the rule list).

#### [F-203] Dataset editor re-renders every caption card on every keystroke and every SSE tick (no per-card component, no memo, no windowing)
- **Source ID:** B4-002
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `apps/web/src/screens/training/DatasetEditorPanel.jsx:588-754` (inline `visibleAssets.map` building ~170 lines of JSX per card, no `React.memo`); `apps/web/src/screens/TrainingStudio.jsx:372,1665-1672` (`updateCaption` sets `captionDraftById` on the 2.6k-line root); `styles.css:6434` (`.asset-tile-windowed` exists but is not applied to `.training-caption-card`)
- **Finding:** Each card contains a controlled `<textarea>`, a `ReadinessBadge`, `AssetThumbnail`, mask overlay and five buttons, all built inline in the parent's render. Typing one character calls `setCaptionDraftById`, re-rendering TrainingStudio (a `useAppContext()` consumer, so it also re-renders on every job/SSE update) and then every card. Dataset sizes of hundreds of images are the normal case (Parquet import, catalog materialize up to 500 rows, character sets). The project already solved the paint side for asset grids with `content-visibility: auto`, but not here and not for the React reconciliation side.
- **Impact:** Visible typing lag in large datasets, and a full-grid reconcile on every training progress tick while the editor is open.
- **Suggested fix:** Extract `<CaptionCard>` wrapped in `React.memo`, passing only that item's `draft`, `readinessEntry`, `savedItem`, `maskStatus` and stable (ref-bridged) callbacks; add a `content-visibility: auto; contain-intrinsic-size` class to `.training-caption-card`; keep caption text in card-local state flushed on blur/debounce if keystroke latency remains.
- **Confidence:** Medium (re-render cost inferred from structure; no profile taken).

#### [F-204] Dataset import uploads files strictly one by one with no partial-failure recovery, no progress, and no project-switch guard; the logic is duplicated for references
- **Source ID:** B4-003
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/web/src/screens/TrainingStudio.jsx:1879-1957` (`handleImport`), `:1692-1716` (`importItemReferences`, same loop copy-pasted), `:1914` (per-file `setUploadedDatasetAssets`), `components/DatasetAddDialog.jsx:117-127` (drop handler)
- **Finding:** (a) `for (const file of imageFiles) await uploadDatasetItem(file)` aborts on the first rejected upload; images already uploaded are in `uploadedDatasetAssets` but never added to `selectedAssetIds` (the `setSelectedAssetIds` happens only after the loop), so a failure on file 80 of 100 leaves 79 orphan server-side uploads and a one-line error with no count. (b) No progress indicator, no concurrency, and `.txt`/`.json` sidecars are read sequentially with `await file.text()` with no size cap. (c) The loop keeps running after the project-switch reset effect (TrainingStudio.jsx:1142-1165 clears the draft), so asset ids from the old project are then written into the new project's `selectedAssetIds`/caption drafts. (d) Each success does `setUploadedDatasetAssets([...filter...])`, O(n^2) over a large drop. (e) The reference variant duplicates the same upload+flag+merge block.
- **Impact:** Large drops are slow and all-or-nothing in practice; a mid-batch failure silently strands uploads; a project switch during import corrupts the new draft.
- **Suggested fix:** Extract `uploadDatasetFiles(files, {signal, onProgress})` used by both paths: bounded concurrency (3-4), collect per-file results, select the successes and report "N of M failed: names" instead of throwing; accumulate assets in a local array and set state once; abort via a controller bumped in the project-reset effect; cap sidecar size before `file.text()`.
- **Confidence:** High on (a)(d)(e); Medium on (c) (depends on switch-guard timing).

#### [F-205] `CatalogCuration` is not keyed by catalog, so a catalog switch mid-open lets the old catalog's fetches write into the new one's state
- **Source ID:** B4-004
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/web/src/screens/DatasetCatalogsScreen.jsx:1224` (`<CatalogCuration catalogId={selected.id} .../>` with no `key`), `:291-307` (reset effect clears only some state), `:365-386` (`openCuration` has no abort or generation guard), `:312` (`run` closes over `catalogId`), `:253-278`
- **Finding:** The reset effect on `catalogId` aborts only `abortRef`/`materializeAbortRef`, and resets `open/items/selectedRecord/materialization`, but not `cursor`, `total`, `facets`, `savedViews`, `projects`, `reviews`, `busy`. `openCuration` awaits saved-views/projects with no signal, then calls `run` from the closure of the catalog that was selected when "Browse catalog" was clicked. If the user switches catalogs during that await, `run` creates a fresh controller (the earlier abort already happened) and loads catalog A's page into the component now displaying catalog B; "Load next page" can also pass A's `cursor` with B's id. A run aborted by the effect leaves `busy` true (`finally` only clears it for non-aborted runs).
- **Impact:** Wrong-catalog thumbnails/saved views/materialize target after a quick switch; stale "Saved views (n)" and facets from the previous catalog until the next run.
- **Suggested fix:** Render `<CatalogCuration key={selected.id} .../>` so all state resets by construction (then most of the reset effect can go), and give `openCuration` the same abort/generation guard `materializeResults` already has.
- **Confidence:** High on the missing key and state not being reset; Medium on hitting the race in practice.

#### [F-206] Pose library create panel bypasses the shared library module its keypoint sibling uses, and maps staged uploads to files by array index
- **Source ID:** B4-005
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `apps/web/src/screens/PoseLibraryScreen.jsx:157-185` (raw `apiFetch("/api/v1/poses/sources")` + `URL.createObjectURL(images[index])`), `:230-258`, `:334`, `:467`, `:539-545`, `:566-593`; contrast `apps/web/src/keypointLibrary.js:51-90` (`stageKeypointSource`, `saveKeypointPreset`, ...)
- **Finding:** `keypointLibrary.js` owns every endpoint and the staging helper for the Key Point screen; `poseLibrary.js` has only constants/record converters, so `PoseLibraryScreen` hand-builds URLs for stage/save/list/status/tags/delete/purge/preview inline. The staging code also posts all images in one multipart request and then pairs `result.sources[index]` with `images[index]` for `displayName` and the preview `objectUrl`; if the server drops or reorders any file (non-decodable image, dedupe), previews and names attach to the wrong source. The two screens also carry near-identical stage -> job -> watch-`jobs`-for-terminal -> review state machines (`terminalStatuses` effect at PoseLibraryScreen.jsx:257-270 vs KeyPointLibraryScreen's capture panel) that should share a hook.
- **Impact:** Endpoint changes need edits in screen code; wrong preview/name pairing on partial server acceptance; two copies of the detect-job watcher to keep in sync (neither times out, so a missed SSE event leaves "detecting" forever).
- **Suggested fix:** Move `stagePoseSources`, `savePoses`, `loadPoseAssets`, `setPoseStatus/Tags`, `deletePose/purgePose` into `poseLibrary.js` (mirroring keypointLibrary), have the server echo the original filename/index so the client matches by key, and factor `useDetectJob(jobId, jobs)`.
- **Confidence:** High on the bypass; Medium on mismatch (depends on the server's response ordering guarantees).

#### [F-207] TrainingStudio is still a 2.6k-line, effect-driven state machine; the F-036 "session" bags are fresh object literals each render
- **Source ID:** B4-007
- **Category:** readability
- **Severity:** Medium
- **Location:** `apps/web/src/screens/TrainingStudio.jsx:372-2607` (single component), `:2487-2530` (inline `datasetSession={{ ...~25 keys }}` / `captionSession` / `config` literals), effects at `:640,767,819,1013,1052,1103,1131,1142,1153,1159,1213,1232,1245,1290,1419,1502,1510,1551`; 7 `eslint-disable react-hooks/exhaustive-deps`
- **Finding:** Previously F-036 (2026-07-12, then 1,567 lines) asked to group the prop bags. The bags were grouped (DatasetEditorPanel now takes `datasetSession/captionSession/...`), but the root grew to 2,607 lines and the grouping is cosmetic: each bag is rebuilt inline every render, so no child can ever skip a render, and the many "mirror state into state" effects (config reseed `:1245-1288`, trigger-follows-captions `:1290`, prompts-follow-trigger, gpu clamp, dataset-reset `:1142`) are derived state kept in sync by effects, which is the shape that produced F-005.
- **Impact:** Every dataset-editor or config change is a three-file edit in one very large component; memoization of children (B4-002) is blocked by the unstable bags.
- **Suggested fix:** Seams: (1) `useDatasetDraft()` owning `activeDataset/draftName/selectedAssetIds/captionDraftById/referenceDraftById/holdoutById` plus `persistDataset/handleImport/openDataset` (about 900 lines); (2) `useTrainingConfigDraft()` owning `configDraft/customizedConfigFields/configBasisRef` and the reseed/follow effects, using `configReseedDecision` as a reducer; (3) move `TrainingLiveProgress/TrainedLoraOfferCard/trainingSampleGroups` (`:201-366`) to `screens/training/`. Return stable, memoized session objects from the hooks.
- **Confidence:** High.

#### [F-208] ModelManagerScreen renders the whole catalog through ~750 lines of inline render closures
- **Source ID:** B5-001
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `apps/web/src/screens/ModelManagerScreen.jsx:1313-1820` (renderModelCard), `:1866-2110` (renderLoraRow), `:604-2535` (whole component), `:625` (useAppContext)
- **Finding:** `renderModelCard` (~510 lines) and `renderLoraRow` (~245 lines) are closures inside the 1,900-line `ModelManagerScreen`. They recompute per-model derived state on every render: job filters, tier logic, license gate, availability badges. They cannot be `React.memo`'d. The screen reads the merged `useAppContext()`, so it re-renders on every live context tick (job progress SSE, worker heartbeat) and re-renders every model and LoRA card each time.
- **Impact:** While a download or convert job streams progress, the full catalog (100+ cards, each running several `jobs.filter` passes) is rebuilt on every tick. This is the file the brief flags as 2.5k lines.
- **Suggested fix:** Extract `<ModelCard>`, `<LoraRow>`, `<LoraImportPanel>` and `<LinkedLibraryStatus>` into `screens/modelManager/*.jsx`, wrapped in `React.memo`. Pass each card only its own `downloadJobs`/`convertJobs` slice, computed once by indexing `jobs` by modelId in a single `useMemo`. Move the delete/relink/rescan handlers to a `useModelManagerActions` hook. The `ModelTierDownloadPanel` extraction at line 314 shows the seam already works.
- **Confidence:** High

#### [F-209] Per-model license acknowledgment lives only in origin-scoped localStorage, and the code itself says it is wiped on desktop relaunch
- **Source ID:** B5-002
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `apps/web/src/licenseAcknowledgment.js` (`readLicenseAck`/`writeLicenseAck`), `apps/web/src/screens/ModelManagerScreen.jsx:734-772,1363-1366`, `apps/web/src/screens/SetupWizard.jsx:114-116`, `apps/web/src/hooks/useModelsAndLoras.js:334,355,411`
- **Finding:** The acceptance gate is a localStorage boolean, and the client then asserts `licenseAcknowledged: true` to the API. The ModelManager comment admits localStorage "does not survive" a desktop relaunch. `SettingsScreen.jsx:248-251` says the same thing about why `ui-preferences.json` exists: the shell origin is `127.0.0.1:<port>` and changes each launch. Theme, accent and default quality were moved to a server-side write-through; license acks were not.
- **Impact:** Desktop users re-accept each gated license on every launch (the F-sc-17227 comment describes this as "unrecoverable through the UI"). The backend only trusts a client boolean, so there is no durable record of acceptance on the server.
- **Suggested fix:** Persist acks through the same `/api/v1/ui-preferences` merge path, or add a small `license-acks` route. Keep localStorage as the instant-paint cache only. Have the server record the acknowledgment with its timestamp, and key it by the id and the `licenseAcknowledgmentSource`.
- **Confidence:** Medium. The origin-wipe behaviour is taken from the repo's own comments; I did not run the desktop shell.

#### [F-210] Pose capture and Key Point capture are near-identical state machines, and both bypass the shared job-creation path
- **Source ID:** B5-003
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/web/src/screens/PoseLibraryScreen.jsx:122-345` and `apps/web/src/screens/KeyPointLibraryScreen.jsx:127-262,640-660`
- **Finding:** Both panels implement the same flow: stage a file or asset, POST `/api/v1/jobs` by hand, watch the SSE-fed `jobs` list for a terminal status, review, then save or discard behind the same `appConfirm`. Both carry the same `objectUrl` lifecycle and the same `hasUnsavedWork` logic. They call `apiFetch("/api/v1/jobs", …)` directly (`PoseLibraryScreen.jsx:232`, `KeyPointLibraryScreen.jsx:645`) instead of the `makeCreateJob` or `CREATE_JOB_DEFINITIONS` factory that Image, Video and Vector use. `PoseLibraryScreen.jsx:559-595` also reimplements status, tags, delete and purge against an arbitrary `projectId`.
- **Impact:** Any fix has to be made twice. Two bugs remain unfixed in both: a job pruned from `jobs` before the watcher sees it leaves the panel stuck in "detecting" or "extracting" forever, and a request that fails after the optimistic stage has no retry. Hand-posted jobs also skip the `setJobs` optimistic insert and the shared error banner.
- **Suggested fix:** Add `useStagedJobCapture({jobType, stage, buildPayload})` to `hooks/`, covering the sources, objectUrl revocation, job watch (with a timeout or `jobs`-miss fallback) and discard-confirm. Register `pose_detect` and `kps_extract` in `CREATE_JOB_DEFINITIONS`. Make the asset mutators in `App` accept `projectId` instead of duplicating them in Pose.
- **Confidence:** High on the duplication, Medium on the stuck-state edge.

#### [F-211] LogsScreen: filter change races the in-flight poll and snapshot, so rows from the old filter can be appended to the new view
- **Source ID:** B5-004
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/web/src/screens/LogsScreen.jsx:168-188` (loadSnapshot), `:191-216` (poll), `:244-267` (scheduler effect)
- **Finding:** `poll()` has no cancellation. When `source` or `level` changes, the scheduler effect's cleanup only sets `closed = true`. A `poll()` already awaiting `fetchLogs` with the old filter still reaches `setEntries(prev.concat(rows))` and `lastSeqRef.current = …`. `loadSnapshot` has no staleness guard either, so two quick filter clicks can resolve out of order. If the snapshot fails, `lastSeqRef` keeps the old filter's seq while `entries` still holds the old rows.
- **Impact:** A user switching Source or Level can briefly see, and then keep, entries from the previous filter. The `seq` bookkeeping then skips or duplicates rows until the next filter change.
- **Suggested fix:** Use a request-generation ref, or an `AbortController` passed to `fetchLogs`. Have `loadSnapshot` and `poll` bail out if the generation changed. Clear `entries` and `lastSeqRef` synchronously when the filter changes.
- **Confidence:** Medium. The race window is small and depends on timing.

#### [F-212] PresetManagerScreen defines its own `slugify` that has drifted from `slugifyPresetId`
- **Source ID:** B5-005
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/web/src/screens/PresetManagerScreen.jsx:209-215,442,543,597` versus `apps/web/src/presetUtils.js:740-746` (`slugifyPresetId`, documented as mirroring the backend `slugify_preset_id`)
- **Finding:** Both use the same regex, but the screen's version strips only leading and trailing `_` (`/^_+|_+$/g`). The shared one strips `[_-]`. The shared one is the helper documented as mirroring `apps/rust-api/src/recipe_presets.rs:1075`. The screen's helper is a separate copy used to build the preset `id` sent to the API. A name like `-Hero` yields `-hero` in the editor and `hero` in the studios' save-as-preset. The empty-slug case (`!!!`) also yields `""` here, where the backend falls back to `"preset"`.
- **Impact:** The same preset name produces different ids depending on which screen created it. A leading-dash id may be rejected or duplicated server-side.
- **Suggested fix:** Delete the local `slugify` and import `slugifyPresetId`. Add a parity test against the backend cases.
- **Confidence:** High

#### [F-213] StatsCharts picks chart colours from the OS colour scheme, not the app theme
- **Source ID:** B5-006
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `apps/web/src/screens/StatsScreen.jsx:192-197`
- **Finding:** `isDark` is read from `window.matchMedia("(prefers-color-scheme: dark)")`. The app applies its own theme as `data-theme` on `<html>` (`App.jsx:1619`) and offers an explicit toggle (`useAppContext().theme`). The grid colour is a hex literal, and the value is computed on every render with no subscription.
- **Impact:** If the user's app theme differs from the OS (Settings, then Light on a dark OS), the grid lines and ticks render wrong or near-invisible. They also do not update when the OS scheme flips mid-session.
- **Suggested fix:** Read `theme` from `useAppStatic()`, or use CSS variables (`stroke="var(--border)"`) as the tooltip style already does, and drop the matchMedia read.
- **Confidence:** High

#### [F-214] SimpleSheet and SimplePreview reimplement Modal without focus trap, focus restore or (Preview) Escape
- **Source ID:** B6-001
- **Category:** architecture-drift
- **Severity:** Medium
- **Location:** `apps/web/src/simple/SimpleSheet.jsx:12-50`, `apps/web/src/simple/SimplePreview.jsx:58`, `apps/web/src/components/Modal.jsx:36-110`
- **Finding:** `Modal.jsx` is the shared primitive ("so every overlay behaves consistently"): Tab trap, Escape, restores focus to the opener on unmount. `SimpleSheet` rolls its own role=dialog and aria-modal, but only focuses the panel on mount and handles Escape. It has no Tab trap and never restores focus. `SimplePreview` is a bare `role="dialog"` div with no aria-modal, no Escape handler, no focus move and no trap. It is a full-screen overlay, so a keyboard user can Tab into the shell behind it.
- **Impact:** Keyboard and screen-reader users in the Simple UI lose their place on close, or tab into hidden content. The advanced UI does not have this problem.
- **Suggested fix:** Build SimpleSheet and SimplePreview on `Modal`, or extract a `useDialogFocus(ref, onClose)` hook from Modal and use it in all three. Add a keyboard test to SimpleShell.test.jsx.
- **Confidence:** High

#### [F-215] The "fetch the prompt guide and read it as text" step is implemented three times with raw fetch
- **Source ID:** B6-002
- **Category:** redundant
- **Severity:** Medium
- **Location:** `apps/web/src/components/RefinePromptControl.jsx:100-113`, `apps/web/src/simple/useSimpleRefine.js:52-65`, `apps/web/src/components/PromptGuideModal.jsx:16-30`
- **Finding:** Three near-identical best-effort `fetch(guidePath)` blocks (AbortError handling, `response.ok`, `.text()`). The first two are line for line the same. The whole refine flow in `useSimpleRefine` also mirrors `RefinePromptControl.handleRefine`: abort the previous run, set busy, fetch the guide, call refinePrompt, check `signal.aborted`. The copies differ on error classification (`err?.name === "AbortError"` here, `isAbortError` from api.js elsewhere).
- **Impact:** Simple and advanced refine can drift. The guide fetch bypasses the `api.js` layer.
- **Suggested fix:** Add `fetchPromptGuide(path, signal)` to `api.js` or a small `promptGuides.js`. Have `RefinePromptControl` consume the Simple hook, or share one `useRefinePrompt` hook. Use `isAbortError` everywhere (6 hand-written `err?.name === "AbortError"` checks remain).
- **Confidence:** High

#### [F-216] LayersPanel.jsx is dead (no importer); only its test keeps it alive
- **Source ID:** B6-004
- **Category:** dead-code
- **Severity:** Medium
- **Location:** `apps/web/src/components/LayersPanel.jsx:1-240`, `apps/web/src/components/LayersPanel.test.jsx`
- **Finding:** A scan of every non-test component and simple file for importers found exactly one with zero: this file. The only reference in src is a comment at `screens/ImageEditor.jsx:1307` ("mirrors LayersPanel"). ImageEditor (4.5k lines) evidently inlines its own layers UI.
- **Impact:** 240 lines plus a test maintained for nothing. The comment in ImageEditor points a reader at a component that is not used.
- **Suggested fix:** Either wire it into ImageEditor, which also shrinks that 4.5k-line file (a split seam), or delete the component and its test, and fix the comment.
- **Confidence:** High

#### [F-217] Dev `web` compose image is missing build inputs that vite.config.js and the app require
- **Source ID:** S1-001
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `docker/web.Dockerfile:12-17`, `apps/web/vite-plugin-editor-guides.js:8-12`, `apps/web/src/data/bundledLicenses.js:9-12`, `apps/web/vite.config.js:26-27`
- **Finding:** web.Dockerfile copies `scripts`, `config`, `packages/schemas` to `/` but not `docs/film-script-writing.md`, `docs/film-editor.md` (read by editor-guides plugin via `../../docs/...`) nor `apps/desktop/licenses` (imported by bundledLicenses.js, also the `server.fs.allow` root). rust.Dockerfile's web-builder stage copies both explicitly (lines 20, 29), so the two web build paths have drifted.
- **Impact:** `docker compose up` web service (Vite dev) would fail on `virtual:editor-guides` (EditorScreen) and the licenses screen with ENOENT/resolve errors. `scripts/check-docker-web-runtime.mjs:19-29` only fetches `/` (index.html), so CI cannot see it.
- **Suggested fix:** COPY the two docs files and `apps/desktop/licenses` into the dev image at the same relative locations, and have check-docker-web-runtime.mjs fetch `/src/components/editor/EditorGuides.jsx` and `/src/data/bundledLicenses.js` (transform requests) to prove the graph resolves.
- **Confidence:** Medium (derived from Dockerfile vs. import paths; not run)

#### [F-218] 192 MB of raw `.rgb` calibration frames tracked in git
- **Source ID:** S1-002
- **Category:** redundant
- **Severity:** Medium
- **Location:** `docs/calibration/sc-19721/*.rgb` (30 files, 94.4 MB), `docs/calibration/sc-18353/*.rgb` (26 files, 81.8 MB), `docs/calibration/sc-15833/*.rgb` (5 files, 15.7 MB)
- **Finding:** 61 uncompressed 1024x1024x3 (3,145,728 B) frames make up 191.9 MB of the 196.7 MB under docs/calibration (668 tracked files). .gitattributes marks `*.rgb binary` but there is no LFS and no ignore. Only the hash-addressed names are referenced (memory_calibration.rs, memory-calibration-harness tests, sdxl evidence JSON), not the bytes.
- **Impact:** every clone/CI checkout pays for this (.git is 91 MB packed here even though raw tree is ~230 MB for tracked files); each re-capture adds another 3 MB per frame permanently.
- **Suggested fix:** if only the digests are load-bearing, store the sha256 in the evidence JSON and drop the frames (or move to release assets / LFS); at minimum compress (png/zstd) and gitignore `docs/calibration/**/*.rgb` going forward so new captures do not land by default.
- **Confidence:** High on sizes; Medium on whether any test reads the bytes (hashing tests may need them)

#### [F-219] Hard-coded personal absolute paths and unreadable minified source in tracked CI scripts
- **Source ID:** S1-003
- **Category:** readability
- **Severity:** Medium
- **Location:** `scripts/ci/moss-first-step-probe.mjs:11-22` (112 lines >300 chars of 89 KB), `scripts/ci/moss-first-step-probe.test.mjs` (61 long lines), `scripts/ci/moss-owned-cache-maintenance.mjs` (25), `scripts/ci/moss-upstream-quality.mjs`, `scripts/ci/moss-upstream-reference-common.mjs`, `scripts/lib/moss-edit-terminal-actions.mjs`
- **Finding:** 38 occurrences of `/Users/michael/...` and `/Volumes/Models/...` (e.g. `PROBE_STAGED`, `WARM`, `CARGO_SEED` at moss-first-step-probe.mjs:11-22) are baked into scripts, and moss-first-step-probe.mjs is written in a compressed no-whitespace style (line 41 is a single ~900-char `run()`), which also hides yet another copy of a spawn-and-tee-to-log helper.
- **Impact:** scripts only work on one machine, cannot be reviewed or linted meaningfully, and leak a developer's home path/layout into a public repo.
- **Suggested fix:** move machine paths into env/config (one `scripts/lib/runner-paths.mjs`), run the files through the formatter used elsewhere, and reuse `scripts/lib` process helpers.
- **Confidence:** High

#### [F-220] Same helpers re-implemented per script instead of living in scripts/lib
- **Source ID:** S1-004
- **Category:** redundant
- **Severity:** Medium
- **Location:** sha256: `scripts/package-sdxl-cuda-endpoints.mjs:18`, `yue2-acceptance.mjs:961`, `extract-memory-anchors.mjs:160`, `generate-memory-matrix.mjs:786`, `scan-inference-provenance.mjs:166`, `check-runpod-staged-image.mjs:14`, `starvector-production-closure.mjs:104`, `inference-closure-digest.mjs:133`, `kv-baseline-harness.mjs:608`, `rearchitecture/capture-catalog-baseline.mjs:65` (98 `createHash("sha256")` calls total; only 6 scripts use `scripts/lib/file-sha256.mjs`). Canonical JSON: `memory-calibration-harness.mjs:400`, `kv-baseline-harness.mjs:581`, `starvector-production-closure.mjs:96` (stableJson), `epic-20738-terminal-cuda-harness.mjs:266` (canonicalize). JSONC read: `audit-hf-cache-liveness.mjs:281`, `check-no-nc-weights.mjs:260`, `check-scaffold.mjs:314`, `check-tier-integrity.mjs:148`, `rearchitecture/capture-catalog-baseline.mjs:12` (a private copy of `stripJsoncComments` despite `scripts/lib/jsonc.mjs`). git/run/fail wrappers: `git()` x4 (inference-closure-digest:129, release:248, rung4-contract-prerequisites:777, epic-20738...:1046), `run()` x6, `fail()` x11.
- **Finding:** these differ subtly (kv-baseline's canonicalJson is indented and numeric-normalised, stableJson sorts by UTF-8 order, canonicalize uses default `sort()` UTF-16 order, memory-calibration appends "\n"), so they are near-twins that can disagree on non-ASCII keys when the same digest is computed by two scripts.
- **Impact:** digest/receipt code is the integrity layer for the evidence bundles; silent divergence in key ordering is a latent hash mismatch, and every fix has to be repeated.
- **Suggested fix:** add `scripts/lib/{sha256,canonical-json,exec}.mjs`, pick one documented ordering, migrate callers; add a lint-style test banning new local `function sha256`.
- **Confidence:** High on duplication; Medium on any current mismatch


### Low findings — web/tooling

#### [F-221] YuE2 licence link is rendered from the catalog without the URL allow-list used everywhere else
- **Source ID:** B1-010
- **Category:** security
- **Severity:** Low
- **Location:** `apps/web/src/screens/audioYue2Lab.jsx:265-269`, `apps/web/src/yue2Policy.js:18`
- **Finding:** `identity.licenseUrl` (any string from `model.licenseUrl`) goes straight to `<a href>`. Every sibling site (`LicenseGateNotice`, `ModelLicenseSummary`, `LicensesScreen`, `ConfigureJobPanel`) wraps the value in `safeExternalUrl()` from `urls.js`.
- **Impact:** A `javascript:` or `data:` URL in a manifest, or one supplied by a future remote or user-provided catalog, would run on click. The Tauri CSP (`script-src 'self'`) blocks inline script, but the Docker/browser build has no equivalent guard. The risk is low because the source is the built-in manifest.
- **Suggested fix:** `href={safeExternalUrl(identity.licenseUrl)}` and omit the link if it is undefined. Do the same in `yue2Policy.js` so the value is normalized at the source.
- **Confidence:** High

#### [F-222] Production Docker builds fetch unpinned toolchains and floating base images
- **Source ID:** S1-005
- **Category:** security
- **Severity:** Low
- **Location:** `docker/rust.Dockerfile:217` (`curl ... https://sh.rustup.rs | sh -s -- -y`), `docker/rust.Dockerfile:16,36,136,172,315` (`node:22-bookworm-slim`, `rust:1-bookworm`, `debian:bookworm-slim`, `ubuntu:24.04`), `docker/web.Dockerfile:1` (`node:22-alpine`); contrast `nvidia/cuda:12.9.1-...` which is version-tagged
- **Finding:** rustup-init is piped to sh with no checksum and installs a toolchain before rust-toolchain.toml pins take over; base images use floating tags, none by digest, while GitHub Actions are carefully SHA-pinned.
- **Impact:** release image content is not reproducible and a compromised tag/rustup script flows straight into published RunPod images.
- **Suggested fix:** pin images by `@sha256:` (Dependabot/renovate can bump), install rustup via a version-pinned rustup-init with a verified sha256.
- **Confidence:** High

#### [F-223] workflow_dispatch inputs interpolated directly into `run:` shell
- **Source ID:** S1-006
- **Category:** security
- **Severity:** Low
- **Location:** `.github/workflows/flux3-quant-matrix.yml:79`, `.github/workflows/flux3-quant-matrix.yml:162`
- **Finding:** `${{ inputs.runner }}` is expanded inside the shell script text; other workflows consistently route inputs through `env:` (e.g. macos-mlx.yml:642-665).
- **Impact:** only users with write access can dispatch, so this is injection-by-maintainer, but it is on self-hosted runners and inconsistent with the rest.
- **Suggested fix:** `env: EXPECTED_RUNNER: ${{ inputs.runner }}` and compare `"$EXPECTED_RUNNER"`.
- **Confidence:** High

#### [F-224] `fetchYue2TranscriptionFile` re-implements `apiFetch` error handling and bypasses the 401 gate
- **Source ID:** B1-006
- **Category:** redundant
- **Severity:** Low
- **Location:** `apps/web/src/api/yue2.js:119-138` (vs `api.js:94-142`); other raw `fetch` of API URLs: `audioWaveform.js:142`, `screens/KeyPointLibraryScreen.jsx:232`
- **Finding:** This function builds its own `Headers`, calls raw `fetch`, and hand-assembles an `ApiError` (without `authRequired`). It never calls `unauthorizedHandler`, so a stale token on this path never re-prompts the access gate, which is the sc-15105 failure mode `apiFetch` exists to prevent.
- **Impact:** A 401 on a transcription export download shows a generic error instead of locking the session. The error shape also drifts from `apiFetch`.
- **Suggested fix:** Give `apiFetch` a `responseType: "blob"` (or `parse: (response) => …`) option and use it here.
- **Confidence:** High

#### [F-225] Dead modules and exports with zero importers
- **Source ID:** B1-008
- **Category:** dead-code
- **Severity:** Low
- **Location:** `apps/web/src/useAudioPreviewGain.js:1-45` (no importer anywhere), `audioPreviewGain.js:1-94` plus `audioPreviewGain.test.js` (imported only by that hook), `timeline.js:1-7` (`aspectOptions`, `transitionOptions`, `speedPresets`; one occurrence each in the tree), `appHydration.js` `GLOBAL_HYDRATION_DOMAINS`, `yue2Lab.js` `YUE2_KINDS`, `audioWaveform.js` `resetAudioPeakCache`; `components/LayersPanel.jsx:1-240` (only a test imports it; `ImageEditor.jsx:1307` just mentions it in a comment)
- **Finding:** The audio preview gain graph (hook, `AudioContext` lease code and test, about 230 lines, last touched in "Fix boosted audio preview gain") is not wired into any component. Likewise the three timeline option tables and the listed constants. Greps were run across `src/` for each name, including `.jsx`.
- **Impact:** Dead code that still gets tested and maintained. The gain code creates an `AudioContext` and would be a leak risk if someone wired it up without noticing it is untested in the app.
- **Suggested fix:** Delete the hook, `audioPreviewGain.js` and its test (or wire the hook in if the boost feature is still wanted). Delete the unused constants. Remove or integrate `LayersPanel.jsx`.
- **Confidence:** High

#### [F-226] Static-only consumers use `useAppContext()` and re-render on every SSE tick
- **Source ID:** B1-009
- **Category:** efficiency
- **Severity:** Low
- **Location:** `apps/web/src/components/ControlOverlayPicker.jsx:12` (`token, activeProject`), `simple/SimplePreview.jsx:31` (`audioModels`), `simple/SimpleLoraSheet.jsx:31` (`loras, createLoraImportJob, activeProject`), `context/AppContext.js:62-90`
- **Finding:** sc-8855 split the context so cold readers can use `useAppStatic()`, but 30 files still call `useAppContext()` against 15 that use `useAppStatic()`. The three above read only fields that live in `appStaticValue` (`token`, `activeProject`, `audioModels`, `loras`, `createLoraImportJob` all appear in the static literal at `App.jsx:3655-3891`). `useAppContext` subscribes to the live context and also spreads about 300 keys into a fresh object whenever either value changes (`useMemo` at `AppContext.js:62`).
- **Impact:** The point of the split is lost for these consumers. Each job or worker tick re-renders them and re-runs the merge.
- **Suggested fix:** Switch those three to `useAppStatic()`. Audit the other 27 for static-only destructures, and add a lint or test that flags a `useAppContext` consumer whose destructured keys are all static.
- **Confidence:** High

#### [F-227] SSE connect surfaces an error from a torn-down connection and cannot abort its ticket request
- **Source ID:** B1-011
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `apps/web/src/hooks/useJobEvents.js:429-440`
- **Finding:** In `connect()`'s catch, `setError(err.message)` runs before the `if (!closed)` check, and the ticket `apiFetch` has no `AbortSignal`. When the effect re-runs (token change, gate re-lock) while a ticket POST is in flight, the old closure can still push its failure notice after cleanup. The `closed` check afterwards (`if (closed) return;`) covers the success path only.
- **Impact:** A spurious error notice (and a reconnect timer that is correctly skipped) during login/lock transitions. This is cosmetic but visible in the same flow F-007 fixed.
- **Suggested fix:** Check `closed` first in the catch, and pass an `AbortController` signal that cleanup aborts.
- **Confidence:** Medium

#### [F-228] Small helpers reimplemented several times: backoff delay and byte formatters
- **Source ID:** B1-012
- **Category:** redundant
- **Severity:** Low
- **Location:** backoff `Math.min(30000, 1000 * 2 ** n)` at `hooks/useJobEvents.js:435,466`, `hooks/useAccessGate.js:197,253,414`, plus variants at `uiPreferences.js:74`, `workflowEmbedPreferences.js:91`, `screens/LogsScreen.jsx:257`; byte formatters `formatting.js:52` (`formatBytes`, binary, "GiB"), `screens/ModelManagerScreen.jsx:121` (`formatTierSize`, binary), `components/RefinePromptControl.jsx:5` (`formatGb`, decimal "GB"), `screens/TrainingStudio.jsx:706` (inline `/1e9`)
- **Finding:** The same exponential-backoff expression is inlined five times in the two gate/stream hooks (and three more variants elsewhere). Size formatting exists in one canonical module and three ad hoc copies that disagree on units (GiB versus GB).
- **Impact:** Users see inconsistent sizes for the same model (a 16 GiB download labeled "17.2 GB" in one panel). A backoff policy change needs eight edits.
- **Suggested fix:** Add `backoffDelay(attempt, {baseMs, maxMs})` to `pollJob.js` or a new `retry.js`. Replace the ad hoc formatters with `formatBytes` and keep one unit convention.
- **Confidence:** High

#### [F-229] Debounced durable studio-settings write is cancelled on unmount
- **Source ID:** B1-013
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `apps/web/src/hooks/useStudioSettings.js:282-283`
- **Finding:** The effect writes the local cache immediately but defers `persistToServer` by 400 ms, and its cleanup is `clearTimeout`. If the studio unmounts, or `serialized` changes again, within the window, the pending durable PUT for the earlier value is dropped. On unmount no later effect run exists to issue it.
- **Impact:** The last edit made within 400 ms of leaving a non-keep-alive studio (or closing the app) is saved to `localStorage` but never reaches the server copy that sc-15425 made durable across relaunch.
- **Suggested fix:** Flush on cleanup when the effect is unmounting (call `persistToServer` instead of only `clearTimeout`), or add a `pagehide`/`visibilitychange` flush.
- **Confidence:** Medium

#### [F-230] Test scaffolding re-declared in many files despite shared helpers
- **Source ID:** B1-015
- **Category:** redundant
- **Severity:** Low
- **Location:** `apps/web/src/hooks/appGateHooks.test.jsx:91-117` (own `FakeEventSource`, `deferred`, `settle`) vs `main.testSupport.jsx:174-213` and `testUtils/dom.js`; local `settle` in 15 test files, `deferred` in 14, `mount`/`render` helpers in 14
- **Finding:** `main.testSupport.jsx` and `testUtils/dom.js` exist as the shared homes (sc-8937, sc-8943), but many files still define their own `settle`, `deferred` and `FakeEventSource` copies. The two `FakeEventSource` versions already differ (`close()` records `closed` in one, is a no-op in the other).
- **Impact:** Test-technique fixes (for example flushing lazy imports in `settle`, as `main.testSupport.jsx:200-215` does) do not reach the local copies, which risks flaky tests.
- **Suggested fix:** Move `deferred`, `settle` and a `FakeEventSource` that records `closed` into `testUtils/`, then codemod the local copies.
- **Confidence:** High

#### [F-231] `useStableImageEditorToolPanelScope` freezes the set of function-valued keys at first render
- **Source ID:** B2-004
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `apps/web/src/screens/imageEditor/ImageEditorToolPanel.jsx:22-45`
- **Finding:** `stableFunctions.current` is built once from the keys that are functions on first render. Later, `typeof value === "function" ? stableFunctions.current[key] : value` returns `undefined` for a key that was non-function at mount but becomes a function later. The reverse case yields a trampoline that throws `TypeError` when the key later becomes null/undefined. Context-supplied callbacks such as `createModelDownloadJob`/`createLoraDownloadJob`, `importAsset` and `tensorAssets` are plausible candidates in partial providers, and tests already pass partial contexts.
- **Impact:** A latent crash that appears only for optional callbacks that arrive after mount (late capability/catalog hydration).
- **Suggested fix:** Build trampolines lazily on first sight of each key (cache per key), and make the trampoline tolerate a missing target. Better still, remove the proxy by passing a stable `actions` object (see B2-003).
- **Confidence:** Medium

#### [F-232] Crop leaks decoded object URLs on partial failure and works from a stale `working` snapshot
- **Source ID:** B2-007
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `apps/web/src/screens/ImageEditor.jsx:2527-2569`
- **Finding:** `Promise.all(working.layers.map(... blobToImage ...))` rejects on the first failure while sibling layers have already created object URLs that are never revoked (the `catch` only sets status). Also, `working` is the closure value from before the awaits, but the commit reads `workingRef.current.layers` for revocation. A layer added or replaced during the encode window is cropped inconsistently (document dims are updated, the new layer is not cropped).
- **Impact:** A small blob-URL and decoded-bitmap leak on failure; a rare inconsistent document after concurrent edits.
- **Suggested fix:** Use `Promise.allSettled`, revoke fulfilled results on any rejection, and bail out (revoke) if `workingRef.current.layers` changed identity between the start and the commit.
- **Confidence:** Medium

#### [F-233] `openAsset` has no abort/ordering guard, and canvases are never released
- **Source ID:** B2-008
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `apps/web/src/screens/ImageEditor.jsx:2198-2222, 2150-2196, 2263, 2427, 2540, 2667, 1045-1100`
- **Finding:** `openAsset` does a raw `fetch(url)` (not `apiFetch`, no `AbortController`) and then `openFromBlob`. Opening asset A then B quickly lets the slower response win, installing the wrong image; after unmount `installWorkingImage` runs and creates an object URL that no cleanup revokes (the unmount effect has already run). `openFromBlob` also reads the whole blob into an ArrayBuffer (L2160) just to sniff alpha. Several scratch `<canvas>` elements (crop, duplicate, blank, composite, cutout) are created per operation and never zeroed (`canvas.width = canvas.height = 0`), which matters because the file itself targets WebKit/WKWebView canvas ceilings (L669, `EDITOR_CANVAS_MAX_AREA`) and Safari holds backing stores until GC.
- **Impact:** Occasionally the wrong image is shown; transient memory pressure on large images on macOS (Tauri/WKWebView).
- **Suggested fix:** Keep a monotonically increasing `openGeneration` ref and drop stale results (revoking their URL). Check `mountedRef` before install. Release scratch canvases after `toBlob`/`drawImage`. Read only the PNG header bytes for the alpha check (`blob.slice(0, N)`).
- **Confidence:** Medium

#### [F-234] Submit and batch preflight validation is duplicated verbatim
- **Source ID:** B2-009
- **Category:** redundant
- **Severity:** Low
- **Location:** `apps/web/src/screens/ImageStudio.jsx:2495-2527` vs `2818-2858`
- **Finding:** Both functions repeat the MOSS-components/edit-adapter error string, the `selectedModelServesMode` message, the decoder-capabilities-pending message, the dimension/hires-fix message and the cfg-interval message. The MOSS message is a long inline literal copied twice.
- **Impact:** A new precondition is easy to add to one path only. F-031 was exactly this class of drift, fixed for the payload builder but not for the guards.
- **Suggested fix:** One `preflightError()` returning a string or null, used by both, with the setter chosen by the caller.
- **Confidence:** High

#### [F-235] Batch enqueue failures are swallowed with no reason
- **Source ID:** B2-010
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `apps/web/src/screens/ImageStudio.jsx:2976-2979` (bare `catch {}`), also `2958`
- **Finding:** Any error from `createImageJob`/`onMagicExpand` only increments `failed` counters. The error message (413, validation, offline, "Auto-generated caption was invalid") is discarded, and a chunk with `n>1` adds `n` failures without saying why.
- **Impact:** A user sees "N failed" in a 500-item batch with no way to learn why. A systematic error (bad model, 422) runs the whole loop before it is visible.
- **Suggested fix:** Capture the first (and last) error message in the batch-run state and show it. Optionally abort after K consecutive identical failures.
- **Confidence:** High

#### [F-236] ImageStudio calls `apiFetch` with a hard-coded endpoint instead of the context/api layer
- **Source ID:** B2-011
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `apps/web/src/screens/ImageStudio.jsx:1, 544-554`; also ImageEditor raw `fetch` at `ImageEditor.jsx:2209, 3167` and `apiFetch(endpoint…)` at `~2955`
- **Finding:** Every other action (`createModelDownloadJob`, `createImageJob`, `imageCaption`, …) comes from `useAppContext`, but `downloadMossAcceleration` builds `/api/v1/models/${model}/conditional-components/…/download` in the screen with the token. It also has no in-flight guard beyond `busy` UI and no unmount guard. ImageEditor fetches asset files with a bare `fetch(assetUrl(...))` (works only because assets are public/unauthenticated; any token-gated asset mode would break it silently), duplicating `createLatestMaskAssetLoader`'s fetch-blob-decode in `imageEditor/maskAssetLoader.js`.
- **Impact:** Endpoint knowledge is spread over screens; the next auth/header change in `api.js` must be hunted down.
- **Suggested fix:** Add `downloadConditionalComponents(model, purpose)` to the models hook/context and a shared `fetchAssetBlob(asset, {signal})` used by `openAsset`, the AI-result loader and the mask loader.
- **Confidence:** Medium

#### [F-237] `consumeEditorLaunch` is dead in production code
- **Source ID:** B2-012
- **Category:** dead-code
- **Severity:** Low
- **Location:** `apps/web/src/screens/ImageEditor.jsx:391-397` (usage only in `ImageEditor.test.jsx:97, 339, 350`)
- **Finding:** The exported function has no non-test caller; `createEditorLaunchGuard` (L398) is what the component uses. A test keeps the older entry point alive.
- **Impact:** Two launch-consumption implementations to keep in sync, with only the unused one covered.
- **Suggested fix:** Delete it and point the tests at `createEditorLaunchGuard().consume`.
- **Confidence:** High (checked by grep for importers).

#### [F-238] `downloadText`/`downloadBlob` copy-pasted twice, plus two more anchor-download implementations
- **Source ID:** B3-004
- **Category:** redundant
- **Severity:** Low
- **Location:** `screens/audioYue2Parts.jsx:35-45`, `screens/audioYue2Recording.jsx:48-57`, `saveExportFile.js:26-29`, `assetActions.js:169`
- **Finding:** The two yue2 files each define the same create-object-URL, hidden anchor, click, remove, revoke sequence. `saveExportFile.js` and `assetActions.js` repeat it again. `downloadText` is exported from audioYue2Parts but `audioYue2Recording` defines its own blob variant.
- **Impact:** Four places to change for download behavior, such as deferring revoke. An immediate `revokeObjectURL` right after `click()` can cancel the download in some browsers.
- **Suggested fix:** Use one `saveExportFile`-style helper that takes a Blob and defers the revoke with `setTimeout`. Delete the local copies.
- **Confidence:** High

#### [F-239] Film components bypass `api/films.js` with inline URL-building `apiFetch` calls
- **Source ID:** B3-005
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `components/editor/FilmWorkspace.jsx:192,229,309,338,369,397,498-499,505,548`, `components/editor/FilmLifecycle.jsx:30-31,63`, `components/editor/FilmReview.jsx:115`, `components/editor/FilmReferences.jsx:163,200,240,348`, `hooks/useTimelines.js:107,157,190,217,252,256,306,324,343,397,402`
- **Finding:** `api/films.js`, `api/filmReview.js` and `api/yue2.js` provide wrappers for the film and yue2 routes, but about 30 film and timeline calls hand-build `/api/v1/projects/${id}/...` strings. Several do not `encodeURIComponent` ids; others do (`FilmWorkspace.jsx:229,309`). There is no `api/timelines.js`.
- **Impact:** Route strings are scattered, so a backend path change needs many edits. Id encoding is inconsistent. Run polling (`.../film-runs/${id}`) is built in three files.
- **Suggested fix:** Add `getFilmRun/listFilmRuns/startFilmRun/...` and `api/timelines.js`, and replace the inline calls.
- **Confidence:** High

#### [F-240] The `AudioContext ?? webkitAudioContext` shim is defined in three modules
- **Source ID:** B3-006
- **Category:** redundant
- **Severity:** Low
- **Location:** `audioPreviewGain.js:3-5`, `audioWaveform.js:129-134`, `timelineAudioPlayback.js:114`
- **Finding:** Each module has its own constructor lookup. They differ on `window` versus `globalThis` and on how a missing constructor is reported.
- **Impact:** Minor drift and inconsistent SSR/test behavior.
- **Suggested fix:** Export one `getAudioContextCtor()` from a small `audioContext.js` module and use it in all three (or two, after B3-003).
- **Confidence:** High

#### [F-241] Yue2 lab `reloadVersions`/`reloadComparisons` have no stale-response guard
- **Source ID:** B3-007
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `screens/audioYue2Lab.jsx:439-466`, `screens/audioYue2Lab.jsx:485-492`
- **Finding:** The two `useCallback` loaders set state after `await` with no live/request-id check, unlike the transcription and detail effects in `audioYue2Recording.jsx:689-740`, which do guard. They are triggered from effects on `finishedKey` and `enabled` and from manual handlers.
- **Impact:** A slow response from a previous project can overwrite the current project's versions and comparisons after a project switch. It also sets state after unmount.
- **Suggested fix:** Add a request counter ref (as `FilmLifecycle` does) or abort on projectId change. The `createdAt` descending sort is written four times (`audioYue2Lab.jsx:440,458`, `audioYue2Recording.jsx:695`) and could share a helper.
- **Confidence:** Medium

#### [F-242] Timeline editor Space shortcut hijacks focused buttons and links
- **Source ID:** B3-008
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `screens/EditorScreen.jsx:205-213`
- **Finding:** The global keydown handler skips only INPUT, TEXTAREA and SELECT targets. For Space it calls `preventDefault` and toggles playback, so pressing Space on a focused button (Delete clip, tabs, transition buttons) toggles playback instead of activating it. Delete/Backspace similarly act when focus is on `contenteditable` or a button.
- **Impact:** Keyboard-only users cannot activate buttons with Space while the editor is the active screen. This is an accessibility regression.
- **Suggested fix:** Also skip `BUTTON`, `A`, `[role=button|tab]` and `isContentEditable`, or scope the listener to the editor container.
- **Confidence:** High

#### [F-243] Unused exports and over-exported helpers in timeline, yue2 and audio modules
- **Source ID:** B3-009
- **Category:** dead-code
- **Severity:** Low
- **Location:** `timeline.js:1-7`, `audioWaveform.js` (`resetAudioPeakCache`), `yue2Lab.js` (about 15 exports), `audioTakes.js`, `screens/audioResults.jsx`
- **Finding:** `aspectOptions`, `transitionOptions` and `speedPresets` in `timeline.js` have zero references anywhere. `resetAudioPeakCache` is documented as a test seam but no test calls it. About 20 other exports (`YUE2_KINDS`, `samplingBody`, `isYue2Job`, `audioJobIsRunning`, `AudioRunGroup`, `AudioInflightStrip`, and similar) are used only inside their own file.
- **Impact:** Stale option tables can drift from the real transition and speed sets (the editor presumably hard-codes them elsewhere). Extra exports obscure the real module APIs.
- **Suggested fix:** Delete the three unused constants and the unused reset function. Drop `export` from file-local helpers.
- **Confidence:** High

#### [F-244] Character and Look selects, advanced-settings state, and prompt-guide wiring are duplicated across Image and Video studios
- **Source ID:** B3-010
- **Category:** redundant
- **Severity:** Low
- **Location:** `screens/VideoStudio.jsx:3094-3115` vs `screens/ImageStudio.jsx:3822-3843`; also `VideoStudio.jsx:337,354,490,495,2446,2948,2963,2975` vs `ImageStudio.jsx:659,515,556,631,4003,4211,4263,4275`
- **Finding:** The markup for the Character and Look pickers is identical except for the placeholder text. The GPU, sampler and scheduler selects, the `saved.*` state initializers and the `resolveJobResultAssets` wrappers are line-for-line twins. `components/generationStudio.jsx` already holds shared pieces (ModeTabs, TierPickerField), but these were not extracted.
- **Impact:** A fix such as look filtering must be applied in both studios, and AudioStudio drifts further.
- **Suggested fix:** Add `CharacterLookFields`, `GpuSelect` and `SamplerSchedulerFields` to `generationStudio.jsx`, and have both studios use them.
- **Confidence:** High

#### [F-245] God files with clear split seams (VideoStudio, AudioStudio, audioYue2*)
- **Source ID:** B3-011
- **Category:** readability
- **Severity:** Low
- **Location:** `screens/VideoStudio.jsx` (3268 lines, one `VideoStudio()` from line 228, 14 effects, 7 `eslint-disable react-hooks/exhaustive-deps` at 914, 1087, 1131, 1186, 1210, 1371, 1389), `screens/AudioStudio.jsx` (1751 lines, `AudioStudio()` from line 217), `screens/audioYue2Lab.jsx` (1286 lines)
- **Finding:** `VideoStudio` is a single component of about 3,000 lines. The 7 dependency-array suppressions mark effects whose inputs are not modeled.
- **Impact:** Hard to review, and stale-closure bugs are likely in the suppressed effects.
- **Suggested fix:** Extract hooks `useVideoModelSelection`, `useVideoRecipeReplay` (the lines 1186-1390 effects), and `useLtx25Controls`. Move the tier and recipe helpers at lines 133-226 to `videoTierHelpers.js`. Extract the settings panels as components.
- **Confidence:** Medium

#### [F-246] Large pickers render every candidate and do O(n*k) selection lookups
- **Source ID:** B4-006
- **Category:** efficiency
- **Severity:** Low
- **Location:** `apps/web/src/components/DatasetAddDialog.jsx:27` (`selectedIds.includes(asset.id)` per card; `CandidateGrid` maps all `libraryCandidates`), `apps/web/src/components/AssetPicker.jsx:565,878,1031` (`isSelected={(asset) => selectedIds.includes(asset.id)}`), `apps/web/src/screens/characterPanels.jsx:1029-1044` (CharacterAssets filter + sort with `new Date()` per comparison on every render, not memoized, over the whole project asset list)
- **Finding:** Candidate lists are the whole project library (thousands of assets) and are mounted unwindowed, with selection tested by array scan per card per render. `CharacterAssets` recomputes `scopedAssets` (filter + date-parsing sort) and the active/trashed splits on every render, unlike `CharacterStudio` which memoizes the equivalent `characterReferenceCandidates`. `CharacterStudio.jsx:374` does the same unmemoized filter on `latestAssets` for the generation panel.
- **Impact:** Slow open/toggle of "Add images" and the Character import dialog in large projects; wasted work on each CharacterStudio re-render (the studio re-renders on SSE job ticks).
- **Suggested fix:** Build `new Set(selectedIds)` once per render, apply the existing `.asset-tile-windowed` class to picker cards, and `useMemo` the CharacterAssets lists keyed on `[assets, characterId, referenceAssetIds]` (pre-parse `createdAt` once).
- **Confidence:** Medium.

#### [F-247] LTX-2.5 training tables live only in the web bundle and the target-contract carve-outs are re-encoded as `baseModel === "ltx_2_5"` literals
- **Source ID:** B4-008
- **Category:** redundant
- **Severity:** Low
- **Location:** `apps/web/src/training/trainingConfig.js:455-470` (`ltx25ValidationDefaults`), `:475-500` (`LTX25_WORKFLOW_PLANS`, 15 workflows), `:546-587` (validation ranges 32..4096, frames 1..257 `% 8 == 1`, STG block 0..47), literals at `:200, :250, :817, :1167, :1279, :1598`, `ConfigureJobPanel.jsx:335,450,585`; Rust twins `crates/sceneworks-core/src/training.rs:2773-2790` (same defaults), `:2824-2845` (workflow list)
- **Finding:** The parity test pins the numeric aux-loss bounds and `depthAnchoringNoVideoLtxWorkflows`, but not the LTX validation defaults (duplicated verbatim in `ltx_2_5_video_lora_target`), the workflow names keyed in `LTX25_WORKFLOW_PLANS` versus `limits.ltxWorkflows`, or the validation ranges (no backend check exists, so the web rule set is the only enforcement and is unverified against the engine). Several carve-outs that the backend already expresses as capability flags (`supportsSubjectMaskLoss` not advertised for LTX-2.5, `supportsResolutionBuckets` withheld) are additionally hardcoded by base-model name in the web.
- **Impact:** A new LTX workflow or default change in Rust silently diverges from the Studio's plan/defaults; `normalRestrictToSubject` for another no-image base would need a web edit.
- **Suggested fix:** Extend `training_contracts.rs` to read `ltx25ValidationDefaults` and the `LTX25_WORKFLOW_PLANS` keys against `ltxWorkflows`; expose a `limits.supportsNormalRestrictToSubject` flag instead of the `baseModel` literal.
- **Confidence:** Medium.

#### [F-248] Failed reference upload / import in the character panels strands state or reports false success
- **Source ID:** B4-009
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `apps/web/src/screens/characterPanels.jsx:613-634` (`onUpload`), `:1013-1020` (`importAssetIds`), `apps/web/src/hooks/useCharacters.js:44-58` (`withCharacterApi` returns `null` on error)
- **Finding:** `addCharacterReference` never throws (it returns `null` after setting the banner), so `onUpload` proceeds to `setReferenceAssetId(asset.id)` and clears its status even when approving failed (the reference-guard effect then snaps the pick back to `[0]`); `importAssetIds` ignores the `null`s, loops sequentially, and the dialog closes (`attach` then `onClose`) as if all were attached. The generation `generate()` has `try/finally` with no `catch`, so a rejecting `createImageJob` is an unhandled rejection from the click handler.
- **Impact:** Partial imports look successful; "Uploading reference..." can be replaced by nothing while the reference was not saved.
- **Suggested fix:** Check the return value and surface "k of n attached"; add `catch` with a status message in `generate`.
- **Confidence:** Medium.

#### [F-249] Dead per-family wrappers and test-only exports in trainingConfig.js
- **Source ID:** B4-010
- **Category:** dead-code
- **Severity:** Low
- **Location:** `apps/web/src/training/trainingConfig.js:1643-1666` (`depthAnchoringIssues`, `depthAnchoringSnapshot`, `faceLossSnapshot` have zero references anywhere, including tests; `bodyLossSnapshot`, `latentPerceptualSnapshot`, `bodyLossIssues`, `faceLossIssues`, `latentPerceptualIssues` are referenced only by tests), `:1127-1131` aside; also `targetSupportsVaeAnchorLoss/targetSupportsLatentLpipsLoss` (`:~1480`, tests only), `optimizerLabels`, `networkTypeLabels` and `qualityPresetLabels` exports (internal use only)
- **Finding:** Exports verified by grep across `apps/web/src` (excluding `*.test.*`): the eight family wrappers have no production importer since `configValidation`/`trainingConfigSnapshot` call `auxLossesIssues/auxLossesSnapshot` directly. Many `export const` bound constants (`vaeAnchor*`, `latentLpips*`, `defaultSampleCount`, `ltx25ValidationDefaults`, `ZIMAGE_TURBO_TRAINING_ADAPTER_REPO`) have no JS importer; those that the Rust parity test reads by `export const NAME = ` text must stay.
- **Impact:** Extra surface and tests keeping unused helpers alive; the "must stay because Rust greps the line" constraint is only documented in comments.
- **Suggested fix:** Delete the 8 wrappers (and their tests); mark parity-read constants in one block with a comment, or move them to `trainingLimits.js` so the Rust `include_str!` target is a small file.
- **Confidence:** High.

#### [F-250] Duplicated upload/file filtering helpers across the three file-drop entry points
- **Source ID:** B4-011
- **Category:** redundant
- **Severity:** Low
- **Location:** `apps/web/src/screens/TrainingStudio.jsx:101-125` (`uploadFileBaseName/Extension/Stem`, `isImageUpload`, `isCaptionUpload`), `PoseLibraryScreen.jsx:160` (`file.type?.startsWith("image/")`), `components/AssetPicker.jsx:537-541,814-818,1019-1023` (three copies of `handleDrop`), `components/DatasetAddDialog.jsx:117-127`
- **Finding:** Each drop surface re-implements its own image-type test (training uses MIME OR extension set, pose uses MIME only, so an extensionless/HEIC-labelled drop is accepted in one and not the other) and the same `preventDefault -> setDragActive(false) -> files` handler, although `FileDropzone` already exists.
- **Impact:** Inconsistent accept rules across screens; fixes to drop handling need N edits.
- **Suggested fix:** Move the predicates to a `uploadFiles.js` util and let `FileDropzone` own `onDrop` normalization (hand callers a plain array).
- **Confidence:** High.

#### [F-251] Download-size label formatting is copy-pasted in six places
- **Source ID:** B5-007
- **Category:** redundant
- **Severity:** Low
- **Location:** `ModelManagerScreen.jsx:111-116`, `SetupWizard.jsx:63-68`, `PresetManagerScreen.jsx:199-201`, `components/RequiredModelsNotice.jsx:30-33`, `components/ModelAvailabilityGate.jsx:23-26`, `simple/SimpleModelManager.jsx:274-277`
- **Finding:** Each one repeats `downloadSizeEstimated ? "~"+label : label`. Only the "unavailable" fallback text differs ("Unavailable" vs "Size unavailable").
- **Impact:** The simple and advanced Model Managers can diverge on wording. Adding a size unit needs six edits.
- **Suggested fix:** Add `downloadSizeText(model, fallback)` to `formatting.js` and use it everywhere.
- **Confidence:** High

#### [F-252] Byte and GiB helpers duplicated beside `formatBytes` and `modelCache.bytesToGib`
- **Source ID:** B5-008
- **Category:** redundant
- **Severity:** Low
- **Location:** `ModelManagerScreen.jsx:121-132` (`formatTierSize`), `QueueScreen.jsx:182-190` (`formatMemory`), `SettingsScreen.jsx:121-129` (`telemetryBytesToGib`/`telemetryGibLabel`), `hooks/useModelsAndLoras.js:18-21` (`uploadLimitLabel`), versus `formatting.js:52` (`formatBytes`) and `modelCache.js:19,305` (`BYTES_PER_GIB`/`bytesToGib`)
- **Finding:** There are four more hand-rolled `/1024/1024/1024` formatters. `ModelManagerScreen` already imports `formatBytes` and still keeps `formatTierSize`.
- **Impact:** Slightly different rounding and unit text across the Models, Queue and Settings screens.
- **Suggested fix:** Give `formatBytes` and `formatMiB` options for the "Size unavailable" fallback, and delete the local copies.
- **Confidence:** High

#### [F-253] Queue screen ships a "Placeholder generation" job composer to production
- **Source ID:** B5-009
- **Category:** dead-code
- **Severity:** Low
- **Location:** `apps/web/src/screens/QueueScreen.jsx:259,288,330-345`; `apps/web/src/App.jsx:2524-2544`
- **Finding:** The first panel on Queue is "Add a job", a free-text prompt (default "Placeholder generation") that POSTs `type: "placeholder"`. This is the Phase-0 scaffold from before real studios existed.
- **Impact:** End users can enqueue a no-op job type. It adds `createPlaceholderJob` to the app context surface, and its tests keep a dead path alive.
- **Suggested fix:** Remove the composer, or gate it behind a dev flag. Keep the GPU and project filter controls, then drop `createPlaceholderJob` from `appStaticValue`.
- **Confidence:** Medium. It may be kept deliberately for smoke-testing.

#### [F-254] `MODEL_IMPORT_ENABLED` is an always-true compile-time flag that still guards dead branches
- **Source ID:** B5-010
- **Category:** dead-code
- **Severity:** Low
- **Location:** `apps/web/src/screens/ModelManagerScreen.jsx:180,823,2139`
- **Finding:** The flag is `const MODEL_IMPORT_ENABLED = true`. The `!MODEL_IMPORT_ENABLED` arms at 823 and 2139 can never run. This was flagged as F-139 in 07-01 when it was `false`. It has since flipped, but the guards were left behind.
- **Impact:** Misleading conditionals; the credential-fetch effect reads as optional but always runs.
- **Suggested fix:** Delete the constant and simplify both conditions (the credential effect gate becomes unconditional).
- **Confidence:** High

#### [F-255] LibraryScreen recomputes filters and tag sets over all assets on every render, from the live context
- **Source ID:** B5-011
- **Category:** efficiency
- **Severity:** Low
- **Location:** `apps/web/src/screens/LibraryScreen.jsx:78-137` (also `PoseLibraryScreen.jsx:610-622`, which is partly memoized)
- **Finding:** `libraryAssets`, `visibleAssets` (`foldUpscaledAssetVariants`), `trashedInView`, `availableTags` and the VQA job filter are unmemoized, and the screen reads the merged `useAppContext()`. Each job-progress tick re-runs several O(n) passes plus `[...new Set(flatMap)].sort()`.
- **Impact:** Slows interaction on large projects while a generation runs. `AssetGrid` receives a fresh `visibleAssets` array each time, which defeats any memoization inside it.
- **Suggested fix:** Wrap the derivations in `useMemo` keyed on `assets` and the filters. Split the `useAppContext` read so only `jobs` (needed for VQA) subscribes live.
- **Confidence:** Medium. I did not profile it.

#### [F-256] `assetMatchesSearch` is exported from a screen and imported by a component in `simple/`
- **Source ID:** B5-012
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `apps/web/src/screens/LibraryScreen.jsx:11` consumed by `apps/web/src/simple/SimpleAssets.jsx:7`
- **Finding:** A pure helper lives in a screen module, so the Simple UI imports the whole `LibraryScreen` module (and, via the asset panels, its dependency graph). That inverts the intended layering of components and utils below screens.
- **Impact:** Any change to `LibraryScreen` imports can create cycles or pull its code into the simple bundle. There is also no single search predicate owned by the asset layer.
- **Suggested fix:** Move it to `assetSearch.js` (next to `assetVariants.js`) and import it from both.
- **Confidence:** High

#### [F-257] VectorStudio is written as dense one-line JSX and does not follow the WorkPanel and page-frame convention
- **Source ID:** B5-013
- **Category:** readability
- **Severity:** Low
- **Location:** `apps/web/src/screens/VectorStudio.jsx:222-305`
- **Finding:** Whole forms and selects are inlined into single 200-300 character lines (e.g. :273-283, :293). It uses a bare `section.vector-studio` with `work-panel` classes instead of the shared `WorkPanel`, `ModeTabs` and `page-frame` building blocks that Library, Queue, Logs and Settings use. The three `useEffect`-driven model pickers (lines 160-163, 165-185) fan out into seven separate model-id states.
- **Impact:** Hard to review or diff; it diverges visually from sibling screens.
- **Suggested fix:** Reformat, use `WorkPanel` and `ModeTabs`, and collapse the model ids into one reducer or `useVectorModels` hook.
- **Confidence:** High

#### [F-258] SettingsScreen (1,409 lines) and PresetManagerScreen (1,714 lines) hold several screens each
- **Source ID:** B5-014
- **Category:** readability
- **Severity:** Low
- **Location:** `SettingsScreen.jsx:209-1409` (tabs at `:223,692`), `PresetManagerScreen.jsx:309-1714`
- **Finding:** Settings is one component with about 25 `useState`s and 12 handlers spanning appearance, workflow-embed, credentials, cache, remote access, GPU, storage and worker restart. Its Unicode-aware remote-password normalizer (`:130-182`) is a security-relevant contract mirror inline in the file. Preset Manager contains the list view, the editor, general versus model presets, the payload builders, and `buildGeneralPayload`/`buildPayload`, which duplicate the prompt and order assembly (`:515-545` and `:567-600`).
- **Impact:** Every settings change touches a god component, and the password contract mirror is not unit-isolated.
- **Suggested fix:** For Settings, extract one component per tab under `screens/settings/` (Credentials, Cache, RemoteAccess, Gpu, Storage). Move the password policy to `remotePasswordPolicy.js` with a parity test against Rust. For Presets, extract `PresetList`, `PresetEditor` and `presetPayload.js`, and unify the two payload builders.
- **Confidence:** High

#### [F-259] Duplicated "resolved-model cache" state and refresh wiring between Settings and ModelManager
- **Source ID:** B5-015
- **Category:** redundant
- **Severity:** Low
- **Location:** `SettingsScreen.jsx:290-336` and `ModelManagerScreen.jsx:884-915`
- **Finding:** Both screens define the same `refreshCache` and `refreshModelCache` callbacks (fetch snapshot, set state, set error string, null on failure) around the shared `useCacheConvergence`. Only the hook was shared.
- **Impact:** The error handling of the two copies can diverge. ModelManager also treats `snapshot.error` while Settings reads `cache.error` only at render time.
- **Suggested fix:** Add `useModelCacheStatus()` returning `{cache, error, refresh, stalled}` and use it in both screens.
- **Confidence:** Medium

#### [F-260] ModelManager duplicated linked-library scan, and three near-identical delete handlers
- **Source ID:** B5-016
- **Category:** redundant
- **Severity:** Low
- **Location:** `ModelManagerScreen.jsx:1231-1249` (`refreshLinkedStatuses`) versus `:1291-1311` (mount effect); `:1063-1115`, `:1117-1160`, `:1162-1190` (deleteModel, deleteModelVariant, deleteLora); `:630-636` (`onResumeDownloadJob`/`onFreshDownloadJob`)
- **Finding:** The mount effect re-implements `refreshLinkedStatuses` (fetch roots, scan each, index). The only differences are cancellation and the empty-roots reset, and the effect does not clear stale statuses when roots become empty. The three delete functions share one confirm, then set-busy, then call, then message pattern. The two download-job callbacks are the same function.
- **Impact:** Divergent edge handling (a stale "Needs relink" can persist after the last root is removed), and more code to maintain.
- **Suggested fix:** Have the effect call a shared `loadLinkedStatuses(signal)`. Factor one `runDeletion({confirm, call, successText})` helper. Drop one of the two retry aliases.
- **Confidence:** High

#### [F-261] PoseLibrary refresh clears `loading` after abort, and `resetSession` revokes object URLs inside a state updater
- **Source ID:** B5-017
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `PoseLibraryScreen.jsx:535-554` (refresh `finally`), `:285-289`, `:204-211`
- **Finding:** When a request is aborted, `finally { setLoading(false) }` still runs, so it can flip `loading` false while the newer request is still in flight (and set state after unmount). `URL.revokeObjectURL` is called inside `setSources` updater functions, which are meant to be pure (StrictMode runs them twice). Mutation helpers call `refresh()` without a signal, so they can overlap the effect's fetch with no ordering guard.
- **Impact:** A minor flicker or stale list; the double revoke is harmless today.
- **Suggested fix:** Guard `setLoading(false)` with `if (!signal?.aborted)`. Do the revoke outside the updater. Serialize refreshes through a ref'd controller, as `useGenerationMetrics` already does.
- **Confidence:** Medium

#### [F-262] Outside-click plus Escape dropdown effect is copy-pasted in 4 components
- **Source ID:** B6-003
- **Category:** redundant
- **Severity:** Low
- **Location:** `apps/web/src/components/CompactSelector.jsx:26-44`, `apps/web/src/components/StylePicker.jsx:122-136`, `apps/web/src/components/AccentPicker.jsx:17-30`, `apps/web/src/components/generationStudio.jsx:1082-1095`
- **Finding:** Each defines identical `onDocMouseDown` / `onDocKey` handlers and add/remove listener boilerplate. `assetPanels.jsx:822` and `SimpleSheet.jsx:28` have variants. None returns focus to the trigger or supports arrow keys.
- **Impact:** Four places to fix any dropdown a11y bug, and none of them gets focus return.
- **Suggested fix:** Extract `useDismissOnOutsideOrEscape(ref, open, close)` into a hooks file, and add focus-return there once.
- **Confidence:** High

#### [F-263] Roughly 130 CSS class selectors in styles.css (and a few in simple.css) are never referenced from source
- **Source ID:** B6-005
- **Category:** dead-code
- **Severity:** Low
- **Location:** `apps/web/src/styles.css` (20k lines), e.g. `.keyboard-card` (6093, 6303), `.form-row` (2233), `.job-row`, `.local-job-card`, `.asset-bin*`, `.asset-tray`, `.tray-*`, `.timeline-panel/-track/-ruler`, `.video-preview-*`, `.video-playback-*`, `.video-review*`, `.preset-rail*`, `.model-row*`, `.recent-clips-*`, `.training-live-*`, `.training-sample-*`; `apps/web/src/simple/simple.css` (`.su-asset-placeholder`, `.su-notice--error`)
- **Finding:** A class-name scan against the concatenated non-test JS/JSX found 173 candidate names. After excluding names built from template-literal prefixes, 131 remain. They look like leftovers from retired screens (older video studio, tray, model-row layout, timeline). Some are duplicated, e.g. `.keyboard-card` is defined twice and `.form-row button` rules are repeated. A few are false positives (`googleapis`, `gstatic`, `woff2` come from URLs).
- **Impact:** Bloats the 20k-line stylesheet and the cascade, and makes real dead rules hard to tell from live ones.
- **Suggested fix:** Run a PurgeCSS or knip-style check with a safelist for dynamic prefixes (`chip-*`, `arch-*`, `su-dot--*`, `su-job-badge--*`), then delete the confirmed selectors.
- **Confidence:** Medium (dynamic class construction through helpers such as `cx(...)` is possible)

#### [F-264] Duplicated micro-helpers instead of shared formatting utils
- **Source ID:** B6-006
- **Category:** redundant
- **Severity:** Low
- **Location:** `apps/web/src/components/RefinePromptControl.jsx:6-11` (`formatGb`), `apps/web/src/screens/ModelManagerScreen.jsx:121` (`formatTierSize`), `apps/web/src/formatting.js:52` (`formatBytes`); `apps/web/src/components/QwenRewritePromptControl.jsx:35-40` and `RefinePromptControl.jsx:18-23` (`isModelMissing`, same regex); `components/AssetPicker.jsx:35-60` (`compactDate`, `titleFor`) vs `components/DatasetAddDialog.jsx:8` (`assetTitle`)
- **Finding:** Byte, size and title formatters are re-declared locally even though `formatting.js` exists. The "model not installed" detector is duplicated verbatim across the two prompt controls. Both controls (216 and 227 lines) also share the same abort, status and review skeleton.
- **Impact:** Inconsistent units (1e9 vs 1024 base) and message detection that can drift.
- **Suggested fix:** Move `isModelMissing` and a `titleForAsset` into shared modules, reuse `formatBytes`, and consider a `usePromptRewriteFlow` hook behind both controls.
- **Confidence:** Medium

#### [F-265] Hard-coded model ids in generic UI code (Simple and editor) duplicate backend capability data
- **Source ID:** B6-007
- **Category:** architecture-drift
- **Severity:** Low
- **Location:** `apps/web/src/simple/SimpleVideoStudio.jsx:167`, `apps/web/src/simple/simpleJobs.js:351` (`model === "minimax_h3" && mode === "image_to_video"`), `apps/web/src/components/editor/FilmSound.jsx:7` (`SPEECH_MODELS`), `components/editor/useEditorGeneration.js:22` (`LTX_VIDEO_MODEL_ID`), `components/editor/FilmWorkspace.jsx:238`
- **Finding:** The last-frame capability is keyed on a literal model id, in both the UI gate and the job-body builder, instead of a manifest or capability flag. The speech-model list is also hard-coded. Advanced VideoStudio presumably uses model capability data for last-frame, so Simple will not pick up the next model that supports it.
- **Impact:** Simple vs advanced parity drift. Adding a model needs UI edits in several places.
- **Suggested fix:** Read a `supportsLastFrame` or `ui.capabilities` flag from the model entry (as `ui.promptGuide` already is) and drop the literals.
- **Confidence:** Medium

#### [F-266] Web bundle config: build-time projection silently swallows all errors; web CI Node differs from Docker
- **Source ID:** S1-007
- **Category:** bad-pattern
- **Severity:** Low
- **Location:** `apps/web/sdxl-cuda-memory-projection.js:45-48`, `apps/web/vite.config.js:10,45`, `.github/workflows/check.yml:49` (`node-version: "20"`) vs `docker/rust.Dockerfile:16` / `web.Dockerfile:1` (Node 22)
- **Finding:** `loadSdxlCudaMemoryProjection` returns `null` on any exception (including a failed `import` of `scripts/package-sdxl-cuda-endpoints.mjs`' dependencies or hash mismatch) and the build proceeds with the conservative estimate; the vite config also depends on a repo-root script outside apps/web. CI lints/tests/builds on Node 20 while shipping on 22 and Vite 8 / vitest 4 floor is Node 20.19/22.12.
- **Impact:** evidence tampering or a broken receipt degrades the UI estimate without any build signal; Node-version-specific bugs would not be caught by the web job.
- **Suggested fix:** log a `console.warn` (or fail when `CI`/production) on the catch path, and align `setup-node` to 22 (or add `engines`).
- **Confidence:** Medium

#### [F-267] Tracked non-source artifacts: design zips, experiment dirs, one-shot spikes
- **Source ID:** S1-008
- **Category:** dead-code
- **Severity:** Low
- **Location:** `design/*.zip` (12 files, 4.3 MB), `docs/sc-3494/` (4.25 MB), `docs/sc-4422-kps-experiment/` (5.3 MB incl. 2 PNG + .py), `docs/sc-16962/` (326 KB), 31 other `docs/sc-NNNN*` dirs/files (2022-era spikes), `scripts/spikes/` (9 py + 2 ort projects, last touched 2026-09-10), `docs/generated/memory-matrix.json` (9.2 MB, generated), root `screenshot*.png` (4 files, ~0.6 MB), `docs/calibration/**/*.log` (164 tracked logs, 881 KB total; 97 in sc-16170, 16 loose `sc-15823-refresh-*.log`)
- **Finding:** these are historical evidence, designer hand-offs and generated output. The 164 logs are intentionally allow-listed in .gitignore:60-64 and marked binary in .gitattributes for hash stability, so they are not a mistake and are small; the 16 top-level `sc-15823-refresh-*.log` and the `sc-15833` logs are the only ones outside the allow-listed sub-dirs' hashing protection. The zips are opaque to review/grep.
- **Impact:** repo bloat (~230 MB working tree dominated by S1-002) and stale-doc noise; nothing here is a correctness risk.
- **Suggested fix:** move `design/*.zip` and finished spike dirs to a release/archive branch or wiki; keep logs but consider one archive per sc-NNNN; leave memory-matrix.json (it is CI-checked via `check:memory-matrix`) but note it is 9 MB of diff noise on every regenerate.
- **Confidence:** High on sizes; Medium on whether to delete (owner call)

#### [F-268] Root `package.json` `check` script is a hand-maintained 100-file list
- **Source ID:** S1-009
- **Category:** readability
- **Severity:** Low
- **Location:** `package.json:11`; guard in `scripts/test-list-coverage.test.mjs`; `scripts/platform-review-contracts.test.mjs` (5.5k lines of regex assertions over workflow/Dockerfile text)
- **Finding:** one ~7 KB `node --test a b c ...` line must list every test; a coverage test exists to catch omissions. Several tests (e.g. platform-review-contracts.test.mjs:1889, :2145) assert exact strings in workflow YAML (`huggingface_hub==0\.36\.0`), so routine workflow edits need matching test edits.
- **Impact:** merge-conflict magnet and brittle string-pin tests; not a defect.
- **Suggested fix:** `node --test "scripts/**/*.test.mjs"` with an explicit exclude list for the one separate invocation; trim text-pin tests to behavioural checks.
- **Confidence:** Medium


### Informational findings — web/tooling

#### [F-269] Access token stays plaintext in localStorage and rides in `?ticket=` query strings (still present, documented)
- **Source ID:** B1-014
- **Category:** security
- **Severity:** Info
- **Location:** `apps/web/src/accessToken.js:1-80`, `api.js:144-177` (`eventUrl`, `withMediaTicket`)
- **Finding:** Previously F-078 (2026-07-12), still present and now explicitly accepted in the module header. The long-lived token never goes into a URL (header only). The URL-borne values are short-lived sliding tickets, which resolves F-008. Residual risk is XSS exfiltration of the stored token. The review found no `dangerouslySetInnerHTML`, `innerHTML`, `postMessage` or `window.open` in `apps/web/src`, and all `target="_blank"` links carry `rel`. `Markdown.jsx` builds React elements and allow-lists `href`.
- **Impact:** None new. The posture is acceptable for loopback/LAN. It would need the httpOnly-cookie exchange noted in the header before any public exposure.
- **Suggested fix:** None required now. Keep the one-file seam, and add a CSP `<meta>` or response header for the Docker/browser build (the Tauri build has one in `tauri.conf.json:25`).
- **Confidence:** High

#### [F-270] Hand-rolled SHA-256 fallback in a protocol path
- **Source ID:** B4-012
- **Category:** security
- **Severity:** Info
- **Location:** `apps/web/src/training/datasetReadiness.js:16-100`
- **Finding:** `captionHash` falls back to a 60-line custom SHA-256 for non-secure contexts. It is the only implementation in the app and is exercised by tests, so this is an observation, not a defect: `expectedCaptionHash` is a concurrency token, not a secret.
- **Impact:** Maintenance cost only.
- **Suggested fix:** Optionally replace with a vetted tiny library (or have the server hash) to drop the custom crypto.
- **Confidence:** High.

#### [F-271] Three independent "load an image to get pixels/dims" helpers
- **Source ID:** B2-013
- **Category:** redundant
- **Severity:** Info
- **Location:** `apps/web/src/screens/ImageEditor.jsx:1031-1043` (`blobToImage`), `apps/web/src/imageDimensions.js:2-22` (`probeImageDimensions`), `apps/web/src/assetBatch.jsx:166-186` (`loadImageDims`)
- **Finding:** Each creates `new Image()`; only `probeImageDimensions` is cancel-safe. `loadImageDims` has no timeout, and on a stuck network load the batch for-loop (L193+) can hang.
- **Impact:** Minor; inconsistent cleanup and error handling.
- **Suggested fix:** Consolidate in `imageDimensions.js` (decode-from-blob with owner-revoked URL; dims probe with abort).
- **Confidence:** Medium

#### [F-272] Mixed token sources inside the same screens
- **Source ID:** B5-018
- **Category:** architecture-drift
- **Severity:** Info
- **Location:** `SettingsScreen.jsx:499` (`serverToken()`), versus `ModelManagerScreen.jsx:625,1235,1295` and `LogsScreen.jsx:121` (context `token`); `credentials.js:23-49`
- **Finding:** Some code reads the access token from context and some from the module-level `serverToken()`. They resolve to the same `accessToken.js` store today, but any caller in a screen can pick either.
- **Impact:** Low; a future change to token rotation must be applied to both paths.
- **Suggested fix:** Standardise on `useAppStatic().token`, or on `readAccessToken()` everywhere, and document the choice in `credentials.js`.
- **Confidence:** Medium

---

#### [F-273] Clickable div in EditorGuides, with link interception that is not keyboard-friendly
- **Source ID:** B6-008
- **Category:** bad-pattern
- **Severity:** Info
- **Location:** `apps/web/src/components/editor/EditorGuides.jsx:39-46`
- **Finding:** `onClick` on a non-interactive div is used to delegate clicks on `<a href="#editor-guide-…">` produced by Markdown. Enter on a link also fires click, so keyboard access works. There is no real security issue, since Markdown is rendered as React elements (`Markdown.jsx:7`).
- **Impact:** Minor lint a11y noise only.
- **Suggested fix:** Optional: pass a link renderer to `Markdown` instead.
- **Confidence:** Medium

#### [F-274] `huggingface_hub==0.36.0` pin repeated inline in 8+ places
- **Source ID:** S1-010
- **Category:** redundant
- **Severity:** Info
- **Location:** `.github/workflows/windows-candle.yml:696,1076,1396`, `macos-mlx.yml:722,733`, `scripts/ci/iris-mlx-real-weights/run.sh:79`, `scripts/ci/supra2-mlx-real-weights/run.sh:49`
- **Finding:** version literal is duplicated and enforced by a test regex rather than a shared constant/requirements file.
- **Impact:** bumping requires editing all and the pin test.
- **Suggested fix:** a single `scripts/ci/requirements-hf.txt` used with `pip install -r`.
- **Confidence:** High


## Themes and systemic observations

### Part 1 themes (Rust backend)

1. **Hardened-then-re-copied helpers (the main source of duplication).** Atomic temp+rename writes exist ~9 times with different durability (D1-001, C1-002); streaming file hashing 13 times and hex encoding 10 (D1-002, W3); `safe_download_dir` three times byte-identical, "kept in sync" by a comment (D1-003, A2-002, W5); eight-plus "safe relative path" predicates and four Hugging Face repo-id validators with different rules (A1-002/006, C1-005); safetensors header reading six times (C2-003); the same `MemoryEvidence` literal ~10 times (W2-003). Root cause per D1: core utilities are `pub(crate)` and there is no public core `hashing`/`env`/`fs` module. Promoting ~6 functions to a public core module fixes most clusters.
2. **MLX ↔ Candle twin drift.** Same tier clamp / quant parse / adapter loop / model-dir resolver once per backend with "mirrors X" comments (W1), diverged floor semantics (W2-001), 15 byte-identical functions plus many near-identical ones in the memory-adapter binaries (X1-006..008). Video is consistently one hardening generation behind image (W1). A backend-neutral shared layer is the highest-leverage structural change.
3. **Shared worker layers only partially adopted.** `analysis_jobs_common`, `progress.rs`, `smoke_support.rs`, `payload::quant_int`, `ort_cuda`: each has a long tail of verbatim clones citing the original in a comment (W4-001/002, W5-003/005, W1-001). Cancel semantics in particular drift in the hand-rolled loops (W4-003), so cancellation correctness is the main recurring runtime risk.
4. **Rules enforced by comments instead of the compiler or tests.** `architecture_tests.rs` checks job-type parity, dispatch arms, family allow-lists and heartbeat wiring, but **not** layering or cfg rules (W5-008); there is nothing to stop the next copy of a helper (C1 suggests extending it to core). Pin tests exist but their premises have gone stale (D1-007). `use super::*` in 42 worker and 44 of 61 rust-api modules hides real dependencies (W5, A1).
5. **Trust-boundary validation applied at entry points, not to values read back or to sibling code.** ETag into a blob path (W3-003), `Link`-header and COCO `file_name` values (W3), `sourcePath` on catalog routes lacking the `require_local_peer` guard that model/checkpoint library has (A1-001), symlink-unsafe asset confinement (W4-005), no timeout/protocol whitelist on the audio-upload ffmpeg (C2-001). Fixes landed per module and did not reach siblings; each needs a crate-wide grep for the same shape.
6. **Loopback trust model.** The desktop sets `SCENEWORKS_TRUST_LOOPBACK`; loopback peers skip the token; no `Origin`/`Host` check on `/api/*` (A2-001) and the `/mcp` defence is off in wildcard-bind LAN mode (X1-001). The 2026-07-12 review raised this for `/mcp` and it was fixed there only. Treat as one finding and fix at the middleware level.
7. **SQLite stores lack a common open/configure/migrate/begin-write entry point** (C1-001/007): the safest choice (`BEGIN IMMEDIATE`) is used by catalog and jobs but not by `project.db`, the one store two processes share most.
8. **God files and `include!`.** `image_jobs/base.rs` 19.6k lines via `include!` (still open since 2026-07-12 F-020), `run_image_generate_job_unexplained` 1,292 lines (since 2026-07-01 F-026), `rust-api/src/lib.rs` 6.3k lines and `models.rs` 15.8k (≈46% inline tests), `memory-adapter/src/bin/mlx.rs` 33.8k lines, core stores of 5–17k lines. Inline tests (30–70% of many files) hide duplication and dead code.
9. **cfg predicate repeated ~1,900 times across ~100 files** (W5-001, W4-010); a `build.rs` cfg alias would remove it and is the biggest single readability win.
10. **Drift guards are strong where they exist** (job types, family allow-lists, geometry, version alignment) **and absent one layer down:** web `fallbackModels` vs `builtin.models.jsonc` (D1-005), env-flag truthiness (D1-004), crate dependency direction. Docs: CLAUDE.md/FEATURE_DEVELOPMENT.md are current; CONTRIBUTING.md and per-crate READMEs lag the deliberate CI/merge-queue/gate teardown (D1).
11. **Positive notes:** all 211 API routes sit behind one `access_control` layer with 5 public paths and constant-time token compare (A2); most prior High findings are verified fixed; dead code is near zero in rust-api; the timeout/SSRF/atomic-promotion work in downloads is solid within each stack (W3).


### Part 2 themes (web app, scripts, CI)

1. **Extract-once, never-migrate.** `pollJobToCompletion`, `useScopedCrudList`, `useAppStatic`, `testUtils/dom.js` and the `api/` layer each cover only a fraction of their callers (2 of 9 refresh hooks, 15 of 45 context consumers, 4 of ~40 endpoint domains; request paths hand-written at 236 call sites) — B1-003/005/009, B3-005, B2-011. Finish the migration or add a lint rule that forbids the old form.
2. **God components that keep growing.** `App.jsx` 2.4k → 4.3k lines (one 3,750-line `App()`), `ImageStudio` 3.2k → 4.9k, `ImageEditor` 4.1k → 4.5k with a ~190-key scope bag, `TrainingStudio` 2.6k, `ModelManagerScreen` renders the whole catalog through ~750 lines of inline closures (B1-001, B2-002/003, B4-007, B5-001). The earlier July decomposition findings are still open and worse; concrete split seams are in the findings.
3. **Polling and lifecycle gaps on error/race paths.** Film run/export loops and three overlapping film polls with no abort or visibility guard (B3-001/002, B1-004), batch runs that outlive the studio (B2-001), import flows with no abort/partial-failure recovery (B4-003/009), stale-response races (B3-007, B4-004, B5-004). Media/object-URL/listener hygiene is otherwise good.
4. **Server-owned state re-implemented client-side.** License acks in `localStorage` that is wiped on desktop relaunch (B5-002), training-config validation that ignores the advertised `limits` (B4-001), `modelCatalog.js` as a test-only mirror of 46/140 manifest models (B1-007; see also F-092), hard-coded model ids in generic UI (B6-007), LTX-2.5 tables only in the web bundle (B4-008). The Rust↔JS parity tests pin constants but not rules.
5. **Parallel UI stacks.** The Simple UI built its own overlay stack without the shared `Modal` focus handling (B6-001), and duplicates refine/guide/dropdown logic (B6-002/003, B2-006). Pose and Key Point capture are near-identical state machines that bypass shared job creation (B5-003).
6. **Dead code is mostly in CSS and a few modules.** ~130 orphan selectors in `styles.css`, `LayersPanel.jsx`, `useAudioPreviewGain.js`/`audioPreviewGain.js`, `consumeEditorLaunch`, an always-true `MODEL_IMPORT_ENABLED` flag (B6-004/005, B1-008, B2-012, B5-010). A `knip`/ESLint unused-export check in CI would catch this class.
7. **Scripts and CI.** `scripts/` is 139k LOC of mostly one-epic harnesses that each re-implement sha256, canonical JSON, JSONC and exec helpers with different key ordering, so the same data can hash differently (S1-004). The dev `web` Docker image lacks build inputs the Vite config needs and its smoke test only proves liveness (S1-001); 192 MB of raw `.rgb` frames and machine-specific paths are tracked in git (S1-002/003); `workflow_dispatch` inputs are interpolated into `run:` shell (S1-006). Otherwise CI hardening is consistently applied (SHA-pinned actions, same-repo gates, empty advisory-ignores).
8. **Security posture of the web app is good.** No HTML sinks, `postMessage`/iframe surface or unsafe `target=_blank` found in screens, components or Simple UI; URLs go through `safeExternalUrl`; credential inputs are write-only. Residual items: access token in `localStorage` and `?ticket=` query strings (B1-014, documented), an un-allow-listed licence link (B1-010).

## Suggested remediation order

1. Security: fix A2-001/X1-001 in the middleware (Origin/Host validation, apply the MCP rebinding check to `/api`, no wildcard-bind exemption under loopback trust); W3-003 (`safe_join` the ETag); A1-001 (`require_local_peer` on catalog `sourcePath`); C2-001 (ffmpeg timeout/protocol whitelist).
2. Correctness: C1-001 (`BEGIN IMMEDIATE` for project.db); W2-001 (decide `any` vs `all` and share one helper); W4-001/003 (route hand-rolled cancel loops through the shared loop).
3. De-duplication via a public core utility module (atomic write, hashing, hex, random id, safe path, safe_download_dir, glob, format_bytes, env bool), then delete the copies; extend `architecture_tests.rs` to forbid new ones.
4. Structural: `build.rs` cfg alias; split `image_jobs`, `rust-api/lib.rs`, `models.rs`; extract inline tests.
5. Part 2 (web/tooling): fix the dev web Dockerfile inputs (S1-001) and `workflow_dispatch` shell interpolation (S1-006); add abort/unmount guards to film loops and batch runs (B3-001, B2-001); make training validation honour `limits` (B4-001); move license acks server-side (B5-002); then migrate callers onto the existing `api/`, `useScopedCrudList` and `useAppStatic` seams and split `App.jsx`/`ImageStudio`/`ImageEditor` along the seams named in B1-001, B2-002, B2-003.


## Coverage notes

Per-area gaps reported by each reviewer (prefixes as above).

**W1**

- Line-level audit of the lane bodies in `image_jobs/{moss,instantid,flux2,krea_imported,checkpoint_plan,supra2,iris,qwen_edit_candle,sdxl_control,krea_control_candle}.rs`, `video_jobs/{minimax_h3,vace,ltx,seedvr2,reference_audio,svd,mochi,krea_realtime}.rs` (only resolver, adapter and tier sections read).
- `image_jobs/base.rs` lines 5400-19649 beyond structure and signatures; `video_admission.rs` ladder selector internals (1203-2450) and veda admission (3332+).
- `generator_cache.rs` lines 130-850 skimmed only; `krea_control_fit.rs` 520-1029 not read.
- The `tests.rs` monoliths (image_jobs 29.8k, video_jobs 21.5k, video_admission 5.7k) were surveyed structurally only. The `*_smoke.rs` files were checked only for duplicated scaffolding (W1-016).

**W2**

- Line-level audit of `mlx_fit_gate.rs` (22k lines) and `candle_memory_strategy.rs` beyond the non-test structure, twin hunting and sampled regions. `memory_anchor.rs` (core) was surveyed only for twins and constants.
- `mlx_tier_admission_tests.rs`, `vector_admission.rs`, `engines.rs`, `engine_capability_facts.rs`, `pinned_engine_geometry.rs`, `footprint_measure.rs`, `inference_runtime.rs`, `ort_cuda.rs` and `preflight.rs` were skimmed only (headers, unwrap/allow scans). I saw no major problems.
- `execution_planner.rs` was read for its enums and the surcharge function only.
- I did not verify W2-001's runtime impact against the pinned gen-core `ComponentPrecisionFloor` declarations.

**W3**

- `iris_training.rs`, `iris_task_training.rs`, `iris_curricula.rs`, `iris_restore.rs`: skimmed headers and function lists only; no line audit of the training logic. `iris_dist.rs` was read for its protocol and timeout design (looked sound: bounded graces, reap-on-drop).
- `checkpoint_catalog_migration.rs`, `checkpoint_weight_facts_host.rs`, `external_library_runtime.rs` (production half), `resolved_cache_promotion.rs`, `converted_install.rs`, `local_derivation.rs`, `refine_model_cache.rs`: structural skim only; they looked well-factored and carry confinement checks.
- `model_jobs.rs` conversion section (`resolve_convert_plan`, `run_model_convert_job`, the per-family `convert_*` cfg triplets) and `run_model_import_job` past the staging setup: not read line by line. Test modules in all files were surveyed structurally only.
- Whether image-job lanes outside this area still call `ensure_hf_cached_file`/`ensure_cached_file_verified` at job time contrary to the epic-17625 rule (`downloads.rs:446-470` doc says no job-time weight downloads). `job_time_download_guard.rs` presumably allow-lists them; not checked.

**W4**

- `yue2_admission.rs` (2,866 lines) and `yue2_memory_profile.rs` (test-only) read only for gating/structure; no line audit.
- `catalog_parquet_scanner.rs` and `catalog_image_fetch.rs` bodies beyond lease/limits/SSRF entry points (SSRF handled by `downloads::fetch_public_source_url_bytes`, not reviewed here).
- Interior of `vector_jobs.rs` SVG sanitizer correctness (security of rules not verified), `audio_jobs.rs` streaming synthesis internals, `media_jobs.rs` timeline/ffmpeg filter builders, `voice_register.rs`, `moss_terminal_observation.rs`, `llm_kv_cache.rs`, `qwen_prompt_rewrite.rs` (skimmed only).
- `*_v1.txt` prompt files: mood-board and single-image variants share large boilerplate by design; judged intentional.
- Lock-across-await analysis was not done exhaustively.

**W5**

- `person_segment_sam3.rs` (1511 lines), `face_likeness.rs` (1306), `upscale_jobs.rs:1315-1676` (the image/dataset job bodies), `pose_jobs.rs:1206-1410` and `iris_restore.rs` were sampled or skimmed, not line-read. `person_track.rs`, `person_replace.rs`, `qwen_alpha.rs`, `openpose_skeleton.rs`, `job_metrics.rs`, `mlx_worker_recovery.rs` and `bin/starvector_terminal_*.rs` got only the unwrap/dead-code/TODO sweep.
- `job_time_download_guard.rs` (3,928 lines): read the header and constants only. I did not verify that the hard-coded ratchet counters (`JOB_TIME_DOWNLOAD_SITES_REMAINING = 41`, `WEIGHTS_CACHE_DESTINATIONS_REMAINING = 25`) match the tree. I did not run `cargo test`.
- Unit-test bodies (`*/tests.rs`) were surveyed structurally only.

**C1**

- `checkpoint_plan_store.rs` (2,352 lines): read only the path and lock helpers; the plan, approved-root and inventory logic was not reviewed.
- `converted_artifacts.rs`, `download_receipt.rs` beyond the first ~200 lines: skimmed, not audited.
- `jobs_store/routing/*` (16k lines) and `jobs_store/tests/*`: structure only.
- `project_store.rs` film/draft/timeline method bodies (roughly lines 900-2700) and the 8,000-line test module: only grepped.
- `catalog_store.rs` query/facet/scanner code beyond open, configure, migrate and the lease.
- `training_store.rs` and `voice_store.rs`: only SQLite opener and the transcode twins.

**C2**

Not line-audited: film_compile.rs, film_review.rs, film_timeline.rs, film_planner.rs (beyond imports), dataset_quality.rs, memory_calibration.rs, sdxl_cuda_memory_endpoints.rs, mlx_tier_completeness.rs, workflow_png/mp4/resolution/parameters, moss_image*.rs, supra2.rs, yue2_score, checkpoint_import/derivative/weight_facts internals, credentials.rs, depth_map.rs, angle_kps.rs, ideogram_caption.rs, resolved_cache*, model_artifacts*. The ~1.8k-line `SIGNATURES` table in lora_family.rs was spot-read, not validated. training/iris.rs not read.

**A1**

- Test bodies in `models.rs` (~7.4k lines) were surveyed structurally, not line-audited.
- Detailed logic of `install_state_for` (`models.rs:7692-7943`), `apply_model_catalog_entry` and the MLX tier probes was not verified.
- `loras.rs` import and validation internals (`validate_lora_specs_for_model`), `recipe_presets.rs`, `prompt_batches.rs`, `yue2_scores.rs`, `iris_curricula.rs` and `iris_training.rs` beyond the sampled regions.
- `catalog_scan_supervisor.rs` was grepped only (panics, locks), not read.
- Worker-side re-validation of `uploadedSourcePath` moves/deletes (`paths.rs:726-760`) was not traced.

**A2**

- Line-level audit of `film_harness.rs` (6.5k), `film_harness/review.rs`, `film_planner.rs`, `film_planning.rs` beyond security/blocking/dup greps.
- `jobs.rs`/`generation.rs` payload canonicalization logic correctness (only skimmed function sizes, the redaction convention, and worker ownership checks).
- `startup.rs`, `timelines.rs`, `workflows.rs`, `control_overlays.rs`, `person.rs`, `keypoints.rs`, `image_restoration.rs`, `manifest*.rs` read only at the grep/structure level.
- Core-side `is_safe_relative_path` and upload extension allow-list (outside area; only skimmed).
- `#[cfg(test)]` modules and all of `tests/` (surveyed for helper duplication only).

**X1**

- `mlx.rs` lines ~1-3900 and 12k-34k were sampled by structure, signature and twin diff only; I did not line-audit any individual arm's logic or the LTX campaign/canary scaffolding (`mlx.rs:13758-17940`, `mlx_ltx25.rs`).
- `lib.rs` (3.9k lines) was surveyed for unwrap/expect/TODO only. There are no non-test `unwrap` calls beyond two documented `expect`s at `lib.rs:585-586`.
- `candle.rs` and the `*_iris`/`*_supra2`/`*_wan_scail2` files were compared to their MLX twins by function-body hashing and spot diffs, not read in full.
- `apps/desktop/src/setup.rs` (5.6k lines) was read for bind/auth, gate_window, and the provisioner hooks. Sidecar PID/reap logic, preflight and Metal/CUDA checks were not audited. `webview_resilience.rs`, `temp_fixture_guard.rs` and `settings.rs` GPU-info code were not read.
- `crates/sceneworks-mcp/src/server.rs` compact_* mappers and the tool arg-schema structs (`:150-540`) were not read line by line; `yue2.rs` was checked only for id and path handling.
- Vendored crates: I diffed only the manifests against `Cargo.toml.orig`; I did not diff the glib source against upstream or re-run `check-glib-variant-iter.mjs`.
- `docs/mcp-server.md` and `docs/memory-calibration-harness.md` were read only where cited.

**D1**

- Did not run `cargo build/test/clippy` or `cargo metadata` (no `target/`, git deps need network); the dependency graph comes from Cargo.tomls and Cargo.lock only.
- Parallel-table drift beyond the web seed: verified `EXPECTED_IMAGE_IDS` (`engines.rs:1808`) and `EXPECTED_VIDEO_IDS` (`pinned_engine_geometry.rs:71`) equal the manifest image/video id sets, and `jobTypesParity` covers the web/core job-type tables; did not diff `config/engine-capabilities/*.json`, `config/backend-capabilities/matrix.json`, or the `routing/catalog.rs` `EXPECTED_*` lists against the manifest/web `candleGating.js` row by row.
- `packages/schemas` enum parity with Rust enums (sampler/scheduler/tier keys) was only spot-read, not diffed.
- Prior reviews: ~28 findings checked of ~170; the Low clusters (F-052..F-060) not re-verified.
- Functional correctness of the duplicates (e.g. whether worker `write_json_value` ever races in practice) is inferred from code, not reproduced.

**B1**

- `apps/web/src/screens/**`, `components/**`, `simple/**`, `training/**` beyond targeted greps (other reviewers). `LayersPanel.jsx` is noted in B1-008 only.
- Line-level audit of root domain modules (`workflowShare.js`, `tierSuggestion.js`, `yue2Lab.js`, `ideogramCaption.js`, `presetUtils.js`) and of `hooks/useTraining.js`, `useWorkflowDrop.js`, `useModelsAndLoras.js` internals; `App.jsx` 2327-2490 (hydration ledger) was skimmed, not traced.
- Test files, surveyed structurally only. `data/*.json` rows (styles, tags, thumbnails) were not audited; the style catalog to manifest derivation is drift-tested and was left alone.
- No `knip`/ESLint run (offline, no install). Dead-code results come from an `rg`/script importer scan limited to this area's exports, and exports used only by tests were not reported.

**B2**

- ImageStudio JSX L3204-4846 and the many MOSS/Iris/Krea branches were only sampled, not line-audited.
- `imageEditor/useMaskTool.js`, `useBoxesTool.js`, `useColorGradeTool.js`, `ImageEditorToolPanel.jsx` (1.4k) internals and `*Math.js`/`maskStroke.js` were not read beyond the lifecycle grep. They have no createObjectURL/listener/timer calls.
- CharacterStudio, VectorStudio, DocumentStudio, ReplacePersonPanel, PoseLibraryScreen (object-URL handling at L174/209/222/285 looked balanced) and KeyPointLibraryScreen were only grep-checked.
- Konva `key={index}` at ImageEditor.jsx:4142/4204 (crop overlay rects, mask lines) look safe (fixed/append-only lists).
- Did not verify bundle impact of Konva or test files.

**B3**

- `simple/` audio and video UIs, `components/audioTakeParts.jsx`, `components/assetPanels.jsx`, and the body of `AudioStudio.jsx` (only structure and greps).
- Line-level review of `VideoStudio.jsx` effects and `yue2Lab.js` logic. `FilmReferences`, `FilmShots`, `FilmPlanning` and `GenerationRail` were not read.
- Test files were surveyed only by grep. Whether the earlier docs/code-review web findings in this area are fixed was not checked.

**B4**

- `ConfigureJobPanel.jsx` (1452 lines), `DatasetDoctor.jsx`, `Iris*` option panels: only grepped for hardcoded ids and numeric inputs, not line-audited.
- `training/irisOptions.js`, `irisTaskOptions.js`, `irisCurricula.js`, `datasetHelpers.js`, `trainedLoraOffer.js`: parity with the Rust Iris validation not compared.
- `CharacterStudio.jsx` lines 520-998 and most of `characterPanels.jsx` (Looks, Loras, Test, LikenessCompare) only skimmed; `KeyPointLibraryScreen.jsx` Collections panel not read.
- `DatasetCatalogsScreen.jsx` lines 830-1379 (polling loop, mutation helpers) read only at the top; no CSS dead-code pass.
- No tests run; no profiler data.

**B5**

- `ModelManagerScreen.jsx` lines ~1420-1820 (model card JSX body) and ~1866-2110 (LoRA row JSX) were skimmed, not line-audited. `PresetManagerScreen.jsx` JSX from ~700 onward was surveyed.
- `DocumentStudio.jsx`, `ReplacePersonPanel.jsx` (past the draft-reseed logic), `audioResults.jsx` and `audioYueControls.jsx` were only grep-audited for effects, listeners, object URLs, HTML sinks and fetches. No problems turned up, but they got no logic review.
- Test files (`*.test.jsx`) were not audited. `imageStudio/` helper files were not reviewed. `components/` (including `LicenseGateNotice.jsx`) was read only where it affects these screens.
- No lint or knip run. Importer checks were by grep.

**B6**

- Line-level review of generationStudio.jsx (1248 lines), assetPanels.jsx (1496), AssetPicker.jsx (1109) and WorkerProgressCard.jsx (996). Only the dropdown and Escape handlers, fetch sites and headers were read.
- Duplication between WorkerProgressCard and SimpleJobCard progress UI, and between SimpleLoraSheet and LoraPickerField (not diffed).
- `components/editor/*` Film* components (only grepped).
- Whether the dead-CSS candidates are referenced from `index.html` or non-src files.

**S1**

- Line-level audit of the ~139k LOC of scripts (read structure, headers and cross-file duplicates only); the four largest (measure-memory-catalog, epic-20738-terminal-cuda-harness, run-ltx-safety-canary, kv-baseline-harness) were not read end to end.
- Field-by-field comparison of each JSON schema against Rust serde DTOs (only consumer wiring checked); the Rust side has its own contract tests (`checkpoint_import_contracts.rs`).
- tests/ pytest harness internals (9.7k LOC) beyond file inventory; apps/web/scripts/generate-style-thumbnails.mjs (613 lines) and vite-plugin-bundle-report.js not read in detail.
- No `npm audit`/lockfile freshness check (offline, no install).


### Per-reviewer systemic observations (verbatim)

**W1**

- The MLX/Candle split is implemented by copy: the same tier clamp, quant parse, adapter loop and model-dir resolver exist once per backend, often with a doc comment saying "mirrors X". Where they have diverged (Wan default tier, Z-Image telemetry keys, video LoRA pinning) the divergence looks accidental because it is documented nowhere. A shared backend-neutral helper layer is the highest-leverage fix.
- Many fixes are applied to the image side and not carried to the video side (LoRA pinning, `quant_int` use, `Prepared*` sources). Video is consistently one hardening generation behind.
- The module layout still relies on `include!` and large aggregation files, and cfg-gating is done with inverse `allow(dead_code)` instead of gating items to their callers. This hides real dead code and makes the 20k-line files hard to navigate.
- Most previously reported items are genuinely fixed and carry sc-/F- citations in comments. The remaining structural debt is readability and duplication rather than correctness or security.
- Test and harness scaffolding (HF-cache discovery, env isolation, catalog-entry loaders) is copied across 15 or more files although `test_env.rs` and `smoke_support.rs` exist as the natural home.

**W2**

- The MLX and Candle lanes are intentionally separate, but the shared pieces (evidence construction, parameter selection, GenerationMemory translation, budget decisions) were copied rather than extracted. Each copy has drifted slightly (W2-001, W2-002, W2-005).
- Policy values are re-declared per module with a "mirrors X" comment (headroom, reserve, GiB, revisions) instead of aliasing one typed constant. Equality is checked by neither the compiler nor tests (W2-004, W2-010, W2-011).
- Hand-maintained stringly-typed provider lists are spread across gate code, the route registry, the model table and the manifest. The defaults fall through permissively instead of failing closed (W2-012).
- Concurrency and process hygiene is good where a helper exists (`run_bounded_command`) and absent where it was bypassed (W2-008). Making the helper the only allowed path, with an `architecture_tests` guard like the existing ones, would close the gap.
- Gate source files have grown to include tens of thousands of lines of inline tests, which hides both duplication and dead code (W2-015, W2-016).

**W3**

- Download correctness is solid in its parts (timeouts on every client, stall watchdog, SSRF pinning, `safe_join`, atomic HF promotion), but it is implemented as parallel stacks (HF snapshot, HF single file, source URL, public dataset URL). Hardening landed on one stack at a time (`no_proxy`, DNS timeout, `.partial` promotion, size caps), which is the root cause of W3-001, W3-002 and W3-006.
- Trust-boundary validation is applied at job entry points (`validate_hf_download_inputs`, `resolve_*_target`, `safe_download_dir`) but not uniformly to values read back from responses (etag, `Link` header, COCO `file_name`). Treat server responses as untrusted in the same helper layer.
- Blocking filesystem or hash work on the async runtime without a heartbeat keepalive is a recurring failure class. It was fixed for `sha256_file` and the control-prep path (sc-13856, sc-11155) but not for the prepared-bundle preflight.
- Test scaffolding (Settings literals, HF cache env isolation, axum API stubs) is copy-pasted widely; a shared `test_env`/`support_stubs` pass would remove several hundred lines and a class of "new field breaks 16 files" churn.
- Giant files (`model_jobs.rs` 9.8k, `training_jobs.rs` 10.5k, `downloads.rs` 3.4k with ~1k lines of tests, `external_library_runtime.rs` 3.6k with 2.4k lines of tests) mix several concerns and interleave test modules; splitting by concern would make the duplicated logic above easier to see and share.

**W4**

- The worker has a strong shared layer (`progress.rs`, `analysis_jobs_common.rs`, `payload.rs`), but newer handlers copy bodies instead of calling it, then paste the sc-/F- rationale into each copy; the shared module headers overstate their coverage.
- Cancel semantics (defer terminal `Canceled` until GPU work joins, treat shutdown as cancel) are correct in the shared loop and drift in hand-rolled loops, so cancellation correctness is the main recurring risk.
- Path confinement and hashing helpers each exist in 2-4 variants with different strictness; consolidating into `sceneworks-core` would also remove worker/API divergence.
- Per-platform `cfg` plumbing is the largest readability tax; a single build-script cfg alias would remove ~1,900 predicate copies.
- Files over ~5k lines (`vector_jobs.rs`, `media_jobs.rs`, `audio_jobs.rs`, `prompt_refine_jobs.rs`) hold test code plus multiple concerns; extracting self-contained engines (SVG sanitizer, ffmpeg runner) would be the cheapest win.

**W5**

- The recurring failure mode is a shared helper that is created but only partially adopted: `smoke_support.rs`, `analysis_jobs_common.rs`, `ort_cuda.rs`, `payload.rs`. Each leaves a long tail of verbatim clones that cite the original in a comment ("mirrors pose_jobs::build_session") rather than importing it.
- `use super::*` (42 non-test modules) plus `pub use paths::*`/`error::*`/`settings::*` at the crate root makes every module's real dependencies invisible. The download-guard header itself lists glob import as the reason a type-level gate was impossible. Moving to explicit imports would make the dependency graph and the cfg set reviewable.
- Safety-critical invariants (poison reset, path confinement, CUDA-EP honesty, heartbeat keepalive) are enforced by comments and source-text tests rather than by a single type or helper. When the rule changes, each copy must be found by grep.
- Cross-crate duplication (worker / rust-api / core / memory-adapter) exists for pure utilities that core already owns, and one duplicate's doc comment asserts byte-for-byte equality that nothing checks.
- The `lib.rs` crate root carries ~1,000 lines of module declarations with narrative history comments (sc-NNNN stories). The comments are valuable context but belong in module docs and a CHANGELOG, not in the file a reader must scroll to find the first `fn`.

**C1**

- Cross-cutting helpers are hoisted in one place and then re-copied by neighbouring modules (atomic write, safe id, relative-path validation, flock). Every earlier fix cites an sc-number, but there is no lint or architecture test that forbids the next copy. The worker `architecture_tests.rs` rules could be extended to core for these.
- The SQLite stores have no common "open, configure, migrate, begin write" entry point. Each store makes its own choices about WAL, synchronous, transaction behavior and migration style, and the safest (catalog and jobs: `BEGIN IMMEDIATE`) is not the one used by the store that two processes share most often (project.db).
- Typed enums exist (`JobStatus`, `StringEnum`) but consumers keep string-slice twins. Drift risk is concentrated in status handling and SQL text.
- `project_store.rs`, `catalog_store.rs` and `jobs_store.rs` are each several thousand production lines; the decomposition done for jobs routing has not been applied to the other two.
- Most `expect`/`unwrap` in non-test code in this area are guarded by a preceding check and documented; no obvious panic-on-user-input was found in the files read.

**C2**

- Identifiers for model families/ids are string literals in many tables (lora_family, base_weights, inspector, film_plan, video_request, h3_*), each hand-synced with the manifests; a typed family registry derived from the manifest would remove most drift risk.
- Core has no shared primitives for header parsing, file hashing, ffmpeg spawning, manifest access or slugging, so each module grew its own copy (C2-003/004/005/012/015).
- Validation placement is inconsistent: video has a full reject-never-clamp gate set in core, images clamp in the parser and gate via files named for video; API and worker re-check separately.
- Large modules mix production code with 50-70% inline tests (video_request 6.9k, lora_family 7.1k, film_plan 6.9k, training 5.1k); splitting `*_tests.rs` out (as model_usage_policy_tests.rs does) would make the real surface reviewable.
- Production lib exports test-support registries (workflow_share builders) and the test tree has no common helper crate.

**A1**

- `use super::*` in 44 of 61 rust-api modules turns `lib.rs` (6.3k lines, ~146 free functions) into an implicit shared namespace. This is why helpers like `pattern_matches`, `format_bytes` and `validate_lora_import_source_path` live in the root and get re-copied into the worker rather than going to core. A rule of "no new shared helpers in lib.rs; add them to core" would stop the bleed.
- Path and URL confinement is done well case-by-case (canonicalize+starts_with, `sanitized_upload_filename`, owned-artifact roots, worker SSRF checks), but it is not centralised. There are about eight relative-path predicates, four repo-id validators and two local-peer guards, and they have already drifted (A1-002, A1-006, A1-001).
- The most recent hardening (local-only on model/checkpoint library, sc-4202 `spawn_blocking` in preferences) was applied per-module and did not reach siblings (catalogs, credentials, training uploads). Applying such a fix needs a crate-wide grep for the same shape.
- `AppState` and handler bodies carry deliberate test-only hooks (`server.rs:~458-500`, `catalogs.rs` ×49 `cfg(test)`, `models.rs` ×35). The design is intentional and documented, but it pushes the longest functions in the crate to be even longer.
- Domain-level 404/typed-error construction is hand-rolled in many places (`ApiError { .. }` 23 times for NOT_FOUND alone). A couple of constructors would remove noise and make audits of status codes grep-able.

**A2**

- Auth architecture is sound for token holders: single global middleware, five-entry default-deny allow-list, constant-time digest compare, per-IP throttle, method-aware exemptions. The weakness is the *trust model around loopback* (no Origin/Host check) and fail-open handling of non-`/api` paths.
- Dead code is clean in this area: no `#[allow(dead_code)]`, no TODO/FIXME, no unreferenced non-test functions found. Remaining debt is duplication, not leftovers.
- Helpers keep being duplicated across rust-api/worker/core and held in sync by comments; core should own every on-disk-contract helper (`safe_download_dir`, glob filter, atomic write, time/hash).
- `use super::*` plus a monolithic lib.rs is the main structural debt; previous extractions (sc-8890, sc-9736) stopped before the big blocks (validators, redaction, file serving, router).
- Error handling is split between sanitized (Catalog) and pass-through (everything else); one policy in `ApiError::internal` would fix ~200 sites.

**X1**

- The cross-lane protocol crate (`lib.rs`) exists, but MLX/Candle twins keep re-growing in the two binaries. Whenever a helper is needed by both lanes, the habit is to copy it; the Cargo.toml comments show this already caused one CI incident (sc-22738). A rule such as "backend-neutral logic goes in `protocol`" would stop the drift.
- "Defense X is on because Y gates it" reasoning needs a threat-model check. The MCP wildcard-bind exemption assumed a rebinding browser is a remote peer, which the loopback-trust setting makes false (X1-001). The same blind spot is probably present in other rust-api routes.
- Timeout hygiene was fixed in places that got a finding (MCP client, GPU provisioner) but not in sibling code paths of the same class (updater download, X1-003; the transcode shell-out, X1-019).
- Security-relevant primitives (safe-id check, SHA-256-verified download, archive extraction, Tauri IPC confirmation) are each implemented several times in slightly different ways (X1-005, X1-010, X1-011). Centralizing them would make future audits shorter.
- Test-support code is the other main duplication source (hand-rolled temp dirs, source-scraping tests); the desktop crate has already moved to `tempfile` and the other crates should follow.

**D1**

- The codebase's dominant failure mode is "helper hardened in one crate, copied unhardened elsewhere": the cause is `pub(crate)` visibility on core utilities (`store_util::atomic_write`, `random_hex`, `is_safe_relative_path`) and no core `hashing`/`env`/`fs` module. Each cluster above (D1-001..004, 009) is fixable by promoting 5-6 functions to a public core module.
- Duplication is increasingly "documented instead of removed" (byte-identical `safe_download_dir`, port of `ltx_frame_count`, transcribed tables with pin tests); the pin tests are good, but the premises in their comments (D1-007) have gone stale faster than the code.
- Drift guards are strong where they exist (job types, family allow-lists, image-axes, geometry, version alignment) and absent for the next layer down (web fallback capabilities/samplers, crate dependency direction, env-flag semantics).
- `image_jobs` (include! ladder, 98.8k lines) and `memory-adapter/src/bin/mlx.rs` (33.8k lines) are the two structural outliers; both keep growing and drive the cfg-trap guidance in CONTRIBUTING.
- Doc prose around CI/merge-queue/gate teardown lags behind the (deliberate) tooling changes; CLAUDE.md/FEATURE_DEVELOPMENT.md are current, CONTRIBUTING.md and per-crate READMEs are not.

**B1**

- Earlier web findings (F-007, 008, 009, 053, 054, 055, 056, 057, 058) are verified fixed. The remaining work is structural: App.jsx grew 80% since F-052 by absorbing features instead of hooks.
- The recurring pattern is "extract once, don't migrate the rest": `pollJobToCompletion`, `useScopedCrudList`, `useAppStatic`, `testUtils/dom.js` and `api/` all exist but cover only a fraction of their potential callers (2 of 9 hooks, 15 of 45 context consumers, 4 of ~40 endpoint domains). Each fix should finish the migration or add a lint rule that prevents the old form.
- Tests are the main consumer of several large data modules (`modelCatalog.js`, `previewSupportDerivation.js`, `entitlementsPlistContract.js`). That is fine when intentional, but a `src/__fixtures__/` or `test-support/` home would make the boundary visible.
- Auth and transport hygiene is good: single token seam, ticketed SSE/media, abort-aware loops, 401 gate. The remaining holes are bypasses (raw `fetch` in `api/yue2.js`) rather than design gaps.
- Recovery and polling loops (gate probe, media ticket, SSE reconnect, film timeline refresh) are each hand-written with their own timers. One small `useRetryLoop` or `createBackoff` utility would remove the duplication and make the unbounded cases, such as B1-004, easier to see.

**B2**

- Listener/object-URL hygiene in this area is good: window listeners are ref-bridged and removed (L2146-2149, 3401-3404), `ResizeObserver` is disconnected, layer URLs are revoked on delete/replace/unmount (`revokeLayerUrls`), the workflow-inspect effect aborts, and the mask loader has generation + abort. The remaining leaks are on error and race paths (B2-007, B2-008).
- Growth continues after the July decomposition findings: ImageStudio 3.2k to 4.9k and ImageEditor 4.1k to 4.5k lines. The extractions done so far (tool hooks, `useStrictControlState`) are shallow; the state, effects and handlers remain in one closure, and a per-model reset-effect web (6 `exhaustive-deps` disables, mostly justified by comments) keeps the coupling.
- Cross-surface duplication is mostly of "small copied helpers plus control clusters" (samplers, tier eligibility, resolution defaults, preflight guards), not of whole components. Shared modules exist (`samplerOptions.js`, `quantTier.js`, `imageJobRequest.js`) but the shared layer sits partly under `simple/`, which forces mirrors instead of imports.
- Exports-for-tests-only from screen files (about 40 in ImageEditor.jsx) are a symptom of pure logic living in the component module; moving it out would also shrink the 4.5k file by about a quarter with no behaviour change.

**B3**

- The recording and AudioContext lifecycles (`MicrophoneTake`, `timelineAudioPlayback`, `audioWaveform` decode) are handled carefully: tracks are stopped, contexts closed, and `live` flags are present. The weak spots are polling, not media.
- Polling is ad hoc: at least 6 different `setInterval`/`setTimeout` loops in film and timeline code, with inconsistent abort, visibility, and backoff handling. A shared `usePolling` hook or SSE would remove most of them.
- The API layer is only partly adopted: yue2 and the film planner use `api/*.js`, while run, draft and timeline calls build URLs inline.
- Manual `live` flags in effects (about 10 in yue2) lack AbortControllers, so requests are not cancelled. A small `useAsyncResource` hook would remove the duplication.
- Dead audio code (B3-003) suggests features landing without a caller. A knip run in CI would catch this.

**B4**

- The best-engineered part of this area is the Rust-to-JS parity test for numeric bounds; the gap is that it covers constants but not the validation *rules* (ranges from `limits`, integer-ness), which is where the web diverges (B4-001).
- Screens build their own list/upload plumbing (sequential upload loops, per-file state sets, `selectedIds.includes`) instead of a shared `useFileImport`/`Set`-based selection; the same shape appears in training, pose, keypoint, character import and dataset-add (B4-003, B4-005, B4-006, B4-011).
- `useAppContext()` consumers (TrainingStudio, CharacterStudio, Pose create panel) re-render on every SSE tick while their big lists are not memoized; only DatasetCatalogs uses `useAppStatic` as the comment intends.
- Components keyed by selected entity are inconsistent: CharacterStudio is remounted by workspace and carefully seeds drafts, while `CatalogCuration` relies on an incomplete manual reset (B4-004). Preferring `key=` over reset effects would remove a class of stale-state bugs.
- The earlier reviews' stale-closure and unmount-guard fixes (F-005, aborts in readiness/subject-mask effects, `generationRef` in the catalogs screen) are present and well commented; remaining risk is concentrated in import/upload flows that do not take an abort signal.

**B5**

- The security posture in this area is good: zero HTML sinks, no postMessage or iframe, URL hardening via `safeExternalUrl`, write-only credential inputs. Dismantled or shipped fixes cite sc-/F- ids and are mostly real (F-033 Logs, F-079 URLs).
- Server-owned state keeps being re-implemented client-side with localStorage or hand-copied contracts: license acks (B5-002), the remote-password policy (`SettingsScreen.jsx:130-182`), `slugify` (B5-005), the preset workflow-segment tables, and the fallback resolution and duration lists in `PresetManagerScreen.jsx:46-95`. Each is a drift risk against the Rust side.
- The "small helper copied per screen" pattern (size label ×6, byte formatters ×5, slugify ×2, cache refresh ×2) suggests a missing `formatting.js` and shared-hook pass. Cleanup would be cheap and mechanical.
- Large screens use inline `renderX()` closures instead of memoizable child components. The big-context consumers (ModelManager, Library) pay for it on every live tick. `useAppStatic` versus `useAppLive` is only used by the lighter screens.
- Staged-capture flows (Pose, Key Point, Vector, Document) each hand-roll job creation and job watching, which is the same architecture-drift seam called out for the studios in the brief.

**B6**

- A shared `Modal` primitive exists with good focus handling, but the Simple UI built a parallel overlay stack, so a11y fixes land in only one of them.
- Small UI behaviours (outside-click dismiss, refine and rewrite flows, guide fetch, abort classification) are copy-pasted rather than hooked. A `hooks/` set of 3-4 primitives would remove most of the duplication in this area.
- Simple and advanced parity is maintained by duplicating logic, not by shared hooks. Model-specific capability checks are hard-coded in the Simple UI.
- Dead-code hygiene is mostly good for JS (only LayersPanel found), but CSS has accumulated about 130 orphan selectors.
- No XSS or unsafe-link problems found in this area.

**S1**

- Evidence/receipt machinery is the dominant weight: scripts/ is 139k LOC, mostly bespoke one-epic harnesses (starvector-terminal-*, moss-*, yue2-*, epic-20738). Shared primitives (hash, canonical JSON, exec, JSONC) were never extracted, so each harness carries its own.
- CI hardening is consistently applied (pins, permissions, same-repo guards, env-routed inputs); the few exceptions found are isolated and cheap to fix.
- Repo hygiene tracks evidence permanently in-tree (calibration frames, logs, zips); the cost is concentrated in a few directories and is easy to cap with an ignore rule plus an archive policy.
- Two parallel web build paths (rust.Dockerfile web-builder vs web.Dockerfile dev image) have no shared source of truth for "what the Vite config needs", which is how S1-001 arose; smoke tests prove liveness, not graph resolution.
- Machine-specific values (user home, /Volumes/Models) leak into scripts under scripts/ci; moving them to runner config would also make those scripts testable off the author's Mac.
