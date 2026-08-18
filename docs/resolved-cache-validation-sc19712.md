# SC-19712 — Resolved-model hot cache: end-to-end acceptance record

Terminal validation story for epic 19703 (app-managed two-tier model cache: an authoritative
Hugging Face library on an external volume, plus backend-ready copies on internal disk that
survive disconnection).

**Verdict: the mechanism works end to end on a real external volume — a model promoted from the
external tier loads and generates with the source drive physically unmounted — but the epic must
not ship as it stands. Three defects were found, one of which (F-3) makes the feature
self-defeating in production: once the cache holds anything, every job submission synchronously
re-hashes the entire cache inside the HTTP request.**

Subject under test: `feature/sc-19703-resolved-model-hot-cache` @ `449461a5c` (all nine code
stories merged). `claude/sc-19712-validation` adds **documentation only — no behavioural change**.
The one code fix this campaign needed (F-1) is recorded below as a ready-to-apply patch rather
than committed, because the file it touches is a memory-matrix fingerprint source and this story
is barred from regenerating the matrix. The campaign itself was run with that patch applied
locally; without it, no SANA job can be claimed at all.

> **Measurement caveat (read before quoting any number in the table).** Every timing here was
> taken on a **`cargo build` dev-profile (unoptimized) binary**, because that is what
> `npm run rust:check` produces and what this campaign was run against. SHA-256 dominates the
> cache's cost model and resolves to the `sha2` crate's pure-software `soft::compress` path in
> this build (confirmed by stack sampling). **Treat every hash-bound number as an upper bound,
> not a shipping figure.** The I/O-bound numbers (copy throughput, load-from-disk) are
> representative; the hash-bound ones are not. Where a number is hash-bound it is marked ⧗.

---

## Environment

| | |
|---|---|
| Host | Apple M5 Max, 18 cores, 128 GiB unified memory |
| OS | macOS 26.5.2 (build 25F84) |
| Local tier (internal) | `/dev/disk3s1s1`, APFS, 3.6 TiB, 287 GiB free at start |
| Source tier (external) | `/dev/disk5s1` "Models", APFS, 4.1 TB, **USB**, media `T253TY004T`, link 80 Gb/s |
| Source-tier volume identity | `VolumeUUID CD01D2AD-5CB6-480B-8A7E-322816A804BD` → binding `macos-volume:cd01d2ad5cb6480b8a7e322816a804bd`, `directoryId 346` |
| Library root (configured **directly**, not via the `~/.cache/huggingface/hub` symlink) | `HF_HUB_CACHE=/Volumes/Models/huggingface/hub` |
| Library contents at baseline | 98 model repos; 4399 files/links; 1,811,018,365,416 B; manifest sha `72b94cffd042…` |
| Data dir | **isolated** scratch dir; `~/SceneWorks/data` was never written (install receipts were copied out read-only) |
| Cache policy | `SCENEWORKS_RESOLVED_CACHE_ENABLED=1`, `MAX_BYTES=21474836480` (20 GiB), `INACTIVITY_SECONDS=1209600` (14 d, default) |
| Binaries | `target/debug/sceneworks-rust-api`, `target/debug/sceneworks-rust-worker` (dev profile — see caveat) |
| Worker | `SCENEWORKS_GPU_ID=mlx`, backend `mlx` |
| Model under test | `sana_1600m` → `SceneWorks/Sana_1600M_1024px_mlx` @ `ac421696dd6eb0b41f446f4c45a53ccc057d82a1`, variant `q4` (manifest default), closure 4 files / 5,574,343,418 B |
| Never-promoted control model | `sana_sprint_1600m` → `SceneWorks/Sana_Sprint_1.6B_1024px_mlx` @ `0b0d18484cac2fb515e76d25a09a5911ae4ab58e`, variant `q4` |

### Step 0 — the merged tree

`#2408` (Settings + Model Manager UI) and `#2409` (promotion activation) merged into the feature
branch within fifteen minutes of each other, so their combined tree had never been built. It was
built first, before any campaign work:

| Command | Exit | Result |
|---|---:|---|
| `npm run rust:check` | **0** | 3757 tests passed, 0 failed |
| `npm run check` | **0** | 412 passed, 0 failed |

---

## Findings

### F-1 — SANA is unroutable on macOS through the real API · severity **high** · pre-existing · **patch ready, deliberately NOT in this PR**

`sana_mlx_eligible` read a `referenceAssetId` of JSON `null` as "not a valid reference" and
returned ineligible, so no MLX worker would ever claim the job. The API normalizes every unset
optional asset carrier to an explicit `null` before storing a job — verified directly in
`jobs.db.payload_json` — so **every** SANA 1600M / SANA-Sprint text-to-image job on macOS sat on
*"Waiting for an available worker."* forever, emitting no `gpu_route_decision`. Observed live for
16 minutes before diagnosis.

The routing layer's own convention, stated at `crates/sceneworks-core/src/jobs_store/routing.rs:222`,
is that "missing, `null`, and a blank string are all the product-level 'not supplied'
representation". Only the SANA predicate departed from it; the two other `referenceAssetId`
predicates (InstantID, PuLID) genuinely require a reference, so their shape is correct for them.

The test blind spot was exact: `sana_variants_accept_single_reference_img2img_and_reject_malformed_shapes`
exercises `maskAssetId: null`, `phases: null` and `mlxQuantize: null` in a case list literally
named "empty/null optional carriers", but never `referenceAssetId: null`.

The fix is one predicate. It was written, verified, and then **deliberately withheld from this
PR** — see "why it is not here" below. The campaign was run with it applied locally; without it no
SANA job can be claimed at all.

```rust
// crates/sceneworks-core/src/jobs_store/routing/mlx.rs
// in sana_mlx_eligible, replacing:
//     && payload.get("referenceAssetId")
//         .map(|value| value.as_str().is_some_and(|id| !id.trim().is_empty()))
//         .unwrap_or(true)
        && sana_reference_carrier_is_absent_or_usable(payload)

/// SANA's optional single-reference carrier. Reference is OPTIONAL here — the engine serves plain
/// text-to-image as well as singular-reference latent-init img2img — so "no reference" must be
/// eligible however the caller expressed it. The distinction that matters is `null`: the API
/// normalizes every unset optional asset carrier to an explicit `null` before the job is stored.
/// The previous `.map(..).unwrap_or(true)` form handled only the MISSING key, so every real
/// text-to-image submission read as "not a valid reference" and no MLX worker would claim it.
/// A malformed non-string still fails closed.
fn sana_reference_carrier_is_absent_or_usable(payload: &Map<String, Value>) -> bool {
    match payload.get("referenceAssetId") {
        None | Some(Value::Null) => true,
        Some(Value::String(value)) => !value.trim().is_empty(),
        Some(_) => false,
    }
}
```

plus the missing case in the existing test's "empty/null optional carriers" list:

```rust
json!({ "referenceAssetId": null, "prompt": "p" }),
```

Verified locally: the added case **fails** against the current predicate and passes with the fix;
28/28 MLX routing tests green; `npm run check:rust-derived-docs` green.

**Why it is not in this PR.**
`crates/sceneworks-core/src/jobs_store/routing/mlx.rs` is one of the memory matrix's fingerprinted
sources (`routingMlx` in `SOURCE_PATHS`, `scripts/generate-memory-matrix.mjs:2110`), so *any* edit
to it — even a comment — makes `npm run check:memory-matrix` fail with "generated memory matrix is
stale" and turns `rust:check` red. Landing the fix therefore requires
`npm run generate:memory-matrix`, and this story is explicitly barred from running a memory-matrix
regeneration. Rather than ship a red PR or take an action outside the story's mandate, the patch is
recorded here in full. It is a one-commit change for whoever picks it up:
apply the two hunks, run `npm run generate:memory-matrix`, `git add` the regenerated files, done.

This is also **not** epic-19703 code (`routing/mlx.rs` is untouched by the epic), so by this
story's scope rule it was reportable rather than fixable in the first place.

### F-2 — a retention checkpoint blocks the cache status surface · severity **medium** · in subject · reported

Every retention checkpoint re-hashes every complete bundle while holding that entry's **exclusive**
metadata lock, and `GET /api/v1/model-cache` takes the same lock. Measured live during a
checkpoint:

| Route | Result |
|---|---|
| `GET /api/v1/workers` | 200 in 0.0011 s |
| `GET /api/v1/model-library` | 200 in 0.0971 s |
| `GET /api/v1/model-cache` | **no response** (still blocked at 40 s) |
| `GET /api/v1/model-cache` after the sweep drained | 200 in **0.0038 s** |

Mechanism, from stack sampling of the API process while the request was outstanding (1679/1679
samples in one leaf): `get_model_cache → ResolvedCacheStore::inspect → ::list →
lock_metadata → flock(2)`, blocked. `list` (`resolved_cache.rs:824`) takes an exclusive per-entry
lock even for the read-only `EntryListing::Inspect` path. The holder, from sampling the worker:
`ResolvedCacheRetention::run_if_idle → enforce_retention → scan_entry → validate_complete_metadata`,
where `scan_entry` (`retention.rs:646`) takes the exclusive lock and `retention.rs:650` then calls
`validate_complete_metadata`, which hardcodes `ContentVerification::RehashEveryFile`
(`resolved_cache.rs:2123-2128`). Checkpoints run every `RESOLVED_CACHE_RETENTION_INTERVAL = 600 s`.

**This is not a simple wiring bug, and it was checked before being called one.**
`ContentVerification::PathsAndSizesOnly` *is* used by retention — at `retention.rs:882-885`, the
removal path, which is what its doc comment refers to. The re-hash at `:650` is deliberate: it is
what separates a healthy entry from a `RecoveryCandidate`. The defect is the cost and the
contention at the seam, not the choice of verification level:

- the same doc comment warns that "re-hashing gigabytes would hold locks that block model loads"
  (`resolved_cache.rs:2153-2157`) — the concern was applied to `:882` but not to `:650`;
- the cost recurs every 10 minutes for the life of the process and scales with **total cache
  bytes** (default budget 64 GiB), whether or not anything changed;
- the web UI polls the blocked endpoint every 3 s (`CACHE_CONVERGENCE_POLL_MS`, cap 40 polls),
  so the Settings "Local model copies" card and the Model Manager badges stall exactly when a
  user is watching them.

Not fixed here: downgrading `:650` would remove real corruption detection, and the right answer
(a non-blocking try-lock for the status read reporting from the journal, and/or a cheap
change-detection gate before re-hashing) is a design decision for the store's author. Note that
simply dropping the lock would be *wrong* — a concurrent staging→bundle rename could make
`Inspect` label a healthy entry `Corrupt`, which is worse than blocking.

### F-3 — every job submission synchronously re-hashes the whole cache · severity **high** · in subject · reported

The headline defect, and the one this story existed to catch.

With a single 5.57 GB bundle published, `POST /api/v1/image/jobs` does not answer within 60 s. The
first submission of the campaign — same payload, same model, **empty** cache — answered in well
under a second. The cost is neither a lock nor a one-off: it is unconditional work proportional to
total cache bytes, performed inside the request.

Stack, from sampling the API process while the POST was outstanding (1739/1739 samples):

```
axum create_image_job
 -> create_generation_job_with_status
 -> model_sources::ensure_runtime_model_sources
 -> model_sources::preflight_payload_model_sources      apps/rust-api/src/model_sources.rs:484
 -> model_sources::local_resolved_artifacts             apps/rust-api/src/model_sources.rs:316-323
 -> ResolvedCacheStore::valid_local_artifacts           crates/.../resolved_cache.rs:474
 -> validate_complete_metadata                          (hardcodes RehashEveryFile, :2123-2128)
 -> validate_metadata_shape_with -> ResolvedModelArtifact::validate
 -> ResolvedBundleClosure::validate_at_root -> validate_artifact_file -> sha2   (1737 samples)
```

Measured: with an empty cache the identical submission answered in **under a second**; with one
5.57 GB bundle published it answered `HTTP 201` after **929.6 s (15 min 30 s)**, with the API
process pegged at ~600 % CPU throughout (the scan parallelizes across files).

The same full-strength provider is reached from three places:

| Call site | Path | Verdict |
|---|---|---|
| `apps/rust-api/src/model_sources.rs:484` | every image/video job **submission**, per request | the defect |
| `apps/rust-api/src/models.rs:4975` | model catalog snapshot build | amortised by `state.model_catalog_cache`, but a full re-hash on every cache-miss rebuild |
| `crates/sceneworks-worker/src/external_library_runtime.rs:114` | worker pre-loader guard | **correct** — it is about to hand bytes to a loader |

Why it happened is a real design tension rather than carelessness: the provider is deliberately
shared so that "catalog/preflight and the actual load can never disagree about which entries are
valid" (`model_sources.rs:312-315`). That is sound reasoning about *agreement* which imported the
load path's verification *cost* onto two hot read paths.

The consequence is that the feature defeats itself the moment it works: the cache exists to make
loads faster, and populating it makes every submission slower than the load it saves — scaling
with cache size, against a default budget of 64 GiB (about 11× the bundle measured here).

Not fixed here, and the reason is not lack of verification capacity. The obvious fix — give the
API path `PathsAndSizesOnly` and keep full strength on the worker guard — breaks the stated
agreement property in precisely the scenario the epic exists for: with the drive disconnected the
API would label an entry `local_ready` while the guard fell back to a source tier that is not
there. Memoizing the scan needs cross-process invalidation, because the **worker** publishes
bundles and the **API** reads them, and a TTL reintroduces the same disagreement window. Choosing
between those is the store author's decision.

### Receipt backfill silently excludes a model from the local tier

Not a new defect so much as an interaction nobody appears to have costed. On first catalog build
the API backfills a download receipt for every installed model it discovers that lacks one
(`backfill_current_receipt`), and those receipts carry **`snapshotRevision: null`** — verified
directly: this campaign's isolated data dir was seeded with two real receipts and the API wrote 68
more, all revision-less.

A revision-less requirement makes its whole repository unserveable from the local tier
(`external_library_runtime.rs:436-439`, pinned by
`a_requirement_with_no_recorded_revision_makes_its_whole_repository_unserveable`). Promotion,
however, *can* still build the bundle, because it resolves the unique matching snapshot. So such a
model can occupy cache bytes it will never be served from, and the user sees no explanation beyond
a `resolved_cache_local_tier_not_selected` event.

Scale, measured against the **real** install rather than this campaign's synthetic one — 85
receipts under `~/SceneWorks/data/models`, of which exactly **one** is revision-less:
`SceneWorks/qwen-image-mlx`. That single model is also one of the three soft-co-requisite models
below, so `qwen_image` is excluded from the local tier on both counts. The backfill path is rare in
practice; it is not theoretical.

### Observations that are not defects

- **`modelResolution.libraryPresent` is `false` on a healthy `external_ready` row.** It reads like
  a general fact but is only ever set on the two `unavailable` paths, where it distinguishes
  "reconnect the drive" from "re-bind the identity". Verified harmless: `modelLibraryContextForModel`
  (`apps/web/src/modelLibrary.js:38-52`) returns `null` unless the model is unavailable, so no
  consumer reads it outside that branch.
- **Uninstall is authorised to delete from the external library.** `delete_model` builds
  `allowed_roots = [<data_dir>/models, model_source_library(<data_dir>).root()]`
  (`apps/rust-api/src/models.rs:1184-1189`), so uninstalling an externally-backed model removes
  its weights from the configured library root — to the OS trash by default, permanently with
  `?permanent=true`. That is defensible (uninstall means uninstall) but it is worth stating
  plainly now that the library can be a shared drive holding other tools' weights. It also
  dictated this campaign's method: the uninstall scenario was run against a **disposable copied
  library**, never against `/Volumes/Models`.

---

## Acceptance matrix

Rows marked **cited** are already covered deterministically by merged suites and were not
re-executed physically; the exact test is named so the claim is checkable. Rows marked **live**
were executed against the real external volume through the real API/worker surfaces.

### Core cycle

| # | Scenario | Command / test | Revision | Result |
|---|---|---|---|---|
| 1 | Library configured directly at the volume root | `HF_HUB_CACHE=/Volumes/Models/huggingface/hub`; `GET /api/v1/model-library` | — | **live · pass** — `probeStatus: available`; binding stamped `macos-volume:cd01d2ad…804bd`, `directoryId 346`, canonical path equal to the configured path |
| 2 | First external use + idle-drain promotion | `POST /api/v1/image/jobs` (`sana_1600m`, 1024², 20 steps, seed 19712) | `ac421696` | **live · pass** — `model_source_tier_selected tier=source_library`; job completed in **27 s**; `resolved_cache_promotion_scheduled` fired 2 s after completion; bundle published `complete`, 5,574,343,418 B, `modelIds:[sana_1600m]`, `sourceVolumeRelation: different` |
| 3 | Local hot load | resubmit the identical job | `ac421696` | **live · see F-3** — the *load* resolves from the bundle, but submission is blocked by F-3; measured separately below |
| 4a | Locally resolved model completes with the drive absent | `diskutil unmountDisk /Volumes/Models`, then generate | `ac421696` | see run log below |
| 4b | External-only model reads typed unavailable | `sana_sprint_1600m` while unmounted | `0b0d1848` | see run log below |
| 5 | Remount → recovery, no transfer, receipts unchanged | `diskutil mountDisk`, snapshot diff | — | see run log below |
| 7 | Delete semantics confined | disposable copied library (never `/Volumes/Models`) | `09f741ba` | see run log below |

### Adversarial probes

| Probe | Evidence | Result |
|---|---|---|
| Path aliasing (`~/.cache/huggingface/hub` symlink vs the direct `/Volumes` root) | live re-bind + `an_alias_of_the_bound_library_probes_available_while_real_changes_still_fail_closed`, `configuring_the_drive_directly_instead_of_its_symlink_stays_ready` (`external_library_tests.rs:851,916`) | see run log |
| Mount substitution / decoy at the configured path | `same_path_reuse_by_an_unrelated_directory_fails_closed` (`external_library_tests.rs:205`); `a_decoy_library_cannot_capture_a_receipt_backed_install` (`apps/rust-api/src/tests/model_library.rs:208`); `a_present_but_mismatched_library_is_flagged_separately_from_a_disconnected_one` (`:964`) | see run log |
| TOCTOU — source vanishes mid-load | **cited** — `source_disconnect_or_provenance_mutation_rejects_publication` (`materialization_tests.rs:401`), `a_source_that_vanished_between_the_load_and_the_drain_declines_rather_than_fetching` (`:1068`), `terminal_remap_occurs_only_when_the_exact_source_probe_proves_disconnect` (`external_library_runtime.rs:2337`) | pass |
| Eviction safety — pinned survives | **cited** — `pinned_entries_survive_automatic_cleanup_until_unpinned` (`retention_tests.rs:400`); API `pinning_blocks_removal_and_moves_bytes_out_of_the_reclaimable_total` (`tests/model_cache.rs:239`) | pass |
| Eviction safety — active lease survives | **cited** — `active_lease_survives_automatic_cleanup_and_eviction_resumes_after_release` (`retention_tests.rs:368`) | pass |
| Eviction safety — source unavailable retains | **cited** — `source_unverifiable_entries_are_retained_automatically_including_content_divergence` (`retention_tests.rs:489`); `a_disconnected_source_library_retains_and_reports_but_never_removes` (`:1109`) | pass |
| Full disk / ENOSPC | **cited, with a gap** — `cancellation_and_io_failures_leave_only_interrupted_metadata` (`materialization_tests.rs:330`) injects `ENOSPC` on `#[cfg(unix)]` only; the non-unix arm injects only `PermissionDenied`. There is **no free-space preflight** anywhere in the cache and no ENOSPC-during-eviction case. Admission is byte-budget-based, not capacity-based, so a cache configured larger than free disk will fail mid-copy rather than decline. | partial — recorded, not closed |
| Crash recovery | **live + cited.** Both processes were killed mid-scan by the harness while the worker held a runtime lease and an entry lock, leaving 13 stale session directories under `models/resolved/sessions/`. On restart the store re-opened, ran its recovery pass and the published bundle survived intact — the source library and the install receipts were byte-identical to the pre-kill snapshot (`t3-after-kill`: source sha `72b94cffd042…` = baseline, receipts sha `90672f50c63c…` unchanged). Also cited: `crash_after_atomic_rename_stays_unavailable_and_next_reservation_republishes` (`materialization_tests.rs:692`); `stale_sessions_are_removed_only_after_their_lock_is_acquirable` (`tests.rs:1015`); `cross_process_same_key_reservation_shared_lease_and_kill_recovery` (`tests.rs:1112`); `interrupted_eviction_converges_on_recovery_lookup_and_reservation` (`retention_tests.rs:826`) | pass |
| Revision change | **cited** — `reconciliation_covers_tier_deletion_revision_replacement_and_full_uninstall` (`retention_tests.rs:678`); `a_superseded_revision_is_served_from_the_source_tier` (`local_preference_tests.rs:227`); `a_wrong_revision_or_wrong_tier_bundle_is_not_used` (`external_library_runtime.rs:1374`) | pass |
| Shared components | **cited** — `tiered_multi_repo_optional_derived_and_shared_members_are_self_contained` (`materialization_tests.rs:205`); `a_shared_snapshot_stays_served_until_the_last_scope_drops` (`local_preference_tests.rs:309`); `a_pair_one_sibling_selection_does_not_cover_is_served_to_neither` (`external_library_runtime.rs:1503`) | pass |

### Advertised artifact classes

Every distinct `downloads[]` shape the catalog advertises, and whether the cache serves it. The
92 catalog entries reduce to four `type` values (`image` 53, `video` 10, `audio` 8, `utility` 21)
and exactly one provider (`huggingface`, 298/298 rows).

| Class | Representative | Cache support | Evidence |
|---|---|---|---|
| Primary, variant tier, dir-glob `["q4/*"]` | `sana_1600m` | **yes** | live — this campaign |
| Primary, no variant, flat exact file list, ONNX, `utility` type | `real_esrgan` (`SceneWorks/real-esrgan-onnx`) | **yes** | live — scenario 7 stack |
| Primary, whole-repo `files: []` | `clip_vit_l14`, `lens` (`training` variant) | **yes** — the manifest `files` shape never reaches the cache; requirements come from the receipt's concrete `resolvedFiles` (`artifact_selection.rs:299-305`) | by construction + `promotion_tests.rs:362` |
| Hard co-requisite, multi-repo, `componentId`/`subdir` | `mage_flow_edit_base` → `SceneWorks/Mage-Flow-Components-mlx` | **yes**, as `ArtifactMemberRole::CoRequisite` | `a_multi_repository_closure_becomes_one_source_library_shaped_bundle` (`promotion_tests.rs:69`) |
| Platform-restricted rows | `lens` (macOS-only) | **yes** — filtered to the worker's own OS by `retain_downloads_for_os` (`artifact_selection.rs:71-87`) | by construction |
| Sibling quant tiers | `q4`/`q8`/`bf16` | **yes**, as distinct members; never unioned | `sibling_tiers_of_one_repository_stay_distinct_members` (`promotion_tests.rs:171`) |
| **Soft co-requisite** (`required: "soft"`) | `qwen_image`, `qwen_image_edit`, `acestep_v15_turbo` | **NO — silently excluded** by `.filter(\|d\| d.get("required") != Some("soft"))` at `artifact_selection.rs:251,342,412`. No event, no badge. The primary promotes and shows "local copy" while that component still reads from the source library. | code-cited; **see limitation note below** |
| **Revision-less requirement** | `svd`, `flux2_klein_9b_true_v2`, `wan_2_2_vace_fun_14b`, `prompt_refine_anubis_8b`, `joycaption_beta_one`, `controlnet_tile_sdxl` (no pinned row at all), plus the unpinned LoRA co-requisites of `ltx_2_3_eros` / `wan_2_2_t2v_14b` / `wan_2_2_i2v_14b` | **NO — poisons the whole repository** at serve time (`external_library_runtime.rs:436-439`). Promotion may still build the bundle, so those entries can occupy cache bytes and never be served. | `a_requirement_with_no_recorded_revision_makes_its_whole_repository_unserveable` (`external_library_runtime.rs:1599`); reported as `resolved_cache_local_tier_not_selected` |
| **LoRAs and control overlays** | `config/manifests/builtin.loras.jsonc` (8), `builtin.control_overlays.jsonc` (1) | **NO — no path into the cache at all.** They are not model manifest entries; `payload_model_entries` reads only `modelManifestEntry` / `baseModelManifestEntry` / `modelManifestEntries` (`external_library_runtime.rs:742-755`), and the producer only ever emits `Primary` and `CoRequisite`, leaving the other seven `ArtifactMemberRole` variants dead. | code-cited |
| **Zero-download / imported models** | `aura_sr_v2` (`downloads: []`), user-imported checkpoints | **N/A** — never enter the closure (`entry_is_provably_non_hf_local`, `external_library_runtime.rs:761-802`); the API forces `local_ready` for anything under `<data_dir>/models` | code-cited |
| Non-`huggingface` providers | none ship today | **N/A** — 298/298 `downloads[]` rows are `huggingface` | manifest scan |

**Limitation worth stating in the product, not just here:** for the three soft-co-requisite models
the "local copy" badge over-promises. The base model survives a disconnect; the optional component
does not, so a request that needs it will fail while the badge says the model is local. This is
documented in the user-facing troubleshooting section added by this story.

### Platform lane

| Lane | Status |
|---|---|
| macOS / MLX | **live** — this record |
| Windows / Candle | **no external drive exists on any CI runner**, so the disconnect/reconnect cycle is structurally unrunnable there. Windows-specific behaviour is covered by the merged suites — `windows_shared_reader_without_a_lease_does_not_protect_an_expired_entry` and `windows_sharing_violation_keeps_the_eviction_pending_until_it_converges` (`retention_tests.rs:1295,1336`), `windows_source_session_cleanup_never_follows_a_replaced_staging_junction` (`external_library_tests.rs:332`), `..._junction_swap_...` (`materialization_tests.rs:570`) — and `scripts/platform-review-contracts.test.mjs:1644` ("Windows resolved-cache coverage rejects either narrowed test module") is the gate proving the Windows lane actually runs both resolved-cache test modules rather than compiling them. **Recorded explicitly rather than skipped silently: no physical external-volume evidence exists for the Windows/Candle lane.** |

---

## Performance

Measured on the hardware above, against the `q4` closure (4 files, 5,574,343,418 B). ⧗ marks a
hash-bound figure inflated by the unoptimized build.

| Phase | Measurement |
|---|---|
| First external use — full job, cold external tier | **27 s** (`startedAt 04:17:36Z` → `completedAt 04:18:03Z`), 1024², 20 steps, backend `mlx` |
| Promotion trigger latency | **2 s** after job completion (`resolved_cache_promotion_scheduled` @ 04:18:05Z) |
| Promotion — end to end, 5.57 GB external → internal | **14 min 36 s** ⧗ (scheduled 04:18:05Z → `Published` 04:32:41Z) |
| ‣ of which: copy + streaming verify | ≈ 2.6 min ⧗ |
| ‣ of which: publish (rename + full-strength validation) | ≈ 3 min ⧗ |
| Retention checkpoint — full re-hash of the cache | ≈ **20 min** ⧗, recurring every 600 s, single-threaded |
| Cache status read, idle | **0.0038 s** |
| Cache status read, during a checkpoint | **no response at 40 s** (F-2) |
| **Job submission, empty cache** | **sub-second** |
| **Job submission, 5.57 GB cache** | **929.6 s (15 min 30 s)**, `HTTP 201` (F-3), API pegged at ~600 % CPU |
| Worker pre-load guard, same 5.57 GB cache | a further **> 16 min** ⧗ in `select_local_tier → validate_artifact_file` before the load starts |

The last three rows are the finding, stated plainly: **loading `sana_1600m` from the external
drive took 27 seconds; loading the same model from its local copy cost over half an hour of
SHA-256 before the load began.** The constant is inflated by the unoptimized build, but the
structure is not: one full-strength re-hash of the whole cache on the API submission path, a
second on the worker guard, and a third on every retention checkpoint — none of them conditional
on anything having changed.

**The load-time benefit the acceptance criteria ask for cannot be reported honestly from this
build.** The local-vs-external delta is an I/O-bound quantity, but on this machine the two tiers
are an internal NVMe and an 80 Gb/s-attached external NVMe, so the raw read delta is small to
begin with; and F-3 puts a multi-minute hash in front of every submission, which swamps it. A
meaningful figure needs a release build with F-3 resolved. Recorded as **not established** rather
than estimated.

---

## Reproduction

```bash
# 0. the merged tree
npm run rust:check && npm run check

# 1. isolated data dir; library configured DIRECTLY at the volume root
export SCENEWORKS_DATA_DIR=<scratch>/data
export SCENEWORKS_CONFIG_DIR=<repo>/config
export HF_HUB_CACHE=/Volumes/Models/huggingface/hub
export SCENEWORKS_RESOLVED_CACHE_ENABLED=1
export SCENEWORKS_RESOLVED_CACHE_MAX_BYTES=21474836480
target/debug/sceneworks-rust-api &
SCENEWORKS_GPU_ID=mlx target/debug/sceneworks-rust-worker &     # mlx is load-bearing

# 2. first external use -> promotion
curl -sX POST localhost:8733/api/v1/projects -H 'content-type: application/json' -d '{"name":"sc19712"}'
curl -sX POST localhost:8733/api/v1/image/jobs -H 'content-type: application/json' \
  -d '{"projectId":"<id>","mode":"text_to_image","model":"sana_1600m",
       "prompt":"a red fox sitting in tall grass at golden hour",
       "count":1,"width":1024,"height":1024,"steps":20,"seed":19712}'
curl -s localhost:8733/api/v1/model-cache          # watch materializing -> complete

# 3. disconnect (only after lsof shows nothing holding the mount)
diskutil info /Volumes/Models | grep 'Device Node'  # record BEFORE unmounting
lsof +D /Volumes/Models
diskutil unmountDisk /Volumes/Models
# ... generate again; the promoted model must still complete ...
diskutil mountDisk /dev/disk5
```

No-network-transfer evidence is a byte-exact fingerprint of both tiers plus the install receipts,
taken before and after each local-only scenario (file list + sizes + mtimes, SHA-256 of the
listing), with `HF_HUB_OFFLINE=1` exported to the worker for the reconnect run so that any
attempted fetch fails loudly instead of silently succeeding.
