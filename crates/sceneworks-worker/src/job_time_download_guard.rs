//! The regression gate for epic 17625: **no new job-time download, no new `<data_dir>/cache`
//! weight destination** (sc-17637, AC9 inherited from sc-17598).
//!
//! ## Why this exists
//!
//! Job-time weight fetching was never designed in. It was added incrementally, one plausible call
//! site at a time, until 24 of them existed across 19 files. Every one of those commits was locally
//! reasonable; the pattern is only visible in aggregate. A rule that lives in a comment regrows the
//! same way, so the rule lives here, as a test that reads the source.
//!
//! ## Why a source scanner and not the type system
//!
//! The obvious design — "make the resolver take no `DownloadContext`, and the compiler is the gate"
//! — does not work in this crate, for three independent reasons:
//!
//! 1. [`crate::downloads`] is glob-imported at the crate root (`use downloads::*;`), so
//!    `ensure_hf_cached_file`, `ensure_cached_file`, `ensure_cached_file_verified` and
//!    [`crate::downloads::DownloadContext`] are in scope, unqualified, in **every** module. Any new
//!    file can also spell them `crate::downloads::…`; several already do.
//! 2. `DownloadContext` is a plain struct with public-in-crate fields and no privileged constructor.
//!    Job-time sites build it as an inline literal. Removing a parameter from one seam does not
//!    remove the ability to construct the context at another.
//! 3. Decisively: the download helpers and all of their call sites are gated
//!    `any(target_os = "macos", all(not(target_os = "macos"), feature = "backend-candle"))`. The
//!    required `parity` CI lane is ubuntu-latest with default features, so **none of that code is
//!    even compiled there**, and the `candle` lane runs `cargo check`/clippy but never `cargo test`.
//!    A gate that inherits those cfgs never fires on any required check.
//!
//! So the gate is plain `#[cfg(test)]` and reads source **text**. It compiles and runs on Linux with
//! default features, on macOS, and on Windows — on every PR. Precedent is
//! [`crate::candle_preview_wiring_tests`], which exists for the same reason, and whose comment
//! stripper this reuses.
//!
//! ## What it asserts
//!
//! 1. [`job_time_download_sites_match_the_allow_list`] — the set of production files that reach a
//!    download primitive equals [`DOWNLOAD_SITES`], **exactly**. Not `⊆`: a migration slice that
//!    removes a site must ratchet the list down in the same commit.
//! 2. [`cache_destinations_match_the_allow_list`] — the set of `<data_dir>/cache/<subdir>`
//!    destinations across every workspace crate that can see the data dir equals
//!    [`CACHE_DESTINATIONS`], exactly, each entry classified and commented.
//! 3. [`downloads_fetch_surface_is_registered`] — [`DOWNLOAD_PRIMITIVES`] is not a hand-written list
//!    of four names. The whole externally-visible fetching surface of `downloads.rs` is rediscovered
//!    structurally and must equal `DOWNLOAD_PRIMITIVES` plus a commented [`NON_WEIGHT_FETCHERS`]
//!    list, so a *new* primitive cannot appear without a decision about which side it belongs on.
//! 4. [`http_client_construction_is_confined`] — the next regrowth shape after "do not call the
//!    helpers" is "do not call anything": `reqwest::Client::new().get(url).send()` written inline in
//!    a lane. Building a client or sending a request is confined to [`HTTP_CLIENT_FILES`].
//! 5. [`parameterized_cache_root_helpers_are_registered`] and [`cache_root_producers_are_registered`]
//!    — a helper that takes its cache subdir as a *parameter*, and a function that hands out the bare
//!    cache root, are the two shapes a literal-keyed allow-list cannot see through. Both registries
//!    are rediscovered structurally and both sets of call sites are scanned for what they pass.
//! 6. [`http_client_file_fetch_surfaces_are_registered`] — confining WHERE a client may be built says
//!    nothing about what those files then fetch. A `fetch_weights(url, dest)` added to `api_client.rs`
//!    is a complete job-time download that every other assertion here would miss, so each
//!    allow-listed file's network surface is registered too.
//! 7. [`scanned_crates_cover_the_workspace`] — [`SCANNED_CRATES`] is tied to `[workspace] members`.
//!    A hand-maintained crate list nobody validates is how the primitive list failed; a new crate
//!    must be scanned or excluded on purpose.
//! 8. [`the_literal_walkers_partition_every_scanned_file_correctly`] — the self-check for the class
//!    of bug where a mis-parsed quote absorbs real code into a phantom literal and silently removes
//!    it from both gates. Delimiter balance, not a length or share threshold; its doc records the
//!    measurements that rule those out. This class shipped twice.
//! 9. The `scanner_*` tests below run the detection functions over hand-written snippets — a fake new
//!    call site, a fake new wrapper, a fake new destination, an aliased cache root, an inline HTTP
//!    client, an `as` rename, a fn-pointer binding, plus doc-comment/`use`/string-literal negatives.
//!    A guard that silently matches nothing passes forever; these are what stop that.
//!
//! ## How far it chases the call graph
//!
//! Full transitive reachability converges on `run_worker_loop` and flags 73 of 133 production files,
//! which is not a gate, it is a census. One hop, on the other hand, is defeated by two lines: add
//! `fn forward(c) { ensure_instantid_file(c) }` to a file already on [`DOWNLOAD_SITES`], then call
//! `forward` from a new lane, and nothing anywhere names a primitive, a known wrapper or a
//! `DownloadContext`.
//!
//! So the walk is a closure that grows only through functions defined in files that are already
//! `JobTime` entries on [`DOWNLOAD_SITES`] — see [`download_wrappers`] for why that loses nothing and
//! why the `Installer` dispatchers are excluded. It converges, and finding the fixpoint added
//! sixteen files the one-hop scan could not see (`kps_jobs.rs`, `face_analysis_jobs.rs`,
//! `catalog_semantic_jobs.rs` and the rest reach weights through two hops, not one).
//!
//! Matching is restricted to **free-function** calls. Method calls have to be excluded or the walk is
//! worthless: `load`, `build` and `detect` are `impl` methods on a dozen unrelated types, and one of
//! them touching a primitive would put `.load(` on the watch list and flag a third of the crate.
//!
//! ## What it still cannot see
//!
//! Kept honest deliberately: a known hole belongs here, not in silence.
//!
//! * A cache root arriving as a *parameter* (`fn f(root: &str, sub: &str) -> d.join(root).join(sub)`)
//!   labels as `{root}` rather than `cache`. Registering the subdir-parameter helpers
//!   ([`PARAMETERIZED_CACHE_HELPERS`]) covers the shape that exists; a root-parameter helper would be
//!   a new one, and its callers would still have to pass a literal `"cache"` from somewhere the scan
//!   does see.
//! * A cache root behind a `const` whose initialiser is neither a plain literal nor `concat!` of
//!   literals — a `const fn` result, an `env!`, an `include_str!`. Those label as `{NAME}`, which is
//!   loud in [`CACHE_DESTINATIONS`] but does not match `cache`, so the ROOT check misses it. Literal
//!   and `concat!` initialisers both resolve; nothing in production source uses another form.
//! * A download wrapper that is a *method* rather than a free function. The file defining it is
//!   flagged on its own, so the escape needs the method's receiver type in a second file.
//! * **A sub-12-character wrapper name, BOUND AS A VALUE rather than called.**
//!   [`value_references`] applies [`MIN_VALUE_REFERENCE_NAME_LEN`] to discovered wrappers, so a short
//!   wrapper added to a file the allow-list already accepts, and then bound (`let grab = stage;`)
//!   rather than called from a new lane, is invisible — demonstrated with a four-character `grab` in
//!   `image_jobs/instantid.rs`. Calling it is still caught; only the value binding escapes. The floor
//!   is kept deliberately: unfloored, a realistic five-character name (`stage`) false-positived six
//!   unrelated production files, and a gate that cries wolf gets deleted by whoever hits it next.
//!   **Empty today** — the only two sub-floor names in the 123-entry watch map are `ensure_one`
//!   (`pose_jobs.rs`) and `resolve_one` (`image_jobs/pulid_candle.rs`), both *private*, so no new
//!   lane can name either.
//! * **A `type Dest = Path;` alias in `downloads.rs`**, used to give a [`NON_WEIGHT_FETCHERS`] member
//!   a destination parameter the signature scan cannot see. The scan reads signature TEXT, so an
//!   alias defeats it; resolving aliases needs a type resolver, which this is not. The two shapes
//!   that do not need a resolver — a `Path` bound hiding in a where clause, and a raw-bytes return
//!   type — are both checked.
//! * **[`invokes_free_function`] cries wolf on a one- or two-character wrapper name.** It matches
//!   `name(` rather than a bare word, so it takes a very short name to bite; a one-character probe
//!   wrapper `f` falsely flagged `person_track.rs` and `segment_jobs.rs`. Pre-existing, noisy rather
//!   than blind, and outside this story's delta — recorded here rather than fixed.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::architecture_tests::{
    char_literal_end, code_without_comments, code_without_comments_or_literals,
};

// ---------------------------------------------------------------------------------------------
// Allow-lists
// ---------------------------------------------------------------------------------------------

/// What a download-reaching file is *for*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DownloadRole {
    /// The Model Manager install path and its plumbing. Epic 17625 makes this the **only** thing
    /// that fetches weights, so these entries are permanent.
    Installer,
    /// A job-time weight fetch. Epic 17625 is migrating every one of these onto
    /// [`crate::downloads::resolve_hf_component_file`] (cache-only, no `DownloadContext`), so this
    /// population only ever shrinks. Each migration slice deletes its entry here.
    JobTime,
}

/// Every production file under `crates/sceneworks-worker/src/` that reaches a download primitive —
/// directly, through a one-hop local wrapper, or by constructing a `DownloadContext`.
///
/// Keyed on the path relative to `src/`, with no line numbers: line numbers rot on the first edit,
/// and `architecture_tests` already refuses them for the capability matrix.
///
/// **Adding an entry to this list is the thing epic 17625 exists to stop.** If a lane needs weights,
/// declare them in the model manifest and resolve them with
/// [`crate::downloads::resolve_hf_component_file`]; a missing component becomes an actionable
/// "install it from the Model Manager" error rather than a silent multi-gigabyte fetch inside a
/// render.
const DOWNLOAD_SITES: &[(&str, DownloadRole)] = &[
    // -- Installer: the Model Manager path. Permanent. ------------------------------------------
    // The primitives themselves (`ensure_cached_file`, `ensure_cached_file_verified`,
    // `ensure_hf_cached_file`, `download_snapshot_into_cache`) and their shared plumbing.
    ("downloads.rs", DownloadRole::Installer),
    // Dispatches `run_model_download_job` / `run_lora_download_job` from the job loop.
    ("lib.rs", DownloadRole::Installer),
    // The Model Manager executor: model + LoRA installs, tier fetches, conversions.
    ("model_jobs.rs", DownloadRole::Installer),
    // -- Job-time: the population epic 17625 is migrating. Only ever shrinks. --------------------
    // Video/image tier fetches via `model_jobs::ensure_hf_files_cached` at generation time.
    ("image_jobs/base.rs", DownloadRole::JobTime),
    ("video_jobs/bernini.rs", DownloadRole::JobTime),
    ("video_jobs/krea_realtime.rs", DownloadRole::JobTime),
    ("video_jobs/ltx.rs", DownloadRole::JobTime),
    ("video_jobs/mochi.rs", DownloadRole::JobTime),
    ("video_jobs/scail2.rs", DownloadRole::JobTime),
    ("video_jobs/wan.rs", DownloadRole::JobTime),
    // Depth-Anything-V2 preprocessor, fetched through `strict_control::ensure_depth_estimator_dir`.
    ("image_jobs/candle_strict_control.rs", DownloadRole::JobTime),
    ("image_jobs/strict_control.rs", DownloadRole::JobTime),
    ("image_jobs/zimage.rs", DownloadRole::JobTime),
    // ControlNet weights, one lane per family.
    ("image_jobs/flux1_control.rs", DownloadRole::JobTime),
    ("image_jobs/flux1_control_candle.rs", DownloadRole::JobTime),
    ("image_jobs/flux2.rs", DownloadRole::JobTime),
    ("image_jobs/flux2_control_candle.rs", DownloadRole::JobTime),
    ("image_jobs/kolors_control.rs", DownloadRole::JobTime),
    ("image_jobs/krea_control.rs", DownloadRole::JobTime),
    ("image_jobs/krea_control_candle.rs", DownloadRole::JobTime),
    ("image_jobs/qwen_control.rs", DownloadRole::JobTime),
    ("image_jobs/zimage_control.rs", DownloadRole::JobTime),
    // IP-Adapter bundles.
    ("image_jobs/flux_ipadapter.rs", DownloadRole::JobTime),
    ("image_jobs/kolors_ipadapter.rs", DownloadRole::JobTime),
    ("image_jobs/sdxl_ipadapter.rs", DownloadRole::JobTime),
    // Identity stacks. `pulid.rs` builds a `DownloadContext` literal of its own, so it is caught
    // without any wrapper resolution; `image_jobs/kolors.rs`, `sdxl.rs` and `sensenova.rs` are the
    // files that genuinely need it — they name no primitive and no context, reaching Hugging Face
    // only through `image_jobs/base.rs`'s `stage_likeness`.
    ("image_jobs/instantid.rs", DownloadRole::JobTime),
    ("image_jobs/pulid.rs", DownloadRole::JobTime),
    ("image_jobs/pulid_candle.rs", DownloadRole::JobTime),
    // Lightning distill LoRA, fetched mid-render via `download_snapshot_into_cache`. Neither of
    // these appears in a grep for `ensure_hf_cached_file`/`ensure_cached_file`.
    ("image_jobs/qwen.rs", DownloadRole::JobTime),
    ("image_jobs/qwen_edit_candle.rs", DownloadRole::JobTime),
    // Imported-SDXL tokenizer/text-encoder components.
    ("image_jobs/sdxl_imported.rs", DownloadRole::JobTime),
    // `upscale_jobs.rs` and `video_jobs/seedvr2.rs` left this list in sc-17633 + sc-17632, and
    // `pose_jobs.rs` in sc-17634: the Real-ESRGAN ONNX, the SeedVR2 checkpoint and the two DWPose
    // graphs are declared in the catalog and resolved with `resolve_hf_component_file`, and none of
    // them reaches any download primitive any more. `pose_jobs.rs` retired the LAST use of
    // `ensure_cached_file`, the non-HF (arbitrary-URL) primitive, so every remaining entry below
    // fetches from Hugging Face.
    // -- Job-time: added by the transitive wrapper walk (sc-17637). These reach a download through a
    // wrapper more than one hop away, which the original one-hop scan could not see. -------------
    // Catalog/dataset image acquisition: fetches remote source URLs into the catalog at job time.
    ("catalog_image_fetch.rs", DownloadRole::JobTime),
    ("catalog_semantic_jobs.rs", DownloadRole::JobTime),
    ("dataset_parquet_jobs.rs", DownloadRole::JobTime),
    // Analyzer lanes that stage the InstantID face stack mid-job. `control_training_jobs.rs` left
    // in sc-17634: DWPose was the only weight it staged, and it now resolves that through the
    // cache-only `pose_jobs::require_dwpose_weights`, which takes no `DownloadContext`. The three
    // below remain because the InstantID face stack is still job-time (sc-17631).
    ("face_analysis_jobs.rs", DownloadRole::JobTime),
    ("face_likeness_compare_jobs.rs", DownloadRole::JobTime),
    ("kps_jobs.rs", DownloadRole::JobTime),
    // Identity likeness staging shared by the SDXL-family lanes (`image_jobs/base.rs`).
    ("image_jobs/kolors.rs", DownloadRole::JobTime),
    ("image_jobs/sdxl.rs", DownloadRole::JobTime),
    ("image_jobs/sensenova.rs", DownloadRole::JobTime),
    (
        "image_jobs/zimage_identity_candle.rs",
        DownloadRole::JobTime,
    ),
    // Bernini tier staging, reached from the image lane as well as the video one.
    ("image_jobs/bernini.rs", DownloadRole::JobTime),
    // The candle video dispatcher stages Mochi/Wan tiers.
    ("video_jobs/candle.rs", DownloadRole::JobTime),
    // The two module-level dispatchers. They contain no download of their own; they call the lane
    // entry points above, which do. They leave this list when the lanes they route to are migrated.
    ("image_jobs.rs", DownloadRole::JobTime),
    ("video_jobs/mod.rs", DownloadRole::JobTime),
];

/// How many [`DownloadSites`](DOWNLOAD_SITES) entries are still `JobTime` — epic 17625's
/// remaining-work number, pinned so a role cannot be quietly changed to `Installer` to keep the
/// population looking flat.
///
/// **Only ever goes down.** Each migration slice deletes its entry and lowers this in the same
/// commit.
const JOB_TIME_DOWNLOAD_SITES_REMAINING: usize = 42;

/// What a `<data_dir>/cache/<subdir>` destination holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CacheRole {
    /// Model weights outside the Hugging Face cache. **This is the class AC9 forbids growing.**
    /// Every entry here is either a job-time destination epic 17625 is migrating, or a read-only
    /// legacy root kept so an existing install keeps working without re-downloading (AC10).
    Weights,
    /// The sanctioned destination: the app-managed Hugging Face hub cache
    /// (`<data_dir>/cache/huggingface/hub`), which the Model Manager installs into.
    HfCache,
    /// Derived, regenerable job output. Deleting it costs compute, never a download.
    Scratch,
    /// User-supplied bytes staged by the API before a job consumes them.
    Upload,
    /// App runtime state (the jobs database, cross-process lock files).
    Runtime,
    /// Code that constructs the cache root itself rather than a destination inside it.
    CacheRoot,
    /// The subdirectory arrives as a **parameter**, so the destination is not visible at this site.
    /// See [`PARAMETERIZED_CACHE_HELPERS`] for how the callers are covered.
    Parameterized,
}

/// Every `<data_dir>/cache/<subdir>` destination in production source across the four crates that
/// can write under the app data dir, as `(crate label, path relative to that crate's src/, subdir,
/// role)`.
///
/// The subdir is the *destination name*, not a line number, so two sites in one function collapse to
/// one entry and a genuinely new destination inside an already-listed file is still a new entry.
/// `const` names are resolved to their string values, so `POSE_UPLOADS_CACHE_DIR` reads as
/// `pose-uploads` and renaming the constant does not churn this list.
///
/// The rust-api / core / desktop crates are covered so a new **weight** destination cannot simply be
/// added outside the worker. None of their current entries is `Weights`, and that is the point: the
/// day one appears, this list has to say so out loud.
const CACHE_DESTINATIONS: &[(&str, &str, &str, CacheRole)] = &[
    // ---- crates/sceneworks-worker/src ---------------------------------------------------------
    // Prepared control-training datasets, regenerated from the source images.
    (
        "worker",
        "control_training_jobs.rs",
        "control-datasets",
        CacheRole::Scratch,
    ),
    // ControlNet weights, one destination per family (sc-17628…sc-17631 migrate these).
    (
        "worker",
        "image_jobs/flux1_control.rs",
        "controlnet-flux1",
        CacheRole::Weights,
    ),
    (
        "worker",
        "image_jobs/flux1_control_candle.rs",
        "controlnet-flux1-candle",
        CacheRole::Weights,
    ),
    (
        "worker",
        "image_jobs/flux2.rs",
        "controlnet-flux2",
        CacheRole::Weights,
    ),
    (
        "worker",
        "image_jobs/flux2_control_candle.rs",
        "controlnet-flux2",
        CacheRole::Weights,
    ),
    (
        "worker",
        "image_jobs/kolors_control.rs",
        "controlnet-kolors",
        CacheRole::Weights,
    ),
    (
        "worker",
        "image_jobs/krea_control.rs",
        "controlnet-krea",
        CacheRole::Weights,
    ),
    (
        "worker",
        "image_jobs/krea_control_candle.rs",
        "controlnet-krea",
        CacheRole::Weights,
    ),
    (
        "worker",
        "image_jobs/qwen_control.rs",
        "controlnet-qwen",
        CacheRole::Weights,
    ),
    (
        "worker",
        "image_jobs/zimage_control.rs",
        "controlnet-zimage",
        CacheRole::Weights,
    ),
    // IP-Adapter bundles.
    (
        "worker",
        "image_jobs/flux_ipadapter.rs",
        "ipadapter-flux",
        CacheRole::Weights,
    ),
    (
        "worker",
        "image_jobs/kolors_ipadapter.rs",
        "ipadapter-kolors",
        CacheRole::Weights,
    ),
    (
        "worker",
        "image_jobs/sdxl_ipadapter.rs",
        "ipadapter-sdxl",
        CacheRole::Weights,
    ),
    // InstantID: face stack, ControlNet and OpenPose bundles.
    (
        "worker",
        "image_jobs/instantid.rs",
        "instantid-controlnet",
        CacheRole::Weights,
    ),
    (
        "worker",
        "image_jobs/instantid.rs",
        "instantid-mlx",
        CacheRole::Weights,
    ),
    (
        "worker",
        "image_jobs/instantid.rs",
        "instantid-openpose",
        CacheRole::Weights,
    ),
    // PuLID identity adapters, MLX and candle lanes.
    (
        "worker",
        "image_jobs/pulid.rs",
        "pulid-flux-mlx",
        CacheRole::Weights,
    ),
    (
        "worker",
        "image_jobs/pulid_candle.rs",
        "pulid-flux",
        CacheRole::Weights,
    ),
    // Tokenizer/text-encoder components staged for an imported SDXL checkpoint.
    (
        "worker",
        "image_jobs/sdxl_imported.rs",
        "sdxl-imported-components",
        CacheRole::Weights,
    ),
    // Depth-Anything-V2 preprocessor.
    (
        "worker",
        "image_jobs/strict_control.rs",
        "depth-anything-v2",
        CacheRole::Weights,
    ),
    // Legacy read-only roots the person/pose analyzers still consult so a pre-migration install
    // keeps working without re-downloading (AC10). Paired with a `models/<same>` root at each site.
    (
        "worker",
        "person_jobs.rs",
        "person-detect",
        CacheRole::Weights,
    ),
    (
        "worker",
        "person_segment.rs",
        "person-segment",
        CacheRole::Weights,
    ),
    (
        "worker",
        "person_segment_sam3_common.rs",
        "person-segment-sam3",
        CacheRole::Weights,
    ),
    // DWPose preprocessor weights: the READ-ONLY legacy root (AC10), kept so a pre-sc-17634 install
    // resolves its existing ~300 MB pair instead of re-downloading. Nothing writes here any more —
    // the graphs install into the HF cache via the `dwpose_pose_detector` entry. Note the sibling
    // legacy root, rtmlib's `~/.cache/rtmlib/hub/checkpoints/`, is keyed off `$HOME` rather than
    // `data_dir`, so it is invisible to this `<data_dir>/cache/<subdir>` inventory by construction.
    // Reclaiming both is sc-17636 (S11).
    ("worker", "pose_jobs.rs", "dwpose", CacheRole::Weights),
    // The two READ-ONLY legacy upscaler roots, both now consulted from `upscale_jobs.rs` (AC10).
    // `upscale` covers the Real-ESRGAN ONNX (`cache/upscale/`, sc-17633) and the image lane's old
    // SeedVR2 copy (`cache/upscale/seedvr2/`); `seedvr2-mlx` is the video lane's old copy of the
    // SAME checkpoint, which moved here in sc-17632 when the two lanes collapsed onto one resolver.
    // Nothing writes to either any more; reclaiming the bytes is sc-17636 (S11).
    ("worker", "upscale_jobs.rs", "upscale", CacheRole::Weights),
    (
        "worker",
        "upscale_jobs.rs",
        "seedvr2-mlx",
        CacheRole::Weights,
    ),
    // API-staged uploads: the worker re-confines the client-supplied source path to these roots.
    ("worker", "imports.rs", "lora-uploads", CacheRole::Upload),
    ("worker", "imports.rs", "model-uploads", CacheRole::Upload),
    (
        "worker",
        "kps_jobs.rs",
        "keypoint-uploads",
        CacheRole::Upload,
    ),
    ("worker", "model_jobs.rs", "lora-uploads", CacheRole::Upload),
    (
        "worker",
        "model_jobs.rs",
        "model-uploads",
        CacheRole::Upload,
    ),
    ("worker", "pose_jobs.rs", "pose-uploads", CacheRole::Upload),
    // Rendered pose-detection previews, regenerated from the source image.
    ("worker", "pose_jobs.rs", "pose_detect", CacheRole::Scratch),
    // Cross-process model-conversion locks.
    (
        "worker",
        "model_jobs.rs",
        "worker-locks",
        CacheRole::Runtime,
    ),
    // The two confinement helpers that take the cache subdir as a parameter. See
    // `PARAMETERIZED_CACHE_HELPERS`.
    (
        "worker",
        "paths.rs",
        "{cache_dir}",
        CacheRole::Parameterized,
    ),
    (
        "worker",
        "paths.rs",
        "{upload_cache}",
        CacheRole::Parameterized,
    ),
    // ---- apps/rust-api/src --------------------------------------------------------------------
    ("rust-api", "assets.rs", "uploads", CacheRole::Upload),
    ("rust-api", "workflows.rs", "uploads", CacheRole::Upload),
    (
        "rust-api",
        "keypoints.rs",
        "keypoint-uploads",
        CacheRole::Upload,
    ),
    ("rust-api", "loras.rs", "lora-uploads", CacheRole::Upload),
    ("rust-api", "models.rs", "model-uploads", CacheRole::Upload),
    ("rust-api", "poses.rs", "pose-uploads", CacheRole::Upload),
    // Generated media thumbnails.
    ("rust-api", "lib.rs", "media-thumbnails", CacheRole::Scratch),
    // The jobs database.
    ("rust-api", "server.rs", "jobs.db", CacheRole::Runtime),
    // Upload-sweep helpers that forward a subdir parameter.
    (
        "rust-api",
        "assets.rs",
        "{subdir}",
        CacheRole::Parameterized,
    ),
    ("rust-api", "lib.rs", "{subdir}", CacheRole::Parameterized),
    (
        "rust-api",
        "startup.rs",
        "{subdir}",
        CacheRole::Parameterized,
    ),
    // ---- crates/sceneworks-core/src -----------------------------------------------------------
    // Constructs `<data_dir>/cache` itself; no destination inside it.
    ("core", "app_paths.rs", "<cache-root>", CacheRole::CacheRoot),
    // The app-managed Hugging Face hub cache — the destination epic 17625 migrates everything ONTO.
    ("core", "hf_home.rs", "huggingface", CacheRole::HfCache),
    (
        "core",
        "project_store.rs",
        "keypoint-uploads",
        CacheRole::Upload,
    ),
    (
        "core",
        "project_store.rs",
        "pose_detect",
        CacheRole::Scratch,
    ),
    // ---- apps/desktop/src ---------------------------------------------------------------------
    // Points `HF_HOME` at the app-managed hub cache.
    ("desktop", "setup.rs", "huggingface", CacheRole::HfCache),
];

/// How many [`CACHE_DESTINATIONS`] entries hold model weights — the population epic 17625 migrates
/// onto the app-managed Hugging Face cache, pinned so a new one cannot be smuggled in under a
/// cheaper role.
///
/// **Only ever goes down.** Each migration slice deletes its destination and lowers this in the same
/// commit.
const WEIGHTS_CACHE_DESTINATIONS_REMAINING: usize = 25;

/// Helpers that build a `<data_dir>/cache/<subdir>` root from a **parameter**, as
/// `(function name, zero-based index of the subdir parameter)`.
///
/// # Why this is called out separately
///
/// This is the documented way an allow-list keyed on string literals gets bypassed. `paths.rs`'s
/// `normalize_app_managed_cache_path(settings, raw_path, cache_dir, label)` never mentions a concrete
/// destination, so scanning for `.join("cache").join("<literal>")` sees nothing at the definition —
/// and a caller that routes a NEW weight destination through it would be invisible.
///
/// The fix is to scan the **call sites**: for each registered helper, the literal (or `const`) passed
/// in the recorded position is recorded as a cache destination attributed to the *calling* file. That
/// is why `poses.rs`/`keypoints.rs` appear in [`CACHE_DESTINATIONS`] at all — neither contains the
/// string `"cache"`.
///
/// [`parameterized_cache_root_helpers_are_registered`] keeps this list honest: it rediscovers the
/// shape structurally (a function whose body joins `"cache"` with one of its own parameters) and
/// fails if the discovered set differs from this one. A new parameterized helper cannot be added
/// without registering it.
///
/// **Known limitation, stated loudly:** a caller that forwards its *own* parameter through one of
/// these helpers records the opaque marker `{subdir}` rather than a destination — `rust-api`'s
/// `sweep_stale_uploads` chain does exactly that. Those are sweep/confinement paths, not write
/// targets for weights, but the scanner cannot prove that; it can only make the opacity visible in
/// the allow-list. A second-order forwarder that *did* stage weights would land as a `{…}` entry,
/// which is still a new entry, and still red.
const PARAMETERIZED_CACHE_HELPERS: &[(&str, usize)] = &[
    ("normalize_app_managed_cache_path", 2),
    ("resolve_import_source_path", 3),
    ("write_upload_field_to_dir", 2),
    ("sweep_stale_uploads_cancellable", 1),
];

/// Functions that hand out the **bare** `<data_dir>/cache` root, as `(crate label, file, fn name)`.
///
/// # Why this is called out separately
///
/// [`CACHE_DESTINATIONS`] is keyed on `(crate, file, subdir)`, so every site that produces the root
/// itself collapses onto one `<cache-root>` entry per file. That is fine as a description and useless
/// as a gate: a brand-new `pub fn cache_root(data_dir) -> data_dir.join("cache")` added to
/// `app_paths.rs` — the natural home for such an accessor, and the file that already owns the only
/// `<cache-root>` entry — is absorbed into the existing entry and changes nothing. A lane can then
/// write `cache_root(d).join("new-weights")`, name no cache literal of its own, and stay green.
///
/// So the producers are registered by **function**, [`cache_root_producers_are_registered`]
/// rediscovers them structurally, and their call sites contribute whatever they join onto the root —
/// the same two-sided treatment [`PARAMETERIZED_CACHE_HELPERS`] gets.
const CACHE_ROOT_PRODUCERS: &[(&str, &str, &str)] = &[
    // The repo-relative fallback used when no home directory is available: `data/cache`.
    ("core", "app_paths.rs", "repo_relative_paths"),
    // The per-platform defaults. Only the Windows body names `cache` as a final segment
    // (`%LOCALAPPDATA%\SceneWorks\cache`); the scan is textual, so all three `cfg`-gated definitions
    // share the one name and the one entry.
    ("core", "app_paths.rs", "platform_default_paths"),
];

/// The byte-fetching entry points a job lane could call. Everything that opens a socket for model
/// bytes bottoms out in one of these.
///
/// **This list is not hand-maintained.** [`downloads_fetch_surface_is_registered`] rediscovers the
/// whole fetching surface of [`crate::downloads`] structurally — every externally-visible `fn` whose
/// body transitively reaches an outbound HTTP request — and requires the discovered set to equal this
/// list plus [`NON_WEIGHT_FETCHERS`]. A fifth (or fifteenth) primitive therefore cannot appear in
/// `downloads.rs` without a deliberate, reviewed decision about which of the two lists it belongs in.
const DOWNLOAD_PRIMITIVES: &[&str] = &[
    // The three helpers sc-17637 names, plus their whole-snapshot peers.
    "ensure_cached_file",
    "ensure_cached_file_verified",
    "ensure_hf_cached_file",
    "download_snapshot_into_cache",
    // `download_snapshot_into_cache`'s inner half: fetches a whole HF snapshot into an arbitrary
    // caller-chosen directory. The two job-time Lightning-LoRA sites use the `_into_cache` wrapper,
    // but this one is `pub(crate)` and equally callable from a lane.
    "download_snapshot",
    // The `sourceUrl` fetchers behind the LoRA/model importers. `download_lora_source_url` and
    // `download_model_source_url` are thin role wrappers over `download_source_url`; all three write
    // remote bytes to a caller-chosen path, which is exactly the shape AC9 forbids in a render.
    "download_source_url",
    "download_lora_source_url",
    "download_model_source_url",
    // Fetches a public URL into memory with a byte cap. Takes NO `DownloadContext`, so neither the
    // context scan nor the wrapper scan would see a lane calling it — the reason it is named here.
    "fetch_public_source_url_bytes",
    "fetch_public_source_url_bytes_with_options",
];

/// Externally-visible functions in `downloads.rs` that reach the network but are **not** a way to
/// pull model bytes into a job. Each one is here because of what it does, not because listing it was
/// convenient.
///
/// This is the escape hatch for [`downloads_fetch_surface_is_registered`], and it is deliberately
/// small: everything else discovered by that test has to go in [`DOWNLOAD_PRIMITIVES`], where the
/// reachability scan covers it.
///
/// # The criterion is machine-checked, not comment-checked
///
/// This was the last list in the module resting on a comment. A `probe_and_save(client, url, dest)`
/// added to `downloads.rs` and listed here with a plausible note would have made its lane callers
/// invisible. So the property every entry must have — *it cannot write remote bytes to a
/// caller-chosen destination* — is asserted: a member whose signature takes a `Path`/`PathBuf`
/// parameter is rejected outright and has to go in [`DOWNLOAD_PRIMITIVES`] instead.
const NON_WEIGHT_FETCHERS: &[&str] = &[
    // The two HTTP client factories. They return a `reqwest::Client` — configured with the download
    // timeouts — and transfer nothing themselves. What a caller then DOES with the client is confined
    // by `http_client_construction_is_confined`, which is the gate that covers this shape.
    "streaming_download_client",
    "build_source_url_client",
    // A `HEAD`/range probe for the size of a `sourceUrl`. Returns a `u64`; writes nothing.
    "lora_source_content_length",
    // `HfSnapshot::resolve` — an `impl` method that pages the HF *file listing* API for a repo. It
    // returns metadata (names, sizes, SHAs) and writes nothing; a caller still needs a real primitive
    // to turn that listing into bytes on disk. It also needs a `reqwest::Client` handed to it, which
    // `http_client_construction_is_confined` covers independently.
    //
    // (An earlier version of this comment said the bare name had to stay out of DOWNLOAD_PRIMITIVES
    // because it would match unrelated `.resolve(` calls. That was wrong — `invokes_free_function`
    // already ignores method calls. The reason is the one above.)
    "resolve",
];

/// The crates whose `src/` trees are scanned, as
/// `(label, path from the repo root, crate root file, minimum production files)`.
///
/// The download gate covers only the worker: the primitives are `pub(crate)`, so no other crate can
/// reach them. The cache gate covers **every** crate in the workspace that can see the app data dir,
/// because a new weight destination could be introduced in any of them. The minimum is a
/// "the walk found nothing" floor, per crate because `apps/rust-worker` is a single `main.rs` and a
/// shared floor would either miss a broken walk elsewhere or fail on that one.
const SCANNED_CRATES: &[(&str, &str, &str, usize)] = &[
    ("worker", "crates/sceneworks-worker/src", "lib.rs", 100),
    ("rust-api", "apps/rust-api/src", "lib.rs", 5),
    ("core", "crates/sceneworks-core/src", "lib.rs", 5),
    ("desktop", "apps/desktop/src", "main.rs", 5),
    ("rust-worker", "apps/rust-worker/src", "main.rs", 1),
    ("mcp", "crates/sceneworks-mcp/src", "lib.rs", 2),
    (
        "image-quality",
        "crates/sceneworks-image-quality/src",
        "lib.rs",
        2,
    ),
    (
        "memory-adapter",
        "crates/sceneworks-memory-adapter/src",
        "lib.rs",
        1,
    ),
];

/// Workspace members deliberately NOT scanned, as `(path, why)`.
///
/// Empty, and [`scanned_crates_cover_the_workspace`] keeps it honest: a crate added to
/// `[workspace] members` must either be scanned or be excluded here on purpose. Without that tie a
/// brand-new crate can hold a brand-new `<data_dir>/cache` weight destination and never be looked
/// at — which is the same "hand-maintained list nobody validates" failure the primitive list had.
const UNSCANNED_WORKSPACE_MEMBERS: &[(&str, &str)] = &[];

/// The externally-visible, network-reaching surface of every [`HTTP_CLIENT_FILES`] member **other
/// than `downloads.rs`**, as `(file, fn name, what it does)`.
///
/// # Why this exists
///
/// [`HTTP_CLIENT_FILES`] confines *where* a client may be built. On its own that is not enough: the
/// comment beside `api_client.rs` saying it "talks to 127.0.0.1, never to a model host" is prose, not
/// a gate. A `pub(crate) async fn fetch_weights(url, dest)` added to `api_client.rs` — build a
/// client, send, write the bytes to a caller-supplied path — is a complete job-time download that no
/// other assertion here sees: the file is allow-listed, and the calling lane names no primitive.
///
/// So every allow-listed file gets the treatment `downloads.rs` gets from
/// [`downloads_fetch_surface_is_registered`]: its network-reaching surface is rediscovered
/// structurally and every member has to be named here.
const HTTP_CLIENT_FILE_SURFACE: &[(&str, &str, &str)] = &[
    (
        "api_client.rs",
        "new",
        "builds the worker's client for the LOCAL API with the download timeouts applied; returns \
         an `ApiClient` and transfers nothing",
    ),
    (
        "api_client.rs",
        "get_json",
        "GETs a LOCAL API path and deserializes JSON. Takes a path, not a URL, and returns a typed \
         value — it cannot write bytes to a caller-chosen destination",
    ),
    (
        "api_client.rs",
        "post_json",
        "POSTs JSON to a LOCAL API path and deserializes the JSON reply; same shape as `get_json`",
    ),
];

/// Worker files allowed to construct an HTTP client or issue an HTTP request directly.
///
/// The gate one level below this — "did a lane call a download primitive" — is defeated by simply
/// not calling one: `reqwest::Client::new().get(url).send()` written inline in a job lane fetches
/// weights while naming nothing on any allow-list. So the ability to *make an HTTP request at all*
/// is confined here, and a new file that wants one has to say why out loud.
const HTTP_CLIENT_FILES: &[(&str, &str)] = &[
    (
        "downloads.rs",
        "owns every download primitive and every HTTP client factory in the crate; this is the file \
         the confinement exists to concentrate network access into",
    ),
    (
        "api_client.rs",
        "the worker's client for the LOCAL SceneWorks API (job claim, progress, completion). It \
         talks to 127.0.0.1, never to a model host, and fetches no weights",
    ),
];

// ---------------------------------------------------------------------------------------------
// Source loading
// ---------------------------------------------------------------------------------------------

/// One production source file: comments stripped, `#[cfg(test)]`-only items removed, string literals
/// intact.
#[derive(Debug, Clone)]
struct SourceFile {
    /// Path relative to the crate's `src/`, slash-separated (`image_jobs/flux2.rs`).
    rel: String,
    code: String,
}

impl SourceFile {
    /// The same text with string literal CONTENTS blanked, for syntax questions where a literal must
    /// not count (`"…ensure_hf_cached_file(…)"` inside an error message is not a call).
    fn code_without_literals(&self) -> String {
        code_without_comments_or_literals(&self.code)
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("the worker crate lives two levels below the repo root")
        .to_path_buf()
}

/// Every `.rs` file under `dir`, as `(path relative to dir, comment-stripped source)`.
///
/// Walked by path rather than enumerated, so a NEW file is scanned the day it lands. A hardcoded
/// file list is how the previous inventory-shaped guards in this repo went stale.
fn read_tree(dir: &Path) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    collect_rs(dir, dir, &mut out);
    out
}

fn collect_rs(root: &Path, dir: &Path, out: &mut BTreeMap<String, String>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_rs(root, &path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let rel = path
                .strip_prefix(root)
                .expect("path under root")
                .to_string_lossy()
                .replace('\\', "/");
            let source =
                std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", e));
            out.insert(rel, code_without_comments(&source));
        }
    }
}

/// The production files of one crate: the whole `src/` tree minus every file reachable only as test
/// code, with inline `#[cfg(test)]` items removed from what remains.
fn production_sources(src_dir: &Path, root_module: &str) -> Vec<SourceFile> {
    let tree = read_tree(src_dir);
    let test_only = test_only_files(&tree, root_module);
    tree.iter()
        .filter(|(rel, _)| !test_only.contains(rel.as_str()))
        .map(|(rel, code)| SourceFile {
            rel: rel.clone(),
            code: strip_cfg_test_items(code),
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Pure detection primitives (unit-tested against synthetic input below)
// ---------------------------------------------------------------------------------------------

/// `bytes[start..end]` as a `String`, clamped to the buffer.
///
/// Byte-indexed rather than `&str`-indexed on purpose: the scanners advance byte by byte, and source
/// text outside comments still carries multi-byte characters (`…` and `—` appear in error strings),
/// so slicing the `&str` directly can panic on a non-boundary index.
fn slice_lossy(bytes: &[u8], start: usize, end: usize) -> String {
    let start = start.min(bytes.len());
    let end = end.clamp(start, bytes.len());
    String::from_utf8_lossy(&bytes[start..end]).into_owned()
}

/// Index just past a `"…"` literal starting at `start`.
fn skip_string(bytes: &[u8], start: usize) -> usize {
    let mut index = start + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index += 2,
            b'"' => return index + 1,
            _ => index += 1,
        }
    }
    index
}

/// Index just past a `'x'` / `'\n'` / `'\''` char literal starting at `start`, or `start + 1` when
/// the quote is a lifetime (`'a`, `'_`) rather than a literal.
///
/// Without this, a `'{'` char literal — eight files in the scanned crates contain one — would throw
/// off every brace match downstream.
///
/// Delegates to [`char_literal_end`] rather than re-deriving the rule: this function previously
/// tested `bytes[start + 2] == '\''` *before* the backslash branch, so `'\''` ended at `start + 3`
/// — on the escaped quote, not the real terminator — and desynchronised every brace, paren and
/// argument walk in this module on any file containing that literal.
fn skip_char(bytes: &[u8], start: usize) -> usize {
    char_literal_end(bytes, start).unwrap_or(start + 1)
}

/// Whether `code` contains an *invocation* of `name` — the identifier, at a token boundary, followed
/// by `(`, and not preceded by `fn` (which would be the definition).
///
/// A `use` item never matches: `use super::{advanced, ensure_hf_cached_file, …};` has a comma after
/// the name, not a paren. Path-qualified calls do match, which is required —
/// `crate::downloads::ensure_hf_cached_file(…)` is how four lanes spell it.
fn invokes(code: &str, name: &str) -> bool {
    invokes_matching(code, name, |_, _| true)
}

/// [`invokes`], restricted to **free-function** calls: `f(…)` and `path::f(…)`, never `value.f(…)`.
///
/// Every download primitive is a free function reachable through the crate-root glob, and so is every
/// wrapper worth chasing. Method calls have to be excluded or the wrapper walk is worthless: names
/// like `load`, `build` and `detect` are defined as `impl` methods in a dozen unrelated types, and
/// one of them calling a primitive would otherwise put `.load(` on the watch list and flag a third of
/// the crate. (The residual shape — a *method* that wraps a download and is called on a receiver from
/// a new file — needs the receiver's type in scope, and the file that defines that method is flagged
/// on its own.)
fn invokes_free_function(code: &str, name: &str) -> bool {
    invokes_matching(code, name, |code, start| {
        let before = code[..start].trim_end();
        !(before.ends_with('.') && !before.ends_with(".."))
    })
}

fn invokes_matching(code: &str, name: &str, accept: impl Fn(&str, usize) -> bool) -> bool {
    let bytes = code.as_bytes();
    let name_bytes = name.as_bytes();
    let mut index = 0usize;
    while let Some(offset) = code[index..].find(name) {
        let start = index + offset;
        let end = start + name_bytes.len();
        index = end;
        let preceded_by_ident = start
            .checked_sub(1)
            .map(|i| bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric())
            .unwrap_or(false);
        if preceded_by_ident {
            continue;
        }
        if !accept(code, start) {
            continue;
        }
        if bytes
            .get(end)
            .is_some_and(|b| *b == b'_' || b.is_ascii_alphanumeric())
        {
            continue;
        }
        let mut after = end;
        while after < bytes.len() && (bytes[after] as char).is_whitespace() {
            after += 1;
        }
        if bytes.get(after) != Some(&b'(') {
            continue;
        }
        // `fn NAME(` is the definition, not a call.
        let before = code[..start].trim_end();
        if before.ends_with("fn") && before[..before.len() - 2].ends_with(char::is_whitespace) {
            continue;
        }
        return true;
    }
    false
}

/// One `fn` item: its name, its parameter names in order, its return-type text and its body text,
/// plus whether it is visible outside its own module.
#[derive(Debug, Clone)]
struct FnItem {
    name: String,
    params: Vec<String>,
    /// The parameter list verbatim, types included — `params` keeps only the names.
    params_raw: String,
    /// Everything between the parameter list and the body brace: the `-> T` and any `where` clause.
    /// A generic bound like `P: AsRef<Path>` lives here, not in `params_raw`.
    ret: String,
    body: String,
    /// `pub`, `pub(crate)`, `pub(super)`, `pub(in …)`. A private `fn` in `downloads.rs` cannot be
    /// named by a job lane at all, which is what lets the fetch-surface registry stay small.
    is_public: bool,
}

/// Strip `kw` from the end of `code` when it is a whole token there.
///
/// `strip_suffix("async")` alone would also fire on `fn_async`, silently mis-reading the modifier
/// chain in front of a `fn` and reporting a `pub` item as private.
fn strip_trailing_keyword<'a>(code: &'a str, kw: &str) -> Option<&'a str> {
    let rest = code.strip_suffix(kw)?;
    let boundary = rest
        .chars()
        .next_back()
        .map(|c| !(c == '_' || c.is_ascii_alphanumeric()))
        .unwrap_or(true);
    boundary.then_some(rest)
}

/// Whether the `fn` whose keyword starts at `fn_keyword` carries a `pub…` visibility.
///
/// Walks back over the modifier chain rustfmt can put between the visibility and the keyword
/// (`pub(crate) async unsafe extern "C" fn`), then checks for `pub` or `pub(…)`.
fn item_is_public(code: &str, fn_keyword: usize) -> bool {
    let mut before = code[..fn_keyword].trim_end();
    loop {
        if let Some(rest) = strip_trailing_keyword(before, "async")
            .or_else(|| strip_trailing_keyword(before, "unsafe"))
            .or_else(|| strip_trailing_keyword(before, "const"))
            .or_else(|| strip_trailing_keyword(before, "extern"))
        {
            before = rest.trim_end();
            continue;
        }
        // `extern "C"` — drop the ABI string, then the keyword on the next turn.
        if let Some(head) = before.strip_suffix('"') {
            if let Some(open) = head.rfind('"') {
                if strip_trailing_keyword(head[..open].trim_end(), "extern").is_some() {
                    before = head[..open].trim_end();
                    continue;
                }
            }
        }
        break;
    }
    if let Some(head) = before.strip_suffix(')') {
        let Some(open) = head.rfind('(') else {
            return false;
        };
        return strip_trailing_keyword(head[..open].trim_end(), "pub").is_some();
    }
    strip_trailing_keyword(before, "pub").is_some()
}

/// Every `fn NAME(params) … { body }` in `code`, with string literals and char literals respected
/// while brace-matching.
///
/// Deliberately loose — trait method signatures without bodies are skipped, and a body that
/// mis-matches would run long rather than short, which over-reports rather than silently passing.
fn fn_items(code: &str) -> Vec<FnItem> {
    let bytes = code.as_bytes();
    let mut items = Vec::new();
    let mut cursor = 0usize;
    while let Some(offset) = code[cursor..].find("fn ") {
        let keyword = cursor + offset;
        cursor = keyword + 3;
        // Must be the `fn` keyword, not the tail of an identifier.
        if keyword
            .checked_sub(1)
            .map(|i| bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric())
            .unwrap_or(false)
        {
            continue;
        }
        let mut index = keyword + 3;
        while index < bytes.len() && (bytes[index] as char).is_whitespace() {
            index += 1;
        }
        let name_start = index;
        while index < bytes.len() && (bytes[index] == b'_' || bytes[index].is_ascii_alphanumeric())
        {
            index += 1;
        }
        if index == name_start {
            continue;
        }
        let name = code[name_start..index].to_owned();
        // Skip a generic parameter list before the argument list.
        while index < bytes.len() && bytes[index] != b'(' {
            if bytes[index] == b'{' || bytes[index] == b';' {
                break;
            }
            index += 1;
        }
        if bytes.get(index) != Some(&b'(') {
            continue;
        }
        let params_start = index;
        let mut depth = 0i32;
        while index < bytes.len() {
            match bytes[index] {
                b'"' => {
                    index = skip_string(bytes, index);
                    continue;
                }
                b'\'' => {
                    index = skip_char(bytes, index);
                    continue;
                }
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        index += 1;
                        break;
                    }
                }
                _ => {}
            }
            index += 1;
        }
        let params_raw = slice_lossy(bytes, params_start, index);
        let params = split_params(&params_raw);
        // Return type / where clause, then the body brace.
        let ret_start = index;
        while index < bytes.len() && bytes[index] != b'{' {
            match bytes[index] {
                b';' => break,
                b'"' => {
                    index = skip_string(bytes, index);
                    continue;
                }
                b'\'' => {
                    index = skip_char(bytes, index);
                    continue;
                }
                _ => index += 1,
            }
        }
        if bytes.get(index) != Some(&b'{') {
            continue;
        }
        let ret = slice_lossy(bytes, ret_start, index);
        let body_start = index + 1;
        let mut braces = 1i32;
        index = body_start;
        while index < bytes.len() && braces > 0 {
            match bytes[index] {
                b'"' => {
                    index = skip_string(bytes, index);
                    continue;
                }
                b'\'' => {
                    index = skip_char(bytes, index);
                    continue;
                }
                b'{' => braces += 1,
                b'}' => braces -= 1,
                _ => {}
            }
            index += 1;
        }
        let body_end = index.saturating_sub(1);
        items.push(FnItem {
            is_public: item_is_public(code, keyword),
            name,
            params,
            params_raw,
            ret,
            body: slice_lossy(bytes, body_start.min(body_end), body_end),
        });
        cursor = body_start;
    }
    items
}

/// Parameter names from a `(a: T, b: U)` list, in order.
///
/// Splits on top-level commas rather than regex-matching `ident:`, because `target: &std::path::Path`
/// contains three colons and a naive scan would report three parameters — silently shifting every
/// index in [`PARAMETERIZED_CACHE_HELPERS`].
fn split_params(raw: &str) -> Vec<String> {
    let trimmed = raw.trim().trim_start_matches('(').trim_end_matches(')');
    let bytes = trimmed.as_bytes();
    // Byte buffers throughout: a parameter list can carry non-ASCII inside a default string literal,
    // and slicing a `&str` mid-character panics.
    let mut parts: Vec<Vec<u8>> = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut depth = 0i32;
    let mut index = 0usize;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                let end = skip_string(bytes, index).min(bytes.len());
                current.extend_from_slice(&bytes[index..end]);
                index = end;
                continue;
            }
            b'\'' => {
                let end = skip_char(bytes, index).min(bytes.len());
                current.extend_from_slice(&bytes[index..end]);
                index = end;
                continue;
            }
            b'(' | b'<' | b'[' | b'{' => depth += 1,
            b')' | b'>' | b']' | b'}' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(std::mem::take(&mut current));
                index += 1;
                continue;
            }
            _ => {}
        }
        current.push(bytes[index]);
        index += 1;
    }
    parts.push(current);
    parts
        .into_iter()
        .map(|part| String::from_utf8_lossy(&part).into_owned())
        .filter_map(|part| {
            let head = part.split(':').next()?.trim().to_owned();
            let head = head.trim_start_matches("mut ").trim().to_owned();
            (!head.is_empty() && head.bytes().all(|b| b == b'_' || b.is_ascii_alphanumeric()))
                .then_some(head)
        })
        .collect()
}

/// Whether a `cfg(..)` predicate means "test builds only".
///
/// `test` and `all(test, …)` qualify. `any(test, …)` and `not(test)` deliberately do NOT — those
/// items ARE compiled in production, and removing them would blind the scan to real code.
fn cfg_is_test_only(predicate: &str) -> bool {
    let predicate = predicate.trim();
    if predicate == "test" {
        return true;
    }
    let Some(inner) = predicate
        .strip_prefix("all(")
        .and_then(|rest| rest.strip_suffix(')'))
    else {
        return false;
    };
    let mut depth = 0i32;
    let mut current = String::new();
    let mut members = Vec::new();
    for character in inner.chars() {
        match character {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => {
                members.push(std::mem::take(&mut current));
                continue;
            }
            _ => {}
        }
        current.push(character);
    }
    members.push(current);
    members.iter().any(|member| cfg_is_test_only(member))
}

/// Whether what follows a `#[cfg(…)]` attribute is a Rust *item* (which owns a `{ … }` or ends in
/// `;`) rather than a struct field or a struct-literal field (which ends at a comma).
fn attribute_precedes_an_item(rest: &str) -> bool {
    const ITEM_KEYWORDS: &[&str] = &[
        "pub",
        "fn",
        "mod",
        "impl",
        "struct",
        "enum",
        "union",
        "trait",
        "use",
        "const",
        "static",
        "type",
        "async",
        "unsafe",
        "extern",
        "macro_rules",
        "let",
    ];
    let rest = rest.trim_start();
    // A further attribute, or a block expression, still resolves to the item walk.
    if rest.starts_with('#') || rest.starts_with('{') {
        return true;
    }
    let word: String = rest
        .chars()
        .take_while(|c| *c == '_' || c.is_ascii_alphanumeric())
        .collect();
    ITEM_KEYWORDS.contains(&word.as_str())
}

/// Index just past the non-item thing a `#[cfg(test)]` at `start` applies to.
///
/// Three shapes reach here and they end differently, which is the whole reason this exists: a struct
/// field or struct-literal field ends at a `,`, a statement ends at a `;`, and a block-tailed
/// statement (`if let … { … }`) ends at its closing brace. Taking whichever comes first at depth zero
/// keeps the deletion tight; the previous "scan on to the next `{` or `;`" walk ran past a field's
/// comma and swallowed the following *function*, and with it any download that function performed.
fn end_of_cfg_member(bytes: &[u8], start: usize) -> usize {
    let mut index = start;
    let mut depth = 0i32;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                index = skip_string(bytes, index);
                continue;
            }
            b'\'' => {
                index = skip_char(bytes, index);
                continue;
            }
            b'{' if depth == 0 => {
                let mut braces = 0i32;
                while index < bytes.len() {
                    match bytes[index] {
                        b'"' => {
                            index = skip_string(bytes, index);
                            continue;
                        }
                        b'\'' => {
                            index = skip_char(bytes, index);
                            continue;
                        }
                        b'{' => braces += 1,
                        b'}' => {
                            braces -= 1;
                            if braces == 0 {
                                return index + 1;
                            }
                        }
                        _ => {}
                    }
                    index += 1;
                }
                return index;
            }
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                if depth == 0 {
                    // The last field before an enclosing closing brace carries no trailing comma.
                    return index;
                }
                depth -= 1;
            }
            b',' | b';' if depth == 0 => return index + 1,
            _ => {}
        }
        index += 1;
    }
    index
}

/// Remove `#[cfg(test)]`-only items (attribute plus the item that follows) from comment-stripped
/// source.
///
/// This is not an escape hatch: production code cannot call a `#[cfg(test)]` item, because it does
/// not exist in a non-test build. Without it, the `#[cfg(test)] mod tests` blocks inside production
/// files (real-weight smokes that legitimately download, temp-dir fixtures that legitimately join
/// `"cache"`) would flood both allow-lists with entries that can never run in production.
fn strip_cfg_test_items(code: &str) -> String {
    let bytes = code.as_bytes();
    // Byte buffer, not a `String`: source outside comments still carries non-ASCII (em dashes in
    // identifiers-adjacent prose, `…` in error strings), and indexing a `&str` at a byte offset that
    // lands inside a multi-byte character panics.
    let mut out: Vec<u8> = Vec::with_capacity(code.len());
    let mut index = 0usize;
    while index < bytes.len() {
        if !bytes[index..].starts_with(b"#[cfg(") {
            out.push(bytes[index]);
            index += 1;
            continue;
        }
        // Matching `]` for the attribute.
        let mut scan = index + 1;
        let mut depth = 0i32;
        while scan < bytes.len() {
            match bytes[scan] {
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        scan += 1;
                        break;
                    }
                }
                _ => {}
            }
            scan += 1;
        }
        let attribute = &code[index..scan.min(code.len())];
        let predicate = attribute
            .strip_prefix("#[cfg(")
            .and_then(|rest| rest.strip_suffix(")]"))
            .unwrap_or("");
        if !cfg_is_test_only(predicate) {
            out.extend_from_slice(attribute.as_bytes());
            index = scan;
            continue;
        }
        // A `#[cfg(test)]` on a struct FIELD or a struct-literal field is not an item: there is no
        // `{ … }` or `;` of its own, so the item walk below would run on to the NEXT item and delete
        // that instead — swallowing whole functions, and with them any download call they made. Those
        // end at a comma, so handle them separately.
        if !attribute_precedes_an_item(&code[scan.min(code.len())..]) {
            index = end_of_cfg_member(bytes, scan);
            continue;
        }
        // Drop the item this attribute applies to: everything through its `{ … }` block, or through
        // the `;` when it is a `mod foo;` / `use …;` / `const … ;`.
        let mut item = scan;
        while item < bytes.len() && bytes[item] != b'{' && bytes[item] != b';' {
            match bytes[item] {
                b'"' => {
                    item = skip_string(bytes, item);
                    continue;
                }
                b'\'' => {
                    item = skip_char(bytes, item);
                    continue;
                }
                _ => item += 1,
            }
        }
        match bytes.get(item) {
            Some(b';') => index = item + 1,
            Some(b'{') => {
                let mut braces = 1i32;
                item += 1;
                while item < bytes.len() && braces > 0 {
                    match bytes[item] {
                        b'"' => {
                            item = skip_string(bytes, item);
                            continue;
                        }
                        b'\'' => {
                            item = skip_char(bytes, item);
                            continue;
                        }
                        b'{' => braces += 1,
                        b'}' => braces -= 1,
                        _ => {}
                    }
                    item += 1;
                }
                index = item;
            }
            _ => index = scan,
        }
    }
    String::from_utf8(out).expect("dropping whole `#[cfg(test)]` items preserves UTF-8")
}

/// Every file in `tree` reachable only as test code.
///
/// Seeded from `#[cfg(test)] mod NAME;` declarations (detected by diffing the module declarations
/// before and after [`strip_cfg_test_items`]) and propagated through both `mod NAME;` and
/// `include!("…")` edges — `src/tests.rs` reaches its four submodules through `include!`, so a
/// `mod`-only walk would leave them classified as production.
///
/// This is what keeps THIS module out of its own scan: it is declared `#[cfg(test)] mod
/// job_time_download_guard;`, and its synthetic fixtures contain both `ensure_hf_cached_file(` and
/// `.join("cache")`.
fn test_only_files(tree: &BTreeMap<String, String>, root_module: &str) -> BTreeSet<String> {
    let mut children: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    let mut includes: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    let mut seeds: BTreeSet<String> = BTreeSet::new();

    for (rel, code) in tree {
        let declared = module_declarations(code);
        let production = module_declarations(&strip_cfg_test_items(code));
        for name in &declared {
            if let Some(target) = resolve_module(rel, name, tree, root_module) {
                children
                    .entry(rel.as_str())
                    .or_default()
                    .push(target.clone());
                if !production.contains(name) {
                    seeds.insert(target);
                }
            }
        }
        for path in include_paths(code) {
            let base = rel.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("");
            let target = if base.is_empty() {
                path
            } else {
                format!("{base}/{path}")
            };
            if tree.contains_key(&target) {
                includes.entry(rel.as_str()).or_default().push(target);
            }
        }
    }

    let mut out = seeds.clone();
    let mut frontier: Vec<String> = seeds.into_iter().collect();
    while let Some(current) = frontier.pop() {
        let downstream = children
            .get(current.as_str())
            .into_iter()
            .chain(includes.get(current.as_str()))
            .flatten();
        for target in downstream {
            if out.insert(target.clone()) {
                frontier.push(target.clone());
            }
        }
    }
    out
}

/// `mod NAME;` declarations in `code` (not `mod NAME { … }`, which needs no file).
fn module_declarations(code: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let bytes = code.as_bytes();
    let mut cursor = 0usize;
    while let Some(offset) = code[cursor..].find("mod ") {
        let keyword = cursor + offset;
        cursor = keyword + 4;
        if keyword
            .checked_sub(1)
            .map(|i| bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric())
            .unwrap_or(false)
        {
            continue;
        }
        let rest = &code[keyword + 4..];
        let name: String = rest
            .chars()
            .take_while(|c| *c == '_' || c.is_ascii_alphanumeric())
            .collect();
        if name.is_empty() {
            continue;
        }
        if rest[name.len()..].trim_start().starts_with(';') {
            out.insert(name);
        }
    }
    out
}

/// `include!("…")` paths in `code`.
fn include_paths(code: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while let Some(offset) = code[cursor..].find("include!(") {
        let start = cursor + offset + "include!(".len();
        cursor = start;
        let rest = code[start..].trim_start();
        let Some(body) = rest.strip_prefix('"') else {
            continue;
        };
        let Some(end) = body.find('"') else { continue };
        out.push(body[..end].to_owned());
    }
    out
}

/// Resolve `mod NAME;` declared in `rel` to a file in `tree`.
fn resolve_module(
    rel: &str,
    name: &str,
    tree: &BTreeMap<String, String>,
    root_module: &str,
) -> Option<String> {
    let dir = if rel == root_module || rel.ends_with("/mod.rs") {
        rel.rsplit_once('/').map(|(dir, _)| dir).unwrap_or("")
    } else {
        rel.strip_suffix(".rs")?
    };
    let prefix = if dir.is_empty() {
        String::new()
    } else {
        format!("{dir}/")
    };
    [
        format!("{prefix}{name}.rs"),
        format!("{prefix}{name}/mod.rs"),
    ]
    .into_iter()
    .find(|candidate| tree.contains_key(candidate))
}

// ---------------------------------------------------------------------------------------------
// Gate 1 — download reachability
// ---------------------------------------------------------------------------------------------

/// Wrappers: functions that reach a [`DOWNLOAD_PRIMITIVES`] member, directly or through other
/// wrappers.
///
/// Computed across the whole crate, not per file, because the wrapper and its callers routinely live
/// apart — `ensure_instantid_file` is defined in `image_jobs/instantid.rs` and called twelve times
/// from there and `image_jobs/pulid.rs`.
///
/// # Why this is a closure and not one hop
///
/// One hop is defeated by two lines: add `fn forward(c) { ensure_instantid_file(c) }` to a file
/// **already on** [`DOWNLOAD_SITES`], then call `forward` from a new lane. The new lane names no
/// primitive, no known wrapper and no `DownloadContext`, and the gate stays green.
///
/// # Why the closure does not become a census
///
/// The full transitive closure converges on `run_worker_loop` and flags 73 of 133 production files.
/// The fix is to grow the closure only **through functions defined in files that are already on
/// [`DOWNLOAD_SITES`]** — with one exception, the first hop, which is taken from anywhere.
///
/// That loses nothing. A wrapper defined in an unlisted file makes *that* file a new download site on
/// its own, because its body invokes a primitive; the gate is already red there. So the only
/// forwarders worth chasing are the ones hiding inside files the allow-list has already accepted, and
/// those are exactly the ones this walks.
///
/// [`DOWNLOAD_SITES`] members that are `Installer` plumbing (`lib.rs`, `model_jobs.rs`) are excluded
/// from the closure past the first hop: they are the dispatch layer, so following them is what
/// produces the census — `run_model_download_job` → the job-loop match arm → `run_worker_loop` →
/// every file that starts a worker. The Model Manager path is *supposed* to download; it is job lanes
/// that must not, and no job lane calls the installer dispatchers.
fn download_wrappers(files: &[SourceFile]) -> BTreeMap<String, String> {
    let primitives: BTreeSet<&str> = DOWNLOAD_PRIMITIVES.iter().copied().collect();
    let closure_files: BTreeSet<&str> = DOWNLOAD_SITES
        .iter()
        .filter(|(_, role)| *role == DownloadRole::JobTime)
        .map(|(rel, _)| *rel)
        .collect();

    // (may the closure grow through this fn, defining file, name, literal-blanked body)
    let items: Vec<(bool, String, String, String)> = files
        .iter()
        .flat_map(|file| {
            let in_closure = closure_files.contains(file.rel.as_str());
            let rel = file.rel.clone();
            fn_items(&file.code).into_iter().map(move |item| {
                let body = code_without_comments_or_literals(&item.body);
                (in_closure, rel.clone(), item.name, body)
            })
        })
        .collect();

    // Hop one, from anywhere: a function whose own body calls a primitive.
    let mut wrappers: BTreeMap<String, String> = BTreeMap::new();
    for (_, rel, name, body) in &items {
        if primitives.contains(name.as_str()) || wrappers.contains_key(name) {
            continue;
        }
        if primitives
            .iter()
            .any(|primitive| invokes_free_function(body, primitive))
        {
            wrappers.insert(name.clone(), rel.clone());
        }
    }

    // Remaining hops, confined. `frontier` keeps each pass proportional to what actually changed.
    let mut frontier: Vec<String> = wrappers.keys().cloned().collect();
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for (in_closure, rel, name, body) in &items {
            if !in_closure || primitives.contains(name.as_str()) || wrappers.contains_key(name) {
                continue;
            }
            if frontier
                .iter()
                .any(|callee| invokes_free_function(body, callee))
            {
                wrappers.insert(name.clone(), rel.clone());
                next.push(name.clone());
            }
        }
        frontier = next;
    }
    wrappers
}

/// Textual markers for "this expression performs an outbound HTTP request".
///
/// Every reqwest request terminates in `.send()`; `reqwest::get` is the one-shot free function. A
/// body that contains neither cannot itself put bytes on the wire.
const NETWORK_TERMINALS: &[&str] = &[".send()", "reqwest::get("];

/// Markers for "this expression obtains an HTTP client of its own".
const HTTP_CLIENT_CONSTRUCTORS: &[&str] = &[
    "reqwest::Client::new",
    "reqwest::Client::builder",
    "reqwest::ClientBuilder",
    "reqwest::blocking",
];

/// Every externally-visible `fn` in `downloads.rs` whose body transitively reaches an outbound HTTP
/// request.
///
/// This is what makes [`DOWNLOAD_PRIMITIVES`] a derived list rather than a hand-maintained one. A
/// private `fn` is excluded because `downloads` is a module: a job lane cannot name it, however much
/// network it touches. Everything else discovered here is callable, unqualified, from any file in the
/// crate — `lib.rs` does `use downloads::*;` — so it has to be classified.
fn network_fetch_surface(file: &SourceFile) -> BTreeSet<String> {
    let items = fn_items(&file.code);
    let bodies: Vec<String> = items
        .iter()
        .map(|item| code_without_comments_or_literals(&item.body))
        .collect();

    let mut reaching: BTreeSet<String> = BTreeSet::new();
    for (item, body) in items.iter().zip(&bodies) {
        if NETWORK_TERMINALS
            .iter()
            .chain(HTTP_CLIENT_CONSTRUCTORS)
            .any(|marker| body.contains(marker))
        {
            reaching.insert(item.name.clone());
        }
    }
    loop {
        let mut grew = false;
        for (item, body) in items.iter().zip(&bodies) {
            if reaching.contains(&item.name) {
                continue;
            }
            if reaching.iter().any(|callee| invokes(body, callee)) {
                reaching.insert(item.name.clone());
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }

    items
        .iter()
        .filter(|item| item.is_public && reaching.contains(&item.name))
        .map(|item| item.name.clone())
        .collect()
}

/// Why a file counts as making HTTP requests of its own.
///
/// Deliberately the two ends of a request and nothing in between: building a client that did not come
/// from `downloads.rs`'s timeout-configured factories, and terminating a request. A lane may still
/// *hold* a `&reqwest::Client` — the download primitives take one — but the moment it constructs one
/// or sends with one, it has left the sanctioned path.
fn http_client_reach(code_without_literals: &str) -> Vec<String> {
    HTTP_CLIENT_CONSTRUCTORS
        .iter()
        .chain(NETWORK_TERMINALS)
        .filter(|marker| code_without_literals.contains(**marker))
        .map(|marker| (*marker).to_owned())
        .collect()
}

/// Why a file counts as reaching a download: the primitive/wrapper names it invokes, plus a
/// `DownloadContext` literal if it builds one.
///
/// Takes the already-blanked code so a mention inside a string literal or a doc comment cannot count.
fn download_reach(code_without_literals: &str, watch: &BTreeMap<String, String>) -> Vec<String> {
    let mut reasons: Vec<String> = watch
        .iter()
        .filter(|(name, _)| invokes_free_function(code_without_literals, name))
        .map(|(name, definer)| format!("{name} <- {definer}"))
        .collect();
    reasons.extend(value_references(code_without_literals, watch));
    if constructs_download_context(code_without_literals) {
        reasons.push("DownloadContext { … }".to_owned());
    }
    reasons
}

/// [`DOWNLOAD_PRIMITIVES`] + [`download_wrappers`] + every `as` rename of anything already in the
/// map, to a fixpoint.
fn watch_map(files: &[SourceFile]) -> BTreeMap<String, String> {
    let mut watch: BTreeMap<String, String> = DOWNLOAD_PRIMITIVES
        .iter()
        .map(|name| ((*name).to_owned(), "downloads.rs".to_owned()))
        .collect();
    watch.extend(download_wrappers(files));
    let blanked: Vec<String> = files
        .iter()
        .map(|file| file.code_without_literals())
        .collect();
    loop {
        let mut grew = false;
        for code in &blanked {
            for (alias, original) in alias_edges(code, &watch) {
                if let std::collections::btree_map::Entry::Vacant(slot) = watch.entry(alias) {
                    slot.insert(format!("`as` rename of {original}"));
                    grew = true;
                }
            }
        }
        if !grew {
            break;
        }
    }
    watch
}

/// Byte ranges of `use …;` items.
///
/// A name inside an import is not a use of it — `use crate::downloads::{ensure_cached_file, …};` has
/// to stay silent, which is the negative [`scanner_detects_a_new_direct_call_site`] pins. Everything
/// OUTSIDE these ranges that names a primitive without calling it is a real value reference.
fn use_item_ranges(code: &str) -> Vec<(usize, usize)> {
    let bytes = code.as_bytes();
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while let Some(offset) = code[cursor..].find("use ") {
        let start = cursor + offset;
        cursor = start + 4;
        let boundary = start
            .checked_sub(1)
            .map(|index| !(bytes[index] == b'_' || bytes[index].is_ascii_alphanumeric()))
            .unwrap_or(true);
        if !boundary {
            continue;
        }
        let mut index = start;
        let mut depth = 0i32;
        while index < bytes.len() {
            match bytes[index] {
                b'{' => depth += 1,
                b'}' => depth -= 1,
                b';' if depth <= 0 => {
                    index += 1;
                    break;
                }
                _ => {}
            }
            index += 1;
        }
        out.push((start, index));
    }
    out
}

/// `… PRIMITIVE as ALIAS …` renames, so `use crate::downloads::ensure_hf_cached_file as fetch_it;`
/// puts `fetch_it` on the watch list.
///
/// An alias is a one-line bypass of any name-based gate, and it need not be spelled in the lane that
/// uses it — `pub use … as …;` in one file arms a call in another — so this is collected crate-wide
/// like the wrapper closure, and iterated so an alias of an alias is caught too.
fn alias_edges(code: &str, watch: &BTreeMap<String, String>) -> Vec<(String, String)> {
    let bytes = code.as_bytes();
    let mut out = Vec::new();
    for name in watch.keys() {
        let mut cursor = 0usize;
        while let Some(offset) = code[cursor..].find(name.as_str()) {
            let start = cursor + offset;
            let end = start + name.len();
            cursor = end;
            let bounded = start
                .checked_sub(1)
                .map(|index| !(bytes[index] == b'_' || bytes[index].is_ascii_alphanumeric()))
                .unwrap_or(true)
                && !bytes
                    .get(end)
                    .is_some_and(|byte| *byte == b'_' || byte.is_ascii_alphanumeric());
            if !bounded {
                continue;
            }
            let Some(after) = code[end..].trim_start().strip_prefix("as") else {
                continue;
            };
            // `as` must be its own token — `assert_eq!` must not read as a rename.
            if after.starts_with(|character: char| {
                character == '_' || character.is_ascii_alphanumeric()
            }) {
                continue;
            }
            let alias: String = after
                .trim_start()
                .chars()
                .take_while(|character| *character == '_' || character.is_ascii_alphanumeric())
                .collect();
            if !alias.is_empty() {
                out.push((alias, name.clone()));
            }
        }
    }
    out
}

/// The shortest discovered-wrapper name the value-reference rule will chase.
///
/// Without a floor the rule cries wolf: a wrapper named `stage` — five letters, and a perfectly real
/// wrapper shape — matched the bare word in six unrelated production files (`audio_jobs.rs`,
/// `caption_jobs.rs`, `progress.rs`, `training_jobs.rs`, `analysis_jobs_common.rs`,
/// `prompt_refine_jobs.rs`). The failure direction is noisy rather than blind, which is the safe
/// one — but a gate that cries wolf gets weakened by the next person who hits it. Every
/// [`DOWNLOAD_PRIMITIVES`] member is chased regardless of length; the floor applies only to
/// discovered wrappers and aliases.
const MIN_VALUE_REFERENCE_NAME_LEN: usize = 12;

/// Watched names referenced as VALUES rather than called: `let grab = ensure_hf_cached_file;`
/// followed by `grab(…)` downloads exactly as much as calling it directly, and matches no `NAME(`
/// pattern anywhere in the file.
///
/// Restricted twice over, because a bare-name match is the loosest rule in this module: the name must
/// be a primitive or long enough not to collide ([`MIN_VALUE_REFERENCE_NAME_LEN`]), and it must sit
/// where a function VALUE is actually bound or passed — after `=`, or directly after `(` or `,` in an
/// argument list.
fn value_references(code: &str, watch: &BTreeMap<String, String>) -> Vec<String> {
    let bytes = code.as_bytes();
    let imports = use_item_ranges(code);
    let mut out = Vec::new();
    for (name, definer) in watch {
        if name.len() < MIN_VALUE_REFERENCE_NAME_LEN
            && !DOWNLOAD_PRIMITIVES.contains(&name.as_str())
        {
            continue;
        }
        let mut cursor = 0usize;
        while let Some(offset) = code[cursor..].find(name.as_str()) {
            let start = cursor + offset;
            let end = start + name.len();
            cursor = end;
            if imports
                .iter()
                .any(|(from, to)| start >= *from && start < *to)
            {
                continue;
            }
            let before = code[..start].trim_end();
            let preceded_by_ident = start
                .checked_sub(1)
                .map(|index| bytes[index] == b'_' || bytes[index].is_ascii_alphanumeric())
                .unwrap_or(false);
            if preceded_by_ident
                || bytes
                    .get(end)
                    .is_some_and(|byte| *byte == b'_' || byte.is_ascii_alphanumeric())
                || (before.ends_with('.') && !before.ends_with(".."))
                || strip_trailing_keyword(before, "fn").is_some()
            {
                continue;
            }
            // A call is `download_reach`'s job and a rename is `alias_edges`'. What is left — the
            // name followed by anything else — is the function itself, taken as a value.
            let next = code[end..].trim_start();
            if next.starts_with('(')
                || next.starts_with("as")
                || next.starts_with('!')
                || next.starts_with("::")
            {
                continue;
            }
            // ...and it has to sit where a function value is bound or passed, or the bare word
            // matches prose-shaped identifiers across the crate.
            if !(before.ends_with('=') && !before.ends_with("==") && !before.ends_with("!="))
                && !before.ends_with('(')
                && !before.ends_with(',')
            {
                continue;
            }
            out.push(format!("{name} (as a value) <- {definer}"));
            break;
        }
    }
    out
}

/// Whether `code` builds a `DownloadContext` struct literal.
///
/// The type is a plain struct with no privileged constructor, so building one is the actual ticket to
/// the network — and it is how a lane reaches a download helper it never names. A `use` import or a
/// `&DownloadContext<'_>` parameter is not a construction and must not count.
fn constructs_download_context(code: &str) -> bool {
    let mut cursor = 0usize;
    let bytes = code.as_bytes();
    while let Some(offset) = code[cursor..].find("DownloadContext") {
        let start = cursor + offset;
        let end = start + "DownloadContext".len();
        cursor = end;
        let preceded_by_ident = start
            .checked_sub(1)
            .map(|i| bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric())
            .unwrap_or(false);
        if preceded_by_ident {
            continue;
        }
        if code[end..].trim_start().starts_with('{') {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------------------------
// Gate 2 — `<data_dir>/cache` destinations
// ---------------------------------------------------------------------------------------------

/// `const NAME: &str = "value"` / `static NAME: &str = "value"` bindings, so a destination hidden
/// behind `POSE_UPLOADS_CACHE_DIR` reads as `pose-uploads` in the allow-list rather than as an opaque
/// constant name that churns when someone renames it.
fn string_constants(files: &[SourceFile]) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for file in files {
        for keyword in ["const ", "static "] {
            let mut cursor = 0usize;
            while let Some(offset) = file.code[cursor..].find(keyword) {
                let start = cursor + offset + keyword.len();
                cursor = start;
                let rest = &file.code[start..];
                let name: String = rest
                    .chars()
                    .take_while(|c| *c == '_' || c.is_ascii_alphanumeric())
                    .collect();
                if name.is_empty() {
                    continue;
                }
                let after = rest[name.len()..].trim_start();
                let Some(after) = after.strip_prefix(':') else {
                    continue;
                };
                let after = after.trim_start();
                if !after.starts_with("&str") && !after.starts_with("&'static str") {
                    continue;
                }
                let Some(equals) = after.find('=') else {
                    continue;
                };
                let value = after[equals + 1..].trim_start();
                if let Some(joined) = concat_literal_value(value) {
                    out.insert(name, joined);
                    continue;
                }
                let Some(literal) = value.strip_prefix('"') else {
                    continue;
                };
                let Some(end) = literal.find('"') else {
                    continue;
                };
                out.insert(name, literal[..end].to_owned());
            }
        }
    }
    out
}

/// The value of a Rust string literal, including the raw forms.
///
/// `r"cache"` and `r#"cache"#` are the same destination as `"cache"` to the filesystem and a
/// different byte sequence to a naive scan, which is exactly how a literal-keyed gate gets walked
/// past. One function so every place that reads a literal agrees.
fn string_literal_value(argument: &str) -> Option<String> {
    let argument = argument.trim();
    if let Some(rest) = argument.strip_prefix('r') {
        let hashes = "#".repeat(rest.len() - rest.trim_start_matches('#').len());
        return rest
            .strip_prefix(&format!("{hashes}\""))?
            .strip_suffix(&format!("\"{hashes}"))
            .map(str::to_owned);
    }
    argument
        .strip_prefix('"')?
        .strip_suffix('"')
        .map(str::to_owned)
}

/// The value of a `concat!("a", "b")` initialiser, when every argument is a plain literal.
///
/// `const ROOT: &str = concat!("ca", "che");` is otherwise an opaque `{ROOT}` to [`subdir_label`],
/// and therefore a cache root the scan cannot recognise.
fn concat_literal_value(value: &str) -> Option<String> {
    let inner = value.trim().strip_prefix("concat!")?.trim_start();
    let inner = inner.strip_prefix('(')?;
    let end = inner.find(')')?;
    let mut joined = String::new();
    for part in inner[..end].split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        joined.push_str(&string_literal_value(part)?);
    }
    Some(joined)
}

/// Render one `.join(…)` argument as an allow-list directory name.
///
/// A literal (plain or raw) becomes itself; a `const` resolves through [`string_constants`]; anything
/// else becomes a braced marker (`{cache_dir}`, `{expr}`) so an opaque destination is loud rather
/// than absent.
///
/// This is applied to the **root** segment as well as the subdir. `const ROOT: &str = "cache";
/// d.join(ROOT).join("weights")` and `d.join(r"cache").join("weights")` are the two cheapest ways to
/// spell a cache root that a `find(".join(\"cache\")")` scan cannot see, and both resolve here.
fn subdir_label(argument: &str, constants: &BTreeMap<String, String>) -> String {
    let argument = argument.trim().trim_start_matches('&').trim();
    // `join(Path::new("cache"))`, `join(std::path::Path::new("cache"))` and
    // `join("cache".to_string())` are all the same path segment as `join("cache")`. Matching the
    // wrapper on a bare prefix left the `::`-qualified spelling and the owned-conversion spelling as
    // one-token bypasses, so the qualifier goes first and the conversion suffix last.
    // Looped to a fixpoint, and conversion BEFORE wrapper: `Path::new("cache").to_owned()` wears
    // both at once, and stripping the wrapper first would take the conversion's closing paren as the
    // wrapper's.
    let mut argument = argument;
    loop {
        let stripped = [
            "to_string()",
            "to_owned()",
            "as_str()",
            "as_ref()",
            "into()",
        ]
        .iter()
        .find_map(|conversion| {
            argument
                .strip_suffix(*conversion)
                .and_then(|rest| rest.strip_suffix('.'))
        })
        .map(str::trim)
        .unwrap_or(argument);
        let unqualified = stripped
            .rsplit_once("::")
            .filter(|(_, tail)| tail.starts_with("new(") || tail.starts_with("from("))
            .map(|(_, tail)| tail)
            .unwrap_or(stripped);
        let unwrapped = ["new(", "from("]
            .iter()
            .find_map(|wrapper| {
                unqualified
                    .strip_prefix(*wrapper)
                    .and_then(|rest| rest.strip_suffix(')'))
            })
            .map(str::trim)
            .unwrap_or(stripped);
        if unwrapped == argument {
            break;
        }
        argument = unwrapped;
    }
    if let Some(literal) = string_literal_value(argument) {
        return literal;
    }
    if !argument.is_empty()
        && argument
            .bytes()
            .all(|b| b == b'_' || b.is_ascii_alphanumeric())
    {
        return constants
            .get(argument)
            .cloned()
            .unwrap_or_else(|| format!("{{{argument}}}"));
    }
    "{expr}".to_owned()
}

/// The balanced first argument of the call whose `(` sits at `open`, and the index just past the
/// matching `)`.
fn balanced_first_argument(bytes: &[u8], open: usize) -> Option<(String, usize)> {
    if bytes.get(open) != Some(&b'(') {
        return None;
    }
    let mut depth = 0i32;
    let mut index = open;
    let mut argument_end: Option<usize> = None;
    while index < bytes.len() {
        match bytes[index] {
            b'"' => {
                index = skip_string(bytes, index);
                continue;
            }
            b'\'' => {
                index = skip_char(bytes, index);
                continue;
            }
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    let end = argument_end.unwrap_or(index);
                    return Some((slice_lossy(bytes, open + 1, end), index + 1));
                }
            }
            b',' if depth == 1 && argument_end.is_none() => argument_end = Some(index),
            _ => {}
        }
        index += 1;
    }
    None
}

/// If `code` continues at `at` (after whitespace/newlines) with `.join(…)` or `.push(…)`, the
/// argument and the index past the call.
fn next_path_segment(code: &str, at: usize) -> Option<(String, usize)> {
    let bytes = code.as_bytes();
    let mut index = at;
    while index < bytes.len() && (bytes[index] as char).is_whitespace() {
        index += 1;
    }
    let rest = &code[index.min(code.len())..];
    let method = if rest.starts_with(".join(") {
        ".join("
    } else if rest.starts_with(".push(") {
        ".push("
    } else {
        return None;
    };
    balanced_first_argument(bytes, index + method.len() - 1)
}

/// Every `.join(…)` / `.push(…)` in `code`, as `(argument, index past the call)`.
fn path_segment_calls(code: &str) -> Vec<(String, usize)> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor < code.len() {
        let Some(offset) = code[cursor..].find('.') else {
            break;
        };
        let dot = cursor + offset;
        cursor = dot + 1;
        let Some((argument, end)) = next_path_segment(code, dot) else {
            continue;
        };
        out.push((argument, end));
    }
    out
}

/// The `<data_dir>/cache/<subdir>` destinations named in `code`, plus whether it produces the bare
/// cache root.
///
/// Three spellings, all live in the tree today:
/// * `.join(<root>).join(<subdir>)` (and the `.push` variant), where `<root>` is anything that
///   *resolves* to `cache` — a plain literal, a raw literal, or a `const`;
/// * `.join(<root>)` with nothing after it, which yields `<cache-root>`; and
/// * a combined `"cache/<subdir>/…"` path literal — `model_jobs.rs`'s `CONVERSION_LOCKS_DIR` and the
///   three legacy person/pose roots use this one, and a scanner that only looked for `.join("cache")`
///   would miss all four.
fn cache_destinations(code: &str, constants: &BTreeMap<String, String>) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (argument, end) in path_segment_calls(code) {
        if subdir_label(&argument, constants) != "cache" {
            continue;
        }
        match next_path_segment(code, end) {
            Some((subdir, _)) => {
                out.insert(subdir_label(&subdir, constants));
            }
            None => {
                out.insert("<cache-root>".to_owned());
            }
        }
    }
    // A `cache/<subdir>` segment inside ANY string literal, not only one that starts with it:
    // `format!("{}/cache/{sub}", data_dir.display())` is the same destination written differently.
    let bytes = code.as_bytes();
    let mut cursor = 0usize;
    while let Some(offset) = code[cursor..].find("cache/") {
        let start = cursor + offset;
        cursor = start + "cache/".len();
        if !matches!(
            start.checked_sub(1).map(|index| bytes[index]),
            Some(b'"') | Some(b'/')
        ) {
            continue;
        }
        let rest = &code[cursor..];
        let end = rest.find(['"', '/']).unwrap_or(rest.len());
        let segment = &rest[..end];
        out.insert(if segment.is_empty() {
            "<cache-root>".to_owned()
        } else if segment.contains(['{', '}']) {
            "{expr}".to_owned()
        } else {
            segment.to_owned()
        });
    }
    out
}

/// Functions that build a cache root from one of their own parameters, as `(name, parameter index)`.
///
/// Rediscovered structurally rather than listed, so [`PARAMETERIZED_CACHE_HELPERS`] cannot silently
/// fall behind a new helper.
fn discover_parameterized_helpers(
    files: &[SourceFile],
    constants: &BTreeMap<String, String>,
) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for file in files {
        for item in fn_items(&file.code) {
            for (argument, end) in path_segment_calls(&item.body) {
                if subdir_label(&argument, constants) != "cache" {
                    continue;
                }
                let Some((subdir, _)) = next_path_segment(&item.body, end) else {
                    continue;
                };
                let subdir = subdir.trim().trim_start_matches('&').trim();
                if let Some(index) = item.params.iter().position(|p| p == subdir) {
                    out.insert(item.name.clone(), index);
                }
            }
        }
    }
    out
}

/// Functions whose body hands out the **bare** cache root, as file-relative `(file, fn name)` pairs.
///
/// The counterpart of [`discover_parameterized_helpers`] for the root rather than the subdir: a
/// function that ends a path expression at `cache` is producing the root for somebody else to join
/// onto, and [`CACHE_ROOT_PRODUCERS`] has to name it.
fn discover_cache_root_producers(
    files: &[SourceFile],
    constants: &BTreeMap<String, String>,
) -> BTreeSet<(String, String)> {
    let mut out = BTreeSet::new();
    for file in files {
        for item in fn_items(&file.code) {
            for (argument, end) in path_segment_calls(&item.body) {
                if subdir_label(&argument, constants) != "cache" {
                    continue;
                }
                if next_path_segment(&item.body, end).is_none() {
                    out.insert((file.rel.clone(), item.name.clone()));
                }
            }
        }
    }
    out
}

/// What each call to a registered [`CACHE_ROOT_PRODUCERS`] function joins onto the root it returns.
///
/// `cache_root(data_dir).join("new-weights")` names a destination while containing no `cache`
/// literal of its own; this attributes it to the calling file.
fn cache_root_producer_destinations(
    code: &str,
    producers: &BTreeSet<&str>,
    constants: &BTreeMap<String, String>,
) -> BTreeSet<String> {
    let bytes = code.as_bytes();
    let mut out = BTreeSet::new();
    for producer in producers {
        let mut cursor = 0usize;
        while let Some(offset) = code[cursor..].find(*producer) {
            let start = cursor + offset;
            let end = start + producer.len();
            cursor = end;
            let preceded_by_ident = start
                .checked_sub(1)
                .map(|i| bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric())
                .unwrap_or(false);
            if preceded_by_ident {
                continue;
            }
            let mut open = end;
            while open < bytes.len() && (bytes[open] as char).is_whitespace() {
                open += 1;
            }
            if bytes.get(open) != Some(&b'(') {
                continue;
            }
            let before = code[..start].trim_end();
            if strip_trailing_keyword(before, "fn").is_some() {
                continue;
            }
            let Some((_, past)) = balanced_first_argument(bytes, open) else {
                continue;
            };
            if let Some((subdir, _)) = next_path_segment(code, past) {
                out.insert(subdir_label(&subdir, constants));
            }
        }
    }
    out
}

/// The literal (or `const`) each call to a registered parameterized helper passes in the subdir
/// position, so the caller — which may contain no `"cache"` text at all — is attributed the
/// destination it is really naming.
fn parameterized_call_destinations(
    code: &str,
    constants: &BTreeMap<String, String>,
) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (helper, index) in PARAMETERIZED_CACHE_HELPERS {
        for args in call_arguments(code, helper) {
            if let Some(argument) = args.get(*index) {
                out.insert(subdir_label(argument, constants));
            }
        }
    }
    out
}

/// Top-level argument expressions of every call to `name` in `code`.
fn call_arguments(code: &str, name: &str) -> Vec<Vec<String>> {
    let bytes = code.as_bytes();
    let mut calls = Vec::new();
    let mut cursor = 0usize;
    while let Some(offset) = code[cursor..].find(name) {
        let start = cursor + offset;
        let end = start + name.len();
        cursor = end;
        let preceded_by_ident = start
            .checked_sub(1)
            .map(|i| bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric())
            .unwrap_or(false);
        if preceded_by_ident {
            continue;
        }
        let mut open = end;
        while open < bytes.len() && (bytes[open] as char).is_whitespace() {
            open += 1;
        }
        if bytes.get(open) != Some(&b'(') {
            continue;
        }
        let before = code[..start].trim_end();
        if before.ends_with("fn") && before[..before.len() - 2].ends_with(char::is_whitespace) {
            continue;
        }
        let mut args: Vec<Vec<u8>> = Vec::new();
        let mut current: Vec<u8> = Vec::new();
        let mut depth = 0i32;
        let mut index = open;
        while index < bytes.len() {
            match bytes[index] {
                b'"' => {
                    let stop = skip_string(bytes, index).min(bytes.len());
                    current.extend_from_slice(&bytes[index..stop]);
                    index = stop;
                    continue;
                }
                b'\'' => {
                    let stop = skip_char(bytes, index).min(bytes.len());
                    current.extend_from_slice(&bytes[index..stop]);
                    index = stop;
                    continue;
                }
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        args.push(std::mem::take(&mut current));
                        break;
                    }
                }
                b',' if depth == 1 => {
                    args.push(std::mem::take(&mut current));
                    index += 1;
                    continue;
                }
                _ => {}
            }
            if !(depth == 1 && bytes[index] == b'(') {
                current.push(bytes[index]);
            }
            index += 1;
        }
        calls.push(
            args.into_iter()
                .map(|arg| String::from_utf8_lossy(&arg).trim().to_owned())
                .filter(|arg| !arg.is_empty())
                .collect(),
        );
    }
    calls
}

// ---------------------------------------------------------------------------------------------
// The gates
// ---------------------------------------------------------------------------------------------

fn worker_sources() -> Vec<SourceFile> {
    let (_, src, root, _) = SCANNED_CRATES[0];
    production_sources(&repo_root().join(src), root)
}

fn worker_file(files: &[SourceFile], rel: &str) -> SourceFile {
    files
        .iter()
        .find(|file| file.rel == rel)
        .unwrap_or_else(|| panic!("the worker's production tree must contain {rel}"))
        .clone()
}

/// The production sources of every scanned crate, as `(label, files)`.
fn all_production_sources() -> Vec<(&'static str, Vec<SourceFile>)> {
    SCANNED_CRATES
        .iter()
        .map(|(label, src, root, minimum)| {
            let files = production_sources(&repo_root().join(src), root);
            // A floor, not a census: this catches "the walk found nothing", which is the only way a
            // set-equality gate can silently pass. Deliberately well below each crate's real count so
            // ordinary refactors do not trip it.
            assert!(
                files.len() >= *minimum,
                "{label}: scanned only {} production files, expected at least {minimum} — the walk \
                 is broken, not the crate",
                files.len()
            );
            (*label, files)
        })
        .collect()
}

/// One `string_constants` map spanning every scanned crate.
///
/// `POSE_UPLOADS_CACHE_DIR` and `KEYPOINT_UPLOADS_CACHE_DIR` are declared in sceneworks-core and used
/// by the worker AND the rust-api, so a per-crate map would leave four real destinations reading as
/// opaque `{CONST}`.
fn all_string_constants(crates: &[(&'static str, Vec<SourceFile>)]) -> BTreeMap<String, String> {
    crates
        .iter()
        .flat_map(|(_, files)| string_constants(files))
        .collect()
}

/// **AC9, half one.** The set of production worker files that reach a download primitive must equal
/// [`DOWNLOAD_SITES`] exactly.
///
/// Exact equality, not containment, in both directions on purpose:
/// * an EXTRA file means a new job-time download landed — the regression this epic exists to stop;
/// * a MISSING file means a migration slice removed one and did not ratchet the list down, which is
///   how an allow-list quietly turns back into a wish list.
#[test]
fn job_time_download_sites_match_the_allow_list() {
    let files = worker_sources();
    assert!(
        files.len() > 100,
        "expected the worker's production tree, scanned only {} files — the walk is broken, not the \
         crate",
        files.len()
    );

    let wrappers = download_wrappers(&files);
    assert!(
        wrappers.contains_key("ensure_instantid_file"),
        "the one-hop wrapper resolver found no `ensure_instantid_file`, so `image_jobs/pulid.rs` — \
         which reaches Hugging Face ONLY through it — would be invisible. Wrapper detection is \
         broken."
    );
    let watch = watch_map(&files);

    let mut found: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for file in &files {
        let reasons = download_reach(&file.code_without_literals(), &watch);
        if !reasons.is_empty() {
            found.insert(file.rel.clone(), reasons);
        }
    }

    let allowed: BTreeMap<&str, DownloadRole> = DOWNLOAD_SITES.iter().copied().collect();
    assert_eq!(
        allowed.len(),
        DOWNLOAD_SITES.len(),
        "DOWNLOAD_SITES has a duplicate path"
    );

    let added: Vec<String> = found
        .iter()
        .filter(|(rel, _)| !allowed.contains_key(rel.as_str()))
        .map(|(rel, reasons)| format!("{rel}  ({})", reasons.join(", ")))
        .collect();
    let migrated: Vec<&str> = allowed
        .keys()
        .filter(|rel| !found.contains_key(**rel))
        .copied()
        .collect();

    assert!(
        added.is_empty(),
        "NEW job-time download site(s) — epic 17625 (AC9) forbids this:\n  {}\n\nA render must not \
         fetch weights. Declare the component in the model manifest and resolve it with \
         `crate::downloads::resolve_hf_component_file` (cache-only, takes no `DownloadContext`), so a \
         missing install becomes an actionable \"install it from the Model Manager\" error instead of \
         a silent multi-gigabyte download inside a job.",
        added.join("\n  ")
    );
    assert!(
        migrated.is_empty(),
        "site(s) migrated off the download helpers but still listed in DOWNLOAD_SITES:\n  {}\n\nThis \
         is the ratchet: remove the entry in the same commit that removes the call, so the list \
         always states the real remaining population.",
        migrated.join("\n  ")
    );

    // The epic's remaining-work number, pinned. Set equality above already forces the allow-list to
    // describe the tree; what it CANNOT see is a role. Classifying a genuine job-time download as
    // `Installer` would otherwise be a silent way to keep the population flat while it grows. Pinning
    // the count means every role edit is a deliberate one-line diff a reviewer must agree with.
    //
    // **This number only ever goes DOWN.** Each migration slice deletes its entry and lowers it in
    // the same commit; raising it is the thing AC9 forbids.
    let job_time = allowed
        .values()
        .filter(|role| **role == DownloadRole::JobTime)
        .count();
    assert_eq!(
        job_time, JOB_TIME_DOWNLOAD_SITES_REMAINING,
        "the number of job-time download sites changed. If a migration slice removed one, lower \
         JOB_TIME_DOWNLOAD_SITES_REMAINING in the same commit. If this went UP, epic 17625 just \
         regressed — see the AC9 note on DOWNLOAD_SITES."
    );
}

/// **AC9, half two.** The set of `<data_dir>/cache/<subdir>` destinations must equal
/// [`CACHE_DESTINATIONS`] exactly, across all four crates that can write under the data dir.
#[test]
fn cache_destinations_match_the_allow_list() {
    let crates = all_production_sources();
    let constants = all_string_constants(&crates);
    let producers: BTreeSet<&str> = CACHE_ROOT_PRODUCERS
        .iter()
        .map(|(_, _, name)| *name)
        .collect();

    let mut found: BTreeSet<(String, String, String)> = BTreeSet::new();
    let mut scanned = 0usize;
    for (label, files) in &crates {
        scanned += files.len();
        for file in files {
            let mut subdirs = cache_destinations(&file.code, &constants);
            subdirs.extend(parameterized_call_destinations(&file.code, &constants));
            subdirs.extend(cache_root_producer_destinations(
                &file.code, &producers, &constants,
            ));
            for subdir in subdirs {
                found.insert(((*label).to_owned(), file.rel.clone(), subdir));
            }
        }
    }
    assert!(
        scanned > 200,
        "expected every scanned production tree, scanned only {scanned} files"
    );

    let allowed: BTreeMap<(String, String, String), CacheRole> = CACHE_DESTINATIONS
        .iter()
        .map(|(label, rel, subdir, role)| {
            (
                ((*label).to_owned(), (*rel).to_owned(), (*subdir).to_owned()),
                *role,
            )
        })
        .collect();
    assert_eq!(
        allowed.len(),
        CACHE_DESTINATIONS.len(),
        "CACHE_DESTINATIONS has a duplicate (crate, file, subdir) entry"
    );

    let added: Vec<String> = found
        .iter()
        .filter(|key| !allowed.contains_key(*key))
        .map(|(label, rel, subdir)| format!("{label}/{rel}  ->  <data_dir>/cache/{subdir}"))
        .collect();
    let removed: Vec<String> = allowed
        .keys()
        .filter(|key| !found.contains(*key))
        .map(|(label, rel, subdir)| format!("{label}/{rel}  ->  <data_dir>/cache/{subdir}"))
        .collect();

    assert!(
        added.is_empty(),
        "NEW `<data_dir>/cache` destination(s) — epic 17625 (AC9) forbids a new weight \
         destination:\n  {}\n\nIf this holds MODEL WEIGHTS, it must not exist: install them into the \
         app-managed Hugging Face cache through the Model Manager and resolve them with \
         `resolve_hf_component_file`. If it genuinely holds scratch/upload/runtime state, add it to \
         CACHE_DESTINATIONS with the matching role and a one-line comment saying what it holds.",
        added.join("\n  ")
    );
    assert!(
        removed.is_empty(),
        "destination(s) listed in CACHE_DESTINATIONS that no longer exist in source:\n  {}\n\nRemove \
         the entry in the same commit that removes the destination, so the list keeps stating the \
         real population.",
        removed.join("\n  ")
    );

    // Set equality above cannot see a ROLE. A new weights destination classified `Scratch` would
    // otherwise slip through as an ordinary allow-list edit, so the weights population is pinned to a
    // literal: any change to it, in either direction, is a diff a reviewer has to agree with.
    //
    // **This number only ever goes DOWN.** Each migration slice deletes its destination and lowers it
    // in the same commit; raising it is precisely the regression AC9 forbids.
    let weights = allowed
        .values()
        .filter(|role| **role == CacheRole::Weights)
        .count();
    assert_eq!(
        weights, WEIGHTS_CACHE_DESTINATIONS_REMAINING,
        "the number of `Weights` destinations under `<data_dir>/cache` changed. If a migration slice \
         removed one, lower WEIGHTS_CACHE_DESTINATIONS_REMAINING in the same commit. If this went UP, \
         a new weight destination just landed outside the app-managed Hugging Face cache, which is \
         what AC9 forbids."
    );
}

/// **AC9, the surface the other two gates stand on.** [`DOWNLOAD_PRIMITIVES`] must name the whole
/// externally-visible fetching surface of `downloads.rs`, minus an explicitly-classified
/// [`NON_WEIGHT_FETCHERS`] list.
///
/// Without this the primitive list is four hand-written strings, and a fifth fetcher added to
/// `downloads.rs` — or three that already existed — is invisible to the reachability scan no matter
/// how many lanes call it.
#[test]
fn downloads_fetch_surface_is_registered() {
    let files = worker_sources();
    let downloads = worker_file(&files, "downloads.rs");

    let discovered = network_fetch_surface(&downloads);
    assert!(
        discovered.contains("ensure_hf_cached_file") && discovered.contains("ensure_cached_file"),
        "the fetch-surface walk did not rediscover the helpers sc-17637 names, so it is broken and \
         proving nothing: {discovered:?}"
    );

    let registered: BTreeSet<String> = DOWNLOAD_PRIMITIVES
        .iter()
        .chain(NON_WEIGHT_FETCHERS)
        .map(|name| (*name).to_owned())
        .collect();
    assert_eq!(
        DOWNLOAD_PRIMITIVES.len() + NON_WEIGHT_FETCHERS.len(),
        registered.len(),
        "a name appears in both DOWNLOAD_PRIMITIVES and NON_WEIGHT_FETCHERS"
    );

    // Two ways a NON_WEIGHT_FETCHERS entry could still put the bytes in a job lane's hands: it takes
    // a filesystem destination and writes there, or it RETURNS the bytes and lets the caller write
    // them. The signature scan covers the parameter list AND the return type and where clause
    // together, because `fn probe<P>(…, dest: P) where P: AsRef<Path>` keeps the giveaway in the
    // where clause rather than in the parameter list.
    let items = fn_items(&downloads.code);
    let smuggled: Vec<String> = NON_WEIGHT_FETCHERS
        .iter()
        .filter_map(|name| {
            let item = items.iter().find(|item| item.name == *name)?;
            let signature =
                format!("{}{}", item.params_raw, item.ret).replace(char::is_whitespace, "");
            let returned = item.ret.replace(char::is_whitespace, "");
            let reason = if signature.contains("Path") {
                "takes a filesystem destination"
            } else if ["Vec<u8>", "Bytes", "[u8]"]
                .iter()
                .any(|shape| returned.contains(shape))
            {
                "returns raw bytes to its caller"
            } else {
                return None;
            };
            Some(format!("{name}{} — {reason}", item.params_raw.trim()))
        })
        .collect();
    assert!(
        smuggled.is_empty(),
        "NON_WEIGHT_FETCHERS member(s) that can put remote bytes in a caller's hands:\n  {}\n\nEvery \
         entry on that list is there because it CANNOT — that is the entire criterion. A function \
         that takes a destination path, or that hands back the body, is a download primitive however \
         it is described, so move it to DOWNLOAD_PRIMITIVES where the reachability scan covers its \
         callers.",
        smuggled.join("\n  ")
    );

    // The two signature shapes the machine check has to see through, pinned on synthetic input so
    // the check itself cannot rot into a default-value assertion.
    let synthetic = SourceFile {
        rel: "downloads.rs".to_owned(),
        code:
            "pub(crate) async fn zz_generic<P>(c: &C, u: &str, dest: P) -> R where P: AsRef<Path> \
               { c.get(u).send().await }\n\
               pub(crate) async fn zz_bytes(c: &C, u: &str) -> WorkerResult<Vec<u8>> { \
               c.get(u).send().await }\n\
               pub(crate) async fn zz_clean(c: &C, u: &str) -> WorkerResult<u64> { \
               c.get(u).send().await }"
                .to_owned(),
    };
    let synthetic_items = fn_items(&synthetic.code);
    let shape = |name: &str| {
        let item = synthetic_items
            .iter()
            .find(|item| item.name == name)
            .expect("fixture fn");
        let signature = format!("{}{}", item.params_raw, item.ret).replace(char::is_whitespace, "");
        let returned = item.ret.replace(char::is_whitespace, "");
        (
            signature.contains("Path"),
            ["Vec<u8>", "Bytes", "[u8]"]
                .iter()
                .any(|s| returned.contains(s)),
        )
    };
    assert_eq!(
        shape("zz_generic"),
        (true, false),
        "a `where P: AsRef<Path>` bound lives in the return/where text, not the parameter list"
    );
    assert_eq!(
        shape("zz_bytes"),
        (false, true),
        "returning the body is the same capability as writing it"
    );
    assert_eq!(shape("zz_clean"), (false, false));

    let unregistered: Vec<&String> = discovered.difference(&registered).collect();
    let stale: Vec<&String> = registered.difference(&discovered).collect();
    assert!(
        unregistered.is_empty(),
        "`downloads.rs` grew externally-visible function(s) that reach the network and are on \
         neither list:\n  {unregistered:?}\n\nEvery one of these is callable, unqualified, from any \
         file in the crate (`lib.rs` does `use downloads::*;`). Put it in DOWNLOAD_PRIMITIVES so the \
         reachability scan covers it, or — if it genuinely cannot pull model bytes into a job — in \
         NON_WEIGHT_FETCHERS with a comment saying what it does instead."
    );
    assert!(
        stale.is_empty(),
        "name(s) registered as part of the `downloads.rs` fetch surface that no longer reach the \
         network there:\n  {stale:?}\n\nRemove the entry in the same commit, so the list keeps \
         stating the real surface."
    );
}

/// **The next regrowth shape.** A lane that writes `reqwest::Client::new().get(url).send()` inline
/// fetches weights while naming nothing on any allow-list, so the ability to make an HTTP request at
/// all is confined to [`HTTP_CLIENT_FILES`].
#[test]
fn http_client_construction_is_confined() {
    let files = worker_sources();
    let mut found: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for file in &files {
        let reasons = http_client_reach(&file.code_without_literals());
        if !reasons.is_empty() {
            found.insert(file.rel.clone(), reasons);
        }
    }

    let allowed: BTreeMap<&str, &str> = HTTP_CLIENT_FILES.iter().copied().collect();
    let added: Vec<String> = found
        .iter()
        .filter(|(rel, _)| !allowed.contains_key(rel.as_str()))
        .map(|(rel, reasons)| format!("{rel}  ({})", reasons.join(", ")))
        .collect();
    let stale: Vec<&str> = allowed
        .keys()
        .filter(|rel| !found.contains_key(**rel))
        .copied()
        .collect();

    assert!(
        added.is_empty(),
        "file(s) outside the confinement now build an HTTP client or issue a request:\n  {}\n\nA job \
         lane that opens its own socket bypasses every download allow-list in this module — it is \
         the same regression as a job-time download, spelled differently. Resolve declared weights \
         with `crate::downloads::resolve_hf_component_file` instead. If a file genuinely needs an \
         HTTP client for something that is not weights, add it to HTTP_CLIENT_FILES with a comment \
         saying what it talks to.",
        added.join("\n  ")
    );
    assert!(
        stale.is_empty(),
        "file(s) allow-listed for HTTP-client construction that no longer build one:\n  {}\n\nRatchet \
         the list down in the same commit.",
        stale.join("\n  ")
    );
}

/// A function that hands out the bare `<data_dir>/cache` root must be registered, or a new one is
/// silently absorbed into an existing file's `<cache-root>` entry and becomes a way to reach a new
/// destination without naming it.
#[test]
fn cache_root_producers_are_registered() {
    let crates = all_production_sources();
    let constants = all_string_constants(&crates);

    let mut discovered: BTreeSet<(String, String, String)> = BTreeSet::new();
    for (label, files) in &crates {
        for (rel, name) in discover_cache_root_producers(files, &constants) {
            discovered.insert(((*label).to_owned(), rel, name));
        }
    }
    let registered: BTreeSet<(String, String, String)> = CACHE_ROOT_PRODUCERS
        .iter()
        .map(|(label, rel, name)| ((*label).to_owned(), (*rel).to_owned(), (*name).to_owned()))
        .collect();
    assert_eq!(
        discovered, registered,
        "the set of functions that hand out a bare `<data_dir>/cache` root changed. A new one is a \
         hole: CACHE_DESTINATIONS keys on (crate, file, subdir), so a second producer in an \
         already-listed file changes nothing there, and its callers can then join a brand-new weight \
         destination onto the root without spelling `cache` at all. Register it as \
         (crate label, file, fn name), which also makes its call sites contribute what they join."
    );
}

/// A helper that takes its cache subdirectory as a **parameter** is the documented way a
/// literal-keyed allow-list gets bypassed, so the registry of those helpers is itself checked.
#[test]
fn parameterized_cache_root_helpers_are_registered() {
    let crates = all_production_sources();
    let constants = all_string_constants(&crates);
    let mut discovered: BTreeMap<String, usize> = BTreeMap::new();
    for (_, files) in &crates {
        discovered.extend(discover_parameterized_helpers(files, &constants));
    }
    let registered: BTreeMap<String, usize> = PARAMETERIZED_CACHE_HELPERS
        .iter()
        .map(|(name, index)| ((*name).to_owned(), *index))
        .collect();
    assert_eq!(
        discovered, registered,
        "the set of functions that build `<data_dir>/cache/<param>` changed. A helper missing from \
         PARAMETERIZED_CACHE_HELPERS is a hole: its callers name the real destination and no scan of \
         string literals can see it. Register it as (name, zero-based index of the subdir parameter), \
         which makes its call sites contribute their literal to CACHE_DESTINATIONS."
    );
    // `paths.rs`'s two confinement helpers are the reason this test exists; losing either means the
    // registry drifted rather than the code improving.
    assert!(
        registered.contains_key("normalize_app_managed_cache_path")
            && registered.contains_key("resolve_import_source_path"),
        "the two `paths.rs` confinement helpers must stay registered"
    );
}

/// This module must never scan itself: its fixtures contain both `ensure_hf_cached_file(` and
/// `.join("cache")` on purpose, and if the module-tree walk ever classified it as production the
/// gates would be permanently, unfixably red.
#[test]
fn the_guard_module_is_excluded_from_its_own_scan() {
    let (_, src, root, _) = SCANNED_CRATES[0];
    let tree = read_tree(&repo_root().join(src));
    let test_only = test_only_files(&tree, root);
    assert!(
        test_only.contains("job_time_download_guard.rs"),
        "the guard module classified itself as production source"
    );
    assert!(
        test_only.contains("tests.rs") && test_only.contains("tests/lora_and_downloads.rs"),
        "the module walk must follow `include!` edges into src/tests/, or the crate's own download \
         tests read as production job-time downloads"
    );
    assert!(
        !test_only.contains("downloads.rs") && !test_only.contains("model_jobs.rs"),
        "production modules must NOT be classified test-only — that would blind both gates"
    );
}

// ---------------------------------------------------------------------------------------------
// Scanner self-tests — the mutation check
// ---------------------------------------------------------------------------------------------

/// A fake NEW job-time call site must be detected, in every spelling the tree actually uses; a doc
/// comment, a `use` item and a string literal that merely NAME a helper must not be.
///
/// Without this, `job_time_download_sites_match_the_allow_list` is a test asserting a default value:
/// it would pass just as happily with the detection ripped out.
#[test]
fn scanner_detects_a_new_direct_call_site() {
    let watch: BTreeMap<String, String> = DOWNLOAD_PRIMITIVES
        .iter()
        .map(|s| ((*s).to_owned(), "downloads.rs".to_owned()))
        .collect();
    let detect = |source: &str| {
        download_reach(
            &code_without_comments_or_literals(&code_without_comments(source)),
            &watch,
        )
    };

    // Unqualified — the crate-root glob (`use downloads::*;`) puts the helpers in scope everywhere.
    assert_eq!(
        detect("async fn new_lane(c: &DownloadContext<'_>) { ensure_hf_cached_file(c, r, v, f, d).await; }"),
        vec!["ensure_hf_cached_file <- downloads.rs".to_owned()]
    );
    // Fully qualified — four lanes already spell it this way.
    assert!(!detect("fn f() { crate::downloads::ensure_cached_file(c, u, t, l, s); }").is_empty());
    // The snapshot primitive, which a gate limited to the two named helpers would miss.
    assert!(!detect("fn f() { download_snapshot_into_cache(&c, &d, r, &s, &mut p); }").is_empty());
    // Line-broken call, as rustfmt writes the real ones.
    assert!(
        !detect("fn f() {\n    ensure_hf_cached_file(\n        &context,\n    );\n}").is_empty()
    );

    // NEGATIVES. ~25 doc comments in this crate name `ensure_hf_cached_file` in files that never
    // call it; `person_jobs.rs`, `person_segment.rs` and `depth.rs` are three of them.
    assert!(
        detect("/// `lfs.oid`, which `ensure_hf_cached_file()` verifies.\nfn f() {}").is_empty()
    );
    assert!(detect("// ensure_cached_file(a, b) used to live here\nfn f() {}").is_empty());
    assert!(detect("/* ensure_hf_cached_file(x) */ fn f() {}").is_empty());
    // A `use` item brings the name into scope but does not call it.
    assert!(
        detect("use super::{advanced, ensure_hf_cached_file, huggingface_snapshot_dir};")
            .is_empty()
    );
    assert!(detect("use crate::downloads::{ensure_cached_file, DownloadContext};").is_empty());
    // A string literal naming the helper is data, not a call.
    assert!(detect(r#"fn f() { log("ensure_hf_cached_file(x) failed"); }"#).is_empty());
    // The definition itself is not a call site.
    assert!(
        detect("pub(crate) async fn ensure_hf_cached_file(c: &C) -> R { unimplemented!() }")
            .is_empty()
    );
    // A longer identifier that merely contains the name.
    assert!(detect("fn f() { my_ensure_hf_cached_file_shim(a); }").is_empty());
}

/// A fake NEW local wrapper must be resolved, including across files — the `instantid.rs` /
/// `pulid.rs` shape, which defeats any gate that only greps the primitive names.
#[test]
fn scanner_detects_a_new_wrapper_and_its_remote_caller() {
    let definer = SourceFile {
        rel: "image_jobs/fake_definer.rs".to_owned(),
        code: "async fn stage_weight(c: &DownloadContext<'_>, f: &str) -> R { \
               ensure_hf_cached_file(c, REPO, REV, f, &dst).await }"
            .to_owned(),
    };
    // The caller contains NO download-helper text at all.
    let caller = SourceFile {
        rel: "image_jobs/fake_caller.rs".to_owned(),
        code: "async fn run(c: &Ctx) { stage_weight(c, \"model.safetensors\").await; }".to_owned(),
    };
    let files = vec![definer.clone(), caller.clone()];

    let wrappers = download_wrappers(&files);
    assert!(
        wrappers.contains_key("stage_weight"),
        "a function whose body calls a primitive is a wrapper: {wrappers:?}"
    );
    assert!(
        !wrappers.contains_key("run"),
        "wrapper resolution must stop after one hop; chasing the call graph flags the whole crate"
    );

    let mut watch: BTreeMap<String, String> = DOWNLOAD_PRIMITIVES
        .iter()
        .map(|s| ((*s).to_owned(), "downloads.rs".to_owned()))
        .collect();
    watch.extend(wrappers);
    assert!(
        !download_reach(&caller.code_without_literals(), &watch).is_empty(),
        "the remote caller must be flagged even though it names no primitive"
    );

    // A `DownloadContext` literal is its own ticket to the network.
    assert!(constructs_download_context(
        "let c = DownloadContext { api, client };"
    ));
    assert!(constructs_download_context(
        "f(&crate::downloads::DownloadContext { api, client })"
    ));
    // A parameter of that type, or an import, is not a construction.
    assert!(!constructs_download_context(
        "fn f(context: &DownloadContext<'_>) {}"
    ));
    assert!(!constructs_download_context(
        "use crate::downloads::DownloadContext;"
    ));
    assert!(!constructs_download_context(
        "struct MyDownloadContext { a: u8 }"
    ));
}

/// A fake NEW `<data_dir>/cache` weight destination must be detected, in each spelling the tree uses;
/// and the parameterized-helper bypass must be covered from the call site.
#[test]
fn scanner_detects_a_new_cache_weight_destination() {
    let mut constants = BTreeMap::new();
    constants.insert("NEW_WEIGHTS_DIR".to_owned(), "brand-new-weights".to_owned());

    let chained = "let dst = settings.data_dir.join(\"cache\").join(\"brand-new-weights\");";
    assert!(cache_destinations(chained, &constants).contains("brand-new-weights"));

    // rustfmt splits the real ones across lines.
    let split = "let dst = settings\n    .data_dir\n    .join(\"cache\")\n    .join(\"brand-new-weights\")\n    .join(&file);";
    assert!(cache_destinations(split, &constants).contains("brand-new-weights"));

    // Combined literal — `CONVERSION_LOCKS_DIR` and the three legacy person/pose roots use this.
    let combined = "for sub in [\"cache/brand-new-weights\", \"models/brand-new-weights\"] {}";
    assert!(cache_destinations(combined, &constants).contains("brand-new-weights"));

    // A `const` resolves to its value, so renaming the constant does not churn the allow-list.
    let via_const = "let dst = data_dir.join(\"cache\").join(NEW_WEIGHTS_DIR);";
    assert!(cache_destinations(via_const, &constants).contains("brand-new-weights"));

    // An unresolvable expression is loud, not absent.
    let dynamic = "let dst = data_dir.join(\"cache\").join(pick_dir(model));";
    assert!(cache_destinations(dynamic, &constants).contains("{expr}"));

    // NEGATIVES: neither a doc comment nor an unrelated join is a destination.
    assert!(
        cache_destinations(
            &code_without_comments("/// writes into `<data>/cache/brand-new-weights/`\nfn f() {}"),
            &constants
        )
        .is_empty(),
        "a doc comment naming a cache path must not register a destination"
    );
    assert!(
        cache_destinations("let p = root.join(\"models\").join(\"x\");", &constants).is_empty()
    );

    // The parameterized bypass: a caller that contains NO `"cache"` text still names the destination.
    let caller =
        "let p = normalize_app_managed_cache_path(settings, raw, \"brand-new-weights\", \"src\")?;";
    assert!(
        cache_destinations(caller, &constants).is_empty(),
        "the literal scan is genuinely blind here — that blindness is what the registry repairs"
    );
    assert!(parameterized_call_destinations(caller, &constants).contains("brand-new-weights"));

    // ...and the registry's discovery half sees a NEW parameterized helper appear.
    let helper = SourceFile {
        rel: "fake_paths.rs".to_owned(),
        code: "fn confine(settings: &Settings, raw: &str, cache_dir: &str, label: &str) -> R { \
               let root = settings.data_dir.join(\"cache\").join(cache_dir); ok(root) }"
            .to_owned(),
    };
    assert_eq!(
        discover_parameterized_helpers(&[helper], &constants).get("confine"),
        Some(&2usize),
        "a new helper that joins `cache` with its own parameter must be discovered, at the right \
         parameter index"
    );
}

/// The three cheapest ways to spell a `<data_dir>/cache` root that a literal scan cannot see, and the
/// registry that catches the fourth.
///
/// Every one of these was a live escape before sc-17637's review: a `const`, a raw literal, and an
/// accessor handing out the bare root from the very file that already owns the only `<cache-root>`
/// allow-list entry.
#[test]
fn scanner_sees_through_an_aliased_cache_root() {
    let mut constants = BTreeMap::new();
    constants.insert("M4_ROOT".to_owned(), "cache".to_owned());
    constants.insert("UNRELATED".to_owned(), "models".to_owned());

    // A `const` that resolves to `cache`.
    assert!(cache_destinations(
        "let d = data_dir.join(M4_ROOT).join(\"brand-new-weights\");",
        &constants
    )
    .contains("brand-new-weights"));
    // Raw literals, both hash flavours.
    assert!(cache_destinations(
        "let d = data_dir.join(r\"cache\").join(\"brand-new-weights\");",
        &constants
    )
    .contains("brand-new-weights"));
    assert_eq!(
        string_literal_value("r#\"cache\"#").as_deref(),
        Some("cache")
    );
    assert_eq!(string_literal_value("\"cache\"").as_deref(), Some("cache"));
    assert_eq!(string_literal_value("CACHE_DIR"), None);
    // A constant that resolves to something else is NOT a cache root.
    assert!(cache_destinations(
        "let d = data_dir.join(UNRELATED).join(\"brand-new-weights\");",
        &constants
    )
    .is_empty());

    // A function that ends a path expression at `cache` is a root producer, and one that keeps going
    // is not — that one is an ordinary destination.
    let producer = SourceFile {
        rel: "fake_app_paths.rs".to_owned(),
        code: "pub fn m8_cache_root(d: &Path) -> PathBuf { d.join(\"cache\") }\n\
               pub fn destination(d: &Path) -> PathBuf { d.join(\"cache\").join(\"uploads\") }"
            .to_owned(),
    };
    let discovered = discover_cache_root_producers(&[producer], &constants);
    assert!(discovered.contains(&("fake_app_paths.rs".to_owned(), "m8_cache_root".to_owned())));
    assert!(!discovered.contains(&("fake_app_paths.rs".to_owned(), "destination".to_owned())));

    // ...and a lane that joins onto a registered producer's result is attributed the destination,
    // even though it contains no `cache` text of its own.
    let producers: BTreeSet<&str> = ["m8_cache_root"].into_iter().collect();
    let caller = "let d = app_paths::m8_cache_root(data_dir).join(\"brand-new-weights\");";
    assert!(cache_destinations(caller, &constants).is_empty());
    assert!(
        cache_root_producer_destinations(caller, &producers, &constants)
            .contains("brand-new-weights")
    );
}

/// A fake NEW fetcher added to `downloads.rs` must be rediscovered, and a private one must not
/// (a job lane cannot name it).
#[test]
fn scanner_rediscovers_the_network_fetch_surface() {
    let module = SourceFile {
        rel: "downloads.rs".to_owned(),
        code: "pub(crate) async fn m17_fetch(client: &reqwest::Client, url: &str) -> R { \
               client.get(url).send().await }\n\
               pub(crate) async fn m17_indirect(c: &C, u: &str) -> R { m17_fetch(c, u).await }\n\
               async fn m17_private(c: &C, u: &str) -> R { m17_fetch(c, u).await }\n\
               pub(crate) fn m17_offline(path: &Path) -> R { std::fs::read(path) }"
            .to_owned(),
    };
    let discovered = network_fetch_surface(&module);
    assert!(
        discovered.contains("m17_fetch"),
        "a new fn that sends a request must be discovered: {discovered:?}"
    );
    assert!(
        discovered.contains("m17_indirect"),
        "reachability is transitive inside the module: {discovered:?}"
    );
    assert!(
        !discovered.contains("m17_private"),
        "a private fn cannot be named by a job lane, so it is not part of the surface"
    );
    assert!(
        !discovered.contains("m17_offline"),
        "a fn that touches no network is not part of the fetch surface"
    );

    // Visibility parsing is what draws that line, so pin it directly.
    let items = fn_items(
        "pub(crate) async fn a() {}\n pub fn b() {}\n async fn c() {}\n pub(super) unsafe fn d() {}",
    );
    let public: Vec<(&str, bool)> = items
        .iter()
        .map(|item| (item.name.as_str(), item.is_public))
        .collect();
    assert_eq!(
        public,
        vec![("a", true), ("b", true), ("c", false), ("d", true)]
    );
}

/// The M18 shape: a lane that opens its own socket, naming nothing on any download allow-list.
#[test]
fn scanner_detects_an_inline_http_client() {
    let detect = |source: &str| {
        http_client_reach(&code_without_comments_or_literals(&code_without_comments(
            source,
        )))
    };

    assert!(
        !detect("async fn f(u: &str) { reqwest::Client::new().get(u).send().await; }").is_empty()
    );
    assert!(!detect("fn f() { let c = reqwest::Client::builder().build(); }").is_empty());
    assert!(
        !detect("async fn f(c: &reqwest::Client, u: &str) { c.get(u).send().await; }").is_empty()
    );
    // Holding a client is not the same as building one or sending with it: every download primitive
    // takes a `&reqwest::Client`, so a parameter of that type must stay quiet.
    assert!(detect(
        "async fn f(client: &reqwest::Client) -> R { ensure_cached_file(client).await }"
    )
    .is_empty());
    // Comments and string literals naming the constructor are not constructions.
    assert!(detect("/// never call reqwest::Client::new() here\nfn f() {}").is_empty());
    assert!(detect(r#"fn f() { log("reqwest::Client::new() is banned"); }"#).is_empty());
}

/// The M2 shape: a forwarder inside an allow-listed file, called from a file that is not.
#[test]
fn scanner_resolves_wrappers_past_one_hop() {
    // `image_jobs/instantid.rs` is a `JobTime` entry, so the closure grows through it.
    let listed = DOWNLOAD_SITES
        .iter()
        .any(|(rel, role)| *rel == "image_jobs/instantid.rs" && *role == DownloadRole::JobTime);
    assert!(
        listed,
        "this test's premise is that instantid.rs is a JobTime site"
    );

    let definer = SourceFile {
        rel: "image_jobs/instantid.rs".to_owned(),
        code: "async fn stage(c: &DownloadContext<'_>) -> R { ensure_hf_cached_file(c).await }\n\
               async fn m2_forward(c: &DownloadContext<'_>) -> R { stage(c).await }"
            .to_owned(),
    };
    // NOT on DOWNLOAD_SITES, and it names no primitive, no context and no first-hop wrapper.
    let new_lane = SourceFile {
        rel: "image_jobs/m2_new_lane.rs".to_owned(),
        code: "async fn run(c: &Ctx) { m2_forward(c).await; }".to_owned(),
    };

    let wrappers = download_wrappers(&[definer, new_lane.clone()]);
    assert!(wrappers.contains_key("stage"), "hop one: {wrappers:?}");
    assert!(
        wrappers.contains_key("m2_forward"),
        "hop two, through a file already on DOWNLOAD_SITES: {wrappers:?}"
    );

    let mut watch: BTreeMap<String, String> = DOWNLOAD_PRIMITIVES
        .iter()
        .map(|s| ((*s).to_owned(), "downloads.rs".to_owned()))
        .collect();
    watch.extend(wrappers);
    assert!(
        !download_reach(&new_lane.code_without_literals(), &watch).is_empty(),
        "the two-hop caller must be flagged"
    );

    // The closure must NOT grow through a file that is not on the list: that file is already red on
    // its own, and following it is what turns the gate into a 73-file census.
    let unlisted = SourceFile {
        rel: "brand_new_file.rs".to_owned(),
        code: "async fn only_hop_one(c: &C) -> R { ensure_hf_cached_file(c).await }\n\
               async fn hop_two(c: &C) -> R { only_hop_one(c).await }"
            .to_owned(),
    };
    let wrappers = download_wrappers(&[unlisted]);
    assert!(wrappers.contains_key("only_hop_one"));
    assert!(
        !wrappers.contains_key("hop_two"),
        "the closure must stop at files that are not JobTime sites: {wrappers:?}"
    );
}

/// A `'"'` char literal and a raw string carrying a `"` both flip string parity for the REST OF THE
/// FILE if the stripper does not step over them — the tail of the file is absorbed into a phantom
/// literal and silently stops being scanned.
///
/// `downloads.rs` contains `.trim_matches('"')` at roughly its midpoint, which is why this is a
/// regression test and not a hypothetical: everything after it, including
/// `lora_source_content_length` and `report_download_progress`, was invisible to both gates.
#[test]
fn scanner_survives_quote_char_literals_and_raw_strings() {
    let source = "fn quoting() { let q = '\"'; }\n\
                  fn hidden(c: &DownloadContext<'_>) { ensure_hf_cached_file(c); }\n";
    let code = code_without_comments(source);
    assert!(
        code.contains("ensure_hf_cached_file"),
        "a `'\"'` char literal must not swallow the rest of the file: {code}"
    );
    assert!(fn_items(&code)
        .iter()
        .any(|item| item.name == "hidden" && item.body.contains("ensure_hf_cached_file")));

    let raw = "fn quoting() { let q = r#\"a \"quoted\" fox\"#; }\n\
               fn hidden() { let d = data_dir.join(\"cache\").join(\"brand-new-weights\"); }\n";
    let code = code_without_comments(raw);
    assert!(
        cache_destinations(&code, &BTreeMap::new()).contains("brand-new-weights"),
        "a raw string carrying a quote must not swallow the rest of the file: {code}"
    );

    // And the same for the blanking pass, which the download scan reads.
    let blanked = code_without_comments_or_literals(source);
    assert!(blanked.contains("ensure_hf_cached_file"));

    // A `#[cfg(test)]` on a struct FIELD ends at its comma, not at the next item's brace. Getting
    // this wrong deleted whole functions from the scan.
    let stripped = strip_cfg_test_items(
        "struct S {\n    real: u8,\n    #[cfg(test)]\n    probe: u64,\n}\n\
         fn kept(c: &C) { ensure_hf_cached_file(c); }\n",
    );
    assert!(
        !stripped.contains("probe"),
        "the cfg'd field must go: {stripped}"
    );
    assert!(
        stripped.contains("fn kept") && stripped.contains("ensure_hf_cached_file"),
        "the function after it must survive: {stripped}"
    );
    // A `#[cfg(test)]` statement still ends at its semicolon.
    let stripped = strip_cfg_test_items(
        "fn f() {\n    #[cfg(test)]\n    COUNTER.set(1);\n    ensure_hf_cached_file(c);\n}",
    );
    assert!(!stripped.contains("COUNTER"));
    assert!(stripped.contains("ensure_hf_cached_file"));
}

/// **MAJOR 4.** [`SCANNED_CRATES`] must cover `[workspace] members`, or a brand-new crate can hold a
/// brand-new weight destination and never be looked at — round one's M10 in a new hat.
#[test]
fn scanned_crates_cover_the_workspace() {
    let manifest =
        std::fs::read_to_string(repo_root().join("Cargo.toml")).expect("root Cargo.toml");
    let members_block = manifest
        .split_once("\nmembers = [")
        .expect("the root manifest must declare [workspace] members")
        .1
        .split_once(']')
        .expect("unterminated members list")
        .0;
    let members: BTreeSet<String> = members_block
        .split(',')
        .filter_map(|entry| string_literal_value(entry.trim()))
        .collect();
    assert!(
        members.len() >= 5,
        "parsed only {} workspace members — the parser is broken, not the manifest",
        members.len()
    );

    let scanned: BTreeSet<String> = SCANNED_CRATES
        .iter()
        .map(|(_, src, _, _)| {
            src.strip_suffix("/src")
                .unwrap_or_else(|| panic!("{src} must be a crate `src` directory"))
                .to_owned()
        })
        .collect();
    let excluded: BTreeSet<String> = UNSCANNED_WORKSPACE_MEMBERS
        .iter()
        .map(|(path, _)| (*path).to_owned())
        .collect();

    let unscanned: Vec<&String> = members
        .iter()
        .filter(|member| !scanned.contains(*member) && !excluded.contains(*member))
        .collect();
    let phantom: Vec<&String> = scanned.difference(&members).collect();
    assert!(
        unscanned.is_empty(),
        "workspace member(s) that no gate in this module looks at:\n  {unscanned:?}\n\nA crate that \
         can see the app data dir can hold a new `<data_dir>/cache` weight destination. Add it to \
         SCANNED_CRATES as (label, `<path>/src`, crate root file, minimum file count), or — if it \
         structurally cannot reach the data dir — to UNSCANNED_WORKSPACE_MEMBERS with the reason."
    );
    assert!(
        phantom.is_empty(),
        "SCANNED_CRATES names path(s) that are no longer workspace members:\n  {phantom:?}"
    );
}

/// **MAJOR 3.** Confining WHERE a client may be built says nothing about what those files fetch, so
/// each [`HTTP_CLIENT_FILES`] member (other than `downloads.rs`, which
/// [`downloads_fetch_surface_is_registered`] covers) must have its network-reaching surface
/// registered in [`HTTP_CLIENT_FILE_SURFACE`].
#[test]
fn http_client_file_fetch_surfaces_are_registered() {
    let files = worker_sources();
    for (rel, _) in HTTP_CLIENT_FILES {
        if *rel == "downloads.rs" {
            continue;
        }
        let discovered = network_fetch_surface(&worker_file(&files, rel));
        assert!(
            !discovered.is_empty(),
            "{rel} is allow-listed for HTTP-client construction but the surface walk found nothing \
             in it, so this assertion would prove nothing"
        );
        let registered: BTreeSet<String> = HTTP_CLIENT_FILE_SURFACE
            .iter()
            .filter(|(file, _, _)| file == rel)
            .map(|(_, name, _)| (*name).to_owned())
            .collect();
        assert_eq!(
            discovered, registered,
            "the network-reaching surface of {rel} changed. This file is allow-listed to build an \
             HTTP client, so a new externally-visible function in it can fetch remote bytes and \
             write them wherever the caller asks, and no other assertion in this module would see \
             it. Register it in HTTP_CLIENT_FILE_SURFACE with what it does — and if what it does is \
             fetch weights, it belongs in `downloads.rs` behind a download primitive instead."
        );
    }
}

/// What the literal walkers made of one file: `(brace balance, paren balance, lowest brace depth,
/// number of string literals containing a newline)`.
fn literal_partition_shape(code: &str) -> (i64, i64, i64, usize) {
    let bytes = code.as_bytes();
    let (mut brace, mut paren, mut floor, mut multiline) = (0i64, 0i64, 0i64, 0usize);
    let mut index = 0usize;
    let note = |slice: &[u8], multiline: &mut usize| {
        if slice.contains(&b'\n') {
            *multiline += 1;
        }
    };
    while index < bytes.len() {
        if bytes[index] == b'\'' {
            index = skip_char(bytes, index);
            continue;
        }
        if (bytes[index] == b'r' || bytes[index] == b'b')
            && index
                .checked_sub(1)
                .map(|i| !(bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric()))
                .unwrap_or(true)
        {
            if let Some((_, end)) = crate::architecture_tests::raw_string_parts(bytes, index) {
                let end = end.min(bytes.len());
                note(&bytes[index..end], &mut multiline);
                index = end;
                continue;
            }
        }
        match bytes[index] {
            b'"' => {
                let end = skip_string(bytes, index).min(bytes.len());
                note(&bytes[index..end], &mut multiline);
                index = end;
                continue;
            }
            b'{' => brace += 1,
            b'}' => brace -= 1,
            b'(' => paren += 1,
            b')' => paren -= 1,
            _ => {}
        }
        floor = floor.min(brace);
        index += 1;
    }
    (brace, paren, floor, multiline)
}

/// **The self-check.** The literal walkers must partition every scanned file into code and literals
/// the way `rustc` does. When they do not, the failure is silent: a mis-paired quote absorbs real
/// code into a phantom literal and removes it from BOTH gates without any assertion firing.
///
/// # Why these two measures, and not the obvious ones
///
/// This class has shipped twice, and the first version of this test was itself a false green — it
/// gated on the LONGEST literal, and would have passed the very defect it was written for. The
/// numbers below are measured across all 240 scanned files, once with the walkers correct and once
/// with round two's broken `char_literal_end` restored.
///
/// * **Longest literal — rejected.** A parity inversion re-syncs at every subsequent `"`, so it
///   makes MANY SHORT phantom literals, never one long one. The longest phantom in `media_jobs.rs`
///   (the file that defect blinded across 1143 lines) was 55 lines, against a real maximum of 52
///   (`jobs_store.rs`'s SQL schema). No ceiling fits in a 3-line gap.
/// * **Literal-byte share — rejected.** The real distribution runs to 42.2% (`worker/preflight.rs`),
///   38.2% (`core/control_weights.rs`), 34.2% (`core/jobs_store/routing/gaps.rs`), while broken
///   `media_jobs.rs` sits near 40%. The populations overlap; no threshold separates them.
/// * **Delimiter balance — exact, no threshold.** Valid Rust balances its braces and parens; code
///   absorbed into a phantom literal takes its delimiters with it. All 240 files balance exactly
///   with the walkers correct, and `core/session_log.rs` goes to `braces=-1` with them broken.
/// * **COUNT of multi-line literals — the same "many short" fact, used as the detector instead of
///   treated as an obstacle.** Real multi-line literals are few and long. Measured by this test's own
///   walk over all 328 scanned files: 128 contain one at all, only 9 exceed 30, and only 3 exceed
///   50 — 60 (`core/catalog_store.rs`), 54 (`core/jobs_store.rs`), 52 (`worker/video_jobs/tests.rs`),
///   then 46, 39, 38, 37, 32, 31, 30. Broken `media_jobs.rs` produces **207**, three and a half times
///   the highest legitimate count in the tree.
///
///   THIS module is currently fifth-highest at 39 and has grown every round, its fixtures being
///   multi-line source snippets. The failure message reports the real worst offender rather than
///   trusting the number written here, so the ceiling can be re-judged from output instead of prose.
///
/// Measured deliberately BEFORE [`strip_cfg_test_items`], over every `.rs` file rather than only the
/// production ones: `rustc` guarantees the whole file's shape, so keeping cfg-stripping out of the
/// comparison leaves this a statement about the literal walkers alone.
#[test]
fn the_literal_walkers_partition_every_scanned_file_correctly() {
    // Legitimate maximum is 60; the observed failure is 207. Sitting between them leaves 1.67x of
    // room for a genuinely fixture-heavy new file — this module itself is at 39 and still growing —
    // without leaving room for an inversion.
    const MAX_MULTILINE_LITERALS: usize = 100;
    let mut broken: Vec<String> = Vec::new();
    let mut scanned = 0usize;
    let mut worst = (0usize, String::new());

    for (label, src, _, _) in SCANNED_CRATES {
        for (rel, code) in read_tree(&repo_root().join(src)) {
            scanned += 1;
            let (brace, paren, floor, multiline) = literal_partition_shape(&code);
            if brace != 0 || paren != 0 || floor < 0 {
                broken.push(format!(
                    "{label}/{rel}  braces={brace:+} parens={paren:+} lowest-depth={floor}"
                ));
            }
            if multiline > worst.0 {
                worst = (multiline, format!("{label}/{rel}"));
            }
            if multiline > MAX_MULTILINE_LITERALS {
                broken.push(format!(
                    "{label}/{rel}  {multiline} string literals span a newline"
                ));
            }
        }
    }

    assert!(
        scanned > 200,
        "walked only {scanned} files — this assertion would prove nothing"
    );
    assert!(
        broken.is_empty(),
        "the literal walkers mis-partitioned {} file(s):\n  {}\n\nValid Rust balances its delimiters \
         exactly, and real multi-line string literals are few. A file failing either check has code \
         absorbed into phantom string literals, which means that code has SILENTLY STOPPED being \
         scanned by both gates and no other assertion will fire. Fix `char_literal_end` / \
         `raw_string_parts` / `skip_string` in `architecture_tests`, not the file named here. \
         (Most multi-line literals seen: {} in {}.)",
        broken.len(),
        broken.join("\n  "),
        worst.0,
        worst.1
    );
}

/// **MINOR 5.** An `as` rename and a bare fn-pointer binding are one-line bypasses of a name gate.
#[test]
fn scanner_sees_through_aliases_and_value_bindings() {
    let aliased = SourceFile {
        rel: "image_jobs/zz_alias_lane.rs".to_owned(),
        code: "use crate::downloads::ensure_hf_cached_file as fetch_it;\n\
               async fn run(c: &Ctx) { fetch_it(c, r, v, f, d).await; }"
            .to_owned(),
    };
    let watch = watch_map(std::slice::from_ref(&aliased));
    assert_eq!(
        watch.get("fetch_it").map(String::as_str),
        Some("`as` rename of ensure_hf_cached_file"),
        "an `as` rename must join the watch map: {watch:?}"
    );
    assert!(
        !download_reach(&aliased.code_without_literals(), &watch).is_empty(),
        "the aliased call must be flagged"
    );

    // A bare value binding matches no `NAME(` anywhere.
    let bound = SourceFile {
        rel: "image_jobs/zz_pointer_lane.rs".to_owned(),
        code:
            "async fn run(c: &Ctx) { let grab = ensure_hf_cached_file; grab(c, r, v, f, d).await; }"
                .to_owned(),
    };
    let watch = watch_map(std::slice::from_ref(&bound));
    let reasons = download_reach(&bound.code_without_literals(), &watch);
    assert!(
        reasons.iter().any(|reason| reason.contains("as a value")),
        "a fn-pointer binding must be flagged: {reasons:?}"
    );

    // NEGATIVE: an import mentions the name without using it. This is the distinction that makes the
    // value-reference rule safe rather than a blanket "the name appears" match.
    let import_only = SourceFile {
        rel: "image_jobs/zz_import_only.rs".to_owned(),
        code: "use crate::downloads::{ensure_cached_file, DownloadContext};\n\
               use super::{advanced, ensure_hf_cached_file, huggingface_snapshot_dir};\n\
               fn run() {}"
            .to_owned(),
    };
    let watch = watch_map(std::slice::from_ref(&import_only));
    let reasons = download_reach(&import_only.code_without_literals(), &watch);
    assert!(
        reasons.is_empty(),
        "a `use` item is not a reference: {reasons:?}"
    );
    assert_eq!(
        use_item_ranges("use a::b;\nfn f() { g(); }\nuse c::{d, e};").len(),
        2
    );
}

/// A short wrapper name must not turn the value-reference rule into a crate-wide word search.
///
/// `stage` is a real wrapper shape, and matching the bare word falsely flagged six unrelated
/// production files. Noisy rather than blind, so fail-safe — but a gate that cries wolf gets weakened
/// by whoever hits it next.
#[test]
fn scanner_does_not_cry_wolf_on_a_short_wrapper_name() {
    let definer = SourceFile {
        rel: "image_jobs/instantid.rs".to_owned(),
        code: "async fn stage(c: &DownloadContext<'_>) -> R { ensure_hf_cached_file(c).await }"
            .to_owned(),
    };
    // Prose that merely contains the word, in the shapes the real crate uses.
    let innocent = SourceFile {
        rel: "audio_jobs.rs".to_owned(),
        code: "fn run(stage: Stage) { match stage { Stage::One => report(stage) } }".to_owned(),
    };
    let watch = watch_map(&[definer, innocent.clone()]);
    assert!(
        watch.contains_key("stage"),
        "the wrapper itself must still be discovered: {watch:?}"
    );
    let reasons = download_reach(&innocent.code_without_literals(), &watch);
    assert!(
        reasons.is_empty(),
        "a short wrapper name must not match prose-shaped identifiers: {reasons:?}"
    );

    // ...while a genuine value binding of a PRIMITIVE is still caught whatever its length.
    let bound = SourceFile {
        rel: "zz_lane.rs".to_owned(),
        code: "fn run() { let grab = ensure_hf_cached_file; grab(a); }".to_owned(),
    };
    assert!(!download_reach(&bound.code_without_literals(), &watch).is_empty());
}

/// **MINOR 6.** Three more spellings of a cache destination, none of which occurs in production
/// source today and all of which would have been invisible.
#[test]
fn scanner_sees_the_remaining_cache_spellings() {
    let mut constants = BTreeMap::new();
    constants.insert("CONCAT_ROOT".to_owned(), "cache".to_owned());

    // A `/cache/<sub>` segment inside a format string rather than at the start of a literal.
    assert!(cache_destinations(
        "let p = format!(\"{}/cache/brand-new-weights\", data_dir.display());",
        &constants
    )
    .contains("brand-new-weights"));
    // ...and an interpolated subdir is loud rather than absent.
    assert!(cache_destinations(
        "let p = format!(\"{}/cache/{sub}\", d.display());",
        &constants
    )
    .contains("{expr}"));
    // `Path::new("cache")` is the same path segment as `"cache"`.
    assert!(cache_destinations(
        "let p = d.join(Path::new(\"cache\")).join(\"brand-new-weights\");",
        &constants
    )
    .contains("brand-new-weights"));
    assert!(cache_destinations(
        "let p = d.join(PathBuf::from(\"cache\")).join(\"brand-new-weights\");",
        &constants
    )
    .contains("brand-new-weights"));
    // A `concat!`-built const resolves to its value.
    assert_eq!(
        concat_literal_value("concat!(\"ca\", \"che\")").as_deref(),
        Some("cache")
    );
    let concat_const = SourceFile {
        rel: "zz.rs".to_owned(),
        code: "const CONCAT_ROOT: &str = concat!(\"ca\", \"che\");".to_owned(),
    };
    assert_eq!(
        string_constants(&[concat_const])
            .get("CONCAT_ROOT")
            .map(String::as_str),
        Some("cache")
    );
    assert!(cache_destinations(
        "let p = d.join(CONCAT_ROOT).join(\"brand-new-weights\");",
        &constants
    )
    .contains("brand-new-weights"));

    // A `::`-qualified wrapper and an owned-conversion suffix are the same segment. Both were
    // one-token bypasses of the bare-prefix match this used to do.
    assert!(cache_destinations(
        "let p = d.join(std::path::Path::new(\"cache\")).join(\"zz-qualified\");",
        &constants
    )
    .contains("zz-qualified"));
    assert!(cache_destinations(
        "let p = d.join(\"cache\".to_string()).join(\"zz-owned\");",
        &constants
    )
    .contains("zz-owned"));
    assert!(cache_destinations(
        "let p = d.join(\"cache\".to_owned()).join(\"zz-owned2\");",
        &constants
    )
    .contains("zz-owned2"));
    // Both wrappers at once — one pass of each leaves this unresolved, so the strips loop.
    assert!(cache_destinations(
        "let p = d.join(Path::new(\"cache\").to_owned()).join(\"zz-both\");",
        &constants
    )
    .contains("zz-both"));

    // NEGATIVE: an unrelated path that merely contains the word.
    assert!(
        cache_destinations("let p = d.join(\"models\").join(\"cachexyz\");", &constants).is_empty()
    );
}

/// The supporting parsers, exercised on the shapes they must judge. Each of these silently returning
/// nothing would make one of the gates above pass forever.
#[test]
fn scanner_parsers_recognise_the_shapes_they_must_judge() {
    // `#[cfg(test)]`-only items are dropped; `any(test, …)` and `not(test)` are NOT — those items are
    // compiled in production and dropping them would blind the scan to real code.
    assert!(cfg_is_test_only("test"));
    assert!(cfg_is_test_only("all(test, target_os = \"macos\")"));
    assert!(cfg_is_test_only(
        "all(test, not(target_os = \"macos\"), feature = \"backend-candle\")"
    ));
    assert!(!cfg_is_test_only("any(test, doc)"));
    assert!(!cfg_is_test_only("not(test)"));
    assert!(!cfg_is_test_only("all(target_os = \"macos\", not(test))"));
    assert!(!cfg_is_test_only("target_os = \"macos\""));

    let stripped = strip_cfg_test_items(
        "fn real() { ensure_hf_cached_file(a); }\n\
         #[cfg(test)]\nmod tests { fn t() { ensure_cached_file(b); } }\n\
         #[cfg(any(test, doc))]\nfn kept() { download_snapshot_into_cache(c); }\n",
    );
    assert!(stripped.contains("fn real"));
    assert!(stripped.contains("fn kept"));
    assert!(
        !stripped.contains("ensure_cached_file(b)"),
        "a #[cfg(test)] mod must be removed: {stripped}"
    );
    assert!(strip_cfg_test_items("#[cfg(test)]\nmod tests;\nfn keep() {}").contains("fn keep"));
    assert!(!strip_cfg_test_items("#[cfg(test)]\nmod tests;\nfn keep() {}").contains("mod tests"));

    // Parameter splitting must not be fooled by `::` inside a type, which would shift every index in
    // PARAMETERIZED_CACHE_HELPERS and silently point the scan at the wrong argument.
    assert_eq!(
        split_params("(settings: &Settings, target: &std::path::Path, cache_dir: &str)"),
        vec!["settings", "target", "cache_dir"]
    );
    assert_eq!(
        split_params("(a: HashMap<String, Vec<u8>>, b: impl Fn() -> bool)"),
        vec!["a", "b"]
    );

    // `'\''` is the escaped-quote case that shipped broken: the terminator is the FOURTH byte, not
    // the third. Getting it wrong orphans a quote, which then eats the opening `"` of the next
    // string literal and hides the rest of the file from both gates.
    assert_eq!(skip_char(b"'\\''", 0), 4);
    assert_eq!(skip_char(b"b'\\''", 1), 5);
    assert_eq!(skip_char(b"'\\n'", 0), 4);
    assert_eq!(skip_char(b"'\\\\'", 0), 4);
    assert_eq!(skip_char("'\\u{1F600}'".as_bytes(), 0), 11);
    assert_eq!(skip_char(b"'\"'", 0), 3);
    assert_eq!(skip_char(b"'{'", 0), 3);
    // A lifetime is not a literal, and a lifetime LIST must not read as one spanning both.
    assert_eq!(skip_char(b"'a>", 0), 1);
    assert_eq!(skip_char(b"'a, 'b>", 0), 1);
    assert_eq!(skip_char(b"'static", 0), 1);

    // A `'{'` char literal must not throw off the brace match — eight files in the scanned crates
    // contain one, and a body that ran to end-of-file would silently swallow every later item.
    let items = fn_items("fn a() { let c = '{'; } fn b() { ensure_hf_cached_file(x); }");
    assert_eq!(
        items.iter().map(|i| i.name.as_str()).collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert!(items[1].body.contains("ensure_hf_cached_file"));
    assert!(!items[0].body.contains("ensure_hf_cached_file"));

    // A brace inside a string literal likewise.
    let braced = fn_items("fn a() { format!(\"{}\", x); } fn b() { y(); }");
    assert_eq!(
        braced.iter().map(|i| i.name.as_str()).collect::<Vec<_>>(),
        vec!["a", "b"]
    );

    // Call-argument extraction: nested calls and string commas must not split an argument.
    let args = call_arguments(
        "let p = helper(settings, raw, \"a, b\", label(x, y))?;",
        "helper",
    );
    assert_eq!(args.len(), 1);
    assert_eq!(args[0].len(), 4);
    assert_eq!(args[0][2], "\"a, b\"");

    // Module-tree edges.
    assert_eq!(
        module_declarations("#[cfg(test)]\nmod tests;\npub mod real;\nmod inline { }"),
        ["real", "tests"]
            .into_iter()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>()
    );
    assert_eq!(
        include_paths("include!(\"tests/hf_and_family.rs\");"),
        vec!["tests/hf_and_family.rs".to_owned()]
    );
}
