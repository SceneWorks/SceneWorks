//! rust-api test suite, split by domain from the former single `tests.rs`
//! (sc-11217, F-030). Shared fixtures/helpers live in `support`.

mod auth;
mod browser_fixture;
mod catalog;
mod checkpoint_library;
mod compression;
mod dataset_catalogs;
#[cfg(feature = "embed-web")]
mod embedded_web;
// `pub(crate)` so the sc-22714 review suite can drive the same in-process API + fake worker
// instead of standing up a second one.
pub(crate) mod film_harness;
mod film_harness_anchoring;
mod film_harness_fixes;
mod film_harness_references;
mod film_harness_review;
mod films;
mod image_restoration;
mod jobs;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod load_progress_acceptance;
mod mcp;
mod media;
mod model_cache;
mod model_library;
mod moss_image;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod moss_training_acceptance;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod moss_training_acceptance_contracts;
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
mod moss_training_native;
mod projects;
mod prompt_batches;
mod recipe_presets;
mod server;
mod startup;
mod subject_masks;
mod supra2_jobs;
// `pub(crate)` so the inline `#[cfg(test)]` test modules that live OUTSIDE this `tests`
// tree (e.g. `crate::models::variant_install_tests`) can reuse the ONE crate-wide
// `isolate_hf_cache` / `HF_ENV_LOCK` guard rather than adding a second lock (sc-13835).
mod iris_curricula;
mod iris_task_training;
mod iris_training;
pub(crate) mod support;
mod training;
mod uploads;
mod workflows;
mod yue2_catalog;
mod yue2_jobs;
mod yue2_scores;
mod yue2_transcriptions;
