//! rust-api test suite, split by domain from the former single `tests.rs`
//! (sc-11217, F-030). Shared fixtures/helpers live in `support`.

mod auth;
mod catalog;
mod checkpoint_library;
mod compression;
mod dataset_catalogs;
#[cfg(feature = "embed-web")]
mod embedded_web;
// `pub(crate)` so the sc-22714 review suite can drive the same in-process API + fake worker
// instead of standing up a second one.
pub(crate) mod film_harness;
mod film_harness_fixes;
mod film_harness_references;
mod film_harness_review;
mod jobs;
mod mcp;
mod media;
mod model_cache;
mod model_library;
mod projects;
mod prompt_batches;
mod recipe_presets;
mod server;
mod startup;
// `pub(crate)` so the inline `#[cfg(test)]` test modules that live OUTSIDE this `tests`
// tree (e.g. `crate::models::variant_install_tests`) can reuse the ONE crate-wide
// `isolate_hf_cache` / `HF_ENV_LOCK` guard rather than adding a second lock (sc-13835).
pub(crate) mod support;
mod training;
mod uploads;
mod workflows;
