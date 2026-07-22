//! rust-api test suite, split by domain from the former single `tests.rs`
//! (sc-11217, F-030). Shared fixtures/helpers live in `support`.

mod auth;
mod catalog;
mod jobs;
mod mcp;
mod media;
mod projects;
mod prompt_batches;
mod recipe_presets;
mod server;
// `pub(crate)` so the inline `#[cfg(test)]` test modules that live OUTSIDE this `tests`
// tree (e.g. `crate::models::variant_install_tests`) can reuse the ONE crate-wide
// `isolate_hf_cache` / `HF_ENV_LOCK` guard rather than adding a second lock (sc-13835).
pub(crate) mod support;
mod training;
mod uploads;
