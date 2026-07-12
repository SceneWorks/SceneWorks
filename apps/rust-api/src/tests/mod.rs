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
mod support;
mod training;
mod uploads;
