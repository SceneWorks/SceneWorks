//! Actual WASM execution of the same metadata-only native corpus. No host or model imports.
#[path = "../tests/support/corpus.rs"]
mod corpus;
fn main() {
    assert!(corpus::run() > 0);
}
