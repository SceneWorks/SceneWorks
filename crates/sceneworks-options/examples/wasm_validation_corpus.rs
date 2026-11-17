//! Actual WASM execution of the same metadata-only native corpus. No host or model imports.
#[path = "../tests/support/corpus.rs"]
mod corpus;
#[path = "../tests/support/frozen_policy.rs"]
mod frozen_policy;
fn main() {
    assert!(corpus::run() > 0);
    frozen_policy::run(&frozen_policy::facts());
}
