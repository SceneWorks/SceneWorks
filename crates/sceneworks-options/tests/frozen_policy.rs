#[path = "support/frozen_policy.rs"]
mod frozen_policy;
#[test]
fn frozen_source_policy_literals_survive_full_resolution() {
    frozen_policy::run(&frozen_policy::facts());
}
