// Strict pose jobs render one image per selected pose. Keep that output fan-out at the Rust API's
// product ceiling: eight ordinary maximum-size (8-image) batches. 64 still fits all 46 shipped
// built-ins plus 18 user-created poses. `the_pose_cap_matches_the_request_and_web_validators` reads
// this declaration and the Rust source so either side moving forces an explicit reconciliation.
export const MAX_JOB_POSES = 64;
