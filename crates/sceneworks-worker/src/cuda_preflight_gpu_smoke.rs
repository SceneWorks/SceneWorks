//! Hardware-gated evidence for the sc-16260 recovery re-check (AC 4). `#[ignore]`d — run by hand on
//! a real CUDA box.
//!
//! sc-16260 makes an unhealthy worker re-run [`crate::cuda_preflight`] on an interval instead of
//! staying stranded until someone restarts the container. That is only worth having if a repeated
//! IN-PROCESS probe actually re-reads driver state — a `cuInit` result the CUDA driver latches for
//! the life of the process would make the re-check dead code no matter how correct the surrounding
//! Rust is. That question cannot be answered by a unit test, and the failing side of it cannot be
//! answered on a healthy box without damaging the driver install, so it is answered here with the
//! one negative path that IS safely reproducible: `CUDA_VISIBLE_DEVICES`.
//!
//! ```text
//! cargo test -p sceneworks-worker --features backend-candle --lib \
//!   cuda_preflight_recovers_within_one_process -- --ignored --nocapture
//! ```
//!
//! Single-threaded by construction: it mutates a process-global env var, so it must not run
//! concurrently with anything else that touches CUDA. Pass `--test-threads=1` if combining it with
//! other GPU tests.

/// Does a probe that FAILED for lack of a visible device succeed again, in the same process, once
/// the device is visible? That is exactly the shape the recovery re-check depends on.
///
/// `CUDA_VISIBLE_DEVICES=-1` is the story's own suggested negative path and the one sc-16247
/// validated on this hardware (it surfaces as `CUDA_ERROR_NO_DEVICE`). Hiding the device is not a
/// driver-stack mismatch, so this does not prove the 803 case recovers in-process — nothing safely
/// can, and 803 needs a host reboot anyway, which restarts the container. What it DOES establish is
/// the property the re-check is built on: a failed probe does not poison the process.
#[test]
#[ignore = "needs a real CUDA device; mutates CUDA_VISIBLE_DEVICES process-wide"]
fn cuda_preflight_recovers_within_one_process() {
    const VAR: &str = "CUDA_VISIBLE_DEVICES";
    let original = std::env::var(VAR).ok();

    // 1. Hide every device and probe. This must fail, or the rest proves nothing.
    std::env::set_var(VAR, "-1");
    let hidden = crate::cuda_preflight();
    let hidden_error = hidden.expect_err("with every device hidden the probe must fail");
    println!("[hidden] {hidden_error}");
    assert!(
        hidden_error.contains("CUDA_ERROR_NO_DEVICE")
            || hidden_error.contains("CUDA_ERROR_INVALID_DEVICE"),
        "hiding the devices must surface as a no-device class error, got: {hidden_error}"
    );
    assert!(
        crate::cuda_failure_is_blocking(&hidden_error),
        "a no-device failure is a definite host problem and must be treated as blocking"
    );

    // 2. Restore visibility and probe AGAIN in the same process — the recovery re-check's move.
    match original.as_deref() {
        Some(value) => std::env::set_var(VAR, value),
        None => std::env::remove_var(VAR),
    }
    let recovered = crate::cuda_preflight();
    println!("[restored] {recovered:?}");
    assert!(
        recovered.is_ok(),
        "a failed probe must not poison the process — the sc-16260 recovery re-check depends on \
         a later in-process probe seeing current driver state. Got: {recovered:?}"
    );

    // 3. And a repeat probe on a healthy device stays healthy, so the re-check cannot degrade a
    //    worker by running (each probe builds and drops its own device).
    assert!(
        crate::cuda_preflight().is_ok(),
        "repeated probes on a healthy device must keep succeeding"
    );
}
