//! Hardware-gated evidence for the CUDA primary-context lease the generator cache's cold load takes
//! (`crate::generator_cache::bind_backend_load_context`). `#[ignore]`d — run by hand on a real CUDA
//! box.
//!
//! The cold load brackets `crate::inference_runtime::load` with two `mem_get_info` snapshots to
//! attribute the load's committed bytes. `mem_get_info` is a bare `cuMemGetInfo_v2`: it reports the
//! CURRENT context's device, and the generator cache runs on a dedicated OS thread that binds a
//! context only as a side effect of building the candle device — INSIDE the load, after the pre-load
//! snapshot. Shipping that snapshot without a lease broke every candle image and video generation
//! with `CUDA_ERROR_INVALID_CONTEXT` before the load was ever attempted.
//!
//! No unit test can see this: the seams stub `capture_backend_committed_bytes` and
//! `bind_backend_load_context` to constants precisely because they create no CUDA context. So this
//! pins the DRIVER behaviour the helper encodes, against the same cudarc the load uses.
//!
//! ```text
//! cargo test -p sceneworks-worker --features backend-candle --lib \
//!   generator_cache_context -- --ignored --nocapture
//! ```

use runtime_cuda::media::candle_core::cuda::cudarc::driver::{result, CudaContext};

/// A thread that has never bound a context cannot take a VRAM snapshot; one that holds the lease
/// can; and the lease must outlive the snapshots, because releasing the last reference destroys the
/// primary context out from under the thread that still has it current.
#[test]
#[ignore = "needs a real CUDA device"]
fn the_cold_load_lease_is_what_makes_a_vram_snapshot_readable() {
    // The load path calls `cuInit` (via the candle device); the preflight already did so process-wide
    // in production. Establish the same starting state: driver up, no context current on this thread.
    result::init().expect("cuInit must succeed on a CUDA box");

    std::thread::Builder::new()
        .name("generator-cache-context-smoke".to_owned())
        .spawn(|| {
            let unbound = result::mem_get_info().expect_err(
                "a thread with no bound context must NOT be able to read device memory — if this \
                 ever starts succeeding, the lease is no longer load-bearing and this smoke is stale",
            );
            println!("[unbound] {unbound:?}");
            assert!(
                format!("{unbound:?}").contains("CUDA_ERROR_INVALID_CONTEXT"),
                "the unbound read must fail as an invalid context, got: {unbound:?}"
            );

            let context = CudaContext::new(0).expect("retain the primary context for device 0");
            context.bind_to_thread().expect("bind it to this thread");
            let (free, total) = result::mem_get_info()
                .expect("with the lease held the pre-load snapshot must read the device");
            println!("[leased] free={free} total={total}");
            assert!(total > 0 && free <= total, "the snapshot must be coherent");

            // Why the lease spans the whole load rather than each snapshot: the primary context is
            // refcounted, and dropping the last reference destroys it while this thread still has it
            // current. Taking and releasing it per snapshot would leave the post-load read reading a
            // corpse.
            drop(context);
            let destroyed = result::mem_get_info()
                .expect_err("releasing the last reference must destroy the context");
            println!("[released] {destroyed:?}");
            assert!(
                format!("{destroyed:?}").contains("CUDA_ERROR_CONTEXT_IS_DESTROYED"),
                "a released primary context must surface as destroyed, got: {destroyed:?}"
            );
        })
        .expect("spawn the smoke thread")
        .join()
        .expect("the smoke thread must not panic");
}
