#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The desktop app launches this same binary a second time as a standalone
    // GPU worker — the Apple-Silicon MLX worker (sc-3289) — by setting
    // SCENEWORKS_WORKER_ONLY=1. The binary already links the mlx-gen engine, so
    // reusing it avoids bundling a second multi-hundred-MB sidecar.
    // One-shot Metal preflight (sc-8411): the desktop re-launches this binary with
    // SCENEWORKS_GPU_CHECK=1 at startup to verify a usable Metal GPU before spawning
    // the long-running worker. Print the user-facing reason to stdout (the desktop
    // relays it onto the setup screen) and exit non-zero on failure, rather than
    // returning Err (whose Debug print would be noise). Checked before WORKER_ONLY so
    // the probe never starts a worker loop.
    // Exit code carries the SEVERITY, stdout the reason (sc-16247): 0 = usable GPU, 2 = a
    // definite host-side problem the desktop must stop on, 1 = the probe failed for a reason
    // that may not be fatal (a transient CUDA OOM, an unrecognized error), which the desktop
    // logs and steps over rather than locking the user out of the whole app. macOS keeps its
    // sc-8411 contract: every Metal failure is blocking, so it always exits 2 here.
    if std::env::var("SCENEWORKS_GPU_CHECK").is_ok_and(|value| value.trim() == "1") {
        match sceneworks_rust_api::gpu_check() {
            Ok(()) => std::process::exit(0),
            Err(failure) => {
                println!("{}", failure.message);
                std::process::exit(if failure.blocking { 2 } else { 1 });
            }
        }
    }
    if std::env::var("SCENEWORKS_WORKER_ONLY").is_ok_and(|value| value.trim() == "1") {
        return sceneworks_rust_api::run_worker().await;
    }
    sceneworks_rust_api::run().await
}
