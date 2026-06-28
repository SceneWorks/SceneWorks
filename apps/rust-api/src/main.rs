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
    if std::env::var("SCENEWORKS_GPU_CHECK").is_ok_and(|value| value.trim() == "1") {
        match sceneworks_rust_api::gpu_check() {
            Ok(()) => std::process::exit(0),
            Err(message) => {
                println!("{message}");
                std::process::exit(1);
            }
        }
    }
    if std::env::var("SCENEWORKS_WORKER_ONLY").is_ok_and(|value| value.trim() == "1") {
        return sceneworks_rust_api::run_worker().await;
    }
    sceneworks_rust_api::run().await
}
