//! Startup hardware preflight (sc-8411).
//!
//! On Apple Silicon the generation worker is MLX-only: every job eventually casts a
//! tensor on the Metal GPU. If MLX can't materialize a default Metal device+stream —
//! a headless / no-window-server session (SSH, a LaunchDaemon), or a transient GPU
//! wedge — the FIRST GPU op fails deep inside a model load with a raw MLX C++
//! assertion (`expected a non-empty mlx_stream`), which also leaks the CI build path
//! baked into MLX's compiled `__FILE__`. The desktop runs [`metal_preflight`] as a
//! one-shot at startup (via the `SCENEWORKS_GPU_CHECK=1` sidecar mode) so an
//! unusable-GPU machine gets a clear, actionable message on the setup screen BEFORE
//! any model load — the macOS counterpart of the Windows `nvidia-smi` `cuda_preflight`.

/// User-facing message when MLX can't acquire a Metal GPU. Authored here (next to the
/// MLX knowledge) and printed to stdout by the `SCENEWORKS_GPU_CHECK=1` probe so the
/// desktop can relay it verbatim onto the setup screen.
#[cfg(target_os = "macos")]
const METAL_UNAVAILABLE: &str = "SceneWorks can't initialize the Metal GPU on this Mac. \
It requires Apple Silicon with GPU access — running over SSH or in a headless session \
(no logged-in graphical session) is not supported. Try opening SceneWorks normally on \
the Mac itself, or reboot and reopen.";

/// Verify this process can acquire a usable Metal GPU by running the smallest MLX op
/// that forces default-device + default-stream acquisition: a 1-element `astype` +
/// `eval` — the exact op that fails in the field. `Ok(())` when the GPU is usable;
/// `Err(message)` is the user-facing reason (with the underlying MLX error appended
/// for the logs).
#[cfg(target_os = "macos")]
pub fn metal_preflight() -> Result<(), String> {
    let probe = mlx_rs::Array::from_slice(&[1.0f32], &[1])
        .as_dtype(mlx_rs::Dtype::Float16)
        .and_then(|array| array.eval());
    match probe {
        Ok(()) => Ok(()),
        Err(error) => Err(format!(
            "{METAL_UNAVAILABLE}\n\nUnderlying MLX error: {error}"
        )),
    }
}

/// Off-Mac the desktop uses its own CUDA preflight (`nvidia-smi`); there is no MLX to
/// probe, so this is a no-op. Present on all targets so the `SCENEWORKS_GPU_CHECK`
/// dispatch in the shared binary compiles everywhere.
#[cfg(not(target_os = "macos"))]
pub fn metal_preflight() -> Result<(), String> {
    Ok(())
}
