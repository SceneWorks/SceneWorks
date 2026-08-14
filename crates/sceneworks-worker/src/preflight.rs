//! Startup hardware preflight (sc-8411, sc-16247).
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
//!
//! Off-Mac, [`cuda_preflight`] is the exact same idea for CUDA (sc-16247, GH #1966). The
//! desktop's `nvidia-smi` preflight probes **NVML** (`libnvidia-ml.so` / `nvml.dll`) and
//! checks a driver-version floor + compute-capability range. That is a genuinely useful
//! pre-download gate, but it never calls `cuInit`, so it cannot see the class of failure
//! where the CUDA **driver API** itself refuses to initialize — most notably
//! `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` (803), raised when `libcuda.so`/`nvcuda.dll` and
//! the loaded NVIDIA kernel module are different versions. Two libraries, two version
//! checks: a host can pass the NVML probe and still fail `cuInit`, which is exactly what
//! GH #1966 reported (a Bazzite box whose atomic image had staged a new driver while the
//! running kernel still held the old module). [`cuda_preflight`] closes that gap by
//! acquiring a real device and running a real kernel — the CUDA twin of the Metal probe's
//! "the exact op that fails in the field".
//!
//! [`cuda_driver_error_guidance`] is the shared, GPU-free token → remedy table used by
//! BOTH the probe above and [`crate::classify_engine_error`], so a driver-class CUDA
//! failure reads the same whether it is caught at startup or escapes into a running job.

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

/// Off-Mac there is no MLX to probe, so this is a no-op; [`cuda_preflight`] is the
/// off-Mac probe. Present on all targets so the `SCENEWORKS_GPU_CHECK` dispatch in the
/// shared binary compiles everywhere.
#[cfg(not(target_os = "macos"))]
pub fn metal_preflight() -> Result<(), String> {
    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA driver-error classification (sc-16247). Pure, GPU-free, all targets.
// ---------------------------------------------------------------------------

/// Fallback when CUDA is unusable for a reason with no specific remedy below.
const CUDA_UNAVAILABLE: &str = "SceneWorks can't initialize the CUDA GPU on this machine. \
Off-Mac generation is NVIDIA/CUDA-only — there is no CPU or AMD fallback. Check that \
`nvidia-smi` lists a supported NVIDIA GPU and that the driver is installed and healthy, \
then restart SceneWorks.";

/// `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` (803) — GH #1966.
const CUDA_DRIVER_MISMATCH: &str = "The NVIDIA kernel driver and the CUDA driver library on \
this machine are different versions, so SceneWorks can't use the GPU. This almost always \
means a driver update has been installed but the machine has not been rebooted onto it yet \
— reboot, then reopen SceneWorks. On an immutable/atomic Linux distribution (Bazzite, \
Silverblue, Bluefin, …) a driver update is staged into the NEXT boot, so a reboot is \
required even if the app has been running for days. Note that `nvidia-smi` can still look \
healthy in this state: it queries NVML, which version-checks separately from CUDA.";

/// `CUDA_ERROR_INSUFFICIENT_DRIVER` (35).
const CUDA_INSUFFICIENT_DRIVER: &str = "The installed NVIDIA driver is older than the CUDA \
runtime SceneWorks ships, so the GPU can't be used. Update the NVIDIA driver — 576.02 or \
newer on Windows, 575.51.03 or newer on Linux — then restart SceneWorks.";

/// `CUDA_ERROR_SYSTEM_NOT_READY` (802).
const CUDA_SYSTEM_NOT_READY: &str = "The NVIDIA driver stack on this machine isn't ready yet \
— the driver or GPU fabric is still initializing. Wait until `nvidia-smi` lists your GPU and \
try again; if it persists, reboot.";

/// No CUDA driver library at all — `libcuda.so.1` / `nvcuda.dll` could not be loaded. Distinct
/// from every code below, which all require a driver that at least loaded. The dominant cause
/// is a container started without GPU access, where the NVIDIA container runtime never
/// injected the driver.
const CUDA_DRIVER_LIBRARY_MISSING: &str = "SceneWorks could not load the NVIDIA CUDA driver \
library (`nvcuda.dll` on Windows, `libcuda.so.1` on Linux), so the GPU can't be used at all. \
If you're running the container/server build, start it with GPU access — `docker run --gpus \
all ...`, or `deploy.resources.reservations.devices` in Compose — and make sure the NVIDIA \
Container Toolkit is installed on the host. Otherwise, install or repair the NVIDIA driver \
and restart SceneWorks.";

/// `CUDA_ERROR_NO_DEVICE` (100) / `CUDA_ERROR_INVALID_DEVICE` (101).
const CUDA_NO_DEVICE: &str = "CUDA started, but no usable NVIDIA GPU is visible to SceneWorks. \
Check that an NVIDIA GPU is present and enabled, that `CUDA_VISIBLE_DEVICES` isn't hiding it, \
and — if you're running the container/server build — that the container was started with GPU \
access (for example `docker run --gpus all`).";

/// Map a stringified CUDA failure onto actionable host-side guidance, or `None` when it
/// isn't a driver-class CUDA error at all.
///
/// **Matched on the stringified error, deliberately — not a typed downcast.**
/// `gen_core::Error::Backend` is a `Box<dyn Error + Send + Sync>` because gen-core cannot
/// name backend types (see `gen-core/src/error.rs`). Recovering the code would mean a
/// downcast chain `gen_core` → `candle_core::Error::Cuda` → `cudarc::driver::DriverError`,
/// which drags a candle dependency into this crate's error path and re-breaks on every
/// candle/cudarc bump. The `CUDA_ERROR_*` token, by contrast, is a stable part of the
/// rendered message on every path: `cudarc`'s `Display for DriverError` forwards to its
/// `Debug`, which prints the `CUresult` variant name — e.g.
/// `DriverError(CUDA_ERROR_SYSTEM_DRIVER_MISMATCH, "system has unsupported display driver /
/// cuda driver combination")`. So this is version-stable, dependency-free, works on any
/// target, and is unit-testable with no GPU — which is why it is also the *only* half of
/// sc-16247 that a non-CUDA machine can verify.
///
/// Order matters only in that every token here is a distinct full identifier; none is a
/// substring of another (`NO_DEVICE` is not a substring of `INVALID_DEVICE`).
pub(crate) fn cuda_driver_error_guidance(message: &str) -> Option<&'static str> {
    // Ordered most-specific first purely for readability; the tokens are disjoint.
    const TABLE: &[(&str, &str)] = &[
        ("CUDA_ERROR_SYSTEM_DRIVER_MISMATCH", CUDA_DRIVER_MISMATCH),
        ("CUDA_ERROR_INSUFFICIENT_DRIVER", CUDA_INSUFFICIENT_DRIVER),
        ("CUDA_ERROR_SYSTEM_NOT_READY", CUDA_SYSTEM_NOT_READY),
        // Both "CUDA is up but there's nothing to run on" shapes. `cuInit` succeeding with
        // zero visible devices surfaces as INVALID_DEVICE from `cuDeviceGet(_, 0)` (that is
        // what `CUDA_VISIBLE_DEVICES=-1` produces), while a driver that enumerates nothing
        // raises NO_DEVICE. Same remedy, so they share a message.
        ("CUDA_ERROR_NO_DEVICE", CUDA_NO_DEVICE),
        ("CUDA_ERROR_INVALID_DEVICE", CUDA_NO_DEVICE),
    ];
    TABLE
        .iter()
        .find(|(token, _)| message.contains(token))
        .map(|(_, guidance)| *guidance)
}

// ---------------------------------------------------------------------------
// CUDA device-acquisition preflight (sc-16247). Off-Mac candle builds only.
// ---------------------------------------------------------------------------

/// Verify this process can acquire a usable CUDA GPU **and run a kernel on it** — the
/// off-Mac counterpart of [`metal_preflight`], and the deeper second gate behind the
/// desktop's `nvidia-smi` check (which stays: it is the cheap pre-download gate for
/// no-GPU / too-old-driver / unsupported-architecture hosts, and it runs before the
/// multi-GB first-run runtime download; this probe needs that runtime present to run at
/// all).
///
/// Goes through `runtime_cuda::media::default_device()` rather than `Device::new_cuda(0)`
/// directly so the probe uses the *identical* device-acquisition seam every provider load
/// uses, then runs the smallest op that forces real work: an H2D copy, a dtype-cast kernel
/// (which also forces candle's PTX module load), and a D2H read that synchronizes. Any of
/// those failing is what the field sees as `backend op failed: DriverError(...)` mid-load.
///
/// Probes device 0 only, which is the right scope for the failure class this exists to
/// catch: a driver/userspace version mismatch is machine-wide, not per-GPU. Per-GPU
/// health on a multi-GPU box remains the `auto` supervisor's concern.
///
/// **Panics are caught, and that is load-bearing, not defensive habit.** cudarc loads the
/// CUDA driver library by `dlopen`/`LoadLibrary` at first use (`dynamic-loading`), and when
/// no candidate resolves its `culib()` calls `panic_no_lib_found` — it does not return a
/// `CUresult`. Missing symbols panic the same way. That is precisely the container-started-
/// without-`--gpus` case, where `libcuda.so.1` is simply absent because the NVIDIA container
/// runtime never injected it. Without [`std::panic::catch_unwind`] this function would abort
/// its caller instead of reporting, which for the in-process server-lane call
/// (`log_cuda_preflight_failure`) means killing the worker at startup over a condition it
/// exists to *describe*. Unwinding across the FFI boundary is not a concern: the panic is
/// raised in cudarc's own Rust code before any `extern "C"` call is made.
///
/// `Ok(())` when the GPU is usable; `Err(message)` is the user-facing reason, with the
/// underlying CUDA error appended for the logs — the same shape `metal_preflight` returns,
/// so the `SCENEWORKS_GPU_CHECK=1` sidecar relays either verbatim.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub fn cuda_preflight() -> Result<(), String> {
    use runtime_cuda::media::candle_core::{DType, Tensor};

    let probe = || -> Result<(), String> {
        let device = runtime_cuda::media::default_device().map_err(|error| error.to_string())?;
        // One closure so the three candle_core::Error tails share a single conversion.
        let op = || -> Result<Vec<f32>, runtime_cuda::media::candle_core::Error> {
            Tensor::new(&[1.0f32], &device)?
                .to_dtype(DType::F16)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()
        };
        op().map(|_| ()).map_err(|error| error.to_string())
    };
    match std::panic::catch_unwind(probe) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(cuda_preflight_message(&error)),
        // A panic here is the driver library itself being unloadable, which is its OWN
        // failure mode — not "no device" and not "mismatched versions" — so it gets its own
        // remedy rather than the generic fallback.
        Err(payload) => Err(compose_preflight_message(
            CUDA_DRIVER_LIBRARY_MISSING,
            &panic_reason(payload.as_ref()),
        )),
    }
}

/// Render a caught panic payload as a one-line reason. `catch_unwind` hands back a
/// `Box<dyn Any>`; the two shapes `panic!` ever produces are `&str` and `String`.
///
/// Compiled on all targets (with the same dead-code carve-out as
/// [`cuda_preflight_message`]) so the container-without-GPU message contract is testable
/// where the probe itself cannot be built.
#[cfg_attr(
    not(all(not(target_os = "macos"), feature = "backend-candle")),
    allow(dead_code)
)]
fn panic_reason(payload: &(dyn std::any::Any + Send)) -> String {
    let detail = payload
        .downcast_ref::<&str>()
        .map(|text| (*text).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "no panic message".to_owned());
    // Named so the logs make clear this was a hard failure inside the CUDA driver loader,
    // not a returned CUresult. `panic_no_lib_found`'s own text names the libraries it tried.
    format!("the CUDA driver library could not be loaded: {detail}")
}

/// Compose the probe's user-facing failure text: the specific host-side remedy when the
/// error carries a known `CUDA_ERROR_*` token, the generic CUDA-unavailable message
/// otherwise, always with the raw error appended so the logs keep the real cause.
///
/// Split out from [`cuda_preflight`] and compiled on all targets so the message assembly
/// (including the no-device vs. driver-mismatch split, which must not collapse) is
/// unit-tested on machines that cannot build the CUDA probe itself — which is the entire
/// point: `npm run rust:check` on a Mac never compiles the probe, so this is where the
/// message contract is actually verified.
///
/// That is also why it needs the dead-code carve-out: its only non-test caller is the
/// candle-gated [`cuda_preflight`], so on a build without the candle lane it is reachable
/// from the tests alone, and `-D warnings` would reject it.
#[cfg_attr(
    not(all(not(target_os = "macos"), feature = "backend-candle")),
    allow(dead_code)
)]
pub(crate) fn cuda_preflight_message(underlying: &str) -> String {
    let guidance = cuda_driver_error_guidance(underlying).unwrap_or(CUDA_UNAVAILABLE);
    compose_preflight_message(guidance, underlying)
}

/// The one place the probe's two-part message shape is defined: actionable remedy first, raw
/// error after a blank line. Shared so the caught-panic path (whose guidance is chosen by the
/// failure mode, not by a `CUDA_ERROR_*` token) produces an identically-shaped message.
pub(crate) fn compose_preflight_message(guidance: &str, underlying: &str) -> String {
    format!("{guidance}\n\nUnderlying CUDA error: {underlying}")
}

/// Does this [`cuda_preflight`] failure describe a host condition that GENUINELY prevents
/// generation, as opposed to one that might clear on its own?
///
/// This is the difference between "show the user a setup screen and refuse to start" and
/// "log it and let the app run", and getting it wrong in the permissive direction costs one
/// failed job while getting it wrong in the strict direction **locks the user out of the whole
/// application** — Library, Settings, everything — over a condition that may not even involve
/// the driver. So only the definite, non-transient host states qualify:
///
/// - any recognized driver-class `CUDA_ERROR_*` code (the family this story exists for), and
/// - the CUDA driver library failing to load at all.
///
/// Everything else — a CUDA OOM because another process (or an orphaned worker from a crashed
/// session) currently holds the GPU, a cuBLAS/cuRAND handle failing to initialize under that
/// same pressure, or any error we do not recognize — is reported and stepped over. The user
/// keeps a usable app, and if they do start a generation the classified message from
/// [`crate::classify_engine_error`] tells them what happened.
///
/// Takes the COMPOSED message (guidance + underlying), which is what crosses the process
/// boundary; the `CUDA_ERROR_*` token survives in its tail.
pub fn cuda_failure_is_blocking(message: &str) -> bool {
    cuda_driver_error_guidance(message).is_some() || message.contains(CUDA_DRIVER_LIBRARY_MISSING)
}

/// Whether this worker's accelerator is usable, as decided by the startup probe (sc-16260).
///
/// The server/Docker lane has no setup screen to refuse startup on, so the verdict has to travel
/// *into* the worker's own behaviour instead: an [`Self::Unusable`] worker withholds the
/// capabilities it can no longer serve (so generation stays queued for a host that gets fixed,
/// rather than being claimed and failed one job at a time) and reports
/// `WorkerStatus::Unhealthy` carrying [`Self::reason`] so an operator sees the host-side remedy
/// without reading container logs.
///
/// [`Self::Usable`] covers both "the probe passed" and "this lane runs no probe" — the CPU
/// utility loops, the macOS `mlx` worker (which has its own Metal gate on the desktop), and any
/// build without the candle lane linked. Those must behave exactly as they did before this
/// existed, so the absence of a probe is deliberately indistinguishable from a passing one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GpuHealth {
    Usable,
    Unusable {
        /// The composed user-facing text from [`cuda_preflight`]: the host-side remedy chosen by
        /// [`cuda_driver_error_guidance`], then the raw CUDA error. Reused verbatim rather than
        /// re-authored, so the startup log, the worker status and a mid-job failure all name the
        /// same fix.
        reason: String,
    },
}

impl GpuHealth {
    /// The remedy text when unusable; `None` when the accelerator is fine.
    pub(crate) fn reason(&self) -> Option<&str> {
        match self {
            Self::Usable => None,
            Self::Unusable { reason } => Some(reason.as_str()),
        }
    }

    pub(crate) fn is_usable(&self) -> bool {
        matches!(self, Self::Usable)
    }
}

/// No-op wherever the candle/CUDA lane isn't linked (macOS, and any off-Mac build without
/// `backend-candle`): there is no CUDA runtime to probe, and the desktop keeps the lane
/// dormant in that state anyway. Present on all targets so the `SCENEWORKS_GPU_CHECK`
/// dispatch in the shared binary compiles everywhere, exactly like [`metal_preflight`].
#[cfg(not(all(not(target_os = "macos"), feature = "backend-candle")))]
pub fn cuda_preflight() -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The literal text GH #1966 reported, as it reaches the worker: `gen_core::Error::Backend`
    /// renders `backend op failed: {0}`, and cudarc's `Display for DriverError` forwards to its
    /// `Debug`, which prints the `CUresult` variant name plus `cuGetErrorString`'s text.
    const REPORTED_803: &str = "backend op failed: DriverError(CUDA_ERROR_SYSTEM_DRIVER_MISMATCH, \
        \"system has unsupported display driver / cuda driver combination\")";

    #[test]
    fn the_reported_driver_mismatch_maps_to_the_reboot_remedy() {
        let guidance = cuda_driver_error_guidance(REPORTED_803)
            .expect("803 must be recognized as a CUDA error");
        assert_eq!(
            guidance, CUDA_DRIVER_MISMATCH,
            "the exact string from GH #1966 must map to the driver-mismatch remedy"
        );
        // The remedy has to actually name the host-side action, not just restate the error.
        assert!(
            guidance.contains("reboot"),
            "the 803 remedy must tell the user to reboot, got: {guidance}"
        );
    }

    #[test]
    fn every_driver_class_code_maps_to_its_own_remedy() {
        // Pinned to DISTINCT expected values, so deleting an arm or collapsing two of them
        // fails here rather than passing on a shared default.
        for (token, expected) in [
            ("CUDA_ERROR_SYSTEM_DRIVER_MISMATCH", CUDA_DRIVER_MISMATCH),
            ("CUDA_ERROR_INSUFFICIENT_DRIVER", CUDA_INSUFFICIENT_DRIVER),
            ("CUDA_ERROR_SYSTEM_NOT_READY", CUDA_SYSTEM_NOT_READY),
            ("CUDA_ERROR_NO_DEVICE", CUDA_NO_DEVICE),
            ("CUDA_ERROR_INVALID_DEVICE", CUDA_NO_DEVICE),
        ] {
            let rendered = format!("backend op failed: DriverError({token}, \"whatever\")");
            assert_eq!(
                cuda_driver_error_guidance(&rendered),
                Some(expected),
                "{token} must map to its own remedy"
            );
        }
    }

    /// The two failure shapes the story explicitly forbids collapsing: "this machine has no
    /// CUDA device" and "this machine's CUDA driver stack is mismatched" want different
    /// remedies, because the user actions are different (check the GPU/container flags vs.
    /// reboot onto the staged driver).
    #[test]
    fn no_device_and_driver_mismatch_do_not_share_a_message() {
        let no_device = cuda_driver_error_guidance("DriverError(CUDA_ERROR_NO_DEVICE, \"x\")")
            .expect("no-device recognized");
        let mismatch = cuda_driver_error_guidance(REPORTED_803).expect("mismatch recognized");
        assert_ne!(
            no_device, mismatch,
            "the no-device and driver-mismatch remedies must stay distinct"
        );
        assert!(
            no_device.contains("--gpus"),
            "the no-device remedy must cover the container case, got: {no_device}"
        );
    }

    #[test]
    fn non_cuda_failures_are_not_claimed_as_driver_errors() {
        // A CUDA OOM is a real CUDA error but NOT a driver-stack problem: rewriting it as
        // "reboot your machine" would be actively misleading. Only the driver-class family
        // may match.
        for unrelated in [
            "backend op failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")",
            "missing tensor: unet.down_blocks.0",
            "backend op failed: cuda error: an illegal memory access was encountered",
            "",
        ] {
            assert_eq!(
                cuda_driver_error_guidance(unrelated),
                None,
                "{unrelated:?} must not be classified as a driver-class CUDA failure"
            );
        }
    }

    #[test]
    fn the_probe_message_keeps_the_underlying_error_for_the_logs() {
        let message = cuda_preflight_message(REPORTED_803);
        assert!(
            message.starts_with(CUDA_DRIVER_MISMATCH),
            "the actionable remedy must lead, got: {message}"
        );
        assert!(
            message.contains(REPORTED_803),
            "the raw CUDA error must be preserved for the logs, got: {message}"
        );
    }

    /// The container-without-`--gpus` contract. cudarc loads the CUDA driver library lazily and
    /// `panic!`s (`panic_no_lib_found`) when no candidate resolves — it does NOT return a
    /// `CUresult`. `cuda_preflight` therefore wraps the probe in `catch_unwind`, and the
    /// server-lane caller runs IN-PROCESS at worker startup, so without that wrapper a GPU-less
    /// container would abort the worker instead of logging why. This pins the two halves that
    /// make the wrapper work — the payload actually renders, and the composed message names the
    /// container remedy — on any machine, with no CUDA and no panic of our own.
    #[test]
    fn a_missing_driver_library_reads_as_the_container_remedy() {
        // Both payload shapes `panic!` can produce. cudarc's is a formatted String.
        let from_string: Box<dyn std::any::Any + Send> =
            Box::new(String::from("Unable to find lib: \"libcuda.so\""));
        let from_str: Box<dyn std::any::Any + Send> = Box::new("boom");
        assert!(
            panic_reason(from_string.as_ref()).contains("libcuda.so"),
            "a String payload must survive into the reason"
        );
        assert!(
            panic_reason(from_str.as_ref()).contains("boom"),
            "a &str payload must survive into the reason"
        );

        let message = compose_preflight_message(
            CUDA_DRIVER_LIBRARY_MISSING,
            &panic_reason(from_string.as_ref()),
        );
        assert!(
            message.contains("--gpus"),
            "the missing-driver-library remedy must name container GPU access, got: {message}"
        );
        assert!(
            message.contains("libcuda.so"),
            "the underlying loader failure must reach the logs, got: {message}"
        );
        // It must NOT be mistaken for the "GPU present but hidden" case: those have different
        // remedies (install/expose the driver vs. check CUDA_VISIBLE_DEVICES).
        assert_ne!(
            CUDA_DRIVER_LIBRARY_MISSING, CUDA_NO_DEVICE,
            "a missing driver library and a hidden device must not share a message"
        );
    }

    /// The severity split that decides whether a probe failure STOPS THE APP or is merely logged
    /// (sc-16247). Over-blocking is the expensive direction: it locks the user out of Library,
    /// Settings and everything else, so only definite, non-transient host states may qualify.
    #[test]
    fn only_definite_host_problems_block_startup() {
        // Blocks: the whole driver-class family, and no driver library at all.
        for token in [
            "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH",
            "CUDA_ERROR_INSUFFICIENT_DRIVER",
            "CUDA_ERROR_SYSTEM_NOT_READY",
            "CUDA_ERROR_NO_DEVICE",
            "CUDA_ERROR_INVALID_DEVICE",
        ] {
            let composed = cuda_preflight_message(&format!("DriverError({token}, \"whatever\")"));
            assert!(
                cuda_failure_is_blocking(&composed),
                "{token} is a definite host problem and must block startup"
            );
        }
        let missing = compose_preflight_message(
            CUDA_DRIVER_LIBRARY_MISSING,
            "the CUDA driver library could not be loaded: Unable to find lib",
        );
        assert!(
            cuda_failure_is_blocking(&missing),
            "no CUDA driver library at all must block startup"
        );

        // Does NOT block: transient or unrecognized failures. A CUDA OOM is the motivating case —
        // another process (or an orphaned worker from a crashed session) holding the GPU must not
        // make the entire application unopenable.
        for transient in [
            "DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")",
            "cuda error: CUBLAS_STATUS_NOT_INITIALIZED",
            "some brand new cudarc failure",
        ] {
            let composed = cuda_preflight_message(transient);
            assert!(
                !cuda_failure_is_blocking(&composed),
                "{transient:?} may be transient and must NOT lock the user out of the app"
            );
        }
    }

    #[test]
    fn an_unrecognized_cuda_failure_still_gets_the_generic_requirement() {
        let message = cuda_preflight_message("some brand new cudarc failure");
        assert!(
            message.starts_with(CUDA_UNAVAILABLE),
            "an unmapped failure must still explain that CUDA is required, got: {message}"
        );
        assert!(message.contains("some brand new cudarc failure"));
    }
}
