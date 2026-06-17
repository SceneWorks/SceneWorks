//! CUDA execution-provider dependency preloading for the off-Mac `ort` (onnxruntime)
//! paths (epic 5482). Shared by every candle GPU-worker `ort` surface — DWPose
//! pose_detect (sc-5496), YOLO person-detect (sc-5498), Real-ESRGAN upscale
//! (sc-5499) — so the CUDA execution provider's runtime dependencies resolve
//! regardless of `PATH`.
//!
//! The problem: the worker links `ort` with `load-dynamic`, dlopening an
//! onnxruntime-gpu build (resolved from `ORT_DYLIB_PATH`) at runtime. That build's
//! `onnxruntime_providers_cuda.dll` in turn needs the CUDA-12 runtime
//! (cudart/cublas/cublasLt/cufft) + cuDNN-9 DLLs. A `torch` import normally arranges
//! `PATH` so these resolve; with the Python stack retired off-Mac, nothing does, so
//! the CUDA EP fails to initialise ("CUDA execution provider is not enabled in this
//! build") and `Detector::load` honestly falls back to the CPU EP.
//!
//! The fix: `ort::ep::cuda::preload_dylibs` dlopens the CUDA + cuDNN DLLs from
//! explicit directories before the CUDA EP is registered, prioritising them in the
//! loader's search order (the off-Mac analogue of bundling the CoreML dylib on Mac,
//! sc-3487). The two directories are resolved independently because they typically
//! differ: the onnxruntime-gpu wheels are CUDA-12 builds whose cudart/cublas/cufft
//! live in the CUDA Toolkit `bin`, while cuDNN-9 ships separately (a cuDNN install,
//! or alongside a torch wheel). The desktop candle bundle stages both into one dir
//! and points both env vars at it (setup.rs); a server/dev box sets the env vars (or
//! relies on the toolkit defaults).
//!
//! Two mechanisms, because onnxruntime + cuDNN load DLLs in two stages:
//!  - `preload_dylibs` immediately dlopens onnxruntime's known CUDA/cuDNN deps from the
//!    resolved dirs, pinning the version-matched copies (CUDA-12) ahead of anything
//!    else on the loader path.
//!  - prepending those dirs to `PATH` (Windows) covers what `preload_dylibs` can't: a
//!    modern cuDNN (9.23) loads its compute-engine sub-libraries LAZILY by name at the
//!    first conv (`cudnn_engines_tensor_ir64_9.dll` et al.), which aren't in ort's fixed
//!    dylib list — without the dir on the loader's standard search those lazy loads fail
//!    mid-inference ("Could not locate cudnn_engines_tensor_ir64_9.dll"). Mirrors the
//!    PATH-prepend the desktop `setup.rs` already does for cudarc's redist DLLs.
//!
//! Best-effort: a preload failure (a missing DLL in the resolved dir) is logged and
//! execution continues — the CUDA EP may still initialise if `PATH` already satisfies
//! it, and if it can't, the `.error_on_failure()` registration surfaces the failure so
//! the detector falls back to CPU and reports `device = "cpu"` honestly.

use std::path::PathBuf;
use std::sync::OnceLock;

/// `SCENEWORKS_ORT_CUDA_DIR` — directory holding the CUDA-12 runtime DLLs the
/// onnxruntime CUDA EP depends on (cudart64_12 / cublas64_12 / cublasLt64_12 /
/// cufft64_11 on Windows; the `.so.12` equivalents on Linux).
const CUDA_DIR_ENV: &str = "SCENEWORKS_ORT_CUDA_DIR";
/// `SCENEWORKS_ORT_CUDNN_DIR` — directory holding the cuDNN-9 DLLs (cudnn64_9 + the
/// graph/ops/heuristic/adv/cnn/engines siblings). Defaults to the CUDA dir when unset
/// (toolkits / bundles that co-locate cuDNN with the CUDA runtime).
const CUDNN_DIR_ENV: &str = "SCENEWORKS_ORT_CUDNN_DIR";

/// Default CUDA Toolkit directory the runtime DLLs are loaded from when neither the
/// env override nor `CUDA_PATH` resolves: the Windows toolkit `bin` / the Linux
/// `lib64`. CUDA 12.9 is the toolchain this candle lane builds + validates against
/// (the onnxruntime-gpu wheels are CUDA-12 builds).
#[cfg(target_os = "windows")]
const DEFAULT_CUDA_DIR: &str = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin";
#[cfg(all(unix, not(target_os = "macos")))]
const DEFAULT_CUDA_DIR: &str = "/usr/local/cuda/lib64";

/// An existing directory from `env_key`, if set + non-empty + present.
fn dir_from_env(env_key: &str) -> Option<PathBuf> {
    let value = std::env::var(env_key).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let path = PathBuf::from(trimmed);
    path.is_dir().then_some(path)
}

/// Resolve the CUDA runtime directory: the explicit `SCENEWORKS_ORT_CUDA_DIR`, then
/// `CUDA_PATH`/bin (a standard toolkit install), then the platform default.
fn resolve_cuda_dir() -> Option<PathBuf> {
    if let Some(dir) = dir_from_env(CUDA_DIR_ENV) {
        return Some(dir);
    }
    if let Some(cuda_path) = std::env::var_os("CUDA_PATH") {
        let bin = if cfg!(windows) {
            PathBuf::from(&cuda_path).join("bin")
        } else {
            PathBuf::from(&cuda_path).join("lib64")
        };
        if bin.is_dir() {
            return Some(bin);
        }
    }
    let default = PathBuf::from(DEFAULT_CUDA_DIR);
    default.is_dir().then_some(default)
}

/// Resolve the cuDNN directory: the explicit `SCENEWORKS_ORT_CUDNN_DIR`, else the
/// CUDA dir (toolkits / bundles that co-locate cuDNN with the CUDA runtime).
fn resolve_cudnn_dir(cuda_dir: Option<&PathBuf>) -> Option<PathBuf> {
    dir_from_env(CUDNN_DIR_ENV).or_else(|| cuda_dir.cloned())
}

/// Prepend the resolved CUDA + cuDNN dirs (deduped, in order) to the process `PATH` so
/// the standard Windows loader search finds both the onnxruntime CUDA provider's direct
/// deps AND cuDNN's lazily-loaded compute-engine sub-DLLs. Windows-only: PATH drives the
/// DLL search there; on Linux the dynamic linker uses `LD_LIBRARY_PATH` (set by the
/// launcher) + the `preload_dylibs` RTLD_GLOBAL handles instead.
#[cfg(target_os = "windows")]
fn prepend_dll_search_path(dirs: &[&PathBuf]) {
    let mut prefix: Vec<PathBuf> = Vec::new();
    for dir in dirs {
        if !prefix.iter().any(|existing| existing == *dir) {
            prefix.push((*dir).clone());
        }
    }
    if prefix.is_empty() {
        return;
    }
    let existing = std::env::var_os("PATH").unwrap_or_default();
    prefix.extend(std::env::split_paths(&existing));
    if let Ok(joined) = std::env::join_paths(prefix) {
        std::env::set_var("PATH", joined);
    }
}

/// Preload the onnxruntime CUDA EP's CUDA + cuDNN dependency DLLs from the resolved
/// directories. Runs at most once per process (idempotent across the det + pose
/// session builds, and across every off-Mac `ort` job path); best-effort.
pub(crate) fn preload_cuda_dylibs() {
    static PRELOADED: OnceLock<()> = OnceLock::new();
    PRELOADED.get_or_init(|| {
        let cuda_dir = resolve_cuda_dir();
        let cudnn_dir = resolve_cudnn_dir(cuda_dir.as_ref());
        if cuda_dir.is_none() && cudnn_dir.is_none() {
            eprintln!(
                "[ort-cuda] no CUDA/cuDNN dir resolved (set {CUDA_DIR_ENV}/{CUDNN_DIR_ENV}); \
                 relying on PATH for the onnxruntime CUDA provider's dependencies"
            );
            return;
        }
        // Put the resolved dirs on the loader search path first, so cuDNN's lazily-loaded
        // sub-engine DLLs (not in `preload_dylibs`' fixed list) resolve at inference time.
        #[cfg(target_os = "windows")]
        {
            let dirs: Vec<&PathBuf> = [cuda_dir.as_ref(), cudnn_dir.as_ref()]
                .into_iter()
                .flatten()
                .collect();
            prepend_dll_search_path(&dirs);
        }
        match ort::ep::cuda::preload_dylibs(cuda_dir.as_deref(), cudnn_dir.as_deref()) {
            Ok(()) => eprintln!(
                "[ort-cuda] preloaded CUDA EP dependencies (cuda={:?}, cudnn={:?})",
                cuda_dir, cudnn_dir
            ),
            // Non-fatal: the CUDA EP may still initialise from PATH; if it can't, the
            // `.error_on_failure()` registration falls the detector back to CPU honestly.
            Err(error) => eprintln!(
                "[ort-cuda] CUDA EP dependency preload failed (cuda={:?}, cudnn={:?}): {error} \
                 — the onnxruntime CUDA provider will fall back to PATH/CPU",
                cuda_dir, cudnn_dir
            ),
        }
    });
}
