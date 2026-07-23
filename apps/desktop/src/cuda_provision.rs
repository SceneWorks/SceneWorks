//! First-run CUDA / onnxruntime redistributable provisioner (Windows candle build).
//!
//! The candle (Windows/CUDA) desktop needs ~2.7 GB of CUDA runtime + cuDNN +
//! onnxruntime-gpu DLLs at runtime: cudarc dynamic-linking `LoadLibrary`s the CUDA
//! runtime by name, and the worker's `ort` paths dlopen a CUDA-enabled onnxruntime
//! (DWPose / YOLO / Real-ESRGAN, epic 5482). We used to bundle these DLLs into the
//! installer (staged by build-sidecar.mjs into the `cuda` + `onnxruntime` resource
//! dirs), but the set blows past NSIS's ~2 GB datablock limit (`makensis`
//! "mmapping datablock" error). Instead we download them once on first run into
//! `%APPDATA%\SceneWorks\gpu-runtime\{cuda,onnxruntime}` and resolve them from
//! there — the same role the bundled resource dirs used to play, just relocated and
//! fetched lazily.
//!
//! The source is the PyPI `nvidia-*-cu12` + `onnxruntime-gpu` wheels (each a zip):
//! the same version-matched CUDA 12.9 runtime + the cuDNN/cuFFT/nvJitLink/nvRTC set
//! onnxruntime-gpu 1.26.0 was validated against (cuDNN 9.23 / cuFFT 11.4, sc-5496).
//! URLs + sha256 are pinned (mirrors the pinned-URL pattern used elsewhere for
//! reproducible downloads): resolved once from the PyPI JSON API and baked in below.
//!
//! Idempotent: a `.redist-marker` written after a full success short-circuits later
//! runs, so the multi-GB fetch happens only on the first launch (or after a version
//! bump that changes the marker).
#![cfg(target_os = "windows")]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use tauri::AppHandle;

use crate::cuda_provision_check::{component_provisioned, dir_has_dll, write_component_marker};
use crate::setup::{emit, gpu_runtime_dir};

/// Bump this when the pinned manifest changes so an existing install re-provisions.
/// Written to `<root>\.redist-marker` after a fully successful provision; a run whose
/// marker already equals this string skips the download entirely.
const REDIST_VERSION: &str = "cuda12.9-ort1.26.0-cudnn9.23-1";

/// `SCENEWORKS_GPU_RUNTIME_DIR` — an optional pre-extracted redist dir to install from
/// instead of downloading (sc-10354, offline install). It mirrors the provisioned
/// root's layout: a `cuda\` + `onnxruntime\` pair of subdirs holding the extracted
/// DLLs (e.g. a copy of another machine's `%APPDATA%\SceneWorks\gpu-runtime`, or a
/// bundle staged per the offline-install guide). When set, `provision` copies it into
/// place and skips the multi-GB PyPI download entirely, so a disconnected machine can
/// complete first run. See `docs/offline-install.md`.
const GPU_RUNTIME_DIR_ENV: &str = "SCENEWORKS_GPU_RUNTIME_DIR";

/// Which provisioned subdir a component's DLLs land in. `Cuda` is the cudarc +
/// onnxruntime CUDA-dep dir (cudart/cublas/curand/nvrtc + cuDNN/cuFFT/nvJitLink);
/// `Onnxruntime` holds onnxruntime's own three DLLs.
#[derive(Clone, Copy)]
enum Dest {
    Cuda,
    Onnxruntime,
}

/// A pinned PyPI wheel to fetch + the DLLs to extract from it.
struct Component {
    /// Human label for progress UI ("cuDNN", "cuBLAS", …).
    label: &'static str,
    /// Stable, filename-safe id used for this component's per-run completion marker
    /// (`.component-<slug>.ok`). Kept separate from `label` so tweaking the display
    /// string never silently orphans an on-disk marker.
    slug: &'static str,
    /// Approximate download size, shown in the progress message.
    approx: &'static str,
    /// Pinned win_amd64 wheel URL (files.pythonhosted.org).
    url: &'static str,
    /// sha256 of the wheel, verified after download.
    sha256: &'static str,
    /// Where the extracted DLLs go.
    dest: Dest,
    /// Specific DLL basenames to extract, or `None` to extract every `*.dll` in the
    /// wheel (used for the single-purpose nvidia-*-cu12 wheels).
    dlls: Option<&'static [&'static str]>,
    /// Sentinel DLL name(s) whose presence in `dest` marks this component's extracted
    /// output. Matched by `dir_has_dll`: an exact filename when it ends in `.dll` (the
    /// onnxruntime DLLs), else a case-insensitive, version-agnostic prefix (`cudart64_`
    /// still resolves a `cudart64_13`). Drives both the pre-staged completeness check
    /// (`is_staged_complete`) and the retry-skip guard (`component_provisioned`,
    /// sc-13614). For a multi-DLL wheel a single prefix here is deliberate: the skip
    /// guard pairs it with a completion marker written only after a verified full
    /// extraction, so a partial unzip is never mistaken for complete.
    sentinels: &'static [&'static str],
}

/// The pinned redist set — matches what the build previously bundled (CUDA 12.9 gen
/// libs + the onnxruntime-gpu 1.26.0 CV-aux set). URLs + sha256 were resolved from
/// the PyPI JSON API (`https://pypi.org/pypi/<pkg>/<ver>/json`, the `*-win_amd64.whl`
/// file's `url` + `digests.sha256`). nvidia-*-cu12 wheels put DLLs under
/// `nvidia/<comp>/bin/*.dll`; onnxruntime-gpu under `onnxruntime/capi/*.dll`. The
/// extractor matches by basename, so the internal path is irrelevant.
const COMPONENTS: &[Component] = &[
    // CUDA 12.9 runtime libs cudarc LoadLibrary's by name (the toolkit-redist set
    // build-sidecar.mjs used to copy). Extract every DLL in each single-purpose wheel.
    Component {
        label: "CUDA runtime",
        slug: "cuda-runtime",
        approx: "≈3 MB",
        url: "https://files.pythonhosted.org/packages/59/df/e7c3a360be4f7b93cee39271b792669baeb3846c58a4df6dfcf187a7ffab/nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl",
        sha256: "8e018af8fa02363876860388bd10ccb89eb9ab8fb0aa749aaf58430a9f7c4891",
        dest: Dest::Cuda,
        dlls: None,
        sentinels: &["cudart64_"],
    },
    Component {
        label: "cuBLAS",
        slug: "cublas",
        approx: "≈530 MB",
        url: "https://files.pythonhosted.org/packages/45/a1/a17fade6567c57452cfc8f967a40d1035bb9301db52f27808167fbb2be2f/nvidia_cublas_cu12-12.9.1.4-py3-none-win_amd64.whl",
        sha256: "1e5fee10662e6e52bd71dec533fbbd4971bb70a5f24f3bc3793e5c2e9dc640bf",
        dest: Dest::Cuda,
        dlls: None,
        // `cublasLt` ships inside this same wheel, so `cublas64_` covers the component.
        sentinels: &["cublas64_"],
    },
    Component {
        label: "cuRAND",
        slug: "curand",
        approx: "≈66 MB",
        url: "https://files.pythonhosted.org/packages/e5/98/1bd66fd09cbe1a5920cb36ba87029d511db7cca93979e635fd431ad3b6c0/nvidia_curand_cu12-10.3.10.19-py3-none-win_amd64.whl",
        sha256: "e8129e6ac40dc123bd948e33d3e11b4aa617d87a583fa2f21b3210e90c743cde",
        dest: Dest::Cuda,
        dlls: None,
        sentinels: &["curand64_"],
    },
    Component {
        label: "NVRTC",
        slug: "nvrtc",
        approx: "≈73 MB",
        url: "https://files.pythonhosted.org/packages/52/de/823919be3b9d0ccbf1f784035423c5f18f4267fb0123558d58b813c6ec86/nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-win_amd64.whl",
        sha256: "72972ebdcf504d69462d3bcd67e7b81edd25d0fb85a2c46d3ea3517666636349",
        dest: Dest::Cuda,
        dlls: None,
        sentinels: &["nvrtc64_"],
    },
    // onnxruntime's CUDA execution provider needs cuDNN-9 (incl. its lazily-loaded
    // sub-engine DLLs), cuFFT, nvJitLink. Extract every DLL.
    Component {
        label: "cuDNN",
        slug: "cudnn",
        approx: "≈660 MB",
        url: "https://files.pythonhosted.org/packages/b7/ec/d95cc4204dd45f40f2d1512f8ff0d4c3fb1810a893fecc79fcea05dfec0e/nvidia_cudnn_cu12-9.23.0.39-py3-none-win_amd64.whl",
        sha256: "357e5d59a1b79d27eef754aa79b3d9e7adf11baf86dc928dc114df0033c2c912",
        dest: Dest::Cuda,
        dlls: None,
        // The wheel also ships cuDNN's lazily-loaded sub-engine DLLs; `cudnn64_` marks
        // the component and the completion marker guarantees the rest extracted.
        sentinels: &["cudnn64_"],
    },
    Component {
        label: "cuFFT",
        slug: "cufft",
        approx: "≈190 MB",
        url: "https://files.pythonhosted.org/packages/20/ee/29955203338515b940bd4f60ffdbc073428f25ef9bfbce44c9a066aedc5c/nvidia_cufft_cu12-11.4.1.4-py3-none-win_amd64.whl",
        sha256: "8e5bfaac795e93f80611f807d42844e8e27e340e0cde270dcb6c65386d795b80",
        dest: Dest::Cuda,
        dlls: None,
        sentinels: &["cufft64_"],
    },
    Component {
        label: "nvJitLink",
        slug: "nvjitlink",
        approx: "≈34 MB",
        url: "https://files.pythonhosted.org/packages/dd/7e/2eecb277d8a98184d881fb98a738363fd4f14577a4d2d7f8264266e82623/nvidia_nvjitlink_cu12-12.9.86-py3-none-win_amd64.whl",
        sha256: "cc6fcec260ca843c10e34c936921a1c426b351753587fdd638e8cff7b16bb9db",
        dest: Dest::Cuda,
        dlls: None,
        // nvJitLink_120_0.dll.
        sentinels: &["nvjitlink"],
    },
    // onnxruntime-gpu's own DLLs. The cp312 wheel: the native DLLs are identical
    // across the cp311/cp312/cp313/cp314 ABI wheels (the cp tag only versions the
    // Python `.pyd` binding we don't ship), so any ABI's DLLs are equivalent. Extract
    // exactly the three the worker dlopens (TensorRT is deliberately not staged).
    Component {
        label: "onnxruntime (GPU)",
        slug: "onnxruntime",
        approx: "≈216 MB",
        url: "https://files.pythonhosted.org/packages/a4/e4/9b378a5466ea0bed65e5beb8e09254973c580a6522810a38afbcc45e5105/onnxruntime_gpu-1.26.0-cp312-cp312-win_amd64.whl",
        sha256: "5f49c44689894650990e4c8a857d2edafc276fbd79bba57ceb224bd18d25d491",
        dest: Dest::Onnxruntime,
        dlls: Some(&[
            "onnxruntime.dll",
            "onnxruntime_providers_cuda.dll",
            "onnxruntime_providers_shared.dll",
        ]),
        // The exact three DLLs the worker dlopens — the full extracted set for this
        // component, so all three double as its sentinels.
        sentinels: &[
            "onnxruntime.dll",
            "onnxruntime_providers_cuda.dll",
            "onnxruntime_providers_shared.dll",
        ],
    },
];

/// Root of the provisioned GPU runtime: `%APPDATA%\SceneWorks\gpu-runtime`.
fn root() -> PathBuf {
    gpu_runtime_dir()
}

/// Provisioned CUDA runtime DLL dir (`<root>\cuda`). The candle worker's PATH is
/// prepended with this so cudarc's `LoadLibrary` and onnxruntime's CUDA provider find
/// cudart/cublas/curand/nvrtc/cuDNN/cuFFT/nvJitLink.
pub(crate) fn cuda_dir() -> PathBuf {
    root().join("cuda")
}

/// Provisioned onnxruntime DLL dir (`<root>\onnxruntime`).
fn onnxruntime_dir() -> PathBuf {
    root().join("onnxruntime")
}

/// The provisioned onnxruntime.dll path (set as `ORT_DYLIB_PATH`).
pub(crate) fn onnxruntime_dll() -> PathBuf {
    onnxruntime_dir().join("onnxruntime.dll")
}

/// The provisioned CUDA dir, but only if the redist has actually been downloaded
/// (probes `cudart64_12.dll`, the marker DLL the bundled resolver also probed). The
/// resolvers in setup.rs gate the candle worker / PATH / ORT wiring on this — before
/// first-run provisioning completes it's None, exactly as the empty bundle dir was.
pub(crate) fn cuda_dir_if_present() -> Option<PathBuf> {
    let dir = cuda_dir();
    dir.join("cudart64_12.dll").exists().then_some(dir)
}

/// The provisioned onnxruntime.dll, but only if it has actually been downloaded.
pub(crate) fn onnxruntime_dll_if_present() -> Option<PathBuf> {
    let dll = onnxruntime_dll();
    dll.exists().then_some(dll)
}

/// True when a prior run already provisioned this exact REDIST_VERSION.
fn already_provisioned(root: &Path) -> bool {
    fs::read_to_string(root.join(".redist-marker"))
        .map(|marker| marker.trim() == REDIST_VERSION)
        .unwrap_or(false)
}

/// Which provisioned dir a component's sentinels live in.
fn dest_dir<'a>(component: &Component, cuda: &'a Path, ort: &'a Path) -> &'a Path {
    match component.dest {
        Dest::Cuda => cuda,
        Dest::Onnxruntime => ort,
    }
}

/// True when both provisioned dirs hold every component's sentinel DLL(s) — i.e. a
/// pre-staged redist is complete enough to run without downloading. Derived from the
/// single `COMPONENTS` table (one source of truth for what a component's DLLs are), so
/// adding a component can't skew this check. Deliberately stricter than the single-DLL
/// `*_if_present` resolver probes (which only gate the candle lane): a partial stage
/// that passed a one-DLL probe but was missing cuDNN/nvJitLink would hard-fail at load.
fn is_staged_complete(cuda: &Path, ort: &Path) -> bool {
    COMPONENTS.iter().all(|component| {
        let dir = dest_dir(component, cuda, ort);
        component.sentinels.iter().all(|dll| dir_has_dll(dir, dll))
    })
}

/// Copy every `*.dll` from `src` into `dst` (flat). Returns how many were copied. Used
/// to install a pre-staged redist subdir into the provisioned location.
fn copy_dlls(src: &Path, dst: &Path) -> Result<usize, String> {
    fs::create_dir_all(dst).map_err(|error| format!("create {}: {error}", dst.display()))?;
    let mut copied = 0usize;
    for entry in fs::read_dir(src).map_err(|error| format!("read {}: {error}", src.display()))? {
        let entry = entry.map_err(|error| format!("read {}: {error}", src.display()))?;
        let path = entry.path();
        let is_dll = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("dll"))
            .unwrap_or(false);
        if !is_dll || !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name() else {
            continue;
        };
        fs::copy(&path, dst.join(name))
            .map_err(|error| format!("copy {}: {error}", name.to_string_lossy()))?;
        copied += 1;
    }
    Ok(copied)
}

/// Install a pre-extracted redist from `source` (a `cuda\` + `onnxruntime\` pair) into
/// the provisioned dirs by copying every DLL across, then verify the full sentinel set
/// landed. Errors clearly if the expected subdirs/DLLs are missing so a mis-staged
/// override doesn't masquerade as success (and silently leave the candle lane broken).
fn install_from_staged(source: &Path, cuda: &Path, ort: &Path) -> Result<(), String> {
    if !source.is_dir() {
        return Err(format!(
            "{GPU_RUNTIME_DIR_ENV} points at a missing directory: {}",
            source.display()
        ));
    }
    let src_cuda = source.join("cuda");
    let src_ort = source.join("onnxruntime");
    if !src_cuda.is_dir() || !src_ort.is_dir() {
        return Err(format!(
            "{GPU_RUNTIME_DIR_ENV} ({}) must contain `cuda` and `onnxruntime` subdirectories of \
             extracted DLLs — see docs/offline-install.md",
            source.display()
        ));
    }
    let copied_cuda = copy_dlls(&src_cuda, cuda)?;
    let copied_ort = copy_dlls(&src_ort, ort)?;
    if copied_cuda == 0 || copied_ort == 0 {
        return Err(format!(
            "{GPU_RUNTIME_DIR_ENV} ({}) had no DLLs to install (cuda: {copied_cuda}, \
             onnxruntime: {copied_ort})",
            source.display()
        ));
    }
    if !is_staged_complete(cuda, ort) {
        return Err(format!(
            "pre-staged GPU runtime from {} is incomplete — one or more required CUDA/onnxruntime \
             DLLs are missing; see docs/offline-install.md for the full redist set",
            source.display()
        ));
    }
    Ok(())
}

/// Write the success marker so a fully provisioned root short-circuits `already_provisioned`
/// on later launches.
fn write_marker(root: &Path) -> Result<(), String> {
    fs::write(root.join(".redist-marker"), REDIST_VERSION)
        .map_err(|error| format!("write marker: {error}"))
}

/// Hex-encode a sha256 digest for comparison against the pinned value.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Download one wheel into `tmp`, verify its sha256, and extract its DLLs into the
/// destination dir. The wheel is STREAMED chunk-by-chunk to disk while the sha256 is fed
/// incrementally (F-129, sc-8931), so a ~660 MB wheel never sits whole in RAM — the peak
/// transient allocation is one HTTP chunk, not the entire file (previously the file was
/// buffered by `bytes()`, then moved into the hash task while ALSO being written to
/// disk). The CPU-bound unzip runs on a blocking thread so it doesn't stall the async
/// runtime.
async fn fetch_component(
    client: &reqwest::Client,
    component: &Component,
    tmp_dir: &Path,
    cuda: &Path,
    ort: &Path,
) -> Result<usize, String> {
    let response = client
        .get(component.url)
        .send()
        .await
        .map_err(|error| format!("download {}: {error}", component.label))?
        .error_for_status()
        .map_err(|error| format!("download {}: {error}", component.label))?;

    let wheel_path = tmp_dir.join(format!(
        "{}.whl",
        component.label.replace(['/', ' ', '(', ')'], "_")
    ));

    // Stream the body to the temp file, hashing each chunk as it lands. Peak RAM is one
    // chunk instead of the whole wheel.
    let mut file =
        fs::File::create(&wheel_path).map_err(|error| format!("create wheel: {error}"))?;
    let mut hasher = Sha256::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| format!("download {}: {error}", component.label))?;
        hasher.update(&chunk);
        file.write_all(&chunk)
            .map_err(|error| format!("write {}: {error}", component.label))?;
    }
    file.flush()
        .map_err(|error| format!("write {}: {error}", component.label))?;
    drop(file);

    let digest = hex(&hasher.finalize());
    if digest != component.sha256 {
        return Err(format!(
            "{}: sha256 mismatch (expected {}, got {digest})",
            component.label, component.sha256
        ));
    }

    let dest = match component.dest {
        Dest::Cuda => cuda.to_path_buf(),
        Dest::Onnxruntime => ort.to_path_buf(),
    };
    let label = component.label.to_owned();
    let dlls: Option<Vec<String>> = component
        .dlls
        .map(|names| names.iter().map(|name| name.to_string()).collect());

    // Unzip is CPU/IO bound — keep it off the async executor. The wheel is already on
    // disk and sha256-verified, so this just extracts from the temp file.
    tauri::async_runtime::spawn_blocking(move || -> Result<usize, String> {
        extract_dlls(&wheel_path, dlls.as_deref(), &dest)
            .map_err(|error| format!("{label}: {error}"))
    })
    .await
    .map_err(|error| format!("{}: extract task failed: {error}", component.label))?
}

/// Extract DLLs from a wheel (zip) into `dest` by basename. `names = None` extracts
/// every `*.dll`; otherwise only the listed basenames. Returns how many were written.
fn extract_dlls(wheel: &Path, names: Option<&[String]>, dest: &Path) -> Result<usize, String> {
    fs::create_dir_all(dest).map_err(|error| format!("create {}: {error}", dest.display()))?;
    let file = fs::File::open(wheel).map_err(|error| format!("open wheel: {error}"))?;
    let mut archive = zip::ZipArchive::new(file).map_err(|error| format!("open zip: {error}"))?;
    let mut written = 0usize;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| format!("read zip entry: {error}"))?;
        // Use the sanitized name so a malicious entry can't traverse out of dest.
        let Some(entry_path) = entry.enclosed_name() else {
            continue;
        };
        let Some(base) = entry_path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !base.to_ascii_lowercase().ends_with(".dll") {
            continue;
        }
        if let Some(names) = names {
            if !names.iter().any(|name| name == base) {
                continue;
            }
        }
        let out_path = dest.join(base);
        // Stream the entry straight to disk (F-129, sc-8931) instead of reading the whole
        // DLL into a Vec first — a cuDNN DLL can be hundreds of MB, and io::copy uses a
        // fixed-size buffer.
        let mut out =
            fs::File::create(&out_path).map_err(|error| format!("write {base}: {error}"))?;
        std::io::copy(&mut entry, &mut out).map_err(|error| format!("extract {base}: {error}"))?;
        written += 1;
    }
    Ok(written)
}

/// Provision the CUDA / onnxruntime redist on first run (idempotent). Emits
/// `setup-status` progress per component while it downloads + verifies + extracts the
/// pinned wheels into `%APPDATA%\SceneWorks\gpu-runtime`. A `.redist-marker` written
/// on full success short-circuits later runs. Returns `Err` with a clear message on
/// any failure so the caller can surface it on the setup screen and abort startup.
///
/// Async so it can be `.await`ed from `run_startup` (a Tauri async command) — driving
/// it via `block_on` from inside the runtime would panic / deadlock. Network IO is
/// async (reqwest stream); the per-component hash + unzip are offloaded to a blocking
/// thread inside `fetch_component`.
pub(crate) async fn provision(app: &AppHandle) -> Result<(), String> {
    let root = root();
    if already_provisioned(&root) {
        return Ok(());
    }

    let cuda = cuda_dir();
    let ort = onnxruntime_dir();
    for dir in [&cuda, &ort] {
        fs::create_dir_all(dir).map_err(|error| format!("create {}: {error}", dir.display()))?;
    }

    // Offline install (sc-10354): if the redist is pre-staged, skip the multi-GB PyPI
    // download entirely so a disconnected machine can complete first run.
    //
    // 1. Explicit override — `SCENEWORKS_GPU_RUNTIME_DIR` points at a pre-extracted
    //    redist (a `cuda\` + `onnxruntime\` pair). Honor it or fail clearly: a typo'd
    //    path silently downloading 2.7 GB — or failing offline with a generic error —
    //    would be baffling on an air-gapped box, so `install_from_staged` errors on a
    //    missing/incomplete source rather than falling through to the network.
    if let Ok(value) = std::env::var(GPU_RUNTIME_DIR_ENV) {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            let source = PathBuf::from(trimmed);
            emit(
                app,
                "provision",
                format!(
                    "Installing pre-staged GPU runtime from {}…",
                    source.display()
                ),
                false,
            );
            install_from_staged(&source, &cuda, &ort)?;
            write_marker(&root)?;
            emit(app, "provision", "GPU runtime ready (pre-staged).", false);
            return Ok(());
        }
    }

    // 2. Implicit — a user (or a prior install) already populated the target dirs with
    //    the full DLL set. Adopt them as-is; no network.
    if is_staged_complete(&cuda, &ort) {
        write_marker(&root)?;
        emit(
            app,
            "provision",
            "Found a pre-staged GPU runtime; skipping download.",
            false,
        );
        return Ok(());
    }

    let tmp_dir = root.join(".download-tmp");
    fs::create_dir_all(&tmp_dir).map_err(|error| format!("create temp dir: {error}"))?;

    emit(
        app,
        "provision",
        "Downloading GPU runtime (first run, ~2.7 GB)…",
        false,
    );

    // Streaming multi-GB fetch: a connect + chunk-level read timeout so a server that
    // accepts the connection but stalls (pre- or mid-stream) can't freeze the first-run
    // setup screen forever. No total `timeout` — that would cap the legitimate ~2.7 GB
    // transfer (sc-11149).
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .read_timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|error| format!("http client: {error}"))?;

    let mut outcome: Result<(), String> = Ok(());
    for (index, component) in COMPONENTS.iter().enumerate() {
        let dest = dest_dir(component, &cuda, &ort);

        // Retry-skip (sc-13614): a component that a prior (partial) run already fully
        // provisioned carries a current-version completion marker AND its sentinel DLL(s)
        // on disk — skip it instead of re-downloading. Without this, a failure on any one
        // of the ~8 components aborted the run and left no per-component state, so the
        // next launch re-fetched the whole ~2.7 GB set. Requiring the marker (written
        // only after a verified full extraction) means a partial/truncated extract is
        // never mistaken for complete.
        if component_provisioned(
            &root,
            dest,
            component.slug,
            REDIST_VERSION,
            component.sentinels,
        ) {
            emit(
                app,
                "provision",
                format!(
                    "GPU runtime [{}/{}]: {} already present; skipping.",
                    index + 1,
                    COMPONENTS.len(),
                    component.label
                ),
                false,
            );
            continue;
        }

        emit(
            app,
            "provision",
            format!(
                "Downloading GPU runtime [{}/{}]: {} ({})…",
                index + 1,
                COMPONENTS.len(),
                component.label,
                component.approx
            ),
            false,
        );
        match fetch_component(&client, component, &tmp_dir, &cuda, &ort).await {
            Ok(written) => {
                if let Some(expected) = component.dlls {
                    if written < expected.len() {
                        outcome = Err(format!(
                            "{}: extracted {written}/{} DLLs",
                            component.label,
                            expected.len()
                        ));
                        break;
                    }
                } else if written == 0 {
                    outcome = Err(format!("{}: no DLLs found in wheel", component.label));
                    break;
                }
                // Download + sha256 + extraction all succeeded with the expected DLL count.
                // Record the per-component completion marker LAST (only here) so a retry
                // after a *later* component fails skips re-downloading this one (sc-13614).
                // A partial/truncated extract never reaches this point, so the marker is a
                // trustworthy completion witness.
                if let Err(error) = write_component_marker(&root, component.slug, REDIST_VERSION) {
                    outcome = Err(error);
                    break;
                }
            }
            Err(error) => {
                outcome = Err(error);
                break;
            }
        }
    }

    // Always clean the temp wheels; ignore failures (best effort).
    let _ = fs::remove_dir_all(&tmp_dir);

    outcome?;

    // Mark success last so a partial/aborted run re-provisions next launch.
    write_marker(&root)?;
    emit(app, "provision", "GPU runtime ready.", false);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pinned set must stay internally consistent: every component has a non-empty
    /// pinned URL pointing at the PyPI CDN and a 64-hex-char sha256, plus a filename-safe,
    /// UNIQUE slug and at least one sentinel (both load-bearing for the per-component
    /// completion marker + retry-skip guard, sc-13614). Cheap, offline.
    #[test]
    fn manifest_is_well_formed() {
        assert!(!COMPONENTS.is_empty());
        let mut slugs = std::collections::HashSet::new();
        for component in COMPONENTS {
            assert!(
                component.url.starts_with("https://files.pythonhosted.org/"),
                "{}: url must be a pinned PyPI wheel",
                component.label
            );
            assert!(
                component.url.ends_with("-win_amd64.whl"),
                "{}: url must be the win_amd64 wheel",
                component.label
            );
            assert_eq!(
                component.sha256.len(),
                64,
                "{}: sha256 must be 64 hex chars",
                component.label
            );
            assert!(
                component
                    .sha256
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "{}: sha256 must be lowercase hex",
                component.label
            );
            // Slug is the marker-filename identity: non-empty, filename-safe, and unique
            // so two components can't collide on `.component-<slug>.ok`.
            assert!(
                !component.slug.is_empty(),
                "{}: empty slug",
                component.label
            );
            assert!(
                component
                    .slug
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-'),
                "{}: slug must be filename-safe (alnum/-): {:?}",
                component.label,
                component.slug
            );
            assert!(
                slugs.insert(component.slug),
                "{}: duplicate slug {:?} would collide markers",
                component.label,
                component.slug
            );
            // At least one sentinel — the retry-skip guard needs something to probe.
            assert!(
                !component.sentinels.is_empty(),
                "{}: needs at least one sentinel",
                component.label
            );
            // When an explicit DLL allow-list is given, its exact names double as the
            // sentinels (nothing else identifies the component's output on disk).
            if let Some(dlls) = component.dlls {
                assert_eq!(
                    component.sentinels, dlls,
                    "{}: explicit-DLL component sentinels must equal its dll list",
                    component.label
                );
            }
        }
    }

    /// A fresh, unique temp dir for a test (offline; cleaned by the caller).
    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("sw-offline-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create scratch");
        dir
    }

    /// Drop empty files named `names` into `dir` (stand-ins for the real DLLs).
    fn touch(dir: &Path, names: &[&str]) {
        fs::create_dir_all(dir).expect("create dir");
        for name in names {
            fs::write(dir.join(name), b"").expect("touch dll");
        }
    }

    /// The DLL basenames a complete stage carries (mirrors what the wheels extract):
    /// versioned CUDA libs + the three exact-named onnxruntime DLLs.
    const STAGED_CUDA_DLLS: &[&str] = &[
        "cudart64_12.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "curand64_10.dll",
        "nvrtc64_120_0.dll",
        "cudnn64_9.dll",
        "cufft64_11.dll",
        "nvJitLink_120_0.dll",
    ];
    const STAGED_ORT_DLLS: &[&str] = &[
        "onnxruntime.dll",
        "onnxruntime_providers_cuda.dll",
        "onnxruntime_providers_shared.dll",
    ];

    /// `is_staged_complete` requires the FULL sentinel set in both dirs — a stage
    /// missing even one component (here cuDNN) is rejected, unlike the one-DLL probes.
    #[test]
    fn is_staged_complete_requires_full_set() {
        let root = scratch("complete");
        let cuda = root.join("cuda");
        let ort = root.join("onnxruntime");
        touch(&cuda, STAGED_CUDA_DLLS);
        touch(&ort, STAGED_ORT_DLLS);
        assert!(is_staged_complete(&cuda, &ort));

        // Remove cuDNN — a one-DLL `cudart64_12` probe would still pass, but the full
        // completeness check must not.
        fs::remove_file(cuda.join("cudnn64_9.dll")).expect("rm cudnn");
        assert!(cuda.join("cudart64_12.dll").is_file());
        assert!(!is_staged_complete(&cuda, &ort));

        let _ = fs::remove_dir_all(&root);
    }

    /// `install_from_staged` copies a well-formed `cuda\` + `onnxruntime\` source into
    /// the provisioned dirs and reports complete.
    #[test]
    fn install_from_staged_copies_full_set() {
        let src = scratch("stage-src");
        touch(&src.join("cuda"), STAGED_CUDA_DLLS);
        touch(&src.join("onnxruntime"), STAGED_ORT_DLLS);

        let dest = scratch("stage-dest");
        let cuda = dest.join("cuda");
        let ort = dest.join("onnxruntime");

        install_from_staged(&src, &cuda, &ort).expect("install succeeds");
        assert!(cuda.join("cudart64_12.dll").is_file());
        assert!(ort.join("onnxruntime_providers_cuda.dll").is_file());
        assert!(is_staged_complete(&cuda, &ort));

        let _ = fs::remove_dir_all(&src);
        let _ = fs::remove_dir_all(&dest);
    }

    /// A source missing the `cuda\`/`onnxruntime\` subdirs, or missing a component, is
    /// rejected with an actionable error rather than a silent partial install.
    #[test]
    fn install_from_staged_rejects_bad_source() {
        // No cuda/onnxruntime subdirs at all.
        let flat = scratch("stage-flat");
        touch(&flat, STAGED_CUDA_DLLS);
        let dest = scratch("stage-flat-dest");
        let error = install_from_staged(&flat, &dest.join("cuda"), &dest.join("onnxruntime"))
            .expect_err("must reject a source without cuda/onnxruntime subdirs");
        assert!(
            error.contains("subdirectories"),
            "unexpected error: {error}"
        );

        // Subdirs present but incomplete (cuDNN missing).
        let partial = scratch("stage-partial");
        let incomplete: Vec<&str> = STAGED_CUDA_DLLS
            .iter()
            .copied()
            .filter(|dll| *dll != "cudnn64_9.dll")
            .collect();
        touch(&partial.join("cuda"), &incomplete);
        touch(&partial.join("onnxruntime"), STAGED_ORT_DLLS);
        let dest2 = scratch("stage-partial-dest");
        let error = install_from_staged(&partial, &dest2.join("cuda"), &dest2.join("onnxruntime"))
            .expect_err("must reject an incomplete source");
        assert!(error.contains("incomplete"), "unexpected error: {error}");

        for dir in [&flat, &dest, &partial, &dest2] {
            let _ = fs::remove_dir_all(dir);
        }
    }

    /// End-to-end smoke of the download → sha256 → unzip path on the SMALLEST pinned
    /// wheel (nvidia-cuda-runtime-cu12, ~3 MB): proves the pinned URL/sha256 + the zip
    /// extractor actually produce the expected DLL. Network-gated (`#[ignore]`) so the
    /// normal offline `cargo test` stays fast; run with
    /// `cargo test -p sceneworks-desktop -- --ignored downloader_smoke`.
    #[test]
    #[ignore = "network: downloads ~3 MB from PyPI"]
    fn downloader_smoke() {
        // The smallest component in the manifest (CUDA runtime, ~3 MB).
        let component = COMPONENTS
            .iter()
            .find(|c| c.label == "CUDA runtime")
            .expect("CUDA runtime component present");

        let tmp = std::env::temp_dir().join(format!("sw-cuda-smoke-{}", std::process::id()));
        let dest = tmp.join("cuda");
        let _ = fs::remove_dir_all(&tmp);
        fs::create_dir_all(&dest).expect("create dest");

        // Download (async reqwest) + verify sha256 + unzip — exercising the same code
        // the real provisioner uses (extract_dlls / hex), minus the AppHandle emit.
        let bytes = tauri::async_runtime::block_on(async {
            let client = reqwest::Client::builder().build().expect("client");
            client
                .get(component.url)
                .send()
                .await
                .expect("send")
                .error_for_status()
                .expect("status")
                .bytes()
                .await
                .expect("bytes")
        });

        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let digest = hex(&hasher.finalize());
        assert_eq!(
            digest, component.sha256,
            "sha256 must match the pinned value"
        );

        let wheel = tmp.join("runtime.whl");
        fs::write(&wheel, &bytes).expect("write wheel");
        let written = extract_dlls(&wheel, None, &dest).expect("extract");
        assert!(written >= 1, "at least one DLL extracted");
        assert!(
            dest.join("cudart64_12.dll").exists(),
            "cudart64_12.dll must be present after extraction"
        );

        let _ = fs::remove_dir_all(&tmp);
    }
}
