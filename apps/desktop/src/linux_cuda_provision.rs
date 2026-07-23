//! First-run Linux CUDA / cuDNN / onnxruntime-gpu provisioner.
//!
//! The NVIDIA driver remains a host prerequisite; pinned user-space libraries are
//! downloaded once into `$XDG_DATA_HOME/SceneWorks/gpu-runtime`. The layout mirrors
//! `docker/rust.Dockerfile` and the runtime contract consumed by `setup.rs`.
#![cfg_attr(not(target_os = "linux"), allow(dead_code))]

use std::fs;
#[cfg(test)]
use std::io::Read;
use std::io::Write;
use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use sha2::{Digest, Sha256};
#[cfg(target_os = "linux")]
use tauri::AppHandle;

use crate::cuda_provision_check::write_component_marker;
#[cfg(target_os = "linux")]
use crate::setup::{emit, gpu_runtime_dir};

const REDIST_VERSION: &str = "cuda12.9-ort1.26.0-cudnn9.23-linux-x86_64-1";
const GPU_RUNTIME_DIR_ENV: &str = "SCENEWORKS_GPU_RUNTIME_DIR";

#[derive(Debug)]
struct Component {
    label: &'static str,
    slug: &'static str,
    approx: &'static str,
    url: &'static str,
    sha256: &'static str,
    dest: &'static str,
    /// `None` retains every `*.so*`, including cuDNN's lazy sub-engines.
    extract_prefixes: Option<&'static [&'static str]>,
    sentinels: &'static [&'static str],
}

/// Linux x86_64 URLs/hashes resolved from authoritative PyPI JSON. Versions match
/// the Docker candle image exactly; runtime provisioning never performs pip solving.
const COMPONENTS: &[Component] = &[
    Component {
        label: "CUDA runtime",
        slug: "cuda-runtime",
        approx: "≈3 MB",
        url: "https://files.pythonhosted.org/packages/bc/46/a92db19b8309581092a3add7e6fceb4c301a3fd233969856a8cbf042cd3c/nvidia_cuda_runtime_cu12-12.9.79-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256: "25bba2dfb01d48a9b59ca474a1ac43c6ebf7011f1b0b8cc44f54eb6ac48a96c3",
        dest: "cuda/lib64",
        extract_prefixes: None,
        sentinels: &["libcudart.so"],
    },
    Component {
        label: "cuBLAS",
        slug: "cublas",
        approx: "≈554 MB",
        url: "https://files.pythonhosted.org/packages/77/3c/aa88abe01f3be3d1f8f787d1d33dc83e76fec05945f9a28fbb41cfb99cd5/nvidia_cublas_cu12-12.9.1.4-py3-none-manylinux_2_27_x86_64.whl",
        sha256: "453611eb21a7c1f2c2156ed9f3a45b691deda0440ec550860290dc901af5b4c2",
        dest: "cublas/lib",
        extract_prefixes: None,
        sentinels: &["libcublas.so", "libcublasLt.so"],
    },
    Component {
        label: "cuRAND",
        slug: "curand",
        approx: "≈65 MB",
        url: "https://files.pythonhosted.org/packages/31/44/193a0e171750ca9f8320626e8a1f2381e4077a65e69e2fb9708bd479e34a/nvidia_curand_cu12-10.3.10.19-py3-none-manylinux_2_27_x86_64.whl",
        sha256: "49b274db4780d421bd2ccd362e1415c13887c53c214f0d4b761752b8f9f6aa1e",
        dest: "curand/lib",
        extract_prefixes: None,
        sentinels: &["libcurand.so"],
    },
    Component {
        label: "NVRTC",
        slug: "nvrtc",
        approx: "≈85 MB",
        url: "https://files.pythonhosted.org/packages/b8/85/e4af82cc9202023862090bfca4ea827d533329e925c758f0cde964cb54b7/nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
        sha256: "210cf05005a447e29214e9ce50851e83fc5f4358df8b453155d5e1918094dcb4",
        dest: "cuda_nvrtc/lib",
        extract_prefixes: None,
        sentinels: &["libnvrtc.so"],
    },
    Component {
        label: "cuDNN",
        slug: "cudnn",
        approx: "≈688 MB",
        url: "https://files.pythonhosted.org/packages/7d/9d/1a383211b0967e702b9e84643986fb31bf35ca07bddc19e0cf139fd3291d/nvidia_cudnn_cu12-9.23.0.39-py3-none-manylinux_2_27_x86_64.whl",
        sha256: "89d53e2a2b0614278afbeda67ac89594bdd74f9f283f22f2d34409d55859846f",
        dest: "cudnn/lib",
        extract_prefixes: None,
        sentinels: &["libcudnn.so"],
    },
    Component {
        label: "cuFFT",
        slug: "cufft",
        approx: "≈192 MB",
        url: "https://files.pythonhosted.org/packages/95/f4/61e6996dd20481ee834f57a8e9dca28b1869366a135e0d42e2aa8493bdd4/nvidia_cufft_cu12-11.4.1.4-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
        sha256: "c67884f2a7d276b4b80eb56a79322a95df592ae5e765cf1243693365ccab4e28",
        dest: "cufft/lib",
        extract_prefixes: None,
        sentinels: &["libcufft.so"],
    },
    Component {
        label: "nvJitLink",
        slug: "nvjitlink",
        approx: "≈38 MB",
        url: "https://files.pythonhosted.org/packages/46/0c/c75bbfb967457a0b7670b8ad267bfc4fffdf341c074e0a80db06c24ccfd4/nvidia_nvjitlink_cu12-12.9.86-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl",
        sha256: "e3f1171dbdc83c5932a45f0f4c99180a70de9bd2718c1ab77d14104f6d7147f9",
        dest: "nvjitlink/lib",
        extract_prefixes: None,
        sentinels: &["libnvJitLink.so"],
    },
    Component {
        label: "onnxruntime (GPU)",
        slug: "onnxruntime",
        approx: "≈264 MB",
        url: "https://files.pythonhosted.org/packages/94/fd/59bee7cffaa435da44fefdeb63e29c61de4dbfa4b279852f59cd02c042ae/onnxruntime_gpu-1.26.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        sha256: "3c01119ed4d9449d60367fa8ccffcd02bd3fe736754284e4b198d131f54edad6",
        dest: "onnxruntime/capi",
        extract_prefixes: Some(&[
            "libonnxruntime.so",
            "libonnxruntime_providers_cuda.so",
            "libonnxruntime_providers_shared.so",
        ]),
        sentinels: &[
            "libonnxruntime.so",
            "libonnxruntime_providers_cuda.so",
            "libonnxruntime_providers_shared.so",
        ],
    },
];

fn dir_has_shared_object(dir: &Path, basename: &str) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            return false;
        };
        entry.path().is_file() && (name == basename || name.starts_with(&format!("{basename}.")))
    })
}

fn runtime_complete(root: &Path) -> bool {
    COMPONENTS.iter().all(|component| {
        component
            .sentinels
            .iter()
            .all(|name| dir_has_shared_object(&root.join(component.dest), name))
    })
}

fn component_complete(root: &Path, component: &Component) -> bool {
    fs::read_to_string(root.join(format!(".component-{}.ok", component.slug)))
        .map(|value| value.trim() == REDIST_VERSION)
        .unwrap_or(false)
        && component
            .sentinels
            .iter()
            .all(|name| dir_has_shared_object(&root.join(component.dest), name))
}

fn already_provisioned(root: &Path) -> bool {
    fs::read_to_string(root.join(".redist-marker"))
        .map(|value| value.trim() == REDIST_VERSION)
        .unwrap_or(false)
        && runtime_complete(root)
}

fn write_marker(root: &Path) -> Result<(), String> {
    fs::write(root.join(".redist-marker"), REDIST_VERSION)
        .map_err(|error| format!("write marker: {error}"))
}

#[cfg(test)]
fn hash_reader(mut reader: impl Read) -> Result<String, String> {
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| format!("read for sha256: {error}"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn extract_shared_objects(
    wheel: &Path,
    prefixes: Option<&[&str]>,
    dest: &Path,
) -> Result<usize, String> {
    fs::create_dir_all(dest).map_err(|error| format!("create {}: {error}", dest.display()))?;
    let file = fs::File::open(wheel).map_err(|error| format!("open wheel: {error}"))?;
    let mut archive = zip::ZipArchive::new(file).map_err(|error| format!("open zip: {error}"))?;
    let mut written = 0;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|error| format!("read zip entry: {error}"))?;
        let Some(path) = entry.enclosed_name() else {
            continue;
        };
        let Some(base) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !base.contains(".so")
            || prefixes.is_some_and(|prefixes| {
                !prefixes
                    .iter()
                    .any(|prefix| base == *prefix || base.starts_with(&format!("{prefix}.")))
            })
        {
            continue;
        }
        let mut output =
            fs::File::create(dest.join(base)).map_err(|error| format!("write {base}: {error}"))?;
        std::io::copy(&mut entry, &mut output)
            .map_err(|error| format!("extract {base}: {error}"))?;
        written += 1;
    }
    Ok(written)
}

fn promote_component(stage: &Path, dest: &Path) -> Result<(), String> {
    fs::create_dir_all(dest).map_err(|error| format!("create {}: {error}", dest.display()))?;
    for entry in
        fs::read_dir(stage).map_err(|error| format!("read {}: {error}", stage.display()))?
    {
        let entry = entry.map_err(|error| format!("read {}: {error}", stage.display()))?;
        let target = dest.join(entry.file_name());
        if target.exists() {
            fs::remove_file(&target)
                .map_err(|error| format!("replace {}: {error}", target.display()))?;
        }
        fs::rename(entry.path(), &target)
            .map_err(|error| format!("promote {}: {error}", target.display()))?;
    }
    Ok(())
}

fn install_from_staged(source: &Path, root: &Path) -> Result<(), String> {
    if !source.is_dir() {
        return Err(format!(
            "{GPU_RUNTIME_DIR_ENV} points at a missing directory: {}",
            source.display()
        ));
    }
    if source == root {
        return runtime_complete(root)
            .then_some(())
            .ok_or_else(|| format!("{GPU_RUNTIME_DIR_ENV} points at an incomplete runtime"));
    }
    for component in COMPONENTS {
        let src = source.join(component.dest);
        if !src.is_dir() {
            return Err(format!(
                "{GPU_RUNTIME_DIR_ENV} ({}) is missing `{}`",
                source.display(),
                component.dest
            ));
        }
        let dest = root.join(component.dest);
        fs::create_dir_all(&dest).map_err(|error| format!("create {}: {error}", dest.display()))?;
        for entry in
            fs::read_dir(&src).map_err(|error| format!("read {}: {error}", src.display()))?
        {
            let entry = entry.map_err(|error| format!("read {}: {error}", src.display()))?;
            if entry.path().is_file() {
                fs::copy(entry.path(), dest.join(entry.file_name()))
                    .map_err(|error| format!("copy {}: {error}", entry.path().display()))?;
            }
        }
    }
    runtime_complete(root).then_some(()).ok_or_else(|| {
        format!(
            "pre-staged GPU runtime from {} is incomplete; required CUDA/cuDNN/onnxruntime \
             shared objects are missing",
            source.display()
        )
    })
}

async fn fetch_component(
    client: &reqwest::Client,
    root: &Path,
    component: &'static Component,
) -> Result<(), String> {
    let tmp = root.join(".download-tmp");
    fs::create_dir_all(&tmp).map_err(|error| format!("create temp dir: {error}"))?;
    let wheel = tmp.join(format!("{}.whl", component.slug));
    let response = client
        .get(component.url)
        .send()
        .await
        .map_err(|error| format!("download {}: {error}", component.label))?
        .error_for_status()
        .map_err(|error| format!("download {}: {error}", component.label))?;
    let mut file = fs::File::create(&wheel).map_err(|error| format!("create wheel: {error}"))?;
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
    let actual = format!("{:x}", hasher.finalize());
    if actual != component.sha256 {
        return Err(format!(
            "{}: sha256 mismatch (expected {}, got {actual})",
            component.label, component.sha256
        ));
    }

    let stage = tmp.join(format!("extract-{}", component.slug));
    let _ = fs::remove_dir_all(&stage);
    let prefixes = component.extract_prefixes;
    let wheel_for_extract = wheel.clone();
    let stage_for_extract = stage.clone();
    let written = tauri::async_runtime::spawn_blocking(move || {
        extract_shared_objects(&wheel_for_extract, prefixes, &stage_for_extract)
    })
    .await
    .map_err(|error| format!("{}: extract task failed: {error}", component.label))??;
    if written == 0
        || !component
            .sentinels
            .iter()
            .all(|name| dir_has_shared_object(&stage, name))
    {
        return Err(format!(
            "{}: wheel did not contain the required shared objects",
            component.label
        ));
    }
    promote_component(&stage, &root.join(component.dest))?;
    write_component_marker(root, component.slug, REDIST_VERSION)?;
    let _ = fs::remove_file(wheel);
    let _ = fs::remove_dir_all(stage);
    Ok(())
}

/// Download or adopt the complete Linux runtime. The top-level marker is written
/// only after every component succeeds; a failed launch remains dormant and retries
/// only missing/corrupt components next time.
#[cfg(target_os = "linux")]
pub(crate) async fn provision(app: &AppHandle) -> Result<(), String> {
    if std::env::consts::ARCH != "x86_64" {
        return Err(format!(
            "automatic Linux GPU runtime provisioning supports x86_64 (found {})",
            std::env::consts::ARCH
        ));
    }
    let root = gpu_runtime_dir();
    fs::create_dir_all(&root).map_err(|error| format!("create {}: {error}", root.display()))?;
    if already_provisioned(&root) {
        emit(app, "provision", "GPU runtime ready (cached).", false);
        return Ok(());
    }
    if let Ok(value) = std::env::var(GPU_RUNTIME_DIR_ENV) {
        let value = value.trim();
        if !value.is_empty() {
            let source = PathBuf::from(value);
            emit(
                app,
                "provision",
                format!(
                    "Installing pre-staged Linux GPU runtime from {}…",
                    source.display()
                ),
                false,
            );
            install_from_staged(&source, &root)?;
            write_marker(&root)?;
            emit(app, "provision", "GPU runtime ready (pre-staged).", false);
            return Ok(());
        }
    }
    if runtime_complete(&root) {
        write_marker(&root)?;
        emit(
            app,
            "provision",
            "Found a complete Linux GPU runtime; skipping download.",
            false,
        );
        return Ok(());
    }
    emit(
        app,
        "provision",
        "Downloading Linux GPU runtime (first run, ~1.9 GB)…",
        false,
    );
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .read_timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|error| format!("http client: {error}"))?;
    for (index, component) in COMPONENTS.iter().enumerate() {
        if component_complete(&root, component) {
            emit(
                app,
                "provision",
                format!(
                    "Linux GPU runtime [{}/{}]: {} already present; skipping.",
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
                "Downloading Linux GPU runtime [{}/{}]: {} ({})…",
                index + 1,
                COMPONENTS.len(),
                component.label,
                component.approx
            ),
            false,
        );
        fetch_component(&client, &root, component).await?;
    }
    let _ = fs::remove_dir_all(root.join(".download-tmp"));
    if !runtime_complete(&root) {
        return Err("Linux GPU runtime is incomplete after extraction".to_owned());
    }
    write_marker(&root)?;
    emit(app, "provision", "Linux GPU runtime ready.", false);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn scratch(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "sw-linux-provision-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create scratch");
        root
    }

    fn touch_runtime(root: &Path) {
        for component in COMPONENTS {
            let dir = root.join(component.dest);
            fs::create_dir_all(&dir).expect("create component dir");
            for sentinel in component.sentinels {
                fs::write(dir.join(format!("{sentinel}.fixture")), b"fixture")
                    .expect("write sentinel");
            }
        }
    }

    fn fixture_wheel(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let cursor = Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(cursor);
        for (name, bytes) in entries {
            zip.start_file(*name, zip::write::SimpleFileOptions::default())
                .expect("start fixture entry");
            zip.write_all(bytes).expect("write fixture entry");
        }
        zip.finish().expect("finish fixture wheel").into_inner()
    }

    #[test]
    fn manifest_is_pinned_and_matches_docker_versions() {
        assert_eq!(COMPONENTS.len(), 8);
        let mut slugs = std::collections::HashSet::new();
        for component in COMPONENTS {
            assert!(component.url.starts_with("https://files.pythonhosted.org/"));
            assert!(component.url.contains("x86_64"));
            assert_eq!(component.sha256.len(), 64);
            assert!(component.sha256.chars().all(|ch| ch.is_ascii_hexdigit()));
            assert!(slugs.insert(component.slug));
            assert!(!component.sentinels.is_empty());
        }
        let docker = include_str!("../../../docker/rust.Dockerfile");
        for version in ["1.26.0", "9.23.0.39", "11.4.1.4", "12.9.86"] {
            assert!(docker.contains(version), "Docker pin missing {version}");
            assert!(
                COMPONENTS
                    .iter()
                    .any(|component| component.url.contains(version)),
                "Linux manifest missing {version}"
            );
        }
    }

    #[test]
    fn sha256_verification_detects_corruption() {
        let digest =
            hash_reader(Cursor::new(b"small deterministic wheel fixture")).expect("hash fixture");
        assert_eq!(
            digest,
            "3500f1c4a8dccd6de360fb9120e394494aba434fea810b45226f59fe54c387d1"
        );
        let corrupt = hash_reader(Cursor::new(b"small deterministic wheel fixturf"))
            .expect("hash corrupt fixture");
        assert_ne!(digest, corrupt);
    }

    #[test]
    fn extraction_filters_shared_objects_and_rejects_traversal() {
        let root = scratch("extract");
        let wheel = root.join("fixture.whl");
        fs::write(
            &wheel,
            fixture_wheel(&[
                ("onnxruntime/capi/libonnxruntime.so.1.26.0", b"ort"),
                ("onnxruntime/capi/libonnxruntime_providers_cuda.so", b"cuda"),
                ("../../escape.so", b"escape"),
                ("onnxruntime/capi/pybind_state.so", b"python"),
            ]),
        )
        .expect("write fixture wheel");
        let dest = root.join("out");
        let count = extract_shared_objects(
            &wheel,
            Some(&["libonnxruntime.so", "libonnxruntime_providers_cuda.so"]),
            &dest,
        )
        .expect("extract fixture");
        assert_eq!(count, 2);
        assert!(dest.join("libonnxruntime.so.1.26.0").is_file());
        assert!(dest.join("libonnxruntime_providers_cuda.so").is_file());
        assert!(!dest.join("pybind_state.so").exists());
        assert!(!root.join("escape.so").exists());
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn marker_reuse_requires_current_version_and_all_sentinels() {
        let root = scratch("marker");
        touch_runtime(&root);
        write_marker(&root).expect("write current marker");
        assert!(already_provisioned(&root));
        fs::remove_file(root.join("cudnn/lib/libcudnn.so.fixture")).expect("remove sentinel");
        assert!(!already_provisioned(&root));
        fs::remove_dir_all(root).expect("remove fixture");
    }

    #[test]
    fn staged_override_copies_complete_layout_and_rejects_partial() {
        let source = scratch("staged-source");
        touch_runtime(&source);
        let target = scratch("staged-target");
        install_from_staged(&source, &target).expect("install complete stage");
        assert!(runtime_complete(&target));

        let partial = scratch("staged-partial");
        touch_runtime(&partial);
        fs::remove_file(partial.join("cufft/lib/libcufft.so.fixture"))
            .expect("remove fixture sentinel");
        let rejected = scratch("staged-rejected");
        let error = install_from_staged(&partial, &rejected).expect_err("reject partial stage");
        assert!(error.contains("incomplete"));
        for root in [source, target, partial, rejected] {
            fs::remove_dir_all(root).expect("remove fixture");
        }
    }

    #[test]
    fn component_marker_does_not_accept_partial_layout() {
        let root = scratch("partial");
        let component = &COMPONENTS[1];
        let dest = root.join(component.dest);
        fs::create_dir_all(&dest).expect("create component dest");
        fs::write(dest.join("libcublas.so.12"), b"fixture").expect("write first sentinel");
        write_component_marker(&root, component.slug, REDIST_VERSION).expect("write marker");
        assert!(!component_complete(&root, component));
        fs::write(dest.join("libcublasLt.so.12"), b"fixture").expect("write second sentinel");
        assert!(component_complete(&root, component));
        fs::remove_dir_all(root).expect("remove fixture");
    }

    /// Opt-in CI/manual seam for the real pinned transport without downloading the
    /// multi-GB set. It exercises the smallest Linux wheel end to end.
    #[test]
    #[ignore = "network: downloads the ~3 MB pinned Linux CUDA runtime wheel"]
    fn linux_downloader_smoke() {
        let root = scratch("network-smoke");
        let component = &COMPONENTS[0];
        let client = reqwest::Client::builder().build().expect("HTTP client");
        tauri::async_runtime::block_on(fetch_component(&client, &root, component))
            .expect("download, hash, and extract pinned Linux wheel");
        assert!(component_complete(&root, component));
        assert!(dir_has_shared_object(
            &root.join("cuda/lib64"),
            "libcudart.so"
        ));
        fs::remove_dir_all(root).expect("remove network smoke runtime");
    }
}
