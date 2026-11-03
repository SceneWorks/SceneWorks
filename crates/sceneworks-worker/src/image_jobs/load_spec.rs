#[cfg(not(target_os = "macos"))]
use super::artifact_resolution::FLUX1_DEV_MLX_TURNKEY_REVISION;
#[cfg(not(target_os = "macos"))]
use super::artifact_resolution::FLUX1_SCHNELL_MLX_TURNKEY_REVISION;
#[cfg(not(target_os = "macos"))]
use super::artifact_resolution::FLUX2_DEV_MLX_TURNKEY_REVISION;
#[cfg(not(target_os = "macos"))]
use super::artifact_resolution::FLUX2_KLEIN_9B_MLX_TURNKEY_REVISION;
#[cfg(not(target_os = "macos"))]
use super::artifact_resolution::ZIMAGE_MLX_TURNKEY_REPO;
#[cfg(not(target_os = "macos"))]
use super::artifact_resolution::ZIMAGE_MLX_TURNKEY_REVISION;
use super::quantization::tier_key_from_resolved_dir;
use super::requested_decoder_id;
use super::AdapterSpec;
#[cfg(all(test, target_os = "macos"))]
use super::Generator;
use super::ImageRequest;
use super::LoadSpec;
use super::Quant;
use super::WeightsSource;
use crate::advanced;
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
use crate::BTreeMap;
use crate::JsonObject;
#[cfg(any(
    all(not(target_os = "macos"), feature = "backend-candle"),
    any(test, all(not(target_os = "macos"), feature = "backend-candle"))
))]
use crate::Path;
use crate::PathBuf;
use crate::Settings;
use crate::Value;
use crate::WorkerError;
use crate::WorkerResult;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_certified_artifact_path(
    engine_id: &str,
    settings: &Settings,
    weights_dir: &Path,
    tier: &str,
) -> bool {
    let (repo, revision) = match engine_id {
        "z_image" | "z_image_turbo" => (ZIMAGE_MLX_TURNKEY_REPO, ZIMAGE_MLX_TURNKEY_REVISION),
        "flux1_schnell" => (
            "SceneWorks/flux1-schnell-mlx",
            FLUX1_SCHNELL_MLX_TURNKEY_REVISION,
        ),
        "flux1_dev" => ("SceneWorks/flux1-dev-mlx", FLUX1_DEV_MLX_TURNKEY_REVISION),
        "flux2_dev" => ("SceneWorks/flux2-dev-mlx", FLUX2_DEV_MLX_TURNKEY_REVISION),
        "flux2_klein_9b" => (
            "SceneWorks/flux2-klein-9b-mlx",
            FLUX2_KLEIN_9B_MLX_TURNKEY_REVISION,
        ),
        // Iris-3B generation (sc-25690). The one dense artifact is the pinned upstream snapshot at
        // its ROOT — there is no tier sub-directory to join — and its text encoder is a manifest
        // co-requisite the worker stages cache-only at its own pinned revision
        // (`attach_required_components`), so the backbone root is the whole caller-chosen identity.
        // Without this arm the route was permanently uncertified: the shared ladder then submits
        // the resident estimate alone, and the staged and bounded-attention rungs
        // `candle-gen-iris` implements could never be selected for a request.
        "iris_3b" => {
            return candle_certified_hf_artifact_path(
                settings,
                gen_core_iris::UPSTREAM_WEIGHTS_REPO,
                gen_core_iris::UPSTREAM_WEIGHTS_REVISION,
                Path::new(""),
                weights_dir,
            );
        }
        _ => return false,
    };
    candle_certified_hf_artifact_path(settings, repo, revision, Path::new(tier), weights_dir)
}

/// The numeric tier key the SHARED memory selector binds a request to.
///
/// The VRAM gate names a tier from the resolved directory's basename, or — for a flat root with no
/// tier basename — from `advanced.mlxQuantize` (q8 by default; `requested_tier_key`). That key only
/// has to index the manifest's per-tier floor rows there. The shared selector is stricter: it turns
/// the key into the `MemoryNumericTier` of the selection, and the provider's safety check refuses a
/// selection whose tier is not the one the generator LOADED. Iris-3B refuses quantization and has
/// exactly one dense artifact, so its loaded tier is bf16 whatever the request's advisory quantize
/// knob says; handing the selector the gate's `q8` would mint a `Q8` selection the loaded bf16
/// generator rejects ("selected tier does not match loaded tier") the moment an optimized rung is
/// chosen.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn candle_shared_memory_tier<'a>(engine_id: &str, gate_tier: &'a str) -> &'a str {
    match engine_id {
        "iris_3b" => "bf16",
        // FLUX 3's gate tier IS its loaded package (sc-26241: `Flux3LoadPlan` names it by
        // identity), so the selection binds the package that loads — FP8R as FP8R, never bf16.
        _ => gate_tier,
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn is_mage_engine(engine_id: &str) -> bool {
    matches!(
        engine_id,
        "mage_flow_base"
            | "mage_flow"
            | "mage_flow_turbo"
            | "mage_flow_edit_base"
            | "mage_flow_edit"
            | "mage_flow_edit_turbo"
    )
}

/// Certify the complete artifact identity consumed by one registered image load. Most providers
/// are self-contained, so their identity is the tier directory alone. Mage is split: its DiT lives
/// in the model-specific snapshot while the text encoder and VAE live in a shared pinned snapshot.
/// Optimized evidence is admissible only when every descriptor-required component matches the
/// exact manifest repo, revision, tier, and subdirectory that the eventual provider load receives.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn candle_certified_load_spec(
    engine_id: &str,
    settings: &Settings,
    spec: &LoadSpec,
    manifest_entry: &JsonObject,
    tier: &str,
) -> bool {
    let WeightsSource::Dir(weights_dir) = &spec.weights else {
        return false;
    };
    let declaration_owned =
        crate::memory_route_registry::candle_declaration_owns_load_shape(engine_id)
            || crate::memory_route_registry::candle_manifest_declares_request_strategy_provider(
                manifest_entry,
                engine_id,
            );
    if !is_mage_engine(engine_id) && !declaration_owned {
        return candle_certified_artifact_path(engine_id, settings, weights_dir, tier);
    }
    let Some(downloads) = manifest_entry.get("downloads").and_then(Value::as_array) else {
        return false;
    };
    let certify = |component_id: Option<&str>, actual: &Path| {
        let mut matches = downloads.iter().filter(|download| {
            download.get("provider").and_then(Value::as_str) == Some("huggingface")
                && (download.get("variant").and_then(Value::as_str) == Some(tier)
                    || (component_id.is_some() && download.get("variant").is_none()))
                && download.get("componentId").and_then(Value::as_str) == component_id
                && download
                    .get("coRequisite")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
                    == component_id.is_some()
        });
        let Some(download) = matches.next() else {
            return false;
        };
        if matches.next().is_some() {
            return false;
        }
        let (Some(repo), Some(revision)) = (
            download.get("repo").and_then(Value::as_str),
            download.get("revision").and_then(Value::as_str),
        ) else {
            return false;
        };
        let relative = download
            .get("subdir")
            .and_then(Value::as_str)
            .or_else(|| {
                component_id
                    .and_then(|_| download.get("files"))
                    .and_then(Value::as_array)
                    .and_then(|files| match files.as_slice() {
                        [Value::String(file)] if !file.contains('*') => Some(file.as_str()),
                        _ => None,
                    })
            })
            .unwrap_or(tier);
        candle_certified_hf_artifact_path(settings, repo, revision, Path::new(relative), actual)
    };

    // Component certification is descriptor-driven, and `media_descriptor` only ever resolves a
    // registered GENERATOR. Several declaration-owned runtime providers are registered as encoder
    // contract ROUTES over a sibling generator instead — `z_image_control`,
    // `z_image_turbo_control`, `qwen_image_edit`, `qwen_image_control` — so no descriptor carries
    // their id and no component contract exists to certify against. Those routes bind their
    // co-requisites outside `spec.components`; the exact certification for them is the one this
    // function has always applied: the primary matches its pinned download AND nothing else is
    // bound. Reading a missing descriptor as `false` (as this did between the SDXL-derivative
    // change and sc-20799) reports every such route uncertified, which silently strips its
    // receipt-priced admission down to resident-only.
    let descriptor = crate::inference_runtime::media_descriptor(engine_id);
    if !is_mage_engine(engine_id) && descriptor.is_none() {
        return spec.components.is_empty()
            && (spec.text_encoder.is_none() || spec.validate_prepared_file_pins().is_ok())
            && certify(None, weights_dir);
    }
    let Some(descriptor) = descriptor else {
        return false;
    };
    let expected_component_count = descriptor.required_components.len()
        + usize::from(
            engine_id == "sdxl"
                && spec.ip_adapter.is_some()
                && spec.components.contains_key("sdxl_ip_image_encoder"),
        );
    if spec.components.len() != expected_component_count {
        return false;
    }

    certify(None, weights_dir)
        && descriptor.required_components.iter().all(|component_id| {
            spec.components.get(*component_id).is_some_and(|source| {
                certify(
                    Some(component_id),
                    crate::weights_source::WeightsSourceExt::path(source),
                )
            })
        })
}

/// Bind the endpoint price to the complete ordinary SDXL assembly, including the shared VAE and
/// tokenizers actually opened by Candle. ArtifactSeal's existing stat-identity memo avoids repeated
/// multi-GB hashing; these exact tokens then participate in cache identity and guard registry load.
/// Different base-tier members or consumed component bytes lose endpoint authority. Unused files in
/// a shared tokenizer/VAE snapshot do not affect the price of the same loaded assembly.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn seal_sdxl_endpoint_load_spec(
    spec: &mut LoadSpec,
    expected: &[sceneworks_core::memory_anchor::sdxl_cuda_memory_endpoints::EndpointArtifactFile],
) -> WorkerResult<bool> {
    use std::collections::BTreeMap;

    const COMPONENTS: [(&str, &str); 3] = [
        ("tokenizer_clip_l", "tokenizer.json"),
        ("tokenizer_clip_bigg", "tokenizer.json"),
        ("vae_fp16_fix", "diffusion_pytorch_model.safetensors"),
    ];
    // The measured assembly has one base directory and these three component sources. The stock
    // co-requisite resolver returns File for a single literal manifest member; providers also accept
    // its snapshot Dir. Extra encoders, controls, adapters, or imports cannot inherit this endpoint.
    if spec.file_source_paths().len() + spec.directory_source_paths().len() != 4
        || spec.components.len() != COMPONENTS.len()
        || COMPONENTS
            .iter()
            .any(|(id, _)| !spec.components.contains_key(*id))
    {
        return Ok(false);
    }
    let WeightsSource::Dir(base) = &spec.weights else {
        return Ok(false);
    };
    let mut components = Vec::new();
    for (id, file) in COMPONENTS {
        let path = match spec.components.get(id) {
            Some(WeightsSource::Dir(root)) => root.join(file),
            Some(WeightsSource::File(path))
                if path.file_name() == Some(std::ffi::OsStr::new(file)) =>
            {
                path.clone()
            }
            _ => return Ok(false),
        };
        components.push((format!("components/{id}/{file}"), path));
    }

    fn visit(
        root: &Path,
        directory: &Path,
        prefix: &str,
        files: &mut BTreeMap<String, PathBuf>,
    ) -> WorkerResult<()> {
        for entry in std::fs::read_dir(directory).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "SDXL endpoint inventory cannot read {}: {error}",
                directory.display()
            ))
        })? {
            let path = entry
                .map_err(|error| {
                    WorkerError::InvalidPayload(format!("SDXL endpoint inventory failed: {error}"))
                })?
                .path();
            let metadata = std::fs::symlink_metadata(&path).map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "SDXL endpoint member cannot be inspected: {error}"
                ))
            })?;
            if metadata.is_dir() {
                visit(root, &path, prefix, files)?;
                continue;
            }
            // HF snapshot links to blob files are normal; directory links are not inventory roots.
            if !std::fs::metadata(&path)
                .map_err(|error| {
                    WorkerError::InvalidPayload(format!(
                        "SDXL endpoint member target cannot be inspected: {error}"
                    ))
                })?
                .is_file()
            {
                return Err(WorkerError::InvalidPayload(format!(
                    "SDXL endpoint member must be a file: {}",
                    path.display()
                )));
            }
            let relative = path
                .strip_prefix(root)
                .expect("inventory child stays below root");
            let relative = relative
                .iter()
                .map(|part| part.to_str())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    WorkerError::InvalidPayload(
                        "SDXL endpoint member has a non-UTF-8 path".to_owned(),
                    )
                })?
                .join("/");
            files.insert(format!("{prefix}/{relative}"), path);
        }
        Ok(())
    }

    let mut actual = BTreeMap::new();
    let base = std::path::absolute(base).map_err(|error| {
        WorkerError::InvalidPayload(format!("SDXL endpoint root cannot be absolutized: {error}"))
    })?;
    visit(&base, &base, "base", &mut actual)?;
    // These are the exact joins used by SDXL's resolve_tokenizer_file/resolve_vae_file. The same HF
    // shared snapshot may contain additional CLIP weights installed for an unrelated model.
    for (name, path) in components {
        if !path.is_file() {
            return Ok(false);
        }
        actual.insert(name, path);
    }
    if expected.is_empty()
        || expected.len() != actual.len()
        || expected.iter().any(|file| !actual.contains_key(&file.path))
    {
        return Ok(false);
    }
    let mut pins = Vec::with_capacity(expected.len());
    for file in expected {
        let pin = gen_core::PinnedWeightsFile::pin(&actual[&file.path])
            .map_err(|error| crate::classify_engine_error("SDXL endpoint load pin", error))?;
        let digest = sceneworks_core::hashing::hex(pin.content_sha256());
        if pin.target_fingerprint().size != file.bytes || digest != file.sha256 {
            return Ok(false);
        }
        pins.push(pin);
    }
    crate::paths::prepare_load_spec_with_file_pins(spec, pins, "SDXL endpoint load preparation")?;
    Ok(true)
}

/// Seal every file recursively-inventoried receipt provider can read before selector evaluation.
/// The same finalized `LoadSpec` reaches the generator cache, so cold/warm execution cannot
/// silently re-resolve a pathname after admission.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn seal_recursive_receipt_load_spec(
    engine_id: &str,
    artifact_is_certified: bool,
    spec: &mut LoadSpec,
) -> WorkerResult<()> {
    if !matches!(
        engine_id,
        "chroma1_hd"
            | "chroma1_base"
            | "chroma1_flash"
            | "sd3_5_large"
            | "sd3_5_large_turbo"
            | "sd3_5_medium"
    ) || !artifact_is_certified
    {
        return Ok(());
    }

    fn visit(root: &Path, files: &mut Vec<PathBuf>) -> WorkerResult<()> {
        for entry in std::fs::read_dir(root).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "Receipt artifact inventory cannot read {}: {error}",
                root.display()
            ))
        })? {
            let path = entry
                .map_err(|error| {
                    WorkerError::InvalidPayload(format!(
                        "Receipt artifact inventory failed: {error}"
                    ))
                })?
                .path();
            let metadata = std::fs::symlink_metadata(&path).map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "Receipt artifact member cannot be inspected: {error}"
                ))
            })?;
            if metadata.file_type().is_symlink() {
                let target = std::fs::metadata(&path).map_err(|error| {
                    WorkerError::InvalidPayload(format!(
                        "Receipt artifact symlink target cannot be inspected: {error}"
                    ))
                })?;
                if target.is_file() {
                    files.push(std::path::absolute(&path).map_err(|error| {
                        WorkerError::InvalidPayload(format!(
                            "Receipt artifact member cannot be absolutized: {error}"
                        ))
                    })?);
                    continue;
                }
                return Err(WorkerError::InvalidPayload(format!(
                    "Receipt artifact symlink must resolve directly to a file: {}",
                    path.display()
                )));
            }
            if metadata.is_dir() {
                visit(&path, files)?;
            } else if metadata.is_file() {
                files.push(std::path::absolute(&path).map_err(|error| {
                    WorkerError::InvalidPayload(format!(
                        "Receipt artifact member cannot be absolutized: {error}"
                    ))
                })?);
            }
        }
        Ok(())
    }

    let mut files = spec
        .file_source_paths()
        .into_iter()
        .map(std::path::absolute)
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "Receipt file source cannot be absolutized: {error}"
            ))
        })?;
    for root in spec.directory_source_paths() {
        visit(root, &mut files)?;
    }
    files.sort();
    files.dedup();
    if files.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Receipt artifact inventory is empty".to_owned(),
        ));
    }
    let pins = files
        .iter()
        .map(|path| {
            gen_core::PinnedWeightsFile::pin(path)
                .map_err(|error| crate::classify_engine_error("Receipt load pin", error))
        })
        .collect::<WorkerResult<Vec<_>>>()?;
    crate::paths::prepare_load_spec_with_file_pins(spec, pins, "Receipt load preparation")?;
    spec.validate_prepared_file_pins()
        .map_err(|error| crate::classify_engine_error("Receipt load preparation", error))
}

/// Seal exactly the files the Ideogram Candle loader reads. Direct component safetensors,
/// transformer configs, the tokenizer, TurboTime residual, ordered user adapters, and PiD assets
/// are load identity; nested files below a component are deliberately ignored because the provider
/// does not enumerate them.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn seal_ideogram_load_spec(
    engine_id: &str,
    artifact_is_certified: bool,
    spec: &mut LoadSpec,
) -> WorkerResult<()> {
    if !matches!(engine_id, "ideogram_4" | "ideogram_4_turbo") || !artifact_is_certified {
        return Ok(());
    }
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(WorkerError::InvalidPayload(
            "Ideogram certified load must resolve to one tier directory".to_owned(),
        ));
    };
    let root = root.clone();
    let mut files = Vec::new();
    for component in [
        "transformer",
        "unconditional_transformer",
        "text_encoder",
        "vae",
    ] {
        let dir = root.join(component);
        if !dir.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(&dir).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "Ideogram artifact inventory cannot read {}: {error}",
                dir.display()
            ))
        })? {
            let path = entry
                .map_err(|error| {
                    WorkerError::InvalidPayload(format!(
                        "Ideogram artifact inventory failed: {error}"
                    ))
                })?
                .path();
            if path.is_file()
                && path.extension().and_then(|extension| extension.to_str()) == Some("safetensors")
            {
                files.push(path);
            }
        }
    }
    files.push(root.join("transformer/config.json"));
    if engine_id == "ideogram_4" {
        files.push(root.join("unconditional_transformer/config.json"));
    } else {
        files.push(root.join("turbo_lora.safetensors"));
    }
    files.push(root.join("tokenizer/tokenizer.json"));
    files.extend(spec.file_source_paths().into_iter().map(Path::to_path_buf));
    if let Some(pid) = spec.pid.as_ref() {
        let WeightsSource::Dir(gemma) = &pid.gemma else {
            return Err(WorkerError::InvalidPayload(
                "Ideogram PiD Gemma must resolve to one directory".to_owned(),
            ));
        };
        let merged = gemma.join("gemma-2-2b-it.safetensors");
        if merged.is_file() {
            files.push(merged);
        } else {
            for entry in std::fs::read_dir(gemma).map_err(|error| {
                WorkerError::InvalidPayload(format!(
                    "Ideogram PiD inventory cannot read {}: {error}",
                    gemma.display()
                ))
            })? {
                let path = entry
                    .map_err(|error| {
                        WorkerError::InvalidPayload(format!(
                            "Ideogram PiD inventory failed: {error}"
                        ))
                    })?
                    .path();
                if path.is_file()
                    && path.extension().and_then(|extension| extension.to_str())
                        == Some("safetensors")
                {
                    files.push(path);
                }
            }
        }
        files.push(gemma.join("tokenizer.json"));
    }
    files = files
        .into_iter()
        .map(std::path::absolute)
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "Ideogram load member cannot be absolutized: {error}"
            ))
        })?;
    files.sort();
    files.dedup();
    if files.is_empty() || files.iter().any(|path| !path.is_file()) {
        return Err(WorkerError::InvalidPayload(
            "Ideogram loader-visible artifact inventory is incomplete".to_owned(),
        ));
    }
    let pins = files
        .iter()
        .map(|path| {
            gen_core::PinnedWeightsFile::pin(path)
                .map_err(|error| crate::classify_engine_error("Ideogram load pin", error))
        })
        .collect::<WorkerResult<Vec<_>>>()?;
    crate::paths::prepare_load_spec_with_file_pins(spec, pins, "Ideogram load preparation")?;
    spec.validate_prepared_file_pins()
        .map_err(|error| crate::classify_engine_error("Ideogram load preparation", error))
}

/// Exact path identity for an artifact inside one immutable Hugging Face snapshot. Optimized memory
/// evidence is artifact-specific: resolving the same filename from `refs/main`, an environment
/// override, or another registered overlay must not inherit the canonical fixture's measurements.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn candle_certified_hf_artifact_path(
    settings: &Settings,
    repo: &str,
    revision: &str,
    relative: &Path,
    actual: &Path,
) -> bool {
    candle_pinned_hf_artifact_path(settings, repo, revision, relative)
        .is_some_and(|expected| candle_artifact_path_matches(actual, &expected))
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn candle_pinned_hf_artifact_path(
    settings: &Settings,
    repo: &str,
    revision: &str,
    relative: &Path,
) -> Option<PathBuf> {
    let root =
        crate::model_jobs::huggingface_pinned_snapshot_dir(&settings.data_dir, repo, revision)?;
    Some(root.join(relative))
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn candle_artifact_path_matches(actual: &Path, expected: &Path) -> bool {
    match (
        std::fs::canonicalize(actual),
        std::fs::canonicalize(expected),
    ) {
        (Ok(actual), Ok(expected)) => actual == expected,
        _ => actual == expected,
    }
}

pub(super) fn load_spec(
    weights_dir: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
    ip_adapter_dir: Option<PathBuf>,
) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(weights_dir));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    if let Some(dir) = ip_adapter_dir {
        spec = spec.with_ip_adapter(WeightsSource::Dir(dir));
    }
    spec
}

/// Validate the API's fresh opaque resolution against this route's inference descriptor and attach
/// the exact prepared source receipt before any planner, fit gate, or provider loader sees the spec.
/// Default/absent selection is an exact no-op.
///
/// The wrapper is intentional typestate: production routes must explicitly consume the attached
/// spec before handing it to a planner or provider. This keeps route wiring compile-enforced while
/// behavior tests exercise the real resolution seam, without source-text assertions.
#[must_use = "a manifest text-encoder attachment must be consumed by the production route"]
#[derive(Debug)]
pub(crate) struct ManifestTextEncoderAttachedSpec(LoadSpec);

impl ManifestTextEncoderAttachedSpec {
    pub(crate) fn into_load_spec(self) -> LoadSpec {
        self.0
    }
}

impl std::ops::Deref for ManifestTextEncoderAttachedSpec {
    type Target = LoadSpec;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for ManifestTextEncoderAttachedSpec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub(crate) fn attach_manifest_text_encoder(
    spec: LoadSpec,
    engine_id: &str,
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<ManifestTextEncoderAttachedSpec> {
    crate::text_encoder_selection::prepare_selected_text_encoder(
        spec,
        engine_id,
        &request.advanced,
        &request.model_manifest_entry,
        settings,
    )
    .map(ManifestTextEncoderAttachedSpec)
}

/// Whether the request carries a non-default authored encoder id. Kept beside the attachment seam
/// so bespoke File routes make the same default/invalid-shape decision as registry routes.
pub(crate) fn has_authored_text_encoder(request: &ImageRequest) -> WorkerResult<bool> {
    match request.advanced.get("textEncoderModel") {
        None => Ok(false),
        Some(Value::String(id)) if id == "default" => Ok(false),
        Some(Value::String(id)) if !id.trim().is_empty() => Ok(true),
        Some(_) => Err(WorkerError::InvalidPayload(
            "advanced.textEncoderModel must be a non-empty string option id returned by GET /api/v1/models"
                .to_owned(),
        )),
    }
}

/// Finalize a complete bespoke File-route spec. A selected encoder's contract export prepares every
/// configured File slot plus its shard/config/tokenizer receipt atomically; the default path keeps
/// the route's existing explicit tokens. Pre-resolved route tokens are compared against the
/// contract-prepared set so a mutation between dispatch resolution and attachment fails closed.
pub(crate) fn prepare_manifest_text_encoder_with_file_pins(
    spec: LoadSpec,
    engine_id: &str,
    request: &ImageRequest,
    settings: &Settings,
    pins: impl IntoIterator<Item = gen_core::PinnedWeightsFile>,
    label: &str,
) -> WorkerResult<LoadSpec> {
    let selected = has_authored_text_encoder(request)?;
    let pins = pins.into_iter().collect::<Vec<_>>();
    let mut attached_spec = attach_manifest_text_encoder(spec, engine_id, request, settings)?;
    if !selected {
        crate::paths::prepare_load_spec_with_file_pins(&mut attached_spec, pins, label)?;
        return Ok(attached_spec.into_load_spec());
    }

    for expected in pins {
        expected
            .ensure_unchanged()
            .map_err(|error| crate::classify_engine_error(label, error))?;
        let received = attached_spec
            .prepared_file_pin_for(expected.loader_path())
            .map_err(|error| crate::classify_engine_error(label, error))?;
        if received != Some(&expected) {
            return Err(WorkerError::InvalidPayload(format!(
                "{label}: source changed while the selected text encoder receipt was prepared: {}",
                expected.loader_path().display()
            )));
        }
    }
    Ok(attached_spec.into_load_spec())
}

/// Select deferred materialization for the native Candle/CUDA Qwen routes. Only the uniform
/// base/edit transformer trunk can be reopened one window at a time: control/IP/PiD and adapter
/// overlays keep the established eager shape and therefore make rung 4 explicitly unavailable.
///
/// Gated to exactly the predicate its call sites live under — `generate_candle_stream`, the
/// `qwen_edit_candle` module, and `candle_qwen_load_shape_tests` all carry
/// `all(not(target_os = "macos"), feature = "backend-candle")`. Note this deliberately does NOT
/// use the `any(test, ...)` spelling of the sibling `apply_request_scoped_candle_residency`: this
/// function's test module is itself macOS-excluded, so admitting bare `test` would leave it unused
/// in a macOS test build and dead-code-error again under `-D warnings`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn apply_declared_candle_image_load_shape(
    engine_id: &str,
    resolved_tier: Option<&str>,
    request_mode: &str,
    manifest: &JsonObject,
    spec: LoadSpec,
    sequential_selected: bool,
) -> LoadSpec {
    let mode = crate::memory_route_registry::MemoryRouteMode::from_request(request_mode);
    let spec = crate::memory_route_registry::evaluate_declared_candle_load_shape(
        engine_id,
        resolved_tier,
        mode,
        manifest,
        spec,
        sequential_selected,
    );
    crate::memory_route_registry::apply_registered_load_shape(
        crate::memory_route_registry::MemoryRouteBackend::Candle,
        engine_id,
        mode.unwrap_or(crate::memory_route_registry::MemoryRouteMode::TextToImage),
        spec,
        sequential_selected,
    )
}

/// Stage a media model's caller-provisioned components (epic 13657, sc-13679) onto its `LoadSpec` —
/// the image/video twin of the audio seam (`audio_jobs.rs run_audio_synthesis_using`). It reads the
/// model's descriptor `required_components` from the media registry and resolves each declared id to
/// its cached `coRequisite` snapshot via [`crate::model_jobs::resolve_co_requisites`] (all-or-nothing:
/// a missing co-requisite fails the job with an actionable error BEFORE the engine load), then stages
/// each in `LoadSpec::components` for the engine's own load-time `require_component` gate.
///
/// Providers with a split artifact layout (Mage-Flow and Candle SDXL at this pin) advertise their
/// component ids here; self-contained providers take the early no-op path without a manifest clone
/// or cache probe.
pub(super) fn attach_required_components(
    spec: LoadSpec,
    model_id: &str,
    manifest_entry: &JsonObject,
    settings: &Settings,
) -> WorkerResult<LoadSpec> {
    let Some(descriptor) = crate::inference_runtime::media_descriptor(model_id) else {
        return Ok(spec);
    };
    if descriptor.required_components.is_empty() {
        return Ok(spec);
    }
    let manifest_value = Value::Object(manifest_entry.clone());
    // Mage-Flow's shared text encoder / VAE are themselves per-tier (sc-14980), so the co-requisite
    // must match the tier actually resolved for the BACKBONE — not the tier the request asked for,
    // which `standard_tier_subdir`'s completeness fallback may have stepped away from. Reading it
    // back off the resolved weights dir keeps the two in lockstep by construction: a q4 request that
    // fell back to q8 gets the q8 text encoder, never a mixed pair.
    let tier = resolved_tier_name(&spec);
    let components = crate::model_jobs::resolve_co_requisites_for_tier(
        &descriptor,
        &manifest_value,
        settings,
        tier,
    )?;
    Ok(components
        .into_iter()
        .fold(spec, |spec, (id, source)| spec.with_component(id, source)))
}

/// Resolve a decoder id through the linked provider descriptor. This is the worker's backend and
/// latent-space gate: the inference registry owns provider eligibility, and its typed compatibility
/// check fails closed for z48 or unknown/learned normalization.
fn selected_decoder_option(
    engine_id: &str,
    decoder_id: &str,
) -> WorkerResult<gen_core::DecoderOption> {
    let descriptor = crate::inference_runtime::media_descriptor(engine_id).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "decoder '{decoder_id}' cannot be selected because image engine '{engine_id}' is not registered"
        ))
    })?;
    descriptor
        .compatible_decoder_options()
        .into_iter()
        .find(|option| option.id == decoder_id)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "decoder '{decoder_id}' is not compatible with image engine '{engine_id}' on backend '{}'",
                descriptor.backend
            ))
        })
}

/// Validate selection before routing or touching model weights. PiD and the alternate decoder both
/// own terminal decode, so accepting both would make the recorded recipe disagree with execution.
pub(super) fn validate_selected_decoder_request(
    engine_id: &str,
    decoder_id: &str,
    advanced: &JsonObject,
) -> WorkerResult<gen_core::DecoderOption> {
    if advanced::flag(advanced, "usePid") {
        return Err(WorkerError::InvalidPayload(
            "advanced.decoder cannot be combined with advanced.usePid; select exactly one decoder"
                .to_owned(),
        ));
    }
    selected_decoder_option(engine_id, decoder_id)
}

/// Stage the selected standalone decoder file without finalizing File identity. Imported routes use
/// this form while assembling their already-confined primary/control/adapter token set atomically.
pub(super) fn attach_selected_decoder_unprepared(
    spec: LoadSpec,
    engine_id: &str,
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<LoadSpec> {
    let Some(decoder_id) = requested_decoder_id(&request.advanced)? else {
        return Ok(spec);
    };
    let option = validate_selected_decoder_request(engine_id, decoder_id, &request.advanced)?;
    let manifest = Value::Object(request.model_manifest_entry.clone());
    let source = crate::model_jobs::resolve_optional_component(
        &manifest,
        option.component_id,
        settings,
    )
    .ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "decoder '{}' needs its standalone pinned component '{}' to be installed; install or repair this model before generating",
            option.label, option.component_id
        ))
    })?;
    Ok(spec.with_component(option.component_id, source))
}

/// Pin the selected decoder through the app-managed model-file trust boundary. The component id is
/// derived from the registered option rather than duplicated in SceneWorks.
pub(super) fn selected_decoder_component_pin(
    spec: &LoadSpec,
    engine_id: &str,
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<gen_core::PinnedWeightsFile>> {
    let Some(decoder_id) = requested_decoder_id(&request.advanced)? else {
        return Ok(None);
    };
    let option = validate_selected_decoder_request(engine_id, decoder_id, &request.advanced)?;
    let source = spec.components.get(option.component_id).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "decoder '{}' component '{}' was not staged",
            option.label, option.component_id
        ))
    })?;
    let WeightsSource::File(path) = source else {
        return Err(WorkerError::InvalidPayload(format!(
            "decoder '{}' component '{}' must resolve to one standalone file",
            option.label, option.component_id
        )));
    };
    crate::paths::pin_app_managed_model_file(
        settings,
        path,
        &format!("{} decoder weights", option.label),
    )
    .map(Some)
}

/// Stage and prepare the selected standalone decoder. Native generation remains runnable without
/// the soft co-requisite; a selected job retains the exact confined donor token and finalizes every
/// configured File identity before cache-key construction.
pub(super) fn attach_selected_decoder(
    spec: LoadSpec,
    engine_id: &str,
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<LoadSpec> {
    let mut spec = attach_selected_decoder_unprepared(spec, engine_id, request, settings)?;
    let Some(decoder_pin) = selected_decoder_component_pin(&spec, engine_id, request, settings)?
    else {
        return Ok(spec);
    };
    let decoder_path = decoder_pin.loader_path().to_path_buf();
    spec.set_prepared_file_pin(decoder_path, decoder_pin)
        .map_err(|error| {
            crate::classify_engine_error("Alternate decoder source preparation failed", error)
        })?;
    spec.prepare_file_sources().map_err(|error| {
        crate::classify_engine_error("Alternate decoder source preparation failed", error)
    })?;
    Ok(spec)
}

/// The tier a [`LoadSpec`]'s weights dir resolved to, for matching a per-tier `coRequisite`'s
/// `variant` (sc-14980).
///
/// Delegates to [`tier_key_from_resolved_dir`] — the established "what tier is this dir" answer, so
/// the co-requisite lookup cannot drift from the tier the backbone actually loads at. `None` for a
/// flat snapshot (whose final segment is a commit sha), which keeps single-row co-requisites — every
/// audio model, and Mage on a legacy flat install — on the tier-agnostic path. A `None` here against
/// a model that DOES declare per-tier rows is refused loudly by `resolve_co_requisites_for_tier`
/// rather than guessed, so an unrecognized layout can never silently pair mismatched weights.
fn resolved_tier_name(spec: &LoadSpec) -> Option<&'static str> {
    let WeightsSource::Dir(dir) = &spec.weights else {
        return None;
    };
    tier_key_from_resolved_dir(dir)
}

/// Resolve SDXL's three caller-staged components (`tokenizer_clip_l` / `tokenizer_clip_bigg` /
/// `vae_fp16_fix`, epic 13657 / sc-13682) for a BESPOKE candle lane whose provider takes explicit
/// component paths (`SdxlEditPaths` / `IpAdapterSdxlPaths` / `InstantIdPaths` — InstantID reuses the
/// candle SDXL conditioner + VAE) rather than a [`LoadSpec`]. Rides the SAME
/// generic [`crate::model_jobs::resolve_co_requisites`] seam the txt2img [`attach_required_components`]
/// path uses: it reads the registered candle `sdxl` descriptor's `required_components` (the exact three
/// ids the edit / IP-Adapter / trainer providers also consume) and maps each to this model's pinned
/// `coRequisite` download by `componentId`. All-or-nothing — a missing component fails the job BEFORE the
/// engine load with the seam's actionable error naming the component id + repo. Candle-only, because the
/// bespoke `SdxlEdit` / `IpAdapterSdxl` providers are (macOS keeps the self-contained MLX SDXL lane).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn resolve_sdxl_components(
    manifest_entry: &JsonObject,
    settings: &Settings,
) -> WorkerResult<(WeightsSource, WeightsSource, WeightsSource)> {
    // The bespoke edit / IP-Adapter providers ARE the candle `sdxl` engine, so resolve that descriptor
    // (id == "sdxl") and let its `required_components` advertisement drive the ids — never hardcoded here.
    let descriptor = crate::inference_runtime::media_descriptor("sdxl").ok_or_else(|| {
        WorkerError::Engine(
            "candle SDXL generator is not registered — cannot resolve its required components"
                .to_owned(),
        )
    })?;
    let manifest_value = Value::Object(manifest_entry.clone());
    let mut components =
        crate::model_jobs::resolve_co_requisites(&descriptor, &manifest_value, settings)?;
    let mut take = |id: &str| -> WorkerResult<WeightsSource> {
        components.remove(id).ok_or_else(|| {
            WorkerError::Engine(format!(
                "the registered candle SDXL descriptor does not advertise its required '{id}' \
                 component — the SceneWorks inference runtime pin is incompatible with this worker"
            ))
        })
    };
    Ok((
        take("tokenizer_clip_l")?,
        take("tokenizer_clip_bigg")?,
        take("vae_fp16_fix")?,
    ))
}

/// Registry-only generator load (epic 3720, sc-3724): resolve `engine_id` through the
/// backend-neutral `crate::inference_runtime::load` seam and return a `Box<dyn gen_core::Generator>`. Optionally
/// installs an IP-Adapter from `ip_adapter_dir` (`LoadSpec::with_ip_adapter`) — the FLUX.1 XLabs
/// IP-Adapter reference path (epic 3621), after which the engine treats a `Conditioning::Reference`
/// as the image prompt. `cfg(target_os)` decides which provider crate registered the engine, not
/// this call.
#[cfg(all(target_os = "macos", test))]
pub(super) fn load_engine(
    engine_id: &str,
    weights_dir: PathBuf,
    quant: Option<Quant>,
    adapters: Vec<AdapterSpec>,
    ip_adapter_dir: Option<PathBuf>,
) -> WorkerResult<Box<dyn Generator>> {
    let spec = load_spec(weights_dir, quant, adapters, ip_adapter_dir);
    load_control_engine(engine_id, &spec)
}

/// Shared real-weight smoke loader: resolve `engine_id` through the backend-neutral
/// `crate::inference_runtime::load` seam and wrap a failure as `WorkerError::Engine`. Every image
/// control/base lane's `#[cfg(test)]` load wrapper funnels through here so the
/// `crate::inference_runtime::load` + `map_err` tail lives in one place (sc-8954). `cfg(target_os)`
/// still decides which provider crate registered the engine, not this call.
#[cfg(all(target_os = "macos", test))]
pub(super) fn load_control_engine(
    engine_id: &str,
    spec: &LoadSpec,
) -> WorkerResult<Box<dyn Generator>> {
    crate::inference_runtime::load(engine_id, spec)
        .map_err(|error| WorkerError::Engine(format!("{engine_id} load failed: {error}")))
}

# [cfg (native_lane)]
# [cfg (test)]
#[path="tests/sdxl_endpoint_inventory_tests.rs"]
mod sdxl_endpoint_inventory_tests;

# [cfg (native_lane)]
# [cfg (test)]
#[path="tests/chroma_load_seal_tests.rs"]
mod chroma_load_seal_tests;

# [cfg (native_lane)]
# [cfg (test)]
#[path="tests/ideogram_load_seal_tests.rs"]
mod ideogram_load_seal_tests;

# [cfg (native_lane)]
# [cfg (all (test , not (target_os = "macos") , feature = "backend-candle"))]
#[path="tests/mage_artifact_certification_tests.rs"]
mod mage_artifact_certification_tests;
