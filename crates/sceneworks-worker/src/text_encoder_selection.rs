//! Descriptor-owned text-encoder discovery and resolution for Image Studio.
//!
//! Public catalog rows contain opaque ids only. The rust-api calls this module again at every job
//! creation boundary and records a server-owned resolution in the worker manifest. The worker then
//! rechecks that resolution against the actual route descriptor, confines it to configured model
//! roots, and exports the inference contract's exact shard/config/tokenizer receipt into the load
//! spec. Candidate discovery is deliberately shallow: the encoder contract, not a SceneWorks file
//! classifier or a recursive walk, owns the direct shard inventory and compatibility decision.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use gen_core::LoadSpec;
use gen_core::WeightsSource;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Digest;

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
use crate::{Settings, WorkerError, WorkerResult};

const DEFAULT_SELECTION_ID: &str = "default";
const SELECTION_ID_PREFIX: &str = "text_encoder_";
const MAX_SHALLOW_ENTRIES: usize = 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageTextEncoderOption {
    pub id: String,
    pub label: String,
    pub description: String,
    pub is_default: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SourceKind {
    File,
    Directory,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ResolvedImageTextEncoder {
    selection_id: String,
    source_kind: SourceKind,
    /// Server-owned resolution metadata. This value is never accepted from the Image Studio
    /// request; the API replaces it from a fresh catalog resolution on create/retry/duplicate.
    path: PathBuf,
}

#[derive(Clone, Debug)]
struct Candidate {
    id: String,
    label: String,
    description: String,
    source: WeightsSource,
}

impl Candidate {
    fn resolution(&self) -> ResolvedImageTextEncoder {
        let (source_kind, path) = match &self.source {
            WeightsSource::File(path) => (SourceKind::File, path.clone()),
            WeightsSource::Dir(path) => (SourceKind::Directory, path.clone()),
        };
        ResolvedImageTextEncoder {
            selection_id: self.id.clone(),
            source_kind,
            path,
        }
    }
}

fn source_path(source: &WeightsSource) -> &Path {
    match source {
        WeightsSource::File(path) | WeightsSource::Dir(path) => path,
    }
}

fn source_kind(source: &WeightsSource) -> SourceKind {
    match source {
        WeightsSource::File(_) => SourceKind::File,
        WeightsSource::Dir(_) => SourceKind::Directory,
    }
}

fn opaque_id(source: &WeightsSource) -> Result<String, std::io::Error> {
    let path = std::path::absolute(source_path(source))?;
    let kind = match source_kind(source) {
        SourceKind::File => "file",
        SourceKind::Directory => "directory",
    };
    let digest = format!(
        "{:x}",
        sha2::Sha256::digest(format!("{kind}\0{}", path.display()))
    );
    Ok(format!("{SELECTION_ID_PREFIX}{}", &digest[..32]))
}

fn display_name(source: &WeightsSource) -> String {
    let path = source_path(source);
    path.file_stem()
        .or_else(|| path.file_name())
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .unwrap_or_else(|| "Installed text encoder".to_owned())
}

fn source_key(source: &WeightsSource) -> Result<(SourceKind, PathBuf), std::io::Error> {
    Ok((
        source_kind(source),
        std::path::absolute(source_path(source))?,
    ))
}

fn push_source(
    sources: &mut BTreeMap<(u8, PathBuf), (WeightsSource, String)>,
    source: WeightsSource,
    scope: &str,
) {
    let Ok((kind, absolute)) = source_key(&source) else {
        return;
    };
    let kind_key = match kind {
        SourceKind::File => 0,
        SourceKind::Directory => 1,
    };
    sources
        .entry((kind_key, absolute))
        .or_insert_with(|| (source, scope.to_owned()));
}

fn is_safetensors(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
}

/// Expand only fixed/direct shapes. There is no recursive traversal: a seed may itself be a File,
/// a component/snapshot Dir, or a model/tier Dir whose direct child or `text_encoder/` child is the
/// selected source. The inference contract decides whether any emitted shape has the required
/// direct shard/config inventory.
fn expand_seed(
    sources: &mut BTreeMap<(u8, PathBuf), (WeightsSource, String)>,
    seed: &Path,
    scope: &str,
) {
    if seed.is_file() {
        if is_safetensors(seed) {
            push_source(sources, WeightsSource::File(seed.to_path_buf()), scope);
        }
        return;
    }
    if !seed.is_dir() {
        return;
    }

    push_source(sources, WeightsSource::Dir(seed.to_path_buf()), scope);
    let text_encoder = seed.join("text_encoder");
    if text_encoder.is_dir() {
        push_source(sources, WeightsSource::Dir(text_encoder), scope);
    }

    let Ok(entries) = std::fs::read_dir(seed) else {
        return;
    };
    let mut children = entries
        .flatten()
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    children.sort();
    children.truncate(MAX_SHALLOW_ENTRIES);
    for child in children {
        if child.is_file() && is_safetensors(&child) {
            push_source(sources, WeightsSource::File(child), scope);
            continue;
        }
        if !child.is_dir() {
            continue;
        }
        push_source(sources, WeightsSource::Dir(child.clone()), scope);
        let nested_component = child.join("text_encoder");
        if nested_component.is_dir() {
            push_source(sources, WeightsSource::Dir(nested_component), scope);
        }
    }
}

fn collect_model_seeds(
    model: &Value,
    sources: &mut BTreeMap<(u8, PathBuf), (WeightsSource, String)>,
) {
    let scope = model
        .get("catalogScope")
        .and_then(Value::as_str)
        .unwrap_or("builtin");
    for key in ["installedPath", "modelPath"] {
        if let Some(path) = model.get(key).and_then(Value::as_str) {
            expand_seed(sources, Path::new(path), scope);
        }
    }
    if let Some(path) = model
        .get("paths")
        .and_then(Value::as_object)
        .and_then(|paths| paths.get("model"))
        .and_then(Value::as_str)
    {
        expand_seed(sources, Path::new(path), scope);
    }
    if let Some(variants) = model.get("variants").and_then(Value::as_array) {
        for variant in variants {
            if let Some(path) = variant.get("installedPath").and_then(Value::as_str) {
                expand_seed(sources, Path::new(path), scope);
            }
        }
    }
    if let Some(components) = model.get("components").and_then(Value::as_array) {
        for component in components {
            if component.get("role").and_then(Value::as_str) != Some("text_encoder") {
                continue;
            }
            if let Some(path) = component.get("path").and_then(Value::as_str) {
                expand_seed(sources, Path::new(path), scope);
            }
        }
    }
}

fn allowed_roots(data_dir: &Path, external_roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut roots = vec![
        data_dir.to_path_buf(),
        sceneworks_core::hf_home::huggingface_hub_cache_dir(data_dir),
    ];
    roots.extend(external_roots.iter().cloned());
    let mut expanded = Vec::new();
    for root in roots {
        if let Ok(absolute) = std::path::absolute(&root) {
            expanded.push(absolute);
        }
        if let Ok(canonical) = std::fs::canonicalize(&root) {
            expanded.push(canonical);
        }
    }
    expanded.sort();
    expanded.dedup();
    expanded
}

fn collect_sources(
    catalog: &[Value],
    data_dir: &Path,
    external_roots: &[PathBuf],
) -> BTreeMap<(u8, PathBuf), (WeightsSource, String)> {
    let mut sources = BTreeMap::new();
    for model in catalog {
        collect_model_seeds(model, &mut sources);
    }
    for (subdir, path) in sceneworks_core::external_roots::comfyui_base_dirs(external_roots) {
        if subdir == "text_encoders" {
            expand_seed(&mut sources, &path, "external");
        }
    }

    let roots = allowed_roots(data_dir, external_roots);
    sources.retain(|(_, path), _| roots.iter().any(|root| path.starts_with(root)));
    sources
}

#[derive(Clone, Debug)]
struct EncoderRoute {
    id: String,
    contract: gen_core::EncoderContract,
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn encoder_route_for_model(model: &Value) -> Option<EncoderRoute> {
    if let Some(id) = model.get("id").and_then(Value::as_str) {
        if let Some(engine_id) = crate::engines::engine_id_for_model(id) {
            if let Some(contract) = crate::inference_runtime::media_encoder_contract(engine_id) {
                return Some(EncoderRoute {
                    id: engine_id.to_owned(),
                    contract,
                });
            }
        }
    }

    let family = model.get("family").and_then(Value::as_str)?;
    let scope = model.get("catalogScope").and_then(Value::as_str);
    let source = match model.get("importSourceShape").and_then(Value::as_str) {
        Some("transformer_file") => gen_core::ImportedModelSource::TransformerFile,
        Some("transformer_directory") => gen_core::ImportedModelSource::TransformerDirectory,
        Some("fused_checkpoint") => gen_core::ImportedModelSource::FusedCheckpoint,
        Some("comfy_ui_tree") => gen_core::ImportedModelSource::ComfyUiTree,
        _ if scope == Some("external") => gen_core::ImportedModelSource::ComfyUiTree,
        _ => return None,
    };
    let descriptor = crate::inference_runtime::imported_model_descriptor(
        family,
        source,
        gen_core::ImportedModelOperation::Generate,
    )?;
    Some(EncoderRoute {
        id: descriptor.id.to_owned(),
        contract: descriptor.encoder_contract?,
    })
}

#[cfg(not(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
)))]
fn encoder_route_for_model(_model: &Value) -> Option<EncoderRoute> {
    None
}

fn candidates_for_model(
    catalog: &[Value],
    model: &Value,
    data_dir: &Path,
    external_roots: &[PathBuf],
) -> Result<(EncoderRoute, Vec<Candidate>), String> {
    let route = encoder_route_for_model(model)
        .ok_or_else(|| "the active runtime has no descriptor for this model".to_owned())?;
    let roots = allowed_roots(data_dir, external_roots);
    let mut candidates = Vec::new();
    for (_, (source, scope)) in collect_sources(catalog, data_dir, external_roots) {
        let Ok(validated) = route.contract.validate_source_for_planning(&source) else {
            continue;
        };
        if validated.ensure_confined_to(&roots).is_err() {
            continue;
        }
        let id = opaque_id(&source).map_err(|error| error.to_string())?;
        let kind = match source_kind(&source) {
            SourceKind::File => "single file",
            SourceKind::Directory => "directory/snapshot",
        };
        candidates.push(Candidate {
            id,
            label: format!("{} ({scope})", display_name(&source)),
            description: format!(
                "Uses a compatible installed {kind}; compatibility is verified by the {engine} engine contract.",
                engine = route.id
            ),
            source,
        });
    }
    candidates.sort_by(|left, right| {
        (left.label.as_str(), left.id.as_str()).cmp(&(right.label.as_str(), right.id.as_str()))
    });
    candidates.dedup_by(|left, right| left.id == right.id);
    Ok((route, candidates))
}

/// Build path-free runtime choices for one catalog model.
pub fn image_text_encoder_options(
    catalog: &[Value],
    model: &Value,
    data_dir: &Path,
    external_roots: &[PathBuf],
) -> Vec<ImageTextEncoderOption> {
    let Ok((route, candidates)) = candidates_for_model(catalog, model, data_dir, external_roots)
    else {
        return Vec::new();
    };
    let mut options = vec![ImageTextEncoderOption {
        id: DEFAULT_SELECTION_ID.to_owned(),
        label: "Model encoder (default)".to_owned(),
        description: format!("Uses the text encoder paired with {}.", route.id),
        is_default: true,
    }];
    options.extend(
        candidates
            .into_iter()
            .map(|candidate| ImageTextEncoderOption {
                id: candidate.id,
                label: candidate.label,
                description: candidate.description,
                is_default: false,
            }),
    );
    options
}

/// Resolve one authored opaque id against a fresh catalog/source snapshot. The returned JSON is
/// server-owned worker metadata; the browser never supplies or observes the resolved path.
pub fn resolve_image_text_encoder_selection(
    catalog: &[Value],
    model: &Value,
    selection_id: &str,
    data_dir: &Path,
    external_roots: &[PathBuf],
) -> Result<Option<Value>, String> {
    if selection_id == DEFAULT_SELECTION_ID {
        return Ok(None);
    }
    let (_, candidates) = candidates_for_model(catalog, model, data_dir, external_roots)?;
    let candidate = candidates
        .into_iter()
        .find(|candidate| candidate.id == selection_id)
        .ok_or_else(|| {
            format!(
                "selected text encoder {selection_id:?} is unavailable; refresh Models and choose an installed option"
            )
        })?;
    serde_json::to_value(candidate.resolution())
        .map(Some)
        .map_err(|error| error.to_string())
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn selected_id(advanced: &serde_json::Map<String, Value>) -> WorkerResult<Option<&str>> {
    match advanced.get("textEncoderModel") {
        None => Ok(None),
        Some(Value::String(id)) if id == DEFAULT_SELECTION_ID => Ok(None),
        Some(Value::String(id)) if !id.trim().is_empty() => Ok(Some(id)),
        Some(_) => Err(WorkerError::InvalidPayload(
            "advanced.textEncoderModel must be a non-empty string option id returned by GET /api/v1/models"
                .to_owned(),
        )),
    }
}

#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
fn base_snapshot_root(spec: &LoadSpec) -> WorkerResult<&Path> {
    if let WeightsSource::Dir(path) = &spec.weights {
        return Ok(path);
    }
    match spec.components.get(gen_core::BASE_SNAPSHOT_COMPONENT) {
        Some(WeightsSource::Dir(path)) => Ok(path),
        _ => Err(WorkerError::InvalidPayload(
            "selected text encoder requires a directory base snapshot for tokenizer binding"
                .to_owned(),
        )),
    }
}

/// Validate and prepare the selected encoder before any memory fit/planner/provider load sees the
/// spec. The same returned spec must be reused for the eventual load.
#[cfg(any(
    target_os = "macos",
    all(not(target_os = "macos"), feature = "backend-candle")
))]
pub(crate) fn prepare_selected_text_encoder(
    mut spec: LoadSpec,
    engine_id: &str,
    advanced: &serde_json::Map<String, Value>,
    manifest_entry: &serde_json::Map<String, Value>,
    settings: &Settings,
) -> WorkerResult<LoadSpec> {
    let Some(selection_id) = selected_id(advanced)? else {
        return Ok(spec);
    };
    let contract =
        crate::inference_runtime::media_encoder_contract(engine_id).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "engine {engine_id:?} does not support selected text encoder {selection_id:?}"
            ))
        })?;
    let resolution = manifest_entry
        .get("resolvedTextEncoder")
        .cloned()
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "selected text encoder {selection_id:?} has no fresh server resolution; retry after refreshing Models"
            ))
        })?;
    let resolution: ResolvedImageTextEncoder =
        serde_json::from_value(resolution).map_err(|_| {
            WorkerError::InvalidPayload(
                "resolved text-encoder metadata is malformed; retry the job".to_owned(),
            )
        })?;
    if resolution.selection_id != selection_id {
        return Err(WorkerError::InvalidPayload(format!(
            "selected text encoder {selection_id:?} does not match fresh server resolution {:?}",
            resolution.selection_id
        )));
    }

    let lexical = crate::paths::normalize_absolute_path(&resolution.path)?;
    if opaque_id(&match resolution.source_kind {
        SourceKind::File => WeightsSource::File(lexical.clone()),
        SourceKind::Directory => WeightsSource::Dir(lexical.clone()),
    })
    .map_err(WorkerError::Io)?
        != selection_id
    {
        return Err(WorkerError::InvalidPayload(
            "resolved text-encoder identity does not match its opaque selection id".to_owned(),
        ));
    }
    let roots = crate::paths::allowed_model_roots(settings)?;
    crate::paths::ensure_path_under(lexical.clone(), &roots, "Selected text encoder")?;
    let canonical = crate::paths::normalize_existing_or_absolute(&lexical)?;
    crate::paths::ensure_path_under(canonical, &roots, "Selected text encoder")?;
    let source = match resolution.source_kind {
        SourceKind::File => WeightsSource::File(lexical),
        SourceKind::Directory => WeightsSource::Dir(lexical),
    };
    let base_root = base_snapshot_root(&spec)?.to_path_buf();
    let validated = contract
        .validate_source_against_base(&source, &base_root)
        .map_err(|error| crate::classify_engine_error("Selected text encoder", error))?;
    validated
        .ensure_confined_to(&roots)
        .map_err(|error| crate::classify_engine_error("Selected text encoder", error))?;
    validated
        .prepare_load_spec(&mut spec)
        .map_err(|error| crate::classify_engine_error("Selected text encoder", error))?;
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collected_source<'a>(
        sources: &'a BTreeMap<(u8, PathBuf), (WeightsSource, String)>,
        kind: SourceKind,
        path: &Path,
    ) -> Option<&'a (WeightsSource, String)> {
        let kind = match kind {
            SourceKind::File => 0,
            SourceKind::Directory => 1,
        };
        sources.get(&(kind, std::path::absolute(path).unwrap()))
    }

    #[test]
    fn opaque_identity_preserves_source_shape_without_exposing_path() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("encoder.safetensors");
        std::fs::write(&file, b"x").unwrap();
        let file_id = opaque_id(&WeightsSource::File(file.clone())).unwrap();
        let dir_id = opaque_id(&WeightsSource::Dir(file.parent().unwrap().to_path_buf())).unwrap();
        assert!(file_id.starts_with(SELECTION_ID_PREFIX));
        assert_ne!(file_id, dir_id);
        assert!(!file_id.contains(dir.path().to_string_lossy().as_ref()));
    }

    #[test]
    fn default_resolution_is_exact_omission() {
        assert_eq!(
            resolve_image_text_encoder_selection(
                &[],
                &serde_json::json!({}),
                DEFAULT_SELECTION_ID,
                Path::new("/tmp"),
                &[],
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn discovery_covers_builtin_imported_external_and_complete_snapshot_shapes() {
        let dir = tempfile::tempdir().unwrap();
        let data_dir = dir.path().join("data");
        let builtin = data_dir.join("builtin");
        let imported = data_dir.join("imported-snapshot");
        let imported_component = imported.join("text_encoder");
        let external_root = dir.path().join("comfyui-models");
        let external = external_root.join("text_encoders/external.safetensors");
        std::fs::create_dir_all(&builtin).unwrap();
        std::fs::create_dir_all(&imported_component).unwrap();
        std::fs::create_dir_all(external.parent().unwrap()).unwrap();
        std::fs::write(builtin.join("builtin.safetensors"), b"builtin").unwrap();
        std::fs::write(imported_component.join("model.safetensors"), b"imported").unwrap();
        std::fs::write(&external, b"external").unwrap();

        let catalog = vec![
            serde_json::json!({
                "id": "builtin-row",
                "catalogScope": "builtin",
                "installedPath": builtin.clone()
            }),
            serde_json::json!({
                "id": "imported-row",
                "catalogScope": "imported",
                "installedPath": imported.clone()
            }),
        ];
        let sources = collect_sources(&catalog, &data_dir, std::slice::from_ref(&external_root));

        assert_eq!(
            collected_source(
                &sources,
                SourceKind::File,
                &data_dir.join("builtin/builtin.safetensors")
            )
            .unwrap()
            .1,
            "builtin"
        );
        assert_eq!(
            collected_source(&sources, SourceKind::Directory, &imported)
                .unwrap()
                .1,
            "imported",
            "the complete snapshot root remains a selectable Dir shape"
        );
        assert_eq!(
            collected_source(&sources, SourceKind::Directory, &imported_component)
                .unwrap()
                .1,
            "imported",
            "the direct text_encoder component remains a selectable Dir shape"
        );
        assert_eq!(
            collected_source(&sources, SourceKind::File, &external)
                .unwrap()
                .1,
            "external"
        );
    }

    #[test]
    fn seed_expansion_is_direct_only_and_never_recursively_scans() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("root");
        let direct = root.join("direct");
        let too_deep = direct.join("nested");
        std::fs::create_dir_all(&too_deep).unwrap();
        let direct_file = root.join("direct.safetensors");
        let deep_file = too_deep.join("deep.safetensors");
        std::fs::write(&direct_file, b"direct").unwrap();
        std::fs::write(&deep_file, b"deep").unwrap();

        let mut sources = BTreeMap::new();
        expand_seed(&mut sources, &root, "builtin");

        assert!(collected_source(&sources, SourceKind::Directory, &root).is_some());
        assert!(collected_source(&sources, SourceKind::Directory, &direct).is_some());
        assert!(collected_source(&sources, SourceKind::File, &direct_file).is_some());
        assert!(
            collected_source(&sources, SourceKind::Directory, &too_deep).is_none(),
            "a grandchild directory must not be discovered"
        );
        assert!(
            collected_source(&sources, SourceKind::File, &deep_file).is_none(),
            "a grandchild shard must not be discovered"
        );
    }

    /// Cross-layer receipt regression for the bespoke Qwen control route. This starts at the same
    /// opaque server resolution the API persists, runs the worker's production attachment seam,
    /// enters the warm-cache identity gate, and finally exercises the provider callback bracket.
    /// The paired inference suite continues through `QwenFunControl::load_with_spec`; keeping this
    /// half here prevents SceneWorks from accidentally stripping or rebuilding that receipt before
    /// the provider sees it.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn worker_preparation_and_cache_reject_file_dir_snapshot_mutations_for_bespoke_qwen_control() {
        use std::cell::Cell;

        const ENGINE_ID: &str = "qwen_image_control";
        for shape in ["file", "dir", "snapshot"] {
            let fixture = tempfile::tempdir().unwrap();
            let base = fixture.path().join("base");
            let selected = fixture.path().join("selected");
            let control = fixture.path().join("control.safetensors");
            std::fs::create_dir_all(&base).unwrap();
            std::fs::write(&control, b"synthetic control overlay").unwrap();

            let contract = crate::inference_runtime::media_encoder_contract(ENGINE_ID)
                .expect("the composed Qwen control route owns an encoder contract");
            gen_core_testkit::write_encoder_contract_tokenizer_fixture(&base, contract).unwrap();
            // The testkit's fixture ends with the encoder's dense multi-gigabyte `set_len`, which
            // NTFS allocates in full — release the never-written tail after each write. Logical
            // size and written bytes are unchanged; see `crate::test_fixture_disk`.
            let source = match shape {
                "file" => {
                    gen_core_testkit::write_encoder_contract_fixture(&selected, contract).unwrap();
                    crate::test_fixture_disk::sparsify_written_safetensors(&selected);
                    WeightsSource::File(selected.join("model.safetensors"))
                }
                "dir" => {
                    gen_core_testkit::write_encoder_contract_fixture(&selected, contract).unwrap();
                    crate::test_fixture_disk::sparsify_written_safetensors(&selected);
                    WeightsSource::Dir(selected.clone())
                }
                "snapshot" => {
                    gen_core_testkit::write_encoder_contract_fixture(
                        &selected.join("text_encoder"),
                        contract,
                    )
                    .unwrap();
                    crate::test_fixture_disk::sparsify_written_safetensors(
                        &selected.join("text_encoder"),
                    );
                    gen_core_testkit::write_encoder_contract_tokenizer_fixture(&selected, contract)
                        .unwrap();
                    WeightsSource::Dir(selected.clone())
                }
                _ => unreachable!(),
            };

            let model = serde_json::json!({
                "id": "qwen_image",
                "installedPath": selected,
            });
            let catalog = vec![model.clone()];
            let selection_id = opaque_id(&source).unwrap();
            let resolution = resolve_image_text_encoder_selection(
                &catalog,
                &model,
                &selection_id,
                fixture.path(),
                &[],
            )
            .unwrap()
            .expect("the compatible opaque selection resolves fresh");
            let advanced = serde_json::json!({ "textEncoderModel": selection_id })
                .as_object()
                .unwrap()
                .clone();
            let mut manifest_entry = model.as_object().unwrap().clone();
            manifest_entry.insert("resolvedTextEncoder".to_owned(), resolution);
            let mut settings = Settings::from_env();
            settings.data_dir = fixture.path().to_path_buf();
            settings.external_model_roots.clear();

            let spec =
                LoadSpec::new(WeightsSource::Dir(base)).with_control(WeightsSource::File(control));
            let prepared = prepare_selected_text_encoder(
                spec,
                ENGINE_ID,
                &advanced,
                &manifest_entry,
                &settings,
            )
            .unwrap();
            crate::generator_cache::LoadIdentity::try_from_load_spec(ENGINE_ID, &prepared)
                .expect("the exact worker-prepared receipt enters warm-cache identity");

            match shape {
                "file" => std::fs::write(selected.join("config.json"), b"{}").unwrap(),
                "dir" => {
                    // Inventory validation runs before shard parsing. Keep this addition tiny: the
                    // sparse production-dimension fixture has a large logical length and must never
                    // be materialized merely to prove that a new direct shard is rejected.
                    std::fs::write(selected.join("added.safetensors"), b"added shard sentinel")
                        .unwrap();
                }
                "snapshot" => {
                    let tokenizer = contract.tokenizer.artifact_candidates[0];
                    std::fs::write(selected.join(tokenizer), b"{}").unwrap();
                }
                _ => unreachable!(),
            }

            let cache_error =
                crate::generator_cache::LoadIdentity::try_from_load_spec(ENGINE_ID, &prepared)
                    .expect_err("a mutated encoder receipt must miss before warm-cache lookup")
                    .to_string();
            assert!(
                cache_error.contains("receipt changed") || cache_error.contains("pinned weights"),
                "{shape}: {cache_error}"
            );

            let provider_opened = Cell::new(false);
            let provider_error = prepared
                .read_prepared_files_unchanged::<(), gen_core::Error>(|| {
                    provider_opened.set(true);
                    Ok(())
                })
                .expect_err("a mutated encoder receipt must fail before provider open")
                .to_string();
            assert!(!provider_opened.get(), "{shape} reached provider open");
            assert!(
                provider_error.contains("receipt changed")
                    || provider_error.contains("pinned weights"),
                "{shape}: {provider_error}"
            );
        }
    }
}
