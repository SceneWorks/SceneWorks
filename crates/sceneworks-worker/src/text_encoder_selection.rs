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

#[cfg(native_lane)]
use gen_core::LoadSpec;
use gen_core::WeightsSource;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::Digest;

#[cfg(native_lane)]
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

fn source_kind(source: &WeightsSource) -> SourceKind {
    match source {
        WeightsSource::File(_) => SourceKind::File,
        WeightsSource::Dir(_) => SourceKind::Directory,
    }
}

pub(crate) fn opaque_id(source: &WeightsSource) -> Result<String, std::io::Error> {
    let path = std::path::absolute(crate::weights_source::WeightsSourceExt::path(source))?;
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
    let path = crate::weights_source::WeightsSourceExt::path(source);
    path.file_stem()
        .or_else(|| path.file_name())
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .unwrap_or_else(|| "Installed text encoder".to_owned())
}

fn source_key(source: &WeightsSource) -> Result<(SourceKind, PathBuf), std::io::Error> {
    Ok((
        source_kind(source),
        crate::paths::normalize_absolute_path(crate::weights_source::WeightsSourceExt::path(
            source,
        ))
        .map_err(std::io::Error::other)?,
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
    // One followed stat answers both `is_file` and `is_dir`.
    let Ok(metadata) = std::fs::metadata(seed) else {
        return;
    };
    if metadata.is_file() {
        if is_safetensors(seed) {
            push_source(sources, WeightsSource::File(seed.to_path_buf()), scope);
        }
        return;
    }
    if !metadata.is_dir() {
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
        .map(|entry| (entry.path(), entry.file_type().ok()))
        .collect::<Vec<_>>();
    children.sort_by(|left, right| left.0.cmp(&right.0));
    children.truncate(MAX_SHALLOW_ENTRIES);
    for (child, file_type) in children {
        // The enumeration already carries each entry's type; only a link (or an entry whose type
        // could not be read) needs the followed stat `is_file`/`is_dir` would have made.
        let (is_file, is_dir) = match file_type {
            Some(file_type) if !file_type.is_symlink() => (file_type.is_file(), file_type.is_dir()),
            _ => std::fs::metadata(&child)
                .map(|metadata| (metadata.is_file(), metadata.is_dir()))
                .unwrap_or((false, false)),
        };
        if is_file && is_safetensors(&child) {
            push_source(sources, WeightsSource::File(child), scope);
            continue;
        }
        if !is_dir {
            continue;
        }
        push_source(sources, WeightsSource::Dir(child.clone()), scope);
        let nested_component = child.join("text_encoder");
        if nested_component.is_dir() {
            push_source(sources, WeightsSource::Dir(nested_component), scope);
        }
    }
}

/// [`expand_seed`] at most once per seed path. Rows routinely share an install path (a model and
/// its variants, a tier family); a repeat expansion of the same path in the same pass could only
/// re-offer keys that are already present, which `push_source` keeps with their first scope.
fn expand_seed_once(
    sources: &mut SourceMap,
    expanded: &mut std::collections::HashSet<PathBuf>,
    seed: &Path,
    scope: &str,
) {
    if expanded.insert(seed.to_path_buf()) {
        expand_seed(sources, seed, scope);
    }
}

fn collect_model_seeds(
    model: &Value,
    sources: &mut BTreeMap<(u8, PathBuf), (WeightsSource, String)>,
    expanded: &mut std::collections::HashSet<PathBuf>,
) {
    let scope = model
        .get("catalogScope")
        .and_then(Value::as_str)
        .unwrap_or("builtin");
    for key in ["installedPath", "modelPath"] {
        if let Some(path) = model.get(key).and_then(Value::as_str) {
            expand_seed_once(sources, expanded, Path::new(path), scope);
        }
    }
    if let Some(path) = model
        .get("paths")
        .and_then(Value::as_object)
        .and_then(|paths| paths.get("model"))
        .and_then(Value::as_str)
    {
        expand_seed_once(sources, expanded, Path::new(path), scope);
    }
    if let Some(variants) = model.get("variants").and_then(Value::as_array) {
        for variant in variants {
            if let Some(path) = variant.get("installedPath").and_then(Value::as_str) {
                expand_seed_once(sources, expanded, Path::new(path), scope);
            }
        }
    }
    if let Some(components) = model.get("components").and_then(Value::as_array) {
        for component in components {
            if component.get("role").and_then(Value::as_str) != Some("text_encoder") {
                continue;
            }
            if let Some(path) = component.get("path").and_then(Value::as_str) {
                expand_seed_once(sources, expanded, Path::new(path), scope);
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
        if let Ok(absolute) = crate::paths::normalize_absolute_path(&root) {
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

type SourceMap = BTreeMap<(u8, PathBuf), (WeightsSource, String)>;

fn collect_sources(catalog: &[Value], data_dir: &Path, external_roots: &[PathBuf]) -> SourceMap {
    collect_sources_within(
        catalog,
        external_roots,
        &allowed_roots(data_dir, external_roots),
    )
}

/// [`collect_sources`] against roots the caller already computed, so one catalog pass can share a
/// single root canonicalization with the discovery validation that follows it.
fn collect_sources_within(
    catalog: &[Value],
    external_roots: &[PathBuf],
    roots: &[PathBuf],
) -> SourceMap {
    let mut sources = BTreeMap::new();
    let mut expanded = std::collections::HashSet::new();
    for model in catalog {
        collect_model_seeds(model, &mut sources, &mut expanded);
    }
    for (subdir, path) in sceneworks_core::external_roots::comfyui_base_dirs(external_roots) {
        if subdir == "text_encoders" {
            expand_seed_once(&mut sources, &mut expanded, &path, "external");
        }
    }

    sources.retain(|(_, path), _| roots.iter().any(|root| path.starts_with(root)));
    sources
}

#[derive(Clone, Debug)]
struct EncoderRoute {
    id: String,
    contract: gen_core::EncoderContract,
}

#[cfg(native_lane)]
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
        contract: crate::inference_runtime::media_encoder_contract(descriptor.id)?,
    })
}

#[cfg(not(native_lane))]
fn encoder_route_for_model(_model: &Value) -> Option<EncoderRoute> {
    None
}

/// The production discovery predicate: the encoder contract, not SceneWorks, decides whether a
/// source is a compatible alternate.
fn validate_for_discovery(
    contract: &gen_core::EncoderContract,
    source: &WeightsSource,
    roots: &[PathBuf],
) -> bool {
    contract
        .validate_source_for_discovery(source, roots)
        .is_ok()
}

fn candidates_for_model(
    catalog: &[Value],
    model: &Value,
    data_dir: &Path,
    external_roots: &[PathBuf],
) -> Result<(EncoderRoute, Vec<Candidate>), String> {
    candidates_for_model_with(
        catalog,
        model,
        data_dir,
        external_roots,
        encoder_route_for_model,
        validate_for_discovery,
    )
}

/// The per-model resolution: collect every catalog source and validate each against this model's
/// contract. Job creation uses it uncached; it is also the reference the batched listing is tested
/// against.
fn candidates_for_model_with(
    catalog: &[Value],
    model: &Value,
    data_dir: &Path,
    external_roots: &[PathBuf],
    route_for: impl Fn(&Value) -> Option<EncoderRoute>,
    validate: impl Fn(&gen_core::EncoderContract, &WeightsSource, &[PathBuf]) -> bool,
) -> Result<(EncoderRoute, Vec<Candidate>), String> {
    let route = route_for(model)
        .ok_or_else(|| "the active runtime has no descriptor for this model".to_owned())?;
    let roots = allowed_roots(data_dir, external_roots);
    let sources = collect_sources(catalog, data_dir, external_roots);
    let candidates = candidates_from_accepted(
        &route,
        sources
            .values()
            .filter(|(source, _)| validate(&route.contract, source, &roots)),
    )?;
    Ok((route, candidates))
}

/// Build one route's ordered, de-duplicated candidate list from the sources its contract accepted,
/// in [`SourceMap`] order. Shared by the per-model resolver and the whole-catalog listing so both
/// produce the same ids, labels, descriptions and order.
fn candidates_from_accepted<'a>(
    route: &EncoderRoute,
    accepted: impl Iterator<Item = &'a (WeightsSource, String)>,
) -> Result<Vec<Candidate>, String> {
    let mut candidates = Vec::new();
    for (source, scope) in accepted {
        let id = opaque_id(source).map_err(|error| error.to_string())?;
        let kind = match source_kind(source) {
            SourceKind::File => "single file",
            SourceKind::Directory => "directory/snapshot",
        };
        candidates.push(Candidate {
            id,
            label: format!("{} ({scope})", display_name(source)),
            description: format!(
                "Uses a compatible installed {kind}; compatibility is verified by the {engine} engine contract.",
                engine = route.id
            ),
            source: source.clone(),
        });
    }
    candidates.sort_by(|left, right| {
        (left.label.as_str(), left.id.as_str()).cmp(&(right.label.as_str(), right.id.as_str()))
    });
    candidates.dedup_by(|left, right| left.id == right.id);
    Ok(candidates)
}

fn options_from_candidates(
    route: &EncoderRoute,
    candidates: Vec<Candidate>,
) -> Vec<ImageTextEncoderOption> {
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

/// Upper bound on memoized discovery sources. The memo is pruned to the latest listing's sources on
/// every call, so this only guards against a pathological catalog; past it, the overflow is
/// validated uncached instead of growing the memo.
const MAX_DISCOVERY_MEMO_SOURCES: usize = 8192;

type ModifiedAt = Option<std::time::SystemTime>;

/// Cheap observation of one file system entry: its own link metadata when it is a symlink or
/// junction, plus the followed (is_file, is_dir, length, modification time).
#[derive(Clone, Debug, PartialEq, Eq)]
struct EntryStamp {
    link: Option<(u64, ModifiedAt)>,
    followed: Option<(bool, bool, u64, ModifiedAt)>,
}

fn entry_stamp(path: &Path) -> Option<EntryStamp> {
    let own = std::fs::symlink_metadata(path).ok()?;
    let link = own
        .file_type()
        .is_symlink()
        .then(|| (own.len(), own.modified().ok()));
    let followed = if link.is_some() {
        std::fs::metadata(path).ok()
    } else {
        Some(own)
    };
    Some(EntryStamp {
        link,
        followed: followed.map(|metadata| {
            (
                metadata.is_file(),
                metadata.is_dir(),
                metadata.len(),
                metadata.modified().ok(),
            )
        }),
    })
}

fn stamp_is_file(stamp: Option<&EntryStamp>) -> bool {
    stamp
        .and_then(|stamp| stamp.followed)
        .is_some_and(|followed| followed.0)
}

fn stamp_is_dir(stamp: Option<&EntryStamp>) -> bool {
    stamp
        .and_then(|stamp| stamp.followed)
        .is_some_and(|followed| followed.1)
}

/// Stat-only fingerprint of everything bounded discovery reads for one source: the source's
/// canonical target, the source entry, the `text_encoder/` component and each `config.json`
/// candidate, and every direct `.safetensors` entry of the directory discovery would inventory.
/// Discovery reads only those files (config bytes and shard headers) and canonicalizes only those
/// paths, so an unchanged fingerprint means an unchanged verdict for a fixed contract and root set,
/// while a staged, rewritten, resized, retargeted or removed shard or config changes it.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SourceFingerprint {
    canonical: Option<PathBuf>,
    entries: Vec<Option<EntryStamp>>,
    inventory: Option<Vec<(std::ffi::OsString, Option<EntryStamp>)>>,
}

fn source_fingerprint(source: &WeightsSource) -> SourceFingerprint {
    let path = crate::weights_source::WeightsSourceExt::path(source);
    let canonical = std::fs::canonicalize(path).ok();
    match source {
        WeightsSource::File(path) => SourceFingerprint {
            canonical,
            entries: vec![
                entry_stamp(path),
                path.parent()
                    .and_then(|parent| entry_stamp(&parent.join("config.json"))),
            ],
            inventory: None,
        },
        WeightsSource::Dir(path) => {
            let own = entry_stamp(path);
            let nested = path.join("text_encoder");
            let nested_stamp = entry_stamp(&nested);
            let nested_config = if stamp_is_dir(nested_stamp.as_ref()) {
                entry_stamp(&nested.join("config.json"))
            } else {
                None
            };
            let direct_config = entry_stamp(&path.join("config.json"));
            let weights =
                if stamp_is_dir(nested_stamp.as_ref()) && stamp_is_file(nested_config.as_ref()) {
                    nested
                } else {
                    path.clone()
                };
            let inventory = std::fs::read_dir(&weights).ok().map(|entries| {
                let mut shards = entries
                    .flatten()
                    .map(|entry| entry.path())
                    .filter(|candidate| is_safetensors(candidate))
                    .map(|candidate| {
                        let stamp = entry_stamp(&candidate);
                        (candidate.file_name().unwrap_or_default().to_owned(), stamp)
                    })
                    .collect::<Vec<_>>();
                shards.sort_by(|left, right| left.0.cmp(&right.0));
                shards
            });
            SourceFingerprint {
                canonical,
                entries: vec![own, nested_stamp, nested_config, direct_config],
                inventory,
            }
        }
    }
}

struct DiscoveryMemoEntry {
    fingerprint: SourceFingerprint,
    verdicts: Vec<(gen_core::EncoderContract, bool)>,
}

#[derive(Default)]
struct DiscoveryMemoState {
    roots: Vec<PathBuf>,
    entries: std::collections::HashMap<(u8, PathBuf), DiscoveryMemoEntry>,
}

/// Process-wide memo of bounded discovery verdicts for the catalog listing.
///
/// Discovery of one source canonicalizes its paths and reads its config and every shard header;
/// across a real library that is seconds of I/O, almost all of it repeated identically on every
/// catalog request. A verdict is keyed by its source, the authorized roots and the encoder contract,
/// and is reused only while the source's [`SourceFingerprint`] is unchanged, so a warm listing only
/// stats files and a newly staged, changed or removed encoder shows up on the next request without
/// a restart.
///
/// The memo serves the listing only. Job creation still resolves a selected id through the
/// uncached per-model path, and the worker re-validates the full sealed source before loading.
#[derive(Default)]
struct DiscoveryMemo {
    state: std::sync::Mutex<DiscoveryMemoState>,
}

impl DiscoveryMemo {
    /// Discovery verdicts for every source (in [`SourceMap`] order) against every contract (in
    /// `contracts` order). Memo entries for sources absent from `sources` are dropped.
    fn verdicts(
        &self,
        sources: &SourceMap,
        roots: &[PathBuf],
        contracts: &[gen_core::EncoderContract],
        validate: impl Fn(&gen_core::EncoderContract, &WeightsSource, &[PathBuf]) -> bool,
    ) -> Vec<Vec<bool>> {
        // Held for the whole pass: concurrent listings then join one warm memo instead of running
        // the same cold inspection side by side.
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.roots != roots {
            state.entries.clear();
            state.roots = roots.to_vec();
        }
        let mut result = Vec::with_capacity(sources.len());
        for (key, (source, _)) in sources {
            // Taken before any inspection, so a change racing this pass misses on the next one.
            let fingerprint = source_fingerprint(source);
            if state
                .entries
                .get(key)
                .is_some_and(|entry| entry.fingerprint != fingerprint)
            {
                state.entries.remove(key);
            }
            if !state.entries.contains_key(key) && state.entries.len() >= MAX_DISCOVERY_MEMO_SOURCES
            {
                result.push(
                    contracts
                        .iter()
                        .map(|contract| validate(contract, source, roots))
                        .collect(),
                );
                continue;
            }
            let entry = state
                .entries
                .entry(key.clone())
                .or_insert_with(|| DiscoveryMemoEntry {
                    fingerprint,
                    verdicts: Vec::new(),
                });
            let row = contracts
                .iter()
                .map(|contract| {
                    if let Some((_, verdict)) =
                        entry.verdicts.iter().find(|(known, _)| known == contract)
                    {
                        return *verdict;
                    }
                    let verdict = validate(contract, source, roots);
                    entry.verdicts.push((*contract, verdict));
                    verdict
                })
                .collect();
            result.push(row);
        }
        state.entries.retain(|key, _| sources.contains_key(key));
        result
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entries
            .len()
    }
}

fn discovery_memo() -> &'static DiscoveryMemo {
    static MEMO: std::sync::OnceLock<DiscoveryMemo> = std::sync::OnceLock::new();
    MEMO.get_or_init(DiscoveryMemo::default)
}

/// Build path-free runtime choices for every model of one catalog in a single pass.
///
/// Element `i` equals [`image_text_encoder_options`]`(catalog, &catalog[i], ..)`: the same options in
/// the same order with the same labels. Only the cost differs. Source collection and root
/// canonicalization run once per call instead of once per model, each source is inspected once per
/// distinct encoder contract instead of once per routed model, and those verdicts are memoized
/// across calls behind a stat-only fingerprint. This is blocking file-system work, so async callers
/// must run it off the runtime's worker threads.
pub fn image_text_encoder_options_for_catalog(
    catalog: &[Value],
    data_dir: &Path,
    external_roots: &[PathBuf],
) -> Vec<Vec<ImageTextEncoderOption>> {
    options_for_catalog_with(
        catalog,
        data_dir,
        external_roots,
        discovery_memo(),
        encoder_route_for_model,
        validate_for_discovery,
    )
}

fn options_for_catalog_with(
    catalog: &[Value],
    data_dir: &Path,
    external_roots: &[PathBuf],
    memo: &DiscoveryMemo,
    route_for: impl Fn(&Value) -> Option<EncoderRoute>,
    validate: impl Fn(&gen_core::EncoderContract, &WeightsSource, &[PathBuf]) -> bool,
) -> Vec<Vec<ImageTextEncoderOption>> {
    let routes = catalog.iter().map(&route_for).collect::<Vec<_>>();
    if routes.iter().all(Option::is_none) {
        return vec![Vec::new(); catalog.len()];
    }
    let roots = allowed_roots(data_dir, external_roots);
    let sources = collect_sources_within(catalog, external_roots, &roots);
    let mut contracts = Vec::<gen_core::EncoderContract>::new();
    for route in routes.iter().flatten() {
        if !contracts.contains(&route.contract) {
            contracts.push(route.contract);
        }
    }
    let verdicts = memo.verdicts(&sources, &roots, &contracts, validate);
    routes
        .iter()
        .map(|route| {
            let Some(route) = route else {
                return Vec::new();
            };
            let Some(contract_index) = contracts
                .iter()
                .position(|contract| *contract == route.contract)
            else {
                return Vec::new();
            };
            let accepted = sources
                .values()
                .zip(&verdicts)
                .filter(|(_, row)| row[contract_index])
                .map(|(source, _)| source);
            match candidates_from_accepted(route, accepted) {
                Ok(candidates) => options_from_candidates(route, candidates),
                Err(_) => Vec::new(),
            }
        })
        .collect()
}

#[cfg(native_lane)]
fn portable_builtin_seed(model: &Value, installed: Option<&Path>) -> Value {
    let mut row = serde_json::Map::new();
    for key in ["id", "family", "importSourceShape", "catalogScope"] {
        if let Some(value) = model.get(key) {
            row.insert(key.to_owned(), value.clone());
        }
    }
    if let Some(installed) = installed {
        row.insert(
            "installedPath".to_owned(),
            Value::String(installed.to_string_lossy().into_owned()),
        );
    }
    Value::Object(row)
}

/// Enumerate only descriptor-compatible registered sources for portable worker advertisements.
/// The API's opaque option id is based on its own absolute path, so a worker advertises the
/// exact source ledger instead of trying to reuse that host-local id.
#[cfg(native_lane)]
pub(crate) fn portable_registered_source_paths(
    settings: &Settings,
) -> Vec<(String, PathBuf, bool)> {
    let mut catalog = Vec::new();
    // Live external_base_* rows are assembled only by the API scan and have host-local IDs.
    // Enroll the corresponding imported descriptor families in the worker's source inventory.
    for registration in crate::inference_runtime::media().imported_models() {
        if registration.source == gen_core::ImportedModelSource::ComfyUiTree
            && registration.operation == gen_core::ImportedModelOperation::Generate
        {
            catalog.push(serde_json::json!({
                "id": format!("external-comfyui:{}", registration.family),
                "family": registration.family,
                "catalogScope": "external",
                "importSourceShape": "comfy_ui_tree",
            }));
        }
    }
    let library = sceneworks_core::hf_home::model_source_library(&settings.data_dir);
    let user_manifest = settings.config_dir.join("manifests/user.models.jsonc");
    if let Ok(raw) = std::fs::read_to_string(user_manifest) {
        // Shared normalizer: a future-version user manifest contributes no encoder sources.
        if let Ok(value) = sceneworks_core::model_manifest_version::parse_user_model_manifest(&raw)
        {
            for model in value
                .get("models")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                catalog.push(model.clone());
                for (_, raw) in crate::portable_source_claims::registered_model_locators(model) {
                    if let Ok(path) = crate::paths::normalize_app_managed_model_entry_path(
                        settings,
                        &raw,
                        "registered text encoder source",
                    ) {
                        catalog.push(portable_builtin_seed(model, Some(&path)));
                    }
                }
                for variant in model
                    .get("variants")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if let Some(raw) = variant.get("installedPath").and_then(Value::as_str) {
                        if let Ok(path) = crate::paths::normalize_app_managed_model_entry_path(
                            settings,
                            Path::new(raw),
                            "registered text encoder variant",
                        ) {
                            catalog.push(portable_builtin_seed(model, Some(&path)));
                        }
                    }
                }
                for download in model
                    .get("downloads")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let (Some(repo), Some(revision)) = (
                        download.get("repo").and_then(Value::as_str),
                        download.get("revision").and_then(Value::as_str),
                    ) else {
                        continue;
                    };
                    if let Ok((_, snapshot)) = library.discover_snapshot(repo, Some(revision)) {
                        let variant = download
                            .get("variant")
                            .and_then(Value::as_str)
                            .unwrap_or("default");
                        let selected = if snapshot.join(variant).is_dir() {
                            snapshot.join(variant)
                        } else {
                            snapshot
                        };
                        catalog.push(portable_builtin_seed(model, Some(&selected)));
                    }
                }
            }
        }
    }
    if let Ok(manifest) =
        sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
    {
        for model in manifest
            .get("models")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(id) = model.get("id").and_then(Value::as_str) else {
                continue;
            };
            catalog.push(portable_builtin_seed(model, None));
            for root in ["mlx", "candle"] {
                let installed = settings.data_dir.join("models").join(root).join(id);
                if installed.exists() {
                    catalog.push(portable_builtin_seed(model, Some(&installed)));
                }
            }
            for download in model
                .get("downloads")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let (Some(repo), Some(revision)) = (
                    download.get("repo").and_then(Value::as_str),
                    download.get("revision").and_then(Value::as_str),
                ) else {
                    continue;
                };
                let Ok((_, snapshot)) = library.discover_snapshot(repo, Some(revision)) else {
                    continue;
                };
                let variant = download
                    .get("variant")
                    .and_then(Value::as_str)
                    .unwrap_or("default");
                let selected = if snapshot.join(variant).is_dir() {
                    snapshot.join(variant)
                } else {
                    snapshot
                };
                catalog.push(portable_builtin_seed(model, Some(&selected)));
            }
        }
    }
    let roots = allowed_roots(&settings.data_dir, &settings.external_model_roots);
    let routes = catalog
        .iter()
        .filter_map(|model| {
            Some((
                model.get("id")?.as_str()?.to_owned(),
                encoder_route_for_model(model)?,
            ))
        })
        .collect::<BTreeMap<_, _>>();
    let mut result = Vec::new();
    for (_, (source, _)) in
        collect_sources(&catalog, &settings.data_dir, &settings.external_model_roots)
    {
        for (model_id, route) in &routes {
            if route
                .contract
                .validate_source_for_discovery(&source, &roots)
                .is_ok()
            {
                result.push((
                    model_id.clone(),
                    crate::weights_source::WeightsSourceExt::path(&source).to_path_buf(),
                    matches!(source, WeightsSource::File(_)),
                ));
            }
        }
    }
    result.sort();
    result.dedup();
    result
}

#[cfg(not(native_lane))]
pub(crate) fn portable_registered_source_paths(
    _settings: &crate::Settings,
) -> Vec<(String, PathBuf, bool)> {
    Vec::new()
}

/// Build path-free runtime choices for one catalog model.
pub fn image_text_encoder_options(
    catalog: &[Value],
    model: &Value,
    data_dir: &Path,
    external_roots: &[PathBuf],
) -> Vec<ImageTextEncoderOption> {
    match candidates_for_model(catalog, model, data_dir, external_roots) {
        Ok((route, candidates)) => options_from_candidates(&route, candidates),
        Err(_) => Vec::new(),
    }
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

#[cfg(all(native_lane, test))]
fn selected_id(advanced: &serde_json::Map<String, Value>) -> WorkerResult<Option<&str>> {
    match advanced.get("textEncoderModel") {
        None => selected_id_value(None),
        Some(Value::String(id)) => selected_id_value(Some(id)),
        Some(_) => Err(WorkerError::InvalidPayload(
            "advanced.textEncoderModel must be a non-empty string option id returned by GET /api/v1/models"
                .to_owned(),
        )),
    }
}

#[cfg(native_lane)]
fn selected_id_value(selection_id: Option<&str>) -> WorkerResult<Option<&str>> {
    match selection_id {
        None | Some(DEFAULT_SELECTION_ID) => Ok(None),
        Some(id) if !id.trim().is_empty() => Ok(Some(id)),
        Some(_) => Err(WorkerError::InvalidPayload(
            "advanced.textEncoderModel must be a non-empty string option id returned by GET /api/v1/models"
                .to_owned(),
        )),
    }
}

#[cfg(native_lane)]
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
#[cfg(all(native_lane, test))]
pub(crate) fn prepare_selected_text_encoder(
    spec: LoadSpec,
    engine_id: &str,
    advanced: &serde_json::Map<String, Value>,
    manifest_entry: &serde_json::Map<String, Value>,
    settings: &Settings,
) -> WorkerResult<LoadSpec> {
    prepare_selected_text_encoder_id(
        spec,
        engine_id,
        selected_id(advanced)?,
        manifest_entry,
        settings,
    )
}

/// Consume the request binder's selected encoder ID without consulting authored advanced JSON.
#[cfg(native_lane)]
pub(crate) fn prepare_selected_text_encoder_id(
    mut spec: LoadSpec,
    engine_id: &str,
    selection_id: Option<&str>,
    manifest_entry: &serde_json::Map<String, Value>,
    settings: &Settings,
) -> WorkerResult<LoadSpec> {
    let Some(selection_id) = selected_id_value(selection_id)? else {
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

    /// A tiny, validation-complete encoder contract. `write_encoder_contract_fixture` writes a
    /// sparse safetensors file with exactly this header surface; discovery reads only the config and
    /// headers, so nothing here loads or runs a model.
    const FIXTURE_CONTRACT_A: gen_core::EncoderContract = gen_core::EncoderContract {
        architecture: "qwen3",
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        vocab_size: 16,
        output_width: 8,
        loaded_hidden_layers: 2,
        requires_final_norm: false,
        requires_lm_head: false,
        hidden_activation: "silu",
        attention_dropout: gen_core::EncoderConfigFloat::new(0.0),
        rms_norm_eps: gen_core::EncoderConfigFloat::new(1e-6),
        qk_norm_eps: Some(gen_core::EncoderConfigFloat::new(1e-6)),
        rope_theta: gen_core::EncoderConfigFloat::new(1_000_000.0),
        max_position_embeddings: 4_096,
        attention_bias: gen_core::EncoderConfigBool::Required(false),
        tie_word_embeddings: gen_core::EncoderConfigBool::Required(true),
        tokenizer: gen_core::EncoderTokenizerContract {
            family: "testkit",
            binding: gen_core::EncoderTokenizerBinding::RetainBase,
            artifact_candidates: &["tokenizer.json"],
            required_tokens: &[],
        },
        prompt_executions: &[gen_core::EncoderPromptExecutionContract {
            purpose: "fixture_prompt",
            template: gen_core::EncoderPromptTemplate::QwenInstruct,
            add_special_tokens: true,
            length: gen_core::EncoderPromptLengthPolicy::RightTruncate { max_tokens: 8 },
            padding: gen_core::EncoderPromptPadding::None,
            prefix_trim: 0,
        }],
        bos_token_id: None,
        eos_token_id: None,
        image_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
        mrope_section: &[],
        mrope_interleaved: None,
        selected_hidden_layers: &[2],
        packing: None,
        dense_storage_dtype_probe: None,
    };

    /// A second contract whose tensor surface differs, so each fixture satisfies exactly one.
    const FIXTURE_CONTRACT_B: gen_core::EncoderContract = gen_core::EncoderContract {
        intermediate_size: 16,
        ..FIXTURE_CONTRACT_A
    };

    /// Test routing: `a*` rows use contract A (two different engine ids share it), `b*` rows use
    /// contract B, anything else has no encoder route.
    fn fixture_route(model: &Value) -> Option<EncoderRoute> {
        let id = model.get("id")?.as_str()?;
        let (engine, contract) = match id {
            "a_one" => ("engine_a_one", FIXTURE_CONTRACT_A),
            "a_two" => ("engine_a_two", FIXTURE_CONTRACT_A),
            "b_one" => ("engine_b_one", FIXTURE_CONTRACT_B),
            _ => return None,
        };
        Some(EncoderRoute {
            id: engine.to_owned(),
            contract,
        })
    }

    struct EncoderLibrary {
        _temp: tempfile::TempDir,
        data_dir: PathBuf,
        alternates: PathBuf,
        catalog: Vec<Value>,
    }

    /// A data dir holding a directory of candidates (an A encoder, a B encoder, a complete A
    /// snapshot and a config-less shard folder), plus one single-file B encoder seeded by path.
    fn encoder_library() -> EncoderLibrary {
        let temp = tempfile::tempdir().unwrap();
        let data_dir = temp.path().join("data");
        let alternates = data_dir.join("alternates");
        gen_core_testkit::write_encoder_contract_fixture(
            &alternates.join("enc-a"),
            FIXTURE_CONTRACT_A,
        )
        .unwrap();
        gen_core_testkit::write_encoder_contract_fixture(
            &alternates.join("enc-b"),
            FIXTURE_CONTRACT_B,
        )
        .unwrap();
        gen_core_testkit::write_encoder_contract_fixture(
            &alternates.join("snapshot-a").join("text_encoder"),
            FIXTURE_CONTRACT_A,
        )
        .unwrap();
        std::fs::create_dir_all(alternates.join("no-config")).unwrap();
        std::fs::write(
            alternates.join("no-config/model.safetensors"),
            b"not an encoder",
        )
        .unwrap();
        let single = data_dir.join("single-b");
        gen_core_testkit::write_encoder_contract_fixture(&single, FIXTURE_CONTRACT_B).unwrap();

        let catalog = vec![
            serde_json::json!({ "id": "a_one", "catalogScope": "builtin", "installedPath": alternates }),
            serde_json::json!({ "id": "unrouted", "catalogScope": "builtin" }),
            serde_json::json!({
                "id": "b_one",
                "catalogScope": "imported",
                "installedPath": single.join("model.safetensors"),
            }),
            serde_json::json!({ "id": "a_two", "catalogScope": "imported" }),
        ];
        EncoderLibrary {
            _temp: temp,
            data_dir,
            alternates,
            catalog,
        }
    }

    fn per_model_options(library: &EncoderLibrary) -> Vec<Vec<ImageTextEncoderOption>> {
        library
            .catalog
            .iter()
            .map(|model| {
                match candidates_for_model_with(
                    &library.catalog,
                    model,
                    &library.data_dir,
                    &[],
                    fixture_route,
                    validate_for_discovery,
                ) {
                    Ok((route, candidates)) => options_from_candidates(&route, candidates),
                    Err(_) => Vec::new(),
                }
            })
            .collect()
    }

    fn batched_options(
        library: &EncoderLibrary,
        memo: &DiscoveryMemo,
        inspections: &std::sync::atomic::AtomicUsize,
    ) -> Vec<Vec<ImageTextEncoderOption>> {
        options_for_catalog_with(
            &library.catalog,
            &library.data_dir,
            &[],
            memo,
            fixture_route,
            |contract, source, roots| {
                inspections.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                validate_for_discovery(contract, source, roots)
            },
        )
    }

    fn option_labels(options: &[ImageTextEncoderOption]) -> Vec<&str> {
        options.iter().map(|option| option.label.as_str()).collect()
    }

    #[test]
    fn batched_catalog_options_equal_the_per_model_computation() {
        let library = encoder_library();
        let reference = per_model_options(&library);
        // The fixture is only meaningful if alternates are actually discovered, per contract.
        assert_eq!(
            option_labels(&reference[0]),
            [
                "Model encoder (default)",
                "enc-a (builtin)",
                "snapshot-a (builtin)",
                "text_encoder (builtin)",
            ]
        );
        assert!(reference[1].is_empty(), "an unrouted model has no options");
        assert_eq!(
            option_labels(&reference[2]),
            [
                "Model encoder (default)",
                "enc-b (builtin)",
                "model (imported)",
            ]
        );
        assert_eq!(
            option_labels(&reference[3]),
            option_labels(&reference[0]),
            "a second route sharing contract A sees the same sources"
        );
        assert!(reference[3][1].description.contains("engine_a_two"));

        let memo = DiscoveryMemo::default();
        let inspections = std::sync::atomic::AtomicUsize::new(0);
        let cold = batched_options(&library, &memo, &inspections);
        assert_eq!(cold, reference);
        let sources = collect_sources(&library.catalog, &library.data_dir, &[]).len();
        assert_eq!(
            inspections.load(std::sync::atomic::Ordering::SeqCst),
            sources * 2,
            "each source is inspected once per distinct contract, not once per routed model"
        );
        let warm = batched_options(&library, &memo, &inspections);
        assert_eq!(warm, reference);
        assert_eq!(
            inspections.load(std::sync::atomic::Ordering::SeqCst),
            sources * 2,
            "an unchanged library is answered from the memo"
        );
        assert_eq!(memo.len(), sources);
        // Production entry point (process memo, real runtime routes) agrees with its own reference.
        for (model, options) in library
            .catalog
            .iter()
            .zip(image_text_encoder_options_for_catalog(
                &library.catalog,
                &library.data_dir,
                &[],
            ))
        {
            assert_eq!(
                options,
                image_text_encoder_options(&library.catalog, model, &library.data_dir, &[])
            );
        }
    }

    #[test]
    fn discovery_memo_invalidates_on_shard_or_config_change_and_removal() {
        use std::sync::atomic::Ordering;

        let library = encoder_library();
        let memo = DiscoveryMemo::default();
        let inspections = std::sync::atomic::AtomicUsize::new(0);
        batched_options(&library, &memo, &inspections);
        let mut seen = inspections.load(Ordering::SeqCst);
        let mut reinspected = |library: &EncoderLibrary| {
            let options = batched_options(library, &memo, &inspections);
            assert_eq!(
                options,
                per_model_options(library),
                "memo diverged from reference"
            );
            let now = inspections.load(Ordering::SeqCst);
            let delta = now - seen;
            seen = now;
            (options, delta)
        };
        assert_eq!(reinspected(&library).1, 0);

        // mtime alone (same size, same bytes) re-inspects that source.
        let shard = library.alternates.join("enc-a/model.safetensors");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&shard)
            .unwrap();
        file.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(120))
            .unwrap();
        drop(file);
        let (options, delta) = reinspected(&library);
        assert!(delta > 0, "an mtime change must miss the memo");
        assert!(option_labels(&options[0]).contains(&"enc-a (builtin)"));

        // A size change that breaks the shard drops the alternate on the next request.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&shard)
            .unwrap()
            .set_len(16)
            .unwrap();
        let (options, delta) = reinspected(&library);
        assert!(delta > 0);
        assert!(!option_labels(&options[0]).contains(&"enc-a (builtin)"));

        // Staging a config into the config-less folder is seen too (it stays incompatible).
        std::fs::write(library.alternates.join("no-config/config.json"), b"{}").unwrap();
        assert!(reinspected(&library).1 > 0);

        // Removing a source directory drops it from the listing and from the memo.
        let before = memo.len();
        std::fs::remove_dir_all(library.alternates.join("enc-b")).unwrap();
        let (options, _) = reinspected(&library);
        assert!(!option_labels(&options[2]).contains(&"enc-b (builtin)"));
        assert_eq!(memo.len(), before - 1);

        // A newly staged compatible encoder appears without any restart or explicit invalidation.
        gen_core_testkit::write_encoder_contract_fixture(
            &library.alternates.join("enc-b-new"),
            FIXTURE_CONTRACT_B,
        )
        .unwrap();
        let (options, delta) = reinspected(&library);
        assert!(delta > 0);
        assert!(option_labels(&options[2]).contains(&"enc-b-new (builtin)"));
        assert_eq!(reinspected(&library).1, 0, "and is memoized from then on");
    }

    fn collected_source<'a>(
        sources: &'a BTreeMap<(u8, PathBuf), (WeightsSource, String)>,
        kind: SourceKind,
        path: &Path,
    ) -> Option<&'a (WeightsSource, String)> {
        let kind = match kind {
            SourceKind::File => 0,
            SourceKind::Directory => 1,
        };
        sources.get(&(kind, crate::paths::normalize_absolute_path(path).unwrap()))
    }

    #[test]
    fn allowed_roots_and_source_keys_share_lexical_normalization() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("child/../models");
        let normalized = crate::paths::normalize_absolute_path(&root).unwrap();
        assert!(allowed_roots(&root, &[]).contains(&normalized));
        let source = WeightsSource::File(root.join("encoder.safetensors"));
        assert_eq!(
            source_key(&source).unwrap().1,
            normalized.join("encoder.safetensors")
        );
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
    #[cfg(native_lane)]
    #[test]
    fn worker_preparation_and_cache_reject_file_dir_snapshot_mutations_for_bespoke_qwen_control() {
        use std::cell::Cell;

        const ENGINE_ID: &str = "qwen_image_control";
        const CONTRACT_ROUTE_IDS: &[&str] = &["qwen_image", ENGINE_ID];

        let production_contract = crate::inference_runtime::media_encoder_contract(ENGINE_ID)
            .expect("the composed Qwen control route owns an encoder contract");
        assert_eq!(
            crate::inference_runtime::media_encoder_contract("qwen_image"),
            Some(production_contract),
            "the bespoke control route must retain the base Qwen encoder contract"
        );
        // This test owns receipt attachment and mutation rejection, not production geometry. Use
        // the same bounded Qwen2.5-VL contract pattern as the provider's sealed-source tests so
        // every file/dir/snapshot shape still crosses the production seal without hashing a
        // multi-gigabyte sparse payload. The provider separately pins its production dimensions.
        let bounded_contract = gen_core::EncoderContract {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 32,
            output_width: 64,
            loaded_hidden_layers: 1,
            max_position_embeddings: 512,
            mrope_section: &[4, 6, 6],
            selected_hidden_layers: &[1],
            ..production_contract
        };
        let _contract_guard = crate::inference_runtime::scoped_test_media_encoder_contract(
            CONTRACT_ROUTE_IDS,
            bounded_contract,
        );

        for shape in ["file", "dir", "snapshot"] {
            let fixture = tempfile::tempdir().unwrap();
            let base = fixture.path().join("base");
            let selected = fixture.path().join("selected");
            let control = fixture.path().join("control.safetensors");
            std::fs::create_dir_all(&base).unwrap();
            std::fs::write(&control, b"synthetic control overlay").unwrap();

            let contract = crate::inference_runtime::media_encoder_contract(ENGINE_ID)
                .expect("the scoped Qwen control route owns an encoder contract");
            gen_core_testkit::write_encoder_contract_tokenizer_fixture(&base, contract).unwrap();
            let source = match shape {
                "file" => {
                    gen_core_testkit::write_encoder_contract_fixture(&selected, contract).unwrap();
                    WeightsSource::File(selected.join("model.safetensors"))
                }
                "dir" => {
                    gen_core_testkit::write_encoder_contract_fixture(&selected, contract).unwrap();
                    WeightsSource::Dir(selected.clone())
                }
                "snapshot" => {
                    gen_core_testkit::write_encoder_contract_fixture(
                        &selected.join("text_encoder"),
                        contract,
                    )
                    .unwrap();
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
            let mut settings = crate::test_env::offline_settings();
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
                    // Inventory validation runs before shard parsing; a new direct shard must
                    // invalidate the prepared directory receipt.
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
