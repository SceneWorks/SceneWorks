//! Compact provenance projections of the existing immutable checkpoint plan. Catalog and client
//! metadata are never generation-time authority. Unknown historical declarations stay unknown.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{json, Map, Value};

use crate::checkpoint_import::ImportPlanV1;
use crate::checkpoint_plan_store::CheckpointPlanStore;

pub const MODEL_PROVENANCE_KEY: &str = "modelProvenance";
pub const MODEL_DECLARATION_KEY: &str = "modelDeclaration";

/// Called once by the import compiler, with the server-normalized import declaration. H3 license
/// owners are read from the trusted builtin component catalog, never from author-provided flags.
pub fn import_declaration_context(
    entry: &Map<String, Value>,
) -> Result<BTreeMap<String, Value>, String> {
    let mut normalized = Map::new();
    let h3 = crate::minimax_h3_import::entry_requires_minimax_h3_family(entry);
    let components = if h3 {
        crate::minimax_h3_import::MiniMaxH3Partition::from_entry(entry)?;
        let components = crate::minimax_h3_import::inherited_component_entries(entry);
        if components.is_empty() {
            return Err("inherited H3 component catalog is unavailable".into());
        }
        components
    } else {
        entry
            .get("baseModel")
            .and_then(Value::as_str)
            .and_then(crate::builtin_manifests::builtin_model_entry)
            .map(|base| {
                vec![crate::license_acknowledgments::component_license_metadata(
                    base,
                )]
            })
            .unwrap_or_default()
    };
    // Clear arbitrary author-provided ancestry even when no trusted base is known.
    crate::license_acknowledgments::apply_inherited_component_licenses(
        &mut normalized,
        &components,
    );
    let mut lineage = Map::new();
    if let Some(base) = entry
        .get("baseModel")
        .filter(|v| v.as_str().is_some_and(|s| !s.is_empty()))
    {
        lineage.insert("baseModel".into(), base.clone());
    }
    // Model IDs here identify license/component ancestry, not the imported denoiser's owner.
    lineage.insert(
        "componentLicenseOwners".into(),
        Value::Array(
            components
                .iter()
                .filter_map(|c| c.get("id").cloned())
                .collect(),
        ),
    );
    let mut semantics = Map::new();
    for key in [
        "family",
        "minimaxH3Partition",
        "importQuantFormat",
        "importComponentTier",
    ] {
        if let Some(value) = entry.get(key) {
            semantics.insert(key.into(), value.clone());
        }
    }
    let mut declaration = json!({
        "schemaVersion":if h3 {2} else {1}, "baseLineage":lineage,
        "declaredSemantics":semantics,
        "inheritedLicenseAcknowledgments":normalized.get("inheritedLicenseAcknowledgments").cloned().unwrap_or_else(||json!([])),
    });
    if h3 {
        declaration["componentDownloads"] = frozen_h3_component_downloads(entry)?;
    }
    Ok(BTreeMap::from([(
        MODEL_DECLARATION_KEY.into(),
        declaration,
    )]))
}

/// Authenticate the existing selection without opening missing original bytes. Only the managed
/// plan store may authorize repair; a path/model-ID match cannot turn linked or builtin weights
/// into application-owned bytes. The binding is also the publication compare-and-swap token.
pub fn managed_repair_binding(
    data_dir: &Path,
    entry: &Map<String, Value>,
) -> Result<Value, String> {
    let id = entry
        .get("id")
        .and_then(Value::as_str)
        .ok_or("repair selection has no ID")?;
    if crate::builtin_manifests::builtin_model_entry(id).is_some() {
        return Err("Builtin models cannot be replaced by a managed import".into());
    }
    let checkpoint = crate::jobs_store::checkpoint_plan_checkpoint_id(entry)
        .ok_or("Only an existing managed checkpoint import can be repaired")?;
    let store = CheckpointPlanStore::open(data_dir);
    let record = store.record(checkpoint).map_err(|e| e.to_string())?;
    let plan = store
        .plan(checkpoint, &record.plan.plan_id)
        .map_err(|e| e.to_string())?;
    record
        .validate_loaded_plan(&plan)
        .map_err(|e| e.to_string())?;
    let install = checkpoint
        .strip_prefix("managed/")
        .ok_or("Linked originals use relink/rescan, not managed repair")?;
    let directory = store.install_dir(install).map_err(|e| e.to_string())?;
    if plan.layers.iter().any(|layer| !matches!(&layer.source,
        crate::checkpoint_import::SourceLocatorV1::Managed { install_id, .. } if install_id == install)) {
        return Err("Repair selection does not own its original managed installation".into());
    }
    let supplied = entry
        .get("paths")
        .and_then(|v| v.get("model"))
        .and_then(Value::as_str)
        .ok_or("Repair selection has no managed installation path")?;
    // Normalize only the parent of a missing install, never a source's filename/extension.
    let normalized = |path: &Path| -> std::path::PathBuf {
        std::fs::canonicalize(path).unwrap_or_else(|_| {
            path.parent()
                .and_then(|parent| std::fs::canonicalize(parent).ok())
                .zip(path.file_name())
                .map(|(parent, name)| parent.join(name))
                .unwrap_or_else(|| path.to_owned())
        })
    };
    if normalized(Path::new(supplied)) != normalized(&directory) {
        return Err("Repair selection does not match its managed installation".into());
    }
    let binding = entry
        .get("importPlan")
        .ok_or("Repair selection has no import plan")?;
    if binding
        .get("planId")
        .and_then(Value::as_str)
        .is_some_and(|id| id != plan.plan_id)
        || binding
            .get("semanticDigest")
            .and_then(Value::as_str)
            .is_some_and(|id| id != record.plan.semantic_digest)
    {
        return Err("Repair selection has a stale import-plan binding".into());
    }
    Ok(
        json!({"modelId":id,"importPlan":binding,"paths":entry.get("paths"),"source":entry.get("source")}),
    )
}

/// Freeze the complete download declaration, not just its tier. Config-only provisioning is
/// omitted because the actual configuration bytes/provenance live in transformerConfig.
pub fn frozen_h3_component_downloads(entry: &Map<String, Value>) -> Result<Value, String> {
    let mut rows: Vec<_> = entry
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|row| {
            row.get("componentId").and_then(Value::as_str) != Some("import_transformer_config")
        })
        .cloned()
        .collect();
    for row in &rows {
        let repo = row
            .get("repo")
            .and_then(Value::as_str)
            .ok_or("component repository is missing")?;
        let revision = row
            .get("revision")
            .and_then(Value::as_str)
            .ok_or("component revision is missing")?;
        crate::model_artifacts::ArtifactIdentity::pinned(
            repo,
            revision,
            row.get("variant")
                .and_then(Value::as_str)
                .unwrap_or("component"),
        )
        .map_err(|e| e.to_string())?;
        if row
            .get("files")
            .and_then(Value::as_array)
            .is_none_or(Vec::is_empty)
        {
            return Err("component download has no file closure".into());
        }
    }
    rows.sort_by_key(|row| serde_json::to_string(row).expect("JSON row"));
    Ok(Value::Array(rows))
}

/// Expand only the frozen component rows in the existing approved source library. The caller
/// hashes once while importing, under cancellation supervision; paths never enter semantic identity.
pub fn h3_component_source_files(
    data_dir: &Path,
    entry: &Map<String, Value>,
) -> Result<Vec<(Value, std::path::PathBuf, Vec<String>)>, String> {
    let rows = frozen_h3_component_downloads(entry)?;
    let rows = rows.as_array().expect("array");
    if rows.is_empty() {
        return Err(
            "H3 component closure is missing; provision components and retry the import".into(),
        );
    }
    let resolver = crate::model_artifacts::ModelArtifactResolver::new(
        crate::hf_home::model_source_library(data_dir),
    );
    let mut result = Vec::new();
    for row in rows {
        let repo = row["repo"].as_str().expect("validated repo");
        let revision = row["revision"].as_str().expect("validated revision");
        let (_, snapshot) = resolver
            .discover_source_snapshot(repo, Some(revision))
            .map_err(|e| {
                format!("component unavailable; finish component downloads and retry: {e}")
            })?;
        if !crate::minimax_h3_import::import_component_files_complete(&snapshot, row) {
            return Err(format!(
                "component {repo}@{revision} is incomplete; finish component downloads and retry"
            ));
        }
        let mut files = std::collections::BTreeSet::new();
        for pattern in row["files"].as_array().expect("validated files") {
            let pattern = pattern.as_str().ok_or("invalid component file pattern")?;
            // Globs are permitted, escapes and absolute paths are not.
            crate::checkpoint_plan_store::portable_relative_path_parts(pattern)?;
            let absolute = snapshot.join(pattern);
            let mut matched = false;
            for path in glob::glob(absolute.to_str().ok_or("non-UTF8 component path")?)
                .map_err(|e| e.to_string())?
            {
                let path = path.map_err(|e| e.to_string())?;
                if !path.is_file()
                    || !path
                        .extension()
                        .and_then(|e| e.to_str())
                        .is_some_and(|ext| {
                            matches!(
                                ext,
                                "safetensors"
                                    | "gguf"
                                    | "json"
                                    | "yaml"
                                    | "yml"
                                    | "model"
                                    | "txt"
                                    | "tiktoken"
                            )
                        })
                {
                    continue;
                }
                let relative = path
                    .strip_prefix(&snapshot)
                    .map_err(|e| e.to_string())?
                    .to_str()
                    .ok_or("non-UTF8 component file")?
                    .replace('\\', "/");
                files.insert(relative);
                matched = true;
            }
            if !matched {
                return Err(format!(
                    "component file {repo}:{pattern} is missing; finish downloads and retry"
                ));
            }
        }
        let files: Vec<_> = files.into_iter().collect();
        // Discovery can select an approved local-preference snapshot rather than the primary
        // library root. Confine links to THAT discovered repository, not a guessed cache path.
        let repository_root = snapshot
            .parent()
            .and_then(Path::parent)
            .ok_or("invalid component snapshot")?;
        let library_root = repository_root
            .parent()
            .ok_or("invalid component library")?;
        let names: Vec<_> = files.iter().map(String::as_str).collect();
        crate::hf_repo_renames::validate_huggingface_snapshot_files(
            library_root,
            repository_root,
            &snapshot,
            &names,
        )
        .ok_or("component files escape their approved source-library repository")?;
        result.push((row.clone(), snapshot, files));
    }
    Ok(result)
}

/// Structural validation at plan serialization/deserialization. Legacy declarations keep their
/// old bytes; v2 must contain actual complete immutable content, never default unknowns to authority.
pub(crate) fn validate_h3_component_declaration(declaration: &Value) -> Result<(), String> {
    let downloads = declaration
        .get("componentDownloads")
        .and_then(Value::as_array)
        .filter(|v| !v.is_empty())
        .ok_or("missing frozen component downloads")?;
    let closure = declaration
        .get("componentClosure")
        .and_then(Value::as_array)
        .filter(|v| !v.is_empty())
        .ok_or("missing frozen component content")?;
    let normalized =
        frozen_h3_component_downloads(json!({"downloads":downloads}).as_object().expect("object"))?;
    if normalized != *declaration.get("componentDownloads").expect("checked")
        || closure.len() != downloads.len()
    {
        return Err("noncanonical/incomplete frozen component closure".into());
    }
    for (download, member) in downloads.iter().zip(closure) {
        if member.get("download") != Some(download)
            || member.get("revision") != download.get("revision")
        {
            return Err("component content does not match its frozen revision/declaration".into());
        }
        let files = member
            .get("files")
            .and_then(Value::as_array)
            .filter(|v| !v.is_empty())
            .ok_or("missing frozen component files")?;
        let mut previous = None;
        for file in files {
            let path = file
                .get("relativePath")
                .and_then(Value::as_str)
                .ok_or("missing frozen component path")?;
            crate::checkpoint_plan_store::portable_relative_path_parts(path)?;
            if previous.is_some_and(|old: &str| old >= path) {
                return Err("component file paths must be uniquely sorted".into());
            }
            previous = Some(path);
            if file.get("sizeBytes").and_then(Value::as_u64).is_none() {
                return Err("missing frozen component size".into());
            }
            let digest = file
                .get("sha256")
                .and_then(Value::as_str)
                .and_then(|s| s.strip_prefix("sha256:"))
                .ok_or("missing frozen component digest")?;
            if digest.len() != 64
                || !digest
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err("invalid frozen component digest".into());
            }
        }
    }
    Ok(())
}

/// One import/rescan acquisition over the platform source closure, with bounded cooperative
/// hashing and source leases. Observations are ephemeral; only revisions/content enter the plan.
pub fn capture_h3_component_closure_with_check(
    data_dir: &Path,
    entry: &Map<String, Value>,
    check: &dyn Fn() -> std::io::Result<()>,
) -> Result<Value, String> {
    use crate::checkpoint_plan_store::{SourceStampV1, StoreClockV1};
    use crate::model_artifacts::{
        ArtifactAvailability, ArtifactFile, ArtifactIdentity, ArtifactMemberRole,
        ModelArtifactResolver, ResolvedBundleClosure, ResolvedBundleMember,
    };
    check().map_err(|e| e.to_string())?;
    let sources = h3_component_source_files(data_dir, entry)?;
    let resolver = ModelArtifactResolver::new(crate::hf_home::model_source_library(data_dir));
    let mut observations = Vec::new();
    let mut leases = Vec::new();
    let mut closure = Vec::new();
    for (row, snapshot, paths) in &sources {
        check().map_err(|e| e.to_string())?;
        let identity = ArtifactIdentity::pinned(
            row["repo"].as_str().expect("validated"),
            row["revision"].as_str().expect("validated"),
            row.get("variant")
                .and_then(Value::as_str)
                .unwrap_or("component"),
        )
        .map_err(|e| e.to_string())?;
        let bundle = ResolvedBundleClosure::new(vec![ResolvedBundleMember {
            role: ArtifactMemberRole::Primary,
            component_id: row
                .get("componentId")
                .and_then(Value::as_str)
                .map(str::to_owned),
            source: identity.clone(),
            tier: row
                .get("variant")
                .and_then(Value::as_str)
                .map(str::to_owned),
            source_subpath: "".into(),
            destination: "".into(),
            files: paths
                .iter()
                .map(ArtifactFile::new)
                .collect::<Result<_, _>>()
                .map_err(|e| e.to_string())?,
        }])
        .map_err(|e| e.to_string())?;
        let artifact = resolver
            .resolve_source(identity, bundle, ArtifactAvailability::Available)
            .map_err(|e| e.to_string())?;
        leases.push(
            resolver
                .acquire_runtime_lease(&std::sync::Arc::new(artifact))
                .map_err(|e| e.to_string())?,
        );
        let mut files = Vec::new();
        for relative in paths {
            check().map_err(|e| e.to_string())?;
            let path = snapshot.join(relative);
            let before =
                SourceStampV1::of(&path, StoreClockV1::SYSTEM).map_err(|e| e.to_string())?;
            let target = std::fs::canonicalize(&path).map_err(|e| e.to_string())?;
            let hash = crate::checkpoint_plan_store::source_file_sha256_with_check(&path, check)
                .map_err(|e| e.to_string())?;
            files.push(json!({"relativePath":relative,"sha256":format!("sha256:{hash}"),"sizeBytes":before.size_bytes}));
            observations.push((path, target, before));
        }
        closure.push(json!({"revision":row["revision"],"download":row,"files":files}));
    }
    for (path, target, before) in observations {
        check().map_err(|e| e.to_string())?;
        let after = SourceStampV1::of(&path, StoreClockV1::SYSTEM).map_err(|e| e.to_string())?;
        if !before.identifies_same_entry(&after)
            || std::fs::canonicalize(&path).map_err(|e| e.to_string())? != target
        {
            return Err("H3 component source changed while binding the assembly; retry after files are stable".into());
        }
    }
    if h3_component_source_files(data_dir, entry)? != sources {
        return Err("H3 component membership changed while binding the assembly; retry after files are stable".into());
    }
    check().map_err(|e| e.to_string())?;
    Ok(json!(closure))
}

/// Compare already acquired component seals to the persisted assembly. No rehash and no fresh
/// authority: a changed cache object, revision, or unconsumed commitment is a hard refusal.
pub fn verify_prepared_h3_component_closure(
    data_dir: &Path,
    assembly: &Value,
    files: &[gen_core::PinnedWeightsFile],
) -> Result<(), String> {
    let closure = assembly.get("componentClosure").and_then(Value::as_array)
        .filter(|rows| !rows.is_empty())
        .ok_or("historical H3 component content is unknown; explicitly repair/reimport or rescan before generation")?;
    let resolver = crate::model_artifacts::ModelArtifactResolver::new(
        crate::hf_home::model_source_library(data_dir),
    );
    fn logical_path(relative: &str) -> &str {
        relative
            .strip_prefix("q4/")
            .or_else(|| relative.strip_prefix("q8/"))
            .or_else(|| relative.strip_prefix("bf16/"))
            .unwrap_or(relative)
    }
    let logical_files: Vec<_> = closure
        .iter()
        .filter_map(|row| row.get("files").and_then(Value::as_array))
        .flatten()
        .filter_map(|file| file.get("relativePath").and_then(Value::as_str))
        .map(logical_path)
        .collect();
    let mut committed_paths = std::collections::BTreeSet::new();
    let mut snapshot_roots = std::collections::BTreeSet::new();
    let mut acquired_component_dirs = std::collections::BTreeSet::new();
    for member in closure {
        let row = member
            .get("download")
            .ok_or("missing frozen component declaration")?;
        let repo = row
            .get("repo")
            .and_then(Value::as_str)
            .ok_or("missing component repo")?;
        let revision = member
            .get("revision")
            .and_then(Value::as_str)
            .ok_or("missing component revision")?;
        if row.get("revision").and_then(Value::as_str) != Some(revision) {
            return Err("component revision commitment is inconsistent".into());
        }
        let (_, snapshot) = resolver
            .discover_source_snapshot(repo, Some(revision))
            .map_err(|e| e.to_string())?;
        snapshot_roots.insert(snapshot.clone());
        for expected in member
            .get("files")
            .and_then(Value::as_array)
            .filter(|v| !v.is_empty())
            .ok_or("empty frozen component closure")?
        {
            let relative = expected
                .get("relativePath")
                .and_then(Value::as_str)
                .ok_or("missing component path")?;
            crate::checkpoint_plan_store::portable_relative_path_parts(relative)?;
            let path = snapshot.join(relative);
            committed_paths.insert(path.clone());
            // Source-library snapshots and supported operator roots can be aliases of the same
            // semantic component. Match its logical relative path, never ownership/root metadata.
            let logical = logical_path(relative);
            let candidates: Vec<_> = files
                .iter()
                .filter(|pin| {
                    let matches = pin.loader_path() == path
                        || pin.loader_path().ends_with(relative)
                        || pin.loader_path().ends_with(logical)
                        || pin.canonical_target_path().ends_with(relative);
                    // The FL2VA source audio descriptor is distinct from audio_vae/config.json.
                    // Choose the most-specific logical binding, never every matching suffix.
                    matches
                        && !logical_files.iter().any(|other| {
                            other.len() > logical.len()
                                && (pin.loader_path().ends_with(other)
                                    || pin.canonical_target_path().ends_with(other))
                        })
                })
                .collect();
            if candidates.is_empty() {
                return Err(format!(
                    "frozen H3 component was not acquired: {}",
                    path.display()
                ));
            }
            for pin in candidates {
                let digest = format!(
                    "sha256:{}",
                    pin.content_sha256()
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>()
                );
                if expected.get("sha256").and_then(Value::as_str) != Some(digest.as_str()) {
                    return Err(format!(
                        "H3 component content drift: {}; explicitly repair/reimport or rescan",
                        pin.loader_path().display()
                    ));
                }
                committed_paths.insert(pin.loader_path().to_owned());
                if let Some(parent) = pin.loader_path().parent() {
                    acquired_component_dirs.insert(parent.to_owned());
                }
            }
        }
    }
    for pin in files {
        if snapshot_roots
            .iter()
            .chain(acquired_component_dirs.iter())
            .any(|root| pin.loader_path().starts_with(root))
            && !committed_paths.contains(pin.loader_path())
        {
            return Err(format!(
                "uncommitted H3 component was acquired: {}",
                pin.loader_path().display()
            ));
        }
    }
    Ok(())
}

/// A portable, compact explanation of a validated immutable plan. The full configuration stays
/// in the plan; assets carry only its content commitment and original source/base provenance.
pub fn checkpoint_model_provenance(plan: &ImportPlanV1, model_id: &str) -> Result<Value, String> {
    let reference = plan.plan_reference().map_err(|e| e.to_string())?;
    let sources = plan
        .layers
        .iter()
        .map(|layer| {
            Ok(json!({"layerId":layer.layer_id,"role":layer.role,
            "sha256":layer.source.content_digest().map_err(|e| e.to_string())?,
            "source":layer.source}))
        })
        .collect::<Result<Vec<Value>, String>>()?;
    let mut provenance = json!({"schemaVersion":1,"modelId":model_id,"family":plan.family,
        "plan":reference,"sources":sources,"inheritedLicenseAcknowledgments":[],"derivativeHistory":[]});
    if let Some(declaration) = plan.context.get(MODEL_DECLARATION_KEY) {
        for key in [
            "baseLineage",
            "declaredSemantics",
            "componentDownloads",
            "componentClosure",
            "inheritedLicenseAcknowledgments",
        ] {
            if let Some(value) = declaration.get(key) {
                provenance[key] = value.clone();
            }
        }
    }
    if let Some(encoding) = plan.context.get("sourceEncoding") {
        let semantics = provenance
            .as_object_mut()
            .expect("object above")
            .entry("declaredSemantics")
            .or_insert_with(|| json!({}));
        semantics["sourceEncoding"] = encoding.clone();
    }
    if let Some(binding) = plan.context.get("transformerConfig") {
        // The source may legitimately differ from the base. Check its own binding, never JSON
        // equality against stock configuration, and do not duplicate its payload in every asset.
        crate::minimax_h3_import::bound_config(&plan.context)?;
        provenance["configuration"] =
            json!({"sha256":binding["sha256"],"provenance":binding["provenance"]});
    }
    Ok(provenance)
}

/// Resolve the record/plan commitment before projecting any provenance. Old imported entries with
/// no plan retain their route and return None. Old plans retain their hashes without invented
/// base/partition declarations. A new bound declaration cannot be overwritten by catalog edits.
pub fn resolved_model_provenance(
    data_dir: &Path,
    entry: &Map<String, Value>,
) -> Result<Option<Value>, String> {
    let Some(checkpoint_id) = crate::jobs_store::checkpoint_plan_checkpoint_id(entry) else {
        return Ok(None);
    };
    let store = CheckpointPlanStore::open(data_dir);
    let record = store.record(checkpoint_id).map_err(|e| e.to_string())?;
    let plan = store
        .plan(checkpoint_id, &record.plan.plan_id)
        .map_err(|e| e.to_string())?;
    record
        .validate_loaded_plan(&plan)
        .map_err(|e| e.to_string())?;
    let binding = entry
        .get("importPlan")
        .ok_or("missing import plan binding")?;
    if binding
        .get("planId")
        .and_then(Value::as_str)
        .is_some_and(|id| id != plan.plan_id)
        || binding
            .get("semanticDigest")
            .and_then(Value::as_str)
            .is_some_and(|digest| digest != record.plan.semantic_digest)
    {
        return Err("model entry no longer matches its immutable import plan".into());
    }
    validate_entry_declaration(&plan, entry)?;
    checkpoint_model_provenance(
        &plan,
        entry.get("id").and_then(Value::as_str).unwrap_or_default(),
    )
    .map(Some)
}

/// Shared by admission and the native provider's pre-load guard. A catalog edit cannot reinterpret
/// the same checkpoint as another partition, compatible base, or resident-component precision.
pub fn validate_entry_declaration(
    plan: &ImportPlanV1,
    entry: &Map<String, Value>,
) -> Result<(), String> {
    if plan.family == crate::minimax_h3_import::MINIMAX_H3_IMPORT_FAMILY {
        let declaration = plan.context.get(MODEL_DECLARATION_KEY).ok_or(
            "historical H3 component content is unknown; explicitly repair/reimport or rescan",
        )?;
        if declaration.get("schemaVersion").and_then(Value::as_f64) != Some(2.0)
            || declaration
                .get("componentClosure")
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty)
        {
            return Err(
                "historical H3 component content is unknown; explicitly repair/reimport or rescan"
                    .into(),
            );
        }
        let downloads = frozen_h3_component_downloads(entry)?;
        if declaration.get("componentDownloads") != Some(&downloads) {
            return Err("component downloads differ from the immutable import declaration; explicitly repair/reimport or rescan".into());
        }
        let closure = declaration["componentClosure"].as_array().expect("checked");
        if closure
            .iter()
            .map(|member| member.get("download"))
            .collect::<Vec<_>>()
            != downloads
                .as_array()
                .expect("array")
                .iter()
                .map(Some)
                .collect::<Vec<_>>()
        {
            return Err("component content closure does not cover the frozen downloads".into());
        }
    }
    if let Some(declaration) = plan.context.get(MODEL_DECLARATION_KEY) {
        if !declaration
            .get("schemaVersion")
            .and_then(Value::as_f64)
            .is_some_and(|version| version == 1.0 || version == 2.0)
        {
            return Err("unsupported immutable model declaration".into());
        }
        for (path, key) in [
            ("/baseLineage/baseModel", "baseModel"),
            (
                "/declaredSemantics/minimaxH3Partition",
                "minimaxH3Partition",
            ),
            (
                "/declaredSemantics/importComponentTier",
                "importComponentTier",
            ),
        ] {
            if declaration
                .pointer(path)
                .is_some_and(|bound| entry.get(key) != Some(bound))
            {
                return Err(format!(
                    "model {key} differs from the immutable import declaration"
                ));
            }
        }
    }
    Ok(())
}

/// Worker facts are mutable reports, not source authority. Every producer uses this same boundary;
/// an absent historical binding clears a forged claim instead of fabricating provenance.
pub fn stamp_asset_provenance(facts: &mut [Value], provenance: Option<&Value>) {
    for fact in facts {
        let Some(object) = fact.as_object_mut() else {
            continue;
        };
        object.remove(MODEL_PROVENANCE_KEY);
        for key in ["extra", "rawAdapterSettings"] {
            if let Some(nested) = object.get_mut(key).and_then(Value::as_object_mut) {
                nested.remove(MODEL_PROVENANCE_KEY);
            }
        }
        if let Some(provenance) = provenance {
            object.insert(MODEL_PROVENANCE_KEY.into(), provenance.clone());
        }
    }
}

/// Redeem only a concrete conversion receipt. No cache scan, guessed codec, or mutable metadata
/// is substituted for evidence of a derivative actually held by the producer.
pub fn verify_asset_derivative_provenance(
    data_dir: &Path,
    facts: &mut [Value],
) -> Result<(), String> {
    let mut store = None;
    let mut receipts: BTreeMap<String, Value> = BTreeMap::new();
    for fact in facts {
        let claim = fact
            .get_mut("rawAdapterSettings")
            .and_then(Value::as_object_mut)
            .and_then(|raw| raw.remove("checkpointDerivative"));
        let Some(claim) = claim else {
            continue;
        };
        // An unfrozen historical job cannot authenticate conversion claims either. Discard the
        // claim instead of inventing ancestry or making terminal persistence retry forever.
        let Some(provenance) = fact.get_mut(MODEL_PROVENANCE_KEY) else {
            continue;
        };
        let cache_key = claim
            .get("cacheKey")
            .and_then(Value::as_str)
            .ok_or("invalid checkpoint derivative identity")?;
        let receipt = if let Some(receipt) = receipts.get(cache_key) {
            receipt.clone()
        } else {
            if store.is_none() {
                store = Some(
                    crate::checkpoint_derivative::CheckpointDerivativeStore::open(data_dir)
                        .map_err(|e| e.to_string())?,
                );
            }
            let receipt = store
                .as_ref()
                .expect("opened above")
                .verified_provenance_receipt(cache_key, provenance)
                .map_err(|e| e.to_string())?;
            receipts.insert(cache_key.to_owned(), receipt.clone());
            receipt
        };
        provenance["derivativeHistory"] = json!([receipt]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint_import::{CheckpointContainerV1, ImportLayerV1, SourceLocatorV1};

    fn plan() -> ImportPlanV1 {
        ImportPlanV1::new(
            format!("checkpoint-plan-{}", "1".repeat(64)),
            "ltx2",
            vec![ImportLayerV1 {
                layer_id: "dit".into(),
                role: "transformer".into(),
                target_path: "model.safetensors".into(),
                container: CheckpointContainerV1::Safetensors,
                source: SourceLocatorV1::linked("root-test", "model.safetensors", "a".repeat(64))
                    .unwrap(),
            }],
        )
        .unwrap()
    }

    #[test]
    fn managed_recovery_transaction_is_bound_but_never_changes_content_equivalence() {
        use crate::checkpoint_import::ManagedProvenanceV1;
        let linked = plan();
        let mut managed = linked.clone();
        let mut provenance = ManagedProvenanceV1::of_source("local-copy");
        provenance.import_job_id = Some("original-job".into());
        provenance.import_commitment = Some(format!("sha256:{}", "b".repeat(64)));
        managed.layers[0].source = SourceLocatorV1::managed(
            "owned-install",
            "model.safetensors",
            "a".repeat(64),
            provenance.clone(),
        )
        .unwrap();
        assert_eq!(
            managed.semantic_digest().unwrap(),
            linked.semantic_digest().unwrap()
        );
        assert_eq!(
            managed.duplicate_equivalence_digest().unwrap(),
            linked.duplicate_equivalence_digest().unwrap()
        );
        let bound = managed.source_binding_identity().unwrap();
        let document = managed.canonical_json().unwrap();
        let decoded: ImportPlanV1 = serde_json::from_str(&document).unwrap();
        assert_eq!(decoded.source_binding_identity().unwrap(), bound);
        provenance.import_job_id = Some("unrelated-job".into());
        managed.layers[0].source = SourceLocatorV1::managed(
            "owned-install",
            "model.safetensors",
            "a".repeat(64),
            provenance.clone(),
        )
        .unwrap();
        assert_ne!(managed.source_binding_identity().unwrap(), bound);
        assert_eq!(
            managed.semantic_digest().unwrap(),
            linked.semantic_digest().unwrap()
        );
        provenance.import_commitment = None;
        assert!(
            provenance.validate().is_err(),
            "half-present legacy fields cannot invent authority"
        );
        let old: ManagedProvenanceV1 =
            serde_json::from_value(json!({"source":"legacy-import"})).unwrap();
        assert!(old.import_job_id.is_none() && old.import_commitment.is_none());
        assert_eq!(
            serde_json::to_value(old).unwrap(),
            json!({"source":"legacy-import"})
        );
    }

    fn frozen_h3_plan() -> (ImportPlanV1, Map<String, Value>) {
        let mut plan = plan();
        plan.family = "minimax-h3".into();
        let entry = json!({"family":"minimax-h3","baseModel":"minimax_h3","minimaxH3Partition":"fl2va","importComponentTier":"q4",
            "downloads":[{"provider":"huggingface","repo":"fixture/components","revision":"1111111111111111111111111111111111111111","files":["tokenizer/tokenizer.json"]}]}).as_object().unwrap().clone();
        plan.context = import_declaration_context(&entry).unwrap();
        let rows = frozen_h3_component_downloads(&entry).unwrap();
        plan.context.get_mut(MODEL_DECLARATION_KEY).unwrap()["componentClosure"] = json!([{"download":rows[0],"revision":rows[0]["revision"],"files":[{"relativePath":"tokenizer/tokenizer.json","sizeBytes":2,"sha256":format!("sha256:{}","a".repeat(64))}]}]);
        (plan, entry)
    }

    #[test]
    fn component_revisions_and_bytes_are_immutable_replay_identity() {
        let (plan, entry) = frozen_h3_plan();
        validate_entry_declaration(&plan, &entry).unwrap();
        let baseline = plan.semantic_digest().unwrap();
        let provenance = checkpoint_model_provenance(&plan, "selection").unwrap();
        let schema_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../packages/schemas/model-manifest.schema.json");
        let schema: Value = serde_json::from_slice(&std::fs::read(schema_path).unwrap()).unwrap();
        let validator = jsonschema::validator_for(
            &json!({"$defs":schema["$defs"],"$ref":"#/$defs/modelProvenance"}),
        )
        .unwrap();
        assert!(
            validator.is_valid(&provenance),
            "full component authority round-trips on the public output schema"
        );
        assert_eq!(
            provenance["componentClosure"],
            plan.context[MODEL_DECLARATION_KEY]["componentClosure"]
        );
        let mut changed_revision = entry.clone();
        changed_revision["downloads"][0]["revision"] = json!("2".repeat(40));
        assert!(validate_entry_declaration(&plan, &changed_revision)
            .unwrap_err()
            .contains("downloads differ"));
        let mut changed_bytes = plan.clone();
        changed_bytes
            .context
            .get_mut(MODEL_DECLARATION_KEY)
            .unwrap()["componentClosure"][0]["files"][0]["sha256"] =
            json!(format!("sha256:{}", "b".repeat(64)));
        assert_ne!(baseline, changed_bytes.semantic_digest().unwrap());
        // Complete source/declaration topology is bound, not a selected handful of component IDs.
        let mut additional = entry;
        additional["downloads"].as_array_mut().unwrap().push(json!({"repo":"fixture/companion","revision":"3".repeat(40),"files":["transformer_ref/model.safetensors"]}));
        assert!(validate_entry_declaration(&plan, &additional).is_err());
    }

    #[test]
    fn unknown_legacy_h3_content_cannot_gain_immutable_authority_by_serde_default() {
        let (mut plan, entry) = frozen_h3_plan();
        plan.context
            .get_mut(MODEL_DECLARATION_KEY)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove("componentClosure");
        assert!(
            plan.canonical_json().is_err(),
            "v2 cannot serialize unknown content as immutable authority"
        );
        plan.context.get_mut(MODEL_DECLARATION_KEY).unwrap()["schemaVersion"] = json!(1);
        let decoded: ImportPlanV1 = serde_json::from_str(&plan.canonical_json().unwrap()).unwrap();
        assert!(validate_entry_declaration(&decoded, &entry)
            .unwrap_err()
            .contains("content is unknown"));
        plan.context.clear();
        assert!(validate_entry_declaration(&plan, &entry)
            .unwrap_err()
            .contains("content is unknown"));
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn component_acquisition_refuses_drift_extra_files_and_cancellation_without_adopting_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        let hub = dir.path().join("hub");
        let _guard = crate::hf_home::test_support::CacheRootGuard::new(data.clone(), hub.clone());
        let (mut plan, mut entry) = frozen_h3_plan();
        entry["downloads"][0]["files"] = json!(["tokenizer/*.json"]);
        let snapshot = hub
            .join("models--fixture--components/snapshots")
            .join("1".repeat(40));
        let file = snapshot.join("tokenizer/tokenizer.json");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"{}").unwrap();
        let closure = capture_h3_component_closure_with_check(&data, &entry, &|| Ok(())).unwrap();
        plan.context = import_declaration_context(&entry).unwrap();
        plan.context.get_mut(MODEL_DECLARATION_KEY).unwrap()["componentClosure"] = closure.clone();
        let baseline = plan.semantic_digest().unwrap();
        let assembly = json!({"componentClosure":closure});
        let first = gen_core::PinnedWeightsFile::pin(&file).unwrap();
        verify_prepared_h3_component_closure(&data, &assembly, &[first]).unwrap();
        let canceled = capture_h3_component_closure_with_check(&data, &entry, &|| {
            Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "canceled",
            ))
        });
        assert!(canceled.unwrap_err().contains("canceled"));
        assert_eq!(
            plan.semantic_digest().unwrap(),
            baseline,
            "cancellation cannot mutate persisted authority"
        );
        std::fs::write(&file, b"[]").unwrap();
        let replacement = gen_core::PinnedWeightsFile::pin(&file).unwrap();
        assert!(verify_prepared_h3_component_closure(
            &data,
            &assembly,
            std::slice::from_ref(&replacement)
        )
        .unwrap_err()
        .contains("content drift"));
        let explicit = capture_h3_component_closure_with_check(&data, &entry, &|| Ok(())).unwrap();
        plan.context.get_mut(MODEL_DECLARATION_KEY).unwrap()["componentClosure"] = explicit.clone();
        assert_ne!(
            plan.semantic_digest().unwrap(),
            baseline,
            "explicit rescan/reimport is a NEW replay identity"
        );
        let extra_path = snapshot.join("tokenizer/new.json");
        std::fs::write(&extra_path, b"{}").unwrap();
        let extra = gen_core::PinnedWeightsFile::pin(extra_path).unwrap();
        assert!(verify_prepared_h3_component_closure(
            &data,
            &json!({"componentClosure":explicit}),
            &[replacement, extra]
        )
        .unwrap_err()
        .contains("uncommitted"));
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn prepared_component_matching_distinguishes_audio_vae_from_nested_source_descriptor() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data");
        let hub = dir.path().join("hub");
        let _guard = crate::hf_home::test_support::CacheRootGuard::new(data.clone(), hub.clone());
        let (_, mut entry) = frozen_h3_plan();
        entry["downloads"][0]["files"] =
            json!(["audio_vae/config.json", "FL2VA/audio_vae/config.json"]);
        let snapshot = hub
            .join("models--fixture--components/snapshots")
            .join("1".repeat(40));
        let paths = [
            snapshot.join("audio_vae/config.json"),
            snapshot.join("FL2VA/audio_vae/config.json"),
        ];
        for (index, path) in paths.iter().enumerate() {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, format!("{{\"component\":{index}}}")).unwrap();
        }
        let closure = capture_h3_component_closure_with_check(&data, &entry, &|| Ok(())).unwrap();
        let pins: Vec<_> = paths
            .iter()
            .map(|path| gen_core::PinnedWeightsFile::pin(path).unwrap())
            .collect();
        verify_prepared_h3_component_closure(&data, &json!({"componentClosure":closure}), &pins)
            .unwrap();
    }

    #[test]
    fn old_plans_keep_identity_without_forging_historical_declarations() {
        let plan = plan();
        let before = plan.canonical_json().unwrap();
        let provenance = checkpoint_model_provenance(&plan, "old-import").unwrap();
        assert_eq!(provenance["sources"][0]["sha256"], "a".repeat(64));
        assert!(provenance.get("baseLineage").is_none());
        assert!(provenance.get("declaredSemantics").is_none());
        assert_eq!(plan.canonical_json().unwrap(), before);
        let decoded: ImportPlanV1 = serde_json::from_str(&before).unwrap();
        assert_eq!(
            decoded.plan_reference().unwrap(),
            plan.plan_reference().unwrap()
        );
    }

    #[test]
    fn declaration_and_provenance_roundtrip_are_immutable_content() {
        let mut plan = plan();
        let entry = json!({"baseModel":"ltx-base", "family":"ltx2"});
        plan.context = import_declaration_context(entry.as_object().unwrap()).unwrap();
        let committed = plan.semantic_digest().unwrap();
        let roundtrip: ImportPlanV1 =
            serde_json::from_str(&plan.canonical_json().unwrap()).unwrap();
        let provenance = checkpoint_model_provenance(&roundtrip, "custom").unwrap();
        assert_eq!(provenance["modelId"], "custom");
        assert_eq!(provenance["baseLineage"]["baseModel"], "ltx-base");
        plan.context.get_mut(MODEL_DECLARATION_KEY).unwrap()["baseLineage"]["baseModel"] =
            json!("another-base");
        assert_ne!(plan.semantic_digest().unwrap(), committed);
    }

    #[test]
    fn every_asset_kind_replaces_worker_claims_and_absent_evidence_stays_absent() {
        let kinds = ["image", "video", "audio", "document", "vector", "depth"];
        let mut facts: Vec<_> = kinds.iter().map(|kind| json!({"type":kind,"modelProvenance":{"forged":true},
            "extra":{"modelProvenance":{"forged":true}}, "rawAdapterSettings":{"modelProvenance":{}}})).collect();
        let trusted = checkpoint_model_provenance(&plan(), "actual-model").unwrap();
        stamp_asset_provenance(&mut facts, Some(&trusted));
        for fact in &facts {
            assert_eq!(fact[MODEL_PROVENANCE_KEY], trusted);
            assert!(fact["extra"].get(MODEL_PROVENANCE_KEY).is_none());
            assert!(fact["rawAdapterSettings"]
                .get(MODEL_PROVENANCE_KEY)
                .is_none());
        }
        stamp_asset_provenance(&mut facts, None);
        assert!(facts
            .iter()
            .all(|fact| fact.get(MODEL_PROVENANCE_KEY).is_none()));
    }

    #[test]
    fn new_h3_declaration_retains_trusted_component_obligations_not_author_flags() {
        let entry = json!({"family":"minimax-h3","baseModel":"minimax_h3_ref",
            "minimaxH3Partition":"ref2va","license":"MIT","requiresLicenseAcknowledgment":false,
            "inheritedLicenseAcknowledgments":[]});
        let context = import_declaration_context(entry.as_object().unwrap()).unwrap();
        let declaration = &context[MODEL_DECLARATION_KEY];
        let components = declaration["inheritedLicenseAcknowledgments"]
            .as_array()
            .unwrap();
        assert!(components
            .iter()
            .any(|component| component["modelId"] == "minimax_h3"
                && component["requiresLicenseAcknowledgment"] == true));
        assert_eq!(declaration["baseLineage"]["baseModel"], "minimax_h3_ref");
        assert_eq!(
            declaration["declaredSemantics"]["minimaxH3Partition"],
            "ref2va"
        );
    }

    #[test]
    fn catalog_and_plan_tampering_cannot_relabel_an_immutable_declaration() {
        use crate::checkpoint_import::{CheckpointCatalogRecordV1, CheckpointInventoryV1};
        let dir = tempfile::tempdir().unwrap();
        let mut plan = plan();
        plan.context =
            import_declaration_context(json!({"baseModel":"declared-base"}).as_object().unwrap())
                .unwrap();
        let checkpoint = "linked/root-test/model.safetensors";
        let record = CheckpointCatalogRecordV1::from_plan(checkpoint, &plan).unwrap();
        let inventory = CheckpointInventoryV1::new(vec![record.clone()]).unwrap();
        let plans_dir = dir.path().join("checkpoints/plans");
        std::fs::create_dir_all(&plans_dir).unwrap();
        let plan_path = plans_dir.join(format!("{}.json", plan.plan_id));
        std::fs::write(&plan_path, plan.canonical_json().unwrap()).unwrap();
        std::fs::write(
            dir.path().join("checkpoints/inventory.json"),
            inventory.canonical_json().unwrap(),
        )
        .unwrap();
        let mut entry = json!({"id":"custom","baseModel":"declared-base","importPlan":{
            "checkpointId":checkpoint,"planId":plan.plan_id,"semanticDigest":record.plan.semantic_digest}});
        assert!(
            resolved_model_provenance(dir.path(), entry.as_object().unwrap())
                .unwrap()
                .is_some()
        );
        entry["baseModel"] = json!("stock-replacement");
        assert!(
            resolved_model_provenance(dir.path(), entry.as_object().unwrap())
                .unwrap_err()
                .contains("immutable import declaration")
        );
        entry["baseModel"] = json!("declared-base");
        plan.context.get_mut(MODEL_DECLARATION_KEY).unwrap()["baseLineage"]["baseModel"] =
            json!("stock-replacement");
        std::fs::write(&plan_path, plan.canonical_json().unwrap()).unwrap();
        assert!(resolved_model_provenance(dir.path(), entry.as_object().unwrap()).is_err());
        assert_eq!(
            resolved_model_provenance(
                dir.path(),
                json!({"id":"pre-plan-import"}).as_object().unwrap()
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn job_provenance_migration_stays_unknown_and_frozen_binding_survives_restart() {
        use crate::jobs_store::{CreateJob, JobsStore};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("jobs.db");
        let store = JobsStore::new(&path);
        store.initialize().unwrap();
        let job = store
            .create_job(CreateJob {
                job_type: crate::contracts::JobType::ImageGenerate,
                project_id: None,
                project_name: None,
                payload: Map::new(),
                requested_gpu: "auto".into(),
                source_job_id: None,
                duplicate_of_job_id: None,
                attempts: 1,
                initial_status: None,
            })
            .unwrap();
        drop(store);
        let connection = rusqlite::Connection::open(&path).unwrap();
        connection
            .execute("alter table jobs drop column model_provenance_json", [])
            .unwrap();
        drop(connection);
        let store = JobsStore::new(&path);
        store.initialize().unwrap();
        assert_eq!(store.model_provenance(&job.id).unwrap(), None);
        let provenance = checkpoint_model_provenance(&plan(), "custom").unwrap();
        let legacy_id = job.id;
        let job = store
            .create_job_with_model_provenance(
                CreateJob {
                    job_type: crate::contracts::JobType::ModelDownload,
                    project_id: None,
                    project_name: None,
                    payload: json!({"modelProvenance":{"caller_forged":true}})
                        .as_object()
                        .unwrap()
                        .clone(),
                    requested_gpu: "cpu".into(),
                    source_job_id: None,
                    duplicate_of_job_id: None,
                    attempts: 1,
                    initial_status: None,
                },
                Some(&provenance),
            )
            .unwrap();
        assert_eq!(job.extra[MODEL_PROVENANCE_KEY], provenance);
        assert_eq!(
            job.extra["revision"],
            store.get_job(&job.id).unwrap().extra["revision"]
        );
        let mut forged = provenance.clone();
        forged["sources"][0]["sha256"] = json!("b".repeat(64));
        store
            .register_worker(crate::jobs_store::RegisterWorker {
                worker_id: "provenance-cpu-worker".into(),
                gpu_id: "cpu".into(),
                gpu_name: None,
                capabilities: vec![crate::contracts::WorkerCapability::ModelDownload],
                loaded_models: Vec::new(),
                utilization: None,
            })
            .unwrap();
        let claimed = store
            .claim_next_job("provenance-cpu-worker")
            .unwrap()
            .unwrap();
        assert_eq!(claimed.id, job.id);
        let updated = store
            .update_job_progress(
                &job.id,
                crate::jobs_store::ProgressUpdate {
                    status: crate::contracts::JobStatus::Saving,
                    stage: crate::contracts::ProgressStage::Saving,
                    progress: 0.75,
                    message: "CPU provenance boundary".into(),
                    error: None,
                    result: Some(
                        json!({"modelProvenance":forged})
                            .as_object()
                            .unwrap()
                            .clone(),
                    ),
                    eta_seconds: None,
                    peak_gpu_memory_pct: None,
                    peak_gpu_load_pct: None,
                    backend: Some("cpu".into()),
                    worker_id: Some("provenance-cpu-worker".into()),
                },
            )
            .unwrap();
        assert_eq!(updated.result[MODEL_PROVENANCE_KEY], provenance);
        assert_eq!(
            store.model_provenance(&job.id).unwrap(),
            Some(provenance.clone())
        );
        assert_eq!(store.model_provenance(&legacy_id).unwrap(), None);
        let retry = store
            .retry_job(
                &job.id,
                crate::jobs_store::RetryJob {
                    payload_changes: Map::new(),
                },
            )
            .unwrap();
        assert_eq!(
            store.model_provenance(&retry.id).unwrap(),
            Some(provenance.clone())
        );
        let duplicate = store
            .duplicate_job(
                &job.id,
                crate::jobs_store::DuplicateJob {
                    payload_changes: json!({"modelProvenance":forged})
                        .as_object()
                        .unwrap()
                        .clone(),
                    requested_gpu: None,
                },
            )
            .unwrap();
        assert_eq!(
            store.model_provenance(&duplicate.id).unwrap(),
            Some(provenance.clone())
        );
        let different_model = store
            .duplicate_job(
                &job.id,
                crate::jobs_store::DuplicateJob {
                    payload_changes: json!({"model":"other-model"}).as_object().unwrap().clone(),
                    requested_gpu: None,
                },
            )
            .unwrap();
        assert_eq!(store.model_provenance(&different_model.id).unwrap(), None);
        drop(store);
        let restarted = JobsStore::new(&path);
        restarted.initialize().unwrap();
        assert_eq!(
            restarted.model_provenance(&job.id).unwrap(),
            Some(provenance)
        );
    }

    #[test]
    fn managed_source_revision_and_hash_survive_without_claiming_linked_ownership() {
        use crate::checkpoint_import::ManagedProvenanceV1;
        let linked = plan();
        let mut managed = linked.clone();
        let revision = "1234567890abcdef1234567890abcdef12345678";
        let mut origin = ManagedProvenanceV1::of_source("huggingface");
        origin.version_id = Some(revision.into());
        origin.reference = Some(format!("author/fine-tune@{revision}"));
        origin.file_id = Some("model.safetensors".into());
        managed.layers[0].source =
            SourceLocatorV1::managed("my-install", "model.safetensors", "a".repeat(64), origin)
                .unwrap();
        let linked = checkpoint_model_provenance(&linked, "user-id").unwrap();
        let managed = checkpoint_model_provenance(&managed, "user-id").unwrap();
        assert_eq!(
            managed["sources"][0]["source"]["provenance"]["versionId"],
            revision
        );
        assert_eq!(
            managed["sources"][0]["sha256"],
            linked["sources"][0]["sha256"]
        );
        assert_eq!(
            managed["plan"]["semanticDigest"],
            linked["plan"]["semanticDigest"]
        );
        assert_ne!(
            managed["plan"]["sourceBindingIdentity"],
            linked["plan"]["sourceBindingIdentity"]
        );
        assert_eq!(managed["sources"][0]["source"]["kind"], "managed");
        assert_eq!(linked["sources"][0]["source"]["kind"], "linked");
        assert_eq!(managed["modelId"], "user-id");
    }

    #[test]
    fn provenance_schema_accepts_legacy_unknowns_but_rejects_invalid_hashes_and_versions() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../packages/schemas/model-manifest.schema.json");
        let schema: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let schema = json!({"$defs":schema["$defs"],"$ref":"#/$defs/modelProvenance"});
        let validator = jsonschema::validator_for(&schema).unwrap();
        let mut value = checkpoint_model_provenance(&plan(), "legacy-import").unwrap();
        assert!(validator.is_valid(&value));
        value["sources"][0]["sha256"] = json!("mutable-main");
        assert!(!validator.is_valid(&value));
        value["sources"][0]["sha256"] = json!("a".repeat(64));
        value["schemaVersion"] = json!(2);
        assert!(!validator.is_valid(&value));
    }

    #[test]
    fn recomputing_mutable_assembly_cannot_change_a_bound_h3_partition() {
        use crate::minimax_h3_import::{
            config_binding, stamp_imported_minimax_h3_entry, verify_entry_assembly,
            MiniMaxH3ImportAssembly,
        };
        let (mut plan, mut entry) = frozen_h3_plan();
        entry.insert("id".into(), json!("my-import"));
        entry.insert("baseModel".into(), json!("minimax_h3_ref"));
        entry.insert("minimaxH3Partition".into(), json!("ref2va"));
        let closure = plan.context[MODEL_DECLARATION_KEY]["componentClosure"].clone();
        plan.context = import_declaration_context(&entry).unwrap();
        plan.context.get_mut(MODEL_DECLARATION_KEY).unwrap()["componentClosure"] = closure;
        plan.context.insert(
            "transformerConfig".into(),
            config_binding(
                crate::minimax_h3_codec::H3Config::published().0,
                json!({"kind":"source"}),
            ),
        );
        entry.insert(
            "importPlan".into(),
            json!({"planId":plan.plan_id,"semanticDigest":plan.semantic_digest().unwrap()}),
        );
        let assembly = MiniMaxH3ImportAssembly::from_declared_plan(&plan, &entry).unwrap();
        stamp_imported_minimax_h3_entry(&mut entry, &assembly);
        verify_entry_assembly(&entry, &plan).unwrap();
        entry.insert("baseModel".into(), json!("minimax_h3"));
        entry.insert("minimaxH3Partition".into(), json!("hybrid"));
        let forged = MiniMaxH3ImportAssembly::from_declared_plan(&plan, &entry).unwrap();
        stamp_imported_minimax_h3_entry(&mut entry, &forged);
        assert!(verify_entry_assembly(&entry, &plan)
            .unwrap_err()
            .contains("immutable import declaration"));
    }

    #[test]
    fn declaration_schema_and_plan_serde_agree_without_closing_unrelated_context() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../packages/schemas/checkpoint-import.schema.json");
        let schema: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let validator = jsonschema::validator_for(&schema).unwrap();
        let mut document = serde_json::to_value(plan()).unwrap();
        assert!(validator.is_valid(&document));
        let valid = json!({"schemaVersion":1,"baseLineage":{},"declaredSemantics":{},"inheritedLicenseAcknowledgments":[]});
        for (key, value) in [
            ("schemaVersion", json!(2)),
            ("baseLineage", json!([])),
            ("declaredSemantics", json!(null)),
            ("inheritedLicenseAcknowledgments", json!([false])),
            ("schemaVersion", json!(1.0)),
        ] {
            let mut declaration = valid.clone();
            declaration[key] = value;
            document["context"] = json!({"modelDeclaration":declaration,"unrelatedLoaderContext":{"opaque":"retained"}});
            assert_eq!(
                serde_json::from_value::<ImportPlanV1>(document.clone()).is_ok(),
                validator.is_valid(&document),
                "{key}: {document}"
            );
        }
        let decoded: ImportPlanV1 = serde_json::from_value(document).unwrap();
        assert_eq!(
            decoded.context["unrelatedLoaderContext"],
            json!({"opaque":"retained"})
        );
    }
}
