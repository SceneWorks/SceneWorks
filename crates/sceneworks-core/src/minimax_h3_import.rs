//! MiniMax-H3 imported checkpoint semantics (epic 26679, sc-26682).
//!
//! Architecture recognition is not task compatibility: FL2VA and Ref2VA have identical tensor
//! shapes. The existing import workflow must record an exact compatible `baseModel` and an explicit
//! `minimaxH3Partition` (`fl2va`, `ref2va`, or `hybrid`). Hybrid is an explicit promise that the
//! imported DiT supports both task partitions, not permission to combine keyframes and references.
//! Every supported operation denoises with the imported bytes; resident components are named
//! load-time companions only. The worker revalidates this binding before loading.

use serde_json::{json, Map, Value};

use crate::checkpoint_import::{CheckpointContainerV1, ImportPlanV1};

/// The catalog family token of every MiniMax-H3 checkpoint, builtin or imported.
pub const MINIMAX_H3_IMPORT_FAMILY: &str = "minimax-h3";
/// The provider-facing `importSourceShape`: a canonical transformer component directory.
/// Native single files reach that shape through the shared content-addressed derivative cache.
pub const MINIMAX_H3_IMPORT_SOURCE_SHAPE: &str = "transformer_directory";
/// Version of the persisted `importPlan.assembly` document.
pub const MINIMAX_H3_ASSEMBLY_SCHEMA_VERSION: u64 = 1;
/// The builtin catalog entry whose partition — and whose generation surface — an imported dense
/// DiT takes over. Used for SEMANTICS only (mode set, frame lattice, geometry); the request keeps
/// the imported model id and the provider loads the imported bytes.
pub const MINIMAX_H3_BASE_PARTITION_MODEL: &str = "minimax_h3";
/// The builtin catalog entry that owns the reference DiT partition.
pub const MINIMAX_H3_REFERENCE_PARTITION_MODEL: &str = "minimax_h3_ref";

/// The inspector role the imported DiT layer carries.
const TRANSFORMER_ROLE: &str = "transformer";

/// The shared components an imported dense H3 DiT is assembled with, and the builtin install each
/// is reused from. Order is the persisted order.
///
/// The opposite DiT is a load-time companion only for single-partition imports. The assembly
/// overrides this row for Ref2VA or hybrid explicitly; it is never a stock operation fallback.
pub const MINIMAX_H3_RESIDENT_COMPONENTS: &[(&str, &str)] = &[
    ("text_encoder", MINIMAX_H3_BASE_PARTITION_MODEL),
    ("tokenizer", MINIMAX_H3_BASE_PARTITION_MODEL),
    ("vae", MINIMAX_H3_BASE_PARTITION_MODEL),
    ("audio_vae", MINIMAX_H3_BASE_PARTITION_MODEL),
    ("transformer_ref", MINIMAX_H3_REFERENCE_PARTITION_MODEL),
];

/// Task compatibility declared independently of architecture and tensor shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiniMaxH3Partition {
    Fl2va,
    Ref2va,
    Hybrid,
}

impl MiniMaxH3Partition {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fl2va => "fl2va",
            Self::Ref2va => "ref2va",
            Self::Hybrid => "hybrid",
        }
    }

    pub fn supports_base(self) -> bool {
        matches!(self, Self::Fl2va | Self::Hybrid)
    }

    pub fn supports_references(self) -> bool {
        matches!(self, Self::Ref2va | Self::Hybrid)
    }

    pub fn from_entry(entry: &Map<String, Value>) -> Result<Self, String> {
        let partition = match entry_str(entry, "minimaxH3Partition") {
            Some("fl2va") => Self::Fl2va,
            Some("ref2va") => Self::Ref2va,
            Some("hybrid") => Self::Hybrid,
            _ => return Err("MiniMax-H3 task compatibility is ambiguous: explicitly declare minimaxH3Partition fl2va, ref2va, or hybrid; shapes and directory names do not prove compatibility".to_owned()),
        };
        let expected = if partition == Self::Ref2va {
            MINIMAX_H3_REFERENCE_PARTITION_MODEL
        } else {
            MINIMAX_H3_BASE_PARTITION_MODEL
        };
        if entry_str(entry, "baseModel") != Some(expected) {
            return Err(format!("MiniMax-H3 {} requires exact compatible baseModel {expected}; incompatible or missing base declaration", partition.as_str()));
        }
        Ok(partition)
    }
}

/// H3 compatibility is also an architecture constraint, even when `family` was omitted.
pub fn entry_requires_minimax_h3_family(entry: &Map<String, Value>) -> bool {
    entry_str(entry, "family") == Some(MINIMAX_H3_IMPORT_FAMILY)
        || entry
            .get("minimaxH3Partition")
            .is_some_and(|value| !value.is_null())
        || matches!(
            entry_str(entry, "baseModel"),
            Some(MINIMAX_H3_BASE_PARTITION_MODEL | MINIMAX_H3_REFERENCE_PARTITION_MODEL)
        )
}

/// Enforce compatibility against source evidence, not against a caller-supplied family stamp.
pub fn validate_declared_source_family(
    entry: &Map<String, Value>,
    detected: Option<&str>,
) -> Result<(), String> {
    if entry_requires_minimax_h3_family(entry) && detected != Some(MINIMAX_H3_IMPORT_FAMILY) {
        return Err(format!("MiniMax-H3 compatibility declaration requires detected source family {MINIMAX_H3_IMPORT_FAMILY}; got {}", detected.unwrap_or("unknown")));
    }
    Ok(())
}

/// A complete configuration snapshot, not a foreign file locator. Source-owned documents are
/// sealed by the plan's source digests; selected-base fallback records its exact download identity.
pub fn config_binding(config: Value, provenance: Value) -> Value {
    use sha2::{Digest, Sha256};
    let digest = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&config).expect("JSON config"))
    );
    json!({"config":config,"sha256":digest,"provenance":provenance})
}

pub fn bound_config(
    context: &std::collections::BTreeMap<String, Value>,
) -> Result<Option<crate::minimax_h3_codec::H3Config>, String> {
    let Some(binding) = context.get("transformerConfig") else {
        return Ok(None);
    };
    let config = binding
        .get("config")
        .filter(|v| v.is_object())
        .ok_or("bound transformer configuration is missing")?;
    let provenance = binding
        .get("provenance")
        .ok_or("bound configuration provenance is missing")?;
    match provenance.get("kind").and_then(Value::as_str) {
        Some("source") => {}
        Some("selected-base") => {
            for field in ["baseModel", "repo", "revision", "path"] {
                if provenance
                    .get(field)
                    .and_then(Value::as_str)
                    .is_none_or(|v| v.trim().is_empty())
                {
                    return Err(format!("bound configuration provenance is missing {field}"));
                }
            }
            let revision = provenance["revision"].as_str().expect("checked");
            if revision.len() != 40 || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("bound configuration requires an exact commit revision".into());
            }
            crate::checkpoint_plan_store::portable_relative_path_parts(
                provenance["path"].as_str().expect("checked"),
            )?;
        }
        _ => return Err("unknown bound configuration provenance".into()),
    }
    if config_binding(config.clone(), provenance.clone()).get("sha256") != binding.get("sha256") {
        return Err("bound configuration digest mismatch".into());
    }
    Ok(Some(crate::minimax_h3_codec::H3Config(config.clone())))
}

/// Resolve task semantics only; the returned stock id must NEVER be used as weight identity.
pub fn entry_partition_model(
    entry: &Map<String, Value>,
    has_references: bool,
) -> Result<&'static str, String> {
    let partition = MiniMaxH3Partition::from_entry(entry)?;
    match has_references {
        true if partition.supports_references() => Ok(MINIMAX_H3_REFERENCE_PARTITION_MODEL),
        false if partition.supports_base() => Ok(MINIMAX_H3_BASE_PARTITION_MODEL),
        _ => Err(format!(
            "declared {} checkpoint does not support the requested operation",
            partition.as_str()
        )),
    }
}

/// A validated dense MiniMax-H3 import assembly — see the module docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniMaxH3ImportAssembly {
    pub partition: MiniMaxH3Partition,
    /// The plan layer that IS the imported DiT.
    pub transformer_layer_id: String,
    /// Every safetensors layer in the DiT inventory, sorted by plan layer id. The primary is the
    /// first member; shard completeness and tensor mapping are validated by the source codec.
    pub transformer_layer_ids: Vec<String>,
    /// Checkpoint-relative source directory (`""` for the checkpoint root). Native sources are
    /// normalized into a separate provider directory without changing this binding.
    pub transformer_dir: String,
    /// Bound JSON descriptors (`config.json` and any shard index) in the source directory.
    pub descriptor_layer_ids: Vec<String>,
    /// Native encoding proven by the complete inventory, never by an approximate header hint.
    pub source_encoding: Option<String>,
    /// Complete source table/basis identity and source/compute precision geometry, when pruned.
    pub pruning: Option<crate::minimax_h3_codec::H3PruningBinding>,
}

impl MiniMaxH3ImportAssembly {
    /// Validate `plan` as a complete dense H3 DiT assembly, or say exactly why it is not one.
    pub fn from_declared_plan(
        plan: &ImportPlanV1,
        entry: &Map<String, Value>,
    ) -> Result<Self, String> {
        if plan.family != MINIMAX_H3_IMPORT_FAMILY {
            return Err(format!(
                "plan {:?} compiles to the {:?} family, not {MINIMAX_H3_IMPORT_FAMILY:?}",
                plan.plan_id, plan.family
            ));
        }
        let transformers: Vec<_> = plan
            .layers
            .iter()
            .filter(|layer| layer.role == TRANSFORMER_ROLE)
            .collect();
        let transformer = transformers.first().copied().ok_or_else(|| format!(
            "plan {:?} carries no {TRANSFORMER_ROLE:?} layer, so there is no MiniMax-H3 DiT to bind", plan.plan_id
        ))?;
        for layer in &transformers {
            if layer.container != CheckpointContainerV1::Safetensors {
                return Err(format!("plan {:?} DiT layer {:?} is a {:?} container; MiniMax-H3 providers open safetensors only", plan.plan_id, layer.target_path, layer.container));
            }
        }
        let transformer_dir = parent_dir(&transformer.target_path).to_owned();
        let mut descriptor_layer_ids = Vec::new();
        for layer in &plan.layers {
            if transformers
                .iter()
                .any(|source| source.layer_id == layer.layer_id)
            {
                if parent_dir(&layer.target_path) != transformer_dir {
                    return Err(format!(
                        "unconsumed H3 transformer {:?}: all shards must share a directory",
                        layer.target_path
                    ));
                }
                continue;
            }
            if layer.container != CheckpointContainerV1::JsonDescriptor
                || parent_dir(&layer.target_path) != transformer_dir
            {
                return Err(format!(
                    "plan {:?} carries a {:?} layer ({:?}, {:?}) that a dense MiniMax-H3 DiT \
                     import does not consume; its text encoder, tokenizer and VAEs are reused from \
                     the installed MiniMax-H3 model",
                    plan.plan_id, layer.role, layer.target_path, layer.container
                ));
            }
            descriptor_layer_ids.push(layer.layer_id.clone());
        }
        descriptor_layer_ids.sort();
        if transformers.len() > 1
            && !plan.layers.iter().any(|layer| {
                layer.container == CheckpointContainerV1::JsonDescriptor
                    && layer.target_path.rsplit('/').next()
                        == Some(crate::minimax_h3_codec::H3_INDEX_FILE)
            })
        {
            return Err(format!(
                "plan {:?} carries {} {TRANSFORMER_ROLE:?} layers without a shard index",
                plan.plan_id,
                transformers.len()
            ));
        }
        let mut transformer_layer_ids: Vec<_> =
            transformers.iter().map(|l| l.layer_id.clone()).collect();
        transformer_layer_ids.sort();
        let source_encoding = match plan.context.get("sourceEncoding") {
            None => None,
            Some(Value::String(value))
                if matches!(value.as_str(), "convrot_int8" | "asym_w4a8_int8") =>
            {
                Some(value.clone())
            }
            Some(_) => return Err("unknown MiniMax-H3 source encoding in plan context".into()),
        };
        let pruning = plan
            .context
            .get("adalnPruning")
            .map(crate::minimax_h3_codec::H3PruningBinding::from_value)
            .transpose()?;
        Ok(Self {
            partition: MiniMaxH3Partition::from_entry(entry)?,
            transformer_layer_id: transformer.layer_id.clone(),
            transformer_layer_ids,
            transformer_dir,
            descriptor_layer_ids,
            source_encoding,
            pruning,
        })
    }

    /// The persisted binding, stamped at `importPlan.assembly`.
    pub fn to_json(&self) -> Value {
        let mut components = Map::new();
        components.insert(
            TRANSFORMER_ROLE.to_owned(),
            json!({
                "source": "plan",
                "layerId": self.transformer_layer_id,
                "layerIds": self.transformer_layer_ids,
                "directory": self.transformer_dir,
                "descriptorLayerIds": self.descriptor_layer_ids,
            }),
        );
        for (component, model) in MINIMAX_H3_RESIDENT_COMPONENTS {
            if *component == "transformer_ref" {
                if self.partition == MiniMaxH3Partition::Hybrid {
                    components.insert(
                        "transformer_ref".to_owned(),
                        components[TRANSFORMER_ROLE].clone(),
                    );
                    continue;
                }
                if self.partition == MiniMaxH3Partition::Ref2va {
                    let imported = components.remove(TRANSFORMER_ROLE).expect("inserted DiT");
                    components.insert("transformer_ref".to_owned(), imported);
                    components.insert(TRANSFORMER_ROLE.to_owned(), json!({"source":"resident", "model":MINIMAX_H3_BASE_PARTITION_MODEL, "purpose":"load_only"}));
                    continue;
                }
            }
            components.insert((*component).to_owned(), json!({ "source": "resident", "model": model, "purpose": if *component == "transformer_ref" { "load_only" } else { "shared" } }));
        }
        let mut binding = json!({
            "schemaVersion": MINIMAX_H3_ASSEMBLY_SCHEMA_VERSION,
            "family": MINIMAX_H3_IMPORT_FAMILY,
            "partition": self.partition.as_str(),
            "components": components,
        });
        if let Some(encoding) = &self.source_encoding {
            binding["sourceEncoding"] = json!(encoding);
        }
        if let Some(pruning) = &self.pruning {
            binding["adalnPruning"] = pruning.to_json();
        }
        binding
    }
}

fn parent_dir(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(parent, _)| parent)
}

fn entry_str<'a>(entry: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    entry
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

/// The stamped `importPlan.assembly` of a catalog entry, if any.
pub fn entry_assembly(entry: &Map<String, Value>) -> Option<&Map<String, Value>> {
    entry
        .get("importPlan")
        .and_then(Value::as_object)
        .and_then(|plan| plan.get("assembly"))
        .and_then(Value::as_object)
}

/// Whether `entry` is an imported, plan-backed dense MiniMax-H3 DiT with a stamped assembly — the
/// one discriminator the scheduler's claim, the catalog surface and the worker route all key on.
///
/// Keyed entirely on the entry, never on the model id: an imported checkpoint's id is the user's,
/// and admitting a BUILTIN id here is refused separately by every caller (a builtin id names the
/// shipped weights, so a plan riding under it would render one checkpoint under another's name).
pub fn entry_is_imported_minimax_h3(entry: &Map<String, Value>) -> bool {
    entry_str(entry, "family") == Some(MINIMAX_H3_IMPORT_FAMILY)
        && entry_str(entry, "importSourceShape") == Some(MINIMAX_H3_IMPORT_SOURCE_SHAPE)
        && crate::jobs_store::checkpoint_plan_checkpoint_id(entry).is_some()
        && entry_assembly(entry).is_some_and(|assembly| {
            assembly.get("schemaVersion").and_then(Value::as_u64)
                == Some(MINIMAX_H3_ASSEMBLY_SCHEMA_VERSION)
                && MiniMaxH3Partition::from_entry(entry).is_ok_and(|partition| {
                    assembly.get("partition").and_then(Value::as_str) == Some(partition.as_str())
                })
        })
}

/// Re-derive the assembly from the plan the store resolved and require it to equal the binding the
/// entry carries. The render-time half of the import's validation: a stamped binding that names a
/// different DiT layer, directory or component source than its plan now proves is refused.
pub fn verify_entry_assembly(
    entry: &Map<String, Value>,
    plan: &ImportPlanV1,
) -> Result<MiniMaxH3ImportAssembly, String> {
    crate::model_artifacts::provenance::validate_entry_declaration(plan, entry)?;
    let assembly = MiniMaxH3ImportAssembly::from_declared_plan(plan, entry)?;
    let config = bound_config(&plan.context)?
        .ok_or("the import plan has no bound configuration; re-import the checkpoint")?;
    config.validate()?;
    if plan.context["transformerConfig"]["provenance"]["kind"] == "selected-base"
        && plan.context["transformerConfig"]["provenance"]["baseModel"] != entry["baseModel"]
    {
        return Err("bound configuration belongs to another compatible base".into());
    }
    if let Some(binding) = entry.get("importPlan").and_then(Value::as_object) {
        if binding
            .get("planId")
            .and_then(Value::as_str)
            .is_some_and(|id| id != plan.plan_id)
            || binding
                .get("semanticDigest")
                .and_then(Value::as_str)
                .is_some_and(|digest| {
                    !plan
                        .semantic_digest()
                        .is_ok_and(|current| current == digest)
                })
        {
            return Err(
                "the catalog entry's plan identity changed; rescan or re-import the checkpoint"
                    .into(),
            );
        }
    }
    let Some(stamped) = entry_assembly(entry) else {
        return Err(
            "the catalog entry carries no importPlan.assembly binding; re-import the checkpoint"
                .to_owned(),
        );
    };
    if Value::Object(stamped.clone()) != assembly.to_json() {
        return Err(format!(
            "the catalog entry's importPlan.assembly no longer matches plan {:?}; re-import the \
             checkpoint",
            plan.plan_id
        ));
    }
    Ok(assembly)
}

/// Bind the plan's pruning declaration to the complete source inventory actually pinned for load.
/// Same-shaped foreign tables, reduced weights, folded biases or precision metadata cannot reuse it.
pub fn verify_imported_pruning(
    assembly: &MiniMaxH3ImportAssembly,
    inventory: &crate::minimax_h3_codec::H3SourceInventory,
) -> Result<(), String> {
    if assembly.source_encoding.as_deref() != inventory.source_encoding() {
        return Err("MiniMax-H3 source encoding no longer matches its import plan; rescan or re-import the checkpoint".into());
    }
    if assembly.pruning != inventory.pruning {
        return Err("MiniMax-H3 source pruning identity/topology no longer matches its import plan; rescan or re-import the checkpoint".into());
    }
    Ok(())
}

/// Stamp a compiled dense H3 DiT import onto its catalog entry: the video type, the source shape,
/// and the validated binding beside the persisted plan identity.
///
/// `importPlan` must already carry the checkpoint identity; this merges the assembly into it rather
/// than replacing it.
pub fn stamp_imported_minimax_h3_entry(
    entry: &mut Map<String, Value>,
    assembly: &MiniMaxH3ImportAssembly,
) {
    entry.insert(
        "minimaxH3Partition".to_owned(),
        json!(assembly.partition.as_str()),
    );
    entry.insert(
        "baseModel".to_owned(),
        json!(if assembly.partition == MiniMaxH3Partition::Ref2va {
            MINIMAX_H3_REFERENCE_PARTITION_MODEL
        } else {
            MINIMAX_H3_BASE_PARTITION_MODEL
        }),
    );
    if let Some(encoding) = &assembly.source_encoding {
        entry.insert("importQuantFormat".into(), json!(encoding));
    }
    entry.insert(
        "family".to_owned(),
        Value::String(MINIMAX_H3_IMPORT_FAMILY.to_owned()),
    );
    // Content decides the type: an H3 DiT renders video, whatever type the request guessed.
    entry.insert("type".to_owned(), Value::String("video".to_owned()));
    entry.insert(
        "importSourceShape".to_owned(),
        Value::String(MINIMAX_H3_IMPORT_SOURCE_SHAPE.to_owned()),
    );
    // Config-only provisioning is complete: its bytes now live in the immutable shared plan.
    // It is not a resident component and must not become an external loose-file dependency.
    if let Some(rows) = entry.get_mut("downloads").and_then(Value::as_array_mut) {
        rows.retain(|row| {
            row.get("componentId").and_then(Value::as_str) != Some("import_transformer_config")
        });
    }
    let import_plan = entry
        .entry("importPlan".to_owned())
        .or_insert_with(|| json!({}));
    if !import_plan.is_object() {
        *import_plan = json!({});
    }
    if let Some(import_plan) = import_plan.as_object_mut() {
        import_plan.insert("assembly".to_owned(), assembly.to_json());
    }
}

fn builtin_base_partition_entry() -> Option<&'static Map<String, Value>> {
    builtin_partition_entry(MINIMAX_H3_BASE_PARTITION_MODEL)
}

fn builtin_partition_entry(id: &str) -> Option<&'static Map<String, Value>> {
    crate::builtin_manifests::builtin_model_entry(id)
}

fn builtin_reference_partition_entry() -> Option<&'static Map<String, Value>> {
    builtin_partition_entry(MINIMAX_H3_REFERENCE_PARTITION_MODEL)
}

/// The persisted shared precision is independent of imported checkpoint precision. A caller may
/// explicitly confirm it, but neither readiness nor generation may rediscover another closure.
pub fn selected_import_component_tier(entry: &Map<String, Value>) -> Result<&'static str, String> {
    let tier = match entry_str(entry, "importComponentTier") {
        Some("q4") => "q4",
        Some("q8") => "q8",
        Some("bf16") => "bf16",
        _ => {
            return Err(
                "the imported checkpoint has no selected shared component tier; re-import it"
                    .into(),
            )
        }
    };
    if entry
        .get("downloads")
        .and_then(Value::as_array)
        .is_some_and(|rows| {
            rows.iter().any(|row| {
                row.get("variant")
                    .and_then(Value::as_str)
                    .is_some_and(|variant| variant != tier)
            })
        })
    {
        return Err("component bindings disagree with the selected shared tier".into());
    }
    Ok(tier)
}

/// Tighten co-requisite globs to the files H3 providers actually consume. A landed config or one
/// shard cannot make a partially downloaded component ready. Literal-only rows retain the shared
/// downloader's all-files check; provider configuration compatibility is checked again at load.
pub fn import_component_files_complete(snapshot: &std::path::Path, row: &Value) -> bool {
    use crate::mlx_tier_completeness as tc;
    let first_pattern = row
        .get("files")
        .and_then(Value::as_array)
        .and_then(|files| {
            files
                .iter()
                .filter_map(Value::as_str)
                .find(|file| file.ends_with("/*"))
        });
    let subdir = row
        .get("subdir")
        .and_then(Value::as_str)
        .or_else(|| first_pattern.map(|file| file.trim_end_matches("/*")));
    let Some(subdir) = subdir else { return true };
    let Ok(parts) = crate::checkpoint_plan_store::portable_relative_path_parts(subdir) else {
        return false;
    };
    let dir = snapshot.join(parts);
    let component = row
        .get("componentId")
        .and_then(Value::as_str)
        .or_else(|| dir.file_name().and_then(|name| name.to_str()))
        .unwrap_or_default();
    let json_present =
        |name: &str| crate::minimax_h3_codec::read_h3_descriptor(&dir.join(name)).is_ok();
    let indexed = |name: &str, omit_dense_tail: bool| {
        let Ok(index) = crate::minimax_h3_codec::read_h3_descriptor(&dir.join(name)) else {
            return false;
        };
        let Some(map) = index
            .get("weight_map")
            .and_then(Value::as_object)
            .filter(|map| !map.is_empty())
        else {
            return false;
        };
        map.values().all(|file| {
            file.as_str().is_some_and(|file| {
                if omit_dense_tail && file == "model-00013-of-00014.safetensors" {
                    return true;
                }
                crate::checkpoint_plan_store::portable_relative_path_parts(file)
                    .is_ok_and(|parts| dir.join(parts).is_file())
            })
        })
    };
    match component {
        "transformer" => dir.parent().is_some_and(tc::minimax_h3_tier_complete),
        "transformer_ref" => dir.parent().is_some_and(tc::minimax_h3_ref_tier_complete),
        "text_encoder" => {
            json_present("config.json")
                && indexed("model.safetensors.index.json", subdir == "text_encoder")
        }
        "tokenizer" => json_present("tokenizer.json") && json_present("tokenizer_config.json"),
        "video_vae" | "vae" | "audio_vae" => {
            json_present("config.json")
                && tc::dir_has_visible_file_ending(&dir, ".safetensors")
                && (!dir.join(crate::minimax_h3_codec::H3_INDEX_FILE).exists()
                    || indexed(crate::minimax_h3_codec::H3_INDEX_FILE, false))
        }
        "fl2va_audio_vae_config" => {
            tc::MINIMAX_H3_AUDIO_VAE_CONFIG_FILES
                .iter()
                .all(|file| dir.join(file).is_file())
                && (first_pattern.is_none()
                    || tc::dir_has_visible_file_ending(&dir, ".safetensors"))
        }
        _ => true,
    }
}

/// Check only the interfaces the imported DiT shares with resident components. Trunk geometry,
/// RoPE and schedules belong to the source architecture, not to a stock-base equality check.
pub fn import_component_configuration_compatible(
    snapshot: &std::path::Path,
    row: &Value,
    config: &Value,
) -> Result<(), String> {
    let subdir = row.get("subdir").and_then(Value::as_str).or_else(|| {
        row.get("files")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .find_map(|file| file.strip_suffix("/*"))
    });
    let component = row
        .get("componentId")
        .and_then(Value::as_str)
        .or_else(|| subdir.and_then(|dir| dir.rsplit('/').next()))
        .unwrap_or_default();
    let (dit_field, component_field) = match component {
        "text_encoder" => ("text_dim", "hidden_size"),
        "vae" | "video_vae" => ("in_channels", "latent_channels"),
        "audio_vae" => ("audio_in_channels", "latent_channels"),
        _ => return Ok(()),
    };
    let subdir = subdir
        .ok_or_else(|| format!("{component}: bound component has no configuration directory"))?;
    let parts = crate::checkpoint_plan_store::portable_relative_path_parts(subdir)?;
    let document =
        crate::minimax_h3_codec::read_h3_descriptor(&snapshot.join(parts).join("config.json"))?;
    let component_config = if component == "text_encoder" {
        document.get("text_config").unwrap_or(&document)
    } else {
        &document
    };
    let actual = component_config
        .get(component_field)
        .and_then(Value::as_u64)
        .filter(|v| *v > 0)
        .ok_or_else(|| format!("{component}/config.json has no positive {component_field}"))?;
    let required = config
        .get(dit_field)
        .and_then(Value::as_u64)
        .filter(|v| *v > 0)
        .ok_or_else(|| format!("bound transformer configuration has no positive {dit_field}"))?;
    if actual != required {
        return Err(format!("incompatible {component}: {component_field}={actual}, but bound transformer {dit_field}={required}"));
    }
    Ok(())
}

/// Select the explicitly compatible base's reusable download closure, never its primary DiT.
/// The caller selects a resident tier (reusing an installed tier before the catalog default).
/// Platform filtering uses the same catalog mechanism as ordinary model installation.
pub fn import_component_downloads(
    entry: &Map<String, Value>,
    base: &Value,
    tier: Option<&str>,
    os: &str,
) -> Result<Vec<Value>, String> {
    use crate::model_artifacts::artifact_selection::{
        model_co_requisite_downloads_for_variant, model_download_for_variant,
        retain_downloads_for_os,
    };
    let partition = MiniMaxH3Partition::from_entry(entry)?;
    if base.get("id").and_then(Value::as_str) != entry_str(entry, "baseModel")
        || base.get("family").and_then(Value::as_str) != Some(MINIMAX_H3_IMPORT_FAMILY)
    {
        return Err("the selected base does not supply the declared MiniMax-H3 contract".into());
    }
    let mut base = base.clone();
    retain_downloads_for_os(&mut base, os);
    if tier.is_some_and(|tier| model_download_for_variant(&base, tier).is_none()) {
        return Err(
            "the selected base does not declare this resident component tier on this platform"
                .into(),
        );
    }
    let mut rows = model_co_requisite_downloads_for_variant(&base, tier);
    rows.retain(|row| {
        row.get("required").and_then(Value::as_str) != Some("soft")
            && !(partition == MiniMaxH3Partition::Hybrid
                && row
                    .get("files")
                    .and_then(Value::as_array)
                    .is_some_and(|files| {
                        files.iter().filter_map(Value::as_str).any(|file| {
                            file.split('/')
                                .any(|part| matches!(part, "transformer" | "transformer_ref"))
                        })
                    }))
    });
    if rows.is_empty() {
        return Err("the selected base declares no reusable MiniMax-H3 components".into());
    }
    Ok(rows)
}

/// Derive the selected base's dense architecture configuration, not the shared resident tier's
/// packed-weight policy. This row is not a primary: completing it cannot install stock DiT.
pub fn import_config_download(base: &Value, tier: &str, os: &str) -> Result<Value, String> {
    use crate::model_artifacts::artifact_selection::{
        model_download_for_variant, retain_downloads_for_os,
    };
    let mut base = base.clone();
    retain_downloads_for_os(&mut base, os);
    let mut row = model_download_for_variant(&base, "bf16")
        .ok_or("selected base has no dense architecture download")?;
    let files = row
        .get("files")
        .and_then(Value::as_array)
        .ok_or("base primary has no file closure")?;
    let configs: std::collections::BTreeSet<_> = files
        .iter()
        .filter_map(Value::as_str)
        .filter_map(|path| {
            if path == "config.json" || path.ends_with("/config.json") {
                Some(path.to_owned())
            } else {
                path.strip_suffix("/*")
                    .map(|dir| format!("{dir}/config.json"))
            }
        })
        .collect();
    if configs.len() != 1 {
        return Err("selected base primary has no unambiguous configuration path".into());
    }
    let path = configs.into_iter().next().expect("one config");
    let revision = row
        .get("revision")
        .and_then(Value::as_str)
        .ok_or("base architecture configuration has no exact revision")?;
    if revision.len() != 40 || !revision.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("base architecture configuration requires an exact commit revision".into());
    }
    crate::checkpoint_plan_store::portable_relative_path_parts(&path)?;
    let object = row.as_object_mut().ok_or("invalid primary download")?;
    for field in [
        "default",
        "estimatedSizeBytes",
        "expectedSizeBytes",
        "expectedSha256",
        "footprint",
        "subdir",
    ] {
        object.remove(field);
    }
    object.insert("files".into(), json!([path]));
    object.insert("coRequisite".into(), json!(true));
    object.insert("componentId".into(), json!("import_transformer_config"));
    object.insert("variant".into(), json!(tier));
    Ok(row)
}

/// Give an imported H3 entry the generation surface its explicit task declaration supports.
///
/// The ROUTE-OWNED fields — `adapter`, `capabilities`, `video`, `defaults`, `limits` — are replaced,
/// not filled: they state what the provider serves for this partition (its modes, its 17n+5
/// duration lattice, its single fps, its step floor), and a value left over from an import request
/// that guessed the wrong type (an image entry's capabilities) would advertise operations nothing
/// can run. Descriptive metadata retains the author's wording; inherited component terms and
/// attribution are unioned afterward and cannot be erased by that wording or a false flag.
/// Deliberately NOT copied: the builtin's downloads and paths (the imported DiT is its own
/// weights), `mlx.repo` (the builtin tier repo), and `loraCompatibility` — adapter and turbo
/// support on imported H3 checkpoints is a later story, and advertising adapters the imported route
/// refuses would admit jobs nothing can run.
pub fn apply_minimax_h3_studio_surface_defaults(entry: &mut Map<String, Value>) {
    let Ok(partition) = MiniMaxH3Partition::from_entry(entry) else {
        entry.insert("capabilities".to_owned(), json!([]));
        return;
    };
    let base = if partition == MiniMaxH3Partition::Ref2va {
        builtin_reference_partition_entry()
    } else {
        builtin_base_partition_entry()
    };
    let Some(base) = base else {
        return;
    };
    for key in ["adapter", "capabilities", "video", "defaults", "limits"] {
        if let Some(value) = base.get(key) {
            entry.insert(key.to_owned(), value.clone());
        }
    }
    for key in [
        "candle",
        "requiresLicenseAcknowledgment",
        "licenseUrl",
        "licenseNotice",
    ] {
        if let Some(value) = base.get(key) {
            entry.entry(key.to_owned()).or_insert_with(|| value.clone());
        }
    }
    if let Some(base_mlx) = base.get("mlx").and_then(Value::as_object) {
        let mlx = entry.entry("mlx".to_owned()).or_insert_with(|| json!({}));
        if let Some(mlx) = mlx.as_object_mut() {
            for key in ["minMemoryGb", "limits"] {
                if let Some(value) = base_mlx.get(key) {
                    mlx.entry(key.to_owned()).or_insert_with(|| value.clone());
                }
            }
        }
    }
    if let Some(base_ui) = base.get("ui").and_then(Value::as_object) {
        let ui = entry.entry("ui".to_owned()).or_insert_with(|| json!({}));
        if let Some(ui) = ui.as_object_mut() {
            for key in [
                "attribution",
                "recommendedFor",
                "durationHint",
                "promptGuide",
            ] {
                if let Some(value) = base_ui.get(key) {
                    ui.entry(key.to_owned()).or_insert_with(|| value.clone());
                }
            }
        }
    }
    let license_components = inherited_component_entries(entry);
    crate::license_acknowledgments::apply_inherited_component_licenses(entry, &license_components);
    if partition == MiniMaxH3Partition::Hybrid {
        entry.insert(
            "capabilities".to_owned(),
            json!([
                "text_to_video",
                "image_to_video",
                "first_last_frame",
                "reference_to_video"
            ]),
        );
        if let Some(reference) = builtin_reference_partition_entry() {
            if let Some(limits) = reference.get("limits") {
                entry.insert("limits".to_owned(), limits.clone());
            }
        }
    }
    // Withdraw the family-wide adapter advertisement `apply_model_manifest_defaults` synthesizes.
    entry.remove("loraCompatibility");
}

/// Trusted license owners of the components reused by this declaration. These rows describe
/// component obligations, not ownership of the imported DiT. Reuse the existing builtin cache.
pub fn inherited_component_entries(entry: &Map<String, Value>) -> Vec<Value> {
    let Ok(partition) = MiniMaxH3Partition::from_entry(entry) else {
        return Vec::new();
    };
    let mut ids = std::collections::BTreeSet::from([MINIMAX_H3_BASE_PARTITION_MODEL]);
    if partition == MiniMaxH3Partition::Fl2va {
        ids.insert(MINIMAX_H3_REFERENCE_PARTITION_MODEL);
    }
    ids.into_iter()
        .filter_map(builtin_partition_entry)
        .map(crate::license_acknowledgments::component_license_metadata)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint_import::{ImportLayerV1, SourceLocatorV1};

    fn layer(id: &str, role: &str, path: &str, container: CheckpointContainerV1) -> ImportLayerV1 {
        ImportLayerV1 {
            layer_id: id.to_owned(),
            role: role.to_owned(),
            target_path: path.to_owned(),
            container,
            source: SourceLocatorV1::linked(
                "root-0123456789abcdef",
                format!("h3/{path}"),
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            )
            .expect("valid linked locator"),
        }
    }

    fn plan(family: &str, layers: Vec<ImportLayerV1>) -> ImportPlanV1 {
        let mut plan =
            ImportPlanV1::new("checkpoint-plan-test", family, layers).expect("valid plan");
        if family == MINIMAX_H3_IMPORT_FAMILY {
            plan.context.insert(
                "transformerConfig".into(),
                config_binding(
                    crate::minimax_h3_codec::H3Config::published().0,
                    json!({"kind":"source"}),
                ),
            );
        }
        plan
    }

    fn dense_dit_plan() -> ImportPlanV1 {
        plan(
            MINIMAX_H3_IMPORT_FAMILY,
            vec![
                layer(
                    "artifact:diffusion_pytorch_model.safetensors",
                    "transformer",
                    "diffusion_pytorch_model.safetensors",
                    CheckpointContainerV1::Safetensors,
                ),
                layer(
                    "artifact:config.json",
                    "descriptor",
                    "config.json",
                    CheckpointContainerV1::JsonDescriptor,
                ),
            ],
        )
    }

    #[test]
    fn identical_weights_require_explicit_compatible_task_declarations() {
        let plan = dense_dit_plan();
        for invalid in [
            json!({}),
            json!({"baseModel":"minimax_h3"}),
            json!({"baseModel":"wan_2_2","minimaxH3Partition":"fl2va"}),
            json!({"baseModel":"minimax_h3","minimaxH3Partition":"ref2va"}),
        ] {
            assert!(MiniMaxH3ImportAssembly::from_declared_plan(
                &plan,
                invalid.as_object().unwrap()
            )
            .is_err());
        }
        for (partition, base, operations) in [
            (
                "fl2va",
                "minimax_h3",
                vec!["text_to_video", "image_to_video", "first_last_frame"],
            ),
            ("ref2va", "minimax_h3_ref", vec!["reference_to_video"]),
            (
                "hybrid",
                "minimax_h3",
                vec![
                    "text_to_video",
                    "image_to_video",
                    "first_last_frame",
                    "reference_to_video",
                ],
            ),
        ] {
            let mut entry = json!({"baseModel":base,"minimaxH3Partition":partition,"importPlan":{"checkpointId":"fixture"}}).as_object().unwrap().clone();
            let assembly = MiniMaxH3ImportAssembly::from_declared_plan(&plan, &entry).unwrap();
            stamp_imported_minimax_h3_entry(&mut entry, &assembly);
            apply_minimax_h3_studio_surface_defaults(&mut entry);
            assert_eq!(entry["capabilities"], json!(operations));
            assert_eq!(
                entry_partition_model(&entry, false).is_ok(),
                partition != "ref2va"
            );
            assert_eq!(
                entry_partition_model(&entry, true).is_ok(),
                partition != "fl2va"
            );
            assert!(verify_entry_assembly(&entry, &plan).is_ok());
            let components = assembly.to_json()["components"].clone();
            for component in ["text_encoder", "tokenizer", "vae", "audio_vae"] {
                assert_eq!(components[component]["model"], "minimax_h3");
            }
            for component in ["transformer", "transformer_ref"] {
                let expected = if partition == "hybrid"
                    || (partition == "fl2va" && component == "transformer")
                    || (partition == "ref2va" && component == "transformer_ref")
                {
                    "plan"
                } else {
                    "resident"
                };
                assert_eq!(components[component]["source"], expected);
            }
            entry.insert("minimaxH3Partition".to_owned(), json!("undeclared"));
            assert!(!entry_is_imported_minimax_h3(&entry));
            assert!(verify_entry_assembly(&entry, &plan).is_err());
        }
    }

    #[test]
    fn a_dense_dit_directory_binds_the_plan_dit_and_every_resident_component() {
        let assembly = MiniMaxH3ImportAssembly::from_declared_plan(
            &dense_dit_plan(),
            serde_json::json!({"baseModel":"minimax_h3","minimaxH3Partition":"fl2va"})
                .as_object()
                .unwrap(),
        )
        .expect("complete");
        let json = assembly.to_json();
        assert_eq!(
            json["components"]["transformer"]["source"],
            json!("plan"),
            "the DiT is the IMPORTED layer, never a resident one: {json}"
        );
        assert_eq!(
            json["components"]["transformer"]["layerId"],
            json!("artifact:diffusion_pytorch_model.safetensors")
        );
        assert_eq!(json["partition"], json!("fl2va"));
        for (component, model) in MINIMAX_H3_RESIDENT_COMPONENTS {
            assert_eq!(
                json["components"][component],
                json!({ "source": "resident", "model": model, "purpose":if *component == "transformer_ref" { "load_only" } else { "shared" } }),
                "{component} must be an explicit resident binding"
            );
        }
        assert_eq!(
            json["components"].as_object().map(Map::len),
            Some(1 + MINIMAX_H3_RESIDENT_COMPONENTS.len())
        );
    }

    #[test]
    fn incomplete_or_ambiguous_assemblies_are_refused_by_name() {
        // No DiT at all.
        let no_dit = plan(
            MINIMAX_H3_IMPORT_FAMILY,
            vec![layer(
                "artifact:config.json",
                "descriptor",
                "config.json",
                CheckpointContainerV1::JsonDescriptor,
            )],
        );
        assert!(
            MiniMaxH3ImportAssembly::from_declared_plan(&no_dit, &Map::new())
                .unwrap_err()
                .contains("no \"transformer\" layer")
        );

        // A second weight file beside the DiT would be mapped unbound by `Weights::from_dir`.
        let mut sharded = dense_dit_plan();
        sharded.layers.push(layer(
            "artifact:extra.safetensors",
            "transformer",
            "extra.safetensors",
            CheckpointContainerV1::Safetensors,
        ));
        assert!(
            MiniMaxH3ImportAssembly::from_declared_plan(&sharded, &Map::new())
                .unwrap_err()
                .contains("2 \"transformer\" layers")
        );

        // An encoder carried by the import would be silently replaced by the resident one.
        let mut with_te = dense_dit_plan();
        with_te.layers.push(layer(
            "artifact:text_encoder/model.safetensors",
            "text_encoder",
            "text_encoder/model.safetensors",
            CheckpointContainerV1::Safetensors,
        ));
        assert!(
            MiniMaxH3ImportAssembly::from_declared_plan(&with_te, &Map::new())
                .unwrap_err()
                .contains("\"text_encoder\" layer")
        );

        // Another family's plan is never an H3 assembly.
        let wan = plan("wan-video", dense_dit_plan().layers);
        assert!(
            MiniMaxH3ImportAssembly::from_declared_plan(&wan, &Map::new())
                .unwrap_err()
                .contains("\"wan-video\" family")
        );
    }

    fn stamped_entry() -> Map<String, Value> {
        let mut entry = json!({
            "id": "my_h3_finetune",
            "type": "image",
            "importPlan": { "checkpointId": "managed/my_h3_finetune" },
        })
        .as_object()
        .cloned()
        .unwrap();
        let assembly = MiniMaxH3ImportAssembly::from_declared_plan(
            &dense_dit_plan(),
            serde_json::json!({"baseModel":"minimax_h3","minimaxH3Partition":"fl2va"})
                .as_object()
                .unwrap(),
        )
        .unwrap();
        stamp_imported_minimax_h3_entry(&mut entry, &assembly);
        entry
    }

    #[test]
    fn the_stamp_is_what_the_discriminator_and_the_render_time_check_read() {
        let entry = stamped_entry();
        assert_eq!(entry["type"], json!("video"), "content decides the type");
        assert_eq!(
            entry["importPlan"]["checkpointId"],
            json!("managed/my_h3_finetune"),
            "the stamp merges into the plan identity rather than replacing it"
        );
        assert!(entry_is_imported_minimax_h3(&entry));
        assert!(verify_entry_assembly(&entry, &dense_dit_plan()).is_ok());

        // Each discriminating field, removed alone, flips the verdict.
        for (path, mutate) in [
            (
                "family",
                Box::new(|e: &mut Map<String, Value>| {
                    e.insert("family".into(), json!("wan-video"));
                }) as Box<dyn Fn(&mut Map<String, Value>)>,
            ),
            (
                "importSourceShape",
                Box::new(|e: &mut Map<String, Value>| {
                    e.remove("importSourceShape");
                }),
            ),
            (
                "checkpointId",
                Box::new(|e: &mut Map<String, Value>| {
                    e["importPlan"]["checkpointId"] = json!("  ");
                }),
            ),
            (
                "assembly",
                Box::new(|e: &mut Map<String, Value>| {
                    e["importPlan"].as_object_mut().unwrap().remove("assembly");
                }),
            ),
        ] {
            let mut mutated = entry.clone();
            mutate(&mut mutated);
            assert!(
                !entry_is_imported_minimax_h3(&mutated),
                "removing {path} must withdraw the imported-H3 verdict"
            );
        }

        // A binding that names a different DiT layer than the plan proves is refused at render.
        let mut tampered = entry.clone();
        tampered["importPlan"]["assembly"]["components"]["transformer"]["layerId"] =
            json!("artifact:other.safetensors");
        assert!(verify_entry_assembly(&tampered, &dense_dit_plan())
            .unwrap_err()
            .contains("no longer matches"));
    }

    #[test]
    fn the_studio_surface_is_the_shipped_base_partitions_and_withdraws_adapters() {
        let mut entry = stamped_entry();
        // The residue an import request that guessed `type: "image"` leaves behind must not survive:
        // route-owned fields are the partition's, not the request's.
        entry.insert("capabilities".to_owned(), json!(["text_to_image"]));
        entry.insert("limits".to_owned(), json!({ "resolutions": ["1024x1024"] }));
        crate::lora_family::apply_model_manifest_defaults(
            &mut entry,
            "video",
            Some(MINIMAX_H3_IMPORT_FAMILY),
        );
        let base = builtin_base_partition_entry().expect("the shipped catalog carries minimax_h3");
        assert_eq!(entry["capabilities"], base["capabilities"]);
        assert_eq!(entry["limits"], base["limits"]);
        assert_eq!(entry["defaults"], base["defaults"]);
        assert_eq!(
            entry["capabilities"],
            json!(["text_to_video", "image_to_video", "first_last_frame"]),
            "an imported dense DiT serves the base partition's modes only"
        );
        assert!(
            entry.get("loraCompatibility").is_none(),
            "adapters are not advertised on imported H3 checkpoints yet"
        );
        assert!(
            entry["mlx"].get("repo").is_none(),
            "the builtin tier repo is not the imported checkpoint's source"
        );
        assert!(entry
            .get("downloads")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty));
    }
}
