//! Dense MiniMax-H3 fine-tune imports (epic 26679, sc-26680).
//!
//! A community H3 fine-tune is a new **DiT** and nothing else: dense canonical Diffusers weights
//! (single or indexed shards), or a native single file normalized by [`crate::minimax_h3_codec`]
//! into the canonical component directory. Every other component the native providers read —
//! the Qwen3-VL text encoder, the tokenizer, the video and audio VAEs with their `FL2VA` constructor
//! documents, and the reference DiT partition the MLX provider probes beside the base one — is the
//! resident MiniMax-H3 install's, reused rather than copied.
//!
//! This module is the one place that decides what such an import IS, so the importer that stamps
//! the catalog entry, the scheduler that claims its jobs and the worker that renders them cannot
//! disagree:
//!
//! * [`MiniMaxH3ImportAssembly::from_plan`] validates a persisted [`ImportPlanV1`] as a COMPLETE
//!   assembly — one dense DiT inventory (possibly several indexed shards), and no encoder/VAE
//!   the provider would replace with resident weights — and names every component's source.
//!   The inspector's tensor codec proves the inventory complete before an assembly is stamped.
//! * The stamped JSON ([`MiniMaxH3ImportAssembly::to_json`], persisted at `importPlan.assembly`) is
//!   the explicit, recorded binding: which layer is the DiT, which partition it serves, and which
//!   builtin install supplies each shared component. [`verify_entry_assembly`] re-derives it from the
//!   plan at render time, so an entry whose binding no longer matches its plan refuses rather than
//!   rendering something other than what was imported.
//! * [`apply_minimax_h3_studio_surface_defaults`] gives the imported entry the base partition's
//!   generation surface (modes, geometry, durations, fps, step floor) read from the SHIPPED builtin
//!   `minimax_h3` entry, so an imported DiT is offered exactly the operations the provider serves.
//!
//! Dense native/single-file/sharded layouts are handled by the task-neutral source codec
//! (sc-26681). Quantized execution, Ref2VA/hybrid operation policy and adapters are separate
//! epic slices; source shape never chooses task semantics.

use std::sync::LazyLock;

use serde_json::{json, Map, Value};

use crate::checkpoint_import::{CheckpointContainerV1, ImportPlanV1};
use crate::jsonc::strip_jsonc_comments;

/// The catalog family token of every MiniMax-H3 checkpoint, builtin or imported.
pub const MINIMAX_H3_IMPORT_FAMILY: &str = "minimax-h3";
/// The provider-facing `importSourceShape`: a canonical transformer component directory.
/// Native single files reach that shape through the shared content-addressed derivative cache.
pub const MINIMAX_H3_IMPORT_SOURCE_SHAPE: &str = "transformer_directory";
/// Version of the persisted `importPlan.assembly` document.
pub const MINIMAX_H3_ASSEMBLY_SCHEMA_VERSION: u64 = 1;
/// The partition an imported dense DiT serves: the base `transformer/` checkpoint (t2va + fl2va).
pub const MINIMAX_H3_ASSEMBLY_PARTITION: &str = "fl2va";
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
/// `transformer_ref` is the one non-obvious row. It is NEVER denoised for a base-partition request;
/// it is bound because `mlx-gen-minimax-h3::load` probes the reference DiT as the staged DiT's
/// sibling on every load (Ref2VA is a first-class task of the provider). Recording it here is what
/// makes that companion explicit rather than an incidental symlink.
pub const MINIMAX_H3_RESIDENT_COMPONENTS: &[(&str, &str)] = &[
    ("text_encoder", MINIMAX_H3_BASE_PARTITION_MODEL),
    ("tokenizer", MINIMAX_H3_BASE_PARTITION_MODEL),
    ("vae", MINIMAX_H3_BASE_PARTITION_MODEL),
    ("audio_vae", MINIMAX_H3_BASE_PARTITION_MODEL),
    ("transformer_ref", MINIMAX_H3_REFERENCE_PARTITION_MODEL),
];

/// A validated dense MiniMax-H3 import assembly — see the module docs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiniMaxH3ImportAssembly {
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
}

impl MiniMaxH3ImportAssembly {
    /// Validate `plan` as a complete dense H3 DiT assembly, or say exactly why it is not one.
    pub fn from_plan(plan: &ImportPlanV1) -> Result<Self, String> {
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
        Ok(Self {
            transformer_layer_id: transformer.layer_id.clone(),
            transformer_layer_ids,
            transformer_dir,
            descriptor_layer_ids,
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
            components.insert(
                (*component).to_owned(),
                json!({ "source": "resident", "model": model }),
            );
        }
        json!({
            "schemaVersion": MINIMAX_H3_ASSEMBLY_SCHEMA_VERSION,
            "family": MINIMAX_H3_IMPORT_FAMILY,
            "partition": MINIMAX_H3_ASSEMBLY_PARTITION,
            "components": components,
        })
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
                && assembly.get("partition").and_then(Value::as_str)
                    == Some(MINIMAX_H3_ASSEMBLY_PARTITION)
        })
}

/// Re-derive the assembly from the plan the store resolved and require it to equal the binding the
/// entry carries. The render-time half of the import's validation: a stamped binding that names a
/// different DiT layer, directory or component source than its plan now proves is refused.
pub fn verify_entry_assembly(
    entry: &Map<String, Value>,
    plan: &ImportPlanV1,
) -> Result<MiniMaxH3ImportAssembly, String> {
    let assembly = MiniMaxH3ImportAssembly::from_plan(plan)?;
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
        "family".to_owned(),
        Value::String(MINIMAX_H3_IMPORT_FAMILY.to_owned()),
    );
    // Content decides the type: an H3 DiT renders video, whatever type the request guessed.
    entry.insert("type".to_owned(), Value::String("video".to_owned()));
    entry.insert(
        "importSourceShape".to_owned(),
        Value::String(MINIMAX_H3_IMPORT_SOURCE_SHAPE.to_owned()),
    );
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

/// The SHIPPED builtin `minimax_h3` entry, parsed once from the embedded manifest — the same
/// authority `crate::minimax_h3_turbo` reads its recipes from, because the base partition's surface
/// is a property of the shipped catalog rather than of the host.
static BUILTIN_BASE_PARTITION_ENTRY: LazyLock<Option<Map<String, Value>>> = LazyLock::new(|| {
    let (_, contents) = crate::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")?;
    let manifest: Value = serde_json::from_str(&strip_jsonc_comments(contents)).ok()?;
    manifest
        .get("models")?
        .as_array()?
        .iter()
        .find(|model| {
            model.get("id").and_then(Value::as_str) == Some(MINIMAX_H3_BASE_PARTITION_MODEL)
        })?
        .as_object()
        .cloned()
});

fn builtin_base_partition_entry() -> Option<&'static Map<String, Value>> {
    BUILTIN_BASE_PARTITION_ENTRY.as_ref()
}

/// Give an imported H3 entry the shipped base partition's generation surface.
///
/// The ROUTE-OWNED fields — `adapter`, `capabilities`, `video`, `defaults`, `limits` — are replaced,
/// not filled: they state what the provider serves for this partition (its modes, its 17n+5
/// duration lattice, its single fps, its step floor), and a value left over from an import request
/// that guessed the wrong type (an image entry's capabilities) would advertise operations nothing
/// can run. Everything descriptive (`ui`, licence text, memory hints) is `or_insert`, so an author's
/// wording wins. Deliberately NOT copied: the builtin's downloads and paths (the imported DiT is its
/// own weights), `mlx.repo` (the builtin tier repo), and `loraCompatibility` — adapter and turbo
/// support on imported H3 checkpoints is a later story, and advertising adapters the imported route
/// refuses would admit jobs nothing can run.
pub fn apply_minimax_h3_studio_surface_defaults(entry: &mut Map<String, Value>) {
    let Some(base) = builtin_base_partition_entry() else {
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
    // Withdraw the family-wide adapter advertisement `apply_model_manifest_defaults` synthesizes.
    entry.remove("loraCompatibility");
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
        ImportPlanV1::new("checkpoint-plan-test", family, layers).expect("valid plan")
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
    fn a_dense_dit_directory_binds_the_plan_dit_and_every_resident_component() {
        let assembly = MiniMaxH3ImportAssembly::from_plan(&dense_dit_plan()).expect("complete");
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
                json!({ "source": "resident", "model": model }),
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
        assert!(MiniMaxH3ImportAssembly::from_plan(&no_dit)
            .unwrap_err()
            .contains("no \"transformer\" layer"));

        // A second weight file beside the DiT would be mapped unbound by `Weights::from_dir`.
        let mut sharded = dense_dit_plan();
        sharded.layers.push(layer(
            "artifact:extra.safetensors",
            "transformer",
            "extra.safetensors",
            CheckpointContainerV1::Safetensors,
        ));
        assert!(MiniMaxH3ImportAssembly::from_plan(&sharded)
            .unwrap_err()
            .contains("2 \"transformer\" layers"));

        // An encoder carried by the import would be silently replaced by the resident one.
        let mut with_te = dense_dit_plan();
        with_te.layers.push(layer(
            "artifact:text_encoder/model.safetensors",
            "text_encoder",
            "text_encoder/model.safetensors",
            CheckpointContainerV1::Safetensors,
        ));
        assert!(MiniMaxH3ImportAssembly::from_plan(&with_te)
            .unwrap_err()
            .contains("\"text_encoder\" layer"));

        // Another family's plan is never an H3 assembly.
        let wan = plan("wan-video", dense_dit_plan().layers);
        assert!(MiniMaxH3ImportAssembly::from_plan(&wan)
            .unwrap_err()
            .contains("\"wan-video\" family"));
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
        let assembly = MiniMaxH3ImportAssembly::from_plan(&dense_dit_plan()).unwrap();
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
