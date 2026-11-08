//! Portable admission routes follow the same request predicates as worker claim.

use sceneworks_core::contracts::{JobType, JsonObject};
use sceneworks_core::jobs_store::{
    native_route_claims_request, CANDLE_NATIVE_CONVERTERS, NATIVE_CONVERTERS, NON_GPU_JOB_TYPES,
};
use sceneworks_core::model_artifacts::ArtifactMemberRole;
use serde_json::Value;

use crate::ApiError;

#[derive(Clone, Debug)]
pub(crate) enum SelectedSourceLocation {
    ManifestDownload(Value),
    LocalPath(String),
    /// Admission resolves the request-selected repository HEAD to an immutable revision.
    DynamicHfRepo(String),
    /// A server-shipped mutable ref is resolved to a commit before the route is frozen.
    DynamicHfRef {
        repo: &'static str,
        reference: &'static str,
        files: &'static [&'static str],
    },
    /// Request-selected repository files are frozen to a commit at admission.
    DynamicHfFiles {
        repo: String,
        reference: String,
        files: Vec<String>,
    },
    /// Admission freezes one worker-advertised FLUX.2 tier's borrowed dense
    /// subtrees. Custom repositories may have different bytes in other tiers.
    ConversionBorrow {
        repo: String,
        preferred_subdir: Option<String>,
    },
    /// An operator Gemma override wins when a registered worker advertises it; otherwise
    /// admission binds the exact selected LTX bundle's pinned sibling Gemma download.
    OperatorOverrideOrManifest {
        source_id: &'static str,
        default: Value,
    },
    PreferredWorkerClaim {
        preferred_id: &'static str,
        fallback_id: &'static str,
    },
    /// A worker-advertised exact digest closure, never an unpinned API download.
    OperatorCachedRepo(&'static str),
}

#[derive(Clone, Debug)]
pub(crate) struct SelectedRouteSource {
    pub model_slot: String,
    pub component_id: String,
    pub role: ArtifactMemberRole,
    pub source: SelectedSourceLocation,
    pub selected_tier: Option<String>,
}

fn push_operator_source(rows: &mut Vec<SelectedRouteSource>, slot: &str, source_id: &'static str) {
    rows.push(SelectedRouteSource {
        model_slot: slot.to_owned(),
        component_id: source_id.to_owned(),
        role: ArtifactMemberRole::OptionalComponent,
        source: SelectedSourceLocation::OperatorCachedRepo(source_id),
        selected_tier: None,
    });
}

fn push_builtin_training_source(
    rows: &mut Vec<SelectedRouteSource>,
    slot: &str,
    catalog: &[Value],
    catalog_id: &str,
    component_id: &str,
) -> Result<(), String> {
    let entry = catalog
        .iter()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(catalog_id))
        .ok_or_else(|| format!("training source {catalog_id} lacks catalog entry"))?;
    let download = entry
        .get("downloads")
        .and_then(Value::as_array)
        .and_then(|rows| rows.first())
        .ok_or_else(|| format!("training source {catalog_id} lacks pinned download"))?;
    rows.push(SelectedRouteSource {
        model_slot: slot.to_owned(),
        component_id: component_id.to_owned(),
        role: ArtifactMemberRole::RequiredComponent,
        source: SelectedSourceLocation::ManifestDownload(download.clone()),
        selected_tier: None,
    });
    Ok(())
}

/// Source additions opened by a route but omitted from its always-required model closure.
/// A missing selected declaration is an admission refusal, never an implicit worker download.
pub(crate) fn selected_route_source_rows(
    job_type: &JobType,
    payload: &JsonObject,
    entries: &[(String, &Value)],
    backend: &str,
    platform: &str,
) -> Result<Vec<SelectedRouteSource>, String> {
    let mut rows = Vec::new();
    if *job_type == JobType::ModelConvert {
        let slot = entries
            .first()
            .map(|(slot, _)| slot.clone())
            .ok_or("conversion lacks a model source carrier")?;
        match payload.get("converter").and_then(Value::as_str) {
            Some("flux2_klein_diffusers") => {
                let repo = payload
                    .get("baseRepo")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or("FLUX.2 conversion lacks baseRepo")?;
                rows.push(SelectedRouteSource {
                    model_slot: slot,
                    component_id: "flux2_convert_base".to_owned(),
                    role: ArtifactMemberRole::RequiredComponent,
                    source: SelectedSourceLocation::ConversionBorrow {
                        repo: repo.to_owned(),
                        preferred_subdir: payload
                            .get("baseSubdir")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                    },
                    selected_tier: None,
                });
            }
            Some("ltx_video") => {
                let repo = payload
                    .get("baseRepo")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or("LTX conversion lacks baseRepo")?;
                let file = payload
                    .get("upscalerFile")
                    .and_then(Value::as_str)
                    .unwrap_or("ltx-2.3-spatial-upscaler-x2-1.1.safetensors");
                rows.push(SelectedRouteSource {
                    model_slot: slot,
                    component_id: "ltx_convert_upscaler".to_owned(),
                    role: ArtifactMemberRole::RequiredComponent,
                    source: SelectedSourceLocation::DynamicHfFiles {
                        repo: repo.to_owned(),
                        reference: "main".to_owned(),
                        files: vec![file.to_owned()],
                    },
                    selected_tier: None,
                });
            }
            _ => {}
        }
    }
    let advanced = payload.get("advanced").and_then(Value::as_object);
    let has_reference = payload
        .get("referenceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|id| !id.trim().is_empty());
    let character_reference =
        payload.get("mode").and_then(Value::as_str) == Some("character_image") && has_reference;
    if matches!(job_type, JobType::LoraTrain | JobType::ControlTraining) {
        let plan: sceneworks_core::training::TrainingPlan = serde_json::from_value(
            payload
                .get("plan")
                .cloned()
                .ok_or("training job lacks resolved plan")?,
        )
        .map_err(|error| format!("invalid resolved training plan: {error}"))?;
        let training_slot = entries
            .iter()
            .find(|(_, entry)| {
                entry.get("id").and_then(Value::as_str) == Some(plan.target.base_model.as_str())
            })
            .map(|(slot, _)| slot.as_str())
            .unwrap_or("");
        let trainer_id = sceneworks_worker::portable_trainer_engine_id_for(
            &plan.target.kernel,
            &plan.target.base_model,
        )
        .ok_or_else(|| {
            format!(
                "no native trainer for {} / {}",
                plan.target.kernel, plan.target.base_model
            )
        })?;
        let aux_models =
            sceneworks_core::portable_sources::selected_training_aux_models(&plan, trainer_id)?;
        if !aux_models.is_empty() {
            if training_slot.is_empty() {
                return Err("training auxiliaries lack a captured base entry".to_owned());
            }
            let manifest =
                sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
                    .map_err(str::to_owned)?;
            let catalog = manifest
                .get("models")
                .and_then(Value::as_array)
                .ok_or("builtin model catalog lacks models")?;
            let mut selected_ids = std::collections::BTreeSet::new();
            for model in aux_models {
                if !selected_ids.insert(model.id) {
                    continue;
                }
                let entry = catalog
                    .iter()
                    .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model.id))
                    .ok_or_else(|| {
                        format!("training auxiliary {} lacks a catalog entry", model.id)
                    })?;
                let download = entry
                    .get("downloads")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .find(|download| {
                        download.get("repo").and_then(Value::as_str) == Some(model.repo)
                            && download.get("revision").and_then(Value::as_str)
                                == Some(model.revision)
                    })
                    .ok_or_else(|| {
                        format!("training auxiliary {} lacks pinned download", model.id)
                    })?;
                rows.push(SelectedRouteSource {
                    model_slot: training_slot.to_owned(),
                    component_id: model.id.to_owned(),
                    role: ArtifactMemberRole::RequiredComponent,
                    source: SelectedSourceLocation::ManifestDownload(download.clone()),
                    selected_tier: None,
                });
            }
        }
        if matches!(
            plan.target.base_model.as_str(),
            "iris_3b_depth" | "iris_3b_restore"
        ) {
            if training_slot.is_empty() {
                return Err("Iris task training lacks a captured task entry".to_owned());
            }
            let manifest =
                sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
                    .map_err(str::to_owned)?;
            let catalog = manifest
                .get("models")
                .and_then(Value::as_array)
                .ok_or("builtin model catalog lacks models")?;
            if plan.target.base_model == "iris_3b_depth" {
                if sceneworks_core::training::iris::depth::depth_init(&plan.config)
                    .map_err(|error| error.to_string())?
                    == sceneworks_core::training::iris::depth::DEPTH_INIT_PARENT
                {
                    push_builtin_training_source(
                        &mut rows,
                        training_slot,
                        catalog,
                        "iris_3b",
                        "iris_parent_backbone",
                    )?;
                    rows.push(SelectedRouteSource {
                        model_slot: training_slot.to_owned(),
                        component_id: "iris_depth_parent".to_owned(),
                        role: ArtifactMemberRole::RequiredComponent,
                        source: SelectedSourceLocation::PreferredWorkerClaim {
                            preferred_id: "iris_depth_export",
                            fallback_id: "iris_generation_text_encoder",
                        },
                        selected_tier: None,
                    });
                }
            } else {
                for (catalog_id, _) in
                    sceneworks_core::training::iris::restore::RESTORE_TRAINING_RESOURCE_MODELS
                {
                    push_builtin_training_source(
                        &mut rows,
                        training_slot,
                        catalog,
                        catalog_id,
                        catalog_id,
                    )?;
                }
                if sceneworks_core::training::iris::restore::restore_options(&plan.config)
                    .map_err(|error| error.to_string())?
                    .get("init")
                    .and_then(Value::as_str)
                    == Some("parent")
                {
                    push_builtin_training_source(
                        &mut rows,
                        training_slot,
                        catalog,
                        "iris_3b",
                        "iris_parent_backbone",
                    )?;
                }
            }
        }
        let trainer_components = sceneworks_core::portable_sources::trainer_required_component_ids(
            &plan.target.base_model,
            backend,
        );
        if !trainer_components.is_empty() {
            let base_entry = entries
                .iter()
                .find(|(slot, _)| slot == training_slot)
                .map(|(_, entry)| *entry)
                .ok_or("trainer components lack a captured base entry")?;
            let downloads = base_entry
                .get("downloads")
                .and_then(Value::as_array)
                .ok_or("trainer base lacks component downloads")?;
            for component_id in trainer_components {
                let candidates: Vec<&Value> = downloads
                    .iter()
                    .filter(|download| {
                        download.get("coRequisite").and_then(Value::as_bool) == Some(true)
                            && download.get("componentId").and_then(Value::as_str)
                                == Some(component_id)
                            && download.get("required").and_then(Value::as_str) != Some("soft")
                    })
                    .collect();
                let selected = if candidates.len() == 1 {
                    candidates[0]
                } else {
                    candidates
                        .iter()
                        .copied()
                        .find(|download| {
                            download
                                .get("variant")
                                .and_then(Value::as_str)
                                .is_some_and(|variant| variant.eq_ignore_ascii_case("bf16"))
                        })
                        .ok_or_else(|| {
                            format!(
                                "trainer component {component_id} has no unambiguous bf16 source"
                            )
                        })?
                };
                rows.push(SelectedRouteSource {
                    model_slot: training_slot.to_owned(),
                    component_id: (*component_id).to_owned(),
                    role: ArtifactMemberRole::RequiredComponent,
                    source: SelectedSourceLocation::ManifestDownload(selected.clone()),
                    selected_tier: Some("bf16".to_owned()),
                });
            }
        }
        if plan.target.base_model == "ltx_2_3" {
            if training_slot.is_empty() {
                return Err("LTX training base lacks a captured manifest entry".to_owned());
            }
            let base_entry = entries
                .iter()
                .find(|(slot, _)| slot == training_slot)
                .map(|(_, entry)| *entry)
                .expect("slot was selected from entries");
            let gemma = base_entry
                .get("downloads")
                .and_then(Value::as_array)
                .and_then(|downloads| {
                    downloads.iter().find(|download| {
                        download.get("repo").and_then(Value::as_str)
                            == Some("SceneWorks/ltx-2.3-mlx")
                            && download.get("files").and_then(Value::as_array).is_some_and(
                                |files| files.iter().any(|file| file.as_str() == Some("gemma/*")),
                            )
                    })
                })
                .ok_or("LTX training base lacks pinned sibling Gemma declaration")?;
            rows.push(SelectedRouteSource {
                model_slot: training_slot.to_owned(),
                component_id: "ltx_training_text_encoder".to_owned(),
                role: ArtifactMemberRole::RequiredComponent,
                source: SelectedSourceLocation::OperatorOverrideOrManifest {
                    source_id: "ltx_training_gemma_override",
                    default: gemma.clone(),
                },
                selected_tier: None,
            });
        }
        if sceneworks_core::training::iris::is_iris_training_kernel(&plan.target.kernel) {
            let options = sceneworks_core::training::iris::iris_options(&plan.config)
                .map_err(|error| error.to_string())?;
            if sceneworks_core::training::iris::curricula::iris_repa_enabled(&options) {
                if training_slot.is_empty() {
                    return Err(
                        "Iris REPA training base lacks a captured manifest entry".to_owned()
                    );
                }
                let manifest =
                    sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
                        .map_err(str::to_owned)?;
                let teacher = manifest.get("models").and_then(Value::as_array)
                    .and_then(|models| models.iter().find(|entry| entry.get("id").and_then(Value::as_str) == Some(sceneworks_core::training::iris::curricula::IRIS_REPA_TEACHER_MODEL_ID)))
                    .and_then(|entry| entry.get("downloads")).and_then(Value::as_array)
                    .and_then(|downloads| downloads.first())
                    .ok_or("missing pinned Iris REPA teacher declaration")?;
                rows.push(SelectedRouteSource {
                    model_slot: training_slot.to_owned(),
                    component_id: "repa_teacher".to_owned(),
                    role: ArtifactMemberRole::OptionalComponent,
                    source: SelectedSourceLocation::ManifestDownload(teacher.clone()),
                    selected_tier: None,
                });
            }
        }
    }
    for (slot, entry) in entries {
        let id = entry
            .get("id")
            .and_then(Value::as_str)
            .ok_or("model entry lacks id")?;
        if entry
            .get("downloads")
            .and_then(Value::as_array)
            .is_none_or(Vec::is_empty)
        {
            if let Some(base) =
                sceneworks_core::model_artifacts::artifact_selection::selected_resident_base_closure(
                    entry, platform,
                )
            {
                for (index, download) in sceneworks_core::model_artifacts::artifact_selection::resident_base_component_downloads(&base).into_iter().enumerate() {
                    rows.push(SelectedRouteSource {
                        model_slot: slot.clone(),
                        component_id: format!("resident_base/downloads/{index}"),
                        role: ArtifactMemberRole::RequiredComponent,
                        source: SelectedSourceLocation::ManifestDownload(download),
                        selected_tier: Some("bf16".to_owned()),
                    });
                }
            }
        }
        if id == "instantid_face_stack"
            && matches!(
                job_type,
                JobType::KpsExtract
                    | JobType::DatasetFaceAnalysis
                    | JobType::FaceLikenessCompare
                    | JobType::CatalogAnalysis
            )
        {
            let download = entry
                .get("downloads")
                .and_then(Value::as_array)
                .and_then(|rows| rows.first())
                .ok_or("face stack lacks pinned download")?;
            rows.push(SelectedRouteSource {
                model_slot: slot.clone(),
                component_id: "utility_face_stack".to_owned(),
                role: ArtifactMemberRole::OptionalComponent,
                source: SelectedSourceLocation::ManifestDownload(download.clone()),
                selected_tier: None,
            });
        }
        if id == "instantid_realvisxl" {
            push_operator_source(&mut rows, slot, "instantid_mlx");
            push_operator_source(&mut rows, slot, "instantid_controlnet");
            if advanced
                .and_then(|value| value.get("poses"))
                .and_then(Value::as_array)
                .is_some_and(|poses| poses.iter().any(Value::is_object))
            {
                push_operator_source(&mut rows, slot, "instantid_openpose");
            }
        }
        if id == "pulid_flux_dev" {
            for source_id in [
                "pulid_adapter",
                "pulid_eva",
                "pulid_bisenet",
                "pulid_scrfd",
                "pulid_arcface",
            ] {
                push_operator_source(&mut rows, slot, source_id);
            }
        }
        if id == "kolors"
            && has_reference
            && (backend == "mlx"
                || !advanced
                    .and_then(|value| value.get("poses"))
                    .and_then(Value::as_array)
                    .is_some_and(|poses| poses.iter().any(Value::is_object)))
        {
            rows.push(SelectedRouteSource {
                model_slot: slot.clone(),
                component_id: "kolors_ip_adapter".to_owned(),
                role: ArtifactMemberRole::OptionalComponent,
                source: SelectedSourceLocation::DynamicHfRef {
                    repo: "Kwai-Kolors/Kolors-IP-Adapter-Plus",
                    reference: "refs/pr/4",
                    files: &[
                        "ip_adapter_plus_general.safetensors",
                        "image_encoder/config.json",
                        "image_encoder/model.safetensors",
                    ],
                },
                selected_tier: None,
            });
        }
        if sceneworks_core::portable_sources::pid_backbone_for(id).is_some()
            && (id != "kolors"
                || (backend == "candle"
                    && advanced
                        .and_then(|value| value.get("poses"))
                        .and_then(Value::as_array)
                        .is_some_and(|poses| poses.iter().any(Value::is_object))))
            && (id != "pulid_flux_dev" || backend == "candle")
            && sceneworks_core::portable_sources::pid_requested(
                advanced.and_then(|value| value.get("usePid")),
            )
            && payload.get("model").and_then(Value::as_str) == Some(id)
        {
            let backbone =
                sceneworks_core::portable_sources::pid_backbone_for(id).expect("checked");
            let manifest =
                sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
                    .map_err(str::to_owned)?;
            let utility = manifest
                .get("models")
                .and_then(Value::as_array)
                .and_then(|models| {
                    models.iter().find(|model| {
                        model.get("id").and_then(Value::as_str) == Some(&format!("pid_{backbone}"))
                    })
                })
                .ok_or_else(|| format!("missing PiD {backbone} utility declaration"))?;
            let downloads = utility
                .get("downloads")
                .and_then(Value::as_array)
                .ok_or("PiD utility has no downloads")?;
            let checkpoint = downloads.first().ok_or("PiD utility has no checkpoint")?;
            let gemma = downloads.get(1).ok_or("PiD utility has no Gemma")?;
            let checkpoint_override = advanced
                .and_then(|value| value.get("pidCheckpoint"))
                .and_then(Value::as_object);
            let checkpoint_source = if let Some(config) = checkpoint_override {
                let authored_repo = config
                    .get("repo")
                    .and_then(Value::as_str)
                    .filter(|value| !value.trim().is_empty());
                let authored_file = config.get("filename").and_then(Value::as_str);
                if let Some(file) = authored_file {
                    if file.is_empty()
                        || file == "."
                        || file == ".."
                        || file.contains(['/', '\\'])
                        || !file.ends_with(".safetensors")
                    {
                        return Err(
                            "PiD checkpoint filename must be a plain safetensors name".to_owned()
                        );
                    }
                }
                if let Some(repo) = authored_repo {
                    SelectedSourceLocation::DynamicHfFiles {
                        repo: repo.to_owned(),
                        reference: "main".to_owned(),
                        files: vec![authored_file.unwrap_or("*").to_owned()],
                    }
                } else if let Some(file) = authored_file {
                    let mut pinned = checkpoint.clone();
                    pinned["files"] = serde_json::json!([file]);
                    SelectedSourceLocation::ManifestDownload(pinned)
                } else {
                    SelectedSourceLocation::ManifestDownload(checkpoint.clone())
                }
            } else {
                SelectedSourceLocation::ManifestDownload(checkpoint.clone())
            };
            let gemma_source = if let Some(repo) = advanced
                .and_then(|value| value.get("pidGemma"))
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
            {
                SelectedSourceLocation::DynamicHfFiles {
                    repo: repo.to_owned(),
                    reference: "main".to_owned(),
                    files: vec!["*".to_owned()],
                }
            } else {
                SelectedSourceLocation::ManifestDownload(gemma.clone())
            };
            for (component_id, source) in [
                ("pid_checkpoint", checkpoint_source),
                ("pid_gemma", gemma_source),
            ] {
                rows.push(SelectedRouteSource {
                    model_slot: slot.clone(),
                    component_id: component_id.to_owned(),
                    role: ArtifactMemberRole::OptionalComponent,
                    source,
                    selected_tier: None,
                });
            }
        }
        let add_component = |rows: &mut Vec<SelectedRouteSource>,
                             component_id: &str,
                             role: ArtifactMemberRole|
         -> Result<(), String> {
            let download = ["downloads", "conditionalComponents"]
                .into_iter()
                .filter_map(|field| entry.get(field).and_then(Value::as_array))
                .flatten()
                .find(|row| row.get("componentId").and_then(Value::as_str) == Some(component_id))
                .ok_or_else(|| format!("{id} lacks selected component {component_id}"))?;
            rows.push(SelectedRouteSource {
                model_slot: slot.clone(),
                component_id: component_id.to_owned(),
                role,
                source: SelectedSourceLocation::ManifestDownload(download.clone()),
                selected_tier: None,
            });
            Ok(())
        };
        if matches!(job_type, JobType::ImageGenerate | JobType::ImageEdit) {
            if slot == "modelManifestEntry" && id.starts_with("moss_image") {
                for download in entry
                    .get("downloads")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let Some(component_id) = download
                        .get("componentId")
                        .and_then(Value::as_str)
                        .filter(|id| {
                            sceneworks_core::moss_image::is_runtime_optional_component_id(id)
                        })
                    else {
                        continue;
                    };
                    rows.push(SelectedRouteSource {
                        model_slot: slot.clone(),
                        component_id: component_id.to_owned(),
                        role: ArtifactMemberRole::OptionalComponent,
                        source: SelectedSourceLocation::ManifestDownload(download.clone()),
                        selected_tier: None,
                    });
                }
            }
            if character_reference && matches!(id, "flux_dev" | "flux_schnell") {
                push_operator_source(&mut rows, slot, "flux_ip_adapter");
                push_operator_source(&mut rows, slot, "flux_ip_encoder");
            }
            if character_reference
                && matches!(
                    id,
                    "sdxl" | "realvisxl" | "illustrious_xl_v1" | "illustrious_xl_v2"
                )
            {
                push_operator_source(&mut rows, slot, "sdxl_ip_adapter");
                push_operator_source(&mut rows, slot, "sdxl_ip_encoder");
            }
            if let Some(decoder) = advanced
                .and_then(|value| value.get("decoder"))
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty() && *value != "native")
            {
                let option = entry
                    .pointer(&format!("/decoders/byBackend/{backend}"))
                    .and_then(Value::as_array)
                    .and_then(|options| {
                        options.iter().find(|option| {
                            option.get("id").and_then(Value::as_str) == Some(decoder)
                        })
                    })
                    .ok_or_else(|| format!("{id} cannot use decoder {decoder} on {backend}"))?;
                let component = option
                    .get("componentId")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("decoder {decoder} has no component"))?;
                add_component(&mut rows, component, ArtifactMemberRole::OptionalComponent)?;
            }
            if matches!(id, "flux3_klein" | "flux3_klein_base")
                && advanced
                    .and_then(|value| value.get("enhancePrompt"))
                    .and_then(Value::as_bool)
                    == Some(true)
            {
                add_component(
                    &mut rows,
                    "prompt_upsampler",
                    ArtifactMemberRole::OptionalComponent,
                )?;
            }
            if id != "instantid_realvisxl"
                && slot == "modelManifestEntry"
                && advanced
                    .and_then(|value| value.get("poses"))
                    .and_then(Value::as_array)
                    .is_some_and(|poses| !poses.is_empty())
            {
                let control = advanced.and_then(|value| value.get("controlWeights"));
                let (source, tier) = if let Some(control) = control {
                    if let Some(path) = control.get("path").and_then(Value::as_str) {
                        (SelectedSourceLocation::LocalPath(path.to_owned()), None)
                    } else {
                        let repo = control
                            .get("repo")
                            .and_then(Value::as_str)
                            .ok_or("control override lacks repo")?;
                        let file = control
                            .get("filename")
                            .and_then(Value::as_str)
                            .ok_or("control override lacks filename")?;
                        let revision = control
                            .get("revision")
                            .and_then(Value::as_str)
                            .ok_or("control override lacks revision")?;
                        (
                            SelectedSourceLocation::ManifestDownload(
                                serde_json::json!({"provider":"huggingface","repo":repo,"revision":revision,"files":[file]}),
                            ),
                            None,
                        )
                    }
                } else if let Some(weight) =
                    sceneworks_core::control_weights::default_control_weight_for_request(
                        id, entry, backend, payload,
                    )
                {
                    let tier = (id == "qwen_image").then(|| {
                        sceneworks_core::control_weights::selected_qwen_control_tier(payload)
                            .to_owned()
                    });
                    (
                        SelectedSourceLocation::ManifestDownload(
                            serde_json::json!({"provider":"huggingface","repo":weight.repo,"revision":weight.revision,"files":[weight.file]}),
                        ),
                        tier,
                    )
                } else {
                    return Err(format!("{id} has poses but no bound strict-control source"));
                };
                rows.push(SelectedRouteSource {
                    model_slot: slot.clone(),
                    component_id: "strict_control".to_owned(),
                    role: ArtifactMemberRole::Control,
                    source,
                    selected_tier: tier,
                });
                if advanced
                    .and_then(|value| value.get("controlMode"))
                    .and_then(Value::as_str)
                    == Some("depth")
                {
                    push_operator_source(&mut rows, slot, "depth_anything_v2_small");
                }
            }
        }
        if *job_type == JobType::ImageDetail && slot == "modelManifestEntry" {
            let custom_repo = advanced
                .and_then(|value| value.get("tileControlNetRepo"))
                .and_then(Value::as_str)
                .filter(|repo| {
                    !repo.trim().is_empty() && *repo != "xinsir/controlnet-tile-sdxl-1.0"
                });
            if let Some(repo) = custom_repo {
                if backend != "mlx" {
                    return Err("custom Detail tile ControlNet is supported only on MLX".to_owned());
                }
                rows.push(SelectedRouteSource {
                    model_slot: slot.clone(),
                    component_id: "sdxl_detail_tile_control".to_owned(),
                    role: ArtifactMemberRole::Control,
                    source: SelectedSourceLocation::DynamicHfRepo(repo.to_owned()),
                    selected_tier: None,
                });
            } else {
                push_operator_source(&mut rows, slot, "sdxl_detail_tile_control");
            }
        }
        if payload.get("editMode").and_then(Value::as_str) == Some("cover")
            && id.starts_with("acestep")
        {
            add_component(
                &mut rows,
                "sft_cover",
                ArtifactMemberRole::OptionalComponent,
            )?;
        }
        if matches!(id, "minimax_h3" | "minimax_h3_ref")
            && advanced
                .and_then(|value| value.get("veda"))
                .and_then(|value| value.get("enabled"))
                .and_then(Value::as_bool)
                == Some(true)
        {
            add_component(
                &mut rows,
                "veda_predictor",
                ArtifactMemberRole::OptionalComponent,
            )?;
        }
        if id.starts_with("ltx")
            && backend == "mlx"
            && advanced.is_some_and(|value| {
                value.get("useUncensoredEnhancer").and_then(Value::as_bool) == Some(true)
                    || value.get("textEncoderModel").and_then(Value::as_str)
                        == Some("ltx_amoral_gemma_3_12b")
            })
        {
            rows.push(SelectedRouteSource {
                model_slot: slot.clone(),
                component_id: "ltx_amoral_gemma_3_12b".to_owned(),
                role: ArtifactMemberRole::OptionalComponent,
                source: SelectedSourceLocation::OperatorCachedRepo("ltx_amoral_gemma_3_12b"),
                selected_tier: None,
            });
        }
    }
    Ok(rows)
}

/// Backend and platform pairs whose full source closure must be frozen at admission. The
/// worker claim still applies the requested GPU id and capability filters; this set never
/// invents a route merely because the catalog advertises some other request shape.
pub(crate) fn eligible_portable_routes(
    job_type: &JobType,
    payload: &JsonObject,
    entries: &[(String, &Value)],
    requested_gpu: &str,
) -> Result<Vec<(String, String)>, ApiError> {
    if entries.is_empty() {
        return Ok(Vec::new());
    }
    if *job_type == JobType::ModelConvert {
        if !matches!(requested_gpu, "auto" | "cpu") {
            return Err(ApiError::bad_request(
                "Model conversion is claimed by a CPU utility worker; requestedGpu must be auto or cpu",
            ));
        }
        let converter = payload
            .get("converter")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !NATIVE_CONVERTERS.contains(&converter) {
            return Err(ApiError::bad_request(
                "Model conversion has no native converter for the selected source",
            ));
        }
        if !native_route_claims_request("cpu", job_type, payload).map_err(ApiError::internal)? {
            return Err(ApiError::bad_request(
                "No CPU utility worker route can claim this conversion request",
            ));
        }
        // The utility worker reports backend `cpu` at claim even when its in-process converter
        // uses MLX or Candle. Only the two Candle-native converters run off macOS.
        let platforms: &[&str] = if CANDLE_NATIVE_CONVERTERS.contains(&converter) {
            &["macos", "linux", "windows"]
        } else {
            &["macos"]
        };
        let routes = platforms
            .iter()
            .filter(|platform| {
                entries.iter().all(|(_, entry)| {
                    entry
                        .get("downloads")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                        .any(|download| {
                            download
                                .get("platforms")
                                .and_then(Value::as_array)
                                .is_none_or(|allowed| {
                                    allowed
                                        .iter()
                                        .any(|value| value.as_str() == Some(*platform))
                                })
                        })
                })
            })
            .map(|platform| ("cpu".to_owned(), (*platform).to_owned()))
            .collect::<Vec<_>>();
        if routes.is_empty() {
            return Err(ApiError::bad_request(
                "No native conversion route can claim the selected source",
            ));
        }
        return Ok(routes);
    }
    if NON_GPU_JOB_TYPES.contains(&job_type.as_str()) {
        return Ok(["macos", "linux", "windows"]
            .into_iter()
            .map(|platform| ("cpu".to_owned(), platform.to_owned()))
            .collect());
    }
    if requested_gpu.eq_ignore_ascii_case("cpu") {
        return Err(ApiError::bad_request(
            "GPU model work cannot be assigned to the CPU utility worker",
        ));
    }
    let mut routes = Vec::new();
    // The MLX worker registers the literal gpu id `mlx`. Every explicit non-CPU id other
    // than that names a CUDA/Candle worker (including `0`, `cuda:0`, and operator ids).
    // An unknown id may still wait in the queue, but it cannot authorize the other backend.
    let explicit_backend = match requested_gpu {
        "auto" => None,
        "mlx" => Some("mlx"),
        _ => Some("candle"),
    };
    for (backend, platforms) in [
        ("mlx", &["macos"][..]),
        ("candle", &["linux", "windows"][..]),
    ] {
        if explicit_backend.is_some_and(|selected| selected != backend) {
            continue;
        }
        if backend == "candle"
            && *job_type == JobType::ImageDetail
            && payload
                .get("advanced")
                .and_then(Value::as_object)
                .and_then(|advanced| advanced.get("tileControlNetRepo"))
                .and_then(Value::as_str)
                .is_some_and(|repo| {
                    !repo.trim().is_empty() && repo != "xinsir/controlnet-tile-sdxl-1.0"
                })
        {
            continue;
        }
        let claims =
            native_route_claims_request(backend, job_type, payload).map_err(ApiError::internal)?;
        if !claims {
            continue;
        }
        routes.extend(
            platforms
                .iter()
                .filter(|platform| entries_support_route(job_type, entries, backend, platform))
                .map(|platform| (backend.to_owned(), (*platform).to_owned())),
        );
    }
    if routes.is_empty() {
        return Err(ApiError::bad_request(
            "No native route can claim this model request and complete source closure",
        ));
    }
    Ok(routes)
}

fn entries_support_route(
    job_type: &JobType,
    entries: &[(String, &Value)],
    backend: &str,
    platform: &str,
) -> bool {
    entries.iter().all(|(slot, entry)| {
        let Some(id) = entry.get("id").and_then(Value::as_str) else {
            return false;
        };
        let downloads: Vec<&Value> = ["downloads", "conditionalComponents"]
            .into_iter()
            .filter_map(|field| entry.get(field).and_then(Value::as_array))
            .flatten()
            .collect();
        let downloads_support_platform = downloads.is_empty()
            || downloads.iter().any(|download| {
                download
                    .get("platforms")
                    .and_then(Value::as_array)
                    .is_none_or(|platforms| {
                        platforms
                            .iter()
                            .any(|value| value.as_str() == Some(platform))
                    })
            });
        if !downloads_support_platform {
            return false;
        }
        if entry.get("type").and_then(Value::as_str) == Some("utility")
            && entry.get("componentOnly").and_then(Value::as_bool) != Some(true)
            && *job_type != JobType::CatalogAnalysis
            && slot != "modelManifestEntry"
            && slot != "baseModelManifestEntry"
        {
            return false;
        }
        if entry.get("type").and_then(Value::as_str) == Some("utility")
            && (entry.get("componentOnly").and_then(Value::as_bool) == Some(true)
                || (*job_type == JobType::CatalogAnalysis
                    && slot != "modelManifestEntry"
                    && slot != "baseModelManifestEntry"))
        {
            return !downloads.is_empty();
        }
        if let Some(spec) = sceneworks_core::registry::builtin_model_spec(id) {
            return spec.routes.contains_key(backend);
        }
        let key = if backend == "mlx" {
            "macSupport"
        } else {
            "candleSupport"
        };
        entry
            .pointer(&format!("/{key}/supported"))
            .and_then(Value::as_bool)
            != Some(false)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imported_checkpoint_freezes_selected_resident_base_components() {
        for (family, expected_repo) in [
            ("krea_2", "SceneWorks/krea-2-turbo-mlx"),
            ("mage-flow", "SceneWorks/Mage-Flow-Components-mlx"),
        ] {
            let entry = serde_json::json!({
                "id": "imported-checkpoint", "family": family,
                "installedPath": "/registered/imported-checkpoint.safetensors"
            });
            let entries = [("modelManifestEntry".to_owned(), &entry)];
            let payload = serde_json::json!({"model":"imported-checkpoint"});
            let selected = selected_route_source_rows(
                &JobType::ImageGenerate,
                payload.as_object().unwrap(),
                &entries,
                "mlx",
                "macos",
            )
            .unwrap();
            let resident: Vec<_> = selected
                .iter()
                .filter(|row| row.component_id.starts_with("resident_base/downloads/"))
                .collect();
            assert!(
                !resident.is_empty(),
                "{family} must freeze its resident base"
            );
            assert!(resident.iter().all(|row| {
                row.role == ArtifactMemberRole::RequiredComponent
                    && row.model_slot == "modelManifestEntry"
                    && row.selected_tier.as_deref() == Some("bf16")
                    && matches!(&row.source, SelectedSourceLocation::ManifestDownload(download)
                        if download["repo"] == expected_repo
                            && download["revision"].as_str().is_some_and(|revision| revision.len() == 40)
                            && download["coRequisite"] == true)
            }));
        }
    }

    #[test]
    fn conversion_freezes_borrowed_base_or_exact_upscaler_source() {
        let entry = serde_json::json!({"id":"model_convert:community/source","downloads":[{"provider":"huggingface","repo":"community/source"}]});
        let entries = [("modelManifestEntries/0".to_owned(), &entry)];
        let klein = serde_json::json!({
            "converter":"flux2_klein_diffusers", "sourceRepo":"community/source",
            "sourceFile":"checkpoint.safetensors", "baseRepo":"other/klein",
            "baseSubdir":"q4"
        });
        assert_eq!(
            eligible_portable_routes(
                &JobType::ModelConvert,
                klein.as_object().unwrap(),
                &entries,
                "auto",
            )
            .unwrap(),
            [
                ("cpu".to_owned(), "macos".to_owned()),
                ("cpu".to_owned(), "linux".to_owned()),
                ("cpu".to_owned(), "windows".to_owned()),
            ],
        );
        assert!(eligible_portable_routes(
            &JobType::ModelConvert,
            klein.as_object().unwrap(),
            &entries,
            "mlx",
        )
        .is_err());
        let rows = selected_route_source_rows(
            &JobType::ModelConvert,
            klein.as_object().unwrap(),
            &entries,
            "mlx",
            "macos",
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model_slot, "modelManifestEntries/0");
        assert_eq!(rows[0].component_id, "flux2_convert_base");
        assert!(
            matches!(&rows[0].source, SelectedSourceLocation::ConversionBorrow { repo, preferred_subdir }
            if repo == "other/klein" && preferred_subdir.as_deref() == Some("q4"))
        );
        let ltx = serde_json::json!({
            "converter":"ltx_video", "sourceRepo":"community/eros",
            "sourceFile":"checkpoint.safetensors", "baseRepo":"other/ltx",
            "upscalerFile":"custom-upscaler.safetensors"
        });
        assert_eq!(
            eligible_portable_routes(
                &JobType::ModelConvert,
                ltx.as_object().unwrap(),
                &entries,
                "auto",
            )
            .unwrap(),
            [("cpu".to_owned(), "macos".to_owned())],
        );
        let rows = selected_route_source_rows(
            &JobType::ModelConvert,
            ltx.as_object().unwrap(),
            &entries,
            "mlx",
            "macos",
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].component_id, "ltx_convert_upscaler");
        assert!(
            matches!(&rows[0].source, SelectedSourceLocation::DynamicHfFiles {repo,reference,files}
            if repo == "other/ltx" && reference == "main" && files == &["custom-upscaler.safetensors"])
        );
    }
    use serde_json::json;

    #[test]
    fn face_utilities_bind_the_catalog_face_stack_download() {
        let entry = json!({"id":"instantid_face_stack","type":"utility","componentOnly":true,
            "downloads":[{"provider":"huggingface","repo":"SceneWorks/instantid-mlx",
                "revision":"bca0cacf8e5e04529bb2b326a521361b02be84fd",
                "files":["scrfd_10g.safetensors","arcface_iresnet100.safetensors"]}]});
        let entries = [("modelManifestEntry".to_owned(), &entry)];
        let payload = JsonObject::new();
        for job in [
            JobType::KpsExtract,
            JobType::DatasetFaceAnalysis,
            JobType::FaceLikenessCompare,
        ] {
            let rows =
                selected_route_source_rows(&job, &payload, &entries, "candle", "linux").unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].component_id, "utility_face_stack");
            assert!(
                matches!(&rows[0].source, SelectedSourceLocation::ManifestDownload(row) if row["repo"] == "SceneWorks/instantid-mlx")
            );
        }
    }

    #[test]
    fn instantid_pose_uses_its_own_openpose_source() {
        let entry = json!({"id":"instantid_realvisxl"});
        let entries = [("modelManifestEntry".to_owned(), &entry)];
        let payload = json!({"model":"instantid_realvisxl","advanced":{"poses":[{"id":"front"}]}});
        let rows = selected_route_source_rows(
            &JobType::ImageGenerate,
            payload.as_object().unwrap(),
            &entries,
            "candle",
            "linux",
        )
        .unwrap();
        assert!(rows
            .iter()
            .any(|row| row.component_id == "instantid_openpose"));
        assert!(!rows.iter().any(|row| row.component_id == "strict_control"));
    }

    #[test]
    fn registered_local_pose_overlay_is_selected_as_strict_control() {
        let entry = json!({"id":"krea_2_turbo"});
        let entries = [("modelManifestEntry".to_owned(), &entry)];
        let payload = json!({
            "model":"krea_2_turbo",
            "advanced":{
                "poses":[{"id":"front"}],
                "controlWeights":{
                    "overlayId":"pose-approved",
                    "_catalogAuthorized":true,
                    "path":"/api-host/pose.safetensors"
                }
            }
        });
        let rows = selected_route_source_rows(
            &JobType::ImageGenerate,
            payload.as_object().unwrap(),
            &entries,
            "mlx",
            "macos",
        )
        .unwrap();
        assert!(rows.iter().any(|row| {
            row.model_slot == "modelManifestEntry"
                && row.component_id == "strict_control"
                && row.role == ArtifactMemberRole::Control
                && matches!(&row.source, SelectedSourceLocation::LocalPath(path) if path == "/api-host/pose.safetensors")
        }));
    }

    #[test]
    fn kolors_selects_ip_only_for_the_reference_loader_and_pid_for_candle_pose() {
        let entry = json!({"id":"kolors"});
        let entries = [("modelManifestEntry".to_owned(), &entry)];
        let pure_reference = json!({"model":"kolors","referenceAssetId":"asset"});
        let rows = selected_route_source_rows(
            &JobType::ImageGenerate,
            pure_reference.as_object().unwrap(),
            &entries,
            "candle",
            "linux",
        )
        .unwrap();
        assert!(rows
            .iter()
            .any(|row| row.component_id == "kolors_ip_adapter"));
        let pose = json!({"model":"kolors","referenceAssetId":"asset","advanced":{"poses":[{"id":"front"}],"usePid":true}});
        let rows = selected_route_source_rows(
            &JobType::ImageGenerate,
            pose.as_object().unwrap(),
            &entries,
            "candle",
            "linux",
        )
        .unwrap();
        assert!(!rows
            .iter()
            .any(|row| row.component_id == "kolors_ip_adapter"));
        assert!(rows.iter().any(|row| row.component_id == "pid_checkpoint"));
        assert!(rows.iter().any(|row| row.component_id == "pid_gemma"));
        let legacy_pose = json!({"model":"kolors","referenceAssetId":"asset","advanced":{"poses":[{"id":"front"}],"usePid":" TrUe "}});
        let legacy_rows = selected_route_source_rows(
            &JobType::ImageGenerate,
            legacy_pose.as_object().unwrap(),
            &entries,
            "candle",
            "linux",
        )
        .unwrap();
        assert!(legacy_rows
            .iter()
            .any(|row| row.component_id == "pid_checkpoint"));
        assert!(legacy_rows
            .iter()
            .any(|row| row.component_id == "pid_gemma"));
    }

    fn routes(job_type: JobType, payload: Value, entry: Value) -> Vec<(String, String)> {
        let entry_ref = [("modelManifestEntry".to_owned(), &entry)];
        eligible_portable_routes(
            &job_type,
            payload.as_object().expect("payload object"),
            &entry_ref,
            "auto",
        )
        .expect("routable job")
    }

    #[test]
    fn auto_image_routes_follow_both_native_claim_predicates() {
        let entry = json!({"id":"flux_schnell"});
        let actual = routes(
            JobType::ImageGenerate,
            json!({"model":"flux_schnell"}),
            entry,
        );
        assert_eq!(
            actual,
            vec![
                ("mlx".to_owned(), "macos".to_owned()),
                ("candle".to_owned(), "linux".to_owned()),
                ("candle".to_owned(), "windows".to_owned()),
            ]
        );
    }

    #[test]
    fn no_route_is_invented_for_unsupported_conditioned_request() {
        let entry = json!({"id":"chroma","type":"image"});
        let payload = json!({"model":"chroma","advanced":{"poses":[{}]}});
        let entry_ref = [("modelManifestEntry".to_owned(), &entry)];
        assert!(eligible_portable_routes(
            &JobType::ImageGenerate,
            payload.as_object().unwrap(),
            &entry_ref,
            "auto",
        )
        .is_err());
    }

    #[test]
    fn explicit_cpu_cannot_claim_generation() {
        let entry = json!({"id":"flux_schnell"});
        let entry_ref = [("modelManifestEntry".to_owned(), &entry)];
        let payload = json!({"model":"flux_schnell"});
        assert!(eligible_portable_routes(
            &JobType::ImageGenerate,
            payload.as_object().unwrap(),
            &entry_ref,
            "cpu",
        )
        .is_err());
    }

    #[test]
    fn explicit_native_gpu_limits_the_frozen_routes() {
        let entry = json!({"id":"flux_schnell"});
        let entry_ref = [("modelManifestEntry".to_owned(), &entry)];
        let payload = json!({"model":"flux_schnell"});
        let payload = payload.as_object().unwrap();
        assert_eq!(
            eligible_portable_routes(&JobType::ImageGenerate, payload, &entry_ref, "mlx").unwrap(),
            vec![("mlx".to_owned(), "macos".to_owned())]
        );
        assert_eq!(
            eligible_portable_routes(&JobType::ImageGenerate, payload, &entry_ref, "0").unwrap(),
            vec![
                ("candle".to_owned(), "linux".to_owned()),
                ("candle".to_owned(), "windows".to_owned())
            ]
        );
    }

    #[test]
    fn selected_sources_bind_default_control_tier_and_override() {
        let entry = json!({"id":"qwen_image"});
        let entries = [("modelManifestEntry".to_owned(), &entry)];
        let payload = json!({"model":"qwen_image","advanced":{"poses":[{}]}});
        let rows = selected_route_source_rows(
            &JobType::ImageGenerate,
            payload.as_object().unwrap(),
            &entries,
            "mlx",
            "macos",
        )
        .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].selected_tier.as_deref(), Some("q8"));
        assert!(
            matches!(&rows[0].source, SelectedSourceLocation::ManifestDownload(value) if value["files"][0] == "q8/model.safetensors")
        );
        let payload = json!({"model":"qwen_image","advanced":{"poses":[{}],"mlxQuantize":4}});
        let rows = selected_route_source_rows(
            &JobType::ImageGenerate,
            payload.as_object().unwrap(),
            &entries,
            "mlx",
            "macos",
        )
        .unwrap();
        assert_eq!(rows[0].selected_tier.as_deref(), Some("q4"));
    }

    #[test]
    fn cover_mode_binds_soft_component_only_when_selected() {
        let entry = json!({"id":"acestep_v15_turbo","downloads":[{"componentId":"sft_cover","required":"soft","repo":"ACE-Step/acestep-v15-xl-sft-diffusers","revision":"4bf7b60a63b27144f539f980927eeb89f5f912b0","files":["transformer/*"]}]});
        let entries = [("modelManifestEntry".to_owned(), &entry)];
        let cover = json!({"editMode":"cover"});
        let plain = json!({"editMode":"inpaint"});
        assert_eq!(
            selected_route_source_rows(
                &JobType::AudioGenerate,
                cover.as_object().unwrap(),
                &entries,
                "mlx",
                "macos"
            )
            .unwrap()
            .len(),
            1
        );
        assert!(selected_route_source_rows(
            &JobType::AudioGenerate,
            plain.as_object().unwrap(),
            &entries,
            "mlx",
            "macos"
        )
        .unwrap()
        .is_empty());
    }

    #[test]
    fn selected_decoder_requires_declared_soft_donor() {
        let entry = json!({"id":"qwen_image","decoders":{"byBackend":{"mlx":[{"id":"wan_2_1_vae","componentId":"vae"}]}},"downloads":[{"componentId":"vae","required":"soft","repo":"donor/vae","revision":"0123456789012345678901234567890123456789","files":["vae.safetensors"]}]});
        let entries = [("modelManifestEntry".to_owned(), &entry)];
        let payload = json!({"advanced":{"decoder":"wan_2_1_vae"}});
        let rows = selected_route_source_rows(
            &JobType::ImageGenerate,
            payload.as_object().unwrap(),
            &entries,
            "mlx",
            "macos",
        )
        .unwrap();
        assert_eq!(rows[0].component_id, "vae");
        assert!(selected_route_source_rows(
            &JobType::ImageGenerate,
            payload.as_object().unwrap(),
            &entries,
            "candle",
            "linux"
        )
        .is_err());
    }

    #[test]
    fn auxiliary_source_rows_follow_route_specific_request_shape() {
        let flux = json!({"id":"flux_dev"});
        let entries = [("modelManifestEntry".to_owned(), &flux)];
        let character =
            json!({"model":"flux_dev","mode":"character_image","referenceAssetId":"asset"});
        let rows = selected_route_source_rows(
            &JobType::ImageGenerate,
            character.as_object().unwrap(),
            &entries,
            "mlx",
            "macos",
        )
        .unwrap();
        assert_eq!(
            rows.iter()
                .map(|row| row.component_id.as_str())
                .collect::<Vec<_>>(),
            vec!["flux_ip_adapter", "flux_ip_encoder"]
        );
        let plain = json!({"model":"flux_dev","mode":"text_to_image"});
        assert!(selected_route_source_rows(
            &JobType::ImageGenerate,
            plain.as_object().unwrap(),
            &entries,
            "mlx",
            "macos"
        )
        .unwrap()
        .is_empty());
        let detail = json!({"model":"sdxl"});
        let sdxl = json!({"id":"sdxl"});
        let entries = [("modelManifestEntry".to_owned(), &sdxl)];
        let rows = selected_route_source_rows(
            &JobType::ImageDetail,
            detail.as_object().unwrap(),
            &entries,
            "candle",
            "linux",
        )
        .unwrap();
        assert_eq!(rows[0].component_id, "sdxl_detail_tile_control");
    }

    #[test]
    fn component_only_utility_entries_are_route_neutral_but_platform_bound() {
        let entry = json!({"id":"person_detector","type":"utility","componentOnly":true,"downloads":[{"provider":"huggingface","repo":"example/person","revision":"0123456789012345678901234567890123456789","files":["model.safetensors"],"platforms":["linux"]}]});
        let entries = [("modelManifestEntries/0".to_owned(), &entry)];
        assert!(entries_support_route(
            &JobType::CatalogAnalysis,
            &entries,
            "candle",
            "linux"
        ));
        assert!(!entries_support_route(
            &JobType::CatalogAnalysis,
            &entries,
            "mlx",
            "macos"
        ));
        let non_component = json!({"id":"unroutable_utility_source","type":"utility","downloads":[{"provider":"huggingface","repo":"example/person","revision":"0123456789012345678901234567890123456789","files":["model.safetensors"],"platforms":["linux"]}]});
        let entries = [("modelManifestEntries/0".to_owned(), &non_component)];
        assert!(entries_support_route(
            &JobType::CatalogAnalysis,
            &entries,
            "candle",
            "linux"
        ));
        assert!(!entries_support_route(
            &JobType::ImageGenerate,
            &entries,
            "candle",
            "linux"
        ));
    }

    #[test]
    fn catalog_analysis_binds_nonroutable_detector_sources_on_native_capability_routes() {
        let person = json!({"id":"person_detector","type":"utility","downloads":[{"provider":"huggingface","repo":"example/person","revision":"0123456789012345678901234567890123456789","files":["person.onnx"]}]});
        let pose = json!({"id":"dwpose_pose_detector","type":"utility","downloads":[{"provider":"huggingface","repo":"example/pose","revision":"0123456789012345678901234567890123456789","files":["pose.onnx"]}]});
        let face = json!({"id":"instantid_face_stack","type":"utility","componentOnly":true,"downloads":[{"provider":"huggingface","repo":"example/face","revision":"0123456789012345678901234567890123456789","files":["face.safetensors"]}]});
        let entries = [
            ("modelManifestEntries/0".to_owned(), &person),
            ("modelManifestEntries/1".to_owned(), &pose),
            ("modelManifestEntries/2".to_owned(), &face),
        ];
        let payload = json!({"catalogId":"catalog"});
        let routes = eligible_portable_routes(
            &JobType::CatalogAnalysis,
            payload.as_object().unwrap(),
            &entries,
            "auto",
        )
        .unwrap();
        assert_eq!(routes.len(), 3);
        assert_eq!(routes[0], ("mlx".to_owned(), "macos".to_owned()));
    }

    #[test]
    fn moss_loader_components_get_stable_bound_member_ids() {
        let entry = json!({"id":"moss_image_sft","downloads":[
            {"componentId":"moss-dmd-time","provider":"huggingface","repo":"test/dmd","revision":"0123456789012345678901234567890123456789","files":["time_embedding.safetensors"]},
            {"componentId":"moss-image-edit","provider":"huggingface","repo":"test/edit","revision":"0123456789012345678901234567890123456789","files":["adapter_config.json","adapter_model.safetensors"]}
        ]});
        let entries = [("modelManifestEntry".to_owned(), &entry)];
        let payload = json!({"model":"moss_image_sft"});
        let rows = selected_route_source_rows(
            &JobType::ImageGenerate,
            payload.as_object().unwrap(),
            &entries,
            "mlx",
            "macos",
        )
        .unwrap();
        assert_eq!(
            rows.iter()
                .map(|row| row.component_id.as_str())
                .collect::<Vec<_>>(),
            vec!["moss-dmd-time", "moss-image-edit"]
        );
        assert!(rows
            .iter()
            .all(|row| matches!(row.source, SelectedSourceLocation::ManifestDownload(_))));
    }

    #[test]
    fn iris_depth_parent_freezes_one_preferred_conditioning_source() {
        let plan = json!({
            "schemaVersion":1,"planVersion":1,"jobId":"job",
            "target":{"targetId":"target","kernel":"iris_train","family":"iris","modality":"image","outputKind":"lora","baseModel":"iris_3b_depth","baseModelPath":"/models/iris"},
            "dataset":{"datasetId":"dataset","datasetVersion":1,"rootPath":"/dataset","items":[]},
            "config":{"rank":16,"alpha":32,"learningRate":0.0001,"steps":10,"batchSize":1,"gradientAccumulation":1,"resolution":512,"saveEvery":10,"seed":1,"optimizer":"adamw8bit","advanced":{"depthInit":"parent"}},
            "output":{"loraId":"lora","outputDir":"/output","fileName":"lora.safetensors","format":"safetensors","triggerWords":[]},
            "provenance":{"datasetId":"dataset","datasetVersion":1,"targetId":"target","baseModel":"iris_3b_depth","configSnapshot":{},"outputLoraId":"lora","sourceJobId":"job","createdAt":"2026-11-07T00:00:00Z"}
        });
        let manifest =
            sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc").unwrap();
        let entry = manifest["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == "iris_3b_depth")
            .unwrap();
        let entries = [("baseModelManifestEntry".to_owned(), entry)];
        let payload = json!({"plan":plan.clone()});
        let rows = selected_route_source_rows(
            &JobType::LoraTrain,
            payload.as_object().unwrap(),
            &entries,
            "mlx",
            "macos",
        )
        .unwrap();
        assert!(rows
            .iter()
            .any(|row| row.component_id == "iris_parent_backbone"));
        assert!(rows
            .iter()
            .any(|row| row.component_id == "iris_depth_parent"
                && matches!(
                    row.source,
                    SelectedSourceLocation::PreferredWorkerClaim {
                        preferred_id: "iris_depth_export",
                        fallback_id: "iris_generation_text_encoder"
                    }
                )));
        let mut restore_plan = plan;
        restore_plan["target"]["baseModel"] = json!("iris_3b_restore");
        restore_plan["config"]["advanced"] = json!({"iris_restore":{"init":"parent"}});
        let restore_entry = manifest["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["id"] == "iris_3b_restore")
            .unwrap();
        let restore_entries = [("baseModelManifestEntry".to_owned(), restore_entry)];
        let restore_payload = json!({"plan":restore_plan});
        let restore_rows = selected_route_source_rows(
            &JobType::LoraTrain,
            restore_payload.as_object().unwrap(),
            &restore_entries,
            "mlx",
            "macos",
        )
        .unwrap();
        let ids: std::collections::BTreeSet<_> = restore_rows
            .iter()
            .map(|row| row.component_id.as_str())
            .collect();
        assert!(ids.contains("iris_parent_backbone"));
        for id in [
            "iris_restore_train_vgg16",
            "iris_restore_train_lpips_lin",
            "iris_restore_train_critic_trunk",
        ] {
            assert!(ids.contains(id), "{id} must be bound");
        }
    }
}
