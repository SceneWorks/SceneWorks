//! Portable admission routes follow the same request predicates as worker claim.

use sceneworks_core::contracts::{JobType, JsonObject};
use sceneworks_core::jobs_store::{native_route_claims_request, NON_GPU_JOB_TYPES};
use sceneworks_core::model_artifacts::ArtifactMemberRole;
use serde_json::Value;

use crate::ApiError;

#[derive(Clone, Debug)]
pub(crate) enum SelectedSourceLocation {
    ManifestDownload(Value),
    LocalPath(String),
    /// Admission resolves the request-selected repository HEAD to an immutable revision.
    DynamicHfRepo(String),
    /// An operator Gemma override wins when a registered worker advertises it; otherwise
    /// admission binds the exact selected LTX bundle's pinned sibling Gemma download.
    OperatorOverrideOrManifest {
        source_id: &'static str,
        default: Value,
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

/// Source additions opened by a route but omitted from its always-required model closure.
/// A missing selected declaration is an admission refusal, never an implicit worker download.
pub(crate) fn selected_route_source_rows(
    job_type: &JobType,
    payload: &JsonObject,
    entries: &[(String, &Value)],
    backend: &str,
    _platform: &str,
) -> Result<Vec<SelectedRouteSource>, String> {
    if matches!(
        job_type,
        JobType::KpsExtract | JobType::DatasetFaceAnalysis | JobType::FaceLikenessCompare
    ) {
        return Err(format!(
            "{} identity detector sources are not yet bound",
            job_type.as_str()
        ));
    }
    let mut rows = Vec::new();
    let advanced = payload.get("advanced").and_then(Value::as_object);
    let has_reference = payload
        .get("referenceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|id| !id.trim().is_empty());
    let character_reference =
        payload.get("mode").and_then(Value::as_str) == Some("character_image") && has_reference;
    if advanced
        .and_then(|value| value.get("usePid"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        return Err("selected PiD decoder sources are not yet bound".to_owned());
    }
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
            .ok_or_else(|| {
                format!(
                    "training base {} lacks a captured manifest entry",
                    plan.target.base_model
                )
            })?;
        if plan.target.base_model == "ltx_2_3" {
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
        if matches!(id, "instantid_realvisxl" | "pulid_flux_dev") {
            return Err(format!("{id} identity provider sources are not yet bound"));
        }
        if id == "kolors" && has_reference {
            return Err("Kolors IP-Adapter source is not yet bound".to_owned());
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
            if slot == "modelManifestEntry"
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
                .filter(|platform| entries_support_route(entries, backend, platform))
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

fn entries_support_route(entries: &[(String, &Value)], backend: &str, platform: &str) -> bool {
    entries.iter().all(|(_, entry)| {
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
                    .map_or(true, |platforms| {
                        platforms
                            .iter()
                            .any(|value| value.as_str() == Some(platform))
                    })
            });
        if !downloads_support_platform {
            return false;
        }
        if entry.get("componentOnly").and_then(Value::as_bool) == Some(true)
            && entry.get("type").and_then(Value::as_str) == Some("utility")
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
    use serde_json::json;

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
        assert!(entries_support_route(&entries, "candle", "linux"));
        assert!(!entries_support_route(&entries, "mlx", "macos"));
    }
}
