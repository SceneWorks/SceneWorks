//! Pure selected auxiliary source IDs shared by API admission and worker guard.

use crate::contracts::{JobType, JsonObject};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PortableAuxSourceShape {
    File,
    Directory,
}

/// The image loader accepts both a JSON boolean and the legacy string spelling.
pub fn pid_requested(value: Option<&Value>) -> bool {
    value.is_some_and(|value| {
        value.as_bool().unwrap_or_else(|| {
            value
                .as_str()
                .is_some_and(|text| text.trim().eq_ignore_ascii_case("true"))
        })
    })
}

/// The Candle Krea loader treats this flag as the INT8-ConvRot tier even without a quant field.
pub fn krea_convrot_requested(
    model_id: &str,
    backend: &str,
    platform: &str,
    flag: Option<&Value>,
) -> bool {
    model_id == "krea_2_turbo"
        && backend == "candle"
        && platform != "macos"
        && flag.and_then(Value::as_bool) == Some(true)
}

/// Path-free key for a worker's registered local digest source. The API computes the same key
/// from the frozen route's logical catalog identity before deciding claim eligibility.
pub fn registered_digest_claim_id(
    kind: &str,
    model_id: &str,
    component_id: Option<&str>,
    file_ledger_sha256: &str,
) -> Result<String, String> {
    if !matches!(
        kind,
        "model" | "model-bytes" | "lora" | "control" | "component"
    ) || model_id.trim().is_empty()
        || model_id != model_id.trim()
        || model_id.chars().any(char::is_control)
        || (kind == "component") != component_id.is_some()
        || component_id.is_some_and(|id| {
            id.trim().is_empty() || id != id.trim() || id.chars().any(char::is_control)
        })
        || file_ledger_sha256
            .strip_prefix("sha256:")
            .is_none_or(|digest| {
                digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
    {
        return Err("invalid registered digest source identity".to_owned());
    }
    let identity =
        serde_json::to_vec(&(kind, model_id, component_id)).map_err(|error| error.to_string())?;
    Ok(format!(
        "registered_digest:{kind}:{}:{file_ledger_sha256}",
        crate::hashing::sha256_bytes(&identity)
    ))
}

/// Private worker claim key for a FLUX.2 conversion base selected from this worker's installed
/// repository and concrete tier. The provenance and file ledger still carry the byte identity.
pub fn flux2_convert_base_claim_id(
    repo: &str,
    revision: &str,
    tier: &str,
) -> Result<String, String> {
    crate::hf_repo_id::parse(repo).map_err(str::to_owned)?;
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("FLUX.2 conversion base revision is not immutable".to_owned());
    }
    if tier.is_empty()
        || matches!(tier, "." | "..")
        || !tier
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err("unsupported FLUX.2 conversion base tier".to_owned());
    }
    Ok(format!("flux2_convert_base:{repo}:{revision}:{tier}"))
}

pub fn parse_flux2_convert_base_claim_id(source_id: &str) -> Option<(&str, &str, &str)> {
    let value = source_id.strip_prefix("flux2_convert_base:")?;
    let (repo_revision, tier) = value.rsplit_once(':')?;
    let (repo, revision) = repo_revision.rsplit_once(':')?;
    (flux2_convert_base_claim_id(repo, revision, tier)
        .ok()
        .as_deref()
        == Some(source_id))
    .then_some((repo, revision, tier))
}

/// Caller-staged trainer components at the pinned native provider registry. The manifest owns
/// repo/revision/files; this is the backend descriptor's required-component ID projection.
pub fn trainer_required_component_ids(base_model: &str, backend: &str) -> &'static [&'static str] {
    match (base_model, backend) {
        ("sdxl", "candle") => &["tokenizer_clip_l", "tokenizer_clip_bigg", "vae_fp16_fix"],
        ("mage_flow_base", "mlx" | "candle") => &["text_encoder", "vae"],
        ("iris_3b", "mlx" | "candle") => &["text_encoder"],
        ("moss_image_base", "mlx" | "candle") => {
            &[gen_core::moss_image::MOSS_SHARED_SNAPSHOT_COMPONENT]
        }
        _ => &[],
    }
}

/// Pinned auxiliary models opened by the selected native training techniques. This uses the
/// same strict plan parsers and model selectors as the trainer, before any loader path is opened.
pub fn selected_training_aux_models(
    plan: &crate::training::TrainingPlan,
    engine_id: &str,
) -> Result<Vec<&'static crate::training::depth_anchoring::AuxTrainingModel>, String> {
    use crate::training::{
        body_losses, depth_anchoring, face_losses, latent_perceptual, training_adapter,
    };
    use depth_anchoring::{AuxTrainingModel, X0DecoderSource};
    let advanced = &plan.config.advanced;
    let mut models: Vec<&'static AuxTrainingModel> = Vec::new();
    let mut need_decoder = false;
    if let Some(depth) =
        depth_anchoring::depth_anchoring_settings(advanced).map_err(|error| error.to_string())?
    {
        need_decoder = true;
        models.push(
            depth_anchoring::depth_anything_v2_model(depth.model)
                .ok_or_else(|| format!("unknown Depth Anything V2 size {}", depth.model))?,
        );
    }
    let body = body_losses::body_loss_settings(advanced).map_err(|error| error.to_string())?;
    if body.any_enabled() {
        need_decoder = true;
        models.push(&body_losses::VITPOSE_PLUS_BASE_MODEL);
        for loss in body_losses::BodyLoss::ALL {
            if body.schedule(loss).is_some() {
                if let Some(model) = loss.own_model() {
                    models.push(model);
                }
            }
        }
    }
    if latent_perceptual::latent_loss_settings(&latent_perceptual::VAE_ANCHOR, advanced)
        .map_err(|error| error.to_string())?
        .is_some()
    {
        need_decoder = true;
        models.push(&latent_perceptual::FLUX2_VAE_MODEL);
    }
    if latent_perceptual::latent_loss_settings(&latent_perceptual::LATENT_LPIPS, advanced)
        .map_err(|error| error.to_string())?
        .is_some()
    {
        models.push(
            latent_perceptual::latent_lpips_model_for_trainer(engine_id)
                .ok_or_else(|| format!("{engine_id} has no E-LatentLPIPS model"))?,
        );
    }
    let identity =
        face_losses::identity_loss_settings(advanced).map_err(|error| error.to_string())?;
    let landmark =
        face_losses::face_landmark_loss_settings(advanced).map_err(|error| error.to_string())?;
    if identity.is_some() || landmark.is_some() {
        need_decoder = true;
        models.push(&face_losses::FACE_STACK_SCRFD);
        if identity.is_some() {
            models.push(&face_losses::FACE_STACK_ARCFACE);
        }
        if landmark.is_some() {
            models.push(&face_losses::FACEMESH_V2_MODEL);
        }
    }
    if need_decoder {
        match depth_anchoring::x0_decoder_for_trainer(engine_id)
            .ok_or_else(|| format!("{engine_id} has no x0 decoder"))?
        {
            X0DecoderSource::Catalog(model) => models.push(model),
            X0DecoderSource::BaseModelVae => {}
        }
    }
    if let Some(model) =
        training_adapter::training_adapter_model(advanced).map_err(|error| error.to_string())?
    {
        models.push(model);
    }
    models.sort_by_key(|model| (model.id, model.file));
    models.dedup_by_key(|model| (model.id, model.file));
    Ok(models)
}

pub fn portable_aux_source_shape(source_id: &str, entry: &Value) -> Option<PortableAuxSourceShape> {
    use PortableAuxSourceShape::{Directory, File};
    // Training selectors return separately cataloged models, not component rows on the base
    // model entry. Their native loaders receive the pinned snapshot directory and open the
    // declared weight file inside it.
    if is_training_aux_model_id(source_id) {
        return Some(Directory);
    }
    match source_id {
        "strict_control"
        | "sdxl_detail_tile_control"
        | "flux_ip_adapter"
        | "sdxl_ip_adapter"
        | "pulid_adapter"
        | "pulid_eva"
        | "pulid_bisenet"
        | "pulid_scrfd"
        | "pulid_arcface"
        | "veda_predictor"
        | "iris_restore_train_vgg16"
        | "iris_restore_train_lpips_lin"
        | "iris_restore_train_critic_trunk" => Some(File),
        "depth_anything_v2_small"
        | "flux_ip_encoder"
        | "sdxl_ip_encoder"
        | "prompt_upsampler"
        | "sft_cover"
        | "repa_teacher"
        | "ltx_amoral_gemma_3_12b"
        | "ltx_training_text_encoder"
        | "iris_depth_parent"
        | "iris_parent_backbone"
        | "flux2_convert_base"
        | "ltx_convert_upscaler" => Some(Directory),
        "instantid_mlx"
        | "instantid_controlnet"
        | "instantid_openpose"
        | "pid_gemma"
        | "kolors_ip_adapter"
        | "utility_face_stack" => Some(Directory),
        "pid_checkpoint" => Some(Directory),
        id if crate::moss_image::is_runtime_optional_component_id(id) => {
            let row = entry
                .get("downloads")?
                .as_array()?
                .iter()
                .find(|row| row.get("componentId").and_then(Value::as_str) == Some(id))?;
            let files = row.get("files")?.as_array()?;
            if files.len() == 1 && files[0].as_str().is_some_and(|file| !file.contains('*')) {
                Some(File)
            } else {
                Some(Directory)
            }
        }
        id if entry
            .get("downloads")
            .and_then(Value::as_array)
            .is_some_and(|downloads| {
                downloads
                    .iter()
                    .any(|row| row.get("componentId").and_then(Value::as_str) == Some(id))
            }) =>
        {
            let row = entry
                .get("downloads")?
                .as_array()?
                .iter()
                .find(|row| row.get("componentId").and_then(Value::as_str) == Some(id))?;
            let files = row.get("files")?.as_array()?;
            if files.len() == 1
                && files[0]
                    .as_str()
                    .is_some_and(|file| !file.contains(['*', '?', '[']))
            {
                Some(File)
            } else {
                Some(Directory)
            }
        }
        _ => entry
            .pointer("/decoders/byBackend")
            .and_then(Value::as_object)
            .into_iter()
            .flat_map(|backends| backends.values())
            .filter_map(Value::as_array)
            .flatten()
            .any(|option| option.get("componentId").and_then(Value::as_str) == Some(source_id))
            .then_some(File),
    }
}

fn is_training_aux_model_id(source_id: &str) -> bool {
    use crate::training::{
        body_losses, depth_anchoring, face_losses, latent_perceptual, training_adapter,
    };
    depth_anchoring::X0_DECODER_MODELS
        .iter()
        .chain(depth_anchoring::DEPTH_ANYTHING_V2_MODELS.iter())
        .chain(latent_perceptual::ELATENTLPIPS_MODELS.iter())
        .chain(training_adapter::TRAINING_ADAPTER_MODELS.iter())
        .chain([
            &body_losses::VITPOSE_PLUS_BASE_MODEL,
            &body_losses::HYBRIK_RESNET34_MODEL,
            &body_losses::SAPIENS_NORMAL_0_3B_MODEL,
            &face_losses::FACE_STACK_SCRFD,
            &face_losses::FACEMESH_V2_MODEL,
            &latent_perceptual::FLUX2_VAE_MODEL,
        ])
        .any(|model| model.id == source_id)
}

pub fn pid_backbone_for(model: &str) -> Option<&'static str> {
    match model {
        // qwenimage (sc-7845): Qwen-Image T2I + its strict-pose control variant (both the `qwen_image`
        // model id), every Qwen-Image-Edit variant (all → the one `qwen_image_edit` engine), Krea 2.
        "qwen_image"
        | "qwen_image_edit"
        | "qwen_image_edit_2509"
        | "qwen_image_edit_2511"
        | "qwen_image_edit_2511_lightning"
        | "krea_2_turbo"
        | "krea_2_raw" => Some("qwenimage"),
        // flux (sc-7846): FLUX.1, Boogu-Image, Chroma, and Z-Image — Z-Image is in the FLUX.1 VAE latent
        // space (PiD's zimage tags alias the flux checkpoint), not the qwenimage space.
        "flux_dev" | "flux_schnell" | "pulid_flux_dev" | "boogu_image" | "boogu_image_turbo"
        | "boogu_image_edit" | "chroma1_hd" | "chroma1_base" | "chroma1_flash"
        | "z_image_turbo" | "z_image_edit" => Some("flux"),
        // flux2 (sc-7847): FLUX.2-dev, every klein-9b variant, Lens, Ideogram 4 (packed 128-ch latent).
        "flux2_dev"
        | "flux2_klein_9b"
        | "flux2_klein_9b_kv"
        | "flux2_klein_9b_true_v2"
        | "lens"
        | "lens_turbo"
        | "ideogram_4"
        | "ideogram_4_turbo" => Some("flux2"),
        // sdxl (sc-7848): SDXL base, RealVisXL (+ Lightning), Kolors (reuses the SDXL VAE). InstantID
        // (sc-8370/8371) composes the SDXL VAE too, so its Angles/Poses latents decode through the same
        // `sdxl` PiD student — the engine's `InstantId::with_pid` loads that one checkpoint.
        // Illustrious-XL (epic 10609) ships the stock SDXL VAE (scaling_factor 0.13025 — verified on
        // the converted turnkey), so its latents decode through the same `sdxl` PiD student.
        "sdxl"
        | "realvisxl"
        | "realvisxl_lightning"
        | "kolors"
        | "instantid_realvisxl"
        | "illustrious_xl_v1"
        | "illustrious_xl_v2" => Some("sdxl"),
        _ => None,
    }
}

pub fn required_portable_aux_source_ids(
    job_type: &JobType,
    payload: &JsonObject,
    entry: &Value,
    backend: &str,
) -> Result<Vec<String>, String> {
    let id = entry
        .get("id")
        .and_then(Value::as_str)
        .ok_or("model entry lacks id")?;
    if id == "instantid_face_stack"
        && matches!(
            job_type,
            JobType::KpsExtract
                | JobType::DatasetFaceAnalysis
                | JobType::FaceLikenessCompare
                | JobType::CatalogAnalysis
        )
    {
        return Ok(vec!["utility_face_stack".to_owned()]);
    }
    let primary_model = payload.get("model").and_then(Value::as_str) == Some(id);
    let advanced = payload.get("advanced").and_then(Value::as_object);
    let mut ids = Vec::new();
    if *job_type == JobType::ModelConvert {
        match payload.get("converter").and_then(Value::as_str) {
            Some("flux2_klein_diffusers") => ids.push("flux2_convert_base".to_owned()),
            Some("ltx_video") => ids.push("ltx_convert_upscaler".to_owned()),
            _ => {}
        }
    }
    let has_ref = payload
        .get("referenceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty());
    let character_reference =
        payload.get("mode").and_then(Value::as_str) == Some("character_image") && has_ref;
    if primary_model && matches!(job_type, JobType::ImageGenerate | JobType::ImageEdit) {
        if id == "instantid_realvisxl" {
            ids.extend(["instantid_mlx", "instantid_controlnet"].map(str::to_owned));
            if advanced
                .and_then(|value| value.get("poses"))
                .and_then(Value::as_array)
                .is_some_and(|poses| poses.iter().any(Value::is_object))
            {
                ids.push("instantid_openpose".to_owned());
            }
        }
        if id == "pulid_flux_dev" {
            ids.extend(
                [
                    "pulid_adapter",
                    "pulid_eva",
                    "pulid_bisenet",
                    "pulid_scrfd",
                    "pulid_arcface",
                ]
                .map(str::to_owned),
            );
        }
        if id == "kolors"
            && has_ref
            && (backend == "mlx"
                || !advanced
                    .and_then(|value| value.get("poses"))
                    .and_then(Value::as_array)
                    .is_some_and(|poses| poses.iter().any(Value::is_object)))
        {
            ids.push("kolors_ip_adapter".to_owned());
        }
        if pid_backbone_for(id).is_some()
            && (id != "kolors"
                || (backend == "candle"
                    && advanced
                        .and_then(|value| value.get("poses"))
                        .and_then(Value::as_array)
                        .is_some_and(|poses| poses.iter().any(Value::is_object))))
            && (id != "pulid_flux_dev" || backend == "candle")
            && pid_requested(advanced.and_then(|value| value.get("usePid")))
        {
            ids.extend(["pid_checkpoint", "pid_gemma"].map(str::to_owned));
        }
        if id.starts_with("moss_image") {
            ids.extend(
                entry
                    .get("downloads")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|row| row.get("componentId").and_then(Value::as_str))
                    .filter(|component| {
                        crate::moss_image::is_runtime_optional_component_id(component)
                    })
                    .map(str::to_owned),
            );
        }
        if character_reference && matches!(id, "flux_dev" | "flux_schnell") {
            ids.extend(["flux_ip_adapter", "flux_ip_encoder"].map(str::to_owned));
        }
        if character_reference
            && matches!(
                id,
                "sdxl" | "realvisxl" | "illustrious_xl_v1" | "illustrious_xl_v2"
            )
        {
            ids.extend(["sdxl_ip_adapter", "sdxl_ip_encoder"].map(str::to_owned));
        }
        if let Some(decoder) = advanced
            .and_then(|value| value.get("decoder"))
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty() && *value != "native")
        {
            let component = entry
                .pointer(&format!("/decoders/byBackend/{backend}"))
                .and_then(Value::as_array)
                .and_then(|options| {
                    options
                        .iter()
                        .find(|option| option.get("id").and_then(Value::as_str) == Some(decoder))
                })
                .and_then(|option| option.get("componentId"))
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{id} cannot use decoder {decoder} on {backend}"))?;
            ids.push(component.to_owned());
        }
        if matches!(id, "flux3_klein" | "flux3_klein_base")
            && advanced
                .and_then(|value| value.get("enhancePrompt"))
                .and_then(Value::as_bool)
                == Some(true)
        {
            ids.push("prompt_upsampler".to_owned());
        }
        if id != "instantid_realvisxl"
            && advanced
                .and_then(|value| value.get("poses"))
                .and_then(Value::as_array)
                .is_some_and(|poses| !poses.is_empty())
        {
            ids.push("strict_control".to_owned());
            if advanced
                .and_then(|value| value.get("controlMode"))
                .and_then(Value::as_str)
                == Some("depth")
            {
                ids.push("depth_anything_v2_small".to_owned());
            }
        }
    }
    if primary_model && *job_type == JobType::ImageDetail {
        ids.push("sdxl_detail_tile_control".to_owned());
    }
    if payload.get("editMode").and_then(Value::as_str) == Some("cover") && id.starts_with("acestep")
    {
        ids.push("sft_cover".to_owned());
    }
    if matches!(id, "minimax_h3" | "minimax_h3_ref")
        && advanced
            .and_then(|value| value.get("veda"))
            .and_then(|value| value.get("enabled"))
            .and_then(Value::as_bool)
            == Some(true)
    {
        ids.push("veda_predictor".to_owned());
    }
    if id.starts_with("ltx")
        && backend == "mlx"
        && advanced.is_some_and(|value| {
            value.get("useUncensoredEnhancer").and_then(Value::as_bool) == Some(true)
                || value.get("textEncoderModel").and_then(Value::as_str)
                    == Some("ltx_amoral_gemma_3_12b")
        })
    {
        ids.push("ltx_amoral_gemma_3_12b".to_owned());
    }
    if matches!(job_type, JobType::LoraTrain | JobType::ControlTraining) {
        let plan: crate::training::TrainingPlan = serde_json::from_value(
            payload
                .get("plan")
                .cloned()
                .ok_or("training job lacks resolved plan")?,
        )
        .map_err(|error| format!("invalid resolved training plan: {error}"))?;
        if plan.target.base_model == id {
            ids.extend(
                trainer_required_component_ids(id, backend)
                    .iter()
                    .map(|id| (*id).to_owned()),
            );
            if id == "iris_3b_depth"
                && crate::training::iris::depth::depth_init(&plan.config)
                    .map_err(|error| error.to_string())?
                    == crate::training::iris::depth::DEPTH_INIT_PARENT
            {
                ids.extend(["iris_parent_backbone", "iris_depth_parent"].map(str::to_owned));
            }
            if id == "iris_3b_restore" {
                ids.extend(
                    crate::training::iris::restore::RESTORE_TRAINING_RESOURCE_MODELS
                        .iter()
                        .map(|(catalog_id, _)| (*catalog_id).to_owned()),
                );
                if crate::training::iris::restore::restore_options(&plan.config)
                    .map_err(|error| error.to_string())?
                    .get("init")
                    .and_then(Value::as_str)
                    == Some("parent")
                {
                    ids.push("iris_parent_backbone".to_owned());
                }
            }
            if id == "ltx_2_3" {
                ids.push("ltx_training_text_encoder".to_owned());
            }
            if crate::training::iris::is_iris_training_kernel(&plan.target.kernel) {
                let options = crate::training::iris::iris_options(&plan.config)
                    .map_err(|error| error.to_string())?;
                if crate::training::iris::curricula::iris_repa_enabled(&options) {
                    ids.push("repa_teacher".to_owned());
                }
            }
        }
    }
    ids.sort();
    ids.dedup();
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_requires_exact_selected_auxiliary_and_directory_shape() {
        let entry = serde_json::json!({"id":"model_convert:community/source"});
        for (converter, expected) in [
            ("flux2_klein_diffusers", "flux2_convert_base"),
            ("ltx_video", "ltx_convert_upscaler"),
        ] {
            let payload = serde_json::json!({"converter":converter});
            assert_eq!(
                required_portable_aux_source_ids(
                    &JobType::ModelConvert,
                    payload.as_object().unwrap(),
                    &entry,
                    "mlx"
                )
                .unwrap(),
                vec![expected]
            );
            assert_eq!(
                portable_aux_source_shape(expected, &entry),
                Some(PortableAuxSourceShape::Directory)
            );
        }
    }
    use serde_json::json;

    #[test]
    fn flux2_conversion_claim_key_preserves_custom_tier_and_revision() {
        let revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let key = flux2_convert_base_claim_id("guardfx/borrow", revision, "custom-q5").unwrap();
        assert_eq!(
            parse_flux2_convert_base_claim_id(&key),
            Some(("guardfx/borrow", revision, "custom-q5"))
        );
        assert!(flux2_convert_base_claim_id("guardfx/borrow", revision, "../q4").is_err());
        assert!(flux2_convert_base_claim_id("guardfx/borrow", "main", "q4").is_err());
    }

    #[test]
    fn registered_digest_claim_key_is_path_free_and_slot_stable() {
        let ledger_a = format!("sha256:{}", "a".repeat(64));
        let ledger_b = format!("sha256:{}", "b".repeat(64));
        let model = registered_digest_claim_id("model", "imported-model", None, &ledger_a).unwrap();
        let component =
            registered_digest_claim_id("component", "imported-model", Some("vae"), &ledger_a)
                .unwrap();
        assert_ne!(model, component);
        assert_eq!(
            model,
            registered_digest_claim_id("model", "imported-model", None, &ledger_a).unwrap()
        );
        assert!(!model.contains("imported-model"));
        assert_ne!(
            model,
            registered_digest_claim_id("model", "imported-model", None, &ledger_b).unwrap()
        );
        let bytes =
            registered_digest_claim_id("model-bytes", "imported-model", None, &ledger_a).unwrap();
        assert_ne!(bytes, model);
        assert!(registered_digest_claim_id(
            "model-bytes",
            "imported-model",
            Some("layer"),
            &ledger_a
        )
        .is_err());
        assert!(
            registered_digest_claim_id("component", "imported-model", None, &ledger_a).is_err()
        );
    }

    #[test]
    fn identity_and_pid_sources_match_selected_native_routes() {
        let face = json!({"id":"instantid_face_stack"});
        let empty = json!({});
        for job in [
            JobType::KpsExtract,
            JobType::DatasetFaceAnalysis,
            JobType::FaceLikenessCompare,
        ] {
            assert_eq!(
                required_portable_aux_source_ids(&job, empty.as_object().unwrap(), &face, "candle")
                    .unwrap(),
                vec!["utility_face_stack"]
            );
        }
        assert_eq!(
            portable_aux_source_shape("utility_face_stack", &face),
            Some(PortableAuxSourceShape::Directory)
        );
        let instant = json!({"id":"instantid_realvisxl"});
        let selected = json!({"model":"instantid_realvisxl","advanced":{"poses":[{"id":"front"}],"usePid":true}});
        let ids = required_portable_aux_source_ids(
            &JobType::ImageGenerate,
            selected.as_object().unwrap(),
            &instant,
            "candle",
        )
        .unwrap();
        for id in [
            "instantid_mlx",
            "instantid_controlnet",
            "instantid_openpose",
            "pid_checkpoint",
            "pid_gemma",
        ] {
            assert!(ids.contains(&id.to_owned()), "missing {id}");
            assert_eq!(
                portable_aux_source_shape(id, &instant),
                Some(PortableAuxSourceShape::Directory)
            );
        }
        let legacy = json!({"model":"instantid_realvisxl","advanced":{"poses":[{"id":"front"}],"usePid":" TrUe "}});
        let legacy_ids = required_portable_aux_source_ids(
            &JobType::ImageGenerate,
            legacy.as_object().unwrap(),
            &instant,
            "candle",
        )
        .unwrap();
        assert_eq!(legacy_ids, ids);
        assert!(!ids.contains(&"strict_control".to_owned()));
        let pulid = json!({"id":"pulid_flux_dev"});
        let selected = json!({"model":"pulid_flux_dev","advanced":{"usePid":true}});
        let candle = required_portable_aux_source_ids(
            &JobType::ImageGenerate,
            selected.as_object().unwrap(),
            &pulid,
            "candle",
        )
        .unwrap();
        let mlx = required_portable_aux_source_ids(
            &JobType::ImageGenerate,
            selected.as_object().unwrap(),
            &pulid,
            "mlx",
        )
        .unwrap();
        assert!(candle.contains(&"pid_checkpoint".to_owned()));
        assert!(!mlx.contains(&"pid_checkpoint".to_owned()));
        for id in [
            "pulid_adapter",
            "pulid_eva",
            "pulid_bisenet",
            "pulid_scrfd",
            "pulid_arcface",
        ] {
            assert!(mlx.contains(&id.to_owned()));
            assert_eq!(
                portable_aux_source_shape(id, &pulid),
                Some(PortableAuxSourceShape::File)
            );
        }
        let kolors = json!({"id":"kolors"});
        let selected = json!({"model":"kolors","referenceAssetId":"asset"});
        assert!(required_portable_aux_source_ids(
            &JobType::ImageGenerate,
            selected.as_object().unwrap(),
            &kolors,
            "mlx"
        )
        .unwrap()
        .contains(&"kolors_ip_adapter".to_owned()));
        let candle_pose = json!({"model":"kolors","referenceAssetId":"asset","advanced":{"poses":[{"id":"front"}]}});
        let mut candle_pid = candle_pose.as_object().unwrap().clone();
        candle_pid.get_mut("advanced").unwrap()["usePid"] = json!(true);
        let ids = required_portable_aux_source_ids(
            &JobType::ImageGenerate,
            &candle_pid,
            &kolors,
            "candle",
        )
        .unwrap();
        assert!(ids.contains(&"pid_checkpoint".to_owned()));
        assert!(ids.contains(&"pid_gemma".to_owned()));
        assert!(!required_portable_aux_source_ids(
            &JobType::ImageGenerate,
            candle_pose.as_object().unwrap(),
            &kolors,
            "candle"
        )
        .unwrap()
        .contains(&"kolors_ip_adapter".to_owned()));
    }

    #[test]
    fn selected_ids_and_shapes_follow_character_and_detail_routes() {
        let flux = json!({"id":"flux_dev"});
        let payload =
            json!({"model":"flux_dev","mode":"character_image","referenceAssetId":"asset"});
        assert_eq!(
            required_portable_aux_source_ids(
                &JobType::ImageGenerate,
                payload.as_object().unwrap(),
                &flux,
                "mlx"
            )
            .unwrap(),
            vec!["flux_ip_adapter", "flux_ip_encoder"]
        );
        assert_eq!(
            portable_aux_source_shape("flux_ip_adapter", &flux),
            Some(PortableAuxSourceShape::File)
        );
        assert_eq!(
            portable_aux_source_shape("flux_ip_encoder", &flux),
            Some(PortableAuxSourceShape::Directory)
        );
        let sdxl = json!({"id":"sdxl"});
        let detail = json!({"model":"sdxl"});
        assert_eq!(
            required_portable_aux_source_ids(
                &JobType::ImageDetail,
                detail.as_object().unwrap(),
                &sdxl,
                "candle"
            )
            .unwrap(),
            vec!["sdxl_detail_tile_control"]
        );
    }

    #[test]
    fn cover_and_decoder_sources_are_selected_only_when_authored() {
        let ace = json!({"id":"acestep_v15_turbo"});
        let cover = json!({"editMode":"cover"});
        assert_eq!(
            required_portable_aux_source_ids(
                &JobType::AudioGenerate,
                cover.as_object().unwrap(),
                &ace,
                "mlx"
            )
            .unwrap(),
            vec!["sft_cover"]
        );
        let plain = json!({"editMode":"inpaint"});
        assert!(required_portable_aux_source_ids(
            &JobType::AudioGenerate,
            plain.as_object().unwrap(),
            &ace,
            "mlx"
        )
        .unwrap()
        .is_empty());
        let qwen = json!({"id":"qwen_image","decoders":{"byBackend":{"mlx":[{"id":"wan_2_1_vae","componentId":"vae"}]}}});
        let decode = json!({"model":"qwen_image","advanced":{"decoder":"wan_2_1_vae"}});
        assert_eq!(
            required_portable_aux_source_ids(
                &JobType::ImageGenerate,
                decode.as_object().unwrap(),
                &qwen,
                "mlx"
            )
            .unwrap(),
            vec!["vae"]
        );
        assert_eq!(
            portable_aux_source_shape("vae", &qwen),
            Some(PortableAuxSourceShape::File)
        );
    }

    #[test]
    fn moss_runtime_components_follow_the_projected_downloads_and_shapes() {
        let entry = json!({"id":"moss_image_sft","downloads":[
            {"componentId":"moss-dmd-lora","files":["lora.safetensors"]},
            {"componentId":"moss-image-edit","files":["adapter_config.json","adapter_model.safetensors"]}
        ]});
        let payload = json!({"model":"moss_image_sft"});
        assert_eq!(
            required_portable_aux_source_ids(
                &JobType::ImageGenerate,
                payload.as_object().unwrap(),
                &entry,
                "mlx"
            )
            .unwrap(),
            vec!["moss-dmd-lora", "moss-image-edit"]
        );
        assert_eq!(
            portable_aux_source_shape("moss-dmd-lora", &entry),
            Some(PortableAuxSourceShape::File)
        );
        assert_eq!(
            portable_aux_source_shape("moss-image-edit", &entry),
            Some(PortableAuxSourceShape::Directory)
        );
        assert_eq!(
            trainer_required_component_ids("sdxl", "candle"),
            &["tokenizer_clip_l", "tokenizer_clip_bigg", "vae_fp16_fix"]
        );
        assert!(trainer_required_component_ids("sdxl", "mlx").is_empty());
    }

    #[test]
    fn training_auxiliary_selection_follows_enabled_losses() {
        let plan: crate::training::TrainingPlan = serde_json::from_value(json!({
            "schemaVersion":1,"planVersion":1,"jobId":"job",
            "target":{"targetId":"target","kernel":"z_image_lora","family":"zimage","modality":"image","outputKind":"lora","baseModel":"z_image_turbo","baseModelPath":"/models/zimage"},
            "dataset":{"datasetId":"dataset","datasetVersion":1,"rootPath":"/dataset","items":[]},
            "config":{"rank":16,"alpha":32,"learningRate":0.0001,"steps":10,"batchSize":1,"gradientAccumulation":1,"resolution":512,"saveEvery":10,"seed":1,"optimizer":"adamw8bit","advanced":{"depthAnchoringWeight":0.1,"depthAnchoringModel":"small","bodyProportionWeight":0.1,"identityLossWeight":0.1}},
            "output":{"loraId":"lora","outputDir":"/output","fileName":"lora.safetensors","format":"safetensors","triggerWords":[]},
            "provenance":{"datasetId":"dataset","datasetVersion":1,"targetId":"target","baseModel":"z_image_turbo","configSnapshot":{},"outputLoraId":"lora","sourceJobId":"job","createdAt":"2026-11-07T00:00:00Z"}
        })).unwrap();
        let selected = selected_training_aux_models(&plan, "z_image_turbo").unwrap();
        let ids: std::collections::BTreeSet<_> = selected.iter().map(|model| model.id).collect();
        assert!(ids.contains("taef1"));
        assert!(ids.contains("depth_anything_v2_small"));
        assert!(ids.contains("vitpose_plus_base"));
        assert!(ids.contains("instantid_face_stack"));
        assert_eq!(ids.len(), 4);
        let base_entry = json!({"id":"z_image_turbo","downloads":[]});
        for model in selected {
            assert_eq!(
                portable_aux_source_shape(model.id, &base_entry),
                Some(PortableAuxSourceShape::Directory),
                "{} must be resolved from its own pinned snapshot, not the training base",
                model.id
            );
        }
        let mut task_plan = plan.clone();
        task_plan.target.base_model = "iris_3b_depth".to_owned();
        task_plan.target.kernel = "iris_train".to_owned();
        task_plan.config.advanced.clear();
        task_plan
            .config
            .advanced
            .insert("depthInit".to_owned(), json!("parent"));
        let payload = json!({"plan":task_plan});
        let entry = json!({"id":"iris_3b_depth"});
        assert_eq!(
            required_portable_aux_source_ids(
                &JobType::LoraTrain,
                payload.as_object().unwrap(),
                &entry,
                "mlx"
            )
            .unwrap(),
            vec!["iris_depth_parent", "iris_parent_backbone"]
        );
    }
}
