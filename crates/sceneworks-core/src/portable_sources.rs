//! Pure selected auxiliary source IDs shared by API admission and worker guard.

use crate::contracts::{JobType, JsonObject};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PortableAuxSourceShape {
    File,
    Directory,
}

pub fn portable_aux_source_shape(source_id: &str, entry: &Value) -> Option<PortableAuxSourceShape> {
    use PortableAuxSourceShape::{Directory, File};
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
        | "veda_predictor" => Some(File),
        "depth_anything_v2_small"
        | "flux_ip_encoder"
        | "sdxl_ip_encoder"
        | "prompt_upsampler"
        | "sft_cover"
        | "repa_teacher"
        | "ltx_amoral_gemma_3_12b"
        | "ltx_training_text_encoder" => Some(Directory),
        "instantid_mlx"
        | "instantid_controlnet"
        | "instantid_openpose"
        | "pid_gemma"
        | "kolors_ip_adapter"
        | "utility_face_stack" => Some(Directory),
        "pid_checkpoint" => Some(Directory),
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
        if id == "kolors" && has_ref {
            ids.push("kolors_ip_adapter".to_owned());
        }
        if pid_backbone_for(id).is_some()
            && id != "kolors"
            && (id != "pulid_flux_dev" || backend == "candle")
            && advanced
                .and_then(|value| value.get("usePid"))
                .and_then(Value::as_bool)
                == Some(true)
        {
            ids.extend(["pid_checkpoint", "pid_gemma"].map(str::to_owned));
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
    use serde_json::json;

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
}
