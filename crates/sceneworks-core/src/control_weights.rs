//! Trusted remote artifacts for strict-control inference.
//!
//! A job payload is not an authority to choose an arbitrary Hugging Face
//! repository. Shipped strict-control artifacts are admitted by the exact
//! engine/repository/file tuple below and are always resolved at the pinned
//! revision. Install flows may fetch those immutable bytes; render paths are
//! cache-only. User and project overlays use the separate registered-overlay
//! catalog path in the Rust API.

/// One immutable strict-control weight artifact shipped by SceneWorks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShippedControlWeight {
    pub engine_id: &'static str,
    pub repo: &'static str,
    pub file: &'static str,
    pub revision: &'static str,
}

/// Pinned defaults for worker-opened auxiliary sources outside a model's hard closure.
/// The worker may select an operator override only for an ID whose loader already permits it;
/// a private source claim freezes its exact bytes and provenance before admission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShippedAuxiliarySource {
    pub source_id: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub files: &'static [&'static str],
}

pub const SHIPPED_AUXILIARY_SOURCES: &[ShippedAuxiliarySource] = &[
    ShippedAuxiliarySource {
        source_id: "flux_ip_adapter",
        repo: "XLabs-AI/flux-ip-adapter",
        revision: "18f6940238ab5dc3744df7a8e30315892279d5f9",
        files: &["ip_adapter.safetensors"],
    },
    ShippedAuxiliarySource {
        source_id: "flux_ip_encoder",
        repo: "openai/clip-vit-large-patch14",
        revision: "32bd64288804d66eefd0ccbe215aa642df71cc41",
        files: &["model.safetensors"],
    },
    ShippedAuxiliarySource {
        source_id: "sdxl_ip_adapter",
        repo: "h94/IP-Adapter",
        revision: "018e402774aeeddd60609b4ecdb7e298259dc729",
        files: &["sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"],
    },
    ShippedAuxiliarySource {
        source_id: "sdxl_ip_encoder",
        repo: "h94/IP-Adapter",
        revision: "018e402774aeeddd60609b4ecdb7e298259dc729",
        files: &[
            "models/image_encoder/config.json",
            "models/image_encoder/model.safetensors",
        ],
    },
    ShippedAuxiliarySource {
        source_id: "sdxl_detail_tile_control",
        repo: "xinsir/controlnet-tile-sdxl-1.0",
        revision: "1ae8d9529efe58f7362a987363ff86a7904dc84f",
        files: &["config.json", "diffusion_pytorch_model.safetensors"],
    },
    ShippedAuxiliarySource {
        source_id: "depth_anything_v2_small",
        repo: "depth-anything/Depth-Anything-V2-Small-hf",
        revision: "5426e4f0f36572d16453bbda7a8389317b1bef99",
        files: &["model.safetensors"],
    },
    ShippedAuxiliarySource {
        source_id: "ltx_training_text_encoder",
        repo: "SceneWorks/ltx-2.3-mlx",
        revision: "01df27d308466533aa09d251e3aebdcc627d07eb",
        files: &["gemma/*"],
    },
];

pub fn shipped_auxiliary_source(source_id: &str) -> Option<&'static ShippedAuxiliarySource> {
    SHIPPED_AUXILIARY_SOURCES
        .iter()
        .find(|source| source.source_id == source_id)
}

/// Exact shipped control-weight allow-list.
///
/// Z-Image base has distinct MLX and Candle filenames in the upstream repo, and
/// Qwen has one artifact per packed quantization tier. They are separate rows so
/// matching never broadens to a repository- or organization-wide trust rule.
pub const SHIPPED_CONTROL_WEIGHTS: &[ShippedControlWeight] = &[
    ShippedControlWeight {
        engine_id: "sdxl",
        repo: "xinsir/controlnet-openpose-sdxl-1.0",
        file: "diffusion_pytorch_model.safetensors",
        revision: "23f966cd5cfdd3f7729c903e243d87152162d2b7",
    },
    ShippedControlWeight {
        engine_id: "flux1_dev_control",
        repo: "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
        file: "diffusion_pytorch_model.safetensors",
        revision: "5d700aaad96c5ddcdf8a38ef9b22a82aac2c38e5",
    },
    ShippedControlWeight {
        engine_id: "flux2_dev_control",
        repo: "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union",
        file: "FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors",
        revision: "b3dcd7836a0e926248dac3ccba8fc0853495764b",
    },
    ShippedControlWeight {
        engine_id: "z_image_turbo_control",
        repo: "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
        file: "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors",
        revision: "5155fc56d17821007d6f62ac192c09e0f0e72016",
    },
    ShippedControlWeight {
        engine_id: "z_image_control",
        repo: "alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1",
        file: "Z-Image-Fun-Controlnet-Union-2.1.safetensors",
        revision: "755999a934909bd5832e20718bb7c639d2a63eb9",
    },
    ShippedControlWeight {
        engine_id: "z_image_control",
        repo: "alibaba-pai/Z-Image-Fun-Controlnet-Union-2.1",
        file: "diffusion_pytorch_model.safetensors",
        revision: "755999a934909bd5832e20718bb7c639d2a63eb9",
    },
    ShippedControlWeight {
        engine_id: "qwen_image_control",
        repo: "SceneWorks/qwen-image-2512-fun-controlnet-union",
        file: "q4/model.safetensors",
        revision: "a061fbc42a4744d6a7ec206370fbd3a37d4a7cca",
    },
    ShippedControlWeight {
        engine_id: "qwen_image_control",
        repo: "SceneWorks/qwen-image-2512-fun-controlnet-union",
        file: "q8/model.safetensors",
        revision: "a061fbc42a4744d6a7ec206370fbd3a37d4a7cca",
    },
    ShippedControlWeight {
        engine_id: "qwen_image_control",
        repo: "SceneWorks/qwen-image-2512-fun-controlnet-union",
        file: "bf16/model.safetensors",
        revision: "a061fbc42a4744d6a7ec206370fbd3a37d4a7cca",
    },
    ShippedControlWeight {
        engine_id: "kolors_control",
        repo: "Kwai-Kolors/Kolors-ControlNet-Pose",
        file: "diffusion_pytorch_model.safetensors",
        revision: "83e35a8033a89d2e75044b412d0e2474111578f7",
    },
    ShippedControlWeight {
        engine_id: "krea_2_turbo_control",
        repo: "SceneWorks/krea2-pose-controlnet-beta",
        file: "control_step5000.safetensors",
        revision: "cb3a0ac7590f5ec594a4eeb43b95ee1da0b5a0ac",
    },
];

/// Find the exact shipped artifact for an engine/repository/file tuple.
pub fn shipped_control_weight(
    engine_id: &str,
    repo: &str,
    file: &str,
) -> Option<&'static ShippedControlWeight> {
    SHIPPED_CONTROL_WEIGHTS.iter().find(|artifact| {
        artifact.engine_id == engine_id && artifact.repo == repo && artifact.file == file
    })
}

/// Find a shipped artifact without choosing an engine.
///
/// The API uses this only to reject arbitrary raw payload repositories before a
/// job is queued. The worker performs the stronger engine-specific check.
pub fn shipped_control_weight_by_repo_file(
    repo: &str,
    file: &str,
) -> Option<&'static ShippedControlWeight> {
    SHIPPED_CONTROL_WEIGHTS
        .iter()
        .find(|artifact| artifact.repo == repo && artifact.file == file)
}

/// Default repository for an engine (all variants for one engine share a repo).
pub fn default_control_repo(engine_id: &str) -> Option<&'static str> {
    SHIPPED_CONTROL_WEIGHTS
        .iter()
        .find(|artifact| artifact.engine_id == engine_id)
        .map(|artifact| artifact.repo)
}

/// Default immutable revision for an engine.
pub fn default_control_revision(engine_id: &str) -> Option<&'static str> {
    SHIPPED_CONTROL_WEIGHTS
        .iter()
        .find(|artifact| artifact.engine_id == engine_id)
        .map(|artifact| artifact.revision)
}

/// The strict-control route's shipped component for this exact request. The worker and API
/// consult the same table; an authored registered overlay replaces this default separately.
/// A control file is selected only when the request actually enters a pose/control route.
pub fn default_control_weight_for_request(
    model_id: &str,
    entry: &serde_json::Value,
    backend: &str,
    payload: &serde_json::Map<String, serde_json::Value>,
) -> Option<&'static ShippedControlWeight> {
    let poses = payload
        .get("advanced")?
        .get("poses")?
        .as_array()
        .is_some_and(|poses| !poses.is_empty());
    if !poses || payload.get("mode").and_then(serde_json::Value::as_str) == Some("edit_image") {
        return None;
    }
    let base_model = entry
        .get("baseModel")
        .and_then(serde_json::Value::as_str)
        .unwrap_or(model_id);
    let engine = match base_model {
        "sdxl"
        | "realvisxl"
        | "realvisxl_lightning"
        | "illustrious_xl_v1"
        | "illustrious_xl_v2" => "sdxl",
        "flux_dev" => "flux1_dev_control",
        "flux2_dev" => "flux2_dev_control",
        "z_image_turbo" => "z_image_turbo_control",
        "z_image" => "z_image_control",
        "qwen_image" => "qwen_image_control",
        "kolors" => "kolors_control",
        "krea_2_turbo" => "krea_2_turbo_control",
        _ => return None,
    };
    let file = match engine {
        "z_image_control" if backend == "candle" => "diffusion_pytorch_model.safetensors",
        "z_image_control" => "Z-Image-Fun-Controlnet-Union-2.1.safetensors",
        "qwen_image_control" => match selected_qwen_control_tier(payload) {
            "q4" => "q4/model.safetensors",
            "bf16" => "bf16/model.safetensors",
            _ => "q8/model.safetensors",
        },
        _ => {
            SHIPPED_CONTROL_WEIGHTS
                .iter()
                .find(|weight| weight.engine_id == engine)?
                .file
        }
    };
    SHIPPED_CONTROL_WEIGHTS
        .iter()
        .find(|weight| weight.engine_id == engine && weight.file == file)
}

/// The portable control tier is independent of this host's installed snapshot. The legacy
/// worker may clamp an implicit Q8 pick to a present tier, but a bound job must keep Q8 across
/// hosts and fetch or refuse when it is missing.
pub fn selected_qwen_control_tier(
    payload: &serde_json::Map<String, serde_json::Value>,
) -> &'static str {
    let bits = payload
        .get("advanced")
        .and_then(serde_json::Value::as_object)
        .and_then(|advanced| crate::payload_util::optional_i64(advanced, "mlxQuantize"));
    match bits {
        Some(bits) if bits <= 0 => "bf16",
        Some(bits) if bits <= 4 => "q4",
        _ => "q8",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn builtin_models() -> Vec<Value> {
        let manifest = crate::builtin_manifests::builtin_manifest("builtin.models.jsonc")
            .expect("valid embedded model catalog");
        manifest["models"]
            .as_array()
            .expect("models array")
            .to_vec()
    }

    #[test]
    fn allow_list_is_exact_and_revisions_are_pinned() {
        assert!(shipped_control_weight(
            "sdxl",
            "xinsir/controlnet-openpose-sdxl-1.0",
            "diffusion_pytorch_model.safetensors"
        )
        .is_some());
        assert!(shipped_control_weight(
            "sdxl",
            "xinsir/controlnet-openpose-sdxl-1.0",
            "other.safetensors"
        )
        .is_none());
        assert!(shipped_control_weight(
            "flux2_dev_control",
            "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union",
            "FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors"
        )
        .is_some());
        assert!(shipped_control_weight(
            "flux2_dev_control",
            "alibaba-pai/other",
            "FLUX.2-dev-Fun-Controlnet-Union-2602.safetensors"
        )
        .is_none());
        assert!(shipped_control_weight(
            "flux2_dev_control",
            "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union",
            "other.safetensors"
        )
        .is_none());
        assert!(SHIPPED_CONTROL_WEIGHTS.iter().all(|artifact| {
            artifact.revision.len() == 40
                && artifact
                    .revision
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        }));
    }

    #[test]
    fn sdxl_openpose_is_one_shared_soft_component_on_the_exact_five_backbones() {
        const SDXL_BACKBONES: [&str; 5] = [
            "sdxl",
            "realvisxl",
            "realvisxl_lightning",
            "illustrious_xl_v1",
            "illustrious_xl_v2",
        ];
        let artifact = shipped_control_weight(
            "sdxl",
            "xinsir/controlnet-openpose-sdxl-1.0",
            "diffusion_pytorch_model.safetensors",
        )
        .expect("shipped SDXL control authority");
        let models = builtin_models();
        assert!(
            models
                .iter()
                .all(|model| model["id"] != "controlnet_openpose_sdxl"),
            "OpenPose is a component of the five accepted SDXL capability rows, not a fake utility model"
        );

        for model_id in SDXL_BACKBONES {
            let model = models
                .iter()
                .find(|model| model["id"] == model_id)
                .unwrap_or_else(|| panic!("builtin model {model_id} exists"));
            let components = model["downloads"]
                .as_array()
                .expect("downloads array")
                .iter()
                .filter(|download| download["componentId"] == "controlnet_openpose")
                .collect::<Vec<_>>();
            assert_eq!(
                components.len(),
                1,
                "{model_id} declares the shared control component exactly once"
            );
            let component = components[0];
            assert_eq!(component["coRequisite"], true, "{model_id}");
            assert_eq!(component["required"], "soft", "{model_id}");
            assert_eq!(component["repo"], artifact.repo, "{model_id}");
            assert_eq!(component["revision"], artifact.revision, "{model_id}");
            assert_eq!(component["files"], json!([artifact.file]), "{model_id}");
            assert_eq!(
                component["platforms"],
                json!(["macos", "windows", "linux"]),
                "{model_id}: the shared route exists on both MLX and Candle bundles"
            );
        }

        let declaring_models = models
            .iter()
            .filter(|model| {
                model["downloads"].as_array().is_some_and(|downloads| {
                    downloads
                        .iter()
                        .any(|download| download["componentId"] == "controlnet_openpose")
                })
            })
            .map(|model| model["id"].as_str().expect("model id"))
            .collect::<Vec<_>>();
        assert_eq!(
            declaring_models.len(),
            SDXL_BACKBONES.len(),
            "no utility or sixth backbone may acquire the component"
        );
    }
}
