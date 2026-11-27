//! Canonical SceneWorks product-to-engine route binding tables.
//!
//! Inference's `sga.capabilities/2` build products own pure semantic capability truth keyed by
//! inference `route_id`, trainer `id`, and provider `id`. SceneWorks owns product model IDs,
//! legacy aliases, multi-engine video dispatch bindings, and training target bindings.
//!
//! Shared between `build.rs` (static `ModelSpec` assembly), `sceneworks_core::registry`,
//! `sceneworks_core::jobs_store::routing`, and `sceneworks-worker` so there is one authoritative
//! binding table across the workspace.

/// Non-identity SceneWorks image model ID -> inference engine `route_id` aliases.
///
/// Any product ID not listed here maps to itself (`product_id == route_id`) unless a backend
/// registers the product ID directly (e.g. Candle registers `qwen_image_edit_2509`,
/// `qwen_image_edit_2511`, and `qwen_image_edit_2511_lightning` directly in `sga.capabilities/2`).
pub const IMAGE_MODEL_ALIASES: &[(&str, &str)] = &[
    ("bernini_image", "bernini"),
    ("flux2_klein_9b_kv", "flux2_klein_9b"),
    ("flux2_klein_9b_true_v2", "flux2_klein_9b"),
    ("flux_dev", "flux1_dev"),
    ("flux_schnell", "flux1_schnell"),
    ("illustrious_xl_v1", "sdxl"),
    ("illustrious_xl_v2", "sdxl"),
    ("qwen_image_edit_2509", "qwen_image_edit"),
    ("qwen_image_edit_2511", "qwen_image_edit"),
    ("qwen_image_edit_2511_lightning", "qwen_image_edit"),
    ("realvisxl", "sdxl"),
    ("realvisxl_lightning", "sdxl"),
    ("sensenova_u1_8b_infographic_v2", "sensenova_u1_8b"),
    (
        "sensenova_u1_8b_infographic_v2_fast",
        "sensenova_u1_8b_fast",
    ),
    ("sensenova_u1_8b_infographic_v3", "sensenova_u1_8b"),
    (
        "sensenova_u1_8b_infographic_v3_fast",
        "sensenova_u1_8b_fast",
    ),
    ("z_image_edit", "z_image_turbo"),
];

/// Direct-dispatch identity/overlay image product IDs whose inference routes are registered under
/// their engine-canonical IDs in `sga.capabilities/2`.
pub const BESPOKE_IMAGE_ROUTE_ALIASES: &[(&str, &str)] = &[
    ("pulid_flux_dev", "pulid_flux"),
    ("instantid_realvisxl", "instantid"),
];

/// Look up the inference generator `route_id` for a SceneWorks image model ID on `backend`.
///
/// Prefers an exact registered `route_id` on that backend (so Candle's registered
/// `qwen_image_edit_2509` / `qwen_image_edit_2511` / `qwen_image_edit_2511_lightning` routes
/// resolve directly to themselves), then falls back to [`IMAGE_MODEL_ALIASES`].
pub fn image_route_id(product_id: &str, has_route: impl Fn(&str) -> bool) -> Option<&str> {
    if has_route(product_id) {
        return Some(product_id);
    }
    IMAGE_MODEL_ALIASES
        .iter()
        .find_map(|&(model, route)| (model == product_id && has_route(route)).then_some(route))
}

/// Static alias target for a SceneWorks image model ID (independent of backend lookup).
pub fn image_alias_target(product_id: &str) -> &str {
    IMAGE_MODEL_ALIASES
        .iter()
        .find_map(|&(model, route)| (model == product_id).then_some(route))
        .unwrap_or(product_id)
}

/// Look up a bespoke direct-dispatch route alias (e.g. `pulid_flux_dev` -> `pulid_flux`).
pub fn bespoke_image_route_id(product_id: &str) -> Option<&'static str> {
    BESPOKE_IMAGE_ROUTE_ALIASES
        .iter()
        .find_map(|&(model, route)| (model == product_id).then_some(route))
}

/// Ordered video modes considered when projecting `ModelSpec.video_routes`.
#[allow(dead_code)]
pub const VIDEO_MODES: &[&str] = &[
    "ads2v",
    "animate_character",
    "extend_clip",
    "first_last_frame",
    "image_to_video",
    "multi_video_to_video",
    "reference_to_video",
    "reference_video_to_video",
    "replace_person",
    "text_to_video",
    "video_bridge",
    "video_to_video",
];

/// Canonical `(backend, model_id, mode)` -> inference video `route_id`s mapping.
///
/// Mirrors the production dispatch ladder in `sceneworks-worker/src/video_jobs/mod.rs`
/// (`runtime_descriptor_engine_ids`) and the routing admission in `jobs_store::routing`.
pub fn video_route_ids(backend: &str, model_id: &str, mode: &str) -> &'static [&'static str] {
    match backend {
        "mlx" => match (model_id, mode) {
            (
                "bernini",
                "ads2v"
                | "multi_video_to_video"
                | "reference_to_video"
                | "reference_video_to_video"
                | "text_to_video"
                | "video_to_video",
            ) => &["bernini"],
            ("krea_realtime_14b", "image_to_video" | "text_to_video" | "video_to_video") => {
                &["krea_realtime_14b"]
            }
            (
                "ltx_2_3" | "ltx_2_3_eros",
                "extend_clip" | "first_last_frame" | "image_to_video" | "replace_person"
                | "text_to_video" | "video_bridge",
            ) => &["ltx_2_3"],
            (
                "ltx_2_5",
                "extend_clip" | "first_last_frame" | "image_to_video" | "replace_person"
                | "text_to_video" | "video_bridge",
            ) => &["ltx_2_5"],
            ("minimax_h3", "first_last_frame" | "image_to_video" | "text_to_video") => {
                &["minimax_h3"]
            }
            ("minimax_h3_ref", "reference_to_video") => &["minimax_h3"],
            ("scail2_14b", "animate_character" | "replace_person") => &["scail2_14b"],
            ("svd", "image_to_video") => &["svd_xt"],
            ("wan_2_2", "extend_clip" | "video_bridge") => &["wan2_2_ti2v_5b", "wan_vace"],
            ("wan_2_2", "first_last_frame" | "image_to_video" | "text_to_video") => {
                &["wan2_2_ti2v_5b"]
            }
            ("wan_2_2", "replace_person") => &["wan_vace"],
            ("wan_2_2_i2v_14b", "image_to_video") => &["wan2_2_i2v_14b"],
            ("wan_2_2_t2v_14b", "text_to_video") => &["wan2_2_t2v_14b"],
            ("wan_2_2_vace_fun_14b", "replace_person") => &["wan2_2_vace_fun_14b"],
            _ => &[],
        },
        "candle" => match (model_id, mode) {
            (
                "bernini",
                "ads2v"
                | "multi_video_to_video"
                | "reference_to_video"
                | "reference_video_to_video"
                | "text_to_video"
                | "video_to_video",
            ) => &["bernini"],
            (
                "ltx_2_3",
                "extend_clip" | "first_last_frame" | "image_to_video" | "replace_person"
                | "text_to_video" | "video_bridge",
            ) => &["ltx_2_3_distilled"],
            (
                "ltx_2_5",
                "extend_clip" | "first_last_frame" | "image_to_video" | "replace_person"
                | "text_to_video" | "video_bridge",
            ) => &["ltx_2_5_distilled"],
            ("minimax_h3", "first_last_frame" | "image_to_video" | "text_to_video") => {
                &["minimax_h3"]
            }
            ("minimax_h3_ref", "reference_to_video") => &["minimax_h3"],
            ("scail2_14b", "animate_character" | "replace_person") => &["scail2_14b"],
            ("svd", "image_to_video") => &["svd_xt"],
            ("wan_2_2", "extend_clip" | "replace_person" | "video_bridge") => &["wan_vace"],
            ("wan_2_2", "first_last_frame" | "image_to_video" | "text_to_video") => {
                &["wan2_2_ti2v_5b"]
            }
            ("wan_2_2_i2v_14b", "extend_clip" | "replace_person" | "video_bridge") => &["wan_vace"],
            ("wan_2_2_i2v_14b", "image_to_video") => &["wan2_2_i2v_14b"],
            ("wan_2_2_t2v_14b", "extend_clip" | "replace_person" | "video_bridge") => &["wan_vace"],
            ("wan_2_2_t2v_14b", "text_to_video") => &["wan2_2_t2v_14b"],
            ("wan_2_2_vace_fun_14b", "replace_person") => &["wan2_2_vace_fun_14b"],
            _ => &[],
        },
        _ => &[],
    }
}

/// Canonical `(target_id, mlx_trainer_id, candle_trainer_id)` bindings for all shipped product
/// training targets. Pure static data (zero cyclic calls into `training::builtin_training_targets`).
#[allow(dead_code)]
pub const TRAINER_TARGET_BINDINGS: &[(&str, &str, &str)] = &[
    ("anima_base_lora", "anima_base", "anima_base"),
    (
        "flux3_klein_edit_lora",
        "flux3_klein_base",
        "flux3_klein_base",
    ),
    ("flux3_klein_lora", "flux3_klein_base", "flux3_klein_base"),
    ("illustrious_xl_v1_lora", "sdxl", "sdxl"),
    ("illustrious_xl_v2_lora", "sdxl", "sdxl"),
    ("iris_3b_depth", "iris_3b_depth", "iris_3b_depth"),
    ("iris_3b_generation", "iris_3b", "iris_3b"),
    ("iris_3b_restore", "iris_3b_restore", "iris_3b_restore"),
    ("kolors_lora", "kolors", "kolors"),
    ("krea_2_control", "krea_2_control", "krea_2_control"),
    ("krea_2_raw_lora", "krea_2_raw", "krea_2_raw"),
    ("lens_turbo_lora", "lens", "lens"),
    ("ltx_2_5_video_lora", "ltx_2_5", "ltx_2_5_distilled"),
    ("ltx_video_lora", "ltx_2_3", "ltx_2_3"),
    ("mage_flow_base_lora", "mage_flow_base", "mage_flow_base"),
    (
        "minimax_h3_image_edit_lora",
        "minimax_h3_image",
        "minimax_h3_image",
    ),
    (
        "minimax_h3_image_lora",
        "minimax_h3_image",
        "minimax_h3_image",
    ),
    ("moss_image_base_lora", "moss_image_base", "moss_image_base"),
    ("moss_image_sft_lora", "moss_image_sft", "moss_image_sft"),
    (
        "qwen_image_2_1_edit_lora",
        "qwen_image_2_1",
        "qwen_image_2_1",
    ),
    ("qwen_image_2_1_lora", "qwen_image_2_1", "qwen_image_2_1"),
    ("sd3_5_large_lora", "sd3_5_large", "sd3_5_large"),
    ("sd3_5_medium_lora", "sd3_5_medium", "sd3_5_medium"),
    ("sdxl_lora", "sdxl", "sdxl"),
    ("wan_i2v_14b_lora", "wan2_2_i2v_14b", "wan2_2_i2v_14b"),
    ("wan_lora", "wan2_2_ti2v_5b", "wan2_2_ti2v_5b"),
    ("wan_t2v_14b_lora", "wan2_2_t2v_14b", "wan2_2_t2v_14b"),
    ("z_image_turbo_lora", "z_image_turbo", "z_image_turbo"),
];

/// Canonical `(backend, kernel, base_model)` -> inference trainer `id` mapping.
///
/// Mirrors `sceneworks-worker/src/training_jobs.rs::engine_trainer_id_for`.
#[allow(dead_code)]
pub fn trainer_engine_id(backend: &str, kernel: &str, base_model: &str) -> Option<&'static str> {
    match kernel {
        "moss_image_lora" => match base_model {
            "moss_image_base" => Some("moss_image_base"),
            "moss_image_sft" => Some("moss_image_sft"),
            _ => None,
        },
        "z_image_lora" => Some("z_image_turbo"),
        "sdxl_lora" => Some("sdxl"),
        "kolors_lora" => (base_model == "kolors").then_some("kolors"),
        "lens_lora" => Some("lens"),
        "krea_lora" => Some("krea_2_raw"),
        "krea_control" => Some("krea_2_control"),
        "sd3_lora" => match base_model {
            "sd3_5_large" => Some("sd3_5_large"),
            "sd3_5_medium" => Some("sd3_5_medium"),
            _ => None,
        },
        "ltx_mlx_lora" => match base_model {
            "ltx_2_3" | "ltx_2_3_eros" => Some("ltx_2_3"),
            "ltx_2_5" => match backend {
                "mlx" => Some("ltx_2_5"),
                "candle" => Some("ltx_2_5_distilled"),
                _ => None,
            },
            _ => None,
        },
        "wan_lora" => (base_model == "wan_2_2").then_some("wan2_2_ti2v_5b"),
        "wan_moe_lora" => match base_model {
            "wan_2_2_t2v_14b" => Some("wan2_2_t2v_14b"),
            "wan_2_2_i2v_14b" => Some("wan2_2_i2v_14b"),
            _ => None,
        },
        "anima_lora" => (base_model == "anima_base").then_some("anima_base"),
        "mage_flow_lora" => (base_model == "mage_flow_base").then_some("mage_flow_base"),
        "iris_train" => match base_model {
            "iris_3b" => Some("iris_3b"),
            "iris_3b_depth" => Some("iris_3b_depth"),
            "iris_3b_restore" => Some("iris_3b_restore"),
            _ => None,
        },
        "flux3_klein_lora" | "flux3_klein_edit_lora" => {
            (base_model == "flux3_klein_base").then_some("flux3_klein_base")
        }
        "qwen_image_2_1_lora" | "qwen_image_2_1_edit_lora" => {
            (base_model == "qwen_image_2_1").then_some("qwen_image_2_1")
        }
        "minimax_h3_image_lora" | "minimax_h3_image_edit_lora" => {
            (base_model == "minimax_h3_image").then_some("minimax_h3_image")
        }
        _ => None,
    }
}

/// Platform worker capabilities advertised by the shipping MLX and Candle GPU workers.
///
/// Combines the shared worker job capabilities with backend-specific precision/dispatch markers
/// derived from `sga.capabilities/2` (`advanced_quant_tiers` + Candle precision capabilities).
#[allow(dead_code)]
pub fn platform_worker_capabilities(backend: &str) -> &'static [&'static str] {
    match backend {
        "mlx" => &[
            "audio_generate",
            "catalog_analysis",
            "dataset_analysis",
            "dataset_face_analysis",
            "dataset_upscale",
            "face_likeness_compare",
            "gpu",
            "image_depth",
            "image_detail",
            "image_edit",
            "image_generate",
            "image_interleave",
            "image_segment",
            "image_upscale",
            "image_vqa",
            "kps_extract",
            "lora_train",
            "lora_train_execute",
            "person_detect",
            "person_replace",
            "person_track",
            "pose_detect",
            "prompt_refine",
            "training_caption",
            "vector_image_to_svg",
            "video_bridge",
            "video_extend",
            "video_generate",
            "video_upscale",
        ],
        "candle" => &[
            "audio_generate",
            "candle",
            "catalog_analysis",
            "dataset_analysis",
            "dataset_face_analysis",
            "dataset_upscale",
            "face_likeness_compare",
            "flux3_nvfp4",
            "fp8r",
            "gpu",
            "image_depth",
            "image_detail",
            "image_edit",
            "image_generate",
            "image_interleave",
            "image_segment",
            "image_upscale",
            "image_vqa",
            "int8_convrot",
            "kps_extract",
            "lora_train",
            "lora_train_execute",
            "nvfp4",
            "person_detect",
            "person_replace",
            "person_track",
            "pose_detect",
            "prompt_refine",
            "training_caption",
            "vector_image_to_svg",
            "video_bridge",
            "video_extend",
            "video_generate",
            "video_upscale",
        ],
        _ => &[],
    }
}
