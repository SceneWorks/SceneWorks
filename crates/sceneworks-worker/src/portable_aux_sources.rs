//! Read-only source selection for private portable worker source claims.
//! Each branch mirrors the corresponding loader's env, HF snapshot, and app-cache order.

use std::path::{Path, PathBuf};

use sceneworks_core::contracts::JsonObject;
use serde_json::Value;

/// Only the verified dispatch clone carries this private map. The worker guard removes every
/// client-supplied copy before adding immutable leased paths for the selected route.
pub(crate) fn bound_aux_path(advanced: &JsonObject, source_id: &str) -> Option<PathBuf> {
    advanced
        .get("__portableAuxPaths")?
        .get(source_id)?
        .as_str()
        .map(PathBuf::from)
}

pub(crate) fn has_bound_aux_paths(advanced: &JsonObject) -> bool {
    advanced.get("__portableAuxPaths").is_some()
}

/// Effective path plus whether the loader selected an authorized operator override.
/// The caller hashes the exact file/dir closure and records pinned-repo provenance otherwise.
pub(crate) fn resolve_selected_aux_source(
    source_id: &str,
    data_dir: &Path,
    payload: &JsonObject,
) -> Option<(PathBuf, bool)> {
    let snapshot = |repo| crate::model_jobs::huggingface_snapshot_dir(data_dir, repo);
    match source_id {
        #[cfg(native_lane)]
        "ltx_training_gemma_override" => {
            crate::video_jobs::ltx::ltx_gemma_env_override().map(|path| (path, true))
        }
        #[cfg(native_lane)]
        "ltx_training_text_encoder" => {
            let model_dir = payload
                .get("plan")?
                .get("target")?
                .get("baseModelPath")?
                .as_str()?;
            let path = crate::video_jobs::ltx::resolve_bundled_ltx_gemma_dir(Path::new(model_dir))?;
            let overridden =
                std::env::var_os("LTX_GEMMA_DIR").is_some_and(|value| path == PathBuf::from(value));
            Some((path, overridden))
        }
        "ltx_amoral_gemma_3_12b" => {
            #[cfg(not(target_os = "macos"))]
            {
                return None;
            }
            #[cfg(target_os = "macos")]
            {
                let path = crate::video_jobs::ltx::resolve_ltx_uncensored_enhancer_dir(data_dir)?;
                let overridden = std::env::var_os("LTX_UNCENSORED_GEMMA_DIR")
                    .is_some_and(|value| path == PathBuf::from(value));
                Some((path, overridden))
            }
        }
        "flux_ip_adapter" | "flux_ip_encoder" => {
            let env = std::env::var_os("SCENEWORKS_IPADAPTER_FLUX").map(PathBuf::from);
            if let Some(root) = env.filter(|root| {
                root.join("ip_adapter.safetensors").is_file()
                    && root.join("image_encoder/model.safetensors").is_file()
            }) {
                return Some((
                    if source_id == "flux_ip_adapter" {
                        root.join("ip_adapter.safetensors")
                    } else {
                        root.join("image_encoder")
                    },
                    true,
                ));
            }
            if source_id == "flux_ip_adapter" {
                if let Some(cached) = snapshot("XLabs-AI/flux-ip-adapter")
                    .map(|root| root.join("ip_adapter.safetensors"))
                {
                    if cached.is_file() {
                        return Some((cached, false));
                    }
                }
                let local = data_dir.join("cache/ipadapter-flux/ip_adapter.safetensors");
                return local.is_file().then_some((local, false));
            }
            if let Some(root) = snapshot("openai/clip-vit-large-patch14") {
                if root.join("model.safetensors").is_file() {
                    return Some((root, false));
                }
            }
            let local = data_dir.join("cache/ipadapter-flux/image_encoder");
            local
                .join("model.safetensors")
                .is_file()
                .then_some((local, false))
        }
        "sdxl_ip_adapter" | "sdxl_ip_encoder" => {
            let env = std::env::var_os("SCENEWORKS_IPADAPTER_SDXL").map(PathBuf::from);
            if let Some(root) = env.filter(|root| {
                root.join("sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors")
                    .is_file()
                    && root.join("models/image_encoder").is_dir()
            }) {
                return Some((
                    if source_id == "sdxl_ip_adapter" {
                        root.join("sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors")
                    } else {
                        root.join("models/image_encoder")
                    },
                    true,
                ));
            }
            if let Some(root) = snapshot("h94/IP-Adapter") {
                let bundle = root.join("sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors");
                let encoder = root.join("models/image_encoder");
                if bundle.is_file() && encoder.join("model.safetensors").is_file() {
                    return Some((
                        if source_id == "sdxl_ip_adapter" {
                            bundle
                        } else {
                            encoder
                        },
                        false,
                    ));
                }
            }
            let cache = data_dir.join("cache/ipadapter-sdxl");
            if source_id == "sdxl_ip_adapter" {
                let file = cache.join("ip-adapter-plus_sdxl_vit-h.safetensors");
                file.is_file().then_some((file, false))
            } else {
                let dir = cache.join("image_encoder");
                dir.join("model.safetensors")
                    .is_file()
                    .then_some((dir, false))
            }
        }
        "sdxl_detail_tile_control" => {
            let repo = payload
                .get("advanced")
                .and_then(Value::as_object)
                .and_then(|advanced| advanced.get("tileControlNetRepo"))
                .and_then(Value::as_str)
                .filter(|repo| !repo.trim().is_empty())
                .unwrap_or("xinsir/controlnet-tile-sdxl-1.0");
            let root = snapshot(repo)?;
            let file = crate::first_safetensors_path(&root)?;
            Some((file, repo != "xinsir/controlnet-tile-sdxl-1.0"))
        }
        "depth_anything_v2_small" => {
            if let Some(dir) = std::env::var_os("SCENEWORKS_DEPTH_ANYTHING_V2")
                .map(PathBuf::from)
                .filter(|dir| dir.join("model.safetensors").is_file())
            {
                return Some((dir, true));
            }
            if let Some(dir) = snapshot("depth-anything/Depth-Anything-V2-Small-hf") {
                if dir.join("model.safetensors").is_file() {
                    return Some((dir, false));
                }
            }
            let dir = data_dir.join("cache/depth-anything-v2");
            dir.join("model.safetensors")
                .is_file()
                .then_some((dir, false))
        }
        _ => None,
    }
}
