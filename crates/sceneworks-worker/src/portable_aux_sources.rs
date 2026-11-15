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

/// A verified portable attempt must use its leased strict-control copy even when an operator
/// env override is present. Legacy, unbound jobs retain the operator override behavior.
#[cfg(any(test, native_lane))]
pub(crate) fn selected_krea_control_override(
    advanced: &JsonObject,
    env_override: Option<&str>,
) -> crate::WorkerResult<Option<PathBuf>> {
    if has_bound_aux_paths(advanced) {
        return bound_aux_path(advanced, "strict_control")
            .map(Some)
            .ok_or_else(|| {
                crate::WorkerError::InvalidPayload("missing bound strict_control source".to_owned())
            });
    }
    Ok(env_override
        .map(str::trim)
        .map(PathBuf::from)
        .filter(|path| path.is_file()))
}

/// The native PuLID providers take a single face directory although the three selected
/// face files belong to two independently authenticated repositories. Compose only from
/// worker-leased immutable paths and keep this temporary directory alive through generation.
#[cfg(native_lane)]
pub(crate) fn bound_pulid_weights(
    advanced: &JsonObject,
) -> crate::WorkerResult<(PathBuf, PathBuf, PathBuf, tempfile::TempDir)> {
    let required = |id| {
        bound_aux_path(advanced, id)
            .ok_or_else(|| crate::WorkerError::InvalidPayload(format!("missing bound {id} source")))
    };
    let adapter = required("pulid_adapter")?;
    let eva = required("pulid_eva")?;
    let bisenet = required("pulid_bisenet")?;
    let scrfd = required("pulid_scrfd")?;
    let arcface = required("pulid_arcface")?;
    let staged = tempfile::tempdir()?;
    for (source, file) in [
        (&bisenet, "bisenet_parsing.safetensors"),
        (&scrfd, "scrfd_10g.safetensors"),
        (&arcface, "arcface_iresnet100.safetensors"),
    ] {
        std::fs::copy(source, staged.path().join(file))?;
    }
    Ok((adapter, eva, staged.path().to_path_buf(), staged))
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
        "instantid_mlx" => {
            let files = [
                "ip-adapter.safetensors",
                "scrfd_10g.safetensors",
                "arcface_iresnet100.safetensors",
            ];
            if let Some(dir) = std::env::var_os("SCENEWORKS_INSTANTID_WEIGHTS")
                .map(PathBuf::from)
                .filter(|dir| files.iter().all(|file| dir.join(file).is_file()))
            {
                return Some((dir, true));
            }
            let dir = data_dir.join("cache/instantid-mlx");
            if files.iter().all(|file| dir.join(file).is_file()) {
                return Some((dir, false));
            }
            snapshot("SceneWorks/instantid-mlx")
                .filter(|dir| files.iter().all(|file| dir.join(file).is_file()))
                .map(|dir| (dir, false))
        }
        "instantid_controlnet" => {
            if let Some(dir) = std::env::var_os("SCENEWORKS_INSTANTID_CONTROLNET")
                .map(PathBuf::from)
                .filter(|dir| {
                    dir.join("config.json").is_file()
                        && dir.join("diffusion_pytorch_model.safetensors").is_file()
                })
            {
                return Some((dir, true));
            }
            if let Some(dir) = snapshot("InstantX/InstantID")
                .map(|dir| dir.join("ControlNetModel"))
                .filter(|dir| {
                    dir.join("config.json").is_file()
                        && dir.join("diffusion_pytorch_model.safetensors").is_file()
                })
            {
                return Some((dir, false));
            }
            let dir = data_dir.join("cache/instantid-controlnet");
            (dir.join("config.json").is_file()
                && dir.join("diffusion_pytorch_model.safetensors").is_file())
            .then_some((dir, false))
        }
        "instantid_openpose" => {
            if let Some(dir) = std::env::var_os("SCENEWORKS_INSTANTID_OPENPOSE")
                .map(PathBuf::from)
                .filter(|dir| {
                    dir.join("config.json").is_file()
                        && dir.join("diffusion_pytorch_model.safetensors").is_file()
                })
            {
                return Some((dir, true));
            }
            if let Some(dir) = snapshot("xinsir/controlnet-openpose-sdxl-1.0").filter(|dir| {
                dir.join("config.json").is_file()
                    && dir.join("diffusion_pytorch_model.safetensors").is_file()
            }) {
                return Some((dir, false));
            }
            let dir = data_dir.join("cache/instantid-openpose");
            (dir.join("config.json").is_file()
                && dir.join("diffusion_pytorch_model.safetensors").is_file())
            .then_some((dir, false))
        }
        "pulid_adapter" | "pulid_eva" | "pulid_bisenet" | "pulid_scrfd" | "pulid_arcface" => {
            let (repo, file) = match source_id {
                "pulid_adapter" => ("guozinan/PuLID", "pulid_flux_v0.9.1.safetensors"),
                "pulid_eva" => ("SceneWorks/pulid-flux-mlx", "eva02_clip_l_336.safetensors"),
                "pulid_bisenet" => ("SceneWorks/pulid-flux-mlx", "bisenet_parsing.safetensors"),
                "pulid_scrfd" => ("SceneWorks/instantid-mlx", "scrfd_10g.safetensors"),
                _ => ("SceneWorks/instantid-mlx", "arcface_iresnet100.safetensors"),
            };
            // The three-variable MLX preset is valid only as a complete group.
            let preset = std::env::var_os("PULID_FLUX_WEIGHTS")
                .map(PathBuf::from)
                .zip(std::env::var_os("PULID_EVA_WEIGHTS").map(PathBuf::from))
                .zip(std::env::var_os("PULID_FACE_WEIGHTS_DIR").map(PathBuf::from))
                .filter(|((adapter, eva), face)| {
                    adapter.is_file() && eva.is_file() && face.is_dir()
                });
            if let Some(((adapter, eva), face)) = preset {
                let selected = match source_id {
                    "pulid_adapter" => adapter,
                    "pulid_eva" => eva,
                    _ => face.join(file),
                };
                if selected.is_file() {
                    return Some((selected, true));
                }
            }
            if let Some(dir) = std::env::var_os("SCENEWORKS_PULID_WEIGHTS")
                .map(PathBuf::from)
                .filter(|dir| {
                    dir.join(file).is_file()
                        && (cfg!(target_os = "macos")
                            || [
                                "pulid_flux_v0.9.1.safetensors",
                                "eva02_clip_l_336.safetensors",
                                "bisenet_parsing.safetensors",
                                "scrfd_10g.safetensors",
                                "arcface_iresnet100.safetensors",
                            ]
                            .iter()
                            .all(|file| dir.join(file).is_file()))
                })
            {
                return Some((dir.join(file), true));
            }
            let dir = data_dir.join(if cfg!(target_os = "macos") {
                "cache/pulid-flux-mlx"
            } else {
                "cache/pulid-flux"
            });
            let local = dir.join(file);
            let cached = snapshot(repo)
                .map(|root| root.join(file))
                .filter(|path| path.is_file());
            if cfg!(target_os = "macos")
                || matches!(source_id, "pulid_bisenet" | "pulid_scrfd" | "pulid_arcface")
            {
                if local.is_file() {
                    return Some((local, false));
                }
                cached.map(|path| (path, false))
            } else {
                cached
                    .map(|path| (path, false))
                    .or_else(|| local.is_file().then_some((local, false)))
            }
        }
        #[cfg(native_lane)]
        "iris_depth_export" => crate::iris_task_training::iris_depth_parent_export_dir(data_dir)
            .map(|path| (path, false)),
        #[cfg(native_lane)]
        "iris_generation_text_encoder" => {
            crate::iris_task_training::installed_generation_text_encoder(data_dir)
                .map(|path| (path, false))
        }
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
            let overridden = std::env::var_os("LTX_GEMMA_DIR")
                .is_some_and(|value| path.as_os_str() == value.as_os_str());
            Some((path, overridden))
        }
        "ltx_amoral_gemma_3_12b" => {
            #[cfg(not(target_os = "macos"))]
            {
                None
            }
            #[cfg(target_os = "macos")]
            {
                let path = crate::video_jobs::ltx::resolve_ltx_uncensored_enhancer_dir(data_dir)?;
                let overridden = std::env::var_os("LTX_UNCENSORED_GEMMA_DIR")
                    .is_some_and(|value| path.as_os_str() == value.as_os_str());
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

#[cfg(all(test, native_lane))]
mod identity_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn pulid_composition_reads_only_bound_files_and_requires_every_source() {
        let source = tempfile::tempdir().unwrap();
        let mut map = serde_json::Map::new();
        for (id, name) in [
            ("pulid_adapter", "adapter"),
            ("pulid_eva", "eva"),
            ("pulid_bisenet", "bisenet"),
            ("pulid_scrfd", "scrfd"),
            ("pulid_arcface", "arcface"),
        ] {
            let file = source.path().join(name);
            std::fs::write(&file, name).unwrap();
            map.insert(id.to_owned(), json!(file));
        }
        let mut advanced = JsonObject::new();
        advanced.insert("__portableAuxPaths".to_owned(), Value::Object(map.clone()));
        let (adapter, eva, face, _lease) = bound_pulid_weights(&advanced).unwrap();
        assert_eq!(std::fs::read_to_string(adapter).unwrap(), "adapter");
        assert_eq!(std::fs::read_to_string(eva).unwrap(), "eva");
        for (file, expected) in [
            ("bisenet_parsing.safetensors", "bisenet"),
            ("scrfd_10g.safetensors", "scrfd"),
            ("arcface_iresnet100.safetensors", "arcface"),
        ] {
            assert_eq!(std::fs::read_to_string(face.join(file)).unwrap(), expected);
        }
        map.remove("pulid_arcface");
        advanced.insert("__portableAuxPaths".to_owned(), Value::Object(map));
        assert!(bound_pulid_weights(&advanced).is_err());
    }
}

#[cfg(test)]
mod strict_control_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn admitted_overlay_precedes_krea_env_override() {
        let temp = tempfile::TempDir::new().unwrap();
        let bound = temp.path().join("bound.safetensors");
        let override_file = temp.path().join("override.safetensors");
        std::fs::write(&bound, b"admitted").unwrap();
        std::fs::write(&override_file, b"other").unwrap();
        let advanced = json!({"__portableAuxPaths":{"strict_control":bound}})
            .as_object()
            .unwrap()
            .clone();
        assert_eq!(
            selected_krea_control_override(&advanced, override_file.to_str())
                .unwrap()
                .as_deref(),
            Some(bound.as_path())
        );
        let missing = json!({"__portableAuxPaths":{}})
            .as_object()
            .unwrap()
            .clone();
        assert!(selected_krea_control_override(&missing, override_file.to_str()).is_err());
        assert_eq!(
            selected_krea_control_override(&JsonObject::new(), override_file.to_str())
                .unwrap()
                .as_deref(),
            Some(override_file.as_path())
        );
    }
}
