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

#[cfg(test)]
mod tests {
    use super::*;

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
}
