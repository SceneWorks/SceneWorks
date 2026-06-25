// Per-generation PiD (pixel-diffusion) super-resolving decoder routing (epic 7840, sc-7849).
//
// PiD is an OPTIONAL, per-generation replacement for a model's VAE decoder: when the request opts in
// (`advanced.usePid`) and a PiD checkpoint for the model's latent space is available, the engine swaps
// `vae.decode(latent)` for a PiD `decode + 4x super-resolve` pass (mlx-gen `PidEngine`, sc-7843/7845).
// PiD is tied to a LATENT SPACE, not a model, so eligibility keys on the model's backbone.
//
// This file is `include!`d on macOS only (the MLX lane); the candle PiD duplicate is Phase 4 (sc-7853).
// It threads the toggle into the load-time `LoadSpec::with_pid` aux-weights + the per-gen
// `GenerationRequest.use_pid` flag; the engine errors loudly if `use_pid` is set without `spec.pid`, so
// the caller resolves the two together (both set, or neither → native VAE).

// Default PiD checkpoint + Gemma-2 caption-encoder provisioning. These are the turnkey re-host
// convention that the Phase-3 provisioning story (sc-7852) finalizes + makes downloadable; until then
// the snapshots are simply absent, so `resolve_pid_weights` returns `None` and decode stays on the
// native VAE (behavior-preserving). Both are overridable per-request via `advanced.pidCheckpoint`
// (`{repo, filename}`) and `advanced.pidGemma` (repo string), mirroring `advanced.controlWeights`.
const PID_QWENIMAGE_REPO: &str = "SceneWorks/pid-qwenimage";
const PID_QWENIMAGE_FILE: &str = "pid_qwenimage_2kto4k.safetensors";
// gemma-2-2b-it is the PiD caption encoder. sc-7852 finalizes the concrete source (likely a SceneWorks
// mirror to avoid the upstream gated repo); the default points at the canonical id for now.
const PID_GEMMA_REPO: &str = "google/gemma-2-2b-it";

/// Map a SceneWorks image model id to its PiD latent-space backbone, or `None` when the model has no
/// PiD backbone (so `usePid` is silently ignored — the guard for SenseNova et al.). Today only the
/// `qwenimage` latent space is wired (sc-7845): Qwen-Image (incl. its strict-pose control variant,
/// which shares the `qwen_image` model id), Qwen-Image-Edit, and Krea 2 Turbo (reuses `QwenVae`). The
/// flux / flux2 / sdxl backbones light up in sc-7846 / 7847 / 7848.
fn pid_backbone_for(model: &str) -> Option<&'static str> {
    match model {
        // Qwen-Image T2I + its strict-pose control variant (both the `qwen_image` model id), every
        // Qwen-Image-Edit variant (all → the one `qwen_image_edit` engine), and Krea 2 Turbo.
        "qwen_image"
        | "qwen_image_edit"
        | "qwen_image_edit_2509"
        | "qwen_image_edit_2511"
        | "qwen_image_edit_2511_lightning"
        | "krea_2_turbo" => Some("qwenimage"),
        _ => None,
    }
}

/// True when the request opted into the PiD decoder via `advanced.usePid` (an opaque pass-through bool,
/// like `mlxQuantize` / `schedulerShift` — no top-level `ImageRequest` field, so zero contract-snapshot
/// drift). Accepts a JSON bool or the string `"true"`.
fn pid_requested(request: &ImageRequest) -> bool {
    request
        .advanced
        .get("usePid")
        .map(|value| {
            value.as_bool().unwrap_or_else(|| {
                value
                    .as_str()
                    .is_some_and(|s| s.trim().eq_ignore_ascii_case("true"))
            })
        })
        .unwrap_or(false)
}

/// Resolve the per-generation PiD decoder weights for `model`, or `None` to keep the native VAE decode.
///
/// Returns `None` whenever ANY of: the request did not opt in (`advanced.usePid` unset/false); the
/// model has no PiD backbone (non-eligible → silently ignore the toggle); or the PiD checkpoint / Gemma
/// snapshot is not cached under `data_dir` yet (provisioning is sc-7852). The checkpoint repo+filename
/// and the Gemma repo default to the turnkey convention and may be overridden via
/// `advanced.pidCheckpoint` (`{repo, filename}`) / `advanced.pidGemma` (repo string).
///
/// The caller sets BOTH `LoadSpec::with_pid(checkpoint, gemma)` and `GenerationRequest.use_pid = true`
/// exactly when this returns `Some` — never one without the other (the engine rejects that).
fn resolve_pid_weights(
    request: &ImageRequest,
    data_dir: &Path,
    model: &str,
) -> Option<gen_core::PidWeights> {
    if !pid_requested(request) {
        return None;
    }
    let backbone = pid_backbone_for(model)?;
    let (default_repo, default_file) = match backbone {
        "qwenimage" => (PID_QWENIMAGE_REPO, PID_QWENIMAGE_FILE),
        // flux / flux2 / sdxl backbones (sc-7846/47/48) register their defaults here as they land.
        _ => return None,
    };

    let ckpt_cfg = request
        .advanced
        .get("pidCheckpoint")
        .and_then(Value::as_object);
    let repo = ckpt_cfg
        .and_then(|c| c.get("repo"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(default_repo);
    let filename = ckpt_cfg
        .and_then(|c| c.get("filename"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(default_file);
    let gemma_repo = request
        .advanced
        .get("pidGemma")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(PID_GEMMA_REPO);

    // Both snapshots must be cached, or fall through to the native VAE (no partial/half-loaded PiD).
    let checkpoint = huggingface_snapshot_dir(data_dir, repo)?.join(filename);
    if !checkpoint.exists() {
        return None;
    }
    let gemma = huggingface_snapshot_dir(data_dir, gemma_repo)?;
    Some(gen_core::PidWeights {
        checkpoint: WeightsSource::File(checkpoint),
        gemma: WeightsSource::Dir(gemma),
    })
}

#[cfg(all(target_os = "macos", test))]
mod pid_tests {
    use super::*;
    use serde_json::json;

    fn request(model: &str, advanced: Value) -> ImageRequest {
        ImageRequest::from_payload(
            json!({ "model": model, "advanced": advanced })
                .as_object()
                .unwrap(),
        )
    }

    #[test]
    fn backbone_map_covers_qwenimage_family_only() {
        assert_eq!(pid_backbone_for("qwen_image"), Some("qwenimage"));
        assert_eq!(pid_backbone_for("qwen_image_edit"), Some("qwenimage"));
        assert_eq!(pid_backbone_for("krea_2_turbo"), Some("qwenimage"));
        // Not yet wired / no PiD backbone → silently ignored.
        assert_eq!(pid_backbone_for("flux_dev"), None);
        assert_eq!(pid_backbone_for("sdxl"), None);
        assert_eq!(pid_backbone_for("sensenova_u1_8b"), None);
    }

    #[test]
    fn pid_requested_reads_bool_and_string() {
        assert!(pid_requested(&request("qwen_image", json!({ "usePid": true }))));
        assert!(pid_requested(&request("qwen_image", json!({ "usePid": "true" }))));
        assert!(!pid_requested(&request("qwen_image", json!({ "usePid": false }))));
        assert!(!pid_requested(&request("qwen_image", json!({}))));
    }

    #[test]
    fn resolve_returns_none_without_opt_in() {
        let dir = tempfile::tempdir().unwrap();
        let req = request("qwen_image", json!({}));
        assert!(resolve_pid_weights(&req, dir.path(), &req.model).is_none());
    }

    #[test]
    fn resolve_returns_none_for_non_eligible_model_even_when_requested() {
        let dir = tempfile::tempdir().unwrap();
        let req = request("sdxl", json!({ "usePid": true }));
        assert!(resolve_pid_weights(&req, dir.path(), &req.model).is_none());
    }

    #[test]
    fn resolve_returns_none_when_checkpoint_not_cached() {
        // Opted-in + eligible, but the PiD checkpoint repo is not in the (empty) HF cache → native VAE.
        let dir = tempfile::tempdir().unwrap();
        let req = request("qwen_image", json!({ "usePid": true }));
        assert!(resolve_pid_weights(&req, dir.path(), &req.model).is_none());
    }
}
