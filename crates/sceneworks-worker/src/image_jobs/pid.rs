// Per-generation PiD (pixel-diffusion) super-resolving decoder routing (epic 7840, sc-7849).
//
// PiD is an OPTIONAL, per-generation replacement for a model's VAE decoder: when the request opts in
// (`advanced.usePid`) and a PiD checkpoint for the model's latent space is available, the engine swaps
// `vae.decode(latent)` for a PiD `decode + 4x super-resolve` pass (mlx-gen `PidEngine`, sc-7843/7845).
// PiD is tied to a LATENT SPACE, not a model, so eligibility keys on the model's backbone.
//
// This file is `include!`d on either face backend (macOS/MLX or off-Mac/candle). `resolve_pid_weights`
// is backend-neutral (it only inspects the request + probes the HF cache), so it feeds BOTH the generic
// MLX `generate*` lanes (base.rs/qwen.rs, macOS-only) and the candle InstantID Angles/Poses lane
// (instantid.rs, sc-8373). It threads the toggle into the load-time `LoadSpec::with_pid` aux-weights +
// the per-gen `GenerationRequest.use_pid` flag; the engine errors loudly if `use_pid` is set without
// `spec.pid`, so the caller resolves the two together (both set, or neither → native VAE).

// Default PiD checkpoint + Gemma-2 caption-encoder provisioning. These are the turnkey re-host
// convention that the Phase-3 provisioning story (sc-7852) finalizes + makes downloadable; until then
// the snapshots are simply absent, so `resolve_pid_weights` returns `None` and decode stays on the
// native VAE (behavior-preserving). Both are overridable per-request via `advanced.pidCheckpoint`
// (`{repo, filename}`) and `advanced.pidGemma` (repo string), mirroring `advanced.controlWeights`.
const PID_QWENIMAGE_REPO: &str = "SceneWorks/pid-qwenimage";
const PID_QWENIMAGE_FILE: &str = "pid_qwenimage_2kto4k.safetensors";
// flux latent space (FLUX.1 / Boogu / Chroma / Z-Image — sc-7846); flux2 (FLUX.2 / klein / Lens /
// Ideogram — sc-7847); sdxl (SDXL / RealVisXL / Kolors — sc-7848). All re-hosted by sc-7852 (the
// `SceneWorks/pid-flux2` repo carries the bug-fixed `_2606` 2kto4k revision; the filenames below are the
// canonical re-host names).
const PID_FLUX_REPO: &str = "SceneWorks/pid-flux";
const PID_FLUX_FILE: &str = "pid_flux_2kto4k.safetensors";
const PID_FLUX2_REPO: &str = "SceneWorks/pid-flux2";
const PID_FLUX2_FILE: &str = "pid_flux2_2kto4k.safetensors";
const PID_SDXL_REPO: &str = "SceneWorks/pid-sdxl";
const PID_SDXL_FILE: &str = "pid_sdxl_2kto4k.safetensors";
// gemma-2-2b-it is the PiD caption encoder (shared by every backbone). sc-8025 re-hosts the stock
// weights (no conversion) at the non-gated `SceneWorks/gemma-2-2b-it` mirror so the in-app download
// needs no Gemma-gated HF token; the catalog `downloads[]` + `pidDecoders.<bb>.gemmaRepo` point here
// too (they must agree, or this snapshot is never cached → native VAE). The `advanced.pidGemma`
// per-request override still wins. Gemma Terms §3.1 permit the mirror (terms ship alongside the weights).
const PID_GEMMA_REPO: &str = "SceneWorks/gemma-2-2b-it";

/// Map a SceneWorks image model id to its PiD latent-space backbone, or `None` when the model has no
/// PiD backbone (so `usePid` is silently ignored — the guard for SenseNova et al.). All four wired
/// latent spaces are mapped here (sc-7845 qwenimage, sc-7846 flux, sc-7847 flux2, sc-7848 sdxl); the
/// returned string selects the default re-host repo in `resolve_pid_weights` (the engine itself picks
/// its backbone from each crate's own `PID_BACKBONE` constant, so this is repo-selection only).
///
/// This routes the standard t2i/i2i path (the generic `ImageRoute::Mlx` `generate_stream`). The
/// bespoke advanced sub-mode streams (control / edit / IP-Adapter / inpaint / InstantID / PuLID) build
/// their own request and do not yet thread PiD — they stay on the native VAE, mirroring the engine-side
/// scope decisions in sc-7846/47/48 (tracked as a follow-up).
fn pid_backbone_for(model: &str) -> Option<&'static str> {
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
        "flux_dev"
        | "flux_schnell"
        | "boogu_image"
        | "boogu_image_turbo"
        | "boogu_image_edit"
        | "chroma1_hd"
        | "chroma1_base"
        | "chroma1_flash"
        | "z_image_turbo"
        | "z_image_edit" => Some("flux"),
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
        "sdxl" | "realvisxl" | "realvisxl_lightning" | "kolors" | "instantid_realvisxl" => {
            Some("sdxl")
        }
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
/// Returns `Ok(None)` whenever ANY of: the request did not opt in (`advanced.usePid` unset/false); the
/// model has no PiD backbone (non-eligible → silently ignore the toggle); or the PiD checkpoint / Gemma
/// snapshot is not cached under `data_dir` yet (provisioning is sc-7852). The checkpoint repo+filename
/// and the Gemma repo default to the turnkey convention and may be overridden via
/// `advanced.pidCheckpoint` (`{repo, filename}`) / `advanced.pidGemma` (repo string). An override
/// filename that is not a plain component is an error (sc-8821 / F-019 — a `../…` filename must not
/// escape the snapshot).
///
/// The caller sets BOTH `LoadSpec::with_pid(checkpoint, gemma)` and `GenerationRequest.use_pid = true`
/// exactly when this returns `Ok(Some)` — never one without the other (the engine rejects that).
fn resolve_pid_weights(
    request: &ImageRequest,
    data_dir: &Path,
    model: &str,
) -> WorkerResult<Option<gen_core::PidWeights>> {
    if !pid_requested(request) {
        return Ok(None);
    }
    let Some(backbone) = pid_backbone_for(model) else {
        return Ok(None);
    };
    let (default_repo, default_file) = match backbone {
        "qwenimage" => (PID_QWENIMAGE_REPO, PID_QWENIMAGE_FILE),
        "flux" => (PID_FLUX_REPO, PID_FLUX_FILE),
        "flux2" => (PID_FLUX2_REPO, PID_FLUX2_FILE),
        "sdxl" => (PID_SDXL_REPO, PID_SDXL_FILE),
        _ => return Ok(None),
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
    let filename = safe_weight_filename(
        ckpt_cfg
            .and_then(|c| c.get("filename"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(default_file),
        "advanced.pidCheckpoint.filename",
    )?;
    let gemma_repo = request
        .advanced
        .get("pidGemma")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(PID_GEMMA_REPO);

    // Both snapshots must be cached, or fall through to the native VAE (no partial/half-loaded PiD).
    let Some(snapshot) = huggingface_snapshot_dir(data_dir, repo) else {
        return Ok(None);
    };
    let checkpoint = snapshot.join(filename);
    if !checkpoint.exists() {
        return Ok(None);
    }
    let Some(gemma) = huggingface_snapshot_dir(data_dir, gemma_repo) else {
        return Ok(None);
    };
    Ok(Some(gen_core::PidWeights {
        checkpoint: WeightsSource::File(checkpoint),
        gemma: WeightsSource::Dir(gemma),
    }))
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
    fn backbone_map_covers_all_wired_latent_spaces() {
        // qwenimage (sc-7845)
        assert_eq!(pid_backbone_for("qwen_image"), Some("qwenimage"));
        assert_eq!(pid_backbone_for("qwen_image_edit"), Some("qwenimage"));
        assert_eq!(pid_backbone_for("krea_2_turbo"), Some("qwenimage"));
        assert_eq!(pid_backbone_for("krea_2_raw"), Some("qwenimage"));
        // flux (sc-7846) — incl. Z-Image, which is in the FLUX.1 VAE latent space.
        assert_eq!(pid_backbone_for("flux_dev"), Some("flux"));
        assert_eq!(pid_backbone_for("boogu_image_turbo"), Some("flux"));
        assert_eq!(pid_backbone_for("chroma1_flash"), Some("flux"));
        assert_eq!(pid_backbone_for("z_image_turbo"), Some("flux"));
        // flux2 (sc-7847) — incl. every klein-9b variant, Lens, Ideogram 4.
        assert_eq!(pid_backbone_for("flux2_dev"), Some("flux2"));
        assert_eq!(pid_backbone_for("flux2_klein_9b_true_v2"), Some("flux2"));
        assert_eq!(pid_backbone_for("lens_turbo"), Some("flux2"));
        assert_eq!(pid_backbone_for("ideogram_4"), Some("flux2"));
        // sdxl (sc-7848)
        assert_eq!(pid_backbone_for("sdxl"), Some("sdxl"));
        assert_eq!(pid_backbone_for("realvisxl_lightning"), Some("sdxl"));
        assert_eq!(pid_backbone_for("kolors"), Some("sdxl"));
        // InstantID (sc-8371): the Angles/Poses identity engine reuses the SDXL VAE latent space.
        assert_eq!(pid_backbone_for("instantid_realvisxl"), Some("sdxl"));
        // No PiD backbone → silently ignored (SenseNova is autoregressive, no VAE latent).
        assert_eq!(pid_backbone_for("sensenova_u1_8b"), None);
        assert_eq!(pid_backbone_for("bernini_image"), None);
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
        assert!(resolve_pid_weights(&req, dir.path(), &req.model)
            .expect("plain default filename resolves")
            .is_none());
    }

    #[test]
    fn resolve_returns_none_for_non_eligible_model_even_when_requested() {
        let dir = tempfile::tempdir().unwrap();
        // SenseNova is autoregressive (no VAE latent) → no PiD backbone, toggle ignored.
        let req = request("sensenova_u1_8b", json!({ "usePid": true }));
        assert!(resolve_pid_weights(&req, dir.path(), &req.model)
            .expect("plain default filename resolves")
            .is_none());
    }

    #[test]
    fn resolve_returns_none_when_checkpoint_not_cached() {
        // Opted-in + eligible (every wired backbone), but the PiD checkpoint repo is not in the (empty)
        // HF cache → native VAE.
        let dir = tempfile::tempdir().unwrap();
        for model in ["qwen_image", "flux_dev", "flux2_dev", "sdxl"] {
            let req = request(model, json!({ "usePid": true }));
            assert!(
                resolve_pid_weights(&req, dir.path(), &req.model)
                    .expect("plain default filename resolves")
                    .is_none(),
                "{model} should resolve None when its checkpoint is not cached"
            );
        }
    }

    /// sc-8821 / F-019: a payload `pidCheckpoint.filename` that is not a plain component (traversal,
    /// absolute, sub-path) is REJECTED with an `InvalidPayload` pointing at the field — never joined
    /// under the snapshot. A plain override filename still resolves (Ok).
    #[test]
    fn resolve_rejects_traversal_pid_checkpoint_filenames() {
        let dir = tempfile::tempdir().unwrap();
        for filename in ["../../etc/hosts", "/etc/hosts", "sub/dir.safetensors", ".."] {
            let req = request(
                "qwen_image",
                json!({ "usePid": true, "pidCheckpoint": { "filename": filename } }),
            );
            let error = resolve_pid_weights(&req, dir.path(), &req.model)
                .expect_err("unsafe pidCheckpoint.filename must be rejected");
            assert!(
                error.to_string().contains("advanced.pidCheckpoint.filename"),
                "error should point at the offending field: {error}"
            );
        }
        // A plain override filename passes validation (snapshot absent → Ok(None), native VAE).
        let req = request(
            "qwen_image",
            json!({ "usePid": true, "pidCheckpoint": { "filename": "custom.safetensors" } }),
        );
        assert!(resolve_pid_weights(&req, dir.path(), &req.model)
            .expect("plain override filename resolves")
            .is_none());
    }
}
