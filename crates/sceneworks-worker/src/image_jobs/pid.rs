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
// res2k (2K-tuned) students — flux + flux2 ONLY (sc-10056 re-host / sc-10057 wiring). NVIDIA ships a
// dedicated `res2k` 4-step student (tuned for 512→2048) for these two latent spaces; the 2K output tier
// loads it in place of the multi-res `2kto4k` student when it's cached. qwenimage + sdxl have NO upstream
// res2k student, so they stay on their 2kto4k file at every tier (the Option-A base-cap still gives them a
// real 2K output). Live alongside the 2kto4k files in the SAME re-host repos (whole-repo download).
const PID_FLUX_FILE_2K: &str = "pid_flux_2k.safetensors";
const PID_FLUX2_FILE_2K: &str = "pid_flux2_2k.safetensors";
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

/// PiD super-resolves the sampler latent by a FIXED 4× (`mlx_gen_pid` `sr_scale`, baked into every
/// released student), so the *output* image is always `effective_base × 4`. There is no engine knob for
/// the factor — the only lever on the output resolution is the base fed to the decode. `advanced.pidTarget`
/// (opt-in, opaque pass-through like `usePid`) picks which tier that base×4 lands on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PidOutputTier {
    /// ~2048-px-ceiling output: the effective base long side is capped to 512 (×4 = 2048). Lower
    /// pixel-space decode peak (F-013 memory-relief valve).
    Res2k,
    /// `base × 4` at the requested base, untouched — the default and the byte-identical pre-sc-10054
    /// behavior (a typical 1024 base → 4096 output).
    Res4k,
}

/// PiD's fixed spatial super-resolution factor (`mlx_gen_pid` `sr_scale`). Output pixels = base × this.
const PID_SR_SCALE: u32 = 4;
/// Base-dimension granularity for the 2K tier. Every PiD-eligible engine requires the base width/height
/// to be a multiple of at least 16 (`mlx-gen-flux` `model.rs` validates `is_multiple_of(16)`; the flux2
/// packed latent is coarser). Snap the down-scaled 2K base to 32 so it stays legal for ALL backbones.
const PID_DIM_MULTIPLE: u32 = 32;

/// Resolve the requested PiD output tier from `advanced.pidTarget` (sc-10054). Default + any
/// unrecognized value → `Res4k` (today's full-resolution behavior); only an explicit `"2k"` opts down.
pub(crate) fn pid_output_tier(request: &ImageRequest) -> PidOutputTier {
    match request
        .advanced
        .get("pidTarget")
        .and_then(Value::as_str)
        .map(str::trim)
    {
        Some(tier) if tier.eq_ignore_ascii_case("2k") => PidOutputTier::Res2k,
        _ => PidOutputTier::Res4k,
    }
}

/// The effective base (pre-super-resolve) dimensions to hand the engine so PiD's `base × 4` output lands
/// on `tier`. Only reshapes when PiD actually runs (`use_pid`) AND the caller asked for `2k`; otherwise
/// returns the request dims unchanged (the byte-identical default path — so a stray `pidTarget` on a
/// non-PiD generation can never shrink a native-VAE image). For `2k`, scales the requested aspect down
/// so the longer output side is ~2048 (base long side ≤ 512), snapping each side to `PID_DIM_MULTIPLE`
/// (min 256). Never upscales: a base already at/under the 2K ceiling is left as-is.
pub(crate) fn pid_effective_dims(
    width: u32,
    height: u32,
    use_pid: bool,
    tier: PidOutputTier,
) -> (u32, u32) {
    if !use_pid || tier == PidOutputTier::Res4k {
        return (width, height);
    }
    let base_cap = 2048 / PID_SR_SCALE; // 512: max base long side for a ~2K output
    let longest = width.max(height);
    if longest <= base_cap {
        return (width, height);
    }
    let scale = f64::from(base_cap) / f64::from(longest);
    let snap = |v: u32| {
        let scaled = (f64::from(v) * scale).round();
        let rounded = (scaled / f64::from(PID_DIM_MULTIPLE)).round() as u32 * PID_DIM_MULTIPLE;
        rounded.max(256)
    };
    (snap(width), snap(height))
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

/// The `res2k` (2K-tuned) student filename for a backbone that ships one — flux + flux2 (sc-10056).
/// qwenimage + sdxl have no upstream res2k student → `None` (they stay on their 2kto4k file at every tier).
fn pid_res2k_file(backbone: &str) -> Option<&'static str> {
    match backbone {
        "flux" => Some(PID_FLUX_FILE_2K),
        "flux2" => Some(PID_FLUX2_FILE_2K),
        _ => None,
    }
}

/// The effective default PiD checkpoint filename for `backbone` at `tier`, given the resolved `snapshot`
/// dir (sc-10057). The 2K tier prefers the 2K-tuned res2k student when this backbone ships one AND it is
/// actually present on disk; otherwise (4K tier, no res2k student, or res2k not yet cached) the multi-res
/// `2kto4k` student. The on-disk check is the graceful fallback: an install that predates the res2k file
/// keeps working on 2kto4k rather than resolving a missing path. A per-request `advanced.pidCheckpoint`
/// override still wins over this (applied by the caller).
fn pid_default_file(
    backbone: &str,
    tier: PidOutputTier,
    snapshot: &Path,
    file_2kto4k: &'static str,
) -> &'static str {
    if tier == PidOutputTier::Res2k {
        if let Some(res2k) = pid_res2k_file(backbone) {
            if snapshot.join(res2k).exists() {
                return res2k;
            }
        }
    }
    file_2kto4k
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
    let (default_repo, default_2kto4k_file) = match backbone {
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
    // Validate a per-request filename override EARLY — before the cache check below — so a traversal /
    // non-plain-component filename is rejected with a field-pointed `InvalidPayload` rather than silently
    // ignored when the repo happens not to be cached (sc-8821 / F-019). `None` ⇒ use the tier default.
    let filename_override = ckpt_cfg
        .and_then(|c| c.get("filename"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| safe_weight_filename(s, "advanced.pidCheckpoint.filename"))
        .transpose()?;
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
    // No override → the tier-aware default (sc-10057): the 2K output tier loads the 2K-tuned res2k student
    // for the backbones that ship one (flux/flux2) when it's actually cached; otherwise the 2kto4k student.
    // Resolved after the snapshot dir so the on-disk check gates the swap — an existing install that
    // predates the res2k file (whole-repo installs mark "installed" without re-fetching) falls back to
    // 2kto4k. The default filenames are trusted consts, so they skip the override's traversal validation.
    let filename = filename_override.unwrap_or_else(|| {
        pid_default_file(backbone, pid_output_tier(request), &snapshot, default_2kto4k_file).to_owned()
    });
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
    fn pid_output_tier_defaults_to_4k_and_reads_2k() {
        // Default (no key) + explicit 4k + garbage → Res4k; only "2k" (case-insensitive) → Res2k.
        assert_eq!(
            pid_output_tier(&request("qwen_image", json!({}))),
            PidOutputTier::Res4k
        );
        assert_eq!(
            pid_output_tier(&request("qwen_image", json!({ "pidTarget": "4k" }))),
            PidOutputTier::Res4k
        );
        assert_eq!(
            pid_output_tier(&request("qwen_image", json!({ "pidTarget": "8k" }))),
            PidOutputTier::Res4k
        );
        assert_eq!(
            pid_output_tier(&request("qwen_image", json!({ "pidTarget": "2K" }))),
            PidOutputTier::Res2k
        );
    }

    #[test]
    fn pid_effective_dims_passthrough_unless_2k_and_pid() {
        // Non-PiD, or 4k tier → request dims untouched (byte-identical default; no shrink of a native
        // VAE image even if pidTarget leaks in).
        assert_eq!(
            pid_effective_dims(1024, 1024, false, PidOutputTier::Res2k),
            (1024, 1024)
        );
        assert_eq!(
            pid_effective_dims(1024, 1024, true, PidOutputTier::Res4k),
            (1024, 1024)
        );
    }

    #[test]
    fn pid_effective_dims_2k_caps_base_to_512_long_side() {
        // 1024² base → 512² (×4 = 2048² output). Aspect preserved, snapped to /32.
        assert_eq!(
            pid_effective_dims(1024, 1024, true, PidOutputTier::Res2k),
            (512, 512)
        );
        // 16:9 (1024×576) → longest 1024 halves to 512; 576→288 (both /32) → 2048×1152 output.
        assert_eq!(
            pid_effective_dims(1024, 576, true, PidOutputTier::Res2k),
            (512, 288)
        );
        // Portrait mirror.
        assert_eq!(
            pid_effective_dims(576, 1024, true, PidOutputTier::Res2k),
            (288, 512)
        );
        // A base already at/under the 2K ceiling is left as-is (never upscaled).
        assert_eq!(
            pid_effective_dims(512, 512, true, PidOutputTier::Res2k),
            (512, 512)
        );
        assert_eq!(
            pid_effective_dims(384, 256, true, PidOutputTier::Res2k),
            (384, 256)
        );
    }

    #[test]
    fn pid_effective_dims_2k_stays_dimension_legal() {
        // Every reshaped side must remain a multiple of 16 (the strictest engine requirement) and ≥256,
        // across a spread of requested bases — else the engine rejects the base (mlx-gen-flux model.rs).
        for (w, h) in [
            (4096u32, 4096u32),
            (2048, 1024),
            (1536, 1024),
            (1024, 768),
            (2048, 512),
            (4096, 256),
        ] {
            let (ew, eh) = pid_effective_dims(w, h, true, PidOutputTier::Res2k);
            assert!(ew.is_multiple_of(16) && eh.is_multiple_of(16), "{ew}x{eh} not /16");
            assert!(ew >= 256 && eh >= 256, "{ew}x{eh} below min");
            assert!(ew.max(eh) <= 512, "{ew}x{eh} base long side exceeds 2K cap");
        }
    }

    #[test]
    fn pid_default_file_selects_res2k_for_2k_when_cached_else_falls_back() {
        let dir = tempfile::tempdir().unwrap();
        let snap = dir.path();
        // 4K tier → always the 2kto4k student, even for a res2k-capable backbone with the file present.
        std::fs::write(snap.join(PID_FLUX_FILE_2K), b"x").unwrap();
        assert_eq!(
            pid_default_file("flux", PidOutputTier::Res4k, snap, PID_FLUX_FILE),
            PID_FLUX_FILE
        );
        // 2K tier + res2k present → the tuned student.
        assert_eq!(
            pid_default_file("flux", PidOutputTier::Res2k, snap, PID_FLUX_FILE),
            PID_FLUX_FILE_2K
        );
        // flux2 too.
        std::fs::write(snap.join(PID_FLUX2_FILE_2K), b"x").unwrap();
        assert_eq!(
            pid_default_file("flux2", PidOutputTier::Res2k, snap, PID_FLUX2_FILE),
            PID_FLUX2_FILE_2K
        );
    }

    #[test]
    fn pid_default_file_falls_back_when_res2k_absent_or_unsupported() {
        let empty = tempfile::tempdir().unwrap();
        let snap = empty.path(); // no res2k file staged
        // 2K tier but the res2k file isn't cached yet → graceful fallback to 2kto4k (existing installs).
        assert_eq!(
            pid_default_file("flux", PidOutputTier::Res2k, snap, PID_FLUX_FILE),
            PID_FLUX_FILE
        );
        // qwenimage + sdxl have no upstream res2k student → 2kto4k at the 2K tier (even if a stray file
        // existed, `pid_res2k_file` returns None so it's never chosen).
        assert_eq!(
            pid_default_file("qwenimage", PidOutputTier::Res2k, snap, PID_QWENIMAGE_FILE),
            PID_QWENIMAGE_FILE
        );
        assert_eq!(
            pid_default_file("sdxl", PidOutputTier::Res2k, snap, PID_SDXL_FILE),
            PID_SDXL_FILE
        );
    }

    #[test]
    fn pid_res2k_file_only_flux_and_flux2() {
        assert_eq!(pid_res2k_file("flux"), Some(PID_FLUX_FILE_2K));
        assert_eq!(pid_res2k_file("flux2"), Some(PID_FLUX2_FILE_2K));
        assert_eq!(pid_res2k_file("qwenimage"), None);
        assert_eq!(pid_res2k_file("sdxl"), None);
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
