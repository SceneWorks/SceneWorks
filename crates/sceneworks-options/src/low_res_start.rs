//! Portable Low-Resolution Start (`advanced.lowResStart`) parsing, capability-driven validation,
//! and quantized two-phase trajectory resolution (SW-6942: `FEAT-1`, `PIPE-2`, `E5`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::contracts::JsonObject;
use crate::request_validation::ValidationRefusal;

pub const LOW_RES_START_KEY: &str = "lowResStart";
pub const LOW_RES_START_REPORT_KEY: &str = "lowResStartExecutionReport";
pub const LOW_RES_START_REPORT_SNAKE_KEY: &str = "low_res_start_execution_report";
pub const LOW_RES_START_VERSION: u32 = 1;
pub const LOW_RES_START_MIN_SCALE: f64 = 0.25;
pub const LOW_RES_START_MAX_SCALE: f64 = 0.9;
pub const LOW_RES_START_QUANTUM: u32 = 16;
pub const LOW_RES_START_MIN_DIM: u32 = 256;

/// Authored `advanced.lowResStart` wire object (`{ version: 1, k, scale }`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LowResStartWireConfig {
    pub version: u32,
    pub k: u32,
    pub scale: f64,
}

/// Validated and quantized two-phase low-resolution start resolution for a concrete request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LowResStartResolved {
    pub version: u32,
    pub k: u32,
    pub scale: f32,
    pub executed_steps: u32,
    pub low_width: u32,
    pub low_height: u32,
    pub full_width: u32,
    pub full_height: u32,
}

fn normalize_serde_path(raw_path: &str, detail: &str) -> String {
    if let Some(missing) = detail
        .strip_prefix("missing field `")
        .and_then(|rest| rest.split('`').next())
    {
        return match missing {
            "version" => "advanced.lowResStart.version".to_owned(),
            "k" => "advanced.lowResStart.k".to_owned(),
            "scale" => "advanced.lowResStart.scale".to_owned(),
            _ => "advanced.lowResStart".to_owned(),
        };
    }
    let trimmed = raw_path.trim_matches('.');
    if trimmed.is_empty() {
        return "advanced.lowResStart".to_owned();
    }
    let head = trimmed.split(['.', '[']).next().unwrap_or_default();
    match head {
        "version" => "advanced.lowResStart.version".to_owned(),
        "k" => "advanced.lowResStart.k".to_owned(),
        "scale" => "advanced.lowResStart.scale".to_owned(),
        _ => "advanced.lowResStart".to_owned(),
    }
}

/// Strictly decode and range-check a raw JSON value as [`LowResStartWireConfig`], rejecting unknown
/// fields, unsupported versions, non-positive `k`, or out-of-range `scale` without clamping.
pub fn parse_low_res_start_wire_value(
    value: &Value,
) -> Result<LowResStartWireConfig, ValidationRefusal> {
    if !value.is_object() {
        return Err(ValidationRefusal::new(
            "advanced.lowResStart",
            format!("advanced.lowResStart must be an object, got {value}"),
        ));
    }
    let wire = serde_path_to_error::deserialize::<_, LowResStartWireConfig>(value.clone())
        .map_err(|error| {
            let inner = error.inner().to_string();
            let field = normalize_serde_path(&error.path().to_string(), &inner);
            ValidationRefusal::new(field, format!("advanced.lowResStart: {inner}"))
        })?;
    wire.validate_bounds()?;
    Ok(wire)
}

impl LowResStartWireConfig {
    pub fn validate_bounds(&self) -> Result<(), ValidationRefusal> {
        if self.version != LOW_RES_START_VERSION {
            return Err(ValidationRefusal::new(
                "advanced.lowResStart.version",
                format!(
                    "advanced.lowResStart.version must be {LOW_RES_START_VERSION} (got {})",
                    self.version
                ),
            ));
        }
        if self.k == 0 {
            return Err(ValidationRefusal::new(
                "advanced.lowResStart.k",
                "advanced.lowResStart.k must be >= 1",
            ));
        }
        let narrowed = self.scale as f32;
        if !self.scale.is_finite()
            || !narrowed.is_finite()
            || !(LOW_RES_START_MIN_SCALE..=LOW_RES_START_MAX_SCALE).contains(&self.scale)
        {
            return Err(ValidationRefusal::new(
                "advanced.lowResStart.scale",
                format!(
                    "advanced.lowResStart.scale must be in [{LOW_RES_START_MIN_SCALE}, {LOW_RES_START_MAX_SCALE}] without clamping (got {})",
                    self.scale
                ),
            ));
        }
        Ok(())
    }
}

/// Quantize `(full_width, full_height)` by `scale` onto `legal_quantum` (`floor((full * scale) / q) * q`)
/// and enforce `low_w >= min_dim`, `low_h >= min_dim`, `low_w < full_width`, `low_h < full_height`
/// without clamping.
pub fn quantize_low_res_dimensions(
    full_width: u32,
    full_height: u32,
    scale: f64,
    legal_quantum: u32,
    min_dim: u32,
) -> Result<(u32, u32), ValidationRefusal> {
    if legal_quantum == 0
        || !full_width.is_multiple_of(legal_quantum)
        || !full_height.is_multiple_of(legal_quantum)
    {
        return Err(ValidationRefusal::new(
            "width",
            format!(
                "target resolution {full_width}x{full_height} must be a multiple of {legal_quantum} for Low Res Start"
            ),
        ));
    }
    let scale_f32 = scale as f32;
    let q_f32 = legal_quantum as f32;
    let low_w = (((full_width as f32 * scale_f32) / q_f32).floor() as u32) * legal_quantum;
    let low_h = (((full_height as f32 * scale_f32) / q_f32).floor() as u32) * legal_quantum;
    if low_w < min_dim || low_h < min_dim {
        return Err(ValidationRefusal::new(
            "advanced.lowResStart.scale",
            format!(
                "advanced.lowResStart.scale {scale} quantizes {full_width}x{full_height} to {low_w}x{low_h}, below minimum dimension {min_dim}x{min_dim}"
            ),
        ));
    }
    if low_w >= full_width || low_h >= full_height {
        return Err(ValidationRefusal::new(
            "advanced.lowResStart.scale",
            format!(
                "advanced.lowResStart.scale {scale} quantizes {full_width}x{full_height} to {low_w}x{low_h}, which must be strictly smaller than {full_width}x{full_height}"
            ),
        ));
    }
    Ok((low_w, low_h))
}

/// PiD super-resolves the sampler latent by a FIXED 4x (`mlx_gen_pid` `sr_scale`, baked into every
/// released student), so the *output* image is always `effective_base * 4`. `advanced.pidTarget`
/// picks which tier that `base * 4` lands on (`SW-7221`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PidOutputTier {
    /// ~2048-px-ceiling output: the effective base long side is capped to 512 (`* 4 = 2048`).
    Res2k,
    /// `base * 4` at the requested base, untouched — the default behavior.
    Res4k,
}

const PID_SR_SCALE: u32 = 4;
const PID_DIM_MULTIPLE: u32 = 32;

/// Map a SceneWorks image model id to its PiD latent-space backbone, or `None` when the model has
/// no PiD backbone (`SW-7221`).
pub fn pid_backbone_for(model: &str) -> Option<&'static str> {
    match model {
        "qwen_image"
        | "qwen_image_edit"
        | "qwen_image_edit_2509"
        | "qwen_image_edit_2511"
        | "qwen_image_edit_2511_lightning"
        | "krea_2_turbo"
        | "krea_2_raw" => Some("qwenimage"),
        "flux_dev" | "flux_schnell" | "pulid_flux_dev" | "boogu_image" | "boogu_image_turbo"
        | "boogu_image_edit" | "chroma1_hd" | "chroma1_base" | "chroma1_flash"
        | "z_image_turbo" | "z_image_edit" => Some("flux"),
        "flux2_dev"
        | "flux2_klein_9b"
        | "flux2_klein_9b_kv"
        | "flux2_klein_9b_true_v2"
        | "lens"
        | "lens_turbo"
        | "ideogram_4"
        | "ideogram_4_turbo" => Some("flux2"),
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

/// Resolve the requested PiD output tier from an optional `advanced.pidTarget` string (`SW-7221`).
/// Default + any unrecognized value => `PidOutputTier::Res4k`; only an explicit `"2k"` opts down.
pub fn pid_output_tier_from_str(target: Option<&str>) -> PidOutputTier {
    match target.map(str::trim) {
        Some(tier) if tier.eq_ignore_ascii_case("2k") => PidOutputTier::Res2k,
        _ => PidOutputTier::Res4k,
    }
}

/// The effective base (pre-super-resolve) dimensions to hand the engine so PiD's `base * 4` output
/// lands on `tier` (`SW-7221`). Shared across Studio/API validation (`sceneworks-options`), core
/// request resolution (`sceneworks-core`), and worker execution (`sceneworks-worker`).
pub fn pid_effective_dims(
    width: u32,
    height: u32,
    use_pid: bool,
    tier: PidOutputTier,
) -> (u32, u32) {
    if !use_pid || tier == PidOutputTier::Res4k {
        return (width, height);
    }
    let base_cap = 2048 / PID_SR_SCALE;
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

/// Resolve the effective base dimensions `(width, height)` for `lowResStart` validation on a raw
/// option map, accounting for PiD base reduction when `usePid` is active on a PiD-eligible model
/// (`SW-7221`).
pub fn effective_base_dimensions_for_low_res_start(
    model: &str,
    width: u32,
    height: u32,
    advanced: &JsonObject,
) -> (u32, u32) {
    let use_pid = pid_backbone_for(model).is_some()
        && advanced.get("usePid").is_some_and(crate::legacy_image_flag);
    let tier = pid_output_tier_from_str(advanced.get("pidTarget").and_then(Value::as_str));
    pid_effective_dims(width, height, use_pid, tier)
}

/// Validate a parsed img2img `strength` value (finite number in `[0.0, 1.0]`, without clamping)
/// (`SW-7220`).
pub fn validate_img2img_strength_value(
    strength: f64,
    field: &'static str,
) -> Result<f64, ValidationRefusal> {
    if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
        return Err(ValidationRefusal::new(
            field,
            format!(
                "{field} must be in [0.0, 1.0] without clamping when Low Res Start is active (got {strength})"
            ),
        ));
    }
    Ok(strength)
}

/// Resolve the raw img2img strength carrier across `advanced.strength` (preferred) and top-level
/// `strength` (`SW-7220`).
pub fn resolve_raw_img2img_strength(
    payload: &JsonObject,
    advanced: &JsonObject,
) -> Result<Option<f64>, ValidationRefusal> {
    let Some(v) = advanced
        .get("strength")
        .filter(|v| !v.is_null())
        .or_else(|| payload.get("strength").filter(|v| !v.is_null()))
    else {
        return Ok(None);
    };
    let strength = v
        .as_f64()
        .or_else(|| v.as_str()?.trim().parse::<f64>().ok())
        .ok_or_else(|| {
            ValidationRefusal::new(
                "advanced.strength",
                "advanced.strength must be a finite number in [0.0, 1.0]",
            )
        })?;
    validate_img2img_strength_value(strength, "advanced.strength").map(Some)
}

/// Compute the executed step count `M` after img2img strength truncation (`§6.13`, `SW-7220`).
pub fn executed_steps_after_img2img_strength(
    num_steps: u32,
    has_source: bool,
    authored_strength: Option<f64>,
) -> u32 {
    if has_source {
        let s = authored_strength.unwrap_or(0.5) as f32;
        let init_time_step = if s > 0.0 {
            ((num_steps as f32 * s) as u32).max(1)
        } else {
            0
        };
        num_steps.saturating_sub(init_time_step)
    } else {
        num_steps
    }
}

/// Resolve the executed step count `M` after img2img strength truncation (`§6.13`, `SW-7220`).
pub fn resolve_executed_steps(
    model: &str,
    payload: &JsonObject,
    advanced: &JsonObject,
    entry: &JsonObject,
) -> Result<u32, ValidationRefusal> {
    let default_steps = entry
        .get("defaults")
        .and_then(|d| d.get("steps"))
        .and_then(Value::as_u64)
        .or_else(|| {
            entry
                .get("feature_details")
                .and_then(|d| d.pointer("/low_res_start/parameter_schema/k/default_executed_steps"))
                .and_then(Value::as_u64)
        })
        .and_then(|steps| u32::try_from(steps).ok())
        .unwrap_or(match model {
            "flux_schnell" | "flux1_schnell" => 4,
            _ => 28,
        });
    let num_steps = crate::limits::requested_steps(advanced).unwrap_or(default_steps);

    let has_source = payload
        .get("sourceAssetId")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.trim().is_empty());
    let authored_strength = resolve_raw_img2img_strength(payload, advanced)?;
    Ok(executed_steps_after_img2img_strength(
        num_steps,
        has_source,
        authored_strength,
    ))
}

fn has_nonempty_str(map: &JsonObject, key: &str) -> bool {
    map.get(key)
        .and_then(Value::as_str)
        .is_some_and(|s| !s.trim().is_empty())
}

fn has_nonempty_array(map: &JsonObject, key: &str) -> bool {
    map.get(key)
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty())
}

fn has_nonnull(map: &JsonObject, key: &str) -> bool {
    map.get(key).is_some_and(|v| !v.is_null())
}

/// Validate `advanced.lowResStart` on a request payload against `entry["features"]["low_res_start"]`,
/// executed step count `M` (after img2img strength truncation), target geometry, and pass-0
/// conditioning exclusions (`FEAT-1`, `PIPE-2`, `E5`).
pub fn requested_low_res_start(
    model: &str,
    payload: &JsonObject,
    advanced: &JsonObject,
    entry: &JsonObject,
) -> Result<Option<LowResStartResolved>, ValidationRefusal> {
    let Some(raw) = advanced
        .get(LOW_RES_START_KEY)
        .filter(|v| !v.is_null() && *v != &Value::Bool(false))
    else {
        return Ok(None);
    };
    let route_supports = entry
        .get("features")
        .and_then(|f| f.get("low_res_start"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !route_supports {
        return Err(ValidationRefusal::new(
            "advanced.lowResStart",
            "The selected model route does not support Low Res Start.",
        ));
    }
    let wire = parse_low_res_start_wire_value(raw)?;

    if advanced
        .get("phases")
        .is_some_and(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
    {
        return Err(ValidationRefusal::new(
            "advanced.lowResStart",
            "Low Res Start cannot be combined with multi-phase sampling (advanced.phases).",
        ));
    }

    let mode = payload
        .get("mode")
        .and_then(Value::as_str)
        .unwrap_or("text_to_image");
    if !matches!(mode, "text_to_image" | "image_generation") {
        return Err(ValidationRefusal::new(
            "advanced.lowResStart",
            format!(
                "Low Res Start is only supported for text-to-image or ordinary single-source img2img, not mode '{mode}'."
            ),
        ));
    }

    let has_forbidden_conditioning = has_nonempty_str(payload, "referenceAssetId")
        || has_nonempty_array(payload, "referenceAssetIds")
        || has_nonempty_array(payload, "sourceAssetIds")
        || has_nonempty_str(payload, "maskAssetId")
        || has_nonempty_str(payload, "characterId")
        || has_nonempty_str(payload, "characterLookId")
        || has_nonempty_array(payload, "controls")
        || has_nonempty_array(payload, "controlnets")
        || has_nonempty_array(advanced, "poses")
        || has_nonempty_str(advanced, "controlImage")
        || has_nonempty_str(advanced, "controlMode")
        || has_nonnull(advanced, "controlWeights")
        || has_nonempty_str(advanced, "ipAdapterRepo");
    if has_forbidden_conditioning {
        return Err(ValidationRefusal::new(
            "advanced.lowResStart",
            "Low Res Start cannot be combined with pass-0 edit, reference, IP-Adapter, identity, or control conditioning.",
        ));
    }

    let executed_steps = resolve_executed_steps(model, payload, advanced, entry)?;
    if executed_steps < 2 || wire.k >= executed_steps {
        return Err(ValidationRefusal::new(
            "advanced.lowResStart.k",
            format!(
                "advanced.lowResStart.k must satisfy 1 <= k < executed_steps (got k={}, executed_steps={executed_steps})",
                wire.k
            ),
        ));
    }

    let (def_w, def_h) = crate::image_request::default_resolution(entry).unwrap_or((1024, 1024));
    let authored_width = crate::image_request::unsigned_or(payload.get("width"), def_w);
    let authored_height = crate::image_request::unsigned_or(payload.get("height"), def_h);
    let (full_width, full_height) = effective_base_dimensions_for_low_res_start(
        model,
        authored_width,
        authored_height,
        advanced,
    );
    let (low_width, low_height) = quantize_low_res_dimensions(
        full_width,
        full_height,
        wire.scale,
        LOW_RES_START_QUANTUM,
        LOW_RES_START_MIN_DIM,
    )?;

    Ok(Some(LowResStartResolved {
        version: wire.version,
        k: wire.k,
        scale: wire.scale as f32,
        executed_steps,
        low_width,
        low_height,
        full_width,
        full_height,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn flux_dev_entry() -> JsonObject {
        json!({
            "id": "flux_dev",
            "family": "flux",
            "defaults": {
                "steps": 28,
                "resolution": "1024x1024"
            },
            "features": {
                "low_res_start": true
            }
        })
        .as_object()
        .unwrap()
        .clone()
    }

    #[test]
    fn validates_wire_bounds_and_quantization_without_clamping() {
        let entry = flux_dev_entry();
        let payload = json!({
            "model": "flux_dev",
            "mode": "text_to_image",
            "width": 1024,
            "height": 1024,
            "advanced": {
                "steps": 8,
                "lowResStart": { "version": 1, "k": 3, "scale": 0.5 }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let adv = payload["advanced"].as_object().unwrap();
        let resolved = requested_low_res_start("flux_dev", &payload, adv, &entry)
            .expect("valid lowResStart")
            .expect("present");
        assert_eq!(resolved.k, 3);
        assert_eq!(resolved.executed_steps, 8);
        assert_eq!((resolved.low_width, resolved.low_height), (512, 512));
        assert_eq!((resolved.full_width, resolved.full_height), (1024, 1024));

        // Non-power-of-two scale quantizes via floor((full * scale) / 16) * 16:
        // floor((1024 * 0.35) / 16) * 16 = floor(358.4 / 16) * 16 = 22 * 16 = 352.
        let (lw, lh) = quantize_low_res_dimensions(1024, 1024, 0.35, 16, 256).unwrap();
        assert_eq!((lw, lh), (352, 352));

        // Scale out of [0.25, 0.9] is refused, never clamped.
        for bad_scale in [0.2, 0.249, 0.91, 1.0] {
            let err = parse_low_res_start_wire_value(
                &json!({ "version": 1, "k": 2, "scale": bad_scale }),
            )
            .unwrap_err();
            assert_eq!(err.field, "advanced.lowResStart.scale");
        }

        // Quantized dimension below 256 is refused (e.g. 512 * 0.45 = 230.4 -> 224 < 256).
        let err = quantize_low_res_dimensions(512, 512, 0.45, 16, 256).unwrap_err();
        assert_eq!(err.field, "advanced.lowResStart.scale");

        // k >= M is refused.
        let bad_k_payload = json!({
            "model": "flux_dev",
            "mode": "text_to_image",
            "width": 1024,
            "height": 1024,
            "advanced": {
                "steps": 4,
                "lowResStart": { "version": 1, "k": 4, "scale": 0.5 }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let err = requested_low_res_start(
            "flux_dev",
            &bad_k_payload,
            bad_k_payload["advanced"].as_object().unwrap(),
            &entry,
        )
        .unwrap_err();
        assert_eq!(err.field, "advanced.lowResStart.k");
    }

    #[test]
    fn img2img_strength_truncates_executed_steps_m_and_refuses_k_at_or_above_m() {
        let entry = flux_dev_entry();
        // 8 steps with strength 0.5 -> init_time_step = 4 -> M = 4 executed steps.
        // k = 3 is valid (1 <= 3 < 4), k = 4 is refused.
        let valid_img2img = json!({
            "model": "flux_dev",
            "mode": "text_to_image",
            "sourceAssetId": "src_1",
            "width": 1024,
            "height": 1024,
            "advanced": {
                "steps": 8,
                "strength": 0.5,
                "lowResStart": { "version": 1, "k": 3, "scale": 0.5 }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let resolved = requested_low_res_start(
            "flux_dev",
            &valid_img2img,
            valid_img2img["advanced"].as_object().unwrap(),
            &entry,
        )
        .expect("valid img2img lowResStart")
        .expect("present");
        assert_eq!(resolved.executed_steps, 4);
        assert_eq!(resolved.k, 3);

        let invalid_img2img = json!({
            "model": "flux_dev",
            "mode": "text_to_image",
            "sourceAssetId": "src_1",
            "width": 1024,
            "height": 1024,
            "advanced": {
                "steps": 8,
                "strength": 0.5,
                "lowResStart": { "version": 1, "k": 4, "scale": 0.5 }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        let err = requested_low_res_start(
            "flux_dev",
            &invalid_img2img,
            invalid_img2img["advanced"].as_object().unwrap(),
            &entry,
        )
        .unwrap_err();
        assert_eq!(err.field, "advanced.lowResStart.k");
    }

    #[test]
    fn refuses_phases_and_pass0_reference_or_control_conditioning() {
        let entry = flux_dev_entry();
        for (label, extra_top, extra_adv) in [
            ("phases", json!({}), json!({ "phases": [{ "steps": 4 }] })),
            (
                "referenceAssetId",
                json!({ "referenceAssetId": "ref_1" }),
                json!({}),
            ),
            (
                "referenceAssetIds",
                json!({ "referenceAssetIds": ["ref_1"] }),
                json!({}),
            ),
            ("maskAssetId", json!({ "maskAssetId": "mask_1" }), json!({})),
            ("characterId", json!({ "characterId": "char_1" }), json!({})),
            ("poses", json!({}), json!({ "poses": [{}] })),
            (
                "controlImage",
                json!({}),
                json!({ "controlImage": "ctrl_1" }),
            ),
            ("controlMode", json!({}), json!({ "controlMode": "canny" })),
            (
                "controlWeights",
                json!({}),
                json!({ "controlWeights": { "overlayId": "ov_1" } }),
            ),
            (
                "ipAdapterRepo",
                json!({}),
                json!({ "ipAdapterRepo": "XLabs-AI/flux-ip-adapter" }),
            ),
            (
                "edit_image mode",
                json!({ "mode": "edit_image", "sourceAssetId": "src_1" }),
                json!({}),
            ),
            (
                "character_image mode",
                json!({ "mode": "character_image" }),
                json!({}),
            ),
        ] {
            let mut payload = json!({
                "model": "flux_dev",
                "mode": "text_to_image",
                "width": 1024,
                "height": 1024,
                "advanced": {
                    "steps": 8,
                    "lowResStart": { "version": 1, "k": 2, "scale": 0.5 }
                }
            });
            for (k, v) in extra_top.as_object().unwrap() {
                payload[k] = v.clone();
            }
            for (k, v) in extra_adv.as_object().unwrap() {
                payload["advanced"][k] = v.clone();
            }
            let obj = payload.as_object().unwrap();
            let err = requested_low_res_start(
                "flux_dev",
                obj,
                obj["advanced"].as_object().unwrap(),
                &entry,
            )
            .expect_err(label);
            assert_eq!(err.field, "advanced.lowResStart", "{label}");
        }

        // Catalog-backed request with both defaults.steps=28 and feature_details.low_res_start.parameter_schema.k.default_executed_steps=25
        // prefers defaults.steps=28 when advanced.steps is omitted, accepting K=27:
        let mut catalog_entry = flux_dev_entry();
        catalog_entry.insert(
            "feature_details".into(),
            json!({
                "low_res_start": {
                    "parameter_schema": {
                        "k": {
                            "default_executed_steps": 25
                        }
                    }
                }
            }),
        );
        let k27_payload = json!({
            "model": "flux_dev",
            "mode": "text_to_image",
            "width": 1024,
            "height": 1024,
            "advanced": {
                "lowResStart": { "version": 1, "k": 27, "scale": 0.5 }
            }
        });
        let k27_obj = k27_payload.as_object().unwrap();
        let resolved_k27 = requested_low_res_start(
            "flux_dev",
            k27_obj,
            k27_obj["advanced"].as_object().unwrap(),
            &catalog_entry,
        )
        .expect("K=27 with no step override must succeed against defaults.steps=28")
        .expect("low_res_start active");
        assert_eq!(resolved_k27.k, 27);
        assert_eq!(resolved_k27.executed_steps, 28);
    }

    #[test]
    fn sw_7220_top_level_strength_resolves_identically_to_advanced_strength() {
        let entry = flux_dev_entry();
        // 8 steps, top-level strength 0.25 -> init_time_step = 2 -> executed_steps = 6 -> K=5 is valid!
        let top_level_payload = json!({
            "model": "flux_dev",
            "mode": "text_to_image",
            "sourceAssetId": "src_1",
            "strength": 0.25,
            "width": 1024,
            "height": 1024,
            "advanced": {
                "steps": 8,
                "lowResStart": { "version": 1, "k": 5, "scale": 0.5 }
            }
        });
        let top_obj = top_level_payload.as_object().unwrap();
        let resolved_top = requested_low_res_start(
            "flux_dev",
            top_obj,
            top_obj["advanced"].as_object().unwrap(),
            &entry,
        )
        .expect("top-level strength 0.25 with 8 steps and K=5 must be valid")
        .expect("present");
        assert_eq!(resolved_top.executed_steps, 6);
        assert_eq!(resolved_top.k, 5);
    }

    #[test]
    fn sw_7221_use_pid_2k_validates_low_res_start_against_pid_effective_base_dimensions() {
        let entry = flux_dev_entry();
        // 1024x1024 with usePid: true and pidTarget: "2k" reduces effective base to 512x512.
        // At scale 0.25, 512 * 0.25 = 128 < 256 -> refused at validation time!
        let pid_2k_too_small = json!({
            "model": "flux_dev",
            "mode": "text_to_image",
            "width": 1024,
            "height": 1024,
            "advanced": {
                "steps": 8,
                "usePid": true,
                "pidTarget": "2k",
                "lowResStart": { "version": 1, "k": 3, "scale": 0.25 }
            }
        });
        let obj = pid_2k_too_small.as_object().unwrap();
        let err = requested_low_res_start(
            "flux_dev",
            obj,
            obj["advanced"].as_object().unwrap(),
            &entry,
        )
        .expect_err("1024x1024 with usePid=true, pidTarget=2k, scale=0.25 must be refused");
        assert_eq!(err.field, "advanced.lowResStart.scale");
        assert!(
            err.detail.contains("512x512 to 128x128"),
            "unexpected detail: {}",
            err.detail
        );

        // At scale 0.5, 512 * 0.5 = 256 >= 256 -> accepted with 256x256 -> 512x512!
        let pid_2k_valid = json!({
            "model": "flux_dev",
            "mode": "text_to_image",
            "width": 1024,
            "height": 1024,
            "advanced": {
                "steps": 8,
                "usePid": true,
                "pidTarget": "2k",
                "lowResStart": { "version": 1, "k": 3, "scale": 0.5 }
            }
        });
        let valid_obj = pid_2k_valid.as_object().unwrap();
        let resolved = requested_low_res_start(
            "flux_dev",
            valid_obj,
            valid_obj["advanced"].as_object().unwrap(),
            &entry,
        )
        .expect("scale 0.5 on 512x512 PiD 2k base is valid")
        .expect("present");
        assert_eq!((resolved.low_width, resolved.low_height), (256, 256));
        assert_eq!((resolved.full_width, resolved.full_height), (512, 512));
    }
}
