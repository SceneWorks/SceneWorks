//! The generic image lane's request controls beyond the positional sampling knobs (sc-25681, epic
//! 25678): the CFG interval, the caption-overflow policy, the CFG-bound negative prompt, plus the
//! per-image facts a render reports back (non-fatal engine warnings, adapter apply reports).
//!
//! Shared by the MLX `generate_stream` arm and the Candle `generate_candle_stream` arm, so both
//! lanes resolve, refuse and record exactly the same thing. Everything is decided against the
//! LINKED engine descriptor (what the engine honours) and the manifest declarations the API's
//! enqueue gate reads ([`sceneworks_core::image_sampling_controls`]); a control the engine does not
//! honour is a typed refusal before load, never accepted and dropped (E3).

use gen_core::{AdapterApplyReport, AdapterKind, AdapterSpec, Capabilities, CaptionOverflowPolicy};
use sceneworks_core::contracts::JsonObject;
use sceneworks_core::image_request::ImageRequest;
use sceneworks_core::image_sampling_controls::{
    self as controls, CaptionOverflow, CAPTION_OVERFLOW_KEY, CFG_INTERVAL_KEY, H3_IMAGE_SAMPLER_KEY,
};
use serde_json::{json, Value};

use crate::error::{WorkerError, WorkerResult};

/// `rawAdapterSettings` key of the non-fatal engine warnings of one image (e.g. Iris's
/// `caption_truncated`).
pub(crate) const GENERATION_WARNINGS_KEY: &str = "generationWarnings";
/// `rawAdapterSettings` key of the applied adapter stack of one image: identity (path, sha256,
/// scale, kind) joined with the engine's own apply report (applied / skipped).
pub(crate) const ADAPTER_APPLY_KEY: &str = "adapterApply";
/// The per-image prefix-KV-cache record (epic 25899): `{selected, pricedBytes, path, cacheBytes,
/// overflow}` — see `docs/observability.md`. The same key the job result summarises
/// (`super::job_prefix_kv_cache_records`), so the two cannot drift.
pub(crate) const PREFIX_KV_CACHE_KEY: &str = super::PREFIX_KV_CACHE_RESULT_KEY;
/// `rawAdapterSettings` key of the completed Perturbed-Attention Guidance receipt (SW-7180).
pub(crate) const PAG_EXECUTION_REPORT_KEY: &str = sceneworks_core::options::pag::PAG_REPORT_KEY;
/// `rawAdapterSettings` key of the completed Low-Resolution Start execution receipt (SW-6942).
pub(crate) const LOW_RES_START_EXECUTION_REPORT_KEY: &str =
    sceneworks_core::options::low_res_start::LOW_RES_START_REPORT_KEY;

/// The resolved request controls one generic-lane render carries onto its `GenerationRequest`.
/// `Default` is "the engine's own" for every field — what every model that declares none sends.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct LaneRequestControls {
    pub(crate) cfg_interval: Option<(f32, f32)>,
    pub(crate) caption_overflow: Option<CaptionOverflowPolicy>,
    /// MiniMax-H3-Image's sampler block (SW-7106), from `advanced.h3ImageSampler`.
    pub(crate) h3_image_sampler: Option<gen_core::h3_image::H3ImageSampler>,
    /// Perturbed-Attention Guidance policy (SW-7180), from `advanced.pag`.
    pub(crate) pag: Option<gen_core::PagConfig>,
    /// Two-phase Low-Resolution Start trajectory policy (SW-6942), from `advanced.lowResStart`.
    pub(crate) low_res_start: Option<gen_core::LowResStartSpec>,
}

/// Record a provider's completed [`gen_core::PagExecutionReport`] into per-image facts under
/// `pagExecutionReport`. Nothing is written when PAG did not execute (`None`).
pub(crate) fn record_pag_execution_report(
    facts: &mut JsonObject,
    pag: Option<&gen_core::PagExecutionReport>,
) {
    let Some(report) = pag else {
        return;
    };
    let wire = sceneworks_core::options::pag::PagExecutionReportWire::from(report);
    if let Ok(value) = serde_json::to_value(wire) {
        facts.insert(PAG_EXECUTION_REPORT_KEY.to_owned(), value);
    }
}

/// Record a provider's completed [`gen_core::LowResStartExecutionReport`] into per-image facts
/// under `lowResStartExecutionReport` and as a canonical [`sceneworks_core::execution_receipt::LowResStartReceipt`]
/// under `workerLowResStartReceipt`. Nothing is written when `lowResStart` did not execute (`None`).
pub(crate) fn record_low_res_start_execution_report(
    facts: &mut JsonObject,
    low_res_start: Option<&gen_core::LowResStartExecutionReport>,
) {
    let Some(report) = low_res_start else {
        return;
    };
    if let Ok(value) = serde_json::to_value(report) {
        facts.insert(LOW_RES_START_EXECUTION_REPORT_KEY.to_owned(), value);
    }
    let receipt = sceneworks_core::execution_receipt::LowResStartReceipt {
        version: report.version,
        k: u32::try_from(report.k).unwrap_or(u32::MAX),
        executed_steps: u32::try_from(report.executed_steps).unwrap_or(u32::MAX),
        scale: sceneworks_core::execution_receipt::normalize_low_res_start_scale(f64::from(
            report.scale,
        )),
        low_w: report.low_width,
        low_h: report.low_height,
        full_w: report.full_width,
        full_h: report.full_height,
        target_pass: report.estimate_provenance.pass_index,
        estimator: report.estimator.clone(),
        schedule_policy: sceneworks_core::execution_receipt::LOW_RES_START_SCHEDULE_POLICY_V1
            .to_owned(),
        transition_arithmetic: report.transition_arithmetic_id.clone(),
        transition_seed_scheme: report.transition_seed_derivation_id.clone(),
        selected_rung: None,
    };
    if let Ok(receipt_value) = serde_json::to_value(receipt) {
        facts.insert("workerLowResStartReceipt".to_owned(), receipt_value);
    }
}

/// Record the sampler the render ACTUALLY runs under `advanced.h3ImageSampler` (SW-7106), even when
/// the request omitted the block or sent a partial one, so "use this recipe" replays the same
/// image. The key is worker-owned here: a request's own block, already validated, is replaced by
/// the resolved (complete) one. Nothing is written when the engine runs no H3-Image sampler.
pub(crate) fn record_effective_h3_image_sampler(
    raw_settings: &mut JsonObject,
    controls: &LaneRequestControls,
) {
    use gen_core::h3_image::H3ImageSampler;
    let Some(sampler) = controls.h3_image_sampler else {
        return;
    };
    let block = match sampler {
        H3ImageSampler::Euler => json!({ "sampler": "euler" }),
        H3ImageSampler::Lcm {
            s_noise,
            s_noise_end,
            noise_clip_std,
        } => json!({
            "sampler": "lcm",
            "sNoise": s_noise,
            "sNoiseEnd": s_noise_end,
            "noiseClipStd": noise_clip_std,
        }),
    };
    raw_settings.insert(controls::H3_IMAGE_SAMPLER_KEY.to_owned(), block);
}

fn engine_policy(policy: CaptionOverflow) -> CaptionOverflowPolicy {
    match policy {
        CaptionOverflow::Warn => CaptionOverflowPolicy::Warn,
        CaptionOverflow::Error => CaptionOverflowPolicy::Error,
        CaptionOverflow::Silent => CaptionOverflowPolicy::Silent,
    }
}

/// Resolve the request controls against the linked engine's `caps`, the resolved `guidance` and
/// the resolved `negative_prompt`, refusing (typed `InvalidPayload`, before any load) a malformed
/// value, a control the engine does not honour, and a control CFG-off would make inert:
///
/// * `advanced.cfgInterval` — needs `supports_cfg_interval`, and guidance != 1.0;
/// * `advanced.captionOverflow` — needs `supports_caption_overflow_policy`;
/// * `advanced.h3ImageSampler` — validated, and needs `supports_h3_image_sampler`;
/// * `advanced.schedulerShift` — validated (> 0) for a model that declares the shift axis;
/// * a non-empty negative prompt on a model declaring `image.negativePromptRequiresGuidance` at
///   guidance 1.0 (the engine refuses it; sc-25679 dropped it, sc-25681 refuses it like the API).
#[cfg(test)]
pub(crate) fn resolve_lane_request_controls(
    request: &ImageRequest,
    caps: &Capabilities,
    guidance: Option<f32>,
    negative_prompt: Option<&str>,
) -> WorkerResult<LaneRequestControls> {
    resolve_lane_request_controls_with_pid(
        request,
        caps,
        guidance,
        negative_prompt,
        request.portable_aux_has_bound_pid(),
    )
}

pub(crate) fn resolve_lane_request_controls_with_pid(
    request: &ImageRequest,
    caps: &Capabilities,
    guidance: Option<f32>,
    negative_prompt: Option<&str>,
    pid_active: bool,
) -> WorkerResult<LaneRequestControls> {
    let refuse =
        |message: String| WorkerError::InvalidPayload(format!("{}: {message}", request.model));
    let cfg_interval = request
        .resolved_advanced
        .as_ref()
        .expect("image options must be resolved")
        .cfg_interval
        .as_ref()
        .map(|bound| {
            let pair = &bound.value;
            match pair.as_slice() {
                [lo, hi]
                    if lo.is_finite() && hi.is_finite() && 0.0 <= *lo && lo < hi && *hi <= 1.0 =>
                {
                    Ok((*lo as f32, *hi as f32))
                }
                _ => Err(refuse(format!(
                    "advanced.{CFG_INTERVAL_KEY} must satisfy 0 <= start < end <= 1"
                ))),
            }
        })
        .transpose()?;
    let caption_overflow = request
        .advanced_text(CAPTION_OVERFLOW_KEY)
        .map(str::trim)
        .filter(|name| !name.is_empty() && *name != "default")
        .map(|name| {
            CaptionOverflow::from_name(name).ok_or_else(|| {
                refuse(format!(
                    "advanced.{CAPTION_OVERFLOW_KEY} is not a caption-overflow policy"
                ))
            })
        })
        .transpose()?;
    if controls::supports_scheduler_shift(&request.model_manifest_entry)
        && request
            .advanced_f64("schedulerShift")
            .is_some_and(|shift| !shift.is_finite() || shift <= 0.0)
    {
        return Err(refuse(
            "advanced.schedulerShift must be a finite value > 0".to_owned(),
        ));
    }
    let h3_image_sampler =
        controls::requested_h3_image_sampler(&request.advanced).map_err(refuse)?;
    if cfg_interval.is_some() && !caps.supports_cfg_interval {
        return Err(refuse(format!(
            "the engine does not honour a guidance interval (advanced.{CFG_INTERVAL_KEY})"
        )));
    }
    if caption_overflow.is_some() && !caps.supports_caption_overflow_policy {
        return Err(refuse(format!(
            "the engine does not honour a caption-overflow policy (advanced.{CAPTION_OVERFLOW_KEY})"
        )));
    }
    if h3_image_sampler.is_some() && !caps.supports_h3_image_sampler {
        return Err(refuse(format!(
            "the engine does not honour the MiniMax-H3-Image sampler (advanced.{H3_IMAGE_SAMPLER_KEY})"
        )));
    }
    // A request that omits the block (and names no other sampler) runs the MiniMax-H3-Image
    // default, LCM 1.1 / 0.9 / 0, the same default the engine applies. An `advanced.sampler` of
    // `euler` keeps meaning Euler.
    let h3_image_sampler = h3_image_sampler.or_else(|| {
        let named =
            sceneworks_core::image_sampling_controls::authored_sampler_name(&request.advanced)
                .map(str::trim)
                .unwrap_or_default();
        if !caps.supports_h3_image_sampler || !(named.is_empty() || named == "default") {
            return None;
        }
        Some(gen_core::h3_image::H3ImageSampler::Lcm {
            s_noise: controls::H3_IMAGE_DEFAULT_S_NOISE,
            s_noise_end: controls::H3_IMAGE_DEFAULT_S_NOISE_END,
            noise_clip_std: controls::H3_IMAGE_DEFAULT_NOISE_CLIP_STD,
        })
    });
    let cfg_off = controls::cfg_off(guidance.map(f64::from), &request.model_manifest_entry);
    if cfg_off && cfg_interval.is_some() {
        return Err(refuse(format!(
            "a guidance interval has no effect at guidance {:.1} (classifier-free guidance is off)",
            guidance.unwrap_or(1.0)
        )));
    }
    if cfg_off
        && controls::negative_prompt_requires_guidance(&request.model_manifest_entry)
        && negative_prompt.is_some_and(|prompt| !prompt.trim().is_empty())
    {
        return Err(refuse(format!(
            "the negative prompt has no effect at guidance {:.1} (classifier-free guidance is off)",
            guidance.unwrap_or(1.0)
        )));
    }
    let pag = request
        .resolved_pag_with_capability(caps.pag.as_ref())
        .map_err(WorkerError::InvalidPayload)?;
    let default_steps = request
        .model_manifest_entry
        .get("defaults")
        .and_then(|d| d.get("steps"))
        .and_then(Value::as_u64)
        .and_then(|s| u32::try_from(s).ok())
        .unwrap_or(match request.model.as_str() {
            "flux_schnell" | "flux1_schnell" => 4,
            _ => 25,
        });
    let resolved_steps = request
        .advanced_i64("steps")
        .and_then(|s| u32::try_from(s).ok())
        .unwrap_or(default_steps);
    let low_res_start_supported = !caps.supports_phases
        && caps.conditioning != [gen_core::ConditioningKind::Control]
        && request
            .model_manifest_entry
            .get("features")
            .and_then(|f| f.get("low_res_start"))
            .and_then(Value::as_bool)
            .unwrap_or(matches!(
                request.model.as_str(),
                "flux_schnell" | "flux1_schnell" | "flux_dev" | "flux1_dev"
            ));
    let low_res_start = request
        .resolved_low_res_start_spec_with_pid(low_res_start_supported, resolved_steps, pid_active)
        .map_err(WorkerError::InvalidPayload)?;
    Ok(LaneRequestControls {
        cfg_interval,
        caption_overflow: caption_overflow.map(engine_policy),
        h3_image_sampler,
        pag,
        low_res_start,
    })
}

/// A native prompt batch resolved for one job (sc-25681): the prompts ONE engine call renders
/// together (`GenerationRequest::prompt_batch`) and the `count` repetitions of the whole list. The
/// engine returns `prompts × count` images in prompt-minor order, image `k` on seed `seed + k`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NativePromptBatch {
    pub(crate) prompts: Vec<String>,
    pub(crate) count: u32,
}

impl NativePromptBatch {
    /// The images the one engine call returns.
    pub(crate) fn images(&self) -> u32 {
        u32::try_from(self.prompts.len())
            .unwrap_or(u32::MAX)
            .saturating_mul(self.count)
    }
}

/// Resolve the request's native prompt batch against the LINKED engine (`caps`), refusing (typed
/// `InvalidPayload`, before any load) a batch the engine cannot render — no `supports_prompt_batch`,
/// more images than its `max_count`, or a shape the API would refuse (the same core check) — and a
/// batch combined with a lane feature that renders per image (`per_image`: name → in use), which the
/// one-call batch would otherwise silently skip. `None` for an ordinary request.
pub(crate) fn resolve_native_prompt_batch(
    request: &ImageRequest,
    caps: &Capabilities,
    per_image: &[(&str, bool)],
) -> WorkerResult<Option<NativePromptBatch>> {
    let refuse =
        |message: String| WorkerError::InvalidPayload(format!("{}: {message}", request.model));
    if request.prompts.is_empty() {
        // A native COUNT batch (sc-24215, `image.nativeCountBatch`): the engine renders `count`
        // images of the one prompt in a single call (`GenerationRequest::count`), image `k` on
        // seed `seed + k` — Supra2-IMG's published `num_images`. Every other model keeps one
        // engine call per image.
        if !controls::native_count_batch(&request.model_manifest_entry) {
            return Ok(None);
        }
        if request.count > caps.max_count {
            return Err(refuse(format!(
                "{} images exceed the {} the engine renders per request",
                request.count, caps.max_count
            )));
        }
        if let Some((feature, _)) = per_image.iter().find(|(_, in_use)| *in_use) {
            return Err(refuse(format!(
                "{feature} renders per image and is not supported by this model"
            )));
        }
        return Ok(Some(NativePromptBatch {
            prompts: vec![request.prompt.clone()],
            count: request.count,
        }));
    }
    if !caps.supports_prompt_batch {
        return Err(refuse(
            "the engine does not render a prompt batch in one pass; send one job per prompt"
                .to_owned(),
        ));
    }
    if let Some(message) = controls::prompt_batch_error(
        &request.model,
        &request.prompt,
        &request.prompts,
        request.count,
        &request.model_manifest_entry,
    ) {
        return Err(WorkerError::InvalidPayload(message));
    }
    let batch = NativePromptBatch {
        prompts: request.prompts.clone(),
        count: request.count,
    };
    if batch.images() > caps.max_count {
        return Err(refuse(format!(
            "a prompt batch of {} images exceeds the {} the engine renders per request",
            batch.images(),
            caps.max_count
        )));
    }
    if let Some((feature, _)) = per_image.iter().find(|(_, in_use)| *in_use) {
        return Err(refuse(format!(
            "{feature} renders per image and cannot ride a prompt batch; send one job per prompt"
        )));
    }
    Ok(Some(batch))
}

/// The output area, in pixels, the declared single-image Candle floors carry: 1024² = 1 MP. The
/// declared activation overlay charges every megapixel (and every image) beyond it.
pub(crate) const DECLARED_ACTIVATION_FLOOR_PIXELS: u64 = 1024 * 1024;

/// `candle.imageActivationGbPerMegapixel` when the entry DECLARES it — the term
/// [`declared_activation_overlay_bytes`] charges. An entry that declares it has every image and
/// megapixel beyond one image at [`DECLARED_ACTIVATION_FLOOR_PIXELS`] priced by that overlay, so no
/// other estimator may scale the same activations by batch or area again
/// (`candle_memory_strategy::law_geometry`).
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub(crate) fn declared_activation_gb_per_megapixel(manifest: &JsonObject) -> Option<f64> {
    manifest
        .get("candle")
        .and_then(|candle| candle.get("imageActivationGbPerMegapixel"))
        .and_then(Value::as_f64)
        .filter(|gb| gb.is_finite() && *gb > 0.0)
}

/// The DECLARED pixel-space activation overlay of a Candle image request, in bytes, charged on top of
/// the single-image floors (`candle.minMemoryGb` / `sequentialPeakGb`, which carry ONE image at
/// <= 1024²): `candle.imageActivationGbPerMegapixel × (images × max(1, width·height / 1024²) − 1)`.
///
/// One term covers both ways a request renders more pixels than the floors were declared at — the
/// extra images of a native prompt batch (sc-25681) and an output above 1024² (sc-25684: Iris allows
/// 2048², whose activations are 4× the 1024² ones; both engines run fused attention, so the transient
/// is linear in tokens). The area never shrinks below 1024² (the floors were declared there). `0` for
/// a single image at <= 1 MP, and for an entry that declares no term on an ordinary request (its
/// legacy floors stand). A batch on an entry that declares no term is refused before load rather than
/// admitted on a floor sized for one image.
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub(crate) fn declared_activation_overlay_bytes(
    request: &ImageRequest,
    batch: Option<&NativePromptBatch>,
    width: u32,
    height: u32,
) -> WorkerResult<u64> {
    let images = batch.map_or(1, NativePromptBatch::images);
    let megapixels =
        (f64::from(width) * f64::from(height) / DECLARED_ACTIVATION_FLOOR_PIXELS as f64).max(1.0);
    let units = f64::from(images) * megapixels - 1.0;
    if units <= 0.0 {
        return Ok(0);
    }
    let per_megapixel_gb = declared_activation_gb_per_megapixel(&request.model_manifest_entry);
    match per_megapixel_gb {
        Some(gb) => Ok((units * gb * crate::fit_gate::BYTES_PER_GIB).round() as u64),
        None if batch.is_some() => Err(WorkerError::InvalidPayload(format!(
            "{}: no declared per-image memory for a prompt batch \
             (candle.imageActivationGbPerMegapixel); send one job per prompt",
            request.model
        ))),
        None => Ok(0),
    }
}

/// One adapter's identity, resolved once per job before load: what was requested, from where,
/// at what strength, and the SHA-256 of the exact file the engine loads.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AdapterIdentity {
    pub(crate) id: Option<String>,
    pub(crate) path: std::path::PathBuf,
    pub(crate) sha256: Option<String>,
    pub(crate) scale: f32,
    pub(crate) kind: AdapterKind,
}

/// The identities of `adapters` (the resolved load-order stack), labelled with the request LoRA
/// ids they came from (`request.loras` is the same order), resolved ONCE per job before load.
///
/// The digest comes from [`super::trusted_lora_hash`] — the same path workflow attribution uses:
/// the install marker's cached `loraFileSha256` when it still describes this exact file (no
/// re-hash), otherwise a SHA-256 on a blocking task with the worker heartbeat kept alive (never on
/// the async executor thread), cached back into the marker. A hashing failure leaves `sha256`
/// `None` — attribution, not generation, fails.
pub(crate) async fn adapter_identities(
    api: &crate::api_client::ApiClient,
    settings: &crate::settings::Settings,
    job: &sceneworks_core::contracts::JobSnapshot,
    request: &ImageRequest,
    adapters: &[AdapterSpec],
) -> Vec<AdapterIdentity> {
    let mut identities = Vec::with_capacity(adapters.len());
    for (index, adapter) in adapters.iter().enumerate() {
        let lora = request.loras.get(index).cloned().unwrap_or(Value::Null);
        let sha256 = super::trusted_lora_hash(api, settings, job, &lora, &adapter.path).await;
        identities.push(AdapterIdentity {
            id: lora
                .as_str()
                .or_else(|| lora.get("id").and_then(Value::as_str))
                .or_else(|| lora.get("loraId").and_then(Value::as_str))
                .map(str::to_owned),
            path: adapter.path.clone(),
            sha256,
            scale: adapter.scale,
            kind: adapter.kind,
        });
    }
    identities
}

/// The per-image facts block a render reports into its asset's `rawAdapterSettings`: the engine's
/// non-fatal warnings and, when adapters were loaded, each one's identity joined (by path) with the
/// engine's apply report. `None` when there is nothing to record, so a plain render's settings are
/// unchanged.
pub(crate) fn image_facts(
    warnings: &[gen_core::GenerationWarning],
    identities: &[AdapterIdentity],
    reports: &[AdapterApplyReport],
    prefix_kv_cache: Option<Value>,
) -> Option<JsonObject> {
    let mut facts = JsonObject::new();
    // Epic 25899: what the admission selected for the provider's prefix KV cache and what the
    // render reports it ran (`execution_planner::prefix_kv_cache_record`). Absent for every
    // provider without a cache.
    if let Some(record) = prefix_kv_cache {
        facts.insert(PREFIX_KV_CACHE_KEY.to_owned(), record);
    }
    if !warnings.is_empty() {
        facts.insert(
            GENERATION_WARNINGS_KEY.to_owned(),
            Value::Array(
                warnings
                    .iter()
                    .map(|warning| json!({ "code": warning.code, "message": warning.message }))
                    .collect(),
            ),
        );
    }
    if !identities.is_empty() {
        facts.insert(
            ADAPTER_APPLY_KEY.to_owned(),
            Value::Array(
                identities
                    .iter()
                    .map(|identity| {
                        let report = reports
                            .iter()
                            .find(|report| report.adapter_path == identity.path);
                        json!({
                            "id": identity.id,
                            "path": identity.path.display().to_string(),
                            "sha256": identity.sha256,
                            "scale": identity.scale,
                            "kind": match identity.kind {
                                AdapterKind::Lora => "lora",
                                AdapterKind::Lokr => "lokr",
                            },
                            // `null` when the engine publishes no report for this file (providers
                            // without a partial-install surface) — never a guessed count.
                            "applied": report.map(|report| report.applied),
                            "skipped": report.map(|report| report.skipped.clone()),
                        })
                    })
                    .collect(),
            ),
        );
    }
    (!facts.is_empty()).then_some(facts)
}

/// The job's distinct warning messages across its written assets, for the completion message.
pub(crate) fn job_warning_messages(asset_writes: &[Value]) -> Vec<String> {
    let mut messages: Vec<String> = Vec::new();
    for write in asset_writes {
        let warnings = write
            .get("rawAdapterSettings")
            .and_then(|settings| settings.get(GENERATION_WARNINGS_KEY))
            .and_then(Value::as_array);
        for warning in warnings.into_iter().flatten() {
            if let Some(message) = warning.get("message").and_then(Value::as_str) {
                if !messages.iter().any(|seen| seen == message) {
                    messages.push(message.to_owned());
                }
            }
        }
    }
    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(advanced: Value, negative: &str, image: Value) -> ImageRequest {
        let payload = json!({
            "projectId": "p",
            "model": "m",
            "prompt": "x",
            "negativePrompt": negative,
            "advanced": advanced,
            "modelManifestEntry": { "id": "m", "image": image },
        });
        let payload = payload.as_object().unwrap();
        ImageRequest::from_resolved(payload, payload).unwrap()
    }

    fn sized_request(candle: Value) -> ImageRequest {
        let payload = json!({
            "projectId": "p",
            "model": "iris_3b",
            "prompt": "x",
            "modelManifestEntry": { "id": "iris_3b", "candle": candle },
        });
        let payload = payload.as_object().unwrap();
        ImageRequest::from_resolved(payload, payload).unwrap()
    }

    /// sc-25684: one declared per-megapixel term charges every image-megapixel beyond the one 1024²
    /// image the floors carry — the extra images of a batch AND the extra area of an output above
    /// 1024² (and of every batched image at that size). Below 1024² nothing is credited back.
    ///
    /// *Mutations that red this:* the sc-25681 formula `(images − 1) × gb × max(1, MP)` (charges
    /// 2048² single images nothing and a 2-image 2048² batch 24 instead of 42 GB); dropping the
    /// `.max(1.0)` (a 512² batch of two would be charged 0.5 instead of 6).
    #[test]
    fn the_declared_overlay_scales_with_images_and_area() {
        let gib = |gb: f64| (gb * crate::fit_gate::BYTES_PER_GIB).round() as u64;
        let declared = sized_request(json!({ "imageActivationGbPerMegapixel": 6.0 }));
        let batch = |prompts: usize, count: u32| NativePromptBatch {
            prompts: (0..prompts).map(|i| format!("p{i}")).collect(),
            count,
        };
        let overlay = |b: Option<&NativePromptBatch>, w: u32, h: u32| {
            declared_activation_overlay_bytes(&declared, b, w, h).unwrap()
        };
        assert_eq!(
            overlay(None, 1024, 1024),
            0,
            "the floors carry one 1024² image"
        );
        assert_eq!(
            overlay(None, 1344, 768),
            0,
            "under 1 MP is inside the floor"
        );
        assert_eq!(
            overlay(None, 512, 512),
            0,
            "nothing below 1024² is credited back"
        );
        assert_eq!(overlay(None, 2048, 2048), gib(18.0), "4 MP: three extra MP");
        assert_eq!(overlay(None, 2048, 1024), gib(6.0), "2 MP: one extra MP");
        assert_eq!(overlay(Some(&batch(2, 1)), 512, 512), gib(6.0));
        assert_eq!(overlay(Some(&batch(2, 4)), 1024, 1024), gib(42.0));
        assert_eq!(
            overlay(Some(&batch(2, 1)), 2048, 2048),
            gib(42.0),
            "two 4 MP images are eight image-MP, seven beyond the floor"
        );

        // An entry that declares no term keeps its legacy floors for one image at any size, but a
        // batch on it is refused rather than admitted on a one-image floor.
        let undeclared = sized_request(json!({ "minMemoryGb": 24 }));
        assert_eq!(
            declared_activation_overlay_bytes(&undeclared, None, 2048, 2048).unwrap(),
            0
        );
        let error = declared_activation_overlay_bytes(&undeclared, Some(&batch(2, 1)), 1024, 1024)
            .unwrap_err();
        assert!(
            error.to_string().contains("imageActivationGbPerMegapixel"),
            "{error}"
        );
    }

    /// sc-24215: a model declaring `image.nativeCountBatch` renders `count` images of its one
    /// prompt in ONE engine call; the engine's `max_count` bounds it and a per-image lane feature is
    /// refused rather than silently skipped. Undeclared ⇒ the per-image path, unchanged.
    #[test]
    fn a_declared_native_count_batch_is_one_call_over_the_one_prompt() {
        let with = |image: Value, count: u32| {
            let payload = json!({
                "projectId": "p", "model": "supra2_img", "prompt": "a fox", "count": count,
                "modelManifestEntry": { "id": "supra2_img", "image": image },
            });
            ImageRequest::from_payload(payload.as_object().unwrap())
        };
        let caps = |max_count: u32| Capabilities {
            max_count,
            ..Default::default()
        };
        let declared = json!({ "nativeCountBatch": true });
        let batch = resolve_native_prompt_batch(&with(declared.clone(), 4), &caps(64), &[])
            .unwrap()
            .expect("declared ⇒ one call");
        assert_eq!(batch.prompts, vec!["a fox".to_owned()]);
        assert_eq!((batch.count, batch.images()), (4, 4));
        assert_eq!(
            resolve_native_prompt_batch(&with(json!({}), 4), &caps(64), &[]).unwrap(),
            None
        );
        let over = resolve_native_prompt_batch(&with(declared.clone(), 4), &caps(2), &[])
            .unwrap_err()
            .to_string();
        assert!(over.contains("4 images exceed the 2"), "{over}");
        let per_image =
            resolve_native_prompt_batch(&with(declared, 2), &caps(64), &[("hires fix", true)])
                .unwrap_err()
                .to_string();
        assert!(per_image.contains("hires fix"), "{per_image}");
    }

    fn honouring() -> Capabilities {
        Capabilities {
            supports_cfg_interval: true,
            supports_caption_overflow_policy: true,
            ..Default::default()
        }
    }

    #[test]
    fn controls_reach_the_request_when_the_engine_honours_them() {
        let request = request(
            json!({ "cfgInterval": [0.25, 0.75], "captionOverflow": "error" }),
            "",
            json!({}),
        );
        let resolved =
            resolve_lane_request_controls(&request, &honouring(), Some(3.0), None).unwrap();
        assert_eq!(
            resolved,
            LaneRequestControls {
                cfg_interval: Some((0.25, 0.75)),
                caption_overflow: Some(CaptionOverflowPolicy::Error),
                h3_image_sampler: None,
                pag: None,
                low_res_start: None,
            }
        );
        let none = resolve_lane_request_controls(
            &self::request(json!({}), "", json!({})),
            &Capabilities::default(),
            Some(3.0),
            None,
        )
        .unwrap();
        assert_eq!(none, LaneRequestControls::default());
    }

    #[test]
    fn an_unhonoured_or_inert_control_is_refused_not_dropped() {
        let cases = [
            // (advanced, negative, image decl, caps honour, guidance, message fragment)
            (
                json!({ "cfgInterval": [0.2, 0.8] }),
                "",
                json!({}),
                false,
                Some(3.0),
                "guidance interval",
            ),
            (
                json!({ "captionOverflow": "warn" }),
                "",
                json!({}),
                false,
                Some(3.0),
                "caption-overflow",
            ),
            (
                json!({ "cfgInterval": [0.2, 0.8] }),
                "",
                json!({}),
                true,
                Some(1.0),
                "guidance 1.0",
            ),
            (
                json!({}),
                "blurry",
                json!({ "negativePromptRequiresGuidance": true }),
                true,
                Some(1.0),
                "negative prompt",
            ),
            (
                json!({ "cfgInterval": [0.8, 0.2] }),
                "",
                json!({}),
                true,
                Some(3.0),
                "0 <= start",
            ),
            (
                json!({ "captionOverflow": "shout" }),
                "",
                json!({}),
                true,
                Some(3.0),
                "not a caption",
            ),
            (
                json!({ "schedulerShift": 0 }),
                "",
                json!({ "supportsSchedulerShift": true }),
                true,
                Some(3.0),
                "> 0",
            ),
        ];
        for (advanced, negative, image, honour, guidance, fragment) in cases {
            let request = request(advanced.clone(), negative, image);
            let caps = if honour {
                honouring()
            } else {
                Capabilities::default()
            };
            let negative = (!negative.is_empty()).then_some(negative);
            match resolve_lane_request_controls(&request, &caps, guidance, negative) {
                Err(WorkerError::InvalidPayload(message)) => {
                    assert!(message.contains(fragment), "{advanced}: {message}")
                }
                other => panic!("{advanced}: expected a typed refusal, got {other:?}"),
            }
        }
        // The undeclared negative-prompt binding never refuses.
        let request = request(json!({}), "blurry", json!({}));
        assert!(
            resolve_lane_request_controls(&request, &honouring(), Some(1.0), Some("blurry"))
                .is_ok()
        );
    }

    /// SW-7106: the asset records the sampler that ran, including when the request omitted the
    /// block (it runs LCM 1.1 / 0.9 / 0) or sent a partial one, so a replay reproduces the image.
    #[test]
    fn the_effective_h3_image_sampler_is_recorded_even_when_the_request_omitted_it() {
        let h3 = Capabilities {
            supports_h3_image_sampler: true,
            ..Default::default()
        };
        let recorded = |advanced: Value, caps: &Capabilities| {
            let request = request(advanced, "", json!({}));
            let controls = resolve_lane_request_controls(&request, caps, None, None).unwrap();
            let mut raw = request.advanced.clone();
            record_effective_h3_image_sampler(&mut raw, &controls);
            raw.get("h3ImageSampler").cloned()
        };
        let lcm_default = json!({ "sampler": "lcm", "sNoise": 1.1f32, "sNoiseEnd": 0.9f32, "noiseClipStd": 0.0f32 });
        assert_eq!(recorded(json!({}), &h3), Some(lcm_default.clone()));
        assert_eq!(
            recorded(json!({ "h3ImageSampler": { "sampler": "lcm" } }), &h3),
            Some(lcm_default)
        );
        assert_eq!(
            recorded(json!({ "h3ImageSampler": { "sampler": "euler" } }), &h3),
            Some(json!({ "sampler": "euler" }))
        );
        // An engine without the sampler, or a request naming Euler through `advanced.sampler`,
        // runs no H3-Image sampler, so nothing is recorded.
        assert_eq!(recorded(json!({}), &Capabilities::default()), None);
        assert_eq!(recorded(json!({ "sampler": "euler" }), &h3), None);
    }

    /// SW-7106: `advanced.h3ImageSampler` maps onto `GenerationRequest::h3_image_sampler` for an
    /// engine that honours it, omitted LCM parameters take the engine defaults (1.1 / 0.9 / 0), and an invalid
    /// value or an engine without the sampler is a typed refusal before load.
    #[test]
    fn the_h3_image_sampler_block_maps_onto_the_engine_sampler() {
        use gen_core::h3_image::H3ImageSampler;
        let h3 = Capabilities {
            supports_h3_image_sampler: true,
            ..Default::default()
        };
        let resolve = |advanced: Value, caps: &Capabilities| {
            resolve_lane_request_controls(&request(advanced, "", json!({})), caps, None, None)
                .map(|controls| controls.h3_image_sampler)
        };
        // An omitted block runs the H3-Image default (LCM 1.1 / 0.9 / 0) on an honouring engine,
        // and nothing on an engine that does not offer the sampler.
        assert_eq!(
            resolve(json!({}), &h3).unwrap(),
            Some(H3ImageSampler::Lcm {
                s_noise: 1.1,
                s_noise_end: 0.9,
                noise_clip_std: 0.0,
            })
        );
        assert_eq!(resolve(json!({}), &Capabilities::default()).unwrap(), None);
        assert_eq!(
            resolve(json!({ "sampler": "euler" }), &h3).unwrap(),
            None,
            "a request that names Euler through advanced.sampler stays Euler"
        );
        assert_eq!(
            resolve(json!({ "h3ImageSampler": { "sampler": "euler" } }), &h3).unwrap(),
            Some(H3ImageSampler::Euler)
        );
        assert_eq!(
            resolve(
                json!({ "h3ImageSampler": {
                    "sampler": "lcm", "sNoise": 1.1, "sNoiseEnd": 0.9, "noiseClipStd": 2.5
                } }),
                &h3
            )
            .unwrap(),
            Some(H3ImageSampler::Lcm {
                s_noise: 1.1,
                s_noise_end: 0.9,
                noise_clip_std: 2.5,
            })
        );
        assert_eq!(
            resolve(json!({ "h3ImageSampler": { "sampler": "lcm" } }), &h3).unwrap(),
            Some(H3ImageSampler::lcm_default())
        );
        for (advanced, fragment) in [
            (
                json!({ "h3ImageSampler": { "sampler": "lcm", "sNoise": 0 } }),
                "sNoise 0 must be greater than 0",
            ),
            (
                json!({ "h3ImageSampler": { "sampler": "lcm", "sNoiseEnd": -0.1 } }),
                "sNoiseEnd -0.1 must be 0 or greater",
            ),
            (
                json!({ "h3ImageSampler": { "sampler": "lcm", "noiseClipStd": -1 } }),
                "noiseClipStd -1 must be 0 (off) or greater",
            ),
            (
                json!({ "h3ImageSampler": { "sampler": "lcm", "sNoise": 1e300 } }),
                "must be finite",
            ),
            (
                json!({ "h3ImageSampler": { "sampler": "dpm" } }),
                "not a MiniMax",
            ),
            (
                json!({ "h3ImageSampler": { "sampler": "euler", "sNoise": 1.1 } }),
                "only to the LCM",
            ),
            (
                json!({ "sampler": "euler", "h3ImageSampler": { "sampler": "lcm" } }),
                "disagrees",
            ),
        ] {
            match resolve(advanced.clone(), &h3) {
                Err(WorkerError::InvalidPayload(message)) => {
                    assert!(message.contains(fragment), "{advanced}: {message}")
                }
                other => panic!("{advanced}: expected a typed refusal, got {other:?}"),
            }
        }
        match resolve(
            json!({ "h3ImageSampler": { "sampler": "lcm" } }),
            &Capabilities::default(),
        ) {
            Err(WorkerError::InvalidPayload(message)) => {
                assert!(message.contains("does not honour"), "{message}")
            }
            other => panic!("expected a typed refusal, got {other:?}"),
        }
    }

    #[test]
    fn lane_cfg_off_matches_the_declared_provider_value() {
        let request = request(
            json!({}),
            "blurry",
            json!({"negativePromptRequiresGuidance":true,"guidanceOffValue":0}),
        );
        assert!(
            resolve_lane_request_controls(&request, &honouring(), Some(1.0), Some("blurry"))
                .is_ok()
        );
        let error =
            resolve_lane_request_controls(&request, &honouring(), Some(0.0), Some("blurry"))
                .unwrap_err();
        assert!(error.to_string().contains("guidance 0.0"));
    }

    #[test]
    fn image_facts_join_identity_with_the_engine_report_by_path() {
        let identities = vec![
            AdapterIdentity {
                id: Some("style".into()),
                path: "/a/style.safetensors".into(),
                sha256: Some("ab".into()),
                scale: 0.8,
                kind: AdapterKind::Lora,
            },
            AdapterIdentity {
                id: Some("lokr".into()),
                path: "/a/lokr.safetensors".into(),
                sha256: None,
                scale: 1.0,
                kind: AdapterKind::Lokr,
            },
        ];
        let reports = vec![AdapterApplyReport {
            adapter_path: "/a/style.safetensors".into(),
            applied: 12,
            skipped: vec!["blocks.3.attn".into()],
        }];
        let warnings = vec![gen_core::GenerationWarning {
            code: "caption_truncated".into(),
            message: "dropped 4 tokens".into(),
        }];
        let facts = image_facts(&warnings, &identities, &reports, None).unwrap();
        assert_eq!(
            Value::Object(facts.clone()),
            json!({
                "generationWarnings": [{ "code": "caption_truncated", "message": "dropped 4 tokens" }],
                "adapterApply": [
                    { "id": "style", "path": "/a/style.safetensors", "sha256": "ab", "scale": 0.800000011920929,
                      "kind": "lora", "applied": 12, "skipped": ["blocks.3.attn"] },
                    { "id": "lokr", "path": "/a/lokr.safetensors", "sha256": null, "scale": 1.0,
                      "kind": "lokr", "applied": null, "skipped": null },
                ],
            })
        );
        assert_eq!(image_facts(&[], &[], &reports, None), None);
        let writes = vec![
            json!({ "rawAdapterSettings": Value::Object(facts.clone()) }),
            json!({ "rawAdapterSettings": Value::Object(facts) }),
            json!({ "rawAdapterSettings": {} }),
        ];
        assert_eq!(
            job_warning_messages(&writes),
            vec!["dropped 4 tokens".to_owned()]
        );
    }

    /// Epic 25899: the render's prefix-KV-cache record rides the image's provenance facts, and the
    /// job result summarises exactly those records in asset order — so a finished job says which
    /// path ran (and any overflow) without opening an asset, and a job with no cache says nothing.
    ///
    /// *Mutation that reds this:* dropping the `prefix_kv_cache` arm of `image_facts`, or reading
    /// another key in `job_prefix_kv_cache_records`.
    #[test]
    fn the_prefix_kv_cache_record_reaches_the_image_facts_and_the_job_result() {
        let cached = json!({ "selected": "cached", "pricedBytes": 10, "path": "cached",
                             "cacheBytes": 8, "overflow": null });
        let shed = json!({ "selected": "cached", "pricedBytes": 10, "path": "prefill",
                           "cacheBytes": 0, "overflow": { "actualBytes": 12, "pricedBytes": 10 } });
        let facts = |record: &Value| {
            Value::Object(image_facts(&[], &[], &[], Some(record.clone())).expect("recorded"))
        };
        assert_eq!(facts(&cached)[PREFIX_KV_CACHE_KEY], cached);
        let writes = vec![
            json!({ "rawAdapterSettings": facts(&cached) }),
            json!({ "rawAdapterSettings": {} }),
            json!({ "rawAdapterSettings": facts(&shed) }),
        ];
        assert_eq!(
            super::super::job_prefix_kv_cache_records(&writes),
            vec![cached, shed]
        );
        assert!(
            super::super::job_prefix_kv_cache_records(&[json!({ "rawAdapterSettings": {} })])
                .is_empty()
        );
    }

    async fn identities_for(loras: Vec<Value>, adapters: &[AdapterSpec]) -> Vec<AdapterIdentity> {
        let settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
        let api = crate::api_client::ApiClient::new(&settings);
        let job: sceneworks_core::contracts::JobSnapshot =
            serde_json::from_value(crate::tests::job_snapshot_json("job-hash", false)).unwrap();
        let mut request = request(json!({}), "", json!({}));
        request.loras = loras;
        adapter_identities(&api, &settings, &job, &request, adapters).await
    }

    /// No install marker: the loaded file is hashed (off the executor) and labelled with its
    /// request id.
    #[tokio::test]
    async fn adapter_identities_hash_the_loaded_file_and_keep_the_request_id() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.safetensors");
        std::fs::write(&file, b"abc").unwrap();
        let identities = identities_for(
            vec![json!({ "id": "style-a", "weight": 0.7 })],
            &[AdapterSpec::new(file.clone(), 0.7, AdapterKind::Lora)],
        )
        .await;
        assert_eq!(identities.len(), 1);
        assert_eq!(identities[0].id.as_deref(), Some("style-a"));
        assert_eq!(
            identities[0].sha256.as_deref(),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
        assert_eq!(identities[0].path, file);
    }

    /// An install marker that still describes the exact file supplies its cached digest — the file
    /// is NOT re-hashed (the marker's value is deliberately not the bytes' real digest).
    ///
    /// *Mutation that reds this:* hash the file unconditionally (bypass `trusted_lora_hash`).
    #[tokio::test]
    async fn adapter_identities_reuse_the_install_markers_cached_digest() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("cached.safetensors");
        std::fs::write(&file, b"abc").unwrap();
        let identity = crate::imports::model_file_identity(&file).unwrap();
        let cached = "f".repeat(64);
        std::fs::write(
            dir.path().join(crate::INSTALL_MARKER),
            json!({
                "loraFileName": identity.name,
                "loraFileBytes": identity.bytes,
                "loraFileModifiedNanos": identity.modified_nanos,
                "loraFileSha256": cached,
            })
            .to_string(),
        )
        .unwrap();
        let identities = identities_for(
            vec![json!({ "id": "cached" })],
            &[AdapterSpec::new(file, 1.0, AdapterKind::Lora)],
        )
        .await;
        assert_eq!(identities[0].sha256.as_deref(), Some(cached.as_str()));
    }

    #[test]
    fn pag_controls_resolve_against_descriptor_capability_and_record_execution_receipt() {
        let mut caps = honouring();
        caps.pag = Some(gen_core::PagCapability {
            num_blocks: 8,
            block_domain: std::borrow::Cow::Borrowed("joint_blocks"),
            default_layers: vec![2, 5],
            default_start: 0.05,
            default_stop: 0.85,
            recommended_scale: 1.5,
            max_scale: 10.0,
        });
        let manifest_with_pag = json!({
            "id": "flux2_dev",
            "type": "image",
            "pag": {
                "numBlocks": 8,
                "blockDomain": "joint_blocks",
                "defaultLayers": [2, 5],
                "defaultStart": 0.05,
                "defaultStop": 0.85,
                "recommendedScale": 1.5,
                "maxScale": 10.0
            }
        });
        let req = request(
            json!({ "pag": { "scale": 1.5, "layers": [2, 7] } }),
            "",
            manifest_with_pag.clone(),
        );
        let resolved = resolve_lane_request_controls(&req, &caps, Some(4.0), None).unwrap();
        assert_eq!(
            resolved.pag,
            Some(gen_core::PagConfig {
                scale: 1.5,
                layers: Some(vec![2, 7]),
                start: None,
                stop: None,
                adaptive_scale: None,
            })
        );

        // Provider without PAG capability refuses active PAG
        let unsupported = resolve_lane_request_controls(&req, &honouring(), Some(4.0), None)
            .expect_err("provider without pag capability must refuse");
        assert!(unsupported
            .to_string()
            .contains("Perturbed-Attention Guidance"));

        // Out-of-bounds layer refused against live descriptor
        let mut tighter_caps = caps.clone();
        tighter_caps.pag.as_mut().unwrap().num_blocks = 6;
        let out_of_bounds =
            resolve_lane_request_controls(&req, &tighter_caps, Some(4.0), None).unwrap_err();
        assert!(out_of_bounds.to_string().contains("pag.layers"));

        // Execution receipt records camelCase report and omits inactive report
        let mut facts = crate::JsonObject::new();
        record_pag_execution_report(
            &mut facts,
            Some(&gen_core::PagExecutionReport {
                scale: 1.5,
                adaptive_scale: 0.25,
                layers: vec![2, 7],
                start: 0.05,
                stop: 0.85,
                active_steps: 18,
                total_steps: 28,
            }),
        );
        assert_eq!(
            facts.get(PAG_EXECUTION_REPORT_KEY),
            Some(&json!({
                "scale": 1.5,
                "adaptiveScale": 0.25,
                "layers": [2, 7],
                "start": 0.05,
                "stop": 0.85,
                "activeSteps": 18,
                "totalSteps": 28
            }))
        );
    }

    #[test]
    fn sw_6942_low_res_start_controls_resolve_against_descriptor_capability_and_record_execution_receipt(
    ) {
        let caps = honouring();
        let payload = json!({
            "projectId": "p",
            "model": "flux_dev",
            "prompt": "a tower",
            "width": 1024,
            "height": 1024,
            "advanced": {
                "steps": 25,
                "lowResStart": { "version": 1, "k": 8, "scale": 0.5 }
            },
            "modelManifestEntry": {
                "id": "flux_dev",
                "type": "image",
                "features": { "low_res_start": true }
            }
        });
        let obj = payload.as_object().unwrap();
        let req = ImageRequest::from_resolved(obj, obj).unwrap();
        let resolved = resolve_lane_request_controls(&req, &caps, Some(3.5), None).unwrap();
        assert_eq!(
            resolved.low_res_start,
            Some(gen_core::LowResStartSpec::new(8, Some(0.5)))
        );

        // Provider whose capabilities conflict with `low_res_start` (e.g. `supports_phases`) refuses before load:
        let mut multiphase_caps = honouring();
        multiphase_caps.supports_phases = true;
        let unsupported = resolve_lane_request_controls(&req, &multiphase_caps, Some(3.5), None)
            .expect_err("provider with conflicting supports_phases must refuse");
        assert!(
            unsupported.to_string().contains("Low Res Start")
                || unsupported.to_string().contains("lowResStart"),
            "{unsupported}"
        );

        // Bespoke conditioned route rejection helper refuses `lowResStart` before load:
        let bespoke_refusal = req
            .reject_low_res_start("FLUX.1-dev strict-control")
            .unwrap_err();
        assert!(
            bespoke_refusal.contains("FLUX.1-dev strict-control"),
            "{bespoke_refusal}"
        );

        // Execution receipt records report and omits inactive report:
        let mut facts = crate::JsonObject::new();
        record_low_res_start_execution_report(&mut facts, None);
        assert!(!facts.contains_key(LOW_RES_START_EXECUTION_REPORT_KEY));
        let resolved_plan = gen_core::LowResStartSpec::new(8, Some(0.5))
            .with_estimate_source(gen_core::EstimateSource::PreviousEvaluation)
            .resolve_geometry("flux1_dev", 1024, 1024, 25, 8, 2, 16, 256)
            .unwrap();
        let sigmas: Vec<f32> = (0..=25).map(|i| 1.0 - (i as f32 / 25.0)).collect();
        let report = resolved_plan
            .execution_report(
                0,
                &sigmas,
                gen_core::sampling::Solver::Euler,
                "embedded_guidance",
                3.5,
                "base_dit",
                "bf16",
                2,
                1,
                16,
                64,
            )
            .unwrap();
        record_low_res_start_execution_report(&mut facts, Some(&report));
        let recorded = facts
            .get(LOW_RES_START_EXECUTION_REPORT_KEY)
            .expect("execution report recorded");
        assert_eq!(recorded["recipe_id"], json!("low_res_start"));
        assert_eq!(recorded["version"], json!(1));
        assert_eq!(recorded["k"], json!(8));
        assert_eq!(recorded["executed_steps"], json!(25));
        assert_eq!(recorded["low_width"], json!(512));
        assert_eq!(recorded["low_height"], json!(512));
        assert_eq!(recorded["full_width"], json!(1024));
        assert_eq!(recorded["full_height"], json!(1024));
        assert_eq!(recorded["estimate_source"], json!("previous_evaluation"));
        assert_eq!(recorded["estimator"], json!("previous_evaluation_v1"));
    }

    #[test]
    fn sw_7220_top_level_strength_shares_resolution_across_validation_image_request_and_worker_img2img(
    ) {
        let data = tempfile::tempdir().expect("data dir");
        let settings = crate::Settings::for_test(data.path().to_path_buf());
        let store = crate::ProjectStore::new(settings.data_dir.clone(), "worker");
        let project = store
            .create_project("SW-7220 top-level strength")
            .expect("project");
        let project_path = std::path::PathBuf::from(&project.path);
        let source_file = data.path().join("flux-source.png");
        image::RgbImage::from_pixel(64, 64, image::Rgb([120, 80, 200]))
            .save(&source_file)
            .expect("source PNG");
        let asset = store
            .import_asset(
                &project.id,
                sceneworks_core::project_store::UploadAsset {
                    filename: "flux-source.png".to_owned(),
                    content_type: Some("image/png".to_owned()),
                    source_path: source_file,
                    source_asset_id: None,
                    provenance: None,
                },
            )
            .expect("source asset");
        let asset_id = asset["id"].as_str().expect("asset id");

        let entry = json!({
            "id": "flux_dev",
            "family": "flux",
            "type": "image",
            "defaults": {
                "steps": 28,
                "resolution": "1024x1024"
            },
            "features": {
                "low_res_start": true
            }
        });
        let entry_obj = entry.as_object().unwrap();

        // Submission payload with ONLY top-level `strength: 0.25`, 8 steps, and K=5:
        // init_time_step = floor(8 * 0.25) = 2 -> executed_steps = 8 - 2 = 6 -> K=5 is valid (1 <= 5 < 6).
        let payload = json!({
            "projectId": project.id,
            "model": "flux_dev",
            "mode": "text_to_image",
            "prompt": "a stone bridge",
            "sourceAssetId": asset_id,
            "strength": 0.25,
            "width": 1024,
            "height": 1024,
            "advanced": {
                "steps": 8,
                "lowResStart": { "version": 1, "k": 5, "scale": 0.5 }
            },
            "modelManifestEntry": entry.clone()
        });
        let authored = payload.as_object().unwrap();

        // 1. Validation (`sceneworks-options::low_res_start::requested_low_res_start`) accepts top-level strength 0.25 and computes 6 executed steps:
        let validated = sceneworks_core::options::low_res_start::requested_low_res_start(
            "flux_dev",
            authored,
            authored["advanced"].as_object().unwrap(),
            entry_obj,
        )
        .expect("validation accepts top-level strength 0.25 with K=5")
        .expect("low_res_start active");
        assert_eq!(validated.executed_steps, 6);
        assert_eq!(validated.k, 5);

        // 2. Core + worker lane controls (`ImageRequest::from_resolved` -> `resolve_lane_request_controls`)
        // share the same top-level strength binding and accept K=5:
        let req = ImageRequest::from_resolved(authored, authored).unwrap();
        assert_eq!(
            req.resolved_authored_img2img_strength().unwrap(),
            Some(0.25)
        );
        let caps = honouring();
        let resolved_controls =
            resolve_lane_request_controls(&req, &caps, Some(3.5), None).unwrap();
        assert_eq!(
            resolved_controls.low_res_start,
            Some(gen_core::LowResStartSpec::new(5, Some(0.5)))
        );

        // 3. Worker img2img source resolver (`resolve_flux1_ordinary_img2img_source`) resolves 0.25 (not default 0.5):
        let (_img, worker_strength) =
            crate::image_jobs::conditioning::resolve_flux1_ordinary_img2img_source(
                &req,
                &settings,
                &project_path,
            )
            .expect("ordinary img2img source resolves")
            .expect("source present");
        assert!(
            (worker_strength - 0.25).abs() < 1e-6,
            "got {worker_strength}"
        );
    }

    #[test]
    fn sw_7221_use_pid_2k_validates_low_res_start_against_pid_effective_base_dimensions_end_to_end()
    {
        let entry = json!({
            "id": "flux_dev",
            "family": "flux",
            "type": "image",
            "ui": { "pid": true },
            "defaults": {
                "steps": 8,
                "resolution": "1024x1024"
            },
            "features": {
                "low_res_start": true
            }
        });
        let entry_obj = entry.as_object().unwrap();

        // 1024x1024 with usePid: true, pidTarget: "2k", and lowResStart.scale: 0.25:
        // PiD 2K reduces effective base to 512x512, so scale 0.25 yields 128x128 < 256x256.
        // Both submission validation AND worker lane control resolution reject it consistently!
        let bad_payload = json!({
            "projectId": "p",
            "model": "flux_dev",
            "mode": "text_to_image",
            "prompt": "a cathedral",
            "width": 1024,
            "height": 1024,
            "advanced": {
                "steps": 8,
                "usePid": true,
                "pidTarget": "2k",
                "lowResStart": { "version": 1, "k": 3, "scale": 0.25 }
            },
            "modelManifestEntry": entry.clone()
        });
        let bad_authored = bad_payload.as_object().unwrap();
        let bad_req = ImageRequest::from_resolved(bad_authored, bad_authored).unwrap();
        let caps = honouring();

        // 1. With cached PiD checkpoint + Gemma snapshot on disk (`resolve_pid_weights` -> `Some`),
        // PiD is effectively active (`use_pid == true`), reducing the base from 1024x1024 to
        // 512x512; both options validation and worker lane controls reject scale 0.25 (128x128 < 256):
        let cached_root = tempfile::tempdir().unwrap();
        let cached_hub = cached_root.path().join("hub");
        let _cached_guard = sceneworks_core::hf_home::test_support::CacheRootGuard::new(
            cached_root.path().to_path_buf(),
            cached_hub.clone(),
        );
        let pid_snap = cached_hub.join(
            "models--SceneWorks--pid-flux/snapshots/1111111111111111111111111111111111111111",
        );
        let gemma_snap = cached_hub.join(
            "models--SceneWorks--gemma-2-2b-it/snapshots/2222222222222222222222222222222222222222",
        );
        std::fs::create_dir_all(&pid_snap).unwrap();
        std::fs::create_dir_all(&gemma_snap).unwrap();
        std::fs::write(pid_snap.join("pid_flux_2k.safetensors"), b"pid").unwrap();
        std::fs::write(gemma_snap.join("config.json"), b"{}").unwrap();

        let active_pid = crate::image_jobs::pid::resolve_pid_weights(
            &bad_req,
            cached_root.path(),
            &bad_req.model,
        )
        .unwrap()
        .is_some();
        assert!(
            active_pid,
            "PiD weights must resolve when checkpoint and Gemma exist"
        );

        let validation_err =
            sceneworks_core::options::low_res_start::requested_low_res_start_with_pid_active(
                "flux_dev",
                bad_authored,
                bad_authored["advanced"].as_object().unwrap(),
                entry_obj,
                active_pid,
            )
            .expect_err(
                "submission validation must reject 128x128 reduced grid under active PiD 2K",
            );
        assert_eq!(validation_err.field, "advanced.lowResStart.scale");
        assert!(
            validation_err.detail.contains("512x512 to 128x128"),
            "{}",
            validation_err.detail
        );
        assert_eq!(
            bad_req.pid_effective_base_dimensions(active_pid),
            (512, 512)
        );
        let worker_err =
            resolve_lane_request_controls_with_pid(&bad_req, &caps, Some(3.5), None, active_pid)
                .expect_err(
                    "worker lane controls must reject 128x128 reduced grid under active PiD 2K",
                );
        assert!(
            worker_err.to_string().contains("512x512 to 128x128"),
            "{worker_err}"
        );

        // 2. When PiD checkpoint or Gemma snapshot is absent on disk (`resolve_pid_weights` -> `None`),
        // execution falls back to the native VAE (`use_pid == false`) and retains the 1024x1024 base,
        // so scale 0.25 produces a valid 256x256 low-resolution grid in both validation and worker execution:
        let empty_root = tempfile::tempdir().unwrap();
        let empty_hub = empty_root.path().join("hub");
        let _empty_guard = sceneworks_core::hf_home::test_support::CacheRootGuard::new(
            empty_root.path().to_path_buf(),
            empty_hub,
        );
        let fallback_pid = crate::image_jobs::pid::resolve_pid_weights(
            &bad_req,
            empty_root.path(),
            &bad_req.model,
        )
        .unwrap()
        .is_some();
        assert!(
            !fallback_pid,
            "missing PiD checkpoint must fall back to native VAE"
        );

        let fallback_validated = sceneworks_core::options::low_res_start::requested_low_res_start(
            "flux_dev",
            bad_authored,
            bad_authored["advanced"].as_object().unwrap(),
            entry_obj,
        )
        .expect("missing-checkpoint fallback must pass static validation at 1024x1024 -> 256x256")
        .expect("present");
        assert_eq!(
            (fallback_validated.low_width, fallback_validated.low_height),
            (256, 256)
        );
        assert_eq!(
            (
                fallback_validated.full_width,
                fallback_validated.full_height
            ),
            (1024, 1024)
        );
        assert_eq!(
            bad_req.pid_effective_base_dimensions(fallback_pid),
            (1024, 1024)
        );
        let fallback_controls =
            resolve_lane_request_controls_with_pid(&bad_req, &caps, Some(3.5), None, fallback_pid)
                .expect("worker lane controls must allow 1024x1024 -> 256x256 when PiD falls back to native VAE");
        assert_eq!(
            fallback_controls.low_res_start,
            Some(gen_core::LowResStartSpec::new(3, Some(0.25)))
        );
    }
}
