use super::flux3_references;
use super::lane_controls;
use super::memory::scrub_untrusted_memory_strategy_disclosure;
use super::CancelFlag;
use super::ImageRequest;
use super::ResolvedModel;
use crate::advanced;
use crate::emit_event;
use crate::JsonObject;
use crate::Path;
use crate::Value;
use crate::WorkerResult;
use serde_json::json;
/// Resolve denoise steps: `advanced.steps` (clamped 1..=max(80, family default)) else the family
/// default.
/// Shared by the MLX path and the candle lane (sc-5096).
#[cfg(native_lane)]
pub(super) fn resolve_steps(request: &ImageRequest, model: &ResolvedModel) -> u32 {
    request
        .advanced
        .get("steps")
        .and_then(|value| {
            value
                .as_u64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        // The historical 80-step ceiling, raised to the model's own default where that is higher
        // (Iris-3B's release default is 100, sc-25679) so the default is always reachable rather
        // than silently clamped below what the model ships with.
        // A declared `limits.hardMaxSteps` (sc-24215) replaces the historical ceiling; the
        // model's own boundary refuses a request above it before this point.
        .map(|steps| {
            let ceiling =
                sceneworks_core::video_request::hard_max_steps(&request.model_manifest_entry)
                    .unwrap_or_else(|| max_requested_steps(model.default_steps()));
            (steps as u32).clamp(1, ceiling)
        })
        .unwrap_or(model.default_steps())
}

/// The largest `advanced.steps` the generic image lane forwards: the historical 80, or the model's
/// own default when that is higher. Mirrored by Image Studio's Steps input (`maxStepsForModel`).
#[cfg(native_lane)]
pub(super) fn max_requested_steps(default_steps: u32) -> u32 {
    default_steps.max(sceneworks_core::image_request::BASE_MAX_IMAGE_STEPS)
}

/// Resolve explicit guidance or the native default for providers that accept this axis.
/// Capability declarations distinguish CFG-off defaults from unsupported guidance.
#[cfg(native_lane)]
pub(super) fn resolve_guidance(request: &ImageRequest, model: &ResolvedModel) -> Option<f32> {
    if !model.supports_guidance() {
        return None;
    }
    let scale = request
        .advanced
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(model.default_guidance());
    Some(scale)
}

/// Resolve an unsigned advanced knob with a manifest fallback (sc-8825). The single mechanism the
/// bespoke edit/adapter/control lanes (`*_edit_candle.rs`, `sdxl_ipadapter.rs`, `kolors_*.rs`,
/// `qwen_control.rs`, `instantid.rs`, `pulid.rs`) had each re-implemented as an inline parse closure:
/// `advanced[key]` (JSON uint OR numeric string) → the manifest `[key]` (same parse) → `default`.
/// The parsed **advanced-or-manifest** value is clamped to `range`; the `default` is returned
/// **unclamped** (it is a trusted per-lane constant, and clamping it would silently change a lane
/// whose default sits outside its own historical range). Each caller passes its OWN `range`/`default`,
/// so the drifting steps bounds (1..=80 / 1..=50 / 1..=100) are preserved byte-for-byte — this is a
/// dedup-of-mechanism refactor, not a policy change.
#[cfg(native_lane)]
pub(super) fn resolve_advanced_or_manifest_u32(
    request: &ImageRequest,
    key: &str,
    default: u32,
    range: std::ops::RangeInclusive<u32>,
) -> u32 {
    resolve_advanced_or_manifest_u32_with(request, key, || default, range)
}

/// Resolve an advanced float with manifest fallback. Guidance remains uncapped; other
/// knobs retain their existing mathematical bounds.
#[cfg(native_lane)]
pub(super) fn resolve_advanced_or_manifest_f32(
    request: &ImageRequest,
    key: &str,
    default: f32,
    range: std::ops::RangeInclusive<f32>,
) -> f32 {
    resolve_advanced_or_manifest_f32_with(request, key, || default, range)
}

/// Closure-default twin of [`resolve_advanced_or_manifest_u32`] (sc-8825). Identical mechanism —
/// `advanced[key]` → manifest `[key]` (parsed, clamped to `range`) — except the fallback is a
/// per-lane `default_fn` closure evaluated **only** when both advanced and manifest are absent (and,
/// like the const twin, returned **unclamped**). This covers the lanes whose default is model-dependent
/// (`flux_ipadapter` variant steps, `qwen_edit_candle`/`zimage_control` per-variant steps) rather than a
/// bare constant. Every caller passes its OWN `range`/`default_fn`, so per-lane bounds stay byte-for-byte.
///
#[cfg(any(native_lane, test))]
pub(super) fn resolve_advanced_or_manifest_u32_with(
    request: &ImageRequest,
    key: &str,
    default_fn: impl FnOnce() -> u32,
    range: std::ops::RangeInclusive<u32>,
) -> u32 {
    let parse = |value: &Value| {
        value
            .as_u64()
            .or_else(|| value.as_str()?.trim().parse().ok())
    };
    request
        .advanced
        .get(key)
        .and_then(parse)
        .or_else(|| request.model_manifest_entry.get(key).and_then(parse))
        .map(|value| value.clamp(*range.start() as u64, *range.end() as u64) as u32)
        .unwrap_or_else(default_fn)
}

/// Closure-default twin of [`resolve_advanced_or_manifest_f32`] (sc-8825). Identical mechanism —
/// manifest `[key]` supplies the effective default (else the per-lane `default_fn` closure, evaluated
/// only when the manifest key is absent), then [`advanced::f32_clamped`] reads `advanced[key]` (falling
/// back to that default) and clamps to `range`. Covers `flux_ipadapter`, whose guidance fallback is a
/// per-variant fn. Each caller passes its OWN `range`/`default_fn`.
#[cfg(any(native_lane, test))]
pub(super) fn resolve_advanced_or_manifest_f32_with(
    request: &ImageRequest,
    key: &str,
    default_fn: impl FnOnce() -> f32,
    range: std::ops::RangeInclusive<f32>,
) -> f32 {
    let manifest_default = request
        .model_manifest_entry
        .get(key)
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or_else(default_fn);
    if key == "guidanceScale" {
        // Guidance recommendations are defaults, not experiment bounds. Native validation
        // owns finite values and the provider's actual mathematical restrictions.
        request
            .advanced
            .get(key)
            .and_then(|value| {
                value
                    .as_f64()
                    .or_else(|| value.as_str()?.trim().parse().ok())
            })
            .map(|value| value as f32)
            .unwrap_or(manifest_default)
    } else {
        advanced::f32_clamped(&request.advanced, key, manifest_default, range)
    }
}

/// True for a TRUE-CFG family whose engine reads the CFG scale from `true_cfg` (with a real
/// negative prompt) and **rejects** the distilled `guidance` scalar — i.e. Chroma (epic 3531),
/// uniquely identified by `supports_guidance=false` + `supports_negative_prompt=true`. The
/// guidance-distilled families (`z_image_turbo`, `flux_schnell`) are `false`/`false` (no CFG at
/// all), and the `guidance`-scalar families (qwen / sdxl / flux2 …) are `true`/*. For a true-CFG
/// family the worker forwards `advanced.guidanceScale` as `true_cfg`, not `guidance`.
#[cfg(native_lane)]
pub(super) fn uses_true_cfg(model: &ResolvedModel) -> bool {
    !model.supports_guidance() && model.supports_negative_prompt()
}

/// Apply the Ideogram placeholder recovery policy to one completed render. The renderer is injected
/// so the MLX and candle lanes share cancellation, reseeding, retry exhaustion, and error behavior.
#[cfg(native_lane)]
pub(super) fn recover_ideogram_placeholder<R>(
    enabled: bool,
    seed: i64,
    cancel: &CancelFlag,
    initial: (u32, u32, Vec<u8>),
    mut render: R,
) -> WorkerResult<(i64, u32, u32, Vec<u8>)>
where
    R: FnMut(i64) -> WorkerResult<(u32, u32, Vec<u8>)>,
{
    recover_ideogram_placeholder_with(
        enabled,
        crate::ideogram_caption::placeholder_recovery_retries(),
        seed,
        cancel,
        initial,
        |pixels, width, height| {
            crate::ideogram_caption::looks_like_placeholder(pixels, width, height)
        },
        &mut render,
    )
}

#[cfg(native_lane)]
pub(super) fn recover_ideogram_placeholder_with<R, P>(
    enabled: bool,
    retries: u32,
    seed: i64,
    cancel: &CancelFlag,
    initial: (u32, u32, Vec<u8>),
    mut is_placeholder: P,
    mut render: R,
) -> WorkerResult<(i64, u32, u32, Vec<u8>)>
where
    R: FnMut(i64) -> WorkerResult<(u32, u32, Vec<u8>)>,
    P: FnMut(&[u8], u32, u32) -> bool,
{
    let (mut width, mut height, mut pixels) = initial;
    if !enabled || !is_placeholder(&pixels, width, height) {
        return Ok((seed, width, height, pixels));
    }

    let mut final_seed = seed;
    for attempt in 0..retries {
        if cancel.is_cancelled() {
            break;
        }
        let retry_seed = crate::ideogram_caption::recovery_seed(seed, attempt);
        tracing::warn!(
            "ideogram 4 placeholder detected (seed {seed}); reseeding {retry_seed} \
             (attempt {}/{retries})",
            attempt + 1,
        );
        (width, height, pixels) = render(retry_seed)?;
        final_seed = retry_seed;
        if !is_placeholder(&pixels, width, height) {
            break;
        }
    }
    Ok((final_seed, width, height, pixels))
}

/// Resolve the true-CFG scale for a true-CFG family (Chroma). `None` for every other family
/// (their CFG, if any, flows through [`resolve_guidance`]). The scale is `advanced.guidanceScale`
/// (the same user knob) else the family default — forwarded to the engine as `GenerationRequest.true_cfg`.
/// Shared by the MLX path and the candle lane (sc-5096).
#[cfg(native_lane)]
pub(super) fn resolve_true_cfg(request: &ImageRequest, model: &ResolvedModel) -> Option<f32> {
    if !uses_true_cfg(model) {
        return None;
    }
    let scale = request
        .advanced
        .get("guidanceScale")
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32)
        .unwrap_or(model.default_guidance());
    Some(scale)
}

/// The negative prompt to pass to the engine. `None` for variants without true CFG
/// (the engine rejects `negative_prompt` on the distilled families) and for an empty
/// prompt (the true-CFG engines fall back to their own neutral negative).
/// Shared by the MLX path and the candle lane (sc-5096); on the Windows lane `supports_negative_prompt`
/// is the candle descriptor, so distilled candle families (z-image, flux schnell) get `None`.
#[cfg(native_lane)]
pub(super) fn resolve_negative_prompt(
    request: &ImageRequest,
    model: &ResolvedModel,
) -> Option<String> {
    if !model.supports_negative_prompt() {
        return None;
    }
    let trimmed = request.negative_prompt.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

pub(super) fn mlx_raw_settings(
    request: &ImageRequest,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
) -> JsonObject {
    mlx_raw_settings_for_bundle(request, None, repo, steps, quant_bits, guidance)
}

/// [`mlx_raw_settings`] for a render whose weights LOADED from `bundle` (the generic lane's resolved
/// `weights_dir`): Supra2's unconditional-context record is read off that bundle's manifest, never
/// re-resolved from the request (sc-24217 pre-merge review).
pub(super) fn mlx_raw_settings_for_bundle(
    request: &ImageRequest,
    bundle: Option<&Path>,
    repo: &str,
    steps: u32,
    quant_bits: Option<i64>,
    guidance: Option<f32>,
) -> JsonObject {
    let mut raw = if sceneworks_core::image_request::flux3::is_model(&request.model) {
        flux3_references::caller_raw_settings(&request.advanced)
    } else {
        request.advanced.clone()
    };
    // This key is worker-owned. A replayed recipe or hostile advanced payload must never be able to
    // forge Auto's rung-4 decision; the trusted Candle path may add it back after final admission.
    scrub_untrusted_memory_strategy_disclosure(&mut raw);
    // Worker-owned too (epic 25899): only the image's own render report adds the record back.
    raw.remove(lane_controls::PREFIX_KV_CACHE_KEY);
    raw.remove(sceneworks_core::moss_image::EDIT_REPORT_KEY);
    raw.remove("moss_image_edit_execution_report");
    raw.remove(sceneworks_core::moss_image::REPORT_KEY);
    raw.remove("moss_image_execution_report");
    // Worker-owned (sc-24215): the pinned weights Supra2 rendered from, never a replayed copy.
    // So are the unconditional context CFG steered against and the rendering backend (epic-review
    // follow-up): the reproducibility record, never a replayed copy.
    raw.remove(sceneworks_core::supra2::MODEL_REVISIONS_KEY);
    if sceneworks_core::supra2::is_supra2(&request.model) {
        raw.insert(
            sceneworks_core::supra2::MODEL_REVISIONS_KEY.to_owned(),
            sceneworks_core::supra2::pinned_revisions(&request.model_manifest_entry),
        );
        raw.insert(
            sceneworks_core::supra2::UNCONDITIONAL_KEY.to_owned(),
            sceneworks_core::supra2::unconditional_source(bundle, guidance),
        );
        raw.insert(
            sceneworks_core::supra2::BACKEND_KEY.to_owned(),
            Value::from(if cfg!(target_os = "macos") {
                "mlx"
            } else {
                "candle"
            }),
        );
    }
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    raw.insert("numInferenceSteps".to_owned(), json!(steps));
    // Distilled variants run without CFG (guidance == None → null in the recipe).
    raw.insert(
        "guidanceScale".to_owned(),
        guidance.map(|value| json!(value)).unwrap_or(Value::Null),
    );
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    if request.hires_fix.enabled {
        raw.insert(
            "hiresFix".to_owned(),
            serde_json::to_value(&request.hires_fix).expect("HiresFixRequest is serializable"),
        );
    }
    raw
}

/// Emit an `image_pipeline_load_{start,complete}` event from inside a blocking
/// generation closure (sc-3450), parity with the Python worker's pipeline-load
/// events. On the backend path `crate::inference_runtime::load` is a single atomic call that also fuses
/// any distill LoRA and applies user LoRAs (`spec.with_adapters`), so there is no
/// separable fuse/apply step to bracket: the adapter total (`adapter_count` =
/// distill + user) is reported here instead of via the torch worker's separate
/// `image_distill_lora_fuse_*` / `image_lora_apply_*` sub-phase events. A `start`
/// with no matching `complete` means the load failed (the error propagates via `?`).
pub(crate) fn emit_load_event(event: &str, job_id: &str, engine: &str, adapter_count: usize) {
    emit_event(
        event,
        json!({
            "jobId": job_id,
            "engine": engine,
            "adapterCount": adapter_count,
        }),
    );
}

/// N3 (epic 7114): a per-generation `sampler` / `scheduler` knob that names something the engine does
/// NOT advertise must never hard-fail the generation. `gen_core::Capabilities::validate_request` (and
/// each engine's own `validate`) rejects an unadvertised name with an `Err`, so the worker pre-filters
/// the knob here against the linked descriptor's advertised surface (`Capabilities.samplers` /
/// `.schedulers`): an advertised name passes through untouched; an unknown one — a stale recipe, a
/// per-backend capability gap (candle advertises a narrower set than mlx until P4), or manifest drift —
/// is dropped back to the engine default (`None`) and a `sampling_knob_unsupported` worker event is
/// emitted for observability. `None` and the `"default"` sentinel are already stripped at the read site,
/// so this only fires on a real, unsupported name. Shared by the MLX (`generate_stream`) + candle
/// (`generate_candle_stream`) image lanes and the video lane (`run_loaded_video_generation`).
pub(crate) fn normalize_sampling_knob(
    requested: Option<String>,
    advertised: &[&str],
    knob: &str,
    model_id: &str,
    job_id: &str,
    engine: &str,
) -> Option<String> {
    let name = requested?;
    if advertised.contains(&name.as_str()) {
        return Some(name);
    }
    tracing::warn!(
        "{engine}: requested {knob} {name:?} is not advertised (supported: {advertised:?}); \
         falling back to the engine default"
    );
    emit_sampling_knob_unsupported(knob, json!(name), advertised, model_id, job_id, engine);
    None
}

/// The `sampling_knob_unsupported` worker event [`normalize_sampling_knob`] emits when it drops a
/// requested knob back to the engine default. Shared with the other per-request drops that remove a
/// value the engine would refuse (Iris-3B's CFG-bound negative prompt and schedule shift, sc-25679),
/// so every silent drop is observable through the one event.
pub(crate) fn emit_sampling_knob_unsupported(
    knob: &str,
    requested: Value,
    advertised: &[&str],
    model_id: &str,
    job_id: &str,
    engine: &str,
) {
    emit_event(
        "sampling_knob_unsupported",
        json!({
            "jobId": job_id,
            "engine": engine,
            "model": model_id,
            "knob": knob,
            "requested": requested,
            "supported": advertised,
        }),
    );
}

/// Read the raw per-generation sampler / scheduler / schedule-shift knobs from a job's `advanced`
/// block (the 1753 front-half carrier). `sampler` / `scheduler` strip the `"default"` sentinel + blanks
/// to `None`, so the engine default — N1's guaranteed no-op — is the ABSENCE of a name, not a magic
/// string; `scheduler_shift` accepts the `schedulerShift` (or legacy `timestepShift`) key as a number or
/// numeric string. Shared by the MLX (`generate_stream`) + candle (`generate_candle_stream`) image lanes
/// — the result is then realvisxl-forced (the lightning checkpoint) and N3-guarded via
/// [`normalize_sampling_knob`]. Returns `(sampler, scheduler, scheduler_shift)`.
pub(crate) fn read_advanced_sampling_knobs(
    advanced: &JsonObject,
) -> (Option<String>, Option<String>, Option<f32>) {
    let name = |key: &str| {
        advanced
            .get(key)
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty() && *value != "default")
            .map(str::to_owned)
    };
    let scheduler_shift = advanced
        .get("schedulerShift")
        .or_else(|| advanced.get("timestepShift"))
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str()?.trim().parse().ok())
        })
        .map(|value| value as f32);
    (name("sampler"), name("scheduler"), scheduler_shift)
}

/// Read the per-generation guidance method (epic 7434 P5, sc-7448) from a job's `advanced` block —
/// the 4th sampling axis (`cfg` / `cfg_rescale` / `apg` / `cfg_pp`), alongside the sampler/scheduler
/// knobs. Strips the `"default"` sentinel + blanks to `None` so the engine default (the N1 no-op) is
/// the ABSENCE of a method. The result is then N3-guarded via [`normalize_sampling_knob`] against the
/// model descriptor's `supported_guidance_methods` and threaded onto `GenerationRequest.guidance_method`.
pub(crate) fn read_advanced_guidance_method(advanced: &JsonObject) -> Option<String> {
    advanced
        .get("guidanceMethod")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty() && *value != "default")
        .map(str::to_owned)
}

/// The curated sampler/scheduler menu (epic 7114 decision 2) the **bespoke** conditioned image paths
/// honor — the shared `gen_core` solver/scheduler vocabulary the unified-sampler engines gate on
/// (`Solver::from_name` / the additive `denoise_curated` path; mlx #537/#538/#539, candle #130). The
/// bespoke per-family paths (InstantID, Kolors-conditioned, PuLID — sc-7432) build CUSTOM request
/// structs OUTSIDE `generate_stream`'s generic plumbing, so they N3-normalize the per-request knob
/// against THIS menu instead of a `Capabilities` list: every engine's advertised set is a superset of
/// it (their native default is the only extra, and `"default"`/`None` already strip to the engine
/// default), so a name that survives [`normalize_sampling_knob`] here also passes the engine's own
/// `validate_request`. This is also the single source of truth the manifest⊆engine drift guard
/// (`engines.rs`) checks these out-of-`MODEL_TABLE` models against, so the runtime and the guard never
/// disagree. Derived from `gen_core` (the engines' own vocab), so it tracks the framework on BOTH
/// backends rather than hard-coding names. Returns `(samplers, schedulers)`.
pub(crate) fn curated_image_menu() -> (Vec<&'static str>, Vec<&'static str>) {
    (
        gen_core::sampling::Solver::ALL
            .iter()
            .map(|solver| solver.name())
            .collect(),
        gen_core::sampling::Scheduler::ALL
            .iter()
            .map(|scheduler| scheduler.name())
            .collect(),
    )
}

#[cfg(native_lane)]
#[cfg(test)]
#[path = "tests/sampling_knob_tests.rs"]
mod sampling_knob_tests;

/// Explicit Advanced values for bespoke paths whose defaults belong to a selected recipe.
/// Deliberately omit manifest fallbacks here: an accelerator can have different defaults from
/// the selected base checkpoint. Its adapter supplies those only when an override is absent.
#[cfg(any(native_lane, test))]
pub(super) fn explicit_sampling_request(request: &ImageRequest) -> gen_core::GenerationRequest {
    let (sampler, scheduler, scheduler_shift) = read_advanced_sampling_knobs(&request.advanced);
    let guidance = sceneworks_core::image_sampling_controls::effective_guidance(
        &request.advanced,
        &JsonObject::new(),
    )
    .map(|value| value as f32);
    gen_core::GenerationRequest {
        sampler,
        scheduler,
        scheduler_shift,
        guidance,
        negative_prompt: (!request.negative_prompt.trim().is_empty())
            .then(|| request.negative_prompt.clone()),
        ..Default::default()
    }
}
