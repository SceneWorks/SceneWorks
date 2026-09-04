// ---------------------------------------------------------------------------
// Krea 2 multi-phase denoise (epic 13879 S4, sc-13884): a `krea_2_raw` t2i job
// carrying an explicit `advanced.phases` list rendered through the new multi-phase engine
// primitive (inference PR #199 on MLX; PR #204 brought the candle sibling, sc-13887). ONE Raw
// denoise trajectory over ONE global sigma schedule,
// where each PHASE owns a contiguous step slice with its own guidance (per-phase true-CFG
// on/off) AND its own active subset of the job's load-time LoRA stack (per-phase toggling).
// The canonical workflow is "N steps Raw with true-CFG on, then M steps Raw + turbo-LoRA with
// CFG off", but the lane is general over any ordered phase list. Routed here from
// `ImageRoute::KreaMultiPhase`.
//
// # The phase list references the job's OWN LoRAs by index
//
// The gen-core contract (`GenerationRequest.phases`) references the load-time adapter stack
// (`LoadSpec::adapters`) by INDEX: a `PhaseAdapter { adapter: usize, weight }` names which of the
// already-loaded adapters this phase activates. The worker builds that stack from `request.loras`
// via `resolve_adapters`, which emits exactly one `AdapterSpec` per lora entry, IN ORDER — so
// `request.loras[i]` maps 1:1 to `LoadSpec::adapters[i]`. The client-supplied `advanced.phases`
// therefore selects a phase's adapters by index into the SAME `loras` array the job already loads;
// [`parse_multiphase_specs`] bounds-checks each index against `request.loras.len()` and
// [`build_generation_phases`] maps it straight to `PhaseAdapter.adapter`.
//
// # Engine surface + what this lane rejects
//
// Multi-phase is the Raw t2i variant only: it renders from PURE NOISE, so the engine
// (`mlx-gen-krea`'s `validate_phases`) rejects phases combined with reference/edit conditioning
// or the PiD decoder, and rejects an empty list / a 0-step phase. This lane mirrors THOSE rejects
// early on the SceneWorks side ([`ensure_multiphase_job_shape`] + [`parse_multiphase_specs`]) so a
// bad shape fails fast with a clear error BEFORE the heavy load, rather than deep in the engine.
//
// NOT mirrored here: the engine ALSO loud-rejects a multi-phase request on a model loaded with a
// ComfyUI `.diff`/`.diff_b` diff-patch adapter (that delta folds irreversibly into the base and
// cannot be toggled off for a base-only phase). That reject is ENGINE-enforced only — it fires LATE,
// after the adapters load, in the engine's `ensure_multiphase_allowed_for` (which sniffs the adapter
// file headers). There is deliberately no early SceneWorks-side probe: the engine already rejects it
// loudly, and the canonical workflow uses the ALLOWED low-rank turbo LoRA/LoKr (which toggles cleanly),
// so an early header sniff here would be redundant work on the happy path.
//
// The lane loads the `krea_2_raw` ENGINE (unlike S3's turbo-on-Raw, which swaps to the `krea_2_turbo`
// engine): multi-phase is a Raw-trajectory decomposition, and the engine keys the driver on the
// `krea_2_raw` descriptor id. This composes with — and takes PRECEDENCE over — the S3 whole-job
// turbo-on-Raw regime: an explicit `advanced.phases` is the finer-grained control, so the router
// checks this lane first (see `resolve_image_route` / `resolve_candle_image_route`).
//
// The whole file is backend-neutral and `include!`d on BOTH the macOS (MLX) build AND the candle
// build (sc-13887, like `krea_turbo_raw.rs`): the shape-detection, the `advanced.phases`
// parse/validate → `GenerationRequest::phases`, and the engine invocation all go through the shared
// registry harness + the backend-agnostic gen-core `phases` contract, so nothing here is
// MLX-specific and nothing needs a per-item cfg. The candle Krea engine honors `phases` too
// (inference PR #204).
// ---------------------------------------------------------------------------

/// Sanity cap on the number of phases in one multi-phase render. The engine imposes none, but a
/// legitimate Raw→Raw+turbo-LoRA workflow is 2–3 phases; a generous cap rejects an obviously
/// malformed / abusive list without constraining any real split. Not a creative limit — a guardrail.
const MAX_MULTIPHASE_PHASES: usize = 8;

/// Sanity cap on the TOTAL denoise-step budget (the sum of every phase's steps, which is the length
/// of the ONE global schedule). Single-phase Raw is 52 steps; a multi-phase split is typically well
/// under that. A generous ceiling catches a runaway/typo total (e.g. a phase asking for thousands of
/// steps) before it becomes a multi-minute render. Not a creative limit — a guardrail.
const MAX_MULTIPHASE_TOTAL_STEPS: u32 = 150;

/// One phase parsed from a job's `advanced.phases` — the SceneWorks-side mirror of gen-core's
/// [`gen_core::GenerationPhase`]. `loras` reference the job's OWN LoRA stack by index (which maps
/// 1:1 to `LoadSpec::adapters`); an EMPTY `loras` is a base-only phase (the common phase-1 case).
#[derive(Clone, Debug, PartialEq)]
struct MultiPhaseSpec {
    /// Contiguous denoise steps this phase runs (≥ 1). The sum across phases is the total budget.
    steps: u32,
    /// Per-phase guidance: `Some(g > 0)` = true-CFG, `Some(0.0)` = CFG off, `None` inherits the
    /// request/Raw default.
    guidance: Option<f32>,
    /// The load-time adapters this phase activates, referencing `request.loras` (== `LoadSpec::adapters`)
    /// by index. Empty = base-only.
    loras: Vec<PhaseLoraRef>,
}

/// One adapter a phase activates: an index into the job's `loras` array (== `LoadSpec::adapters`,
/// since `resolve_adapters` emits one spec per lora in order) plus an optional per-phase weight
/// override (`None` uses the adapter's load-time scale).
#[derive(Clone, Copy, Debug, PartialEq)]
struct PhaseLoraRef {
    index: usize,
    weight: Option<f32>,
}

/// True when a job carries an explicit `advanced.phases` (a present, non-null value). Cheap presence
/// check for the router gate — the shape is fully parsed + validated later by [`parse_multiphase_specs`],
/// so a malformed `phases` still routes HERE and fails loudly rather than being silently dropped by
/// falling through to another lane.
fn request_has_multiphase(request: &ImageRequest) -> bool {
    matches!(request.advanced.get("phases"), Some(value) if !value.is_null())
}

/// True when this is a Krea 2 **Raw** job carrying an explicit `advanced.phases` list and its Raw
/// weights resolve locally — the multi-phase denoise lane (S4). Keyed on model `krea_2_raw` + a present
/// `advanced.phases`, so EVERY job without `advanced.phases` is byte-for-byte unaffected. Placed FIRST
/// among the `krea_2_raw` lanes in [`prepare_image_route`] so an explicit phase list takes precedence
/// over the S3 whole-job turbo-on-Raw regime and the generic Raw t2i. The lane itself rejects
/// edit/pose/reference/PiD shapes loudly (multi-phase renders from pure noise), so the gate claims the
/// job and surfaces a clear error rather than diverting a conflicting shape elsewhere.
fn krea_multiphase_available(request: &ImageRequest, settings: &Settings) -> bool {
    request.model == KREA_RAW_MODEL_ID
        && request_has_multiphase(request)
        && matches!(resolve_weights_dir(request, settings), Ok(Some(_)))
}

/// Reject the job SHAPES the multi-phase engine rejects, loudly and BEFORE the heavy load (mirrors
/// `mlx-gen-krea`'s `validate_phases`): multi-phase renders from pure noise, so edit mode, strict-pose
/// conditioning, an img2img reference, and the PiD decoder are all unsupported in v1. Surfacing these
/// here keeps a conflicting request from being silently rendered as bare-noise multi-phase (dropping
/// the reference/pose/edit the user asked for) — the router sends any `krea_2_raw` + `advanced.phases`
/// job to this lane, so this is where the conflict is caught.
///
/// AUTHORITATIVE guard, not a convenience: [`generate_krea_multiphase_stream`] builds a t2i-only
/// `GenerationRequest` (empty `conditioning`, no `use_pid`), so the engine's own conditioning backstop
/// (`validate_phases`' `!req.conditioning.is_empty()` / `use_pid` rejects) can NEVER fire — the engine
/// only ever sees a clean t2i request from this lane. That makes THESE checks the sole protection
/// against silently dropping a Raw conditioning source. Any future Raw conditioning entry point
/// (a new img2img/edit/mask/reference shape that reaches multi-phase) MUST add its reject here, or it
/// will be dropped with no error.
fn ensure_multiphase_job_shape(request: &ImageRequest) -> WorkerResult<()> {
    if request.mode == "edit_image" {
        return Err(WorkerError::InvalidPayload(
            "Krea multi-phase denoise renders from pure noise — edit mode (edit_image) is not \
             supported (sc-13884). Remove advanced.phases to run an edit."
                .to_owned(),
        ));
    }
    if !pose_entries(request).is_empty() {
        return Err(WorkerError::InvalidPayload(
            "Krea multi-phase denoise renders from pure noise — strict-pose conditioning is not \
             supported (sc-13884)."
                .to_owned(),
        ));
    }
    if request_carries_img2img_reference(request) {
        return Err(WorkerError::InvalidPayload(
            "Krea multi-phase denoise renders from pure noise — an img2img reference is not \
             supported (sc-13884)."
                .to_owned(),
        ));
    }
    if advanced::flag(&request.advanced, "usePid") {
        return Err(WorkerError::InvalidPayload(
            "Krea multi-phase denoise does not support the PiD decoder yet (sc-13884 follow-on)."
                .to_owned(),
        ));
    }
    Ok(())
}

/// Parse + validate `advanced.phases` into the SceneWorks-side [`MultiPhaseSpec`] plan, bounds-checking
/// each phase's lora index against the job's own `loras` stack (== `LoadSpec::adapters`).
///
/// Rejects, loudly, exactly what the engine would (so the failure is early + clear, not deep in the
/// driver): a non-array / empty `phases`, more than [`MAX_MULTIPHASE_PHASES`] phases, a phase that is
/// not an object, a missing / 0-step `steps`, a non-finite or negative `guidance`, a `loras` that is
/// not an array, a lora entry missing `index`, a lora `index` out of range of the job's loras, a
/// non-finite lora `weight`, or a total step budget over [`MAX_MULTIPHASE_TOTAL_STEPS`]. `steps` and
/// numeric fields accept a JSON number or a numeric string (matching the rest of the `advanced` knobs).
fn parse_multiphase_specs(request: &ImageRequest) -> WorkerResult<Vec<MultiPhaseSpec>> {
    let raw = request.advanced.get("phases").ok_or_else(|| {
        WorkerError::InvalidPayload("Krea multi-phase job is missing 'advanced.phases'.".to_owned())
    })?;
    let list = raw.as_array().ok_or_else(|| {
        WorkerError::InvalidPayload(
            "'advanced.phases' must be an array of phase objects.".to_owned(),
        )
    })?;
    if list.is_empty() {
        return Err(WorkerError::InvalidPayload(
            "'advanced.phases' must contain at least one phase.".to_owned(),
        ));
    }
    if list.len() > MAX_MULTIPHASE_PHASES {
        return Err(WorkerError::InvalidPayload(format!(
            "'advanced.phases' has {} phases; at most {MAX_MULTIPHASE_PHASES} are supported.",
            list.len()
        )));
    }

    let lora_count = request.loras.len();
    let mut specs = Vec::with_capacity(list.len());
    let mut total_steps: u64 = 0;

    for (i, entry) in list.iter().enumerate() {
        let obj = entry.as_object().ok_or_else(|| {
            WorkerError::InvalidPayload(format!("phase {i}: each phase must be a JSON object."))
        })?;

        // steps: required, ≥ 1.
        let steps_raw = obj
            .get("steps")
            .and_then(|value| value.as_u64().or_else(|| value.as_str()?.trim().parse().ok()))
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "phase {i}: 'steps' is required (a positive integer)."
                ))
            })?;
        if steps_raw == 0 {
            return Err(WorkerError::InvalidPayload(format!(
                "phase {i}: 'steps' must be at least 1."
            )));
        }
        // Clamp the cast; the total-budget cap below rejects any abusive value regardless.
        let steps = steps_raw.min(u64::from(u32::MAX)) as u32;
        total_steps += u64::from(steps);

        // guidance: optional; finite and ≥ 0 when present (0 = CFG off, > 0 = CFG on).
        let guidance = match obj.get("guidance") {
            None | Some(Value::Null) => None,
            Some(value) => {
                let g = value
                    .as_f64()
                    .or_else(|| value.as_str()?.trim().parse().ok())
                    .ok_or_else(|| {
                        WorkerError::InvalidPayload(format!(
                            "phase {i}: 'guidance' must be a number."
                        ))
                    })? as f32;
                if !g.is_finite() || g < 0.0 {
                    return Err(WorkerError::InvalidPayload(format!(
                        "phase {i}: 'guidance' must be a finite value >= 0 (0 = CFG off, > 0 = CFG on)."
                    )));
                }
                Some(g)
            }
        };

        // loras: optional array; empty / absent = a base-only phase.
        let loras = match obj.get("loras") {
            None | Some(Value::Null) => Vec::new(),
            Some(value) => {
                let arr = value.as_array().ok_or_else(|| {
                    WorkerError::InvalidPayload(format!(
                        "phase {i}: 'loras' must be an array of {{ index, weight? }} entries."
                    ))
                })?;
                let mut refs = Vec::with_capacity(arr.len());
                for (m, lora) in arr.iter().enumerate() {
                    let lora_obj = lora.as_object().ok_or_else(|| {
                        WorkerError::InvalidPayload(format!(
                            "phase {i} lora {m}: each entry must be a JSON object."
                        ))
                    })?;
                    let index = lora_obj
                        .get("index")
                        .and_then(|value| {
                            value.as_u64().or_else(|| value.as_str()?.trim().parse().ok())
                        })
                        .ok_or_else(|| {
                            WorkerError::InvalidPayload(format!(
                                "phase {i} lora {m}: 'index' is required (an index into the job's loras)."
                            ))
                        })? as usize;
                    if index >= lora_count {
                        return Err(WorkerError::InvalidPayload(format!(
                            "phase {i} lora {m}: index {index} is out of range — the job loads \
                             {lora_count} LoRA(s) (valid indices 0..{}).",
                            lora_count.saturating_sub(1)
                        )));
                    }
                    let weight = match lora_obj.get("weight") {
                        None | Some(Value::Null) => None,
                        Some(value) => {
                            let w = value
                                .as_f64()
                                .or_else(|| value.as_str()?.trim().parse().ok())
                                .ok_or_else(|| {
                                    WorkerError::InvalidPayload(format!(
                                        "phase {i} lora {m}: 'weight' must be a number."
                                    ))
                                })? as f32;
                            if !w.is_finite() {
                                return Err(WorkerError::InvalidPayload(format!(
                                    "phase {i} lora {m}: 'weight' must be finite."
                                )));
                            }
                            Some(w)
                        }
                    };
                    refs.push(PhaseLoraRef { index, weight });
                }
                refs
            }
        };

        specs.push(MultiPhaseSpec {
            steps,
            guidance,
            loras,
        });
    }

    if total_steps > u64::from(MAX_MULTIPHASE_TOTAL_STEPS) {
        return Err(WorkerError::InvalidPayload(format!(
            "multi-phase total step budget is {total_steps}; at most {MAX_MULTIPHASE_TOTAL_STEPS} \
             are supported."
        )));
    }

    Ok(specs)
}

/// Map the parsed [`MultiPhaseSpec`] plan to the gen-core `phases` a `GenerationRequest` carries. Each
/// phase's lora `index` becomes `PhaseAdapter.adapter` VERBATIM — the index into `LoadSpec::adapters`
/// the engine bounds-checks against the loaded stack (which the worker builds from the same
/// `request.loras`, in order). `steps`/`guidance`/`weight` pass through unchanged.
fn build_generation_phases(specs: &[MultiPhaseSpec]) -> Vec<gen_core::GenerationPhase> {
    specs
        .iter()
        .map(|phase| gen_core::GenerationPhase {
            steps: phase.steps,
            guidance: phase.guidance,
            adapters: phase
                .loras
                .iter()
                .map(|lora| gen_core::PhaseAdapter {
                    adapter: lora.index,
                    weight: lora.weight,
                })
                .collect(),
        })
        .collect()
}

/// Flat telemetry for a multi-phase generation (parity with [`krea_turbo_on_raw_raw_settings`] /
/// [`mlx_raw_settings`]). Records the Raw repo/tier that loaded, the TOTAL step budget (the sum of the
/// phases' steps — the flat `numInferenceSteps`), a null job-level `guidanceScale` (per-phase guidance
/// drives CFG), a `samplingRegime` marker, a `multiPhase` flag, and a compact replay-friendly summary
/// of the resolved phase plan.
fn krea_multiphase_raw_settings(
    request: &ImageRequest,
    repo: &str,
    total_steps: u32,
    quant_bits: Option<i64>,
    specs: &[MultiPhaseSpec],
) -> JsonObject {
    let mut raw = request.advanced.clone();
    raw.insert("realModelInference".to_owned(), Value::Bool(true));
    raw.insert("repo".to_owned(), Value::String(repo.to_owned()));
    // The engine ignores the flat `steps` when phases are present; record the SUM it actually runs.
    raw.insert("numInferenceSteps".to_owned(), json!(total_steps));
    // No single job-level guidance scale — each phase carries its own (CFG on/off).
    raw.insert("guidanceScale".to_owned(), Value::Null);
    raw.insert(
        "mlxQuantize".to_owned(),
        quant_bits.map(|bits| json!(bits)).unwrap_or(Value::Null),
    );
    raw.insert(
        "samplingRegime".to_owned(),
        Value::String(format!("{KREA_RAW_MODEL_ID}_multiphase")),
    );
    raw.insert("multiPhase".to_owned(), Value::Bool(true));
    // Replace the raw input `phases` with the resolved/normalized plan (indices + weights + steps) for
    // lineage / replay.
    let phases: Vec<Value> = specs
        .iter()
        .map(|phase| {
            json!({
                "steps": phase.steps,
                "guidance": phase.guidance,
                "loras": phase
                    .loras
                    .iter()
                    .map(|lora| json!({ "index": lora.index, "weight": lora.weight }))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    raw.insert("phases".to_owned(), Value::Array(phases));
    raw
}

/// Generate one multi-phase image (plain t2i, no reference/edit conditioning). The `phases` drive the
/// engine's multi-phase driver: ONE global schedule for the total step budget, each phase a contiguous
/// slice with its own guidance (CFG on/off) + its own active LoRA subset (indices into the loaded
/// stack). The flat `steps` is left `None` (ignored under `phases`); `guidance` is the request-level
/// default a `None`-guidance phase inherits.
#[allow(clippy::too_many_arguments)]
fn krea_multiphase_generate_one(
    generator: &dyn Generator,
    prompt: &str,
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    seed: i64,
    guidance: Option<f32>,
    text_style_gain: Option<f32>,
    phases: Vec<gen_core::GenerationPhase>,
    memory: Option<gen_core::GenerationMemory>,
    memory_strategy_context: Option<&gen_core::MemoryRunContext>,
    preview: gen_core::PreviewSink,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> WorkerResult<(u32, u32, Vec<u8>)> {
    let mut request = GenerationRequest {
        prompt: prompt.to_owned(),
        // Raw supports a negative prompt for the true-CFG phases; forwarded when the job carries one.
        negative_prompt,
        width,
        height,
        count: 1,
        seed: Some(seed as u64),
        // Ignored when `phases` is present — the engine sums the phases' steps for the ONE schedule.
        steps: None,
        // Request-level default guidance a phase with `guidance: None` inherits (per-phase guidance
        // overrides it). Matches the single-phase Raw default (`resolve_guidance`).
        guidance,
        text_style_gain,
        phases: Some(phases),
        memory,
        preview,
        cancel: cancel.clone(),
        ..Default::default()
    };
    let output = crate::memory_strategy::generate_with_scope(
        generator,
        &mut request,
        memory_strategy_context,
        on_progress,
    )
    .map_err(|error| {
        WorkerError::Engine(format!("Krea multi-phase generation failed: {error}"))
    })?;
    match output {
        GenerationOutput::Images(mut images) => {
            let image = images.pop().ok_or_else(|| {
                WorkerError::Engine("Krea multi-phase generator produced no image".to_owned())
            })?;
            Ok((image.width, image.height, image.pixels))
        }
        _ => Err(WorkerError::Engine(
            "Krea multi-phase generator returned non-image output".to_owned(),
        )),
    }
}

/// The `LoadSpec::quantize` this lane hands the engine, per backend.
///
/// A named function rather than an inline `cfg` split because the two arms silently diverged once
/// already: the candle arm used to hardcode `None`, dropping the job's requested tier on the floor
/// where the sibling `krea_edit.rs` candle arm forwards it. Tier is a whole-pipeline contract — q4
/// means q4 for every segment — so the lane must not decide the tier behind the engine's back.
///
/// * **macOS/MLX** delegates to [`mlx_load_quant_for_resolved_artifact`], which maps the request to
///   what the RESOLVED per-tier artifact actually needs (the MLX turnkeys ship pre-quantized, so a
///   packed tier resolves to a `None` load quant on purpose).
/// * **Candle** forwards the request unchanged, matching `krea_edit.rs`. At pin 931366f62,
///   `candle-gen-krea`'s `actual_quant_tier` ignores `spec.quantize` entirely for a
///   `WeightsSource::Dir` spec (the builtin per-tier turnkey — the tier IS the staged directory), so
///   this is inert for the catalog path. It is load-bearing for an imported `WeightsSource::File`
///   native checkpoint: there, `(companion = Some(tier), requested = None)` is a hard
///   `Unsupported` — "no quant request, but its companion snapshot is packed" — so hardcoding `None`
///   made a legitimate matching-tier multi-phase request unrunnable while telling the user to
///   "request the matching tier" they had already requested. `spec.quantize` Q4/Q8 is documented
///   ACCEPTED by that provider (inference sc-9607).
fn multiphase_load_quant(engine_id: &str, quant: Option<Quant>) -> Option<Quant> {
    #[cfg(target_os = "macos")]
    {
        mlx_load_quant_for_resolved_artifact(engine_id, quant)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = engine_id;
        quant
    }
}

/// Real multi-phase generation (epic 13879 S4, sc-13884): validate the job shape + phase list, resolve
/// the RAW weights + tier + the job's LoRA stack (the SAME stack the phases reference by index), load
/// the `krea_2_raw` ENGINE (the multi-phase driver keys on its descriptor id), and drive one output per
/// requested count through the phase plan. Mirrors [`generate_krea_turbo_on_raw_stream`]'s
/// blocking-thread + streamed-events shape and reuses [`consume_gen_events`]; differs in loading the Raw
/// (not Turbo) engine and passing `phases: Some(..)`.
async fn generate_krea_multiphase_stream(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    plan: &ImagePlan,
    project_path: &Path,
    backend: &str,
    asset_writes: &mut Vec<Value>,
) -> WorkerResult<()> {
    let request = &plan.request;

    // Reject shapes the engine will reject (edit / pose / img2img reference / PiD), loudly, before any
    // heavy work — the router sends every `krea_2_raw` + `advanced.phases` job here, so this is the
    // single place a conflicting shape is caught rather than silently rendered from bare noise.
    ensure_multiphase_job_shape(request)?;
    // Parse + validate the phase list against the job's own LoRA stack (indices bounds-checked here).
    let specs = parse_multiphase_specs(request)?;

    let raw_model = mlx_model(&request.model)
        .ok_or_else(|| WorkerError::InvalidPayload("not an MLX-backed model".to_owned()))?;
    let weights_dir = resolve_weights_dir(request, settings)?
        .ok_or_else(|| WorkerError::InvalidPayload("Krea 2 Raw weights not found".to_owned()))?;
    // The `krea_2_raw` engine IS the multi-phase driver — its descriptor id (`krea_2_raw`) is what the
    // engine's `validate_phases` requires (multi-phase is the Raw t2i variant only). Distinct from the
    // S3 turbo-on-Raw lane, which swaps to the `krea_2_turbo` engine.
    let engine_id = raw_model.engine_id();

    let (quant, quant_bits) = resolve_quant(request, Some(&weights_dir));
    // The SAME adapter stack the phases reference by index: `resolve_adapters` emits one `AdapterSpec`
    // per `request.loras` entry, in order, so `LoadSpec::adapters[i]` is `request.loras[i]` and a
    // phase's lora `index` selects it directly.
    let adapters = resolve_adapters(request, settings)?;
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let adapter_resident_bytes = adapters.iter().fold(0_u64, |total, adapter| {
        total.saturating_add(gen_core::safetensors_path_bytes(&adapter.path))
    });
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let offload_policy = admit_candle_base(
        request,
        settings,
        &weights_dir,
        "Krea multi-phase",
        CandleBaseEvidence::Catalog,
        adapter_resident_bytes,
        crate::mlx_fit_gate::engine_supports_sequential(engine_id),
    )
    .await?;
    let repo = model_repo(request, &raw_model);
    let adapter_label = raw_model.adapter_label();
    let text_style_gain = resolve_text_style_gain(request);
    // Request-level default guidance for any phase that omits its own (the engine inherits
    // `req.guidance.unwrap_or(DEFAULT_RAW_GUIDANCE)`) — the same Raw default the single-phase path uses.
    let guidance = resolve_guidance(request, &raw_model);
    let negative_prompt =
        (!request.negative_prompt.trim().is_empty()).then(|| request.negative_prompt.clone());

    let phases = build_generation_phases(&specs);
    let total_steps: u32 = specs.iter().map(|phase| phase.steps).sum();

    // Plain per-image work: `request.count` renders, each its own seed + the base prompt.
    let work: Vec<(i64, String)> = (0..request.count as usize)
        .map(|index| (resolve_seed(request, index), request.prompt.clone()))
        .collect();
    let total = work.len();
    let raw_settings = krea_multiphase_raw_settings(request, &repo, total_steps, quant_bits, &specs);

    let (width, height) = (request.width, request.height);
    let adapter_count = adapters.len();
    let load_quant = multiphase_load_quant(engine_id, quant);
    let mut spec = attach_selected_decoder(
        load_spec(weights_dir.clone(), load_quant, adapters, None)
            .with_resolved_route(request.model.clone()),
        engine_id,
        request,
        settings,
    )?;
    spec = attach_required_components(
        spec,
        engine_id,
        &request.model_manifest_entry,
        settings,
    )?;
    let unattached_spec = spec;
    let attached_spec =
        attach_manifest_text_encoder(unattached_spec, engine_id, request, settings)?;
    let spec = attached_spec.into_load_spec();
    #[cfg(target_os = "macos")]
    let spec = {
        let resolved_tier = resolved_mlx_artifact_tier(&weights_dir, quant_bits);
        let route_mode = crate::memory_route_registry::MemoryRouteMode::TextToImage;
        let shaped = crate::memory_route_registry::evaluate_declared_mlx_load_shape_for_request(
            engine_id,
            resolved_tier,
            Some(route_mode),
            &request.model_manifest_entry,
            spec,
            crate::memory_route_registry::MemoryRouteRequestContext {
                mode: route_mode,
                reference_count: 0,
                use_pid: false,
                has_phases: true,
            },
        );
        if let Some(warning) =
            crate::memory_route_registry::mlx_load_shape_declaration_warning(&shaped)
        {
            tracing::warn!(
                event = "mlx_load_shape_declaration_warning",
                provider = engine_id,
                ?warning,
                "provider refused deferred materialization; retaining the safe eager multi-phase load path"
            );
        }
        shaped
    };
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let spec = spec.with_offload_policy(offload_policy);

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let (generation_memory, memory_strategy_context) = {
        let tier = candle_resolved_tier_key(request, &weights_dir, false);
        let raw_budget = crate::vram_gate::apply_vram_cap(
            crate::gpu::nvidia_vram_budget_gb(&settings.gpu_id).await,
            crate::vram_gate::cuda_vram_cap_gb(),
        );
        let reclaimable_gb = crate::vram_gate::reclaimable_pool_gb(&settings.gpu_id);
        let budget = raw_budget
            .map(|budget| crate::vram_gate::with_reclaimable(budget, reclaimable_gb));
        let predicted_peak_gb = crate::vram_gate::predicted_peak_gb_with_adapter_bytes(
            &request.model_manifest_entry,
            tier,
            adapter_resident_bytes,
        );
        let evaluation = crate::candle_memory_strategy::evaluate_shared_image(
            engine_id,
            &request.model,
            &spec,
            candle_certified_load_spec(
                engine_id,
                settings,
                &spec,
                &request.model_manifest_entry,
                tier,
            ),
            &request.model_manifest_entry,
            tier,
            "text_to_image",
            (adapter_count > 0).then_some("lora"),
            gen_core::MemoryGeometry {
                width,
                height,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            false,
            false,
            false,
            true,
            budget,
            raw_budget.map_or(0.0, crate::vram_gate::ladder_reserve_gb),
            predicted_peak_gb,
            adapter_resident_bytes,
            if reclaimable_gb > 0.0 {
                gen_core::MemoryCacheState::Warm
            } else {
                gen_core::MemoryCacheState::Cold
            },
        )?;
        (
            evaluation
                .as_ref()
                .and_then(|evaluation| evaluation.memory),
            evaluation.and_then(|evaluation| {
                optimized_shared_memory_context(engine_id, evaluation.context)
            }),
        )
    };
    #[cfg(target_os = "macos")]
    let memory_plan = crate::mlx_fit_gate::MlxRequestPlan::for_spec_and_manifest(
        engine_id,
        &request.model,
        &spec,
        Some(&request.model_manifest_entry),
        None,
    )
    .with_resolved_artifact_tier(resolved_mlx_artifact_tier(&weights_dir, quant_bits))?;
    #[cfg(target_os = "macos")]
    let memory_inputs = crate::mlx_fit_gate::MlxRequestInputs {
        width,
        height,
        count: request.count,
        mode: request.mode.clone(),
        overlay: (adapter_count > 0).then(|| format!("adapters:{adapter_count}")),
        adapter_count,
        has_reference: false,
        reference_count: 0,
        use_pid: false,
        has_phases: true,
    };

    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    let (cancel, rx, blocking) = start_cached_gen_stream(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        format!("{engine_id} load failed"),
        move |generator, tx, cancel| {
            drive_gen_items(tx, work, move |_index, (seed, prompt), preview, on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let (out_w, out_h, pixels) = krea_multiphase_generate_one(
                    generator,
                    &prompt,
                    negative_prompt.clone(),
                    width,
                    height,
                    seed,
                    guidance,
                    text_style_gain,
                    phases.clone(),
                    generation_memory,
                    memory_strategy_context.as_ref(),
                    preview,
                    &cancel,
                    on_progress,
                )?;
                Ok(Some((seed, out_w, out_h, pixels)))
            })
        },
    );
    #[cfg(target_os = "macos")]
    let (cancel, rx, blocking) = start_cached_gen_stream_with_request_state(
        job.id.clone(),
        engine_id,
        adapter_count,
        spec,
        format!("{engine_id} load failed"),
        move |generator,
              cache_state,
              loaded_policy,
              warm_policy,
              external_committed_bytes,
              tx,
              cancel| {
            let mut request_cache_state = cache_state;
            // sc-18317: ONE warm hit is ONE decision, but this lane evaluates the request once per
            // item. Hand the real proposal to the first evaluation and an inert one to the rest, so a
            // multi-image job settles exactly one decision and emits exactly one event.
            let mut warm_policy = crate::execution_planner::WarmPolicyOnce::new(warm_policy);
            drive_gen_items(tx, work, move |_index, (seed, prompt), preview, on_progress| {
                if cancel.is_cancelled() {
                    return Ok(None);
                }
                let memory_evaluation = crate::mlx_fit_gate::evaluate_request(
                    generator,
                    &memory_plan,
                    &memory_inputs,
                    request_cache_state,
                    loaded_policy.offload_policy,
                    warm_policy.take(),
                    external_committed_bytes,
                )?;
                request_cache_state = gen_core::MemoryCacheState::Warm;
                let _request_memory_limit = memory_evaluation
                    .process_limit_bytes
                    .and_then(crate::generator_cache::apply_request_gpu_memory_limit);
                let (out_w, out_h, pixels) = krea_multiphase_generate_one(
                    generator,
                    &prompt,
                    negative_prompt.clone(),
                    width,
                    height,
                    seed,
                    guidance,
                    text_style_gain,
                    phases.clone(),
                    Some(memory_evaluation.memory),
                    Some(&memory_evaluation.context),
                    preview,
                    &cancel,
                    on_progress,
                )?;
                Ok(Some((seed, out_w, out_h, pixels)))
            })
        },
    );

    consume_gen_events(
        api,
        settings,
        job,
        plan,
        project_path,
        backend,
        adapter_label,
        &raw_settings,
        total,
        rx,
        cancel,
        blocking,
        asset_writes,
    )
    .await
}
