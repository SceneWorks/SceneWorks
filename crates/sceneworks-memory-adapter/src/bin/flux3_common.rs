//! Tensor-neutral FLUX3 producer binding and scoped lifecycle, shared by both lanes.
//! Nothing in this module opens a device. Only the lane's `run` loads a registry generator.
use super::*;
#[cfg(not(target_os = "macos"))]
use runtime_cuda::gen_core as core;
#[cfg(target_os = "macos")]
use runtime_macos::gen_core as core;

#[path = "flux3_boundary_self_test.rs"]
pub(crate) mod boundary_self_test;
#[path = "flux3_resource.rs"]
mod resource;

pub(super) const PROVIDER: &str = "flux3_klein";
pub(super) const FLUX3_REPOSITORY: &str = "black-forest-labs/flux-3-image-klein";
pub(super) const REVISION: &str = "8454017f9ebaada30c998fccd9f67a80c7580f68";
pub(super) const SEED: u64 = 26_243;
pub(super) const STEPS: u32 = 16;
pub(super) const BASE_PROVIDER: &str = "flux3_klein_base";
pub(super) const FLUX3_BASE_REPOSITORY: &str = "black-forest-labs/flux-3-image-klein-base";
pub(super) const BASE_REVISION: &str = "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4";

#[derive(Clone, Copy, Debug)]
pub(super) struct Checkpoint {
    pub provider: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
    pub label: &'static str,
    pub steps: u32,
    pub guidance: Option<f32>,
}

fn checkpoint(provider: &str) -> Result<Checkpoint, String> {
    match provider {
        PROVIDER => Ok(Checkpoint {
            provider: PROVIDER,
            repository: FLUX3_REPOSITORY,
            revision: REVISION,
            label: "flux3-klein",
            steps: STEPS,
            guidance: None,
        }),
        BASE_PROVIDER => Ok(Checkpoint {
            provider: BASE_PROVIDER,
            repository: FLUX3_BASE_REPOSITORY,
            revision: BASE_REVISION,
            label: "flux3-klein-base",
            steps: 50,
            guidance: Some(3.5),
        }),
        other => Err(format!("{other}: not a native FLUX3 checkpoint")),
    }
}

pub(super) fn quant(lane: &str, tier: &str) -> Result<Option<Quant>, String> {
    match (lane, tier) {
        ("candle" | "mlx", "bf16") => Ok(None),
        ("candle" | "mlx", "q4") => Ok(Some(Quant::Q4)),
        ("candle" | "mlx", "q8") => Ok(Some(Quant::Q8)),
        ("candle", "fp8r") => Ok(Some(Quant::Fp8Rowwise)),
        ("candle", "nvfp4") => Ok(Some(Quant::Nvfp4)),
        _ => Err(format!(
            "unsupported FLUX3 native tier {lane}:{tier}; no dense fallback"
        )),
    }
}

pub(super) fn conversion(lane: &str, tier: &str) -> Option<&'static str> {
    match (lane, tier) {
        ("candle", "q4") => Some("flux3-candle-ggml-q4_1-v1"),
        ("candle", "q8") => Some("flux3-candle-ggml-q8_0-v1"),
        ("mlx", "q4") => Some("flux3-mlx-affine-q4-g64-v1"),
        ("mlx", "q8") => Some("flux3-mlx-affine-q8-g64-v1"),
        _ => None,
    }
}

#[cfg(test)]
pub(super) fn fingerprint(lane: &str, tier: &str) -> Result<String, String> {
    fingerprint_for(lane, tier, PROVIDER)
}

pub(super) fn fingerprint_for(lane: &str, tier: &str, provider: &str) -> Result<String, String> {
    let checkpoint = checkpoint(provider)?;
    quant(lane, tier)?;
    let suffix = match (lane, tier) {
        (_, "bf16") => "",
        ("candle", "q4") => "-ggml-q4-1",
        ("candle", "q8") => "-ggml-q8-0",
        ("mlx", "q4") => "-mlx-affine-q4-g64",
        ("mlx", "q8") => "-mlx-affine-q8-g64",
        (_, "fp8r") => "-fp8r",
        (_, "nvfp4") => "-nvfp4",
        _ => unreachable!("quant validated the tier"),
    };
    let base = if checkpoint.provider == BASE_PROVIDER {
        // Exact native base contract identities, NOT a distilled calibration alias.
        format!("flux3-klein-base-{lane}-cfg-source-v1")
    } else {
        format!("flux3-klein-{lane}-derived-2026-10-15-v1")
    };
    Ok(format!("{base}{suffix}"))
}

pub(super) struct Plan {
    pub lane: String,
    pub checkpoint: Checkpoint,
    pub tier: String,
    pub fingerprint: String,
    pub selection: MemorySelection,
    resource_case: Option<resource::BoundCase>,
}

pub(super) fn plan(request: &Value, lane: &str) -> Result<Plan, String> {
    let planned = protocol::planned(request)?;
    let checkpoint = checkpoint(
        planned
            .pointer("/target/provider")
            .and_then(Value::as_str)
            .ok_or_else(|| "FLUX3 target.provider must name a native checkpoint".to_owned())?,
    )?;
    protocol::validate_still_geometry(request, "FLUX3 calibration")?;
    protocol::validate_plain_overlay_target(request, "the native FLUX3 text-to-image path")?;
    for (pointer, expected) in [
        ("/backend", lane),
        ("/target/provider", checkpoint.provider),
        ("/target/modelId", checkpoint.provider),
        ("/target/mode", "text_to_image"),
        ("/loadShape", protocol::LOAD_SHAPE_EAGER),
    ] {
        if planned.pointer(pointer).and_then(Value::as_str) != Some(expected) {
            return Err(format!("FLUX3 planned{pointer} must be {expected:?}"));
        }
    }
    if protocol::target_geometry(request)? != (1024, 1024) {
        return Err(
            "FLUX3 producer measures only its checkpoint-specific 1024x1024 seed26243 fixture"
                .to_owned(),
        );
    }
    let tier = planned
        .pointer("/target/tier")
        .and_then(Value::as_str)
        .ok_or_else(|| "FLUX3 planned.target.tier must be a string".to_owned())?;
    let fingerprint = fingerprint_for(lane, tier, checkpoint.provider)?;
    let fixture = format!(
        "{}-{lane}-{tier}-1024-seed26243-step{}",
        checkpoint.label, checkpoint.steps
    );
    if planned.get("fixture").and_then(Value::as_str) != Some(fixture.as_str()) {
        return Err(format!("FLUX3 planned.fixture must be {fixture:?}"));
    }
    if planned
        .get("calibrationFingerprint")
        .and_then(Value::as_str)
        != Some(fingerprint.as_str())
    {
        return Err(format!(
            "FLUX3 plan/provider calibration mismatch: expected {fingerprint}"
        ));
    }
    if !protocol::strategy_parameters(request)?.is_empty() {
        return Err("FLUX3 anchor producer accepts no unmeasured strategy parameters".to_owned());
    }
    let selection = MemorySelection {
        strategy: planned_memory_strategy(request)?,
        parameters: MemoryStrategyParameters::default(),
        tier: MemoryNumericTier {
            precision: Precision::Bf16,
            quant: quant(lane, tier)?,
            component_precision_floors: &[],
        },
    };
    let anchor = if lane == "candle" {
        MemoryStrategy::StagedResidency
    } else {
        MemoryStrategy::Resident
    };
    if selection.strategy != anchor {
        return Err(format!(
            "FLUX3 {lane} producer measures {anchor:?}, not {:?}",
            selection.strategy
        ));
    }
    let resource_case = request
        .get("resourceCase")
        .map(|case| -> Result<resource::BoundCase, String> {
            let root = protocol::required_env("SCENEWORKS_FLUX3_RESOURCE_INPUT_ROOT")?;
            let bound = resource::bind(case, lane, checkpoint, tier, std::path::Path::new(&root))?;
            Ok(bound)
        })
        .transpose()?;
    Ok(Plan {
        lane: lane.to_owned(),
        checkpoint,
        tier: tier.to_owned(),
        fingerprint,
        selection,
        resource_case,
    })
}

/// The harness passes the two actual roots, not a guessed snapshot/<tier>. Packed
/// outputs own the DiT only; both external components ALWAYS come from the original.
pub(super) fn spec(lane: &str, plan: &Plan) -> Result<LoadSpec, String> {
    let repo = protocol::required_env("SCENEWORKS_FLUX3_KLEIN_REPOSITORY")?;
    let rev = protocol::required_env("SCENEWORKS_FLUX3_KLEIN_REVISION")?;
    if repo != plan.checkpoint.repository || rev != plan.checkpoint.revision {
        return Err(format!(
            "FLUX3 requires pinned original {}@{}, got {repo}@{rev}",
            plan.checkpoint.repository, plan.checkpoint.revision
        ));
    }
    let root = |name| -> Result<PathBuf, String> {
        let raw = protocol::required_env(name).map_err(|error| {
            format!("weights_missing: {error}; run the local-install harness probe")
        })?;
        std::fs::canonicalize(&raw).map_err(|error| {
            format!("weights_missing: {name}={raw}: {error}; provision this exact native package")
        })
    };
    let original = root("SCENEWORKS_FLUX3_KLEIN_ORIGINAL_ROOT")?;
    protocol::validate_huggingface_revision_root(
        &original,
        &repo,
        &rev,
        plan.checkpoint.repository,
    )?;
    let package = root("SCENEWORKS_FLUX3_KLEIN_ROOT")?;
    validate_package_root(lane, &plan.tier, &original, &package)?;
    for file in [
        package.join("config.json"),
        package.join("model.safetensors"),
        original.join("ae.safetensors"),
        original.join("text_encoder/config.json"),
        original.join("text_encoder/tokenizer.json"),
        original.join("text_encoder/tokenizer_config.json"),
    ] {
        if !file.is_file() {
            return Err(format!("weights_missing: FLUX3 requires {}; install this exact package and its original external TE/AE", file.display()));
        }
    }
    if conversion(lane, &plan.tier).is_some()
        && !package.join("flux3-conversion-receipt.json").is_file()
    {
        return Err(format!(
            "weights_missing: {} has no native conversion receipt; derive this exact backend tier",
            package.display()
        ));
    }
    if !original.join("text_encoder/model.safetensors").is_file()
        && !original
            .join("text_encoder/model.safetensors.index.json")
            .is_file()
    {
        return Err(format!(
            "weights_missing: {} requires its text encoder single-file weights or shard index",
            original.display()
        ));
    }
    let mut spec = load_spec(lane, &plan.tier, package, original)?;
    if let Some(case) = &plan.resource_case {
        spec.adapters = case.adapters.clone();
        if let Some(root) = &case.pup_root {
            spec.components.insert(
                core::flux3::pup::UPSAMPLER_COMPONENT.into(),
                WeightsSource::Dir(root.clone()),
            );
        }
    }
    Ok(spec)
}

pub(super) fn validate_package_root(
    lane: &str,
    tier: &str,
    original: &std::path::Path,
    package: &std::path::Path,
) -> Result<(), String> {
    quant(lane, tier)?;
    let valid = if let Some(id) = conversion(lane, tier) {
        package.ends_with(std::path::Path::new(tier).join(id)) && package != original
    } else if tier == "bf16" {
        package == original
    } else {
        package == original.join("variants").join(tier)
    };
    if !valid {
        return Err(format!("FLUX3 {lane}:{tier} package {} is not the exact native install (conversion {:?}); no snapshot tier-dir or dense substitution", package.display(), conversion(lane, tier)));
    }
    Ok(())
}

pub(super) fn load_spec(
    lane: &str,
    tier: &str,
    package: PathBuf,
    original: PathBuf,
) -> Result<LoadSpec, String> {
    let mut spec = LoadSpec::new(WeightsSource::Dir(package))
        .with_offload_policy(if lane == "candle" {
            OffloadPolicy::Sequential
        } else {
            OffloadPolicy::Resident
        })
        .with_load_shape(LoadShape::EagerMaterialization)
        .with_component(
            "text_encoder",
            WeightsSource::Dir(original.join("text_encoder")),
        )
        .with_component(
            "autoencoder",
            WeightsSource::File(original.join("ae.safetensors")),
        );
    spec.precision = Precision::Bf16;
    spec.quantize = quant(lane, tier)?;
    Ok(spec)
}

/// Snapshot the SAME borrowed object immediately before catalog.load. No reconstruction from
/// caller intent and no physical-device assertion: LoadSpec does not expose that observation.
pub(super) fn actual_spec_observation(plan: &Plan, spec: &LoadSpec) -> Result<Value, String> {
    use sha2::{Digest, Sha256};
    let expected_policy = if plan.lane == "candle" {
        OffloadPolicy::Sequential
    } else {
        OffloadPolicy::Resident
    };
    if spec.precision != Precision::Bf16
        || spec.quantize != quant(&plan.lane, &plan.tier)?
        || spec.offload_policy != expected_policy
        || spec.load_shape != LoadShape::EagerMaterialization
    {
        return Err("BLOCKED: actual LoadSpec differs from admitted native tier/residency".into());
    }
    fn source(value: &WeightsSource) -> Value {
        match value {
            WeightsSource::Dir(path) => json!({"kind":"Dir","path":path}),
            WeightsSource::File(path) => json!({"kind":"File","path":path}),
        }
    }
    let adapters = |values: &[core::AdapterSpec]| {
        values
            .iter()
            .map(|a| {
                json!({"path":a.path,"scale":a.scale,"kind":format!("{:?}",a.kind),
            "passScales":a.pass_scales,"moeExpert":format!("{:?}",a.moe_expert)})
            })
            .collect::<Vec<_>>()
    };
    let intended_adapters = plan
        .resource_case
        .as_ref()
        .map(|c| adapters(&c.adapters))
        .unwrap_or_default();
    if adapters(&spec.adapters) != intended_adapters {
        return Err("BLOCKED: actual LoadSpec adapters differ from bound request".into());
    }
    let expected_components = 2 + usize::from(
        plan.resource_case
            .as_ref()
            .is_some_and(|c| c.pup_root.is_some()),
    );
    if spec.components.len() != expected_components
        || !spec.components.contains_key("text_encoder")
        || !spec.components.contains_key("autoencoder")
    {
        return Err(
            "BLOCKED: actual LoadSpec component inventory differs from bound request".into(),
        );
    }
    if let Some(root) = plan
        .resource_case
        .as_ref()
        .and_then(|c| c.pup_root.as_ref())
    {
        if spec
            .components
            .get(core::flux3::pup::UPSAMPLER_COMPONENT)
            .map(source)
            != Some(json!({"kind":"Dir","path":root}))
        {
            return Err("BLOCKED: actual PUP LoadSpec differs from bound inventory".into());
        }
    }
    let components = spec
        .components
        .iter()
        .map(|(name, value)| (name.clone(), source(value)))
        .collect::<serde_json::Map<_, _>>();
    let mut value = json!({"schemaVersion":1,"provenanceKind":"actual_borrowed_load_spec_before_catalog_load",
        "backend":plan.lane,"provider":plan.checkpoint.provider,"tier":plan.tier,"weights":source(&spec.weights),
        "components":components,"adapters":adapters(&spec.adapters),"precision":format!("{:?}",spec.precision),
        "quantization":format!("{:?}",spec.quantize),"offloadPolicy":format!("{:?}",spec.offload_policy),
        "loadShape":format!("{:?}",spec.load_shape),"nativeDebugFormatVersion":1,
        "nativeDebugSha256":format!("{:x}",Sha256::digest(format!("{spec:?}").as_bytes()))});
    value["sha256"] = json!(resource::json_digest(
        b"flux3_actual_load_spec_v1\0",
        &value
    )?);
    Ok(value)
}

pub(super) fn attach_loaded_spec(
    mut fragment: Value,
    actual: Value,
    generator: &dyn Generator,
) -> Result<Value, String> {
    if fragment["evidenceKind"].as_str().is_some_and(|kind| {
        kind == "flux3_supplemental_resource_observation_v1"
            || kind == "flux3_native_resource_capture_envelope_v1"
    }) {
        fragment["actualLoadSpec"] = actual;
        fragment["loadedGenerator"] =
            json!({"provider":generator.descriptor().id,"backend":generator.descriptor().backend});
        let invocation = fragment
            .pointer("/nativeOutput/reservation/invocationId")
            .and_then(Value::as_str)
            .ok_or_else(|| "resource capture has no actual invocation receipt".to_owned())?
            .to_owned();
        fragment["captureId"] = json!(invocation);
        fragment["captureBindingSha256"] = json!(resource::json_digest(
            b"flux3_native_capture_binding_v1\0",
            &fragment
        )?);
    }
    Ok(fragment)
}

pub(super) fn context(plan: &Plan, abi: u32, hardware_bytes: u64) -> MemoryRunContext {
    let mut context = MemoryRunContext {
        selection: plan.selection,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: abi,
        calibration_fingerprint: plan.fingerprint.clone(),
        load_shape: LoadShape::EagerMaterialization,
        mode: MemoryMode::TextToImage,
        has_reference: false,
        reference_resources: None,
        use_pid: false,
        has_phases: true,
        geometry: MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        overlay: None,
        budget: MemoryBudget {
            total_bytes: hardware_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        // This is a capture request, NOT an estimated peak or synthetic measurement.
        // Admission's zero-budget and fitting-control probes are recorded separately.
        predicted_peak_bytes: 1,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: format!("sc-26241@{}", protocol::INFERENCE_PIN),
    };
    if let Some(case) = &plan.resource_case {
        let count = case.request.image_reference_count();
        context.mode = if count == 0 {
            MemoryMode::TextToImage
        } else {
            MemoryMode::Edit
        };
        context.has_reference = count > 0;
        context.geometry.width = case.request.width;
        context.geometry.height = case.request.height;
        context.geometry.reference_count = count;
        context.reference_resources = case.reference_resources.clone();
        context.overlay = (!case.adapters.is_empty()).then(|| "lora".to_owned());
        context.evidence_revision =
            format!("supplemental:{}:{}", case.value["caseId"], case.digest);
    }
    context
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum BoundaryError {
    MissingPreDenoise,
    InvalidZeroStep,
}
impl std::fmt::Display for BoundaryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::MissingPreDenoise => "missing_phase_boundary: native FLUX3 must emit typed ConditioningComplete after materialized, stream-fenced positive/negative/reference conditioning and before the first DiT forward; neither Renderer loading nor completed Step1 can label conditioning",
            Self::InvalidZeroStep => "invalid_phase_boundary: Progress::Step is 1-based; Step0 is not a conditioning-completion event",
        })
    }
}
impl std::error::Error for BoundaryError {}

/// Reducer for the native typed ConditioningComplete event, shared by both capture
/// adapters. Duplicate completion never resets a phase's peak or reopens conditioning.
pub(super) fn conditioning_complete(
    phase: &mut protocol::ReferencePhase,
) -> Option<protocol::ReferencePhase> {
    if *phase != protocol::ReferencePhase::Conditioning {
        return None;
    }
    *phase = protocol::ReferencePhase::Denoise;
    Some(*phase)
}

/// Cross only real pre-denoise boundaries, never completed solver Step1. Older
/// pins without the typed completion event fail rather than mislabeling phases.
pub(super) fn boundary(
    phase: &mut protocol::ReferencePhase,
    progress: &Progress,
) -> Result<Option<protocol::ReferencePhase>, BoundaryError> {
    if matches!(progress, Progress::Step { current: 0, .. }) {
        return Err(BoundaryError::InvalidZeroStep);
    }
    // Renderer loading can precede reference encoding. Only the provider's typed,
    // synchronized completion covers ALL conditioning on resident and staged routes.
    if matches!(progress, Progress::ConditioningComplete) {
        return Ok(conditioning_complete(phase));
    }
    if *phase == protocol::ReferencePhase::Conditioning
        && matches!(
            progress,
            Progress::Step { current: 1.., .. } | Progress::Decoding
        )
    {
        return Err(BoundaryError::MissingPreDenoise);
    }
    let reference = match progress {
        Progress::Decoding => protocol::ReferenceBoundary::Decoding,
        _ => return Ok(None),
    };
    let Some(next) = protocol::next_reference_phase(*phase, reference) else {
        return Ok(None);
    };
    *phase = next;
    Ok(Some(next))
}

fn generation_request(plan: &Plan) -> GenerationRequest {
    if let Some(case) = &plan.resource_case {
        return case.request.clone();
    }
    GenerationRequest {
        prompt: "a portrait of a person in a sunlit studio, editorial photograph".to_owned(),
        width: 1024,
        height: 1024,
        count: 1,
        seed: Some(SEED),
        steps: Some(plan.checkpoint.steps),
        guidance: plan.checkpoint.guidance,
        ..Default::default()
    }
}

/// Identity of the returned native RGB8 host carrier, not a saved asset or quality claim.
fn output_receipt(output: &GenerationOutput, width: u32, height: u32) -> Result<Value, String> {
    use sha2::{Digest, Sha256};
    let GenerationOutput::Images(images) = output else {
        return Err("FLUX3 did not return an image carrier".to_owned());
    };
    if images.len() != 1 {
        return Err("FLUX3 did not return exactly one real image".to_owned());
    }
    let image = &images[0];
    let expected = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| "FLUX3 RGB8 byte length overflow".to_owned())?;
    if image.width != width || image.height != height || image.pixels.len() as u64 != expected {
        return Err("FLUX3 returned image geometry/byte length differs from request".to_owned());
    }
    let mut hash = Sha256::new();
    hash.update(b"flux3_native_output_v1\0Images\0RGB8\0interleaved_host_u8\0");
    hash.update(image.width.to_le_bytes());
    hash.update(image.height.to_le_bytes());
    hash.update(expected.to_le_bytes());
    hash.update(&image.pixels);
    Ok(
        json!({"schemaVersion":1,"kind":"Images","width":image.width,"height":image.height,
        "format":"RGB8","representation":"interleaved_host_u8","bytes":expected,
        "sha256":format!("{:x}",hash.finalize()),"identityDomain":"flux3_native_output_v1"}),
    )
}

pub(super) fn render(
    plan: &Plan,
    generator: &dyn Generator,
    context: &MemoryRunContext,
    observed: &mut dyn FnMut(protocol::ReferencePhase),
) -> Result<Value, String> {
    render_guarded(plan, generator, context, observed, &mut resource::reserve)
}

fn render_guarded(
    plan: &Plan,
    generator: &dyn Generator,
    context: &MemoryRunContext,
    observed: &mut dyn FnMut(protocol::ReferencePhase),
    reserve: &mut dyn FnMut(&Plan, &GenerationRequest) -> Result<Value, String>,
) -> Result<Value, String> {
    let mut request = generation_request(plan);
    let mut scope = generator
        .begin_memory_strategy_request(context)
        .map_err(|error| format!("begin FLUX3 memory request: {error}"))?
        .ok_or_else(|| "FLUX3 provider did not open its native memory request scope".to_owned())?;
    // Every post-open exit finishes the scope, including configure/enter errors.
    let mut canceled = false;
    let result = (|| {
        scope
            .configure_request(&mut request)
            .map_err(|error| error.to_string())?;
        scope
            .enter_phase(MemoryPhase::Conditioning)
            .map_err(|error| error.to_string())?;
        let mut phase = protocol::ReferencePhase::Conditioning;
        let mut phase_error = None;
        let cancel = request.cancel.clone();
        let reservation = reserve(plan, &request)?;
        let mut pup_pending = request.enhance_prompt;
        let mut pup_complete = false;
        let output = generator.generate(&request, &mut |progress| {
            if phase_error.is_some() {
                return;
            }
            if matches!(progress, Progress::PromptUpsamplingComplete) {
                if pup_pending {
                    pup_pending = false;
                    pup_complete = true;
                    // Observation only: keep the native scope's complete conditioning envelope.
                    observed(protocol::ReferencePhase::Conditioning);
                } else if !pup_complete {
                    phase_error = Some("unexpected_phase_boundary: PUP bypass emitted completion".to_owned());
                    cancel.cancel();
                }
                return;
            }
            if pup_pending && matches!(progress, Progress::Loading(core::LoadPhase::TextEncoder)
                | Progress::ConditioningComplete | Progress::Step { .. } | Progress::Decoding) {
                phase_error = Some("missing_phase_boundary: PromptUpsamplingComplete required before text conditioning".to_owned());
                cancel.cancel();
                return;
            }
            let next = match boundary(&mut phase, &progress) {
                Ok(next) => next,
                Err(error) => {
                    phase_error = Some(error.to_string());
                    cancel.cancel();
                    return;
                }
            };
            if let Some(next) = next {
                observed(next);
                let (before, after) = match next {
                    protocol::ReferencePhase::Denoise => {
                        (MemoryPhase::Conditioning, MemoryPhase::Denoise)
                    }
                    protocol::ReferencePhase::Decode => (MemoryPhase::Denoise, MemoryPhase::Decode),
                    _ => return,
                };
                if phase_error.is_none() {
                    if let Err(error) = scope
                        .leave_phase(before)
                        .and_then(|_| scope.enter_phase(after))
                    {
                        phase_error = Some(error.to_string());
                    }
                }
            }
        });
        if let Some(error) = phase_error {
            return Err(error);
        }
        canceled = matches!(&output, Err(core::Error::Canceled));
        let output = output.map_err(|error| format!("native FLUX3 generation failed: {error}"))?;
        if phase != protocol::ReferencePhase::Decode {
            return Err("FLUX3 generation omitted a measured phase boundary".to_owned());
        }
        scope
            .leave_phase(MemoryPhase::Decode)
            .map_err(|error| error.to_string())?;
        let mut receipt = output_receipt(&output, request.width, request.height)?;
        receipt["reservation"] = reservation;
        receipt["actualGenerationRequest"] = resource::actual_request_observation(&request);
        receipt["promptUpsampling"] = json!({"requested":request.enhance_prompt,
            "typedCompletionObserved":pup_complete,"positiveEnhancementAttested":false,
            "scope":"whole-request residency/baseline; completion includes honest fallback, not positive enhancement proof"});
        Ok(receipt)
    })();
    let outcome = match &result {
        Ok(_) => MemoryRunOutcome::Complete,
        Err(_) if canceled => MemoryRunOutcome::Canceled,
        Err(message) => MemoryRunOutcome::Error {
            message: message.clone(),
        },
    };
    let finished = scope
        .finish(outcome)
        .map_err(|error| format!("finish FLUX3 memory scope: {error}"));
    match (result, finished) {
        (Ok(receipt), Ok(())) => Ok(receipt),
        (Ok(_), Err(error)) => Err(error),
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(finish)) => Err(format!("{error}; {finish}")),
    }
}

/// Admission controls against the loaded native generator, never a copied fit formula.
pub(super) fn admit(
    generator: &dyn Generator,
    plan: &Plan,
    request: &Value,
) -> Result<MemoryRunContext, String> {
    generator
        .validate(&generation_request(plan))
        .map_err(|error| format!("BLOCKED: native FLUX3 request validation: {error}"))?;
    let contract = generator
        .memory_strategy_contract()
        .ok_or_else(|| "loaded FLUX3 provider exposes no memory contract".to_owned())?;
    if contract.provider_id != plan.checkpoint.provider {
        return Err(format!(
            "FLUX3 selected {} but loaded {}",
            plan.checkpoint.provider, contract.provider_id
        ));
    }
    contract
        .validate_selection(&plan.selection)
        .map_err(|error| error.to_string())?;
    let calibration = contract
        .calibration
        .as_ref()
        .ok_or_else(|| "loaded FLUX3 provider exposes no calibration identity".to_owned())?;
    if calibration.fingerprint != plan.fingerprint
        || calibration.load_shape != LoadShape::EagerMaterialization
    {
        return Err(format!(
            "FLUX3 plan/provider calibration mismatch: planned {}, loaded {:?}",
            plan.fingerprint, calibration
        ));
    }
    let hardware = request
        .pointer("/hardware/memoryBytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| "FLUX3 request.hardware.memoryBytes must be an integer".to_owned())?;
    let context = context(plan, calibration.abi, hardware);
    if !matches!(
        generator.memory_strategy_safety_check(&context),
        MemorySafetyDecision::Accept
    ) {
        return Err(
            "FLUX3 admission refused the fitting control; no capture is attested".to_owned(),
        );
    }
    let mut unknown = context.clone();
    unknown.budget.total_bytes = 0;
    if !matches!(
        generator.memory_strategy_safety_check(&unknown),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("FLUX3 admission accepted an unknown budget".to_owned());
    }
    let mut stale = context.clone();
    stale.calibration_fingerprint.push_str("-mutation");
    if !matches!(
        generator.memory_strategy_safety_check(&stale),
        MemorySafetyDecision::Reject { .. }
    ) {
        return Err("FLUX3 admission accepted a mutated calibration identity".to_owned());
    }
    Ok(context)
}

pub(super) fn fragment(
    request: &Value,
    lane: &str,
    plan: &Plan,
    strategy: Value,
    observed: Value,
    diagnostics: Value,
    output: Value,
) -> Result<Value, String> {
    if let Some(case) = &plan.resource_case {
        return Ok(json!({
            "schemaVersion":1,"evidenceKind":"flux3_supplemental_resource_observation_v1",
            "anchorEligible":false,"caseId":case.value["caseId"],"status":"OBSERVED",
            "resourceCase":case.value,"requestSha256":case.digest,"inputInventory":case.inputs,
            "nativeOutput":output,
            "actualRequest":{
                "prompt":case.request.prompt,"width":case.request.width,"height":case.request.height,
                "count":case.request.count,"seed":case.request.seed,"steps":case.request.steps,
                "guidance":case.request.guidance,"enhancePrompt":case.request.enhance_prompt,
                "referenceCount":case.request.image_reference_count(),
                "layout":case.value["generation"].get("layout"),
                "adapters":case.value["adapters"],
            },
            "artifact":{"repository":plan.checkpoint.repository,"resolvedRevision":plan.checkpoint.revision,"variant":plan.tier},
            "componentInventory":resource::model_inventory()?,
            "coverage":{
                "wholeRequestCheckpoint":plan.checkpoint.provider,"crossCheckpointPeakInheritance":false,
                "imagePrecision":"bf16","imageQuantization":plan.tier,
                "residencyPolicy":if lane=="candle" {"sequential"} else {"resident"},
                "upsamplerPrecisionPolicy":"native-package-validated-no-override",
                "scope":"only this exact input/layout/adapter/component inventory and request; auxiliary profiles never imply distilled whole-request peaks"
            },
            "observedMemory":observed,"strategy":strategy,"diagnostics":diagnostics,
            "captureRequest":request,"inferencePin":protocol::INFERENCE_PIN,
            "capturedAt":protocol::captured_at(),
            "notAttested":["saved product assets","quality acceptance","canonical calibration anchor","warm repeat/cancellation/error scenarios"],
        }));
    }
    let mut observed = observed;
    if let Some(metrics) = observed.as_object_mut() {
        metrics.remove("promptUpsampling");
    }
    let blocker = "native FLUX3 anchor producer: real scoped generation and synchronized phase observations only; promotion quality, repeat/cancel/error lifecycle scenarios and fault injection have NOT been attested by this capture";
    let mut fragment = json!({
        "status": "gated", "strategy": strategy,
        "loadShape": protocol::LOAD_SHAPE_EAGER,
        "artifact": { "repository": plan.checkpoint.repository, "resolvedRevision": plan.checkpoint.revision, "variant": plan.tier },
        "sweep": protocol::reference_sweep(request, "passed")?,
        "scenarios": [
            { "name": "exact_fit", "result": "not_run", "reason": blocker },
            { "name": "unknown_budget", "result": "passed" },
            { "name": "stale_evidence", "result": "passed" },
            { "name": "warm_repeat", "result": "not_run", "reason": blocker },
            { "name": "cancel", "result": "not_run", "reason": blocker },
            { "name": "error", "result": "not_run", "reason": blocker },
            { "name": "loadability", "result": "passed" },
            { "name": "overlay", "result": "not_applicable", "reason": "settled below" }
        ],
        "predictedPeakBytes": null, "observedMemory": observed,
        "quality": { "result": "not_run" }, "negativeMutation": null,
        "loadability": { "result": "passed", "resolvedPathFingerprint": format!("{}@{}:{}:{}:external-original-te-ae", plan.checkpoint.repository, plan.checkpoint.revision, plan.tier, conversion(lane, &plan.tier).unwrap_or("publisher")) },
        "diagnostics": diagnostics, "capturedAt": protocol::captured_at(),
    });
    protocol::settle_plain_overlay_scenario(
        request,
        &mut fragment,
        "the native FLUX3 text-to-image path",
    )?;
    if request["resourceEnvelope"] == true {
        // Accounting/spec/output evidence is NEVER inserted into a canonical anchor fragment.
        return Ok(
            json!({"schemaVersion":1,"evidenceKind":"flux3_native_resource_capture_envelope_v1",
            "anchorEligible":false,"caseId":output["reservation"]["caseId"],
            "requestSha256":output["reservation"]["requestSha256"],
            "anchorFragmentSha256":resource::json_digest(b"flux3_canonical_anchor_fragment_v1\0", &fragment)?,
            "anchorFragment":fragment,"nativeOutput":output,
            "componentInventory":resource::model_inventory()?}),
        );
    }
    Ok(fragment)
}

#[cfg(test)]
mod tests {
    use super::core::LoadPhase;
    use super::*;

    // Explicit CPU fixture seam; production render ALWAYS uses the real atomic ledger helper.
    fn render(
        plan: &Plan,
        generator: &dyn Generator,
        context: &MemoryRunContext,
        observed: &mut dyn FnMut(protocol::ReferencePhase),
    ) -> Result<Value, String> {
        render_guarded(plan, generator, context, observed, &mut |_, _| {
            Ok(json!({"fixtureOnly":true}))
        })
    }

    fn request(lane: &str, tier: &str) -> Value {
        json!({ "planned": {
            "backend": lane,
            "target": { "provider": PROVIDER, "modelId": PROVIDER, "tier": tier, "mode": "text_to_image", "overlay": "none", "geometry": { "width": 1024, "height": 1024, "batch": 1, "frames": 1 } },
            "loadShape": protocol::LOAD_SHAPE_EAGER,
            "calibrationFingerprint": fingerprint(lane, tier).unwrap(),
            "fixture": format!("flux3-klein-{lane}-{tier}-1024-seed26243-step16"),
            "strategy": { "rung": if lane == "candle" { "staged_residency" } else { "resident" }, "parameters": {} }
        } })
    }

    #[test]
    fn flux3_plan_validation_is_exact_and_precedes_any_weights_or_device() {
        for lane in ["candle", "mlx"] {
            for tier in ["bf16", "q4", "q8"] {
                let valid = request(lane, tier);
                assert_eq!(
                    plan(&valid, lane).unwrap().selection.tier.quant,
                    quant(lane, tier).unwrap()
                );
                for (pointer, mutation) in [
                    ("/planned/target/mode", json!("edit_image")),
                    ("/planned/target/overlay", json!("lora")),
                    ("/planned/target/geometry/batch", json!(2)),
                    ("/planned/loadShape", json!("deferred_materialization")),
                    ("/planned/fixture", json!("wrong-seed")),
                    ("/planned/calibrationFingerprint", json!("wrong-tier")),
                ] {
                    let mut invalid = valid.clone();
                    *invalid.pointer_mut(pointer).unwrap() = mutation;
                    assert!(plan(&invalid, lane).is_err(), "accepted mutation {pointer}");
                }
            }
        }
        for tier in ["fp8r", "nvfp4"] {
            assert_eq!(
                plan(&request("candle", tier), "candle")
                    .unwrap()
                    .selection
                    .tier
                    .quant,
                quant("candle", tier).unwrap()
            );
        }
    }

    #[test]
    fn flux3_phase_boundaries_follow_native_staged_and_resident_progress() {
        for staged in [false, true] {
            let mut phase = protocol::ReferencePhase::Conditioning;
            assert_eq!(
                boundary(&mut phase, &Progress::Loading(LoadPhase::TextEncoder)),
                Ok(None)
            );
            if staged {
                assert_eq!(
                    boundary(&mut phase, &Progress::Loading(LoadPhase::Renderer)),
                    Ok(None),
                    "reference encoding can still follow Renderer loading"
                );
            }
            assert_eq!(phase, protocol::ReferencePhase::Conditioning);
            assert_eq!(
                boundary(&mut phase, &Progress::ConditioningComplete),
                Ok(Some(protocol::ReferencePhase::Denoise))
            );
            assert_eq!(
                boundary(&mut phase, &Progress::ConditioningComplete),
                Ok(None),
                "duplicate completion must not reset peaks"
            );
            for current in 1..=STEPS {
                assert_eq!(
                    boundary(
                        &mut phase,
                        &Progress::Step {
                            current,
                            total: STEPS
                        }
                    ),
                    Ok(None)
                );
                assert_eq!(phase, protocol::ReferencePhase::Denoise);
            }
            assert_eq!(
                boundary(&mut phase, &Progress::Decoding),
                Ok(Some(protocol::ReferencePhase::Decode))
            );
            assert_eq!(
                boundary(&mut phase, &Progress::ConditioningComplete),
                Ok(None)
            );
            assert_eq!(phase, protocol::ReferencePhase::Decode);
        }
    }

    #[test]
    fn flux3_missing_or_late_boundary_and_zero_steps_fail_closed() {
        for staged in [false, true] {
            for current in [1, STEPS] {
                let mut phase = protocol::ReferencePhase::Conditioning;
                if staged {
                    assert_eq!(
                        boundary(&mut phase, &Progress::Loading(LoadPhase::Renderer)),
                        Ok(None)
                    );
                }
                assert_eq!(
                    boundary(
                        &mut phase,
                        &Progress::Step {
                            current,
                            total: STEPS
                        }
                    ),
                    Err(BoundaryError::MissingPreDenoise)
                );
                assert_eq!(
                    boundary(&mut phase, &Progress::Decoding),
                    Err(BoundaryError::MissingPreDenoise)
                );
                assert_eq!(phase, protocol::ReferencePhase::Conditioning);
            }
        }
        for mut phase in [
            protocol::ReferencePhase::Conditioning,
            protocol::ReferencePhase::Denoise,
            protocol::ReferencePhase::Decode,
        ] {
            assert_eq!(
                boundary(
                    &mut phase,
                    &Progress::Step {
                        current: 0,
                        total: STEPS
                    }
                ),
                Err(BoundaryError::InvalidZeroStep)
            );
        }
    }

    /// Scripted tensor-neutral provider: exercise the real render callback and scope
    /// cleanup without a registry, weights, allocator, or device.
    struct ScriptedGenerator {
        descriptor: core::ModelDescriptor,
        events: Vec<Progress>,
        cancel_at_end: bool,
        lifecycle: std::cell::RefCell<Vec<String>>,
        outcomes: std::cell::RefCell<Vec<MemoryRunOutcome>>,
        cancel_seen: std::cell::Cell<bool>,
        generate_calls: std::cell::Cell<usize>,
        sampling: std::cell::RefCell<Option<(Option<u32>, Option<f32>)>>,
    }

    impl ScriptedGenerator {
        fn new(events: Vec<Progress>, cancel_at_end: bool) -> Self {
            Self {
                descriptor: core::ModelDescriptor {
                    encoder_contract: None,
                    denoiser_output_latent_space: None,
                    id: "flux3-capture-script",
                    family: "flux3",
                    backend: "script",
                    modality: core::Modality::Image,
                    capabilities: core::Capabilities::default(),
                    required_components: &[],
                    control_kinds: None,
                },
                events,
                cancel_at_end,
                lifecycle: Default::default(),
                outcomes: Default::default(),
                cancel_seen: Default::default(),
                generate_calls: Default::default(),
                sampling: Default::default(),
            }
        }
    }

    struct ScriptedScope<'a>(&'a ScriptedGenerator);

    impl core::MemoryRequestScope for ScriptedScope<'_> {
        fn configure_request(&mut self, _: &mut GenerationRequest) -> core::Result<()> {
            Ok(())
        }
        fn enter_phase(&mut self, phase: MemoryPhase) -> core::Result<()> {
            self.0
                .lifecycle
                .borrow_mut()
                .push(format!("enter:{phase:?}"));
            Ok(())
        }
        fn leave_phase(&mut self, phase: MemoryPhase) -> core::Result<()> {
            self.0
                .lifecycle
                .borrow_mut()
                .push(format!("leave:{phase:?}"));
            Ok(())
        }
        fn configure_decode(&mut self, _: u32, _: u32, _: MemoryGeometry) -> core::Result<()> {
            Ok(())
        }
        fn configure_attention(&mut self, _: u32) -> core::Result<()> {
            Ok(())
        }
        fn materialize_transformer_window(&mut self, _: u32, _: u32) -> core::Result<()> {
            Ok(())
        }
        fn finish(&mut self, outcome: MemoryRunOutcome) -> core::Result<()> {
            self.0.outcomes.borrow_mut().push(outcome);
            Ok(())
        }
    }

    impl Generator for ScriptedGenerator {
        fn descriptor(&self) -> &core::ModelDescriptor {
            &self.descriptor
        }
        fn validate(&self, _: &GenerationRequest) -> core::Result<()> {
            Ok(())
        }
        fn begin_memory_strategy_request(
            &self,
            _: &MemoryRunContext,
        ) -> core::Result<Option<Box<dyn core::MemoryRequestScope + '_>>> {
            Ok(Some(Box::new(ScriptedScope(self))))
        }
        fn generate(
            &self,
            request: &GenerationRequest,
            progress: &mut dyn FnMut(Progress),
        ) -> core::Result<GenerationOutput> {
            self.generate_calls.set(self.generate_calls.get() + 1);
            *self.sampling.borrow_mut() = Some((request.steps, request.guidance));
            for event in &self.events {
                // Deliberately deliver late callbacks even after cancellation: the
                // capture must preserve the FIRST refusal rather than relabel peaks.
                progress(*event);
            }
            self.cancel_seen.set(request.cancel.is_cancelled());
            if self.cancel_at_end || request.cancel.is_cancelled() {
                return Err(core::Error::Canceled);
            }
            Ok(GenerationOutput::Images(vec![core::Image {
                width: 1024,
                height: 1024,
                pixels: vec![0; 1024 * 1024 * 3],
            }]))
        }
    }

    #[test]
    fn flux3_actual_generation_projection_includes_all_native_sampler_axes_and_restores() {
        let p = plan(&request("candle", "bf16"), "candle").unwrap();
        let req = generation_request(&p);
        let original = resource::actual_request_observation(&req);
        assert_eq!(original["guidance"], Value::Null);
        assert_eq!(original["negativePrompt"], Value::Null);
        for field in ["negative", "true_cfg", "sampler", "scheduler", "shift"] {
            let mut changed = req.clone();
            match field {
                "negative" => changed.negative_prompt = Some("different".into()),
                "true_cfg" => changed.true_cfg = Some(2.),
                "sampler" => changed.sampler = Some("different".into()),
                "scheduler" => changed.scheduler = Some("different".into()),
                _ => changed.scheduler_shift = Some(2.),
            }
            assert_ne!(original, resource::actual_request_observation(&changed));
        }
        assert_eq!(original, resource::actual_request_observation(&req));
    }

    #[test]
    fn flux3_anchor_request_digest_matches_shared_node_witness_for_both_checkpoints() {
        let mut p = plan(&request("candle", "bf16"), "candle").unwrap();
        for (provider, expected) in [
            (
                PROVIDER,
                "144040ed814a0836315c27e126a027fd750fe74a2400aceb2406346befbb455f",
            ),
            (
                BASE_PROVIDER,
                "a5a90974bf6d058f30e6ca9899c3b58ca6053b006234fbd28b83237f19543760",
            ),
        ] {
            p.checkpoint = checkpoint(provider).unwrap();
            assert_eq!(
                resource::anchor_digest(&p, &generation_request(&p)).unwrap(),
                expected
            );
        }
        assert_eq!(
            String::from_utf8(
                resource::canonical_bytes(&json!({"z":1.0,"a":[-0.0,1e-7,1e-6,1e21]})).unwrap()
            )
            .unwrap(),
            "{\"a\":[0,1e-7,0.000001,1e+21],\"z\":1}"
        );
    }

    #[test]
    fn flux3_actual_spec_projection_binds_real_quant_components_and_keeps_legacy_output_unchanged()
    {
        let mut p = plan(&request("candle", "bf16"), "candle").unwrap();
        let mut actual = load_spec(
            "candle",
            "bf16",
            PathBuf::from("fixture-package"),
            PathBuf::from("fixture-original"),
        )
        .unwrap();
        let baseline = actual_spec_observation(&p, &actual).unwrap();
        actual.components.insert(
            "text_encoder".into(),
            WeightsSource::Dir(PathBuf::from("different-actual-component")),
        );
        let changed = actual_spec_observation(&p, &actual).unwrap();
        assert_ne!(baseline["sha256"], changed["sha256"]);
        assert_ne!(baseline["components"], changed["components"]);
        actual.quantize = quant("candle", "q4").unwrap();
        assert!(
            actual_spec_observation(&p, &actual).is_err(),
            "intended/actual tier mismatch refuses"
        );
        p.tier = "q4".into();
        let packed = actual_spec_observation(&p, &actual).unwrap();
        assert_ne!(changed["sha256"], packed["sha256"]);
        let generator = ScriptedGenerator::new(vec![], false);
        let legacy = json!({"status":"gated","quality":{"result":"not_run"},"fixtureOnly":true});
        assert_eq!(
            attach_loaded_spec(legacy.clone(), packed, &generator).unwrap(),
            legacy
        );
    }

    #[test]
    fn flux3_actual_scripted_generate_loop_cannot_bypass_or_reuse_inner_permit() {
        for canceled in [false, true] {
            let plan = plan(&request("candle", "bf16"), "candle").unwrap();
            let generator = ScriptedGenerator::new(
                vec![
                    Progress::ConditioningComplete,
                    Progress::Step {
                        current: 1,
                        total: 16,
                    },
                    Progress::Decoding,
                ],
                canceled,
            );
            let mut spent = false;
            let mut guard = |_: &Plan, _: &GenerationRequest| {
                if spent {
                    return Err("fixture permit already spent".to_owned());
                }
                spent = true;
                assert_eq!(generator.generate_calls.get(), 0, "consume BEFORE generate");
                Ok(json!({"fixtureOnly":true,"invocationId":"one"}))
            };
            for _ in 0..3 {
                let _ = render_guarded(
                    &plan,
                    &generator,
                    &context(&plan, 1, 1 << 34),
                    &mut |_| {},
                    &mut guard,
                );
            }
            assert_eq!(
                generator.generate_calls.get(),
                1,
                "failed/canceled attempts also spend the only permit"
            );
            assert!(spent);
            let denied = ScriptedGenerator::new(vec![], false);
            assert!(render_guarded(
                &plan,
                &denied,
                &context(&plan, 1, 1 << 34),
                &mut |_| {},
                &mut |_, _| Err("missing immutable approval".to_owned())
            )
            .is_err());
            assert_eq!(denied.generate_calls.get(), 0);
        }
    }

    #[test]
    fn flux3_pup_meter_requires_the_real_typed_boundary_before_conditioning() {
        for include_boundary in [false, true] {
            let mut plan = plan(&request("candle", "bf16"), "candle").unwrap();
            let mut req = generation_request(&plan);
            req.enhance_prompt = true;
            plan.resource_case = Some(resource::BoundCase {
                value: json!({"fixtureOnly":true}),
                digest: "fixture".into(),
                inputs: vec![],
                request: req,
                adapters: vec![],
                pup_root: None,
                reference_resources: None,
            });
            let mut events = vec![];
            if include_boundary {
                events.push(Progress::PromptUpsamplingComplete);
            }
            events.extend([
                Progress::Loading(LoadPhase::TextEncoder),
                Progress::ConditioningComplete,
                Progress::Step {
                    current: 1,
                    total: 16,
                },
                Progress::Decoding,
            ]);
            let generator = ScriptedGenerator::new(events, false);
            let mut observed = vec![];
            let result = render(
                &plan,
                &generator,
                &context(&plan, 1, 1 << 34),
                &mut |phase| observed.push(phase),
            );
            if include_boundary {
                let receipt = result.unwrap();
                assert_eq!(
                    observed,
                    [
                        protocol::ReferencePhase::Conditioning,
                        protocol::ReferencePhase::Denoise,
                        protocol::ReferencePhase::Decode
                    ]
                );
                assert_eq!(receipt["promptUpsampling"]["typedCompletionObserved"], true);
                assert_eq!(
                    receipt["promptUpsampling"]["positiveEnhancementAttested"],
                    false
                );
            } else {
                assert!(result.unwrap_err().contains("PromptUpsamplingComplete"));
                assert!(observed.is_empty());
            }
        }
    }

    #[test]
    fn flux3_native_output_receipt_binds_pixels_dimensions_and_real_carrier() {
        let output = |width, height, pixels| {
            GenerationOutput::Images(vec![core::Image {
                width,
                height,
                pixels,
            }])
        };
        let original = output(2, 1, vec![0, 1, 2, 3, 4, 5]);
        let receipt = output_receipt(&original, 2, 1).unwrap();
        assert_eq!(receipt["bytes"], 6);
        assert_eq!(receipt["format"], "RGB8");
        let changed = output(2, 1, vec![0, 1, 2, 3, 4, 6]);
        assert_ne!(
            receipt["sha256"],
            output_receipt(&changed, 2, 1).unwrap()["sha256"]
        );
        let reshaped = output(1, 2, vec![0, 1, 2, 3, 4, 5]);
        assert_ne!(
            receipt["sha256"],
            output_receipt(&reshaped, 1, 2).unwrap()["sha256"]
        );
        assert!(output_receipt(&reshaped, 2, 1).is_err());
        assert!(output_receipt(&output(2, 1, vec![0; 5]), 2, 1).is_err());
        assert!(output_receipt(&GenerationOutput::Images(vec![]), 2, 1).is_err());
        assert_eq!(receipt, output_receipt(&original, 2, 1).unwrap());
    }

    #[test]
    fn flux3_both_constructed_requests_satisfy_the_real_native_parameter_contract() {
        use super::core::flux3::{
            DitConfig, Flux3Config, GenerationParams, SamplerConfig, TextEncoderConfig, VaeConfig,
        };
        for lane in ["candle", "mlx"] {
            for provider in [PROVIDER, BASE_PROVIDER] {
                let mut value = request(lane, "bf16");
                let checkpoint = checkpoint(provider).unwrap();
                value["planned"]["target"]["provider"] = json!(provider);
                value["planned"]["target"]["modelId"] = json!(provider);
                value["planned"]["calibrationFingerprint"] =
                    json!(fingerprint_for(lane, "bf16", provider).unwrap());
                value["planned"]["fixture"] = json!(format!(
                    "{}-{lane}-bf16-1024-seed26243-step{}",
                    checkpoint.label, checkpoint.steps
                ));
                let plan = plan(&value, lane).unwrap();
                let req = generation_request(&plan);
                let flow_map = provider == PROVIDER;
                let config = Flux3Config {
                    format_version: 1,
                    model: provider.into(),
                    quantization: None,
                    torch_dtype: "bfloat16".into(),
                    dit: DitConfig {
                        hidden: 3072,
                        heads: 24,
                        depth: 6,
                        depth_single: 24,
                        mlp_ratio: 4.,
                        axes_dim: vec![32, 48, 48],
                        theta: 10000.,
                        image_in_channels: 64,
                        image_cond_in_channels: 64,
                        text_in_channels: 10240,
                        out_modes: vec!["image".into()],
                        vec_in_dim: None,
                        image_rope_scale: 1.,
                        flow_map,
                    },
                    text_encoder: TextEncoderConfig {
                        path: "text_encoder".into(),
                        tokenizer: "text_encoder".into(),
                        layers: vec![9, 18, 27],
                        pad_multiple: 32,
                        max_tokens: 8192,
                        dtype: "bfloat16".into(),
                    },
                    vae: VaeConfig {
                        path: "ae.safetensors".into(),
                        dtype: "bfloat16".into(),
                    },
                    sampler: SamplerConfig {
                        steps: checkpoint.steps,
                        guidance: checkpoint.guidance.map(f64::from),
                        alpha: f64::from(core::flux3::DEFAULT_ALPHA),
                        size: [1024, 1024],
                    },
                };
                let resolved = GenerationParams::resolve(&req, SEED, &config).unwrap();
                assert_eq!(req.guidance, if flow_map { None } else { Some(3.5) });
                assert_eq!(resolved.guidance, req.guidance);
                assert_eq!(resolved.steps, checkpoint.steps);
                if flow_map {
                    let mut invalid = req;
                    invalid.guidance = Some(1.0);
                    assert!(GenerationParams::resolve(&invalid, SEED, &config).is_err());
                }
            }
        }
    }

    #[test]
    fn flux3_base_producer_uses_its_own_identity_sampling_and_calibration() {
        for lane in ["candle", "mlx"] {
            let mut value = request(lane, "bf16");
            value["planned"]["target"]["provider"] = json!(BASE_PROVIDER);
            value["planned"]["target"]["modelId"] = json!(BASE_PROVIDER);
            value["planned"]["calibrationFingerprint"] =
                json!(format!("flux3-klein-base-{lane}-cfg-source-v1"));
            value["planned"]["fixture"] = json!(format!(
                "flux3-klein-base-{lane}-bf16-1024-seed26243-step50"
            ));
            let plan = plan(&value, lane).unwrap();
            assert_eq!(plan.checkpoint.provider, BASE_PROVIDER);
            assert_eq!(
                plan.checkpoint.repository,
                "black-forest-labs/flux-3-image-klein-base"
            );
            assert_eq!(
                plan.checkpoint.revision,
                "1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4"
            );
            assert_eq!(plan.checkpoint.steps, 50);
            assert_eq!(plan.checkpoint.guidance, Some(3.5));
            assert_ne!(plan.fingerprint, fingerprint(lane, "bf16").unwrap());
            let generator = ScriptedGenerator::new(
                vec![
                    Progress::ConditioningComplete,
                    Progress::Step {
                        current: 50,
                        total: 50,
                    },
                    Progress::Decoding,
                ],
                false,
            );
            render(&plan, &generator, &context(&plan, 1, 1 << 34), &mut |_| {}).unwrap();
            assert_eq!(*generator.sampling.borrow(), Some((Some(50), Some(3.5))));
            let mut wrong = value.clone();
            wrong["planned"]["target"]["modelId"] = json!(PROVIDER);
            assert!(super::plan(&wrong, lane).is_err());
            wrong = value;
            wrong["planned"]["calibrationFingerprint"] = json!(fingerprint(lane, "bf16").unwrap());
            assert!(super::plan(&wrong, lane).is_err());
        }
    }

    #[test]
    fn flux3_render_callback_orders_scopes_and_ignores_duplicate_boundaries() {
        for lane in ["candle", "mlx"] {
            let plan = plan(&request(lane, "bf16"), lane).unwrap();
            let mut events = vec![Progress::Loading(LoadPhase::TextEncoder)];
            if lane == "candle" {
                events.push(Progress::Loading(LoadPhase::Renderer));
            }
            events.extend([
                Progress::ConditioningComplete,
                Progress::ConditioningComplete,
                Progress::Step {
                    current: 1,
                    total: STEPS,
                },
                Progress::Step {
                    current: STEPS,
                    total: STEPS,
                },
                Progress::Decoding,
                Progress::ConditioningComplete,
            ]);
            let generator = ScriptedGenerator::new(events, false);
            let mut observed = Vec::new();
            render(
                &plan,
                &generator,
                &context(&plan, 1, 1 << 34),
                &mut |phase| observed.push(phase),
            )
            .unwrap();
            assert_eq!(
                observed,
                [
                    protocol::ReferencePhase::Denoise,
                    protocol::ReferencePhase::Decode
                ]
            );
            assert_eq!(
                *generator.lifecycle.borrow(),
                [
                    "enter:Conditioning",
                    "leave:Conditioning",
                    "enter:Denoise",
                    "leave:Denoise",
                    "enter:Decode",
                    "leave:Decode",
                ]
            );
            assert!(matches!(
                generator.outcomes.borrow().as_slice(),
                [MemoryRunOutcome::Complete]
            ));
            assert!(!generator.cancel_seen.get());
        }
    }

    #[test]
    fn flux3_render_missing_or_late_boundary_cancels_and_preserves_typed_refusal() {
        let plan = plan(&request("candle", "bf16"), "candle").unwrap();
        for current in [0, 1] {
            let generator = ScriptedGenerator::new(
                vec![
                    Progress::Loading(LoadPhase::Renderer),
                    Progress::Step {
                        current,
                        total: STEPS,
                    },
                    Progress::ConditioningComplete,
                    Progress::Decoding,
                ],
                false,
            );
            let mut observed = Vec::new();
            let error = render(
                &plan,
                &generator,
                &context(&plan, 1, 1 << 34),
                &mut |phase| observed.push(phase),
            )
            .unwrap_err();
            let expected = if current == 0 {
                BoundaryError::InvalidZeroStep
            } else {
                BoundaryError::MissingPreDenoise
            };
            assert_eq!(error, expected.to_string());
            assert!(generator.cancel_seen.get());
            assert!(
                observed.is_empty(),
                "late completion must not recover a failed capture"
            );
            assert_eq!(*generator.lifecycle.borrow(), ["enter:Conditioning"]);
            assert!(
                matches!(generator.outcomes.borrow().as_slice(), [MemoryRunOutcome::Error { message }] if message == &error)
            );
        }
    }

    #[test]
    fn flux3_render_provider_cancellation_finishes_once_without_fabricating_phases() {
        let plan = plan(&request("mlx", "bf16"), "mlx").unwrap();
        for after_conditioning in [false, true] {
            let events = if after_conditioning {
                vec![Progress::ConditioningComplete]
            } else {
                vec![]
            };
            let generator = ScriptedGenerator::new(events, true);
            let mut observed = Vec::new();
            let error = render(
                &plan,
                &generator,
                &context(&plan, 1, 1 << 34),
                &mut |phase| observed.push(phase),
            )
            .unwrap_err();
            assert_eq!(error, "native FLUX3 generation failed: cancelled");
            assert_eq!(
                observed,
                if after_conditioning {
                    vec![protocol::ReferencePhase::Denoise]
                } else {
                    vec![]
                }
            );
            assert!(matches!(
                generator.outcomes.borrow().as_slice(),
                [MemoryRunOutcome::Canceled]
            ));
        }
    }

    #[test]
    fn flux3_exact_native_specs_and_external_components() {
        for lane in ["candle", "mlx"] {
            for (tier, expected) in [
                ("bf16", None),
                ("q4", Some(Quant::Q4)),
                ("q8", Some(Quant::Q8)),
            ] {
                let spec = load_spec(lane, tier, "derived".into(), "original".into()).unwrap();
                assert_eq!(spec.quantize, expected);
                assert_eq!(spec.precision, Precision::Bf16);
                assert_eq!(
                    spec.components.get("text_encoder"),
                    Some(&WeightsSource::Dir(PathBuf::from("original/text_encoder")))
                );
                assert_eq!(
                    spec.components.get("autoencoder"),
                    Some(&WeightsSource::File(PathBuf::from(
                        "original/ae.safetensors"
                    )))
                );
                if tier != "bf16" {
                    assert!(conversion(lane, tier)
                        .unwrap()
                        .starts_with(&format!("flux3-{lane}-")));
                }
            }
        }
        assert!(quant("mlx", "fp8r").is_err());
        assert!(quant("mlx", "nvfp4").is_err());
        assert_eq!(quant("candle", "fp8r").unwrap(), Some(Quant::Fp8Rowwise));
        assert_eq!(quant("candle", "nvfp4").unwrap(), Some(Quant::Nvfp4));
    }

    #[test]
    fn flux3_rejects_wrong_backend_package_and_snapshot_tier_guess() {
        let original = std::path::Path::new("original");
        for lane in ["candle", "mlx"] {
            for tier in ["q4", "q8"] {
                let id = conversion(lane, tier).unwrap();
                assert!(validate_package_root(
                    lane,
                    tier,
                    original,
                    &PathBuf::from("derived").join(tier).join(id)
                )
                .is_ok());
                assert!(validate_package_root(lane, tier, original, &original.join(tier)).is_err());
                let other = if lane == "candle" { "mlx" } else { "candle" };
                assert!(validate_package_root(
                    lane,
                    tier,
                    original,
                    &PathBuf::from("derived")
                        .join(tier)
                        .join(conversion(other, tier).unwrap())
                )
                .is_err());
            }
        }
    }

    // ---- sc-26243 zero-budget native-boundary self-test ----
    use super::boundary_self_test as boundary;

    fn self_test_request(lane: &str, binary: &str, cases: Vec<Value>) -> Value {
        json!({"action": boundary::SELF_TEST_ACTION, "lane": lane, "expectedBinarySha256": binary,
            "buildReceiptSha256": "c".repeat(64), "buildReceiptBinarySha256": binary,
            "sourceSha": "a".repeat(40), "inferenceMainSha": protocol::INFERENCE_PIN, "cases": cases})
    }

    fn anchor_case(lane: &str, tier: &str) -> Value {
        let planned = request(lane, tier);
        let p = plan(&planned, lane).unwrap();
        let digest = resource::anchor_digest(&p, &generation_request(&p)).unwrap();
        json!({"invocationId": format!("{lane}.distilled.resource-native.{tier}.generation-1"),
            "caseId": format!("{lane}.distilled.resource-native.{tier}"), "kind": "resource_anchor",
            "requestSha256": digest, "request": planned})
    }

    fn env_of(pairs: Vec<(&'static str, &'static str)>) -> impl Fn(&str) -> Option<String> {
        move |key: &str| {
            pairs
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| (*v).to_owned())
        }
    }

    #[test]
    fn native_boundary_self_test_is_unreachable_without_its_explicit_cpu_guard() {
        let guard_env = boundary::SELF_TEST_ENV;
        assert!(boundary::guard(&env_of(vec![("CUDA_VISIBLE_DEVICES", "-1")])).is_err());
        assert!(boundary::guard(&env_of(vec![
            (guard_env, "true"),
            ("CUDA_VISIBLE_DEVICES", "-1")
        ]))
        .is_err());
        assert!(boundary::guard(&env_of(vec![
            (guard_env, "1"),
            ("CUDA_VISIBLE_DEVICES", "1")
        ]))
        .is_err());
        assert!(
            boundary::guard(&env_of(vec![(guard_env, "1")])).is_err(),
            "unset CVD exposes devices"
        );
        for production in [
            "SCENEWORKS_FLUX3_INVOCATION",
            "SCENEWORKS_FLUX3_BUDGET_NODE",
            "SCENEWORKS_FLUX3_KLEIN_ROOT",
        ] {
            assert!(
                boundary::guard(&env_of(vec![
                    (guard_env, "1"),
                    ("CUDA_VISIBLE_DEVICES", "-1"),
                    (production, "{}")
                ]))
                .is_err(),
                "{production} present must refuse"
            );
        }
        assert!(boundary::guard(&env_of(vec![
            (guard_env, "1"),
            ("CUDA_VISIBLE_DEVICES", "-1"),
            ("SCENEWORKS_GPU_ID", "1")
        ]))
        .is_err());
        assert!(boundary::guard(&env_of(vec![
            (guard_env, "1"),
            ("CUDA_VISIBLE_DEVICES", "-1")
        ]))
        .is_ok());
    }

    #[test]
    fn native_boundary_self_test_proves_one_permit_one_generate_and_refusals_with_the_production_loop(
    ) {
        let binary = "b".repeat(64);
        for lane in ["candle", "mlx"] {
            let receipt = boundary::run_self_test(
                &self_test_request(
                    lane,
                    &binary,
                    vec![anchor_case(lane, "bf16"), anchor_case(lane, "q4")],
                ),
                lane,
                render_guarded,
                &binary,
            )
            .unwrap();
            assert_eq!(receipt["binarySha256"], binary);
            assert_eq!(receipt["campaignPermitsConsumed"], 0);
            assert_eq!(receipt["inferenceMainSha"], protocol::INFERENCE_PIN);
            for proof in receipt["cases"].as_array().unwrap() {
                assert_eq!(
                    proof["valid"],
                    json!({"reserveCalls":1,"generateCalls":1,"order":["reserve","generate"]})
                );
                for refusal in ["reused", "missing", "mismatchedDigest", "mismatchedCase"] {
                    assert_eq!(
                        proof[refusal],
                        json!({"refused":true,"generateCalls":0}),
                        "{refusal}"
                    );
                }
            }
        }
    }

    #[test]
    fn native_boundary_self_test_refuses_a_binary_hash_or_planner_digest_mismatch() {
        let binary = "b".repeat(64);
        let request = self_test_request("candle", &binary, vec![anchor_case("candle", "bf16")]);
        let error = boundary::run_self_test(&request, "candle", render_guarded, &"e".repeat(64))
            .unwrap_err();
        assert!(error.contains("sealed binary hash mismatch"), "{error}");
        let mut wrong_receipt = request.clone();
        wrong_receipt["buildReceiptBinarySha256"] = json!("d".repeat(64));
        assert!(
            boundary::run_self_test(&wrong_receipt, "candle", render_guarded, &binary).is_err()
        );
        let mut wrong_digest = request.clone();
        wrong_digest["cases"][0]["requestSha256"] = json!("0".repeat(64));
        assert!(
            boundary::run_self_test(&wrong_digest, "candle", render_guarded, &binary).is_err(),
            "a planner digest differing from the binary's own native request digest yields no receipt"
        );
        let mut wrong_pin = request.clone();
        wrong_pin["inferenceMainSha"] = json!("f".repeat(40));
        assert!(boundary::run_self_test(&wrong_pin, "candle", render_guarded, &binary).is_err());
        assert!(
            boundary::run_self_test(&request, "mlx", render_guarded, &binary).is_err(),
            "lane mismatch"
        );
    }

    fn generate_twice(
        plan: &Plan,
        generator: &dyn Generator,
        _: &MemoryRunContext,
        _: &mut dyn FnMut(protocol::ReferencePhase),
        reserve: &mut dyn FnMut(&Plan, &GenerationRequest) -> Result<Value, String>,
    ) -> Result<Value, String> {
        let request = generation_request(plan);
        reserve(plan, &request)?;
        let _ = generator.generate(&request, &mut |_| {});
        let _ = generator.generate(&request, &mut |_| {});
        Ok(json!({}))
    }

    fn generate_before_reserve(
        plan: &Plan,
        generator: &dyn Generator,
        _: &MemoryRunContext,
        _: &mut dyn FnMut(protocol::ReferencePhase),
        reserve: &mut dyn FnMut(&Plan, &GenerationRequest) -> Result<Value, String>,
    ) -> Result<Value, String> {
        let request = generation_request(plan);
        let _ = generator.generate(&request, &mut |_| {});
        reserve(plan, &request)?;
        Ok(json!({}))
    }

    fn generate_without_reserve(
        plan: &Plan,
        generator: &dyn Generator,
        _: &MemoryRunContext,
        _: &mut dyn FnMut(protocol::ReferencePhase),
        _: &mut dyn FnMut(&Plan, &GenerationRequest) -> Result<Value, String>,
    ) -> Result<Value, String> {
        let _ = generator.generate(&generation_request(plan), &mut |_| {});
        Ok(json!({}))
    }

    #[test]
    fn native_boundary_self_test_catches_double_generate_late_or_missing_reservation_mutants() {
        let binary = "b".repeat(64);
        let request = self_test_request("candle", &binary, vec![anchor_case("candle", "bf16")]);
        let mutants: [(&str, boundary::Render); 3] = [
            ("two generates per permit", generate_twice),
            ("generate before reserve", generate_before_reserve),
            ("generate without reserve", generate_without_reserve),
        ];
        for (label, mutant) in mutants {
            assert!(
                boundary::run_self_test(&request, "candle", mutant, &binary).is_err(),
                "{label} must be caught"
            );
        }
    }
}
