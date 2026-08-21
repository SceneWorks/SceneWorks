//! Terminal Windows/CUDA acceptance for SC-19054's geometry-aware Candle scalar gate.
//!
//! This Candle-only module is invoked explicitly by `windows-candle.yml`. The backend-neutral
//! source/workflow contract runs in `scripts/platform-review-contracts.test.mjs`. One value returned
//! by [`admission_contract`] drives both the machine-countable admission receipt and the
//! request-scoped [`gen_core::GenerationMemory`] consumed by the real `flux1_schnell` provider, so
//! the decision and render cannot pass independently.

use std::path::{Path, PathBuf};

use gen_core::{GenerationMemory, LoadSpec, WeightsSource};
use serde_json::{json, Value};

use crate::candle_scalar_gate::{self, LoadPlan, VramBudget};

const STORY_ID: &str = "SC-19054";
const EVENT: &str = "sc19054_candle_admission_selected";
const MODEL_ID: &str = "flux_schnell";
const ENGINE_ID: &str = "flux1_schnell";
const TIER: &str = "q4";
const WIDTH: u32 = 1024;
const HEIGHT: u32 = 1024;
const BUDGET_GB: f64 = 24.0;
const STEPS: u32 = 4;
const SEED: u64 = 42;
const PROMPT: &str = "a rusty robot holding a lit candle, studio lighting";

#[derive(Clone, Copy, Debug, PartialEq)]
struct AdmissionContract {
    sequential_capable: bool,
    old_plan: LoadPlan,
    selected_plan: LoadPlan,
    raw_resident_peak_gb: f64,
    widened_resident_peak_gb: f64,
    raw_sequential_peak_gb: f64,
    widened_sequential_peak_gb: f64,
}

#[derive(Clone, Debug)]
struct AcceptanceExecution {
    spec: LoadSpec,
    memory: GenerationMemory,
}

fn builtin_manifest_entry() -> serde_json::Map<String, Value> {
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc is embedded");
    let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
        .expect("builtin.models.jsonc parses");
    manifest["models"]
        .as_array()
        .expect("builtin manifest has a models array")
        .iter()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(MODEL_ID))
        .and_then(Value::as_object)
        .cloned()
        .expect("builtin manifest contains flux_schnell")
}

fn request_geometry() -> gen_core::MemoryGeometry {
    gen_core::MemoryGeometry {
        width: WIDTH,
        height: HEIGHT,
        batch: 1,
        frames: 1,
        reference_count: 0,
    }
}

/// Reproduce the exact decision-diff cell from the shipped manifest and production scalar gate.
fn admission_contract(sequential_capable: bool) -> AdmissionContract {
    let manifest = builtin_manifest_entry();
    let budget = Some(VramBudget {
        free_gb: BUDGET_GB,
        total_gb: BUDGET_GB,
    });
    let raw_resident_peak_gb = candle_scalar_gate::predicted_peak_gb(&manifest, TIER)
        .expect("flux_schnell q4 has a resident peak");
    let raw_sequential_peak_gb = candle_scalar_gate::predicted_sequential_peak_gb(&manifest, TIER)
        .expect("flux_schnell q4 has a sequential peak");
    let widened_resident_peak_gb =
        candle_scalar_gate::predicted_peak_gb_for_request(&manifest, TIER, 0, request_geometry())
            .expect("flux_schnell q4 has a graded resident peak");
    let widened_sequential_peak_gb = candle_scalar_gate::predicted_sequential_peak_gb_for_request(
        &manifest,
        TIER,
        0,
        request_geometry(),
    )
    .expect("flux_schnell q4 has a graded sequential peak");
    AdmissionContract {
        sequential_capable,
        old_plan: candle_scalar_gate::load_plan(
            Some(raw_resident_peak_gb),
            Some(raw_sequential_peak_gb),
            budget,
            sequential_capable,
        ),
        selected_plan: candle_scalar_gate::load_plan(
            Some(widened_resident_peak_gb),
            Some(widened_sequential_peak_gb),
            budget,
            sequential_capable,
        ),
        raw_resident_peak_gb,
        widened_resident_peak_gb,
        raw_sequential_peak_gb,
        widened_sequential_peak_gb,
    }
}

/// Bind the admitted plan to the exact two-part provider execution contract.
///
/// The packed q4 tier is detected from `transformer/config.json`; setting `LoadSpec::quantize`
/// requests unsupported on-the-fly quantization and is therefore deliberately forbidden. FLUX.1
/// consumes Sequential at generation time through `GenerationMemory::stage_residency`, so that bit
/// is derived directly from the admitted plan and carried into the real request.
fn acceptance_execution(root: &Path, admission: AdmissionContract) -> AcceptanceExecution {
    assert_eq!(
        admission.selected_plan,
        LoadPlan::Sequential,
        "the exact SC-19054 cell must select Sequential before real weights are touched"
    );
    let spec = LoadSpec::new(WeightsSource::Dir(root.to_owned())).with_resolved_route(MODEL_ID);
    assert!(
        spec.quantize.is_none(),
        "packed FLUX q4 must be inferred from transformer/config.json"
    );
    AcceptanceExecution {
        spec,
        memory: GenerationMemory {
            stage_residency: true,
            ..Default::default()
        },
    }
}

fn plan_label(plan: LoadPlan) -> &'static str {
    match plan {
        LoadPlan::Resident => "resident",
        LoadPlan::Sequential => "sequential",
        LoadPlan::Reject => "reject",
    }
}

fn admission_event(admission: AdmissionContract) -> Value {
    json!({
        "event": EVENT,
        "story": STORY_ID,
        "modelId": MODEL_ID,
        "engineId": ENGINE_ID,
        "tier": TIER,
        "geometry": { "width": WIDTH, "height": HEIGHT, "batch": 1, "frames": 1 },
        "budgetGb": BUDGET_GB,
        "sequentialCapable": admission.sequential_capable,
        "oldPlan": plan_label(admission.old_plan),
        "selectedPlan": plan_label(admission.selected_plan),
        "rawResidentPeakGb": admission.raw_resident_peak_gb,
        "widenedResidentPeakGb": admission.widened_resident_peak_gb,
        "rawSequentialPeakGb": admission.raw_sequential_peak_gb,
        "widenedSequentialPeakGb": admission.widened_sequential_peak_gb,
    })
}

fn required_env(name: &str) -> String {
    let value = std::env::var(name).unwrap_or_else(|_| panic!("set ${name}"));
    let value = value.trim();
    assert!(!value.is_empty(), "${name} must not be empty");
    value.to_owned()
}

fn sha256_file(path: &Path) -> String {
    use sha2::{Digest, Sha256};

    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("read {} for SHA-256: {error}", path.display()));
    format!("{:x}", Sha256::digest(bytes))
}

/// Exact real-weight terminal spot-check. The workflow provisions and identity-checks the pinned
/// public q4 snapshot, records host/GPU/cache facts, then invokes this test alone.
#[allow(dead_code)]
fn run_sc19054_flux_candle_acceptance() {
    use std::time::Instant;

    use gen_core::{GenerationOutput, GenerationRequest, Progress};

    let root = PathBuf::from(required_env("SC19054_FLUX_Q4_ROOT"));
    let out_dir = PathBuf::from(required_env("SC19054_OUTPUT_DIR"));
    let scene_works_revision = required_env("SC19054_SCENEWORKS_REVISION");
    let inference_revision = required_env("SC19054_INFERENCE_REVISION");
    let artifact_revision = required_env("SC19054_FLUX_REVISION");
    let runtime_cap_gb = required_env("SCENEWORKS_CUDA_VRAM_CAP_GB")
        .parse::<f64>()
        .expect("SCENEWORKS_CUDA_VRAM_CAP_GB must be numeric");
    assert_eq!(runtime_cap_gb, BUDGET_GB, "acceptance must emulate 24 GiB");
    assert!(root.is_dir(), "exact q4 root missing: {}", root.display());
    assert_eq!(root.file_name().and_then(|name| name.to_str()), Some(TIER));
    assert!(
        scene_works_revision.len() == 40
            && scene_works_revision
                .chars()
                .all(|ch| ch.is_ascii_hexdigit()),
        "SceneWorks revision must be exact 40-hex"
    );
    assert_eq!(
        inference_revision,
        crate::catalog_semantic_jobs::INFERENCE_RUNTIME_REVISION,
        "workflow inference receipt must match the linked runtime pin"
    );
    let pin = crate::manifest_pins::builtin_model_pin(MODEL_ID);
    assert_eq!(pin.repo, "SceneWorks/flux1-schnell-mlx");
    assert_eq!(artifact_revision, pin.revision);
    assert_eq!(pin.files, ["q4/*"]);
    std::fs::create_dir_all(&out_dir).expect("create SC-19054 output directory");

    // Resolve the capability from the exact linked inference descriptor before weight I/O. This is
    // the same source the product gate uses; a worker-side allowlist would make the acceptance pass
    // even if the pinned Candle provider stopped honoring Sequential.
    let descriptor = crate::inference_runtime::media_descriptor(ENGINE_ID)
        .expect("the pinned Candle bundle must register flux1_schnell");
    assert_eq!(descriptor.backend, "candle", "acceptance requires Candle");
    let admission = admission_contract(descriptor.capabilities.supports_sequential_offload);
    let event = admission_event(admission);
    println!(
        "SC19054_ADMISSION_EVENT={}",
        serde_json::to_string(&event).expect("serialize admission event")
    );
    let execution = acceptance_execution(&root, admission);
    let stage_residency = execution.memory.stage_residency;
    assert!(stage_residency, "Sequential must engage staged residency");
    let load_started = Instant::now();
    let generator = crate::inference_runtime::load(ENGINE_ID, &execution.spec)
        .unwrap_or_else(|error| panic!("load exact packed FLUX.1 Schnell q4: {error}"));
    let load_seconds = load_started.elapsed().as_secs_f64();
    assert_eq!(generator.descriptor().id, ENGINE_ID);
    assert_eq!(generator.descriptor().backend, descriptor.backend);
    assert!(
        generator
            .descriptor()
            .capabilities
            .supports_sequential_offload,
        "the selected Sequential plan must reach a provider that honors it"
    );

    let request = GenerationRequest {
        prompt: PROMPT.to_owned(),
        width: WIDTH,
        height: HEIGHT,
        count: 1,
        seed: Some(SEED),
        steps: Some(STEPS),
        memory: Some(execution.memory),
        ..Default::default()
    };
    let render_started = Instant::now();
    let mut last_step = 0;
    let mut saw_decode = false;
    let output = generator
        .generate(&request, &mut |progress| match progress {
            Progress::Step { current, total } => {
                assert!(current >= last_step && current <= total);
                assert_eq!(
                    total, STEPS,
                    "FLUX Schnell must keep the fixed four-step request"
                );
                last_step = current;
                println!("[SC-19054] denoise {current}/{total}");
            }
            Progress::Decoding => {
                saw_decode = true;
                println!("[SC-19054] decoding");
            }
            Progress::Loading(phase) => println!("[SC-19054] loading {phase:?}"),
        })
        .unwrap_or_else(|error| panic!("render exact packed FLUX.1 Schnell q4: {error}"));
    let render_seconds = render_started.elapsed().as_secs_f64();
    assert_eq!(
        last_step, STEPS,
        "render did not complete all denoise steps"
    );
    assert!(saw_decode, "render emitted no decode progress");
    let GenerationOutput::Images(mut images) = output else {
        panic!("FLUX Schnell returned a non-image output")
    };
    assert_eq!(images.len(), 1, "acceptance must render exactly one image");
    let image = images.pop().expect("one image");
    assert_eq!((image.width, image.height), (WIDTH, HEIGHT));
    assert_eq!(image.pixels.len(), (WIDTH * HEIGHT * 3) as usize);
    let mean = crate::smoke_support::image_mean(&image);
    let std = crate::smoke_support::image_std(&image);
    assert!(
        std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
        "FLUX Schnell render is degenerate (mean {mean:.2}, std {std:.2})"
    );
    let image_path = out_dir.join("sc19054-flux-schnell-q4-1024.png");
    crate::smoke_support::save_png(&image, &image_path);
    let image_sha256 = sha256_file(&image_path);

    let receipt = json!({
        "verdict": "PASS",
        "admission": event,
        "sceneWorksRevision": scene_works_revision,
        "inferenceRevision": inference_revision,
        "artifact": {
            "repository": pin.repo,
            "revision": artifact_revision,
            "tier": TIER,
            "root": root,
        },
        "request": {
            "prompt": PROMPT,
            "width": WIDTH,
            "height": HEIGHT,
            "count": 1,
            "seed": SEED,
            "steps": STEPS,
        },
        "runtime": {
            "engineId": generator.descriptor().id,
            "backend": generator.descriptor().backend,
            "supportsSequentialOffload": generator.descriptor().capabilities.supports_sequential_offload,
            "stageResidency": stage_residency,
            "vramCapGb": runtime_cap_gb,
            "loadSeconds": load_seconds,
            "renderSeconds": render_seconds,
        },
        "output": {
            "path": image_path,
            "width": image.width,
            "height": image.height,
            "pixelMean": mean,
            "pixelStd": std,
            "sha256": image_sha256,
        }
    });
    let receipt_path = out_dir.join("sc19054-flux-acceptance.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_vec_pretty(&receipt).expect("serialize acceptance receipt"),
    )
    .expect("write acceptance receipt");
    println!("SC19054_ACCEPTANCE_RECEIPT={}", receipt_path.display());
    println!("SC19054_OUTPUT_SHA256={image_sha256}");
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[test]
#[ignore = "terminal SC-19054 acceptance; needs exact packed FLUX.1 Schnell q4 weights and CUDA"]
fn sc19054_flux_candle_acceptance() {
    run_sc19054_flux_candle_acceptance();
}

#[cfg(test)]
mod contract_tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn shipped_manifest_exact_cell_flips_resident_to_sequential_at_the_widened_floor() {
        let admission = admission_contract(true);
        assert_eq!(admission.old_plan, LoadPlan::Resident);
        assert_eq!(admission.selected_plan, LoadPlan::Sequential);
        close(admission.raw_resident_peak_gb, 23.3);
        close(admission.widened_resident_peak_gb, 24.232);
        close(admission.raw_sequential_peak_gb, 20.4);
        close(admission.widened_sequential_peak_gb, 21.216);
        let event = admission_event(admission);
        assert_eq!(event["event"], EVENT);
        assert_eq!(event["selectedPlan"], "sequential");
        assert_eq!(event["geometry"]["width"], WIDTH);
        assert_eq!(event["budgetGb"], BUDGET_GB);
        assert_eq!(event["sequentialCapable"], true);
        assert_eq!((STEPS, SEED), (4, 42));
        assert_eq!(
            PROMPT,
            "a rusty robot holding a lit candle, studio lighting"
        );
    }

    #[test]
    fn selected_admission_is_load_bearing_for_the_exact_q4_provider_spec() {
        let root = PathBuf::from("exact-snapshot").join(TIER);
        let execution = acceptance_execution(&root, admission_contract(true));
        assert_eq!(execution.spec.weights, WeightsSource::Dir(root));
        assert_eq!(execution.spec.quantize, None);
        assert_eq!(execution.spec.resolved_route.as_deref(), Some(MODEL_ID));
        assert!(execution.memory.stage_residency);
        assert_eq!(
            execution.memory,
            GenerationMemory {
                stage_residency: true,
                ..Default::default()
            }
        );

        let mut disconnected = admission_contract(true);
        disconnected.selected_plan = LoadPlan::Resident;
        assert!(
            std::panic::catch_unwind(|| acceptance_execution(Path::new("wrong"), disconnected))
                .is_err(),
            "a resident-plan mutation must stop before provider load"
        );
        assert_eq!(admission_contract(false).selected_plan, LoadPlan::Reject);
    }

    #[test]
    fn catalog_pin_is_the_public_exact_q4_turnkey() {
        let pin = crate::manifest_pins::builtin_model_pin(MODEL_ID);
        assert_eq!(pin.repo, "SceneWorks/flux1-schnell-mlx");
        assert_eq!(pin.revision, "bba3ae01dfd94089f173c05edd4e1a4c551f2599");
        assert_eq!(pin.files, ["q4/*"]);
    }
}
