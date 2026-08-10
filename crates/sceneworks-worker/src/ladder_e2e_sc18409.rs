//! On-device validation of z_image_turbo's DeferredMaterialization Sequential cold load (sc-18409).
//!
//! PR #2215 (`with_selected_sequential_shape`, `mlx_fit_gate.rs`) changed z_image_turbo's
//! production cold-load path: when `apply_residency_policy` selects `OffloadPolicy::Sequential`,
//! the returned `LoadSpec` now also carries `LoadShape::DeferredMaterialization` — with runtime
//! evidence from qwen only. This scenario exercises that exact branch on real weights: it derives
//! the cap boundary from the gate's own decision (no hand-copied numbers), asserts the deferred
//! shape provably came from the Sequential branch, loads, admits a request through the real
//! `evaluate_request` seam, renders, and compares the observed MLX peak to the admitted ceiling.
//!
//! This is a VALIDATION harness, not a calibration capture: it writes logs and numbers under
//! `~/SceneWorks/render-validation-sc18409/` and never touches evidence records or the closure
//! table. Like the sc-18101 scenarios it samples process-global MLX peak counters, so run it in
//! its own `cargo test` invocation.
//!
//! ```text
//! cargo test -p sceneworks-worker --release --lib -- --ignored --exact --nocapture \
//!     ladder_e2e_sc18409::z_image_deferred_sequential_cold_load_renders_within_the_admitted_ceiling
//! ```

use std::io::Write as _;
use std::path::PathBuf;

use gen_core::{LoadSpec, WeightsSource};

use crate::ladder_e2e_sc18101::{
    admission_path_line, cached_tier_dir, field_in_line, gib, inputs, shipped_manifest_entry,
    with_captured_tracing,
};
use crate::mlx_fit_gate::MlxRequestPlan;

const REPO_DIR: &str = "models--SceneWorks--z-image-turbo-mlx";
const ENGINE: &str = "z_image_turbo";
const TIER: &str = "bf16";
const SENTINEL: &str = "model_index.json";

fn out_dir() -> PathBuf {
    let dir = PathBuf::from(crate::smoke_support::env_or(
        "SC18409_OUT_DIR",
        &format!(
            "{}/SceneWorks/render-validation-sc18409",
            std::env::var("HOME").expect("HOME")
        ),
    ));
    std::fs::create_dir_all(&dir).expect("artifact dir");
    dir
}

fn write_log(name: &str, text: &str) -> PathBuf {
    let path = out_dir().join(format!("{name}.log"));
    std::fs::write(&path, text).expect("write log");
    eprintln!("[sc18409] log -> {}", path.display());
    path
}

fn record_row(row: &serde_json::Value) {
    let line = serde_json::to_string(row).expect("row serializes");
    eprintln!("[[SC18409]] {line}");
    let path = out_dir().join("evidence.jsonl");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .expect("evidence file");
    writeln!(file, "{line}").expect("write evidence row");
}

/// sc-18409. Validate the changed cold-load path on real weights.
///
/// The subject is the bf16 tier: it is the tier VERIFIED on this Mac, and
/// `with_selected_sequential_shape` keys on the ENGINE id alone, so bf16 exercises the identical
/// changed branch the shipped q4 tier would take. The shipped `z_image_turbo` bindings all name
/// tier q4 at 768², so this bf16 request carries no provenance and is admitted through the
/// estimate ladder — which is fine: the subject under test is the LOAD-time gate and the load +
/// render it admits, not the evidence path.
///
/// The cap is DERIVED, not guessed: the test walks integer caps downward through the real
/// `apply_residency_policy` and uses the highest cap at which the gate itself selects Sequential
/// (override with `SC18409_CAP_GB`). That anchors the scenario to the gate's own manifest- and
/// disk-derived arithmetic, so a weights or headroom change moves the cap instead of silently
/// invalidating the scenario.
#[test]
#[ignore = "sc-18409: real-weight MLX render; needs SceneWorks/z-image-turbo-mlx bf16 cached"]
fn z_image_deferred_sequential_cold_load_renders_within_the_admitted_ceiling() {
    let Some((tier_dir, revision)) = cached_tier_dir(REPO_DIR, TIER, SENTINEL) else {
        panic!("SKIP-AS-FAILURE: no {REPO_DIR} {TIER} weights cached; sc-18409 cannot be validated without them");
    };
    let entry = shipped_manifest_entry(ENGINE);
    // No shipped binding names the bf16 tier, so there is no provenance to prove — and if one ever
    // appears this scenario must start proving it (mirrors sc-18101's provenance discipline).
    let has_bf16_binding = entry
        .get("mlx")
        .and_then(serde_json::Value::as_object)
        .and_then(|mlx| mlx.get("calibrations"))
        .and_then(serde_json::Value::as_array)
        .is_some_and(|items| {
            items
                .iter()
                .any(|item| item.get("tier").and_then(serde_json::Value::as_str) == Some(TIER))
        });
    assert!(
        !has_bf16_binding,
        "a shipped z_image_turbo bf16 binding now exists; supply its provenance here instead of None"
    );

    // The production base spec for a bf16 z_image_turbo directory load: dense (no quant), and the
    // default EAGER shape — z_image_turbo is not one of `image_jobs`'s deferred-materialization
    // routes, so before PR #2215 nothing on this path ever set the deferred shape.
    let base = LoadSpec::new(WeightsSource::Dir(tier_dir.clone()));
    assert!(
        matches!(base.load_shape, gen_core::LoadShape::EagerMaterialization),
        "the pre-policy spec must be eager, or the deferred assertion below proves nothing"
    );

    let policy_at = |cap_gb: f64| {
        crate::test_env::temp_env_var("SCENEWORKS_MLX_MEMORY_CAP_GB", &format!("{cap_gb}"), || {
            with_captured_tracing(|| {
                crate::mlx_fit_gate::apply_residency_policy(base.clone(), ENGINE)
            })
        })
    };

    // Derive the cap from the gate's own decisions: the highest integer cap that selects
    // Sequential. Walking downward also records the Resident cap immediately above it, proving the
    // scenario sits at the genuine Resident/Sequential boundary rather than in Reject territory.
    let cap_gb: f64 = match crate::smoke_support::env_or("SC18409_CAP_GB", "").parse::<f64>() {
        Ok(cap) => cap,
        Err(_) => {
            let mut derived = None;
            for cap in (8..=128).rev() {
                let cap = cap as f64;
                let (result, _) = policy_at(cap);
                match result {
                    Ok(spec) if spec.offload_policy == gen_core::OffloadPolicy::Sequential => {
                        derived = Some(cap);
                        break;
                    }
                    Ok(_) => continue, // Resident — keep descending.
                    Err(error) => panic!(
                        "the sweep hit Reject at {cap} GB before ever selecting Sequential: {error}"
                    ),
                }
            }
            let cap = derived.expect("some cap in 8..=128 GB must select Sequential");
            let (above, _) = policy_at(cap + 1.0);
            assert_eq!(
                above.expect("cap+1 must admit").offload_policy,
                gen_core::OffloadPolicy::Resident,
                "the derived cap must sit at the Resident/Sequential boundary"
            );
            cap
        }
    };

    // THE changed path (PR #2215): under this cap the Sequential branch fires, and for
    // z_image_turbo `with_selected_sequential_shape` must stamp DeferredMaterialization onto the
    // spec the cold load uses.
    let (policy_result, policy_log) = policy_at(cap_gb);
    let spec = policy_result.expect("the derived cap must admit the load");
    assert!(
        policy_log.contains("mlx_sequential_residency_selected"),
        "the Sequential branch must be the one that fired:\n{policy_log}"
    );
    assert_eq!(
        spec.offload_policy,
        gen_core::OffloadPolicy::Sequential,
        "sc-18409 exercises the Sequential branch"
    );
    assert!(
        matches!(
            spec.load_shape,
            gen_core::LoadShape::DeferredMaterialization
        ),
        "PR #2215's coupling must hold: the Sequential branch stamps DeferredMaterialization for {ENGINE}"
    );
    write_log("policy-selection", &policy_log);
    eprintln!(
        "[sc18409] cap={cap_gb} GB -> Sequential + DeferredMaterialization for {ENGINE} {TIER} \
         from {} (revision {revision})",
        tier_dir.display()
    );

    // Load with EXACTLY that spec and drive the real request seam under the same cap.
    let width: u32 = crate::smoke_support::env_or("SC18409_W", "1024")
        .parse()
        .expect("SC18409_W");
    let height: u32 = crate::smoke_support::env_or("SC18409_H", "1024")
        .parse()
        .expect("SC18409_H");
    let request_inputs = inputs(width, height);
    let plan = MlxRequestPlan::for_spec_and_manifest(ENGINE, ENGINE, &spec, Some(&entry), None);
    let generator = crate::inference_runtime::load(ENGINE, &spec).expect("load z_image_turbo");
    let (result, log) =
        crate::test_env::temp_env_var("SCENEWORKS_MLX_MEMORY_CAP_GB", &format!("{cap_gb}"), || {
            with_captured_tracing(|| {
                crate::mlx_fit_gate::evaluate_request(
                    &*generator,
                    &plan,
                    &request_inputs,
                    gen_core::MemoryCacheState::Cold,
                    spec.offload_policy,
                    0,
                )
            })
        });
    write_log("request-selection", &log);
    let evaluation = result.unwrap_or_else(|error| {
        panic!("sc-18409: the request must be ADMITTED under the {cap_gb} GB cap, not refused: {error}")
    });
    let strategy = evaluation.context.selection.strategy;
    let raw_peak_bytes = evaluation.context.predicted_peak_bytes;

    // The admitted ceiling: for an estimate-backed selection it is the WIDENED peak the admission
    // arithmetic graded against (sc-18101's convention); a measured selection is graded raw.
    let widened = field_in_line(
        &log,
        "memory-strategy selection uses an estimate-backed candidate at the widened peak",
        "widened_peak_bytes=",
    );
    let admitted_ceiling_bytes = widened.unwrap_or(raw_peak_bytes);

    let render_memory = evaluation.memory;
    let request = gen_core::GenerationRequest {
        prompt: crate::smoke_support::env_or(
            "SC18409_PROMPT",
            "a lighthouse on a rocky headland under storm clouds, late golden light",
        ),
        width,
        height,
        count: 1,
        seed: Some(18409),
        steps: Some(
            crate::smoke_support::env_or("SC18409_STEPS", "4")
                .parse()
                .expect("SC18409_STEPS"),
        ),
        // Z-Image Turbo is guidance-distilled: no CFG.
        guidance: None,
        memory: Some(render_memory),
        ..Default::default()
    };

    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let resident_before = mlx_rs::memory::get_active_memory() as u64;
    let started = std::time::Instant::now();
    let image = match generator
        .generate(&request, &mut |_| {})
        .expect("sc-18409: the admitted Sequential+Deferred load must complete a real render")
    {
        gen_core::GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected images, got {other:?}"),
    };
    let elapsed = started.elapsed();
    let observed_peak_bytes = mlx_rs::memory::get_peak_memory() as u64;
    mlx_rs::memory::clear_cache();
    let resident_after = mlx_rs::memory::get_active_memory() as u64;

    let png = out_dir().join(format!("z-image-{TIER}-{width}x{height}.png"));
    crate::smoke_support::save_png(&image, &png);
    let std_dev = crate::smoke_support::image_std(&image);

    record_row(&serde_json::json!({
        "story": "sc-18409",
        "engine": ENGINE,
        "tier": TIER,
        "artifactRevision": revision,
        "geometry": format!("{width}x{height}"),
        "capGb": cap_gb,
        "loadPolicy": format!("{:?}", spec.offload_policy),
        "loadShape": format!("{:?}", spec.load_shape),
        "rung": format!("{strategy:?}"),
        "parameters": format!("{:?}", evaluation.context.selection.parameters),
        "admissionPath": admission_path_line(&log),
        "rawPredictedPeakBytes": raw_peak_bytes,
        "rawPredictedPeakGib": gib(raw_peak_bytes),
        "admittedCeilingBytes": admitted_ceiling_bytes,
        "admittedCeilingGib": gib(admitted_ceiling_bytes),
        "ceilingIsWidened": widened.is_some(),
        "observedPeakBytes": observed_peak_bytes,
        "observedPeakGib": gib(observed_peak_bytes),
        "residentBeforeGib": gib(resident_before),
        "residentAfterGib": gib(resident_after),
        "renderSeconds": elapsed.as_secs_f64(),
        "steps": request.steps,
        "imageStd": std_dev,
        "png": png.display().to_string(),
    }));

    assert_eq!((image.width, image.height), (width, height));
    assert!(
        !crate::smoke_support::is_all_zero(&image),
        "sc-18409: the render is entirely black"
    );
    assert!(
        std_dev > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
        "sc-18409: the render looks degenerate (std {std_dev:.2})"
    );
    assert!(
        observed_peak_bytes <= admitted_ceiling_bytes,
        "CEILING BREACH: observed peak {:.2} GiB EXCEEDS the admitted ceiling {:.2} GiB \
         (rung {strategy:?}, cap {cap_gb} GB). Do not tune anything to make this pass — report it.",
        gib(observed_peak_bytes),
        gib(admitted_ceiling_bytes),
    );
}
