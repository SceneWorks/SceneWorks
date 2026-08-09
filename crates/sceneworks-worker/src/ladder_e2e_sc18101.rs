//! On-device end-to-end validation of the epic 18093 memory-ladder apparatus (sc-18101).
//!
//! Every test here is `#[ignore]`d: each one needs real weights in the HF hub cache and a real
//! Apple-Silicon Metal device, and the peak counters they sample
//! (`mlx_rs::memory::{reset_peak_memory, get_peak_memory}`) are PROCESS-GLOBAL, so a peak-bearing
//! test must own its process. Run one test per `cargo test` invocation.
//!
//! The four scenarios map 1:1 onto the epic's success criteria:
//!
//! | test | criterion |
//! |---|---|
//! | [`c1_unmeasured_cell_engages_a_deep_rung_and_renders`] | an UNMEASURED cell under `SCENEWORKS_MLX_MEMORY_CAP_GB` engages a deep rung and completes a real render |
//! | [`c2_measured_current_cell_selection`] | a MEASURED-CURRENT cell's selection is unchanged from pre-epic main |
//! | [`c3_stale_lane_admits_at_the_widened_peak`] | a stale-closure lane still admits, at the widened peak, and logs it |
//! | [`c4_oversized_request_is_refused_not_oom_killed`] | a genuinely oversized request is refused with the actionable message |
//!
//! ## Why this file compiles on pre-epic main too
//!
//! [`c2_measured_current_cell_selection`] is a CROSS-COMMIT driver: the same source is dropped into
//! a checkout of the epic's base commit and run there, and the two selection logs are diffed. It
//! therefore touches only API that both revisions share — `mlx_fit_gate::{MlxRequestPlan,
//! MlxRequestInputs, evaluate_request}`, `inference_runtime::load`, `test_env::temp_env_vars`. In
//! particular it does NOT import [`crate::ladder_margin_policy`] (which does not exist before
//! sc-18094); the margins appear here as literals, pinned to the policy module by
//! [`margin_literals_match_the_policy`] below, which is the one test in this file that is neither
//! ignored nor portable.
//!
//! ## What the cap does and does not emulate
//!
//! `SCENEWORKS_MLX_MEMORY_CAP_GB` replaces the machine's real unified-memory total in
//! `mlx_fit_gate::resolve_budget`, so it drives ADMISSION arithmetic exactly as a smaller Mac
//! would. It does not shrink the Metal heap: the render that follows still runs against the real
//! 128 GiB pool, so an observed peak measured under a cap is the peak that geometry and rung
//! genuinely cost, and re-materialization costs on a machine that really was that small are a
//! LOWER bound on what this measures. That is the right direction for validating a margin — the
//! emulated run cannot flatter the prediction.

use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use gen_core::{LoadSpec, WeightsSource};

use crate::mlx_fit_gate::{MlxRequestInputs, MlxRequestPlan};

/// Where renders, selection logs, and the machine-readable evidence rows are written. Stable and
/// outside the repo so a run's artifacts survive a `git clean` and can be diffed across commits.
fn out_dir() -> PathBuf {
    let dir = PathBuf::from(crate::smoke_support::env_or(
        "SC18101_OUT_DIR",
        &format!(
            "{}/SceneWorks/render-validation-sc18101",
            std::env::var("HOME").expect("HOME")
        ),
    ));
    std::fs::create_dir_all(&dir).expect("evidence dir");
    dir
}

/// A tag distinguishing one run's artifacts from another's — set to the commit under test by the
/// cross-commit driver so a main run and a baseline run land side by side.
fn run_tag() -> String {
    crate::smoke_support::env_or("SC18101_TAG", "run")
}

/// Tee `tracing` output to stderr (so `--nocapture` shows it live) and into a buffer the test can
/// assert on. `with_default` scopes it to one closure, which keeps the capture attributable to a
/// single `evaluate_request` call rather than to the whole process.
#[derive(Clone, Default)]
struct Tee(Arc<Mutex<Vec<u8>>>);

impl Write for Tee {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        std::io::stderr().write_all(buf)?;
        self.0.lock().expect("capture lock").extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stderr().flush()
    }
}

impl Tee {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().expect("capture lock")).into_owned()
    }
}

/// Run `body` with an INFO-level subscriber whose output is both printed and captured.
///
/// Field ordering inside a `tracing` event is the macro's, not a map's, so the captured text is
/// deterministic for a given build — which is what makes the cross-commit diff meaningful.
fn with_captured_tracing<T>(body: impl FnOnce() -> T) -> (T, String) {
    let tee = Tee::default();
    let writer = tee.clone();
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(move || writer.clone())
        .with_ansi(false)
        .with_target(false)
        .without_time()
        .finish();
    let value = tracing::subscriber::with_default(subscriber, body);
    (value, tee.text())
}

/// The HF hub snapshot subdirectory for one repo + tier, or `None` when that tier was never pulled.
///
/// A tier DIRECTORY existing is not proof the tier is present, so this probes for the weights file
/// the loader actually opens rather than for the directory.
fn cached_tier_dir(repo_dir: &str, tier: &str, sentinel: &str) -> Option<(PathBuf, String)> {
    let snapshots = PathBuf::from(std::env::var("HOME").ok()?)
        .join(".cache/huggingface/hub")
        .join(repo_dir)
        .join("snapshots");
    std::fs::read_dir(&snapshots).ok()?.flatten().find_map(|e| {
        let revision = e.file_name().to_str()?.to_owned();
        let dir = e.path().join(tier);
        dir.join(sentinel).is_file().then_some((dir, revision))
    })
}

/// The shipped manifest entry for one catalog model id.
fn shipped_manifest_entry(model_id: &str) -> serde_json::Map<String, Value> {
    let raw = include_str!("../../../config/manifests/builtin.models.jsonc");
    let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
        .expect("builtin.models.jsonc parses");
    manifest
        .get("models")
        .and_then(Value::as_array)
        .expect("models array")
        .iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some(model_id))
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("no manifest entry for {model_id}"))
        .clone()
}

/// The `Quant` a tier token names, or `None` for the dense tier. Mirrors what
/// `image_jobs::base::resolve_quant` derives from a manifest's `mlxQuantize`; spelled out here so a
/// tier override on the command line resolves the same way the router would.
fn quant_for_tier(tier: &str) -> Option<gen_core::Quant> {
    match tier {
        "q4" => Some(gen_core::Quant::Q4),
        "q8" => Some(gen_core::Quant::Q8),
        _ => None,
    }
}

/// Routes for which `image_jobs::apply_measured_mlx_load_shape` (`image_jobs/base.rs`) forces
/// `LoadShape::DeferredMaterialization` on a directory load — mirrored here rather than called.
///
/// WHY MIRRORED, given that calling the real function would be strictly better: `base.rs` is one of
/// the sixteen sources `docs/generated/memory-matrix.json` fingerprints (`SOURCE_PATHS.imageRouting`
/// in `scripts/generate-memory-matrix.mjs`). Widening that function's visibility by one keyword
/// rotates `generatedFrom.sceneWorksRevision`, which reds `npm run check:memory-matrix` until the
/// whole 860 KB artifact and every manifest binding's `matrixSourceRevision` are regenerated — a
/// large, merge-queue-hostile diff for a test-only visibility change. A test harness is not worth
/// that, so the predicate is restated here and [`load_shape_mirror_matches_the_documented_routes`]
/// keeps the restatement honest.
///
/// This matters because the load shape decides which rungs exist at all: mlx-gen-z-image and
/// mlx-gen-lens declare `BoundedTransformerResidency` `Implemented` only under
/// `DeferredMaterialization` (mlx-gen-krea gates its own on `OffloadPolicy::Sequential` instead,
/// which comes from the load-time residency gate rather than from here).
const DEFERRED_MATERIALIZATION_ROUTES: &[&str] =
    &["qwen_image", "qwen_image_edit", "lens", "lens_turbo"];

/// Build the `LoadSpec` production would build for one engine + tier directory, up to (but not
/// including) the load-time residency decision.
fn production_spec(engine: &str, tier_dir: &std::path::Path, tier: &str) -> LoadSpec {
    let quant = quant_for_tier(tier);
    let mut spec = LoadSpec::new(WeightsSource::Dir(tier_dir.to_path_buf()));
    if let Some(quant) = quant {
        spec = spec.with_quant(quant);
    }
    // `base.rs`'s `lens_native` arm additionally requires `lens` to be q4 and `lens_turbo` to be
    // dense; its `qwen_native` arm accepts any tier. Neither arm fires unless the weights are a
    // directory and the precision is bf16, which is true of every spec this harness builds.
    let deferred = match engine {
        "lens" => quant == Some(gen_core::Quant::Q4),
        "lens_turbo" => quant.is_none(),
        other => DEFERRED_MATERIALIZATION_ROUTES.contains(&other),
    };
    if deferred {
        spec = spec.with_load_shape(gen_core::LoadShape::DeferredMaterialization);
    }
    spec
}

fn inputs(width: u32, height: u32) -> MlxRequestInputs {
    MlxRequestInputs {
        width,
        height,
        count: 1,
        mode: "text_to_image".to_owned(),
        overlay: None,
        adapter_count: 0,
        has_reference: false,
        reference_count: 0,
        use_pid: false,
        has_phases: false,
    }
}

/// Append one machine-readable row to the run's evidence file and echo it.
fn record_row(row: &Value) {
    let line = serde_json::to_string(row).expect("row serializes");
    eprintln!("[[SC18101]] {line}");
    let path = out_dir().join(format!("evidence-{}.jsonl", run_tag()));
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .expect("evidence file");
    writeln!(file, "{line}").expect("write evidence row");
}

/// Write the captured selection log verbatim, so the cross-commit diff is over a file rather than
/// over scrollback.
fn write_log(name: &str, text: &str) -> PathBuf {
    let path = out_dir().join(format!("{name}-{}.log", run_tag()));
    std::fs::write(&path, text).expect("write selection log");
    eprintln!("[sc18101] selection log -> {}", path.display());
    path
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB
}

// ---------------------------------------------------------------------------------------------
// Criterion 1 — an UNMEASURED cell under an emulated cap engages a deep rung and really renders.
// ---------------------------------------------------------------------------------------------

/// The unmeasured subject: `krea_2_turbo` bf16, text-to-image, no overlay.
///
/// Why this cell is genuinely unmeasured rather than merely uncalibrated-for-this-geometry: the
/// whole MLX corpus contains exactly three providers (`qwen_image`, `z_image_turbo`,
/// `krea_2_turbo_control`). `krea_2_turbo` — the plain text-to-image route, no control overlay —
/// has ZERO evidence records and ZERO manifest bindings at ANY tier or geometry. The shipped
/// `krea_2_turbo` manifest entry does declare `mlx.calibrations`, but every binding in it names
/// provider `krea_2_turbo_control` at q4 with `overlay: "control:1"`, so the identity filter in
/// `evidence_admission_route` matches nothing and the request routes to `AdmissionPath::Legacy`
/// with no measured basis at all. That is precisely the epic 18093 R1 "weights + headroom floor"
/// path.
///
/// Why bf16: it is the only tier of `SceneWorks/krea-2-turbo-mlx` in this machine's cache, and
/// Krea 2 Turbo is a CFG-free distilled model, so a real render is 8 steps.
///
/// Overridable with `SC18101_C1_{REPO,ENGINE,TIER}` so the same driver can probe a sibling route
/// without a rebuild.
fn c1_repo_dir() -> String {
    crate::smoke_support::env_or("SC18101_C1_REPO", "models--SceneWorks--krea-2-turbo-mlx")
}

fn c1_engine() -> &'static str {
    // The plan wants a `&'static str` engine id, so an override is resolved against the registry's
    // own ids rather than leaked from a `String`.
    let requested = crate::smoke_support::env_or("SC18101_C1_ENGINE", C1_ENGINE);
    crate::inference_runtime::generators()
        .map(|registration| (registration.descriptor)().id)
        .find(|id| *id == requested)
        .unwrap_or_else(|| panic!("no registered MLX generator with id {requested}"))
}

fn c1_tier() -> String {
    crate::smoke_support::env_or("SC18101_C1_TIER", "bf16")
}

const C1_ENGINE: &str = "krea_2_turbo";
/// The loader's own entry file. A tier DIRECTORY existing proves nothing (a half-deleted or
/// half-downloaded tier leaves the directory behind), and the Krea bf16 transformer is sharded, so
/// there is no single `diffusion_pytorch_model.safetensors` to probe for either.
const C1_SENTINEL: &str = "model_index.json";

/// sc-18101 criterion 1. Drive the REAL production seam — `mlx_fit_gate::evaluate_request`, the
/// call `image_jobs/base.rs` makes immediately before every generation — on a live loaded provider
/// for a cell nobody has ever measured, under an emulated unified-memory cap that no shallow rung
/// fits, and then actually render at whatever rung the estimate ladder picked.
///
/// `SC18101_C1_CAP_GB` selects the emulated cap. Leave it unset to run the CAP SWEEP first: the
/// test prints, for a ladder of caps, which rung each one selects and at what widened peak, then
/// fails with the sweep table so the operator can pin a cap. With the cap set it asserts a deep
/// rung, renders, samples `mlx_rs::memory::get_peak_memory`, and compares the observed peak to the
/// predicted-with-margin ceiling the gate admitted against.
///
/// ## The cap is applied to the WHOLE production chain, not just the request ladder
///
/// Which rungs a provider even declares `Implemented` is a function of the `LoadSpec` it was loaded
/// with, not of the request: mlx-gen-krea only advertises `BoundedTransformerResidency` when the
/// spec carries `OffloadPolicy::Sequential`, and mlx-gen-z-image / mlx-gen-lens gate theirs on
/// `LoadShape::DeferredMaterialization`. Production reaches those states through the LOAD-time gate
/// (`mlx_fit_gate::apply_residency_policy`, called on the generator cache's cold-load path), which
/// reads the same cap. So this test runs the cap through that gate FIRST and loads with whatever
/// spec it returns, exactly as a cold load on a genuinely smaller Mac would. Skipping that step and
/// loading `Resident` would produce a flat three-rung ladder whose floors are all equal, and the
/// "deep rung" would be unreachable for reasons that have nothing to do with epic 18093.
///
/// ```text
/// SC18101_C1_CAP_GB=<gb> cargo test -p sceneworks-worker --lib -- \
///     --ignored --nocapture c1_unmeasured_cell_engages_a_deep_rung_and_renders
/// ```
#[test]
#[ignore = "sc-18101 criterion 1: real-weight MLX render; needs SceneWorks/krea-2-turbo-mlx bf16 cached"]
fn c1_unmeasured_cell_engages_a_deep_rung_and_renders() {
    let (repo_dir, engine, tier) = (c1_repo_dir(), c1_engine(), c1_tier());
    let Some((tier_dir, revision)) = cached_tier_dir(&repo_dir, &tier, C1_SENTINEL) else {
        panic!("SKIP-AS-FAILURE: no {repo_dir} {tier} weights cached; criterion 1 cannot be validated without them");
    };
    let entry = shipped_manifest_entry(engine);
    let width: u32 = crate::smoke_support::env_or("SC18101_C1_W", "1024")
        .parse()
        .expect("SC18101_C1_W");
    let height: u32 = crate::smoke_support::env_or("SC18101_C1_H", "1024")
        .parse()
        .expect("SC18101_C1_H");
    let request_inputs = inputs(width, height);

    // One whole cold-load-plus-request pass under one cap. The generator is rebuilt per cap because
    // the residency policy the cap chooses is part of the `LoadSpec` the contract is derived from.
    // MLX materializes lazily, so a rebuilt generator costs a directory scan, not a re-read of the
    // weights.
    let attempt = |cap_gb: f64| {
        crate::test_env::temp_env_var("SCENEWORKS_MLX_MEMORY_CAP_GB", &format!("{cap_gb}"), || {
            with_captured_tracing(|| {
                let base = production_spec(engine, &tier_dir, &tier);
                let spec = crate::mlx_fit_gate::apply_residency_policy(base, engine)?;
                let load_policy = spec.offload_policy;
                // No `resolved_artifact`: the shipped `krea_2_turbo` bindings name provider
                // `krea_2_turbo_control` at q4 with a control overlay, so there is nothing for
                // provenance to prove about THIS cell. The plan therefore carries no covering
                // calibration and the request is admitted from synthesized estimates — the state
                // under test.
                let plan = MlxRequestPlan::for_spec_and_manifest(
                    engine,
                    engine,
                    &spec,
                    Some(&entry),
                    None,
                );
                let generator = crate::inference_runtime::load(engine, &spec)
                    .map_err(|error| crate::WorkerError::InvalidPayload(error.to_string()))?;
                let evaluation = crate::mlx_fit_gate::evaluate_request(
                    &*generator,
                    &plan,
                    &request_inputs,
                    gen_core::MemoryCacheState::Cold,
                    load_policy,
                    0,
                )?;
                Ok::<_, crate::WorkerError>((generator, evaluation, load_policy))
            })
        })
    };

    let Ok(cap_gb) = crate::smoke_support::env_or("SC18101_C1_CAP_GB", "").parse::<f64>() else {
        let mut table = String::from("\ncap_gb  outcome\n");
        for cap_gb in sweep_caps() {
            let (result, _log) = attempt(cap_gb);
            let outcome = match &result {
                Ok((_, evaluation, load_policy)) => format!(
                    "{:?} load_policy={load_policy:?} params={:?} predicted_peak={:.2} GiB",
                    evaluation.context.selection.strategy,
                    evaluation.context.selection.parameters,
                    gib(evaluation.context.predicted_peak_bytes),
                ),
                Err(error) => format!("REFUSED: {error}"),
            };
            table.push_str(&format!("{cap_gb:6.1}  {outcome}\n"));
        }
        panic!("SC18101_C1_CAP_GB is unset — cap sweep follows; pin a cap that selects a deep rung:{table}");
    };

    let (result, log) = attempt(cap_gb);
    let (generator, evaluation, load_policy) = result.unwrap_or_else(|error| {
        panic!("criterion 1: the estimate ladder must ADMIT under a {cap_gb} GB cap, not refuse: {error}")
    });
    eprintln!(
        "[sc18101/c1] {engine} {tier} from {} (revision {revision}), load_policy={load_policy:?}",
        tier_dir.display()
    );
    write_log("c1-selection", &log);

    let strategy = evaluation.context.selection.strategy;
    assert!(
        strategy.is_optimized(),
        "criterion 1 requires a DEEP rung; the ladder selected {strategy:?}"
    );
    // The estimate margin must be visible in the selection, not merely believed. Both the
    // per-candidate admission line and the selection line name the basis and both peaks.
    assert!(
        log.contains("estimate-backed memory-strategy candidate admitted with widened margin"),
        "criterion 1 requires estimate-scoped admission in the logs:\n{log}"
    );
    assert!(
        log.contains(
            "memory-strategy selection uses an estimate-backed candidate at the widened peak"
        ),
        "criterion 1 requires the SELECTION to be estimate-scoped:\n{log}"
    );
    assert!(
        log.contains("estimate_margin=0.1"),
        "criterion 1 requires the applied MLX estimate margin (0.10) in the logs:\n{log}"
    );

    // The admitted ceiling is the WIDENED peak, not `context.predicted_peak_bytes`.
    //
    // `mlx_fit_gate.rs:2404-2417` deliberately puts the estimate's RAW peak on the run context: the
    // context carries the rung's incremental working-set demand, while the margin lives in the
    // admission arithmetic (`memory_strategy::select_strategy` grades against
    // `raw * (1 + estimate_margin)` and the refusal message quotes that widened number). Comparing
    // the observed peak against the raw context value would be checking the prediction WITHOUT its
    // margin — a stricter test than the policy promises, and not the one the story specifies. Both
    // numbers are recorded; the assertion is against the widened one.
    let raw_peak_bytes = field_in_line(
        &log,
        "memory-strategy selection uses an estimate-backed candidate at the widened peak",
        "raw_peak_bytes=",
    )
    .expect("the selection event carries raw_peak_bytes");
    let admitted_ceiling_bytes = field_in_line(
        &log,
        "memory-strategy selection uses an estimate-backed candidate at the widened peak",
        "widened_peak_bytes=",
    )
    .expect("the selection event carries widened_peak_bytes");
    assert_eq!(
        admitted_ceiling_bytes,
        (raw_peak_bytes as f64 * 1.1).ceil() as u64,
        "the admitted ceiling must be the raw estimate times the 10% MLX estimate margin"
    );

    let render_memory = evaluation.memory;
    let request = gen_core::GenerationRequest {
        prompt: crate::smoke_support::env_or(
            "SC18101_C1_PROMPT",
            "a windswept basalt sea stack at dawn, low mist, long exposure water",
        ),
        width,
        height,
        count: 1,
        seed: Some(18101),
        steps: Some(
            crate::smoke_support::env_or("SC18101_C1_STEPS", "8")
                .parse()
                .expect("SC18101_C1_STEPS"),
        ),
        // Krea 2 Turbo is CFG-free (distilled).
        guidance: None,
        memory: Some(render_memory),
        ..Default::default()
    };

    // Re-base the process-global high-water mark so the sampled peak is THIS render's, on top of
    // the already-resident weights, rather than the load's.
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let resident_before = mlx_rs::memory::get_active_memory() as u64;
    let started = std::time::Instant::now();
    let image = match generator
        .generate(&request, &mut |_| {})
        .expect("criterion 1: the admitted deep rung must complete a real render")
    {
        gen_core::GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected images, got {other:?}"),
    };
    let elapsed = started.elapsed();
    let observed_peak_bytes = mlx_rs::memory::get_peak_memory() as u64;
    mlx_rs::memory::clear_cache();
    let resident_after = mlx_rs::memory::get_active_memory() as u64;

    let path = out_dir().join(format!(
        "c1-{engine}-{tier}-{width}x{height}-{}.png",
        run_tag()
    ));
    crate::smoke_support::save_png(&image, &path);
    let std_dev = crate::smoke_support::image_std(&image);

    record_row(&serde_json::json!({
        "criterion": 1,
        "engine": engine,
        "tier": tier,
        "artifactRevision": revision,
        "geometry": format!("{width}x{height}"),
        "capGb": cap_gb,
        "rung": format!("{strategy:?}"),
        "parameters": format!("{:?}", evaluation.context.selection.parameters),
        "loadPolicy": format!("{load_policy:?}"),
        "rawEstimateBytes": raw_peak_bytes,
        "rawEstimateGib": gib(raw_peak_bytes),
        "admittedCeilingBytes": admitted_ceiling_bytes,
        "admittedCeilingGib": gib(admitted_ceiling_bytes),
        "estimateMargin": 0.10,
        "contextPredictedPeakGib": gib(evaluation.context.predicted_peak_bytes),
        "observedPeakBytes": observed_peak_bytes,
        "observedPeakGib": gib(observed_peak_bytes),
        "residentBeforeGib": gib(resident_before),
        "residentAfterGib": gib(resident_after),
        "renderSeconds": elapsed.as_secs_f64(),
        "imageStd": std_dev,
        "png": path.display().to_string(),
    }));

    assert_eq!((image.width, image.height), (width, height));
    assert!(
        !crate::smoke_support::is_all_zero(&image),
        "criterion 1: the render is entirely black"
    );
    assert!(
        std_dev > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
        "criterion 1: the render looks degenerate (std {std_dev:.2})"
    );

    // THE margin check the story makes load-bearing. An overrun here is a margin-derivation defect
    // in sc-18094 and must reopen that story — it must never be papered over by widening a margin
    // here.
    assert!(
        observed_peak_bytes <= admitted_ceiling_bytes,
        "MARGIN BREACH (reopen sc-18094): observed peak {:.2} GiB EXCEEDS the admitted \
         predicted-with-margin ceiling {:.2} GiB at rung {strategy:?} under a {cap_gb} GB cap. \
         Do not widen a margin to make this pass.",
        gib(observed_peak_bytes),
        gib(admitted_ceiling_bytes),
    );
}

/// The cap ladder the criterion-1 sweep walks when no cap is pinned. Coarse at the top (where the
/// resident baseline lives) and fine at the bottom (where the deep rungs separate).
fn sweep_caps() -> Vec<f64> {
    let mut caps = vec![
        128.0, 96.0, 80.0, 72.0, 64.0, 56.0, 48.0, 44.0, 40.0, 36.0, 32.0,
    ];
    caps.extend([
        30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0,
    ]);
    caps
}

// ---------------------------------------------------------------------------------------------
// Criteria 2 and 3 — the measured `qwen_image` q8 1024² cell, current and stale.
// ---------------------------------------------------------------------------------------------

const MEASURED_REPO_DIR: &str = "models--SceneWorks--qwen-image-mlx";
const MEASURED_ENGINE: &str = "qwen_image";
const MEASURED_TIER: &str = "q8";
const MEASURED_SENTINEL: &str = "model_index.json";

/// Build the plan for the measured `qwen_image` q8 1024² cell from the SHIPPED manifest binding and
/// the snapshot actually on disk, optionally rewriting the binding's `inferenceClosureDigest`.
///
/// `closure_digest_override` is the currency seam. The worker compares
/// `binding.query.inference_closure_digest` (mlx_fit_gate.rs:1257 — the candidate's digest comes
/// from the BINDING, not from the evidence record) against
/// `sceneworks_core::memory_calibration::packaged_closure_digest`. Rewriting the binding to the
/// live digest is exactly the edit `docs/calibration-runbook.md` §7d prescribes for moving a lane
/// to `current`, applied here in memory instead of to the shipped file — so a measured-CURRENT cell
/// can be exercised on a corpus that today contains none.
fn measured_plan() -> Option<(MlxRequestPlan, LoadSpec, PathBuf, String, String)> {
    let (tier_dir, revision) =
        cached_tier_dir(MEASURED_REPO_DIR, MEASURED_TIER, MEASURED_SENTINEL)?;
    let entry = shipped_manifest_entry(MEASURED_ENGINE);
    let calibrations = entry
        .get("mlx")
        .and_then(Value::as_object)
        .expect("qwen_image declares an mlx block")
        .get("calibrations")
        .and_then(Value::as_array)
        .expect("qwen_image declares mlx.calibrations");
    let binding = calibrations
        .iter()
        .find(|item| item.get("tier").and_then(Value::as_str) == Some(MEASURED_TIER))
        .expect("a q8 binding")
        .clone();
    let declared = |key: &str| {
        binding
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("binding is missing {key}"))
            .to_owned()
    };
    assert_eq!(
        revision,
        declared("artifactResolvedRevision"),
        "the on-disk snapshot must be the exact artifact the shipped opt-in names"
    );
    let provenance = crate::model_jobs::ResolvedArtifactProvenance {
        identity: crate::model_jobs::ResolvedArtifactIdentity {
            repository: declared("artifactRepository"),
            revision: revision.clone(),
            variant: declared("artifactVariant"),
            fingerprint: declared("resolvedPathFingerprint"),
        },
        fixed_artifact_tier: Some(declared("tier")),
    };
    // `SC18101_MEASURED_EAGER=1` skips `apply_measured_mlx_load_shape`, which forces
    // `DeferredMaterialization` on every `qwen_image` directory load. Which load shape the spec
    // carries decides WHICH shipped binding can serve the cell: the q8 ladder ships one
    // `bounded_attention` binding captured `eager_materialization` and one
    // `bounded_transformer_residency` binding captured `deferred_materialization`.
    let spec = if crate::smoke_support::env_or("SC18101_MEASURED_EAGER", "") == "1" {
        LoadSpec::new(WeightsSource::Dir(tier_dir.clone())).with_quant(gen_core::Quant::Q8)
    } else {
        production_spec(MEASURED_ENGINE, &tier_dir, MEASURED_TIER)
    };
    let plan = MlxRequestPlan::for_spec_and_manifest(
        MEASURED_ENGINE,
        MEASURED_ENGINE,
        &spec,
        Some(&entry),
        Some(provenance),
    );
    Some((
        plan,
        spec,
        tier_dir,
        revision,
        declared("inferenceClosureDigest"),
    ))
}

/// The live packaged closure digest for the MLX `qwen_image` lane — the value a binding must carry
/// to read as CURRENT.
fn live_qwen_closure_digest() -> String {
    sceneworks_core::memory_calibration::packaged_closure_digest("mlx", MEASURED_ENGINE)
        .expect("mlx:qwen_image is a declared lane")
}

/// sc-18101 criterion 2. A MEASURED-CURRENT cell must select exactly what pre-epic main selected.
///
/// ## Which reading of "measured-current" this uses, and why
///
/// The corpus as shipped has **zero** closure-current lanes (`npm run report:stale-lanes`: "8
/// stale, 0 current, 1 unmeasured"), so "run the same request on a calibrated cell" has two
/// possible readings:
///
/// 1. *measured and admitted* — take a cell as it ships. Rejected: every shipped MLX cell is
///    closure-STALE, and pre-epic main EXCLUDED stale candidates outright. Comparing that against
///    main would compare the epic's intended behaviour change with itself and prove nothing.
/// 2. *measured and closure-current* — the happy path the epic promises not to disturb. This is
///    the reading used here. Since no shipped lane is current, the cell is MADE current with a
///    SCRATCH CLOSURE TABLE: `config/inference-provider-closures.json` is temporarily edited so
///    the live digest for `mlx:qwen_image` equals the digest its q8 ladder was captured under,
///    i.e. the world in which nobody has touched the provider since the measurement. That is the
///    single edit that makes a cell current without disturbing anything else.
///
/// Moving the MANIFEST BINDING instead — the runbook's §7d edit — does NOT work for a test,
/// and finding that out is worth recording: `EvidenceBundle::evidence_for` compares the binding's
/// `inferenceClosureDigest` against the RECORD's own stamp as well, so rewriting only the binding
/// makes the record unfindable and the cell routes to `StaleIdentity` with no measured candidate
/// at all (observed: `path=Legacy fallback_reason=Some(StaleIdentity)`, selection falls to the
/// resident floor estimate). A real re-measurement moves both halves together, which is exactly
/// why the runbook insists on both.
///
/// Reading 2 is the only one that can distinguish "the epic left the calibrated path alone" from
/// "the epic changed everything", which is what the criterion is for.
///
/// The comparison itself is a file diff over the captured selection log, run twice from the same
/// source: once in this checkout and once in a checkout of the epic's base commit.
///
/// ```text
/// SC18101_TAG=main     cargo test -p sceneworks-worker --release --lib -- --ignored --nocapture c2_measured_current
/// SC18101_TAG=baseline cargo test -p sceneworks-worker --release --lib -- --ignored --nocapture c2_measured_current   # in the base checkout
/// diff <(…)-main.log <(…)-baseline.log
/// ```
#[test]
#[ignore = "sc-18101 criterion 2: needs SceneWorks/qwen-image-mlx q8 cached (~36 GB)"]
fn c2_measured_current_cell_selection() {
    let live = live_qwen_closure_digest();
    let Some((plan, spec, tier_dir, revision, binding_digest)) = measured_plan() else {
        panic!("SKIP-AS-FAILURE: no {MEASURED_REPO_DIR} {MEASURED_TIER} weights cached");
    };
    assert_eq!(
        live, binding_digest,
        "criterion 2 needs a measured-CURRENT cell, and the shipped corpus has none. Put the \
         scratch closure table in place first — set `providers[\"mlx:qwen_image\"].digest` in \
         config/inference-provider-closures.json to the digest the q8 ladder was captured under \
         ({binding_digest}) and rebuild — then re-run. Moving the BINDING instead does not work: \
         `EvidenceBundle::evidence_for` also compares the binding's digest against the record's \
         own stamp, so rewriting one half makes the record unfindable and the cell routes to \
         `StaleIdentity` with no measured candidate at all."
    );
    let request_inputs = inputs(1024, 1024);
    eprintln!(
        "[sc18101/c2] loading {MEASURED_ENGINE} {MEASURED_TIER} from {} (revision {revision})",
        tier_dir.display()
    );
    let generator =
        crate::inference_runtime::load(MEASURED_ENGINE, &spec).expect("load qwen_image");

    let cap_gb = crate::smoke_support::env_or("SC18101_C2_CAP_GB", "64");
    let (result, log) =
        crate::test_env::temp_env_var("SCENEWORKS_MLX_MEMORY_CAP_GB", &cap_gb, || {
            with_captured_tracing(|| {
                crate::mlx_fit_gate::evaluate_request(
                    &*generator,
                    &plan,
                    &request_inputs,
                    gen_core::MemoryCacheState::Warm,
                    gen_core::OffloadPolicy::Resident,
                    0,
                )
            })
        });

    // The selection fingerprint: everything the criterion says must be identical across the two
    // commits, in a stable single-line form a diff can compare without tripping over pointer
    // addresses or wall-clock noise.
    let fingerprint = match &result {
        Ok(evaluation) => format!(
            "SELECTED strategy={:?} parameters={:?} tier={:?} predicted_peak_bytes={} \
             process_limit_bytes={:?} stage_residency={} tile_vae_decode={} chunk_attention={} \
             stream_transformer_blocks={}\n",
            evaluation.context.selection.strategy,
            evaluation.context.selection.parameters,
            evaluation.context.selection.tier,
            evaluation.context.predicted_peak_bytes,
            evaluation.process_limit_bytes,
            evaluation.memory.stage_residency,
            evaluation.memory.tile_vae_decode,
            evaluation.memory.chunk_attention,
            evaluation.memory.stream_transformer_blocks,
        ),
        Err(error) => format!("REFUSED {error}\n"),
    };
    eprintln!("[sc18101/c2] {fingerprint}");
    write_log(
        "c2-selection",
        &format!("{fingerprint}\n--- tracing ---\n{log}"),
    );
    write_log("c2-fingerprint", &fingerprint);
    record_row(&serde_json::json!({
        "criterion": 2,
        "engine": MEASURED_ENGINE,
        "tier": MEASURED_TIER,
        "artifactRevision": revision,
        "geometry": "1024x1024",
        "capGb": cap_gb,
        "closureDigest": live,
        "currency": "measured-current (scratch closure table: the live lane digest set to the captured one)",
        "fingerprint": fingerprint.trim(),
    }));

    let evaluation = result.expect("criterion 2: a measured-current cell must be admitted");
    assert!(
        evaluation.process_limit_bytes.is_some(),
        "a measured-current cell is admitted through the Evidence path, which supplies a \
         request-scoped ceiling; `None` means it degraded to the legacy estimator"
    );
    // A current candidate is graded at its RAW peak: no widening line may appear.
    assert!(
        !log.contains("widened margin") && !log.contains("widened peak"),
        "criterion 2: a measured-CURRENT cell must be graded at its raw peak, unwidened:\n{log}"
    );
}

/// sc-18101 criterion 3. The shipped `mlx:qwen_image` lane is closure-STALE today (live
/// `9930aa53…` versus the binding's `54a7b45b…`). Under sc-18095 that no longer excludes it: the
/// candidate reaches the selector carrying the digest it was measured under and is graded at its
/// peak widened by `MLX_STALE_MEASURED_MARGIN` (5%).
///
/// The test proves the widening is REAL rather than merely logged by bracketing it: at a cap whose
/// effective budget sits between the raw peak and the widened peak the request must REFUSE, and at
/// a cap above the widened peak it must admit. A zeroed margin would admit both.
#[test]
#[ignore = "sc-18101 criterion 3: needs SceneWorks/qwen-image-mlx q8 cached (~36 GB)"]
fn c3_stale_lane_admits_at_the_widened_peak() {
    let Some((plan, spec, tier_dir, revision, binding_digest)) = measured_plan() else {
        panic!("SKIP-AS-FAILURE: no {MEASURED_REPO_DIR} {MEASURED_TIER} weights cached");
    };
    assert_ne!(
        live_qwen_closure_digest(),
        binding_digest,
        "criterion 3 needs a STALE lane. `mlx:qwen_image` is stale as shipped, so this only fires \
         if the scratch closure table from criterion 2 was left in place — revert \
         config/inference-provider-closures.json and rebuild."
    );
    let request_inputs = inputs(1024, 1024);
    eprintln!(
        "[sc18101/c3] loading {MEASURED_ENGINE} {MEASURED_TIER} from {} (revision {revision})",
        tier_dir.display()
    );
    let generator =
        crate::inference_runtime::load(MEASURED_ENGINE, &spec).expect("load qwen_image");

    let evaluate = |cap_gb: f64| {
        crate::test_env::temp_env_var("SCENEWORKS_MLX_MEMORY_CAP_GB", &format!("{cap_gb}"), || {
            with_captured_tracing(|| {
                crate::mlx_fit_gate::evaluate_request(
                    &*generator,
                    &plan,
                    &request_inputs,
                    gen_core::MemoryCacheState::Warm,
                    gen_core::OffloadPolicy::Resident,
                    0,
                )
            })
        })
    };

    let admit_cap: f64 = crate::smoke_support::env_or("SC18101_C3_ADMIT_CAP_GB", "96")
        .parse()
        .expect("SC18101_C3_ADMIT_CAP_GB");
    let (admitted, log) = evaluate(admit_cap);
    write_log("c3-selection", &log);
    let evaluation = admitted
        .unwrap_or_else(|error| panic!("criterion 3: a stale lane must still ADMIT: {error}"));

    assert!(
        log.contains("stale-closure memory-strategy candidate admitted with widened margin"),
        "criterion 3 requires the stale admission in the logs:\n{log}"
    );
    assert!(
        log.contains("memory-strategy selection uses stale-closure evidence at the widened peak"),
        "criterion 3 requires the SELECTION to be stale-scoped:\n{log}"
    );
    assert!(
        log.contains("stale_margin=0.05"),
        "criterion 3 requires the applied MLX stale margin (0.05) in the logs:\n{log}"
    );

    // Pull the raw and widened peaks straight out of the emitted event so the recorded numbers are
    // the gate's, not a restatement.
    const C3_SELECTION: &str =
        "memory-strategy selection uses stale-closure evidence at the widened peak";
    let raw_peak_bytes =
        field_in_line(&log, C3_SELECTION, "raw_peak_bytes=").expect("raw_peak_bytes in the log");
    let widened_peak_bytes = field_in_line(&log, C3_SELECTION, "widened_peak_bytes=")
        .expect("widened_peak_bytes in the log");
    assert_eq!(
        widened_peak_bytes,
        (raw_peak_bytes as f64 * 1.05).ceil() as u64,
        "the widened peak must be the raw peak times the 5% MLX stale-measured margin"
    );

    record_row(&serde_json::json!({
        "criterion": 3,
        "lane": "mlx:qwen_image",
        "engine": MEASURED_ENGINE,
        "tier": MEASURED_TIER,
        "artifactRevision": revision,
        "geometry": "1024x1024",
        "admitCapGb": admit_cap,
        "rung": format!("{:?}", evaluation.context.selection.strategy),
        "rawPeakBytes": raw_peak_bytes,
        "rawPeakGib": gib(raw_peak_bytes),
        "widenedPeakBytes": widened_peak_bytes,
        "widenedPeakGib": gib(widened_peak_bytes),
        "staleMargin": 0.05,
    }));
}

/// Read `key=<digits>` from the first log LINE containing `marker`.
///
/// Scoped to a line because the admission events and the selection event carry the same field
/// names for different rungs: a whole-buffer search would return whichever candidate happened to be
/// logged first, not the one that was selected.
fn field_in_line(log: &str, marker: &str, key: &str) -> Option<u64> {
    log.lines()
        .find(|line| line.contains(marker))
        .and_then(|line| field_after(line, key))
}

/// Read the integer value of the first `key=<digits>` occurrence in a captured tracing log.
fn field_after(log: &str, key: &str) -> Option<u64> {
    let start = log.find(key)? + key.len();
    let rest = &log[start..];
    let end = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

// ---------------------------------------------------------------------------------------------
// Criterion 4 — a genuinely oversized request refuses instead of entering MLX's fatal path.
// ---------------------------------------------------------------------------------------------

/// sc-18101 criterion 4. Under a cap below EVERY rung's widened floor, the gate must return the
/// actionable `InvalidPayload` refusal rather than admitting and letting MLX's default error
/// handler terminate the process.
///
/// The proof that this is a refusal and not an averted OOM is structural, not statistical: the
/// refusal is produced by `evaluate_request`, which runs BEFORE `generator.generate`, so no
/// allocation for the request is ever attempted — and the test then keeps running in the same
/// process, which an MLX allocator overshoot would not permit (the default handler calls
/// `exit(-1)`). The test asserts both: the message shape, and that the process is still alive and
/// able to admit the same request at a larger cap.
#[test]
#[ignore = "sc-18101 criterion 4: needs SceneWorks/krea-2-turbo-mlx bf16 cached"]
fn c4_oversized_request_is_refused_not_oom_killed() {
    let (repo_dir, engine, tier) = (c1_repo_dir(), c1_engine(), c1_tier());
    let Some((tier_dir, revision)) = cached_tier_dir(&repo_dir, &tier, C1_SENTINEL) else {
        panic!("SKIP-AS-FAILURE: no {repo_dir} {tier} weights cached");
    };
    let entry = shipped_manifest_entry(engine);
    // Loaded `Sequential` deliberately: the DEEPEST ladder the provider can offer. A refusal from a
    // ladder that already has every rung available is a refusal because nothing fits, not because a
    // rung was missing.
    let spec = production_spec(engine, &tier_dir, &tier)
        .with_offload_policy(gen_core::OffloadPolicy::Sequential);
    let plan = MlxRequestPlan::for_spec_and_manifest(engine, engine, &spec, Some(&entry), None);
    let generator = crate::inference_runtime::load(engine, &spec).expect("load the C1 subject");

    // A genuinely oversized request: a large geometry (whose area-scaled activation headroom alone
    // is substantial) against a cap no rung can fit.
    let request_inputs = inputs(2048, 2048);
    let refuse_cap: f64 = crate::smoke_support::env_or("SC18101_C4_CAP_GB", "6")
        .parse()
        .expect("SC18101_C4_CAP_GB");

    let (result, log) = crate::test_env::temp_env_var(
        "SCENEWORKS_MLX_MEMORY_CAP_GB",
        &format!("{refuse_cap}"),
        || {
            with_captured_tracing(|| {
                crate::mlx_fit_gate::evaluate_request(
                    &*generator,
                    &plan,
                    &request_inputs,
                    gen_core::MemoryCacheState::Warm,
                    gen_core::OffloadPolicy::Resident,
                    0,
                )
            })
        },
    );
    write_log("c4-refusal", &log);
    let error = result
        .err()
        .unwrap_or_else(|| panic!("criterion 4: a {refuse_cap} GB budget must refuse 2048x2048"))
        .to_string();
    eprintln!("[sc18101/c4] refusal: {error}");

    assert!(
        error.contains("2048x2048"),
        "criterion 4: the refusal must quote the request geometry: {error}"
    );
    assert!(
        error.contains("is safely available") || error.contains("no structurally admissible"),
        "criterion 4: the refusal must be one of the two actionable MLX refusals: {error}"
    );

    // Still alive, and not wedged: the same provider admits the same geometry at a cap that fits.
    // An OOM-killed process cannot reach this line.
    let (recovered, _) =
        crate::test_env::temp_env_var("SCENEWORKS_MLX_MEMORY_CAP_GB", "512", || {
            with_captured_tracing(|| {
                crate::mlx_fit_gate::evaluate_request(
                    &*generator,
                    &plan,
                    &request_inputs,
                    gen_core::MemoryCacheState::Warm,
                    gen_core::OffloadPolicy::Resident,
                    0,
                )
            })
        });
    assert!(
        recovered.is_ok(),
        "criterion 4: the refusal must be request-scoped — the same request admits at a larger cap"
    );

    record_row(&serde_json::json!({
        "criterion": 4,
        "engine": engine,
        "tier": tier,
        "artifactRevision": revision,
        "geometry": "2048x2048",
        "capGb": refuse_cap,
        "refusal": error,
        "processSurvived": true,
    }));
}

// ---------------------------------------------------------------------------------------------

/// The deferred-materialization mirror must name exactly the routes `image_jobs/base.rs` names.
///
/// `apply_measured_mlx_load_shape` is private and lives in an `include!`d, macOS-gated file, so the
/// coupling cannot be a direct call without rotating the memory-matrix source fingerprint (see
/// [`DEFERRED_MATERIALIZATION_ROUTES`]). It can still be a TEXT coupling: the engine ids are
/// single-line double-quoted literals in that source, which is exactly the shape the matrix's own
/// comment-stripping fingerprint guarantees stays load-bearing. A route added to or removed from
/// the production list therefore reds this test instead of silently changing what the ignored
/// scenarios exercise.
#[test]
fn load_shape_mirror_matches_the_documented_routes() {
    let base = include_str!("image_jobs/base.rs");
    let start = base
        .find("fn apply_measured_mlx_load_shape")
        .expect("image_jobs/base.rs declares apply_measured_mlx_load_shape");
    // Bound the slice to the function itself: the first line-start `}` after the signature. A
    // fixed-width window would run into the neighbouring test module, whose fixtures are full of
    // quoted component names.
    let end = base[start..]
        .find("\n}\n")
        .expect("the function has a line-start closing brace");
    let body = &base[start..start + end];
    for route in DEFERRED_MATERIALIZATION_ROUTES {
        assert!(
            body.contains(&format!("\"{route}\"")),
            "{route} is mirrored here but no longer named by apply_measured_mlx_load_shape"
        );
    }
    // And the converse: no OTHER engine id appears in that function. Counting quoted literals is
    // enough — the function names engine ids and nothing else.
    let quoted = body.match_indices('"').count();
    assert_eq!(
        quoted,
        DEFERRED_MATERIALIZATION_ROUTES.len() * 2,
        "apply_measured_mlx_load_shape names a route this harness does not mirror"
    );
}

/// The margin literals this file asserts on are the policy's. Not `#[ignore]`d and not portable to
/// the base commit: it is the coupling that stops the ignored tests above from silently drifting
/// into asserting a margin nobody ships.
#[test]
fn margin_literals_match_the_policy() {
    assert_eq!(
        crate::ladder_margin_policy::MLX_ESTIMATE_MARGIN,
        0.10,
        "c1 asserts `estimate_margin=0.1` in the tracing output"
    );
    assert_eq!(
        crate::ladder_margin_policy::MLX_STALE_MEASURED_MARGIN,
        0.05,
        "c3 asserts `stale_margin=0.05` and recomputes the widened peak with 1.05"
    );
}
