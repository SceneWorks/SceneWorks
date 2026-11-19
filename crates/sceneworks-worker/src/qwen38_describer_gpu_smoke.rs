//! Real-weight CUDA validation of the Qwen3.8 27B dataset describer (SW-7120, feature SW-F365),
//! and its MLX twin `qwen38_describer_describes_a_synthetic_scene_on_mlx` (SW-7202) at the end of
//! this file, which `qwen38-describer-mlx-validation.yml` runs on nax-macos-2.
//!
//! Runs ONLY from the dispatch-only `qwen38-describer-validation.yml` workflow (backend=cuda), under the exact-UUID
//! CUDA guard (`scripts/ci/windows_cuda_execution.py`); it is never part of ordinary PR or push CI.
//! It resolves a Candle tier (pre-quantised q4/q8 from the re-host, or full-precision bf16 from the
//! upstream repository) through the production path (`qwen38_describer_weights_dir`), checks it is
//! the catalog's pinned snapshot for that tier, describes one synthetic scene generated here with the production
//! describer (`VisionCaptionerDescriber`, the same one a `dataset_batch_describe` job runs), and
//! requires the described fields to be filled (`pose` and `camera` may be empty: the scene has no
//! people and no camera character) from a complete reply: a reply cut off at the reply budget fails
//! the run instead of being recovered. A sampler thread polls `nvidia-smi memory.used`
//! on the job's card every 100 ms for the peak; the guard holds that card exclusively and verified
//! it idle first, so the card-wide high-water is this run's peak (a transient shorter than the poll
//! interval can be missed).
//!
//! Fail-loud: the tier, the cache and the pinned snapshot must all be present. Nothing is downloaded
//! (`qwen38-describer-provision.yml` fills the cache).
//!
//! The scene is 512x384 by default; `SCENEWORKS_QWEN38_DESCRIBER_IMAGE_SIDE=1536` draws the same
//! scene at 1536x1536, so the peak of a large image can be measured.
//!
//! SW-7204: the describer names the installed block draft (DFlash2 / DSpine) as a job does, and the
//! evidence records the decode block (proposer, draft, precision, target passes, rows checked and
//! accepted tokens per pass). `SCENEWORKS_QWEN38_BLOCK_DRAFT=on` requires the draft to run (block
//! draft, the tier's draft precision, tree checking, more than one accepted token per pass, no
//! fallback); `no_fit` caps the engine's load budget so the draft cannot fit beside the target and
//! requires the describe to complete without it, its record saying why. The reply must be valid
//! describer JSON either way (the strict parse below).
//!
//! ```text
//! $env:HF_HUB_CACHE = "E:\huggingface\hub"
//! $env:SCENEWORKS_QWEN38_DESCRIBER_TIER = "q4"
//! $env:SCENEWORKS_QWEN38_DESCRIBER_IMAGE_SIDE = "default"
//! cargo test -p sceneworks-worker --features backend-candle --release --lib qwen38_describer_gpu_smoke -- --ignored --nocapture --test-threads=1
//! ```

use std::path::PathBuf;
#[cfg(not(target_os = "macos"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(not(target_os = "macos"))]
use std::sync::Arc;

use gen_core::core_llm::CancelFlag;
use sceneworks_core::dataset_batch_describe::DESCRIBE_FIELDS;

use super::*;
use crate::batch_describe_jobs::{
    qwen38_describer_weights_dir, BatchDescriber, DescribeModel, DescribePrompt,
    VisionCaptionerDescriber,
};
use crate::qwen38_block_draft::SmokeDraftCheck;
use sceneworks_core::dataset_batch_describe::QWEN38_27B_DESCRIBER_MODEL_ID;

/// The block draft the describe names, as a job resolves it, and (`no_fit`) the load-budget cap
/// that keeps it from fitting, held for the describe.
async fn smoke_block_draft(
    check: SmokeDraftCheck,
    settings: &Settings,
    tier: &str,
    weights_dir: &std::path::Path,
) -> (
    Option<crate::qwen38_block_draft::BlockDraftChoice>,
    Option<crate::qwen38_block_draft::EnvOverride>,
) {
    let draft = check.installed_draft(
        &settings.data_dir,
        QWEN38_27B_DESCRIBER_MODEL_ID,
        Some(tier),
    );
    let cap = match (check, &draft) {
        (SmokeDraftCheck::NoFit, Some(draft)) => {
            Some(crate::qwen38_block_draft::force_draft_no_fit(weights_dir, draft).await)
        }
        _ => None,
    };
    (draft, cap)
}

/// The describer's decode block, checked against `check`; panics with the record when it fails.
fn checked_decode_record(check: SmokeDraftCheck, describer: &VisionCaptionerDescriber) -> Value {
    let decode = describer
        .decode_record()
        .expect("the Qwen3.8 27B describer records its decode");
    eprintln!("DECODE block_draft_check={} {decode}", check.label());
    check
        .check(&decode, describer.block_draft.as_ref())
        .unwrap_or_else(|error| panic!("block-draft check {}: {error}", check.label()));
    decode
}

const TIER_ENV: &str = "SCENEWORKS_QWEN38_DESCRIBER_TIER";
const IMAGE_SIDE_ENV: &str = "SCENEWORKS_QWEN38_DESCRIBER_IMAGE_SIDE";
const CATALOG_ID: &str = "vision_describe_qwen3_8_27b";

/// The catalog's pinned snapshot of `tier` on this host's lane: the folder the re-host row names for
/// q4 and q8 (MLX re-host on macOS, Candle re-host elsewhere), the upstream repository's root for
/// full precision (`bf16`).
fn pinned_snapshot(tier: &str) -> crate::batch_describe_jobs::Qwen38PinnedSnapshot {
    use sceneworks_core::dataset_batch_describe::{
        QWEN38_27B_DESCRIBER_CANDLE_REPO, QWEN38_27B_DESCRIBER_MLX_REPO,
    };
    let rehost = if cfg!(target_os = "macos") {
        QWEN38_27B_DESCRIBER_MLX_REPO
    } else {
        QWEN38_27B_DESCRIBER_CANDLE_REPO
    };
    let entry = crate::test_env::shipped_catalog_entry(CATALOG_ID);
    crate::batch_describe_jobs::qwen38_pinned_snapshot(&entry, tier, rehost)
        .unwrap_or_else(|| panic!("the catalog has no pinned {rehost} {tier} row for {CATALOG_ID}"))
}

/// `$SCENEWORKS_QWEN38_DESCRIBER_TIER`, checked to be one of the catalog's tiers.
fn requested_tier() -> String {
    let tier =
        std::env::var(TIER_ENV).unwrap_or_else(|_| panic!("set ${TIER_ENV} to q4, q8 or bf16"));
    let tier = tier.trim().to_owned();
    assert!(
        matches!(tier.as_str(), "q4" | "q8" | "bf16"),
        "${TIER_ENV} must be q4, q8 or bf16, got {tier:?}"
    );
    tier
}

/// The scene size `$SCENEWORKS_QWEN38_DESCRIBER_IMAGE_SIDE` selects: unset, blank or `default` is
/// 512x384; `1536` is 1536x1536. Anything else fails the run.
fn scene_size() -> (u32, u32) {
    let side = std::env::var(IMAGE_SIDE_ENV).unwrap_or_default();
    match side.trim() {
        "" | "default" => (512, 384),
        "1536" => (1536, 1536),
        other => panic!("${IMAGE_SIDE_ENV} must be default or 1536, got {other:?}"),
    }
}

/// A simple synthetic scene: a red house with a brown roof and a door on green grass, under a blue
/// sky with a yellow sun. Generated, so the run needs no checked-in or personal image. The scene is
/// laid out on a 512x384 plan and stretched to `width`x`height`; at 512x384 every pixel is the plan's.
fn synthetic_scene(width: u32, height: u32) -> image::DynamicImage {
    let (scale_x, scale_y) = (512.0 / width as f32, 384.0 / height as f32);
    let scene = image::RgbImage::from_fn(width, height, |x, y| {
        // This pixel's position on the 512x384 plan.
        let (xf, yf) = (x as f32 * scale_x, y as f32 * scale_y);
        let sun = (xf - 410.0).powi(2) + (yf - 70.0).powi(2) <= 38.0_f32.powi(2);
        let wall = (120.0..=250.0).contains(&xf) && (190.0..=290.0).contains(&yf);
        let door = (170.0..=200.0).contains(&xf) && (235.0..=290.0).contains(&yf);
        // Roof: a triangle with its apex at (185, 120) over the wall's top edge.
        let roof =
            (120.0..190.0).contains(&yf) && (xf - 185.0).abs() <= (yf - 120.0) * (75.0 / 70.0);
        if door {
            image::Rgb([90, 50, 25])
        } else if wall {
            image::Rgb([190, 35, 30])
        } else if roof {
            image::Rgb([110, 70, 40])
        } else if sun {
            image::Rgb([250, 215, 60])
        } else if yf >= 270.0 {
            image::Rgb([60, 150 - ((yf - 270.0) / 4.0) as u8, 50])
        } else {
            let t = yf / 270.0;
            image::Rgb([(110.0 + 80.0 * t) as u8, (160.0 + 60.0 * t) as u8, 235])
        }
    });
    image::DynamicImage::ImageRgb8(scene)
}

#[cfg(not(target_os = "macos"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight describer GPU validation; needs a pinned Candle tier of the Qwen3.8 27B describer under HF_HUB_CACHE and a CUDA device"]
async fn qwen38_describer_describes_a_synthetic_scene_on_cuda() {
    let tier = requested_tier();
    let hf_hub_cache = PathBuf::from(
        std::env::var("HF_HUB_CACHE").expect("set $HF_HUB_CACHE to the real-weights hub"),
    );
    assert!(
        hf_hub_cache.is_dir(),
        "HF_HUB_CACHE is not a directory: {}",
        hf_hub_cache.display()
    );

    // The production resolution, then proof it landed on the catalog's pinned snapshot rather than
    // another cached revision.
    let temp = tempfile::tempdir().expect("smoke tempdir creates");
    let settings = Settings::for_test(temp.path().join("data"));
    let weights_dir = qwen38_describer_weights_dir(&settings, &tier)
        .unwrap_or_else(|error| panic!("the {tier} tier resolves from HF_HUB_CACHE: {error}"));
    let pinned = pinned_snapshot(&tier);
    let pinned_suffix = &pinned.relative_dir;
    assert!(
        weights_dir.ends_with(pinned_suffix),
        "the {tier} tier resolved to {} instead of the pinned snapshot {}",
        weights_dir.display(),
        pinned_suffix.display()
    );
    let draft_check = SmokeDraftCheck::from_env();
    let (draft, _no_fit_cap) = smoke_block_draft(draft_check, &settings, &tier, &weights_dir).await;

    // The worker reads the job's card by its nvidia-smi (PCI-bus-ordered) index; under
    // `CUDA_VISIBLE_DEVICES` that is the visible device's index.
    let gpu_index = std::env::var("CUDA_VISIBLE_DEVICES")
        .ok()
        .and_then(|devices| devices.split(',').next().map(|d| d.trim().to_owned()))
        .filter(|device| !device.is_empty())
        .unwrap_or_else(|| "0".to_owned());
    let baseline_mib = crate::iris_gpu_smoke::nvidia_memory_used_mib(&gpu_index)
        .expect("nvidia-smi memory.used reads");
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let stop = Arc::clone(&stop);
        let index = gpu_index.clone();
        std::thread::spawn(move || {
            let mut peak = 0_u64;
            while !stop.load(Ordering::Relaxed) {
                if let Some(used) = crate::iris_gpu_smoke::nvidia_memory_used_mib(&index) {
                    peak = peak.max(used);
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            peak
        })
    };

    let (scene_width, scene_height) = scene_size();
    let scene = synthetic_scene(scene_width, scene_height);
    let describer = VisionCaptionerDescriber::new(
        DescribeModel::Qwen38,
        weights_dir.clone(),
        "sw-7120-qwen38-describer-smoke".to_owned(),
    )
    .with_block_draft(draft);
    let (prompt, _) = DescribePrompt::for_instruction(DescribeModel::Qwen38, None);
    let started = std::time::Instant::now();
    let reply = describer.describe(scene, &prompt, CancelFlag::new()).await;
    let wall_seconds = started.elapsed().as_secs_f64();
    stop.store(true, Ordering::Relaxed);
    let peak_mib = sampler.join().expect("sampler thread joins");
    eprintln!(
        "RESULT model=qwen38_27b backend=candle tier={tier} image={scene_width}x{scene_height} \
         gpu={gpu_index} baseline_mib={baseline_mib} \
         peak_mib={peak_mib} peak_delta_mib={} wall_s={wall_seconds:.1}",
        peak_mib.saturating_sub(baseline_mib)
    );
    let raw = reply
        .expect("the describer model loads")
        .expect("the describer generates");
    eprintln!("raw reply:\n{raw}");
    assert!(!raw.contains("<think>"), "thinking must be off: {raw}");
    // The strict parse, never `recover`: a reply cut off at the reply budget fails here rather than
    // passing as a partial description with a notice.
    let parsed = describer
        .parse(&raw)
        .expect("the reply is the complete description JSON");
    assert!(
        parsed.notice.is_none(),
        "a complete reply carries no cut-off notice: {:?}",
        parsed.notice
    );
    let empty: Vec<&str> = DESCRIBE_FIELDS
        .iter()
        .zip(&parsed.fields)
        // The output contract lets `camera` be empty when the image shows no camera character, and
        // `pose` when it shows no people (the synthetic scene has none).
        .filter(|(name, text)| !matches!(**name, "camera" | "pose") && text.is_empty())
        .map(|(name, _)| *name)
        .collect();
    assert!(empty.is_empty(), "described fields are empty: {empty:?}");
    let decode = checked_decode_record(draft_check, &describer);

    let test_fqn = std::thread::current()
        .name()
        .expect("the libtest case has an actual FQN")
        .to_owned();
    if let Some(output_dir) = std::env::var_os("MODEL_PROOF_OUTPUT_DIR") {
        let output_dir = PathBuf::from(output_dir);
        assert!(output_dir.is_absolute() && output_dir.is_dir());
        let fields: serde_json::Map<String, Value> = DESCRIBE_FIELDS
            .iter()
            .zip(&parsed.fields)
            .map(|(name, text)| ((*name).to_owned(), json!(text)))
            .collect();
        let result = json!({
            "schema_version": 1,
            "test_fqn": test_fqn,
            "repo": pinned.repo,
            "revision": pinned.revision,
            "tier": tier,
            "weights_dir": weights_dir,
            "image": { "kind": "synthetic-scene", "width": scene_width, "height": scene_height },
            "fields": fields,
            "block_draft": { "check": draft_check.label(), "decode": decode },
            "wall_seconds": wall_seconds,
            "wall_time_scope": "one describe call on a cold process: model load plus one image",
            "gpu_memory": {
                "gpu_index": gpu_index,
                "baseline_mib": baseline_mib,
                "peak_mib": peak_mib,
                "peak_delta_mib": peak_mib.saturating_sub(baseline_mib),
                "method": "nvidia-smi memory.used on the job's card, polled every 100 ms; card-wide high-water"
            }
        });
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output_dir.join("describer-result.json"))
            .expect("create the owned describer result, never replace an existing file");
        serde_json::to_writer_pretty(&mut file, &result).expect("persist the describer result");
    }
    // The CUDA guard samples the card as soon as this process exits and refuses a card above 5%
    // utilisation. `nvidia-smi` averages utilisation over the last second, so exiting straight after
    // generation reads as a busy card (q4 and q8 runs, 2026-11-14). Stay idle until the average decays.
    std::thread::sleep(std::time::Duration::from_secs(5));
    // A missing fixture, a failed load or generation, or an empty field cannot emit this.
    eprintln!("MODEL_INFERENCE_BODY_COMPLETED {test_fqn}");
}

/// Directory of real images the throughput benchmark describes (the first
/// `$SCENEWORKS_QWEN38_BENCH_IMAGES_COUNT`, default 4, in name order). Never checked in.
#[cfg(not(target_os = "macos"))]
const BENCH_IMAGES_ENV: &str = "SCENEWORKS_QWEN38_BENCH_IMAGES";
#[cfg(not(target_os = "macos"))]
const BENCH_COUNT_ENV: &str = "SCENEWORKS_QWEN38_BENCH_IMAGES_COUNT";

/// Throughput benchmark of the describe path on one tier: real images described one after another
/// on a resident model, as a batch job does. Each generation's `describe_generation_stats` event
/// (prompt and generated tokens, prefill and decode seconds, sampler, CUDA-graph replays) prints on
/// stdout; this adds the per-image wall time and the raw reply so the output is read, not just timed.
///
/// ```text
/// $env:HF_HUB_CACHE = "E:\huggingface\hub"
/// $env:SCENEWORKS_QWEN38_DESCRIBER_TIER = "q4"
/// $env:SCENEWORKS_QWEN38_BENCH_IMAGES = "<a directory of images>"
/// cargo test -p sceneworks-worker --features backend-candle --release --lib qwen38_describer_throughput -- --ignored --nocapture --test-threads=1
/// ```
#[cfg(not(target_os = "macos"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight describer GPU benchmark; needs a pinned Candle tier under HF_HUB_CACHE, a CUDA device and a directory of images"]
async fn qwen38_describer_throughput_on_cuda() {
    // The worker's stdout event sink, so each `describe_generation_stats` line prints.
    sceneworks_core::observability::init_logging();
    let tier = std::env::var(TIER_ENV).unwrap_or_else(|_| panic!("set ${TIER_ENV}"));
    let tier = tier.trim().to_owned();
    let images_dir = PathBuf::from(
        std::env::var(BENCH_IMAGES_ENV).unwrap_or_else(|_| panic!("set ${BENCH_IMAGES_ENV}")),
    );
    let count: usize = std::env::var(BENCH_COUNT_ENV)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(4);
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&images_dir)
        .expect("the bench image directory reads")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| {
                    matches!(
                        ext.to_ascii_lowercase().as_str(),
                        "png" | "jpg" | "jpeg" | "webp"
                    )
                })
        })
        .collect();
    paths.sort();
    paths.truncate(count);
    assert!(!paths.is_empty(), "no images in {}", images_dir.display());

    let temp = tempfile::tempdir().expect("bench tempdir creates");
    let settings = Settings::for_test(temp.path().join("data"));
    let weights_dir = qwen38_describer_weights_dir(&settings, &tier)
        .unwrap_or_else(|error| panic!("the {tier} tier resolves from HF_HUB_CACHE: {error}"));
    let (prompt, _) = DescribePrompt::for_instruction(DescribeModel::Qwen38, None);
    // As a job does: the installed block draft rides the load (SW-7204).
    let draft = crate::qwen38_block_draft::installed_block_draft(
        &settings.data_dir,
        QWEN38_27B_DESCRIBER_MODEL_ID,
        Some(&tier),
    );
    let describer = VisionCaptionerDescriber::new(
        DescribeModel::Qwen38,
        weights_dir,
        "qwen38-describer-bench".to_owned(),
    )
    .with_block_draft(draft);

    // Warm-up: the load and first-call graph capture are paid here, not by a measured image.
    let started = std::time::Instant::now();
    describer
        .describe(synthetic_scene(512, 384), &prompt, CancelFlag::new())
        .await
        .expect("the describer loads")
        .expect("the warm-up generates");
    eprintln!(
        "BENCH warmup tier={tier} wall_s={:.1}",
        started.elapsed().as_secs_f64()
    );

    let mut total = 0.0;
    for path in &paths {
        let image = crate::image_decode::decode_image_any(path).expect("the image decodes");
        let started = std::time::Instant::now();
        let raw = describer
            .describe(image, &prompt, CancelFlag::new())
            .await
            .expect("the describer is loaded")
            .expect("the describer generates");
        let wall = started.elapsed().as_secs_f64();
        total += wall;
        let parses = describer.parse(&raw).is_ok();
        eprintln!(
            "BENCH tier={tier} image={} wall_s={wall:.1} chars={} parses={parses}",
            path.file_name().unwrap().to_string_lossy(),
            raw.len()
        );
        eprintln!("BENCH_REPLY {raw}");
    }
    eprintln!(
        "BENCH_TOTAL tier={tier} images={} wall_s={total:.1} per_image_s={:.1}",
        paths.len(),
        total / paths.len() as f64
    );
}

/// Decode-path breakdown on one tier: the same text-only prompt decoded greedy without a
/// constraint (on-device sampling), with the describer's temperature and no constraint, and with
/// the describer's temperature under the JSON constraint. Comparing the three
/// `describe_generation_stats` lines separates the model step from sampling and constraint cost.
#[cfg(not(target_os = "macos"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight decode-path benchmark; needs a pinned Candle tier under HF_HUB_CACHE and a CUDA device"]
async fn qwen38_decode_paths_on_cuda() {
    use gen_core::core_llm::{Constraint, Message, Sampling, TextLlmRequest, ThinkingMode};
    sceneworks_core::observability::init_logging();
    let tier = std::env::var(TIER_ENV).unwrap_or_else(|_| panic!("set ${TIER_ENV}"));
    let tier = tier.trim().to_owned();
    let temp = tempfile::tempdir().expect("bench tempdir creates");
    let settings = Settings::for_test(temp.path().join("data"));
    let weights_dir = qwen38_describer_weights_dir(&settings, &tier)
        .unwrap_or_else(|error| panic!("the {tier} tier resolves from HF_HUB_CACHE: {error}"));
    let user = "Describe, as a JSON object with the keys \"subject\", \"setting\", \"lighting\", \
                \"mood\" and \"details\", an old lighthouse on a rocky coast at dusk during a storm. \
                Each value is a long paragraph of at least eighty words.";
    use gen_core::core_llm::{Speculative, SpeculativeProposer};
    let mtp = Some(Speculative::proposer(SpeculativeProposer::Mtp, 3));
    let cases: [(&str, f32, Option<Constraint>, Option<Speculative>); 7] = [
        ("warmup", 0.0, None, None),
        ("greedy_free", 0.0, None, None),
        ("t0.4_free", 0.4, None, None),
        ("t0.4_json", 0.4, Some(Constraint::Json), None),
        ("greedy_free_mtp", 0.0, None, mtp),
        ("t0.4_free_mtp", 0.4, None, mtp),
        ("t0.4_json_mtp", 0.4, Some(Constraint::Json), mtp),
    ];
    for (label, temperature, constraint, speculative) in cases {
        let (spec, requirements) =
            crate::batch_describe_jobs::vision_describer_load(&weights_dir, None);
        let started = std::time::Instant::now();
        let output = crate::refine_model_cache::with_cached_refiner(
            spec,
            requirements,
            "bench load failed",
            move |model| {
                let request = TextLlmRequest {
                    messages: vec![Message::user(user.to_owned())],
                    sampling: Sampling {
                        temperature,
                        top_p: if temperature > 0.0 { 0.9 } else { 1.0 },
                        ..Sampling::default()
                    },
                    max_new_tokens: 400,
                    constraint,
                    cancel: CancelFlag::new(),
                    thinking: ThinkingMode::Disabled,
                    speculative,
                    ..Default::default()
                };
                Ok(model.generate(&request, &mut |_| {}))
            },
        )
        .await
        .expect("the model loads")
        .expect("the model generates");
        let decode = output
            .timings
            .as_ref()
            .map(|t| t.decode.as_secs_f64())
            .unwrap_or(0.0);
        let report = output.decode.as_ref();
        eprintln!(
            "DECODE case={label} tier={tier} wall_s={:.1} generated={} decode_s={decode:.2} \
             tok_s={:.1} sampler={:?} graphs_replayed={:?} graphs_eager={:?} path={:?} attention={:?} fused={:?} graph_path={:?} proposer={:?} proposed={:?} accepted={:?} target_forwards={:?} demoted_at={:?} monitor={:?} fallbacks={:?}",
            started.elapsed().as_secs_f64(),
            output.usage.generated_tokens,
            f64::from(output.usage.generated_tokens) / decode.max(1e-9),
            report.map(|r| r.sampler.clone()),
            report.map(|r| r.cuda_graphs.replayed),
            report.map(|r| r.cuda_graphs.eager),
            report.map(|r| r.path.clone()),
            report.map(|r| r.attention.clone()),
            report.map(|r| r.fused_primitives.clone()),
            report.map(|r| r.graph_path.clone()),
            report.map(|r| r.proposer),
            report.map(|r| r.proposed_tokens),
            report.map(|r| r.accepted_tokens),
            report.map(|r| r.target_forwards),
            report.and_then(|r| r.speculative_demoted_at),
            report.and_then(|r| r.speculative_monitor),
            report.map(|r| r.fallbacks.clone()),
        );
    }
}

/// The MLX twin of the CUDA smoke (SW-7202): the same production describer, tier resolution, pinned
/// snapshot check, synthetic scene and strict parse, on macOS. It does not sample memory itself:
/// `qwen38-describer-mlx-validation.yml` runs it on nax-macos-2 under the external phys_footprint
/// guard (`scripts/ci/iris-mlx-real-weights/tree_guard.py`), whose `root_lifetime_max` is the
/// kernel's exact high-water mark of this process (`ri_lifetime_max_phys_footprint`, Metal buffers
/// included): the peak the MLX floor is set from.
///
/// ```text
/// HF_HUB_CACHE=<hub> SCENEWORKS_QWEN38_DESCRIBER_TIER=q4 SCENEWORKS_QWEN38_DESCRIBER_IMAGE_SIDE=1536 \
///   cargo test -p sceneworks-worker --release --lib qwen38_describer_gpu_smoke -- --ignored --nocapture --test-threads=1
/// ```
#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight describer MLX validation; needs a pinned MLX tier of the Qwen3.8 27B describer under HF_HUB_CACHE"]
async fn qwen38_describer_describes_a_synthetic_scene_on_mlx() {
    let tier = requested_tier();
    let hf_hub_cache = PathBuf::from(
        std::env::var("HF_HUB_CACHE").expect("set $HF_HUB_CACHE to the real-weights hub"),
    );
    assert!(
        hf_hub_cache.is_dir(),
        "HF_HUB_CACHE is not a directory: {}",
        hf_hub_cache.display()
    );
    let temp = tempfile::tempdir().expect("smoke tempdir creates");
    let settings = Settings::for_test(temp.path().join("data"));
    let weights_dir = qwen38_describer_weights_dir(&settings, &tier)
        .unwrap_or_else(|error| panic!("the {tier} tier resolves from HF_HUB_CACHE: {error}"));
    let pinned = pinned_snapshot(&tier);
    assert!(
        weights_dir.ends_with(&pinned.relative_dir),
        "the {tier} tier resolved to {} instead of the pinned snapshot {}",
        weights_dir.display(),
        pinned.relative_dir.display()
    );
    let draft_check = SmokeDraftCheck::from_env();
    let (draft, _no_fit_cap) = smoke_block_draft(draft_check, &settings, &tier, &weights_dir).await;

    let (scene_width, scene_height) = scene_size();
    let describer = VisionCaptionerDescriber::new(
        DescribeModel::Qwen38,
        weights_dir.clone(),
        "sw-7202-qwen38-describer-mlx-smoke".to_owned(),
    )
    .with_block_draft(draft);
    let (prompt, _) = DescribePrompt::for_instruction(DescribeModel::Qwen38, None);
    let started = std::time::Instant::now();
    let reply = describer
        .describe(
            synthetic_scene(scene_width, scene_height),
            &prompt,
            CancelFlag::new(),
        )
        .await;
    let wall_seconds = started.elapsed().as_secs_f64();
    eprintln!(
        "RESULT model=qwen38_27b backend=mlx tier={tier} image={scene_width}x{scene_height} \
         wall_s={wall_seconds:.1}"
    );
    let raw = reply
        .expect("the describer model loads")
        .expect("the describer generates");
    eprintln!("raw reply:\n{raw}");
    assert!(!raw.contains("<think>"), "thinking must be off: {raw}");
    let parsed = describer
        .parse(&raw)
        .expect("the reply is the complete description JSON");
    assert!(
        parsed.notice.is_none(),
        "a complete reply carries no cut-off notice: {:?}",
        parsed.notice
    );
    let empty: Vec<&str> = DESCRIBE_FIELDS
        .iter()
        .zip(&parsed.fields)
        .filter(|(name, text)| !matches!(**name, "camera" | "pose") && text.is_empty())
        .map(|(name, _)| *name)
        .collect();
    assert!(empty.is_empty(), "described fields are empty: {empty:?}");
    let decode = checked_decode_record(draft_check, &describer);
    let test_fqn = std::thread::current()
        .name()
        .expect("the libtest case has an actual FQN")
        .to_owned();
    if let Some(output_dir) = std::env::var_os("MODEL_PROOF_OUTPUT_DIR") {
        let output_dir = PathBuf::from(output_dir);
        assert!(output_dir.is_absolute() && output_dir.is_dir());
        let fields: serde_json::Map<String, Value> = DESCRIBE_FIELDS
            .iter()
            .zip(&parsed.fields)
            .map(|(name, text)| ((*name).to_owned(), json!(text)))
            .collect();
        let result = json!({
            "schema_version": 1,
            "test_fqn": test_fqn,
            "backend": "mlx",
            "repo": pinned.repo,
            "revision": pinned.revision,
            "tier": tier,
            "weights_dir": weights_dir,
            "image": { "kind": "synthetic-scene", "width": scene_width, "height": scene_height },
            "fields": fields,
            "block_draft": { "check": draft_check.label(), "decode": decode },
            "wall_seconds": wall_seconds,
            "wall_time_scope": "one describe call on a cold process: model load plus one image",
            "memory": "measured outside the process by tree_guard.py (root_lifetime_max)"
        });
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output_dir.join("describer-result.json"))
            .expect("create the owned describer result, never replace an existing file");
        serde_json::to_writer_pretty(&mut file, &result).expect("persist the describer result");
    }
    eprintln!("MODEL_INFERENCE_BODY_COMPLETED {test_fqn}");
}
