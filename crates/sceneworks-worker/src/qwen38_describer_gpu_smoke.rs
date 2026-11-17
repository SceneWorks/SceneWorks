//! Real-weight CUDA validation of the Qwen3.8 27B dataset describer (SW-7120, feature SW-F365).
//!
//! Runs ONLY from the dispatch-only `qwen38-describer-validation.yml` workflow, under the exact-UUID
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
//! ```text
//! $env:HF_HUB_CACHE = "E:\huggingface\hub"
//! $env:SCENEWORKS_QWEN38_DESCRIBER_TIER = "q4"
//! $env:SCENEWORKS_QWEN38_DESCRIBER_IMAGE_SIDE = "default"
//! cargo test -p sceneworks-worker --features backend-candle --release --lib qwen38_describer_gpu_smoke -- --ignored --nocapture --test-threads=1
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use gen_core::core_llm::CancelFlag;
use sceneworks_core::dataset_batch_describe::DESCRIBE_FIELDS;

use super::*;
use crate::batch_describe_jobs::{
    qwen38_describer_weights_dir, BatchDescriber, DescribeModel, DescribePrompt,
    VisionCaptionerDescriber,
};

const TIER_ENV: &str = "SCENEWORKS_QWEN38_DESCRIBER_TIER";
const IMAGE_SIDE_ENV: &str = "SCENEWORKS_QWEN38_DESCRIBER_IMAGE_SIDE";
const CATALOG_ID: &str = "vision_describe_qwen3_8_27b";

/// The catalog's pinned snapshot of `tier` on the Candle lane: the re-host's `<tier>/` folder for
/// q4 and q8, the upstream repository's root for full precision (`bf16`).
fn pinned_candle_snapshot(tier: &str) -> crate::batch_describe_jobs::Qwen38PinnedSnapshot {
    use sceneworks_core::dataset_batch_describe::QWEN38_27B_DESCRIBER_CANDLE_REPO;
    let entry = crate::test_env::shipped_catalog_entry(CATALOG_ID);
    crate::batch_describe_jobs::qwen38_pinned_snapshot(
        &entry,
        tier,
        QWEN38_27B_DESCRIBER_CANDLE_REPO,
    )
    .unwrap_or_else(|| panic!("the catalog has no pinned Candle {tier} row for {CATALOG_ID}"))
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight describer GPU validation; needs a pinned Candle tier of the Qwen3.8 27B describer under HF_HUB_CACHE and a CUDA device"]
async fn qwen38_describer_describes_a_synthetic_scene_on_cuda() {
    let tier =
        std::env::var(TIER_ENV).unwrap_or_else(|_| panic!("set ${TIER_ENV} to q4, q8 or bf16"));
    let tier = tier.trim().to_owned();
    assert!(
        matches!(tier.as_str(), "q4" | "q8" | "bf16"),
        "${TIER_ENV} must be q4, q8 or bf16, got {tier:?}"
    );
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
    let pinned = pinned_candle_snapshot(&tier);
    let pinned_suffix = &pinned.relative_dir;
    assert!(
        weights_dir.ends_with(pinned_suffix),
        "the {tier} tier resolved to {} instead of the pinned snapshot {}",
        weights_dir.display(),
        pinned_suffix.display()
    );

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
    let describer = VisionCaptionerDescriber {
        model: DescribeModel::Qwen38,
        weights_dir: weights_dir.clone(),
        job_id: "sw-7120-qwen38-describer-smoke".to_owned(),
    };
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
const BENCH_IMAGES_ENV: &str = "SCENEWORKS_QWEN38_BENCH_IMAGES";
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
    let describer = VisionCaptionerDescriber {
        model: DescribeModel::Qwen38,
        weights_dir,
        job_id: "qwen38-describer-bench".to_owned(),
    };

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
        let (spec, requirements) = crate::batch_describe_jobs::vision_describer_load(&weights_dir);
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
