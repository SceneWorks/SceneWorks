//! Real-weight CUDA validation of the Qwen3.8 27B dataset describer (SW-7120, feature SW-F365).
//!
//! Runs ONLY from the dispatch-only `qwen38-describer-validation.yml` workflow, under the exact-UUID
//! CUDA guard (`scripts/ci/windows_cuda_execution.py`); it is never part of ordinary PR or push CI.
//! It resolves a pre-quantised Candle tier through the production path
//! (`qwen38_describer_weights_dir`), describes one synthetic scene generated here with the production
//! describer (`VisionCaptionerDescriber`, the same one a `dataset_batch_describe` job runs), and
//! requires all six described fields to be filled. A sampler thread polls `nvidia-smi memory.used`
//! on the job's card every 100 ms for the peak; the guard holds that card exclusively and verified
//! it idle first, so the card-wide high-water is this run's peak (a transient shorter than the poll
//! interval can be missed).
//!
//! Fail-loud: the tier, the cache and the pinned snapshot must all be present. Nothing is downloaded
//! (`qwen38-describer-provision.yml` fills the cache).
//!
//! ```text
//! $env:HF_HUB_CACHE = "E:\huggingface\hub"
//! $env:SCENEWORKS_QWEN38_DESCRIBER_TIER = "q4"
//! cargo test -p sceneworks-worker --features backend-candle --release --lib qwen38_describer_gpu_smoke -- --ignored --nocapture --test-threads=1
//! ```

use std::path::{Path, PathBuf};
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
const CATALOG_ID: &str = "vision_describe_qwen3_8_27b";

/// The catalog's pinned revision of the Candle re-host for `tier`.
fn catalog_candle_revision(tier: &str) -> String {
    use sceneworks_core::dataset_batch_describe::QWEN38_27B_DESCRIBER_CANDLE_REPO;
    let entry = crate::test_env::shipped_catalog_entry(CATALOG_ID);
    entry
        .get("downloads")
        .and_then(Value::as_array)
        .and_then(|rows| {
            rows.iter().find(|row| {
                row.get("repo").and_then(Value::as_str) == Some(QWEN38_27B_DESCRIBER_CANDLE_REPO)
                    && row.get("variant").and_then(Value::as_str) == Some(tier)
            })
        })
        .and_then(|row| row.get("revision").and_then(Value::as_str))
        .unwrap_or_else(|| panic!("the catalog has no pinned Candle {tier} row for {CATALOG_ID}"))
        .to_owned()
}

/// A simple synthetic scene: a red house with a brown roof and a door on green grass, under a blue
/// sky with a yellow sun. Generated, so the run needs no checked-in or personal image.
fn synthetic_scene() -> image::DynamicImage {
    const WIDTH: u32 = 512;
    const HEIGHT: u32 = 384;
    let scene = image::RgbImage::from_fn(WIDTH, HEIGHT, |x, y| {
        let (xf, yf) = (x as f32, y as f32);
        let sun = (xf - 410.0).powi(2) + (yf - 70.0).powi(2) <= 38.0_f32.powi(2);
        let wall = (120..=250).contains(&x) && (190..=290).contains(&y);
        let door = (170..=200).contains(&x) && (235..=290).contains(&y);
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
        } else if y >= 270 {
            image::Rgb([60, 150 - ((y - 270) / 4) as u8, 50])
        } else {
            let t = y as f32 / 270.0;
            image::Rgb([(110.0 + 80.0 * t) as u8, (160.0 + 60.0 * t) as u8, 235])
        }
    });
    image::DynamicImage::ImageRgb8(scene)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight describer GPU validation; needs a pinned Candle tier of the Qwen3.8 27B describer under HF_HUB_CACHE and a CUDA device"]
async fn qwen38_describer_describes_a_synthetic_scene_on_cuda() {
    use sceneworks_core::dataset_batch_describe::QWEN38_27B_DESCRIBER_CANDLE_REPO;

    let tier = std::env::var(TIER_ENV).unwrap_or_else(|_| panic!("set ${TIER_ENV} to q4 or q8"));
    let tier = tier.trim().to_owned();
    assert!(
        tier == "q4" || tier == "q8",
        "${TIER_ENV} must be q4 or q8, got {tier:?}"
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
    let revision = catalog_candle_revision(&tier);
    let pinned_suffix = Path::new(&format!(
        "models--{}",
        QWEN38_27B_DESCRIBER_CANDLE_REPO.replace('/', "--")
    ))
    .join("snapshots")
    .join(&revision)
    .join(&tier);
    assert!(
        weights_dir.ends_with(&pinned_suffix),
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

    let scene = synthetic_scene();
    let (scene_width, scene_height) = (scene.width(), scene.height());
    let describer = VisionCaptionerDescriber {
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
        "RESULT model=qwen38_27b backend=candle tier={tier} gpu={gpu_index} baseline_mib={baseline_mib} \
         peak_mib={peak_mib} peak_delta_mib={} wall_s={wall_seconds:.1}",
        peak_mib.saturating_sub(baseline_mib)
    );
    let raw = reply
        .expect("the describer model loads")
        .expect("the describer generates");
    eprintln!("raw reply:\n{raw}");
    assert!(!raw.contains("<think>"), "thinking must be off: {raw}");
    let parsed = describer
        .parse(&raw)
        .expect("the reply is the six-field JSON");
    let empty: Vec<&str> = DESCRIBE_FIELDS
        .iter()
        .zip(&parsed.fields)
        // The output contract lets `camera` be empty when the image shows no camera character.
        .filter(|(name, text)| **name != "camera" && text.is_empty())
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
            "repo": QWEN38_27B_DESCRIBER_CANDLE_REPO,
            "revision": revision,
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
