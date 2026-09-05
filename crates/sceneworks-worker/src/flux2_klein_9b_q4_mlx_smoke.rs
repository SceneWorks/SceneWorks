//! Local real-weight MLX smoke for the FLUX.2 Klein 9B **q4** worker lane (sc-22765, epic 8506).
//! `#[ignore]`d — run by hand on an Apple-Silicon Mac.
//!
//! This is the worker-side proof for sc-22760: the MLX worker could not load either Klein rehost
//! from the raw HF cache tier path (`models--…/snapshots/<rev>/q4`). Two things had to move
//! together — the inference engine (symlink-aware discovery roots, sharded bf16 components) and the
//! rehost artifacts (the q4/q8 `text_encoder/config.json` carried a false `quantization` block over
//! a dense BF16 Qwen3). The pin bump lands both; this smoke drives the seams `generate_stream` uses,
//! minus the API/job plumbing, against the pinned cache snapshot:
//!   1. `huggingface_pinned_snapshot_dir` → the manifest's default-download revision in the cache,
//!   2. `resolve_weights_dir` → the tier subdir, for BOTH tier picks a real job produces: a plain
//!      default job (no `advanced.mlxQuantize`, which `standard_tier_subdir` answers with the
//!      app-wide `q8/` default of epic 10721 / sc-10726 — NOT the manifest's default DOWNLOAD row,
//!      which is q4) and an explicit q4 pick,
//!   3. `resolve_quant` → `(None, None)` on both (`mlx.denseTextEncoderTier` keeps the Qwen3 text
//!      encoder dense at every tier),
//!   4. `inference_runtime::load("flux2_klein_9b", &spec)` — the real engine load on the raw cache
//!      path (no re-host, no flattened copy),
//!   5. a short distilled render off the q4 tier, checked non-degenerate.
//!
//! Setup — with the q4 AND q8 tiers of `SceneWorks/flux2-klein-9b-mlx` at the manifest revision in
//! the HF cache, no env is needed (`HF_HUB_CACHE` / `HF_HOME` are honoured, else `~/.cache/huggingface`):
//! ```text
//! # optional: KLEIN_Q4_STEPS=4 KLEIN_Q4_W=768 KLEIN_Q4_H=768 KLEIN_Q4_PROMPT="..." KLEIN_Q4_OUT_DIR=/tmp/klein_q4_smoke
//! cargo test -p sceneworks-worker --release flux2_klein_9b_q4_mlx_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use gen_core::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use sceneworks_core::image_request::ImageRequest;
use serde_json::{json, Value};

use super::smoke_support::{
    env_or, image_mean, image_std, is_all_zero, save_png, DEGENERATE_STD_FLOOR_TIGHT,
};

const MODEL_ID: &str = "flux2_klein_9b";

/// The hub cache root this smoke reads from: whatever the environment already points at, else the
/// standard `~/.cache/huggingface/hub` that `huggingface_hub` downloads into.
fn hub_cache_root() -> Option<PathBuf> {
    for key in ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"] {
        if let Ok(value) = std::env::var(key) {
            if !value.trim().is_empty() {
                return Some(PathBuf::from(value));
            }
        }
    }
    if let Ok(value) = std::env::var("HF_HOME") {
        if !value.trim().is_empty() {
            return Some(PathBuf::from(value).join("hub"));
        }
    }
    Some(PathBuf::from(std::env::var("HOME").ok()?).join(".cache/huggingface/hub"))
}

/// The shipped catalog entry for `model_id`, read straight from the embedded builtin manifest — the
/// same bytes the worker resolves against.
fn builtin_manifest_entry(model_id: &str) -> Value {
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc is embedded");
    let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
        .expect("builtin.models.jsonc parses");
    manifest["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|entry| entry.get("id").and_then(Value::as_str) == Some(model_id))
        .cloned()
        .unwrap_or_else(|| panic!("builtin entry {model_id} present"))
}

/// The entry's `"default": true` download row: `(repo, revision, variant)`. `variant` is the tier the
/// catalog DOWNLOADS by default; it is deliberately NOT the tier a default job resolves (see
/// `standard_tier_subdir`'s app-wide q8 default), and this smoke keeps the two apart.
fn default_download(entry: &Value) -> (String, String, String) {
    let row = entry["downloads"]
        .as_array()
        .expect("downloads array")
        .iter()
        .find(|row| row.get("default").and_then(Value::as_bool) == Some(true))
        .expect("one default download row");
    let field = |key: &str| {
        row[key]
            .as_str()
            .unwrap_or_else(|| panic!("default download `{key}` is a string"))
            .to_owned()
    };
    (field("repo"), field("revision"), field("variant"))
}

/// The exact artifact defect the sc-22760 pin bump corrects: the packed tiers' Qwen3 text encoder is
/// dense BF16, so its `config.json` must NOT declare a `quantization` block. gen-core refuses a
/// marker that contradicts the tensor surface, which is how the old revisions failed to load.
fn assert_text_encoder_config_is_dense(tier: &Path) {
    let config = tier.join("text_encoder/config.json");
    let raw = std::fs::read_to_string(&config)
        .unwrap_or_else(|error| panic!("read {}: {error}", config.display()));
    let parsed: Value = serde_json::from_str(&raw).expect("text_encoder/config.json parses");
    assert!(
        parsed.get("quantization").is_none() && parsed.get("quantization_config").is_none(),
        "{} declares a quantization block over the dense BF16 Qwen3 text encoder — the manifest is \
         pinned at a rehost revision that predates the sc-22760 config fix",
        config.display()
    );
}

#[test]
#[ignore = "real-weight MLX smoke; needs the SceneWorks/flux2-klein-9b-mlx q4 tier cached at the manifest revision + an Apple-Silicon Mac"]
fn flux2_klein_9b_q4_mlx_gpu_smoke() {
    let hub = hub_cache_root().expect("a resolvable HF hub cache root");
    // Pin the worker's cache resolution to the SAME hub this smoke discovered in, so the pinned
    // resolver reads the snapshot we are about to assert on rather than a different data_dir's cache.
    let _env = crate::test_env::EnvVars::set(&[("HF_HUB_CACHE", &hub.display().to_string())]);
    let settings = crate::Settings::from_env();

    let manifest = builtin_manifest_entry(MODEL_ID);
    let (repo, revision, variant) = default_download(&manifest);
    assert_eq!(
        variant, "q4",
        "the shipped default DOWNLOAD row is the q4 tier"
    );

    // 1. The worker's own pinned resolver (F-029): the exact manifest revision, never `refs/main`.
    let snapshot =
        crate::model_jobs::huggingface_pinned_snapshot_dir(&settings.data_dir, &repo, &revision)
            .unwrap_or_else(|| {
                panic!(
                    "{repo}@{revision} is not in the HF cache at {} — pull it with \
             `hf download {repo} --revision {revision} --include '{variant}/*'`",
                    hub.display()
                )
            });
    for expected_tier in ["q4", "q8"] {
        let dir = snapshot.join(expected_tier);
        assert!(
            dir.join("transformer/diffusion_pytorch_model.safetensors")
                .is_file(),
            "{} must carry the packed transformer — pull it with `hf download {repo} --revision \
             {revision} --include '{expected_tier}/*'`",
            dir.display()
        );
    }

    // 2-3. The router's decisions, for BOTH tier picks a real job produces. `resolve_weights_dir`
    // answers a plain default job with the app-wide `q8/` default (epic 10721 / sc-10726) — the
    // manifest's `"default": true` DOWNLOAD row is q4, which is a different question — and answers an
    // explicit `advanced.mlxQuantize: 4` with `q4/`. On both, the load quant is forced to None by
    // `mlx.denseTextEncoderTier`, so the dense bf16 Qwen3 text encoder is never re-quantized.
    let request_for = |advanced: Value| {
        let payload = json!({
            "model": MODEL_ID,
            "prompt": "smoke",
            "width": 1024,
            "height": 1024,
            "advanced": advanced,
            "modelManifestEntry": manifest,
        });
        ImageRequest::from_payload(payload.as_object().unwrap())
    };
    let default_job = request_for(json!({}));
    let q4_job = request_for(json!({ "mlxQuantize": 4, "mlxQuantizeExplicit": true }));

    let mut q4_loaded = None;
    for (label, request, expected_tier) in [
        ("default job", &default_job, "q8"),
        ("explicit q4 pick", &q4_job, "q4"),
    ] {
        let resolved = crate::image_jobs::resolve_weights_dir(request, &settings)
            .expect("resolve_weights_dir")
            .unwrap_or_else(|| panic!("{MODEL_ID} ({label}): the worker resolved no weights dir"));
        assert!(
            resolved.starts_with(&snapshot) && resolved.ends_with(expected_tier),
            "{MODEL_ID} ({label}): the worker must resolve {}/{expected_tier}, got {}",
            snapshot.display(),
            resolved.display()
        );
        // The exact artifact defect the pin bump corrects, asserted on the tier that is about to load.
        assert_text_encoder_config_is_dense(&resolved);
        assert_eq!(
            crate::image_jobs::resolve_quant(request, Some(&resolved)),
            (None, None),
            "{MODEL_ID} ({label}) is a dense-TE turnkey: the load quant must be None"
        );
        println!(
            "[worker-smoke] {label}: resolve_weights_dir -> {}",
            resolved.display()
        );

        // 4. The real engine load on the raw HF cache tier path (symlinks into blobs/ and all). This
        // is the load that failed before sc-22760: the engine's discovery roots refused the symlinked
        // cache entries, and the packed tiers' text_encoder/config.json declared a quantization block
        // over dense BF16 tensors.
        let spec = LoadSpec::new(WeightsSource::Dir(resolved.clone()));
        let generator = crate::inference_runtime::load(MODEL_ID, &spec).unwrap_or_else(|error| {
            panic!(
                "{MODEL_ID} refused the pinned {expected_tier} tier at {} — this is the sc-22760 \
                 failure: {error}",
                resolved.display()
            )
        });
        println!("[worker-smoke] {label}: loaded {expected_tier} through the worker lane");
        if expected_tier == "q4" {
            q4_loaded = Some((resolved, generator));
        }
    }

    // 5. A short distilled render off the q4 tier (manifest defaults: 4 steps, guidance 1.0).
    let (tier, generator) = q4_loaded.expect("the explicit q4 pick loaded");
    let steps: u32 = env_or("KLEIN_Q4_STEPS", "4")
        .parse()
        .expect("KLEIN_Q4_STEPS");
    let w: u32 = env_or("KLEIN_Q4_W", "768").parse().expect("KLEIN_Q4_W");
    let h: u32 = env_or("KLEIN_Q4_H", "768").parse().expect("KLEIN_Q4_H");
    let prompt = env_or(
        "KLEIN_Q4_PROMPT",
        "a red fox sitting in a sunlit autumn forest, sharp focus, shallow depth of field",
    );
    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        guidance: Some(1.0),
        ..Default::default()
    };
    println!(
        "[worker-smoke] rendering {w}x{h} @ {steps} steps from {} ...",
        tier.display()
    );
    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |p| {
            let s = format!("{p:?}");
            if s != last {
                println!("[progress] {s}");
                last = s;
            }
        })
        .expect("flux2_klein_9b generate");
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };

    let mean = image_mean(&image);
    let std = image_std(&image);
    let out_dir = PathBuf::from(env_or("KLEIN_Q4_OUT_DIR", "/tmp/klein_q4_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let png = out_dir.join(format!("flux2_klein_9b_q4_{steps}step.png"));
    save_png(&image, &png);
    println!(
        "[worker-smoke] {MODEL_ID} q4 {}x{} mean {mean:.2} std {std:.2} -> {}",
        image.width,
        image.height,
        png.display()
    );
    assert_eq!(
        (image.width, image.height),
        (w, h),
        "wrong output dimensions"
    );
    assert!(!is_all_zero(&image), "{MODEL_ID} q4 decode is ALL-ZERO");
    assert!(
        std > DEGENERATE_STD_FLOOR_TIGHT,
        "{MODEL_ID} q4 render looks degenerate (std {std:.2})"
    );
    println!(
        "[worker-smoke] DONE: {MODEL_ID} loaded natively from the HF cache at both the default q8 \
         tier and an explicit q4 pick, and rendered coherently at q4"
    );
}
