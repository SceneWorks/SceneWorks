//! Local real-weight MLX smoke for an imported **bare-keyed** Krea 2 Turbo transformer (sc-25855).
//! `#[ignore]`d — run by hand on an Apple-Silicon Mac.
//!
//! Community Krea 2 finetunes such as `lodestones/Kroma` name their tensors `blocks.0.attn.wq.weight`
//! with no `model.diffusion_model.` prefix. Candle always resolved that; MLX refused every bare key
//! until inference #1237, so such a file passed the import gate and then failed at load on a Mac.
//!
//! It drives the load the imported-Krea lane builds (`generate_krea_imported_stream`), minus the
//! API/job plumbing and path confinement:
//!   1. `imported_model_descriptor("krea_2", TransformerFile, Generate)` — the registry row,
//!   2. `LoadSpec::new(File(<checkpoint>))` + the Turbo `bf16/` tier as `BASE_SNAPSHOT_COMPONENT`,
//!   3. `inference_runtime::load` — the real engine load,
//!   4. one Euler render at the lane's 8-step default, checked non-degenerate.
//!
//! Setup — with `SceneWorks/krea-2-turbo-mlx` `bf16/` in the HF cache:
//! ```text
//! # optional: KREA_BARE_STEPS=8 KREA_BARE_W=1024 KREA_BARE_H=1024 KREA_BARE_PROMPT="..." KREA_BARE_OUT_DIR=/tmp/krea_bare_smoke
//! KREA_BARE_CHECKPOINT=/path/to/kroma-v0.3.1-turbo-opd.safetensors \
//!   cargo test -p sceneworks-worker --release krea_imported_bare_keys_mlx_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use gen_core::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};

use super::smoke_support::{env_or, image_mean, image_std, is_all_zero, save_png};

const BASE_REPO: &str = "SceneWorks/krea-2-turbo-mlx";

/// The cached Turbo snapshot's dense `bf16/` tier: the text encoder, VAE, tokenizer and DiT config
/// the transformer-only checkpoint is paired with.
fn cached_base_tier() -> Option<PathBuf> {
    let hub = ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"]
        .iter()
        .find_map(|key| std::env::var(key).ok().filter(|v| !v.trim().is_empty()))
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HF_HOME")
                .ok()
                .filter(|v| !v.trim().is_empty())
                .map(|home| PathBuf::from(home).join("hub"))
        })
        .or_else(|| {
            Some(PathBuf::from(std::env::var("HOME").ok()?).join(".cache/huggingface/hub"))
        })?;
    let snapshots = hub
        .join(format!("models--{}", BASE_REPO.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|entry| entry.ok())
        .find_map(|entry| {
            let tier = entry.path().join("bf16");
            tier.join("transformer/config.json")
                .is_file()
                .then_some(tier)
        })
}

#[test]
#[ignore = "real-weight MLX smoke; needs KREA_BARE_CHECKPOINT (a bare-keyed Krea 2 Turbo DiT, ~26GB) + the SceneWorks/krea-2-turbo-mlx bf16 tier cached + an Apple-Silicon Mac"]
fn krea_imported_bare_keys_mlx_gpu_smoke() {
    let checkpoint = PathBuf::from(env_or("KREA_BARE_CHECKPOINT", ""));
    assert!(
        checkpoint.is_file(),
        "set KREA_BARE_CHECKPOINT to a bare-keyed Krea 2 Turbo transformer .safetensors"
    );
    let base = cached_base_tier().unwrap_or_else(|| {
        panic!("{BASE_REPO} bf16 tier is not in the HF cache — pull it before running this smoke")
    });

    let descriptor = crate::inference_runtime::imported_model_descriptor(
        "krea_2",
        gen_core::ImportedModelSource::TransformerFile,
        gen_core::ImportedModelOperation::Generate,
    )
    .expect("an imported Krea 2 transformer-file provider is registered");
    let spec = LoadSpec::new(WeightsSource::File(checkpoint.clone()))
        .with_component(gen_core::BASE_SNAPSHOT_COMPONENT, WeightsSource::Dir(base));
    let model = crate::inference_runtime::load(descriptor.id, &spec).unwrap_or_else(|error| {
        panic!(
            "{} must load on MLX — an engine pinned before inference #1237 refuses its bare keys: \
             {error}",
            checkpoint.display()
        )
    });

    let steps: u32 = env_or("KREA_BARE_STEPS", "8").parse().expect("steps");
    let width: u32 = env_or("KREA_BARE_W", "1024").parse().expect("width");
    let height: u32 = env_or("KREA_BARE_H", "1024").parse().expect("height");
    let prompt = env_or(
        "KREA_BARE_PROMPT",
        "a red fox sitting in a snowy pine forest at dawn, photograph",
    );
    let request = GenerationRequest {
        prompt,
        width,
        height,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        sampler: Some("euler".to_owned()),
        ..Default::default()
    };
    let mut last = String::new();
    let output = model
        .generate(&request, &mut |progress| {
            let line = format!("{progress:?}");
            if line != last {
                println!("[progress] {line}");
                last = line;
            }
        })
        .expect("bare-keyed imported Krea 2 generation");
    let GenerationOutput::Images(images) = output else {
        panic!("the imported Krea 2 provider must produce images");
    };
    let image = images.first().expect("one image");

    assert!(!is_all_zero(image), "the render must not be all-zero");
    let (mean, std) = (image_mean(image), image_std(image));
    println!("render: mean={mean:.2} std={std:.2} ({width}x{height}, {steps} steps)");
    // A sanity floor only: it catches a flat or black frame, not a wrong image. Whether the
    // checkpoint's weights reached the right modules is the saved PNG's job.
    assert!(
        std >= 20.0 && (10.0..245.0).contains(&mean),
        "the render looks degenerate (mean {mean:.2}, std {std:.2})"
    );

    let out_dir = PathBuf::from(env_or("KREA_BARE_OUT_DIR", "/tmp/krea_bare_smoke"));
    std::fs::create_dir_all(&out_dir).expect("out dir");
    let out = out_dir.join("krea_imported_bare_keys.png");
    save_png(image, &out);
    println!("wrote {}", out.display());
}
