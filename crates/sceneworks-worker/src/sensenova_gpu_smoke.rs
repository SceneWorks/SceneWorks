//! Local real-weight GPU smoke for the candle **SenseNova-U1 8B** worker lane (sc-13817, epic 13678).
//! `#[ignore]`d — run by hand on the RTX PRO 6000. It drives the real candle SenseNova engine via
//! `crate::inference_runtime::load("sensenova_u1_8b{,_fast}")` with a `LoadSpec` pointed at the tier
//! `resolve_weights_dir` resolves — the exact runtime seam `generate_candle_stream` uses, minus the
//! API/job plumbing.
//!
//! **Why this smoke exists.** Every other candle image family (anima / flux2_dev / sana / sdxl) has a
//! GPU smoke; SenseNova had none, so its candle lane had never been executed on hardware. It had
//! also never *worked*: the loader (`candle-gen-sensenova`) dense-loads a flat `*.safetensors`
//! backbone via `f32_vb` and hard-rejects `spec.quantize`, but every sensenova catalog entry flags
//! `mlx.standardTierLayout: true` with no `mlx.minQualityTier`, so `standard_tier_subdir` resolved
//! the app-wide **q8** default — an MLX-packed tier the engine cannot read. sc-13817 forces the lane
//! onto the dense `bf16/` tier; this smoke is the hardware evidence for that fix.
//!
//! **The tier is the point.** `SENSENOVA_DIR` must be the **dense bf16** tier dir (a flat
//! `model.safetensors` + `tokenizer.json`, plus `distill_merged.json` on the `_fast` turnkey). The
//! smoke asserts that shape up front rather than letting a packed q4/q8 dir reach the loader and fail
//! with an opaque tensor-name error — pointing this at a packed tier is a *setup* mistake and should
//! say so.
//!
//! Note the dense tier is ~35 GB of bf16 and `f32_vb` mmaps it as **F32**, so expect roughly double
//! that resident on the device. This is a 96 GB-class-card smoke.
//!
//! Build with `CUDA_COMPUTE_CAP=120` (native Blackwell sm_120).
//!
//! Setup (PowerShell):
//! ```text
//! $env:SENSENOVA_DIR="E:\huggingface\hub\models--SceneWorks--sensenova-u1-8b-fast-mlx\snapshots\<hash>\bf16"
//! $env:SENSENOVA_OUT_DIR="D:\sceneworks-sensenova-validate"
//! # optional: SENSENOVA_VARIANT=base|fast  SENSENOVA_W=1024 SENSENOVA_H=1024
//! #           SENSENOVA_STEPS=..  SENSENOVA_GUIDANCE=..  SENSENOVA_SEED=42  SENSENOVA_PROMPT="..."
//! cargo test -p sceneworks-worker --no-default-features --features backend-candle --release \
//!   sensenova_candle_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use gen_core::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};

use super::smoke_support::{env_or, image_std, save_png, DEGENERATE_STD_FLOOR_DEFAULT};

/// Resolve the SenseNova variant → engine id + its shipped defaults (steps/guidance).
///
/// `fast` is the 8-step distilled turnkey (CFG-free, guidance 1.0 — the distill LoRA is pre-merged
/// into the tier); `base` is the undistilled 50-step model at guidance 4.0. Both match the catalog
/// `defaults` for the corresponding manifest entry.
fn resolve_variant(raw: &str) -> (&'static str, u32, f32) {
    match raw.trim().to_ascii_lowercase().as_str() {
        "base" => ("sensenova_u1_8b", 50, 4.0),
        "fast" => ("sensenova_u1_8b_fast", 8, 1.0),
        other => panic!("SENSENOVA_VARIANT must be base|fast, got {other:?}"),
    }
}

/// Whether `dir` holds a flat dense backbone — the shape `candle-gen-sensenova`'s `f32_vb` mmaps
/// (`*.safetensors` directly under the tier dir, no `transformer/` component tree).
///
/// A packed MLX `q4/`/`q8/` tier also carries `*.safetensors`, so this is a *shape* check, not a
/// precision check; the caller pairs it with the `bf16` path hint in its failure message. A tier
/// holding only configs/tokenizer (the torn shape a partial download leaves) fails it.
fn has_flat_safetensors(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.flatten().any(|entry| {
            let path = entry.path();
            !sceneworks_core::lora_family::is_hidden_file(&path)
                && path.extension().is_some_and(|ext| ext == "safetensors")
        })
    })
}

fn env_path(key: &str) -> PathBuf {
    // Trim: a cmd `set VAR=value && ...` keeps the trailing space before `&&`
    // (reference_candle_build_toolchain_win — cost two build cycles on sc-9994).
    PathBuf::from(
        std::env::var(key)
            .unwrap_or_else(|_| panic!("set ${key}"))
            .trim(),
    )
}

#[cfg(test)]
mod variant_tests {
    use super::{has_flat_safetensors, resolve_variant};

    #[test]
    fn resolve_variant_maps_the_two_ids_with_their_shipped_defaults() {
        // The distilled turnkey is 8-step CFG-free; the base model is 50-step at guidance 4.0.
        assert_eq!(resolve_variant("base"), ("sensenova_u1_8b", 50, 4.0));
        assert_eq!(resolve_variant(" FAST ").0, "sensenova_u1_8b_fast");
        assert_eq!(resolve_variant("fast").1, 8);
        assert_eq!(resolve_variant("fast").2, 1.0);
    }

    #[test]
    #[should_panic(expected = "SENSENOVA_VARIANT must be base|fast")]
    fn resolve_variant_rejects_unknown() {
        let _ = resolve_variant("turbo");
    }

    #[test]
    fn flat_safetensors_probe_rejects_a_config_only_tier() {
        // The torn shape a partial download leaves — configs present, weights absent. Handing this to
        // the loader produces an opaque "no .safetensors found"; the smoke must catch it as setup.
        let torn = tempfile::tempdir().unwrap();
        std::fs::write(torn.path().join("config.json"), "{}").unwrap();
        std::fs::write(torn.path().join("tokenizer.json"), "{}").unwrap();
        assert!(!has_flat_safetensors(torn.path()));

        let dense = tempfile::tempdir().unwrap();
        std::fs::write(dense.path().join("model.safetensors"), "weights").unwrap();
        assert!(has_flat_safetensors(dense.path()));
    }
}

#[test]
#[ignore = "real-weight GPU smoke; needs the DENSE bf16 SenseNova-U1 tier (~35GB) + a CUDA device (cap=120)"]
fn sensenova_candle_gpu_smoke() {
    let weights_dir = env_path("SENSENOVA_DIR");
    let (variant_raw, _) = (env_or("SENSENOVA_VARIANT", "fast"), ());
    let (engine_id, default_steps, default_guidance) = resolve_variant(&variant_raw);

    // The tier is the whole point of sc-13817 — fail loudly on a packed/torn dir rather than letting
    // it reach the loader as an opaque tensor error.
    assert!(
        has_flat_safetensors(&weights_dir),
        "SENSENOVA_DIR must be the DENSE bf16 tier (a flat model.safetensors), got no .safetensors \
         in {}. The q4/q8 tiers are MLX-packed and unreadable by the candle engine.",
        weights_dir.display()
    );
    assert!(
        weights_dir.join("tokenizer.json").is_file(),
        "SENSENOVA_DIR must hold tokenizer.json (a self-contained SenseNova-U1 tier): {}",
        weights_dir.display()
    );
    if engine_id == "sensenova_u1_8b_fast" {
        // The `_fast` turnkey bakes the 8-step distill LoRA into its weights and ships this marker;
        // without it the loader would try to resolve+merge a distill LoRA that the tier does not ship
        // (the sc-13787 marker-skip is what makes the pre-merged tier loadable at all).
        assert!(
            weights_dir.join("distill_merged.json").is_file(),
            "the _fast tier must ship distill_merged.json (pre-merged 8-step distill, sc-8775/sc-13787): {}",
            weights_dir.display()
        );
    }

    let out_dir = PathBuf::from(env_or("SENSENOVA_OUT_DIR", "/tmp/sensenova_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let steps: u32 = env_or("SENSENOVA_STEPS", &default_steps.to_string())
        .parse()
        .expect("SENSENOVA_STEPS");
    let w: u32 = env_or("SENSENOVA_W", "1024").parse().expect("SENSENOVA_W");
    let h: u32 = env_or("SENSENOVA_H", "1024").parse().expect("SENSENOVA_H");
    let guidance: f32 = env_or("SENSENOVA_GUIDANCE", &default_guidance.to_string())
        .parse()
        .expect("SENSENOVA_GUIDANCE");
    let seed: u64 = env_or("SENSENOVA_SEED", "42")
        .parse()
        .expect("SENSENOVA_SEED");
    let prompt = env_or(
        "SENSENOVA_PROMPT",
        "a red vintage bicycle leaning against a weathered brick wall, golden hour light, sharp focus",
    );

    // Same seam as `generate_candle_stream`: a registry load of the candle sensenova engine with a
    // DENSE `LoadSpec` (no `.with_quant(..)` — the descriptor advertises `supported_quants: &[]` and
    // the engine hard-rejects a quantize request, so the worker resolves `(None, None)`).
    println!(
        "[smoke] loading {engine_id} (dense) from {} ...",
        weights_dir.display()
    );
    let spec = LoadSpec::new(WeightsSource::Dir(weights_dir.clone()));
    let generator = crate::inference_runtime::load(engine_id, &spec)
        .unwrap_or_else(|error| panic!("load candle {engine_id} provider: {error}"));

    let req = GenerationRequest {
        prompt: prompt.clone(),
        width: w,
        height: h,
        count: 1,
        seed: Some(seed),
        steps: Some(steps),
        guidance: Some(guidance),
        ..Default::default()
    };
    println!("[smoke] rendering {w}x{h} @ {steps} steps, guidance {guidance} ...");
    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |p| {
            let s = format!("{p:?}");
            if s != last {
                println!("[progress] {s}");
                last = s;
            }
        })
        .unwrap_or_else(|error| panic!("{engine_id} generate: {error}"));
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };

    let std = image_std(&image);
    let png = out_dir.join(format!("sensenova_{variant_raw}_{steps}step_{w}x{h}.png"));
    save_png(&image, &png);
    println!(
        "[smoke] {engine_id} {}x{} std {:.2} -> {}",
        image.width,
        image.height,
        std,
        png.display()
    );
    assert_eq!((image.width, image.height), (w, h));
    assert!(
        std > DEGENERATE_STD_FLOOR_DEFAULT,
        "{engine_id} render looks degenerate (std {std:.2}) — possible NaN / all-black decode \
         (check CUDA_COMPUTE_CAP=120 and that SENSENOVA_DIR is the DENSE bf16 tier)"
    );
    println!("[smoke] DONE: {engine_id} dense render coherent at {steps} steps");
}
