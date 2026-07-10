//! sc-10618 — real-weight MLX **train → apply** smoke for the Illustrious-XL SDXL-family lane.
//!
//! This is the E2E evidence the epic 10609 training half demands: not "a registry entry + a green
//! unit test" but an actual round-trip on real weights. One `#[ignore]`d test, driven by env, that for
//! one `(model, network_type)` cell:
//!   1. loads the **native-MLX SDXL trainer** (`mlx_gen_sdxl::load_trainer`) from the Illustrious
//!      turnkey's **dense `bf16/` tier** — the exact tier `resolve_base_model_path` hands the worker
//!      (`tiered_turnkey_train_dir` descends to `bf16/`), never a quantized tier,
//!   2. trains a tiny LoRA or LoKr adapter over a small synthetic dataset (few steps — the point is the
//!      seam, not convergence),
//!   3. inspects the emitted `.safetensors` — the dual-text-encoder key signature that makes it an
//!      `sdxl`-family adapter, and (for LoKr) that **no `mid_block` factors were emitted** (sc-2640: the
//!      SDXL LoKr surface is down/up attention only, so `mid_block` factors would be dead weight),
//!   4. renders the SAME prompt/seed WITHOUT and WITH the adapter applied (`LoadSpec::with_adapters`)
//!      and asserts the adapter **visibly changes the output** — the step that catches a trainer
//!      emitting keys no inference path reads (a zero delta), and
//!   5. records wall-clock + peak MLX memory for the manifest footprint record.
//!
//! Direct provider constructors (`mlx_gen_sdxl::load{,_trainer}`), NOT the `gen_core` registry: the
//! worker test build links a candle-backed `sdxl` **trainer stub** (engines.rs, exercises the candle
//! training gate) and an `sdxl` generator, so the first-wins registry could resolve the stub. The
//! direct constructors are the same trainer/generator the registry resolves on the real app, minus the
//! id lookup — the same sidestep `lora_train_driver` makes with `mlx_gen_z_image::load`.
//!
//! macOS/Metal + `RUST_TEST_THREADS=1` (MLX is `!Send`). Run one cell per invocation (each loads the
//! full SDXL model), e.g. the v1 LoRA cell from the scratch-built turnkey, applying on the dense tier:
//! ```sh
//! ILL_BF16_DIR=/path/to/ill-v1-fixed/bf16 ILL_MODEL=illustrious_xl_v1 ILL_NETWORK=lora \
//!   RUST_TEST_THREADS=1 cargo test -p sceneworks-worker --release --lib \
//!   illustrious_train_apply_mlx_smoke -- --ignored --nocapture
//! ```
//! Knobs: `ILL_STEPS` `ILL_RANK` `ILL_RES` `ILL_SEED` `ILL_SCALE` `ILL_OUT_DIR`.
//!
//! Apply tier — this smoke applies on the DENSE tier (`ILL_GEN_QUANT` defaults to `bf16`). Applying on a
//! packed tier (`ILL_GEN_QUANT=q8`/`q4`) currently fails at load: mlx-gen-sdxl applies a LoRA by MERGING
//! the delta into the base (`merge_dense_delta`), which requires a dense base and rejects a quantized one
//! ("a LoRA must be merged before quantization"). That is a pre-existing SDXL-family limitation (not
//! Illustrious-specific) and belongs to the LoRA-inference story (sc-10616), not this training E2E — so
//! the packed knob exists for that investigation but is expected to fail until sc-10616 addresses it.

use std::path::{Path, PathBuf};
use std::time::Instant;

use gen_core::{
    AdapterKind, AdapterSpec, CancelFlag, GenerationOutput, GenerationRequest, Image, LoadSpec,
    NetworkType, Quant, TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest,
    WeightsSource,
};

use super::smoke_support::{env_or, image_std, is_all_zero, save_png, DEGENERATE_STD_FLOOR_TIGHT};

/// A distinct anime-styled swatch dataset: `n` procedurally-varied images so the tiny run has real
/// (non-flat) gradient signal to bias the output, captioned with a shared trigger token. Deterministic.
fn write_synthetic_dataset(dir: &Path, n: u32, res: u32, trigger: &str) -> Vec<TrainingItem> {
    std::fs::create_dir_all(dir).expect("dataset dir");
    let caption = format!("{trigger}, 1girl, solo, anime, vibrant, masterpiece");
    (0..n)
        .map(|i| {
            let phase = i as f32 * 1.7;
            let path = dir.join(format!("swatch_{i:02}.png"));
            image::RgbImage::from_fn(res, res, |x, y| {
                let fx = x as f32 / res as f32;
                let fy = y as f32 / res as f32;
                // Per-image varied bands + a bright focal blob — distinct across the set, not a flat fill.
                let r = ((fx * 6.0 + phase).sin() * 0.5 + 0.5) * 255.0;
                let g = ((fy * 5.0 + phase * 1.3).cos() * 0.5 + 0.5) * 255.0;
                let d = ((fx - 0.5).powi(2) + (fy - 0.5).powi(2)).sqrt();
                let b = ((1.0 - d).max(0.0) * 255.0).min(255.0);
                image::Rgb([r as u8, g as u8, b as u8])
            })
            .save(&path)
            .expect("write swatch");
            TrainingItem {
                image_path: path,
                caption: caption.clone(),
            }
        })
        .collect()
}

/// Tensor names inside a `.safetensors` file, read straight from the header (8-byte LE length prefix +
/// JSON map of `name -> {...}`, plus an optional `__metadata__`). Dependency-free so the smoke inspects
/// exactly what the trainer wrote, without a model load.
fn safetensors_tensor_names(path: &Path) -> Vec<String> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(bytes.len() >= 8, "safetensors {} too short", path.display());
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len])
        .unwrap_or_else(|e| panic!("parse safetensors header {}: {e}", path.display()));
    header
        .as_object()
        .expect("safetensors header is an object")
        .keys()
        .filter(|k| k.as_str() != "__metadata__")
        .cloned()
        .collect()
}

/// Load the SDXL generator from `dir` at the requested quant and render one image for `prompt`/`seed`,
/// optionally with `adapter` applied. Each call loads + drops its own model (MLX cache cleared after) so
/// the baseline and adapter renders don't co-reside.
fn render(
    dir: &Path,
    quant: Option<Quant>,
    adapter: Option<AdapterSpec>,
    prompt: &str,
    seed: u64,
    res: u32,
) -> Image {
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()));
    if let Some(q) = quant {
        spec = spec.with_quant(q);
    }
    if let Some(a) = adapter {
        spec = spec.with_adapters(vec![a]);
    }
    let generator = mlx_gen_sdxl::load(&spec).expect("load mlx sdxl generator");
    let req = GenerationRequest {
        prompt: prompt.to_owned(),
        negative_prompt: Some("blurry, low quality, deformed, watermark".to_owned()),
        width: res,
        height: res,
        count: 1,
        seed: Some(seed),
        steps: Some(
            env_or("ILL_GEN_STEPS", "24")
                .parse()
                .expect("ILL_GEN_STEPS"),
        ),
        guidance: Some(7.0),
        ..Default::default()
    };
    let output = generator
        .generate(&req, &mut |_p| {})
        .expect("sdxl generate");
    mlx_rs::memory::clear_cache();
    match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned an image"),
        other => panic!("expected Images output, got {other:?}"),
    }
}

/// Mean absolute per-byte difference between two same-shape RGB images (0–255 scale).
fn mean_abs_delta(a: &Image, b: &Image) -> f64 {
    assert_eq!(
        (a.width, a.height, a.pixels.len()),
        (b.width, b.height, b.pixels.len()),
        "before/after images must share dimensions"
    );
    let n = a.pixels.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    a.pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as f64)
        .sum::<f64>()
        / n
}

fn quant_from_env() -> (Option<Quant>, &'static str) {
    match env_or("ILL_GEN_QUANT", "none").as_str() {
        "q8" => (Some(Quant::Q8), "q8"),
        "q4" => (Some(Quant::Q4), "q4"),
        _ => (None, "bf16"),
    }
}

#[test]
#[ignore = "real-weight MLX train→apply smoke (sc-10618); needs an Illustrious turnkey bf16 tier + an Apple-Silicon Mac. Set ILL_BF16_DIR"]
fn illustrious_train_apply_mlx_smoke() {
    let bf16_dir = PathBuf::from(
        std::env::var("ILL_BF16_DIR")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| {
                panic!(
                    "set ILL_BF16_DIR to the Illustrious turnkey's dense bf16 tier \
                     (e.g. .../illustrious-xl-v1-mlx/bf16 — the tier resolve_base_model_path trains from)"
                )
            })
            .trim(),
    );
    assert!(
        bf16_dir.join("unet").is_dir(),
        "ILL_BF16_DIR must be a diffusers component tree with unet/ (the dense bf16 tier); got {}",
        bf16_dir.display()
    );

    let model = env_or("ILL_MODEL", "illustrious");
    let network_str = env_or("ILL_NETWORK", "lora");
    let network = NetworkType::parse(&network_str);
    let is_lokr = matches!(network, NetworkType::Lokr);
    let gen_dir = std::env::var("ILL_GEN_DIR")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(|v| PathBuf::from(v.trim()))
        .unwrap_or_else(|| bf16_dir.clone());
    let (gen_quant, gen_quant_label) = quant_from_env();

    let steps: u32 = env_or("ILL_STEPS", "24").parse().expect("ILL_STEPS");
    let rank: u32 = env_or("ILL_RANK", "8").parse().expect("ILL_RANK");
    let res: u32 = env_or("ILL_RES", "768").parse().expect("ILL_RES");
    let seed: u64 = env_or("ILL_SEED", "7").parse().expect("ILL_SEED");
    let scale: f32 = env_or("ILL_SCALE", "1.4").parse().expect("ILL_SCALE");
    let out_dir = PathBuf::from(env_or("ILL_OUT_DIR", "/tmp/ill_train_smoke"));
    std::fs::create_dir_all(&out_dir).expect("out dir");

    let trigger = "sks_ill";
    let data_dir = out_dir.join(format!("{model}_{network_str}_data"));
    let items = write_synthetic_dataset(&data_dir, 6, res, trigger);
    let file_name = format!("{model}_{network_str}.safetensors");

    // ── 1. TRAIN on the dense bf16 tier ────────────────────────────────────────────────────────────
    let config = TrainingConfig {
        rank,
        alpha: rank as f32,
        learning_rate: 1e-4,
        steps,
        batch_size: 1,
        gradient_accumulation: 1,
        gradient_checkpointing: false,
        train_dtype: "bf16".to_owned(),
        resolution: res,
        save_every: 0,
        seed,
        optimizer: "adamw".to_owned(),
        weight_decay: 0.0,
        network_type: network,
        // -1 = the trainer's auto Kronecker block-split for LoKr (ignored for LoRA).
        decompose_factor: -1,
        // SDXL attention projections — the sdxl_lora target's loraTargetModules (sc-10617).
        lora_target_modules: vec![
            "to_q".to_owned(),
            "to_k".to_owned(),
            "to_v".to_owned(),
            "to_out.0".to_owned(),
        ],
        loss_type: "mse".to_owned(),
        trigger_word: Some(trigger.to_owned()),
        // A mid-run preview at POSITIVE guidance — SDXL uses real CFG (acceptance #6), unlike a
        // distilled base sampling at 0.0. `sample_every` must land on a NON-final step: the engine
        // skips sampling on the final step (the candle sample-count finding), so `sample_every = steps`
        // never previews. Half-way through fires at least once mid-run.
        sample_every: (steps / 2).max(1),
        sample_prompts: vec![format!("{trigger}, 1girl, solo, anime portrait")],
        sample_steps: 8,
        sample_guidance_scale: 7.0,
        ..Default::default()
    };
    let request = TrainingRequest {
        items,
        config,
        output_dir: out_dir.clone(),
        file_name: file_name.clone(),
        trigger_words: vec![trigger.to_owned()],
        cancel: CancelFlag::new(),
    };

    println!(
        "[ill-train {model}/{network_str}] loading SDXL trainer from bf16 tier {} ...",
        bf16_dir.display()
    );
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let t0 = Instant::now();
    let mut trainer =
        mlx_gen_sdxl::load_trainer(&LoadSpec::new(WeightsSource::Dir(bf16_dir.clone())))
            .expect("mlx sdxl trainer loads from the Illustrious bf16 tier");
    trainer
        .validate(&request)
        .expect("trainer accepts the Illustrious training plan");
    let mut last_loss = f32::NAN;
    let mut sample_count = 0u32;
    let mut first_sample: Option<Image> = None;
    let output = trainer
        .train(&request, &mut |progress| match progress {
            TrainingProgress::Training { step, total, loss } => {
                last_loss = loss;
                if step == 1 || step % 5 == 0 || step == total {
                    println!("[ill-train {model}/{network_str}] step {step}/{total} loss {loss:.4}");
                }
            }
            // The trainer streams in-training previews as events carrying a decoded RGB bitmap (the
            // worker persists them via `write_training_sample`); capture the first to validate #6.
            TrainingProgress::Sample {
                step,
                prompt,
                image,
                ..
            } => {
                sample_count += 1;
                if first_sample.is_none() {
                    println!(
                        "[ill-train {model}/{network_str}] preview sample @ step {step} prompt {prompt:?}"
                    );
                    first_sample = Some(image);
                }
            }
            _ => {}
        })
        .expect("training runs to completion");
    let train_secs = t0.elapsed().as_secs_f64();
    let train_peak_gb = mlx_rs::memory::get_peak_memory() as f64 / 1e9;
    drop(trainer);
    mlx_rs::memory::clear_cache();

    let adapter_path = output.adapter_path.clone();
    println!(
        "[ill-train {model}/{network_str}] DONE steps={} final_loss={} last={last_loss:.4} \
         wall={train_secs:.1}s peak={train_peak_gb:.1}GB adapter={}",
        output.steps,
        output.final_loss,
        adapter_path.display()
    );
    assert!(output.steps >= 1, "at least one training step ran");
    assert!(output.final_loss.is_finite(), "final loss is finite");
    assert!(last_loss.is_finite(), "a per-step loss was observed");
    assert!(
        adapter_path.exists(),
        "trained adapter written at {}",
        adapter_path.display()
    );

    // In-training preview at POSITIVE guidance (acceptance #6): SDXL uses real CFG (guidance 7.0),
    // unlike a distilled base sampling at 0.0. The trainer emitted TrainingProgress::Sample event(s);
    // assert at least one fired and rendered a non-degenerate image at the configured positive guidance.
    let preview = first_sample.expect(
        "no TrainingProgress::Sample emitted — expected an in-training preview at positive guidance \
         (sample_every>0, sample_prompts set)",
    );
    let preview_std = image_std(&preview);
    let preview_png = out_dir.join(format!("{model}_{network_str}_preview.png"));
    save_png(&preview, &preview_png);
    println!(
        "[ill-train {model}/{network_str}] {sample_count} positive-guidance preview(s); first std {preview_std:.1} -> {}",
        preview_png.display()
    );
    assert!(
        preview_std > DEGENERATE_STD_FLOOR_TIGHT && !is_all_zero(&preview),
        "in-training preview at positive guidance looks degenerate (std {preview_std:.1})"
    );

    // ── 2. INSPECT the adapter: sdxl-family signature + (LoKr) no dead mid_block factors ────────────
    let names = safetensors_tensor_names(&adapter_path);
    assert!(!names.is_empty(), "adapter carries tensors");
    let has_te1 = names
        .iter()
        .any(|k| k.contains("te1") || k.contains("text_encoder."));
    let has_te2 = names
        .iter()
        .any(|k| k.contains("te2") || k.contains("text_encoder_2"));
    let has_unet = names
        .iter()
        .any(|k| k.contains("unet") || k.contains("down_blocks") || k.contains("up_blocks"));
    let mid_block_keys: Vec<&String> = names.iter().filter(|k| k.contains("mid_block")).collect();
    let mut prefixes: Vec<String> = names
        .iter()
        .map(|k| k.split(['.', '_']).take(3).collect::<Vec<_>>().join("_"))
        .collect();
    prefixes.sort();
    prefixes.dedup();
    println!(
        "[ill-inspect {model}/{network_str}] {} tensors; te1={has_te1} te2={has_te2} unet={has_unet} \
         mid_block_keys={} sample_prefixes={:?}",
        names.len(),
        mid_block_keys.len(),
        prefixes.iter().take(8).collect::<Vec<_>>()
    );
    assert!(
        (has_te1 && has_te2) || has_unet,
        "adapter must carry the SDXL dual-text-encoder or UNet key signature (family=sdxl); prefixes: {prefixes:?}"
    );
    if is_lokr {
        // sc-2640: the SDXL LoKr inference surface is down/up attention only; emitting mid_block LoKr
        // factors would produce weights no inference path reads. Confirm the trainer honors that.
        assert!(
            mid_block_keys.is_empty(),
            "LoKr adapter emitted {} mid_block factor(s) — sc-2640 says the SDXL LoKr surface excludes \
             mid_block, so these would be dead weight: {mid_block_keys:?}",
            mid_block_keys.len()
        );
    }

    // ── 3. APPLY back at generation: same seed/prompt, without vs with the adapter ──────────────────
    let gen_prompt = format!("{trigger}, 1girl, solo, anime portrait, detailed, sharp focus");
    println!(
        "[ill-apply {model}/{network_str}] rendering baseline (no adapter) from {} tier {} ...",
        gen_quant_label,
        gen_dir.display()
    );
    let baseline = render(&gen_dir, gen_quant, None, &gen_prompt, seed, res);
    save_png(
        &baseline,
        &out_dir.join(format!("{model}_{network_str}_baseline.png")),
    );

    println!("[ill-apply {model}/{network_str}] rendering WITH adapter (scale {scale}) ...");
    let with_adapter = render(
        &gen_dir,
        gen_quant,
        Some(AdapterSpec::new(
            adapter_path.clone(),
            scale,
            if is_lokr {
                AdapterKind::Lokr
            } else {
                AdapterKind::Lora
            },
        )),
        &gen_prompt,
        seed,
        res,
    );
    let after_png = out_dir.join(format!("{model}_{network_str}_with_adapter.png"));
    save_png(&with_adapter, &after_png);

    let base_std = image_std(&baseline);
    let after_std = image_std(&with_adapter);
    let delta = mean_abs_delta(&baseline, &with_adapter);
    println!(
        "[ill-apply {model}/{network_str}] base_std={base_std:.1} after_std={after_std:.1} \
         mean_abs_delta={delta:.2} -> {}",
        after_png.display()
    );

    // Both renders must be alive (a broken decode / all-zero would trivially "differ" or "match").
    assert!(
        !is_all_zero(&baseline),
        "baseline render is all-zero (broken decode)"
    );
    assert!(
        !is_all_zero(&with_adapter),
        "adapter render is all-zero (broken decode)"
    );
    assert!(
        base_std > DEGENERATE_STD_FLOOR_TIGHT && after_std > DEGENERATE_STD_FLOOR_TIGHT,
        "a render looks degenerate (base_std {base_std:.1}, after_std {after_std:.1})"
    );
    // The load-bearing assertion: the adapter is actually READ at inference. A trainer that emits keys
    // no inference path consumes yields an identical image (delta ~0). Same seed + same prompt isolate
    // the adapter as the only variable, so any real applied adapter moves the pixels well clear of noise.
    assert!(
        delta > 1.0,
        "adapter did not change the output (mean_abs_delta {delta:.2} on 0–255) — the trained {network_str} \
         adapter is not being applied at inference (keys the loader doesn't read?)"
    );
    println!(
        "[ill-DONE {model}/{network_str}] train→apply CLOSED: trained on bf16, adapter applies on {gen_quant_label}, \
         family=sdxl, delta={delta:.2}, train {train_secs:.1}s peak {train_peak_gb:.1}GB"
    );
}

/// sc-10616 — real THIRD-PARTY **kohya** LoRA inference validation. Distinct from the training smoke
/// above: it loads a community Illustrious LoRA in kohya format (`lora_te1_`/`lora_te2_`/`lora_unet_`,
/// the Civitai/HF convention — NOT the PEFT naming the trainer emits) and proves (a) `lora_family`
/// classifies it as `sdxl` off the dual-text-encoder key split (the real detection path, not the
/// happy path where metadata declares the family), and (b) it applies at generation and visibly
/// changes output. Applies on the DENSE bf16 tier — a packed q4/q8 base fails `merge_dense_delta`
/// (a pre-existing family-wide constraint, filed as the sc-10616 finding).
///
/// ```sh
/// ILL_BF16_DIR=/path/ill-v1-fixed/bf16 ILL_MODEL=illustrious_xl_v1 \
///   ILL_LORA_PATH="/path/The Artist - Style for Illustrious XL.safetensors" \
///   RUST_TEST_THREADS=1 cargo test -p sceneworks-worker --lib \
///   illustrious_external_kohya_lora_apply_mlx_smoke -- --ignored --nocapture
/// ```
#[test]
#[ignore = "real-weight MLX inference smoke (sc-10616); needs an Illustrious bf16 tier + a kohya LoRA. Set ILL_BF16_DIR, ILL_LORA_PATH"]
fn illustrious_external_kohya_lora_apply_mlx_smoke() {
    let bf16_dir = PathBuf::from(
        std::env::var("ILL_BF16_DIR")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .expect("set ILL_BF16_DIR to the Illustrious dense bf16 tier")
            .trim(),
    );
    assert!(
        bf16_dir.join("unet").is_dir(),
        "ILL_BF16_DIR must be a diffusers component tree with unet/; got {}",
        bf16_dir.display()
    );
    let lora_path = PathBuf::from(
        std::env::var("ILL_LORA_PATH")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .expect("set ILL_LORA_PATH to a kohya Illustrious LoRA .safetensors")
            .trim(),
    );
    assert!(
        lora_path.is_file(),
        "ILL_LORA_PATH is not a file: {}",
        lora_path.display()
    );
    let model = env_or("ILL_MODEL", "illustrious");
    let res: u32 = env_or("ILL_RES", "1024").parse().expect("ILL_RES");
    let seed: u64 = env_or("ILL_SEED", "7").parse().expect("ILL_SEED");
    let scale: f32 = env_or("ILL_SCALE", "1.0").parse().expect("ILL_SCALE");
    let out_dir = PathBuf::from(env_or("ILL_OUT_DIR", "/tmp/ill_ext_lora"));
    std::fs::create_dir_all(&out_dir).expect("out dir");

    // (1) Kohya dual-text-encoder format + the REAL family-detection path (not a declared-metadata
    // shortcut): `detect_model_family` must classify it `sdxl` off the `lora_te1_`/`lora_te2_` split.
    let names = safetensors_tensor_names(&lora_path);
    let has_te1 = names.iter().any(|k| k.starts_with("lora_te1_"));
    let has_te2 = names.iter().any(|k| k.starts_with("lora_te2_"));
    let has_unet = names.iter().any(|k| k.starts_with("lora_unet_"));
    println!(
        "[ill-ext {model}] adapter {} tensors: kohya lora_te1_={has_te1} lora_te2_={has_te2} lora_unet_={has_unet}",
        names.len()
    );
    assert!(
        has_te1 && has_te2 && has_unet,
        "expected a kohya SDXL dual-text-encoder LoRA (lora_te1_/lora_te2_/lora_unet_)"
    );
    let detected = sceneworks_core::lora_family::detect_model_family(&lora_path)
        .expect("family detection reads the safetensors header");
    assert_eq!(
        detected.as_deref(),
        Some("sdxl"),
        "lora_family must classify a kohya dual-TE Illustrious LoRA as `sdxl` (got {detected:?})"
    );
    println!("[ill-ext {model}] lora_family detected: {detected:?}");

    // (2) Apply at generation on the dense tier — same seed, without vs with the LoRA.
    let prompt = env_or(
        "ILL_PROMPT",
        "1girl, solo, silver hair, anime portrait, detailed, masterpiece",
    );
    let baseline = render(&bf16_dir, None, None, &prompt, seed, res);
    save_png(
        &baseline,
        &out_dir.join(format!("{model}_extlora_baseline.png")),
    );
    let with_lora = render(
        &bf16_dir,
        None,
        Some(AdapterSpec::new(
            lora_path.clone(),
            scale,
            AdapterKind::Lora,
        )),
        &prompt,
        seed,
        res,
    );
    let after_png = out_dir.join(format!("{model}_extlora_with.png"));
    save_png(&with_lora, &after_png);

    let (bstd, astd) = (image_std(&baseline), image_std(&with_lora));
    let delta = mean_abs_delta(&baseline, &with_lora);
    println!(
        "[ill-ext {model}] base_std={bstd:.1} after_std={astd:.1} mean_abs_delta={delta:.2} -> {}",
        after_png.display()
    );
    assert!(
        !is_all_zero(&baseline) && !is_all_zero(&with_lora),
        "a render is all-zero (broken decode)"
    );
    assert!(
        bstd > DEGENERATE_STD_FLOOR_TIGHT && astd > DEGENERATE_STD_FLOOR_TIGHT,
        "a render looks degenerate (base_std {bstd:.1}, after_std {astd:.1})"
    );
    assert!(
        delta > 1.0,
        "the kohya LoRA did not change output (mean_abs_delta {delta:.2}) — not applied at inference"
    );
    println!(
        "[ill-ext-DONE {model}] third-party kohya LoRA: family=sdxl, applies on bf16, delta={delta:.2}"
    );
}
