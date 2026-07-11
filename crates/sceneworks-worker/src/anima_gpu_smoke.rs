//! Local real-weight GPU smoke for the candle **Anima 2B** worker lane (epic 10512, sc-10625 — the
//! hardware-gated acceptance extracted from sc-10525). `#[ignore]`d — run by hand on the RTX PRO 6000.
//! It drives the real candle Anima engine via `gen_core::load("anima_{base,aesthetic,turbo}")` with a
//! `LoadSpec` pointed at the ungated `circlestone-labs/Anima` `split_files/` snapshot — the exact
//! runtime seam `generate_candle_stream` uses (minus the API/job plumbing) once the router routes Anima
//! to candle off-Mac.
//!
//! This is the "the candle lane has actually been RUN on real CUDA" evidence that unblocks flipping
//! `macOnly: false` / `candle_routed = true` (the whole point of sc-10625): the candle port landed
//! CPU-validated against sc-10524's framework-independent goldens but had NEVER been generated on a
//! GPU, because the dev machine was an Apple M5 Max.
//!
//! Build with `CUDA_COMPUTE_CAP=120` (native Blackwell sm_120).
//!
//! Weight layout — the candle loader wants `<root>/{diffusion_models,text_encoders,vae}/...`, which is
//! exactly the raw HF repo's `split_files/` tree (the loader also accepts a dir whose child is
//! `split_files/`). Point `ANIMA_DIR` at the snapshot dir (the parent of `split_files/`) OR at
//! `split_files/` itself.
//!
//! Setup (PowerShell):
//! ```text
//! $env:ANIMA_DIR="D:\.cache\huggingface\hub\models--circlestone-labs--Anima\snapshots\<hash>"
//! $env:ANIMA_OUT_DIR="D:\sceneworks-anima-validate"
//! # optional: ANIMA_VARIANT=base|aesthetic|turbo  ANIMA_W=1024 ANIMA_H=1024 ANIMA_STEPS=..
//! #           ANIMA_GUIDANCE=..  ANIMA_SEED=42  ANIMA_PROMPT="..."  ANIMA_NEG="..."
//! #           ANIMA_LORA=<path.safetensors>  ANIMA_LORA_KIND=lora|lokr  ANIMA_LORA_SCALE=1.0
//! cargo test -p sceneworks-worker --no-default-features --features backend-candle --release \
//!   anima_candle_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use gen_core::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};

use super::smoke_support::{env_or, image_mean, image_std, save_png, DEGENERATE_STD_FLOOR_DEFAULT};

/// Resolve the Anima variant → engine id + its shipped defaults (steps/guidance). Turbo is the merged
/// CFG-free student (guidance forced 1.0 inside the engine); base/aesthetic run true CFG.
fn resolve_variant(raw: &str) -> (&'static str, u32, f32) {
    match raw.trim().to_ascii_lowercase().as_str() {
        "base" => ("anima_base", 30, 4.5),
        "aesthetic" => ("anima_aesthetic", 30, 4.5),
        "turbo" => ("anima_turbo", 10, 1.0),
        other => panic!("ANIMA_VARIANT must be base|aesthetic|turbo, got {other:?}"),
    }
}

#[cfg(test)]
mod variant_tests {
    use super::resolve_variant;

    #[test]
    fn resolve_variant_maps_the_three_ids() {
        assert_eq!(resolve_variant("base").0, "anima_base");
        assert_eq!(resolve_variant("AESTHETIC").0, "anima_aesthetic");
        assert_eq!(resolve_variant(" turbo ").0, "anima_turbo");
    }

    #[test]
    #[should_panic(expected = "ANIMA_VARIANT must be base|aesthetic|turbo")]
    fn resolve_variant_rejects_unknown() {
        let _ = resolve_variant("preview");
    }
}

/// The optional LoRA/LoKr adapter from env (`ANIMA_LORA` path + `ANIMA_LORA_KIND` + `ANIMA_LORA_SCALE`).
/// Returns the built `AdapterSpec` and a filename-safe tag; `None` when `ANIMA_LORA` is unset/empty.
fn adapter_from_env() -> Option<(AdapterSpec, String)> {
    let path = std::env::var("ANIMA_LORA")
        .ok()
        .map(|p| p.trim().to_string());
    let path = path.filter(|p| !p.is_empty())?;
    let lora_path = PathBuf::from(&path);
    assert!(
        lora_path.is_file(),
        "ANIMA_LORA is set but not a file: {}",
        lora_path.display()
    );
    let kind = match env_or("ANIMA_LORA_KIND", "lora")
        .to_ascii_lowercase()
        .as_str()
    {
        "lora" => AdapterKind::Lora,
        "lokr" => AdapterKind::Lokr,
        other => panic!("ANIMA_LORA_KIND must be lora|lokr, got {other:?}"),
    };
    let scale: f32 = env_or("ANIMA_LORA_SCALE", "1.0")
        .parse()
        .expect("ANIMA_LORA_SCALE");
    let stem = lora_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("lora")
        .replace(['.', ' '], "_");
    let tag = format!("{stem}_{:?}", kind).to_ascii_lowercase();
    Some((AdapterSpec::new(lora_path, scale, kind), tag))
}

#[test]
#[ignore = "real-weight GPU smoke; needs the circlestone-labs/Anima split_files snapshot + a CUDA device (cap=120)"]
fn anima_candle_gpu_smoke() {
    let root = PathBuf::from(
        std::env::var("ANIMA_DIR")
            .expect("set $ANIMA_DIR to the Anima snapshot dir (parent of split_files/)")
            .trim(),
    );
    // Accept either the snapshot dir (child split_files/) or the split_files/ dir directly.
    let split = if root.join("diffusion_models").is_dir() {
        root.clone()
    } else if root.join("split_files").join("diffusion_models").is_dir() {
        root.join("split_files")
    } else {
        panic!(
            "ANIMA_DIR has no diffusion_models/ or split_files/diffusion_models/: {}",
            root.display()
        );
    };
    assert!(
        split
            .join("text_encoders/qwen_3_06b_base.safetensors")
            .is_file(),
        "missing shared Qwen3 TE under {}",
        split.display()
    );
    assert!(
        split.join("vae/qwen_image_vae.safetensors").is_file(),
        "missing shared Qwen-Image VAE under {}",
        split.display()
    );

    let out_dir = PathBuf::from(env_or("ANIMA_OUT_DIR", "/tmp/anima_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let variant_raw = env_or("ANIMA_VARIANT", "base");
    let (id, def_steps, def_guidance) = resolve_variant(&variant_raw);
    let steps: u32 = env_or("ANIMA_STEPS", &def_steps.to_string())
        .parse()
        .expect("ANIMA_STEPS");
    let w: u32 = env_or("ANIMA_W", "1024").parse().expect("ANIMA_W");
    let h: u32 = env_or("ANIMA_H", "1024").parse().expect("ANIMA_H");
    let guidance: f32 = env_or("ANIMA_GUIDANCE", &def_guidance.to_string())
        .parse()
        .expect("ANIMA_GUIDANCE");
    let seed: u64 = env_or("ANIMA_SEED", "42").parse().expect("ANIMA_SEED");
    let prompt = env_or(
        "ANIMA_PROMPT",
        "masterpiece, best quality, score_7, safe, 1girl, silver hair, blue eyes, kimono, \
         cherry blossoms, standing, looking at viewer, detailed background",
    );
    let negative = env_or(
        "ANIMA_NEG",
        "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, \
         jpeg artifacts, chromatic aberration",
    );

    // Point `WeightsSource::Dir` at the split_files root; the candle loader resolves the variant DiT
    // + shared TE/VAE from it (dense bf16 — the candle lane has no Windows path to a packed tier, since
    // the q4/q8 converter is macOS-only and Anima's NC posture means no packed tier is published).
    let mut spec = LoadSpec::new(WeightsSource::Dir(split.clone()));
    let adapter = adapter_from_env();
    if let Some((a, _)) = &adapter {
        spec = spec.with_adapters(vec![a.clone()]);
    }

    let adapter_tag = adapter
        .as_ref()
        .map(|(_, t)| format!("_{t}"))
        .unwrap_or_default();
    println!(
        "[smoke] loading {id} (bf16{}) from {} ...",
        adapter
            .as_ref()
            .map(|(_, t)| format!(", +{t}"))
            .unwrap_or_default(),
        split.display()
    );
    let generator = gen_core::load(id, &spec).unwrap_or_else(|e| panic!("load candle {id}: {e}"));

    // Turbo is the merged CFG-free student — its descriptor advertises `supports_negative_prompt: false`
    // and `supports_guidance: false`, so passing either is (correctly) rejected by `validate`. Only the
    // CFG variants (base/aesthetic) carry a negative prompt + guidance.
    let uses_cfg = id != "anima_turbo";
    let req = GenerationRequest {
        prompt: prompt.clone(),
        negative_prompt: uses_cfg.then(|| negative.clone()),
        width: w,
        height: h,
        count: 1,
        seed: Some(seed),
        steps: Some(steps),
        guidance: uses_cfg.then_some(guidance),
        ..Default::default()
    };
    println!(
        "[smoke] rendering {id} {w}x{h} @ {steps} steps, guidance {guidance}, seed {seed} ..."
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
        .unwrap_or_else(|e| panic!("{id} generate: {e}"));
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };

    let mean = image_mean(&image);
    let std = image_std(&image);
    let png = out_dir.join(format!("{id}_bf16_{w}x{h}_{steps}step{adapter_tag}.png"));
    save_png(&image, &png);
    println!(
        "[smoke] {id} {}x{} mean {:.2} std {:.2} -> {}",
        image.width,
        image.height,
        mean,
        std,
        png.display()
    );
    assert_eq!((image.width, image.height), (w, h));
    assert!(
        std > DEGENERATE_STD_FLOOR_DEFAULT,
        "{id} render looks degenerate (std {std:.2}) — possible NaN / all-black decode \
         (check CUDA_COMPUTE_CAP=120)"
    );
    println!(
        "[smoke] DONE: {id} bf16 render coherent at {steps} steps (eyeball {})",
        png.display()
    );
}

/// sc-10676 worker-lane E2E: drive the ACTUAL worker weight-resolution + dense-load + render for all
/// three Anima variants on real CUDA — the piece the engine-seam [`anima_candle_gpu_smoke`] above does
/// NOT cover. Unlike that smoke (which hands the engine a hand-built `split_files/` path), this calls
/// the WORKER's [`crate::image_jobs::resolve_weights_dir`] the way `generate_candle_stream` does, asserts
/// it returns the dense `split_files/` dir off-Mac (NOT a converted q4/q8/bf16 tier), then loads DENSE
/// (Quant `None` — the decision `generate_candle_stream` forces for Anima; passing a quant would hit the
/// loader's dense-DiT reject) and renders a coherent image. Point the HF cache env at the real Anima
/// snapshot — no `ANIMA_DIR` (the resolver finds it):
/// ```text
/// $env:HF_HUB_CACHE="D:\.cache\huggingface\hub"
/// $env:ANIMA_OUT_DIR="D:\sceneworks-anima-validate"
/// cargo test -p sceneworks-worker --no-default-features --features backend-candle --release \
///   anima_worker_lane_gpu_smoke -- --ignored --nocapture
/// ```
#[test]
#[ignore = "real-weight worker-lane GPU smoke; needs the circlestone-labs/Anima snapshot in the HF cache \
            (HF_HUB_CACHE/HF_HOME) + a CUDA device (cap=120)"]
fn anima_worker_lane_gpu_smoke() {
    use crate::image_jobs::resolve_weights_dir;
    use crate::settings::Settings;
    use gen_core::GenerationRequest;
    use sceneworks_core::image_request::ImageRequest;
    use serde_json::json;

    let out_dir = PathBuf::from(env_or("ANIMA_OUT_DIR", "/tmp/anima_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let settings = Settings::from_env();
    let prompt = "masterpiece, best quality, 1girl, silver hair, blue eyes, kimono, \
                  cherry blossoms, standing, looking at viewer, detailed background";
    let negative = "worst quality, low quality, blurry, jpeg artifacts";

    // (id, steps, guidance, uses_cfg) — turbo is the merged CFG-free student (no negative / guidance).
    for (id, steps, guidance, uses_cfg) in [
        ("anima_base", 30u32, 4.5f32, true),
        ("anima_aesthetic", 30, 4.5, true),
        ("anima_turbo", 10, 1.0, false),
    ] {
        // A minimal job payload — resolve_weights_dir keys off `model` (+ absent modelPath), the exact
        // shape the router hands `generate_candle_stream`. No `mlxQuantize`: the web UI never sends it.
        let payload = json!({ "model": id, "prompt": prompt, "width": 1024, "height": 1024 });
        let request = ImageRequest::from_payload(payload.as_object().unwrap());

        // THE worker resolver (my sc-10676 change): off-Mac Anima → the dense `split_files/` tree, never
        // a converted tier (there is none off-Mac). This is the seam the engine-only smoke skips.
        let dir = resolve_weights_dir(&request, &settings)
            .expect("resolve_weights_dir")
            .unwrap_or_else(|| {
                panic!("{id}: Anima snapshot not in the HF cache — set HF_HUB_CACHE/HF_HOME")
            });
        assert!(
            dir.ends_with("split_files") && dir.join("diffusion_models").is_dir(),
            "{id}: worker must resolve the dense split_files/ dir, got {}",
            dir.display()
        );
        println!(
            "[worker-smoke] {id}: resolve_weights_dir -> {}",
            dir.display()
        );

        // DENSE load (Quant None) — exactly what generate_candle_stream forces for Anima.
        let spec = LoadSpec::new(WeightsSource::Dir(dir));
        let generator = gen_core::load(id, &spec).unwrap_or_else(|e| panic!("load {id}: {e}"));
        let req = GenerationRequest {
            prompt: prompt.to_string(),
            negative_prompt: uses_cfg.then(|| negative.to_string()),
            width: 1024,
            height: 1024,
            count: 1,
            seed: Some(42),
            steps: Some(steps),
            guidance: uses_cfg.then_some(guidance),
            ..Default::default()
        };
        println!("[worker-smoke] {id}: rendering 1024x1024 @ {steps} steps ...");
        let output = generator
            .generate(&req, &mut |_| {})
            .unwrap_or_else(|e| panic!("{id} generate: {e}"));
        let image = match output {
            GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
            other => panic!("expected Images output, got {other:?}"),
        };
        let std = image_std(&image);
        let png = out_dir.join(format!("{id}_worker_lane_1024.png"));
        save_png(&image, &png);
        println!(
            "[worker-smoke] {id}: {}x{} mean {:.2} std {:.2} -> {}",
            image.width,
            image.height,
            image_mean(&image),
            std,
            png.display()
        );
        assert_eq!((image.width, image.height), (1024, 1024));
        assert!(
            std > DEGENERATE_STD_FLOOR_DEFAULT,
            "{id} render looks degenerate (std {std:.2})"
        );
    }
    println!("[worker-smoke] DONE: all three Anima variants rendered through the worker resolver + dense lane");
}
