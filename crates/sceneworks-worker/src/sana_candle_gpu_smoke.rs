//! Local real-weight GPU smoke for the candle **SANA 1600M** worker lane (epic 8485, sc-11780 — the
//! hardware-gated acceptance for the candle half) + the **SANA-Sprint** sibling (sc-11781). `#[ignore]`d —
//! run by hand on the RTX PRO 6000 (Blackwell sm_120). Each drives the ACTUAL worker weight-resolution +
//! `gen_core::load(<id>)` + render — the exact runtime seam `generate_candle_stream` uses (minus the
//! API/job plumbing) once the router routes SANA / SANA-Sprint to candle off-Mac.
//!
//! This is the "the candle SANA lane has actually been RUN on real CUDA" evidence that backs flipping
//! `macOnly: false` / `candle_routed = true`: the candle-gen-sana port (candle-gen #495) landed
//! CPU-validated against framework-independent goldens, but the worker lane had never been generated on a
//! GPU (the candle-gen dev machine was Apple Silicon).
//!
//! Unlike an engine-seam smoke that hands the engine a hand-built path, this calls the WORKER's
//! [`crate::image_jobs::resolve_weights_dir`] the way `generate_candle_stream` does, asserts it returns
//! the WHOLE `Efficient-Large-Model/Sana_1600M_1024px_diffusers` diffusers snapshot ROOT off-Mac (NOT the
//! MLX-packed turnkey, NOT a q4/q8/bf16 tier subdir), then loads DENSE and renders a coherent true-CFG
//! image at 1024².
//!
//! Build with `CUDA_COMPUTE_CAP=120` (native Blackwell sm_120). Point the HF cache env at the real
//! snapshot — no explicit weights path (the resolver finds it):
//! ```text
//! $env:HF_HUB_CACHE="E:\huggingface\hub"
//! $env:SANA_OUT_DIR="D:\sceneworks-sana-validate"
//! # optional: SANA_STEPS=20  SANA_GUIDANCE=4.5  SANA_SEED=42  SANA_PROMPT="..."  SANA_NEG="..."
//! cargo test -p sceneworks-worker --no-default-features --features backend-candle --release \
//!   sana_worker_lane_gpu_smoke -- --ignored --nocapture
//! # SANA-Sprint (CFG-free, 1–4 step; optional SANA_SPRINT_STEPS=2 SANA_SPRINT_GUIDANCE=4.5):
//! cargo test -p sceneworks-worker --no-default-features --features backend-candle --release \
//!   sana_sprint_worker_lane_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use gen_core::{GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};

use super::smoke_support::{env_or, image_mean, image_std, save_png, DEGENERATE_STD_FLOOR_DEFAULT};

/// sc-11780 worker-lane E2E: drive the ACTUAL worker weight-resolution + dense-load + render for the
/// candle SANA 1600M lane on real CUDA. Calls the WORKER's [`crate::image_jobs::resolve_weights_dir`]
/// the way `generate_candle_stream` does, asserts it resolves the whole `Efficient-Large-Model/
/// Sana_1600M_1024px_diffusers` diffusers snapshot ROOT off-Mac (the candle-sana branch — NOT the MLX
/// turnkey the macOS path loads, NOT a tier subdir), then loads DENSE and renders a coherent true-CFG
/// 1024² image. SANA is a true-CFG flow-match model (20 steps / guidance 4.5 + negative prompt).
#[test]
#[ignore = "real-weight worker-lane GPU smoke; needs the Efficient-Large-Model/Sana_1600M_1024px_diffusers \
            snapshot in the HF cache (HF_HUB_CACHE/HF_HOME) + a CUDA device (cap=120)"]
fn sana_worker_lane_gpu_smoke() {
    use crate::image_jobs::resolve_weights_dir;
    use crate::settings::Settings;
    use sceneworks_core::image_request::ImageRequest;
    use serde_json::json;

    let out_dir = PathBuf::from(env_or("SANA_OUT_DIR", "/tmp/sana_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let settings = Settings::from_env();

    let steps: u32 = env_or("SANA_STEPS", "20").parse().expect("SANA_STEPS");
    let guidance: f32 = env_or("SANA_GUIDANCE", "4.5")
        .parse()
        .expect("SANA_GUIDANCE");
    let seed: u64 = env_or("SANA_SEED", "42").parse().expect("SANA_SEED");
    let w: u32 = env_or("SANA_W", "1024").parse().expect("SANA_W");
    let h: u32 = env_or("SANA_H", "1024").parse().expect("SANA_H");
    let prompt = env_or(
        "SANA_PROMPT",
        "a photorealistic red fox sitting in a sunlit autumn forest, sharp focus, detailed fur, \
         golden hour lighting",
    );
    let negative = env_or("SANA_NEG", "blurry, low quality, jpeg artifacts, deformed");

    // A minimal job payload — resolve_weights_dir keys off `model` (+ absent modelPath), the exact shape
    // the router hands `generate_candle_stream`. No `mlxQuantize`: the candle SANA base path is dense.
    let payload = json!({ "model": "sana_1600m", "prompt": prompt, "width": w, "height": h });
    let request = ImageRequest::from_payload(payload.as_object().unwrap());

    // THE worker resolver (the sc-11780 change): off-Mac SANA → the whole diffusers snapshot ROOT, never
    // the MLX turnkey and never a tier subdir. This is the seam an engine-only smoke skips.
    let dir = resolve_weights_dir(&request, &settings)
        .expect("resolve_weights_dir")
        .unwrap_or_else(|| {
            panic!(
                "sana_1600m: Efficient-Large-Model/Sana_1600M_1024px_diffusers snapshot not in the HF \
                 cache — set HF_HUB_CACHE/HF_HOME"
            )
        });
    assert!(
        dir.join("transformer").is_dir()
            && dir.join("vae").is_dir()
            && dir.join("text_encoder").is_dir(),
        "sana_1600m: worker must resolve the whole diffusers snapshot root (transformer/ + vae/ + \
         text_encoder/), got {}",
        dir.display()
    );
    assert!(
        !dir.ends_with("q4") && !dir.ends_with("q8") && !dir.ends_with("bf16"),
        "sana_1600m candle must NOT descend into a tier subdir, got {}",
        dir.display()
    );
    println!(
        "[worker-smoke] sana_1600m: resolve_weights_dir -> {}",
        dir.display()
    );

    // DENSE load (no quant) — the candle SANA base path advertises no packed tier off-Mac.
    let spec = LoadSpec::new(WeightsSource::Dir(dir));
    let generator =
        gen_core::load("sana_1600m", &spec).unwrap_or_else(|e| panic!("load sana_1600m: {e}"));
    assert_eq!(
        generator.descriptor().backend,
        "candle",
        "sana_1600m must resolve the candle generator off-Mac"
    );

    let req = GenerationRequest {
        prompt: prompt.clone(),
        negative_prompt: Some(negative.clone()),
        width: w,
        height: h,
        count: 1,
        seed: Some(seed),
        steps: Some(steps),
        guidance: Some(guidance),
        ..Default::default()
    };
    println!(
        "[worker-smoke] sana_1600m: rendering {w}x{h} @ {steps} steps, guidance {guidance}, seed {seed} ..."
    );
    let started = std::time::Instant::now();
    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |p| {
            let s = format!("{p:?}");
            if s != last {
                println!("[progress] {s}");
                last = s;
            }
        })
        .unwrap_or_else(|e| panic!("sana_1600m generate: {e}"));
    let elapsed = started.elapsed();
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };

    let mean = image_mean(&image);
    let std = image_std(&image);
    let png = out_dir.join(format!("sana_1600m_candle_{w}x{h}_{steps}step.png"));
    save_png(&image, &png);
    println!(
        "[worker-smoke] sana_1600m: {}x{} mean {:.2} std {:.2} in {:.1}s -> {}",
        image.width,
        image.height,
        mean,
        std,
        elapsed.as_secs_f32(),
        png.display()
    );
    assert_eq!((image.width, image.height), (w, h));
    assert!(
        mean.is_finite() && std.is_finite(),
        "sana_1600m render produced non-finite stats (mean {mean}, std {std}) — NaN in the pipeline"
    );
    assert!(
        std > DEGENERATE_STD_FLOOR_DEFAULT,
        "sana_1600m render looks degenerate (std {std:.2}) — possible NaN / all-black decode \
         (check CUDA_COMPUTE_CAP=120)"
    );
    println!(
        "[worker-smoke] DONE: sana_1600m candle true-CFG render coherent at {steps} steps (eyeball {})",
        png.display()
    );
}

/// sc-11781 worker-lane E2E: the SANA-**Sprint** sibling of `sana_worker_lane_gpu_smoke`. Drives the
/// ACTUAL worker weight-resolution + dense-load + render for the candle SANA-Sprint lane on real CUDA.
/// Calls the WORKER's [`crate::image_jobs::resolve_weights_dir`] the way `generate_candle_stream` does,
/// asserts it resolves the whole `Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers` diffusers
/// snapshot ROOT off-Mac (the candle-sana branch — NOT the MLX turnkey, NOT a tier subdir), then loads
/// DENSE and renders a coherent image. Sprint is CFG-FREE (guidance embedded, no negative-prompt second
/// pass) and runs in the 1–4 step SCM/TrigFlow band — this smoke drives a 2-step render.
#[test]
#[ignore = "real-weight worker-lane GPU smoke; needs the Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers \
            snapshot in the HF cache (HF_HUB_CACHE/HF_HOME) + a CUDA device (cap=120)"]
fn sana_sprint_worker_lane_gpu_smoke() {
    use crate::image_jobs::resolve_weights_dir;
    use crate::settings::Settings;
    use sceneworks_core::image_request::ImageRequest;
    use serde_json::json;

    let out_dir = PathBuf::from(env_or("SANA_OUT_DIR", "/tmp/sana_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let settings = Settings::from_env();

    // Sprint operating band is 1–4 steps; default the smoke to 2. Guidance is the embedded CFG-free
    // scalar (NOT a true-CFG second pass), so there is no negative prompt.
    let steps: u32 = env_or("SANA_SPRINT_STEPS", "2")
        .parse()
        .expect("SANA_SPRINT_STEPS");
    let guidance: f32 = env_or("SANA_SPRINT_GUIDANCE", "4.5")
        .parse()
        .expect("SANA_SPRINT_GUIDANCE");
    let seed: u64 = env_or("SANA_SEED", "42").parse().expect("SANA_SEED");
    let w: u32 = env_or("SANA_W", "1024").parse().expect("SANA_W");
    let h: u32 = env_or("SANA_H", "1024").parse().expect("SANA_H");
    let prompt = env_or(
        "SANA_PROMPT",
        "a photorealistic red fox sitting in a sunlit autumn forest, sharp focus, detailed fur, \
         golden hour lighting",
    );

    // A minimal job payload — resolve_weights_dir keys off `model` (+ absent modelPath), the exact shape
    // the router hands `generate_candle_stream`. No `mlxQuantize`: the candle Sprint path is dense.
    let payload =
        json!({ "model": "sana_sprint_1600m", "prompt": prompt, "width": w, "height": h });
    let request = ImageRequest::from_payload(payload.as_object().unwrap());

    // THE worker resolver (the sc-11781 change): off-Mac SANA-Sprint → the whole diffusers snapshot ROOT,
    // never the MLX turnkey and never a tier subdir.
    let dir = resolve_weights_dir(&request, &settings)
        .expect("resolve_weights_dir")
        .unwrap_or_else(|| {
            panic!(
                "sana_sprint_1600m: Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers snapshot \
                 not in the HF cache — set HF_HUB_CACHE/HF_HOME"
            )
        });
    assert!(
        dir.join("transformer").is_dir()
            && dir.join("vae").is_dir()
            && dir.join("text_encoder").is_dir(),
        "sana_sprint_1600m: worker must resolve the whole diffusers snapshot root (transformer/ + vae/ \
         + text_encoder/), got {}",
        dir.display()
    );
    assert!(
        !dir.ends_with("q4") && !dir.ends_with("q8") && !dir.ends_with("bf16"),
        "sana_sprint_1600m candle must NOT descend into a tier subdir, got {}",
        dir.display()
    );
    println!(
        "[worker-smoke] sana_sprint_1600m: resolve_weights_dir -> {}",
        dir.display()
    );

    // DENSE load (no quant) — the candle Sprint path advertises no packed tier off-Mac.
    let spec = LoadSpec::new(WeightsSource::Dir(dir));
    let generator = gen_core::load("sana_sprint_1600m", &spec)
        .unwrap_or_else(|e| panic!("load sana_sprint_1600m: {e}"));
    assert_eq!(
        generator.descriptor().backend,
        "candle",
        "sana_sprint_1600m must resolve the candle generator off-Mac"
    );

    // CFG-free: no negative prompt (the guidance scalar is a trunk embedding).
    let req = GenerationRequest {
        prompt: prompt.clone(),
        negative_prompt: None,
        width: w,
        height: h,
        count: 1,
        seed: Some(seed),
        steps: Some(steps),
        guidance: Some(guidance),
        ..Default::default()
    };
    println!(
        "[worker-smoke] sana_sprint_1600m: rendering {w}x{h} @ {steps} steps (CFG-free), guidance \
         {guidance}, seed {seed} ..."
    );
    let started = std::time::Instant::now();
    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |p| {
            let s = format!("{p:?}");
            if s != last {
                println!("[progress] {s}");
                last = s;
            }
        })
        .unwrap_or_else(|e| panic!("sana_sprint_1600m generate: {e}"));
    let elapsed = started.elapsed();
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };

    let mean = image_mean(&image);
    let std = image_std(&image);
    let png = out_dir.join(format!("sana_sprint_1600m_candle_{w}x{h}_{steps}step.png"));
    save_png(&image, &png);
    println!(
        "[worker-smoke] sana_sprint_1600m: {}x{} mean {:.2} std {:.2} in {:.1}s -> {}",
        image.width,
        image.height,
        mean,
        std,
        elapsed.as_secs_f32(),
        png.display()
    );
    assert_eq!((image.width, image.height), (w, h));
    assert!(
        mean.is_finite() && std.is_finite(),
        "sana_sprint_1600m render produced non-finite stats (mean {mean}, std {std}) — NaN in the pipeline"
    );
    assert!(
        std > DEGENERATE_STD_FLOOR_DEFAULT,
        "sana_sprint_1600m render looks degenerate (std {std:.2}) — possible NaN / all-black decode \
         (check CUDA_COMPUTE_CAP=120)"
    );
    println!(
        "[worker-smoke] DONE: sana_sprint_1600m candle CFG-free render coherent at {steps} steps \
         (eyeball {})",
        png.display()
    );
}
