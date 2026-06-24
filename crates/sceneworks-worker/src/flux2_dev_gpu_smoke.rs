//! Local real-weight GPU smoke for the candle FLUX.2-dev worker lane (epic 6564 sc-7458, the worker
//! half of the sc-7457 provider). `#[ignore]`d — run by hand on the RTX PRO 6000. It drives the real
//! candle FLUX.2-dev engine via `gen_core::load("flux2_dev")` with a **Q4** `LoadSpec` — the exact
//! runtime seam `generate_candle_stream` uses (minus the API/job plumbing) once the router routes
//! `flux2_dev` to candle off-Mac. The 32B doesn't fit the GPU dense, so the dev quant path stages the
//! dense diffusers snapshot in system RAM and quantizes each projection onto the GPU at load
//! (candle-gen-flux2 sc-7457) — this is the end-to-end worker-lane validation backing the routing wire.
//!
//! Build with `CUDA_COMPUTE_CAP=120` (native Blackwell sm_120): the cap=80 PTX baseline JIT-no-ops
//! candle's CUDA quantized matmul on sm_120 (sc-7457 black-image root cause → sc-7544 packaging).
//!
//! Setup (PowerShell; point at the dense FLUX.2-dev diffusers snapshot — `transformer/ text_encoder/
//! vae/ tokenizer/ model_index.json`):
//! ```text
//! $env:FLUX2_DEV_DIR="D:\models\FLUX.2-dev"
//! $env:FLUX2_DEV_OUT_DIR="D:\sceneworks-sampler-validate\flux2-dev"
//! # optional: FLUX2_DEV_STEPS=28 FLUX2_DEV_W=1024 FLUX2_DEV_H=1024 FLUX2_DEV_GUIDANCE=4.0
//! #           FLUX2_DEV_QUANT=q4|q8  FLUX2_DEV_PROMPT="..."
//! cargo test -p sceneworks-worker --features backend-candle --release flux2_dev_candle_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use gen_core::{GenerationOutput, GenerationRequest, Image, LoadSpec, Quant, WeightsSource};

fn env_path(key: &str) -> PathBuf {
    // Trim: a cmd `set VAR=value && ...` keeps the trailing space before `&&`.
    PathBuf::from(
        std::env::var(key)
            .unwrap_or_else(|_| panic!("set ${key}"))
            .trim(),
    )
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .map(|v| v.trim().to_string())
        .unwrap_or_else(|_| default.to_string())
}

/// Mean per-pixel std-dev across the RGB channels — a cheap "is the image non-degenerate" check. The
/// sc-7457 dev-quant bug (CUDA-Q4 matmul no-op at cap=80) produced an all-black decode whose std
/// collapses toward 0; this guards that degenerate floor (the real quality call is the saved-PNG eyeball).
fn image_std(img: &Image) -> f64 {
    let n = img.pixels.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = img.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let var = img
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    var.sqrt()
}

fn save_png(img: &Image, path: &Path) {
    image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .expect("rgb buffer")
        .save(path)
        .unwrap_or_else(|e| panic!("save {}: {e}", path.display()));
}

#[test]
#[ignore = "real-weight GPU smoke; needs the dense FLUX.2-dev diffusers snapshot + a CUDA device (cap=120)"]
fn flux2_dev_candle_gpu_smoke() {
    let weights_dir = env_path("FLUX2_DEV_DIR");
    assert!(
        weights_dir.join("model_index.json").is_file(),
        "FLUX2_DEV_DIR must point at the dense FLUX.2-dev diffusers snapshot (model_index.json missing): {}",
        weights_dir.display()
    );
    let out_dir = PathBuf::from(env_or("FLUX2_DEV_OUT_DIR", "flux2-dev-out"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let steps: u32 = env_or("FLUX2_DEV_STEPS", "28")
        .parse()
        .expect("FLUX2_DEV_STEPS");
    let w: u32 = env_or("FLUX2_DEV_W", "1024").parse().expect("FLUX2_DEV_W");
    let h: u32 = env_or("FLUX2_DEV_H", "1024").parse().expect("FLUX2_DEV_H");
    let guidance: f32 = env_or("FLUX2_DEV_GUIDANCE", "4.0")
        .parse()
        .expect("FLUX2_DEV_GUIDANCE");
    // The shipped worker forces Q4 (manifest `mlx.quantize: 4` → `resolve_quant`); allow Q8 for an
    // optional contrast run. dev advertises only Q4/Q8 — dense doesn't fit the GPU.
    let quant = match env_or("FLUX2_DEV_QUANT", "q4").as_str() {
        "q8" | "Q8" => Quant::Q8,
        _ => Quant::Q4,
    };
    let prompt = env_or(
        "FLUX2_DEV_PROMPT",
        "a rusty robot holding a lit candle in a dark workshop, cinematic, sharp focus",
    );

    // Same seam as `generate_candle_stream`: a registry load of the candle `flux2_dev` engine + a
    // Q4 `LoadSpec` pointed at the dense diffusers snapshot. The dev quant path CPU-stages the dense
    // weights and quantizes each projection onto the GPU — the 32B never lands on the GPU dense.
    println!(
        "[smoke] loading flux2_dev ({quant:?}) from {} ...",
        weights_dir.display()
    );
    let spec = LoadSpec::new(WeightsSource::Dir(weights_dir.clone())).with_quant(quant);
    let generator = gen_core::load("flux2_dev", &spec).expect("load candle flux2_dev provider");

    let req = GenerationRequest {
        prompt: prompt.clone(),
        width: w,
        height: h,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        // dev is guidance-distilled (embedded scalar, single forward — no negative pass).
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
        .expect("flux2_dev generate");
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };

    let std = image_std(&image);
    let png = out_dir.join(format!(
        "flux2_dev_{}_{steps}step.png",
        env_or("FLUX2_DEV_QUANT", "q4")
    ));
    save_png(&image, &png);
    println!(
        "[smoke] flux2_dev {}x{} std {:.2} -> {}",
        image.width,
        image.height,
        std,
        png.display()
    );
    assert_eq!((image.width, image.height), (w, h));
    assert!(
        std > 5.0,
        "flux2_dev render looks degenerate (std {std:.2}) — possible NaN / all-black decode \
         (check CUDA_COMPUTE_CAP=120: cap=80 JIT-no-ops the quantized matmul on sm_120, sc-7457)"
    );
    println!("[smoke] DONE: flux2_dev {quant:?} render coherent at {steps} steps");
}
