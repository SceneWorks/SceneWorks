//! Local real-weight MLX smoke for the SDXL base 1.0 **Q8** worker lane (sc-8746, epic 8506 Group-B).
//! `#[ignore]`d — run by hand on an Apple-Silicon Mac. It drives the real native-MLX `sdxl` engine via
//! `gen_core::load("sdxl")` with a **Q8** `LoadSpec` pointed at the packed `q8/` turnkey subdir — the
//! exact runtime seam `generate_stream` uses (minus the API/job plumbing) when the router routes an
//! `sdxl` MLX job (`standard_tier_subdir` → the `q8/` subdir; `resolve_quant` → `Quant::Q8`).
//!
//! Purpose: this is the on-device evidence that CLOSES the stale sc-1975 Q8-on-SDXL loop. sc-1975
//! recorded that **Apple's vendored `mlx-examples` Python Q8 recipe** produced an ALL-ZERO (degenerate)
//! decode on SDXL base 1.0, so the manifest deferred a Q8 default. That vendored in-process MLX-SDXL
//! adapter was RETIRED by sc-3060 (the Python worker always routes torch for SDXL now); MLX-SDXL is the
//! Rust `mlx-gen-sdxl` path, whose Q4/Q8 quantization was FIXED + MERGED by mlx-gen PR #62 (sc-2641,
//! "SDXL Q4/Q8 quantization — both hold parity on base-1.0"). This smoke asserts the Q8 render is (a)
//! non-degenerate (per-pixel std well above the degenerate floor) and (b) SPECIFICALLY NOT all-zero —
//! the exact failure mode the original Apple recipe exhibited.
//!
//! Setup — point at the published turnkey `SceneWorks/sdxl-base-mlx` (the worker default). With the q8
//! tier already in the HF cache, no env is needed: the smoke auto-resolves the cached snapshot's `q8/`
//! subdir (the same selection `image_jobs::base::standard_tier_subdir` makes for `mlxQuantize: 8`).
//! Override `SDXL_Q8_DIR` to point at a snapshot root or a `q8/`-bearing dir directly.
//! ```text
//! # optional: SDXL_Q8_DIR=/path/to/sdxl-base-mlx  (root containing q8/, or the q8/ dir itself)
//! # optional: SDXL_STEPS=30 SDXL_W=768 SDXL_H=768 SDXL_PROMPT="..." SDXL_OUT_DIR=/tmp/sdxl_q8_smoke
//! cargo test -p sceneworks-worker --release sdxl_base_q8_mlx_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use gen_core::{GenerationOutput, GenerationRequest, Image, LoadSpec, Quant, WeightsSource};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

/// The engine-complete packed subdir to load: mirror `image_jobs::base::standard_tier_subdir`'s q8
/// selection — prefer `<root>/q8` (SDXL-family turnkeys pack the backbone under `unet/`), else `root`
/// itself if it already *is* a q8 root. Errors loud if neither resolves so a half-download surfaces as
/// a clear failure rather than a confusing engine load error.
fn resolve_q8_dir(root: &Path) -> PathBuf {
    let is_engine_root = |d: &Path| d.join("unet/diffusion_pytorch_model.safetensors").is_file();
    let q8 = root.join("q8");
    if is_engine_root(&q8) {
        return q8;
    }
    assert!(
        is_engine_root(root),
        "SDXL_Q8_DIR must point at the turnkey root (containing q8/) or a q8/ dir with a packed \
         unet/diffusion_pytorch_model.safetensors; neither found under {}",
        root.display()
    );
    root.to_path_buf()
}

/// Auto-discover the cached `SceneWorks/sdxl-base-mlx` turnkey snapshot in the HF hub cache, returning
/// the snapshot whose `q8/` subdir carries the packed UNet. `None` if the q8 tier hasn't been pulled
/// (the smoke then panics with the `SDXL_Q8_DIR` hint).
fn cached_turnkey_root() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--SceneWorks--sdxl-base-mlx/snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let dir = e.path();
            dir.join("q8/unet/diffusion_pytorch_model.safetensors")
                .is_file()
                .then_some(dir)
        })
}

/// Per-pixel mean over the RGB buffer — used both as the "is it black?" floor check and reported for
/// the record. The original Apple Q8 bug produced an ALL-ZERO decode, so a near-zero mean is the smoking
/// gun; a coherent render sits well inside the mid-tones.
fn image_mean(img: &Image) -> f64 {
    let n = img.pixels.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    img.pixels.iter().map(|&p| p as f64).sum::<f64>() / n
}

/// Mean per-pixel std-dev across the RGB channels — a cheap "is the image non-degenerate" check. A NaN /
/// all-black / flat decode collapses the std toward 0; this guards that degenerate floor. The real
/// quality call is the saved-PNG eyeball.
fn image_std(img: &Image) -> f64 {
    let n = img.pixels.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = image_mean(img);
    let var = img
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    var.sqrt()
}

/// Whether EVERY pixel byte is exactly 0 — the precise degenerate signature of the retired Apple Q8
/// recipe (sc-1975). A coherent SDXL render can never be all-zero.
fn is_all_zero(img: &Image) -> bool {
    !img.pixels.is_empty() && img.pixels.iter().all(|&p| p == 0)
}

fn save_png(img: &Image, path: &Path) {
    image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .expect("rgb buffer")
        .save(path)
        .unwrap_or_else(|e| panic!("save {}: {e}", path.display()));
}

#[test]
#[ignore = "real-weight MLX smoke; needs the SceneWorks/sdxl-base-mlx q8 turnkey cached + an Apple-Silicon Mac"]
fn sdxl_base_q8_mlx_gpu_smoke() {
    let root = match std::env::var("SDXL_Q8_DIR") {
        Ok(p) if !p.trim().is_empty() => PathBuf::from(p.trim()),
        _ => cached_turnkey_root().unwrap_or_else(|| {
            panic!(
                "no cached SceneWorks/sdxl-base-mlx q8 turnkey found; download it via the manifest \
                 (`hf download SceneWorks/sdxl-base-mlx --include 'q8/*'`) or set SDXL_Q8_DIR to the \
                 turnkey root (containing q8/)"
            )
        }),
    };
    let q8_dir = resolve_q8_dir(&root);

    let out_dir = PathBuf::from(env_or("SDXL_OUT_DIR", "/tmp/sdxl_q8_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let steps: u32 = env_or("SDXL_STEPS", "30").parse().expect("SDXL_STEPS");
    let w: u32 = env_or("SDXL_W", "768").parse().expect("SDXL_W");
    let h: u32 = env_or("SDXL_H", "768").parse().expect("SDXL_H");
    let prompt = env_or(
        "SDXL_PROMPT",
        "a photorealistic portrait of a red fox sitting in a sunlit autumn forest, sharp focus, \
         shallow depth of field",
    );

    // Same seam as the worker's MLX image path: a registry load of the `sdxl` generator (the worker-crate
    // force-link anchor `use mlx_gen_sdxl as _;` keeps it registered) + a Q8 `LoadSpec` pointed at the
    // packed q8 turnkey subdir. The packed weights auto-detect their quant, so `with_quant(Q8)` matches
    // the manifest's `mlx.quantize: 8` tier.
    println!("[smoke] loading sdxl (Q8) from {} ...", q8_dir.display());
    let spec = LoadSpec::new(WeightsSource::Dir(q8_dir.clone())).with_quant(Quant::Q8);
    let generator = gen_core::load("sdxl", &spec).expect("load mlx sdxl generator");

    // SDXL base 1.0: real CFG (negative prompt + guidance 7.0), EulerDiscrete default schedule.
    let req = GenerationRequest {
        prompt: prompt.clone(),
        negative_prompt: Some("blurry, low quality, deformed, watermark".to_owned()),
        width: w,
        height: h,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        guidance: Some(7.0),
        ..Default::default()
    };
    println!("[smoke] rendering {w}x{h} @ {steps} steps (CFG 7.0) ...");
    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |p| {
            let s = format!("{p:?}");
            if s != last {
                println!("[progress] {s}");
                last = s;
            }
        })
        .expect("sdxl generate");
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };

    let mean = image_mean(&image);
    let std = image_std(&image);
    let all_zero = is_all_zero(&image);
    let png = out_dir.join(format!("sdxl_base_q8_{steps}step.png"));
    save_png(&image, &png);
    println!(
        "[smoke] sdxl base Q8 {}x{} mean {:.2} std {:.2} all_zero={} -> {}",
        image.width,
        image.height,
        mean,
        std,
        all_zero,
        png.display()
    );
    assert_eq!(
        (image.width, image.height),
        (w, h),
        "engine returned the wrong dimensions"
    );
    // The original Apple Q8 recipe (sc-1975) produced an ALL-ZERO decode. Assert the fixed mlx-gen path
    // (sc-2641) specifically does NOT reproduce that signature.
    assert!(
        !all_zero,
        "sdxl base Q8 decode is ALL-ZERO — the retired Apple recipe's exact failure (sc-1975); the \
         mlx-gen Q8 path (sc-2641) must not reproduce it"
    );
    assert!(
        std > 20.0,
        "sdxl base Q8 render looks degenerate (std {std:.2}) — possible NaN / all-black / flat decode"
    );
    println!(
        "[smoke] DONE: sdxl base Q8 render coherent (mean {mean:.2}, std {std:.2}, NOT all-zero) \
         at {steps} steps through the worker lane"
    );
}
