//! Local real-weight GPU smoke for the candle **Z-Image + PiD super-resolving decode** (sc-8033,
//! epic 7840). `#[ignore]`d — run by hand on the RTX PRO 6000. Proves the end-to-end claim sc-8033
//! exists to close: `z_image_turbo` is engine-ready for PiD but had never actually been *run* through
//! the decoder.
//!
//! The routing is already wired — `image_jobs/pid.rs` maps `z_image_turbo` / `z_image_edit` →
//! backbone `"flux"` → the `SceneWorks/pid-flux` checkpoint (Z-Image is the **FLUX.1** VAE latent
//! space; PiD's `zimage`/`zimage-turbo` tags alias the flux checkpoint — the 2026-06-24 epic scope
//! correction, so there is NO dedicated `pid_zimage`), and the manifest gives both Z-Image models
//! `ui.pid.checkpointId = "pid_flux"`. This smoke drives the exact generic candle t2i seam
//! `generate_candle_stream` uses (sc-9727): `gen_core::load("z_image_turbo", spec.with_pid(pid_flux,
//! gemma))` + `GenerationRequest.use_pid`, decoding the SAME Z-Image latent through the native VAE
//! (render size) then the `pid_flux` student (4× → 2K/4K).
//!
//! Setup (PowerShell; the upstream Z-Image-Turbo diffusers snapshot + the NC `pid-flux` checkpoint +
//! gemma-2-2b-it are all in the HF cache):
//! ```text
//! $H="D:\.cache\huggingface\hub"
//! $env:ZIMAGE_PID_BASE="$H\models--Tongyi-MAI--Z-Image-Turbo\snapshots\f332072aa78be7aecdf3ee76d5c247082da564a6"
//! $env:ZIMAGE_PID_CKPT="$H\models--SceneWorks--pid-flux\snapshots\<rev>\pid_flux_2kto4k.safetensors"
//! $env:ZIMAGE_PID_GEMMA="$H\models--SceneWorks--gemma-2-2b-it\snapshots\684c553b5b41a1c835989d89f62f585e6269a7de"
//! $env:ZIMAGE_PID_OUT="D:\sceneworks-pid-validate\zimage"
//! # optional: ZIMAGE_PID_STEPS=8 ZIMAGE_PID_SIZE=1024 (render side; PiD output is 4x) ZIMAGE_PID_PROMPT="..."
//! cargo test -p sceneworks-worker --features backend-candle --release zimage_pid_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use gen_core::{GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};

fn env_path(key: &str) -> PathBuf {
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

/// Mean per-pixel std-dev — the cheap non-degenerate floor (an all-black / NaN decode collapses to ~0).
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

fn render(spec: &LoadSpec, w: u32, h: u32, steps: u32, use_pid: bool) -> Image {
    let generator =
        gen_core::load("z_image_turbo", spec).expect("load candle z_image_turbo provider");
    let req = GenerationRequest {
        prompt: env_or(
            "ZIMAGE_PID_PROMPT",
            "a serene mountain lake at golden hour, crisp reflections, highly detailed",
        ),
        width: w,
        height: h,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        use_pid,
        ..Default::default()
    };
    let output = generator
        .generate(&req, &mut |_| {})
        .unwrap_or_else(|e| panic!("z_image_turbo generate (use_pid={use_pid}): {e}"));
    match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    }
}

#[test]
#[ignore = "real-weight GPU smoke; needs the Z-Image-Turbo diffusers snapshot + NC pid-flux + gemma-2-2b-it (cap=120)"]
fn zimage_pid_gpu_smoke() {
    let base = env_path("ZIMAGE_PID_BASE");
    assert!(
        base.join("transformer").is_dir(),
        "ZIMAGE_PID_BASE must be a Tongyi-MAI/Z-Image-Turbo diffusers snapshot (transformer/ missing): {}",
        base.display()
    );
    let ckpt = env_path("ZIMAGE_PID_CKPT");
    let gemma = env_path("ZIMAGE_PID_GEMMA");
    let out_dir = PathBuf::from(env_or("ZIMAGE_PID_OUT", "zimage-pid-out"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let size: u32 = env_or("ZIMAGE_PID_SIZE", "1024")
        .parse()
        .expect("ZIMAGE_PID_SIZE");
    let (w, h) = (size, size);
    let steps: u32 = env_or("ZIMAGE_PID_STEPS", "8")
        .parse()
        .expect("ZIMAGE_PID_STEPS");

    // Native VAE decode (render-sized) — the byte-exact default path, spec WITHOUT pid.
    println!("[smoke] Z-Image native-VAE {w}x{h} @ {steps} steps ...");
    let native_spec = LoadSpec::new(WeightsSource::Dir(base.clone()));
    let native = render(&native_spec, w, h, steps, false);
    let native_std = image_std(&native);
    save_png(&native, &out_dir.join("zimage_native.png"));
    println!(
        "[smoke] Z-Image native {}x{} std {:.2}",
        native.width, native.height, native_std
    );
    assert_eq!(
        (native.width, native.height),
        (w, h),
        "native VAE decode must be render-sized"
    );

    // PiD decode (super-resolving) — the SAME Z-Image latent decoded through the flux-aliased `pid_flux`
    // student at 4×. `spec.with_pid` + `use_pid` is exactly the generic candle lane (sc-9727) for a
    // usePid Z-Image request.
    println!("[smoke] Z-Image PiD {w}x{h} -> expect 4x ...");
    let pid_spec = LoadSpec::new(WeightsSource::Dir(base.clone())).with_pid(
        WeightsSource::File(ckpt.clone()),
        WeightsSource::Dir(gemma.clone()),
    );
    let pid = render(&pid_spec, w, h, steps, true);
    let pid_std = image_std(&pid);
    save_png(&pid, &out_dir.join("zimage_pid.png"));
    println!(
        "[smoke] Z-Image PiD {}x{} std {:.2} -> {}",
        pid.width,
        pid.height,
        pid_std,
        out_dir.join("zimage_pid.png").display()
    );

    assert!(
        native_std > 5.0,
        "native Z-Image render degenerate (std {native_std:.2})"
    );
    assert!(
        pid_std > 5.0,
        "PiD Z-Image render degenerate (std {pid_std:.2}) — check pid-flux + gemma snapshots + cap=120"
    );
    assert!(
        pid.width > native.width && pid.height > native.height,
        "PiD decode must super-resolve past the native size (native {}x{}, pid {}x{})",
        native.width,
        native.height,
        pid.width,
        pid.height
    );
    println!(
        "[smoke] DONE: Z-Image (flux latent) PiD super-resolve {}x{} -> {}x{} via pid_flux — coherent",
        native.width, native.height, pid.width, pid.height
    );
}
