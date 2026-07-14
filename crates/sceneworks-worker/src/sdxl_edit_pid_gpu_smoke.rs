//! Local real-weight GPU smoke for the candle **SDXL edit + PiD super-resolving decode** (sc-8044,
//! epic 7840). `#[ignore]`d — run by hand on the RTX PRO 6000. Drives the bespoke
//! `runtime_cuda::providers::sdxl::SdxlEdit` provider (the same one `sdxl_edit_candle.rs` loads) with the PiD decoder
//! attached (`with_pid`), exercising the **inpaint** path (`generate_masked`) — a latent-space
//! mask-blend that ends in a single decode, so PiD sees the same final latent and emits it at 4× (2K/4K)
//! instead of the native SDXL VAE. Represents the whole `sdxl`-latent bespoke group (SdxlEdit /
//! IpAdapterSdxl / KolorsControl / InstantID) — they all reuse this one `pid_sdxl` student + decode seam.
//!
//! Contrast: the same request with `use_pid = false` decodes on the native VAE at the render size, so the
//! smoke asserts (a) the native decode is render-sized + coherent, and (b) the PiD decode is 4× that size
//! + coherent — proving the decode routes through PiD and super-resolves without collapsing.
//!
//! Setup (PowerShell; all three snapshots are in the HF cache — the SDXL base + the NC `pid_sdxl` +
//! gemma-2-2b-it):
//! ```text
//! $env:SDXL_PID_BASE="D:\.cache\huggingface\hub\models--stabilityai--stable-diffusion-xl-base-1.0\snapshots\462165984030d82259a11f4367a4eed129e94a7b"
//! $env:SDXL_PID_CKPT="D:\.cache\huggingface\hub\models--SceneWorks--pid-sdxl\snapshots\70b494831561dc2c181f04a7f057260b8785419a\pid_sdxl_2kto4k.safetensors"
//! $env:SDXL_PID_GEMMA="D:\.cache\huggingface\hub\models--SceneWorks--gemma-2-2b-it\snapshots\684c553b5b41a1c835989d89f62f585e6269a7de"
//! $env:SDXL_PID_OUT="D:\sceneworks-pid-validate\sdxl-edit"
//! # optional: SDXL_PID_SIZE=512 (render size; PiD output is 4x)
//! cargo test -p sceneworks-worker --features backend-candle --release sdxl_edit_pid_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use gen_core::runtime::CancelFlag;
use gen_core::{Image, PidWeights, WeightsSource};
use runtime_cuda::providers::sdxl::{SdxlEdit, SdxlEditPaths, SdxlEditRequest};

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

/// A deterministic non-flat source: a smooth RGB gradient (so the VAE-encode has real structure to keep
/// in the black region and the smoke isn't decoding a constant).
fn synthetic_source(w: u32, h: u32) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x * 255 / w.max(1)) as u8);
            pixels.push((y * 255 / h.max(1)) as u8);
            pixels.push(((x + y) * 255 / (w + h).max(1)) as u8);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// Center-box inpaint mask: white (repaint) in the middle third, black (keep) elsewhere.
fn center_box_mask(w: u32, h: u32) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let inside = x >= w / 3 && x < 2 * w / 3 && y >= h / 3 && y < 2 * h / 3;
            let v = if inside { 255u8 } else { 0u8 };
            pixels.extend_from_slice(&[v, v, v]);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

fn inpaint(model: &SdxlEdit, source: &Image, mask: &Image, w: u32, h: u32, use_pid: bool) -> Image {
    let req = SdxlEditRequest {
        prompt: env_or(
            "SDXL_PID_PROMPT",
            "a red vintage sports car, studio product photo, sharp focus, high detail",
        ),
        negative: "blurry, lowres, deformed, watermark".to_owned(),
        width: w,
        height: h,
        steps: env_or("SDXL_PID_STEPS", "30")
            .parse()
            .expect("SDXL_PID_STEPS"),
        guidance: 5.0,
        strength: 0.85,
        seed: 42,
        use_pid,
        cancel: CancelFlag::new(),
    };
    model
        .generate_masked(&req, source, mask, &mut |_| {})
        .expect("sdxl inpaint generate")
}

#[test]
#[ignore = "real-weight GPU smoke; needs the SDXL base + NC pid_sdxl + gemma-2-2b-it snapshots (cap=120)"]
fn sdxl_edit_pid_gpu_smoke() {
    let base = env_path("SDXL_PID_BASE");
    assert!(
        base.join("unet").is_dir(),
        "SDXL_PID_BASE must be a diffusers SDXL snapshot (unet/ missing): {}",
        base.display()
    );
    let ckpt = env_path("SDXL_PID_CKPT");
    let gemma = env_path("SDXL_PID_GEMMA");
    let out_dir = PathBuf::from(env_or("SDXL_PID_OUT", "sdxl-edit-pid-out"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let size: u32 = env_or("SDXL_PID_SIZE", "512")
        .parse()
        .expect("SDXL_PID_SIZE");
    let (w, h) = (size, size);
    let source = synthetic_source(w, h);
    let mask = center_box_mask(w, h);

    println!("[smoke] loading SdxlEdit from {} ...", base.display());
    let model = SdxlEdit::load(&SdxlEditPaths {
        sdxl_base: base.clone(),
    })
    .expect("load candle SdxlEdit");
    let model = model
        .with_pid(&PidWeights {
            checkpoint: WeightsSource::File(ckpt.clone()),
            gemma: WeightsSource::Dir(gemma.clone()),
        })
        .expect("attach pid_sdxl decoder");

    // Native VAE decode (render-sized): the byte-exact default path.
    println!("[smoke] native-VAE inpaint {w}x{h} ...");
    let native = inpaint(&model, &source, &mask, w, h, false);
    let native_std = image_std(&native);
    save_png(&native, &out_dir.join("sdxl_inpaint_native.png"));
    println!(
        "[smoke] native {}x{} std {:.2}",
        native.width, native.height, native_std
    );
    assert_eq!(
        (native.width, native.height),
        (w, h),
        "native VAE decode must be render-sized"
    );

    // PiD decode (super-resolving): the same final latent, decoded through the `sdxl` PiD student at 4x.
    println!("[smoke] PiD inpaint {w}x{h} -> expect 4x ...");
    let pid = inpaint(&model, &source, &mask, w, h, true);
    let pid_std = image_std(&pid);
    save_png(&pid, &out_dir.join("sdxl_inpaint_pid.png"));
    println!(
        "[smoke] PiD {}x{} std {:.2} -> {}",
        pid.width,
        pid.height,
        pid_std,
        out_dir.join("sdxl_inpaint_pid.png").display()
    );

    assert!(
        native_std > 5.0,
        "native inpaint render degenerate (std {native_std:.2})"
    );
    assert!(
        pid_std > 5.0,
        "PiD inpaint render degenerate (std {pid_std:.2}) — check pid_sdxl + gemma snapshots + cap=120"
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
        "[smoke] DONE: SDXL inpaint PiD super-resolve {}x{} -> {}x{} coherent",
        native.width, native.height, pid.width, pid.height
    );
}
