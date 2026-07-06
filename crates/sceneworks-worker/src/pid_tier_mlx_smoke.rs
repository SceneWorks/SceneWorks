//! Local real-weight MLX smoke for the **PiD 2K/4K output tier** (epic 7840, sc-10054). `#[ignore]`d —
//! run by hand on an Apple-Silicon Mac. It is the on-device evidence that the sc-10054 base-resolution
//! mapping actually yields a 2K image at `pidTarget:"2k"` and a 4K image at `"4k"`, on real weights.
//!
//! What it proves end-to-end: PiD super-resolves the sampler latent by a FIXED 4×, so the only lever on
//! output size is the base fed to the decode. The worker translates `advanced.pidTarget` into that base
//! via `pid_output_tier` + `pid_effective_dims` (the REAL functions, called here — not a re-derivation).
//! For a 1024² request: `"4k"` keeps base 1024 → 4096² output; `"2k"` caps base to 512 → 2048² output.
//! Rendering through the same `gen_core::load(...).with_pid(...)` + `GenerationRequest.use_pid` seam the
//! worker lanes use confirms the engine emits exactly `effective_base × 4`.
//!
//! Uses `z_image_turbo` (8-step distilled, CFG-free — the fastest PiD-eligible MLX model) whose FLUX.1
//! latent space decodes through the `pid-flux` student (Z-Image aliases `pid_flux`; there is no dedicated
//! `pid_zimage`). All three snapshots (z-image-turbo-mlx bf16 + NC pid-flux + gemma-2-2b-it) auto-resolve
//! from the HF cache; override with env if needed.
//! ```text
//! # optional: PID_TIER_BASE=/path/to/z-image-turbo-mlx (root containing bf16/, or the bf16/ dir)
//! # optional: PID_TIER_CKPT=/path/to/pid_flux_2kto4k.safetensors  PID_TIER_GEMMA=/path/to/gemma-2-2b-it
//! # optional: PID_TIER_STEPS=8  PID_TIER_OUT=/tmp/pid_tier_smoke  PID_TIER_PROMPT="..."
//! cargo test -p sceneworks-worker --release pid_tier_mlx_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use gen_core::{GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
use sceneworks_core::image_request::ImageRequest;
use serde_json::json;

use super::image_jobs::{pid_effective_dims, pid_output_tier};
use super::smoke_support::{env_or, image_std, save_png, DEGENERATE_STD_FLOOR_DEFAULT};

/// Find the newest snapshot dir of a cached `SceneWorks/<repo>` HF model whose `probe` relative path
/// exists, or `None` if the repo/tier isn't pulled.
fn cached_snapshot(repo_dir: &str, probe: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo_dir)
        .join("snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|dir| dir.join(probe).exists())
}

/// Resolve the z-image-turbo-mlx engine root: prefer `<root>/bf16` (the tiered turnkey layout), else the
/// root itself if it already carries the engine (`transformer/`).
fn resolve_base_dir(root: &Path) -> PathBuf {
    let bf16 = root.join("bf16");
    if bf16.join("transformer").is_dir() {
        return bf16;
    }
    assert!(
        root.join("transformer").is_dir(),
        "PID_TIER_BASE must be a z-image-turbo-mlx turnkey root (containing bf16/) or a bf16/ engine dir \
         with transformer/; neither found under {}",
        root.display()
    );
    root.to_path_buf()
}

/// Render one image through the z_image_turbo MLX generator with the PiD decoder attached (`use_pid`),
/// at the given base `w`×`h`. The PiD student super-resolves the decode 4×, so the returned image is
/// `w*4`×`h*4`.
fn render_pid(spec: &LoadSpec, w: u32, h: u32, steps: u32, prompt: &str) -> Image {
    let generator =
        gen_core::load("z_image_turbo", spec).expect("load mlx z_image_turbo generator");
    let req = GenerationRequest {
        prompt: prompt.to_owned(),
        width: w,
        height: h,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        use_pid: true,
        ..Default::default()
    };
    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |p| {
            let s = format!("{p:?}");
            if s != last {
                println!("[progress] {s}");
                last = s;
            }
        })
        .expect("z_image_turbo PiD generate");
    match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    }
}

/// Build the `ImageRequest` a `pidTarget` job carries, so the tier resolution below runs the REAL parse
/// (`advanced.pidTarget` → `PidOutputTier`) and not a hand-rolled copy.
fn request_with_tier(tier: &str) -> ImageRequest {
    ImageRequest::from_payload(
        json!({
            "model": "z_image_turbo",
            "width": 1024,
            "height": 1024,
            "advanced": { "usePid": true, "pidTarget": tier },
        })
        .as_object()
        .unwrap(),
    )
}

#[test]
#[ignore = "real-weight MLX smoke; needs SceneWorks/{z-image-turbo-mlx bf16, pid-flux, gemma-2-2b-it} cached + an Apple-Silicon Mac"]
fn pid_tier_mlx_gpu_smoke() {
    let base_root = match std::env::var("PID_TIER_BASE") {
        Ok(p) if !p.trim().is_empty() => PathBuf::from(p.trim()),
        _ => cached_snapshot("models--SceneWorks--z-image-turbo-mlx", "bf16/transformer")
            .unwrap_or_else(|| {
                panic!("no cached SceneWorks/z-image-turbo-mlx bf16 tier; set PID_TIER_BASE")
            }),
    };
    let base_dir = resolve_base_dir(&base_root);
    let ckpt = match std::env::var("PID_TIER_CKPT") {
        Ok(p) if !p.trim().is_empty() => PathBuf::from(p.trim()),
        _ => cached_snapshot(
            "models--SceneWorks--pid-flux",
            "pid_flux_2kto4k.safetensors",
        )
        .map(|d| d.join("pid_flux_2kto4k.safetensors"))
        .unwrap_or_else(|| panic!("no cached SceneWorks/pid-flux; set PID_TIER_CKPT")),
    };
    let gemma = match std::env::var("PID_TIER_GEMMA") {
        Ok(p) if !p.trim().is_empty() => PathBuf::from(p.trim()),
        _ => cached_snapshot("models--SceneWorks--gemma-2-2b-it", "config.json")
            .unwrap_or_else(|| panic!("no cached SceneWorks/gemma-2-2b-it; set PID_TIER_GEMMA")),
    };
    let out_dir = PathBuf::from(env_or("PID_TIER_OUT", "/tmp/pid_tier_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let steps: u32 = env_or("PID_TIER_STEPS", "8")
        .parse()
        .expect("PID_TIER_STEPS");
    let prompt = env_or(
        "PID_TIER_PROMPT",
        "a serene mountain lake at golden hour, crisp reflections, highly detailed",
    );

    let spec = LoadSpec::new(WeightsSource::Dir(base_dir.clone())).with_pid(
        WeightsSource::File(ckpt.clone()),
        WeightsSource::Dir(gemma.clone()),
    );

    // Drive the REAL sc-10054 mapping for each tier: a 1024² request → effective base → base×4 output.
    for (tier, expected_out) in [("2k", 2048u32), ("4k", 4096u32)] {
        let req = request_with_tier(tier);
        let (bw, bh) = pid_effective_dims(req.width, req.height, true, pid_output_tier(&req));
        println!(
            "[smoke] pidTarget={tier}: 1024x1024 request -> effective base {bw}x{bh} -> expect {expected_out}x{expected_out} out"
        );
        assert_eq!(
            (bw, bh),
            (expected_out / 4, expected_out / 4),
            "sc-10054 mapping produced the wrong effective base for pidTarget={tier}"
        );

        let img = render_pid(&spec, bw, bh, steps, &prompt);
        let std = image_std(&img);
        let png = out_dir.join(format!("pid_{tier}_{}x{}.png", img.width, img.height));
        save_png(&img, &png);
        println!(
            "[smoke] pidTarget={tier}: rendered {}x{} std {:.2} -> {}",
            img.width,
            img.height,
            std,
            png.display()
        );
        assert_eq!(
            (img.width, img.height),
            (expected_out, expected_out),
            "PiD output for pidTarget={tier} must be effective_base x4 ({expected_out}); got {}x{}",
            img.width,
            img.height
        );
        assert!(
            std > DEGENERATE_STD_FLOOR_DEFAULT,
            "PiD {tier} render looks degenerate (std {std:.2}) — check pid-flux + gemma snapshots"
        );
    }
    println!(
        "[smoke] DONE: sc-10054 tier mapping verified on real weights — 2k -> 2048² and 4k -> 4096² via pid_flux"
    );
}
