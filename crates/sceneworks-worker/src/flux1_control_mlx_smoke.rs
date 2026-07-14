//! Local real-weight MLX smokes for the FLUX.1-dev strict-control worker lane (sc-8244; engine E2
//! sc-8239). `#[ignore]`d — run by hand on an Apple-Silicon Mac with the gated FLUX.1-dev snapshot + the
//! Shakker `FLUX.1-dev-ControlNet-Union-Pro-2.0` checkpoint. They drive the real native-MLX
//! `flux1_dev_control` engine via `crate::inference_runtime::load("flux1_dev_control")` with a `Dir` base + `with_control`
//! overlay — the exact runtime seam `generate_flux1_dev_control_stream` builds (`flux1_control_spec`),
//! minus the API/job plumbing. The point is to prove the **worker drives the bundled engine
//! end-to-end** per control mode through `runtime-macos`, not just the engine crate in isolation (that's
//! `mlx-gen-flux`'s own `control_real_weights` test).
//!
//! Each mode asserts a control-vs-control-free **steer**: the same prompt/seed rendered control-free vs
//! with the `kind`-tagged control conditioning must differ meaningfully (mean abs Δ over the decode) —
//! the structural hint actually bent the output. pose/canny/depth all flow through the SAME input-agnostic
//! Shakker Union-Pro-2.0 branch (only the control image differs), so one harness drives all three.
//!
//! Setup (point at the gated dense FLUX.1-dev diffusers snapshot + the Shakker control checkpoint):
//! ```text
//! FLUX1_DEV_DIR=~/.cache/huggingface/.../FLUX.1-dev \
//! FLUX1_CONTROL=~/.cache/huggingface/.../diffusion_pytorch_model.safetensors \
//! cargo test -p sceneworks-worker --release flux1_dev_control_mlx_smoke -- --ignored --nocapture
//! ```
//!
//! The `mod flux1_control_mlx_smoke;` declaration in `lib.rs` already carries the same
//! `#[cfg(all(test, target_os = "macos"))]`, so no inner `#![cfg]` is needed here (sc-8952).

use std::path::PathBuf;

use gen_core::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource,
};

use super::smoke_support::{env_or, image_std, DEGENERATE_STD_FLOOR_DEFAULT};

fn env_path(key: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(key)
            .unwrap_or_else(|_| panic!("set ${key}"))
            .trim(),
    )
}

/// A synthetic single-channel-ish control fixture (a smooth top-to-bottom luminance ramp broadcast to
/// RGB) — enough to exercise the VAE-encode of the control hint on real weights without shipping a
/// per-mode fixture (the engine treats pose/canny/depth maps the same way: VAE-encode + pack). Distinct
/// from a flat field so the control latent isn't degenerate.
fn ramp_rgb(w: u32, h: u32) -> Image {
    let (wu, hu) = (w as usize, h as usize);
    let mut pixels = vec![0u8; wu * hu * 3];
    for y in 0..hu {
        let v = ((y * 255) / hu.max(1)) as u8;
        for x in 0..wu {
            let i = (y * wu + x) * 3;
            pixels[i] = v;
            pixels[i + 1] = v;
            pixels[i + 2] = v;
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// Mean absolute per-pixel difference between two same-shape decodes — the control-vs-control-free steer
/// metric (0 = identical = the control did nothing). Panics on a shape mismatch (a bug, not a render).
fn mean_abs_delta(a: &Image, b: &Image) -> f64 {
    assert_eq!(
        (a.width, a.height, a.pixels.len()),
        (b.width, b.height, b.pixels.len()),
        "decodes must share a shape to diff"
    );
    if a.pixels.is_empty() {
        return 0.0;
    }
    a.pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| (x as f64 - y as f64).abs())
        .sum::<f64>()
        / a.pixels.len() as f64
}

/// Render one image through the loaded `flux1_dev_control` generator with the given conditioning (empty =
/// control-free baseline).
fn render(
    generator: &dyn gen_core::Generator,
    prompt: &str,
    w: u32,
    h: u32,
    steps: u32,
    guidance: f32,
    conditioning: Vec<Conditioning>,
) -> Image {
    let req = GenerationRequest {
        prompt: prompt.to_owned(),
        width: w,
        height: h,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        guidance: Some(guidance),
        conditioning,
        ..Default::default()
    };
    let output = generator
        .generate(&req, &mut |_| {})
        .expect("flux1_dev_control generate");
    match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    }
}

/// Real-weight MLX smoke: load `flux1_dev_control` once, render a control-free baseline, then a render per
/// control mode (pose/canny/depth), and assert each mode (a) is non-degenerate and (b) steers the output
/// away from the control-free baseline (the structural hint bent the result). The control image is the
/// same synthetic ramp for every mode (Union-Pro-2.0 is input-agnostic — pose/canny/depth differ only by
/// the upstream preprocessor, which these worker smokes deliberately bypass: the per-preprocessor
/// derivation is unit-tested in `image_jobs::tests`).
#[test]
#[ignore = "real-weight MLX smoke; needs the gated FLUX.1-dev snapshot + the Shakker Union-Pro-2.0 ckpt on an Apple-Silicon Mac"]
fn flux1_dev_control_mlx_smoke() {
    let weights_dir = env_path("FLUX1_DEV_DIR");
    let control_ckpt = env_path("FLUX1_CONTROL");
    assert!(
        weights_dir.join("model_index.json").is_file() || weights_dir.join("transformer").is_dir(),
        "FLUX1_DEV_DIR must point at the dense FLUX.1-dev diffusers snapshot: {}",
        weights_dir.display()
    );
    let out_dir = PathBuf::from(env_or("FLUX1_OUT_DIR", "/tmp/flux1_control_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let w: u32 = env_or("FLUX1_W", "768").parse().expect("FLUX1_W");
    let h: u32 = env_or("FLUX1_H", "768").parse().expect("FLUX1_H");
    let steps: u32 = env_or("FLUX1_STEPS", "28").parse().expect("FLUX1_STEPS");
    let guidance: f32 = env_or("FLUX1_GUIDANCE", "3.5")
        .parse()
        .expect("FLUX1_GUIDANCE");
    let control_scale: f32 = env_or("FLUX1_CONTROL_SCALE", "0.7")
        .parse()
        .expect("FLUX1_CONTROL_SCALE");
    let prompt = env_or(
        "FLUX1_PROMPT",
        "a knight in ornate steel armor, dramatic cinematic lighting, sharp focus",
    );

    // Same seam the worker's `flux1_control_spec` builds: a Dir base + the Shakker control overlay.
    let spec = LoadSpec::new(WeightsSource::Dir(weights_dir.clone()))
        .with_control(WeightsSource::File(control_ckpt.clone()));
    let generator = crate::inference_runtime::load("flux1_dev_control", &spec)
        .expect("load flux1_dev_control provider");

    // Control-free baseline (no conditioning → the union ControlNet branch is inert).
    let baseline = render(&*generator, &prompt, w, h, steps, guidance, Vec::new());
    let base_std = image_std(&baseline);
    image::RgbImage::from_raw(w, h, baseline.pixels.clone())
        .expect("rgb")
        .save(out_dir.join("flux1_control_baseline.png"))
        .expect("save baseline");
    assert!(
        base_std > DEGENERATE_STD_FLOOR_DEFAULT,
        "control-free baseline looks degenerate (std {base_std:.2})"
    );

    let control_map = ramp_rgb(w, h);
    for kind in [ControlKind::Pose, ControlKind::Canny, ControlKind::Depth] {
        let label = match kind {
            ControlKind::Pose => "pose",
            ControlKind::Canny => "canny",
            ControlKind::Depth => "depth",
            ControlKind::Other(_) => "other",
        };
        // Depth uses the dedicated `Depth` conditioning; pose/canny use `Control { kind, scale }` — the
        // same split the worker's `build_control_conditioning` makes.
        let conditioning = match kind {
            ControlKind::Depth => vec![Conditioning::Depth {
                image: control_map.clone(),
            }],
            kind => vec![Conditioning::Control {
                image: control_map.clone(),
                kind,
                scale: Some(control_scale),
            }],
        };
        let image = render(&*generator, &prompt, w, h, steps, guidance, conditioning);
        let std = image_std(&image);
        let steer = mean_abs_delta(&image, &baseline);
        image::RgbImage::from_raw(w, h, image.pixels.clone())
            .expect("rgb")
            .save(out_dir.join(format!("flux1_control_{label}.png")))
            .expect("save control render");
        println!("[smoke] flux1 {label}: std {std:.2}, steer(meanAbsΔ vs control-free) {steer:.2}");
        assert_eq!((image.width, image.height), (w, h));
        assert!(
            std > DEGENERATE_STD_FLOOR_DEFAULT,
            "flux1 {label} control render looks degenerate (std {std:.2})"
        );
        assert!(
            steer > 1.0,
            "flux1 {label} control did not steer the output away from the control-free baseline \
             (meanAbsΔ {steer:.2}) — the structural hint was inert"
        );
    }
    println!("[smoke] DONE: flux1_dev_control pose/canny/depth all coherent + steering");
}
