//! Local real-weight MLX smoke for the Krea 2 Turbo pose-ControlNet worker lane on a **packed Q8 base**
//! (sc-11796). `#[ignore]`d — run by hand on an Apple-Silicon Mac with the `SceneWorks/krea-2-turbo-mlx`
//! turnkey + the `SceneWorks/krea2-pose-controlnet-beta` overlay cached. It drives the real native-MLX
//! `krea_2_turbo_control` engine via `gen_core::load("krea_2_turbo_control")` with the EXACT `LoadSpec`
//! the worker's `krea_control_spec` builds after sc-11796: a `Dir` pointed at the packed **q8** turnkey
//! tier subdir (`krea_model_subdir`'s default) + the pose overlay as the required `Control` checkpoint +
//! `with_quant(Q8)`. This proves two things the old dense-only lane could not. First, the control branch
//! loads + renders on a PACKED q8 base (mlx-gen sc-11730's true packed forward), not just the dense bf16
//! tree the phantom `krea/Krea-2-Turbo` repo used to demand. Second, the pose actually STEERS the output —
//! the same pose skeleton at `control_scale = 0.6` must differ meaningfully from `control_scale = 0.0`
//! (base passthrough); if pose selection were being silently dropped (the reported sc-11796 bug), the two
//! renders would be identical.
//!
//! Same force-link anchor idiom as the sibling smokes: `use mlx_gen_krea as _;` in `image_jobs.rs` keeps
//! the `krea_2_turbo_control` `inventory::submit!` from being GC'd so `gen_core::load` resolves it.
//!
//! Setup — with the turnkey + overlay already in the HF cache, no env is needed (auto-discovered). The
//! pose defaults to a repo skeleton PNG (`poses/dance_01.png`), already an OpenPose render.
//! ```text
//! # optional: KREA_CTRL_POSE=/path/to/skeleton.png KREA_CTRL_STEPS=8 KREA_CTRL_SIZE=512
//! #           KREA_CTRL_SCALE=0.6 KREA_CTRL_SEED=1234 KREA_CTRL_OUT_DIR=/tmp/krea_control_smoke
//! cargo test -p sceneworks-worker krea_control_q8_mlx_smoke -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use gen_core::{
    CancelFlag, Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec,
    Quant, WeightsSource,
};

use super::smoke_support::{env_or, image_std, DEGENERATE_STD_FLOOR_DEFAULT};

/// Auto-discover the cached `SceneWorks/krea-2-turbo-mlx` q8 tier subdir (the packed transformer root the
/// worker's `krea_model_subdir` resolves by default). `None` if the turnkey q8 tier hasn't been pulled.
fn cached_q8_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--SceneWorks--krea-2-turbo-mlx/snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let q8 = e.path().join("q8");
            q8.join("transformer/diffusion_pytorch_model.safetensors")
                .is_file()
                .then_some(q8)
        })
}

/// Auto-discover the cached `SceneWorks/krea2-pose-controlnet-beta` overlay checkpoint.
fn cached_overlay() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--SceneWorks--krea2-pose-controlnet-beta/snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let f = e.path().join("control_step5000.safetensors");
            f.is_file().then_some(f)
        })
}

/// Load an RGB skeleton PNG into a gen_core `Image`.
fn load_rgb(path: &std::path::Path) -> Image {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("open pose skeleton {}: {e}", path.display()))
        .to_rgb8();
    let (width, height) = img.dimensions();
    Image {
        width,
        height,
        pixels: img.into_raw(),
    }
}

/// Mean absolute per-byte delta between two same-size RGB buffers (the pose-steer metric).
fn mean_abs_delta(a: &Image, b: &Image) -> f64 {
    assert_eq!(
        a.pixels.len(),
        b.pixels.len(),
        "compared images differ in size"
    );
    let sum: u64 = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u64)
        .sum();
    sum as f64 / a.pixels.len() as f64
}

fn render_pose(
    generator: &dyn gen_core::Generator,
    prompt: &str,
    size: u32,
    steps: u32,
    seed: u64,
    skeleton: &Image,
    scale: f32,
) -> Image {
    let request = GenerationRequest {
        prompt: prompt.to_owned(),
        width: size,
        height: size,
        count: 1,
        seed: Some(seed),
        steps: Some(steps),
        // Krea Turbo is CFG-free (distilled) — no guidance, one forward per step.
        guidance: None,
        conditioning: vec![Conditioning::Control {
            image: skeleton.clone(),
            kind: ControlKind::Pose,
            scale: Some(scale),
        }],
        cancel: CancelFlag::new(),
        ..Default::default()
    };
    match generator
        .generate(&request, &mut |_| {})
        .unwrap_or_else(|e| panic!("krea_2_turbo_control generate at scale {scale}: {e}"))
    {
        GenerationOutput::Images(mut images) => images.pop().expect("one image"),
        other => panic!("expected Images output, got {other:?}"),
    }
}

#[test]
#[ignore = "real-weight MLX smoke; needs the SceneWorks/krea-2-turbo-mlx q8 turnkey + krea2-pose-controlnet-beta overlay cached on an Apple-Silicon Mac"]
fn krea_control_q8_mlx_smoke() {
    let q8_dir = cached_q8_dir().unwrap_or_else(|| {
        panic!("no cached SceneWorks/krea-2-turbo-mlx q8 turnkey found — download the Krea 2 Turbo q8 tier")
    });
    let overlay = cached_overlay()
        .unwrap_or_else(|| panic!("no cached SceneWorks/krea2-pose-controlnet-beta overlay found"));
    let pose_path = PathBuf::from(env_or(
        "KREA_CTRL_POSE",
        "/Users/michael/Repos/SceneWorks/poses/dance_01.png",
    ));
    let out_dir = PathBuf::from(env_or("KREA_CTRL_OUT_DIR", "/tmp/krea_control_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let steps: u32 = env_or("KREA_CTRL_STEPS", "8")
        .parse()
        .expect("KREA_CTRL_STEPS");
    let size: u32 = env_or("KREA_CTRL_SIZE", "512")
        .parse()
        .expect("KREA_CTRL_SIZE");
    let scale: f32 = env_or("KREA_CTRL_SCALE", "0.6")
        .parse()
        .expect("KREA_CTRL_SCALE");
    let seed: u64 = env_or("KREA_CTRL_SEED", "1234")
        .parse()
        .expect("KREA_CTRL_SEED");
    let prompt = env_or(
        "KREA_CTRL_PROMPT",
        "a full-body studio photo of a person, plain grey background, sharp focus",
    );

    let skeleton = load_rgb(&pose_path);

    // The EXACT seam `krea_control_spec` builds after sc-11796: the packed q8 tier subdir as the base
    // `Dir` + the pose overlay as the required control checkpoint + `with_quant(Q8)` (the resolved tier's
    // packed bits, a no-op match for the pre-packed subdir). Proves the control branch runs a true packed
    // forward on the q8 base the plain txt2img lane already installs.
    println!(
        "[smoke] loading krea_2_turbo_control (packed Q8) base {} overlay {}",
        q8_dir.display(),
        overlay.display()
    );
    let spec = LoadSpec::new(WeightsSource::Dir(q8_dir))
        .with_control(WeightsSource::File(overlay))
        .with_quant(Quant::Q8);
    let generator = crate::inference_runtime::load("krea_2_turbo_control", &spec)
        .expect("load krea_2_turbo_control generator");

    // A/B on the SAME pose/prompt/seed: pose-locked (scale) vs base passthrough (0.0). Only the control
    // residual differs, so a meaningful delta proves the pose actually steers the render.
    let locked = render_pose(&*generator, &prompt, size, steps, seed, &skeleton, scale);
    let passthrough = render_pose(&*generator, &prompt, size, steps, seed, &skeleton, 0.0);

    for (label, img) in [("scale", &locked), ("base", &passthrough)] {
        image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
            .expect("rgb")
            .save(out_dir.join(format!("krea_control_q8_{label}.png")))
            .expect("save render");
    }

    let locked_std = image_std(&locked);
    let base_std = image_std(&passthrough);
    let steer = mean_abs_delta(&locked, &passthrough);
    println!(
        "[smoke] q8 pose-lock std {locked_std:.2} / base std {base_std:.2} / steer(meanAbsΔ) {steer:.2} -> {}",
        out_dir.display()
    );

    assert_eq!((locked.width, locked.height), (size, size));
    assert!(
        locked_std > DEGENERATE_STD_FLOOR_DEFAULT && base_std > DEGENERATE_STD_FLOOR_DEFAULT,
        "a render looks degenerate (pose-lock std {locked_std:.2}, base std {base_std:.2})"
    );
    // The core sc-11796 assertion: the pose STEERS the output on the packed q8 base. If pose selection
    // were being dropped (silent t2i), the two renders would be identical (steer ~= 0).
    assert!(
        steer > 1.0,
        "pose did not steer the packed-q8 render vs base passthrough (meanAbsΔ {steer:.2}) — the pose \
         was effectively ignored"
    );
    println!("[smoke] DONE: krea_2_turbo_control renders pose-locked on a PACKED Q8 base");
}
