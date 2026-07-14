//! Local real-weight GPU smoke for the candle **InstantID + PiD super-resolving decode** (sc-8386,
//! epic 7840). `#[ignore]`d — run by hand on the RTX PRO 6000. Drives the bespoke
//! `runtime_cuda::providers::instantid::InstantId` provider (the same one `image_jobs/instantid.rs` loads off-Mac)
//! with the PiD decoder attached (`with_pid`), across **all three InstantID modes** the story names:
//!
//!   * **Identity** (`generate`) — the reference face's ArcFace embedding + its detected kps,
//!   * **Angles** (`generate_angle`) — the same identity re-posed to a canonical view-angle kps pack,
//!   * **Poses** (`generate_pose` + OpenPose CN) — the identity driven onto a COCO-18 body skeleton.
//!
//! Each mode ends in a single latent decode. With `use_pid = false` that latent decodes on the native
//! SDXL VAE at the render size; with `use_pid = true` the same final latent routes through the `sdxl`
//! PiD student (4× → 2K/4K). InstantID reuses the `sdxl` PiD student (it composes the SDXL VAE latent
//! space), so this is the InstantID-lane cousin of `sdxl_edit_pid_gpu_smoke` — same `pid_sdxl`
//! checkpoint + decode seam, different driver (identity conditioning + a likeness assertion).
//!
//! Asserts, per the sc-8386 acceptance:
//!   1. the native Identity decode is render-sized + coherent (the byte-exact baseline),
//!   2. every PiD decode (Identity / Angle / Pose) super-resolves the render size + stays coherent, and
//!   3. **likeness survives PiD**: the ArcFace cosine between the reference face and each PiD output
//!      clears a preservation floor (InstantID's identity envelope is ~0.82 @1024²; a conservative
//!      floor keeps the smoke non-flaky while proving the identity clearly transferred).
//!
//! Setup (PowerShell; all six weight snapshots are in the HF cache — the RealVisXL base + InstantX
//! IdentityNet + the converted `instantid-mlx` face/ip-adapter bundle + the NC `pid_sdxl` + gemma-2-2b-it;
//! Pose mode additionally needs the OpenPose SDXL ControlNet. The reference face defaults to an InstantX
//! example portrait already in the cache):
//! ```text
//! $H="D:\.cache\huggingface\hub"
//! $env:INSTANTID_PID_BASE="$H\models--SG161222--RealVisXL_V5.0\snapshots\ac93e0dda1f6d448cae19bbfab8c5e720a5e48bc"
//! $env:INSTANTID_PID_IDENTITYNET="$H\models--InstantX--InstantID\snapshots\57b32dfee076092ad2930c71fd6d439c2c3b1820\ControlNetModel"
//! $env:INSTANTID_PID_FACEDIR="$H\models--SceneWorks--instantid-mlx\snapshots\bca0cacf8e5e04529bb2b326a521361b02be84fd"
//! $env:INSTANTID_PID_CKPT="$H\models--SceneWorks--pid-sdxl\snapshots\70b494831561dc2c181f04a7f057260b8785419a\pid_sdxl_2kto4k.safetensors"
//! $env:INSTANTID_PID_GEMMA="$H\models--SceneWorks--gemma-2-2b-it\snapshots\684c553b5b41a1c835989d89f62f585e6269a7de"
//! # Pose mode (optional — omit to run Identity + Angle only):
//! $env:INSTANTID_PID_OPENPOSE="$H\models--xinsir--controlnet-openpose-sdxl-1.0\snapshots\23f966cd5cfdd3f7729c903e243d87152162d2b7"
//! # reference face: defaults to $INSTANTID_PID_IDENTITYNET's sibling examples/0.png; override with a portrait:
//! # $env:INSTANTID_PID_FACE="$H\models--InstantX--InstantID\snapshots\57b32dfee076092ad2930c71fd6d439c2c3b1820\examples\0.png"
//! $env:INSTANTID_PID_OUT="D:\sceneworks-pid-validate\instantid"
//! # optional: INSTANTID_PID_SIZE=1024 (render side; PiD output is 4x), INSTANTID_PID_STEPS=30
//! cargo test -p sceneworks-worker --features backend-candle --release instantid_pid_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use gen_core::runtime::CancelFlag;
use gen_core::{Image, PidWeights, WeightsSource};
use runtime_cuda::providers::instantid::{BodyPoint, InstantId, InstantIdPaths, InstantIdRequest};

/// The InstantID identity-preservation floor. InstantID's target envelope is ArcFace-cosine ~0.82
/// @1024²; a conservative 0.35 floor proves the reference identity clearly transferred (an unrelated
/// face sits near 0) without making the smoke flaky on prompt/seed/pose variance.
const LIKENESS_FLOOR: f64 = 0.35;

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

/// ArcFace cosine similarity between two 512-d identity embeddings (both come off the same face stack,
/// so this is the InstantID identity-preservation metric: 1.0 = same face, ~0 = unrelated).
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

fn load_png(path: &Path) -> Image {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("open reference {}: {e}", path.display()))
        .to_rgb8();
    let (width, height) = img.dimensions();
    Image {
        width,
        height,
        pixels: img.into_raw(),
    }
}

fn save_png(img: &Image, path: &Path) {
    image::RgbImage::from_raw(img.width, img.height, img.pixels.clone())
        .expect("rgb buffer")
        .save(path)
        .unwrap_or_else(|e| panic!("save {}: {e}", path.display()));
}

fn base_request(w: u32, h: u32, use_pid: bool) -> InstantIdRequest {
    InstantIdRequest {
        prompt: env_or(
            "INSTANTID_PID_PROMPT",
            "a cinematic studio portrait, dramatic lighting, sharp focus, high detail",
        ),
        negative: "blurry, lowres, deformed, disfigured, watermark".to_owned(),
        width: w,
        height: h,
        steps: env_or("INSTANTID_PID_STEPS", "30")
            .parse()
            .expect("INSTANTID_PID_STEPS"),
        seed: 42,
        use_pid,
        cancel: CancelFlag::new(),
        // Everything else (guidance 5.0, the InstantID IP/controlnet/openpose scales, no sampler /
        // scheduler override) keeps the engine's vendored identity defaults.
        ..Default::default()
    }
}

/// Detect + embed the largest face in `img` through the loaded InstantID face stack; the ArcFace
/// embedding is the identity vector the likeness assertion compares.
fn face_embedding(model: &InstantId, img: &Image, what: &str) -> Vec<f32> {
    model
        .largest_face(img)
        .unwrap_or_else(|e| panic!("no detectable face in {what}: {e}"))
        .embedding
}

/// A canonical frontal standing COCO-18 skeleton, normalized `[0,1]` (x centered at 0.5, y down). The
/// head box sits near the top so `generate_pose` places the reference face there; arms + legs give the
/// OpenPose ControlNet a real body to condition on.
fn standing_skeleton() -> Vec<BodyPoint> {
    // COCO-18 order: 0 nose,1 neck,2 r-sho,3 r-elb,4 r-wri,5 l-sho,6 l-elb,7 l-wri,8 r-hip,9 r-knee,
    // 10 r-ank,11 l-hip,12 l-knee,13 l-ank,14 r-eye,15 l-eye,16 r-ear,17 l-ear.
    vec![
        Some((0.50, 0.20)), // 0 nose
        Some((0.50, 0.28)), // 1 neck
        Some((0.42, 0.30)), // 2 r shoulder
        Some((0.38, 0.45)), // 3 r elbow
        Some((0.35, 0.58)), // 4 r wrist
        Some((0.58, 0.30)), // 5 l shoulder
        Some((0.62, 0.45)), // 6 l elbow
        Some((0.65, 0.58)), // 7 l wrist
        Some((0.45, 0.55)), // 8 r hip
        Some((0.45, 0.72)), // 9 r knee
        Some((0.45, 0.90)), // 10 r ankle
        Some((0.55, 0.55)), // 11 l hip
        Some((0.55, 0.72)), // 12 l knee
        Some((0.55, 0.90)), // 13 l ankle
        Some((0.47, 0.18)), // 14 r eye
        Some((0.53, 0.18)), // 15 l eye
        Some((0.44, 0.19)), // 16 r ear
        Some((0.56, 0.19)), // 17 l ear
    ]
}

/// Assert one PiD output: non-degenerate, super-resolved past the render side, and identity-preserving.
fn assert_pid_output(
    img: &Image,
    ref_embedding: &[f32],
    model: &InstantId,
    side: u32,
    label: &str,
    out_dir: &Path,
) {
    let std = image_std(img);
    let cos = cosine(
        ref_embedding,
        &face_embedding(model, img, &format!("{label} PiD output")),
    );
    save_png(img, &out_dir.join(format!("instantid_{label}_pid.png")));
    println!(
        "[smoke] {label} PiD {}x{} std {:.2} likeness {:.3}",
        img.width, img.height, std, cos
    );
    assert!(std > 5.0, "{label} PiD render degenerate (std {std:.2})");
    assert!(
        img.width > side && img.height > side,
        "{label} PiD decode must super-resolve past the render side {side} (got {}x{})",
        img.width,
        img.height
    );
    assert!(
        cos > LIKENESS_FLOOR,
        "{label} PiD likeness {cos:.3} below floor {LIKENESS_FLOOR} — the 4x PiD decode scrubbed the face"
    );
}

#[test]
#[ignore = "real-weight GPU smoke; needs RealVisXL + InstantX IdentityNet + instantid-mlx face/ip + NC pid_sdxl + gemma-2-2b-it (+ optional OpenPose CN) (cap=120)"]
fn instantid_pid_gpu_smoke() {
    let base = env_path("INSTANTID_PID_BASE");
    assert!(
        base.join("unet").is_dir(),
        "INSTANTID_PID_BASE must be a diffusers SDXL/RealVisXL snapshot (unet/ missing): {}",
        base.display()
    );
    let identitynet = env_path("INSTANTID_PID_IDENTITYNET");
    let face_dir = env_path("INSTANTID_PID_FACEDIR");
    let ip_adapter = face_dir.join("ip-adapter.safetensors");
    assert!(
        ip_adapter.is_file(),
        "converted ip-adapter.safetensors missing from INSTANTID_PID_FACEDIR: {}",
        ip_adapter.display()
    );
    let ckpt = env_path("INSTANTID_PID_CKPT");
    let gemma = env_path("INSTANTID_PID_GEMMA");
    // Pose mode is optional: attach OpenPose only when its snapshot is provided; otherwise the smoke
    // runs Identity + Angle (both share the `generate_with` decode seam) and skips Pose.
    let openpose = std::env::var("INSTANTID_PID_OPENPOSE")
        .ok()
        .map(|v| PathBuf::from(v.trim()))
        .filter(|p| p.is_dir());
    // Reference portrait: default to an InstantX example that ships in the same snapshot as IdentityNet
    // (examples/0.png), overridable with any face photo.
    let reference_path = PathBuf::from(env_or(
        "INSTANTID_PID_FACE",
        identitynet
            .parent()
            .map(|p| p.join("examples").join("0.png"))
            .unwrap_or_else(|| PathBuf::from("0.png"))
            .to_string_lossy()
            .as_ref(),
    ));
    let out_dir = PathBuf::from(env_or("INSTANTID_PID_OUT", "instantid-pid-out"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let size: u32 = env_or("INSTANTID_PID_SIZE", "1024")
        .parse()
        .expect("INSTANTID_PID_SIZE");
    let (w, h) = (size, size);

    let reference = load_png(&reference_path);
    println!(
        "[smoke] reference {} ({}x{})",
        reference_path.display(),
        reference.width,
        reference.height
    );

    println!("[smoke] loading InstantId from {} ...", base.display());
    let model = InstantId::load(&InstantIdPaths {
        sdxl_base: base.clone(),
        identitynet: WeightsSource::Dir(identitynet.clone()),
        ip_adapter: ip_adapter.clone(),
        adapters: Vec::new(),
    })
    .expect("load candle InstantId");
    // Attach OpenPose (pose mode) BEFORE the face stack — the engine's documented order. Harmless for
    // Identity / Angle (only `generate_pose` uses it), so one model serves all three modes.
    let model = match &openpose {
        Some(dir) => {
            println!("[smoke] attaching OpenPose CN from {} ...", dir.display());
            model
                .with_openpose(&WeightsSource::Dir(dir.clone()))
                .expect("attach OpenPose ControlNet")
        }
        None => {
            println!("[smoke] INSTANTID_PID_OPENPOSE unset/not-a-dir -> skipping Pose mode");
            model
        }
    };
    let mut model = model
        .with_face(&face_dir)
        .expect("attach SCRFD + ArcFace face stack")
        .with_pid(&PidWeights {
            checkpoint: WeightsSource::File(ckpt.clone()),
            gemma: WeightsSource::Dir(gemma.clone()),
        })
        .expect("attach pid_sdxl decoder");

    // Reference identity embedding — the target every output is scored against.
    let ref_embedding = face_embedding(&model, &reference, "reference portrait");

    // --- Identity: native VAE baseline (render-sized) then PiD (super-resolving) ---
    println!("[smoke] Identity native-VAE {w}x{h} ...");
    let native = model
        .generate(&base_request(w, h, false), &reference, &mut |_| {})
        .expect("instantid identity native generate");
    let native_std = image_std(&native);
    let native_cos = cosine(
        &ref_embedding,
        &face_embedding(&model, &native, "identity native output"),
    );
    save_png(&native, &out_dir.join("instantid_identity_native.png"));
    println!(
        "[smoke] Identity native {}x{} std {:.2} likeness {:.3}",
        native.width, native.height, native_std, native_cos
    );
    assert_eq!(
        (native.width, native.height),
        (w, h),
        "native VAE decode must be render-sized"
    );
    assert!(
        native_std > 5.0,
        "native identity render degenerate (std {native_std:.2})"
    );
    assert!(
        native_cos > LIKENESS_FLOOR,
        "native identity likeness {native_cos:.3} below floor {LIKENESS_FLOOR} — InstantID not conditioning on the reference"
    );

    println!("[smoke] Identity PiD {w}x{h} -> expect 4x ...");
    let identity_pid = model
        .generate(&base_request(w, h, true), &reference, &mut |_| {})
        .expect("instantid identity PiD generate");
    assert_pid_output(
        &identity_pid,
        &ref_embedding,
        &model,
        size,
        "identity",
        &out_dir,
    );

    // --- Angles: canonical view-angle kps pack, PiD decode ---
    let angle = env_or("INSTANTID_PID_ANGLE", "front");
    println!("[smoke] Angle {angle:?} PiD side {size} -> expect 4x ...");
    let angle_pid = model
        .generate_angle(
            &base_request(size, size, true),
            &reference,
            &angle,
            &mut |_| {},
        )
        .unwrap_or_else(|e| panic!("instantid angle {angle:?} PiD generate: {e}"));
    assert_pid_output(&angle_pid, &ref_embedding, &model, size, "angle", &out_dir);

    // --- Poses: COCO-18 body skeleton + OpenPose CN, PiD decode (only when OpenPose was attached) ---
    if openpose.is_some() {
        let skeleton = standing_skeleton();
        println!("[smoke] Pose (standing skeleton) PiD side {size} -> expect 4x ...");
        let pose_pid = model
            .generate_pose(
                &base_request(size, size, true),
                &reference,
                &skeleton,
                &mut |_| {},
            )
            .expect("instantid pose PiD generate");
        assert_pid_output(&pose_pid, &ref_embedding, &model, size, "pose", &out_dir);
    }

    println!(
        "[smoke] DONE: InstantID PiD super-resolve {size}x{size} -> 4x across Identity/Angle{} — likeness preserved (native identity {:.3})",
        if openpose.is_some() { "/Pose" } else { "" },
        native_cos
    );
}
