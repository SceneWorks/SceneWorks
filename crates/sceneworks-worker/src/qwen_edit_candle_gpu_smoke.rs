//! Local real-weight GPU smoke for the candle **Qwen-Image-Edit** worker lane (sc-13534 — the
//! hardware-gated acceptance for the tier/gate reconciliation). `#[ignore]`d — run by hand on the RTX
//! PRO 6000 (Blackwell sm_120). It drives the ACTUAL worker seam this story fixed —
//! [`crate::image_jobs::resolve_qwen_edit_candle_base`] — then loads the bespoke
//! `runtime_cuda::providers::qwen_image::QwenEdit` provider at the resolved dir and renders one real
//! reference-conditioned edit.
//!
//! ## Why this smoke exists
//!
//! Before sc-13534 the lane resolved the UPSTREAM `Qwen/Qwen-Image-Edit(-2511)` snapshot — a repo no
//! download flow fetches. So on a normal install the lane was dead, and on a dev box that happened to
//! have the dense upstream snapshot cached it loaded DENSE bf16 while the fit-gate had sized q4. This
//! smoke is the "the packed turnkey tier actually resolves + loads + renders" evidence:
//!
//!   1. The worker resolver lands on the `q4/` tier subdir of `SceneWorks/qwen-image-edit-2511-mlx`
//!      (NOT the upstream repo, NOT the turnkey root that holds no `transformer/`).
//!   2. `QwenEdit::load` packed-detects that tier from `transformer/config.json` and renders a coherent,
//!      non-degenerate edit — proving the packed q4 path the sc-11019 `vramGbByTier` rows were measured
//!      against is the path that runs.
//!
//! Build with `CUDA_COMPUTE_CAP=120` (native Blackwell sm_120). The HF cache env points at the real
//! snapshot — no explicit weights path (the worker resolver finds it and, on Windows, hardlink-repairs
//! the `hf`-populated blob symlinks candle can't traverse):
//! ```text
//! $env:HF_HOME="E:\huggingface"                # or HF_HUB_CACHE="E:\huggingface\hub"
//! $env:QWEN_EDIT_OUT_DIR="D:\sceneworks-qwen-edit-validate"
//! # optional: QWEN_EDIT_STEPS=8  QWEN_EDIT_GUIDANCE=4.0  QWEN_EDIT_SEED=42  QWEN_EDIT_W=768  QWEN_EDIT_H=768
//! #           QWEN_EDIT_MODEL=qwen_image_edit_2511  QWEN_EDIT_PROMPT="..."
//! cargo test -p sceneworks-worker --no-default-features --features backend-candle --release \
//!   qwen_edit_worker_lane_gpu_smoke -- --ignored --nocapture
//! ```

use gen_core::{CancelFlag, Image, OffloadPolicy, Progress};
use runtime_cuda::providers::qwen_image::{QwenEdit, QwenEditPaths, QwenEditRequest};

use super::smoke_support::{env_or, image_std, save_png, DEGENERATE_STD_FLOOR_DEFAULT};

/// sc-13534 worker-lane E2E: drive the ACTUAL worker base resolution + `QwenEdit::load` + render for the
/// candle Qwen-Image-Edit lane on real CUDA. Calls the WORKER's
/// [`crate::image_jobs::resolve_qwen_edit_candle_base`] the way `generate_candle_qwen_edit_stream` does,
/// asserts it resolves the `q4/` tier subdir of the `SceneWorks/qwen-image-edit-2511-mlx` turnkey (NOT
/// the upstream snapshot the pre-fix code reached, NOT the tier-less root), then packed-loads q4 and
/// renders a coherent true-CFG edit against a synthetic reference.
#[test]
#[ignore = "real-weight worker-lane GPU smoke; needs the SceneWorks/qwen-image-edit-2511-mlx turnkey in \
            the HF cache (HF_HUB_CACHE/HF_HOME) + a CUDA device (cap=120)"]
fn qwen_edit_worker_lane_gpu_smoke() {
    use crate::image_jobs::resolve_qwen_edit_candle_base;
    use crate::settings::Settings;
    use sceneworks_core::image_request::ImageRequest;
    use serde_json::json;

    let out_dir = std::path::PathBuf::from(env_or("QWEN_EDIT_OUT_DIR", "/tmp/qwen_edit_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let settings = Settings::from_env();

    let model = env_or("QWEN_EDIT_MODEL", "qwen_image_edit_2511");
    let steps: usize = env_or("QWEN_EDIT_STEPS", "8")
        .parse()
        .expect("QWEN_EDIT_STEPS");
    let guidance: f32 = env_or("QWEN_EDIT_GUIDANCE", "4.0")
        .parse()
        .expect("QWEN_EDIT_GUIDANCE");
    let seed: u64 = env_or("QWEN_EDIT_SEED", "42")
        .parse()
        .expect("QWEN_EDIT_SEED");
    let w: u32 = env_or("QWEN_EDIT_W", "768").parse().expect("QWEN_EDIT_W");
    let h: u32 = env_or("QWEN_EDIT_H", "768").parse().expect("QWEN_EDIT_H");
    let prompt = env_or(
        "QWEN_EDIT_PROMPT",
        "turn the plain gray card into a lush watercolor painting of a mountain lake at sunrise",
    );
    let negative = env_or(
        "QWEN_EDIT_NEG",
        "blurry, low quality, jpeg artifacts, deformed",
    );

    // The minimal job payload the router hands `generate_candle_qwen_edit_stream`: an `edit_image` job on
    // a Qwen-Image-Edit id with a source asset. `mlxQuantize: 4` pins the q4 tier deterministically —
    // both the receipt-variant step and the `standard_tier_subdir` descent resolve q4 regardless of
    // whether an app download receipt exists in this cache (an `hf`-populated cache has none). No
    // manifest `repo`, so `qwen_edit_candle_repo` reads the turnkey off the MODEL_TABLE row — the exact
    // production default this story restored.
    let payload = json!({
        "model": model,
        "mode": "edit_image",
        "sourceAssetId": "smoke-src",
        "prompt": prompt,
        "width": w,
        "height": h,
        "advanced": { "mlxQuantize": 4 },
    });
    let request = ImageRequest::from_payload(payload.as_object().unwrap());

    // THE worker resolver (the sc-13534 change): edit lane → the q4 tier subdir of the packed turnkey.
    let dir = resolve_qwen_edit_candle_base(&request, &settings)
        .expect("resolve_qwen_edit_candle_base")
        .unwrap_or_else(|| {
            panic!(
                "{model}: SceneWorks/qwen-image-edit-2511-mlx turnkey not in the HF cache — set \
                 HF_HUB_CACHE/HF_HOME"
            )
        });
    assert!(
        dir.ends_with("q4"),
        "{model}: worker must descend into the q4 tier subdir, got {}",
        dir.display()
    );
    assert!(
        dir.join("transformer").is_dir(),
        "{model}: resolved tier dir must hold transformer/ (packed-detect reads its config.json), got {}",
        dir.display()
    );
    // The exact defect this story fixed: never the upstream snapshot, never the tier-less turnkey root.
    let dir_str = dir.to_string_lossy().replace('\\', "/");
    assert!(
        dir_str.contains("qwen-image-edit-2511-mlx"),
        "{model}: must resolve the SceneWorks turnkey, not an upstream snapshot, got {dir_str}"
    );
    assert!(
        !dir_str.contains("models--Qwen--Qwen-Image-Edit"),
        "{model}: resolved the UPSTREAM repo — the pre-sc-13534 bug, got {dir_str}"
    );
    println!(
        "[worker-smoke] {model}: resolve_qwen_edit_candle_base -> {}",
        dir.display()
    );

    // Packed q4 load, resident residency (the default cross-request-cached path). No adapters — this is
    // the production (non-distilled) true-CFG edit path; QwenEdit packed-detects q4 from the tier's
    // transformer/config.json (group_size 64).
    let model_engine = QwenEdit::load(&QwenEditPaths {
        root: dir.clone(),
        adapters: Vec::new(),
        offload_policy: OffloadPolicy::Resident,
    })
    .unwrap_or_else(|e| panic!("QwenEdit::load({}): {e}", dir.display()));

    // A synthetic reference: a flat mid-gray card. The prompt asks for a strong stylization, so a live
    // edit must diverge from the flat input (a degenerate/black decode stays flat and fails the floor).
    let reference = Image {
        width: w,
        height: h,
        pixels: vec![128u8; (w * h * 3) as usize],
    };

    let cancel = CancelFlag::new();
    let req = QwenEditRequest {
        prompt: prompt.clone(),
        negative: negative.clone(),
        width: w,
        height: h,
        steps,
        guidance,
        seed,
        lightning: false,
        cancel: cancel.clone(),
    };
    println!(
        "[worker-smoke] {model}: rendering {w}x{h} @ {steps} steps, guidance {guidance}, seed {seed} ..."
    );
    let started = std::time::Instant::now();
    let mut steps_seen = 0u32;
    let out = model_engine
        .generate(&req, std::slice::from_ref(&reference), &mut |p| {
            if let Progress::Step { current, .. } = p {
                steps_seen = steps_seen.max(current);
            }
        })
        .unwrap_or_else(|e| panic!("{model} generate: {e}"));
    let elapsed = started.elapsed();

    assert_eq!((out.width, out.height), (w, h), "output dims");
    assert_eq!(out.pixels.len(), (w * h * 3) as usize, "RGB8 pixel count");
    assert!(steps_seen >= 1, "expected denoise step progress");

    // Degenerate-decode floor: a coherent edit clears the general per-pixel std-dev bar by a wide
    // margin; a NaN / all-black / flat collapse pulls it toward 0.
    let std = image_std(&out);
    assert!(
        std >= DEGENERATE_STD_FLOOR_DEFAULT,
        "{model}: decode looks degenerate (per-pixel std {std:.3} < {DEGENERATE_STD_FLOOR_DEFAULT}) — \
         the packed q4 tier did not render a live image"
    );

    let png = out_dir.join(format!("{model}_q4_{w}x{h}_s{seed}.png"));
    save_png(&out, &png);
    println!(
        "[worker-smoke] {model}: q4 edit OK in {:.1}s (std {std:.2}) -> {}",
        elapsed.as_secs_f32(),
        png.display()
    );
}
