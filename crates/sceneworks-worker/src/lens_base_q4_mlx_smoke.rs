//! Local real-weight MLX smoke for the recovered base **Lens** **Q4** worker lane (sc-8767, epic 8506
//! Group-B). `#[ignore]`d — run by hand on an Apple-Silicon Mac. It drives the real native-MLX `lens`
//! engine via `gen_core::load("lens")` with a **Q4** `LoadSpec` pointed at the packed `q4/` turnkey
//! subdir — the exact runtime seam `generate_stream` uses (minus the API/job plumbing) when the router
//! routes a base `lens` MLX job (`standard_tier_subdir` → the `q4/` subdir; `resolve_quant` →
//! `Quant::Q4`).
//!
//! Purpose: on-device evidence that the recovered SceneWorks/lens-mlx pre-quantized q4 turnkey loads
//! through the worker packed path (`mlx.standardTierLayout` → `standard_tier_subdir` resolves `q4/`) and
//! renders a non-degenerate image. Base Lens packs BOTH the transformer DiT and the gpt-oss-20b MoE text
//! encoder per-tier (it is NOT a dense-TE model), so the q4 load-quant is a harmless no-op on the
//! already-packed weights. The base differs from lens_turbo only in its 20-step / CFG 5.0 defaults.
//!
//! Setup — point at the published turnkey `SceneWorks/lens-mlx` (the worker default). With the q4 tier
//! already in the HF cache, no env is needed: the smoke auto-resolves the cached snapshot's `q4/` subdir
//! (the same selection `image_jobs::base::standard_tier_subdir` makes for `mlxQuantize: 4`). Override
//! `LENS_BASE_Q4_DIR` to point at a snapshot root or a `q4/`-bearing dir directly. Env keys are
//! `LENS_BASE_*`-prefixed so this smoke and `lens_turbo_q4_mlx_smoke` (its `LENS_TURBO_*` sibling) can
//! share one `--ignored` run without bleeding each other's step count / out dir (sc-8924).
//! ```text
//! # optional: LENS_BASE_Q4_DIR=/path/to/lens-mlx  (root containing q4/, or the q4/ dir itself)
//! # optional: LENS_BASE_STEPS=20 LENS_BASE_W=1024 LENS_BASE_H=1024 LENS_BASE_PROMPT="..." LENS_BASE_OUT_DIR=/tmp/lens_base_q4_smoke
//! cargo test -p sceneworks-worker --release lens_base_q4_mlx_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use gen_core::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};

use super::smoke_support::{
    env_or, image_mean, image_std, is_all_zero, save_png, DEGENERATE_STD_FLOOR_LENS,
};

/// The engine-complete packed subdir to load: mirror `image_jobs::base::standard_tier_subdir`'s q4
/// selection — prefer `<root>/q4` (lens turnkeys pack the backbone under `transformer/`), else `root`
/// itself if it already *is* a q4 root. Errors loud if neither resolves so a half-download surfaces as
/// a clear failure rather than a confusing engine load error.
fn resolve_q4_dir(root: &Path) -> PathBuf {
    let is_engine_root = |d: &Path| {
        d.join("transformer/diffusion_pytorch_model.safetensors")
            .is_file()
    };
    let q4 = root.join("q4");
    if is_engine_root(&q4) {
        return q4;
    }
    assert!(
        is_engine_root(root),
        "LENS_BASE_Q4_DIR must point at the turnkey root (containing q4/) or a q4/ dir with a packed \
         transformer/diffusion_pytorch_model.safetensors; neither found under {}",
        root.display()
    );
    root.to_path_buf()
}

/// Auto-discover the cached `SceneWorks/lens-mlx` turnkey snapshot in the HF hub cache, returning the
/// snapshot whose `q4/` subdir carries the packed transformer. `None` if the q4 tier hasn't been pulled
/// (the smoke then panics with the `LENS_BASE_Q4_DIR` hint).
fn cached_turnkey_root() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        PathBuf::from(home).join(".cache/huggingface/hub/models--SceneWorks--lens-mlx/snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let dir = e.path();
            dir.join("q4/transformer/diffusion_pytorch_model.safetensors")
                .is_file()
                .then_some(dir)
        })
}

#[test]
#[ignore = "real-weight MLX smoke; needs the SceneWorks/lens-mlx q4 turnkey cached + an Apple-Silicon Mac"]
fn lens_base_q4_mlx_gpu_smoke() {
    let root = match std::env::var("LENS_BASE_Q4_DIR") {
        Ok(p) if !p.trim().is_empty() => PathBuf::from(p.trim()),
        _ => cached_turnkey_root().unwrap_or_else(|| {
            panic!(
                "no cached SceneWorks/lens-mlx q4 turnkey found; download it via the manifest \
                 (`hf download SceneWorks/lens-mlx --include 'q4/*'`) or set LENS_BASE_Q4_DIR to the \
                 turnkey root (containing q4/)"
            )
        }),
    };
    let q4_dir = resolve_q4_dir(&root);

    // Per-test env prefix `LENS_BASE_*` (sc-8924): previously this smoke and lens_turbo_q4 shared the
    // bare `LENS_OUT_DIR` / `LENS_STEPS` / `LENS_W` / `LENS_H` / `LENS_PROMPT` keys, so co-running both
    // under one `--ignored` invocation bled the turbo smoke's 4-step config into this 20-step base run.
    let out_dir = PathBuf::from(env_or("LENS_BASE_OUT_DIR", "/tmp/lens_base_q4_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let steps: u32 = env_or("LENS_BASE_STEPS", "20")
        .parse()
        .expect("LENS_BASE_STEPS");
    let w: u32 = env_or("LENS_BASE_W", "1024").parse().expect("LENS_BASE_W");
    let h: u32 = env_or("LENS_BASE_H", "1024").parse().expect("LENS_BASE_H");
    let prompt = env_or(
        "LENS_BASE_PROMPT",
        "a photorealistic portrait of a red fox sitting in a sunlit autumn forest, sharp focus, \
         shallow depth of field",
    );

    // Same seam as the worker's MLX image path: a registry load of the base `lens` generator (the
    // worker-crate force-link anchor `use mlx_gen_lens as _;` keeps it registered) + a Q4 `LoadSpec`
    // pointed at the packed q4 turnkey subdir. The packed weights auto-detect their quant, so
    // `with_quant(Q4)` matches the manifest's `mlx.quantize: 4` tier.
    println!(
        "[smoke] loading lens (base, Q4) from {} ...",
        q4_dir.display()
    );
    let spec = LoadSpec::new(WeightsSource::Dir(q4_dir.clone())).with_quant(Quant::Q4);
    let generator = gen_core::load("lens", &spec).expect("load mlx lens (base) generator");

    // Base Lens: non-distilled 20-step, CFG 5.0.
    let req = GenerationRequest {
        prompt: prompt.clone(),
        width: w,
        height: h,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        guidance: Some(5.0),
        ..Default::default()
    };
    println!("[smoke] rendering {w}x{h} @ {steps} steps ...");
    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |p| {
            let s = format!("{p:?}");
            if s != last {
                println!("[progress] {s}");
                last = s;
            }
        })
        .expect("lens (base) generate");
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };

    let mean = image_mean(&image);
    let std = image_std(&image);
    let all_zero = is_all_zero(&image);
    let png = out_dir.join(format!("lens_base_q4_{steps}step.png"));
    save_png(&image, &png);
    println!(
        "[smoke] lens (base) Q4 {}x{} mean {:.2} std {:.2} all_zero={} -> {}",
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
    assert!(
        !all_zero,
        "lens (base) Q4 decode is ALL-ZERO — a broken packed load/decode"
    );
    assert!(
        std > DEGENERATE_STD_FLOOR_LENS,
        "lens (base) Q4 render looks degenerate (std {std:.2}) — possible NaN / all-black / flat decode"
    );
    println!(
        "[smoke] DONE: lens (base) Q4 render coherent (mean {mean:.2}, std {std:.2}, NOT all-zero) \
         at {steps} steps through the worker packed lane"
    );
}
