//! Local real-weight MLX smoke for the SenseNova-U1 `_fast` worker lane (sc-17396).
//! `#[ignore]`d — run by hand on an Apple-Silicon Mac. It drives the real native-MLX
//! `sensenova_u1_8b_fast` engine via `crate::inference_runtime::load("sensenova_u1_8b_fast")` with a
//! **Q8** `LoadSpec` pointed at the packed tier subdir — the exact runtime seam
//! `image_jobs::sensenova` reaches through `start_cached_gen_stream` (minus the API/job plumbing).
//!
//! **Why this exists rather than another `load_raw` assertion.** `sensenova_jobs`' packed-tier
//! smokes build the model with `load_sensenova_model`, which replicates the engine's private
//! `load_inner` from public re-exports (`load_raw` + `T2iModel::from_weights`) because the VQA /
//! interleave lanes need a dense, unquantized, LoRA-free model. That replication is correct for
//! those lanes, but it means NO worker test drove the engine's own `load`/`load_fast` — so when the
//! shared memory ladder added a pinned-artifact branch inside `load_inner`, every packed-tier smoke
//! stayed green while every real HF-cached generation failed at load with
//! `backend op failed: Unsupported file format` (sc-17396: `PinnedArtifact::open_weights` opened the
//! canonicalized, EXTENSIONLESS `blobs/<sha>` object, and mlx-rs dispatches the safetensors loader
//! on the path extension). This smoke closes that gap by loading through the registry.
//!
//! It is deliberately pointed at the **HF cache**, not a copied-out directory: the bug only exists
//! when `model.safetensors` is a symlink into `blobs/`, which is exactly what the cache produces and
//! what a fixture with real regular files cannot reproduce.
//!
//! Setup — with a packed tier of `SceneWorks/sensenova-u1-8b-fast-mlx` already in the HF cache no env
//! is needed; the smoke auto-resolves the cached snapshot's tier subdir (preferring `q4/`, falling
//! back to `q8/`, matching `image_jobs::base::standard_tier_subdir`'s clamp to what is installed).
//! ```text
//! # optional: SENSENOVA_FAST_DIR=/path/to/sensenova-u1-8b-fast-mlx  (root containing q4//q8/, or the tier dir itself)
//! # optional: SENSENOVA_FAST_STEPS=8 SENSENOVA_FAST_W=512 SENSENOVA_FAST_H=512 SENSENOVA_FAST_PROMPT="..." SENSENOVA_FAST_OUT_DIR=/tmp/sensenova_fast_q8_smoke
//! cargo test -p sceneworks-worker --release sensenova_fast_q8_mlx_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use gen_core::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};

use super::smoke_support::{env_or, image_mean, image_std, is_all_zero, save_png};

/// The packed tier subdir to load. SenseNova-U1 is a UNIFIED turnkey: the whole backbone is a flat
/// `model.safetensors` directly in the tier dir (no `transformer/` component), so probe the tier root
/// itself. Prefer `q4/`, fall back to `q8/`, else accept `root` when it already IS a tier dir.
fn resolve_packed_tier_dir(root: &Path) -> PathBuf {
    let is_tier = |dir: &Path| dir.join("model.safetensors").is_file();
    for tier in ["q4", "q8"] {
        let dir = root.join(tier);
        if is_tier(&dir) {
            return dir;
        }
    }
    assert!(
        is_tier(root),
        "SENSENOVA_FAST_DIR must point at the turnkey root (containing q4/ or q8/) or a tier dir \
         with a packed model.safetensors; neither found under {}",
        root.display()
    );
    root.to_path_buf()
}

/// Auto-discover the cached `SceneWorks/sensenova-u1-8b-fast-mlx` turnkey snapshot in the HF hub
/// cache. `None` if no packed tier has been pulled (the smoke then panics with a download hint).
///
/// Returning the cache path is the POINT (see the module doc): the snapshot's `model.safetensors` is
/// a symlink into `blobs/`, and that shape is what sc-17396 regressed on.
fn cached_turnkey_root() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--SceneWorks--sensenova-u1-8b-fast-mlx/snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|entry| entry.ok())
        .find_map(|entry| {
            let dir = entry.path();
            ["q4", "q8"]
                .iter()
                .any(|tier| dir.join(tier).join("model.safetensors").is_file())
                .then_some(dir)
        })
}

#[test]
#[ignore = "real-weight MLX smoke; needs a SceneWorks/sensenova-u1-8b-fast-mlx packed tier cached + an Apple-Silicon Mac"]
fn sensenova_fast_q8_mlx_gpu_smoke() {
    let root = match std::env::var("SENSENOVA_FAST_DIR") {
        Ok(path) if !path.trim().is_empty() => PathBuf::from(path.trim()),
        _ => cached_turnkey_root().unwrap_or_else(|| {
            panic!(
                "no cached SceneWorks/sensenova-u1-8b-fast-mlx packed tier found; download one \
                 (`hf download SceneWorks/sensenova-u1-8b-fast-mlx --include 'q8/*'`) or set \
                 SENSENOVA_FAST_DIR to the turnkey root (containing q4/ or q8/)"
            )
        }),
    };
    let tier_dir = resolve_packed_tier_dir(&root);

    // The regression this smoke exists for is invisible unless the checkpoint entry is a symlink into
    // an extensionless HF blob. Say so out loud when it is not, so a green run against a copied-out
    // directory is never mistaken for coverage of the sc-17396 lane.
    let checkpoint = tier_dir.join("model.safetensors");
    let canonical = std::fs::canonicalize(&checkpoint).expect("canonicalize the packed checkpoint");
    println!(
        "[smoke] checkpoint {} -> {} (symlink={}, canonical extension={:?})",
        checkpoint.display(),
        canonical.display(),
        checkpoint.is_symlink(),
        canonical.extension()
    );
    if !checkpoint.is_symlink() || canonical.extension().is_some() {
        println!(
            "[smoke] NOTE: this tier is a regular file, so it does NOT exercise the sc-17396 \
             extensionless-HF-blob path; run against the HF cache for that coverage"
        );
    }

    let out_dir = PathBuf::from(env_or(
        "SENSENOVA_FAST_OUT_DIR",
        "/tmp/sensenova_fast_q8_smoke",
    ));
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let steps: u32 = env_or("SENSENOVA_FAST_STEPS", "8")
        .parse()
        .expect("SENSENOVA_FAST_STEPS");
    // Multiples of `CELL` (32) per side, per the engine's `validate_dims_and_steps`.
    let w: u32 = env_or("SENSENOVA_FAST_W", "512")
        .parse()
        .expect("SENSENOVA_FAST_W");
    let h: u32 = env_or("SENSENOVA_FAST_H", "512")
        .parse()
        .expect("SENSENOVA_FAST_H");
    let prompt = env_or(
        "SENSENOVA_FAST_PROMPT",
        "a single red circle on a white background, flat vector illustration",
    );

    println!(
        "[smoke] loading sensenova_u1_8b_fast (Q8) from {} ...",
        tier_dir.display()
    );
    let spec = LoadSpec::new(WeightsSource::Dir(tier_dir.clone())).with_quant(Quant::Q8);
    let generator = crate::inference_runtime::load("sensenova_u1_8b_fast", &spec)
        .expect("load mlx sensenova_u1_8b_fast generator");

    // Distilled defaults: 8 NFE at CFG 1.0 (the `_fast` variant's manifest defaults). No `true_cfg`
    // — that is image guidance and the engine rejects it without Reference conditioning.
    let req = GenerationRequest {
        prompt: prompt.clone(),
        width: w,
        height: h,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        guidance: Some(1.0),
        ..Default::default()
    };
    println!("[smoke] rendering {w}x{h} @ {steps} steps ...");
    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |progress| {
            let rendered = format!("{progress:?}");
            if rendered != last {
                println!("[progress] {rendered}");
                last = rendered;
            }
        })
        .expect("sensenova_u1_8b_fast generate");
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };

    let mean = image_mean(&image);
    let std = image_std(&image);
    let all_zero = is_all_zero(&image);
    let png = out_dir.join(format!("sensenova_fast_{steps}step.png"));
    save_png(&image, &png);
    println!(
        "[smoke] sensenova_u1_8b_fast {}x{} mean {:.2} std {:.2} all_zero={} -> {}",
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
        "sensenova_u1_8b_fast decode is ALL-ZERO — a broken packed load/decode"
    );
    // The `sensenova_jobs` packed-tier smokes use a 10.0 floor for this same distilled 8-step render;
    // hold to it rather than the shared 5.0 default, which a bad distill merge could still clear.
    assert!(
        std > 10.0,
        "sensenova_u1_8b_fast render looks degenerate (std {std:.2}, mean {mean:.2}) — possible NaN \
         / all-black / flat decode or a bad distill merge"
    );
    println!(
        "[smoke] DONE: sensenova_u1_8b_fast render coherent (mean {mean:.2}, std {std:.2}, NOT \
         all-zero) at {steps} steps through the worker registry lane"
    );
}
