//! Local real-weight MLX smoke for the Mochi 1 **quant-matrix** worker lane (epic 1788 / sc-11992).
//! `#[ignore]`d — run by hand on an Apple-Silicon Mac. It drives the real native-MLX `mochi_1` engine
//! via `crate::inference_runtime::load(id)` with a per-tier `LoadSpec` pointed at the tier subdir —
//! the exact runtime seam `generate_mochi` uses (minus the API/job plumbing).
//!
//! Purpose: on-device evidence that the `SceneWorks/mochi-1-mlx` pre-quantized turnkey loads through
//! the worker tier path and renders a non-degenerate CLIP at every downloaded tier. Mochi ships its
//! AsymmDiT pre-quantized per tier (`q4/`/`q8/`/`bf16/`), with the T5-XXL text encoder + AsymmVAE
//! SHARED as siblings of the tier dir — so this also covers the thing most likely to break in the
//! worker: that `WeightsSource::Dir` points at the TIER dir while the provider resolves the shared
//! components from that dir's PARENT (`resolve_component_root`).
//!
//! It additionally asserts the three worker-owned facts the unit tests can only assert in the
//! abstract: the tier dir the resolver picks really loads, `mochi_frame_count`'s `6k+1` lattice is
//! what the engine's `validate_request` accepts (the Wan stride lands OFF that lattice for the
//! shipped 5 s default and is hard-rejected — see `frames_are_the_mochi_lattice_not_wans`), and that
//! progress is monotonic, reaches the decode, and terminates.
//!
//! Setup — point at the published turnkey (the worker default) or a local staging dir:
//! ```text
//! hf download SceneWorks/mochi-1-mlx --include 'q4/*' 'text_encoder/*' 'tokenizer/*' 'vae/*'
//! # or point at any model root laid out as <root>/{q4,q8,bf16}/ + {text_encoder,tokenizer,vae}/
//! export MOCHI_MODEL_DIR=/path/to/mochi-root
//! cargo test -p sceneworks-worker --release mochi_mlx_gpu_smoke -- --ignored --nocapture
//! ```
//! Optional overrides: `MOCHI_W`/`MOCHI_H` (default 848x480 — Mochi's only trained bucket; both axes
//! must divide by 16), `MOCHI_FRAMES` (default 7 = `6·1+1`, the cheapest on-lattice clip — the
//! untiled AsymmVAE decode peak grows LINEARLY in frames, ~60 GiB at the shipped 151-frame default,
//! so the smoke deliberately renders the shortest legal clip), `MOCHI_STEPS` (default 4 — this is a
//! load/decode smoke, not a quality bar), `MOCHI_OUT_DIR` (default `/tmp/mochi_mlx_smoke`).

use std::path::{Path, PathBuf};

use gen_core::{GenerationOutput, GenerationRequest, LoadSpec, Progress, Quant, WeightsSource};

use super::smoke_support::{
    env_or, image_mean, image_std, is_all_zero, mean_abs_frame_delta, save_png,
    DEGENERATE_STD_FLOOR_DEFAULT,
};

/// The engine id BOTH backends register (no `_distilled`-style split).
const MOCHI_ENGINE_ID: &str = "mochi_1";

/// The three matrix tiers + the load `Quant` each asserts. Mirrors `video_jobs::mochi_tier_quant`,
/// which derives this from the tier's `split_model.json`: q4→Q4, q8→Q8, bf16→dense/None. The provider
/// only ASSERTS this against the tier's manifest (a mismatch is a hard load error) — it never
/// re-quantizes, so a wrong value here fails loud rather than silently rendering the wrong tier.
const TIERS: &[(&str, Option<Quant>)] = &[
    ("q4", Some(Quant::Q4)),
    ("q8", Some(Quant::Q8)),
    ("bf16", None),
];

/// The shared components the provider resolves from the tier dir's PARENT (the A6 sibling layout).
const SHARED_COMPONENTS: &[&str] = &["text_encoder", "tokenizer", "vae"];

/// Motion floor: the minimum [`mean_abs_frame_delta`] (0–255 scale) every CONSECUTIVE frame pair of a
/// live Mochi clip must clear (sc-11992 review).
///
/// Why this exists: a FROZEN clip — the 6× temporal path collapsed to N copies of one still — scores
/// exactly 0.0 here yet passes every other assertion this smoke makes (frame count, dimensions,
/// non-all-zero, per-pixel std), because each individual frame is a perfectly good image. The temporal
/// path IS Mochi's distinguishing feature, so a smoke that cannot tell a moving clip from a frozen one
/// is not evidence the lane works.
///
/// Why the per-PAIR minimum and not just `first` vs `last`: the pair minimum is strictly stronger. A
/// clip that moves once and then freezes for the rest of the run still shows a large first→last span,
/// but its frozen stretch drives a consecutive pair to ~0. Requiring EVERY adjacent pair to clear the
/// floor asserts the temporal path stayed alive for the whole clip, not just at its ends.
///
/// Calibration — the observed 848×480 × 7f @ 4-step run on this Mac (seed 42), per-pair deltas:
///   q4   [7.58, 5.58, 3.60, 3.41, 2.60, 5.21]  → min 2.597 (first→last span 14.65)
///   q8   [30.04, 6.11, 2.57, 7.23, 14.93, 11.89] → min 2.573 (span 20.41)
///   bf16 [30.87, 6.47, 2.75, 7.19, 16.24, 12.67] → min 2.752 (span 19.90)
/// The tightest real observation across all three tiers is 2.573. 1.0 sits ~2.6× below it — enough
/// margin that a different prompt/step count/seed (all env-overridable here) will not flake — while a
/// frozen or near-frozen pair (0.0, or a sub-1/255 average pixel change) cannot reach it. This is a
/// structural liveness floor, deliberately NOT a motion-quality bar: the saved-PNG eyeball remains the
/// quality call, exactly like `DEGENERATE_STD_FLOOR_DEFAULT`.
const MOTION_MIN_PAIR_DELTA_FLOOR: f64 = 1.0;

/// Whether `root` is a Mochi model root: the shared co-requisite plus at least one complete tier.
fn is_model_root(root: &Path) -> bool {
    SHARED_COMPONENTS
        .iter()
        .all(|component| root.join(component).is_dir())
        && TIERS
            .iter()
            .any(|(tier, _)| tier_is_complete(&root.join(tier)))
}

/// Whether `dir` is a loadable tier dir (mirrors `video_jobs::mochi_tier_dir_is_complete`).
fn tier_is_complete(dir: &Path) -> bool {
    dir.join("split_model.json").is_file()
        && dir.join("transformer").join("model.safetensors").is_file()
}

/// The Mochi model root: `$MOCHI_MODEL_DIR`, else the cached `SceneWorks/mochi-1-mlx` snapshot.
fn model_root() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("MOCHI_MODEL_DIR") {
        let root = PathBuf::from(dir.trim());
        if is_model_root(&root) {
            return Some(root);
        }
        println!(
            "[smoke] $MOCHI_MODEL_DIR={} is not a complete Mochi model root; falling back to the HF cache",
            root.display()
        );
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--SceneWorks--mochi-1-mlx/snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .find(|dir| is_model_root(dir))
}

#[test]
#[ignore = "real-weight MLX smoke; needs the SceneWorks/mochi-1-mlx tiers cached + an Apple-Silicon Mac"]
fn mochi_mlx_gpu_smoke() {
    let out_dir = PathBuf::from(env_or("MOCHI_OUT_DIR", "/tmp/mochi_mlx_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let w: u32 = env_or("MOCHI_W", "848").parse().expect("MOCHI_W");
    let h: u32 = env_or("MOCHI_H", "480").parse().expect("MOCHI_H");
    let frames: u32 = env_or("MOCHI_FRAMES", "7").parse().expect("MOCHI_FRAMES");
    let steps: u32 = env_or("MOCHI_STEPS", "4").parse().expect("MOCHI_STEPS");
    let prompt = "a calico kitten padding through tall sunlit grass, shallow depth of field";

    assert_eq!(
        frames % 6,
        1,
        "MOCHI_FRAMES must sit on the 6k+1 lattice the engine's validate_request enforces \
         (mochi_frame_count snaps to this); got {frames}"
    );
    assert!(
        w % 16 == 0 && h % 16 == 0,
        "Mochi requires width/height divisible by 16; got {w}x{h}"
    );

    let Some(root) = model_root() else {
        panic!(
            "no Mochi model root found — set $MOCHI_MODEL_DIR to a dir laid out as \
             <root>/{{q4,q8,bf16}}/ + {{text_encoder,tokenizer,vae}}/, or pull the turnkey: \
             hf download SceneWorks/mochi-1-mlx --include 'q4/*' 'text_encoder/*' 'tokenizer/*' 'vae/*'"
        );
    };
    println!("[smoke] Mochi model root: {}", root.display());

    let mut verified: Vec<String> = Vec::new();
    for (tier, quant) in TIERS {
        let tier_dir = root.join(tier);
        if !tier_is_complete(&tier_dir) {
            println!("[smoke] mochi_1 {tier} not downloaded — skipping");
            continue;
        }

        // The worker seam: `WeightsSource::Dir` points at the TIER dir, and the provider resolves the
        // shared T5/tokenizer/VAE from its PARENT. `with_quant` mirrors `mochi_tier_quant` — advisory
        // in the sense that the tier dir dictates the real precision, but ASSERTED by the loader.
        let mut spec = LoadSpec::new(WeightsSource::Dir(tier_dir.clone()));
        if let Some(q) = quant {
            spec = spec.with_quant(*q);
        }
        println!(
            "[smoke] loading mochi_1 ({tier}) from {} ...",
            tier_dir.display()
        );
        let generator = crate::inference_runtime::load(MOCHI_ENGINE_ID, &spec)
            .expect("load mlx mochi generator for tier");

        let req = GenerationRequest {
            prompt: prompt.to_owned(),
            width: w,
            height: h,
            count: 1,
            frames: Some(frames),
            fps: Some(30),
            seed: Some(42),
            steps: Some(steps),
            guidance: Some(4.5),
            ..Default::default()
        };
        println!("[smoke] rendering {w}x{h} x {frames}f @ {steps} steps (mochi_1 {tier}) ...");

        // Progress must be MONOTONIC and reach the decode — the job lane has NO background heartbeat
        // during a generation, so this callback IS the keepalive; a silent engine would stall the job.
        let mut last_step = 0_u32;
        let mut saw_decoding = false;
        let output = generator
            .generate(&req, &mut |progress| match progress {
                Progress::Step { current, total } => {
                    assert!(
                        current >= last_step,
                        "progress went BACKWARDS: {current} after {last_step}"
                    );
                    assert!(current <= total, "step {current} exceeds total {total}");
                    last_step = current;
                }
                Progress::Decoding => saw_decoding = true,
                Progress::Loading(_) => {}
            })
            .expect("mochi generate");
        assert!(last_step > 0, "engine emitted no denoise Step progress");
        assert!(saw_decoding, "engine never reported the Decoding phase");

        let GenerationOutput::Video {
            frames: rendered,
            fps,
            audio,
        } = output
        else {
            panic!("expected Video output from a video generator");
        };
        assert!(
            audio.is_none(),
            "mochi has no audio branch — a Some(audio) means the wrong engine answered"
        );
        assert_eq!(
            rendered.len(),
            frames as usize,
            "mochi_1 {tier} returned {} frames for a {frames}-frame request",
            rendered.len()
        );
        assert_eq!(fps, 30, "mochi renders ~30 fps");

        // Every frame must be a real decode, and the clip must actually MOVE — a frozen clip would
        // pass a per-frame std check while proving the temporal path is broken.
        for (index, frame) in rendered.iter().enumerate() {
            assert_eq!(
                (frame.width, frame.height),
                (w, h),
                "mochi_1 {tier} frame {index} has the wrong dimensions"
            );
            assert!(
                !is_all_zero(frame),
                "mochi_1 {tier} frame {index} is ALL-ZERO — a broken tier load or decode"
            );
            let std = image_std(frame);
            assert!(
                std > DEGENERATE_STD_FLOOR_DEFAULT,
                "mochi_1 {tier} frame {index} looks degenerate (std {std:.2}) — possible NaN / \
                 all-black / flat decode"
            );
        }
        // The clip must actually MOVE. Every per-frame check above is structurally blind to a frozen
        // clip (N copies of one good still), and the 6× temporal path is the thing this lane exists to
        // prove — so assert a real motion floor on EVERY consecutive pair, not just the ends. A frozen
        // pair scores exactly 0.0. See `MOTION_MIN_PAIR_DELTA_FLOOR` for the calibration.
        let pair_deltas: Vec<f64> = rendered
            .windows(2)
            .map(|pair| mean_abs_frame_delta(&pair[0], &pair[1]))
            .collect();
        let span = mean_abs_frame_delta(&rendered[0], &rendered[rendered.len() - 1]);
        println!(
            "[smoke] mochi_1 {tier} {w}x{h} x{} frames, mean {:.2} -> {:.2}, std {:.2}",
            rendered.len(),
            image_mean(&rendered[0]),
            image_mean(&rendered[rendered.len() - 1]),
            image_std(&rendered[0])
        );
        println!(
            "[smoke-motion] mochi_1 {tier} pair_deltas={:?} span={span:.3}",
            pair_deltas
                .iter()
                .map(|d| format!("{d:.3}"))
                .collect::<Vec<_>>()
        );
        for (index, delta) in pair_deltas.iter().enumerate() {
            assert!(
                *delta > MOTION_MIN_PAIR_DELTA_FLOOR,
                "mochi_1 {tier} frames {index}->{} are FROZEN (mean abs delta {delta:.3} <= floor \
                 {MOTION_MIN_PAIR_DELTA_FLOOR}) — the temporal path is broken; per-pair deltas \
                 {pair_deltas:?}",
                index + 1
            );
        }

        for (index, frame) in rendered.iter().enumerate() {
            save_png(
                frame,
                &out_dir.join(format!("mochi_1_{tier}_{index:03}.png")),
            );
        }
        verified.push(format!("mochi_1:{tier}"));
    }

    assert!(
        !verified.is_empty(),
        "no Mochi tiers were verified — download at least one tier plus the shared components \
         (hf download SceneWorks/mochi-1-mlx --include 'q4/*' 'text_encoder/*' 'tokenizer/*' 'vae/*')"
    );
    println!(
        "[smoke] DONE: Mochi renders a coherent clip through the worker tier lane for [{}]",
        verified.join(", ")
    );
}
