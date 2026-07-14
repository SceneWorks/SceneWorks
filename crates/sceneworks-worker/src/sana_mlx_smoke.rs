//! Local real-weight MLX smoke for the SANA **quant-matrix** worker lane (sc-8489/sc-8513, epic 8506
//! Group-B). `#[ignore]`d — run by hand on an Apple-Silicon Mac. It drives the real native-MLX
//! `sana_1600m` + `sana_sprint_1600m` engines via `crate::inference_runtime::load(id)` with a per-tier `LoadSpec`
//! pointed at the packed/dense turnkey subdir — the exact runtime seam `generate_stream` uses (minus
//! the API/job plumbing) when the router routes a SANA MLX job (`standard_tier_subdir` → the
//! `q4/`|`q8/`|`bf16/` subdir; `resolve_quant` → `Quant::Q4`|`Quant::Q8`|`None`).
//!
//! Purpose: on-device evidence that the `SceneWorks/Sana_1600M_1024px_mlx` /
//! `SceneWorks/Sana_Sprint_1.6B_1024px_mlx` pre-quantized turnkeys load through the worker packed
//! path (`STANDARD_TIER_MODELS` → `standard_tier_subdir` resolves the tier) and render a
//! non-degenerate image at EVERY downloaded tier. SANA packs the Linear-DiT transformer + the Gemma-2
//! CHI text encoder per-tier (the DC-AE VAE stays dense); the q4/q8 load-quant is a harmless no-op on
//! the already-packed weights (packed-detected from `{base}.scales`) and bf16 loads dense
//! (`Quant::None`), so this covers both the packed and dense load mechanisms.
//!
//! Setup — point at the published turnkeys (the worker defaults). With the tier subdirs in the HF
//! cache the smoke auto-resolves each cached snapshot's `q4/`|`q8/`|`bf16/` subdir and verifies every
//! tier it finds (a partial download verifies whatever is present). Pull them via the manifest, e.g.
//! ```text
//! hf download SceneWorks/Sana_1600M_1024px_mlx      --include 'q4/*' 'q8/*' 'bf16/*'
//! hf download SceneWorks/Sana_Sprint_1.6B_1024px_mlx --include 'q4/*' 'q8/*' 'bf16/*'
//! cargo test -p sceneworks-worker --release sana_mlx_gpu_smoke -- --ignored --nocapture
//! ```
//! Optional overrides: `SANA_W`/`SANA_H` (default 512, a 32-multiple within the DC-AE 256..1024
//! envelope), `SANA_OUT_DIR` (default `/tmp/sana_mlx_smoke`).

use std::path::PathBuf;

use gen_core::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};

use super::smoke_support::{
    env_or, image_mean, image_std, is_all_zero, save_png, DEGENERATE_STD_FLOOR_DEFAULT,
};

/// A SANA model to smoke: its registry id, its cache repo folder, and its per-render recipe. Base is
/// true-CFG 20-step guidance 4.5; Sprint is the CFG-free ~2-step SCM distillation.
struct SanaModel {
    id: &'static str,
    repo_folder: &'static str,
    steps: u32,
    guidance: f32,
}

const MODELS: &[SanaModel] = &[
    SanaModel {
        id: "sana_1600m",
        repo_folder: "models--SceneWorks--Sana_1600M_1024px_mlx",
        steps: 20,
        guidance: 4.5,
    },
    SanaModel {
        id: "sana_sprint_1600m",
        repo_folder: "models--SceneWorks--Sana_Sprint_1.6B_1024px_mlx",
        steps: 2,
        guidance: 4.5,
    },
];

/// The three matrix tiers + the load `Quant` each maps to — mirroring `image_jobs::base`'s
/// `standard_tier_subdir` (subdir) + `resolve_quant` (load quant): q4→Q4, q8→Q8, bf16→dense/None.
const TIERS: &[(&str, Option<Quant>)] = &[
    ("q4", Some(Quant::Q4)),
    ("q8", Some(Quant::Q8)),
    ("bf16", None),
];

/// The cached snapshot dir for `repo_folder` that carries a loadable `<tier>/transformer/` — SANA
/// turnkeys pack the backbone under `transformer/diffusion_pytorch_model.safetensors`. `None` if the
/// repo isn't cached at all.
fn cached_snapshot(repo_folder: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo_folder)
        .join("snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|dir| {
            TIERS.iter().any(|(t, _)| {
                dir.join(t)
                    .join("transformer/diffusion_pytorch_model.safetensors")
                    .is_file()
            })
        })
}

#[test]
#[ignore = "real-weight MLX smoke; needs the SceneWorks SANA q4/q8/bf16 turnkeys cached + an Apple-Silicon Mac"]
fn sana_mlx_gpu_smoke() {
    let out_dir = PathBuf::from(env_or("SANA_OUT_DIR", "/tmp/sana_mlx_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");
    let w: u32 = env_or("SANA_W", "512").parse().expect("SANA_W");
    let h: u32 = env_or("SANA_H", "512").parse().expect("SANA_H");
    let prompt =
        "a photorealistic portrait of a red fox sitting in a sunlit autumn forest, sharp focus";

    // Verified (model, tier) pairs — the run must cover at least one so a stale/empty cache fails loud.
    let mut verified: Vec<String> = Vec::new();

    for model in MODELS {
        let Some(snapshot) = cached_snapshot(model.repo_folder) else {
            println!(
                "[smoke] {} — no cached tiers found ({}); skipping (pull via `hf download`)",
                model.id, model.repo_folder
            );
            continue;
        };
        for (tier, quant) in TIERS {
            let tier_dir = snapshot.join(tier);
            if !tier_dir
                .join("transformer/diffusion_pytorch_model.safetensors")
                .is_file()
            {
                println!("[smoke] {} {tier} not downloaded — skipping", model.id);
                continue;
            }

            // Same seam as the worker MLX path: catalog load of the SANA generator + a `LoadSpec`
            // pointed at the tier subdir. Packed tiers
            // (q4/q8) carry `.scales` and auto-detect; bf16 loads dense. `with_quant` mirrors
            // `resolve_quant` (advisory post-#653 — the on-disk tier dictates the real precision).
            let mut spec = LoadSpec::new(WeightsSource::Dir(tier_dir.clone()));
            if let Some(q) = quant {
                spec = spec.with_quant(*q);
            }
            println!(
                "[smoke] loading {} ({tier}) from {} ...",
                model.id,
                tier_dir.display()
            );
            let generator = crate::inference_runtime::load(model.id, &spec)
                .expect("load mlx sana generator for tier");

            let req = GenerationRequest {
                prompt: prompt.to_owned(),
                width: w,
                height: h,
                count: 1,
                seed: Some(42),
                steps: Some(model.steps),
                guidance: Some(model.guidance),
                ..Default::default()
            };
            println!(
                "[smoke] rendering {}x{} @ {} steps ({}) ...",
                w, h, model.steps, model.id
            );
            let output = generator
                .generate(&req, &mut |_p| {})
                .expect("sana generate");
            let image = match output {
                GenerationOutput::Images(mut images) => {
                    images.pop().expect("engine returned no image")
                }
                other => panic!("expected Images output, got {other:?}"),
            };

            let mean = image_mean(&image);
            let std = image_std(&image);
            let all_zero = is_all_zero(&image);
            let png = out_dir.join(format!("{}_{tier}.png", model.id));
            save_png(&image, &png);
            println!(
                "[smoke] {} {tier} {}x{} mean {:.2} std {:.2} all_zero={} -> {}",
                model.id,
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
                "{} {tier} returned the wrong dimensions",
                model.id
            );
            assert!(
                !all_zero,
                "{} {tier} decode is ALL-ZERO — a broken packed/dense load or decode",
                model.id
            );
            assert!(
                std > DEGENERATE_STD_FLOOR_DEFAULT,
                "{} {tier} render looks degenerate (std {std:.2}) — possible NaN / all-black / flat decode",
                model.id
            );
            verified.push(format!("{}:{tier}", model.id));
        }
    }

    assert!(
        !verified.is_empty(),
        "no SANA tiers were verified — download the SceneWorks/Sana_*_mlx q4/q8/bf16 turnkeys first \
         (hf download SceneWorks/Sana_1600M_1024px_mlx --include 'q4/*' 'q8/*' 'bf16/*')"
    );
    println!(
        "[smoke] DONE: SANA quant-matrix renders coherent through the worker packed lane for [{}]",
        verified.join(", ")
    );
}
