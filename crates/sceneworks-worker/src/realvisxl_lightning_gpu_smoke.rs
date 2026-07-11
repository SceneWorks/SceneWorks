//! Local real-weight GPU smoke for the candle RealVisXL Lightning worker lane (sc-7176, the worker
//! half of sc-6128). `#[ignore]`d — run by hand on the RTX PRO 6000. It drives the real candle SDXL
//! engine via `gen_core::load("sdxl")` — the same runtime seam `generate_candle_stream` uses, minus
//! the API/job plumbing — with the **forced** few-step `lightning` (Euler-trailing, CFG-off) sampler
//! the worker pins for `realvisxl_lightning`, against the distilled RealVisXL_V5.0_Lightning weights.
//! This is the end-to-end worker-lane validation that backs the macOnly drop.
//!
//! `REALVISXL_LIGHTNING_DIR` may point at EITHER a dense diffusers snapshot (the fp16 components under a
//! `model_index.json` root) OR a packed **q4/q8 standard tier** dir (the `q4/`/`q8/` subdir of the
//! `SceneWorks/realvisxl-lightning-mlx` turnkey, which carries engine-complete `unet/`+`text_encoder/`+
//! `vae/`). `candle-gen-sdxl::load` packed-detects the tier from disk (sc-9416/9527), so the SAME seam
//! serves both — this is the sc-10812 packed-tier eyeball: the routing flip to `candle_quant_lora` keeps
//! a quant tier-select on candle, and a packed lightning render must stay coherent at 5-step. Point it
//! at the cached `SceneWorks/sdxl-base-mlx` `q8/` tier (with `RVXL_SAMPLER=` unset → engine default,
//! since the non-distilled base isn't a lightning checkpoint) to exercise the shared candle-gen-sdxl
//! packed loader on this box when the lightning turnkey isn't pulled.
//!
//! Setup (PowerShell; the components must be in the HF cache — download via the Model Manager / the
//! manifest `realvisxl_lightning` entry):
//! ```text
//! # dense: the snapshot dir holding model_index.json + unet/ text_encoder/ ... *.fp16.safetensors
//! $env:REALVISXL_LIGHTNING_DIR="C:\Users\Michael\.cache\huggingface\hub\models--SG161222--RealVisXL_V5.0_Lightning\snapshots\<hash>"
//! # OR packed (sc-10812): a q4/q8 tier dir carrying unet/ text_encoder/ vae/
//! # $env:REALVISXL_LIGHTNING_DIR="D:\.cache\huggingface\hub\models--SceneWorks--realvisxl-lightning-mlx\snapshots\<hash>\q8"
//! $env:RVXL_OUT_DIR="D:\sceneworks-sampler-validate\rvxl-lightning"
//! # optional: RVXL_STEPS=5 RVXL_W=1024 RVXL_H=1024 RVXL_PROMPT="..." RVXL_CONTRAST=1 (also render ddim@steps)
//! cargo test -p sceneworks-worker --features backend-candle --release realvisxl_lightning_candle_gpu_smoke -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use gen_core::{GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};

use super::smoke_support::{env_or, image_std, save_png, DEGENERATE_STD_FLOOR_DEFAULT};

fn env_path(key: &str) -> PathBuf {
    // Trim: a cmd `set VAR=value && ...` keeps the trailing space before `&&`.
    PathBuf::from(
        std::env::var(key)
            .unwrap_or_else(|_| panic!("set ${key}"))
            .trim(),
    )
}

fn render(
    generator: &dyn gen_core::Generator,
    prompt: &str,
    w: u32,
    h: u32,
    steps: u32,
    guidance: f32,
    sampler: Option<&str>,
) -> Image {
    // The generator is loaded ONCE by the caller and reused across the lightning + optional ddim
    // contrast render (sc-8925): the SDXL snapshot is multi-GB, so re-loading it per render() (as the
    // old per-call `gen_core::load` did) doubled the load wall-clock + VRAM on the RVXL_CONTRAST=1 run.
    let req = GenerationRequest {
        prompt: prompt.to_owned(),
        width: w,
        height: h,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        guidance: Some(guidance),
        sampler: sampler.map(str::to_owned),
        ..Default::default()
    };
    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |p| {
            let s = format!("{p:?}");
            if s != last {
                println!("[progress {}] {s}", sampler.unwrap_or("default"));
                last = s;
            }
        })
        .expect("sdxl generate");
    match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    }
}

#[test]
#[ignore = "real-weight GPU smoke; needs the candle RealVisXL_V5.0_Lightning diffusers snapshot"]
fn realvisxl_lightning_candle_gpu_smoke() {
    let weights_dir = env_path("REALVISXL_LIGHTNING_DIR");
    // Accept EITHER a dense diffusers snapshot (model_index.json root) OR a packed q4/q8 standard-tier
    // dir (sc-10812: engine-complete unet/+text_encoder/+vae/, no model_index.json). candle-gen-sdxl
    // packed-detects the tier from disk, so both load through the same `gen_core::load("sdxl")` seam.
    let is_diffusers_snapshot = weights_dir.join("model_index.json").is_file();
    let is_packed_tier = weights_dir.join("unet").is_dir()
        && weights_dir.join("text_encoder").is_dir()
        && weights_dir.join("vae").is_dir();
    assert!(
        is_diffusers_snapshot || is_packed_tier,
        "REALVISXL_LIGHTNING_DIR must be a diffusers snapshot (model_index.json) or a packed q4/q8 tier \
         (unet/+text_encoder/+vae/): {}",
        weights_dir.display()
    );
    let out_dir = PathBuf::from(env_or("RVXL_OUT_DIR", "/tmp/rvxl_lightning_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let steps: u32 = env_or("RVXL_STEPS", "5").parse().expect("RVXL_STEPS");
    let w: u32 = env_or("RVXL_W", "1024").parse().expect("RVXL_W");
    let h: u32 = env_or("RVXL_H", "1024").parse().expect("RVXL_H");
    let prompt = env_or(
        "RVXL_PROMPT",
        "a photorealistic portrait of a red fox in a snowy forest, golden hour, sharp focus",
    );

    // Same seam as `generate_candle_stream`: registry load of the candle `sdxl` engine + a dense
    // (no-quant, no-adapter) LoadSpec pointed at the RealVisXL Lightning diffusers components. Loaded
    // ONCE here and reused across both renders (sc-8925) — the multi-GB snapshot is not re-loaded for
    // the RVXL_CONTRAST=1 ddim run.
    let spec = LoadSpec::new(WeightsSource::Dir(weights_dir.to_path_buf()));
    let generator = gen_core::load("sdxl", &spec).expect("load candle sdxl provider");

    // The shipped behavior: realvisxl_lightning forces the few-step `lightning` sampler, CFG-off
    // (guidance 1.0 — the distilled checkpoint is trained CFG-free). Overridable so this harness can also
    // exercise the shared candle-gen-sdxl packed loader against the cached non-distilled `sdxl-base-mlx`
    // q8 tier (sc-10812): `RVXL_SAMPLER=` (empty) → engine default, `RVXL_GUIDANCE=7.5` for real CFG.
    let sampler_name = env_or("RVXL_SAMPLER", "lightning");
    let sampler = (!sampler_name.trim().is_empty()).then_some(sampler_name.trim());
    let guidance: f32 = env_or("RVXL_GUIDANCE", "1.0")
        .parse()
        .expect("RVXL_GUIDANCE");
    println!(
        "[smoke] rendering {} @ {steps} steps ({w}x{h}, guidance {guidance}) ...",
        sampler.unwrap_or("engine-default")
    );
    let lightning = render(&*generator, &prompt, w, h, steps, guidance, sampler);
    let lightning_std = image_std(&lightning);
    save_png(
        &lightning,
        &out_dir.join(format!("lightning_{steps}step.png")),
    );
    println!(
        "[smoke] lightning {}x{} std {:.2} -> {}",
        lightning.width,
        lightning.height,
        lightning_std,
        out_dir.join(format!("lightning_{steps}step.png")).display()
    );

    // Optional contrast: the SAME checkpoint on the default `ddim` schedule (real CFG) at the same low
    // step count is visibly under-denoised — the wrong-schedule result the forced lightning sampler
    // avoids. ddim needs real CFG (guidance > 1; the lightning path is CFG-off, ddim+1.0 is invalid),
    // so render it at the base SDXL default 7.5. Saved for the eyeball only.
    if env_or("RVXL_CONTRAST", "0") == "1" {
        println!("[smoke] rendering ddim @ {steps} steps (contrast) ...");
        let ddim = render(&*generator, &prompt, w, h, steps, 7.5, Some("ddim"));
        save_png(&ddim, &out_dir.join(format!("ddim_{steps}step.png")));
        println!(
            "[smoke] ddim contrast std {:.2} -> {}",
            image_std(&ddim),
            out_dir.join(format!("ddim_{steps}step.png")).display()
        );
    }

    assert!(
        lightning_std > DEGENERATE_STD_FLOOR_DEFAULT,
        "lightning render looks degenerate (std {lightning_std:.2}) — possible NaN / all-black decode"
    );
    println!("[smoke] DONE: lightning render coherent at {steps} steps");
}
