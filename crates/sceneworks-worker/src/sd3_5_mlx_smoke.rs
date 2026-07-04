//! Local real-weight MLX smokes for the SD3.5 worker lane (epic 7841 S6 sc-7875 — the validation
//! boundary for the native MLX path). `#[ignore]`d — run by hand on an Apple-Silicon Mac with the
//! gated `stabilityai/stable-diffusion-3.5-{large,large-turbo,medium}` snapshots cached.
//!
//! These are the **worker-layer** counterpart to the engine-layer E5/E6/M3 smokes (which live in
//! `mlx-gen-sd3`'s own `tests/`). The point here is NOT to re-prove the engine in isolation — it is to
//! prove the **`sceneworks-worker` crate links + drives the registered SD3.5 generators end-to-end**:
//! the `use mlx_gen_sd3 as _;` force-link anchor in `image_jobs.rs` keeps the three `inventory::submit!`
//! `ModelRegistration`s from being GC'd, so `gen_core::load("sd3_5_large" | "sd3_5_large_turbo" |
//! "sd3_5_medium")` resolves the generator from inside this crate exactly as `generate_stream` does at
//! runtime (minus the API/job plumbing). A coherent 1024² render through this seam is the worker-path
//! proof.
//!
//! SD3.5 loads directly from the raw `stabilityai/*` diffusers snapshot (the triple TE + MMDiT + 16-ch
//! VAE), quantizing on load — the same `LoadSpec { WeightsSource::Dir(snapshot), quant }` the worker's
//! image path builds (`image_jobs::base::load_spec`). No pre-quantized turnkey dir (unlike Krea), so the
//! smoke points straight at the cached gated snapshot.
//!
//! ```text
//! # all three t2i round-trips (Q8 by default):
//! cargo test -p sceneworks-worker --release sd3_5 -- --ignored --nocapture
//! # footprint profiling (peak OS memory footprint vs the manifest minMemoryGb):
//! /usr/bin/time -l cargo test -p sceneworks-worker --release sd3_5_large_mlx_gpu_smoke -- --ignored --nocapture
//! # LoRA-apply worker-seam smoke (set SD3_LORA to a real community sd3 LoRA, else it self-skips):
//! SD3_LORA=/path/to/sd3_style_lora.safetensors cargo test -p sceneworks-worker --release sd3_5_lora_apply_mlx_gpu_smoke -- --ignored --nocapture
//! ```
//!
//! Env overrides (per model): `SD3_LARGE_SNAPSHOT` / `SD3_TURBO_SNAPSHOT` / `SD3_MEDIUM_SNAPSHOT`
//! (snapshot root dir); `SD3_QUANT` (`q8`|`q4`, default q8); `SD3_W` / `SD3_H` (default 1024);
//! `SD3_STEPS` (default per model: Large 28, Turbo 4, Medium 40); `SD3_PROMPT`; `SD3_OUT_DIR`
//! (default `/tmp/sd3_5_smoke`).

use std::path::PathBuf;

use gen_core::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, Image, LoadSpec, Quant,
    WeightsSource,
};

use super::smoke_support::{env_or, save_png};

/// `q4` -> Q4, `q8` -> Q8 (the manifest default for all three SD3.5 tiers). An unrecognized value
/// panics with a clear hint rather than silently defaulting to Q8 (sc-8924): a hand-run tier
/// validation with a typo'd `SD3_QUANT` would otherwise PASS the wrong tier and record a bogus result.
/// Pure over the raw string so the reject behavior is unit-testable without touching process env.
fn parse_quant(raw: &str) -> Quant {
    match raw.trim().to_ascii_lowercase().as_str() {
        "q4" => Quant::Q4,
        "q8" => Quant::Q8,
        other => panic!("SD3_QUANT must be q4 or q8, got {other:?}"),
    }
}

fn quant_from_env() -> Quant {
    parse_quant(&env_or("SD3_QUANT", "q8"))
}

#[cfg(test)]
mod quant_tests {
    use super::{parse_quant, Quant};

    #[test]
    fn parse_quant_accepts_q4_q8_case_insensitively() {
        assert!(matches!(parse_quant("q4"), Quant::Q4));
        assert!(matches!(parse_quant("Q4"), Quant::Q4));
        assert!(matches!(parse_quant(" q8 "), Quant::Q8));
    }

    #[test]
    #[should_panic(expected = "SD3_QUANT must be q4 or q8")]
    fn parse_quant_rejects_unknown_value() {
        // A typo'd tier must NOT silently default (sc-8924) — it would record a bogus tier validation.
        let _ = parse_quant("q5");
    }
}

/// Resolve a gated `stabilityai/<repo>` snapshot dir: the `<env_override>` if set, else the first
/// snapshot subdir in the HF hub cache (mirrors the engine's own e2e smoke + the worker's
/// `${HF_CACHE}/...` path resolution). Panics loud if neither resolves so a missing/half-downloaded
/// gated pull surfaces clearly rather than as a confusing engine load error.
fn snapshot(env_override: &str, hub_dir: &str) -> PathBuf {
    if let Ok(p) = std::env::var(env_override) {
        if !p.trim().is_empty() {
            return PathBuf::from(p.trim());
        }
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(hub_dir)
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| {
            panic!(
                "no {hub_dir} snapshots under {snaps:?}; accept the Stability AI Community License + \
                 download the gated repo, or set {env_override}"
            )
        })
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir() && p.join("model_index.json").is_file())
        .unwrap_or_else(|| panic!("no diffusers snapshot (with model_index.json) under {snaps:?}"))
}

/// Per-channel mean + std over an RGB8 image — flat/constant output collapses std≈0.
fn channel_stats(img: &Image) -> [(f64, f64); 3] {
    let n = (img.width * img.height) as usize;
    let mut out = [(0.0f64, 0.0f64); 3];
    if n == 0 {
        return out;
    }
    for (c, stat) in out.iter_mut().enumerate() {
        let mut sum = 0.0f64;
        for i in 0..n {
            sum += img.pixels[i * 3 + c] as f64;
        }
        let mean = sum / n as f64;
        let mut var = 0.0f64;
        for i in 0..n {
            let d = img.pixels[i * 3 + c] as f64 - mean;
            var += d * d;
        }
        *stat = (mean, (var / n as f64).sqrt());
    }
    out
}

/// Mean absolute horizontal+vertical neighbor gradient on the luma plane — a crude spatial-coherence
/// signal. White noise ≈85/255, a flat image ≈0, a coherent photo is LOW-but-nonzero (~5–25). A
/// coherent render lands well below the noise floor and well above flat. (Same gate the engine e2e
/// smoke uses — the coordinator can't view the saved PNG, so this is the machine-readable proof.)
fn mean_neighbor_gradient(img: &Image) -> f64 {
    let (wi, hi) = (img.width as usize, img.height as usize);
    if wi < 2 || hi < 2 {
        return 0.0;
    }
    let luma = |x: usize, y: usize| -> f64 {
        let i = (y * wi + x) * 3;
        0.299 * img.pixels[i] as f64
            + 0.587 * img.pixels[i + 1] as f64
            + 0.114 * img.pixels[i + 2] as f64
    };
    let mut sum = 0.0f64;
    let mut cnt = 0u64;
    for y in 0..hi {
        for x in 0..wi {
            if x + 1 < wi {
                sum += (luma(x, y) - luma(x + 1, y)).abs();
                cnt += 1;
            }
            if y + 1 < hi {
                sum += (luma(x, y) - luma(x, y + 1)).abs();
                cnt += 1;
            }
        }
    }
    sum / cnt as f64
}

/// Drive the worker-lane load→generate seam for one SD3.5 engine id and assert a coherent image.
/// `engine_id` is the `gen_core::load` id (`sd3_5_large` | `sd3_5_large_turbo` | `sd3_5_medium`),
/// resolved through the worker's force-link anchor. `guidance` mirrors the worker's `resolve_guidance`:
/// `None` for the CFG-free Turbo (the descriptor advertises supports_guidance=false), `Some(scale)` for
/// the true-CFG Large/Medium. `adapters` exercises the `LoadSpec::with_adapters` apply seam.
#[allow(clippy::too_many_arguments)]
fn run_worker_smoke(
    engine_id: &str,
    snapshot: PathBuf,
    quant: Quant,
    w: u32,
    h: u32,
    steps: u32,
    guidance: Option<f32>,
    adapters: Vec<AdapterSpec>,
    out_name: &str,
) -> Image {
    let out_dir = PathBuf::from(env_or("SD3_OUT_DIR", "/tmp/sd3_5_smoke"));
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let prompt = env_or(
        "SD3_PROMPT",
        "a photograph of a red fox sitting in a green meadow, sharp focus, daylight",
    );

    println!(
        "\n=== [{engine_id}] worker-lane smoke {w}x{h} steps={steps} quant={quant:?} \
         guidance={guidance:?} adapters={} ===\nsnapshot: {}",
        adapters.len(),
        snapshot.display()
    );

    // The exact runtime seam `image_jobs::base::load_spec` builds: a Dir LoadSpec (+ quant, + any
    // adapters via `with_adapters`). `gen_core::load` resolves the generator the worker force-link
    // keeps registered — if the `use mlx_gen_sd3 as _;` anchor were dropped this would fail with
    // "no generator registered for <id>", which is precisely the worker-wiring failure this guards.
    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot)).with_quant(quant);
    if !adapters.is_empty() {
        spec = spec.with_adapters(adapters);
    }
    let t_load = std::time::Instant::now();
    let generator =
        gen_core::load(engine_id, &spec).unwrap_or_else(|e| panic!("load {engine_id}: {e:?}"));
    println!("loaded in {:.1}s", t_load.elapsed().as_secs_f32());

    // True-CFG Large/Medium take a negative prompt; CFG-free Turbo's is inert (guidance=None).
    let negative_prompt = guidance.map(|_| "blurry, low quality, distorted".to_string());
    let req = GenerationRequest {
        prompt,
        negative_prompt,
        width: w,
        height: h,
        count: 1,
        seed: Some(7),
        steps: Some(steps),
        guidance,
        ..Default::default()
    };

    let t_gen = std::time::Instant::now();
    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |p| {
            let s = format!("{p:?}");
            if s != last {
                println!("[progress] {s}");
                last = s;
            }
        })
        .unwrap_or_else(|e| panic!("{engine_id} generate: {e:?}"));
    let gen_secs = t_gen.elapsed().as_secs_f32();

    let image = match output {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1, "count=1 -> one image");
            v.pop().expect("engine returned no image")
        }
        other => panic!("[{engine_id}] expected Images output, got {other:?}"),
    };
    assert_eq!(
        (image.width, image.height),
        (w, h),
        "[{engine_id}] engine returned the wrong dimensions"
    );

    let png = out_dir.join(out_name);
    save_png(&image, &png);

    let stats = channel_stats(&image);
    let grad = mean_neighbor_gradient(&image);
    let max_std = stats.iter().fold(0.0f64, |m, s| m.max(s.1));
    println!(
        "generated in {gen_secs:.1}s ({:.2}s/step) -> {}",
        gen_secs / steps.max(1) as f32,
        png.display()
    );
    println!(
        "channel mean/std: R {:.1}/{:.1}  G {:.1}/{:.1}  B {:.1}/{:.1}",
        stats[0].0, stats[0].1, stats[1].0, stats[1].1, stats[2].0, stats[2].1
    );
    println!("mean neighbor gradient (luma): {grad:.2} (white-noise≈85, flat≈0, coherent≈5–25)");

    // --- coherence gates (same honest signals as the engine e2e smoke) -----------------------------
    assert!(
        max_std > 8.0,
        "[{engine_id}] output looks flat/constant (max channel std {max_std:.2}); a real image has \
         contrast — likely a worker-lane load/dispatch wiring break"
    );
    assert!(
        grad < 60.0,
        "[{engine_id}] neighbor gradient {grad:.2} near the white-noise floor (~85) — the render is \
         noise, NOT a coherent image (forward/sampler/CFG/VAE break reaching through the worker)"
    );
    assert!(
        grad > 1.0,
        "[{engine_id}] neighbor gradient {grad:.2} ≈ 0 — the render is essentially flat, not an image"
    );
    println!("[{engine_id}] PASS: coherent worker-lane render (contrast {max_std:.1}, gradient {grad:.2})");
    image
}

/// SD3.5 Large (true-CFG flagship) through the worker lane. Q8 default, 1024²/28 steps / guidance 3.5
/// (the MODEL_TABLE recipe). Also the footprint-profiling target: run under `/usr/bin/time -l` and
/// compare "peak memory footprint" to the manifest `minMemoryGb: 64`.
#[test]
#[ignore = "real-weight MLX smoke; needs the gated stabilityai/stable-diffusion-3.5-large snapshot + an Apple-Silicon Mac"]
fn sd3_5_large_mlx_gpu_smoke() {
    let quant = quant_from_env();
    let snap = snapshot(
        "SD3_LARGE_SNAPSHOT",
        "models--stabilityai--stable-diffusion-3.5-large",
    );
    let w: u32 = env_or("SD3_W", "1024").parse().expect("SD3_W");
    let h: u32 = env_or("SD3_H", "1024").parse().expect("SD3_H");
    let steps: u32 = env_or("SD3_STEPS", "28").parse().expect("SD3_STEPS");
    let q = if matches!(quant, Quant::Q4) {
        "q4"
    } else {
        "q8"
    };
    run_worker_smoke(
        "sd3_5_large",
        snap,
        quant,
        w,
        h,
        steps,
        Some(3.5),
        Vec::new(),
        &format!("sd3_5_large_{q}_{steps}step.png"),
    );
}

/// SD3.5 Large Turbo (ADD-distilled, CFG-free) through the worker lane. 4 steps, `guidance: None`
/// (mirrors `resolve_guidance` — the descriptor advertises supports_guidance=false, no negative pass).
#[test]
#[ignore = "real-weight MLX smoke; needs the gated stabilityai/stable-diffusion-3.5-large-turbo snapshot + an Apple-Silicon Mac"]
fn sd3_5_large_turbo_mlx_gpu_smoke() {
    let quant = quant_from_env();
    let snap = snapshot(
        "SD3_TURBO_SNAPSHOT",
        "models--stabilityai--stable-diffusion-3.5-large-turbo",
    );
    let w: u32 = env_or("SD3_W", "1024").parse().expect("SD3_W");
    let h: u32 = env_or("SD3_H", "1024").parse().expect("SD3_H");
    let steps: u32 = env_or("SD3_STEPS", "4").parse().expect("SD3_STEPS");
    let q = if matches!(quant, Quant::Q4) {
        "q4"
    } else {
        "q8"
    };
    run_worker_smoke(
        "sd3_5_large_turbo",
        snap,
        quant,
        w,
        h,
        steps,
        None,
        Vec::new(),
        &format!("sd3_5_large_turbo_{q}_{steps}step.png"),
    );
}

/// SD3.5 Medium (MMDiT-X, true-CFG) through the worker lane. 40 steps / guidance 5.0 (the MODEL_TABLE
/// recipe). The S6 worker-lane validation measured the OS peak at ~52 GB Q8 / ~48.6 GB Q4 (the dense
/// bf16 T5-XXL TE + VAE + load-time quant dominate the 2.5B backbone, disproving the earlier ~16 GB
/// estimate), so the manifest minMemoryGb is 56 — the lightest SD3.5 tier; verify the footprint stays
/// consistent with that.
#[test]
#[ignore = "real-weight MLX smoke; needs the gated stabilityai/stable-diffusion-3.5-medium snapshot + an Apple-Silicon Mac"]
fn sd3_5_medium_mlx_gpu_smoke() {
    let quant = quant_from_env();
    let snap = snapshot(
        "SD3_MEDIUM_SNAPSHOT",
        "models--stabilityai--stable-diffusion-3.5-medium",
    );
    let w: u32 = env_or("SD3_W", "1024").parse().expect("SD3_W");
    let h: u32 = env_or("SD3_H", "1024").parse().expect("SD3_H");
    let steps: u32 = env_or("SD3_STEPS", "40").parse().expect("SD3_STEPS");
    let q = if matches!(quant, Quant::Q4) {
        "q4"
    } else {
        "q8"
    };
    run_worker_smoke(
        "sd3_5_medium",
        snap,
        quant,
        w,
        h,
        steps,
        Some(5.0),
        Vec::new(),
        &format!("sd3_5_medium_{q}_{steps}step.png"),
    );
}

/// Community-LoRA application smoke (AC #5) — exercises the worker apply seam (`LoadSpec::with_adapters`
/// -> the engine's `apply_sd3_adapters`) at SD3.5 generation. Set `SD3_LORA` to an sd3 LoRA
/// `.safetensors` (a real community adapter, or — when none is cached — the adapter the T3 trainer smoke
/// `train_sd3_lora` produces; the S5 `sd3` family accepts both, see `lora_family.rs`). `SD3_LORA_ENGINE`
/// picks the base (`sd3_5_large` default, or `sd3_5_medium` to match a Medium-trained adapter's block
/// layout). If `SD3_LORA` is unset the smoke self-skips with a hint (the S5 family-acceptance + worker
/// apply-path wiring already have unit coverage). A coherent render with the adapter applied proves the
/// family-gated LoRA flows through the worker `with_adapters` seam into the MMDiT's `apply_sd3_adapters`.
#[test]
#[ignore = "real-weight MLX smoke; needs the SD3.5 snapshot + an sd3 LoRA (SD3_LORA, e.g. the train_sd3_lora output) + an Apple-Silicon Mac"]
fn sd3_5_lora_apply_mlx_gpu_smoke() {
    let lora = match std::env::var("SD3_LORA") {
        Ok(p) if !p.trim().is_empty() => PathBuf::from(p.trim()),
        _ => {
            // Make the self-skip UNMISTAKABLE in a recorded validation run (sc-8926): libtest reports a
            // bare `return` as `ok`, so a green line here would falsely read as "the LoRA apply path was
            // exercised". Emit a loud, greppable `[SKIPPED]` marker naming the test + the missing lever on
            // BOTH streams (stdout is captured and printed on the hand-run; stderr shows even without
            // --nocapture) so a story-recorded run cannot mistake this skip for a real pass.
            let msg = "[SKIPPED] sd3_5_lora_apply_mlx_gpu_smoke: SD3_LORA not set — the worker LoRA \
                       apply seam (with_adapters -> apply_sd3_adapters) was NOT exercised. Point SD3_LORA \
                       at an sd3 LoRA .safetensors (a community adapter, or the `train_sd3_lora` smoke \
                       output) to run it.";
            println!("{msg}");
            eprintln!("{msg}");
            return;
        }
    };
    assert!(
        lora.is_file(),
        "SD3_LORA does not point at a file: {}",
        lora.display()
    );

    let engine_id = env_or("SD3_LORA_ENGINE", "sd3_5_large");
    // `default_guidance` mirrors `resolve_guidance`: `Some(scale)` for the true-CFG Large/Medium
    // (supports_guidance=true), `None` for the CFG-free Turbo (supports_guidance=false) — so the smoke
    // sends the SAME request shape the shipped worker sends (and the dedicated turbo smoke passes None).
    let (hub_dir, env_snap, default_steps, default_guidance): (&str, &str, &str, Option<f32>) =
        match engine_id.as_str() {
            "sd3_5_medium" => (
                "models--stabilityai--stable-diffusion-3.5-medium",
                "SD3_MEDIUM_SNAPSHOT",
                "40",
                Some(5.0),
            ),
            "sd3_5_large_turbo" => (
                "models--stabilityai--stable-diffusion-3.5-large-turbo",
                "SD3_TURBO_SNAPSHOT",
                "4",
                None,
            ),
            _ => (
                "models--stabilityai--stable-diffusion-3.5-large",
                "SD3_LARGE_SNAPSHOT",
                "28",
                Some(3.5),
            ),
        };

    let scale: f32 = env_or("SD3_LORA_SCALE", "1.0")
        .parse()
        .expect("SD3_LORA_SCALE");
    let snap = snapshot(env_snap, hub_dir);
    let quant = quant_from_env();
    let w: u32 = env_or("SD3_W", "1024").parse().expect("SD3_W");
    let h: u32 = env_or("SD3_H", "1024").parse().expect("SD3_H");
    let steps: u32 = env_or("SD3_STEPS", default_steps)
        .parse()
        .expect("SD3_STEPS");

    // Mirror `image_jobs::base::resolve_adapters`: AdapterSpec::new(file, scale, kind=Lora).
    let adapters = vec![AdapterSpec::new(lora.clone(), scale, AdapterKind::Lora)];
    println!(
        "[smoke] applying sd3 LoRA {} @ scale {scale} at {engine_id} via the worker with_adapters seam",
        lora.display()
    );
    run_worker_smoke(
        &engine_id,
        snap,
        quant,
        w,
        h,
        steps,
        default_guidance,
        adapters,
        "sd3_5_lora.png",
    );
}
