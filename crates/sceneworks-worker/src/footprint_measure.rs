//! On-device per-tier memory-footprint measurement harness (sc-8516, epic 8506).
//!
//! Purpose: produce REAL steady-state resident + peak GPU-memory numbers per (model × quant tier)
//! so the sc-8509 RAM→tier suggestion (apps/web/src/tierSuggestion.js) can be calibrated from
//! measured data instead of the disk×multiplier estimate, and so the sc-8508 manifest footprint
//! fields (`footprint.residentMemoryBytes` / `peakMemoryBytes`) can be populated.
//!
//! Each test drives the EXACT worker runtime seam a real job uses — a registry `gen_core::load(id)`
//! of a packed tier subdir (the same one `image_jobs::base::standard_tier_subdir` resolves) + ONE
//! generation — while sampling the MLX process-global memory counters that generator_cache.rs already
//! publishes to telemetry:
//!   * `mlx_rs::memory::get_active_memory()` — bytes currently live on the GPU allocator. RESIDENT is
//!     sampled from this AFTER the generation AND AFTER a `clear_cache()` that releases the gen's
//!     transient working buffers, so it is the steady-state weight footprint ONLY, NOT the transient.
//!     (MLX is lazy: directly after `load` NOTHING is materialized — `get_active_memory()` reads ~0 —
//!     so resident is only observable once a gen has forced the weights resident. The credibility fix
//!     is the `clear_cache()` before the sample: the old harness read active post-gen WITHOUT it, which
//!     folded the freeable transient INTO resident — inflating resident, understating peak−resident.)
//!   * `mlx_rs::memory::get_peak_memory()` — high-water mark since process start / last reset. Reset
//!     BEFORE load and read AFTER the generation, so the reported peak covers load + the single
//!     generation (the true install-time ceiling the RAM suggestion must budget for).
//!
//! RUN ONE TIER PER PROCESS. The MLX counters + allocator peak high-water mark are process-global and
//! persist across tests in the same binary invocation, so each tier MUST be measured in its OWN
//! `cargo test … footprint_<x>` invocation for a clean allocator + peak counter — otherwise a heavier
//! earlier tier's peak leaks into a lighter later one. The manifest numbers were each captured
//! fresh-process this way.
//!
//! These are `#[ignore]`d real-weight smokes — run by hand on an Apple-Silicon Mac with the tier's
//! turnkey cached (same convention as the *_mlx_smoke tests). Each prints a single machine-parseable
//! `[[FOOTPRINT]] {json}` line (model, tier, diskSizeBytes, residentMemoryBytes, peakMemoryBytes) so
//! a run over the measurable set can be scraped straight into builtin.models.jsonc.
//!
//! ```text
//! # One tier per invocation (fresh MLX allocator + peak counter each time):
//! cargo test -p sceneworks-worker --release footprint_sdxl_q8         -- --ignored --nocapture
//! cargo test -p sceneworks-worker --release footprint_z_image_turbo_q4 -- --ignored --nocapture
//! cargo test -p sceneworks-worker --release footprint_z_image_q4       -- --ignored --nocapture
//! cargo test -p sceneworks-worker --release footprint_lens_turbo_q4    -- --ignored --nocapture
//! ```
//!
//! CALIBRATION SET (prefer already-on-disk tiers; do NOT trigger a 30GB+ download sweep — sc-8516
//! records un-measured tiers for a backfill story and the estimate keeps their suggestion accurate):
//!   * sdxl            q8  (SceneWorks/sdxl-base-mlx      q8/)
//!   * z_image_turbo   q4  (SceneWorks/z-image-turbo-mlx  q4/)
//!   * z_image         q4  (SceneWorks/z-image-mlx        q4/)
//!   * lens_turbo      q4  (SceneWorks/lens-turbo-mlx     q4/)
//!
//! Extra tiers (other quants / models) auto-measure if their subdir is cached; otherwise the test
//! panics with a download hint and is simply not part of the run.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use gen_core::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};

use super::smoke_support::{env_or, image_std, DEGENERATE_STD_FLOOR_DEFAULT};

/// One-tier-per-process guard (sc-8925). The MLX memory counters + allocator peak high-water mark are
/// PROCESS-GLOBAL and persist across tests in the same binary, so measuring a second tier in the same
/// process folds the first tier's residue into its numbers. `measure_footprint` flips this once and
/// asserts it was previously false — an unfiltered `cargo test … footprint -- --ignored` run (more than
/// one tier in one process) now fails loudly instead of silently emitting corrupt `[[FOOTPRINT]]` lines.
static FOOTPRINT_RAN: AtomicBool = AtomicBool::new(false);

/// Locate a cached `SceneWorks/<repo>` turnkey snapshot whose `<tier>/` subdir carries the packed
/// backbone file `sentinel` (e.g. `unet/diffusion_pytorch_model.safetensors`). Returns the `<tier>/`
/// dir itself, ready to hand to `WeightsSource::Dir`. `None` if the tier hasn't been pulled.
fn cached_tier_dir(repo: &str, tier: &str, sentinel: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let repo_cache = repo.replace('/', "--");
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(format!("models--{repo_cache}"))
        .join("snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let tier_dir = e.path().join(tier);
            tier_dir.join(sentinel).is_file().then_some(tier_dir)
        })
}

/// Resolve the tier dir from an explicit override env (a turnkey root OR the tier dir itself) or the
/// HF cache, panicking with a download hint if neither resolves so a half/no download surfaces clearly.
fn resolve_tier_dir(env_key: &str, repo: &str, tier: &str, sentinel: &str) -> PathBuf {
    if let Ok(p) = std::env::var(env_key) {
        let p = p.trim();
        if !p.is_empty() {
            let root = PathBuf::from(p);
            let sub = root.join(tier);
            if sub.join(sentinel).is_file() {
                return sub;
            }
            if root.join(sentinel).is_file() {
                return root;
            }
            panic!(
                "{env_key}={} has no packed {tier}/{sentinel} (nor is it itself a {tier} root)",
                root.display()
            );
        }
    }
    cached_tier_dir(repo, tier, sentinel).unwrap_or_else(|| {
        panic!(
            "no cached {repo} {tier} turnkey found; download it \
             (`hf download {repo} --include '{tier}/*'`) or set {env_key} to the turnkey root"
        )
    })
}

/// The load-quant for a tier, or `None` for the dense `bf16` tier (loaded without `.with_quant`, the
/// same as the worker's dense path — the packed q4/q8 subdirs auto-detect their quant, so `with_quant`
/// just names the tier).
fn quant_for(tier: &str) -> Option<Quant> {
    match tier {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        _ => None,
    }
}

/// Run one real load + generation for `(model, tier)` at `dir`, sampling the MLX memory counters
/// around it, and emit the machine-parseable `[[FOOTPRINT]]` line. Returns (residentBytes, peakBytes).
///
/// Sequence (mirrors the worker's per-job memory lifecycle in generator_cache.rs, and separates the
/// steady-state weight footprint from the generation transient — the sc-8516 credibility fix):
///   1. `clear_cache()` then `reset_peak_memory()` — start the peak high-water mark from a clean base
///      so it captures load + this one generation, not leftovers from a prior test in the same process.
///   2. record `baseline = get_active_memory()` — anything already live (should be ~0 in a
///      one-tier-per-process run; subtracted so each line is the marginal footprint of THIS tier).
///   3. load the packed tier + generate once. PEAK = `get_peak_memory()` read right after gen — the
///      load+gen high-water ceiling (peak was reset in step 1 and never since, so it spans the window).
///   4. `clear_cache()` to RELEASE the generation's transient working buffers (VAE-decode scratch,
///      attention activations, latents) back to the OS, THEN sample
///      `residentBytes = get_active_memory() - baseline` — the STEADY-STATE resident WEIGHTS only.
///
/// Why resident is sampled POST-gen-and-clear rather than pre-gen: MLX evaluates lazily, so directly
/// after `gen_core::load` NOTHING is materialized on the GPU allocator (`get_active_memory()` reads
/// ~0 — verified) — the weights are only realized when the first forward pass touches them. So a true
/// steady-state resident is only observable AFTER a generation has forced materialization. The fix
/// versus the original harness is the `clear_cache()` on line below: the old code sampled
/// `get_active_memory()` post-gen WITHOUT releasing the cache, folding the gen's freeable transient
/// INTO resident (inflating resident, understating peak−resident). Dropping the cache first leaves
/// only the live weight arrays the generator still holds.
// Every caller discards the result — the resident/peak numbers are consumed off the printed
// `[[FOOTPRINT]]` scrape line, not the return value — so this reports through stdout and returns
// nothing (sc-8952).
fn measure_footprint(model: &str, tier: &str, engine_id: &str, dir: &Path, req: GenerationRequest) {
    // Enforce the "RUN ONE TIER PER PROCESS" invariant (sc-8925): the MLX counters are process-global,
    // so a second measurement in the same binary invocation would report numbers contaminated by the
    // first tier's allocator residue + peak high-water mark. Fail loudly instead of scraping garbage.
    assert!(
        !FOOTPRINT_RAN.swap(true, Ordering::SeqCst),
        "footprint harness measured a second tier in one process — the MLX peak/active counters are \
         process-global, so run exactly one `footprint_<x>` per `cargo test … -- --ignored` invocation \
         (see the module doc). The just-attempted {model} {tier} measurement would be corrupt."
    );

    println!(
        "[footprint] loading {model} ({tier}) from {} ...",
        dir.display()
    );

    // 1. Clean baseline: drop the allocator's free cache and zero the peak high-water mark so the
    //    numbers below are this tier's own load+gen, not a prior test's residue.
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let baseline = mlx_rs::memory::get_active_memory() as u64;

    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()));
    if let Some(q) = quant_for(tier) {
        spec = spec.with_quant(q);
    }
    let generator = gen_core::load(engine_id, &spec)
        .unwrap_or_else(|e| panic!("load {engine_id} ({tier}): {e:?}"));

    // 3. One generation, then the load+gen peak high-water mark (reset in step 1, never since).
    let mut last = String::new();
    let output = generator
        .generate(&req, &mut |p| {
            let s = format!("{p:?}");
            if s != last {
                println!("[progress] {s}");
                last = s;
            }
        })
        .unwrap_or_else(|e| panic!("{model} {tier} generate: {e:?}"));
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };
    let std = image_std(&image);
    assert!(
        std > DEGENERATE_STD_FLOOR_DEFAULT,
        "{model} {tier} render looks degenerate (std {std:.2}) — measured footprint would be bogus"
    );
    let peak = mlx_rs::memory::get_peak_memory() as u64;

    // 4. STEADY-STATE RESIDENT: release the gen's transient working buffers, THEN sample. This is the
    //    credibility fix — the old harness sampled active WITHOUT this clear_cache, so resident carried
    //    the freeable transient and peak−resident was an artifact, not the real transient.
    mlx_rs::memory::clear_cache();
    let active_resident = mlx_rs::memory::get_active_memory() as u64;
    let resident = active_resident.saturating_sub(baseline);
    let transient = peak.saturating_sub(active_resident);

    // Machine-parseable line — scrape `[[FOOTPRINT]]` to backfill builtin.models.jsonc.
    println!(
        "[[FOOTPRINT]] {{\"model\":\"{model}\",\"tier\":\"{tier}\",\"residentMemoryBytes\":{resident},\"peakMemoryBytes\":{peak}}}"
    );
    println!(
        "[footprint] {model} {tier}: resident {:.2} GiB (active {:.2}, baseline {:.2}) | peak {:.2} GiB | transient (peak−resident) {:.2} GiB | render std {:.1}",
        resident as f64 / 1024.0 / 1024.0 / 1024.0,
        active_resident as f64 / 1024.0 / 1024.0 / 1024.0,
        baseline as f64 / 1024.0 / 1024.0 / 1024.0,
        peak as f64 / 1024.0 / 1024.0 / 1024.0,
        transient as f64 / 1024.0 / 1024.0 / 1024.0,
        std,
    );
}

/// SDXL-family default request: 1024², real CFG. Kept modest on steps to keep the harness quick; the
/// footprint is dominated by resident weights + a single denoise's activations, not step count.
fn sdxl_request() -> GenerationRequest {
    GenerationRequest {
        prompt: env_or(
            "FP_PROMPT",
            "a photorealistic portrait of a red fox in a sunlit autumn forest, sharp focus",
        ),
        width: env_or("FP_W", "1024").parse().expect("FP_W"),
        height: env_or("FP_H", "1024").parse().expect("FP_H"),
        count: 1,
        seed: Some(42),
        steps: Some(env_or("FP_STEPS", "12").parse().expect("FP_STEPS")),
        guidance: Some(7.0),
        ..Default::default()
    }
}

/// Distilled/turbo request: few steps. `guidance`: `Some(1.0)` for lens (CFG-scale 1 accepted), or
/// `None` for guidance-DISTILLED engines like z_image_turbo that reject any `guidance` value.
fn turbo_request(steps: u32, guidance: Option<f32>) -> GenerationRequest {
    GenerationRequest {
        prompt: env_or(
            "FP_PROMPT",
            "a photorealistic portrait of a red fox in a sunlit autumn forest, sharp focus",
        ),
        width: env_or("FP_W", "1024").parse().expect("FP_W"),
        height: env_or("FP_H", "1024").parse().expect("FP_H"),
        count: 1,
        seed: Some(42),
        steps: Some(
            env_or("FP_STEPS", &steps.to_string())
                .parse()
                .expect("FP_STEPS"),
        ),
        guidance,
        ..Default::default()
    }
}

const SDXL_SENTINEL: &str = "unet/diffusion_pytorch_model.safetensors";
// The dense bf16 tier packs the UNet under the `.fp16.safetensors` variant filename (the turnkey
// recipe: quantized tiers use the plain name, the dense tier uses `.fp16`). `resolve_weight_file`
// tries both at load, but the tier-presence sentinel must name the file that actually exists.
const SDXL_BF16_SENTINEL: &str = "unet/diffusion_pytorch_model.fp16.safetensors";
// Lens turnkeys pack the DiT under transformer/diffusion_pytorch_model.safetensors …
const LENS_SENTINEL: &str = "transformer/diffusion_pytorch_model.safetensors";
// … but the DENSE bf16 lens-turbo DiT is SHARDED (diffusion_pytorch_model-0000N-of-0000M.safetensors),
// so its tier-presence sentinel is the shard index file that the plain name would otherwise miss.
const LENS_SHARDED_SENTINEL: &str = "transformer/diffusion_pytorch_model.safetensors.index.json";
// … while Z-Image AND Qwen-Image turnkeys pack the DiT under transformer/model.safetensors.
const ZIMAGE_SENTINEL: &str = "transformer/model.safetensors";

#[test]
#[ignore = "footprint measurement; needs SceneWorks/sdxl-base-mlx q8 cached + an Apple-Silicon Mac"]
fn footprint_sdxl_q8() {
    let dir = resolve_tier_dir(
        "FP_SDXL_Q8_DIR",
        "SceneWorks/sdxl-base-mlx",
        "q8",
        SDXL_SENTINEL,
    );
    measure_footprint("sdxl", "q8", "sdxl", &dir, sdxl_request());
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/sdxl-base-mlx q4 cached + an Apple-Silicon Mac"]
fn footprint_sdxl_q4() {
    let dir = resolve_tier_dir(
        "FP_SDXL_Q4_DIR",
        "SceneWorks/sdxl-base-mlx",
        "q4",
        SDXL_SENTINEL,
    );
    measure_footprint("sdxl", "q4", "sdxl", &dir, sdxl_request());
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/sdxl-base-mlx bf16 cached + an Apple-Silicon Mac"]
fn footprint_sdxl_bf16() {
    let dir = resolve_tier_dir(
        "FP_SDXL_BF16_DIR",
        "SceneWorks/sdxl-base-mlx",
        "bf16",
        SDXL_SENTINEL,
    );
    measure_footprint("sdxl", "bf16", "sdxl", &dir, sdxl_request());
}

// Illustrious-XL v1.0 / v2.0 (epic 10609, sc-10619): SDXL-family turnkeys, so the same `sdxl` engine,
// the same `unet/` sentinel, and the same per-tier resolution as sdxl-base. `FP_ILL_V{1,2}_DIR` points
// at the turnkey ROOT (the dir holding q4/ q8/ bf16/); resolve_tier_dir joins the tier. Set an anime
// `FP_PROMPT` for a representative render — the footprint is prompt-independent, but the non-degenerate
// gate is more meaningful on an in-distribution image. One tier per process (the process-global MLX
// counter invariant), so run these one `-- --ignored footprint_illustrious_<x>` at a time.
#[test]
#[ignore = "footprint measurement; needs SceneWorks/illustrious-xl-v1-mlx q4 cached + an Apple-Silicon Mac"]
fn footprint_illustrious_v1_q4() {
    let dir = resolve_tier_dir(
        "FP_ILL_V1_DIR",
        "SceneWorks/illustrious-xl-v1-mlx",
        "q4",
        SDXL_SENTINEL,
    );
    measure_footprint("illustrious_xl_v1", "q4", "sdxl", &dir, sdxl_request());
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/illustrious-xl-v1-mlx q8 cached + an Apple-Silicon Mac"]
fn footprint_illustrious_v1_q8() {
    let dir = resolve_tier_dir(
        "FP_ILL_V1_DIR",
        "SceneWorks/illustrious-xl-v1-mlx",
        "q8",
        SDXL_SENTINEL,
    );
    measure_footprint("illustrious_xl_v1", "q8", "sdxl", &dir, sdxl_request());
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/illustrious-xl-v1-mlx bf16 cached + an Apple-Silicon Mac"]
fn footprint_illustrious_v1_bf16() {
    let dir = resolve_tier_dir(
        "FP_ILL_V1_DIR",
        "SceneWorks/illustrious-xl-v1-mlx",
        "bf16",
        SDXL_BF16_SENTINEL,
    );
    measure_footprint("illustrious_xl_v1", "bf16", "sdxl", &dir, sdxl_request());
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/illustrious-xl-v2-mlx q4 cached + an Apple-Silicon Mac"]
fn footprint_illustrious_v2_q4() {
    let dir = resolve_tier_dir(
        "FP_ILL_V2_DIR",
        "SceneWorks/illustrious-xl-v2-mlx",
        "q4",
        SDXL_SENTINEL,
    );
    measure_footprint("illustrious_xl_v2", "q4", "sdxl", &dir, sdxl_request());
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/illustrious-xl-v2-mlx q8 cached + an Apple-Silicon Mac"]
fn footprint_illustrious_v2_q8() {
    let dir = resolve_tier_dir(
        "FP_ILL_V2_DIR",
        "SceneWorks/illustrious-xl-v2-mlx",
        "q8",
        SDXL_SENTINEL,
    );
    measure_footprint("illustrious_xl_v2", "q8", "sdxl", &dir, sdxl_request());
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/illustrious-xl-v2-mlx bf16 cached + an Apple-Silicon Mac"]
fn footprint_illustrious_v2_bf16() {
    let dir = resolve_tier_dir(
        "FP_ILL_V2_DIR",
        "SceneWorks/illustrious-xl-v2-mlx",
        "bf16",
        SDXL_BF16_SENTINEL,
    );
    measure_footprint("illustrious_xl_v2", "bf16", "sdxl", &dir, sdxl_request());
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/z-image-turbo-mlx q4 cached + an Apple-Silicon Mac"]
fn footprint_z_image_turbo_q4() {
    let dir = resolve_tier_dir(
        "FP_ZIMAGE_TURBO_Q4_DIR",
        "SceneWorks/z-image-turbo-mlx",
        "q4",
        ZIMAGE_SENTINEL,
    );
    measure_footprint(
        "z_image_turbo",
        "q4",
        "z_image_turbo",
        &dir,
        turbo_request(8, None),
    );
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/z-image-mlx q4 cached + an Apple-Silicon Mac"]
fn footprint_z_image_q4() {
    let dir = resolve_tier_dir(
        "FP_ZIMAGE_Q4_DIR",
        "SceneWorks/z-image-mlx",
        "q4",
        ZIMAGE_SENTINEL,
    );
    // Non-turbo Z-Image runs real CFG at more steps; still a single denoise for the footprint.
    measure_footprint("z_image", "q4", "z_image", &dir, sdxl_request());
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/lens-turbo-mlx q4 cached + an Apple-Silicon Mac"]
fn footprint_lens_turbo_q4() {
    let dir = resolve_tier_dir(
        "FP_LENS_Q4_DIR",
        "SceneWorks/lens-turbo-mlx",
        "q4",
        LENS_SENTINEL,
    );
    measure_footprint(
        "lens_turbo",
        "q4",
        "lens_turbo",
        &dir,
        turbo_request(4, Some(1.0)),
    );
}

// ── sc-10863 HEADROOM calibration set ─────────────────────────────────────────────────────────────
// Four already-on-disk tiers spanning a small→large hold-all spread (illustrious q8 ~5 GiB → qwen-image
// q8 ~36 GiB) drive the fit-gate `HEADROOM_GB` calibration: the gate predicts peak = Σon-disk-weights +
// HEADROOM, so the measured `headroom = peakMemoryBytes − diskSizeBytes` (Σsafetensors) per tier fixes
// the constant. Each RESIDENT hold-all peak is measured through the SAME `measure_footprint` seam as the
// tiers above (default `LoadSpec` ⇒ `OffloadPolicy::Resident`, no fit-gate in the loop, no memory cap),
// so the peak is the true hold-all ceiling over summed weights — not a Sequential-staged number.
// illustrious q8 is covered by `footprint_illustrious_v{1,2}_q8` above; the other three are here.

#[test]
#[ignore = "footprint measurement; needs SceneWorks/lens-mlx q4 cached + an Apple-Silicon Mac"]
fn footprint_lens_q4() {
    // Base (non-turbo) lens: real CFG at 20 steps / guidance 5.0 (the `lens_base_q4_mlx_smoke` defaults).
    let dir = resolve_tier_dir(
        "FP_LENS_BASE_Q4_DIR",
        "SceneWorks/lens-mlx",
        "q4",
        LENS_SENTINEL,
    );
    measure_footprint("lens", "q4", "lens", &dir, turbo_request(20, Some(5.0)));
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/lens-turbo-mlx bf16 cached + an Apple-Silicon Mac"]
fn footprint_lens_turbo_bf16() {
    // Dense bf16 lens-turbo: SHARDED DiT ⇒ the shard-index sentinel. Turbo schedule (4 steps, CFG-scale 1).
    let dir = resolve_tier_dir(
        "FP_LENS_TURBO_BF16_DIR",
        "SceneWorks/lens-turbo-mlx",
        "bf16",
        LENS_SHARDED_SENTINEL,
    );
    measure_footprint(
        "lens_turbo",
        "bf16",
        "lens_turbo",
        &dir,
        turbo_request(4, Some(1.0)),
    );
}

#[test]
#[ignore = "footprint measurement; needs SceneWorks/qwen-image-mlx q8 cached + an Apple-Silicon Mac"]
fn footprint_qwen_image_q8() {
    // Qwen-Image base is a non-distilled true-CFG model (packs the DiT under transformer/model.safetensors
    // like Z-Image). 1024², guidance 4.0; few steps (the peak is set by load + one denoise, not step count).
    let dir = resolve_tier_dir(
        "FP_QWEN_IMAGE_Q8_DIR",
        "SceneWorks/qwen-image-mlx",
        "q8",
        ZIMAGE_SENTINEL,
    );
    measure_footprint(
        "qwen_image",
        "q8",
        "qwen_image",
        &dir,
        turbo_request(8, Some(4.0)),
    );
}

// ── sc-10863 emulation A/B: the calibrated HEADROOM_GB through the REAL cold-load gate ─────────────
/// End-to-end check that the sc-10863-calibrated `HEADROOM_GB` behaves correctly through the exact gate
/// the worker's cold-load path runs — `mlx_fit_gate::apply_residency_policy`, which `generator_cache.rs`
/// invokes right before `gen_core::load` (the piece the pure `mlx_fit_gate` unit tests can't cover: the
/// real on-disk byte sums + the live `SCENEWORKS_MLX_MEMORY_CAP_GB` budget resolution). Emulates three
/// Mac sizes via the cap and asserts the decision at each, then runs a REAL generation on the Resident
/// path so the calibrated value is confirmed against an actual load+gen, not just arithmetic:
///   * cap BELOW even the staged peak ⇒ Reject with the actionable over-budget message (the reject event);
///   * cap BETWEEN the staged and resident peaks ⇒ Sequential selected (emits
///     `mlx_sequential_residency_selected`, surfaced on stdout by the dev-only tracing subscriber);
///   * cap ABOVE the resident peak ⇒ Resident (spec unchanged) AND a real render succeeds.
///
/// Caps are derived from the SAME predictor the gate uses (`predicted_peak_gb` /
/// `predicted_sequential_peak_gb`), so the A/B stays valid if `HEADROOM_GB` is retuned. Mutates the
/// process-global cap env ⇒ `#[ignore]`d and run one-per-process like the footprint smokes.
#[test]
#[ignore = "sc-10863 fit-gate A/B; needs SceneWorks/illustrious-xl-v1-mlx q8 cached + an Apple-Silicon Mac"]
fn fit_gate_ab_illustrious_q8() {
    use crate::mlx_fit_gate::{
        apply_residency_policy, predicted_peak_gb, predicted_sequential_peak_gb,
        resolve_text_encoder_bytes, sum_safetensors_bytes, MLX_MEMORY_CAP_ENV,
    };
    use gen_core::OffloadPolicy;

    // Dev-only stdout subscriber so the real `mlx_sequential_residency_selected` INFO event is visible
    // under --nocapture (the gate emits it via `tracing::info!` — silent without a subscriber).
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .without_time()
        .try_init();

    let dir = resolve_tier_dir(
        "FP_ILL_V1_DIR",
        "SceneWorks/illustrious-xl-v1-mlx",
        "q8",
        SDXL_SENTINEL,
    );
    // Illustrious loads on the sequential-capable `sdxl` engine, so all three outcomes are reachable.
    let engine = "sdxl";
    let make_spec = || LoadSpec::new(WeightsSource::Dir(dir.to_path_buf())).with_quant(Quant::Q8);
    let disk = sum_safetensors_bytes(&dir);
    // Derive `te` through the SAME footprint-preferred path the gate uses
    // (`resolve_text_encoder_bytes`), NOT the bare `text_encoder*` subdir scan. If sdxl's declared
    // per-component footprint TE differs from the subdir sum, the gate's internal staged peak would
    // shift off this test's midpoint `cap_seq` window (~0.7 GiB each side) and the Sequential assertion
    // would flake. Querying the footprint here keeps the derived caps exactly on the gate's staged peak
    // (sc-10863).
    let footprint_te = gen_core::footprint(engine, &make_spec())
        .ok()
        .flatten()
        .map(|fp| fp.text_encoder);
    let te = resolve_text_encoder_bytes(footprint_te, &dir);
    let resident_peak = predicted_peak_gb(disk).expect("weights measured");
    let staged_peak = predicted_sequential_peak_gb(disk, te).expect("weights measured");
    let gib = 1024.0 * 1024.0 * 1024.0;
    assert!(
        staged_peak < resident_peak,
        "need a text-encoder split for a Sequential window (staged {staged_peak:.1} < resident {resident_peak:.1})"
    );
    println!(
        "[ab] illustrious q8: disk {:.2} GiB, te {:.2} GiB ⇒ predicted resident peak {resident_peak:.1} GiB, staged peak {staged_peak:.1} GiB",
        disk as f64 / gib,
        te as f64 / gib,
    );

    let set_cap = |gb: f64| std::env::set_var(MLX_MEMORY_CAP_ENV, format!("{gb}"));

    // (C) cap BELOW the staged peak ⇒ reject even one-component-at-a-time, with the actionable message.
    let cap_reject = staged_peak - 2.0;
    set_cap(cap_reject);
    let err = apply_residency_policy(make_spec(), engine)
        .expect_err("cap below the staged peak must reject");
    let crate::WorkerError::InvalidPayload(msg) = &err else {
        panic!("expected an actionable InvalidPayload reject, got {err:?}");
    };
    assert!(msg.contains("unified memory"), "actionable reject: {msg}");
    assert!(
        msg.contains("one component at a time"),
        "reject names the staged requirement: {msg}"
    );
    println!("[ab] cap {cap_reject:.1} GiB (< staged) ⇒ REJECT: {msg}");

    // (B) cap BETWEEN the staged and resident peaks ⇒ Sequential (won't fit resident, staged will).
    let cap_seq = (staged_peak + resident_peak) / 2.0;
    set_cap(cap_seq);
    let spec = apply_residency_policy(make_spec(), engine).expect("staged fits ⇒ Ok(Sequential)");
    assert_eq!(
        spec.offload_policy,
        OffloadPolicy::Sequential,
        "a cap between staged and resident selects Sequential"
    );
    println!("[ab] cap {cap_seq:.1} GiB (staged < cap < resident) ⇒ SEQUENTIAL (mlx_sequential_residency_selected)");

    // (A) cap ABOVE the resident peak ⇒ Resident, and a real generation confirms the calibrated value
    //     lets the hold-all path actually load + render.
    let cap_resident = resident_peak + 8.0;
    set_cap(cap_resident);
    let spec = apply_residency_policy(make_spec(), engine).expect("fits resident ⇒ Ok(Resident)");
    assert_eq!(
        spec.offload_policy,
        OffloadPolicy::Resident,
        "a cap above the resident peak keeps the warm Resident path"
    );
    println!(
        "[ab] cap {cap_resident:.1} GiB (> resident) ⇒ RESIDENT; running a real generation ..."
    );
    let generator =
        gen_core::load(engine, &spec).unwrap_or_else(|e| panic!("resident load: {e:?}"));
    let output = generator
        .generate(&sdxl_request(), &mut |_| {})
        .unwrap_or_else(|e| panic!("resident generate: {e:?}"));
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };
    let std = image_std(&image);
    assert!(
        std > DEGENERATE_STD_FLOOR_DEFAULT,
        "resident render looks degenerate (std {std:.2})"
    );
    println!(
        "[ab] RESIDENT generation ok (render std {std:.1}) — calibrated HEADROOM_GB confirmed through the real gate"
    );

    std::env::remove_var(MLX_MEMORY_CAP_ENV);
}
