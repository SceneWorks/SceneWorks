//! On-device RESOLUTION sweep of the MLX activation transient (sc-16195, epic 15448).
//!
//! Purpose: supply the measurements [`crate::mlx_fit_gate`]'s request-scoped generic estimator needs
//! in order to stop scaling its whole flat activation headroom with megapixels. The sc-10863
//! calibration behind `HEADROOM_GB` measured four tiers at 1024² ONLY, so the constant is a
//! 1024²-worst-case with no shape, and `generic_total_peak_bytes` multiplied all of it by `w·h/1024²`
//! — including the fixed OS/app reserve, which does not scale with anything.
//!
//! The story that commissioned this expected the sweep to justify a SUBLINEAR area term, from two
//! sc-5567 points. It did the opposite: above 1024² the transient is proportional to area to within
//! 1.97% across seven tiers, so only the fixed reserve was pulled out and the area term was left
//! linear. Full results and the refutation in `docs/sc-16195/README.md`.
//!
//! This harness is the sibling of [`crate::footprint_measure`] with the axis rotated: that one
//! measures ONE resolution across tiers, this one measures ONE tier across resolutions. It samples
//! the same process-global MLX counters (`mlx_rs::memory::{reset_peak_memory, get_active_memory,
//! get_peak_memory, clear_cache}`) that `generator_cache.rs` publishes to telemetry — `ps` RSS and
//! `getrusage` maxrss do NOT observe the Metal allocator's high-water mark, so an out-of-process
//! sampler cannot produce these numbers.
//!
//! ## What one row measures
//!
//! Per resolution cell, on a WARM generator (weights already materialized):
//!
//! ```text
//!   clear_cache(); reset_peak_memory();      // peak starts at the resident weights
//!   resident_before = get_active_memory()    // the weights, no transient
//!   generate(w, h)                           // one real render
//!   peak = get_peak_memory()                 // weights + this cell's activation high-water
//!   clear_cache(); resident_after = get_active_memory()
//!   transient = peak - resident_after        // the AREA-DEPENDENT term the estimator must model
//! ```
//!
//! `transient` is the quantity the fit gate's headroom stands for, isolated from the weights. The
//! `clear_cache()` before sampling `resident_after` is the sc-8516 credibility fix — without it the
//! generation's freeable scratch is folded INTO resident and the transient is understated.
//!
//! WARM is the right regime for this seam: `mlx_fit_gate::evaluate_request` runs per request, after
//! the generator-cache lookup and immediately before generation, and its formula is
//! `asset_bytes + headroom·scale` — i.e. resident weights plus the marginal activation cost of THIS
//! request's geometry. The cold load+gen path is gated separately and resolution-blind
//! (`decide_residency_for_spec`, the flat `HEADROOM_GB`), so its row is reported as a bonus
//! (`"cell":"cold"`) rather than mixed into the fit.
//!
//! Read that cold row as a LOWER BOUND on the load+gen ceiling, not as the ceiling. MLX materializes
//! lazily, so on the first pass the weights are still being realized while the activations allocate —
//! the two are never fully co-resident. Measured: the cold peak comes in BELOW the warm peak at the
//! same geometry in all seven swept tiers.
//!
//! ## Running it
//!
//! RUN ONE TIER PER PROCESS — the same rule [`crate::footprint_measure`] enforces (sc-8925), and for
//! the same reason: the MLX counters are process-global, so a second TIER's weights land on top of
//! the first tier's allocator residue. [`SWEEP_RAN`] enforces it, so an unfiltered
//! `cargo test … sweep -- --ignored` fails loudly instead of emitting corrupt rows.
//!
//! What is relaxed here is one tier per *cell*, not per tier. Within a single tier the weights are
//! the same arrays throughout and `reset_peak_memory()` re-bases the high-water mark per cell, so a
//! whole resolution sweep is measured in one load. Narrowing `RS_CELLS` to a single cell therefore
//! gives a fresh-process cross-check of that cell's in-process row — worth doing for the last cell of
//! a sweep, the one that could in principle have carried allocator state from the earlier ones.
//! `docs/sc-16195/README.md` §2 records such a cross-check.
//!
//! ```text
//! cargo test -p sceneworks-worker --release sweep_illustrious_v1_q8 -- --ignored --nocapture
//! cargo test -p sceneworks-worker --release sweep_krea_2_turbo_bf16 -- --ignored --nocapture
//!
//! # fresh-process cross-check of one in-process row:
//! RS_CELLS=2048x2048 cargo test -p sceneworks-worker --release \
//!     sweep_illustrious_v1_q8 -- --ignored --nocapture
//! ```
//!
//! Each cell prints one machine-parseable line for the fit:
//!
//! ```text
//! [[RESSWEEP]] {"model":"…","tier":"…","cell":"1024x1024","width":1024,"height":1024,
//!               "megapixels":1.0,"diskBytes":…,"residentBytes":…,"peakBytes":…,"transientBytes":…}
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use gen_core::{GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};

use super::smoke_support::{env_or, image_std, is_all_zero, DEGENERATE_STD_FLOOR_DEFAULT};

/// One-tier-per-process guard, mirroring `footprint_measure`'s `FOOTPRINT_RAN` (sc-8925). The MLX
/// counters and the allocator's peak high-water mark are PROCESS-GLOBAL, so measuring a second tier
/// in the same binary invocation folds the first tier's resident weights into its numbers. `sweep_tier`
/// flips this once and asserts it was previously false, so an unfiltered
/// `cargo test … sweep -- --ignored` run fails loudly instead of silently emitting corrupt
/// `[[RESSWEEP]]` lines. Multiple RESOLUTIONS in one process are fine and are the point — see the
/// module doc.
static SWEEP_RAN: AtomicBool = AtomicBool::new(false);

/// The sweep's default resolution cells — the story's required minimum set: a sub-1024² point, the
/// 1024² calibration anchor, a portrait 1.5 MP, the 1152×2048 cell from the sc-16194 report, and a
/// 4 MP square. Square and non-square points at similar areas are both present on purpose: they
/// separate "scales with AREA" from "scales with the LONGEST SIDE" (attention over a 1D token
/// sequence is area-shaped; a tiled VAE decode is not). Override with
/// `RS_CELLS=512x512,2048x2048`.
const DEFAULT_CELLS: &[(u32, u32)] = &[
    (512, 512),
    (1024, 1024),
    (1024, 1536),
    (1152, 2048),
    (2048, 2048),
];

/// Parse `RS_CELLS` (`WxH,WxH,…`) or fall back to [`DEFAULT_CELLS`].
fn sweep_cells() -> Vec<(u32, u32)> {
    let raw = env_or("RS_CELLS", "");
    let raw = raw.trim();
    if raw.is_empty() {
        return DEFAULT_CELLS.to_vec();
    }
    raw.split(',')
        .map(str::trim)
        .filter(|cell| !cell.is_empty())
        .map(|cell| {
            let (w, h) = cell
                .split_once(['x', 'X'])
                .unwrap_or_else(|| panic!("RS_CELLS entry {cell:?} must look like 1024x1536"));
            (
                w.trim()
                    .parse()
                    .unwrap_or_else(|_| panic!("RS_CELLS width {w:?}")),
                h.trim()
                    .parse()
                    .unwrap_or_else(|_| panic!("RS_CELLS height {h:?}")),
            )
        })
        .collect()
}

/// Locate a cached `SceneWorks/<repo>` turnkey snapshot whose `<tier>/` subdir carries `sentinel`.
/// Same resolution rule as [`crate::footprint_measure`] so both harnesses see the same on-disk set.
fn cached_tier_dir(repo: &str, tier: &str, sentinel: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let repo_cache = repo.replace('/', "--");
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(format!("models--{repo_cache}"))
        .join("snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|entry| entry.ok())
        .find_map(|entry| {
            let tier_dir = entry.path().join(tier);
            tier_dir.join(sentinel).is_file().then_some(tier_dir)
        })
}

/// Resolve the tier dir from an explicit override env (a turnkey root OR the tier dir itself) or the
/// HF cache, panicking with a download hint if neither resolves.
fn resolve_tier_dir(env_key: &str, repo: &str, tier: &str, sentinel: &str) -> PathBuf {
    if let Ok(path) = std::env::var(env_key) {
        let path = path.trim();
        if !path.is_empty() {
            let root = PathBuf::from(path);
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

/// The load-quant for a tier, or `None` for the dense `bf16` tier (the worker's dense path). Packed
/// q4/q8 subdirs auto-detect their quant, so `with_quant` just names the tier.
fn quant_for(tier: &str) -> Option<Quant> {
    match tier {
        "q4" => Some(Quant::Q4),
        "q8" => Some(Quant::Q8),
        _ => None,
    }
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// One generation at `(width, height)` with the peak counter re-based first. Returns
/// `(peak_bytes, resident_after_bytes, render_std)`; the caller derives
/// `transient = peak − resident_after`.
fn measure_cell(
    label: &str,
    generator: &dyn gen_core::Generator,
    request: &GenerationRequest,
) -> (u64, u64, f64) {
    // Re-base: drop the allocator's free cache so the high-water mark starts at the live weights and
    // this cell's peak is its own, not the previous (possibly larger) cell's leftovers.
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let resident_before = mlx_rs::memory::get_active_memory() as u64;

    let output = generator
        .generate(request, &mut |_| {})
        .unwrap_or_else(|error| panic!("{label} generate: {error:?}"));
    let image = match output {
        GenerationOutput::Images(mut images) => images.pop().expect("engine returned no image"),
        other => panic!("expected Images output, got {other:?}"),
    };
    // A BLANK frame is the only render outcome that invalidates the measurement: it means the
    // pipeline bailed and never allocated a real working set. A low-contrast-but-real render at an
    // out-of-distribution aspect (several families here list only 1024²-ish resolutions, and this
    // sweep deliberately probes past them) is still a genuine activation profile, so the softer
    // `DEGENERATE_STD_FLOOR_DEFAULT` is REPORTED on the row rather than asserted — dropping such a
    // cell would silently thin the sweep exactly where the estimator most needs points.
    assert!(
        !is_all_zero(&image),
        "{label} rendered a blank frame — the pipeline bailed, so its measured transient is not a \
         real activation working set"
    );
    let std = image_std(&image);
    if std <= DEGENERATE_STD_FLOOR_DEFAULT {
        println!(
            "[sweep] WARN {label}: low-contrast render (std {std:.2}) — kept, flagged on the row"
        );
    }

    let peak = mlx_rs::memory::get_peak_memory() as u64;
    // Release this cell's freeable scratch BEFORE sampling resident, so `peak − resident` is the
    // activation transient and not an artifact of retained buffers (the sc-8516 credibility fix).
    mlx_rs::memory::clear_cache();
    let resident_after = mlx_rs::memory::get_active_memory() as u64;
    // The whole credibility of `transient = peak − resident` rests on that `clear_cache()` having
    // actually drained the generation's freeable scratch. `get_cache_memory` is the direct proof:
    // anything still sitting in the allocator's free cache here would be scratch wrongly counted as
    // resident, understating the transient. Asserted rather than printed, because a silent
    // regression would corrupt the calibration this harness exists to produce.
    let cached_after = mlx_rs::memory::get_cache_memory() as u64;
    assert!(
        cached_after <= resident_after / 100,
        "{label}: clear_cache() left {cached_after} bytes in the allocator cache against \
         {resident_after} resident — the transient would be understated by whatever did not drain"
    );
    println!(
        "[sweep] {label}: resident {:.2} GiB (pre-gen {:.2}, cache after drain {:.3}) | peak {:.2} GiB | transient {:.2} GiB | std {std:.1}",
        resident_after as f64 / GIB,
        resident_before as f64 / GIB,
        cached_after as f64 / GIB,
        peak as f64 / GIB,
        peak.saturating_sub(resident_after) as f64 / GIB,
    );
    (peak, resident_after, std)
}

/// Emit one machine-parseable sweep row.
#[allow(clippy::too_many_arguments)]
fn emit_row(
    model: &str,
    tier: &str,
    cell: &str,
    width: u32,
    height: u32,
    disk: u64,
    resident: u64,
    peak: u64,
    std: f64,
) {
    let megapixels = f64::from(width) * f64::from(height) / (1024.0 * 1024.0);
    let transient = peak.saturating_sub(resident);
    println!(
        "[[RESSWEEP]] {{\"model\":\"{model}\",\"tier\":\"{tier}\",\"cell\":\"{cell}\",\
         \"width\":{width},\"height\":{height},\"megapixels\":{megapixels:.4},\
         \"diskBytes\":{disk},\"residentBytes\":{resident},\"peakBytes\":{peak},\
         \"transientBytes\":{transient},\"renderStd\":{std:.2},\"lowContrast\":{}}}",
        std <= DEGENERATE_STD_FLOOR_DEFAULT
    );
}

/// Load `(model, tier)` once, then measure the activation transient at every sweep cell.
///
/// The first generation doubles as the COLD row: MLX materializes lazily, so nothing is on the
/// allocator until a forward pass touches the weights, and that first pass therefore carries the
/// load transient on top of its own activations. It is emitted as `"cell":"cold"` and then repeated
/// warm, so the warm rows that feed the fit are all in the same regime.
fn sweep_tier(
    model: &str,
    tier: &str,
    engine_id: &str,
    dir: &Path,
    make_request: impl Fn(u32, u32) -> GenerationRequest,
) {
    // Enforce the one-tier-per-process invariant before touching the GPU (see `SWEEP_RAN`): a second
    // tier in this process would report numbers contaminated by the first tier's resident weights.
    assert!(
        !SWEEP_RAN.swap(true, Ordering::SeqCst),
        "resolution sweep measured a second TIER in one process — the MLX peak/active counters are \
         process-global, so run exactly one `sweep_<x>` per `cargo test … -- --ignored` invocation \
         (see the module doc; multiple RESOLUTIONS per process are fine and are the point). The \
         just-attempted {model} {tier} sweep would be corrupt."
    );

    let cells = sweep_cells();
    assert!(!cells.is_empty(), "no sweep cells");

    let disk = crate::mlx_fit_gate::sum_safetensors_bytes(dir);
    println!(
        "[sweep] {model} ({tier}) from {} — disk {:.2} GiB, cells {cells:?}",
        dir.display(),
        disk as f64 / GIB,
    );

    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.to_path_buf()));
    if let Some(quant) = quant_for(tier) {
        spec = spec.with_quant(quant);
    }
    let generator = crate::inference_runtime::load(engine_id, &spec)
        .unwrap_or_else(|error| panic!("load {engine_id} ({tier}): {error:?}"));

    // COLD row: the first pass is what materializes the weights, so its peak spans load + one
    // generation — the regime the resolution-blind load-time gate budgets for. It is a LOWER bound on
    // that ceiling, not the ceiling: lazy materialization means the weights and the full activation
    // set are never simultaneously live, which is why every swept tier's cold peak lands below its
    // warm peak at the same geometry. Measured at the FIRST sweep cell so the caller controls which
    // geometry pays for materialization.
    let (cold_width, cold_height) = cells[0];
    let (cold_peak, cold_resident, cold_std) = measure_cell(
        &format!("{model} {tier} cold {cold_width}x{cold_height}"),
        generator.as_ref(),
        &make_request(cold_width, cold_height),
    );
    emit_row(
        model,
        tier,
        "cold",
        cold_width,
        cold_height,
        disk,
        cold_resident,
        cold_peak,
        cold_std,
    );

    for (width, height) in cells {
        let label = format!("{model} {tier} {width}x{height}");
        let (peak, resident, std) =
            measure_cell(&label, generator.as_ref(), &make_request(width, height));
        emit_row(
            model,
            tier,
            &format!("{width}x{height}"),
            width,
            height,
            disk,
            resident,
            peak,
            std,
        );
    }
}

/// Shared prompt/seed; `steps` and `guidance` follow each engine's own recipe (`engines.rs`
/// `ModelRow` defaults), because a guidance-distilled engine rejects a guidance value outright and a
/// true-CFG engine runs a second uncond forward whose activations are part of what we are measuring.
fn request_for(steps: u32, guidance: Option<f32>) -> impl Fn(u32, u32) -> GenerationRequest {
    let prompt = env_or(
        "RS_PROMPT",
        "a photorealistic portrait of a red fox in a sunlit autumn forest, sharp focus",
    );
    let steps: u32 = env_or("RS_STEPS", &steps.to_string())
        .parse()
        .expect("RS_STEPS");
    move |width, height| GenerationRequest {
        prompt: prompt.clone(),
        width,
        height,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        guidance,
        ..Default::default()
    }
}

const SDXL_SENTINEL: &str = "unet/diffusion_pytorch_model.safetensors";
const SDXL_BF16_SENTINEL: &str = "unet/diffusion_pytorch_model.fp16.safetensors";
// Lens AND Krea pack the DiT under transformer/diffusion_pytorch_model.safetensors …
const LENS_SENTINEL: &str = "transformer/diffusion_pytorch_model.safetensors";
// … but Krea's DENSE bf16 DiT is SHARDED (…-0000N-of-0000M.safetensors), so its tier-presence
// sentinel is the shard index the plain name would miss (the same split lens-turbo bf16 has).
const SHARDED_DIT_SENTINEL: &str = "transformer/diffusion_pytorch_model.safetensors.index.json";
// … while Z-Image AND Qwen-Image pack the DiT under transformer/model.safetensors.
const ZIMAGE_SENTINEL: &str = "transformer/model.safetensors";

// ── The sweep set ────────────────────────────────────────────────────────────────────────────────
// Chosen to span the two axes the estimator has to survive: ARCHITECTURE (SDXL UNet vs single-stream
// DiT vs a DiT whose VAE decode is tiled) and TIER SHAPE (packed q4/q8 vs dense bf16). Every entry
// is already on disk, so the sweep triggers no download.

/// SDXL UNet, packed q8 — one of the two 14.04 GiB@1024² transients the sc-10863 `HEADROOM_GB`
/// calibration is built on, so its curve is what the constant must stay conservative against.
#[test]
#[ignore = "sc-16195 resolution sweep; needs SceneWorks/illustrious-xl-v1-mlx q8 cached + an Apple-Silicon Mac"]
fn sweep_illustrious_v1_q8() {
    let dir = resolve_tier_dir(
        "RS_ILL_V1_DIR",
        "SceneWorks/illustrious-xl-v1-mlx",
        "q8",
        SDXL_SENTINEL,
    );
    sweep_tier(
        "illustrious_xl_v1",
        "q8",
        "sdxl",
        &dir,
        request_for(30, Some(7.0)),
    );
}

/// SDXL UNet, DENSE bf16 — the same architecture as the row above with the tier shape flipped, so a
/// difference between the two is a tier effect rather than an architecture effect.
#[test]
#[ignore = "sc-16195 resolution sweep; needs SceneWorks/sdxl-base-mlx bf16 cached + an Apple-Silicon Mac"]
fn sweep_sdxl_bf16() {
    let dir = resolve_tier_dir(
        "RS_SDXL_BF16_DIR",
        "SceneWorks/sdxl-base-mlx",
        "bf16",
        SDXL_BF16_SENTINEL,
    );
    sweep_tier("sdxl", "bf16", "sdxl", &dir, request_for(30, Some(7.0)));
}

/// Z-Image Turbo, packed q4 — a small guidance-distilled DiT (no uncond forward), the light end of
/// the architecture spread.
#[test]
#[ignore = "sc-16195 resolution sweep; needs SceneWorks/z-image-turbo-mlx q4 cached + an Apple-Silicon Mac"]
fn sweep_z_image_turbo_q4() {
    let dir = resolve_tier_dir(
        "RS_ZIMAGE_TURBO_Q4_DIR",
        "SceneWorks/z-image-turbo-mlx",
        "q4",
        ZIMAGE_SENTINEL,
    );
    sweep_tier(
        "z_image_turbo",
        "q4",
        "z_image_turbo",
        &dir,
        request_for(8, None),
    );
}

/// Krea 2 Turbo, packed q8 — the family in the sc-16194 report (Qwen3-VL-4B TE + 28-block
/// single-stream DiT + Qwen-Image VAE), measured on the cheap tier.
#[test]
#[ignore = "sc-16195 resolution sweep; needs SceneWorks/krea-2-turbo-mlx q8 cached + an Apple-Silicon Mac"]
fn sweep_krea_2_turbo_q8() {
    let dir = resolve_tier_dir(
        "RS_KREA_TURBO_DIR",
        "SceneWorks/krea-2-turbo-mlx",
        "q8",
        LENS_SENTINEL,
    );
    sweep_tier(
        "krea_2_turbo",
        "q8",
        "krea_2_turbo",
        &dir,
        request_for(8, None),
    );
}

/// Krea 2 Turbo, DENSE bf16 — the exact cell sc-16194 reported (1152×2048 on a 33.22 GiB tier), so
/// the retuned estimator can be checked against a measured number for the reported request.
#[test]
#[ignore = "sc-16195 resolution sweep; needs SceneWorks/krea-2-turbo-mlx bf16 cached + an Apple-Silicon Mac"]
fn sweep_krea_2_turbo_bf16() {
    let dir = resolve_tier_dir(
        "RS_KREA_TURBO_DIR",
        "SceneWorks/krea-2-turbo-mlx",
        "bf16",
        SHARDED_DIT_SENTINEL,
    );
    sweep_tier(
        "krea_2_turbo",
        "bf16",
        "krea_2_turbo",
        &dir,
        request_for(8, None),
    );
}

/// Qwen-Image, packed q8 — the sc-10863 outlier whose VAE decode is TILED (sc-11747), i.e. the one
/// family already known to break the "transient grows with output area" assumption.
#[test]
#[ignore = "sc-16195 resolution sweep; needs SceneWorks/qwen-image-mlx q8 cached + an Apple-Silicon Mac"]
fn sweep_qwen_image_q8() {
    let dir = resolve_tier_dir(
        "RS_QWEN_IMAGE_Q8_DIR",
        "SceneWorks/qwen-image-mlx",
        "q8",
        ZIMAGE_SENTINEL,
    );
    sweep_tier(
        "qwen_image",
        "q8",
        "qwen_image",
        &dir,
        request_for(20, Some(4.0)),
    );
}

/// Lens, packed q4 — the other 14.04 GiB@1024² anchor from sc-10863, and the largest packed tier in
/// the set (17.67 GiB on disk).
#[test]
#[ignore = "sc-16195 resolution sweep; needs SceneWorks/lens-mlx q4 cached + an Apple-Silicon Mac"]
fn sweep_lens_q4() {
    let dir = resolve_tier_dir("RS_LENS_Q4_DIR", "SceneWorks/lens-mlx", "q4", LENS_SENTINEL);
    sweep_tier("lens", "q4", "lens", &dir, request_for(20, Some(5.0)));
}
