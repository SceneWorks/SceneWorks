//! mlx_fit_gate video owner.
use super::*;

/// Resolve the numeric tier that the loaded video checkpoint actually carries.
///
/// Provider-owned resolution runs first for video loaders that carry their tier outside
/// `LoadSpec.quantize` (Wan TI2V-5B and Bernini). LTX then resolves its immutable `split_model.json`.
/// An explicit assertion that disagrees with the checkpoint fails closed before selection.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn resolved_video_numeric_tier(
    engine_id: &str,
    spec: &LoadSpec,
) -> WorkerResult<MemoryNumericTier> {
    // Provider-owned load-exact video tiers win whenever the selected runtime exposes one. Wan and
    // Bernini read immutable packed config/header surfaces (Wan also carries the Q4 text-encoder Q8
    // floor); deriving from `LoadSpec.quantize` would misprice their normal prepacked loads, whose
    // request-side quant hint is deliberately `None`.
    #[cfg(target_os = "macos")]
    if let Some(tier) =
        runtime_macos::resolved_video_memory_numeric_tier(engine_id, spec).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "{engine_id} admission cannot resolve the provider-owned numeric tier: {error}"
            ))
        })?
    {
        return Ok(tier);
    }
    if engine_id != "ltx_2_3" {
        return Ok(spec_numeric_tier(engine_id, spec));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(path) => path,
        WeightsSource::File(path) => {
            return Err(WorkerError::InvalidPayload(format!(
                "ltx_2_3 admission expected a model directory, got {}",
                path.display()
            )));
        }
    };
    let manifest_path = root.join("split_model.json");
    let resolved_quant = if manifest_path.exists() {
        let raw = std::fs::read_to_string(&manifest_path).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "ltx_2_3 admission cannot read {}: {error}",
                manifest_path.display()
            ))
        })?;
        let value: Value = serde_json::from_str(&raw).map_err(|error| {
            WorkerError::InvalidPayload(format!(
                "ltx_2_3 admission cannot parse {}: {error}",
                manifest_path.display()
            ))
        })?;
        if value
            .get("quantized")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            match value
                .get("quantization_bits")
                .and_then(Value::as_i64)
                .unwrap_or(4)
            {
                4 => Some(gen_core::Quant::Q4),
                8 => Some(gen_core::Quant::Q8),
                bits => {
                    return Err(WorkerError::InvalidPayload(format!(
                        "ltx_2_3 admission refuses unsupported packed checkpoint tier q{bits} in {}",
                        manifest_path.display()
                    )));
                }
            }
        } else {
            None
        }
    } else {
        None
    };
    if spec.quantize != resolved_quant && spec.quantize.is_some() {
        return Err(WorkerError::InvalidPayload(format!(
            "ltx_2_3 admission quant assertion {:?} disagrees with checkpoint tier {:?} in {}",
            spec.quantize,
            resolved_quant,
            manifest_path.display()
        )));
    }
    Ok(MemoryNumericTier {
        precision: spec.precision,
        quant: resolved_quant,
        component_precision_floors: active_component_floors(
            declared_component_floors(engine_id),
            resolved_quant,
        ),
    })
}

// ---------------------------------------------------------------------------------------------
// Mochi 1: the FRAME-DEPENDENT decode fit gate (epic 1788 / sc-11992)
// ---------------------------------------------------------------------------------------------
//
// Why Mochi needs its own gate rather than riding the generic one above:
//
//  1. The generic gate is deliberately RESOLUTION-BLIND. `predicted_peak_gb` = Σweights +
//     HEADROOM_GB, where HEADROOM_GB is a 1024²-calibrated CONSTANT — the load-time seam cannot see
//     the request geometry (the generator is cached across resolutions, see the HEADROOM_GB note).
//     That structure is right for image models, whose transient is roughly request-independent once
//     calibrated. It is WRONG for Mochi: its AsymmVAE decode is UNTILED (`vae.decode(latents)`
//     materializes the whole clip — sc-12291), so the peak grows LINEARLY IN CLIP LENGTH. A 7-frame
//     and a 151-frame request differ by ~55 GiB on the same model. A constant cannot express that.
//
//  2. Mochi's `supports_sequential_offload: false` ⇒ the legacy admission override admits on WEIGHTS ALONE
//     (18.73 GiB fits almost any Mac), so the generic gate would happily admit a 151-frame job that
//     then needs ~79 GiB. The floor is correct for its own purpose (sc-12179: never wall-reject a
//     machine that used to work) but it is precisely what makes the generic gate unable to protect
//     Mochi.
//
//  3. This MUST be a pre-flight rejection, not a caught error. MLX's default error handler is
//     `exit(-1)` — a hard process kill (sc-12178/12179, GitHub #1544). An `exit(-1)` cannot be
//     mapped to a job error after the fact, so the only place to honor the epic's "actionable error,
//     not a crash" AC is BEFORE the decode allocates.
//
// The gate is therefore a per-REQUEST admission check (it sees frames + geometry), layered beside —
// not inside — the generic per-LOAD one, and it budgets on a DERIVED architectural formula rather
// than a calibration constant, because nothing has been measured on-device yet (B5/sc-11995 backfills
// `footprint.residentMemoryBytes`/`peakMemoryBytes`; sc-12291 tiles the decode and should then cut
// the frame term sharply).
//
// All three points above hold verbatim for the candle lane, which grew the same gate in sc-12306 —
// except (3), where a CUDA OOM is catchable rather than a SIGKILL, so the pre-flight buys an
// actionable message and minutes of un-wasted denoise rather than process survival. What remains here
// is MLX-SPECIFIC: the unified-memory budget probe, the typed legacy/evidence reserve, and the Mac-worded
// message. The shared ARITHMETIC lives in `crate::fit_gate` (`mochi_decode_peak_gb` /
// `mochi_needed_gb`); the candle half is `vram_gate::mochi_fit_error`. The on-disk scan
// (`mochi_resident_bytes`) is shared too — the hosted tier layout is one repo serving both lanes.

/// Predicted whole-generation peak (GiB) for an MLX Mochi request, in UNIFIED memory: the shared
/// backend-neutral arithmetic ([`crate::fit_gate::mochi_needed_gb`]) with the caller's unified reserve,
/// because on unified memory the OS draws from the same pool the model does.
///
/// The formula itself moved to [`crate::fit_gate`] when the candle video lane needed the identical
/// arithmetic (sc-12306) — every term in it is a fact about the MODEL (tensor shapes × the f32 dtype
/// both decoders pin), so duplicating it per lane would let the two gates disagree about one model.
/// The RESERVE is what stays lane-specific; see that function's note.
pub(super) fn mochi_needed_gb(
    weight_bytes: u64,
    frames: u32,
    width: u32,
    height: u32,
    reserve_gb: f64,
) -> Option<f64> {
    crate::fit_gate::mochi_needed_gb(weight_bytes, frames, width, height, reserve_gb)
}

/// The pure Mochi admission decision: `Some(error)` when the predicted peak overflows the budget,
/// `None` to admit. Missing either signal (unmeasurable weights / no budget) admits — the gate never
/// blocks without evidence, exactly like [`fit_decision`].
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn mochi_fit_error(
    model_label: &str,
    weight_bytes: u64,
    frames: u32,
    width: u32,
    height: u32,
    budget: Option<MlxMemoryBudget>,
) -> Option<WorkerError> {
    let budget = budget?;
    let reserve = crate::fit_gate::legacy_unified_reserve(budget.total_gb);
    let needed_gb = mochi_needed_gb(weight_bytes, frames, width, height, reserve.gb)?;
    crate::fit_gate::exceeds_budget(needed_gb, budget.total_gb).then(|| {
        mochi_too_big_error(
            model_label,
            needed_gb,
            budget.total_gb,
            frames,
            width,
            height,
            weight_bytes as f64 / BYTES_PER_GIB,
        )
    })
}

/// Build Mochi's actionable over-budget rejection. Follows the [`too_big_error`] convention — name
/// the model, explain the constraint ("unified memory"), state what it needs and what the machine
/// has — and adds the lever that is UNIQUE to Mochi: the clip length. The generic message's advice
/// ("choose a smaller quant tier, lower the resolution") is nearly useless here — Mochi has one
/// trained bucket (848×480) and the decode dwarfs the tier delta (q4→bf16 is ~11 GiB against a
/// ~60 GiB decode) — so the message leads with shortening the clip, which is the only lever that
/// moves the dominant term.
#[allow(clippy::too_many_arguments)]
pub(super) fn mochi_too_big_error(
    model_label: &str,
    needed_gb: f64,
    available_gb: f64,
    frames: u32,
    width: u32,
    height: u32,
    weights_gb: f64,
) -> WorkerError {
    WorkerError::InvalidPayload(format!(
        "{model_label} needs ~{needed} GB of unified memory to render a {frames}-frame \
         {width}x{height} clip (~{weights} GB of weights, held resident for the whole run, plus an \
         untiled VAE decode whose peak grows with clip length) but this machine has ~{available} \
         GB. Shorten the clip — the decode peak scales roughly linearly with duration — or run on a \
         Mac with more memory.",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
        weights = weights_gb.round() as i64,
    ))
}

/// Live pre-flight Mochi admission check (the seam the worker's Mochi generation arm calls before
/// loading). Resolves the same budget the generic gate uses — the real `sysctl hw.memsize` total,
/// overridable by [`MLX_MEMORY_CAP_ENV`] so a small Mac can be emulated in tests — and sums the
/// on-disk bytes the load will actually hold resident: the TIER dir (the AsymmDiT) plus the SHARED
/// `text_encoder/` + `vae/` siblings, which the provider resolves from the tier dir's PARENT
/// (`resolve_component_root`, mlx-gen-mochi/src/model.rs). Summing only the tier dir would miss the
/// ~9.7 GiB T5-XXL + VAE — over half the resident footprint.
///
/// `Ok(())` admits (including whenever there is no signal); `Err` is the actionable pre-crash
/// rejection.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn mochi_fit_check(
    model_label: &str,
    tier_dir: &Path,
    frames: u32,
    width: u32,
    height: u32,
) -> WorkerResult<()> {
    let budget = resolve_budget(probe_total_unified_memory_gib(), mlx_memory_cap_gb());
    match mochi_fit_error(
        model_label,
        mochi_resident_bytes(tier_dir),
        frames,
        width,
        height,
        budget,
    ) {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// The on-disk bytes a Mochi load holds resident: the tier dir (AsymmDiT) + the shared `text_encoder/`
/// and `vae/` components resolved from the tier dir's parent (the A6 shared-sibling layout). A
/// self-contained dir — the raw snapshot, where the components live under the dir itself — is summed
/// once, never double-counted, because the parent scan only adds SIBLING dirs of `tier_dir`.
///
/// **Shared by both lanes** (sc-12306), despite the module name: this describes the HOSTED REPO LAYOUT,
/// not MLX. `SceneWorks/mochi-1-mlx` serves candle too — A6's `.scales`-detect seam ingests the same
/// mlx-affine tiers 1:1 through the same `resolve_mochi_model_dir` — so the resident byte total is
/// identical off-Mac, and both providers set `supports_sequential_offload: false`, so both hold all
/// three components for the whole run. It stays here beside its `sum_safetensors_bytes` helper (which
/// has several other MLX callers) rather than moving to `fit_gate` with the arithmetic; the candle
/// video gate calls it through this path, exactly as `image_jobs/base.rs` already calls
/// `mlx_fit_gate::engine_supports_sequential` from the candle image lane (sc-12130).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn mochi_resident_bytes(tier_dir: &Path) -> u64 {
    let mut total = sum_safetensors_bytes(tier_dir);
    if let Some(parent) = tier_dir.parent() {
        for component in ["text_encoder", "vae"] {
            let dir = parent.join(component);
            // Only count a shared sibling — a self-contained tier dir already summed its own
            // components above (`sum_safetensors_bytes` recurses).
            if dir.is_dir() && !tier_dir.join(component).is_dir() {
                total += sum_safetensors_bytes(&dir);
            }
        }
    }
    total
}
