//! mlx_fit_gate full finetune owner.
use super::*;

// ---------------------------------------------------------------------------------------------
// Full base fine-tune training memory-envelope gate (epic 14034 Mage-Flow / sc-14056)
// ---------------------------------------------------------------------------------------------
//
// A LoRA/LoKr run freezes the base and trains a tiny adapter, so it rides the base-installed check and
// the generation-style residency of a frozen model. A **full base fine-tune** (sc-14056) is a
// different memory regime: it makes EVERY DiT weight trainable, so the resident state is the
// full-precision master weights PLUS the optimizer's per-parameter state PLUS the gradients, and —
// because gradient (activation) checkpointing is not yet ported (sc-14989) — the dense retained-graph
// backward holds the whole forward's activations at once. That envelope grows far beyond the on-disk
// weight bytes and, at production resolution, exceeds any consumer Mac.
//
// This gate is the training analogue of the generation `apply_residency_policy` admission check, and
// it REUSES this module's machinery per sc-14056: `sum_safetensors_bytes` for the dense DiT bytes,
// `resolve_budget` + the `sysctl hw.memsize` probe (+ `SCENEWORKS_MLX_MEMORY_CAP_ENV` emulation) for
// the platform-aware unified-memory budget, and the typed legacy unified reserve. It is a pre-flight
// admission check (an MLX overcommit SIGKILLs the process — it cannot be caught after the fact, the
// same reason the generation gate and the Mochi gate are pre-flight), so a full-tune this machine
// cannot hold is refused up front with an actionable message rather than an uncatchable mid-run kill.
//
// The envelope is deliberately CONSERVATIVE (it errs toward rejection): under-predicting is the
// dangerous direction — it would advertise a production-resolution full-tune as fitting when it will
// not, exactly the failure sc-14056 calls out. The state multiplier is exact arithmetic *given the
// configuration* — it is 8× at `gradient_accumulation == 1` and 10× above it, because the engine's
// accumulator holds one more full f32 gradient map for the length of the window. The activation terms
// are a documented pre-grad-checkpointing ESTIMATE, uncalibrated in both directions against MLX's
// fast-SDPA vjp; sc-15038 tracks measuring them on-device with `mlx::get_peak_memory` the way
// sc-14053 calibrated the quant envelopes.

/// Peak resident training state as a multiple of the dense DiT's **bf16 on-disk** bytes: f32 master
/// weights (2×) + AdamW first/second moments in f32 (4×) + f32 gradients (2×). The in-trace bf16
/// weight reconstruction is a transient freed before the optimizer step, so this 8× step-peak
/// dominates. This term is resolution-independent and is exact arithmetic (a 4B-parameter model needs
/// ~8× its bf16 bytes of state regardless of resolution), so it alone gates full fine-tunes off Macs
/// that are simply too small for the optimizer state.
///
/// It is the **`gradient_accumulation == 1`** figure — see
/// [`FULL_FINETUNE_ACCUM_MULTIPLIER`] for the extra map an accumulation window holds.
pub(super) const FULL_FINETUNE_STATE_MULTIPLIER: f64 = 8.0;

/// Extra peak state, again as a multiple of the bf16 on-disk DiT bytes, held **only** when
/// `gradient_accumulation > 1`.
///
/// The engine's `accumulate_grads` (mlx-gen `train/lora.rs`) keeps a running accumulator that is a
/// FULL f32 gradient map (2× the bf16 bytes) alive for the whole window, *in addition to* the
/// per-micro-step gradient map already counted in [`FULL_FINETUNE_STATE_MULTIPLIER`] — both are live
/// simultaneously inside the summing loop. So the real step peak is 10×, not 8×, and it is the
/// accumulator that pushes a 4B full tune from ~61 GiB of state to ~77 GiB.
///
/// This was the omission that made the "exact arithmetic" claim wrong in the first cut of this gate
/// (sc-14056 review): `gradient_accumulation` is a user-exposed knob (Training Studio "Gradient
/// accumulation"), so a 64–80 GB Mac was predicted at ~63 GiB, admitted, and then hard-killed —
/// an MLX overcommit is a SIGKILL the worker cannot catch, which is the whole reason this gate is
/// pre-flight. `average_grads` afterwards holds at most the same two full maps (its input and its
/// output), so it needs no term of its own; 10× bounds both phases.
pub(super) const FULL_FINETUNE_ACCUM_MULTIPLIER: f64 = 2.0;

/// Retained attention bytes per (query,key) token pair, summed over all heads and blocks, that the
/// dense backward keeps WITHOUT gradient checkpointing (sc-14989): Mage's `num_heads = 24` × `depth =
/// 12` × f32 (4 bytes) × a conservative ×3 retain factor for the score / softmax / scaled-score
/// intermediates. This term is QUADRATIC in the packed token count and is what makes a
/// production-resolution full-tune exceed even a large Mac until checkpointing lands.
pub(super) const FULL_FINETUNE_ATTN_BYTES_PER_TOKEN_PAIR: f64 = 24.0 * 12.0 * 4.0 * 3.0;

/// Retained feature-map bytes per token, summed over the block stack: ≈24 retained intermediates ×
/// `hidden_size = 3072` × `depth = 12` × f32 (4 bytes). Linear in the packed token count.
pub(super) const FULL_FINETUNE_FEATURE_BYTES_PER_TOKEN: f64 = 24.0 * 3072.0 * 12.0 * 4.0;

/// The packed token count a full fine-tune trains at `resolution`: the trainer buckets the edge to a
/// multiple of 32, divides by the 16× Mage-VAE stride, and — `patch_size == 1` — packs exactly the
/// square latent grid (`grid²`).
pub(super) fn full_finetune_tokens(resolution: u32) -> f64 {
    let edge = (resolution / 32).max(1) * 32;
    let grid = (edge / 16) as f64;
    grid * grid
}

/// Predicted unified-memory peak (GiB) of a full base fine-tune of a `dit_bytes` (bf16 on-disk) DiT at
/// `resolution` with `gradient_accumulation` micro-steps per optimizer update, WITHOUT gradient
/// checkpointing. `None` when `dit_bytes == 0` (nothing installed → no signal → never block),
/// mirroring [`predicted_peak_gb`].
pub(super) fn full_finetune_peak_gb(
    dit_bytes: u64,
    resolution: u32,
    gradient_accumulation: u32,
) -> Option<f64> {
    if dit_bytes == 0 {
        return None;
    }
    let dit_gb = dit_bytes as f64 / BYTES_PER_GIB;
    let tokens = full_finetune_tokens(resolution);
    // The accumulator map exists only when the window is longer than one micro-step; `0` and `1` both
    // mean "step every micro-step" to the engine, so neither pays the term.
    let state_multiplier = if gradient_accumulation > 1 {
        FULL_FINETUNE_STATE_MULTIPLIER + FULL_FINETUNE_ACCUM_MULTIPLIER
    } else {
        FULL_FINETUNE_STATE_MULTIPLIER
    };
    let state = dit_gb * state_multiplier;
    let attention = tokens * tokens * FULL_FINETUNE_ATTN_BYTES_PER_TOKEN_PAIR / BYTES_PER_GIB;
    let feature = tokens * FULL_FINETUNE_FEATURE_BYTES_PER_TOKEN / BYTES_PER_GIB;
    Some(state + attention + feature + crate::fit_gate::LEGACY_UNIFIED_FALLBACK_RESERVE_GB)
}

/// The pure full-fine-tune admission decision: `Some(message)` when the predicted peak overflows the
/// machine's unified-memory budget, `None` to admit. Missing either signal admits (never block without
/// evidence — the fit-gate invariant shared with [`fit_decision`] and [`mochi_fit_error`]).
pub(super) fn full_finetune_fit_message(
    model_label: &str,
    dit_bytes: u64,
    resolution: u32,
    gradient_accumulation: u32,
    budget: Option<MlxMemoryBudget>,
) -> Option<String> {
    let (needed_gb, budget) = (
        full_finetune_peak_gb(dit_bytes, resolution, gradient_accumulation)?,
        budget?,
    );
    crate::fit_gate::exceeds_budget(needed_gb, budget.total_gb).then(|| {
        full_finetune_too_big_message(
            model_label,
            needed_gb,
            budget.total_gb,
            resolution,
            gradient_accumulation,
        )
    })
}

/// The actionable over-budget rejection for a full base fine-tune — names the model, the constraint
/// (unified memory), what it needs and what the machine has, and the levers that actually move the
/// dominant terms: a lower training resolution, a LoRA adapter instead of a full fine-tune, or a bigger
/// Mac — and states plainly that a production-resolution full fine-tune needs gradient checkpointing
/// (sc-14989), which is not yet available.
pub(super) fn full_finetune_too_big_message(
    model_label: &str,
    needed_gb: f64,
    available_gb: f64,
    resolution: u32,
    gradient_accumulation: u32,
) -> String {
    // Naming gradient accumulation only when it is actually costing memory keeps the advice
    // actionable: at accumulation 1 there is no accumulator map and "lower it" would be noise.
    let accum_lever = if gradient_accumulation > 1 {
        format!(
            " Gradient accumulation is {gradient_accumulation}, which holds an extra full-size \
             gradient buffer for the whole window — setting it to 1 removes that buffer."
        )
    } else {
        String::new()
    };
    format!(
        "A full base fine-tune of {model_label} at {resolution}px needs ~{needed} GB of unified \
         memory (the full-precision master weights, the optimizer state, and — without gradient \
         checkpointing — the whole retained backward graph) but this machine has ~{available} GB. \
         Lower the training resolution, train a LoRA adapter instead of a full fine-tune, or run on \
         a Mac with more memory.{accum_lever} A production-resolution full fine-tune needs gradient \
         (activation) checkpointing (sc-14989), which is not yet available.",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
    )
}

/// Live, platform-aware **full base fine-tune** memory pre-flight for the training submit gate
/// (sc-14056). Sums the base model's dense DiT bytes (`<base_model_dir>/transformer`), resolves this
/// machine's unified-memory budget with the same `sysctl hw.memsize` probe + `SCENEWORKS_MLX_MEMORY_
/// CAP_GB` emulation the generation fit gate uses, and returns a clear rejection message when a full
/// fine-tune at `resolution` with `gradient_accumulation` micro-steps per update won't fit — or
/// `None` to permit. `gradient_accumulation` is load-bearing, not cosmetic: a window longer than one
/// micro-step keeps an extra full-size f32 gradient buffer resident, which is the difference between
/// admitting and refusing a 4B full tune on a 64–80 GB Mac. Permits (returns `None`) whenever there
/// is no signal: the base isn't installed (`transformer/` sums to 0 bytes — the base-installed gate
/// covers that case), or the platform probe yields no budget (off-macOS, where a full-tune would run
/// on a different backend with its own envelope). Only the full-fine-tune path needs this; a LoRA run
/// trains against a frozen base and is covered by the base-installed check.
///
/// Re-exported publicly from the crate root as `sceneworks_worker::full_finetune_memory_error` so the
/// rust-api training submit gate can call it alongside `training_base_model_status` /
/// `training_disk_space_error`.
pub fn full_finetune_memory_error(
    base_model_dir: &Path,
    resolution: u32,
    gradient_accumulation: u32,
    model_label: &str,
) -> Option<String> {
    let dit_bytes = sum_safetensors_bytes(&base_model_dir.join("transformer"));
    let budget = resolve_budget(probe_total_unified_memory_gib(), mlx_memory_cap_gb());
    full_finetune_fit_message(
        model_label,
        dit_bytes,
        resolution,
        gradient_accumulation,
        budget,
    )
}
