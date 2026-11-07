//! vram_gate video owner.
use super::*;

// ---------------------------------------------------------------------------------------------
// Mochi 1: the FRAME-DEPENDENT decode fit gate, candle/CUDA half (epic 1788 / sc-12306)
// ---------------------------------------------------------------------------------------------
//
// The candle twin of `mlx_fit_gate`'s Mochi gate. Both call the SAME arithmetic
// (`crate::fit_gate::mochi_needed_gb`); this half supplies the CUDA budget, the CUDA reserve, and the
// CUDA-worded message.
//
// Why Mochi needs its own gate here rather than riding `fit_decision` above:
//
//  1. `predicted_peak_gb` is MANIFEST-driven (`candle.vramGbByTier`), and **`mochi_1` has no `candle`
//     block at all** — it ships `mlx` only (builtin.models.jsonc:7233). So the generic gate returns
//     `None` ⇒ `Unknown` ⇒ admit. It could not protect Mochi even if the video lane called it. (No
//     candle VIDEO model has such a block; closing that is sc-12344.)
//
//  2. Even given a block, the generic gate is deliberately RESOLUTION-BLIND: a per-tier constant
//     calibrated at load time, where the seam cannot see the request geometry (the generator is cached
//     across resolutions). Mochi's AsymmVAE decode is UNTILED (candle-gen-mochi pipeline.rs:151 —
//     `vae.decode(&latents)` materializes the whole clip; sc-12291), so its peak grows LINEARLY IN CLIP
//     LENGTH: a 7-frame and a 151-frame request differ by ~55 GiB on the same model and the same tier.
//     A constant cannot express that.
//
//  3. Unlike the MLX lane, a CUDA OOM here is CATCHABLE (`classify_engine_error`) rather than MLX's
//     unmappable `exit(-1)` (sc-12178/12179). So this gate is not preventing a process kill — it is
//     preventing a raw allocator error that arrives only AFTER the full 64-step denoise has burned
//     minutes of GPU time, and that says nothing about the one lever that actually fixes it. At the
//     shipped 5 s / 151-frame default Mochi needs ~81 GB; an RTX 5090 has 32 GB, so on consumer
//     hardware this is not an edge case but the DEFAULT path.

/// The pure candle Mochi admission decision: `Some(error)` when the predicted peak overflows the VRAM
/// budget, `None` to admit. Missing either signal (unmeasurable weights / no budget) admits — the gate
/// never blocks without evidence, exactly like [`fit_decision`].
///
/// Budgets against `free_gb` (what is allocatable now), like [`fit_decision`] and unlike the MLX gate,
/// which has only a unified `total_gb`. [`HEADROOM_GB`] comes from
/// [`crate::fit_gate::dedicated_vram_reserve`]: it covers allocator slack and CUDA context
/// overhead because the OS does not draw from dedicated VRAM. The MLX gate instead uses
/// [`crate::fit_gate::legacy_unified_reserve`] for its unified-memory reserve.
///
/// Pure (no GPU probe, no env) so the whole decision is unit-testable without CUDA; the caller resolves
/// the budget.
pub(crate) fn mochi_fit_error(
    model_label: &str,
    weight_bytes: u64,
    frames: u32,
    width: u32,
    height: u32,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    let (needed_gb, budget) = (
        crate::fit_gate::mochi_needed_gb(weight_bytes, frames, width, height, HEADROOM_GB)?,
        budget?,
    );
    crate::fit_gate::exceeds_budget(needed_gb, budget.free_gb).then(|| {
        mochi_too_big_error(
            model_label,
            needed_gb,
            budget.free_gb,
            frames,
            width,
            height,
            weight_bytes as f64 / BYTES_PER_GIB,
            gpu_id,
        )
    })
}

/// Build Mochi's actionable over-budget rejection for the candle lane. Follows this module's existing
/// reject convention — name the model, state what it needs and what the GPU has — and adds the lever
/// that is UNIQUE to Mochi: the clip length.
///
/// The generic candle message's advice ("choose a smaller quant tier, lower the resolution") is nearly
/// useless here: Mochi has one trained bucket (848×480), and the decode dwarfs the tier delta (q4→bf16
/// is ~11 GiB against a ~60 GiB decode), so neither lever closes a 49 GB gap. The message therefore
/// leads with shortening the clip — the only lever that moves the dominant term.
///
/// Deliberately does NOT reuse `mlx_fit_gate::mochi_too_big_error`: that one says "unified memory" and
/// "run on a Mac with more memory", both false on CUDA. The shared thing between the lanes is the
/// arithmetic, not the prose.
#[allow(clippy::too_many_arguments)]
pub(super) fn mochi_too_big_error(
    model_label: &str,
    needed_gb: f64,
    available_gb: f64,
    frames: u32,
    width: u32,
    height: u32,
    weights_gb: f64,
    gpu_id: &str,
) -> WorkerError {
    WorkerError::InvalidPayload(format!(
        "{model_label} needs ~{needed} GB of VRAM to render a {frames}-frame {width}x{height} clip \
         (~{weights} GB of weights, held resident for the whole run, plus an untiled VAE decode whose \
         peak grows with clip length) but GPU {gpu_id} has ~{available} GB available. Shorten the \
         clip — the decode peak scales roughly linearly with duration — or run on a card with more \
         VRAM.",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
        weights = weights_gb.round() as i64,
    ))
}

// ---------------------------------------------------------------------------------------------
// SVD candle-video long-burst gate (sc-14492)
// ---------------------------------------------------------------------------------------------

/// Pure SVD CUDA admission decision for the candle lane.
///
/// A 32 GB RTX PRO 4500 completed the supported/default 25-frame, decode-chunk-8, 25-step 1024x576
/// render after sc-14625's sequential CFG/residency and live-free VAE tiling changes. That complete
/// measured profile (or lower settings) is admitted on 32 GB; requests beyond any measured dimension
/// remain rejected on that card class. Missing telemetry admits, matching every other fit gate: no
/// invented signal means no rejection.
///
/// The check keys off physical `total_gb`, not momentary `free_gb`. This story establishes a hardware
/// boundary, not a trustworthy transient-working-set threshold for a busy card.
pub(crate) fn svd_fit_error(
    frames: u32,
    decode_chunk_size: u32,
    steps: u32,
    width: u32,
    height: u32,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    let budget = budget?;
    if !crate::fit_gate::svd_profile_needs_larger_card(
        frames,
        decode_chunk_size,
        steps,
        width,
        height,
        budget.total_gb,
    ) {
        return None;
    }
    Some(WorkerError::InvalidPayload(format!(
        "Stable Video Diffusion's {frames}-frame, decode-chunk-{decode_chunk_size}, {steps}-step \
         {width}x{height} CUDA profile is outside the measured admission envelope on \
         GPU {gpu_id}'s {total} GB VRAM class. A real 32 GB RTX PRO 4500 completed up to \
         {validated_width}x{validated_height}, {validated_frames} frames, \
         decodeChunkSize={validated_chunk}, and {validated_steps} steps at a 17.521 GiB peak, so \
         even that bounded profile requires at least an {minimum} GB physical-card class. Requests \
         beyond any measured dimension remain unvalidated through 32 GB. Use a card and recipe \
         inside that measured envelope, or choose a larger GPU. The job was rejected before model \
         load.",
        total = budget.total_gb.round() as i64,
        validated_width = crate::fit_gate::SVD_32GB_VALIDATED_MAX_WIDTH,
        validated_height = crate::fit_gate::SVD_32GB_VALIDATED_MAX_HEIGHT,
        validated_frames = crate::fit_gate::SVD_32GB_VALIDATED_MAX_FRAMES,
        validated_chunk = crate::fit_gate::SVD_32GB_VALIDATED_MAX_DECODE_CHUNK,
        validated_steps = crate::fit_gate::SVD_32GB_VALIDATED_MAX_STEPS,
        minimum = crate::fit_gate::SVD_VALIDATED_PROFILE_MIN_VRAM_GB.round() as i64,
    )))
}

// ---------------------------------------------------------------------------------------------
// The Wan candle video weights-floor gate (epic 1788 / sc-12344)
// ---------------------------------------------------------------------------------------------
//
// Mochi above gets a bespoke gate because its decode peak is frame-dependent. Every OTHER candle video
// engine was admitted with NO pre-flight check at all until this: `predicted_peak_gb` is manifest-driven
// off `candle.vramGbByTier`, and **no candle video model carries a `candle` block**, so wiring the
// generic gate would return `None` ⇒ `Unknown` ⇒ admit for every one of them — dead code that reads as
// coverage (the failure `tests/gpu_and_manifest.rs`'s flux2 test warns about).
//
// So this budgets on the one signal that exists WITHOUT a measurement campaign: the on-disk weight
// bytes of the components the loader actually reads. Three facts make that a sound admission check for
// **Wan** rather than a guess:
//
//  1. **The engines this FLOOR still gates hold all components resident for the whole run.** ltx / svd /
//     mochi declare `supports_sequential_offload: false` (candle-gen-ltx / -svd / -mochi), so
//     `resolve_offload` is a no-op for them and Σweights is a genuine LOWER BOUND. **The Wan A14B T2V/I2V
//     and the dense TI2V-5B are the exception as of sc-12631 / epic sc-12732 (5B: sc-13175):** the candle
//     engine stages components **one-resident-at-a-time** (`render_sequential`,
//     `supports_sequential_offload: true`) — the A14B swaps TE → high-expert → low-expert → VAE (only ONE
//     14B expert live at a time, not both), the 5B flushes its TE + z48 VAE off-GPU around the dense
//     denoise (sc-12757) — so each is sized by its MEASURED sequential `candle.vramGbByTier` peak and this
//     floor never gates it (`wan_video_fit_error` takes the measured branch), while its load is forced
//     `OffloadPolicy::Sequential` (`video_jobs::candle_wan_offload_policy`) so the sized peak is the one
//     actually loaded. The floor stays the fallback for a Wan tier that was never measured.
//  2. **There is no host paging on CUDA.** Weights that do not fit VRAM cannot be demand-paged the way
//     MLX's unified pool swaps, so Σweights is a genuine LOWER BOUND on the job's need. This is the
//     asymmetry that makes the weights floor safe HERE but not on MLX's legacy transition override
//     exists to stop a pageable transient from wall-rejecting a small Mac (sc-12179), and that whole
//     class of false reject cannot arise on a discrete card.
//  3. Therefore rejecting when the weights alone overflow can never wall-reject a machine that would
//     have rendered — the sc-12179 regression this lane must not repeat.
//
// ⚠️ **(2) holds only where the on-disk bytes ARE the loaded set** — which is why [`wan_weight_bytes`]
// sums NAMED component subdirs and why `ltx`/`svd` are exempt. See that function's note; getting this
// wrong turns the floor from a lower bound into an over-count, i.e. straight into the sc-12179
// wall-reject.
//
// ⚠️ **The manifest's `footprint.peakMemoryBytes` for these models is an MLX measurement and MUST NOT be
// reused here.** `wan_2_2_t2v_14b`'s recorded 24.5 GiB peak sits BELOW its own `diskSizeBytes` because,
// as that manifest comment says, "A14B is MoE — only ONE 14B expert is resident at a time, not both".
// The candle A14B now ALSO holds one expert at a time (sc-12733, per (1)), but that number is still an
// MLX measurement at a different geometry/dtype and must not be reused: the candle lane is sized by its
// own MEASURED sequential `candle.vramGbByTier` (sc-12631). (Before sc-12732 the candle engine co-located
// both experts, which is why the pre-rework floor under-predicted it by a whole ~28 GiB bf16 expert.)
//
// ## This is a FLOOR — it under-predicts, deliberately — and it is now the FALLBACK, not the gate
// It counts weights only: no activation transient, no VAE decode, no attention working set. A MARGINAL
// job is still admitted and can still CUDA-OOM (catchably, via `classify_engine_error` — not MLX's
// unmappable `exit(-1)`). That was the honest bound from the data that existed before a measurement.
// [`wan_video_fit_error`] now prefers the MEASURED per-tier peak (`candle.vramGbByTier`, sc-12402) and
// only falls back here when a model/tier was never measured.
//
// ⚠️ **Point (1) above is WRONG for the DENSE tier, and sc-12402 measured how wrong.** "On-disk bytes ARE
// the loaded set" holds only for the PACKED q4/q8 tiers. `candle-gen-wan` casts dtypes on load
// (`wan14b.rs:49-52`: `DIT_DTYPE = BF16`, `ENC_DTYPE = F32`, applied per-tensor by `mmap_var_builder`),
// so the dense `Wan-AI/*-Diffusers` experts — which ship **fp32**, not bf16: `transformer/…index.json`
// declares `total_size` 57,153,966,336 = 53.2 GiB for ONE 14B expert — HALVE on load, while the UMT5 TE
// DOUBLES (bf16 10.58 → f32 21.16 GiB) on EVERY tier. Measured error vs the real device set:
//
//   * packed q4/q8 — floor UNDER-counts the co-resident set by the transient it omits. Safe direction:
//     it just admits, the pre-gate status quo. For the sequential-offload Wan engines the relationship
//     even INVERTS: the 5B's on-disk q4 floor is 18.15 GiB, but its MEASURED SEQUENTIAL peak is only
//     ~12.1 GB pool high-water / 10.61 GiB USED_MEM_HIGH (sc-13175 — the denoise transient, with the bf16
//     TE + z48 VAE flushed off-GPU), so there the floor OVER-counts and the measured `candle.vramGbByTier`
//     block must win (it does; see below).
//   * dense bf16 — floor OVER-counts by ~44 GiB (117.5 GiB summed vs a ~75 GiB device set), which
//     wall-rejects a 96 GB card that would render. That is the sc-12179 class this gate's note claims
//     cannot arise here; it can, on this one tier.
//
// So the floor is kept for its ORIGINAL job — refusing the clearly-impossible where nothing was measured
// — and a measurement supersedes it wherever one exists. The dense over-count is why
// [`wan_video_fit_error`] does NOT `max` the two.
//
// Deliberately NOT reusing Mochi's `mochi_decode_peak_gb`: that term is specific to Mochi's untiled
// AsymmVAE (peak linear in clip length). Wan tiles/chunks differently, so applying it here would
// over-predict badly and wall-reject working hardware — hence this gate is frame-blind by construction.

/// The exact component subdirs a candle **Wan** engine's loader reads, or `None` for an engine whose
/// on-disk bytes are NOT its loaded set (see below) ⇒ no signal ⇒ the gate admits.
///
/// Each list is the set of dirs the provider enumerates with `sorted_safetensors`, which loads **every**
/// `.safetensors` it finds — no variant selection — so the sum over these dirs IS the resident set:
///  * 5B (`candle-gen-wan` lib.rs:122/141/124) — `transformer` + `text_encoder` + `vae`;
///  * A14B T2V/I2V (wan14b.rs:288/301/302/328) — the same plus `transformer_2`, the second MoE expert,
///    which is co-resident (see the module note) and must be counted.
///
/// Packed q4/q8 tiers and the dense `Wan-AI/*-Diffusers` snapshot use the SAME subdir names (the tier is
/// detected from tensor content — the `proj_out.scales` marker — not from the directory layout), so one
/// list serves both.
///
/// **Why NAMED subdirs and not a blind recursive sum of `model_dir`.** A blind sum cannot go inert,
/// which is attractive — but it can OVER-count, and over-counting is the direction that hurts: it
/// wall-rejects hardware that renders fine (sc-12179), whereas an under-count merely admits (the
/// pre-gate status quo). Naming the dirs makes a wrong name read `0` for that component — permissive,
/// the safe direction — and it also makes the tier-ROOT fallback safe: if no tier subdir resolved and
/// `model_dir` is a root holding `q4/`+`q8/`, there is no top-level `transformer/`, so this reads 0 and
/// admits rather than summing two tiers at once.
///
/// **Why `ltx_2_3_distilled` and `svd_xt` are `None` (sc-12344's recorded exemptions).** Their on-disk
/// bytes are categorically not their loaded set, so ANY floor built from a directory sum would
/// wall-reject working cards:
///  * **ltx dense** — `ltx_checkpoint()` (candle-gen-ltx lib.rs:129-159) picks exactly ONE root-level
///    file by substring rank (`distilled` > `bf16` > largest), skipping the `fp8`/`mixed`/lora/upscaler
///    siblings shipped beside it. The hosted `Lightricks/LTX-2.3` is ~146 GiB on disk
///    (`estimatedSizeBytes: 157004895813`) against a single-file load — summing it would refuse LTX on
///    every GPU in existence.
///  * **ltx packed tier** — reads 3 exact files (`transformer` / `connector` / `vae_decoder`,
///    tier.rs:152-173) while the tier dir also ships `vae_encoder` + `audio_vae` + `vocoder` +
///    `upsampler`, which the T2V render never loads (tier.rs:30).
///  * **svd_xt** — resolves ONE exact file per component (candle-gen-svd lib.rs:128-141), but the
///    upstream `stabilityai/stable-video-diffusion-img2vid-xt` snapshot ships `X.safetensors` AND
///    `X.fp16.safetensors` side by side in each of `unet`/`vae`/`image_encoder`, so a dir sum roughly
///    DOUBLES a ~8.9 GiB model — enough to false-reject a small card that runs it today.
///
/// Closing those two honestly needs the provider to own its split (`register_generators! { …;
/// footprint = … }` — gen-core's `PerComponentBytes`, whose own doc explains that a consumer guessing
/// component sizes is exactly this failure). None of the video crates registers one today, so that is a
/// cross-repo change on the inference monorepo: **sc-12397**. Not a fudge factor — the LTX dense case is
/// off by ~7x, not by a constant.
pub(super) fn wan_weight_components(engine_id: &str) -> Option<&'static [&'static str]> {
    match engine_id {
        "wan2_2_ti2v_5b" => Some(&["transformer", "text_encoder", "vae"]),
        "wan2_2_t2v_14b" | "wan2_2_i2v_14b" => {
            Some(&["transformer", "transformer_2", "text_encoder", "vae"])
        }
        // `mochi_1` has its own frame-dependent gate above; `ltx_2_3_distilled` / `svd_xt` are exempt.
        _ => None,
    }
}

/// The on-disk bytes a candle Wan load holds resident: [`wan_weight_components`] summed under
/// `model_dir`. `0` for a non-Wan engine, or when nothing could be scanned ⇒ no signal ⇒ admit.
///
/// Reuses [`crate::mlx_fit_gate::sum_safetensors_bytes`] per component so the HF-cache symlink handling
/// (shards are symlinks into `blobs/`) and the AppleDouble `._*` skip are shared with the MLX lane
/// rather than re-implemented — the same reason sc-12306's Mochi gate reuses `mochi_resident_bytes`.
pub(crate) fn wan_weight_bytes(engine_id: &str, model_dir: &Path) -> u64 {
    wan_weight_components(engine_id).map_or(0, |components| {
        components
            .iter()
            .map(|component| crate::mlx_fit_gate::sum_safetensors_bytes(&model_dir.join(component)))
            .sum()
    })
}

/// Exact resident-weight floor for the dedicated sequential Wan2.2 VACE-Fun provider. The shared
/// UMT5+VAE preparation phase is co-resident, then each complete expert is loaded one at a time. The
/// peak is therefore `max(shared, high + adapters, low + adapters)`, never the sum of both experts.
/// Missing any required component fails closed because this route has no calibrated CUDA peak yet.
pub(crate) fn wan_vace_fun_sequential_weight_bytes(
    model_dir: &Path,
    adapter_stage_bytes: u64,
) -> Result<u64, WorkerError> {
    let component = |name: &str| crate::mlx_fit_gate::sum_safetensors_bytes(&model_dir.join(name));
    let high = component("transformer");
    let low = component("transformer_2");
    let text_encoder = component("text_encoder");
    let vae = component("vae");
    if [high, low, text_encoder, vae].contains(&0) {
        return Err(WorkerError::InvalidPayload(
            "Wan2.2 VACE-Fun admission cannot verify transformer, transformer_2, text_encoder, and vae weights; repair the model install before retrying."
                .to_owned(),
        ));
    }
    let shared = text_encoder.checked_add(vae).ok_or_else(|| {
        WorkerError::InvalidPayload("Wan2.2 VACE-Fun shared footprint overflowed u64.".to_owned())
    })?;
    let high = high.checked_add(adapter_stage_bytes).ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Wan2.2 VACE-Fun high-expert footprint overflowed u64.".to_owned(),
        )
    })?;
    let low = low.checked_add(adapter_stage_bytes).ok_or_else(|| {
        WorkerError::InvalidPayload(
            "Wan2.2 VACE-Fun low-expert footprint overflowed u64.".to_owned(),
        )
    })?;
    Ok(shared.max(high).max(low))
}

/// The pure Wan candle video admission decision: `Some(error)` when the model's RESIDENT WEIGHTS alone
/// cannot fit the VRAM budget, `None` to admit. The non-Mochi twin of [`mochi_fit_error`], and pure for
/// the same reason — the caller resolves the budget, so the whole decision is unit-testable with no CUDA
/// driver and no GPU.
///
/// Missing either signal admits: no budget (`nvidia-smi` unreadable) or unmeasurable weights
/// (`weight_bytes == 0` — an exempt engine, or a dir that could not be scanned) ⇒ `None`, exactly like
/// [`fit_decision`]'s [`FitDecision::Unknown`]. A gate that blocks without evidence is a regression, not
/// a safety net.
pub(crate) fn video_weights_fit_error(
    model_label: &str,
    weight_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    let (needed_gb, budget) = (video_weights_needed_gb(weight_bytes)?, budget?);
    crate::fit_gate::exceeds_budget(needed_gb, budget.free_gb).then(|| {
        video_weights_too_big_error(
            model_label,
            needed_gb,
            budget.free_gb,
            weight_bytes as f64 / BYTES_PER_GIB,
            gpu_id,
        )
    })
}

/// The candle **video** admission decision (sc-12402): budget on the MEASURED per-tier peak
/// (`candle.vramGbByTier[tier_key]` + [`HEADROOM_GB`], via [`predicted_peak_gb`]) when the manifest
/// carries one, else fall back to the sc-12344 on-disk weights FLOOR ([`video_weights_fit_error`]).
///
/// This is the seam that makes the measured blocks LIVE. Until sc-12402 the video lane called the
/// floor directly and nothing read `candle.vramGbByTier` — so adding the blocks alone would have been
/// dead data that reads as coverage (the failure `tests/gpu_and_manifest.rs`'s flux2 test exists to
/// catch). Pure, like both halves it composes: the caller resolves the budget and the tier.
///
/// ## Why a measurement REPLACES the floor rather than composing with it (`max`)
///
/// A `max(measured, floor)` looks safer and is not: **the floor OVER-counts the dense tier**, so
/// maxing would re-introduce the exact wall-reject the measurement exists to remove.
///
/// The floor's premise — `wan_weight_components`' "on-disk bytes ARE the loaded set" — holds only for
/// the PACKED q4/q8 tiers. The dense `Wan-AI/*-Diffusers` snapshot the manifest routes `bf16` to ships
/// its experts in **fp32** (`transformer/…index.json` declares `total_size` 57,153,966,336 = 53.2 GiB
/// for ONE 14B expert), and `candle-gen-wan` casts on load — `DIT_DTYPE = BF16`, `ENC_DTYPE = F32`
/// (wan14b.rs:49-52), applied per-tensor by `mmap_var_builder`. So the experts HALVE (57.2 → 28.6 GB
/// each) while the UMT5 TE DOUBLES (11.4 → 22.7 GB). Net, the dense floor sums ~117 GiB against a real
/// ~81 GB resident set and refuses a 96 GB card that renders it. (The packed tiers fail the other way
/// — the f32 TE doubling makes their floor UNDER-count — which is safe: a floor that under-counts just
/// admits, the pre-gate status quo.)
///
/// A real measurement is not a better proxy for the peak, it IS the peak; deferring to a byte-sum that
/// is provably wrong about dtype would be superstition. Where no measurement exists the floor is still
/// the best available signal, so it stays as the fallback.
///
/// Missing either signal admits, exactly like [`fit_decision`]'s [`FitDecision::Unknown`]: no budget
/// (`nvidia-smi` unreadable) ⇒ `None`, and an engine with neither a `candle` block nor countable
/// weights (`ltx`/`svd`) ⇒ `None` through the floor.
#[cfg(any(test, doc))]
pub(crate) fn wan_video_fit_error(
    model_label: &str,
    manifest_entry: &JsonObject,
    tier_key: &str,
    weight_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    wan_video_fit_error_with_adapter_bytes(
        model_label,
        manifest_entry,
        tier_key,
        weight_bytes,
        0,
        gpu_id,
        budget,
    )
}

/// Adapter-aware Wan admission. Packed callers pass the independently resident user stack; dense
/// callers pass zero because their factors are folded. The calibrated Lightning stack is already in
/// the manifest peak and must not be included here.
pub(crate) fn wan_video_fit_error_with_adapter_bytes(
    model_label: &str,
    manifest_entry: &JsonObject,
    tier_key: &str,
    weight_bytes: u64,
    adapter_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    // Unmeasured (no `candle` block, or no row for this tier and no `minMemoryGb`) ⇒ the sc-12344
    // floor, byte-for-byte the shipped behavior.
    let Some(needed_gb) = predicted_peak_gb(manifest_entry, tier_key) else {
        return video_weights_fit_error(
            model_label,
            weight_bytes.saturating_add(adapter_bytes),
            gpu_id,
            budget,
        );
    };
    let needed_gb = needed_gb + adapter_bytes as f64 / BYTES_PER_GIB;
    let budget = budget?;
    crate::fit_gate::exceeds_budget(needed_gb, budget.free_gb)
        .then(|| video_peak_too_big_error(model_label, tier_key, needed_gb, budget.free_gb, gpu_id))
}

/// Archived MiniMax-H3 catalog-row regression fixture. Keep the measured baseline and its
/// adapter arithmetic testable; production uses resolved source/request/strategy admission.
#[cfg(test)]
pub(crate) fn minimax_h3_fit_error(
    manifest_entry: &JsonObject,
    tier_key: &str,
    adapter_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    minimax_h3_fit_error_with_sparse(
        manifest_entry,
        tier_key,
        adapter_bytes,
        gpu_id,
        budget,
        None,
    )
}

// Historical catalog-row fixture only. Production H3 admission uses the resolved source,
// request, live pools and selected strategy in video_admission before tensor materialization.
#[cfg(test)]
pub(crate) fn minimax_h3_fit_error_with_sparse(
    manifest_entry: &JsonObject,
    tier_key: &str,
    adapter_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
    sparse: Option<&gen_core_h3::veda::workspace::WorkspaceBound>,
) -> Option<WorkerError> {
    const MEASURED_PIXELS: u64 = 1_032_192;
    let candle = manifest_entry.get("candle");
    let measured = candle
        .and_then(|value| value.get("measured"))
        .and_then(Value::as_bool);
    let measured_pixels = candle
        .and_then(|value| value.get("vramMeasuredPixels"))
        .and_then(Value::as_u64);
    let peak_gb = candle
        .and_then(|value| value.get("vramGbByTier"))
        .and_then(|tiers| tiers.get(tier_key))
        .and_then(json_f64)
        .filter(|value| value.is_finite() && *value > 0.0);
    let (true, Some(MEASURED_PIXELS), Some(peak_gb)) =
        (measured == Some(true), measured_pixels, peak_gb)
    else {
        return Some(WorkerError::InvalidPayload(format!(
            "MiniMax-H3 Candle admission is unavailable because the installed catalog has no \
             complete measured {tier_key} CUDA row at 1344x768. Refusing to load unproved \
             weights; update SceneWorks before retrying."
        )));
    };
    let Some(budget) = budget else {
        return Some(WorkerError::InvalidPayload(
            "MiniMax-H3 Candle admission could not read free GPU VRAM from nvidia-smi. Refusing \
             to load the measured tier without proving that it fits; verify the NVIDIA \
             driver/runtime and retry."
                .to_owned(),
        ));
    };
    // This catalog row remains the dense baseline; sparse has its own explicit additional
    // requirement including predictor loading, retained state and forced dense fallback.
    let needed_gb = crate::video_admission::veda_cold_peak_gb(
        peak_gb + HEADROOM_GB + adapter_bytes as f64 / BYTES_PER_GIB,
        sparse,
    );
    let sparse_reason = if sparse.is_some() {
        " plus the Veda predictor/workspace and safe dense fallback"
    } else {
        ""
    };
    crate::fit_gate::exceeds_budget(needed_gb, budget.free_gb).then(|| {
        WorkerError::InvalidPayload(format!(
            "MiniMax-H3 needs ~{needed} GB of free GPU VRAM for its measured {tier_key} render \
             peak plus CUDA reserve{sparse_reason}, but GPU {gpu_id} has ~{available} GB available. Select a \
             smaller quant tier or use a GPU with more free VRAM.",
            needed = needed_gb.ceil() as u64,
            available = budget.free_gb.round() as u64,
        ))
    })
}

/// Fail-closed admission for the shared SCAIL-2 Candle bf16 package. Unlike the generic Wan gate,
/// SCAIL has no lower-memory Candle tier and no sequential/offload lifecycle, so falling back to
/// on-disk bytes or admitting an absent row would turn an incomplete catalog into a real OOM. The
/// exact production render owns `candle.vramGbByTier.bf16`; [`HEADROOM_GB`] remains the common CUDA
/// reserve added by every measured-peak gate.
#[cfg(test)]
pub(crate) fn scail2_video_fit_error(
    manifest_entry: &JsonObject,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    scail2_video_fit_error_with_adapter_bytes(manifest_entry, "bf16", 0, gpu_id, budget)
}

pub(crate) fn scail2_video_fit_error_with_adapter_bytes(
    manifest_entry: &JsonObject,
    tier_key: &str,
    adapter_bytes: u64,
    gpu_id: &str,
    budget: Option<VramBudget>,
) -> Option<WorkerError> {
    const MEASURED_PIXELS: u64 = 832 * 480;
    let candle = manifest_entry.get("candle");
    let measured = candle
        .and_then(|value| value.get("measured"))
        .and_then(Value::as_bool);
    let peak_gb = candle
        .and_then(|value| value.get("vramGbByTier"))
        .and_then(|tiers| tiers.get(tier_key))
        .and_then(json_f64)
        .filter(|value| value.is_finite() && *value > 0.0);
    let min_memory_gb = candle
        .and_then(|value| value.get("minMemoryGb"))
        .and_then(Value::as_u64);
    let measured_pixels = candle
        .and_then(|value| value.get("vramMeasuredPixels"))
        .and_then(Value::as_u64);
    let (true, Some(peak_gb), Some(min_memory_gb), Some(MEASURED_PIXELS)) = (
        measured == Some(true),
        peak_gb,
        min_memory_gb,
        measured_pixels,
    ) else {
        return Some(WorkerError::InvalidPayload(
            format!(
                "SCAIL-2 Candle admission is unavailable because the installed catalog has no complete \
                 measured {tier_key} CUDA row (positive peak, 832x480 geometry, and minMemoryGb floor). \
                 Refusing to load the selected {tier_key} package; update SceneWorks before retrying."
            ),
        ));
    };
    let base_needed_gb = peak_gb + HEADROOM_GB;
    if (min_memory_gb as f64) + f64::EPSILON < base_needed_gb.ceil() {
        return Some(WorkerError::InvalidPayload(format!(
            "SCAIL-2 Candle admission is unavailable because catalog minMemoryGb={min_memory_gb} \
             is below the measured {tier_key} CUDA peak plus reserve (~{} GB). Refusing to load the \
             selected {tier_key} package; update SceneWorks before retrying.",
            base_needed_gb.ceil() as u64,
        )));
    }
    let needed_gb = base_needed_gb + adapter_bytes as f64 / BYTES_PER_GIB;
    let Some(budget) = budget else {
        return Some(WorkerError::InvalidPayload(
            format!(
                "SCAIL-2 Candle admission could not read free GPU VRAM from nvidia-smi. Refusing to \
                 load the selected {tier_key} package without proving that its measured render peak \
                 fits; verify the NVIDIA driver/runtime and retry."
            ),
        ));
    };
    crate::fit_gate::exceeds_budget(needed_gb, budget.free_gb).then(|| {
        WorkerError::InvalidPayload(format!(
            "SCAIL-2 shared {tier_key} needs ~{needed} GB of free GPU VRAM (the measured 832x480, \
             81-frame render peak plus CUDA reserve), but GPU {gpu_id} has ~{available} GB \
             available. Select a measured SCAIL-2 tier that fits, or use a GPU with more free VRAM.",
            needed = needed_gb.ceil() as i64,
            available = budget.free_gb.round() as i64,
        ))
    })
}

/// Build the MEASURED-peak candle video rejection (sc-12402). The measured sibling of
/// [`video_weights_too_big_error`], and worded apart from it on purpose: that one can only honestly
/// speak about weights, this one is the whole render's ceiling (weights + denoise + the budget-tiled
/// VAE decode), so it names the render rather than the weights.
///
/// Names the TIER, because the tier is the lever that moves this number and the gate now knows which
/// one it sized. Like the floor's message it does NOT offer "lower the resolution": the measured peak
/// is a per-tier CONSTANT taken at the model's default geometry (the manifest schema's "video =
/// default frames"), and Wan's decode is budget-tiled (`auto_tiling_budgeted_wan22`) so the weights
/// dominate and a smaller clip cannot move the number the user was just shown. That is the same
/// resolution-blind contract the image lane's `vramGbByTier` gate has always had, and the reason Mochi
/// needs its own frame-scaled gate instead of this one (its untiled AsymmVAE decode IS the peak).
pub(super) fn video_peak_too_big_error(
    model_label: &str,
    tier_key: &str,
    needed_gb: f64,
    available_gb: f64,
    gpu_id: &str,
) -> WorkerError {
    WorkerError::InvalidPayload(format!(
        "{model_label} needs ~{needed} GB of VRAM to render at its {tier_key} tier (the measured peak of \
         the whole render — the resident weights plus the denoise and VAE-decode transient, at whatever \
         component residency the engine uses) but GPU {gpu_id} has ~{available} GB available. \
         Select a smaller quant tier, or run on a GPU with more VRAM.",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
    ))
}

/// The predicted resident FLOOR (GiB) for a candle video job: the on-disk weights every component of
/// this engine holds for the whole run, plus [`HEADROOM_GB`].
///
/// [`HEADROOM_GB`] is the CUDA reserve — allocator slack + CUDA context overhead — not MLX's
/// unified reserve (the OS does not draw from discrete VRAM). The two may agree numerically but mean
/// different things; this is the same split `fit_gate::mochi_needed_gb` takes its `reserve_gb`
/// parameter for (sc-12306).
///
/// `None` when nothing was measured (`weight_bytes == 0`) ⇒ no signal ⇒ never block.
pub(super) fn video_weights_needed_gb(weight_bytes: u64) -> Option<f64> {
    (weight_bytes > 0).then(|| weight_bytes as f64 / BYTES_PER_GIB + HEADROOM_GB)
}

/// Build the generic candle video weights-floor rejection. Names the model, what its weights alone
/// need, and what the card has.
///
/// The levers are WEIGHTS levers, and only those. It deliberately does NOT say "lower the output
/// resolution": resolution has exactly ZERO effect on weight bytes, so offering it here would send
/// the user to a knob that cannot move the number they were just shown. Nor does it say Mochi's
/// "shorten the clip" — same reason (there is no decode term in this floor). The tier IS the lever
/// that moves weights, plus the card itself. The image lane's `vram_reject_tail` used to offer the
/// knob unconditionally against its equally resolution-blind `vramGbByTier` rows; issue #3197 is
/// what that cost, and it now takes an explicit `ResolutionLever` for the same reason stated here.
pub(super) fn video_weights_too_big_error(
    model_label: &str,
    needed_gb: f64,
    available_gb: f64,
    weights_gb: f64,
    gpu_id: &str,
) -> WorkerError {
    WorkerError::InvalidPayload(format!(
        "{model_label} needs at least ~{needed} GB of VRAM just to hold its weights (~{weights} GB, \
         every component held resident for the whole run — this engine does not stage components) but \
         GPU {gpu_id} has ~{available} GB available. Select a smaller quant tier, or run on a GPU with \
         more VRAM.",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
        weights = weights_gb.round() as i64,
    ))
}
