#[cfg(not(target_os = "macos"))]
use super::has_candle_tier_peak_row;
#[cfg(not(target_os = "macos"))]
use super::is_dense_te_tier;
#[cfg(not(target_os = "macos"))]
use super::load_spec::apply_declared_candle_image_load_shape;
#[cfg(not(target_os = "macos"))]
use super::load_spec::candle_certified_load_spec;
#[cfg(not(target_os = "macos"))]
use super::nvfp4_host_eligible;
#[cfg(not(target_os = "macos"))]
use super::nvfp4_selected;
#[cfg(not(target_os = "macos"))]
use super::quantization::gate_tier_key;
#[cfg(not(target_os = "macos"))]
use super::quantization::resolve_quant;
use super::quantization::resolve_tier_dir;
#[cfg(not(target_os = "macos"))]
use super::render::hires_fix_reference_count;
#[cfg(not(target_os = "macos"))]
use super::render::lane_reference_count;
use super::tier_quality_rank;
use super::ImageRequest;
#[cfg(not(target_os = "macos"))]
use super::LoadSpec;
#[cfg(not(target_os = "macos"))]
use super::Quant;
#[cfg(not(target_os = "macos"))]
use super::WeightsSource;
use crate::JsonObject;
#[cfg(not(target_os = "macos"))]
use crate::Path;
use crate::Settings;
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
use crate::Value;
#[cfg(not(target_os = "macos"))]
use crate::WorkerError;
#[cfg(not(target_os = "macos"))]
use crate::WorkerResult;
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
use serde_json::json;
/// Identity of the Krea Turbo calibration this request was admitted under, built from the two
/// constants the selector itself compares against (sc-17097).
///
/// It used to be a literal in two places, so a re-measurement had to remember to edit both; the
/// tracing line and the run context would otherwise disagree about which capture was in force.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn krea_evidence_revision() -> String {
    format!(
        "{}@{}",
        crate::vram_gate::KREA_TURBO_SCENEWORKS_REVISION,
        // sc-17774: the lane's compile-closure digest replaced the frozen inference SHA here too, so
        // the traced receipt names the same thing the selector actually compared.
        sceneworks_core::memory_calibration::packaged_closure_digest("candle", "krea_2_turbo")
            .unwrap_or_default()
    )
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn optimized_shared_memory_context(
    engine_id: &str,
    context: gen_core::MemoryRunContext,
) -> Option<gen_core::MemoryRunContext> {
    crate::memory_strategy::executable_context(engine_id, context)
}

/// The registered FLUX.2-dev generator is a text-to-image provider even when the catalog surface
/// originated from a style-variation prompt. Keep that provider-specific canonicalization here so
/// Z-Image, Qwen, and FLUX.1 retain their established mode/evidence semantics.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn candle_base_memory_request_mode<'a>(
    engine_id: &str,
    request_mode: &'a str,
) -> &'a str {
    if engine_id == "flux2_dev"
        || (engine_id == "flux2_klein_9b"
            && matches!(request_mode, "image_generation" | "text_to_image"))
        || (matches!(engine_id, "chroma1_hd" | "chroma1_base" | "chroma1_flash")
            && matches!(request_mode, "image_generation" | "text_to_image"))
        || (matches!(engine_id, "ideogram_4" | "ideogram_4_turbo")
            && matches!(request_mode, "image_generation" | "text_to_image"))
        || (matches!(engine_id, "sana_1600m" | "sana_sprint_1600m")
            && matches!(request_mode, "image_generation" | "text_to_image"))
        || (matches!(
            engine_id,
            "sd3_5_large" | "sd3_5_large_turbo" | "sd3_5_medium"
        ) && matches!(request_mode, "image_generation" | "text_to_image"))
    {
        "text_to_image"
    } else {
        request_mode
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn shared_image_reference_count(
    edit_reference_count: usize,
    has_single_reference: bool,
    has_edit_mask: bool,
    has_hires_fix: bool,
) -> u32 {
    if has_hires_fix {
        hires_fix_reference_count()
    } else {
        lane_reference_count(has_single_reference, edit_reference_count, has_edit_mask)
    }
}

#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn flux2_klein_reference_bearing_mode(mode: &str) -> bool {
    matches!(
        mode,
        "edit_image" | "reference" | "image_to_image" | "character_image" | "style_variations"
    )
}

/// The generation tiers (`bf16`/`q8`/`q4`) currently INSTALLED for `request`'s model, in DESCENDING
/// fidelity (bf16 → q8 → q4). Each is confirmed by [`resolve_tier_dir`] (a forced-bits probe of the
/// real family resolver), so the list is exactly what the loader could load. Empty for a model with no
/// quant-tier layout (a flat / `modelPath` snapshot).
///
/// Candle-only: the sc-12090 reject-message enumeration. The MLX downtier reject names the single
/// smallest evaluated tier from `choose_downtier`, so it needs no separate installed-list scan.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn installed_tier_keys(
    request: &ImageRequest,
    settings: &Settings,
) -> Vec<&'static str> {
    ["bf16", "q8", "q4"]
        .into_iter()
        .filter(|tier| resolve_tier_dir(request, settings, tier).is_some())
        .collect()
}

/// A per-tier capability-fit result for the downtier chooser (sc-10733) — the lane-agnostic reduction
/// of each lane's richer fit decision (candle's resident/offload/reject, MLX's resident/sequential/
/// reject) to "does this tier run at all on this machine."
#[cfg(any(all(target_os = "macos", test), feature = "backend-candle"))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum TierFit {
    /// Runs — resident or (where the provider stages components) sequentially.
    Fits,
    /// Won't run even sequentially. `needed_gb`/`available_gb` are for the reject message.
    TooBig { needed_gb: f64, available_gb: f64 },
}

/// The capability-downtier decision (sc-10733).
#[cfg(any(all(target_os = "macos", test), feature = "backend-candle"))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum DowntierPick {
    /// The resolved default tier fits — load it unchanged.
    Keep,
    /// A LOWER installed tier is the highest-fidelity one that fits — load it instead.
    Downtier(&'static str),
    /// Nothing in `[floor, default]` fits — reject, naming the SMALLEST (least-demanding) tier
    /// evaluated and what it needed.
    Reject {
        tier: &'static str,
        needed_gb: f64,
        available_gb: f64,
    },
}

/// The pure capability-downtier chooser (sc-10733), shared by the candle and MLX gates. `candidates`
/// are the INSTALLED tiers in `[floor, default]` with their per-lane [`TierFit`], in DESCENDING fidelity
/// (so the default tier is first). Returns the highest-fidelity tier that fits: [`DowntierPick::Keep`]
/// when that is the default itself, [`DowntierPick::Downtier`] when a lower tier is the best that fits,
/// or [`DowntierPick::Reject`] (naming the smallest — least-demanding — tier evaluated) when nothing in
/// range fits.
///
/// The floor + installed clamping live in the CANDIDATE list (the caller filters to `rank >= floor` and
/// `installed`), so the quality floor always wins over the downtier — a floor-q8 model's candidates
/// never include q4, so it rejects rather than silently rendering q4 (acceptance #5). An explicit user
/// pick never reaches here (the caller skips the downtier for it, honoring the pick — acceptance #7).
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
// Scoring a candle tier is a manifest arithmetic lookup, so that lane keeps the eager form and its
// straight-line read. On macOS the only callers left are the ordering tests — the MLX gate scores
// lazily now, because there scoring a tier hashes an encoder.
pub(super) fn choose_downtier(
    default_tier: &str,
    candidates: &[(&'static str, TierFit)],
) -> DowntierPick {
    let tiers: Vec<&'static str> = candidates.iter().map(|(tier, _)| *tier).collect();
    let mut fits = candidates.iter().map(|(_, fit)| *fit);
    match choose_downtier_lazy::<_, std::convert::Infallible>(default_tier, &tiers, |_| {
        Ok(fits.next())
    }) {
        Ok(pick) => pick,
        Err(never) => match never {},
    }
}

/// [`choose_downtier`] with the fit scored ON DEMAND, stopping at the first tier that fits.
///
/// Same ordering rule, same result — this is the single implementation and the eager form above
/// delegates to it, so the two cannot drift. What changes is how many tiers get SCORED.
///
/// That matters because on the MLX lane scoring a tier is not cheap. `mlx_tier_fit` reaches
/// `ProviderRegistry::footprint`, which resolves the encoder through `EncoderContract::source_for_load`
/// and seals it with `ArtifactSeal::read_unchanged` — a full SHA-256 of that tier's text encoder. A
/// stack sample of a krea_2_turbo job caught 62% of a 30-second window inside that subtree, 96% of it
/// in the SHA-256 compressor itself and only 3% in `read`: it is CPU, not I/O.
///
/// The caller then collected EVERY candidate before choosing, so a machine with room to spare paid for
/// bf16, q8 and q4 in turn — three encoders hashed to answer a question the first one had already
/// settled. Measured on one job: 15s + 10s + 6s, against a budget with 61 GB unused.
///
/// `Ok(None)` from `fit` SKIPS a tier rather than scoring it, preserving the caller's pre-existing
/// filter for a candidate whose directory does not resolve: such a tier was absent from the eager
/// slice entirely, so it must not become the named rejection either.
#[cfg(any(all(target_os = "macos", test), feature = "backend-candle"))]
fn choose_downtier_lazy<F, E>(
    default_tier: &str,
    tiers: &[&'static str],
    mut fit: F,
) -> Result<DowntierPick, E>
where
    F: FnMut(&'static str) -> Result<Option<TierFit>, E>,
{
    let mut smallest_reject: Option<(&'static str, f64, f64)> = None;
    for &tier in tiers {
        let Some(fit) = fit(tier)? else {
            continue;
        };
        match fit {
            // DESCENDING fidelity ⇒ the first that fits is the highest-fidelity fitting tier, and
            // nothing below it can win — so nothing below it is scored.
            TierFit::Fits => {
                return Ok(if tier == default_tier {
                    DowntierPick::Keep
                } else {
                    DowntierPick::Downtier(tier)
                });
            }
            TierFit::TooBig {
                needed_gb,
                available_gb,
            } => smallest_reject = Some((tier, needed_gb, available_gb)),
        }
    }
    // Nothing fit — reject, naming the LAST (smallest / least-demanding) tier we tried. `None` only when
    // the candidate list was empty (no installed tier in range — defensive; the default itself is always
    // installed & in range), in which case Keep lets the plain gate handle it.
    Ok(match smallest_reject {
        Some((tier, needed_gb, available_gb)) => DowntierPick::Reject {
            tier,
            needed_gb,
            available_gb,
        },
        None => DowntierPick::Keep,
    })
}

/// The installed tiers in `[floor, default]` for `request`'s model, in DESCENDING fidelity, ready for
/// [`choose_downtier`] once each is paired with its per-lane [`TierFit`] (sc-10733). `default_tier` is
/// the disk-clamped resolved tier (the downtier ceiling — the clamp only ever lowers fidelity); `floor`
/// is the per-model quality floor ([`min_quality_floor`], defaulting to `q4`). Excludes tiers not on
/// disk so the downtier never lands on an uninstalled tier.
#[cfg(native_lane)]
pub(super) fn downtier_candidate_tiers(
    request: &ImageRequest,
    settings: &Settings,
    default_tier: &str,
    floor: Option<&str>,
) -> Vec<&'static str> {
    let default_rank = tier_quality_rank(default_tier);
    // Floor at least q4 (rank 1) — a downtier must never fall below the lowest real tier.
    let floor_rank = floor.map_or(1, tier_quality_rank).max(1);
    ["bf16", "q8", "q4"]
        .into_iter()
        .filter(|tier| {
            let rank = tier_quality_rank(tier);
            rank <= default_rank && rank >= floor_rank
        })
        .filter(|tier| resolve_tier_dir(request, settings, tier).is_some())
        .collect()
}

/// Whether the GB figure a candle reject just quoted can move with the output geometry — the
/// difference between naming a knob and sending the operator round a loop that cannot pass
/// (issue #3197).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResolutionLever {
    /// The number is a per-tier manifest ROW read VERBATIM — `candle.vramGbByTier[tier]` or
    /// `candle.sequentialPeakGb[tier]`, plus [`crate::vram_gate::HEADROOM_GB`], stated once at
    /// `candle.vramMeasuredPixels`. [`crate::vram_gate::predicted_peak_gb`] and
    /// [`crate::vram_gate::predicted_sequential_peak_gb`] take no width and no height, so the
    /// verdict is identical at every output size: a retry at a smaller one re-derives the same GB
    /// and rejects again. This is the resolution-blind contract `vram_gate`'s own
    /// `video_peak_too_big_error` already documents (and already declines to print the advice
    /// against); the image lane kept offering the knob anyway, which is how #3197's reporter spent
    /// a 1024x1024 and a 768x768 attempt on byte-identical errors.
    Blind,
    /// The number came from a geometry-aware selector, so a smaller output can genuinely change the
    /// verdict — including by bringing an out-of-envelope request back inside the measured hull the
    /// shared memory-strategy ladder can price at all. Keep offering it.
    Priced,
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
impl ResolutionLever {
    /// The resolution clause for a reject tail: the knob when it is real, otherwise the sentence
    /// that stops the retry. Saying it outright beats dropping the clause silently — the estimate
    /// reads as activation-shaped to anyone looking at it, so silence invites the very retry this
    /// explains away. The rule is "name a knob only when the gate priced it", never "don't mention
    /// resolution": the lane that CAN price a smaller geometry (Krea Turbo's calibrated ladder)
    /// resolves a concrete `krea_turbo_smaller_fit_with_runtime` target and offers that size.
    pub(super) fn for_selector(selector_priced: bool) -> Self {
        if selector_priced {
            Self::Priced
        } else {
            Self::Blind
        }
    }

    fn clause(self) -> &'static str {
        match self {
            Self::Blind => {
                " Lowering the output resolution will not help: this gate prices each tier from a \
                 fixed per-tier peak that does not scale with the output size."
            }
            Self::Priced => "",
        }
    }

    /// The lever list that closes a reject tail, in the tail's two historical shapes: `joined`
    /// follows a comma (a smaller tier was just offered), `leading` opens the clause on its own.
    fn levers(self) -> (&'static str, &'static str) {
        match self {
            Self::Blind => (
                "or run on a GPU with more VRAM.",
                "run on a GPU with more VRAM.",
            ),
            Self::Priced => (
                "lower the output resolution, or run on a GPU with more VRAM.",
                "lower the output resolution or run on a GPU with more VRAM.",
            ),
        }
    }
}

/// The actionable tail for a candle VRAM-fit reject (sc-12090). Suggests only the tiers that are
/// actually INSTALLED and smaller than the rejected one (`installed_smaller`, descending fidelity),
/// else states that none is installed. Never names the rejected tier itself, and never points at the
/// quant picker — which is hidden by design when ≤1 tier is installed, exactly the case that produced
/// the misleading "Pick a lower tier (Q4/Q8)" text on a single-tier install. `lever` decides whether
/// the resolution knob is offered or explained away ([`ResolutionLever`], issue #3197).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn vram_reject_tail(installed_smaller: &[&str], lever: ResolutionLever) -> String {
    let (joined, leading) = lever.levers();
    if installed_smaller.is_empty() {
        return format!(
            "No smaller tier is installed — {leading}{note}",
            note = lever.clause(),
        );
    }
    format!(
        "Select a smaller installed tier ({tiers}), {joined}{note}",
        tiers = installed_smaller
            .iter()
            .map(|tier| tier.to_uppercase())
            .collect::<Vec<_>>()
            .join(" / "),
        note = lever.clause(),
    )
}

/// Build the candle reject advice for `rejected_tier` from the tiers the loader can actually resolve.
/// Kept pure so bespoke candle lanes can share the same truthfulness contract as the main image lane.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn vram_reject_tail_for_tier(
    installed: Vec<&'static str>,
    rejected_tier: &str,
    lever: ResolutionLever,
) -> String {
    let smaller: Vec<&'static str> = installed
        .into_iter()
        .filter(|candidate| tier_quality_rank(candidate) < tier_quality_rank(rejected_tier))
        .collect();
    vram_reject_tail(&smaller, lever)
}

/// Candle per-tier capability fit for the downtier chooser (sc-10733): fold the full candle fit decision
/// (predicted resident peak vs the live budget, plus the sequential-residency second stage where the
/// provider stages components) down to [`TierFit`]. `Fits` = runs resident OR sequentially; `TooBig` =
/// won't run even one-component-at-a-time. `Unknown` (no budget / unmeasured tier) counts as `Fits` — the
/// gate never blocks without a signal.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_tier_fit(
    manifest_entry: &JsonObject,
    tier: &'static str,
    budget: Option<crate::vram_gate::VramBudget>,
    sequential_capable: bool,
    adapter_bytes: u64,
) -> TierFit {
    let needed =
        crate::vram_gate::predicted_peak_gb_with_adapter_bytes(manifest_entry, tier, adapter_bytes);
    match crate::vram_gate::resolve_offload(
        crate::vram_gate::fit_decision(needed, budget),
        sequential_capable,
    ) {
        crate::vram_gate::FitDecision::Fits | crate::vram_gate::FitDecision::Unknown => {
            TierFit::Fits
        }
        crate::vram_gate::FitDecision::Offload { available_gb, .. } => {
            // Resident won't fit but the provider stages — fits only if the MEASURED sequential peak
            // fits (unmeasured ⇒ best-effort run, so `sequential_overflow_gb` yields None ⇒ Fits).
            let seq_needed = crate::vram_gate::predicted_sequential_peak_gb_with_adapter_bytes(
                manifest_entry,
                tier,
                adapter_bytes,
            );
            match crate::vram_gate::sequential_overflow_gb(seq_needed, budget) {
                Some(seq_gb) => TierFit::TooBig {
                    needed_gb: seq_gb,
                    available_gb,
                },
                None => TierFit::Fits,
            }
        }
        crate::vram_gate::FitDecision::TooBig {
            needed_gb,
            available_gb,
        } => TierFit::TooBig {
            needed_gb,
            available_gb,
        },
    }
}

/// A generic SDXL request has a narrower, artifact-bound staging contract than its descriptor.
/// Bespoke edit/IP/detail lanes never reach this path. Keep the legacy tier verdict unless the
/// exact request selector has admitted the candidate; a declaration alone grants no saving.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_tier_fit_after_shared_selection(
    legacy: TierFit,
    selection: Option<&crate::candle_memory_strategy::CandleMemoryEvaluation>,
) -> TierFit {
    if selection.is_some() {
        TierFit::Fits
    } else {
        legacy
    }
}

/// FLUX.2-dev's caption-upsample lifecycle has exact bounded high-waters only for the Q4 and Q8
/// turnkey tiers. `predicted_peak_gb` intentionally falls back to `minMemoryGb` for legacy sparse
/// manifests, but doing that here would admit the unmeasured 113 GB BF16 tier at the 48 GB Q4 floor.
/// Keep this route-specific until another provider rotates its calibration identity and needs the
/// same evidence rule.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn validate_candle_tier_memory_evidence(
    model: &str,
    manifest_entry: &JsonObject,
    tier: &str,
) -> WorkerResult<()> {
    if model == "flux2_dev" && !has_candle_tier_peak_row(manifest_entry, tier) {
        return Err(WorkerError::InvalidPayload(format!(
            "{model} cannot run the resolved {tier} Candle tier because no current caption-aware \
             candle.vramGbByTier evidence exists for it; select Q4/Q8 or update SceneWorks"
        )));
    }
    Ok(())
}

/// Carry the legacy Candle fit gate's resident/staged choice into the request-scoped runtime
/// contract. The load-time policy remains populated for compatibility with older and non-image
/// providers, but current image providers use this bit as the sole lifecycle authority.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn apply_request_scoped_candle_residency(
    use_sequential: bool,
    memory: &mut Option<gen_core::GenerationMemory>,
) {
    if use_sequential {
        memory.get_or_insert_with(Default::default).stage_residency = true;
    }
}

#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn rejects_unverified_shared_memory_fallback(
    engine_id: &str,
    optimized_strategy_selected: bool,
) -> bool {
    matches!(
        engine_id,
        "z_image"
            | "z_image_turbo"
            | "flux2_dev"
            | "flux2_klein_9b"
            | "chroma1_hd"
            | "chroma1_base"
            | "chroma1_flash"
    ) && !optimized_strategy_selected
}

#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn verified_only_memory_family_label(engine_id: &str) -> Option<&'static str> {
    match engine_id {
        "z_image" | "z_image_turbo" => Some("Z-Image"),
        "flux2_dev" => Some("FLUX.2-dev"),
        "flux2_klein_9b" => Some("FLUX.2 Klein"),
        "chroma1_hd" | "chroma1_base" | "chroma1_flash" => Some("Chroma1"),
        _ => None,
    }
}

/// Caller-visible explanation for the one Krea memory decision whose latency would otherwise look
/// like a hung render (sc-16104). This is deliberately presentation-small: the worker folds the note
/// into its ordinary progress message rather than creating a dialog, toast, or separate UI surface.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct AutoTierStreamingDisclosure {
    pub(super) tier: String,
    pub(super) measured_materialization_ms_per_block: Option<f64>,
    pub(super) measured_materialization_ms_per_step: Option<u64>,
}

pub(super) const AUTO_TIER_STREAMING_DISCLOSURE_KEY: &str = "memoryStrategyDisclosure";

pub(super) fn scrub_untrusted_memory_strategy_disclosure(raw_settings: &mut JsonObject) {
    raw_settings.remove(AUTO_TIER_STREAMING_DISCLOSURE_KEY);
}

impl AutoTierStreamingDisclosure {
    pub(super) fn progress_message(&self, progress: &str) -> String {
        format!(
            "{progress} Streaming transformer blocks to hold the auto-selected {} tier.",
            self.tier
        )
    }

    #[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
    pub(super) fn record(&self) -> Value {
        json!({
            "strategy": "bounded_transformer_residency",
            "cause": "streaming_transformer_blocks_to_hold_auto_selected_tier",
            "tier": self.tier,
            "measuredMaterializationMsPerBlock": self.measured_materialization_ms_per_block,
            "measuredMaterializationMsPerStep": self.measured_materialization_ms_per_step,
            "measurementSource": "sc-16096-rtx-pro-6000-blackwell",
        })
    }

    #[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
    pub(super) fn telemetry(&self, job_id: &str, model: &str) -> Value {
        let mut telemetry = self.record();
        let fields = telemetry
            .as_object_mut()
            .expect("memory-strategy disclosure record is an object");
        fields.insert("jobId".to_owned(), Value::String(job_id.to_owned()));
        fields.insert("model".to_owned(), Value::String(model.to_owned()));
        telemetry
    }
}

pub(super) fn progress_with_auto_tier_streaming(
    progress: &str,
    disclosure: Option<&AutoTierStreamingDisclosure>,
) -> String {
    disclosure.map_or_else(
        || progress.to_owned(),
        |disclosure| disclosure.progress_message(progress),
    )
}

#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AutoTierFitDecision {
    StreamedBlocks,
    Other,
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn auto_tier_fit_decision(
    final_fit: Option<&crate::vram_gate::KreaTurboFit>,
) -> AutoTierFitDecision {
    if matches!(
        final_fit,
        Some(crate::vram_gate::KreaTurboFit::Fits {
            selection: gen_core::MemorySelection {
                strategy: gen_core::MemoryStrategy::BoundedTransformerResidency,
                ..
            },
            ..
        })
    ) {
        AutoTierFitDecision::StreamedBlocks
    } else {
        AutoTierFitDecision::Other
    }
}

/// SC-16096 measured Krea's real packed q4/q8 window materialization after the device-format sidecar
/// fix. Krea Turbo is a fixed 30-block, CFG-free single-forward denoiser, so block median x 30 is the
/// measured materialization component paid by each rung-4 denoise step. Dense bf16 was not measured;
/// disclose the strategy there without inventing a number.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn krea_streaming_measurement(tier: &str) -> (Option<f64>, Option<u64>) {
    let block_ms: Option<f64> = match tier {
        "q4" => Some(56.2),
        "q8" => Some(101.7),
        _ => None,
    };
    let step_ms = block_ms.map(|milliseconds| (milliseconds * 30.0).round() as u64);
    (block_ms, step_ms)
}

/// A disclosure exists only for Auto plus the final rung-4 selection. Explicit picks keep bypassing
/// the capability downtier and do not claim that Auto retained their tier; cheaper rungs do not claim
/// transformer streaming. These two guards are mutation-pinned below.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
pub(super) fn auto_tier_streaming_disclosure(
    explicit_pick: bool,
    tier: &str,
    final_fit: AutoTierFitDecision,
) -> Option<AutoTierStreamingDisclosure> {
    if explicit_pick || final_fit != AutoTierFitDecision::StreamedBlocks {
        return None;
    }
    let (measured_materialization_ms_per_block, measured_materialization_ms_per_step) =
        krea_streaming_measurement(tier);
    Some(AutoTierStreamingDisclosure {
        tier: tier.to_owned(),
        measured_materialization_ms_per_block,
        measured_materialization_ms_per_step,
    })
}

/// Whether a candle job may use Krea Turbo's request-scoped, quality-preserving memory ladder.
/// Keep every exclusion explicit: these surfaces have distinct component graphs or denoise contracts.
#[cfg(any(test, all(not(target_os = "macos"), feature = "backend-candle")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn krea_turbo_memory_route(
    engine_id: &str,
    has_convrot: bool,
    has_edit_reference: bool,
    has_img2img_reference: bool,
    has_edit_references: bool,
    has_edit_mask: bool,
    has_hires_fix: bool,
    use_pid: bool,
) -> bool {
    engine_id == "krea_2_turbo"
        && !has_convrot
        && !has_edit_reference
        && !has_img2img_reference
        && !has_edit_references
        && !has_edit_mask
        && !has_hires_fix
        && !use_pid
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn krea_runtime_evidence_context(
    request: &ImageRequest,
    settings: &Settings,
    tier: &str,
    resolved_artifact_root: &Path,
) -> Option<crate::vram_gate::KreaRuntimeEvidenceContext> {
    let download = request
        .model_manifest_entry
        .get("downloads")?
        .as_array()?
        .iter()
        .find(|download| download.get("variant").and_then(Value::as_str) == Some(tier))?;
    let provider = download.get("provider")?.as_str()?;
    let repository = download.get("repo")?.as_str()?;
    let revision = download.get("revision")?.as_str()?;
    let pinned_snapshot_root = crate::model_jobs::huggingface_pinned_snapshot_dir(
        &settings.data_dir,
        repository,
        revision,
    )?;
    crate::vram_gate::KreaRuntimeEvidenceContext::inspect_turbo_fit(
        "krea_2_turbo",
        "candle",
        &settings.gpu_id,
        crate::gpu::cached_compute_cap(),
        provider,
        repository,
        revision,
        tier,
        resolved_artifact_root,
        &pinned_snapshot_root,
    )
}

/// The constrained Krea ladder preserves the precision selected for the request. Its memory rungs
/// change residency only; they must never make the generic capability clamp silently cross from Q8
/// to Q4 before the ladder can return its measured reject/lower-resolution result.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn capability_downtier_floor<'a>(
    krea_turbo_ladder: bool,
    resolved_tier: &'a str,
    manifest_floor: Option<&'a str>,
) -> Option<&'a str> {
    if krea_turbo_ladder {
        Some(resolved_tier)
    } else {
        manifest_floor
    }
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn log_krea_turbo_evidence_exclusion(
    model: &str,
    tier: &str,
    width: u32,
    height: u32,
    consumer: &str,
    reason: Option<gen_core::MemoryEvidenceVerdict>,
) {
    tracing::warn!(
        model,
        tier,
        width,
        height,
        consumer,
        reason = ?reason.unwrap_or(gen_core::MemoryEvidenceVerdict::Missing),
        "shared Krea memory-strategy evidence excluded; optimized execution is disabled"
    );
}

/// Unverified Krea evidence may retain only the baseline resident behavior. In particular, a legacy
/// sequential estimate fitting does not authorize an optimized load after the shared contract failed.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn krea_unverified_resident_decision(
    needed_gb: Option<f64>,
    budget: Option<crate::vram_gate::VramBudget>,
) -> crate::vram_gate::FitDecision {
    crate::vram_gate::fit_decision(needed_gb, budget)
}

/// Truthful Krea Turbo capability-clamp rejection. The ladder's same-tier floor prevents a silent
/// quality downgrade, but lower installed tiers remain a valid MANUAL escape and must be named when
/// lowering resolution cannot cross the measured fixed-weight floor.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn krea_turbo_capability_reject_message(
    model: &str,
    tier: &str,
    needed_gb: f64,
    available_gb: f64,
    gpu_id: &str,
    lower_resolution: Option<(u32, u32)>,
    installed_smaller: &[&str],
) -> String {
    let mut options = Vec::new();
    if let Some((smaller_w, smaller_h)) = lower_resolution {
        options.push(format!(
            "lower the output resolution to {smaller_w}x{smaller_h} or below"
        ));
    }
    if !installed_smaller.is_empty() {
        options.push(format!(
            "select a smaller installed tier ({})",
            installed_smaller
                .iter()
                .map(|candidate| candidate.to_uppercase())
                .collect::<Vec<_>>()
                .join(" / ")
        ));
    }
    options.push("run on a GPU with more VRAM".to_owned());
    format!(
        "{model} at its {tier} precision floor needs {needed_gb:.2} GiB of VRAM even after every \
         measured constrained-memory rung, but GPU {gpu_id} has {available_gb:.2} GiB available. \
         {options}.",
        options = options.join(", or "),
    )
}

/// Truthful capability-clamp rejection for the generic candle lane (sc-10733), reached once every
/// installed tier in `[floor, default]` has been scored by [`candle_tier_fit`] and none fits.
///
/// `smallest_tier` is the least-demanding tier that was evaluated, so a smaller tier is not an
/// escape. Whether a smaller output is one depends on `lever`: `candle_tier_fit` prices a manifest
/// row and never reads the request geometry ([`ResolutionLever::Blind`]), but the SDXL admission
/// path overrides that verdict with the shared selector, which prices the geometry
/// ([`ResolutionLever::Priced`]). Kept pure so the contract is testable without a GPU, exactly
/// like [`krea_turbo_capability_reject_message`] beside it.
///
/// It also says WHAT the number covers, in `video_peak_too_big_error`'s established spelling. #3197's
/// reporter read a ~14 GB demand against a tier Model Manager lists at 2.5 GiB and concluded the gate
/// was broken; those are a whole-render peak and an on-disk size, and the SDXL candle lane's own
/// `config/memory-anchors.json` rows (q4 at 10.87 GiB for 768x768, q8 at 13.78 GiB for 1024x1024,
/// decode-dominated) say the big one is real. A reject that quotes a number nobody can reconcile with
/// the model card produces the bug report instead of the answer.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_capability_reject_message(
    model: &str,
    smallest_tier: &str,
    needed_gb: f64,
    available_gb: f64,
    gpu_id: &str,
    lever: ResolutionLever,
) -> String {
    let (_, leading) = lever.levers();
    let mut action = leading.to_owned();
    action[..1].make_ascii_uppercase();
    format!(
        "{model} needs ~{needed} GB of VRAM even at the smallest installed tier it can run \
         ({smallest_tier}) — the peak of the whole render, the resident weights plus the denoise and \
         VAE-decode transient, not the tier's on-disk size — but GPU {gpu_id} has ~{available} GB \
         available. {action}{note}",
        needed = needed_gb.round() as i64,
        available = available_gb.round() as i64,
        note = lever.clause(),
    )
}

/// Resolve the Candle load quant + recipe bit count from the FINAL tier the disk resolver (or the
/// capability clamp) selected. That tier is also handed to the shared memory evaluator, so the
/// provider's `LoadSpec` and request scope cannot describe different numeric artifacts.
///
/// Dense-TE turnkeys keep their full-precision text encoder while still recording the packed DiT's
/// resolved bits. Opaque/flat paths retain the request-derived behavior because there is no tier
/// basename to make a stronger artifact claim.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn candle_resolved_tier_key(
    request: &ImageRequest,
    weights_dir: &Path,
    convrot_resolved: bool,
) -> &'static str {
    gate_tier_key(
        convrot_resolved,
        weights_dir,
        &request.advanced,
        &request.model_manifest_entry,
        nvfp4_selected(request, nvfp4_host_eligible(), Some(weights_dir)),
    )
}

/// Run the shared selector for an uncached, bespoke SDXL provider after every physical component
/// has been assembled. This is the only admission that can authorize the subsequent loader; the
/// older scalar gate remains useful telemetry but cannot mint an optimized execution plan.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn admit_sdxl_bespoke_memory(
    request: &ImageRequest,
    settings: &Settings,
    mut spec: LoadSpec,
    request_mode: &str,
    width: u32,
    height: u32,
    reference_count: u32,
    use_pid: bool,
    additional_artifacts_certified: bool,
) -> WorkerResult<(
    crate::candle_memory_strategy::CandleMemoryEvaluation,
    LoadSpec,
)> {
    let WeightsSource::Dir(weights_dir) = &spec.weights else {
        return Err(WorkerError::InvalidPayload(
            "SDXL optimized admission requires a catalog snapshot directory".to_owned(),
        ));
    };
    let tier = candle_resolved_tier_key(request, weights_dir, false);
    spec = match tier {
        "q4" => spec.with_quant(Quant::Q4),
        "q8" => spec.with_quant(Quant::Q8),
        "bf16" => spec,
        other => {
            return Err(WorkerError::InvalidPayload(format!(
                "SDXL Candle route resolved unsupported physical tier {other:?}"
            )))
        }
    };
    spec = apply_declared_candle_image_load_shape(
        "sdxl",
        Some(tier),
        request_mode,
        &request.model_manifest_entry,
        spec,
        false,
    );
    spec.prepare_file_sources().map_err(|error| {
        WorkerError::InvalidPayload(format!(
            "SDXL could not retain exact component/adapter receipts: {error}"
        ))
    })?;
    let contract = runtime_cuda::providers::sdxl::provider_contract_for_spec(&spec)
        .map_err(|error| WorkerError::InvalidPayload(format!("SDXL receipt refused: {error}")))?;
    let runtime_overlay_bytes = contract.asset_facts.overlay_bytes;
    let artifact_is_certified = additional_artifacts_certified
        && candle_certified_load_spec("sdxl", settings, &spec, &request.model_manifest_entry, tier);
    let budget = crate::vram_gate::live_capped_budget(&settings.gpu_id).await;
    let predicted_peak = crate::vram_gate::predicted_peak_gb_with_adapter_bytes(
        &request.model_manifest_entry,
        tier,
        runtime_overlay_bytes,
    );
    let overlay = if spec.ip_adapter.is_some() {
        Some("identity")
    } else if spec.control.is_some() {
        Some("control")
    } else if !spec.adapters.is_empty() {
        Some("lora")
    } else {
        None
    };
    let evaluation = crate::candle_memory_strategy::evaluate_shared_bespoke_image(
        "sdxl",
        &request.model,
        &spec,
        artifact_is_certified,
        &request.model_manifest_entry,
        tier,
        request_mode,
        overlay,
        gen_core::MemoryGeometry {
            width,
            height,
            batch: 1,
            frames: 1,
            reference_count,
        },
        reference_count > 0,
        use_pid,
        false,
        budget,
        // This lane gates on the raw probe (no reclaimable credit), so the reserve derives from it.
        budget.map_or(0.0, crate::vram_gate::ladder_reserve_gb),
        predicted_peak,
        runtime_overlay_bytes,
        gen_core::MemoryCacheState::Cold,
        contract,
        crate::candle_memory_strategy::SDXL_REQUEST_EVIDENCE_REVISION,
    )?
    .ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "SDXL {} {} at {tier} has no exact shared memory strategy before load",
            request.model, request_mode
        ))
    })?;
    Ok((evaluation, spec))
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn candle_quant_for_resolved_tier(
    request: &ImageRequest,
    tier: &str,
    weights_dir: &Path,
    supports_quant: bool,
    force_dense: bool,
) -> (Option<Quant>, Option<i64>) {
    if matches!(request.model.as_str(), "sana_1600m" | "sana_sprint_1600m") {
        return (None, None);
    }
    let resolved_bits = match tier {
        "bf16" => None,
        "q4" => Some(4),
        "q8" => Some(8),
        // Distinct identities such as NVFP4 and opaque modelPath roots keep the existing resolver,
        // but against the FINAL directory so NVFP4 cannot survive a fallback to another artifact.
        _ if !force_dense && supports_quant => return resolve_quant(request, Some(weights_dir)),
        _ => return (None, None),
    };
    if is_dense_te_tier(request) {
        return (None, resolved_bits);
    }
    // Two families reach here with physically packed q4/q8 transformer tensors that the provider
    // discovers from the selected directory rather than from `LoadSpec.quantize`:
    //   * the Chroma/SD3.5 turnkeys, detected from their exact header/config receipt — setting
    //     `quantize` would request a second on-the-fly quantization and the provider refuses it;
    //   * the pinned candle-gen-flux descriptor, which advertises `supported_quants: []` and
    //     rejects a nonempty `LoadSpec.quantize` outright.
    //   * sc-22732: the two Ideogram 4 routes. `candle-gen-ideogram`'s descriptor advertises
    //     `supported_quants: [Q4, Q8]` so this A/B tier toggle engages, but its exact directory
    //     route — `IdeogramLoadReceipt::capture` -> `validate_load_shape`, on the PRODUCTION
    //     `load`/`load_turbo` path, not a test-only one — refuses a spec whose `quantize` is
    //     `Some(_)` by name ("exact Ideogram directory route requires precision=Bf16 and
    //     quantize=None") and proves the tier off the packed safetensors headers instead. Sending
    //     the resolved quant therefore failed every candle Ideogram q4/q8 load at the loader; the
    //     artifact is the authority on the tier, so the load instruction stays empty and the
    //     resolved bits still reach the recipe.
    // Keep the load instruction empty while retaining the resolved artifact bits for the recipe
    // and later telemetry. This is deliberately narrower than `!supports_quant`: other descriptors
    // with an empty list are dense-only and must not acquire a packed-tier receipt.
    if matches!(
        request.model.as_str(),
        "chroma1_hd"
            | "chroma1_base"
            | "chroma1_flash"
            | "sd3_5_large"
            | "sd3_5_large_turbo"
            | "sd3_5_medium"
            | "flux_schnell"
            | "flux_dev"
            | "ideogram_4"
            | "ideogram_4_turbo"
    ) {
        return (None, resolved_bits);
    }
    if force_dense || !supports_quant {
        return (None, None);
    }
    match tier {
        "bf16" => (None, None),
        "q4" => (Some(Quant::Q4), Some(4)),
        "q8" => (Some(Quant::Q8), Some(8)),
        _ => unreachable!("resolved bits above accepts only bf16/q4/q8"),
    }
}

/// sc-13960 — the evict-then-reclaim gate driver the bespoke (non-cache-evicting) candle image lanes
/// (`qwen_edit_candle` / `krea_control_candle`) share. Runs the lane's PURE `gate(budget)` against a
/// **two-pass** budget and returns the plan to act on plus the budget it was resolved against:
///
///  1. **Raw pass.** Gate against raw live free VRAM (no reclaim). Correct as-is: a resident txt2img
///     generator stays live and co-resident with the bespoke load, so its cudarc pool pages are NOT
///     free — crediting them would over-admit an OOM (sc-13588's documented reason these lanes budget
///     raw). If nothing is reclaimable (cold pool) — or the plan the raw budget yields already stands
///     after folding the pool — return it here, and **the warm txt2img generator cache is preserved**
///     (the deliberate tradeoff: we do NOT evict on every render, only when it changes the outcome).
///  2. **Reclaim pass.** Otherwise gate again against `free + reclaimable_pool` and, when that yields a
///     BETTER plan (`reclaim_improves` — the budget only ever grows, so a changed plan is always a
///     higher residency / an admit-instead-of-reject), EVICT the single-slot generator cache so the
///     resident generator's pages become genuinely free, making the reclaim credit honest, then act on
///     the reclaimed plan. This is the missing half of sc-11023 for the bespoke lanes: a second
///     edit/control render in the same worker no longer sees the first render's dropped-but-pooled
///     pages as unavailable.
///
/// `reclaim_improves(&raw, &reclaimed)` returns whether the reclaimed plan is worth an evict — `false`
/// when it is the same action as raw (including two rejects that differ only in their reported
/// free-VRAM number, which must NOT trigger a pointless evict). Mirrors the candle video comfyui lane's
/// `generator_cache::with_uncached_generator` precedent (evict, then reclaim) — see that function and
/// `video_jobs::candle_video_vram_budget`.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) async fn gate_with_evict_reclaim<D>(
    gpu_id: &str,
    raw_budget: Option<crate::vram_gate::VramBudget>,
    gate: impl Fn(Option<crate::vram_gate::VramBudget>) -> D,
    reclaim_improves: impl Fn(&D, &D) -> bool,
) -> WorkerResult<(D, Option<crate::vram_gate::VramBudget>)> {
    let raw = gate(raw_budget);
    let reclaimable = crate::vram_gate::reclaimable_pool_gb(gpu_id);
    // Cold pool: nothing this process pooled to reclaim, hence nothing resident to evict — gate exactly
    // as the pre-sc-13960 lanes did.
    if reclaimable <= 0.0 {
        return Ok((raw, raw_budget));
    }
    let reclaimed_budget =
        raw_budget.map(|budget| crate::vram_gate::with_reclaimable(budget, reclaimable));
    let reclaimed = gate(reclaimed_budget);
    if !reclaim_improves(&raw, &reclaimed) {
        // Reclaim would not change the plan — keep the warm txt2img cache rather than evict for nothing.
        return Ok((raw, raw_budget));
    }
    let evicted = crate::generator_cache::evict_cached_generator().await?;
    tracing::info!(
        gpu_id,
        evicted,
        reclaimable_gb = reclaimable,
        "candle bespoke-lane VRAM gate: evicted the resident generator to reclaim its cudarc pool so \
         this render admits at a higher residency (sc-13960)"
    );
    Ok((reclaimed, reclaimed_budget))
}

#[cfg(native_lane)]
#[cfg(test)]
#[path = "tests/candle_request_residency_tests.rs"]
mod candle_request_residency_tests;

#[cfg(native_lane)]
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
#[path = "tests/krea_turbo_memory_route_tests.rs"]
mod krea_turbo_memory_route_tests;

#[cfg(native_lane)]
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
#[path = "tests/candle_resolved_tier_contract_tests.rs"]
mod candle_resolved_tier_contract_tests;

#[cfg(native_lane)]
#[cfg(all(test, native_lane))]
#[path = "tests/capability_downtier_tests.rs"]
mod capability_downtier_tests;

#[cfg(native_lane)]
#[cfg(all(test, target_os = "macos"))]
#[path = "tests/mlx_downtier_emulation_tests.rs"]
mod mlx_downtier_emulation_tests;

#[cfg(native_lane)]
#[cfg(all(test, not(target_os = "macos"), feature = "backend-candle"))]
#[path = "tests/candle_downtier_emulation_tests.rs"]
mod candle_downtier_emulation_tests;
