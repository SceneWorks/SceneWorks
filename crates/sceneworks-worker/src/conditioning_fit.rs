//! Candle **conditioning-overlay** VRAM admission gate (sc-16069, epic 15448).
//!
//! Every candle route that overlays a SECOND network on the base model — strict-pose ControlNet,
//! IP-Adapter, face/identity encoder — is diverted by `resolve_candle_image_route` BEFORE the generic
//! `CandleTxt2Img` arm (the divert is deliberate: those lanes share candle txt2img model ids, so
//! without it a pose job would be silently rendered as plain txt2img with the poses dropped). The
//! consequence was unintended: the shared admission check the generic arm reaches
//! (`vram_gate` inside `generate_candle_stream`) and the one every registry-loaded generator reaches
//! (`mlx_fit_gate::apply_residency_policy` inside `generator_cache`) were BOTH bypassed, so eleven
//! conditioning routes allocated with **no pre-flight check at all** — no rejection, no warning, just a
//! reactive CUDA OOM mid-render. This module is the gate they were missing.
//!
//! ## Why these lanes cannot use the existing gates
//!
//! On candle each conditioning model loads as a plain concrete type through a bespoke entrypoint that
//! takes a private `*Paths` struct (`QwenFunControl::load(&QwenFunControlPaths)`,
//! `IpAdapterSdxl::load(&IpAdapterSdxlPaths)`, `PulidFlux::load(&PulidFluxPaths)`, …) — never a
//! `LoadSpec`, never a `Box<dyn Generator>`. `apply_residency_policy` reads a `LoadSpec`; there is not
//! even a `LoadSpec` for it to read. And `vram_gate::predicted_peak_gb` reads
//! `candle.vramGbByTier[tier]`, which prices the BASE model alone — it cannot express a co-resident
//! overlay, and of the 4360 matrix cells with `overlay != "none"` exactly ZERO carry any historical,
//! current-environment or strategy-parameter verification. So neither existing gate can be pointed at
//! these routes as-is.
//!
//! ## What this gate is: an on-disk weights FLOOR
//!
//! `Σ(resident base weights) + Σ(overlay weights) + `[`HEADROOM_GB`] vs the live (or capped) free VRAM.
//! Deliberately crude, and deliberately a **lower bound** on the real peak: it prices no activations, no
//! denoise steady state and no VAE-decode spike, because nothing has measured an overlay cell (see
//! `docs/memory-strategy-contract.md`). A floor is the
//! only honest thing available, and for an admission gate the floor's direction is the SAFE one:
//!
//! * **It does not false-reject, PROVIDED every declared path is loaded in full.** A reject then means the
//!   weights the load must hold, alone, already overflow free VRAM — impossible however the transients are
//!   priced. That satisfies the house never-block-without-evidence posture
//!   ([`crate::vram_gate::FitDecision::Unknown`], [`crate::vram_gate::video_weights_fit_error`]).
//!   The proviso is load-bearing and is the whole reason the summation is careful: counting a path twice,
//!   or scanning a directory that holds checkpoints this render will not load, turns the floor into an
//!   OVER-count that can refuse a render which fits. [`ConditioningFootprint::from_paths`] therefore
//!   deduplicates nested paths, and each lane declares FILES wherever it knows them rather than a
//!   directory that might hold alternates. One residual route remains, shared with the loader: an
//!   operator `modelPath` override is used verbatim with no tier descent, so pointing it at a multi-tier
//!   turnkey root sums every tier — the loader reads the same directory, so the two at least agree.
//! * **It does under-admit protection.** A marginal render whose transients are what overflow still
//!   reaches the reactive CUDA OOM. That is strictly better than today (where *nothing* is refused) and
//!   is the honest limit of unmeasured evidence — inflating the floor with an invented activation fudge
//!   factor would manufacture rejections no measurement supports.
//!
//! This is the same "crude, honest floor now; principled peak once the contract has vocabulary for a
//! second resident network and one overlay cell is measured" split the epic declares, and the same
//! weights-floor shape [`crate::vram_gate::video_weights_fit_error`] (sc-12344) already ships for
//! candle video.
//!
//! ## Relationship to the Krea control ladder
//!
//! `krea_2_turbo`'s pose-ControlNet lane has the epic's only MEASURED gate ([`crate::krea_control_fit`])
//! and keeps it. It ALSO runs this floor, in its own preamble rather than through the shared driver — see
//! [`ConditioningAdmission::GatedInPreamble`] for the ordering reason. The two are not redundant: the
//! current measured tiers can reject from transient-aware peaks, and for a tier with no priced row
//! (`KreaControlFit::Unverified`) it cannot price the render at all — leaving the floor as the only check
//! that can refuse a host which cannot hold the weights. The route declares only the files it actually
//! loads so the floor does not over-count unused or repacked checkpoints.
//!
//! ## What sc-19055 migrated, and what it deliberately did not
//!
//! Epic 19048 R1 forbids a prediction law living in a backend-local module, and sc-19055's brief is
//! that the overlay accounting must survive the migration "as candidate composition in the shared
//! mechanism, not by flattening it away". Two things moved:
//!
//! * **The composition.** `base + overlay` was a hand-written `saturating_add` here. It is now the
//!   auxiliary term of [`crate::estimate_synthesis::compose_resident_floor_bytes`] under
//!   [`CONDITIONING_ENGAGED`] — the SAME law
//!   [`crate::estimate_synthesis::floor_weights_bytes`] applies to a provider contract's declared
//!   components, with the bytes sourced from disk instead of from a contract. Byte-identical today;
//!   the point is that an overlay a future contract declares `bounded_by` an engaged rung now drops
//!   out of the floor on its own, in one place, for both lanes.
//! * **The headroom.** [`HEADROOM_GB`] was a third literal `2.0`; it is now the candle lane's one
//!   constant.
//!
//! What did NOT move, and why the inventory still reports this mechanism geometry-blind:
//!
//! * **No estimate-margin grading.** [`crate::estimate_synthesis::graded_scalar_gb`] widens a
//!   `DeclaredFloor` by the lane's estimate margin, and that is right for a gate whose number is a
//!   *declared peak* being honest about its uncertainty. It is wrong here. This floor's whole
//!   contract is that it cannot false-reject (see above): a refusal means the weights ALONE
//!   overflow, which is true however the transients are priced. Widening it by 4% manufactures
//!   refusals no measurement supports — exactly the "invented activation fudge factor" the module
//!   note rules out — and it is the same call sc-19055 made for
//!   [`crate::vram_gate::video_weights_fit_error`], the video lane's on-disk weights sum, which
//!   stays ungraded for the same reason. Both are floors *by construction*, not declared peaks.
//! * **No geometry axis.** On-disk weight bytes do not vary with width, height or frames, so
//!   claiming this gate reads geometry would be a false claim in a generated artifact. `no` in the
//!   inventory's geometry column is the correct answer for a weights floor, and it stays `no`.
//! * **No shared SELECTOR.** [`crate::memory_strategy::select_strategy`] grades candidates against a
//!   `MemoryProviderContract`, and these lanes have none to give it — that is the premise of this
//!   whole module (see "Why these lanes cannot use the existing gates"). Reaching the selector here
//!   is blocked upstream, in the same way and for the same reason as the unreached image class
//!   `candle_admission_decisions::the_unreached_image_class_is_exactly_the_routes_declaring_no_provider_contract`
//!   pins: no provider contract, no candidates. Sharing the composition law is what IS reachable at
//!   this pin, and it is what makes the eventual selector migration a rewiring rather than a rewrite.
//!
//! Everything here is pure and unit-tested; the live `nvidia-smi` reading lives in [`crate::gpu`] and
//! the per-lane wiring is in `image_jobs/conditioning_gate.rs`.

use super::*;

use crate::fit_gate::BYTES_PER_GIB;
use crate::vram_gate::VramBudget;

/// Fixed transient/runtime headroom (GB) added on top of the on-disk weights floor: allocator slack
/// plus CUDA context overhead, NOT an activation estimate (this gate makes no claim about
/// activations; see the module note on why the floor stays a floor).
///
/// **The candle lane's ONE reserve** (sc-19055, epic 19048 R1). This was a literal `2.0` documented
/// as "mirrors [`crate::vram_gate`]'s and [`crate::krea_control_fit`]'s own `HEADROOM_GB`" — three
/// copies of one number, which is the drift hazard R1 forbids and the same one
/// [`crate::estimate_synthesis::binding_phase`] was extracted to close ("three mirrors have three
/// chances to stop mirroring"). [`crate::krea_control_fit`] already aliased the constant; this is
/// the last copy. The value is unchanged, so no admission decision moves.
const HEADROOM_GB: f64 = crate::vram_gate::HEADROOM_GB;

/// The composition a conditioning render actually runs.
///
/// These lanes load through bespoke concrete entrypoints (`QwenFunControl::load(&QwenFunControlPaths)`
/// and friends) with no rung selection and no `LoadSpec`, so nothing stages, tiles, chunks or windows:
/// the load is whole-model resident with the overlay co-resident beside it. That is *why* the two
/// terms are additive, and stating it as the engaged composition — rather than as a hand-written `+`
/// — is what lets [`crate::estimate_synthesis::compose_resident_floor_bytes`] drop the overlay out on
/// its own the day one of these lanes gains a contract that bounds it.
const CONDITIONING_ENGAGED: &[gen_core::MemoryStrategy] = &[gen_core::MemoryStrategy::Resident];

/// WHERE one candle conditioning lane is admitted — not whether.
///
/// Either the lane hands its on-disk footprint to the shared driver, which gates it, or the lane gates
/// ITSELF earlier in its own preamble and says so, naming the module involved.
/// `CandleStrictControl::conditioning_admission` returns this with no default body, so adding a lane means
/// choosing at compile time. Both arms end in an admission check; neither is an opt-out.
///
/// **Why the second arm has to exist.** The Krea pose-control lane must run its check BEFORE the shared
/// driver would: its preamble walks the measured [`crate::krea_control_fit`] ladder and then records the
/// admitted peak with [`crate::vram_gate::note_loaded_peak`]. A rejection arriving after that call would
/// leave a reclaimable high-water standing for a load that never allocated — over-crediting the pool,
/// which lets the NEXT gate over-admit an OOM. Ordering, not exemption, is what that lane needs, so it
/// calls the floor itself up front and the driver stands down to avoid probing the GPU and re-gating
/// twice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ConditioningAdmission {
    /// The shared driver gates this lane on the on-disk weights floor. Every conditioning lane except
    /// Krea pose-control.
    Floor(ConditioningFootprint),
    /// This lane already gated itself, earlier, in its own preamble — because it must sequence its check
    /// before something else it does (see the enum note). `gate` names the module that decides alongside
    /// the floor there.
    ///
    /// `image_jobs/tests.rs::every_candle_conditioning_route_is_admitted_through_a_gate` bounds this to
    /// exactly ONE lane and asserts that lane really does call both the floor and the named gate, so it
    /// cannot become a quiet way to skip admission.
    GatedInPreamble { gate: &'static str },
}

/// The resident on-disk footprint of one candle conditioning render, plus the labels its rejection
/// message needs.
///
/// Built by each lane from the paths it has ALREADY resolved (its base tier dir and its overlay weight
/// file), so the gate performs no resolution of its own and stays a pure decision over bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ConditioningFootprint {
    /// The base model as the user selected it (`request.model`) — the thing whose tier they can change.
    pub(crate) model_label: String,
    /// What the overlay IS, in the user's terms ("strict-pose ControlNet branch", "IP-Adapter",
    /// "face-identity encoder"). Named separately from the model because it is the term that explains
    /// why this render needs more VRAM than a plain txt2img of the same model at the same tier.
    pub(crate) overlay_label: &'static str,
    /// On-disk bytes of the base weights this render holds resident.
    pub(crate) base_bytes: u64,
    /// On-disk bytes of the overlay network(s) held co-resident with the base.
    pub(crate) overlay_bytes: u64,
}

impl ConditioningFootprint {
    /// Sum a base weights DIRECTORY and any number of overlay paths (each either a single weight file
    /// or a directory of them).
    ///
    /// Reuses [`crate::mlx_fit_gate::sum_safetensors_bytes`] for directories so the HF-cache symlink
    /// handling (shards are symlinks into `blobs/`) and the AppleDouble `._*` skip are shared rather
    /// than re-implemented — the same reuse [`crate::vram_gate::wan_weight_bytes`] makes.
    ///
    /// An overlay path that is a plain FILE is measured with `metadata().len()`, because the packed
    /// control/adapter overlays are single files addressed directly (`q8/model.safetensors`,
    /// `ip-adapter-plus_sdxl_vit-h.safetensors`) and a directory scan of the file's parent would sum
    /// every sibling tier. Unreadable / missing ⇒ contributes 0, i.e. less evidence, never a phantom
    /// requirement.
    ///
    /// **NESTED paths are counted once.** Overlay paths are deduplicated by CONTAINMENT before summing:
    /// any path that sits inside another declared path (or inside `base_dir`, which is summed
    /// recursively) is dropped, because the enclosing directory scan already counts it. This is a
    /// property of the summation rather than of the caller, so it lives here — a lane cannot introduce
    /// the bug by declaring its paths in the obvious way.
    ///
    /// It is not hypothetical: PuLID's `ensure_pulid_candle_weights` returns `(adapter, eva, bundle)`
    /// where `bundle` IS the provider's `face_dir`, and `resolve_one` returns `bundle.join(file)` on the
    /// download-on-first-use path — so on a fresh install `adapter` and `eva` live INSIDE `bundle` and a
    /// naive sum counted ~2 GB twice. On a warm whole-repo-snapshot install the same two paths resolve
    /// OUTSIDE `bundle` and must still be counted, so the caller cannot simply omit them; only
    /// containment at sum time gets both installs right. An over-count is the one direction this gate
    /// must never have (see the module note), because it can refuse a render that fits.
    pub(crate) fn from_paths(
        model_label: &str,
        overlay_label: &'static str,
        base_dir: &Path,
        overlays: &[&Path],
    ) -> Self {
        Self {
            model_label: model_label.to_owned(),
            overlay_label,
            base_bytes: crate::mlx_fit_gate::sum_safetensors_bytes(base_dir),
            overlay_bytes: dedupe_contained(base_dir, overlays)
                .into_iter()
                .map(path_weight_bytes)
                .sum(),
        }
    }

    /// Total resident on-disk bytes for the composition this render runs, composed by the shared
    /// mechanism (sc-19055, epic 19048 R1/R2).
    ///
    /// The additive base+overlay accounting this gate exists for is not flattened away by the
    /// migration — it becomes the auxiliary term of
    /// [`crate::estimate_synthesis::compose_resident_floor_bytes`], the same law
    /// [`crate::estimate_synthesis::floor_weights_bytes`] applies to a provider contract's declared
    /// components. Under [`CONDITIONING_ENGAGED`] that is `base + overlay`, byte-identical to the
    /// `saturating_add` this replaced.
    ///
    /// **`conditioning_bytes: 0` is a claim about the EVIDENCE, not about the model.** A recursive
    /// `.safetensors` scan of a base directory cannot separate the text encoder from the DiT, so
    /// this gate does not know the split. Declaring it zero (rather than guessing) is what keeps the
    /// composition honest: were a rung ever to engage `StagedResidency` here, the law's
    /// `conditioning.max(heavy)` would return the whole base unreduced — no phantom saving from a
    /// split nothing measured. `transformer_bytes` is zero for the same reason.
    fn total_bytes(&self) -> u64 {
        crate::estimate_synthesis::compose_resident_floor_bytes(
            &crate::estimate_synthesis::ResidentWeights {
                conditioning_bytes: 0,
                heavy_bytes: self.base_bytes,
                transformer_bytes: 0,
            },
            CONDITIONING_ENGAGED,
            // The overlay network(s) held beside the base. `None` — nothing here declares a rung
            // that bounds them, because nothing here selects a rung at all.
            [(self.overlay_bytes, None)],
        )
    }
}

/// Drop every overlay path that is CONTAINED in `base_dir` or in another overlay path, so a recursive
/// directory scan and an explicit file inside it cannot both be counted (sc-16069 review finding).
///
/// Shortest-path-first: a path is kept only when no already-kept path is an ancestor of it, which keeps
/// the enclosing DIRECTORY (whose scan covers the descendant) and drops the descendant. Comparison is
/// lexical `Path::starts_with`, not canonicalized — every caller builds these paths from the same
/// `settings.data_dir` prefix, so the lexical test is the one that matches how they were constructed, and
/// canonicalizing would resolve HF-cache blob symlinks into a shared `blobs/` directory where unrelated
/// files would look nested.
fn dedupe_contained<'a>(base_dir: &Path, overlays: &[&'a Path]) -> Vec<&'a Path> {
    let mut ordered: Vec<&Path> = overlays.to_vec();
    // Shorter paths first so an ancestor is always considered before its descendants. `sort_by_key` is
    // stable, so equal-length paths keep their declared order.
    ordered.sort_by_key(|path| path.as_os_str().len());
    let mut kept: Vec<&Path> = Vec::with_capacity(ordered.len());
    for path in ordered {
        // `base_dir` is summed recursively by the caller, so anything inside it is already counted.
        if path.starts_with(base_dir) {
            continue;
        }
        if kept.iter().any(|k| path.starts_with(k)) {
            continue;
        }
        kept.push(path);
    }
    kept
}

/// The path inside a [`gen_core::WeightsSource`], for the overlay list. Both variants carry one path;
/// [`path_weight_bytes`] already distinguishes file from directory on disk, so the variant itself does
/// not need to be threaded through.
pub(crate) fn weights_source_path(source: &gen_core::WeightsSource) -> &Path {
    match source {
        gen_core::WeightsSource::Dir(path) | gen_core::WeightsSource::File(path) => path.as_path(),
    }
}

/// The extra CO-RESIDENT paths an opted-in PiD decoder adds to a render (epic 7840): the PiD student
/// checkpoint plus the `gemma-2-2b-it` caption-encoder snapshot. Empty when PiD is off.
///
/// Counted because they are genuinely held alongside the base and the overlay — the gemma snapshot alone
/// is several GB, so a floor that ignored it would under-state a PiD render by more than the overlay
/// itself on some lanes. Omitting them would still be *safe* (a lower floor only admits), but this gate's
/// job is to be as honest as the on-disk evidence allows.
pub(crate) fn pid_paths(pid: Option<&gen_core::PidWeights>) -> Vec<&Path> {
    pid.map(|pid| {
        vec![
            weights_source_path(&pid.checkpoint),
            weights_source_path(&pid.gemma),
        ]
    })
    .unwrap_or_default()
}

/// On-disk weight bytes at `path`: the file's own length when it is a file, else a recursive
/// `.safetensors` sum when it is a directory. `0` when neither (missing / unreadable) ⇒ no signal.
fn path_weight_bytes(path: &Path) -> u64 {
    match std::fs::metadata(path) {
        Ok(meta) if meta.is_file() => meta.len(),
        Ok(meta) if meta.is_dir() => crate::mlx_fit_gate::sum_safetensors_bytes(path),
        _ => 0,
    }
}

/// The predicted resident FLOOR (GB) for a conditioning render: the co-resident base + overlay weights
/// plus [`HEADROOM_GB`].
///
/// `None` when the weights are unmeasurable (`total == 0` — a dir that could not be scanned, an
/// operator-supplied root in a format the scan does not recognize) ⇒ no signal ⇒ the caller admits,
/// exactly like [`crate::vram_gate::FitDecision::Unknown`] and
/// [`crate::fit_gate::mochi_needed_gb`]'s `weight_bytes == 0` arm. A gate that blocks without evidence
/// is a regression, not a safety net.
pub(crate) fn conditioning_floor_gb(footprint: &ConditioningFootprint) -> Option<f64> {
    let total = footprint.total_bytes();
    (total > 0).then(|| total as f64 / BYTES_PER_GIB + HEADROOM_GB)
}

/// The outcome of the conditioning admission check. Reported by [`decide`] so the caller can log the
/// reason it admitted — the epic's "no evidence, no rejection, and the reason is logged" requirement —
/// rather than the gate silently returning `Ok(())` for three materially different reasons.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ConditioningFit {
    /// No countable weights ⇒ nothing to compare. Admit.
    NoWeightEvidence,
    /// No live VRAM budget (non-NVIDIA host, `gpu_id == "cpu"`, or an unreadable `nvidia-smi`). Admit.
    NoBudget { floor_gb: f64 },
    /// The weights floor fits the budget. Admit. (This says nothing about the transients — see the
    /// module note; a marginal render can still reach the reactive CUDA OOM.)
    Fits { floor_gb: f64, available_gb: f64 },
    /// The weights ALONE overflow the budget: the render is impossible however its transients are
    /// priced. Reject before allocating.
    TooBig { floor_gb: f64, available_gb: f64 },
}

/// The pure conditioning admission decision. Both "no signal" arms admit and are DISTINGUISHED (rather
/// than collapsed into one `Unknown`) so the caller's log line names which evidence was missing.
pub(crate) fn decide(
    footprint: &ConditioningFootprint,
    budget: Option<VramBudget>,
) -> ConditioningFit {
    let Some(floor_gb) = conditioning_floor_gb(footprint) else {
        return ConditioningFit::NoWeightEvidence;
    };
    let Some(budget) = budget else {
        return ConditioningFit::NoBudget { floor_gb };
    };
    if budget.free_gb + f64::EPSILON < floor_gb {
        ConditioningFit::TooBig {
            floor_gb,
            available_gb: budget.free_gb,
        }
    } else {
        ConditioningFit::Fits {
            floor_gb,
            available_gb: budget.free_gb,
        }
    }
}

/// Build the actionable pre-flight rejection for a conditioning render that cannot fit.
///
/// Follows the house [`crate::mlx_fit_gate`] `too_big_error` convention — name the model, state what it
/// needs and what the machine has, offer the levers — and adds the two facts that are UNIQUE to a
/// conditioning route, because without them the number reads like the wrong model's total:
///
/// * the **overlay** is named and priced separately, so the user can see that this render is heavier
///   than a plain txt2img of the same model at the same tier, and why;
/// * the figure is declared a **floor** ("at least"), not a prediction. Claiming a measured peak here
///   would be a lie — no overlay cell has been measured — and a user told "needs ~34 GB" who then frees
///   34 GB and still OOMs has been misled. Saying "at least" is both honest and still actionable.
fn too_big_error(
    footprint: &ConditioningFootprint,
    floor_gb: f64,
    available_gb: f64,
    gpu_id: &str,
) -> WorkerError {
    // A lane whose conditioning adds no second CHECKPOINT (Z-Image identity-init reconditions the base
    // rather than overlaying a network) has nothing to price separately, and printing "the 0.0 GB
    // identity-init" would read as a bug and offer a lever that does not exist. Say what is true instead.
    let (what, breakdown, lever) = if footprint.overlay_bytes > 0 {
        (
            format!(
                "{} with a {}",
                footprint.model_label, footprint.overlay_label
            ),
            format!(
                "the {base} GB base model and the {overlay_gb} GB {overlay} are held co-resident, \
                 plus headroom",
                base = round_gb(footprint.base_bytes as f64 / BYTES_PER_GIB),
                overlay_gb = round_gb(footprint.overlay_bytes as f64 / BYTES_PER_GIB),
                overlay = footprint.overlay_label,
            ),
            format!("drop the {}", footprint.overlay_label),
        )
    } else {
        (
            format!("{} ({})", footprint.model_label, footprint.overlay_label),
            format!(
                "{base} GB of base weights held resident, plus headroom",
                base = round_gb(footprint.base_bytes as f64 / BYTES_PER_GIB),
            ),
            "render without conditioning".to_owned(),
        )
    };
    WorkerError::InvalidPayload(format!(
        "{what} needs at least ~{floor} GB of VRAM — {breakdown} — but GPU {gpu_id} has ~{available} \
         GB available. That is the weights alone, before the denoise and VAE-decode transients, so \
         this render cannot fit. Select a smaller quant tier, {lever}, or run on a GPU with more VRAM.",
        floor = round_gb(floor_gb),
        available = round_gb(available_gb),
    ))
}

/// One decimal place. The overlay term is often ~1 GB (an IP-Adapter is far smaller than its base), and
/// whole-GB rounding — which the base-model gates use, where every term is tens of GB — would print it
/// as "0 GB" and make the message look broken.
fn round_gb(value: f64) -> String {
    format!("{:.1}", (value * 10.0).round() / 10.0)
}

/// Turn a [`ConditioningFit`] into the lane's result, logging the reason on every path.
///
/// Split from [`decide`] so the whole decision stays testable with no GPU and no tracing subscriber,
/// and so the three admit reasons are individually assertable.
pub(crate) fn admit(
    fit: &ConditioningFit,
    footprint: &ConditioningFootprint,
    gpu_id: &str,
) -> WorkerResult<()> {
    match fit {
        // No countable weights: the gate has no evidence, so it must not block (the house posture).
        // Logged at WARN because it means a conditioning route is running UNGATED — the condition this
        // whole module exists to make visible rather than invisible.
        ConditioningFit::NoWeightEvidence => {
            tracing::warn!(
                model = %footprint.model_label,
                overlay = footprint.overlay_label,
                gpu_id,
                "candle conditioning fit-gate: admitted WITHOUT evidence — no countable on-disk \
                 weights for the base or the overlay, so no floor could be computed. The reactive \
                 CUDA-OOM backstop is the only protection for this render (sc-16069)."
            );
            Ok(())
        }
        // No live budget (non-NVIDIA host / unreadable nvidia-smi). Same posture, different missing
        // half.
        ConditioningFit::NoBudget { floor_gb } => {
            tracing::info!(
                model = %footprint.model_label,
                overlay = footprint.overlay_label,
                gpu_id,
                floor_gb,
                "candle conditioning fit-gate: admitted without a VRAM budget — no live reading for \
                 this device, so the weights floor has nothing to compare against (sc-16069)."
            );
            Ok(())
        }
        ConditioningFit::Fits {
            floor_gb,
            available_gb,
        } => {
            tracing::info!(
                model = %footprint.model_label,
                overlay = footprint.overlay_label,
                gpu_id,
                floor_gb,
                available_gb,
                base_gb = footprint.base_bytes as f64 / BYTES_PER_GIB,
                overlay_gb = footprint.overlay_bytes as f64 / BYTES_PER_GIB,
                "candle conditioning fit-gate: admitted — the co-resident base + overlay weights fit \
                 free VRAM. This is a FLOOR, not a measured peak: the denoise and VAE-decode \
                 transients are unpriced (no overlay cell has been measured), so a marginal render can \
                 still reach the reactive CUDA-OOM backstop (sc-16069)."
            );
            Ok(())
        }
        ConditioningFit::TooBig {
            floor_gb,
            available_gb,
        } => Err(too_big_error(footprint, *floor_gb, *available_gb, gpu_id)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn footprint(base_bytes: u64, overlay_bytes: u64) -> ConditioningFootprint {
        ConditioningFootprint {
            model_label: "qwen_image".to_owned(),
            overlay_label: "strict-pose ControlNet branch",
            base_bytes,
            overlay_bytes,
        }
    }

    fn budget(free_gb: f64) -> Option<VramBudget> {
        Some(VramBudget {
            free_gb,
            total_gb: 96.0,
        })
    }

    /// One GiB, so the fixtures below read in the unit the floor is expressed in.
    const GIB: u64 = 1_073_741_824;

    // ── The floor arithmetic ─────────────────────────────────────────────────────────────────────

    /// The floor is base + overlay + [`HEADROOM_GB`], and BOTH terms count. A mutation that drops the
    /// overlay term — the whole point of this gate, since the base alone is what the existing
    /// `vram_gate` already prices — moves this number and turns the test red.
    #[test]
    fn floor_is_base_plus_overlay_plus_headroom() {
        let floor = conditioning_floor_gb(&footprint(20 * GIB, 7 * GIB)).expect("weights present");
        assert!(
            (floor - (27.0 + HEADROOM_GB)).abs() < 1e-9,
            "floor {floor} should be 20 + 7 + {HEADROOM_GB}"
        );
        // Dropping the overlay must lower the floor by exactly the overlay's bytes — pins that the
        // overlay is a real term and not decoration.
        let base_only = conditioning_floor_gb(&footprint(20 * GIB, 0)).expect("weights present");
        assert!(
            (floor - base_only - 7.0).abs() < 1e-9,
            "the overlay must contribute its own bytes: {floor} vs {base_only}"
        );
    }

    /// **sc-19055: the floor is the MECHANISM's composition and the LANE's headroom, not this
    /// module's arithmetic.**
    ///
    /// Both halves are pinned against their sources rather than against a literal, so a re-hardcoded
    /// copy of either is red:
    ///
    /// * the reserve is compared to [`crate::vram_gate::HEADROOM_GB`] — the candle lane's one
    ///   constant. MUTATION PROOF: restoring the pre-sc-19055 `const HEADROOM_GB: f64 = 2.0;` and
    ///   moving `vram_gate`'s value turns this red, which the old literal could not do.
    /// * the composed total is compared to
    ///   [`crate::estimate_synthesis::compose_resident_floor_bytes`] driven with the same component
    ///   shape, so the gate and the mechanism cannot disagree about what a resident composition
    ///   holds.
    ///
    /// The interval is the open band between the base-only floor and the composed floor: a budget
    /// inside it admits one and refuses the other, so neither assertion can be satisfied vacuously.
    #[test]
    fn the_floor_is_the_shared_composition_plus_the_lane_headroom() {
        let f = footprint(20 * GIB, 7 * GIB);
        let composed = crate::estimate_synthesis::compose_resident_floor_bytes(
            &crate::estimate_synthesis::ResidentWeights {
                conditioning_bytes: 0,
                heavy_bytes: 20 * GIB,
                transformer_bytes: 0,
            },
            CONDITIONING_ENGAGED,
            [(7 * GIB, None)],
        );
        assert_eq!(composed, 27 * GIB, "resident composition holds both terms");
        let expected = composed as f64 / BYTES_PER_GIB + crate::vram_gate::HEADROOM_GB;
        let floor = conditioning_floor_gb(&f).expect("weights present");
        assert!(
            (floor - expected).abs() < 1e-9,
            "floor {floor} must be the shared composition plus the lane's headroom ({expected})"
        );

        // The open band between the base-only floor and the composed floor. A budget inside it must
        // separate the two, which is what stops either assertion above from being vacuous.
        let base_only = conditioning_floor_gb(&footprint(20 * GIB, 0)).expect("weights present");
        let inside = (base_only + floor) / 2.0;
        assert!(
            base_only < inside && inside < floor,
            "the band must be open: {base_only} < {inside} < {floor}"
        );
        assert!(matches!(
            decide(&footprint(20 * GIB, 0), budget(inside)),
            ConditioningFit::Fits { .. }
        ));
        assert!(matches!(
            decide(&f, budget(inside)),
            ConditioningFit::TooBig { .. }
        ));
    }

    /// A composition that engages `StagedResidency` promises NO saving here, because this gate
    /// cannot see the conditioning/heavy split a directory scan does not expose (sc-19055).
    ///
    /// The mechanism's law reduces a staged base to `conditioning.max(heavy)`; with
    /// `conditioning_bytes: 0` that is `heavy` unchanged, so the floor cannot shrink on the strength
    /// of a split nothing measured. Pinned because the tempting "improvement" — guessing a split so
    /// staging looks cheaper — would under-predict toward an OOM.
    #[test]
    fn a_staged_composition_promises_no_saving_without_a_measured_split() {
        let weights = crate::estimate_synthesis::ResidentWeights {
            conditioning_bytes: 0,
            heavy_bytes: 20 * GIB,
            transformer_bytes: 0,
        };
        let resident = crate::estimate_synthesis::compose_resident_floor_bytes(
            &weights,
            CONDITIONING_ENGAGED,
            [(7 * GIB, None)],
        );
        let staged = crate::estimate_synthesis::compose_resident_floor_bytes(
            &weights,
            &[
                gen_core::MemoryStrategy::Resident,
                gen_core::MemoryStrategy::StagedResidency,
            ],
            [(7 * GIB, None)],
        );
        assert_eq!(
            staged, resident,
            "an unmeasured conditioning split must not become a staging saving"
        );
    }

    /// Unmeasurable weights ⇒ no signal ⇒ `None`, and the caller admits. The never-block-without-
    /// evidence posture, stated as a test. A single byte on EITHER side is enough signal to gate on.
    #[test]
    fn no_weights_means_no_signal() {
        assert_eq!(conditioning_floor_gb(&footprint(0, 0)), None);
        assert!(conditioning_floor_gb(&footprint(1, 0)).is_some());
        assert!(conditioning_floor_gb(&footprint(0, 1)).is_some());
    }

    // ── The decision ─────────────────────────────────────────────────────────────────────────────

    /// Every no-signal path ADMITS, and each is distinguishable so the caller can log which evidence
    /// was missing. A mutation that rejects on missing evidence fails here.
    #[test]
    fn missing_evidence_always_admits_and_says_which_is_missing() {
        // No weights, even against a hopeless 0 GB budget.
        assert_eq!(
            decide(&footprint(0, 0), budget(0.0)),
            ConditioningFit::NoWeightEvidence
        );
        assert!(admit(
            &decide(&footprint(0, 0), budget(0.0)),
            &footprint(0, 0),
            "0"
        )
        .is_ok());
        // Weights, but no budget: a 900 GB floor still admits.
        let huge = footprint(900 * GIB, 0);
        assert!(matches!(
            decide(&huge, None),
            ConditioningFit::NoBudget { .. }
        ));
        assert!(admit(&decide(&huge, None), &huge, "0").is_ok());
    }

    /// A machine that can hold the weights is never blocked — including exactly AT the boundary, which
    /// is where an off-by-one comparison would false-reject. The floor for (20 + 7) GiB is 29.0 GB.
    #[test]
    fn a_machine_that_fits_the_weights_is_admitted() {
        let f = footprint(20 * GIB, 7 * GIB);
        for free in [29.0, 29.1, 40.0, 96.0] {
            let fit = decide(&f, budget(free));
            assert!(
                matches!(fit, ConditioningFit::Fits { .. }),
                "{free} GB free must admit a 29 GB floor, got {fit:?}"
            );
            assert!(admit(&fit, &f, "0").is_ok());
        }
    }

    /// Below the floor the weights alone cannot fit, so the render is refused BEFORE it allocates.
    /// This is the defect sc-16069 fixes, stated as a test: before this module the same request
    /// allocated and died on a reactive CUDA OOM.
    #[test]
    fn weights_that_cannot_fit_are_rejected_before_allocating() {
        let f = footprint(20 * GIB, 7 * GIB);
        let fit = decide(&f, budget(24.0));
        assert_eq!(
            fit,
            ConditioningFit::TooBig {
                floor_gb: 29.0,
                available_gb: 24.0
            }
        );
        let error = admit(&fit, &f, "0").expect_err("must reject");
        let message = error.to_string();
        // Actionable, house-convention: names the model, the overlay, the need, the machine, the levers.
        assert!(message.contains("qwen_image"), "{message}");
        assert!(
            message.contains("strict-pose ControlNet branch"),
            "{message}"
        );
        assert!(message.contains("29.0 GB"), "{message}");
        assert!(message.contains("24.0 GB"), "{message}");
        assert!(message.contains("GPU 0"), "{message}");
        assert!(message.contains("smaller quant tier"), "{message}");
        // Declared a FLOOR, never a measured peak — the message must not overclaim what is unmeasured.
        assert!(
            message.contains("at least"),
            "the rejection must present the figure as a floor: {message}"
        );
        assert!(
            message.contains("weights alone"),
            "the rejection must say the transients are not included: {message}"
        );
    }

    /// The overlay is priced in the message, at a precision that can actually show it. An IP-Adapter is
    /// ~1 GB against a ~20 GB base, so whole-GB rounding would print "0 GB" and read as a bug.
    #[test]
    fn the_message_prices_a_small_overlay_visibly() {
        let f = ConditioningFootprint {
            model_label: "sdxl".to_owned(),
            overlay_label: "IP-Adapter",
            base_bytes: 6 * GIB,
            // ~0.8 GiB — the real order of magnitude for `ip-adapter-plus_sdxl_vit-h.safetensors`.
            overlay_bytes: 858_993_459,
        };
        let error = admit(&decide(&f, budget(1.0)), &f, "0").expect_err("must reject");
        let message = error.to_string();
        assert!(message.contains("0.8 GB IP-Adapter"), "{message}");
        assert!(!message.contains("0.0 GB IP-Adapter"), "{message}");
    }

    // ── The on-disk scan ─────────────────────────────────────────────────────────────────────────

    /// A base DIRECTORY is summed recursively over `.safetensors`; an overlay FILE is measured by its
    /// own length, NOT by scanning its parent — the packed overlays live as `<tier>/model.safetensors`
    /// siblings, so a parent scan would sum every tier and over-count several-fold.
    #[test]
    fn from_paths_sums_a_base_dir_and_measures_an_overlay_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let base = tmp.path().join("base");
        std::fs::create_dir_all(base.join("transformer")).expect("mkdir");
        std::fs::write(
            base.join("transformer").join("a.safetensors"),
            vec![0u8; 2048],
        )
        .expect("write");
        std::fs::write(base.join("b.safetensors"), vec![0u8; 1024]).expect("write");
        // Not weights — must be ignored.
        std::fs::write(base.join("config.json"), b"{}").expect("write");

        // Two sibling overlay tiers; only the selected one may be counted.
        let overlays = tmp.path().join("overlay");
        std::fs::create_dir_all(overlays.join("q8")).expect("mkdir");
        std::fs::create_dir_all(overlays.join("bf16")).expect("mkdir");
        let selected = overlays.join("q8").join("model.safetensors");
        std::fs::write(&selected, vec![0u8; 512]).expect("write");
        std::fs::write(
            overlays.join("bf16").join("model.safetensors"),
            vec![0u8; 4096],
        )
        .expect("write");

        let f = ConditioningFootprint::from_paths("qwen_image", "overlay", &base, &[&selected]);
        assert_eq!(f.base_bytes, 3072, "base dir sums its safetensors only");
        assert_eq!(
            f.overlay_bytes, 512,
            "the overlay FILE is measured alone — a parent scan would have summed the bf16 sibling"
        );
    }

    /// A missing base / overlay path contributes 0 rather than erroring, which lands on
    /// [`ConditioningFit::NoWeightEvidence`] ⇒ admit. Less evidence must never become a phantom
    /// requirement.
    #[test]
    fn from_paths_treats_missing_paths_as_no_signal() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let missing_base = tmp.path().join("nope");
        let missing_overlay = tmp.path().join("also-nope.safetensors");
        let f = ConditioningFootprint::from_paths(
            "qwen_image",
            "overlay",
            &missing_base,
            &[&missing_overlay],
        );
        assert_eq!(f.base_bytes, 0);
        assert_eq!(f.overlay_bytes, 0);
        assert_eq!(decide(&f, budget(0.0)), ConditioningFit::NoWeightEvidence);
    }

    /// A path declared BOTH as a file and inside a declared directory is counted ONCE (sc-16069 review).
    ///
    /// This is the real PuLID shape: `ensure_pulid_candle_weights` returns `(adapter, eva, bundle)`, and
    /// on the download-on-first-use path `resolve_one` puts `adapter`/`eva` INSIDE `bundle` — which is
    /// also the provider's `face_dir`. Summing all three naively counted ~2 GB twice, an OVER-count, the
    /// one direction that can refuse a render which fits.
    ///
    /// MUTATION PROOF: removing the containment dedupe doubles the two nested files and this goes red.
    #[test]
    fn nested_overlay_paths_are_counted_once() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let base = tmp.path().join("base");
        std::fs::create_dir_all(&base).expect("mkdir");
        std::fs::write(base.join("m.safetensors"), vec![0u8; 4096]).expect("write");

        // The PuLID bundle: face-stack files staged in it, plus the adapter + EVA downloaded INTO it.
        let bundle = tmp.path().join("pulid-flux");
        std::fs::create_dir_all(&bundle).expect("mkdir");
        let adapter = bundle.join("pulid_flux_v0.9.1.safetensors");
        let eva = bundle.join("eva02_clip_l_336.safetensors");
        std::fs::write(&adapter, vec![0u8; 1024]).expect("write");
        std::fs::write(&eva, vec![0u8; 512]).expect("write");
        std::fs::write(bundle.join("scrfd_10g.safetensors"), vec![0u8; 256]).expect("write");

        let nested = ConditioningFootprint::from_paths(
            "PuLID-FLUX",
            "face-identity encoder stack",
            &base,
            &[adapter.as_path(), eva.as_path(), bundle.as_path()],
        );
        assert_eq!(
            nested.overlay_bytes, 1792,
            "the bundle scan (1024 + 512 + 256) must be counted once, not once per nested declaration"
        );

        // The WARM install: the same two files resolve to a whole-repo snapshot OUTSIDE the bundle, so
        // they are NOT nested and MUST still be counted. Containment — not omission at the call site —
        // is what gets both installs right.
        let snapshot = tmp.path().join("snapshot");
        std::fs::create_dir_all(&snapshot).expect("mkdir");
        let far_adapter = snapshot.join("pulid_flux_v0.9.1.safetensors");
        std::fs::write(&far_adapter, vec![0u8; 1024]).expect("write");
        let separate = ConditioningFootprint::from_paths(
            "PuLID-FLUX",
            "face-identity encoder stack",
            &base,
            &[far_adapter.as_path(), bundle.as_path()],
        );
        assert_eq!(
            separate.overlay_bytes,
            1024 + 1792,
            "a path outside the declared directory is a real, separate resident cost"
        );

        // An overlay inside the BASE dir is likewise already counted by the base scan.
        let in_base = base.join("m.safetensors");
        let overlapping =
            ConditioningFootprint::from_paths("PuLID-FLUX", "overlay", &base, &[in_base.as_path()]);
        assert_eq!(overlapping.base_bytes, 4096);
        assert_eq!(overlapping.overlay_bytes, 0);

        // A sibling whose NAME merely shares a prefix is NOT nested — `Path::starts_with` is
        // component-wise, so a lexical string prefix must not silently drop a real cost.
        let sibling = tmp.path().join("pulid-flux-extra.safetensors");
        std::fs::write(&sibling, vec![0u8; 64]).expect("write");
        let prefixed = ConditioningFootprint::from_paths(
            "PuLID-FLUX",
            "overlay",
            &base,
            &[bundle.as_path(), sibling.as_path()],
        );
        assert_eq!(prefixed.overlay_bytes, 1792 + 64);
    }

    /// Several overlays (a control branch plus its own encoder, say) all count. A mutation that reads
    /// only the first path loses the rest and turns this red.
    #[test]
    fn from_paths_sums_every_overlay_path() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let base = tmp.path().join("base");
        std::fs::create_dir_all(&base).expect("mkdir");
        std::fs::write(base.join("m.safetensors"), vec![0u8; 1024]).expect("write");
        let one = tmp.path().join("one.safetensors");
        let two = tmp.path().join("two.safetensors");
        std::fs::write(&one, vec![0u8; 256]).expect("write");
        std::fs::write(&two, vec![0u8; 768]).expect("write");
        let f = ConditioningFootprint::from_paths("m", "overlay", &base, &[&one, &two]);
        assert_eq!(f.overlay_bytes, 1024);
    }
}
