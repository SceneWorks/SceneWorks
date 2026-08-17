//! Request-scoped execution planning (sc-18317, epic 18304 P2).
//!
//! Two decisions live here, both backend-neutral and both pure:
//!
//! 1. **Warm-hit execution-policy switching.** The generator cache is single-resident and keyed by
//!    `generator_cache::LoadIdentity`, which deliberately excludes residency and materialization
//!    policy (sc-18305): changing a policy must not force the same weights to reload. That split left
//!    an obligation this module discharges — a warm hit whose request asks for a *different*
//!    [`ExecutionPolicy`] than the resident generator was loaded under used to be served under the
//!    cold-load policy with nothing but a `tracing::warn`, so no caller could tell a granted switch
//!    from a silently substituted one. [`decide_warm_policy`] turns that into a real, fail-closed
//!    decision, [`WarmPolicyProposal`] carries it to the request-scoped planner that can act on it,
//!    and the planner floors the memory ladder's candidate set so a granted switch actually changes
//!    what runs. `WarmPolicyProposal::requires_staged_residency` is the seam the planner reads.
//!
//! 2. **Typed execution-domain selection.** gen-core carries [`GraphEvalCadence`] / [`FfnChunk`] /
//!    [`CfgBatching`] on [`GenerationMemory`], declared per provider through
//!    `Capabilities::execution` and refused fail-closed at `Capabilities::validate_request`.
//!    [`select_execution_domains`] chooses values ONLY from what a provider declares, so the worker
//!    can never trip that refusal.
//!
//! ## Why the warm decision is shaped by memory monotonicity rather than by a cost model
//!
//! gen-core's residency driver (`gen_core::residency`) is already request-scoped: the same loaded
//! generator serves both unstaged and staged requests, evicting and rebuilding its warm component
//! pair when the request's shape differs (`Residency::ensure_warm_locked` drops the pair whenever
//! `pair.streamable != streamable`, and `run_request_scoped` runs text before heavy without
//! overlapping their lifetimes when `stage_residency` is set). A policy switch therefore costs
//! *component* re-materialization inside the loaded generator, never a new generator and never a
//! cache miss — so there is no reload-versus-free tradeoff for this module to price, and inference
//! exposes no transition-cost query for one either (`mlx-gen/src/residency.rs` is a 116-line adapter
//! whose only cost statement is a unit test's allocator-flush count).
//!
//! What a switch does change is the request's PEAK, and admission was proved against the resident
//! generator's loaded shape. Staged residency and deferred materialization only ever LOWER the peak,
//! so honoring a request that asks for *more* of either stays inside the admitted envelope. The
//! reverse — a staged load asked to run fully resident — would execute above what admission proved,
//! so it is served as loaded. That asymmetry, not a cost model, is the decision.
//!
//! ## Why a grant needs TWO signals, and why neither is enough alone
//!
//! `gen_core::Residency::resident(text, heavy)` has no reload loaders (`rebuildable == false`), and
//! any request that asks it to stage or re-materialize returns
//! `Error::Unsupported("resident-only component source cannot stage or rematerialize components …")`.
//! Granting such a switch would fail the job with a capability error part-way through generation, so
//! the switch must be refused before the provider is asked. Two independent things can put an
//! instance in that state, and they need separate questions:
//!
//! - The SOURCE is a single file. [`SourceReopenability`], read off the [`LoadSpec`]. This is the
//!   "single-file interlock" the story names — treat imported checkpoints as stream-ineligible, fail
//!   closed, never assume — and it is deliberately conservative: a prepared pin does NOT lift it,
//!   because at the pinned revision `mlx-gen-sdxl`'s fused-LDM Resident load obtains a valid pin and
//!   then discards it along with any means of using it.
//! - The INSTANCE declares rung 1 unimplemented, or publishes no contract at all. That is
//!   [`StagingAttestation`], read off the loaded generator's own `MemoryProviderContract`.
//!
//! A grant requires both to admit it, and each refusal names its own cause so the event never blames
//! the source for a declaration decision or the reverse.
//!
//! Neither signal is a substitute for the other, and — verified, not assumed — the attestation cannot
//! see the sdxl case: that instance declares rung 1 `Implemented` and passes `validate_selection`. The
//! source proxy is what stops it. `StagingAttestation`'s docs record that gap and name the upstream
//! fix; do not "simplify" this to one signal on the strength of the contract looking authoritative.
//!
//! ## Why a grant is a proposal until the selection confirms it
//!
//! The cache seam knows a switch is safe and performable. It does NOT know whether honoring it changes
//! anything, because per-request staging is not chosen by this policy at all — it is chosen by the
//! memory ladder in [`crate::mlx_fit_gate`], which picks the first fitting candidate in
//! resident → staged → bounded-decode → … order. A policy the ladder never reads cannot move a
//! request, and the first version of this module emitted `rematerialized` from the seam while the
//! request went on to run fully resident.
//!
//! So the grant is threaded into the planner as a request-scoped FLOOR on the candidate set: when a
//! grant asks for more staging than the load chose, only candidates that engage
//! `MemoryStrategy::StagedResidency` are offered to the selector. The selector applies the same budget
//! and margins it always does, so a floored selection that comes back `Selected` has passed exactly
//! the same fit check — monotonicity is preserved by construction rather than asserted. The event is
//! then emitted by [`WarmPolicyProposal::settle_with_selection`] from the outcome the selector
//! actually produced, so `rematerialized` means the selection moved and nothing else.

use gen_core::{
    CfgBatching, CfgBatchingDomain, ExecutionSurface, ExecutionValueDomain, FfnChunk,
    GenerationMemory, GraphEvalCadence, LoadShape, LoadSpec, OffloadPolicy, WeightsSource,
};

use crate::generator_cache::ExecutionPolicy;

/// Whether the LOADED generator's own contract declares staged residency implemented.
///
/// A per-instance signal — the contract is built from the spec the generator was actually loaded with
/// — and it catches a provider that declares rung 1 `Missing` or publishes no contract at all. Both
/// absent cases are `NotImplemented`, so it is fail-closed in its own right.
///
/// **What it CANNOT detect, verified against the pinned revision.** Rebuildability is a property of
/// the `gen_core::Residency` the provider CONSTRUCTED, and no published surface reports it. At
/// 717f43b5 exactly one production site builds the non-rebuildable `Residency::resident`:
/// `mlx-gen-sdxl`'s `load_from_ldm_file` under `OffloadPolicy::Resident`, which reads through a
/// perfectly valid file pin and then drops it, installing error-returning loaders. That instance
/// nonetheless reports `staged_residency_availability() == Selectable` from its static descriptor,
/// declares rung 1 `Implemented` through this contract (its builder derives only `load_shape` and
/// rung 4 from the spec), and passes `validate_selection` — rung 1 declares no prerequisite, and
/// `MemoryStrategyPrerequisite` has no variant that could express "this load retained reload
/// loaders". The failure surfaces only inside `generate`, from `Residency::ensure_rebuildable`, as
/// `Error::Unsupported`.
///
/// So this attestation is necessary but NOT sufficient, and it is deliberately not the thing standing
/// between a grant and that crash. [`SourceReopenability`] is, and it is conservative for exactly the
/// shape that triggers it. Closing the gap properly is an upstream change (sdxl declaring rung 1
/// `Missing` for `File` + `Resident`, mirroring what it already does for rung 4, or retaining the pin
/// and using `from_policy_with_resident`); until then the worker must not trust the declaration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StagingAttestation {
    Implemented,
    NotImplemented,
}

impl StagingAttestation {
    /// Read the attestation off the loaded generator's contract.
    pub(crate) fn of_contract(contract: Option<&gen_core::MemoryProviderContract>) -> Self {
        let implemented = contract
            .and_then(|contract| contract.capability(gen_core::MemoryStrategy::StagedResidency))
            .is_some_and(|capability| {
                matches!(
                    capability.support,
                    gen_core::MemoryStrategySupport::Implemented
                )
            });
        if implemented {
            Self::Implemented
        } else {
            Self::NotImplemented
        }
    }

    pub(crate) fn of_generator(generator: &dyn gen_core::Generator) -> Self {
        Self::of_contract(generator.memory_strategy_contract())
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Implemented => "staging_implemented",
            Self::NotImplemented => "staging_not_implemented",
        }
    }
}

/// Whether the weights behind a resident generator can be re-opened for component
/// re-materialization.
///
/// Derived from the [`LoadSpec`] the generator was loaded from, never from the engine id: one engine
/// serves both a snapshot directory and an imported single file.
///
/// **A single-file base source is classified NOT re-openable even when it carries a valid prepared
/// pin, and that is deliberate rather than an oversight.** A pin proves the FILE can be re-read under
/// a stable identity; it does not prove the PROVIDER retained loaders that would use it. At the pinned
/// revision the one provider that accepts a `WeightsSource::File` base under
/// `OffloadPolicy::Resident` -- `mlx-gen-sdxl`'s fused-LDM path -- obtains exactly such a pin, reads
/// through it once, drops it, and installs `Residency::resident`'s error-returning loaders. Trusting
/// the pin there would hand that instance a staging request and fail the job mid-generation with
/// `Error::Unsupported`, because nothing between here and the tensor code refuses it (see
/// [`StagingAttestation`] for why the contract does not).
///
/// So this is the conservative half of the interlock and the half that actually holds the line: `Dir`
/// re-opens, every `File` does not. It costs a grant on some future single-file provider that does
/// retain its loaders, which is the right way round -- a missed optimization, never a failed render.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SourceReopenability {
    /// A snapshot directory. Every provider that takes one retains reload loaders at the pinned
    /// revision.
    Reopenable,
    /// Any single-file / imported base source, pinned or not. The one provider that loads this shape
    /// resident holds it resident-only and cannot rebuild a component it drops.
    SingleFileNotReopenable,
}

impl SourceReopenability {
    /// Classify the base weights source of `spec`.
    ///
    /// Only `spec.weights` is consulted: it is the component the residency driver stages and
    /// re-materializes. A companion overlay (control, adapter, external text encoder) that happens to
    /// be a bare file does not make the base transformer un-reopenable, and treating it as if it did
    /// would refuse staging for every LoRA request against a snapshot directory.
    pub(crate) fn of_spec(spec: &LoadSpec) -> Self {
        match &spec.weights {
            WeightsSource::Dir(_) => Self::Reopenable,
            // Prepared or not. See the type docs: a pin proves the file re-reads, not that the
            // provider kept a loader to re-read it with, and the one provider taking this shape under
            // Resident does not.
            WeightsSource::File(_) => Self::SingleFileNotReopenable,
        }
    }

    pub(crate) const fn is_reopenable(self) -> bool {
        matches!(self, Self::Reopenable)
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Reopenable => "reopenable",
            Self::SingleFileNotReopenable => "single_file_not_reopenable",
        }
    }
}

/// What the worker decided about a warm hit whose requested policy differs from the loaded one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WarmPolicyDecision {
    /// The requested and loaded policies are identical. No decision, no event.
    Unchanged,
    /// The resident generator keeps running under its loaded policy. The request's own intent is
    /// recorded but not executed.
    ServedAsIs(ServedAsIsReason),
    /// The requested policy is granted: the provider re-materializes components for this request
    /// inside the already-loaded generator. No new generator is constructed and the cache entry is
    /// untouched.
    Rematerialized,
    /// A switch the resident source cannot perform. Refused explicitly and served under the loaded
    /// policy — never sent to the provider, which would answer `Error::Unsupported` mid-generation.
    RefusedSwitch(RefusalReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ServedAsIsReason {
    /// Only `gen_core::LoadShapeDeclarationResult` differs. That is a declaration-authority marker,
    /// not an execution shape, so nothing about the run changes.
    DeclarationAuthorityOnly,
    /// The request asks to run less staged / more resident than the load chose. Admission was proved
    /// against the loaded shape and the requested shape peaks above it.
    LoadedPolicyBoundsThePeak,
    /// The LOADED generator's own contract does not implement staged residency, so `Sequential` is
    /// advisory-only for this instance and granting the switch would change nothing it honors. Read
    /// from the instance's `MemoryProviderContract`, not from the engine's static descriptor bit —
    /// see [`StagingAttestation`].
    LoadedContractDoesNotImplementStaging,
    /// The grant survived the decision but the request-scoped ladder had no candidate that engages
    /// staged residency, so honoring it would have changed nothing about the selection.
    NoStagedCandidateForThisRequest,
    /// The ladder's staged candidate did not fit this request's budget, so the baseline selection
    /// stands. Not a refusal: the fit check is the same one every selection passes.
    StagedCandidateDidNotFit,
    /// The grant survived and a staged candidate existed, but the selection it produced is the one the
    /// baseline had already chosen — the request was ALREADY going to stage.
    SelectionAlreadyStaged,
    /// This route assembles its provider request without a request-scoped `GenerationMemory`, so it
    /// has no seam through which a policy switch could take effect.
    RouteHasNoRequestScopedMemory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RefusalReason {
    /// The resident weights are a bare single-file/imported source. Staging or deferred
    /// materialization would return `Error::Unsupported` from the provider.
    SourceNotReopenable,
    /// The switch tightens ONLY `LoadShape`, and the worker has no seam that executes that axis
    /// per-request.
    ///
    /// `OffloadPolicy` reaches the provider through the memory ladder's `stage_residency` selection,
    /// which is what [`WarmPolicyProposal::requires_staged_residency`] floors. `LoadShape` has no such
    /// request-scoped selector: it is fixed at load time and the ladder has no rung that toggles it. So
    /// granting a load-shape-only switch would report a re-materialization while the deferred half
    /// executed nowhere. Refused with this reason until an execution seam for the axis exists, rather
    /// than reported as a half-performed switch.
    NoExecutionSeamForLoadShapeAlone,
}

impl WarmPolicyDecision {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Unchanged => "unchanged",
            Self::ServedAsIs(_) => "served_as_is",
            Self::Rematerialized => "rematerialized",
            Self::RefusedSwitch(_) => "refused_switch",
        }
    }

    pub(crate) const fn reason(self) -> &'static str {
        match self {
            Self::Unchanged => "policies_match",
            Self::ServedAsIs(ServedAsIsReason::DeclarationAuthorityOnly) => {
                "declaration_authority_only"
            }
            Self::ServedAsIs(ServedAsIsReason::LoadedPolicyBoundsThePeak) => {
                "loaded_policy_bounds_the_peak"
            }
            Self::ServedAsIs(ServedAsIsReason::LoadedContractDoesNotImplementStaging) => {
                "loaded_contract_does_not_implement_staging"
            }
            Self::ServedAsIs(ServedAsIsReason::NoStagedCandidateForThisRequest) => {
                "no_staged_candidate_for_this_request"
            }
            Self::ServedAsIs(ServedAsIsReason::StagedCandidateDidNotFit) => {
                "staged_candidate_did_not_fit"
            }
            Self::ServedAsIs(ServedAsIsReason::SelectionAlreadyStaged) => {
                "selection_already_staged"
            }
            Self::ServedAsIs(ServedAsIsReason::RouteHasNoRequestScopedMemory) => {
                "route_has_no_request_scoped_memory"
            }
            Self::Rematerialized => "components_rematerialized_in_place",
            Self::RefusedSwitch(RefusalReason::SourceNotReopenable) => "source_not_reopenable",
            Self::RefusedSwitch(RefusalReason::NoExecutionSeamForLoadShapeAlone) => {
                "no_execution_seam_for_load_shape_alone"
            }
        }
    }

    /// Whether the request may execute under its own requested policy.
    pub(crate) const fn grants_requested_policy(self) -> bool {
        matches!(self, Self::Rematerialized)
    }
}

/// Rank a policy by how tightly it bounds peak memory. Higher is cheaper at peak.
///
/// `Sequential` releases whole components between phases; `DeferredMaterialization` re-opens
/// transformer blocks instead of holding a bulk resident stack. Both are reductions and they compose,
/// so this sum is an honest partial order for "is the requested shape inside the admitted envelope".
/// It deliberately does not rank the two levers against each other: a sideways trade proves nothing.
const fn peak_containment_rank(policy: ExecutionPolicy) -> u8 {
    let staged = match policy.offload_policy {
        OffloadPolicy::Sequential => 1,
        OffloadPolicy::Resident => 0,
    };
    let deferred = match policy.load_shape {
        LoadShape::DeferredMaterialization => 1,
        LoadShape::EagerMaterialization => 0,
    };
    staged + deferred
}

/// Whether two policies differ in anything the provider actually executes.
///
/// `load_shape_declaration_result` is excluded on purpose: it records who decided the shape, not what
/// runs.
fn differs_in_execution(loaded: ExecutionPolicy, requested: ExecutionPolicy) -> bool {
    loaded.offload_policy != requested.offload_policy || loaded.load_shape != requested.load_shape
}

/// Decide what a warm hit does with a requested policy that differs from the loaded one.
///
/// A `Rematerialized` result here is a PROPOSAL, not a fact: it says the switch is safe and the
/// instance can perform it. Only the request-scoped selection can confirm that honoring it actually
/// changes what runs, which is why the event is emitted by
/// [`WarmPolicyProposal::settle`] and not here. Pure.
///
/// Both `reopenability` (a property of the SOURCE) and `attestation` (a property of the loaded
/// INSTANCE) must admit the switch. They answer different questions and either one alone is
/// insufficient — see the doc comments on both types.
pub(crate) fn decide_warm_policy(
    loaded: ExecutionPolicy,
    requested: ExecutionPolicy,
    reopenability: SourceReopenability,
    attestation: StagingAttestation,
) -> WarmPolicyDecision {
    if loaded == requested {
        return WarmPolicyDecision::Unchanged;
    }
    if !differs_in_execution(loaded, requested) {
        return WarmPolicyDecision::ServedAsIs(ServedAsIsReason::DeclarationAuthorityOnly);
    }
    if peak_containment_rank(requested) <= peak_containment_rank(loaded) {
        return WarmPolicyDecision::ServedAsIs(ServedAsIsReason::LoadedPolicyBoundsThePeak);
    }
    // From here the request asks for a strictly tighter shape than the load chose, which is inside
    // the admitted envelope.
    //
    // First: is there a seam that EXECUTES the tightening it asks for? Only the `OffloadPolicy` half
    // has one (the ladder's staged-residency selection). A switch that tightens `LoadShape` alone has
    // nowhere to take effect, so granting it would report a re-materialization that never happened.
    if requested.offload_policy == loaded.offload_policy {
        return WarmPolicyDecision::RefusedSwitch(RefusalReason::NoExecutionSeamForLoadShapeAlone);
    }
    // Then: can the resident instance deliver it? Answered before the provider is asked rather than by
    // letting the provider refuse mid-generation.
    if !reopenability.is_reopenable() {
        return WarmPolicyDecision::RefusedSwitch(RefusalReason::SourceNotReopenable);
    }
    if attestation == StagingAttestation::NotImplemented {
        // The source can reopen but THIS instance's contract cannot stage — the sdxl-loaded-Resident
        // shape. Reporting it as a source refusal would blame the wrong thing, so it is a truthful
        // served-as-is and the grant dies here.
        return WarmPolicyDecision::ServedAsIs(
            ServedAsIsReason::LoadedContractDoesNotImplementStaging,
        );
    }
    WarmPolicyDecision::Rematerialized
}

/// A warm-hit policy decision travelling from the cache seam to the request-scoped planner that can
/// act on it.
///
/// This type exists because a grant is only half a fact at the cache seam. The seam knows the switch
/// is SAFE (monotone at peak) and PERFORMABLE (source re-openable, instance attests staging); it does
/// not know whether honoring it changes anything, because per-request staging is chosen by the memory
/// ladder from the candidate set, not by this policy. Emitting `rematerialized` at the seam therefore
/// claimed a re-materialization that the ladder might never perform.
///
/// So the seam produces a proposal and the consumer settles it exactly once:
///
/// - [`Self::settle_with_selection`] — the MLX request planner, which floored the candidate set with
///   the grant and can say whether the selection actually moved.
/// - [`Self::decline`] — a route with no request-scoped `GenerationMemory` seam, which downgrades a
///   grant to a truthful served-as-is rather than reporting a switch it cannot perform.
///
/// Marked `#[must_use]` so a consumer that neither settles nor declines is a compiler warning rather
/// than a silently dropped observation.
#[must_use = "a warm-policy proposal must be settled or declined, or the decision goes unreported"]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WarmPolicyProposal {
    engine_id: &'static str,
    decision: WarmPolicyDecision,
    loaded: ExecutionPolicy,
    requested: ExecutionPolicy,
    reopenability: SourceReopenability,
    attestation: StagingAttestation,
    /// Whether settling this copy emits the event.
    ///
    /// A multi-item job evaluates one request per image and every one of them must be floored by the
    /// grant — the decision is about the resident generator, so it holds for the whole job. But the
    /// EVENT describes one cache access, so only the first copy reports. Separating the two is what
    /// lets `WarmPolicyOnce` silence the duplicates without also silencing the optimization.
    report: bool,
}

/// One-shot holder for a multi-item job's warm-policy proposal.
///
/// A job renders N images by calling the request planner once per seed or pose, and the proposal is
/// `Copy`, so a naive lane settles it N times and emits N identical events for ONE warm hit. The
/// decision is a property of the cache access, not of the item, so it must be settled exactly once.
///
/// [`Self::take`] yields the same decision to EVERY item — a grant floors the ladder for the whole job,
/// because it is a fact about the resident generator rather than about one image — but only the first
/// copy reports. Rationing the decision itself would have been the wrong fix: items 2..N would have
/// silently lost the staging the request asked for and run at the higher peak the switch existed to
/// avoid.
pub(crate) struct WarmPolicyOnce {
    engine_id: &'static str,
    /// The reporting copy, until it is handed out.
    proposal: Option<WarmPolicyProposal>,
    /// The silent copy handed to every later item.
    silenced: Option<WarmPolicyProposal>,
}

impl WarmPolicyOnce {
    pub(crate) fn new(proposal: WarmPolicyProposal) -> Self {
        Self {
            engine_id: proposal.engine_id,
            proposal: Some(proposal),
            silenced: None,
        }
    }

    /// The decision, reporting on the first call and silent on every call after it.
    pub(crate) fn take(&mut self) -> WarmPolicyProposal {
        match self.proposal.take() {
            Some(proposal) => {
                // Keep the decision for later items; drop only its right to log again.
                self.silenced = Some(proposal.silenced());
                proposal
            }
            None => self
                .silenced
                .unwrap_or_else(|| WarmPolicyProposal::inert(self.engine_id).silenced()),
        }
    }

    /// Settle the proposal without evaluating any request, for a route that turns out to have no
    /// request-scoped memory plan at all. A no-op once [`Self::take`] has handed it out.
    ///
    /// A holder that is simply DROPPED unsettled reports nothing, and that is the intended outcome for
    /// a job cancelled before its first evaluation: no request ran, so there is no decision to report.
    /// The alternative — logging a policy decision for work that never happened — would be noise.
    pub(crate) fn decline_if_unsettled(&mut self, reason: ServedAsIsReason) {
        if let Some(proposal) = self.proposal.take() {
            proposal.decline(reason);
        }
    }
}

/// What the request-scoped selection did with a granted proposal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GrantOutcome {
    /// The grant changed the selection: the request now engages staged residency where the baseline
    /// candidate set would have run it resident. This is the only outcome that reports
    /// `rematerialized`.
    SelectionMovedToStaged,
    /// No candidate in this request's ladder engages staged residency.
    NoStagedCandidate,
    /// A staged candidate existed but did not fit the budget, so the baseline selection stands.
    StagedCandidateDidNotFit,
    /// The baseline selection already engaged staged residency, so the floor was a no-op — the request
    /// was going to stage anyway.
    AlreadyStaged,
}

impl WarmPolicyProposal {
    /// The proposal for one warm (or cold) access.
    pub(crate) fn new(
        engine_id: &'static str,
        decision: WarmPolicyDecision,
        loaded: ExecutionPolicy,
        requested: ExecutionPolicy,
        reopenability: SourceReopenability,
        attestation: StagingAttestation,
    ) -> Self {
        Self {
            engine_id,
            decision,
            loaded,
            requested,
            reopenability,
            attestation,
            report: true,
        }
    }

    /// The same decision, settling silently. See the `report` field.
    fn silenced(self) -> Self {
        Self {
            report: false,
            ..self
        }
    }

    /// A proposal with nothing to decide, for a route or test seam whose execution policy never
    /// varies from the loaded one. Settling it is silent.
    pub(crate) fn inert(engine_id: &'static str) -> Self {
        let policy = ExecutionPolicy {
            offload_policy: OffloadPolicy::Resident,
            load_shape: LoadShape::EagerMaterialization,
            load_shape_declaration_result: gen_core::LoadShapeDeclarationResult::NotEvaluated,
        };
        Self::new(
            engine_id,
            WarmPolicyDecision::Unchanged,
            policy,
            policy,
            SourceReopenability::Reopenable,
            StagingAttestation::Implemented,
        )
    }

    /// The policy this request executes under. For a granted proposal this is the tighter requested
    /// policy; otherwise the loaded one.
    ///
    /// This is an EXECUTION intent, never an admission input: admission reads the separate
    /// `loaded_policy` slot the cache seam passes alongside this proposal, because that names the shape
    /// the resident weights are actually in.
    #[cfg(test)]
    pub(crate) fn effective_policy(self) -> ExecutionPolicy {
        if self.decision.grants_requested_policy() {
            self.requested
        } else {
            self.loaded
        }
    }

    /// Whether the request-scoped ladder must be floored to candidates that engage staged residency.
    ///
    /// Equivalent to "this proposal was granted": [`decide_warm_policy`] refuses a switch that does not
    /// move the `OffloadPolicy` axis (`NoExecutionSeamForLoadShapeAlone`), precisely because the floor
    /// is the only seam a grant executes through. The policy comparison is kept as a debug assertion so
    /// the two cannot drift apart silently.
    pub(crate) fn requires_staged_residency(self) -> bool {
        let granted = self.decision.grants_requested_policy();
        debug_assert!(
            !granted
                || (self.requested.offload_policy == OffloadPolicy::Sequential
                    && self.loaded.offload_policy == OffloadPolicy::Resident),
            "a granted proposal must move the staging axis: {self:?}"
        );
        granted
    }

    /// Settle a proposal against what the request-scoped selection actually did.
    ///
    /// `outcome` is ignored for a non-granted proposal: its decision was already final at the seam.
    pub(crate) fn settle_with_selection(self, outcome: GrantOutcome) {
        let decision = if self.decision.grants_requested_policy() {
            match outcome {
                GrantOutcome::SelectionMovedToStaged => WarmPolicyDecision::Rematerialized,
                GrantOutcome::NoStagedCandidate => WarmPolicyDecision::ServedAsIs(
                    ServedAsIsReason::NoStagedCandidateForThisRequest,
                ),
                GrantOutcome::StagedCandidateDidNotFit => {
                    WarmPolicyDecision::ServedAsIs(ServedAsIsReason::StagedCandidateDidNotFit)
                }
                GrantOutcome::AlreadyStaged => {
                    WarmPolicyDecision::ServedAsIs(ServedAsIsReason::SelectionAlreadyStaged)
                }
            }
        } else {
            self.decision
        };
        self.log(decision);
    }

    /// Settle a proposal on a route that has no request-scoped memory seam to honor it through.
    pub(crate) fn decline(self, reason: ServedAsIsReason) {
        let decision = if self.decision.grants_requested_policy() {
            WarmPolicyDecision::ServedAsIs(reason)
        } else {
            self.decision
        };
        self.log(decision);
    }

    fn log(self, decision: WarmPolicyDecision) {
        if !self.report {
            return;
        }
        log_warm_policy_decision(
            self.engine_id,
            decision,
            self.loaded,
            self.requested,
            self.reopenability,
            self.attestation,
        );
    }

    #[cfg(test)]
    pub(crate) fn decision(self) -> WarmPolicyDecision {
        self.decision
    }

    #[cfg(test)]
    pub(crate) fn reports(self) -> bool {
        self.report
    }
}

/// Emit the single documented warm-policy decision event (`docs/observability.md`).
///
/// Private: every caller goes through [`WarmPolicyProposal::settle_with_selection`] or
/// [`WarmPolicyProposal::decline`], so a `rematerialized` event cannot be emitted without a selection
/// that actually moved. `Unchanged` is silent — the overwhelmingly common warm hit must not log once
/// per request.
fn log_warm_policy_decision(
    engine_id: &str,
    decision: WarmPolicyDecision,
    loaded: ExecutionPolicy,
    requested: ExecutionPolicy,
    reopenability: SourceReopenability,
    attestation: StagingAttestation,
) {
    if matches!(decision, WarmPolicyDecision::Unchanged) {
        return;
    }
    // A refusal is the only outcome an operator may need to act on — an imported checkpoint that can
    // never receive the memory ladder. Granted and bounded outcomes are routine planning.
    if matches!(decision, WarmPolicyDecision::RefusedSwitch(_)) {
        tracing::warn!(
            event = "generator_cache_warm_policy_decision",
            engine = engine_id,
            decision = decision.label(),
            reason = decision.reason(),
            source = reopenability.label(),
            staging = attestation.label(),
            loadedOffloadPolicy = ?loaded.offload_policy,
            loadedLoadShape = ?loaded.load_shape,
            loadedLoadShapeDeclarationResult = ?loaded.load_shape_declaration_result,
            requestedOffloadPolicy = ?requested.offload_policy,
            requestedLoadShape = ?requested.load_shape,
            requestedLoadShapeDeclarationResult = ?requested.load_shape_declaration_result,
            "refused a warm execution-policy switch the resident source cannot perform"
        );
    } else {
        tracing::info!(
            event = "generator_cache_warm_policy_decision",
            engine = engine_id,
            decision = decision.label(),
            reason = decision.reason(),
            source = reopenability.label(),
            staging = attestation.label(),
            loadedOffloadPolicy = ?loaded.offload_policy,
            loadedLoadShape = ?loaded.load_shape,
            loadedLoadShapeDeclarationResult = ?loaded.load_shape_declaration_result,
            requestedOffloadPolicy = ?requested.offload_policy,
            requestedLoadShape = ?requested.load_shape,
            requestedLoadShapeDeclarationResult = ?requested.load_shape_declaration_result,
            "decided a warm execution-policy switch"
        );
    }
}

/// The typed execution-domain values one request selects. Every field is `None` unless a provider's
/// own declaration pins exactly one legal value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ExecutionDomainSelection {
    pub(crate) graph_eval_cadence: Option<GraphEvalCadence>,
    pub(crate) ffn_chunk: Option<FfnChunk>,
    pub(crate) cfg_batching: Option<CfgBatching>,
}

impl ExecutionDomainSelection {
    pub(crate) const fn is_empty(&self) -> bool {
        self.graph_eval_cadence.is_none() && self.ffn_chunk.is_none() && self.cfg_batching.is_none()
    }

    /// Write the selection onto a request's memory block. A field left `None` is untouched, so a value
    /// some other layer already chose is never replaced with "unset".
    pub(crate) fn apply(self, memory: &mut GenerationMemory) {
        if let Some(cadence) = self.graph_eval_cadence {
            memory.graph_eval_cadence = Some(cadence);
        }
        if let Some(chunk) = self.ffn_chunk {
            memory.ffn_chunk = Some(chunk);
        }
        if let Some(mode) = self.cfg_batching {
            memory.cfg_batching = Some(mode);
        }
    }
}

/// The single legal value of a numeric domain, or `None` when the domain is unsupported or leaves a
/// choice open.
///
/// `ExecutionValueDomain::AtLeast` is an open half-line: every value in it is *declared*, but nothing
/// in it is *measured*, and both a wider graph-evaluation cadence and a larger FFN chunk trade peak
/// memory for speed. The memory corpus carries no cadence or chunk axis at all — neither
/// `docs/generated/memory-calibration-evidence.json` nor the manifest `calibrations[].parameters`
/// blocks it derives from name one, and `mlx_fit_gate::parse_evidence_parameters` rejects any
/// parameter key outside the five ladder knobs — so there is no admitted grid point to pick and the
/// honest selection is the provider's own default, `None`. A multi-member `Candidates` list is the
/// same situation with a finite grid.
///
/// A `Candidates` list with exactly one member is different: it IS the grid, it has one point, and
/// selecting it is what the provider would have done itself.
fn single_declared_value(domain: &ExecutionValueDomain) -> Option<u32> {
    match domain {
        ExecutionValueDomain::Candidates(candidates) if candidates.len() == 1 => {
            candidates.first().copied()
        }
        ExecutionValueDomain::Unsupported
        | ExecutionValueDomain::AtLeast(_)
        | ExecutionValueDomain::Candidates(_) => None,
    }
}

/// Select typed execution-domain values for one request from what the provider declares.
///
/// Fail-closed by construction: every returned value satisfies `domain.accepts(..)`, so
/// `Capabilities::validate_request` — which refuses an undeclared or out-of-domain selection with
/// `Error::Unsupported` — can never fire on a plan this function produced. An undeclared or unmeasured
/// domain yields `None`, which every provider resolves to its own historical constant, so a request
/// that selects nothing renders byte-for-byte as it did before sc-18317.
pub(crate) fn select_execution_domains(surface: &ExecutionSurface) -> ExecutionDomainSelection {
    ExecutionDomainSelection {
        graph_eval_cadence: single_declared_value(&surface.graph_eval_cadence_blocks)
            .and_then(|blocks| GraphEvalCadence::new(blocks).ok()),
        ffn_chunk: single_declared_value(&surface.ffn_chunk_rows)
            .and_then(|rows| FfnChunk::new(rows).ok()),
        cfg_batching: match &surface.cfg_batching {
            CfgBatchingDomain::Modes(modes) if modes.len() == 1 => modes.first().copied(),
            CfgBatchingDomain::Unsupported | CfgBatchingDomain::Modes(_) => None,
        },
    }
}

/// Plan the typed execution domains onto one already-assembled provider request.
///
/// Called from `memory_strategy::generate_with_scope`, the one seam every image/video generation on
/// either backend passes through. Deliberately a no-op when `request.memory` is `None`: an absent
/// memory block already means "every request-scoped knob is the provider's own default", which is
/// exactly what an unset domain means, so manufacturing a block to carry a value equal to the default
/// would change the `memory.is_some()` signal a provider may read as "the caller opted into
/// request-scoped adaptation" while changing nothing about the render.
pub(crate) fn plan_request_execution_domains(
    generator: &dyn gen_core::Generator,
    request: &mut gen_core::GenerationRequest,
) -> ExecutionDomainSelection {
    let Some(memory) = request.memory.as_mut() else {
        return ExecutionDomainSelection::default();
    };
    let selection = select_execution_domains(&generator.descriptor().capabilities.execution);
    selection.apply(memory);
    selection
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::LoadShapeDeclarationResult;
    use std::num::NonZeroU32;

    fn policy(offload: OffloadPolicy, shape: LoadShape) -> ExecutionPolicy {
        ExecutionPolicy {
            offload_policy: offload,
            load_shape: shape,
            load_shape_declaration_result: LoadShapeDeclarationResult::NotEvaluated,
        }
    }

    /// The policy a decision executes under — the accessor's rule, exercised directly.
    fn effective_policy(
        loaded: ExecutionPolicy,
        requested: ExecutionPolicy,
        decision: WarmPolicyDecision,
    ) -> ExecutionPolicy {
        if decision.grants_requested_policy() {
            requested
        } else {
            loaded
        }
    }

    fn resident_eager() -> ExecutionPolicy {
        policy(OffloadPolicy::Resident, LoadShape::EagerMaterialization)
    }

    fn staged_deferred() -> ExecutionPolicy {
        policy(
            OffloadPolicy::Sequential,
            LoadShape::DeferredMaterialization,
        )
    }

    #[test]
    fn identical_policies_are_unchanged_and_silent() {
        assert_eq!(
            decide_warm_policy(
                resident_eager(),
                resident_eager(),
                SourceReopenability::Reopenable,
                StagingAttestation::Implemented,
            ),
            WarmPolicyDecision::Unchanged
        );
    }

    #[test]
    fn a_declaration_only_difference_changes_no_execution() {
        let mut requested = resident_eager();
        requested.load_shape_declaration_result = LoadShapeDeclarationResult::Refused;
        let decision = decide_warm_policy(
            resident_eager(),
            requested,
            SourceReopenability::Reopenable,
            StagingAttestation::Implemented,
        );
        assert_eq!(
            decision,
            WarmPolicyDecision::ServedAsIs(ServedAsIsReason::DeclarationAuthorityOnly)
        );
        assert!(!decision.grants_requested_policy());
        assert_eq!(
            effective_policy(resident_eager(), requested, decision),
            resident_eager(),
            "a declaration marker must not become the executed policy"
        );
    }

    #[test]
    fn a_tighter_request_on_a_reopenable_source_is_granted_without_a_reload() {
        let decision = decide_warm_policy(
            resident_eager(),
            staged_deferred(),
            SourceReopenability::Reopenable,
            StagingAttestation::Implemented,
        );
        assert_eq!(decision, WarmPolicyDecision::Rematerialized);
        assert!(decision.grants_requested_policy());
        assert_eq!(
            effective_policy(resident_eager(), staged_deferred(), decision),
            staged_deferred()
        );
    }

    #[test]
    fn a_looser_request_is_served_under_the_admitted_loaded_policy() {
        // The memory-safety direction: admission was proved against the staged/deferred load, and
        // running fully resident would peak above it.
        let decision = decide_warm_policy(
            staged_deferred(),
            resident_eager(),
            SourceReopenability::Reopenable,
            StagingAttestation::Implemented,
        );
        assert_eq!(
            decision,
            WarmPolicyDecision::ServedAsIs(ServedAsIsReason::LoadedPolicyBoundsThePeak)
        );
        assert_eq!(
            effective_policy(staged_deferred(), resident_eager(), decision),
            staged_deferred()
        );
    }

    #[test]
    fn a_sideways_switch_that_trades_one_lever_for_the_other_is_not_granted() {
        // Equal containment rank, different levers. Nothing proves the swap stays under the admitted
        // peak, so the loaded shape wins.
        let loaded = policy(OffloadPolicy::Sequential, LoadShape::EagerMaterialization);
        let requested = policy(OffloadPolicy::Resident, LoadShape::DeferredMaterialization);
        assert_eq!(
            decide_warm_policy(
                loaded,
                requested,
                SourceReopenability::Reopenable,
                StagingAttestation::Implemented,
            ),
            WarmPolicyDecision::ServedAsIs(ServedAsIsReason::LoadedPolicyBoundsThePeak)
        );
    }

    #[test]
    fn a_non_reopenable_source_refuses_every_tighter_switch() {
        // Only switches that MOVE THE STAGING AXIS reach the source question; a load-shape-only
        // switch is refused earlier, for having no execution seam at all (see the test below).
        for requested in [
            staged_deferred(),
            policy(OffloadPolicy::Sequential, LoadShape::EagerMaterialization),
        ] {
            let decision = decide_warm_policy(
                resident_eager(),
                requested,
                SourceReopenability::SingleFileNotReopenable,
                StagingAttestation::Implemented,
            );
            assert_eq!(
                decision,
                WarmPolicyDecision::RefusedSwitch(RefusalReason::SourceNotReopenable),
                "requested {requested:?} against an imported single file"
            );
            assert!(!decision.grants_requested_policy());
            assert_eq!(
                effective_policy(resident_eager(), requested, decision),
                resident_eager()
            );
        }
    }

    #[test]
    fn a_non_reopenable_source_still_reports_a_looser_request_as_bounded_not_refused() {
        // The refusal must name the real obstacle. A looser request is declined because admission
        // bounds it, and it would be declined on a snapshot directory too.
        assert_eq!(
            decide_warm_policy(
                staged_deferred(),
                resident_eager(),
                SourceReopenability::SingleFileNotReopenable,
                StagingAttestation::Implemented,
            ),
            WarmPolicyDecision::ServedAsIs(ServedAsIsReason::LoadedPolicyBoundsThePeak)
        );
    }

    /// MINOR 2 (review cycle 2): a switch that tightens ONLY `LoadShape` is REFUSED, not reported as a
    /// served-as-is or a re-materialization.
    ///
    /// `OffloadPolicy` reaches the provider through the ladder's staged-residency selection, which the
    /// grant floors. `LoadShape` has no request-scoped selector at all, so granting such a switch would
    /// claim a re-materialization whose deferred half executed nowhere. Refusing names the real
    /// situation. Mutation sentinel: delete the guard in `decide_warm_policy` and this goes red.
    #[test]
    fn a_load_shape_only_switch_is_refused_for_having_no_execution_seam() {
        for (loaded, requested) in [
            (
                resident_eager(),
                policy(OffloadPolicy::Resident, LoadShape::DeferredMaterialization),
            ),
            (
                policy(OffloadPolicy::Sequential, LoadShape::EagerMaterialization),
                staged_deferred(),
            ),
        ] {
            // Both fail-closed signals admit it, so nothing else can be the cause of the refusal.
            let decision = decide_warm_policy(
                loaded,
                requested,
                SourceReopenability::Reopenable,
                StagingAttestation::Implemented,
            );
            assert_eq!(
                decision,
                WarmPolicyDecision::RefusedSwitch(RefusalReason::NoExecutionSeamForLoadShapeAlone),
                "load-shape-only {loaded:?} -> {requested:?} must be refused, never half-granted"
            );
            assert!(!decision.grants_requested_policy());
            assert_eq!(
                effective_policy(loaded, requested, decision),
                loaded,
                "a refused switch executes under the loaded policy"
            );
            assert_eq!(decision.reason(), "no_execution_seam_for_load_shape_alone");
        }
    }

    /// Every granted proposal moves the staging axis, so the ladder floor is the seam it executes
    /// through. This is the invariant `requires_staged_residency`'s debug assertion encodes.
    #[test]
    fn a_grant_always_requires_the_staged_residency_floor() {
        let decision = decide_warm_policy(
            resident_eager(),
            staged_deferred(),
            SourceReopenability::Reopenable,
            StagingAttestation::Implemented,
        );
        let proposal = WarmPolicyProposal::new(
            "fixture",
            decision,
            resident_eager(),
            staged_deferred(),
            SourceReopenability::Reopenable,
            StagingAttestation::Implemented,
        );
        assert!(proposal.requires_staged_residency());
        assert!(!WarmPolicyProposal::inert("fixture").requires_staged_residency());
    }

    /// The one-shot holder rations a `Copy` proposal across a multi-item job: the real decision goes to
    /// the first evaluation, every later item gets an inert one, so N images settle ONE decision.
    #[test]
    fn the_one_shot_holder_yields_the_real_proposal_exactly_once() {
        let granted = decide_warm_policy(
            resident_eager(),
            staged_deferred(),
            SourceReopenability::Reopenable,
            StagingAttestation::Implemented,
        );
        let mut once = WarmPolicyOnce::new(WarmPolicyProposal::new(
            "fixture",
            granted,
            resident_eager(),
            staged_deferred(),
            SourceReopenability::Reopenable,
            StagingAttestation::Implemented,
        ));
        let first = once.take();
        assert_eq!(first.decision(), granted);
        assert!(first.requires_staged_residency());
        assert!(first.reports());
        for item in 1..4 {
            let later = once.take();
            assert_eq!(
                later.decision(),
                granted,
                "item {item} must keep the grant: it is a fact about the resident generator, and \
                 dropping it would run later items at the peak the switch exists to avoid"
            );
            assert!(
                later.requires_staged_residency(),
                "item {item} must still be floored to staged candidates"
            );
            assert!(
                !later.reports(),
                "item {item} must not re-emit the one cache access's decision"
            );
        }
    }

    /// `decline_if_unsettled` covers the path where no request is ever evaluated, and is one-shot too.
    #[test]
    fn declining_an_unused_holder_is_one_shot_as_well() {
        let mut once = WarmPolicyOnce::new(WarmPolicyProposal::inert("fixture"));
        once.decline_if_unsettled(ServedAsIsReason::RouteHasNoRequestScopedMemory);
        let after = once.take();
        assert_eq!(after.decision(), WarmPolicyDecision::Unchanged);
        assert!(!after.reports(), "the decision was already settled");
    }

    #[test]
    fn a_loaded_instance_whose_contract_omits_staging_is_not_offered_a_no_op_switch() {
        let decision = decide_warm_policy(
            resident_eager(),
            policy(OffloadPolicy::Sequential, LoadShape::EagerMaterialization),
            SourceReopenability::Reopenable,
            StagingAttestation::NotImplemented,
        );
        assert_eq!(
            decision,
            WarmPolicyDecision::ServedAsIs(ServedAsIsReason::LoadedContractDoesNotImplementStaging)
        );
    }

    #[test]
    fn a_staging_instance_grants_a_deferred_shape() {
        let decision = decide_warm_policy(
            resident_eager(),
            staged_deferred(),
            SourceReopenability::Reopenable,
            StagingAttestation::Implemented,
        );
        assert_eq!(decision, WarmPolicyDecision::Rematerialized);
    }

    #[test]
    fn every_outcome_reports_a_distinct_stable_reason() {
        let all = [
            WarmPolicyDecision::Unchanged,
            WarmPolicyDecision::ServedAsIs(ServedAsIsReason::DeclarationAuthorityOnly),
            WarmPolicyDecision::ServedAsIs(ServedAsIsReason::LoadedPolicyBoundsThePeak),
            WarmPolicyDecision::ServedAsIs(ServedAsIsReason::LoadedContractDoesNotImplementStaging),
            WarmPolicyDecision::Rematerialized,
            WarmPolicyDecision::RefusedSwitch(RefusalReason::SourceNotReopenable),
        ];
        let reasons: std::collections::BTreeSet<&str> =
            all.iter().map(|decision| decision.reason()).collect();
        assert_eq!(
            reasons.len(),
            all.len(),
            "reasons must be 1:1 with outcomes"
        );
        assert_eq!(
            all.iter()
                .filter(|decision| decision.grants_requested_policy())
                .count(),
            1,
            "exactly one outcome may execute the requested policy"
        );
        for decision in all {
            assert!(!decision.label().is_empty() && !decision.reason().is_empty());
        }
    }

    #[test]
    fn every_single_file_base_source_is_conservatively_not_reopenable() {
        let dir = tempfile::tempdir().expect("weights tempdir");
        assert_eq!(
            SourceReopenability::of_spec(&LoadSpec::new(WeightsSource::Dir(
                dir.path().to_path_buf()
            ))),
            SourceReopenability::Reopenable
        );
        let file = dir.path().join("model.safetensors");
        std::fs::write(&file, b"imported checkpoint bytes").expect("write single file");
        assert_eq!(
            SourceReopenability::of_spec(&LoadSpec::new(WeightsSource::File(file.clone()))),
            SourceReopenability::SingleFileNotReopenable,
            "an unprepared imported checkpoint cannot prove it reopens"
        );
        // A prepared pin does NOT lift the classification, and that is the load-bearing case: at the
        // pinned revision `mlx-gen-sdxl`'s fused-LDM Resident load obtains exactly such a pin, reads
        // through it once, drops it, and installs `Residency::resident`'s error-returning loaders. Its
        // contract still declares rung 1 implemented, so this proxy is the only thing between a grant
        // and an `Error::Unsupported` part-way through generation.
        let mut prepared = LoadSpec::new(WeightsSource::File(file.clone()));
        prepared
            .prepare_with_file_pins([
                gen_core::PinnedWeightsFile::pin(&file).expect("pin the single file")
            ])
            .expect("install the prepared pin");
        assert_eq!(
            SourceReopenability::of_spec(&prepared),
            SourceReopenability::SingleFileNotReopenable,
            "a pin proves the FILE re-reads, never that the provider kept a loader to use it"
        );
        assert_eq!(
            SourceReopenability::of_spec(&LoadSpec::new(WeightsSource::File(
                dir.path().join("absent.safetensors")
            ))),
            SourceReopenability::SingleFileNotReopenable
        );
    }

    #[test]
    fn an_inert_surface_selects_nothing_and_leaves_the_request_byte_identical() {
        let selection = select_execution_domains(&ExecutionSurface::default());
        assert_eq!(selection, ExecutionDomainSelection::default());
        assert!(selection.is_empty());
        let mut memory = GenerationMemory::default();
        selection.apply(&mut memory);
        assert_eq!(memory, GenerationMemory::default());
    }

    #[test]
    fn an_open_half_line_declaration_is_declared_but_unmeasured_so_nothing_is_selected() {
        let surface = ExecutionSurface {
            graph_eval_cadence_blocks: ExecutionValueDomain::ANY_POSITIVE,
            ffn_chunk_rows: ExecutionValueDomain::AtLeast(
                NonZeroU32::new(512).expect("nonzero floor"),
            ),
            cfg_batching: CfgBatchingDomain::Unsupported,
        };
        assert_eq!(
            select_execution_domains(&surface),
            ExecutionDomainSelection::default(),
            "AtLeast declares a half-line; the corpus measures no point on it"
        );
    }

    #[test]
    fn a_multi_point_declaration_waits_for_evidence_that_does_not_exist_yet() {
        let surface = ExecutionSurface {
            graph_eval_cadence_blocks: ExecutionValueDomain::Candidates(vec![1, 2, 5, 10]),
            ffn_chunk_rows: ExecutionValueDomain::Candidates(vec![1024, 2048]),
            cfg_batching: CfgBatchingDomain::Modes(vec![
                CfgBatching::Batched,
                CfgBatching::Sequential,
            ]),
        };
        assert_eq!(
            select_execution_domains(&surface),
            ExecutionDomainSelection::default()
        );
    }

    #[test]
    fn a_single_point_declaration_is_the_grid_and_is_selected() {
        let surface = ExecutionSurface {
            graph_eval_cadence_blocks: ExecutionValueDomain::Candidates(vec![4]),
            ffn_chunk_rows: ExecutionValueDomain::Candidates(vec![2048]),
            cfg_batching: CfgBatchingDomain::Modes(vec![CfgBatching::Batched]),
        };
        let selection = select_execution_domains(&surface);
        assert_eq!(
            selection.graph_eval_cadence.map(GraphEvalCadence::blocks),
            Some(4)
        );
        assert_eq!(selection.ffn_chunk.map(FfnChunk::rows), Some(2048));
        assert_eq!(selection.cfg_batching, Some(CfgBatching::Batched));
        assert!(!selection.is_empty());
    }

    #[test]
    fn a_malformed_zero_candidate_is_dropped_rather_than_forwarded() {
        let surface = ExecutionSurface {
            graph_eval_cadence_blocks: ExecutionValueDomain::Candidates(vec![0]),
            ffn_chunk_rows: ExecutionValueDomain::Candidates(vec![0]),
            cfg_batching: CfgBatchingDomain::Unsupported,
        };
        assert_eq!(
            select_execution_domains(&surface),
            ExecutionDomainSelection::default(),
            "zero is not a cadence or a chunk; the typed constructors refuse it"
        );
    }

    #[test]
    fn every_selected_value_passes_the_validation_that_would_otherwise_refuse_it() {
        // The property that keeps the worker off the provider's refusal path, over every declaration
        // shape gen-core can express — including a malformed zero candidate.
        for surface in [
            ExecutionSurface::default(),
            ExecutionSurface {
                graph_eval_cadence_blocks: ExecutionValueDomain::Candidates(vec![1]),
                ffn_chunk_rows: ExecutionValueDomain::Candidates(vec![1]),
                cfg_batching: CfgBatchingDomain::Modes(vec![CfgBatching::Sequential]),
            },
            ExecutionSurface {
                graph_eval_cadence_blocks: ExecutionValueDomain::Candidates(vec![0]),
                ffn_chunk_rows: ExecutionValueDomain::Candidates(vec![0]),
                cfg_batching: CfgBatchingDomain::Unsupported,
            },
            ExecutionSurface {
                graph_eval_cadence_blocks: ExecutionValueDomain::ANY_POSITIVE,
                ffn_chunk_rows: ExecutionValueDomain::Candidates(vec![64, 128]),
                cfg_batching: CfgBatchingDomain::Modes(vec![CfgBatching::Batched]),
            },
        ] {
            let selection = select_execution_domains(&surface);
            let mut memory = GenerationMemory::default();
            selection.apply(&mut memory);
            surface
                .validate("planner_property", Some(&memory))
                .expect("a planned selection must pass the provider's own validation");
        }
    }

    #[test]
    fn a_selection_never_unsets_a_value_another_layer_already_chose() {
        let mut memory = GenerationMemory {
            cfg_batching: Some(CfgBatching::Batched),
            ..Default::default()
        };
        ExecutionDomainSelection::default().apply(&mut memory);
        assert_eq!(memory.cfg_batching, Some(CfgBatching::Batched));
    }

    #[test]
    fn peak_containment_orders_the_four_shapes() {
        assert_eq!(peak_containment_rank(resident_eager()), 0);
        assert_eq!(peak_containment_rank(staged_deferred()), 2);
        for middle in [
            policy(OffloadPolicy::Sequential, LoadShape::EagerMaterialization),
            policy(OffloadPolicy::Resident, LoadShape::DeferredMaterialization),
        ] {
            assert_eq!(peak_containment_rank(middle), 1);
        }
    }
}

/// Corpus-replay acceptance for the execution planner (sc-18317).
///
/// Replays the requests the SHIPPED evidence already describes — the promoted calibration bundle and
/// the manifest's admitted/refused decode-quality rows — through the planner and asserts two things:
/// the selections are stable, and every one of them is inside the declared domain the provider would
/// validate. No measurement is taken and no GPU is touched: every input is a checked-in artifact or a
/// weights-free registry descriptor.
#[cfg(test)]
mod corpus_replay_tests {
    use super::*;
    use sceneworks_core::memory_calibration::{load_packaged_bundle, BundleLoad};
    use serde_json::Value;

    /// The three execution-domain keys, in the spellings a calibration `parameters` block would use
    /// if the corpus ever grew one.
    const EXECUTION_DOMAIN_KEYS: [&str; 3] = ["graphEvalCadence", "ffnChunk", "cfgBatching"];

    fn shipped_manifest() -> Value {
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(include_str!(
            "../../../config/manifests/builtin.models.jsonc"
        )))
        .expect("builtin.models.jsonc parses")
    }

    /// The premise the "unmeasured ⇒ leave unset" rule rests on: no shipped measurement describes an
    /// execution domain, on either side of the evidence system.
    ///
    /// Asserted as a SHAPE — "no row names one of these keys" — not as a population count, so a
    /// re-collection that widens or narrows a sweep does not disturb it. If a future capture DOES add
    /// one of these axes, this goes red with the exact key, which is the signal that
    /// [`single_declared_value`] may start honoring a measured grid point.
    #[test]
    fn no_shipped_measurement_names_an_execution_domain_axis() {
        let BundleLoad::Ready(bundle) = load_packaged_bundle().expect("the packaged bundle parses")
        else {
            // A stale bundle is a legitimate shipped state (schema/ABI ahead of the corpus) and every
            // consumer falls back to the legacy path; there is nothing to replay through the planner.
            return;
        };
        for record in &bundle.records {
            for key in record.strategy.parameters.keys() {
                assert!(
                    !EXECUTION_DOMAIN_KEYS.contains(&key.as_str()),
                    "calibration record {} names execution-domain parameter {key:?}; the planner's \
                     unmeasured-domain rule must be revisited before this can ship",
                    record.id
                );
            }
        }

        let manifest = shipped_manifest();
        for model in manifest["models"].as_array().expect("manifest models") {
            let model_id = model["id"].as_str().unwrap_or("(unnamed)");
            for backend in ["mlx", "candle"] {
                let Some(calibrations) = model[backend]["calibrations"].as_array() else {
                    continue;
                };
                for calibration in calibrations {
                    let Some(parameters) = calibration["parameters"].as_object() else {
                        continue;
                    };
                    for key in parameters.keys() {
                        assert!(
                            !EXECUTION_DOMAIN_KEYS.contains(&key.as_str()),
                            "{model_id}/{backend} calibration names execution-domain parameter \
                             {key:?}"
                        );
                    }
                }
            }
        }
    }

    /// Every geometry the shipped decode-quality corpus admits or refuses, replayed as a request.
    ///
    /// The ladder parameters come from the row itself, so these are the exact request shapes P9
    /// measured. The planner must leave every one of them untouched (it owns only the three execution
    /// domains) and must not add a value the provider would refuse.
    fn decode_quality_request_memories() -> Vec<(String, GenerationMemory)> {
        let manifest = shipped_manifest();
        let mut out = Vec::new();
        for model in manifest["models"].as_array().expect("manifest models") {
            let model_id = model["id"].as_str().unwrap_or("(unnamed)").to_owned();
            for backend in ["mlx", "candle"] {
                let Some(implementations) =
                    model[backend]["memoryStrategyContract"]["implementations"].as_array()
                else {
                    continue;
                };
                for implementation in implementations {
                    let Some(policies) =
                        implementation["parameterRanges"]["decodeGeometryPolicies"].as_array()
                    else {
                        continue;
                    };
                    for policy in policies {
                        let kind = policy["disposition"]["kind"].as_str().unwrap_or_default();
                        // Both dispositions are replayed on purpose. A refused row is a request the
                        // ladder must NOT bound, and the planner must be just as inert on it.
                        let engages_decode = kind == "admitted";
                        out.push((
                            format!(
                                "{model_id}/{backend}/{kind}/{}x{}",
                                policy["geometry"]["width"], policy["geometry"]["height"]
                            ),
                            GenerationMemory {
                                stage_residency: false,
                                tile_vae_decode: engages_decode,
                                decode_tile_edge: engages_decode
                                    .then(|| policy["tileEdge"].as_u64())
                                    .flatten()
                                    .and_then(|edge| u32::try_from(edge).ok()),
                                decode_overlap: engages_decode
                                    .then(|| policy["overlap"].as_u64())
                                    .flatten()
                                    .and_then(|overlap| u32::try_from(overlap).ok()),
                                ..Default::default()
                            },
                        ));
                    }
                }
            }
        }
        out
    }

    #[test]
    fn the_shipped_decode_quality_corpus_yields_replayable_requests() {
        // Shape, not population: the corpus must be reachable at all, or the replay below is vacuous
        // and would pass for the wrong reason.
        assert!(
            !decode_quality_request_memories().is_empty(),
            "no decode-quality rows were reachable from the shipped manifest"
        );
    }

    /// The planner is idempotent and deterministic over every shipped request shape.
    ///
    /// Runs against every declaration shape gen-core can express rather than only the registry's, so
    /// the property holds on a host with no backend linked too.
    #[test]
    fn replayed_requests_get_stable_selections_that_never_disturb_the_ladder() {
        let surfaces = [
            ExecutionSurface::default(),
            ExecutionSurface {
                graph_eval_cadence_blocks: ExecutionValueDomain::ANY_POSITIVE,
                ffn_chunk_rows: ExecutionValueDomain::ANY_POSITIVE,
                cfg_batching: CfgBatchingDomain::Unsupported,
            },
            ExecutionSurface {
                graph_eval_cadence_blocks: ExecutionValueDomain::Unsupported,
                ffn_chunk_rows: ExecutionValueDomain::Unsupported,
                cfg_batching: CfgBatchingDomain::Modes(vec![CfgBatching::Batched]),
            },
            ExecutionSurface {
                graph_eval_cadence_blocks: ExecutionValueDomain::Candidates(vec![8]),
                ffn_chunk_rows: ExecutionValueDomain::Candidates(vec![2048]),
                cfg_batching: CfgBatchingDomain::Modes(vec![CfgBatching::Sequential]),
            },
        ];
        for (label, baseline) in decode_quality_request_memories() {
            for surface in &surfaces {
                let selection = select_execution_domains(surface);
                assert_eq!(
                    selection,
                    select_execution_domains(surface),
                    "{label}: the selection must not depend on when it is asked"
                );

                let mut planned = baseline;
                selection.apply(&mut planned);
                let mut twice = planned;
                selection.apply(&mut twice);
                assert_eq!(
                    twice, planned,
                    "{label}: applying a plan twice must be inert"
                );

                assert_eq!(
                    (
                        planned.stage_residency,
                        planned.tile_vae_decode,
                        planned.chunk_attention,
                        planned.stream_transformer_blocks,
                        planned.decode_tile_edge,
                        planned.decode_overlap,
                        planned.attention_chunk_size,
                        planned.transformer_window_size,
                        planned.transformer_window_component,
                    ),
                    (
                        baseline.stage_residency,
                        baseline.tile_vae_decode,
                        baseline.chunk_attention,
                        baseline.stream_transformer_blocks,
                        baseline.decode_tile_edge,
                        baseline.decode_overlap,
                        baseline.attention_chunk_size,
                        baseline.transformer_window_size,
                        baseline.transformer_window_component,
                    ),
                    "{label}: the planner owns only the three execution domains"
                );

                surface
                    .validate(&label, Some(&planned))
                    .unwrap_or_else(|error| {
                        panic!("{label}: the provider would refuse the planned request: {error}")
                    });
            }
        }
    }

    /// Against the REGISTRY this host links: no shipped provider refuses what the planner selects for
    /// it, and the selection stays inside its declared domain.
    ///
    /// This is the test that proves the worker never triggers `Capabilities::validate_request`'s
    /// execution-domain refusal in production — the refusal path exists, is exercised upstream, and
    /// must remain unreachable from here.
    #[cfg(any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    ))]
    #[test]
    fn no_registered_provider_refuses_what_the_planner_selects_for_it() {
        let mut declaring = 0usize;
        let mut providers = 0usize;
        for registration in crate::inference_runtime::media().generators() {
            let descriptor = (registration.descriptor)();
            let surface = &descriptor.capabilities.execution;
            providers += 1;
            if !surface.is_inert() {
                declaring += 1;
                assert!(
                    surface.declaration_errors().is_empty(),
                    "{} publishes a malformed execution surface: {:?}",
                    descriptor.id,
                    surface.declaration_errors()
                );
            }
            let selection = select_execution_domains(surface);
            if let Some(cadence) = selection.graph_eval_cadence {
                assert!(
                    surface.graph_eval_cadence_blocks.accepts(cadence.blocks()),
                    "{}: selected cadence {} is outside {}",
                    descriptor.id,
                    cadence.blocks(),
                    surface.graph_eval_cadence_blocks.describe()
                );
            }
            if let Some(chunk) = selection.ffn_chunk {
                assert!(
                    surface.ffn_chunk_rows.accepts(chunk.rows()),
                    "{}: selected chunk {} is outside {}",
                    descriptor.id,
                    chunk.rows(),
                    surface.ffn_chunk_rows.describe()
                );
            }
            if let Some(mode) = selection.cfg_batching {
                assert!(
                    surface.cfg_batching.accepts(mode),
                    "{}: selected cfg_batching {} is outside {}",
                    descriptor.id,
                    mode.label(),
                    surface.cfg_batching.describe()
                );
            }
            // Replay every shipped decode-quality request shape through this provider's own
            // validation, so the planner's selection is graded by the code that would refuse it.
            for (label, baseline) in decode_quality_request_memories() {
                let mut planned = baseline;
                selection.apply(&mut planned);
                surface
                    .validate(descriptor.id, Some(&planned))
                    .unwrap_or_else(|error| {
                        panic!(
                            "{}/{label}: provider refusal reachable: {error}",
                            descriptor.id
                        )
                    });
            }
        }
        assert!(
            providers > 0,
            "the linked registry published no generators, so this replay proved nothing"
        );
        // Shape, not a pinned population: at least one shipped provider must declare an execution
        // surface, or the planner has nothing to select from and the assertions above are vacuous.
        assert!(
            declaring > 0,
            "no linked provider declares an execution surface; sc-18317's planner has no reachable \
             input on this backend"
        );
    }
}
