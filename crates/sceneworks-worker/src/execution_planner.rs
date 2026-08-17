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
//!    decision plus one structured event, and [`effective_policy`] names the policy the request
//!    actually executes under.
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
//! ## Why re-openability is a hard interlock
//!
//! `gen_core::Residency::resident(text, heavy)` has no reload loaders (`rebuildable == false`), and
//! any request that asks it to stage or re-materialize returns
//! `Error::Unsupported("resident-only component source cannot stage or rematerialize components …")`.
//! An imported single-file checkpoint with no prepared pin is exactly that source. Granting a switch
//! for one would fail the job with a capability error part-way through generation, so
//! [`SourceReopenability`] classifies the [`LoadSpec`] up front and the switch is refused explicitly
//! instead. That is the "single-file interlock" the story names: treat imported checkpoints as
//! stream-ineligible, fail closed, never assume.

use gen_core::{
    CfgBatching, CfgBatchingDomain, ExecutionSurface, ExecutionValueDomain, FfnChunk,
    GenerationMemory, GraphEvalCadence, LoadShape, LoadSpec, OffloadPolicy,
    StagedResidencyAvailability, WeightsSource,
};

use crate::generator_cache::ExecutionPolicy;

/// Whether the weights behind a resident generator can be re-opened for component
/// re-materialization.
///
/// Derived from the [`LoadSpec`] the generator was loaded from, never from the engine id: one engine
/// serves both a snapshot directory and an imported single file, and only the former can rebuild a
/// component it dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SourceReopenability {
    /// A snapshot directory, or a single file carried with a prepared re-openable pin
    /// (`gen_core::PinnedWeightsFile`, which exists precisely so a streamed provider can reopen a
    /// lexical path under the same identity on every phase).
    Reopenable,
    /// A bare single-file / imported source with no prepared pin. The provider holds it resident-only
    /// and cannot rebuild a component it drops.
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
            WeightsSource::File(path) => match spec.prepared_file_pin_for(path) {
                Ok(Some(_)) => Self::Reopenable,
                // An unprepared spec and an unreadable prepared receipt are the same answer here:
                // neither PROVES the source reopens, and "cannot prove it" must not become "assume
                // it does".
                Ok(None) | Err(_) => Self::SingleFileNotReopenable,
            },
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
    /// The provider declares no selectable staged residency, so `Sequential` is advisory-only for it
    /// and granting the switch would change nothing the provider honors.
    ProviderDoesNotSelectStaging,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RefusalReason {
    /// The resident weights are a bare single-file/imported source. Staging or deferred
    /// materialization would return `Error::Unsupported` from the provider.
    SourceNotReopenable,
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
            Self::ServedAsIs(ServedAsIsReason::ProviderDoesNotSelectStaging) => {
                "provider_does_not_select_staging"
            }
            Self::Rematerialized => "components_rematerialized_in_place",
            Self::RefusedSwitch(RefusalReason::SourceNotReopenable) => "source_not_reopenable",
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
/// `staged_availability` is the provider's own attestation
/// (`gen_core::Capabilities::staged_residency_availability`); `reopenability` classifies the resident
/// weights. Pure — the caller owns the event and the effective policy.
pub(crate) fn decide_warm_policy(
    loaded: ExecutionPolicy,
    requested: ExecutionPolicy,
    reopenability: SourceReopenability,
    staged_availability: StagedResidencyAvailability,
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
    // the admitted envelope. Whether the resident source can deliver it is the remaining question,
    // and it is answered before the provider is asked rather than by letting the provider refuse.
    if !reopenability.is_reopenable() {
        return WarmPolicyDecision::RefusedSwitch(RefusalReason::SourceNotReopenable);
    }
    let wants_more_staging = requested.offload_policy == OffloadPolicy::Sequential
        && loaded.offload_policy == OffloadPolicy::Resident;
    if wants_more_staging && staged_availability == StagedResidencyAvailability::Absent {
        return WarmPolicyDecision::ServedAsIs(ServedAsIsReason::ProviderDoesNotSelectStaging);
    }
    WarmPolicyDecision::Rematerialized
}

/// The policy the request actually executes under, after the decision.
pub(crate) fn effective_policy(
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

/// Emit the single documented warm-policy decision event (`docs/observability.md`).
///
/// `Unchanged` is silent: the overwhelmingly common warm hit must not log once per request.
pub(crate) fn log_warm_policy_decision(
    engine_id: &str,
    decision: WarmPolicyDecision,
    loaded: ExecutionPolicy,
    requested: ExecutionPolicy,
    reopenability: SourceReopenability,
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
                StagedResidencyAvailability::Selectable,
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
            StagedResidencyAvailability::Selectable,
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
            StagedResidencyAvailability::Selectable,
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
            StagedResidencyAvailability::Selectable,
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
                StagedResidencyAvailability::Selectable,
            ),
            WarmPolicyDecision::ServedAsIs(ServedAsIsReason::LoadedPolicyBoundsThePeak)
        );
    }

    #[test]
    fn a_non_reopenable_source_refuses_every_tighter_switch() {
        for requested in [
            staged_deferred(),
            policy(OffloadPolicy::Sequential, LoadShape::EagerMaterialization),
            policy(OffloadPolicy::Resident, LoadShape::DeferredMaterialization),
        ] {
            let decision = decide_warm_policy(
                resident_eager(),
                requested,
                SourceReopenability::SingleFileNotReopenable,
                StagedResidencyAvailability::Selectable,
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
                StagedResidencyAvailability::Selectable,
            ),
            WarmPolicyDecision::ServedAsIs(ServedAsIsReason::LoadedPolicyBoundsThePeak)
        );
    }

    #[test]
    fn a_provider_without_selectable_staging_is_not_offered_a_no_op_switch() {
        let decision = decide_warm_policy(
            resident_eager(),
            policy(OffloadPolicy::Sequential, LoadShape::EagerMaterialization),
            SourceReopenability::Reopenable,
            StagedResidencyAvailability::Absent,
        );
        assert_eq!(
            decision,
            WarmPolicyDecision::ServedAsIs(ServedAsIsReason::ProviderDoesNotSelectStaging)
        );
    }

    #[test]
    fn an_unconditionally_staging_provider_still_grants_a_deferred_shape() {
        let decision = decide_warm_policy(
            resident_eager(),
            staged_deferred(),
            SourceReopenability::Reopenable,
            StagedResidencyAvailability::UnconditionallyEngaged,
        );
        assert_eq!(decision, WarmPolicyDecision::Rematerialized);
    }

    #[test]
    fn every_outcome_reports_a_distinct_stable_reason() {
        let all = [
            WarmPolicyDecision::Unchanged,
            WarmPolicyDecision::ServedAsIs(ServedAsIsReason::DeclarationAuthorityOnly),
            WarmPolicyDecision::ServedAsIs(ServedAsIsReason::LoadedPolicyBoundsThePeak),
            WarmPolicyDecision::ServedAsIs(ServedAsIsReason::ProviderDoesNotSelectStaging),
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
    fn reopenability_is_read_off_the_base_weights_source() {
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
        let mut prepared = LoadSpec::new(WeightsSource::File(file.clone()));
        prepared
            .prepare_with_file_pins([
                gen_core::PinnedWeightsFile::pin(&file).expect("pin the single file")
            ])
            .expect("install the prepared pin");
        assert_eq!(
            SourceReopenability::of_spec(&prepared),
            SourceReopenability::Reopenable,
            "a prepared pin is exactly the re-openable single-file receipt"
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
