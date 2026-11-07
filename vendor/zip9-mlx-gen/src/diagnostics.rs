//! Request-scoped diagnostics for the cross-family MLX benchmark harness.
//!
//! The production generator contract deliberately stays tensor-neutral and does not expose
//! implementation details such as compiled-handle lifetime or provider-local memoization.  The
//! performance harness still needs those details to distinguish an optimization that really ran
//! from a requested toggle that silently fell back.  This module provides a narrow, opt-in seam:
//!
//! * [`begin_request`] installs a collector for the current synchronous render thread;
//! * hot-path [`record_compile`], [`record_cache`], [`record_fallback`], [`record_toggle`], and
//!   [`record_exact_epilogue`] calls are no-ops when no collector is active;
//! * [`DiagnosticScope::finish`] returns stable, aggregated counters for that one request.
//!
//! MLX generation is synchronous and SceneWorks serializes work on the process-default Metal
//! device.  Thread-local ownership therefore gives each request an isolated collector without a
//! process-global lock or a cross-request toggle.  Nested scopes are rejected rather than merged.
//!
//! # Toggles versus capabilities (sc-18316 / sc-18318)
//!
//! Two members of [`BENCHMARK_TOGGLES`] — [`RETAINED_COMPILATION`] and [`EXACT_EPILOGUES`] — are no
//! longer benchmark-only requests: they are real [`crate::capability`] switches whose production
//! default is **on**.  A diagnostic scope keeps A/B authority over them because the P6 matrix
//! contract already treats a variant's requested-toggle set as an *exact and exclusive*
//! specification (`validate_toggle_receipts` rejects "toggle-free variants" that emit any toggle
//! receipt at all).  [`capability_override`] therefore reports membership in both directions for a
//! scope opened with [`CapabilityAuthority::Override`], while [`begin_observed_request`] opens a
//! collector that only *records* — it leaves the production default in force, so what it observes is
//! the production path rather than a path the observer selected.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::time::Instant;

/// P1 retained compiled-operation handles.
pub const RETAINED_COMPILATION: &str = "retained_compilation";
/// P3 exact elementwise/normalization/matmul epilogue fusions.
pub const EXACT_EPILOGUES: &str = "exact_epilogues";
/// P3 biased 2-D convolution operation identity.
pub const EXACT_CONV2D_BIAS: &str = "conv2d_bias";
/// P3 biased 3-D convolution operation identity.
pub const EXACT_CONV3D_BIAS: &str = "conv3d_bias";
/// P3 group-aware GroupNorm plus affine operation identity.
pub const EXACT_GROUP_NORM_AFFINE: &str = "group_norm_affine";
/// P3 packed affine Q4/Q8 matmul plus dense bias operation identity.
pub const EXACT_QUANTIZED_MATMUL_BIAS: &str = "quantized_matmul_bias";
/// P3 eager-compatible SiLU operation identity.
pub const EXACT_SILU: &str = "silu";
/// P3 eager-compatible tanh-GELU operation identity.
pub const EXACT_GELU_TANH: &str = "gelu_tanh";
/// Complete phase-1 operation inventory. Provider request inventories are strict subsets of this
/// list because already-compiled glue and operations absent from a family are not P3 gains.
pub const EXACT_EPILOGUE_OPERATIONS: [&str; 6] = [
    EXACT_CONV2D_BIAS,
    EXACT_CONV3D_BIAS,
    EXACT_GROUP_NORM_AFFINE,
    EXACT_QUANTIZED_MATMUL_BIAS,
    EXACT_SILU,
    EXACT_GELU_TANH,
];
/// P4 fused QK-normalization, RoPE/layout, and projection primitives.
pub const FUSED_ATTENTION_PRIMITIVES: &str = "fused_attention_primitives";
/// P5 indexed/scatter tiled-decode accumulation.
pub const INDEXED_DECODE_ACCUMULATOR: &str = "indexed_decode_accumulator";
/// P9 geometry-aware decode admission and policy.
pub const GEOMETRY_AWARE_DECODE: &str = "geometry_aware_decode";
/// Complete independently selectable optimization surface consumed by the P6 matrix.
pub const BENCHMARK_TOGGLES: [&str; 5] = [
    RETAINED_COMPILATION,
    EXACT_EPILOGUES,
    FUSED_ATTENTION_PRIMITIVES,
    INDEXED_DECODE_ACCUMULATOR,
    GEOMETRY_AWARE_DECODE,
];

/// How a compiled operation was obtained for one invocation.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CompileDisposition {
    /// A fresh `compile(...)` handle was constructed at the call site and dropped after the call.
    OneShot,
    /// A retained handle did not yet contain a graph for the requested shape/signature.
    RetainedMiss,
    /// A retained handle reused an already-compiled shape/signature.
    RetainedHit,
}

/// Outcome of a provider-local cache lookup.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum CacheDisposition {
    Hit,
    Miss,
    Bypass,
}

/// Whether an independently selectable benchmark optimization actually ran.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ToggleDisposition {
    Applied,
    Fallback,
    Unavailable,
}

/// The P9 geometry-aware policy's semantic decision for one request.
///
/// `Unchanged` means P9 deliberately preserved the provider's production/default decoder choice;
/// it does not imply a physical single-pass decoder because Wan may already auto-tile. A
/// `GeometryTiled` decision may change output bytes and therefore also carries the immutable
/// production evidence identity that admitted that policy.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DecodePolicyDisposition {
    Unchanged,
    GeometryTiled,
}

/// Physical decoder path selected for a request carrying a P9 policy receipt.
///
/// This is separate from [`DecodePolicyDisposition`]: Wan can truthfully preserve its production
/// policy (`Unchanged`) while that pre-existing policy auto-tiles the concrete request.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum DecodePathDisposition {
    Dense,
    Tiled,
}

/// Provider-neutral compute boundaries consumed by the P6 benchmark harness.
///
/// Providers emit these at the exact point where pre-denoise work ends and where denoise hands its
/// final latents to the decoder. They are deliberately separate from [`gen_core::Progress`]: the UI
/// progress contract has historically allowed providers to emit `Step` before or after a solver
/// evaluation, so it cannot also be a comparable timing boundary.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum BenchmarkPhaseBoundary {
    DenoiseStart,
    DecodeStart,
}

/// Benchmark-only fixed tile geometry used to compare P5's accumulator mechanics against the old
/// accumulator without also changing P9's production admission policy. It exists only inside an
/// active request-local diagnostic scope; ordinary generation can never observe one.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BenchmarkDecodeControl {
    pub spatial_tile_px: i32,
    pub spatial_overlap_px: i32,
    pub temporal_tile_frames: Option<i32>,
    pub temporal_overlap_frames: Option<i32>,
}

impl BenchmarkDecodeControl {
    pub fn tiling_config(self) -> crate::tiling::TilingConfig {
        crate::tiling::TilingConfig {
            spatial: Some(crate::tiling::SpatialTiling {
                tile_px: self.spatial_tile_px,
                overlap_px: self.spatial_overlap_px,
            }),
            temporal: self
                .temporal_tile_frames
                .map(|tile_frames| crate::tiling::TemporalTiling {
                    tile_frames,
                    overlap_frames: self
                        .temporal_overlap_frames
                        .expect("validated temporal benchmark control has an overlap"),
                }),
        }
    }
}

/// One provider-emitted phase boundary measured from the beginning of the diagnostic request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PhaseBoundaryRecord {
    pub boundary: BenchmarkPhaseBoundary,
    pub elapsed_nanos: u64,
}

/// One aggregated diagnostic counter from a request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DiagnosticCounter {
    Compile {
        site: &'static str,
        disposition: CompileDisposition,
        count: u64,
    },
    Cache {
        site: &'static str,
        disposition: CacheDisposition,
        count: u64,
    },
    Fallback {
        site: &'static str,
        reason: &'static str,
        count: u64,
    },
    Toggle {
        toggle: &'static str,
        disposition: ToggleDisposition,
        count: u64,
    },
    /// One exact P3 operation's concrete request-local path. A request may carry both Applied and
    /// Fallback records for the same operation when different shapes select different dispatchers.
    ExactEpilogue {
        operation: &'static str,
        disposition: ToggleDisposition,
        reason: Option<&'static str>,
        count: u64,
    },
    DecodePolicy {
        disposition: DecodePolicyDisposition,
        decode_path: DecodePathDisposition,
        production_evidence_sha256: Option<String>,
        count: u64,
    },
}

/// Completed diagnostics for exactly one benchmarked request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiagnosticReport {
    pub request_id: String,
    pub family: String,
    pub counters: Vec<DiagnosticCounter>,
    pub phase_boundaries: Vec<PhaseBoundaryRecord>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum CounterKey {
    Compile(&'static str, CompileDisposition),
    Cache(&'static str, CacheDisposition),
    Fallback(&'static str, &'static str),
    Toggle(&'static str, ToggleDisposition),
    ExactEpilogue(&'static str, ToggleDisposition, Option<&'static str>),
    DecodePolicy(
        DecodePolicyDisposition,
        DecodePathDisposition,
        Option<String>,
    ),
}

/// How an active diagnostic scope relates to the production [`crate::capability`] switches.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapabilityAuthority {
    /// The requested-toggle set is an exact, exclusive specification: a capability-backed toggle is
    /// forced **on** by membership and **off** by absence, overriding the production default in both
    /// directions. This is what the P6 A/B matrix needs and what its receipt validator enforces.
    Override,
    /// The scope records receipts only. Capability resolution keeps the production default, so the
    /// receipts describe the path production would have taken with no scope installed at all.
    Observe,
}

struct Collector {
    request_id: String,
    family: String,
    capability_authority: CapabilityAuthority,
    requested_toggles: BTreeSet<&'static str>,
    decode_control: Option<BenchmarkDecodeControl>,
    counters: BTreeMap<CounterKey, u64>,
    started: Instant,
    phase_boundaries: Vec<PhaseBoundaryRecord>,
    phase_observer: Option<Box<dyn FnMut(BenchmarkPhaseBoundary)>>,
}

thread_local! {
    static COLLECTOR: RefCell<Option<Collector>> = const { RefCell::new(None) };
}

/// Returned when a benchmark tries to overlap two diagnostic requests on one render thread.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScopeAlreadyActive;

impl fmt::Display for ScopeAlreadyActive {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("an MLX diagnostic request is already active on this thread")
    }
}

impl std::error::Error for ScopeAlreadyActive {}

/// RAII owner for one request's diagnostic collector.
#[must_use = "finish the scope to retain its diagnostics; dropping it clears the request collector"]
#[derive(Debug)]
pub struct DiagnosticScope {
    active: bool,
}

/// Begin diagnostics for one synchronous render request.
///
/// This is an [`CapabilityAuthority::Override`] scope with an empty toggle set, so every
/// capability-backed toggle is forced **off** for its lifetime — the P6 "toggle-free" baseline.
/// To record receipts for the production defaults instead, use [`begin_observed_request`].
pub fn begin_request(
    request_id: impl Into<String>,
    family: impl Into<String>,
) -> Result<DiagnosticScope, ScopeAlreadyActive> {
    begin_request_with_toggles(request_id, family, &[])
}

/// Begin a collector that **observes** the production path instead of selecting one.
///
/// Every other constructor in this module carries [`CapabilityAuthority::Override`]: its
/// requested-toggle set decides whether each capability-backed toggle runs, which is exactly what
/// the benchmark's A/B matrix needs and exactly what makes it useless as a reachability proof — the
/// scope, not production, chose the path. This constructor installs the same collector with
/// [`CapabilityAuthority::Observe`], so [`crate::capability::enabled`] keeps returning the
/// production default and the receipts describe the arm a production render actually takes.
pub fn begin_observed_request(
    request_id: impl Into<String>,
    family: impl Into<String>,
) -> Result<DiagnosticScope, ScopeAlreadyActive> {
    install_collector(
        request_id,
        family,
        CapabilityAuthority::Observe,
        &[],
        None,
        None,
    )
}

/// Begin diagnostics and expose the requested benchmark toggles to provider code on this render
/// thread. Providers must call [`record_toggle`] after choosing their concrete path; merely finding
/// a name in [`toggle_requested`] is not an activation receipt.
pub fn begin_request_with_toggles(
    request_id: impl Into<String>,
    family: impl Into<String>,
    requested_toggles: &[&'static str],
) -> Result<DiagnosticScope, ScopeAlreadyActive> {
    begin_request_with_toggles_and_phase_observer(request_id, family, requested_toggles, None, None)
}

/// Begin request diagnostics with a synchronous observer for provider-neutral phase boundaries.
///
/// The observer runs on the generation thread before the provider proceeds into the next phase.
/// P6 uses that stop-the-world seam to stop and join the old phase's memory sampler, reset MLX's
/// native active-memory high-water mark, and start the next sampler without misattributing work.
pub fn begin_request_with_phase_observer(
    request_id: impl Into<String>,
    family: impl Into<String>,
    requested_toggles: &[&'static str],
    phase_observer: impl FnMut(BenchmarkPhaseBoundary) + 'static,
) -> Result<DiagnosticScope, ScopeAlreadyActive> {
    begin_request_with_toggles_and_phase_observer(
        request_id,
        family,
        requested_toggles,
        None,
        Some(Box::new(phase_observer)),
    )
}

/// Begin the complete P6 request scope with an optional fixed-tile control and synchronous phase
/// observer. This is intentionally distinct from production-facing diagnostic constructors: only
/// the benchmark harness supplies the control.
pub fn begin_benchmark_request(
    request_id: impl Into<String>,
    family: impl Into<String>,
    requested_toggles: &[&'static str],
    decode_control: Option<BenchmarkDecodeControl>,
    phase_observer: impl FnMut(BenchmarkPhaseBoundary) + 'static,
) -> Result<DiagnosticScope, ScopeAlreadyActive> {
    begin_request_with_toggles_and_phase_observer(
        request_id,
        family,
        requested_toggles,
        decode_control,
        Some(Box::new(phase_observer)),
    )
}

fn begin_request_with_toggles_and_phase_observer(
    request_id: impl Into<String>,
    family: impl Into<String>,
    requested_toggles: &[&'static str],
    decode_control: Option<BenchmarkDecodeControl>,
    phase_observer: Option<Box<dyn FnMut(BenchmarkPhaseBoundary)>>,
) -> Result<DiagnosticScope, ScopeAlreadyActive> {
    install_collector(
        request_id,
        family,
        CapabilityAuthority::Override,
        requested_toggles,
        decode_control,
        phase_observer,
    )
}

fn install_collector(
    request_id: impl Into<String>,
    family: impl Into<String>,
    capability_authority: CapabilityAuthority,
    requested_toggles: &[&'static str],
    decode_control: Option<BenchmarkDecodeControl>,
    phase_observer: Option<Box<dyn FnMut(BenchmarkPhaseBoundary)>>,
) -> Result<DiagnosticScope, ScopeAlreadyActive> {
    COLLECTOR.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err(ScopeAlreadyActive);
        }
        *slot = Some(Collector {
            request_id: request_id.into(),
            family: family.into(),
            capability_authority,
            requested_toggles: requested_toggles.iter().copied().collect(),
            decode_control,
            counters: BTreeMap::new(),
            started: Instant::now(),
            phase_boundaries: Vec::new(),
            phase_observer,
        });
        Ok(DiagnosticScope { active: true })
    })
}

/// Whether the active benchmark request selected `toggle`. Always `false` in production requests
/// that did not install a diagnostic scope.
///
/// This is the correct predicate for the three benchmark-only toggles
/// ([`FUSED_ATTENTION_PRIMITIVES`], [`INDEXED_DECODE_ACCUMULATOR`], [`GEOMETRY_AWARE_DECODE`]). It is
/// **not** how the two capability-backed toggles resolve — see [`capability_override`] and
/// [`crate::capability::enabled`], whose production default is on.
pub fn toggle_requested(toggle: &str) -> bool {
    COLLECTOR.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|collector| collector.requested_toggles.contains(toggle))
    })
}

/// The active scope's A/B decision for a capability-backed `toggle`, or `None` when the production
/// default stands.
///
/// `Some(true)`/`Some(false)` are returned only for an [`CapabilityAuthority::Override`] scope, where
/// membership of the requested set is an exact and exclusive specification. An
/// [`CapabilityAuthority::Observe`] scope and the no-scope production case both return `None`.
pub fn capability_override(toggle: &str) -> Option<bool> {
    debug_assert!(
        BENCHMARK_TOGGLES.contains(&toggle),
        "capability_override is only defined for the benchmark toggle surface"
    );
    COLLECTOR.with(|slot| {
        let slot = slot.borrow();
        let collector = slot.as_ref()?;
        match collector.capability_authority {
            CapabilityAuthority::Override => Some(collector.requested_toggles.contains(toggle)),
            CapabilityAuthority::Observe => None,
        }
    })
}

/// Fixed tiled-decode control for the active P6 request, or `None` for production/default and P9
/// policy comparisons. This is request-local and always `None` outside the benchmark scope.
pub fn benchmark_decode_control() -> Option<BenchmarkDecodeControl> {
    COLLECTOR.with(|slot| {
        slot.borrow()
            .as_ref()
            .and_then(|collector| collector.decode_control)
    })
}

fn increment(key: CounterKey) {
    COLLECTOR.with(|slot| {
        let mut slot = slot.borrow_mut();
        if let Some(collector) = slot.as_mut() {
            *collector.counters.entry(key).or_insert(0) += 1;
        }
    });
}

/// Record one compiled-operation invocation. No-op outside an active request.
pub fn record_compile(site: &'static str, disposition: CompileDisposition) {
    increment(CounterKey::Compile(site, disposition));
}

/// Record one provider-local cache decision. No-op outside an active request.
pub fn record_cache(site: &'static str, disposition: CacheDisposition) {
    increment(CounterKey::Cache(site, disposition));
}

/// Record an explicit fallback and its stable reason. No-op outside an active request.
pub fn record_fallback(site: &'static str, reason: &'static str) {
    increment(CounterKey::Fallback(site, reason));
}

/// Record whether one benchmark toggle actually ran. A runner must not infer this from an
/// environment variable or from output timing alone.
pub fn record_toggle(toggle: &'static str, disposition: ToggleDisposition) {
    increment(CounterKey::Toggle(toggle, disposition));
}

/// Record one P3 operation's concrete selection. Applied records carry no reason; rejected native
/// paths carry a stable Fallback/Unavailable reason while the caller executes the original composed
/// expression. The aggregate P3 terminal receipt is derived at [`DiagnosticScope::finish`]: any
/// admitted operation makes the toggle Applied, while an all-fallback or empty request remains
/// Fallback/Unavailable. The P6 validator separately requires the exact provider operation inventory,
/// so one successful helper cannot certify an incomplete request.
pub fn record_exact_epilogue(
    operation: &'static str,
    disposition: ToggleDisposition,
    reason: Option<&'static str>,
) {
    debug_assert!(EXACT_EPILOGUE_OPERATIONS.contains(&operation));
    debug_assert_eq!(reason.is_none(), disposition == ToggleDisposition::Applied);
    increment(CounterKey::ExactEpilogue(operation, disposition, reason));
}

/// Record P9's request-local policy decision. No-op outside an active diagnostic request.
///
/// The benchmark contract rejects `GeometryTiled` without both a physical tiled path and a
/// lower-hex SHA-256 identity, and rejects any evidence identity on `Unchanged`; keeping validation
/// in the versioned P6 schema lets legacy or malformed serialized evidence fail closed as well.
pub fn record_decode_policy(
    disposition: DecodePolicyDisposition,
    decode_path: DecodePathDisposition,
    production_evidence_sha256: Option<&str>,
) {
    increment(CounterKey::DecodePolicy(
        disposition,
        decode_path,
        production_evidence_sha256.map(str::to_owned),
    ));
}

/// Emit P9's complete activation receipt for one concrete decoder decision.
///
/// Providers call this only at the point where their request-local decoder path has been resolved.
/// Keeping the `Applied` toggle and semantic policy record together prevents an implementation from
/// accidentally satisfying only half of the P6 receipt contract. Outside an active scope, or when
/// P9 was not requested, this is a no-op and ordinary generation remains untouched.
pub fn record_geometry_decode_decision(
    disposition: DecodePolicyDisposition,
    decode_path: DecodePathDisposition,
    production_evidence_sha256: Option<&str>,
) {
    if !toggle_requested(GEOMETRY_AWARE_DECODE) {
        return;
    }
    record_toggle(GEOMETRY_AWARE_DECODE, ToggleDisposition::Applied);
    record_decode_policy(disposition, decode_path, production_evidence_sha256);
}

/// Record and synchronously publish one provider-neutral phase boundary. No-op outside an active
/// request, so the production path pays only the request-local diagnostic branch.
pub fn record_phase_boundary(boundary: BenchmarkPhaseBoundary) {
    COLLECTOR.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(collector) = slot.as_mut() else {
            return;
        };
        let elapsed_nanos = collector.started.elapsed().as_nanos();
        collector.phase_boundaries.push(PhaseBoundaryRecord {
            boundary,
            elapsed_nanos: u64::try_from(elapsed_nanos).unwrap_or(u64::MAX),
        });
        if let Some(observer) = collector.phase_observer.as_mut() {
            observer(boundary);
        }
    });
}

impl DiagnosticScope {
    /// Finish this request and return its counters in deterministic key order.
    pub fn finish(mut self) -> DiagnosticReport {
        self.active = false;
        COLLECTOR.with(|slot| {
            let mut collector = slot
                .borrow_mut()
                .take()
                .expect("an active diagnostic scope owns one collector");
            // The P3 terminal receipt is derived whenever the capability was in force for this
            // request: because an Override scope requested it, or because an Observe scope let the
            // production default stand. An all-fallback or empty request still terminates
            // Fallback/Unavailable, so an observed request cannot certify a path it never took.
            let epilogues_in_force = collector.requested_toggles.contains(EXACT_EPILOGUES)
                || collector.capability_authority == CapabilityAuthority::Observe;
            if epilogues_in_force
                && !collector.counters.keys().any(|key| {
                    matches!(key, CounterKey::Toggle(toggle, _) if *toggle == EXACT_EPILOGUES)
                })
            {
                let has_applied = collector.counters.keys().any(|key| {
                    matches!(
                        key,
                        CounterKey::ExactEpilogue(_, ToggleDisposition::Applied, None)
                    )
                });
                let has_fallback = collector.counters.keys().any(|key| {
                    matches!(
                        key,
                        CounterKey::ExactEpilogue(_, ToggleDisposition::Fallback, Some(_))
                    )
                });
                let disposition = if has_applied {
                    ToggleDisposition::Applied
                } else if has_fallback {
                    ToggleDisposition::Fallback
                } else {
                    ToggleDisposition::Unavailable
                };
                collector
                    .counters
                    .insert(CounterKey::Toggle(EXACT_EPILOGUES, disposition), 1);
            }
            let counters = collector
                .counters
                .into_iter()
                .map(|(key, count)| match key {
                    CounterKey::Compile(site, disposition) => DiagnosticCounter::Compile {
                        site,
                        disposition,
                        count,
                    },
                    CounterKey::Cache(site, disposition) => DiagnosticCounter::Cache {
                        site,
                        disposition,
                        count,
                    },
                    CounterKey::Fallback(site, reason) => DiagnosticCounter::Fallback {
                        site,
                        reason,
                        count,
                    },
                    CounterKey::Toggle(toggle, disposition) => DiagnosticCounter::Toggle {
                        toggle,
                        disposition,
                        count,
                    },
                    CounterKey::ExactEpilogue(operation, disposition, reason) => {
                        DiagnosticCounter::ExactEpilogue {
                            operation,
                            disposition,
                            reason,
                            count,
                        }
                    }
                    CounterKey::DecodePolicy(
                        disposition,
                        decode_path,
                        production_evidence_sha256,
                    ) => DiagnosticCounter::DecodePolicy {
                        disposition,
                        decode_path,
                        production_evidence_sha256,
                        count,
                    },
                })
                .collect();
            DiagnosticReport {
                request_id: collector.request_id,
                family: collector.family,
                counters,
                phase_boundaries: collector.phase_boundaries,
            }
        })
    }
}

impl Drop for DiagnosticScope {
    fn drop(&mut self) {
        if self.active {
            COLLECTOR.with(|slot| {
                slot.borrow_mut().take();
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_are_aggregated_and_sorted_per_request() {
        let scope = begin_request("case/0", "qwen_image").unwrap();
        record_compile("z", CompileDisposition::OneShot);
        record_compile("z", CompileDisposition::OneShot);
        record_cache("a", CacheDisposition::Miss);
        record_cache("a", CacheDisposition::Hit);
        record_fallback("decoder", "geometry_not_admitted");
        record_toggle("retained_compilation", ToggleDisposition::Unavailable);
        let report = scope.finish();

        assert_eq!(report.request_id, "case/0");
        assert_eq!(report.family, "qwen_image");
        assert_eq!(report.counters.len(), 5);
        assert!(report.counters.contains(&DiagnosticCounter::Compile {
            site: "z",
            disposition: CompileDisposition::OneShot,
            count: 2,
        }));
        assert!(report.counters.contains(&DiagnosticCounter::Cache {
            site: "a",
            disposition: CacheDisposition::Hit,
            count: 1,
        }));
    }

    #[test]
    fn nested_scope_is_rejected_and_drop_clears_the_slot() {
        let scope = begin_request("outer", "wan").unwrap();
        assert_eq!(
            begin_request("nested", "wan").unwrap_err(),
            ScopeAlreadyActive
        );
        drop(scope);
        let next = begin_request("next", "sdxl").unwrap();
        assert!(next.finish().counters.is_empty());
    }

    #[test]
    fn calls_outside_a_scope_do_not_leak_into_the_next_request() {
        record_compile("outside", CompileDisposition::OneShot);
        record_cache("outside", CacheDisposition::Hit);
        let scope = begin_request("clean", "wan").unwrap();
        assert!(scope.finish().counters.is_empty());
    }

    #[test]
    fn requested_toggles_are_request_local_but_not_implicit_receipts() {
        assert!(!toggle_requested("retained_compilation"));
        let scope = begin_request_with_toggles(
            "toggle",
            "wan",
            &["retained_compilation", "indexed_decode_accumulator"],
        )
        .unwrap();
        assert!(toggle_requested("retained_compilation"));
        assert!(!toggle_requested("geometry_aware_decode"));
        assert!(scope.finish().counters.is_empty());
        assert!(!toggle_requested("retained_compilation"));
    }

    #[test]
    fn p3_exact_epilogues_preserve_per_operation_fallback_and_derive_terminal_status() {
        let scope = begin_request_with_toggles("p3", "qwen_image", &[EXACT_EPILOGUES]).unwrap();
        record_exact_epilogue(EXACT_CONV2D_BIAS, ToggleDisposition::Applied, None);
        record_exact_epilogue(EXACT_CONV2D_BIAS, ToggleDisposition::Applied, None);
        record_exact_epilogue(
            EXACT_CONV2D_BIAS,
            ToggleDisposition::Fallback,
            Some("unsupported_dtype_shape_or_dispatch"),
        );
        let report = scope.finish();

        assert!(report.counters.contains(&DiagnosticCounter::ExactEpilogue {
            operation: EXACT_CONV2D_BIAS,
            disposition: ToggleDisposition::Applied,
            reason: None,
            count: 2,
        }));
        assert!(report.counters.contains(&DiagnosticCounter::ExactEpilogue {
            operation: EXACT_CONV2D_BIAS,
            disposition: ToggleDisposition::Fallback,
            reason: Some("unsupported_dtype_shape_or_dispatch"),
            count: 1,
        }));
        assert!(report.counters.contains(&DiagnosticCounter::Toggle {
            toggle: EXACT_EPILOGUES,
            disposition: ToggleDisposition::Applied,
            count: 1,
        }));

        let scope =
            begin_request_with_toggles("p3-fallback", "qwen_image", &[EXACT_EPILOGUES]).unwrap();
        record_exact_epilogue(
            EXACT_CONV2D_BIAS,
            ToggleDisposition::Fallback,
            Some("unsupported_dtype_shape_or_dispatch"),
        );
        let report = scope.finish();
        assert!(report.counters.contains(&DiagnosticCounter::ExactEpilogue {
            operation: EXACT_CONV2D_BIAS,
            disposition: ToggleDisposition::Fallback,
            reason: Some("unsupported_dtype_shape_or_dispatch"),
            count: 1,
        }));
        assert!(report.counters.contains(&DiagnosticCounter::Toggle {
            toggle: EXACT_EPILOGUES,
            disposition: ToggleDisposition::Fallback,
            count: 1,
        }));
    }

    #[test]
    fn p3_exact_epilogues_without_an_admitted_operation_are_not_reported_applied() {
        let scope = begin_request_with_toggles("p3-empty", "wan", &[EXACT_EPILOGUES]).unwrap();
        let report = scope.finish();
        assert_eq!(
            report.counters,
            [DiagnosticCounter::Toggle {
                toggle: EXACT_EPILOGUES,
                disposition: ToggleDisposition::Unavailable,
                count: 1,
            }]
        );
    }

    #[test]
    fn fixed_decode_control_is_benchmark_only_and_request_local() {
        let control = BenchmarkDecodeControl {
            spatial_tile_px: 256,
            spatial_overlap_px: 64,
            temporal_tile_frames: Some(32),
            temporal_overlap_frames: Some(8),
        };
        assert_eq!(benchmark_decode_control(), None);

        let scope = begin_benchmark_request(
            "fixed-tile",
            "wan_video",
            &[INDEXED_DECODE_ACCUMULATOR],
            Some(control),
            |_| {},
        )
        .unwrap();
        assert_eq!(benchmark_decode_control(), Some(control));
        assert!(toggle_requested(INDEXED_DECODE_ACCUMULATOR));
        let tiling = benchmark_decode_control().unwrap().tiling_config();
        let spatial = tiling.spatial.unwrap();
        let temporal = tiling.temporal.unwrap();
        assert_eq!((spatial.tile_px, spatial.overlap_px), (256, 64));
        assert_eq!((temporal.tile_frames, temporal.overlap_frames), (32, 8));

        let report = scope.finish();
        assert!(report.counters.is_empty());
        assert_eq!(benchmark_decode_control(), None);
    }

    #[test]
    fn p9_policy_receipt_preserves_decision_and_evidence_identity() {
        let evidence = "a".repeat(64);
        let scope =
            begin_request_with_toggles("geometry-policy", "image_dit", &[GEOMETRY_AWARE_DECODE])
                .unwrap();
        record_decode_policy(
            DecodePolicyDisposition::GeometryTiled,
            DecodePathDisposition::Tiled,
            Some(evidence.as_str()),
        );
        let report = scope.finish();

        assert_eq!(
            report.counters,
            [DiagnosticCounter::DecodePolicy {
                disposition: DecodePolicyDisposition::GeometryTiled,
                decode_path: DecodePathDisposition::Tiled,
                production_evidence_sha256: Some(evidence),
                count: 1,
            }]
        );
    }

    #[test]
    fn geometry_decode_decision_is_atomic_and_request_scoped() {
        record_geometry_decode_decision(
            DecodePolicyDisposition::Unchanged,
            DecodePathDisposition::Dense,
            None,
        );

        let scope =
            begin_request_with_toggles("geometry", "sdxl", &[GEOMETRY_AWARE_DECODE]).unwrap();
        record_geometry_decode_decision(
            DecodePolicyDisposition::Unchanged,
            DecodePathDisposition::Dense,
            None,
        );
        let report = scope.finish();
        assert_eq!(
            report.counters,
            vec![
                DiagnosticCounter::Toggle {
                    toggle: GEOMETRY_AWARE_DECODE,
                    disposition: ToggleDisposition::Applied,
                    count: 1,
                },
                DiagnosticCounter::DecodePolicy {
                    disposition: DecodePolicyDisposition::Unchanged,
                    decode_path: DecodePathDisposition::Dense,
                    production_evidence_sha256: None,
                    count: 1,
                },
            ]
        );
    }

    #[test]
    fn phase_boundaries_are_explicit_ordered_and_synchronously_observed() {
        use std::cell::RefCell;
        use std::rc::Rc;

        let observed = Rc::new(RefCell::new(Vec::new()));
        let observer = Rc::clone(&observed);
        let scope = begin_request_with_phase_observer("phase", "qwen_image", &[], move |phase| {
            observer.borrow_mut().push(phase);
        })
        .unwrap();
        record_phase_boundary(BenchmarkPhaseBoundary::DenoiseStart);
        record_phase_boundary(BenchmarkPhaseBoundary::DecodeStart);
        let report = scope.finish();

        assert_eq!(
            *observed.borrow(),
            [
                BenchmarkPhaseBoundary::DenoiseStart,
                BenchmarkPhaseBoundary::DecodeStart,
            ]
        );
        assert_eq!(report.phase_boundaries.len(), 2);
        assert_eq!(
            report.phase_boundaries[0].boundary,
            BenchmarkPhaseBoundary::DenoiseStart
        );
        assert_eq!(
            report.phase_boundaries[1].boundary,
            BenchmarkPhaseBoundary::DecodeStart
        );
        assert!(
            report.phase_boundaries[0].elapsed_nanos <= report.phase_boundaries[1].elapsed_nanos
        );
    }
}
