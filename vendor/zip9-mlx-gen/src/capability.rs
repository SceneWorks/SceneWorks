//! Production capability switches for the shared MLX optimization surface (sc-18316, sc-18318).
//!
//! P1 retained compilation and P3 exact epilogues both shipped behind
//! [`crate::diagnostics::toggle_requested`], which is `false` unless a diagnostic scope names the
//! toggle — and the only non-test installer was the benchmark binary. Every production render
//! therefore took the discarded-handle / composed arm: exactly the pathology the two stories were
//! filed to remove. This module is the seam that fixes that. Both are now real capabilities whose
//! **production default is on**, with the benchmark keeping A/B authority through
//! [`crate::diagnostics::capability_override`] and any path that genuinely cannot run one keeping a
//! truthful opt-out ([`CapabilityOptOut`]) instead of a silent fallback.
//!
//! # Resolution order
//!
//! [`enabled`] answers in this order, and the order is load-bearing:
//!
//! 1. **An installed [`CapabilityOptOut`] wins.** An opt-out exists because a path cannot correctly
//!    run the capability, so a benchmark request must not be able to force it back on. When a
//!    benchmark did request it, the opt-out emits [`ToggleDisposition::Unavailable`] — the campaign
//!    then fails its receipt validation loudly rather than measuring a path it did not select.
//! 2. **An [`crate::diagnostics::CapabilityAuthority::Override`] scope decides.** Membership of the
//!    requested-toggle set forces on *and* off, which is what the P6 matrix compares.
//! 3. **Otherwise the production default: on.**
//!
//! # Why enabling these cannot change output
//!
//! Neither capability is an approximation. P1 replaces a per-call `compile(f, true)(args)` — a fresh
//! handle, traced and leased per call — with a retained handle over the **same** graph function, so
//! the executed kernel is identical and only its construction is amortized. P3 substitutes MLX's
//! native fused epilogue for the composed expression only where it is **bit-exact**, and reports a
//! per-operation [`ToggleDisposition::Fallback`] otherwise. The accepted MLX tile-shape ULP context
//! (fused-vs-unfused GEMM tiling differing by 1 ULP) does **not** bite here — but not because the
//! epilogues avoid matmuls: three of the six P3 ops **are** matmuls (`quantized_matmul_bias` →
//! `affine_qmm_bias_t`; `conv2d`/`conv3d` → implicit GEMM). What makes them exact is (a) each fused
//! kernel is the **same template instantiation** as its composed counterpart, with `has_bias` as a
//! template parameter over identical tile constants — so the accumulation order cannot change — and
//! (b) `ops.cpp` carries admission guards mirroring MLX's own qmv/split-K/Winograd kernel
//! selection, rejecting any shape that would route to a differently-tiled kernel with a `Fallback`
//! receipt instead of running it. On that admitted set the epilogues are asserted `array_eq`
//! against the composed baseline across every supported dtype, layout, group size, and bit width in
//! `nn`'s P3 suite. The P1 sites are proven bit-identical to their one-shot compiled forms, and —
//! at the level of each individual compiled op — to their eager forms.
//!
//! # Memory tradeoff of a default-on retention
//!
//! Retention keeps MLX compiled-graph state alive for the owning **thread's** lifetime instead of
//! erasing it after every call, and with a production default that now happens on every render
//! thread rather than only under the benchmark. That is bounded and does not disturb the memory
//! contracts:
//!
//! * **Nothing capturing a tensor is retained.** Every retained handle in this workspace wraps a
//!   plain, non-capturing `fn` item over its inputs (`gated_impl`, `modulate_impl`,
//!   `rope_rotate_impl`, `gelu_ffn_impl`, `swiglu_impl`, `add_hint_impl`, `silu_glue_impl`, and this
//!   crate's `compiled_*` graph functions). A retained handle therefore cannot pin a request's
//!   inputs/outputs or a component's weights, so it cannot keep an **evicted** component resident and
//!   the residency/eviction paths need no new release hook.
//! * **What it does hold** is compiler metadata, the built kernels, and any constant traced *inside*
//!   the graph function — a handful of four-byte scalars such as the literal `1` in the adaLN affine.
//!   That is bytes per signature, not the megabytes the fit gate budgets, so a fit decision computed
//!   from weights plus activations stays valid. `mlx::get_peak_memory` reports **active** rather than
//!   cached bytes, so it too is unaffected by kernel caching.
//! * **Measured, not argued**: `nn`'s serialized
//!   `retained_compile_handles_do_not_pin_request_arrays` audit, run against this change, reports
//!   `retained_active=4` bytes and `retained_cache=0` while the whole 25-handle inventory is alive
//!   after a 4 MiB (`request_bytes=4194304`) request — four bytes, the one traced f32 constant. The
//!   benchmark's retention-memory receipt bounds the same aggregate independently.
//! * **Release is deliberately not per-request.** Reuse *across* requests is the entire P1 gain, so
//!   [`crate::nn::release_retained_compile_glue`] is an explicit thread-level reclaim for a worker
//!   shutting down or trimming, not something a request scope calls on `finish`.

use std::cell::Cell;

use crate::diagnostics::{self, ToggleDisposition};

/// An independently switchable MLX pipeline capability whose production default is **on**.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelineCapability {
    /// P1: retained `mx.compile` handles reused across calls, blocks, steps, and requests.
    RetainedCompilation,
    /// P3: MLX's native fused conv+bias / GroupNorm+affine / quantized-matmul+bias / SiLU /
    /// tanh-GELU epilogues in place of the composed expressions.
    ExactEpilogues,
}

/// Every capability this module resolves, in a stable order.
pub const PIPELINE_CAPABILITIES: [PipelineCapability; 2] = [
    PipelineCapability::RetainedCompilation,
    PipelineCapability::ExactEpilogues,
];

impl PipelineCapability {
    /// The benchmark toggle identity this capability shares with the P6 matrix and its receipts.
    pub const fn toggle(self) -> &'static str {
        match self {
            Self::RetainedCompilation => diagnostics::RETAINED_COMPILATION,
            Self::ExactEpilogues => diagnostics::EXACT_EPILOGUES,
        }
    }
}

thread_local! {
    /// Thread-local, matching the surrounding design: the compiled-glue setting and the retained
    /// handles themselves are thread-local so one concurrent render cannot change another's numeric
    /// path. A process-global opt-out would reintroduce exactly that cross-request coupling.
    static RETAINED_COMPILATION_OPT_OUT: Cell<bool> = const { Cell::new(false) };
    static EXACT_EPILOGUES_OPT_OUT: Cell<bool> = const { Cell::new(false) };
}

fn with_opt_out<T>(capability: PipelineCapability, f: impl FnOnce(&Cell<bool>) -> T) -> T {
    match capability {
        PipelineCapability::RetainedCompilation => RETAINED_COMPILATION_OPT_OUT.with(f),
        PipelineCapability::ExactEpilogues => EXACT_EPILOGUES_OPT_OUT.with(f),
    }
}

/// Whether `capability` runs for the work about to execute on this thread.
///
/// See the module documentation for the resolution order. With no opt-out installed and no
/// [`crate::diagnostics::CapabilityAuthority::Override`] scope active — the production case — this is
/// `true`.
pub fn enabled(capability: PipelineCapability) -> bool {
    let toggle = capability.toggle();
    if with_opt_out(capability, Cell::get) {
        // Truthful, not silent: a benchmark that asked for a capability this path opted out of gets
        // an Unavailable receipt, which its validator rejects rather than scoring as a measurement.
        if diagnostics::capability_override(toggle) == Some(true) {
            diagnostics::record_toggle(toggle, ToggleDisposition::Unavailable);
        }
        return false;
    }
    diagnostics::capability_override(toggle).unwrap_or(true)
}

/// Whether `capability` is opted out on this thread, independently of any active scope.
pub fn opted_out(capability: PipelineCapability) -> bool {
    with_opt_out(capability, Cell::get)
}

/// RAII opt-out: disables one capability for this thread for the guard's lifetime and **restores the
/// prior value on drop**, even on an early `?` return.
///
/// This is the sanctioned way for a path that genuinely cannot run a capability to say so. It is
/// deliberately explicit and deliberately auditable: while it is installed, a benchmark request for
/// the same capability records [`ToggleDisposition::Unavailable`] rather than silently measuring the
/// unoptimized arm. No provider installs one today; the parity baselines in the P1/P3 suites use it
/// to obtain the pre-promotion composed/one-shot path to compare against.
#[must_use = "dropping the guard restores the prior capability setting; bind it for the scope that cannot run it"]
#[derive(Debug)]
pub struct CapabilityOptOut {
    capability: PipelineCapability,
    prev: bool,
}

impl CapabilityOptOut {
    /// Opt this thread out of `capability` until the returned guard drops.
    pub fn install(capability: PipelineCapability) -> Self {
        let prev = with_opt_out(capability, |cell| cell.replace(true));
        Self { capability, prev }
    }
}

impl Drop for CapabilityOptOut {
    fn drop(&mut self) {
        with_opt_out(self.capability, |cell| cell.set(self.prev));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::{DiagnosticCounter, EXACT_EPILOGUES, RETAINED_COMPILATION};

    /// The whole point of sc-18316/sc-18318's promotion: with no scope and no opt-out — production —
    /// both capabilities are on. Before this seam existed, both were `false` here.
    #[test]
    fn production_default_is_on_for_every_capability() {
        for capability in PIPELINE_CAPABILITIES {
            assert!(!opted_out(capability));
            assert!(
                enabled(capability),
                "{capability:?} must be enabled by default in a production render"
            );
        }
    }

    #[test]
    fn an_opt_out_is_scoped_and_restores_the_prior_value() {
        for capability in PIPELINE_CAPABILITIES {
            {
                let _guard = CapabilityOptOut::install(capability);
                assert!(opted_out(capability));
                assert!(!enabled(capability));
                {
                    let _nested = CapabilityOptOut::install(capability);
                    assert!(!enabled(capability));
                }
                assert!(
                    !enabled(capability),
                    "a nested opt-out must restore the outer opt-out, not the default"
                );
            }
            assert!(enabled(capability), "the guard must restore the default");
            assert!(!opted_out(capability));
        }
    }

    /// The benchmark keeps A/B authority in **both** directions: an override scope that names the
    /// toggle forces it on, and one that omits it forces it off despite the production default.
    #[test]
    fn an_override_scope_forces_both_states() {
        for capability in PIPELINE_CAPABILITIES {
            let scope = diagnostics::begin_request_with_toggles(
                "capability-on",
                "test",
                &[capability.toggle()],
            )
            .unwrap();
            assert!(enabled(capability));
            let _ = scope.finish();

            let scope = diagnostics::begin_request("capability-off", "test").unwrap();
            assert!(
                !enabled(capability),
                "{capability:?} must be forced off by a toggle-free override variant"
            );
            let _ = scope.finish();
        }
    }

    /// An observation scope must not change what production does — otherwise it could not be used to
    /// prove what production does.
    #[test]
    fn an_observation_scope_preserves_the_production_default() {
        let scope = diagnostics::begin_observed_request("observed", "test").unwrap();
        for capability in PIPELINE_CAPABILITIES {
            assert!(enabled(capability));
            assert_eq!(diagnostics::capability_override(capability.toggle()), None);
        }
        let report = scope.finish();
        // Resolution alone is not a receipt: only the call sites record one. The P3 terminal
        // Unavailable is the derived "in force but never admitted an operation" statement.
        assert_eq!(
            report.counters,
            [DiagnosticCounter::Toggle {
                toggle: EXACT_EPILOGUES,
                disposition: ToggleDisposition::Unavailable,
                count: 1,
            }]
        );
    }

    /// An opt-out outranks a benchmark request and says so on the receipt.
    #[test]
    fn an_opt_out_outranks_a_benchmark_request_and_reports_unavailable() {
        let _guard = CapabilityOptOut::install(PipelineCapability::RetainedCompilation);
        let scope = diagnostics::begin_request_with_toggles(
            "opt-out-vs-request",
            "test",
            &[RETAINED_COMPILATION],
        )
        .unwrap();
        assert!(!enabled(PipelineCapability::RetainedCompilation));
        let report = scope.finish();
        assert_eq!(
            report.counters,
            [DiagnosticCounter::Toggle {
                toggle: RETAINED_COMPILATION,
                disposition: ToggleDisposition::Unavailable,
                count: 1,
            }],
            "an opted-out capability must report Unavailable, never a silent fallback"
        );
    }

    /// An opt-out that nobody asked about stays silent: production pays no receipt cost.
    #[test]
    fn an_unrequested_opt_out_records_nothing() {
        let _guard = CapabilityOptOut::install(PipelineCapability::RetainedCompilation);
        let scope = diagnostics::begin_request("opt-out-unrequested", "test").unwrap();
        assert!(!enabled(PipelineCapability::RetainedCompilation));
        let report = scope.finish();
        assert!(!report.counters.iter().any(|counter| matches!(
            counter,
            DiagnosticCounter::Toggle {
                toggle: RETAINED_COMPILATION,
                ..
            }
        )));
    }

    #[test]
    fn capability_toggles_are_the_benchmark_identities() {
        assert_eq!(
            PipelineCapability::RetainedCompilation.toggle(),
            RETAINED_COMPILATION
        );
        assert_eq!(PipelineCapability::ExactEpilogues.toggle(), EXACT_EPILOGUES);
        for capability in PIPELINE_CAPABILITIES {
            assert!(diagnostics::BENCHMARK_TOGGLES.contains(&capability.toggle()));
        }
    }
}
