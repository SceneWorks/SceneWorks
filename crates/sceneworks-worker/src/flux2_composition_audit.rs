//! sc-17607: the COMPOSITION half of the SC-15833 FLUX.2 calibration audit.
//!
//! `inference_compatibility_audit` proves the FLUX.2 *codegen* has not moved: it compares git
//! objects over the measurement binary's compile closure and, when one moves, the SHA-256 of the
//! linked binary itself. That instrument answers exactly one question — "is the compiled code the
//! measurements ran still the same code?" — and it is blind to a second one by construction.
//!
//! The second question is composition. The worker does not call `candle-gen-flux2` directly; it
//! resolves `flux2_dev` through
//!
//! ```text
//! runtime-cuda -> candle-gen-catalog -> candle-gen-flux2
//! ```
//!
//! and either of the first two crates could stop registering that id, or register something else
//! under it, without moving one byte of FLUX.2's compiled code. A digest cannot see that, in either
//! direction: the measurement binary links neither crate, so an unchanged digest says nothing about
//! them and a changed one would convict them of nothing.
//!
//! sc-15833 nonetheless put `crates/bundles/runtime-cuda` in the compile closure, on the argument
//! that the worker links it. `candle-gen-catalog` is "linked by the worker" in exactly the same
//! sense and was in no list at all — it moved 988 lines (+949/-39) inside the window sc-17524
//! certified, unexamined, while the audited bundle did not move at all across that same window —
//! so the line between them was not drawn on purpose. The remedy the closure could
//! offer was also not a remedy: a bundle edit registering some unrelated provider is unadjudicable
//! by construction, and the only way out of it was the ~47.6 GB re-capture the artifact layer
//! exists to stop demanding.
//!
//! So both crates left the codegen closure (sc-17607) and the composition question is asked here,
//! where it can actually be answered: against the LINKED bundle, whose `flux2_dev` entry must be
//! `candle-gen-flux2`'s own registration and nothing else. This costs no build beyond the one the
//! candle lane already does, needs no weights and no GPU, and — unlike a re-capture — it answers
//! the question that was asked.
//!
//! **Where this runs.** `windows-candle.yml` (`cargo test -p sceneworks-worker --features
//! backend-candle`). The whole point is to inspect the composition the worker actually links, so
//! there is no platform-neutral version of it; the macOS/MLX bundle composes a different FLUX.2
//! lane whose calibration is not this record's.

use gen_core::{MemoryRegistration, ModelRegistration, ProviderRegistry, ProviderRegistryBuilder};

/// The id SC-15833's measurements were captured for. The whole audit is about this one route.
const FLUX2_DEV: &str = "flux2_dev";

/// The registered generator for `id`, or `None` when the registry has no such route.
///
/// `ProviderRegistry` resolves a load by scanning descriptors for the id, so this is the same
/// lookup a real `load("flux2_dev", spec)` performs — minus the weights.
fn generator<'a>(registry: &'a ProviderRegistry, id: &str) -> Option<&'a ModelRegistration> {
    let mut matching = registry
        .generators()
        .filter(|registration| (registration.descriptor)().id == id);
    let found = matching.next();
    assert!(
        matching.next().is_none(),
        "{id} is registered more than once, which `ProviderRegistryBuilder::build` must reject"
    );
    found
}

/// The registered memory-strategy contract for `id`, or `None`.
fn memory_strategy<'a>(registry: &'a ProviderRegistry, id: &str) -> Option<&'a MemoryRegistration> {
    registry
        .memory_strategy_registrations()
        .find(|registration| registration.provider_id == id)
}

/// `candle-gen-flux2`'s registrations, with nothing composed on top.
///
/// The comparison target. Built from the provider crate the calibration measured, through the same
/// `register_providers` entry point the catalog is supposed to be calling — so "the bundle's
/// `flux2_dev` is this one" is decided by identity, not by a transcribed copy of the descriptor
/// that would go stale the first time an unrelated field moved.
fn provider_only_registry() -> ProviderRegistry {
    runtime_cuda::providers::flux2::register_providers(ProviderRegistryBuilder::new())
        .build()
        .expect("candle-gen-flux2's own registrations must form a valid registry")
}

/// Where the `register_providers` above is actually DEFINED, re-exports notwithstanding.
///
/// The one hole an identity comparison alone leaves open, and it is not a small one: the control is
/// reached through `runtime_cuda::providers::flux2`, which is `candle-gen-catalog`'s own
/// `pub use candle_gen_flux2 as flux2` — the crate this module exists to police owns the alias on
/// BOTH sides of every comparison. Re-point it at a wrapper that registers `flux2_dev` and the two
/// sides move together and stay green.
///
/// `sceneworks-worker` reaches every provider through the bundle by design and does not depend on
/// `candle-gen-flux2` directly, so the answer is read out of the type system instead:
/// `type_name_of_val` on a function ITEM renders its defining path, crate first, and an alias
/// cannot rewrite it (verified: a `pub use inner as aliased` renders `krate::inner::f` through
/// both spellings). So this names the crate the control genuinely came from.
fn control_registrar_definition() -> &'static str {
    std::any::type_name_of_val(&runtime_cuda::providers::flux2::register_providers)
}

/// The shipped CUDA bundle still routes `flux2_dev` to `candle-gen-flux2`, unwrapped.
///
/// Function-pointer identity rather than a frozen descriptor snapshot, deliberately. A snapshot
/// re-asserts what the compiled-artifact digest already covers (the descriptor is FLUX.2's own
/// code) and goes red on every innocuous field addition; identity asserts the one fact the digest
/// cannot reach — that this id is answered by that crate, not by a wrapper, a substitute or a
/// second registration composed over it.
///
/// `std::ptr::fn_addr_eq` rather than `==`: rustc's `unpredictable_function_pointer_comparisons`
/// lint refuses the latter, and under `-D warnings` that is a build failure. Both sides here are
/// copies of the same `const` in `candle-gen-flux2`, so the addresses agree for the only reason
/// that matters — they are the same function.
#[test]
fn the_bundle_routes_flux2_dev_to_the_provider_the_calibration_measured() {
    let bundle = crate::inference_runtime::media();
    let provider_only = provider_only_registry();

    let composed = generator(bundle, FLUX2_DEV).unwrap_or_else(|| {
        panic!(
            "the shipped CUDA bundle registers no '{FLUX2_DEV}' generator at all — the SC-15833 \
             calibration describes a route this worker can no longer resolve"
        )
    });
    let own = generator(&provider_only, FLUX2_DEV)
        .expect("candle-gen-flux2 must register the id its own calibration was captured for");

    assert!(
        std::ptr::fn_addr_eq(composed.descriptor, own.descriptor),
        "'{FLUX2_DEV}' resolves to a descriptor candle-gen-flux2 did not register"
    );
    assert!(
        std::ptr::fn_addr_eq(composed.load, own.load),
        "'{FLUX2_DEV}' loads through something other than candle-gen-flux2's own provider, so the \
         SC-15833 measurements describe code this id no longer reaches"
    );
    assert_eq!(
        composed.footprint.map(|f| f as usize),
        own.footprint.map(|f| f as usize),
        "the composed route declares a different on-disk footprint than the provider it names"
    );

    // A purity tripwire, and nothing more: after the pointer check above this compares `f()` with
    // `f()`. It is here because the whole scheme reads the id out of `(descriptor)()` — a descriptor
    // that answered differently on successive calls would make every lookup in this module, and in
    // `ProviderRegistry::load` itself, non-deterministic. It adds no composition coverage.
    assert_eq!(
        format!("{:?}", (composed.descriptor)()),
        format!("{:?}", (own.descriptor)()),
        "same function, different output — the descriptor is not a pure function of the crate"
    );

    // The comparison has to be able to say NO, or everything above passes by construction. Two
    // pointers that always compare equal is not a hypothetical failure: identical function bodies
    // can be folded together by the linker (MSVC's ICF), and a comparison that had been quietly
    // reduced to `true` would accept a `flux2_dev` answered by any provider at all. `flux2_klein_9b`
    // is the strongest available negative — the SIBLING registration in the very same crate.
    let klein = generator(bundle, "flux2_klein_9b")
        .expect("candle-gen-flux2 registers both FLUX.2 descriptors");
    assert!(
        !std::ptr::fn_addr_eq(composed.load, klein.load),
        "flux2_dev and flux2_klein_9b resolve to the same loader, so this test cannot tell the \
         audited route apart from any other and proves nothing"
    );
    assert!(
        !std::ptr::fn_addr_eq(composed.descriptor, klein.descriptor),
        "the two FLUX.2 descriptors are the same function, so descriptor identity is meaningless here"
    );
}

/// The memory-strategy route is the one the calibration actually exercises.
///
/// A fit measurement is an admission decision: `MemoryRegistration::contract` and `safety_check`
/// are what decide whether a tier is servable, and they are what SC-15833's rungs were captured
/// against. A composition that kept the generator but dropped or replaced this pair would leave
/// every published `flux2_dev` fit describing an admission path the worker no longer runs — and
/// the codegen digest would be byte-identical throughout.
///
/// Presence is asserted, not just agreement: `candle-gen-flux2` registers this pair only under its
/// `cuda` feature, so a comparison alone would pass — twice `None` — on a bundle that had lost the
/// whole memory-strategy lane.
#[test]
fn the_bundle_keeps_flux2_devs_memory_strategy_route_intact() {
    let bundle = crate::inference_runtime::media();
    let provider_only = provider_only_registry();

    let composed = memory_strategy(bundle, FLUX2_DEV).unwrap_or_else(|| {
        panic!(
            "the shipped CUDA bundle registers no '{FLUX2_DEV}' memory strategy, so the SC-15833 \
             fits describe an admission path this worker does not run"
        )
    });
    let own = memory_strategy(&provider_only, FLUX2_DEV)
        .expect("candle-gen-flux2 registers flux2_dev's memory strategy under its cuda feature");

    assert!(
        std::ptr::fn_addr_eq(composed.contract, own.contract),
        "'{FLUX2_DEV}' declares a memory contract candle-gen-flux2 did not register"
    );
    assert!(
        std::ptr::fn_addr_eq(composed.safety_check, own.safety_check),
        "'{FLUX2_DEV}' admits loads through a safety check the calibration never measured"
    );

    // The behaviour fixture rides the same route and is how conformance replays it weights-free.
    // `build()` already rejects a duplicate provider id, so the count is 0 or 1 and pinning it at 1
    // is the whole assertion — a pairwise comparison against `provider_only` could not fail
    // independently of it.
    assert_eq!(
        bundle
            .memory_behavior_registrations()
            .filter(|registration| registration.provider_id == FLUX2_DEV)
            .count(),
        1,
        "flux2_dev's memory-behavior fixture is what replays the measured route without weights"
    );
}

/// The registry under test is the composed CUDA catalog, and the control is the provider crate.
///
/// Both halves are about what the assertions above are *comparing*, not about what they assert.
/// Everything above would be satisfied by a bundle that contained FLUX.2 and nothing else —
/// `inference_runtime::media()` has a second, deliberately empty arm for builds that link no
/// backend, and a future refactor could route the candle lane through something equally
/// degenerate — so the generator floor pins that this is the real catalog. (It does NOT catch a
/// lane that lost `backend-candle`: this module is cfg'd out entirely there, which is what the
/// liveness guard in `scripts/platform-review-contracts.test.mjs` is for. That guard used to live in
/// `scripts/inference-artifact-audit.test.mjs` and moved when sc-17774 deleted the flux2-only audit
/// tooling that file graded — the guard was never about that tooling, and this check survived the
/// removal because "which provider is registered" is a composition question, not an invalidation
/// one.)
#[test]
fn the_audited_composition_is_the_full_cuda_bundle() {
    assert_eq!(runtime_cuda::PLATFORM, "cuda");
    assert_eq!(runtime_cuda::BACKEND, "candle");
    // ...and the control really is candle-gen-flux2's registrar. Both sides of the identity
    // assertions reach it through `candle-gen-catalog`'s own alias, so without this a catalog that
    // re-pointed that alias at a wrapper would move them together and stay green.
    let registrar = control_registrar_definition();
    assert!(
        registrar.starts_with("candle_gen_flux2::"),
        "the flux2 registrar this module compares against is defined in {registrar}, not \
         candle-gen-flux2 — every identity assertion here is then comparing the catalog's own \
         composition against itself"
    );
    let bundle = crate::inference_runtime::media();
    assert!(
        bundle.generators().len() > 40,
        "this is not the composed CUDA media catalog, so nothing it says about flux2_dev's \
         composition means anything: {} generators",
        bundle.generators().len()
    );
    assert!(
        generator(&provider_only_registry(), "flux2_klein_9b").is_some(),
        "the provider-only registry must be candle-gen-flux2's whole surface, not a subset"
    );
}
