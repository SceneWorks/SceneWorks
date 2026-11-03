use super::*;

// ------------------------------------------------------------------------------------------
// Measured lower bounds, capture side (sc-22738, epic 22723)
// ------------------------------------------------------------------------------------------

/// The digest the test declarations carry for `bernini:mlx` — the bound below is CURRENT when
/// its own key equals this and STALE otherwise.
const DECLARED: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
/// A key no declaration carries: the closure moved under a pin bump.
const MOVED: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

/// Loader-closure declarations naming exactly the lanes given, each at the stated digest — the
/// currency half of the comparison, stated rather than read from the packaged file so a test
/// can say "current" and "stale" outright.
fn closures_declaring(
    lanes: &[(&str, &str)],
) -> sceneworks_core::memory_anchor::AnchorLoaderClosures {
    use sceneworks_core::memory_anchor as anchor;
    anchor::AnchorLoaderClosures {
        comment: String::new(),
        digest_version: anchor::ANCHOR_LOADER_CLOSURE_VERSION.to_owned(),
        inference_revision: "0".repeat(40),
        models: lanes
            .iter()
            .map(|(lane, digest)| {
                (
                    (*lane).to_owned(),
                    anchor::AnchorLoaderClosure {
                        engine_id: None,
                        entry_points: vec!["crates/x/src/lib.rs".to_owned()],
                        digest: (*digest).to_owned(),
                        closure_file_count: 1,
                        closure_files: vec!["crates/x/src/lib.rs".to_owned()],
                    },
                )
            })
            .collect(),
    }
}

/// Declarations under which the Bernini bound below is CURRENT.
fn current_closures() -> sceneworks_core::memory_anchor::AnchorLoaderClosures {
    closures_declaring(&[("bernini:mlx", DECLARED)])
}

/// One Bernini stop at `frames`, keyed at `digest`, observed at `footprint` bytes.
fn bernini_bound(
    id: &str,
    frames: u32,
    footprint: u64,
    digest: &str,
) -> sceneworks_core::memory_anchor::ExceededBound {
    use sceneworks_core::memory_anchor as anchor;
    anchor::ExceededBound {
        id: id.to_owned(),
        model_id: "bernini".to_owned(),
        model_family: "bernini".to_owned(),
        route: "bernini".to_owned(),
        provider: "bernini".to_owned(),
        backend: anchor::AnchorBackend::Mlx,
        tier: "bf16".to_owned(),
        transformer_variant: None,
        decoder: None,
        mode: "text_to_video".to_owned(),
        overlay: None,
        reference_count: 0,
        load_shape: anchor::AnchorLoadShape::EagerMaterialization,
        geometry: anchor::AnchorGeometry {
            width: 848,
            height: 480,
            frames,
            fps: None,
        },
        observed_footprint_bytes: footprint,
        ceiling_bytes: 94_822_600_832,
        host_memory_bytes: 137_438_953_472,
        reason: format!("physical_footprint_at_or_above_94822600832:observed_{footprint}"),
        source: anchor::AnchorSource {
            path: "docs/calibration/sc-22738/bernini-bf16-mlx-exceeded-evidence.json".to_owned(),
            sha256: "0".repeat(64),
            record_id: "exc-0".to_owned(),
            calibration_fingerprint: "fp".to_owned(),
            loader_closure_digest: digest.to_owned(),
            currency_attestation: None,
        },
    }
}

fn store_of(
    bounds: Vec<sceneworks_core::memory_anchor::ExceededBound>,
) -> sceneworks_core::memory_anchor::MemoryAnchorStore {
    use sceneworks_core::memory_anchor as anchor;
    anchor::MemoryAnchorStore {
        schema_version: anchor::MEMORY_ANCHOR_SCHEMA_VERSION,
        anchors: Vec::new(),
        analytic_only: Vec::new(),
        component_deltas: Vec::new(),
        observed_operating_envelopes: Vec::new(),
        exceeded_bounds: bounds,
    }
}

/// A store carrying the Bernini stop, keyed CURRENT against [`current_closures`]: the
/// currency comparison the lookup makes is the real one, over stated declarations.
fn bernini_bound_store() -> sceneworks_core::memory_anchor::MemoryAnchorStore {
    use sceneworks_core::memory_anchor as anchor;
    anchor::MemoryAnchorStore {
        schema_version: anchor::MEMORY_ANCHOR_SCHEMA_VERSION,
        anchors: Vec::new(),
        analytic_only: Vec::new(),
        component_deltas: Vec::new(),
        observed_operating_envelopes: Vec::new(),
        exceeded_bounds: vec![anchor::ExceededBound {
            id: "exceeded:bernini:mlx:bf16:base:base:fp:exc-0".to_owned(),
            model_id: "bernini".to_owned(),
            model_family: "bernini".to_owned(),
            route: "bernini".to_owned(),
            provider: "bernini".to_owned(),
            backend: anchor::AnchorBackend::Mlx,
            tier: "bf16".to_owned(),
            transformer_variant: None,
            decoder: None,
            mode: "text_to_video".to_owned(),
            overlay: None,
            reference_count: 0,
            load_shape: anchor::AnchorLoadShape::EagerMaterialization,
            geometry: anchor::AnchorGeometry {
                width: 848,
                height: 480,
                frames: 49,
                fps: None,
            },
            observed_footprint_bytes: 97_147_294_328,
            ceiling_bytes: 94_822_600_832,
            host_memory_bytes: 137_438_953_472,
            reason: "physical_footprint_at_or_above_94822600832:observed_97147294328".to_owned(),
            source: anchor::AnchorSource {
                path: "docs/calibration/sc-22738/bernini-bf16-mlx-exceeded-evidence.json"
                    .to_owned(),
                sha256: "0".repeat(64),
                record_id: "exc-0".to_owned(),
                calibration_fingerprint: "fp".to_owned(),
                // CURRENT under `current_closures()` (sc-22738): the tooling seam refuses only
                // a current bound, so the store is keyed to the declaration on purpose.
                // Production binds regardless of currency; this mirror does not.
                loader_closure_digest: DECLARED.to_owned(),
                currency_attestation: None,
            },
        }],
    }
}

fn bernini_capture_request(tier: &str, frames: u64, memory_bytes: u64) -> Value {
    json!({
        "action": "run",
        "hardware": { "probe": "test", "memoryBytes": memory_bytes },
        "planned": {
            "backend": "mlx",
            "loadShape": "eager_materialization",
            "target": {
                "provider": "bernini",
                "modelId": "bernini",
                "tier": tier,
                "mode": "text_to_video",
                "overlay": "none",
                "geometry": { "width": 848, "height": 480, "batch": 1, "frames": frames },
            },
        },
    })
}

/// sc-22738: the capture side refuses the render production refuses. The `bernini:bf16:mlx`
/// anchor spent 72 minutes reaching a footprint this host could not carry and was killed; with
/// the stop on file, the next campaign is told so in milliseconds instead of repeating it.
#[test]
fn a_bounded_capture_is_refused_before_the_load_on_a_host_that_cannot_carry_it() {
    let store = bernini_bound_store();
    let refusal = exceeded_bound_capture_refusal_in(
        &store,
        &current_closures(),
        &bernini_capture_request("bf16", 49, 137_438_953_472),
    )
    .expect("the predicate reads the request")
    .expect("the same 128 GiB Mac that could not finish this render must not be given it again");
    assert!(
        refusal.contains("CURRENT measured lower bound"),
        "{refusal}"
    );
    assert!(
        refusal.contains("97147294328"),
        "it states the footprint: {refusal}"
    );
    assert!(refusal.contains("Refusing before the load"), "{refusal}");
    // The message states the SAME contract the runner states: current ⇒ exceeded_current and
    // never scheduled; stale ⇒ runnable and re-measured; production refuses either way.
    assert!(refusal.contains("exceeded_current"), "{refusal}");
    assert!(refusal.contains("the bound stales"), "{refusal}");
    assert!(refusal.contains("runtime never demotes"), "{refusal}");
}

/// sc-22738: a STALE bound refuses nothing at this seam. The bound was measured under a loader
/// closure the pin no longer loads, so the only way to learn whether the cell still cannot
/// complete is to run it — under the watchdog — and let the outcome supersede the bound. Before
/// this conjunct the seam refused on `refuses_host` alone: the runner listed the cell runnable,
/// the adapter exited 1, the runner logged `capture_failed`, and no engine fix could ever lift a
/// bound (`bernini_image:bf16:mlx` at e16c6a55e). MUTATION: drop the currency conjunct from the
/// lookup (consult `store` instead of the current-only view) and this test reds.
#[test]
fn a_stale_bound_does_not_refuse_the_capture() {
    let request = bernini_capture_request("bf16", 49, 137_438_953_472);
    let stale = store_of(vec![bernini_bound("exc-stale", 49, 97_147_294_328, MOVED)]);
    assert_eq!(
        exceeded_bound_capture_refusal_in(&stale, &current_closures(), &request)
            .expect("the predicate reads it"),
        None,
        "a bound whose loader closure moved must be re-measured, not refused",
    );
    // The SAME bound, keyed to the declaration, refuses — so the only thing that flipped the
    // answer is currency.
    let current = store_of(vec![bernini_bound(
        "exc-current",
        49,
        97_147_294_328,
        DECLARED,
    )]);
    assert!(
        exceeded_bound_capture_refusal_in(&current, &current_closures(), &request)
            .expect("the predicate reads it")
            .is_some(),
        "the identical bound under the declared closure refuses",
    );
}

/// A lane the declarations no longer name is NOT current — the runner's `readExceededBounds`
/// reads `declared !== undefined && declared === digest`, and the mirror must not read a missing
/// declaration as a match.
#[test]
fn a_bound_on_an_undeclared_lane_is_stale_not_current() {
    let request = bernini_capture_request("bf16", 49, 137_438_953_472);
    let store = bernini_bound_store();
    assert_eq!(
        exceeded_bound_capture_refusal_in(
            &store,
            &closures_declaring(&[("bernini:candle", DECLARED)]),
            &request
        )
        .expect("the predicate reads it"),
        None,
        "the mlx bound is undeclared under a candle-only declaration set",
    );
    assert_eq!(
        exceeded_bound_capture_refusal_in(&store, &closures_declaring(&[]), &request)
            .expect("the predicate reads it"),
        None,
    );
}

/// Currency is applied BEFORE the binding lookup. `binding_exceeded_bound` returns the LARGEST
/// covering footprint; if currency were graded on that one alone, a stale wider stop would
/// shadow a current narrower one and the seam would refuse nothing while the runner classified
/// the cell `exceeded_current`. MUTATION: grade currency after the lookup and this reds.
#[test]
fn a_stale_wider_bound_does_not_shadow_a_current_narrower_one() {
    let request = bernini_capture_request("bf16", 49, 137_438_953_472);
    let store = store_of(vec![
        // Stale, at the request's own geometry, the larger footprint: the lookup's favourite.
        bernini_bound("exc-stale-wide", 49, 97_147_294_328, MOVED),
        // Current, measured at 25 frames (covers 49), smaller footprint — still refuses this
        // 128 GiB host outright on `refuses_host`'s first disjunct.
        bernini_bound("exc-current-narrow", 25, 90_000_000_000, DECLARED),
    ]);
    let refusal = exceeded_bound_capture_refusal_in(&store, &current_closures(), &request)
        .expect("the predicate reads it")
        .expect("the current bound binds and refuses");
    assert!(
        refusal.contains("exc-current-narrow"),
        "the refusal names the CURRENT bound, not the stale one: {refusal}"
    );
    assert!(!refusal.contains("exc-stale-wide"), "{refusal}");
}

/// sc-22738 — both Bernini routes are graded on all three declared lifecycle phases again. The
/// pin before e16c6a55e emitted no `Progress::Loading(_)` from `mlx-gen-bernini`, so the still
/// member filed an overall-only ceiling with the conditioning phase declared unattributed;
/// inference #963 emits `Loading(TextEncoder)` before the planner load and `Loading(Renderer)`
/// at the expert loads on both routes, and this arm cuts on them. Read the way the sibling
/// guards in this file read `run_bernini` — the measured render needs real weights.
#[test]
fn both_bernini_routes_cut_every_declared_phase_on_an_engine_boundary() {
    let source = include_str!("mlx.rs");
    let start = source
        .find("\nfn run_bernini(")
        .expect("the arm still exists");
    let body = &source[start
        ..start
            + source[start..]
                .find("\n}\n")
                .expect("the arm's body closes")];
    for cut in [
        "Progress::Loading(LoadPhase::TextEncoder) => {",
        "Progress::Loading(LoadPhase::Renderer) => {",
        "Progress::Decoding => {",
        "if [conditioning.active, denoise.active, decode.active].contains(&0) {",
        "(\"conditioningActivePeak\", \"bytes\", conditioning.active)",
    ] {
        assert!(
            body.contains(cut),
            "the strict three-phase capture lost {cut}"
        );
    }
    for relaxation in [
        "PhaseUnattributed",
        "overall_only_json",
        "BERNINI_VIDEO_MODEL_ID ==",
        "requires_bounded",
    ] {
        assert!(
            !body.contains(relaxation),
            "the Bernini arm must not grade one route more loosely than the other again \
                 ({relaxation})"
        );
    }
}

/// Everything the bound does NOT claim still captures: a bigger host, a shorter clip, another
/// tier. A measurement gate would have refused these too, and that is precisely what epic
/// 22723's E5 forbids.
#[test]
fn an_unbounded_capture_is_untouched_by_the_bound() {
    let store = bernini_bound_store();
    for (label, request) in [
        (
            "a host large enough to carry it",
            bernini_capture_request("bf16", 49, 549_755_813_888),
        ),
        (
            "a clip below the measured point",
            bernini_capture_request("bf16", 25, 137_438_953_472),
        ),
        (
            "another tier of the same model",
            bernini_capture_request("q4", 49, 137_438_953_472),
        ),
    ] {
        assert_eq!(
            exceeded_bound_capture_refusal_in(&store, &current_closures(), &request)
                .expect("the predicate reads it"),
            None,
            "{label} must still be capturable",
        );
    }
    // And an EMPTY store refuses nothing at all — the posture for every cell no hard stop has
    // ever bounded, which is all of them until one is.
    let empty = sceneworks_core::memory_anchor::MemoryAnchorStore {
        exceeded_bounds: Vec::new(),
        ..bernini_bound_store()
    };
    assert_eq!(
        exceeded_bound_capture_refusal_in(
            &empty,
            &current_closures(),
            &bernini_capture_request("bf16", 49, 137_438_953_472)
        )
        .expect("the predicate reads it"),
        None,
    );
}

/// The capture-side allowance is PRODUCTION'S, read from the shared declarations rather than
/// restated: `mlx_fit_gate::spec_headroom_bytes` for a generic engine less the fixed unified
/// reserve, which is what `video_jobs/wan.rs` hands the video gate as `headroom_bytes`.
#[test]
fn the_capture_side_allowance_is_the_workers_video_admission_headroom() {
    assert_eq!(exceeded_bound_headroom_bytes(), 17_179_869_184);
    assert_eq!(
        exceeded_bound_headroom_bytes(),
        sceneworks_core::memory_anchor::gib_to_bytes(
            sceneworks_core::memory_anchor::MLX_GENERIC_HEADROOM_GB
        ) - sceneworks_core::memory_anchor::gib_to_bytes(
            sceneworks_core::memory_anchor::LEGACY_UNIFIED_FALLBACK_RESERVE_GB
        ),
    );
}

/// THE WIRING, not the predicate. `exceeded_bound_capture_refusal_in` can be perfectly correct
/// and reach nothing: deleting the lines that call it out of `run_with` leaves every other test
/// in this module green, because they all drive the predicate directly. This test drives the
/// entry point with a CURRENT bound and a request it covers, and asserts the refusal comes out
/// of the ENTRY POINT — before any provider is dispatched, any weight is opened, or any
/// 72-minute render begins. Driven through `run_with` with a stated store: the packaged
/// Bernini bound is stale on every pin after the one it was measured at, so `run` over the
/// packaged store would prove nothing here (see the sibling test for what it proves instead).
#[test]
fn the_capture_refusal_is_wired_into_the_run_entry_point() {
    let error = run_with(
        Some(&bernini_bound_store()),
        &current_closures(),
        &bernini_capture_request("bf16", 49, 137_438_953_472),
    )
    .expect_err("a bounded request must not reach a provider arm at all");
    assert!(
        error.contains("measured lower bound"),
        "the refusal must be the bound's, not a downstream arm's: {error}",
    );
}

/// The other half of the wiring: a STALE bound lets the request THROUGH the seam into the
/// provider arm. The arm then fails on the test request's missing plan fields — which is the
/// point: the failure is the arm's, not the bound's, so the render would have been attempted.
/// (No weight is opened: `run_bernini` refuses a plan without a selection before any load.)
#[test]
fn a_stale_bound_lets_the_capture_through_the_run_entry_point() {
    let stale = store_of(vec![bernini_bound("exc-stale", 49, 97_147_294_328, MOVED)]);
    let error = run_with(
        Some(&stale),
        &current_closures(),
        &bernini_capture_request("bf16", 49, 137_438_953_472),
    )
    .expect_err("the bare test request cannot complete a Bernini plan");
    assert!(
        !error.contains("measured lower bound"),
        "a stale bound must not be the reason the capture stops: {error}",
    );
}

/// The packaged store's own Bernini stop, through `run`: it was measured at inference
/// 3b922bac6 and the pin has moved since, so under the packaged declarations it is STALE and
/// `run` lets the request through to the arm. This is the shipped behaviour the story fixes —
/// a re-measurement can now lift the bound. If a later pin re-declares the closure at the
/// bound's digest (an attestation, or the loader returning to that source), the bound becomes
/// current again and this test's expectation flips with it: read the assertion message.
#[test]
fn the_packaged_bernini_bound_refuses_through_run_exactly_when_it_is_current() {
    let store = sceneworks_core::memory_anchor::packaged_memory_anchors()
        .expect("the packaged anchor store loads");
    let closures = packaged_anchor_loader_closures().expect("the packaged closures parse");
    let bound = store
        .exceeded_bounds
        .iter()
        .find(|bound| {
            bound.model_id == "bernini"
                && bound.tier == "bf16"
                && bound.backend == sceneworks_core::memory_anchor::AnchorBackend::Mlx
        })
        .expect("the bernini:bf16:mlx footprint hard stop is on file");
    let current = exceeded_bound_is_current(bound, &closures);
    let error = run(&bernini_capture_request("bf16", 49, 137_438_953_472))
        .expect_err("either the bound refuses or the bare plan fails in the arm");
    assert_eq!(
        error.contains("measured lower bound"),
        current,
        "packaged bernini:bf16:mlx bound current={current} at this pin; run said: {error}",
    );
}

/// The shipped store carries the stop this story recorded. A SHAPE assertion, not a count: the
/// claim is that `bernini:bf16:mlx` at 848x480x49 has a measured lower bound on file, which is
/// what makes both refusals above reachable at all.
#[test]
fn the_packaged_store_carries_the_bernini_bf16_mlx_stop() {
    let store = sceneworks_core::memory_anchor::packaged_memory_anchors()
        .expect("the packaged anchor store loads");
    let bound = store
        .exceeded_bounds
        .iter()
        .find(|bound| {
            bound.model_id == "bernini"
                && bound.tier == "bf16"
                && bound.backend == sceneworks_core::memory_anchor::AnchorBackend::Mlx
        })
        .expect("the bernini:bf16:mlx footprint hard stop is on file");
    assert_eq!(bound.mode, "text_to_video");
    assert_eq!(
        (
            bound.geometry.width,
            bound.geometry.height,
            bound.geometry.frames
        ),
        (848, 480, 49)
    );
    assert_eq!(bound.observed_footprint_bytes, 97_147_294_328);
    assert_eq!(bound.host_memory_bytes, 137_438_953_472);
    assert!(bound.observed_footprint_bytes >= bound.ceiling_bytes);
}
