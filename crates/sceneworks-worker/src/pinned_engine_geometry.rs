//! Ties the shipped video manifest's geometry (`limits.maxPixels`, and the wan-family
//! `requiresDimensionsMultipleOf`) to the constants of the engine that ACTUALLY SHIPS — the one
//! pinned by `Cargo.toml`'s `runtime-*` tag — instead of to a hand-copied literal.
//!
//! **sc-12409.** sc-12308 corrected both the engine cap and the catalog, but the catalog PR
//! merged first, so `main` advertised a `1280x720` A14B geometry the pinned engine still
//! rejected (candle `validate()`) or silently refit (mlx). No lane caught it: the
//! `sceneworks-core` guard [`shipped_manifest_matches_each_engines_real_geometry`] checks the
//! manifest against the `ENGINE_GEOMETRY` *literals*, and those literals had drifted in lockstep
//! with the manifest — both internally consistent, both ahead of the binary that ships. The
//! adjacent `constants.js` mirror had a parity guard (`videoGeometryParity.test.js`) and caught
//! its half immediately; this is the same class of drift, one layer down, previously unguarded.
//!
//! These tests read the real `MAX_AREA_*` / `SIZE_MULTIPLE_14B` constants out of the pinned
//! backend, so a manifest that gets ahead of the engine — OR a `runtime-*` pin bump that gets
//! ahead of the manifest — is RED here rather than a silent job-time reject in production.
//!
//! They run on whichever backend the current lane compiles, each verifying the binary its own
//! platform runs:
//! * **macOS** — mlx, via `runtime-macos` (`macos-mlx.yml`: `cargo test -p sceneworks-worker`).
//! * **off-mac `backend-candle`** — candle, via `runtime-cuda` (`windows-candle.yml`:
//!   `cargo test -p sceneworks-worker --features backend-candle`).
//!
//! The consts come from the same pinned tag and each backend's `config.rs` documents them as
//! mirrors of the other, so a single mapping serves both lanes.
//!
//! **Scope (sc-12409 point 4 — "does the class generalize?").** `maxPixels` is tied for every
//! video model. `requiresDimensionsMultipleOf` is tied for the wan 14B grid-16 family only —
//! that is the axis reachable through the `wan::config` import this test already holds. The
//! ltx / mochi / svd strides live in other engine crates (each a new, candle-uncompilable-here
//! import), and the 5B / scail2 `None`-stride needs a different `None == core_default` assertion;
//! extending the pinned tie to them is tracked in **sc-12587**. Until then those strides stay
//! guarded only by `ENGINE_GEOMETRY`'s literals in `sceneworks-core`.

// The pinned inference bundle for this platform — the same cfg-selected alias
// `inference_runtime.rs` uses as its composition root.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda as platform_runtime;
#[cfg(target_os = "macos")]
use runtime_macos as platform_runtime;

// `wan::config` re-exported off the platform bundle's `providers` (present under the default
// `media` feature). candle-gen-bernini / mlx-gen-bernini and the scail2 engines import this same
// `MAX_AREA_14B` rather than declaring their own, so it is authoritative for the whole 14B family.
// `SIZE_MULTIPLE_14B` is the same `pub const ... = 16` on both backends.
use platform_runtime::providers::minimax_h3::pipeline::CANVAS_MAX_PIXELS;
use platform_runtime::providers::wan::config::{MAX_AREA_14B, MAX_AREA_5B, SIZE_MULTIPLE_14B};
use serde_json::Value;

/// Every video model in the shipped `builtin.models.jsonc`. A model added or removed without
/// updating this list trips the count guard in each test, so the check cannot silently stop
/// covering a model — the same tripwire the sibling
/// [`shipped_manifest_matches_each_engines_real_geometry`] uses (`models.len() == …`).
const EXPECTED_VIDEO_IDS: &[&str] = &[
    "ltx_2_3",
    "ltx_2_3_eros",
    "svd",
    "wan_2_2",
    "wan_2_2_t2v_14b",
    "wan_2_2_i2v_14b",
    "wan_2_2_vace_fun_14b",
    "bernini",
    "scail2_14b",
    "krea_realtime_14b",
    "minimax_h3",
    "minimax_h3_ref",
];

/// The pinned engine area cap a video model's `limits.maxPixels` must equal. Derived from the
/// engine source (`candle-gen-wan` / `mlx-gen-wan` `config.rs`), NOT read back from the manifest —
/// that independence is the whole point.
///
/// * The 14B family — `wan_2_2_t2v_14b`, `wan_2_2_i2v_14b`, `wan_2_2_vace_fun_14b`, `bernini`,
///   `scail2_14b` — all validate against [`MAX_AREA_14B`]. bernini and scail2 import that exact
///   const rather than declaring their own.
/// * The TI2V-5B (`wan_2_2`) validates against [`MAX_AREA_5B`] (its z48 VAE's 32-px grid, a
///   genuinely lower budget than the 14B family's).
/// * `ltx_2_3` / `ltx_2_3_eros` / `svd` have no `maxPixels`-expressible area cap in
///   either backend, so the manifest must NOT invent one — expected `None` (key absent).
/// * `krea_realtime_14b` is a Wan-2.1-14B derivative but is deliberately NOT in the
///   [`MAX_AREA_14B`] group: `mlx-gen-krea-realtime` imports no `MAX_AREA_*` and has no
///   `reject_over_area`, and gen-core's `validate_request` carries no area term — only the
///   descriptor's per-EDGE `max_size: 1280`. Sharing the backbone is not sharing the check, so it
///   is `None` (key absent) like ltx/svd. This is the axis where "inherit the family's cap"
///   would have been wrong.
///
/// An unmapped id panics: adding a video model is a deliberate act that must derive its own cap
/// from its engine, never inherit one by default.
fn expected_max_pixels(id: &str) -> Option<u64> {
    match id {
        "wan_2_2_t2v_14b"
        | "wan_2_2_i2v_14b"
        | "wan_2_2_vace_fun_14b"
        | "bernini"
        | "scail2_14b" => Some(MAX_AREA_14B as u64),
        "wan_2_2" => Some(MAX_AREA_5B as u64),
        "ltx_2_3" | "ltx_2_3_eros" | "svd" | "krea_realtime_14b" => None,
        "minimax_h3" | "minimax_h3_ref" => Some(u64::from(CANVAS_MAX_PIXELS)),
        other => panic!(
            "video model {other:?} is not mapped to a pinned engine area cap — derive its \
             MAX_AREA_* from that model's engine `config.rs` and add it to \
             `expected_max_pixels`; do not blanket-apply a default (sc-12409)"
        ),
    }
}

/// The pinned engine stride a video model's `requiresDimensionsMultipleOf` must equal, for the
/// models whose stride this guard ties to a pinned const. `Some(n)` = assert the manifest
/// declares exactly `n`; a model absent here is not stride-tied yet (see the module note /
/// sc-12587) and remains covered only by `ENGINE_GEOMETRY`.
///
/// The wan 14B grid-16 family renders on the `SIZE_MULTIPLE_14B = 16` lattice: the three A14B ids
/// through their own engine, and `bernini` through a Wan2.2-T2V-A14B snapshot (patch 2 × vae 8).
/// `scail2_14b` and the 5B declare no stride (their `None` means "engine stride == core's default
/// floor"), and ltx / svd / mochi live in other engine crates — all deferred to sc-12587.
///
/// **`krea_realtime_14b` (epic 8431 / sc-8444) is deliberately NOT tied here**, even though it
/// renders on the same ÷16 lattice and its manifest declares 16. The tie would be a FICTION: the
/// value coincides with `SIZE_MULTIPLE_14B`, but it is not sourced from it. `mlx-gen-krea-realtime`
/// imports no `SIZE_MULTIPLE_*` — it hardcodes its own `const SPATIAL_STRIDE: usize = 8` (`t2v.rs`)
/// and takes the patch from `cfg.wan.patch_size`. So a change to *krea's* stride would NOT go red
/// here (the guard would still be reading wan's const), while a change to *wan's* const WOULD go red
/// spuriously on a model that never consulted it — a guard wrong in both directions is worse than
/// none. It therefore joins the ltx / svd / mochi "engine lives in another crate" group and stays
/// covered by `ENGINE_GEOMETRY`'s literal until sc-12587 extends the pinned tie to those crates.
fn pinned_stride(id: &str) -> Option<u64> {
    match id {
        "wan_2_2_t2v_14b" | "wan_2_2_i2v_14b" | "wan_2_2_vace_fun_14b" | "bernini" => {
            Some(SIZE_MULTIPLE_14B as u64)
        }
        _ => None,
    }
}

/// The `models` array of the SHIPPED `builtin.models.jsonc` — the exact bytes the app embeds and
/// seeds — filtered to one `type`.
fn shipped_models_of_type(kind: &str) -> Vec<Value> {
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc present in BUILTIN_MANIFESTS");
    let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
        .expect("builtin.models.jsonc parses as JSON");
    manifest
        .get("models")
        .and_then(Value::as_array)
        .expect("builtin.models.jsonc has a models array")
        .iter()
        .filter(|m| m.get("type").and_then(Value::as_str) == Some(kind))
        .cloned()
        .collect()
}

/// The shipped `type == "video"` entries.
fn shipped_video_models() -> Vec<Value> {
    shipped_models_of_type("video")
}

/// Assert the shipped video set is exactly [`EXPECTED_VIDEO_IDS`] and return each entry's `limits`
/// object keyed by id. The count guard catches a model added or removed from the manifest so this
/// test can't silently stop covering one; the per-id lookup catches a rename.
fn shipped_video_limits() -> std::collections::HashMap<String, Value> {
    let models = shipped_video_models();
    assert_eq!(
        models.len(),
        EXPECTED_VIDEO_IDS.len(),
        "a video model was added/removed in builtin.models.jsonc — update EXPECTED_VIDEO_IDS and \
         map its cap/stride to its engine const (sc-12409); do not let it go unchecked"
    );
    EXPECTED_VIDEO_IDS
        .iter()
        .map(|id| {
            let entry = models
                .iter()
                .find(|m| m.get("id").and_then(Value::as_str) == Some(*id))
                .unwrap_or_else(|| panic!("{id} present in builtin.models.jsonc"));
            let limits = entry
                .get("limits")
                .cloned()
                .unwrap_or_else(|| panic!("{id} declares a limits object"));
            ((*id).to_owned(), limits)
        })
        .collect()
}

/// For every shipped video model, `limits.maxPixels` must equal the area cap of the engine
/// PINNED by `Cargo.toml` — not a literal transcribed from it. Flip either side (drift the
/// manifest, or bump the `runtime-*` pin without the manifest) and this goes RED.
#[test]
fn manifest_max_pixels_matches_the_pinned_engine_area_cap() {
    for (id, limits) in shipped_video_limits() {
        let declared = limits.get("maxPixels").and_then(Value::as_u64);
        assert_eq!(
            declared,
            expected_max_pixels(&id),
            "{id}: manifest `limits.maxPixels` disagrees with the PINNED engine's area cap \
             (this backend: MAX_AREA_14B={MAX_AREA_14B}, MAX_AREA_5B={MAX_AREA_5B}). The catalog \
             and the `runtime-*` tag in Cargo.toml must move together — see sc-12409."
        );
    }
}

/// For the wan 14B grid-16 family, `limits.requiresDimensionsMultipleOf` must equal the engine's
/// PINNED `SIZE_MULTIPLE_14B`. Same drift class as the area cap, one axis over; the remaining
/// video strides are tracked in sc-12587.
#[test]
fn manifest_dimension_multiple_matches_the_pinned_engine_stride() {
    for (id, limits) in shipped_video_limits() {
        let Some(want) = pinned_stride(&id) else {
            continue;
        };
        let declared = limits
            .get("requiresDimensionsMultipleOf")
            .and_then(Value::as_u64);
        assert_eq!(
            declared,
            Some(want),
            "{id}: manifest `limits.requiresDimensionsMultipleOf` disagrees with the PINNED \
             engine stride (this backend: SIZE_MULTIPLE_14B={SIZE_MULTIPLE_14B}). The catalog and \
             the `runtime-*` tag in Cargo.toml must move together — see sc-12409."
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// sc-20573 — the IMAGE-lane sibling: envelope ∩ manifest limits
// ─────────────────────────────────────────────────────────────────────────────
//
// The two tests above tie a manifest SCALAR to a pinned engine CONST. The image lane's drift is a
// different shape: the geometry statement is not a `pub const` but an admission GATE — the route
// gate a provider hands `standard_memory_strategy_safety_check`. sc-20569 and sc-20570 both shipped
// such a gate keyed to a coordinate no product request can reach:
//
// * **sensenova** (sc-20569, engine-side) — the gate admitted exactly 1024x1024 / batch 1. The
//   shipped manifest advertises seven geometries from 1152 to 2048 per side and counts 1/2/4; none
//   of them is 1024x1024, so the permitted envelope and the advertised envelope had an EMPTY
//   intersection and every product-legal SenseNova request was refused in production.
// * **mage_flow** (sc-20570, worker-side) — the same class, one axis over: the request scope
//   demanded a calibration fingerprint the provider never emits.
//
// So the assertion here is the one sc-20573 asks for: **for every image model whose engine publishes
// a calibration-scoped admission gate, the set of shipped `limits.resolutions` x `limits.count`
// cells the gate ADMITS must be non-empty.**
//
// ## Why this drives the real gate instead of a declared envelope constant
//
// sc-20573's description proposed declaring each provider's envelope as importable constants and
// intersecting those. That would have re-created the drift it exists to catch: a declaration is a
// second copy of the rule, and the copy can agree with the manifest while the gate does not. The
// pinned bundle already exposes the gate itself weights-free —
// [`gen_core::MemoryRegistration::safety_check`] is documented as "the provider's real admission
// check, callable before weights are loaded", and it is the SAME function the loaded generator
// delegates to. Paired with the provider-owned weights-free fixtures
// ([`gen_core::MemoryBehaviorRegistration::valid_fixtures`]) and the shipped manifest's own bytes,
// the intersection can be MEASURED rather than mirrored. No engine-side declaration is needed, and
// there is no constant that can drift.
//
// It also closes the mirror the engine crates carry today: `mlx-gen-sensenova`'s and
// `mlx-gen-flux`'s own sc-20569 regression tests hand-copy `MANIFEST_RESOLUTIONS` into the engine
// repo. Those copies go stale the moment SceneWorks edits `limits.resolutions`; this test reads the
// shipped bytes.
//
// ## What it deliberately does NOT assert
//
// Only that the intersection is NON-EMPTY — the exact bar sc-20573 names, and the exact bar that
// separates "bricked for all product traffic" from "narrower than advertised". A gate that admits
// 1024x1024 but refuses counts 2 and 4 is a partial narrowing, not an outage, and is out of scope
// here. (`mlx-gen-flux`'s pre-#700 state was exactly that.) The three route-gate shapes epic
// 19048's defect-class sweep (activity-20582) classified as BY DESIGN are recorded below rather
// than skipped in silence.
//
// It also probes the ENGINE's gate, which is where sc-20569 lived. sc-20570's mage_flow defect lived
// one layer up, in `mlx_fit_gate`'s `MAGE_CALIBRATION_FINGERPRINT` pairing check, and is NOT on this
// path: its fix reads the expectation off the provider crate
// (`runtime_macos::providers::mage::model::MEMORY_CALIBRATION_FINGERPRINT`) and is guarded there by
// `mage_calibration_expectation_is_read_from_the_provider_crate`. Do not read a green here as
// coverage of the worker-side pairing constants.

use gen_core::{
    MemoryBehaviorRegistration, MemoryOptimizationAuthority, MemoryProviderContract,
    MemoryRegistration, MemoryRunContext, MemorySafetyDecision, MemoryStrategySupport,
};
use std::collections::{BTreeMap, BTreeSet};

/// Every image model in the shipped `builtin.models.jsonc`, in manifest order. Same tripwire as
/// [`EXPECTED_VIDEO_IDS`]: a model added or removed without touching this list trips the set guard,
/// so the check cannot silently stop covering a model.
const EXPECTED_IMAGE_IDS: &[&str] = &[
    "mage_flow_edit_base",
    "mage_flow_edit",
    "mage_flow_edit_turbo",
    "mage_flow_base",
    "mage_flow",
    "mage_flow_turbo",
    "z_image_turbo",
    "z_image",
    "z_image_edit",
    "qwen_image",
    "qwen_image_edit_2511",
    "qwen_image_edit_2511_lightning",
    "lens",
    "lens_turbo",
    "sensenova_u1_8b",
    "sensenova_u1_8b_infographic_v2",
    "sensenova_u1_8b_infographic_v3",
    "sensenova_u1_8b_fast",
    "sensenova_u1_8b_infographic_v2_fast",
    "sensenova_u1_8b_infographic_v3_fast",
    "flux_schnell",
    "flux_dev",
    "ideogram_4",
    "ideogram_4_turbo",
    "boogu_image",
    "boogu_image_turbo",
    "boogu_image_edit",
    "krea_2_turbo",
    "krea_2_raw",
    "flux2_klein_9b",
    "flux2_klein_9b_kv",
    "flux2_klein_9b_true_v2",
    "flux2_dev",
    "chroma1_hd",
    "chroma1_base",
    "chroma1_flash",
    "kolors",
    "sd3_5_large",
    "sd3_5_large_turbo",
    "sd3_5_medium",
    "sana_1600m",
    "sana_sprint_1600m",
    "anima_base",
    "anima_aesthetic",
    "anima_turbo",
    "sdxl",
    "realvisxl",
    "realvisxl_lightning",
    "illustrious_xl_v1",
    "illustrious_xl_v2",
    "instantid_realvisxl",
    "pulid_flux_dev",
    "bernini_image",
];

/// Route-gate shapes epic 19048's defect-class sweep (epic comment activity-20582, 2026-08-19)
/// audited and classified **by design**, recorded here so a future reader does not re-open them and
/// so this test cannot quietly start flagging them.
///
/// * **`candle-gen-wan` / `mlx-gen-ltx`** — video engines, so they are outside this image lane by
///   construction (`type == "video"`, covered by the two tests above). Their non-T2V capabilities
///   bypass the gate through `video_admission.rs`'s caller-side pre-filter, which is epic decision 3
///   / sc-18814, not a refusal.
/// * **`mlx-gen-chroma`** — `route_mode_and_references` fixes the route at `(TextToImage, 0)`. That
///   is a true STRUCTURAL claim (all three chroma ids declare only `text_to_image`) and the gate has
///   no geometry-cell narrowing at all, so its intersection is the whole manifest. It needs no
///   exception: it passes the assertion below on its own terms. Listed only so "chroma is missing"
///   is never mistaken for an omission.
///
/// The general principle: this test probes the GEOMETRY axis only. Route mode, reference count,
/// overlay, PiD and phases come from the provider's OWN weights-free fixture, so a structural
/// mode pre-filter can never be misread here as a geometry refusal.
const BY_DESIGN_PRE_FILTERS: &[&str] = &["wan", "ltx", "chroma"];

// There are no known gate defects at the inference revision currently pinned by `Cargo.toml`.
// In particular, the six shipped SenseNova ids now admit manifest cells at pin `401304976`; every
// probed provider is therefore required to pass the ordinary non-empty-intersection assertion.

/// The pinned provider id for the bespoke PuLID route on this lane. PuLID deliberately has no
/// [`crate::engines::MODEL_TABLE`] row: MLX owns a registered `pulid_flux` generator while Candle
/// owns a path-shaped provider API with the same memory provider id.
#[cfg(target_os = "macos")]
const PULID_MEMORY_PROVIDER_ID: &str = platform_runtime::providers::pulid::pulid_flux::MODEL_ID;
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const PULID_MEMORY_PROVIDER_ID: &str =
    platform_runtime::providers::pulid::memory_strategy::PROVIDER_ID;

/// The product-visible disposition of an image model that bypasses `MODEL_TABLE`.
///
/// This is deliberately explicit rather than an exclusion list. InstantID owns no calibration-
/// scoped memory gate on either lane, so there is no envelope to intersect. PuLID does own one and
/// must be dispatched to its lane-specific provider surface at the exact shipped campaign cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BespokeImageGateDispatch {
    NotCalibrationScoped,
    CalibrationScoped {
        provider_id: &'static str,
        width: u32,
        height: u32,
        count: u32,
        authority: MemoryOptimizationAuthority,
    },
}

fn bespoke_image_gate_dispatch(model_id: &str) -> Option<BespokeImageGateDispatch> {
    match model_id {
        "instantid_realvisxl" => Some(BespokeImageGateDispatch::NotCalibrationScoped),
        "pulid_flux_dev" => Some(BespokeImageGateDispatch::CalibrationScoped {
            provider_id: PULID_MEMORY_PROVIDER_ID,
            width: 1024,
            height: 1024,
            count: 1,
            authority: MemoryOptimizationAuthority::Calibrated,
        }),
        _ => None,
    }
}

/// Mutation-sensitive ownership check for both bespoke image models. This fails if either dispatch
/// disappears, if PuLID is pointed at a base-FLUX/worker telemetry id, or if its exact manifest cell
/// or evidence authority is widened. It also records the product decision that InstantID is not a
/// calibration-scoped route rather than pretending it owns an envelope that can be probed.
#[test]
fn bespoke_image_gate_dispatch_is_complete_and_exact() {
    assert_eq!(
        bespoke_image_gate_dispatch("instantid_realvisxl"),
        Some(BespokeImageGateDispatch::NotCalibrationScoped),
        "InstantID has no calibration-scoped memory gate on either lane"
    );
    assert_eq!(
        bespoke_image_gate_dispatch("pulid_flux_dev"),
        Some(BespokeImageGateDispatch::CalibrationScoped {
            provider_id: "pulid_flux",
            width: 1024,
            height: 1024,
            count: 1,
            authority: MemoryOptimizationAuthority::Calibrated,
        }),
        "PuLID must reach its provider-owned calibration surface at the exact shipped cell"
    );
    for id in ["instantid_realvisxl", "pulid_flux_dev"] {
        assert!(EXPECTED_IMAGE_IDS.contains(&id), "{id} must remain shipped");
        assert_eq!(
            crate::engines::engine_id_for_model(id),
            None,
            "{id} gained a MODEL_TABLE row; delete its bespoke dispatch and let the ordinary \
             registry sweep own it"
        );
    }
}

/// The EXACT set of shipped image models the off-mac `backend-candle` lane probes, as measured on
/// the pinned CUDA bundle (`cargo test -p sceneworks-worker --features backend-candle --lib
/// pinned_engine_geometry -- --nocapture`, reconciled at inference rev `401304976`).
///
/// This is the vacuity guard with teeth. `!probed.is_empty()` stayed green after 21 of these 22 fell
/// out of the sweep; set equality does not. It is a TRIPWIRE, not a spec — the same idiom as
/// [`EXPECTED_IMAGE_IDS`]. A provider that legitimately gains or loses a weights-free gate+fixture
/// pair in a pin bump moves this list *deliberately*, in the same commit, with the census on stderr
/// as the evidence. What it forbids is that movement happening silently.
///
/// Everything else on this lane is `unprobed` because `candle-gen-*` publishes no weights-free
/// gate+fixture pair for it (sensenova, sdxl, sana, sd3, chroma, kolors, anima, boogu, ideogram,
/// bernini), or because its bespoke route is explicitly non-calibration-scoped (InstantID).
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
const EXPECTED_PROBED_IMAGE_MODELS: &[&str] = &[
    "flux2_dev",
    "flux2_klein_9b",
    "flux2_klein_9b_kv",
    "flux2_klein_9b_true_v2",
    "flux_dev",
    "flux_schnell",
    "krea_2_raw",
    "krea_2_turbo",
    "lens",
    "lens_turbo",
    "mage_flow",
    "mage_flow_base",
    "mage_flow_edit",
    "mage_flow_edit_base",
    "mage_flow_edit_turbo",
    "mage_flow_turbo",
    "pulid_flux_dev",
    "qwen_image",
    "qwen_image_edit_2511",
    "qwen_image_edit_2511_lightning",
    "z_image",
    "z_image_edit",
    "z_image_turbo",
];

/// Candle lane: exact set equality against the measured census.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn assert_probe_census(
    probed: &BTreeSet<&str>,
    unprobed: &BTreeMap<&str, &'static str>,
    gate_count: usize,
    behavior_count: usize,
) {
    let expected: BTreeSet<&str> = EXPECTED_PROBED_IMAGE_MODELS.iter().copied().collect();
    assert_eq!(
        *probed, expected,
        "sc-20573: this lane's probed census moved. The bundle publishes {gate_count} memory \
         registrations and {behavior_count} behavior registrations. A model that DISAPPEARED from \
         the census stopped being covered (a renamed provider id, or a registration that no longer \
         publishes a weights-free gate+fixture pair) — that is the silent-coverage-loss this guard \
         exists for, and it is the same rot path that would hollow out \
         the required image-model census. A model that APPEARED is new coverage: confirm it, then add it \
         here in the same commit. Unprobed: {unprobed:?}"
    );
}

/// Models the macOS/MLX lane MUST probe. Deliberately a FLOOR, not the exact census.
///
/// The exact mlx census cannot be measured from the Windows dev box this guard was extended on —
/// MLX does not build there at all — and an mlx census derived by reading `mlx-gen-*` registration
/// sources would be a guess whose only failure mode is reddening the hosted macOS lane on a change
/// that broke nothing. So the mlx lane asserts the part that is PROVABLE off-Mac instead:
///
/// * PuLID must be probed through its explicit provider dispatch; and
/// * these six SenseNova ids must be probed. All six resolve (via `MODEL_TABLE`) to two engines;
///   `mlx-gen-sensenova` registers both a `MemoryRegistration`
///   and a `MemoryBehaviorRegistration` for each, its fixtures come from gen-core's shared
///   `standard_memory_behavior_context` (fixed at 1024x1024 / batch 1), and the engine's own
///   `registered_safety_check(... &fixtures[0].context) == Accept` test pins that the gate admits
///   them unmodified. So `probe_count > 0` holds by construction at the pinned rev;
///
/// **To tighten this to set equality** (preferred, and cheap): run the macOS lane once with
/// `cargo test -p sceneworks-worker --lib pinned_engine_geometry -- --nocapture`, paste the printed
/// census here, and switch this lane's [`assert_probe_census`] to the candle lane's `assert_eq!`.
#[cfg(target_os = "macos")]
const REQUIRED_PROBED_IMAGE_MODELS: &[&str] = &[
    "pulid_flux_dev",
    "sensenova_u1_8b",
    "sensenova_u1_8b_fast",
    "sensenova_u1_8b_infographic_v2",
    "sensenova_u1_8b_infographic_v2_fast",
    "sensenova_u1_8b_infographic_v3",
    "sensenova_u1_8b_infographic_v3_fast",
];

/// macOS lane: the documented weaker form — a required floor rather than set equality (see
/// [`REQUIRED_PROBED_IMAGE_MODELS`] for why, and for how to tighten it).
#[cfg(target_os = "macos")]
fn assert_probe_census(
    probed: &BTreeSet<&str>,
    unprobed: &BTreeMap<&str, &'static str>,
    gate_count: usize,
    behavior_count: usize,
) {
    // `contains(*id)` (not `contains(id)`) so the lookup borrows as `str` and does not force the
    // set's element lifetime to unify with `&'static str`.
    let missing: Vec<&str> = REQUIRED_PROBED_IMAGE_MODELS
        .iter()
        .copied()
        .filter(|id| !probed.contains(*id))
        .collect();
    assert!(
        missing.is_empty(),
        "sc-20573: {missing:?} must be probed on the MLX lane but were not. The bundle publishes \
         {gate_count} memory registrations and {behavior_count} behavior registrations; either a \
         provider id was renamed, or a registration stopped publishing a weights-free \
         gate+fixture pair. Coverage vanished silently. Unprobed: {unprobed:?}"
    );
    assert!(
        probed.len() >= REQUIRED_PROBED_IMAGE_MODELS.len(),
        "sc-20573: the MLX lane probed only {} image models, below the {} this lane is required to \
         cover. Unprobed: {unprobed:?}",
        probed.len(),
        REQUIRED_PROBED_IMAGE_MODELS.len(),
    );
}

/// Assert the shipped image set is exactly [`EXPECTED_IMAGE_IDS`] and return each entry's `limits`
/// object keyed by id.
fn shipped_image_limits() -> BTreeMap<String, Value> {
    let models = shipped_models_of_type("image");
    let shipped: BTreeSet<String> = models
        .iter()
        .map(|m| {
            m.get("id")
                .and_then(Value::as_str)
                .expect("every manifest model declares an id")
                .to_owned()
        })
        .collect();
    let expected: BTreeSet<String> = EXPECTED_IMAGE_IDS
        .iter()
        .map(|id| (*id).to_owned())
        .collect();
    assert_eq!(
        shipped, expected,
        "an image model was added/removed/renamed in builtin.models.jsonc — update \
         EXPECTED_IMAGE_IDS (sc-20573); do not let it go unchecked"
    );
    // The set comparison above collapses a DUPLICATE id — two manifest entries sharing one id
    // compare equal to the single expected entry, and the later one silently wins the map below.
    // The video sibling has always counted (`models.len() == EXPECTED_VIDEO_IDS.len()`); count here
    // too so the image lane cannot be fooled by the same shape.
    assert_eq!(
        models.len(),
        EXPECTED_IMAGE_IDS.len(),
        "builtin.models.jsonc declares {} image entries against {} EXPECTED_IMAGE_IDS while the two \
         ID SETS agree — so one side carries a DUPLICATE id, which the set comparison above cannot \
         see. Remove it; two manifest entries under one id also make the catalog's own lookup \
         order-dependent (sc-20573)",
        models.len(),
        EXPECTED_IMAGE_IDS.len(),
    );
    models
        .iter()
        .map(|m| {
            let id = m
                .get("id")
                .and_then(Value::as_str)
                .expect("every manifest model declares an id")
                .to_owned();
            let limits = m
                .get("limits")
                .cloned()
                .unwrap_or_else(|| panic!("{id} declares a limits object"));
            (id, limits)
        })
        .collect()
}

/// `limits.resolutions` as `(width, height)` pairs. A malformed entry is a manifest bug, not
/// something to skip past.
fn manifest_resolutions(id: &str, limits: &Value) -> Vec<(u32, u32)> {
    limits
        .get("resolutions")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{id}: limits.resolutions is an array"))
        .iter()
        .map(|entry| {
            let text = entry
                .as_str()
                .unwrap_or_else(|| panic!("{id}: limits.resolutions entries are strings"));
            let (width, height) = text
                .split_once('x')
                .unwrap_or_else(|| panic!("{id}: malformed resolution {text:?}"));
            (
                width
                    .parse()
                    .unwrap_or_else(|_| panic!("{id}: malformed resolution {text:?}")),
                height
                    .parse()
                    .unwrap_or_else(|_| panic!("{id}: malformed resolution {text:?}")),
            )
        })
        .collect()
}

/// `limits.count` — the advertised batch axis.
fn manifest_counts(id: &str, limits: &Value) -> Vec<u32> {
    limits
        .get("count")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{id}: limits.count is an array"))
        .iter()
        .map(|entry| {
            entry
                .as_u64()
                .unwrap_or_else(|| panic!("{id}: limits.count entries are integers"))
                as u32
        })
        .collect()
}

/// Every authority a real SceneWorks admission can present. `Calibrated` is the measured claim;
/// `Estimated` and `Resident` are `AdmissionPath::Legacy` in the fit gate — an out-of-envelope or
/// stale-identity request that the caller has ALREADY told the provider is not riding the measured
/// ladder. A gate that refuses all three at every advertised cell is bricked, which is the sc-20569
/// production outage exactly.
const ADMISSION_AUTHORITIES: [MemoryOptimizationAuthority; 3] = [
    MemoryOptimizationAuthority::Calibrated,
    MemoryOptimizationAuthority::Estimated,
    MemoryOptimizationAuthority::Resident,
];

/// Usable probes for one provider surface: provider-owned weights-free contexts the gate ADMITS
/// unmodified, so any refusal after moving only `width`/`height`/`batch` is attributable to the
/// geometry envelope and to nothing else. One per rung the contract declares `Implemented`.
///
/// Taking the route from the provider's OWN fixture — rather than synthesizing a text-to-image
/// context here — is what keeps a structural mode/overlay pre-filter (the wan / ltx / chroma class)
/// from being misread as a geometry refusal.
fn geometry_probes(
    registration: &MemoryRegistration,
    behavior: &MemoryBehaviorRegistration,
    spec: &gen_core::LoadSpec,
    contract: &MemoryProviderContract,
) -> Vec<MemoryRunContext> {
    let mut probes = Vec::new();
    for capability in &contract.strategies {
        if capability.support != MemoryStrategySupport::Implemented {
            continue;
        }
        let Ok(fixtures) = (behavior.valid_fixtures)(spec, contract, capability.strategy) else {
            // A provider may decline to build a fixture for a rung on this surface. That is the
            // provider's own statement about its route, not a geometry refusal.
            continue;
        };
        for fixture in fixtures {
            if (registration.safety_check)(spec, contract, &fixture.context)
                == MemorySafetyDecision::Accept
            {
                probes.push(fixture.context);
            }
        }
    }
    probes
}

fn require_pulid_manifest_cell(
    id: &str,
    limits: &Value,
    width: u32,
    height: u32,
    count: u32,
) -> Result<(), String> {
    let resolutions = manifest_resolutions(id, limits);
    let counts = manifest_counts(id, limits);
    if !resolutions.contains(&(width, height)) || !counts.contains(&count) {
        return Err(format!(
            "{id}: bespoke PuLID provider requires the exact {width}x{height} count {count} \
             intersection, but the shipped manifest advertises {resolutions:?} x {counts:?}"
        ));
    }
    Ok(())
}

/// MLX owns an ordinary provider registration for PuLID even though SceneWorks deliberately keeps
/// the model out of `MODEL_TABLE`. Drive that registration's weights-free contract and provider-
/// owned fixture directly so this remains the same gate the production generator delegates to.
#[cfg(target_os = "macos")]
fn probe_bespoke_pulid_manifest_cell(
    id: &str,
    limits: &Value,
    provider_id: &str,
    width: u32,
    height: u32,
    count: u32,
    authority: MemoryOptimizationAuthority,
) -> Result<String, String> {
    require_pulid_manifest_cell(id, limits, width, height, count)?;
    let registry = platform_runtime::providers::pulid::provider_registry()
        .map_err(|error| format!("{id}: build MLX PuLID registry: {error}"))?;
    let registration = registry
        .memory_strategy_registrations()
        .find(|registration| registration.provider_id == provider_id)
        .ok_or_else(|| {
            format!("{id}: MLX PuLID registry has no memory registration {provider_id:?}")
        })?;
    let behavior = registry
        .memory_behavior_registrations()
        .find(|registration| registration.provider_id == provider_id)
        .ok_or_else(|| {
            format!("{id}: MLX PuLID registry has no behavior registration {provider_id:?}")
        })?;
    let surfaces = registry
        .memory_contract_surfaces()
        .map_err(|error| format!("{id}: build MLX PuLID weights-free surfaces: {error}"))?;

    let mut checked = 0_usize;
    let mut refusals = Vec::new();
    for surface in &surfaces {
        if surface.contract.provider_id != provider_id {
            continue;
        }
        for mut context in geometry_probes(registration, behavior, &surface.spec, &surface.contract)
        {
            checked += 1;
            context.optimization_authority = authority;
            context.geometry.width = width;
            context.geometry.height = height;
            context.geometry.batch = count;
            match (registration.safety_check)(&surface.spec, &surface.contract, &context) {
                MemorySafetyDecision::Accept => {
                    return Ok(format!(
                        "{width}x{height} count {count} {authority:?} via {provider_id}"
                    ));
                }
                MemorySafetyDecision::Reject { reason } => {
                    if refusals.len() < 4 {
                        refusals.push(reason);
                    }
                }
            }
        }
    }
    Err(format!(
        "{id}: MLX provider {provider_id:?} supplied {checked} accepted weights-free fixtures, but \
         refused the exact shipped {width}x{height} count {count} {authority:?} cell: {refusals:?}"
    ))
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn write_pulid_safetensors_fixture(path: &std::path::Path) -> Result<(), String> {
    std::fs::create_dir_all(
        path.parent()
            .ok_or_else(|| format!("{} has no parent", path.display()))?,
    )
    .map_err(|error| format!("create {}: {error}", path.display()))?;
    let mut header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_vec();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend([0_u8; 4]);
    std::fs::write(path, bytes).map_err(|error| format!("write {}: {error}", path.display()))
}

/// Candle owns no invented generator registration for PuLID. Its production route constructs a
/// path-shaped provider contract, so reproduce that exact public API with tiny valid safetensors;
/// this reads headers only and never constructs weights, touches CUDA, or reaches the network.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn probe_bespoke_pulid_manifest_cell(
    id: &str,
    limits: &Value,
    provider_id: &str,
    width: u32,
    height: u32,
    count: u32,
    authority: MemoryOptimizationAuthority,
) -> Result<String, String> {
    use gen_core::{MemoryBehaviorRoute, MemoryMode, MemoryStrategy};

    require_pulid_manifest_cell(id, limits, width, height, count)?;
    let temp = tempfile::tempdir().map_err(|error| format!("{id}: tempdir: {error}"))?;
    let root = temp.path().join("pulid-memory");
    for component in ["text_encoder", "text_encoder_2", "transformer", "vae"] {
        write_pulid_safetensors_fixture(
            &root.join("base").join(component).join("model.safetensors"),
        )?;
    }
    std::fs::write(
        root.join("base/transformer/config.json"),
        r#"{"quantization":{"bits":4,"group_size":64}}"#,
    )
    .map_err(|error| format!("{id}: write packed FLUX config: {error}"))?;
    for name in [
        "scrfd_10g.safetensors",
        "arcface_iresnet100.safetensors",
        "bisenet_parsing.safetensors",
    ] {
        write_pulid_safetensors_fixture(&root.join("face").join(name))?;
    }
    write_pulid_safetensors_fixture(&root.join("pulid.safetensors"))?;
    write_pulid_safetensors_fixture(&root.join("eva.safetensors"))?;

    let paths = platform_runtime::providers::pulid::PulidFluxPaths {
        flux_base: root.join("base"),
        pulid_weights: root.join("pulid.safetensors"),
        eva_weights: root.join("eva.safetensors"),
        face_dir: root.join("face"),
        adapters: Vec::new(),
    };
    let contract =
        platform_runtime::providers::pulid::memory_strategy::provider_contract(&paths)
            .map_err(|error| format!("{id}: build Candle PuLID provider contract: {error}"))?;
    if contract.provider_id != provider_id {
        return Err(format!(
            "{id}: Candle PuLID contract registered {:?}, expected {provider_id:?}",
            contract.provider_id
        ));
    }
    let tier = platform_runtime::providers::pulid::memory_strategy::resolved_numeric_tier(&paths)
        .map_err(|error| format!("{id}: resolve Candle PuLID numeric tier: {error}"))?;
    let mut context = gen_core::standard_memory_behavior_context(
        &contract,
        MemoryStrategy::StagedResidency,
        tier,
        MemoryBehaviorRoute {
            mode: MemoryMode::Other("character_image".to_owned()),
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: Some("identity".to_owned()),
        },
    )
    .map_err(|error| format!("{id}: build Candle PuLID weights-free context: {error}"))?;
    context.optimization_authority = authority;
    context.geometry.width = width;
    context.geometry.height = height;
    context.geometry.batch = count;
    match platform_runtime::providers::pulid::memory_strategy::safety_check(
        &paths, &contract, &context,
    ) {
        MemorySafetyDecision::Accept => Ok(format!(
            "{width}x{height} count {count} {authority:?} via {provider_id}"
        )),
        MemorySafetyDecision::Reject { reason } => Err(format!(
            "{id}: Candle provider {provider_id:?} refused the exact shipped {width}x{height} \
             count {count} {authority:?} cell: {reason}"
        )),
    }
}

/// **The sc-20573 assertion.** For every shipped image model whose engine publishes a
/// calibration-scoped admission gate in the PINNED bundle, at least one advertised
/// `limits.resolutions` x `limits.count` cell must be admitted by that gate.
///
/// Runs on whichever backend the current lane compiles, exactly like its video siblings: macOS
/// probes the MLX bundle's gates, off-mac `backend-candle` probes the CUDA bundle's. Weights-free
/// and geometry-only — no GPU, no model files, no network.
#[test]
fn every_calibration_scoped_image_gate_admits_a_shipped_manifest_cell() {
    let registry = crate::inference_runtime::memory_contract_surface_registry()
        .expect("the linked bundle publishes weights-free memory-contract surfaces");
    let surfaces = registry
        .memory_contract_surfaces()
        .expect("every memory registration has its paired weights-free surface");
    let gates: BTreeMap<&str, &MemoryRegistration> = registry
        .memory_strategy_registrations()
        .map(|registration| (registration.provider_id, registration))
        .collect();
    let behaviors: BTreeMap<&str, &MemoryBehaviorRegistration> = registry
        .memory_behavior_registrations()
        .map(|registration| (registration.provider_id, registration))
        .collect();

    // Census, so a lane that covers nothing is visible rather than vacuously green.
    let mut probed: BTreeSet<&str> = BTreeSet::new();
    let mut unprobed: BTreeMap<&str, &'static str> = BTreeMap::new();
    // `(engine_id, rendered failure line)`. The engine id is kept STRUCTURED rather than recovered
    // from the rendered line: the by-design pre-filter below keys off it, and a gate's refusal
    // `reason` is free prose that can mention any other family's name.
    let mut failures: Vec<(&'static str, String)> = Vec::new();

    let shipped_limits = shipped_image_limits();
    for id in EXPECTED_IMAGE_IDS {
        let limits = shipped_limits
            .get(*id)
            .unwrap_or_else(|| panic!("{id} present in builtin.models.jsonc"));
        let Some(engine_id) = crate::engines::engine_id_for_model(id) else {
            match bespoke_image_gate_dispatch(id) {
                Some(BespokeImageGateDispatch::NotCalibrationScoped) => {
                    unprobed.insert(id, "bespoke route is explicitly non-calibration-scoped");
                }
                Some(BespokeImageGateDispatch::CalibrationScoped {
                    provider_id,
                    width,
                    height,
                    count,
                    authority,
                }) => {
                    probed.insert(id);
                    match probe_bespoke_pulid_manifest_cell(
                        id,
                        limits,
                        provider_id,
                        width,
                        height,
                        count,
                        authority,
                    ) {
                        Ok(admitted) => {
                            eprintln!("sc-20573: {id} bespoke intersection admitted {admitted}");
                        }
                        Err(reason) => failures.push((provider_id, reason)),
                    }
                }
                None => {
                    unprobed.insert(id, "no MODEL_TABLE row and no bespoke gate disposition");
                }
            }
            continue;
        };
        let (Some(registration), Some(behavior)) = (
            gates.get(engine_id).copied(),
            behaviors.get(engine_id).copied(),
        ) else {
            // Either the engine is not in THIS lane's bundle (an MLX-only family on the candle
            // lane, or the reverse), or it publishes no optimized rung and therefore no executable
            // weights-free fixture. Neither is a calibration-scoped geometry gate.
            unprobed.insert(
                id,
                "engine publishes no weights-free gate+fixture pair in this lane's bundle",
            );
            continue;
        };

        let resolutions = manifest_resolutions(id, limits);
        let counts = manifest_counts(id, limits);
        let mut probe_count = 0_usize;
        let mut admitted: Option<String> = None;
        let mut refusals: Vec<String> = Vec::new();

        'surfaces: for surface in &surfaces {
            if surface.contract.provider_id != engine_id {
                continue;
            }
            for probe in geometry_probes(registration, behavior, &surface.spec, &surface.contract) {
                probe_count += 1;
                for (width, height) in &resolutions {
                    for count in &counts {
                        for authority in ADMISSION_AUTHORITIES {
                            let mut context = probe.clone();
                            context.optimization_authority = authority;
                            context.geometry.width = *width;
                            context.geometry.height = *height;
                            context.geometry.batch = *count;
                            match (registration.safety_check)(
                                &surface.spec,
                                &surface.contract,
                                &context,
                            ) {
                                MemorySafetyDecision::Accept => {
                                    admitted = Some(format!(
                                        "{width}x{height} count {count} {authority:?}"
                                    ));
                                    break 'surfaces;
                                }
                                MemorySafetyDecision::Reject { reason } => {
                                    if refusals.len() < 4 {
                                        refusals.push(format!(
                                            "{width}x{height} count {count} {authority:?}: {reason}"
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if probe_count == 0 {
            unprobed.insert(
                id,
                "no provider-owned fixture that the gate accepts unmodified — nothing to attribute \
                 a geometry refusal to",
            );
            continue;
        }
        probed.insert(id);
        if admitted.is_none() {
            failures.push((
                engine_id,
                format!(
                    "{id} (engine {engine_id}): the pinned engine's admission gate refuses EVERY \
                     advertised cell — {} resolutions x {} counts x {} authorities, all rejected. \
                     The model is bricked for all product traffic (the sc-20569 class). First \
                     refusals: {}",
                    resolutions.len(),
                    counts.len(),
                    ADMISSION_AUTHORITIES.len(),
                    refusals.join(" | ")
                ),
            ));
        }
    }

    // Visible under `--nocapture` so a reader can see WHICH models this lane actually probed
    // without having to break the test to find out.
    eprintln!(
        "sc-20573: probed {} image models on this lane ({probed:?}); {} unprobed ({unprobed:?})",
        probed.len(),
        unprobed.len(),
    );

    // A refusal from one of the sweep's by-design families is not an ordinary failure: it means the
    // 2026-08-19 audit's classification no longer holds, and the fix is to re-audit rather than to
    // widen or waive the assertion. Say so instead of burying it in the generic list.
    //
    // Matched against the STRUCTURED engine id, never the rendered line: the line carries the gate's
    // own refusal `reason`, which is free prose. A SenseNova refusal that happens to name "wan" or
    // "ltx" while explaining a route would otherwise swap in this re-audit message and send the
    // reader to audit a gate that never failed.
    for family in BY_DESIGN_PRE_FILTERS {
        if let Some((_, failure)) = failures.iter().find(|(engine, _)| engine.contains(family)) {
            panic!(
                "{failure}\n\nEpic 19048's route-gate defect-class sweep (epic comment \
                 activity-20582, 2026-08-19) classified the {family:?} gate as BY DESIGN, not a \
                 defect. An empty intersection there contradicts that audit — re-audit the gate \
                 before widening or waiving this assertion (sc-20573)."
            );
        }
    }

    let rendered_failures: Vec<&str> = failures.iter().map(|(_, line)| line.as_str()).collect();
    assert!(
        failures.is_empty(),
        "sc-20573 envelope ∩ manifest intersection ({} probed on this lane; the census is on \
         stderr, run with --nocapture):\n  EMPTY INTERSECTIONS:\n    {}",
        probed.len(),
        rendered_failures.join("\n    "),
    );

    assert_eq!(
        unprobed.get("instantid_realvisxl"),
        Some(&"bespoke route is explicitly non-calibration-scoped"),
        "InstantID must remain an explicit non-calibration disposition, not an unswept gate"
    );
    assert!(
        probed.contains("pulid_flux_dev"),
        "PuLID is calibration-scoped and must reach the bespoke provider probe on this lane"
    );

    // (3) Per-lane expected-probed census — the real vacuity guard. `!probed.is_empty()` would stay
    // green after 22 of 23 models fell out of the sweep.
    assert_probe_census(&probed, &unprobed, gates.len(), behaviors.len());
}

/// Every MiniMax-H3 resolution the catalog ADVERTISES must fit the pinned engine's per-edge
/// ceiling, not just its area budget (sc-19721).
///
/// **This is the guard the 21:9 caveat was standing in for.** `1536x672` is 1,032,192 px — byte for
/// byte what `1344x768` is — so it always satisfied `CANVAS_MAX_PIXELS`, and every area-based check
/// in this repo passed it while the engine refused it: MiniMax-H3 bounds each EDGE independently,
/// and that ceiling used to sit below 1536. For two stories the only thing standing between the
/// catalog and a canvas the engine rejects was a prose caveat in the manifest and the prompt guide,
/// with a JS test asserting the caveat's WORDING. inference #640 raised the ceiling and sc-19721's
/// pin bump brought it here, so the caveat is discharged — and rather than delete the coverage with
/// it, the claim is re-expressed as the thing it should always have been: a comparison against the
/// engine.
///
/// `max_size` is read off the REGISTERED descriptor rather than `pipeline::MAX_CANVAS_EDGE`, because
/// the descriptor field is what `gen_core::validate_request` actually enforces; the const is only
/// what the provider chose to build it from. They agree at this pin, and that agreement is asserted
/// here so neither can be substituted for the other by accident.
#[test]
fn advertised_minimax_h3_resolutions_fit_the_pinned_engine_edge_cap() {
    let descriptor = crate::inference_runtime::media_descriptor("minimax_h3")
        .expect("the pinned bundle registers minimax_h3 (sc-19721)");
    let max_edge = u64::from(descriptor.capabilities.max_size);
    assert_eq!(
        max_edge,
        u64::from(platform_runtime::providers::minimax_h3::pipeline::MAX_CANVAS_EDGE),
        "the descriptor's enforced max_size and the pipeline const have diverged; this guard must \
         follow the one the request validator reads"
    );
    assert!(
        max_edge > u64::from(CANVAS_MAX_PIXELS).isqrt(),
        "a per-edge cap at or below the square root of the area budget would make the area budget \
         unreachable — if that is ever true the model's geometry story has changed, not this test"
    );

    let mut checked = 0_usize;
    for (id, limits) in shipped_video_limits() {
        if !id.starts_with("minimax_h3") {
            continue;
        }
        let resolutions = limits
            .get("resolutions")
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("{id} advertises a limits.resolutions list"));
        assert!(!resolutions.is_empty(), "{id}: empty resolutions list");
        for entry in resolutions {
            let (width, height) = match entry {
                Value::Object(map) => (
                    map.get("width").and_then(Value::as_u64),
                    map.get("height").and_then(Value::as_u64),
                ),
                Value::String(text) => {
                    let (w, h) = text
                        .split_once('x')
                        .unwrap_or_else(|| panic!("{id}: unparseable resolution {text:?}"));
                    (w.trim().parse().ok(), h.trim().parse().ok())
                }
                other => panic!("{id}: unexpected resolution entry {other}"),
            };
            let (Some(width), Some(height)) = (width, height) else {
                panic!("{id}: resolution entry has no width/height: {entry}");
            };
            assert!(
                width.max(height) <= max_edge,
                "{id}: advertises {width}x{height}, whose long edge exceeds the PINNED engine's \
                 per-edge cap ({max_edge}). The area budget does NOT settle this — the engine \
                 bounds each edge independently. Either the catalog got ahead of the engine or the \
                 pin moved behind the catalog; they must move together (sc-17152 / sc-19721)."
            );
            assert!(
                width * height <= u64::from(CANVAS_MAX_PIXELS),
                "{id}: advertises {width}x{height}, over the engine's area budget"
            );
            checked += 1;
        }
    }
    // The loop must have had something to check: a renamed key or an entry shape this parser skips
    // would otherwise make the whole guard a vacuous pass.
    assert!(
        checked >= 10,
        "expected both MiniMax-H3 partitions to advertise their full bucket list; checked only \
         {checked} resolutions"
    );
}

/// The MiniMax-H3 install-integrity mirror in `sceneworks-core` is bound to the engine's own
/// component names (sc-19721).
///
/// `mlx_tier_completeness` transcribed `transformer` / `transformer_ref` / `text_encoder` by READING
/// the engine source, and said so: at the old pin `mlx-gen-minimax-h3` was not in the tree, so there
/// was no symbol to import and nothing could go red when the engine renamed one. The recorded
/// obligation was to bind them once the pin carried the crate. It does now, so they are bound —
/// `sceneworks-core` cannot depend on an inference crate, so the tie lives here, in the worker,
/// where the pinned bundle is already in scope.
#[test]
fn minimax_h3_component_names_are_the_pinned_engines_own() {
    use platform_runtime::providers::minimax_h3::model as engine;
    use sceneworks_core::mlx_tier_completeness::MINIMAX_H3_PARTITIONS;
    #[cfg(target_os = "macos")]
    use sceneworks_core::mlx_tier_completeness::MINIMAX_H3_TEXT_ENCODER_DIR;

    assert_eq!(
        MINIMAX_H3_PARTITIONS,
        [
            ("minimax_h3", engine::BASE_DIT_PARTITION),
            ("minimax_h3_ref", engine::REFERENCE_DIT_PARTITION),
        ],
        "the install-integrity mirror's DiT partition dirs must be the engine's own constants — a \
         rename upstream has to red here, not at load time on a user's machine"
    );
    // The staged text-encoder dir has a `pub const` on MLX (`TEXT_ENCODER_COMPONENT`) but not on
    // candle: `candle-gen-minimax-h3::model` spells it as a bare `"text_encoder"` literal inside a
    // private `REQUIRED_COMPONENT_DIRS`, so there is no symbol to bind to on that lane.
    #[cfg(target_os = "macos")]
    assert_eq!(
        MINIMAX_H3_TEXT_ENCODER_DIR,
        engine::TEXT_ENCODER_COMPONENT,
        "the staged text-encoder component name is the engine's, not a transcription"
    );
}
