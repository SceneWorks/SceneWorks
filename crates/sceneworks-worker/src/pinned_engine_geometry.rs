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
// here. (`mlx-gen-flux`'s pre-#700 state was exactly that, which is why it is not in the ledger
// below.) The three route-gate shapes epic 19048's defect-class sweep (activity-20582) classified as
// BY DESIGN are recorded below rather than skipped in silence.
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

/// Engines whose gate is KNOWN to violate the intersection invariant **at the inference rev
/// `Cargo.toml` currently pins**, with the upstream fix that is already merged on inference `main`.
///
/// This is a ledger, not a waiver. Epic 19048's terminal phase performs exactly ONE pin bump, to an
/// inference rev >= `0843ff092` (activity-20593); until that bump lands, the pinned MLX bundle still
/// carries the pre-sc-20569 SenseNova gate, whose envelope clauses refuse unconditionally instead of
/// degrading on a non-`Calibrated` authority. Asserting the invariant for these engines today would
/// red on a defect that is already fixed upstream and cannot be fixed from this repo.
///
/// So the assertion is INVERTED for them: they must STILL violate. The moment the pin moves past the
/// fix, this test goes red telling the reader to delete the entry — the ledger cannot rot into a
/// permanent hole, and the pin bump cannot silently leave a stale exception behind.
///
/// * `sensenova_u1_8b` / `sensenova_u1_8b_fast` — sc-20569. Fixed by inference PR #699
///   (`0f9808e42`, merge of `18876051e`), which is NOT an ancestor of the pinned
///   `6f47e6707589158b89d0f238c5ef56be6f1bdd7a`. All six shipped SenseNova product ids advertise the
///   same seven geometries (1152..2048 per side); the pinned gate admits only 1024x1024.
///
/// `mlx-gen-flux`'s twin (inference PR #700, `517269a1f`) is deliberately NOT here: `flux_dev`
/// advertises 1024x1024 and count 1, so even the pre-fix gate leaves a non-empty intersection. Its
/// defect was a partial narrowing (counts 2/4 and the other three resolutions), which is a real bug
/// — fixed upstream — but not the empty-intersection class this test asserts.
const PINNED_ENGINE_GATE_DEFECTS: &[&str] = &["sensenova_u1_8b", "sensenova_u1_8b_fast"];

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
    let mut failures: Vec<String> = Vec::new();
    let mut retired_ledger_entries: Vec<String> = Vec::new();

    let shipped_limits = shipped_image_limits();
    for id in EXPECTED_IMAGE_IDS {
        let limits = shipped_limits
            .get(*id)
            .unwrap_or_else(|| panic!("{id} present in builtin.models.jsonc"));
        let Some(engine_id) = crate::engines::engine_id_for_model(id) else {
            unprobed.insert(id, "no MODEL_TABLE row (not an engine-backed image family)");
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

        let ledgered = PINNED_ENGINE_GATE_DEFECTS.contains(&engine_id);
        match (&admitted, ledgered) {
            (Some(_), false) => {}
            (None, true) => {}
            (Some(cell), true) => retired_ledger_entries.push(format!(
                "{id} (engine {engine_id}): the PINNED engine now admits {cell}, so the fix has \
                 landed in the pinned rev — DELETE {engine_id:?} from PINNED_ENGINE_GATE_DEFECTS \
                 (sc-20573)"
            )),
            (None, false) => failures.push(format!(
                "{id} (engine {engine_id}): the pinned engine's admission gate refuses EVERY \
                 advertised cell — {} resolutions x {} counts x {} authorities, all rejected. The \
                 model is bricked for all product traffic (the sc-20569 class). First refusals: {}",
                resolutions.len(),
                counts.len(),
                ADMISSION_AUTHORITIES.len(),
                refusals.join(" | ")
            )),
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
    for family in BY_DESIGN_PRE_FILTERS {
        if let Some(failure) = failures.iter().find(|line| line.contains(family)) {
            panic!(
                "{failure}\n\nEpic 19048's route-gate defect-class sweep (epic comment \
                 activity-20582, 2026-08-19) classified the {family:?} gate as BY DESIGN, not a \
                 defect. An empty intersection there contradicts that audit — re-audit the gate \
                 before widening or waiving this assertion (sc-20573)."
            );
        }
    }

    assert!(
        failures.is_empty() && retired_ledger_entries.is_empty(),
        "sc-20573 envelope ∩ manifest intersection ({} probed on this lane; the census is on \
         stderr, run with --nocapture):\n  EMPTY INTERSECTIONS:\n    {}\n  STALE LEDGER \
         ENTRIES:\n    {}",
        probed.len(),
        failures.join("\n    "),
        retired_ledger_entries.join("\n    "),
    );

    assert!(
        !probed.is_empty(),
        "sc-20573: no image model was probed on this lane, so the intersection check is vacuous. \
         The linked bundle publishes {} memory registrations and {} behavior registrations; either \
         the bundle stopped composing or `engine_id_for_model` no longer agrees with the registry's \
         provider ids. Unprobed: {unprobed:?}",
        gates.len(),
        behaviors.len(),
    );
}
