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
    // MiniMax-H3, both partitions (epic 17137, sc-17158). Neither is tie-able YET — see
    // [`PinnedAreaCap::NotInThePinnedBundle`].
    "minimax_h3",
    "minimax_h3_ref",
];

/// What this guard can say about one model's `limits.maxPixels`.
///
/// The two-state `Option<u64>` this used to be conflated "the manifest must declare exactly this
/// pinned value" with "the manifest must declare nothing", and had no way to express the third
/// case sc-17158 introduced: a model whose engine **is not in the pinned inference bundle at all**,
/// so there is no const to tie to and `None` would assert the opposite of the truth.
///
/// Naming it is the point. A literal quietly returned as `Some(1_032_192)` would look tied and be
/// a fiction — exactly the failure mode sc-12409 exists to prevent — so the third state is
/// explicit, carries the id of the story that closes it, and the test says out loud which models
/// it is currently only counting rather than checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PinnedAreaCap {
    /// The manifest must declare exactly this value, read from the pinned engine's own const.
    Pinned(u64),
    /// The engine has no `maxPixels`-expressible area cap, so the manifest must NOT invent one.
    EngineHasNone,
    /// The engine is not compiled into the pinned `runtime-*` bundle, so nothing here can be
    /// verified against it. The manifest value is still guarded — by `ENGINE_GEOMETRY`'s literals
    /// in `sceneworks-core` and by that crate's `minimax_h3_over_cap_canvas_is_refit_onto_the_area_budget`
    /// — just not against the shipping binary.
    NotInThePinnedBundle,
}

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
/// * `minimax_h3` / `minimax_h3_ref` declare a real, enforced area budget
///   (`CANVAS_MAX_PIXELS = 1_032_192 = 768 × 1344`, wired by sc-17152) — but the MiniMax-H3
///   provider is **not in the pinned inference bundle**: SceneWorks still pins `014134e3…` and
///   sc-18650 owns the bump, which needs an off-Mac candle capability dump. There is no
///   `providers::minimax_h3::config` to import, so this guard can only COUNT these two, not check
///   them, and says so with [`PinnedAreaCap::NotInThePinnedBundle`] rather than transcribing the
///   number and pretending it is tied. **sc-18650's pin bump must convert both to `Pinned(…)`**
///   sourced from the engine const — that conversion is the whole reason this state is named.
///
/// An unmapped id panics: adding a video model is a deliberate act that must derive its own cap
/// from its engine, never inherit one by default.
fn expected_max_pixels(id: &str) -> PinnedAreaCap {
    match id {
        "wan_2_2_t2v_14b"
        | "wan_2_2_i2v_14b"
        | "wan_2_2_vace_fun_14b"
        | "bernini"
        | "scail2_14b" => PinnedAreaCap::Pinned(MAX_AREA_14B as u64),
        "wan_2_2" => PinnedAreaCap::Pinned(MAX_AREA_5B as u64),
        "ltx_2_3" | "ltx_2_3_eros" | "svd" | "krea_realtime_14b" => PinnedAreaCap::EngineHasNone,
        "minimax_h3" | "minimax_h3_ref" => PinnedAreaCap::NotInThePinnedBundle,
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
///
/// **MiniMax-H3 (sc-17158) falls in the untied `_` arm for two independent reasons**, and both
/// have to hold for that to be right: its `SIZE_MULTIPLE = 32` is not `SIZE_MULTIPLE_14B`, and its
/// engine is not in the pinned bundle at all (see [`PinnedAreaCap::NotInThePinnedBundle`]). It
/// declares no `requiresDimensionsMultipleOf` — 32 IS core's default floor — so "not tied" and
/// "declares nothing" agree here, which is why this needs no third state the way the area cap did.
fn pinned_stride(id: &str) -> Option<u64> {
    match id {
        "wan_2_2_t2v_14b" | "wan_2_2_i2v_14b" | "wan_2_2_vace_fun_14b" | "bernini" => {
            Some(SIZE_MULTIPLE_14B as u64)
        }
        _ => None,
    }
}

/// The `models` array of the SHIPPED `builtin.models.jsonc` — the exact bytes the app embeds and
/// seeds — filtered to `type == "video"`.
fn shipped_video_models() -> Vec<Value> {
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
        .filter(|m| m.get("type").and_then(Value::as_str) == Some("video"))
        .cloned()
        .collect()
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
    let mut only_counted: Vec<String> = Vec::new();
    for (id, limits) in shipped_video_limits() {
        let declared = limits.get("maxPixels").and_then(Value::as_u64);
        let want = match expected_max_pixels(&id) {
            PinnedAreaCap::Pinned(cap) => Some(cap),
            PinnedAreaCap::EngineHasNone => None,
            PinnedAreaCap::NotInThePinnedBundle => {
                // Not skipped silently: assert the manifest DOES declare a budget (an absent one
                // would be a real regression this guard would otherwise wave through), then record
                // the id so the list below stays honest about what is unchecked.
                assert!(
                    declared.is_some(),
                    "{id}: declares no `limits.maxPixels`. Its engine is not in the pinned bundle \
                     so the value cannot be tied here, but an area budget must still be declared \
                     — see sceneworks-core's ENGINE_GEOMETRY."
                );
                only_counted.push(id.clone());
                continue;
            }
        };
        assert_eq!(
            declared, want,
            "{id}: manifest `limits.maxPixels` disagrees with the PINNED engine's area cap \
             (this backend: MAX_AREA_14B={MAX_AREA_14B}, MAX_AREA_5B={MAX_AREA_5B}). The catalog \
             and the `runtime-*` tag in Cargo.toml must move together — see sc-12409."
        );
    }

    // The set that is COUNTED but not CHECKED, pinned as a value rather than left implicit: when
    // sc-18650 bumps the inference pin and the MiniMax-H3 provider becomes importable, this goes
    // red and forces the mapping to move to `PinnedAreaCap::Pinned`. A guard that quietly shrinks
    // its own coverage is the thing sc-12409 was written about.
    only_counted.sort();
    assert_eq!(
        only_counted,
        vec!["minimax_h3".to_owned(), "minimax_h3_ref".to_owned()],
        "the set of video models whose area cap cannot be tied to the pinned engine changed"
    );
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
