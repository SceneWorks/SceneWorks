//! **The candle admission DECISION BASELINE, emitted by the gate itself** (sc-19049, epic 19048).
//!
//! Epic 19048 requirement R6 says later slices must go red on any admission decision they did not
//! enumerate and justify. That is only true if the committed baseline is the output of the code the
//! request actually reaches. A baseline re-derived in the generator that writes it is circular: it
//! reds when it disagrees with itself, and stays green while production drifts out from under it.
//!
//! So this module is the producer. It drives [`crate::candle_scalar_gate`]'s real functions —
//! `predicted_peak_gb`, `predicted_sequential_peak_gb`, `fit_decision`, `resolve_offload`,
//! `sequential_overflow_gb`, `load_plan` — over the shipped manifest corpus and writes
//! `docs/generated/candle-admission-decisions.json`.
//! `scripts/generate-candle-admission-inventory.mjs` **consumes** that file to build
//! `docs/generated/candle-admission-baseline.json`; it no longer computes the decisions itself, and
//! its retained JS mirror is reconciled row-for-row against this output by
//! `generate-candle-admission-inventory.test.mjs`.
//!
//! Regenerate with:
//!
//! ```text
//! SCENEWORKS_REGENERATE_CANDLE_ADMISSION=1 \
//!   cargo test -p sceneworks-worker candle_admission_decisions
//! npm run generate:candle-admission     # rebuild the baseline from the new decisions
//! ```
//!
//! # Not every candle route is evaluated, and the ones that are not say so
//!
//! Only the legacy scalar gate is a pure function of manifest JSON and a synthetic budget. It is
//! what the base txt2img request of every base-routed candle image model reaches, so 52 of the 63
//! routes are evaluated end to end. The remaining 11 reach a DIFFERENT gate that needs inputs no CPU
//! lane can produce, and stamping them with the scalar gate's answer would be a fabricated column —
//! decisions their gates never produced:
//!
//! * **Video routes** are admitted by the flat per-family fit errors, every one of which takes
//!   `weight_bytes: u64` measured off the RESOLVED MODEL DIR on disk (`wan_weight_bytes(engine_id,
//!   model_dir)`, `wan_vace_fun_sequential_weight_bytes(model_dir, …)`). No manifest number
//!   reproduces them.
//! * **`krea_2_turbo`'s ladder** (`krea_turbo_fit_with_runtime`) takes
//!   `Option<&KreaRuntimeEvidenceContext>`, whose only non-test constructor canonicalizes an
//!   artifact root and walks the tier's component tree; `None` short-circuits to `Unverified`, so
//!   passing it would record the ABSENCE of evidence as a decision.
//! * **The Krea control ladder and the shared selector** need a registered
//!   `MemoryProviderContract`, resolved from a `WeightsSource::Dir`; without one
//!   `fit_ladder_for_entry_with_runtime` returns `Unknown` immediately.
//!
//! Those rows carry `resolution: "not_evaluated"` with the gate that owns them and the exact input
//! that is missing — the same shape the MLX rows use for their own deferral to sc-19050 — and they
//! carry NO geometry or budget axis, because a row that does not respond to a coordinate must not
//! publish one.
//!
//! # Why the sequential-capability input is enumerated rather than resolved
//!
//! `base.rs`'s candle txt2img gate takes its `sequential_capable` from
//! [`crate::mlx_fit_gate::engine_supports_sequential`], a query against the LINKED provider bundle's
//! descriptor (sc-12130, whose comment explicitly forbids maintaining a second engine-id allowlist
//! in the worker). The manifest's advisory `candle.supportsSequentialOffload` key is NOT that input
//! and production never reads it.
//!
//! The linked bundle differs per build: macOS links the MLX providers, `backend-candle` links the
//! CUDA catalog, and every other build links none. The candle answer therefore exists only on a
//! bundle that needs a CUDA toolchain to link, and committing the MLX bundle's answer — or the
//! manifest key — as "the" capability would be exactly the fabricated column this baseline exists to
//! prevent. So each evaluated row records the plan for BOTH values of the input
//! (`loadPlanSequentialCapable` / `loadPlanSequentialIncapable`). The decision surface is complete,
//! nothing is invented, and the day the capability resolves the row narrows to one branch — a
//! visible, enumerable diff, which is what R6 asks for.
//!
//! [`sequential_capability_divergence`] records what the LINKED bundle actually says next to the
//! manifest key, so the divergence is measured on whatever lane runs rather than asserted here.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::{json, Map, Value};

use crate::candle_scalar_gate::{
    fit_decision, fit_decision_label, load_plan, predicted_peak_gb_for_request,
    predicted_sequential_peak_gb_for_request, scalar_peak_class, VramBudget, HEADROOM_GB,
};
use crate::estimate_synthesis::DeclaredScalarClass;
use crate::fit_gate::resolve_offload;

/// The generated inventory supplies the route UNIVERSE and each route's mechanism classification.
/// Reading it here rather than re-deriving the universe is deliberate: `memory_route_registry.rs`'s
/// sc-19049 tests already reconcile that file against the routing catalog as CODE, so a route this
/// module never sees is a failure THERE, not a silent omission here.
const INVENTORY: &str = include_str!("../../../docs/generated/candle-admission-inventory.json");

/// The committed decisions this module both produces and verifies.
const DECISIONS: &str = include_str!("../../../docs/generated/candle-admission-decisions.json");

const DECISIONS_RELATIVE_PATH: &str = "docs/generated/candle-admission-decisions.json";
const REGENERATE_ENV: &str = "SCENEWORKS_REGENERATE_CANDLE_ADMISSION";

/// Tier axis. Must equal `scripts/lib/manifest-memory-declarations.mjs`'s `TIER_ORDER`; the Node
/// test pins the two together so a tier added on one side cannot quietly halve the corpus.
const TIERS: &[&str] = &["bf16", "q8", "q4", "nvfp4"];

/// Synthetic budgets (GB), `free_gb == total_gb` — a cold card, the shape `vram_gate`'s own unit
/// tests use.
const BUDGETS_GB: &[f64] = &[12.0, 24.0, 48.0, 96.0];

/// Square rungs spanning the calibration ladder.
const IMAGE_GEOMETRIES: &[(u32, u32, u32)] = &[(1024, 1024, 1), (1536, 1536, 1), (2048, 2048, 1)];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Resolution {
    /// The row is the output of a gate this lane can drive end to end.
    Evaluated,
    /// The gate that owns this route needs an input no CPU lane can produce. The row records the
    /// gate and the missing input instead of another gate's answer.
    NotEvaluated,
}

struct RouteGate {
    /// The gate the route's BASE (unconditioned txt2img / base video) request reaches.
    gate: String,
    resolution: Resolution,
    /// For `NotEvaluated`: the exact input that cannot be produced here.
    missing_input: Option<&'static str>,
}

/// Which gate a route's BASE request reaches.
///
/// "Base" is load-bearing. A row here is keyed by model id, and a model can be an admission
/// coordinate twice: `z_image` has an ordinary candle txt2img route AND is served by
/// `CandleImageLane::ZImageControl`. The txt2img decision is the one this row records, so an overlay
/// lane must not displace it — `base_routed` (the routing catalog's own
/// `candle_routed_image_models()` oracle, read as CODE) is what separates a model that merely ALSO
/// has an overlay from one like `instantid_realvisxl` or `pulid_flux_dev` that has no base candle
/// route at all and is only ever reached through its bespoke lane.
///
/// Ordering within the base path is the dispatch order in `image_jobs/base.rs`: the Krea turbo
/// ladder is consulted BEFORE the scalar gate and short-circuits it, so a route carrying
/// `krea_turbo_fit` is not a scalar-gate route even though it also publishes `vramGbByTier`.
fn route_gate(
    modality: &str,
    mechanisms: &[String],
    video_fit_symbol: Option<&str>,
    base_routed: bool,
) -> RouteGate {
    let has = |id: &str| mechanisms.iter().any(|m| m == id);

    if modality == "video" {
        return RouteGate {
            gate: video_fit_symbol.unwrap_or("none").to_owned(),
            resolution: Resolution::NotEvaluated,
            missing_input: Some(
                "weight_bytes: u64, measured off the resolved model dir on disk (wan_weight_bytes / \
                 wan_vace_fun_sequential_weight_bytes); no manifest number reproduces it",
            ),
        };
    }
    if has("krea_turbo_fit") {
        return RouteGate {
            gate: "krea_turbo_fit_with_runtime".to_owned(),
            resolution: Resolution::NotEvaluated,
            missing_input: Some(
                "runtime: Option<&KreaRuntimeEvidenceContext>, built only by canonicalizing a \
                 resolved artifact root and walking the tier's component tree; None short-circuits \
                 to KreaTurboFit::Unverified",
            ),
        };
    }
    // Only when the model has NO base candle txt2img route does an overlay module own its one and
    // only admission decision. A base-routed model that also has an overlay lane keeps the txt2img
    // row; the overlay is a second coordinate, tracked by the inventory's `conditionedLanes` and
    // owned by a later slice.
    if !base_routed {
        if has("krea_control_fit") {
            return RouteGate {
                gate: "krea_control_fit::fit_ladder_for_entry_with_runtime".to_owned(),
                resolution: Resolution::NotEvaluated,
                missing_input: Some(
                    "contract: Option<&MemoryProviderContract>, resolved from a WeightsSource::Dir; \
                     None returns KreaControlFit::Unknown before any arithmetic runs",
                ),
            };
        }
        if has("conditioning_fit") {
            return RouteGate {
                gate: "conditioning_fit::decide".to_owned(),
                resolution: Resolution::NotEvaluated,
                missing_input: Some(
                    "ConditioningFootprint::from_paths(base_dir, overlays) — the base and overlay \
                     byte counts are stat()ed off disk, and the manifest publishes neither",
                ),
            };
        }
        if has("shared_selector_bespoke_override") {
            return RouteGate {
                gate: "evaluate_shared_bespoke_image".to_owned(),
                resolution: Resolution::NotEvaluated,
                missing_input: Some(
                    "spec: &LoadSpec and contract: MemoryProviderContract, both provisioned from a \
                     WeightsSource::Dir the declared-selector lookups probe",
                ),
            };
        }
    }
    // Everything else is an ordinary candle image route: `resolve_candle_image_route` did not divert
    // it to a bespoke edit/control lane and it is not on the Krea ladder, so its base txt2img request
    // reaches the scalar gate. That INCLUDES a route whose manifest publishes no candle number: the
    // gate is REACHED and returns `Unknown -> Resident` ("never block without evidence"), which is a
    // real decision and one a later slice will change the day it gives that route a number. Marking
    // it `not_evaluated` would hide the decision the request actually gets; the row instead records
    // `predictedPeakGb: null`, which says precisely that the gate found nothing to read.
    RouteGate {
        gate: "candle_scalar_gate::load_plan".to_owned(),
        resolution: Resolution::Evaluated,
        missing_input: None,
    }
}

/// Whether the named gate's decision can vary with request width/height/frames — resolved PER GATE
/// FUNCTION from the survey of its signature, including geometry that arrives inside a struct
/// parameter (`geometry: MemoryGeometry`), which a scan of the parameter tokens cannot see.
///
/// The Node generator derives the same fact from the sources and
/// `generate-candle-admission-inventory.test.mjs` reconciles the two, so this is a cross-check on a
/// derivation rather than a second declaration of it.
fn gate_is_geometry_aware(gate: &str) -> bool {
    match gate {
        // sc-19054 (epic 19048 R3): the admission peak `load_plan` compares is now computed by
        // `predicted_peak_gb_for_request(…, geometry: MemoryGeometry)` — the scalar is the peak
        // only where its `measured`/`vramMeasuredPixels` capture covers the request, the widened
        // declared floor everywhere else — so the composition this label names IS geometry-aware.
        // The Node side derives the same fact by composing `load_plan` with the prediction entry's
        // signature (`deriveGateGeometry`), and the reconciliation test pins the two together.
        "candle_scalar_gate::load_plan" => true,
        // (…, width: u32, height: u32, …) — direct scalars.
        "krea_turbo_fit_with_runtime" => true,
        // (…, frames: u32, …, width: u32, height: u32, …) — direct scalars.
        "svd_fit_error" | "mochi_fit_error" => true,
        // sc-19055 (epic 19048 R2): geometry: MemoryGeometry — STRUCT-WRAPPED, and load-bearing.
        // These were the resolution- AND frame-blind per-tier constants. They now hand the request
        // geometry to `video_admission::graded_video_peak_gb`, which asks the mechanism what the
        // manifest row may claim for that shape. For a video request the answer is always the
        // estimate-margin-widened declared floor, because `vramMeasuredPixels` is a pixels-only
        // capture; the geometry is nonetheless a real input, and the day a candle video curve is
        // packaged (sc-19057) it becomes the identity that selects a cell.
        "wan_video_fit_error"
        | "wan_video_fit_error_with_adapter_bytes"
        | "scail2_video_fit_error"
        | "scail2_video_fit_error_with_adapter_bytes" => true,
        // The numeric byte floor remains deliberately UNGRADED, but the migrated compatibility
        // route now carries its resolved request geometry as selector identity. Geometry is
        // therefore a real typed input even though it cannot widen or scale this legacy ceiling.
        "video_weights_fit_error" => true,
        // VACE-Fun shares the arithmetic but owns no approved compatibility route and therefore
        // acquires no invented request coordinates.
        "unscoped_video_weights_fit_error" => false,
        // geometry: MemoryGeometry — STRUCT-WRAPPED, invisible to a token scan.
        "krea_control_fit::fit_ladder_for_entry_with_runtime" => true,
        // ConditioningFootprint carries labels and byte counts only.
        "conditioning_fit::decide" => false,
        "none" => false,
        other => panic!(
            "candle_admission_decisions: gate {other:?} has no recorded geometry-awareness. Resolve \
             it from the gate function's own signature (including struct parameter types) and add \
             it here and to ADMISSION_GATE_GEOMETRY in the generator."
        ),
    }
}

fn round6(value: f64) -> f64 {
    (value * 1e6).round() / 1e6
}

fn json_number(value: Option<f64>) -> Value {
    match value {
        Some(value) if value.is_finite() => json!(round6(value)),
        _ => Value::Null,
    }
}

fn manifest_entries() -> BTreeMap<String, Map<String, Value>> {
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc is embedded");
    let manifest: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw))
        .expect("builtin.models.jsonc parses as JSON");
    manifest
        .get("models")
        .and_then(Value::as_array)
        .expect("the builtin manifest publishes a models array")
        .iter()
        .filter_map(|model| {
            let id = model.get("id")?.as_str()?.to_owned();
            Some((id, model.as_object()?.clone()))
        })
        .collect()
}

/// What the LINKED provider bundle says about sequential residency for each routed engine, beside
/// the manifest's advisory key. Measured on whatever lane runs rather than declared, so the record
/// is a reading and not a claim.
fn sequential_capability_divergence(
    engines: &[(String, String)],
    entries: &BTreeMap<String, Map<String, Value>>,
) -> Value {
    let mut rows = Vec::new();
    for (model_id, engine_id) in engines {
        let registry = crate::mlx_fit_gate::engine_supports_sequential(engine_id);
        let manifest = entries
            .get(model_id)
            .and_then(|entry| entry.get("candle"))
            .and_then(|candle| candle.get("supportsSequentialOffload"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if registry != manifest {
            rows.push(json!({
                "modelId": model_id,
                "engineId": engine_id,
                "linkedRegistrySays": registry,
                "manifestKeySays": manifest,
            }));
        }
    }
    json!(rows)
}

/// The identity of the provider bundle this build links, so a reading of
/// [`crate::mlx_fit_gate::engine_supports_sequential`] is never mistaken for a lane-independent fact.
fn linked_bundle() -> String {
    match crate::inference_runtime::capability_snapshot_json() {
        Some(snapshot) => format!(
            "{}/{}",
            snapshot
                .get("platform")
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
            snapshot
                .get("backend")
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
        ),
        None => "unlinked".to_owned(),
    }
}

/// Build the decision document from the real gate.
fn build_decisions() -> Value {
    let inventory: Value =
        serde_json::from_str(INVENTORY).expect("the generated inventory is valid JSON");
    let entries = manifest_entries();

    let routes = inventory["routes"]
        .as_array()
        .expect("the inventory publishes a routes array");

    let mut rows = Vec::new();
    let mut engines = Vec::new();
    let mut evaluated_routes = 0usize;
    let mut not_evaluated_routes = 0usize;

    for route in routes {
        let model_id = route["modelId"].as_str().expect("route without a modelId");
        let engine_id = route["engineId"].as_str();
        let modality = route["modality"]
            .as_str()
            .expect("route without a modality");
        let mechanisms: Vec<String> = route["mechanisms"]
            .as_array()
            .map(|list| {
                list.iter()
                    .filter_map(|m| m.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();
        let video_fit_symbol = route["videoFitSymbol"].as_str();
        if let Some(engine_id) = engine_id {
            engines.push((model_id.to_owned(), engine_id.to_owned()));
        }

        // The routing catalog's own oracle, read as CODE rather than through the inventory: the
        // UNCONDITIONED base txt2img set. A model outside it is only ever reached through a bespoke
        // lane.
        let base_routed =
            sceneworks_core::jobs_store::candle_routed_image_models().contains(&model_id);
        let gate = route_gate(modality, &mechanisms, video_fit_symbol, base_routed);
        let geometry_sensitive = gate_is_geometry_aware(&gate.gate);
        let entry = entries.get(model_id);

        match gate.resolution {
            Resolution::NotEvaluated => {
                not_evaluated_routes += 1;
                for tier in TIERS {
                    rows.push(json!({
                        "modelId": model_id,
                        "engineId": engine_id,
                        "modality": modality,
                        "tier": tier,
                        "gate": gate.gate,
                        "resolution": "not_evaluated",
                        "geometrySensitive": geometry_sensitive,
                        "missingInput": gate.missing_input,
                    }));
                }
            }
            Resolution::Evaluated => {
                evaluated_routes += 1;
                // A candle-routed id with no manifest entry at all is a recorded divergence, not an
                // impossibility. The gate reads an empty object and returns None, exactly as it does
                // in production for that id.
                let empty = Map::new();
                let entry = entry.unwrap_or(&empty);
                for tier in TIERS {
                    for (width, height, frames) in IMAGE_GEOMETRIES {
                        // THE REAL GATE (sc-19054: geometry-aware — the scalar is presented as
                        // the peak only where its `measured`/`vramMeasuredPixels` capture covers
                        // this cell, the widened declared floor everywhere else). Everything below
                        // this line is `candle_scalar_gate`'s own code.
                        let geometry = gen_core::MemoryGeometry {
                            width: *width,
                            height: *height,
                            batch: 1,
                            frames: *frames,
                            reference_count: 0,
                        };
                        let needed = predicted_peak_gb_for_request(entry, tier, 0, geometry);
                        let sequential =
                            predicted_sequential_peak_gb_for_request(entry, tier, 0, geometry);
                        let scalar_class =
                            match scalar_peak_class(entry, "vramGbByTier", tier, geometry) {
                                DeclaredScalarClass::MeasuredPeak => "measured_peak",
                                DeclaredScalarClass::DeclaredFloor => "declared_floor",
                            };
                        for budget_gb in BUDGETS_GB {
                            let budget = Some(VramBudget {
                                free_gb: *budget_gb,
                                total_gb: *budget_gb,
                            });
                            let capable = resolve_offload(fit_decision(needed, budget), true);
                            let incapable = resolve_offload(fit_decision(needed, budget), false);
                            rows.push(json!({
                                "modelId": model_id,
                                "engineId": engine_id,
                                "modality": modality,
                                "tier": tier,
                                "gate": gate.gate,
                                "resolution": "evaluated",
                                "width": width,
                                "height": height,
                                "frames": frames,
                                "budgetGb": budget_gb,
                                "scalarClass": scalar_class,
                                "predictedPeakGb": json_number(needed),
                                "predictedSequentialPeakGb": json_number(sequential),
                                "fitDecisionSequentialCapable": fit_decision_label(&capable),
                                "fitDecisionSequentialIncapable": fit_decision_label(&incapable),
                                "loadPlanSequentialCapable":
                                    load_plan(needed, sequential, budget, true).label(),
                                "loadPlanSequentialIncapable":
                                    load_plan(needed, sequential, budget, false).label(),
                                "geometrySensitive": geometry_sensitive,
                            }));
                        }
                    }
                }
            }
        }
    }

    json!({
        "schemaVersion": 1,
        "story": "sc-19049",
        "generatedBy": format!(
            "{REGENERATE_ENV}=1 cargo test -p sceneworks-worker candle_admission_decisions"
        ),
        "producedBy": {
            "module": "crates/sceneworks-worker/src/candle_admission_decisions.rs",
            "gateModule": "crates/sceneworks-worker/src/candle_scalar_gate.rs",
            "entryPoints": [
                "predicted_peak_gb_for_request",
                "predicted_sequential_peak_gb_for_request",
                "scalar_peak_class",
                "fit_decision",
                "fit_gate::resolve_offload",
                "sequential_overflow_gb",
                "load_plan",
            ],
            "headroomGb": HEADROOM_GB,
            "routeUniverse": "docs/generated/candle-admission-inventory.json",
        },
        "corpus": {
            "tiers": TIERS,
            "budgetsGb": BUDGETS_GB,
            "imageGeometries": IMAGE_GEOMETRIES
                .iter()
                .map(|(width, height, frames)| json!({
                    "width": width, "height": height, "frames": frames
                }))
                .collect::<Vec<_>>(),
        },
        "sequentialCapability": {
            "productionInput": "crate::mlx_fit_gate::engine_supports_sequential(engine_id) \
                                (image_jobs/base.rs, sc-12130)",
            "manifestKeyIsNotTheInput": true,
            "resolution": "enumerated_both_ways",
            "why": "The input is a query against the LINKED provider bundle's descriptor, which \
                    differs per build (macOS links MLX, backend-candle links the CUDA catalog, every \
                    other build links none). The candle answer exists only on a bundle that needs a \
                    CUDA toolchain to link, so each evaluated row records the plan for BOTH values \
                    rather than committing another bundle's answer or the manifest's advisory key.",
        },
        "summary": {
            "rows": rows.len(),
            "evaluatedRoutes": evaluated_routes,
            "notEvaluatedRoutes": not_evaluated_routes,
        },
        "rows": rows,
    })
}

fn rendered_decisions() -> String {
    format!(
        "{}\n",
        serde_json::to_string_pretty(&build_decisions()).expect("decisions serialize")
    )
}

fn decisions_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(DECISIONS_RELATIVE_PATH)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The producer/verifier. Under `SCENEWORKS_REGENERATE_CANDLE_ADMISSION=1` it WRITES the
    /// artifact; otherwise it fails on any drift between the committed file and what the gate
    /// produces right now. This is the assertion that makes the baseline non-circular: the numbers
    /// in `docs/generated/candle-admission-baseline.json` are built from this file, and this file is
    /// built by `candle_scalar_gate`.
    #[test]
    fn committed_decisions_match_the_gate() {
        let rendered = rendered_decisions();
        if std::env::var(REGENERATE_ENV).is_ok_and(|value| value == "1") {
            // Rendered successfully before any write — never truncate the checked-in artifact with
            // a partial or failed generation.
            std::fs::write(decisions_path(), &rendered).expect("write the decisions artifact");
            return;
        }
        assert_eq!(
            DECISIONS.replace("\r\n", "\n"),
            rendered,
            "{DECISIONS_RELATIVE_PATH} no longer matches what the candle scalar gate produces. \
             A gate change is exactly what epic 19048 R6 wants surfaced: enumerate and justify the \
             decision change, then regenerate with \
             `{REGENERATE_ENV}=1 cargo test -p sceneworks-worker candle_admission_decisions` \
             followed by `npm run generate:candle-admission`."
        );
    }

    /// Every candle route in the inventory is represented, and every row states which gate produced
    /// it. A route that quietly vanished from the decisions would leave the baseline claiming
    /// coverage it does not have.
    #[test]
    fn every_inventory_route_is_represented_with_a_named_gate() {
        let decisions = build_decisions();
        let inventory: Value = serde_json::from_str(INVENTORY).expect("inventory parses");
        let expected: std::collections::BTreeSet<&str> = inventory["routes"]
            .as_array()
            .expect("routes")
            .iter()
            .filter_map(|route| route["modelId"].as_str())
            .collect();
        let mut seen = std::collections::BTreeSet::new();
        for row in decisions["rows"].as_array().expect("rows") {
            let model = row["modelId"].as_str().expect("row without a modelId");
            seen.insert(model);
            let gate = row["gate"].as_str().expect("row without a gate");
            assert!(!gate.is_empty(), "{model}: empty gate label");
            let resolution = row["resolution"]
                .as_str()
                .expect("row without a resolution");
            assert!(
                matches!(resolution, "evaluated" | "not_evaluated"),
                "{model}: unknown resolution {resolution}"
            );
            if resolution == "not_evaluated" {
                assert!(
                    row["missingInput"].as_str().is_some_and(|s| !s.is_empty()),
                    "{model}: a not_evaluated row must name the input it cannot produce"
                );
                // The whole point of the distinct resolution: no fabricated coordinate response.
                assert!(
                    row.get("width").is_none() && row.get("budgetGb").is_none(),
                    "{model}: a not_evaluated row must not publish a geometry or budget axis"
                );
            }
        }
        assert_eq!(
            seen, expected,
            "the decision baseline and the inventory disagree about the candle route universe"
        );
    }

    /// **The sc-19054 successor of `a_geometry_blind_gate_yields_one_plan_across_the_geometry_axis`**
    /// (that test's own doc said this story has to break it — this is the deliberate replacement,
    /// enumerated in the story's decision ledger). The evaluated scalar gate is now
    /// geometry-aware: every evaluated row says so, and the PREDICTION moves with the geometry
    /// axis exactly where the scalar's declared measurement stops covering the request —
    /// `measured: true` routes captured at 1024² predict the raw peak at 1024² and the
    /// margin-widened declared floor at 1536²/2048², while `measured: false` routes are
    /// floor-classed at every geometry (uniformly widened, still geometry-INVARIANT in value).
    #[test]
    fn the_scalar_gate_grades_the_geometry_axis_through_the_declared_measurement() {
        let decisions = build_decisions();
        let entries = manifest_entries();
        let mut peaks: BTreeMap<(String, String, String), BTreeMap<u64, f64>> = BTreeMap::new();
        for row in decisions["rows"].as_array().expect("rows") {
            if row["resolution"].as_str() != Some("evaluated") {
                continue;
            }
            assert_eq!(
                row["geometrySensitive"].as_bool(),
                Some(true),
                "{}: since sc-19054 the evaluated scalar gate is geometry-aware",
                row["modelId"]
            );
            let class = row["scalarClass"].as_str().expect("scalarClass");
            assert!(
                matches!(class, "measured_peak" | "declared_floor"),
                "{}: unknown scalarClass {class}",
                row["modelId"]
            );
            let Some(peak) = row["predictedPeakGb"].as_f64() else {
                continue;
            };
            let pixels =
                row["width"].as_u64().expect("width") * row["height"].as_u64().expect("height");
            peaks
                .entry((
                    row["modelId"].as_str().expect("modelId").to_owned(),
                    row["tier"].as_str().expect("tier").to_owned(),
                    class.to_owned(),
                ))
                .or_default()
                .insert(pixels, peak);
        }
        assert!(!peaks.is_empty(), "no evaluated rows to check");

        // The axis is LIVE: at least one measured=true route must predict a strictly larger peak
        // beyond its declared capture geometry than inside it, and inside it the prediction must
        // be byte-for-byte the pre-story raw scalar + headroom.
        let mut witnessed_split = false;
        for ((model_id, tier, _), by_pixels) in peaks
            .iter()
            .filter(|((_, _, class), _)| class == "measured_peak")
        {
            let entry = entries.get(model_id).expect("evaluated model has an entry");
            let declared = entry["candle"]["vramMeasuredPixels"]
                .as_u64()
                .expect("a measured_peak class requires a declared capture geometry");
            let raw = entry["candle"]["vramGbByTier"][tier.as_str()]
                .as_f64()
                .expect("a measured_peak class requires the tier's own row")
                + HEADROOM_GB;
            for (pixels, peak) in by_pixels {
                assert!(
                    *pixels <= declared,
                    "measured_peak beyond the declared capture"
                );
                assert_eq!(
                    round6(raw),
                    *peak,
                    "{model_id}/{tier}: inside the declared capture the prediction is the raw \
                     measured scalar, byte-for-byte the pre-sc-19054 number"
                );
            }
            // The same model+tier's floor-classed cells (beyond the capture) must sit strictly
            // above the covered raw peak — the widened declared floor.
            if let Some(floor_cells) =
                peaks.get(&(model_id.clone(), tier.clone(), "declared_floor".to_owned()))
            {
                for (pixels, peak) in floor_cells {
                    assert!(
                        *pixels > declared,
                        "{model_id}/{tier}: a floor-classed cell inside the declared capture \
                         means the measured flag or the pixel comparison was dropped"
                    );
                    assert!(
                        *peak > round6(raw),
                        "{model_id}/{tier}: beyond the declared capture the scalar is a floor \
                         and must be widened above the raw peak (got {peak} vs raw {raw})"
                    );
                    witnessed_split = true;
                }
            }
        }
        assert!(
            witnessed_split,
            "the corpus must witness at least one measured=true route split across the geometry \
             axis — a corpus that cannot is not exercising sc-19054 at all"
        );
    }

    /// The baseline must never take its sequential-capability input from the manifest. This asserts
    /// the structural property (no resolved `sequentialCapable` column) and RECORDS the measured
    /// divergence between the linked bundle and the manifest key on whatever lane runs.
    #[test]
    fn sequential_capability_is_never_sourced_from_the_manifest_key() {
        let decisions = build_decisions();
        for row in decisions["rows"].as_array().expect("rows") {
            assert!(
                row.get("sequentialCapable").is_none(),
                "{}: a resolved sequentialCapable column would be a fabricated input — production \
                 reads mlx_fit_gate::engine_supports_sequential, which this lane cannot answer for \
                 the candle bundle",
                row["modelId"]
            );
            if row["resolution"].as_str() == Some("evaluated") {
                assert!(row.get("loadPlanSequentialCapable").is_some());
                assert!(row.get("loadPlanSequentialIncapable").is_some());
            }
        }
        assert_eq!(
            decisions["sequentialCapability"]["manifestKeyIsNotTheInput"].as_bool(),
            Some(true)
        );

        // Measured, not asserted: on a lane that links a bundle, this is the set of routes where the
        // manifest's advisory key and the production input disagree. It is printed rather than
        // pinned because the answer is a property of the linked bundle, not of this tree.
        let inventory: Value = serde_json::from_str(INVENTORY).expect("inventory parses");
        let engines: Vec<(String, String)> = inventory["routes"]
            .as_array()
            .expect("routes")
            .iter()
            .filter_map(|route| {
                Some((
                    route["modelId"].as_str()?.to_owned(),
                    route["engineId"].as_str()?.to_owned(),
                ))
            })
            .collect();
        let divergence = sequential_capability_divergence(&engines, &manifest_entries());
        eprintln!(
            "candle sequential-capability divergence on bundle {}: {}",
            linked_bundle(),
            serde_json::to_string(&divergence).expect("divergence serializes")
        );
    }

    /// **SC-19059 owner policy: pin the fourteen evidence-free image routes as Unverified.**
    ///
    /// The source-backed classes now reach the selector through an honest resident-only
    /// `compatibility_default`: fourteen scalar routes plus InstantID and Bernini image. The
    /// remaining fourteen publish neither a provider contract nor a source-owned capacity truth.
    /// The approved policy sends them through the selector with no candidate and preserves resident
    /// execution, so an exact-set assertion prevents fabricated evidence or accidental disablement.
    #[test]
    fn all_image_routes_reach_the_selector_and_exactly_fourteen_are_unverified() {
        let inventory: Value = serde_json::from_str(INVENTORY).expect("inventory parses");
        let routes = inventory["routes"].as_array().expect("routes");

        let mut unreached_image = Vec::new();
        let mut unverified_image = Vec::new();
        let mut reached_without_declaration = Vec::new();
        for route in routes {
            if route["modality"].as_str() != Some("image") {
                continue;
            }
            let model_id = route["modelId"].as_str().expect("modelId");
            let reached = route["sharedSelector"]["reached"]
                .as_bool()
                .expect("sharedSelector.reached");
            let declares = route["evidence"]["declaresProviderContract"]
                .as_bool()
                .unwrap_or(false);
            // A bespoke override reaches the selector through a caller-supplied revision rather
            // than through a manifest declaration, so it is legitimately reached without one.
            let bespoke = route["sharedSelector"]["via"].as_str() == Some("bespoke_override");
            let named = route["sharedSelector"]["via"].as_str() == Some("named_revision");
            let compatibility = matches!(
                route["sharedSelector"]["via"].as_str(),
                Some(
                    "legacy_scalar_compatibility"
                        | "structural_floor_compatibility"
                        | "unverified_compatibility"
                )
            );
            if route["sharedSelector"]["via"].as_str() == Some("unverified_compatibility") {
                unverified_image.push(model_id.to_owned());
            }
            if !reached {
                assert!(
                    !declares,
                    "{model_id}: declares a provider contract but is classified unreached — the \
                     inventory's reachability key and this partition disagree"
                );
                unreached_image.push(model_id.to_owned());
            } else if !declares && !bespoke && !named && !compatibility {
                reached_without_declaration.push(model_id.to_owned());
            }
        }

        assert!(
            reached_without_declaration.is_empty(),
            "these image routes reach the selector through neither a named revision, a bespoke \
             override, a compatibility basis, nor a manifest provider contract: \
             {reached_without_declaration:?}"
        );
        assert!(
            unreached_image.is_empty(),
            "all image routes must reach the selector"
        );
        unverified_image.sort();
        assert_eq!(
            unverified_image,
            [
                "anima_aesthetic",
                "anima_base",
                "anima_turbo",
                "boogu_image_edit",
                "chroma1_base",
                "chroma1_flash",
                "chroma1_hd",
                "illustrious_xl_v1",
                "illustrious_xl_v2",
                "realvisxl",
                "realvisxl_lightning",
                "sana_1600m",
                "sana_sprint_1600m",
                "sdxl",
            ],
            "the approved evidence-free Unverified image class changed"
        );
    }
}
