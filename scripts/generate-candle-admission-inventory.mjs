#!/usr/bin/env node
/**
 * CANDLE ADMISSION ROUTE INVENTORY + DECISION BASELINE (sc-19049, epic 19048 slice 1).
 *
 * Epic 19048 converges candle and MLX memory admission on one evidence-based mechanism. Before any
 * of it moves, two things have to exist, and both have to be MECHANICAL:
 *
 *  1. **The inventory.** Every candle image/video route, and which admission gate it actually
 *     reaches. Epic 18472 is adding candle routes serially, so a hand-written snapshot is stale on
 *     arrival â€” the story rejects one explicitly. This script derives the route universe from the
 *     routing catalog (`jobs_store/routing/*.rs`), NOT from the manifest's advisory per-backend
 *     hints, and joins it to the admission surface parsed out of the worker.
 *
 *  2. **The decision baseline.** Admission decisions over a model x tier x geometry x budget
 *     corpus, committed as a diffable artifact. Later slices red on any decision change they did
 *     not enumerate and justify (epic requirement R6).
 *
 *     **The decisions are NOT computed here.** They are read from
 *     `docs/generated/candle-admission-decisions.json`, produced by
 *     `crates/sceneworks-worker/src/candle_admission_decisions.rs` driving
 *     `crates/sceneworks-worker/src/candle_scalar_gate.rs` â€” the functions the request actually
 *     reaches. A baseline re-derived by the generator that writes it is circular: it reds when it
 *     disagrees with itself and stays green while production drifts. Nothing needs a GPU either
 *     way; the scalar gate is a pure function of the manifest entry, the tier key and a synthetic
 *     `VramBudget`, which is why it was lifted out of the candle-only `vram_gate` module and can
 *     now be driven from the ordinary `cargo test` lane.
 *
 *     The JS mirror further down this file is RETAINED as a readable statement of the same law and
 *     is reconciled row-for-row against the Rust output by the test suite; it is no longer what the
 *     artifact is built from.
 *
 * Usage:
 *   node scripts/generate-candle-admission-inventory.mjs             # regenerate the artifacts
 *   node scripts/generate-candle-admission-inventory.mjs --check     # verify + fail on drift (CI)
 *   node scripts/generate-candle-admission-inventory.mjs --self-test # prove the guards actually fail
 *
 * ## Provenance: the fingerprint covers every source this script READS
 *
 * `generatedFrom.sceneWorksRevision` is a `source-tree:<sha256>` over the SEMANTIC body of every
 * entry in [`SOURCE_PATHS`], NUL-joined in declaration order â€” the same discipline as
 * `docs/generated/memory-matrix.json` (`scripts/lib/source-revision.mjs`). Unlike the matrix,
 * nothing is excluded from the hash: this artifact stamps no value back into any of its own inputs,
 * so there is no self-stamping fixed point to break.
 *
 * The invariant that makes the fingerprint meaningful is enforced, not asserted: every read goes
 * through [`readSources`], which returns the exact key set it read, and
 * `generate-candle-admission-inventory.test.mjs` pins that set equal to `Object.keys(SOURCE_PATHS)`.
 * A parser that starts reading a file nobody fingerprinted fails the test rather than silently
 * producing an artifact that cannot go stale.
 *
 * ## Terminal fingerprint closure (sc-19059)
 *
 * The inventory originally recorded that the memory-matrix fingerprint omitted production
 * admission inputs. The terminal acceptance closes that gap once, after the feature train is
 * integrated: `scripts/generate-memory-matrix.mjs` now fingerprints `candle_scalar_gate.rs`,
 * `candle_memory_strategy.rs`, `video_admission.rs`, `conditioning_fit.rs`,
 * `krea_control_fit.rs`, `sceneworks-core/src/video_memory_curves.rs`, and `payload.rs`.
 * A change to any of those inputs now rotates the matrix provenance instead of leaving a falsely
 * current artifact.
 *
 * ## Why the mechanism taxonomy is derived rather than declared
 *
 * The three facts that classify a route are each read from code or manifest data:
 *
 *   - **Does it reach the shared selector, and how?** `candle_memory_strategy.rs`'s dispatch is
 *     parsed for its named `*_REQUEST_EVIDENCE_REVISION` arms. Since sc-18456 that match ends in a
 *     catch-all (`_ => DECLARATION_REQUEST_EVIDENCE_REVISION`), so "reaches the selector" and
 *     "reaches it with a named evidence revision" are DIFFERENT questions and are reported
 *     separately. The catch-all only fires for a provider whose manifest declares a
 *     `candle.memoryStrategyContract`; without one the request never enters the shared path at all.
 *   - **Which numbers does it use?** From the manifest `candle` block: `vramGbByTier` (a measured
 *     per-tier peak), `sequentialPeakGb`, `minMemoryGb` (a declared floor), `turboFit`, `control`.
 *   - **Is the gate geometry-aware?** From EACH gate function's own signature, wherever it is
 *     defined â€” whether it takes `width`/`height`/`frames`, as scalars OR inside a struct parameter
 *     (`geometry: MemoryGeometry`, `query: VideoCurveQuery`). Resolved per SYMBOL and never unioned
 *     across a mechanism. Not asserted in prose here; parsed.
 *
 * ## The `measured` flag is a BARE BOOLEAN, not an evidence class
 *
 * Epic 18472 was expected to land an evidence-CLASS flag. What actually shipped is a bare
 * `measured: true|false` at `<model>.candle`, `<model>.candle.turboFit` and `<model>.candle.control`
 * (47 occurrences). There is no `evidenceClass`/`evidence_class` anywhere in the tree. The
 * inventory records the boolean it can actually see and reports the absence as a known gap; it does
 * not invent an enum that no producer writes.
 */

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { stripJsoncComments } from "./lib/jsonc.mjs";
import { canonicalSourceText, semanticSourceBody } from "./lib/source-revision.mjs";
import { TIER_ORDER, declaredProviders } from "./lib/manifest-memory-declarations.mjs";
import { routedLanes } from "./check-tier-integrity.mjs";
import { CANDLE_HARD_FLOOR, ESTIMATE_WIDENING_MULTIPLIER } from "./derive-ladder-margins.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const OUTPUT_INVENTORY_JSON = "docs/generated/candle-admission-inventory.json";
const OUTPUT_INVENTORY_MD = "docs/generated/candle-admission-inventory.md";
const OUTPUT_BASELINE_JSON = "docs/generated/candle-admission-baseline.json";

export const OUTPUT_PATHS = Object.freeze({
  inventoryJson: OUTPUT_INVENTORY_JSON,
  inventoryMarkdown: OUTPUT_INVENTORY_MD,
  baselineJson: OUTPUT_BASELINE_JSON,
});

/**
 * Every source this script reads, and therefore every source its fingerprint covers. Exported so
 * the tests derive the list instead of mirroring it, and so a later slice of epic 19048 can add a
 * file here and get a rotated fingerprint for free.
 *
 * Declaration order is load-bearing: it is the order the bodies are NUL-joined in before hashing.
 * Appending is safe; reordering rotates the fingerprint for no semantic reason.
 */
export const SOURCE_PATHS = Object.freeze({
  manifest: "config/manifests/builtin.models.jsonc",

  // The route UNIVERSE. The story is explicit that this reconciles against the routing catalog, not
  // the manifest's advisory `candle` block: 9 catalog entries carry a `candle` block while the
  // router serves no candle image/video lane for them, and several routed ids carry no block at all.
  routingCatalog: "crates/sceneworks-core/src/jobs_store/routing/catalog.rs",
  routingCandle: "crates/sceneworks-core/src/jobs_store/routing/candle.rs",
  routingMlx: "crates/sceneworks-core/src/jobs_store/routing/mlx.rs",

  // SceneWorks model id -> engine/provider id, which is the key `candle_memory_strategy.rs`'s
  // dispatch and `memory_route_registry.rs`'s rules are both keyed on.
  engines: "crates/sceneworks-worker/src/engines.rs",

  // The admission surface itself. `candle_scalar_gate.rs` is the PURE half of the legacy scalar
  // gate, lifted out of `vram_gate.rs` by sc-19049 so it compiles (and can therefore be driven by
  // the Rust decision emitter) off a CUDA lane; `vram_gate.rs` re-exports every item.
  fitGate: "crates/sceneworks-worker/src/fit_gate.rs",
  candleScalarGate: "crates/sceneworks-worker/src/candle_scalar_gate.rs",
  vramGate: "crates/sceneworks-worker/src/vram_gate.rs",
  candleMemoryStrategy: "crates/sceneworks-worker/src/candle_memory_strategy.rs",
  memoryRouteRegistry: "crates/sceneworks-worker/src/memory_route_registry.rs",
  kreaControlFit: "crates/sceneworks-worker/src/krea_control_fit.rs",
  conditioningFit: "crates/sceneworks-worker/src/conditioning_fit.rs",
  videoAdmission: "crates/sceneworks-worker/src/video_admission.rs",
  videoMemoryCurves: "crates/sceneworks-core/src/video_memory_curves.rs",
  videoMemoryCurvesData: "docs/generated/video-memory-curves.json",

  // Call sites. Every module that invokes an admission entry symbol is fingerprinted, so a route
  // switching gates rotates this artifact even when no gate function changed.
  imageRouting: "crates/sceneworks-worker/src/image_jobs/base.rs",
  imageBaseAdmission: "crates/sceneworks-worker/src/image_jobs/base_admission.rs",
  imageConditioningGate: "crates/sceneworks-worker/src/image_jobs/conditioning_gate.rs",
  imageStrictControl: "crates/sceneworks-worker/src/image_jobs/candle_strict_control.rs",
  imageFlux1Control: "crates/sceneworks-worker/src/image_jobs/flux1_control_candle.rs",
  imageFlux2Control: "crates/sceneworks-worker/src/image_jobs/flux2_control_candle.rs",
  imageFlux2Edit: "crates/sceneworks-worker/src/image_jobs/flux2_edit_candle.rs",
  imageFluxIpAdapter: "crates/sceneworks-worker/src/image_jobs/flux_ipadapter.rs",
  imageKolorsControl: "crates/sceneworks-worker/src/image_jobs/kolors_control.rs",
  imageQwenControl: "crates/sceneworks-worker/src/image_jobs/qwen_control.rs",
  imageKreaControl: "crates/sceneworks-worker/src/image_jobs/krea_control_candle.rs",
  imageKreaEdit: "crates/sceneworks-worker/src/image_jobs/krea_edit_candle.rs",
  imageKreaMultiphase: "crates/sceneworks-worker/src/image_jobs/krea_multiphase.rs",
  imageQwenEdit: "crates/sceneworks-worker/src/image_jobs/qwen_edit_candle.rs",
  imageZimageControl: "crates/sceneworks-worker/src/image_jobs/zimage_control.rs",
  imageInstantId: "crates/sceneworks-worker/src/image_jobs/instantid.rs",
  imageBernini: "crates/sceneworks-worker/src/image_jobs/bernini.rs",
  imagePulid: "crates/sceneworks-worker/src/image_jobs/pulid_candle.rs",
  imageDetail: "crates/sceneworks-worker/src/image_jobs/detail.rs",
  videoRouteCandle: "crates/sceneworks-worker/src/video_jobs/candle.rs",
  videoRouteBernini: "crates/sceneworks-worker/src/video_jobs/bernini.rs",
  videoRouteWan: "crates/sceneworks-worker/src/video_jobs/wan.rs",
});

/**
 * The admission mechanisms a candle route can land on, in migration order â€” roughly worst to best,
 * which is also the order epic 19048's slices retire them in.
 *
 * Two symbol lists per mechanism, because they answer different questions:
 *
 *   - `definitionSymbols` are the unqualified `fn` names in `source`. `geometryAware` is derived
 *     from THEIR parameter lists, never declared here â€” declaring it would be a restatement that
 *     drifts the moment sc-19054 makes the resident path geometry-aware.
 *   - `callPattern` is the regex a call site is recognised by. It is separate because several of
 *     these functions have names too generic to scan for unqualified (`decide`, `admit`,
 *     `evaluate`); production reaches them through a module-qualified path, and that is what is
 *     matched.
 */
export const ADMISSION_MECHANISMS = Object.freeze([
  {
    id: "legacy_scalar_gate",
    // The pure arithmetic lives in `candle_scalar_gate.rs` since sc-19049; `vram_gate.rs`
    // re-exports it, so every call site below is unchanged.
    source: "candleScalarGate",
    definitionSymbols: Object.freeze([
      "predicted_peak_gb",
      "predicted_peak_gb_with_adapter_bytes",
      "predicted_peak_gb_for_request",
      "predicted_sequential_peak_gb",
      "predicted_sequential_peak_gb_with_adapter_bytes",
      "predicted_sequential_peak_gb_for_request",
      "scalar_peak_class",
      "sequential_overflow_gb",
      "load_plan",
    ]),
    callPattern:
      "(?<!fn\\s)\\b(?:predicted_peak_gb|predicted_peak_gb_with_adapter_bytes|predicted_peak_gb_for_request|predicted_sequential_peak_gb|predicted_sequential_peak_gb_with_adapter_bytes|predicted_sequential_peak_gb_for_request|scalar_peak_class|sequential_overflow_gb|load_plan)\\s*\\(",
    summary:
      "Per-tier manifest scalar (`candle.vramGbByTier` + headroom, else `candle.minMemoryGb`) compared to a live VRAM budget. Since sc-19054 the scalar is graded per request geometry: a covering `measured`/`vramMeasuredPixels` capture compares raw, everything else compares as the estimate-margin-widened declared floor.",
  },
  {
    id: "shared_selector_named_revision",
    source: "candleMemoryStrategy",
    definitionSymbols: Object.freeze(["evaluate_shared_image"]),
    callPattern: "(?<!fn\\s)\\bevaluate_shared_image\\s*\\(",
    summary:
      "The shared memory-strategy selector, entered under an engine-id-keyed `*_REQUEST_EVIDENCE_REVISION` scope.",
  },
  {
    id: "shared_selector_bespoke_override",
    source: "candleMemoryStrategy",
    definitionSymbols: Object.freeze(["evaluate_shared_bespoke_image"]),
    callPattern: "(?<!fn\\s)\\bevaluate_shared_bespoke_image\\s*\\(",
    summary:
      "The shared selector entered with a caller-supplied `request_evidence_revision_override`, bypassing the engine-id match.",
  },
  {
    id: "shared_selector_declaration_catch_all",
    source: "candleMemoryStrategy",
    definitionSymbols: Object.freeze(["evaluate_shared_image"]),
    callPattern: "(?<!fn\\s)\\bevaluate_shared_image\\s*\\(",
    summary:
      "The shared selector reached through the sc-18456 catch-all arm, scoped by the provider's manifest `candle.memoryStrategyContract`.",
  },
  {
    id: "shared_selector_legacy_scalar_compatibility",
    source: "candleMemoryStrategy",
    definitionSymbols: Object.freeze(["select_compatibility_resident"]),
    callPattern: "(?<!fn\\s)\\bselect_compatibility_resident\\s*\\(",
    summary:
      "Resident-only shared-selector compatibility candidate carrying an existing Candle scalar ceiling without a second reserve or estimate widening.",
  },
  {
    id: "shared_selector_structural_floor_compatibility",
    source: "candleMemoryStrategy",
    definitionSymbols: Object.freeze(["select_compatibility_resident"]),
    callPattern: "(?<!fn\\s)\\bselect_compatibility_resident\\s*\\(",
    summary:
      "Resident-only shared-selector compatibility candidate carrying a source-backed structural lower bound with no calibration claim or estimate widening.",
  },
  {
    id: "shared_selector_unverified_compatibility",
    source: "candleMemoryStrategy",
    definitionSymbols: Object.freeze(["select_compatibility_resident"]),
    callPattern: "(?<!fn\\s)\\bselect_compatibility_resident\\s*\\(",
    summary:
      "Evidence-free resident route submitted to the shared selector with no candidate; its Unverified verdict preserves the legacy best-effort resident execution.",
  },
  {
    id: "shared_selector_legacy_video_compatibility",
    source: "videoAdmission",
    definitionSymbols: Object.freeze(["select_legacy_video_resident"]),
    callPattern: "(?<!fn\\s)\\bselect_legacy_video_resident\\s*\\(",
    summary:
      "Typed Candle-lane resident compatibility candidate carrying one flat video's already-normalized legacy ceiling without widening it again.",
  },
  {
    id: "krea_turbo_fit",
    source: "vramGate",
    definitionSymbols: Object.freeze([
      "krea_turbo_fit_with_runtime",
      "krea_turbo_smaller_fit_with_runtime",
    ]),
    callPattern: "(?<!fn\\s)\\bkrea_turbo_(?:smaller_)?fit_with_runtime\\s*\\(",
    summary:
      "krea_2_turbo's bespoke declared phase curves (`candle.turboFit.phaseCurvesByTier`), the one geometry-aware candle image path today.",
  },
  {
    id: "krea_control_fit",
    source: "kreaControlFit",
    definitionSymbols: Object.freeze([
      "fit_ladder",
      "fit_ladder_for_entry_with_runtime",
      "predicted_control_peak_gb",
      "predicted_control_sequential_peak_gb",
      "incurred_peak_gb",
    ]),
    // Namespace reference, not just the ladder entry points. A module that reaches for ANY of this
    // fit module's items â€” the ladder, the incurred peak, the branch tier, its result type â€” is on
    // its admission path; requiring the entry function alone would score `conditioning_gate.rs`
    // (which asks for the incurred peak) as on the path and the lane module that owns the decision
    // as off it. `::tests` is excluded so a test-module path reference is not read as a call site.
    callPattern: "krea_control_fit::(?!tests\\b)\\w+",
    summary: "The Krea ControlNet overlay's own fit module.",
  },
  {
    id: "conditioning_fit",
    source: "conditioningFit",
    definitionSymbols: Object.freeze(["decide", "admit", "conditioning_floor_gb"]),
    // Same namespace-reference rule as `krea_control_fit`. The DECISION (`decide`/`admit`) is taken
    // once, in `conditioning_gate.rs`; the overlay lanes reach the module by building the footprint
    // it decides over (`weights_source_path`, `pid_paths`, `ConditioningFootprint`). Scoring only
    // the decision would report one route on this mechanism when eleven are.
    callPattern: "conditioning_fit::(?!tests\\b)\\w+",
    summary: "The shared conditioning-overlay fit module used by the strict-control lanes.",
  },
  {
    id: "flat_video_fit_error",
    source: "vramGate",
    definitionSymbols: Object.freeze([
      "svd_fit_error",
      "mochi_fit_error",
      "wan_video_fit_error",
      "wan_video_fit_error_with_adapter_bytes",
      "scail2_video_fit_error",
      "scail2_video_fit_error_with_adapter_bytes",
      "video_weights_fit_error",
      "unscoped_video_weights_fit_error",
    ]),
    callPattern:
      "(?<!fn\\s)\\b(?:svd_fit_error|mochi_fit_error|wan_video_fit_error(?:_with_adapter_bytes)?|scail2_video_fit_error(?:_with_adapter_bytes)?|video_weights_fit_error|unscoped_video_weights_fit_error)\\s*\\(",
    summary: "Flat per-model video fit errors, one bespoke function per family.",
  },
  {
    id: "video_memory_curve_bundle",
    source: "videoMemoryCurves",
    definitionSymbols: Object.freeze(["evaluate"]),
    callPattern: "(?<!fn\\s)\\bpackaged_video_memory_curves\\s*\\(",
    summary:
      "Epic 18803's shared, lane-tagged video curve container in `sceneworks-core`. Fails closed on a foreign lane.",
  },
  {
    id: "unreached",
    source: null,
    definitionSymbols: Object.freeze([]),
    callPattern: null,
    summary:
      "The router serves this model on the candle lane, but no admission gate keyed to it was found.",
  },
]);

/**
 * Video engine id -> the flat fit-error symbol that admits it, plus the model ids that reach it.
 *
 * This binding is DECLARED and then MECHANICALLY VERIFIED: [`deriveVideoBindings`] requires every
 * symbol named here to be defined in `vram_gate.rs` AND called from `video_jobs/candle.rs`, and
 * requires every engine id to be produced by `candle_video_engine_id`'s match (or to be one of the
 * two ids reached off that match â€” `scail2_14b` through the generator-cache cold-load admission and
 * `wan_2_2_vace_fun_14b` through its own dual-expert weights preflight). It throws otherwise. The
 * verification is what keeps it from becoming the hand-written snapshot the story rejects: the
 * binding cannot silently survive the symbol being renamed, the call site being deleted, or the
 * engine leaving the router.
 */
export const VIDEO_ADMISSION_BINDINGS = Object.freeze([
  Object.freeze({ engineId: "svd_xt", symbol: "svd_fit_error", offMatch: false }),
  Object.freeze({ engineId: "mochi_1", symbol: "mochi_fit_error", offMatch: false }),
  Object.freeze({
    engineId: "wan2_2_ti2v_5b",
    symbol: "wan_video_fit_error_with_adapter_bytes",
    offMatch: false,
  }),
  Object.freeze({
    engineId: "wan2_2_t2v_14b",
    symbol: "wan_video_fit_error_with_adapter_bytes",
    offMatch: false,
  }),
  Object.freeze({
    engineId: "wan2_2_i2v_14b",
    symbol: "wan_video_fit_error_with_adapter_bytes",
    offMatch: false,
  }),
  Object.freeze({
    engineId: "ltx_2_3_distilled",
    symbol: "video_weights_fit_error",
    offMatch: false,
  }),
  Object.freeze({
    engineId: "scail2_14b",
    symbol: "scail2_video_fit_error_with_adapter_bytes",
    offMatch: true,
  }),
  Object.freeze({
    engineId: "wan_2_2_vace_fun_14b",
    symbol: "unscoped_video_weights_fit_error",
    offMatch: true,
  }),
]);

/**
 * Overlay/bespoke image lanes whose admission is decided by their OWN module rather than by the
 * engine-keyed data the rest of this script derives, mapped to the [`SOURCE_PATHS`] key that module
 * lives under.
 *
 * The base txt2img gate is engine-and-manifest-keyed, so it needs no lane binding â€” a model reaches
 * it exactly when its manifest publishes a number for it. These lanes are different: whether a
 * request hits `conditioning_fit`, `krea_control_fit` or the bespoke PuLID selector override is a
 * property of WHICH HANDLER MODULE serves it, and nothing in the manifest says which one that is.
 *
 * Declared, then verified by [`deriveLaneBindings`]: the lane must still exist in
 * `CANDLE_IMAGE_ROUTES`, and its module must still call at least one admission entry symbol. Lanes
 * with no binding are not silently dropped â€” they are published as
 * `summary.imageLanesWithoutAdmissionBinding`, which is the list a later slice shortens.
 */
export const IMAGE_LANE_BINDINGS = Object.freeze({
  InstantId: "imageInstantId",
  Flux2Edit: "imageFlux2Edit",
  QwenEdit: "imageQwenEdit",
  KreaEdit: "imageKreaEdit",
  FluxIpAdapter: "imageFluxIpAdapter",
  QwenControl: "imageQwenControl",
  KolorsControl: "imageKolorsControl",
  ZImageControl: "imageZimageControl",
  Flux1Control: "imageFlux1Control",
  Flux2Control: "imageFlux2Control",
  KreaControl: "imageKreaControl",
  Pulid: "imagePulid",
});

/**
 * Mechanisms a lane binding may contribute. The engine-keyed and manifest-keyed mechanisms are
 * deliberately excluded: they are derived from data that says whether the gate is LIVE for a model,
 * and a lane's call site cannot answer that (`flux1_control_candle.rs` calls the scalar gate, but
 * for a model with no `candle` block the call returns `None` and admits).
 */
const MODULE_KEYED_MECHANISMS = Object.freeze([
  "conditioning_fit",
  "krea_control_fit",
  "shared_selector_bespoke_override",
  "video_memory_curve_bundle",
]);

/** The synthetic budgets the baseline gates against, in GB. `free_gb == total_gb` (a cold card). */
export const BASELINE_BUDGETS_GB = Object.freeze([12, 24, 48, 96]);

/**
 * Image geometry corpus. Square rungs spanning the ladder the calibration plan uses.
 *
 * There is no video geometry corpus any more. Every candle video route resolves to
 * `not_evaluated` â€” the flat per-family fit errors take weight bytes measured off the resolved
 * model dir â€” and a row that publishes no decision must not publish a coordinate either. The two
 * clip lengths this file used to cross every video route with produced 0-of-N variance, which is
 * the signature of a fabricated axis rather than a measured one.
 *
 * Kept in lockstep with `IMAGE_GEOMETRIES` in
 * `crates/sceneworks-worker/src/candle_admission_decisions.rs`, which is what actually drives the
 * gate; `generate-candle-admission-inventory.test.mjs` pins the two equal.
 */
export const BASELINE_IMAGE_GEOMETRIES = Object.freeze([
  Object.freeze({ width: 1024, height: 1024, frames: 1 }),
  Object.freeze({ width: 1536, height: 1536, frames: 1 }),
  Object.freeze({ width: 2048, height: 2048, frames: 1 }),
]);

function sha256(body) {
  return createHash("sha256").update(body).digest("hex");
}

/**
 * Read every declared source, returning `{ bodies, read }`. `read` is the exact key set that was
 * touched; the test pins it equal to `Object.keys(SOURCE_PATHS)`, which is what turns "the
 * fingerprint covers every source it reads" from a claim into an invariant.
 */
export async function readSources({ overrides = {} } = {}) {
  const bodies = {};
  const read = [];
  for (const [name, relative] of Object.entries(SOURCE_PATHS)) {
    read.push(name);
    bodies[name] = canonicalSourceText(
      Object.hasOwn(overrides, name)
        ? overrides[name]
        : await readFile(path.join(ROOT, relative), "utf8"),
    );
  }
  return { bodies, read };
}

/**
 * The Rust-emitted decision artifact this generator CONSUMES.
 *
 * Deliberately NOT in [`SOURCE_PATHS`]. `SOURCE_PATHS` is the set of sources the inventory's facts
 * are DERIVED from, and its hash is what makes the inventory go stale; this file is derived from the
 * inventory (the emitter reads the route universe out of it), so fingerprinting it would close a
 * loop â€” regenerate, hash, regenerate. It is instead identified by its own digest, recorded in the
 * baseline's provenance, so a decision change is still visible in the committed diff and still fails
 * `--check`.
 */
export const DECISIONS_PATH = "docs/generated/candle-admission-decisions.json";

/**
 * Read the Rust-emitted decisions. Produced by
 * `SCENEWORKS_REGENERATE_CANDLE_ADMISSION=1 cargo test -p sceneworks-worker candle_admission_decisions`,
 * which drives `crates/sceneworks-worker/src/candle_scalar_gate.rs` â€” the code the request actually
 * reaches â€” rather than any re-implementation of it here.
 */
export async function readDecisions({ override } = {}) {
  const raw =
    override ?? canonicalSourceText(await readFile(path.join(ROOT, DECISIONS_PATH), "utf8"));
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed.rows) || parsed.rows.length === 0) {
    throw new Error(
      `${DECISIONS_PATH} carries no decision rows. Produce it with ` +
        "`SCENEWORKS_REGENERATE_CANDLE_ADMISSION=1 cargo test -p sceneworks-worker " +
        "candle_admission_decisions`.",
    );
  }
  return { ...parsed, digest: `sha256:${sha256(raw)}` };
}

/** `source-tree:<sha256>` over the semantic body of every declared source, NUL-joined in order. */
export function sourceRevisionOf(bodies) {
  const entries = Object.entries(SOURCE_PATHS);
  const semantic = entries.map(([name, relative]) => semanticSourceBody(relative, bodies[name]));
  return {
    revision: `source-tree:${sha256(semantic.join("\0"))}`,
    perSource: Object.fromEntries(
      entries.map(([name, relative], index) => [
        name,
        { path: relative, sha256: sha256(semantic[index]) },
      ]),
    ),
  };
}

/** Drop a Rust file's `#[cfg(test)] mod tests { â€¦ }` tail so test call sites are not counted. */
export function productionBody(body) {
  const marker = body.search(/^#\[cfg\(test\)\]\s*\r?\nmod tests\b/m);
  return marker === -1 ? body : body.slice(0, marker);
}

/**
 * The engine ids `candle_memory_strategy.rs` names explicitly, and the revision constant each one
 * lands on. Parsed from the dispatch itself so a new arm is picked up without editing this script.
 *
 * Returns `{ named: Map<engineId, {constant, value}>, catchAll: {constant, value}, constants }`.
 * A dispatch with no catch-all arm throws: since sc-18456 the catch-all IS the mechanism by which a
 * declaration-scoped provider reaches the shared selector, and an inventory that silently reported
 * "unreached" for all of them would be wrong in the direction that hides work.
 */
export function parseRequestScopeDispatch(source) {
  const constants = new Map();
  for (const match of source.matchAll(
    /const\s+([A-Z0-9_]*REQUEST_EVIDENCE_REVISION)\s*:\s*&str\s*=\s*"([^"]+)"\s*;/g,
  )) {
    constants.set(match[1], match[2]);
  }
  if (constants.size === 0) {
    throw new Error("candle_memory_strategy.rs: no *_REQUEST_EVIDENCE_REVISION constants found");
  }

  const dispatch = source.match(
    /request_evidence_revision_override\s*\.unwrap_or\(match engine_id \{([\s\S]*?)\n {4}\}\)/,
  );
  if (!dispatch) {
    throw new Error("candle_memory_strategy.rs: could not locate the engine-id request-scope match");
  }

  const named = new Map();
  let catchAll = null;
  // Arms are `<patterns> => CONSTANT,` or `<patterns> => {\n CONSTANT \n}`; rustfmt wraps long
  // pattern lists across lines, so the pattern half is matched non-greedily up to its `=>`.
  for (const arm of dispatch[1].matchAll(
    /([^=>{}]+?)=>\s*(?:\{\s*([A-Z0-9_]+)\s*\}|([A-Z0-9_]+))\s*,?/g,
  )) {
    const constant = arm[2] ?? arm[3];
    if (!constants.has(constant)) continue;
    const value = constants.get(constant);
    const patterns = arm[1];
    if (/(^|\|)\s*_\s*$/.test(patterns.trim()) || patterns.trim() === "_") {
      catchAll = { constant, value };
      continue;
    }
    for (const id of patterns.matchAll(/"([^"]+)"/g)) {
      named.set(id[1], { constant, value });
    }
  }
  if (!catchAll) {
    throw new Error(
      "candle_memory_strategy.rs: the request-scope match has no catch-all arm. Since sc-18456 the " +
        "catch-all is how a declaration-scoped provider reaches the shared selector; classifying " +
        "routes without it would silently under-report reachability.",
    );
  }
  // A constant that is declared but never reached from the match is an OVERRIDE-only scope (today:
  // PuLID-FLUX, injected at the `evaluate_shared_bespoke_image` call site). Report it as such.
  const reached = new Set([...named.values()].map((entry) => entry.constant));
  reached.add(catchAll.constant);
  const overrideOnly = [...constants.entries()]
    .filter(([name]) => !reached.has(name))
    .map(([constant, value]) => ({ constant, value }));

  return { named, catchAll, overrideOnly, constants };
}

/** Parse one source-owned exact `&[&str]` compatibility-route set. */
export function parseCompatibilityRoutes(source, constant) {
  const escaped = constant.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = source.match(
    new RegExp(`(?:pub\\(crate\\)\\s+)?const\\s+${escaped}\\s*:\\s*&\\[&str\\]\\s*=\\s*&\\[([\\s\\S]*?)\\];`),
  );
  if (!match) throw new Error(`could not locate ${constant} source-owned route set`);
  const routes = [...match[1].matchAll(/"([^"]+)"/g)].map((entry) => entry[1]);
  if (routes.length === 0 || new Set(routes).size !== routes.length) {
    throw new Error(`${constant} must be a non-empty unique route set`);
  }
  return new Set(routes);
}

/** Extract a Rust function body with balanced braces; throws instead of treating missing code as wiring. */
function rustFunctionBody(source, symbol) {
  const signature = new RegExp(
    `(?:pub\\(crate\\)\\s+)?fn\\s+${symbol}(?:<[^>]*>)?\\s*\\(`,
  ).exec(source);
  if (!signature) throw new Error(`could not locate production function ${symbol}`);
  const open = source.indexOf("{", signature.index);
  if (open === -1) throw new Error(`${symbol}: no function body`);
  let depth = 0;
  for (let index = open; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    if (source[index] === "}") depth -= 1;
    if (depth === 0) return source.slice(open + 1, index);
  }
  throw new Error(`${symbol}: unterminated function body`);
}

/** Extract one call's balanced argument list from a production function body. */
function rustCallArguments(source, symbol) {
  const call = new RegExp(`\\b${symbol}\\s*\\(`).exec(source);
  if (!call) throw new Error(`could not locate production call ${symbol}`);
  const open = source.indexOf("(", call.index);
  let depth = 0;
  for (let index = open; index < source.length; index += 1) {
    if (source[index] === "(") depth += 1;
    if (source[index] === ")") depth -= 1;
    if (depth === 0) return source.slice(open + 1, index);
  }
  throw new Error(`${symbol}: unterminated call`);
}

/** Load-bearing wiring guards for the compatibility migration, driven from production source. */
export function validateCompatibilityWiring(bodies) {
  const compatibilityBody = rustFunctionBody(
    bodies.candleMemoryStrategy,
    "select_compatibility_resident",
  );
  if (!compatibilityBody.includes("reserved_headroom_gb: 0.0")) {
    throw new Error("compatibility selector must normalize its already-reserved ceiling with zero reserve");
  }
  for (const basis of ["LegacyScalar", "StructuralFloor", "LegacyVideo"]) {
    if (!compatibilityBody.includes(`CandidateBasis::${basis}`)) {
      throw new Error(`compatibility selector no longer accepts ${basis}`);
    }
  }
  const imageBody = productionBody(bodies.imageRouting);
  if (
    !imageBody.includes("is_legacy_scalar_compatibility_route") ||
    !imageBody.includes("CandidateBasis::LegacyScalar") ||
    !imageBody.includes("select_compatibility_resident")
  ) {
    throw new Error("legacy scalar image routes are not wired to their typed compatibility candidate");
  }
  if (
    !imageBody.includes("is_unverified_image_compatibility_route") ||
    !compatibilityBody.includes("let candidates = evidence") ||
    !compatibilityBody.includes("&candidates")
  ) {
    throw new Error("evidence-free image routes must invoke the shared selector with an empty candidate set");
  }
  if (
    !/legacy_scalar_compatibility\s*&&\s*crate::candle_scalar_gate::scalar_measurement_lane_is_candle/.test(
      imageBody,
    ) ||
    !imageBody.includes(".then_some(gated_needed)")
  ) {
    throw new Error("legacy scalar compatibility candidates must reject missing/foreign lanes as Unverified");
  }
  const sharedImage = rustFunctionBody(bodies.candleMemoryStrategy, "evaluate_shared_image_inner");
  const scalarProvenance = rustFunctionBody(
    bodies.candleMemoryStrategy,
    "select_unverified_scalar_provenance",
  );
  if (
    !sharedImage.includes("scalar_measurement_lane_is_candle(manifest)") ||
    !sharedImage.includes("select_unverified_scalar_provenance") ||
    !scalarProvenance.includes("&[]") ||
    !bodies.candleScalarGate.includes("== Some(\"candle\")")
  ) {
    throw new Error("scalar selector provenance must reject missing/foreign lanes as Unverified");
  }
  for (const [source, symbol] of [[bodies.conditioningFit, "decide_via_compatibility_selector"]]) {
    const body = rustFunctionBody(source, symbol);
    if (!body.includes("CandidateBasis::StructuralFloor") || !body.includes("select_compatibility_resident")) {
      throw new Error(`${symbol} is not wired to its structural-floor compatibility candidate`);
    }
  }
  const baseFloorSelector = rustFunctionBody(
    bodies.imageBaseAdmission,
    "compatibility_base_floor_plan",
  );
  if (
    !baseFloorSelector.includes("CandidateBasis::StructuralFloor") ||
    !baseFloorSelector.includes("select_compatibility_resident")
  ) {
    throw new Error("compatibility_base_floor_plan is not wired to its structural-floor candidate");
  }
  const videoSelector = rustFunctionBody(bodies.videoAdmission, "select_legacy_video_resident");
  if (
    !videoSelector.includes("lane != VideoLane::Candle") ||
    !videoSelector.includes("CandidateBasis::LegacyVideo")
  ) {
    throw new Error("legacy video selector lost its typed Candle lane or LegacyVideo basis");
  }
  if (!baseFloorSelector.includes("scope.structural_floor_applies.then_some(floor_gb)")) {
    throw new Error("structural-floor selector must withhold its candidate on unsupported modes");
  }
  if (
    !baseFloorSelector.includes("if !scope.structural_floor_applies") ||
    !baseFloorSelector.includes("LoadPlan::Resident")
  ) {
    throw new Error("unsupported Bernini modes must preserve resident execution after Unverified");
  }
  const berniniVideo = rustFunctionBody(bodies.videoRouteBernini, "generate_candle_bernini");
  const berniniFloor = rustFunctionBody(
    bodies.videoRouteBernini,
    "bernini_structural_floor_applies",
  );
  if (!berniniFloor.includes('mode_key == "t2v" && reference_count == 0')) {
    throw new Error("Bernini structural floor must be exact T2V with zero references");
  }
  for (const marker of [
    "bernini_adapter_resident_bytes(&adapters, quant)",
    "width: request.width",
    "height: request.height",
    "frames: wan_frame_count(request.raw_frame_count())",
    "reference_count: u32::try_from(conditioning.len())",
    "structural_floor_applies: bernini_structural_floor_applies(",
    "admit_candle_base_floor_with_resident_overlay_via_selector",
  ]) {
    if (!berniniVideo.includes(marker)) {
      throw new Error(`Bernini video structural selector lost production input: ${marker}`);
    }
  }
  const videoSelection = rustFunctionBody(bodies.videoAdmission, "fitted_or_floor_phase_peaks");
  if (
    !videoSelection.includes("selector.identity.lane == VideoLane::Candle") ||
    !videoSelection.includes("return None")
  ) {
    throw new Error("Candle optimized video candidates must remain Unverified without a fitted curve");
  }
  if (
    !videoSelection.includes("let headroom_bytes = selector.headroom_bytes?") ||
    videoSelection.includes("headroom_bytes.unwrap_or(0)")
  ) {
    throw new Error("video fallback allowance absence must not create an estimate-floor candidate");
  }
  const wanFallback = rustFunctionBody(bodies.videoRouteWan, "admission_fallback_headroom_bytes");
  if (
    !bodies.videoRouteWan.includes("fallback_headroom_bytes: Option<u64>") ||
    !wanFallback.includes("checked_sub(runtime.budget.reserved_headroom_bytes)") ||
    wanFallback.includes("unwrap_or(0)")
  ) {
    throw new Error("Wan fallback headroom must preserve Option absence and normalize real reserve once");
  }
  if (
    !bodies.videoRouteWan.includes("fallback_headroom_bytes") ||
    !bodies.videoRouteWan.includes(
      'if cfg!(all(not(target_os = "macos"), feature = "backend-candle")) {\n                        None',
    ) ||
    !bodies.videoRouteWan.includes("admission_fallback_headroom_bytes(")
  ) {
    throw new Error("Wan production callback must pass an absent Candle fallback allowance");
  }
  for (const symbol of [
    "mochi_fit_error",
    "svd_fit_error",
    "video_weights_fit_error",
    "wan_video_fit_error_with_adapter_bytes",
    "scail2_video_fit_error_with_adapter_bytes",
  ]) {
    const body = rustFunctionBody(bodies.vramGate, symbol);
    if (
      !body.includes("select_legacy_video_resident") ||
      !body.includes("VideoLane::Candle")
    ) {
      throw new Error(`${symbol} is not wired to the typed Candle legacy-video selector`);
    }
  }


  // The generic weights-floor helper is shared with VACE-Fun, which owns no approved SC-19055
  // compatibility route. Eligible callers therefore provide an explicit request scope, and the
  // selector must consume every field from that scope rather than stamping placeholder evidence.
  const weightsFloorBody = rustFunctionBody(bodies.vramGate, "video_weights_fit_error");
  const weightsSelectorArgs = rustCallArguments(weightsFloorBody, "select_legacy_video_resident");
  for (const field of [
    "request_scope.tier_key",
    "request_scope.mode_key",
    "request_scope.geometry",
  ]) {
    if (!weightsSelectorArgs.includes(field)) {
      throw new Error(`video_weights_fit_error does not propagate ${field} to the selector`);
    }
  }

  // LTX is the production path that resolves a packed tier and full request coordinates before
  // invoking the floor. Keep those exact values co-located in the call so a literal bf16/T2V/1x1x1
  // regression cannot retain a green generated inventory.
  const candleVideoBody = rustFunctionBody(bodies.videoRouteCandle, "generate_candle_video_using");
  const ltxAdmissionArgs = rustCallArguments(candleVideoBody, "admit_candle_ltx");
  for (const [field, expected] of [
    ["tier", 'ltx_tier_key.expect("an LTX model path must resolve an admission tier")'],
    ["mode", "request.mode.as_str()"],
    [
      "geometry",
      "crate::video_admission::video_gate_geometry(request.width, request.height, frames)",
    ],
  ]) {
    if (!ltxAdmissionArgs.includes(expected)) {
      throw new Error(`Candle LTX admission does not propagate its resolved ${field}`);
    }
  }
}

/**
 * The engine ids passed to `evaluate_shared_bespoke_image` as string literals â€” the override-only
 * scopes. Scanned across every call-site source so a second bespoke override cannot appear
 * unnoticed.
 */
export function parseBespokeOverrideEngines(bodies) {
  const ids = new Set();
  for (const [name, body] of Object.entries(bodies)) {
    if (!SOURCE_PATHS[name].endsWith(".rs")) continue;
    // The defining module is not a call site; scanning it would read the signature, not an argument.
    if (name === "candleMemoryStrategy") continue;
    for (const call of productionBody(body).matchAll(
      /evaluate_shared_bespoke_image\s*\(([\s\S]{0,600}?)\)\s*[;?]/g,
    )) {
      for (const literal of call[1].matchAll(/"([a-z0-9_]+)"/g)) ids.add(literal[1]);
    }
  }
  return ids;
}

// -------------------------------------------------------------------------------------------------
// Geometry-awareness, resolved PER GATE FUNCTION
//
// "Is this gate geometry-aware?" is a property of ONE function's signature, not of a mechanism. Two
// mistakes are easy here and both were made before sc-19049's review:
//
//  1. **Unioning across a mechanism.** `flat_video_fit_error` names seven symbols and they do
//     NOT agree. `svd_fit_error` and `mochi_fit_error` take `(frames, width, height)` as
//     scalars; `wan_video_fit_error*` and `scail2_video_fit_error*` take
//     `geometry: MemoryGeometry` since sc-19055. `video_weights_fit_error` now carries geometry as
//     compatibility-selector identity without scaling its ungraded byte floor, while the separate
//     `unscoped_video_weights_fit_error` used by VACE-Fun stays resolution-blind. A union would still
//     erase that distinction, which is why this stays a PER-SYMBOL derivation even now that most of
//     the mechanism's symbols happen to agree.
//  2. **Scanning parameter TOKENS for `width:`.** Geometry also arrives inside a struct:
//     `krea_control_fit::fit_ladder_for_entry_with_runtime` takes `geometry: MemoryGeometry` and
//     `candle_memory_strategy::evaluate_shared_image` takes the same, while
//     `VideoMemoryCurveBundle::evaluate` takes `query: VideoCurveQuery` whose `geometry` field is a
//     `VideoCurveGeometry`. A token scan reports all three as geometry-blind, which is the exact
//     inverse of the truth.
//
// So a symbol's axes are resolved from its parsed parameter list, with struct-typed parameters
// resolved to their FIELDS â€” recursively, and from either the type's `struct` definition or a
// struct-literal construction of it anywhere in the fingerprinted tree (`gen_core::MemoryGeometry`
// is declared in the pinned inference dependency, but it is BUILT in `vram_gate.rs` and
// `krea_control_fit.rs`, so its field set is recoverable from sources this artifact already hashes).
// -------------------------------------------------------------------------------------------------

/** Field names that carry request geometry. `pixels`/`voxels` are pre-multiplied forms of the same. */
const GEOMETRY_FIELDS = Object.freeze(["width", "height", "frames", "pixels", "voxels"]);

/** Rust primitives â€” a parameter of one of these types has no fields to resolve. */
const SCALAR_TYPE = /^(?:bool|char|str|String|u8|u16|u32|u64|u128|usize|i8|i16|i32|i64|isize|f32|f64)$/;

/** Split a Rust parameter/field list on TOP-LEVEL commas (generics and tuples nest). */
function splitTopLevel(text) {
  const parts = [];
  let depth = 0;
  let current = "";
  for (const char of text) {
    if ("<([{".includes(char)) depth += 1;
    else if (">)]}".includes(char)) depth -= 1;
    if (char === "," && depth <= 0) {
      parts.push(current);
      current = "";
      continue;
    }
    current += char;
  }
  if (current.trim()) parts.push(current);
  return parts.map((part) => part.trim()).filter(Boolean);
}

/**
 * Reduce a Rust type expression to the bare type NAME whose fields we might resolve: strip
 * references, `mut`, lifetimes, `dyn`/`impl`, and unwrap the single-argument container generics that
 * do not change the shape of the value (`Option`, `Box`, `Rc`, `Arc`, `Cow`).
 */
export function bareTypeName(type) {
  let text = type
    .replace(/\bdyn\b|\bimpl\b|\bmut\b/g, " ")
    .replace(/&\s*'[a-z_][\w]*/g, " ")
    .replace(/&/g, " ")
    .trim();
  for (let guard = 0; guard < 6; guard += 1) {
    const wrapper = text.match(/^(Option|Box|Rc|Arc|Cow)\s*<([\s\S]*)>$/);
    if (!wrapper) break;
    text = wrapper[2].replace(/^\s*'[a-z_][\w]*\s*,\s*/, "").trim();
  }
  text = text.replace(/<[\s\S]*$/, "").trim();
  const segments = text.split("::").filter(Boolean);
  return segments.length ? segments[segments.length - 1].trim() : "";
}

/** Parse `name: Type` pairs out of a struct body or a parameter list. */
function parseNamedFields(text) {
  const fields = [];
  for (const part of splitTopLevel(text)) {
    const cleaned = part.replace(/#\[[\s\S]*?\]/g, "").replace(/\/\/[^\n]*/g, "");
    const match = cleaned.match(/(?:pub(?:\s*\([^)]*\))?\s+)?([A-Za-z_]\w*)\s*:\s*([\s\S]+)$/);
    if (!match) continue;
    fields.push({ name: match[1], type: match[2].trim() });
  }
  return fields;
}

/**
 * The field list of `typeName`, from its `struct` definition in a fingerprinted source or â€” for a
 * type declared in the pinned inference dependency â€” from a struct-literal construction of it.
 * `null` when neither exists, which the caller reports rather than silently reading as "no fields".
 */
export function resolveStructFields(bodies, typeName) {
  if (!typeName || SCALAR_TYPE.test(typeName)) return null;
  for (const [name, body] of Object.entries(bodies)) {
    if (!SOURCE_PATHS[name].endsWith(".rs")) continue;
    const definition = body.match(
      new RegExp(`struct\\s+${typeName}\\s*(?:<[^>{]*>)?\\s*\\{([\\s\\S]*?)\\n\\}`),
    );
    if (definition) return { fields: parseNamedFields(definition[1]), via: "definition" };
  }
  for (const [name, body] of Object.entries(bodies)) {
    if (!SOURCE_PATHS[name].endsWith(".rs")) continue;
    const literal = body.match(new RegExp(`\\b${typeName}\\s*\\{([^{}]*)\\}`));
    if (!literal) continue;
    const named = parseNamedFields(literal[1]).map((field) => field.name);
    // Shorthand initializers (`MemoryGeometry { width, height, .. }`) carry the field name alone.
    const shorthand = splitTopLevel(literal[1])
      .filter((part) => /^[A-Za-z_]\w*$/.test(part))
      .map((part) => part.trim());
    const fields = [...new Set([...named, ...shorthand])].map((field) => ({
      name: field,
      type: "",
    }));
    return fields.length ? { fields, via: "struct_literal" } : null;
  }
  return null;
}

/**
 * The geometry axes a struct type carries, following nested geometry-bearing fields.
 *
 * `seen` is the CURRENT PATH, not a global visit log, and the bound is on DEPTH. A shared visit set
 * bounded by size silently truncates a wide struct: `VideoCurveQuery` names four unrelated enums
 * before its `geometry` field, so a five-entry global budget stops looking exactly one field early
 * and reports the one geometry-aware gate in `sceneworks-core` as geometry-blind.
 */
function structGeometryAxes(bodies, typeName, depth = 0, seen = new Set()) {
  if (depth > 3 || seen.has(typeName)) return { axes: new Set(), resolved: true };
  const resolved = resolveStructFields(bodies, typeName);
  if (!resolved) return { axes: new Set(), resolved: false };
  const path = new Set([...seen, typeName]);
  const axes = new Set();
  for (const field of resolved.fields) {
    if (GEOMETRY_FIELDS.includes(field.name)) axes.add(field.name);
    const nested = bareTypeName(field.type);
    if (!nested || SCALAR_TYPE.test(nested)) continue;
    // A nested type that resolves to nothing is not a failure of THIS parameter — enums and opaque
    // handles are expected. Only the parameter's own type failing to resolve is reportable.
    for (const axis of structGeometryAxes(bodies, nested, depth + 1, path).axes) axes.add(axis);
  }
  return { axes, resolved: true };
}

/**
 * Resolve ONE gate function's geometry-awareness. Returns the axes, how they were reached, and any
 * parameter type that could not be resolved â€” published rather than silently read as "no geometry".
 */
export function resolveSymbolGeometry(bodies, sourceKey, symbol) {
  const body = bodies[sourceKey];
  if (!body) return null;
  const signature = body.match(
    new RegExp(`fn\\s+${symbol}\\s*(?:<[^>{(]*>)?\\s*\\(([\\s\\S]*?)\\)\\s*(?:->|\\{)`),
  );
  if (!signature) return null;

  const axes = new Set();
  const reachedVia = new Set();
  const unresolved = [];
  for (const parameter of parseNamedFields(signature[1])) {
    const type = bareTypeName(parameter.type);
    if (GEOMETRY_FIELDS.includes(parameter.name) && (!type || SCALAR_TYPE.test(type))) {
      axes.add(parameter.name);
      reachedVia.add("scalar_parameter");
      continue;
    }
    if (!type || SCALAR_TYPE.test(type)) continue;
    const { axes: structAxes, resolved } = structGeometryAxes(bodies, type);
    if (!resolved) {
      unresolved.push({ parameter: parameter.name, type });
      // A parameter NAMED for geometry whose type cannot be resolved is the failure mode that
      // produced this whole class of bug. Refuse rather than under-report it.
      if (/geom/i.test(parameter.name)) {
        throw new Error(
          `${symbol}: parameter \`${parameter.name}: ${type}\` looks like request geometry, but ` +
            `${type}'s fields could not be resolved from any fingerprinted source (no struct ` +
            "definition and no struct-literal construction). Add the declaring or constructing " +
            "source to SOURCE_PATHS; reporting the gate as geometry-blind would invert the truth.",
        );
      }
      continue;
    }
    if (structAxes.size > 0) {
      for (const axis of structAxes) axes.add(axis);
      reachedVia.add(`struct_parameter:${parameter.name}:${type}`);
    }
  }

  return {
    symbol,
    geometryAware: axes.size > 0,
    geometryAxes: [...axes].sort(),
    reachedVia: [...reachedVia].sort(),
    unresolvedParameterTypes: unresolved,
  };
}

/**
 * Per-mechanism derived facts: the modules that call its entry symbols, and â€” per SYMBOL, never
 * unioned â€” whether that gate function is geometry-aware.
 *
 * `geometryAware` at the mechanism level is retained only as "at least one of my symbols is", which
 * is what the summary table reports. Route rows and baseline rows bind to `symbols[]` instead, so a
 * `wan_video_fit_error` route is never labelled geometry-aware because `svd_fit_error` is.
 */
export function deriveMechanismFacts(bodies) {
  const facts = [];
  for (const mechanism of ADMISSION_MECHANISMS) {
    const callers = new Set();
    if (mechanism.callPattern) {
      const call = new RegExp(mechanism.callPattern);
      for (const [name, body] of Object.entries(bodies)) {
        if (!SOURCE_PATHS[name].endsWith(".rs")) continue;
        if (call.test(productionBody(body))) callers.add(name);
      }
    }
    const symbols = [];
    if (mechanism.source) {
      for (const symbol of mechanism.definitionSymbols) {
        const resolved = resolveSymbolGeometry(bodies, mechanism.source, symbol);
        if (resolved) symbols.push(resolved);
      }
      if (symbols.length === 0) {
        throw new Error(
          `mechanism "${mechanism.id}": none of its definition symbols exist in ` +
            `${SOURCE_PATHS[mechanism.source]}. The taxonomy has drifted from the tree.`,
        );
      }
    }
    const unionAxes = new Set(symbols.flatMap((entry) => entry.geometryAxes));
    facts.push({
      id: mechanism.id,
      summary: mechanism.summary,
      definedIn: mechanism.source ? SOURCE_PATHS[mechanism.source] : null,
      definitionSymbols: [...mechanism.definitionSymbols],
      symbols,
      // "Some symbol of this mechanism is geometry-aware" â€” a summary, NOT a per-route claim.
      geometryAware: symbols.some((entry) => entry.geometryAware),
      geometryAxes: [...unionAxes].sort(),
      geometryAwareSymbols: symbols.filter((entry) => entry.geometryAware).map((e) => e.symbol),
      geometryBlindSymbols: symbols.filter((entry) => !entry.geometryAware).map((e) => e.symbol),
      calledFrom: [...callers].sort().map((name) => SOURCE_PATHS[name]),
    });
  }
  return facts;
}

/**
 * `gate label -> geometry fact`, keyed by the labels the Rust emitter stamps on every decision row.
 * This is the join that lets `candle_admission_decisions.rs`'s `gate_is_geometry_aware` match be
 * checked against a derivation instead of standing as a second hand-written claim: `buildBaseline`
 * throws when the two disagree for any gate that appears in the decisions.
 */
export function deriveGateGeometry(mechanismFacts) {
  const gates = new Map();
  for (const mechanism of mechanismFacts) {
    for (const entry of mechanism.symbols) {
      const labels = [entry.symbol];
      // The Rust side labels the scalar gate and the two module-owned gates by their module path.
      if (entry.symbol === "load_plan") labels.push("candle_scalar_gate::load_plan");
      if (entry.symbol === "fit_ladder_for_entry_with_runtime") {
        labels.push("krea_control_fit::fit_ladder_for_entry_with_runtime");
      }
      if (entry.symbol === "decide") labels.push("conditioning_fit::decide");
      for (const label of labels) {
        gates.set(label, {
          gate: label,
          mechanism: mechanism.id,
          definedIn: mechanism.definedIn,
          geometryAware: entry.geometryAware,
          geometryAxes: entry.geometryAxes,
          reachedVia: entry.reachedVia,
        });
      }
    }
    // sc-19054 (epic 19048 R3): the module-qualified `candle_scalar_gate::load_plan` label names
    // the route-reached COMPOSITION — since this story the peak `load_plan` compares is computed
    // by `predicted_peak_gb_for_request(…, geometry: MemoryGeometry)`, so the label's geometry
    // fact is the composition of the two signatures. The bare `load_plan` symbol entry above
    // keeps its own signature-derived (geometry-blind) fact: this is a dataflow composition
    // within one gate path, not a union across mechanism members.
    const plan = mechanism.symbols.find((entry) => entry.symbol === "load_plan");
    const prediction = mechanism.symbols.find(
      (entry) => entry.symbol === "predicted_peak_gb_for_request",
    );
    if (plan && prediction) {
      gates.set("candle_scalar_gate::load_plan", {
        gate: "candle_scalar_gate::load_plan",
        mechanism: mechanism.id,
        definedIn: mechanism.definedIn,
        geometryAware: plan.geometryAware || prediction.geometryAware,
        geometryAxes: [...new Set([...plan.geometryAxes, ...prediction.geometryAxes])].sort(),
        reachedVia: [
          ...new Set([
            ...plan.reachedVia,
            ...prediction.reachedVia.map((via) => `composed:predicted_peak_gb_for_request:${via}`),
          ]),
        ].sort(),
      });
    }
  }
  // The sentinel the emitter uses for a route no gate is keyed to.
  gates.set("none", {
    gate: "none",
    mechanism: "unreached",
    definedIn: null,
    geometryAware: false,
    geometryAxes: [],
    reachedVia: [],
  });
  return gates;
}

/** `IMAGE_MODEL_CAPS` / `VIDEO_MODEL_CAPS` rows, split by modality so a route knows its own axes. */
export function parseCatalogModalities(routingCatalog) {
  const image = new Set();
  const video = new Set();
  for (const [constructor, target] of [
    ["ModelCaps", image],
    ["VideoModelCaps", video],
  ]) {
    const pattern = new RegExp(
      `${constructor}::new\\(\\s*"([^"]+)"\\s*,\\s*(true|false)\\s*,\\s*(true|false)\\s*,`,
      "g",
    );
    for (const match of routingCatalog.matchAll(pattern)) {
      if (match[3] === "true") target.add(match[1]);
    }
  }
  // Bespoke direct-slice video routes (`CANDLE_SCAIL2_VIDEO_MODELS`), resolved the same way the
  // tier-integrity lane oracle resolves them so the two universes cannot disagree.
  const slices = new Map();
  for (const match of routingCatalog.matchAll(
    /(?:pub\(crate\)\s+)?const\s+([A-Z][A-Z0-9_]*)\s*:\s*&\[&str\]\s*=\s*&\[([^\]]*)\]/g,
  )) {
    slices.set(match[1], [...match[2].matchAll(/"([^"]+)"/g)].map((id) => id[1]));
  }
  const candleVideoRouteBody = routingCatalog.match(
    /fn\s+video_model_has_candle_video_route\([^)]*\)[^{]*\{([\s\S]*?)\n\}/,
  )?.[1];
  for (const match of (candleVideoRouteBody ?? "").matchAll(
    /([A-Z][A-Z0-9_]*)\.contains\(&model\)/g,
  )) {
    for (const id of slices.get(match[1]) ?? []) video.add(id);
  }
  if (image.size === 0 || video.size === 0) {
    throw new Error("routing/catalog.rs: parsed zero candle image or video routes");
  }
  return { image, video };
}

/**
 * `CANDLE_IMAGE_ROUTES` as `Map<laneVariant, Set<modelId>>`, resolving both `ModelMatch::Any` and
 * `ModelMatch::Family(is_*)` exactly as the tier-integrity lane oracle resolves them, so the two
 * readings of the same table cannot disagree.
 */
export function parseCandleImageLanes(routingCandle) {
  const table = routingCandle.match(
    /const CANDLE_IMAGE_ROUTES: &\[CandleImageRoute\] = &\[([\s\S]*?)\n\];/,
  );
  if (!table) throw new Error("routing/candle.rs: could not locate CANDLE_IMAGE_ROUTES");
  const familyIds = new Map();
  for (const match of routingCandle.matchAll(
    /fn (is_\w+)\(model: &str\) -> bool \{\s*matches!\(\s*model,([^)]*)\)/g,
  )) {
    familyIds.set(match[1], [...match[2].matchAll(/"([^"]+)"/g)].map((id) => id[1]));
  }
  const lanes = new Map();
  for (const row of table[1].matchAll(/CandleImageRoute \{([\s\S]*?)\n {4}\},/g)) {
    const lane = row[1].match(/lane: CandleImageLane::(\w+)/)?.[1];
    if (!lane) continue;
    if (!lanes.has(lane)) lanes.set(lane, new Set());
    const target = lanes.get(lane);
    const any = row[1].match(/ModelMatch::Any\(&\[([\s\S]*?)\]\)/);
    for (const id of (any?.[1] ?? "").matchAll(/"([^"]+)"/g)) target.add(id[1]);
    const family = row[1].match(/ModelMatch::Family\((\w+)\)/)?.[1];
    for (const id of familyIds.get(family) ?? []) target.add(id);
  }
  if (lanes.size === 0) throw new Error("routing/candle.rs: CANDLE_IMAGE_ROUTES parsed to zero lanes");
  return lanes;
}

/**
 * Resolve [`IMAGE_LANE_BINDINGS`] against the parsed lane table and the derived mechanism facts.
 * Throws when a bound lane has left the route table or its module no longer calls any admission
 * entry symbol â€” the two ways a declared binding rots into a stale hand-written claim.
 */
export function deriveLaneBindings({ imageLanes, mechanismFacts }) {
  const mechanismsBySource = new Map();
  for (const mechanism of mechanismFacts) {
    for (const relative of mechanism.calledFrom) {
      if (!mechanismsBySource.has(relative)) mechanismsBySource.set(relative, new Set());
      mechanismsBySource.get(relative).add(mechanism.id);
    }
  }
  const bindings = [];
  for (const [lane, sourceKey] of Object.entries(IMAGE_LANE_BINDINGS)) {
    if (!imageLanes.has(lane)) {
      throw new Error(
        `image lane binding "${lane}": CANDLE_IMAGE_ROUTES no longer serves that lane`,
      );
    }
    const relative = SOURCE_PATHS[sourceKey];
    const called = mechanismsBySource.get(relative);
    if (!called || called.size === 0) {
      throw new Error(
        `image lane binding "${lane}": ${relative} calls no admission entry symbol, so the lane's ` +
          "gate can no longer be derived from it",
      );
    }
    bindings.push({
      lane,
      module: relative,
      models: [...imageLanes.get(lane)].sort(),
      mechanisms: MODULE_KEYED_MECHANISMS.filter((id) => called.has(id)),
      allMechanisms: [...called].sort(),
    });
  }
  const unbound = [...imageLanes.keys()].filter((lane) => !(lane in IMAGE_LANE_BINDINGS)).sort();
  return { bindings, unbound };
}

/** SceneWorks model id -> candle video engine id, from `candle_video_engine_id`'s match. */
export function parseCandleVideoEngines(videoRouteCandle) {
  const body = videoRouteCandle.match(
    /fn candle_video_engine_id\(model: &str\) -> Option<&'static str> \{\s*match model \{([\s\S]*?)\n {4}\}/,
  );
  if (!body) {
    throw new Error("video_jobs/candle.rs: could not locate candle_video_engine_id's match");
  }
  const engines = new Map();
  for (const arm of body[1].matchAll(/"([^"]+)"\s*=>\s*Some\("([^"]+)"\)/g)) {
    engines.set(arm[1], arm[2]);
  }
  if (engines.size === 0) {
    throw new Error("video_jobs/candle.rs: candle_video_engine_id parsed to zero engines");
  }
  return engines;
}

/**
 * Verify [`VIDEO_ADMISSION_BINDINGS`] against the tree and return it resolved. Throws on any drift,
 * which is what stops the declared half from rotting into a stale hand-written list.
 */
export function deriveVideoBindings({ bodies, videoEngines }) {
  const vramGate = bodies.vramGate;
  const videoCallSites = productionBody(bodies.videoRouteCandle);
  const routedEngines = new Set(videoEngines.values());
  const resolved = [];
  for (const binding of VIDEO_ADMISSION_BINDINGS) {
    if (!new RegExp(`fn\\s+${binding.symbol}\\s*\\(`).test(vramGate)) {
      throw new Error(
        `video admission binding ${binding.engineId}: vram_gate.rs defines no fn ${binding.symbol}`,
      );
    }
    if (!new RegExp(`(?<!fn\\s)\\b${binding.symbol}\\s*\\(`).test(videoCallSites)) {
      throw new Error(
        `video admission binding ${binding.engineId}: video_jobs/candle.rs never calls ${binding.symbol}`,
      );
    }
    if (!binding.offMatch && !routedEngines.has(binding.engineId)) {
      throw new Error(
        `video admission binding ${binding.engineId}: candle_video_engine_id no longer produces it`,
      );
    }
    resolved.push({ ...binding });
  }
  // Every routed video engine must be bound, or the inventory would under-report a live gate.
  for (const engineId of [...routedEngines].sort()) {
    if (!resolved.some((binding) => binding.engineId === engineId)) {
      throw new Error(
        `candle video engine "${engineId}" is routed but has no VIDEO_ADMISSION_BINDINGS entry`,
      );
    }
  }
  return resolved;
}

/** SceneWorks model id -> engine/provider id, from the worker's static image routing table. */
export function parseModelTable(engines) {
  const table = engines.match(/pub\(crate\) const MODEL_TABLE:[\s\S]*?=\s*&\[([\s\S]*?)\n\];/);
  if (!table) throw new Error("engines.rs: could not locate MODEL_TABLE");
  const routes = new Map();
  for (const row of table[1].matchAll(/ModelRow\s*\{([\s\S]*?)\n\s*\},/g)) {
    const model = row[1].match(/sceneworks_id:\s*"([^"]+)"/)?.[1];
    const engine = row[1].match(/engine_id:\s*"([^"]+)"/)?.[1];
    if (model && engine) routes.set(model, engine);
  }
  if (routes.size === 0) throw new Error("engines.rs: MODEL_TABLE parsed to zero routes");
  return routes;
}

/** The candle providers `memory_route_registry.rs` declares load-shape rules for. */
export function parseRegistryCandleProviders(memoryRouteRegistry) {
  const table = memoryRouteRegistry.match(/const RULES: &\[MemoryRouteRule\] = &\[([\s\S]*?)\n\];/);
  if (!table) throw new Error("memory_route_registry.rs: could not locate RULES");
  const providers = new Set();
  for (const row of table[1].matchAll(/MemoryRouteRule \{([\s\S]*?)\n {4}\},/g)) {
    const backend = row[1].match(/backend: MemoryRouteBackend::(\w+)/)?.[1];
    const provider = row[1].match(/provider: "([a-z0-9_]+)"/)?.[1];
    if (backend === "Candle" && provider) providers.add(provider);
  }
  if (providers.size === 0) {
    throw new Error("memory_route_registry.rs: RULES parsed to zero candle providers");
  }
  return providers;
}

/** `DEDICATED_VRAM_ALLOCATOR_SLACK_GB` â€” the headroom `vram_gate::HEADROOM_GB` aliases. */
export function parseHeadroomGb(fitGate) {
  const match = fitGate.match(
    /const DEDICATED_VRAM_ALLOCATOR_SLACK_GB:\s*f64\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*;/,
  );
  if (!match) throw new Error("fit_gate.rs: could not read DEDICATED_VRAM_ALLOCATOR_SLACK_GB");
  return Number(match[1]);
}

/** The lanes the packaged video curve bundles were measured on, from the committed curve data. */
export function parseVideoCurveLanes(videoMemoryCurvesData) {
  const parsed = JSON.parse(videoMemoryCurvesData);
  const bundles = Array.isArray(parsed) ? parsed : (parsed.bundles ?? parsed.curves ?? []);
  const lanes = new Map();
  const visit = (node) => {
    if (!node || typeof node !== "object") return;
    if (Array.isArray(node)) return node.forEach(visit);
    const backend = typeof node.backend === "string" ? node.backend.toLowerCase() : null;
    if (backend) lanes.set(backend, (lanes.get(backend) ?? 0) + 1);
    for (const value of Object.values(node)) visit(value);
  };
  visit(bundles.length ? bundles : parsed);
  return lanes;
}

// -------------------------------------------------------------------------------------------------
// The candle legacy scalar gate, MIRRORED. Every branch below is a line-for-line reading of
// `crates/sceneworks-worker/src/candle_scalar_gate.rs`.
//
// This mirror does NOT produce the baseline any more â€” `candle_admission_decisions.rs` does, by
// calling the Rust functions themselves. It is kept because a readable statement of the admission
// law beside the inventory is worth having, and because a second independent implementation catches
// a class of transcription error a single implementation cannot. It is only defensible while it is
// CHECKED: `generate-candle-admission-inventory.test.mjs`'s "the JS gate mirror reproduces the
// Rust-emitted decision for every evaluated row" reconciles the two coordinate by coordinate and
// fails on any drift. Delete the mirror rather than let that test lapse.
// -------------------------------------------------------------------------------------------------

const NVFP4_TIER = "nvfp4";

/** `vram_gate::predicted_peak_gb`. */
export function predictedPeakGb(entry, tierKey, headroomGb) {
  const candle = entry?.candle;
  if (!candle) return null;
  const measured = (key) => {
    const value = candle.vramGbByTier?.[key];
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  };
  const direct = measured(tierKey);
  if (direct !== null) return direct + headroomGb;
  if (tierKey === NVFP4_TIER) {
    const q8 = measured("q8");
    if (q8 !== null) return q8 + headroomGb;
  }
  const floor = candle.minMemoryGb;
  return typeof floor === "number" && Number.isFinite(floor) ? floor : null;
}

/** `vram_gate::predicted_sequential_peak_gb`. */
export function predictedSequentialPeakGb(entry, tierKey, headroomGb) {
  const sequential = entry?.candle?.sequentialPeakGb;
  if (!sequential) return null;
  const measured = (key) => {
    const value = sequential[key];
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  };
  const direct = measured(tierKey);
  const resolved = direct !== null ? direct : tierKey === NVFP4_TIER ? measured("q8") : null;
  return resolved === null ? null : resolved + headroomGb;
}

/** `vram_gate::fit_decision`. `budget` is `{ freeGb, totalGb }`, or null for "no live budget". */
export function fitDecision(neededGb, budget) {
  if (neededGb === null || budget === null) return "unknown";
  return budget.freeGb + Number.EPSILON < neededGb ? "too_big" : "fits";
}

/** `fit_gate::resolve_offload`. */
export function resolveOffload(decision, sequentialCapable) {
  return decision === "too_big" && sequentialCapable ? "offload" : decision;
}

/** `vram_gate::sequential_overflow_gb` â€” truthy when the staged peak ALSO overflows. */
export function sequentialOverflows(sequentialNeededGb, budget) {
  if (sequentialNeededGb === null || budget === null) return false;
  return budget.freeGb + Number.EPSILON < sequentialNeededGb;
}

/** `vram_gate::load_plan`. */
export function loadPlan(neededGb, sequentialNeededGb, budget, sequentialCapable) {
  const decision = resolveOffload(fitDecision(neededGb, budget), sequentialCapable);
  if (decision === "offload") {
    return sequentialOverflows(sequentialNeededGb, budget) ? "reject" : "sequential";
  }
  if (decision === "too_big") return "reject";
  return "resident";
}

// ── sc-19054 (epic 19048 R3): the geometry-aware scalar grade, MIRRORED. ────────────────────────

const BYTES_PER_GIB = 1024 ** 3;

/**
 * `ladder_margin_policy::CANDLE_ESTIMATE_MARGIN`, restated through its own derivation chain
 * (`derive-ladder-margins.mjs`: the candle corpus has zero repeat pairs, so the stale margin IS
 * the hard floor, and the estimate margin doubles it) rather than as a second literal.
 */
export const CANDLE_ESTIMATE_MARGIN = CANDLE_HARD_FLOOR * ESTIMATE_WIDENING_MULTIPLIER;

/** `candle_scalar_gate::scalar_peak_class` + `estimate_synthesis::declared_scalar_class`. */
export function scalarPeakClass(entry, rowsKey, tierKey, geometry) {
  const candle = entry?.candle;
  const own = candle?.[rowsKey]?.[tierKey];
  if (!(typeof own === "number" && Number.isFinite(own))) return "declared_floor";
  const measured = candle.measured === true;
  const declaredPixels =
    typeof candle.vramMeasuredPixels === "number" &&
    Number.isInteger(candle.vramMeasuredPixels) &&
    candle.vramMeasuredPixels >= 0
      ? candle.vramMeasuredPixels
      : null;
  const covered =
    measured &&
    geometry.batch === 1 &&
    geometry.frames === 1 &&
    declaredPixels !== null &&
    declaredPixels > 0 &&
    geometry.width * geometry.height <= declaredPixels;
  return covered ? "measured_peak" : "declared_floor";
}

/**
 * `candle_scalar_gate::graded_scalar_gb`: a covering measured peak compares raw; a declared floor
 * is widened by the candle ESTIMATE margin over integer bytes — the selector's own
 * `widened_peak_bytes` arithmetic, never a second law.
 */
export function gradedScalarGb(peakGb, scalarClass) {
  if (scalarClass === "measured_peak") return peakGb;
  const bytes = Math.ceil(peakGb * BYTES_PER_GIB);
  return Math.ceil(bytes * (1 + CANDLE_ESTIMATE_MARGIN)) / BYTES_PER_GIB;
}

/** `vram_gate::predicted_peak_gb_for_request` (adapter bytes held at the corpus's zero). */
export function predictedPeakGbForRequest(entry, tierKey, headroomGb, geometry) {
  const raw = predictedPeakGb(entry, tierKey, headroomGb);
  if (raw === null) return null;
  return gradedScalarGb(raw, scalarPeakClass(entry, "vramGbByTier", tierKey, geometry));
}

/** `vram_gate::predicted_sequential_peak_gb_for_request`. */
export function predictedSequentialPeakGbForRequest(entry, tierKey, headroomGb, geometry) {
  const raw = predictedSequentialPeakGb(entry, tierKey, headroomGb);
  if (raw === null) return null;
  return gradedScalarGb(raw, scalarPeakClass(entry, "sequentialPeakGb", tierKey, geometry));
}

// -------------------------------------------------------------------------------------------------
// Document assembly
// -------------------------------------------------------------------------------------------------

function manifestEntries(manifestBody) {
  const parsed = JSON.parse(stripJsoncComments(manifestBody));
  return new Map((parsed.models ?? []).map((model) => [model.id, model]));
}

function candleEvidence(entry) {
  const candle = entry?.candle;
  if (!candle) {
    return {
      hasCandleBlock: false,
      measured: null,
      measurementLane: null,
      measuredTiers: [],
      sequentialTiers: [],
      minMemoryGb: null,
      supportsSequentialOffload: false,
      declaresProviderContract: false,
      declaresTurboFit: false,
      declaresControlFit: false,
    };
  }
  return {
    hasCandleBlock: true,
    // A BARE BOOLEAN, not an evidence class. See the header note.
    measured: typeof candle.measured === "boolean" ? candle.measured : null,
    measurementLane: typeof candle.measurementLane === "string" ? candle.measurementLane : null,
    measuredTiers: Object.keys(candle.vramGbByTier ?? {}).sort(),
    sequentialTiers: Object.keys(candle.sequentialPeakGb ?? {}).sort(),
    minMemoryGb: typeof candle.minMemoryGb === "number" ? candle.minMemoryGb : null,
    supportsSequentialOffload: candle.supportsSequentialOffload === true,
    declaresProviderContract: Boolean(candle.memoryStrategyContract),
    declaresTurboFit: Boolean(candle.turboFit),
    declaresControlFit: Boolean(candle.control),
  };
}

/**
 * Classify one candle-routed model. Returns the ordered mechanism id list plus the shared-selector
 * verdict, which the story requires to distinguish three states that used to be conflated:
 * a NAMED evidence revision, the sc-18456 DECLARATION catch-all, and never reaching it at all.
 */
function classifyRoute({
  modelId,
  engineId,
  modality,
  entry,
  evidence,
  dispatch,
  laneMechanisms,
  lanes,
  conditionedLanes,
  videoBinding,
  registryProviders,
  compatibility,
}) {
  const mechanisms = [...laneMechanisms];
  const named = engineId ? dispatch.named.get(engineId) : undefined;
  let sharedSelector;
  if (compatibility.legacyScalar.has(modelId)) {
    sharedSelector = {
      reached: true,
      via: "legacy_scalar_compatibility",
      evidenceRevision: "sc-19055-candle-compatibility-v1",
    };
    mechanisms.push("shared_selector_legacy_scalar_compatibility");
  } else if (compatibility.unverifiedImage.has(modelId)) {
    sharedSelector = {
      reached: true,
      via: "unverified_compatibility",
      evidenceRevision: null,
    };
    mechanisms.push("shared_selector_unverified_compatibility");
  } else if (compatibility.structuralFloor.has(modelId)) {
    sharedSelector = {
      reached: true,
      via: "structural_floor_compatibility",
      evidenceRevision: "sc-19055-candle-compatibility-v1",
    };
    mechanisms.push("shared_selector_structural_floor_compatibility");
  } else if (compatibility.legacyVideo.has(modelId)) {
    sharedSelector = {
      reached: true,
      via: "legacy_video_compatibility",
      evidenceRevision: "sc-19055-candle-compatibility-v1",
    };
    mechanisms.push("shared_selector_legacy_video_compatibility");
  } else if (laneMechanisms.includes("shared_selector_bespoke_override")) {
    const override = dispatch.overrideOnly[0] ?? null;
    sharedSelector = {
      reached: true,
      via: "bespoke_override",
      evidenceRevision: override?.value ?? null,
    };
  } else if (named) {
    sharedSelector = { reached: true, via: "named_revision", evidenceRevision: named.value };
    mechanisms.push("shared_selector_named_revision");
  } else if (evidence.declaresProviderContract) {
    sharedSelector = {
      reached: true,
      via: "declaration_catch_all",
      evidenceRevision: dispatch.catchAll.value,
    };
    mechanisms.push("shared_selector_declaration_catch_all");
  } else {
    sharedSelector = { reached: false, via: null, evidenceRevision: null };
  }

  if (evidence.declaresTurboFit) mechanisms.push("krea_turbo_fit");
  if (evidence.declaresControlFit) mechanisms.push("krea_control_fit");
  if (videoBinding) mechanisms.push("flat_video_fit_error");
  // The scalar gate is live for any image route whose manifest supplies a number for it to read.
  if (
    modality === "image" &&
    evidence.hasCandleBlock &&
    (evidence.measuredTiers.length > 0 || evidence.minMemoryGb !== null)
  ) {
    mechanisms.push("legacy_scalar_gate");
  }
  if (mechanisms.length === 0) mechanisms.push("unreached");

  return {
    modelId,
    engineId,
    modality,
    manifestEntry: Boolean(entry),
    loadShapeRule: Boolean(engineId && registryProviders.has(engineId)),
      // Every conditioned lane the router serves this model on, and â€” separately â€” the subset whose
    // handler module is bound in IMAGE_LANE_BINDINGS and therefore contributed a mechanism. The
    // difference between the two lists is exactly the classification debt a later slice pays down.
    conditionedLanes: [...conditionedLanes].sort(),
    overlayLanes: [...lanes].sort(),
    sharedSelector,
    videoFitSymbol: videoBinding?.symbol ?? null,
    mechanisms: [...new Set(mechanisms)].sort(),
    evidence,
  };
}

/**
 * Build the inventory document. Pure over `bodies` so `--self-test` and the unit tests can drive it
 * with mutated sources.
 */
export function buildInventory(bodies) {
  validateCompatibilityWiring(bodies);
  const entries = manifestEntries(bodies.manifest);
  const scalarEntries = [...entries.values()].filter((entry) => {
    const candle = entry.candle;
    return Boolean(
      candle &&
        ("minMemoryGb" in candle || "vramGbByTier" in candle || "sequentialPeakGb" in candle),
    );
  });
  if (scalarEntries.length !== 45) {
    throw new Error(`expected the exact 45 shipped Candle scalar manifests, found ${scalarEntries.length}`);
  }
  const wrongLane = scalarEntries.filter((entry) => entry.candle.measurementLane !== "candle");
  if (wrongLane.length > 0) {
    throw new Error(
      `Candle scalar manifests require measurementLane=candle: ${wrongLane.map((entry) => entry.id).join(", ")}`,
    );
  }
  const compatibility = {
    legacyScalar: parseCompatibilityRoutes(
      bodies.candleMemoryStrategy,
      "LEGACY_SCALAR_COMPATIBILITY_ROUTES",
    ),
    unverifiedImage: parseCompatibilityRoutes(
      bodies.candleMemoryStrategy,
      "UNVERIFIED_IMAGE_COMPATIBILITY_ROUTES",
    ),
    structuralFloor: parseCompatibilityRoutes(
      bodies.candleMemoryStrategy,
      "STRUCTURAL_FLOOR_COMPATIBILITY_ROUTES",
    ),
    legacyVideo: parseCompatibilityRoutes(
      bodies.videoAdmission,
      "LEGACY_VIDEO_COMPATIBILITY_ROUTES",
    ),
  };
  const lanes = routedLanes({
    routingCatalog: bodies.routingCatalog,
    routingCandle: bodies.routingCandle,
    routingMlx: bodies.routingMlx,
  });
  const modalities = parseCatalogModalities(bodies.routingCatalog);
  const dispatch = parseRequestScopeDispatch(bodies.candleMemoryStrategy);
  const bespokeOverrides = parseBespokeOverrideEngines(bodies);
  const modelTable = parseModelTable(bodies.engines);
  const registryProviders = parseRegistryCandleProviders(bodies.memoryRouteRegistry);
  const videoEngines = parseCandleVideoEngines(bodies.videoRouteCandle);
  const videoBindings = deriveVideoBindings({ bodies, videoEngines });
  const headroomGb = parseHeadroomGb(bodies.fitGate);
  const curveLanes = parseVideoCurveLanes(bodies.videoMemoryCurvesData);
  const mechanismFacts = deriveMechanismFacts(bodies);
  const imageLanes = parseCandleImageLanes(bodies.routingCandle);
  const { bindings: laneBindings, unbound: unboundLanes } = deriveLaneBindings({
    imageLanes,
    mechanismFacts,
  });
  const { revision, perSource } = sourceRevisionOf(bodies);

  // model id -> every conditioned lane the router serves it on, bound or not.
  const conditionedLanesByModel = new Map();
  for (const [lane, models] of imageLanes) {
    for (const modelId of models) {
      if (!conditionedLanesByModel.has(modelId)) conditionedLanesByModel.set(modelId, new Set());
      conditionedLanesByModel.get(modelId).add(lane);
    }
  }

  // model id -> { lanes, mechanisms } contributed by the overlay/bespoke lanes that serve it.
  const laneContributions = new Map();
  for (const binding of laneBindings) {
    for (const modelId of binding.models) {
      if (!laneContributions.has(modelId)) {
        laneContributions.set(modelId, { lanes: new Set(), mechanisms: new Set() });
      }
      const contribution = laneContributions.get(modelId);
      contribution.lanes.add(binding.lane);
      for (const id of binding.mechanisms) contribution.mechanisms.add(id);
    }
  }
  // The bespoke-override engine literals must be reachable from a bound lane, or the inventory
  // would report an override scope that no route can actually enter.
  if (bespokeOverrides.size > 0) {
    const overrideLanes = laneBindings.filter((binding) =>
      binding.mechanisms.includes("shared_selector_bespoke_override"),
    );
    if (overrideLanes.length === 0) {
      throw new Error(
        `evaluate_shared_bespoke_image is called with ${[...bespokeOverrides].join(", ")} but no ` +
          "bound image lane reaches the calling module; add the lane to IMAGE_LANE_BINDINGS",
      );
    }
  }

  const candleIds = [...lanes.entries()]
    .filter(([, laneSet]) => laneSet.has("candle"))
    .map(([id]) => id)
    .sort();

  const bindingByEngine = new Map(videoBindings.map((binding) => [binding.engineId, binding]));
  const routes = candleIds.map((modelId) => {
    const entry = entries.get(modelId) ?? null;
    const videoEngineId = videoEngines.get(modelId) ?? null;
    const modality = modalities.video.has(modelId) ? "video" : "image";
    const engineId = videoEngineId ?? modelTable.get(modelId) ?? null;
    const videoBinding =
      modality === "video"
        ? (bindingByEngine.get(videoEngineId ?? "") ?? bindingByEngine.get(modelId) ?? null)
        : null;
    const contribution = laneContributions.get(modelId);
    return classifyRoute({
      modelId,
      engineId,
      modality,
      entry,
      evidence: candleEvidence(entry),
      dispatch,
      laneMechanisms: [...(contribution?.mechanisms ?? [])],
      lanes: contribution?.lanes ?? new Set(),
      conditionedLanes: conditionedLanesByModel.get(modelId) ?? new Set(),
      videoBinding,
      registryProviders,
      compatibility,
    });
  });

  // ---------------------------------------------------------------------------------------------
  // Overlay providers: admission routes that are NOT a model id.
  //
  // A control provider has no catalog row of its own â€” `z_image_control` is reached as an overlay
  // on `z_image`'s route, and the manifest declares it as a `runtimeProvider` split on the base
  // model's contract. It is nonetheless a distinct admission coordinate: it has its own named
  // request scope in `candle_memory_strategy.rs` and its own load-shape rule in
  // `memory_route_registry.rs`. Keying the inventory only on model ids would drop it, and dropping
  // it is how a migration ships having "converged every route" while a control overlay still
  // predicts on the old mechanism.
  // ---------------------------------------------------------------------------------------------
  const routedEngineIds = new Set(routes.map((route) => route.engineId).filter(Boolean));
  const overlayProviderIds = new Set(
    [...registryProviders, ...dispatch.named.keys()].filter((id) => !routedEngineIds.has(id)),
  );
  const overlayProviders = [...overlayProviderIds].sort().map((providerId) => {
    const named = dispatch.named.get(providerId);
    const baseModels = [...entries.values()]
      .filter(
        (model) =>
          lanes.get(model.id)?.has("candle") && declaredProviders(model, "candle").has(providerId),
      )
      .map((model) => model.id)
      .sort();
    return {
      providerId,
      loadShapeRule: registryProviders.has(providerId),
      sharedSelector: named
        ? { reached: true, via: "named_revision", evidenceRevision: named.value }
        : baseModels.length > 0
          ? {
              reached: true,
              via: "declaration_catch_all",
              evidenceRevision: dispatch.catchAll.value,
            }
          : { reached: false, via: null, evidenceRevision: null },
      baseModels,
    };
  });

  // Divergences: the ledger sc-19059 must drive to empty. Each one is a concrete, checkable claim.
  const divergences = [];
  for (const route of routes) {
    if (!route.manifestEntry) {
      divergences.push({
        kind: "routed_without_manifest_entry",
        modelId: route.modelId,
        detail:
          "the router serves a candle lane for this id, but the builtin manifest has no entry, so " +
          "no declared number and no provider contract can be read for it",
      });
    }
    if (!route.sharedSelector.reached) {
      divergences.push({
        kind: "shared_selector_unreached",
        modelId: route.modelId,
        detail: `no named request scope for engine "${route.engineId ?? "<unrouted>"}" and no candle.memoryStrategyContract`,
      });
    } else if (route.sharedSelector.via === "declaration_catch_all") {
      divergences.push({
        kind: "shared_selector_via_catch_all",
        modelId: route.modelId,
        detail: `reaches the selector only through ${dispatch.catchAll.constant}, with no engine-keyed evidence revision`,
      });
    }
    if (route.mechanisms.includes("legacy_scalar_gate") && route.evidence.measuredTiers.length === 0) {
      divergences.push({
        kind: "declared_floor_only",
        modelId: route.modelId,
        detail: "the scalar gate has no candle.vramGbByTier row and falls through to candle.minMemoryGb",
      });
    }
    if (
      route.modality === "video" &&
      !route.videoFitSymbol &&
      route.sharedSelector.via !== "structural_floor_compatibility"
    ) {
      divergences.push({
        kind: "video_route_without_fit_gate",
        modelId: route.modelId,
        detail:
          "the routing catalog serves a candle video lane for this id, but candle_video_engine_id " +
          "produces no engine for it and no flat fit error is bound to it, so none of the " +
          "enumerated admission gates covers this route",
      });
    }
  }

  for (const provider of overlayProviders) {
    if (!provider.sharedSelector.reached) {
      divergences.push({
        kind: "overlay_provider_unreached",
        modelId: `provider:${provider.providerId}`,
        detail:
          "this candle provider has a load-shape rule but no named request scope and no base model " +
          "declaring it as a runtimeProvider, so nothing routes it into the shared selector",
      });
    }
  }

  for (const lane of unboundLanes) {
    divergences.push({
      kind: "image_lane_without_admission_binding",
      modelId: `lane:${lane}`,
      detail:
        "this conditioned candle image lane has no handler module bound in IMAGE_LANE_BINDINGS, so " +
        "its overlay-specific admission (if any) is not enumerated â€” only the engine-keyed and " +
        "manifest-keyed mechanisms are reported for the models it serves",
    });
  }

  // The manifest's advisory candle blocks that the router does NOT serve a candle lane for. Not a
  // route, but it is exactly the class of stale declaration a migration trips over.
  const advisoryOnly = [...entries.values()]
    .filter((model) => model.candle && !lanes.get(model.id)?.has("candle"))
    .map((model) => model.id)
    .sort();

  const byMechanism = Object.fromEntries(
    ADMISSION_MECHANISMS.map((mechanism) => [
      mechanism.id,
      routes.filter((route) => route.mechanisms.includes(mechanism.id)).length,
    ]),
  );

  return {
    schemaVersion: 1,
    generatedBy: "scripts/generate-candle-admission-inventory.mjs",
    story: "sc-19049",
    generatedFrom: { sceneWorksRevision: revision, sources: perSource },
    constants: {
      headroomGb,
      requestScopeConstants: Object.fromEntries([...dispatch.constants.entries()].sort()),
      catchAllRequestScope: dispatch.catchAll,
      overrideOnlyRequestScopes: dispatch.overrideOnly,
    },
    mechanisms: mechanismFacts,
    // `gate label -> geometry fact`, keyed by the labels the Rust emitter stamps on decision rows.
    // Published so the baseline can bind each row's `geometrySensitive` to the gate that row's route
    // actually reaches, instead of to whether ANY symbol of ANY mechanism it touches is geometry-aware.
    gateGeometry: [...deriveGateGeometry(mechanismFacts).values()].sort((a, b) =>
      a.gate.localeCompare(b.gate),
    ),
    videoBindings,
    imageLaneBindings: laneBindings,
    routes,
    overlayProviders,
    summary: {
      candleRoutes: routes.length,
      imageRoutes: routes.filter((route) => route.modality === "image").length,
      videoRoutes: routes.filter((route) => route.modality === "video").length,
      byMechanism,
      sharedSelector: {
        namedRevision: routes.filter((route) => route.sharedSelector.via === "named_revision").length,
        declarationCatchAll: routes.filter(
          (route) => route.sharedSelector.via === "declaration_catch_all",
        ).length,
        bespokeOverride: routes.filter((route) => route.sharedSelector.via === "bespoke_override")
          .length,
        legacyScalarCompatibility: routes.filter(
          (route) => route.sharedSelector.via === "legacy_scalar_compatibility",
        ).length,
        unverifiedCompatibility: routes.filter(
          (route) => route.sharedSelector.via === "unverified_compatibility",
        ).length,
        structuralFloorCompatibility: routes.filter(
          (route) => route.sharedSelector.via === "structural_floor_compatibility",
        ).length,
        legacyVideoCompatibility: routes.filter(
          (route) => route.sharedSelector.via === "legacy_video_compatibility",
        ).length,
        unreached: routes.filter((route) => !route.sharedSelector.reached).length,
      },
      geometryAwareMechanisms: mechanismFacts
        .filter((mechanism) => mechanism.geometryAware)
        .map((mechanism) => mechanism.id),
      packagedVideoCurveLanes: Object.fromEntries([...curveLanes.entries()].sort()),
      advisoryCandleBlocksWithoutRoute: advisoryOnly,
      imageLanesWithoutAdmissionBinding: unboundLanes,
      overlayProviders: overlayProviders.length,
    },
    divergences: divergences.sort(
      (a, b) => a.kind.localeCompare(b.kind) || a.modelId.localeCompare(b.modelId),
    ),
    knownGaps: knownGaps({ curveLanes, routes }),
  };
}

/**
 * Structural gaps this inventory records but does NOT close, each with the story that owns it.
 * Recorded in the artifact rather than in prose so a later slice can assert on them mechanically.
 */
function knownGaps({ curveLanes }) {
  return [
    {
      id: "measured-flag-is-a-bare-boolean",
      owner: "sc-19053",
      severity: "medium",
      detail:
        "The manifest carries `measured: true|false` at <model>.candle, .candle.turboFit and " +
        ".candle.control. There is no evidenceClass/evidence_class field anywhere in the tree, so " +
        "epic 19048 R3's 'adopt epic 18472's evidence-class flag' has no producer to adopt yet; a " +
        "boolean cannot distinguish a fitted curve from a single measured point from a declared floor.",
      paths: ["config/manifests/builtin.models.jsonc"],
    },
    {
      id: "zero-candle-video-memory-curves-packaged",
      owner: "sc-19057",
      severity: "high",
      detail:
        "sceneworks-core's lane-tagged VideoMemoryCurveBundle exists and fails closed on a foreign " +
        `lane, but the packaged curve data covers ${JSON.stringify(
          Object.fromEntries([...curveLanes.entries()].sort()),
        )} â€” there are no candle curves at all. Optimized Candle candidates are therefore ` +
        "Unverified; the seven legacy ceilings and Bernini's T2V structural pre-load floor remain " +
        "the only source-backed resident admission inputs.",
      paths: ["docs/generated/video-memory-curves.json"],
    },
    {
      id: "candle-sequential-capability-needs-the-linked-candle-bundle",
      owner: "sc-19050",
      severity: "high",
      detail:
        "The candle txt2img gate's `sequential_capable` input is " +
        "`mlx_fit_gate::engine_supports_sequential(engine_id)` â€” a query against the LINKED " +
        "provider bundle's descriptor (sc-12130 forbids a second engine-id allowlist in the " +
        "worker). The manifest's advisory `candle.supportsSequentialOffload` key is NOT that input " +
        "and diverges from it in BOTH directions on the bundles that can be linked here (the MLX " +
        "bundle answers true for sdxl/lens/krea_2_raw/krea_2_turbo/qwen_image_edit and friends " +
        "where the manifest says false, and false for the six mage_flow ids where the manifest " +
        "says true). The candle bundle needs a CUDA toolchain to link, so no CPU lane can resolve " +
        "it; every evaluated baseline row therefore records the plan for BOTH values of the input " +
        "rather than committing another bundle's answer. CORRECTED by sc-19050: the extracted " +
        "mechanism does NOT consume this input and cannot resolve it. Staged residency reaches " +
        "the shared mechanism as a CONTRACT-DECLARED rung, not as a gate-level boolean, so the " +
        "capability stays an input of `candle_scalar_gate::load_plan` alone. It becomes a " +
        "first-class input only when that route stops using the scalar gate (sc-19053/sc-19054), " +
        "and OBSERVING it still requires a CUDA-linked bundle either way — a toolchain blocker, " +
        "not a code one.",
      paths: [
        "crates/sceneworks-worker/src/image_jobs/base.rs",
        "crates/sceneworks-worker/src/mlx_fit_gate.rs",
      ],
    },
  ];
}

/**
 * Build the decision baseline from the RUST-EMITTED decisions.
 *
 * ## The candle rows are not computed here
 *
 * They are `docs/generated/candle-admission-decisions.json`, produced by
 * `crates/sceneworks-worker/src/candle_admission_decisions.rs` driving
 * `crates/sceneworks-worker/src/candle_scalar_gate.rs` â€” the functions the request actually reaches.
 * This function joins each Rust row to the inventory context for the same route (mechanisms, shared
 * selector verdict, whether the tier is advertised) and stamps the backend. The JS mirror further up
 * this file is retained as a readable statement of the law, and
 * `generate-candle-admission-inventory.test.mjs` reconciles it row-for-row against these decisions,
 * so it cannot drift silently.
 *
 * ## Three resolutions, stated per row rather than blurred
 *
 * * **`evaluated`** â€” the row is the output of the gate that route reaches. Today that is the
 *   legacy scalar gate, on the image routes whose base txt2img request lands there.
 * * **`not_evaluated`** â€” the route reaches a DIFFERENT gate, and that gate needs an input no CPU
 *   lane can produce (weight bytes measured off the resolved model dir, a runtime artifact probe, a
 *   registered `MemoryProviderContract`). The row names the gate and the missing input, and carries
 *   NO geometry or budget axis, because a decision that was never taken must not publish a
 *   coordinate response. Stamping these with the scalar gate's answer â€” which is what this file did
 *   before â€” records decisions their gates never produced.
 * * **`declared`** (MLX) â€” the manifest floor and evidence-ladder reachability, without re-deriving
 *   `mlx_fit_gate`'s estimate synthesis. Mirroring that here would fork the very prediction law
 *   sc-19050 is about to extract into shared code; the fork, not the absence, is what would make a
 *   later diff meaningless. Once sc-19050 lands these gain `evaluated` through the extracted module,
 *   and THAT slice's empty-diff obligation is against these rows.
 *
 * ## `sequentialCapable` is enumerated, not resolved
 *
 * Production takes it from `mlx_fit_gate::engine_supports_sequential(engine_id)` â€” a query against
 * the LINKED provider bundle's descriptor (sc-12130 explicitly forbids a second engine-id allowlist
 * in the worker). The manifest's advisory `candle.supportsSequentialOffload` key is NOT that input,
 * and reading it here produced hundreds of rows whose `loadPlan` flipped `reject`â†”`sequential` on
 * tight budgets. The candle bundle can only be linked with a CUDA toolchain, so every evaluated row
 * instead records the plan for BOTH values of the input; nothing is invented, and the day the
 * capability resolves each row narrows to one branch â€” a visible, enumerable diff.
 */
export function buildBaseline(bodies, inventory, decisions) {
  const entries = manifestEntries(bodies.manifest);
  const lanes = routedLanes({
    routingCatalog: bodies.routingCatalog,
    routingCandle: bodies.routingCandle,
    routingMlx: bodies.routingMlx,
  });
  const { revision } = sourceRevisionOf(bodies);

  const routesById = new Map(inventory.routes.map((route) => [route.modelId, route]));
  const gateGeometry = new Map(inventory.gateGeometry.map((gate) => [gate.gate, gate]));

  // Every gate the emitter stamped must be a gate this generator independently resolved, and the two
  // must AGREE about geometry-awareness. This is what stops `gate_is_geometry_aware`'s Rust match
  // from becoming a second hand-written claim.
  for (const row of decisions.rows) {
    const gate = gateGeometry.get(row.gate);
    if (!gate) {
      throw new Error(
        `the decision emitter stamped gate "${row.gate}" (route ${row.modelId}), which this ` +
          "generator resolved no geometry fact for. Add its defining symbol to ADMISSION_MECHANISMS.",
      );
    }
    if (gate.geometryAware !== row.geometrySensitive) {
      throw new Error(
        `gate "${row.gate}": the Rust emitter records geometrySensitive=${row.geometrySensitive} ` +
          `but its signature resolves to geometryAware=${gate.geometryAware} ` +
          `(axes ${JSON.stringify(gate.geometryAxes)}, via ${JSON.stringify(gate.reachedVia)}). ` +
          "One of the two is wrong; they are not allowed to disagree.",
      );
    }
  }

  const candleRows = decisions.rows.map((row) => {
    const route = routesById.get(row.modelId);
    if (!route) {
      throw new Error(
        `the decisions carry a row for "${row.modelId}", which is not an inventory route`,
      );
    }
    const shared = {
      backend: "candle",
      resolution: row.resolution,
      modelId: row.modelId,
      engineId: row.engineId ?? null,
      modality: row.modality,
      tier: row.tier,
      tierAdvertised: route.evidence.measuredTiers.includes(row.tier),
      gate: row.gate,
      mechanisms: route.mechanisms,
      sharedSelectorVia: route.sharedSelector.via,
      geometrySensitive: row.geometrySensitive,
      geometryAxes: gateGeometry.get(row.gate).geometryAxes,
    };
    if (row.resolution === "not_evaluated") {
      return { ...shared, missingInput: row.missingInput };
    }
    return {
      ...shared,
      width: row.width,
      height: row.height,
      frames: row.frames,
      budgetGb: row.budgetGb,
      // sc-19054: what the scalar may claim at this cell — `measured_peak` (covering capture,
      // compared raw) or `declared_floor` (estimate-margin-widened).
      scalarClass: row.scalarClass,
      predictedPeakGb: row.predictedPeakGb,
      predictedSequentialPeakGb: row.predictedSequentialPeakGb,
      // Both branches of the registry-derived input. See the header note.
      fitDecisionSequentialCapable: row.fitDecisionSequentialCapable,
      fitDecisionSequentialIncapable: row.fitDecisionSequentialIncapable,
      loadPlanSequentialCapable: row.loadPlanSequentialCapable,
      loadPlanSequentialIncapable: row.loadPlanSequentialIncapable,
    };
  });

  const mlxIds = [...lanes.entries()]
    .filter(([, laneSet]) => laneSet.has("mlx"))
    .map(([id]) => id)
    .sort();
  const mlxRows = [];
  for (const modelId of mlxIds) {
    const entry = entries.get(modelId) ?? null;
    const mlx = entry?.mlx ?? null;
    for (const tier of TIER_ORDER) {
      const measured = mlx?.vramGbByTier?.[tier];
      mlxRows.push({
        backend: "mlx",
        resolution: "declared",
        modelId,
        tier,
        tierAdvertised: typeof measured === "number",
        declaredPeakGb: round(typeof measured === "number" ? measured : null),
        declaredFloorGb: round(typeof mlx?.minMemoryGb === "number" ? mlx.minMemoryGb : null),
        reachesEvidenceLadder: Boolean(mlx?.memoryStrategyContract),
        supportsSequentialOffload: mlx?.supportsSequentialOffload === true,
      });
    }
  }

  const evaluated = candleRows.filter((row) => row.resolution === "evaluated");
  const notEvaluated = candleRows.filter((row) => row.resolution === "not_evaluated");

  return {
    schemaVersion: 2,
    generatedBy: "scripts/generate-candle-admission-inventory.mjs",
    story: "sc-19049",
    generatedFrom: {
      sceneWorksRevision: revision,
      // The candle decisions are NOT derived here; this identifies the artifact they came from.
      decisions: {
        path: DECISIONS_PATH,
        digest: decisions.digest,
        producedBy: decisions.producedBy,
        generatedBy: decisions.generatedBy,
      },
    },
    corpus: {
      tiers: [...TIER_ORDER],
      budgetsGb: [...BASELINE_BUDGETS_GB],
      imageGeometries: BASELINE_IMAGE_GEOMETRIES.map((geometry) => ({ ...geometry })),
      resolutions: {
        evaluated:
          "the output of the gate this route reaches, emitted by " +
          "crates/sceneworks-worker/src/candle_admission_decisions.rs driving candle_scalar_gate.rs",
        not_evaluated:
          "the route reaches a different gate whose inputs no CPU lane can produce; the row names " +
          "the gate and the missing input, and publishes no geometry or budget axis",
        declared:
          "MLX â€” manifest floors and evidence-ladder reachability. sc-19050 verified these CANNOT " +
          "become `evaluated` here: the synthesis floor is a pure function of " +
          "`MemoryProviderContract::asset_facts`, and every weights-free contract builder in the " +
          "pinned inference bundle injects ZERO asset facts by contract, while production fills " +
          "them from real stats. No committed table backfills them. sc-19050 instead emits " +
          "`docs/generated/mlx-estimate-synthesis-decisions.json` from the real mechanism over the " +
          "shipped MLX contract universe, ENUMERATING the asset facts it cannot resolve — the same " +
          "posture this baseline takes for `sequentialCapable`.",
      },
      sequentialCapability: decisions.sequentialCapability,
    },
    summary: {
      candleRows: candleRows.length,
      evaluatedCandleRows: evaluated.length,
      notEvaluatedCandleRows: notEvaluated.length,
      mlxRows: mlxRows.length,
      candlePlansSequentialCapable: tally(evaluated, (row) => row.loadPlanSequentialCapable),
      candlePlansSequentialIncapable: tally(evaluated, (row) => row.loadPlanSequentialIncapable),
      candleDecisionsSequentialCapable: tally(
        evaluated,
        (row) => row.fitDecisionSequentialCapable,
      ),
      candleDecisionsSequentialIncapable: tally(
        evaluated,
        (row) => row.fitDecisionSequentialIncapable,
      ),
      geometrySensitiveCandleRows: candleRows.filter((row) => row.geometrySensitive).length,
      notEvaluatedByGate: tally(notEvaluated, (row) => row.gate),
    },
    candle: candleRows,
    mlx: mlxRows,
  };
}

function round(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return null;
  return Math.round(value * 1e6) / 1e6;
}

function tally(rows, key) {
  const counts = {};
  for (const row of rows) {
    const bucket = key(row);
    counts[bucket] = (counts[bucket] ?? 0) + 1;
  }
  return Object.fromEntries(Object.entries(counts).sort());
}

/** Escape a value for a Markdown table cell. A literal `|` would otherwise split the column â€” and
 *  one of the recorded gaps quotes `measured: true|false`, which is exactly that case. */
function cell(text) {
  return String(text).replaceAll("|", "\\|");
}

export function renderMarkdown(inventory) {
  const lines = [];
  lines.push("# Candle admission route inventory");
  lines.push("");
  lines.push(
    "Generated by `scripts/generate-candle-admission-inventory.mjs` (sc-19049). Do not edit by hand â€”",
    "run `npm run generate:candle-admission` and commit the result.",
  );
  lines.push("");
  lines.push(`- SceneWorks revision: \`${inventory.generatedFrom.sceneWorksRevision}\``);
  lines.push(
    `- Candle routes: ${inventory.summary.candleRoutes} (${inventory.summary.imageRoutes} image, ${inventory.summary.videoRoutes} video)`,
  );
  lines.push(`- Headroom: \`${inventory.constants.headroomGb}\` GB`);
  lines.push("");

  lines.push("## Shared-selector reachability");
  lines.push("");
  lines.push(
    "Three distinct states, deliberately not conflated. Since sc-18456 the request-scope match ends",
    "in a catch-all, so \"reaches the shared selector\" no longer implies \"has an engine-keyed",
    "evidence revision\".",
  );
  lines.push("");
  lines.push("| state | routes |");
  lines.push("| --- | ---: |");
  for (const [state, count] of Object.entries(inventory.summary.sharedSelector)) {
    lines.push(`| ${state} | ${count} |`);
  }
  lines.push("");

  lines.push("## Mechanisms");
  lines.push("");
  lines.push(
    "`geometry-aware` here means *at least one of this mechanism's symbols is*. It is a summary, not",
    "a per-route claim â€” see the per-gate table below, which is what the baseline rows bind to.",
  );
  lines.push("");
  lines.push("| mechanism | any symbol geometry-aware | axes | routes | defined in |");
  lines.push("| --- | :---: | --- | ---: | --- |");
  for (const mechanism of inventory.mechanisms) {
    lines.push(
      `| \`${mechanism.id}\` | ${mechanism.geometryAware ? "yes" : "no"} | ${
        mechanism.geometryAxes.length ? mechanism.geometryAxes.join(", ") : "â€”"
      } | ${inventory.summary.byMechanism[mechanism.id]} | ${
        mechanism.definedIn ? `\`${mechanism.definedIn}\`` : "â€”"
      } |`,
    );
  }
  lines.push("");

  lines.push("## Geometry-awareness, per gate function");
  lines.push("");
  lines.push(
    "Resolved from each gate's own parameter list, INCLUDING geometry that arrives inside a struct",
    "parameter (`geometry: MemoryGeometry`, `query: VideoCurveQuery`) â€” a scan of the parameter",
    "tokens sees none of those. Unioning these per mechanism is what previously labelled the",
    "wan/scail2/LTX video routes geometry-aware on the strength of `svd_fit_error`'s signature.",
    "sc-19055 migrated the wan/scail2 gates onto `geometry: MemoryGeometry`; LTX's",
    "`video_weights_fit_error` now also carries resolved geometry as selector identity while",
    "preserving its ungraded flat ceiling. VACE-Fun uses the separate resolution-blind",
    "`unscoped_video_weights_fit_error`, so a mechanism union would still misreport it.",
  );
  lines.push("");
  lines.push("| gate | mechanism | geometry-aware | axes | reached via |");
  lines.push("| --- | --- | :---: | --- | --- |");
  for (const gate of inventory.gateGeometry) {
    lines.push(
      `| \`${gate.gate}\` | \`${gate.mechanism}\` | ${gate.geometryAware ? "yes" : "no"} | ${
        gate.geometryAxes.length ? gate.geometryAxes.join(", ") : "â€”"
      } | ${gate.reachedVia.length ? gate.reachedVia.map((via) => `\`${via}\``).join("<br>") : "â€”"} |`,
    );
  }
  lines.push("");

  lines.push("## Routes");
  lines.push("");
  lines.push("| model | engine | modality | shared selector | mechanisms | measured tiers | floor (GB) |");
  lines.push("| --- | --- | --- | --- | --- | --- | ---: |");
  for (const route of inventory.routes) {
    lines.push(
      `| \`${route.modelId}\` | ${route.engineId ? `\`${route.engineId}\`` : "â€”"} | ${route.modality} | ${
        route.sharedSelector.via ?? "**unreached**"
      } | ${route.mechanisms.map((id) => `\`${id}\``).join("<br>")} | ${
        route.evidence.measuredTiers.length ? route.evidence.measuredTiers.join(", ") : "â€”"
      } | ${route.evidence.minMemoryGb ?? "â€”"} |`,
    );
  }
  lines.push("");

  lines.push("## Overlay providers");
  lines.push("");
  lines.push(
    "Admission coordinates that are not a catalog model id: a control or identity provider reached",
    "as an overlay on a base model's route, declared in the manifest as a `runtimeProvider`.",
  );
  lines.push("");
  lines.push("| provider | load-shape rule | shared selector | base models |");
  lines.push("| --- | :---: | --- | --- |");
  for (const provider of inventory.overlayProviders) {
    lines.push(
      `| \`${provider.providerId}\` | ${provider.loadShapeRule ? "yes" : "no"} | ${
        provider.sharedSelector.via ?? "**unreached**"
      } | ${provider.baseModels.map((id) => `\`${id}\``).join(", ") || "â€”"} |`,
    );
  }
  lines.push("");

  lines.push("## Divergences");
  lines.push("");
  lines.push(
    "The ledger epic 19048's acceptance story (sc-19059) must drive to empty. Each row is a route",
    "that is not yet on the converged mechanism.",
  );
  lines.push("");
  lines.push("| kind | model | detail |");
  lines.push("| --- | --- | --- |");
  for (const divergence of inventory.divergences) {
    lines.push(`| \`${divergence.kind}\` | \`${divergence.modelId}\` | ${cell(divergence.detail)} |`);
  }
  lines.push("");

  lines.push("## Known gaps");
  lines.push("");
  lines.push("| id | owner | severity | detail |");
  lines.push("| --- | --- | --- | --- |");
  for (const gap of inventory.knownGaps) {
    lines.push(`| \`${gap.id}\` | ${gap.owner} | ${gap.severity} | ${cell(gap.detail)} |`);
  }
  lines.push("");
  return `${lines.join("\n")}\n`;
}

/** Render all three artifact bodies from a set of source bodies plus the Rust-emitted decisions. */
export function renderArtifacts(bodies, decisions) {
  const inventory = buildInventory(bodies);
  const baseline = buildBaseline(bodies, inventory, decisions);
  return {
    inventory,
    baseline,
    inventoryJson: `${JSON.stringify(inventory, null, 2)}\n`,
    inventoryMarkdown: renderMarkdown(inventory),
    baselineJson: `${JSON.stringify(baseline, null, 2)}\n`,
  };
}

/**
 * Prove the guards fail when violated. A checker nobody has watched fail is a checker nobody knows
 * works â€” the same discipline `check-tier-integrity.mjs --self-test` applies.
 */
export async function selfTest() {
  const { bodies } = await readSources();
  const decisions = await readDecisions();
  const failures = [];
  const expectFailure = (label, mutate, mutateDecisions) => {
    let mutated;
    let mutatedDecisions = decisions;
    try {
      mutated = { ...bodies, ...mutate() };
      if (mutateDecisions) mutatedDecisions = mutateDecisions();
    } catch (error) {
      failures.push(`${label}: could not build the mutation (${error.message})`);
      return;
    }
    try {
      renderArtifacts(mutated, mutatedDecisions);
      failures.push(`${label}: the generator ACCEPTED a source it must reject`);
    } catch {
      /* expected */
    }
  };

  expectFailure("request-scope dispatch without a catch-all arm", () => ({
    candleMemoryStrategy: bodies.candleMemoryStrategy.replace(
      /_ => DECLARATION_REQUEST_EVIDENCE_REVISION,/,
      "",
    ),
  }));
  expectFailure("a video fit symbol renamed out of vram_gate.rs", () => ({
    vramGate: bodies.vramGate.replaceAll("fn svd_fit_error", "fn svd_fit_error_renamed"),
  }));
  expectFailure("a video fit call site deleted from video_jobs/candle.rs", () => ({
    videoRouteCandle: bodies.videoRouteCandle.replaceAll(
      "vram_gate::unscoped_video_weights_fit_error(",
      "vram_gate::unscoped_video_weights_fit_error_removed_call(",
    ),
  }));
  expectFailure("a new candle video engine with no admission binding", () => ({
    videoRouteCandle: bodies.videoRouteCandle.replace(
      '"mochi_1" => Some("mochi_1"),',
      '"mochi_1" => Some("mochi_1"),\n        "unbound_model" => Some("unbound_engine"),',
    ),
  }));
  expectFailure("the headroom constant removed from fit_gate.rs", () => ({
    fitGate: bodies.fitGate.replace(
      /const DEDICATED_VRAM_ALLOCATOR_SLACK_GB:\s*f64\s*=\s*[0-9.]+\s*;/,
      "",
    ),
  }));
  expectFailure("LTX compatibility evidence stamped with a placeholder tier", () => ({
    videoRouteCandle: bodies.videoRouteCandle.replace(
      'ltx_tier_key.expect("an LTX model path must resolve an admission tier")',
      '"bf16"',
    ),
  }));
  expectFailure("LTX compatibility evidence stamped with a placeholder mode", () => ({
    videoRouteCandle: bodies.videoRouteCandle.replace(
      "request.mode.as_str(),\n            crate::video_admission::video_gate_geometry(request.width, request.height, frames)",
      '"text_to_video",\n            crate::video_admission::video_gate_geometry(request.width, request.height, frames)',
    ),
  }));
  expectFailure("LTX compatibility evidence stamped with placeholder geometry", () => ({
    videoRouteCandle: bodies.videoRouteCandle.replace(
      "request.mode.as_str(),\n            crate::video_admission::video_gate_geometry(request.width, request.height, frames)",
      "request.mode.as_str(),\n            crate::video_admission::video_gate_geometry(1, 1, 1)",
    ),
  }));

  // The guards the sc-19049 review asked for. The Rust emitter's `gate_is_geometry_aware` match may
  // not stand as a second hand-written claim beside this file's derivation, and a decision row may
  // not name a route or a gate nobody resolved.
  expectFailure(
    "a decision row whose geometrySensitive contradicts its gate's signature",
    () => ({}),
    () => ({
      ...decisions,
      rows: decisions.rows.map((row, index) =>
        index === 0 ? { ...row, geometrySensitive: !row.geometrySensitive } : row,
      ),
    }),
  );
  expectFailure(
    "a decision row for a model that is not an inventory route",
    () => ({}),
    () => ({
      ...decisions,
      rows: [...decisions.rows, { ...decisions.rows[0], modelId: "not_a_route" }],
    }),
  );
  expectFailure(
    "a decision row stamped with an unresolvable gate label",
    () => ({}),
    () => ({
      ...decisions,
      rows: decisions.rows.map((row, index) =>
        index === 0 ? { ...row, gate: "some_gate_nobody_declared" } : row,
      ),
    }),
  );
  // Geometry hidden inside a struct parameter must not read as "geometry-blind": when the struct's
  // fields become unresolvable the generator has to REFUSE, not silently downgrade the gate. This is
  // the mutation that reproduces the original defect.
  //
  // The mutation removes every struct-LITERAL construction of `gen_core::MemoryGeometry` from the
  // fingerprinted sources, leaving the `geometry: MemoryGeometry` parameters in place. That is
  // exactly the state the type is in when nothing in this tree builds it: its fields become
  // unrecoverable, and a generator that shrugged would report `fit_ladder_for_entry_with_runtime`
  // and `evaluate_shared_image` as geometry-blind — the inverse of the truth.
  expectFailure("a geometry struct parameter whose fields cannot be resolved", () => {
    const dropLiterals = (body) => body.replaceAll("MemoryGeometry {", "UnbuiltGeometryLiteral {");
    return Object.fromEntries(
      Object.keys(SOURCE_PATHS)
        .filter((name) => SOURCE_PATHS[name].endsWith(".rs"))
        .map((name) => [name, dropLiterals(bodies[name])]),
    );
  });

  // A positive control: the unmutated tree must build.
  try {
    renderArtifacts(bodies, decisions);
  } catch (error) {
    failures.push(`the unmutated tree does not build: ${error.message}`);
  }

  if (failures.length > 0) {
    throw new Error(`self-test failed:\n  - ${failures.join("\n  - ")}`);
  }
  return { checks: 10 };
}

async function main() {
  if (process.argv.includes("--help") || process.argv.includes("-h")) {
    console.log(
      "Generate the candle admission route inventory and decision baseline (sc-19049).\n\n" +
        "  node scripts/generate-candle-admission-inventory.mjs             regenerate\n" +
        "  node scripts/generate-candle-admission-inventory.mjs --check     fail on drift (CI)\n" +
        "  node scripts/generate-candle-admission-inventory.mjs --self-test prove the guards fail\n",
    );
    return;
  }

  if (process.argv.includes("--self-test")) {
    const { checks } = await selfTest();
    console.log(`candle-admission self-test: ${checks} guards verified.`);
    return;
  }

  const { bodies } = await readSources();
  const decisions = await readDecisions();
  const rendered = renderArtifacts(bodies, decisions);

  if (process.argv.includes("--check")) {
    const committed = await Promise.all(
      [OUTPUT_INVENTORY_JSON, OUTPUT_INVENTORY_MD, OUTPUT_BASELINE_JSON].map(async (relative) =>
        canonicalSourceText(await readFile(path.join(ROOT, relative), "utf8")),
      ),
    );
    const expected = [rendered.inventoryJson, rendered.inventoryMarkdown, rendered.baselineJson];
    for (const [index, relative] of [
      OUTPUT_INVENTORY_JSON,
      OUTPUT_INVENTORY_MD,
      OUTPUT_BASELINE_JSON,
    ].entries()) {
      if (committed[index] !== expected[index]) {
        throw new Error(
          `${relative} is stale; run npm run generate:candle-admission and commit the result`,
        );
      }
    }
    return;
  }

  await mkdir(path.join(ROOT, "docs/generated"), { recursive: true });
  await writeFile(path.join(ROOT, OUTPUT_INVENTORY_JSON), rendered.inventoryJson, "utf8");
  await writeFile(path.join(ROOT, OUTPUT_INVENTORY_MD), rendered.inventoryMarkdown, "utf8");
  await writeFile(path.join(ROOT, OUTPUT_BASELINE_JSON), rendered.baselineJson, "utf8");
  console.log(
    `candle-admission: ${rendered.inventory.summary.candleRoutes} routes, ` +
      `${rendered.baseline.summary.evaluatedCandleRows} evaluated candle decisions + ` +
      `${rendered.baseline.summary.notEvaluatedCandleRows} recorded non-evaluated rows, ` +
      `${rendered.inventory.divergences.length} divergences.`,
  );
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
