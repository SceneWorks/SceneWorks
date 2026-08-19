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
 *     not enumerate and justify (epic requirement R6). Nothing here needs a GPU: the candle gate is
 *     a pure function of the manifest entry, the tier key and a synthetic `VramBudget`, exactly as
 *     `vram_gate.rs`'s own unit tests drive it.
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
 * ## KNOWN GAP, deliberately NOT fixed here: the memory-matrix fingerprint under-covers this surface
 *
 * `scripts/generate-memory-matrix.mjs`'s own `SOURCE_PATHS` does **not** include
 * `candle_memory_strategy.rs`, `video_admission.rs`, `conditioning_fit.rs`, `krea_control_fit.rs`
 * or `sceneworks-core/src/video_memory_curves.rs` â€” five files that decide candle admission. A
 * change to any of them leaves `docs/generated/memory-matrix.json`'s revision unrotated, so the
 * matrix can claim currency it does not have on exactly the surface this epic rewrites.
 *
 * It is recorded here (`knownGaps[]`, and the "Known gaps" section of the generated Markdown)
 * rather than fixed, because adding entries to the matrix's `SOURCE_PATHS` rotates ITS fingerprint
 * and forces a full memory-matrix regeneration into every subsequent story's PR in this epic. The
 * epic's terminal acceptance story (sc-19059) owns closing it, once, at the end.
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
 *   - **Is the gate geometry-aware?** From the gate function's own signature in `vram_gate.rs` â€”
 *     whether it takes `width`/`height`/`frames`. Not asserted in prose here; parsed.
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

  // The admission surface itself.
  fitGate: "crates/sceneworks-worker/src/fit_gate.rs",
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
  imagePulid: "crates/sceneworks-worker/src/image_jobs/pulid_candle.rs",
  imageDetail: "crates/sceneworks-worker/src/image_jobs/detail.rs",
  videoRouteCandle: "crates/sceneworks-worker/src/video_jobs/candle.rs",
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
    source: "vramGate",
    definitionSymbols: Object.freeze([
      "predicted_peak_gb",
      "predicted_peak_gb_with_adapter_bytes",
      "predicted_sequential_peak_gb",
      "predicted_sequential_peak_gb_with_adapter_bytes",
      "sequential_overflow_gb",
      "load_plan",
    ]),
    callPattern:
      "(?<!fn\\s)\\b(?:predicted_peak_gb|predicted_peak_gb_with_adapter_bytes|predicted_sequential_peak_gb|predicted_sequential_peak_gb_with_adapter_bytes|sequential_overflow_gb|load_plan)\\s*\\(",
    summary:
      "Per-tier manifest scalar (`candle.vramGbByTier` + headroom, else `candle.minMemoryGb`) compared to a live VRAM budget.",
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
    ]),
    callPattern:
      "(?<!fn\\s)\\b(?:svd_fit_error|mochi_fit_error|wan_video_fit_error(?:_with_adapter_bytes)?|scail2_video_fit_error(?:_with_adapter_bytes)?|video_weights_fit_error)\\s*\\(",
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
    symbol: "video_weights_fit_error",
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

/** Image geometry corpus. Square rungs spanning the ladder the calibration plan uses. */
export const BASELINE_IMAGE_GEOMETRIES = Object.freeze([
  Object.freeze({ width: 1024, height: 1024, frames: 1 }),
  Object.freeze({ width: 1536, height: 1536, frames: 1 }),
  Object.freeze({ width: 2048, height: 2048, frames: 1 }),
]);

/** Video geometry corpus: two clip lengths at two resolutions, so a frames term is visible. */
export const BASELINE_VIDEO_GEOMETRIES = Object.freeze([
  Object.freeze({ width: 768, height: 512, frames: 49 }),
  Object.freeze({ width: 1280, height: 720, frames: 81 }),
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

/**
 * Per-mechanism derived facts: the modules that call its entry symbols, and whether the gate is
 * geometry-aware â€” read off the entry symbol's own parameter list rather than declared here.
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
    const definitionSource = mechanism.source ? bodies[mechanism.source] : null;
    const geometryParameters = new Set();
    let definitionsSeen = 0;
    if (definitionSource) {
      for (const symbol of mechanism.definitionSymbols) {
        const definition = definitionSource.match(
          new RegExp(`fn\\s+${symbol}\\s*(?:<[^>]*>)?\\s*\\(([\\s\\S]*?)\\)\\s*->`),
        );
        if (!definition) continue;
        definitionsSeen += 1;
        for (const axis of ["width", "height", "frames"]) {
          if (new RegExp(`\\b${axis}\\s*:`).test(definition[1])) geometryParameters.add(axis);
        }
      }
      if (definitionsSeen === 0) {
        throw new Error(
          `mechanism "${mechanism.id}": none of its definition symbols exist in ` +
            `${SOURCE_PATHS[mechanism.source]}. The taxonomy has drifted from the tree.`,
        );
      }
    }
    facts.push({
      id: mechanism.id,
      summary: mechanism.summary,
      definedIn: mechanism.source ? SOURCE_PATHS[mechanism.source] : null,
      definitionSymbols: [...mechanism.definitionSymbols],
      geometryAware: geometryParameters.size > 0,
      geometryAxes: [...geometryParameters].sort(),
      calledFrom: [...callers].sort().map((name) => SOURCE_PATHS[name]),
    });
  }
  return facts;
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
// The candle legacy scalar gate, mirrored exactly. Every branch below is a line-for-line reading of
// `vram_gate.rs`; the source fingerprint above is what forces this artifact to be regenerated (and
// the mirror re-reviewed) the moment any of it changes.
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
}) {
  const mechanisms = [...laneMechanisms];
  const named = engineId ? dispatch.named.get(engineId) : undefined;
  let sharedSelector;
  if (laneMechanisms.includes("shared_selector_bespoke_override")) {
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
  const entries = manifestEntries(bodies.manifest);
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
    if (route.modality === "video" && !route.videoFitSymbol) {
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
function knownGaps({ curveLanes, routes }) {
  const missingFingerprint = [
    "crates/sceneworks-worker/src/candle_memory_strategy.rs",
    "crates/sceneworks-worker/src/video_admission.rs",
    "crates/sceneworks-worker/src/conditioning_fit.rs",
    "crates/sceneworks-worker/src/krea_control_fit.rs",
    "crates/sceneworks-core/src/video_memory_curves.rs",
  ];
  return [
    {
      id: "memory-matrix-source-paths-under-cover-candle-admission",
      owner: "sc-19059",
      severity: "high",
      detail:
        "scripts/generate-memory-matrix.mjs's SOURCE_PATHS omits five files that decide candle " +
        "admission, so a change to any of them leaves docs/generated/memory-matrix.json's " +
        "source-tree revision unrotated and the matrix claims a currency it does not have. Adding " +
        "them rotates the matrix fingerprint and forces a full regeneration, so epic 19048's " +
        "terminal acceptance story owns the fix rather than every slice paying for it.",
      paths: missingFingerprint,
    },
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
        )} â€” there are no candle curves at all, so every candle video route falls back to the ` +
        "weights+headroom lower bound.",
      paths: ["docs/generated/video-memory-curves.json"],
    },
    {
      id: "candle-image-admission-is-geometry-blind",
      owner: "sc-19054",
      severity: "high",
      detail:
        `${routes.filter((route) => route.modality === "image" && route.mechanisms.includes("legacy_scalar_gate")).length} ` +
        "image routes are admitted by the per-tier scalar gate, whose signature takes no width, " +
        "height or frames â€” a 1024x1024 and a 2048x2048 request are admitted identically. The " +
        "decision baseline records that explicitly (`geometrySensitive: false`) so the slice that " +
        "fixes it produces a visible, enumerable diff.",
      paths: ["crates/sceneworks-worker/src/vram_gate.rs"],
    },
  ];
}

/**
 * Build the decision baseline: admission decisions over model x tier x geometry x budget.
 *
 * ## Two resolutions, stated per row rather than blurred
 *
 * Candle rows are `evaluated`: every field is the output of the mirrored gate above, which is a
 * complete reading of the candle scalar path. MLX rows are `declared`: they record the floor the
 * manifest publishes and whether the model reaches the evidence ladder, but they do NOT re-derive
 * `mlx_fit_gate`'s estimate synthesis. Mirroring that here would fork the very prediction law
 * sc-19050 is about to extract into shared code â€” the fork, not the absence, is what would make a
 * later diff meaningless. Once sc-19050 lands, the MLX rows gain the same `evaluated` resolution
 * through the extracted module, and THAT slice's empty-diff obligation is against these rows.
 *
 * MLX rows carry no geometry or budget axis, because a declared floor has no geometry response to
 * record; emitting one would fabricate a column of constants.
 */
export function buildBaseline(bodies, inventory) {
  const entries = manifestEntries(bodies.manifest);
  const lanes = routedLanes({
    routingCatalog: bodies.routingCatalog,
    routingCandle: bodies.routingCandle,
    routingMlx: bodies.routingMlx,
  });
  const headroomGb = parseHeadroomGb(bodies.fitGate);
  const { revision } = sourceRevisionOf(bodies);

  const candleRows = [];
  for (const route of inventory.routes) {
    const entry = entries.get(route.modelId) ?? null;
    const geometries =
      route.modality === "video" ? BASELINE_VIDEO_GEOMETRIES : BASELINE_IMAGE_GEOMETRIES;
    const sequentialCapable = route.evidence.supportsSequentialOffload;
    for (const tier of TIER_ORDER) {
      const neededGb = predictedPeakGb(entry, tier, headroomGb);
      const sequentialGb = predictedSequentialPeakGb(entry, tier, headroomGb);
      for (const geometry of geometries) {
        for (const budgetGb of BASELINE_BUDGETS_GB) {
          const budget = { freeGb: budgetGb, totalGb: budgetGb };
          candleRows.push({
            backend: "candle",
            resolution: "evaluated",
            modelId: route.modelId,
            engineId: route.engineId,
            modality: route.modality,
            tier,
            tierAdvertised: route.evidence.measuredTiers.includes(tier),
            width: geometry.width,
            height: geometry.height,
            frames: geometry.frames,
            budgetGb,
            mechanisms: route.mechanisms,
            sharedSelectorVia: route.sharedSelector.via,
            predictedPeakGb: round(neededGb),
            predictedSequentialPeakGb: round(sequentialGb),
            sequentialCapable,
            fitDecision: resolveOffload(fitDecision(neededGb, budget), sequentialCapable),
            loadPlan: loadPlan(neededGb, sequentialGb, budget, sequentialCapable),
            // The whole point of the baseline: today this is false for every row whose mechanism
            // list does not contain a geometry-aware gate. sc-19054 flips it, visibly.
            geometrySensitive: route.mechanisms.some((id) =>
              inventory.mechanisms.find((mechanism) => mechanism.id === id)?.geometryAware,
            ),
          });
        }
      }
    }
  }

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

  return {
    schemaVersion: 1,
    generatedBy: "scripts/generate-candle-admission-inventory.mjs",
    story: "sc-19049",
    generatedFrom: { sceneWorksRevision: revision },
    corpus: {
      tiers: [...TIER_ORDER],
      budgetsGb: [...BASELINE_BUDGETS_GB],
      imageGeometries: BASELINE_IMAGE_GEOMETRIES.map((geometry) => ({ ...geometry })),
      videoGeometries: BASELINE_VIDEO_GEOMETRIES.map((geometry) => ({ ...geometry })),
      resolutions: {
        candle: "evaluated â€” every field is the mirrored vram_gate decision for this coordinate",
        mlx: "declared â€” manifest floors and evidence-ladder reachability; sc-19050 upgrades these to evaluated",
      },
    },
    summary: {
      candleRows: candleRows.length,
      mlxRows: mlxRows.length,
      candlePlans: tally(candleRows, (row) => row.loadPlan),
      candleDecisions: tally(candleRows, (row) => row.fitDecision),
      geometrySensitiveCandleRows: candleRows.filter((row) => row.geometrySensitive).length,
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
  lines.push("| mechanism | geometry-aware | axes | routes | defined in |");
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

/** Render all three artifact bodies from a set of source bodies. */
export function renderArtifacts(bodies) {
  const inventory = buildInventory(bodies);
  const baseline = buildBaseline(bodies, inventory);
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
  const failures = [];
  const expectFailure = (label, mutate) => {
    let mutated;
    try {
      mutated = { ...bodies, ...mutate() };
    } catch (error) {
      failures.push(`${label}: could not build the mutation (${error.message})`);
      return;
    }
    try {
      renderArtifacts(mutated);
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
      "vram_gate::scail2_video_fit_error_with_adapter_bytes(",
      "vram_gate::scail2_video_fit_error_removed_call(",
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

  // A positive control: the unmutated tree must build.
  try {
    renderArtifacts(bodies);
  } catch (error) {
    failures.push(`the unmutated tree does not build: ${error.message}`);
  }

  if (failures.length > 0) {
    throw new Error(`self-test failed:\n  - ${failures.join("\n  - ")}`);
  }
  return { checks: 6 };
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
  const rendered = renderArtifacts(bodies);

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
      `${rendered.baseline.summary.candleRows} evaluated candle decisions, ` +
      `${rendered.inventory.divergences.length} divergences.`,
  );
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
