// Tests for the candle admission route inventory + decision baseline (sc-19049, epic 19048).
//
// Two things are being pinned, and they are different in kind:
//
//   1. The artifacts are GENERATED and GUARDED — the fingerprint covers every source the generator
//      reads, the committed files match a fresh build, and each derivation guard actually fails
//      when the tree drifts under it. A hand-written inventory is stale on arrival; these tests are
//      what make "generated" mean something.
//   2. The decisions come from the RUST GATE, not from this generator. The candle rows are read out
//      of `docs/generated/candle-admission-decisions.json`, emitted by
//      `crates/sceneworks-worker/src/candle_admission_decisions.rs` driving `candle_scalar_gate.rs`.
//      The JS mirror that remains in the generator is reconciled against those rows coordinate by
//      coordinate, and separately pinned on the properties the gate documents invariants for — the
//      NVFP4 tier fallback, the declared-floor fallback, and `load_plan`'s budget monotonicity.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { stripJsoncComments } from "./lib/jsonc.mjs";
import { canonicalSourceText } from "./lib/source-revision.mjs";
import { TIER_ORDER } from "./lib/manifest-memory-declarations.mjs";
import {
  ADMISSION_MECHANISMS,
  BASELINE_BUDGETS_GB,
  BASELINE_IMAGE_GEOMETRIES,
  DECISIONS_PATH,
  IMAGE_LANE_BINDINGS,
  OUTPUT_PATHS,
  SOURCE_PATHS,
  VIDEO_ADMISSION_BINDINGS,
  buildBaseline,
  buildInventory,
  deriveLaneBindings,
  deriveMechanismFacts,
  fitDecision,
  loadPlan,
  parseCandleImageLanes,
  parseCandleVideoEngines,
  parseCatalogModalities,
  parseRequestScopeDispatch,
  predictedPeakGb,
  predictedPeakGbForRequest,
  predictedSequentialPeakGb,
  predictedSequentialPeakGbForRequest,
  readDecisions,
  readSources,
  renderArtifacts,
  resolveSymbolGeometry,
  scalarPeakClass,
  selfTest,
  sourceRevisionOf,
} from "./generate-candle-admission-inventory.mjs";

const sources = await readSources();
const bodies = sources.bodies;
const decisions = await readDecisions();
const artifacts = renderArtifacts(bodies, decisions);

async function committed(relative) {
  return canonicalSourceText(await readFile(new URL(`../${relative}`, import.meta.url), "utf8"));
}

// -------------------------------------------------------------------------------------------------
// Provenance
// -------------------------------------------------------------------------------------------------

test("the fingerprint covers exactly the sources the generator reads", () => {
  // The invariant, not a restatement of it: `readSources` reports the key set it actually touched,
  // and every parser in the generator is handed bodies from that map. A parser that starts reading
  // an unfingerprinted file has nowhere to get the body from.
  assert.deepEqual(sources.read.sort(), Object.keys(SOURCE_PATHS).sort());
  assert.deepEqual(
    Object.fromEntries(
      Object.entries(artifacts.inventory.generatedFrom.sources).map(([name, entry]) => [
        name,
        entry.path,
      ]),
    ),
    { ...SOURCE_PATHS },
  );
});

test("the source revision is a source-tree sha256 and is shared by both artifacts", () => {
  const revision = artifacts.inventory.generatedFrom.sceneWorksRevision;
  assert.match(revision, /^source-tree:[0-9a-f]{64}$/);
  assert.equal(artifacts.baseline.generatedFrom.sceneWorksRevision, revision);
});

test("a semantic change to any fingerprinted Rust source rotates the revision", () => {
  const baseline = sourceRevisionOf(bodies).revision;
  for (const [name, relative] of Object.entries(SOURCE_PATHS)) {
    if (!relative.endsWith(".rs")) continue;
    const mutated = sourceRevisionOf({
      ...bodies,
      [name]: `${bodies[name]}\nconst SC_19049_PROVENANCE_PROBE: &str = "probe";\n`,
    }).revision;
    assert.notEqual(mutated, baseline, `${relative} does not contribute to the fingerprint`);
  }
});

test("an inert comment-only edit does not rotate the revision", () => {
  const baseline = sourceRevisionOf(bodies).revision;
  const mutated = sourceRevisionOf({
    ...bodies,
    vramGate: `${bodies.vramGate}\n// sc-19049 provenance regression: an inert comment line\n`,
  }).revision;
  assert.equal(mutated, baseline);
});

test("the committed artifacts match a fresh build", async () => {
  assert.equal(await committed(OUTPUT_PATHS.inventoryJson), artifacts.inventoryJson);
  assert.equal(await committed(OUTPUT_PATHS.inventoryMarkdown), artifacts.inventoryMarkdown);
  assert.equal(await committed(OUTPUT_PATHS.baselineJson), artifacts.baselineJson);
});

test("a rebuild from the same sources is byte-identical", () => {
  const again = renderArtifacts(bodies, decisions);
  assert.equal(again.inventoryJson, artifacts.inventoryJson);
  assert.equal(again.baselineJson, artifacts.baselineJson);
  assert.equal(again.inventoryMarkdown, artifacts.inventoryMarkdown);
});

// -------------------------------------------------------------------------------------------------
// Route universe reconciliation
// -------------------------------------------------------------------------------------------------

test("the route universe reconciles against the routing catalog, not the manifest", () => {
  const modalities = parseCatalogModalities(bodies.routingCatalog);
  const rows = new Map(artifacts.inventory.routes.map((route) => [route.modelId, route]));
  assert.equal(rows.size, artifacts.inventory.routes.length, "duplicate route rows");
  for (const id of [...modalities.image, ...modalities.video]) {
    assert.ok(rows.has(id), `catalog routes ${id} on candle but the inventory has no row`);
  }
  for (const route of artifacts.inventory.routes) {
    assert.equal(
      route.modality,
      modalities.video.has(route.modelId) ? "video" : "image",
      `${route.modelId}: modality disagrees with the catalog`,
    );
  }
  // The manifest's advisory `candle` blocks are deliberately NOT the universe: nine entries carry
  // one without the router serving any candle lane for them.
  const manifest = JSON.parse(stripJsoncComments(bodies.manifest));
  for (const id of artifacts.inventory.summary.advisoryCandleBlocksWithoutRoute) {
    assert.ok(
      manifest.models.some((model) => model.id === id && model.candle),
      `${id} is reported as an advisory-only candle block but has no candle block`,
    );
    assert.ok(!rows.has(id), `${id} is reported as unrouted yet has a route row`);
  }
});

test("every route carries exactly one shared-selector verdict, and the three states are distinct", () => {
  const dispatch = parseRequestScopeDispatch(bodies.candleMemoryStrategy);
  for (const route of artifacts.inventory.routes) {
    const { reached, via, evidenceRevision } = route.sharedSelector;
    assert.equal(reached, via !== null, `${route.modelId}: reached/via disagree`);
    if (via === "named_revision") {
      assert.equal(evidenceRevision, dispatch.named.get(route.engineId).value);
    } else if (via === "declaration_catch_all") {
      assert.equal(evidenceRevision, dispatch.catchAll.value);
      assert.ok(
        route.evidence.declaresProviderContract,
        `${route.modelId}: catch-all scope without a candle.memoryStrategyContract`,
      );
    } else if (via === null) {
      assert.equal(evidenceRevision, null);
    }
  }
  const summary = artifacts.inventory.summary.sharedSelector;
  assert.equal(
    summary.namedRevision + summary.declarationCatchAll + summary.bespokeOverride + summary.unreached,
    artifacts.inventory.routes.length,
  );
});

test("the named request scopes are exactly the dispatch's named arms", () => {
  const dispatch = parseRequestScopeDispatch(bodies.candleMemoryStrategy);
  const namedEngines = new Set(
    artifacts.inventory.routes
      .filter((route) => route.sharedSelector.via === "named_revision")
      .map((route) => route.engineId),
  );
  for (const engineId of namedEngines) {
    assert.ok(dispatch.named.has(engineId), `${engineId} is not a named dispatch arm`);
  }
  // sc-18456's catch-all is present and reported separately from the named arms, which is the
  // distinction the story requires and the epic text (written before it landed) does not make.
  assert.ok(dispatch.catchAll.constant.length > 0);
  assert.ok(!dispatch.named.has("__catch_all__"));
});

test("every candle video engine is bound to a flat fit error that still exists and is still called", () => {
  const engines = parseCandleVideoEngines(bodies.videoRouteCandle);
  const bound = new Set(VIDEO_ADMISSION_BINDINGS.map((binding) => binding.engineId));
  for (const engineId of engines.values()) {
    assert.ok(bound.has(engineId), `${engineId} has no VIDEO_ADMISSION_BINDINGS entry`);
  }
  for (const binding of artifacts.inventory.videoBindings) {
    assert.match(bodies.vramGate, new RegExp(`fn\\s+${binding.symbol}\\s*\\(`));
    assert.match(bodies.videoRouteCandle, new RegExp(`${binding.symbol}\\s*\\(`));
  }
});

test("every bound image lane exists in CANDLE_IMAGE_ROUTES and the unbound lanes are published", () => {
  const lanes = parseCandleImageLanes(bodies.routingCandle);
  for (const lane of Object.keys(IMAGE_LANE_BINDINGS)) {
    assert.ok(lanes.has(lane), `${lane} is bound but is not a CANDLE_IMAGE_ROUTES lane`);
  }
  const { unbound } = deriveLaneBindings({
    imageLanes: lanes,
    mechanismFacts: deriveMechanismFacts(bodies),
  });
  assert.deepEqual(artifacts.inventory.summary.imageLanesWithoutAdmissionBinding, unbound);
  for (const lane of unbound) {
    assert.ok(
      artifacts.inventory.divergences.some(
        (divergence) =>
          divergence.kind === "image_lane_without_admission_binding" &&
          divergence.modelId === `lane:${lane}`,
      ),
      `${lane} is unbound but is not recorded as a divergence`,
    );
  }
});

test("overlay providers are recorded as admission coordinates in their own right", () => {
  // `z_image_control` has a candle load-shape rule AND a named request scope, but no catalog row —
  // it is reached as an overlay on `z_image`. An inventory keyed only on model ids drops it, and a
  // dropped coordinate is how a migration ships "every route converged" with a control overlay
  // still on the old mechanism. Reconciled from the Rust side too, in memory_route_registry.rs.
  const routedEngines = new Set(
    artifacts.inventory.routes.map((route) => route.engineId).filter(Boolean),
  );
  const dispatch = parseRequestScopeDispatch(bodies.candleMemoryStrategy);
  const overlay = new Map(
    artifacts.inventory.overlayProviders.map((provider) => [provider.providerId, provider]),
  );
  for (const engineId of dispatch.named.keys()) {
    assert.ok(
      routedEngines.has(engineId) || overlay.has(engineId),
      `${engineId} has a named request scope but appears in neither routes nor overlayProviders`,
    );
  }
  for (const provider of overlay.values()) {
    assert.ok(!routedEngines.has(provider.providerId), "an overlay provider must not be a route");
    if (provider.sharedSelector.via === "declaration_catch_all") {
      assert.ok(provider.baseModels.length > 0);
    }
  }
  assert.equal(artifacts.inventory.summary.overlayProviders, overlay.size);
});

test("every route lists the conditioned lanes that serve it, and the bound subset", () => {
  const lanes = parseCandleImageLanes(bodies.routingCandle);
  for (const route of artifacts.inventory.routes) {
    const expected = [...lanes.entries()]
      .filter(([, models]) => models.has(route.modelId))
      .map(([lane]) => lane)
      .sort();
    assert.deepEqual(route.conditionedLanes, expected, `${route.modelId}: conditioned lanes`);
    for (const bound of route.overlayLanes) {
      assert.ok(
        route.conditionedLanes.includes(bound),
        `${route.modelId}: bound lane ${bound} is not one of its conditioned lanes`,
      );
      assert.ok(bound in IMAGE_LANE_BINDINGS);
    }
  }
});

test("geometry awareness is read off the gate signatures, not declared", () => {
  const facts = deriveMechanismFacts(bodies);
  const byId = new Map(facts.map((fact) => [fact.id, fact]));
  // sc-19054 landed epic requirement R3: the legacy scalar gate's prediction entry now takes the
  // request geometry (`predicted_peak_gb_for_request(…, geometry: MemoryGeometry)`), so the
  // mechanism is geometry-aware — the exact flip the pre-sc-19054 assertion here said only that
  // story was allowed to make.
  assert.equal(byId.get("legacy_scalar_gate").geometryAware, true);
  assert.ok(byId.get("legacy_scalar_gate").geometryAxes.includes("width"));
  // krea_2_turbo's declared phase curves are the one geometry-aware candle IMAGE path today.
  assert.equal(byId.get("krea_turbo_fit").geometryAware, true);
  assert.ok(byId.get("krea_turbo_fit").geometryAxes.includes("width"));
  assert.equal(byId.get("flat_video_fit_error").geometryAware, true);
  assert.ok(byId.get("flat_video_fit_error").geometryAxes.includes("frames"));
  // Regression pin for the struct-parameter blind spot: these three take their geometry as
  // `geometry: MemoryGeometry`, and a scan of the parameter tokens reported every one of them as
  // geometry-blind — the inverse of the truth, on the three gates epic 19048 converges ONTO.
  for (const id of [
    "krea_control_fit",
    "shared_selector_named_revision",
    "shared_selector_bespoke_override",
    "video_memory_curve_bundle",
  ]) {
    assert.equal(byId.get(id).geometryAware, true, `${id} must be geometry-aware`);
    assert.deepEqual(byId.get(id).geometryAxes.includes("width"), true, id);
  }
  // ...and the conditioning overlay genuinely is not: `ConditioningFootprint` carries byte counts.
  assert.equal(byId.get("conditioning_fit").geometryAware, false);
  // Every mechanism except the sentinel names a real definition site.
  for (const fact of facts) {
    if (fact.id === "unreached") {
      assert.equal(fact.definedIn, null);
      continue;
    }
    assert.ok(fact.definedIn, `${fact.id} has no definition site`);
    assert.ok(fact.definitionSymbols.length > 0);
  }
  assert.deepEqual(
    facts.map((fact) => fact.id),
    ADMISSION_MECHANISMS.map((mechanism) => mechanism.id),
  );
});

// -------------------------------------------------------------------------------------------------
// Guards: each must FAIL when the tree drifts under it
// -------------------------------------------------------------------------------------------------

test("a request-scope dispatch with no catch-all arm is rejected", () => {
  assert.throws(
    () =>
      parseRequestScopeDispatch(
        bodies.candleMemoryStrategy.replace(/_ => DECLARATION_REQUEST_EVIDENCE_REVISION,/, ""),
      ),
    /no catch-all arm/,
  );
});

test("a video fit symbol renamed out of vram_gate.rs is rejected", () => {
  assert.throws(
    () =>
      buildInventory({
        ...bodies,
        vramGate: bodies.vramGate.replaceAll("fn mochi_fit_error", "fn mochi_fit_error_renamed"),
      }),
    /defines no fn mochi_fit_error/,
  );
});

test("a candle video engine with no admission binding is rejected", () => {
  assert.throws(
    () =>
      buildInventory({
        ...bodies,
        videoRouteCandle: bodies.videoRouteCandle.replace(
          '"svd" => Some("svd_xt"),',
          '"svd" => Some("svd_xt"),\n        "unbound_model" => Some("unbound_engine"),',
        ),
      }),
    /unbound_engine.*no VIDEO_ADMISSION_BINDINGS entry/s,
  );
});

test("a bound image lane that leaves CANDLE_IMAGE_ROUTES is rejected", () => {
  assert.throws(
    () =>
      buildInventory({
        ...bodies,
        routingCandle: bodies.routingCandle.replaceAll(
          "CandleImageLane::Pulid,",
          "CandleImageLane::PulidRenamed,",
        ),
      }),
    /no longer serves that lane/,
  );
});

test("a mechanism whose definition symbols all vanish is rejected", () => {
  assert.throws(
    () =>
      deriveMechanismFacts({
        ...bodies,
        kreaControlFit: bodies.kreaControlFit
          .replaceAll("fn fit_ladder", "fn gone_a")
          .replaceAll("fn predicted_control_peak_gb", "fn gone_b")
          .replaceAll("fn predicted_control_sequential_peak_gb", "fn gone_c")
          .replaceAll("fn incurred_peak_gb", "fn gone_d"),
      }),
    /none of its definition symbols exist/,
  );
});

test("the missing headroom constant is rejected", () => {
  assert.throws(
    () =>
      buildInventory({
        ...bodies,
        fitGate: bodies.fitGate.replace(
          /const DEDICATED_VRAM_ALLOCATOR_SLACK_GB:\s*f64\s*=\s*[0-9.]+\s*;/,
          "",
        ),
      }),
    /DEDICATED_VRAM_ALLOCATOR_SLACK_GB/,
  );
});

test("--self-test passes on the current tree", async () => {
  const result = await selfTest();
  assert.ok(result.checks > 0);
});

// -------------------------------------------------------------------------------------------------
// The mirrored candle gate
// -------------------------------------------------------------------------------------------------

const HEADROOM = 2.0;

test("predicted_peak_gb reads the measured tier, then the nvfp4 -> q8 fallback, then the floor", () => {
  const entry = {
    candle: { vramGbByTier: { q4: 10, q8: 20 }, minMemoryGb: 7 },
  };
  assert.equal(predictedPeakGb(entry, "q4", HEADROOM), 12);
  // NVFP4 with no measured row degrades to the q8 row (a deliberate over-prediction), NOT to
  // minMemoryGb, which would size an FP4 render against the lightest tier and fail PERMISSIVELY.
  assert.equal(predictedPeakGb(entry, "nvfp4", HEADROOM), 22);
  // A tier with no row and no NVFP4 escape lands on the declared floor, WITHOUT headroom — the
  // manifest already pads that number.
  assert.equal(predictedPeakGb(entry, "bf16", HEADROOM), 7);
  // No candle block at all: the gate is inert and never blocks.
  assert.equal(predictedPeakGb({}, "q4", HEADROOM), null);
  assert.equal(predictedPeakGb({ candle: { minMemoryGb: 7 } }, "nvfp4", HEADROOM), 7);
});

test("predicted_sequential_peak_gb has no declared-floor fallback", () => {
  const entry = { candle: { sequentialPeakGb: { q8: 15 }, minMemoryGb: 7 } };
  assert.equal(predictedSequentialPeakGb(entry, "q8", HEADROOM), 17);
  assert.equal(predictedSequentialPeakGb(entry, "nvfp4", HEADROOM), 17);
  // Unmeasured for this tier keeps the best-effort staged run, so it must be null, not the floor.
  assert.equal(predictedSequentialPeakGb(entry, "q4", HEADROOM), null);
  assert.equal(predictedSequentialPeakGb({ candle: {} }, "q8", HEADROOM), null);
});

test("fit_decision never blocks without both a prediction and a budget", () => {
  assert.equal(fitDecision(null, { freeGb: 8, totalGb: 8 }), "unknown");
  assert.equal(fitDecision(40, null), "unknown");
  assert.equal(fitDecision(40, { freeGb: 8, totalGb: 8 }), "too_big");
  assert.equal(fitDecision(4, { freeGb: 8, totalGb: 8 }), "fits");
});

test("load_plan is monotonic in the budget: reject -> sequential -> resident", () => {
  // The invariant sc-13960's evict-then-reclaim two-pass leans on. A larger free_gb may only
  // improve the plan.
  const rank = { reject: 0, sequential: 1, resident: 2 };
  for (const sequentialCapable of [true, false]) {
    for (const [needed, staged] of [
      [40, 20],
      [40, null],
      [null, 20],
      [12, 12],
    ]) {
      let previous = -1;
      for (const freeGb of [4, 8, 16, 24, 48, 96]) {
        const plan = loadPlan(needed, staged, { freeGb, totalGb: freeGb }, sequentialCapable);
        assert.ok(
          rank[plan] >= previous,
          `plan regressed at ${freeGb} GB (needed=${needed}, staged=${staged}, seq=${sequentialCapable})`,
        );
        previous = rank[plan];
      }
    }
  }
  // A lane that cannot stage turns a resident overflow straight into a reject.
  assert.equal(loadPlan(40, 10, { freeGb: 20, totalGb: 20 }, false), "reject");
  assert.equal(loadPlan(40, 10, { freeGb: 20, totalGb: 20 }, true), "sequential");
  // A staged peak that ALSO overflows rejects rather than staging best-effort.
  assert.equal(loadPlan(40, 30, { freeGb: 20, totalGb: 20 }, true), "reject");
  // An unmeasured staged peak keeps today's best-effort staging.
  assert.equal(loadPlan(40, null, { freeGb: 20, totalGb: 20 }, true), "sequential");
});

// -------------------------------------------------------------------------------------------------
// The decision baseline
// -------------------------------------------------------------------------------------------------

test("the baseline is built from the Rust-emitted decisions, not re-derived here", () => {
  const { baseline } = artifacts;
  assert.equal(baseline.generatedFrom.decisions.path, DECISIONS_PATH);
  assert.match(baseline.generatedFrom.decisions.digest, /^sha256:[0-9a-f]{64}$/);
  // The producer must be the Rust module driving the real gate, named in the artifact itself.
  assert.equal(
    baseline.generatedFrom.decisions.producedBy.module,
    "crates/sceneworks-worker/src/candle_admission_decisions.rs",
  );
  assert.equal(
    baseline.generatedFrom.decisions.producedBy.gateModule,
    "crates/sceneworks-worker/src/candle_scalar_gate.rs",
  );
  assert.equal(baseline.candle.length, decisions.rows.length);
  assert.equal(baseline.summary.candleRows, baseline.candle.length);
});

test("the evaluated corpus is exactly gate-reaching routes x tier x geometry x budget", () => {
  const { baseline } = artifacts;
  const evaluated = baseline.candle.filter((row) => row.resolution === "evaluated");
  const notEvaluated = baseline.candle.filter((row) => row.resolution === "not_evaluated");
  const evaluatedRoutes = new Set(evaluated.map((row) => row.modelId));
  const notEvaluatedRoutes = new Set(notEvaluated.map((row) => row.modelId));

  // Shape, not a frozen magic number: an evaluated route contributes the full cross product and a
  // non-evaluated one contributes one row per tier and NO coordinate axes.
  assert.equal(
    evaluated.length,
    evaluatedRoutes.size *
      TIER_ORDER.length *
      BASELINE_IMAGE_GEOMETRIES.length *
      BASELINE_BUDGETS_GB.length,
  );
  assert.equal(notEvaluated.length, notEvaluatedRoutes.size * TIER_ORDER.length);
  // Every candle route is in exactly one bucket, and every route is covered.
  assert.equal(evaluatedRoutes.size + notEvaluatedRoutes.size, artifacts.inventory.routes.length);
  for (const modelId of evaluatedRoutes) assert.ok(!notEvaluatedRoutes.has(modelId));

  const seen = new Set();
  for (const row of evaluated) {
    const key = `${row.modelId}|${row.tier}|${row.width}x${row.height}x${row.frames}|${row.budgetGb}`;
    assert.ok(!seen.has(key), `duplicate baseline coordinate ${key}`);
    seen.add(key);
    assert.ok(TIER_ORDER.includes(row.tier));
    assert.ok(BASELINE_BUDGETS_GB.includes(row.budgetGb));
  }
});

test("a not_evaluated row names its gate and the input that gate needs, and publishes no axes", () => {
  // The fix for the row class that used to claim `resolution: "evaluated"` with decision columns
  // its actual gate never produced. A row that records no decision must not record a coordinate.
  const notEvaluated = artifacts.baseline.candle.filter(
    (row) => row.resolution === "not_evaluated",
  );
  assert.ok(notEvaluated.length > 0, "every candle route is now evaluated — update this test");
  for (const row of notEvaluated) {
    assert.ok(row.gate && row.gate.length > 0, `${row.modelId}: no gate named`);
    assert.ok(
      typeof row.missingInput === "string" && row.missingInput.length > 40,
      `${row.modelId}: a not_evaluated row must justify itself`,
    );
    for (const axis of ["width", "height", "frames", "budgetGb"]) {
      assert.ok(!(axis in row), `${row.modelId}: a not_evaluated row must not publish ${axis}`);
    }
    for (const column of [
      "predictedPeakGb",
      "loadPlanSequentialCapable",
      "loadPlanSequentialIncapable",
    ]) {
      assert.ok(!(column in row), `${row.modelId}: a not_evaluated row must not publish ${column}`);
    }
  }
  // Every video route is in this bucket — the flat fit errors read weight bytes off disk.
  for (const row of artifacts.baseline.candle) {
    if (row.modality !== "video") continue;
    assert.equal(row.resolution, "not_evaluated", `${row.modelId}: a video row cannot be evaluated`);
  }
});

test("the JS gate mirror reproduces the Rust-emitted decision for every evaluated row", () => {
  // THE reconciliation. The mirror further up the generator is a readable statement of the law; the
  // committed decisions are the output of `candle_scalar_gate.rs` itself. If the two disagree on any
  // coordinate, the mirror has drifted and this fails — which is the property that makes keeping a
  // JS mirror at all defensible.
  const headroom = artifacts.inventory.constants.headroomGb;
  const entries = new Map(
    JSON.parse(stripJsoncComments(bodies.manifest)).models.map((model) => [model.id, model]),
  );
  let checked = 0;
  for (const row of decisions.rows) {
    if (row.resolution !== "evaluated") continue;
    const entry = entries.get(row.modelId) ?? null;
    const budget = { freeGb: row.budgetGb, totalGb: row.budgetGb };
    // sc-19054: the mirror grades per geometry, exactly like the Rust gate.
    const geometry = { width: row.width, height: row.height, batch: 1, frames: row.frames };
    const needed = predictedPeakGbForRequest(entry, row.tier, headroom, geometry);
    const staged = predictedSequentialPeakGbForRequest(entry, row.tier, headroom, geometry);
    const label = `${row.modelId}/${row.tier}/${row.width}x${row.height}/${row.budgetGb}`;
    assert.equal(
      scalarPeakClass(entry, "vramGbByTier", row.tier, geometry),
      row.scalarClass,
      `${label}: scalarClass`,
    );
    assert.equal(round(needed), row.predictedPeakGb, `${label}: predictedPeakGb`);
    assert.equal(round(staged), row.predictedSequentialPeakGb, `${label}: predictedSequentialPeakGb`);
    for (const [capable, fitKey, planKey] of [
      [true, "fitDecisionSequentialCapable", "loadPlanSequentialCapable"],
      [false, "fitDecisionSequentialIncapable", "loadPlanSequentialIncapable"],
    ]) {
      const decision = fitDecision(needed, budget);
      const resolved = decision === "too_big" && capable ? "offload" : decision;
      assert.equal(row[fitKey], resolved, `${label}: ${fitKey} drifted from the JS mirror`);
      assert.equal(
        row[planKey],
        loadPlan(needed, staged, budget, capable),
        `${label}: ${planKey} drifted from the JS mirror`,
      );
    }
    checked += 1;
  }
  assert.ok(checked > 0, "no evaluated rows were reconciled");
});

function round(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return null;
  return Math.round(value * 1e6) / 1e6;
}

test("the JS corpus axes match the Rust emitter's, so neither side can halve the grid", () => {
  assert.deepEqual(decisions.corpus.tiers, [...TIER_ORDER]);
  assert.deepEqual(decisions.corpus.budgetsGb, [...BASELINE_BUDGETS_GB]);
  assert.deepEqual(
    decisions.corpus.imageGeometries,
    BASELINE_IMAGE_GEOMETRIES.map((geometry) => ({ ...geometry })),
  );
});

test("the sequential-capability input is the registry descriptor, never the manifest key", () => {
  // The defect this replaces: `sequentialCapable` was read from `candle.supportsSequentialOffload`
  // while production reads `mlx_fit_gate::engine_supports_sequential`, flipping loadPlan
  // reject<->sequential on tight budgets for every route where the two disagree.
  assert.equal(decisions.sequentialCapability.manifestKeyIsNotTheInput, true);
  assert.match(decisions.sequentialCapability.productionInput, /engine_supports_sequential/);
  for (const row of artifacts.baseline.candle) {
    assert.ok(
      !("sequentialCapable" in row),
      `${row.modelId}: a resolved sequentialCapable column would be a fabricated input`,
    );
  }
  // Both branches are enumerated, and they actually differ somewhere — an enumeration that
  // collapsed to one answer everywhere would be recording nothing.
  const evaluated = artifacts.baseline.candle.filter((row) => row.resolution === "evaluated");
  assert.ok(
    evaluated.some((row) => row.loadPlanSequentialCapable !== row.loadPlanSequentialIncapable),
    "the two capability branches never differ — the enumeration is inert",
  );
  const gap = artifacts.inventory.knownGaps.find(
    (entry) => entry.id === "candle-sequential-capability-needs-the-linked-candle-bundle",
  );
  assert.ok(gap, "the unresolved sequential-capability input is not recorded as a gap");
  assert.equal(gap.owner, "sc-19050");
});

test("candle scalar admission is geometry-aware since sc-19054, and the baseline shows the split", () => {
  // The sc-19054 successor of "candle admission is geometry-blind today" — that test's own doc
  // said sc-19054 is the slice that breaks it, and this is the enumerated replacement. Every
  // evaluated coordinate publishes the full geometry axis; the scalar is classed per cell
  // (`measured_peak` where the declared capture covers the request, `declared_floor` otherwise),
  // floor cells predict strictly above their coordinate's covered peak, and at least one
  // coordinate must witness the axis actually splitting the prediction.
  const byCoordinate = new Map();
  for (const row of artifacts.baseline.candle) {
    if (row.resolution !== "evaluated") continue;
    assert.equal(
      row.geometrySensitive,
      true,
      `${row.modelId}: every evaluated row rides the geometry-aware scalar gate since sc-19054`,
    );
    const key = `${row.modelId}|${row.tier}|${row.budgetGb}`;
    if (!byCoordinate.has(key)) byCoordinate.set(key, []);
    byCoordinate.get(key).push(row);
  }
  let witnessedSplit = false;
  for (const [key, rows] of byCoordinate) {
    assert.equal(rows.length, BASELINE_IMAGE_GEOMETRIES.length, `${key}: geometry axis is missing`);
    const covered = rows.filter((row) => row.scalarClass === "measured_peak");
    const floors = rows.filter((row) => row.scalarClass === "declared_floor");
    assert.equal(covered.length + floors.length, rows.length, `${key}: unclassified rows`);
    if (covered.length > 0 && floors.length > 0) {
      const coveredPeaks = new Set(covered.map((row) => row.predictedPeakGb));
      assert.equal(coveredPeaks.size, 1, `${key}: covered cells must share the one measured peak`);
      const coveredPeak = covered[0].predictedPeakGb;
      for (const row of floors) {
        if (row.predictedPeakGb === null) continue;
        assert.ok(
          row.predictedPeakGb > coveredPeak,
          `${key}@${row.width}x${row.height}: a floor-classed cell must be widened above the ` +
            `covered measured peak (${row.predictedPeakGb} vs ${coveredPeak})`,
        );
        witnessedSplit = true;
      }
    }
  }
  assert.ok(byCoordinate.size > 0, "no evaluated coordinates — the corpus is empty");
  assert.ok(
    witnessedSplit,
    "no coordinate split across the geometry axis — the corpus is not exercising sc-19054",
  );
});

// -------------------------------------------------------------------------------------------------
// Geometry-awareness is resolved PER GATE FUNCTION, including struct-wrapped geometry
// -------------------------------------------------------------------------------------------------

test("struct-wrapped geometry is seen, and per-symbol facts are never unioned across a mechanism", () => {
  const gates = new Map(artifacts.inventory.gateGeometry.map((gate) => [gate.gate, gate]));

  // A scan of the parameter TOKENS reports all three of these as geometry-blind: the geometry
  // arrives inside `geometry: MemoryGeometry` (declared in the pinned inference dependency, but
  // BUILT in fingerprinted sources) and inside `query: VideoCurveQuery`, whose `geometry` field is
  // four fields deep behind four unrelated enums.
  for (const symbol of [
    "krea_control_fit::fit_ladder_for_entry_with_runtime",
    "evaluate_shared_image",
    "evaluate_shared_bespoke_image",
    "evaluate",
  ]) {
    const gate = gates.get(symbol);
    assert.ok(gate, `${symbol} has no resolved geometry fact`);
    assert.equal(gate.geometryAware, true, `${symbol} must be geometry-aware`);
    assert.ok(
      gate.reachedVia.some((via) => via.startsWith("struct_parameter:")),
      `${symbol}: geometry must be reached through a struct parameter, got ${gate.reachedVia}`,
    );
  }

  // The mechanism-union defect: `flat_video_fit_error` names seven symbols, two of which take
  // geometry scalars. The other five are deliberately resolution-blind and must say so.
  for (const symbol of ["svd_fit_error", "mochi_fit_error"]) {
    assert.equal(gates.get(symbol).geometryAware, true, symbol);
    assert.deepEqual(gates.get(symbol).geometryAxes, ["frames", "height", "width"]);
  }
  for (const symbol of [
    "wan_video_fit_error",
    "wan_video_fit_error_with_adapter_bytes",
    "scail2_video_fit_error",
    "scail2_video_fit_error_with_adapter_bytes",
    "video_weights_fit_error",
  ]) {
    assert.equal(
      gates.get(symbol).geometryAware,
      false,
      `${symbol} is resolution-blind; a mechanism union labelled it aware from svd_fit_error`,
    );
  }
  // The module-qualified scalar-gate label is the sc-19054 COMPOSITION (`load_plan` graded by
  // `predicted_peak_gb_for_request`) and is geometry-aware, while the bare `load_plan` symbol
  // stays signature-derived and blind — the dataflow composition is labelled, never unioned.
  assert.equal(gates.get("candle_scalar_gate::load_plan").geometryAware, true);
  assert.equal(gates.get("load_plan").geometryAware, false);
  assert.equal(gates.get("conditioning_fit::decide").geometryAware, false);

  // The mechanism-level flag survives only as a summary, and it must be exactly "any symbol is".
  for (const mechanism of artifacts.inventory.mechanisms) {
    assert.equal(
      mechanism.geometryAware,
      mechanism.symbols.some((entry) => entry.geometryAware),
      `${mechanism.id}: the mechanism flag is not the OR of its symbols`,
    );
  }
  const videoMechanism = artifacts.inventory.mechanisms.find(
    (mechanism) => mechanism.id === "flat_video_fit_error",
  );
  assert.ok(
    videoMechanism.geometryAwareSymbols.length > 0 &&
      videoMechanism.geometryBlindSymbols.length > 0,
    "flat_video_fit_error must publish BOTH lists — that split is the whole point",
  );
});

test("every decision row's geometrySensitive comes from the gate that route reaches", () => {
  const gates = new Map(artifacts.inventory.gateGeometry.map((gate) => [gate.gate, gate]));
  for (const row of artifacts.baseline.candle) {
    const gate = gates.get(row.gate);
    assert.ok(gate, `${row.modelId}: unresolved gate ${row.gate}`);
    assert.equal(
      row.geometrySensitive,
      gate.geometryAware,
      `${row.modelId}: geometrySensitive does not match its own gate ${row.gate}`,
    );
    assert.deepEqual(row.geometryAxes, gate.geometryAxes);
  }
  // The evaluated surface is the scalar gate, geometry-AWARE since sc-19054: the peak `load_plan`
  // compares is computed by `predicted_peak_gb_for_request`, so an evaluated row that claimed
  // geometry-blindness would be publishing an axis it responds to without saying so — the inverse
  // of the 0-of-N-variance class the original assertion replaced.
  for (const row of artifacts.baseline.candle) {
    if (row.resolution !== "evaluated") continue;
    assert.equal(
      row.geometrySensitive,
      true,
      `${row.modelId}: the evaluated scalar gate is geometry-aware since sc-19054`,
    );
  }
});

test("a gate function's geometry facts are resolvable for every declared symbol", () => {
  for (const mechanism of ADMISSION_MECHANISMS) {
    if (!mechanism.source) continue;
    for (const symbol of mechanism.definitionSymbols) {
      const resolved = resolveSymbolGeometry(bodies, mechanism.source, symbol);
      if (!resolved) continue;
      // A parameter type that could not be resolved is PUBLISHED, never silently read as
      // "no geometry". Any that appear must be non-geometry handles.
      for (const unresolved of resolved.unresolvedParameterTypes) {
        assert.ok(
          !/geom/i.test(unresolved.parameter),
          `${symbol}: geometry parameter ${unresolved.parameter} is unresolved`,
        );
      }
    }
  }
});

test("mlx baseline rows are declared-resolution and carry no fabricated geometry axis", () => {
  for (const row of artifacts.baseline.mlx) {
    assert.equal(row.resolution, "declared");
    assert.ok(!("width" in row), "an MLX declared row must not fabricate a geometry column");
    assert.ok(!("budgetGb" in row), "an MLX declared row must not fabricate a budget column");
    assert.ok(TIER_ORDER.includes(row.tier));
  }
  assert.ok(artifacts.baseline.mlx.length > 0);
  assert.match(artifacts.baseline.corpus.resolutions.declared, /sc-19050/);
});

test("the baseline is deterministic under a re-derivation from the same inputs", () => {
  const again = buildBaseline(bodies, artifacts.inventory, decisions);
  assert.equal(JSON.stringify(again), JSON.stringify(artifacts.baseline));
});

test("the baseline refuses decisions that contradict the derived gate facts", () => {
  // The reconciliation is a REFUSAL, not a report: the Rust emitter's `gate_is_geometry_aware` match
  // may not stand as a second hand-written claim beside this file's signature-derived one.
  assert.throws(
    () =>
      buildBaseline(bodies, artifacts.inventory, {
        ...decisions,
        rows: decisions.rows.map((row, index) =>
          index === 0 ? { ...row, geometrySensitive: !row.geometrySensitive } : row,
        ),
      }),
    /not allowed to disagree/,
  );
  assert.throws(
    () =>
      buildBaseline(bodies, artifacts.inventory, {
        ...decisions,
        rows: decisions.rows.map((row, index) =>
          index === 0 ? { ...row, gate: "a_gate_nobody_resolved" } : row,
        ),
      }),
    /resolved no geometry fact for/,
  );
  assert.throws(
    () =>
      buildBaseline(bodies, artifacts.inventory, {
        ...decisions,
        rows: [...decisions.rows, { ...decisions.rows[0], modelId: "not_a_route" }],
      }),
    /not an inventory route/,
  );
});

// -------------------------------------------------------------------------------------------------
// Recorded gaps
// -------------------------------------------------------------------------------------------------

test("the memory-matrix fingerprint gap is recorded, owned, and still real", async () => {
  const gap = artifacts.inventory.knownGaps.find(
    (entry) => entry.id === "memory-matrix-source-paths-under-cover-candle-admission",
  );
  assert.ok(gap, "the memory-matrix SOURCE_PATHS gap is not recorded");
  assert.equal(gap.owner, "sc-19059");
  // Recorded gaps must be checked, not just asserted: the day the matrix DOES fingerprint these
  // files, this test fails and the stale gap gets deleted rather than quietly outliving its truth.
  const matrix = await readFile(new URL("./generate-memory-matrix.mjs", import.meta.url), "utf8");
  const declared = matrix.slice(matrix.indexOf("export const SOURCE_PATHS"));
  for (const relative of gap.paths) {
    assert.ok(
      !declared.includes(`"${relative}"`),
      `${relative} is now in the matrix SOURCE_PATHS — delete the recorded gap`,
    );
  }
});

test("the manifest carries a bare `measured` boolean and no evidence-class enum", async () => {
  // The story and the epic assume epic 18472 landed an evidence-CLASS flag to adopt. It did not.
  // Pinning the actual shape stops a later slice from designing against a field with no producer.
  const gap = artifacts.inventory.knownGaps.find(
    (entry) => entry.id === "measured-flag-is-a-bare-boolean",
  );
  assert.ok(gap);
  const manifest = JSON.parse(stripJsoncComments(bodies.manifest));
  let measuredCount = 0;
  const visit = (node) => {
    if (!node || typeof node !== "object") return;
    if (Array.isArray(node)) return node.forEach(visit);
    for (const [key, value] of Object.entries(node)) {
      assert.ok(
        key !== "evidenceClass" && key !== "evidence_class",
        "an evidence-class field now exists — update the recorded gap and adopt it",
      );
      if (key === "measured") {
        measuredCount += 1;
        assert.equal(typeof value, "boolean", "`measured` is no longer a bare boolean");
      }
      visit(value);
    }
  };
  visit(manifest.models);
  assert.ok(measuredCount > 0);
});

test("the packaged video curves are MLX-only, which is why no candle route reaches the bundle", () => {
  const lanes = artifacts.inventory.summary.packagedVideoCurveLanes;
  assert.ok(Object.keys(lanes).length > 0, "no packaged curve lanes at all");
  assert.equal(lanes.candle ?? 0, 0, "a candle curve now exists — update the recorded gap");
  assert.equal(artifacts.inventory.summary.byMechanism.video_memory_curve_bundle, 0);
  assert.ok(
    artifacts.inventory.knownGaps.some(
      (gap) => gap.id === "zero-candle-video-memory-curves-packaged",
    ),
  );
});

test("every divergence names a route or a lane that exists", () => {
  const routes = new Set(artifacts.inventory.routes.map((route) => route.modelId));
  const lanes = new Set(
    artifacts.inventory.summary.imageLanesWithoutAdmissionBinding.map((lane) => `lane:${lane}`),
  );
  const providers = new Set(
    artifacts.inventory.overlayProviders.map((provider) => `provider:${provider.providerId}`),
  );
  for (const divergence of artifacts.inventory.divergences) {
    assert.ok(
      routes.has(divergence.modelId) ||
        lanes.has(divergence.modelId) ||
        providers.has(divergence.modelId),
      `divergence ${divergence.kind} names unknown subject ${divergence.modelId}`,
    );
    assert.ok(divergence.detail.length > 20, "a divergence must say what is wrong");
  }
  assert.ok(artifacts.inventory.divergences.length > 0, "epic 19048 has not started yet");
});
