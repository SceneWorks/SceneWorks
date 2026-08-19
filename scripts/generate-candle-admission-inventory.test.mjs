// Tests for the candle admission route inventory + decision baseline (sc-19049, epic 19048).
//
// Two things are being pinned, and they are different in kind:
//
//   1. The artifacts are GENERATED and GUARDED — the fingerprint covers every source the generator
//      reads, the committed files match a fresh build, and each derivation guard actually fails
//      when the tree drifts under it. A hand-written inventory is stale on arrival; these tests are
//      what make "generated" mean something.
//   2. The mirrored candle gate arithmetic agrees with `vram_gate.rs` on the properties that gate
//      has documented invariants for — the NVFP4 tier fallback, the declared-floor fallback, and
//      `load_plan`'s monotonicity in the budget.

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
  BASELINE_VIDEO_GEOMETRIES,
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
  predictedSequentialPeakGb,
  readSources,
  renderArtifacts,
  selfTest,
  sourceRevisionOf,
} from "./generate-candle-admission-inventory.mjs";

const sources = await readSources();
const bodies = sources.bodies;
const artifacts = renderArtifacts(bodies);

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
  const again = renderArtifacts(bodies);
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
  // The legacy scalar gate takes (manifest_entry, tier_key) and nothing else. This assertion is the
  // whole premise of epic requirement R3; if it ever passes as `true` without sc-19054 landing, the
  // parse is lying.
  assert.equal(byId.get("legacy_scalar_gate").geometryAware, false);
  assert.deepEqual(byId.get("legacy_scalar_gate").geometryAxes, []);
  // krea_2_turbo's declared phase curves are the one geometry-aware candle IMAGE path today.
  assert.equal(byId.get("krea_turbo_fit").geometryAware, true);
  assert.ok(byId.get("krea_turbo_fit").geometryAxes.includes("width"));
  assert.equal(byId.get("flat_video_fit_error").geometryAware, true);
  assert.ok(byId.get("flat_video_fit_error").geometryAxes.includes("frames"));
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

test("the baseline covers the declared corpus exactly", () => {
  const { baseline, inventory } = artifacts;
  const imageRoutes = inventory.routes.filter((route) => route.modality === "image").length;
  const videoRoutes = inventory.routes.filter((route) => route.modality === "video").length;
  const expected =
    (imageRoutes * BASELINE_IMAGE_GEOMETRIES.length + videoRoutes * BASELINE_VIDEO_GEOMETRIES.length) *
    TIER_ORDER.length *
    BASELINE_BUDGETS_GB.length;
  assert.equal(baseline.candle.length, expected);
  assert.equal(baseline.summary.candleRows, baseline.candle.length);

  const seen = new Set();
  for (const row of baseline.candle) {
    const key = `${row.modelId}|${row.tier}|${row.width}x${row.height}x${row.frames}|${row.budgetGb}`;
    assert.ok(!seen.has(key), `duplicate baseline coordinate ${key}`);
    seen.add(key);
    assert.ok(TIER_ORDER.includes(row.tier));
    assert.ok(BASELINE_BUDGETS_GB.includes(row.budgetGb));
    assert.equal(row.resolution, "evaluated");
  }
});

test("every baseline decision is reproducible from its own row", () => {
  // The row records its inputs, so the decision it records must be the one those inputs produce.
  // This is what makes a later slice's diff attributable rather than merely visible.
  for (const row of artifacts.baseline.candle) {
    const budget = { freeGb: row.budgetGb, totalGb: row.budgetGb };
    assert.equal(
      row.fitDecision,
      loadPlanDecision(row.predictedPeakGb, budget, row.sequentialCapable),
      `${row.modelId}/${row.tier}/${row.budgetGb}: fitDecision is not reproducible`,
    );
    assert.equal(
      row.loadPlan,
      loadPlan(row.predictedPeakGb, row.predictedSequentialPeakGb, budget, row.sequentialCapable),
      `${row.modelId}/${row.tier}/${row.budgetGb}: loadPlan is not reproducible`,
    );
  }
});

function loadPlanDecision(needed, budget, sequentialCapable) {
  const decision = fitDecision(needed, budget);
  return decision === "too_big" && sequentialCapable ? "offload" : decision;
}

test("candle admission is geometry-blind today, and the baseline says so explicitly", () => {
  // The single most important property this baseline records. Every image route on the scalar gate
  // must decide identically across the geometry axis; sc-19054 is the slice that breaks this, and
  // when it does the diff is the deliverable.
  const byCoordinate = new Map();
  for (const row of artifacts.baseline.candle) {
    if (row.geometrySensitive) continue;
    const key = `${row.modelId}|${row.tier}|${row.budgetGb}`;
    if (!byCoordinate.has(key)) byCoordinate.set(key, []);
    byCoordinate.get(key).push(row);
  }
  for (const [key, rows] of byCoordinate) {
    const plans = new Set(rows.map((row) => row.loadPlan));
    assert.equal(plans.size, 1, `${key}: a geometry-blind coordinate produced ${plans.size} plans`);
  }
  assert.ok(byCoordinate.size > 0, "no geometry-blind coordinates — the corpus is empty");
});

test("mlx baseline rows are declared-resolution and carry no fabricated geometry axis", () => {
  for (const row of artifacts.baseline.mlx) {
    assert.equal(row.resolution, "declared");
    assert.ok(!("width" in row), "an MLX declared row must not fabricate a geometry column");
    assert.ok(!("budgetGb" in row), "an MLX declared row must not fabricate a budget column");
    assert.ok(TIER_ORDER.includes(row.tier));
  }
  assert.ok(artifacts.baseline.mlx.length > 0);
  assert.match(artifacts.baseline.corpus.resolutions.mlx, /sc-19050/);
});

test("the baseline is deterministic under a re-derivation from the same inventory", () => {
  const again = buildBaseline(bodies, artifacts.inventory);
  assert.equal(JSON.stringify(again), JSON.stringify(artifacts.baseline));
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
