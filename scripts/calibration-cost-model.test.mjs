import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  COLLAPSING_AXES,
  DEFAULT_PARAMETERS,
  NA_SELECTION_RULE,
  OVERLAY_LOAD_CONTRACT,
  PRODUCIBILITY_GATES,
  SOURCE_PATHS,
  batchedSessionKeysFromPlan,
  buildCostModel,
  cellCensus,
  collapseToRuns,
  fitSensitivity,
  fitVersusExhaustive,
  kreaFitPrecedent,
  naSensitivity,
  producibility,
  rankUncertainties,
} from "./calibration-cost-model.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";
import { stripInertLines } from "./lib/source-revision.mjs";

const RUNGS = [
  "resident",
  "staged_residency",
  "bounded_decode",
  "bounded_attention",
  "bounded_transformer_residency",
];

test("comment-only manifest edits produce no calibration cost-model change", async () => {
  const manifestUrl = new URL("../config/manifests/builtin.models.jsonc", import.meta.url);
  const manifest = readFileSync(manifestUrl, "utf8");
  const baseline = await buildCostModel();
  const commentOnly = await buildCostModel({
    sourceOverrides: {
      manifest: `${manifest}\n// SC-16129 regression: semantically inert comment\n`,
    },
  });
  const withoutAnyComments = await buildCostModel({
    sourceOverrides: {
      manifest: JSON.stringify(JSON.parse(stripJsoncComments(manifest))),
    },
  });
  assert.deepEqual(commentOnly, baseline);
  assert.deepEqual(withoutAnyComments, baseline);

  const parsed = JSON.parse(stripJsoncComments(manifest));
  parsed.models[0].name = `${parsed.models[0].name} semantic mutation`;
  const semanticChange = await buildCostModel({
    sourceOverrides: { manifest: JSON.stringify(parsed) },
  });
  assert.notEqual(
    semanticChange.generatedFrom.sceneWorksRevision,
    baseline.generatedFrom.sceneWorksRevision,
    "the override must be exercised and semantic manifest changes must remain visible",
  );
});

test("comment-only Rust and generator-script edits produce no calibration cost-model change", async () => {
  // sc-16268: this document hashes `vram_gate.rs`, both memory adapters and two generator scripts,
  // so wiring `check:calibration-cost-model` into `rust:check` without this would just move the
  // inert churn from CI to the pre-push gate.
  // DERIVED from the generator's own list, so a source dropped from the fingerprint cannot leave
  // this test green.
  const sources = Object.fromEntries(
    Object.entries(SOURCE_PATHS).filter(
      ([, relative]) => relative.endsWith(".rs") || relative.endsWith(".mjs"),
    ),
  );
  assert.deepEqual(Object.keys(sources).sort(), [
    "candleAdapter",
    "harness",
    "jsonc",
    "matrixGenerator",
    "mlxAdapter",
    "sourceRevision",
    "vramGate",
  ]);
  const read = (relative) => readFileSync(new URL(`../${relative}`, import.meta.url), "utf8");
  const baseline = await buildCostModel();

  for (const [name, relative] of Object.entries(sources)) {
    const commentOnly = await buildCostModel({
      sourceOverrides: {
        [name]: `${read(relative)}\n// sc-16268 regression: semantically inert comment\n`,
      },
    });
    assert.deepEqual(commentOnly, baseline, `${name}: an appended comment must be inert`);
  }

  // The load-bearing half, mirroring the matrix suite: regenerating from FULLY comment-stripped
  // sources must be identical. This exercises the parsers, not just the hash, so a future
  // cost-model parse anchored on comment text fails here instead of quietly letting a semantic
  // change past the staleness tripwire.
  const stripped = await buildCostModel({
    sourceOverrides: Object.fromEntries(
      Object.entries(sources).map(([name, relative]) => [
        name,
        stripInertLines(read(relative), "//"),
      ]),
    ),
  });
  assert.deepEqual(stripped, baseline);

  // Mutation check: the override seam is real and a semantic edit to the same file is still visible.
  const semanticChange = await buildCostModel({
    sourceOverrides: {
      vramGate: `${read(sources.vramGate)}\nconst SC_16268_PROBE: f64 = 1.0;\n`,
    },
  });
  assert.notEqual(
    semanticChange.generatedFrom.sceneWorksRevision,
    baseline.generatedFrom.sceneWorksRevision,
  );
});

test("control coverage follows CONTROL_LANE_MODELS, not backend measurement blocks", async () => {
  const model = await buildCostModel();
  assert.ok(model.controlCoverage.declaredControlModels.includes("krea_2_turbo"));
  assert.ok(model.controlCoverage.emittedControlPairs.includes("krea_2_turbo|mlx"));
  assert.ok(model.controlCoverage.emittedControlPairs.includes("kolors|candle"));
  assert.ok(model.controlCoverage.emittedControlPairs.includes("z_image|candle"));
  assert.match(model.controlCoverage.citation, /CONTROL_LANE_MODELS/);
  assert.doesNotMatch(
    JSON.stringify(model.controlCoverage),
    /model\[backend\]\.control|knownOmission|manifestDeclarations/,
  );
});

test("shipped batch coverage is empty while complete opted-in groups remain derivable", () => {
  const plan = JSON.parse(readFileSync(
    new URL("../config/memory-calibration-plan.json", import.meta.url),
    "utf8",
  ));
  assert.deepEqual(batchedSessionKeysFromPlan(plan), []);

  const optedIn = structuredClone(plan);
  for (const provider of optedIn.providers.filter(
    (item) => item.fixture === "fresh-five-rung-krea-q4-1024-seed16402-step2",
  )) {
    provider.modelLoadPolicy = "batch_rungs";
    provider.modelLoadGroup = "synthetic-complete-krea-group";
  }
  assert.deepEqual(batchedSessionKeysFromPlan(optedIn), [[
    "krea_2_turbo", "candle", "q4", "text_to_image", "none",
  ].join("\0")]);

  const incomplete = structuredClone(optedIn);
  incomplete.providers = incomplete.providers.filter(
    (provider) => provider.name !== "candle-krea-q4-fresh-reference-resident",
  );
  assert.throws(() => batchedSessionKeysFromPlan(incomplete), /every canonical rung/);
});

test("published overlay prose derives its census and load ratio from the current matrix", async () => {
  const model = await buildCostModel();
  const overlayCells = model.cells.total - model.cells.byOverlay.none;
  const overlayAxis = model.collapsing.axes.find((axis) => axis.axis === "overlay");
  const overlayUncertainty = model.biggestUncertainties.find(
    (entry) => entry.gate === "overlay-declined",
  );

  assert.match(overlayAxis.rule, new RegExp(`${overlayCells} of the ${model.cells.total} cells`));
  assert.match(
    overlayAxis.factor,
    new RegExp(`${model.runs.unavailableOverlayLoadPriceTag.collapseFactor}x`),
  );
  assert.match(
    overlayUncertainty.why,
    new RegExp(`${model.producibility.blockerRanking.overlayChargedByFirstMatch} cells`),
  );
  assert.doesNotMatch(JSON.stringify(model), /3,925|6,595|2\.47x|SC-16073/);
});

/**
 * A small synthetic matrix with the shape that matters:
 *
 * - `alpha` is mlx with TWO tiers and a TWO-resolution envelope -> 2 sessions, 10 cells, 20
 *   geometry-expanded cells. It is the fixture that separates "tier collapses" from "tier does
 *   not", and "geometry multiplies" from "geometry does not".
 * - `beta` is candle with ONE tier and a ONE-resolution envelope -> 1 session, 5 cells, and one of
 *   its cells is already `Structurally N/A`, so the exemption path is exercised too.
 *
 * Every number asserted below is hand-computable from this fixture, which is the point: a cost
 * model nobody can falsify is just a number in a document.
 */
function syntheticCells({ overlays = ["none"] } = {}) {
  const cells = [];
  const push = (modelId, backend, tier, resolutions, state = "Missing") => {
    for (const overlay of overlays) {
      for (const rung of RUNGS) {
        cells.push({
          id: [modelId, modelId, backend, tier, "text_to_image", overlay, rung].join(":"),
          modelId,
          backend,
          tier,
          mode: "text_to_image",
          overlay,
          rung,
          geometryEnvelope: { resolutions },
          state:
            modelId === "beta" && overlay === "none" && rung === "bounded_transformer_residency"
              ? "Structurally N/A"
              : state,
        });
      }
    }
  };
  push("alpha", "mlx", "q4", ["1024x1024", "512x512"]);
  push("alpha", "mlx", "q8", ["1024x1024", "512x512"]);
  push("beta", "candle", "q4", ["1024x1024"]);
  return cells;
}

test("cell census counts cells, sessions and the geometry collapse without interpreting them", () => {
  const census = cellCensus(syntheticCells());

  assert.equal(census.total, 15);
  assert.deepEqual(census.byBackend, { candle: 5, mlx: 10 });
  assert.deepEqual(census.byState, { Missing: 14, "Structurally N/A": 1 });

  // 3 session keys: (alpha, mlx, q4), (alpha, mlx, q8), (beta, candle, q4).
  assert.equal(census.sessions.total, 3);
  assert.deepEqual(census.sessions.byBackend, { candle: 1, mlx: 2 });

  // The rung span must be a CONSTANT for the model to report a collapse factor at all.
  assert.deepEqual(census.sessions.rungsPerSession, [5]);

  // Geometry: alpha's 10 cells advertise 2 resolutions each, beta's 5 advertise 1.
  assert.equal(census.geometryExpandedCells, 10 * 2 + 5 * 1);
  assert.equal(census.geometryCollapseFactor, 1.667);

  // Fit groups drop the tier: (alpha, mlx, t2i, none) and (beta, candle, t2i, none).
  assert.equal(census.fitGroups.total, 2);
  assert.equal(census.fitGroups.tierSum, 3);
});

/**
 * THE COLLAPSING RULE, pinned.
 *
 * Two facts, both derived from harness/generator code and both easy to get wrong:
 *   1. cells -> certifying records is 1:1 (tier, mode and overlay do NOT collapse);
 *   2. sessions are COUNTED from surviving session keys, never DIVIDED out of the record count.
 */
test("the collapsing rule maps cells 1:1 to records and counts sessions rather than dividing", () => {
  const runs = collapseToRuns(syntheticCells(), DEFAULT_PARAMETERS);

  // 15 cells minus the one already-exempt `Structurally N/A` cell.
  assert.deepEqual(runs.certifyingRecords, { total: 14, mlx: 10, candle: 4 });
  assert.equal(runs.exemptions.alreadyStructurallyNotApplicable, 1);
  assert.equal(runs.exemptions.projectedAdditional, 0);

  // Model loads: all three session keys still have work, including beta's (4 of 5 rungs survive).
  assert.deepEqual(runs.warmSessions, { total: 3, mlx: 2, candle: 1 });
  assert.deepEqual(runs.shippedModelLoads, { total: 14, mlx: 10, candle: 4 });
  assert.equal(runs.shippedLoadCollapseFactor, 1);
  assert.equal(runs.rungCollapseFactor, 5);
  // 14 records over 3 loads — NOT the rung span, because one cell is already exempt. Reported to
  // three places precisely so an exempt cell cannot round away into a clean "5".
  assert.equal(runs.recordsPerWarmSession, 4.667);

  // The declared axes must stay in sync with what the numbers above actually demonstrate.
  const verdicts = Object.fromEntries(COLLAPSING_AXES.map((axis) => [axis.axis, axis.verdict]));
  assert.equal(verdicts.cell, "NOT collapsed");
  // sc-16060: geometry collapses for the implementation claim and does NOT for the characterization
  // claim. A verdict that still read a bare "collapsed" would be re-asserting the conflation this
  // axis was rewritten to remove, so the two halves are matched independently rather than pinned as
  // one opaque string.
  assert.match(verdicts.geometry, /^collapsed for the implementation claim/);
  assert.match(verdicts.geometry, /NOT collapsed for the characterization claim$/);
});

/**
 * MUTATION PROOF for fact 1 (cells -> records is 1:1).
 *
 * A plausible wrong rule is "tier collapses, because one loaded generator could measure every
 * tier". Reimplement that rule here and prove it produces a DIFFERENT answer on the same fixture.
 * If the real implementation ever drifts toward it, the assertion above goes red.
 */
test("mutation: a tier-collapsing rule disagrees with the shipped rule", () => {
  const cells = syntheticCells();
  const real = collapseToRuns(cells, DEFAULT_PARAMETERS);

  const tierCollapsed = new Set(
    cells
      .filter((cell) => cell.state !== "Structurally N/A")
      .map((cell) => [cell.modelId, cell.backend, cell.mode, cell.overlay, cell.rung].join(" ")),
  );

  assert.equal(tierCollapsed.size, 9, "the perturbed rule must actually differ on this fixture");
  assert.notEqual(
    real.certifyingRecords.total,
    tierCollapsed.size,
    "tier must not collapse — a q4 record cannot certify a q8 cell",
  );
});

test("only an explicitly covered complete rung group reduces shipped model loads", () => {
  const betaKey = ["beta", "candle", "q4", "text_to_image", "none"].join("\0");
  const runs = collapseToRuns(syntheticCells(), DEFAULT_PARAMETERS, {
    batchedSessionKeys: [betaKey],
  });
  assert.deepEqual(runs.shippedModelLoads, { total: 11, mlx: 10, candle: 1 });
  assert.deepEqual(runs.shippedBatchCoverage, {
    sessionKeys: [betaKey],
    sessions: 1,
    records: 4,
    savedLoads: 3,
  });
});

/**
 * MUTATION PROOF for fact 2 (sessions are counted, not divided).
 *
 * Dividing the record count by the rung span is the obvious shortcut and it is wrong: a session
 * survives as long as ANY of its rungs needs measuring. Exempting four of five rungs on `alpha`
 * makes the two rules disagree by 2x, which is exactly the error the N/A sensitivity table would
 * have inherited.
 */
test("mutation: dividing records by the rung span understates the model loads", () => {
  const cells = syntheticCells();
  const onlyBaseline = {
    ...DEFAULT_PARAMETERS,
    structurallyNotApplicableFractionByRung: {
      resident: 0,
      staged_residency: 1,
      bounded_decode: 1,
      bounded_attention: 1,
      bounded_transformer_residency: 1,
    },
  };
  const runs = collapseToRuns(cells, onlyBaseline);

  // One resident cell per session key survives.
  assert.deepEqual(runs.certifyingRecords, { total: 3, mlx: 2, candle: 1 });
  // ...and every session key still needs its own model load.
  assert.deepEqual(runs.warmSessions, { total: 3, mlx: 2, candle: 1 });

  const divided = Math.ceil(runs.certifyingRecords.total / runs.rungCollapseFactor);
  assert.equal(divided, 1);
  assert.notEqual(
    runs.warmSessions.total,
    divided,
    "sessions must be counted from surviving session keys, not divided out of the record count",
  );
});

/**
 * MUTATION PROOF for the geometry collapse: widening an envelope must move the geometry-expanded
 * count and must NOT move the record count. A rule that keyed cells per resolution would move both.
 */
test("mutation: widening a geometry envelope moves the expansion but not the run count", () => {
  const narrow = syntheticCells();
  const wide = syntheticCells().map((cell) =>
    cell.modelId === "alpha"
      ? { ...cell, geometryEnvelope: { resolutions: ["1024x1024", "512x512", "768x768", "1536x1536"] } }
      : cell,
  );

  assert.notEqual(
    cellCensus(narrow).geometryExpandedCells,
    cellCensus(wide).geometryExpandedCells,
    "the fixture must actually differ on the geometry axis",
  );
  assert.equal(
    collapseToRuns(narrow, DEFAULT_PARAMETERS).certifyingRecords.total,
    collapseToRuns(wide, DEFAULT_PARAMETERS).certifyingRecords.total,
    "geometry is an envelope, not a cell key — widening it must not add runs",
  );
});

test("N/A sensitivity moves the record count and leaves the model loads alone", () => {
  const scenarios = naSensitivity(syntheticCells());
  const byName = Object.fromEntries(scenarios.map((scenario) => [scenario.name, scenario]));

  assert.equal(byName.current.kind, "fact");
  assert.equal(byName.current.certifyingRecords.total, 14);
  assert.equal(byName["everything-but-baseline"].kind, "bound");
  assert.equal(byName["everything-but-baseline"].certifyingRecords.total, 3);

  // The headline finding: no reclassification removes a model load, because the resident baseline
  // can never be Structurally N/A.
  for (const scenario of scenarios) {
    assert.equal(scenario.warmSessions.total, 3, `${scenario.name} must keep all three model loads`);
  }
});

test("the Krea precedent reports under-determined tiers instead of quoting a headline record count", () => {
  const manifest = {
    models: [
      {
        id: "krea_2_turbo",
        candle: {
          turboFit: {
            evidenceRecords: [
              { tier: "q4", width: 768, height: 768 },
              { tier: "q4", width: 1024, height: 1024 },
              { tier: "q8", width: 1024, height: 1024 },
            ],
            phaseCurvesByTier: {
              q4: {
                threeStage: {
                  text: { fixedGb: 4, perMpxGb: 0 },
                  denoise: { fixedGb: 7.15, perMpxGb: 7.98 },
                  decode: { fixedGb: 5.04, perMpxGb: 10.03 },
                },
              },
              q8: {
                threeStage: {
                  text: { fixedGb: 6.45, perMpxGb: 0 },
                  denoise: { fixedGb: 13.76, perMpxGb: 7.98 },
                  decode: { fixedGb: 16.62, perMpxGb: 0 },
                },
              },
            },
          },
        },
      },
    ],
  };

  const precedent = kreaFitPrecedent(manifest);
  assert.equal(precedent.measurementRecords, 3);
  assert.deepEqual(precedent.geometriesByTier, { q4: ["1024x1024", "768x768"], q8: ["1024x1024"] });
  assert.deepEqual(precedent.underDeterminedTiers, ["q8"]);

  // A flat `text` slope is physically correct and must NOT be counted as an undetermined fit;
  // a flat `decode` slope on a single-point tier must be.
  assert.equal(precedent.slopesByTier.q4.flatSlopes, 0);
  assert.equal(precedent.slopesByTier.q8.flatSlopes, 1);
  assert.match(precedent.honestyNote, /two geometry points per tier/);
});

/**
 * PRODUCIBILITY. The gates are ordered and must partition the matrix exactly — a cell counted twice
 * or dropped would silently change the "how much of this is code work" headline.
 */
test("the producibility gates partition every cell exactly once, in priority order", () => {
  const cells = syntheticCells({ overlays: ["none", "lora"] });
  // 3 (model, backend, tier) x 2 overlays x 5 rungs = 30 cells; one is Structurally N/A.
  assert.equal(cells.length, 30);

  const result = producibility(cells, {
    adapterPairs: ["alpha|mlx"],
    pairsWithCompleteRecords: [],
  });
  const byId = Object.fromEntries(result.partition.map((gate) => [gate.id, gate.cells]));

  assert.equal(byId["no-run-needed"], 1);
  // 15 lora cells; none of them is the N/A cell, which is an overlay=none cell.
  assert.equal(byId["overlay-declined"], 15);
  // beta|candle has no adapter: 5 none-overlay rungs, minus the one already charged to N/A.
  assert.equal(byId["no-provider-adapter"], 4);
  // alpha|mlx has an adapter but no complete record: 2 tiers x 5 rungs of overlay=none.
  assert.equal(byId["adapter-gated"], 10);
  assert.equal(byId["producible-today"], 0);

  assert.equal(
    result.partition.reduce((sum, gate) => sum + gate.cells, 0),
    cells.length,
    "the gates must partition the matrix exactly",
  );
  assert.equal(result.behindProviderCode, 29);
  assert.equal(result.producibleToday, 0);
  assert.match(result.headline, /ZERO cells are producible today/);
  assert.match(result.headline, /0 provider pair\(s\) have emitted a `complete` record/);
  assert.match(result.headline, /1 provider pair\(s\) have an adapter at all/);

  // The declared gate list and the computed partition must not drift apart.
  assert.deepEqual(
    result.partition.map((gate) => gate.id),
    PRODUCIBILITY_GATES.map((gate) => gate.id),
  );
});

test("a complete record for a covered pair moves its cells into producible-today", () => {
  const cells = syntheticCells({ overlays: ["none", "lora"] });
  const result = producibility(cells, {
    adapterPairs: ["alpha|mlx"],
    pairsWithCompleteRecords: ["alpha|mlx"],
  });
  const byId = Object.fromEntries(result.partition.map((gate) => [gate.id, gate.cells]));

  assert.equal(byId["adapter-gated"], 0);
  assert.equal(byId["producible-today"], 10);
  assert.equal(result.producibleToday, 10);
  assert.equal(result.behindProviderCode, 19);
  assert.doesNotMatch(result.headline, /ZERO/);
});

test("a complete passed overlay record unblocks only its exact matrix overlay", () => {
  const cells = syntheticCells({ overlays: ["none", "lora"] });
  const result = producibility(cells, {
    adapterPairs: ["alpha|mlx"],
    pairsWithCompleteRecords: ["alpha|mlx"],
    overlayKeysWithCompleteRecords: ["alpha|mlx|lora"],
  });
  const byId = Object.fromEntries(result.partition.map((gate) => [gate.id, gate.cells]));

  assert.equal(byId["overlay-declined"], 5, "beta's uncovered lora cells stay declined");
  assert.equal(byId["producible-today"], 20, "alpha's none and proven lora cells are runnable");
});

test("published cost model distinguishes complete history from runtime-current evidence", async () => {
  const model = await buildCostModel();
  assert.equal(model.completedBaseline.completeRecords, 24);
  assert.equal(
    model.completedBaseline.matrixSummaryCurrentCalibrationRuns,
    0,
    "the Qwen captures remain complete history after the inference runtime pin advances",
  );
  assert.doesNotMatch(model.completedBaseline.note, /Zero calibration records|WHOLE POPULATION/);
  assert.match(model.completedBaseline.note, /24 complete record\(s\) exist/);
  assert.match(model.completedBaseline.note, /0 current calibration run\(s\)/);
  assert.match(model.completedBaseline.note, /Exact records remain narrower/);
  assert.match(model.biggestUncertainties[0].why, /Only 3 of 53 catalog entries/);
  const allProse = JSON.stringify(model);
  assert.doesNotMatch(
    allProse,
    /no adapter has ever emitted|zero cells are producible|Zero calibration records|no overlay handling at all/i,
    "no generated section may retain the pre-promotion zero-baseline claims",
  );
});

/**
 * MUTATION PROOF for the gate ORDER. If `overlay-declined` were checked after `no-provider-adapter`,
 * the overlay bucket would shrink to only the covered pairs and the "59.5% of the matrix is behind
 * overlay code" claim would silently become a much smaller number.
 */
test("mutation: reordering the overlay gate after the adapter gate changes the answer", () => {
  const cells = syntheticCells({ overlays: ["none", "lora"] });
  const adapters = new Set(["alpha|mlx"]);
  const real = producibility(cells, { adapterPairs: [...adapters], pairsWithCompleteRecords: [] });
  const overlayFirst = Object.fromEntries(real.partition.map((gate) => [gate.id, gate.cells]))[
    "overlay-declined"
  ];

  let adapterFirstOverlay = 0;
  for (const cell of cells) {
    if (cell.state === "Structurally N/A") continue;
    if (!adapters.has(`${cell.modelId}|${cell.backend}`)) continue;
    if (cell.overlay !== "none") adapterFirstOverlay += 1;
  }

  assert.equal(overlayFirst, 15);
  assert.equal(adapterFirstOverlay, 10);
  assert.notEqual(
    overlayFirst,
    adapterFirstOverlay,
    "gate order is load-bearing — the overlay bucket must be charged before adapter coverage",
  );
});

/**
 * The overlay axis. Records do NOT collapse. The overlay-amortised LOAD count is still computed,
 * because SC-16072 needs the price tag — but it prices a capability the shipped load contract does
 * not offer, and the verdict must say so rather than calling it "collapsible".
 */
test("overlay multiplies records; the amortised load count is a price tag, not an available collapse", () => {
  const withoutOverlays = collapseToRuns(syntheticCells({ overlays: ["none"] }), DEFAULT_PARAMETERS);
  const withOverlays = collapseToRuns(
    syntheticCells({ overlays: ["none", "lora", "identity"] }),
    DEFAULT_PARAMETERS,
  );

  // Records triple: an overlay cell needs its own record.
  assert.equal(withoutOverlays.certifyingRecords.total, 14);
  assert.equal(withOverlays.certifyingRecords.total, 44);

  // Overlay-keyed loads triple too...
  assert.equal(withoutOverlays.warmSessions.total, 3);
  assert.equal(withOverlays.warmSessions.total, 9);

  // ...but amortising overlay into one load collapses them back to the base count.
  assert.equal(withOverlays.unavailableOverlayLoadPriceTag.sessions.total, 3);
  assert.equal(withOverlays.unavailableOverlayLoadPriceTag.collapseFactor, 3);
  assert.equal(withoutOverlays.unavailableOverlayLoadPriceTag.collapseFactor, 1);
  assert.equal(withOverlays.unavailableOverlayLoadPriceTag.availability, "unavailable");

  const overlayAxis = COLLAPSING_AXES.find((axis) => axis.axis === "overlay");
  assert.equal(
    overlayAxis.verdict,
    "NOT collapsed for records; NOT collapsible for model loads in the shipped contract",
  );
});

/**
 * MAJOR 2, pinned. The load-axis collapse was previously asserted ("DOES collapse", "mostly yes",
 * "legitimate for a LoRA a warm generator can swap") against the evidence in this repository. The
 * decisive argument is not "the feature is missing" but "the feature would perturb the measurement":
 * a resident adapter inflates the `none` baseline this campaign exists to measure. If any of these
 * four legs is dropped from the artifact, the old comfortable reading becomes available again.
 */
test("the overlay load-axis collapse is recorded as unavailable in the shipped load contract", () => {
  assert.equal(OVERLAY_LOAD_CONTRACT.verdict, "unavailable in the shipped load contract");

  // Adapters are fixed at construction, and changing them forces a cold reload by design.
  assert.match(OVERLAY_LOAD_CONTRACT.adaptersAreLoadTime, /resolve_adapters/);
  assert.match(OVERLAY_LOAD_CONTRACT.adaptersAreLoadTime, /LoadSpec::adapters/);

  // The negative claim that makes the collapse impossible rather than merely unimplemented.
  assert.match(OVERLAY_LOAD_CONTRACT.noRuntimeSwapApi, /no detach \/ unload/);

  // Adapters void the measured ladder — the origin of this matrix's 6 Structurally N/A cells.
  assert.match(OVERLAY_LOAD_CONTRACT.adaptersDisableARung, /Structurally N\/A/);

  // THE DECISIVE LEG. Without this the collapse looks like a missing feature rather than a
  // measurement error, which is exactly the misreading that shipped.
  assert.match(OVERLAY_LOAD_CONTRACT.baselinePerturbation, /DECISIVE/);
  assert.match(OVERLAY_LOAD_CONTRACT.baselinePerturbation, /resident/);
  assert.match(OVERLAY_LOAD_CONTRACT.baselinePerturbation, /inflated by the adapter's own bytes/);

  // The column is retained, so the artifact must say why it is retained.
  assert.match(OVERLAY_LOAD_CONTRACT.counterfactualPriceTagDisposition, /column is dropped/i);

  assert.deepEqual(Object.keys(OVERLAY_LOAD_CONTRACT.perOverlayKind), [
    "lora",
    "identity",
    "control",
  ]);
  for (const verdict of Object.values(OVERLAY_LOAD_CONTRACT.perOverlayKind)) {
    assert.equal(verdict.warmAttachDetachWithoutBaseReload, false);
    assert.equal(verdict.noneBaselineComparison.status, "not_applicable");
    assert.equal(verdict.noneBaselineComparison.toleranceBytes, 0);
    assert.equal(verdict.loadSharingAvailable, false);
  }

  const publishedMarkdown = readFileSync(
    new URL("../docs/generated/calibration-cost-model.md", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(publishedMarkdown, /Warm sessions, overlays also amortised/);
  assert.doesNotMatch(publishedMarkdown, /Rungs \+ overlays amortised/);

  // And the axis prose must not reintroduce the collapse claim.
  const overlayAxis = COLLAPSING_AXES.find((axis) => axis.axis === "overlay");
  assert.doesNotMatch(overlayAxis.rule, /DOES collapse/);
  assert.match(overlayAxis.rule, /perturbs the/);
});

/**
 * SC-16059 resolves the rung uncertainty per backend. Candle proves a one-load batch can execute but
 * fails the fresh/reused tolerance gate. MLX remains fresh because its eager and deferred rungs carry
 * distinct calibration identities.
 */
test("the rung axis records both backends' explicit inability to amortize", () => {
  const rung = COLLAPSING_AXES.find((axis) => axis.axis === "rung");

  assert.equal(rung.verdict, "fresh per rung on both measured backends");
  assert.match(rung.factor, /warm floor remains counterfactual/);
  assert.match(rung.rule, /run_batch/);
  assert.match(rung.rule, /exceeded the/);
  assert.match(rung.rule, /eager and deferred calibration fingerprints/);
  assert.match(rung.rule, /No target is priced as batched/);
  assert.match(rung.citation, /run_five_rung_batch/);
  assert.match(rung.citation, /assess_z_image_batch/);
  assert.match(rung.caveat, /256 MiB/);
  assert.match(rung.caveat, /5%/);
});

test("the perRunSeconds remediation derives current completion counts and asks for timing", () => {
  const prod = producibility(syntheticCells({ overlays: ["none", "lora"] }), {
      adapterPairs: ["alpha|mlx"],
      pairsWithCompleteRecords: ["alpha|mlx"],
    });
  const uncertainties = rankUncertainties(prod, { completeRecords: 2 });
  const perRun = uncertainties.find((item) => item.input === "perRunSeconds");
  assert.ok(perRun, "perRunSeconds must still be reported as an uncertainty");

  assert.doesNotMatch(perRun.howToResolve, /One run\s+collapses this uncertainty entirely/);
  assert.match(perRun.why, /2 complete record/);
  assert.match(perRun.why, new RegExp(`${prod.producibleToday} producible cell`));
  assert.match(perRun.howToResolve, /per-scenario timing instrumentation/);
  assert.match(perRun.howToResolve, /now-complete Krea MLX adapter path/);

  // It is a multiplier, not a cell gate, so it must carry no blocking count.
  assert.equal(perRun.gate, undefined);
});

/**
 * MAJOR 1 — CO-BLOCKING, pinned.
 *
 * The ordered partition charges every cell to the FIRST gate that blocks it, which is a partition but
 * not a measure of remediation value. On this fixture 10 of the 15 overlay cells are ALSO blocked by
 * `adapter-gated` and the other 5 by `no-provider-adapter`, so overlay support alone frees NOTHING.
 * The multiset is what makes that visible, and the real matrix has the same shape (3,835 of 3,919
 * overlay cells co-blocked by having no adapter at all).
 */
test("producibility emits blockers-per-cell as an order-independent multiset", () => {
  const cells = syntheticCells({ overlays: ["none", "lora"] });
  const result = producibility(cells, {
    adapterPairs: ["alpha|mlx"],
    pairsWithCompleteRecords: [],
  });

  assert.deepEqual(
    result.blockersPerCell.map((entry) => [entry.description, entry.cells]),
    [
      ["adapter-gated", 10],
      ["adapter-gated + overlay-declined", 10],
      ["no-provider-adapter + overlay-declined", 5],
      ["no-provider-adapter", 4],
      ["(exempt — no run needed)", 1],
    ],
  );

  // The multiset must still account for every cell exactly once, like the partition does.
  assert.equal(
    result.blockersPerCell.reduce((sum, entry) => sum + entry.cells, 0),
    cells.length,
  );

  // Backend splits must survive into the multiset, since the two hardware pools do not substitute.
  const byDescription = Object.fromEntries(result.blockersPerCell.map((entry) => [entry.description, entry]));
  assert.deepEqual(
    { mlx: byDescription["adapter-gated + overlay-declined"].mlx, candle: byDescription["adapter-gated + overlay-declined"].candle },
    { mlx: 10, candle: 0 },
  );
});

test("independently-blocking counts show the overlay gate frees no cell on its own", () => {
  const result = producibility(syntheticCells({ overlays: ["none", "lora"] }), {
    adapterPairs: ["alpha|mlx"],
    pairsWithCompleteRecords: [],
  });
  const byId = Object.fromEntries(result.independentBlocking.map((entry) => [entry.id, entry]));

  // THE LOAD-BEARING NUMBER: the overlay gate touches 15 cells and is the sole blocker of zero.
  assert.equal(byId["overlay-declined"].cellsTouched, 15);
  assert.equal(byId["overlay-declined"].soleBlockerCells, 0);
  assert.equal(byId["overlay-declined"].coBlockedCells, 15);
  assert.equal(byId["overlay-declined"].coBlockedShare, 1);
  assert.deepEqual(byId["overlay-declined"].coBlockedWith, {
    "adapter-gated": 10,
    "no-provider-adapter": 5,
  });

  assert.equal(byId["adapter-gated"].soleBlockerCells, 10);
  assert.equal(byId["no-provider-adapter"].soleBlockerCells, 4);

  // Emitted in descending sole-blocker order, which is the ranking basis.
  assert.deepEqual(result.blockerRanking.ranking, [
    "adapter-gated",
    "no-provider-adapter",
    "overlay-declined",
  ]);

  // The four figures the finding is built from.
  assert.equal(result.blockerRanking.overlayChargedByFirstMatch, 15);
  assert.equal(result.blockerRanking.overlayAlsoBlockedByAdapterCoverage, 15);
  assert.equal(result.blockerRanking.overlayIndependentlyBlocking, 0);
  assert.equal(result.blockerRanking.overlayBindingAfterAdapterCoverage, 10);
});

test("gate-order sensitivity is emitted as numbers, not merely disclosed in prose", () => {
  const result = producibility(syntheticCells({ overlays: ["none", "lora"] }), {
    adapterPairs: ["alpha|mlx"],
    pairsWithCompleteRecords: [],
  });

  assert.deepEqual(result.gateOrderSensitivity.shippedOrder, {
    "adapter-gated": 10,
    "no-provider-adapter": 4,
    "no-run-needed": 1,
    "overlay-declined": 15,
  });
  assert.deepEqual(result.gateOrderSensitivity.adapterCoverageFirst, {
    "adapter-gated": 10,
    "no-provider-adapter": 9,
    "no-run-needed": 1,
    "overlay-declined": 10,
  });

  // Both orders must partition the matrix exactly — that is what makes the comparison fair.
  for (const order of [result.gateOrderSensitivity.shippedOrder, result.gateOrderSensitivity.adapterCoverageFirst]) {
    assert.equal(
      Object.values(order).reduce((sum, count) => sum + count, 0),
      30,
    );
  }
});

/**
 * MAJOR 1 — THE RE-RANKING, pinned.
 *
 * Ranked by cells a gate is the ONLY thing wrong with. The invariant is asserted rather than the
 * literal order, so the rule survives fixture changes; the `no-provider-adapter`-dominant case below
 * is the one that reproduces the real matrix's shape.
 */
test("uncertainties are ranked by independently-blocking cells, so overlay ranks last of the code gates", () => {
  const result = producibility(syntheticCells({ overlays: ["none", "lora"] }), {
    adapterPairs: ["alpha|mlx"],
    pairsWithCompleteRecords: [],
  });
  const ranked = rankUncertainties(result);
  const codeGated = ranked.filter((item) => item.gate);

  // Non-increasing in the ranking basis.
  for (let index = 1; index < codeGated.length; index += 1) {
    assert.ok(
      codeGated[index - 1].independentlyBlockingCells >= codeGated[index].independentlyBlockingCells,
      `code gates must be ordered by independently-blocking cells: ${codeGated[index - 1].gate} then ${codeGated[index].gate}`,
    );
  }

  // Overlay is last among the code gates precisely because it independently blocks nothing.
  assert.equal(codeGated.at(-1).gate, "overlay-declined");
  assert.equal(codeGated.at(-1).independentlyBlockingCells, 0);

  // The overlay entry must carry the demotion and must not reclaim the "largest bucket" framing.
  const overlay = ranked.find((item) => item.gate === "overlay-declined");
  assert.match(overlay.why, /DEMOTED/);
  assert.doesNotMatch(overlay.why, /largest single bucket/);
  assert.match(overlay.howToResolve, /sequenced AFTER catalog adapter coverage/);
});

test("when most entries have no adapter — the real matrix's shape — provider-adapter coverage ranks #1", () => {
  // No adapter anywhere: `no-provider-adapter` becomes the dominant independent blocker, exactly as
  // it is in the generated artifact (2,610 sole-blocked of 6,445 touched).
  const result = producibility(syntheticCells({ overlays: ["none", "lora"] }), {
    adapterPairs: [],
    pairsWithCompleteRecords: [],
  });
  const ranked = rankUncertainties(result);

  assert.equal(ranked[0].gate, "no-provider-adapter");
  assert.equal(ranked[0].independentlyBlockingCells, 14);
  assert.equal(ranked[0].cellsTouched, 29);
  // perRunSeconds sits directly behind the top code gate: it multiplies whatever that gate unlocks.
  assert.equal(ranked[1].input, "perRunSeconds");
  // Overlay still frees nothing on its own.
  assert.equal(
    ranked.find((item) => item.gate === "overlay-declined").independentlyBlockingCells,
    0,
  );
});

/**
 * MUTATION PROOF for the CO-BLOCKING count.
 *
 * The error this guards is treating the first-match bucket as the blocker set — i.e. crediting the
 * overlay gate with cells that would not move if overlay support shipped. Reimplement that mistake
 * and prove it disagrees.
 */
test("mutation: counting the first-match bucket as the blocker set overstates overlay by 15 cells", () => {
  const cells = syntheticCells({ overlays: ["none", "lora"] });
  const result = producibility(cells, {
    adapterPairs: ["alpha|mlx"],
    pairsWithCompleteRecords: [],
  });

  // The perturbed rule: "a cell's blockers are just the gate it was charged to."
  const firstMatchOnly = result.partition.find((gate) => gate.id === "overlay-declined").cells;
  const trueIndependent = result.independentBlocking.find((entry) => entry.id === "overlay-declined")
    .soleBlockerCells;

  assert.equal(firstMatchOnly, 15, "the fixture must actually charge overlay a nonzero bucket");
  assert.notEqual(
    firstMatchOnly,
    trueIndependent,
    "co-blocking must be quantified — the first-match bucket is not the number of cells overlay work frees",
  );
  assert.equal(firstMatchOnly - trueIndependent, 15);

  // And the co-blocked cells must be attributed, not just counted, or the ranking cannot be derived.
  assert.equal(
    result.blockersPerCell
      .filter((entry) => entry.blockers.includes("overlay-declined") && entry.blockers.length > 1)
      .reduce((sum, entry) => sum + entry.cells, 0),
    15,
  );
});

/**
 * MUTATION PROOF for the RE-RANKING.
 *
 * Ranking by first-match bucket size is the shipped mistake this PR corrects: it puts overlay #1.
 * Ranking by independent blocking puts it last. Prove the two orders differ, so a regression to the
 * bucket-size rule cannot pass.
 */
test("mutation: ranking by first-match bucket size puts overlay first and contradicts the shipped ranking", () => {
  const result = producibility(syntheticCells({ overlays: ["none", "lora"] }), {
    adapterPairs: ["alpha|mlx"],
    pairsWithCompleteRecords: [],
  });

  const byBucketSize = result.partition
    .filter((gate) => gate.blockedBy !== "nothing; the epic exempts these" && gate.cells > 0)
    .filter((gate) => gate.id !== "producible-today")
    .sort((left, right) => right.cells - left.cells)
    .map((gate) => gate.id);
  const byIndependentBlocking = result.blockerRanking.ranking;

  // The perturbed rule must actually differ on this fixture, and it must differ AT THE TOP.
  assert.equal(byBucketSize[0], "overlay-declined", "bucket-size ranking must put overlay first");
  assert.notEqual(
    byIndependentBlocking[0],
    byBucketSize[0],
    "the ranking basis is load-bearing — bucket size and independent blocking must disagree on #1",
  );
  assert.equal(byIndependentBlocking.at(-1), "overlay-declined");

  // ...and the shipped uncertainty list must follow the independent-blocking order, not bucket size.
  const ranked = rankUncertainties(result).filter((item) => item.gate);
  assert.notEqual(ranked[0].gate, "overlay-declined");
  assert.deepEqual(ranked.map((item) => item.gate), byIndependentBlocking);
});

/**
 * Minor 5. These three parameters were the only unswept ones in the file, contrary to its own policy,
 * and they move the exact-geometry/fit ratio by ~3x. The DIRECTION of the inversion is what the
 * document concludes from, so that must be shown to hold across the whole grid.
 */
test("the fit ratio is swept over its three previously-fixed parameters and the inversion holds everywhere", () => {
  const cells = syntheticCells();
  const sensitivity = fitSensitivity(cells);

  // 3 geometry points x 2 sharing modes x 3 validation fractions.
  assert.equal(sensitivity.rows.length, 18);
  assert.ok(sensitivity.defaultRow, "the default parameter combination must appear in the grid");
  assert.equal(sensitivity.defaultRow.geometryPointsPerFit, DEFAULT_PARAMETERS.geometryPointsPerFit);
  assert.equal(sensitivity.defaultRow.slopeSharedAcrossTiers, DEFAULT_PARAMETERS.slopeSharedAcrossTiers);

  // The ratio genuinely moves — otherwise the sweep would be decoration.
  const [low, high] = sensitivity.exactGeometryOverFitRange;
  assert.ok(high > low, "the unswept parameters must actually move the ratio");
  assert.ok(high / low > 2, "the movement must be large enough to matter to a planner");

  // ...and the DIRECTION does not: fitting costs more than the per-cell campaign everywhere.
  assert.equal(sensitivity.inversionHoldsEverywhere, true);
  assert.ok(sensitivity.fitOverPerCellRange[0] > 1);
});

/**
 * Minor 6. The justification for `slopeSharedAcrossTiers: false` must not rest on the never-fitted
 * q8/bf16 decode zeros, which is circular. It now rests on absence of evidence.
 */
test("the slope-sharing default is justified on absence of evidence, not on the never-fitted zeros", () => {
  const source = readFileSync(new URL("./calibration-cost-model.mjs", import.meta.url), "utf8");
  const comment = source.slice(
    source.indexOf("// Whether the geometry slope may be shared across tiers"),
    source.indexOf("slopeSharedAcrossTiers: false"),
  );

  assert.match(comment, /ABSENCE OF EVIDENCE/);
  assert.match(comment, /CIRCULAR/);
  // The informative comparison, recorded with the real numbers from the manifest.
  assert.match(comment, /7\.98 \(q4\) \/ 7\.98 \(q8\) \/ 7\.90 \(bf16\)/);
  assert.match(comment, /0\.59 \(q4\) \/ 1\.18 \(q8\) \/ 0\.22 \(bf16\)/);
  assert.match(comment, /unestablished, not disproven/);
});

/**
 * Minor 7. The old comment claimed the arbitrary N/A selection was "arbitrary in a way the output
 * says out loud"; grep of both generated artifacts returned zero hits. It must now actually be
 * emitted, and the per-backend split of a projection must be marked as partly a selection artifact.
 */
test("the N/A projection's selection rule is emitted into the model, not just described in source", () => {
  assert.match(NA_SELECTION_RULE, /ASCENDING CELL ID/);
  assert.match(NA_SELECTION_RULE, /ARBITRARY/);
  assert.match(NA_SELECTION_RULE, /partly an artifact of this rule/);

  const scenarios = naSensitivity(syntheticCells());
  const current = scenarios.find((scenario) => scenario.name === "current");
  const projection = scenarios.find((scenario) => scenario.name === "rung4-all");

  // The `current` row is a fact, so its split is never an artifact.
  assert.equal(current.kind, "fact");
  assert.equal(current.backendSplitIsSelectionArtifact, false);
  assert.deepEqual(current.exemptedByBackend, { mlx: 0, candle: 0 });

  // Every projection row must report both its actual and its proportional split so the gap is visible.
  for (const scenario of scenarios.filter((item) => item.kind !== "fact")) {
    assert.ok(scenario.exemptedByBackend, `${scenario.name} must report its actual split`);
    assert.ok(
      scenario.exemptedByBackendIfProportional,
      `${scenario.name} must report the proportional comparison`,
    );
  }
  assert.ok(projection.exemptedByBackend.mlx + projection.exemptedByBackend.candle > 0);
});

/**
 * Minor 9. The gate's own doc comment claims slopes "fitted from real renders at multiple
 * resolutions", which is false for the two tiers carrying one geometry point each. Recorded in the
 * artifact rather than left as a passing remark.
 */
test("the Krea slope-provenance contradiction is recorded against the tiers it is false for", () => {
  const precedent = kreaFitPrecedent({
    models: [
      {
        id: "krea_2_turbo",
        candle: {
          turboFit: {
            evidenceRecords: [
              { tier: "q4", width: 768, height: 768 },
              { tier: "q4", width: 1024, height: 1024 },
              { tier: "q8", width: 1024, height: 1024 },
            ],
            phaseCurvesByTier: {
              q4: { threeStage: { denoise: { fixedGb: 7.15, perMpxGb: 7.98 }, decode: { fixedGb: 5.04, perMpxGb: 10.03 } } },
              q8: { threeStage: { denoise: { fixedGb: 13.76, perMpxGb: 7.98 }, decode: { fixedGb: 16.62, perMpxGb: 0 } } },
            },
          },
        },
      },
    ],
  });

  const contradiction = precedent.slopeProvenanceContradiction;
  assert.ok(contradiction, "an under-determined tier must produce a recorded contradiction");
  assert.match(contradiction.claim, /fitted from real renders at multiple\s+resolutions/);
  assert.match(contradiction.claim, /vram_gate\.rs:460-463/);
  assert.match(contradiction.reality, /true for q4 only/);
  assert.match(contradiction.reality, /q8/);
  // The nuance matters: the manifest's prose DOES claim both geometries, so this is a comment that
  // over-promises relative to the machine-readable records the gate consumes.
  assert.match(contradiction.nuance, /machine-readable/);
  assert.match(contradiction.nuance, /one geometry cell for q8/);

  // A fully-determined precedent must NOT manufacture a contradiction.
  const determined = kreaFitPrecedent({
    models: [
      {
        id: "krea_2_turbo",
        candle: {
          turboFit: {
            evidenceRecords: [
              { tier: "q4", width: 768, height: 768 },
              { tier: "q4", width: 1024, height: 1024 },
            ],
            phaseCurvesByTier: {
              q4: { threeStage: { denoise: { fixedGb: 7.15, perMpxGb: 7.98 } } },
            },
          },
        },
      },
    ],
  });
  assert.equal(determined.slopeProvenanceContradiction, null);
});

test("fit-versus-exhaustive is sensitive to the geometry points parameter, not hardcoded", () => {
  const cells = syntheticCells();
  const precedent = { measurementRecords: 0, tiers: [], curves: 0, coefficients: 0 };
  const sessionsFor = (overrides) =>
    Object.fromEntries(
      fitVersusExhaustive(cells, precedent, { ...DEFAULT_PARAMETERS, ...overrides }).strategies.map(
        (strategy) => [strategy.name, strategy.sessions],
      ),
    );

  const twoPoints = sessionsFor({ geometryPointsPerFit: 2, validationSampleFraction: 0 });
  const threePoints = sessionsFor({ geometryPointsPerFit: 3, validationSampleFraction: 0 });

  // Exhaustive strategies are geometry-policy facts and must not move with a fit parameter.
  // Session-level, not cell-level: alpha's two session keys advertise 2 resolutions each and
  // beta's advertises 1. Both backends pay five fresh invocations per session point.
  assert.equal(twoPoints["exhaustive-exact-geometry"], 5);
  assert.equal(threePoints["exhaustive-exact-geometry"], 5);
  assert.equal(twoPoints["exhaustive-per-cell"], 3);

  const invocations = Object.fromEntries(
    fitVersusExhaustive(cells, precedent, {
      ...DEFAULT_PARAMETERS,
      geometryPointsPerFit: 2,
      validationSampleFraction: 0,
    }).strategies.map((strategy) => [strategy.name, strategy.providerInvocations]),
  );
  assert.equal(invocations["exhaustive-exact-geometry"], 25);
  assert.equal(invocations["exhaustive-per-cell"], 14);

  // 3 (group, tier) pairs x points.
  assert.equal(twoPoints["fit-then-validate"], 6);
  assert.equal(threePoints["fit-then-validate"], 9);

  // Sharing the slope across tiers costs one extra point per GROUP rather than per tier.
  const shared = sessionsFor({
    geometryPointsPerFit: 2,
    validationSampleFraction: 0,
    slopeSharedAcrossTiers: true,
  });
  assert.equal(shared["fit-then-validate"], 5);
});
