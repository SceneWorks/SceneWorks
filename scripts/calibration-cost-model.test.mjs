import assert from "node:assert/strict";
import test from "node:test";

import {
  COLLAPSING_AXES,
  DEFAULT_PARAMETERS,
  PRODUCIBILITY_GATES,
  cellCensus,
  collapseToRuns,
  fitVersusExhaustive,
  kreaFitPrecedent,
  naSensitivity,
  producibility,
} from "./calibration-cost-model.mjs";

const RUNGS = [
  "resident",
  "staged_residency",
  "bounded_decode",
  "bounded_attention",
  "bounded_transformer_residency",
];

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
  assert.equal(runs.rungCollapseFactor, 5);
  // 14 records over 3 loads — NOT the rung span, because one cell is already exempt. Reported to
  // three places precisely so an exempt cell cannot round away into a clean "5".
  assert.equal(runs.recordsPerWarmSession, 4.667);

  // The declared axes must stay in sync with what the numbers above actually demonstrate.
  const verdicts = Object.fromEntries(COLLAPSING_AXES.map((axis) => [axis.axis, axis.verdict]));
  assert.equal(verdicts.cell, "NOT collapsed");
  assert.equal(verdicts.geometry, "collapsed");
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
 * The overlay axis: records do NOT collapse, model loads DO. Both halves asserted, because the
 * honest answer to "does the 3,925 evaporate?" is "on one axis only".
 */
test("overlay multiplies records but not model loads", () => {
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
  assert.equal(withOverlays.overlayAmortisedSessions.total, 3);
  assert.equal(withOverlays.overlayLoadCollapseFactor, 3);
  assert.equal(withoutOverlays.overlayLoadCollapseFactor, 1);

  const overlayAxis = COLLAPSING_AXES.find((axis) => axis.axis === "overlay");
  assert.equal(overlayAxis.verdict, "not collapsed for records; collapsible for model loads");
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
  // beta's advertises 1, so measuring every advertised resolution is 5 model loads (25 provider
  // invocations once each load's five rungs are counted).
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
