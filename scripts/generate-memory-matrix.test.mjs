import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  CONTROL_LANE_MODELS,
  FAMILY_STORIES,
  MODEL_STORIES,
  SOURCE_PATHS,
  activeCalibrationPlan,
  assertMinimaxH3CalibrationPlan,
  assertCellOwnershipIsBackendScoped,
  assertCharacterizationIsConsistent,
  assertCalibrationPlanTargetsResolvedCoordinates,
  assertMlxStagedCoverageIsStructurallyConsistent,
  assertCellInventoryMatchesCatalog,
  assertPublishedDocumentIsClosed,
  hoistManifestScopes,
  planEntryTargetsCoordinate,
  assertTwinCoverage,
  backendScopes,
  buildMatrix,
  buildStoryBackendScope,
  catalogFamilyBackends,
  isImplemented,
  isPublishableCell,
  memoryCharacterization,
  OUT_OF_MATRIX_CELL_STATES,
  plannedCellIds,
  RUNG4_APPLICABILITIES,
  RUNG4_IMPLEMENTATIONS,
  RUNG4_REQUEST_PEAKS,
  SHARED_RUNG4_PREREQUISITES,
  assertGeneratorSourceDoesNotRestateTheRemovedEdge,
  assertRung4CalibrationsDeclareTheRequiredLoadShape,
  assertRung4PrerequisiteRecordsCoverEveryFamily,
  deriveOutOfMatrixApplicability,
  familyGroup,
  parseRung4ContractPrerequisites,
  parseRung4Survey,
  rung4ContractAdmits,
  familyStory,
  parseVideoEngineIds,
  parseInternalCandleVideoRoutes,
  parseVideoRoutes,
  assertOwnershipRegistriesAreDisjoint,
  assertUnroutedEntriesAreDeclared,
  assertVideoOwnership,
  assertRung4SurveyCoversEveryFamily,
  PENDING_RUNG4_SURVEYS,
  UNION_ONLY_MLX_ROUTES,
  UNROUTED_CATALOG_ENTRIES,
  VIDEO_FAMILY_STORIES,
  measuredGeometryKey,
  mlxRequiredHostBytes,
  observedPeakBytes,
  modelStory,
  providerFor,
  declarationModelForCoordinate,
  stagedResidencyIsAvailable,
  strategyStatus,
} from "./generate-memory-matrix.mjs";
import { logicalCaseId, recordId } from "./memory-calibration-harness.mjs";
import { recordsNeedingDigest } from "./backfill-closure-digests.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";
import { stripInertLines } from "./lib/source-revision.mjs";
import { routedLanes } from "./check-tier-integrity.mjs";

async function memoryContractSource(name) {
  return JSON.parse(
    await readFile(new URL(`../${SOURCE_PATHS[name]}`, import.meta.url), "utf8"),
  );
}

test("control provider resolution uses one exact declaration and rejects ambiguity", () => {
  // Real routes carry sc-18815's per-backend `engineFor`; the scalar stays for the alias checks.
  const route = { engine: "z_image", engineFor: () => "z_image" };
  const legacy = { id: "z_image", candle: {} };
  assert.equal(providerFor(legacy, "candle", "control", route), "z_image");

  const declared = {
    id: "z_image",
    candle: {
      memoryStrategyContract: {
        provider: "z_image",
        implementations: [{
          runtimeProvider: "z_image_control",
          overlays: ["control"],
        }],
      },
    },
  };
  assert.equal(
    providerFor(declared, "candle", "control", route),
    "z_image_control",
  );

  declared.candle.memoryStrategyContract.implementations.push({
    runtimeProvider: "z_image_turbo_control",
    overlays: ["control"],
  });
  assert.throws(
    () => providerFor(declared, "candle", "control", route),
    /ambiguous runtime providers: z_image_control, z_image_turbo_control/,
  );
});

test("route-local alias declarations win only on their exact coordinate", () => {
  const target = {
    id: "z_image_turbo",
    candle: {
      memoryStrategyContract: {
        provider: "z_image_turbo",
        implementations: [{
          rung: "bounded_decode",
          tiers: ["q4"],
          modes: ["edit_image"],
          overlays: ["none"],
        }],
      },
    },
  };
  const alias = {
    id: "z_image_edit",
    candle: {
      memoryStrategyContract: {
        provider: "z_image_turbo",
        implementations: [{
          rung: "bounded_transformer_residency",
          tiers: ["q4"],
          modes: ["edit_image"],
          overlays: ["none"],
        }],
      },
    },
  };
  const input = {
    backend: "candle",
    route: { engine: "z_image_turbo" },
    provider: "z_image_turbo",
    model: alias,
    tier: "q4",
    mode: "edit_image",
    overlay: "none",
    manifestById: new Map([[target.id, target], [alias.id, alias]]),
  };
  assert.equal(
    declarationModelForCoordinate({ ...input, rung: "bounded_transformer_residency" }),
    alias,
  );
  assert.equal(
    declarationModelForCoordinate({ ...input, rung: "bounded_decode" }),
    target,
  );
  assert.equal(
    declarationModelForCoordinate({ ...input, rung: "bounded_transformer_residency", mode: "text_to_image" }),
    target,
  );
});

test("exact declaration composition supplies staged residency without a legacy flag", () => {
  const model = {
    id: "lens",
    candle: {
      memoryStrategyContract: {
        provider: "lens",
        implementations: [{
          rung: "bounded_transformer_residency",
          tiers: ["q4"],
          modes: ["text_to_image"],
          overlays: ["none"],
          engagedRungs: [
            "resident",
            "staged_residency",
            "bounded_decode",
            "bounded_attention",
            "bounded_transformer_residency",
          ],
        }],
      },
    },
  };
  assert.equal(model.candle.supportsSequentialOffload, undefined);
  const input = {
    backend: "candle",
    model,
    route: { engine: "lens" },
    provider: "lens",
    tier: "q4",
    mode: "text_to_image",
    overlay: "none",
    sequentialEngines: new Set(),
    manifestById: new Map([[model.id, model]]),
  };
  assert.equal(stagedResidencyIsAvailable(input), true);
  assert.equal(stagedResidencyIsAvailable({ ...input, provider: "lens_turbo" }), false);
  assert.equal(stagedResidencyIsAvailable({ ...input, mode: "style_variations" }), false);

  const statusInput = {
    backend: "candle",
    rung: "staged_residency",
    route: { engine: "lens", kind: "registry" },
    provider: "lens",
    sequentialEngines: new Set(),
    model,
    tier: "q4",
    mode: "text_to_image",
    overlay: "none",
    rung4Survey: new Map(),
    manifestById: new Map([[model.id, model]]),
    inferenceClosureDigests: new Map(),
  };
  assert.equal(strategyStatus(statusInput).state, "Implemented/unverified");

  const mutated = structuredClone(model);
  mutated.candle.memoryStrategyContract.implementations[0].engagedRungs = ["resident"];
  assert.equal(
    strategyStatus({
      ...statusInput,
      model: mutated,
      manifestById: new Map([[mutated.id, mutated]]),
    }).state,
    "Missing",
  );
});


async function memoryContractPinOverrides(pin) {
  const [mlx, candle] = await Promise.all([
    memoryContractSource("engineCapabilitiesMlx"),
    memoryContractSource("engineCapabilitiesCandle"),
  ]);
  mlx.generatedFrom.inferenceRevision = pin;
  candle.generatedFrom.inferenceRevision = pin;
  return {
    engineCapabilitiesMlx: JSON.stringify(mlx),
    engineCapabilitiesCandle: JSON.stringify(candle),
  };
}

// Line-ending and comment normalisation now lives in `scripts/lib/source-revision.mjs` and is unit
// tested there; these tests cover the same rules end to end, through the real generator.
test("a comment-only manifest edit produces no generated matrix change", async () => {
  const manifestUrl = new URL("../config/manifests/builtin.models.jsonc", import.meta.url);
  const manifest = await readFile(manifestUrl, "utf8");
  const baseline = await buildMatrix({ publish: false });
  const commentOnly = await buildMatrix({
    publish: false,
    sourceOverrides: {
      manifest: `${manifest}\n// SC-16129 regression: provenance-only comment\n`,
    },
  });
  const withoutAnyComments = await buildMatrix({
    publish: false,
    sourceOverrides: {
      // This also removes every comment block introduced by #1977, proving replacements and
      // deletions are inert rather than covering only an appended-comment special case.
      manifest: JSON.stringify(JSON.parse(stripJsoncComments(manifest))),
    },
  });

  assert.deepEqual(commentOnly, baseline);
  assert.deepEqual(withoutAnyComments, baseline);
});

test("runtime-only CUDA telemetry surfaces its measured active-byte peak", () => {
  assert.equal(observedPeakBytes({ observedMemory: { overall: { activeBytes: 1234 } } }), 1234);
  // sc-18864: full phase telemetry publishes the allocator bound, which is what `deviceBytes`
  // carried before it was removed, so every published cell keeps its value.
  assert.equal(
    observedPeakBytes({ observedMemory: { overall: { activeBytes: 1234, allocatorBytes: 5678 } } }),
    5678,
  );
  assert.equal(observedPeakBytes({ observedMemory: { overall: {} } }), null);
});

test("self-stamped manifest matrix revisions do not rotate the source fingerprint", async () => {
  const manifestUrl = new URL("../config/manifests/builtin.models.jsonc", import.meta.url);
  const manifest = await readFile(manifestUrl, "utf8");
  const baseline = await buildMatrix({ publish: false });
  const mutated = await buildMatrix({
    publish: false,
    sourceOverrides: {
      manifest: manifest.replaceAll(
        /"matrixSourceRevision":\s*"source-tree:[0-9a-f]{64}"/g,
        `"matrixSourceRevision": "source-tree:${"f".repeat(64)}"`,
      ),
    },
  });

  assert.equal(
    mutated.generatedFrom.sceneWorksRevision,
    baseline.generatedFrom.sceneWorksRevision,
  );
  assert.equal(
    mutated.generatedFrom.sources.manifest.sha256,
    baseline.generatedFrom.sources.manifest.sha256,
  );
});

// The Rust/TOML half of the same principle (sc-16268). DERIVED from the generator's own
// `SOURCE_PATHS` rather than mirrored, so dropping a source from the fingerprint cannot leave these
// tests green — the tripwire's coverage is the thing under test.
const RUST_SOURCE_PATHS = Object.freeze(
  Object.fromEntries(
    Object.entries(SOURCE_PATHS).filter(
      ([, relative]) => relative.endsWith(".rs") || relative.endsWith(".toml"),
    ),
  ),
);

async function readRustSources() {
  const entries = await Promise.all(
    Object.entries(RUST_SOURCE_PATHS).map(async ([name, relative]) => [
      name,
      await readFile(new URL(`../${relative}`, import.meta.url), "utf8"),
    ]),
  );
  return Object.fromEntries(entries);
}

test("the fingerprint covers every declared source, and the artifact publishes that set", async () => {
  // Pins the tripwire's COVERAGE. Deleting a hash-only source (say `memoryStrategy`) is otherwise
  // invisible: every inertness test still passes while the fingerprint quietly stops watching a file
  // the memory numbers depend on.
  assert.deepEqual(Object.keys(SOURCE_PATHS).sort(), [
    "calibrationEvidence",
    "calibrationPlan",
    "cargo",
    "engineCapabilitiesCandle",
    "engineCapabilitiesMlx",
    "engines",
    "imageRouting",
    // sc-17774: `inferenceCompatibility` left with the flux2-only artifact audit; the per-provider
    // closure digests that replaced it are declared here instead, for every lane.
    "inferenceClosures",
    "instantId",
    "manifest",
    "memoryStrategy",
    "mlxFitGate",
    "routingCandle",
    "routingCatalog",
    "routingMlx",
    // sc-19542: the rung-4 arm admits from these per-provider prerequisite records, so they decide
    // cell state and belong inside the fingerprint like every other deciding source.
    "rung4ContractPrerequisites",
    "rung4Survey",
    // sc-18815: the video lane's route resolvers. Each `*_engine_id` function is where a video
    // model-id -> provider-id mapping actually lives, so the fingerprint watches them for the same
    // reason it watches `engines.rs#MODEL_TABLE` on the image side.
    "videoRouteBernini",
    "videoRouteCandle",
    "videoRouteKreaRealtime",
    "videoRouteLtx",
    "videoRouteScail2",
    "videoRouteSvd",
    "videoRouteWan",
    "vramGate",
  ]);
  assert.deepEqual(Object.keys(RUST_SOURCE_PATHS).sort(), [
    "cargo",
    "engines",
    "imageRouting",
    "instantId",
    "memoryStrategy",
    "mlxFitGate",
    "routingCandle",
    "routingCatalog",
    "routingMlx",
    "videoRouteBernini",
    "videoRouteCandle",
    "videoRouteKreaRealtime",
    "videoRouteLtx",
    "videoRouteScail2",
    "videoRouteSvd",
    "videoRouteWan",
    "vramGate",
  ]);

  const matrix = await buildMatrix({ publish: false });
  assert.deepEqual(
    Object.fromEntries(
      Object.entries(matrix.generatedFrom.sources).map(([name, entry]) => [name, entry.path]),
    ),
    { ...SOURCE_PATHS },
  );
});

test("a comment-only Rust or Cargo edit produces no generated matrix change", async () => {
  const sources = await readRustSources();
  const baseline = await buildMatrix({ publish: false });

  for (const [name, body] of Object.entries(sources)) {
    const marker = name === "cargo" ? "#" : "//";
    const appended = await buildMatrix({
      publish: false,
      sourceOverrides: {
        [name]: `${body}\n${marker} sc-16268 regression: semantically inert comment\n`,
      },
    });
    assert.deepEqual(appended, baseline, `${name}: an appended comment must be inert`);
  }
});

test("the matrix regenerates identically from fully comment-stripped sources", async () => {
  // The load-bearing invariant behind hashing stripped bodies: nothing the generator PARSES may
  // live in a strippable comment. Feeding the stripped sources back in exercises the parsers, not
  // just the hash, so a future comment-anchored regex (there was one — `parseMlxSequentialEngines`)
  // fails here instead of silently letting a semantic change slip past the staleness tripwire.
  const sources = await readRustSources();
  const baseline = await buildMatrix({ publish: false });
  const stripped = await buildMatrix({
    publish: false,
    sourceOverrides: Object.fromEntries(
      Object.entries(sources).map(([name, body]) => [
        name,
        stripInertLines(body, name === "cargo" ? "#" : "//"),
      ]),
    ),
  });

  assert.deepEqual(stripped, baseline);
});

test("a Rust value the generator never parses still rotates the source fingerprint", async () => {
  // The mutation check. `HEADROOM_GB` is not parsed by any of this generator's regexes, yet every
  // MLX staged-residency number depends on it: the fingerprint is a whole-source tripwire, and
  // narrowing it to only-parsed-values would trade noisy-but-safe for quiet-and-stale.
  const sources = await readRustSources();
  const baseline = await buildMatrix({ publish: false });
  const headroom = sources.mlxFitGate.match(/const HEADROOM_GB: f64 = ([0-9.]+);/);
  assert.ok(headroom, "fixture needs the HEADROOM_GB constant in mlx_fit_gate.rs");
  const mutated = await buildMatrix({
    publish: false,
    sourceOverrides: {
      mlxFitGate: sources.mlxFitGate.replace(
        headroom[0],
        `const HEADROOM_GB: f64 = ${Number(headroom[1]) + 0.25};`,
      ),
    },
  });

  assert.notEqual(
    mutated.generatedFrom.sceneWorksRevision,
    baseline.generatedFrom.sceneWorksRevision,
  );
  assert.notEqual(
    mutated.generatedFrom.sources.mlxFitGate.sha256,
    baseline.generatedFrom.sources.mlxFitGate.sha256,
  );
});

test("a commented-out line carrying a string literal still rotates the source fingerprint", async () => {
  // The quote guard. A comment that could feed one of the generator's `"([^"]+)"` parsers is not
  // inert, so it stays hashed even though it is a comment.
  const sources = await readRustSources();
  const baseline = await buildMatrix({ publish: false });
  const quoted = await buildMatrix({
    publish: false,
    sourceOverrides: {
      engines: `${sources.engines}\n// "some_commented_out_id",\n`,
    },
  });

  assert.notEqual(
    quoted.generatedFrom.sceneWorksRevision,
    baseline.generatedFrom.sceneWorksRevision,
  );
});

test("provenance is stamped once on the document, never per row", async () => {
  // sc-16268: the per-row copy was one constant repeated ~7,360 times, which turned every
  // fingerprint rotation into a ~14,700-line rewrite of a file that can only be regenerated.
  const matrix = await buildMatrix({ publish: false });
  assert.equal(matrix.schemaVersion, 8);
  assert.match(matrix.generatedFrom.sceneWorksRevision, /^source-tree:[0-9a-f]{64}$/);
  assert.ok(matrix.cells.length > 1000);
  assert.equal(
    matrix.cells.filter((cell) => "evidenceRevision" in cell).length,
    0,
    "document-scoped provenance must not be duplicated into cells",
  );
  assert.ok(
    !JSON.stringify(matrix.cells).includes(matrix.generatedFrom.sceneWorksRevision),
    "no cell may embed the source revision under any other key either",
  );
});

test("catalog-relative inventory guard rejects whole-scope loss without freezing today's total", async () => {
  const matrix = await buildMatrix({ publish: false });
  const candleCells = matrix.cells.filter((cell) => cell.backend === "candle");
  assert.ok(candleCells.length > matrix.cells.length / 4, "mutation must drop a substantial fraction");
  await assert.rejects(
    buildMatrix({
      publish: false,
      // Preserve the bespoke tier-override scope so its older, narrower guard does not mask the new
      // catalog-wide assertion. Every other Candle scope disappears from the real generator path.
      cellFilter: (cell) => cell.backend !== "candle" || cell.modelId === "instantid_realvisxl",
      sourceOverrides: {
        calibrationEvidence: JSON.stringify({
          schemaVersion: 5,
          harnessVersion: "sceneworks-memory-v5",
          records: [],
        }),
      },
    }),
    /:candle: catalog cross-product expects .* emitted 0/,
  );

  // The expectation moves with a legitimate new catalog scope; no committed 7,360-cell ratchet needs
  // editing. The generator's separate 53-entry and ownership guards still decide whether that scope is
  // part of this epic, while this guard checks only that an accepted scope emits its full cross-product.
  const driftRungs = [
    "resident",
    "staged_residency",
    "bounded_decode",
    "bounded_attention",
    "bounded_transformer_residency",
  ];
  const driftCells = driftRungs.map((rung) => ({
    id: `future:future:mlx:bf16:text_to_image:none:${rung}`,
    modelId: "future",
    backend: "mlx",
  }));
  const driftExpectations = new Map([
    ["future:mlx", { tiers: 1, modes: 1, overlays: 1, rungs: driftRungs.length, cells: driftRungs.length }],
  ]);
  assert.doesNotThrow(() => assertCellInventoryMatchesCatalog(driftCells, driftExpectations));
});

test("inventory guard rejects duplicate and unknown-scope cells", async () => {
  const cells = [
    {
      id: "known:known:mlx:bf16:text_to_image:none:resident",
      modelId: "known",
      backend: "mlx",
    },
  ];
  const expected = new Map([
    ["known:mlx", { tiers: 1, modes: 1, overlays: 1, rungs: 1, cells: 1 }],
  ]);

  assert.throws(
    () => assertCellInventoryMatchesCatalog([...cells, cells[0]], expected),
    /duplicate memory-matrix cell id/,
  );
  assert.throws(
    () =>
      assertCellInventoryMatchesCatalog(
        [
          ...cells,
          {
            id: "other:other:mlx:bf16:text_to_image:none:resident",
            modelId: "other",
            backend: "mlx",
          },
        ],
        expected,
      ),
    /unexpected catalog scope other:mlx/,
  );
});

test("a calibration-relevant manifest value rotates affected fallback fingerprints", async () => {
  const manifestUrl = new URL("../config/manifests/builtin.models.jsonc", import.meta.url);
  const manifest = await readFile(manifestUrl, "utf8");
  const parsed = JSON.parse(stripJsoncComments(manifest));
  const baseline = await buildMatrix({ publish: false });
  const model = parsed.models.find((candidate) =>
    ["mlx", "candle"].some(
      (backend) =>
        candidate[backend]?.vramGbByTier &&
        candidate[backend]?.turboFit?.calibrationFingerprint === undefined,
    ),
  );
  assert.ok(model, "fixture needs a derived-fingerprint model with per-tier memory floors");
  const backend = ["mlx", "candle"].find(
    (candidate) =>
      model[candidate]?.vramGbByTier &&
      model[candidate]?.turboFit?.calibrationFingerprint === undefined,
  );
  const tier = Object.keys(model[backend].vramGbByTier)[0];
  model[backend].vramGbByTier[tier] += 0.01;
  const changed = await buildMatrix({
    publish: false,
    sourceOverrides: { manifest: JSON.stringify(parsed) },
  });

  const baselineFingerprints = new Map(
    baseline.cells.map((cell) => [cell.id, cell.calibrationFingerprint]),
  );
  const affected = changed.cells.filter(
    (cell) =>
      cell.modelId === model.id &&
      cell.backend === backend &&
      cell.tier === tier &&
      cell.calibrationFingerprint !== null,
  );
  assert.ok(affected.length > 0);
  assert.ok(
    affected.every(
      (cell) => cell.calibrationFingerprint !== baselineFingerprints.get(cell.id),
    ),
  );
  assert.ok(
    changed.cells
      .filter(
        (cell) =>
          cell.modelId !== model.id || cell.backend !== backend || cell.tier !== tier,
      )
      .every(
        (cell) => cell.calibrationFingerprint === baselineFingerprints.get(cell.id),
      ),
    "a per-tier floor change must not invalidate unrelated cells",
  );
});

test("Krea bounded-decode and bounded-attention matrix identities match their cumulative measurements", async () => {
  const manifestUrl = new URL("../config/manifests/builtin.models.jsonc", import.meta.url);
  const manifest = JSON.parse(stripJsoncComments(await readFile(manifestUrl, "utf8")));
  const krea = manifest.models.find((model) => model.id === "krea_2_turbo");
  const expected = "krea-turbo-cuda-phase-curves-v1";
  assert.equal(krea.candle.turboFit.calibrationFingerprint, expected);

  for (const tier of ["q4", "q8", "bf16"]) {
    for (const rung of ["tiledVae", "chunkedAttention"]) {
      assert.deepEqual(
        Object.keys(krea.candle.turboFit.phaseCurvesByTier[tier][rung]).sort(),
        ["decode", "denoise", "text"],
        `${tier}.${rung} preserves all three measured phase curves`,
      );
    }
  }

  const matrix = await buildMatrix({ publish: false });
  const affected = matrix.cells.filter(
    (cell) =>
      cell.modelId === "krea_2_turbo" &&
      cell.backend === "candle" &&
      cell.mode === "text_to_image" &&
      cell.overlay === "none" &&
      ["q4", "q8", "bf16"].includes(cell.tier) &&
      ["bounded_decode", "bounded_attention"].includes(cell.rung),
  );
  assert.equal(affected.length, 6);
  assert.ok(
    affected.every(
      (cell) =>
        cell.calibrationFingerprint === expected &&
        cell.state === "Implemented/unverified" &&
        cell.engagedRungs.includes("staged_residency") &&
        cell.engagedRungs.includes(cell.rung) &&
        cell.evidence.currentEnvironmentVerification.length === 0 &&
        cell.evidence.historicalVerification.every(
          (verification) =>
            verification.engagedRungs.includes("staged_residency") &&
            verification.engagedRungs.includes(cell.rung),
        ) &&
        cell.evidence.strategyParameterVerification.every(
          (verification) =>
            verification.engagedRungs.includes("staged_residency") &&
            verification.engagedRungs.includes(cell.rung),
        ),
    ),
    "all six catalog cells must publish the exact cumulative identity while remaining runtime-unverified",
  );

  const characterized = matrix.cells.filter(
    (cell) =>
      cell.modelId === "krea_2_turbo" &&
      cell.backend === "candle" &&
      cell.mode === "text_to_image" &&
      cell.overlay === "none" &&
      ["q4", "q8", "bf16"].includes(cell.tier) &&
      [
        "resident",
        "staged_residency",
        "bounded_decode",
        "bounded_attention",
        "bounded_transformer_residency",
      ].includes(cell.rung),
  );
  assert.equal(characterized.length, 15);
  assert.ok(
    characterized.every(
      (cell) =>
        cell.memoryCharacterization.status === "fitted" &&
        cell.memoryCharacterization.coveredPixelBound === 1024 * 1024 &&
        cell.memoryCharacterization.measuredGeometries.length === 2 &&
        cell.memoryCharacterization.measuredGeometries.includes("768x768") &&
        cell.memoryCharacterization.measuredGeometries.includes("1024x1024"),
    ),
    "all three Krea CUDA tiers and five measured rungs must publish two-point fitted characterization",
  );
  for (const cell of characterized.filter((candidate) => ["q8", "bf16"].includes(candidate.tier))) {
    const lower = cell.evidence.historicalVerification.find(
      (record) => record.geometry === "768x768",
    );
    const upper = cell.evidence.historicalVerification.find(
      (record) => record.geometry === "1024x1024",
    );
    assert.equal(lower.evidenceScope, "phase_fit_only");
    assert.equal(lower.runtimeAdmission, false);
    assert.equal(lower.parity, undefined);
    assert.equal(upper.evidenceScope, "exact_request");
    assert.equal(upper.runtimeAdmission, true);
    assert.equal(upper.parity.result, "passed");
  }
});

test("Qwen MLX static ladder contracts expose every shipped entry and promote only exact evidence", async () => {
  const manifest = JSON.parse(stripJsoncComments(await readFile(
    new URL("../config/manifests/builtin.models.jsonc", import.meta.url),
    "utf8",
  )));
  const matrix = await buildMatrix({ publish: false });
  const qwenEntries = [
    "qwen_image",
    "qwen_image_edit_2511",
    "qwen_image_edit_2511_lightning",
  ];
  for (const modelId of qwenEntries) {
    const implementations = manifest.models.find((model) => model.id === modelId)
      .mlx.memoryStrategyContract.implementations;
    // sc-20246: scoped to the HAND-AUTHORED rows. The engine-derived projection
    // (`scripts/generate-manifest-memory-declarations.mjs`) publishes rung x tier coverage and
    // nothing else — the capability dumps carry no fingerprint, so a projected row has none to
    // carry and asserting one would be asserting a value the generator would have to invent. The
    // exemption cannot hide a hand row that dropped its fingerprint: the second assertion pins every
    // fingerprint-less row to the engine dump as its source.
    const [projected, handAuthored] = [
      implementations.filter((implementation) => implementation.fingerprint === undefined),
      implementations.filter((implementation) => implementation.fingerprint !== undefined),
    ];
    assert.ok(
      handAuthored.every(
        (implementation) =>
          implementation.fingerprint === "qwen-image-mlx-shared-ladder-2026-08-01-v1",
      ),
      `${modelId} must keep load shape separate from the provider content fingerprint`,
    );
    assert.ok(
      projected.every((implementation) =>
        implementation.source?.startsWith("config/engine-capabilities/"),
      ),
      `${modelId} rows without a fingerprint must be engine-projected, not hand-authored`,
    );
  }
  const boundedRungs = [
    "bounded_decode",
    "bounded_attention",
    "bounded_transformer_residency",
  ];
  const cells = matrix.cells.filter(
    (cell) =>
      qwenEntries.includes(cell.modelId) &&
      cell.backend === "mlx" &&
      boundedRungs.includes(cell.rung) &&
      cell.evidence.staticImplementation.some((entry) =>
        entry.source.includes("mlx-gen-qwen-image/src/memory_strategy.rs"),
      ),
  );

  assert.deepEqual([...new Set(cells.map((cell) => cell.modelId))].sort(), qwenEntries);
  assert.ok(cells.length > 0);
  assert.ok(
    cells.every(
      (cell) =>
        cell.strategyParameters.publishedRanges.decodeTileEdges.join(",") ===
          "768,640,512,448,384,320,256" &&
        cell.strategyParameters.publishedRanges.decodeOverlaps.join(",") === "64",
    ),
    "the static production contract must inventory the complete shipped ladder",
  );
  // `cells` is filtered to entries whose static implementation is sourced from the PROVIDER contract
  // (`mlx-gen-qwen-image/src/memory_strategy.rs`). Promotion rewrites that source to the evidence
  // route (`mlx_fit_gate.rs#evidence_admission_route`), so a Verified cell leaves this set by
  // construction — the empty result holds whether or not the evidence is current, and says the
  // static-contract inventory never claims verification on its own.
  assert.deepEqual(
    cells.filter((cell) => cell.state === "Verified").map((cell) => cell.id),
    [],
    "the static production contract must never assert verification by itself",
  );

  // The manifest binding must be re-stamped alongside the records: sc-17774 made currency the
  // provider's compile closure, and a record carrying a LIVE digest cannot match a binding still
  // carrying a superseded one. Overriding only the evidence leaves the pair mismatched, so nothing
  // promotes and this test reads as a scope regression when it is really a stale fixture.
  const onCurrentPin = await buildMatrix({
    publish: false,
    sourceOverrides: {
      calibrationEvidence: await qwenRung4OnCurrentPin(),
      manifest: await currentManifestCalibrationFixture({
        select: (binding) => binding.provider === "qwen_image",
      }),
    },
  });
  const verified = onCurrentPin.cells.filter(
    (cell) =>
      qwenEntries.includes(cell.modelId) &&
      cell.backend === "mlx" &&
      boundedRungs.includes(cell.rung) &&
      cell.state === "Verified",
  );
  // This is the SYNTHETIC current-closure fixture: it re-stamps the shipped records onto whatever
  // closure is live, so it measures the promotion RULE rather than today's shipped currency. The
  // q4/bf16 receipts rejoin here because the pin advance moved them into the re-stamped set; the
  // shipped artifact keeps them historical, which the checked-in assertions below pin separately.
  // Pinned as the SET, not a count: neither admitting a stale sibling nor losing a refreshed
  // binding may read as green.
  assert.deepEqual(
    verified.map((cell) => cell.id).sort(),
    [
      "qwen_image:qwen_image:mlx:bf16:text_to_image:none:bounded_attention",
      "qwen_image:qwen_image:mlx:bf16:text_to_image:none:bounded_decode",
      "qwen_image:qwen_image:mlx:bf16:text_to_image:none:bounded_transformer_residency",
      "qwen_image:qwen_image:mlx:q4:text_to_image:none:bounded_attention",
      "qwen_image:qwen_image:mlx:q4:text_to_image:none:bounded_transformer_residency",
      "qwen_image:qwen_image:mlx:q8:text_to_image:none:bounded_attention",
      "qwen_image:qwen_image:mlx:q8:text_to_image:none:bounded_transformer_residency",
    ],
    "current evidence must promote only its exact entry, tier, mode, overlay, and rung",
  );
  // Exact per-cell counts, because `>= 1` would accept a cell that had silently lost evidence.
  //
  // q8 carries three records per bound rung because the fixture re-stamps the superseded q8 records
  // and also includes SC-18237's production-deferred pair. The new q4/bf16 cells carry one exact
  // current record each. Exact counts keep both facts visible.
  //
  // sc-19721 bumped the inference pin 014134e3 -> 75d66db5, which staled every one of these cells
  // (the closure digest is what currency keys on, and mlx-gen/src/residency.rs moved +272 lines).
  // The re-capture adds exactly ONE current record per cell, so every count below rises by one and
  // no other shape changes. The uniform +1 is the signature of a clean sweep: a partial or
  // duplicated ingest would show a ragged delta here, and the seven-cell SET assertion above still
  // pins which cells may appear at all.
  assert.deepEqual(
    Object.fromEntries(
      verified.map((cell) => [cell.id, cell.evidence.currentEnvironmentVerification.length]),
    ),
    {
      "qwen_image:qwen_image:mlx:bf16:text_to_image:none:bounded_attention": 2,
      "qwen_image:qwen_image:mlx:bf16:text_to_image:none:bounded_decode": 2,
      "qwen_image:qwen_image:mlx:bf16:text_to_image:none:bounded_transformer_residency": 2,
      "qwen_image:qwen_image:mlx:q4:text_to_image:none:bounded_attention": 2,
      "qwen_image:qwen_image:mlx:q4:text_to_image:none:bounded_transformer_residency": 2,
      "qwen_image:qwen_image:mlx:q8:text_to_image:none:bounded_attention": 4,
      "qwen_image:qwen_image:mlx:q8:text_to_image:none:bounded_transformer_residency": 4,
    },
    "every Verified Qwen cell must carry its exact dynamic evidence",
  );
  assert.ok(
    cells
      .filter((cell) => cell.state !== "Verified")
      .every(
        (cell) =>
          cell.state === "Implemented/unverified" &&
          cell.evidence.currentEnvironmentVerification.length === 0,
      ),
    "unmeasured siblings must remain unverified",
  );
  assert.ok(
    cells
      .filter((cell) => cell.rung === "bounded_transformer_residency")
      .every(
        (cell) =>
          cell.strategyParameters.transformerWindowSize === 1 &&
          cell.strategyParameters.transformerWindowComponent === "Dit" &&
          cell.engagedRungs.includes("staged_residency"),
      ),
  );
  assert.equal(
    matrix.cells.some(
      (cell) =>
        cell.modelId === "qwen_image" &&
        cell.provider === "qwen_image_control" &&
        cell.evidence.staticImplementation.length > 0,
    ),
    false,
    "the separate unbounded control provider must not inherit the shared ladder declaration",
  );
});

test("decode geometry receipts stay semantic and never publish calibration ranges", async () => {
  const matrix = await buildMatrix({ publish: false });
  const fingerprint = "800d06acf579a36e604f91955fd6a6852ec70bc39701f7a320f1fdd2bf5ff29d";
  const targetModels = [
    "chroma1_base",
    "chroma1_flash",
    "chroma1_hd",
    "illustrious_xl_v1",
    "illustrious_xl_v2",
    "kolors",
    "realvisxl",
    "realvisxl_lightning",
    "sdxl",
  ];
  const cells = matrix.cells.filter(
    (cell) =>
      targetModels.includes(cell.modelId) &&
      cell.backend === "mlx" &&
      cell.tier === "q4" &&
      cell.mode === "text_to_image" &&
      cell.overlay === "none" &&
      cell.rung === "bounded_decode" &&
      cell.calibrationFingerprint === fingerprint,
  );

  assert.deepEqual([...new Set(cells.map((cell) => cell.modelId))].sort(), targetModels);
  assert.ok(
    cells.every(
      (cell) =>
        cell.state === "Implemented/unverified" &&
        cell.requiresCurrentCalibrationBinding !== true &&
        !Object.hasOwn(cell.strategyParameters.publishedRanges, "decodeGeometryPolicies") &&
        cell.evidence.currentEnvironmentVerification.length === 0,
    ),
    "quality receipts must not create a published range, calibration requirement, or Verified evidence",
  );
});

test("FLUX.2-dev MLX exposes only the captured q4/q8 T2I Resident cells and keeps stale captures historical", async () => {
  const manifest = JSON.parse(stripJsoncComments(await readFile(
    new URL("../config/manifests/builtin.models.jsonc", import.meta.url),
    "utf8",
  )));
  const contract = manifest.models.find((model) => model.id === "flux2_dev")
    .mlx.memoryStrategyContract;
  assert.equal(contract.provider, "flux2_dev");
  assert.equal(contract.exhaustive, true);
  assert.deepEqual(
    contract.implementations.map(({ rung, tiers, modes, overlays, engagedRungs }) => ({
      rung,
      tiers,
      modes,
      overlays,
      engagedRungs,
    })),
    [{
      rung: "resident",
      tiers: ["q4", "q8"],
      modes: ["text_to_image"],
      overlays: ["none"],
      engagedRungs: ["resident"],
    }],
  );

  const shipped = await buildMatrix({ publish: false });
  const shippedCells = shipped.cells.filter(
    (cell) => cell.modelId === "flux2_dev" && cell.backend === "mlx",
  );
  assert.equal(
    shippedCells.length,
    135,
    "the full 3-tier x 3-active-mode x 3-overlay x 5-rung slice must exist after style retirement",
  );
  assert.deepEqual(
    shippedCells.filter((cell) => cell.state !== "Missing").map((cell) => cell.id).sort(),
    [
      "flux2_dev:flux2_dev:mlx:q4:text_to_image:none:resident",
      "flux2_dev:flux2_dev:mlx:q8:text_to_image:none:resident",
    ],
    "BF16 and every sibling mode, overlay, and rung must remain Missing",
  );
  // TWO retained cohorts per cell after the sc-17137 main sync: SC-18218's originals (10831e4ca)
  // and sc-19721's re-captures (75d66db5), each covering the 768 and 1024 geometries — four
  // historical rows per cell. The sc-20523 pin moved the shared gen-core closure past BOTH, so
  // neither cohort is re-stamped current: the superseded captures stay historical until a real
  // re-capture arrives, which is the property this has always been protecting.
  assert.ok(
    shippedCells
      .filter((cell) => cell.state !== "Missing")
      .every(
        (cell) =>
          cell.state === "Implemented/unverified" &&
          cell.calibrationFingerprint === "sc-18218-flux2-dev-t2i-resident-evidence-v1" &&
          cell.evidence.currentEnvironmentVerification.length === 0 &&
          cell.evidence.historicalVerification.length === 4,
      ),
    "the exact 768 and 1024 captures of both cohorts must remain attributable but historical after the shared gen-core closure moved",
  );

  // Promotion is tested against a mechanically re-stamped fixture. The checked-in physical
  // captures are deliberately never re-stamped when inference source changes: doing so would turn
  // provenance into an assertion instead of evidence.
  const onCurrentClosure = await buildMatrix({
    publish: false,
    sourceOverrides: {
      calibrationEvidence: await currentEvidenceFixture({
        select: (record) =>
          record.target.provider === "flux2_dev" &&
          record.calibrationFingerprint === "sc-18218-flux2-dev-t2i-resident-evidence-v1",
      }),
      manifest: await currentManifestCalibrationFixture({
        select: (binding) => binding.provider === "flux2_dev",
      }),
    },
  });
  const currentCells = onCurrentClosure.cells.filter(
    (cell) => cell.modelId === "flux2_dev" && cell.backend === "mlx" && cell.state !== "Missing",
  );
  assert.ok(
    currentCells.every(
      (cell) =>
        cell.state === "Runtime verified" &&
        cell.calibrationFingerprint === "sc-18218-flux2-dev-t2i-resident-evidence-v1" &&
        cell.evidence.currentEnvironmentVerification.length === 4,
    ),
    "each admitted cell must be backed by its exact 768 and 1024 current captures",
  );
});

test("Z-Image MLX static contracts cover every bounded rung through the actual provider", async () => {
  const manifest = JSON.parse(stripJsoncComments(await readFile(
    new URL("../config/manifests/builtin.models.jsonc", import.meta.url),
    "utf8",
  )));
  const matrix = await buildMatrix({ publish: false });
  const zImageIds = ["z_image", "z_image_edit", "z_image_turbo"];
  const boundedRungs = [
    "bounded_decode",
    "bounded_attention",
    "bounded_transformer_residency",
  ];
  const bounded = matrix.cells.filter(
    (cell) =>
      zImageIds.includes(cell.modelId) &&
      cell.backend === "mlx" &&
      boundedRungs.includes(cell.rung),
  );

  assert.ok(bounded.length > 0);
  assert.deepEqual([...new Set(bounded.map((cell) => cell.modelId))].sort(), zImageIds);
  assert.ok(
    bounded.every((cell) => ["Implemented/unverified", "Verified"].includes(cell.state)),
    "a shipped Z-Image MLX rung may not remain Missing once its provider contract is exported",
  );
  assert.ok(
    bounded.every((cell) =>
      cell.calibrationFingerprint.startsWith("z-image-mlx-independent-materialization-v4") &&
      cell.evidence.staticImplementation.some((entry) =>
        entry.source.includes("mlx-gen-z-image/src/memory_strategy.rs") ||
        entry.source.includes("mlx_fit_gate.rs#evidence_admission_route"),
      ),
    ),
    "every bounded cell must resolve through either the pinned provider contract or exact admitted evidence",
  );

  const turboContract = manifest.models.find((model) => model.id === "z_image_turbo")
    .mlx.memoryStrategyContract;
  assert.equal(turboContract.provider, "z_image_turbo");
  assert.ok(
    bounded
      .filter((cell) => cell.modelId === "z_image_edit")
      .every((cell) => cell.resolvedRoute === "z_image_turbo"),
    "the edit catalog entry must inherit the Turbo provider contract, not invent a provider",
  );

  for (const cell of bounded) {
    const ranges = cell.strategyParameters.publishedRanges;
    assert.deepEqual(ranges.decodeTileEdges, [2048, 768, 640, 512, 448, 384, 320, 256]);
    assert.deepEqual(ranges.decodeOverlaps, [256, 64]);
    if (cell.rung !== "bounded_decode") {
      assert.deepEqual(ranges.attentionChunkSizes, [67108864]);
    }
    if (cell.rung === "bounded_transformer_residency") {
      assert.deepEqual(ranges.transformerWindowSizes, [1]);
      assert.deepEqual(ranges.transformerWindowComponents, ["Dit", "TextEncoder", "Both"]);
      assert.deepEqual(cell.engagedRungs, [
        "resident",
        "bounded_decode",
        "bounded_attention",
        "bounded_transformer_residency",
      ]);
    }
  }

  // Shared implementation does not certify the catalog siblings. Only their own exact evidence can
  // lift them out of Implemented/unverified.
  assert.ok(
    bounded
      .filter((cell) => cell.modelId !== "z_image_turbo")
      .every((cell) => cell.state === "Implemented/unverified"),
  );

  const allZImageMlx = matrix.cells.filter(
    (cell) => zImageIds.includes(cell.modelId) && cell.backend === "mlx",
  );
  const verified = allZImageMlx.filter((cell) => cell.state === "Verified");
  // SC-19753's five q4 rungs were captured at the closure live at the time. The epic's inference
  // pin has since advanced past it, so they are an ACCEPTED FLOOR and no longer promote — a pin
  // bump staling calibration records is the fail-closed design, not a re-capture work order. Kept
  // as an exact empty SET rather than a count so a record that silently survives the drift as
  // current still fails here.
  const expectedVerified = [];
  assert.deepEqual(
    verified.map((cell) => cell.id).sort(),
    expectedVerified,
    "no Z-Image rung may verify while its capture closure is superseded",
  );
  assert.deepEqual(
    allZImageMlx
      .filter((cell) => cell.evidence.currentEnvironmentVerification.length > 0)
      .map((cell) => cell.id)
      .sort(),
    expectedVerified,
    "a superseded capture must not survive the closure change as current evidence",
  );
  assert.ok(
    allZImageMlx
      .filter((cell) => cell.state !== "Verified")
      .every((cell) =>
        cell.state === "Implemented/unverified" &&
        cell.evidence.currentEnvironmentVerification.length === 0
      ),
    "unmeasured Z-Image sibling tuples must remain implemented but unverified",
  );
});

test("MLX generated evidence derives the same exact static host boundary as runtime", () => {
  const record = {
    backend: "mlx",
    hardware: {
      memoryBytes: 8 * 1024 ** 3,
      mlxMemoryLimitBytes: 6 * 1024 ** 3,
      wiredLimitBytes: 7 * 1024 ** 3,
    },
    predictedPeakBytes: { overall: 5 * 1024 ** 3 },
    observedMemory: {
      overall: {
        // sc-18864: the non-reclaimable residency, formerly recovered as wired - reclaimable.
        activeBytes: 3 * 1024 ** 3,
      },
    },
  };
  assert.equal(
    mlxRequiredHostBytes(record),
    7_158_278_827,
    "ceil(5 GiB * 8 GiB / (8 - 2) GiB) is the true proportional minimum",
  );
  assert.equal(
    mlxRequiredHostBytes({
      ...record,
      hardware: {
        memoryBytes: 137_438_953_472,
        mlxMemoryLimitBytes: 130_567_005_798,
        wiredLimitBytes: 87_044_670_532,
      },
      predictedPeakBytes: { overall: 46_305_116_160 },
      observedMemory: {
        overall: { activeBytes: 40_203_970_608 },
      },
    }),
    73_113_341_306,
    "the 48 GiB host-specific sum (60.725 GiB) must not be published as a portable minimum",
  );
  assert.equal(
    mlxRequiredHostBytes({
      ...record,
      predictedPeakBytes: { overall: 7 * 1024 ** 3 },
    }),
    9 * 1024 ** 3,
    "a proportional solution above the capture host uses the larger-host absolute reserve",
  );
  assert.equal(mlxRequiredHostBytes({ ...record, backend: "candle" }), null);
  assert.equal(
    mlxRequiredHostBytes({
      ...record,
      hardware: { ...record.hardware, memoryBytes: 0 },
    }),
    null,
    "an invalid zero-byte capture host fails closed instead of dividing by zero",
  );
  assert.equal(
    mlxRequiredHostBytes({
      ...record,
      observedMemory: { overall: { activeBytes: 9 * 1024 ** 3 } },
    }),
    11 * 1024 ** 3,
    "observed non-reclaimable wired peak wins when it exceeds prediction",
  );
});

test("backend scopes preserve the routing oracle's canonical lane order", () => {
  const lanes = new Map([
    ["dual", new Set(["candle", "mlx"])],
    ["mlx_only", new Set(["mlx"])],
  ]);
  assert.deepEqual(backendScopes({ id: "dual" }, lanes), ["mlx", "candle"]);
  assert.deepEqual(backendScopes({ id: "mlx_only" }, lanes), ["mlx"]);
  assert.deepEqual(backendScopes({ id: "unrouted" }, lanes), []);
});

test("backend scopes follow real routing even when manifest tuning blocks are absent", async () => {
  const [manifestBody, routingCatalog, routingCandle, routingMlx] = await Promise.all([
    readFile(new URL("../config/manifests/builtin.models.jsonc", import.meta.url), "utf8"),
    readFile(
      new URL("../crates/sceneworks-core/src/jobs_store/routing/catalog.rs", import.meta.url),
      "utf8",
    ),
    readFile(
      new URL("../crates/sceneworks-core/src/jobs_store/routing/candle.rs", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../crates/sceneworks-core/src/jobs_store/routing/mlx.rs", import.meta.url), "utf8"),
  ]);
  const manifest = JSON.parse(stripJsoncComments(manifestBody));
  const byId = new Map(manifest.models.map((model) => [model.id, model]));
  const lanes = routedLanes({ routingCatalog, routingCandle, routingMlx });

  assert.equal(byId.get("anima_base").candle, undefined, "fixture premise: no candle tuning block");
  assert.deepEqual(backendScopes(byId.get("anima_base"), lanes), ["mlx", "candle"]);
  assert.deepEqual(
    backendScopes({ id: "scail2_14b" }, lanes),
    ["mlx", "candle"],
    "the bespoke Candle SCAIL-2 route is part of the canonical backend predicate",
  );
  assert.deepEqual(
    backendScopes({ id: "krea_realtime_14b" }, lanes),
    ["mlx"],
    "the routing oracle must not collapse to both lanes for every model",
  );
});

// SC-15812: ownership is keyed by backend. The generator used to assign one story pair per MODEL and
// copy it onto every cell, so all 2,260 candle cells in the shipped matrix named MLX-scoped stories
// (15450-15501) and zero Candle twins appeared anywhere. An MLX story cannot be closed from CUDA
// hardware, so that inventory claimed coverage no story would ever provide.
test("ownership lookups resolve per backend and refuse to invent a twin", () => {
  // A dual-backend entry has two distinct owners, one per backend.
  assert.equal(modelStory("boogu_image_turbo", "mlx"), 15475);
  assert.equal(modelStory("boogu_image_turbo", "candle"), 15910);
  assert.equal(familyStory("boogu_image_turbo", "mlx"), 15516);
  assert.equal(familyStory("boogu_image_turbo", "candle"), 15827);

  // Bespoke routes are keyed the same way as registry ones.
  assert.equal(modelStory("instantid_realvisxl", "candle"), 15934);
  assert.equal(familyStory("pulid_flux_dev", "candle"), 15839);

  // SC-17422: these entries had real Candle routes but no manifest candle tuning block, so the old
  // lane oracle hid their Candle cells. Every one now has an independent Candle model owner.
  for (const formerlyHidden of [
    "lens",
    "boogu_image_edit",
    "flux2_klein_9b_kv",
    "flux2_klein_9b_true_v2",
  ]) {
    assert.ok(Number.isInteger(modelStory(formerlyHidden, "candle")));
    assert.ok(Number.isInteger(familyStory(formerlyHidden, "candle")));
  }

  assert.equal(familyStory("chroma1_hd", "candle"), 17410);
  assert.throws(() => modelStory("not_a_model", "mlx"), /no owning model story/);

  // The routed image inventory is dual-backend throughout: 53 model twins across 20 families.
  const candleModels = Object.values(MODEL_STORIES).filter((stories) => stories.candle);
  const candleFamilies = Object.values(FAMILY_STORIES).filter((stories) => stories.candle);
  assert.equal(candleModels.length, 53);
  assert.equal(candleFamilies.length, 20);
});

test("the ownership tables scope every story to exactly one backend", () => {
  // 53 model stories and 20 family stories per backend, all distinct.
  const scope = buildStoryBackendScope();
  assert.equal(scope.size, 146);
  assert.deepEqual(scope.get(15475), { backend: "mlx", role: "model story", owner: "boogu_image_turbo" });
  assert.deepEqual(scope.get(15827), {
    backend: "candle",
    role: "family story",
    owner: "family SC-15516",
  });

  // The per-cell guard below resolves a story id through this map and compares both the backend AND
  // the owner it finds, so the map must answer unambiguously. `scope` is a plain Map, so a repeated id
  // is last-write-wins: the guard would compare every cell naming it against whichever claimant landed
  // last, passing that one's cells and rejecting the other's. Not vacuous — ambiguous, which is worse,
  // because it still looks green. So the table itself fails closed on any repeat.
  assert.throws(
    () => buildStoryBackendScope({ alpha: { mlx: 1, candle: 1 } }, {}),
    /exactly one owner and one backend/,
  );
  assert.throws(
    () => buildStoryBackendScope({ alpha: { mlx: 7 } }, { 7: { mlx: 7 } }),
    /exactly one owner and one backend/,
  );
  assert.throws(() => buildStoryBackendScope({ alpha: { mlx: "15450" } }, {}), /not an integer/);
  // Two models sharing one story would under-count the split, so that is rejected too.
  assert.throws(
    () => buildStoryBackendScope({ alpha: { mlx: 1 }, beta: { mlx: 1 } }, {}),
    /exactly one owner and one backend/,
  );
});

test("a cell may never name a story scoped to the other backend", () => {
  assertCellOwnershipIsBackendScoped([
    {
      id: "boogu_image_turbo:mlx",
      modelId: "boogu_image_turbo",
      backend: "mlx",
      owningModelStory: 15475,
      owningFamilyStory: 15516,
    },
    {
      id: "boogu_image_turbo:candle",
      modelId: "boogu_image_turbo",
      backend: "candle",
      owningModelStory: 15910,
      owningFamilyStory: 15827,
    },
  ]);

  // The exact defect this story fixes: the candle cell carrying the model's MLX story.
  assert.throws(
    () =>
      assertCellOwnershipIsBackendScoped([
        {
          id: "boogu_image_turbo:candle",
          modelId: "boogu_image_turbo",
          backend: "candle",
          owningModelStory: 15475,
          owningFamilyStory: 15827,
        },
      ]),
    /owningModelStory SC-15475 is scoped to mlx, but the cell is candle/,
  );
  // ...and the mirror image, an MLX cell attributed to a Candle twin.
  assert.throws(
    () =>
      assertCellOwnershipIsBackendScoped([
        {
          id: "boogu_image_turbo:mlx",
          modelId: "boogu_image_turbo",
          backend: "mlx",
          owningModelStory: 15475,
          owningFamilyStory: 15827,
        },
      ]),
    /owningFamilyStory SC-15827 is scoped to candle, but the cell is mlx/,
  );
  // An id from outside the tables is a defect too, not an unknown to be tolerated.
  assert.throws(
    () =>
      assertCellOwnershipIsBackendScoped([
        { id: "x:candle", backend: "candle", owningModelStory: 99999, owningFamilyStory: 15827 },
      ]),
    /not a known backend-scoped ownership story/,
  );
  assert.throws(
    () =>
      assertCellOwnershipIsBackendScoped([
        { id: "x:candle", backend: "candle", owningModelStory: null, owningFamilyStory: 15827 },
      ]),
    /not an ownership story id/,
  );
});

// The backend check alone shipped first and was NOT sufficient. Review of sc-15812 mutated the
// generator's own table lookup so `boogu_image_turbo`'s candle cells named SC-15907 — `boogu_image`'s
// Candle twin, correctly scoped to candle, belonging to a different model — and the generator exited 0
// after writing 30 mis-attributed cells. Same failure class as the original defect (a cell credited to
// a story that cannot close it), reached by assignment instead of by backend. The original defect was
// itself an assignment bug, so this is the mutation the guard has to survive.
test("a cell may never name another entry's story, even on the right backend", () => {
  // The reviewer's exact mutation: turbo's candle cells pointed at the base model's Candle twin.
  assert.throws(
    () =>
      assertCellOwnershipIsBackendScoped([
        {
          id: "boogu_image_turbo:candle",
          modelId: "boogu_image_turbo",
          backend: "candle",
          owningModelStory: 15907,
          owningFamilyStory: 15827,
        },
      ]),
    /owningModelStory SC-15907 is the candle model story of boogu_image, but this cell needs the candle model story of boogu_image_turbo/,
  );
  // The same hole on the MLX side, and across families rather than within one: a sibling's story is
  // still a story that cannot close this cell.
  assert.throws(
    () =>
      assertCellOwnershipIsBackendScoped([
        {
          id: "boogu_image_turbo:mlx",
          modelId: "boogu_image_turbo",
          backend: "mlx",
          owningModelStory: 15474,
          owningFamilyStory: 15516,
        },
      ]),
    /owningModelStory SC-15474 is the mlx model story of boogu_image, but this cell needs the mlx model story of boogu_image_turbo/,
  );
  assert.throws(
    () =>
      assertCellOwnershipIsBackendScoped([
        {
          id: "boogu_image_turbo:candle",
          modelId: "boogu_image_turbo",
          backend: "candle",
          owningModelStory: 15910,
          owningFamilyStory: 15813,
        },
      ]),
    /owningFamilyStory SC-15813 is the candle family story of family SC-15509, but this cell needs the candle family story of family SC-15516/,
  );
  // A family story in the model slot is same-backend and same-family, and still wrong: it is the
  // family's story, so closing it would not close this model's Candle work.
  assert.throws(
    () =>
      assertCellOwnershipIsBackendScoped([
        {
          id: "boogu_image_turbo:candle",
          modelId: "boogu_image_turbo",
          backend: "candle",
          owningModelStory: 15827,
          owningFamilyStory: 15827,
        },
      ]),
    /owningModelStory SC-15827 is the candle family story of family SC-15516, but this cell needs the candle model story of boogu_image_turbo/,
  );
});

// The story's original reconcile AC pinned an absolute epic story count ("100 -> ~147"), which was
// already stale twice over when this landed. Twin coverage is asserted RELATIVE to what the catalog
// advertises instead, so filing an unrelated story cannot make it wrong.
test("twin coverage reconciles against the catalog, not an absolute story count", () => {
  const models = Object.entries(MODEL_STORIES).map(([id, stories]) => ({
    id,
    backends: stories.candle ? ["mlx", "candle"] : ["mlx"],
  }));
  assert.deepEqual(assertTwinCoverage(models), { dualModels: 53, dualFamilies: 20 });

  // A dual model with a missing Candle twin must stop generation, not quietly reuse the MLX story.
  assert.throws(
    () => assertTwinCoverage(models, { ...MODEL_STORIES, lens: { mlx: 15462 } }),
    /lens: advertises candle but has no Candle twin/,
  );
  // And an empty Candle twin on an mlx-only entry is equally a defect: it could never be closed.
  assert.throws(
    () =>
      assertTwinCoverage(
        models.map((model) => (model.id === "lens" ? { ...model, backends: ["mlx"] } : model)),
      ),
    /lens: advertises mlx only but carries Candle twin SC-17489/,
  );
  const chromaMlxOnly = models.map((model) =>
    model.id.startsWith("chroma1") ? { ...model, backends: ["mlx"] } : model,
  );
  const chromaModelStories = {
    ...MODEL_STORIES,
    chroma1_hd: { mlx: 15483 },
    chroma1_base: { mlx: 15484 },
    chroma1_flash: { mlx: 15485 },
  };
  assert.throws(
    () => assertTwinCoverage(chromaMlxOnly, chromaModelStories),
    /family SC-15520: owns no dual model but carries Candle twin SC-17410/,
  );
  assert.throws(
    () => assertTwinCoverage(models, MODEL_STORIES, { ...FAMILY_STORIES, 15516: { mlx: 15516 } }),
    /family SC-15516: owns dual models but has no Candle twin/,
  );
  // Two dual models sharing one Candle twin would under-count the split silently.
  assert.throws(
    () => assertTwinCoverage(models, { ...MODEL_STORIES, boogu_image: { mlx: 15474, candle: 15910 } }),
    /53 dual models map onto only 52 distinct Candle model twins/,
  );
});

test("a shipping control lane is declared, not inferred from having been measured (sc-16069)", async () => {
  // sc-18099 slimmed the ARTIFACT to planned-or-evidenced cells, so the committed file no longer
  // carries a cell for every declared-but-unmeasured control coordinate — which is the very thing
  // this test is about. Two sources are read, deliberately:
  //
  //   * the RESOLVED document (`publish: false`) for the state and provider-binding claims, so a
  //     declared lane's cells are still there to assert on. Slimming must not quietly turn this test
  //     into a scan of whatever survived.
  //   * the COMMITTED artifact for the existence claim, through `models[].axes`, because "a shipping
  //     lane must not be invisible in the published matrix" is a property OF the artifact and would
  //     be untested if this test moved off it entirely.
  const matrix = await buildMatrix({ publish: false });
  const published = JSON.parse(
    await readFile(new URL("../docs/generated/memory-matrix.json", import.meta.url), "utf8"),
  );
  const control = matrix.cells.filter((cell) => cell.overlay === "control");

  // The named regression: the MLX Krea control lane SHIPS (mlx-gen-krea registers
  // `krea_2_turbo_control`, image_jobs/krea_control.rs routes it) but has no manifest `mlx.control`
  // measurement block, so keying the overlay off that block gave it ZERO cells — a shipping feature
  // invisible to the matrix that exists to show what is unmeasured.
  const kreaMlx = control.filter((cell) => cell.modelId === "krea_2_turbo" && cell.backend === "mlx");
  assert.ok(
    kreaMlx.length > 0,
    "the shipping MLX Krea control lane must have control cells even with no mlx.control block",
  );
  const kreaBounded = kreaMlx.find(
    (cell) =>
      cell.tier === "q4" &&
      cell.mode === "text_to_image" &&
      cell.rung === "bounded_decode",
  );
  assert.equal(
    kreaBounded?.resolvedRoute,
    "krea_2_turbo_control",
    "Krea control evidence must bind to the distinct production provider",
  );
  // sc-16915 reran both geometries on the current runtime, which is what the previous wording
  // ("until rerun on the new runtime") was waiting for, so current goes 0 -> 2.
  //
  // History goes 2 -> 0, which looks like a loss and is not. The superseded runs are still in the
  // bundle, but they carry the pre-full-ladder fingerprint
  // `krea-control-mlx-v4-q4-pose-bounded-decode-512-64`, and history attaches per cell by the
  // fingerprint the cell now claims. A record measured against a different provider identity is not
  // this cell's history — attaching it would be the same category error as re-dating it onto the
  // new pin.
  // Current has gone 2 -> 0 again, and correctly. sc-17774 made currency the provider's compile
  // closure rather than the pin, and `mlx:krea_2_turbo_control` moved in this pin's window because
  // two commits touched the shared `crates/media/mlx-gen` crate, which is a first-party dependency of
  // every MLX provider's closure. The two runs are unchanged and still in the bundle; they were
  // measured against a closure that is no longer live. Re-capturing them is the Krea calibration
  // story's work. What this test still pins is the ROUTE binding above — that the cell resolves to
  // the distinct `krea_2_turbo_control` provider at all — which is what sc-16069 is actually about.
  assert.equal(
    kreaBounded?.evidence.currentEnvironmentVerification.length,
    0,
    "the Krea control runs were measured against a superseded provider closure",
  );
  // History is the mirror of the line above: the two sc-16915 runs carry this cell's own calibration
  // fingerprint, so when their closure went superseded they moved from `current` into `historical`
  // rather than detaching. The pre-full-ladder runs
  // (`krea-control-mlx-v4-q4-pose-bounded-decode-512-64`) still do not appear here — they were
  // measured against a different provider identity, which is a category apart from a live identity
  // whose closure has merely moved on.
  assert.equal(
    kreaBounded?.evidence.historicalVerification.length,
    2,
    "this cell's own runs become its history once their closure is superseded",
  );

  for (const [modelId, provider] of [
    ["z_image", "z_image_control"],
    ["z_image_turbo", "z_image_turbo_control"],
  ]) {
    const cells = control.filter((cell) => cell.modelId === modelId && cell.backend === "mlx");
    assert.ok(cells.length > 0, `${modelId}/mlx must expose its shipping control lane`);
    assert.ok(
      cells.every((cell) => cell.resolvedRoute === provider),
      `${modelId}/mlx control cells must target the registered ${provider} provider`,
    );
  }

  // Every declared lane is represented on every backend the entry advertises — the declaration is what
  // generates cells now, so a lane can be unmeasured without being invisible. Asserted against BOTH
  // the resolved cells and the published `axes`: the artifact is where a reader looks, and a lane the
  // slim made unreadable there is the sc-16069 defect all over again.
  for (const id of CONTROL_LANE_MODELS) {
    const model = matrix.models.find((entry) => entry.id === id);
    const publishedModel = published.models.find((entry) => entry.id === id);
    assert.ok(model, `${id} must be a catalog entry`);
    assert.ok(publishedModel, `${id} must be a catalog entry in the published artifact`);
    for (const backend of model.backends) {
      assert.ok(
        control.some((cell) => cell.modelId === id && cell.backend === backend),
        `${id}/${backend} ships a control lane, so it must have control cells`,
      );
      assert.ok(
        publishedModel.axes[backend].overlays.includes("control"),
        `${id}/${backend} ships a control lane, so the published axes must show it`,
      );
    }
  }

  // No control cells for anything undeclared: the overlay axis must not grow by accident. Checked on
  // the published axes too, so an undeclared lane cannot appear there either.
  const undeclared = [...new Set(control.map((cell) => cell.modelId))].filter(
    (id) => !CONTROL_LANE_MODELS.includes(id),
  );
  assert.deepEqual(undeclared, [], "control cells exist only for declared lanes");
  assert.deepEqual(
    published.models
      .filter((entry) =>
        Object.values(entry.axes).some((axes) => axes.overlays.includes("control")),
      )
      .map((entry) => entry.id)
      .filter((id) => !CONTROL_LANE_MODELS.includes(id)),
    [],
    "the published axes advertise a control lane only for declared lanes",
  );

  // Declaring a lane must NOT fabricate evidence. The current Z-Image captures measure the base
  // no-overlay provider, so none may attach to the distinct control providers; every declared
  // control lane stays unverified until a current control capture exists.
  //
  // sc-16060: until the promotion producer existed this assertion was green for the trivial reason
  // that NO cell could hold `Verified` — `strategyStatus` never returned it and the cell copied that
  // verbatim. It now asserts something: a declared lane with no measurement stays unverified while a
  // measured one is promoted, and the sc-16060 tests below prove the promotion path is live. What
  // this test forbids is DECLARATION alone producing verification.
  // sc-16915 gave the MLX Krea control lane a current capture and this read `[KREA_CONTROL_CELL]`.
  // It is back to empty because sc-17774 made currency the provider's compile closure and
  // `mlx:krea_2_turbo_control` moved in this pin's window (the shared `crates/media/mlx-gen` crate is
  // a first-party dependency of every MLX provider).
  //
  // On its own that is the vacuous shape the paragraph above warns about — with no cell able to hold
  // `Verified`, "declaration alone never verifies" asserts nothing. It is NOT vacuous overall,
  // because the promotion path is asserted against a promoted FIXTURE in
  // "current evidence promotes a cell to Verified, and historical evidence does not (sc-16060)",
  // which was deliberately rebased off the shipped bundle for exactly this reason: shipped currency
  // is not a stable thing to assert on, and that mutation has already gone inert twice by depending
  // on it. What remains pinned here is the scope claim — that no UNDECLARED lane produces a control
  // cell, checked above — which is what sc-16069 is actually about.
  assert.deepEqual(
    control.filter((cell) => cell.state === "Verified").map((cell) => cell.id),
    [],
    "no declared control lane holds a current capture at this closure",
  );
  const attachedZImageControl = control.filter(
    (cell) =>
      cell.modelId === "z_image" &&
      cell.backend === "candle" &&
      cell.evidence.historicalVerification.length > 0,
  );
  assert.equal(attachedZImageControl.length, 0, "base evidence must not attach to control cells");
});

test("every advertised MLX and Candle control route must be declared (sc-16073)", async () => {
  const { imageRouting } = await readRustSources();

  for (const sourceName of ["WIRED_MLX_POSE_FAMILIES", "WIRED_CANDLE_POSE_FAMILIES"]) {
    const marker = `const ${sourceName}: &[&str] = &[`;
    assert.ok(imageRouting.includes(marker), `fixture needs ${sourceName}`);
    const mutated = imageRouting.replace(marker, `${marker}\n    "lens",`);
    await assert.rejects(
      buildMatrix({ publish: false, sourceOverrides: { imageRouting: mutated } }),
      new RegExp(
        `${sourceName.includes("MLX") ? "mlx" : "candle"} control routes and ` +
          "CONTROL_LANE_MODELS disagree .*advertised but undeclared=lens",
      ),
      `${sourceName}: adding a shipping route without a declaration must fail generation`,
    );
  }
});

// ---------------------------------------------------------------------------
// SC-15969 — the rung-4 applicability survey.
// ---------------------------------------------------------------------------

const SURVEY_URL = new URL("../config/rung4-applicability-survey.json", import.meta.url);

async function surveyFixture() {
  return JSON.parse(await readFile(SURVEY_URL, "utf8"));
}

test("every advertised family/backend has a rung-4 verdict, and it reaches its cells", async () => {
  const matrix = await buildMatrix({ publish: false });
  const rung4 = matrix.cells.filter(
    (cell) => cell.rung === "bounded_transformer_residency",
  );
  assert.ok(rung4.length > 1000);

  // The field rides exactly the rung-4 cells. Asserted in BOTH directions: a rung-4 cell that
  // escaped the survey, and the field drifting onto another rung where a consumer would read a
  // verdict that was never made about it.
  assert.ok(rung4.every((cell) => cell.rung4Survey?.story === 15969));
  assert.equal(
    matrix.cells.filter(
      (cell) => cell.rung !== "bounded_transformer_residency" && "rung4Survey" in cell,
    ).length,
    0,
  );

  // Coverage is over what the CATALOG advertises, not over what the survey happens to list, so a
  // family added without a verdict fails here rather than reporting Missing as though surveyed.
  const advertised = new Set(
    matrix.models.flatMap((model) =>
      model.backends.map((backend) => `${familyGroup(model.id)}:${backend}`),
    ),
  );
  const surveyed = new Set(
    matrix.rung4SurveyRows.map((row) => `${row.familyStory}:${row.backend}`),
  );
  // sc-18815: surveyed + pending PARTITION what the catalog advertises. Neither half may be a
  // superset and nothing may fall between them — a family that is in the universe, has no verdict
  // and is not declared pending is exactly the silent hole this assertion exists to catch, and the
  // generator throws on it. Asserting `surveyed == advertised` outright is no longer possible
  // without freezing the epic's ordering, but asserting the partition is stronger than either half.
  const pending = new Set(
    matrix.summary.rung4Survey.pendingFamilyBackends.map((row) => `${row.family}:${row.backend}`),
  );
  assert.deepEqual([...new Set([...surveyed, ...pending])].sort(), [...advertised].sort());
  assert.equal(surveyed.size + pending.size, advertised.size, "no pair may be both surveyed and pending");
  assert.equal(matrix.summary.rung4Survey.surveyedFamilyBackends, surveyed.size);

  // Every pending pair's cells say so, and every surveyed pair's cells say the opposite. Without
  // this the discriminator could be published as a constant and still satisfy the partition above.
  for (const cell of rung4) {
    const key = `${familyGroup(cell.modelId)}:${cell.backend}`;
    assert.equal(cell.rung4Survey.surveyed, surveyed.has(key), `${cell.id}: surveyed flag disagrees with the rows`);
    if (!cell.rung4Survey.surveyed) {
      assert.equal(cell.state, "Missing", `${cell.id}: an unsurveyed family may make no rung-4 claim`);
      assert.equal(cell.rung4Survey.requestPeak, "unsurveyed");
      assert.equal(cell.rung4Survey.structuralApplicability, null);
      assert.equal(cell.rung4Survey.implementation, null);
      assert.ok(Number.isInteger(cell.rung4Survey.pendingSurveyStory));
    }
  }
});

test("video surveys distinguish an unrouted backend from five Missing rungs (sc-18828)", async () => {
  const survey = await surveyFixture();
  assert.deepEqual(Object.keys(survey.families["krea-realtime"].backends), ["mlx"]);
  assert.equal(survey.families["krea-realtime"].unroutedBackends.candle.owningStory, 18828);

  const undeclared = structuredClone(survey);
  delete undeclared.families["krea-realtime"].unroutedBackends;
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(undeclared) } }),
    /family krea-realtime has no candle route — declare it in unroutedBackends/,
  );

  const stale = structuredClone(survey);
  stale.families.scail2.unroutedBackends = {
    candle: structuredClone(survey.families["krea-realtime"].unroutedBackends.candle),
  };
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(stale) } }),
    /SCAIL-2 unrouted candle: the same backend has both a routed verdict and an unrouted declaration/,
  );

  const wrongOwner = structuredClone(survey);
  wrongOwner.families["krea-realtime"].unroutedBackends.candle.owningStory = 18826;
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(wrongOwner) } }),
    /owningStory sc-18826 is not family krea-realtime's owner sc-18828/,
  );
});

test("Bernini's shared provider relationship adds video truth without changing a published image cell (sc-18827)", async () => {
  const survey = await surveyFixture();
  assert.deepEqual(survey.families["15528"].modalityRelationship.entries, {
    image: ["bernini_image"],
    video: ["bernini"],
  });

  const beforeSurvey = structuredClone(survey);
  const previousMlxVerdict = beforeSurvey.families["15528"].backends.mlx;
  previousMlxVerdict.implementation = "none";
  delete previousMlxVerdict.implementedEntries;
  delete previousMlxVerdict.implementedOverlays;
  delete previousMlxVerdict.strategyParameters;
  const [before, after] = await Promise.all([
    buildMatrix({ sourceOverrides: { rung4Survey: JSON.stringify(beforeSurvey) } }),
    buildMatrix(),
  ]);
  assert.deepEqual(
    after.cells.filter((cell) => cell.modelId === "bernini_image"),
    before.cells.filter((cell) => cell.modelId === "bernini_image"),
    "SC-18827 may reconcile the shared provider row, but it may not move a published image cell",
  );

  const wrongModality = structuredClone(survey);
  wrongModality.families["15528"].modalityRelationship.entries.video = ["bernini_image"];
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(wrongModality) } }),
    /says bernini_image is video, but the admitted catalog does not/,
  );

  const missingVideo = structuredClone(survey);
  delete missingVideo.families["15528"].modalityRelationship.entries.video;
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(missingVideo) } }),
    /entries must name at least one image and one video catalog id/,
  );
});

test("video staged-residency findings preserve selectable versus unconditional truth", async () => {
  const survey = await surveyFixture();
  const findings = (family, backend) =>
    survey.families[family].backends[backend].findings.join("\n");
  assert.match(
    findings("15528", "mlx"),
    /corrected SC-18816 descriptor tuple is `supports_sequential_offload = false`, `unconditionally_engages_staged_residency = true`/,
  );
  assert.match(findings("scail2", "mlx"), /corrected tuple is `\(false, true\)`/);
  assert.match(findings("krea-realtime", "mlx"), /corrected tuple is `\(false, true\)`/);
  assert.match(
    findings("wan-video", "mlx"),
    /`\(supports_sequential_offload, unconditionally_engages_staged_residency\) = \(true, true\)`/,
  );
});

test("every video survey assertion is pinned to source lines", async () => {
  const survey = await surveyFixture();
  const sources = [];
  for (const family of ["15528", "wan-video", "scail2", "krea-realtime", "svd"]) {
    const entry = survey.families[family];
    for (const verdict of Object.values(entry.backends ?? {})) {
      for (const section of ["structural", "evidence"]) {
        sources.push(...(verdict[section] ?? []).map((item) => item.source));
      }
    }
    for (const declaration of Object.values(entry.unroutedBackends ?? {})) {
      sources.push(...declaration.evidence.map((item) => item.source));
    }
    sources.push(...(entry.modalityRelationship?.evidence ?? []).map((item) => item.source));
  }
  assert.ok(sources.length > 0);
  for (const source of sources) {
    assert.match(source, /:\d/, `${source}: video survey evidence must cite exact source lines`);
  }
});

test("SVD is partial on both lanes because its U-Net contains 16 paired nested stacks (sc-18828)", async () => {
  const survey = await surveyFixture();
  for (const backend of ["mlx", "candle"]) {
    const verdict = survey.families.svd.backends[backend];
    assert.equal(verdict.structuralApplicability, "partial", `${backend}: not Structurally N/A`);
    assert.equal(verdict.implementation, "none", `${backend}: no block-window path is wired`);
    assert.deepEqual(
      verdict.blockStacks.map(({ blocks, windowable }) => ({ blocks, windowable })),
      [
        { blocks: "16 × 1 BasicBlock", windowable: true },
        { blocks: "16 × 1 TemporalBlock", windowable: true },
        { blocks: "4 down + 1 mid + 4 up stages", windowable: false },
      ],
      `${backend}: inventory must retain both nested transformer stacks and the non-windowable trunk`,
    );
    assert.ok(
      verdict.structural.some((item) => /3 × 2 \+ 1 \+ 3 × 3 = 16/.test(item.reason)),
      `${backend}: the 16-module count must be source-derived`,
    );
  }

  const matrix = await buildMatrix({ publish: false });
  const rows = matrix.rung4SurveyRows.filter((row) => row.familyStory === "svd");
  assert.deepEqual(rows.map((row) => row.backend).sort(), ["candle", "mlx"]);
  assert.ok(rows.every((row) => row.structuralApplicability === "partial"));
  assert.ok(rows.every((row) => row.implementation === "none"));

  const cells = matrix.cells.filter(
    (cell) => cell.modelId === "svd" && cell.rung === "bounded_transformer_residency",
  );
  assert.equal(cells.length, 4, "two lanes × two overlay coordinates must be resolved");
  assert.ok(cells.every((cell) => cell.state === "Missing"));
  assert.ok(cells.every((cell) => cell.rung4Survey.structuralApplicability === "partial"));
  assert.ok(cells.every((cell) => cell.rung4Survey.implementation === "none"));
});

test("the two rung-4 findings stay separate: structural applicability never implies the peak moved", async () => {
  // The epic's non-negotiable in test form. `partial`/`full` says the architecture CAN be windowed;
  // it must never be readable as evidence that doing so is worth selecting. Only families with
  // measured request-peak evidence may claim `moves`.
  const matrix = await buildMatrix({ publish: false });
  const moves = matrix.rung4SurveyRows.filter((row) => row.requestPeak === "moves");
  assert.deepEqual(
    moves.map((row) => `${row.familyStory}:${row.backend}`).sort(),
    [
      // SC-15524 (Anima), SC-15525 (SDXL + derivatives) and SC-15521 (Kolors) join the measured set
      // with their MLX ladders: Anima moves the request peak 5.229 -> 4.151 GiB at window 1; the SDXL
      // family moves it -6.97% (q4) to -21.40% (bf16) per entry per tier; Kolors moves it -7.21% /
      // -12.72% / -21.37% by tier, and its `TextEncoder`/`Both` scopes move it a further -22.22% /
      // -60.02% at bf16/512 where conditioning carries the peak.
      //
      // SC-15520 (Chroma1) joins with its MLX ladder: rung 4 at `Dit` scope moves the staged request
      // peak 19.2065 -> 14.6932 GiB (-23.50%) on Chroma1-Base q4 at 1024^2, byte-identical output at
      // every cadence in [1, 2, 5, 10]. The measured scope is exactly one cell
      // (chroma1_base/q4/text_to_image/none); the family row reads `moves` through that scope while
      // every sibling entry, tier, mode and overlay stays `unmeasured`.
      "15510:candle", "15510:mlx", "15511:mlx", "15512:candle", "15512:mlx", "15517:candle", "15517:mlx",
      "15519:candle", "15520:mlx", "15521:mlx", "15524:mlx", "15525:mlx",
    ],
  );
  assert.equal(
    matrix.rung4SurveyRows.find((row) => row.familyStory === 15511 && row.backend === "mlx")
      .requestPeak,
    "moves",
  );
  assert.ok(
    matrix.rung4SurveyRows.every(
      (row) => row.requestPeak === "unmeasured" || row.implementation !== "none",
    ),
    "a request-peak verdict can only come from a family that has something to measure",
  );

  // Applicable-but-unmeasured is the common case and must remain expressible: these families are
  // structurally capable and still carry no peak claim at all.
  const capableUnmeasured = matrix.rung4SurveyRows.filter(
    (row) => row.structuralApplicability !== "none" && row.requestPeak === "unmeasured",
  );
  assert.ok(capableUnmeasured.length > 25);
});

test("partial applicability is recorded rather than rounded to Implemented or Structurally N/A", async () => {
  const matrix = await buildMatrix({ publish: false });

  // SDXL is the story's named trap: a U-Net whose lowest-resolution level is a genuine 10-deep
  // transformer stack. Rounding it to Structurally N/A would exempt it from the ladder outright.
  const sdxl = matrix.rung4SurveyRows.find(
    (row) => row.familyStory === 15525 && row.backend === "mlx",
  );
  assert.equal(sdxl.structuralApplicability, "partial");
  const sdxlCell = matrix.cells.find(
    (cell) => cell.modelId === "sdxl" && cell.rung === "bounded_transformer_residency",
  );
  assert.equal(sdxlCell.state, "Missing", "applicable-but-unimplemented is Missing, not N/A");
  // sc-18099: the block-stack inventory is a family-level fact and now rides `rung4SurveyRows`, so
  // it survives an entry whose every rung-4 cell is elided — which is SDXL's case exactly.
  assert.ok(
    sdxl.blockStacks.some(
      (stack) => stack.windowable && /10 per Transformer2D/.test(stack.blocks),
    ),
    "the windowable sub-stack must be named, since that is what makes it partial rather than N/A",
  );
  assert.ok(
    sdxl.blockStacks.some((stack) => !stack.windowable),
    "and so must the remainder that stays resident",
  );

  // Kolors reuses the same U-Net module, so it must reach the same verdict rather than being
  // classified from its family name.
  const kolors = matrix.rung4SurveyRows.find(
    (row) => row.familyStory === 15521 && row.backend === "mlx",
  );
  assert.equal(kolors.structuralApplicability, "partial");
});

test("an implemented family is Implemented/unverified only where the provider actually exposes it", async () => {
  const matrix = await buildMatrix({ publish: false });
  const implemented = matrix.cells.filter(
    (cell) =>
      cell.rung === "bounded_transformer_residency" &&
      cell.state === "Implemented/unverified",
  );
  assert.ok(implemented.length > 0);

  // MLX Z-Image ships rung 4 (SC-15754) and the matrix reported it Missing until this survey.
  const zImage = implemented.filter(
    (cell) => cell.backend === "mlx" && cell.owningFamilyStory === 15510,
  );
  assert.deepEqual(
    [...new Set(zImage.map((cell) => cell.modelId))].sort(),
    ["z_image", "z_image_edit", "z_image_turbo"],
  );
  assert.ok(
    zImage.every(
      (cell) =>
        cell.strategyParameters.transformerWindowSize === 1 &&
        cell.strategyParameters.transformerWindowComponent === "Dit",
    ),
    "the published window size and default component scope travel with the cell",
  );

  // The Candle half lands independently under SC-15815. Base Z-Image's SC-16170 certification adds
  // adapter and strict-control overlays; Turbo/Edit retain their previously narrower plain surface.
  const zImageCandle = matrix.cells.filter(
    (cell) =>
      cell.rung === "bounded_transformer_residency" &&
      ["Implemented/unverified", "Verified"].includes(cell.state) &&
      cell.backend === "candle" &&
      cell.owningFamilyStory === 15815,
  );
  assert.deepEqual(
    [...new Set(zImageCandle.map((cell) => cell.modelId))].sort(),
    ["z_image", "z_image_edit", "z_image_turbo"],
  );
  assert.deepEqual(
    [...new Set(zImageCandle.filter((cell) => cell.modelId === "z_image").map((cell) => cell.overlay))].sort(),
    ["control", "lora", "none"],
  );
  assert.ok(
    zImageCandle
      .filter((cell) => cell.modelId === "z_image")
      .every((cell) => ["control", "lora", "none"].includes(cell.overlay)),
  );
  assert.ok(
    zImageCandle.every(
      (cell) =>
        cell.strategyParameters.transformerWindowSize === 1 &&
        cell.strategyParameters.transformerWindowComponent === "Dit",
      ),
  );
  for (const rung of ["bounded_decode", "bounded_attention"]) {
    const cells = matrix.cells.filter(
      (cell) =>
        cell.backend === "candle" &&
        cell.owningFamilyStory === 15815 &&
        cell.rung === rung,
    );
    assert.ok(cells.length > 0);
    // RESTATED 2026-08-17: the Turbo control cell used to be required Missing, which read the absence
    // of a declaration as the invariant. Engine truth at pin 931366f62 overrides that —
    // `candle-gen-z-image` registers `z_image_turbo_control` as its own `provider_id`/`route_id` with
    // its own `control_contract`, and the candle dump exports all five rungs for it. SC-18460 declared
    // it (`runtimeProvider: "z_image_turbo_control"`), so it is now Implemented by its OWN contract.
    //
    // The anti-leak intent is what mattered and is preserved exactly: base Z-Image's declaration must
    // not reach the Turbo control route. That is now asserted as identity rather than as absence — a
    // Turbo control cell may be Implemented, but ONLY while resolving to `z_image_turbo_control`. If it
    // ever went Implemented under the base `z_image_turbo` identity, that IS the leak, and this reds.
    assert.ok(
      cells.every((cell) =>
        cell.modelId === "z_image"
          ? ["Implemented/unverified", "Verified"].includes(cell.state)
          : cell.overlay === "control"
            ? cell.resolvedRoute === "z_image_turbo_control" &&
              ["Implemented/unverified", "Verified"].includes(cell.state)
            : cell.state === "Implemented/unverified",
      ),
      `${rung} must reach base control without leaking to the Turbo control route`,
    );
  }

  // Lens publishes two exact provider identities rather than a Cartesian family claim: base Q4
  // exposes the full ladder with Both/window=1, while Turbo BF16 retains TextEncoder/window=1.
  // Crossing either entry with the other tier must remain Missing.
  const lens = implemented.filter(
    (cell) => cell.backend === "mlx" && cell.owningFamilyStory === 15512,
  );
  assert.deepEqual(
    [...new Set(lens.map((cell) => cell.modelId))].sort(),
    ["lens", "lens_turbo"],
  );
  assert.ok(lens.length > 0);
  assert.ok(
    lens
      .filter((cell) => cell.modelId === "lens")
      .every(
        (cell) =>
          cell.tier === "q4" &&
          cell.overlay === "none" &&
          cell.strategyParameters.transformerWindowSize === 1 &&
          cell.strategyParameters.transformerWindowComponent === "Both" &&
          cell.rung4Survey.requestPeak === "moves",
      ),
  );
  assert.ok(
    lens
      .filter((cell) => cell.modelId === "lens_turbo")
      .every(
      (cell) =>
        cell.tier === "bf16" &&
        cell.overlay === "none" &&
        cell.strategyParameters.transformerWindowSize === 1 &&
        cell.strategyParameters.transformerWindowComponent === "TextEncoder" &&
        cell.rung4Survey.requestPeak === "moves",
      ),
  );
  assert.equal(
    matrix.cells.filter(
      (cell) =>
        cell.owningFamilyStory === 15512 &&
        cell.rung === "bounded_transformer_residency" &&
        cell.state === "Implemented/unverified" &&
        !(
          (cell.modelId === "lens" && cell.tier === "q4" && cell.overlay === "none") ||
          (cell.modelId === "lens_turbo" &&
            cell.tier === "bf16" &&
            cell.overlay === "none")
        ),
    ).length,
    0,
    "Lens implementation scopes must not create unsupported entry/tier cross-products",
  );
  const lensCells = matrix.cells.filter(
    (cell) =>
      cell.backend === "mlx" &&
      cell.owningFamilyStory === 15512 &&
      cell.rung === "bounded_transformer_residency",
  );
  assert.ok(
    lensCells
      .filter(
        (cell) =>
          cell.modelId === "lens_turbo" && cell.tier === "q4" && cell.overlay === "none",
      )
      .every(
        (cell) =>
          cell.state === "Missing" && cell.rung4Survey.requestPeak === "does-not-move",
      ),
    "Lens-Turbo Q4 carries the measured request non-win and remains Missing",
  );
  assert.ok(
    lensCells
      .filter(
        (cell) =>
          (cell.modelId === "lens" && cell.tier === "bf16") || cell.tier === "q8",
      )
      .every(
        (cell) => cell.state === "Missing" && cell.rung4Survey.requestPeak === "unmeasured",
      ),
    "Lens BF16 and Q8 cells carry no base-provider measurement claim and remain Missing",
  );
  assert.ok(
    lensCells
      .filter((cell) => cell.overlay !== "none")
      .every((cell) => cell.state === "Missing"),
    "Lens adapter overlays remain exactly Missing",
  );
  const candleLens = matrix.cells.filter(
    (cell) =>
      cell.backend === "candle" &&
      ["lens", "lens_turbo"].includes(cell.modelId) &&
      cell.rung === "bounded_transformer_residency",
  );
  assert.ok(candleLens.length > 0);
  assert.ok(
    candleLens
      .filter(
        (cell) =>
          cell.modelId === "lens_turbo" &&
          ["q4", "q8"].includes(cell.tier) &&
          cell.overlay === "none",
      )
      .every(
      (cell) =>
        cell.owningFamilyStory === 15819 &&
        cell.state === "Implemented/unverified" &&
        cell.rung4Survey.implementation === "shared-primitive" &&
        cell.strategyParameters.transformerWindowSize === 1 &&
        cell.strategyParameters.transformerWindowComponent === "Dit",
      ),
    "SC-15819 exposes the exact packed q4/q8 plain Candle Lens-Turbo ladder",
  );
  assert.ok(
    candleLens
      .filter((cell) => cell.tier === "bf16" || cell.overlay !== "none")
      .every((cell) => cell.state === "Missing"),
    "dense and adapter-bearing Candle Lens-Turbo cells remain fail-closed",
  );
  assert.ok(
    candleLens.every(
      (cell) =>
        cell.owningFamilyStory === 15819 &&
        cell.rung4Survey.implementation === "shared-primitive",
    ),
    "the MLX reconciliation must preserve SC-15819's independent Candle ownership",
  );

  // MLX Krea registers the contract on all four base descriptors, so both catalog entries and both
  // modes are covered at every measured tier. Low-rank overlays replay; the distinct pose-control
  // provider remains outside this claim.
  const mlxKrea = implemented.filter(
    (cell) => cell.backend === "mlx" && cell.owningFamilyStory === 15517,
  );
  assert.deepEqual(
    [...new Set(mlxKrea.map((cell) => cell.modelId))].sort(),
    ["krea_2_raw", "krea_2_turbo"],
  );
  assert.deepEqual([...new Set(mlxKrea.map((cell) => cell.mode))].sort(), ["edit_image", "text_to_image"]);
  assert.ok(mlxKrea.every((cell) => ["none", "lora"].includes(cell.overlay)));
  assert.ok(
    mlxKrea.every(
      (cell) =>
        cell.strategyParameters.transformerWindowSize === 1 &&
        cell.rung4Survey.requestPeak === "moves",
    ),
  );
  assert.equal(
    implemented.filter(
      (cell) => cell.backend === "mlx" && cell.owningFamilyStory === 15517 && cell.overlay === "control",
    ).length,
    0,
    "the separately registered Krea pose-control route does not inherit the base DiT claim",
  );

  // Candle Krea remains narrower: its contract is gated on the turbo descriptor id, and its edit
  // modes route to descriptors that do not return it.
  const candleKrea = implemented.filter(
    (cell) => cell.backend === "candle" && cell.owningFamilyStory === 15517,
  );
  assert.ok(candleKrea.every((cell) => cell.modelId === "krea_2_turbo"));
  assert.ok(candleKrea.every((cell) => cell.mode === "text_to_image"));
  assert.equal(
    matrix.cells.filter(
      (cell) =>
        cell.backend === "candle" &&
        cell.modelId === "krea_2_raw" &&
        cell.rung === "bounded_transformer_residency" &&
        cell.state === "Implemented/unverified",
    ).length,
    0,
    "Candle krea_2_raw does not get its sibling's contract",
  );
});

test("Candle PuLID exposes every rung only for its exact identity route", async () => {
  const matrix = await buildMatrix({ publish: false });
  const pulid = matrix.cells.filter(
    (cell) => cell.modelId === "pulid_flux_dev" && cell.backend === "candle",
  );

  assert.deepEqual([...new Set(pulid.map((cell) => cell.tier))].sort(), ["bf16", "q4", "q8"]);
  assert.deepEqual([...new Set(pulid.map((cell) => cell.mode))], ["character_image"]);
  assert.deepEqual(
    [...new Set(pulid.map((cell) => cell.rung))].sort(),
    [
      "bounded_attention",
      "bounded_decode",
      "bounded_transformer_residency",
      "resident",
      "staged_residency",
    ],
  );
  assert.ok(
    pulid.every((cell) =>
      cell.overlay === "identity"
        ? cell.state === "Implemented/unverified"
        : cell.state === "Missing",
    ),
    "base and LoRA cells must not inherit the bespoke identity provider's capability",
  );
});

test("PuLID's closed overlay contract does not redefine legacy Candle resident coverage", async () => {
  const matrix = await buildMatrix({ publish: false });
  const expectedResident = [
    "flux_dev:flux1_dev:candle:q4:text_to_image:lora:resident",
    "qwen_image:qwen_image:candle:q4:text_to_image:control:resident",
    // RESTATED 2026-08-17: this coordinate's provider segment moved from the base `z_image_turbo` to
    // the registered `z_image_turbo_control` when SC-18460 declared the control route's own runtime
    // provider (`candle-gen-z-image` registers it with its own `control_contract`; the candle dump
    // exports all five rungs). The cell and its state are unchanged — only the identity it resolves
    // under. The point of this test, that PuLID's closed identity contract does not redefine somebody
    // else's generic resident fallback, is untouched.
    "z_image_turbo:z_image_turbo_control:candle:q4:text_to_image:control:resident",
  ];
  for (const id of expectedResident) {
    assert.equal(
      matrix.cells.find((cell) => cell.id === id)?.state,
      "Implemented/unverified",
      `${id} keeps its pre-PuLID generic resident fallback`,
    );
  }

  assert.equal(
    matrix.cells.find(
      (cell) =>
        cell.id === "flux_dev:flux1_dev:candle:q4:text_to_image:lora:staged_residency",
    )?.state,
    "Missing",
    "the scope fix must not broaden legacy staged-overlay support",
  );
});

test("Mage Candle bounded decode is structurally exempt rather than advertised or left Missing", async () => {
  const matrix = await buildMatrix({ publish: false });
  const cells = matrix.cells.filter(
    (cell) =>
      cell.modelId.startsWith("mage_flow") &&
      cell.backend === "candle" &&
      cell.rung === "bounded_decode",
  );

  assert.equal(cells.length, 27, "fixture covers every Mage Candle tier and mode coordinate");
  assert.ok(cells.every((cell) => cell.state === "Structurally N/A"));
  assert.ok(
    cells.every((cell) =>
      cell.evidence.structural.some(
        (item) =>
          item.source.endsWith("candle-gen-mage/src/memory_strategy.rs") &&
          /StructurallyNotApplicable/.test(item.reason),
      ),
    ),
    "every exemption must carry the provider architecture evidence that justifies it",
  );
});

test("Candle Krea's Implemented cells report the shared backend that makes them reachable", async () => {
  const matrix = await buildMatrix({ publish: false });
  const source = await surveyFixture();
  const sourceReport = source.families["15517"].backends.candle;
  const krea = matrix.rung4SurveyRows.find(
    (row) => row.familyStory === 15517 && row.backend === "candle",
  );
  const kreaCell = matrix.cells.find(
    (cell) =>
      cell.modelId === "krea_2_turbo" &&
      cell.backend === "candle" &&
      cell.rung === "bounded_transformer_residency" &&
      cell.state === "Implemented/unverified",
  );
  const report = kreaCell.rung4Survey;

  assert.equal(krea.implementation, "shared-primitive");
  assert.equal(report.implementation, "shared-primitive");
  assert.match(krea.summary, /candle_gen::block_window::run_windowed/);
  assert.ok(
    sourceReport.evidence.some(
      (item) =>
        item.source.endsWith("candle-gen/src/block_window.rs") &&
        /BlockWindowBackend/.test(item.reason),
    ),
    "the audit must name the Candle backend that makes the Implemented declaration reachable",
  );

  const serialized = JSON.stringify(sourceReport);
  assert.doesNotMatch(serialized, /provider-local/);
  assert.doesNotMatch(serialized, /does not go through.*BlockWindowBackend/i);
});

// ---------------------------------------------------------------------------
// sc-19542 — the rung-4 arm admits from each provider's OWN declared prerequisite graph.
// ---------------------------------------------------------------------------

const PREREQUISITES_URL = new URL("../config/rung4-contract-prerequisites.json", import.meta.url);

async function prerequisitesFixture() {
  return JSON.parse(await readFile(PREREQUISITES_URL, "utf8"));
}

/** `${group}:${backend}` -> whether that provider appends the rung-1 edge at the pinned revision. */
async function declaredRung1Edges() {
  const records = await prerequisitesFixture();
  const declares = new Map();
  for (const [group, family] of Object.entries(records.families)) {
    for (const [backend, record] of Object.entries(family.backends)) {
      declares.set(
        `${group}:${backend}`,
        record.additionalPrerequisites.some(
          (edge) => edge.kind === "rung" && edge.rung === "staged_residency",
        ),
      );
    }
  }
  return declares;
}

test("rung 4 is refused exactly where the family's OWN prerequisite graph refuses it", async () => {
  // sc-19542. The arm used to apply `stagedResidencyIsAvailable` to every family — a rung-1
  // AVAILABILITY proxy, where rung 4's shared contract rule
  // (`BOUNDED_TRANSFORMER_RESIDENCY_REQUIRES`) names `LoadShape::DeferredMaterialization` and no
  // rung edge at all. The rung-1 edge is real but PER PROVIDER: it is appended through
  // `MemoryProviderContract::additional_prerequisites`, and 21 of the 40 (family, backend) pairs
  // append it while 19 do not.
  //
  // The expectation below is read out of those records rather than listed here, so this test
  // maintains no names and no counts: whichever families declare the edge are the families the
  // rung-1 predicate may hold back.
  //
  // What this one actually grades, stated so it is not over-read: the RECORDS' discrimination
  // against the live catalog. Collapse them so every provider declares the edge — which is the data
  // form of the blanket proxy — and the `exempt` branch empties and this reds. It does NOT catch the
  // code-level regression: the two tests below it do, and were each mutation-checked against a
  // restored blanket call site and against a neutered rung evaluator.
  const matrix = await buildMatrix({ publish: false });
  const declares = await declaredRung1Edges();
  const cellsOf = (modelId, backend, rung) =>
    matrix.cells.filter(
      (cell) => cell.modelId === modelId && cell.backend === backend && cell.rung === rung,
    );

  const lanes = [
    ...new Set(matrix.cells.map((cell) => `${cell.modelId}:${cell.backend}`)),
  ].map((lane) => {
    const [modelId, backend] = lane.split(":");
    return { modelId, backend, key: `${familyGroup(modelId)}:${backend}` };
  });

  let heldBack = 0;
  let exempt = 0;
  for (const { modelId, backend, key } of lanes) {
    const rung1 = cellsOf(modelId, backend, "staged_residency");
    if (!rung1.length || rung1.some((cell) => cell.state !== "Missing")) continue;
    if (declares.get(key)) {
      heldBack += 1;
      assert.ok(
        cellsOf(modelId, backend, "bounded_transformer_residency").every(
          (cell) => cell.state === "Missing",
        ),
        `${modelId}:${backend} appends the rung-1 edge and cannot stage, so rung 4 must be Missing`,
      );
    } else {
      exempt += 1;
    }
  }
  // Both partitions have to be occupied or the assertion above grades nothing — a lane set that is
  // all-declaring would make it a restatement of the old blanket proxy, and an all-exempt one would
  // make its loop body unreachable.
  assert.ok(heldBack > 0, "no lane exercises the held-back branch");
  assert.ok(exempt > 0, "no lane exercises the exempt branch — the gate would be indistinguishable from the blanket proxy");
});

test("the rung-1 predicate reaches exactly the families whose provider declares that edge", async () => {
  // AC5 as a behavioural property rather than a source-text convention, and the direct mutation
  // check for the defect: switch the arm back to a blanket `stagedResidencyIsAvailable` and every
  // record becomes sensitive, so the 19 that declare no edge fail here; drop the predicate entirely
  // and the 21 that do declare one fail. The expectation is each record's own prerequisite list —
  // the gate's own predicate — so there is nothing to keep in step by hand.
  const declares = await declaredRung1Edges();
  const records = await prerequisitesFixture();
  const context = (rung1Available) => ({
    backend: "candle",
    // A Candle lane, because `stagedResidencyIsAvailable` reads its answer off the manifest block
    // there, which makes the two contexts differ in exactly one fact.
    model: { id: "fixture", candle: rung1Available ? { supportsSequentialOffload: true } : {} },
    route: { engine: "fixture" },
    sequentialEngines: new Set(),
    manifestById: new Map(),
  });

  let sensitive = 0;
  let insensitive = 0;
  for (const [group, family] of Object.entries(records.families)) {
    for (const [backend, record] of Object.entries(family.backends)) {
      const key = `${group}:${backend}`;
      const withRung1 = rung4ContractAdmits(record, context(true));
      const withoutRung1 = rung4ContractAdmits(record, context(false));
      assert.equal(
        withRung1 !== withoutRung1,
        declares.get(key),
        `${family.name} (${backend}): the gate's sensitivity to rung-1 availability must match ` +
          "whether this provider appends that edge",
      );
      if (withRung1 !== withoutRung1) sensitive += 1;
      else insensitive += 1;
    }
  }
  assert.ok(sensitive > 0 && insensitive > 0, "both branches must be occupied or this grades nothing");
});

test("a rung-4 implementation claim survives an absent rung 1 exactly when the provider allows it", async () => {
  // The same property end to end, through the real generator. Three lanes, each claiming a rung-4
  // implementation the real provider does not have:
  //
  //   Mage-Flow MLX     appends the rung-1 edge, cannot stage  -> refused
  //   SenseNova U1 MLX  appends none,             cannot stage  -> HONOURED (this is the arm that
  //                                                               reds if the blanket proxy returns)
  //   SANA MLX          appends the rung-1 edge,  can stage     -> honoured (so the refusal above is
  //                                                               about the prerequisite, not about
  //                                                               the claim being ignored)
  const claimImplemented = async (group, entry) => {
    const survey = await surveyFixture();
    const verdict = survey.families[group].backends.mlx;
    verdict.implementation = "shared-primitive";
    verdict.implementedEntries = [entry];
    verdict.strategyParameters = { transformerWindowSize: 1 };
    const sourceOverrides = { rung4Survey: JSON.stringify(survey) };
    const matrix = await buildMatrix({
      publish: false,
      sourceOverrides,
    });
    const of = (rung) =>
      matrix.cells.filter(
        (cell) => cell.modelId === entry && cell.backend === "mlx" && cell.rung === rung,
      );
    return { rung1: of("staged_residency"), rung4: of("bounded_transformer_residency") };
  };
  const declares = await declaredRung1Edges();

  assert.equal(declares.get("15509:mlx"), true, "fixture assumes Mage-Flow MLX appends the edge");
  const heldBack = await claimImplemented("15509", "mage_flow");
  assert.ok(heldBack.rung1.length > 0 && heldBack.rung4.length > 0);
  assert.ok(
    heldBack.rung1.every((cell) => cell.state === "Missing"),
    "fixture assumes Mage-Flow advertises no MLX rung 1",
  );
  assert.ok(
    heldBack.rung4.every((cell) => cell.state === "Missing"),
    "a rung-4 claim must not survive an absent rung 1 where the provider declares that edge",
  );

  assert.equal(declares.get("15513:mlx"), false, "fixture assumes SenseNova MLX appends no edge");
  const exempt = await claimImplemented("15513", "sensenova_u1_8b");
  assert.ok(
    exempt.rung1.every((cell) => cell.state === "Missing"),
    "fixture assumes SenseNova advertises no MLX rung 1 either",
  );
  assert.ok(
    exempt.rung4.every(
      (cell) =>
        cell.state === "Implemented/unverified" &&
        cell.strategyParameters.transformerWindowSize === 1,
    ),
    "a provider that appends no rung-1 edge must not be held back by one — this is the cell the " +
      "blanket proxy got wrong",
  );

  assert.equal(declares.get("15523:mlx"), true, "fixture assumes SANA MLX appends the edge");
  const satisfied = await claimImplemented("15523", "sana_1600m");
  assert.ok(
    satisfied.rung1.every((cell) => cell.state === "Implemented/unverified"),
    "fixture assumes SANA advertises MLX rung 1",
  );
  assert.ok(
    satisfied.rung4.every(
      (cell) =>
        cell.state === "Implemented/unverified" &&
        cell.strategyParameters.transformerWindowSize === 1,
    ),
    "with the prerequisite satisfied the same claim IS honoured — so the refusal above is not vacuous",
  );
});

test("a structurally-N/A rung 1 satisfies the provider's own rung-4 edge vacuously", async () => {
  // `validate_selection` has TWO accepting arms for a `Rung { .. EngagedInSameRequest }` edge, and
  // the gate shipped only the first. inference
  // `crates/contracts/gen-core/src/memory_strategy.rs` at pinned `75d66db5` (`1468-1481` at
  // the `014134e3` this was written against):
  //
  //     if self.engages(selection.strategy, rung) { continue; }
  //     // `StructurallyNotApplicable` satisfies the edge vacuously: it asserts the
  //     // architecture has no such component to shed.
  //     if matches!(self.support(rung), Some(StructurallyNotApplicable { .. })) { continue; }
  //
  // Missing arm 2, a provider that both appends the edge and declares rung 1 structurally N/A is
  // UNDER-admitted: the contract accepts the composition and the matrix reports it Missing.
  //
  // MUTATION: delete the `structurally_not_applicable` line from the `rung` evaluator and the first
  // assertion below reds. Nothing else in the suite moves, which is the point — the arm is
  // unreachable on today's records (`mlx-gen-sensenova` is the only N/A declarer and it appends no
  // edge), so only a test that CONSTRUCTS the combination can grade it.
  const edge = {
    kind: "rung",
    rung: "staged_residency",
    scope: "engaged_in_same_request",
    source: "crates/media/mlx-gen/mlx-gen-fixture/src/memory_strategy.rs",
    conditional: false,
  };
  // A context in which rung 1 is NOT available, so arm 1 cannot be what admits.
  const cannotStage = {
    backend: "candle",
    model: { id: "fixture", candle: {} },
    route: { engine: "fixture" },
    sequentialEngines: new Set(),
    manifestById: new Map(),
  };
  const admits = (stagedResidencySupport) =>
    rung4ContractAdmits(
      { crate: "crates/media/mlx-gen/mlx-gen-fixture", additionalPrerequisites: [edge], stagedResidencySupport },
      cannotStage,
    );

  assert.equal(admits("structurally_not_applicable"), true, "arm 2 must satisfy the edge");
  // Each of the other declarations individually, because "N/A admits" would also hold if the gate
  // simply stopped consulting the record at all.
  assert.equal(admits("implemented"), false);
  assert.equal(admits("missing"), false);
  assert.equal(
    admits(null),
    false,
    "an unreadable declaration must fall through to the engagement term, never admit",
  );
  // ...and arm 1 still admits on its own when rung 1 IS available, so arm 2 is an addition rather
  // than a replacement.
  assert.equal(
    rung4ContractAdmits(
      {
        crate: "crates/media/mlx-gen/mlx-gen-fixture",
        additionalPrerequisites: [edge],
        stagedResidencySupport: "implemented",
      },
      { ...cannotStage, model: { id: "fixture", candle: { supportsSequentialOffload: true } } },
    ),
    true,
  );
  // The declaration this arm exists for is a fact about a REAL provider, not an invented one: the
  // records must still carry exactly the N/A declaration the pinned tree contains.
  const records = await prerequisitesFixture();
  const naLanes = Object.entries(records.families).flatMap(([group, family]) =>
    Object.entries(family.backends)
      .filter(([, record]) => record.stagedResidencySupport === "structurally_not_applicable")
      .map(([backend]) => `${group}:${backend}`),
  );
  assert.ok(
    naLanes.length > 0,
    "no record declares rung 1 structurally N/A, so the extractor half of this arm grades nothing",
  );
  // Stated rather than assumed: those lanes append no rung edge today, so the arm is currently
  // unreachable through `buildMatrix` and the assertions above are the only thing grading it.
  for (const lane of naLanes) {
    const [group, backend] = lane.split(":");
    assert.equal(
      records.families[group].backends[backend].additionalPrerequisites.length,
      0,
      `${lane} now appends an edge, so the N/A arm is reachable end to end — grade it there too`,
    );
  }
});

test("the rung-4 prerequisite records cover every advertised lane, exactly and currently", async () => {
  const matrix = await buildMatrix({ publish: false });
  // The IMAGE half of the universe, matching the fence's own scope: the derivation script walks
  // the 20 image families, and the video lanes sc-18815 admitted fall back to the direct
  // predicate until their provider graphs are derived (sc-17137 main-sync reconciliation — the
  // fence's comment carries the full account).
  const advertised = new Set(
    matrix.models
      .filter((model) => model.modality === "image")
      .flatMap((model) => model.backends.map((backend) => `${familyGroup(model.id)}:${backend}`)),
  );
  const base = await prerequisitesFixture();
  const parse = (mutate) => {
    const records = JSON.parse(JSON.stringify(base));
    mutate(records);
    return parseRung4ContractPrerequisites(JSON.stringify(records), {
      pin: records.inferenceRevision,
    });
  };
  const cover = (mutate) =>
    assertRung4PrerequisiteRecordsCoverEveryFamily(parse(mutate), matrix.models);

  // The clean file parses and covers every advertised lane, so each rejection below is graded
  // against a passing baseline rather than against another failure.
  assert.equal(parse(() => {}).size, advertised.size);
  assert.doesNotThrow(() => cover(() => {}));

  // Each guard mutated individually — a set-wide mutation would prove the set, not the members.
  const anyGroup = Object.keys(base.families)[0];
  assert.throws(
    () =>
      cover((records) => {
        delete records.families[anyGroup].backends.mlx;
      }),
    /has no mlx rung-4 contract-prerequisite record/,
  );
  assert.throws(
    () =>
      cover((records) => {
        records.families["99999"] = {
          name: "invented",
          backends: { mlx: { crate: "x", additionalPrerequisites: [] } },
        };
      }),
    /the record reaches no cell/,
  );
  assert.throws(
    () =>
      parseRung4ContractPrerequisites(JSON.stringify(base), {
        pin: "0".repeat(40),
      }),
    /is keyed to .* but Cargo pins/,
  );
  assert.throws(
    () =>
      parse((records) => {
        records.families[anyGroup].backends.mlx.additionalPrerequisites = [
          { kind: "invented-kind", source: "x" },
        ];
      }),
    /has no evaluator in this gate/,
  );
  assert.throws(
    () =>
      parse((records) => {
        delete records.families[anyGroup].backends.mlx.additionalPrerequisites;
      }),
    /must be an array/,
  );
  assert.throws(
    () =>
      parse((records) => {
        delete records.families[anyGroup].backends.mlx.crate;
      }),
    /must name the inference crate/,
  );
  assert.throws(
    () =>
      parse((records) => {
        records.families[anyGroup].backends.mlx.additionalPrerequisites = [
          { kind: "rung", rung: "staged_residency", scope: "engaged_in_same_request" },
        ];
      }),
    /must cite the provider file/,
  );
  // sc-19542 review: an absent `conditional` is a CLAIM that the construction is unconditional, so
  // it may not be defaulted in. Same for a support spelling this gate has no arm for.
  assert.throws(
    () =>
      parse((records) => {
        records.families[anyGroup].backends.mlx.additionalPrerequisites = [
          {
            kind: "rung",
            rung: "staged_residency",
            scope: "engaged_in_same_request",
            source: "crates/x/src/lib.rs",
          },
        ];
      }),
    /must say whether the construction that appends it is conditional/,
  );
  assert.throws(
    () =>
      parse((records) => {
        records.families[anyGroup].backends.mlx.stagedResidencySupport = "probably-not";
      }),
    /is not a gen-core MemoryStrategySupport this gate knows/,
  );

  // ...and the fence is WIRED, not merely correct. The assertions above call it directly, so they
  // would all stay green if nothing in the generator ran it; this drives the same mutation through
  // `buildMatrix`.
  const dropped = JSON.parse(JSON.stringify(base));
  delete dropped.families[anyGroup].backends.mlx;
  await assert.rejects(
    buildMatrix({
      publish: false,
      sourceOverrides: { rung4ContractPrerequisites: JSON.stringify(dropped) },
    }),
    /rung-4 contract-prerequisite record/,
  );
});

test("rung 4's shared LoadShape edge is graded on every declared rung-4 calibration binding", async () => {
  // The shared half of the prerequisite graph. `rung4ContractAdmits` cannot demote a coordinate on
  // it — a cell has no load shape — but a calibration BINDING does, and today's rung-4 bindings
  // carry `deferred_materialization` while their rungs 0-3 siblings carry `eager_materialization`.
  const manifest = JSON.parse(
    stripJsoncComments(await readFile(new URL(`../${SOURCE_PATHS.manifest}`, import.meta.url), "utf8")),
  );
  const images = manifest.models.filter((model) => model.type === "image");
  const rung4Bindings = images.flatMap((model) =>
    ["mlx", "candle"].flatMap((backend) =>
      (model[backend]?.calibrations ?? []).filter(
        (binding) => binding.rung === "bounded_transformer_residency",
      ),
    ),
  );
  // Non-vacuity, derived: with no rung-4 binding in the catalog the guard would grade nothing and
  // stay green forever.
  assert.ok(rung4Bindings.length > 0, "the catalog declares no rung-4 calibration binding to grade");
  assert.doesNotThrow(() => assertRung4CalibrationsDeclareTheRequiredLoadShape(images));

  // ...and it bites. Mutate ONE binding's load shape to the shape rung 4's shared edge forbids.
  const mutated = JSON.parse(JSON.stringify(images));
  const target = mutated
    .flatMap((model) =>
      ["mlx", "candle"].flatMap((backend) => model[backend]?.calibrations ?? []),
    )
    .find((binding) => binding.rung === "bounded_transformer_residency");
  target.loadShape = "eager_materialization";
  assert.throws(
    () => assertRung4CalibrationsDeclareTheRequiredLoadShape(mutated),
    /rung 4's shared prerequisite requires deferred_materialization/,
  );

  // The required shape is READ OUT OF the gate's constant, not restated in the guard.
  assert.deepEqual(
    SHARED_RUNG4_PREREQUISITES.map((prerequisite) => prerequisite.kind),
    ["load-shape"],
  );

  // ...and it is WIRED. The assertions above call the guard directly and would all stay green if
  // nothing in the generator ran it, so the same mutation is driven through `buildMatrix` — through
  // the JSONC manifest, which is the form the generator actually reads.
  const manifestBody = await readFile(
    new URL(`../${SOURCE_PATHS.manifest}`, import.meta.url),
    "utf8",
  );
  const eager = manifestBody.replace(
    /"loadShape": "deferred_materialization"/,
    '"loadShape": "eager_materialization"',
  );
  assert.notEqual(eager, manifestBody, "the manifest fixture must actually change");
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { manifest: eager } }),
    /rung 4's shared prerequisite requires deferred_materialization/,
  );
});

// ---------------------------------------------------------------------------
// sc-18664 — the corrected prerequisite note, and the families the matrix cannot carry.
// ---------------------------------------------------------------------------

const surveyRejects = async (mutate, pattern) => {
  const survey = await surveyFixture();
  mutate(survey);
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(survey) } }),
    pattern,
  );
};

const h3 = (survey) => survey.outOfMatrixFamilies["17137"];

// The projection the generator's derive-check compares against `memoryCharacterization(...)` — the
// three image-shaped keys, PLUS `coveredFrameBound` whenever the record carries one (sc-18663). It
// is spelled once here because a test that hard-coded three keys would keep passing while the
// generator compared four, which is the shape of the defect sc-18663 fixed.
const declaredCharacterization = (characterization) => ({
  status: characterization.status,
  measuredGeometries: characterization.measuredGeometries,
  coveredPixelBound: characterization.coveredPixelBound,
  ...(Object.hasOwn(characterization, "coveredFrameBound")
    ? { coveredFrameBound: characterization.coveredFrameBound }
    : {}),
});

// One temporal characterization, derived by the shipped helper rather than hand-written, so a test
// fixture can never assert a bound the geometries do not support.
const temporalCharacterization = (record, geometries) => ({
  ...record.memoryCharacterization,
  ...memoryCharacterization(geometries),
});

// The out-of-matrix catalog entries, read through the generator's own parse rather than re-spelled
// here: it is the set `buildMatrix` subtracts from the coordinate universe AND the set the
// calibration-plan guard exempts, and a test that spelled its own could green-light an exemption
// production does not make (sc-18663).
const shippedOutOfMatrixEntries = (body) =>
  parseRung4Survey(body, { familyGroups: familyGroup }).outOfMatrixCatalogEntries;

test("the survey's notes state rung 4's real prerequisite and cannot restate the removed one", async () => {
  const survey = await surveyFixture();
  const notes = survey.notes.join("\n");

  // The positive half. `BOUNDED_TRANSFORMER_RESIDENCY_REQUIRES` names exactly one edge at the pinned
  // revision (gen-core memory_strategy.rs:290-293), and SC-15998 is the story that made it so.
  assert.match(notes, /LoadShape::DeferredMaterialization/);
  assert.match(notes, /SC-15998/);
  assert.doesNotMatch(notes, /rung 4 requires rung 1/i);

  // Each guard mutated on its own, so a green suite is evidence about every one of them rather than
  // about the set. All four run through the real generator, which is what proves the reach.
  await surveyRejects((mutated) => {
    mutated.notes.push("Rung 4 requires rung 1 engaged in the same request.");
  }, /restate the rung-1 prerequisite SC-15998 removed/);
  await surveyRejects((mutated) => {
    mutated.notes.push("The window additionally requires rung 1 engaged in the same request.");
  }, /restate the rung-1 prerequisite SC-15998 removed/);
  await surveyRejects((mutated) => {
    mutated.notes = mutated.notes.map((note) =>
      note.replaceAll("LoadShape::DeferredMaterialization", "a deferred load"),
    );
  }, /must name LoadShape::DeferredMaterialization/);
  await surveyRejects((mutated) => {
    mutated.notes = mutated.notes.map((note) => note.replaceAll("SC-15998", "an earlier story"));
  }, /must name SC-15998/);
});

test("a provider's OWN rung-1 edge stays sayable per family — the ban is on the blanket claim", async () => {
  // The scope of the guard is the point, not an accident. mlx-gen-anima pushes a
  // `BoundedTransformerResidency -> StagedResidency (EngagedInSameRequest)` edge through
  // `additional_prerequisites` when the load is streamable, so Anima's entry saying its window needs
  // rung 1 in the same request is TRUE. A document-wide ban would have forced that correct verdict to
  // be rewritten, which is precisely what sc-18664 was told not to do.
  const survey = await surveyFixture();
  const anima = survey.families["15524"].backends.mlx;
  assert.match(anima.summary, /rung 1 engaged in the same request/);
  await buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(survey) } });
});

test("MiniMax-H3 is surveyed per stack, and the family verdict is derived from those stacks", async () => {
  const survey = await surveyFixture();
  const record = h3(survey);
  assert.deepEqual(record.catalogEntries, ["minimax_h3", "minimax_h3_ref"]);
  assert.throws(() => familyGroup("minimax_h3"), /no family story/);

  for (const backend of ["mlx", "candle"]) {
    const verdict = record.backends[backend];
    // `partial` is the honest answer and it is COMPUTED: three-plus separately-indexed windowable
    // stacks needing a plan each, and a conv remainder that cannot be windowed at all.
    assert.equal(verdict.structuralApplicability, "partial");
    assert.equal(deriveOutOfMatrixApplicability(verdict.stacks), "partial");
    // sc-18662 landed the MLX arm on the shared `gen_core::block_window` driver, so the two backends
    // no longer agree here. Asserted per backend rather than loosened to "whatever it says": a
    // backend-local reimplementation is the review failure SC-15792 records, so `shared-primitive`
    // is the specific value that must hold once MLX claims an implementation at all.
    assert.equal(
      verdict.implementation,
      backend === "mlx" ? "shared-primitive" : "none",
    );
    if (backend === "mlx") {
      // sc-18662 closed this axis on MLX: `generate_impl` now routes a whole render through the
      // deferred loaders when the request selects the rung, and one streamed t2va request was
      // measured end to end — 5.80 GB against the resident request's 53.07 GB conditioning mark,
      // decode-bound. `moves` is the specific value that must hold, and the reason must carry the
      // measured figure and its scope (the q4/q4 cell), not restate the phase-arm cells.
      assert.equal(verdict.requestPeak.finding, "moves");
      assert.match(verdict.requestPeak.reason, /5\.80 GB/);
      assert.match(verdict.requestPeak.reason, /RUNG4_REQUEST_PEAK_Q4_BYTES/);
      assert.ok(
        verdict.requestPeak.evidence?.some((entry) =>
          entry.source.includes("streamed_generate_real.rs"),
        ),
        "the requestPeak verdict must cite the end-to-end streamed harness",
      );
      // sc-18662 landed both deferred loaders, so the shape rung 4 requires is now satisfiable and
      // the contract implements it on a `DeferredMaterialization` load.
      assert.equal(verdict.contractSupport, "implemented");
      assert.match(verdict.contractReason, /DeferredMaterialization/);
      assert.match(verdict.contractReason, /transformer_window_sizes: \[1\]/);
      // `Both`, not `Dit` — the encoder arm is the larger absolute saving and AC3 refuses a
      // DiT-only result.
      assert.match(verdict.contractReason, /Both/);
    } else {
      // Candle's REQUEST axis stays `unmeasured`: the rung-3 lesson is that a measurement does not
      // transfer across backends, and candle has no deferred loader to route a request through.
      assert.equal(verdict.requestPeak.finding, "unmeasured");
      // Candle is untouched: the rung-3 lesson is that a verdict does not transfer across backends,
      // and neither does an implementation.
      assert.equal(verdict.contractSupport, "missing");
      assert.match(verdict.contractReason, /EagerMaterialization/);
    }
    assert.deepEqual(verdict.nonWindowableStacks, ["vae.encoder", "audio_vae.decode"]);
    assert.ok(verdict.stacks.every((stack) => stack.reason));
  }

  const mlx = record.backends.mlx;
  // The DiT trunk and the conditioning tower are separate stacks with separate answers, which is the
  // whole reason a single family-level verdict would have been wrong in one direction or the other.
  const byId = new Map(mlx.stacks.map((stack) => [stack.id, stack]));
  assert.equal(byId.get("transformer.blocks").structuralApplicability, "full");
  assert.equal(byId.get("text_encoder.language_model").structuralApplicability, "full");
  assert.equal(byId.get("vae.decoder").structuralApplicability, "full");
  assert.equal(byId.get("audio_vae.decode").windowable, false);
  assert.ok(byId.get("vae.encoder").structural.length > 0);
  // The candle crate ships no text encoder (sc-17156), so its stack list must not inherit one.
  assert.ok(record.backends.candle.stacks.every((stack) => !stack.id.startsWith("text_encoder.")));

  // The retraction rides the row rather than living only in a story comment.
  assert.match(mlx.requestPeak.reason, /RETRACTED/);
});

test("the MiniMax-H3 record is validated on every generation, not merely stored", async () => {
  // Each guard mutated ALONE. A record nothing reads is not a delivered record, so every one of
  // these goes through `buildMatrix` — the same path `--check` and the pre-push hook take.
  await surveyRejects((survey) => {
    h3(survey).backends.mlx.structuralApplicability = "full";
  }, /its stacks derive partial/);
  await surveyRejects((survey) => {
    h3(survey).backends.mlx.nonWindowableStacks = ["vae.encoder"];
  }, /nonWindowableStacks is/);
  await surveyRejects((survey) => {
    h3(survey).backends.mlx.contractSupport = "structurally-not-applicable";
  }, /StructurallyNotApplicable while this survey names a windowable stack/);
  // sc-18662 made this record `contractSupport: "implemented"`, and the guard below only applies
  // when the contract does NOT implement rung 4 — so the non-implemented state is restored as the
  // mutation's PRECONDITION and the deleted field stays the sole variable. The control arm proves
  // the precondition alone does not reject, or these two would pass with the guard deleted.
  const notImplemented = (survey) => {
    h3(survey).backends.mlx.contractSupport = "missing";
    h3(survey).backends.mlx.implementation = "none";
    // sc-17153: `state` must agree with `contractSupport` under isImplemented(), so the
    // precondition flips it too — keeping the field under test the sole variable, as before.
    h3(survey).backends.mlx.state = "Missing";
  };
  {
    const control = await surveyFixture();
    notImplemented(control);
    await buildMatrix({
      publish: false,
      sourceOverrides: { rung4Survey: JSON.stringify(control) },
    });
  }
  await surveyRejects((survey) => {
    notImplemented(survey);
    delete h3(survey).backends.mlx.contractReason;
  }, /record contractSource and contractReason/);
  await surveyRejects((survey) => {
    // The guard fires only when the contract does NOT implement rung 4, which the MLX record now
    // does (sc-18662) — so the mutation claims an implementation on the CANDLE arm, which is still
    // `missing`. Same guard, same single variable, and it stays exercisable without weakening the
    // MLX record.
    h3(survey).backends.candle.implementation = "shared-primitive";
  }, /while the contract does not declare rung 4 Implemented/);
  await surveyRejects((survey) => {
    const stack = h3(survey).backends.mlx.stacks.find((entry) => entry.id === "audio_vae.decode");
    delete stack.structural;
  }, /Structurally N\/A claim, which the epic accepts only with static provider evidence/);
  await surveyRejects((survey) => {
    h3(survey).backends.mlx.stacks.find((entry) => entry.id === "vae.encoder").windowable = true;
  }, /contradicts structuralApplicability/);
  await surveyRejects((survey) => {
    delete h3(survey).backends.mlx.stacks[0].reason;
  }, /a per-stack verdict without a stated reason is an assertion/);
  await surveyRejects((survey) => {
    h3(survey).backends.mlx.stacks[1].id = h3(survey).backends.mlx.stacks[0].id;
  }, /ids must not repeat/);
  await surveyRejects((survey) => {
    h3(survey).backends.mlx.stacks = [];
  }, /the family verdict is derived from per-stack verdicts/);
  await surveyRejects((survey) => {
    delete h3(survey).backends.mlx.evidence;
  }, /must cite at least one source/);
});

test("the out-of-matrix record carries the cell's two distinct claims, and each is validated (sc-17153)", async () => {
  // The positive half: the shipped record parses through the real generator with both claims
  // present, in the cell vocabulary, and the CODE claim classifies through the shared predicate —
  // the same `isImplemented()` every coverage surface uses — rather than by string comparison.
  const survey = await surveyFixture();
  const mlx = h3(survey).backends.mlx;
  const candle = h3(survey).backends.candle;
  // sc-18663: the MLX row is the one epic 17137's terminal campaign measures, so what is pinned
  // about it is the RELATIONSHIP between its two claims, never a snapshot of either value. The
  // candle row is the opposite — `Missing` is permanent, and everything about it is pinned exactly.
  assert.ok(isImplemented(mlx.state), "the MLX code claim classifies as implemented");
  assert.equal(candle.state, "Missing");
  assert.ok(!isImplemented(candle.state), "the candle code claim classifies as not implemented");
  for (const backend of [mlx, candle]) {
    assert.ok(OUT_OF_MATRIX_CELL_STATES.includes(backend.state));
    // The PEAKS claim is a FUNCTION of the measured geometries, whatever they turn out to be. The
    // projection must carry `coveredFrameBound` too (sc-18663): the shared helper emits it as soon
    // as a geometry is temporal, and a three-key projection would silently stop comparing the whole
    // claim the moment the campaign records its first `WxHxfF` point.
    assert.deepEqual(
      declaredCharacterization(backend.memoryCharacterization),
      memoryCharacterization(backend.memoryCharacterization.measuredGeometries),
    );
    // ...and peaks may only be characterized where the CODE claim says the rung works.
    if (backend.memoryCharacterization.status !== "unmeasured") {
      assert.ok(
        isImplemented(backend.state),
        `${backend.state}: a characterized record must carry an implemented state`,
      );
    }
  }
  // Candle must stay `unmeasured` forever — the validator refuses a characterized non-implemented
  // record — and must say WHY, which is the half a status alone cannot carry.
  assert.equal(candle.memoryCharacterization.status, "unmeasured");
  assert.match(candle.memoryCharacterization.reason, /not implemented/i);
  // MLX: while the numbers are still owed, the reason names who owes them. Once the campaign fills
  // the row in, the derivation above is what keeps the claim honest.
  if (mlx.memoryCharacterization.status === "unmeasured") {
    assert.match(mlx.memoryCharacterization.reason, /terminal/i);
  }
  await buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(survey) } });

  // Each guard mutated ALONE, through the real generator — the same path `--check` takes.
  await surveyRejects((mutated) => {
    delete h3(mutated).backends.mlx.state;
  }, /unknown state/);
  await surveyRejects((mutated) => {
    // A receipt-backed state on a contract that does not implement the rung: the two claims
    // describe one contract and must agree UNDER THE PREDICATE (both `Verified` and
    // `Implemented/unverified` are implemented states, so either would contradict candle).
    h3(mutated).backends.candle.state = "Verified";
  }, /disagrees with contractSupport/);
  await surveyRejects((mutated) => {
    h3(mutated).backends.mlx.state = "Missing";
  }, /disagrees with contractSupport/);
  await surveyRejects((mutated) => {
    delete h3(mutated).backends.mlx.memoryCharacterization;
  }, /memoryCharacterization with a non-empty reason is required/);
  await surveyRejects((mutated) => {
    h3(mutated).backends.mlx.memoryCharacterization.reason = "";
  }, /memoryCharacterization with a non-empty reason is required/);
  await surveyRejects((mutated) => {
    // Asserting a status the geometries do not support: `point` with zero measured geometries is
    // exactly the drift the derivation rule exists to refuse.
    h3(mutated).backends.mlx.memoryCharacterization.status = "point";
  }, /does not derive from its measuredGeometries/);
  await surveyRejects((mutated) => {
    // A single geometry DOES derive `point` — so this mutation isolates the other rule: peaks on
    // a non-implemented state are refused, mirroring the published matrix's own cell rule.
    h3(mutated).backends.candle.memoryCharacterization = {
      ...h3(mutated).backends.candle.memoryCharacterization,
      status: "point",
      measuredGeometries: ["384x224"],
      coveredPixelBound: null,
    };
  }, /while the state is not implemented/);
});

test("an out-of-matrix record can record TEMPORAL coverage, and cannot assert it (sc-18663)", async () => {
  // The defect: the derive-check projected three keys while `memoryCharacterization()` emits a
  // FOURTH — `coveredFrameBound` — as soon as any geometry is temporal. The declared side could
  // therefore never equal the derived side for a multi-frame record, so a video family could not
  // record the temporal coverage epic 17137's terminal campaign exists to produce. The helper's
  // shape is asserted first, or the rest of this test would be pinning a defect that moved.
  assert.ok(
    Object.hasOwn(memoryCharacterization(["1344x768xf124"]), "coveredFrameBound"),
    "a temporal geometry makes the derived characterization four-keyed",
  );
  assert.ok(!Object.hasOwn(memoryCharacterization(["1344x768"]), "coveredFrameBound"));

  // Positive half, through the real generator: one temporal point, and a temporal FIT crossing two
  // areas and two frame counts — the rank-3 design `memoryCharacterization` grades against.
  const fitted = ["1344x768xf124", "1344x768xf61", "832x480xf124"];
  for (const geometries of [["1344x768xf124"], fitted]) {
    const survey = await surveyFixture();
    const mlx = h3(survey).backends.mlx;
    mlx.memoryCharacterization = temporalCharacterization(mlx, geometries);
    // The fixture is the campaign's own shape: the MLX row measured, the frame bound present.
    assert.equal(mlx.memoryCharacterization.status, geometries.length === 1 ? "point" : "fitted");
    assert.ok(Object.hasOwn(mlx.memoryCharacterization, "coveredFrameBound"));
    await buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(survey) } });
  }
  const fittedRecord = (survey) => {
    const mlx = h3(survey).backends.mlx;
    mlx.memoryCharacterization = temporalCharacterization(mlx, fitted);
    return mlx.memoryCharacterization;
  };
  assert.deepEqual(
    memoryCharacterization(fitted),
    { status: "fitted", measuredGeometries: fitted, coveredPixelBound: 1032192, coveredFrameBound: 124 },
    "the fixture's own bounds come from the shipped helper, not from this file",
  );

  // Negative half. Each guard mutated ALONE, through the real generator — a frame bound that does
  // not follow from the geometries is refused in every direction it can be wrong in.
  await surveyRejects((mutated) => {
    delete fittedRecord(mutated).coveredFrameBound;
  }, /does not derive from its measuredGeometries/);
  await surveyRejects((mutated) => {
    fittedRecord(mutated).coveredFrameBound = 61;
  }, /does not derive from its measuredGeometries/);
  await surveyRejects((mutated) => {
    // Smuggled the other way: a frame bound on a record whose geometries carry no frame axis at
    // all. The derived object has no such key, so the claim is the record's own.
    const mlx = h3(mutated).backends.mlx;
    mlx.memoryCharacterization = {
      ...temporalCharacterization(mlx, ["1344x768"]),
      coveredFrameBound: null,
    };
  }, /does not derive from its measuredGeometries/);
  await surveyRejects((mutated) => {
    // The status still follows from the geometries on the temporal axis too: three points at ONE
    // area cannot determine three coefficients, so `fitted` there is an assertion.
    const mlx = h3(mutated).backends.mlx;
    mlx.memoryCharacterization = {
      ...temporalCharacterization(mlx, ["1344x768xf121", "1344x768xf124", "1344x768xf61"]),
      status: "fitted",
      coveredPixelBound: 1032192,
      coveredFrameBound: 124,
    };
  }, /does not derive from its measuredGeometries/);

  // `sortedUnique` is part of the comparison, so ordering and repeats are part of the claim — and
  // the diagnostic has to SAY so rather than report two arrays that read as identical.
  await surveyRejects((mutated) => {
    fittedRecord(mutated).measuredGeometries = ["832x480xf124", "1344x768xf124", "1344x768xf61"];
  }, /must be an array that is sorted and duplicate-free/);
  await surveyRejects((mutated) => {
    const mlx = h3(mutated).backends.mlx;
    mlx.memoryCharacterization = {
      ...temporalCharacterization(mlx, ["1344x768xf124"]),
      measuredGeometries: ["1344x768xf124", "1344x768xf124"],
    };
  }, /must be an array that is sorted and duplicate-free/);

  // And the OTHER claim still fences it: a measured record on a backend whose state is not
  // implemented is refused, temporal or not. Candle's `Missing` row may never carry these numbers.
  await surveyRejects((mutated) => {
    const candle = h3(mutated).backends.candle;
    candle.memoryCharacterization = temporalCharacterization(candle, ["1344x768xf124"]);
  }, /while the state is not implemented/);
});

test("every remaining out-of-matrix throw site is mutated on its own (sc-18664 review)", async () => {
  // The rest of `parseOutOfMatrixRung4Families`. Split from the block above only for readability —
  // the rule is the same and it is the whole point of the file: ONE mutation per throw site, so a
  // green run is evidence about each guard rather than about the set. Every one of these went red
  // on its own before this test was committed.

  // The record has to say what it is a survey OF. An empty array, not a delete: `?.length` treats
  // them alike, and the empty array is the mutation a careless edit actually produces.
  await surveyRejects((survey) => {
    h3(survey).catalogEntries = [];
  }, /must name the catalog entries it is a survey OF/);

  // The three verdict-level vocabularies, each alone. `RUNG4_APPLICABILITIES` /
  // `RUNG4_IMPLEMENTATIONS` / `RUNG4_REQUEST_PEAKS` are the same frozen lists the `families`
  // verdicts are held to, and an out-of-matrix record has to be held to them too or the two halves
  // of the survey drift into different vocabularies.
  await surveyRejects((survey) => {
    h3(survey).backends.mlx.structuralApplicability = "mostly";
  }, /\(mlx\): unknown structuralApplicability "mostly"/);
  await surveyRejects((survey) => {
    h3(survey).backends.mlx.implementation = "partial";
  }, /unknown implementation "partial"/);
  await surveyRejects((survey) => {
    h3(survey).backends.mlx.requestPeak.finding = "probably-moves";
  }, /unknown requestPeak finding "probably-moves"/);

  // The stack-level vocabulary is a separate throw site from the verdict-level one, and the match
  // is anchored on the `stacks[...]` prefix so this cannot pass by hitting the verdict-level guard.
  await surveyRejects((survey) => {
    h3(survey).backends.mlx.stacks.find((entry) => entry.id === "vae.decoder").structuralApplicability =
      "windowable-ish";
  }, /stacks\["vae\.decoder"\]: unknown structuralApplicability "windowable-ish"/);

  // `contractSupport` mirrors `gen_core::MemoryStrategySupport`, so its vocabulary is closed too.
  await surveyRejects((survey) => {
    h3(survey).backends.mlx.contractSupport = "not-yet";
  }, /unknown contractSupport "not-yet"/);

  // `contractSource` deleted INDEPENDENTLY of `contractReason`. The guard is `!(reason && source)`,
  // so deleting only the reason (covered above) proves one conjunct and this proves the other.
  await surveyRejects((survey) => {
    // Same precondition restoration as the `contractReason` twin above (sc-18662).
    h3(survey).backends.mlx.contractSupport = "missing";
    h3(survey).backends.mlx.implementation = "none";
    delete h3(survey).backends.mlx.contractSource;
  }, /record contractSource and contractReason/);

  // The `implemented && derived === "none"` contradiction. Reachable, so it is tested rather than
  // deleted — but only by driving the whole record to a consistent `none`, which is why it had no
  // coverage: every stack unwindowable with its own structural evidence, the family verdict `none`
  // to match the derivation, `nonWindowableStacks` naming all of them, and only THEN the contract
  // claiming Implemented. Every earlier guard passes; this one is what catches it.
  await surveyRejects((survey) => {
    const verdict = h3(survey).backends.mlx;
    verdict.stacks = verdict.stacks.map((stack) => ({
      ...stack,
      structuralApplicability: "none",
      windowable: false,
      structural: stack.structural ?? [{ source: "inference:fixture", reason: "fixture" }],
    }));
    verdict.structuralApplicability = "none";
    verdict.nonWindowableStacks = verdict.stacks.map((stack) => stack.id);
    verdict.contractSupport = "implemented";
  }, /Implemented while this survey finds no windowable stack/);
});

test("an out-of-matrix record has to date the tree its evidence resolves in (sc-18664 review)", async () => {
  // The record cites crates the matrix's own pin does not contain, so `generatedFrom.
  // inferenceRevision` does not date it and something on the record has to.
  const survey = await surveyFixture();
  const cargo = await readFile(new URL("../Cargo.toml", import.meta.url), "utf8");
  // ANTI-VACUITY, not a snapshot. This used to read
  // `assert.equal(pin, "014134e3035ad7e4eca5c2ed7bded2375dc3c071")`, which pinned a literal that
  // carries no meaning of its own and goes stale on every inference bump — the exact defect class
  // sc-19751/sc-19758 removed elsewhere. The load-bearing assertion is `notEqual(revision, pin)`
  // below; all this needs to establish is that a pin was really parsed, so a regex that stopped
  // matching cannot turn that comparison into `undefined !== "79f02e..."` and pass vacuously.
  const pin = /rev = "([0-9a-f]{40})"/.exec(cargo)?.[1];
  // The pinned revision is the committed SC-20757 inference feature head. Keep this literal
  // alongside the current generated receipt: the assertion below only means something while the
  // pin is known, and `assert.notEqual(revision, pin)` is the claim this exists to make.
  assert.equal(pin, "c943f0e60ba70586dfe5309f60af7d6164f893d3");

  // The two backends now resolve at DIFFERENT revisions, per field's own definition: sc-18662's
  // streamed-request measurement re-surveyed the MLX record against the story branch, while the
  // Candle record was last surveyed at 79f02e6d0 and re-stamping it without re-surveying would be
  // provenance theater.
  const expected = {
    mlx: "e09f46aafb10126b14172a148acb26c619cf9213",
    candle: "79f02e6d0eaca861a0698ee490b70daa7441e321",
  };
  for (const backend of ["mlx", "candle"]) {
    const revision = h3(survey).backends[backend].contractRevision;
    assert.equal(revision, expected[backend]);
    // The field earns its place only because it DIFFERS from the pin. Asserting a value that
    // happened to equal the pin would be a green that proves nothing.
    assert.notEqual(revision, pin);
  }
  // And the divergence is stated in prose too, not left for a reader to infer from two shas —
  // BOTH revisions, since the record's paths no longer date at a single one.
  assert.match(h3(survey).whyOutOfMatrix, /e09f46aafb10126b14172a148acb26c619cf9213/);
  assert.match(h3(survey).whyOutOfMatrix, /79f02e6d0eaca861a0698ee490b70daa7441e321/);

  // Each guard mutated alone.
  await surveyRejects((mutated) => {
    delete h3(mutated).backends.mlx.contractRevision;
  }, /contractRevision must name the inference revision/);
  await surveyRejects((mutated) => {
    h3(mutated).backends.candle.contractRevision = "main";
  }, /contractRevision must name the inference revision/);
  await surveyRejects((mutated) => {
    // Too short to identify a commit unambiguously — a 7-character abbreviation is not provenance.
    h3(mutated).backends.mlx.contractRevision = "79f02e6";
  }, /at least 9 hex characters/);
});

test("the out-of-matrix record self-invalidates the day the matrix learns the family", async () => {
  // The guard that keeps this section from becoming a museum. It is keyed by epic id precisely
  // because `familyGroup` has no answer for a video entry; if it ever gains one, the record has to
  // move into `families`, where the coverage fence can see it.
  await surveyRejects((survey) => {
    h3(survey).catalogEntries = ["sdxl"];
  }, /familyGroup now resolves sdxl to family SC-15525/);
  await surveyRejects((survey) => {
    survey.outOfMatrixFamilies["15525"] = h3(survey);
  }, /is also a `families` key/);

  // And the reason it cannot simply live in `families` today: withdrawing the out-of-matrix record
  // admits the MiniMax entries into the modality-aware universe (the sc-17137 main sync keyed the
  // universe exclusion on these records), where route resolution fails outright — `familyGroup` has
  // no arm and no video-route resolver row exists — rather than sitting inert.
  await surveyRejects((survey) => {
    delete survey.outOfMatrixFamilies["17137"];
    survey.families["17137"] = {
      name: "MiniMax-H3",
      backends: {
        mlx: {
          structuralApplicability: "partial",
          implementation: "none",
          requestPeak: { finding: "unmeasured" },
          evidence: [{ source: "inference:crates/media/mlx-gen/mlx-gen-minimax-h3/src/memory_strategy.rs" }],
        },
      },
    };
  }, /minimax_h3: no resolved route\/provider/);
});

test("deriveOutOfMatrixApplicability restates the survey's vocabulary for unconditionally resident stacks", () => {
  const stack = (structuralApplicability) => ({ structuralApplicability });
  assert.equal(deriveOutOfMatrixApplicability([stack("full")]), "full");
  // Two separately-indexed stacks need a plan each, which is `partial` by the vocabulary's first
  // clause even though both are individually uniform.
  assert.equal(deriveOutOfMatrixApplicability([stack("full"), stack("full")]), "partial");
  // A remainder that cannot be windowed is `partial` by its third clause.
  assert.equal(deriveOutOfMatrixApplicability([stack("full"), stack("none")]), "partial");
  assert.equal(deriveOutOfMatrixApplicability([stack("partial")]), "partial");
  assert.equal(deriveOutOfMatrixApplicability([stack("none"), stack("none")]), "none");
});

test("the derivation reproduces 38 of the 40 `families` verdicts, and route scoping is the exception", async () => {
  // sc-18664 review. The claim used to be that this function is simply the survey's vocabulary
  // restated. It is not quite, so the gap is measured here rather than asserted away. Applying the
  // derivation to every verdict `families` already carries — `blockStacks[].windowable` mapped onto
  // the applicability vocabulary — reproduces all but Qwen-Image's two.
  const survey = await surveyFixture();
  const disagreements = [];
  let verdicts = 0;
  for (const [group, family] of Object.entries(survey.families)) {
    for (const [backend, verdict] of Object.entries(family.backends ?? {})) {
      verdicts += 1;
      const stacks = (verdict.blockStacks ?? []).map((entry) => ({
        structuralApplicability: entry.windowable ? "full" : "none",
        routeScoped: (entry.entries ?? []).length > 0,
      }));
      assert.ok(stacks.length, `SC-${group} ${backend} carries no blockStacks to derive from`);
      const derived = deriveOutOfMatrixApplicability(stacks);
      if (derived !== verdict.structuralApplicability) {
        disagreements.push({ group, backend, derived, recorded: verdict.structuralApplicability, stacks });
      }
    }
  }
  // 40 image-lane verdicts plus the 9 video (family, backend) verdicts sc-18815 added on main;
  // the two route-scoped disagreements below are unchanged.
  assert.equal(verdicts, 49);
  assert.equal(verdicts - disagreements.length, 47);

  // And the two are the SAME family on both backends, disagreeing for the SAME reason: the second
  // stack is control-route-only, so it is not resident on the routes the `full` verdict covers, and
  // this function has no route axis to see that with.
  assert.deepEqual(
    disagreements.map((entry) => `${entry.group}:${entry.backend}`).sort(),
    ["15511:candle", "15511:mlx"],
  );
  for (const entry of disagreements) {
    assert.equal(entry.recorded, "full");
    assert.equal(entry.derived, "partial");
    assert.equal(entry.stacks.filter((stack) => stack.routeScoped).length, 1);
  }
  for (const backend of ["mlx", "candle"]) {
    assert.deepEqual(survey.families["15511"].backends[backend].blockStacks[1].entries, [
      "qwen_image_control",
    ]);
  }

  // Why route scoping is not added rather than merely undone: no out-of-matrix record needs it.
  // MiniMax-H3 has a route-conditional stack of its own — `text_encoder.vision_tower`, the `fl2va`
  // keyframe path only — and drops out to the same answer with it removed, because its conv
  // remainder and its several separately-indexed denoise stacks each force `partial` unaided.
  const mlx = h3(survey).backends.mlx;
  const vision = mlx.stacks.find((stack) => stack.id === "text_encoder.vision_tower");
  assert.match(vision.blocks, /`fl2va` keyframe path only/);
  assert.equal(deriveOutOfMatrixApplicability(mlx.stacks), "partial");
  assert.equal(
    deriveOutOfMatrixApplicability(mlx.stacks.filter((stack) => stack !== vision)),
    "partial",
  );
});

test("the ban on the removed rung-1 claim reaches the generator's OWN prose, not just the notes", async () => {
  // sc-18664 review. The corrected sentence survived ~500 lines below the guard that bans it, in
  // the `stagedResidencyIsAvailable` docstring — the site a reader of the gate itself lands on.
  // Banning a sentence in the data file while it lives on in the file the guard is written in is
  // the "one half of a pair moved" defect, so the generator's own source is now scanned too.
  const source = await readFile(new URL("./generate-memory-matrix.mjs", import.meta.url), "utf8");
  assertGeneratorSourceDoesNotRestateTheRemovedEdge(source);
  // The generator's prose names rung 4's actual sole shared prerequisite, and names the constant it
  // comes from. sc-19542 moved that statement from the `stagedResidencyIsAvailable` docstring — where
  // it described a proxy — onto `SHARED_RUNG4_PREREQUISITES`, which is the gate's own copy of it, so
  // the value is asserted as a VALUE here rather than only as prose.
  assert.match(source, /BOUNDED_TRANSFORMER_RESIDENCY_REQUIRES/);
  assert.match(
    source,
    /`&\[MemoryStrategyPrerequisite::LoadShape\(LoadShape::DeferredMaterialization\)\]`/,
  );
  assert.deepEqual(
    SHARED_RUNG4_PREREQUISITES.map((prerequisite) => [prerequisite.kind, prerequisite.shape]),
    [["load-shape", "deferred_materialization"]],
  );
  assert.match(source, /additional_prerequisites/);

  // The exact text the review found, verbatim from the pre-correction docstring and wrapped across
  // two comment lines exactly as it was.
  const RESTORED = [
    " * Whether this entry advertises rung 1 on this backend. Rung 4 requires rung 1 engaged in the same",
    " * request (`gen_core::memory_strategy`'s `BOUNDED_TRANSFORMER_RESIDENCY_REQUIRES`), so the rung-4 arm",
    " * below reads the prerequisite from the SAME predicate the rung-1 arm uses.",
  ].join("\n");
  assert.throws(
    () => assertGeneratorSourceDoesNotRestateTheRemovedEdge(`/**\n${RESTORED}\n */\n`),
    /restates the rung-1 prerequisite SC-15998 removed/,
  );

  // Each pattern in the shared set proven on its own, so a green run is evidence about both rather
  // than about the set.
  assert.throws(
    () => assertGeneratorSourceDoesNotRestateTheRemovedEdge("// rung 4 requires rung 1.\n"),
    /rung 4 requires rung 1/,
  );
  assert.throws(
    () =>
      assertGeneratorSourceDoesNotRestateTheRemovedEdge(
        "// The window also requires rung 1 engaged in the same request.\n",
      ),
    /requires rung 1 engaged in the same request/,
  );

  // The flattening, proven on its OWN. The fixture above does not prove it: its first line carries
  // `Rung 4 requires rung 1` whole, so the shorter pattern catches it line-oriented and the guard
  // stays green with the flattening deleted — which is what a mutation run of this file showed.
  // This fixture trips ONLY the long pattern and only once comment continuations are joined, so it
  // is the one that goes green if the flattening is removed. It is also the real defect's shape:
  // the surviving docstring wrapped between `same` and `request`, which is exactly why a
  // line-oriented grep of the generator reported no match at all.
  const WRAPPED_ONLY = "/**\n * The window additionally requires rung 1 engaged in the same\n * request, per this provider's own edge.\n */\n";
  assert.doesNotMatch(WRAPPED_ONLY, /rung 4 requires rung 1/i, "must not trip the short pattern");
  assert.doesNotMatch(
    WRAPPED_ONLY,
    /requires rung 1 engaged in the same request/i,
    "and must be invisible to a line-oriented scan before flattening",
  );
  assert.throws(
    () => assertGeneratorSourceDoesNotRestateTheRemovedEdge(WRAPPED_ONLY),
    /requires rung 1 engaged in the same request/,
  );

  // The excision that lets the file scan itself fails CLOSED. Rename the pattern constant and the
  // literals stop being excised, so the guard throws instead of going quietly green — the failure
  // direction that matters for a self-scanning gate.
  assert.throws(
    () =>
      assertGeneratorSourceDoesNotRestateTheRemovedEdge(
        source.replace(
          "export const STALE_RUNG1_PREREQUISITE_PATTERNS = Object.freeze([",
          "export const RENAMED_PATTERNS = Object.freeze([",
        ),
      ),
    /restates the rung-1 prerequisite SC-15998 removed/,
  );

  // Reach. `parseRung4Survey` is what `buildMatrix` calls on every generation — every other survey
  // mutation in this file proves that edge — so wiring the source scan into it is what puts the
  // generator's prose behind `--check` and the pre-push hook. Injecting the source proves the call
  // happens without touching the real file.
  const survey = await readFile(SURVEY_URL, "utf8");
  parseRung4Survey(survey, { familyGroups: familyGroup, generatorSource: source });
  assert.throws(
    () =>
      parseRung4Survey(survey, {
        familyGroups: familyGroup,
        generatorSource: `${source}\n// rung 4 requires rung 1.\n`,
      }),
    /restates the rung-1 prerequisite SC-15998 removed/,
  );
});

test("overlay incompatibility is a provider fact, applied where evidenced and nowhere else", async () => {
  const matrix = await buildMatrix({ publish: false });
  const na = matrix.cells.filter(
    (cell) =>
      cell.rung === "bounded_transformer_residency" && cell.state === "Structurally N/A",
  );
  assert.ok(na.length > 0);
  assert.ok(
    na.every((cell) => cell.backend === "candle" && cell.modelId === "krea_2_turbo"),
    "only Krea's Candle loader folds adapters into the base weight at load",
  );
  assert.ok(na.every((cell) => cell.overlay !== "none"));
  assert.ok(
    na.every((cell) => cell.evidence.structural.length >= 2),
    "a Structurally N/A verdict carries its static evidence, in both repos",
  );

  // The exemption reaches only cells whose entry AND mode actually have the streaming path. Without
  // that scoping it spread to krea_2_raw, which has no streaming path at all, and to krea_2_turbo's
  // edit modes — exempting cells from the calibration workload on the strength of a path that does
  // not exist there.
  assert.ok(na.every((cell) => cell.mode === "text_to_image"));
  assert.equal(
    matrix.cells.filter(
      (cell) =>
        cell.modelId === "krea_2_raw" &&
        cell.rung === "bounded_transformer_residency" &&
        cell.state === "Structurally N/A",
    ).length,
    0,
    "krea_2_raw has no streaming path, so its overlay cells are Missing, not exempt",
  );

  // The flag is its own field. `structuralApplicability` keeps reporting the ARCHITECTURE, which for
  // Krea is a windowable 28-block trunk however its adapters are installed.
  assert.ok(na.every((cell) => cell.rung4Survey.overlayIncompatible === true));
  assert.ok(na.every((cell) => cell.rung4Survey.structuralApplicability === "partial"));
  const structurallyInapplicable = matrix.cells.filter(
    (cell) => cell.rung4Survey?.structuralApplicability === "none",
  );
  assert.deepEqual(
    structurallyInapplicable,
    [],
    "no surveyed family is architecturally inapplicable; per-provider overlay exemptions remain separate",
  );

  // The contrast that makes this a provider fact rather than a rung fact: MLX Z-Image replays
  // forward-time residuals onto each materialized block, so its overlay cells stream fine.
  const zImageOverlay = matrix.cells.filter(
    (cell) =>
      cell.modelId === "z_image_turbo" &&
      cell.backend === "mlx" &&
      cell.rung === "bounded_transformer_residency" &&
      cell.overlay === "lora",
  );
  assert.ok(zImageOverlay.length > 0);
  assert.ok(zImageOverlay.every((cell) => cell.state === "Implemented/unverified"));
});

test("a Structurally N/A survey verdict without structural evidence is rejected", async () => {
  // AC4: the epic accepts a static N/A verdict BECAUSE the evidence is present. An `applicability:
  // "none"` with an empty `structural` array would turn that allowance into a bare assertion, so it
  // must fail generation rather than emit an unevidenced exemption.
  const survey = await surveyFixture();
  survey.families["15525"].backends.mlx.structuralApplicability = "none";
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(survey) } }),
    /static provider evidence/,
  );

  // With evidence it is accepted, and it reaches the cell as Structurally N/A carrying that
  // evidence — the positive control, so the rejection above is not passing for an unrelated reason.
  survey.families["15525"].backends.mlx.structural = [
    { source: "inference:crates/media/mlx-gen/mlx-gen-sdxl/src/unet/mod.rs", reason: "fixture" },
  ];
  const matrix = await buildMatrix({
    publish: false,
    sourceOverrides: { rung4Survey: JSON.stringify(survey) },
  });
  const cells = matrix.cells.filter(
    (cell) =>
      cell.modelId === "sdxl" &&
      cell.backend === "mlx" &&
      cell.rung === "bounded_transformer_residency",
  );
  assert.ok(cells.length > 0);
  assert.ok(
    cells.every(
      (cell) =>
        cell.state === "Structurally N/A" &&
        cell.evidence.structural.some((entry) => entry.reason === "fixture") &&
        cell.rung4Survey.structuralApplicability === "none",
    ),
  );
});

test("a survey that contradicts itself or misses a family fails generation", async () => {
  const survey = await surveyFixture();

  const missing = await surveyFixture();
  delete missing.families["15525"];
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(missing) } }),
    /no rung-4 survey verdict/,
  );

  const contradictory = await surveyFixture();
  contradictory.families["15509"].backends.mlx.implementedEntries = ["mage_flow"];
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(contradictory) } }),
    /the two must agree/,
  );

  const foreign = await surveyFixture();
  foreign.families["15510"].backends.mlx.implementedEntries.push("qwen_image");
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(foreign) } }),
    /belongs to another family/,
  );

  const unknown = await surveyFixture();
  unknown.families["15509"].backends.mlx.structuralApplicability = "probably";
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(unknown) } }),
    /unknown structuralApplicability/,
  );

  const unevidenced = await surveyFixture();
  unevidenced.families["15509"].backends.mlx.evidence = [];
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(unevidenced) } }),
    /cite at least one source/,
  );

  const emptyTiers = await surveyFixture();
  emptyTiers.families["15517"].backends.mlx.implementedTiers = [];
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(emptyTiers) } }),
    /implementedTiers is empty/,
  );

  const unknownOverlay = await surveyFixture();
  unknownOverlay.families["15517"].backends.mlx.implementedOverlays = ["mystery"];
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(unknownOverlay) } }),
    /implementedOverlays contains an overlay outside the matrix vocabulary/,
  );

  const emptyScope = await surveyFixture();
  emptyScope.families["15512"].backends.mlx.implementationScopes[0].entries = [];
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(emptyScope) } }),
    /entries must name at least one catalog entry/,
  );

  const overlappingScopes = await surveyFixture();
  overlappingScopes.families["15512"].backends.mlx.implementationScopes.push({
    entries: ["lens"],
    tiers: ["q4"],
    strategyParameters: { transformerWindowSize: 2 },
  });
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(overlappingScopes) } }),
    /overlaps implementationScopes\[0\]/,
  );

  const foreignScope = await surveyFixture();
  foreignScope.families["15512"].backends.mlx.implementationScopes[0].entries = ["qwen_image"];
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(foreignScope) } }),
    /belongs to another family/,
  );

  const mixedSelectors = await surveyFixture();
  mixedSelectors.families["15512"].backends.mlx.implementedEntries = ["lens"];
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(mixedSelectors) } }),
    /either legacy implementedEntries fields or exact implementationScopes/,
  );

  // Positive control: the shipped survey builds.
  await buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(survey) } });
});

test("a survey edit rotates the source fingerprint", async () => {
  // The survey is now a generated-matrix input, so it has to be inside the staleness tripwire.
  // Without this, editing a verdict would leave `sceneWorksRevision` claiming the same provenance.
  const baseline = await buildMatrix({ publish: false });
  const survey = await surveyFixture();
  survey.families["15523"].backends.mlx.summary = `${survey.families["15523"].backends.mlx.summary}.`;
  const edited = await buildMatrix({
    publish: false,
    sourceOverrides: { rung4Survey: JSON.stringify(survey) },
  });
  assert.notEqual(
    edited.generatedFrom.sceneWorksRevision,
    baseline.generatedFrom.sceneWorksRevision,
  );
  assert.equal(edited.generatedFrom.sources.rung4Survey.path, SOURCE_PATHS.rung4Survey);
});

test("the survey vocabulary, the generator's enums and the published schema agree", async () => {
  // Three copies of the same vocabulary — the survey file documents it, the generator validates
  // against it, the schema constrains the artifact. Nothing connected them, so a value added to one
  // could be silently rejected by another; the generator would then fail on a survey the schema
  // considers valid, or vice versa.
  const schema = JSON.parse(
    await readFile(new URL("../packages/schemas/memory-matrix.schema.json", import.meta.url), "utf8"),
  );
  assert.deepEqual(schema.$defs.rung4StructuralApplicability.enum, [...RUNG4_APPLICABILITIES]);
  assert.deepEqual(schema.$defs.rung4Implementation.enum, [...RUNG4_IMPLEMENTATIONS]);
  assert.deepEqual(schema.$defs.rung4RequestPeak.enum, [...RUNG4_REQUEST_PEAKS]);

  // Both places the schema constrains these values must go through the SAME definition. A second
  // inline copy is what the test above cannot see drifting.
  // sc-18815 wrapped the CELL verdict's three fields in a `oneOf` so an unsurveyed family can
  // publish nulls and `unsurveyed`. The shared definition still has to be the only source of the
  // real vocabulary in both places, so resolve through the wrapper rather than dropping the check —
  // an inline enum copy hiding inside a `oneOf` is exactly the drift this test exists to catch.
  const refsOf = (schemaNode) =>
    (schemaNode.$ref ? [schemaNode.$ref] : (schemaNode.oneOf ?? []).map((branch) => branch.$ref)).filter(Boolean);
  for (const properties of [
    schema.$defs.rung4SurveyVerdict.properties,
    schema.properties.rung4SurveyRows.items.properties,
  ]) {
    assert.ok(refsOf(properties.structuralApplicability).includes("#/$defs/rung4StructuralApplicability"));
    assert.ok(refsOf(properties.implementation).includes("#/$defs/rung4Implementation"));
    assert.ok(refsOf(properties.requestPeak).includes("#/$defs/rung4RequestPeak"));
    // No branch may inline its own enum: that is how a fourth value would appear in the artifact
    // while the three constants above still agreed with each other.
    for (const field of ["structuralApplicability", "implementation", "requestPeak"]) {
      for (const branch of properties[field].oneOf ?? []) {
        assert.ok(
          branch.$ref || branch.type === "null" || typeof branch.const === "string",
          `${field}: a oneOf branch must be the shared $ref, null, or a named const — not an inline enum`,
        );
      }
    }
  }

  // The survey file documents the same vocabulary. Compared as SETS: the documentation order is
  // presentational, and coupling it to the enum order would redden on a harmless reorder.
  const survey = await surveyFixture();
  const sorted = (values) => [...values].sort();
  assert.deepEqual(
    sorted(Object.keys(survey.vocabulary.structuralApplicability)),
    sorted(RUNG4_APPLICABILITIES),
  );
  assert.deepEqual(sorted(Object.keys(survey.vocabulary.implementation)), sorted(RUNG4_IMPLEMENTATIONS));
  assert.deepEqual(sorted(Object.keys(survey.vocabulary.requestPeak)), sorted(RUNG4_REQUEST_PEAKS));
});

test("a block stack may not name another family's catalog entry", async () => {
  // `blockStacks[].entries` is published on the family's `rung4SurveyRows` row (sc-18099 moved it
  // there from every rung-4 cell), so a typo'd or foreign id becomes a per-entry "fact" about entries
  // that are not in the family at all. `implementedEntries` was checked from the start; this field
  // was added later and inherited nothing. Moving where it is published changes nothing about that:
  // the id is still asserted to belong to the owning family at generation time.
  const foreign = await surveyFixture();
  foreign.families["15522"].backends.mlx.blockStacks[0].entries = ["qwen_image"];
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(foreign) } }),
    /blockStacks.*names qwen_image, which belongs to another family/,
  );

  const invented = await surveyFixture();
  invented.families["15522"].backends.mlx.blockStacks[0].entries = ["totally_made_up_entry"];
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(invented) } }),
    /names totally_made_up_entry/,
  );
});

test("every control-advertising family inventories its control-branch stack", async () => {
  // A control route holds a SECOND transformer resident alongside the trunk, which is exactly the
  // quantity a partial verdict exists to state. z-image was inventoried first and the rest were not,
  // so the inventory said different things about the same shape depending on the family.
  const matrix = await buildMatrix({ publish: false });
  const controlFamilies = new Set(
    matrix.cells
      .filter((cell) => cell.rung === "bounded_transformer_residency" && cell.overlay === "control")
      .map((cell) => `${familyGroup(cell.modelId)}:${cell.backend}`),
  );
  assert.ok(controlFamilies.size >= 10);
  for (const key of controlFamilies) {
    // sc-18099: block stacks are family-level and live on `rung4SurveyRows`. The key is already the
    // family/backend pair the control cells resolved to, so the row is the same verdict the cell
    // carried, read where it is now published.
    const [familyStory, backend] = key.split(":");
    const row = matrix.rung4SurveyRows.find(
      (candidate) => candidate.familyStory === Number(familyStory) && candidate.backend === backend,
    );
    assert.ok(row, `${key}: advertises a control overlay but has no rung-4 survey row`);
    const controlStacks = row.blockStacks.filter((stack) =>
      /control|ControlNet|IdentityNet/i.test(stack.name),
    );
    assert.ok(
      controlStacks.length > 0,
      `${key}: advertises a control overlay but its block-stack inventory names no control stack`,
    );
  }
});

test("only the independently wired base Z-Image Candle control route exposes staged residency", async () => {
  const matrix = await buildMatrix({ publish: false });
  const cells = matrix.cells.filter(
    (cell) =>
      ["z_image", "z_image_turbo"].includes(cell.modelId) &&
      cell.backend === "candle" &&
      cell.overlay === "control" &&
      cell.rung === "staged_residency",
  );
  assert.ok(cells.length > 0);
  assert.ok(
    cells
      .filter((cell) => cell.modelId === "z_image")
      .every((cell) => ["Implemented/unverified", "Verified"].includes(cell.state)),
  );
  // RESTATED 2026-08-17 (sc-20246), for the same reason and in the same shape as the sibling
  // restatement in "an implemented family is Implemented/unverified only where the provider actually
  // exposes it": the Turbo control cell used to be required Missing, which read the absence of a
  // declaration as the invariant. `candle-gen-z-image` registers `z_image_turbo_control` as its own
  // provider with its own contract, and the candle dump exports staged residency for it, so the
  // engine-derived projection declares it under `runtimeProvider: "z_image_turbo_control"`.
  //
  // The anti-leak intent is what mattered and is preserved exactly: base Z-Image's declaration must
  // not reach the Turbo control route. That is asserted as identity rather than as absence — a Turbo
  // control cell may be Implemented, but ONLY while resolving to `z_image_turbo_control`. If it ever
  // went Implemented under the base `z_image_turbo` identity, that IS the leak, and this reds.
  assert.ok(
    cells
      .filter((cell) => cell.modelId === "z_image_turbo")
      .every(
        (cell) =>
          cell.resolvedRoute === "z_image_turbo_control" &&
          ["Implemented/unverified", "Verified"].includes(cell.state),
      ),
  );
});

test("a survey verdict that reaches no cell is rejected, not silently carried", async () => {
  // Coverage runs both ways. `rung4SurveyRows` is derived from the generated cells, so a verdict for
  // a family or backend the catalog does not advertise appears nowhere at all — it would sit in the
  // file being maintained, reviewed and trusted while having no effect.
  const survey = await surveyFixture();
  survey.families["15523"].backends.cuda = {
    ...survey.families["15523"].backends.mlx,
    summary: "fixture: cuda is not a matrix backend",
  };
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(survey) } }),
    /the verdict reaches no cell/,
  );
});

test("a verdict may run ahead of the model universe, but no further than the routing catalog (sc-18813)", async () => {
  // The two-state coverage check cannot express the ORDER epic 18803 chose. Admitting video to the
  // model universe (sc-18815) destructures a survey verdict for every rung-4 cell it creates, so the
  // verdict has to exist by then; landing the verdict first — sc-18813, so the survey is reviewable
  // on its own — hits the survey -> catalog arm, because the universe is still image-only. That is
  // not a structural circle: sc-18815 could have gone first had it carried the verdict too, and once
  // `ltx_2_3` is in `advertised` the strict arm covers it with no gate change at all. The third
  // state buys the slicing. So a verdict is allowed to run AHEAD of admission — bounded by what the
  // routing catalog actually routes, and by nothing else.
  const [manifestBody, routingCatalog, routingCandle, routingMlx] = await Promise.all([
    readFile(new URL("../config/manifests/builtin.models.jsonc", import.meta.url), "utf8"),
    readFile(
      new URL("../crates/sceneworks-core/src/jobs_store/routing/catalog.rs", import.meta.url),
      "utf8",
    ),
    readFile(
      new URL("../crates/sceneworks-core/src/jobs_store/routing/candle.rs", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../crates/sceneworks-core/src/jobs_store/routing/mlx.rs", import.meta.url), "utf8"),
  ]);
  const manifestModels = JSON.parse(stripJsoncComments(manifestBody)).models;
  const routed = catalogFamilyBackends(
    manifestModels,
    routedLanes({ routingCatalog, routingCandle, routingMlx }),
  );

  // The LTX family remains routed on both lanes through base `ltx_2_3`; Eros contributes only MLX.
  // Family-level survey verdicts are therefore still inside both real routing fences.
  assert.ok(routed.has("ltx-video:mlx"));
  assert.ok(routed.has("ltx-video:candle"));
  // A lane the catalog does not route is NOT.
  assert.ok(!routed.has("ltx-video:cuda"));
  // That line alone does not pin the ORACLE, only the gate's existence: `routedLanes` emits nothing
  // but "mlx" and "candle", so `:cuda` is absent under EVERY implementation — including one that
  // ignored the routing map and admitted any lane of any group `familyGroup` knows. Nor does real
  // data discriminate, because no shipped family group is routed on exactly one lane. A synthetic
  // routing map is the only case that separates the two: `ltx_2_3` routes mlx, `ltx_2_3_eros` routes
  // nothing, so a fence that CONSULTS the map admits mlx and refuses candle, while a fence that
  // assumed "every lane" admits both.
  const synthetic = catalogFamilyBackends(manifestModels, new Map([["ltx_2_3", new Set(["mlx"])]]));
  assert.ok(synthetic.has("ltx-video:mlx"), "the routed lane is admitted");
  assert.ok(
    !synthetic.has("ltx-video:candle"),
    "the fence is the routing oracle, not the family key: an unrouted lane of a routed family stays out",
  );

  // sc-18815 landed, so the fence SELF-CLEARED for ltx-video exactly as its doc comment promised:
  // the family is in `advertised` now, the strict arm covers it, and the verdict reaches real cells.
  // Pinned as a state change rather than deleted — "the third state is temporary" is a claim, and
  // this is where it is checked.
  const survey = await surveyFixture();
  assert.ok(survey.families["ltx-video"], "the shipped survey carries the ltx-video verdict");
  const matrix = await buildMatrix({ publish: false });
  const advertised = new Set(
    matrix.models.flatMap((model) =>
      model.backends.map((backend) => `${familyGroup(model.id)}:${backend}`),
    ),
  );
  assert.ok(advertised.has("ltx-video:mlx") && advertised.has("ltx-video:candle"));
  assert.deepEqual(
    matrix.rung4SurveyRows.filter((row) => row.familyStory === "ltx-video").map((row) => row.backend).sort(),
    ["candle", "mlx"],
  );
  assert.ok(matrix.cells.some((cell) => cell.modelId === "ltx_2_3"));
  assert.equal(matrix.summary.rung4Survey.surveyedFamilyBackends, matrix.rung4SurveyRows.length);

  // And a group the catalog maps nothing to is still rejected, so the third state is a fence rather
  // than a hole. `familyGroup` throws on an unknown id, which is what keeps the fence closed.
  const orphan = await surveyFixture();
  orphan.families["99999"] = {
    name: "fixture: a family the catalog knows nothing about",
    backends: {
      mlx: {
        structuralApplicability: "partial",
        implementation: "none",
        summary: "fixture",
        blockStacks: [],
        evidence: [{ source: "fixture", reason: "fixture" }],
        requestPeak: { finding: "unmeasured", reason: "fixture" },
        findings: [],
      },
    },
  };
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(orphan) } }),
    /the verdict reaches no cell/,
  );
});

test("`requires-different-primitive` is a finding, never an exemption or an implementation", async () => {
  // AC5 as a machine check. This value exists so a family the primitive cannot express in its
  // current SHAPE is recorded as a finding rather than rounded to Structurally N/A. Two ways that
  // could rot: recording it with no finding (indistinguishable from a bare Missing), or recording it
  // alongside an implementation claim (a contradiction). Both must fail generation.
  const bare = await surveyFixture();
  bare.families["15511"].backends.mlx.structuralApplicability = "requires-different-primitive";
  bare.families["15511"].backends.mlx.implementation = "none";
  delete bare.families["15511"].backends.mlx.implementedEntries;
  delete bare.families["15511"].backends.mlx.implementedModes;
  delete bare.families["15511"].backends.mlx.implementedTiers;
  delete bare.families["15511"].backends.mlx.implementedOverlays;
  bare.families["15511"].backends.mlx.findings = [];
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(bare) } }),
    /must state the shape gap as a finding/,
  );

  const contradictory = await surveyFixture();
  contradictory.families["15510"].backends.mlx.structuralApplicability =
    "requires-different-primitive";
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(contradictory) } }),
    /declaring the primitive's shape insufficient/,
  );

  // Stated as a finding, it builds — and the cell reports Missing with the verdict attached, NOT
  // Structurally N/A. That distinction is the whole point of the value.
  const stated = await surveyFixture();
  stated.families["15511"].backends.mlx.structuralApplicability = "requires-different-primitive";
  stated.families["15511"].backends.mlx.implementation = "none";
  delete stated.families["15511"].backends.mlx.implementedEntries;
  delete stated.families["15511"].backends.mlx.implementedModes;
  delete stated.families["15511"].backends.mlx.implementedTiers;
  delete stated.families["15511"].backends.mlx.implementedOverlays;
  stated.families["15511"].backends.mlx.findings = ["fixture: the driver's shape cannot express it"];
  const sourceOverrides = { rung4Survey: JSON.stringify(stated) };
  const matrix = await buildMatrix({
    publish: false,
    sourceOverrides,
  });
  const cells = matrix.cells.filter(
    (cell) =>
      cell.modelId === "qwen_image" &&
      cell.backend === "mlx" &&
      cell.rung === "bounded_transformer_residency",
  );
  assert.ok(cells.length > 0);
  assert.ok(
    cells.every(
      (cell) =>
        cell.state === "Missing" &&
        cell.rung4Survey.structuralApplicability === "requires-different-primitive",
    ),
  );
  // The finding is what separates this value from a bare N/A, so it must actually be published —
  // on the family row since sc-18099, where it survives the family's cells being elided.
  const row = matrix.rung4SurveyRows.find(
    (candidate) => candidate.familyStory === 15511 && candidate.backend === "mlx",
  );
  assert.equal(row.structuralApplicability, "requires-different-primitive");
  assert.ok(row.findings.length > 0);
});

test("`does-not-move` is carried through to the cell rather than collapsed into `unmeasured`", async () => {
  // The epic's non-negotiable has a positive form: a family MEASURED not to move the request peak is
  // a different fact from one nobody has measured, and the selector must be able to tell them apart.
  const survey = await surveyFixture();
  survey.families["15511"].backends.mlx.requestPeak = {
    finding: "does-not-move",
    reason: "fixture",
  };
  const matrix = await buildMatrix({
    publish: false,
    sourceOverrides: { rung4Survey: JSON.stringify(survey) },
  });
  assert.ok(
    matrix.cells
      .filter(
        (cell) =>
          cell.modelId === "qwen_image" &&
          cell.backend === "mlx" &&
          cell.rung === "bounded_transformer_residency",
      )
      .every((cell) => cell.rung4Survey.requestPeak === "does-not-move"),
  );
  assert.equal(
    matrix.rung4SurveyRows.find((row) => row.familyStory === 15511 && row.backend === "mlx")
      .requestPeak,
    "does-not-move",
  );
});

test("request-peak measurements can be scoped to the exact entry, mode, and overlay exercised", async () => {
  const survey = await surveyFixture();
  survey.families["15511"].backends.mlx.requestPeak = {
    finding: "unmeasured",
    reason: "fixture",
    scopes: [
      {
        entries: ["qwen_image"],
        tiers: ["bf16", "q4", "q8"],
        modes: ["text_to_image"],
        overlays: ["none"],
        finding: "does-not-move",
      },
    ],
  };
  const matrix = await buildMatrix({
    publish: false,
    sourceOverrides: { rung4Survey: JSON.stringify(survey) },
  });
  const qwen = matrix.cells.filter(
    (cell) => cell.backend === "mlx" && cell.rung === "bounded_transformer_residency" &&
      cell.modelId.startsWith("qwen_image"),
  );
  assert.ok(qwen.length > 0);
  assert.ok(
    qwen
      .filter(
        (cell) =>
          cell.modelId === "qwen_image" && cell.mode === "text_to_image" && cell.overlay === "none",
      )
      .every((cell) => cell.rung4Survey.requestPeak === "does-not-move"),
  );
  assert.ok(
    qwen
      .filter(
        (cell) =>
          cell.modelId !== "qwen_image" || cell.mode !== "text_to_image" || cell.overlay !== "none",
      )
      .every((cell) => cell.rung4Survey.requestPeak === "unmeasured"),
  );
});

test("request-peak tier overrides fail closed", async () => {
  const badTier = await surveyFixture();
  badTier.families["15512"].backends.mlx.requestPeak.byTier = { fp8: "moves" };
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(badTier) } }),
    /requestPeak\.byTier contains a tier outside the matrix vocabulary/,
  );

  const badFinding = await surveyFixture();
  badFinding.families["15512"].backends.mlx.requestPeak.byTier = { q4: "phase-only" };
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { rung4Survey: JSON.stringify(badFinding) } }),
    /requestPeak\.byTier\.q4 has unknown finding/,
  );
});

// SC-16060 ------------------------------------------------------------------------------------
// `Verified` was a listed conformance state with nothing able to emit it: `strategyStatus` returns
// only Implemented/unverified, Structurally N/A and Missing, and the cell copied that verbatim. So
// the guard in `validateMatrix` was unreachable and the overlay test asserting "no cell is Verified"
// was green for the trivial reason that no cell COULD be. These tests exist to make that class of
// green impossible: every assertion below is paired with a mutation that must flip it.

const KREA_CONTROL_CELL =
  "krea_2_turbo:krea_2_turbo_control:mlx:q4:text_to_image:control:bounded_decode";

/// The shipped bundle re-stamped onto the current inference pin. Every shipped record is
/// `historical` — its inference revision predates the pin — so promotion, which is a property of
/// CURRENT evidence, has to be asserted against a re-stamped fixture rather than the shipped bundle.
///
/// `select` chooses which records are re-stamped, defaulting to Krea. It exists because the Qwen
/// rung-4 records ingested by sc-16353 were current at the `8ffa211a` pin and stopped being current
/// the moment sc-16962 moved it to `d4802320` — the fail-closed rule working as designed, and the
/// reason the two families are re-stamped separately: the geometry tests below depend on ONLY the
/// Krea cell moving relative to the shipped matrix, and [`qwenRung4OnCurrentPin`] must move exactly
/// the two retained q8 cells rather than every Qwen record in the bundle.
async function currentEvidenceFixture({
  keepGeometries = null,
  select = (record) => record.target.provider === "krea_2_turbo_control",
} = {}) {
  const [bundle, cargo, closureBody] = await Promise.all([
    readFile(new URL(`../${SOURCE_PATHS.calibrationEvidence}`, import.meta.url), "utf8"),
    readFile(new URL(`../${SOURCE_PATHS.cargo}`, import.meta.url), "utf8"),
    readFile(new URL(`../${SOURCE_PATHS.inferenceClosures}`, import.meta.url), "utf8"),
  ]);
  const pin = cargo.match(
    /candle-kernels\s*=\s*\{[^}]*?github\.com\/SceneWorks\/inference[^}]*?rev\s*=\s*"([0-9a-f]+)"/,
  )[1];
  // sc-17774: "current" means the provider's LIVE compile closure, not the pin. Stamping only the
  // revision — as this fixture used to — now makes a record no more current than it was, so the
  // tests built on it would assert promotion against evidence that never became eligible.
  const liveClosures = JSON.parse(closureBody).providers;
  const parsed = JSON.parse(bundle);
  parsed.records = parsed.records
    .filter(
      (record) =>
        !select(record) ||
        !keepGeometries ||
        keepGeometries.includes(`${record.target.geometry.width}x${record.target.geometry.height}`),
    )
    .map((record) => {
      if (!select(record)) return record;
      // `repositories` is part of the record's deterministic identity, so re-stamping the revision
      // without re-deriving the id produces a bundle the harness rejects outright. Recomputing it
      // through the real `recordId` keeps the fixture a VALID record rather than a shape that only
      // this test would accept.
      const restamped = {
        ...record,
        repositories: {
          ...record.repositories,
          inference: {
            ...record.repositories.inference,
            revision: pin,
            closureDigest:
              liveClosures[`${record.backend}:${record.target.provider}`]?.digest ??
              record.repositories.inference.closureDigest,
          },
        },
      };
      return { ...restamped, id: recordId(restamped) };
    });
  return JSON.stringify(parsed);
}

/// The inverse of [`currentEvidenceFixture`]: the shipped records re-stamped onto a SUPERSEDED
/// inference revision.
///
/// Before sc-16915 the shipped bundle was entirely historical, so "historical evidence does not
/// promote" could be asserted by simply reading the shipped matrix. Now that the evidence is
/// current, that half has to be produced deliberately — otherwise the negative claim would quietly
/// become untested the moment the positive one started holding.
/// The revision the superseded krea records were captured at.
const SUPERSEDED_KREA_REVISION = "96b13b6630132410a29ae1bcdf4d8738db7af28a";

/// The `mlx:krea_2_turbo_control` closure digest at [`SUPERSEDED_KREA_REVISION`], READ OUT of the
/// shipped bundle rather than transcribed (sc-17989).
///
/// It used to be a hardcoded 64-hex constant under a "derive it with `inference-closure-digest.mjs`"
/// comment, which nothing re-derived. A wrong value there does not fail — it makes every test built
/// on this fixture assert "historical evidence does not promote" for the wrong reason, because ANY
/// digest that is not the live one demotes. That is exactly how `820bf106…` hid in the sc-15833
/// FLUX.2 evidence one-shot's test through the whole of sc-17774 (deleted in sc-18100).
///
/// Reading it from the bundle inherits a real gate: `backfill-closure-digests.mjs --verify` in
/// `check.yml` re-derives every record digest from inference source at the revision it names, so a
/// value that disagrees with `96b13b66` fails CI at the source instead of silently here. The
/// not-live assertion pins the property the fixture actually needs on top of that.
function supersededKreaClosureDigest(parsed, liveClosures) {
  // `recordsNeedingDigest` is the gate's OWN eligibility predicate, not an approximation of it: it
  // is what `--verify` re-derives. Filtering on `(backend, provider, revision)` alone would read a
  // digest out of a record the gate skips — one demoted to `status: "superseded"` or a non-
  // authoritative scope — and the claim that CI grades this value would quietly stop being true.
  const digests = new Set(
    recordsNeedingDigest(parsed)
      .filter(
        (record) =>
          record.backend === "mlx" &&
          record.target.provider === "krea_2_turbo_control" &&
          record.repositories.inference.revision === SUPERSEDED_KREA_REVISION,
      )
      .map((record) => record.repositories.inference.closureDigest),
  );
  assert.equal(
    digests.size,
    1,
    `expected exactly one mlx:krea_2_turbo_control closure digest at ${SUPERSEDED_KREA_REVISION.slice(0, 8)}, ` +
      `found ${digests.size}. The fixture's superseded digest is read from these records.`,
  );
  const [digest] = digests;
  assert.notEqual(
    digest,
    liveClosures["mlx:krea_2_turbo_control"].digest,
    "the superseded fixture digest equals the LIVE one, so 'historical evidence does not promote' " +
      "would be asserting against evidence that is current. Re-point the fixture at a genuinely " +
      "superseded capture.",
  );
  return digest;
}

async function historicalEvidenceFixture({
  select = (record) => record.target.provider === "krea_2_turbo_control",
} = {}) {
  const [bundle, closureBody] = await Promise.all([
    readFile(new URL(`../${SOURCE_PATHS.calibrationEvidence}`, import.meta.url), "utf8"),
    readFile(new URL(`../${SOURCE_PATHS.inferenceClosures}`, import.meta.url), "utf8"),
  ]);
  const parsed = JSON.parse(bundle);
  const supersededDigest = supersededKreaClosureDigest(parsed, JSON.parse(closureBody).providers);
  parsed.records = parsed.records.map((record) => {
    if (!select(record)) return record;
    const restamped = {
      ...record,
      repositories: {
        ...record.repositories,
        // sc-17774: re-stamp onto a superseded CLOSURE, not a superseded pin. Both values come
        // from this bundle's own history (the krea capture at `96b13b66`), so the fixture still
        // exercises the exact comparison production performs — but the term that decides it is now
        // the provider's compile closure. Moving only the revision demotes nothing, by design.
        inference: {
          ...record.repositories.inference,
          revision: SUPERSEDED_KREA_REVISION,
          closureDigest: supersededDigest,
        },
      },
    };
    return { ...restamped, id: recordId(restamped) };
  });
  return JSON.stringify(parsed);
}

/// Re-stamp the manifest half of an exact calibration binding alongside a synthetic current record.
/// Production promotion requires both halves to name the linked inference runtime revision.
async function currentManifestCalibrationFixture({
  select = (binding) => binding.provider === "krea_2_turbo_control",
} = {}) {
  const [manifest, cargo, closureBody] = await Promise.all([
    readFile(new URL("../config/manifests/builtin.models.jsonc", import.meta.url), "utf8"),
    readFile(new URL(`../${SOURCE_PATHS.cargo}`, import.meta.url), "utf8"),
    readFile(new URL(`../${SOURCE_PATHS.inferenceClosures}`, import.meta.url), "utf8"),
  ]);
  const pin = cargo.match(
    /candle-kernels\s*=\s*\{[^}]*?github\.com\/SceneWorks\/inference[^}]*?rev\s*=\s*"([0-9a-f]+)"/,
  )[1];
  // sc-17774 moved currency off the pin and onto the provider's compile closure. The evidence
  // fixture was migrated with it; this one was not, so it kept stamping only the revision. A record
  // carrying a LIVE closure digest can never match a binding still carrying a stale one, which made
  // every fixture built here un-promotable the moment any provider's closure moved — regardless of
  // whether the provider under test was the one that moved.
  const liveClosures = JSON.parse(closureBody).providers;
  const parsed = JSON.parse(stripJsoncComments(manifest));
  for (const model of parsed.models) {
    for (const backend of ["candle", "mlx"]) {
      for (const binding of model[backend]?.calibrations ?? []) {
        if (!select(binding)) continue;
        binding.inferenceRevision = pin;
        binding.inferenceClosureDigest =
          liveClosures[`${backend}:${binding.provider}`]?.digest ?? binding.inferenceClosureDigest;
      }
    }
  }
  return JSON.stringify(parsed);
}

/// Qwen records re-stamped current: the retained q8 rung-4 fixture, plus SC-18353's physical
/// captures.
///
/// Selected by their calibration fingerprint, which is what separates them from the 22 older Qwen
/// records in the bundle: the rung-4 ingest and SC-18237's production-deferred pair carry the bare
/// `qwen-image-mlx-shared-ladder-2026-08-01-v1`, while the earlier captures carry the `-eager` /
/// `-deferred` load-shape variants. Q4/BF16 are deliberately not re-stamped: their physical source
/// sessions bind the superseded closure and cannot truthfully be made current by a synthetic fixture.
const QWEN_RUNG4_FINGERPRINT = "qwen-image-mlx-shared-ladder-2026-08-01-v1";
const QWEN_PRODUCTION_DEFERRED_REVISION = "014134e3035ad7e4eca5c2ed7bded2375dc3c071";
const qwenRung4OnCurrentPin = () =>
  currentEvidenceFixture({
    select: (record) =>
      record.target.provider === "qwen_image" &&
      (record.sourceProvenance === "physical_mlx_v1" ||
        (record.target.tier === "q8" &&
          record.calibrationFingerprint === QWEN_RUNG4_FINGERPRINT)),
  });

test("current evidence promotes a cell to Verified, and historical evidence does not (sc-16060)", async () => {
  const shipped = await buildMatrix({ publish: false });
  const shippedCell = shipped.cells.find((cell) => cell.id === KREA_CONTROL_CELL);
  // sc-16915 re-collected both families at the then-current pin, so the POSITIVE half of this test
  // was briefly observable on the shipped matrix. It has gone back to `Implemented/unverified`, and
  // correctly: under sc-17774 currency is the provider's compile closure, and `mlx:krea_2_turbo_control`
  // moved (records carry d355971e/3064f675, live is cbb83ed8) because two commits in this pin's
  // window touched the shared `crates/media/mlx-gen` crate, which is a first-party dependency of
  // every MLX provider's closure. Re-capturing is the Krea calibration story's work, not this PR's.
  //
  // The POSITIVE half is not lost — it is asserted below against `promotedQwen`, built from a fixture
  // stamped with live closure digests. That is the stronger arrangement anyway: it stays meaningful
  // whether or not the shipped bundle happens to be current.
  assert.equal(
    shippedCell.state,
    "Implemented/unverified",
    "the shipped Krea records were captured against a superseded provider closure",
  );
  assert.equal(
    shippedCell.evidence.currentEnvironmentVerification.length,
    0,
    "a cell with no current evidence must carry no current-environment verification",
  );
  const verifiedQwen = (matrix) =>
    matrix.cells.filter(
      (cell) => cell.modelId === "qwen_image" && cell.backend === "mlx" && cell.state === "Verified",
    ).length;

  const qwenOnCurrentPin = await qwenRung4OnCurrentPin();
  const qwenManifestOnCurrentPin = await currentManifestCalibrationFixture({
    select: (binding) => binding.provider === "qwen_image",
  });
  const promotedQwen = await buildMatrix({
    publish: false,
    sourceOverrides: {
      calibrationEvidence: qwenOnCurrentPin,
      manifest: qwenManifestOnCurrentPin,
    },
  });
  assert.equal(
    verifiedQwen(promotedQwen),
    9,
    "every re-stamped Qwen binding must verify once evidence and manifest share the live closure",
  );
  // sc-19721 re-captured at pin 75d66db5, so the shipped bundle verifies all nine bindings on its
  // own now. The captures stayed historical until a real re-capture arrived rather than being
  // re-stamped, which is the property this protects.
  assert.equal(
    verifiedQwen(shipped),
    0,
    "SC-18311 moves the Qwen provider closure, so every shipped capture is historical until recaptured",
  );

  const verifiedZ = (matrix) =>
    matrix.cells.filter(
      (cell) => cell.modelId === "z_image_turbo" && cell.backend === "mlx" &&
        cell.state === "Verified",
    ).length;
  // The shipped Z-Image ladder is an accepted floor now: its capture closure was superseded by the
  // pin advance, so nothing promotes from the checked-in artifact.
  assert.equal(
    verifiedZ(shipped),
    0,
    "a Z-Image ladder whose capture closure was superseded must not ship as Verified",
  );
  // ...which would leave the mismatched-binding control below comparing 0 against 0. Re-establish a
  // real positive first: with BOTH evidence and manifest binding re-stamped onto the live closure,
  // all five rungs promote. That is the baseline the mismatch must then destroy.
  const promotedZ = await buildMatrix({
    publish: false,
    sourceOverrides: {
      calibrationEvidence: await currentEvidenceFixture({
        select: (record) => record.target.provider === "z_image_turbo",
      }),
      manifest: await currentManifestCalibrationFixture({
        select: (binding) => binding.provider === "z_image_turbo",
      }),
    },
  });
  assert.equal(
    verifiedZ(promotedZ),
    5,
    "on a shared live closure every Z-Image rung must still promote — the rule is intact, only the shipped capture is stale",
  );

  const manifestWithMismatchedZBinding = JSON.parse(stripJsoncComments(await readFile(
    new URL("../config/manifests/builtin.models.jsonc", import.meta.url),
    "utf8",
  )));
  for (const binding of manifestWithMismatchedZBinding.models
    .find((model) => model.id === "z_image_turbo").mlx.calibrations) {
    binding.inferenceClosureDigest = "0".repeat(64);
  }
  const mismatchedBindingZ = await buildMatrix({
    publish: false,
    sourceOverrides: {
      manifest: JSON.stringify(manifestWithMismatchedZBinding),
    },
  });
  assert.equal(
    verifiedZ(mismatchedBindingZ),
    0,
    "current evidence cannot promote through a manifest binding with a different closure identity",
  );

  // sc-17774: moving the PIN must no longer demote anything — that was the whole defect. Moving
  // THIS provider's compile closure must, and moving another provider's must not. All three
  // directions are asserted, because a unit that absolved everything would be indistinguishable
  // from a broken one.
  const closures = JSON.parse(
    await readFile(new URL(`../${SOURCE_PATHS.inferenceClosures}`, import.meta.url), "utf8"),
  );
  const cargo = await readFile(new URL(`../${SOURCE_PATHS.cargo}`, import.meta.url), "utf8");
  const withPin = (pin) =>
    cargo.replace(
      /(candle-kernels\s*=\s*\{[^}]*?github\.com\/SceneWorks\/inference[^}]*?rev\s*=\s*")[0-9a-f]+(")/,
      `$1${pin}$2`,
    );
  const movedPin = "0".repeat(40);
  const movedMemoryContractSources = await memoryContractPinOverrides(movedPin);
  const withClosures = (mutate) => {
    const next = structuredClone(closures);
    next.inferenceRevision = movedPin;
    mutate(next.providers);
    return JSON.stringify(next, null, 2);
  };
  // sc-19542: the rung-4 prerequisite records are keyed to the pin the same way the closures are, so
  // a moved-pin fixture has to re-key both or generation fails on the stale-config guard before it
  // reaches the currency question this test is about.
  const prerequisitesOnMovedPin = JSON.stringify({
    ...(await prerequisitesFixture()),
    inferenceRevision: movedPin,
  });

  const pinOnlyQwen = await buildMatrix({
    publish: false,
    sourceOverrides: {
      calibrationEvidence: qwenOnCurrentPin,
      manifest: qwenManifestOnCurrentPin,
      cargo: withPin(movedPin),
      rung4ContractPrerequisites: prerequisitesOnMovedPin,
      inferenceClosures: withClosures(() => {}),
      ...movedMemoryContractSources,
    },
  });
  assert.equal(
    verifiedQwen(pinOnlyQwen),
    9,
    "a pin move that leaves Qwen's compile closure alone must NOT demote its retained measurements",
  );

  const otherProviderMoved = await buildMatrix({
    publish: false,
    sourceOverrides: {
      calibrationEvidence: qwenOnCurrentPin,
      manifest: qwenManifestOnCurrentPin,
      cargo: withPin(movedPin),
      rung4ContractPrerequisites: prerequisitesOnMovedPin,
      inferenceClosures: withClosures((providers) => {
        providers["mlx:z_image_turbo"].digest = "f".repeat(64);
      }),
      ...movedMemoryContractSources,
    },
  });
  assert.equal(
    verifiedQwen(otherProviderMoved),
    9,
    "another model's code path moving must never demote retained Qwen evidence",
  );

  const staleQwen = await buildMatrix({
    publish: false,
    sourceOverrides: {
      calibrationEvidence: qwenOnCurrentPin,
      manifest: qwenManifestOnCurrentPin,
      cargo: withPin(movedPin),
      rung4ContractPrerequisites: prerequisitesOnMovedPin,
      inferenceClosures: withClosures((providers) => {
        providers["mlx:qwen_image"].digest = "e".repeat(64);
      }),
      ...movedMemoryContractSources,
    },
  });
  assert.equal(
    verifiedQwen(staleQwen),
    0,
    "Qwen's OWN compile closure moving must fail closed instead of carrying verification forward",
  );

  // MUTATION, and the negative half of this test's title. The SAME Krea records must verify their
  // cell when their closure is live and stop verifying it when it is superseded. Asserting only one
  // direction would pass on a generator that promoted unconditionally — precisely the failure the
  // `Verified` state exists to rule out.
  //
  // Both halves now run against FIXTURES rather than against the shipped bundle, because the shipped
  // bundle's currency is not a stable thing to assert on. The mutation has already gone inert twice
  // for that reason: once when the evidence was re-collected at the pin (the promote-direction became
  // a no-op), and again once `mlx:krea_2_turbo_control`'s closure moved (the demote-direction became
  // a no-op, since the shipped cell was already `Implemented/unverified`). Anchoring the baseline to
  // a promoted fixture makes the pair meaningful regardless of what the shipped bundle happens to be.
  const promotedKrea = await buildMatrix({
    publish: false,
    sourceOverrides: {
      calibrationEvidence: await currentEvidenceFixture(),
      manifest: await currentManifestCalibrationFixture(),
    },
  });
  const promotedKreaCell = promotedKrea.cells.find((cell) => cell.id === KREA_CONTROL_CELL);
  assert.equal(
    promotedKreaCell.state,
    "Verified",
    "records carrying the live provider closure must verify their cell",
  );
  assert.ok(
    promotedKreaCell.evidence.currentEnvironmentVerification.length > 0,
    "a Verified cell must carry the dynamic evidence its guard requires",
  );

  const demoted = await buildMatrix({
    publish: false,
    sourceOverrides: { calibrationEvidence: await historicalEvidenceFixture() },
  });
  const demotedCell = demoted.cells.find((cell) => cell.id === KREA_CONTROL_CELL);
  assert.equal(
    demotedCell.state,
    "Implemented/unverified",
    "records re-dated onto a superseded pin must stop verifying their cell",
  );
  assert.equal(
    demotedCell.evidence.currentEnvironmentVerification.length,
    0,
    "a demoted cell must carry no current-environment verification",
  );

  // Demotion is scoped to the cells those records bind. Nothing else may move.
  const movedIds = demoted.cells
    .filter(
      (cell) => cell.state !== promotedKrea.cells.find((other) => other.id === cell.id).state,
    )
    .map((cell) => cell.id);
  assert.deepEqual(movedIds, [KREA_CONTROL_CELL]);
});

test("Verified never implies geometry coverage — one point certifies one point (sc-16060)", async () => {
  // The story's central case: a record whose geometry is INSIDE the envelope but far from what the
  // cell will be asked to render. Under the old single-field model this cell read as certified
  // across all 17 advertised resolutions on the strength of one 768x768 capture.
  const single = await buildMatrix({
    publish: false,
    sourceOverrides: {
      calibrationEvidence: await currentEvidenceFixture({ keepGeometries: ["768x768"] }),
      manifest: await currentManifestCalibrationFixture(),
    },
  });
  const cell = single.cells.find((candidate) => candidate.id === KREA_CONTROL_CELL);

  assert.equal(cell.state, "Verified", "one current record establishes the implementation claim");
  assert.equal(cell.memoryCharacterization.status, "point");
  assert.deepEqual(cell.memoryCharacterization.measuredGeometries, ["768x768"]);
  assert.equal(
    cell.memoryCharacterization.coveredPixelBound,
    null,
    "one point determines no curve, so it bounds no coverage",
  );

  // The envelope is genuinely much wider than the measurement — otherwise this test would pass for
  // the uninteresting reason that there was nothing to over-claim.
  const measured = 768 * 768;
  const widest = Math.max(
    ...cell.geometryEnvelope.resolutions.map((resolution) => {
      const [width, height] = resolution.split("x").map(Number);
      return width * height;
    }),
  );
  assert.ok(
    widest >= measured * 4,
    `envelope must dwarf the measurement for this to mean anything (widest ${widest}, measured ${measured})`,
  );
});

test("a second geometry is what makes a curve determinable (sc-16060)", async () => {
  const fitted = await buildMatrix({
    publish: false,
    sourceOverrides: {
      calibrationEvidence: await currentEvidenceFixture({
        keepGeometries: ["768x768", "896x896"],
      }),
      manifest: await currentManifestCalibrationFixture(),
    },
  });
  const fittedCell = fitted.cells.find((cell) => cell.id === KREA_CONTROL_CELL);
  assert.equal(fittedCell.memoryCharacterization.status, "fitted");
  assert.deepEqual(fittedCell.memoryCharacterization.measuredGeometries, ["768x768", "896x896"]);
  assert.equal(fittedCell.memoryCharacterization.coveredPixelBound, 896 * 896);

  // MUTATION: drop one geometry. `fitted` must fall back to `point` and surrender its bound — a
  // status that survived losing half its evidence would be asserting nothing.
  const single = await buildMatrix({
    publish: false,
    sourceOverrides: {
      calibrationEvidence: await currentEvidenceFixture({ keepGeometries: ["896x896"] }),
      manifest: await currentManifestCalibrationFixture(),
    },
  });
  const singleCell = single.cells.find((cell) => cell.id === KREA_CONTROL_CELL);
  assert.equal(singleCell.memoryCharacterization.status, "point");
  assert.equal(singleCell.memoryCharacterization.coveredPixelBound, null);
  assert.equal(
    singleCell.state,
    "Verified",
    "losing a geometry costs the curve, never the implementation claim",
  );
});

test("the shipped Krea tiers report their recovered two-point fits (sc-16514)", async () => {
  // SC-16514 recovered the 768² q8/bf16 captures into the manifest records, so the generated claim
  // must promote all three tier curves together. Dropping either recovered record mutates this back
  // to `point` through the general one-vs-two-point test above.
  const matrix = await buildMatrix({ publish: false });
  const status = (tier) =>
    matrix.cells.find(
      (cell) => cell.id === `krea_2_turbo:krea_2_turbo:candle:${tier}:text_to_image:none:bounded_decode`,
    ).memoryCharacterization.status;

  assert.equal(status("q4"), "fitted", "q4 carries 768x768 and 1024x1024");
  assert.equal(status("q8"), "fitted", "q8 carries 768x768 and 1024x1024");
  assert.equal(status("bf16"), "fitted", "bf16 carries 768x768 and 1024x1024");
});

test("promotion is available only to an implemented cell (sc-16060)", async () => {
  const matrix = await buildMatrix({
    publish: false,
    sourceOverrides: {
      calibrationEvidence: await currentEvidenceFixture(),
      manifest: await currentManifestCalibrationFixture(),
    },
  });
  for (const cell of matrix.cells) {
    if (cell.state === "Verified") {
      assert.ok(
        cell.evidence.currentEnvironmentVerification.length > 0,
        `${cell.id}: Verified without current evidence`,
      );
    }
    // Neither of the two non-implementation states may carry measured geometry: there is no code to
    // verify and nothing to measure, so evidence reaching one of them means a record bound a cell it
    // should not have.
    if (cell.state === "Missing" || cell.state === "Structurally N/A") {
      assert.equal(
        cell.memoryCharacterization.status,
        "unmeasured",
        `${cell.id}: ${cell.state} cell carries measured geometry`,
      );
    }
  }
});

test("every conformance and characterization state carries a definition (sc-16060)", async () => {
  // The defect underneath this story: `conformanceStates` was a bare string list, so `Verified` had
  // no definition anywhere in the artifact, the generator, or the docs — and two parts of the
  // pipeline answered "does one geometry certify the envelope?" differently without either being
  // wrong about a rule that had never been written down.
  const matrix = await buildMatrix({ publish: false });
  const states = new Set(matrix.cells.map((cell) => cell.state));
  for (const state of states) {
    const declared = matrix.conformanceStates.find((entry) => entry.state === state);
    assert.ok(declared, `${state} appears on a cell but is not declared`);
    assert.ok(declared.definition.length > 20, `${state} has no usable definition`);
  }
  for (const status of new Set(matrix.cells.map((cell) => cell.memoryCharacterization.status))) {
    const declared = matrix.memoryCharacterizationStates.find((entry) => entry.status === status);
    assert.ok(declared, `${status} appears on a cell but is not declared`);
    assert.ok(declared.definition.length > 20, `${status} has no usable definition`);
  }
  // The two claims are declared as what they are: one geometry-sensitive, one not. Collapsing them
  // is what produced this story.
  assert.equal(matrix.claims.state.geometrySensitive, false);
  assert.equal(matrix.claims.memoryCharacterization.geometrySensitive, true);
});

// ── sc-18099: publication ──────────────────────────────────────────────────────────────────────
//
// Everything above asserts GENERATION and reads `publish: false`, because most coordinates are
// elided and asserting them against the published subset would quietly hollow out thirteen
// behavioural tests. These assert the publication step itself, and are the only tests here that read
// the published document.

test("publication keeps every planned, measured, bound and cited coordinate — and nothing else", async () => {
  const resolved = await buildMatrix({ publish: false });
  const publishedDocument = await buildMatrix();
  const plan = activeCalibrationPlan(JSON.parse(
    await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url), "utf8"),
  ));

  const planned = plannedCellIds(plan, resolved.cells);
  assert.ok(planned.size > 100, "the shipped plan must target a substantial set of coordinates");
  const calibrationRunCellIds = new Set(resolved.calibrationRuns.map((run) => run.cellId));
  assert.ok(calibrationRunCellIds.size > 0);

  const expected = resolved.cells
    .filter((cell) => isPublishableCell(cell, { plannedCellIds: planned, calibrationRunCellIds }))
    .map((cell) => cell.id);
  assert.deepEqual(publishedDocument.cells.map((cell) => cell.id), expected);

  // Both directions of the predicate, spelled out rather than inferred from the set equality above:
  // an arm that silently stopped matching would still produce a self-consistent document.
  const publishedIds = new Set(expected);
  for (const cell of resolved.cells) {
    const reasons = [
      planned.has(cell.id),
      calibrationRunCellIds.has(cell.id),
      cell.memoryCharacterization.status !== "unmeasured",
      cell.evidence.historicalVerification.length > 0,
      cell.evidence.currentEnvironmentVerification.length > 0,
      cell.evidence.strategyParameterVerification.length > 0,
      cell.evidence.structural.length > 0,
    ];
    assert.equal(
      publishedIds.has(cell.id),
      reasons.some(Boolean),
      `${cell.id}: publication disagrees with the stated predicate`,
    );
  }

  // Six of the seven arms must actually carry cells of their own. An arm that admitted nothing would
  // be a dead clause, and the predicate would then mean something narrower than it says.
  for (const [name, arm] of [
    ["planned", (cell) => planned.has(cell.id)],
    ["bound to a record", (cell) => calibrationRunCellIds.has(cell.id)],
    ["measured", (cell) => cell.memoryCharacterization.status !== "unmeasured"],
    ["historical evidence", (cell) => cell.evidence.historicalVerification.length > 0],
    ["strategy parameters", (cell) => cell.evidence.strategyParameterVerification.length > 0],
    ["structural evidence", (cell) => cell.evidence.structural.length > 0],
  ]) {
    assert.ok(resolved.cells.some(arm), `the "${name}" arm admits no coordinate at all`);
  }

  // The seventh arm, `currentEnvironmentVerification`, admits NOTHING at the sc-20523 pin: it moved
  // every provider closure past every retained capture — SC-18218's and sc-19721's FLUX.2 cohorts,
  // the Qwen ladder re-capture, and SC-19753's Z-Image rungs alike — so every capture is historical
  // until the bump-time re-capture. Two facts keep this assertion useful:
  //
  //   1. It is exact: NO row may survive the closure change as current — a record silently keeping
  //      currency across a pin bump is the failure this pins.
  //   2. It is SUBSUMED. A current run is an eligible run, and `memoryCharacterization` counts every
  //      eligible run's geometry, so a cell carrying current evidence is `point` or `fitted` and the
  //      measured arm already admits it. The arm therefore cannot uniquely admit or elide anything.
  //
  // Asserted as an exact set so a recapture flips this test rather than silently passing, and
  // the field's presence is asserted separately so a rename cannot make the arm quietly vanish.
  assert.deepEqual(
    resolved.cells
      .filter((cell) => cell.evidence.currentEnvironmentVerification.length > 0)
      .map((cell) => cell.id)
      .sort(),
    [],
    "no retained capture may carry current evidence at a pin whose closures have moved past it; a historical row surviving as current is the failure this pins",
  );
  assert.ok(
    resolved.cells.every((cell) => Array.isArray(cell.evidence.currentEnvironmentVerification)),
    "the arm's field must exist on every cell, or a rename would silently retire it",
  );
  assert.ok(
    resolved.cells.every(
      (cell) =>
        cell.evidence.currentEnvironmentVerification.length === 0 ||
        cell.memoryCharacterization.status !== "unmeasured",
    ),
    "current evidence must imply measured geometry, which is what makes the empty arm harmless",
  );

  // A Structurally N/A verdict may never be elided: an absent coordinate reads as "nobody has done
  // this yet", and that verdict says the opposite.
  const exempt = resolved.cells.filter((cell) => cell.state === "Structurally N/A");
  assert.ok(exempt.length > 0);
  assert.ok(exempt.every((cell) => publishedIds.has(cell.id)));

  // And a bare route declaration is NOT a reason on its own — otherwise the slim would republish the
  // cross-product under another name.
  assert.ok(
    resolved.cells.some((cell) => isImplemented(cell.state) && !publishedIds.has(cell.id)),
    "an implemented-but-unplanned, unmeasured, uncited coordinate must still be elided",
  );
});

test("the elision is counted, never silent (sc-18099)", async () => {
  const resolved = await buildMatrix({ publish: false });
  const matrix = await buildMatrix();

  assert.equal(matrix.summary.cells, resolved.cells.length, "`cells` stays the resolved total");
  assert.equal(matrix.summary.publishedCells, matrix.cells.length);
  assert.equal(matrix.summary.publishedCells + matrix.summary.elidedCells, matrix.summary.cells);
  assert.ok(matrix.summary.elidedCells > 0, "this artifact does elide, so the count must be real");
  assert.equal(
    Object.values(matrix.summary.elidedByState).reduce((total, count) => total + count, 0),
    matrix.summary.elidedCells,
  );
  assert.match(matrix.summary.publicationPredicate, /PLANNED/);

  // The census answers every coverage question the cross-product used to, including for lanes that
  // published nothing at all.
  const censusStates = new Map();
  for (const row of matrix.coverage) {
    for (const [state, count] of Object.entries(row.states)) {
      censusStates.set(state, (censusStates.get(state) ?? 0) + count);
    }
  }
  const resolvedStates = new Map();
  for (const cell of resolved.cells) {
    resolvedStates.set(cell.state, (resolvedStates.get(cell.state) ?? 0) + 1);
  }
  assert.deepEqual([...censusStates].sort(), [...resolvedStates].sort());
  assert.equal(
    matrix.coverage.reduce((total, row) => total + row.implemented, 0),
    resolved.cells.filter((cell) => isImplemented(cell.state)).length,
  );

  // A lane the slim published nothing from is still fully described. Without this the artifact would
  // report absence for a lane that is merely unmeasured — the sc-16069 failure, reintroduced.
  const silent = matrix.coverage.filter((row) => row.published === 0);
  assert.ok(silent.length > 0, "some lane must publish nothing, or this asserts nothing");
  for (const row of silent) {
    assert.equal(row.elided, row.coordinates);
    const model = matrix.models.find((entry) => entry.id === row.modelId);
    assert.ok(model.axes[row.backend].rungs.includes(row.rung));
  }

  // `mlxStagedStaticCoverage` is a claim about all 53 IMAGE entries and must not shrink to the
  // published sample — nor (sc-18815) grow by absorbing the video lane. The video `bernini` entry
  // shares the `bernini` engine with the already-counted `bernini_image`, so a modality-blind
  // recount reads 39/53 for a lane that gained nothing; the two censuses are kept apart here for the
  // same reason the generator keeps them apart.
  const imageIds = new Set(
    matrix.models.filter((model) => model.modality === "image").map((model) => model.id),
  );
  const stagedFromCensus = new Set(
    matrix.coverage
      .filter(
        (row) =>
          row.backend === "mlx" &&
          row.rung === "staged_residency" &&
          row.implemented > 0 &&
          imageIds.has(row.modelId),
      )
      .map((row) => row.modelId),
  );
  assert.equal(stagedFromCensus.size, matrix.summary.mlxStagedStaticCoverage);
  assert.equal(
    matrix.coverage.filter(
      (row) =>
        row.backend === "mlx" &&
        row.rung === "staged_residency" &&
        row.implemented > 0 &&
        !imageIds.has(row.modelId),
    ).length,
    matrix.summary.videoMlxStagedStaticCoverage,
  );
  assert.equal(
    stagedFromCensus.size,
    new Set(
      resolved.cells
        .filter(
          (cell) =>
            cell.backend === "mlx" &&
            cell.rung === "staged_residency" &&
            isImplemented(cell.state) &&
            imageIds.has(cell.modelId),
        )
        .map((cell) => cell.modelId),
    ).size,
  );
});

// sc-18100: rehomed from `calibration-cost-model.test.mjs`, the only place that ever re-derived the
// published headline counts from raw inputs. The generator computes `currentCalibrationRuns` by
// summing `cell.evidence.currentEnvironmentVerification`; this recomputes it straight from the
// evidence bundle and the closure ledger, so a promotion bug in that path cannot agree with itself.
// The sc-17774 rule under test: a record authorizes the live runtime only while its provider's
// compile closure is unchanged — the pin is not the term.
test("the published summary re-derives from the evidence bundle and the closure ledger (sc-17774)", async () => {
  const root = new URL("../", import.meta.url);
  const matrix = JSON.parse(await readFile(new URL("docs/generated/memory-matrix.json", root)));
  const evidence = JSON.parse(
    await readFile(new URL("docs/generated/memory-calibration-evidence.json", root)),
  );
  const closures = JSON.parse(
    await readFile(new URL("config/inference-provider-closures.json", root)),
  );

  const tally = { complete: 0, runtimeComplete: 0 };
  for (const record of evidence.records) {
    if (record.status === "complete") tally.complete += 1;
    if (record.status === "runtime_complete") tally.runtimeComplete += 1;
  }
  assert.deepEqual(matrix.summary.calibrationRunsByStatus, tally);
  assert.equal(matrix.summary.calibrationRuns, tally.complete + tally.runtimeComplete);

  // `binding.eligible` is the generator's own coordinate match; everything else here is recomputed.
  const eligibleByRecord = new Map(
    matrix.calibrationRuns.map(({ binding, record }) => [record.id, binding.eligible]),
  );
  const currentRuns = (providers) => evidence.records.filter((record) => {
    if (!["complete", "runtime_complete"].includes(record.status)) return false;
    if (["fixture", "candidate"].includes(record.evidenceScope)) return false;
    if (!eligibleByRecord.get(record.id)) return false;
    const lane = `${record.backend}:${record.target.provider}`;
    return record.repositories.inference.closureDigest === providers[lane]?.digest;
  }).length;

  assert.equal(matrix.summary.currentCalibrationRuns, currentRuns(closures.providers));

  // The equality above is only as good as the recomputation being a live derivation. Today the
  // corpus is fully fail-closed (every eligible record sits at a superseded closure, so both sides
  // are 0) and a hardwired `0` would satisfy it. Prove sensitivity in BOTH directions against two
  // synthetic ledgers, so the proof holds whatever the real corpus currently says.
  //
  // Deliberately NOT compared against `matrix.summary.currentCalibrationRuns`: refreshing every lane
  // to the live closure is the stated end state of epic 18093, and at that point the as-captured
  // ledger and the live ledger agree. A strict `>` against the live count would red on that entirely
  // correct state — the same "fails on correct change" defect that disqualified the census literals
  // this test replaced.
  const eligibleRecords = evidence.records.filter((record) =>
    ["complete", "runtime_complete"].includes(record.status) &&
    !["fixture", "candidate"].includes(record.evidenceScope) &&
    eligibleByRecord.get(record.id),
  );
  assert.ok(eligibleRecords.length > 0, "the corpus must hold eligible records to prove anything");

  // Admits: a ledger stamped with the records' own captured digests makes eligible records current.
  // Not an equality against `eligibleRecords.length` — the ledger is keyed by LANE, and a lane whose
  // records were captured at more than one closure collapses to whichever digest is written last, so
  // some eligible records legitimately stay non-current here (31 of 35 today). The claim that has to
  // hold is that the count is driven off zero at all.
  const asCaptured = Object.fromEntries(eligibleRecords.map((record) => [
    `${record.backend}:${record.target.provider}`,
    { digest: record.repositories.inference.closureDigest },
  ]));
  assert.ok(
    currentRuns(asCaptured) > 0,
    "a ledger matching the captured digests must make eligible records current",
  );

  // Refuses: a ledger where no digest can match makes nothing current. Together with the admitting
  // case this pins the comparison itself, not the count it happens to produce today.
  const allSuperseded = Object.fromEntries(
    Object.keys(closures.providers).map((lane) => [lane, { digest: "0".repeat(64) }]),
  );
  assert.equal(currentRuns(allSuperseded), 0);
});

test("the published document is closed: no reference outlives its row (sc-18099)", async () => {
  const matrix = await buildMatrix();
  const publishedIds = new Set(matrix.cells.map((cell) => cell.id));

  assert.ok(matrix.calibrationRuns.length > 0);
  for (const run of matrix.calibrationRuns) {
    assert.ok(publishedIds.has(run.cellId), `${run.record.id}: dangling cellId ${run.cellId}`);
    assert.match(run.record.source, /^docs\/generated\/memory-calibration-evidence\.json#/);
    assert.ok(run.record.source.endsWith(run.record.id));
  }
  for (const [modelId, slice] of Object.entries(matrix.modelSlices)) {
    for (const id of slice) assert.ok(publishedIds.has(id), `${modelId}: dangling slice id ${id}`);
  }
  assert.deepEqual(
    Object.keys(matrix.modelSlices).sort(),
    matrix.models.map((model) => model.id).sort(),
    "every entry keeps a slice, even an empty one — a missing key would read as a missing entry",
  );

  // The closure guard is a real gate, not a formality: a dangling reference must fail generation.
  assert.throws(
    () =>
      assertPublishedDocumentIsClosed(
        {
          ...matrix,
          cells: matrix.cells.filter((cell) => cell.id !== matrix.calibrationRuns[0].cellId),
        },
        matrix.summary.cells,
      ),
    /calibration run names cell .* which the slim did not publish/,
  );
  assert.throws(
    () =>
      assertPublishedDocumentIsClosed(
        { ...matrix, coverage: matrix.coverage.slice(1) },
        matrix.summary.cells,
      ),
    /coverage census covers .* but the catalog resolved/,
  );

  // The census breakdown is gated too, in both directions: a marginal that no longer accounts for
  // the count it explains, and a breakdown present or absent on the wrong kind of row.
  const mixedIndex = matrix.coverage.findIndex((row) => Object.hasOwn(row, "implementedBy"));
  assert.ok(mixedIndex >= 0);
  const withBadMarginal = matrix.coverage.map((row, index) =>
    index === mixedIndex
      ? { ...row, implementedBy: { ...row.implementedBy, overlay: { none: row.implemented + 1 } } }
      : row,
  );
  assert.throws(
    () => assertPublishedDocumentIsClosed({ ...matrix, coverage: withBadMarginal }, matrix.summary.cells),
    /implementedBy\.overlay sums to/,
  );
  const withoutBreakdown = matrix.coverage.map((row, index) => {
    if (index !== mixedIndex) return row;
    const { implementedBy, ...rest } = row;
    return rest;
  });
  assert.throws(
    () => assertPublishedDocumentIsClosed({ ...matrix, coverage: withoutBreakdown }, matrix.summary.cells),
    /implementedBy must be present on exactly the rows/,
  );

  // ...and so is the manifest-scope join, from both ends.
  const orphanCell = matrix.cells.map((cell, index) =>
    index === 0 ? { ...cell, evidence: { ...cell.evidence, manifestScope: "not:a:scope" } } : cell,
  );
  assert.throws(
    () => assertPublishedDocumentIsClosed({ ...matrix, cells: orphanCell }, matrix.summary.cells),
    /evidence\.manifestScope not:a:scope is not published/,
  );
  assert.throws(
    () =>
      assertPublishedDocumentIsClosed(
        { ...matrix, manifestScopes: { ...matrix.manifestScopes, "orphan:mlx:q4": { declaredCalibration: [], loadability: [] } } },
        matrix.summary.cells,
      ),
    /manifestScopes\.orphan:mlx:q4 is referenced by no published cell/,
  );
});

test("the published axes are the cross-product, so no lane can be invisible (sc-18099)", async () => {
  const resolved = await buildMatrix({ publish: false });
  const matrix = await buildMatrix();

  for (const model of matrix.models) {
    for (const [backend, axes] of Object.entries(model.axes)) {
      const lane = resolved.cells.filter(
        (cell) => cell.modelId === model.id && cell.backend === backend,
      );
      assert.deepEqual(axes.tiers, [...new Set(lane.map((cell) => cell.tier))].sort());
      assert.deepEqual(axes.modes, [...new Set(lane.map((cell) => cell.mode))].sort());
      assert.deepEqual(axes.overlays, [...new Set(lane.map((cell) => cell.overlay))].sort());
      assert.equal(
        axes.tiers.length * axes.modes.length * axes.overlays.length * axes.rungs.length,
        lane.length,
      );
    }
  }

  // The concrete case: InstantID's Candle lane resolves to bf16 only, and publishes no cell at all.
  // Reading its tiers off `cells` would now return the empty set.
  const instantId = matrix.models.find((model) => model.id === "instantid_realvisxl");
  assert.deepEqual(instantId.axes.candle.tiers, ["bf16"]);
  assert.equal(
    matrix.cells.filter(
      (cell) => cell.modelId === "instantid_realvisxl" && cell.backend === "candle",
    ).length,
    0,
  );
});

test("a calibration-plan entry that addresses no coordinate fails generation (sc-18099)", async () => {
  // The defect: nine sc-15817 entries carried `mode: "edit"` while the catalog's mode axis spells
  // that capability `edit_image`, so they matched ZERO coordinates. Nothing caught it —
  // `expectedEngagedRungs` just returned null, and `memory-calibration.schema.json` types `mode` as a
  // free string — and a capture run against them would have produced records binding to nothing.
  const rawPlan = JSON.parse(
    await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url), "utf8"),
  );
  const retiredStyleRows = rawPlan.providers.filter(
    (entry) => entry.target.mode === "style_variations",
  );
  assert.ok(retiredStyleRows.length > 0, "historical style captures remain provenance");
  const plan = activeCalibrationPlan(rawPlan);
  assert.ok(
    plan.providers.every((entry) => entry.target.mode !== "style_variations"),
    "retired product modes are excluded from the active calibration plan",
  );

  // The shipped plan is clean, which is the property worth pinning: every entry addresses something
  // — or names a family the matrix cannot carry a coordinate for at all (sc-18663), which is the
  // set the generator itself exempts and is read here from the same parse the generator reads.
  const resolved = await buildMatrix({ publish: false });
  const outOfMatrixEntries = shippedOutOfMatrixEntries(await readFile(SURVEY_URL, "utf8"));
  assert.doesNotThrow(() =>
    assertCalibrationPlanTargetsResolvedCoordinates(plan, resolved.cells, { outOfMatrixEntries }),
  );

  // And the guard discriminates, on each axis independently — a guard that only caught `mode` would
  // be a fix for one typo rather than for the class.
  for (const [axis, mutate] of [
    ["mode", (entry) => { entry.target.mode = "edit"; }],
    ["tier", (entry) => { entry.target.tier = "fp8"; }],
    ["modelId", (entry) => { entry.target.modelId = "not_a_catalog_entry"; }],
    ["provider", (entry) => { entry.target.provider = "not_a_provider"; }],
    ["backend", (entry) => { entry.backend = "rocm"; }],
    ["rung", (entry) => { entry.rung = "resident_but_wrong"; }],
  ]) {
    const broken = JSON.parse(JSON.stringify(plan));
    mutate(broken.providers[0]);
    assert.throws(
      () => assertCalibrationPlanTargetsResolvedCoordinates(broken, resolved.cells),
      (error) =>
        /match no resolved matrix coordinate/.test(error.message) &&
        error.message.includes(broken.providers[0].name),
      `a plan entry with an unresolvable ${axis} must fail closed`,
    );
  }

  // Wired into generation, not just exported: a bad plan must stop the artifact being written.
  const badPlan = JSON.parse(JSON.stringify(plan));
  badPlan.providers[0].target.mode = "edit";
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { calibrationPlan: JSON.stringify(badPlan) } }),
    /match no resolved matrix coordinate/,
  );

  // The nine sc-15817 entries specifically: they resolve now, and their targets are published.
  const qwenEdit = plan.providers.filter((entry) => entry.name.startsWith("candle-qwen-edit"));
  assert.equal(qwenEdit.length, 9);
  assert.ok(qwenEdit.every((entry) => entry.target.mode === "edit_image"));
  const matrix = await buildMatrix();
  for (const entry of qwenEdit) {
    assert.ok(
      matrix.cells.some((cell) => planEntryTargetsCoordinate(entry, cell)),
      `${entry.name} is a shipped plan target, so its coordinate must be published`,
    );
  }
});

test("a plan row for an OUT-OF-MATRIX family is exempt, and nothing else is (sc-18663)", async () => {
  // The category error: `buildMatrix` subtracts the survey's declared out-of-matrix catalog entries
  // from the coordinate universe, so a MiniMax-H3 row can never match a coordinate — and the
  // sc-18099 guard read that as "addresses nothing" and threw. Epic 17137's terminal campaign has to
  // plan captures against exactly those entries, so the first such row failed generation outright.
  const resolved = await buildMatrix({ publish: false });
  const outOfMatrixEntries = shippedOutOfMatrixEntries(await readFile(SURVEY_URL, "utf8"));
  assert.ok(outOfMatrixEntries.has("minimax_h3") && outOfMatrixEntries.has("minimax_h3_ref"));
  // The premise of the exemption, asserted rather than assumed: these entries resolve to no cell.
  for (const id of outOfMatrixEntries) {
    assert.equal(
      resolved.cells.filter((cell) => cell.modelId === id).length,
      0,
      `${id}: an out-of-matrix entry may not carry a coordinate — the exemption would then hide a real cell`,
    );
  }

  const plan = activeCalibrationPlan(
    JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url), "utf8")),
  );
  const minimaxRows = plan.providers.filter((entry) => entry.target.modelId === "minimax_h3");
  assert.ok(minimaxRows.length > 0, "the terminal campaign has checked-in MiniMax-H3 plan rows");
  assert.doesNotThrow(() =>
    assertCalibrationPlanTargetsResolvedCoordinates(plan, resolved.cells, { outOfMatrixEntries }),
  );
  await buildMatrix({ publish: false });

  // ...and it is an exemption, not a hole. A family in NEITHER the matrix nor the survey's
  // out-of-matrix set still fails closed, and so does a row that merely borrows the exempt family's
  // provider — the exemption is keyed on `target.modelId`, the same field the subtraction is.
  for (const [label, mutate] of [
    ["a family in neither set", (entry) => {
      entry.target.modelId = "hailuo_4";
      entry.target.provider = "hailuo_4";
    }],
    ["an exempt provider on an unknown model", (entry) => {
      entry.target.modelId = "not_a_catalog_entry";
    }],
  ]) {
    const planned = structuredClone(plan);
    const donor = planned.providers.find((entry) => entry.target.modelId !== "minimax_h3");
    mutate(donor);
    assert.throws(
      () =>
        assertCalibrationPlanTargetsResolvedCoordinates(planned, resolved.cells, { outOfMatrixEntries }),
      /match no resolved matrix coordinate/,
      `${label} must still fail closed`,
    );
    await assert.rejects(
      buildMatrix({ publish: false, sourceOverrides: { calibrationPlan: JSON.stringify(planned) } }),
      /match no resolved matrix coordinate/,
    );
  }

  // The exemption is opt-in and defaults to the strict behaviour, so a caller that forgets to pass
  // the set gets the old guard rather than a silently widened one.
  assert.throws(
    () => assertCalibrationPlanTargetsResolvedCoordinates(plan, resolved.cells),
    /match no resolved matrix coordinate/,
  );
});

test("MiniMax-H3 has one legal measured-spanning grid per implemented tier/rung family (sc-18663)", async () => {
  const plan = activeCalibrationPlan(
    JSON.parse(await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url), "utf8")),
  );
  const rows = plan.providers.filter((entry) => entry.target.modelId === "minimax_h3");
  assert.equal(rows.length, 72, "3 tiers × 4 implemented rungs × 2 areas × 3 frame levels");
  assert.doesNotThrow(() => assertMinimaxH3CalibrationPlan(plan));
  await buildMatrix({ publish: false });

  const families = new Map();
  for (const row of rows) {
    const key = `${row.target.tier}:${row.rung}`;
    if (!families.has(key)) families.set(key, []);
    families.get(key).push(row);
    assert.equal(row.target.geometry.frames % 17, 5, `${row.name}: frame lattice`);
  }
  assert.equal(families.size, 12, "only the three tiers and four implemented rungs are planned");
  for (const [family, entries] of families) {
    assert.equal(entries.length, 6, `${family}: the full 2×3 grid is planned`);
    assert.equal(new Set(entries.map((entry) => entry.target.geometry.width * entry.target.geometry.height)).size, 2);
    assert.equal(new Set(entries.map((entry) => entry.target.geometry.frames)).size, 3);
  }

  // Bounded attention is structurally unavailable for every MiniMax-H3 tier.  It must not slip
  // into a declared composition just because it appears earlier in a generic five-rung ladder.
  const impossible = structuredClone(plan);
  impossible.providers.find((entry) =>
    entry.target.modelId === "minimax_h3" && entry.rung === "bounded_decode",
  ).engagedRungs = ["resident", "bounded_decode", "bounded_attention"];
  assert.throws(
    () => assertMinimaxH3CalibrationPlan(impossible),
    /declares engaged rungs/,
  );
  await assert.rejects(
    buildMatrix({ publish: false, sourceOverrides: { calibrationPlan: JSON.stringify(impossible) } }),
    /declares engaged rungs/,
    "the generator must reject an impossible declared engaged rung before terminal capture",
  );
});

test("an out-of-matrix family's RECEIPT is unbound rather than fatal (sc-18663)", async () => {
  // The same category error one step later, and the wall the campaign hits immediately after the
  // plan one: `calibrationRuns` binds every bundle record to a cell, and an out-of-matrix family has
  // no cell BY CONSTRUCTION, so the first MiniMax-H3 receipt failed generation outright. The record
  // is skipped instead — its numbers live on the family's derived `memoryCharacterization` — while a
  // record naming a family in NEITHER set still throws.
  const bundle = JSON.parse(
    await readFile(new URL("../docs/generated/memory-calibration-evidence.json", import.meta.url), "utf8"),
  );
  const donor = bundle.records.find((record) => record.backend === "mlx");
  assert.ok(donor, "the shipped bundle must carry an MLX record to re-target");
  // Re-targeted through the harness's OWN identity functions: the bundle is schema- and
  // identity-validated before the generator ever reaches the binding step, so a hand-written id
  // would fail somewhere else and this test would pin the wrong wall.
  const withRecord = (modelId, provider) => {
    const clone = JSON.parse(JSON.stringify(donor));
    clone.target.modelId = modelId;
    clone.target.provider = provider;
    clone.logicalCaseId = logicalCaseId(clone);
    clone.id = recordId(clone);
    const copy = JSON.parse(JSON.stringify(bundle));
    copy.records.push(clone);
    return { body: JSON.stringify(copy), id: clone.id };
  };

  const baseline = await buildMatrix({ publish: false });
  assert.equal(
    baseline.calibrationRuns.length,
    baseline.summary.calibrationRuns,
    "today every shipped receipt binds, so the two counts start equal",
  );

  for (const modelId of ["minimax_h3", "minimax_h3_ref"]) {
    const { body, id } = withRecord(modelId, "minimax_h3");
    const matrix = await buildMatrix({
      publish: false,
      sourceOverrides: { calibrationEvidence: body },
    });
    assert.ok(
      !matrix.calibrationRuns.some((run) => run.record.id === id),
      `${modelId}: an out-of-matrix receipt binds to no cell`,
    );
    assert.equal(matrix.calibrationRuns.length, baseline.calibrationRuns.length);
    // The pair stays honest in both halves: the bundle count still sees the receipt, the bound
    // count does not. A skip that quietly shrank BOTH would hide the receipt entirely.
    assert.equal(matrix.summary.calibrationRuns, baseline.summary.calibrationRuns + 1);
  }

  // ...and the requirement is narrowed, not removed. Both arms matter, and the second is what pins
  // the KEY: the skip reads `target.modelId`, the same field the universe subtraction reads. Keying
  // it on `target.provider` instead would agree on every well-formed row — `minimax_h3_ref` rides
  // provider `minimax_h3` — and silently swallow this one, which is precisely the drift the
  // shared-field rule exists to prevent.
  for (const [label, modelId, provider] of [
    ["a family in neither set", "hailuo_4", "hailuo_4"],
    ["an exempt provider on an unknown model", "hailuo_4", "minimax_h3"],
  ]) {
    const unbindable = withRecord(modelId, provider);
    await assert.rejects(
      buildMatrix({ publish: false, sourceOverrides: { calibrationEvidence: unbindable.body } }),
      /calibration record does not map to a matrix cell/,
      `${label} must still fail closed`,
    );
  }
});

test("a partially implemented lane says WHICH coordinates, per axis (sc-18099)", async () => {
  // A census row spans tier x mode x overlay, so a bare `implemented` count is unambiguous only at 0
  // or `coordinates`. In between it hid the sc-16069 question: `krea_2_turbo:mlx` rung 4 reads
  // implemented 12/18 while its CONTROL overlay is 0/6, and before the slim that was answered by a
  // published `Missing` control cell.
  const resolved = await buildMatrix({ publish: false });
  const matrix = await buildMatrix();

  const mixed = matrix.coverage.filter(
    (row) => row.implemented > 0 && row.implemented < row.coordinates,
  );
  assert.ok(mixed.length > 0, "some lane must be partially implemented, or this asserts nothing");
  assert.ok(
    matrix.coverage.every(
      (row) =>
        Object.hasOwn(row, "implementedBy") ===
        (row.implemented > 0 && row.implemented < row.coordinates),
    ),
    "implementedBy rides exactly the rows a bare count cannot answer",
  );

  // Every marginal accounts for the whole count, and every axis value present in the lane appears —
  // including the zeroes, which are the answer the sc-16069 case needs.
  for (const row of mixed) {
    const lane = resolved.cells.filter(
      (cell) =>
        cell.modelId === row.modelId && cell.backend === row.backend && cell.rung === row.rung,
    );
    for (const axis of ["tier", "mode", "overlay"]) {
      const counts = row.implementedBy[axis];
      assert.deepEqual(
        Object.keys(counts).sort(),
        [...new Set(lane.map((cell) => cell[axis]))].sort(),
        `${row.modelId}:${row.backend}:${row.rung}: implementedBy.${axis} must name every value the lane spans`,
      );
      assert.equal(
        Object.values(counts).reduce((sum, count) => sum + count, 0),
        row.implemented,
      );
      for (const [value, count] of Object.entries(counts)) {
        assert.equal(
          count,
          lane.filter((cell) => cell[axis] === value && isImplemented(cell.state)).length,
          `${row.modelId}:${row.backend}:${row.rung}: implementedBy.${axis}.${value} must be the real count`,
        );
      }
    }
  }

  // The named case, asserted directly: a control lane that publishes NO cell is still legible as
  // "declared but not implemented at this rung" rather than as "not there".
  const krea = matrix.coverage.find(
    (row) =>
      row.modelId === "krea_2_turbo" &&
      row.backend === "mlx" &&
      row.rung === "bounded_transformer_residency",
  );
  assert.ok(krea.implemented > 0 && krea.implemented < krea.coordinates);
  assert.equal(krea.implementedBy.overlay.control, 0);
  assert.ok(krea.implementedBy.overlay.none > 0);
  assert.ok(
    matrix.models
      .find((model) => model.id === "krea_2_turbo")
      .axes.mlx.overlays.includes("control"),
    "and the lane itself is still declared, which is what makes the zero readable",
  );
  assert.equal(
    matrix.cells.filter(
      (cell) =>
        cell.modelId === "krea_2_turbo" &&
        cell.backend === "mlx" &&
        cell.overlay === "control" &&
        cell.rung === "bounded_transformer_residency",
    ).length,
    0,
    "no published cell answers this — the census is the only thing that does",
  );
});

test("manifest-derived evidence is published once per scope, and the join is closed (sc-18099)", async () => {
  const resolved = await buildMatrix({ publish: false });
  const matrix = await buildMatrix();

  // The hoist is lossless: each published cell's scope carries exactly what the resolved cell had.
  for (const cell of matrix.cells) {
    const source = resolved.cells.find((candidate) => candidate.id === cell.id);
    const scope = matrix.manifestScopes[cell.evidence.manifestScope];
    assert.ok(scope, `${cell.id}: unpublished manifestScope`);
    assert.equal(cell.evidence.manifestScope, `${cell.modelId}:${cell.backend}:${cell.tier}`);
    assert.deepEqual(scope.declaredCalibration, source.evidence.declaredCalibration);
    assert.deepEqual(scope.loadability, source.evidence.loadability);
    assert.ok(!Object.hasOwn(cell.evidence, "declaredCalibration"));
    assert.ok(!Object.hasOwn(cell.evidence, "loadability"));
  }
  // Worth doing at all: far fewer scopes than cells.
  assert.ok(Object.keys(matrix.manifestScopes).length < matrix.cells.length / 3);
  // No orphan scopes either — a table that outgrew its references is the same defect mirrored.
  assert.deepEqual(
    Object.keys(matrix.manifestScopes).sort(),
    [...new Set(matrix.cells.map((cell) => cell.evidence.manifestScope))].sort(),
  );
  // `evidenceDimensions` still names all six: the change is where two of them are written.
  assert.ok(matrix.evidenceDimensions.includes("declaredCalibration"));
  assert.ok(matrix.evidenceDimensions.includes("loadability"));

  // The hoist is only sound because these really are functions of (entry, backend, tier). If that
  // stopped being true the generator must refuse rather than publish whichever copy it saw first.
  const drifted = resolved.cells
    .filter((cell) => cell.modelId === matrix.cells[0].modelId)
    .map((cell, index) =>
      index === 1
        ? { ...cell, evidence: { ...cell.evidence, loadability: [{ repository: "drift", revision: null, variant: null }] } }
        : cell,
    );
  if (drifted.length > 1) {
    assert.throws(
      () => hoistManifestScopes(drifted),
      /manifest-derived evidence differs between coordinates of scope/,
    );
  }
});

// The two scopings applied to `assertMlxStagedCoverageIsStructurallyConsistent` on 2026-08-17 (see the
// honest-scope note at the top of generate-memory-matrix.mjs) each narrow it, so each needs a control
// proving the narrowing did not become a hole. Driven against the exported assertion over a synthetic
// census rather than the real catalog: the boundary being tested is the assertion's own logic, and a
// fixture states the boundary in one screen instead of depending on whatever the catalog happens to
// declare this week.
function stagedCensusFixture() {
  const model = (id, { route = id, routeKind = "registry", tiers = ["q4"] } = {}) => ({
    id,
    resolvedRoute: route,
    routeKind,
    axes: { mlx: { tiers } },
  });
  const cell = (modelId, { overlay = "none", state = "Implemented/unverified" } = {}) => ({
    backend: "mlx",
    rung: "staged_residency",
    modelId,
    tier: "q4",
    mode: "text_to_image",
    overlay,
    state,
  });
  // Minimum shape the other assertions in the function demand: bernini in the census, flux2_dev out of
  // it, and the census neither empty nor the whole catalog.
  return {
    models: [model("bernini_image"), model("filler_a"), model("filler_b")],
    cells: [cell("bernini_image")],
    model,
    cell,
  };
}

test("the staged-coverage census fixture is green before any perturbation", () => {
  const { models, cells } = stagedCensusFixture();
  assertMlxStagedCoverageIsStructurallyConsistent({ models, cells });
});

test("a drifting tiered route-mate still reds after the single-dense-tier exemption", () => {
  const { models, cells, model, cell } = stagedCensusFixture();
  // Two TIERED entries sharing one route, disagreeing: the exemption must not reach this.
  const drifted = {
    models: [...models, model("tiered_staged", { route: "shared" }), model("tiered_bare", { route: "shared" })],
    cells: [...cells, cell("tiered_staged")],
  };
  assert.throws(
    () => assertMlxStagedCoverageIsStructurallyConsistent(drifted),
    /MLX staged coverage disagrees within resolved route\(s\) shared/,
    "per-route drift between tiered entries must still red",
  );

  // Same disagreement, except the entry that lacks staged coverage advertises NO tier ladder — the
  // `flux2_klein_9b_true_v2` shape. Structurally fixed, not drift, so it is exempt.
  const exempt = {
    models: [
      ...models,
      model("tiered_staged", { route: "shared" }),
      model("single_dense", { route: "shared", tiers: ["default"] }),
    ],
    cells: [...cells, cell("tiered_staged")],
  };
  assertMlxStagedCoverageIsStructurallyConsistent(exempt);

  // And the exemption is keyed on the axis, not on the id: a single-dense entry is only exempt from the
  // CROSS-ENTRY comparison. Add a third TIERED route-mate that disagrees and the route reds again.
  const exemptPlusDrift = {
    models: [...exempt.models, model("tiered_bare", { route: "shared" })],
    cells: exempt.cells,
  };
  assert.throws(
    () => assertMlxStagedCoverageIsStructurallyConsistent(exemptPlusDrift),
    /disagrees within resolved route\(s\) shared/,
    "an exempt entry on the route must not suppress drift between its tiered siblings",
  );
});

test("a bespoke route claiming the generic staged ladder still reds", () => {
  const { models, cells, model, cell } = stagedCensusFixture();
  // PuLID's actual shape: bespoke dispatch, staged coverage ONLY on its own closed overlay. Allowed.
  const closedOverlayOnly = {
    models: [...models, model("bespoke_identity", { routeKind: "bespoke" })],
    cells: [...cells, cell("bespoke_identity", { overlay: "identity" })],
  };
  assertMlxStagedCoverageIsStructurallyConsistent(closedOverlayOnly);

  // The generic coordinates are the ones a bespoke route may never claim.
  for (const overlay of ["none", "lora"]) {
    const generic = {
      models: [...models, model("bespoke_identity", { routeKind: "bespoke" })],
      cells: [...cells, cell("bespoke_identity", { overlay })],
    };
    assert.throws(
      () => assertMlxStagedCoverageIsStructurallyConsistent(generic),
      /bespoke route\(s\) bespoke_identity claim generic MLX staged coverage/,
      `a bespoke route claiming the ${overlay} overlay must red`,
    );
  }
});

// ---------------------------------------------------------------------------------------------
// sc-18815 — the modality-aware model universe.
// ---------------------------------------------------------------------------------------------

test("the universe is modality-aware, and every video entry is IN it (sc-18815)", async () => {
  // The defect this replaces: `manifest.models.filter((m) => m.type === "image")`. Video entries
  // were not reported as `Missing` — they were outside the universe, so the matrix read as complete
  // while covering one modality. The fix has to be checked as REACH, not as a filter edit: every
  // video entry in the shipped manifest resolves a route, a family, an owner and a cell population.
  const manifestBody = await readFile(
    new URL("../config/manifests/builtin.models.jsonc", import.meta.url),
    "utf8",
  );
  // MINUS the survey's declared out-of-matrix entries (sc-18664 x sc-18815, reconciled by the
  // sc-17137 main sync): MiniMax-H3's two video entries are validated out-of-matrix records —
  // `familyGroup` has no arm and no route resolver row exists — so the universe deliberately
  // excludes them until the epic promotes the family (the record's own guard forces the move).
  const surveyBody = await readFile(
    new URL("../config/rung4-applicability-survey.json", import.meta.url),
    "utf8",
  );
  const outOfMatrixEntries = new Set(
    Object.values(JSON.parse(surveyBody).outOfMatrixFamilies ?? {}).flatMap(
      (family) => family.catalogEntries ?? [],
    ),
  );
  const manifestVideo = JSON.parse(stripJsoncComments(manifestBody))
    .models.filter((model) => model.type === "video" && !outOfMatrixEntries.has(model.id))
    .map((model) => model.id)
    .sort();
  assert.equal(manifestVideo.length, 10);

  const matrix = await buildMatrix({ publish: false });
  const inMatrix = matrix.models.filter((model) => model.modality === "video").map((model) => model.id);
  assert.deepEqual(inMatrix.sort(), manifestVideo, "every manifest video entry is in the universe");
  assert.equal(matrix.models.filter((model) => model.modality === "image").length, 53);
  assert.equal(matrix.summary.catalogEntries, matrix.models.length);
  assert.deepEqual(matrix.summary.catalogEntriesByModality, { image: 53, video: 10 });

  // Cells exist for every routed video lane, on every rung — the "per-rung coverage rows" the story
  // asks for — and their states are the ordinary vocabulary, not a video-specific one.
  const videoCells = matrix.cells.filter((cell) => inMatrix.includes(cell.modelId));
  assert.ok(videoCells.length > 1000);
  for (const model of matrix.models.filter((entry) => entry.modality === "video")) {
    for (const backend of model.backends) {
      for (const rung of ["resident", "staged_residency", "bounded_decode", "bounded_attention", "bounded_transformer_residency"]) {
        assert.ok(
          videoCells.some(
            (cell) => cell.modelId === model.id && cell.backend === backend && cell.rung === rung,
          ),
          `${model.id}:${backend}:${rung}: no cell`,
        );
      }
    }
  }
  assert.ok(
    videoCells.some((cell) => cell.state === "Missing"),
    "an unmeasured video coordinate must report Missing rather than vanish",
  );
  assert.ok(videoCells.some((cell) => isImplemented(cell.state)));
});

test("video providers are per BACKEND, derived from the worker's own route resolvers (sc-18815)", async () => {
  // LTX is `ltx_2_3` on MLX and `ltx_2_3_distilled` on candle. A single scalar route would be wrong
  // on one backend, and a wrong provider is not cosmetic — it is the key calibration evidence, the
  // calibration plan and the per-provider closure digests all bind on. `mlx:ltx_2_3` is the exact
  // key sc-18808 committed to `config/inference-provider-closures.json`.
  const matrix = await buildMatrix({ publish: false });
  const ltx = matrix.models.find((model) => model.id === "ltx_2_3");
  assert.deepEqual(ltx.resolvedRoutes, { mlx: "ltx_2_3", candle: "ltx_2_3_distilled" });
  const eros = matrix.models.find((model) => model.id === "ltx_2_3_eros");
  assert.deepEqual(
    eros.resolvedRoutes,
    { mlx: "ltx_2_3" },
    "sc-18902: Eros keeps its validated MLX recipe and must not regain the failed Candle route",
  );
  const providers = new Set(
    matrix.cells
      .filter((cell) => cell.modelId === "ltx_2_3")
      .map((cell) => `${cell.backend}:${cell.provider}`),
  );
  assert.deepEqual([...providers].sort(), ["candle:ltx_2_3_distilled", "mlx:ltx_2_3"]);

  const closures = JSON.parse(
    await readFile(new URL("../config/inference-provider-closures.json", import.meta.url), "utf8"),
  );
  assert.ok(
    closures.providers["mlx:ltx_2_3"],
    "the provider a cell binds on must be the one the closure table already names",
  );
});

test("a video route parser handles every form the worker spells, consts included (sc-18815)", () => {
  // Three syntactic forms and a const-spelled one, because all four ship. A parser that silently
  // skipped a form would drop a whole family from the universe and read as "the worker does not
  // route it" — the same silent absence, one layer down. Synthetic input: real data cannot show a
  // parser missing a form it does not contain.
  const matchForm = `
fn candle_video_engine_id(model: &str) -> Option<&'static str> {
    match model {
        "wan_2_2" => Some("wan2_2_ti2v_5b"),
        "ltx_2_3" | "ltx_2_3_eros" => Some("ltx_2_3_distilled"),
        _ => None,
    }
}
`;
  assert.deepEqual(
    [...parseVideoEngineIds(matchForm, "candle_video_engine_id")].sort(),
    [
      ["ltx_2_3", "ltx_2_3_distilled"],
      ["ltx_2_3_eros", "ltx_2_3_distilled"],
      ["wan_2_2", "wan2_2_ti2v_5b"],
    ],
  );

  const eqForm = `fn svd_engine_id(model: &str) -> Option<&'static str> {
    (model == "svd").then_some("svd_xt")
}
`;
  assert.deepEqual([...parseVideoEngineIds(eqForm, "svd_engine_id")], [["svd", "svd_xt"]]);

  const matchesForm = `fn ltx_engine_id(model: &str) -> Option<&'static str> {
    matches!(model, "ltx_2_3" | "ltx_2_3_eros").then_some("ltx_2_3")
}
`;
  assert.deepEqual(
    [...parseVideoEngineIds(matchesForm, "ltx_engine_id")].sort(),
    [["ltx_2_3", "ltx_2_3"], ["ltx_2_3_eros", "ltx_2_3"]],
  );

  // `krea_realtime_engine_id` is spelled ENTIRELY in consts. A literal-only parser returns an empty
  // map for it, which is why the empty map is a throw rather than a family quietly disappearing.
  const constForm = `pub(super) const KREA_REALTIME_MODEL_ID: &str = "krea_realtime_14b";
fn krea_realtime_engine_id(model: &str) -> Option<&'static str> {
    (model == KREA_REALTIME_MODEL_ID).then_some(KREA_REALTIME_MODEL_ID)
}
`;
  assert.deepEqual(
    [...parseVideoEngineIds(constForm, "krea_realtime_engine_id")],
    [["krea_realtime_14b", "krea_realtime_14b"]],
  );

  // Fails closed on a rename and on a body that yields nothing.
  assert.throws(() => parseVideoEngineIds(eqForm, "renamed_engine_id"), /could not locate/);
  assert.throws(
    () => parseVideoEngineIds(`fn empty_engine_id(model: &str) -> Option<&'static str> {\n    None\n}\n`, "empty_engine_id"),
    /declared no model -> engine arm/,
  );
  // And on a const it cannot resolve, rather than inventing an id.
  assert.throws(
    () =>
      parseVideoEngineIds(
        `fn x_engine_id(model: &str) -> Option<&'static str> {\n    (model == MISSING_CONST).then_some("x")\n}\n`,
        "x_engine_id",
      ),
    /resolves to no &str const/,
  );
});

test("the two worker declarations of a video route must agree (sc-18815)", async () => {
  // `engines.rs#video_engine_ids` and the per-family `*_engine_id` functions are independent
  // declarations of the same fact. The generator consults BOTH, so one of them drifting is a
  // generation failure rather than a provider silently swapped underneath a cell's evidence.
  const bodies = Object.fromEntries(
    await Promise.all(
      Object.entries(SOURCE_PATHS).map(async ([name, relative]) => [
        name,
        await readFile(new URL(`../${relative}`, import.meta.url), "utf8"),
      ]),
    ),
  );
  const routes = parseVideoRoutes(bodies);
  assert.equal(routes.get("ltx_2_3").mlx, "ltx_2_3");
  assert.equal(routes.get("bernini").candle, "bernini", "candle routes Bernini off the model id");

  const drifted = {
    ...bodies,
    videoRouteSvd: bodies.videoRouteSvd.replace('.then_some("svd_xt")', '.then_some("svd_xt_v2")'),
  };
  assert.throws(
    () => parseVideoRoutes(drifted),
    /video route svd:mlx resolves to provider svd_xt_v2.*video route declarations disagree/s,
  );
});

test("MiniMax-H3 Candle dispatch exposes the authorized base and Ref2VA routes", async () => {
  const [dispatch, minimax, routingCatalog, routingCandle, routingMlx, bodies] = await Promise.all([
    readFile(new URL("../crates/sceneworks-worker/src/video_jobs/mod.rs", import.meta.url), "utf8"),
    readFile(new URL("../crates/sceneworks-worker/src/video_jobs/minimax_h3.rs", import.meta.url), "utf8"),
    readFile(new URL("../crates/sceneworks-core/src/jobs_store/routing/catalog.rs", import.meta.url), "utf8"),
    readFile(new URL("../crates/sceneworks-core/src/jobs_store/routing/candle.rs", import.meta.url), "utf8"),
    readFile(new URL("../crates/sceneworks-core/src/jobs_store/routing/mlx.rs", import.meta.url), "utf8"),
    Object.fromEntries(
      await Promise.all(
        Object.entries(SOURCE_PATHS).map(async ([name, relative]) => [
          name,
          await readFile(new URL(`../${relative}`, import.meta.url), "utf8"),
        ]),
      ),
    ),
  ]);
  const internal = parseInternalCandleVideoRoutes(dispatch, minimax);
  assert.deepEqual([...internal], [
    ["minimax_h3", "minimax_h3"],
    ["minimax_h3_ref", "minimax_h3"],
  ]);
  const publicRoutes = parseVideoRoutes(bodies);
  const routedBackends = routedLanes({ routingCatalog, routingCandle, routingMlx });
  // sc-20755 authorizes the base t2va/fl2va executor and sc-20756 authorizes Ref2VA.
  // The worker keeps this family on its dedicated direct resolver rather than the generic
  // `*_engine_id` parsers, so public authorization is the catalog lane, not `publicRoutes`.
  assert.equal(publicRoutes.get("minimax_h3")?.candle, undefined);
  assert.equal(publicRoutes.get("minimax_h3_ref")?.candle, undefined);
  assert.deepEqual([...routedBackends.get("minimax_h3")], ["mlx", "candle"]);
  assert.deepEqual([...routedBackends.get("minimax_h3_ref")], ["mlx", "candle"]);

  assert.throws(
    () => parseInternalCandleVideoRoutes(
      dispatch.replace(
        "} else if let Some(engine_id) = minimax_h3_engine_id(&request.model) {\n        CandleVideoRoute::MiniMaxH3(engine_id)\n    } else if is_candle_video_engine(&request.model) {",
        "} else if is_candle_video_engine(&request.model) {",
      ),
      minimax,
    ),
    /direct Candle dispatch no longer selects CandleVideoRoute::MiniMaxH3/,
  );
});

test("a routed backend with no *_engine_id arm fails HERE, naming the resolver (sc-18815)", async () => {
  // The review mutation. Deleting `ltx_2_3` from `ltx_engine_id`'s arm used to let generation SUCCEED:
  // `resolvedRoutes.mlx` went `null`, the scalar fell back to `video.mlx ?? video.candle`, and all 180
  // MLX cells were stamped `ltx_2_3_distilled` — CANDLE's provider on an MLX cell, which is the exact
  // substitution this resolver exists to prevent. The only thing that noticed was JSON-schema
  // validation two lanes downstream, reporting `None is not of type 'string'` and naming neither the
  // resolver nor the cause. It has to fail at the resolver, with the owing function named.
  const ltx = await readFile(new URL(`../${SOURCE_PATHS.videoRouteLtx}`, import.meta.url), "utf8");
  const withoutMlxArm = ltx.replace(
    'matches!(model, "ltx_2_3" | "ltx_2_3_eros")',
    'matches!(model, "ltx_2_3_eros")',
  );
  assert.notEqual(withoutMlxArm, ltx, "the mutation must actually change ltx.rs");

  await assert.rejects(
    () => buildMatrix({ publish: false, sourceOverrides: { videoRouteLtx: withoutMlxArm } }),
    /ltx_2_3: the routing catalog routes mlx, but no mlx provider resolved — expected an arm in ltx_engine_id/,
  );

  // The named resolver is per BACKEND, not one generic complaint: the same entry losing its candle
  // arm must point at `candle_video_engine_id` instead.
  const candle = await readFile(new URL(`../${SOURCE_PATHS.videoRouteCandle}`, import.meta.url), "utf8");
  const withoutCandleArm = candle.replace(
    '"ltx_2_3" => Some("ltx_2_3_distilled")',
    '"ltx_2_3_missing" => Some("ltx_2_3_distilled")',
  );
  assert.notEqual(withoutCandleArm, candle, "the mutation must actually change candle.rs");
  await assert.rejects(
    () => buildMatrix({ publish: false, sourceOverrides: { videoRouteCandle: withoutCandleArm } }),
    /ltx_2_3: the routing catalog routes candle, but no candle provider resolved — expected an arm in candle_video_engine_id/,
  );
});

test("the union-only MLX fallback is an allowlist, not a shape (sc-18815)", async () => {
  // `engines.rs#video_engine_ids` is the ONLY source for `wan_2_2_vace_fun_14b` (its MLX dispatch is
  // the native VACE arm, which carries no id string). The fallback was written for that one entry but
  // keyed on `declared.length === 1`, which is a SHAPE thousands of ids can satisfy — and `mochi_1`
  // demonstrably took it, acquiring an MLX provider derived from `mochi.rs`, a file this generator
  // neither reads nor fingerprints. Inert (mochi has no manifest entry) but it downgraded the
  // "two independent declarations must agree" invariant to single-source for whoever hit it.
  const bodies = Object.fromEntries(
    await Promise.all(
      Object.entries(SOURCE_PATHS).map(async ([name, relative]) => [
        name,
        await readFile(new URL(`../${relative}`, import.meta.url), "utf8"),
      ]),
    ),
  );
  assert.deepEqual([...UNION_ONLY_MLX_ROUTES], ["wan_2_2_vace_fun_14b"]);

  const routes = parseVideoRoutes(bodies);
  assert.equal(routes.get("wan_2_2_vace_fun_14b").mlx, "wan2_2_vace_fun_14b");
  // `mochi_1` has a candle arm and no MLX one. Its MLX route must stay absent rather than be
  // synthesised from the union alone.
  assert.deepEqual(routes.get("mochi_1"), { candle: "mochi_1" });

  // The allowlist is load-bearing in BOTH directions: empty it and VACE loses the only source it
  // has, widen it and mochi acquires the single-source route the guard exists to withhold.
  assert.equal(parseVideoRoutes(bodies, new Set()).get("wan_2_2_vace_fun_14b"), undefined);
  assert.equal(
    parseVideoRoutes(bodies, new Set([...UNION_ONLY_MLX_ROUTES, "mochi_1"])).get("mochi_1").mlx,
    "mochi_1",
  );
});

test("the two ownership registries are disjoint, in the silent direction too (sc-18815)", () => {
  // `familyStory` consults `VIDEO_FAMILY_STORIES` FIRST. An IMAGE group added to the video registry
  // reds most of this suite; a VIDEO name added to the image registry is silent dead code nothing
  // else catches, so that direction needs its own assertion.
  assertOwnershipRegistriesAreDisjoint();
  assert.throws(
    () =>
      assertOwnershipRegistriesAreDisjoint(
        { ...FAMILY_STORIES, svd: { mlx: 1, candle: 2 } },
        VIDEO_FAMILY_STORIES,
      ),
    /family group svd is declared in BOTH FAMILY_STORIES and VIDEO_FAMILY_STORIES/,
  );
});

test("the VACE-Fun routing defect is closed and the wholly-unrouted guard stays fail-closed (sc-18826)", async () => {
  // The manifest and engine were always real. SC-18826 adds the missing MLX-only VideoModelCaps row,
  // so the old temporary exception must disappear and the route must now produce coordinates.
  const matrix = await buildMatrix({ publish: false });
  assert.deepEqual(matrix.summary.unroutedEntries, []);
  assert.deepEqual([...UNROUTED_CATALOG_ENTRIES], []);
  const vace = matrix.models.find((model) => model.id === "wan_2_2_vace_fun_14b");
  assert.deepEqual(vace.backends, ["mlx"]);
  assert.ok(Object.keys(vace.axes).length > 0);
  assert.ok(matrix.cells.some((cell) => cell.modelId === vace.id));

  // A NEW unrouted entry cannot appear undeclared...
  assert.throws(
    () =>
      assertUnroutedEntriesAreDeclared(
        [...matrix.models, { id: "brand_new_entry", backends: [] }],
        UNROUTED_CATALOG_ENTRIES,
      ),
    /brand_new_entry: the routing catalog routes it on no backend/,
  );
  // ...and a synthetic declaration cannot outlive the defect it explains.
  assert.throws(
    () =>
      assertUnroutedEntriesAreDeclared(
        matrix.models,
        new Map([
          [
            "wan_2_2_vace_fun_14b",
            { reason: "fixture: stale declaration", owningStory: 18826 },
          ],
        ]),
      ),
    /declared unrouted .* but the catalog now routes mlx/,
  );
});

test("mochi_1 is routed but deliberately outside the universe (sc-18815)", async () => {
  // The inverse defect: a `VIDEO_MODEL_CAPS` row and a worker route with NO manifest entry. Mochi is
  // frozen with no weights lane and epic 18803 lists it out of scope, so it must NOT be admitted —
  // and the reason has to be the MANIFEST, not a coincidence of some other filter that a later
  // change could remove. Pinned from both sides.
  const [manifestBody, routingCatalog, routingCandle, routingMlx] = await Promise.all([
    readFile(new URL("../config/manifests/builtin.models.jsonc", import.meta.url), "utf8"),
    readFile(new URL("../crates/sceneworks-core/src/jobs_store/routing/catalog.rs", import.meta.url), "utf8"),
    readFile(new URL("../crates/sceneworks-core/src/jobs_store/routing/candle.rs", import.meta.url), "utf8"),
    readFile(new URL("../crates/sceneworks-core/src/jobs_store/routing/mlx.rs", import.meta.url), "utf8"),
  ]);
  const routed = routedLanes({ routingCatalog, routingCandle, routingMlx });
  assert.ok(routed.get("mochi_1")?.size, "mochi is routed, so its absence is not a routing accident");
  assert.ok(
    !JSON.parse(stripJsoncComments(manifestBody)).models.some((model) => model.id === "mochi_1"),
    "mochi has no manifest entry — that is what keeps it out",
  );

  const matrix = await buildMatrix({ publish: false });
  assert.ok(!matrix.models.some((model) => model.id === "mochi_1"));
  assert.ok(!matrix.cells.some((cell) => cell.modelId === "mochi_1"));
  assert.ok(!matrix.summary.unroutedEntries.some((entry) => entry.id === "mochi_1"));
});

test("video cells name no per-entry story, and their family owner is the real one (sc-18815)", async () => {
  // Epic 18803 filed no per-(entry, backend) video stories — measurement is a runbook (epic 18093) —
  // so an integer in `owningModelStory` would name a story that cannot close the cell. That is
  // SC-15812's defect reached from the other direction, so it is checked with the same force.
  const matrix = await buildMatrix({ publish: false });
  const video = matrix.models.filter((model) => model.modality === "video");
  assert.ok(video.length === 10);
  for (const model of video) {
    for (const backend of model.backends) {
      assert.equal(model.owningModelStories[backend], null);
      assert.equal(model.owningFamilyStories[backend], familyStory(model.id, backend));
    }
  }
  // `bernini` is NOT in `VIDEO_FAMILY_STORIES`: same engine and same block stack as `bernini_image`,
  // so it stays in image family 15528 and takes that family's per-backend owners. Anything else
  // would survey one architecture twice and count the family twice.
  //
  // Both spellings, because the group key and the model id are different strings and only one of them
  // is the shape a mistaken entry would take. `familyGroup("bernini")` is `15528`, so the group-key
  // assertion alone is near-vacuous — it never fires on the actual error, which is someone adding a
  // literal `bernini:` row alongside the other video family NAMES (sc-18815 review).
  assert.ok(!Object.hasOwn(VIDEO_FAMILY_STORIES, "bernini"));
  assert.ok(!Object.hasOwn(VIDEO_FAMILY_STORIES, familyGroup("bernini")));
  assert.equal(familyGroup("bernini"), familyGroup("bernini_image"));
  const bernini = matrix.models.find((model) => model.id === "bernini");
  assert.deepEqual(bernini.owningFamilyStories, FAMILY_STORIES[15528]);

  // A fabricated per-entry owner is rejected...
  assert.throws(
    () =>
      assertVideoOwnership(
        matrix.models.map((model) =>
          model.id === "ltx_2_3"
            ? { ...model, owningModelStories: { ...model.owningModelStories, mlx: 18813 } }
            : model,
        ),
      ),
    /ltx_2_3:mlx: video entries carry no per-entry ownership story/,
  );
  // ...and so is a family owner that is not this family's.
  assert.throws(
    () =>
      assertVideoOwnership(
        matrix.models.map((model) =>
          model.id === "ltx_2_3"
            ? { ...model, owningFamilyStories: { ...model.owningFamilyStories, mlx: 18826 } }
            : model,
        ),
      ),
    /owningFamilyStory SC-18826 is not family ltx-video's mlx owner SC-18813/,
  );
});

test("the video survey debt is closed while its fail-closed lifecycle remains enforced", async () => {
  const matrix = await buildMatrix({ publish: false });
  const pending = matrix.summary.rung4Survey.pendingFamilyBackends;
  assert.deepEqual(pending, []);
  assert.deepEqual([...PENDING_RUNG4_SURVEYS], []);
  for (const family of ["wan-video", "scail2", "krea-realtime", "svd"]) {
    assert.ok(
      matrix.rung4SurveyRows.some((row) => row.familyStory === family),
      `${family} must now have a real survey row`,
    );
  }
  const unsurveyedCells = matrix.cells.filter(
    (cell) => cell.rung === "bounded_transformer_residency" && cell.rung4Survey.surveyed === false,
  );
  assert.deepEqual(unsurveyedCells, []);

  // The mechanism stays checked even though the current debt set is empty.
  assert.throws(
    () =>
      assertRung4SurveyCoversEveryFamily(
        new Map(),
        [{ id: "wan_2_2", backends: ["mlx"] }],
        { pendingSurveys: new Map() },
      ),
    /family wan-video has no mlx rung-4 survey verdict/,
  );
  // A pending row for a family the catalog advertises nowhere is a leftover, not a licence.
  assert.throws(
    () =>
      assertRung4SurveyCoversEveryFamily(new Map(), [], {
        pendingSurveys: new Map([["wan-video", 18826]]),
      }),
    /the catalog advertises no entry in that family — remove the pending row/,
  );
  // And it expires the moment the verdict lands, so sc-18826 cannot leave it behind.
  assert.throws(
    () =>
      assertRung4SurveyCoversEveryFamily(
        new Map([["wan-video:mlx", { structuralApplicability: "full" }]]),
        [{ id: "wan_2_2", backends: ["mlx"] }],
        { pendingSurveys: new Map([["wan-video", 18826]]) },
      ),
    /now carries a verdict for every advertised backend, so its pending row \(sc-18826\) is stale/,
  );
});

test("admitting video leaves the image lane's guards at full strength (sc-18815)", async () => {
  // The video lane is checked DIFFERENTLY, not less. Three ways the image guards could have been
  // widened instead of partitioned, each one a real regression, each one still rejected.
  const matrix = await buildMatrix({ publish: false });
  const scope = buildStoryBackendScope();
  const modalityByModelId = new Map(matrix.models.map((model) => [model.id, model.modality]));

  // 1. An image cell may still not carry a null owner, even though video cells do.
  const imageCell = matrix.cells.find((cell) => modalityByModelId.get(cell.modelId) === "image");
  assert.throws(
    () =>
      assertCellOwnershipIsBackendScoped(
        [{ ...imageCell, owningModelStory: null }],
        scope,
        modalityByModelId,
      ),
    /owningModelStory is null, not an ownership story id/,
  );
  // 2. An image cell may still not name a sibling's same-backend story.
  assert.throws(
    () =>
      assertCellOwnershipIsBackendScoped(
        [{ ...imageCell, owningModelStory: MODEL_STORIES.pulid_flux_dev[imageCell.backend] }],
        scope,
        modalityByModelId,
      ),
    /a cell credited to another entry's story/,
  );
  // 3. Twin coverage still holds over the image families — the video exclusion is by MODALITY, so
  // dropping an image family's candle twin is still rejected.
  assert.throws(
    () => assertTwinCoverage(matrix.models, MODEL_STORIES, { ...FAMILY_STORIES, 15516: { mlx: 15516 } }),
    /family SC-15516: owns dual models but has no Candle twin/,
  );
});

test("a video cell's geometry envelope carries its temporal axis (sc-18815)", async () => {
  // Publishing only the spatial half would claim the envelope is fully described while omitting the
  // axis a video peak actually scales on. These are the catalog's DECLARED limits and nothing more —
  // how the phase CURVE represents time is sc-18810/sc-18812's to measure and decide.
  const matrix = await buildMatrix({ publish: false });
  const temporal = ["defaultDuration", "durations", "hardMaxDuration", "defaultFps", "fps"];
  const video = matrix.cells.filter((cell) => cell.modelId === "ltx_2_3");
  assert.ok(video.length > 0);
  for (const cell of video) {
    for (const key of temporal) {
      assert.ok(key in cell.geometryEnvelope, `${cell.id}: geometryEnvelope omits ${key}`);
    }
    assert.deepEqual(cell.geometryEnvelope.durations, [4, 6, 8, 10, 12, 15]);
    assert.deepEqual(cell.geometryEnvelope.fps, [24, 25, 30]);
  }
  // And no IMAGE envelope gains one: image entries declare none of these keys, so an empty array or
  // a null appearing there would be the generator inventing an axis the catalog never declared.
  const imageIds = new Set(
    matrix.models.filter((model) => model.modality === "image").map((model) => model.id),
  );
  for (const cell of matrix.cells.filter((entry) => imageIds.has(entry.modelId))) {
    for (const key of temporal) {
      assert.ok(!(key in cell.geometryEnvelope), `${cell.id}: image envelope gained ${key}`);
    }
  }
});

// ── sc-18812: the temporal axis reaches the characterization ───────────────────────────────────

test("the measured-geometry key carries frames, and only above one (sc-18812)", () => {
  // The `WxH` form has to survive verbatim: it is what all 208 published cells already carry, and
  // the migration claim is that admitting a temporal axis moves none of them.
  assert.equal(measuredGeometryKey({ width: 1024, height: 1024, frames: 1 }), "1024x1024");
  assert.equal(measuredGeometryKey({ width: 768, height: 768, frames: 0 }), "768x768");
  assert.equal(measuredGeometryKey({ width: 768, height: 512, frames: 241 }), "768x512xf241");
});

test("frames DISTINGUISH two measured geometries rather than collapsing (sc-18812)", () => {
  // The seam this story was written against: two records that differ only temporally used to key
  // to one `768x512` string, so a cell with real temporal coverage reported `point`.
  const characterization = memoryCharacterization(["768x512xf121", "768x512xf241"]);
  assert.deepEqual(characterization.measuredGeometries, ["768x512xf121", "768x512xf241"]);
});

test("temporal geometries at ONE area cannot determine the curve (sc-18812)", () => {
  // Three points, three frame counts, and still singular: with one area the `mpx` and `mpx*frames`
  // columns are proportional. sc-18810 established this the expensive way — crossing two areas is
  // what makes the candidate forms identifiable at all — so counting geometries is the wrong test.
  const characterization = memoryCharacterization([
    "768x512xf121",
    "768x512xf241",
    "768x512xf361",
  ]);
  assert.equal(characterization.status, "point");
  assert.equal(characterization.coveredPixelBound, null);
  assert.equal(characterization.coveredFrameBound, null);
});

test("two areas crossed with two frame counts ARE determinable (sc-18812)", () => {
  const characterization = memoryCharacterization([
    "768x512xf121",
    "768x512xf241",
    "1280x704xf121",
  ]);
  assert.equal(characterization.status, "fitted");
  assert.equal(characterization.coveredPixelBound, 1280 * 704);
  assert.equal(characterization.coveredFrameBound, 241);
});

test("`coveredFrameBound` appears only where a temporal geometry does (sc-18812)", () => {
  // Mirrors `geometryEnvelope`'s durations/fps convention. An image cell that gained a null
  // temporal bound would be the generator inventing an axis its records never carried.
  assert.ok(!("coveredFrameBound" in memoryCharacterization(["768x768", "1024x1024"])));
  assert.ok(!("coveredFrameBound" in memoryCharacterization([])));
  assert.ok("coveredFrameBound" in memoryCharacterization(["768x512xf121", "1280x704xf241"]));
});

test("two resolutions of ONE area do not determine the image curve either (sc-18812)", () => {
  // A latent defect the rank rule fixes on the way past: `768x512` and `512x768` are two distinct
  // geometry strings carrying one area, and counting called that `fitted`. It publishes no cell
  // today — verified by the regenerated artifact being byte-identical — but it would have been the
  // first thing a portrait/landscape video sweep hit.
  assert.equal(memoryCharacterization(["768x512", "512x768"]).status, "point");
  assert.equal(memoryCharacterization(["768x512", "512x768"]).coveredPixelBound, null);
  assert.equal(memoryCharacterization(["768x512", "1280x704"]).status, "fitted");
});

test("admitting the temporal axis moves no published image cell (sc-18812)", async () => {
  // The migration guard, over the REAL published population rather than a fixture. Every shipped
  // cell must still key its geometries in the two-number form and carry no temporal bound.
  const matrix = await buildMatrix({ publish: false });
  let measured = 0;
  for (const cell of matrix.cells) {
    assert.ok(
      !("coveredFrameBound" in cell.memoryCharacterization),
      `${cell.id}: an image cell gained a temporal bound`,
    );
    for (const geometry of cell.memoryCharacterization.measuredGeometries) {
      assert.match(geometry, /^[1-9][0-9]*x[1-9][0-9]*$/, `${cell.id}: ${geometry}`);
      measured += 1;
    }
  }
  assert.ok(measured > 0, "the guard must have graded real measured geometries");
});

// ── sc-18812 review pass: the DECLARED form, and evidence that can state its frames ────────────

/** The builtin manifest, parsed, ready to be mutated and fed back through `sourceOverrides`. */
async function parsedBuiltinManifest() {
  return JSON.parse(
    stripJsoncComments(
      await readFile(new URL("../config/manifests/builtin.models.jsonc", import.meta.url), "utf8"),
    ),
  );
}

const KREA_Q8_THREE_STAGE = "krea_2_turbo:krea_2_turbo:candle:q8:text_to_image:none:staged_residency";
// `resident` is the control: it is the one rung with NO entry in `phaseCurvesByTier`, so it shares
// every evidence record with the cell above and differs only in the declared curve form.
const KREA_Q8_RESIDENT = "krea_2_turbo:krea_2_turbo:candle:q8:text_to_image:none:resident";

const cellById = (matrix, id) => {
  const cell = matrix.cells.find((candidate) => candidate.id === id);
  assert.ok(cell, `${id} must exist for this test to mean anything`);
  return cell;
};

test("the coefficient count comes from the DECLARED curve, not from what was measured (sc-18812)", () => {
  // Two areas, one frame each. Against the two-coefficient image form that is a determinable
  // curve; against a declared three-coefficient form it is two points short of one, and calling it
  // `fitted` would publish a temporal coefficient nobody measured.
  const geometries = ["768x768", "1024x1024"];
  const image = memoryCharacterization(geometries);
  assert.equal(image.status, "fitted");
  assert.equal(image.coveredPixelBound, 1024 * 1024);
  assert.ok(!("coveredFrameBound" in image));

  const declaredTemporal = memoryCharacterization(geometries, { declaresTemporalCurve: true });
  assert.equal(declaredTemporal.status, "point");
  assert.equal(declaredTemporal.coveredPixelBound, null);
  assert.equal(declaredTemporal.coveredFrameBound, null);

  // ...and the flag can only ever ADD the axis. Multi-frame measurements are temporal whatever the
  // flag says, so an unset flag cannot smuggle a video cell back onto the two-coefficient rule.
  assert.equal(
    memoryCharacterization(["768x512xf121", "768x512xf241", "1280x704xf121"], {
      declaresTemporalCurve: false,
    }).status,
    "fitted",
  );
  assert.equal(memoryCharacterization(["768x768"], { declaresTemporalCurve: true }).status, "point");
  assert.equal(
    memoryCharacterization([], { declaresTemporalCurve: true }).status,
    "unmeasured",
    "a declared temporal curve with no measurements is still unmeasured, not point",
  );
});

test("a declared temporal curve regrades its own cell against three coefficients (sc-18812)", async () => {
  const baseline = await buildMatrix({ publish: false });
  assert.equal(cellById(baseline, KREA_Q8_THREE_STAGE).memoryCharacterization.status, "fitted");

  const manifest = await parsedBuiltinManifest();
  const krea = manifest.models.find((model) => model.id === "krea_2_turbo");
  krea.candle.turboFit.phaseCurvesByTier.q8.threeStage.decode.perMpxFrameGb = 0.2998482076533136;
  const changed = await buildMatrix({
    publish: false,
    sourceOverrides: { manifest: JSON.stringify(manifest) },
  });

  // The cell whose curve gained the term: same two measured geometries, now one short of the rank
  // its own form needs.
  const declared = cellById(changed, KREA_Q8_THREE_STAGE).memoryCharacterization;
  assert.deepEqual(declared.measuredGeometries, ["1024x1024", "768x768"]);
  assert.equal(declared.status, "point");
  assert.equal(declared.coveredPixelBound, null);
  assert.equal(declared.coveredFrameBound, null);

  // The control, on the SAME evidence: no declared temporal term, so nothing about it moves.
  assert.deepEqual(
    cellById(changed, KREA_Q8_RESIDENT).memoryCharacterization,
    cellById(baseline, KREA_Q8_RESIDENT).memoryCharacterization,
  );
  assert.equal(cellById(changed, KREA_Q8_RESIDENT).memoryCharacterization.status, "fitted");
});

test("an evidence record can state its frame count, and it reaches characterization (sc-18812)", async () => {
  const baseline = await buildMatrix({ publish: false });
  assert.deepEqual(cellById(baseline, KREA_Q8_THREE_STAGE).memoryCharacterization.measuredGeometries, [
    "1024x1024",
    "768x768",
  ]);

  const manifest = await parsedBuiltinManifest();
  const krea = manifest.models.find((model) => model.id === "krea_2_turbo");
  const record = krea.candle.turboFit.evidenceRecords.find(
    (candidate) => candidate.tier === "q8" && candidate.width === 768,
  );
  assert.ok(record, "fixture needs a q8 768x768 evidence record");
  record.frames = 241;
  const changed = await buildMatrix({
    publish: false,
    sourceOverrides: { manifest: JSON.stringify(manifest) },
  });

  // The point of the fix: `historicalVerification` no longer hand-builds `WxH`, so a record that
  // declares frames is characterized at the geometry it was actually captured at rather than as a
  // one-frame design point that was never measured.
  const characterization = cellById(changed, KREA_Q8_THREE_STAGE).memoryCharacterization;
  assert.deepEqual(characterization.measuredGeometries, ["1024x1024", "768x768xf241"]);
  assert.equal(characterization.status, "point");
  assert.equal(characterization.coveredFrameBound, null);
  // The evidence row itself carries the frames too — a consumer reading the row must see the same
  // geometry the characterization was computed from.
  const row = cellById(changed, KREA_Q8_THREE_STAGE).evidence.historicalVerification.find(
    (candidate) => candidate.geometry.startsWith("768x768"),
  );
  assert.equal(row.geometry, "768x768xf241");

  // Untouched tiers are untouched: q4 shares the code path and moves only if its own records do.
  assert.deepEqual(
    cellById(changed, "krea_2_turbo:krea_2_turbo:candle:q4:text_to_image:none:staged_residency")
      .memoryCharacterization.measuredGeometries,
    ["1024x1024", "768x768"],
  );
});

test("`fitted` may not be published on fewer geometries than the form has coefficients (sc-18812)", () => {
  // The validator's guard, exercised on a forged cell rather than by trusting that
  // `memoryCharacterization` never emits one. The two are deliberately independent: this is what
  // would catch a rank bug that answered `fitted` on too few points.
  const forged = (characterization) => ({
    id: "forged:cell",
    state: "Implemented/unverified",
    memoryCharacterization: characterization,
  });
  assert.throws(
    () =>
      assertCharacterizationIsConsistent(
        forged({
          status: "fitted",
          measuredGeometries: ["1024x1024", "768x768"],
          coveredPixelBound: 1024 * 1024,
          coveredFrameBound: 241,
        }),
      ),
    /fitted on 2 measured geometries, but its curve has 3 coefficients/,
  );
  // The SAME cell without the temporal marker is a legitimate two-coefficient fit and passes, so
  // the rejection above is attributable to the coefficient count and not to the forging.
  assert.doesNotThrow(() =>
    assertCharacterizationIsConsistent(
      forged({
        status: "fitted",
        measuredGeometries: ["1024x1024", "768x768"],
        coveredPixelBound: 1024 * 1024,
      }),
    ),
  );
  // Three geometries satisfy the three-coefficient floor, so the guard is a floor and not a blanket
  // refusal of temporal cells.
  assert.doesNotThrow(() =>
    assertCharacterizationIsConsistent(
      forged({
        status: "fitted",
        measuredGeometries: ["1024x1024", "768x512xf121", "768x512xf241"],
        coveredPixelBound: 1024 * 1024,
        coveredFrameBound: 241,
      }),
    ),
  );
  // And the temporal bound obeys the same "null unless fitted" rule the pixel bound does.
  assert.throws(
    () =>
      assertCharacterizationIsConsistent(
        forged({
          status: "point",
          measuredGeometries: ["768x512xf121", "768x512xf241"],
          coveredPixelBound: null,
          coveredFrameBound: 241,
        }),
      ),
    /coveredFrameBound is only meaningful on a fitted curve/,
  );
});
