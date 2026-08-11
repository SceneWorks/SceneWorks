import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  CONTROL_LANE_MODELS,
  FAMILY_STORIES,
  MODEL_STORIES,
  SOURCE_PATHS,
  assertCellOwnershipIsBackendScoped,
  assertCalibrationPlanTargetsResolvedCoordinates,
  assertCellInventoryMatchesCatalog,
  assertPublishedDocumentIsClosed,
  hoistManifestScopes,
  planEntryTargetsCoordinate,
  assertTwinCoverage,
  backendScopes,
  buildMatrix,
  buildStoryBackendScope,
  isImplemented,
  isPublishableCell,
  plannedCellIds,
  RUNG4_APPLICABILITIES,
  RUNG4_IMPLEMENTATIONS,
  RUNG4_REQUEST_PEAKS,
  familyGroup,
  familyStory,
  mlxRequiredHostBytes,
  observedPeakBytes,
  modelStory,
} from "./generate-memory-matrix.mjs";
import { recordId } from "./memory-calibration-harness.mjs";
import { recordsNeedingDigest } from "./backfill-closure-digests.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";
import { stripInertLines } from "./lib/source-revision.mjs";
import { routedLanes } from "./check-tier-integrity.mjs";

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
  assert.equal(
    observedPeakBytes({ observedMemory: { overall: { activeBytes: 1234, deviceBytes: 5678 } } }),
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
    "rung4Survey",
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
  assert.equal(matrix.schemaVersion, 7);
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
          schemaVersion: 4,
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
    assert.ok(
      implementations.every(
        (implementation) =>
          implementation.fingerprint === "qwen-image-mlx-shared-ladder-2026-08-01-v1",
      ),
      `${modelId} must keep load shape separate from the provider content fingerprint`,
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
  // SC-18353 adds production-deferred physical MLX receipts for q4/bf16. Pinned as the SET, not a
  // count: neither admitting an unbound sibling nor losing any current binding may read as green.
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
  assert.deepEqual(
    Object.fromEntries(
      verified.map((cell) => [cell.id, cell.evidence.currentEnvironmentVerification.length]),
    ),
    {
      "qwen_image:qwen_image:mlx:bf16:text_to_image:none:bounded_attention": 1,
      "qwen_image:qwen_image:mlx:bf16:text_to_image:none:bounded_decode": 1,
      "qwen_image:qwen_image:mlx:bf16:text_to_image:none:bounded_transformer_residency": 1,
      "qwen_image:qwen_image:mlx:q4:text_to_image:none:bounded_attention": 1,
      "qwen_image:qwen_image:mlx:q4:text_to_image:none:bounded_transformer_residency": 1,
      "qwen_image:qwen_image:mlx:q8:text_to_image:none:bounded_attention": 3,
      "qwen_image:qwen_image:mlx:q8:text_to_image:none:bounded_transformer_residency": 3,
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

test("FLUX.2-dev MLX exposes only the captured q4/q8 T2I Resident cells", async () => {
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
    180,
    "the full 3-tier x 4-mode x 3-overlay x 5-rung slice must exist",
  );
  assert.deepEqual(
    shippedCells.filter((cell) => cell.state !== "Missing").map((cell) => cell.id).sort(),
    [
      "flux2_dev:flux2_dev:mlx:q4:text_to_image:none:resident",
      "flux2_dev:flux2_dev:mlx:q8:text_to_image:none:resident",
    ],
    "BF16 and every sibling mode, overlay, and rung must remain Missing",
  );
  assert.ok(
    shippedCells
      .filter((cell) => cell.state !== "Missing")
      .every(
        (cell) =>
          cell.state === "Implemented/unverified" &&
          cell.evidence.currentEnvironmentVerification.length === 0,
      ),
    "the checked-in captures must become historical when the provider closure moves",
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
        cell.evidence.currentEnvironmentVerification.length === 2,
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
      cell.calibrationFingerprint.startsWith("z-image-mlx-independent-materialization-v3") &&
      cell.evidence.staticImplementation.some((entry) =>
        entry.source.includes("mlx-gen-z-image/src/memory_strategy.rs"),
      ),
    ),
    "historical bindings must not mask the pinned MLX provider contract",
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
    assert.deepEqual(ranges.decodeTileEdges, [2048, 768, 640, 512]);
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
  assert.deepEqual(verified, [], "the d480 Z-Image records must remain historical at pin bf06");
  assert.ok(
    allZImageMlx.every((cell) => cell.evidence.currentEnvironmentVerification.length === 0),
    "no historical Z-Image capture may be promoted across an exact inference-pin change",
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
        wiredBytes: 4 * 1024 ** 3,
        reclaimableBytes: 1 * 1024 ** 3,
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
        overall: {
          wiredBytes: 44_056_333_980,
          reclaimableBytes: 3_852_363_372,
        },
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
      observedMemory: { overall: { wiredBytes: 9 * 1024 ** 3, reclaimableBytes: 0 } },
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
  assert.deepEqual([...surveyed].sort(), [...advertised].sort());
  assert.equal(matrix.summary.rung4Survey.surveyedFamilyBackends, advertised.size);
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
    assert.ok(
      cells.every((cell) =>
        cell.modelId === "z_image"
          ? ["Implemented/unverified", "Verified"].includes(cell.state)
          : cell.overlay === "control"
            ? cell.state === "Missing"
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
    "z_image_turbo:z_image_turbo:candle:q4:text_to_image:control:resident",
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

test("rung 4 is not claimed on MLX where its declared rung-1 prerequisite is absent", async () => {
  // gen_core::memory_strategy makes rung 1 a prerequisite of rung 4, so a family with a perfectly
  // windowable trunk still reports Missing where the entry cannot stage its phases. Mage-Flow and
  // SenseNova are the epic's own uncovered-rung-1 MLX families; Candle capability is independent.
  const matrix = await buildMatrix({ publish: false });
  for (const modelId of ["mage_flow", "sensenova_u1_8b"]) {
    const staged = matrix.cells.filter(
      (cell) =>
        cell.modelId === modelId &&
        cell.backend === "mlx" &&
        cell.rung === "staged_residency",
    );
    assert.ok(staged.length > 0);
    assert.ok(
      staged.every((cell) => cell.state === "Missing"),
      `${modelId}: fixture assumes rung 1 is unavailable`,
    );
    const rung4 = matrix.cells.filter(
      (cell) =>
        cell.modelId === modelId &&
        cell.backend === "mlx" &&
        cell.rung === "bounded_transformer_residency",
    );
    assert.ok(
      rung4.every((cell) => cell.state === "Missing"),
      `${modelId}: rung 4 cannot be reachable without rung 1`,
    );
    assert.ok(
      rung4.every((cell) => cell.rung4Survey.structuralApplicability !== "none"),
      `${modelId}: the architecture is fine — the verdict must not read as Structurally N/A`,
    );
  }
});

test("the rung-1 prerequisite gates the rung-4 claim, and is the ONLY thing separating these two families", async () => {
  // The mutation check for the prerequisite. Both fixtures below claim a rung-4 implementation the
  // real providers do not have; the ONLY difference between them is whether the entry advertises
  // rung 1. Mage-Flow does not (the epic's uncovered-rung-1 set), SANA does. If the rung-4 arm
  // stopped consulting `stagedResidencyIsAvailable`, the Mage-Flow half would go green — which is
  // exactly the false claim `gen_core::memory_strategy`'s prerequisite edge exists to prevent.
  const claimImplemented = async (group, entry) => {
    const survey = await surveyFixture();
    const verdict = survey.families[group].backends.mlx;
    verdict.implementation = "shared-primitive";
    verdict.implementedEntries = [entry];
    verdict.strategyParameters = { transformerWindowSize: 1 };
    const matrix = await buildMatrix({
      publish: false,
      sourceOverrides: { rung4Survey: JSON.stringify(survey) },
    });
    const of = (rung) =>
      matrix.cells.filter(
        (cell) => cell.modelId === entry && cell.backend === "mlx" && cell.rung === rung,
      );
    return { rung1: of("staged_residency"), rung4: of("bounded_transformer_residency") };
  };

  const withoutRung1 = await claimImplemented("15509", "mage_flow");
  assert.ok(withoutRung1.rung1.length > 0 && withoutRung1.rung4.length > 0);
  assert.ok(
    withoutRung1.rung1.every((cell) => cell.state === "Missing"),
    "fixture assumes Mage-Flow advertises no MLX rung 1",
  );
  assert.ok(
    withoutRung1.rung4.every((cell) => cell.state === "Missing"),
    "a rung-4 implementation claim must not survive an absent rung-1 prerequisite",
  );

  const withRung1 = await claimImplemented("15523", "sana_1600m");
  assert.ok(
    withRung1.rung1.every((cell) => cell.state === "Implemented/unverified"),
    "fixture assumes SANA advertises MLX rung 1",
  );
  assert.ok(
    withRung1.rung4.every(
      (cell) =>
        cell.state === "Implemented/unverified" &&
        cell.strategyParameters.transformerWindowSize === 1,
    ),
    "with the prerequisite present the same claim IS honoured — so the assertion above is not vacuous",
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
  assert.equal(
    matrix.cells.filter((cell) => cell.rung4Survey?.structuralApplicability === "none").length,
    0,
    "no family in the catalog is architecturally inapplicable, so no cell may publish `none`",
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
  for (const properties of [
    schema.$defs.rung4SurveyVerdict.properties,
    schema.properties.rung4SurveyRows.items.properties,
  ]) {
    assert.equal(properties.structuralApplicability.$ref, "#/$defs/rung4StructuralApplicability");
    assert.equal(properties.implementation.$ref, "#/$defs/rung4Implementation");
    assert.equal(properties.requestPeak.$ref, "#/$defs/rung4RequestPeak");
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
  assert.ok(cells.filter((cell) => cell.modelId === "z_image_turbo").every((cell) => cell.state === "Missing"));
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
  const matrix = await buildMatrix({
    publish: false,
    sourceOverrides: { rung4Survey: JSON.stringify(stated) },
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

/// Retained Qwen production-deferred records using the shared fingerprint, re-stamped current.
///
/// Selected by their calibration fingerprint, which is what separates them from the 22 older Qwen
/// records in the bundle: the rung-4 ingest and SC-18237/SC-18353 production-deferred records carry
/// the bare `qwen-image-mlx-shared-ladder-2026-08-01-v1`, while older captures carry the `-eager` /
/// `-deferred` fingerprint variants. Q8 retains the historical rung-4 set used by this fixture;
/// Q4/BF16 select only SC-18353's final capture revision so older same-fingerprint attempts cannot
/// silently increase the exact per-cell evidence counts.
const QWEN_RUNG4_FINGERPRINT = "qwen-image-mlx-shared-ladder-2026-08-01-v1";
const QWEN_PRODUCTION_DEFERRED_REVISION = "014134e3035ad7e4eca5c2ed7bded2375dc3c071";
const qwenRung4OnCurrentPin = () =>
  currentEvidenceFixture({
    select: (record) =>
      record.target.provider === "qwen_image" &&
      record.calibrationFingerprint === QWEN_RUNG4_FINGERPRINT &&
      (record.target.tier === "q8" ||
        record.repositories.inference.revision === QWEN_PRODUCTION_DEFERRED_REVISION),
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
    "the retained q8 fixture plus the physical q4/bf16 captures must verify all nine bindings",
  );
  assert.equal(
    verifiedQwen(shipped),
    0,
    "the shipped Qwen captures must become historical when the provider closure moves",
  );

  const evidenceOnlyZ = await buildMatrix({
    publish: false,
    sourceOverrides: {
      calibrationEvidence: await currentEvidenceFixture({
        select: (record) => record.target.provider === "z_image_turbo",
      }),
    },
  });
  assert.equal(
    evidenceOnlyZ.cells.filter(
      (cell) => cell.modelId === "z_image_turbo" && cell.backend === "mlx" &&
        cell.state === "Verified",
    ).length,
    0,
    "current evidence cannot promote through a historical exact manifest binding",
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
  const withClosures = (mutate) => {
    const next = structuredClone(closures);
    next.inferenceRevision = movedPin;
    mutate(next.providers);
    return JSON.stringify(next, null, 2);
  };

  const pinOnlyQwen = await buildMatrix({
    publish: false,
    sourceOverrides: {
      calibrationEvidence: qwenOnCurrentPin,
      manifest: qwenManifestOnCurrentPin,
      cargo: withPin(movedPin),
      inferenceClosures: withClosures(() => {}),
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
      inferenceClosures: withClosures((providers) => {
        providers["mlx:z_image_turbo"].digest = "f".repeat(64);
      }),
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
      inferenceClosures: withClosures((providers) => {
        providers["mlx:qwen_image"].digest = "e".repeat(64);
      }),
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
  const plan = JSON.parse(
    await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url), "utf8"),
  );

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

  // The seventh arm, `currentEnvironmentVerification`, is empty at this pin because the rich
  // capability-descriptor change moved the shared MLX provider closure. The physical Qwen and
  // FLUX.2 captures remain historical until they are actually re-captured; they are not re-stamped.
  // Two facts keep this assertion useful:
  //
  //   1. It is exact: no historical or sibling-rung row may join merely because the pin moved.
  //   2. It is SUBSUMED. A current run is an eligible run, and `memoryCharacterization` counts every
  //      eligible run's geometry, so a cell carrying current evidence is `point` or `fitted` and the
  //      measured arm already admits it. The arm being empty therefore cannot elide anything.
  //
  // Asserted as an exact set so another recapture flips this test rather than silently passing, and
  // the field's presence is asserted separately so a rename cannot make the arm quietly vanish.
  assert.deepEqual(
    resolved.cells
      .filter((cell) => cell.evidence.currentEnvironmentVerification.length > 0)
      .map((cell) => cell.id)
      .sort(),
    [],
    "provider closure drift must demote every formerly current capture without re-stamping it",
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

  // `mlxStagedStaticCoverage` is a claim about all 53 entries and must not shrink to the published
  // sample. Recomputed here from the census with the generator's own implementation predicate.
  const stagedFromCensus = new Set(
    matrix.coverage
      .filter((row) => row.backend === "mlx" && row.rung === "staged_residency" && row.implemented > 0)
      .map((row) => row.modelId),
  );
  assert.equal(stagedFromCensus.size, matrix.summary.mlxStagedStaticCoverage);
  assert.equal(
    stagedFromCensus.size,
    new Set(
      resolved.cells
        .filter(
          (cell) =>
            cell.backend === "mlx" && cell.rung === "staged_residency" && isImplemented(cell.state),
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
  const plan = JSON.parse(
    await readFile(new URL("../config/memory-calibration-plan.json", import.meta.url), "utf8"),
  );

  // The shipped plan is clean, which is the property worth pinning: every entry addresses something.
  const resolved = await buildMatrix({ publish: false });
  assert.doesNotThrow(() => assertCalibrationPlanTargetsResolvedCoordinates(plan, resolved.cells));

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
