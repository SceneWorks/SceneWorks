import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test, { before } from "node:test";
import { fileURLToPath } from "node:url";

import { buildMatrix } from "./generate-memory-matrix.mjs";
import {
  ANALYTIC_BASES,
  MEMORY_ANCHOR_SCHEMA_VERSION,
  STORE_PATH,
  anchorCandidate,
  buildAnchorStore,
  catalogCells,
  cellKey,
  identityKey,
  inferencePin,
  manifestTierEvidence,
  providerByteConstants,
  selectRepresentative,
  serialiseStore,
} from "./extract-memory-anchors.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// `buildMatrix` resolves the whole catalog; resolve it once and share it, and pin the inference
// checkout OFF so every assertion below is a property of the repo rather than of this host.
let matrix;
let store;
let cells;

before(async () => {
  matrix = await buildMatrix();
  store = await buildAnchorStore({ matrix, inferenceRoot: null });
  cells = await catalogCells(matrix);
});

// ---------------------------------------------------------------------------------------------
// AC 1 — every routing-catalog cell resolves to an anchor or an explicit analytic-only entry, with
// zero unclassified cells. SHAPE only: this must never pin how many models or anchors exist, or it
// becomes a frozen-corpus gate that a catalog addition fails for the wrong reason.
// ---------------------------------------------------------------------------------------------

test("every catalog model x tier x lane is classified, with nothing unclassified", () => {
  const anchored = new Set(
    store.anchors.map((anchor) => cellKey(anchor.modelId, anchor.backend, anchor.tier)),
  );
  const analytic = new Set(
    store.analyticOnly.map((entry) => cellKey(entry.modelId, entry.backend, entry.tier)),
  );
  const catalog = new Set(cells.map((cell) => cellKey(cell.modelId, cell.backend, cell.tier)));

  const unclassified = [...catalog].filter((cell) => !anchored.has(cell) && !analytic.has(cell));
  assert.deepEqual(unclassified, [], "every catalog cell must carry a classification");

  const both = [...catalog].filter((cell) => anchored.has(cell) && analytic.has(cell));
  assert.deepEqual(both, [], "a cell is classified exactly once, never twice");

  const foreign = [...anchored, ...analytic].filter((cell) => !catalog.has(cell));
  assert.deepEqual(foreign, [], "the store must not carry a row the routing catalog cannot reach");
});

test("the classification is shaped like the catalog, not like one corpus", () => {
  // Shape guards so the coverage assertion above cannot pass vacuously: both classifications are
  // populated, both backend lanes appear, and more than one model is covered.
  assert.ok(store.anchors.length > 0, "the migration must extract anchors");
  assert.ok(store.analyticOnly.length > 0, "unanchorable cells must be declared, not omitted");
  assert.equal(store.schemaVersion, MEMORY_ANCHOR_SCHEMA_VERSION);
  const lanes = new Set([
    ...store.anchors.map((anchor) => anchor.backend),
    ...store.analyticOnly.map((entry) => entry.backend),
  ]);
  assert.deepEqual([...lanes].sort(), ["candle", "mlx"], "both backend lanes are classified");
  assert.ok(
    new Set(store.anchors.map((anchor) => anchor.modelId)).size > 1,
    "anchors must cover more than one model",
  );
  for (const entry of store.analyticOnly) {
    assert.ok(ANALYTIC_BASES.includes(entry.basis), `${entry.id}: unknown basis ${entry.basis}`);
    assert.ok(entry.reason.trim().length > 0, `${entry.id}: a classification must state why`);
    assert.equal(
      entry.evidence === null,
      entry.basis === "no_retained_evidence",
      `${entry.id}: the basis and the cited evidence must agree`,
    );
  }
  for (const anchor of store.anchors) {
    for (const phase of ["conditioning", "denoise", "decode"]) {
      assert.ok(
        Number.isInteger(anchor.phaseActivePeakBytes[phase]) &&
          anchor.phaseActivePeakBytes[phase] > 0,
        `${anchor.id}: an anchor carries a measured ${phase} peak`,
      );
    }
    assert.ok(anchor.source.path.length > 0 && anchor.source.sha256.length === 64);
  }
});

test("the checked-in store is what the extractor produces", async () => {
  const committed = await readFile(path.join(ROOT, STORE_PATH), "utf8");
  assert.equal(
    serialiseStore(store),
    committed,
    `${STORE_PATH} is stale — re-run scripts/extract-memory-anchors.mjs`,
  );
});

// ---------------------------------------------------------------------------------------------
// AC 2 — extraction is deterministic.
// ---------------------------------------------------------------------------------------------

test("re-running over the same corpora reproduces the store byte-identically", async () => {
  const first = serialiseStore(await buildAnchorStore({ matrix, inferenceRoot: null }));
  const second = serialiseStore(await buildAnchorStore({ matrix, inferenceRoot: null }));
  assert.equal(first, second);
});

test("the store is a pure function of its inputs, not of iteration order", async () => {
  // Feeding the run its own output must not move it: idempotence is what lets the extractor be
  // re-run after a concurrent story lands an anchor.
  const once = await buildAnchorStore({ matrix, inferenceRoot: null });
  const twice = await buildAnchorStore({ matrix, existingStore: once, inferenceRoot: null });
  assert.equal(serialiseStore(once), serialiseStore(twice));
  const ids = store.anchors.map((anchor) => anchor.id);
  assert.deepEqual(ids, [...ids].sort(), "anchors are emitted in a stable sorted order");
  const analyticIds = store.analyticOnly.map((entry) => entry.id);
  assert.deepEqual(analyticIds, [...analyticIds].sort(), "analytic rows are emitted sorted");
});

test("an anchor this run does not produce is carried forward, never clobbered", async () => {
  // sc-22509 lands candle proving-model anchors concurrently; a re-extraction must union with
  // them rather than delete every row it did not write itself.
  const foreignCell = cells.find(
    (cell) =>
      cell.backend === "candle" &&
      !store.anchors.some(
        (anchor) => cellKey(anchor.modelId, anchor.backend, anchor.tier) === cellKey(cell.modelId, cell.backend, cell.tier),
      ),
  );
  assert.ok(foreignCell, "the catalog has an unanchored candle cell to stand in for sc-22509");
  const foreign = {
    ...store.anchors[0],
    id: "foreign:concurrent-story",
    modelId: foreignCell.modelId,
    backend: foreignCell.backend,
    tier: foreignCell.tier,
    transformerVariant: null,
    decoder: null,
    // Cites evidence OUTSIDE the corpora this run walks, which is what makes it another story's
    // row rather than one this run withdrew.
    source: { ...store.anchors[0].source, path: "docs/proving/sc-22509-candle.json" },
  };
  const merged = await buildAnchorStore({
    matrix,
    existingStore: { schemaVersion: 1, anchors: [foreign], analyticOnly: [] },
    inferenceRoot: null,
  });
  assert.ok(
    merged.anchors.some((anchor) => anchor.id === "foreign:concurrent-story"),
    "a foreign anchor identity survives re-extraction",
  );
  assert.ok(
    !merged.analyticOnly.some(
      (entry) => cellKey(entry.modelId, entry.backend, entry.tier) === cellKey(foreignCell.modelId, foreignCell.backend, foreignCell.tier),
    ),
    "the cell the foreign anchor covers stops being analytic-only",
  );
  assert.equal(
    merged.anchors.length + merged.analyticOnly.length,
    store.anchors.length + store.analyticOnly.length,
    "the union reclassifies a cell, it does not duplicate it",
  );
});

test("a row this run rejected is withdrawn, not made immortal by the carry-forward", async () => {
  // The counterpart of the union rule: authority is scoped to the corpora the run walked, so a
  // stale row citing one of them (an overlay render, a record that lost its phase peaks) does not
  // survive by sitting in the previous output.
  // A REAL catalog cell this run classifies as analytic-only, so the row is withdrawn by the
  // authority rule rather than by the catalog filter.
  const withdrawn = store.analyticOnly.find(
    (entry) => entry.evidence?.values?.overlay || entry.basis === "measured_envelope",
  );
  assert.ok(withdrawn, "a catalog cell the run declines to anchor exists");
  const stale = {
    ...store.anchors[0],
    id: "stale:withdrawn",
    modelId: withdrawn.modelId,
    backend: withdrawn.backend,
    tier: withdrawn.tier,
    transformerVariant: null,
    decoder: null,
    // Cites a corpus this run WALKED: the run saw that evidence and did not anchor from it.
    source: { ...store.anchors[0].source, path: store.anchors[0].source.path },
  };
  const merged = await buildAnchorStore({
    matrix,
    existingStore: { schemaVersion: 1, anchors: [stale], analyticOnly: [] },
    inferenceRoot: null,
  });
  assert.ok(!merged.anchors.some((anchor) => anchor.id === "stale:withdrawn"));
});

test("an overlay render does not anchor the base cell", () => {
  // krea's q4 MLX evidence is control-branch-only, under its own `*_control` provider: it measures
  // a different resident set, so it may bound the cell but never anchor it.
  for (const anchor of store.anchors) {
    assert.equal(anchor.overlay, null, `${anchor.id}: an overlay render must not anchor a cell`);
  }
  const overlayEvidenced = store.analyticOnly.filter(
    (entry) => entry.evidence?.values?.overlay,
  );
  assert.ok(
    overlayEvidenced.length > 0,
    "the overlay-only evidence must still be cited, not discarded",
  );
  for (const entry of overlayEvidenced) {
    assert.equal(entry.basis, "measured_envelope");
    assert.match(entry.reason, /overlay/, `${entry.id}: the row must say why it is not anchored`);
  }
});

// ---------------------------------------------------------------------------------------------
// The extraction rules themselves.
// ---------------------------------------------------------------------------------------------

const corpus = { path: "docs/generated/example.json", sha256: "a".repeat(64) };

const record = (overrides = {}) => ({
  id: "imc-example",
  backend: "mlx",
  loadShape: "eager_materialization",
  calibrationFingerprint: "example-v1",
  observedMemory: { overall: { allocatorBytes: 10 } },
  strategy: { engagedRungs: ["resident"] },
  target: {
    modelId: "example",
    tier: "q4",
    mode: "text_to_image",
    provider: "example",
    geometry: { width: 1024, height: 1024, frames: 1 },
  },
  diagnostics: {
    measurements: [
      { name: "conditioningActivePeak", value: 1 },
      { name: "denoiseActivePeak", value: 2 },
      { name: "decodeActivePeak", value: 3 },
    ],
  },
  ...overrides,
});

test("a record missing any phase peak cannot anchor", () => {
  assert.ok(anchorCandidate(record(), corpus) !== null);
  const partial = record();
  partial.diagnostics.measurements.pop();
  assert.equal(anchorCandidate(partial, corpus), null);
  assert.equal(
    anchorCandidate(record({ observedMemory: { overall: {} } }), corpus),
    null,
    "an envelope-free record cannot anchor",
  );
});

test("the representative is the largest envelope, tie-broken by path then record id", () => {
  const make = (id, envelope, sourcePath) => ({
    overallAllocatorEnvelopeBytes: envelope,
    recordId: id,
    sourcePath,
  });
  assert.equal(
    selectRepresentative([make("b", 1, "a.json"), make("a", 2, "z.json")]).recordId,
    "a",
  );
  assert.equal(
    selectRepresentative([make("b", 2, "z.json"), make("a", 2, "a.json")]).recordId,
    "a",
  );
  assert.equal(
    selectRepresentative([make("b", 2, "a.json"), make("a", 2, "a.json")]).recordId,
    "a",
  );
  // Total: the same candidates in any order select the same record.
  const candidates = [make("b", 2, "a.json"), make("a", 2, "a.json"), make("c", 1, "a.json")];
  assert.equal(
    selectRepresentative([...candidates].reverse()).recordId,
    selectRepresentative(candidates).recordId,
  );
});

test("an axis-free record keys on a spelling no stated axis can produce", () => {
  const axisFree = anchorCandidate(record(), corpus);
  const stated = anchorCandidate(
    record({
      target: { ...record().target, transformerVariant: "dev", decoder: "conv" },
    }),
    corpus,
  );
  assert.notEqual(identityKey(axisFree), identityKey(stated));
  assert.ok(identityKey(axisFree).endsWith(":-:-"));
});

test("only measured manifest tier tables become evidence", () => {
  const manifest = {
    models: [
      {
        id: "example",
        candle: { measured: true, vramGbByTier: { q4: 18.4 }, sequentialPeakGb: { q4: 5.7 } },
        mlx: { measured: false, vramGbByTier: { q4: 18.4 } },
      },
    ],
  };
  const cell = { modelId: "example", backend: "candle", tier: "q4" };
  const evidence = manifestTierEvidence(manifest, "manifest.jsonc", "b".repeat(64), cell);
  assert.deepEqual(evidence.values, { vramGbByTier: "18.4", sequentialPeakGb: "5.7" });
  assert.equal(
    manifestTierEvidence(manifest, "manifest.jsonc", "b".repeat(64), { ...cell, backend: "mlx" }),
    null,
    "an unmeasured declaration is not evidence",
  );
  assert.equal(
    manifestTierEvidence(manifest, "manifest.jsonc", "b".repeat(64), { ...cell, tier: "bf16" }),
    null,
    "a tier the table does not declare is not evidence",
  );
});

test("only top-level provider constants are read as provider facts", () => {
  const source = [
    "pub const TEXT_ENCODER_BYTES: u64 = 66_714_912_872;",
    "pub const NOT_BYTES: u64 = 5;",
    "    const FIXTURE_BYTES: u64 = 1_000;",
    "fn helper() { const INNER_BYTES: u64 = 7; }",
  ].join("\n");
  assert.deepEqual(providerByteConstants(source), {
    TEXT_ENCODER_BYTES: "66714912872",
    NOT_BYTES: "5",
  });
});

test("the inference pin is read from the inference remote alone", () => {
  const cargo = [
    'candle-gen = { git = "https://github.com/SceneWorks/inference", rev = "' + "a".repeat(40) + '" }',
    'mlx-rs = { git = "https://github.com/michaeltrefry/mlx-rs", rev = "' + "b".repeat(40) + '" }',
  ].join("\n");
  assert.equal(inferencePin(cargo), "a".repeat(40));
  assert.throws(
    () =>
      inferencePin(
        cargo +
          '\ncandle-gen-ltx = { git = "https://github.com/SceneWorks/inference", rev = "' +
          "c".repeat(40) +
          '" }',
      ),
    /distinct inference revisions/,
  );
});
