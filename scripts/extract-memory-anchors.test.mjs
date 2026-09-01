import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test, { before } from "node:test";
import { fileURLToPath } from "node:url";

import { buildMatrix } from "./generate-memory-matrix.mjs";
import {
  ANALYTIC_BASES,
  MEMORY_ANCHOR_SCHEMA_VERSION,
  PACKAGED_SOURCES_PATH,
  STORE_PATH,
  anchorCandidate,
  assertPackagedSources,
  buildAnchorStore,
  catalogCells,
  cellKey,
  envelopeEvidence,
  identityKey,
  inferencePin,
  inferenceProviderConstants,
  locateInferenceCheckout,
  manifestTierEvidence,
  packagedAnchorSources,
  providerByteConstants,
  selectRepresentative,
  serialiseStore,
} from "./extract-memory-anchors.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

// `buildMatrix` resolves the whole catalog; resolve it once and share it. Every build below uses
// the DEFAULT resolution — the same one `main()` and `npm run check:memory-anchors` use — so these
// assertions cover the shipped code path and are a property of the repo rather than of this host.
let matrix;
let store;
let cells;

before(async () => {
  matrix = await buildMatrix();
  store = await buildAnchorStore({ matrix });
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
  const first = serialiseStore(await buildAnchorStore({ matrix }));
  const second = serialiseStore(await buildAnchorStore({ matrix }));
  assert.equal(first, second);
});

test("the store is a pure function of the committed evidence, not of iteration order", async () => {
  // The previous output is not an input: there is no carry-forward, so `--check` (which rebuilds
  // and compares) asks exactly the question a regeneration answers, and no row can survive on the
  // strength of having been written once.
  // A hand-inserted row — the shape `--check` used to seed itself with — does not survive a
  // rebuild, whether it cites a walked corpus or one this run never sees.
  const unanchored = cells.find(
    (cell) =>
      !store.anchors.some(
        (anchor) =>
          cellKey(anchor.modelId, anchor.backend, anchor.tier) ===
          cellKey(cell.modelId, cell.backend, cell.tier),
      ),
  );
  assert.ok(unanchored, "the catalog has a cell no anchor covers");
  const handWritten = (id, source) => ({
    ...store.anchors[0],
    id,
    modelId: unanchored.modelId,
    backend: unanchored.backend,
    tier: unanchored.tier,
    transformerVariant: null,
    decoder: null,
    source: { ...store.anchors[0].source, ...source },
  });
  const seeded = await buildAnchorStore({
    matrix,
    existingStore: {
      schemaVersion: 1,
      anchors: [
        // One citing a corpus this run walks, one citing evidence it never sees: neither is
        // re-derivable from the committed tree, so neither may reach the store.
        handWritten("hand:inserted", {}),
        handWritten("hand:outside-the-corpora", { path: "docs/proving/sc-22509-candle.json" }),
      ],
      analyticOnly: [],
    },
  });
  assert.deepEqual(
    seeded.anchors.filter((anchor) => anchor.id.startsWith("hand:")),
    [],
    "no row survives a rebuild on the strength of having been written once",
  );
  assert.equal(serialiseStore(seeded), serialiseStore(store));

  const explicitlyOff = await buildAnchorStore({ matrix, inferenceRoot: null });
  assert.equal(
    serialiseStore(store),
    serialiseStore(explicitlyOff),
    "the default resolution reads no inference checkout, so the output is host-independent",
  );
  const ids = store.anchors.map((anchor) => anchor.id);
  assert.deepEqual(ids, [...ids].sort(), "anchors are emitted in a stable sorted order");
  const analyticIds = store.analyticOnly.map((entry) => entry.id);
  assert.deepEqual(analyticIds, [...analyticIds].sort(), "analytic rows are emitted sorted");
});

// ---------------------------------------------------------------------------------------------
// The anchor store and the Rust loader's compiled-in evidence list are two halves of one contract.
// Nothing else cross-checks them: `validate_anchor` hard-rejects an anchor whose `source.path` is
// absent from `PACKAGED_MEMORY_ANCHOR_SOURCES`, so a store citing anything else is unloadable.
// This is also how a concurrent story's anchors (sc-22509) reach the store: its corpus lands under
// a walked root AND in that list, and the regeneration derives its rows again.
// ---------------------------------------------------------------------------------------------

test("every anchor cites a corpus compiled into PACKAGED_MEMORY_ANCHOR_SOURCES", async () => {
  const packaged = packagedAnchorSources(
    await readFile(path.join(ROOT, PACKAGED_SOURCES_PATH), "utf8"),
  );
  assert.ok(packaged.size > 0, "the compiled-in evidence list must parse");
  const foreign = store.anchors
    .filter((anchor) => !packaged.has(anchor.source.path))
    .map((anchor) => `${anchor.id} -> ${anchor.source.path}`);
  assert.deepEqual(
    foreign,
    [],
    `every anchor's source must be compiled in via ${PACKAGED_SOURCES_PATH}, or the Rust loader ` +
      "rejects the store as citing a file that is not retained evidence",
  );
});

test("an anchor citing evidence outside the compiled-in list is refused, not emitted", () => {
  const packaged = new Set(["docs/generated/memory-calibration-evidence.json"]);
  assert.doesNotThrow(() =>
    assertPackagedSources(
      [{ id: "ok", source: { path: "docs/generated/memory-calibration-evidence.json" } }],
      packaged,
    ),
  );
  assert.throws(
    () =>
      assertPackagedSources(
        [{ id: "foreign:concurrent-story", source: { path: "docs/proving/sc-22509-candle.json" } }],
        packaged,
      ),
    /docs\/proving\/sc-22509-candle\.json/,
    "a row whose corpus is not compiled in would not load, so it must not be written",
  );
});

test("the compiled-in evidence list is parsed from its declaration, not guessed", () => {
  const source = [
    "const PACKAGED_MEMORY_ANCHOR_SOURCES: &[(&str, &str)] = &[",
    '    ("docs/calibration/one.json", include_str!("../../../docs/calibration/one.json")),',
    '    ("docs/generated/two.json", include_str!("../../../docs/generated/two.json")),',
    "];",
    "",
    'const OTHER: &str = include_str!("../../../docs/generated/three.json");',
  ].join("\n");
  assert.deepEqual(
    [...packagedAnchorSources(source)].sort(),
    ["docs/calibration/one.json", "docs/generated/two.json"],
    "an include_str! outside the list is not part of the anchors' source domain",
  );
  assert.throws(
    () => packagedAnchorSources("const SOMETHING_ELSE: u32 = 1;"),
    /no longer declares PACKAGED_MEMORY_ANCHOR_SOURCES/,
  );
});

test("an overlay render does not anchor the base cell", () => {
  // krea's q4 MLX evidence is control-branch-only, under its own `*_control` provider: it measures
  // a different resident set, so it may bound the cell but never anchor it.
  for (const anchor of store.anchors) {
    assert.equal(anchor.overlay, null, `${anchor.id}: an overlay render must not anchor a cell`);
  }
  // SHAPE, not census: if the overlay-only evidence ever leaves the corpus there is nothing to
  // assert about, and failing here would red a measurement's retirement rather than a defect.
  const overlayEvidenced = store.analyticOnly.filter((entry) => entry.evidence?.values?.overlay);
  for (const entry of overlayEvidenced) {
    assert.equal(entry.basis, "measured_envelope");
    assert.match(entry.reason, /overlay/, `${entry.id}: the row must say why it is not anchored`);
  }
});

test("a clean render outranks an overlay render of the same cell, whatever its envelope", () => {
  // The rule anchors already follow, applied to the weaker evidence too: an overlay envelope is a
  // measurement of a DIFFERENT resident set, so it may only speak for a cell nothing clean covers.
  const cell = { modelId: "example", backend: "mlx", tier: "q4" };
  const render = (id, allocatorBytes, overlay) => ({
    id,
    backend: "mlx",
    observedMemory: { overall: { allocatorBytes } },
    target: { modelId: "example", tier: "q4", provider: "example", overlay },
  });
  const corpora = [
    {
      path: "docs/generated/example.json",
      sha256: "a".repeat(64),
      records: [render("overlay", 900, "control:1"), render("clean", 100, "none")],
    },
  ];
  const chosen = envelopeEvidence(corpora, cell);
  assert.equal(chosen.recordId, "clean", "the smaller CLEAN envelope wins over a larger overlay");
  assert.equal(chosen.values.overlay, undefined, "a clean row cites no overlay");

  const overlayOnly = envelopeEvidence(
    [{ ...corpora[0], records: [render("overlay", 900, "control:1")] }],
    cell,
  );
  assert.equal(overlayOnly.recordId, "overlay", "overlay evidence is still cited when it is all");
  assert.equal(overlayOnly.values.overlay, "control:1");
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

test("a catalog cell that cannot name its family or route fails by name, not by serde", async () => {
  // Both fields are non-optional on the Rust side; `JSON.stringify` DROPS an undefined value, so an
  // unguarded miss would ship a store whose only symptom is a "missing field" from the loader.
  const model = {
    id: "example",
    backends: ["mlx"],
    axes: { mlx: { tiers: ["q4"] } },
    family: "example-family",
    familyGroup: null,
    resolvedRoute: null,
    resolvedRoutes: { mlx: "example_route" },
    modality: "image",
  };
  assert.deepEqual(
    (await catalogCells({ models: [model] })).map((cell) => [cell.modelFamily, cell.route]),
    [["example-family", "example_route"]],
  );
  await assert.rejects(
    () => catalogCells({ models: [{ ...model, family: null }] }),
    /example \(mlx\): the routing catalog publishes null for family\/familyGroup/,
  );
  await assert.rejects(
    () => catalogCells({ models: [{ ...model, resolvedRoutes: {} }] }),
    /example \(mlx\): the routing catalog publishes null for resolvedRoutes\.mlx\/resolvedRoute/,
  );
});

const writeProviderCrate = async (root, crate, body) => {
  const dir = path.join(root, "crates/media/mlx-gen", crate, "src");
  await mkdir(dir, { recursive: true });
  await writeFile(path.join(dir, "memory_strategy.rs"), body);
};

test("provider constants are read from the longest-prefix crate, at the cited revision", async () => {
  // Drives the checkout limb against a directory SHAPED like the pinned inference tree, so the
  // opt-in path is exercised without a real cargo checkout of it.
  const checkout = await mkdtemp(path.join(os.tmpdir(), "anchor-inference-"));
  await writeProviderCrate(checkout, "mlx-gen-flux", "pub const FLUX_BYTES: u64 = 1;\n");
  await writeProviderCrate(checkout, "mlx-gen-flux2", "pub const FLUX2_BYTES: u64 = 2_000;\n");
  await writeProviderCrate(checkout, "mlx-gen-z", "pub const SHORT_PREFIX_BYTES: u64 = 4;\n");
  await writeProviderCrate(checkout, "mlx-gen-z-image", "pub const Z_BYTES: u64 = 3;\n");
  await writeProviderCrate(checkout, "mlx-gen-empty", "pub fn nothing() {}\n");
  const revision = "d".repeat(40);
  const constants = await inferenceProviderConstants(
    checkout,
    revision,
    new Set(["flux2_dev", "z_image_turbo", "empty", "unmatched_model"]),
  );

  assert.deepEqual([...constants.keys()].sort(), ["flux2_dev", "z_image_turbo"]);
  const flux2 = constants.get("flux2_dev");
  assert.deepEqual(
    flux2.values,
    { FLUX2_BYTES: "2000" },
    "`flux2_dev` matches mlx-gen-flux2, not the shorter mlx-gen-flux prefix",
  );
  assert.equal(flux2.repo, "SceneWorks/inference");
  assert.equal(flux2.revision, revision, "a foreign citation names the revision it was read at");
  assert.equal(flux2.path, "crates/media/mlx-gen/mlx-gen-flux2/src/memory_strategy.rs");
  assert.match(flux2.sha256, /^[0-9a-f]{64}$/);
  assert.deepEqual(
    constants.get("z_image_turbo").values,
    { Z_BYTES: "3" },
    "`z_image_turbo` matches mlx-gen-z-image, though mlx-gen-z is also a legal prefix of it",
  );
  assert.equal(constants.get("empty"), undefined, "a crate with no constants contributes nothing");

  assert.equal(
    (await inferenceProviderConstants(null, revision, new Set(["flux2_dev"]))).size,
    0,
    "with no checkout the limb reads nothing at all — the default for every generation",
  );
});

test("the pinned checkout is located by the pin's own short hash, or not at all", async () => {
  const home = await mkdtemp(path.join(os.tmpdir(), "anchor-home-"));
  const revision = "0123456789abcdef0123456789abcdef01234567";
  const checkouts = path.join(home, ".cargo/git/checkouts");
  await mkdir(path.join(checkouts, "inference-9f0e1d2c", revision.slice(0, 7)), {
    recursive: true,
  });
  await mkdir(path.join(checkouts, "inference-9f0e1d2c", "badc0de"), { recursive: true });
  await mkdir(path.join(checkouts, "mlx-rs-1a2b3c4d", revision.slice(0, 7)), { recursive: true });

  assert.equal(
    await locateInferenceCheckout(revision, home),
    path.join(checkouts, "inference-9f0e1d2c", revision.slice(0, 7)),
    "only a directory whose name prefixes THIS revision, under an inference remote, answers",
  );
  assert.equal(
    await locateInferenceCheckout("f".repeat(40), home),
    null,
    "a host holding some other revision resolves to nothing rather than to the wrong tree",
  );
  assert.equal(await locateInferenceCheckout(revision, path.join(home, "absent")), null);
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
