import assert from "node:assert/strict";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test, { before } from "node:test";
import { fileURLToPath } from "node:url";

import { stripJsoncComments } from "./lib/jsonc.mjs";
import { buildMatrix } from "./generate-memory-matrix.mjs";
import {
  ANALYTIC_BASES,
  ANCHOR_LOADER_CLOSURES_PATH,
  CONTRACT_LADDER_BACKENDS,
  MANIFEST_PATH,
  stagedResidencyExemptLanes,
  MEMORY_ANCHOR_SCHEMA_VERSION,
  PACKAGED_SOURCES_PATH,
  STORE_PATH,
  anchorCandidate,
  assertEveryDerivableCorpusIsPackaged,
  assertPackagedSources,
  buildAnchorStore,
  catalogCells,
  cellKey,
  envelopeEvidence,
  identityKey,
  inferencePin,
  isDerivable,
  inferenceProviderConstants,
  loadCorpora,
  LTX25_VARIANT_COMPONENTS,
  LTX25_WEIGHTS_INVENTORY_PATH,
  ltx25ComponentDeltas,
  phaseAllocatorEnvelopes,
  underivedReasonFor,
  loaderClosureDigestFor,
  locateInferenceCheckout,
  manifestTierEvidence,
  manifestSequentialRow,
  isReceiptPricedRoute,
  RECEIPT_PRICED_ROUTES,
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
    store.anchors.map((anchor) =>
      cellKey(anchor.modelId, anchor.backend, anchor.tier),
    ),
  );
  const analytic = new Set(
    store.analyticOnly.map((entry) =>
      cellKey(entry.modelId, entry.backend, entry.tier),
    ),
  );
  const catalog = new Set(
    cells.map((cell) => cellKey(cell.modelId, cell.backend, cell.tier)),
  );

  const unclassified = [...catalog].filter(
    (cell) => !anchored.has(cell) && !analytic.has(cell),
  );
  assert.deepEqual(
    unclassified,
    [],
    "every catalog cell must carry a classification",
  );

  const both = [...catalog].filter(
    (cell) => anchored.has(cell) && analytic.has(cell),
  );
  assert.deepEqual(both, [], "a cell is classified exactly once, never twice");

  const foreign = [...anchored, ...analytic].filter(
    (cell) => !catalog.has(cell),
  );
  assert.deepEqual(
    foreign,
    [],
    "the store must not carry a row the routing catalog cannot reach",
  );
});

test("the classification is shaped like the catalog, not like one corpus", () => {
  // SHAPE ONLY. Population counts ("anchors exist", "more than one model is anchored") were
  // removed under sc-22512/E8: they red when a measurement is retired or a corpus shrinks, which
  // is never a defect. What survives grades data that IS present — the lane spelling, and each
  // row's basis/reason/evidence agreement.
  assert.equal(store.schemaVersion, MEMORY_ANCHOR_SCHEMA_VERSION);
  const lanes = new Set([
    ...store.anchors.map((anchor) => anchor.backend),
    ...store.analyticOnly.map((entry) => entry.backend),
  ]);
  // SUBSET, not an exact roster: `deepEqual(["candle","mlx"])` is a population gate wearing a
  // shape gate's clothes — it reds when a lane's rows are all retired, which is absence. What is
  // actually enforceable at any population size is that no row spells a lane the loader cannot
  // route.
  for (const lane of lanes) {
    assert.ok(["candle", "mlx"].includes(lane), `unknown backend lane ${lane}`);
  }
  for (const entry of store.analyticOnly) {
    assert.ok(
      ANALYTIC_BASES.includes(entry.basis),
      `${entry.id}: unknown basis ${entry.basis}`,
    );
    assert.ok(
      entry.reason.trim().length > 0,
      `${entry.id}: a classification must state why`,
    );
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
    assert.ok(
      anchor.source.path.length > 0 && anchor.source.sha256.length === 64,
    );
  }
});

// sc-22512 / E8: absence is an INPUT to the classification, never a failure. A model no corpus, no
// manifest tier table and no provider constant says anything about must still come out classified,
// with a stated reason — and the extractor must not throw on the way there. Built entirely in-test
// (nothing is committed) so this keeps asking its question no matter what the real corpus holds.
test("a catalog model with zero evidence of any kind is classified, not refused", async () => {
  const model = {
    id: "sc22512_unmeasured_model",
    backends: ["mlx"],
    axes: { mlx: { tiers: ["q4"] } },
    family: "sc22512-unmeasured-family",
    familyGroup: null,
    resolvedRoute: null,
    resolvedRoutes: { mlx: "sc22512_unmeasured_route" },
    modality: "image",
  };
  const built = await buildAnchorStore({ matrix: { models: [model] } });

  assert.deepEqual(built.anchors, [], "no retained render covers this cell, so it anchors nothing");
  assert.equal(built.analyticOnly.length, 1, "the cell is classified exactly once");
  const [row] = built.analyticOnly;
  assert.equal(row.modelId, model.id);
  assert.equal(row.backend, "mlx");
  assert.equal(row.tier, "q4");
  assert.equal(row.basis, "no_retained_evidence", "total absence is an explicit basis, not a gap");
  assert.equal(row.evidence, null, "a no-evidence row cites nothing");
  assert.ok(row.reason.trim().length > 0, "the classification states why it is analytic-only");
});

test("the checked-in store is what the extractor produces", async () => {
  const committed = await readFile(path.join(ROOT, STORE_PATH), "utf8");
  assert.equal(
    serialiseStore(store),
    committed,
    `${STORE_PATH} is stale — re-run scripts/extract-memory-anchors.mjs`,
  );
});

test("an anchor's currency key is carried forward, never re-derived at the pin", async () => {
  const committed = JSON.parse(await readFile(path.join(ROOT, STORE_PATH), "utf8"));
  const closures = JSON.parse(
    await readFile(path.join(ROOT, ANCHOR_LOADER_CLOSURES_PATH), "utf8"),
  );
  assert.ok(store.anchors.length > 0);
  for (const anchor of store.anchors) {
    const recorded = committed.anchors.find((entry) => entry.id === anchor.id);
    assert.ok(recorded, `${anchor.id}: must already exist in the committed store`);
    assert.equal(anchor.source.loaderClosureDigest, recorded.source.loaderClosureDigest);
  }
  // THE POINT: the key records the loader AT MEASUREMENT, so it is NOT simply the pin's declared
  // digest. A store where the two always agreed would be one where currency compares a value with
  // itself and can never report a moved loader.
  const declaredAtPin = store.anchors.map(
    (anchor) => closures.models[`${anchor.modelId}:${anchor.backend}`]?.digest,
  );
  assert.ok(
    store.anchors.some((anchor, index) => anchor.source.loaderClosureDigest !== declaredAtPin[index]),
    "at least one packaged anchor must be measured against a loader the pin has since moved — " +
      "otherwise this generator is stamping the pin and currency means nothing",
  );
  // A new anchor with nothing to carry forward fails LOUDLY rather than borrowing the pin's digest.
  assert.throws(
    () => loaderClosureDigestFor(committed, "brand:new:anchor:id"),
    /has no recorded loader-closure digest/,
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
  // A cell the rebuild will NOT re-derive an anchor for. Preferably a real unanchored catalog cell;
  // if coverage is ever total, a coordinate outside the catalog serves the same purpose — either
  // way the hand-written row must not survive. Asserting that an unanchored cell EXISTS would be an
  // inverted absence gate: it reds the day coverage becomes complete (sc-22512/E8).
  const unanchored =
    cells.find(
      (cell) =>
        !store.anchors.some(
          (anchor) =>
            cellKey(anchor.modelId, anchor.backend, anchor.tier) ===
            cellKey(cell.modelId, cell.backend, cell.tier),
        ),
    ) ?? { modelId: "sc22512_no_such_model", backend: "mlx", tier: "q4" };
  // Likewise absence-tolerant: with an empty anchor set there is no row to copy the shape from, and
  // an empty template still exercises the question ("does a written row survive a rebuild?").
  const template = store.anchors[0] ?? { source: {} };
  const handWritten = (id, source) => ({
    ...template,
    id,
    modelId: unanchored.modelId,
    backend: unanchored.backend,
    tier: unanchored.tier,
    transformerVariant: null,
    decoder: null,
    source: { ...template.source, ...source },
  });
  const seeded = await buildAnchorStore({
    matrix,
    existingStore: {
      schemaVersion: 1,
      anchors: [
        // One citing a corpus this run walks, one citing evidence it never sees: neither is
        // re-derivable from the committed tree, so neither may reach the store.
        handWritten("hand:inserted", {}),
        handWritten("hand:outside-the-corpora", {
          path: "docs/proving/sc-22509-candle.json",
        }),
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
  assert.deepEqual(
    ids,
    [...ids].sort(),
    "anchors are emitted in a stable sorted order",
  );
  const analyticIds = store.analyticOnly.map((entry) => entry.id);
  assert.deepEqual(
    analyticIds,
    [...analyticIds].sort(),
    "analytic rows are emitted sorted",
  );
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
      [
        {
          id: "ok",
          source: { path: "docs/generated/memory-calibration-evidence.json" },
        },
      ],
      packaged,
    ),
  );
  assert.throws(
    () =>
      assertPackagedSources(
        [
          {
            id: "foreign:concurrent-story",
            source: { path: "docs/proving/sc-22509-candle.json" },
          },
        ],
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
    assert.equal(
      anchor.overlay,
      null,
      `${anchor.id}: an overlay render must not anchor a cell`,
    );
  }
  // SHAPE, not census: if the overlay-only evidence ever leaves the corpus there is nothing to
  // assert about, and failing here would red a measurement's retirement rather than a defect.
  const overlayEvidenced = store.analyticOnly.filter(
    (entry) => entry.evidence?.values?.overlay,
  );
  for (const entry of overlayEvidenced) {
    assert.equal(entry.basis, "measured_envelope");
    assert.match(
      entry.reason,
      /overlay/,
      `${entry.id}: the row must say why it is not anchored`,
    );
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
      records: [
        render("overlay", 900, "control:1"),
        render("clean", 100, "none"),
      ],
    },
  ];
  const chosen = envelopeEvidence(corpora, cell);
  assert.equal(
    chosen.recordId,
    "clean",
    "the smaller CLEAN envelope wins over a larger overlay",
  );
  assert.equal(
    chosen.values.overlay,
    undefined,
    "a clean row cites no overlay",
  );

  const overlayOnly = envelopeEvidence(
    [{ ...corpora[0], records: [render("overlay", 900, "control:1")] }],
    cell,
  );
  assert.equal(
    overlayOnly.recordId,
    "overlay",
    "overlay evidence is still cited when it is all",
  );
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
    selectRepresentative([make("b", 1, "a.json"), make("a", 2, "z.json")])
      .recordId,
    "a",
  );
  assert.equal(
    selectRepresentative([make("b", 2, "z.json"), make("a", 2, "a.json")])
      .recordId,
    "a",
  );
  assert.equal(
    selectRepresentative([make("b", 2, "a.json"), make("a", 2, "a.json")])
      .recordId,
    "a",
  );
  // Total: the same candidates in any order select the same record.
  const candidates = [
    make("b", 2, "a.json"),
    make("a", 2, "a.json"),
    make("c", 1, "a.json"),
  ];
  assert.equal(
    selectRepresentative([...candidates].reverse()).recordId,
    selectRepresentative(candidates).recordId,
  );
});

test("an MLX window opened above its resident set outranks a cold-request render of the same cell", () => {
  // sc-22667 (epic 22657 D3). The retained z_image_turbo q4 MLX corpus shape: the cold-request
  // record's overall envelope (19.14 GB) is within allocator jitter of a warm re-capture's, so
  // envelope-first would pick between them by chance — and the cold record's conditioning level
  // (2.27 GB) sits below the 5.83 GB resident set the core law subtracts, so it derives nothing.
  const mlx = (id, envelope, measurements) => ({
    backend: "mlx",
    overallAllocatorEnvelopeBytes: envelope,
    recordId: id,
    sourcePath: "docs/generated/memory-calibration-evidence.json",
    ...measurements,
  });
  const cold = mlx("cold", 19_138_127_416, { residentSetSeen: false });
  const warm = mlx("warm", 18_900_000_000, { residentSetSeen: true });
  assert.equal(selectRepresentative([cold, warm]).recordId, "warm");
  assert.equal(selectRepresentative([warm, cold]).recordId, "warm");
  // Between two records of the same kind the envelope still decides; a record predating the flag
  // (no field at all) ranks with the cold ones.
  const legacy = mlx("legacy", 19_200_000_000, {});
  assert.equal(selectRepresentative([legacy, cold]).recordId, "legacy");
  assert.equal(selectRepresentative([legacy, warm]).recordId, "warm");
  // …and the flag is read off the record's own diagnostics, only on the MLX lane.
  const record = (backend, value) => ({
    id: `r-${backend}`,
    backend,
    target: {
      modelId: "z_image_turbo",
      provider: "z_image_turbo",
      tier: "q4",
      geometry: { width: 768, height: 768, frames: 1 },
    },
    loadShape: "eager_materialization",
    strategy: { engagedRungs: ["resident"] },
    observedMemory: { overall: { allocatorBytes: 1 } },
    diagnostics: {
      measurements: [
        ...(backend === "mlx"
          ? [
              { name: "conditioningActivePeak", unit: "bytes", value: 1 },
              { name: "denoiseActivePeak", unit: "bytes", value: 1 },
              { name: "decodeActivePeak", unit: "bytes", value: 1 },
            ]
          : [
              { name: "conditioningDevicePeakDelta", unit: "bytes", value: 1 },
              { name: "denoiseDevicePeakDelta", unit: "bytes", value: 1 },
              { name: "decodeDevicePeakDelta", unit: "bytes", value: 1 },
            ]),
        ...(value === null
          ? []
          : [{ name: "residentSetMaterializedBeforeWindow", unit: "count", value }]),
      ],
    },
  });
  const corpus = { path: "docs/generated/example.json", sha256: "0".repeat(64) };
  assert.equal(anchorCandidate(record("mlx", 1), corpus).residentSetSeen, true);
  assert.equal(anchorCandidate(record("mlx", 0), corpus).residentSetSeen, false);
  assert.equal(anchorCandidate(record("mlx", null), corpus).residentSetSeen, false);
  assert.equal(anchorCandidate(record("candle", 1), corpus).residentSetSeen, false);
});

test("derivability outranks envelope, so a cell is never anchored by a render its law refuses", () => {
  const candle = (id, envelope, regime) => ({
    backend: "candle",
    transformerVariant: null,
    decoder: null,
    geometry: { width: 1024, height: 1024, frames: 1, fps: null },
    measuredRegime: {
      decodeTiled: false,
      transformerWindowed: false,
      staged: false,
      attentionChunked: false,
      ...regime,
    },
    overallAllocatorEnvelopeBytes: envelope,
    recordId: id,
    sourcePath: "docs/generated/example.json",
  });
  // The shape of the retained Krea corpus: the LARGEST envelope is the resident-only capture, which
  // is the one composition `derive_image_phase_peaks` refuses. Envelope-first would anchor the cell
  // from a row rejected on every lookup.
  const resident = candle("resident", 25_171_066_880, {});
  const staged = candle("staged", 22_352_494_592, { staged: true });
  assert.equal(
    isDerivable(resident),
    false,
    "a resident candle composition is not derivable",
  );
  assert.equal(isDerivable(staged), true, "the shallow staged composition is");
  assert.equal(selectRepresentative([resident, staged]).recordId, "staged");
  assert.equal(selectRepresentative([staged, resident]).recordId, "staged");

  // Every deeper rung is refused too: the anchor must be the SHALLOWEST optimized composition.
  for (const deeper of [
    "decodeTiled",
    "attentionChunked",
    "transformerWindowed",
  ]) {
    assert.equal(
      isDerivable(candle("deeper", 1, { staged: true, [deeper]: true })),
      false,
      `${deeper} is deeper than the law prices`,
    );
  }
  // The MLX video law rejects no composition outright — its regime guards are anchor-vs-request —
  // so this rule must not silently withdraw MLX rows.
  assert.equal(isDerivable({ backend: "mlx", measuredRegime: {} }), true);
});

test("a candle lane whose engine has no staged composition is anchored by its resident render", async () => {
  // sc-22734. `staged_residency` is a STRUCTURAL exemption for SenseNova on both lanes: one fused
  // dual-path checkpoint with no separable conditioning component, so no staged render exists to
  // anchor from and the resident one is the only composition the cell can ever be captured in.
  // Before this the extractor discarded those captures and the cells fell to analytic-only, which
  // would have spent a real GPU campaign producing renders nothing could use.
  const manifest = {
    models: [
      {
        id: "sensenova_u1_8b",
        candle: {
          memoryStrategyStructuralExemptions: {
            staged_residency: { overlays: ["none"], evidence: [] },
          },
        },
        mlx: {
          memoryStrategyStructuralExemptions: {
            staged_residency: { overlays: ["none"], evidence: [] },
          },
        },
      },
      { id: "qwen_image", candle: {} },
    ],
  };
  const lanes = stagedResidencyExemptLanes(manifest);
  assert.deepEqual(
    [...lanes].sort(),
    ["sensenova_u1_8b:candle", "sensenova_u1_8b:mlx"],
    "the exempt lanes are derived from the manifest, never hand-kept",
  );

  const candle = (modelId, regime) => ({
    modelId,
    backend: "candle",
    transformerVariant: null,
    decoder: null,
    geometry: { width: 1024, height: 1024, frames: 1, fps: null },
    measuredRegime: {
      decodeTiled: false,
      transformerWindowed: false,
      staged: false,
      attentionChunked: false,
      ...regime,
    },
    overallAllocatorEnvelopeBytes: 1,
    recordId: modelId,
    sourcePath: "docs/generated/example.json",
  });

  // The exempt cell: resident derives, and the staged shape it can never actually be measured in
  // is refused rather than silently preferred.
  assert.equal(isDerivable(candle("sensenova_u1_8b", {}), lanes), true);
  assert.equal(
    isDerivable(candle("sensenova_u1_8b", { staged: true }), lanes),
    false,
    "a staged anchor on a cell that declares staging impossible is not a record the law can price",
  );
  // Deeper rungs stay refused on the exempt lane too.
  for (const deeper of ["decodeTiled", "attentionChunked", "transformerWindowed"]) {
    assert.equal(
      isDerivable(candle("sensenova_u1_8b", { [deeper]: true }), lanes),
      false,
      `${deeper} is deeper than the law prices, exemption or not`,
    );
  }
  // And NOTHING moves for an ordinary cell — the exemption is per (model, lane), not a mode.
  assert.equal(isDerivable(candle("qwen_image", {}), lanes), false);
  assert.equal(isDerivable(candle("qwen_image", { staged: true }), lanes), true);
  // The default argument is what every existing caller and every packaged row still takes.
  assert.equal(isDerivable(candle("sensenova_u1_8b", {})), false);
  assert.equal(isDerivable(candle("sensenova_u1_8b", { staged: true })), true);

  // Representative selection follows the same rule, so an exempt cell's resident capture wins.
  assert.equal(
    selectRepresentative(
      [candle("sensenova_u1_8b", { staged: true }), candle("sensenova_u1_8b", {})],
      lanes,
    ).measuredRegime.staged,
    false,
  );

  // The real manifest declares the exemption for all six SenseNova ids on both lanes, and for
  // nothing that already has a packaged candle anchor.
  const realManifest = JSON.parse(
    stripJsoncComments(await readFile(path.join(ROOT, MANIFEST_PATH), "utf8")),
  );
  const realLanes = stagedResidencyExemptLanes(realManifest);
  for (const modelId of realManifest.models
    .filter((model) => model.id.startsWith("sensenova_u1_8b"))
    .map((model) => model.id)) {
    for (const backend of ["mlx", "candle"]) {
      assert.ok(
        realLanes.has(`${modelId}:${backend}`),
        `${modelId}:${backend} must carry the structural exemption the arm relies on`,
      );
    }
  }
  const store = await buildAnchorStore({ matrix });
  for (const anchor of store.anchors) {
    assert.ok(
      !realLanes.has(`${anchor.modelId}:${anchor.backend}`),
      `${anchor.id}: no packaged anchor is on an exempt lane yet, so no packaged row moved`,
    );
    assert.equal(
      anchor.stagedResidencyStructurallyNotApplicable,
      undefined,
      `${anchor.id}: the new field is emitted only when true, so packaged rows stay byte-identical`,
    );
  }
});

test("every emitted anchor cites a compiled-in corpus, and every retained corpus is compiled in", async () => {
  const store = await buildAnchorStore({ matrix });
  const packaged = packagedAnchorSources(
    await readFile(path.join(ROOT, PACKAGED_SOURCES_PATH), "utf8"),
  );
  for (const anchor of store.anchors) {
    assert.ok(
      packaged.has(anchor.source.path),
      `${anchor.id} cites an unpackaged corpus`,
    );
  }
  // THE CONVERSE (sc-22666, epic 22657 E5). Packaging used to be an opt-in a story could defer,
  // because the image lane priced cells with slopes fitted on Krea Turbo and anchoring another
  // model from a freshly committed corpus would have borrowed them. The law fits nothing since
  // sc-22663, so an unpackaged corpus that could anchor a catalog cell is now a defect: the
  // generator must fail rather than classify the cell analytic-only beside its own evidence.
  const corpora = await loadCorpora(ROOT);
  const catalogByCell = new Map(
    (await catalogCells(matrix)).map((cell) => [
      cellKey(cell.modelId, cell.backend, cell.tier),
      cell,
    ]),
  );
  assert.doesNotThrow(() =>
    assertEveryDerivableCorpusIsPackaged(corpora, packaged, catalogByCell),
  );
  // SHAPE, not a census: whichever corpora are retained, dropping any ONE of them from the
  // packaged list must be caught, and the failure must name the file and the cells it strands.
  const anchoredPaths = [
    ...new Set(store.anchors.map((anchor) => anchor.source.path)),
  ].sort();
  assert.ok(anchoredPaths.length > 0, "the store must cite at least one corpus");
  for (const dropped of anchoredPaths) {
    const narrowed = new Set([...packaged].filter((item) => item !== dropped));
    assert.throws(
      () =>
        assertEveryDerivableCorpusIsPackaged(corpora, narrowed, catalogByCell),
      (error) =>
        error.message.includes(dropped) &&
        /not compiled into PACKAGED_MEMORY_ANCHOR_SOURCES/.test(error.message),
      `dropping ${dropped} from the packaged list must fail the run`,
    );
  }
});

test("a newly packaged candle corpus anchors its own cells rather than bounding them", async () => {
  const store = await buildAnchorStore({ matrix });
  // sc-15859's three Z-Image-Turbo candle captures and sc-15817's qwen candle ladder were retained
  // but unpackaged before sc-22666. They are compiled in now, so they ANCHOR their cells, and
  // those cells must no longer appear on the analytic-only side (a cell is classified once).
  for (const [modelId, tiers] of [
    ["z_image_turbo", ["bf16", "q4", "q8"]],
    ["qwen_image", ["q4"]],
  ]) {
    for (const tier of tiers) {
      assert.ok(
        store.anchors.some(
          (anchor) =>
            anchor.modelId === modelId &&
            anchor.backend === "candle" &&
            anchor.tier === tier,
        ),
        `${modelId}:candle:${tier} must be anchored from its packaged corpus`,
      );
      assert.equal(
        store.analyticOnly.some(
          (row) =>
            row.modelId === modelId &&
            row.backend === "candle" &&
            row.tier === tier,
        ),
        false,
        `${modelId}:candle:${tier} is anchored, so it cannot also be analytic-only`,
      );
    }
  }
});

// ---------------------------------------------------------------------------------------------
// `contract_estimate` (sc-22666, epic 22657 E5): a cell whose backend block publishes a
// `memoryStrategyContract` is priced by the worker as a CONTRACT-ONLY per-rung ladder (sc-22664),
// not as one manifest scalar repeated, so classifying it `manifest_tier_declaration` would put its
// evidence in the wrong place.
// ---------------------------------------------------------------------------------------------

test("a published memoryStrategyContract outranks a bare manifest tier declaration", async () => {
  const store = await buildAnchorStore({ matrix });
  const manifest = JSON.parse(
    stripJsoncComments(await readFile(path.join(ROOT, MANIFEST_PATH), "utf8")),
  );
  const publishesContract = (row) =>
    Boolean(
      manifest.models?.find((model) => model.id === row.modelId)?.[row.backend]
        ?.memoryStrategyContract,
    );
  // SHAPE: whichever cells fall through to a manifest-only basis, none of them may be one whose
  // contract is published, whose lane implements the ladder AND which carries the ladder's
  // inputs (sc-22667: the staged row, on a non-receipt-priced route) — that is the precedence
  // claim, stated without pinning a count. A published contract on a lane with no ladder, on a
  // receipt-priced route, or without the row is NOT misplaced on a manifest row; the worker never
  // rescales those (see the lane-restriction and ladder-input tests).
  const misplaced = store.analyticOnly
    .filter((row) => row.basis === "manifest_tier_declaration")
    .filter(
      (row) =>
        publishesContract(row) &&
        CONTRACT_LADDER_BACKENDS.includes(row.backend) &&
        !isReceiptPricedRoute(row.route) &&
        manifestSequentialRow(manifest, row) !== null,
    )
    .map((row) => row.id);
  assert.deepEqual(
    misplaced,
    [],
    "a ladder-lane cell whose contract is published is a contract_estimate, never a bare manifest row",
  );
  for (const row of store.analyticOnly.filter(
    (item) => item.basis === "contract_estimate",
  )) {
    assert.ok(publishesContract(row), `${row.id} cites a contract that is not published`);
    assert.match(row.reason, /per-rung ladder/);
    assert.ok(
      row.evidence?.path?.endsWith("/memoryStrategyContract"),
      `${row.id} must cite the contract block it was classified from`,
    );
    assert.ok(
      (row.evidence?.values?.declaredRungs ?? "").length > 0,
      `${row.id} must carry the rungs the contract declares`,
    );
    // The reason says the ladder rescales the MANIFEST ROW, so where the manifest declares that
    // row the evidence must carry it: a row cannot assert a rescale of figures it drops.
    const declared = manifestTierEvidence(manifest, MANIFEST_PATH, "sha", {
      modelId: row.modelId,
      backend: row.backend,
      tier: row.tier,
    });
    if (declared !== null) {
      for (const [key, value] of Object.entries(declared.values)) {
        assert.equal(
          row.evidence?.values?.[key],
          value,
          `${row.id} must carry the manifest ${key} its reason says the ladder rescales`,
        );
      }
    }
  }
  assert.ok(
    ANALYTIC_BASES.indexOf("contract_estimate") <
      ANALYTIC_BASES.indexOf("manifest_tier_declaration"),
    "the basis order IS the precedence",
  );
});

test("every contract_estimate row is on a lane that implements the per-rung ladder", async () => {
  // The `contract_estimate` reason asserts a specific worker mechanism: the manifest row rescaled
  // by the image law's per-rung ratios. That is `floor_pseudo_anchor` in the CANDLE strategy; the
  // mlx fit gate has no such path. Read the worker sources so the claim is checked against the
  // code that would have to change, not against a literal repeated in two places.
  const laneSources = {
    candle: "crates/sceneworks-worker/src/candle_memory_strategy.rs",
    mlx: "crates/sceneworks-worker/src/mlx_fit_gate.rs",
  };
  const implementing = [];
  for (const [backend, relative] of Object.entries(laneSources)) {
    const source = await readFile(path.join(ROOT, relative), "utf8");
    if (/fn floor_pseudo_anchor\b/.test(source)) implementing.push(backend);
  }
  assert.deepEqual(
    [...CONTRACT_LADDER_BACKENDS].sort(),
    implementing.sort(),
    "CONTRACT_LADDER_BACKENDS must name exactly the lanes whose source implements the ladder",
  );
  const store = await buildAnchorStore({ matrix });
  // SHAPE: no row on a lane without the mechanism, whatever the counts are.
  const offLane = store.analyticOnly
    .filter((row) => row.basis === "contract_estimate")
    .filter((row) => !implementing.includes(row.backend))
    .map((row) => `${row.id} (${row.backend})`);
  assert.deepEqual(
    offLane,
    [],
    "a contract_estimate row asserts a ladder its lane does not implement",
  );
});

// ---------------------------------------------------------------------------------------------
// sc-22667 (epic 22657 feature-end round, AT-4/E5): `contract_estimate` asserts a worker
// mechanism — `floor_anchor` in `candle_memory_strategy.rs` — that runs only when the manifest
// declares the raw staged row AND the route is not receipt-priced. The basis is keyed on exactly
// those two inputs, and the receipt-priced list is read back from the worker source.
// ---------------------------------------------------------------------------------------------

test("RECEIPT_PRICED_ROUTES mirrors the worker's is_receipt_priced, read from its source", async () => {
  const source = await readFile(
    path.join(ROOT, "crates/sceneworks-worker/src/candle_memory_strategy.rs"),
    "utf8",
  );
  const body = source.match(
    /pub\(crate\) fn is_receipt_priced\(engine_id: &str\) -> bool \{([\s\S]*?)\n\}/,
  );
  assert.ok(body, "the worker still names its receipt-priced families in is_receipt_priced");
  const routes = new Set();
  for (const [, literal] of body[1].matchAll(/engine_id == "([^"]+)"/g)) {
    routes.add(literal);
  }
  const helpers = [...body[1].matchAll(/\b(is_[a-z0-9_]+)\(engine_id\)/g)].map(
    ([, helper]) => helper,
  );
  assert.ok(helpers.length > 0, "is_receipt_priced delegates to per-family helpers");
  for (const helper of helpers) {
    const helperBody = source.match(
      new RegExp(`fn ${helper}\\(engine_id: &str\\) -> bool \\{([\\s\\S]*?)\\n\\}`),
    );
    assert.ok(helperBody, `${helper} is defined on the worker`);
    const literals = helperBody[1].match(/matches!\(\s*engine_id,([\s\S]*?)\)\s*$/);
    assert.ok(literals, `${helper} is a matches! over engine-id literals`);
    for (const [, literal] of literals[1].matchAll(/"([^"]+)"/g)) routes.add(literal);
  }
  assert.deepEqual(
    [...RECEIPT_PRICED_ROUTES].sort(),
    [...routes].sort(),
    "the extractor's receipt-priced list must equal the worker's, in both directions",
  );
  assert.ok(routes.size >= 10, `the worker names ${routes.size} receipt-priced routes`);
});

test("contract_estimate is keyed on the ladder's inputs: a sequential row on a non-receipt-priced route", async () => {
  const store = await buildAnchorStore({ matrix });
  const manifest = JSON.parse(
    stripJsoncComments(await readFile(path.join(ROOT, MANIFEST_PATH), "utf8")),
  );
  const publishesContract = (row) =>
    Boolean(
      manifest.models?.find((model) => model.id === row.modelId)?.[row.backend]
        ?.memoryStrategyContract,
    );
  const candle = store.analyticOnly.filter((row) =>
    CONTRACT_LADDER_BACKENDS.includes(row.backend),
  );
  const estimates = candle.filter((row) => row.basis === "contract_estimate");
  assert.ok(estimates.length > 0, "the catalog has contract-only ladder cells");
  // SHAPE: every contract_estimate row carries the row it rescales and sits on a route the
  // worker would actually rescale for.
  for (const row of estimates) {
    const declared = manifestSequentialRow(manifest, row);
    assert.notEqual(declared, null, `${row.id} has no sequentialPeakGb row to rescale`);
    assert.equal(
      row.evidence?.values?.sequentialPeakGb,
      String(declared),
      `${row.id} must carry the staged row the reason says the ladder rescales`,
    );
    assert.equal(
      isReceiptPricedRoute(row.route),
      false,
      `${row.id} is receipt-priced; its floor is a sealed receipt, never a rescaled row`,
    );
    assert.match(row.reason, /sequentialPeakGb/);
    assert.match(row.reason, /not receipt-priced/);
  }
  // …and the converse: no cell that HAS both inputs (and a published contract) fell through.
  const fellThrough = candle
    .filter((row) => row.basis !== "contract_estimate")
    .filter(
      (row) =>
        publishesContract(row) &&
        !isReceiptPricedRoute(row.route) &&
        manifestSequentialRow(manifest, row) !== null &&
        row.basis !== "measured_envelope",
    )
    .map((row) => `${row.id} (${row.basis})`);
  assert.deepEqual(fellThrough, [], "a cell with the ladder's inputs is a contract_estimate");
  // The two exclusions each exist in the catalog, or the keying is vacuous: at least one
  // receipt-priced cell and at least one row-less cell with a published contract are classified
  // on a manifest basis.
  const excluded = candle.filter(
    (row) =>
      publishesContract(row) &&
      ["manifest_tier_declaration", "no_retained_evidence"].includes(row.basis),
  );
  assert.ok(
    excluded.some((row) => isReceiptPricedRoute(row.route)),
    "a receipt-priced route with a published contract stays on a manifest basis",
  );
  assert.ok(
    excluded.some((row) => manifestSequentialRow(manifest, row) === null),
    "a published contract with no sequentialPeakGb row stays on a manifest basis",
  );
});

test("manifestSequentialRow reads the tier's own row, with the worker's nvfp4 -> q8 fallback", () => {
  const manifest = {
    models: [
      {
        id: "m",
        candle: { sequentialPeakGb: { q4: 5.5, q8: 7.25, bf16: "9" } },
      },
    ],
  };
  const cell = (tier) => ({ modelId: "m", backend: "candle", tier });
  assert.equal(manifestSequentialRow(manifest, cell("q4")), 5.5);
  assert.equal(manifestSequentialRow(manifest, cell("nvfp4")), 7.25);
  assert.equal(manifestSequentialRow(manifest, cell("bf16")), null, "a string is not a row");
  assert.equal(manifestSequentialRow(manifest, cell("int8")), null);
  assert.equal(
    manifestSequentialRow(manifest, { modelId: "m", backend: "mlx", tier: "q4" }),
    null,
  );
});

test("an axis-free record keys on a spelling no stated axis can produce", () => {
  const axisFree = anchorCandidate(record(), corpus);
  const stated = anchorCandidate(
    record({
      target: {
        ...record().target,
        transformerVariant: "dev",
        decoder: "conv",
      },
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
        candle: {
          measured: true,
          vramGbByTier: { q4: 18.4 },
          sequentialPeakGb: { q4: 5.7 },
        },
        mlx: { measured: false, vramGbByTier: { q4: 18.4 } },
      },
    ],
  };
  const cell = { modelId: "example", backend: "candle", tier: "q4" };
  const evidence = manifestTierEvidence(
    manifest,
    "manifest.jsonc",
    "b".repeat(64),
    cell,
  );
  assert.deepEqual(evidence.values, {
    vramGbByTier: "18.4",
    sequentialPeakGb: "5.7",
  });
  assert.equal(
    manifestTierEvidence(manifest, "manifest.jsonc", "b".repeat(64), {
      ...cell,
      backend: "mlx",
    }),
    null,
    "an unmeasured declaration is not evidence",
  );
  assert.equal(
    manifestTierEvidence(manifest, "manifest.jsonc", "b".repeat(64), {
      ...cell,
      tier: "bf16",
    }),
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
    (await catalogCells({ models: [model] })).map((cell) => [
      cell.modelFamily,
      cell.route,
    ]),
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
  await writeProviderCrate(
    checkout,
    "mlx-gen-flux",
    "pub const FLUX_BYTES: u64 = 1;\n",
  );
  await writeProviderCrate(
    checkout,
    "mlx-gen-flux2",
    "pub const FLUX2_BYTES: u64 = 2_000;\n",
  );
  await writeProviderCrate(
    checkout,
    "mlx-gen-z",
    "pub const SHORT_PREFIX_BYTES: u64 = 4;\n",
  );
  await writeProviderCrate(
    checkout,
    "mlx-gen-z-image",
    "pub const Z_BYTES: u64 = 3;\n",
  );
  await writeProviderCrate(checkout, "mlx-gen-empty", "pub fn nothing() {}\n");
  const revision = "d".repeat(40);
  const constants = await inferenceProviderConstants(
    checkout,
    revision,
    new Set(["flux2_dev", "z_image_turbo", "empty", "unmatched_model"]),
  );

  assert.deepEqual([...constants.keys()].sort(), [
    "flux2_dev",
    "z_image_turbo",
  ]);
  const flux2 = constants.get("flux2_dev");
  assert.deepEqual(
    flux2.values,
    { FLUX2_BYTES: "2000" },
    "`flux2_dev` matches mlx-gen-flux2, not the shorter mlx-gen-flux prefix",
  );
  assert.equal(flux2.repo, "SceneWorks/inference");
  assert.equal(
    flux2.revision,
    revision,
    "a foreign citation names the revision it was read at",
  );
  assert.equal(
    flux2.path,
    "crates/media/mlx-gen/mlx-gen-flux2/src/memory_strategy.rs",
  );
  assert.match(flux2.sha256, /^[0-9a-f]{64}$/);
  assert.deepEqual(
    constants.get("z_image_turbo").values,
    { Z_BYTES: "3" },
    "`z_image_turbo` matches mlx-gen-z-image, though mlx-gen-z is also a legal prefix of it",
  );
  assert.equal(
    constants.get("empty"),
    undefined,
    "a crate with no constants contributes nothing",
  );

  assert.equal(
    (await inferenceProviderConstants(null, revision, new Set(["flux2_dev"])))
      .size,
    0,
    "with no checkout the limb reads nothing at all — the default for every generation",
  );
});

test("the pinned checkout is located by the pin's own short hash, or not at all", async () => {
  const home = await mkdtemp(path.join(os.tmpdir(), "anchor-home-"));
  const revision = "0123456789abcdef0123456789abcdef01234567";
  const checkouts = path.join(home, ".cargo/git/checkouts");
  await mkdir(
    path.join(checkouts, "inference-9f0e1d2c", revision.slice(0, 7)),
    {
      recursive: true,
    },
  );
  await mkdir(path.join(checkouts, "inference-9f0e1d2c", "badc0de"), {
    recursive: true,
  });
  await mkdir(path.join(checkouts, "mlx-rs-1a2b3c4d", revision.slice(0, 7)), {
    recursive: true,
  });

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
  assert.equal(
    await locateInferenceCheckout(revision, path.join(home, "absent")),
    null,
  );
});

test("the inference pin is read from the inference remote alone", () => {
  const cargo = [
    'candle-gen = { git = "https://github.com/SceneWorks/inference", rev = "' +
      "a".repeat(40) +
      '" }',
    'mlx-rs = { git = "https://github.com/michaeltrefry/mlx-rs", rev = "' +
      "b".repeat(40) +
      '" }',
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

// ── Epic 22505 feature-end fix round: allocator levels, underived reasons, component deltas ────

test("phase allocator levels are extracted exactly when the record reports all three", () => {
  const record = {
    observedMemory: {
      conditioning: { allocatorBytes: 1 },
      denoise: { allocatorBytes: 2 },
      decode: { allocatorBytes: 3 },
      overall: { allocatorBytes: 3 },
    },
  };
  assert.deepEqual(phaseAllocatorEnvelopes(record), { conditioning: 1, denoise: 2, decode: 3 });
  const partial = structuredClone(record);
  delete partial.observedMemory.denoise.allocatorBytes;
  assert.equal(phaseAllocatorEnvelopes(partial), null);
  const zeroed = structuredClone(record);
  zeroed.observedMemory.decode.allocatorBytes = 0;
  assert.equal(phaseAllocatorEnvelopes(zeroed), null);
});

test("an underived reason names the measured REGIME, never a missing geometry spread", () => {
  const candidate = (overrides) => ({
    modelId: "m",
    backend: "mlx",
    geometry: { frames: 1 },
    loadShape: "eager_materialization",
    measuredRegime: {
      staged: false,
      decodeTiled: false,
      attentionChunked: false,
      transformerWindowed: false,
    },
    transformerVariant: null,
    decoder: null,
    ...overrides,
  });
  // THE BORROWED-SLOPE REFUSAL IS GONE (sc-22666, epic 22657 E5). An MLX image anchor used to be
  // refused unless the model's OWN retained records varied geometry within a cell, because the
  // lane priced cells with per-pixel slopes fitted on some model's spread. The image law fits
  // nothing since sc-22663 — it decomposes THIS anchor's measured peaks against THIS contract's
  // component bytes — so a single-geometry anchor derives, and the reason no longer depends on the
  // corpus at all (the signature takes only the candidate).
  assert.equal(underivedReasonFor(candidate({})), null);
  assert.equal(
    underivedReasonFor.length,
    1,
    "the refusal's corpus inputs are gone with it",
  );
  // A deep measured regime still cannot upper-bound the ladder: that guard is about WHICH
  // composition was measured, not about fitting anything.
  assert.match(
    underivedReasonFor(
      candidate({
        measuredRegime: {
          staged: false,
          decodeTiled: true,
          attentionChunked: true,
          transformerWindowed: false,
        },
      }),
    ),
    /bounded rungs/,
  );
  assert.match(
    underivedReasonFor(candidate({ loadShape: "deferred_materialization" })),
    /bounded rungs/,
  );
  // An axis-free VIDEO anchor cannot answer the pipeline-keyed video law.
  assert.match(
    underivedReasonFor(candidate({ geometry: { frames: 145 } })),
    /pipeline axes/,
  );
  // A video anchor with stated axes takes no reason.
  assert.equal(
    underivedReasonFor(
      candidate({
        geometry: { frames: 145 },
        transformerVariant: "dev",
        decoder: "diffvae",
      }),
    ),
    null,
  );
  // A candle anchor takes no reason at all: `isDerivable` already refused the compositions the
  // candle law rejects, so every candle anchor that exists derives.
  assert.equal(underivedReasonFor(candidate({ backend: "candle" })), null);
});

test("the LTX-2.5 component deltas are priced from the committed weights inventory, per tier and axis", async () => {
  const inventoryBody = await readFile(
    path.join(ROOT, LTX25_WEIGHTS_INVENTORY_PATH),
    "utf8",
  );
  const inventory = JSON.parse(inventoryBody);
  const rows = ltx25ComponentDeltas(inventoryBody, ["q4", "q8", "bf16"]);
  // 3 tiers x (2 variant targets + 2 decoder targets).
  assert.equal(rows.length, 12);
  for (const row of rows) {
    assert.equal(
      row.bytes,
      row.files.reduce((total, file) => total + inventory.files[file], 0),
      `${row.id}: bytes must be the sum of the shipped files it names`,
    );
    // A row is either a priced crossing (files, positive bytes) or an explicit ZERO crossing (no
    // files, no bytes). Bytes without files is the doctoring shape and never emitted.
    assert.equal(row.bytes > 0, row.files.length > 0, `${row.id}: bytes and files must agree`);
    assert.equal(row.source.path, LTX25_WEIGHTS_INVENTORY_PATH);
  }
  // The variant deltas are keyed on what the ENGINE materializes, not on the variant's name: the
  // capture arm pushes the distillation LoRA under the Dev arm and the production worker resolves
  // no adapter for LTX-2.5 distilled, so DEV prices the LoRA and DISTILLED crosses nothing. The
  // stock enhancer is inserted for both variants, so it appears in neither row.
  const byId = new Map(rows.map((row) => [row.id, row]));
  assert.deepEqual(byId.get("ltx_2_5:mlx:q4:transformer_variant:dev").files, [
    "distilled_lora/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
  ]);
  assert.deepEqual(byId.get("ltx_2_5:mlx:q4:transformer_variant:distilled").files, []);
  assert.equal(byId.get("ltx_2_5:mlx:q4:transformer_variant:distilled").bytes, 0);
  for (const row of rows) {
    assert.ok(
      !row.files.some((file) => file.startsWith("enhancer/")),
      `${row.id}: the enhancer is resident in both variants and crosses nothing`,
    );
  }
  // The table the rows are keyed on is the mirror the Rust cross-check compares to the engine.
  assert.deepEqual(
    LTX25_VARIANT_COMPONENTS.map(({ to, components }) => [to, components]),
    [
      ["dev", ["distilled_lora"]],
      ["distilled", []],
    ],
  );
  // The decoder deltas take the WIDER variant's files, so one row upper-bounds both variants.
  const convQ4 = byId.get("ltx_2_5:mlx:q4:decoder:conv");
  assert.equal(convQ4.files.length, 1);
  assert.ok(convQ4.files[0].endsWith("vae_decoder.safetensors"));
  const diffvaeQ4 = byId.get("ltx_2_5:mlx:q4:decoder:diffvae");
  assert.equal(diffvaeQ4.files.length, 2);
  // And the packaged store carries exactly these rows — the extractor and the artifact agree.
  const packagedStore = JSON.parse(await readFile(path.join(ROOT, STORE_PATH), "utf8"));
  assert.deepEqual(packagedStore.componentDeltas, rows);
});
