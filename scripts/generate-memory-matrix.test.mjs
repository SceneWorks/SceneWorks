import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  CONTROL_LANE_MODELS,
  FAMILY_STORIES,
  MODEL_STORIES,
  SOURCE_PATHS,
  assertCellOwnershipIsBackendScoped,
  assertCellInventoryMatchesCatalog,
  assertTwinCoverage,
  backendScopes,
  buildMatrix,
  buildStoryBackendScope,
  RUNG4_APPLICABILITIES,
  RUNG4_IMPLEMENTATIONS,
  RUNG4_REQUEST_PEAKS,
  familyGroup,
  familyStory,
  mlxRequiredHostBytes,
  modelStory,
} from "./generate-memory-matrix.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";
import { stripInertLines } from "./lib/source-revision.mjs";

// Line-ending and comment normalisation now lives in `scripts/lib/source-revision.mjs` and is unit
// tested there; these tests cover the same rules end to end, through the real generator.
test("a comment-only manifest edit produces no generated matrix change", async () => {
  const manifestUrl = new URL("../config/manifests/builtin.models.jsonc", import.meta.url);
  const manifest = await readFile(manifestUrl, "utf8");
  const baseline = await buildMatrix();
  const commentOnly = await buildMatrix({
    sourceOverrides: {
      manifest: `${manifest}\n// SC-16129 regression: provenance-only comment\n`,
    },
  });
  const withoutAnyComments = await buildMatrix({
    sourceOverrides: {
      // This also removes every comment block introduced by #1977, proving replacements and
      // deletions are inert rather than covering only an appended-comment special case.
      manifest: JSON.stringify(JSON.parse(stripJsoncComments(manifest))),
    },
  });

  assert.deepEqual(commentOnly, baseline);
  assert.deepEqual(withoutAnyComments, baseline);
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
    "instantId",
    "manifest",
    "memoryStrategy",
    "mlxFitGate",
    "rung4Survey",
    "vramGate",
  ]);
  assert.deepEqual(Object.keys(RUST_SOURCE_PATHS).sort(), [
    "cargo",
    "engines",
    "instantId",
    "memoryStrategy",
    "mlxFitGate",
    "vramGate",
  ]);

  const matrix = await buildMatrix();
  assert.deepEqual(
    Object.fromEntries(
      Object.entries(matrix.generatedFrom.sources).map(([name, entry]) => [name, entry.path]),
    ),
    { ...SOURCE_PATHS },
  );
});

test("a comment-only Rust or Cargo edit produces no generated matrix change", async () => {
  const sources = await readRustSources();
  const baseline = await buildMatrix();

  for (const [name, body] of Object.entries(sources)) {
    const marker = name === "cargo" ? "#" : "//";
    const appended = await buildMatrix({
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
  const baseline = await buildMatrix();
  const stripped = await buildMatrix({
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
  const baseline = await buildMatrix();
  const headroom = sources.mlxFitGate.match(/const HEADROOM_GB: f64 = ([0-9.]+);/);
  assert.ok(headroom, "fixture needs the HEADROOM_GB constant in mlx_fit_gate.rs");
  const mutated = await buildMatrix({
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
  const baseline = await buildMatrix();
  const quoted = await buildMatrix({
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
  const matrix = await buildMatrix();
  assert.equal(matrix.schemaVersion, 4);
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
  const matrix = await buildMatrix();
  const candleCells = matrix.cells.filter((cell) => cell.backend === "candle");
  assert.ok(candleCells.length > matrix.cells.length / 4, "mutation must drop a substantial fraction");
  await assert.rejects(
    buildMatrix({
      // Preserve the bespoke tier-override scope so its older, narrower guard does not mask the new
      // catalog-wide assertion. Every other Candle scope disappears from the real generator path.
      cellFilter: (cell) => cell.backend !== "candle" || cell.modelId === "instantid_realvisxl",
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
  const baseline = await buildMatrix();
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

test("Krea cumulative bounded-decode and bounded-attention curves preserve their measured compositions", async () => {
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
        `${tier}.${rung} remains only as historical curve provenance`,
      );
    }
  }

  const matrix = await buildMatrix();
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
        !cell.engagedRungs.includes("staged_residency") &&
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
    "all six catalog cells must remain unverified while retaining the cumulative measured sets",
  );
});

test("MLX generated evidence derives the same exact additive host requirement as runtime", () => {
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
  assert.equal(mlxRequiredHostBytes(record), 7 * 1024 ** 3);
  assert.equal(mlxRequiredHostBytes({ ...record, backend: "candle" }), null);
  assert.equal(
    mlxRequiredHostBytes({
      ...record,
      observedMemory: { overall: { wiredBytes: 9 * 1024 ** 3, reclaimableBytes: 0 } },
    }),
    11 * 1024 ** 3,
    "observed non-reclaimable wired peak wins when it exceeds prediction",
  );
});

// SC-15510: `z_image_edit` is a catalog id, not a provider. Both backends serve it from the
// `z_image_turbo` provider (MLX: `jobs_store::routing::mlx` maps `z_image_turbo | z_image_edit` to the
// same eligibility and engine; Candle: the `ZImageEdit` lane runs on Turbo weights), so its advertised
// backend scopes must be INHERITED from `z_image_turbo` rather than read off its own manifest entry —
// which carries no `mlx`/`candle` block of its own.
//
// Without the inheritance the entry silently advertises zero backends, and every one of its 150 matrix
// cells disappears instead of failing loudly. That is exactly the "route unavailable" state the epic
// distinguishes from "verified", so it is worth a test rather than a comment.
test("z_image_edit inherits its backend scopes from the z_image_turbo provider", () => {
  const manifestById = new Map([
    ["z_image_turbo", { id: "z_image_turbo", mlx: { quantize: 4 }, candle: { quantize: 4 } }],
    ["z_image_edit", { id: "z_image_edit" }],
  ]);
  const edit = manifestById.get("z_image_edit");
  assert.deepEqual(backendScopes(edit, manifestById), ["mlx", "candle"]);

  // The inheritance is specific, not a blanket fallback: an ordinary entry with no backend blocks
  // advertises nothing, and the alias tracks whatever Turbo actually advertises.
  assert.deepEqual(backendScopes({ id: "some_other_model" }, manifestById), []);
  const mlxOnly = new Map([
    ["z_image_turbo", { id: "z_image_turbo", mlx: { quantize: 4 } }],
    ["z_image_edit", { id: "z_image_edit" }],
  ]);
  assert.deepEqual(backendScopes(mlxOnly.get("z_image_edit"), mlxOnly), ["mlx"]);
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

  // An mlx-only entry has no candle key, so a candle lookup FAILS rather than silently returning the
  // MLX story. These four sit in families that DID get Candle twins but advertise mlx only, which is
  // exactly where a fallback would have looked plausible.
  for (const mlxOnly of [
    "lens",
    "boogu_image_edit",
    "flux2_klein_9b_kv",
    "flux2_klein_9b_true_v2",
  ]) {
    assert.equal(MODEL_STORIES[mlxOnly].candle, undefined);
    assert.throws(() => modelStory(mlxOnly, "candle"), /no candle owning model story/);
    // Their family twin exists, so the family lookup must NOT be what stops them — the model does.
    assert.ok(Number.isInteger(familyStory(mlxOnly, "candle")));
  }

  // A family owning no dual model has no candle twin, and asking for one throws.
  assert.equal(FAMILY_STORIES[15520].candle, undefined);
  assert.throws(() => familyStory("chroma1_hd", "candle"), /owns no candle story/);
  assert.throws(() => modelStory("not_a_model", "mlx"), /no owning model story/);

  // The applied split: 35 dual models, 15 dual families, 18 and 5 respectively left unsplit.
  const candleModels = Object.values(MODEL_STORIES).filter((stories) => stories.candle);
  const candleFamilies = Object.values(FAMILY_STORIES).filter((stories) => stories.candle);
  assert.equal(candleModels.length, 35);
  assert.equal(Object.keys(MODEL_STORIES).length - candleModels.length, 18);
  assert.equal(candleFamilies.length, 15);
  assert.equal(Object.keys(FAMILY_STORIES).length - candleFamilies.length, 5);
});

test("the ownership tables scope every story to exactly one backend", () => {
  // 53 MLX + 35 Candle model stories, 20 MLX + 15 Candle family stories, all distinct.
  const scope = buildStoryBackendScope();
  assert.equal(scope.size, 123);
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
  assert.deepEqual(assertTwinCoverage(models), { dualModels: 35, dualFamilies: 15 });

  // A newly dual model with no Candle twin must stop generation, not quietly reuse the MLX story.
  assert.throws(
    () =>
      assertTwinCoverage(
        models.map((model) => (model.id === "lens" ? { ...model, backends: ["mlx", "candle"] } : model)),
      ),
    /lens: advertises candle but has no Candle twin/,
  );
  // And an empty Candle twin on an mlx-only entry is equally a defect: it could never be closed.
  assert.throws(
    () => assertTwinCoverage(models, { ...MODEL_STORIES, lens: { mlx: 15462, candle: 99999 } }),
    /lens: advertises mlx only but carries Candle twin SC-99999/,
  );
  assert.throws(
    () => assertTwinCoverage(models, MODEL_STORIES, { ...FAMILY_STORIES, 15520: { mlx: 15520, candle: 99999 } }),
    /family SC-15520: owns no dual model but carries Candle twin SC-99999/,
  );
  assert.throws(
    () => assertTwinCoverage(models, MODEL_STORIES, { ...FAMILY_STORIES, 15516: { mlx: 15516 } }),
    /family SC-15516: owns dual models but has no Candle twin/,
  );
  // Two dual models sharing one Candle twin would under-count the split silently.
  assert.throws(
    () => assertTwinCoverage(models, { ...MODEL_STORIES, boogu_image: { mlx: 15474, candle: 15910 } }),
    /35 dual models map onto only 34 distinct Candle model twins/,
  );
});

test("a shipping control lane is declared, not inferred from having been measured (sc-16069)", async () => {
  const matrix = JSON.parse(
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
  assert.equal(
    kreaBounded?.evidence.currentEnvironmentVerification.length,
    0,
    "exact Krea control runs from the prior inference revision must not remain current after a pin bump",
  );
  assert.equal(
    kreaBounded?.evidence.historicalVerification.length,
    2,
    "both exact Krea control geometries remain attached as history until rerun on the new runtime",
  );

  // Every declared lane is represented on every backend the entry advertises — the declaration is what
  // generates cells now, so a lane can be unmeasured without being invisible.
  for (const id of CONTROL_LANE_MODELS) {
    const model = matrix.models.find((entry) => entry.id === id);
    assert.ok(model, `${id} must be a catalog entry`);
    for (const backend of model.backends) {
      assert.ok(
        control.some((cell) => cell.modelId === id && cell.backend === backend),
        `${id}/${backend} ships a control lane, so it must have control cells`,
      );
    }
  }

  // No control cells for anything undeclared: the overlay axis must not grow by accident.
  const undeclared = [...new Set(control.map((cell) => cell.modelId))].filter(
    (id) => !CONTROL_LANE_MODELS.includes(id),
  );
  assert.deepEqual(undeclared, [], "control cells exist only for declared lanes");

  // Declaring a lane must NOT fabricate evidence: these cells are honestly unverified.
  assert.equal(
    control.filter((cell) => cell.state === "Verified").length,
    0,
    "declaring a lane must not manufacture verification — no overlay cell has been measured",
  );
});

// ---------------------------------------------------------------------------
// SC-15969 — the rung-4 applicability survey.
// ---------------------------------------------------------------------------

const SURVEY_URL = new URL("../config/rung4-applicability-survey.json", import.meta.url);

async function surveyFixture() {
  return JSON.parse(await readFile(SURVEY_URL, "utf8"));
}

test("every advertised family/backend has a rung-4 verdict, and it reaches its cells", async () => {
  const matrix = await buildMatrix();
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
  // it must never be readable as evidence that doing so is worth selecting. The only two families
  // claiming `moves` are the two with measured request-peak evidence.
  const matrix = await buildMatrix();
  const moves = matrix.rung4SurveyRows.filter((row) => row.requestPeak === "moves");
  assert.deepEqual(
    moves.map((row) => `${row.familyStory}:${row.backend}`).sort(),
    ["15510:mlx", "15517:candle"],
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
  const matrix = await buildMatrix();

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
  assert.ok(
    sdxlCell.rung4Survey.blockStacks.some(
      (stack) => stack.windowable && /10 per Transformer2D/.test(stack.blocks),
    ),
    "the windowable sub-stack must be named, since that is what makes it partial rather than N/A",
  );
  assert.ok(
    sdxlCell.rung4Survey.blockStacks.some((stack) => !stack.windowable),
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
  const matrix = await buildMatrix();
  const implemented = matrix.cells.filter(
    (cell) =>
      cell.rung === "bounded_transformer_residency" &&
      cell.state === "Implemented/unverified",
  );
  assert.ok(implemented.length > 0);

  // MLX Z-Image ships rung 4 (SC-15754) and the matrix reported it Missing until this survey.
  const zImage = implemented.filter((cell) => cell.backend === "mlx");
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

  // Candle Krea's contract is gated on the turbo descriptor id, and its edit modes route to
  // descriptors that do not return it. A family- or entry-level claim alone would over-report both.
  const krea = implemented.filter((cell) => cell.backend === "candle");
  assert.ok(krea.every((cell) => cell.modelId === "krea_2_turbo"));
  assert.ok(krea.every((cell) => cell.mode === "text_to_image"));
  assert.equal(
    matrix.cells.filter(
      (cell) =>
        cell.modelId === "krea_2_raw" &&
        cell.rung === "bounded_transformer_residency" &&
        cell.state === "Implemented/unverified",
    ).length,
    0,
    "krea_2_raw does not get its sibling's contract",
  );
});

test("Candle Krea's Implemented cells report the shared backend that makes them reachable", async () => {
  const matrix = await buildMatrix();
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
  assert.match(report.summary, /candle_gen::block_window::run_windowed/);
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

test("rung 4 is not claimed where its declared rung-1 prerequisite is absent", async () => {
  // gen_core::memory_strategy makes rung 1 a prerequisite of rung 4, so a family with a perfectly
  // windowable trunk still reports Missing where the entry cannot stage its phases. Mage-Flow and
  // SenseNova are the epic's own uncovered-rung-1 families.
  const matrix = await buildMatrix();
  for (const modelId of ["mage_flow", "sensenova_u1_8b"]) {
    const staged = matrix.cells.filter(
      (cell) => cell.modelId === modelId && cell.rung === "staged_residency",
    );
    assert.ok(staged.length > 0);
    assert.ok(
      staged.every((cell) => cell.state === "Missing"),
      `${modelId}: fixture assumes rung 1 is unavailable`,
    );
    const rung4 = matrix.cells.filter(
      (cell) => cell.modelId === modelId && cell.rung === "bounded_transformer_residency",
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
  const matrix = await buildMatrix();
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
    buildMatrix({ sourceOverrides: { rung4Survey: JSON.stringify(survey) } }),
    /static provider evidence/,
  );

  // With evidence it is accepted, and it reaches the cell as Structurally N/A carrying that
  // evidence — the positive control, so the rejection above is not passing for an unrelated reason.
  survey.families["15525"].backends.mlx.structural = [
    { source: "inference:crates/media/mlx-gen/mlx-gen-sdxl/src/unet/mod.rs", reason: "fixture" },
  ];
  const matrix = await buildMatrix({
    sourceOverrides: { rung4Survey: JSON.stringify(survey) },
  });
  const cells = matrix.cells.filter(
    (cell) => cell.modelId === "sdxl" && cell.rung === "bounded_transformer_residency",
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
    buildMatrix({ sourceOverrides: { rung4Survey: JSON.stringify(missing) } }),
    /no rung-4 survey verdict/,
  );

  const contradictory = await surveyFixture();
  contradictory.families["15509"].backends.mlx.implementedEntries = ["mage_flow"];
  await assert.rejects(
    buildMatrix({ sourceOverrides: { rung4Survey: JSON.stringify(contradictory) } }),
    /the two must agree/,
  );

  const foreign = await surveyFixture();
  foreign.families["15510"].backends.mlx.implementedEntries.push("qwen_image");
  await assert.rejects(
    buildMatrix({ sourceOverrides: { rung4Survey: JSON.stringify(foreign) } }),
    /belongs to another family/,
  );

  const unknown = await surveyFixture();
  unknown.families["15509"].backends.mlx.structuralApplicability = "probably";
  await assert.rejects(
    buildMatrix({ sourceOverrides: { rung4Survey: JSON.stringify(unknown) } }),
    /unknown structuralApplicability/,
  );

  const unevidenced = await surveyFixture();
  unevidenced.families["15509"].backends.mlx.evidence = [];
  await assert.rejects(
    buildMatrix({ sourceOverrides: { rung4Survey: JSON.stringify(unevidenced) } }),
    /cite at least one source/,
  );

  // The positive control for all five: the shipped survey builds.
  await buildMatrix({ sourceOverrides: { rung4Survey: JSON.stringify(survey) } });
});

test("a survey edit rotates the source fingerprint", async () => {
  // The survey is now a generated-matrix input, so it has to be inside the staleness tripwire.
  // Without this, editing a verdict would leave `sceneWorksRevision` claiming the same provenance.
  const baseline = await buildMatrix();
  const survey = await surveyFixture();
  survey.families["15523"].backends.mlx.summary = `${survey.families["15523"].backends.mlx.summary}.`;
  const edited = await buildMatrix({
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
  // `blockStacks[].entries` is published onto every rung-4 cell of the family, so a typo'd or foreign
  // id becomes a per-entry "fact" about entries that are not in the family at all. `implementedEntries`
  // was checked from the start; this field was added later and inherited nothing.
  const foreign = await surveyFixture();
  foreign.families["15522"].backends.mlx.blockStacks[0].entries = ["qwen_image"];
  await assert.rejects(
    buildMatrix({ sourceOverrides: { rung4Survey: JSON.stringify(foreign) } }),
    /blockStacks.*names qwen_image, which belongs to another family/,
  );

  const invented = await surveyFixture();
  invented.families["15522"].backends.mlx.blockStacks[0].entries = ["totally_made_up_entry"];
  await assert.rejects(
    buildMatrix({ sourceOverrides: { rung4Survey: JSON.stringify(invented) } }),
    /names totally_made_up_entry/,
  );
});

test("every control-advertising family inventories its control-branch stack", async () => {
  // A control route holds a SECOND transformer resident alongside the trunk, which is exactly the
  // quantity a partial verdict exists to state. z-image was inventoried first and the rest were not,
  // so the inventory said different things about the same shape depending on the family.
  const matrix = await buildMatrix();
  const controlFamilies = new Set(
    matrix.cells
      .filter((cell) => cell.rung === "bounded_transformer_residency" && cell.overlay === "control")
      .map((cell) => `${familyGroup(cell.modelId)}:${cell.backend}`),
  );
  assert.ok(controlFamilies.size >= 10);
  for (const key of controlFamilies) {
    const cell = matrix.cells.find(
      (candidate) =>
        candidate.rung === "bounded_transformer_residency" &&
        candidate.overlay === "control" &&
        `${familyGroup(candidate.modelId)}:${candidate.backend}` === key,
    );
    const controlStacks = cell.rung4Survey.blockStacks.filter((stack) =>
      /control|ControlNet|IdentityNet/i.test(stack.name),
    );
    assert.ok(
      controlStacks.length > 0,
      `${key}: advertises a control overlay but its block-stack inventory names no control stack`,
    );
  }
});

test("a survey verdict that reaches no cell is rejected, not silently carried", async () => {
  // Coverage runs both ways. `rung4SurveyRows` is derived from the generated cells, so a verdict for
  // a family or backend the catalog does not advertise appears nowhere at all — it would sit in the
  // file being maintained, reviewed and trusted while having no effect.
  const survey = await surveyFixture();
  survey.families["15523"].backends.candle = {
    ...survey.families["15523"].backends.mlx,
    summary: "fixture: SANA advertises no candle entry",
  };
  await assert.rejects(
    buildMatrix({ sourceOverrides: { rung4Survey: JSON.stringify(survey) } }),
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
  bare.families["15511"].backends.mlx.findings = [];
  await assert.rejects(
    buildMatrix({ sourceOverrides: { rung4Survey: JSON.stringify(bare) } }),
    /must state the shape gap as a finding/,
  );

  const contradictory = await surveyFixture();
  contradictory.families["15510"].backends.mlx.structuralApplicability =
    "requires-different-primitive";
  await assert.rejects(
    buildMatrix({ sourceOverrides: { rung4Survey: JSON.stringify(contradictory) } }),
    /declaring the primitive's shape insufficient/,
  );

  // Stated as a finding, it builds — and the cell reports Missing with the verdict attached, NOT
  // Structurally N/A. That distinction is the whole point of the value.
  const stated = await surveyFixture();
  stated.families["15511"].backends.mlx.structuralApplicability = "requires-different-primitive";
  stated.families["15511"].backends.mlx.findings = ["fixture: the driver's shape cannot express it"];
  const matrix = await buildMatrix({
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
        cell.rung4Survey.structuralApplicability === "requires-different-primitive" &&
        cell.rung4Survey.findings.length > 0,
    ),
  );
});

test("`does-not-move` is carried through to the cell rather than collapsed into `unmeasured`", async () => {
  // The epic's non-negotiable has a positive form: a family MEASURED not to move the request peak is
  // a different fact from one nobody has measured, and the selector must be able to tell them apart.
  // No shipped family records it yet, so this pins the path rather than the data.
  const survey = await surveyFixture();
  survey.families["15511"].backends.mlx.requestPeak = {
    finding: "does-not-move",
    reason: "fixture",
  };
  const matrix = await buildMatrix({
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
