import assert from "node:assert/strict";
import test from "node:test";

import {
  FAMILY_STORIES,
  MODEL_STORIES,
  assertCellOwnershipIsBackendScoped,
  assertTwinCoverage,
  backendScopes,
  buildStoryBackendScope,
  canonicalSourceText,
  familyStory,
  mlxRequiredHostBytes,
  modelStory,
} from "./generate-memory-matrix.mjs";

test("memory-strategy source hashing is independent of platform line endings", () => {
  const canonical = "alpha\nbeta\ngamma\n";
  assert.equal(canonicalSourceText(canonical), canonical);
  assert.equal(canonicalSourceText("alpha\r\nbeta\r\ngamma\r\n"), canonical);
  assert.equal(canonicalSourceText("alpha\rbeta\rgamma\r"), canonical);
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
  // MLX story. These five sit in families that DID get Candle twins but advertise mlx only, which is
  // exactly where a fallback would have looked plausible.
  for (const mlxOnly of [
    "z_image",
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

  // The applied split: 33 dual models, 14 dual families, 20 and 6 respectively left unsplit.
  const candleModels = Object.values(MODEL_STORIES).filter((stories) => stories.candle);
  const candleFamilies = Object.values(FAMILY_STORIES).filter((stories) => stories.candle);
  assert.equal(candleModels.length, 33);
  assert.equal(Object.keys(MODEL_STORIES).length - candleModels.length, 20);
  assert.equal(candleFamilies.length, 14);
  assert.equal(Object.keys(FAMILY_STORIES).length - candleFamilies.length, 6);
});

test("the ownership tables scope every story to exactly one backend", () => {
  // 53 MLX + 33 Candle model stories, 20 MLX + 14 Candle family stories, all distinct.
  const scope = buildStoryBackendScope();
  assert.equal(scope.size, 120);
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
  assert.deepEqual(assertTwinCoverage(models), { dualModels: 33, dualFamilies: 14 });

  // A newly dual model with no Candle twin must stop generation, not quietly reuse the MLX story.
  assert.throws(
    () =>
      assertTwinCoverage(
        models.map((model) => (model.id === "z_image" ? { ...model, backends: ["mlx", "candle"] } : model)),
      ),
    /z_image: advertises candle but has no Candle twin/,
  );
  // And an empty Candle twin on an mlx-only entry is equally a defect: it could never be closed.
  assert.throws(
    () => assertTwinCoverage(models, { ...MODEL_STORIES, z_image: { mlx: 15457, candle: 99999 } }),
    /z_image: advertises mlx only but carries Candle twin SC-99999/,
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
    /33 dual models map onto only 32 distinct Candle model twins/,
  );
});
