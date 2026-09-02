
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { stripJsoncComments } from "./lib/jsonc.mjs";
import { stripInertLines } from "./lib/source-revision.mjs";
import { routedLanes } from "./check-tier-integrity.mjs";
import {
  assertCellInventoryMatchesCatalog,
  assertCellOwnershipIsBackendScoped,
  assertMlxStagedCoverageIsStructurallyConsistent,
  assertOwnershipRegistriesAreDisjoint,
  assertOutOfMatrixEntriesAreStillUnroutable,
  assertPublishedDocumentIsClosed,
  assertTwinCoverage,
  assertUnroutedEntriesAreDeclared,
  assertVideoOwnership,
  backendScopes,
  buildMatrix,
  buildStoryBackendScope,
  cellState,
  declarationModelForCoordinate,
  familyGroup,
  familyStory,
  FAMILY_STORIES,
  implementationVerdict,
  IMPLEMENTED_STATES,
  indexAnchorStore,
  indexLoaderClosures,
  isImplemented,
  isPublishableCell,
  MODEL_STORIES,
  modelStory,
  OUT_OF_MATRIX_CATALOG_ENTRIES,
  parseAnchorDerivationLanes,
  parseCandleBespokeStagedLanes,
  parseInternalCandleVideoRoutes,
  parseVideoEngineIds,
  parseVideoRoutes,
  providerFor,
  renderMarkdown,
  SOURCE_PATHS,
  stagedResidencyIsAvailable,
  UNION_ONLY_MLX_ROUTES,
  UNROUTED_CATALOG_ENTRIES,
  VIDEO_FAMILY_STORIES,
} from "./generate-memory-matrix.mjs";

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

test("provenance is stamped once on the document, never per row", async () => {
  // sc-16268: the per-row copy was one constant repeated ~7,360 times, which turned every
  // fingerprint rotation into a ~14,700-line rewrite of a file that can only be regenerated.
  const matrix = await buildMatrix({ publish: false });
  assert.equal(matrix.schemaVersion, 11);
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

  // sc-20799 gave every remaining image entry a candle tuning block, so the absent-block premise no
  // longer has a live manifest instance. The claim under test is unchanged — the oracle keys on
  // ROUTING, not on tuning blocks — so exercise it with a block-less synthetic entry directly, and
  // keep the real-manifest read to prove the id exists in the catalog.
  assert.ok(byId.get("anima_base"), "anima_base must remain a catalog entry");
  assert.deepEqual(backendScopes({ id: "anima_base" }, lanes), ["mlx", "candle"]);
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

  // sc-22512 removed the pinned `candleModels.length === 53` / `candleFamilies.length === 20`. They
  // were frozen catalog populations and carried nothing beyond the numbers: a catalog that grows,
  // or one entry that ships mlx-only before its Candle lane exists, reddened them without anything
  // being wrong with the rows present. The claim worth keeping — an entry advertising `candle` must
  // have a real Candle twin, and an mlx-only entry must not carry one — belongs to
  // `assertTwinCoverage`, which reds on that CONTRADICTION at any population size and is exercised
  // by its own throw-mutation cases below.
});

test("the ownership tables scope every story to exactly one backend", () => {
  // One scope entry per (story, backend) across both tables, all distinct.
  //
  // sc-22512 replaced the pinned `146` with that relation, derived from the tables themselves: the
  // number was 2*(53+20) and reddened on any catalog growth, while the claim worth keeping is that
  // no story id is shared between two owners — which the throw at the bottom of this test pins.
  const scope = buildStoryBackendScope();
  const scopedStoryIds = [
    ...Object.values(MODEL_STORIES),
    ...Object.values(FAMILY_STORIES),
  ].flatMap((stories) => Object.values(stories));
  assert.equal(
    scope.size,
    scopedStoryIds.length,
    "every declared (story, backend) pair resolves to exactly one scope entry",
  );
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
  // sc-22512: the pinned `{dualModels: 53, dualFamilies: 20}` is replaced by the same figures
  // derived from the fixture the call was handed. The pinned pair reddened on a catalog that grew;
  // the claim worth keeping is that `assertTwinCoverage` REPORTS the dual population it was given,
  // which holds at any size including zero. The throw-mutation cases below are untouched — those
  // red on a declaration that contradicts itself, not on one that is missing.
  assert.deepEqual(assertTwinCoverage(models), {
    dualModels: models.filter((model) => model.backends.includes("candle")).length,
    dualFamilies: new Set(
      models
        .filter((model) => model.backends.includes("candle"))
        .map((model) => familyGroup(model.id)),
    ).size,
  });

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
    /^Error: (\d+) dual models map onto only (?!\1\b)\d+ distinct Candle model twins$/,
  );
});

const SURVEY_URL = new URL("../config/rung4-applicability-survey.json", import.meta.url);

async function surveyFixture() {
  return JSON.parse(await readFile(SURVEY_URL, "utf8"));
}

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
  // The exempt partition has to be occupied or the record set has collapsed into the old blanket
  // proxy (every family declaring the edge). The held-back partition, by contrast, may be honestly
  // EMPTY: its sole occupant used to be `flux2_klein_9b_true_v2:mlx`, whose staged column was
  // all-Missing only because the entry's converter-packed bf16 tier was invisible to the tier
  // universe and it surveyed at the `default` pseudo-tier (fixed in sc-21510). A declaring lane
  // whose staged column CAN stage is refused nothing, so an empty held-back partition just means
  // every declaring lane stages today; require instead that declaring lanes still exist at all, so
  // a records collapse to nobody-declares cannot pass silently, and keep the per-lane rung-4
  // assertion above live the moment any declaring lane stops staging.
  assert.ok(exempt > 0, "no lane exercises the exempt branch — the gate would be indistinguishable from the blanket proxy");
  assert.ok(
    lanes.some(({ key }) => declares.get(key)),
    "no lane belongs to a declaring family — the records no longer discriminate against the catalog",
  );
});

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
const QWEN_PRODUCTION_DEFERRED_REVISION = "014134e3035ad7e4eca5c2ed7bded2375dc3c071";
// ── sc-18099: publication ──────────────────────────────────────────────────────────────────────
//
// Everything above asserts GENERATION and reads `publish: false`, because most coordinates are
// elided and asserting them against the published subset would quietly hollow out thirteen
// behavioural tests. These assert the publication step itself, and are the only tests here that read
// the published document.

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

  const lightning = matrix.models.find((model) => model.id === "qwen_image_edit_2511_lightning");
  assert.deepEqual(
    lightning.axes.candle.overlays,
    ["identity", "lora"],
    "Qwen Lightning Candle has no plain route; built-in-LoRA and character overlays remain",
  );
  assert.equal(
    matrix.cells.some(
      (cell) =>
        cell.modelId === "qwen_image_edit_2511_lightning" &&
        cell.backend === "candle" &&
        cell.overlay === "none" &&
        ["resident", "staged_residency", "bounded_decode", "bounded_attention"].includes(
          cell.rung,
        ),
    ),
    false,
    "plain Lightning cells must not claim any Candle rung",
  );
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

function stagedCensusFixture() {
  const model = (id, { route = id, routeKind = "registry", tiers = ["q4"] } = {}) => ({
    id,
    resolvedRoute: route,
    routeKind,
    axes: { mlx: { tiers } },
  });
  const cell = (modelId, { overlay = "none", state = "Implemented" } = {}) => ({
    backend: "mlx",
    rung: "staged_residency",
    modelId,
    tier: "q4",
    mode: "text_to_image",
    overlay,
    state,
  });
  // Minimum shape the other assertions in the function demand: bernini AND flux2_dev in the census
  // (sc-20799 flipped flux2_dev's direction — at pin ebcdc7da7 all three Dev MLX providers declare
  // selectable staged residency), and the census neither empty nor the whole catalog.
  return {
    models: [model("bernini_image"), model("flux2_dev"), model("filler_a"), model("filler_b")],
    cells: [cell("bernini_image"), cell("flux2_dev")],
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
  // sc-22512 removed the three pinned populations here (11 manifest video entries, 53 image
  // entries, and the `{image: 53, video: 11}` summary breakdown). Every one of them reddened on a
  // catalog that SHIPPED AN EXTRA MODEL — the exact case that story has to let through. The claim
  // they were carrying is a partition, and it is stated below from the catalog's own enumeration:
  // the manifest video ids and the universe's video ids are the same set (both directions), and the
  // published breakdown is the recomputed truth about whatever the catalog contains.
  const matrix = await buildMatrix({ publish: false });
  const inMatrix = matrix.models.filter((model) => model.modality === "video").map((model) => model.id);
  assert.deepEqual(inMatrix.sort(), manifestVideo, "every manifest video entry is in the universe");
  assert.equal(matrix.summary.catalogEntries, matrix.models.length);
  assert.deepEqual(matrix.summary.catalogEntriesByModality, {
    image: matrix.models.filter((model) => model.modality === "image").length,
    video: matrix.models.filter((model) => model.modality === "video").length,
  });
  assert.equal(
    matrix.summary.catalogEntriesByModality.image + matrix.summary.catalogEntriesByModality.video,
    matrix.summary.catalogEntries,
    "image and video partition the catalog with nothing counted twice or lost",
  );

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
  // Each LTX generation has distinct MLX and Candle providers. A single scalar route would be wrong
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

  const ltx25 = matrix.models.find((model) => model.id === "ltx_2_5");
  assert.equal(ltx25.familyGroup, "ltx-video");
  assert.deepEqual(ltx25.resolvedRoutes, { mlx: "ltx_2_5", candle: "ltx_2_5_distilled" });
  const ltx25Providers = new Set(
    matrix.cells
      .filter((cell) => cell.modelId === "ltx_2_5")
      .map((cell) => `${cell.backend}:${cell.provider}`),
  );
  assert.deepEqual([...ltx25Providers].sort(), ["candle:ltx_2_5_distilled", "mlx:ltx_2_5"]);

  const closures = JSON.parse(
    await readFile(new URL("../config/inference-provider-closures.json", import.meta.url), "utf8"),
  );
  for (const provider of ["mlx:ltx_2_3", "mlx:ltx_2_5", "candle:ltx_2_5_distilled"]) {
    assert.ok(
      closures.providers[provider],
      `the provider ${provider} that a cell binds on must be named by the closure table`,
    );
  }
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
  // sc-22512 dropped the two "the debt register is empty" rosters (`unroutedEntries` and
  // `UNROUTED_CATALOG_ENTRIES` both deepEqual `[]`). They pinned a frozen population of exceptions:
  // adding a model that has no routing row yet reddened the suite instead of letting the entry ship
  // as unrouted-and-declared. The mechanism that matters — an unrouted entry must be DECLARED, and
  // an undeclared one still throws — is exercised by the fail-closed cases below and by
  // `assertUnroutedEntriesAreDeclared`. What is asserted here instead is the both-directions
  // agreement between the published register and the declared one, which holds at any size.
  assert.deepEqual(
    matrix.summary.unroutedEntries.map((entry) => entry.id).sort(),
    [...UNROUTED_CATALOG_ENTRIES]
      .filter(([id]) => matrix.models.some((model) => model.id === id))
      .map(([id]) => id)
      .sort(),
    "the published unrouted register is exactly the declared one, restricted to the universe",
  );
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
  // sc-22512: the pinned `11` was a frozen catalog population and reddened on an added entry. The
  // ownership rule below is universally quantified over the video lane, whatever its size.
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

test("receipt-backed bespoke Candle staged coverage is visible and census-fenced (sc-20799)", async () => {
  const matrix = await buildMatrix({ publish: false });
  const staged = matrix.cells.filter(
    (cell) =>
      cell.modelId === "kolors" && cell.backend === "candle" && cell.rung === "staged_residency",
  );
  const implemented = staged.filter((cell) => isImplemented(cell.state));
  // SC-20790 delivered request-authoritative admission for the Kolors ip/control bespoke lanes
  // with no manifest declaration (writing one would flip `declared_candle_request_strategy_contract`
  // relevance on the base lane). SHAPE, not a count: the delivered lanes must be visible, must be
  // exactly the conditioning overlays, and the undelivered plain lane must stay Missing.
  assert.ok(
    implemented.length > 0,
    "the delivered SC-20790 kolors bespoke lanes must not publish as Missing",
  );
  assert.ok(
    implemented.every((cell) => ["identity", "control"].includes(cell.overlay)),
    "bespoke request authority covers only the conditioning overlays",
  );
  assert.ok(
    implemented.some((cell) => cell.overlay === "identity") &&
      implemented.some((cell) => cell.overlay === "control"),
    "both censused providers must surface, not just one",
  );
  assert.ok(
    staged
      .filter((cell) => cell.overlay === "none")
      .every((cell) => cell.state === "Missing"),
    "plain kolors Candle staged stays Missing — no borrowed bespoke evidence",
  );

  // The census fence: a worker-censused provider without a generator lane map fails generation
  // closed instead of silently publishing a delivered lane as Missing.
  assert.throws(
    () =>
      parseCandleBespokeStagedLanes(
        'const CANDLE_BESPOKE_REQUEST_PROVIDERS: &[&str] = &["candle_new_provider"];',
      ),
    /unmapped=candle_new_provider/,
  );
  assert.throws(() => parseCandleBespokeStagedLanes("// nothing here"), /could not derive/);
});

test("the staged column tells structural truth apart from undelivered work (sc-20799)", async () => {
  const matrix = await buildMatrix();
  const markdown = renderMarkdown(matrix);
  const row = (id, backend) =>
    markdown
      .split("\n")
      .find((line) => line.startsWith(`| \`${id}\` |`) && line.includes(`| ${backend} |`));
  // SenseNova's staged rung is an architecture fact (fused dual-path transformer, no separable
  // conditioning component) on BOTH backends — the column must say so, not "Missing".
  for (const backend of ["mlx", "candle"]) {
    assert.match(
      row("sensenova_u1_8b", backend),
      /Structurally N\/A/,
      `sensenova_u1_8b ${backend} staged column must render the structural verdict`,
    );
  }
  // The delivered SC-20790 Kolors Candle bespoke lanes are receipt-backed coverage.
  assert.match(row("kolors", "candle"), /\| Implemented \|/);
  // And a genuinely undelivered lane still reads Missing — the column did not go soft.
  assert.match(row("svd", "candle"), /Missing/);
});

test("an Anchored rollup carries (stale) exactly when every anchor behind it is stale (sc-22513)", async () => {
  const matrix = await buildMatrix();
  const row = (markdown, id, backend) =>
    markdown
      .split("\n")
      .find((line) => line.startsWith(`| \`${id}\` |`) && line.includes(`| ${backend} |`));
  const anchored = matrix.models.flatMap((model) =>
    model.backends
      .filter((backend) =>
        matrix.anchors.some((anchor) => anchor.modelId === model.id && anchor.backend === backend),
      )
      .map((backend) => [model.id, backend]),
  );
  assert.ok(anchored.length > 0, "no anchored lane — the fixture would be vacuous");
  const anchoredRollups = anchored.filter(([id, backend]) =>
    /\| Anchored/.test(row(renderMarkdown(matrix), id, backend)),
  );
  assert.ok(anchoredRollups.length > 0, "no lane rolls up as Anchored — the marker is unreachable");

  const withCurrency = (current) => ({
    ...matrix,
    anchors: matrix.anchors.map((anchor) => ({ ...anchor, current })),
  });
  // Every anchor stale -> every Anchored rollup marked. Every anchor current -> none marked. The
  // marker is therefore driven by currency and by nothing else about the lane.
  const allStale = renderMarkdown(withCurrency(false));
  const allCurrent = renderMarkdown(withCurrency(true));
  for (const [id, backend] of anchoredRollups) {
    assert.match(row(allStale, id, backend), /Anchored[^|]*\(stale\)/, `${id}:${backend} stale`);
    assert.doesNotMatch(row(allCurrent, id, backend), /\(stale\)/, `${id}:${backend} current`);
  }
  // ONE current anchor on a lane is enough to clear its marker — the claim is "every anchor behind
  // it", not "any".
  const [id, backend] = anchoredRollups[0];
  const firstOfLane = matrix.anchors.find(
    (anchor) => anchor.modelId === id && anchor.backend === backend,
  );
  const mixed = renderMarkdown({
    ...matrix,
    anchors: matrix.anchors.map((anchor) => ({
      ...anchor,
      current: anchor.id === firstOfLane.id,
    })),
  });
  assert.doesNotMatch(row(mixed, id, backend), /\(stale\)/);
  // A lane the marker must never reach: no anchor behind it at all.
  const unanchored = matrix.models
    .flatMap((model) => model.backends.map((backend) => [model.id, backend]))
    .find(
      ([candidateId, candidateBackend]) =>
        !matrix.anchors.some(
          (anchor) => anchor.modelId === candidateId && anchor.backend === candidateBackend,
        ),
    );
  assert.doesNotMatch(row(allStale, unanchored[0], unanchored[1]), /\(stale\)/);
  // The shipped artifact agrees with the renderer.
  const shipped = await readFile(new URL("../docs/generated/memory-matrix.md", import.meta.url), "utf8");
  assert.equal(shipped, renderMarkdown(matrix));
});

// ── sc-22513: the collapse ─────────────────────────────────────────────────────────────────────

test("the cell state is a pure function of (implementation, anchor, derivation) and nothing else", () => {
  // The whole domain, exhaustively. Three implementation verdicts x anchored x derivation-defined
  // x anchor-derivable (epic 22505 feature-end fix round, E5), stated as a table here so a change
  // to the function has to be a change to this table too.
  const table = [
    [{ implementation: "missing", anchorPresent: false, derivationDefined: false, anchorDerivable: false }, "Missing"],
    [{ implementation: "missing", anchorPresent: true, derivationDefined: true, anchorDerivable: true }, "Missing"],
    [{ implementation: "structurally-na", anchorPresent: false, derivationDefined: false, anchorDerivable: false }, "Structurally N/A"],
    [{ implementation: "structurally-na", anchorPresent: true, derivationDefined: true, anchorDerivable: true }, "Structurally N/A"],
    [{ implementation: "implemented", anchorPresent: false, derivationDefined: false, anchorDerivable: false }, "Implemented"],
    [{ implementation: "implemented", anchorPresent: false, derivationDefined: true, anchorDerivable: false }, "Implemented"],
    [{ implementation: "implemented", anchorPresent: true, derivationDefined: false, anchorDerivable: true }, "Anchored/underived"],
    // Anchor-level derivability: a wired lane whose LAW refuses this anchor (an axis-free video
    // anchor, a single-geometry image model) publishes the honest Anchored/underived.
    [{ implementation: "implemented", anchorPresent: true, derivationDefined: true, anchorDerivable: false }, "Anchored/underived"],
    [{ implementation: "implemented", anchorPresent: true, derivationDefined: true, anchorDerivable: true }, "Anchored"],
  ];
  for (const [facts, expected] of table) {
    assert.equal(cellState(facts), expected, JSON.stringify(facts));
  }
  // An unknown verdict is a defect, not a silent `Missing`: a fourth implementation value would
  // otherwise inherit whichever arm happened to be last.
  assert.throws(
    () => cellState({ implementation: "unverified", anchorPresent: false, derivationDefined: false, anchorDerivable: false }),
    /unknown implementation verdict/,
  );
  // The retired vocabulary is unreachable, and the implemented set is exactly the three states that
  // assert code.
  const produced = new Set(table.map(([, state]) => state));
  for (const retired of ["Verified", "Runtime verified", "Implemented/unverified"]) {
    assert.ok(!produced.has(retired), retired);
  }
  assert.deepEqual([...IMPLEMENTED_STATES].sort(), ["Anchored", "Anchored/underived", "Implemented"]);
  for (const state of IMPLEMENTED_STATES) assert.ok(isImplemented(state), state);
  for (const state of ["Missing", "Structurally N/A"]) assert.ok(!isImplemented(state), state);
});

test("every generated cell's state re-derives from its own three published facts (sc-22513)", async () => {
  const matrix = await buildMatrix({ publish: false });
  const byFacts = new Map();
  for (const cell of matrix.cells) {
    const facts = {
      implementation: cell.implementation,
      anchorPresent: cell.anchor !== null,
      derivationDefined: cell.derivationDefined,
      anchorDerivable: cell.anchor !== null && cell.anchor.derivable,
    };
    assert.equal(cell.state, cellState(facts), cell.id);
    // Nothing that used to carry per-geometry evidence bookkeeping may survive on a cell.
    for (const retired of [
      "memoryCharacterization",
      "calibrationFingerprint",
      "engagedRungs",
      "plannedPipelineIdentities",
      "pipelineCharacterizations",
      "rung4Survey",
      "maxPixels",
    ]) {
      assert.ok(!(retired in cell), `${cell.id} still carries ${retired}`);
    }
    assert.deepEqual(Object.keys(cell.evidence).sort(), ["anchor", "staticImplementation", "structural"]);
    const key = JSON.stringify(facts);
    byFacts.set(key, (byFacts.get(key) ?? new Set()).add(cell.state));
  }
  // The half a re-derivation cannot fake: the mapping is SINGLE-VALUED. A state that acquired a
  // fourth input — a record, a plan row, a geometry, a currency digest — would split at least one
  // fact triple into two states, and this is what would catch it.
  for (const [key, states] of byFacts) {
    assert.equal(states.size, 1, `${key} produced ${[...states].join(" and ")}`);
  }
  // ...and the corners are actually populated, or the assertions above are vacuous.
  const produced = new Set(matrix.cells.map((cell) => cell.state));
  assert.deepEqual(
    [...produced].sort(),
    ["Anchored", "Anchored/underived", "Implemented", "Missing", "Structurally N/A"],
  );

  // THE NEGATIVE CASE. Everything above is a within-one-build check, and a fourth input that
  // happened to be CONSTANT across this build — a campaign flag, a plan row, a hardcoded per-lane
  // adjustment — would keep the mapping single-valued and slip straight through it. So move one of
  // the three facts and require the cells that ACTUALLY moved state to be exactly the cells
  // `cellState` PREDICTS should move from their own before/after triples. A cell that moved without
  // its triple predicting it is reading a fourth input; a cell whose triple predicts a move and
  // stayed put is not a function of its triple at all. (The predicted set is a strict subset of the
  // cells whose triple moved: dropping an anchor changes `anchorPresent` on `Missing` cells too, and
  // `cellState` is deliberately insensitive to the anchor there.)
  const store = JSON.parse(
    await readFile(new URL(`../${SOURCE_PATHS.anchorStore}`, import.meta.url), "utf8"),
  );
  const target = matrix.cells.find((cell) => cell.anchor !== null);
  assert.ok(target, "no anchored cell to flip — the negative case would be vacuous");
  const flipped = {
    ...store,
    anchors: store.anchors.filter(
      (anchor) =>
        !(
          anchor.modelId === target.modelId &&
          anchor.backend === target.backend &&
          anchor.tier === target.tier
        ),
    ),
  };
  const after = await buildMatrix({
    publish: false,
    sourceOverrides: { anchorStore: JSON.stringify(flipped) },
  });
  const tripleOf = (cell) => ({
    implementation: cell.implementation,
    anchorPresent: cell.anchor !== null,
    derivationDefined: cell.derivationDefined,
    anchorDerivable: cell.anchor !== null && cell.anchor.derivable,
  });
  const before = new Map(matrix.cells.map((cell) => [cell.id, cell]));
  assert.equal(after.cells.length, matrix.cells.length);
  const movedState = after.cells
    .filter((cell) => cell.state !== before.get(cell.id).state)
    .map((cell) => cell.id)
    .sort();
  const movedTriple = after.cells.filter(
    (cell) => JSON.stringify(tripleOf(cell)) !== JSON.stringify(tripleOf(before.get(cell.id))),
  );
  const predicted = movedTriple
    .filter((cell) => cellState(tripleOf(cell)) !== cellState(tripleOf(before.get(cell.id))))
    .map((cell) => cell.id)
    .sort();
  assert.ok(movedTriple.length > 0, "the flip must actually move a fact triple");
  assert.ok(predicted.length > 0, "the flip must actually predict a state move");
  assert.deepEqual(movedState, predicted);
  // And the mapping stays single-valued across BOTH builds together, so the flip did not introduce a
  // second state for a triple the baseline already produced.
  for (const cell of after.cells) {
    const key = JSON.stringify(tripleOf(cell));
    byFacts.set(key, (byFacts.get(key) ?? new Set()).add(cell.state));
  }
  for (const [key, states] of byFacts) {
    assert.equal(states.size, 1, `${key} produced ${[...states].join(" and ")}`);
  }
});

test("anchor CURRENCY is reported and cannot move a state (sc-22511, sc-22513)", async () => {
  const [store, closures] = await Promise.all(
    [SOURCE_PATHS.anchorStore, SOURCE_PATHS.anchorLoaderClosures].map(async (source) =>
      JSON.parse(await readFile(new URL(`../${source}`, import.meta.url), "utf8")),
    ),
  );
  // The checked-in receipts may all be historical at a newer reviewed pin. Make one closure current
  // synthetically so this invariant continues to exercise an actual currency transition without
  // treating pin-only currency drift as a request to recapture a measurement.
  const target = store.anchors[0];
  const targetKey = `${target.modelId}:${target.backend}`;
  assert.ok(closures.models[targetKey], `missing closure entry for ${targetKey}`);
  const currentized = {
    ...closures,
    models: {
      ...closures.models,
      [targetKey]: { ...closures.models[targetKey], digest: target.source.loaderClosureDigest },
    },
  };
  const baseline = await buildMatrix({
    publish: false,
    sourceOverrides: { anchorLoaderClosures: JSON.stringify(currentized) },
  });
  // Stale EVERY anchor's loader closure at once. Currency is a report, so the state of every cell
  // must be byte-identical; only the reported `current` flags may move.
  const staled = {
    ...currentized,
    models: Object.fromEntries(
      Object.entries(currentized.models).map(([key, entry]) => [key, { ...entry, digest: "0".repeat(64) }]),
    ),
  };
  const mutated = await buildMatrix({
    publish: false,
    sourceOverrides: { anchorLoaderClosures: JSON.stringify(staled) },
  });
  assert.deepEqual(
    mutated.cells.map((cell) => [cell.id, cell.state]),
    baseline.cells.map((cell) => [cell.id, cell.state]),
  );
  assert.ok(baseline.cells.some((cell) => cell.anchor?.current === true));
  assert.ok(mutated.cells.filter((cell) => cell.anchor).every((cell) => cell.anchor.current === false));
  assert.equal(mutated.summary.staleAnchors, mutated.anchors.length);
});

test("removing an anchor demotes exactly its own coordinates, and nothing else (sc-22513)", async () => {
  const baseline = await buildMatrix({ publish: false });
  const store = JSON.parse(
    await readFile(new URL(`../${SOURCE_PATHS.anchorStore}`, import.meta.url), "utf8"),
  );
  const target = baseline.cells.find((cell) => cell.state === "Anchored");
  assert.ok(target, "no Anchored cell to demote — the fixture would be vacuous");
  const withoutAnchor = {
    ...store,
    anchors: store.anchors.filter(
      (anchor) =>
        !(
          anchor.modelId === target.modelId &&
          anchor.backend === target.backend &&
          anchor.tier === target.tier
        ),
    ),
  };
  const mutated = await buildMatrix({
    publish: false,
    sourceOverrides: { anchorStore: JSON.stringify(withoutAnchor) },
  });
  const before = new Map(baseline.cells.map((cell) => [cell.id, cell]));
  const moved = mutated.cells.filter((cell) => cell.state !== before.get(cell.id).state);
  assert.ok(moved.length > 0);
  for (const cell of moved) {
    assert.equal(cell.modelId, target.modelId);
    assert.equal(cell.backend, target.backend);
    assert.equal(cell.tier, target.tier);
    assert.equal(cell.anchor, null);
    // The implementation verdict is untouched by the store: only the memory axis moved.
    assert.equal(cell.implementation, before.get(cell.id).implementation);
    assert.equal(cell.state, cell.implementation === "implemented" ? "Implemented" : before.get(cell.id).state);
  }
});

test("the derivation is defined per LANE, read off the Rust that declares and wires it", () => {
  const declares = "pub fn derive_video_phase_peaks(&self, request: AnchorDeriveRequest) {}";
  const wires = `
    let backend = match request.lane {
      VideoLane::Mlx => sceneworks_core::memory_anchor::AnchorBackend::Mlx,
      VideoLane::Candle => sceneworks_core::memory_anchor::AnchorBackend::Candle,
    };
    anchor.derive_video_phase_peaks(request)
  `;
  const laneMap = (source) => ({
    "video:mlx": { law: "video", sources: [source] },
    "video:candle": { law: "video", sources: [source] },
  });
  assert.deepEqual([...parseAnchorDerivationLanes(declares, laneMap(wires))].sort(), [
    "video:candle",
    "video:mlx",
  ]);
  // Declared but never called is NOT defined for any lane: an anchor priced by nothing prices
  // nothing, and a cell claiming `Anchored` off an unwired derivation would be the false green.
  assert.deepEqual(
    [
      ...parseAnchorDerivationLanes(
        declares,
        laneMap(wires.replace(/derive_video_phase_peaks\(request\)/, "floor()")),
      ),
    ],
    [],
  );
  // Unwiring one lane collapses that lane only.
  assert.deepEqual(
    [
      ...parseAnchorDerivationLanes(
        declares,
        laneMap(
          wires.replace(
            "VideoLane::Candle => sceneworks_core::memory_anchor::AnchorBackend::Candle,",
            "",
          ),
        ),
      ),
    ],
    ["video:mlx"],
  );
  // And a law nothing declares defines nothing, so the fixtures above are not a parallel universe.
  assert.ok(!parseAnchorDerivationLanes("", laneMap(wires)).size);
});

test("the shipped derivation reaches every lane through its REAL admission source (epic 22505)", async () => {
  const [derivation, admission, vram, candleStrategy, mlxFitGate] = await Promise.all([
    readFile(new URL(`../${SOURCE_PATHS.anchorDerivation}`, import.meta.url), "utf8"),
    readFile(new URL(`../${SOURCE_PATHS.anchorAdmission}`, import.meta.url), "utf8"),
    readFile(new URL(`../${SOURCE_PATHS.anchorAdmissionImageVram}`, import.meta.url), "utf8"),
    readFile(new URL(`../${SOURCE_PATHS.anchorAdmissionImageCandle}`, import.meta.url), "utf8"),
    readFile(new URL(`../${SOURCE_PATHS.mlxFitGate}`, import.meta.url), "utf8"),
  ]);
  const lanes = parseAnchorDerivationLanes(derivation, {
    "video:mlx": { law: "video", sources: [admission] },
    "video:candle": { law: "video", sources: [admission] },
    "image:candle": { law: "image", sources: [vram, candleStrategy] },
    "image:mlx": { law: "mlx_image", sources: [mlxFitGate] },
  });
  assert.deepEqual(
    [...lanes].sort(),
    ["image:candle", "image:mlx", "video:candle", "video:mlx"],
    "each lane's real admission source wires it (E5): video via video_admission.rs, image via " +
      "vram_gate.rs / candle_memory_strategy.rs / mlx_fit_gate.rs",
  );
  const matrix = await buildMatrix({ publish: false });
  for (const cell of matrix.cells) {
    const modality = matrix.models.find((model) => model.id === cell.modelId).modality;
    assert.equal(cell.derivationDefined, lanes.has(`${modality}:${cell.backend}`), cell.id);
  }
  // Anchored image cells now split on ANCHOR-LEVEL derivability: an anchor the lane's law accepts
  // publishes `Anchored`; one the store marks underived (single measured geometry, axis-free
  // video record) publishes `Anchored/underived` with its reason.
  const anchoredImplemented = matrix.cells.filter(
    (cell) => cell.anchor && cell.implementation === "implemented" && cell.derivationDefined,
  );
  assert.ok(anchoredImplemented.length > 0);
  for (const cell of anchoredImplemented) {
    assert.equal(cell.state, cell.anchor.derivable ? "Anchored" : "Anchored/underived", cell.id);
    if (!cell.anchor.derivable) {
      assert.ok(cell.anchor.underivedReason, `${cell.id} must state why it is underived`);
    }
  }
  // The headline cells of the feature-end fix round, pinned by name: the candle image anchor
  // prices its lane (item 3's acceptance), the flux2 image-MLX anchors price theirs (item 1), and
  // the refused anchors publish honestly (item 4).
  const stateOf = (modelId, backend, tier) =>
    new Set(
      matrix.cells
        .filter(
          (cell) =>
            cell.modelId === modelId &&
            cell.backend === backend &&
            cell.tier === tier &&
            cell.implementation === "implemented",
        )
        .map((cell) => cell.state),
    );
  assert.deepEqual([...stateOf("krea_2_turbo", "candle", "q4")], ["Anchored"]);
  assert.deepEqual([...stateOf("flux2_dev", "mlx", "q4")], ["Anchored"]);
  assert.deepEqual([...stateOf("qwen_image", "mlx", "q8")], ["Anchored/underived"]);
  assert.deepEqual([...stateOf("ltx_2_3", "mlx", "q8")], ["Anchored/underived"]);
});

test("the anchor store is indexed by (model, backend lane, tier), and analytic-only is not coverage", () => {
  const indexed = indexAnchorStore(
    JSON.stringify({
      anchors: [
        { id: "b", modelId: "m", backend: "mlx", tier: "q4" },
        { id: "a", modelId: "m", backend: "mlx", tier: "q4" },
        { id: "c", modelId: "m", backend: "candle", tier: "q4" },
      ],
      analyticOnly: [{ id: "analytic:m:mlx:q8" }],
    }),
  );
  assert.equal(indexed.total, 3);
  assert.equal(indexed.analyticOnly, 1);
  // Two anchors on one identity cell: the first by id is cited, deterministically, and they are
  // never merged into a coordinate no render measured.
  assert.equal(indexed.anchors.get("m:mlx:q4").id, "a");
  assert.equal(indexed.anchors.get("m:candle:q4").id, "c");
  // An analytic-only row is the store SAYING there is no anchor here. It is not a third state.
  assert.equal(indexed.anchors.get("m:mlx:q8"), undefined);
  assert.equal(indexed.anchors.size, 2);
});

test("the loader-closure digests are indexed by the anchor's own currency key (sc-22511)", () => {
  const closures = indexLoaderClosures(
    JSON.stringify({ models: { "m:mlx": { digest: "abc" }, "m:candle": {} } }),
  );
  assert.equal(closures.get("m:mlx"), "abc");
  assert.equal(closures.get("m:candle"), null);
  assert.equal(closures.get("m:absent"), undefined);
});

test("publication keeps anchored and structurally exempt coordinates, and counts the rest", async () => {
  const full = await buildMatrix({ publish: false });
  const published = await buildMatrix();
  const publishedIds = new Set(published.cells.map((cell) => cell.id));
  for (const cell of full.cells) {
    assert.equal(publishedIds.has(cell.id), isPublishableCell(cell), cell.id);
  }
  assert.ok(published.cells.every((cell) => cell.anchor !== null || cell.evidence.structural.length));
  // A `Missing` coordinate is never published, even when an anchor covers its (model, tier, lane):
  // an anchor rides every rung of a lane, including rungs the route does not implement.
  assert.ok(published.cells.every((cell) => cell.state !== "Missing"));
  assert.ok(full.cells.some((cell) => cell.state === "Missing" && cell.anchor !== null));
  // The elision is counted, never silent, and the census still covers every resolved coordinate.
  assert.equal(published.summary.publishedCells + published.summary.elidedCells, published.summary.cells);
  assert.equal(
    Object.values(published.summary.elidedByState).reduce((total, count) => total + count, 0),
    published.summary.elidedCells,
  );
  assert.equal(
    published.coverage.reduce((total, row) => total + row.coordinates, 0),
    published.summary.cells,
  );
  assert.ok(published.summary.elidedByState.Implemented > 0);
});

test("the fingerprint covers exactly the anchor and catalog sources, and the artifact publishes that set", async () => {
  // sc-22513 shrank this from 26 sources to 21. What LEFT is the per-record evidence join the matrix
  // no longer performs: the calibration plan, the calibration evidence bundle, the provider closure
  // ledger, both rung-4 survey artifacts, the two engine capability dumps and the Cargo pin. None of
  // them can move a cell any more.
  assert.deepEqual(SOURCE_PATHS, {
    manifest: "config/manifests/builtin.models.jsonc",
    routingCatalog: "crates/sceneworks-core/src/jobs_store/routing/catalog.rs",
    routingCandle: "crates/sceneworks-core/src/jobs_store/routing/candle.rs",
    routingMlx: "crates/sceneworks-core/src/jobs_store/routing/mlx.rs",
    engines: "crates/sceneworks-worker/src/engines.rs",
    imageRouting: "crates/sceneworks-worker/src/image_jobs/base.rs",
    videoRouteWan: "crates/sceneworks-worker/src/video_jobs/wan.rs",
    videoRouteLtx: "crates/sceneworks-worker/src/video_jobs/ltx.rs",
    videoRouteSvd: "crates/sceneworks-worker/src/video_jobs/svd.rs",
    videoRouteBernini: "crates/sceneworks-worker/src/video_jobs/bernini.rs",
    videoRouteScail2: "crates/sceneworks-worker/src/video_jobs/scail2.rs",
    videoRouteKreaRealtime: "crates/sceneworks-worker/src/video_jobs/krea_realtime.rs",
    videoRouteCandle: "crates/sceneworks-worker/src/video_jobs/candle.rs",
    mlxFitGate: "crates/sceneworks-worker/src/mlx_fit_gate.rs",
    memoryRouteRegistry: "crates/sceneworks-worker/src/memory_route_registry.rs",
    instantId: "crates/sceneworks-worker/src/image_jobs/instantid.rs",
    anchorStore: "config/memory-anchors.json",
    anchorLoaderClosures: "config/anchor-loader-closures.json",
    anchorDerivation: "crates/sceneworks-core/src/memory_anchor.rs",
    anchorAdmission: "crates/sceneworks-worker/src/video_admission.rs",
    anchorAdmissionImageVram: "crates/sceneworks-worker/src/vram_gate.rs",
    anchorAdmissionImageCandle: "crates/sceneworks-worker/src/candle_memory_strategy.rs",
    anchorExtractor: "scripts/extract-memory-anchors.mjs",
  });
  const matrix = await buildMatrix({ publish: false });
  assert.deepEqual(
    Object.fromEntries(
      Object.entries(matrix.generatedFrom.sources).map(([name, entry]) => [name, entry.path]),
    ),
    { ...SOURCE_PATHS },
  );
  // The generator may not READ a removed source either. A path that survives only in a comment is
  // fine; a `readFile` of it would be a source outside the fingerprint deciding a cell.
  const generator = await readFile(new URL("./generate-memory-matrix.mjs", import.meta.url), "utf8");
  const readPaths = [...generator.matchAll(/readFile\(path\.join\(ROOT, "([^"]+)"/g)].map(
    (match) => match[1],
  );
  assert.deepEqual(
    readPaths.filter((candidate) => !Object.values(SOURCE_PATHS).includes(candidate)),
    [],
  );
  for (const removed of [
    "config/memory-calibration-plan.json",
    "docs/generated/memory-calibration-evidence.json",
    "config/inference-provider-closures.json",
    "config/rung4-applicability-survey.json",
    "config/rung4-contract-prerequisites.json",
    "config/engine-capabilities/capabilities.mlx.json",
    "docs/generated/video-memory-curves.json",
  ]) {
    assert.ok(!generator.includes(`"${removed}"`), `${removed} is still named by the generator`);
  }
  // The mutate-regenerate-restore half of this claim lives in
  // `scripts/matrix-removed-sources.test.mjs`, which `npm run check` runs in its own SERIAL
  // `node --test` segment: it edits shared repo artifacts on disk, and several sibling suites hash
  // those same bytes, so running it in the parallel pool would make them flake.
});

test("the out-of-matrix subtraction may not outlive its reason (sc-18663, sc-22513)", () => {
  const entries = [...OUT_OF_MATRIX_CATALOG_ENTRIES.keys()];
  assert.deepEqual(entries, ["minimax_h3", "minimax_h3_ref"]);
  const catalog = entries.map((id) => ({ id }));
  const unroutable = () => {
    throw new Error("no video-route resolver row");
  };
  assertOutOfMatrixEntriesAreStillUnroutable(catalog, unroutable);
  // The day the generator can resolve one of them, it must join the universe rather than stay
  // silently subtracted.
  assert.throws(
    () => assertOutOfMatrixEntriesAreStillUnroutable(catalog, () => ({ engine: "minimax_h3" })),
    /now resolves its route/,
  );
  // And a declaration for an entry the catalog no longer carries is stale, not inert.
  assert.throws(
    () => assertOutOfMatrixEntriesAreStillUnroutable([{ id: "minimax_h3" }], unroutable),
    /the catalog no longer carries/,
  );
});

test("the implementation verdict is a claim about CODE, with no memory term in it", () => {
  // Every arm of `implementationVerdict` answers from the manifest and the worker sources. The
  // signature is the guarantee: it takes no anchor store, no derivation, no evidence of any kind, so
  // a memory fact cannot reach it even by accident.
  const source = implementationVerdict.toString();
  for (const forbidden of ["anchor", "derivation", "calibrationFingerprint", "record"]) {
    assert.ok(!source.includes(`${forbidden}Store`), forbidden);
  }
  const model = {
    id: "fixture",
    mlx: {
      memoryStrategyStructuralExemptions: {
        bounded_decode: {
          overlays: ["none"],
          evidence: [{ source: "crates/fixture.rs#no_separable_decode" }],
        },
      },
    },
  };
  const route = { kind: "registry", engine: "fixture", engineFor: () => "fixture" };
  const call = (rung) =>
    implementationVerdict({
      backend: "mlx",
      rung,
      route,
      provider: "fixture",
      stagedResidencyEngines: new Map(),
      model,
      tier: "q4",
      mode: "text_to_image",
      overlay: "none",
      manifestById: new Map([["fixture", model]]),
      candleBespokeStagedLanes: new Map(),
    });
  assert.equal(call("bounded_decode").implementation, "structurally-na");
  assert.equal(call("bounded_decode").structural.length, 1);
  assert.equal(call("resident").implementation, "implemented");
  assert.equal(call("bounded_attention").implementation, "missing");
  // A lane with no per-backend manifest block at all is wholly untriaged, never inferred from the
  // other backend.
  assert.equal(
    implementationVerdict({
      backend: "candle",
      rung: "resident",
      route,
      provider: "fixture",
      stagedResidencyEngines: new Map(),
      model,
      tier: "q4",
      mode: "text_to_image",
      overlay: "none",
      manifestById: new Map([["fixture", model]]),
      candleBespokeStagedLanes: new Map(),
    }).implementation,
    "missing",
  );
});

test("the anchor inventory is closed against the cells, in both directions (sc-22513)", async () => {
  const matrix = await buildMatrix();
  const inventory = new Map(matrix.anchors.map((anchor) => [anchor.id, anchor]));
  for (const cell of matrix.cells) {
    if (!cell.anchor) continue;
    const anchor = inventory.get(cell.anchor.id);
    assert.ok(anchor, `${cell.id} cites an unpublished anchor`);
    assert.deepEqual(
      [anchor.modelId, anchor.backend, anchor.tier],
      [cell.modelId, cell.backend, cell.tier],
    );
    assert.equal(cell.anchor.current, anchor.current);
  }
  const cited = new Set(matrix.cells.filter((cell) => cell.anchor).map((cell) => cell.anchor.id));
  // The `cells` column is the COUNT of resolved coordinates the anchor covers, so it is recomputed
  // from the pre-publication document — a row that overstated its reach would otherwise read as
  // wider evidence than the store has.
  const resolved = await buildMatrix({ publish: false });
  for (const anchor of matrix.anchors) {
    assert.equal(
      anchor.cells,
      resolved.cells.filter(
        (cell) => cell.anchor?.id === anchor.id && cell.implementation !== "missing",
      ).length,
      anchor.id,
    );
    // An anchor that reaches coordinates must be cited by a published one, and one that reaches
    // none must be cited by nothing. Silence in either direction is the store drifting away.
    assert.equal(anchor.cells > 0, cited.has(anchor.id), anchor.id);
  }
  // The closure guard is REACHABLE, not decorative: a document that lost a cited anchor's row, or
  // kept a row nothing cites, is refused before it can be written.
  assert.throws(
    () =>
      assertPublishedDocumentIsClosed(
        { ...matrix, anchors: matrix.anchors.filter((row) => row.cells === 0) },
        matrix.summary.cells,
      ),
    /which the document does not publish/,
  );
  assert.throws(
    () =>
      assertPublishedDocumentIsClosed(
        { ...matrix, cells: matrix.cells.map((cell) => ({ ...cell, anchor: null })) },
        matrix.summary.cells,
      ),
    /but no published cell cites it/,
  );
  assert.equal(matrix.summary.anchors, matrix.anchors.length);
  assert.ok(matrix.summary.anchoredCells > 0);
  // The rendered page carries the same inventory, so a reader of the markdown sees what the JSON says.
  const markdown = renderMarkdown(matrix);
  for (const anchor of matrix.anchors) assert.ok(markdown.includes(anchor.id), anchor.id);
  assert.ok(markdown.includes("Measured anchors"));
  // The retired vocabulary may not appear as a rendered STATE (the prose says it was retired).
  assert.ok(!markdown.includes("| Runtime verified |"));
  assert.ok(!markdown.includes("| Implemented/unverified |"));
});
