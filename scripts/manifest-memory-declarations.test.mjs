import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { stripJsoncComments } from "./lib/jsonc.mjs";
import {
  GENERATED_BEGIN,
  GENERATED_END,
  catalogAxes,
  clearProjection,
  engineContractInventory,
  modelSpans,
  parseEngineModelTable,
  parseRequestContextLanes,
  parseStrictControlEngines,
  projectManifestBody,
  projectProviderRows,
  routeWitnessInventory,
  rustStringSlice,
  withheldRungs,
  witnessCoordinatesForTier,
} from "./lib/manifest-memory-declarations.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const read = (relative) => readFileSync(path.join(ROOT, relative), "utf8");

// --- fixtures ------------------------------------------------------------------------------------
// Deliberately tiny and synthetic. Every assertion below is about SHAPE — which rungs, tiers, modes
// and overlays a projection does or does not emit, and which text survives a round trip. None asserts
// a population count against the real dumps: those are frozen-corpus assertions that a re-dump
// invalidates wholesale.

const surface = (tier, implementedRungs, deferred = []) => ({
  selector: { tier, offloadPolicy: "resident", loadShape: "deferred_materialization" },
  implementedRungs,
  structurallyNotApplicableRungs: [],
  deferredMaterializationRungs: deferred,
});

const witness = (provider, tier, mode, overlay, loadProfile) => ({
  provider,
  tier,
  mode,
  overlay,
  loadProfile,
});

function fixtureFacts({ rungs = ["resident", "staged_residency", "bounded_decode"] } = {}) {
  return [
    {
      backend: "mlx",
      generatedFrom: { inferenceRevision: "deadbeef" },
      memoryContracts: [
        {
          id: "widget",
          composed: false,
          surfaces: [surface("bf16", rungs), surface("q4", rungs)],
        },
        {
          id: "widget_control",
          composed: false,
          surfaces: [surface("q4", rungs)],
        },
      ],
      memoryRouteWitnesses: [
        witness("widget", "bf16", "text_to_image", "none", "plain"),
        witness("widget", "q4", "text_to_image", "none", "plain"),
        witness("widget", "q4", "text_to_image", "lora", "lora"),
        witness("widget", "q4", "edit_image", "none", "plain"),
        witness("widget", "q4", "edit_image", "lora", "lora"),
        witness("widget_control", "q4", "text_to_image", "control", "single_control"),
      ],
    },
    {
      backend: "candle",
      generatedFrom: { inferenceRevision: "deadbeef" },
      memoryContracts: [],
      memoryRouteWitnesses: [witness("widget", "q4", "text_to_image", "none", "plain")],
    },
  ];
}

const FIXTURE_ENGINES = `
pub(crate) const MODEL_TABLE: &[ModelRow] = &[
    ModelRow {
        sceneworks_id: "widget",
        engine_id: "widget",
        default_repo: "Example/widget",
    },
];
`;

const FIXTURE_STRICT_CONTROL = `
const STRICT_CONTROL_ENGINES: &[StrictControlEngine] = &[
    StrictControlEngine {
        engine_id: "widget_control",
        repo: "Example/widget-control",
        supported_kinds: &[ControlKind::Pose],
    },
];
`;

const FIXTURE_IMAGE_ROUTING = `
const WIRED_MLX_POSE_FAMILIES: &[&str] = &["widget"];
const WIRED_CANDLE_POSE_FAMILIES: &[&str] = &["widget"];
`;

// `legacy_shaping: true` everywhere, so the fixture lanes take projected rows. The
// request-context test below flips one row to false.
const fixtureRouteRegistry = (legacy = "true") => `
const RULES: &[MemoryRouteRule] = &[
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "widget",
        tiers: ALL_TIERS,
        modes: TEXT_ONLY,
        load_profiles: PLAIN,
        requires_sequential_selection: false,
        legacy_shaping: ${legacy},
    },
    MemoryRouteRule {
        backend: MemoryRouteBackend::Mlx,
        provider: "widget_control",
        tiers: ALL_TIERS,
        modes: TEXT_ONLY,
        load_profiles: SINGLE_CONTROL,
        requires_sequential_selection: false,
        legacy_shaping: true,
    },
];
`;

const FIXTURE_MANIFEST = `{
  // A hand-written comment that must survive every projection.
  "schemaVersion": 1,
  "models": [
    {
      "id": "widget",
      "type": "image",
      "capabilities": ["text_to_image", "edit_image"],
      "loraCompatibility": true,
      "downloads": [{ "repo": "Example/widget", "variant": "q4" }],
      "mlx": {
        // Measured, hand-authored, never rewritten.
        "quantize": 4,
        "vramGbByTier": { "q4": 4.2 }
      },
      "limits": {
        "samplers": ["default"]
      },
      "ui": { "label": "Widget" }
    }
  ]
}
`;

const fixtureInput = (overrides = {}) => ({
  body: FIXTURE_MANIFEST,
  engineFacts: fixtureFacts(),
  enginesSource: FIXTURE_ENGINES,
  strictControlSource: FIXTURE_STRICT_CONTROL,
  imageRoutingSource: FIXTURE_IMAGE_ROUTING,
  routeRegistrySource: fixtureRouteRegistry(),
  ...overrides,
});

const parse = (body) => JSON.parse(stripJsoncComments(body));
const widgetContract = (body, backend = "mlx") =>
  parse(body).models[0][backend].memoryStrategyContract;

// --- engine inventory ----------------------------------------------------------------------------

test("the engine inventory keys rungs by tier and never declares the resident floor", () => {
  const inventory = engineContractInventory(fixtureFacts());
  const widget = inventory.get("mlx:widget");
  assert.deepEqual([...widget.rungs.keys()], ["staged_residency", "bounded_decode"]);
  assert.deepEqual(widget.rungs.get("staged_residency"), ["bf16", "q4"]);
  assert.equal(widget.backend, "mlx");
});

test("a rung implemented under any selector for a tier counts for that tier", () => {
  const facts = fixtureFacts();
  // Two surfaces for one tier, the rung present in only one of them.
  facts[0].memoryContracts[0].surfaces = [
    surface("q4", ["resident"]),
    {
      selector: { tier: "q4", offloadPolicy: "sequential", loadShape: "eager_materialization" },
      implementedRungs: ["resident", "bounded_attention"],
      structurallyNotApplicableRungs: [],
      deferredMaterializationRungs: [],
    },
  ];
  const widget = engineContractInventory(facts).get("mlx:widget");
  assert.deepEqual(widget.rungs.get("bounded_attention"), ["q4"]);
});

test("route witnesses index per tier and report an unwitnessed tier as absent", () => {
  const witnesses = routeWitnessInventory(fixtureFacts()).get("mlx:widget");
  const q4 = witnessCoordinatesForTier(witnesses, "q4");
  assert.deepEqual(q4.modes, ["text_to_image", "edit_image"]);
  assert.deepEqual(q4.overlays, ["none", "lora"]);
  assert.deepEqual(q4.loadProfiles, ["plain", "lora"]);
  assert.equal(witnessCoordinatesForTier(witnesses, "q8"), null);
});

// --- catalog reach -------------------------------------------------------------------------------

test("catalog axes come from the entry's own capabilities, lora flag and advertised tiers", () => {
  const model = parse(FIXTURE_MANIFEST).models[0];
  const poseFamilies = {
    mlx: rustStringSlice(FIXTURE_IMAGE_ROUTING, "WIRED_MLX_POSE_FAMILIES"),
    candle: rustStringSlice(FIXTURE_IMAGE_ROUTING, "WIRED_CANDLE_POSE_FAMILIES"),
  };
  const axes = catalogAxes(model, "mlx", poseFamilies);
  assert.deepEqual([...axes.modes], ["text_to_image", "edit_image"]);
  assert.deepEqual([...axes.overlays], ["none", "lora", "control"]);
  assert.ok(axes.tiers.has("q4"));
  assert.ok(!axes.tiers.has("q8"), "a tier the entry never advertises is not catalog-reachable");
});

test("an entry outside the wired pose families has no control overlay to declare", () => {
  const model = parse(FIXTURE_MANIFEST).models[0];
  const axes = catalogAxes(model, "mlx", { mlx: new Set(), candle: new Set() });
  assert.ok(!axes.overlays.has("control"));
});

// --- source-of-truth parsers ---------------------------------------------------------------------

test("the model table and strict-control table parse from their Rust sources", () => {
  assert.equal(parseEngineModelTable(FIXTURE_ENGINES).get("widget"), "widget");
  assert.ok(parseStrictControlEngines(FIXTURE_STRICT_CONTROL).has("widget_control"));
});

test("a missing source table is a hard failure, never an empty projection", () => {
  assert.throws(() => parseEngineModelTable("// nothing here"), /MODEL_TABLE/);
  assert.throws(() => parseStrictControlEngines("// nothing here"), /STRICT_CONTROL_ENGINES/);
  assert.throws(() => rustStringSlice("// nothing here", "WIRED_MLX_POSE_FAMILIES"), /base\.rs/);
  assert.throws(() => parseRequestContextLanes("// nothing here"), /RULES/);
});

test("only non-legacy MLX route rules mark a lane as requiring request contexts", () => {
  assert.deepEqual([...parseRequestContextLanes(fixtureRouteRegistry("true"))], []);
  assert.deepEqual([...parseRequestContextLanes(fixtureRouteRegistry("false"))], ["mlx:widget"]);
});

test("an MLX lane requiring requestContexts takes no projected row at all", () => {
  // A row without `requestContexts` on such a lane is read as MALFORMED and fails the whole load
  // closed to Refused + Eager, so this is a correctness constraint, not a tidiness one.
  const result = projectManifestBody(
    fixtureInput({ routeRegistrySource: fixtureRouteRegistry("false") }),
  );
  const rows = widgetContract(result.body).implementations.filter(
    (row) => (row.runtimeProvider ?? "widget") === "widget",
  );
  assert.equal(rows.length, 0, "the base lane is skipped");
  assert.ok(result.skipped.some((entry) => entry.reason === "requires-request-contexts"));
  // The sibling control lane is still legacy-shaped, so it is unaffected.
  assert.ok(
    widgetContract(result.body).implementations.some(
      (row) => row.runtimeProvider === "widget_control",
    ),
  );
});

// --- row projection ------------------------------------------------------------------------------

const projectWidget = (options = {}) => {
  const inventory = engineContractInventory(options.facts ?? fixtureFacts());
  const witnesses = routeWitnessInventory(options.facts ?? fixtureFacts());
  const host = options.host ?? parse(FIXTURE_MANIFEST).models[0];
  const provider = options.provider ?? "widget";
  return projectProviderRows({
    backend: "mlx",
    provider,
    contractProvider: options.contractProvider ?? "widget",
    host,
    engine: inventory.get(`mlx:${provider}`),
    witnesses: witnesses.get(`mlx:${provider}`),
    withhold: withheldRungs(host, "mlx"),
    axes: catalogAxes(host, "mlx", {
      mlx: rustStringSlice(FIXTURE_IMAGE_ROUTING, "WIRED_MLX_POSE_FAMILIES"),
      candle: rustStringSlice(FIXTURE_IMAGE_ROUTING, "WIRED_CANDLE_POSE_FAMILIES"),
    }),
  });
};

test("projected rows carry only engine-derived axes and never guess measured shape", () => {
  const { rows } = projectWidget();
  assert.ok(rows.length > 0);
  for (const row of rows) {
    assert.deepEqual(Object.keys(row), ["rung", "tiers", "modes", "overlays", "loadProfiles", "source"]);
    assert.ok(row.tiers.length > 0 && row.modes.length > 0 && row.overlays.length > 0);
    assert.match(row.source, /^config\/engine-capabilities\/capabilities\.mlx\.json#memoryContracts\//);
  }
});

test("a tier the catalog does not advertise is skipped and reported, not declared", () => {
  const { rows, skipped } = projectWidget();
  assert.ok(
    rows.every((row) => !row.tiers.includes("bf16")),
    "bf16 is engine-implemented but the fixture entry advertises only q4",
  );
  assert.ok(skipped.some((entry) => entry.reason === "tier-not-advertised"));
});

test("a rung with no route witness at a tier is skipped and reported", () => {
  const facts = fixtureFacts();
  facts[0].memoryRouteWitnesses = facts[0].memoryRouteWitnesses.filter(
    (row) => row.tier !== "q4" || row.provider !== "widget",
  );
  const { rows, skipped } = projectWidget({ facts });
  assert.equal(rows.length, 0);
  assert.ok(skipped.some((entry) => entry.reason === "no-route-witness"));
});

test("runtimeProvider appears only when the row is not the contract's own provider", () => {
  const base = projectWidget();
  assert.ok(base.rows.every((row) => row.runtimeProvider === undefined));
  const control = projectWidget({ provider: "widget_control" });
  assert.ok(control.rows.length > 0);
  assert.ok(control.rows.every((row) => row.runtimeProvider === "widget_control"));
  assert.ok(control.rows.every((row) => row.overlays.includes("control")));
});

test("a generated row never claims an overlay another runtime provider already owns", () => {
  const host = parse(FIXTURE_MANIFEST).models[0];
  host.mlx.memoryStrategyContract = {
    abi: 1,
    provider: "widget",
    implementations: [
      {
        rung: "bounded_decode",
        runtimeProvider: "widget_sibling",
        tiers: ["q4"],
        modes: ["text_to_image"],
        overlays: ["lora"],
      },
    ],
  };
  const { rows } = projectWidget({ host });
  assert.ok(rows.length > 0);
  assert.ok(
    rows.every((row) => !row.overlays.includes("lora")),
    "the lora overlay is owned by widget_sibling on this contract",
  );
});

test("a tier already covered by a hand-authored row for the same rung is not re-declared", () => {
  const facts = fixtureFacts();
  // q4 only, so the hand-authored q4 row below covers the rung's whole engine tier set.
  facts[0].memoryContracts[0].surfaces = [surface("q4", ["resident", "bounded_decode"])];
  const host = parse(FIXTURE_MANIFEST).models[0];
  host.mlx.memoryStrategyContract = {
    abi: 1,
    provider: "widget",
    implementations: [
      {
        rung: "bounded_decode",
        tiers: ["q4"],
        modes: ["text_to_image"],
        overlays: ["none"],
      },
    ],
  };
  const { rows, skipped } = projectWidget({ facts, host });
  assert.ok(rows.every((row) => row.rung !== "bounded_decode"));
  assert.ok(skipped.some((entry) => entry.rung === "bounded_decode" && entry.reason === "already-declared"));
});

test("a declared withhold is honored and reported rather than overridden", () => {
  const host = parse(FIXTURE_MANIFEST).models[0];
  host.mlx.memoryDeclarationWithhold = {
    rungs: ["bounded_decode"],
    story: "SC-0000",
    reason: "measured quality regression",
  };
  const { rows, skipped } = projectWidget({ host });
  assert.ok(rows.every((row) => row.rung !== "bounded_decode"));
  const withheld = skipped.find((entry) => entry.reason === "withheld");
  assert.equal(withheld.rung, "bounded_decode");
  assert.equal(withheld.declaration.story, "SC-0000");
});

test("a whole-backend withhold suppresses every rung on that block", () => {
  const host = parse(FIXTURE_MANIFEST).models[0];
  host.mlx.memoryDeclarationWithhold = { rungs: "all", story: "SC-0000", reason: "measured" };
  assert.equal(projectWidget({ host }).rows.length, 0);
});

test("a malformed withhold fails loudly instead of silently projecting everything", () => {
  const host = parse(FIXTURE_MANIFEST).models[0];
  host.mlx.memoryDeclarationWithhold = { rungs: ["not_a_rung"] };
  assert.throws(() => withheldRungs(host, "mlx"), /unknown rung/);
  host.mlx.memoryDeclarationWithhold = { rungs: [] };
  assert.throws(() => withheldRungs(host, "mlx"), /non-empty array/);
});

test("tiers whose witnessed coordinates differ land in separate rows, never a union", () => {
  const facts = fixtureFacts();
  facts[0].memoryContracts[0].surfaces = [
    surface("q4", ["resident", "bounded_decode"]),
    surface("q8", ["resident", "bounded_decode"]),
  ];
  // q8 witnesses only the plain text lane; q4 also witnesses lora and edit.
  facts[0].memoryRouteWitnesses.push(witness("widget", "q8", "text_to_image", "none", "plain"));
  const host = parse(FIXTURE_MANIFEST).models[0];
  host.downloads.push({ repo: "Example/widget", variant: "q8" });
  host.mlx.vramGbByTier = { q4: 4.2, q8: 6.1 };
  const { rows } = projectWidget({ facts, host });
  const decode = rows.filter((row) => row.rung === "bounded_decode");
  assert.equal(decode.length, 2, "one row per distinct witnessed coordinate set");
  const q8 = decode.find((row) => row.tiers.includes("q8"));
  assert.deepEqual(q8.modes, ["text_to_image"]);
  assert.deepEqual(q8.overlays, ["none"]);
});

// --- mutation: the projection tracks the dump surface ---------------------------------------------

test("adding a rung to a dump surface adds a generated row", () => {
  const before = projectManifestBody(fixtureInput());
  const widened = fixtureFacts({
    rungs: ["resident", "staged_residency", "bounded_decode", "bounded_attention"],
  });
  const after = projectManifestBody(fixtureInput({ engineFacts: widened }));
  const rungsOf = (body) => widgetContract(body).implementations.map((row) => row.rung);
  assert.ok(!rungsOf(before.body).includes("bounded_attention"));
  assert.ok(rungsOf(after.body).includes("bounded_attention"));
});

test("removing a rung from a dump surface removes its generated row", () => {
  const narrowed = fixtureFacts({ rungs: ["resident", "staged_residency"] });
  const after = projectManifestBody(fixtureInput({ engineFacts: narrowed }));
  const rungs = widgetContract(after.body).implementations.map((row) => row.rung);
  assert.ok(!rungs.includes("bounded_decode"));
  assert.ok(rungs.includes("staged_residency"));
});

test("removing a provider from the dump removes its rows and reports nothing stale", () => {
  const facts = fixtureFacts();
  facts[0].memoryContracts = facts[0].memoryContracts.filter((row) => row.id !== "widget_control");
  const after = projectManifestBody(fixtureInput({ engineFacts: facts }));
  const providers = widgetContract(after.body).implementations.map(
    (row) => row.runtimeProvider ?? "widget",
  );
  assert.ok(!providers.includes("widget_control"));
});

test("a provider with no image-manifest host is reported, never silently dropped", () => {
  const facts = fixtureFacts();
  facts[0].memoryContracts.push({
    id: "orphan",
    composed: false,
    surfaces: [surface("q4", ["resident", "bounded_decode"])],
  });
  const result = projectManifestBody(fixtureInput({ engineFacts: facts }));
  assert.ok(result.unhosted.some((entry) => entry.provider === "orphan"));
});

// --- idempotence and hand-content preservation ---------------------------------------------------

test("regeneration is byte-identical on the fixture manifest", () => {
  const first = projectManifestBody(fixtureInput()).body;
  const second = projectManifestBody(fixtureInput({ body: first })).body;
  assert.equal(second, first);
});

test("clearing a projection restores the fixture body byte-for-byte", () => {
  const projected = projectManifestBody(fixtureInput()).body;
  assert.notEqual(projected, FIXTURE_MANIFEST);
  assert.equal(clearProjection(projected), FIXTURE_MANIFEST);
});

test("hand-authored non-memory content survives a projection cycle unchanged", () => {
  const projected = projectManifestBody(fixtureInput()).body;
  const before = parse(FIXTURE_MANIFEST).models[0];
  const after = parse(projected).models[0];
  for (const key of ["id", "type", "capabilities", "loraCompatibility", "downloads", "limits", "ui"]) {
    assert.deepEqual(after[key], before[key], `${key} is untouched by the projection`);
  }
  const { memoryStrategyContract: _generated, ...mlx } = after.mlx;
  assert.deepEqual(mlx, before.mlx, "the backend block keeps every hand-authored key");
  assert.match(projected, /A hand-written comment that must survive/);
});

test("generated regions are delimited and never nested", () => {
  const projected = projectManifestBody(fixtureInput()).body;
  const begins = projected.split(GENERATED_BEGIN).length - 1;
  const ends = projected.split(GENERATED_END).length - 1;
  assert.equal(begins, ends, "every BEGIN marker has exactly one END marker");
  let cursor = 0;
  for (let index = 0; index < begins; index += 1) {
    const begin = projected.indexOf(GENERATED_BEGIN, cursor);
    const end = projected.indexOf(GENERATED_END, begin);
    assert.ok(end > begin, "the region closes after it opens");
    assert.equal(
      projected.indexOf(GENERATED_BEGIN, begin + GENERATED_BEGIN.length) > end ||
        projected.indexOf(GENERATED_BEGIN, begin + GENERATED_BEGIN.length) === -1,
      true,
      "no BEGIN marker opens inside an unclosed region",
    );
    cursor = end + GENERATED_END.length;
  }
});

// --- the committed manifest ----------------------------------------------------------------------

test("the committed manifest is a fixed point of the committed dumps", () => {
  const body = read("config/manifests/builtin.models.jsonc");
  const result = projectManifestBody({
    body,
    engineFacts: ["mlx", "candle"].map((backend) =>
      JSON.parse(read(`config/engine-capabilities/capabilities.${backend}.json`)),
    ),
    enginesSource: read("crates/sceneworks-worker/src/engines.rs"),
    strictControlSource: read("crates/sceneworks-worker/src/image_jobs/strict_control.rs"),
    imageRoutingSource: read("crates/sceneworks-worker/src/image_jobs/base.rs"),
    routeRegistrySource: read("crates/sceneworks-worker/src/memory_route_registry.rs"),
  });
  assert.equal(
    result.body,
    body,
    "run `npm run generate:manifest-memory-declarations` — the committed projection is stale",
  );
  const manifest = parse(result.body);
  // Shape, not population: every generated row must be a usable declaration.
  for (const model of manifest.models) {
    for (const backend of ["mlx", "candle"]) {
      for (const row of model[backend]?.memoryStrategyContract?.implementations ?? []) {
        if (!row.source?.startsWith("config/engine-capabilities/")) continue;
        assert.ok(row.tiers?.length, `${model.id}:${backend}:${row.rung} declares tiers`);
        assert.ok(row.modes?.length, `${model.id}:${backend}:${row.rung} declares modes`);
        assert.ok(row.overlays?.length, `${model.id}:${backend}:${row.rung} declares overlays`);
        assert.ok(row.loadProfiles?.length, `${model.id}:${backend}:${row.rung} declares profiles`);
      }
    }
  }
});

test("every model object in the committed manifest is locatable by the text walker", () => {
  const body = read("config/manifests/builtin.models.jsonc");
  const spans = modelSpans(body);
  for (const model of parse(body).models) {
    assert.ok(spans.has(model.id), `${model.id} has a located span`);
    const span = spans.get(model.id);
    assert.deepEqual(parse(body.slice(span.start, span.end)), model);
  }
});
