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
  contractIsLoraOnly,
  engineContractInventory,
  modelSpans,
  parseEngineModelTable,
  parseRequestContextLanes,
  parseStrictControlEngines,
  projectManifestBody,
  projectProviderRows,
  routeWitnessInventory,
  splitTrailingTrivia,
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

// Same entry, but the LAST key of the `mlx` block carries a trailing same-line note. The
// whole-contract insert appends after that key, so its separator comma must land on the value and not
// past the note — a comma appended after `// ...` becomes comment text and the manifest stops parsing.
const FIXTURE_MANIFEST_TRAILING_NOTE = FIXTURE_MANIFEST.replace(
  '"vramGbByTier": { "q4": 4.2 }',
  '"vramGbByTier": { "q4": 4.2 } // measured on an M3 Max, do not round',
);

// A hand-authored `implementations` array whose last line before `]` is a STANDALONE comment. The
// in-array separator comma has the same hazard there.
const FIXTURE_MANIFEST_TRAILING_COMMENT_LINE = FIXTURE_MANIFEST.replace(
  '        "vramGbByTier": { "q4": 4.2 }',
  `        "vramGbByTier": { "q4": 4.2 },
        "memoryStrategyContract": {
          "abi": 1,
          "provider": "widget",
          "implementations": [
            {
              "rung": "bounded_decode",
              "fingerprint": "widget-hand-authored-v1",
              "tiers": ["q4"],
              "modes": ["text_to_image"],
              "overlays": ["none"],
              "engagedRungs": ["resident", "bounded_decode"],
              "parameters": { "decodeTileEdge": 512 },
              "parameterRanges": { "decodeTileEdges": [512] },
              "source": "inference:crates/media/mlx-gen/mlx-gen-widget/src/memory_strategy.rs"
            }
            // A parting thought about the row above, which must not swallow the next comma.
          ]
        }`,
);

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

test("a LoRA-only backend contract removes the unreachable plain overlay", () => {
  // STRUCTURAL, not id-keyed (sc-20799): any model whose exact contract names only the LoRA load
  // profile for its base provider is suppressed — the fixture keeps its own id to prove no
  // magic-id list exists.
  const model = parse(FIXTURE_MANIFEST).models[0];
  model.candle = {
    memoryStrategyContract: {
      provider: "widget",
      implementations: [{ overlays: ["lora"] }],
    },
  };
  model.loraCompatibility = { families: ["widget"] };
  const axes = catalogAxes(model, "candle", { mlx: new Set(), candle: new Set() }, "widget");
  assert.deepEqual([...axes.overlays], ["lora"]);
});

test("contractIsLoraOnly keys on the contract's shape alone", () => {
  const loraOnly = {
    candle: {
      memoryStrategyContract: { provider: "widget", implementations: [{ overlays: ["lora"] }] },
    },
  };
  assert.ok(contractIsLoraOnly(loraOnly, "candle", "widget"));
  // A single row serving the plain profile too is NOT lora-only.
  const mixedRow = {
    candle: {
      memoryStrategyContract: {
        provider: "widget",
        implementations: [{ overlays: ["none", "lora"] }],
      },
    },
  };
  assert.ok(!contractIsLoraOnly(mixedRow, "candle", "widget"));
  // One plain row among lora rows is NOT lora-only.
  const mixedRows = {
    candle: {
      memoryStrategyContract: {
        provider: "widget",
        implementations: [{ overlays: ["lora"] }, { overlays: ["none"] }],
      },
    },
  };
  assert.ok(!contractIsLoraOnly(mixedRows, "candle", "widget"));
  // No contract / no implementations: nothing to suppress on.
  assert.ok(!contractIsLoraOnly({}, "candle", "widget"));
  assert.ok(!contractIsLoraOnly(
    { candle: { memoryStrategyContract: { provider: "widget", implementations: [] } } },
    "candle",
    "widget",
  ));
  // The predicate is per-backend: a lora-only candle contract says nothing about mlx.
  assert.ok(!contractIsLoraOnly(loraOnly, "mlx", "widget"));
  // A sibling runtimeProvider's lora row describes ITS lane, not the base model's plain route.
  const siblingOnly = {
    candle: {
      memoryStrategyContract: {
        provider: "widget",
        implementations: [{ runtimeProvider: "widget_sibling", overlays: ["lora"] }],
      },
    },
  };
  assert.ok(!contractIsLoraOnly(siblingOnly, "candle", "widget"));
  // A contract whose OWN provider is a sibling lane (a route-local edit provider, the Krea Turbo
  // shape) never suppresses the base model's plain route.
  const siblingContract = {
    candle: {
      memoryStrategyContract: {
        provider: "widget_edit",
        implementations: [{ overlays: ["lora"] }],
      },
    },
  };
  assert.ok(!contractIsLoraOnly(siblingContract, "candle", "widget"));
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
  const cited = { story: "SC-0000", reason: "measured" };
  host.mlx.memoryDeclarationWithhold = { ...cited, rungs: ["not_a_rung"] };
  assert.throws(() => withheldRungs(host, "mlx"), /unknown rung/);
  host.mlx.memoryDeclarationWithhold = { ...cited, rungs: [] };
  assert.throws(() => withheldRungs(host, "mlx"), /non-empty array/);
  // An UNCITED withhold is the dangerous one: it is indistinguishable from the untriaged gap the
  // projection exists to close, so both fields are required here and in the authoring schema.
  host.mlx.memoryDeclarationWithhold = { rungs: "all", reason: "measured" };
  assert.throws(() => withheldRungs(host, "mlx"), /non-empty story/);
  host.mlx.memoryDeclarationWithhold = { rungs: "all", story: "SC-0000", reason: "   " };
  assert.throws(() => withheldRungs(host, "mlx"), /non-empty reason/);
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

test("a trailing same-line note at the insert anchor keeps the manifest parseable", () => {
  const projected = projectManifestBody(
    fixtureInput({ body: FIXTURE_MANIFEST_TRAILING_NOTE }),
  ).body;
  // The comma is on the value, the note is reproduced verbatim after it.
  assert.match(projected, /"vramGbByTier": \{ "q4": 4\.2 \}, \/\/ measured on an M3 Max, do not round/);
  assert.ok(widgetContract(projected).implementations.length > 0);
  assert.equal(clearProjection(projected), FIXTURE_MANIFEST_TRAILING_NOTE);
  assert.equal(
    projectManifestBody(fixtureInput({ body: projected })).body,
    projected,
    "still idempotent with a note at the anchor",
  );
});

test("a standalone trailing comment line in implementations keeps the manifest parseable", () => {
  const projected = projectManifestBody(
    fixtureInput({ body: FIXTURE_MANIFEST_TRAILING_COMMENT_LINE }),
  ).body;
  const rows = widgetContract(projected).implementations;
  assert.ok(rows.some((row) => row.fingerprint === "widget-hand-authored-v1"), "hand row survives");
  assert.ok(
    rows.some((row) => row.source?.startsWith("config/engine-capabilities/")),
    "projected rows were appended",
  );
  // The comma went after the row, NOT after the parting thought.
  assert.match(projected, /\},\n\s*\/\/ A parting thought about the row above/);
  assert.equal(clearProjection(projected), FIXTURE_MANIFEST_TRAILING_COMMENT_LINE);
  assert.equal(
    projectManifestBody(fixtureInput({ body: projected })).body,
    projected,
    "still idempotent with a standalone comment line before the closing bracket",
  );
});

test("the trivia splitter never places a comma inside a comment or a string", () => {
  assert.deepEqual(splitTrailingTrivia('"a": 1'), ['"a": 1', ""]);
  assert.deepEqual(splitTrailingTrivia('"a": 1 // note'), ['"a": 1', " // note"]);
  assert.deepEqual(splitTrailingTrivia('"a": 1\n  // note\n'), ['"a": 1', "\n  // note\n"]);
  // A `//` inside a string is not a comment, so the whole line is content.
  assert.deepEqual(
    splitTrailingTrivia('"url": "https://example.com/x"'),
    ['"url": "https://example.com/x"', ""],
  );
  assert.deepEqual(
    splitTrailingTrivia('"url": "https://example.com/x" // and a real note'),
    ['"url": "https://example.com/x"', " // and a real note"],
  );
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

// The committed manifest's FRESHNESS is deliberately not asserted here.
//
// A "the committed projection is a fixed point of the committed dumps" test would be a blocking
// freshness invariant in `npm run check` — a new gate, which is exactly what Michael's 2026-08-17
// decision rules out, and it would contradict this generator's own "NOT A GATE" contract. The
// freshness signal lives in `npm run report:memory-contract-reconciliation`, which prints whether the
// manifest is a fixed point of the committed dumps and always exits 0.
//
// Nothing is lost in coverage: the row-shape invariant that test also carried (every projected row
// declares non-empty tiers/modes/overlays/loadProfiles) is asserted on the fixture projection by
// "projected rows carry only engine-derived axes and never guess measured shape" above, where it
// tests the generator instead of the committed artifact.

test("every model object in the committed manifest is locatable by the text walker", () => {
  const body = read("config/manifests/builtin.models.jsonc");
  const spans = modelSpans(body);
  for (const model of parse(body).models) {
    assert.ok(spans.has(model.id), `${model.id} has a located span`);
    const span = spans.get(model.id);
    assert.deepEqual(parse(body.slice(span.start, span.end)), model);
  }
});

// sc-22730. A candle engine may WITHHOLD the production calibration identity for a load shape it
// still supports. `candle-gen-sd3`'s `production_calibration_identity` returns `None` the moment
// `!receipt.adapters.is_empty()` (inference `crates/media/candle-gen/candle-gen-sd3/src/
// memory_strategy.rs:876-886`, merged at `63056d4e9`): a loaded adapter stack adds resident bytes
// no anchor measured, so only the clean base cell carries an identity.
//
// A manifest row that declares a `fingerprint` for an overlay-bearing implementation therefore
// promises a string the shipped engine will never publish. Nothing else catches it — the row is
// structurally valid, the reconciliation walks fingerprints that EXIST, and the mismatch surfaces
// only as a mid-campaign refusal after a 28-57 GB load.
//
// The rule is read from the engine source when INFERENCE_REPO points at a checkout that carries it;
// otherwise it is asserted for the sd3_5 family explicitly, against the file:line cited above.
test("no overlay-bearing manifest row promises an identity its candle engine withholds under adapters", () => {
  // model id -> the engine crate whose `production_calibration_identity` we are claiming about.
  const withholding = {
    sd3_5_large: "candle-gen-sd3",
    sd3_5_large_turbo: "candle-gen-sd3",
    sd3_5_medium: "candle-gen-sd3",
  };

  // Derive the rule from the engine when the configured checkout has it, so a future engine that
  // starts publishing under adapters makes this test fail instead of silently over-asserting.
  const inferenceRepo = process.env.INFERENCE_REPO;
  let derived = false;
  if (inferenceRepo) {
    const enginePath = path.join(
      inferenceRepo,
      "crates/media/candle-gen/candle-gen-sd3/src/memory_strategy.rs",
    );
    let source = "";
    try {
      source = readFileSync(enginePath, "utf8");
    } catch {
      source = "";
    }
    const fn = /fn production_calibration_identity\([\s\S]*?\n\}/.exec(source);
    if (fn) {
      derived = true;
      assert.match(
        fn[0],
        /if !receipt\.adapters\.is_empty\(\) \{\s*return None;/,
        "candle-gen-sd3 no longer withholds the identity under adapters; this table is stale",
      );
    }
  }
  // At the COMPILED PIN the function does not exist yet, so a non-derived run is expected and the
  // citation above is the authority. Either way the manifest claim below is asserted.
  assert.equal(typeof derived, "boolean");

  const models = parse(read("config/manifests/builtin.models.jsonc")).models;
  let withheld = 0;
  let published = 0;
  for (const [id, crate] of Object.entries(withholding)) {
    const model = models.find((candidate) => candidate.id === id);
    assert.ok(model, `${id} is still a shipped model`);
    const rows = model.candle?.memoryStrategyContract?.implementations ?? [];
    assert.ok(rows.length > 0, `${id} still declares a candle memoryStrategyContract`);
    for (const row of rows) {
      const overlays = (row.overlays ?? []).filter((overlay) => overlay !== "none");
      const adapterBearing = overlays.length > 0 || (row.providerOverlay ?? "none") !== "none";
      if (adapterBearing) {
        assert.equal(
          row.fingerprint,
          undefined,
          `${id}: ${crate} publishes no identity under adapters, but the ${row.rung} ` +
            `${JSON.stringify(row.overlays)} row declares ${row.fingerprint}`,
        );
        withheld += 1;
      } else {
        assert.equal(
          typeof row.fingerprint,
          "string",
          `${id}: the clean base ${row.rung} row must still declare its identity`,
        );
        published += 1;
      }
    }
  }
  // Shape, not a frozen count: both halves of the split must be non-empty or the test asserts
  // nothing about one of them.
  assert.ok(withheld > 0, "no overlay-bearing row is covered; this test guards nothing");
  assert.ok(published > 0, "no clean base row is covered; this test guards nothing");
});
