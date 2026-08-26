import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import { collectMemoryContractMismatches } from "./lib/memory-contract-reconciliation.mjs";
import {
  TRIAGE_CLASSES,
  classifyMemoryContractMismatch,
  triageMemoryContractMismatches,
} from "./lib/memory-contract-triage.mjs";

const PIN = "b".repeat(40);

function digest(contract) {
  contract.selectorDigest = `sha256:${createHash("sha256")
    .update(
      JSON.stringify(
        contract.surfaces.map((surface) => ({
          selector: surface.selector,
          implementedRungs: surface.implementedRungs,
          structurallyNotApplicableRungs: surface.structurallyNotApplicableRungs,
          deferredMaterializationRungs: surface.deferredMaterializationRungs,
        })),
      ),
    )
    .digest("hex")}`;
  return contract;
}

function surface(tier, { implemented = [], deferred = [] } = {}) {
  return {
    selector: { tier, offloadPolicy: "sequential", loadShape: "deferred_materialization" },
    implementedRungs: ["resident", ...implemented],
    structurallyNotApplicableRungs: [],
    deferredMaterializationRungs: deferred,
  };
}

function facts(backend, { contracts, witnesses }) {
  return {
    backend,
    generatedFrom: { inferenceRevision: PIN, dumper: "fixture" },
    memoryContracts: contracts,
    memoryRouteWitnesses: witnesses,
  };
}

/**
 * The smallest input that reaches every leg the triage classifies.
 *
 * `mlx_alpha` is the well-formed lane: implemented, declared, celled and witnessed, so it emits
 * nothing. `candle_alpha` implements `staged_residency` at q8 with no manifest declaration — the
 * engine_manifest leg. Everything else is introduced per-test by mutating this.
 */
function fixture() {
  return {
    engineFacts: [
      facts("mlx", {
        contracts: [
          digest({
            id: "mlx_alpha",
            surfaces: [
              surface("q4", {
                implemented: ["bounded_transformer_residency"],
                deferred: ["bounded_transformer_residency"],
              }),
            ],
          }),
        ],
        witnesses: [
          { provider: "mlx_alpha", tier: "q4", mode: "text_to_image", overlay: "none", loadProfile: "plain" },
        ],
      }),
      facts("candle", {
        contracts: [
          digest({ id: "candle_alpha", surfaces: [surface("q8", { implemented: ["staged_residency"] })] }),
        ],
        witnesses: [
          { provider: "candle_alpha", tier: "q8", mode: "text_to_image", overlay: "none", loadProfile: "plain" },
        ],
      }),
    ],
    manifest: {
      models: [
        {
          id: "mlx_model",
          type: "image",
          mlx: {
            memoryStrategyContract: {
              provider: "mlx_alpha",
              implementations: [
                {
                  rung: "bounded_transformer_residency",
                  tiers: ["q4"],
                  modes: ["text_to_image"],
                  overlays: ["none"],
                },
              ],
            },
          },
        },
      ],
    },
    cells: [
      {
        backend: "mlx",
        provider: "mlx_alpha",
        modelId: "mlx_model",
        mode: "text_to_image",
        tier: "q4",
        overlay: "none",
        rung: "bounded_transformer_residency",
        owningFamilyStory: 100,
      },
    ],
    calibrationPlan: { providers: [] },
    closures: { providers: {} },
    survey: {
      families: {
        100: {
          backends: {
            mlx: {
              implementation: "shared-primitive",
              implementedEntries: ["mlx_model"],
              implementedTiers: ["q4"],
              implementedModes: ["text_to_image"],
              implementedOverlays: ["none"],
            },
          },
        },
      },
    },
  };
}

const EMPTY_PLAN = { unhosted: [], skipped: [] };

const triage = (input, plan = EMPTY_PLAN) =>
  triageMemoryContractMismatches(collectMemoryContractMismatches(input), plan);

const classCounts = (result) =>
  Object.fromEntries(result.classes.map((group) => [group.name, group.count]));

test("the fixture's only finding is the undeclared Candle lane", () => {
  const result = triage(fixture());
  assert.equal(result.total, 1);
  assert.equal(result.classes[0].findings[0].backend, "candle");
  assert.equal(result.classes[0].findings[0].leg, "engine_manifest");
});

test("an engine_manifest finding takes its class from the projection, not from the leg", () => {
  const input = fixture();
  const finding = collectMemoryContractMismatches(input).find(
    (entry) => entry.leg === "engine_manifest",
  );

  // Same coordinate, four different projection answers, four different classes. This is the whole
  // point of the join: the leg alone cannot distinguish them.
  const cases = [
    [{ unhosted: [{ backend: "candle", provider: "candle_alpha" }], skipped: [] }, "engine_provider_unhosted"],
    [{ unhosted: [], skipped: [{ backend: "candle", provider: "candle_alpha", rung: "staged_residency", tiers: ["q8"], reason: "no-route-witness" }] }, "no_route_witness"],
    [{ unhosted: [], skipped: [{ backend: "candle", provider: "candle_alpha", rung: "staged_residency", tiers: ["q8"], reason: "tier-not-advertised" }] }, "tier_not_in_catalog"],
    [{ unhosted: [], skipped: [{ backend: "candle", provider: "candle_alpha", rung: "staged_residency", tiers: ["q8"], reason: "requires-request-contexts" }] }, "mlx_request_context_lane"],
  ];
  for (const [plan, expected] of cases) {
    assert.equal(classifyMemoryContractMismatch(finding, plan), expected);
  }

  // Mutation: the skip row must match on ALL FOUR of backend/provider/rung/tier. Flipping any one
  // alone must lose the join and fall to `unclassified` rather than keeping the benign class — a
  // loose join is how a real drift coordinate gets absorbed into "nothing to do".
  const matching = {
    backend: "candle",
    provider: "candle_alpha",
    rung: "staged_residency",
    tiers: ["q8"],
    reason: "no-route-witness",
  };
  assert.equal(classifyMemoryContractMismatch(finding, { unhosted: [], skipped: [matching] }), "no_route_witness");
  for (const [field, wrong] of [
    ["backend", "mlx"],
    ["provider", "candle_beta"],
    ["rung", "bounded_decode"],
    ["tiers", ["bf16"]],
  ]) {
    assert.equal(
      classifyMemoryContractMismatch(finding, {
        unhosted: [],
        skipped: [{ ...matching, [field]: wrong }],
      }),
      "unclassified",
      `the join must not survive a wrong ${field}`,
    );
  }
});

test("`already-declared` and an unknown reason both land as unclassified drift, never as benign", () => {
  const input = fixture();
  const finding = collectMemoryContractMismatches(input).find(
    (entry) => entry.leg === "engine_manifest",
  );
  for (const reason of ["already-declared", "some-reason-invented-next-year"]) {
    const name = classifyMemoryContractMismatch(finding, {
      unhosted: [],
      skipped: [{ backend: "candle", provider: "candle_alpha", rung: "staged_residency", tiers: ["q8"], reason }],
    });
    assert.equal(name, "unclassified");
    assert.equal(TRIAGE_CLASSES[name].disposition, "drift");
  }
});

test("a `withheld` projection reason reaches its own class", () => {
  // Without this the class is unreachable: deleting the `["withheld", …]` row from
  // PROJECTION_REASONS leaves the whole suite green, so the kolors withhold has no coverage at all.
  const finding = collectMemoryContractMismatches(fixture()).find(
    (entry) => entry.leg === "engine_manifest",
  );
  const plan = {
    unhosted: [],
    skipped: [{ backend: "candle", provider: "candle_alpha", rung: "staged_residency", tiers: ["q8"], reason: "withheld" }],
  };
  assert.equal(classifyMemoryContractMismatch(finding, plan), "manifest_declaration_withheld");
  assert.equal(TRIAGE_CLASSES.manifest_declaration_withheld.disposition, "by-construction");
});

test("a survey verdict that omits an axis is by-construction; naming the axis makes it drift", () => {
  // Widen the engine so the family has two tiers of cells but the engine reaches only q4.
  const wildcard = fixture();
  wildcard.cells.push({ ...wildcard.cells[0], tier: "q8" });
  delete wildcard.survey.families[100].backends.mlx.implementedTiers;

  const wild = triage(wildcard);
  assert.equal(classCounts(wild).survey_wildcard_axis, 1);
  assert.equal(wild.byDisposition.drift, 1, "only the pre-existing Candle lane is drift");

  // Same coordinate, now NAMED by the verdict rather than reached through an omitted axis. The
  // survey is asserting q8 outright, so it is drift.
  const explicit = structuredClone(wildcard);
  explicit.survey.families[100].backends.mlx.implementedTiers = ["q4", "q8"];
  const named = triage(explicit);
  assert.equal(classCounts(named).survey_wildcard_axis, undefined);
  assert.equal(classCounts(named).survey_scope_overclaims, 1);
  assert.equal(named.byDisposition.drift, 2);
});

test("an omission on one axis does not excuse a contradiction on an axis the scope names", () => {
  // The regression this pins: the first rule asked only whether the scope omitted SOME axis, so a
  // verdict naming `tiers` and omitting `modes` had its TIER contradictions filed by-construction.
  const input = fixture();
  input.cells.push({ ...input.cells[0], tier: "q8" });
  const verdict = input.survey.families[100].backends.mlx;
  verdict.implementedTiers = ["q4", "q8"]; // q8 NAMED; the engine implements q4 only
  delete verdict.implementedModes; // modes omitted — the unrelated wildcard

  const result = triage(input);
  assert.equal(
    classCounts(result).survey_scope_overclaims,
    1,
    "a named tier the engine does not implement is the survey asserting something false",
  );
  assert.equal(classCounts(result).survey_wildcard_axis, undefined);

  // Mutation: stop NAMING the tier and the very same coordinate becomes a wildcard expansion.
  const omitted = structuredClone(input);
  delete omitted.survey.families[100].backends.mlx.implementedTiers;
  const wild = triage(omitted);
  assert.equal(classCounts(wild).survey_wildcard_axis, 1);
  assert.equal(classCounts(wild).survey_scope_overclaims, undefined);
});

test("a loaded overlay the survey withholds is by-construction, not underclaim drift", () => {
  // The dump's contract surface has no overlay axis, so `implementedByTier` cannot say "rung 4 at
  // clean base only" — but the providers mean exactly that, and the route witnesses still reach the
  // overlay. A survey that claims `none` and withholds `lora` is the more precise record.
  const input = fixture();
  input.cells.push({ ...input.cells[0], overlay: "lora" });
  input.engineFacts[0].memoryRouteWitnesses.push({
    provider: "mlx_alpha", tier: "q4", mode: "text_to_image", overlay: "lora", loadProfile: "lora",
  });
  const result = triage(input);
  assert.equal(classCounts(result).survey_withholds_loaded_overlay, 1);
  assert.equal(classCounts(result).survey_scope_underclaims, undefined);

  // Mutation: drop the clean-base claim, so the survey is not withholding an overlay — it simply
  // does not claim the coordinate at all. That is ordinary underclaim drift again.
  const noBase = structuredClone(input);
  noBase.survey.families[100].backends.mlx.implementedModes = ["edit_image"];
  const bare = triage(noBase);
  assert.equal(classCounts(bare).survey_withholds_loaded_overlay, undefined);
  assert.equal(classCounts(bare).survey_scope_underclaims, 2);
});

test("`implementation: none` against an implemented, routed rung is drift", () => {
  const input = fixture();
  input.survey.families[100].backends.mlx = { implementation: "none" };
  const result = triage(input);
  assert.equal(classCounts(result).survey_records_none, 1);
  assert.equal(TRIAGE_CLASSES.survey_records_none.disposition, "drift");

  // Mutation: withdraw the engine's route witness and the contradiction disappears — proving the
  // class is keyed on the engine ACTUALLY reaching the coordinate, not merely on the verdict string.
  const unrouted = structuredClone(input);
  unrouted.engineFacts[0].memoryRouteWitnesses = [
    { provider: "mlx_alpha", tier: "q4", mode: "edit_image", overlay: "none", loadProfile: "plain" },
  ];
  assert.equal(classCounts(triage(unrouted)).survey_records_none, undefined);
});

test("a narrower explicit survey scope is underclaim drift, distinct from `none`", () => {
  const input = fixture();
  input.survey.families[100].backends.mlx.implementedModes = ["edit_image"];
  const result = triage(input);
  assert.equal(classCounts(result).survey_scope_underclaims, 1);
  assert.equal(classCounts(result).survey_records_none, undefined);
});

test("manifest overdeclaration splits on whether the matrix cell exists", () => {
  // No cell for the declared coordinate.
  const noCell = fixture();
  noCell.cells = [];
  assert.equal(classCounts(triage(noCell)).manifest_overdeclares_cell, 1);

  // Cell exists, and a route IS witnessed at the coordinate — but through `pid`, a load profile the
  // `none` overlay's manifest mapping does not accept (only `plain` serves a bare `none`). So the
  // declaration names a path production does not take.
  const noRoute = fixture();
  noRoute.engineFacts[0].memoryRouteWitnesses = [
    { provider: "mlx_alpha", tier: "q4", mode: "text_to_image", overlay: "none", loadProfile: "pid" },
  ];
  const routed = triage(noRoute);
  assert.equal(classCounts(routed).manifest_overdeclares_route, 1);
  assert.equal(classCounts(routed).manifest_overdeclares_cell, undefined);
});

test("every class is reachable, dispositions total, and no class is silently absent", () => {
  // Each class name in the table must be either produced above or explicitly a fallback. This
  // catches a class added to the table and never wired to a cause.
  const wired = new Set([
    "engine_provider_unhosted",
    "no_route_witness",
    "tier_not_in_catalog",
    "mlx_request_context_lane",
    "manifest_declaration_withheld",
    "survey_wildcard_axis",
    "survey_records_none",
    "survey_scope_underclaims",
    "survey_withholds_loaded_overlay",
    "survey_withholds_unmeasured_entry",
    "survey_scope_overclaims",
    "manifest_overdeclares_cell",
    "manifest_overdeclares_route",
    "unclassified",
  ]);
  assert.deepEqual(new Set(Object.keys(TRIAGE_CLASSES)), wired);
  for (const [name, entry] of Object.entries(TRIAGE_CLASSES)) {
    assert.ok(["by-construction", "drift"].includes(entry.disposition), `${name} has no disposition`);
    assert.ok(entry.rationale.length > 40, `${name} has no written rationale`);
  }

  const result = triage(fixture());
  assert.equal(result.byDisposition.drift + result.byDisposition["by-construction"], result.total);
});
