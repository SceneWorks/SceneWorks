import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import {
  collectMemoryContractMismatches,
  reconcileMemoryContracts,
  routeEligibilityFromEngineFacts,
} from "./lib/memory-contract-reconciliation.mjs";

const PIN = "a".repeat(40);

function contract(id) {
  return recomputeDigest({
    id,
    composed: false,
    selectorDigest: null,
    surfaces: [
      {
        selector: {
          tier: "q4",
          offloadPolicy: "sequential",
          loadShape: "deferred_materialization",
        },
        implementedRungs: ["resident", "bounded_transformer_residency"],
        structurallyNotApplicableRungs: [],
        deferredMaterializationRungs: ["bounded_transformer_residency"],
      },
    ],
  });
}

function recomputeDigest(contractValue) {
  contractValue.selectorDigest = `sha256:${createHash("sha256")
    .update(JSON.stringify(contractValue.surfaces))
    .digest("hex")}`;
  return contractValue;
}

function backendFacts(backend) {
  return {
    backend,
    generatedFrom: { inferenceRevision: PIN, dumper: "fixture" },
    engines: [{ id: `${backend}_alpha` }],
    memoryContracts: [contract(`${backend}_alpha`)],
    memoryRouteWitnesses: [{
      provider: `${backend}_alpha`,
      tier: "q4",
      mode: "text_to_image",
      overlay: "none",
      loadProfile: "plain",
    }],
    ...(backend === "candle" ? {
      bespokeMemoryRouteWaivers: [{
        providerId: "bespoke_identity",
        crateName: "identity",
        owner: "candle-gen-identity",
        reason: "Worker-owned path-shaped identity route with no Generator registration.",
        contractPath: "crates/media/candle-gen/candle-gen-identity/src/memory_strategy.rs",
        verificationPath: "crates/media/candle-gen/candle-gen-identity/src/identity.rs",
      }],
    } : {}),
  };
}

function manifestModel(backend) {
  return {
    id: `${backend}_model`,
    type: "image",
    [backend]: {
      memoryStrategyContract: {
        provider: `${backend}_alpha`,
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
  };
}

function fixture() {
  const input = {
    pin: PIN,
    engineFacts: [backendFacts("mlx"), backendFacts("candle")],
    manifest: {
      models: [
        manifestModel("mlx"),
        manifestModel("candle"),
        { id: "bespoke_model", type: "image", candle: { memoryStrategyCapabilities: {} } },
      ],
    },
    cells: [
      ...["mlx", "candle"].map((backend) => ({
        backend,
        provider: `${backend}_alpha`,
        modelId: `${backend}_model`,
        mode: "text_to_image",
        tier: "q4",
        overlay: "none",
        rung: "bounded_transformer_residency",
        owningFamilyStory: 100,
      })),
      {
        backend: "candle",
        provider: "bespoke_identity",
        modelId: "bespoke_model",
        mode: "character_image",
        tier: "q4",
        overlay: "identity",
        rung: "resident",
        owningFamilyStory: 200,
      },
    ],
    // sc-22514: the anchor plan, keyed `<modelId>:<tier>:<backend>`, one entry per cell.
    calibrationPlan: {
      anchors: Object.fromEntries(["mlx", "candle"].map((backend) => [
        `${backend}_model:q4:${backend}`,
        { provider: `${backend}_alpha`, mode: "text_to_image" },
      ])),
    },
    closures: {
      providers: {
        "mlx:mlx_alpha": {},
        "candle:candle_alpha": {},
      },
    },
    survey: {
      families: {
        100: {
          backends: {
            mlx: {
              implementation: "provider-local",
              implementedEntries: ["mlx_model"],
              implementedTiers: ["q4"],
              implementedModes: ["text_to_image"],
              implementedOverlays: ["none"],
            },
            candle: {
              implementation: "shared-primitive",
              implementationScopes: [{
                entries: ["candle_model"],
                tiers: ["q4"],
                modes: ["text_to_image"],
                overlays: ["none"],
              }],
            },
          },
        },
      },
    },
  };
  return input;
}

// The reconciliation is REPORT-ONLY (Michael, 2026-08-17): a disagreement is a FINDING, never a
// throw. The waiver ledger and its bijection are gone, so `/unwaived mismatch/` no longer exists as an
// error string. What these tests still prove — and what matters — is DETECTION: the clean fixture
// reconciles with zero findings, each mutation is noticed, and reverting clears it. That is the
// property the gate was only ever a delivery mechanism for.
function detectsMismatch(mutator) {
  const clean = fixture();
  assert.equal(reconcileMemoryContracts(clean).mismatches, 0);
  const mutated = structuredClone(clean);
  mutator(mutated);
  const result = reconcileMemoryContracts(mutated);
  assert.ok(
    result.mismatches > 0,
    "the mutation must be reported as a mismatch, not silently reconciled",
  );
  // A finding is not just a count: the report enumerates coordinates, so every finding must carry one.
  assert.equal(result.findings.length, result.mismatches);
  for (const finding of result.findings) {
    assert.ok(finding.leg && finding.direction, `finding is under-keyed: ${JSON.stringify(finding)}`);
  }
  assert.equal(reconcileMemoryContracts(fixture()).mismatches, 0);
}

// Structural malformation of the ENGINE FACTS is still an error rather than a finding: the dumps are
// generated input, and a malformed inventory means the report itself would be wrong. Nothing here can
// fail a build — `generate-memory-matrix.mjs` wraps the whole call in a report-only seam — but the lib
// must still refuse to reconcile against garbage instead of reporting a confident zero.
function rejectsStructurally(mutator, pattern) {
  const clean = fixture();
  assert.equal(reconcileMemoryContracts(clean).mismatches, 0);
  const mutated = structuredClone(clean);
  mutator(mutated);
  assert.throws(() => reconcileMemoryContracts(mutated), pattern);
  assert.equal(reconcileMemoryContracts(fixture()).mismatches, 0);
}

test("engine to manifest is mutation-proven green-red-green", () => {
  detectsMismatch(
    (input) => delete input.manifest.models[0].mlx.memoryStrategyContract,
  );
  detectsMismatch(
    (input) => input.manifest.models[0].mlx.memoryStrategyContract.implementations[0].tiers = ["q8"],
  );
});

test("typed bespoke Candle route waivers are exact, reachable, and cannot mask registrations", () => {
  const clean = fixture();
  assert.equal(reconcileMemoryContracts(clean).bespokeWaivers, 1);

  for (const mutate of [
    (input) => input.engineFacts[1].bespokeMemoryRouteWaivers.push(
      structuredClone(input.engineFacts[1].bespokeMemoryRouteWaivers[0]),
    ),
    (input) => input.engineFacts[1].bespokeMemoryRouteWaivers[0].providerId = "*",
    (input) => input.engineFacts[1].bespokeMemoryRouteWaivers[0].contractPath =
      "crates/media/candle-gen/other/src/memory_strategy.rs",
    (input) => input.cells.splice(input.cells.findIndex((cell) => cell.provider === "bespoke_identity"), 1),
    (input) => input.engineFacts[1].memoryContracts.push(contract("bespoke_identity")),
    (input) => input.engineFacts[1].memoryRouteWitnesses.push({
      provider: "bespoke_identity",
      tier: "q4",
      mode: "character_image",
      overlay: "identity",
      loadProfile: "identity",
    }),
  ]) {
    const mutated = structuredClone(clean);
    mutate(mutated);
    assert.throws(
      () => reconcileMemoryContracts(mutated),
      /duplicate bespoke|wildcard|invalid contractPath|stale bespoke|masks an ordinary/,
    );
  }

  const wrongBackend = structuredClone(clean);
  wrongBackend.engineFacts[0].bespokeMemoryRouteWaivers =
    wrongBackend.engineFacts[1].bespokeMemoryRouteWaivers;
  assert.throws(
    () => reconcileMemoryContracts(wrongBackend),
    /cannot publish Candle bespoke-memory waivers/,
  );
});

test("manifest to route is mutation-proven green-red-green", () => {
  // An absent witness inventory is malformed generated input, not a finding.
  rejectsStructurally(
    (input) => input.engineFacts[0].memoryRouteWitnesses.splice(0, 1),
    /no memoryRouteWitnesses/,
  );
  detectsMismatch(
    (input) => input.manifest.models[0].mlx.memoryStrategyContract.implementations[0].overlays = ["lora"],
  );
  detectsMismatch(
    (input) => input.engineFacts[0].memoryRouteWitnesses[0].tier = "q8",
  );
  detectsMismatch(
    (input) => input.engineFacts[0].memoryRouteWitnesses[0].mode = "edit_image",
  );
  // A witness whose overlay contradicts its own load profile is internally inconsistent generated
  // input — the dumper cannot emit it — so it stays an error rather than a finding.
  rejectsStructurally(
    (input) => input.engineFacts[0].memoryRouteWitnesses[0].overlay = "lora",
    /belongs to overlay none/,
  );
  // `pid` maps to the same `none` overlay the witness already carries, so this is internally
  // consistent input that simply no longer matches the manifest's declared profile set — a finding.
  detectsMismatch(
    (input) => input.engineFacts[0].memoryRouteWitnesses[0].loadProfile = "pid",
  );
  detectsMismatch(
    (input) => {
      input.engineFacts[0].memoryRouteWitnesses[0].overlay = "control";
      input.engineFacts[0].memoryRouteWitnesses[0].loadProfile = "multi_control";
      input.manifest.models[0].mlx.memoryStrategyContract.implementations[0].overlays = ["control"];
      input.cells[0].overlay = "control";
    },
  );
});

test("runtimeProvider is the exact composed provider identity", () => {
  const input = fixture();
  const candleFacts = input.engineFacts[1];
  const implementation = input.manifest.models[1].candle.memoryStrategyContract.implementations[0];
  implementation.runtimeProvider = "candle_control";
  candleFacts.engines[0].id = "candle_control";
  candleFacts.memoryContracts[0].id = "candle_control";
  candleFacts.memoryRouteWitnesses[0].provider = "candle_control";
  input.cells[1].provider = "candle_control";
  input.calibrationPlan.anchors["candle_model:q4:candle"].provider = "candle_control";
  input.closures.providers["candle:candle_control"] = input.closures.providers["candle:candle_alpha"];
  delete input.closures.providers["candle:candle_alpha"];
  assert.equal(reconcileMemoryContracts(input).mismatches, 0);

  for (const runtimeProvider of [undefined, "candle_alpha", "crossed_control"]) {
    const mutated = structuredClone(input);
    if (runtimeProvider === undefined) {
      delete mutated.manifest.models[1].candle.memoryStrategyContract.implementations[0]
        .runtimeProvider;
    } else {
      mutated.manifest.models[1].candle.memoryStrategyContract.implementations[0]
        .runtimeProvider = runtimeProvider;
    }
    assert.ok(
      reconcileMemoryContracts(mutated).mismatches > 0,
      "a crossed or missing runtimeProvider identity must be reported",
    );
  }
});

test("route facts are independent of syntax-equivalent Rust source", () => {
  const facts = fixture().engineFacts;
  const blockForm = "if eligible { spec.with_load_shape(deferred) } else { spec }";
  const expressionForm = "eligible.then(|| spec.with_load_shape(deferred)).unwrap_or(spec)";
  assert.notEqual(blockForm, expressionForm);
  assert.deepEqual(
    routeEligibilityFromEngineFacts(facts, blockForm),
    routeEligibilityFromEngineFacts(facts, expressionForm),
  );
});

test("plan and closure to engine is mutation-proven green-red-green", () => {
  detectsMismatch(
    (input) => input.calibrationPlan.anchors["mlx_model:q4:mlx"].provider = "renamed_provider",
  );
  detectsMismatch(
    (input) => {
      input.closures.providers["mlx:renamed_provider"] = input.closures.providers["mlx:mlx_alpha"];
      delete input.closures.providers["mlx:mlx_alpha"];
    },
  );
});

test("survey to engine is mutation-proven green-red-green", () => {
  detectsMismatch(
    (input) => input.survey.families[100].backends.mlx.implementation = "none",
  );
  detectsMismatch(
    (input) => {
      const contractValue = input.engineFacts[0].memoryContracts[0];
      contractValue.surfaces[0].implementedRungs = ["resident"];
      recomputeDigest(contractValue);
    },
  );
  detectsMismatch(
    (input) => input.survey.families[100].backends.mlx.implementedTiers = ["q8"],
  );
  detectsMismatch(
    (input) => input.survey.families[100].backends.candle.implementationScopes[0].modes = ["edit_image"],
  );
  detectsMismatch(
    (input) => {
      input.cells.push({
        ...input.cells[1],
        tier: "q8",
      });
      input.survey.families[100].backends.candle.implementationScopes.push({
        entries: ["candle_model"],
        tiers: ["q8"],
        modes: ["text_to_image"],
        overlays: ["none"],
      });
    },
  );
});

test("survey and engine scope mismatches carry exact coordinates in both directions", () => {
  const missingEngine = fixture();
  const contractValue = missingEngine.engineFacts[0].memoryContracts[0];
  contractValue.surfaces[0].implementedRungs = ["resident"];
  recomputeDigest(contractValue);
  assert.deepEqual(
    collectMemoryContractMismatches(missingEngine).filter((row) => row.leg === "survey_engine"),
    [{
      leg: "survey_engine",
      direction: "survey_to_engine",
      backend: "mlx",
      provider: "mlx_alpha",
      modelId: "mlx_model",
      familyStory: 100,
      mode: "text_to_image",
      tier: "q4",
      overlay: "none",
      rung: "bounded_transformer_residency",
      selectorDigest: contractValue.selectorDigest,
      // sc-21505: the survey names every axis here, so the over-claim is an assertion it made
      // outright rather than a wildcard expansion. `memory-contract-triage.mjs` reads this.
      cause: "survey_scope_overclaims",
    }],
  );

  const missingSurvey = fixture();
  missingSurvey.survey.families[100].backends.candle.implementationScopes = [];
  assert.deepEqual(
    collectMemoryContractMismatches(missingSurvey).filter((row) => row.leg === "survey_engine"),
    [{
      leg: "survey_engine",
      direction: "engine_to_survey",
      backend: "candle",
      provider: "candle_alpha",
      modelId: "candle_model",
      familyStory: 100,
      mode: "text_to_image",
      tier: "q4",
      overlay: "none",
      rung: "bounded_transformer_residency",
      selectorDigest: missingSurvey.engineFacts[1].memoryContracts[0].selectorDigest,
      cause: "survey_scope_underclaims",
    }],
  );
});

test("a loaded overlay withheld at a different mode is still a withheld overlay, not an underclaim", () => {
  // sc-21510: FLUX.1's identity coordinates ride `character_image` while the survey's clean-base
  // claim rides `text_to_image`. Rung capability is a property of the loaded overlay spec, not the
  // request mode, so the mode difference must not demote the withhold to an underclaim.
  const input = fixture();
  input.cells.push({ ...input.cells[0], mode: "character_image", overlay: "identity" });
  input.engineFacts[0].memoryRouteWitnesses.push({
    provider: "mlx_alpha",
    tier: "q4",
    mode: "character_image",
    overlay: "identity",
    loadProfile: "ip_adapter",
  });
  const rows = collectMemoryContractMismatches(input).filter((row) => row.leg === "survey_engine");
  assert.deepEqual(
    rows.map((row) => [row.mode, row.overlay, row.cause]),
    [["character_image", "identity", "survey_withholds_loaded_overlay"]],
  );

  // Mutation: drop the clean-base cell (and with it the survey's clean-base claim), and the same
  // coordinate is an ordinary underclaim again — the relaxation must not fire without a real claim.
  const noBase = structuredClone(input);
  noBase.cells.splice(0, 1);
  const bare = collectMemoryContractMismatches(noBase).filter((row) => row.leg === "survey_engine");
  assert.deepEqual(
    bare.map((row) => [row.mode, row.overlay, row.cause]),
    [["character_image", "identity", "survey_scope_underclaims"]],
  );
});

test("a coordinate the request-peak record marks unmeasured is a recorded withhold", () => {
  // sc-21510: SC-15525 marks illustrious entries `unmeasured` in `requestPeak.scopes`; publishing
  // the coordinates would overturn that record, so the absence is a verdict rather than drift.
  const input = fixture();
  const verdict = input.survey.families[100].backends.mlx;
  verdict.implementedEntries = [];
  verdict.requestPeak = {
    finding: "moves",
    scopes: [{ entries: ["mlx_model"], tiers: ["q4"], finding: "unmeasured" }],
  };
  const rows = collectMemoryContractMismatches(input).filter((row) => row.leg === "survey_engine");
  assert.deepEqual(
    rows.map((row) => [row.modelId, row.cause]),
    [["mlx_model", "survey_withholds_unmeasured_entry"]],
  );

  // Mutation 1: the scope names a different tier, so it says nothing about this coordinate.
  const wrongTier = structuredClone(input);
  wrongTier.survey.families[100].backends.mlx.requestPeak.scopes[0].tiers = ["q8"];
  assert.deepEqual(
    collectMemoryContractMismatches(wrongTier)
      .filter((row) => row.leg === "survey_engine")
      .map((row) => row.cause),
    ["survey_scope_underclaims"],
  );

  // Mutation 2: no scope at all — a bare unmeasured finding is not a per-coordinate record.
  const noScope = structuredClone(input);
  noScope.survey.families[100].backends.mlx.requestPeak = { finding: "unmeasured" };
  assert.deepEqual(
    collectMemoryContractMismatches(noScope)
      .filter((row) => row.leg === "survey_engine")
      .map((row) => row.cause),
    ["survey_scope_underclaims"],
  );
});

test("independent valid revision labels do not invalidate capability content", () => {
  const input = fixture();
  input.engineFacts[0].generatedFrom.inferenceRevision = "f".repeat(40);
  assert.equal(reconcileMemoryContracts(input).mismatches, 0);
});

test("malformed revisions and duplicate-provider failures are mutation-proven", () => {
  rejectsStructurally(
    (input) => input.engineFacts[0].generatedFrom.inferenceRevision = "not-a-sha",
    /no valid inference revision/,
  );
  rejectsStructurally(
    (input) => input.engineFacts[0].memoryContracts.push(structuredClone(input.engineFacts[0].memoryContracts[0])),
    /duplicate mlx memory-contract provider/,
  );
  rejectsStructurally(
    (input) => input.engineFacts[0].memoryContracts[0].selectorDigest = `sha256:${"f".repeat(64)}`,
    /selectorDigest does not bind/,
  );
});
