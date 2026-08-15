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
    calibrationPlan: {
      providers: ["mlx", "candle"].map((backend) => ({
        name: `${backend}-alpha`,
        backend,
        target: {
          provider: `${backend}_alpha`,
          modelId: `${backend}_model`,
          mode: "text_to_image",
        },
      })),
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
  input.waiverLedger = {
    $schema: "../packages/schemas/memory-contract-reconciliation-waivers.schema.json",
    schemaVersion: 1,
    inferenceRevision: PIN,
    waivers: [],
  };
  return input;
}

function greenRedGreen(mutator, pattern) {
  const clean = fixture();
  assert.equal(reconcileMemoryContracts(clean).mismatches, 0);
  const mutated = structuredClone(clean);
  mutator(mutated);
  assert.throws(() => reconcileMemoryContracts(mutated), pattern);
  assert.equal(reconcileMemoryContracts(fixture()).mismatches, 0);
}

test("engine to manifest is mutation-proven green-red-green", () => {
  greenRedGreen(
    (input) => delete input.manifest.models[0].mlx.memoryStrategyContract,
    /unwaived mismatch/,
  );
  greenRedGreen(
    (input) => input.manifest.models[0].mlx.memoryStrategyContract.implementations[0].tiers = ["q8"],
    /unwaived mismatch/,
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
  greenRedGreen(
    (input) => input.engineFacts[0].memoryRouteWitnesses.splice(0, 1),
    /no memoryRouteWitnesses/,
  );
  greenRedGreen(
    (input) => input.manifest.models[0].mlx.memoryStrategyContract.implementations[0].overlays = ["lora"],
    /unwaived mismatch/,
  );
  greenRedGreen(
    (input) => input.engineFacts[0].memoryRouteWitnesses[0].tier = "q8",
    /unwaived mismatch/,
  );
  greenRedGreen(
    (input) => input.engineFacts[0].memoryRouteWitnesses[0].mode = "edit_image",
    /unwaived mismatch/,
  );
  greenRedGreen(
    (input) => input.engineFacts[0].memoryRouteWitnesses[0].overlay = "lora",
    /belongs to overlay none/,
  );
  greenRedGreen(
    (input) => input.engineFacts[0].memoryRouteWitnesses[0].loadProfile = "pid",
    /belongs to overlay none|unwaived mismatch/,
  );
  greenRedGreen(
    (input) => {
      input.engineFacts[0].memoryRouteWitnesses[0].overlay = "control";
      input.engineFacts[0].memoryRouteWitnesses[0].loadProfile = "multi_control";
      input.manifest.models[0].mlx.memoryStrategyContract.implementations[0].overlays = ["control"];
      input.cells[0].overlay = "control";
    },
    /unwaived mismatch/,
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
  input.calibrationPlan.providers[1].target.provider = "candle_control";
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
    assert.throws(() => reconcileMemoryContracts(mutated), /unwaived mismatch/);
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
  greenRedGreen(
    (input) => input.calibrationPlan.providers[0].target.provider = "renamed_provider",
    /unwaived mismatch/,
  );
  greenRedGreen(
    (input) => {
      input.closures.providers["mlx:renamed_provider"] = input.closures.providers["mlx:mlx_alpha"];
      delete input.closures.providers["mlx:mlx_alpha"];
    },
    /unwaived mismatch/,
  );
});

test("survey to engine is mutation-proven green-red-green", () => {
  greenRedGreen(
    (input) => input.survey.families[100].backends.mlx.implementation = "none",
    /unwaived mismatch/,
  );
  greenRedGreen(
    (input) => {
      const contractValue = input.engineFacts[0].memoryContracts[0];
      contractValue.surfaces[0].implementedRungs = ["resident"];
      recomputeDigest(contractValue);
    },
    /unwaived mismatch/,
  );
  greenRedGreen(
    (input) => input.survey.families[100].backends.mlx.implementedTiers = ["q8"],
    /unwaived mismatch/,
  );
  greenRedGreen(
    (input) => input.survey.families[100].backends.candle.implementationScopes[0].modes = ["edit_image"],
    /unwaived mismatch/,
  );
  greenRedGreen(
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
    /unwaived mismatch/,
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
    }],
  );
});

test("pin and duplicate-provider failures are mutation-proven", () => {
  greenRedGreen(
    (input) => input.engineFacts[0].generatedFrom.inferenceRevision = "f".repeat(40),
    /Cargo pins/,
  );
  greenRedGreen(
    (input) => input.engineFacts[0].memoryContracts.push(structuredClone(input.engineFacts[0].memoryContracts[0])),
    /duplicate mlx memory-contract provider/,
  );
  greenRedGreen(
    (input) => input.engineFacts[0].memoryContracts[0].selectorDigest = `sha256:${"f".repeat(64)}`,
    /selectorDigest does not bind/,
  );
});

test("under-keyed and stale waivers fail exactly", () => {
  const input = fixture();
  input.survey.families[100].backends.mlx.implementation = "none";
  const [entry] = collectMemoryContractMismatches(input);
  input.waiverLedger.waivers = [{
    ...entry,
    ownerStory: "sc-18460",
    reason: "Synthetic exact waiver for mutation proof.",
  }];
  assert.equal(reconcileMemoryContracts(input).mismatches, 1);

  const underKeyed = structuredClone(input);
  delete underKeyed.waiverLedger.waivers[0].mode;
  assert.throws(() => reconcileMemoryContracts(underKeyed), /under-keyed: missing mode/);

  const duplicate = structuredClone(input);
  duplicate.waiverLedger.waivers.push(structuredClone(duplicate.waiverLedger.waivers[0]));
  assert.throws(() => reconcileMemoryContracts(duplicate), /duplicate waiver/);

  const missingSchema = structuredClone(input);
  delete missingSchema.waiverLedger.$schema;
  assert.throws(() => reconcileMemoryContracts(missingSchema), /\$schema must be/);

  const extraTopLevel = structuredClone(input);
  extraTopLevel.waiverLedger.comment = "not in the closed schema";
  assert.throws(() => reconcileMemoryContracts(extraTopLevel), /unknown field comment/);

  const extraWaiverField = structuredClone(input);
  extraWaiverField.waiverLedger.waivers[0].comment = "not in the closed schema";
  assert.throws(() => reconcileMemoryContracts(extraWaiverField), /waiver 0 has unknown field comment/);

  const stale = fixture();
  stale.waiverLedger.waivers = structuredClone(input.waiverLedger.waivers);
  assert.throws(() => reconcileMemoryContracts(stale), /stale waiver/);
  assert.equal(reconcileMemoryContracts(fixture()).mismatches, 0);
});
