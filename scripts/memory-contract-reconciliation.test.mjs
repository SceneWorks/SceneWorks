import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import {
  collectMemoryContractMismatches,
  reconcileMemoryContracts,
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
    manifest: { models: [manifestModel("mlx"), manifestModel("candle")] },
    cells: ["mlx", "candle"].map((backend) => ({
      backend,
      provider: `${backend}_alpha`,
      modelId: `${backend}_model`,
      mode: "text_to_image",
      tier: "q4",
      overlay: "none",
      rung: "bounded_transformer_residency",
      owningFamilyStory: 100,
    })),
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
            mlx: { implementation: "provider-local" },
            candle: { implementation: "shared-primitive" },
          },
        },
      },
    },
    routeEligibility: ["mlx", "candle"].map((backend) => ({
      backend,
      provider: `${backend}_alpha`,
      mode: null,
      overlay: "none",
    })),
  };
  input.waiverLedger = { schemaVersion: 1, inferenceRevision: PIN, waivers: [] };
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

test("manifest to route is mutation-proven green-red-green", () => {
  greenRedGreen(
    (input) => input.routeEligibility.splice(0, 1),
    /unwaived mismatch/,
  );
  greenRedGreen(
    (input) => input.manifest.models[0].mlx.memoryStrategyContract.implementations[0].overlays = ["lora"],
    /unwaived mismatch/,
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

  const stale = fixture();
  stale.waiverLedger.waivers = structuredClone(input.waiverLedger.waivers);
  assert.throws(() => reconcileMemoryContracts(stale), /stale waiver/);
  assert.equal(reconcileMemoryContracts(fixture()).mismatches, 0);
});
