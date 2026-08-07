import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { stripJsoncComments } from "./lib/jsonc.mjs";
import {
  evidenceSemantics, logicalCaseId, recordId, validateBundle, validateRecord,
} from "./memory-calibration-harness.mjs";
import {
  flux1CalibrationPlans, flux1EvidenceRecords, flux1SourceSessions, updateBundle,
} from "./sc-15823-flux1-evidence.mjs";

const INFERENCE_REVISION = "5f973a73bf00307240afd81d2778ba9d89349e51";
const SCENEWORKS_REVISION = "f936e7e6a17a0b592752f634d21db17f5e8f2db7";
const BOUNDED = new Set(["bounded_decode", "bounded_attention", "bounded_transformer_residency"]);
const EXPECTED = {
  flux1_schnell: {
    revision: "bba3ae01dfd94089f173c05edd4e1a4c551f2599",
    inventory: "3157f5cdd80246daf0dd5f7c07694e8e8ee2845ec01bec8b9edb8a02b4bd8f62",
    maximum: 29, mean: 0.2289120356241862, rmse: 0.5452312653086621,
  },
  flux1_dev: {
    revision: "323fd12d79f78ad444e882e8d8e871914584f2b9",
    inventory: "38892dfdd0177068e4996834a3ef6666309db5a0034f2548bd13f814e742b341",
    maximum: 21, mean: 0.26471201578776044, rmse: 0.5180919471938068,
  },
};

test("SC-15823 emits ten current, exact base-only FLUX.1 records", () => {
  const records = flux1EvidenceRecords();
  assert.equal(records.length, 10);
  for (const record of records) {
    validateRecord(record);
    assert.equal(record.status, "runtime_complete");
    assert.equal(record.repositories.sceneWorks.revision, SCENEWORKS_REVISION);
    assert.equal(record.repositories.inference.revision, INFERENCE_REVISION);
    assert.equal(record.target.overlay, "none");
    assert.equal(record.target.mode, "text_to_image");
    assert.equal(record.target.tier, "q4");
    assert.deepEqual(record.target.geometry, { width: 1024, height: 1024, batch: 1, frames: 1 });
    // sc-17774: currency is the lane's compile closure, not the pin. Feeding the record's own
    // captured digest back as the live one is what "nothing has moved" means under the new policy.
    assert.equal(
      evidenceSemantics(record, {
        inference: INFERENCE_REVISION,
        inferenceClosureDigests: {
          [`${record.backend}:${record.target.provider}`]:
            record.repositories.inference.closureDigest,
        },
      }),
      "current",
    );
    assert.equal(record.artifact.resolvedRevision, EXPECTED[record.target.provider].revision);
    assert.equal(record.artifact.inventorySha256, EXPECTED[record.target.provider].inventory);
    assert.equal(record.scenarios.find(({ name }) => name === "overlay").result, "not_applicable");
    for (const name of ["warm_repeat", "cancel", "error"]) {
      assert.equal(record.scenarios.find((scenario) => scenario.name === name).result, "not_run");
    }
    assert.equal(record.negativeMutation, null);
  }
  assert.deepEqual(
    Object.fromEntries(Object.keys(EXPECTED).map((provider) => [
      provider, records.filter((record) => record.target.provider === provider).map((record) => record.strategy.rung).sort(),
    ])),
    Object.fromEntries(Object.keys(EXPECTED).map((provider) => [provider, [
      "bounded_attention", "bounded_decode", "bounded_transformer_residency", "resident", "staged_residency",
    ]])),
  );
});

test("SC-15823 preserves exact staged parity and measured bounded tolerances", () => {
  for (const record of flux1EvidenceRecords()) {
    const expected = EXPECTED[record.target.provider];
    if (BOUNDED.has(record.strategy.rung)) {
      assert.deepEqual(
        [record.quality.maximumError, record.quality.meanError, record.quality.rootMeanSquareError],
        [expected.maximum, expected.mean, expected.rmse],
      );
      assert.deepEqual(
        [record.quality.maximumErrorThreshold, record.quality.meanErrorThreshold, record.quality.rootMeanSquareErrorThreshold],
        [expected.maximum, expected.mean, expected.rmse],
      );
    } else {
      assert.deepEqual(
        [record.quality.maximumError, record.quality.meanError, record.quality.rootMeanSquareError],
        [0, 0, 0],
      );
      assert.deepEqual(
        [record.quality.maximumErrorThreshold, record.quality.meanErrorThreshold, record.quality.rootMeanSquareErrorThreshold],
        [0, 0, 0],
      );
    }
  }
});

test("SC-15823 plan declares the same ten exact compositions", () => {
  const plans = flux1CalibrationPlans();
  const records = flux1EvidenceRecords();
  assert.equal(plans.length, 10);
  for (const plan of plans) {
    const record = records.find((candidate) =>
      candidate.target.provider === plan.target.provider && candidate.strategy.rung === plan.rung,
    );
    assert.ok(record);
    assert.deepEqual(plan.engagedRungs, record.strategy.engagedRungs);
    assert.deepEqual(plan.cases, [{ parameters: record.strategy.parameters, expectedResult: "passed" }]);
    assert.equal(plan.fixture, record.fixture);
  }
});

test("SC-15823 replaces only its exact provider records", async () => {
  const existing = JSON.parse(await readFile(new URL("../docs/generated/memory-calibration-evidence.json", import.meta.url)));
  const updated = updateBundle(existing);
  assert.equal(updated.records.length, existing.records.length);
  assert.equal(updated.records.filter((record) => Object.hasOwn(EXPECTED, record.target.provider)).length, 10);
  const existingRefreshSessions = (existing.sourceSessions ?? []).filter(
    (session) => session.sourcePath.startsWith("docs/calibration/sc-15823-refresh-"),
  );
  assert.equal(
    updated.sourceSessions.length,
    (existing.sourceSessions ?? []).length - existingRefreshSessions.length + 10,
  );
  assert.equal(
    updated.sourceSessions.filter((session) => session.sourcePath.startsWith("docs/calibration/sc-15823-refresh-")).length,
    10,
  );
  assert.deepEqual(
    updated.sourceSessions.filter((session) => !session.sourcePath.startsWith("docs/calibration/sc-15823-refresh-")),
    (existing.sourceSessions ?? []).filter(
      (session) => !session.sourcePath.startsWith("docs/calibration/sc-15823-refresh-"),
    ),
    "PuLID and every unrelated source session are preserved exactly",
  );
});

test("SC-15823 source sessions bind the ten immutable current-pin logs", async () => {
  const sessions = flux1SourceSessions();
  assert.equal(sessions.length, 10);
  for (const session of sessions) {
    const bytes = await readFile(new URL(`../${session.sourcePath}`, import.meta.url));
    assert.equal(createHash("sha256").update(bytes).digest("hex"), session.stdoutSha256);
    assert.equal(session.repositories.sceneWorks.revision, SCENEWORKS_REVISION);
    assert.equal(session.repositories.inference.revision, INFERENCE_REVISION);
    assert.deepEqual(session.claims, ["memory", "loadability", "overlay"]);
    assert.equal(session.inputs.length, 1);
    assert.equal(session.inputs[0].variant, "q4");
  }
});

test("FLUX.1 manifest has ten base bindings and zero overlay bindings", async () => {
  const source = await readFile(new URL("../config/manifests/builtin.models.jsonc", import.meta.url), "utf8");
  const manifest = JSON.parse(stripJsoncComments(source));
  const models = manifest.models.filter(({ id }) => ["flux_schnell", "flux_dev"].includes(id));
  assert.equal(models.length, 2);
  const bindings = models.flatMap((model) => model.candle.calibrations ?? []);
  assert.equal(bindings.length, 10);
  assert.equal(bindings.filter(({ overlay }) => overlay !== "none").length, 0);
  for (const model of models) {
    const expectedProvider = model.id === "flux_schnell" ? "flux1_schnell" : "flux1_dev";
    assert.deepEqual(
      model.candle.calibrations.map(({ rung }) => rung).sort(),
      ["bounded_attention", "bounded_decode", "bounded_transformer_residency", "resident", "staged_residency"],
    );
    for (const binding of model.candle.calibrations) {
      assert.equal(binding.provider, expectedProvider);
      assert.equal(binding.inferenceRevision, INFERENCE_REVISION);
      assert.equal(binding.sceneWorksRevision, SCENEWORKS_REVISION);
      assert.equal(binding.overlay, "none");
      assert.equal(binding.artifactResolvedRevision, EXPECTED[expectedProvider].revision);
    }
  }
});

test("generated matrix retains superseded runtime evidence without authorizing the new pin", async () => {
  const matrix = JSON.parse(await readFile(new URL("../docs/generated/memory-matrix.json", import.meta.url)));
  assert.equal(matrix.schemaVersion, 6);
  const runs = matrix.calibrationRuns.filter(({ record }) =>
    ["flux1_schnell", "flux1_dev"].includes(record.target.provider),
  );
  assert.equal(runs.length, 10);
  assert.deepEqual(
    runs.map(({ record }) => `${record.target.provider}:${record.strategy.rung}`).sort(),
    Object.keys(EXPECTED).flatMap((provider) => [
      "bounded_attention", "bounded_decode", "bounded_transformer_residency", "resident", "staged_residency",
    ].map((rung) => `${provider}:${rung}`)).sort(),
  );
  assert.ok(runs.every(({ record }) => record.status === "runtime_complete"));
  assert.ok(runs.every(({ semantics }) => semantics === "historical"));
  assert.ok(matrix.summary.calibrationRunsByStatus.complete > 0);
  const cells = matrix.cells.filter((cell) =>
    ["flux_schnell", "flux_dev"].includes(cell.modelId) &&
    cell.backend === "candle" && cell.tier === "q4" &&
    cell.mode === "text_to_image" && cell.overlay === "none",
  );
  assert.equal(cells.length, 10);
  assert.ok(cells.every((cell) => cell.state === "Implemented/unverified"));
  assert.equal(cells.filter((cell) => cell.state === "Verified").length, 0);
  assert.ok(cells.every((cell) =>
    cell.evidence.currentEnvironmentVerification.length === 0 &&
    cell.evidence.historicalVerification.some(
      (evidence) => evidence.recordStatus === "runtime_complete",
    ),
  ));
  const overlays = matrix.cells.filter((cell) =>
    ["flux_schnell", "flux_dev"].includes(cell.modelId) &&
    cell.backend === "candle" && cell.overlay !== "none",
  );
  assert.equal(overlays.filter((cell) => ["Runtime verified", "Verified"].includes(cell.state)).length, 0);

  const markdown = await readFile(new URL("../docs/generated/memory-matrix.md", import.meta.url), "utf8");
  assert.match(markdown, /`Runtime verified` means the exact base-only coordinate is production-admissible/);
  assert.match(markdown, /\| `flux_dev` \| candle .*\| Implemented\/unverified \|/);
  assert.match(markdown, /\| `flux_schnell` \| candle .*\| Implemented\/unverified \|/);
});

test("runtime_complete rejects overclaimed lifecycle or overlay coverage", () => {
  const record = structuredClone(flux1EvidenceRecords()[0]);
  record.scenarios.find(({ name }) => name === "warm_repeat").result = "passed";
  assert.throws(() => validateRecord(record), /warm_repeat must remain explicitly not_run/);

  const overlay = structuredClone(flux1EvidenceRecords()[0]);
  overlay.target.overlay = "identity";
  overlay.logicalCaseId = logicalCaseId(overlay);
  overlay.id = recordId(overlay);
  assert.throws(() => validateRecord(overlay), /base-only none overlay/);
});

test("runtime_complete accepts only its one exact passed strategy case", () => {
  const record = flux1EvidenceRecords()[0];
  const secondCase = structuredClone(record);
  secondCase.sweep.cases.push({ parameters: { unexpected: 1 }, result: "passed" });
  assert.throws(
    () => validateRecord(secondCase),
    /exactly one passed case matching its strategy parameters/,
  );
  assert.throws(
    () => validateBundle({
      schemaVersion: 4,
      harnessVersion: record.harnessVersion,
      sourceSessions: [],
      records: [secondCase],
    }),
    /schema validation failed: .*sweep\.cases: array has too many items/,
  );

  const mismatch = structuredClone(record);
  mismatch.sweep.cases[0].parameters = { unexpected: 1 };
  assert.throws(
    () => validateRecord(mismatch),
    /exactly one passed case matching its strategy parameters/,
  );
  assert.throws(
    () => validateBundle({
      schemaVersion: 4,
      harnessVersion: record.harnessVersion,
      sourceSessions: [],
      records: [mismatch],
    }),
    /exactly one passed case matching its strategy parameters/,
  );
});
