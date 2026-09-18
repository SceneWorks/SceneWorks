import assert from "node:assert/strict";
import test from "node:test";
import { LIMIT_CASE_SCENARIOS, materializeLimitCases, validateLimitCases } from "./starvector-terminal-limit-cases.mjs";

test("materializes six executable tier-specific limit scenarios", () => {
  for (const tuple of ["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"]) {
    const tier = tuple.split(":")[1], records = materializeLimitCases(tuple);
    assert.equal(validateLimitCases(records, tier), records);
    assert.deepEqual(records.map((record) => record.scenario), LIMIT_CASE_SCENARIOS);
    assert.deepEqual(records.map((record) => record.source_case_index), [12, 11, 11, 11, 11, 11]);
    assert.deepEqual(records.map((record) => record.case_index), [0, 1, 2, 3, 4, 5]);
    assert.deepEqual(records.map((record) => record.sampling), Array(6).fill({ temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1, seed: 7 }));
    assert.equal(records[1].detailBudget.maxNewTokens, 16);
    assert.equal(records[2].detailBudget.maxSvgBytes, 1024);
    assert.equal(records[3].detailBudget.maxWallTimeMs, 1000);
    assert.equal(records[0].detailBudget.maxNewTokens, tier === "1b" ? 7933 : 15422);
    assert.equal(records[0].detailBudget.maxWallTimeMs, 300000);
    assert.deepEqual(records.slice(4).map((record) => record.detailBudget), [records[0].detailBudget, records[0].detailBudget]);
    assert.equal(records[4].cancel_after_create, true);
    assert.equal(records[4].worker_unloaded, true);
    assert.equal(records[5].cancel_after_progress, true);
  }
});

test("rejects the old outcome-label-only limit index before execution", () => {
  const legacy = ["complete_root", "eos", "token_limit", "byte_limit", "wall_time_limit", "cancelled"]
    .map((finish_reason, case_index) => ({ case_id: `limit-${case_index}`, case_index, finish_reason }));
  assert.throws(() => validateLimitCases(legacy, "1b"), /label-only legacy scenario/);
});

test("rejects missing cancellation controls and non-executable budgets", () => {
  const missingControl = materializeLimitCases("mlx:1b");
  delete missingControl[4].worker_unloaded;
  assert.throws(() => validateLimitCases(missingControl, "1b"), /worker_unloaded/);
  const labelInsteadOfControl = materializeLimitCases("mlx:1b");
  labelInsteadOfControl[1].detailBudget.maxNewTokens = 7933;
  assert.throws(() => validateLimitCases(labelInsteadOfControl, "1b"), /does not execute token/);
});
