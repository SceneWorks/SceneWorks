import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { readPlanAndLock, validateMetricsLock, validatePlan } from "./starvector-terminal-campaign.mjs";

const plan = JSON.parse(readFileSync("release/starvector-terminal-campaign-v1.json"));
const lock = JSON.parse(readFileSync("release/starvector-terminal-metrics-lock-v1.json"));

test("terminal campaign is fixed, serial, and fail closed", async () => {
  assert.equal(validatePlan(plan).inference_contract.revision, "c7f10e2191cc9df221c852cb5a32605fae9e94b2");
  assert.match((await readPlanAndLock("release/starvector-terminal-campaign-v1.json")).metrics_lock_sha256, /^[0-9a-f]{64}$/);
});

test("terminal campaign rejects count, lock, and LPIPS weight drift", () => {
  const badPlan = structuredClone(plan); badPlan.counts.hostile_sanitizer = 199;
  assert.throws(() => validatePlan(badPlan), /counts/);
  const badLock = structuredClone(lock); badLock.lpips.alexnet_weights_sha256 = "0".repeat(64);
  assert.throws(() => validateMetricsLock(badLock), /LPIPS/);
});
