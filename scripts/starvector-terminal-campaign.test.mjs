import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { readPlanAndLock, validateMetricsLock, validatePlan, validateTerminalDispatchInputs } from "./starvector-terminal-campaign.mjs";

const plan = JSON.parse(readFileSync("release/starvector-terminal-campaign-v1.json"));
const lock = JSON.parse(readFileSync("release/starvector-terminal-metrics-lock-v1.json"));

test("terminal campaign is fixed, serial, and fail closed", async () => {
  const validated = validatePlan(plan);
  const workerManifest = readFileSync("crates/sceneworks-worker/Cargo.toml", "utf8");
  const nativePin = workerManifest.match(/sceneworks-gen-core = \{ git = "https:\/\/github.com\/SceneWorks\/inference", rev = "([a-f0-9]{40})"/)[1];
  assert.equal(validated.inference_contract.revision, nativePin, "campaign follows the actual shipping provider");
  assert.equal(validated.inference_preflight.head_sha, nativePin);
  assert.equal(validated.inference_preflight.artifact.name,
    `starvector-terminal-preflight-${nativePin}-${validated.inference_preflight.workflow_run_id}-${validated.inference_preflight.workflow_run_attempt}`);
  assert.match((await readPlanAndLock("release/starvector-terminal-campaign-v1.json")).metrics_lock_sha256, /^[0-9a-f]{64}$/);
});

test("terminal campaign rejects count, lock, and LPIPS weight drift", () => {
  const badPlan = structuredClone(plan); badPlan.counts.hostile_sanitizer = 199;
  assert.throws(() => validatePlan(badPlan), /counts/);
  const legacySchema = structuredClone(plan); legacySchema.inference_contract.receipt_schema = "release/starvector-terminal-receipt-v2.schema.json";
  assert.throws(() => validatePlan(legacySchema), /outcome-parity receipt schema/);
  const wrongSchemaHash = structuredClone(plan); wrongSchemaHash.inference_contract.receipt_schema_sha256 = "0".repeat(64);
  assert.throws(() => validatePlan(wrongSchemaHash), /outcome-parity receipt schema/);
  const badLock = structuredClone(lock); badLock.lpips.alexnet_weights_sha256 = "0".repeat(64);
  assert.throws(() => validateMetricsLock(badLock), /LPIPS/);
});

test("plan validates preflight structure while allowing newly recorded exact evidence", () => {
  const next = structuredClone(plan);
  next.inference_preflight.artifact.id += 1;
  next.inference_preflight.artifact.digest = `sha256:${"a".repeat(64)}`;
  assert.equal(validatePlan(next), next);
  for (const change of [p => p.inference_preflight.head_sha = "0".repeat(40), p => p.inference_preflight.artifact.digest = "invalid", p => p.inference_preflight.workflow_run_attempt = 0, p => p.inference_preflight.hook_logs.pop()]) {
    const invalid = structuredClone(plan); change(invalid); assert.throws(() => validatePlan(invalid), /preflight provenance/);
  }
});

test("dispatch identities reject Bash and PowerShell injection-shaped payloads", () => {
  const pin = plan.inference_contract.revision;
  assert.deepEqual(validateTerminalDispatchInputs(plan, pin, "campaign-safe-1"), {
    permanent_pin: pin,
    campaign_run_id: "campaign-safe-1",
  });
  for (const value of [
    "$(touch /tmp/starvector-shell-injection)",
    "campaign'; Write-Output pwned; #",
    'campaign\"; Start-Process calc; #',
    "campaign;rm",
  ]) {
    assert.throws(
      () => validateTerminalDispatchInputs(plan, pin, value),
      /bounded portable identifier/,
      value,
    );
  }
  assert.throws(
    () => validateTerminalDispatchInputs(plan, `$(printf ${pin})`, "campaign"),
    /permanent pin/,
  );
});
