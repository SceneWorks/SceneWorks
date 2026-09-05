import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { readPlanAndLock, validateMetricsLock, validatePlan, validateTerminalDispatchInputs } from "./starvector-terminal-campaign.mjs";

const plan = JSON.parse(readFileSync("release/starvector-terminal-campaign-v1.json"));
const lock = JSON.parse(readFileSync("release/starvector-terminal-metrics-lock-v1.json"));

test("terminal campaign is fixed, serial, and fail closed", async () => {
  const validated = validatePlan(plan);
  assert.equal(validated.inference_contract.revision, "1cd0e393863f7d3d880400e409519bcadfb43959");
  assert.deepEqual(validated.inference_preflight, {
    repository: "SceneWorks/inference",
    workflow: {
      id: 312370029,
      name: "Real-weight validation",
      path: ".github/workflows/real-weights.yml",
      event: "workflow_dispatch",
    },
    workflow_run_id: "33999207731",
    workflow_run_attempt: 1,
    head_sha: "1cd0e393863f7d3d880400e409519bcadfb43959",
    artifact: {
      id: 9979002999,
      name: "starvector-terminal-preflight-1cd0e393863f7d3d880400e409519bcadfb43959-33999207731-1",
      size_in_bytes: 6355,
      digest: "sha256:c70a0ffe4a8951a7caf24f95e2cbfca1f3d71a295cb2196265a27eac483955f1",
    },
    inventory_artifacts: [
      { tier: "1b", path: "inventory/starvector-1b-inventory.json", sha256: "f4b8345ae7b6aa535080191c05694bba68fb3bbfe0391ff95f5bfd9b381812da" },
      { tier: "8b", path: "inventory/starvector-8b-inventory.json", sha256: "af1bcb4c38b86bbe1a973aedcba2ca72b03485d2c5877457192878beeb5989a2" },
    ],
    hook_logs: [
      { backend: "mlx", tier: "1b", path: "hooks/mlx-starvector-1b.log", sha256: "f5ca9ff987c3aa43d16c227025b5bc5ccd108b77ec4313b2d0d24157282871b0" },
      { backend: "mlx", tier: "8b", path: "hooks/mlx-starvector-8b.log", sha256: "b7dabfd4a0a65faf5420f18e6985b0fc4a5c9753c275d15cf105347dc0a8769c" },
      { backend: "candle-cuda", tier: "1b", path: "hooks/candle-cuda-starvector-1b.log", sha256: "0cc32c25bc9108b1595d15ab9878df48e855dc7f89f1f2a595a0dc812372d412" },
      { backend: "candle-cuda", tier: "8b", path: "hooks/candle-cuda-starvector-8b.log", sha256: "119a6380aa7af1fe796c14cc6c6b74e30fe6c97df4886db6cf839812ec0ba387" },
    ],
  });
  assert.match((await readPlanAndLock("release/starvector-terminal-campaign-v1.json")).metrics_lock_sha256, /^[0-9a-f]{64}$/);
});

test("terminal campaign rejects count, lock, and LPIPS weight drift", () => {
  const badPlan = structuredClone(plan); badPlan.counts.hostile_sanitizer = 199;
  assert.throws(() => validatePlan(badPlan), /counts/);
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
  const pin = "1cd0e393863f7d3d880400e409519bcadfb43959";
  assert.deepEqual(validateTerminalDispatchInputs(plan, pin, "campaign-33999207731"), {
    permanent_pin: pin,
    campaign_run_id: "campaign-33999207731",
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
