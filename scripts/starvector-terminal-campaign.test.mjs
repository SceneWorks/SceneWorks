import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { readPlanAndLock, validateMetricsLock, validatePlan, validateTerminalDispatchInputs } from "./starvector-terminal-campaign.mjs";

const plan = JSON.parse(readFileSync("release/starvector-terminal-campaign-v1.json"));
const lock = JSON.parse(readFileSync("release/starvector-terminal-metrics-lock-v1.json"));

test("terminal campaign is fixed, serial, and fail closed", async () => {
  const validated = validatePlan(plan);
  assert.equal(validated.inference_contract.revision, "42ab6f2b8b9815205bc215c6d19c2b7714c908fe");
  assert.deepEqual(validated.inference_preflight, {
    repository: "SceneWorks/inference",
    workflow: {
      id: 312370029,
      name: "Real-weight validation",
      path: ".github/workflows/real-weights.yml",
      event: "workflow_dispatch",
    },
    workflow_run_id: "33928871038",
    workflow_run_attempt: 1,
    head_sha: "42ab6f2b8b9815205bc215c6d19c2b7714c908fe",
    artifact: {
      id: 9957850431,
      name: "starvector-terminal-preflight-42ab6f2b8b9815205bc215c6d19c2b7714c908fe-33928871038-1",
      size_in_bytes: 6456,
      digest: "sha256:609a2850118c206a4a38698e1680e81b78666d6b72be07d66ba677b0f50a9831",
    },
    inventory_artifacts: [
      { tier: "1b", path: "inventory/starvector-1b-inventory.json", sha256: "f4b8345ae7b6aa535080191c05694bba68fb3bbfe0391ff95f5bfd9b381812da" },
      { tier: "8b", path: "inventory/starvector-8b-inventory.json", sha256: "af1bcb4c38b86bbe1a973aedcba2ca72b03485d2c5877457192878beeb5989a2" },
    ],
    hook_logs: [
      { backend: "mlx", tier: "1b", path: "hooks/mlx-starvector-1b.log", sha256: "61fa4d975619bf4fd5fe7935f328926ec1f1454f43b8fcce10c20cc6d7182f5d" },
      { backend: "mlx", tier: "8b", path: "hooks/mlx-starvector-8b.log", sha256: "63ddfbaf05a9f299de3e5688586e4e3ac2cdb91053131192771c4754d672bc7d" },
      { backend: "candle-cuda", tier: "1b", path: "hooks/candle-cuda-starvector-1b.log", sha256: "9dfec9f0651396e8e9d0e92cabde7ea9b0c6af393694647e7cc3fd8d855d7da6" },
      { backend: "candle-cuda", tier: "8b", path: "hooks/candle-cuda-starvector-8b.log", sha256: "86b069670b6515227a6e7a016f6c5bb3c27f701dee5b799d24f9a53c8feaf9b2" },
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
  const pin = "42ab6f2b8b9815205bc215c6d19c2b7714c908fe";
  assert.deepEqual(validateTerminalDispatchInputs(plan, pin, "campaign-33928871038"), {
    permanent_pin: pin,
    campaign_run_id: "campaign-33928871038",
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
