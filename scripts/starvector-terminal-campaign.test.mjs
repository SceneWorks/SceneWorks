import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { readPlanAndLock, validateMetricsLock, validatePlan, validateTerminalDispatchInputs } from "./starvector-terminal-campaign.mjs";

const plan = JSON.parse(readFileSync("release/starvector-terminal-campaign-v1.json"));
const lock = JSON.parse(readFileSync("release/starvector-terminal-metrics-lock-v1.json"));

test("terminal campaign is fixed, serial, and fail closed", async () => {
  const validated = validatePlan(plan);
  assert.equal(validated.inference_contract.revision, "e11fd9f0fd26a0eee3a0eb1f4ca7f81c32b5aeb8");
  assert.deepEqual(validated.inference_preflight, {
    repository: "SceneWorks/inference",
    workflow: {
      id: 312370029,
      name: "Real-weight validation",
      path: ".github/workflows/real-weights.yml",
      event: "workflow_dispatch",
    },
    workflow_run_id: "34478405706",
    workflow_run_attempt: 1,
    head_sha: "e11fd9f0fd26a0eee3a0eb1f4ca7f81c32b5aeb8",
    artifact: {
      id: 10152562888,
      name: "starvector-terminal-preflight-e11fd9f0fd26a0eee3a0eb1f4ca7f81c32b5aeb8-34478405706-1",
      size_in_bytes: 6275,
      digest: "sha256:c8c50df391786aae4a2cf96cb1f41b3c4c296348295481f22c6423474bd08f69",
    },
    inventory_artifacts: [
      { tier: "1b", path: "inventory/starvector-1b-inventory.json", sha256: "f4b8345ae7b6aa535080191c05694bba68fb3bbfe0391ff95f5bfd9b381812da" },
      { tier: "8b", path: "inventory/starvector-8b-inventory.json", sha256: "af1bcb4c38b86bbe1a973aedcba2ca72b03485d2c5877457192878beeb5989a2" },
    ],
    hook_logs: [
      { backend: "mlx", tier: "1b", path: "hooks/mlx-starvector-1b.log", sha256: "8947c8dc1c87103e991c4af08e9b40827ce43cea24a2fddd0d39a74b0c05b545" },
      { backend: "mlx", tier: "8b", path: "hooks/mlx-starvector-8b.log", sha256: "811f42c6dc9d2c2e5c82bbc1b25e7aa4b05e6577ea1555ec90f053e9e920b1ce" },
      { backend: "candle-cuda", tier: "1b", path: "hooks/candle-cuda-starvector-1b.log", sha256: "ffc32559c2f2a645d5c5f419c3fe621892893dc1a704d81c36e596fd786b3ad5" },
      { backend: "candle-cuda", tier: "8b", path: "hooks/candle-cuda-starvector-8b.log", sha256: "75f68f0dd1a805ad373629dc4af456b34777f7685dcda1ff0784a78bfd1e7159" },
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
  const pin = "e11fd9f0fd26a0eee3a0eb1f4ca7f81c32b5aeb8";
  assert.deepEqual(validateTerminalDispatchInputs(plan, pin, "campaign-34478405706"), {
    permanent_pin: pin,
    campaign_run_id: "campaign-34478405706",
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
