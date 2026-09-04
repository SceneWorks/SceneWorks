import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { readPlanAndLock, validateMetricsLock, validatePlan } from "./starvector-terminal-campaign.mjs";

const plan = JSON.parse(readFileSync("release/starvector-terminal-campaign-v1.json"));
const lock = JSON.parse(readFileSync("release/starvector-terminal-metrics-lock-v1.json"));

test("terminal campaign is fixed, serial, and fail closed", async () => {
  const validated = validatePlan(plan);
  assert.equal(validated.inference_contract.revision, "c6d6a4dbd61ab09c26ff5526632cae2cefea60ed");
  assert.deepEqual(validated.inference_preflight, {
    workflow_run_id: "33851645747",
    workflow_run_attempt: 1,
    head_sha: "c6d6a4dbd61ab09c26ff5526632cae2cefea60ed",
    artifact: {
      id: 9928624696,
      name: "starvector-terminal-preflight-c6d6a4dbd61ab09c26ff5526632cae2cefea60ed-33851645747-1",
      digest: "sha256:4df39fc45d36ef11f968aa82c48eda6292f48c54086a4beee4ff3f6e8ba48226",
    },
    inventory_artifacts: [
      { tier: "1b", path: "inventory/starvector-1b-inventory.json", sha256: "f4b8345ae7b6aa535080191c05694bba68fb3bbfe0391ff95f5bfd9b381812da" },
      { tier: "8b", path: "inventory/starvector-8b-inventory.json", sha256: "af1bcb4c38b86bbe1a973aedcba2ca72b03485d2c5877457192878beeb5989a2" },
    ],
    hook_logs: [
      { backend: "mlx", tier: "1b", path: "hooks/mlx-starvector-1b.log", sha256: "4c06aa10dfa65cf575e741df0632baa03970a277485eb93ca69dca3070d3828b" },
      { backend: "mlx", tier: "8b", path: "hooks/mlx-starvector-8b.log", sha256: "4c2e426c53ac0728fa20765dbedbac1440766acc4b52670acdf0e7c574bb2da3" },
      { backend: "candle-cuda", tier: "1b", path: "hooks/candle-cuda-starvector-1b.log", sha256: "c55e5ed6d44c670ccdb06315106e136e769b67ce16de3a43a1260c7f75e47d03" },
      { backend: "candle-cuda", tier: "8b", path: "hooks/candle-cuda-starvector-8b.log", sha256: "0f34fe938c1b9491a8380b5b27583735546b1afd39f1c21abdaedb09550bddc4" },
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

test("terminal campaign rejects every sealed native-preflight provenance mutation", () => {
  for (const [label, mutate] of [
    ["run", (value) => { value.inference_preflight.workflow_run_id = "33851645748"; }],
    ["attempt", (value) => { value.inference_preflight.workflow_run_attempt = 2; }],
    ["head", (value) => { value.inference_preflight.head_sha = "0".repeat(40); }],
    ["artifact id", (value) => { value.inference_preflight.artifact.id += 1; }],
    ["artifact name", (value) => { value.inference_preflight.artifact.name += "-other"; }],
    ["artifact digest", (value) => { value.inference_preflight.artifact.digest = `sha256:${"0".repeat(64)}`; }],
    ["inventory", (value) => { value.inference_preflight.inventory_artifacts[0].sha256 = "0".repeat(64); }],
    ["hook", (value) => { value.inference_preflight.hook_logs[0].sha256 = "0".repeat(64); }],
  ]) {
    const changed = structuredClone(plan);
    mutate(changed);
    assert.throws(() => validatePlan(changed), /preflight provenance/, label);
  }
});
