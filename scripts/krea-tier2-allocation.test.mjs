import assert from "node:assert/strict";
import test from "node:test";
import { childEnvironments, dispatchOptions, exitCodeFor, readPatternEvidence, resolveConfig } from "./tier2-ab.mjs";
import { KREA_ALLOCATION_POOL_BYTES, KREA_WIDE_IM2COL_PIN, kreaAllocationReceipt } from "./krea-tier2-allocation.mjs";
import { kreaWindowsMemoryFixture } from "./lib/krea-windows-worker-memory-fixture.mjs";
import { CHECKPOINT, KREA_BASE_APP, KREA_SUITE, kreaAssetInventory, kreaExpectedStepEvents, kreaRoutes } from "./krea-tier2-proof.mjs";

const SHA = "1".repeat(40);
function dispatch(backend, options, pin = KREA_WIDE_IM2COL_PIN) {
  return dispatchOptions({ profile: "tier2-ab", ref: SHA, inference_sha: pin, download_missing: true, backend,
    moss_options_json: JSON.stringify({ suite: KREA_SUITE, head_sha: SHA, preflight_only: false, ...options }) }, SHA);
}

test("the opt-in Candle allocation needs one linked Hires route and no Raw weights", async () => {
  assert.equal(dispatch("candle", { allocation: true }).krea_allocation, true);
  assert.equal(dispatch("candle", {}).krea_allocation, false);
  assert.throws(() => dispatch("mlx", { allocation: true }), /requires Candle/);
  assert.throws(() => dispatch("candle", { allocation: true, performance: true }), /separate from performance/);
  assert.throws(() => dispatch("candle", { allocation: true }, "2".repeat(40)), /reviewed CUDA im2col fix/);
  const env = { TIER2_AB_SUITE: KREA_SUITE, TIER2_AB_BACKEND: "cuda", TIER2_AB_BASE_SHA: KREA_BASE_APP, TIER2_AB_HEAD_SHA: SHA, TIER2_AB_INFERENCE_SHA: KREA_WIDE_IM2COL_PIN, TIER2_AB_BASE_DIR: "E:\\base", TIER2_AB_HEAD_DIR: "E:\\head", TIER2_AB_WORK: "E:\\work", TIER2_AB_KREA_ALLOCATION: "true" };
  const cfg = resolveConfig(env, { platform: "win32" });
  assert.equal(cfg.routes.length, 1);
  assert.equal(cfg.repairDir, null);
  const [allocation] = cfg.routes;
  const hires = kreaRoutes("cuda")[1];
  assert.deepEqual(allocation.body, { ...hires.body, width: 1120, height: 1120, hiresFix: { ...hires.body.hiresFix, upscaleBy: 2 } });
  assert.equal(allocation.finalSize, 2240);
  assert.deepEqual(kreaExpectedStepEvents(allocation), Array.from({ length: 10 }, (_, index) => [index + 1, 12]));
  const inventory = kreaAssetInventory(await readPatternEvidence("."), cfg.routes);
  assert.equal(inventory.length, 11);
  assert.deepEqual(inventory.find(({ file }) => file.includes("diffusion_models/")), CHECKPOINT);
  assert.ok(inventory.every(({ file }) => !file.startsWith("q4/") && !file.startsWith("loras/")));
});

test("a fixed pool cap is passed only to the owned GPU worker", () => {
  const inputs = { backend: "cuda", state: "/state", hub: "/weights/hub", xet: "/xet", workerId: "worker" };
  const inherited = { SCENEWORKS_CUDA_PROOF_POOL_BYTES: "5" };
  for (const allocation of [false, true]) {
    const env = childEnvironments(inherited, { ...inputs, allocation });
    assert.equal(env.api.SCENEWORKS_CUDA_PROOF_POOL_BYTES, undefined);
    assert.equal(env.worker("http://127.0.0.1:54321").SCENEWORKS_CUDA_PROOF_POOL_BYTES, allocation ? String(KREA_ALLOCATION_POOL_BYTES) : undefined);
  }
});

function receiptFields() {
  return [1120, 2240].flatMap((size, index) => {
    const admission = { event: "sgaf336_krea_allocation_admission", width: size, height: size, reference_count: index, context_present: true };
    Object.assign(admission, { context_revision: "imported-krea-hires-floor", context_width: size, context_height: size, context_reference_count: index, predicted_additional_bytes: 1000, available_additional_bytes: 1100 });
    const scope = { event: "generation_memory_policy_applied", engine: "krea_2_turbo", scope_configured: true };
    const pool = ["first_step_after_conditioning", "before_decode", "after_provider_return"].map((boundary, stage) => {
      const usedHigh = 2000 + index * 100 + stage * 10;
      return { event: "sgaf336_krea_allocation_pool", width: size, height: size, boundary, logical_ordinal: 0, cap_bytes: KREA_ALLOCATION_POOL_BYTES, used_bytes: 1800, used_high_bytes: usedHigh, reserved_bytes: 2150, reserved_high_bytes: usedHigh + 500 };
    });
    return [admission, scope, ...pool];
  });
}

const log = (fields) => fields.map((entry) => JSON.stringify({ fields: entry })).join("\n");
const memory = kreaWindowsMemoryFixture({ peakBytes: 2400, hostPeakBytes: 3000 });

test("actual pass context, cumulative native pool counters and owned process peaks keep their own domains", () => {
  const measured = kreaAllocationReceipt(log(receiptFields()), memory, 3200);
  const workerLog = receiptFields().map((entry) => JSON.stringify(entry)).join("\n");
  assert.deepEqual(kreaAllocationReceipt(workerLog, memory, 3200), measured);
  assert.equal(measured.status, "observed");
  assert.equal(measured.pool.length, 6);
  assert.ok(measured.ownedGpuPeakBytes > measured.admissions[1].available_additional_bytes);
  const noContext = receiptFields();
  noContext[0].context_present = false;
  assert.throws(() => kreaAllocationReceipt(log(noContext), memory, 3200), /context is missing/);
  assert.throws(() => kreaAllocationReceipt(log(receiptFields()), { ...memory, wddmDedicated: null }, 3200), /incomplete/);
  assert.throws(() => kreaAllocationReceipt(log(receiptFields()), memory, 2000), /containment limit/);
  const bad = receiptFields();
  bad.at(-1).reserved_high_bytes = KREA_ALLOCATION_POOL_BYTES + 1;
  assert.throws(() => kreaAllocationReceipt(log(bad), memory, 3200), /cap do not agree/);
  bad.at(-1).reserved_high_bytes = 2500;
  assert.throws(() => kreaAllocationReceipt(log(bad), memory, 3200), /counters regressed/);
});

test("allocation success requires the single pinned product and its attributed observations", () => {
  const run = { kind: KREA_SUITE, mode: "run", allocationProtocol: "bounded", routes: [{ status: "ran", verdict: "observed", allocationWitness: { status: "observed" } }] };
  assert.equal(exitCodeFor(run), 0);
  run.routes[0].allocationWitness.status = "inconclusive";
  assert.equal(exitCodeFor(run), 1);
});
