import assert from "node:assert/strict";
import test from "node:test";
import { dispatchOptions, requireKreaCudaRunLease } from "./tier2-ab.mjs";
import { KREA_WDDM_SOURCE } from "./lib/krea-windows-worker-memory.mjs";
import { kreaWindowsMemoryFixture } from "./lib/krea-windows-worker-memory-fixture.mjs";
import { KREA_SUITE, kreaRoutes } from "./krea-tier2-proof.mjs";
import {
  KREA_PERF_ORDER, bindKreaSamplerOverlay, compareKreaPerformance, kreaOwnedCudaAttachment,
  kreaPerformanceObservation, kreaSamplerReceipt,
} from "./krea-tier2-performance.mjs";

const GPU = "GPU-2160ecaf-a2b7-7321-87a2-1234567890ab";
const OTHER = "GPU-3160ecaf-a2b7-7321-87a2-1234567890ab";
const [, HIRES] = kreaRoutes("cuda");
const SHA = "1".repeat(40);

test("performance arming is opt-in and the overlay rejects unreviewed source", async () => {
  const options = { suite: KREA_SUITE, head_sha: SHA, preflight_only: false };
  const dispatch = (body) => dispatchOptions({ profile: "tier2-ab", backend: "candle", ref: SHA, inference_sha: "2".repeat(40), download_missing: true, moss_options_json: JSON.stringify(body) }, SHA);
  assert.equal(dispatch(options).krea_performance, false);
  assert.equal(dispatch({ ...options, performance: true }).krea_performance, true);
  assert.throws(() => dispatch({ ...options, performance: "true" }), /must be boolean/);
  for (const side of ["base", "head"]) await assert.rejects(bindKreaSamplerOverlay(Buffer.from("unexpected"), side), /unreviewed/);
});

test("NVIDIA compute app attests the exact worker PID and leased UUID", () => {
  assert.equal(kreaOwnedCudaAttachment(`37, ${GPU}\n9, ${GPU}\n8, ${OTHER}`, 37, GPU), true);
  assert.equal(kreaOwnedCudaAttachment(`9, ${GPU}`, 37, GPU), false);
  assert.throws(() => kreaOwnedCudaAttachment(`37, ${OTHER}`, 37, GPU), /unleased device/);
  assert.throws(() => kreaOwnedCudaAttachment(`37, ${GPU}\n37, ${GPU}`, 37, GPU), /duplicate/);
  assert.throws(() => kreaOwnedCudaAttachment(`37, 92a, ${GPU}`, 37, GPU), /invalid/);
});

function nativePasses() {
  return [
    { engine: "krea_2_turbo", width: 512, height: 512, steps: 8, elapsed_ns: 8e9, valid: true },
    { engine: "krea_2_turbo", width: 768, height: 768, steps: 2, elapsed_ns: 3e9, valid: true },
  ];
}
const logFor = (passes) => passes.map((fields) => JSON.stringify({ fields: { event: "sgaf336_krea_sampler", ...fields } })).join("\n");

test("Hires requires independently fenced timings and a persisted terminal SSE, with no metric fallback", () => {
  const passes = kreaSamplerReceipt(logFor(nativePasses()), HIRES);
  const workerLog = nativePasses().map((entry) => JSON.stringify({ event: "sgaf336_krea_sampler", ...entry })).join("\n");
  assert.deepEqual(kreaSamplerReceipt(workerLog, HIRES), passes);
  assert.throws(() => kreaSamplerReceipt(logFor(nativePasses().slice(0, 1)), HIRES), /separate native timing/);
  assert.throws(() => kreaSamplerReceipt(logFor(nativePasses().map((pass) => ({ ...pass, valid: false }))), HIRES), /missing or invalid/);
  const memory = kreaWindowsMemoryFixture({ gpuUuid: GPU });
  const proof = { route: HIRES, passes, postedMonotonicMs: 1000, memory, metrics: { peakMemoryBytes: 98 }, events: [
    { status: "completed", publishedAssets: 0, receivedMonotonicMs: 11500 },
    { status: "completed", publishedAssets: 1, receivedMonotonicMs: 16500 },
  ] };
  const result = kreaPerformanceObservation(proof);
  assert.equal(result.secondsPerOuterStep, 1.1);
  assert.equal(result.productSeconds, 15.5);
  assert.equal(result.devicePeakBytes, 80);
  assert.throws(() => kreaPerformanceObservation({ ...proof, events: proof.events.slice(0, 1) }), /persisted terminal/);
  assert.equal(result.physicalResidencyClaim, false);
  assert.equal(result.deviceSource, KREA_WDDM_SOURCE);
  assert.throws(() => kreaPerformanceObservation({ ...proof, memory: { ...memory, ownedGpuIdentityError: "wrong UUID" } }), /incomplete/);
  assert.throws(() => kreaPerformanceObservation({ ...proof, memory: { ...memory, wddmDedicated: null } }), /incomplete/);
  const mlx = kreaPerformanceObservation({ ...proof, route: { ...HIRES, backend: "mlx" }, machine: "mac", memory: { hostLifetimePeakBytes: 100 } });
  assert.equal(mlx.devicePeakBytes, 98);
  assert.equal(mlx.deviceSource, "MLX allocator native high-water reset per job");
  assert.equal(mlx.wddmAdapter, undefined);
});

function arm(ratio, memory = 102, device = GPU) {
  const wddmAdapter = kreaWindowsMemoryFixture({ gpuUuid: device }).wddmDedicated.rows[0].adapter;
  return { status: "completed", requestSha256: SHA, performance: { eligibility: "valid", device, deviceSource: KREA_WDDM_SOURCE,
    wddmAdapter, secondsPerOuterStep: ratio, productSeconds: 100 * ratio, hostPeakBytes: memory, devicePeakBytes: memory } };
}

function six(ratios, memory = 102) {
  return KREA_PERF_ORDER.map((order, index) => ({ order, base: arm(1, 100), head: arm(ratios[index], memory) }));
}

test("frozen six-pair method passes wholly below the ceiling, fails wholly above, and stops on overlap", () => {
  assert.deepEqual(KREA_PERF_ORDER, [["base", "head"], ["head", "base"], ["base", "head"], ["head", "base"], ["base", "head"], ["head", "base"]]);
  const pass = compareKreaPerformance(six([1, 1.01, 1.02, 0.98, 0.97, 1.02]), "cuda");
  assert.equal(pass.status, "pass");
  assert.equal(pass.intervalCoverage, 0.96875);
  assert.deepEqual(pass.secondsPerOuterStep.interval, [0.97, 1.02]);
  assert.equal(pass.secondsPerOuterStep.medianRatio, 1.005);
  assert.equal(compareKreaPerformance(six([1, 1, 1, 1, 1, 1.03]), "cuda").status, "inconclusive");
  assert.equal(compareKreaPerformance(six(Array(6).fill(1.03)), "cuda").status, "fail");
  assert.equal(compareKreaPerformance(six(Array(6).fill(1), 103), "cuda").deviceOrAllocatorPeak.status, "fail");
  const pairs = six(Array(6).fill(1));
  assert.equal(compareKreaPerformance(pairs.slice(0, 5), "cuda").status, "inconclusive");
  pairs[0].head.status = "error";
  assert.match(compareKreaPerformance(pairs, "cuda").reason, /completed render/);
  pairs[0].head.status = "completed";
  pairs[0].head = arm(1, 100, OTHER);
  assert.equal(compareKreaPerformance(pairs, "cuda").status, "inconclusive");
});

test("CUDA memory compares one predeclared accounting method and adapter across all arms", () => {
  const pairs = six(Array(6).fill(1));
  pairs[0].head.performance.deviceSource = "nvidia-smi";
  assert.match(compareKreaPerformance(pairs, "cuda").reason, /accounting source/);
  pairs[0].head.performance.deviceSource = KREA_WDDM_SOURCE;
  pairs[1].head.performance.wddmAdapter = { ...pairs[1].head.performance.wddmAdapter, luid: "luid_0x00000000_0x0001e267" };
  assert.match(compareKreaPerformance(pairs, "cuda").reason, /adapter changed/);
});


test("Windows Krea performance and allocation require the exact CUDA lease only for execution", () => {
  for (const options of [{ perf: true }, { allocation: true }]) {
    const cfg = { backend: "cuda", ...options };
    assert.throws(() => requireKreaCudaRunLease(cfg, "run", {}), /exact-UUID CUDA execution guard/);
    assert.throws(() => requireKreaCudaRunLease(cfg, "run", { SW_CI_CUDA_EXECUTION_ACTIVE: "true" }), /exact-UUID CUDA execution guard/);
    assert.doesNotThrow(() => requireKreaCudaRunLease(cfg, "run", { SW_CI_CUDA_EXECUTION_ACTIVE: "1234" }));
    for (const mode of ["preflight", "stage", "build"]) {
      assert.doesNotThrow(() => requireKreaCudaRunLease(cfg, mode, {}));
    }
  }
  assert.doesNotThrow(() => requireKreaCudaRunLease({ backend: "cuda", perf: false, allocation: false }, "run", {}));
  assert.doesNotThrow(() => requireKreaCudaRunLease({ backend: "mlx", perf: true }, "run", {}));
});
