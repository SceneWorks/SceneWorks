import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { KREA_WDDM_SOURCE, kreaWindowsDedicatedPeak } from "./lib/krea-windows-worker-memory.mjs";
import { sha256 } from "./lib/file-sha256.mjs";
import { kreaAllocationOverlay, kreaWorkerLogEvents } from "./krea-tier2-allocation.mjs";

export const SAMPLER_SOURCE_FILE = "crates/sceneworks-worker/src/memory_strategy.rs";
const BASE_SAMPLER_SOURCE_SHA = "c69878bde04838b8a2b51cbbefba32c22fa5279aea86cd0cec981bd37f63b2cf";
// Candle's repaired plan-aware scope hook preserves the same provider progress callback.
// Reviewed at app bed9b3553aaf7f9435bea50254e717dc334f0280.
const HEAD_SAMPLER_SOURCE_SHA = "75bd47b689482be4aab4a099162cb4623af3af34bd2b6b890f3c9751409cb0f9";
const SAMPLER_HOOK_FILE = new URL("./ci/krea-perf-timer.rs.in", import.meta.url);
const HIRES_PROGRESS_HOOK_FILE = new URL("./ci/krea-hires-progress.rs.in", import.meta.url);
const ANCHOR = `) -> gen_core::Result<T> {\n    // A reference render may never take the legacy no-context shortcut`;
const OBSERVE = `    let mut krea_performance_sampler = KreaPerformanceSampler::new(generator, request);
    let mut observe_krea_progress = |progress| {
        if let Some(sampler) = krea_performance_sampler.as_mut() {
            sampler.observe(&progress);
        }
        on_progress(progress);
    };
    let on_progress = &mut observe_krea_progress;
`;
const OBSERVE_HIRES = `    let mut krea_hires_progress = (generator.descriptor().id == "krea_2_turbo")
        .then(KreaHiresProgress::default);
    let mut observe_hires_progress = |progress| {
        if let Some(trace) = krea_hires_progress.as_mut() {
            trace.observe(&progress);
        }
        on_progress(progress);
    };
    let on_progress = &mut observe_hires_progress;
`;

export async function bindKreaSamplerOverlay(original, side, { allocation = false, hiresProgress = false, perf = true } = {}) {
  const source = original.toString("utf8").replaceAll("\r\n", "\n");
  const sourceSha256 = sha256(source);
  assert.ok(["base", "head"].includes(side));
  const known = sourceSha256 === (side === "base" ? BASE_SAMPLER_SOURCE_SHA : HEAD_SAMPLER_SOURCE_SHA);
  assert.ok(known, `unreviewed ${side} file for the Krea sampler observer: ${sourceSha256}`);
  assert.equal(source.split(ANCHOR).length, 2, "Krea sampler observer requires exactly one native callback owner");
  if (allocation) {
    assert.equal(side, "head", "the wide-im2col allocation witness builds only the corrected candidate");
    const content = await kreaAllocationOverlay(source, ANCHOR);
    return { content, sourceSha256, overlaySha256: sha256(content) };
  }
  const [prefix, rest] = ANCHOR.split("\n");
  assert.ok(perf || hiresProgress, "Krea overlay needs a selected observer");
  let materialized = source.replace(ANCHOR, `${prefix}\n${perf ? OBSERVE : ""}${hiresProgress ? OBSERVE_HIRES : ""}${rest}`);
  if (hiresProgress) {
    // The frozen Candle path has no request scope. The candidate still uses its applied scope.
    const direct = "        return run(generator, request, None, on_progress);";
    const scoped = "    if matches!(result, Err(gen_core::Error::ProcessPoisoned(_))) {";
    assert.equal(materialized.split(direct).length, 2, "expected one direct provider return");
    assert.equal(materialized.split(scoped).length, 2, "expected one scoped provider return");
    materialized = materialized.replace(direct, `        let result = run(generator, request, None, on_progress);
        if let (Ok(_), Some(trace)) = (&result, krea_hires_progress) {
            trace.publish(request, None, false);
        }
        return result;`);
    materialized = materialized.replace(scoped, `    if let (Ok(_), Some(trace)) = (&result, krea_hires_progress) {
        trace.publish(request, Some(context), scope.is_some());
    }
${scoped}`);
    materialized += `\n${await readFile(HIRES_PROGRESS_HOOK_FILE, "utf8")}`;
  }
  if (perf) materialized += `\n${await readFile(SAMPLER_HOOK_FILE, "utf8")}`;
  return { content: materialized, sourceSha256, overlaySha256: sha256(materialized) };
}

// NVIDIA can still identify compute applications and their GPU UUID on WDDM.
// Dedicated process bytes come solely from the separate PDH observation.
export function kreaOwnedCudaAttachment(csv, ownedPid, uuid) {
  const uuidPattern = /^GPU-[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/i;
  assert.ok(uuidPattern.test(uuid) && Number.isSafeInteger(ownedPid) && ownedPid > 0);
  const seen = new Set();
  let attached = false;
  for (const line of csv.trim().split(/\r?\n/).filter(Boolean)) {
    const [pid, device, ...unexpected] = line.split(",").map((value) => value.trim());
    assert.ok(unexpected.length === 0 && /^\d{1,10}$/.test(pid) && Number(pid) > 0 && uuidPattern.test(device), "invalid compute-app process/UUID row");
    const identity = `${pid}/${device.toLowerCase()}`;
    assert.ok(!seen.has(identity), "duplicate compute-app process/UUID row");
    seen.add(identity);
    if (Number(pid) !== ownedPid) continue;
    assert.equal(device.toLowerCase(), uuid.toLowerCase(), "the owned worker appeared on an unleased device");
    attached = true;
  }
  return attached;
}

export function kreaSamplerReceipt(log, route) {
  const expected = route.proof === "raw" ? [[512, 512, 10]] : [[512, 512, 8], [768, 768, 2]];
  const desiredEngine = route.proof === "raw" ? "krea_2_raw" : "krea_2_turbo";
  const passes = kreaWorkerLogEvents(log)
    .filter(({ event }) => event === "sgaf336_krea_sampler")
    .map(({ engine, width, height, steps, elapsed_ns: elapsedNs, valid }) => ({ engine, width, height, steps, elapsedNs, valid }));
  assert.deepEqual(passes.map(({ width, height, steps }) => [width, height, steps]), expected, "expected separate native timing for each actual Krea pass");
  assert.ok(passes.every(({ engine, valid, elapsedNs }) => engine === desiredEngine && valid && Number.isSafeInteger(elapsedNs) && elapsedNs > 0), "native synchronized sampler interval missing or invalid");
  return passes;
}

export function kreaHiresProgressReceipt(log, route, { requireScope = false } = {}) {
  assert.equal(route.proof, "hires");
  const passes = kreaWorkerLogEvents(log).filter(({ event }) => event === "sgaf336_krea_hires_progress");
  assert.equal(passes.length, 2, "expected two completed native Hires.fix passes");
  const expected = [
    [route.body.width, route.body.height, route.body.advanced.steps, 0, 8],
    [route.finalSize, route.finalSize, route.body.hiresFix.steps, 1, 2],
  ];
  return passes.map((pass, index) => {
    const [width, height, authoredSteps, referenceCount, steps] = expected[index];
    assert.deepEqual([pass.engine, pass.width, pass.height, pass.authored_steps, pass.reference_count],
      ["krea_2_turbo", width, height, authoredSteps, referenceCount], "native Hires.fix pass geometry or reference changed");
    assert.equal(typeof pass.context_present, "boolean", "native Hires.fix pass is missing admission context evidence");
    assert.equal(typeof pass.scope_configured, "boolean", "native Hires.fix pass is missing scope evidence");
    assert.ok((!requireScope || pass.scope_configured) && (!pass.scope_configured || pass.context_present),
      "native Hires.fix pass did not configure its required scope");
    const contextGeometry = [pass.context_width, pass.context_height, pass.context_reference_count];
    assert.deepEqual(contextGeometry, pass.context_present ? [width, height, referenceCount] : [0, 0, 0],
      "native Hires.fix admission geometry differs from its provider pass");
    assert.ok(pass.valid === true && pass.decode_count === 1, "native Hires.fix pass needs one decode after its steps");
    assert.ok(typeof pass.coordinates === "string" && pass.coordinates.length <= 128, "invalid native Hires.fix coordinate record");
    const coordinates = JSON.parse(pass.coordinates);
    assert.deepEqual(coordinates, Array.from({ length: steps }, (_, step) => [step + 1, steps]),
      "native Hires.fix step coordinates do not match the exact clipped schedule");
    return { width, height, authoredSteps, referenceCount, coordinates, decodeCount: pass.decode_count,
      contextPresent: pass.context_present, scopeConfigured: pass.scope_configured };
  });
}

// Port of SceneWorks/inference b2452ead6:scripts/ci/sga_paired_performance.py.
// The existing JavaScript driver has no common Python executable on both job hosts.
export const KREA_PERF_PAIRS = 6;
const CEILING = 1.02;
export const KREA_PERF_ORDER = Object.freeze(Array.from({ length: KREA_PERF_PAIRS }, (_, index) => index % 2 === 0 ? ["base", "head"] : ["head", "base"]));
export const KREA_PERF_COVERAGE = 1 - 2 / 2 ** KREA_PERF_PAIRS;

export function compareKreaPerformance(pairs, backend) {
  if (pairs.length !== KREA_PERF_PAIRS) return { status: "inconclusive", reason: "all six ordered pairs are required" };
  if (!pairs.every(({ order }, index) => JSON.stringify(order) === JSON.stringify(KREA_PERF_ORDER[index]))) return { status: "inconclusive", reason: "pair order departed from the frozen protocol" };
  const sides = ["base", "head"];
  if (!pairs.every((pair) => sides.every((side) => pair[side]?.status === "completed" && pair[side].performance?.eligibility === "valid"))) return { status: "inconclusive", reason: "at least one arm lacks a completed render or complete attributed measurements" };
  if (pairs.some(({ base, head }) => base.performance.device !== head.performance.device || base.requestSha256 !== head.requestSha256)) return { status: "inconclusive", reason: "paired device or authored request differed" };
  const runs = pairs.flatMap((pair) => sides.map((side) => pair[side]));
  const ordered = runs.map(({ performance }) => performance);
  if (new Set(runs.map(({ requestSha256 }) => requestSha256)).size !== 1) return { status: "inconclusive", reason: "authored request changed between pairs" };
  const positive = (value) => Number.isFinite(value) && value > 0;
  if (!ordered.every(({ secondsPerOuterStep, productSeconds, hostPeakBytes, devicePeakBytes }) => positive(secondsPerOuterStep) && positive(productSeconds) && Number.isSafeInteger(hostPeakBytes) && hostPeakBytes > 0 && Number.isSafeInteger(devicePeakBytes) && devicePeakBytes > 0)) return { status: "inconclusive", reason: "invalid timing or memory sample" };
  if (new Set(ordered.map(({ device }) => device)).size !== 1) return { status: "inconclusive", reason: "device changed between pairs" };
  if (backend === "cuda" && (!ordered.every(({ deviceSource, wddmAdapter }) => deviceSource === KREA_WDDM_SOURCE && wddmAdapter)
    || new Set(ordered.map(({ wddmAdapter }) => `${wddmAdapter.luid}/${wddmAdapter.physical_node}`)).size !== 1)) {
    return { status: "inconclusive", reason: "Windows device accounting source or adapter changed between arms" };
  }
  const timing = (field) => {
    const pairRatios = pairs.map(({ base, head }) => head.performance[field] / base.performance[field]);
    const sorted = [...pairRatios].sort((a, b) => a - b);
    const lower = sorted[0], upper = sorted.at(-1);
    return { status: upper <= CEILING ? "pass" : lower > CEILING ? "fail" : "inconclusive", pairRatios, medianRatio: (sorted[2] + sorted[3]) / 2, interval: [lower, upper] };
  };
  const peak = (field) => {
    const [basePeakBytes, headPeakBytes] = sides.map((side) => Math.max(...pairs.map((pair) => pair[side].performance[field])));
    return { status: 100n * BigInt(headPeakBytes) <= 102n * BigInt(basePeakBytes) ? "pass" : "fail", basePeakBytes, headPeakBytes, ratio: headPeakBytes / basePeakBytes };
  };
  const results = { secondsPerOuterStep: timing("secondsPerOuterStep"), productSeconds: timing("productSeconds"), hostOrUnifiedPeak: peak("hostPeakBytes"), deviceOrAllocatorPeak: { ...peak("devicePeakBytes"), ...(backend === "cuda" ? { source: KREA_WDDM_SOURCE, physicalResidencyClaim: false } : {}) } };
  const statuses = Object.values(results).map(({ status }) => status);
  return { status: statuses.includes("fail") ? "fail" : statuses.includes("inconclusive") ? "inconclusive" : "pass", backend, ceiling: CEILING, intervalCoverage: KREA_PERF_COVERAGE, ...results };
}

export function kreaPerformanceObservation({ route, passes, postedMonotonicMs, events, memory, metrics, machine }) {
  const productEnd = events.find(({ status, publishedAssets }) => status === "completed" && publishedAssets === 1)?.receivedMonotonicMs;
  assert.ok(Number.isFinite(productEnd) && productEnd > postedMonotonicMs, "persisted terminal SSE measurement missing");
  const hostPeakBytes = memory.hostLifetimePeakBytes;
  const mlx = route.backend === "mlx";
  const dedicated = mlx ? null : kreaWindowsDedicatedPeak(memory);
  const devicePeakBytes = mlx ? metrics.peakMemoryBytes : dedicated.peakBytes;
  const device = mlx ? machine : memory.gpuUuid;
  assert.ok(device && [hostPeakBytes, devicePeakBytes].every((value) => Number.isSafeInteger(value) && value > 0), "matched memory peaks are incomplete");
  const secondsPerOuterStep = passes.reduce((total, { elapsedNs }) => total + elapsedNs, 0) / (1e9 * route.executedSteps);
  assert.ok(Number.isFinite(secondsPerOuterStep) && secondsPerOuterStep > 0);
  return { eligibility: "valid", secondsPerOuterStep, passes, productSeconds: (productEnd - postedMonotonicMs) / 1000, hostPeakBytes, devicePeakBytes, device,
    hostSource: mlx ? "Darwin proc_pid_rusage lifetime physical footprint of worker" : "Windows PeakWorkingSet64 of worker",
    deviceSource: mlx ? "MLX allocator native high-water reset per job" : KREA_WDDM_SOURCE,
    ...(dedicated ? { wddmAdapter: dedicated.adapter, physicalResidencyClaim: false } : {}) };
}
