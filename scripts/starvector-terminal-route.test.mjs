import assert from "node:assert/strict";
import test from "node:test";
import { assembleRun, assembleSuites, validateBundle, vectorRequest } from "./starvector-terminal-route.mjs";

test("route runner can only construct the typed project-owned vector_generate request", () => {
  const request = vectorRequest({ projectId: "project", sourceAssetId: "asset", model: "starvector_1b", prompt: "icon" });
  assert.deepEqual(request, { projectId: "project", projectName: undefined, mode: "image_to_svg", model: "starvector_1b", sourceAssetId: "asset", prompt: "icon", sampling: undefined, detailBudget: undefined });
  assert.throws(() => vectorRequest({ projectId: "project", model: "starvector_1b" }), /sourceAssetId/);
});

test("route refuses a count-only or incomplete terminal bundle before product calls", () => {
  process.env.STARVECTOR_TERMINAL_PERMANENT_PIN = "81fda3bd5a9d5920ad9cdc62796df3305be96742";
  const records = (count, prefix) => Array.from({ length: count }, (_, index) => ({ case_id: `${prefix}-${index}`, projectId: "p", sourceAssetId: `a-${index}`, model: "starvector_8b" }));
  const bundle = { schema_version: 1, inference_revision: process.env.STARVECTOR_TERMINAL_PERMANENT_PIN, corpus_sha256: "a".repeat(64), tuples: { "candle-cuda:8b": { image_quality: records(120, "quality"), deterministic_parity: records(20, "parity"), lifecycle: records(4, "lifecycle"), limits: ["complete_root", "eos", "token_limit", "byte_limit", "wall_time_limit", "cancelled"].map((finish_reason, index) => ({ ...records(1, "limit")[0], case_id: `limit-${index}`, finish_reason })) } }, hostile_sanitizer: records(200, "hostile"), prompt_composition: records(60, "prompt") };
  assert.equal(validateBundle(bundle, "candle-cuda:8b"), bundle.tuples["candle-cuda:8b"]);
  bundle.hostile_sanitizer.pop(); assert.throws(() => validateBundle(bundle, "candle-cuda:8b"), /200 hostile/);
});

test("route refuses missing raw metric facts and missing terminal suite output", () => {
  assert.throws(() => assembleRun("mlx:1b", { image_quality: [], run_identity: {}, hardware: {} }, { image_quality: [], deterministic_parity: [], lifecycle: [], limits: [] }, []), /120 unique raw quality facts/);
  assert.throws(() => assembleSuites({}, {}, {}), /lacks source-owned hostile\/prompt suite evidence/);
});

test("terminal suite identities come from the same-run controller and observed workflow", () => {
  process.env.STARVECTOR_TERMINAL_RUN_ID = "campaign";
  process.env.STARVECTOR_TERMINAL_PERMANENT_PIN = "6".repeat(40);
  process.env.STARVECTOR_TERMINAL_ROUTE_CLOSURE_SHA256 = "a".repeat(64);
  const prompt = Array.from({ length: 60 }, (_, case_index) => ({ case_index, case_id: `prompt-v1-${case_index}`, prompt_sha256: "b".repeat(64), raster_model: "raster", expected_raster_revision: "revision" }));
  const bundle = { hostile_sanitizer: Array.from({ length: 200 }, () => ({ input_sha256: "c".repeat(64) })), prompt_composition: prompt };
  const events = { hostile_sanitizer: Array(200).fill({}), prompt_composition: prompt.map(() => ({ job: { workflow: { disclosure: "raster_to_vector", rasterStage: { model: "raster", revision: "revision" } }, terminalRasterObservation: { providerId: "native-raster", model: "raster", revision: "revision" } } })) };
  const observation = { packages: { "scikit-image": "0.25.2", lpips: "0.1.4" }, metrics_lock_sha256: "d".repeat(64), lpips_linear_sha256: "e".repeat(64), alexnet_sha256: "f".repeat(64), metric_transcript_path: "/tmp/metric-transcript", metric_transcript_sha256: "1".repeat(64), clip: { provider_id: "open-clip-torch", model: "ViT-B-32", revision: "clip-revision", inventory_sha256: "2".repeat(64) } };
  const metrics = { terminal_suite_measurements: { hostile_cases: [], prompt_cases: [], metric_observation: observation } };
  const context = { campaign_run_id: "campaign", permanent_pin: "6".repeat(40), workflow_run_id: "123", workflow_run_attempt: 1, controller_started_at: "2026-08-29T00:00:00Z", service: { started_at: "2026-08-29T00:00:00Z", sceneworks_revision: "7".repeat(40) }, route: { path: "scripts/starvector-terminal-route.mjs", sha256: "a".repeat(64), sceneworks_revision: "7".repeat(40) }, metrics: { metrics_lock_sha256: "d".repeat(64), packages: [{ name: "scikit-image", version: "0.25.2" }, { name: "lpips", version: "0.1.4" }], weights: { lpips_linear: { sha256: "e".repeat(64) }, alexnet: { sha256: "f".repeat(64) } }, clip: { provider_id: "open-clip-torch", model: "ViT-B-32", revision: "clip-revision", checkpoint: { sha256: "2".repeat(64) } } }, prompt_raster: { provider_id: "native-raster", model: "raster", revision: "revision", inventory_sha256: "3".repeat(64) }, inference_preflight: { workflow_run_id: "9" } };
  const result = assembleSuites(bundle, events, metrics, context, "4".repeat(64), "2026-08-29T01:00:00Z", "sha256:fixture");
  assert.equal(result.execution.head_sha, "7".repeat(40));
  assert.equal(result.prompt_composition.raster_provider_id, "native-raster");
  assert.equal(result.metric_identity.ssim.package_version, "0.25.2");
  const driftedWorkflow = structuredClone(events); driftedWorkflow.prompt_composition[0].job.workflow.rasterStage.revision = "drifted";
  assert.throws(() => assembleSuites(bundle, driftedWorkflow, metrics, context, "4".repeat(64), "2026-08-29T01:00:00Z", "sha256:fixture"), /actual prompt workflow/);
  const driftedProvider = structuredClone(events); driftedProvider.prompt_composition[0].job.terminalRasterObservation.providerId = "claimed-only";
  assert.throws(() => assembleSuites(bundle, driftedProvider, metrics, context, "4".repeat(64), "2026-08-29T01:00:00Z", "sha256:fixture"), /raster provider\/model identity/);
  const driftedMetric = structuredClone(metrics); driftedMetric.terminal_suite_measurements.metric_observation.clip.inventory_sha256 = "5".repeat(64);
  assert.throws(() => assembleSuites(bundle, events, driftedMetric, context, "4".repeat(64), "2026-08-29T01:00:00Z", "sha256:fixture"), /OpenCLIP identity drifted/);
  const driftedRun = structuredClone(context); driftedRun.workflow_run_id = null;
  assert.throws(() => assembleSuites(bundle, events, metrics, driftedRun, "4".repeat(64), "2026-08-29T01:00:00Z", "sha256:fixture"), /workflow execution identity/);
});

// Host observations must not inherit the provider's accelerator-only metric.
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { observeTerminalMemory, startTerminalMemorySampler, terminalHardwareFromSamples } from "./lib/starvector-terminal-memory.mjs";
import { runLifecycle } from "./starvector-terminal-route.mjs";
const memorySample = (available = 800, rss = 100) => ({ observed_at: "2026-09-10T10:00:00Z", total_bytes: 1000, available_bytes: available, processes: [{ pid: 12, rss_bytes: 20 }, { pid: 13, rss_bytes: rss }], accelerator: { uuid: "GPU-abc", index: "0", name: "GPU", driver: "driver", total_bytes: 500, free_bytes: 450, used_bytes: 50 } });

test("runtime receipt separates OS RSS, baseline pressure, allocator and selected CUDA memory", () => {
  const samples = [{ ...memorySample(), phase: "before_first_provider_load" }, { ...memorySample(700, 120), phase: "running" }, { ...memorySample(850, 90), phase: "after_native_suites" }];
  const common = { samples, allocatorPeaks: [330], arch: "arm64", runnerName: "fixture" };
  const mac = terminalHardwareFromSamples({ ...common, platform: "darwin" });
  assert.equal(mac.peak_process_rss_bytes, 140);
  assert.equal(mac.baseline_available_bytes, 800);
  assert.equal(mac.accelerator.baseline_free_bytes, 800);
  assert.equal(mac.accelerator.peak_used_bytes, 300);
  const cuda = terminalHardwareFromSamples({ ...common, platform: "win32" });
  assert.equal(cuda.peak_process_rss_bytes, 140);
  assert.equal(cuda.accelerator.peak_used_bytes, 330);
  assert.equal(cuda.accelerator.baseline_free_bytes, 450);
  const drift = structuredClone(samples); drift[1].accelerator.uuid = "GPU-def";
  assert.throws(() => terminalHardwareFromSamples({ ...common, samples: drift, platform: "win32" }), /identity drifted/);
  assert.throws(() => terminalHardwareFromSamples({ ...common, samples: samples.slice(1), platform: "darwin" }), /baseline/);
});

test("OS sampler reads known RSS units, rejects missing processes and binds CUDA UUID", async () => {
  const service = { api_pid: 12, worker_pid: 13, worker: { gpu_id: "0" }, gpu_binding: { uuid: "GPU-abc", gpu_id: "0", name: "GPU" } };
  const execute = async (command) => ({ stdout: command === "sysctl" ? "1000000" : command === "vm_stat" ? "Mach Virtual Memory Statistics: (page size of 4096 bytes)\nPages free: 20.\nPages inactive: 10.\nPages speculative: 5.\n" : "12 10\n13 20\n" });
  const mac = await observeTerminalMemory(service, { platform: "darwin", execute });
  assert.equal(mac.available_bytes, 35 * 4096);
  assert.deepEqual(mac.processes, [{ pid: 12, rss_bytes: 10240 }, { pid: 13, rss_bytes: 20480 }]);
  await assert.rejects(() => observeTerminalMemory(service, { platform: "darwin", execute: async (command) => command === "ps" ? { stdout: "12 10\n" } : execute(command) }), /unobservable/);
  const windows = { platform: "win32", execute: async () => ({ stdout: JSON.stringify(memorySample()) }), cuda: async () => memorySample().accelerator };
  assert.equal((await observeTerminalMemory(service, windows)).processes[1].rss_bytes, 100);
  await assert.rejects(() => observeTerminalMemory(service, { ...windows, cuda: async () => ({ ...memorySample().accelerator, uuid: "GPU-def" }) }), /identity drifted/);
});

test("sampler baseline precedes work, records unload transitions and awaits final probe before hashing", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "terminal-memory-"));
  const file = path.join(root, "samples.ndjson");
  let active = 0, count = 0;
  try {
    const sampler = await startTerminalMemorySampler({ file, intervalMs: 5, observe: async () => { active++; await new Promise((resolve) => setTimeout(resolve, 8)); count++; active--; return memorySample(); } });
    assert.equal(count, 1); // The first provider request can only follow baseline.
    await new Promise((resolve) => setTimeout(resolve, 6));
    await sampler.transition(async () => { assert.equal(active, 0); });
    const result = await sampler.stop();
    assert.equal(active, 0);
    assert.equal(result.samples[0].phase, "before_first_provider_load");
    assert.equal(result.samples.at(-1).phase, "after_native_suites");
    const bytes = await readFile(file, "utf8");
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.equal(await readFile(file, "utf8"), bytes);
    assert.equal(bytes.trim().split("\n").length, result.samples.length);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("sampler fails closed on lost telemetry and leaves its authentic partial samples", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "terminal-memory-loss-"));
  let count = 0;
  try {
    const file = path.join(root, "samples.ndjson");
    const sampler = await startTerminalMemorySampler({ file, intervalMs: 1, observe: async () => { if (count++) throw new Error("owned worker unobservable"); return memorySample(); } });
    await new Promise((resolve) => setTimeout(resolve, 10));
    await assert.rejects(() => sampler.stop(), /unobservable/);
    assert.equal((await readFile(file, "utf8")).trim().split("\n").length, 1);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test("lifecycle requires observed worker exit and a new process before provider reload", async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), "terminal-lifecycle-")), order = [];
  const records = ["load", "unload", "reload", "memory_reported"].map((operation) => ({ operation, case_id: operation }));
  const dependencies = {
    unload: async () => { order.push("exit"); return { status: "succeeded", exited: true }; },
    reload: async () => { order.push("spawn"); return { status: "succeeded", previous_worker_pid: 13, worker_pid: 14 }; },
    submit: async (_base, record) => { order.push(record.operation); return { terminalEvidence: { accepted: true, finishReason: "complete_root" }, terminalMetrics: { peakMemoryBytes: 300 } }; },
  };
  try {
    const result = await runLifecycle("http://unused", records, path.join(root, "transcript"), {}, dependencies);
    assert.deepEqual(order, ["load", "exit", "spawn", "reload", "memory_reported"]);
    assert.equal(result[1].observation.exited, true);
    await assert.rejects(() => runLifecycle("http://unused", records, path.join(root, "rejected"), {}, { ...dependencies, unload: async () => ({ ok: true }) }), /did not exit/);
    await assert.rejects(() => runLifecycle("http://unused", records, path.join(root, "reused"), {}, { ...dependencies, reload: async () => ({ status: "succeeded", previous_worker_pid: 13, worker_pid: 13 }) }), /new observed PID/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
