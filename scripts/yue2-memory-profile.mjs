#!/usr/bin/env node
// YuE2 memory profiling (sc-23001, epic 22988): the case list, capture runner, record checker and
// corpus ingest the terminal story (sc-23002) uses for its ONE calibration campaign on Mac Metal and
// Windows CUDA. Nothing here runs in CI, and nothing here gates a build: records only ever improve
// the admission estimate (crates/sceneworks-worker/src/yue2_admission.rs) when a later change
// recalibrates it from them.
//
// YuE2 has no image five-rung memory contract, so it is not an anchor of
// config/memory-calibration-plan.json and the five-rung adapters cannot capture it. What it shares
// with that harness is the part that matters for evidence:
//
//   * CURRENCY — a record carries the `candle:yue2` compile-closure digest it was captured under,
//     derived by scripts/inference-closure-digest.mjs from the pinned inference source, and is
//     `current` only while that equals the declared digest in config/inference-provider-closures.json
//     (the sc-17774 rule `evidenceSemantics` applies). The pin itself is provenance, never compared.
//   * IDENTITY — the exact model repo@revision and tier weights (the derived tier's pinned sha256),
//     the decoder repo@revision, the engine closure, the tier and the backend. A record whose
//     identity no longer matches the catalog or the declared closure is STALE: `check` reports it
//     and `ingest` refuses it. Staleness is tooling-only (FEATURE_DEVELOPMENT.md "Gate teardown"):
//     it never changes runtime behaviour.
//   * THE GUARD — Darwin captures run under scripts/memory-calibration-watchdog.py (phys_footprint
//     sampled, a stated footprint ceiling, a wall-clock budget), exactly as measure-memory-catalog
//     guards its captures.
//
// The native half is the ignored test `yue2_memory_profile::capture_case` in the worker: it resolves
// the installed weights as the YuE2 job does, runs the job's admission gate against the live budget,
// loads the registered `yue2` provider and renders, writing admission.json, stages.jsonl (stage
// marks on the sampler's clock) and outcome.json. This controller turns the sampler's samples plus
// those marks into per-stage measured peaks beside the per-stage estimate.
//
//   node scripts/yue2-memory-profile.mjs plan [--backend metal|cuda]
//   node scripts/yue2-memory-profile.mjs capture --case <id> --inference-repo <dir> --out <dir>
//        [--data-dir <app data dir>] [--gpu-id N] [--budget-minutes N] [--dry-run]
//   node scripts/yue2-memory-profile.mjs run --backend metal|cuda --inference-repo <dir> --out <dir> [...]
//   node scripts/yue2-memory-profile.mjs check <record.json ...>
//   node scripts/yue2-memory-profile.mjs ingest <record.json ...>
//
// `--out` must be OUTSIDE the repository (a capture from a dirty checkout is not evidence).
import { spawn, execFileSync } from "node:child_process";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { inferencePinFromCargo, providerClosureDigest } from "./inference-closure-digest.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
export const PLAN_PATH = "config/yue2-memory-profile-plan.json";
export const CLOSURES_PATH = "config/inference-provider-closures.json";
export const MANIFEST_PATH = "config/manifests/builtin.models.jsonc";
export const PLAN_SCHEMA = "sceneworks-yue2-memory-profile-plan-v1";
export const RECORD_SCHEMA = "sceneworks-yue2-memory-profile-record-v1";
export const BACKENDS = Object.freeze(["metal", "cuda"]);
export const TIERS = Object.freeze(["bf16", "q8", "q4"]);
export const STAGES = Object.freeze(["load", "plan", "semantic", "acoustic", "decode"]);
export const DECODERS = Object.freeze({ standard: "vae", legacy: "vae_legacy" });
/** The dedicated-VRAM allocator reserve admission charges on a CUDA card (fit_gate). */
export const CUDA_RESERVE_BYTES = 2 * 1024 ** 3;
const PLANNINGS = new Set(["full", "melody", "off"]);
export const REQUEST_KEYS = new Set([
  "planning", "style", "lyrics", "seed", "cfgScale", "steps", "scoreSampling", "semanticSampling",
]);

export function fail(message) {
  throw new Error(`yue2-memory-profile: ${message}`);
}

// ---- Plan ------------------------------------------------------------------------------------------

export async function readSources(root = ROOT) {
  const [plan, closures, manifest] = await Promise.all([
    readFile(path.join(root, PLAN_PATH), "utf8").then(JSON.parse),
    readFile(path.join(root, CLOSURES_PATH), "utf8").then(JSON.parse),
    readFile(path.join(root, MANIFEST_PATH), "utf8").then((body) => JSON.parse(stripJsoncComments(body))),
  ]);
  return { plan, closures, manifest };
}

export function manifestEntry(manifest, modelId) {
  const entry = manifest.models?.find((model) => model.id === modelId);
  if (!entry) fail(`the catalog has no ${modelId} entry`);
  return entry;
}

function validateRequest(request, label) {
  if (!request || typeof request !== "object") fail(`${label}: request must be an object`);
  for (const key of Object.keys(request)) {
    if (!REQUEST_KEYS.has(key)) fail(`${label}: unknown request field ${key}`);
  }
  if (!PLANNINGS.has(request.planning)) fail(`${label}: planning must be full, melody or off`);
  for (const key of ["style", "lyrics"]) {
    if (typeof request[key] !== "string" || !request[key].trim()) fail(`${label}: ${key} is required`);
  }
  if (!Number.isSafeInteger(request.seed) || request.seed < 0) fail(`${label}: seed must be a non-negative integer`);
  for (const key of ["scoreSampling", "semanticSampling"]) {
    const sampling = request[key];
    if (sampling === undefined) continue;
    for (const [field, value] of Object.entries(sampling)) {
      if (!["minTokens", "maxTokens"].includes(field)) fail(`${label}: ${key}.${field} is not a profile field`);
      if (!Number.isSafeInteger(value) || value < 0) fail(`${label}: ${key}.${field} must be a count`);
    }
    if (sampling.minTokens !== undefined && sampling.maxTokens !== undefined && sampling.minTokens > sampling.maxTokens) {
      fail(`${label}: ${key}.minTokens exceeds maxTokens`);
    }
  }
}

/** Validate the plan against the catalog and the closure table. */
export function validatePlan(plan, { closures, manifest }) {
  if (plan?.schema !== PLAN_SCHEMA) fail(`the plan is not ${PLAN_SCHEMA}`);
  if (!closures.providers?.[plan.lane]) {
    fail(`lane ${plan.lane} is not declared in ${CLOSURES_PATH}; a record could never be current`);
  }
  const entry = manifestEntry(manifest, plan.modelId);
  const variants = new Set(entry.downloads.filter((row) => !row.coRequisite).map((row) => row.variant));
  if (!Array.isArray(plan.backends) || !plan.backends.length) fail("the plan names no backends");
  for (const backend of plan.backends) {
    if (!BACKENDS.includes(backend)) fail(`unknown backend ${backend}`);
  }
  if (!plan.requests || !Object.keys(plan.requests).length) fail("the plan names no requests");
  for (const [name, spec] of Object.entries(plan.requests)) {
    if (!/^[a-z][a-z0-9-]*$/.test(name)) fail(`request name ${name} must be kebab-case`);
    if (!Array.isArray(spec.tiers) || !spec.tiers.length) fail(`${name}: no tiers`);
    for (const tier of spec.tiers) {
      if (!TIERS.includes(tier) || !variants.has(tier)) fail(`${name}: ${tier} is not a catalog tier`);
    }
    if (!DECODERS[spec.decoder]) fail(`${name}: unknown decoder ${spec.decoder}`);
    validateRequest(spec.request, name);
  }
  return plan;
}

/** The case list: every request at each of its tiers on each backend. */
export function expandCases(plan) {
  const cases = [];
  for (const [name, spec] of Object.entries(plan.requests)) {
    for (const tier of spec.tiers) {
      for (const backend of plan.backends) {
        cases.push({
          id: `${plan.modelId}:${tier}:${backend}:${name}`,
          modelId: plan.modelId,
          tier,
          backend,
          requestName: name,
          decoder: spec.decoder,
          request: spec.request,
        });
      }
    }
  }
  const ids = new Set(cases.map((item) => item.id));
  if (ids.size !== cases.length) fail("duplicate case ids");
  return cases;
}

/** The exact model and decoder identity a case measures, from the catalog. */
export function caseIdentity(item, manifest) {
  const entry = manifestEntry(manifest, item.modelId);
  const row = entry.downloads.find((download) => !download.coRequisite && download.variant === item.tier);
  if (!row) fail(`${item.id}: the catalog has no ${item.tier} row`);
  const derivation = row.localDerivation ?? null;
  const decoderRow = entry.downloads.find((download) => download.componentId === DECODERS[item.decoder]);
  if (!decoderRow) fail(`${item.id}: the catalog has no ${DECODERS[item.decoder]} decoder`);
  return {
    model: {
      id: item.modelId,
      repo: row.repo,
      revision: row.revision,
      tier: item.tier,
      // The derived tier's pinned weights; the released bf16 file is identified by repo@revision.
      weightsSha256: derivation?.weightsSha256 ?? null,
      weightsBytes: derivation?.weightsBytes ?? null,
      conversion: derivation?.conversion ?? null,
    },
    decoder: {
      choice: item.decoder,
      componentId: decoderRow.componentId,
      repo: decoderRow.repo,
      revision: decoderRow.revision,
    },
  };
}

/** The case file the native entrypoint reads (`yue2_memory_profile::ProfileCase`). */
export function caseFile(item) {
  return { id: item.id, tier: item.tier, decoder: item.decoder, request: item.request };
}

// ---- Measurement -----------------------------------------------------------------------------------

/** Parse `stages.jsonl`: `{stage, at}` marks in time order. */
export function parseStageMarks(body) {
  const marks = [];
  for (const line of String(body).split("\n")) {
    if (!line.trim()) continue;
    const mark = JSON.parse(line);
    if (![...STAGES, "done"].includes(mark.stage)) fail(`unknown stage mark ${mark.stage}`);
    if (!Number.isFinite(mark.at)) fail("a stage mark has no time");
    if (marks.length && mark.at < marks.at(-1).at) fail("stage marks go back in time");
    marks.push(mark);
  }
  if (!marks.length || marks[0].stage !== "load") fail("the stage marks do not start at the load");
  return marks;
}

/** Watchdog event log → `{at, bytes}` footprint samples. */
export function parseWatchdogSamples(body) {
  const samples = [];
  for (const line of String(body).split("\n")) {
    if (!line.trim()) continue;
    const event = JSON.parse(line);
    if (event.event !== "sample") continue;
    if (!Number.isSafeInteger(event.physicalFootprintBytes) || event.physicalFootprintBytes <= 0) {
      fail("a watchdog sample states no footprint");
    }
    samples.push({ at: event.at, bytes: event.physicalFootprintBytes });
  }
  return samples;
}

/**
 * Attribute each sample to the stage in force at its time and take each stage's peak. Samples
 * before the load mark or after the `done` mark belong to no stage (process start-up, teardown).
 */
export function stagePeaks(samples, marks) {
  const peaks = {};
  for (const sample of samples) {
    let stage = null;
    for (const mark of marks) {
      if (mark.at <= sample.at) stage = mark.stage;
      else break;
    }
    if (!stage || stage === "done") continue;
    const peak = (peaks[stage] ??= { peakBytes: 0, samples: 0 });
    peak.samples += 1;
    peak.peakBytes = Math.max(peak.peakBytes, sample.bytes);
  }
  return peaks;
}

// ---- Records -----------------------------------------------------------------------------------------

export function buildRecord({
  item, identity, engine, sceneworks, hardware, admission, outcome, sampler, samples, marks, capturedAt,
}) {
  const measured = marks ? stagePeaks(samples, marks) : {};
  const peak = Object.values(measured).reduce((max, stage) => Math.max(max, stage.peakBytes), 0);
  return {
    schema: RECORD_SCHEMA,
    caseId: item.id,
    lane: engine.lane,
    backend: item.backend,
    identity: { ...identity, engine, sceneworks },
    hardware,
    request: { name: item.requestName, ...item.request },
    admission,
    measured: { sampler, stages: measured, peakBytes: peak || null },
    outcome,
    capturedAt,
  };
}

export function validateRecord(record) {
  if (record?.schema !== RECORD_SCHEMA) fail("not a YuE2 memory-profile record");
  for (const field of ["caseId", "lane", "backend", "identity", "admission", "measured", "outcome"]) {
    if (record[field] === undefined) fail(`record ${record.caseId ?? "?"} has no ${field}`);
  }
  const { engine, sceneworks, model, decoder } = record.identity;
  if (!/^[0-9a-f]{64}$/.test(engine?.closureDigest ?? "")) fail(`${record.caseId}: no captured closure digest`);
  if (!/^[0-9a-f]{40}$/.test(engine?.inferenceRevision ?? "")) fail(`${record.caseId}: no inference revision`);
  if (!/^[0-9a-f]{40}$/.test(sceneworks?.revision ?? "")) fail(`${record.caseId}: no SceneWorks revision`);
  if (!model?.repo || !model?.revision || !decoder?.repo || !decoder?.revision) {
    fail(`${record.caseId}: model and decoder must be identified by repo@revision`);
  }
  if (!["completed", "refused", "failed"].includes(record.outcome.status)) {
    fail(`${record.caseId}: unknown outcome ${record.outcome.status}`);
  }
  if (record.outcome.status === "completed") {
    for (const stage of Object.keys(record.measured.stages)) {
      if (!STAGES.includes(stage)) fail(`${record.caseId}: unknown measured stage ${stage}`);
    }
    if (!record.measured.stages.load) fail(`${record.caseId}: a completed record measured no load`);
  }
  return record;
}

/**
 * Grade a record against the current catalog, closure table and plan. `current` needs the captured
 * closure digest (and digest version) to equal the declared one, the model/decoder identity to equal
 * the catalog's, a clean SceneWorks checkout, and a case the plan still declares.
 */
export function gradeRecord(record, { plan, closures, manifest }) {
  validateRecord(record);
  const reasons = [];
  const declared = closures.providers?.[record.lane];
  const engine = record.identity.engine;
  if (!declared) reasons.push(`lane ${record.lane} is no longer declared`);
  else if (engine.closureDigest !== declared.digest) {
    reasons.push(
      `stale closure: captured ${engine.closureDigest.slice(0, 12)}, declared ${declared.digest.slice(0, 12)} ` +
        "(the engine code this record measured has changed)",
    );
  }
  if (engine.digestVersion !== closures.digestVersion) {
    reasons.push(`closure digest version ${engine.digestVersion} is not ${closures.digestVersion}`);
  }
  const item = expandCases(plan).find((candidate) => candidate.id === record.caseId);
  if (!item) {
    reasons.push(`case ${record.caseId} is not in the plan`);
  } else {
    const expected = caseIdentity(item, manifest);
    for (const part of ["model", "decoder"]) {
      for (const [field, value] of Object.entries(expected[part])) {
        if (record.identity[part][field] !== value) {
          reasons.push(`${part}.${field} ${JSON.stringify(record.identity[part][field])} is now ${JSON.stringify(value)}`);
        }
      }
    }
  }
  if (record.identity.sceneworks.dirty) reasons.push("captured from a dirty SceneWorks checkout");
  return { status: reasons.length ? "stale" : "current", reasons };
}

/**
 * Measured vs estimated, per stage, for a completed record. A unified (Metal) pool compares the
 * process footprint with the stage's total; a CUDA card compares `nvidia-smi` used memory (which
 * includes the CUDA context) with the stage's device bytes plus the allocator reserve admission
 * charges on top.
 */
export function coverage(record) {
  if (record.outcome.status !== "completed") return [];
  const stages = record.admission.estimate?.stages ?? {};
  return Object.entries(record.measured.stages).map(([stage, measured]) => {
    const estimate = stages[stage];
    const estimatedBytes = !estimate
      ? null
      : record.backend === "cuda"
        ? estimate.deviceBytes + CUDA_RESERVE_BYTES
        : estimate.totalBytes;
    return {
      stage,
      measuredBytes: measured.peakBytes,
      estimatedBytes,
      ratio: estimatedBytes ? measured.peakBytes / estimatedBytes : null,
      covered: estimatedBytes !== null && measured.peakBytes <= estimatedBytes,
    };
  });
}

/** Per case: the current record the corpus holds, or why it needs a capture. */
export function corpusStatus(plan, records, sources) {
  return expandCases(plan).map((item) => {
    const mine = records.filter((record) => record.caseId === item.id);
    const graded = mine.map((record) => ({ record, grade: gradeRecord(record, sources) }));
    const current = graded.find(({ grade }) => grade.status === "current");
    return {
      id: item.id,
      backend: item.backend,
      status: current ? "current" : graded.length ? "stale" : "missing",
      reasons: current ? [] : graded.flatMap(({ grade }) => grade.reasons),
    };
  });
}

export async function readCorpus(root = ROOT, corpus) {
  const dir = path.join(root, corpus);
  let names = [];
  try {
    names = await readdir(dir);
  } catch (error) {
    if (error.code === "ENOENT") return [];
    throw error;
  }
  const records = [];
  for (const name of names.filter((file) => file.endsWith(".json")).sort()) {
    records.push(JSON.parse(await readFile(path.join(dir, name), "utf8")));
  }
  return records;
}

/** The corpus file name a record is ingested under. */
export function corpusFileName(record) {
  return `${record.caseId.replaceAll(":", "__")}.json`;
}

/** Refuse anything that is not current, complete evidence. */
export function admitToCorpus(record, sources) {
  const grade = gradeRecord(record, sources);
  if (grade.status !== "current") {
    fail(`rejected ${record.caseId}: ${grade.reasons.join("; ")}`);
  }
  if (record.outcome.status === "failed") fail(`rejected ${record.caseId}: the capture failed`);
  return record;
}

// ---- Capture -----------------------------------------------------------------------------------------

function run(command, args, options = {}) {
  return execFileSync(command, args, { encoding: "utf8", ...options }).trim();
}

export function repositoryIdentity(root = ROOT) {
  return {
    revision: run("git", ["-C", root, "rev-parse", "HEAD"]),
    dirty: run("git", ["-C", root, "status", "--porcelain"]).length > 0,
  };
}

/** The engine identity: the pinned inference revision and its `candle:yue2` closure digest. */
export async function engineIdentity({ root = ROOT, inferenceRepo, closures, lane }) {
  const revision = inferencePinFromCargo(await readFile(path.join(root, "Cargo.toml"), "utf8"));
  const declared = closures.providers[lane];
  const derived = providerClosureDigest({
    repo: path.resolve(inferenceRepo),
    revision,
    provider: lane,
    crateDir: declared.crate,
  });
  return {
    lane,
    inferenceRevision: revision,
    closureDigest: derived.digest,
    digestVersion: closures.digestVersion,
  };
}

/** The native entrypoint's cargo invocation (`--no-run` compiles it first, outside any guard). */
export function cargoTestArgv(backend, { noRun = false } = {}) {
  return [
    "test", "--release", "-p", "sceneworks-worker", "--lib",
    ...(backend === "cuda" ? ["--features", "backend-candle"] : []),
    ...(noRun ? ["--no-run"] : []),
    "yue2_memory_profile::capture_case",
    ...(noRun ? [] : ["--", "--ignored", "--exact", "--nocapture", "--test-threads", "1"]),
  ];
}

export function captureEnv({ caseFilePath, outDir, dataDir, gpuId, base = process.env }) {
  return {
    ...base,
    SCENEWORKS_ENABLE_YUE2_MEMORY_PROFILE: "1",
    SCENEWORKS_YUE2_PROFILE_CASE: caseFilePath,
    SCENEWORKS_YUE2_PROFILE_OUT: outDir,
    ...(dataDir ? { SCENEWORKS_DATA_DIR: dataDir } : {}),
    ...(gpuId !== undefined ? { SCENEWORKS_GPU_ID: String(gpuId) } : {}),
  };
}

function probeHardware(backend, gpuId) {
  if (backend === "metal") {
    return {
      memoryBytes: Number(run("sysctl", ["-n", "hw.memsize"])),
      chip: run("sysctl", ["-n", "machdep.cpu.brand_string"]),
      os: `${os.type()} ${os.release()}`,
    };
  }
  const [name, totalMb, computeCap, driver] = run("nvidia-smi", [
    "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader,nounits",
    "-i", String(gpuId),
  ]).split(",").map((part) => part.trim());
  return { gpu: name, memoryBytes: Number(totalMb) * 1024 * 1024, computeCap, driver, os: `${os.type()} ${os.release()}` };
}

/** Sample the card's used memory every `intervalMs` until stopped (CUDA). */
function startNvidiaSampler(gpuId, intervalMs = 250) {
  const samples = [];
  const timer = setInterval(() => {
    try {
      const used = Number(run("nvidia-smi", [
        "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", String(gpuId),
      ]));
      if (Number.isFinite(used)) samples.push({ at: Date.now() / 1000, bytes: used * 1024 * 1024 });
    } catch {
      // A missed tick is a gap in the series, never a failed capture.
    }
  }, intervalMs);
  return { samples, stop: () => clearInterval(timer) };
}

function spawnAndWait(command, args, options) {
  return new Promise((resolve) => {
    const child = spawn(command, args, { stdio: "inherit", ...options });
    child.on("exit", (code, signal) => resolve({ code, signal }));
    child.on("error", (error) => resolve({ code: -1, signal: null, error }));
  });
}

async function readOptional(file) {
  try {
    return await readFile(file, "utf8");
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
}

/** Plan one capture: the case, its files and the exact commands, without running anything. */
export async function planCapture({ caseId, outDir, dataDir, gpuId, budgetMinutes = 240, sources, root = ROOT }) {
  const { plan, manifest } = sources;
  validatePlan(plan, sources);
  const item = expandCases(plan).find((candidate) => candidate.id === caseId);
  if (!item) fail(`unknown case ${caseId}; see \`plan\``);
  const resolvedOut = path.resolve(outDir, item.id.replaceAll(":", "__"));
  if (!path.relative(root, resolvedOut).startsWith("..")) {
    fail("--out must be outside the repository: a capture from a dirty checkout is not evidence");
  }
  const caseFilePath = path.join(resolvedOut, "case.json");
  const env = captureEnv({ caseFilePath, outDir: resolvedOut, dataDir, gpuId });
  const test = ["cargo", ...cargoTestArgv(item.backend)];
  const guarded = item.backend === "metal"
    ? { eventFile: path.join(resolvedOut, "watchdog.jsonl"), budgetMinutes }
    : null;
  return {
    item,
    identity: caseIdentity(item, manifest),
    outDir: resolvedOut,
    caseFilePath,
    env,
    compile: ["cargo", ...cargoTestArgv(item.backend, { noRun: true })],
    test,
    guarded,
  };
}

/** Run one capture and write its record to `<out>/<case>/record.json`. */
export async function captureCase(options) {
  const sources = options.sources ?? await readSources();
  const planned = await planCapture({ ...options, sources });
  const { item } = planned;
  await mkdir(planned.outDir, { recursive: true });
  await writeFile(planned.caseFilePath, `${JSON.stringify(caseFile(item), null, 2)}\n`);
  const sceneworks = repositoryIdentity();
  if (sceneworks.dirty) fail("the SceneWorks checkout is dirty; commit before capturing");
  const engine = await engineIdentity({
    inferenceRepo: options.inferenceRepo, closures: sources.closures, lane: sources.plan.lane,
  });
  const hardware = probeHardware(item.backend, options.gpuId ?? 0);
  // Compile outside the guard so the build is never sampled or killed as the render.
  const built = await spawnAndWait(planned.compile[0], planned.compile.slice(1), { cwd: ROOT, env: planned.env });
  if (built.code !== 0) fail(`${item.id}: the capture entrypoint does not build`);
  let sampler;
  let samples = [];
  let result;
  if (planned.guarded) {
    const { watchdogGuard } = await import("./measure-memory-catalog.mjs");
    const guard = watchdogGuard({
      hardware: { memoryBytes: hardware.memoryBytes },
      eventFile: planned.guarded.eventFile,
      budgetMinutes: planned.guarded.budgetMinutes,
    });
    sampler = "memory-calibration-watchdog phys_footprint";
    result = await spawnAndWait("python3", [...guard, "--", ...planned.test], { cwd: ROOT, env: planned.env });
    samples = parseWatchdogSamples((await readOptional(planned.guarded.eventFile)) ?? "");
  } else {
    sampler = "nvidia-smi memory.used";
    const nvidia = startNvidiaSampler(options.gpuId ?? 0);
    result = await spawnAndWait(planned.test[0], planned.test.slice(1), { cwd: ROOT, env: planned.env });
    nvidia.stop();
    samples = nvidia.samples;
  }
  const admissionBody = await readOptional(path.join(planned.outDir, "admission.json"));
  const admission = admissionBody ? JSON.parse(admissionBody) : { outcome: "unknown" };
  const outcomeBody = await readOptional(path.join(planned.outDir, "outcome.json"));
  const marksBody = await readOptional(path.join(planned.outDir, "stages.jsonl"));
  let outcome;
  if (admission.outcome === "refused") outcome = { status: "refused", message: admission.message };
  else if (result.code === 0 && outcomeBody) outcome = { status: "completed", ...JSON.parse(outcomeBody) };
  else outcome = { status: "failed", exit: result.code, signal: result.signal };
  const record = buildRecord({
    item,
    identity: planned.identity,
    engine,
    sceneworks,
    hardware,
    admission,
    outcome,
    sampler,
    samples,
    marks: marksBody ? parseStageMarks(marksBody) : null,
    capturedAt: new Date().toISOString(),
  });
  validateRecord(record);
  const recordPath = path.join(planned.outDir, "record.json");
  await writeFile(recordPath, `${JSON.stringify(record, null, 2)}\n`);
  return { record, recordPath };
}

// ---- CLI ---------------------------------------------------------------------------------------------

export function parseArgs(argv) {
  const [command, ...rest] = argv;
  const options = { command, files: [] };
  for (let index = 0; index < rest.length; index += 1) {
    const arg = rest[index];
    const value = () => {
      const next = rest[index + 1];
      if (next === undefined || next.startsWith("--")) fail(`${arg} needs a value`);
      index += 1;
      return next;
    };
    if (arg === "--case") options.caseId = value();
    else if (arg === "--backend") options.backend = value();
    else if (arg === "--inference-repo") options.inferenceRepo = value();
    else if (arg === "--out") options.outDir = value();
    else if (arg === "--data-dir") options.dataDir = value();
    else if (arg === "--gpu-id") options.gpuId = Number(value());
    else if (arg === "--budget-minutes") options.budgetMinutes = Number(value());
    else if (arg === "--dry-run") options.dryRun = true;
    else if (arg.startsWith("--")) fail(`unknown option ${arg}`);
    else options.files.push(arg);
  }
  if (options.backend !== undefined && !BACKENDS.includes(options.backend)) fail(`unknown backend ${options.backend}`);
  return options;
}

async function readRecords(files) {
  return Promise.all(files.map(async (file) => JSON.parse(await readFile(file, "utf8"))));
}

function formatBytes(bytes) {
  return bytes === null || bytes === undefined ? "—" : `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
}

export async function main(argv = process.argv.slice(2), root = ROOT) {
  const options = parseArgs(argv);
  const sources = await readSources(root);
  validatePlan(sources.plan, sources);
  const corpus = sources.plan.corpus;
  switch (options.command) {
    case "plan": {
      const status = corpusStatus(sources.plan, await readCorpus(root, corpus), sources);
      for (const row of status.filter((item) => !options.backend || item.backend === options.backend)) {
        console.log(`${row.id.padEnd(36)} ${row.status}${row.reasons.length ? `  (${row.reasons.join("; ")})` : ""}`);
      }
      for (const note of sources.plan.unreachable ?? []) console.log(`unreachable: ${note.what} — ${note.why}`);
      return 0;
    }
    case "capture": {
      if (!options.caseId || !options.outDir) fail("capture needs --case and --out");
      if (options.dryRun) {
        const planned = await planCapture({ ...options, sources, root });
        console.log(JSON.stringify({ case: caseFile(planned.item), identity: planned.identity, compile: planned.compile, test: planned.test, guarded: planned.guarded }, null, 2));
        return 0;
      }
      if (!options.inferenceRepo) fail("capture needs --inference-repo (the pinned inference checkout)");
      const { record, recordPath } = await captureCase({ ...options, sources });
      console.log(`${record.caseId}: ${record.outcome.status} → ${recordPath}`);
      return record.outcome.status === "failed" ? 1 : 0;
    }
    case "run": {
      if (!options.backend || !options.outDir) fail("run needs --backend and --out");
      const status = corpusStatus(sources.plan, await readCorpus(root, corpus), sources);
      const todo = status.filter((row) => row.backend === options.backend && row.status !== "current");
      console.log(`${todo.length} case(s) to capture on ${options.backend}; ${status.filter((row) => row.backend === options.backend).length - todo.length} already current`);
      let failures = 0;
      for (const row of todo) {
        if (options.dryRun) {
          console.log(`would capture ${row.id}`);
          continue;
        }
        if (!options.inferenceRepo) fail("run needs --inference-repo");
        try {
          const { record, recordPath } = await captureCase({ ...options, caseId: row.id, sources });
          console.log(`${record.caseId}: ${record.outcome.status} → ${recordPath}`);
          if (record.outcome.status === "failed") failures += 1;
        } catch (error) {
          failures += 1;
          console.error(`${row.id}: ${error.message}`);
        }
      }
      console.log(`ingest the records: node ${sources.plan.harness} ingest ${options.outDir}/*/record.json`);
      return failures ? 1 : 0;
    }
    case "check": {
      let stale = 0;
      for (const record of await readRecords(options.files)) {
        const grade = gradeRecord(record, sources);
        if (grade.status !== "current") stale += 1;
        console.log(`${record.caseId}: ${grade.status} ${record.outcome.status}${grade.reasons.length ? ` — ${grade.reasons.join("; ")}` : ""}`);
        for (const row of coverage(record)) {
          console.log(
            `  ${row.stage.padEnd(9)} measured ${formatBytes(row.measuredBytes)}  estimated ${formatBytes(row.estimatedBytes)}  ` +
              `${row.covered ? "covered" : "UNDER-PRICED"}${row.ratio ? ` (${row.ratio.toFixed(2)}×)` : ""}`,
          );
        }
      }
      return stale ? 1 : 0;
    }
    case "ingest": {
      const dir = path.join(root, corpus);
      await mkdir(dir, { recursive: true });
      for (const record of await readRecords(options.files)) {
        admitToCorpus(record, sources);
        await writeFile(path.join(dir, corpusFileName(record)), `${JSON.stringify(record, null, 2)}\n`);
        console.log(`ingested ${record.caseId}`);
      }
      return 0;
    }
    default:
      fail("usage: plan | capture | run | check | ingest (see the header of this script)");
  }
  return 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().then(
    (code) => {
      process.exitCode = code;
    },
    (error) => {
      console.error(error.message);
      process.exitCode = 1;
    },
  );
}
