#!/usr/bin/env node
// Fixed-purpose, Windows-only StarVector CUDA diagnostic. This exercises the
// production product route with sealed local inputs, but cannot author terminal
// acceptance, preflight, or performance evidence.
import { createHash } from "node:crypto";
import { execFile as execFileCallback } from "node:child_process";
import { lstat, mkdir, readFile, readdir, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { isExecutedModule } from "./starvector-terminal-cli.mjs";
import { stripJsoncComments } from "./lib/jsonc.mjs";
import { terminalSourceRowsSha256 } from "./starvector-terminal-campaign.mjs";
import { acquireStableLease } from "./starvector-terminal-producer.mjs";
import { importTupleAssets } from "./starvector-terminal-assets.mjs";
import { assertTerminalProductWorkerReady, startProductService, stopProductService } from "./starvector-terminal-product-service.mjs";
import { preserveTerminalDiagnostics, submitAndPoll, vectorRequest } from "./starvector-terminal-route.mjs";

export const DIAGNOSTIC_TUPLE = "candle-cuda:1b";
export const DIAGNOSTIC_CASES = Object.freeze([6, 9, 11, 12, 13, 15]);
export const DIAGNOSTIC_BUDGET = Object.freeze({ maxNewTokens: 7933, maxSvgBytes: 262144, maxWallTimeMs: 120000 });
export const DIAGNOSTIC_STOP = Object.freeze({ accepted: 3, rejected: 4 });
export const DIAGNOSTIC_LEGACY_CORPUS = Object.freeze({
  schema_version: 1,
  inference_revision: "252e6a7a0fee08d2e07b0e8652234e4cea28aa59",
  row_identity_sha256: "f9529c2e5a86bef6644054c909c4f621991f6384d9b33a029ad46ff2e6cd3b88",
  index_sha256: "dbfd6b6ef972f3104f7a20c2033af6157b720e49adf46c506e0b34ef612d3a59",
  readiness: Object.freeze({ workflow_run_id: "34914687576", workflow_run_attempt: 1, artifact_id: 10375713957, artifact_sha256: "2af3f6e6154c218d854022a95ae641f687b3f3b7cc4d5bbd9c30488f1f373796" }),
});
const MODEL = Object.freeze({
  id: "starvector_1b",
  provider: "candle-starvector-1b",
  backend: "candle",
  repository: "starvector/starvector-1b-im2svg",
  revision: "380ab95d25a8e9ab1dc825debe238b4953ae13b9",
});
const REVISION = /^[a-f0-9]{40}$/;
const SHA256 = /^[a-f0-9]{64}$/;
const execFile = promisify(execFileCallback);
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const stable = (value) => Array.isArray(value) ? `[${value.map(stable).join(",")}]` : value && typeof value === "object" ? `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(",")}}` : JSON.stringify(value);
const die = (message) => { throw new Error(`starvector terminal diagnostic: ${message}`); };

function confined(root, target) {
  const relative = path.relative(path.resolve(root), path.resolve(target));
  return relative && relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

export function validateDiagnosticInvocation({ root, output, weightsRoot, corpusAssetsRoot, permanentPin, corpusSourcePin, expectedSceneWorksRevision, leaseRoot, leaseHelper, platform = process.platform, runnerTemp = process.env.RUNNER_TEMP, noJobDownloads = process.env.STARVECTOR_TERMINAL_NO_JOB_DOWNLOADS, gpuId = process.env.STARVECTOR_TERMINAL_GPU_ID }) {
  if (platform !== "win32") die("this product-route diagnostic requires Windows CUDA");
  for (const [label, value] of [["SceneWorks root", root], ["output", output], ["weights root", weightsRoot], ["corpus assets root", corpusAssetsRoot], ["lease root", leaseRoot], ["lease helper", leaseHelper]]) if (typeof value !== "string" || !path.isAbsolute(value)) die(`${label} must be absolute`);
  if (!REVISION.test(permanentPin ?? "") || !REVISION.test(corpusSourcePin ?? "") || !REVISION.test(expectedSceneWorksRevision ?? "")) die("exact source, inference, and corpus revisions are required");
  if (!runnerTemp || !path.isAbsolute(runnerTemp) || !confined(runnerTemp, output)) die("diagnostic output must be a confined runner temporary path");
  if (noJobDownloads !== "1") die("no-job-downloads guard is required");
  if (gpuId !== "0") die("the fixed CUDA diagnostic requires GPU 0");
  return { root: path.resolve(root), output: path.resolve(output), weightsRoot: path.resolve(weightsRoot), corpusAssetsRoot: path.resolve(corpusAssetsRoot), permanentPin, corpusSourcePin, expectedSceneWorksRevision, leaseRoot: path.resolve(leaseRoot), leaseHelper: path.resolve(leaseHelper) };
}

export function validateDiagnosticWeightsManifest(manifest) {
  const model = manifest?.models?.["starvector-1b"];
  if (manifest?.schema_version !== 1 || model?.relative_path !== "models/starvector-1b" || model?.revision !== MODEL.revision || !SHA256.test(model?.inventory_sha256 ?? "")) die("selected Windows weights identity is invalid");
  return model.inventory_sha256;
}

export function validateDiagnosticService(service, { sceneWorksRevision, permanentPin, modelInventory }) {
  const model = service?.models?.["starvector-1b"];
  const worker = service?.worker;
  if (service?.sceneworks_revision !== sceneWorksRevision || service?.inference_revision !== permanentPin || service?.tuple !== DIAGNOSTIC_TUPLE) die("product service source identity drifted");
  if (!SHA256.test(modelInventory ?? "") || model?.revision !== MODEL.revision || model?.inventory_sha256 !== modelInventory) die("product service model inventory drifted");
  if (worker?.model_id !== MODEL.id || worker?.provider_id !== MODEL.provider || worker?.backend !== MODEL.backend || worker?.gpu_id !== "0" || worker?.status !== "idle" || worker?.current_job_id != null) die("product worker route identity drifted");
  if (!SHA256.test(service.api_binary_sha256 ?? "") || !service.api_url?.startsWith("http://127.0.0.1:")) die("product service binary or endpoint identity is missing");
  return service;
}

export async function readCurrentInferencePin(root) {
  const cargo = await readFile(path.join(root, "Cargo.toml"), "utf8");
  const pins = [...cargo.matchAll(/SceneWorks\/inference",\s*rev\s*=\s*"([a-f0-9]{40})"/g)].map((match) => match[1]);
  if (pins.length === 0 || new Set(pins).size !== 1) die("current SceneWorks source does not have one exact inference pin");
  return pins[0];
}

export function validateDiagnosticManifest(manifest) {
  const selected = manifest?.models?.filter((entry) => entry?.id === MODEL.id);
  if (!Array.isArray(selected) || selected.length !== 1) die("built-in manifest does not uniquely select StarVector 1B");
  const entry = selected[0], vector = entry.vector, download = entry.downloads?.find((item) => item?.provider === "huggingface");
  if (entry.type !== "vector" || entry.adapter !== "starvector" || vector?.providers?.candle?.id !== MODEL.provider || vector.providers.candle.available !== true) die("built-in manifest product route identity drifted");
  if (vector.maxNewTokens !== DIAGNOSTIC_BUDGET.maxNewTokens || vector.maxSvgBytes !== DIAGNOSTIC_BUDGET.maxSvgBytes || vector.maxWallTimeMs !== DIAGNOSTIC_BUDGET.maxWallTimeMs) die("built-in manifest Detailed budget drifted");
  if (download?.repo !== MODEL.repository || download.revision !== MODEL.revision || !Array.isArray(download.files) || download.files.length === 0) die("built-in manifest model snapshot identity drifted");
  if (vector.deviceAdmission?.staticWeightFloorBytes !== 5142705320 || vector.deviceAdmission?.deviceClasses?.candle !== "nvidia_dedicated_vram") die("built-in manifest CUDA admission identity drifted");
  return { maxNewTokens: vector.maxNewTokens, maxSvgBytes: vector.maxSvgBytes, maxWallTimeMs: vector.maxWallTimeMs };
}

export function parseDiagnosticCudaGpu(stdout) {
  const lines = String(stdout).trim().split(/\r?\n/).filter(Boolean);
  if (lines.length !== 1) die("CUDA occupancy probe must return exactly GPU 0");
  const [index, uuid, utilization, total, free, used, ...extra] = lines[0].split(",").map((item) => item.trim());
  const values = [utilization, total, free, used].map(Number);
  if (extra.length || index !== "0" || !/^GPU-[a-f0-9-]+$/i.test(uuid ?? "") || values.some((value) => !Number.isFinite(value) || value < 0) || values[0] > 100 || values[2] > values[1] || values[3] > values[1]) die("CUDA occupancy GPU sample is malformed");
  return { index, uuid, utilization_percent: values[0], total_bytes: values[1] * 1024 * 1024, free_bytes: values[2] * 1024 * 1024, used_bytes: values[3] * 1024 * 1024 };
}

export function parseDiagnosticCudaProcesses(stdout, expectedUuid, ownedPids = []) {
  const owned = new Set(ownedPids.filter((pid) => Number.isSafeInteger(pid) && pid > 0));
  const body = String(stdout).trim();
  if (!body || /^no running processes found\.?$/i.test(body)) return [];
  const processes = body.split(/\r?\n/).filter(Boolean).map((line) => {
    const [uuid, pidText, name, usedText, ...extra] = line.split(",").map((item) => item.trim());
    const pid = Number(pidText), usedMib = /^\d+$/.test(usedText) ? Number(usedText) : null;
    if (extra.length || uuid !== expectedUuid || !Number.isSafeInteger(pid) || pid <= 0 || !name || (usedMib !== null && !Number.isSafeInteger(usedMib))) die("CUDA occupancy process sample is malformed");
    return { gpu_uuid: uuid, pid, process_name: name, used_gpu_memory_bytes: usedMib === null ? null : usedMib * 1024 * 1024, owned: owned.has(pid) };
  });
  if (new Set(processes.map((item) => item.pid)).size !== processes.length) die("CUDA occupancy process identities are duplicated");
  return processes;
}

export async function captureDiagnosticCudaOccupancy({ ownedPids = [], execFileImpl = execFile, wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds)) } = {}) {
  const samples = [];
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const gpuResult = await execFileImpl("nvidia-smi", ["--id=0", "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"], { timeout: 10_000, maxBuffer: 64 * 1024, windowsHide: true });
    const gpu = parseDiagnosticCudaGpu(gpuResult.stdout);
    const processResult = await execFileImpl("nvidia-smi", ["--id=0", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], { timeout: 10_000, maxBuffer: 64 * 1024, windowsHide: true });
    const processes = parseDiagnosticCudaProcesses(processResult.stdout, gpu.uuid, ownedPids);
    samples.push({ observed_at: new Date().toISOString(), gpu, processes });
    if (attempt < 2) await wait(500);
  }
  if (new Set(samples.map((sample) => sample.gpu.uuid)).size !== 1) die("CUDA occupancy GPU identity changed across samples");
  const foreign = samples.flatMap((sample) => sample.processes.filter((item) => !item.owned));
  // Before the service starts, GPU activity alongside a foreign process is
  // attributable and defers the diagnostic. Once our worker exists, aggregate
  // utilization cannot distinguish its kernels from an idle foreign context;
  // keep functional evidence but mark all timing unusable instead.
  const activityAttributable = ownedPids.length === 0;
  const activeForeignCompute = activityAttributable && foreign.length > 0 && samples.some((sample) => sample.gpu.utilization_percent > 5);
  const enoughFreeMemory = samples.every((sample) => sample.gpu.free_bytes >= 5142705320);
  return { schema_version: 1, samples, foreign_processes_observed: foreign.length > 0, activity_attribution: activityAttributable ? "foreign_only_before_service" : "ambiguous_with_owned_service", active_foreign_compute: activeForeignCompute, enough_free_memory_for_static_weight_floor: enoughFreeMemory, functional_execution_allowed: !activeForeignCompute && enoughFreeMemory, timing_usable_for_performance: foreign.length === 0 && !activeForeignCompute };
}

export function selectDiagnosticRecordView(index, indexBytes, binding, corpusSourcePin, { legacyCorpus = DIAGNOSTIC_LEGACY_CORPUS } = {}) {
  if (!Buffer.isBuffer(indexBytes) || indexBytes.length === 0 || index?.inference_revision !== corpusSourcePin || !Array.isArray(index.rows) || index.rows.length !== 120) die("immutable corpus row index identity/count drifted");
  const indexSha256 = sha(indexBytes), rowIdentity = terminalSourceRowsSha256(index.rows);
  if (!SHA256.test(index.row_identity_sha256 ?? "") || rowIdentity !== index.row_identity_sha256) die("immutable corpus row identities drifted");
  const legacy = index.schema_version === 1;
  if (legacy) {
    if (legacyCorpus?.schema_version !== 1 || legacyCorpus.inference_revision !== corpusSourcePin || legacyCorpus.row_identity_sha256 !== index.row_identity_sha256 || legacyCorpus.index_sha256 !== indexSha256 || legacyCorpus.readiness?.workflow_run_id !== "34914687576" || legacyCorpus.readiness.workflow_run_attempt !== 1 || legacyCorpus.readiness.artifact_id !== 10375713957 || legacyCorpus.readiness.artifact_sha256 !== "2af3f6e6154c218d854022a95ae641f687b3f3b7cc4d5bbd9c30488f1f373796") die("legacy corpus index is not the authenticated Windows readiness input");
  } else if (index.schema_version !== 2) {
    die("immutable corpus row index schema drifted");
  }
  for (const [position, row] of index.rows.entries()) {
    if (row?.case_index !== position || row.row_index !== position % 30 || typeof row.dataset !== "string" || !row.dataset || !REVISION.test(row.revision ?? "") || typeof row.filename !== "string" || !row.filename || !SHA256.test(row.svg_sha256 ?? "") || !SHA256.test(row.png_sha256 ?? "") || !SHA256.test(row.reference_png_sha256 ?? "") || typeof row.input_png_path !== "string" || !row.input_png_path) die(`diagnostic corpus row ${position} immutable identity drifted`);
    const budget = legacy ? row.detail_budget : row.detail_budgets?.["1b"];
    const expected = legacy ? { maxNewTokens: 4000, maxSvgBytes: 262144, maxWallTimeMs: 120000 } : DIAGNOSTIC_BUDGET;
    if (stable(budget) !== stable(expected)) die(`diagnostic corpus row ${position} source budget drifted`);
  }
  if (!binding?.project_id || binding.tuple !== DIAGNOSTIC_TUPLE || !Array.isArray(binding.assets) || binding.assets.length !== 120 || binding.aggregate_sha256 !== sha(JSON.stringify(binding.assets))) die("tuple-local imported asset binding drifted");
  const assets = new Map(binding.assets.map((entry) => [entry.case_index, entry]));
  if (assets.size !== 120) die("tuple-local imported assets are not unique");
  const records = DIAGNOSTIC_CASES.map((caseIndex) => {
    const row = index.rows[caseIndex], asset = assets.get(caseIndex), budget = legacy ? DIAGNOSTIC_BUDGET : row?.detail_budgets?.["1b"];
    if (row?.case_index !== caseIndex || !SHA256.test(row.png_sha256 ?? "") || !row.input_png_path || !asset?.asset_id || asset.input_png_sha256 !== row.png_sha256) die(`diagnostic case ${caseIndex} asset identity drifted`);
    if (stable(budget) !== stable(DIAGNOSTIC_BUDGET)) die(`diagnostic case ${caseIndex} does not use the shipping Detailed budget`);
    if (!row.sampling || typeof row.sampling !== "object" || Array.isArray(row.sampling)) die(`diagnostic case ${caseIndex} sampling identity is missing`);
    return { case_id: `diagnostic-quality-v1-${caseIndex}`, case_index: caseIndex, projectId: binding.project_id, sourceAssetId: asset.asset_id, model: MODEL.id, input_png_sha256: row.png_sha256, sampling: row.sampling, detailBudget: { ...budget } };
  });
  return {
    records,
    provenance: {
      source_schema_version: index.schema_version,
      source_index_sha256: indexSha256,
      source_row_identity_sha256: rowIdentity,
      view: legacy ? "authenticated_legacy_rows_with_current_diagnostic_budget" : "current_schema_rows",
      request_detail_budget: { ...DIAGNOSTIC_BUDGET },
      readiness: legacy ? { ...legacyCorpus.readiness } : null,
    },
  };
}

export function selectDiagnosticRecords(index, binding, corpusSourcePin, options = {}) {
  const indexBytes = options.indexBytes ?? Buffer.from(`${JSON.stringify(index, null, 2)}\n`);
  return selectDiagnosticRecordView(index, indexBytes, binding, corpusSourcePin, options).records;
}

export function validateDiagnosticOutcome(record, job) {
  const item = job?.result?.terminalEvidence ?? job?.terminalEvidence;
  const expected = vectorRequest(record), payload = job?.payload;
  if (job?.status !== "completed" || job.type !== "vector_generate" || !payload || payload.mode !== expected.mode || payload.projectId !== expected.projectId || (payload.projectName ?? null) !== (expected.projectName ?? null) || payload.sourceAssetId !== expected.sourceAssetId || payload.model !== expected.model || payload.prompt !== expected.prompt || stable(payload.sampling) !== stable(expected.sampling) || stable(payload.detailBudget) !== stable(expected.detailBudget)) die(`case ${record.case_index} stored job payload differs from the exact request`);
  validateDiagnosticManifest({ models: [payload.modelManifestEntry] });
  if (!item || item.providerId !== MODEL.provider || item.backend !== MODEL.backend || item.modelId !== MODEL.id || item.modelRepository !== MODEL.repository || item.modelRevision !== MODEL.revision || item.sourceRasterSha256 !== record.input_png_sha256) die(`case ${record.case_index} terminal identity drifted`);
  if (typeof item.providerTranscriptPath !== "string" || !SHA256.test(item.providerTranscriptSha256 ?? "")) die(`case ${record.case_index} provider transcript identity is missing`);
  if (!Number.isSafeInteger(item.generatedTokens) || item.generatedTokens < 0 || item.generatedTokens > DIAGNOSTIC_BUDGET.maxNewTokens || !Number.isSafeInteger(item.generatedBytes) || item.generatedBytes < 0 || item.generatedBytes > DIAGNOSTIC_BUDGET.maxSvgBytes || typeof item.latencySeconds !== "number" || item.latencySeconds < 0) die(`case ${record.case_index} terminal bounds are invalid`);
  const accepted = item.accepted === true;
  if (accepted) {
    if (!["complete_root", "eos"].includes(item.finishReason) || !SHA256.test(item.canonicalSvgSha256 ?? "") || !SHA256.test(item.previewPngSha256 ?? "") || !item.canonicalSvgPath || !item.previewPngPath) die(`case ${record.case_index} accepted outcome is incomplete`);
  } else {
    const typedLimit = item.rejectionStage === "generation_limit" && ["token_limit", "byte_limit", "wall_time_limit"].includes(item.finishReason);
    const sanitized = item.rejectionStage === "sanitizer" && ["complete_root", "eos"].includes(item.finishReason);
    if ((!typedLimit && !sanitized) || !item.rejectionCode || item.canonicalSvgPath != null || item.previewPngPath != null) die(`case ${record.case_index} rejected outcome is not a sealed non-conversion`);
  }
  return { accepted, evidence: item };
}

export function diagnosticShouldStop(results) {
  const accepted = results.filter((item) => item.accepted).length;
  const rejected = results.length - accepted;
  return accepted >= DIAGNOSTIC_STOP.accepted || rejected >= DIAGNOSTIC_STOP.rejected;
}

async function materializeAccepted(output, record, evidence) {
  const files = [["canonical.svg", evidence.canonicalSvgPath, evidence.canonicalSvgSha256, DIAGNOSTIC_BUDGET.maxSvgBytes], ["preview.png", evidence.previewPngPath, evidence.previewPngSha256, 16 * 1024 * 1024], ["provider-transcript.json", evidence.providerTranscriptPath, evidence.providerTranscriptSha256, 2 * 1024 * 1024]];
  const artifacts = [];
  for (const [name, source, expected, maxBytes] of files) {
    const info = await lstat(source);
    if (!info.isFile() || info.isSymbolicLink() || info.size < 1 || info.size > maxBytes) die(`case ${record.case_index} accepted ${name} is not a bounded regular file`);
    const bytes = await readFile(source); if (sha(bytes) !== expected) die(`case ${record.case_index} accepted ${name} hash drifted`);
    const destination = path.join(output, "accepted", String(record.case_index), name); await mkdir(path.dirname(destination), { recursive: true }); await writeFile(destination, bytes, { flag: "wx" });
    const role = name === "canonical.svg" ? "canonical_svg" : name === "preview.png" ? "preview_png" : "provider_transcript";
    artifacts.push({ role, path: path.relative(output, destination).split(path.sep).join("/"), bytes: info.size, sha256: expected });
  }
  return artifacts;
}

async function writeRecord(file, value) {
  const temporary = `${file}.tmp`;
  await writeFile(temporary, JSON.stringify(value, null, 2) + "\n", { mode: 0o600 });
  await rename(temporary, file);
}

function diagnosticBootstrap(inputs) {
  return { schema_version: 1, kind: "starvector_cuda_product_route_diagnostic_bootstrap", acceptance_use: "diagnostic_only", usable_for_terminal_acceptance: false, sceneworks_revision: inputs.expectedSceneWorksRevision, inference_revision: inputs.permanentPin, corpus_source_revision: inputs.corpusSourcePin, tuple: DIAGNOSTIC_TUPLE, status: "source_bootstrap", workflow: { run_id: String(process.env.GITHUB_RUN_ID ?? "local"), run_attempt: Number(process.env.GITHUB_RUN_ATTEMPT ?? 0) }, recorded_at: new Date().toISOString() };
}

export async function prepareDiagnosticOutput(options) {
  const inputs = validateDiagnosticInvocation(options);
  await mkdir(inputs.output, { recursive: false });
  const bootstrap = diagnosticBootstrap(inputs);
  await writeFile(path.join(inputs.output, "diagnostic-bootstrap.json"), JSON.stringify(bootstrap, null, 2) + "\n", { flag: "wx", mode: 0o600 });
  return bootstrap;
}

async function verifyDiagnosticBootstrap(inputs) {
  const entries = await readdir(inputs.output);
  if (JSON.stringify(entries.sort()) !== JSON.stringify(["diagnostic-bootstrap.json"])) die("diagnostic output is not a fresh prepared directory");
  const file = path.join(inputs.output, "diagnostic-bootstrap.json"), info = await lstat(file);
  if (!info.isFile() || info.isSymbolicLink()) die("diagnostic bootstrap is not a regular file");
  const observed = JSON.parse(await readFile(file, "utf8")), expected = diagnosticBootstrap(inputs);
  for (const key of ["schema_version", "kind", "acceptance_use", "usable_for_terminal_acceptance", "sceneworks_revision", "inference_revision", "corpus_source_revision", "tuple", "status"]) if (observed[key] !== expected[key]) die(`diagnostic bootstrap identity drifted at ${key}`);
  if (observed.workflow?.run_id !== expected.workflow.run_id || observed.workflow?.run_attempt !== expected.workflow.run_attempt || typeof observed.recorded_at !== "string") die("diagnostic bootstrap workflow identity drifted");
  return observed;
}

export async function runDiagnostic(options, dependencies = {}) {
  const inputs = validateDiagnosticInvocation(options);
  const modelInventory = validateDiagnosticWeightsManifest(JSON.parse(await readFile(path.join(inputs.weightsRoot, "starvector-terminal-weights-v1.json"), "utf8")));
  const apiUrl = "http://127.0.0.1:17831";
  const runId = `diagnostic-${process.env.GITHUB_RUN_ID ?? "local"}-${process.env.GITHUB_RUN_ATTEMPT ?? "0"}`;
  const acquire = dependencies.acquire ?? acquireStableLease, start = dependencies.start ?? startProductService, stop = dependencies.stop ?? stopProductService, ready = dependencies.ready ?? assertTerminalProductWorkerReady, importAssets = dependencies.importAssets ?? importTupleAssets, submit = dependencies.submit ?? submitAndPoll, preserve = dependencies.preserve ?? preserveTerminalDiagnostics, occupancy = dependencies.occupancy ?? captureDiagnosticCudaOccupancy;
  await verifyDiagnosticBootstrap(inputs);
  const resultPath = path.join(inputs.output, "diagnostic-results.json"), transcript = path.join(inputs.output, "diagnostic-route.ndjson");
  const result = { schema_version: 1, kind: "starvector_cuda_product_route_diagnostic", acceptance_use: "diagnostic_only", usable_for_terminal_acceptance: false, tuple: DIAGNOSTIC_TUPLE, sceneworks_revision: inputs.expectedSceneWorksRevision, inference_revision: inputs.permanentPin, corpus_source_revision: inputs.corpusSourcePin, model: { ...MODEL, inventory: modelInventory }, detail_budget: DIAGNOSTIC_BUDGET, selected_case_indexes: [...DIAGNOSTIC_CASES], stop_rule: DIAGNOSTIC_STOP, workflow: { run_id: String(process.env.GITHUB_RUN_ID ?? "local"), run_attempt: Number(process.env.GITHUB_RUN_ATTEMPT ?? 0) }, status: "starting", results: [] };
  await writeRecord(resultPath, result);
  let release, serviceStarted = false, primaryError;
  try {
    release = await acquire(inputs.leaseRoot, inputs.leaseHelper, inputs.permanentPin, runId);
    const manifest = JSON.parse(stripJsoncComments(await readFile(path.join(inputs.root, "config", "manifests", "builtin.models.jsonc"), "utf8")));
    validateDiagnosticManifest(manifest);
    result.cuda_occupancy_before_start = await occupancy();
    result.timing_usable_for_performance = result.cuda_occupancy_before_start.timing_usable_for_performance;
    await writeRecord(resultPath, result);
    if (!result.cuda_occupancy_before_start.functional_execution_allowed) die("CUDA occupancy does not permit uncontended functional execution");
    const service = await start({ root: inputs.root, output: inputs.output, permanentPin: inputs.permanentPin, url: apiUrl, weightsRoot: inputs.weightsRoot, tuple: DIAGNOSTIC_TUPLE }); serviceStarted = true;
    validateDiagnosticService(service, { sceneWorksRevision: inputs.expectedSceneWorksRevision, permanentPin: inputs.permanentPin, modelInventory });
    const live = await ready(apiUrl, DIAGNOSTIC_TUPLE, service.worker.worker_id);
    if (JSON.stringify(live) !== JSON.stringify(service.worker)) die("live product worker changed after startup");
    const bindingPath = path.join(inputs.output, "imported-assets.json");
    const binding = await importAssets({ assetsRoot: inputs.corpusAssetsRoot, apiUrl, tuple: DIAGNOSTIC_TUPLE, output: bindingPath });
    const indexBytes = await readFile(path.join(inputs.corpusAssetsRoot, "starvector-terminal-row-index-v1.json"));
    const index = JSON.parse(indexBytes);
    const view = selectDiagnosticRecordView(index, indexBytes, binding, inputs.corpusSourcePin, { legacyCorpus: dependencies.legacyCorpus ?? DIAGNOSTIC_LEGACY_CORPUS });
    const records = view.records;
    result.corpus = view.provenance;
    result.status = "running"; result.product_service = service; await writeRecord(resultPath, result);
    for (const record of records) {
      const job = await submit(apiUrl, record, transcript);
      const outcome = validateDiagnosticOutcome(record, job);
      const diagnosticArtifacts = outcome.accepted ? await materializeAccepted(inputs.output, record, outcome.evidence) : (await preserve(inputs.output, "image_quality", record.case_id, job))?.artifacts ?? [];
      result.results.push({ case_index: record.case_index, request: vectorRequest(record), accepted: outcome.accepted, finish_reason: outcome.evidence.finishReason, generated_tokens: outcome.evidence.generatedTokens, generated_bytes: outcome.evidence.generatedBytes, latency_seconds: outcome.evidence.latencySeconds, rejection_stage: outcome.evidence.rejectionStage ?? null, rejection_code: outcome.evidence.rejectionCode ?? null, provider_transcript_sha256: outcome.evidence.providerTranscriptSha256, artifacts: diagnosticArtifacts });
      const currentOccupancy = await occupancy({ ownedPids: [service.api_pid, service.worker_pid] });
      result.cuda_occupancy_after_cases ??= [];
      result.cuda_occupancy_after_cases.push({ case_index: record.case_index, ...currentOccupancy });
      if (currentOccupancy.foreign_processes_observed) result.timing_usable_for_performance = false;
      await writeRecord(resultPath, result);
      if (!currentOccupancy.functional_execution_allowed) { result.status = "stopped_for_resource_contention"; break; }
      if (diagnosticShouldStop(result.results)) break;
    }
    if (result.status === "running") result.status = "completed";
  } catch (error) {
    primaryError = error; result.status = "failed"; result.error = error.message;
  } finally {
    if (serviceStarted) {
      try { await stop(inputs.output); result.product_service_cleanup = "stopped_and_state_removed"; } catch (error) { result.product_service_cleanup = "failed"; result.cleanup_error = error.message; primaryError ??= error; }
    }
    if (release) {
      try { await release(); result.lease = "released"; } catch (error) { result.lease = "release_failed"; result.lease_error = error.message; primaryError ??= error; }
    }
    result.completed_at = new Date().toISOString(); await writeRecord(resultPath, result);
  }
  if (primaryError) throw primaryError;
  return result;
}

if (isExecutedModule(import.meta.url)) {
  const [command, ...args] = process.argv.slice(2);
  const invocation = () => { const [root, output, weightsRoot, corpusAssetsRoot, permanentPin, corpusSourcePin, expectedSceneWorksRevision, leaseRoot, leaseHelper] = args; return { root, output, weightsRoot, corpusAssetsRoot, permanentPin, corpusSourcePin, expectedSceneWorksRevision, leaseRoot, leaseHelper }; };
  const run = command === "source-pin" ? readCurrentInferencePin(args[0]).then((pin) => console.log(pin)) : command === "prepare" ? prepareDiagnosticOutput(invocation()).then((result) => console.log(JSON.stringify(result))) : command === "run" ? runDiagnostic(invocation()).then((result) => console.log(JSON.stringify({ status: result.status, accepted: result.results.filter((item) => item.accepted).length, rejected: result.results.filter((item) => !item.accepted).length }))) : Promise.reject(new Error("usage: source-pin <root> | prepare|run <root> <output> <weights-root> <corpus-assets-root> <pin> <corpus-pin> <sceneworks-revision> <lease-root> <lease-helper>"));
  run.catch((error) => { console.error(error.message); process.exitCode = 1; });
}
