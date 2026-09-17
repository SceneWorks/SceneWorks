import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { stripJsoncComments } from "./lib/jsonc.mjs";
import { terminalSourceRowsSha256 } from "./starvector-terminal-campaign.mjs";
import {
  DIAGNOSTIC_BUDGET,
  DIAGNOSTIC_CASES,
  DIAGNOSTIC_LEGACY_CORPUS,
  DIAGNOSTIC_SAMPLING,
  captureDiagnosticCudaOccupancy,
  diagnosticShouldStop,
  parseDiagnosticCudaGpu,
  parseDiagnosticCudaProcesses,
  prepareDiagnosticOutput,
  readCurrentInferencePin,
  runDiagnostic,
  selectDiagnosticRecordView,
  selectDiagnosticRecords,
  validateDiagnosticInvocation,
  validateDiagnosticManifest,
  validateDiagnosticOutcome,
  validateDiagnosticService,
  validateDiagnosticWeightsManifest,
} from "./starvector-terminal-diagnostic.mjs";

const sha = (character) => character.repeat(64);
const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
const revision = (character) => character.repeat(40);
const corpusPin = revision("a");
const sceneWorksRevision = revision("b");
const permanentPin = revision("c");
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

function indexFixture() {
  const index = {
    schema_version: 2,
    inference_revision: corpusPin,
    rows: Array.from({ length: 120 }, (_, case_index) => ({
      case_index,
      dataset: `starvector/source-${Math.floor(case_index / 30)}`,
      revision: revision(String(Math.floor(case_index / 30) + 1)),
      row_index: case_index % 30,
      filename: `source-${case_index}.svg`,
      svg_sha256: sha(((case_index + 1) % 10).toString()),
      input_png_path: `rows/${case_index}.png`,
      png_sha256: sha((case_index % 10).toString()),
      reference_png_sha256: sha(((case_index + 2) % 10).toString()),
      sampling: { temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1, seed: 7 },
      detail_budgets: { "1b": { ...DIAGNOSTIC_BUDGET } },
    })),
  };
  index.row_identity_sha256 = terminalSourceRowsSha256(index.rows);
  return index;
}

function legacyIndexFixture() {
  const index = indexFixture();
  index.schema_version = 1;
  for (const row of index.rows) {
    delete row.detail_budgets;
    row.detail_budget = { maxNewTokens: 4000, maxSvgBytes: 262144, maxWallTimeMs: 120000 };
  }
  return index;
}

function bindingFixture(index = indexFixture()) {
  const assets = index.rows.map((row) => ({ case_index: row.case_index, asset_id: `asset-${row.case_index}`, input_png_sha256: row.png_sha256, input_png_bytes: 10 }));
  return { project_id: "project", tuple: "candle-cuda:1b", assets, aggregate_sha256: "unused" };
}

async function sealedBinding(index = indexFixture()) {
  const { createHash } = await import("node:crypto");
  const binding = bindingFixture(index);
  binding.aggregate_sha256 = createHash("sha256").update(JSON.stringify(binding.assets)).digest("hex");
  return binding;
}

function serviceFixture() {
  return {
    sceneworks_revision: sceneWorksRevision,
    inference_revision: permanentPin,
    tuple: "candle-cuda:1b",
    api_binary_sha256: sha("d"),
    api_url: "http://127.0.0.1:17831",
    models: { "starvector-1b": { revision: "380ab95d25a8e9ab1dc825debe238b4953ae13b9", inventory_sha256: "a1ab79c36eec58747bf40e1de981497e15ecb0e2af15c62a4f033030555766e4" } },
    worker: { model_id: "starvector_1b", provider_id: "candle-starvector-1b", backend: "candle", gpu_id: "0", status: "idle", current_job_id: null },
  };
}

function weightsFixture() {
  return { schema_version: 1, models: { "starvector-1b": { relative_path: "models/starvector-1b", revision: "380ab95d25a8e9ab1dc825debe238b4953ae13b9", inventory_sha256: "a1ab79c36eec58747bf40e1de981497e15ecb0e2af15c62a4f033030555766e4" } } };
}

function manifestFixture() {
  return { models: [{ id: "starvector_1b", type: "vector", adapter: "starvector", vector: { ...DIAGNOSTIC_BUDGET, deviceAdmission: { staticWeightFloorBytes: 5142705320, deviceClasses: { candle: "nvidia_dedicated_vram" } }, providers: { candle: { id: "candle-starvector-1b", available: true } } }, downloads: [{ provider: "huggingface", repo: "starvector/starvector-1b-im2svg", revision: "380ab95d25a8e9ab1dc825debe238b4953ae13b9", files: ["config.json"] }] }] };
}

function outcomeFixture(record, accepted = true) {
  const request = { projectId: record.projectId ?? "project", projectName: null, mode: "image_to_svg", model: record.model ?? "starvector_1b", sourceAssetId: record.sourceAssetId ?? "asset-6", prompt: "", sampling: structuredClone(record.sampling ?? { seed: 7 }), detailBudget: structuredClone(record.detailBudget ?? DIAGNOSTIC_BUDGET), modelManifestEntry: manifestFixture().models[0] };
  return { status: "completed", type: "vector_generate", payload: request, result: { terminalEvidence: { accepted, finishReason: accepted ? "complete_root" : "wall_time_limit", generatedTokens: 5000, generatedBytes: 6000, latencySeconds: 119.8, providerId: "candle-starvector-1b", backend: "candle", modelId: "starvector_1b", modelRepository: "starvector/starvector-1b-im2svg", modelRevision: "380ab95d25a8e9ab1dc825debe238b4953ae13b9", sourceRasterSha256: record.input_png_sha256, providerTranscriptPath: "/tmp/provider-transcript.json", providerTranscriptSha256: sha("e"), canonicalSvgPath: accepted ? "/tmp/canonical.svg" : null, canonicalSvgSha256: accepted ? sha("f") : null, previewPngPath: accepted ? "/tmp/preview.png" : null, previewPngSha256: accepted ? sha("1") : null, rejectionStage: accepted ? null : "generation_limit", rejectionCode: accepted ? null : "wall_time_limit" } } };
}

test("fixed diagnostic invocation is Windows-only, offline, and runner-temp confined", () => {
  const options = { root: "/repo", output: "/runner/output", weightsRoot: "/weights", corpusAssetsRoot: "/corpus", permanentPin, corpusSourcePin: corpusPin, expectedSceneWorksRevision: sceneWorksRevision, leaseRoot: "/leases", leaseHelper: "/repo/lease" };
  assert.equal(validateDiagnosticInvocation({ ...options, platform: "win32", runnerTemp: "/runner", noJobDownloads: "1", gpuId: "0" }).permanentPin, permanentPin);
  for (const mutation of [
    { platform: "darwin", runnerTemp: "/runner", noJobDownloads: "1", gpuId: "0" },
    { platform: "win32", runnerTemp: "/runner", noJobDownloads: "0", gpuId: "0" },
    { platform: "win32", runnerTemp: "/runner", noJobDownloads: "1", gpuId: "1" },
    { platform: "win32", runnerTemp: "/other", noJobDownloads: "1", gpuId: "0" },
  ]) assert.throws(() => validateDiagnosticInvocation({ ...options, ...mutation }), /requires Windows CUDA|no-job-downloads|requires GPU 0|confined runner/);
});

test("diagnostic prepares authenticated failure evidence before source compilation", async () => {
  const runnerTemp = await mkdtemp(path.join(os.tmpdir(), "starvector-diagnostic-"));
  const output = path.join(runnerTemp, "output"), options = { root: path.join(runnerTemp, "repo"), output, weightsRoot: path.join(runnerTemp, "weights"), corpusAssetsRoot: path.join(runnerTemp, "corpus"), permanentPin, corpusSourcePin: corpusPin, expectedSceneWorksRevision: sceneWorksRevision, leaseRoot: path.join(runnerTemp, "leases"), leaseHelper: path.join(runnerTemp, "lease"), platform: "win32", runnerTemp, noJobDownloads: "1", gpuId: "0" };
  try {
    const value = await prepareDiagnosticOutput(options);
    assert.equal(value.status, "source_bootstrap");
    assert.equal(value.usable_for_terminal_acceptance, false);
    assert.deepEqual(JSON.parse(await readFile(path.join(output, "diagnostic-bootstrap.json"), "utf8")), value);
    await assert.rejects(() => prepareDiagnosticOutput(options), /EEXIST/);
  } finally { await rm(runnerTemp, { recursive: true, force: true }); }
});

test("selected records are the fixed former failures with the exact Detailed budget", async () => {
  const index = indexFixture(), binding = await sealedBinding(index);
  const records = selectDiagnosticRecords(index, binding, corpusPin);
  assert.deepEqual(records.map((record) => record.case_index), DIAGNOSTIC_CASES);
  assert.ok(records.every((record) => record.model === "starvector_1b" && JSON.stringify(record.detailBudget) === JSON.stringify(DIAGNOSTIC_BUDGET)));
  assert.ok(records.every((record) => JSON.stringify(record.sampling) === JSON.stringify(DIAGNOSTIC_SAMPLING)));
  for (const mutate of [
    (next) => { next.rows[DIAGNOSTIC_CASES[0]].detail_budgets["1b"].maxNewTokens = 4000; },
    (next) => { next.rows[DIAGNOSTIC_CASES[0]].png_sha256 = sha("a"); },
    (next) => { next.inference_revision = revision("f"); },
  ]) {
    const next = structuredClone(index); mutate(next);
    assert.throws(() => selectDiagnosticRecords(next, binding, corpusPin), /Detailed budget|source budget|asset identity|row (?:index )?identit/);
  }
  const wrongTuple = structuredClone(binding); wrongTuple.tuple = "mlx:1b";
  assert.throws(() => selectDiagnosticRecords(index, wrongTuple, corpusPin), /binding drifted/);
});

test("authenticated legacy Windows rows produce only a diagnostic current-budget view", async () => {
  const index = legacyIndexFixture(), indexBytes = Buffer.from(`${JSON.stringify(index, null, 2)}\n`), binding = await sealedBinding(index);
  const legacyCorpus = { ...DIAGNOSTIC_LEGACY_CORPUS, inference_revision: corpusPin, row_identity_sha256: index.row_identity_sha256, index_sha256: digest(indexBytes) };
  const view = selectDiagnosticRecordView(index, indexBytes, binding, corpusPin, { legacyCorpus });
  assert.equal(view.provenance.source_schema_version, 1);
  assert.equal(view.provenance.source_index_sha256, digest(indexBytes));
  assert.equal(view.provenance.source_row_identity_sha256, index.row_identity_sha256);
  assert.equal(view.provenance.view, "authenticated_legacy_rows_with_current_diagnostic_budget");
  assert.deepEqual(view.records.map((record) => record.case_index), DIAGNOSTIC_CASES);
  assert.ok(view.records.every((record) => record.detailBudget.maxNewTokens === 7933));
  for (const mutate of [
    (next) => { next.index_sha256 = sha("0"); },
    (next) => { next.row_identity_sha256 = sha("1"); },
    (next) => { next.readiness.artifact_sha256 = sha("2"); },
  ]) {
    const next = structuredClone(legacyCorpus); mutate(next);
    assert.throws(() => selectDiagnosticRecordView(index, indexBytes, binding, corpusPin, { legacyCorpus: next }), /authenticated Windows readiness input/);
  }
  const staleBudget = structuredClone(index); staleBudget.rows[0].detail_budget.maxNewTokens = 3999;
  const staleBytes = Buffer.from(`${JSON.stringify(staleBudget, null, 2)}\n`);
  const staleReceipt = { ...legacyCorpus, index_sha256: digest(staleBytes) };
  assert.throws(() => selectDiagnosticRecordView(staleBudget, staleBytes, binding, corpusPin, { legacyCorpus: staleReceipt }), /source budget drifted/);
});

test("driver exercises authenticated legacy corpus through exact requests, both terminal outcomes, stop, and cleanup", async () => {
  const runnerTemp = await mkdtemp(path.join(os.tmpdir(), "starvector-diagnostic-route-"));
  const repo = path.join(runnerTemp, "repo"), output = path.join(runnerTemp, "output"), weightsRoot = path.join(runnerTemp, "weights"), corpus = path.join(runnerTemp, "corpus");
  const options = { root: repo, output, weightsRoot, corpusAssetsRoot: corpus, permanentPin, corpusSourcePin: corpusPin, expectedSceneWorksRevision: sceneWorksRevision, leaseRoot: path.join(runnerTemp, "leases"), leaseHelper: path.join(repo, "lease"), platform: "win32", runnerTemp, noJobDownloads: "1", gpuId: "0" };
  const index = legacyIndexFixture(), indexBytes = Buffer.from(`${JSON.stringify(index, null, 2)}\n`), binding = await sealedBinding(index);
  const legacyCorpus = { ...DIAGNOSTIC_LEGACY_CORPUS, inference_revision: corpusPin, row_identity_sha256: index.row_identity_sha256, index_sha256: digest(indexBytes) };
  const acceptedFiles = { transcript: path.join(runnerTemp, "provider.json"), svg: path.join(runnerTemp, "canonical.svg"), png: path.join(runnerTemp, "preview.png") };
  let stopped = false, released = false, submitted = 0;
  try {
    await mkdir(path.join(repo, "config", "manifests"), { recursive: true }); await mkdir(weightsRoot, { recursive: true }); await mkdir(corpus, { recursive: true });
    await writeFile(path.join(repo, "config", "manifests", "builtin.models.jsonc"), JSON.stringify(manifestFixture()));
    await writeFile(path.join(weightsRoot, "starvector-terminal-weights-v1.json"), JSON.stringify(weightsFixture()));
    await writeFile(path.join(corpus, "starvector-terminal-row-index-v1.json"), indexBytes);
    await writeFile(acceptedFiles.transcript, "{}\n"); await writeFile(acceptedFiles.svg, "<svg/>\n"); await writeFile(acceptedFiles.png, Buffer.from([137, 80, 78, 71]));
    await prepareDiagnosticOutput(options);
    const service = serviceFixture(); service.worker.worker_id = "worker-1";
    const occupancy = { schema_version: 1, samples: [], foreign_processes_observed: false, activity_attribution: "foreign_only_before_service", active_foreign_compute: false, enough_free_memory_for_static_weight_floor: true, functional_execution_allowed: true, timing_usable_for_performance: true };
    const result = await runDiagnostic(options, {
      legacyCorpus,
      acquire: async () => async () => { released = true; },
      start: async () => service,
      ready: async (_url, tuple, workerId) => { assert.equal(tuple, "candle-cuda:1b"); assert.equal(workerId, "worker-1"); return service.worker; },
      importAssets: async ({ tuple }) => { assert.equal(tuple, "candle-cuda:1b"); return binding; },
      submit: async (_url, record) => {
        assert.equal(record.case_index, DIAGNOSTIC_CASES[submitted]); assert.equal(record.projectId, binding.project_id); assert.equal(record.sourceAssetId, binding.assets[record.case_index].asset_id); assert.equal(record.input_png_sha256, index.rows[record.case_index].png_sha256); assert.deepEqual(record.detailBudget, DIAGNOSTIC_BUDGET);
        const accepted = submitted !== 1; submitted += 1;
        const job = outcomeFixture(record, accepted);
        Object.assign(job.result.terminalEvidence, { providerTranscriptPath: acceptedFiles.transcript, providerTranscriptSha256: digest(await readFile(acceptedFiles.transcript)) });
        if (accepted) Object.assign(job.result.terminalEvidence, { canonicalSvgPath: acceptedFiles.svg, canonicalSvgSha256: digest(await readFile(acceptedFiles.svg)), previewPngPath: acceptedFiles.png, previewPngSha256: digest(await readFile(acceptedFiles.png)) });
        return job;
      },
      preserve: async (_output, suite, caseId, job) => { assert.equal(suite, "image_quality"); assert.equal(caseId, "diagnostic-quality-v1-11"); assert.equal(job.result.terminalEvidence.accepted, false); return { artifacts: [] }; },
      occupancy: async () => occupancy,
      stop: async () => { stopped = true; },
    });
    assert.equal(result.status, "completed"); assert.equal(submitted, 4); assert.equal(result.results.filter((item) => item.accepted).length, 3); assert.equal(result.results.filter((item) => !item.accepted).length, 1);
    assert.equal(result.corpus.source_schema_version, 1); assert.equal(result.corpus.source_index_sha256, digest(indexBytes)); assert.deepEqual(result.corpus.request_detail_budget, DIAGNOSTIC_BUDGET);
    assert.equal(stopped, true); assert.equal(released, true); assert.equal(result.product_service_cleanup, "stopped_and_state_removed"); assert.equal(result.lease, "released");
  } finally { await rm(runnerTemp, { recursive: true, force: true }); }
});

test("service and terminal results bind exact source, model, provider, backend, and limits", () => {
  assert.deepEqual(validateDiagnosticManifest(manifestFixture()), DIAGNOSTIC_BUDGET);
  for (const mutate of [
    (next) => { next.models[0].vector.maxNewTokens = 4000; },
    (next) => { next.models[0].vector.providers.candle.id = "other"; },
    (next) => { next.models[0].downloads[0].revision = revision("0"); },
  ]) {
    const next = structuredClone(manifestFixture()); mutate(next);
    assert.throws(() => validateDiagnosticManifest(next), /budget drifted|route identity drifted|snapshot identity drifted/);
  }
  const modelInventory = validateDiagnosticWeightsManifest(weightsFixture());
  assert.equal(modelInventory, "a1ab79c36eec58747bf40e1de981497e15ecb0e2af15c62a4f033030555766e4");
  for (const mutate of [
    (next) => { next.schema_version = 2; },
    (next) => { next.models["starvector-1b"].relative_path = "models/starvector-8b"; },
    (next) => { next.models["starvector-1b"].revision = revision("0"); },
    (next) => { next.models["starvector-1b"].inventory_sha256 = "invalid"; },
  ]) {
    const next = weightsFixture(); mutate(next);
    assert.throws(() => validateDiagnosticWeightsManifest(next), /Windows weights identity is invalid/);
  }
  assert.equal(validateDiagnosticService(serviceFixture(), { sceneWorksRevision, permanentPin, modelInventory }).tuple, "candle-cuda:1b");
  for (const [field, value] of [["sceneworks_revision", revision("0")], ["inference_revision", revision("1")], ["tuple", "mlx:1b"]]) {
    const next = serviceFixture(); next[field] = value;
    assert.throws(() => validateDiagnosticService(next, { sceneWorksRevision, permanentPin, modelInventory }), /source identity drifted/);
  }
  for (const [field, value] of [["provider_id", "other"], ["backend", "mlx"], ["model_id", "starvector_8b"]]) {
    const next = serviceFixture(); next.worker[field] = value;
    assert.throws(() => validateDiagnosticService(next, { sceneWorksRevision, permanentPin, modelInventory }), /worker route identity drifted/);
  }
  const wrongInventory = serviceFixture(); wrongInventory.models["starvector-1b"].inventory_sha256 = sha("0");
  assert.throws(() => validateDiagnosticService(wrongInventory, { sceneWorksRevision, permanentPin, modelInventory }), /model inventory drifted/);
  const record = { case_index: 6, input_png_sha256: sha("6"), projectId: "project", sourceAssetId: "asset-6", model: "starvector_1b", sampling: { seed: 7 }, detailBudget: { ...DIAGNOSTIC_BUDGET } };
  assert.equal(validateDiagnosticOutcome(record, outcomeFixture(record)).accepted, true);
  for (const [field, value] of [["providerId", "other"], ["backend", "mlx"], ["modelId", "starvector_8b"], ["sourceRasterSha256", sha("7")]]) {
    const next = outcomeFixture(record); next.result.terminalEvidence[field] = value;
    assert.throws(() => validateDiagnosticOutcome(record, next), /terminal identity drifted/);
  }
  const overBudget = outcomeFixture(record); overBudget.result.terminalEvidence.generatedTokens = 7934;
  assert.throws(() => validateDiagnosticOutcome(record, overBudget), /terminal bounds/);
  for (const mutate of [
    (next) => { next.status = "failed"; },
    (next) => { next.payload.detailBudget.maxNewTokens = 4000; },
    (next) => { next.payload.sourceAssetId = "other"; },
    (next) => { next.payload.sampling.seed = 8; },
  ]) {
    const next = outcomeFixture(record); mutate(next);
    assert.throws(() => validateDiagnosticOutcome(record, next), /stored job payload differs/);
  }
  const generic = outcomeFixture(record, false); generic.result.terminalEvidence.rejectionStage = "infrastructure";
  assert.throws(() => validateDiagnosticOutcome(record, generic), /sealed non-conversion/);
  const sanitizer = outcomeFixture(record, false); Object.assign(sanitizer.result.terminalEvidence, { finishReason: "complete_root", rejectionStage: "sanitizer", rejectionCode: "provider_svg_text" });
  assert.equal(validateDiagnosticOutcome(record, sanitizer).accepted, false);
  const mismatchedSanitizer = structuredClone(sanitizer); mismatchedSanitizer.result.terminalEvidence.finishReason = "wall_time_limit";
  assert.throws(() => validateDiagnosticOutcome(record, mismatchedSanitizer), /sealed non-conversion/);
});

test("CUDA occupancy records activity without treating an idle resident context as active", async () => {
  const uuid = "GPU-12345678-abcd-1234-abcd-123456789abc";
  assert.equal(parseDiagnosticCudaGpu(`0, ${uuid}, 7, 32768, 30000, 2768`).utilization_percent, 7);
  assert.deepEqual(parseDiagnosticCudaProcesses("No running processes found", uuid), []);
  assert.equal(parseDiagnosticCudaProcesses(`${uuid}, 42, other.exe, 512`, uuid)[0].used_gpu_memory_bytes, 512 * 1024 * 1024);
  const outputs = Array.from({ length: 3 }, () => [{ stdout: `0, ${uuid}, 0, 32768, 30000, 2768` }, { stdout: `${uuid}, 42, other.exe, 512` }]).flat();
  const idleResident = await captureDiagnosticCudaOccupancy({ execFileImpl: async () => outputs.shift(), wait: async () => {} });
  assert.equal(idleResident.functional_execution_allowed, true);
  assert.equal(idleResident.timing_usable_for_performance, false);
  const activeOutputs = Array.from({ length: 3 }, () => [{ stdout: `0, ${uuid}, 17, 32768, 30000, 2768` }, { stdout: `${uuid}, 42, other.exe, 512` }]).flat();
  const active = await captureDiagnosticCudaOccupancy({ execFileImpl: async () => activeOutputs.shift(), wait: async () => {} });
  assert.equal(active.active_foreign_compute, true);
  assert.equal(active.functional_execution_allowed, false);
  const ambiguousOutputs = Array.from({ length: 3 }, () => [{ stdout: `0, ${uuid}, 17, 32768, 30000, 2768` }, { stdout: `${uuid}, 42, other.exe, 512\n${uuid}, 99, owned.exe, 12000` }]).flat();
  const ambiguous = await captureDiagnosticCudaOccupancy({ ownedPids: [99], execFileImpl: async () => ambiguousOutputs.shift(), wait: async () => {} });
  assert.equal(ambiguous.active_foreign_compute, false);
  assert.equal(ambiguous.functional_execution_allowed, true);
  assert.equal(ambiguous.timing_usable_for_performance, false);
});

test("current built-in manifest supplies the exact diagnostic model and Detailed budget", async () => {
  const manifest = JSON.parse(stripJsoncComments(await readFile(new URL("../config/manifests/builtin.models.jsonc", import.meta.url), "utf8")));
  assert.deepEqual(validateDiagnosticManifest(manifest), DIAGNOSTIC_BUDGET);
});

test("current source derives one exact inference pin without a workflow input", async () => {
  const pin = await readCurrentInferencePin(root);
  assert.equal(pin, "8e2d9671fd28ab1b34aa22c8fc49de221d43b000");
  const checkWorkflow = await readFile(path.join(root, ".github/workflows/check.yml"), "utf8");
  const fetch = checkWorkflow.slice(checkWorkflow.indexOf("      - name: Fetch the exact public inference terminal contract"));
  assert.match(fetch, new RegExp(`git -C \\"\\$inference_root\\" fetch --depth=1 origin ${pin}`));
  assert.match(fetch, new RegExp(`rev-parse HEAD\\)\\" = ${pin}`));
});

test("diagnostic stops only at three accepted or three rejected across the five remaining cases", () => {
  assert.equal(diagnosticShouldStop([{ accepted: true }, { accepted: true }]), false);
  assert.equal(diagnosticShouldStop([{ accepted: true }, { accepted: true }, { accepted: true }]), true);
  assert.equal(diagnosticShouldStop([{ accepted: false }, { accepted: false }]), false);
  assert.equal(diagnosticShouldStop([{ accepted: false }, { accepted: false }, { accepted: false }]), true);
});

test("workflow exposes one fixed Windows diagnostic with offline cleanup and diagnostic-only artifacts", async () => {
  const workflow = await readFile(new URL("../.github/workflows/server-candle-linux.yml", import.meta.url), "utf8");
  const job = workflow.slice(workflow.indexOf("  starvector-diagnostic-candle-1b:"));
  assert.match(workflow, /options: \[standard, source, provision, readiness, campaign, diagnostic-candle-1b\]/);
  assert.match(job, /runs-on: \[self-hosted, Windows, X64, cuda, real-weights\]/);
  assert.match(job, /STARVECTOR_TERMINAL_NO_JOB_DOWNLOADS: "1"/);
  assert.match(job, /shell: cmd/);
  assert.match(job, /Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64\.bat/);
  assert.match(job, /set "NVCC_CCBIN=%VCToolsInstallDir%bin\\Hostx64\\x64"/);
  assert.match(job, /where cl\.exe >nul \|\| exit \/b 1/);
  assert.ok(job.indexOf("diagnostic.mjs prepare") < job.indexOf("vcvars64.bat"));
  assert.ok(job.indexOf("vcvars64.bat") < job.indexOf("cargo fetch --locked"));
  assert.match(job, /diagnostic\.mjs prepare/);
  assert.match(job, /cargo fetch --locked/);
  assert.match(job, /cargo build --release --locked/);
  assert.doesNotMatch(job, /CARGO_NET_OFFLINE|cargo fetch[^\n]*--manifest-path|huggingface-cli|wget|curl/);
  assert.match(job, /starvector-terminal-diagnostic\.mjs/);
  assert.match(job, /source-pin "\$env:GITHUB_WORKSPACE"/);
  assert.match(job, /STARVECTOR_DIAGNOSTIC_PERMANENT_PIN=\$currentPin/);
  assert.match(job, /starvector_terminal_lease\.exe/);
  assert.match(job, /Verify owned product service cleanup\n\s+if: \$\{\{ always\(\) \}\}/);
  assert.match(job, /Upload diagnostic-only CUDA evidence\n\s+if: \$\{\{ always\(\) \}\}/);
  assert.doesNotMatch(job, /macOS|ARM64|campaign_run_id|STARVECTOR_TERMINAL_METRICS/);
  assert.doesNotMatch(job.slice(job.indexOf("    steps:")), /\$\{\{ inputs\./);
});
