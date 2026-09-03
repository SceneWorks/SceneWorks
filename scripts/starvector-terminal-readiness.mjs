#!/usr/bin/env node
// Read-only host gate for the SC-22261 terminal campaign. This validates every
// pre-provisioned input without starting SceneWorks, running inference, or
// claiming a campaign/tuple marker.
import { createHash } from "node:crypto";
import { execFile as execFileCallback } from "node:child_process";
import { lstat, mkdir, readFile, readdir, stat, writeFile } from "node:fs/promises";
import { promisify } from "node:util";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { INFERENCE_REVISION, TUPLES, readPlanAndLock } from "./starvector-terminal-campaign.mjs";
import { fileSha256 } from "./lib/file-sha256.mjs";
import { sortTerminalTreeEntries, terminalTreeEntry, terminalTreeSha256 } from "./lib/terminal-tree-identity.mjs";
import {
  validateInferencePreflight,
  validateMetricsEnvironment,
  validateWeightsEnvironment,
  verifyInferenceCheckout,
  verifyPermanentPin,
  verifyRouteClosure,
} from "./starvector-terminal-producer.mjs";

const execFile = promisify(execFileCallback);
const SHA256 = /^[a-f0-9]{64}$/;
const sha = (value) => createHash("sha256").update(value).digest("hex");
const json = async (file) => JSON.parse(await readFile(file, "utf8"));
const die = (message) => { throw new Error(`starvector terminal readiness: ${message}`); };
const stable = (value) => Array.isArray(value) ? `[${value.map(stable).join(",")}]` : value && typeof value === "object" ? `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(",")}}` : JSON.stringify(value);

function safeRelative(relative, label) {
  if (typeof relative !== "string" || !relative || path.isAbsolute(relative) || path.win32.isAbsolute(relative) || relative.split(/[\\/]/).includes("..")) die(`${label} path is unsafe`);
  return relative.split(/[\\/]/);
}

async function regularFile(root, relative, expected, label) {
  if (!SHA256.test(expected ?? "")) die(`${label} hash is invalid`);
  const file = path.join(root, ...safeRelative(relative, label));
  const info = await lstat(file);
  if (!info.isFile() || info.isSymbolicLink()) die(`${label} must be a regular non-symlink file`);
  const bytes = await readFile(file);
  if (sha(bytes) !== expected) die(`${label} hash mismatch`);
  return { path: relative.split(/[\\/]/).join("/"), byte_size: info.size, sha256: expected };
}

export async function treeIdentity(root, relative, label, digestFile = fileSha256) {
  const directory = path.join(root, ...safeRelative(relative, label));
  const rootInfo = await lstat(directory);
  if (!rootInfo.isDirectory() || rootInfo.isSymbolicLink()) die(`${label} must be a regular directory tree`);
  const rows = [];
  for (const name of await readdir(directory, { recursive: true })) {
    const file = path.join(directory, name); const info = await lstat(file);
    if (info.isSymbolicLink()) die(`${label} rejects symlink ${name}`);
    if (!info.isFile() && !info.isDirectory()) die(`${label} rejects non-regular entry ${name}`);
    if (info.isFile()) rows.push(terminalTreeEntry(name.split(path.sep).join("/"), info.size, await digestFile(file)));
  }
  const canonicalRows = sortTerminalTreeEntries(rows);
  return { file_count: canonicalRows.length, sha256: terminalTreeSha256(canonicalRows) };
}

export async function validateTerminalServiceClosure(weightsRoot, weights) {
  const expectedModels = ["starvector-1b", "starvector-8b"];
  if (JSON.stringify(Object.keys(weights.models ?? {}).sort()) !== JSON.stringify(expectedModels)) die("weights manifest must contain exactly the 1B and 8B snapshots");
  const closure = weights.terminal_service_closure;
  if (!closure || typeof closure.app_data_relative_path !== "string" || typeof closure.hf_home_relative_path !== "string" || !SHA256.test(closure.app_data_sha256 ?? "") || !SHA256.test(closure.hf_home_sha256 ?? "")) die("weights manifest lacks the exact terminal service closure");
  const appData = await treeIdentity(weightsRoot, closure.app_data_relative_path, "terminal app-data closure");
  const hfHome = await treeIdentity(weightsRoot, closure.hf_home_relative_path, "terminal HF closure");
  if (appData.file_count < 1 || hfHome.file_count < 1) die("terminal service closure trees must not be empty");
  if (appData.sha256 !== closure.app_data_sha256 || hfHome.sha256 !== closure.hf_home_sha256) die("terminal service closure tree hash mismatch");
  return { app_data: appData, hf_home: hfHome };
}

export async function validateMetricRuntime(metricsPython, packages) {
  if (!metricsPython || !path.isAbsolute(metricsPython)) die("absolute pre-provisioned metrics Python is required");
  const info = await stat(metricsPython);
  if (!info.isFile()) die("metrics Python must resolve to a regular executable");
  const names = packages.map((entry) => entry.name);
  const script = `import importlib.metadata as m,json\nprint(json.dumps({name:m.version(name) for name in ${JSON.stringify(names)}}))`;
  const { stdout } = await execFile(metricsPython, ["-c", script], { maxBuffer: 1024 * 1024 });
  let observed;
  try { observed = JSON.parse(stdout); } catch { die("metrics Python did not emit its package identity"); }
  for (const entry of packages) if (observed?.[entry.name] !== entry.version) die(`installed metric package ${entry.name} is not exactly ${entry.version}`);
  if (Object.keys(observed).length !== packages.length) die("installed metric package probe returned an inexact identity");
  return observed;
}

function validateHashFields(value, label) {
  if (Array.isArray(value)) return value.forEach((entry, index) => validateHashFields(entry, `${label}[${index}]`));
  if (!value || typeof value !== "object") return;
  for (const [key, entry] of Object.entries(value)) {
    if (key.endsWith("sha256") && !SHA256.test(entry ?? "")) die(`${label}.${key} is not a SHA-256 identity`);
    validateHashFields(entry, `${label}.${key}`);
  }
}

function exactCases(index, key, expected, identity) {
  const observed = {};
  for (const tuple of TUPLES) {
    const records = index[key]?.[tuple];
    if (!Array.isArray(records) || records.length !== expected.length) die(`${tuple} must carry exactly ${expected.length} ${key} records`);
    const names = records.map((entry) => entry?.[identity]);
    if (new Set(names).size !== expected.length || expected.some((name) => !names.includes(name))) die(`${tuple} ${key} identities are incomplete`);
    records.forEach((entry, position) => {
      if (typeof entry.case_id !== "string" || !entry.case_id) die(`${tuple} ${key}[${position}] lacks a case id`);
      validateHashFields(entry, `${tuple}.${key}[${position}]`);
    });
    observed[tuple] = sha(stable(records));
  }
  return observed;
}

export async function validateCorpusAssets(inferenceRoot, corpusRelative, assetsRoot, permanentPin) {
  if (permanentPin !== INFERENCE_REVISION) die("corpus assets do not bind the exact permanent inference revision");
  const corpusPath = path.join(inferenceRoot, ...safeRelative(corpusRelative, "terminal corpus"));
  const corpusInfo = await lstat(corpusPath);
  if (!corpusInfo.isFile() || corpusInfo.isSymbolicLink()) die("terminal corpus must be a regular non-symlink file");
  const validator = await import(pathToFileURL(path.join(inferenceRoot, "scripts", "release", "starvector_terminal_evidence.mjs")).href);
  const corpus = await json(corpusPath), corpusSha256 = validator.validatePlan(corpus);
  const indexPath = path.join(assetsRoot, "starvector-terminal-row-index-v1.json"), indexInfo = await lstat(indexPath), indexBytes = await readFile(indexPath);
  if (!indexInfo.isFile() || indexInfo.isSymbolicLink()) die("terminal row index must be a regular non-symlink file");
  const index = JSON.parse(indexBytes);
  if (index.inference_revision !== permanentPin || index.row_identity_sha256 !== corpus.upstream_image_quality_cases.row_identity_sha256 || !Array.isArray(index.rows) || index.rows.length !== 120) die("terminal row index pin, identity, or cardinality is invalid");

  const assetEntries = [], rows = [];
  for (const [position, row] of index.rows.entries()) {
    const source = corpus.upstream_image_quality_cases.sources[Math.floor(position / 30)];
    if (row.case_index !== position || row.dataset !== source?.dataset || row.revision !== source.revision || row.row_index !== position % 30 || typeof row.filename !== "string" || !row.filename || !SHA256.test(row.svg_sha256 ?? "") || !SHA256.test(row.png_sha256 ?? "") || !SHA256.test(row.reference_png_sha256 ?? "")) die(`terminal row ${position} immutable identity is invalid`);
    assetEntries.push(await regularFile(assetsRoot, row.svg_path, row.svg_sha256, `row ${position} source SVG`));
    assetEntries.push(await regularFile(assetsRoot, row.input_png_path, row.png_sha256, `row ${position} input PNG`));
    assetEntries.push(await regularFile(assetsRoot, row.reference_png, row.reference_png_sha256, `row ${position} reference PNG`));
    if (row.preview_png !== undefined || row.preview_png_sha256 !== undefined) assetEntries.push(await regularFile(assetsRoot, row.preview_png, row.preview_png_sha256, `row ${position} preview PNG`));
    rows.push(row);
  }
  const rowRecord = (row) => JSON.stringify({ dataset: row.dataset, revision: row.revision, row_index: row.row_index, filename: row.filename, svg_sha256: row.svg_sha256 });
  const rowIdentity = sha(`${rows.map(rowRecord).join("\n")}\n`);
  if (rowIdentity !== corpus.upstream_image_quality_cases.row_identity_sha256) die("terminal row identities drifted from the pinned corpus");
  for (const [sourceIndex, source] of corpus.upstream_image_quality_cases.sources.entries()) if (sha(`${rows.slice(sourceIndex * 30, sourceIndex * 30 + 30).map(rowRecord).join("\n")}\n`) !== source.row_identity_sha256) die(`terminal rows for ${source.dataset} drifted`);
  const parityRows = corpus.upstream_image_quality_cases.sources.flatMap((_, sourceIndex) => rows.slice(sourceIndex * 30, sourceIndex * 30 + 5));
  if (sha(`${parityRows.map(rowRecord).join("\n")}\n`) !== corpus.deterministic_parity_cases.row_identity_sha256) die("deterministic parity row identities drifted from the pinned corpus");

  const lifecycle = exactCases(index, "lifecycle_cases", ["load", "unload", "reload", "memory_reported"], "operation");
  const limits = exactCases(index, "limit_cases", ["complete_root", "eos", "token_limit", "byte_limit", "wall_time_limit", "cancelled"], "finish_reason");
  if (!Array.isArray(index.prompt_composition) || index.prompt_composition.length !== 60) die("terminal index must carry exactly 60 prompt-composition records");
  const promptHashes = [];
  index.prompt_composition.forEach((entry, caseIndex) => {
    if (entry?.case_index !== caseIndex || entry.case_id !== `prompt-v1-${caseIndex}` || typeof entry.prompt !== "string" || !entry.prompt || sha(entry.prompt) !== entry.prompt_sha256 || typeof entry.raster_model !== "string" || !entry.raster_model || typeof entry.vector_model !== "string" || !entry.vector_model || typeof entry.expected_raster_revision !== "string" || !entry.expected_raster_revision || typeof entry.expected_vector_revision !== "string" || !entry.expected_vector_revision) die(`prompt-composition record ${caseIndex} is incomplete or drifted`);
    validateHashFields(entry, `prompt_composition[${caseIndex}]`); promptHashes.push(entry.prompt_sha256);
  });
  const promptIdentity = sha(promptHashes.join("\n"));
  if (promptIdentity !== corpus.sceneworks_owned_suites.prompt_composition.content_identity_sha256) die("prompt-composition identities drifted from the pinned corpus");
  assetEntries.sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0);
  return { corpus_sha256: corpusSha256, row_identity_sha256: rowIdentity, index_sha256: sha(indexBytes), assets_sha256: sha(stable(assetEntries)), asset_file_references: assetEntries.length, lifecycle_sha256: lifecycle, limits_sha256: limits, prompt_sha256: promptIdentity };
}

export async function terminalReadiness({ sceneWorksRoot, planPath, inferenceRoot, weightsRoot, metricsRoot, metricsPython, assetsRoot }) {
  const { plan, metrics_lock_sha256 } = await readPlanAndLock(planPath);
  const permanentPin = plan.inference_contract.revision;
  await verifyInferenceCheckout(inferenceRoot);
  await verifyPermanentPin(sceneWorksRoot, permanentPin, permanentPin);
  const route = await verifyRouteClosure(sceneWorksRoot, path.join(sceneWorksRoot, "scripts", "starvector-terminal-route.mjs"));
  const weights = await validateWeightsEnvironment(weightsRoot, plan.model_snapshot_revisions);
  const service = await validateTerminalServiceClosure(weightsRoot, weights);
  const metrics = await validateMetricsEnvironment(metricsRoot, metrics_lock_sha256);
  const packages = await validateMetricRuntime(metricsPython, metrics.packages);
  const preflight = await validateInferencePreflight(inferenceRoot, permanentPin);
  const corpus = await validateCorpusAssets(inferenceRoot, plan.inference_contract.corpus, assetsRoot, permanentPin);
  return { permanent_pin: permanentPin, sceneworks_revision: route.sceneworks_revision, route: { path: route.path, sha256: route.sha256, metric_path: route.metric_path, metric_sha256: route.metric_sha256 }, weights: { models: Object.fromEntries(Object.entries(weights.models).map(([key, value]) => [key, { revision: value.revision, inventory_sha256: value.inventory_sha256 }])), prompt_raster: { provider_id: weights.prompt_raster.provider_id, model: weights.prompt_raster.model, revision: weights.prompt_raster.revision, inventory_sha256: weights.prompt_raster.inventory_sha256 }, terminal_service_closure: service }, metrics: { metrics_lock_sha256, packages, lpips_linear_sha256: metrics.weights.lpips_linear.sha256, alexnet_sha256: metrics.weights.alexnet.sha256, clip_sha256: metrics.clip.checkpoint.sha256 }, inference_preflight: preflight.receipt, corpus };
}

async function main() {
  const [sceneWorksRoot, planPath, inferenceRoot, weightsRoot, metricsRoot, metricsPython, assetsRoot, output] = process.argv.slice(2);
  if (!output) { console.error("usage: <sceneworks-root> <plan> <inference-root> <weights-root> <metrics-root> <metrics-python> <assets-root> <output>"); process.exitCode = 2; return; }
  const report = { schema_version: 1, kind: "starvector_terminal_readiness", runner: { name: process.env.RUNNER_NAME ?? null, os: process.platform, arch: process.arch }, checked_at: new Date().toISOString(), status: "failed" };
  try { report.readiness = await terminalReadiness({ sceneWorksRoot, planPath, inferenceRoot, weightsRoot, metricsRoot, metricsPython, assetsRoot }); report.status = "ready"; }
  catch (error) { report.error = String(error?.message ?? error); process.exitCode = 1; }
  await mkdir(path.dirname(output), { recursive: true }); await writeFile(output, JSON.stringify(report, null, 2) + "\n");
  console.log(`starvector terminal readiness: ${report.status}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) main().catch((error) => { console.error(error.message); process.exitCode = 1; });
