#!/usr/bin/env node
// This controller never loads a model. A dispatch supplies already-provisioned weights,
// metrics and a command which must exercise SceneWorks' vector_generate HTTP route.
import { createHash, randomUUID } from "node:crypto";
import { mkdir, open, readdir, readFile, stat, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { COUNTS, TUPLES, readPlanAndLock } from "./starvector-terminal-campaign.mjs";

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const json = async (file) => JSON.parse(await readFile(file, "utf8"));
const die = (message) => { throw new Error(`starvector terminal producer: ${message}`); };
const hashPattern = /^[a-f0-9]{64}$/;
const gitRevisionPattern = /^[a-f0-9]{40}$/;
const finishes = new Set(["complete_root", "eos", "token_limit", "byte_limit", "wall_time_limit", "cancelled"]);
const numeric = (value, label, min = -Infinity, max = Infinity) => { if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) die(`${label} is not in range`); };
const digest = (value, label) => { if (typeof value !== "string" || !hashPattern.test(value)) die(`${label} must be a sha256`); };
const nullableDigest = (value, label) => { if (value !== null) digest(value, label); };

export async function inventory(root) {
  const names = (await readdir(root, { recursive: true })).sort();
  const entries = [];
  for (const name of names) {
    const file = path.join(root, name);
    if ((await stat(file)).isFile()) entries.push({ path: name, bytes: (await stat(file)).size, sha256: sha(await readFile(file)) });
  }
  return { entries, sha256: sha(JSON.stringify(entries)) };
}

export async function acquireLease(leasePath, identity) {
  await mkdir(path.dirname(leasePath), { recursive: true });
  let handle;
  try {
    handle = await open(leasePath, "wx");
    await handle.writeFile(JSON.stringify({ identity, pid: process.pid, created_at: new Date().toISOString() }));
  } catch { die(`cross-repo terminal lease already held: ${leasePath}`); }
  return async () => { await handle.close(); await unlink(leasePath); };
}

export async function validateMetricsEnvironment(metricsRoot, metricsLockSha) {
  const environment = await json(path.join(metricsRoot, "starvector-terminal-metrics-environment-v1.json"));
  if (environment.metrics_lock_sha256 !== metricsLockSha) die("pre-provisioned metric environment does not match checked-in lock");
  if (!Array.isArray(environment.packages) || environment.packages.length !== 4) die("metric environment package closure is incomplete");
  for (const name of ["scikit-image", "lpips", "torch", "torchvision"]) {
    const item = environment.packages.find((entry) => entry?.name === name);
    if (!item || typeof item.version !== "string" || item.version.length === 0) die(`metric package ${name} identity missing`);
  }
  for (const key of ["lpips_linear", "alexnet"]) {
    const weight = environment.weights?.[key];
    if (!weight?.path || !hashPattern.test(weight.sha256)) die(`metric ${key} weight identity missing`);
    const actual = sha(await readFile(path.join(metricsRoot, weight.path)));
    if (actual !== weight.sha256) die(`metric ${key} weight content drifted`);
  }
  return environment;
}

export async function validateWeightsEnvironment(weightsRoot, snapshotRevisions) {
  const environment = await json(path.join(weightsRoot, "starvector-terminal-weights-v1.json"));
  for (const [key, revision] of Object.entries(snapshotRevisions)) {
    const model = environment.models?.[key];
    if (!model?.relative_path || model.revision !== revision || !hashPattern.test(model.inventory_sha256)) die(`${key} snapshot identity or inventory missing`);
    if ((await inventory(path.join(weightsRoot, model.relative_path))).sha256 !== model.inventory_sha256) die(`${key} snapshot inventory drifted`);
  }
  return environment;
}

export function validateIndexed(records, expected, label, checker) {
  if (!Array.isArray(records) || records.length !== expected) die(`${label} count must be ${expected}`);
  records.forEach((record, index) => { if (record?.case_index !== index || typeof record.case_id !== "string" || !record.case_id) die(`${label} must be ordered and identity-bound at ${index}`); checker(record, index); });
  return records;
}

export function validateTupleEvidence(raw, tuple, campaignRunId) {
  if (raw?.schema_version !== 1 || raw.campaign_run_id !== campaignRunId || raw.tuple !== tuple || raw.production_route !== "vector_generate") die(`raw ${tuple} route identity mismatch`);
  if (!gitRevisionPattern.test(raw.execution?.sceneworks_revision) || typeof raw.execution?.workflow_run_id !== "string" || !raw.execution.workflow_run_id || !Number.isInteger(raw.execution.workflow_run_attempt) || typeof raw.execution.head_ref !== "string" || !raw.execution.head_ref) die(`${tuple} execution provenance missing`);
  const [backend, tier] = tuple.split(":");
  if (raw.model?.backend !== backend || raw.model?.tier !== tier || typeof raw.model.provider_id !== "string" || !raw.model.provider_id || !hashPattern.test(raw.model.inventory_sha256)) die(`${tuple} provider/model closure missing`);
  validateIndexed(raw.image_quality, COUNTS.image_quality, `${tuple} image quality`, (record) => {
    for (const key of ["source_svg_sha256", "source_png_sha256", "provider_transcript_sha256"]) digest(record[key], `${tuple} ${key}`);
    if (!finishes.has(record.finish_reason) || typeof record.accepted !== "boolean") die(`${tuple} typed output missing`);
    numeric(record.latency_seconds, `${tuple} latency`, 0);
    if (record.accepted) {
      if (!["complete_root", "eos"].includes(record.finish_reason)) die(`${tuple} accepted output has nonterminal finish`);
      for (const key of ["canonical_svg_sha256", "preview_png_sha256"]) digest(record[key], `${tuple} ${key}`);
      numeric(record.ssim, `${tuple} ssim`, 0, 1); numeric(record.lpips, `${tuple} lpips`, 0);
    } else for (const key of ["canonical_svg_sha256", "preview_png_sha256", "ssim", "lpips"]) if (record[key] !== null) die(`${tuple} rejected output carries ${key}`);
  });
  validateIndexed(raw.deterministic_parity, COUNTS.deterministic_parity, `${tuple} deterministic parity`, (record) => {
    if (!Number.isInteger(record.seed)) die(`${tuple} parity seed missing`);
    for (const key of ["first_preview_sha256", "second_preview_sha256", "provider_transcript_sha256"]) digest(record[key], `${tuple} ${key}`);
    numeric(record.rendered_ssim, `${tuple} rendered_ssim`, 0, 1);
  });
  if (!raw.hardware || typeof raw.hardware.runner_name !== "string" || !raw.hardware.runner_name || !raw.hardware.raw_probe_sha256 || !hashPattern.test(raw.hardware.raw_probe_sha256)) die(`${tuple} raw hardware probe missing`);
  return raw;
}

export function validateSuites(suites, campaignRunId) {
  if (suites?.schema_version !== 1 || suites.campaign_run_id !== campaignRunId) die("suite campaign identity mismatch");
  const hostile = suites.hostile_sanitizer;
  if (!hashPattern.test(hostile?.corpus_sha256)) die("hostile corpus identity missing");
  validateIndexed(hostile.cases, COUNTS.hostile_sanitizer, "hostile suite", (record) => {
    digest(record.input_sha256, "hostile input");
    if (!["rejected", "sanitized_inert"].includes(record.actual_disposition) || record.result_contains_inline_svg !== false || record.staging_residue !== false || !Array.isArray(record.published_paths)) die("hostile raw disposition invalid");
    if (record.actual_disposition === "rejected") { if (record.published_paths.length || record.output_svg_sha256 !== null || record.output_png_sha256 !== null) die("rejected hostile case published output"); }
    else { if (record.published_paths.length !== 2) die("inert hostile case lacks canonical pair"); digest(record.output_svg_sha256, "inert svg"); digest(record.output_png_sha256, "inert preview"); }
    if (record.error_code !== null && typeof record.error_code !== "string") die("hostile error code invalid");
    digest(record.transcript_sha256, "hostile transcript");
  });
  const prompt = suites.prompt_composition;
  if (!hashPattern.test(prompt?.corpus_sha256)) die("prompt corpus identity missing");
  let accepted = 0;
  validateIndexed(prompt.cases, COUNTS.prompt_composition, "prompt suite", (record) => {
    for (const key of ["prompt_sha256", "raster_asset_sha256", "transcript_sha256"]) digest(record[key], `prompt ${key}`);
    if (!Number.isInteger(record.seed) || typeof record.accepted !== "boolean") die("prompt identity missing");
    if (record.accepted) { accepted++; for (const key of ["svg_sha256", "preview_sha256"]) digest(record[key], `prompt ${key}`); numeric(record.clip_cosine, "prompt cosine", -1, 1); numeric(record.alignment_loss, "prompt alignment", -2, 2); numeric(record.latency_seconds, "prompt latency", 0); }
    else for (const key of ["svg_sha256", "preview_sha256", "clip_cosine", "alignment_loss"]) if (record[key] !== null) die(`rejected prompt carries ${key}`);
  });
  if (accepted < 57) die("prompt suite accepted fewer than 57 cases");
  return suites;
}

export async function preflight({ planPath, inferenceRoot, weightsRoot, metricsRoot, command }) {
  const { plan, metrics_lock_sha256 } = await readPlanAndLock(planPath);
  if (!inferenceRoot || !weightsRoot || !metricsRoot || !command?.length) die("plan, pre-provisioned inference/weights/metrics, and production route command required");
  for (const item of [plan.inference_contract.receipt_schema, plan.inference_contract.corpus]) try { await stat(path.join(inferenceRoot, item)); } catch { die(`missing pinned inference contract ${item}`); }
  const weights = await inventory(weightsRoot); if (!weights.entries.length) die("pre-provisioned weights inventory empty");
  const weight_environment = await validateWeightsEnvironment(weightsRoot, plan.model_snapshot_revisions);
  const metrics = await validateMetricsEnvironment(metricsRoot, metrics_lock_sha256);
  return { plan, weights, weight_environment, metrics, metrics_lock_sha256 };
}

const invoke = async (command, env) => {
  const child = spawn(command[0], command.slice(1), { stdio: "inherit", env: { ...process.env, ...env } });
  const code = await new Promise((resolve, reject) => child.once("error", reject).once("exit", resolve));
  if (code !== 0) die(`production route command exited ${code}`);
};

export async function executeTuple({ planPath, inferenceRoot, weightsRoot, metricsRoot, leasePath, command, output, tuple, campaignRunId = randomUUID(), permanentPin }) {
  const pre = await preflight({ planPath, inferenceRoot, weightsRoot, metricsRoot, command });
  if (!TUPLES.includes(tuple) || !permanentPin) die("exact tuple and permanent pin required");
  const release = await acquireLease(leasePath, `${permanentPin}:${campaignRunId}`);
  await mkdir(output, { recursive: true }); let failure = null;
  try {
    await invoke(command, { STARVECTOR_TERMINAL_RUN_ID: campaignRunId, STARVECTOR_TERMINAL_PLAN: planPath, STARVECTOR_TERMINAL_OUTPUT: output, STARVECTOR_TERMINAL_TUPLE: tuple, STARVECTOR_TERMINAL_PERMANENT_PIN: permanentPin });
    await validateTupleEvidence(await json(path.join(output, "raw-results.json")), tuple, campaignRunId);
  } catch (error) { failure = error; }
  const artifacts = await inventory(output);
  await writeFile(path.join(output, "tuple-controller-receipt.json"), JSON.stringify({ schema_version: 1, campaign_run_id: campaignRunId, permanent_pin: permanentPin, tuple, status: failure ? "failed" : "succeeded", error: failure ? String(failure.message ?? failure) : null, inference_revision: pre.plan.inference_contract.revision, metrics_lock_sha256: pre.metrics_lock_sha256, weights_inventory_sha256: pre.weights.sha256, artifacts_manifest_sha256: artifacts.sha256 }, null, 2) + "\n");
  await writeFile(path.join(output, "tuple-artifacts.json"), JSON.stringify(artifacts, null, 2) + "\n");
  await release(); if (failure) throw failure;
}

export async function sealReceipt({ planPath, inferenceRoot, evidenceRoot, output, campaignRunId, permanentPin }) {
  const { plan, metrics_lock_sha256 } = await readPlanAndLock(planPath);
  if (!campaignRunId || !permanentPin) die("campaign run and permanent pin required to seal receipt");
  for (const item of [plan.inference_contract.receipt_schema, plan.inference_contract.corpus]) try { await stat(path.join(inferenceRoot, item)); } catch { die(`missing pinned inference contract ${item}`); }
  const rows = await readdir(evidenceRoot, { recursive: true });
  const rawFiles = rows.filter((entry) => entry.endsWith("raw-results.json")).map((entry) => path.join(evidenceRoot, entry));
  const runs = new Map(); for (const file of rawFiles) { const raw = await json(file); if (TUPLES.includes(raw.tuple)) runs.set(raw.tuple, validateTupleEvidence(raw, raw.tuple, campaignRunId)); }
  if (runs.size !== TUPLES.length) die("four tuple raw evidence files required");
  const execution = runs.get(TUPLES[0]).execution;
  for (const tuple of TUPLES.slice(1)) {
    const candidate = runs.get(tuple).execution;
    if (candidate.sceneworks_revision !== execution.sceneworks_revision || candidate.workflow_run_id !== execution.workflow_run_id || candidate.workflow_run_attempt !== execution.workflow_run_attempt || candidate.head_ref !== execution.head_ref) die("tuple execution provenance is mixed");
  }
  const suites = validateSuites(await json(path.join(evidenceRoot, "terminal-suites.json")), campaignRunId);
  await mkdir(output, { recursive: true });
  const artifacts = await inventory(evidenceRoot);
  const receipt = { schema_version: 1, campaign_run_id: campaignRunId, permanent_pin: permanentPin, execution: { repository: "SceneWorks/SceneWorks", ...execution, status: "succeeded" }, producer: { route: "vector_generate", artifact_manifest_sha256: artifacts.sha256 }, inference_revision: plan.inference_contract.revision, inference_preflight: { model_inventory_sha256_by_key: Object.fromEntries(TUPLES.map((tuple) => [tuple, runs.get(tuple).model.inventory_sha256])), provider_ids: TUPLES.map((tuple) => runs.get(tuple).model.provider_id) }, metric_identity: { metrics_lock_sha256 }, artifact_manifest: artifacts, runs: TUPLES.map((tuple) => runs.get(tuple)), hostile_sanitizer: suites.hostile_sanitizer, prompt_composition: suites.prompt_composition };
  await writeFile(path.join(output, "terminal-receipt.json"), JSON.stringify(receipt, null, 2) + "\n");
  return receipt;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [mode, ...args] = process.argv.slice(2);
  const failCli = (error) => { console.error(error.message); process.exitCode = 1; };
  if (mode === "run") {
    const [planPath, inferenceRoot, weightsRoot, metricsRoot, leasePath, output, tuple, campaignRunId, permanentPin, ...command] = args;
    executeTuple({ planPath, inferenceRoot, weightsRoot, metricsRoot, leasePath, output, tuple, campaignRunId, permanentPin, command }).catch(failCli);
  } else if (mode === "seal") {
    const [planPath, inferenceRoot, evidenceRoot, output, campaignRunId, permanentPin] = args;
    sealReceipt({ planPath, inferenceRoot, evidenceRoot, output, campaignRunId, permanentPin }).catch(failCli);
  } else failCli(new Error("usage: run|seal ..."));
}
