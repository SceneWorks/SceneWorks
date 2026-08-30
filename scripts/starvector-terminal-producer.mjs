#!/usr/bin/env node
// SC-22261 controller: all threshold decisions belong to the pinned inference validator.
import { createHash, randomUUID } from "node:crypto";
import { execFile as execFileCallback } from "node:child_process";
import { mkdir, open, readdir, readFile, rmdir, stat, writeFile } from "node:fs/promises";
import { promisify } from "node:util";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { INFERENCE_REVISION, readPlanAndLock } from "./starvector-terminal-campaign.mjs";

const execFile = promisify(execFileCallback);
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const json = async (file) => JSON.parse(await readFile(file, "utf8"));
const die = (message) => { throw new Error(`starvector terminal producer: ${message}`); };
const SHA256 = /^[a-f0-9]{64}$/;
const REVISION = /^[a-f0-9]{40}$/;
const TUPLES = ["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"];

export async function inventory(root) {
  const entries = [];
  for (const name of (await readdir(root, { recursive: true })).sort()) {
    const file = path.join(root, name); const info = await stat(file);
    if (info.isFile()) entries.push({ path: name, byte_size: info.size, sha256: sha(await readFile(file)) });
  }
  return { entries, aggregate_sha256: sha(JSON.stringify(entries)) };
}

async function git(root, args) { return (await execFile("git", ["-C", root, ...args])).stdout.trim(); }
export async function verifyInferenceCheckout(inferenceRoot) {
  if (!inferenceRoot) die("pinned inference checkout required");
  if (await git(inferenceRoot, ["rev-parse", "HEAD"]) !== INFERENCE_REVISION) die("inference checkout is not the exact PR #891 revision");
  if (await git(inferenceRoot, ["status", "--porcelain"])) die("inference checkout must be clean");
  for (const item of ["release/starvector-terminal-receipt-v1.schema.json", "release/starvector-terminal-corpus-v1.json", "scripts/release/starvector_terminal_evidence.mjs"]) try { await stat(path.join(inferenceRoot, item)); } catch { die(`missing pinned inference contract ${item}`); }
  return path.join(inferenceRoot, "scripts/release/starvector_terminal_evidence.mjs");
}

export async function verifyPermanentPin(sceneWorksRoot, permanentPin) {
  if (!REVISION.test(permanentPin)) die("permanent pin must be an exact 40-character SHA");
  const cargo = await readFile(path.join(sceneWorksRoot, "Cargo.toml"), "utf8");
  const match = cargo.match(/SceneWorks\/inference",\s*rev\s*=\s*"([a-f0-9]{40})"/);
  if (!match || match[1] !== permanentPin) die("permanent pin does not equal this SceneWorks Cargo inference pin");
  return permanentPin;
}

// A stable directory lock is atomic at the host filesystem level. It intentionally never
// reclaims an existing owner: a stale record is reported with its identity for explicit human
// release, avoiding the unsafe practice of unlinking another process's lock.
export async function acquireStableLease(leaseRoot, permanentPin, campaignRunId) {
  if (!leaseRoot || !REVISION.test(permanentPin) || !campaignRunId) die("stable lease root, pin, and campaign id required");
  await mkdir(leaseRoot, { recursive: true });
  const lock = path.join(leaseRoot, `starvector-terminal-${permanentPin}`);
  const owner = { permanent_pin: permanentPin, campaign_run_id: campaignRunId, pid: process.pid, hostname: process.env.RUNNER_NAME ?? process.env.HOSTNAME ?? "unknown", created_at: new Date().toISOString() };
  try { await mkdir(lock); await writeFile(path.join(lock, "owner.json"), JSON.stringify(owner, null, 2) + "\n", { flag: "wx" }); }
  catch {
    let existing = "unreadable"; try { existing = await readFile(path.join(lock, "owner.json"), "utf8"); } catch { /* preserve unknown ownership */ }
    die(`stable cross-repo lease exists; never reclaimed automatically: ${lock}; owner=${existing}`);
  }
  return async () => {
    const current = await json(path.join(lock, "owner.json"));
    if (JSON.stringify(current) !== JSON.stringify(owner)) die("stable lease ownership changed; refusing release");
    await rmdir(lock, { recursive: true });
  };
}

export async function claimCampaignMarker(leaseRoot, permanentPin, campaignRunId) {
  await mkdir(leaseRoot, { recursive: true });
  const marker = path.join(leaseRoot, `starvector-terminal-${permanentPin}.campaign.json`);
  const identity = { permanent_pin: permanentPin, campaign_run_id: campaignRunId };
  try { const handle = await open(marker, "wx"); await handle.writeFile(JSON.stringify(identity, null, 2) + "\n"); await handle.close(); }
  catch {
    const existing = await json(marker);
    if (existing.permanent_pin !== permanentPin || existing.campaign_run_id !== campaignRunId) die(`permanent pin already has a different terminal campaign marker: ${marker}`);
  }
  return identity;
}

export async function validateMetricsEnvironment(metricsRoot, metricsLockSha) {
  const environment = await json(path.join(metricsRoot, "starvector-terminal-metrics-environment-v1.json"));
  if (environment.metrics_lock_sha256 !== metricsLockSha || !Array.isArray(environment.packages)) die("metric environment lock identity missing");
  const required = new Map([["scikit-image", "0.25.2"], ["lpips", "0.1.4"], ["torch", "2.7.0"], ["torchvision", "0.22.0"]]);
  for (const [name, version] of required) if (!environment.packages.some((entry) => entry?.name === name && entry.version === version)) die(`metric package ${name} is not exactly pinned`);
  for (const [key, expected] of [["lpips_linear", "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"], ["alexnet", "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"]]) {
    const weight = environment.weights?.[key];
    if (!weight?.path || weight.sha256 !== expected || sha(await readFile(path.join(metricsRoot, weight.path))) !== expected) die(`official ${key} weight does not match fixed hash`);
  }
  if (!SHA256.test(environment.metric_transcript_sha256)) die("metric transcript identity missing");
  return environment;
}

export async function validateWeightsEnvironment(weightsRoot, revisions) {
  const environment = await json(path.join(weightsRoot, "starvector-terminal-weights-v1.json"));
  for (const [key, revision] of Object.entries(revisions)) {
    const model = environment.models?.[key];
    if (!model?.relative_path || model.revision !== revision || !SHA256.test(model.inventory_sha256)) die(`${key} snapshot identity missing`);
    if ((await inventory(path.join(weightsRoot, model.relative_path))).aggregate_sha256 !== model.inventory_sha256) die(`${key} inventory drifted`);
  }
  return environment;
}

export async function verifyRouteClosure(sceneWorksRoot, command) {
  if (!command || !path.isAbsolute(command)) die("source-owned current-tree production route command required");
  const resolved = path.resolve(command), root = path.resolve(sceneWorksRoot) + path.sep;
  if (!resolved.startsWith(root)) die("production route command must be built from this SceneWorks tree");
  const contents = await readFile(resolved);
  if (/(?:npm|pip|cargo)\s+install|huggingface-cli|\bwget\b|\bcurl\b/i.test(String(contents))) die("production route closure may not download during the campaign");
  const closure = { path: path.relative(sceneWorksRoot, resolved), sha256: sha(contents), sceneworks_revision: await git(sceneWorksRoot, ["rev-parse", "HEAD"]) };
  if (!REVISION.test(closure.sceneworks_revision)) die("route source tree revision missing");
  return closure;
}

export async function preflight({ sceneWorksRoot, planPath, inferenceRoot, weightsRoot, metricsRoot, permanentPin, command }) {
  const { plan, metrics_lock_sha256 } = await readPlanAndLock(planPath);
  await verifyInferenceCheckout(inferenceRoot); await verifyPermanentPin(sceneWorksRoot, permanentPin);
  if (!weightsRoot || !metricsRoot) die("pre-provisioned weights and metrics roots required; network acquisition is forbidden");
  return { plan, metrics_lock_sha256, weights: await validateWeightsEnvironment(weightsRoot, plan.model_snapshot_revisions), metrics: await validateMetricsEnvironment(metricsRoot, metrics_lock_sha256), route: await verifyRouteClosure(sceneWorksRoot, command) };
}

async function failureArtifact(output, context, error) {
  await mkdir(output, { recursive: true });
  const transcript = JSON.stringify({ ...context, status: "failed", error: String(error?.message ?? error), completed_at: new Date().toISOString() }, null, 2) + "\n";
  await writeFile(path.join(output, "controller-failure.json"), transcript);
  await writeFile(path.join(output, "controller-transcript.sha256"), sha(transcript) + "\n");
  await writeFile(path.join(output, "raw-probes-and-transcripts.json"), JSON.stringify({ hardware_probe: process.env.STARVECTOR_TERMINAL_HARDWARE_PROBE ?? null, route_transcript: "route-command-transcript.json", available_before_route_result: true }, null, 2) + "\n");
  await writeFile(path.join(output, "controller-artifacts.json"), JSON.stringify(await inventory(output), null, 2) + "\n");
}

export async function executeTuple({ sceneWorksRoot, planPath, inferenceRoot, weightsRoot, metricsRoot, leaseRoot, command, output, tuple, campaignRunId = randomUUID(), permanentPin }) {
  await mkdir(output, { recursive: true }); let release;
  const context = { campaign_run_id: campaignRunId, permanent_pin: permanentPin, tuple, command };
  try {
    if (!TUPLES.includes(tuple)) die("unsupported tuple");
    const pre = await preflight({ sceneWorksRoot, planPath, inferenceRoot, weightsRoot, metricsRoot, permanentPin, command });
    await claimCampaignMarker(leaseRoot, permanentPin, campaignRunId);
    release = await acquireStableLease(leaseRoot, permanentPin, campaignRunId);
    await writeFile(path.join(output, "preflight-provenance.json"), JSON.stringify({ inference_revision: INFERENCE_REVISION, permanent_pin: permanentPin, route: pre.route, metric_transcript_sha256: pre.metrics.metric_transcript_sha256, model_revisions: pre.plan.model_snapshot_revisions }, null, 2) + "\n");
    try {
      const commandResult = await execFile(command, [], { env: { ...process.env, STARVECTOR_TERMINAL_RUN_ID: campaignRunId, STARVECTOR_TERMINAL_TUPLE: tuple, STARVECTOR_TERMINAL_OUTPUT: output, STARVECTOR_TERMINAL_PERMANENT_PIN: permanentPin, STARVECTOR_TERMINAL_ROUTE_CLOSURE_SHA256: pre.route.sha256 }, maxBuffer: 1024 * 1024 });
      await writeFile(path.join(output, "route-command-transcript.json"), JSON.stringify({ stdout: commandResult.stdout, stderr: commandResult.stderr }, null, 2) + "\n");
    } catch (error) {
      await writeFile(path.join(output, "route-command-transcript.json"), JSON.stringify({ stdout: error.stdout ?? "", stderr: error.stderr ?? "", error: String(error.message ?? error) }, null, 2) + "\n");
      throw error;
    }
    await stat(path.join(output, "raw-results.json"));
    await writeFile(path.join(output, "tuple-controller.json"), JSON.stringify({ ...context, status: "succeeded", route: pre.route, metrics_lock_sha256: pre.metrics_lock_sha256 }, null, 2) + "\n");
  } catch (error) { await failureArtifact(output, context, error); throw error; }
  finally { if (release) await release(); }
}

function canonicalRun(raw, tuple) {
  if (raw?.tuple !== tuple || !raw.run || typeof raw.run !== "object") die(`raw artifact for ${tuple} must carry one canonical run`);
  if (`${raw.run.backend}:${raw.run.tier}` !== tuple) die(`raw ${tuple} run identity mismatches`);
  return raw.run;
}
async function canonicalModule(inferenceRoot) { return import(pathToFileURL(path.join(inferenceRoot, "scripts/release/starvector_terminal_evidence.mjs")).href); }

export async function sealReceipt({ sceneWorksRoot, planPath, inferenceRoot, evidenceRoot, output, campaignRunId, permanentPin }) {
  const { plan } = await readPlanAndLock(planPath); await verifyInferenceCheckout(inferenceRoot); await verifyPermanentPin(sceneWorksRoot, permanentPin);
  const rows = await readdir(evidenceRoot, { recursive: true });
  const rawFiles = rows.filter((name) => name.endsWith("raw-results.json"));
  const suiteFiles = rows.filter((name) => name.endsWith("terminal-suites.json"));
  if (suiteFiles.length !== 1) die("downloaded evidence must contain exactly one nested terminal-suites.json");
  const byTuple = new Map();
  for (const entry of rawFiles) { const raw = await json(path.join(evidenceRoot, entry)); if (TUPLES.includes(raw.tuple)) { if (byTuple.has(raw.tuple)) die(`duplicate raw tuple ${raw.tuple}`); byTuple.set(raw.tuple, canonicalRun(raw, raw.tuple)); } }
  if (byTuple.size !== TUPLES.length) die("downloaded evidence must contain exactly four canonical tuple runs");
  const suites = await json(path.join(evidenceRoot, suiteFiles[0]));
  const validator = await canonicalModule(inferenceRoot);
  const corpus = await json(path.join(inferenceRoot, plan.inference_contract.corpus));
  const sceneworksRevision = await git(sceneWorksRoot, ["rev-parse", "HEAD"]);
  const receipt = { schema_version: 1, campaign_run_id: campaignRunId, inference_revision: INFERENCE_REVISION, sceneworks_revision: sceneworksRevision, corpus_sha256: validator.validatePlan(corpus), execution: suites.execution, producer: suites.producer, metric_identity: suites.metric_identity, inference_preflight: suites.inference_preflight, runs: TUPLES.map((tuple) => byTuple.get(tuple)), hostile_sanitizer: suites.hostile_sanitizer, prompt_composition: suites.prompt_composition, artifact_manifest: null };
  receipt.execution.head_sha = sceneworksRevision;
  const manifest = validator.buildArtifactManifest(receipt, corpus); receipt.artifact_manifest = manifest; receipt.producer.artifact_manifest_sha256 = manifest.aggregate_sha256;
  await mkdir(output, { recursive: true }); const receiptPath = path.join(output, "terminal-receipt.json"); await writeFile(receiptPath, JSON.stringify(receipt, null, 2) + "\n");
  const validatorPath = path.join(inferenceRoot, "scripts/release/starvector_terminal_evidence.mjs");
  await execFile(process.execPath, [validatorPath, "validate-receipt", "--corpus", path.join(inferenceRoot, plan.inference_contract.corpus), "--receipt", receiptPath, "--inference-revision", INFERENCE_REVISION, "--sceneworks-revision", sceneworksRevision]);
  await writeFile(path.join(output, "terminal-artifacts.json"), JSON.stringify(await inventory(evidenceRoot), null, 2) + "\n");
  return receipt;
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [mode, ...args] = process.argv.slice(2); const fail = (error) => { console.error(error.message); process.exitCode = 1; };
  if (mode === "run") { const [sceneWorksRoot, planPath, inferenceRoot, weightsRoot, metricsRoot, leaseRoot, output, tuple, campaignRunId, permanentPin, command] = args; executeTuple({ sceneWorksRoot, planPath, inferenceRoot, weightsRoot, metricsRoot, leaseRoot, output, tuple, campaignRunId, permanentPin, command }).catch(fail); }
  else if (mode === "seal") { const [sceneWorksRoot, planPath, inferenceRoot, evidenceRoot, output, campaignRunId, permanentPin] = args; sealReceipt({ sceneWorksRoot, planPath, inferenceRoot, evidenceRoot, output, campaignRunId, permanentPin }).catch(fail); }
  else fail(new Error("usage: run|seal ..."));
}
