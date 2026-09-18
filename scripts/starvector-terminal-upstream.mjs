#!/usr/bin/env node
import { execFile as callback } from "node:child_process";
import { createHash } from "node:crypto";
import { lstat, mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { promisify } from "node:util";
import path from "node:path";
import { acquireStableLease, claimTupleMarker, verifyPermanentPin, inventory } from "./starvector-terminal-producer.mjs";
import { claimTerminalAttempt } from "./lib/starvector-terminal-attempt.mjs";
import { verifyExecutionPredecessor, verifyRecovery } from "./starvector-terminal-recovery.mjs";
import { readPlanAndLock, validateTerminalDispatchInputs } from "./starvector-terminal-campaign.mjs";
import { terminalGpuBinding, terminalGpuEnvironment } from "./lib/starvector-terminal-gpu.mjs";
import { loadUpstreamReference, PARITY_SOURCE_INDICES } from "./lib/starvector-terminal-upstream-reference.mjs";
import { isExecutedModule } from "./starvector-terminal-cli.mjs";
const execFile = promisify(callback);
const json = async file => JSON.parse(await readFile(file, "utf8"));
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
export const UPSTREAM_TIER_TIMEOUT_MS = 7_300 * 1_000;

async function regularFile(file, description) {
  const info = await lstat(file);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error(`upstream reference: ${description} is not a regular file`);
  return file;
}

async function regularDirectory(directory, description) {
  const info = await lstat(directory);
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error(`upstream reference: ${description} is not a regular directory`);
  return directory;
}

export async function verifyCollectedRejections(output, tier, rows) {
  if (!path.isAbsolute(output) || !["1b", "8b"].includes(tier) || rows.length < 95) throw new Error("upstream reference: invalid rejection verification inputs");
  const tierRoot = path.join(output, `upstream-${tier}`);
  await regularDirectory(tierRoot, "rejection tier root");
  const transcript = await regularFile(path.join(tierRoot, "transcript.jsonl"), "rejection transcript");
  let events;
  try { events = (await readFile(transcript, "utf8")).trimEnd().split(/\r?\n/).map(line => JSON.parse(line)); }
  catch { throw new Error("upstream reference: rejection transcript is invalid"); }
  const starts = events.filter(event => event?.event === "case_started");
  const results = events.filter(event => ["case_completed", "case_rejected"].includes(event?.event));
  if (starts.length !== 20 || results.length !== 20) throw new Error("upstream reference: rejection transcript does not cover all planned cases");
  for (let index = 0; index < 20; index += 1) {
    const source = PARITY_SOURCE_INDICES[index], row = rows[source];
    for (const event of [starts[index], results[index]]) {
      if (event.case_index !== index || event.source_case_index !== source || event.seed !== index || event.input_png_sha256 !== row?.png_sha256) throw new Error("upstream reference: rejection transcript case identity differs");
    }
  }
  const rejected = results.filter(event => event.event === "case_rejected");
  const keys = ["case_index", "source_case_index", "seed", "input_png_sha256", "error_code", "raw_svg_sha256", "sanitizer_stdout", "sanitizer_stderr"];
  const summaries = rejected.map(event => Object.fromEntries(keys.map(key => [key, event[key]])));
  const final = events.at(-1);
  if (!rejected.length || final?.event !== "failed" || final.failure_kind !== "svg_case_rejections" || final.collected_cases !== 20 || final.completed_cases !== 20 - rejected.length || JSON.stringify(final.rejected_cases) !== JSON.stringify(summaries)) throw new Error("upstream reference: rejection transcript has no authenticated terminal rejection summary");
  for (const item of summaries) {
    const caseRoot = path.join(tierRoot, `case-${String(item.case_index).padStart(2, "0")}`);
    await regularDirectory(caseRoot, "rejected case root");
    const raw = await regularFile(path.join(caseRoot, "raw.svg"), "rejected raw SVG");
    const stdout = await regularFile(path.join(caseRoot, "sanitizer.stdout.log"), "rejected sanitizer stdout");
    await regularFile(path.join(caseRoot, "sanitizer.stderr.log"), "rejected sanitizer stderr");
    const sanitizer = JSON.parse(await readFile(stdout, "utf8"));
    if (hash(await readFile(raw)) !== item.raw_svg_sha256 || item.sanitizer_stdout !== path.relative(output, stdout).split(path.sep).join("/") || item.sanitizer_stderr !== path.relative(output, path.join(caseRoot, "sanitizer.stderr.log")).split(path.sep).join("/") || sanitizer.outcome !== "rejected" || sanitizer.error_code !== item.error_code || sanitizer.canonical_svg_sha256 !== null || sanitizer.preview_png_sha256 !== null || JSON.stringify(sanitizer.published_paths) !== "[]" || JSON.stringify(sanitizer.staging_residue) !== "[]" || sanitizer.result_contains_inline_svg !== false) throw new Error("upstream reference: rejected case evidence differs from transcript");
  }
  return summaries;
}

export async function promoteUpstreamManifests(output, operations = {}) {
  const move = operations.rename ?? rename;
  const read = operations.readFile ?? readFile;
  const write = operations.writeFile ?? writeFile;
  const remove = operations.unlink ?? unlink;
  const inspect = operations.lstat ?? lstat;
  const entries = ["1b", "8b"].map(tier => ({
    pending: path.join(output, `upstream-reference-${tier}.pending.json`),
    canonical: path.join(output, `upstream-reference-${tier}.json`),
  }));
  for (const entry of entries) entry.bytes = await read(entry.pending);
  const promoted = [];
  try {
    for (const entry of entries) {
      await move(entry.pending, entry.canonical);
      promoted.push(entry);
    }
  } catch (promotionError) {
    const rollbackErrors = [];
    for (const entry of promoted.reverse()) {
      try {
        await move(entry.canonical, entry.pending);
      } catch (renameError) {
        try {
          await write(entry.pending, entry.bytes, { flag: "wx" });
          await remove(entry.canonical);
        } catch (fallbackError) {
          rollbackErrors.push(renameError, fallbackError);
        }
      }
    }
    const residues = [];
    for (const entry of entries) {
      try { await inspect(entry.canonical); residues.push(entry.canonical); }
      catch (error) { if (error?.code !== "ENOENT") rollbackErrors.push(error); }
    }
    if (rollbackErrors.length || residues.length) throw new AggregateError([promotionError, ...rollbackErrors], `upstream reference: manifest promotion rollback failed; canonical residue: ${residues.join(", ") || "unknown"}`);
    throw promotionError;
  }
}

export function upstreamOptions(sceneWorksRoot, env = process.env) {
  const root = env.STARVECTOR_TERMINAL_ROOT;
  if (!root || !path.isAbsolute(root)) throw new Error("upstream host root must be absolute");
  return { sceneWorksRoot, python: env.STARVECTOR_TERMINAL_UPSTREAM_PYTHON ?? path.join(root, "upstream-env", "Scripts", "python.exe"), upstreamRoot: path.join(root, "upstream-source"), componentsRoot: path.join(root, "upstream-components"), weightsRoot: env.STARVECTOR_TERMINAL_WEIGHTS_ROOT ?? path.join(root, "weights"), assetsRoot: env.STARVECTOR_TERMINAL_CORPUS_ASSETS_ROOT, sanitizer: env.STARVECTOR_TERMINAL_SANITIZER ?? path.join(sceneWorksRoot, "target", "release", "starvector_terminal_sanitize.exe") };
}

export async function validateUpstreamInputs(options, output, execute = execFile) {
  const args = [path.join(options.sceneWorksRoot, "scripts/starvector-terminal-upstream-oracle.py"), "validate", "--upstream-root", options.upstreamRoot, "--weights-root", options.weightsRoot, "--assets-root", options.assetsRoot, "--output", output, "--components-root", options.componentsRoot, "--sanitizer", options.sanitizer];
  const reports = [];
  for (const tier of ["1b", "8b"]) reports.push(JSON.parse((await execute(options.python, [...args, "--tier", tier], { env: { ...process.env, HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" }, timeout: 30 * 60 * 1000, maxBuffer: 1024 * 1024 })).stdout));
  return reports;
}

export async function produceUpstreamReferences(options, output, binding, rows, execute = execFile, loadReference = loadUpstreamReference) {
  const rejectedTiers = [];
  for (const tier of ["1b", "8b"]) {
    const args = [path.join(options.sceneWorksRoot, "scripts/starvector-terminal-upstream-oracle.py"), "prepare", "--upstream-root", options.upstreamRoot, "--weights-root", options.weightsRoot, "--assets-root", options.assetsRoot, "--output", output, "--components-root", options.componentsRoot, "--sanitizer", options.sanitizer, "--tier", tier, "--device", "cuda:0", "--defer-manifest"];
    try {
      await execute(options.python, args, { env: { ...process.env, ...terminalGpuEnvironment(binding), HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" }, timeout: UPSTREAM_TIER_TIMEOUT_MS, maxBuffer: 1024 * 1024 });
    } catch (error) {
      if (error?.code !== 3) throw error;
      await verifyCollectedRejections(output, tier, rows);
      rejectedTiers.push(tier);
    }
  }
  if (rejectedTiers.length) {
    const error = new Error(`upstream SVG case collection rejected one or more cases for tier(s): ${rejectedTiers.join(", ")}; preserved all bounded case evidence`);
    error.code = 3;
    throw error;
  }
  for (const tier of ["1b", "8b"]) await loadReference(output, tier, rows, { pending: true });
  await promoteUpstreamManifests(output);
}

export async function runUpstream(sceneWorksRoot, output) {
  const options = upstreamOptions(sceneWorksRoot), pin = process.env.STARVECTOR_TERMINAL_PERMANENT_PIN, campaign = process.env.STARVECTOR_TERMINAL_CAMPAIGN_RUN_ID;
  const { plan } = await readPlanAndLock(path.join(sceneWorksRoot, "release/starvector-terminal-campaign-v1.json"));
  validateTerminalDispatchInputs(plan, pin, campaign); await verifyPermanentPin(sceneWorksRoot, pin);
  const validated = await validateUpstreamInputs(options, output);
  const binding = await terminalGpuBinding();
  if (binding.backend !== "candle") throw new Error("upstream reference requires the qualified CUDA lane");
  const recovery = await json(path.join(sceneWorksRoot, "release/starvector-terminal-recovery-v1.json")), recoveryRoot = process.env.STARVECTOR_TERMINAL_RECOVERY_ROOT ?? path.join(process.env.RUNNER_TEMP, "starvector-recovery");
  const nativePredecessor = await verifyRecovery(recovery, recoveryRoot, { campaignRunId: campaign, permanentPin: pin });
  const predecessor = await verifyExecutionPredecessor(recovery, recoveryRoot, nativePredecessor);
  const release = await acquireStableLease(process.env.STARVECTOR_TERMINAL_LEASE_ROOT, process.env.STARVECTOR_TERMINAL_LEASE_HELPER, pin, campaign);
  try {
    await claimTerminalAttempt(process.env.STARVECTOR_TERMINAL_LEASE_ROOT, pin, campaign, { workflowRunId: process.env.GITHUB_RUN_ID, workflowRunAttempt: Number(process.env.GITHUB_RUN_ATTEMPT), predecessor });
    await claimTupleMarker(process.env.STARVECTOR_TERMINAL_LEASE_ROOT, pin, campaign, "upstream-reference");
    await mkdir(output, { recursive: true });
    const rows = (await json(path.join(options.assetsRoot, "starvector-terminal-row-index-v1.json"))).rows;
    await produceUpstreamReferences(options, output, binding, rows);
    await writeFile(path.join(output, "upstream-controller.json"), JSON.stringify({ schema_version: 1, campaign_run_id: campaign, inference_revision: pin, workflow_run_id: process.env.GITHUB_RUN_ID, workflow_run_attempt: Number(process.env.GITHUB_RUN_ATTEMPT), sceneworks_revision: process.env.GITHUB_SHA, gpu_binding: binding, validated, artifacts: await inventory(output) }, null, 2) + "\n", { flag: "wx" });
  } finally { await release(); }
}
if (isExecutedModule(import.meta.url)) {
  const [mode, root, output] = process.argv.slice(2);
  (mode === "validate" ? validateUpstreamInputs(upstreamOptions(root), output) : mode === "run" ? runUpstream(root, output) : Promise.reject(new Error("usage: validate|run <sceneworks-root> <output>"))).catch(error => { console.error(error.message); process.exitCode = error?.code === 3 ? 3 : 1; });
}
