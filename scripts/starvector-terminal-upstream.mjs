#!/usr/bin/env node
import { execFile as callback } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { promisify } from "node:util";
import path from "node:path";
import { acquireStableLease, claimTupleMarker, verifyPermanentPin, inventory } from "./starvector-terminal-producer.mjs";
import { claimTerminalAttempt } from "./lib/starvector-terminal-attempt.mjs";
import { verifyRecovery } from "./starvector-terminal-recovery.mjs";
import { readPlanAndLock, validateTerminalDispatchInputs } from "./starvector-terminal-campaign.mjs";
import { terminalGpuBinding, terminalGpuEnvironment } from "./lib/starvector-terminal-gpu.mjs";
import { loadUpstreamReference } from "./lib/starvector-terminal-upstream-reference.mjs";
import { isExecutedModule } from "./starvector-terminal-cli.mjs";
const execFile = promisify(callback);
const json = async file => JSON.parse(await readFile(file, "utf8"));

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

export async function runUpstream(sceneWorksRoot, output) {
  const options = upstreamOptions(sceneWorksRoot), pin = process.env.STARVECTOR_TERMINAL_PERMANENT_PIN, campaign = process.env.STARVECTOR_TERMINAL_CAMPAIGN_RUN_ID;
  const { plan } = await readPlanAndLock(path.join(sceneWorksRoot, "release/starvector-terminal-campaign-v1.json"));
  validateTerminalDispatchInputs(plan, pin, campaign); await verifyPermanentPin(sceneWorksRoot, pin);
  const validated = await validateUpstreamInputs(options, output);
  const binding = await terminalGpuBinding();
  if (binding.backend !== "candle") throw new Error("upstream reference requires the qualified CUDA lane");
  const predecessor = await verifyRecovery(await json(path.join(sceneWorksRoot, "release/starvector-terminal-recovery-v1.json")), process.env.STARVECTOR_TERMINAL_RECOVERY_ROOT ?? path.join(process.env.RUNNER_TEMP, "starvector-recovery"), { campaignRunId: campaign, permanentPin: pin });
  const release = await acquireStableLease(process.env.STARVECTOR_TERMINAL_LEASE_ROOT, process.env.STARVECTOR_TERMINAL_LEASE_HELPER, pin, campaign);
  try {
    await claimTerminalAttempt(process.env.STARVECTOR_TERMINAL_LEASE_ROOT, pin, campaign, { workflowRunId: process.env.GITHUB_RUN_ID, workflowRunAttempt: Number(process.env.GITHUB_RUN_ATTEMPT), predecessor });
    await claimTupleMarker(process.env.STARVECTOR_TERMINAL_LEASE_ROOT, pin, campaign, "upstream-reference");
    await mkdir(output, { recursive: true });
    for (const tier of ["1b", "8b"]) {
      const args = [path.join(sceneWorksRoot, "scripts/starvector-terminal-upstream-oracle.py"), "prepare", "--upstream-root", options.upstreamRoot, "--weights-root", options.weightsRoot, "--assets-root", options.assetsRoot, "--output", output, "--components-root", options.componentsRoot, "--sanitizer", options.sanitizer, "--tier", tier, "--device", "cuda:0"];
      await execFile(options.python, args, { env: { ...process.env, ...terminalGpuEnvironment(binding), HF_HUB_OFFLINE: "1", TRANSFORMERS_OFFLINE: "1" }, timeout: 3700 * 1000, maxBuffer: 1024 * 1024 });
    }
    const rows = (await json(path.join(options.assetsRoot, "starvector-terminal-row-index-v1.json"))).rows;
    for (const tier of ["1b", "8b"]) await loadUpstreamReference(output, tier, rows);
    await writeFile(path.join(output, "upstream-controller.json"), JSON.stringify({ schema_version: 1, campaign_run_id: campaign, inference_revision: pin, workflow_run_id: process.env.GITHUB_RUN_ID, workflow_run_attempt: Number(process.env.GITHUB_RUN_ATTEMPT), sceneworks_revision: process.env.GITHUB_SHA, gpu_binding: binding, validated, artifacts: await inventory(output) }, null, 2) + "\n", { flag: "wx" });
  } finally { await release(); }
}
if (isExecutedModule(import.meta.url)) {
  const [mode, root, output] = process.argv.slice(2);
  (mode === "validate" ? validateUpstreamInputs(upstreamOptions(root), output) : mode === "run" ? runUpstream(root, output) : Promise.reject(new Error("usage: validate|run <sceneworks-root> <output>"))).catch(error => { console.error(error.message); process.exitCode = 1; });
}
