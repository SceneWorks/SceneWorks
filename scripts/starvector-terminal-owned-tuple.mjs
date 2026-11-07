#!/usr/bin/env node
// One workflow-owned start -> unchanged assets/bundle/producer -> stop lifetime. Required by
// Windows Job ownership: guarding a short-lived start CLI alone would wait forever for services.
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { withMacMetalExecutionLease, scopedMetalChildOptions } from "./lib/apple-metal-lease.mjs";

export function ownedTupleCommands({ root, output, permanentPin, url, weightsRoot, tuple, campaignRunId, env = process.env, pathImpl = path }) {
  const required = [root, output, permanentPin, url, weightsRoot, tuple, campaignRunId,
    env.STARVECTOR_TERMINAL_INFERENCE_ROOT, env.STARVECTOR_TERMINAL_CORPUS_ASSETS_ROOT,
    env.STARVECTOR_TERMINAL_CASE_BUNDLE, env.STARVECTOR_TERMINAL_METRICS_ROOT,
    env.STARVECTOR_TERMINAL_LEASE_ROOT, env.STARVECTOR_TERMINAL_LEASE_HELPER,
    env.STARVECTOR_TERMINAL_PRODUCT_ROUTE_COMMAND];
  if (required.some((value) => typeof value !== "string" || value.length === 0)) throw new Error("exact existing terminal tuple inputs are required");
  const script = (name) => pathImpl.join(root, "scripts", name);
  const assets = pathImpl.join(output, "imported-assets.json");
  return {
    start: { root, output, permanentPin, url, weightsRoot, tuple },
    commands: [
      [script("starvector-terminal-assets.mjs"), [env.STARVECTOR_TERMINAL_CORPUS_ASSETS_ROOT, url, tuple, assets]],
      [script("starvector-terminal-case-bundle.mjs"), [pathImpl.join(env.STARVECTOR_TERMINAL_INFERENCE_ROOT, "release", "starvector-terminal-corpus-v1.json"), env.STARVECTOR_TERMINAL_CORPUS_ASSETS_ROOT, env.STARVECTOR_TERMINAL_CASE_BUNDLE, permanentPin, assets]],
      [script("starvector-terminal-producer.mjs"), ["run", root, "release/starvector-terminal-campaign-v1.json", env.STARVECTOR_TERMINAL_INFERENCE_ROOT, weightsRoot, env.STARVECTOR_TERMINAL_METRICS_ROOT, env.STARVECTOR_TERMINAL_LEASE_ROOT, env.STARVECTOR_TERMINAL_LEASE_HELPER, output, tuple, campaignRunId, permanentPin, env.STARVECTOR_TERMINAL_PRODUCT_ROUTE_COMMAND]],
    ],
  };
}
async function runNode(script, args, env) {
  const child = spawn(process.execPath, [script, ...args], scopedMetalChildOptions({ env, stdio: "inherit", windowsHide: true }));
  await new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("close", (code, signal) => code === 0 && !signal ? resolve() : reject(new Error(`terminal child failed: ${path.basename(script)} code=${code} signal=${signal}`)));
  });
}
export async function runOwnedTuple(options, { start, stop, execute = runNode } = {}) {
  const calls = ownedTupleCommands(options);
  if (!start || !stop) {
    const service = await import("./starvector-terminal-product-service.mjs");
    start ??= service.startProductService; stop ??= service.stopProductService;
  }
  let primary = null;
  try {
    await start(calls.start);
    for (const [script, args] of calls.commands) await execute(script, args, options.env ?? process.env);
  } catch (error) { primary = error; }
  // Exactly one old stop operation, as the workflow's old always() step did. Do not assume its
  // non-idempotent API can be called twice; no new service API or PID signalling is introduced.
  try { await stop(options.output); }
  catch (error) { if (primary) primary.message += `; service stop also failed: ${error.message}`; else primary = error; }
  if (primary) throw primary;
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [root, output, permanentPin, url, weightsRoot, tuple, campaignRunId, budgetMinutes] = process.argv.slice(2);
  const budget = Number(budgetMinutes);
  const run = async () => {
    if (!Number.isFinite(budget) || budget <= 0) throw new Error("existing workflow acquisition budget is required");
    await withMacMetalExecutionLease(budget * 60, async () => {
      // Only defer once execution ownership exists. A cancellation while still queued must not
      // later start a model. Windows cancellation remains owned by the closed Windows Job guard.
      let interrupted = null;
      const interrupt = () => { interrupted ??= 130; }, terminate = () => { interrupted ??= 143; };
      if (process.platform === "darwin") { process.on("SIGINT", interrupt); process.on("SIGTERM", terminate); }
      try {
        await runOwnedTuple({ root, output, permanentPin, url, weightsRoot, tuple, campaignRunId });
        if (interrupted !== null) process.exitCode = interrupted;
      } finally {
        if (process.platform === "darwin") { process.removeListener("SIGINT", interrupt); process.removeListener("SIGTERM", terminate); }
      }
    });
  };
  run().catch((error) => { console.error(error.message); process.exitCode = 1; });
}
