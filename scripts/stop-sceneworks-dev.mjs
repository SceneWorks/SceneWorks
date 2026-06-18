#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = new Set(process.argv.slice(2));
const dryRun = args.has("--dry-run");
const noForce = args.has("--no-force");
const help = args.has("--help") || args.has("-h");
const timeoutMs = Number.parseInt(process.env.SCENEWORKS_DEV_STOP_TIMEOUT_MS || "3000", 10);

if (help) {
  console.log(`Usage: npm run dev:stop -- [--dry-run] [--no-force]

Stops host-mode SceneWorks API/worker processes launched from this checkout.

Options:
  --dry-run   Print matching processes without sending signals.
  --no-force  Do not send SIGKILL to processes that survive SIGTERM.

Environment:
  SCENEWORKS_DEV_STOP_TIMEOUT_MS  Milliseconds to wait after SIGTERM before SIGKILL. Default: 3000.`);
  process.exit(0);
}

if (process.platform === "win32") {
  console.error("dev:stop is currently implemented for macOS/Linux host-mode processes.");
  process.exit(1);
}

function readProcesses() {
  let output;
  try {
    output = execFileSync("ps", ["-axo", "pid=,ppid=,command="], { encoding: "utf8" });
  } catch (error) {
    console.error(`Unable to inspect local processes with ps: ${error.message}`);
    process.exit(1);
  }
  return output
    .split("\n")
    .map((line) => line.match(/^\s*(\d+)\s+(\d+)\s+(.+)$/))
    .filter(Boolean)
    .map((match) => ({
      pid: Number.parseInt(match[1], 10),
      ppid: Number.parseInt(match[2], 10),
      command: match[3],
    }));
}

function isInitialSceneWorksTarget(proc) {
  if (proc.pid === process.pid || proc.pid === process.ppid) {
    return false;
  }
  if (proc.command.includes("scripts/stop-sceneworks-dev.mjs")) {
    return false;
  }

  const command = proc.command.replaceAll("\\ ", " ");
  const repoPath = `${repoRoot}/`;
  const launchedFromRepo = command.includes(repoPath) || command.includes(`cd ${repoRoot};`);
  const sceneWorksRuntime =
    /\bsceneworks-rust-(api|worker)\b/.test(command) ||
    command.includes("SCENEWORKS_WORKER_ONLY=1") ||
    command.includes("apps/worker/scene_worker");

  return launchedFromRepo && sceneWorksRuntime;
}

function collectTargets(processes) {
  const byParent = new Map();
  for (const proc of processes) {
    if (!byParent.has(proc.ppid)) {
      byParent.set(proc.ppid, []);
    }
    byParent.get(proc.ppid).push(proc);
  }

  const targets = new Map();
  const queue = processes.filter(isInitialSceneWorksTarget);
  while (queue.length > 0) {
    const proc = queue.shift();
    if (!proc || targets.has(proc.pid) || proc.pid === process.pid || proc.pid === process.ppid) {
      continue;
    }
    targets.set(proc.pid, proc);
    for (const child of byParent.get(proc.pid) || []) {
      queue.push(child);
    }
  }
  return [...targets.values()].sort((a, b) => b.pid - a.pid);
}

function signalProcess(pid, signal) {
  try {
    process.kill(pid, signal);
    return true;
  } catch (error) {
    if (error.code === "ESRCH") {
      return false;
    }
    throw error;
  }
}

function sleep(ms) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

function liveTargetPids(pids) {
  const live = new Set(readProcesses().map((proc) => proc.pid));
  return pids.filter((pid) => live.has(pid));
}

const targets = collectTargets(readProcesses());
if (targets.length === 0) {
  console.log("No host-mode SceneWorks API/worker processes found for this checkout.");
  process.exit(0);
}

console.log(`Matched ${targets.length} SceneWorks dev process${targets.length === 1 ? "" : "es"}:`);
for (const proc of targets) {
  console.log(`  ${proc.pid}\t${proc.command}`);
}

if (dryRun) {
  process.exit(0);
}

for (const proc of targets) {
  signalProcess(proc.pid, "SIGTERM");
}

const deadline = Date.now() + (Number.isFinite(timeoutMs) && timeoutMs >= 0 ? timeoutMs : 3000);
let remaining = liveTargetPids(targets.map((proc) => proc.pid));
while (remaining.length > 0 && Date.now() < deadline) {
  sleep(100);
  remaining = liveTargetPids(remaining);
}

if (remaining.length > 0 && !noForce) {
  for (const pid of remaining) {
    signalProcess(pid, "SIGKILL");
  }
  sleep(100);
  remaining = liveTargetPids(remaining);
}

if (remaining.length > 0) {
  console.error(
    `Failed to stop ${remaining.length} process${remaining.length === 1 ? "" : "es"}: ${remaining.join(", ")}`,
  );
  process.exit(1);
}

console.log("Stopped SceneWorks host-mode API/worker processes.");
