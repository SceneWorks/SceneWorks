import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { isExecutedModule } from "./starvector-terminal-cli.mjs";
import { sourceInstallEnvironment, sourceInstallRequests, stopSourceChildren, windowsTaskkillArguments } from "./starvector-terminal-source-install.mjs";

test("terminal CLIs compare platform paths through canonical file URLs", () => {
  const executable = path.resolve("scripts/starvector-terminal-source-install.mjs");
  assert.equal(isExecutedModule(pathToFileURL(executable).href, executable), true);
  assert.equal(isExecutedModule(pathToFileURL(`${executable}.other`).href, executable), false);
});

test("source preparation uses only the three typed Model Manager installs", () => {
  assert.deepEqual(sourceInstallRequests(), [
    { modelId: "starvector_1b", body: { requestedGpu: "cpu" } },
    { modelId: "starvector_8b", body: { requestedGpu: "cpu" } },
    { modelId: "flux_schnell", body: { requestedGpu: "cpu", variant: "q4" } },
  ]);
});

test("source preparation clears terminal/offline inheritance and isolates both product roots", () => {
  const root = path.resolve("source-root");
  const { env, dataDir, hfHome } = sourceInstallEnvironment(root, "http://127.0.0.1:17831", {
    HF_HUB_OFFLINE: "1",
    TRANSFORMERS_OFFLINE: "1",
    SCENEWORKS_TERMINAL_CAMPAIGN: "1",
    SCENEWORKS_TERMINAL_NO_JOB_DOWNLOADS: "1",
  });
  assert.equal(dataDir, path.join(root, "app-data"));
  assert.equal(hfHome, path.join(root, "hf-home"));
  assert.equal(env.HF_HOME, hfHome);
  assert.equal(env.HF_HUB_CACHE, path.join(hfHome, "hub"));
  assert.equal(env.SCENEWORKS_GPU_ID, "cpu");
  for (const name of ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "SCENEWORKS_TERMINAL_CAMPAIGN", "SCENEWORKS_TERMINAL_NO_JOB_DOWNLOADS"]) assert.equal(env[name], undefined);
});

test("source preparation escalates stubborn Windows children through taskkill", async () => {
  const signals = [];
  const child = Object.assign(new EventEmitter(), {
    exitCode: null,
    signalCode: null,
    pid: 4242,
    kill(signal) { signals.push(signal); return true; },
  });
  const taskkill = async (pid) => {
    assert.equal(pid, 4242);
    child.signalCode = "SIGKILL";
    child.emit("exit", null, "SIGKILL");
  };
  await stopSourceChildren([child], { platform: "win32", taskkill, timeoutMs: 1 });
  assert.deepEqual(signals, ["SIGTERM"]);
  assert.deepEqual(windowsTaskkillArguments(4242), ["/PID", "4242", "/T", "/F"]);
});

test("source preparation escalates stubborn POSIX children through SIGKILL", async () => {
  const signals = [];
  const child = Object.assign(new EventEmitter(), {
    exitCode: null,
    signalCode: null,
    pid: 4243,
    kill(signal) {
      signals.push(signal);
      if (signal === "SIGKILL") {
        this.signalCode = signal;
        this.emit("exit", null, signal);
      }
      return true;
    },
  });
  await stopSourceChildren([child], { platform: "linux", timeoutMs: 1 });
  assert.deepEqual(signals, ["SIGTERM", "SIGKILL"]);
});

test("source preparation recognizes signal-terminated children as stopped", async () => {
  const child = {
    exitCode: null,
    signalCode: "SIGTERM",
    pid: 4244,
    kill() { assert.fail("a signal-terminated child must not be killed again"); },
  };
  await stopSourceChildren([child], { timeoutMs: 1 });
});

test("source preparation remains bounded when taskkill itself hangs", async () => {
  const child = Object.assign(new EventEmitter(), {
    exitCode: null,
    signalCode: null,
    pid: 4245,
    kill() { return true; },
  });
  await assert.rejects(
    stopSourceChildren([child], {
      platform: "win32",
      taskkill: async () => new Promise(() => {}),
      timeoutMs: 1,
    }),
    /did not stop \(pids: 4245\)/,
  );
});

test("source workflow is preparation-only on both real-weight hosts", async () => {
  const workflow = await readFile(new URL("../.github/workflows/starvector-terminal-source.yml", import.meta.url), "utf8");
  assert.match(workflow, /runs-on: \[self-hosted, macOS, ARM64, rw-starvector\]/);
  assert.match(workflow, /runs-on: \[self-hosted, Windows, X64, cuda, real-weights\]/);
  assert.equal((workflow.match(/starvector-terminal-source-install\.mjs/g) ?? []).length, 2);
  assert.doesNotMatch(workflow, /starvector-terminal-(?:producer|campaign|route)\.mjs|vector_generate|campaign_run_id/i);
});

test("Windows terminal workflows use the in-box PowerShell host", async () => {
  const workflows = await Promise.all([
    "source",
    "provision",
    "readiness",
    "",
  ].map((suffix) => readFile(new URL(`../.github/workflows/starvector-terminal${suffix ? `-${suffix}` : ""}.yml`, import.meta.url), "utf8")));
  for (const workflow of workflows) assert.doesNotMatch(workflow, /shell: pwsh/);
  assert.equal(workflows.reduce((count, workflow) => count + (workflow.match(/shell: powershell/g) ?? []).length, 0), 19);
});

test("existing default-branch workflow bridges every pre-merge terminal operation", async () => {
  const bridge = await readFile(new URL("../.github/workflows/server-candle-linux.yml", import.meta.url), "utf8");
  for (const operation of ["source", "provision", "readiness", "campaign"]) {
    assert.match(bridge, new RegExp(`starvector_terminal_operation == '${operation}'`));
  }
  for (const reusable of ["source", "provision", "readiness", ""]) {
    const suffix = reusable ? `-${reusable}` : "";
    assert.match(bridge, new RegExp(`uses: \\.\\/.github/workflows/starvector-terminal${suffix}\\.yml`));
  }
  assert.match(bridge, /options: \[standard, source, provision, readiness, campaign\]/);
  assert.match(bridge, /starvector-readiness:[\s\S]*?uses: \.\/.github\/workflows\/starvector-terminal-readiness\.yml[\s\S]*?with:[\s\S]*?permanent_pin: \$\{\{ inputs\.permanent_pin \}\}/);
});
