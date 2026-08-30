import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { isExecutedModule } from "./starvector-terminal-cli.mjs";
import { sourceInstallEnvironment, sourceInstallRequests } from "./starvector-terminal-source-install.mjs";

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
  assert.equal(workflows.reduce((count, workflow) => count + (workflow.match(/shell: powershell/g) ?? []).length, 0), 11);
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
});
