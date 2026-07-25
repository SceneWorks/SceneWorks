import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function source(path) {
  return readFile(new URL(`../${path}`, import.meta.url), "utf8");
}

test("Windows workflows watch the local Rust runner action", async () => {
  for (const workflow of [
    ".github/workflows/windows-candle.yml",
    ".github/workflows/desktop-windows.yml",
  ]) {
    assert.match(
      await source(workflow),
      /^\s+- "\.github\/actions\/prepare-rust-runner\/\*\*"/m,
      workflow,
    );
  }
});

test("Docker relevance gate paginates and checks for truncated file lists", async () => {
  const workflow = await source(".github/workflows/check.yml");
  assert.match(workflow, /gh api --paginate/);
  assert.match(workflow, /docker-smoke-relevance\.mjs --expected-count/);
  assert.doesNotMatch(workflow, /gh pr view .*--json files/);
});

test("every release job is confined to refs/tags/v", async () => {
  const workflow = await source(".github/workflows/release.yml");
  const jobConditions = [...workflow.matchAll(/^\s{4}if:\s*(.+)$/gm)].map((match) => match[1]);
  assert.equal(jobConditions.length, 3);
  for (const condition of jobConditions) {
    assert.match(condition, /startsWith\(github\.ref, 'refs\/tags\/v'\)/);
  }
  assert.ok(
    workflow.includes('if [[ "${TAG#v}" == *-* ]]; then'),
    "prerelease classification must use the validated v-tag",
  );
});

test("Lens smoke only terminates processes it started", async () => {
  const script = await source("scripts/smoke-lens.ps1");
  assert.doesNotMatch(script, /Get-Process/);
  assert.match(script, /taskkill \/F \/T \/PID \$\(\$p\.Id\)/);
});

test("health check defaults to the compose API port", async () => {
  assert.match(
    await source("scripts/check-health.mjs"),
    /http:\/\/localhost:8010/,
  );
});

test("Docker cleanup relies on the configured host uid instead of a root container", async () => {
  const script = await source("scripts/check-docker-api-runtime.mjs");
  assert.doesNotMatch(script, /--entrypoint", "rm"/);
  assert.match(script, /SCENEWORKS_UID/);
});

test("all three manifest scripts import the shared JSONC parser", async () => {
  for (const scriptPath of [
    "scripts/check-scaffold.mjs",
    "scripts/check-download-patterns.mjs",
    "scripts/check-no-nc-weights.mjs",
  ]) {
    const script = await source(scriptPath);
    assert.match(script, /import \{ stripJsoncComments \} from "\.\/lib\/jsonc\.mjs";/);
    assert.doesNotMatch(script, /function stripJsoncComments/);
  }
});
