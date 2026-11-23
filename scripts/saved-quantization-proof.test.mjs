import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve, relative, isAbsolute } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

const supported = process.platform !== "win32" || /^D:[\\/]/i.test(tmpdir());
const controller = resolve("scripts/ci/saved-quantization-proof.mjs");
function fixture(t) {
  const base = mkdtempSync(join(tmpdir(), "saved-quantization-proof-test-"));
  t.after(() => {
    const child = relative(tmpdir(), base);
    assert.ok(!isAbsolute(child) && child.startsWith("saved-quantization-proof-test-"));
    rmSync(base, { recursive: true, force: true });
  });
  const root = join(base, "proof");
  const workspace = join(base, "workspace");
  mkdirSync(root);
  mkdirSync(workspace);
  writeFileSync(join(root, ".saved-quantization-proof"), "123:1\n");
  const env = { ...process.env, GITHUB_ACTIONS: "true", GITHUB_RUN_ID: "123", GITHUB_RUN_ATTEMPT: "1", GITHUB_WORKSPACE: workspace, RUNNER_TEMP: base };
  const run = (overrides = {}) => spawnSync(process.execPath, [controller, "--stage-receipts", "--backend", "candle", "--root", root], { env: { ...env, ...overrides }, encoding: "utf8" });
  return { root, workspace, base, run };
}

test("receipt staging keeps only this attempt's JSON metadata inside the artifact workspace", { skip: !supported }, (t) => {
  const { root, workspace, base, run } = fixture(t);
  writeFileSync(join(root, "warm.json"), JSON.stringify({ cacheHits: 2, quantizerRuns: 0 }));
  writeFileSync(join(root, "weights.safetensors"), "packed weights stay out of artifacts");
  const guard = join(base, "sceneworks-cuda-execution-123-1-saved-quantization");
  mkdirSync(guard);
  writeFileSync(join(guard, "resource-receipt.json"), JSON.stringify({ resource_completed: true }));
  const child = run();
  assert.equal(child.status, 0, child.stderr);
  const destination = join(workspace, "saved-quantization-receipts", "123-1");
  assert.deepEqual(readdirSync(destination).sort(), ["resource-receipt.json", "warm.json"]);
  assert.equal(JSON.parse(readFileSync(join(destination, "warm.json"))).quantizerRuns, 0);
  assert.equal(readFileSync(join(root, "weights.safetensors"), "utf8"), "packed weights stay out of artifacts");
});

test("receipt staging refuses a foreign attempt before creating artifact paths", { skip: !supported }, (t) => {
  const { root, workspace, run } = fixture(t);
  writeFileSync(join(root, ".saved-quantization-proof"), "another-run:1");
  const child = run();
  assert.notEqual(child.status, 0);
  assert.match(child.stderr, /ownership marker/);
  assert.equal(existsSync(join(workspace, "saved-quantization-receipts")), false);
});

test("receipt staging cannot accidentally enter numerical execution outside Actions", { skip: !supported }, (t) => {
  const { workspace, run } = fixture(t);
  const child = run({ GITHUB_ACTIONS: "false" });
  assert.notEqual(child.status, 0);
  assert.match(child.stderr, /requires an existing owned Actions proof/);
  assert.equal(existsSync(join(workspace, "saved-quantization-receipts")), false);
});
