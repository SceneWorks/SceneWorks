import assert from "node:assert/strict";
import { mkdtemp, writeFile, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { executeNodeRoute, verifyTupleExecution } from "./starvector-terminal-producer.mjs";

test("route uses the Node executable even when the script is not executable and its path has spaces", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "terminal execution "));
  t.after(() => rm(root, { recursive: true, force: true }));
  const script = path.join(root, "route without shebang.mjs");
  await writeFile(script, 'console.log(JSON.stringify({runtime:process.execPath,result:"executed"}));', { mode: 0o600 });
  const observed = JSON.parse((await executeNodeRoute(script)).stdout);
  assert.equal(observed.runtime, process.execPath);
  assert.equal(observed.result, "executed");
});

test("current tuple provenance permits equal deterministic bytes but rejects historical run identity", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "terminal proof "));
  t.after(() => rm(root, { recursive: true, force: true }));
  const raw = '{"same":"deterministic output may recur"}\n';
  await writeFile(path.join(root, "raw-results.json"), raw);
  const expected = { campaign_run_id: "fresh", permanent_pin: "a".repeat(40), tuple: "mlx:1b", workflow_run_id: "12", workflow_run_attempt: 1, sceneworks_revision: "b".repeat(40) };
  const record = { ...expected, status: "succeeded", raw_results_sha256: createHash("sha256").update(raw).digest("hex") };
  await writeFile(path.join(root, "tuple-controller.json"), JSON.stringify(record));
  await verifyTupleExecution(root, expected);
  await assert.rejects(verifyTupleExecution(root, { ...expected, workflow_run_id: "13" }), /another workflow_run_id/);
  await writeFile(path.join(root, "raw-results.json"), raw + "tamper");
  await assert.rejects(verifyTupleExecution(root, expected), /output bytes/);
});
