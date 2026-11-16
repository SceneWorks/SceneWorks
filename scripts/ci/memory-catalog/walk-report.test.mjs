import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "../../..");
const REPORT = path.join(HERE, "walk-report.mjs");

const report = (...args) => execFileSync(process.execPath, [REPORT, ...args], { encoding: "utf8" });

test("a failed walk prints the wrapper diagnostics, the walk log tail and every failing case", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "walk-report-"));
  const workDir = path.join(dir, "work");
  const receipt = path.join(dir, "resource-receipt.json");
  await writeFile(receipt, JSON.stringify({
    resource_completed: false, admission_policy: "strict-idle", lock_wait_seconds: 3,
    child: { returncode: 2, timeout: false, seconds: 14000, owned_tree_empty: true },
  }));
  const log = Array.from({ length: 300 }, (_, index) => `walk line ${index + 1}`);
  log.push("--- minimax_h3_image:q4: ingest failed (adapter); last 40 stderr lines:", "CUDA_ERROR_OUT_OF_MEMORY", "--- end minimax_h3_image:q4 ingest stderr");
  await mkdir(workDir);
  await writeFile(path.join(workDir, "campaign.log"), `${log.join("\n")}\n`);
  await writeFile(path.join(workDir, "summary-2026-11-16T04-00-00-000Z.json"), JSON.stringify({ results: [] }));
  await writeFile(path.join(workDir, "summary-2026-11-16T05-00-00-000Z.json"), JSON.stringify({ results: [
    { key: "minimax_h3_image:q8", status: "committed" },
    { key: "minimax_h3_image:q4", status: "ingest_failed", reason: "adapter exited 101" },
    { key: "minimax_h3_image:bf16", status: "runtime_budget_exceeded", reason: "120 min probe budget" },
  ] }));

  const output = report("--work-dir", workDir, "--receipt", receipt);
  assert.match(output, /resource_completed: false/);
  assert.match(output, /child: returncode=2 timeout=false seconds=14000/);
  assert.match(output, /last 200 of 303 line\(s\):/);
  assert.match(output, /^walk line 104$/m);
  assert.doesNotMatch(output, /^walk line 103$/m, "only the tail is printed");
  assert.match(output, /CUDA_ERROR_OUT_OF_MEMORY\n--- end minimax_h3_image:q4 ingest stderr/);
  assert.match(output, /summary-2026-11-16T05-00-00-000Z\.json: 3 result\(s\), 2 failing/);
  assert.match(output, /^- minimax_h3_image:q4: ingest_failed \(adapter exited 101\)$/m);
  assert.match(output, /^- minimax_h3_image:bf16: runtime_budget_exceeded \(120 min probe budget\)$/m);
  assert.doesNotMatch(output, /^- minimax_h3_image:q8/m, "a committed anchor is not a failure");
  assert.deepEqual(JSON.parse(await readFile(path.join(workDir, "cuda-resource-receipt.json"), "utf8")).child.returncode, 2,
    "the receipt is copied into the uploaded work dir");
});

test("a truncated summary keeps the walk log tail and says why the cases are missing", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "walk-report-"));
  const workDir = path.join(dir, "work");
  await mkdir(workDir);
  await writeFile(path.join(workDir, "campaign.log"), "walk line 1\nENOSPC: no space left on device\n");
  await writeFile(path.join(workDir, "summary-2026-11-16T05-00-00-000Z.json"), '{"results": [{"key": "minimax_h3_image:q4", "sta');
  const output = report("--work-dir", workDir);
  assert.match(output, /ENOSPC: no space left on device/);
  assert.match(output, /summary-2026-11-16T05-00-00-000Z\.json is not valid JSON/);
});

test("a wrapper that never launched the walk reports its own error", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "walk-report-"));
  const workDir = path.join(dir, "work");
  const receipt = path.join(dir, "resource-receipt.json");
  await writeFile(receipt, JSON.stringify({
    resource_completed: false, error: "TimeoutError: exact GPU UUID lock/idle wait exhausted the existing execution budget",
  }));
  const output = report("--work-dir", workDir, "--receipt", receipt, "--tail", "5");
  assert.match(output, /error: TimeoutError: exact GPU UUID lock\/idle wait exhausted/);
  assert.match(output, /child: never launched/);
  assert.match(output, /no campaign\.log in .*: run\.sh never started the walk/);
  assert.match(output, /no summary JSON/);
  assert.match(report("--work-dir", workDir, "--receipt", path.join(dir, "absent.json")), /no receipt at /);
});

test("the candle walk step prints the walk report before it fails", async () => {
  const workflow = await readFile(path.join(ROOT, ".github/workflows/memory-catalog-campaign.yml"), "utf8");
  const candle = workflow.slice(workflow.indexOf("\n  candle:"));
  const start = candle.indexOf("      - name: Walk the catalog\n");
  const step = candle.slice(start, candle.indexOf("\n      - name:", start + 1));
  const wrapper = step.indexOf("windows_cuda_execution.py --actor memory-catalog-candle");
  const reportCall = step.indexOf("node scripts/ci/memory-catalog/walk-report.mjs");
  const thrown = step.indexOf("throw ");
  assert.ok(wrapper >= 0 && reportCall > wrapper && thrown > reportCall, "wrapper, then report, then throw");
  assert.match(step, /--work-dir "\$env:WORK_DIR_NATIVE"/, "reads and writes the uploaded work dir");
  // windows_cuda_execution.py writes its receipt to RUNNER_TEMP/sceneworks-cuda-execution-<run>-<attempt>-<actor>.
  assert.match(step, /--receipt "\$env:RUNNER_TEMP\\sceneworks-cuda-execution-\$env:GITHUB_RUN_ID-\$env:GITHUB_RUN_ATTEMPT-memory-catalog-candle\\resource-receipt\.json"/);
  assert.doesNotMatch(step, /Get-ChildItem Env:|\bprintenv\b|\benv\b *$/m, "never echoes the environment");
});
