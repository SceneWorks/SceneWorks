import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, readFile, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "node:net";
import { assertActionsIdentity, assertActionsTimeBudget, emitOwnedApiToken, probeReadAuthority, requirePreviousJobClosed, importOwnedCorpus, redactReceipt } from "./h3-veda-actions.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const H = "a".repeat(40);
function m(root, backend = "mlx") {
  return { sourceFreeze: { sceneworks: { root: ROOT, head: H }, inference: { head: H }, inferencePin: H }, runners: { [backend]: { apiUrl: "http://127.0.0.1:13579", guardian: { port: 13579, dataDir: path.join(root, "data"), configDir: path.join(root, "config") } } } };
}
const options = (root, backend = "mlx") => ({ backend, runnerName: backend === "mlx" ? "nax-macos-2" : "cuda-windows-2", expectedRunner: backend === "mlx" ? "nax-macos-2" : "cuda-windows-2", sceneWorksSha: H, inferenceSha: H, manifest: m(root, backend), runDir: root, cacheRoot: path.join(root, "cache") });

test("Actions host/source/run roots refuse unknown or mismatched physical identity", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "s9-actions-")); t.after(() => rm(root, { recursive: true, force: true }));
  assertActionsIdentity(options(root)); assertActionsIdentity(options(root, "candle_cuda"));
  for (const change of [
    (o) => o.runnerName = "nax-macos",
    (o) => o.expectedRunner = "nax-macos",
    (o) => o.inferenceSha = "b".repeat(40),
    (o) => o.manifest.sourceFreeze.sceneworks.root = "/untrusted/source",
    (o) => o.manifest.runners.mlx.guardian.dataDir = "/shared/data",
    (o) => o.manifest.runners.mlx.apiUrl = "http://0.0.0.0:13579",
  ]) { const o = structuredClone(options(root)); change(o); assert.throws(() => assertActionsIdentity(o)); }
  const win = options(root, "candle_cuda"); win.expectedRunner = "cuda-windows-3"; win.runnerName = "cuda-windows-3"; assert.throws(() => assertActionsIdentity(win));
});

test("owned import refuses missing private token before creating data or starting API", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "s9-actions-")); t.after(() => rm(root, { recursive: true, force: true }));
  let starts = 0;
  await assert.rejects(importOwnedCorpus({ manifest: m(root), plan: { assets: {} }, planPath: "plan", corpusRoot: root, runDir: root, cacheRoot: root, backend: "mlx", token: "", spawnApi: () => { starts++; } }), /token absent/);
  assert.equal(starts, 0);
  await assert.rejects(readFile(path.join(root, "data", "s9-workspace-owner.json")), /ENOENT/);
});

test("job-owned API token is random-length hex, masked before job env write, and never returned", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "s9-token-")); t.after(() => rm(root, { recursive: true, force: true }));
  const prior = process.env.GITHUB_ACTIONS; process.env.GITHUB_ACTIONS = "true";
  try {
    const output = [], envFile = path.join(root, "job-env");
    const result = await emitOwnedApiToken({ envFile, stdout: { write: (value) => { output.push(value); } }, random: (n) => { assert.equal(n, 32); return Buffer.alloc(n, 0xab); } });
    assert.equal(result, undefined);
    assert.deepEqual(output, [`::add-mask::${"ab".repeat(32)}\n`]);
    assert.equal(await readFile(envFile, "utf8"), `SCENEWORKS_TOKEN=${"ab".repeat(32)}\n`);
    await assert.rejects(emitOwnedApiToken({ envFile: "relative", random: () => Buffer.alloc(32) }), /owned Actions environment file/);
  } finally { if (prior === undefined) delete process.env.GITHUB_ACTIONS; else process.env.GITHUB_ACTIONS = prior; }
});

test("existing credential probe requires exact live job, physical runner and cross-repository queue readbacks", () => {
  const before = Object.fromEntries(["GITHUB_ACTIONS", "GH_TOKEN", "GITHUB_REPOSITORY", "GITHUB_EVENT_NAME", "GITHUB_SHA", "GITHUB_WORKFLOW_REF", "GITHUB_JOB", "RUNNER_NAME"].map((key) => [key, process.env[key]]));
  Object.assign(process.env, { GITHUB_ACTIONS: "true", GH_TOKEN: "not-logged-test-token", GITHUB_REPOSITORY: "SceneWorks/SceneWorks", GITHUB_EVENT_NAME: "workflow_dispatch", GITHUB_SHA: H, GITHUB_WORKFLOW_REF: `SceneWorks/SceneWorks/.github/workflows/h3-veda-terminal.yml@${H}`, GITHUB_JOB: "mac2", RUNNER_NAME: "nax-macos-2" });
  const responses = (endpoint) => {
    if (endpoint.endsWith("/actions/runs/123")) return { id: 123, run_attempt: 1, head_sha: H, status: "in_progress", conclusion: null, event: "workflow_dispatch", name: "H3 Veda terminal owned campaign" };
    if (endpoint.includes("/actions/runs/123/jobs")) return [{ jobs: [{ id: 456, run_id: 123, run_attempt: 1, head_sha: H, name: "mac2", runner_name: "nax-macos-2", runner_id: 5281, status: "in_progress", conclusion: null }] }];
    if (endpoint.startsWith("orgs/")) return [{ runners: [{ id: 5281, name: "nax-macos-2", status: "online", busy: true }] }];
    if (endpoint.startsWith("repos/michaeltrefry/LensWorks/")) return [{ workflow_runs: [] }];
    throw Error("unexpected endpoint");
  };
  const seen = [], exec = (_bin, args) => { const endpoint = args.at(-1); seen.push(endpoint); return JSON.stringify(responses(endpoint)); };
  try {
    assert.equal(probeReadAuthority({ backend: "mlx", runnerName: "nax-macos-2", sceneWorksSha: H, runId: "123", attempt: "1", exec }).runnerId, 5281);
    assert.equal(seen.length, 5);
    assert.throws(() => probeReadAuthority({ backend: "mlx", runnerName: "nax-macos", sceneWorksSha: H, runId: "123", attempt: "1", exec }), /workflow\/source\/job\/runner differs/);
    const failed = (_bin, args) => { if (args.at(-1).startsWith("orgs/")) throw Error("credential value must not surface"); return JSON.stringify(responses(args.at(-1))); };
    assert.throws(() => probeReadAuthority({ backend: "mlx", runnerName: "nax-macos-2", sceneWorksSha: H, runId: "123", attempt: "1", exec: failed }), (error) => error.message.includes("cannot read orgs/SceneWorks/actions/runners") && !error.message.includes("credential value"));
    const wrong = (_bin, args) => { const result = responses(args.at(-1)); if (args.at(-1).includes("/jobs")) result[0].jobs[0].runner_id = 999; return JSON.stringify(result); };
    assert.throws(() => probeReadAuthority({ backend: "mlx", runnerName: "nax-macos-2", sceneWorksSha: H, runId: "123", attempt: "1", exec: wrong }), /runner readback missing/);
  } finally { for (const [key, value] of Object.entries(before)) { if (value === undefined) delete process.env[key]; else process.env[key] = value; } }
});

test("new job token refuses retained guardian before fresh physical admission", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "s9-token-")); t.after(() => rm(root, { recursive: true, force: true }));
  await requirePreviousJobClosed(root);
  await mkdir(path.join(root, "physical-guardian"));
  await writeFile(path.join(root, "physical-guardian/control.json"), "{}");
  await assert.rejects(requirePreviousJobClosed(root), /previous job API token/);
  await writeFile(path.join(root, "physical-guardian/closed.json"), "{}");
  await requirePreviousJobClosed(root);
});

test("owned import refuses a foreign listener before creating data or spawning API", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "s9-actions-")); t.after(() => rm(root, { recursive: true, force: true }));
  const server = createServer(); await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve)); t.after(() => server.close());
  const manifest = m(root); manifest.runners.mlx.guardian.port = server.address().port;
  let starts = 0;
  await assert.rejects(importOwnedCorpus({ manifest, plan: { assets: {} }, planPath: "plan", corpusRoot: root, runDir: root, cacheRoot: root, backend: "mlx", token: "owned-token", spawnApi: () => { starts++; } }), /port already claimed/);
  assert.equal(starts, 0);
  await assert.rejects(readFile(path.join(root, "data", "s9-workspace-owner.json")), /ENOENT/);
});

test("live Actions queue and owned-child deadlines are bounded by exact run/job identity", () => {
  const start = Date.parse("2026-10-13T20:00:00Z"), run = { id: 123, run_attempt: 1, head_sha: H, status: "in_progress", event: "workflow_dispatch", created_at: new Date(start).toISOString() };
  const job = { run_id: 123, run_attempt: 1, head_sha: H, name: "mac2", runner_name: "nax-macos-2", status: "in_progress", started_at: new Date(start + 29 * 60_000).toISOString() };
  const options = { now: start + 38 * 60_000, runId: "123", attempt: "1", backend: "mlx", runnerName: "nax-macos-2", sceneWorksSha: H };
  assertActionsTimeBudget(run, job, options);
  assert.throws(() => assertActionsTimeBudget(run, { ...job, started_at: new Date(start + 31 * 60_000).toISOString() }, options), /30 minutes/);
  assert.throws(() => assertActionsTimeBudget(run, job, { ...options, now: start + 40 * 60_000 }), /insufficient owned-child/);
  assert.throws(() => assertActionsTimeBudget(run, { ...job, runner_name: "nax-macos" }, options), /job admission identity/);
  assert.throws(() => assertActionsTimeBudget({ ...run, status: "completed" }, job, options), /workflow admission identity/);
});

test("artifact transport is allowlisted summary and hashed closure, never raw media, token, or request", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "s9-actions-")); t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(path.join(root, "physical-guardian"));
  await writeFile(path.join(root, "campaign.json"), JSON.stringify({ campaignId: "owned", fingerprint: "f", namedCells: 5400, qualityAcceptance: "pending Michael", completedAcceptance: false, secret: "PRIVATE_TOKEN", cells: { a: { status: "unresolved", request: "PRIVATE_PROMPT" } } }));
  await writeFile(path.join(root, "physical-guardian/closed.json"), JSON.stringify({ closed: true, servicesDrained: true, physicalLockReleased: true, ownerTokenSha256: "h", fingerprint: "f", pid: 123, token: "PRIVATE_TOKEN" }));
  const output = path.join(root, "upload", "receipt.json"); await redactReceipt({ runDir: root, output });
  const text = await readFile(output, "utf8"), receipt = JSON.parse(text);
  assert.equal(receipt.summary.cellStatusCounts.unresolved, 1);
  assert.equal(receipt.guardianClosure.physicalLockReleased, true);
  assert.equal(receipt.mediaUploaded, false);
  assert.doesNotMatch(text, /PRIVATE_TOKEN|PRIVATE_PROMPT|request/);
});

test("workflow binds explicit runner, source, owned import and guardian, with no blanket artifact upload", async () => {
  const workflow = await readFile(path.join(ROOT, ".github/workflows/h3-veda-terminal.yml"), "utf8");
  assert.match(workflow, /runs-on: \[self-hosted, macOS, ARM64, rw-mage\]/);
  assert.match(workflow, /runs-on: \[self-hosted, Windows, X64, cuda, real-weights\]/);
  assert.match(workflow, /test "\$RUNNER_NAME" = 'nax-macos-2'/);
  assert.match(workflow, /S9_EXPECTED_RUNNER: \$\{\{ inputs.expected_runner \}\}/);
  assert.match(workflow, /h3-veda-actions\.mjs prepare/);
  assert.match(workflow, /vcvars64\.bat/);
  assert.match(workflow, /set NVCC_CCBIN=%VCToolsInstallDir%bin\\Hostx64\\x64/);
  assert.equal((workflow.match(/vcvars64\.bat/g) ?? []).length, 3, "Windows build, owned import, and GPU guardian must inherit VS/NVCC setup");
  assert.match(workflow, /h3-veda-acceptance\.mjs run/);
  assert.match(workflow, /h3-veda-actions\.mjs redact/);
  assert.equal((workflow.match(/h3-veda-actions\.mjs probe-auth/g) ?? []).length, 2);
  assert.equal((workflow.match(/h3-veda-actions\.mjs owned-token/g) ?? []).length, 2);
  assert.doesNotMatch(workflow, /SCENEWORKS_S9_TOKEN|S9_RUNNER_READ_TOKEN/);
  assert.doesNotMatch(workflow, /path:.*\*\*|path:.*\$\{\{ inputs.run_dir \}\}/);
});
