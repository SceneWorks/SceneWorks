import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, writeFile, rm, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { bindRecoveryLineage, checkedRecoveryFile, prepareRecovery, safeRecoveryPath, stable, validateExecutionPredecessor, verifyExecutionPredecessor, verifyRecovery } from "./starvector-terminal-recovery.mjs";

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
async function fixture(t) {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-recovery-test-")); t.after(() => rm(root, { recursive: true, force: true }));
  const archive = path.join(root, "77.zip");
  execFileSync("python3", ["-c", "import sys,zipfile\nwith zipfile.ZipFile(sys.argv[1],'w') as z:\n z.writestr('hostile-inputs/0.svg','noise-0<svg/>'); z.writestr('service.stderr.log',''); z.writestr('transcript.json','historical transcript bytes')", archive]);
  const bytes = await readFile(archive), content = '{"campaign_run_id":"retired","permanent_pin":"' + "a".repeat(40) + '"}\n';
  const marker = { path: "campaign.json", size: Buffer.byteLength(content), sha256: sha(content), content };
  const config = { schema_version: 1, campaign_id: "retired", inference_revision: "a".repeat(40), sceneworks_revision: "b".repeat(40), workflow: { repository: "SceneWorks/SceneWorks", path: ".github/workflows/server-candle-linux.yml", run_id: "100", run_attempt: 1, head_sha: "b".repeat(40), conclusion: "cancelled" }, failure: { code: "worker_cpu_fallback", phase: "execution", tuple: "mlx:1b" }, markers: { campaign: marker, tuple: { ...marker, path: "tuple.json" } }, source_artifacts: [{ role: "raw", repository: "SceneWorks/SceneWorks", workflow_run_id: "100", workflow_run_attempt: 1, head_sha: "b".repeat(40), api_workflow_run: { id: "100", head_sha: "b".repeat(40) }, id: "77", name: "raw-retired", size: bytes.length, digest: `sha256:${sha(bytes)}` }], authority: { reason: "Corrected worker identity; historical bytes cannot serve current execution" } };
  const output = path.join(root, "quarantined");
  await prepareRecovery(config, output, { archiveRoot: root });
  return { root, output, config };
}
test("production recovery preserves complete archive contents including empty stderr", async (t) => {
  const { config, output } = await fixture(t);
  const value = await verifyRecovery(config, output, { campaignRunId: "fresh", permanentPin: "c".repeat(40) });
  assert.equal(value.source_artifacts[0].content_inventory.length, 3);
  assert.equal(value.source_artifacts[0].content_inventory.find((entry) => entry.path === "service.stderr.log").byte_size, 0);
  await assert.rejects(() => verifyRecovery(config, output, { campaignRunId: "retired" }), /retired campaign/);
  await verifyRecovery(config, output, { campaignRunId: "fresh", permanentPin: "a".repeat(40) });
});
test("recovery rejects self-consistent extracted-file tampering against original archive", async (t) => {
  const { config, output } = await fixture(t);
  const predecessorPath = path.join(output, "recovery-predecessor.json"), value = JSON.parse(await readFile(predecessorPath));
  const entry = value.source_artifacts[0].content_inventory.find((entry) => entry.path === "transcript.json"), tamper = "tampered historical transcript";
  entry.byte_size = Buffer.byteLength(tamper); entry.sha256 = sha(tamper);
  const file = path.join(output, value.quarantine.root, "source-artifacts/raw/77/extracted/transcript.json");
  await writeFile(file, tamper); await writeFile(predecessorPath, JSON.stringify(value));
  await assert.rejects(() => verifyRecovery(config, output), /immutable archive/);
});
test("new lineage binds successor identity while permanent marker bytes remain unchanged", async (t) => {
  const { config, output, root } = await fixture(t), canonical = path.join(root, "canonical");
  const receipt = { campaign_run_id: "fresh", inference_revision: "c".repeat(40), sceneworks_revision: "d".repeat(40), execution: { repository: "SceneWorks/SceneWorks", workflow_run_id: "101", workflow_run_attempt: 1, head_sha: "d".repeat(40) }, producer: {} };
  await bindRecoveryLineage(receipt, config, output, canonical);
  assert.equal(receipt.schema_version, 2); assert.equal(receipt.campaign_lineage.kind, "failed_campaign_supersession");
  assert.equal(receipt.campaign_lineage.failed_predecessors[0].superseded_by, "fresh");
  assert.equal(receipt.producer.campaign_lineage_sha256, sha(stable(receipt.campaign_lineage)));
  const { authority } = receipt.campaign_lineage.supersession_records[0];
  assert.equal(JSON.parse(await checkedRecoveryFile(canonical, authority.path, authority)).successor_inference_revision, receipt.inference_revision);
  await verifyRecovery(config, output);
});
test("unsafe paths, symlink parents and missing permanent markers fail closed", async (t) => {
  const { config, output, root } = await fixture(t);
  for (const name of ["../out", "/out", "a/../out", "a\\out", "a//out"]) assert.throws(() => safeRecoveryPath(name), /unsafe/);
  await mkdir(path.join(root, "leases"));
  await assert.rejects(() => verifyRecovery(config, output, { leaseRoot: path.join(root, "leases") }), /ENOENT/);
  await symlink(output, path.join(root, "linked"));
  await assert.rejects(() => checkedRecoveryFile(root, "linked/recovery-predecessor.json", { size: 1, sha256: "a".repeat(64) }), /symlink/);
});

test("authenticated upstream-only failure advances execution without rewriting native quarantine", async (t) => {
  const { config, output, root } = await fixture(t), original = await readFile(path.join(output, "recovery-predecessor.json"), "utf8");
  const bytes = await readFile(path.join(root, "77.zip")); await writeFile(path.join(root, "88.zip"), bytes);
  config.execution_predecessor = { stage: "upstream-reference", campaign_id: "failed-upstream", predecessor_campaign_id: "retired", inference_revision: "c".repeat(40), sceneworks_revision: "d".repeat(40), workflow: { repository: "SceneWorks/SceneWorks", path: ".github/workflows/server-candle-linux.yml", run_id: "101", run_attempt: 1, head_sha: "d".repeat(40), conclusion: "failure" }, source_artifact: { id: "88", name: "starvector-upstream-failed-upstream", size: bytes.length, digest: `sha256:${sha(bytes)}` } };
  const value = config.execution_predecessor;
  const run = { id: 101, run_attempt: 1, head_sha: value.sceneworks_revision, path: value.workflow.path, event: "workflow_dispatch", status: "completed", conclusion: "failure" };
  const artifact = { id: 88, name: value.source_artifact.name, size_in_bytes: bytes.length, digest: value.source_artifact.digest, expired: false, workflow_run: { id: 101, head_sha: run.head_sha } };
  const jobs = { total_count: 5, jobs: ["upstream-reference", "mlx-1b", "mlx-8b", "cuda-1b", "cuda-8b"].map(stage => ({ name: `starvector-campaign / ${stage}`, head_sha: run.head_sha, conclusion: stage === "upstream-reference" ? "failure" : "skipped" })) };
  const urls = [];
  const fetchImpl = async (url, options) => { urls.push(url); assert.equal(options.headers.Authorization, "Bearer fixture-token"); return { ok: true, json: async () => url.includes("/artifacts/") ? artifact : url.includes("/jobs?") ? jobs : run }; };
  await prepareRecovery(config, output, { archiveRoot: root, token: "fixture-token", fetchImpl });
  assert(urls.some(url => url.endsWith("/actions/runs/101/attempts/1")));
  assert.equal(await readFile(path.join(output, "recovery-predecessor.json"), "utf8"), original);
  const native = await verifyRecovery(config, output, { campaignRunId: "next", permanentPin: value.inference_revision });
  assert.equal(native.campaign_id, "retired"); assert.equal(native.execution_predecessor, undefined);
  assert.deepEqual(await verifyExecutionPredecessor(config, output, native), value);
  assert.throws(() => validateExecutionPredecessor(config, { ...run, conclusion: "success" }, artifact, jobs), /workflow differs/);
  assert.throws(() => validateExecutionPredecessor(config, { ...run, run_attempt: 2 }, artifact, jobs), /workflow differs/);
  assert.throws(() => validateExecutionPredecessor(config, run, { ...artifact, workflow_run: { id: 99, head_sha: run.head_sha } }, jobs), /artifact differs/);
  assert.throws(() => validateExecutionPredecessor(config, run, artifact, { ...jobs, total_count: 6 }), /census is incomplete/);
  const executed = structuredClone(jobs); executed.jobs[1].conclusion = "failure";
  assert.throws(() => validateExecutionPredecessor(config, run, artifact, executed), /mlx-1b/);
  await writeFile(path.join(output, "execution-attempts/failed-upstream/upstream.zip"), "substituted archive");
  await assert.rejects(() => verifyExecutionPredecessor(config, output, native), /evidence bytes differ/);
});
