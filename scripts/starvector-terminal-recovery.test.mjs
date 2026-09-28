import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, writeFile, rm, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { bindRecoveryLineage, checkedRecoveryFile, prepareRecovery, safeRecoveryPath, stable, validateCancelledAfterUpstreamArchive, validateExecutionPredecessor, validateNativeExecutionArchives, verifyExecutionPredecessor, verifyRecovery } from "./starvector-terminal-recovery.mjs";

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
// Hosted Windows recovery receives this exact workflow-owned interpreter.  The
// local archive fixture also needs an interpreter because recovery validates
// ZIP contents with Python; use the installed command only when the workflow
// value is absent from a native Windows test process.
if (process.platform === "win32" && !process.env.STARVECTOR_TERMINAL_METRICS_PYTHON) {
  process.env.STARVECTOR_TERMINAL_METRICS_PYTHON = "python";
}
async function writeZip(root, id, files) {
  const source = path.join(root, `zip-${id}`);
  for (const [relative, content] of Object.entries(files)) {
    const file = path.join(source, relative); await mkdir(path.dirname(file), { recursive: true }); await writeFile(file, content);
  }
  const archive = path.join(root, `${id}.zip`);
  execFileSync(process.env.STARVECTOR_TERMINAL_METRICS_PYTHON ?? "python3", ["-c", "import os,sys,zipfile\nroot,out=sys.argv[1:]\nwith zipfile.ZipFile(out,'w') as z:\n for base,_,names in os.walk(root):\n  for name in names:\n   p=os.path.join(base,name); z.write(p,os.path.relpath(p,root))", source, archive]);
  return readFile(archive);
}
async function fixture(t) {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-recovery-test-")); t.after(() => rm(root, { recursive: true, force: true }));
  const archive = path.join(root, "77.zip");
  execFileSync(process.env.STARVECTOR_TERMINAL_METRICS_PYTHON ?? "python3", ["-c", "import sys,zipfile\nwith zipfile.ZipFile(sys.argv[1],'w') as z:\n z.writestr('hostile-inputs/0.svg','noise-0<svg/>'); z.writestr('service.stderr.log',''); z.writestr('transcript.json','historical transcript bytes')", archive]);
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
  // A second upstream failure retains the first authentic archive and links
  // through it, while the old native quarantine remains byte-identical.
  const oldMetadata = await readFile(path.join(output, "execution-attempts/failed-upstream/metadata.json"));
  const next = structuredClone(value);
  next.campaign_id = "second-failed-upstream"; next.predecessor_campaign_id = value.campaign_id;
  next.workflow.run_id = "102"; next.source_artifact.id = "89"; next.source_artifact.name = `starvector-upstream-${next.campaign_id}`;
  config.execution_history = [value]; config.execution_predecessor = next;
  await writeFile(path.join(root, "89.zip"), bytes);
  const secondRun = { ...run, id: 102 }, secondArtifact = { ...artifact, id: 89, name: next.source_artifact.name, workflow_run: { ...artifact.workflow_run, id: 102 } };
  const secondFetch = async (url) => ({ ok: true, json: async () => url.includes("/artifacts/89") ? secondArtifact : url.includes("/artifacts/") ? artifact : url.includes("/jobs?") ? jobs : url.includes("/runs/102/") ? secondRun : run });
  await prepareRecovery(config, output, { archiveRoot: root, token: "fixture-token", fetchImpl: secondFetch });
  assert.deepEqual(await verifyExecutionPredecessor(config, output, native), next);
  assert.deepEqual(await readFile(path.join(output, "execution-attempts/failed-upstream/metadata.json")), oldMetadata);
  assert.equal(await readFile(path.join(output, "recovery-predecessor.json"), "utf8"), original);
  await verifyRecovery(config, output);
  await assert.rejects(() => verifyExecutionPredecessor({ ...config, execution_history: [] }, output, native), /ordered successor chain/);
  await assert.rejects(() => verifyExecutionPredecessor({ ...config, execution_history: [value, value] }, output, native), /ordered successor chain/);
  await writeFile(path.join(output, "execution-attempts/failed-upstream/upstream.zip"), "substituted archive");
  await assert.rejects(() => verifyExecutionPredecessor(config, output, native), /evidence bytes differ/);
});

test("cancelled campaign after authenticated upstream completion advances exactly one claim", async (t) => {
  const { config, root } = await fixture(t);
  const campaign = "cancelled-after-upstream", revision = "c".repeat(40), sceneWorksRevision = "d".repeat(40);
  const models = {
    "1b": { repository: "starvector/starvector-1b-im2svg", revision: "380ab95d25a8e9ab1dc825debe238b4953ae13b9", inventory: "1".repeat(64) },
    "8b": { repository: "starvector/starvector-8b-im2svg", revision: "518beea8dcb5f7a37c5911e92d1d62a76beee7f9", inventory: "8".repeat(64) },
  };
  const manifests = {}, archiveFiles = {};
  for (const [tier, model] of Object.entries(models)) {
    for (const role of ["config", "processor", "transcript"]) archiveFiles[`upstream-${tier}/${role}.${role === "transcript" ? "jsonl" : "json"}`] = `${tier}-${role}`;
    const cases = Array.from({ length: 20 }, (_, case_index) => {
      const common = { case_index, source_case_index: [0, 1, 2, 3, 4, 30, 31, 32, 33, 34, 60, 61, 62, 63, 64, 90, 91, 92, 93, 94][case_index], seed: case_index, input_png_sha256: String(case_index).padStart(64, "0") };
      if (case_index === 0) {
        const raw = `upstream-${tier}/case-00/raw.svg`; archiveFiles[raw] = "<svg";
        return { ...common, outcome: "rejected", rejection_stage: "generation_limit", rejection_code: "wall_time_limit", rejection_reason: "upstream StarVector stopped at the wall_time_limit", upstream_raw_svg: raw, upstream_raw_svg_sha256: sha(archiveFiles[raw]) };
      }
      const svg = `upstream-${tier}/case-${String(case_index).padStart(2, "0")}/rendered/canonical.svg`, png = `upstream-${tier}/case-${String(case_index).padStart(2, "0")}/rendered/preview.png`;
      archiveFiles[svg] = `<svg id="${case_index}"/>`; archiveFiles[png] = `png-${case_index}`;
      return { ...common, outcome: "accepted", upstream_svg: svg, upstream_svg_sha256: sha(archiveFiles[svg]), upstream_preview_png: png, upstream_preview_png_sha256: sha(archiveFiles[png]) };
    });
    const reference = { implementation_repository: "https://github.com/joanrod/star-vector", implementation_revision: "0e083c1911760aa31bc576ca7f337a7f8ee605ec", checkpoint_repository: model.repository, checkpoint_revision: model.revision, checkpoint_inventory_sha256: model.inventory, config_sha256: sha(archiveFiles[`upstream-${tier}/config.json`]), processor_sha256: sha(archiveFiles[`upstream-${tier}/processor.json`]), transcript_sha256: sha(archiveFiles[`upstream-${tier}/transcript.jsonl`]) };
    const name = `upstream-reference-${tier}.json`;
    manifests[name] = JSON.stringify({ schema_version: 2, upstream_reference: reference, config_path: `upstream-${tier}/config.json`, processor_path: `upstream-${tier}/processor.json`, transcript_path: `upstream-${tier}/transcript.jsonl`, cases });
    archiveFiles[name] = manifests[name];
  }
  const inventory = Object.entries(archiveFiles).map(([path, bytes]) => ({ path, byte_size: Buffer.byteLength(bytes), sha256: sha(bytes) })).sort((left, right) => left.path.localeCompare(right.path));
  const controller = {
    schema_version: 1,
    campaign_run_id: campaign,
    inference_revision: revision,
    sceneworks_revision: sceneWorksRevision,
    workflow_run_id: "101",
    workflow_run_attempt: 1,
    validated: Object.entries(models).map(([tier, model]) => ({ status: "validated", tier, model_inventory_sha256: model.inventory, source_sha256: "a".repeat(64), cases: 20 })),
    artifacts: { entries: inventory, aggregate_sha256: sha(JSON.stringify(inventory)) },
  };
  const upstreamBytes = await writeZip(root, "88", { ...archiveFiles, "upstream-controller.json": JSON.stringify(controller) });
  const recoveryBytes = await writeZip(root, "87", { "recovery-predecessor.json": "authenticated predecessor bytes" });
  const value = {
    stage: "upstream-reference",
    predecessor_campaign_id: config.campaign_id,
    campaign_id: campaign,
    inference_revision: revision,
    sceneworks_revision: sceneWorksRevision,
    workflow: { repository: "SceneWorks/SceneWorks", path: ".github/workflows/server-candle-linux.yml", run_id: "101", run_attempt: 1, head_sha: sceneWorksRevision, conclusion: "cancelled" },
    failure: { code: "campaign_cancelled_after_upstream", phase: "execution", evidence_schema_version: 1 },
    source_artifacts: [
      { role: "recovery", id: "87", name: `starvector-recovery-${campaign}`, size: recoveryBytes.length, digest: `sha256:${sha(recoveryBytes)}` },
      { role: "upstream", id: "88", name: `starvector-upstream-${campaign}`, size: upstreamBytes.length, digest: `sha256:${sha(upstreamBytes)}` },
    ],
  };
  config.execution_predecessor = value;
  const run = { id: 101, run_attempt: 1, head_sha: sceneWorksRevision, path: value.workflow.path, event: "workflow_dispatch", status: "completed", conclusion: "cancelled" };
  const artifacts = value.source_artifacts.map((input) => ({ id: Number(input.id), name: input.name, size_in_bytes: input.size, digest: input.digest, expired: false, workflow_run: { id: 101, head_sha: sceneWorksRevision } }));
  const artifactCensus = { total_count: artifacts.length, artifacts };
  const conclusions = new Map([
    ["starvector-campaign / prepare-recovery", "success"],
    ["starvector-provision", "skipped"],
    ["starvector-diagnostic-candle-1b", "skipped"],
    ["starvector-readiness", "skipped"],
    ["starvector-source-closure", "skipped"],
    ["build-candle", "skipped"],
    ["starvector-campaign / upstream-reference", "success"],
    ["starvector-campaign / mlx-1b", "cancelled"],
    ["starvector-campaign / mlx-8b", "cancelled"],
    ["starvector-campaign / cuda-1b", "cancelled"],
    ["starvector-campaign / cuda-8b", "cancelled"],
    ["starvector-campaign / seal-receipt", "failure"],
  ]);
  const jobs = { total_count: conclusions.size, jobs: [...conclusions].map(([name, conclusion]) => ({ name, conclusion, head_sha: sceneWorksRevision })) };
  const fetchImpl = async (url) => ({ ok: true, json: async () => url.includes("/artifacts?") ? artifactCensus : url.includes("/jobs?") ? jobs : run });
  const output = path.join(root, "cancelled-after-upstream-valid");
  await prepareRecovery(config, output, { archiveRoot: root, token: "fixture-token", fetchImpl });
  const native = await verifyRecovery(config, output, { campaignRunId: "next", permanentPin: "e".repeat(40) });
  assert.deepEqual(await verifyExecutionPredecessor(config, output, native), value);

  const missingTuple = structuredClone(jobs); missingTuple.jobs = missingTuple.jobs.filter((job) => !job.name.endsWith("cuda-8b")); missingTuple.total_count--;
  assert.throws(() => validateExecutionPredecessor(config, run, artifactCensus, missingTuple), /job census differs/);
  const extraTuple = structuredClone(jobs); extraTuple.jobs.push({ name: "starvector-campaign / cuda-extra", conclusion: "cancelled", head_sha: sceneWorksRevision }); extraTuple.total_count++;
  assert.throws(() => validateExecutionPredecessor(config, run, artifactCensus, extraTuple), /job census differs/);
  const wrongTuple = structuredClone(jobs); wrongTuple.jobs.find((job) => job.name.endsWith("mlx-1b")).conclusion = "success";
  assert.throws(() => validateExecutionPredecessor(config, run, artifactCensus, wrongTuple), /job differs: starvector-campaign \/ mlx-1b/);
  assert.throws(() => validateExecutionPredecessor({ ...config, execution_predecessor: { ...value, failure: { ...value.failure, reason: "operator assertion" } } }, run, artifactCensus, jobs), /failure identity/);
  assert.throws(() => validateExecutionPredecessor({ ...config, execution_predecessor: { ...value, workflow: { ...value.workflow, conclusion: "failure" } } }, { ...run, conclusion: "failure" }, artifactCensus, jobs), /requires a cancelled workflow/);
  assert.throws(() => validateExecutionPredecessor(config, { ...run, status: "in_progress" }, artifactCensus, jobs), /workflow differs/);
  const missingArtifact = { total_count: 1, artifacts: artifacts.slice(1) };
  assert.throws(() => validateExecutionPredecessor(config, run, missingArtifact, jobs), /artifact census differs/);
  const extraArtifact = { total_count: 3, artifacts: [...artifacts, { ...artifacts[1], id: 99, name: "unrelated" }] };
  assert.throws(() => validateExecutionPredecessor(config, run, extraArtifact, jobs), /artifact census differs/);
  const unrelated = { total_count: 2, artifacts: [artifacts[0], { ...artifacts[1], id: 99, name: "unrelated" }] };
  assert.throws(() => validateExecutionPredecessor(config, run, unrelated, jobs), /upstream artifact differs/);
  const duplicateRole = structuredClone(value); duplicateRole.source_artifacts[0] = { ...duplicateRole.source_artifacts[1] };
  assert.throws(() => validateExecutionPredecessor({ ...config, execution_predecessor: duplicateRole }, run, artifactCensus, jobs), /recovery artifact differs/);

  let mutation = 0;
  const expectArchiveFailure = async (files, pattern) => {
    const id = `cancelled-mutation-${mutation++}`;
    await writeZip(root, id, files);
    const generated = path.join(root, `${id}.zip`);
    await assert.rejects(() => validateCancelledAfterUpstreamArchive(value, generated), pattern);
  };
  const controllerWith = (change) => ({ ...archiveFiles, "upstream-controller.json": JSON.stringify(change(structuredClone(controller))) });
  await expectArchiveFailure(controllerWith((item) => { item.campaign_run_id = "unrelated-campaign"; return item; }), /controller identity differs/);
  await expectArchiveFailure(controllerWith((item) => { item.validated.pop(); return item; }), /validation or inventory is incomplete|validated tier differs/);
  await expectArchiveFailure(controllerWith((item) => { item.validated[1].tier = "1b"; return item; }), /validated tier differs/);
  await expectArchiveFailure(controllerWith((item) => { item.validated[0].status = "failed"; return item; }), /validated tier differs/);
  await expectArchiveFailure(controllerWith((item) => { item.validated[0].cases = 19; return item; }), /validated tier differs/);
  await expectArchiveFailure(controllerWith((item) => { item.artifacts.aggregate_sha256 = "f".repeat(64); return item; }), /validation or inventory is incomplete/);
  await expectArchiveFailure(controllerWith((item) => { item.artifacts.entries[0].sha256 = "f".repeat(64); item.artifacts.aggregate_sha256 = sha(JSON.stringify(item.artifacts.entries)); return item; }), /archive inventory differs/);
  const missingManifest = { ...archiveFiles, "upstream-controller.json": JSON.stringify(controller) }; delete missingManifest["upstream-reference-8b.json"];
  await expectArchiveFailure(missingManifest, /missing, ambiguous, or oversized/);
  const wrongManifest = structuredClone(JSON.parse(manifests["upstream-reference-1b.json"])); wrongManifest.upstream_reference.checkpoint_revision = "f".repeat(40);
  await expectArchiveFailure({ ...archiveFiles, "upstream-reference-1b.json": JSON.stringify(wrongManifest), "upstream-controller.json": JSON.stringify(controller) }, /archive inventory differs|manifest inventory differs|manifest identity differs/);
});

test("checked-in recovery advances through the authenticated publication failure", async () => {
  const config = JSON.parse(await readFile(path.join(process.cwd(), "release/starvector-terminal-recovery-v1.json")));
  const chain = [...config.execution_history, config.execution_predecessor];
  let previous = config.campaign_id;
  for (const value of chain) {
    assert.equal(value.predecessor_campaign_id, previous);
    previous = value.campaign_id;
  }
  assert.equal(config.execution_history.at(-1).campaign_id, "sc22261-8e2d967-848eb1a-0605dc4993d41332");
  assert.equal(config.execution_predecessor.campaign_id, "sc22261-0b084cb-2326819-9869bf8aa45664a2");
  assert.equal(config.execution_predecessor.failure.code, "native_asset_publication_missing_display_name");
  assert.equal(config.execution_predecessor.failure.record_type, "publication_failure_predecessor");
  assert.ok(!chain.some((value) => value.campaign_id === "sc22261-c5c8c2a-db676be-5438020f56714566"));
});

test("authenticated native failure retains upstream, raw, and combined archives as one successor", async (t) => {
  const { config, output, root } = await fixture(t);
  const campaign = "failed-native";
  const value = {
    stage: "native",
    predecessor_campaign_id: config.campaign_id,
    campaign_id: campaign,
    inference_revision: "c".repeat(40),
    sceneworks_revision: "d".repeat(40),
    workflow: { repository: "SceneWorks/SceneWorks", path: ".github/workflows/server-candle-linux.yml", run_id: "101", run_attempt: 1, head_sha: "d".repeat(40), conclusion: "failure" },
    failure: { code: "native_model_receipt_unproven", phase: "execution", tuple: "mlx:1b", evidence_schema_version: 1, model_id: "starvector_1b", model_repository: "starvector/starvector-1b-im2svg", model_revision: "e".repeat(40) },
  };
  const upstream = { "upstream-controller.json": JSON.stringify({ schema_version: 1, campaign_run_id: campaign, inference_revision: value.inference_revision, sceneworks_revision: value.sceneworks_revision, workflow_run_id: value.workflow.run_id, workflow_run_attempt: value.workflow.run_attempt }) };
  const records = {
    "controller-failure.json": JSON.stringify({ campaign_run_id: campaign, permanent_pin: value.inference_revision, tuple: value.failure.tuple, status: "failed" }),
    "preflight-provenance.json": JSON.stringify({ campaign_run_id: campaign, inference_revision: value.inference_revision, permanent_pin: value.inference_revision, tuple: value.failure.tuple, workflow_run_id: value.workflow.run_id, workflow_run_attempt: value.workflow.run_attempt, service: { sceneworks_revision: value.sceneworks_revision, inference_revision: value.inference_revision, tuple: value.failure.tuple, models: { "starvector-1b": { revision: value.failure.model_revision } }, worker: { model_id: value.failure.model_id, provider_id: "mlx-starvector-1b" } } }),
    "case-bundle.json": JSON.stringify({ schema_version: 1, inference_revision: value.inference_revision, tuples: { [value.failure.tuple]: { image_quality: [{ model: value.failure.model_id }], deterministic_parity: [{ model: value.failure.model_id }], upstream_reference: { checkpoint_revision: value.failure.model_revision } } } }),
    "product-service-worker.stdout.log": [
      JSON.stringify({ event: "model_source_tier_selected", repository: value.failure.model_repository, revision: value.failure.model_revision }),
      JSON.stringify({ event: "utility_job_failed", error: `vector_model_unavailable: exact receipt-backed snapshot ${value.failure.model_repository}@${value.failure.model_revision} is missing or unproven` }),
    ].join("\n"),
  };
  const bytesByRole = {
    upstream: await writeZip(root, "88", upstream),
    raw: await writeZip(root, "89", records),
    combined: await writeZip(root, "90", Object.fromEntries(Object.entries(records).map(([name, content]) => [`evidence/${name}`, content]))),
  };
  const sourceArtifacts = [
    ["upstream", "88", `starvector-upstream-${campaign}`],
    ["raw", "89", `starvector-terminal-mlx-1b-${campaign}`],
    ["combined", "90", `starvector-terminal-receipt-${campaign}`],
  ].map(([role, id, name]) => ({ role, id, name, size: bytesByRole[role].length, digest: `sha256:${sha(bytesByRole[role])}` }));
  value.source_artifacts = sourceArtifacts;
  config.execution_predecessor = value;
  const run = { id: 101, run_attempt: 1, head_sha: value.sceneworks_revision, path: value.workflow.path, event: "workflow_dispatch", status: "completed", conclusion: "failure" };
  const artifacts = sourceArtifacts.map((input) => ({ id: Number(input.id), name: input.name, size_in_bytes: input.size, digest: input.digest, expired: false, workflow_run: { id: 101, head_sha: run.head_sha } }));
  const conclusions = { "upstream-reference": "success", "mlx-1b": "failure", "mlx-8b": "skipped", "cuda-1b": "skipped", "cuda-8b": "skipped", "seal-receipt": "failure" };
  const jobs = { total_count: 6, jobs: Object.entries(conclusions).map(([stage, conclusion]) => ({ name: `starvector-campaign / ${stage}`, head_sha: run.head_sha, conclusion })) };
  const fetchImpl = async (url) => {
    const artifactIndex = sourceArtifacts.findIndex((input) => url.includes(`/artifacts/${input.id}`));
    return { ok: true, json: async () => artifactIndex >= 0 ? artifacts[artifactIndex] : url.includes("/jobs?") ? jobs : run };
  };
  await prepareRecovery(config, output, { archiveRoot: root, token: "fixture-token", fetchImpl });
  const native = await verifyRecovery(config, output, { campaignRunId: "next", permanentPin: value.inference_revision });
  assert.deepEqual(await verifyExecutionPredecessor(config, output, native), value);
  for (const input of sourceArtifacts) await checkedRecoveryFile(output, `execution-attempts/${campaign}/${input.role}.zip`, { size: input.size, sha256: input.digest.slice(7) });

  assert.throws(() => validateExecutionPredecessor({ ...config, execution_predecessor: { ...value, source_artifacts: value.source_artifacts.slice(0, 2) } }, run, artifacts, jobs), /artifact census/);
  const wrongArtifact = artifacts.map((entry) => ({ ...entry })); wrongArtifact[2].digest = `sha256:${"e".repeat(64)}`;
  assert.throws(() => validateExecutionPredecessor(config, run, wrongArtifact, jobs), /combined artifact differs/);
  const wrongJobs = structuredClone(jobs); wrongJobs.jobs.find((job) => job.name.endsWith("mlx-8b")).conclusion = "success";
  assert.throws(() => validateExecutionPredecessor(config, run, artifacts, wrongJobs), /mlx-8b/);
  assert.throws(() => validateExecutionPredecessor({ ...config, execution_predecessor: { ...value, failure: { ...value.failure, code: "generic_error" } } }, run, artifacts, jobs), /failure identity/);

  let mutationIndex = 0;
  const expectSemanticFailure = async (role, files, pattern) => {
    const mutated = structuredClone(value), observed = artifacts.map((entry) => ({ ...entry }));
    const roles = role === "raw+combined" ? ["raw", "combined"] : [role];
    for (const currentRole of roles) {
      const input = mutated.source_artifacts.find((entry) => entry.role === currentRole);
      const archiveFiles = currentRole === "combined" ? Object.fromEntries(Object.entries(files).map(([name, content]) => [`evidence/${name}`, content])) : files;
      const bytes = await writeZip(root, input.id, archiveFiles); input.size = bytes.length; input.digest = `sha256:${sha(bytes)}`;
      const observedArtifact = observed.find((entry) => String(entry.id) === input.id);
      observedArtifact.size_in_bytes = input.size; observedArtifact.digest = input.digest;
    }
    const mutatedFetch = async (url) => {
      const found = mutated.source_artifacts.findIndex((entry) => url.includes(`/artifacts/${entry.id}`));
      return { ok: true, json: async () => found >= 0 ? observed[found] : url.includes("/jobs?") ? jobs : run };
    };
    const mutatedConfig = { ...config, execution_predecessor: mutated };
    try {
      await assert.rejects(() => prepareRecovery(mutatedConfig, path.join(root, `semantic-${mutationIndex++}`), { archiveRoot: root, token: "fixture-token", fetchImpl: mutatedFetch }), pattern);
    } finally {
      for (const currentRole of roles) {
        const input = value.source_artifacts.find((entry) => entry.role === currentRole);
        await writeFile(path.join(root, `${input.id}.zip`), bytesByRole[currentRole]);
      }
    }
  };
  await expectSemanticFailure("upstream", { ...upstream, "upstream-controller.json": JSON.stringify({ ...JSON.parse(upstream["upstream-controller.json"]), inference_revision: "f".repeat(40) }) }, /upstream controller identity/);
  await expectSemanticFailure("raw+combined", { ...records, "preflight-provenance.json": JSON.stringify({ ...JSON.parse(records["preflight-provenance.json"]), tuple: "mlx:8b" }) }, /preflight provenance/);
  await expectSemanticFailure("raw+combined", { ...records, "product-service-worker.stdout.log": JSON.stringify({ event: "utility_job_failed", error: "generic infrastructure error" }) }, /receipt rejection/);
  await expectSemanticFailure("combined", { ...records, "case-bundle.json": JSON.stringify({ schema_version: 1, inference_revision: value.inference_revision, tuples: {} }) }, /combined archive substituted case-bundle/);
});

test("publication failure predecessor proves completed asset writes, displayName rejection, clean stop, and no seal", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-publication-recovery-")); t.after(() => rm(root, { recursive: true, force: true }));
  const value = {
    stage: "native",
    predecessor_campaign_id: "prior",
    campaign_id: "failed-publication",
    inference_revision: "c".repeat(40),
    sceneworks_revision: "d".repeat(40),
    workflow: { repository: "SceneWorks/SceneWorks", path: ".github/workflows/starvector-terminal.yml", run_id: "101", run_attempt: 1, head_sha: "d".repeat(40), conclusion: "cancelled" },
    failure: {
      code: "native_asset_publication_missing_display_name",
      record_type: "publication_failure_predecessor",
      phase: "execution",
      tuple: "mlx:1b",
      evidence_schema_version: 1,
      model_id: "starvector_1b",
      model_repository: "starvector/starvector-1b-im2svg",
      model_revision: "e".repeat(40),
      completed_case_id: "quality-v1-0",
      route_record_count: 3,
      completed_poll_count: 1,
      generation_set_id: "genset_fixture",
      asset_id: "asset_fixture",
      api_status: 400,
      api_detail: "Missing required field: displayName",
      api_job_id: "job_fixture",
      api_error_count: 1,
      recovery_failure_count: 1,
      stopped_instance_token: "f".repeat(64),
      stopped_api_pid: 201,
      stopped_worker_pid: 202,
    },
  };
  const upstream = { "upstream-controller.json": JSON.stringify({ schema_version: 1, campaign_run_id: value.campaign_id, inference_revision: value.inference_revision, sceneworks_revision: value.sceneworks_revision, workflow_run_id: value.workflow.run_id, workflow_run_attempt: 1 }) };
  const routeRows = [
    { case_id: value.failure.completed_case_id, phase: "created", job: { status: "queued" } },
    { case_id: value.failure.completed_case_id, phase: "polled", job: { status: "running" } },
    { case_id: value.failure.completed_case_id, phase: "polled", job: { status: "completed", result: { generationSetId: value.failure.generation_set_id, assetWrites: [{ assetId: value.failure.asset_id, model: value.failure.model_id, type: "vector" }] } } },
  ];
  const apiLog = [
    { event: "api_error", status: 400, detail: value.failure.api_detail },
    { event: "terminal_progress_side_effect_recovery_failed", status: 400, detail: value.failure.api_detail, retry_deferred: true, job_id: value.failure.api_job_id },
  ].map((record) => JSON.stringify(record)).join("\n") + "\n";
  const stopped = { schema_version: 1, status: "stopped", instance_token: value.failure.stopped_instance_token, api_pid: value.failure.stopped_api_pid, worker_pid: value.failure.stopped_worker_pid };
  const records = {
    "controller-failure.json": JSON.stringify({ campaign_run_id: value.campaign_id, permanent_pin: value.inference_revision, tuple: value.failure.tuple, status: "failed" }),
    "preflight-provenance.json": JSON.stringify({ campaign_run_id: value.campaign_id, inference_revision: value.inference_revision, permanent_pin: value.inference_revision, tuple: value.failure.tuple, workflow_run_id: value.workflow.run_id, workflow_run_attempt: 1, service: { sceneworks_revision: value.sceneworks_revision, inference_revision: value.inference_revision, tuple: value.failure.tuple, instance_token: value.failure.stopped_instance_token, api_pid: value.failure.stopped_api_pid, worker_pid: value.failure.stopped_worker_pid, models: { "starvector-1b": { revision: value.failure.model_revision } }, worker: { model_id: value.failure.model_id, provider_id: "mlx-starvector-1b" } } }),
    "case-bundle.json": JSON.stringify({ schema_version: 1, inference_revision: value.inference_revision, tuples: { "mlx:1b": { image_quality: [{ model: value.failure.model_id }], deterministic_parity: [], upstream_reference: { checkpoint_revision: value.failure.model_revision } } } }),
    "product-service-worker.stdout.log": "",
    "product-service-api.stdout.log": apiLog,
    "product-service-stopped.json": JSON.stringify(stopped),
    "vector-generate-route.ndjson": routeRows.map((record) => JSON.stringify(record)).join("\n") + "\n",
  };
  value.failure.controller_failure_sha256 = sha(records["controller-failure.json"]);
  const writeArchives = async (suffix, rawRecords = records, combinedRecords = rawRecords) => {
    await writeZip(root, `upstream-publication-${suffix}`, upstream);
    await writeZip(root, `raw-publication-${suffix}`, rawRecords);
    await writeZip(root, `combined-publication-${suffix}`, Object.fromEntries(Object.entries(combinedRecords).map(([name, content]) => [`evidence/${name}`, content])));
    return { upstream: path.join(root, `upstream-publication-${suffix}.zip`), raw: path.join(root, `raw-publication-${suffix}.zip`), combined: path.join(root, `combined-publication-${suffix}.zip`) };
  };
  const validArchives = await writeArchives("valid");
  await validateNativeExecutionArchives(value, validArchives);

  const sourceArtifacts = [
    { role: "upstream", id: "88", name: `starvector-upstream-${value.campaign_id}`, size: 1, digest: `sha256:${"1".repeat(64)}` },
    { role: "raw", id: "89", name: `starvector-terminal-mlx-1b-${value.campaign_id}`, size: 1, digest: `sha256:${"2".repeat(64)}` },
    { role: "combined", id: "90", name: `starvector-terminal-receipt-${value.campaign_id}`, size: 1, digest: `sha256:${"3".repeat(64)}` },
  ];
  value.source_artifacts = sourceArtifacts;
  const config = { campaign_id: "retired", execution_predecessor: value };
  const run = { id: 101, run_attempt: 1, head_sha: value.sceneworks_revision, path: value.workflow.path, event: "workflow_dispatch", status: "completed", conclusion: "cancelled" };
  const artifacts = sourceArtifacts.map((input) => ({ id: Number(input.id), name: input.name, size_in_bytes: input.size, digest: input.digest, expired: false, workflow_run: { id: 101, head_sha: run.head_sha } }));
  const conclusions = new Map([
    ["starvector-campaign / prepare-recovery", "success"], ["starvector-provision", "skipped"], ["starvector-diagnostic-candle-1b", "skipped"], ["starvector-readiness", "skipped"], ["starvector-source-closure", "skipped"], ["build-candle", "skipped"], ["starvector-campaign / upstream-reference", "success"], ["starvector-campaign / mlx-1b", "cancelled"], ["starvector-campaign / mlx-8b", "cancelled"], ["starvector-campaign / cuda-1b", "cancelled"], ["starvector-campaign / cuda-8b", "cancelled"], ["starvector-campaign / seal-receipt", "failure"],
  ]);
  const jobs = { total_count: conclusions.size, jobs: [...conclusions].map(([name, conclusion]) => ({ name, conclusion, head_sha: run.head_sha })) };
  assert.deepEqual(validateExecutionPredecessor(config, run, artifacts, jobs), value);
  assert.throws(() => validateExecutionPredecessor(config, { ...run, id: 102 }, artifacts, jobs), /workflow differs/);
  assert.throws(() => validateExecutionPredecessor(config, { ...run, head_sha: "a".repeat(40) }, artifacts, jobs), /workflow differs/);
  assert.throws(() => validateExecutionPredecessor(config, { ...run, conclusion: "failure" }, artifacts, jobs), /workflow differs/);
  const acceptanceClaim = { ...config, execution_predecessor: { ...value, failure: { ...value.failure, record_type: "terminal_acceptance" } } };
  assert.throws(() => validateExecutionPredecessor(acceptanceClaim, run, artifacts, jobs), /failure identity/);
  const completedJob = structuredClone(jobs); completedJob.jobs.find((job) => job.name.endsWith("mlx-1b")).conclusion = "success";
  assert.throws(() => validateExecutionPredecessor(config, run, artifacts, completedJob), /job differs/);

  const wrongPin = { ...records, "preflight-provenance.json": records["preflight-provenance.json"].replace(value.inference_revision, "a".repeat(40)) };
  const wrongPinArchives = await writeArchives("wrong-pin", wrongPin);
  await assert.rejects(() => validateNativeExecutionArchives(value, wrongPinArchives), /preflight provenance/);
  const missingDisplayName = { ...records, "product-service-api.stdout.log": apiLog.replaceAll("Missing required field: displayName", "different API failure") };
  const missingDisplayNameArchives = await writeArchives("missing-display-name", missingDisplayName);
  await assert.rejects(() => validateNativeExecutionArchives(value, missingDisplayNameArchives), /displayName failure/);
  const summaryNeutralApiSubstitution = { ...records, "product-service-api.stdout.log": `${apiLog}${JSON.stringify({ event: "unrelated_valid_event" })}\n` };
  const summaryNeutralApiArchives = await writeArchives("summary-neutral-api", records, summaryNeutralApiSubstitution);
  await assert.rejects(() => validateNativeExecutionArchives(value, summaryNeutralApiArchives), /substituted product service API evidence/);
  const noCompleted = { ...records, "vector-generate-route.ndjson": routeRows.slice(0, 2).map((record) => JSON.stringify(record)).join("\n") + "\n" };
  const noCompletedArchives = await writeArchives("no-completed", noCompleted);
  await assert.rejects(() => validateNativeExecutionArchives(value, noCompletedArchives), /completed generation/);
  const noAssetWrites = structuredClone(routeRows); noAssetWrites[2].job.result.assetWrites = [];
  const noWrites = { ...records, "vector-generate-route.ndjson": noAssetWrites.map((record) => JSON.stringify(record)).join("\n") + "\n" };
  const noWritesArchives = await writeArchives("no-writes", noWrites);
  await assert.rejects(() => validateNativeExecutionArchives(value, noWritesArchives), /completed asset writes/);
  const wrongStopped = { ...records, "product-service-stopped.json": JSON.stringify({ ...stopped, instance_token: "a".repeat(64) }) };
  const wrongStoppedArchives = await writeArchives("wrong-stopped", wrongStopped);
  await assert.rejects(() => validateNativeExecutionArchives(value, wrongStoppedArchives), /stop identity/);
  const forgedSeal = { ...records, "terminal-receipt.json": JSON.stringify({ status: "accepted" }) };
  const forgedSealArchives = await writeArchives("forged-seal", forgedSeal);
  await assert.rejects(() => validateNativeExecutionArchives(value, forgedSealArchives), /forged terminal seal/);
  const mismatchedCombined = { ...records, "controller-failure.json": JSON.stringify({ campaign_run_id: value.campaign_id, permanent_pin: value.inference_revision, tuple: value.failure.tuple, status: "failed", error: "substituted" }) };
  const mismatchArchives = await writeArchives("mismatch", records, mismatchedCombined);
  await assert.rejects(() => validateNativeExecutionArchives(value, mismatchArchives), /substituted controller-failure/);
});

test("underprovisioned native campaign is bound to exact budgets and terminal outcomes", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "starvector-budget-recovery-")); t.after(() => rm(root, { recursive: true, force: true }));
  const value = { campaign_id: "failed-budget", inference_revision: "c".repeat(40), sceneworks_revision: "d".repeat(40), workflow: { run_id: "101", run_attempt: 1 }, failure: { code: "native_quality_budget_underprovisioned", phase: "execution", tuple: "mlx:1b", evidence_schema_version: 1, model_id: "starvector_1b", model_repository: "starvector/starvector-1b-im2svg", model_revision: "e".repeat(40), configured_max_new_tokens: 4000, required_max_new_tokens: 7933, observed_quality_cases: 20, accepted_quality_cases: 11, rejected_quality_cases: 9, token_limit_quality_cases: 7, sanitizer_quality_cases: 2, required_accepted_quality_cases: 114, max_possible_accepted_quality_cases: 111 } };
  const upstream = { "upstream-controller.json": JSON.stringify({ schema_version: 1, campaign_run_id: value.campaign_id, inference_revision: value.inference_revision, sceneworks_revision: value.sceneworks_revision, workflow_run_id: value.workflow.run_id, workflow_run_attempt: 1 }) };
  const evidence = (index) => ({ accepted: index < 11, finishReason: index < 11 || index >= 18 ? "complete_root" : "token_limit", generatedTokens: index < 11 ? 100 : index >= 18 ? 200 : 4000, generatedBytes: 100, rejectionStage: index < 11 ? null : index >= 18 ? "sanitizer" : "generation_limit", rejectionCode: index < 11 || index >= 18 ? null : "token_limit", rejectionReason: index < 11 ? null : index >= 18 ? "SVG policy rejection" : "native StarVector stopped at the token_limit", providerId: "mlx-starvector-1b", backend: "mlx", modelId: value.failure.model_id, modelRepository: value.failure.model_repository, modelRevision: value.failure.model_revision });
  const route = Array.from({ length: 20 }, (_, index) => JSON.stringify({ case_id: `quality-v1-${index}`, job: { result: { terminalEvidence: evidence(index) } } })).join("\n") + "\n";
  const bundle = { schema_version: 1, inference_revision: value.inference_revision, tuples: { "mlx:1b": { image_quality: Array.from({ length: 120 }, () => ({ model: value.failure.model_id, detailBudget: { maxNewTokens: 4000, maxSvgBytes: 262144, maxWallTimeMs: 120000 } })), deterministic_parity: [], upstream_reference: { checkpoint_revision: value.failure.model_revision } } } };
  const records = { "controller-failure.json": JSON.stringify({ campaign_run_id: value.campaign_id, permanent_pin: value.inference_revision, tuple: value.failure.tuple, status: "failed" }), "preflight-provenance.json": JSON.stringify({ campaign_run_id: value.campaign_id, inference_revision: value.inference_revision, permanent_pin: value.inference_revision, tuple: value.failure.tuple, workflow_run_id: value.workflow.run_id, workflow_run_attempt: 1, service: { sceneworks_revision: value.sceneworks_revision, inference_revision: value.inference_revision, tuple: value.failure.tuple, models: { "starvector-1b": { revision: value.failure.model_revision } }, worker: { model_id: value.failure.model_id, provider_id: "mlx-starvector-1b" } } }), "case-bundle.json": JSON.stringify(bundle), "product-service-worker.stdout.log": "", "vector-generate-route.ndjson": route };
  value.failure.controller_failure_sha256 = sha(records["controller-failure.json"]);
  const writeArchives = async (suffix, rawRecords = records, combinedRecords = rawRecords) => ({ upstream: path.join(root, `upstream-${suffix}.zip`), raw: path.join(root, `raw-${suffix}.zip`), combined: path.join(root, `combined-${suffix}.zip`), ...await (async () => { await writeZip(root, `upstream-${suffix}`, upstream); await writeZip(root, `raw-${suffix}`, rawRecords); await writeZip(root, `combined-${suffix}`, Object.fromEntries(Object.entries(combinedRecords).map(([name, content]) => [`evidence/${name}`, content]))); return {}; })() });
  const archives = await writeArchives("valid"); await validateNativeExecutionArchives(value, archives);
  const wrongRoute = { ...records, "vector-generate-route.ndjson": route.replace('"generatedTokens":4000', '"generatedTokens":3999') };
  const routeArchives = await writeArchives("route", wrongRoute);
  await assert.rejects(() => validateNativeExecutionArchives(value, routeArchives), /underprovisioned quality budget/);
  const wrongBundle = structuredClone(bundle); wrongBundle.tuples["mlx:1b"].image_quality[0].detailBudget.maxNewTokens = 7933;
  const wrongBudget = { ...records, "case-bundle.json": JSON.stringify(wrongBundle) };
  const budgetArchives = await writeArchives("budget", wrongBudget);
  await assert.rejects(() => validateNativeExecutionArchives(value, budgetArchives), /underprovisioned budget/);
  const substituteArchives = await writeArchives("substitute", records, wrongRoute);
  await assert.rejects(() => validateNativeExecutionArchives(value, substituteArchives), /substituted vector route/);
  const wrongProvider = { ...records, "vector-generate-route.ndjson": route.replace('"providerId":"mlx-starvector-1b"', '"providerId":"other-provider"') };
  const wrongProviderArchives = await writeArchives("provider", wrongProvider);
  await assert.rejects(() => validateNativeExecutionArchives(value, wrongProviderArchives), /provider\/model identity/);
  const wrongController = { ...records, "controller-failure.json": JSON.stringify({ campaign_run_id: value.campaign_id, permanent_pin: value.inference_revision, tuple: value.failure.tuple, status: "failed", error: "different failure" }) };
  const wrongControllerArchives = await writeArchives("controller", wrongController);
  await assert.rejects(() => validateNativeExecutionArchives(value, wrongControllerArchives), /controller failure bytes/);
});
