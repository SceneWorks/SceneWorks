import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, writeFile, rm, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { bindRecoveryLineage, checkedRecoveryFile, prepareRecovery, safeRecoveryPath, stable, validateExecutionPredecessor, validateNativeExecutionArchives, verifyExecutionPredecessor, verifyRecovery } from "./starvector-terminal-recovery.mjs";

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
async function writeZip(root, id, files) {
  const source = path.join(root, `zip-${id}`);
  for (const [relative, content] of Object.entries(files)) {
    const file = path.join(source, relative); await mkdir(path.dirname(file), { recursive: true }); await writeFile(file, content);
  }
  const archive = path.join(root, `${id}.zip`);
  execFileSync("python3", ["-c", "import os,sys,zipfile\nroot,out=sys.argv[1:]\nwith zipfile.ZipFile(out,'w') as z:\n for base,_,names in os.walk(root):\n  for name in names:\n   p=os.path.join(base,name); z.write(p,os.path.relpath(p,root))", source, archive]);
  return readFile(archive);
}
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
