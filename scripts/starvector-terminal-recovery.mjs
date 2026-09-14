// Historical evidence is copied into quarantine, never into current tuple inputs.
import { createHash } from "node:crypto";
import { lstat, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { execFile as execFileCallback } from "node:child_process";
import { promisify } from "node:util";
import path from "node:path";
import { isExecutedModule } from "./starvector-terminal-cli.mjs";

const execFile = promisify(execFileCallback);
export const stable = (value) => Array.isArray(value) ? `[${value.map(stable).join(",")}]` : value && typeof value === "object" ? `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(",")}}` : JSON.stringify(value);
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const fail = (message) => { throw new Error(`terminal recovery: ${message}`); };
const python = () => process.platform === "win32" ? process.env.STARVECTOR_TERMINAL_METRICS_PYTHON : "python3";
export function safeRecoveryPath(relative) {
  if (typeof relative !== "string" || !relative || relative.includes("\\") || relative.split("/").some((part) => !/^[A-Za-z0-9._:-]+$/.test(part) || [".", ".."].includes(part))) fail("unsafe evidence path");
  return relative;
}
export async function checkedRecoveryFile(root, relative, expected) {
  safeRecoveryPath(relative);
  let file = root;
  for (const part of relative.split("/")) {
    file = path.join(file, part);
    if ((await lstat(file)).isSymbolicLink()) fail(`symlink evidence ${relative}`);
  }
  const info = await lstat(file), bytes = await readFile(file);
  if (!info.isFile() || bytes.length !== expected.size || sha(bytes) !== expected.sha256) fail(`evidence bytes differ: ${relative}`);
  return bytes;
}
async function put(root, relative, bytes) {
  safeRecoveryPath(relative);
  const file = path.join(root, ...relative.split("/"));
  await mkdir(path.dirname(file), { recursive: true });
  try { await writeFile(file, bytes, { flag: "wx" }); } catch (error) {
    if (error.code !== "EEXIST") throw error;
    await checkedRecoveryFile(root, relative, { size: Buffer.byteLength(bytes), sha256: sha(bytes) });
  }
  return { path: relative, size: Buffer.byteLength(bytes), sha256: sha(bytes) };
}
async function treeInventory(root) {
  const entries = [];
  for (const name of await readdir(root, { recursive: true })) {
    const relative = name.split(path.sep).join("/"); safeRecoveryPath(relative);
    const info = await lstat(path.join(root, name));
    if (info.isSymbolicLink()) fail("archive contains symlink");
    if (info.isFile()) { const bytes = await readFile(path.join(root, name)); entries.push({ path: relative, byte_size: bytes.length, sha256: sha(bytes) }); }
  }
  return entries.sort((a, b) => a.path.localeCompare(b.path));
}

const EXECUTION_RECORDS = ["controller-failure.json", "preflight-provenance.json", "case-bundle.json", "product-service-worker.stdout.log"];
async function boundedArchiveRecords(archive, required) {
  const program = String.raw`import base64,json,pathlib,stat,sys,zipfile
archive=pathlib.Path(sys.argv[1]); required=json.loads(sys.argv[2]); result={}; names=set(); total=0
with zipfile.ZipFile(archive) as z:
 infos=z.infolist()
 if len(infos)>20000: raise ValueError('archive entry limit')
 for i in infos:
  p=pathlib.PurePosixPath(i.filename); total+=i.file_size
  if p.is_absolute() or '..' in p.parts or '\\' in i.filename or i.filename in names or stat.S_ISLNK(i.external_attr>>16) or total>512*1024*1024: raise ValueError('unsafe failed execution archive')
  names.add(i.filename)
 for suffix in required:
  matches=[i for i in infos if not i.is_dir() and (i.filename==suffix or i.filename.endswith('/'+suffix))]
  if len(matches)!=1 or matches[0].file_size>2*1024*1024: raise ValueError('missing, ambiguous, or oversized failed execution record '+suffix)
  result[suffix]=base64.b64encode(z.read(matches[0])).decode('ascii')
print(json.dumps(result,separators=(',',':')))`;
  let encoded;
  try {
    encoded = JSON.parse((await execFile(python(), ["-c", program, archive, JSON.stringify(required)], { maxBuffer: 8 * 1024 * 1024 })).stdout);
  } catch (error) {
    fail(`cannot inspect failed execution archive: ${error.message}`);
  }
  return Object.fromEntries(required.map((name) => [name, Buffer.from(encoded[name], "base64")]));
}
const parseRecord = (records, name) => {
  try { return JSON.parse(records[name]); } catch { fail(`invalid failed execution ${name}`); }
};

export async function validateNativeExecutionArchives(value, archives) {
  if (!archives?.upstream || !archives?.raw || !archives?.combined) fail("native execution archives are incomplete");
  const upstreamRecords = await boundedArchiveRecords(archives.upstream, ["upstream-controller.json"]);
  const rawRecords = await boundedArchiveRecords(archives.raw, EXECUTION_RECORDS);
  const combinedRecords = await boundedArchiveRecords(archives.combined, EXECUTION_RECORDS);
  for (const name of EXECUTION_RECORDS) if (!rawRecords[name].equals(combinedRecords[name])) fail(`combined archive substituted ${name}`);

  const expected = value.failure, upstream = parseRecord(upstreamRecords, "upstream-controller.json");
  if (upstream.schema_version !== expected.evidence_schema_version || upstream.campaign_run_id !== value.campaign_id || upstream.inference_revision !== value.inference_revision || upstream.sceneworks_revision !== value.sceneworks_revision || String(upstream.workflow_run_id) !== value.workflow.run_id || upstream.workflow_run_attempt !== value.workflow.run_attempt) fail("upstream controller identity differs from failed execution");
  const bundle = parseRecord(rawRecords, "case-bundle.json");
  const tuple = bundle.tuples?.[expected.tuple];
  const modelRows = [...(tuple?.image_quality ?? []), ...(tuple?.deterministic_parity ?? [])];
  if (bundle.schema_version !== expected.evidence_schema_version || bundle.inference_revision !== value.inference_revision || modelRows.length < 1 || modelRows.some((row) => row.model !== expected.model_id) || tuple?.upstream_reference?.checkpoint_revision !== expected.model_revision) fail("native case bundle identity differs from failed execution");
  const provenance = parseRecord(rawRecords, "preflight-provenance.json");
  if (provenance.campaign_run_id !== value.campaign_id || provenance.inference_revision !== value.inference_revision || provenance.permanent_pin !== value.inference_revision || provenance.tuple !== expected.tuple || String(provenance.workflow_run_id) !== value.workflow.run_id || provenance.workflow_run_attempt !== value.workflow.run_attempt || provenance.service?.sceneworks_revision !== value.sceneworks_revision || provenance.service?.inference_revision !== value.inference_revision || provenance.service?.tuple !== expected.tuple || provenance.service?.worker?.model_id !== expected.model_id || provenance.service?.worker?.provider_id !== "mlx-starvector-1b" || provenance.service?.models?.["starvector-1b"]?.revision !== expected.model_revision) fail("native preflight provenance differs from failed execution");
  const controller = parseRecord(rawRecords, "controller-failure.json");
  if (controller.campaign_run_id !== value.campaign_id || controller.permanent_pin !== value.inference_revision || controller.tuple !== expected.tuple || controller.status !== "failed") fail("native controller failure identity differs");
  const logRecords = rawRecords["product-service-worker.stdout.log"].toString("utf8").trim().split("\n").filter(Boolean).map((line) => { try { return JSON.parse(line); } catch { fail("invalid native worker evidence log"); } });
  const exactError = `vector_model_unavailable: exact receipt-backed snapshot ${expected.model_repository}@${expected.model_revision} is missing or unproven`;
  const failures = logRecords.filter((record) => record.event === "utility_job_failed");
  const selections = logRecords.filter((record) => record.event === "model_source_tier_selected");
  if (failures.length < 1 || failures.some((record) => record.error !== exactError) || selections.length < 1 || selections.some((record) => record.repository !== expected.model_repository || record.revision !== expected.model_revision)) fail("native worker failure does not prove the declared receipt rejection");
  return value;
}

// Invoked in a hosted preparation job, before the first hardware job. ZIPs are
// GitHub artifacts, not model downloads. A local archive directory supports the
// same production path for a CPU-only integration dry run.
export async function prepareRecovery(config, output, { archiveRoot, token = process.env.GH_TOKEN, fetchImpl = fetch } = {}) {
  if (config.schema_version !== 1) fail("unsupported recovery configuration");
  const predecessor = structuredClone(config);
  delete predecessor.schema_version; delete predecessor.authority; delete predecessor.execution_predecessor; delete predecessor.execution_history;
  const root = `quarantine/${safeRecoveryPath(predecessor.campaign_id)}`, entries = [];
  for (const [role, marker] of Object.entries(predecessor.markers)) {
    const bytes = Buffer.from(marker.content); delete marker.content;
    if (bytes.length !== marker.size || sha(bytes) !== marker.sha256) fail("marker copy differs from permanent digest");
    entries.push(await put(output, `${root}/markers/${role}/${marker.path}`, bytes));
  }
  entries.push(await put(output, `${root}/workflow-run.json`, stable(predecessor.workflow)));
  for (const artifact of predecessor.source_artifacts) {
    let bytes;
    if (archiveRoot) bytes = await readFile(path.join(archiveRoot, `${artifact.id}.zip`));
    else {
      if (!token) fail("Actions read token required for historical artifact selection");
      const response = await fetch(`https://api.github.com/repos/${artifact.repository}/actions/artifacts/${artifact.id}/zip`, { headers: { Authorization: `Bearer ${token}`, Accept: "application/vnd.github+json" }, signal: AbortSignal.timeout(60000) });
      if (!response.ok) fail(`historical artifact ${artifact.id}: HTTP ${response.status}`);
      bytes = Buffer.from(await response.arrayBuffer());
    }
    if (bytes.length !== artifact.size || `sha256:${sha(bytes)}` !== artifact.digest) fail(`archive identity mismatch ${artifact.id}`);
    const archive = `${root}/source-artifacts/${artifact.role}/${artifact.id}/${artifact.name}`;
    entries.push(await put(output, archive, bytes));
    const extracted = `${root}/source-artifacts/${artifact.role}/${artifact.id}/extracted`;
    // Reject traversal, duplicate entries, symlinks and archive bombs before extracting.
    await execFile(python(), ["-c", "import pathlib,sys,zipfile,stat\nz=zipfile.ZipFile(sys.argv[1]); names=set(); total=0\nfor i in z.infolist():\n p=pathlib.PurePosixPath(i.filename); total+=i.file_size\n if p.is_absolute() or '..' in p.parts or '\\\\' in i.filename or i.filename in names or stat.S_ISLNK(i.external_attr>>16) or total>512*1024*1024: raise ValueError('unsafe historical archive')\n names.add(i.filename)\n if len(names)>20000: raise ValueError('archive entry limit')\nz.extractall(sys.argv[2])", path.join(output, archive), path.join(output, extracted)]);
    artifact.content_inventory = await treeInventory(path.join(output, extracted));
    for (const entry of artifact.content_inventory) entries.push({ path: `${extracted}/${entry.path}`, size: entry.byte_size, sha256: entry.sha256 });
  }
  entries.sort((a, b) => a.path.localeCompare(b.path));
  predecessor.quarantine = { root, entries, aggregate_sha256: sha(stable({ root, entries })) };
  await put(output, `${root}/aggregate.json`, stable({ root, entries }));
  await put(output, "recovery-predecessor.json", stable(predecessor));
  for (const value of executionChain(config)) await prepareExecutionPredecessor({ ...config, execution_predecessor: value }, output, { archiveRoot, token, fetchImpl });
  return predecessor;
}

// Execution claims and historical native acceptance evidence have different
// scopes. An upstream-only failure has no native tuple receipt to quarantine.
export function validateExecutionPredecessor(config, run, artifact, jobs) {
  const value = config.execution_predecessor, workflow = value?.workflow, input = value?.source_artifact;
  if (!value || !["upstream-reference", "native"].includes(value.stage) || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value.campaign_id ?? "") || value.campaign_id === config.campaign_id || !/^[a-f0-9]{40}$/.test(value.inference_revision ?? "") || !/^[a-f0-9]{40}$/.test(value.sceneworks_revision ?? "")) fail("invalid execution predecessor identity");
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value.predecessor_campaign_id ?? "") || value.predecessor_campaign_id === value.campaign_id) fail("failed execution lacks its original claim predecessor");
  if (workflow?.repository !== "SceneWorks/SceneWorks" || ![".github/workflows/starvector-terminal.yml", ".github/workflows/server-candle-linux.yml"].includes(workflow.path) || !/^[1-9][0-9]*$/.test(workflow.run_id ?? "") || !Number.isSafeInteger(workflow.run_attempt) || workflow.run_attempt < 1 || workflow.head_sha !== value.sceneworks_revision || !["failure", "cancelled", "timed_out"].includes(workflow.conclusion)) fail("invalid failed execution workflow");
  if (String(run?.id) !== workflow.run_id || run.run_attempt !== workflow.run_attempt || run.head_sha !== workflow.head_sha || run.path !== workflow.path || run.event !== "workflow_dispatch" || run.status !== "completed" || run.conclusion !== workflow.conclusion) fail("authenticated failed execution workflow differs");
  if (!Array.isArray(jobs?.jobs) || jobs.total_count !== jobs.jobs.length) fail("failed execution job census is incomplete");
  if (value.stage === "upstream-reference") {
    if (!/^[1-9][0-9]*$/.test(input?.id ?? "") || input.name !== `starvector-upstream-${value.campaign_id}` || !Number.isSafeInteger(input.size) || input.size < 1 || !/^sha256:[a-f0-9]{64}$/.test(input.digest ?? "") || String(artifact?.id) !== input.id || artifact.name !== input.name || artifact.size_in_bytes !== input.size || artifact.digest !== input.digest || artifact.expired !== false || String(artifact.workflow_run?.id) !== workflow.run_id || artifact.workflow_run?.head_sha !== workflow.head_sha) fail("authenticated upstream artifact differs");
    for (const stage of ["upstream-reference", "mlx-1b", "mlx-8b", "cuda-1b", "cuda-8b"]) {
      const matches = jobs.jobs.filter(job => job.name === stage || job.name.endsWith(` / ${stage}`));
      if (matches.length !== 1 || matches[0].head_sha !== workflow.head_sha || (stage === "upstream-reference" ? !["failure", "cancelled", "timed_out"].includes(matches[0].conclusion) : matches[0].conclusion !== "skipped")) fail(`execution predecessor did not leave ${stage} in the required state`);
    }
    return value;
  }
  if (value.failure?.code !== "native_model_receipt_unproven" || value.failure.phase !== "execution" || value.failure.tuple !== "mlx:1b" || value.failure.evidence_schema_version !== 1 || value.failure.model_id !== "starvector_1b" || value.failure.model_repository !== "starvector/starvector-1b-im2svg" || !/^[a-f0-9]{40}$/.test(value.failure.model_revision ?? "")) fail("invalid native execution failure identity");
  const expectedRoles = ["upstream", "raw", "combined"], inputs = value.source_artifacts, artifacts = Array.isArray(artifact) ? artifact : [];
  if (!Array.isArray(inputs) || inputs.length !== expectedRoles.length || artifacts.length !== expectedRoles.length) fail("native execution artifact census differs");
  for (const [index, role] of expectedRoles.entries()) {
    const expected = inputs[index], observed = artifacts[index];
    const name = role === "upstream" ? `starvector-upstream-${value.campaign_id}` : role === "raw" ? `starvector-terminal-mlx-1b-${value.campaign_id}` : `starvector-terminal-receipt-${value.campaign_id}`;
    if (expected?.role !== role || !/^[1-9][0-9]*$/.test(expected.id ?? "") || expected.name !== name || !Number.isSafeInteger(expected.size) || expected.size < 1 || !/^sha256:[a-f0-9]{64}$/.test(expected.digest ?? "") || String(observed?.id) !== expected.id || observed.name !== expected.name || observed.size_in_bytes !== expected.size || observed.digest !== expected.digest || observed.expired !== false || String(observed.workflow_run?.id) !== workflow.run_id || observed.workflow_run?.head_sha !== workflow.head_sha) fail(`authenticated native ${role} artifact differs`);
  }
  for (const [stage, conclusion] of [["upstream-reference", "success"], ["mlx-1b", "failure"], ["mlx-8b", "skipped"], ["cuda-1b", "skipped"], ["cuda-8b", "skipped"], ["seal-receipt", "failure"]]) {
    const matches = jobs.jobs.filter(job => job.name === stage || job.name.endsWith(` / ${stage}`));
    if (matches.length !== 1 || matches[0].head_sha !== workflow.head_sha || matches[0].conclusion !== conclusion) fail(`native execution predecessor did not leave ${stage} in the required state`);
  }
  return value;
}

async function prepareExecutionPredecessor(config, output, { archiveRoot, token, fetchImpl }) {
  const value = config.execution_predecessor, workflow = value.workflow;
  if (!token) fail("Actions read token required for failed execution verification");
  const get = async (endpoint, binary = false) => {
    const response = await fetchImpl(`https://api.github.com/repos/SceneWorks/SceneWorks/${endpoint}`, { headers: { Authorization: `Bearer ${token}`, Accept: "application/vnd.github+json" }, signal: AbortSignal.timeout(60000) });
    if (!response.ok) fail(`failed execution metadata: HTTP ${response.status}`);
    return binary ? Buffer.from(await response.arrayBuffer()) : response.json();
  };
  // The attempt-specific endpoint never substitutes a newer re-run's outcome.
  const base = `actions/runs/${workflow.run_id}/attempts/${workflow.run_attempt}`, inputs = value.stage === "native" ? value.source_artifacts : [value.source_artifact];
  const run = await get(base), artifacts = await Promise.all(inputs.map((input) => get(`actions/artifacts/${input.id}`))), jobs = await get(`${base}/jobs?per_page=100`);
  validateExecutionPredecessor(config, run, value.stage === "native" ? artifacts : artifacts[0], jobs);
  const root = `execution-attempts/${value.campaign_id}`;
  const archivePaths = {};
  for (const input of inputs) {
    const bytes = archiveRoot ? await readFile(path.join(archiveRoot, `${input.id}.zip`)) : await get(`actions/artifacts/${input.id}/zip`, true);
    if (bytes.length !== input.size || `sha256:${sha(bytes)}` !== input.digest) fail("failed execution archive identity differs");
    const role = value.stage === "native" ? input.role : "upstream";
    await put(output, `${root}/${role}.zip`, bytes);
    archivePaths[role] = path.join(output, root, `${role}.zip`);
  }
  if (value.stage === "native") await validateNativeExecutionArchives(value, archivePaths);
  await put(output, `${root}/metadata.json`, stable({ predecessor: value, run, ...(value.stage === "native" ? { artifacts } : { artifact: artifacts[0] }), jobs }));
}

// Ordered upstream-only history is retained separately from native receipts.
function executionChain(config) {
  const history = config.execution_history ?? [];
  if (!Array.isArray(history) || (history.length && !config.execution_predecessor)) fail("invalid execution history");
  const chain = [...history, ...(config.execution_predecessor ? [config.execution_predecessor] : [])];
  let previous = config.campaign_id;
  const seen = new Set([previous]);
  for (const value of chain) {
    if (!value || seen.has(value.campaign_id) || value.predecessor_campaign_id !== previous) fail("execution history is not an ordered successor chain");
    seen.add(value.campaign_id); previous = value.campaign_id;
  }
  return chain;
}

export async function verifyExecutionPredecessor(config, root, nativePredecessor) {
  for (const value of executionChain(config)) {
    const relative = `execution-attempts/${safeRecoveryPath(value.campaign_id)}`;
    const metadataPath = `${relative}/metadata.json`, info = await lstat(path.join(root, metadataPath));
    const bytes = await checkedRecoveryFile(root, metadataPath, { size: info.size, sha256: sha(await readFile(path.join(root, metadataPath))) });
    const metadata = JSON.parse(bytes);
    if (stable(metadata.predecessor) !== stable(value)) fail("failed execution declaration differs from prepared evidence");
    validateExecutionPredecessor({ ...config, execution_predecessor: value }, metadata.run, value.stage === "native" ? metadata.artifacts : metadata.artifact, metadata.jobs);
    const inputs = value.stage === "native" ? value.source_artifacts : [value.source_artifact];
    for (const input of inputs) await checkedRecoveryFile(root, `${relative}/${value.stage === "native" ? input.role : "upstream"}.zip`, { size: input.size, sha256: input.digest.slice(7) });
    if (value.stage === "native") await validateNativeExecutionArchives(value, Object.fromEntries(inputs.map((input) => [input.role, path.join(root, relative, `${input.role}.zip`)])));
  }
  return config.execution_predecessor ?? nativePredecessor;
}

export async function verifyRecovery(config, root, { campaignRunId, permanentPin, leaseRoot } = {}) {
  if (campaignRunId === config.campaign_id) fail("retired campaign identity cannot execute again");
  if (permanentPin !== undefined && !/^[a-f0-9]{40}$/.test(permanentPin)) fail("exact successor inference pin required");
  const predecessor = JSON.parse(await readFile(path.join(root, "recovery-predecessor.json"), "utf8"));
  for (const key of ["campaign_id", "inference_revision", "sceneworks_revision", "workflow", "failure"]) if (stable(predecessor[key]) !== stable(config[key])) fail(`historical ${key} differs`);
  for (const [role, marker] of Object.entries(config.markers)) {
    const { content, ...expected } = marker;
    if (stable(predecessor.markers[role]) !== stable(expected)) fail("historical marker identity differs");
    if (leaseRoot) await checkedRecoveryFile(leaseRoot, marker.path, marker);
  }
  if (predecessor.source_artifacts.length !== config.source_artifacts.length) fail("historical artifact selection differs");
  const expectedEntries = [];
  const q = predecessor.quarantine, expectedRoot = `quarantine/${config.campaign_id}`;
  if (q.root !== expectedRoot) fail("historical quarantine root differs");
  for (const [role, marker] of Object.entries(predecessor.markers)) expectedEntries.push({ ...marker, path: `${expectedRoot}/markers/${role}/${marker.path}` });
  const workflow = stable(predecessor.workflow);
  expectedEntries.push({ path: `${expectedRoot}/workflow-run.json`, size: Buffer.byteLength(workflow), sha256: sha(workflow) });
  for (let i = 0; i < predecessor.source_artifacts.length; i++) {
    const artifact = predecessor.source_artifacts[i], { content_inventory, ...metadata } = artifact;
    if (stable(metadata) !== stable(config.source_artifacts[i])) fail("historical artifact metadata differs");
    const prefix = `${expectedRoot}/source-artifacts/${artifact.role}/${artifact.id}`;
    expectedEntries.push({ path: `${prefix}/${artifact.name}`, size: artifact.size, sha256: artifact.digest.slice(7) });
    const actual = await treeInventory(path.join(root, prefix, "extracted"));
    if (stable(actual) !== stable(content_inventory)) fail("historical inventory differs");
    const archivePath = path.join(root, prefix, artifact.name);
    await checkedRecoveryFile(root, `${prefix}/${artifact.name}`, { size: artifact.size, sha256: artifact.digest.slice(7) });
    const zipped = JSON.parse((await execFile(python(), ["-c", "import hashlib,json,sys,zipfile\nz=zipfile.ZipFile(sys.argv[1]); print(json.dumps([{'path':i.filename,'byte_size':i.file_size,'sha256':hashlib.sha256(z.read(i)).hexdigest()} for i in z.infolist() if not i.is_dir()]))", archivePath], { maxBuffer: 8 * 1024 * 1024 })).stdout).sort((a, b) => a.path.localeCompare(b.path));
    if (stable(zipped) !== stable(actual)) fail("extracted files differ from immutable archive");
    for (const entry of actual) expectedEntries.push({ path: `${prefix}/extracted/${entry.path}`, size: entry.byte_size, sha256: entry.sha256 });
  }
  expectedEntries.sort((a, b) => a.path.localeCompare(b.path));
  if (stable(q.entries) !== stable(expectedEntries) || q.aggregate_sha256 !== sha(stable({ root: q.root, entries: q.entries }))) fail("historical quarantine closure differs");
  for (const entry of q.entries) await checkedRecoveryFile(root, entry.path, entry);
  await checkedRecoveryFile(root, `${q.root}/aggregate.json`, { size: Buffer.byteLength(stable({ root: q.root, entries: q.entries })), sha256: q.aggregate_sha256 });
  return predecessor;
}

export async function bindRecoveryLineage(receipt, config, recoveryRoot, canonicalRoot) {
  const predecessor = await verifyRecovery(config, recoveryRoot, { campaignRunId: receipt.campaign_run_id, permanentPin: receipt.inference_revision });
  predecessor.superseded_by = receipt.campaign_run_id;
  const current = { campaign_id: receipt.campaign_run_id, inference_revision: receipt.inference_revision, sceneworks_revision: receipt.sceneworks_revision, repository: receipt.execution.repository, path: process.env.GITHUB_WORKFLOW_REF?.match(/^[^/]+\/[^/]+\/(.+)@/)?.[1] ?? ".github/workflows/starvector-terminal.yml", run_id: receipt.execution.workflow_run_id, run_attempt: receipt.execution.workflow_run_attempt, head_sha: receipt.execution.head_sha };
  const record = { predecessor_campaign_id: predecessor.campaign_id, successor_campaign_id: receipt.campaign_run_id, predecessor_inference_revision: predecessor.inference_revision, predecessor_sceneworks_revision: predecessor.sceneworks_revision, successor_inference_revision: receipt.inference_revision, successor_sceneworks_revision: receipt.sceneworks_revision };
  record.authority = await put(canonicalRoot, `lineage/supersession-records/${predecessor.campaign_id}-to-${receipt.campaign_run_id}.json`, stable({ ...record, ...config.authority, current_workflow: current }));
  for (const entry of predecessor.quarantine.entries) await put(canonicalRoot, entry.path, await checkedRecoveryFile(recoveryRoot, entry.path, entry));
  await put(canonicalRoot, `${predecessor.quarantine.root}/aggregate.json`, stable({ root: predecessor.quarantine.root, entries: predecessor.quarantine.entries }));
  const lineage = { kind: "failed_campaign_supersession", current_campaign_id: receipt.campaign_run_id, current_workflow: current, failed_predecessors: [predecessor], supersession_records: [record] };
  receipt.schema_version = 2; receipt.campaign_lineage = lineage; receipt.producer.campaign_lineage_sha256 = sha(stable(lineage));
  await put(canonicalRoot, "lineage/current-workflow.json", stable(current));
  await put(canonicalRoot, "lineage/campaign-lineage.json", stable(lineage));
  return lineage;
}

if (isExecutedModule(import.meta.url)) {
  const [mode, configPath, output, archives] = process.argv.slice(2);
  Promise.resolve().then(async () => {
    const config = JSON.parse(await readFile(configPath, "utf8"));
    if (mode === "prepare") await prepareRecovery(config, output, { archiveRoot: archives });
    else if (mode === "verify") await verifyRecovery(config, output, { campaignRunId: process.env.STARVECTOR_TERMINAL_CAMPAIGN_RUN_ID, permanentPin: process.env.STARVECTOR_TERMINAL_PERMANENT_PIN });
    else fail("usage: prepare|verify config output [archives]");
  }).catch((error) => { console.error(error.message); process.exitCode = 1; });
}
