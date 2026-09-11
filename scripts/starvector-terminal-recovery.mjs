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

// Invoked in a hosted preparation job, before the first hardware job. ZIPs are
// GitHub artifacts, not model downloads. A local archive directory supports the
// same production path for a CPU-only integration dry run.
export async function prepareRecovery(config, output, { archiveRoot, token = process.env.GH_TOKEN, fetchImpl = fetch } = {}) {
  if (config.schema_version !== 1) fail("unsupported recovery configuration");
  const predecessor = structuredClone(config);
  delete predecessor.schema_version; delete predecessor.authority; delete predecessor.execution_predecessor;
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
  if (config.execution_predecessor) await prepareExecutionPredecessor(config, output, { archiveRoot, token, fetchImpl });
  return predecessor;
}

// Execution claims and historical native acceptance evidence have different
// scopes. An upstream-only failure has no native tuple receipt to quarantine.
export function validateExecutionPredecessor(config, run, artifact, jobs) {
  const value = config.execution_predecessor, workflow = value?.workflow, input = value?.source_artifact;
  if (!value || value.stage !== "upstream-reference" || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value.campaign_id ?? "") || value.campaign_id === config.campaign_id || !/^[a-f0-9]{40}$/.test(value.inference_revision ?? "") || !/^[a-f0-9]{40}$/.test(value.sceneworks_revision ?? "")) fail("invalid execution predecessor identity");
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value.predecessor_campaign_id ?? "") || value.predecessor_campaign_id === value.campaign_id) fail("failed execution lacks its original claim predecessor");
  if (workflow?.repository !== "SceneWorks/SceneWorks" || ![".github/workflows/starvector-terminal.yml", ".github/workflows/server-candle-linux.yml"].includes(workflow.path) || !/^[1-9][0-9]*$/.test(workflow.run_id ?? "") || !Number.isSafeInteger(workflow.run_attempt) || workflow.run_attempt < 1 || workflow.head_sha !== value.sceneworks_revision || !["failure", "cancelled", "timed_out"].includes(workflow.conclusion)) fail("invalid failed execution workflow");
  if (String(run?.id) !== workflow.run_id || run.run_attempt !== workflow.run_attempt || run.head_sha !== workflow.head_sha || run.path !== workflow.path || run.event !== "workflow_dispatch" || run.status !== "completed" || run.conclusion !== workflow.conclusion) fail("authenticated failed execution workflow differs");
  if (!/^[1-9][0-9]*$/.test(input?.id ?? "") || input.name !== `starvector-upstream-${value.campaign_id}` || !Number.isSafeInteger(input.size) || input.size < 1 || !/^sha256:[a-f0-9]{64}$/.test(input.digest ?? "") || String(artifact?.id) !== input.id || artifact.name !== input.name || artifact.size_in_bytes !== input.size || artifact.digest !== input.digest || artifact.expired !== false || String(artifact.workflow_run?.id) !== workflow.run_id || artifact.workflow_run?.head_sha !== workflow.head_sha) fail("authenticated upstream artifact differs");
  if (!Array.isArray(jobs?.jobs) || jobs.total_count !== jobs.jobs.length) fail("failed execution job census is incomplete");
  for (const stage of ["upstream-reference", "mlx-1b", "mlx-8b", "cuda-1b", "cuda-8b"]) {
    const matches = jobs.jobs.filter(job => job.name === stage || job.name.endsWith(` / ${stage}`));
    if (matches.length !== 1 || matches[0].head_sha !== workflow.head_sha || (stage === "upstream-reference" ? !["failure", "cancelled", "timed_out"].includes(matches[0].conclusion) : matches[0].conclusion !== "skipped")) fail(`execution predecessor did not leave ${stage} in the required state`);
  }
  return value;
}

async function prepareExecutionPredecessor(config, output, { archiveRoot, token, fetchImpl }) {
  const value = config.execution_predecessor, workflow = value.workflow, input = value.source_artifact;
  if (!token) fail("Actions read token required for failed execution verification");
  const get = async (endpoint, binary = false) => {
    const response = await fetchImpl(`https://api.github.com/repos/SceneWorks/SceneWorks/${endpoint}`, { headers: { Authorization: `Bearer ${token}`, Accept: "application/vnd.github+json" }, signal: AbortSignal.timeout(60000) });
    if (!response.ok) fail(`failed execution metadata: HTTP ${response.status}`);
    return binary ? Buffer.from(await response.arrayBuffer()) : response.json();
  };
  // The attempt-specific endpoint never substitutes a newer re-run's outcome.
  const base = `actions/runs/${workflow.run_id}/attempts/${workflow.run_attempt}`;
  const run = await get(base), artifact = await get(`actions/artifacts/${input.id}`), jobs = await get(`${base}/jobs?per_page=100`);
  validateExecutionPredecessor(config, run, artifact, jobs);
  const bytes = archiveRoot ? await readFile(path.join(archiveRoot, `${input.id}.zip`)) : await get(`actions/artifacts/${input.id}/zip`, true);
  if (bytes.length !== input.size || `sha256:${sha(bytes)}` !== input.digest) fail("failed execution archive identity differs");
  const root = `execution-attempts/${value.campaign_id}`;
  await put(output, `${root}/upstream.zip`, bytes);
  await put(output, `${root}/metadata.json`, stable({ predecessor: value, run, artifact, jobs }));
}

export async function verifyExecutionPredecessor(config, root, nativePredecessor) {
  const value = config.execution_predecessor;
  if (!value) return nativePredecessor;
  const relative = `execution-attempts/${safeRecoveryPath(value.campaign_id)}`;
  const metadataPath = `${relative}/metadata.json`, info = await lstat(path.join(root, metadataPath));
  const bytes = await checkedRecoveryFile(root, metadataPath, { size: info.size, sha256: sha(await readFile(path.join(root, metadataPath))) });
  const metadata = JSON.parse(bytes);
  if (stable(metadata.predecessor) !== stable(value)) fail("failed execution declaration differs from prepared evidence");
  validateExecutionPredecessor(config, metadata.run, metadata.artifact, metadata.jobs);
  await checkedRecoveryFile(root, `${relative}/upstream.zip`, { size: value.source_artifact.size, sha256: value.source_artifact.digest.slice(7) });
  return value;
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
