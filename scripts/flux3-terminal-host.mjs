// sc-26243: per-host MLX terminal steps behind the manual `flux3-image-klein-terminal-host` workflow.
// Runs ONLY on the Mac the dispatch names, under a fixed $HOME/sc26243-run, from a hash-pinned local
// control file. Nothing is uploaded: the repository and its Actions logs are PUBLIC, so this script
// prints hashes and counts only, and every child's own output goes to a local log file.
import assert from 'node:assert/strict';
import { spawn, execFile as execFileCb } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { copyFile, mkdir, open, readFile, readdir, stat, statfs, writeFile } from 'node:fs/promises';
import { hostname, userInfo } from 'node:os';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { promisify } from 'node:util';
import { MAC_RUNNERS } from './flux3-image-klein-terminal-session.mjs';
const execFile = promisify(execFileCb);
export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const STEPS = Object.freeze(['prepare', 'self-test', 'run']);
export const NONE = 'none';
export const HOST_DIRECTORY = 'sc26243-run';
// Personal inputs live ONLY on the dataset host and are verified against this manifest first.
export const DATASET_HOST = 'nax-macos';
export const DATASET_INPUTS_MANIFEST_SHA256 = 'a9551ff98b7088c3653052079ba19e00644190f0f7aac878b58d8835e886f0d9';
export const MLX_CONVERSIONS = Object.freeze(['flux3-mlx-affine-q4-g64-v1', 'flux3-mlx-affine-q8-g64-v1']);
const sha = /^[a-f0-9]{64}$/, revision = /^[a-f0-9]{40}$/;
const blocked = message => new Error(`BLOCKED: ${message}`);
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
export async function fileSha256(file) { const h = createHash('sha256'); for await (const chunk of createReadStream(file)) h.update(chunk); return h.digest('hex'); }

/** Typed dispatch inputs. GitHub turns ANY empty dispatch value into the declared default, so every
 *  input defaults to the sentinel `none`, which is refused: absence can never select a host/step. */
export function resolveDispatch(env) {
  const host = env.FLUX3_TERMINAL_HOST, step = env.FLUX3_TERMINAL_STEP, controlSha256 = env.FLUX3_TERMINAL_CONTROL_SHA256;
  if (!host || host === NONE) throw blocked('flux3_terminal_host is the "none" sentinel; name nax-macos or nax-macos-2 explicitly');
  if (!step || step === NONE) throw blocked('flux3_terminal_step is the "none" sentinel; name prepare, self-test or run explicitly');
  if (!controlSha256 || controlSha256 === NONE) throw blocked('flux3_terminal_control_sha256 is the "none" sentinel; pin the local control file by its sha256');
  if (!Object.hasOwn(MAC_RUNNERS, host)) throw blocked(`unknown terminal host ${host}`);
  if (!STEPS.includes(step)) throw blocked(`unknown terminal step ${step}`);
  if (!sha.test(controlSha256)) throw blocked('control sha256 must be 64 lowercase hex');
  if (!revision.test(env.FLUX3_TERMINAL_SOURCE_SHA ?? '') || !revision.test(env.FLUX3_TERMINAL_INFERENCE_SHA ?? '')) throw blocked('exact 40-hex source and inference SHAs required');
  return { host, step, controlSha256, sourceSha: env.FLUX3_TERMINAL_SOURCE_SHA, inferenceMainSha: env.FLUX3_TERMINAL_INFERENCE_SHA, ...MAC_RUNNERS[host] };
}
/** Wrong-host refusal: the job's actual runner must be the dispatched host (a label is not a name). */
export function hostRoot(dispatch, env) {
  if (env.RUNNER_NAME !== dispatch.host) throw blocked(`dispatched for ${dispatch.host} but running on ${env.RUNNER_NAME ?? 'an unnamed runner'}; refusing before any work`);
  if (!path.isAbsolute(env.HOME ?? '')) throw blocked('absolute HOME required');
  return path.join(env.HOME, HOST_DIRECTORY);
}
/** The hash-pinned local control file for exactly this host and step. */
export async function readControl(root, dispatch) {
  const file = path.join(root, 'control', dispatch.host, `${dispatch.step}.json`);
  let bytes; try { bytes = await readFile(file); } catch { throw blocked(`control file for ${dispatch.host}/${dispatch.step} is not recorded on this host`); }
  if (digest(bytes) !== dispatch.controlSha256) throw blocked('control file differs from the dispatched sha256');
  const control = JSON.parse(bytes);
  assert.equal(control.kind, 'flux3-terminal-host-control'); assert.equal(control.host, dispatch.host); assert.equal(control.step, dispatch.step);
  assert.equal(control.sourceSha, dispatch.sourceSha); assert.equal(control.inferenceMainSha, dispatch.inferenceMainSha);
  return control;
}
/** Read a sibling control artifact (approval/addendum), pinned by the control file's own digest. */
async function pinned(root, dispatch, name, expected) {
  assert(/^[a-zA-Z0-9_.-]+\.json$/.test(name ?? ''), 'control artifacts are plain file names');
  const bytes = await readFile(path.join(root, 'control', dispatch.host, name));
  if (digest(bytes) !== expected) throw blocked(`${name} differs from the control-pinned sha256`);
  return JSON.parse(bytes);
}

/** Verify every personal input against the approved manifest; refuse on any mismatch or extra file. */
export async function verifyInputsManifest(inputsRoot, expectedSha256 = DATASET_INPUTS_MANIFEST_SHA256) {
  const bytes = await readFile(path.join(inputsRoot, 'inputs-manifest.json'));
  if (digest(bytes) !== expectedSha256) throw blocked('inputs manifest differs from the approved manifest');
  const manifest = JSON.parse(bytes), expected = new Map();
  for (const r of manifest.references) expected.set(`references/${r.file}`, r);
  for (const t of manifest.training) { expected.set(`training/${t.file}`, t); expected.set(`training/${t.file.replace(/\.png$/, '.txt')}`, { sha256: t.captionSha256 }); }
  for (const directory of ['references', 'training']) {
    for (const entry of await readdir(path.join(inputsRoot, directory), { withFileTypes: true })) {
      const rel = `${directory}/${entry.name}`;
      if (!entry.isFile() || !expected.has(rel)) throw blocked('unexpected input entry; inputs must match the manifest exactly');
    }
  }
  let count = 0;
  for (const [rel, row] of expected) {
    const file = path.join(inputsRoot, rel);
    let info; try { info = await stat(file); } catch { throw blocked('a manifest input file is missing'); }
    if ((row.bytes !== undefined && info.size !== row.bytes) || await fileSha256(file) !== row.sha256) throw blocked('an input file differs from the manifest');
    count++;
  }
  return { manifestSha256: expectedSha256, references: manifest.references.length, training: manifest.training.length, files: count };
}

/** `run` is refused until THIS host lane's parent-recorded addendum is present and valid. */
export async function laneAddendum(approval, addendum, host) {
  const { validateInputSpecAddendum, ledgerLane } = await import('./flux3-image-klein-terminal-budget.mjs');
  if (!addendum) throw blocked(`run refused: no recorded addendum for mlx@${host}`);
  const bound = validateInputSpecAddendum(approval, addendum);
  if (bound.ledgerLane !== ledgerLane(approval, 'mlx', host)) throw blocked(`run refused: the addendum binds ${bound.ledgerLane}, not mlx@${host}`);
  return bound;
}

// ---------------------------------------------------------------- host execution (not unit-tested)
const say = value => console.log(JSON.stringify(value));
async function child(root, name, command, args, { env = process.env, cwd = ROOT } = {}) {
  await mkdir(path.join(root, 'logs'), { recursive: true });
  const logFile = path.join(root, 'logs', `${new Date().toISOString().replace(/[:.]/g, '-')}-${name}.log`), log = await open(logFile, 'wx');
  try {
    const code = await new Promise((resolve, reject) => { const c = spawn(command, args, { cwd, env, stdio: ['ignore', log.fd, log.fd] }); c.once('error', reject); c.once('close', resolve); });
    say({ action: name, exitCode: code });
    if (code !== 0) throw blocked(`${name} failed (exit ${code}); its log stays on this host only`);
  } finally { await log.close(); }
  return logFile;
}
async function readJson(file) { return JSON.parse(await readFile(file, 'utf8')); }
async function exists(file) { try { await stat(file); return true; } catch { return false; } }
const node = process.execPath, driver = path.join(ROOT, 'scripts/flux3-image-klein-terminal-driver.mjs');
async function inferenceCheckout(root, inferenceMainSha) {
  const repo = path.join(root, 'inference');
  if (!await exists(path.join(repo, '.git'))) await child(root, 'inference-clone', 'git', ['clone', '--filter=blob:none', '--no-checkout', 'https://github.com/SceneWorks/inference.git', repo]);
  await child(root, 'inference-fetch', 'git', ['-C', repo, 'fetch', '--filter=blob:none', 'origin', 'main']);
  await child(root, 'inference-ancestry', 'git', ['-C', repo, 'merge-base', '--is-ancestor', inferenceMainSha, 'refs/remotes/origin/main']);
  return { repo, tip: (await execFile('git', ['-C', repo, 'rev-parse', 'refs/remotes/origin/main'])).stdout.trim() };
}
async function toolchain() {
  const which = async tool => (await execFile('rustup', ['which', tool], { cwd: ROOT })).stdout.trim();
  const cargo = await which('cargo'), rustc = await which('rustc');
  return { cargo, rustc, cargoSha256: await fileSha256(cargo), rustcSha256: await fileSha256(rustc) };
}
function state(root, dispatch, name) { return path.join(root, 'state', `${dispatch.sourceSha.slice(0, 12)}-${dispatch.host}-${name}.json`); }
async function build(root, dispatch, control, role, inference, tools) {
  const resultFile = state(root, dispatch, `${role}-build`);
  if (await exists(resultFile)) return readJson(resultFile);
  const lock = await readFile(path.join(ROOT, 'Cargo.lock'));
  const outputRoot = path.join(root, 'builds', dispatch.sourceSha.slice(0, 12), role); await mkdir(outputRoot, { recursive: true });
  const admission = { actualBuild: true, userBuildApprovalId: control.userBuildApprovalId, resourceAdmissionId: control.resourceAdmissionId, binaryRole: role, backend: 'mlx',
    sourceSha: dispatch.sourceSha, inferenceMainSha: dispatch.inferenceMainSha, inferenceRepository: inference.repo, cargoExecutable: tools.cargo, cargoExecutableSha256: tools.cargoSha256,
    rustcExecutable: tools.rustc, rustcExecutableSha256: tools.rustcSha256, targetDir: path.join(root, 'target'), buildOutputRoot: outputRoot, cargoLockSha256: digest(lock), rustflags: '', timeoutMs: 10_800_000 };
  const admissionFile = state(root, dispatch, `${role}-build-admission`); await writeFile(admissionFile, JSON.stringify(admission, null, 2) + '\n', { flag: 'wx' });
  const log = await child(root, `build-${role}`, node, [driver, 'produce-build', admissionFile], { env: { ...process.env, CUDA_VISIBLE_DEVICES: '-1' } });
  const result = JSON.parse((await readFile(log, 'utf8')).trim().split('\n').at(-1));
  const transcript = await readJson(path.join(path.dirname(result.receiptPath), 'producer-transcript.json'));
  const value = { ...result, binding: { receiptSha256: result.receiptSha256, transcriptSha256: result.receipt.transcriptSha256, binarySha256: result.receipt.binary.sha256, cargoLockSha256: result.receipt.cargoLockSha256,
    rustflagsSha256: digest(transcript.rustflags), cargoExecutableSha256: transcript.cargoExecutableSha256, rustcExecutableSha256: transcript.rustcExecutableSha256 } };
  await writeFile(resultFile, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
  say({ action: `build-${role}`, receiptSha256: value.binding.receiptSha256, binarySha256: value.binding.binarySha256 });
  return value;
}
async function preparation(root, dispatch, control, api, inference) {
  const resultFile = state(root, dispatch, 'preparation');
  if (await exists(resultFile)) return readJson(resultFile);
  const request = { backend: 'mlx', sourceSha: dispatch.sourceSha, inferenceMainSha: dispatch.inferenceMainSha, modelIds: control.modelIds };
  const requestFile = state(root, dispatch, 'preparation-request'); await writeFile(requestFile, JSON.stringify(request) + '\n', { flag: 'wx' });
  const planLog = await child(root, 'preparation-plan', node, [driver, 'prepare', 'plan', requestFile]);
  const plan = JSON.parse(await readFile(planLog, 'utf8')); const planFile = state(root, dispatch, 'preparation-plan'); await writeFile(planFile, JSON.stringify(plan, null, 2) + '\n', { flag: 'wx' });
  const account = userInfo().username, outputRoot = path.join(root, 'preparation', dispatch.sourceSha.slice(0, 12)); await mkdir(outputRoot, { recursive: true });
  const a = { actualProvision: true, userProvisionApprovalId: control.userProvisionApprovalId, admissionId: control.resourceAdmissionId, approvedBy: control.approvedBy, runner: dispatch.host, runnerAccount: account, backend: 'mlx',
    sourceSha: dispatch.sourceSha, inferenceMainSha: dispatch.inferenceMainSha, inferenceMainTipSha: inference.tip, inferenceRepository: inference.repo, preparationPlanSha256: plan.planSha256, outputRoot, hfHome: control.hfHome,
    binary: api.receipt.binary.path, binarySourceSha: dispatch.sourceSha, binarySha256: api.receipt.binary.sha256, binaryBuildReceipt: { root: path.dirname(api.receiptPath), path: api.receiptPath, sha256: api.receiptSha256 },
    preparationBuildReceiptBinding: api.binding, timeoutSeconds: 10800, diskBudgetBytes: plan.downloadBytes + 60 * 1024 ** 3, generationAttempts: 0,
    downloadApprovals: plan.repositories.map(r => ({ repository: r.repository, resolvedRevision: r.resolvedRevision, userDownloadApprovalId: `${control.userProvisionApprovalId}-${r.modelId}`, wholeRepository: true, licenseAcknowledged: true, accountGateAcknowledged: true, diskAcknowledged: true, downloadBytes: r.downloadBytes, runnerAccount: account, hfHome: control.hfHome })) };
  const admissionFile = state(root, dispatch, 'preparation-admission'); await writeFile(admissionFile, JSON.stringify(a, null, 2) + '\n', { flag: 'wx' });
  const log = await child(root, 'preparation-run', node, [driver, 'prepare', 'run', admissionFile, planFile], { env: { ...process.env, CUDA_VISIBLE_DEVICES: '-1' } });
  const result = JSON.parse(await readFile(log, 'utf8')); const reportSha256 = await fileSha256(result.file);
  const value = { reportFile: result.file, reportSha256, status: result.status }; await writeFile(resultFile, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
  say({ action: 'preparation', status: result.status, reportSha256, repositories: plan.repositories.length });
  return value;
}
async function conversions(root, dispatch, report) {
  const resultFile = state(root, dispatch, 'conversions');
  if (await exists(resultFile)) return readJson(resultFile);
  const source = report.inventories.find(i => i.modelId === 'flux3_klein').root, out = path.join(root, 'conversions');
  await child(root, 'conversions', 'cargo', ['test', '--offline', '--locked', '--release', '-p', 'sceneworks-worker', '--lib', '-j2', 'local_derivation::tests::flux3_capture_conversion_artifacts', '--', '--ignored', '--exact', '--nocapture'],
    { env: { ...process.env, CUDA_VISIBLE_DEVICES: '-1', FLUX3_CAPTURE_MODEL_ID: 'flux3_klein', FLUX3_CAPTURE_SOURCE_DIR: source, FLUX3_CAPTURE_OUT_DIR: out, FLUX3_CAPTURE_CONVERSIONS: MLX_CONVERSIONS.join(',') } });
  const value = { root: out, packages: {} };
  for (const id of MLX_CONVERSIONS) { const tier = id.includes('-q4-') ? 'q4' : 'q8'; value.packages[tier] = { root: path.join(out, 'flux3_klein', tier, id), files: await listTree(path.join(out, 'flux3_klein', tier, id)) }; }
  await writeFile(resultFile, JSON.stringify(value, null, 2) + '\n', { flag: 'wx' });
  say({ action: 'conversions', packages: Object.fromEntries(Object.entries(value.packages).map(([t, p]) => [t, digest(JSON.stringify(p.files))])) });
  return value;
}
async function listTree(root, prefix = '') {
  const out = [];
  for (const e of await readdir(path.join(root, prefix), { withFileTypes: true })) {
    const rel = prefix ? `${prefix}/${e.name}` : e.name;
    if (e.isDirectory()) out.push(...await listTree(root, rel)); else { const full = path.join(root, rel); out.push({ path: rel, bytes: (await stat(full)).size, sha256: await fileSha256(full) }); }
  }
  return out.sort((a, b) => a.path.localeCompare(b.path));
}
/** The lane's MLX input specification from genuine host observations only (inventories, install
 *  receipts, host conversions and, on the dataset host, the verified personal inputs). */
export async function hostInputSpec({ report, reportFile, hfHome, host, converted, inputsRoot }) {
  const AUTH = 'parent-approved-pinned-inventory', stateDir = path.join(path.dirname(reportFile), `state-${path.basename(reportFile).split('-observed-inventories.json')[0]}`);
  const inv = id => report.inventories.find(i => i.modelId === id), plain = files => files.map(({ path: p, bytes, sha256 }) => ({ path: p, bytes, sha256 }));
  const strip = (files, prefix) => files.filter(f => f.path.startsWith(prefix)).map(f => ({ path: f.path.slice(prefix.length), bytes: f.bytes, sha256: f.sha256 }));
  const safe = repo => repo.replace(/[^A-Za-z0-9._-]+/g, '__').replace(/^_+|_+$/g, '');
  const receipt = async repo => { const file = path.join(stateDir, 'data/models', safe(repo), '.sceneworks-download-complete.json'); return { root: stateDir, path: file, sha256: await fileSha256(file) }; };
  const p = inv('flux3_image_klein_pup'), pup = { root: p.root, containmentRoot: hfHome, repository: p.repository, resolvedRevision: p.resolvedRevision, authority: AUTH, files: plain(p.files), installationReceipt: await receipt(p.repository) };
  const dense = async id => { const i = inv(id); return { root: i.root, containmentRoot: hfHome, repository: i.repository, resolvedRevision: i.resolvedRevision, authority: AUTH, files: plain(i.files), installationReceipt: await receipt(i.repository), components: { prompt_upsampler: pup } }; };
  const external = i => [{ role: 'text_encoder', root: path.join(i.root, 'text_encoder'), containmentRoot: hfHome, repository: i.repository, resolvedRevision: i.resolvedRevision, authority: AUTH, files: strip(i.files, 'text_encoder/') },
    { role: 'autoencoder', root: i.root, containmentRoot: hfHome, repository: i.repository, resolvedRevision: i.resolvedRevision, authority: AUTH, files: plain(i.files.filter(f => f.path === 'ae.safetensors')) }];
  const packages = { 'flux3_klein:mlx:bf16': await dense('flux3_klein'), 'flux3_klein_base:mlx:bf16': await dense('flux3_klein_base') };
  for (const [tier, pkg] of Object.entries(converted?.packages ?? {})) { const i = inv('flux3_klein'); packages[`flux3_klein:mlx:${tier}`] = { root: pkg.root, containmentRoot: pkg.root, repository: i.repository, resolvedRevision: i.resolvedRevision, authority: AUTH, files: pkg.files, externalComponents: external(i), components: { prompt_upsampler: pup } }; }
  const spec = { schemaVersion: 1, backend: 'mlx', packages, images: {} };
  if (host === DATASET_HOST) {
    const manifest = JSON.parse(await readFile(path.join(inputsRoot, 'inputs-manifest.json'), 'utf8')), items = [];
    for (const r of manifest.references) spec.images[`reference-${r.index}`] = { root: inputsRoot, path: `references/${r.file}`, sha256: r.sha256 };
    for (const t of manifest.training) { spec.images[`training-${t.index}`] = { root: inputsRoot, path: `training/${t.file}`, sha256: t.sha256 }; items.push({ binding: `training-${t.index}`, caption: (await readFile(path.join(inputsRoot, 'training', t.file.replace(/\.png$/, '.txt')), 'utf8')).trim() }); }
    spec.training = { captioned: { steps: 4, saveEvery: 1, rank: 4, items } };
  }
  return spec;
}
async function prepare(root, dispatch, control) {
  for (const key of ['userBuildApprovalId', 'resourceAdmissionId', 'userProvisionApprovalId', 'approvedBy']) assert(typeof control[key] === 'string' && control[key], `control ${key} required`);
  assert(path.isAbsolute(control.hfHome ?? '') && Array.isArray(control.modelIds) && control.modelIds.length, 'control hfHome/modelIds required');
  const inference = await inferenceCheckout(root, dispatch.inferenceMainSha), tools = await toolchain();
  const api = await build(root, dispatch, control, 'api-worker', inference, tools);
  await build(root, dispatch, control, 'resource-adapter', inference, tools);
  const prep = await preparation(root, dispatch, control, api, inference), report = await readJson(prep.reportFile);
  const converted = dispatch.host === DATASET_HOST ? null : await conversions(root, dispatch, report);
  const specFile = state(root, dispatch, 'input-spec');
  if (!await exists(specFile)) await writeFile(specFile, JSON.stringify(await hostInputSpec({ report, reportFile: prep.reportFile, hfHome: control.hfHome, host: dispatch.host, converted, inputsRoot: path.join(root, 'inputs') })), { flag: 'wx' });
  say({ action: 'input-spec', inputSpecSha256: digest(await readFile(specFile)) });
}
async function resourceInputRoot(root, dispatch, approval, report) {
  const target = path.join(root, 'resource-inputs', dispatch.host); await mkdir(target, { recursive: true });
  if (dispatch.host !== DATASET_HOST) return target; // anchors only: no supplemental inputs
  const row = approval.plan.resourceCases.find(r => r.backend === 'mlx' && r.kind === 'resource_supplemental'), pupRoot = report.inventories.find(i => i.modelId === 'flux3_image_klein_pup').root;
  for (const r of row.resourceCase.references) { const to = path.join(target, r.path); await mkdir(path.dirname(to), { recursive: true }); if (!await exists(to)) await copyFile(path.join(root, 'inputs', r.path), to); if (await fileSha256(to) !== r.sha256) throw blocked('staged resource reference differs'); }
  const pup = row.resourceCase.components.prompt_upsampler;
  for (const f of pup.inventory) { const to = path.join(target, pup.root, f.path); await mkdir(path.dirname(to), { recursive: true }); if (!await exists(to)) await copyFile(path.join(pupRoot, f.path), to); if (await fileSha256(to) !== f.sha256) throw blocked('staged PUP file differs'); }
  return target;
}
async function selfTest(root, dispatch, control) {
  const approval = await pinned(root, dispatch, control.approvalFile, control.approvalSha256);
  const { validateApprovedPlan, planHash } = await import('./flux3-image-klein-terminal-budget.mjs'); validateApprovedPlan(approval);
  const resource = await readJson(state(root, dispatch, 'resource-adapter-build')), api = await readJson(state(root, dispatch, 'api-worker-build'));
  const report = await readJson((await readJson(state(root, dispatch, 'preparation'))).reportFile);
  const planFile = state(root, dispatch, `self-test-plan-${planHash(approval).slice(0, 12)}`); if (!await exists(planFile)) await writeFile(planFile, JSON.stringify(approval.plan), { flag: 'wx' });
  const { runNativeBoundarySelfTest } = await import('./flux3-native-boundary-self-test.mjs');
  const result = await runNativeBoundarySelfTest({ lane: 'mlx', host: dispatch.host, binary: resource.receipt.binary.path, buildReceipt: { root: path.dirname(resource.receiptPath), path: resource.receiptPath, sha256: resource.receiptSha256 },
    sourceSha: dispatch.sourceSha, inferenceMainSha: dispatch.inferenceMainSha, planFile, resourceInputRoot: await resourceInputRoot(root, dispatch, approval, report) });
  const material = { kind: 'flux3-terminal-host-addendum-material', host: dispatch.host, approvedPlanSha256: planHash(approval), inputSpecSha256: digest(await readFile(state(root, dispatch, 'input-spec'))),
    buildReceipt: { receiptSha256: api.binding.receiptSha256, binarySha256: api.binding.binarySha256 }, resourceBuildReceipt: { receiptSha256: resource.binding.receiptSha256, binarySha256: resource.binding.binarySha256 },
    nativeBoundaryReceiptSha256: result.receiptSha256, nativeBoundaryReceipt: result.receipt };
  await writeFile(state(root, dispatch, `addendum-material-${planHash(approval).slice(0, 12)}`), JSON.stringify(material, null, 2) + '\n', { flag: 'wx' });
  say(material); // hashes, IDs and digests only: the receipt carries no input content
}
async function run(root, dispatch, control) {
  const approval = await pinned(root, dispatch, control.approvalFile, control.approvalSha256);
  const addendum = control.addendumFile ? await pinned(root, dispatch, control.addendumFile, control.addendumSha256) : undefined;
  await laneAddendum(approval, addendum, dispatch.host);
  const budget = await import('./flux3-image-klein-terminal-budget.mjs'), session = await import('./flux3-image-klein-terminal-session.mjs');
  const lane = budget.ledgerLane(approval, 'mlx', dispatch.host), ledgerRoot = path.join(root, 'ledgers', 'mlx');
  assert.equal(path.resolve(approval.ledgerRoots[lane]), path.resolve(ledgerRoot), 'approved lane ledger root must be this host\'s fixed root');
  if (!await exists(ledgerRoot)) say({ action: 'init-ledger', ...await budget.initializeLedger({ approval, backend: 'mlx', host: dispatch.host, ledgerRoot }) });
  const specFile = state(root, dispatch, 'input-spec'), inputSpec = await readJson(specFile);
  assert.equal(digest(await readFile(specFile)), addendum.inputSpecSha256, 'host input spec differs from the recorded addendum');
  const api = await readJson(state(root, dispatch, 'api-worker-build')), resource = await readJson(state(root, dispatch, 'resource-adapter-build'));
  const report = await readJson((await readJson(state(root, dispatch, 'preparation'))).reportFile), inference = await inferenceCheckout(root, dispatch.inferenceMainSha);
  const account = userInfo().username, hostName = hostname().split('.')[0], outputRoot = path.join(root, 'output'); await mkdir(outputRoot, { recursive: true });
  // Coordinator-attested exclusive Metal workload, observed now: the guard refuses any later process.
  const table = await session.observeProcessTable((c, a, o) => execFile(c, a, o), process.platform), ancestors = session.processAncestors(table, process.pid);
  const deadline = Date.now() + control.timeoutSeconds * 1000 * (control.sequence.length + 1);
  const lease = { kind: 'flux3-secondary-mac-exclusive-workload-lease', coordinatorApprovalId: control.coordinatorLeaseApprovalId, approvedBy: control.approvedBy, exclusiveGpuWorkload: control.exclusiveGpuWorkload === true, noForeignGpuWorkload: control.noForeignGpuWorkload === true,
    admissionId: control.admissionId, sourceSha: dispatch.sourceSha, inferenceMainSha: dispatch.inferenceMainSha, runnerAccount: account, hostName, observedAt: new Date().toISOString(), expiresAt: new Date(deadline).toISOString(),
    ancestorRunnerPids: table.filter(p => /^Runner\.Worker$/.test(path.basename(p.command)) && ancestors.has(p.pid)).map(p => p.pid), baselineProcesses: table };
  const leaseRoot = path.join(root, 'leases'); await mkdir(leaseRoot, { recursive: true });
  const leaseFile = path.join(leaseRoot, `${control.admissionId}-${Date.now()}.json`), leaseBytes = Buffer.from(JSON.stringify(lease)); await writeFile(leaseFile, leaseBytes, { flag: 'wx' });
  const registry = (await import('./flux3-image-klein-terminal-plan.mjs')).targetRegistry(await (await import('./flux3-image-klein-terminal-plan.mjs')).loadProductTargets());
  const present = {}, conversionOutputBytes = {};
  for (const t of registry.filter(t => t.backend === 'mlx')) for (const d of t.model.downloads) { if (report.inventories.some(i => i.repository === d.repo && i.resolvedRevision === d.revision)) present[`${d.repo}@${d.revision}`] = d.estimatedSizeBytes; if (d.localDerivation) conversionOutputBytes[`${t.modelId}:mlx:${d.variant}`] = Math.max(d.localDerivation.weightsBytes, d.footprint.diskSizeBytes - d.estimatedSizeBytes); }
  const fs = await statfs(outputRoot, { bigint: true });
  const base = { actualRun: true, backend: 'mlx', admissionId: control.admissionId, approvedBy: control.approvedBy, phases: approval.plan.phases, runner: dispatch.host, runnerTarget: dispatch.runnerTarget, runnerLabel: dispatch.runnerLabel,
    runnerAccount: account, hostName, sourceSha: dispatch.sourceSha, inferenceMainSha: dispatch.inferenceMainSha, inferenceRepository: inference.repo, inferenceMainTipSha: inference.tip, outputRoot, hfHome: control.hfHome,
    timeoutSeconds: control.timeoutSeconds, approvedPlanSha256: budget.planHash(approval), ledgerId: approval.ledgerId, ledgerRoot, inputSpecSha256: addendum.inputSpecSha256,
    disk: { freeBytes: Number(fs.bavail * fs.bsize), present, conversionOutputBytes }, trainingLicenseAcknowledged: control.trainingLicenseAcknowledged === true,
    macWorkloadLease: { root: leaseRoot, path: leaseFile, sha256: digest(leaseBytes) } };
  const sealed = b => ({ binary: b.receipt.binary.path, binarySha256: b.receipt.binary.sha256, binarySourceSha: dispatch.sourceSha, binaryBuildReceipt: { root: path.dirname(b.receiptPath), path: b.receiptPath, sha256: b.receiptSha256 } });
  const work = path.join(root, 'run', budget.planHash(approval).slice(0, 12)); await mkdir(work, { recursive: true });
  const write = async (name, value) => { const file = path.join(work, `${Date.now()}-${name}.json`); await writeFile(file, JSON.stringify(value), { flag: 'wx' }); return file; };
  const approvalFile = await write('approval', approval), addendumFile = await write('addendum', addendum), specCopy = await write('input-spec', inputSpec);
  const { productionResourceRequest } = await import('./flux3-native-boundary-self-test.mjs'), calibrationPlan = await readJson(path.join(ROOT, 'config/memory-calibration-plan.json'));
  const inv = id => report.inventories.find(i => i.modelId === id), conversionsState = dispatch.host === DATASET_HOST ? null : await readJson(state(root, dispatch, 'conversions'));
  // Ordered, explicit groups; stop at the first failure. A failed permit stays spent: no retry.
  for (const [index, entry] of control.sequence.entries()) {
    if (entry.caseIds) {
      const admissionFile = await write(`admission-${index}`, { ...base, ...sealed(api) }), casesFile = await write(`cases-${index}`, entry.caseIds);
      await child(root, `run-${index}`, node, [driver, 'run', admissionFile, specCopy, casesFile, approvalFile, addendumFile]);
    } else {
      const invocation = approval.plan.generationInvocations.find(i => i.invocationId === entry.resourceInvocationId); assert(invocation?.kind?.startsWith('resource_'), 'resource sequence entries name a resource invocation');
      const dense = inv(invocation.modelId), packageRoot = invocation.tier === 'bf16' ? dense.root : conversionsState.packages[invocation.tier].root;
      const launch = { admission: { ...base, ...sealed(resource), approvedPlan: approval, inputSpecAddendum: addendum }, request: productionResourceRequest(calibrationPlan, approval.plan, invocation), invocationId: invocation.invocationId,
        resourceEnvironment: { SCENEWORKS_FLUX3_KLEIN_ROOT: packageRoot, SCENEWORKS_FLUX3_KLEIN_ORIGINAL_ROOT: dense.root, SCENEWORKS_FLUX3_KLEIN_REPOSITORY: dense.repository, SCENEWORKS_FLUX3_KLEIN_REVISION: dense.resolvedRevision,
          SCENEWORKS_FLUX3_RESOURCE_INPUT_ROOT: await resourceInputRoot(root, dispatch, approval, report) } };
      await child(root, `resource-${index}`, node, [path.join(ROOT, 'scripts/flux3-resource-launcher.mjs'), await write(`resource-${index}`, launch)]);
    }
  }
  say({ action: 'ledger', ...(({ state: s }) => ({ sequence: s.sequence, generationCount: s.generationCount, headSha256: s.headSha256 }))(await budget.auditLedger({ approval, backend: 'mlx', host: dispatch.host, ledgerRoot, inputSpecAddendum: addendum })) });
}
export async function main(env = process.env) {
  const dispatch = resolveDispatch(env);
  if (env.FLUX3_TERMINAL_GATE_ONLY === '1') { say({ gate: 'ok', host: dispatch.host, step: dispatch.step }); return; }
  const root = hostRoot(dispatch, env);
  assert.equal((await execFile('git', ['rev-parse', 'HEAD'], { cwd: ROOT })).stdout.trim(), dispatch.sourceSha, 'checkout differs from the dispatched source');
  const control = await readControl(root, dispatch);
  // The dataset host verifies every personal input before anything else, on every step.
  if (dispatch.host === DATASET_HOST) say({ action: 'inputs-verified', ...await verifyInputsManifest(path.join(root, 'inputs')) });
  await mkdir(path.join(root, 'state'), { recursive: true });
  await ({ prepare, 'self-test': selfTest, run })[dispatch.step](root, dispatch, control);
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) main().catch(error => { console.error(error.message.split('\n')[0].slice(0, 500)); process.exitCode = 1; });
