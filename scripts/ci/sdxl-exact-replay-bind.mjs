// SW-7148. Bind a queued request on an isolated API; never start a worker or a model.
import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { createServer } from 'node:http';
import { mkdir, open, readFile, readdir, realpath, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fileSha256 } from '../lib/file-sha256.mjs';
import { inferenceRevision } from '../lib/inference-provenance.mjs';
import { REQUEST, digest, sourceIdentity, verifyMembers } from './sdxl-exact-replay-proof.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const PIN = '886b915c2830e64d5887297e3b797c816ada0e38';
// Derived from AlphaSchedule/DiscreteModelSampling/Normal 20-step source and
// NumericScheduleDigest::f32_values, then canonicalized by the proof harness.
// AlphaSchedule, DiscreteModelSampling, Normal scheduling and the SDXL provider
// remain unchanged from b387b95a to this pin. The proof also compares the
// completed runtime schedule to this digest, so later numerical drift refuses.
const SCHEDULE_SHA256 = 'e6c4fabcef6944e589759bd092184dc47baa5786c50cd92870c695bd36092289';
let stage = 'dispatch';
let failureHint = '';

export function validateBindingDispatch(raw) {
  assert(raw?.startsWith('{"proof":"sdxl-exact-replay",'), 'canonical SDXL marker required');
  const value = JSON.parse(raw);
  assert.deepEqual(Object.keys(value).sort(), ['ownerGo', 'phase', 'preparationPath', 'proof'], 'unexpected binding input');
  assert.equal(value.proof, 'sdxl-exact-replay');
  assert.equal(value.phase, 'bind');
  assert.equal(value.ownerGo, false, 'source binding cannot carry model approval');
  assert(typeof value.preparationPath === 'string' && path.win32.isAbsolute(value.preparationPath), 'private preparation path required');
  return value;
}

async function candidateRoots(hub, names) {
  const roots = [];
  for (const name of names) {
    assert(/^models--[^\\/]+$/i.test(name), 'unexpected cache entry');
    const snapshots = path.join(hub, name, 'snapshots');
    if (!(await stat(snapshots).catch(() => null))?.isDirectory()) continue;
    for (const revision of await readdir(snapshots)) {
      const root = path.join(snapshots, revision);
      if ((await stat(root).catch(() => null))?.isDirectory()) roots.push(root);
    }
  }
  return roots.sort();
}

export async function mapSelectedFiles(route, hub, roots) {
  const files = [];
  for (const model of route.models) for (const [memberIndex, member] of model.members.entries()) {
    const matches = [];
    for (const root of roots) {
      const candidate = [];
      for (const file of member.files) {
        const relative = file.relativePath.replaceAll('\\', '/');
        assert(!path.win32.isAbsolute(relative) && !relative.startsWith('/') &&
          relative.split('/').every((part) => part && part !== '.' && part !== '..'), 'unsafe bound member path');
        const selected = path.join(root, ...relative.split('/'));
        const info = await stat(selected).catch(() => null);
        if (!info?.isFile() || info.size !== file.sizeBytes || `sha256:${await fileSha256(selected)}` !== file.sha256) {
          candidate.length = 0; break;
        }
        candidate.push({ slot: model.slot, memberIndex, relativePath: file.relativePath, path: selected });
      }
      if (candidate.length === member.files.length) matches.push(candidate);
    }
    assert(matches.length > 0, 'selected member absent from offline cache');
    files.push(...matches[0]);
  }
  await verifyMembers(route, { hub, files, sourceIdentitySha256: digest(sourceIdentity(route)) });
  return files;
}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function freePort() {
  const server = createServer();
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  await new Promise((resolve) => server.close(resolve));
  return port;
}

async function main() {
  assert.equal(process.platform, 'win32', 'Windows runner required');
  assert.equal(process.env.GITHUB_EVENT_NAME, 'workflow_dispatch', 'manual dispatch required');
  const input = validateBindingDispatch(process.env.SDXL_REPLAY_DISPATCH);
  assert(process.env.SDXL_REPLAY_BIND_REPORT && path.win32.isAbsolute(process.env.SDXL_REPLAY_BIND_REPORT), 'owned report path missing');
  const sourceSha = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim();
  assert.equal(sourceSha, process.env.GITHUB_SHA, 'source changed');
  assert.equal(execFileSync('git', ['status', '--porcelain'], { cwd: ROOT, encoding: 'utf8' }).trim(), '', 'checkout dirty');
  const pin = await inferenceRevision(ROOT);
  assert.equal(pin, PIN, 'schedule metadata requires a new pinned-source calculation');
  stage = 'preparation';
  const privateDir = path.join(process.env.LOCALAPPDATA, 'SceneWorks', 'sdxl-exact-replay');
  const trustedDir = await realpath(privateDir);
  const priorPath = await realpath(input.preparationPath);
  assert.equal(path.dirname(priorPath).toLowerCase(), trustedDir.toLowerCase(), 'preparation escaped private directory');
  const prior = JSON.parse(await readFile(priorPath, 'utf8'));
  assert.equal(prior.version, 0);
  assert.equal(prior.kind, 'metadata-only-preparation');
  assert.equal(prior.ownerGo, false);
  assert.equal(prior.inferenceRevision, pin);
  assert.equal(prior.requestSha256, digest(REQUEST));
  assert.equal(prior.hubExists, true, 'offline hub missing');
  assert.equal(prior.cacheEntriesTruncated, false, 'cache candidate inventory incomplete');
  assert.equal(await fileSha256(prior.driver.path), prior.driver.sha256, 'driver bytes changed');
  const currentDevices = execFileSync('nvidia-smi', ['--query-gpu=index,uuid', '--format=csv,noheader'], { encoding: 'utf8' })
    .trim().split(/\r?\n/).map((line) => {
      const [index, uuid] = line.split(',').map((value) => value.trim());
      return { physicalIndex: Number(index), uuid };
    });
  assert.deepEqual(currentDevices, prior.devices.map(({ physicalIndex, uuid }) => ({ physicalIndex, uuid })), 'physical host changed');
  const roots = await candidateRoots(prior.hub, prior.cacheEntries);
  assert(roots.length > 0, 'no cached snapshot roots');
  const binary = path.resolve(process.env.SDXL_REPLAY_BINARY);
  const binarySha256 = await fileSha256(binary);
  const state = path.join(process.env.RUNNER_TEMP, `sdxl-exact-replay-bind-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}`);
  await mkdir(state);
  const log = await open(path.join(privateDir, `api-bind-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}.log`), 'wx');
  const port = await freePort();
  const inherited = Object.fromEntries(Object.entries(process.env).filter(([key]) =>
    !key.startsWith('SCENEWORKS_') && !/TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY/.test(key)));
  const env = { ...inherited, SCENEWORKS_DATA_DIR: path.join(state, 'data'),
    SCENEWORKS_CONFIG_DIR: path.join(state, 'config'), SCENEWORKS_JOBS_DB_PATH: path.join(state, 'jobs.db'),
    SCENEWORKS_CREDENTIALS_DIR: path.join(state, 'credentials'), SCENEWORKS_WORKER_PROOF_MASTER: randomBytes(32).toString('hex'),
    SCENEWORKS_WORKER_CHILD: '1', SCENEWORKS_WORKER_ONLY: '0', SCENEWORKS_GPU_CHECK: '0',
    SCENEWORKS_PARENT_PID: String(process.pid), SCENEWORKS_TRUST_LOOPBACK: '1',
    SCENEWORKS_API_HOST: '127.0.0.1', SCENEWORKS_API_PORT: String(port), SCENEWORKS_RUN_UTILITY_INPROCESS: '0',
    SCENEWORKS_BACKEND_CANDLE_ENABLED: 'true', SCENEWORKS_BACKEND_MLX_ENABLED: 'false',
    HF_HUB_CACHE: prior.hub, HUGGINGFACE_HUB_CACHE: prior.hub, HF_HOME: path.dirname(prior.hub),
    HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1' };
  // WORKER_CHILD treats stdin EOF as shutdown on Windows; keep the child pipe open.
  const child = spawn(binary, [], { cwd: ROOT, env, stdio: ['pipe', log.fd, log.fd] });
  let spawnFailed = false;
  child.on('error', () => { spawnFailed = true; });
  const done = new Promise((resolve) => { child.once('exit', resolve); child.once('error', resolve); });
  const api = async (method, route, body) => {
    const response = await fetch(`http://127.0.0.1:${port}${route}`, { method,
      ...(body ? { headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) } : {}),
      signal: AbortSignal.timeout(10000) });
    failureHint = response.ok ? '' : `HTTP ${method} ${route} ${response.status}`;
    assert(response.ok, `HTTP ${method} ${response.status}`);
    return response.json();
  };
  let result;
  try {
    stage = 'api-start';
    let healthy = false;
    for (let attempt = 0; attempt < 240; attempt++) {
      assert(!spawnFailed && child.exitCode === null && child.signalCode === null, 'API exited');
      try { healthy = (await api('GET', '/api/v1/health')).status === 'ok'; } catch { /* startup */ }
      if (healthy) break;
      await delay(500);
    }
    assert(healthy, 'API did not become healthy');
    assert.deepEqual(await api('GET', '/api/v1/workers'), [], 'isolated API has a worker');
    stage = 'create-project';
    const project = await api('POST', '/api/v1/projects', { name: 'SDXL metadata-only source binding' });
    stage = 'admit-image';
    const accepted = await api('POST', '/api/v1/image/jobs', { ...REQUEST, projectId: project.id, requestedGpu: '0' });
    stage = 'open-job-db';
    const { DatabaseSync } = await import('node:sqlite');
    const db = new DatabaseSync(env.SCENEWORKS_JOBS_DB_PATH, { readOnly: true });
    let row;
    stage = 'query-job-row';
    try { row = db.prepare('select status, portable_model_binding_json from jobs where id=?').get(accepted.id); }
    finally { db.close(); }
    stage = 'check-queued-status';
    failureHint = row ? (['queued', 'pending_caption', 'failed', 'canceled'].includes(row.status) ? row.status : 'unexpected-status') : 'missing-row';
    assert.equal(row?.status, 'queued', 'request was not left queued');
    failureHint = '';
    stage = 'parse-binding';
    const binding = JSON.parse(row.portable_model_binding_json);
    stage = 'select-route';
    const routes = Object.values(binding.routes).filter((route) => route.backend === 'candle' && route.platform === 'windows');
    failureHint = `${routes.length} Windows Candle routes`;
    assert.equal(routes.length, 1, 'ambiguous server-issued route');
    failureHint = '';
    const route = routes[0];
    stage = 'verify-source-identity';
    let sourceIdentitySha256;
    try {
      sourceIdentitySha256 = digest(sourceIdentity(route));
    } catch (error) {
      const publicReasons = new Set([
        'backend fallback', 'unexpected model/adapter slots', 'model substitution',
        'model backend substitution', 'model slot substitution', 'provider substitution',
        'source members missing', 'adapter/optional source outside witness',
        'member bytes missing', 'invalid member digest',
      ]);
      failureHint = publicReasons.has(error?.message) ? error.message : 'invalid source identity';
      throw error;
    }
    stage = 'members';
    const files = await mapSelectedFiles(route, prior.hub, roots);
    assert.deepEqual(await api('GET', '/api/v1/workers'), [], 'worker appeared during binding');
    const bindingPath = path.join(privateDir, `binding-${sourceSha}-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}.json`);
    const record = { version: 0, kind: 'metadata-only-binding', sourceSha, inferenceRevision: pin,
      requestSha256: digest(REQUEST), preparationPath: priorPath, hub: prior.hub, devices: prior.devices,
      driver: prior.driver, sourceIdentitySha256, scheduleSha256: SCHEDULE_SHA256, files,
      binarySha256, ownerGo: false, missing: ['loadedRuntimeDllCensus', 'ownerGpuWindow'] };
    const report = { version: 1, kind: 'metadata-only-binding',
      sourceSha, inferenceRevision: pin, bindingPath, sourceIdentitySha256, scheduleSha256: SCHEDULE_SHA256,
      selectedFileCount: files.length, binarySha256, missing: record.missing };
    result = { bindingPath, record, report };
  } finally {
    child.kill();
    const drained = await Promise.race([done.then(() => true), delay(10000).then(() => false)]);
    await log.close();
    await rm(state, { recursive: true, force: true });
    assert(drained, 'isolated API did not drain');
  }
  stage = 'write';
  await writeFile(result.bindingPath, JSON.stringify(result.record, null, 2) + '\n', { flag: 'wx', mode: 0o600 });
  await writeFile(process.env.SDXL_REPLAY_BIND_REPORT, JSON.stringify(result.report, null, 2) + '\n', { flag: 'wx' });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(() => { console.error(`SDXL metadata binding failed at ${stage}${failureHint ? ` (${failureHint})` : ''}. No worker or model was started.`); process.exitCode = 1; });
}
