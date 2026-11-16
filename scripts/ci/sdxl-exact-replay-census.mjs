#!/usr/bin/env node
// Metadata-only idle-worker census. No job is created, so no weights can be loaded.
import assert from 'node:assert/strict';
import { createHmac, randomBytes } from 'node:crypto';
import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdir, open, readFile, realpath, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fileSha256 } from '../lib/file-sha256.mjs';
import { inferenceRevision } from '../lib/inference-provenance.mjs';
import { REQUEST, digest } from './sdxl-exact-replay-proof.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const PIN = '101975c301686467f089cf8c0e3c4a9e33846b6d';
let stage = 'dispatch';
const publicReasons = new Set([
  'binding escaped private directory', 'inference pin changed', 'bound runtime source changed',
  'bound API binary changed', 'physical host changed', 'physical index zero unavailable',
  'driver bytes changed', 'isolated API has a worker', 'API did not become healthy',
  'API exited', 'worker exited', 'idle worker did not register',
  'census cannot run queued jobs', 'census created a job', 'CUDA driver not loaded',
  'duplicate runtime library names', 'isolated runtime did not drain',
]);

export function validateCensusDispatch(raw) {
  assert(typeof raw === 'string' && raw.startsWith('{"proof":"sdxl-exact-replay",'), 'canonical census marker required');
  const input = JSON.parse(raw);
  assert.deepEqual(Object.keys(input).sort(), ['bindingPath', 'ownerGo', 'phase', 'proof'], 'unknown census fields');
  assert.equal(input.proof, 'sdxl-exact-replay');
  assert.equal(input.phase, 'census');
  assert.equal(input.ownerGo, false, 'census cannot inherit model approval');
  assert(typeof input.bindingPath === 'string' && path.win32.isAbsolute(input.bindingPath), 'private binding path required');
  return input;
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
  assert.equal(process.platform, 'win32', 'Windows census only');
  assert.equal(process.env.GITHUB_EVENT_NAME, 'workflow_dispatch', 'manual dispatch required');
  const input = validateCensusDispatch(process.env.SDXL_REPLAY_DISPATCH);
  assert(process.env.SDXL_REPLAY_CENSUS_REPORT && path.win32.isAbsolute(process.env.SDXL_REPLAY_CENSUS_REPORT), 'owned report path missing');
  const sourceSha = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim();
  assert.equal(sourceSha, process.env.GITHUB_SHA, 'source changed');
  assert.equal(execFileSync('git', ['status', '--porcelain'], { cwd: ROOT, encoding: 'utf8' }).trim(), '', 'checkout dirty');
  const pin = await inferenceRevision(ROOT);
  assert.equal(pin, PIN, 'inference pin changed');

  stage = 'binding-path';
  const privateDir = await realpath(path.join(process.env.LOCALAPPDATA, 'SceneWorks', 'sdxl-exact-replay'));
  const bindingPath = await realpath(input.bindingPath);
  assert.equal(path.dirname(bindingPath).toLowerCase(), privateDir.toLowerCase(), 'binding escaped private directory');
  stage = 'binding-contract';
  const binding = JSON.parse(await readFile(bindingPath, 'utf8'));
  assert.equal(binding.version, 0);
  assert.equal(binding.kind, 'metadata-only-binding');
  assert.equal(binding.ownerGo, false);
  assert.equal(binding.inferenceRevision, pin);
  assert.equal(binding.requestSha256, digest(REQUEST));
  assert.equal(binding.missing.includes('loadedRuntimeDllCensus'), true);
  stage = 'binding-source';
  // Only proof/workflow files may change after the server issued this source binding.
  const runtimeInputs = ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', '.cargo', 'vendor',
    'crates', 'apps', 'packages', 'config', 'docs/generated', 'docs/calibration', 'docker'];
  const sourceDiff = spawnSync('git', ['diff', '--quiet', binding.sourceSha, sourceSha, '--', ...runtimeInputs], { cwd: ROOT });
  assert.equal(sourceDiff.status, 0, 'bound runtime source changed');
  stage = 'binding-binary';
  const binary = path.resolve(process.env.SDXL_REPLAY_BINARY);
  assert.equal(await fileSha256(binary), binding.binarySha256, 'bound API binary changed');
  stage = 'binding-device';
  const devices = execFileSync('nvidia-smi', ['--query-gpu=index,uuid', '--format=csv,noheader'], { encoding: 'utf8' })
    .trim().split(/\r?\n/).map((line) => {
      const [index, uuid] = line.split(',').map((value) => value.trim());
      return { physicalIndex: Number(index), uuid };
    });
  assert.deepEqual(devices, binding.devices.map(({ physicalIndex, uuid }) => ({ physicalIndex, uuid })), 'physical host changed');
  const device = devices[0];
  assert(device && device.physicalIndex === 0, 'physical index zero unavailable');
  stage = 'binding-driver';
  assert.equal(await fileSha256(binding.driver.path), binding.driver.sha256, 'driver bytes changed');

  const state = path.join(process.env.RUNNER_TEMP, 'sdxl-exact-replay-census-' + process.env.GITHUB_RUN_ID + '-' + process.env.GITHUB_RUN_ATTEMPT);
  await mkdir(state);
  const log = await open(path.join(privateDir, 'census-' + process.env.GITHUB_RUN_ID + '-' + process.env.GITHUB_RUN_ATTEMPT + '.log'), 'wx');
  const port = await freePort();
  const workerId = 'sdxl-census-' + process.env.GITHUB_RUN_ID + '-' + process.env.GITHUB_RUN_ATTEMPT;
  const master = randomBytes(32).toString('hex');
  const workerKey = createHmac('sha256', Buffer.from(master, 'utf8')).update(workerId).digest('hex');
  const inherited = Object.fromEntries(Object.entries(process.env).filter(([key]) =>
    !key.startsWith('SCENEWORKS_') && !/TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY/.test(key)));
  const env = { ...inherited, SCENEWORKS_DATA_DIR: path.join(state, 'data'),
    SCENEWORKS_CONFIG_DIR: path.join(state, 'config'), SCENEWORKS_JOBS_DB_PATH: path.join(state, 'jobs.db'),
    SCENEWORKS_CREDENTIALS_DIR: path.join(state, 'credentials'), SCENEWORKS_WORKER_PROOF_MASTER: master,
    SCENEWORKS_WORKER_CHILD: '1', SCENEWORKS_PARENT_PID: String(process.pid), SCENEWORKS_TRUST_LOOPBACK: '1',
    SCENEWORKS_API_HOST: '127.0.0.1', SCENEWORKS_WORKER_ID: workerId,
    SCENEWORKS_GPU_ID: String(device.physicalIndex), SCENEWORKS_RUST_WORKER_GPU_ID: String(device.physicalIndex),
    SCENEWORKS_BACKEND_CANDLE_ENABLED: 'true', SCENEWORKS_BACKEND_MLX_ENABLED: 'false',
    SCENEWORKS_POLL_SECONDS: '1', SCENEWORKS_HEARTBEAT_SECONDS: '1',
    HF_HUB_CACHE: binding.hub, HUGGINGFACE_HUB_CACHE: binding.hub, HF_HOME: path.dirname(binding.hub),
    HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', CUDA_VISIBLE_DEVICES: device.uuid };
  const children = [];
  const launch = (overrides) => {
    const child = spawn(binary, [], { cwd: ROOT, env: { ...env, ...overrides }, stdio: ['ignore', log.fd, log.fd] });
    child.spawnFailed = false;
    child.on('error', () => { child.spawnFailed = true; });
    child.done = new Promise((resolve) => { child.once('exit', resolve); child.once('error', resolve); });
    children.push(child);
    return child;
  };
  const api = async (route) => {
    const response = await fetch('http://127.0.0.1:' + port + route, { signal: AbortSignal.timeout(10000) });
    assert(response.ok, 'isolated API refusal');
    return response.json();
  };
  let result;
  try {
    stage = 'api-start';
    const apiChild = launch({ SCENEWORKS_WORKER_ONLY: '0', SCENEWORKS_API_PORT: String(port),
      SCENEWORKS_RUN_UTILITY_INPROCESS: '0' });
    let healthy = false;
    for (let attempt = 0; attempt < 240; attempt++) {
      assert(!apiChild.spawnFailed && apiChild.exitCode === null && apiChild.signalCode === null, 'API exited');
      try { healthy = (await api('/api/v1/health')).status === 'ok'; } catch { /* startup */ }
      if (healthy) break;
      await delay(500);
    }
    assert(healthy, 'API did not become healthy');
    assert.deepEqual(await api('/api/v1/workers'), [], 'isolated API has a worker');
    const { DatabaseSync } = await import('node:sqlite');
    const db = new DatabaseSync(env.SCENEWORKS_JOBS_DB_PATH, { readOnly: true });
    try {
      assert.equal(db.prepare('select count(*) as n from jobs').get().n, 0, 'census cannot run queued jobs');
      stage = 'idle-worker';
      const worker = launch({ SCENEWORKS_WORKER_ONLY: '1',
        SCENEWORKS_API_URL: 'http://127.0.0.1:' + port, SCENEWORKS_WORKER_PROOF_KEY: workerKey });
      let registered = false;
      for (let attempt = 0; attempt < 360; attempt++) {
        assert(!worker.spawnFailed && worker.exitCode === null && worker.signalCode === null, 'worker exited');
        const workers = await api('/api/v1/workers');
        const owned = workers.find((entry) => entry.id === workerId);
        registered = owned?.status === 'idle' && owned.gpuId === String(device.physicalIndex)
          && db.prepare('select source_proof_authenticated from workers where id=?').get(workerId)?.source_proof_authenticated === 1;
        if (registered) break;
        await delay(500);
      }
      assert(registered, 'idle worker did not register');
      assert.equal(db.prepare('select count(*) as n from jobs').get().n, 0, 'census created a job');
      stage = 'runtime-modules';
      const command = '(Get-Process -Id ' + worker.pid + ').Modules | Select-Object -ExpandProperty FileName | ConvertTo-Json -Compress';
      const modules = [JSON.parse(execFileSync('powershell', ['-NoProfile', '-NonInteractive', '-Command', command], { encoding: 'utf8' }))].flat();
      const selected = modules.filter((file) => /^(?:nv|cuda|cublas|cudnn|cufft|curand|cusolver|cusparse|vcruntime|msvcp).*\.dll$/i.test(path.basename(file)));
      assert(selected.some((file) => /^nvcuda/i.test(path.basename(file))), 'CUDA driver not loaded');
      const runtimeFiles = await Promise.all(selected.map(async (file) => ({ path: file, sha256: await fileSha256(file) })));
      const names = runtimeFiles.map((file) => path.basename(file.path).toLowerCase());
      assert.equal(new Set(names).size, names.length, 'duplicate runtime library names');
      assert.equal(db.prepare('select count(*) as n from jobs').get().n, 0, 'census created a job');
      const publicFiles = runtimeFiles.map((file) => ({ name: path.basename(file.path).toLowerCase(), sha256: file.sha256 }))
        .sort((a, b) => a.name.localeCompare(b.name));
      const censusPath = path.join(privateDir, 'census-' + sourceSha + '-' + process.env.GITHUB_RUN_ID + '-' + process.env.GITHUB_RUN_ATTEMPT + '.json');
      const record = { version: 0, kind: 'metadata-only-runtime-census', sourceSha,
        bindingSourceSha: binding.sourceSha, bindingPath, bindingSha256: await fileSha256(bindingPath),
        inferenceRevision: pin, requestSha256: binding.requestSha256, hub: binding.hub,
        sourceIdentitySha256: binding.sourceIdentitySha256, scheduleSha256: binding.scheduleSha256,
        files: binding.files, binarySha256: binding.binarySha256, driver: binding.driver,
        deviceUuid: device.uuid, physicalIndex: device.physicalIndex, runtimeFiles,
        ownerGo: false, missing: ['ownerGpuWindow'] };
      const report = { version: 1, kind: 'metadata-only-runtime-census', sourceSha,
        inferenceRevision: pin, censusPath, deviceUuid: device.uuid, physicalIndex: device.physicalIndex,
        selectedFileCount: binding.files.length, runtimeLibraryCount: publicFiles.length,
        runtimeLibrariesSha256: digest(publicFiles), binarySha256: binding.binarySha256, missing: record.missing };
      result = { censusPath, record, report };
    } finally { db.close(); }
  } finally {
    for (const child of children.reverse()) child.kill();
    const drained = await Promise.all(children.map((child) => Promise.race([child.done.then(() => true), delay(10000).then(() => false)])));
    await log.close();
    await rm(state, { recursive: true, force: true });
    assert(drained.every(Boolean), 'isolated runtime did not drain');
  }
  stage = 'write';
  await writeFile(result.censusPath, JSON.stringify(result.record, null, 2) + '\n', { flag: 'wx', mode: 0o600 });
  await writeFile(process.env.SDXL_REPLAY_CENSUS_REPORT, JSON.stringify(result.report, null, 2) + '\n', { flag: 'wx' });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    const reason = publicReasons.has(error?.message) ? ' (' + error.message + ')' : '';
    const fileError = stage === 'binding-binary' && ['ENOENT', 'EACCES', 'EPERM', 'EBUSY'].includes(error?.code)
      ? ` (${error.code})` : '';
    console.error('SDXL runtime census failed at ' + stage + (reason || fileError) + '. No job or model was started.');
    process.exitCode = 1;
  });
}
