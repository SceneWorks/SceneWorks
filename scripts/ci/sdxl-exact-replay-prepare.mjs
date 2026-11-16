// SW-7148. Host discovery only: no API, worker, CUDA context, weights or model execution.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdir, readdir, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fileSha256 } from '../lib/file-sha256.mjs';
import { inferenceRevision } from '../lib/inference-provenance.mjs';
import { REQUEST, digest } from './sdxl-exact-replay-proof.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
export function validatePreparationDispatch(raw) {
  assert(raw?.startsWith('{"proof":"sdxl-exact-replay",'), 'canonical SDXL marker required');
  const value = JSON.parse(raw);
  assert.deepEqual(Object.keys(value).sort(), ['ownerGo', 'phase', 'proof'], 'unexpected preparation input');
  assert.equal(value.proof, 'sdxl-exact-replay');
  assert.equal(value.phase, 'prepare');
  assert.equal(value.ownerGo, false, 'metadata preparation cannot carry model approval');
  return value;
}

async function main() {
  assert.equal(process.platform, 'win32', 'Windows runner required');
  assert.equal(process.env.GITHUB_EVENT_NAME, 'workflow_dispatch', 'manual dispatch required');
  validatePreparationDispatch(process.env.SDXL_REPLAY_DISPATCH);
  const sourceSha = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim();
  assert.equal(sourceSha, process.env.GITHUB_SHA, 'checkout changed');
  assert.equal(execFileSync('git', ['status', '--porcelain'], { cwd: ROOT, encoding: 'utf8' }).trim(), '', 'checkout dirty');
  const inferencePin = await inferenceRevision(ROOT);
  const hub = path.resolve(process.env.HF_HUB_CACHE?.trim() || process.env.HUGGINGFACE_HUB_CACHE?.trim() || 'E:\\huggingface\\hub');
  const hubExists = (await stat(hub).catch(() => null))?.isDirectory() === true;
  const allCacheEntries = hubExists ? (await readdir(hub)).filter((name) => /^models--/i.test(name) && /sdxl|clip|vae/i.test(name)).sort() : [];
  const cacheEntries = allCacheEntries.slice(0, 50);
  const gpuLines = execFileSync('nvidia-smi', ['--query-gpu=index,uuid,name,driver_version,pci.bus_id', '--format=csv,noheader,nounits'], { encoding: 'utf8' }).trim().split(/\r?\n/);
  const devices = gpuLines.map((line) => {
    const cells = line.split(',').map((value) => value.trim());
    assert.equal(cells.length, 5, 'unexpected NVIDIA metadata');
    const [index, uuid, name, driverVersion, busId] = cells;
    assert(/^\d+$/.test(index) && /^GPU-[a-f0-9-]{36}$/i.test(uuid), 'invalid physical device index or UUID');
    return { physicalIndex: Number(index), uuid, name, driverVersion, busId };
  });
  assert(devices.length > 0 && new Set(devices.map((device) => device.uuid)).size === devices.length, 'ambiguous GPUs');
  const driver = path.join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'nvcuda.dll');
  const driverSha256 = await fileSha256(driver);
  assert(process.env.LOCALAPPDATA && path.win32.isAbsolute(process.env.LOCALAPPDATA), 'private user profile missing');
  assert(process.env.SDXL_REPLAY_PREPARATION_REPORT && path.win32.isAbsolute(process.env.SDXL_REPLAY_PREPARATION_REPORT), 'owned report path missing');
  const privateDir = path.join(process.env.LOCALAPPDATA, 'SceneWorks', 'sdxl-exact-replay');
  await mkdir(privateDir, { recursive: true });
  const preparationPath = path.join(privateDir, `preparation-${sourceSha}-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}.json`);
  const preparation = { version: 0, kind: 'metadata-only-preparation', sourceSha, inferenceRevision: inferencePin,
    requestSha256: digest(REQUEST), hub, hubExists, cacheEntries, cacheEntryCount: allCacheEntries.length,
    cacheEntriesTruncated: allCacheEntries.length > cacheEntries.length, devices, driver: { path: driver, sha256: driverSha256 },
    runnerName: process.env.RUNNER_NAME, ownerGo: false,
    missing: ['serverIssuedSourceBinding', 'selectedMemberHashes', 'providerSchedule', 'loadedRuntimeDllCensus', 'ownerGpuWindow'] };
  await writeFile(preparationPath, JSON.stringify(preparation, null, 2) + '\n', { flag: 'wx', mode: 0o600 });
  const report = { version: 1, kind: 'metadata-only-preparation', sourceSha, inferenceRevision: inferencePin,
    requestSha256: preparation.requestSha256, runnerName: preparation.runnerName, preparationPath,
    hubExists, candidateCacheEntries: cacheEntries, candidateCacheEntryCount: allCacheEntries.length,
    candidateCacheEntriesTruncated: preparation.cacheEntriesTruncated, devices, driverSha256, missing: preparation.missing };
  await writeFile(process.env.SDXL_REPLAY_PREPARATION_REPORT, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(() => { console.error('SDXL metadata preparation failed. No model was started.'); process.exitCode = 1; });
}
