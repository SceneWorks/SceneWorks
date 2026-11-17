#!/usr/bin/env node
// Turn the private metadata census into a time-bounded plan only after owner go.
// This phase never starts an API, worker, job, CUDA context, or model.
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { realpath, readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fileSha256 } from '../lib/file-sha256.mjs';
import { inferenceRevision } from '../lib/inference-provenance.mjs';
import { RUNTIME_INPUTS, selectProofDevice } from './sdxl-exact-replay-census.mjs';
import { REQUEST, digest, validatePlan } from './sdxl-exact-replay-proof.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
let stage = 'dispatch';

export function validatePlanDispatch(raw) {
  assert(typeof raw === 'string' && raw.startsWith('{"proof":"sdxl-exact-replay",'), 'canonical plan marker required');
  const input = JSON.parse(raw);
  assert.deepEqual(Object.keys(input).sort(), ['censusPath', 'deviceUuid', 'ownerGo', 'phase', 'proof']);
  assert.equal(input.proof, 'sdxl-exact-replay');
  assert.equal(input.phase, 'plan');
  assert.equal(input.ownerGo, true, 'owner GPU go required');
  assert(typeof input.censusPath === 'string' && path.win32.isAbsolute(input.censusPath), 'private census path required');
  assert(/^GPU-[a-f0-9-]{36}$/i.test(input.deviceUuid), 'physical UUID required');
  return input;
}

export function boundedPlanWindow(now = Date.now()) {
  assert(Number.isSafeInteger(now) && now > 0, 'invalid plan clock');
  return { goStartsAt: new Date(now).toISOString(), goExpiresAt: new Date(now + 120 * 60 * 1000).toISOString() };
}

async function main() {
  assert.equal(process.platform, 'win32', 'Windows plan only');
  assert.equal(process.env.GITHUB_EVENT_NAME, 'workflow_dispatch', 'manual dispatch required');
  const input = validatePlanDispatch(process.env.SDXL_REPLAY_DISPATCH);
  const sourceSha = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim();
  assert.equal(sourceSha, process.env.GITHUB_SHA, 'source changed');
  assert.equal(execFileSync('git', ['status', '--porcelain'], { cwd: ROOT, encoding: 'utf8' }).trim(), '', 'checkout dirty');
  const pin = await inferenceRevision(ROOT);

  stage = 'private-census';
  const privateDir = await realpath(path.join(process.env.LOCALAPPDATA, 'SceneWorks', 'sdxl-exact-replay'));
  const censusPath = await realpath(input.censusPath);
  assert.equal(path.dirname(censusPath).toLowerCase(), privateDir.toLowerCase(), 'census escaped private directory');
  const census = JSON.parse(await readFile(censusPath, 'utf8'));
  assert.equal(census.version, 0);
  assert.equal(census.kind, 'metadata-only-runtime-census');
  assert.equal(census.ownerGo, false);
  assert.deepEqual(census.missing, ['ownerGpuWindow']);
  assert.equal(census.inferenceRevision, pin);
  assert.equal(census.requestSha256, digest(REQUEST));
  assert.equal(census.deviceUuid, input.deviceUuid);
  assert.equal(census.physicalIndex, 1, 'only approved physical index one');
  const sourceDiff = spawnSync('git', ['diff', '--quiet', census.sourceSha, sourceSha, '--', ...RUNTIME_INPUTS], { cwd: ROOT });
  assert.equal(sourceDiff.status, 0, 'census runtime source changed');
  const bindingPath = await realpath(census.bindingPath);
  assert.equal(path.dirname(bindingPath).toLowerCase(), privateDir.toLowerCase(), 'binding escaped private directory');
  assert.equal(await fileSha256(bindingPath), census.bindingSha256, 'private binding changed');

  stage = 'metadata';
  const devices = execFileSync('nvidia-smi', ['--query-gpu=index,uuid', '--format=csv,noheader'], { encoding: 'utf8' })
    .trim().split(/\r?\n/).map((line) => {
      const [index, uuid] = line.split(',').map((value) => value.trim());
      return { physicalIndex: Number(index), uuid };
    });
  assert.equal(selectProofDevice(devices).uuid, input.deviceUuid, 'physical device changed');
  assert.equal(await fileSha256(census.driver.path), census.driver.sha256, 'driver bytes changed');
  assert(Array.isArray(census.files) && census.files.length > 0, 'selected model files missing');
  for (const file of census.files) assert((await stat(file.path)).isFile(), 'selected model file missing');
  assert(Array.isArray(census.runtimeFiles) && census.runtimeFiles.length > 0, 'runtime census missing');
  for (const file of census.runtimeFiles) {
    assert.equal(await fileSha256(file.path), file.sha256, 'runtime library changed');
  }

  stage = 'write';
  assert(process.env.SDXL_REPLAY_PLAN_REPORT && path.win32.isAbsolute(process.env.SDXL_REPLAY_PLAN_REPORT), 'owned report path missing');
  const planPath = path.join(privateDir, `plan-${sourceSha}-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}.json`);
  const now = Date.now();
  const plan = { version: 1, sourceSha, inferenceRevision: pin, requestSha256: census.requestSha256,
    sourceIdentitySha256: census.sourceIdentitySha256, scheduleSha256: census.scheduleSha256,
    hub: census.hub, files: census.files, runtimeFiles: census.runtimeFiles,
    deviceUuid: census.deviceUuid, physicalIndex: census.physicalIndex, driver: census.driver,
    censusSourceSha: census.sourceSha, censusSha256: await fileSha256(censusPath),
    ownerGo: true, ...boundedPlanWindow(now) };
  validatePlan(plan, sourceSha, pin, 'qualify', now);
  await writeFile(planPath, JSON.stringify(plan, null, 2) + '\n', { flag: 'wx', mode: 0o600 });
  const report = { version: 1, kind: 'approved-private-plan', sourceSha, inferenceRevision: pin,
    planPath, deviceUuid: plan.deviceUuid, physicalIndex: plan.physicalIndex,
    goStartsAt: plan.goStartsAt, goExpiresAt: plan.goExpiresAt,
    selectedFileCount: plan.files.length, runtimeLibraryCount: plan.runtimeFiles.length,
    censusSha256: plan.censusSha256 };
  await writeFile(process.env.SDXL_REPLAY_PLAN_REPORT, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(() => { console.error(`SDXL private plan failed at ${stage}. No API, worker, job, or model was started.`); process.exitCode = 1; });
}
