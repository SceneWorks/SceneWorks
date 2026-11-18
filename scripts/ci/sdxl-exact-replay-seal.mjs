#!/usr/bin/env node
// Preserve a successful physical qualification for the separate replay run.
// This step only reads metadata and writes a private, bounded plan.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFile, realpath, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { inferenceRevision } from '../lib/inference-provenance.mjs';
import { SCOPE, digest, requireReplayQualification, validatePlan } from './sdxl-exact-replay-proof.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const REVIEW_URL = 'https://github.com/SceneWorks/inference/pull/1464';

export function makeReplayPlan(plan, qualification, sourceSha, pin, now = Date.now()) {
  validatePlan(plan, sourceSha, pin, 'qualify', Date.parse(plan.goStartsAt));
  assert.equal(plan.physicalIndex, 1, 'only physical GPU one is approved');
  assert.equal(qualification.sourceSha, sourceSha, 'qualified app source changed');
  assert.equal(qualification.inferenceRevision, pin, 'qualified inference source changed');
  assert.equal(qualification.physicalIndex, plan.physicalIndex, 'qualified physical GPU changed');
  for (const job of qualification.jobs ?? []) {
    assert.deepEqual(job.support, { kind: 'deterministic', scope: SCOPE }, 'provider did not declare exact replay support');
  }
  const starts = new Date(now).toISOString();
  const expires = new Date(now + 120 * 60 * 1000).toISOString();
  const replayPlan = { ...plan, qualificationSha256: digest(qualification),
    declaration: { scope: SCOPE, candidateSourceSha: sourceSha,
      candidateInferenceRevision: pin, reviewUrl: REVIEW_URL },
    goStartsAt: starts, goExpiresAt: expires };
  requireReplayQualification(replayPlan, qualification);
  validatePlan(replayPlan, sourceSha, pin, 'replay', now);
  return replayPlan;
}

async function main() {
  assert.equal(process.platform, 'win32', 'Windows runner required');
  assert.equal(process.env.GITHUB_EVENT_NAME, 'workflow_dispatch', 'manual dispatch required');
  assert.equal(process.env.SDXL_REPLAY_PHASE, 'qualify', 'only qualification can seal replay');
  const sourceSha = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim();
  assert.equal(sourceSha, process.env.GITHUB_SHA, 'source changed');
  assert.equal(execFileSync('git', ['status', '--porcelain'], { cwd: ROOT, encoding: 'utf8' }).trim(), '', 'checkout dirty');
  const pin = await inferenceRevision(ROOT);
  const privateDir = await realpath(path.join(process.env.LOCALAPPDATA, 'SceneWorks', 'sdxl-exact-replay'));
  const planPath = await realpath(process.env.SDXL_REPLAY_PLAN);
  assert.equal(path.dirname(planPath).toLowerCase(), privateDir.toLowerCase(), 'plan escaped private directory');
  const plan = JSON.parse(await readFile(planPath, 'utf8'));
  const qualification = JSON.parse(await readFile(path.join(process.env.SDXL_REPLAY_OUTPUT, 'evidence.json'), 'utf8'));
  const replayPlan = makeReplayPlan(plan, qualification, sourceSha, pin);
  const suffix = `${sourceSha}-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}`;
  const qualificationFile = path.join(privateDir, `qualification-${suffix}.json`);
  const replayPath = path.join(privateDir, `replay-plan-${suffix}.json`);
  await writeFile(qualificationFile, JSON.stringify(qualification, null, 2) + '\n', { flag: 'wx', mode: 0o600 });
  replayPlan.qualificationFile = qualificationFile;
  await writeFile(replayPath, JSON.stringify(replayPlan, null, 2) + '\n', { flag: 'wx', mode: 0o600 });
  assert(process.env.SDXL_REPLAY_SEAL_REPORT && path.win32.isAbsolute(process.env.SDXL_REPLAY_SEAL_REPORT), 'report path missing');
  await writeFile(process.env.SDXL_REPLAY_SEAL_REPORT, JSON.stringify({
    version: 1, kind: 'reviewed-replay-plan', sourceSha, inferenceRevision: pin,
    planPath: replayPath, qualificationSha256: replayPlan.qualificationSha256,
    deviceUuid: replayPlan.deviceUuid, physicalIndex: replayPlan.physicalIndex,
    goStartsAt: replayPlan.goStartsAt, goExpiresAt: replayPlan.goExpiresAt,
  }, null, 2) + '\n', { flag: 'wx' });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch((error) => { console.error(`SDXL replay plan seal failed: ${error.message}`); process.exitCode = 1; });
}
