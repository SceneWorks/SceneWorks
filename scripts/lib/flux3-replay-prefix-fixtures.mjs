// DECLARED_SYNTHETIC CPU-only persistence fixtures. Never model output, device evidence,
// user authorization, or a real terminal-run receipt. No API/model/GPU access occurs here.
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { deflateSync } from 'node:zlib';
import { imageRequest, loadProductTargets, targetRegistry } from '../flux3-image-klein-terminal-plan.mjs';
import { planHash } from '../flux3-image-klein-terminal-budget.mjs';
import { assertFinishedPng, objectHash, pngCrc32, replayRequest, RETAINED_REPLAY_CASE, RETAINED_REPLAY_FIRST, RETAINED_REPLAY_NEXT, sha256 } from '../flux3-image-klein-terminal-driver.mjs';

const projectId = 'DECLARED_SYNTHETIC_OLD_PROJECT';
const oldJobId = 'DECLARED_SYNTHETIC_OLD_JOB';
const oldWorkerId = 'DECLARED_SYNTHETIC_OLD_WORKER';
const oldWorkerPid = 731001;
const timestamp = '2001-01-01T00:00:00.000Z';
// Public checked-in request defaults, not a read of any actual run/artifact.
const fixtureTarget = targetRegistry(await loadProductTargets()).find(t => t.backend === 'candle' && t.modelId === 'flux3_klein');

// Split preparation lets a caller reserve the first transport hash in its OWN CPU
// fixture ledger before constructing a report bound to the resulting old receipt.
export function prepareReplayRequestFixture(approval) {
  const cell = structuredClone(approval.plan.cases.find(c => c.caseId === RETAINED_REPLAY_CASE));
  assert(cell && cell.backend === 'candle' && cell.kind === 'replay');
  const first = imageRequest(cell, fixtureTarget, projectId, {});
  const recipe = { mode: first.mode, model: first.model, adapter: 'candle_flux3', prompt: first.prompt, negativePrompt: first.negativePrompt, seed: first.seed, loras: [], stylePreset: 'none',
    normalizedSettings: { width: first.width, height: first.height, count: first.count, family: 'flux3', characterId: null, characterLookId: null, characterConditioningActive: false, characterConditioningNote: 'DECLARED_SYNTHETIC schema assertion only' },
    rawAdapterSettings: { enhancePrompt: false, guidanceScale: null, mlxQuantize: null, numInferenceSteps: first.advanced.steps, quantTier: 'bf16', realModelInference: true, repo: cell.repository, sampler: first.advanced.sampler, steps: first.advanced.steps } };
  // These runtime-native fields are declared synthetic schema assertions, NOT
  // hardware/inference evidence. Preserve insertion order from both real builders.
  const second = replayRequest(recipe, projectId);
  assert.deepEqual(first, second); assert.notEqual(objectHash(first), objectHash(second));
  return { cell, recipe, imageRequest: first, replayRequest: second, firstRequestSha256: objectHash(first), secondRecordedRequestSha256: objectHash(second) };
}
function syntheticPng() {
  const width = 1024, height = 1024, rowBytes = width * 3 + 1, pixels = Buffer.alloc(rowBytes * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = y * rowBytes + 1 + x * 3;
    pixels[i] = x % 256; pixels[i + 1] = y % 256; pixels[i + 2] = (x + y) % 256;
  }
  const chunk = (type, data) => {
    const bytes = Buffer.alloc(data.length + 12); bytes.writeUInt32BE(data.length); bytes.write(type, 4, 'ascii'); data.copy(bytes, 8);
    bytes.writeUInt32BE(pngCrc32(bytes.subarray(4, 8 + data.length)), 8 + data.length); return bytes;
  };
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(width); ihdr.writeUInt32BE(height, 4); ihdr[8] = 8; ihdr[9] = 2;
  return Buffer.concat([Buffer.from('89504e470d0a1a0a', 'hex'), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(pixels)), chunk('IEND', Buffer.alloc(0))]);
}
export async function makeReplayBaselineFixture({ root, approval, previousReceipt }) {
  assert(path.isAbsolute(root), 'fixture root must be an owned absolute temporary directory');
  const prepared = prepareReplayRequestFixture(approval), { cell, recipe, firstRequestSha256, secondRecordedRequestSha256 } = prepared;
  assert.equal(previousReceipt.events[0].requestSha256, firstRequestSha256, 'reserve prepareReplayRequestFixture(approval).firstRequestSha256 before finalizing fixture');
  await mkdir(root, { recursive: true });
  const paths = { report: path.join(root, 'DECLARED_SYNTHETIC-report.json'), png: path.join(root, 'DECLARED_SYNTHETIC-pattern.png'), sidecar: path.join(root, 'DECLARED_SYNTHETIC-pattern.sceneworks.json'), readback: path.join(root, 'DECLARED_SYNTHETIC-readback.json') };
  const bytes = syntheticPng(), pngSha256 = sha256(bytes), pixelValidity = assertFinishedPng(bytes, 1024, 1024);
  const saved = { schemaVersion: 1, id: 'DECLARED_SYNTHETIC_ASSET', projectId, type: 'image', origin: 'generated', displayName: 'DECLARED_SYNTHETIC CPU pattern', createdAt: timestamp, generationSetId: 'DECLARED_SYNTHETIC_GENERATION_SET',
    file: { path: 'DECLARED_SYNTHETIC-pattern.png', width: 1024, height: 1024, mimeType: 'image/png', duration: null, fps: null },
    lineage: { jobId: oldJobId, parents: [], sourceAssetId: null, sourceTimestamp: null }, recipe,
    status: { favorite: false, rating: 0, rejected: false, trashed: false } };
  const sidecarBytes = JSON.stringify(saved, null, 2), sidecarSha256 = sha256(sidecarBytes);
  const job = { id: oldJobId, type: 'image_generate', status: 'completed', projectId, projectName: null,
    payload: { ...structuredClone(prepared.imageRequest), referenceAssetIds: [], referenceAssetId: null, sourceAssetId: null, maskAssetId: null, characterId: null, characterLookId: null, projectName: null, fitMode: 'contain', stylePreset: 'none' },
    result: { adapter: 'candle_flux3', model: 'flux3_klein', assetIds: [saved.id], assets: [saved], expectedCount: 1, generationSetId: saved.generationSetId },
    requestedGpu: '1', assignedGpu: '1', workerId: oldWorkerId, progress: 1, stage: 'completed', message: 'DECLARED_SYNTHETIC completed schema', error: null, etaSeconds: null, elapsedSeconds: 0.125, attempts: 1, sourceJobId: null, duplicateOfJobId: null, cancelRequested: false,
    createdAt: timestamp, updatedAt: timestamp, startedAt: timestamp, completedAt: timestamp, canceledAt: null, lastHeartbeatAt: timestamp, backend: '1', title: 'DECLARED_SYNTHETIC', queueRank: 0, revision: 1 };
  const output = { job_id: job.id, asset_id: saved.id, generation_set_id: saved.generationSetId, project_id: projectId, backend: '1', adapter: 'candle_flux3', model: 'flux3_klein', media_path: paths.png,
    semanticAcceptance: 'PENDING', pixelValidity, visualObligations: cell.visualObligations ?? [], assetId: saved.id, pngSha256, sidecarSha256, width: 1024, height: 1024, retainedPng: paths.png, retainedSidecar: paths.sidecar, recipe,
    device: { physicalGpuId: '1', gpuUuid: 'DECLARED_SYNTHETIC_GPU_UUID', logicalCudaOrdinal: 0, ownComputePids: [oldWorkerPid], foreignGraphics: { luid: 'DECLARED_SYNTHETIC_LUID', processes: 0, totalBytes: 0, maxProcessBytes: 0, processLimitBytes: 731, totalLimitBytes: 1462 } } };
  const recorded = { ...cell, executionStatus: 'FAILED', sourceSha: approval.sourceSha, inferenceMainSha: approval.inferenceMainSha, resourceAdmissionId: 'DECLARED_SYNTHETIC_ADMISSION',
    events: [{ id: job.id, status: 'running', stage: 'denoising', progress: 0.5, workerId: oldWorkerId, result: {} }],
    requests: [{ body: prepared.imageRequest, sha256: firstRequestSha256 }, { body: prepared.replayRequest, sha256: secondRecordedRequestSha256 }], jobs: [job], outputs: [output], adapters: [], observations: [], timeoutSeconds: 10, visualInspection: 'NOT_INSPECTED',
    error: 'DECLARED_SYNTHETIC Get-Counter CounterApiError: performance counter samples is not valid' };
  const report = { schemaVersion: 1, kind: 'flux3-image-klein-terminal-report', declaredSynthetic: true, admissionId: 'DECLARED_SYNTHETIC_ADMISSION', ledgerId: approval.ledgerId, approvedPlanSha256: planHash(approval), sourceSha: approval.sourceSha, inferenceMainSha: approval.inferenceMainSha,
    runner: 'DECLARED_SYNTHETIC_CPU', runnerAccount: 'DECLARED_SYNTHETIC_ACCOUNT', status: 'FAILED', cases: [recorded], ledgerReceipt: structuredClone(previousReceipt) };
  const readback = { caseId: cell.caseId, jobStatus: job.status, adapter: 'candle_flux3', model: 'flux3_klein', pngSha256, width: 1024, height: 1024, pixelValidity, retainedSidecarHashVerified: true, requestHashesVerified: true, seedAndReplayReconstructionVerified: true, secondRequestRecordedButNoSecondJobOrPermit: true, physicalGpuId: '1', ownComputePids: [oldWorkerPid], semanticAcceptance: 'PENDING', visualInspection: 'NOT_INSPECTED' };
  const reportBytes = JSON.stringify(report, null, 2), readbackBytes = JSON.stringify(readback, null, 2);
  await Promise.all([writeFile(paths.png, bytes, { flag: 'wx' }), writeFile(paths.sidecar, sidecarBytes, { flag: 'wx' }), writeFile(paths.report, reportBytes, { flag: 'wx' }), writeFile(paths.readback, readbackBytes, { flag: 'wx' })]);
  const baseline = { caseId: RETAINED_REPLAY_CASE, consumedInvocationId: RETAINED_REPLAY_FIRST, nextInvocationId: RETAINED_REPLAY_NEXT, allowedLineageRemap: ['projectId'],
    report: { path: paths.report, sha256: sha256(reportBytes) }, png: { path: paths.png, sha256: pngSha256 }, sidecar: { path: paths.sidecar, sha256: sidecarSha256 }, readback: { path: paths.readback, sha256: sha256(readbackBytes) },
    firstRequestSha256, secondRecordedRequestSha256, oldJobId, oldWorkerId, oldWorkerPid };
  return { ...prepared, baseline, paths, report, recorded, readback, saved, job, output, pngBytes: bytes };
}
