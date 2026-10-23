// DECLARED_SYNTHETIC CPU-only retained-prefix tests. Zero model/API/device access.
import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { loadProductTargets, terminalPlan } from './flux3-image-klein-terminal-plan.mjs';
import { auditLedger, initializeLedger, planHash, reserveInvocation } from './flux3-image-klein-terminal-budget.mjs';
import { assertFinishedPng, executeRetainedReplayPrefix, objectHash, retainedReplayBody, RETAINED_REPLAY_FIRST, RETAINED_REPLAY_NEXT, sha256, validateReplayPrefixBaseline } from './flux3-image-klein-terminal-driver.mjs';
import { selectProductInvocation } from './flux3-image-klein-terminal-session.mjs';
import { makeReplayBaselineFixture, prepareReplayRequestFixture } from './lib/flux3-replay-prefix-fixtures.mjs';

const profile = terminalPlan(await loadProductTargets());
async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'DECLARED_SYNTHETIC-replay-prefix-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const approval = { schemaVersion: 1, kind: 'flux3-terminal-plan-approval', approvalId: 'DECLARED_SYNTHETIC_APPROVAL', userPlanApprovalId: 'DECLARED_SYNTHETIC_NOT_USER_AUTHORIZATION', approvedBy: 'DECLARED_SYNTHETIC CPU fixture', sourceSha: 'a'.repeat(40), inferenceMainSha: 'b'.repeat(40), ledgerId: 'DECLARED_SYNTHETIC_LEDGER', ledgerRoots: { mlx: path.join(root, 'mlx-ledger'), candle: path.join(root, 'candle-ledger') }, inputSpecSha256: { mlx: 'd'.repeat(64), candle: 'd'.repeat(64) }, plan: structuredClone(profile) };
  const prepared = prepareReplayRequestFixture(approval), invocation = approval.plan.generationInvocations.find(i => i.invocationId === RETAINED_REPLAY_FIRST);
  await initializeLedger({ approval, backend: 'candle', ledgerRoot: approval.ledgerRoots.candle });
  await reserveInvocation({ approval, backend: 'candle', ledgerRoot: approval.ledgerRoots.candle, ledgerId: approval.ledgerId, approvedPlanSha256: planHash(approval), invocationId: invocation.invocationId, caseId: invocation.caseId, kind: invocation.kind, requestSha256: prepared.firstRequestSha256 });
  const previousReceipt = await auditLedger({ approval, backend: 'candle', ledgerRoot: approval.ledgerRoots.candle });
  const data = await makeReplayBaselineFixture({ root: path.join(root, 'baseline'), approval, previousReceipt });
  return { root, approval, previousReceipt, ...data, authorization: { baseline: data.baseline } };
}
const validate = f => validateReplayPrefixBaseline({ previousApproval: f.approval, previousReceipt: f.previousReceipt, authorization: f.authorization, ownedRoot: f.root });
async function persist(f) {
  for (const [name, record] of [['report', f.report], ['sidecar', f.saved], ['readback', f.readback]]) {
    const bytes = JSON.stringify(record, null, 2); await writeFile(f.paths[name], bytes); f.baseline[name].sha256 = sha256(bytes);
  }
}
// Rebind transport hashes when testing semantic guards rather than stale-hash guards.
function rehashRequests(f) {
  const [first, second] = f.recorded.requests;
  first.sha256 = objectHash(first.body); second.sha256 = objectHash(second.body);
  f.baseline.firstRequestSha256 = first.sha256; f.baseline.secondRecordedRequestSha256 = second.sha256;
  f.previousReceipt.events[0].requestSha256 = first.sha256; f.report.ledgerReceipt = structuredClone(f.previousReceipt);
}

test('CPU synthetic retained baseline uses the real validator without promoting historical failure or semantic acceptance', async t => {
  const f = await fixture(t), result = await validate(f);
  assert.equal(result.historicalExecutionStatus, 'FAILED'); assert.equal(result.historicalFailureKind, 'GPU_COUNTER_MEASUREMENT_FAILED');
  assert.equal(result.semanticAcceptance, 'PENDING'); assert.equal(f.recorded.visualInspection, 'NOT_INSPECTED');
  assert.deepEqual(result.firstRequest, result.recordedReplayRequest);
  assert.notEqual(result.firstRequestSha256, result.secondRecordedRequestSha256);
  assert.equal(result.firstRequestSha256, f.previousReceipt.events[0].requestSha256);
  assert.equal(f.previousReceipt.events.length, 1); assert.equal(f.previousReceipt.state.generationCount, 1);
  assert.equal(result.pngSha256, sha256(await readFile(f.paths.png)));
  assert.equal(assertFinishedPng(f.pngBytes, 1024, 1024).nonblank, true);
  assert.equal(f.report.declaredSynthetic, true); assert.match(result.oldWorkerId, /DECLARED_SYNTHETIC/);
  assert.deepEqual(Object.keys(f.baseline).sort(), ['caseId', 'consumedInvocationId', 'nextInvocationId', 'allowedLineageRemap', 'report', 'png', 'sidecar', 'readback', 'firstRequestSha256', 'secondRecordedRequestSha256', 'oldJobId', 'oldWorkerId', 'oldWorkerPid'].sort());
});

test('artifact bytes and exact binding hashes cannot be changed after authorization', async t => {
  for (const name of ['png', 'sidecar', 'report', 'readback']) await t.test(name, async t => {
    const f = await fixture(t); await writeFile(f.paths[name], Buffer.concat([await readFile(f.paths[name]), Buffer.from('\nDECLARED_SYNTHETIC_MUTATION')]));
    await assert.rejects(validate(f), /evidence bytes changed/);
  });
});

const mutations = [
  ['wrong retained case', f => f.baseline.caseId = 'candle.DECLARED_SYNTHETIC.other.bf16'],
  ['wrong consumed invocation', f => f.baseline.consumedInvocationId = f.baseline.nextInvocationId],
  ['wrong next invocation', f => f.baseline.nextInvocationId = f.baseline.consumedInvocationId],
  ['unapproved seed remap', f => f.baseline.allowedLineageRemap = ['projectId', 'seed']],
  ['other field instead of project lineage', f => f.baseline.allowedLineageRemap = ['prompt']],
  ['missing project lineage permission', f => f.baseline.allowedLineageRemap = []],
  ['extra artifact binding field', f => f.baseline.png.syntheticSkip = true],
  ['invalid artifact hash', f => f.baseline.png.sha256 = 'DECLARED_SYNTHETIC_INVALID_HASH'],
  ['relative artifact path', f => f.baseline.png.path = 'DECLARED_SYNTHETIC-pattern.png'],
  ['source identity', f => f.report.sourceSha = 'c'.repeat(40)],
  ['inference identity', f => f.report.inferenceMainSha = 'c'.repeat(40)],
  ['approved plan hash', f => f.report.approvedPlanSha256 = 'c'.repeat(64)],
  ['ledger identity', f => f.report.ledgerId = 'DECLARED_SYNTHETIC_OTHER_LEDGER'],
  ['old receipt binding', f => f.report.ledgerReceipt.events[0].requestSha256 = 'c'.repeat(64)],
  ['case settings', f => f.recorded.options.enhancePrompt = true],
  ['historical PASS relabel', f => f.recorded.executionStatus = 'PASS'],
  ['wrong historical failure', f => f.recorded.error = 'DECLARED_SYNTHETIC unrelated execution failure'],
  ['second output not retained prefix', f => f.recorded.outputs.push(structuredClone(f.output))],
  ['second job not retained prefix', f => f.recorded.jobs.push(structuredClone(f.job))],
  ['second request absent', f => f.recorded.requests.pop()],
  ['first transport hash', f => f.baseline.firstRequestSha256 = 'c'.repeat(64)],
  ['second transport hash', f => f.baseline.secondRecordedRequestSha256 = 'c'.repeat(64)],
  ['false raw request equality', f => { f.recorded.requests[1].body = structuredClone(f.recorded.requests[0].body); rehashRequests(f); }],
  ['changed replay seed despite rebound hash', f => { f.recorded.requests[1].body.seed++; rehashRequests(f); }],
  ['coherent seed drift from approved authored request', f => { for (const request of f.recorded.requests) request.body.seed++; f.saved.recipe.seed++; f.job.payload.seed++; rehashRequests(f); }],
  ['unknown inference field in both rehashed request bodies', f => { for (const request of f.recorded.requests) request.body.advanced.unknownSyntheticField = true; rehashRequests(f); }],
  ['output path differing from approved binding', f => f.output.retainedPng = path.join(f.root, 'DECLARED_SYNTHETIC_NOT_THE_APPROVED_PNG.png')],
  ['unapproved project remap in historical pair', f => { f.recorded.requests[1].body.projectId = 'DECLARED_SYNTHETIC_NEW_PROJECT'; rehashRequests(f); }],
  ['reference-bearing request', f => { for (const request of f.recorded.requests) request.body.referenceAssetIds = ['DECLARED_SYNTHETIC_REFERENCE']; rehashRequests(f); }],
  ['adapter-bearing request', f => { for (const request of f.recorded.requests) request.body.loras = [{ id: 'DECLARED_SYNTHETIC_LORA', weight: 1 }]; rehashRequests(f); }],
  ['PUP-enabled request', f => { for (const request of f.recorded.requests) request.body.advanced.enhancePrompt = true; rehashRequests(f); }],
  ['layout-bearing request', f => { for (const request of f.recorded.requests) request.body.advanced.flux3_layout = { version: 1, boxes: [] }; rehashRequests(f); }],
  ['job not completed', f => f.job.status = 'failed'],
  ['job stage incomplete', f => f.job.stage = 'denoising'],
  ['job has error', f => f.job.error = 'DECLARED_SYNTHETIC error'],
  ['wrong job', f => f.baseline.oldJobId = 'DECLARED_SYNTHETIC_OTHER_JOB'],
  ['foreign worker', f => f.baseline.oldWorkerId = 'DECLARED_SYNTHETIC_OTHER_WORKER'],
  ['wrong device backend', f => f.job.backend = '0'],
  ['non-native adapter', f => f.job.result.adapter = 'DECLARED_SYNTHETIC_OTHER_ADAPTER'],
  ['unpublished worker assetWrites', f => f.job.result.assetWrites = []],
  ['asset identity', f => f.job.result.assetIds[0] = 'DECLARED_SYNTHETIC_OTHER_ASSET'],
  ['sidecar project lineage', f => f.saved.projectId = 'DECLARED_SYNTHETIC_OTHER_PROJECT'],
  ['sidecar job lineage', f => f.saved.lineage.jobId = 'DECLARED_SYNTHETIC_OTHER_JOB'],
  ['saved model inference assertion absent', f => f.saved.recipe.rawAdapterSettings.realModelInference = false],
  ['saved steps', f => f.saved.recipe.rawAdapterSettings.numInferenceSteps++],
  ['saved repository', f => f.saved.recipe.rawAdapterSettings.repo = 'DECLARED_SYNTHETIC_OTHER_REPOSITORY'],
  ['saved native tier', f => f.saved.recipe.rawAdapterSettings.quantTier = 'q4'],
  ['saved quantization bits', f => f.saved.recipe.rawAdapterSettings.mlxQuantize = 4],
  ['saved family', f => f.saved.recipe.normalizedSettings.family = 'DECLARED_SYNTHETIC_OTHER_FAMILY'],
  ['saved dimensions', f => f.saved.recipe.normalizedSettings.width = 512],
  ['saved sampler', f => f.saved.recipe.rawAdapterSettings.sampler = 'DECLARED_SYNTHETIC_OTHER_SAMPLER'],
  ['unexpected enhancer outcome', f => f.saved.recipe.rawAdapterSettings.promptEnhancement = { outcome: 'enhanced' }],
  ['saved PNG dimensions', f => f.saved.file.width = 512],
  ['pixel validity', f => f.output.pixelValidity = { ...f.output.pixelValidity, nonblank: false }],
  ['readback pixel validity', f => f.readback.pixelValidity = { ...f.readback.pixelValidity, decodedContentSha256: 'c'.repeat(64) }],
  ['readback PNG digest', f => f.readback.pngSha256 = 'c'.repeat(64)],
  ['unverified retained sidecar', f => f.readback.retainedSidecarHashVerified = false],
  ['unverified seed reconstruction', f => f.readback.seedAndReplayReconstructionVerified = false],
  ['output semantic PASS', f => f.output.semanticAcceptance = 'PASS'],
  ['readback semantic PASS', f => f.readback.semanticAcceptance = 'PASS'],
  ['visual PASS relabel', f => f.recorded.visualInspection = 'PASS'],
  ['wrong physical device', f => f.output.device.physicalGpuId = '0'],
  ['wrong logical device', f => f.output.device.logicalCudaOrdinal = 1],
  ['missing owned worker PID', f => f.output.device.ownComputePids = []],
];
for (const [name, mutate] of mutations) test(`CPU baseline rejects ${name}, including coherently rehashed metadata`, async t => {
  const f = await fixture(t); mutate(f);
  // Keep the saved sidecar's exact hash synchronized for semantic mutations;
  // otherwise the test could pass on a stale digest before reaching the guard.
  f.output.sidecarSha256 = sha256(JSON.stringify(f.saved, null, 2));
  await persist(f); await assert.rejects(validate(f));
});

test('encoded PNG corruption remains rejected even if all encoded hashes are rebound', async t => {
  const f = await fixture(t), bytes = Buffer.from(f.pngBytes); bytes[bytes.length - 1] ^= 1;
  await writeFile(f.paths.png, bytes); f.baseline.png.sha256 = sha256(bytes); f.output.pngSha256 = f.baseline.png.sha256; f.readback.pngSha256 = f.baseline.png.sha256;
  await persist(f); await assert.rejects(validate(f), /CRC mismatch/);
});

test('baseline artifacts cannot escape the explicit owned temporary root', async t => {
  const f = await fixture(t);
  await assert.rejects(validateReplayPrefixBaseline({ previousApproval: f.approval, previousReceipt: f.previousReceipt, authorization: f.authorization, ownedRoot: f.approval.ledgerRoots.candle }), /escap|outside|contain/i);
});

async function resumeFixture(t) {
  const f = await fixture(t), prefix = await validate(f);
  // Isolated helper binding ONLY, not proof of source-transition authorization.
  // Full authorization validation belongs to the budget tests.
  f.approval.candleSourceTransitionSha256 = sha256('DECLARED_SYNTHETIC_ISOLATED_HELPER_AUTHORIZATION');
  prefix.authorizationSha256 = f.approval.candleSourceTransitionSha256;
  const project = { id: 'DECLARED_SYNTHETIC_FRESH_PROJECT' }, body = { ...structuredClone(prefix.firstRequest), projectId: project.id };
  const report = { outputs: [], jobs: [] }, identity = { workerId: 'DECLARED_SYNTHETIC_FRESH_WORKER' };
  // Any lifecycle/API/DB access fails; only the CPU generate callback is available.
  const session = new Proxy({ replayPrefix: prefix, identity }, { get(target, key) { assert(Object.hasOwn(target, key), `unexpected session access: ${String(key)}`); return target[key]; } });
  return { ...f, prefix, project, body, report, session };
}

test('retainedReplayBody permits only projectId remapping for the fixed known reference-free prefix', async t => {
  const f = await resumeFixture(t), request = retainedReplayBody(f.prefix, f.project.id, f.body);
  assert.equal(request.projectId, f.project.id); assert.notEqual(request.projectId, f.prefix.firstRequest.projectId);
  assert.deepEqual(request, { ...f.prefix.recordedReplayRequest, projectId: f.project.id });
  assert.equal(f.prefix.firstRequest.projectId, f.saved.projectId, 'original retained request is not rewritten');
  const changes = [
    b => b.seed++, b => b.prompt += ' changed', b => b.negativePrompt = 'DECLARED_SYNTHETIC_CHANGED',
    b => b.width = 512, b => b.height = 512, b => b.count = 2, b => b.mode = 'edit_image', b => b.model = 'flux3_klein_base',
    b => b.advanced.steps++, b => b.advanced.sampler = 'DECLARED_SYNTHETIC_OTHER', b => b.advanced.guidanceScale = 1,
    b => b.advanced.quantTier = 'q4', b => b.advanced.enhancePrompt = true,
    b => b.advanced.flux3_layout = { version: 1, boxes: [] }, b => b.advanced.unknownSyntheticField = true,
    b => b.referenceAssetIds = ['DECLARED_SYNTHETIC_REFERENCE'], b => b.referenceAssetIds = [],
    b => b.loras = [{ id: 'DECLARED_SYNTHETIC_LORA', weight: 1 }], b => b.unknownSyntheticField = true,
  ];
  for (const mutate of changes) {
    const body = structuredClone(f.body); mutate(body);
    assert.throws(() => retainedReplayBody(f.prefix, f.project.id, body));
    assert.throws(() => selectProductInvocation(f.approval, f.cell, 0, body, f.prefix));
  }
  for (const projectId of ['', '../DECLARED_SYNTHETIC', 'DECLARED SYNTHETIC', null]) assert.throws(() => retainedReplayBody(f.prefix, projectId, { ...f.body, projectId }));
  for (const mutate of [p => p.caseId = 'DECLARED_SYNTHETIC_OTHER_CASE', p => p.consumedInvocationId = RETAINED_REPLAY_NEXT, p => p.nextInvocationId = RETAINED_REPLAY_FIRST, p => p.recipe.seed++, p => p.recordedReplayRequest.advanced.steps++]) {
    const prefix = structuredClone(f.prefix); mutate(prefix); assert.throws(() => retainedReplayBody(prefix, f.project.id, f.body));
  }
});

test('selectProductInvocation offsets only the fixed authorized prefix to generation-2; never generic skip-consumed', async t => {
  const f = await resumeFixture(t);
  assert.equal(selectProductInvocation(f.approval, f.cell, 0, f.body, f.prefix).invocationId, RETAINED_REPLAY_NEXT);
  assert.equal(selectProductInvocation(f.approval, f.cell, 0, f.body).invocationId, RETAINED_REPLAY_FIRST, 'without prefix consumed permits are not silently skipped');
  assert.equal(selectProductInvocation(f.approval, f.cell, 1, f.body).invocationId, RETAINED_REPLAY_NEXT);
  for (const index of [-1, 0.5, 1, 2]) assert.throws(() => selectProductInvocation(f.approval, f.cell, index, f.body, f.prefix));
  const other = f.approval.plan.cases.find(c => c.backend === 'candle' && c.caseId !== f.cell.caseId && f.approval.plan.generationInvocations.some(i => i.caseId === c.caseId && i.kind === 'product_generation'));
  const otherInvocations = f.approval.plan.generationInvocations.filter(i => i.backend === other.backend && i.caseId === other.caseId && i.kind === 'product_generation');
  for (const index of [0, 1]) assert.deepEqual(selectProductInvocation(f.approval, other, index, {}, f.prefix), otherInvocations[index]);
  assert.throws(() => selectProductInvocation(f.approval, other, 0, {}, { ...f.prefix, caseId: other.caseId }));
  for (const mutate of [p => p.authorizationSha256 = 'c'.repeat(64), p => delete p.authorizationSha256, p => p.consumedInvocationId = RETAINED_REPLAY_NEXT, p => p.nextInvocationId = RETAINED_REPLAY_FIRST]) {
    const prefix = structuredClone(f.prefix); mutate(prefix); assert.throws(() => selectProductInvocation(f.approval, f.cell, 0, f.body, prefix));
  }
  for (const mutate of [c => c.kind = 'image', c => c.backend = 'mlx']) { const cell = structuredClone(f.cell); mutate(cell); assert.throws(() => selectProductInvocation(f.approval, cell, 0, f.body, f.prefix)); }
});

test('retained replay invokes only generation-2 once and carries old FAILED/PENDING evidence separately from new outputs', async t => {
  const f = await resumeFixture(t), calls = [];
  await executeRetainedReplayPrefix(f.session, f.cell, f.body, f.project, f.report, async request => {
    calls.push({ invocation: selectProductInvocation(f.approval, f.cell, calls.length, request, f.prefix), request });
    f.report.jobs.push({ id: 'DECLARED_SYNTHETIC_NEW_JOB', workerId: f.session.identity.workerId });
    f.report.outputs.push({ assetId: 'DECLARED_SYNTHETIC_NEW_ASSET', pngSha256: f.prefix.pngSha256, semanticAcceptance: 'PENDING' });
  });
  assert.equal(calls.length, 1); assert.equal(calls[0].invocation.invocationId, RETAINED_REPLAY_NEXT);
  assert.deepEqual(calls[0].request, f.body); assert.equal(f.report.jobs.length, 1); assert.equal(f.report.outputs.length, 1);
  assert.equal(f.report.outputs[0].assetId, 'DECLARED_SYNTHETIC_NEW_ASSET'); assert.notEqual(f.report.outputs[0].assetId, f.saved.id);
  assert.equal(f.report.carriedBaseline.invocationId, RETAINED_REPLAY_FIRST); assert.equal(f.report.carriedBaseline.jobId, f.prefix.oldJobId);
  assert.equal(f.report.carriedBaseline.historicalExecutionStatus, 'FAILED'); assert.equal(f.report.carriedBaseline.semanticAcceptance, 'PENDING');
  assert.equal(f.report.carriedBaseline.evidenceRole, 'RETAINED_RENDER_BASELINE_NOT_NEW_GENERATION'); assert.equal(f.report.replayCompletedFromRetainedBaseline, true);
  assert.equal(f.report.outputs.includes(f.report.carriedBaseline), false); assert.equal(f.recorded.executionStatus, 'FAILED');
  assert.equal(f.previousReceipt.events.length, 1); assert.equal(f.previousReceipt.state.generationCount, 1, 'isolated callback does not write even the fixture ledger');
  await assert.rejects(executeRetainedReplayPrefix(f.session, f.cell, f.body, f.project, f.report, async () => assert.fail('must not generate again')));
  assert.equal(calls.length, 1);
});

test('retained replay refuses wrong baseline status, absent authorization or old worker before its callback', async t => {
  const original = await resumeFixture(t);
  for (const mutate of [
    f => f.session.identity.workerId = f.prefix.oldWorkerId,
    f => f.prefix.historicalExecutionStatus = 'PASS', f => f.prefix.semanticAcceptance = 'PASS',
    f => delete f.prefix.authorizationSha256, f => f.prefix.caseId = 'DECLARED_SYNTHETIC_OTHER_CASE',
    f => f.cell.kind = 'image', f => f.cell.backend = 'mlx',
    f => f.report.jobs.push({ id: 'DECLARED_SYNTHETIC_PREEXISTING_JOB' }),
    f => f.report.outputs.push({ assetId: 'DECLARED_SYNTHETIC_PREEXISTING_ASSET' }),
  ]) {
    const prefix = structuredClone(original.prefix), f = { prefix, cell: structuredClone(original.cell), report: { outputs: [], jobs: [] }, session: { replayPrefix: prefix, identity: { workerId: 'DECLARED_SYNTHETIC_FRESH_WORKER' } } };
    mutate(f); let calls = 0;
    await assert.rejects(executeRetainedReplayPrefix(f.session, f.cell, original.body, original.project, f.report, async () => { calls++; }));
    assert.equal(calls, 0);
  }
});

test('retained replay rejects generation-2 hash mismatch or wrong new job/worker identity without claiming completion', async t => {
  const original = await resumeFixture(t);
  for (const mutation of ['hash', 'old-job', 'foreign-worker', 'missing-output', 'two-outputs', 'missing-job']) {
    const report = { outputs: [], jobs: [] }; let calls = 0;
    await assert.rejects(executeRetainedReplayPrefix(original.session, original.cell, original.body, original.project, report, async () => {
      calls++;
      if (mutation !== 'missing-job') report.jobs.push({ id: mutation === 'old-job' ? original.prefix.oldJobId : 'DECLARED_SYNTHETIC_NEW_JOB', workerId: mutation === 'foreign-worker' ? original.prefix.oldWorkerId : original.session.identity.workerId });
      if (mutation !== 'missing-output') report.outputs.push({ pngSha256: mutation === 'hash' ? 'c'.repeat(64) : original.prefix.pngSha256 });
      if (mutation === 'two-outputs') report.outputs.push({ pngSha256: original.prefix.pngSha256 });
    }));
    assert.equal(calls, 1); assert.notEqual(report.replayCompletedFromRetainedBaseline, true);
    assert.equal(report.carriedBaseline.historicalExecutionStatus, 'FAILED'); assert.equal(report.carriedBaseline.semanticAcceptance, 'PENDING');
  }
});

test('CPU cross-device replay preserves GPU1 provenance and requires actual GPU0 launch/output proofs', async t => {
  const f = await resumeFixture(t);
  const historical = { physicalGpuId: '1', gpuUuid: 'GPU-e4b79931-7be6-f216-460a-f5405cfafffe', pciBusId: '00000000:C1:00.0' };
  const destination = { physicalGpuId: '0', gpuUuid: 'GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0', pciBusId: '00000000:21:00.0', logicalCudaOrdinal: 0 };
  f.prefix.deviceTransition = { previousDevice: historical, nextDevice: destination };
  f.prefix.oldDevice = { ...historical, logicalCudaOrdinal: 0 };
  const session = { replayPrefix: f.prefix, identity: f.session.identity, deviceAtLaunch: destination };
  const generate = async () => {
    f.report.jobs.push({ id: 'DECLARED_SYNTHETIC_GPU0_JOB', workerId: session.identity.workerId });
    f.report.outputs.push({ pngSha256: f.prefix.pngSha256, device: { ...destination } });
  };
  await executeRetainedReplayPrefix(session, f.cell, f.body, f.project, f.report, generate);
  assert.equal(f.report.replayComparison.kind, 'CROSS_SOURCE_CROSS_DEVICE_FRESH_PROCESS_EXACT_OUTPUT_COMPARISON');
  assert.equal(f.report.replayComparison.samePhysicalDeviceClaim, false);
  assert.equal(f.report.replayComparison.sameBinaryClaim, false);
  assert.equal(f.report.replayComparison.sameSourceClaim, false);
  assert.equal(f.report.replayComparison.historicalMemoryTransferred, false);
  assert.equal(f.report.replayComparison.historicalDevice.physicalGpuId, '1');
  assert.equal(f.report.replayComparison.destinationDevice.physicalGpuId, '0');
  for (const mutate of [d => d.physicalGpuId = '1', d => d.gpuUuid = historical.gpuUuid, d => d.pciBusId = historical.pciBusId, d => d.logicalCudaOrdinal = 1]) {
    const bad = structuredClone(destination); mutate(bad); let calls = 0;
    await assert.rejects(executeRetainedReplayPrefix({ ...session, deviceAtLaunch: bad }, f.cell, f.body, f.project, { outputs: [], jobs: [] }, async () => { calls++; }));
    assert.equal(calls, 0);
    const report = { outputs: [], jobs: [] };
    await assert.rejects(executeRetainedReplayPrefix(session, f.cell, f.body, f.project, report, async () => {
      report.jobs.push({ id: 'DECLARED_SYNTHETIC_BAD_DEVICE_JOB', workerId: session.identity.workerId });
      report.outputs.push({ pngSha256: f.prefix.pngSha256, device: bad });
    }));
  }
});

test('CPU historical GPU UUID cannot be rebound to destination when cross-device authorization is supplied', async t => {
  const f = await fixture(t);
  f.authorization.deviceTransition = { previousDevice: { physicalGpuId: '1', gpuUuid: 'GPU-e4b79931-7be6-f216-460a-f5405cfafffe', pciBusId: '00000000:C1:00.0' } };
  await assert.rejects(validate(f), /historical device/);
  f.output.device.gpuUuid = f.authorization.deviceTransition.previousDevice.gpuUuid;
  await persist(f);
  const baseline = await validate(f);
  assert.equal(baseline.oldDevice.physicalGpuId, '1');
  assert.deepEqual(baseline.deviceTransition, f.authorization.deviceTransition);
});
