// Metadata only: no service, provider, CUDA context or model is started by this suite.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, writeFile, readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { REQUEST, SCOPE, digest, sourceIdentity, parseDispatch, validateDispatch, validatePlan, verifyMembers,
  completedWitness, verifyQueuedPayload, verifyReplayCandidate, compareFresh, requireReplayQualification, verifyDeviceSelection,
  finalizeEvidence } from './sdxl-exact-replay-proof.mjs';
import { validatePreparationDispatch } from './sdxl-exact-replay-prepare.mjs';

const hash = (s) => createHash('sha256').update(s).digest('hex');
const sha = 'a'.repeat(40), pin = 'b'.repeat(40), uuid = 'GPU-00000000-0000-0000-0000-000000000001';
const resolved = (value) => ({ state: 'resolved', value });
function fixture() {
  const png = Buffer.from([137,80,78,71,13,10,26,10,1,2,3]); // metadata stand-in, never rendered
  const route = { backend: 'candle', platform: 'windows', expectedProvider: {
    modelSlot: 'modelManifestEntry', modelId: 'sdxl', operation: 'image_generate', providerId: 'sdxl' },
  models: [{ schemaVersion: 1, slot: 'modelManifestEntry', modelId: 'sdxl', backend: 'candle',
    variant: 'base', tier: '4', authoredOptionsSha256: hash('authored'), members: [{ role: 'primary',
      componentId: 'unet', source: { kind: 'digest', sha256: hash('source') }, files: [
        { relativePath: 'unet/model.bin', sizeBytes: 3, sha256: hash('abc') } ] }] }] };
  const schedule = [{ sampler: 'ddim', schedule: { element: 'f32_le_bits', count: 21, sha256: hash('sigmas') }, estimator: 'eps' }];
  const execution = { state: 'resolved', identity: { version: 1, provider_id: 'sdxl', backend: 'candle', runtime_id: 'candle-execution-v1', output_units: 1,
    sampling: resolved(['ddim']), schedule: resolved(schedule), estimator: resolved('eps'),
    geometry: resolved([{ width: 1024, height: 1024, channels: 3, format: 'rgb8' }]),
    quality_policy: resolved({ compute: [{ device: 'Cuda(0)', dtype: 'F16' }], operations: [] }), output_policy: resolved(['rgb8']),
    randomness: { kind: 'draws', scheme: 'fixture', draws: [{ output_index: 0, purpose: 'initial', seed: 16402, noise_sha256: hash('noise') }] },
    deterministic_replay: { kind: 'non_deterministic', reason: 'not qualified' } } };
  const job = { id: 'job-one', status: 'completed', workerId: 'worker', assignedGpu: '1', result: { adapter: 'candle_sdxl', assets: [{ id: 'asset' }] } };
  const receipt = { version: 1, jobId: job.id, workerId: 'worker', attemptId: hash('attempt-one'), acceptedBackend: 'candle', acceptedDevice: '1', inferenceRevision: pin,
    acceptedSourceRoute: route, acceptedBindingSha256: hash('binding'), execution,
    outputs: [{ outputIndex: 0, artifactIds: ['asset'], artifactSha256: [hash(png)] }] };
  const input = { job, receipt, route, png, workerId: 'worker', device: '1', pin, scheduleSha256: digest(schedule), payloadSha256: hash('payload') };
  const plan = { version: 1, sourceSha: sha, inferenceRevision: pin, requestSha256: digest(REQUEST), sourceIdentitySha256: digest(sourceIdentity(route)), scheduleSha256: digest(schedule),
    deviceUuid: uuid, physicalIndex: 1, ownerGo: true, goExpiresAt: '2030-01-01T00:00:00Z',
    files: [{}], runtimeFiles: [{ path: 'private', sha256: hash('runtime') }] };
  return { input, plan };
}
function pair() {
  const { input, plan } = fixture();
  const first = completedWitness(input);
  input.job.id = input.receipt.jobId = 'job-two'; input.receipt.attemptId = hash('attempt-two');
  return { first, second: completedWitness(input), plan };
}
test('completed facts and distinct fresh outputs qualify; retries cannot reuse attempt identity', () => {
  const { first, second } = pair(); compareFresh(first, second);
  assert.throws(() => compareFresh(first, { ...second, attemptId: first.attemptId }), /reused completed attempt/);
  for (const key of ['pngSha256', 'executionSha256', 'payloadSha256', 'acceptedBindingSha256', 'sourceIdentitySha256', 'scheduleSha256', 'qualitySha256'])
    assert.throws(() => compareFresh(first, { ...second, [key]: hash('different') }), new RegExp(key));
});
test('completed witness rejects cancellation, fallback, missing facts and spoofed output association', () => {
  const mutations = [
    (x) => { x.job.status = 'canceled'; }, (x) => { x.job.workerId = 'other'; },
    (x) => { x.job.assignedGpu = '0'; }, (x) => { x.job.result.adapter = 'mlx_sdxl'; },
    (x) => { x.receipt.inferenceRevision = sha; }, (x) => { x.receipt.attemptId = ''; },
    (x) => { x.receipt.outputs[0].artifactIds = ['other']; },
    (x) => { x.receipt.outputs[0].outputIndex = 1; },
    (x) => { x.receipt.outputs.push(x.receipt.outputs[0]); },
    (x) => { x.receipt.outputs[0].stages = [{}]; },
    (x) => { x.png = Buffer.from('not PNG'); }, (x) => { x.png[9]++; },
    (x) => { x.receipt.execution.state = 'unavailable'; },
    (x) => { x.receipt.execution.identity.version = 2; },
    (x) => { delete x.receipt.execution.identity.deterministic_replay; },
    (x) => { x.receipt.execution.identity.sampling.value = ['euler']; },
    (x) => { x.receipt.execution.identity.schedule.value[0].schedule.count = 20; },
    (x) => { x.receipt.execution.identity.randomness.draws[0].seed++; },
    (x) => { x.receipt.execution.identity.geometry.state = 'unavailable'; },
    (x) => { x.receipt.execution.identity.quality_policy.value.compute = [{ device: 'Cpu' }]; },
  ];
  for (const mutate of mutations) { const { input } = fixture(); mutate(input); assert.throws(() => completedWitness(input)); }
});
test('source identity excludes only request binding and refuses extra adapter/provider routes', () => {
  const { input } = fixture(); const before = digest(sourceIdentity(input.route));
  input.route.models[0].authoredOptionsSha256 = hash('new project');
  assert.equal(digest(sourceIdentity(input.route)), before);
  input.route.models[0].members[0].role = 'lora'; assert.throws(() => sourceIdentity(input.route));
  input.route.models[0].members[0].role = 'primary'; input.route.expectedProvider.operation = 'image_edit';
  assert.throws(() => sourceIdentity(input.route));
});
test('real member bytes must match the exact private route and cache, before and after execution', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'sdxl-proof-members-'));
  try {
    const { input, plan } = fixture(); const file = path.join(root, 'model.bin'); await writeFile(file, 'abc');
    Object.assign(plan, { hub: root, files: [{ slot: 'modelManifestEntry', memberIndex: 0, relativePath: 'unet/model.bin', path: file }] });
    await verifyMembers(input.route, plan);
    await writeFile(file, 'abd'); await assert.rejects(verifyMembers(input.route, plan), /bytes changed/);
    await writeFile(file, 'abc'); plan.files.push(plan.files[0]);
    await assert.rejects(verifyMembers(input.route, plan), /cardinality/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
test('source, pin, owner window and separate reviewed declaration are mandatory', () => {
  const { plan } = fixture(); const now = Date.parse('2026-11-14'); validatePlan(plan, sha, pin, 'qualify', now);
  for (const mutation of [ { ownerGo: false }, { goExpiresAt: '2020-01-01' }, { sourceSha: pin }, { inferenceRevision: sha }, { requestSha256: hash('other') }, { scheduleSha256: '' }, { runtimeFiles: [] } ])
    assert.throws(() => validatePlan({ ...plan, ...mutation }, sha, pin, 'qualify', now));
  assert.throws(() => validatePlan(plan, sha, pin, 'replay', now));
  Object.assign(plan, { qualificationSha256: hash('qualification'), declaration: { scope: SCOPE, candidateSourceSha: sha, candidateInferenceRevision: pin, reviewUrl: 'https://github.com/SceneWorks/inference/pull/1' } });
  validatePlan(plan, sha, pin, 'replay', now);
});
test('qualification is immutable and cannot manufacture a deterministic support declaration', () => {
  const { first, second, plan } = pair();
  const qualification = { accepted: true, phase: 'qualify', deviceUuid: uuid, requestSha256: plan.requestSha256, sourceIdentitySha256: plan.sourceIdentitySha256, scheduleSha256: plan.scheduleSha256, jobs: [first, second] };
  plan.qualificationSha256 = digest(qualification); requireReplayQualification(plan, qualification);
  assert.equal(first.support.kind, 'non_deterministic');
  qualification.jobs[0].support.kind = 'deterministic';
  assert.throws(() => requireReplayQualification(plan, qualification), /evidence changed/);
});
test('physical ordinal zero, NVML worker label and final guard UUID must agree', () => {
  const { plan } = fixture();
  const metadata = { context_created: false, devices: [{ ordinal: 0, uuid }] };
  verifyDeviceSelection(metadata, 1, plan);
  assert.throws(() => verifyDeviceSelection(metadata, 0, plan));
  assert.throws(() => verifyDeviceSelection({ ...metadata, devices: [{ ordinal: 1, uuid }] }, 1, plan));
  const evidence = { bodyAccepted: true, deviceUuid: uuid, physicalIndex: 1 };
  const resource = { resource_completed: true, binding: { kind: 'context-free-cuda-ordinal-uuid', devices: [{ index: 1, uuid }], mapping: [{ cuda_ordinal: 0, uuid, nvml_index: 1 }] } };
  assert.equal(finalizeEvidence(evidence, resource).accepted, true);
  for (const wrong of [{ ...resource, resource_completed: false }, { ...resource, binding: { ...resource.binding, mapping: [{ cuda_ordinal: 0, uuid, nvml_index: 0 }] } }])
    assert.throws(() => finalizeEvidence({ ...evidence }, wrong));
});
test('task DB uses supported read-only SQLite and cannot mutate worker/receipt records', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'sdxl-proof-sqlite-'));
  try {
    const file = path.join(root, 'jobs.db'); const writer = new DatabaseSync(file);
    writer.exec('create table jobs (id text, execution_receipt_json text)');
    writer.prepare('insert into jobs values (?,?)').run('one', '{"version":1}'); writer.close();
    const before = hash(await readFile(file)); const reader = new DatabaseSync(file, { readOnly: true });
    assert.equal(reader.prepare('select execution_receipt_json from jobs where id=?').get('one').execution_receipt_json, '{"version":1}');
    assert.throws(() => reader.exec('delete from jobs'), /readonly/); reader.close();
    assert.equal(hash(await readFile(file)), before);
  } finally { await rm(root, { recursive: true, force: true }); }
});
test('workflow only dispatches manually and uploads bounded evidence, never private state', async () => {
  const workflow = await readFile(new URL('../../.github/workflows/server-candle-linux.yml', import.meta.url), 'utf8');
  assert.match(workflow, /on:\n  workflow_dispatch:/);
  assert.match(workflow, /  sdxl-exact-replay-prepare:\n    if:.*phase == 'prepare'.*ownerGo == false/);
  assert.match(workflow, /node scripts\/ci\/sdxl-exact-replay-prepare.mjs/);
  assert.match(workflow, /  sdxl-exact-replay:\n    if:.*github.event_name == 'workflow_dispatch'.*inputs.starvector_terminal_operation == 'standard'.*startsWith.*ownerGo == true/);
  assert.match(workflow, /  build-candle:\n    if:.*!startsWith/);
  assert.match(workflow, /cancel-in-progress:.*!startsWith/);
  assert.doesNotMatch(workflow, /options:.*sdxl-exact-replay/);
  assert.doesNotMatch(workflow, /\n  (push|pull_request|schedule|workflow_call):/);
  assert.match(workflow, /runs-on: \[self-hosted, Windows, X64, cuda, real-weights\]/);
  assert.match(workflow, /windows_cuda_execution.py --actor sdxl-exact-replay --budget-minutes 120 --cuda-ordinals 0/);
  assert.match(workflow, /path: \$\{\{ env.SDXL_REPLAY_OUTPUT \}\}\/evidence.json/);
});

test('metadata preparation cannot inherit a real-model approval or command input', () => {
  const marker = JSON.stringify({ proof: 'sdxl-exact-replay', phase: 'prepare', ownerGo: false });
  validatePreparationDispatch(marker);
  assert.throws(() => validatePreparationDispatch(JSON.stringify({ proof: 'sdxl-exact-replay', phase: 'prepare', ownerGo: true })));
  assert.throws(() => validatePreparationDispatch(JSON.stringify({ proof: 'sdxl-exact-replay', phase: 'prepare', ownerGo: false, command: 'run' })));
});

test('accepted queued request preserves the fixed controls before claims are released', () => {
  verifyQueuedPayload({ ...REQUEST, projectId: 'owned', modelManifestEntry: {} });
  assert.throws(() => verifyQueuedPayload({ ...REQUEST, advanced: { ...REQUEST.advanced, sampler: 'euler' } }));
  assert.throws(() => verifyQueuedPayload({ ...REQUEST, prompt: 'rewritten prompt' }));
});

test('replay candidate requires actual provider declaration and unchanged qualified bytes/identity', () => {
  const { first } = pair();
  assert.throws(() => verifyReplayCandidate(first, first), /declaration/);
  const candidate = { ...first, support: { kind: 'deterministic', scope: SCOPE } };
  verifyReplayCandidate(candidate, first);
  for (const key of ['pngSha256', 'executionWithoutSupportSha256', 'sourceIdentitySha256', 'scheduleSha256', 'qualitySha256'])
    assert.throws(() => verifyReplayCandidate({ ...candidate, [key]: hash('changed') }, first), /behavior changed/);
  assert.throws(() => verifyReplayCandidate({ ...candidate, support: { kind: 'deterministic', scope: 'all_devices' } }, first));
});

test('registered dispatcher accepts only the fixed proof fields, never a script or command', () => {
  const input = { proof: 'sdxl-exact-replay', phase: 'qualify', planPath: 'C:\\private\\sdxl-plan.json', deviceUuid: uuid, ownerGo: true };
  validateDispatch(input);
  assert.deepEqual(parseDispatch(JSON.stringify(input)), input);
  for (const raw of ['plain-campaign', '{"proof":"sdxl-exact-replay",', JSON.stringify(input, null, 2)]) assert.throws(() => parseDispatch(raw));
  for (const extra of [{ command: 'anything' }, { script: 'anything' }, { sourceRef: 'anything' }])
    assert.throws(() => validateDispatch({ ...input, ...extra }), /unknown dispatch fields/);
  for (const bad of [{ proof: 'other' }, { phase: 'other' }, { planPath: 'relative.json' }, { ownerGo: false }, { ownerGo: 'true' }])
    assert.throws(() => validateDispatch({ ...input, ...bad }));
});

test('standard marker selects no unrelated model job and malformed plans fail before execution', async () => {
  const workflow = await readFile(new URL('../../.github/workflows/server-candle-linux.yml', import.meta.url), 'utf8');
  const condition = (job) => workflow.slice(workflow.indexOf(`  ${job}:\n`)).split('\n').find((line) => line.startsWith('    if:')).trim().slice(8, -3);
  const evaluate = (job, plan, event = 'workflow_dispatch') => Function('github', 'inputs', 'fromJSON', 'startsWith', `return (${condition(job)});`)(
    { event_name: event, repository: 'SceneWorks/SceneWorks' }, { starvector_terminal_operation: 'standard', campaign_run_id: plan }, JSON.parse, (value, prefix) => String(value ?? '').toLowerCase().startsWith(prefix.toLowerCase()));
  const marker = JSON.stringify({ proof: 'sdxl-exact-replay', ownerGo: true });
  assert.equal(evaluate('sdxl-exact-replay', marker), true);
  assert.equal(evaluate('build-candle', marker), false);
  assert.equal(evaluate('h3-cuda-memory', marker), false);
  const noGo = JSON.stringify({ proof: 'sdxl-exact-replay', ownerGo: false });
  assert.equal(evaluate('sdxl-exact-replay', noGo), false);
  assert.equal(evaluate('build-candle', noGo), false);
  for (const plan of ['', '{}', '{"proof":"other"}', 'starvector-campaign-2026-11', '{', 'not-json']) {
    assert.equal(evaluate('sdxl-exact-replay', plan), false);
    assert.equal(evaluate('build-candle', plan), true);
    assert.equal(evaluate('h3-cuda-memory', plan), false);
  }
  for (const plan of ['{"proof":"sdxl-exact-replay",', '{"proof":"sdxl-exact-replay","ownerGo":']) {
    assert.throws(() => evaluate('sdxl-exact-replay', plan));
    assert.equal(evaluate('build-candle', plan), false);
    assert.equal(evaluate('h3-cuda-memory', plan), false);
  }
  for (const event of ['push', 'pull_request', 'schedule']) assert.equal(evaluate('sdxl-exact-replay', marker, event), false);
});
