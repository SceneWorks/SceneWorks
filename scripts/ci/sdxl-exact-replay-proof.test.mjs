// Metadata only: no service, provider, CUDA context or model is started by this suite.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { REQUEST, SCOPE, digest, sourceIdentity, parseDispatch, validateDispatch, validatePlan, verifyMembers,
  completedWitness, verifyQueuedPayload, verifyReplayCandidate, compareFresh, requireReplayQualification, verifyDeviceSelection,
  finalizeEvidence, finalizeArtifacts } from './sdxl-exact-replay-proof.mjs';
import { failureDiagnostic } from './sdxl-exact-replay-diagnostic.mjs';
import { validatePreparationDispatch } from './sdxl-exact-replay-prepare.mjs';
import { validateBindingDispatch, mapSelectedFiles } from './sdxl-exact-replay-bind.mjs';
import { selectProofDevice, validateCensusDispatch } from './sdxl-exact-replay-census.mjs';
import { boundedPlanWindow, validatePlanDispatch } from './sdxl-exact-replay-plan.mjs';
import { makeReplayPlan } from './sdxl-exact-replay-seal.mjs';

const hash = (s) => createHash('sha256').update(s).digest('hex');
const sha = 'a'.repeat(40), pin = 'b'.repeat(40), uuid = 'GPU-00000000-0000-0000-0000-000000000001';
const resolved = (value) => ({ state: 'resolved', value });
function fixture() {
  const png = Buffer.from([137,80,78,71,13,10,26,10,1,2,3]); // metadata stand-in, never rendered
  const route = { backend: 'candle', platform: 'windows', expectedProvider: {
    modelSlot: 'modelManifestEntry', modelId: 'sdxl', operation: 'image_generate', providerId: 'sdxl' },
  models: [{ schemaVersion: 1, slot: 'modelManifestEntry', modelId: 'sdxl', backend: 'candle',
    variant: 'base', tier: '4', authoredOptionsSha256: hash('authored'), members: [{ role: 'primary',
      componentId: 'unet', source: { kind: 'digest', sha256: `sha256:${hash('source')}` }, files: [
        { relativePath: 'unet/model.bin', sizeBytes: 3, sha256: `sha256:${hash('abc')}` } ] }] }] };
  const schedule = [{ sampler: 'ddim', schedule: { element: 'f32_le_bits', count: 21, sha256: hash('sigmas') }, estimator: 'eps' }];
  const execution = { state: 'resolved', identity: { version: 1, provider_id: 'sdxl', backend: 'candle', runtime_id: 'candle-execution-v1', output_units: 1,
    sampling: resolved(['ddim']), schedule: resolved(schedule), estimator: resolved('eps'),
    geometry: resolved([{ width: 1024, height: 1024, channels: 3, format: 'rgb8' }]),
    quality_policy: resolved({ compute: [{ device: 'Cuda(0)', dtype: 'F16' }], operations: [] }), output_policy: resolved(['rgb8']),
    randomness: { kind: 'draws', scheme: 'candle-ordered-provider-streams-v1:rand-0.9-StdRng-rand_distr-StandardNormal-f32',
      draws: [{ output_index: 0, purpose: 'sdxl/pipeline', seed: 16402, noise_sha256: null }] },
    deterministic_replay: { kind: 'non_deterministic', reason: 'not qualified' } } };
  const job = { id: 'job-one', status: 'completed', workerId: 'worker', assignedGpu: '1', result: { adapter: 'candle_sdxl', assets: [{ id: 'asset' }] } };
  const receipt = { version: 1, jobId: job.id, workerId: 'worker', attemptId: hash('attempt-one'), acceptedBackend: 'candle', acceptedDevice: '1', inferenceRevision: pin,
    acceptedSourceRoute: route, acceptedBindingSha256: hash('binding'), execution,
    outputs: [{ outputIndex: 0, artifactIds: ['asset'], artifactSha256: [hash(png)] }] };
  const input = { job, receipt, route, png, workerId: 'worker', device: '1', pin, scheduleSha256: digest(schedule), payloadSha256: hash('payload') };
  const plan = { version: 1, sourceSha: sha, inferenceRevision: pin, requestSha256: digest(REQUEST), sourceIdentitySha256: digest(sourceIdentity(route)), scheduleSha256: digest(schedule),
    deviceUuid: uuid, physicalIndex: 1, ownerGo: true,
    goStartsAt: '2026-11-13T23:50:00Z', goExpiresAt: '2026-11-14T01:50:00Z',
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
    (x) => { x.receipt.execution.identity.randomness.scheme = 'unverified-rng'; },
    (x) => { x.receipt.execution.identity.randomness.draws[0].purpose = 'other'; },
    (x) => { x.receipt.execution.identity.geometry.state = 'unavailable'; },
    (x) => { x.receipt.execution.identity.quality_policy.value.compute = [{ device: 'Cpu' }]; },
  ];
  for (const mutate of mutations) { const { input } = fixture(); mutate(input); assert.throws(() => completedWitness(input)); }
});
test('source identity excludes only request binding and refuses extra adapter/provider routes', () => {
  const { input } = fixture(); const before = digest(sourceIdentity(input.route));
  input.route.models[0].authoredOptionsSha256 = hash('new project');
  assert.equal(digest(sourceIdentity(input.route)), before);
  input.route.models[0].members[0].files[0].sha256 = hash('abc');
  assert.throws(() => sourceIdentity(input.route), /invalid member digest/);
  input.route.models[0].members[0].files[0].sha256 = `sha256:${hash('abc')}`;
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
  for (const mutation of [ { ownerGo: false }, { goStartsAt: '2026-11-14T00:01:00Z' },
    { goExpiresAt: '2020-01-01' }, { goExpiresAt: '2026-11-14T02:01:00Z' },
    { sourceSha: pin }, { inferenceRevision: sha }, { requestSha256: hash('other') },
    { scheduleSha256: '' }, { runtimeFiles: [] }, { physicalIndex: 0 } ])
    assert.throws(() => validatePlan({ ...plan, ...mutation }, sha, pin, 'qualify', now));
  assert.throws(() => validatePlan(plan, sha, pin, 'replay', now));
  Object.assign(plan, { qualificationSha256: hash('qualification'), declaration: { scope: SCOPE, candidateSourceSha: sha, candidateInferenceRevision: pin, reviewUrl: 'https://github.com/SceneWorks/inference/pull/1' } });
  validatePlan(plan, sha, pin, 'replay', now);
});
test('qualification is immutable and cannot manufacture a deterministic support declaration', () => {
  const { first, second, plan } = pair();
  const qualification = { accepted: true, phase: 'qualify', deviceUuid: uuid, requestSha256: plan.requestSha256, sourceIdentitySha256: plan.sourceIdentitySha256,
    scheduleSha256: plan.scheduleSha256, loadedRuntimeLibrariesSha256: hash('loaded-runtime'), jobs: [first, second] };
  plan.qualificationSha256 = digest(qualification); requireReplayQualification(plan, qualification);
  assert.equal(first.support.kind, 'non_deterministic');
  qualification.jobs[0].support.kind = 'deterministic';
  assert.throws(() => requireReplayQualification(plan, qualification), /evidence changed/);
});
test('a completed deterministic qualification seals a separate bounded replay plan', () => {
  const { first, second, plan } = pair();
  const support = { kind: 'deterministic', scope: SCOPE };
  const qualification = { accepted: true, phase: 'qualify', sourceSha: sha, inferenceRevision: pin,
    deviceUuid: uuid, physicalIndex: 1, requestSha256: plan.requestSha256,
    sourceIdentitySha256: plan.sourceIdentitySha256, scheduleSha256: plan.scheduleSha256,
    loadedRuntimeLibrariesSha256: hash('loaded-runtime'),
    jobs: [{ ...first, support }, { ...second, support }] };
  const now = Date.parse('2026-11-14T01:00:00Z');
  const replayPlan = makeReplayPlan(plan, qualification, sha, pin, now);
  assert.equal(replayPlan.qualificationSha256, digest(qualification));
  assert.equal(replayPlan.declaration.reviewUrl, 'https://github.com/SceneWorks/inference/pull/1464');
  assert.equal(Date.parse(replayPlan.goExpiresAt) - Date.parse(replayPlan.goStartsAt), 120 * 60 * 1000);
  assert.throws(() => makeReplayPlan(plan, { ...qualification, accepted: false }, sha, pin, now));
  assert.throws(() => makeReplayPlan(plan, { ...qualification, physicalIndex: 0 }, sha, pin, now), /physical GPU changed/);
  assert.throws(() => makeReplayPlan(plan, { ...qualification, jobs: [first, second] }, sha, pin, now),
    /provider did not declare/);
});
test('logical ordinal zero, physical GPU1 label and final guard UUID must agree', () => {
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
test('failure summaries allow only fixed stages, error classes and bounded resource metadata', () => {
  const privateText = 'C:\\private\\model.bin pid=12345 secret';
  const resource = { schema_version: 1, kind: 'cuda-execution-resource-lease-v1', actor: 'sdxl-exact-replay',
    resource_completed: false, error: `TimeoutError: ${privateText}`, lock_wait_seconds: 1.23456,
    child: { timeout: true, stdout: privateText, pid: 12345 }, argv: [privateText], binding: { privateText } };
  const report = failureDiagnostic({ failureStage: 'render', failureErrorClass: 'Error', failureCheck: privateText, jobs: [privateText] }, resource);
  assert.deepEqual(report, { version: 1, kind: 'sdxl-exact-replay-failure', accepted: false, bodyAccepted: false,
    proof: { status: 'available', stage: 'render', errorClass: 'Error' },
    resource: { status: 'available', stage: 'child-result-recorded', errorClass: 'TimeoutError',
      lockWaitSeconds: 1.235, childTimeout: true, resource_completed: false } });
  assert.doesNotMatch(JSON.stringify(report), /private|model|12345|secret|stdout|argv/);
  const unknown = failureDiagnostic({ failureStage: privateText, failureErrorClass: privateText }, { ...resource, error: privateText });
  assert.equal(unknown.proof.stage, 'unknown'); assert.equal(unknown.proof.errorClass, 'unknown'); assert.equal(unknown.resource.errorClass, 'unknown');
  for (const wait of ['123', -1, 86401, Infinity, NaN, {}, null]) {
    assert.equal(failureDiagnostic(null, { ...resource, lock_wait_seconds: wait }).resource.lockWaitSeconds, null);
  }
  assert.equal(failureDiagnostic(null, { ...resource, child: { timeout: 'true' } }).resource.childTimeout, null);
  assert.equal(failureDiagnostic(null, { ...resource, error: `resource release failed: OSError: ${privateText}` }).resource.stage, 'resource-release');
  assert.deepEqual(failureDiagnostic(null, { ...resource, actor: 'other' }).resource, { status: 'unavailable' });
});
test('finalize writes a public refusal after a guard timeout even without child evidence', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'sdxl-finalize-'));
  try {
    const output = path.join(root, 'output'), resourceFile = path.join(root, 'resource.json');
    const resource = { schema_version: 1, kind: 'cuda-execution-resource-lease-v1', actor: 'sdxl-exact-replay',
      resource_completed: false, lock_wait_seconds: 2, child: { timeout: true, pid: 12345 } };
    await writeFile(resourceFile, JSON.stringify(resource));
    await assert.rejects(finalizeArtifacts(output, resourceFile), /did not complete/);
    const file = path.join(output, 'evidence.json');
    const report = JSON.parse(await readFile(file, 'utf8'));
    assert.equal(report.accepted, false); assert.equal(report.bodyAccepted, false);
    assert.deepEqual(report.proof, { status: 'unavailable' });
    assert.equal(report.resource.childTimeout, true);
    assert.doesNotMatch(JSON.stringify(report), /12345|pid/);
    for (const content of ['{malformed', ' '.repeat(1024 * 1024 + 1), 'null', '[]']) {
      await writeFile(file, content); await writeFile(resourceFile, content);
      await assert.rejects(finalizeArtifacts(output, resourceFile));
      const invalid = JSON.parse(await readFile(file, 'utf8'));
      assert.deepEqual(invalid.proof, { status: 'unavailable' });
      assert.deepEqual(invalid.resource, { status: 'unavailable' });
      assert.equal(invalid.accepted, false);
    }
    await rm(file); await rm(resourceFile);
    await assert.rejects(finalizeArtifacts(output, resourceFile));
    assert.equal(JSON.parse(await readFile(file, 'utf8')).accepted, false);
  } finally { await rm(root, { recursive: true, force: true }); }
});
test('finalize retains valid acceptance and sanitizes failed existing proof evidence', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'sdxl-finalize-valid-'));
  try {
    const file = path.join(root, 'evidence.json'), resourceFile = path.join(root, 'resource.json');
    const evidence = { bodyAccepted: true, deviceUuid: uuid, physicalIndex: 1 };
    const resource = { resource_completed: true, binding: { kind: 'context-free-cuda-ordinal-uuid',
      devices: [{ index: 1, uuid }], mapping: [{ cuda_ordinal: 0, uuid, nvml_index: 1 }] } };
    await writeFile(file, JSON.stringify(evidence)); await writeFile(resourceFile, JSON.stringify(resource));
    assert.equal((await finalizeArtifacts(root, resourceFile)).accepted, true);
    assert.equal(JSON.parse(await readFile(file, 'utf8')).resourceReceiptSha256, digest(resource));
    await writeFile(file, JSON.stringify({ ...evidence, bodyAccepted: false, failureStage: 'completed_identity',
      failureErrorClass: 'AssertionError', failureCheck: 'C:/private/model.bin', jobs: ['secret'] }));
    await assert.rejects(finalizeArtifacts(root, resourceFile));
    const failure = JSON.parse(await readFile(file, 'utf8'));
    assert.equal(failure.accepted, false); assert.equal(failure.proof.stage, 'completed_identity');
    assert.equal(failure.proof.errorClass, 'AssertionError');
    assert.doesNotMatch(JSON.stringify(failure), /private|model.bin|secret/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
test('workflow only dispatches manually and uploads bounded evidence, never private state', async () => {
  const workflow = await readFile(new URL('../../.github/workflows/server-candle-linux.yml', import.meta.url), 'utf8');
  assert.match(workflow, /on:\n  workflow_dispatch:/);
  assert.match(workflow, /  sdxl-exact-replay-prepare:\n    if:.*phase == 'prepare'.*ownerGo == false/);
  assert.match(workflow, /node scripts\/ci\/sdxl-exact-replay-prepare.mjs/);
  assert.match(workflow, /  sdxl-exact-replay-bind:\n    if:.*phase == 'bind'.*phase == 'census'.*ownerGo == false/);
  assert.match(workflow, /node scripts\/ci\/sdxl-exact-replay-bind.mjs/);
  assert.match(workflow, /node scripts\/ci\/sdxl-exact-replay-census.mjs/);
  assert.match(workflow, /  sdxl-exact-replay-plan:\n    if:.*phase == 'plan'.*ownerGo == true/);
  assert.match(workflow, /node scripts\/ci\/sdxl-exact-replay-plan.mjs/);
  assert.match(workflow, /  sdxl-exact-replay:\n    if:.*github.event_name == 'workflow_dispatch'.*inputs.starvector_terminal_operation == 'standard'.*startsWith.*phase == 'qualify'.*phase == 'replay'.*ownerGo == true/);
  assert.match(workflow, /  build-candle:\n    if:.*!startsWith/);
  assert.match(workflow, /cancel-in-progress:.*!startsWith/);
  assert.doesNotMatch(workflow, /options:.*sdxl-exact-replay/);
  assert.doesNotMatch(workflow, /\n  (push|pull_request|schedule|workflow_call):/);
  assert.match(workflow, /runs-on: \[self-hosted, Windows, X64, cuda, real-weights, cuda-windows-2\]/);
  assert.match(workflow, /windows_cuda_execution.py --actor sdxl-exact-replay --budget-minutes 30 --min-free-mib 16384 --require-physical-index 1 --cuda-ordinals 0/);
  assert(workflow.indexOf('sdxl-exact-replay-proof.mjs finalize') < workflow.indexOf('sdxl-exact-replay-seal.mjs'));
  assert.match(workflow, /name: Seal the reviewed qualification into a private replay plan\n        if: \$\{\{ success\(\) && fromJSON\(inputs.campaign_run_id\).phase == 'qualify' \}\}/);
  assert.match(workflow, /path: \$\{\{ env.SDXL_REPLAY_SEAL_REPORT \}\}/);
  assert.equal((workflow.match(/\$targetDirectory = \(cargo metadata --format-version 1 --no-deps \| ConvertFrom-Json\)\.target_directory/g) ?? []).length, 3);
  assert.equal((workflow.match(/\$env:SDXL_REPLAY_BINARY = Join-Path \$targetDirectory 'release\\sceneworks-rust-api\.exe'/g) ?? []).length, 3);
  assert.equal((workflow.match(/Compiled SDXL API binary missing/g) ?? []).length, 3);
  assert.match(workflow, /path: \$\{\{ env.SDXL_REPLAY_OUTPUT \}\}\/evidence.json/);
});

test('metadata preparation cannot inherit a real-model approval or command input', () => {
  const marker = JSON.stringify({ proof: 'sdxl-exact-replay', phase: 'prepare', ownerGo: false });
  validatePreparationDispatch(marker);
  assert.throws(() => validatePreparationDispatch(JSON.stringify({ proof: 'sdxl-exact-replay', phase: 'prepare', ownerGo: true })));
  assert.throws(() => validatePreparationDispatch(JSON.stringify({ proof: 'sdxl-exact-replay', phase: 'prepare', ownerGo: false, command: 'run' })));
});

test('metadata binding accepts only a private path and maps the actual cached member bytes', async () => {
  const marker = JSON.stringify({ proof: 'sdxl-exact-replay', phase: 'bind', preparationPath: 'C:\\private\\record.json', ownerGo: false });
  validateBindingDispatch(marker);
  assert.throws(() => validateBindingDispatch(JSON.stringify({ proof: 'sdxl-exact-replay', phase: 'bind', preparationPath: 'record.json', ownerGo: false })));
  assert.throws(() => validateBindingDispatch(JSON.stringify({ proof: 'sdxl-exact-replay', phase: 'bind', preparationPath: 'C:\\private\\record.json', ownerGo: true })));
  const hub = await mkdtemp(path.join(os.tmpdir(), 'sdxl-proof-binding-'));
  try {
    const root = path.join(hub, 'models--sdxl', 'snapshots', 'revision');
    await mkdir(path.join(root, 'unet'), { recursive: true });
    await writeFile(path.join(root, 'unet', 'model.bin'), 'abc');
    const { input } = fixture();
    const files = await mapSelectedFiles(input.route, hub, [root]);
    assert.equal(files.length, 1);
    await writeFile(path.join(root, 'unet', 'model.bin'), 'abd');
    await assert.rejects(mapSelectedFiles(input.route, hub, [root]), /absent/);
  } finally { await rm(hub, { recursive: true, force: true }); }
});

test('idle-worker census accepts only a private binding and no model approval', () => {
  const marker = JSON.stringify({ proof: 'sdxl-exact-replay', phase: 'census', bindingPath: 'C:\\private\\binding.json', ownerGo: false });
  assert.deepEqual(validateCensusDispatch(marker), JSON.parse(marker));
  for (const changed of [{ ownerGo: true }, { bindingPath: 'binding.json' }, { command: 'run' }, { phase: 'qualify' }]) {
    assert.throws(() => validateCensusDispatch(JSON.stringify({ ...JSON.parse(marker), ...changed })));
  }
});

test('metadata census and plan select only the approved physical GPU1 UUID', () => {
  const gpu0 = { physicalIndex: 0, uuid: 'GPU-00000000-0000-0000-0000-000000000000' };
  const gpu1 = { physicalIndex: 1, uuid };
  assert.deepEqual(selectProofDevice([gpu1, gpu0]), gpu1);
  assert.throws(() => selectProofDevice([gpu0]), /index one unavailable/);
  assert.throws(() => selectProofDevice([gpu1, gpu1]), /index one unavailable/);
  assert.throws(() => selectProofDevice([{ ...gpu1, uuid: 'bad' }]), /UUID invalid/);
});

test('private plan derives one bounded execution window from standing owner go', () => {
  const input = { proof: 'sdxl-exact-replay', phase: 'plan', censusPath: 'C:\\private\\census.json',
    deviceUuid: uuid, ownerGo: true };
  assert.deepEqual(validatePlanDispatch(JSON.stringify(input)), input);
  const window = boundedPlanWindow(Date.parse('2026-11-17T12:00:00Z'));
  assert.deepEqual(window, { goStartsAt: '2026-11-17T12:00:00.000Z', goExpiresAt: '2026-11-17T14:00:00.000Z' });
  assert.throws(() => boundedPlanWindow(NaN), /invalid plan clock/);
  for (const change of [{ ownerGo: false }, { censusPath: 'census.json' },
    { goStartsAt: '2026-11-17T12:00:00Z' }, { goExpiresAt: '2026-11-17T14:00:00Z' },
    { command: 'run' }]) {
    assert.throws(() => validatePlanDispatch(JSON.stringify({ ...input, ...change })));
  }
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
  const marker = JSON.stringify({ proof: 'sdxl-exact-replay', phase: 'qualify', ownerGo: true });
  assert.equal(evaluate('sdxl-exact-replay', marker), true);
  assert.equal(evaluate('sdxl-exact-replay-plan', marker), false);
  assert.equal(evaluate('build-candle', marker), false);
  assert.equal(evaluate('h3-cuda-memory', marker), false);
  const approvedPlan = JSON.stringify({ proof: 'sdxl-exact-replay', phase: 'plan', ownerGo: true });
  assert.equal(evaluate('sdxl-exact-replay-plan', approvedPlan), true);
  assert.equal(evaluate('sdxl-exact-replay', approvedPlan), false);
  assert.equal(evaluate('build-candle', approvedPlan), false);
  const noGo = JSON.stringify({ proof: 'sdxl-exact-replay', phase: 'qualify', ownerGo: false });
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
