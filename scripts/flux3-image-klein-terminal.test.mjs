// All evidence in this suite is DECLARED SYNTHETIC. No models, downloads, real services or GPUs.
import assert from 'node:assert/strict';
import test from 'node:test';
import { EventEmitter } from 'node:events';
import { mkdtemp, mkdir, readFile, writeFile, rm, realpath } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { deflateSync } from 'node:zlib';
import { loadProductTargets, terminalCatalog as terminalPlan, terminalPlan as representativePlan, targetRegistry, imageRequest, requestedDiskBudget, SCENE_PROMPT, REFERENCE_PROMPT } from './flux3-image-klein-terminal-plan.mjs';
import { launchTerminalSession, offlineChildEnv, validateAdmission, guardHardware } from './flux3-image-klein-terminal-session.mjs';
import { assertSavedAsset, ownedApiController, runnerHfCredential, authorizeFlux3Repository, redactSensitiveText } from './flux3-image-klein-app-lifecycle.mjs';
import { assertImageObservations, executeCell, trainingSequence, retainImage, waitJob, sha256, objectHash, Blocked, runTerminal, replayRequest, assertCheckpoint, pngCrc32, assertFinishedPng, provisioningApproval, provisionCell, importAdapter, ownedNegativeProbe, uploadBinding, assertPackage, boundAdapterSpec } from './flux3-image-klein-terminal-driver.mjs';
import { initializeLedger, planHash, auditLedger } from './flux3-image-klein-terminal-budget.mjs';
const source = 'a'.repeat(40), inference = 'b'.repeat(40), tip = 'c'.repeat(40), digest = 'd'.repeat(64);
const sources = await loadProductTargets(), registry = targetRegistry(sources), plan = terminalPlan(sources);
const find = (scenario, model = 'base', backend = 'candle') => plan.cases.find(c => c.caseId === `${backend}.${model}.${scenario}.bf16`);
const target = cell => registry.find(t => t.backend === cell.backend && t.modelId === cell.modelId);
const report = () => ({ events: [], requests: [], jobs: [], outputs: [], adapters: [], observations: [], timeoutSeconds: 10 });
async function temporary(t) { const root = await mkdtemp(path.join(os.tmpdir(), 's10-terminal-declared-synthetic-')); t.after(() => rm(root, { recursive: true, force: true })); return root; }
function admission(root) { return { actualRun: true, backend: 'candle', admissionId: 'parent-fixture-only', approvedBy: 'declared synthetic', phases: plan.phases, sourceSha: source, inferenceMainSha: inference, inferenceMainTipSha: tip, inferenceRepository: root, binary: path.join(root, 'api.exe'), binarySha256: digest, binarySourceSha: source, binaryBuildReceipt: { root, path: path.join(root, 'producer-receipt.json'), sha256: digest }, outputRoot: path.join(root, 'output'), hfHome: path.join(root, 'hf'), runner: 'fixture', runnerAccount: os.userInfo().username, physicalGpuId: '1', cudaVisibleDevices: '1', logicalCudaOrdinal: 0, gpuUuid: 'GPU-1234', timeoutSeconds: 60 }; }
async function approvedFixture(root, a, inputSpec) {
  const approval = { schemaVersion: 1, kind: 'flux3-terminal-plan-approval', approvalId: 'declared-fixture-only', userPlanApprovalId: 'SYNTHETIC_NOT_EXECUTION_AUTHORIZATION', approvedBy: 'CPU fixture', sourceSha: source, inferenceMainSha: inference, ledgerId: 'fixture-terminal-ledger', ledgerRoots: { mlx: path.join(root, 'ledger-mlx'), candle: path.join(root, 'ledger-candle') }, inputSpecSha256: { mlx: objectHash(inputSpec), candle: objectHash(inputSpec) }, plan: representativePlan(sources) };
  Object.assign(a, { ledgerRoot: approval.ledgerRoots[a.backend], ledgerId: approval.ledgerId, approvedPlanSha256: planHash(approval) }); await initializeLedger({ approval, backend: a.backend, ledgerRoot: a.ledgerRoot }); return approval;
}
function child(pid) { const c = new EventEmitter(); Object.assign(c, { pid, exitCode: null, signalCode: null, stdin: { end() { c.exitCode = 0; c.emit('exit', 0, null); } }, kill() { c.signalCode = 'SIGKILL'; c.emit('exit', null, 'SIGKILL'); } }); return c; }
function ready(state) { return { status: 'ok', service: 'sceneworks-api', runtime: 'rust', authRequired: false, readiness: { status: 'ready' }, directories: { data: path.join(state, 'data'), config: path.join(state, 'config'), projects: path.join(state, 'data/projects'), jobsDb: path.join(state, 'data/cache/jobs.db') } }; }
function savedFixture(body, root, workerId = 'w') {
  const sidecar = { schemaVersion: 1, id: 'asset', projectId: body.projectId, type: 'image', generationSetId: 'set', lineage: { jobId: 'image-job' }, file: { path: 'assets/images/image-job.png', mimeType: 'image/png', width: body.width, height: body.height }, recipe: { model: body.model, mode: body.mode, prompt: body.prompt, adapter: 'candle_flux3', seed: body.seed, loras: body.loras, normalizedSettings: { family: 'flux3', width: body.width, height: body.height, count: 1 }, rawAdapterSettings: { ...body.advanced, realModelInference: true, numInferenceSteps: body.advanced.steps, repo: target(find('t2i')).downloads[0].repo } } };
  const generationSet = { schemaVersion: 1, id: 'set', projectId: body.projectId, jobId: 'image-job', mode: body.mode, model: body.model, prompt: body.prompt, negativePrompt: '', count: 1, createdAt: 'declared synthetic timestamp', recipe: structuredClone(sidecar.recipe) };
  const asset = { ...structuredClone(sidecar), url: `/api/v1/projects/${body.projectId}/files/${sidecar.file.path}`, sidecarPath: 'assets/images/image-job.sceneworks.json', generationSet };
  const job = { id: 'image-job', type: 'image_generate', projectId: body.projectId, payload: body, status: 'completed', stage: 'completed', workerId, backend: '1', result: { model: body.model, adapter: 'candle_flux3', generationSetId: 'set', assetIds: ['asset'], assets: [sidecar] } };
  const chunk = (name, data) => { const nameAndData = Buffer.concat([Buffer.from(name), data]), prefix = Buffer.alloc(4), crc = Buffer.alloc(4); prefix.writeUInt32BE(data.length); crc.writeUInt32BE(pngCrc32(nameAndData)); return Buffer.concat([prefix, nameAndData, crc]); };
  const header = Buffer.alloc(13); header.writeUInt32BE(body.width); header.writeUInt32BE(body.height, 4); header[8] = 8; header[9] = 2;
  const pixels = Buffer.alloc((body.width * 3 + 1) * body.height); pixels[1] = body.loras.some(l => l.weight > 0) ? 2 : 1;
  const png = Buffer.concat([Buffer.from('89504e470d0a1a0a', 'hex'), chunk('IHDR', header), chunk('IDAT', deflateSync(pixels)), chunk('IEND', Buffer.alloc(0))]);
  return { sidecar, asset, job, png, project: { id: body.projectId, path: root } };
}

test('planner is pure and covers both checkpoints/backends, all explicit obligations without PASS', () => {
  assert.equal(new Set(plan.cases.map(c => c.caseId)).size, plan.cases.length);
  for (const backend of ['mlx', 'candle']) for (const model of ['base', 'distilled']) {
    for (const scenario of ['t2i', 'refs-1', 'refs-10', 'layout-placement', 'box-move-remove', 'pup-t2i', 'pup-i2i', 'pup-t2i_bbox', 'pup-i2i_bbox', 'lora-zero', 'lora-on', 'portrait', 'landscape', 'max-resolution', 'cancel', 'saved-settings-repro', 'install-missing', 'authentication-denied', 'bad-reference', 'reference-count', 'bad-box', 'checkpoint-compatibility']) assert(find(scenario, model, backend), `${backend}/${model}/${scenario}`);
    for (const tier of ['q4', 'q8', ...(backend === 'candle' ? ['fp8r', 'nvfp4'] : [])]) assert(plan.cases.some(c => c.caseId === `${backend}.${model}.native-tier.${tier}`));
    if (model === 'base') for (const mode of ['captioned', 'edit']) assert(find(`train-interrupt-resume-export-${mode}`, model, backend));
  }
  assert(plan.cases.every(c => ['NOT_RUN', 'BLOCKED'].includes(c.status)));
  // Base q4/q8 now have production rows pinned to genuine local conversions: runnable-but-unrun,
  // not BLOCKED and never PASS.
  assert.deepEqual(plan.cases.filter(c => c.status === 'BLOCKED').map(c => c.caseId), []);
  for (const caseId of ['mlx.base.native-tier.q4', 'mlx.base.native-tier.q8', 'candle.base.native-tier.q4', 'candle.base.native-tier.q8']) {
    const cell = plan.cases.find(c => c.caseId === caseId); assert.equal(cell?.status, 'NOT_RUN', caseId); assert.equal(cell.blocker, null);
  }
  assert(!plan.cases.some(c => c.backend === 'mlx' && ['fp8r', 'nvfp4'].includes(c.tier)));
});
test('payload uses shipped defaults and actual quantTier/enhancePrompt/ordered-layout carriers', () => {
  const cell = find('box-move-remove'), inputs = { 'reference-0': { assetId: 'owned-1' } }, body = imageRequest(cell, target(cell), 'p', inputs);
  assert.equal(body.advanced.steps, 50); assert.equal(body.advanced.guidanceScale, 3.5); assert.equal(body.advanced.quantTier, 'bf16'); assert.equal(body.advanced.enhancePrompt, false);
  assert.deepEqual(body.referenceAssetIds, ['owned-1']); assert.deepEqual(body.advanced.flux3_reference_ids, ['slot-0']); assert.equal(body.advanced.flux3_layout.boxes[1].tgt_bbox, null);
  const ten = find('refs-10'); const tenInputs = Object.fromEntries(ten.inputBindings.map(b => [b.binding, { assetId: `a-${b.index}` }]));
  assert.deepEqual(imageRequest(ten, target(ten), 'p', tenInputs).referenceAssetIds, Array.from({ length: 10 }, (_, i) => `a-${i}`));
  for (const scale of [0, 0.75]) { const c = find(scale ? 'lora-on' : 'lora-zero'); assert.deepEqual(imageRequest(c, target(c), 'p', {}, { id: 'trusted', name: 'Real adapter' }).loras, [{ id: 'trusted', name: 'Real adapter', weight: scale }]); }
  assert.throws(() => imageRequest(find('refs-1'), target(cell), 'p', {}), /missing/);
});
test('disk budget deduplicates whole repos, adds optional PUP/conversion/build/state; no invented sizes', () => {
  const cells = [find('t2i'), find('pup-t2i'), find('t2i', 'distilled'), plan.cases.find(c => c.caseId === 'candle.distilled.native-tier.q4')];
  const disk = requestedDiskBudget(registry, cells, { freeBytes: 1e12, conversionOutputBytes: { 'flux3_klein:candle:q4': 5e9 } });
  assert.equal(disk.repositoryBytes, 43_317_649_602 + 43_229_093_855 + 18_839_746_673); assert.equal(disk.conversionBytes, 5e9); assert.equal(disk.buildBytes, 32 * 1024 ** 3); assert.equal(disk.stateBytes, 16 * 1024 ** 3);
  assert.throws(() => requestedDiskBudget(registry, cells, { freeBytes: 1e12 }), /conversion/);
  assert.throws(() => requestedDiskBudget(registry, cells, { freeBytes: disk.requiredBytes - 1, conversionOutputBytes: { 'flux3_klein:candle:q4': 5e9 } }), /disk admission/);
});
test('actual mode refuses absent approval/source/hash and checks inference MAIN ancestry, not labels', async t => {
  const root = await temporary(t), a = admission(root), calls = [];
  const git = async (cwd, ...args) => { calls.push([cwd, ...args]); if (args[0] === 'merge-base') return ''; if (args.includes('refs/remotes/origin/main')) return tip; if (args.includes('HEAD')) return source; return ''; };
  await validateAdmission(a, 'candle', { git, root: path.join(root, 'checkout') });
  assert(calls.some(call => call.includes('merge-base') && call.includes(inference) && call.includes('refs/remotes/origin/main')));
  for (const mutate of [a => a.actualRun = false, a => a.sourceSha = 'feature/sc-26233', a => a.inferenceMainSha = 'main', a => a.binarySourceSha = tip, a => a.physicalGpuId = '0', a => a.cudaVisibleDevices = '', a => a.outputRoot = path.join(root, 'checkout/output')]) { const bad = structuredClone(a); mutate(bad); await assert.rejects(validateAdmission(bad, 'candle', { git, root: path.join(root, 'checkout') })); }
  await assert.rejects(validateAdmission(a, 'candle', { root, git: async (cwd, ...args) => { if (args[0] === 'merge-base') throw new Error('feature-only commit is not an ancestor of MAIN'); return git(cwd, ...args); } }), /not an ancestor/);
});
test('child environment is offline and scrubbed; no inherited token/cache/backend/GPU leaks', () => {
  const a = admission('D:/synthetic'), identity = { state: 'D:/synthetic/nonce', workerId: 'owned' };
  const env = offlineChildEnv({ HF_TOKEN: 'DECLARED_SYNTHETIC_DO_NOT_USE', HF_TOKEN_PATH: 'ambient', API_KEY: 'ambient', SCENEWORKS_GPU_ID: '0', CUDA_VISIBLE_DEVICES: '', HF_ENDPOINT: 'https://foreign.invalid', PATH: 'safe' }, a, identity, 'http://127.0.0.1:12345');
  assert.equal(env.CUDA_VISIBLE_DEVICES, '1'); assert.equal(env.SCENEWORKS_GPU_ID, '1'); assert.equal(env.HF_TOKEN, undefined); assert.equal(env.HF_TOKEN_PATH, undefined); assert.equal(env.API_KEY, undefined); assert.equal(env.HF_ENDPOINT, undefined); assert.equal(env.HF_HUB_OFFLINE, '1'); assert.equal(env.SCENEWORKS_API_PORT, '12345');
});
test('GPU guard rejects foreign Runner.Worker/compute, UUID mismatch and actual worker on GPU0', async () => {
  const a = admission('D:/synthetic'), processTable = async () => [{ pid: process.pid, ppid: 0, command: 'node.exe' }, {pid:123,ppid:process.pid,command:'api.exe',startedAt:'2000-01-01T00:00:00Z',sessionId:1}];
  const exec = async (command, args) => { if (command === 'powershell.exe') return { stdout: `${a.gpuUuid} luid=0xdeadbeef_0x00000001` }; if (args.includes('--query-gpu=index,uuid,pci.bus_id')) return { stdout: `1,${a.gpuUuid},00000000:C1:00.0` }; if(args.includes('--query-gpu=memory.total,memory.free'))return{stdout:'96000,90000'}; if (args.includes('-x')) return { stdout: `<uuid>${a.gpuUuid}</uuid><processes><process_info><pid>123</pid><type>C</type></process_info></processes>` }; return { stdout: `123,${a.gpuUuid}` }; };
  const deps = { exec, platform: 'win32', processTable };
  assert.deepEqual((await guardHardware(a, [123], deps)).ownComputePids, [123]);
  await assert.rejects(guardHardware(a, [], deps), /JSON/); // Foreign C is measured, never categorically refused; this mock supplies no memory JSON.
  await assert.rejects(guardHardware(a, [123], { ...deps, processTable: async () => [...await processTable(), { pid: 99, ppid: 0, command: 'Runner.Worker.exe' }] }), /Runner.Worker/);
  await assert.rejects(guardHardware(a, [123], { ...deps, exec: async (c, args) => c === 'powershell.exe' ? { stdout: 'GPU-other' } : exec(c, args) }), /masked CUDA0/);
  await assert.rejects(guardHardware(a, [123], { ...deps, exec: async (c, args) => args.includes('--query-compute-apps=pid,gpu_uuid') ? { stdout: '123,GPU-0000' } : exec(c, args) }), /wrong physical GPU UUID/);
});
test('MLX guard requires coordinator exclusive lease in addition to physical secondary labels', async () => {
  const a = { ...admission('D:/synthetic'), backend: 'mlx', runner: 'nax-macos-2', runnerTarget: 'secondary', runnerLabel: 'rw-mage', hostName: 'nax-macos-2' }, table = [{ pid: process.pid, ppid: 0, command: 'node', startedAt: 'fixture-start' }];
  const lease = { kind: 'flux3-secondary-mac-exclusive-workload-lease', coordinatorApprovalId: 'SYNTHETIC_LEASE_ONLY', approvedBy: 'fixture', exclusiveGpuWorkload: true, noForeignGpuWorkload: true, admissionId: a.admissionId, sourceSha: a.sourceSha, inferenceMainSha: a.inferenceMainSha, runnerAccount: a.runnerAccount, hostName: a.hostName, observedAt: new Date(Date.now()-1000).toISOString(), expiresAt: new Date(Date.now()+120000).toISOString(), ancestorRunnerPids: [], baselineProcesses: table };
  const deps = { platform: 'darwin', arch: 'arm64', env: { RUNNER_NAME: 'nax-macos-2' }, exec: async () => ({ stdout: 'nax-macos-2.local' }), processTable: async () => table, leaseReader: async () => lease };
  await guardHardware(a, [], deps);
  await assert.rejects(guardHardware(a, [], { ...deps, env: { RUNNER_NAME: 'rw-starvector' } }));
  await assert.rejects(guardHardware({ ...a, runner: 'primary' }, [], deps));
  await assert.rejects(guardHardware(a, [], { ...deps, platform: 'win32' }));
  await assert.rejects(guardHardware(a, [], { ...deps, leaseReader: async () => ({ ...lease, expiresAt: new Date(0).toISOString() }) }), /lease expired/);
});
test('concrete launcher control flow starts only owned binary, resolves port-zero listener, restarts fresh worker and cleans children', async t => {
  const root = await temporary(t), a = admission(root), launched = [], calls = []; let workerId, guardCount = 0;
  const session = await launchTerminalSession(a, {
    fixtureOnly: true, skipPinReadForFixture: true, hash: async () => digest, mkdir: async () => {}, cp: async () => {}, open: async () => ({ fd: 42, close: async () => {} }),
    guard: async () => { guardCount++; return { ownComputePids: launched.map(c => c.pid) }; },
    readFile: async () => '{"event":"api_listening","address":"127.0.0.1:43123"}\n',
    spawn: (binary, args, options) => { calls.push({ binary, args, options }); workerId = options.env.SCENEWORKS_WORKER_ID; const c = child(100 + launched.length); launched.push(c); return c; },
    controller: ({ url, state }) => { assert.equal(url, 'http://127.0.0.1:43123'); assert(state.includes('state-')); return { verify: async () => true, startWorker: async fn => fn(), call: async () => [{ id: workerId, gpuId: '1', status: 'idle', capabilities: ['gpu', 'candle'] }], close: () => {} }; },
  });
  try { assert.equal(launched.length, 2); const previous = session.workerPid; await session.restartWorker(); assert.notEqual(session.workerPid, previous); assert.equal(launched[1].exitCode, 0); assert(session.identity.workerId.includes('resume')); assert(guardCount >= 3); await session.assertDeviceProof(); } finally { await session.close(); }
  assert(launched.every(c => c.exitCode === 0)); assert(calls.every(c => c.binary === a.binary && c.args.length === 0)); assert.equal(calls[0].options.env.SCENEWORKS_API_PORT, '0'); assert.equal(calls[1].options.env.SCENEWORKS_API_PORT, '43123'); assert.equal(calls[1].options.env.CUDA_VISIBLE_DEVICES, '1');
});
test('launcher refuses hash/guard before spawning and closes after readiness failure', async t => {
  const root = await temporary(t); let count = 0;
  await assert.rejects(launchTerminalSession(admission(root), { fixtureOnly: true, hash: async () => 'wrong', spawn: () => { count++; } }), /binary mismatch/); assert.equal(count, 0);
  await assert.rejects(launchTerminalSession(admission(root), { fixtureOnly: true, hash: async () => digest, skipPinReadForFixture: true, guard: async () => { throw new Error('foreign GPU'); }, spawn: () => { count++; } }), /foreign GPU/); assert.equal(count, 0);
});
test('negative API observations dispatch real public routes and do not accept status-only wrong rejection', async () => {
  for (const scenario of ['bad-reference', 'reference-count', 'bad-box']) {
    const cell = find(scenario), r = report(), calls = []; let paused = false, canceled = false;
    const s = { identity: { workerId: 'owned' }, wait: async fn => fn(), pauseWorker: async () => { paused = true; }, restartWorker: async () => { assert(paused); paused = false; return { pid: 2 }; }, failClosed: () => { assert(paused); }, endpoint: {
      call: async (method, route) => { assert(paused); if (route.startsWith('/api/v1/jobs?')) return []; if (route.endsWith('/cancel')) { canceled = true; return {}; } if (route === '/api/v1/jobs/bad') return { id: 'bad', status: canceled ? 'canceled' : 'queued', type: 'image_generate' }; throw new Error('unexpected negative fixture route'); },
      exchange: async (...args) => { assert(paused); calls.push(args); return { status: 400, body: { error: { 'bad-reference': 'reference 1 is not an asset of this project', 'reference-count': 'accepts 1–10 ordered references; request supplies 11', 'bad-box': 'layout needs positive YXYX boxes' }[scenario] } }; } } };
    await executeCell(s, cell, target(cell), { id: 'p' }, {}, {}, 'unused', r); assert.equal(calls[0][1], '/api/v1/image/jobs');
    if (scenario === 'reference-count') assert.equal(calls[0][2].referenceAssetIds.length, 11);
    s.endpoint.exchange = async () => ({ status: 201, body: { id: 'bad' } }); await assert.rejects(executeCell(s, cell, target(cell), { id: 'p' }, {}, {}, 'unused', report()), /unexpectedly accepted/); assert(canceled && paused, 'unexpected job is terminal-canceled before any worker resume');
  }
});
test('cancel requires owned running progress then product cancel and canceled terminal state', async () => {
  const cell = find('cancel'), r = report(), calls = []; let state = 0;
  const s = { identity: { workerId: 'w' }, wait: async fn => { for (let i = 0; i < 6; i++) { const j = await fn(); if (j) return j; } throw new Error('fixture exhausted'); }, endpoint: { call: async (method, route, body) => { calls.push([method, route]); if (route === '/api/v1/image/jobs') return { id: 'j', projectId: 'p', type: 'image_generate', payload: body }; if (route.endsWith('/cancel')) { state = 1; return {}; } if (route.endsWith('/assets')) return []; return { id: 'j', type: 'image_generate', workerId: 'w', status: state ? 'canceled' : 'running', progress: 0.125 }; } } };
  await executeCell(s, cell, target(cell), { id: 'p' }, {}, {}, 'unused', r);
  assert(calls.some(c => c[0] === 'POST' && c[1] === '/api/v1/jobs/j/cancel')); assert.equal(r.jobs[0].status, 'canceled'); assert(r.events.some(e => e.progress === 0.125));
});
test('foreign worker or early completion cannot manufacture progress/cancel proof', async () => {
  const s = { identity: { workerId: 'owned' }, wait: async fn => fn(), endpoint: { call: async () => ({ id: 'j', type: 'image_generate', workerId: 'foreign', status: 'running' }) } };
  await assert.rejects(waitJob(s, 'j', report()), /foreign worker/);
  s.endpoint.call = async () => ({ id: 'j', type: 'image_generate', workerId: 'owned', status: 'completed' });
  await assert.rejects(waitJob(s, 'j', report(), { until: j => j.status === 'running' }), /terminated before/);
});
test('retained proof uses GET library + actual sidecar + served PNG and rejects altered bytes/progress', async t => {
  const root = await temporary(t), cell = find('t2i'), body = imageRequest(cell, target(cell), 'p', {}), f = savedFixture(body, root);
  await mkdir(path.join(root, 'assets/images'), { recursive: true }); await writeFile(path.join(root, f.sidecar.file.path), f.png); await writeFile(path.join(root, f.asset.sidecarPath), JSON.stringify(f.sidecar));
  const s = { identity: { workerId: 'w' }, worker: { gpuId: '1' }, endpoint: { call: async () => [f.asset], served: async () => ({ status: 200, bytes: f.png }) }, assertDeviceProof: async () => ({ gpuUuid: 'declared synthetic' }) };
  const r = report(); r.events.push({ id: f.job.id, status: 'running', progress: 0.25 });
  const proof = await retainImage(s, cell, body, f.job, f.project, path.join(root, 'retained'), r, {}); assert.equal(proof.semanticAcceptance, 'PENDING'); assert.equal(proof.pixelValidity.nonblank, true); assert(proof.visualObligations.length > 0); assert.equal(proof.pngSha256, sha256(f.png)); assert.equal(await readFile(proof.retainedSidecar, 'utf8'), JSON.stringify(f.sidecar));
  await assert.rejects(retainImage(s, cell, body, f.job, f.project, path.join(root, 'retained'), report(), {}), /running progress/);
  s.endpoint.served = async () => ({ status: 200, bytes: Buffer.concat([f.png, Buffer.from('changed')]) });
  await assert.rejects(retainImage(s, cell, body, f.job, f.project, path.join(root, 'retained'), r, {}), /exactly match/);
});
test('shared saved-asset assertion accepts explicit base/portrait request while preserving ordinary defaults', () => {
  const cell = find('portrait'), body = imageRequest(cell, target(cell), 'p', {}), f = savedFixture(body, 'unused');
  assert.equal(assertSavedAsset(f.job, f.asset, f.sidecar, { body, projectId: 'p', workerId: 'w', backend: '1', adapter: 'candle_flux3' }).model, 'flux3_klein_base');
  const bad = structuredClone(f); bad.asset.recipe.rawAdapterSettings.numInferenceSteps = 16;
  assert.throws(() => assertSavedAsset(bad.job, bad.asset, bad.sidecar, { body, projectId: 'p', workerId: 'w', backend: '1', adapter: 'candle_flux3' }));
});
test('PUP/locked annotations/reference order/content hashes/LoRA weight have mutation-sensitive assertions', () => {
  const cell = find('pup-i2i_bbox'), body = imageRequest(cell, target(cell), 'p', { 'reference-0': { assetId: 'ref' } }), rows = '[{"id":"locked"}]', inputs = { 'reference-0': { sha256: 'e'.repeat(64), decodedContentSha256: digest, orientedSize: [4, 3] } };
  const asset = { recipe: { rawAdapterSettings: { quantTier: 'bf16', flux3ReferenceEditing: { references: [{ assetId: 'ref', ordinal: 0, orientedSize: [4, 3], contentSha256: digest }] }, flux3_reference_ids: ['slot-0'], flux3_layout: body.advanced.flux3_layout, flux3Layout: { authoring: body.advanced.flux3_layout, referenceInstanceIds: ['slot-0'], references: [{ assetId: 'ref', ordinal: 0, contentSha256: digest }], lockedAnnotations: rows }, promptEnhancement: { originalPrompt: body.prompt, effectivePrompt: `effective ${rows}`, outcome: 'enhanced', mode: 'i2i_bbox', lockedAnnotations: rows } } } };
  assertImageObservations(cell, body, asset, inputs);
  for (const mutate of [a => a.recipe.rawAdapterSettings.flux3ReferenceEditing.references[0].contentSha256 = 'wrong', a => a.recipe.rawAdapterSettings.flux3ReferenceEditing.references[0].assetId = 'wrong', a => a.recipe.rawAdapterSettings.flux3_reference_ids = ['other'], a => a.recipe.rawAdapterSettings.promptEnhancement.mode = 't2i', a => a.recipe.rawAdapterSettings.promptEnhancement.outcome = 'fallback', a => a.recipe.rawAdapterSettings.promptEnhancement.lockedAnnotations = '[]', a => a.recipe.rawAdapterSettings.flux3_layout = {}]) { const changed = structuredClone(asset); mutate(changed); assert.throws(() => assertImageObservations(cell, body, changed, inputs)); }
});
test('missing gate/installation/checkpoint prerequisites remain actionable BLOCKED, never fallback PASS', async () => {
  const cell = find('authentication-denied');
  await assert.rejects(executeCell({}, cell, target(cell), { id: 'p' }, {}, {}, 'unused', report()), e => e instanceof Blocked && /separate explicit hf403/.test(e.message));
  const checkpoint = find('checkpoint-compatibility'); await assert.rejects(executeCell({}, checkpoint, target(checkpoint), { id: 'p' }, {}, {}, 'unused', report()), /bounded base training/);
});

function encodedSafetensors(header) {
  const json = Buffer.from(JSON.stringify(header)), text = Buffer.alloc(Math.ceil(json.length / 8) * 8, 0x20), prefix = Buffer.alloc(8); json.copy(text); prefix.writeBigUInt64LE(BigInt(text.length));
  const data = Buffer.alloc(Math.max(...Object.entries(header).filter(([key]) => key !== '__metadata__').map(([, value]) => value.data_offsets[1])));
  if (header.__metadata__?.cursor && JSON.parse(header.__metadata__.cursor).update_step > 1) data[0] = 1;
  return Buffer.concat([prefix, text, data]);
}
function checkpointHeader(cell, step) {
  // Tensor-neutral synthetic views of the ACTUAL native checkpoint formats/keys;
  // never supplied to a real worker or presented as native training evidence.
  const tensors = { 'factor/test': { dtype: 'F32', shape: [1, 1], data_offsets: [0, 4] },
    ...(cell.backend === 'candle' ? { 'optim.0.m': { dtype: 'F32', shape: [1, 1], data_offsets: [4, 8] }, 'optim.0.v': { dtype: 'F32', shape: [1, 1], data_offsets: [8, 12] } } : { 'moment_first/test': { dtype: 'F32', shape: [1, 1], data_offsets: [4, 8] }, 'moment_second/test': { dtype: 'F32', shape: [1, 1], data_offsets: [8, 12] } }) };
  const data = Buffer.alloc(12); if (step > 1) data[0] = 1;
  const manifest = Object.fromEntries(Object.entries(tensors).map(([key, value]) => [key, { dtype: value.dtype, shape: value.shape, sha256: sha256(data.subarray(...value.data_offsets)) }]));
  return { __metadata__: { format: cell.backend === 'candle' ? 'flux3-native-training-resume-v1' : 'flux3-mlx-training-resume-v1', policy: 'micro_steps=4;accumulation=1;sampler_len=1', gain_bits: '3ff0000000000000', pending_keys: '[]', identity: JSON.stringify({ schema: 1, training_reference: '7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941', base_repo: cell.repository, base_revision: cell.resolvedRevision, artifacts: { dit: digest }, dataset: digest, preprocessing: digest, configuration: { optimizer: 'adamw' }, factors: { test: [1, 1] } }), cursor: JSON.stringify({ schema: 1, micro_step: step, update_step: step, accumulation_count: 0, sampler_epoch: step - 1, sampler_cursor: 1, rng_state: [1, 2, 3], last_loss_bits: 1065353216 }), tensor_manifest: JSON.stringify(manifest), ...(cell.backend === 'candle' ? { optimizer: 'adamw', optimizer_config: 'adamw;lr=0.0001;wd=0.0001;b1=0.9;b2=0.999;eps=0.00000001', 'optim.step': String(step), optimizer_order: '["test"]' } : { optimizer_config: 'adamw;lr=38d1b717;wd=38d1b717;beta1=0.9;beta2=0.999;eps=1e-8;bias_correction=true', optimizer_steps: JSON.stringify({ test: step }) }) }, ...tensors };
}
test('training refuses reused checkpoint/export output before any model execution', async t => {
  const root = await temporary(t), cell = find('train-interrupt-resume-export-captioned'), outputDir = path.join(root, 'training'); await mkdir(outputDir);
  const initial = path.join(outputDir, 'lora-step000001.resume.safetensors'), final = path.join(outputDir, 'lora-step000004.resume.safetensors'), exported = path.join(outputDir, 'lora.safetensors');
  await writeFile(initial, encodedSafetensors(checkpointHeader(cell, 1))); await writeFile(final, encodedSafetensors(checkpointHeader(cell, 4, 'f'.repeat(64)))); await writeFile(exported, encodedSafetensors({ 'layer.lora_A.weight': { shape: [4, 4], dtype: 'F32', data_offsets: [0, 64] } }));
  const registryFixture = JSON.parse(await readFile(new URL('../tests/fixtures/rust_migration_contracts/training/target-registry.json', import.meta.url), 'utf8'));
  const calls = [], r = report(); let canceled = false, resumedPolls = 0, imagePolls = 0, fixture, restarted = 0, paused = false;
  const persistedPlan = { output: { fileName: 'lora.safetensors', outputDir }, dataset: { datasetId: 'dataset' }, config: { rank: 4 }, target: { targetId: 'flux3_klein_lora', baseModel: 'flux3_klein_base' } };
  const s = { identity: { workerId: 'w' }, worker: { gpuId: '1' }, workerPid: 100, wait: async fn => { for (let i = 0; i < 10; i++) { const j = await fn(); if (j) return j; } throw new Error('fixture exhausted'); }, assertDeviceProof: async () => ({ gpuUuid: 'declared synthetic' }), async pauseWorker() { paused = true; }, failClosed(error) { throw error; }, async restartWorker() { restarted++; paused = false; this.workerPid++; this.identity.workerId = `w-${restarted}`; return { pid: this.workerPid }; }, endpoint: {
    async exchange(method, route, body) { assert(paused); calls.push([method, route, body]); return { status: 400, body: { error: 'checkpoint identity mismatch' } }; },
    async call(method, route, body) {
      calls.push([method, route, body]);
      if (route.startsWith('/api/v1/jobs?')) { assert(paused); return []; }
      if (route.endsWith('/training/datasets')) return { id: 'dataset' };
      if (route === '/api/v1/training/targets') return registryFixture;
      if (route.endsWith('/training/jobs')) { assert.equal(body.dryRun, false); assert.equal(body.config.steps, 4); assert.equal(body.config.saveEvery, 1); return { id: 'train', payload: { dryRun: false, plan: persistedPlan } }; }
      if (route === '/api/v1/jobs/train' && method === 'GET') return { id: 'train', type: 'training', workerId: s.identity.workerId, status: canceled ? 'canceled' : 'running', progress: 0.25, result: { resumeCheckpointStep: 1 }, payload: { plan: persistedPlan } };
      if (route.endsWith('/cancel')) { canceled = true; return {}; }
      if (route.endsWith('/retry')) { assert.equal(body.resume, true); assert.equal(restarted, 1); return { id: 'resumed', payload: { plan: { config: { advanced: { resume: true } } } } }; }
      if (route === '/api/v1/jobs/resumed') return { id: 'resumed', type: 'training', workerId: s.identity.workerId, status: ++resumedPolls === 1 ? 'running' : 'completed', progress: resumedPolls === 1 ? 0.5 : 1, result: { loraId: 'trained', outputPath: exported, resumeCheckpointStep: 4 } };
      if (route.startsWith('/api/v1/loras?')) return [{ id: 'trained', name: 'Trained adapter' }];
      if (route === '/api/v1/image/jobs') { assert.equal(restarted, 2); assert.deepEqual(body.loras, [{ id: 'trained', name: 'Trained adapter', weight: 0.75 }]); fixture = savedFixture(body, root, s.identity.workerId); for (const recipe of [fixture.sidecar.recipe, fixture.asset.recipe, fixture.asset.generationSet.recipe]) recipe.loras[0].hash = sha256(await readFile(exported)); await mkdir(path.join(root, 'assets/images'), { recursive: true }); await writeFile(path.join(root, fixture.sidecar.file.path), fixture.png); await writeFile(path.join(root, fixture.asset.sidecarPath), JSON.stringify(fixture.sidecar)); return fixture.job; }
      if (route === '/api/v1/jobs/image-job') return ++imagePolls === 1 ? { ...fixture.job, status: 'running', progress: 0.5 } : fixture.job;
      if (route.endsWith('/assets')) return [fixture.asset]; throw new Error(`unexpected declared fixture route ${route}`);
    }, served: async () => ({ status: 200, bytes: fixture.png }),
  } };
  s.identity.state = root;
  await assert.rejects(trainingSequence(s, cell, target(cell), { id: 'p', path: root }, { 'train-0': { assetId: 'owned' } }, { training: { captioned: { steps: 4, saveEvery: 1, items: [{ binding: 'train-0', caption: 'real user caption in actual run; synthetic here' }] } } }, path.join(root, 'retained'), r), /fresh training output already contains/);
  assert.equal(restarted, 0); assert(canceled && paused); assert.equal(r.checkpoint, undefined);
});
test('checkpoint proof rejects wrong model, missing optimizer tensors, missing RNG and fabricated advancement', async t => {
  const root = await temporary(t), cell = find('train-interrupt-resume-export-captioned'), file = path.join(root, 'synthetic.safetensors');
  for (const mutate of [h => h.__metadata__.format = 'not-resumable', h => delete h['optim.0.m'], h => { const cursor = JSON.parse(h.__metadata__.cursor); cursor.rng_state = []; h.__metadata__.cursor = JSON.stringify(cursor); }, h => { const identity = JSON.parse(h.__metadata__.identity); identity.base_revision = source; h.__metadata__.identity = JSON.stringify(identity); }]) { const h = checkpointHeader(cell, 1); mutate(h); await writeFile(file, encodedSafetensors(h)); await assert.rejects(assertCheckpoint(file, cell, 1)); }
});
test('saved replay reconstructs persisted nondefault knobs instead of resending original request', () => {
  const recipe = { mode: 'text_to_image', model: 'flux3_klein_base', prompt: 'saved prompt', seed: 937, normalizedSettings: { width: 768, height: 1024, count: 1 }, rawAdapterSettings: { numInferenceSteps: 37, sampler: 'euler', guidanceScale: 4.25, enhancePrompt: false, quantTier: 'bf16' } };
  const replay = replayRequest(recipe, 'owned'); assert.equal(replay.seed, 937); assert.equal(replay.width, 768); assert.equal(replay.advanced.steps, 37); assert.equal(replay.advanced.guidanceScale, 4.25);
  delete recipe.rawAdapterSettings.sampler; assert.throws(() => replayRequest(recipe, 'owned'), /saved sampler/);
});
test('owned exchange rechecks actual canonical health before multipart/rejection and blocks foreign endpoint', async t => {
  const root = await temporary(t), state = path.join(root, 'state'); await mkdir(path.join(state, 'data/projects'), { recursive: true }); await mkdir(path.join(state, 'config')); await mkdir(path.join(state, 'data/cache')); await writeFile(path.join(state, 'data/cache/jobs.db'), 'declared synthetic');
  const api = child(123), controller = new AbortController(), h = ready(state), routes = []; let foreign = false;
  const endpoint = ownedApiController({ url: 'http://127.0.0.1:43123', state, api, controller, fetchImpl: async (url, options) => { routes.push([url, options.method]); return new Response(JSON.stringify(url.endsWith('/health') ? foreign ? { ...h, directories: { ...h.directories, data: root } } : h : { error: 'declared rejection' }), { status: url.endsWith('/health') ? 200 : 400, headers: { 'content-type': 'application/json' } }); } });
  try { const response = await endpoint.exchange('POST', '/api/v1/image/jobs', { invalid: true }); assert.equal(response.status, 400); foreign = true; await assert.rejects(endpoint.exchange('POST', '/api/v1/projects/p/assets', new FormData(), { multipart: true }), /identity refused/); assert.equal(routes.filter(([u]) => u.endsWith('/assets')).length, 0); } finally { endpoint.close(); }
});
test('run report is retained when concrete launcher refuses; no case becomes PASS', async t => {
  const root = await temporary(t), a = admission(root); await mkdir(a.outputRoot); await mkdir(a.hfHome); const inputSpec = { images: {} }; a.inputSpecSha256 = objectHash(inputSpec); a.disk = { freeBytes: 1e12, present: {}, conversionOutputBytes: {} };
  const git = async (cwd, ...args) => { if (args[0] === 'merge-base') return ''; if (args.includes('refs/remotes/origin/main')) return tip; if (args.includes('HEAD')) return source; return ''; };
  const approvedPlan = await approvedFixture(root, a, inputSpec), printed = [], caseId = approvedPlan.plan.cases.find(c => c.backend === a.backend && c.kind === 'replay').caseId;
  const result = await runTerminal({ admission: a, inputSpec, sources, caseIds: [caseId], approvedPlan }, { admission: { git, root: path.join(root, 'checkout') }, diskFree: async () => 1e12, print: text => printed.push(JSON.parse(text)), launch: async () => { throw new Error('declared fixture binary not admitted'); } });
  assert.equal(printed[0].invocations.length, 38); assert.equal(printed[0].budget.unallocatedNotAuthorized, 10);
  assert.equal(result.report.status, 'FAILED'); assert(result.report.cases.every(c => c.status !== 'PASSED')); assert.equal(JSON.parse(await readFile(path.join(result.output, 'report.json'), 'utf8')).error, 'declared fixture binary not admitted');
});
test('concrete MLX launcher and explicit HF403 fault profile use owned offline children/empty hub only', async t => {
  const root = await temporary(t), a = { ...admission(root), backend: 'mlx', faultProfile: 'hf403', installationFaultAdmission: true }, launches = []; let workerId, handler, serverClosed = false;
  const session = await launchTerminalSession(a, { fixtureOnly: true, skipPinReadForFixture: true, hash: async () => digest, mkdir: async () => {}, cp: async () => {}, open: async () => ({ fd: 42, close: async () => {} }), guard: async () => ({}),
    readFile: async () => '{"event":"api_listening","address":"127.0.0.1:43124"}',
    createServer: fn => { handler = fn; return { once() {}, listen(port, host, ready) { assert.equal(port, 0); assert.equal(host, '127.0.0.1'); ready(); }, address: () => ({ port: 43125 }), close(fn) { serverClosed = true; fn(); } }; },
    spawn: (binary, args, options) => { workerId = options.env.SCENEWORKS_WORKER_ID; const c = child(200 + launches.length); launches.push({ c, options }); return c; },
    controller: () => ({ verify: async () => true, startWorker: async fn => fn(), call: async () => [{ id: workerId, gpuId: 'mlx', status: 'idle', capabilities: ['gpu'] }], close() {} }),
  });
  try { assert.equal(session.faultProfile, 'hf403'); assert.equal(launches[1].options.env.SCENEWORKS_BACKEND_MLX_ENABLED, 'true'); assert.equal(launches[1].options.env.CUDA_VISIBLE_DEVICES, '-1'); assert.equal(launches[1].options.env.HF_ENDPOINT, 'http://127.0.0.1:43125'); assert(launches[1].options.env.HF_HOME.endsWith('fault-hf')); const response = { writeHead(status) { assert.equal(status, 403); }, end(text) { assert.match(text, /gated-repository denial/); } }; handler({ method: 'GET', url: '/declared-synthetic/config.json', get headers() { throw new Error('must not read credential headers'); } }, response); assert.equal(session.faultRequests.length, 1); } finally { await session.close(); }
  assert(serverClosed); assert(launches.every(({ c }) => c.exitCode === 0));
});
test('explicit failure profiles require intended actionable errors and distinguish injected denial from real auth success', async () => {
  for (const scenario of ['install-missing', 'authentication-denied']) {
    const cell = find(scenario), r = report(), calls = [], profile = scenario === 'install-missing' ? 'missing-install' : 'hf403';
    const s = { identity: { nonce: 'nonce' }, faultProfile: profile, faultRequests: [], endpoint: { call: async (method, route) => route === '/api/v1/workers' ? [{ id: 'flux3-app-nonce-utility', gpuId: 'cpu', status: 'idle', capabilities: ['model_download'] }] : [{ id: cell.modelId, installState: 'not_installed' }], exchange: async (...args) => { calls.push(args); return { status: scenario === 'install-missing' ? 400 : 403, body: { error: scenario === 'install-missing' ? 'Install missing model' : 'Gated repository credential denied (403)' } }; } } };
    await executeCell(s, cell, target(cell), { id: 'p' }, {}, {}, 'unused', r); assert(r.rejection.status >= 400);
    assert(calls[0][1].endsWith('/models/flux3_klein_base/download'));
    if (scenario === 'authentication-denied') assert(r.observations.some(text => text.includes('NOT evidence')));
    s.endpoint.exchange = async () => ({ status: 400, body: { error: 'unrelated malformed sampler' } }); await assert.rejects(executeCell(s, cell, target(cell), { id: 'p' }, {}, {}, 'unused', report()));
  }
});
test('full PNG proof rejects header-only/truncated/CRC-invalid files even when served and saved hashes match', () => {
  const cell = find('t2i'), f = savedFixture(imageRequest(cell, target(cell), 'p', {}), 'unused'); assertFinishedPng(f.png, 1024, 1024);
  for (const bad of [f.png.subarray(0, 24), f.png.subarray(0, f.png.length - 12), Buffer.concat([f.png, Buffer.from('trailing')])]) assert.throws(() => assertFinishedPng(bad, 1024, 1024));
  const changed = Buffer.from(f.png); changed[changed.length - 1] ^= 1; assert.throws(() => assertFinishedPng(changed, 1024, 1024), /CRC/);
});
test('case-plan schema requires all emitted identities and cannot accept a fabricated plan PASS', async () => {
  const schema = JSON.parse(await readFile(new URL('../docs/testing/sc-26243-terminal-case-plan.schema.json', import.meta.url), 'utf8'));
  assert.equal(schema.properties.status.const, 'NOT_RUN'); assert.deepEqual(schema.$defs.case.properties.status.enum, ['NOT_RUN', 'BLOCKED']);
  for (const cell of plan.cases) { for (const key of schema.$defs.case.required) assert(Object.hasOwn(cell, key)); assert(new RegExp(schema.$defs.case.properties.caseId.pattern).test(cell.caseId)); }
});
test('positive provisioning adds exactly six explicit cells and refuses absent/mismatched per-repo approvals', () => {
  const cells = plan.cases.filter(c => c.kind === 'provision'); assert.equal(cells.length, 6); assert.equal(new Set(cells.map(c => `${c.backend}:${c.repository}`)).size, 6);
  for (const cell of cells) {
    const t = target(cell); assert.throws(() => provisioningApproval(cell, t, {}), /explicit user whole-repository/);
    const row = t.model.downloads.find(d => d.repo === cell.repository), approved = { repository: row.repo, resolvedRevision: row.revision, userDownloadApprovalId: 'declared-synthetic-approval', wholeRepository: true, accountGateAcknowledged: true, diskAcknowledged: true, downloadBytes: row.estimatedSizeBytes };
    assert.equal(provisioningApproval(cell, t, { downloadApprovals: [approved] }).repository, cell.repository);
    for (const mutation of [{ resolvedRevision: source }, { wholeRepository: false }, { accountGateAcknowledged: false }, { diskAcknowledged: false }, { downloadBytes: 0 }]) assert.throws(() => provisioningApproval(cell, t, { downloadApprovals: [{ ...approved, ...mutation }] }));
  }
});
test('shared legacy credential boundary is mocked, frozen-HF-only, and never puts token in authorization receipt/errors', async () => {
  const declared = 'DECLARED_SYNTHETIC_TOKEN_ONLY/+', files = [];
  const token = await runnerHfCredential({ HF_HOME: 'D:/declared-synthetic/hf' }, async file => { files.push(file); return ` ${declared}\n`; }); assert.equal(token, declared); assert.equal(files.length, 1);
  const cell = find('install-provision'), calls = [];
  const authorization = await authorizeFlux3Repository(cell.repository, cell.resolvedRevision, token, { fetchImpl: async (url, options) => { calls.push([url, options]); return new Response('{"declared":"synthetic config"}', { status: 200 }); } });
  assert.equal(calls.length, 1); assert(calls[0][0].startsWith('https://huggingface.co/black-forest-labs/')); assert.equal(calls[0][1].headers.Authorization, `Bearer ${declared}`); assert(!JSON.stringify(authorization).includes(declared));
  await assert.rejects(authorizeFlux3Repository('arbitrary/exfiltration', cell.resolvedRevision, token, { fetchImpl: async () => { throw new Error('must not fetch'); } }), /refuses non-frozen/);
  await assert.rejects(authorizeFlux3Repository(cell.repository, cell.resolvedRevision, token, { fetchImpl: async () => new Response('', { status: 403 }) }), e => /HTTP 403/.test(e.message) && !e.message.includes(declared));
  assert.equal(redactSensitiveText(`${declared} ${encodeURIComponent(declared)}`, [declared]), '[REDACTED] [REDACTED]');
});
test('provisioning concrete launcher reads only mocked account credential after admission and passes it only to owned API utility env', async t => {
  const root = await temporary(t), cell = find('install-provision'), a = { ...admission(root), approvedProvisioning: [{ repository: cell.repository, resolvedRevision: cell.resolvedRevision }] }, calls = [], written = []; let workerId, credentialReads = 0;
  const declared = 'DECLARED_SYNTHETIC_PROVISION_TOKEN/+';
  const session = await launchTerminalSession(a, { fixtureOnly: true, skipPinReadForFixture: true, hash: async () => digest, mkdir: async () => {}, cp: async () => {}, open: async () => ({ fd: 42, close: async () => {} }), guard: async () => ({}),
    readCredential: async env => { credentialReads++; assert.equal(env.HF_HOME, a.hfHome); return declared; }, authorize: async (repo, rev, token) => { assert.equal(token, declared); return { repository: repo, resolvedRevision: rev, configSha256: digest, httpStatus: 200 }; },
    readFile: async () => `{"event":"api_listening","address":"127.0.0.1:43126"}\n${declared} ${encodeURIComponent(declared)}`, writeFile: async (file, text) => written.push(text),
    spawn: (binary, args, options) => { workerId = options.env.SCENEWORKS_WORKER_ID; calls.push(options); return child(300 + calls.length); },
    controller: () => ({ verify: async () => true, startWorker: async fn => fn(), call: async () => [{ id: workerId, gpuId: '1', status: 'idle', capabilities: ['gpu', 'candle'] }], close() {} }),
  });
  try { assert.equal(credentialReads, 1); assert.equal(calls[0].env.HF_TOKEN, declared); assert.equal(calls[0].env.HF_HUB_OFFLINE, undefined); assert.equal(calls[0].env.HF_ENDPOINT, 'https://huggingface.co'); assert.equal(calls[1].env.HF_TOKEN, undefined); assert.equal(calls[1].env.HF_HUB_OFFLINE, '1'); assert(!JSON.stringify(session.authorizations).includes(declared)); assert.equal(session.redact(declared), '[REDACTED]'); } finally { await session.close(); }
  assert(written.length === 2 && written.every(text => !text.includes(declared) && !text.includes(encodeURIComponent(declared))));
});
test('positive ordinary download requires owned job-bound v2 receipt, full inventory and authenticated config hash', async t => {
  const root = await temporary(t), cell = find('install-provision'), packageRoot = path.join(root, 'snapshot'), state = path.join(root, 'state'), modelId = cell.options.provisionModelId; await mkdir(packageRoot); await mkdir(path.join(state, 'data/models', modelId), { recursive: true });
  const config = Buffer.from('{"declared":"synthetic config"}'), license = Buffer.from('FLUX Kommunity License - declared synthetic fixture, not real model evidence'); await writeFile(path.join(packageRoot, 'config.json'), config); await writeFile(path.join(packageRoot, 'LICENSE.md'), license);
  const inventory = { root: packageRoot, repository: cell.repository, resolvedRevision: cell.resolvedRevision, authority: 'parent-approved-pinned-inventory', files: [{ path: 'config.json', bytes: config.length, sha256: sha256(config) }, { path: 'LICENSE.md', bytes: license.length, sha256: sha256(license) }] };
  const receipt = { schemaVersion: 2, repo: cell.repository, modelId, snapshotRevision: cell.resolvedRevision, jobId: 'download-job', completedAt: 'declared synthetic timestamp', manifestFiles: [], resolvedFiles: ['config.json', 'LICENSE.md'] }, marker = path.join(state, 'data/models', modelId, '.sceneworks-download-complete.json'); await writeFile(marker, JSON.stringify(receipt));
  const utility = { id: 'flux3-app-nonce-utility', gpuId: 'cpu', status: 'idle', capabilities: ['model_download'] }, requests = [];
  const session = { identity: { nonce: 'nonce', state }, execution: { sourceSha: source, inferenceMainSha: inference, runnerAccount: 'declared synthetic' }, authorizations: [{ repository: cell.repository, resolvedRevision: cell.resolvedRevision, httpStatus: 200, configSha256: sha256(config), sourceSha: source, inferenceMainSha: inference, runnerAccount: 'declared synthetic' }], wait: async fn => fn(), endpoint: { async call(method, route, body) { requests.push([method, route, body]); if (route === '/api/v1/workers') return [utility]; if (route.endsWith('/download')) return { id: 'download-job', type: 'model_download', payload: { modelId, repo: cell.repository, revision: cell.resolvedRevision, files: [] } }; if (route === '/api/v1/jobs/download-job') return { id: 'download-job', type: 'model_download', status: 'completed', workerId: utility.id }; if (route === '/api/v1/models') return [{ id: modelId, installState: 'installed', installedPath: packageRoot }]; throw new Error('unexpected synthetic route'); } } };
  const r = report(); await provisionCell(session, cell, target(cell), { packages: { [cell.artifactKey]: inventory } }, r); assert.equal(r.provisioning.jobId, 'download-job'); assert.equal(r.provisioning.credentialIncluded, false); assert(requests.some(([method, route]) => method === 'POST' && route.endsWith('/download')));
  for (const mutation of [{ jobId: 'preexisting-job' }, { snapshotRevision: source }, { schemaVersion: 1 }, { resolvedFiles: ['config.json'] }, { manifestFiles: ['config.json'] }]) { await writeFile(marker, JSON.stringify({ ...receipt, ...mutation })); await assert.rejects(provisionCell(session, cell, target(cell), { packages: { [cell.artifactKey]: inventory } }, report())); }
});
test('environment scrub filters secret keys before touching their values', () => {
  const env = { PATH: 'declared-safe' }; Object.defineProperty(env, 'HF_TOKEN', { enumerable: true, get() { throw new Error('ambient credential value must not be read'); } });
  const clean = offlineChildEnv(env, admission('D:/synthetic'), { state: 'D:/synthetic/state', workerId: 'owned' }); assert.equal(clean.HF_TOKEN, undefined);
});
test('real-file adapter import rank/hash and scale zero/on dispatch retain distinct image effects, not just completed status', async t => {
  const root = await temporary(t), zero = find('lora-zero'), on = find('lora-on'), adapterFile = path.join(root, 'declared-synthetic-adapter.safetensors'), bytes = encodedSafetensors({ 'layer.lora_A.weight': { shape: [4, 8], dtype: 'F32', data_offsets: [0, 128] } }); await writeFile(adapterFile, bytes);
  const adapter = { root, path: adapterFile, sha256: sha256(bytes), bytes: bytes.length, rank: 4, repository: on.repository, resolvedRevision: on.resolvedRevision }, spec = { adapters: { [on.modelId]: adapter } }; let current, polls = 0;
  const s = { identity: { nonce: 'nonce', workerId: 'owned' }, worker: { gpuId: '1' }, wait: async fn => { for (let i = 0; i < 10; i++) { const v = await fn(); if (v) return v; } throw new Error('fixture exhausted'); }, assertDeviceProof: async () => ({ gpuUuid: 'declared synthetic' }), endpoint: { async call(method, route, body) {
    if (route === '/api/v1/loras/import') { assert.equal(body.expectedSha256, adapter.sha256); assert.equal(body.sourcePath, await realpath(adapterFile)); return { id: 'import-job' }; }
    if (route === '/api/v1/jobs/import-job') return { id: 'import-job', type: 'lora_import', status: 'completed', workerId: 'flux3-app-nonce-utility', result: { loraId: 'provisioned' } };
    if (route.startsWith('/api/v1/loras?')) return [{ id: 'provisioned', name: 'Provisioned adapter', installState: 'installed' }];
    if (route === '/api/v1/image/jobs') { current = savedFixture(body, root, 'owned'); polls = 0; for (const recipe of [current.sidecar.recipe, current.asset.recipe, current.asset.generationSet.recipe]) recipe.loras[0].hash = adapter.sha256; await mkdir(path.join(root, 'assets/images'), { recursive: true }); await writeFile(path.join(root, current.sidecar.file.path), current.png); await writeFile(path.join(root, current.asset.sidecarPath), JSON.stringify(current.sidecar)); return current.job; }
    if (route === '/api/v1/jobs/image-job') return ++polls === 1 ? { ...current.job, status: 'running', progress: 0.5 } : current.job;
    if (route.endsWith('/assets')) return [current.asset]; throw new Error(`unexpected adapter fixture route ${route}`);
  }, served: async () => ({ status: 200, bytes: current.png }) } };
  const rz = report(), ro = report(); await executeCell(s, zero, target(zero), { id: 'p', path: root }, {}, spec, path.join(root, 'zero'), rz); await executeCell(s, on, target(on), { id: 'p', path: root }, {}, spec, path.join(root, 'on'), ro);
  assert.notEqual(rz.outputs[0].pngSha256, ro.outputs[0].pngSha256); assert.equal(ro.adapterEffect.adapterSha256, adapter.sha256); assert.equal(ro.requests.at(-1).body.loras[0].weight, 0.75);
  await assert.rejects(importAdapter(s, { ...adapter, rank: 8 }, on, { id: 'p' }, report()), /rank mismatch/);
  await assert.rejects(importAdapter(s, { ...adapter, sha256: digest }, on, { id: 'p' }, report()));
});
test('checkpoint parser traces both actual native optimizer formats and verifies tensor payload hashes, not invented state aliases', async t => {
  const root = await temporary(t), file = path.join(root, 'synthetic-native-format.safetensors');
  for (const backend of ['candle', 'mlx']) {
    const cell = find('train-interrupt-resume-export-captioned', 'base', backend), header = checkpointHeader(cell, 1); await writeFile(file, encodedSafetensors(header)); const state = await assertCheckpoint(file, cell, 1); assert.equal(state.cursor.update_step, 1);
    const bytes = encodedSafetensors(header); bytes[bytes.length - 1] ^= 1; await writeFile(file, bytes); await assert.rejects(assertCheckpoint(file, cell, 1), /tensor payload differs/);
    const bad = structuredClone(header); if (backend === 'mlx') { delete bad['moment_first/test']; bad['optim.0.m'] = header['moment_first/test']; } else { delete bad['optim.0.v']; bad['optimizerState'] = header['optim.0.v']; } await writeFile(file, encodedSafetensors(bad)); await assert.rejects(assertCheckpoint(file, cell, 1));
  }
});
test('PNG inspection reconstructs all five scanline filters and nonblank ranges without claiming semantic acceptance', () => {
  const chunk = (name, data) => { const size = Buffer.alloc(4), crc = Buffer.alloc(4), encoded = Buffer.concat([Buffer.from(name), data]); size.writeUInt32BE(data.length); crc.writeUInt32BE(pngCrc32(encoded)); return Buffer.concat([size, encoded, crc]); };
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(2); ihdr.writeUInt32BE(2, 4); ihdr[8] = 8; ihdr[9] = 2;
  const samples = [
    [0, 10, 20, 30, 40, 50, 60, 0, 11, 21, 31, 42, 52, 62],
    [1, 10, 20, 30, 30, 30, 30, 1, 11, 21, 31, 31, 31, 31],
    [2, 10, 20, 30, 40, 50, 60, 2, 1, 1, 1, 2, 2, 2],
    [3, 10, 20, 30, 35, 40, 45, 3, 6, 11, 16, 17, 17, 17],
    [4, 10, 20, 30, 30, 30, 30, 4, 1, 1, 1, 2, 2, 2],
  ];
  const png = data => Buffer.concat([Buffer.from('89504e470d0a1a0a', 'hex'), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(Buffer.from(data))), chunk('IEND', Buffer.alloc(0))]);
  for (const sample of samples) { const proof = assertFinishedPng(png(sample), 2, 2); assert.deepEqual(proof.minimum, [10, 20, 30]); assert.deepEqual(proof.maximum, [42, 52, 62]); assert.equal(proof.semanticAcceptance, 'PENDING'); }
  const prefix = Buffer.alloc(9); prefix[0] = 0; prefix.writeUInt32LE(2, 1); prefix.writeUInt32LE(2, 5);
  const decoded = Buffer.from([10, 20, 30, 40, 50, 60, 11, 21, 31, 42, 52, 62]), carrierHash = sha256(Buffer.concat([prefix, decoded]));
  for (const sample of samples) { const image = png(sample), proof = assertFinishedPng(image, 2, 2, { forReference: true }); assert.equal(proof.decodedContentSha256, carrierHash); assert.notEqual(proof.decodedContentSha256, sha256(image), 'native carrier hash is NOT encoded file hash'); }
  assert.throws(() => assertFinishedPng(png(Array(14).fill(0)), 2, 2), /blank/);
});
test('concrete launcher consumes actual single-use ledger permit BEFORE every generation POST, including fresh-process repro', async t => {
  const root = await temporary(t), a = admission(root), approval = await approvedFixture(root, a, {}); a.approvedPlan = approval;
  const cell = approval.plan.cases.find(c => c.backend === 'candle' && c.kind === 'replay'), body = imageRequest(cell, target(cell), 'p', {}), children = []; let workerId, posts = 0;
  const session = await launchTerminalSession(a, { fixtureOnly: true, skipPinReadForFixture: true, hash: async () => digest, mkdir: async () => {}, cp: async () => {}, open: async () => ({ fd: 42, close: async () => {} }), guard: async () => ({}), readFile: async () => '{"event":"api_listening","address":"127.0.0.1:43127"}',
    spawn: (binary, args, options) => { workerId = options.env.SCENEWORKS_WORKER_ID; const c = child(400 + children.length); children.push(c); return c; },
    controller: () => ({ verify: async () => true, startWorker: async fn => fn(), async call(method, route) { if (route === '/api/v1/workers') return [{ id: workerId, gpuId: '1', status: 'idle', capabilities: ['gpu', 'candle'] }]; assert.equal(route, '/api/v1/image/jobs'); const ledger = await auditLedger({ approval, backend: 'candle', ledgerRoot: a.ledgerRoot }); assert.equal(ledger.state.generationCount, ++posts, 'permit must be durable before dispatch'); return { id: `declared-job-${posts}` }; }, close() {} }),
  });
  try { session.beginCase(cell); await session.endpoint.call('POST', '/api/v1/image/jobs', body); await session.restartWorker(); await session.endpoint.call('POST', '/api/v1/image/jobs', body); await assert.rejects(session.endpoint.call('POST', '/api/v1/image/jobs', body), /no approved single-use/); assert.equal(posts, 2); assert.equal(session.permits.length, 2); } finally { await session.close(); }
  assert.equal((await auditLedger({ approval, backend: 'candle', ledgerRoot: a.ledgerRoot })).state.generationCount, 2); assert(children.every(c => c.exitCode === 0));
});
test('concrete negative probe pauses ONLY its worker and cancels unexpected accepted queue jobs before any resume, spending zero generation permits', async t => {
  const root = await temporary(t), a = admission(root), approval = await approvedFixture(root, a, {}); a.approvedPlan = approval;
  const cell = approval.plan.cases.find(c => c.backend === 'candle' && c.options.invalid === 'bad-reference'), children = []; let workerId, accepted = false, canceled = false;
  const session = await launchTerminalSession(a, { fixtureOnly: true, skipPinReadForFixture: true, hash: async () => digest, mkdir: async () => {}, cp: async () => {}, open: async () => ({ fd: 42, close: async () => {} }), guard: async () => ({}), readFile: async () => '{"event":"api_listening","address":"127.0.0.1:43128"}',
    spawn: (binary, args, options) => { workerId = options.env.SCENEWORKS_WORKER_ID; const c = child(500 + children.length); children.push(c); return c; },
    controller: () => ({ verify: async () => true, startWorker: async fn => fn(), async call(method, route) {
      if (route === '/api/v1/workers') return [{ id: workerId, gpuId: '1', status: 'idle', capabilities: ['gpu', 'candle'] }];
      assert.equal(children[0].exitCode, null, 'owned API must remain alive'); assert.equal(children[1].exitCode, 0, 'only owned GPU worker must be stopped before any probe/cancel');
      if (route.startsWith('/api/v1/jobs?')) return accepted ? [{ id: 'unexpected', status: canceled ? 'canceled' : 'queued' }] : [];
      if (route.endsWith('/cancel')) { canceled = true; return {}; }
      if (route === '/api/v1/jobs/unexpected') return { id: 'unexpected', type: 'image_generate', status: canceled ? 'canceled' : 'queued' }; throw new Error('unexpected fixture route');
    }, async exchange() { accepted = true; return { status: 201, body: { id: 'unexpected' } }; }, close() {} }),
  });
  try { session.beginCase(cell); await assert.rejects(session.endpoint.exchange('POST', '/api/v1/image/jobs', {}), /owned GPU worker stopped/);
    await assert.rejects(ownedNegativeProbe(session, '/api/v1/image/jobs', {}, () => { throw new Error('must not accept unexpected job'); }, report()), /unexpectedly accepted/); assert(canceled); assert.equal(children.length, 2, 'no worker restart after unsafe acceptance'); assert.throws(() => session.alive(), /unexpectedly accepted/);
  } finally { await session.close(); }
  assert.equal((await auditLedger({ approval, backend: 'candle', ledgerRoot: a.ledgerRoot })).state.generationCount, 0);
});
test('retained reference bindings separate encoded SHA from exact native RGB/RGBA carrier hash, retaining hidden alpha pixels', async t => {
  const root = await temporary(t), uploads = [], chunk = (name, data) => { const prefix = Buffer.alloc(4), crc = Buffer.alloc(4), body = Buffer.concat([Buffer.from(name), data]); prefix.writeUInt32BE(data.length); crc.writeUInt32BE(pngCrc32(body)); return Buffer.concat([prefix, body, crc]); };
  const png = (color, pixels, extra = []) => { const ihdr = Buffer.alloc(13), channels = color === 6 ? 4 : 3; ihdr.writeUInt32BE(2); ihdr.writeUInt32BE(2, 4); ihdr[8] = 8; ihdr[9] = color; const scanlines = Buffer.concat([Buffer.from([0]), pixels.subarray(0, 2 * channels), Buffer.from([0]), pixels.subarray(2 * channels)]); return Buffer.concat([Buffer.from('89504e470d0a1a0a', 'hex'), chunk('IHDR', ihdr), ...extra, chunk('IDAT', deflateSync(scanlines)), chunk('IEND', Buffer.alloc(0))]); };
  const session = { endpoint: { async exchange(method, route, form, options) { assert.equal(method, 'POST'); assert.equal(route, '/api/v1/projects/p/assets'); assert.equal(options.multipart, true); uploads.push(Buffer.from(await form.get('file').arrayBuffer())); return { status: 201, body: { id: `owned-${uploads.length}` } }; } } };
  for (const [mode, pixels] of [['rgb', Buffer.from([10, 20, 30, 40, 50, 60, 11, 21, 31, 42, 52, 62])], ['rgba', Buffer.from([10, 20, 30, 0, 40, 50, 60, 255, 11, 21, 31, 0, 42, 52, 62, 255])]]) {
    const bytes = png(mode === 'rgba' ? 6 : 2, pixels), file = path.join(root, `${mode}.png`); await writeFile(file, bytes);
    const binding = await uploadBinding(session, { id: 'p' }, { root, path: file, sha256: sha256(bytes) }), prefix = Buffer.alloc(9); prefix[0] = mode === 'rgba' ? 1 : 0; prefix.writeUInt32LE(2, 1); prefix.writeUInt32LE(2, 5);
    assert.equal(binding.decodedContentSha256, sha256(Buffer.concat([prefix, pixels]))); assert.equal(binding.sha256, sha256(bytes)); assert.notEqual(binding.sha256, binding.decodedContentSha256); assert.deepEqual(binding.orientedSize, [2, 2]); assert.equal(binding.sourceMode, mode); assert.deepEqual(uploads.at(-1), bytes);
  }
  const bytes = png(2, Buffer.alloc(12), [chunk('eXIf', Buffer.from('declared ambiguous fixture'))]), file = path.join(root, 'ambiguous.png'); await writeFile(file, bytes);
  await assert.rejects(uploadBinding(session, { id: 'p' }, { root, path: file, sha256: sha256(bytes) }), e => e instanceof Blocked && /EXIF/.test(e.message)); assert.equal(uploads.length, 2);
});
test('external publisher hashes never pass without owned resolver mapping and actual device-floor advertisement', async t => {
  const root = await temporary(t), cell = plan.cases.find(c => c.backend === 'candle' && c.modelId === 'flux3_klein_base' && c.tier === 'fp8r'), snapshot = path.join(root, 'snapshot'), variant = path.join(snapshot, 'variants/fp8r');
  async function inventory(directory, names, role) { await mkdir(directory, { recursive: true }); const files = []; for (const name of names) { const bytes = Buffer.from(`DECLARED SYNTHETIC ${name}`); await writeFile(path.join(directory, name), bytes); files.push({ path: name, bytes: bytes.length, sha256: sha256(bytes) }); } return { root: directory, repository: cell.repository, resolvedRevision: cell.resolvedRevision, authority: 'parent-approved-pinned-inventory', ...(role ? { role } : {}), files }; }
  const primary = await inventory(variant, ['config.json', 'manifest.json', 'model.safetensors', 'hf_quant_config.json']);
  const license = await inventory(snapshot, ['LICENSE.md', 'NOTICE.md'], 'license'), ae = await inventory(snapshot, ['ae.safetensors'], 'autoencoder'), encoder = await inventory(path.join(snapshot, 'text_encoder'), ['model.safetensors'], 'text_encoder'); primary.externalComponents = [license, ae, encoder];
  const s = { worker: { capabilities: ['gpu', 'candle', 'fp8r'] }, endpoint: { call: async () => [{ id: cell.modelId, installState: 'installed', installedPath: snapshot }] } };
  await assert.rejects(assertPackage(s, cell, target(cell), primary), /no resolver-visible materialization/);
  const noLicense = structuredClone(primary); noLicense.externalComponents = [ae, encoder]; await assert.rejects(assertPackage(s, cell, target(cell), noLicense));
  s.worker.capabilities = ['gpu', 'candle']; await assert.rejects(assertPackage(s, cell, target(cell), primary), e => e instanceof Blocked && /device floor/.test(e.message));
});
test('explicit public PUP-only provisioning does not read any account token and keeps the GPU worker offline', async t => {
  const root = await temporary(t), a = { ...admission(root), approvedProvisioning: [{ repository: sources.upsampler.downloads[0].repo, resolvedRevision: sources.upsampler.downloads[0].revision }] }, envs = []; let workerId;
  const publicAuthorization = await authorizeFlux3Repository(a.approvedProvisioning[0].repository, a.approvedProvisioning[0].resolvedRevision, undefined, { fetchImpl: async (url, options) => { assert.equal(options.headers, undefined); return new Response('{"declared":"public fixture"}', { status: 200 }); } }); assert.equal(publicAuthorization.mechanism, 'public_pinned_repository');
  const s = await launchTerminalSession(a, { fixtureOnly: true, skipPinReadForFixture: true, hash: async () => digest, mkdir: async () => {}, cp: async () => {}, open: async () => ({ fd: 42, close: async () => {} }), guard: async () => ({}), readFile: async () => '{"event":"api_listening","address":"127.0.0.1:43129"}', readCredential: async () => { throw new Error('public component must not read credential'); }, authorize: async (repo, rev, token) => { assert.equal(token, undefined); return publicAuthorization; },
    spawn: (binary, args, options) => { workerId = options.env.SCENEWORKS_WORKER_ID; envs.push(options.env); return child(600 + envs.length); }, controller: () => ({ verify: async () => true, startWorker: async fn => fn(), call: async () => [{ id: workerId, gpuId: '1', status: 'idle', capabilities: ['gpu', 'candle'] }], close() {} }),
  });
  try { assert.equal(envs[0].HF_TOKEN, undefined); assert.equal(envs[0].HF_ENDPOINT, 'https://huggingface.co'); assert.equal(envs[0].HF_HUB_OFFLINE, undefined); assert.equal(envs[1].HF_TOKEN, undefined); assert.equal(envs[1].HF_HUB_OFFLINE, '1'); } finally { await s.close(); }
});

test('a bound adapter case uses only its own backend training export on the trained checkpoint', () => {
  const representative = representativePlan(sources);
  const on = representative.cases.find(c => c.caseId === 'candle.base.lora-on.bf16');
  assert.deepEqual(on.adapterBinding, { source: 'training-export', lifecycleId: 'candle.base.train-interrupt-resume-export-captioned.bf16', backend: 'candle', modelId: 'flux3_klein_base' });
  const exported = { lifecycleId: on.adapterBinding.lifecycleId, modelId: 'flux3_klein_base', sha256: digest, bytes: 1, rank: 4, root: 'r', path: 'r/a', repository: on.repository, resolvedRevision: on.resolvedRevision };
  const provisioned = { sha256: 'e'.repeat(64) };
  assert.equal(boundAdapterSpec({ trainingExports: { candle: exported } }, on, provisioned), exported, 'the exported LoRA, never a provisioned substitute');
  for (const [session, cell, pattern] of [
    [{}, on, /run its bound training lifecycle .* first/],
    [{ trainingExports: { mlx: { ...exported } } }, on, /run its bound training lifecycle/],
    [{ trainingExports: { candle: { ...exported, lifecycleId: 'candle.base.other.bf16' } } }, on, /run its bound training lifecycle/],
    [{ trainingExports: { candle: exported } }, { ...on, modelId: 'flux3_klein' }, /cannot drive flux3_klein/],
    [{ trainingExports: { candle: { ...exported, modelId: 'flux3_klein' } } }, on, /checkpoint differs/],
    [{ trainingExports: { candle: exported } }, { ...on, adapterBinding: { ...on.adapterBinding, backend: 'mlx' } }, /own backend/],
  ]) assert.throws(() => boundAdapterSpec(session, cell, provisioned), error => error instanceof Blocked && pattern.test(error.message));
  const legacy = { ...on }; delete legacy.adapterBinding;
  assert.equal(boundAdapterSpec({}, legacy, provisioned), provisioned, 'unbound diagnostic cases keep their provisioned adapter');
});
test('reference cases use a neutral, identity-preserving, setting-only prompt; text-only cases stay person-free', () => {
  const representative = representativePlan(sources);
  for (const prompt of [REFERENCE_PROMPT]) for (const banned of [/nud|naked|nsfw|undress|lingerie|sexy/i, /(age|younger|older|child|teen|kid)/i, /different (person|face|identity)|swap/i]) assert.doesNotMatch(prompt, banned);
  assert.match(REFERENCE_PROMPT, /unchanged identity/);
  for (const invocation of representative.generationInvocations) {
    const references = invocation.options.references ?? 0;
    assert.equal(invocation.requestTemplate.prompt, references ? REFERENCE_PROMPT : SCENE_PROMPT, invocation.invocationId);
    for (const box of invocation.requestTemplate.advanced.flux3_layout?.boxes ?? []) {
      if (references) assert.doesNotMatch(box.desc, /teapot|cup/, 'reference layout descriptors are subject-agnostic');
    }
  }
  const moveRemove = representative.generationInvocations.find(i => i.options.references === 10 && i.requestTemplate.advanced.flux3_layout);
  assert.deepEqual(moveRemove.requestTemplate.advanced.flux3_layout.boxes.map(b => [b.id, b.src_bbox, b.tgt_bbox]), [
    ['move', [0.125, 0.25, 0.75, 0.875], [0.25, 0.125, 0.875, 0.75]], ['remove', [0.01, 0.02, 0.11, 0.12], null]], 'layout geometry is unchanged');
});
