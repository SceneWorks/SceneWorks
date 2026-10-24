import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, readFile, rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import { ARTIFACTS, SOURCE, requiredCases, validatePlan, validateDelivery, hashFile, verifyInputs, compareNumeric, executedTests, decodeAV, decodedDelta, validateEvidence, runProcess, runCase, verifyCampaign } from './h3-finetune-terminal.mjs';
import { runScenario, pointer } from './h3-finetune-terminal-api.mjs';
const sw = 'a'.repeat(40), inf = 'b'.repeat(40);
const log = 'test result: ok. 1 passed; 0 failed; 0 ignored;';
async function fixture(t, group = 'artifact') {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'h3-terminal-fixture-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const input = path.join(dir, 'synthetic-checkpoint'); await writeFile(input, 'not a model');
  const hash = await hashFile(input);
  const p = { version: 1, sceneworks: sw, inference: inf, source: SOURCE, output: dir, responsibleTask: 'sc-26691', nextAuthorizedAction: 'Coordinator authorizes exact next Actions case; no automatic retry.', cases: requiredCases().map(c => ({ ...c, command: ['node', 'controlled-fixture.mjs'], timeoutSeconds: 10, evidence: 'evidence.json', loadHostBudgetBytes: 1000, loadDeviceBudgetBytes: 2000, minimumControlDelta: 0.1, controlStream: 'video', referenceRevision: 'c'.repeat(40), referenceCommand: ['node', 'controlled-reference.mjs'], referenceOutput: 'reference.json', tolerance: { atol: 0.001, rtol: 0.001 }, inputs: [{ path: input, sha256: hash, role: 'checkpoint', artifact: c.artifact }, { path: input, sha256: hash, role: 'reference' }] })) };
  const c = p.cases.find(c => c.group === group);
  const e = { status: 'executed', sceneworks: sw, inference: inf, caseId: c.id, backend: c.backend, source: SOURCE, artifact: c.artifact, hardware: { device: 'synthetic CPU fixture only', driver: 'fixture' }, runtimeIdentity: { checkpointSha256: hash, planId: 'synthetic-plan', semanticDigest: hash, topology: 'fixture', quantization: 'fixture', components: [], adapters: [], operation: 'fixture', samplingPolicy: { steps: 2, schedulerShift: null }, seed: 42 }, metrics: { hostPeakBytes: 100, devicePeakBytes: 200, loadHostPeakBytes: 90, loadDevicePeakBytes: 180, sampleIntervalMs: 50, hostMethod: 'sampled-rss', deviceMethod: 'native-allocated-high-water' }, files: ['trace.json'], av: 'synthetic-av.bin', control: { intervention: 'replace conditioning', checkpointSha256: hash, seed: 42, av: 'control-av.bin' }, reference: { revision: c.referenceRevision, sha256: hash }, numeric: { candidate: 'candidate.json', reference: 'reference.json' } };
  e.testInputSha256 = hash;
  e.completedJob = { id: 'synthetic-job', status: 'completed', providerId: 'synthetic-native-provider', output: e.av };
  e.loadEvent = { event: 'h3_import_prepared_load', jobId: e.completedJob.id, provider: e.completedJob.providerId, backend: c.backend, native_backend: c.backend, seal_validated: true, cache_state: 'Cold', identity: JSON.stringify({ checkpointId: 'synthetic-checkpoint', planId: e.runtimeIdentity.planId, semanticDigest: hash, assembly: {}, sourceFiles: [{ path: input, sha256: hash }] }), prepared_sources: JSON.stringify([{ path: input, sha256: hash }]), operation: e.runtimeIdentity.operation, seed: 42, steps: 'Some(2)', scheduler_shift: 'None' };
  await writeFile(path.join(dir, 'source-catalog.json'), JSON.stringify({ sha: SOURCE.revision, siblings: ARTIFACTS.map(rfilename => ({ rfilename, lfs: { sha256: hash } })) }));
  await writeFile(path.join(dir, 'trace.json'), '{}');
  await writeFile(path.join(dir, e.av), 'synthetic primary AV');
  await writeFile(path.join(dir, e.control.av), 'synthetic control AV');
  await writeFile(path.join(dir, 'candidate.json'), '[1,2,3]');
  await writeFile(path.join(dir, 'reference.json'), '[1,2,3]');
  await writeFile(path.join(dir, 'reference.log'), 'controlled executable reference');
  await writeFile(path.join(dir, 'reference-execution.json'), JSON.stringify({ command: c.referenceCommand, revision: c.referenceRevision, outputSha256: await hashFile(path.join(dir, 'reference.json')) }));
  const save = () => writeFile(path.join(dir, c.evidence), JSON.stringify(e)); await save();
  const decode = async file => ({ sha256: await hashFile(file), streams: 'controlled process fixture, NOT real AV' });
  return { p, c, e, dir, save, decode, verify: () => validateEvidence(p, c, dir, log, decode, async () => 0.25) };
}
test('complete exact inventory includes eight frozen artifact/backend executions and every mechanism', async t => {
  const { p } = await fixture(t);
  assert.equal(validatePlan(p), p);
  assert.equal(requiredCases().filter(c => c.group === 'artifact').length, 8);
  assert.equal(new Set(ARTIFACTS).size, 4);
  for (const mutate of [p => p.cases.pop(), p => p.cases[0].artifact = 'stock.safetensors', p => p.source.revision = 'd'.repeat(40)]) {
    const copy = structuredClone(p); mutate(copy); assert.throws(() => validatePlan(copy));
  }
});
test('actual streamed content detects input drift, not filename or metadata', async t => {
  const { c } = await fixture(t); await verifyInputs(c);
  await writeFile(c.inputs[0].path, 'different checkpoint');
  await assert.rejects(verifyInputs(c), /content changed/);
});
test('numeric comparison discriminates scaling, NaN, empty and tolerance mutations', () => {
  assert.equal(compareNumeric([1, 2.001], [1, 2], { atol: 0.01, rtol: 0 }).elements, 2);
  for (const [a, b, tol] of [[[2,4],[1,2],{atol:0.01,rtol:0}], [[NaN],[1],{atol:0,rtol:0}], [[],[],{atol:0,rtol:0}], [[1],[1],{atol:-1,rtol:0}]]) assert.throws(() => compareNumeric(a,b,tol));
});
test('test execution evidence rejects ignored, zero, missing and failure summaries', () => {
  assert.equal(executedTests(log), 1);
  for (const s of ['', 'test result: ok. 0 passed; 0 failed; 0 ignored;', 'test result: ok. 1 passed; 0 failed; 1 ignored;', 'test result: ok. 1 passed; 1 failed; 0 ignored;']) assert.throws(() => executedTests(s));
});
for (const [name, mutate] of [
  ['wrong source revision', e => e.inference = sw],
  ['wrong checkpoint', e => e.runtimeIdentity.checkpointSha256 = 'f'.repeat(64)],
  ['missing sealed checkpoint identity', e => { const identity = JSON.parse(e.loadEvent.identity); identity.sourceFiles = []; e.loadEvent.identity = JSON.stringify(identity); }],
  ['unvalidated cache seal', e => e.loadEvent.seal_validated = false],
  ['CPU fallback', e => e.loadEvent.backend = 'cpu'],
  ['wrong native backend', e => e.loadEvent.native_backend = 'cuda'],
  ['wrong native job binding', e => e.loadEvent.jobId = 'unrelated-job'],
  ['changed observed sampling policy', e => e.loadEvent.steps = 'Some(99)'],
  ['wrong output binding', e => e.completedJob.output = 'unrelated-av.bin'],
  ['missing AV', e => delete e.av],
  ['missing metrics', e => delete e.metrics],
  ['residency deltas mislabeled as peaks', e => e.metrics.deviceMethod = 'pre-post-residency-delta'],
  ['missing load peak', e => delete e.metrics.loadDevicePeakBytes],
  ['loading budget exceeded', e => e.metrics.loadHostPeakBytes = 1001],
  ['missing runtime component hash', e => e.runtimeIdentity.components = [{ sha256: 'e'.repeat(64) }]],
  ['not actually executed', e => e.status = 'planned'],
  ['missing hardware', e => e.hardware = {}],
]) test(`receipt mutation rejects ${name}`, async t => {
  const f = await fixture(t); await f.verify(); mutate(f.e); await f.save(); await assert.rejects(f.verify());
});
test('numeric case executes scale-sensitive array assertion and pinned reference revision', async t => {
  const f = await fixture(t, 'numeric'); const result = await f.verify(); assert.equal(result.numeric.elements, 3);
  await writeFile(path.join(f.dir, 'candidate.json'), '[2,4,6]'); await assert.rejects(f.verify(), /numeric mismatch/);
  await writeFile(path.join(f.dir, 'candidate.json'), '[1,2,3]'); f.e.reference.revision = sw; await f.save(); await assert.rejects(f.verify(), /reference revision drift/);
  await assert.rejects(validateEvidence(f.p, f.c, f.dir, 'test result: ok. 0 passed; 0 failed; 1 ignored;', f.decode), /ignored/);
});
test('conditioning controls reject dropped conditioning and changed seed/checkpoint', async t => {
  const f = await fixture(t, 'capability'); await f.verify();
  await assert.rejects(validateEvidence(f.p, f.c, f.dir, log, f.decode, async () => 0), /ignored/);
  f.e.control.seed = 43; await f.save(); await assert.rejects(f.verify());
  f.e.control.seed = 42; f.e.control.checkpointSha256 = '1'.repeat(64); await f.save(); await assert.rejects(f.verify());
});
test('AV verifier fully decodes both streams and rejects metadata-only/missing audio', async t => {
  const f = await fixture(t); const calls = [];
  const command = async (bin, args) => { calls.push({ bin, args }); return { stdout: JSON.stringify({ streams: [{ codec_type: 'video', nb_read_frames: '4', duration: '1' }, { codec_type: 'audio', nb_read_frames: '4', duration: '1' }] }) }; };
  await decodeAV(path.join(f.dir, f.e.av), command);
  assert.equal(calls[1].bin, 'ffmpeg'); assert.ok(calls[1].args.includes('0:a:0')); assert.ok(calls[1].args.includes('-xerror'));
  await assert.rejects(decodeAV(path.join(f.dir, f.e.av), async () => ({ stdout: '{"streams":[{"codec_type":"video","nb_read_frames":"4","duration":"1"}]}' })), /audio/);
});
test('decoded scale-sensitive intervention metric is computed from media process bytes', async () => {
  const command = async (_bin, args) => ({ stdout: Buffer.from(args.includes('primary') ? [0, 0, 0] : [255, 255, 255]) });
  assert.equal(await decodedDelta('primary', 'control', 'video', command), 1);
  assert.equal(await decodedDelta('primary', 'primary', 'video', command), 0);
});
test('controlled subprocess executes, propagates failures and refuses log overwrite', async t => {
  const { dir } = await fixture(t);
  const options = { cwd: dir, env: process.env, logPath: path.join(dir, 'process.log'), timeoutSeconds: 2 };
  await runProcess([process.execPath, '-e', 'console.log("actually ran")'], options);
  assert.match(await readFile(options.logPath, 'utf8'), /actually ran/);
  await assert.rejects(runProcess([process.execPath, '-e', 'process.exit(0)'], options), /EEXIST/);
  await assert.rejects(runProcess([process.execPath, '-e', 'process.exit(7)'], { ...options, logPath: path.join(dir, 'failed.log') }), /failed/);
  await assert.rejects(runProcess([process.execPath, '-e', 'setTimeout(()=>{},100)'], { ...options, timeoutSeconds: 0.01, logPath: path.join(dir, 'timeout.log') }), /timeout/);
});
test('missing executions cannot yield campaign acceptance; local native run is refused', async t => {
  const { p } = await fixture(t); await assert.rejects(verifyCampaign(p), /ENOENT/);
  if (process.env.GITHUB_ACTIONS !== 'true') await assert.rejects(runCase(p, p.cases[0].id, '.', '.'), /Actions-only/);
});
const response = (value, status = 200) => ({ status, text: async () => JSON.stringify(value) });
const scenario = () => ({ baseUrl: 'http://127.0.0.1:43210', caseId: 'mlx.generalization.ltx-add-select-generate', timeoutSeconds: 5, steps: [
  { id: 'import', method: 'POST', path: '/api/v1/models/import', expectedStatus: 201, body: { name: 'Imported H3', sourcePath: '/private-fixture/model.safetensors', baseModel: 'minimax_h3', minimaxH3Partition: 'ref2va' }, assert: [{ pointer: '/id', equals: 'import-job' }] },
  { id: 'installed', waitJob: { $capture: 'import/id' }, assert: [{ pointer: '/result/modelId', equals: 'imported-model' }] },
  { id: 'generate', method: 'POST', path: '/api/v1/video/jobs', expectedStatus: 201, body: { projectId: 'private-project', model: { $capture: 'installed/result/modelId' }, mode: 'reference_to_video', prompt: 'A person waves', referenceAssetIds: ['private-ref'] }, assert: [{ pointer: '/payload/model', equals: 'imported-model' }] },
  { id: 'generated', waitJob: { $capture: 'generate/id' }, assert: [{ pointer: '/status', equals: 'completed' }] },
] });
test('real app sequence uses existing import/select/video job endpoints and captured imported model', async () => {
  const calls = [];
  const results = [response({ id: 'import-job' }, 201), response({ status: 'completed', result: { modelId: 'imported-model' } }), response({ id: 'video-job', payload: { model: 'imported-model' } }, 201), response({ status: 'completed' })];
  const result = await runScenario(scenario(), { fetchImpl: async (url, options) => { calls.push([url.pathname, options]); return results.shift(); } });
  assert.equal(calls[0][0], '/api/v1/models/import'); assert.equal(calls[2][0], '/api/v1/video/jobs');
  assert.equal(JSON.parse(calls[2][1].body).model, 'imported-model');
  assert.equal(result.transcript.filter(r => r.assertion).length, 4);
});
test('API failure/timeout cancels actual submitted job, not the GPU process', async () => {
  const s = scenario(); s.steps = s.steps.slice(0, 2); const calls = [];
  await assert.rejects(runScenario(s, { fetchImpl: async (url, options) => { calls.push([url.pathname, options.method]); return calls.length === 1 ? response({ id: 'import-job' }, 201) : response({ status: 'failed' }); } }), /did not complete/);
  assert.deepEqual(calls.at(-1), ['/api/v1/jobs/import-job/cancel', 'POST']);
  let time = 0;
  await assert.rejects(runScenario(s, { now: () => time, sleep: async ms => { time += ms; }, fetchImpl: async url => url.pathname.endsWith('/import') ? response({ id: 'import-job' }, 201) : response({ status: 'running' }) }), /timeout/);
});
test('API assertions fail on dropped selected-model propagation and external services', async () => {
  const s = scenario(); s.baseUrl = 'https://unrelated.example'; await assert.rejects(runScenario(s));
  s.baseUrl = 'http://localhost:43210'; s.steps = [s.steps[0]];
  await assert.rejects(runScenario(s, { fetchImpl: async () => response({ id: 'wrong-job' }, 201) }));
  assert.throws(() => pointer({}, '/missing'), /missing/);
});
test('manual workflow is execution-only and uses existing runner/token infrastructure without global locks', async () => {
  const workflow = await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml', import.meta.url), 'utf8');
  assert.match(workflow, /workflow_dispatch:/); assert.doesNotMatch(workflow, /^  (push|pull_request|schedule):/m);
  assert.match(workflow, /rw-chroma/); assert.match(workflow, /real-weights/); assert.match(workflow, /cancel-in-progress: false/);
  assert.match(workflow, /SCENEWORKS_INFERENCE_READ_TOKEN/); assert.match(workflow, /stationary_pair_authorized/);
  assert.doesNotMatch(workflow, /registration_only|pkill|flock|gpu.token/i);
});
test('delivery records reject old CI, wrong main pin and unresolved limitations', async t => {
  const { p } = await fixture(t);
  p.delivery = { requiredCI: [{ headSha: sw, conclusion: 'success', url: 'https://example.test/ci' }], featureEndReview: { sceneworks: sw, inference: inf, unresolvedFindings: 0, url: 'https://example.test/review' }, inferenceMain: { sha: inf, url: 'https://example.test/inference-main' }, sceneworksMain: { sha: 'e'.repeat(40), sourceCommit: sw, inferencePin: inf, url: 'https://example.test/sw-main' }, shortcutReadbacks: [26679,26691].map(id => ({ id, observedState: 'In Progress', readAt: '2026-10-24T00:00:00Z', url: `https://app.shortcut.com/trefry/${id}` })), limitations: [] };
  validateDelivery(p);
  for (const mutate of [d => d.requiredCI[0].headSha = 'f'.repeat(40), d => d.sceneworksMain.inferencePin = sw, d => d.limitations.push('Missing CUDA AV')]) {
    const copy = structuredClone(p); mutate(copy.delivery); assert.throws(() => validateDelivery(copy));
  }
});
test('newly prepared native derivatives are verified by bytes without invented pre-run hashes', async t => {
  const f = await fixture(t);
  const derivative = path.join(f.dir, 'native-derivative'); await writeFile(derivative, 'actual new derivative');
  f.e.loadEvent.prepared_sources = JSON.stringify([{ path: derivative, sha256: await hashFile(derivative) }]);
  await f.save(); await f.verify();
  await writeFile(derivative, 'drifted derivative'); await assert.rejects(f.verify(), /prepared source content changed/);
});
test('replaced executable reference output cannot pass numerical fidelity', async t => {
  const f = await fixture(t, 'numeric'); await f.verify();
  await writeFile(path.join(f.dir, 'reference.json'), '[2,4,6]');
  await writeFile(path.join(f.dir, 'candidate.json'), '[2,4,6]');
  await assert.rejects(f.verify(), /reference output was replaced/);
});
test('renamed stock weights cannot impersonate a frozen HF artifact with internally consistent receipts', async t => {
  const f = await fixture(t); await f.verify();
  const catalog = JSON.parse(await readFile(path.join(f.dir, 'source-catalog.json'), 'utf8'));
  catalog.siblings[0].lfs.sha256 = 'f'.repeat(64);
  await writeFile(path.join(f.dir, 'source-catalog.json'), JSON.stringify(catalog));
  await assert.rejects(f.verify(), /not the frozen source artifact/);
});
test('generation reclassification cannot bypass mandatory AV or metrics', async t => {
  const f = await fixture(t); f.c.group = 'lifecycle'; delete f.e.av; delete f.e.metrics; await f.save();
  assert.throws(() => validatePlan(f.p)); await assert.rejects(f.verify(), /reclassification/);
});
test('refusal lifecycle and arithmetic fixtures need executed input proof, not invented successful load receipts', async t => {
  for (const group of ['lifecycle', 'numeric']) {
    const f = await fixture(t, group);
    delete f.e.runtimeIdentity; delete f.e.completedJob; delete f.e.loadEvent; delete f.e.metrics;
    await f.save(); await f.verify();
    delete f.e.testInputSha256; await f.save(); await assert.rejects(f.verify(), /test input identity/);
  }
});
test('three individual source mutations are killed by executed assertions, not syntax errors', async t => {
  const { dir } = await fixture(t);
  const original = await readFile(new URL('./h3-finetune-terminal.mjs', import.meta.url), 'utf8');
  for (const file of ['h3-finetune-terminal-api.mjs', 'h3-finetune-terminal.test.mjs']) await writeFile(path.join(dir, file), await readFile(new URL(file, import.meta.url)));
  for (const [needle, pattern] of [
    ['assert.equal(e.inference, p.inference);', 'receipt mutation rejects wrong source revision'],
    ["assert.equal(+m[3], 0, 'ignored tests are not evidence');", 'test execution evidence rejects ignored'],
    ['assert.ok(Math.abs(v - r) <= tolerance.atol + tolerance.rtol * Math.abs(r), `numeric mismatch at ${i}`);', 'numeric comparison discriminates scaling'],
  ]) {
    assert.equal(original.split(needle).length, 2, 'mutation must reach exactly one guard');
    const mutated = path.join(dir, 'h3-finetune-terminal.mjs');
    await writeFile(mutated, original.replace(needle, '/* source mutation: removed guard */'));
    assert.equal(spawnSync(process.execPath, ['--check', mutated]).status, 0, 'non-compiling mutation is invalid');
    const childEnv = { ...process.env }; delete childEnv.NODE_TEST_CONTEXT;
    const run = spawnSync(process.execPath, ['--test', `--test-name-pattern=${pattern}`, path.join(dir, 'h3-finetune-terminal.test.mjs')], { encoding: 'utf8', env: childEnv });
    assert.equal(run.status, 1, `mutation survived: ${pattern}\n${run.stdout}\n${run.stderr}`);
    assert.match(run.stdout + run.stderr, /Missing expected (exception|rejection)/, 'must fail the discriminating assertion');
  }
});
