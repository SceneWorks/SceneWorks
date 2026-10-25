import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, readFile, rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import os from 'node:os';
import { runInNewContext } from 'node:vm';
import path from 'node:path';
import { ARTIFACTS, SOURCE, requiredCases, validatePlan, validateDelivery, validatePeakMetrics, hashFile, verifyInputs, compareNumeric, executedTests, decodeAV, decodedDelta, validateEvidence, runProcess, runCase, verifyCampaign } from './h3-finetune-terminal.mjs';
import { runScenario, pointer } from './h3-finetune-terminal-api.mjs';
const sw = 'a'.repeat(40), inf = 'b'.repeat(40);
const log = 'test result: ok. 1 passed; 0 failed; 0 ignored;';
function peakObservation(bytes, method, phase) {
  const start = '2026-10-24T00:00:00.000Z', end = '2026-10-24T00:00:00.050Z';
  const cumulative = ['mlx-allocator-cumulative-high-water', 'os-rss-high-water'].includes(method);
  return { method, sampled: method.startsWith('sampled-'), intervalMs: method.startsWith('sampled-') ? 50 : null,
    bounds: [...(phase === 'loading' ? ['full-case-load-inclusive-upper-bound'] : []), ...(method === 'sampled-device-used' ? ['observed-device-global-bound'] : [])],
    samples: [{ at: start, bytes: bytes / 2 }, { at: end, bytes }], window: { start, end, phase, coverage: phase === 'loading' ? 'full-case-load-inclusive' : 'full-case' },
    attribution: { pid: process.pid, deviceId: 'synthetic-device', scope: cumulative ? 'process-lifetime' : method === 'sampled-device-used' ? 'device' : 'process', meaning: 'Synthetic CPU observation fixture, not real measurement', includesPriorAllocations: cumulative, includesOtherProcesses: method === 'sampled-device-used' },
    blindspots: method.startsWith('sampled-') ? ['Transients between samples can be missed'] : cumulative ? ['Counter includes allocations before the observation window'] : [] };
}
async function fixture(t, group = 'artifact') {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'h3-terminal-fixture-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const input = path.join(dir, 'synthetic-checkpoint'); await writeFile(input, 'not a model');
  const hash = await hashFile(input);
  const p = { version: 1, sceneworks: sw, inference: inf, source: SOURCE, output: dir, responsibleTask: 'sc-26691', nextAuthorizedAction: 'Coordinator authorizes exact next Actions case; no automatic retry.', cases: requiredCases().map(c => ({ ...c, command: ['node', 'controlled-fixture.mjs'], timeoutSeconds: 10, evidence: 'evidence.json', loadHostBudgetBytes: 1000, loadDeviceBudgetBytes: 2000, minimumControlDelta: 0.1, controlStream: 'video', referenceRevision: 'c'.repeat(40), referenceCommand: ['node', 'controlled-reference.mjs'], referenceOutput: 'reference.json', tolerance: { atol: 0.001, rtol: 0.001 }, inputs: [{ path: input, sha256: hash, role: 'checkpoint', artifact: c.artifact }, { path: input, sha256: hash, role: 'reference' }] })) };
  const c = p.cases.find(c => c.group === group);
  const e = { status: 'executed', sceneworks: sw, inference: inf, caseId: c.id, backend: c.backend, source: SOURCE, artifact: c.artifact, hardware: { device: 'synthetic CPU fixture only', driver: 'fixture' }, runtimeIdentity: { checkpointSha256: hash, planId: 'synthetic-plan', semanticDigest: hash, topology: 'fixture', quantization: 'fixture', components: [], adapters: [], operation: 'fixture', samplingPolicy: { steps: 2, schedulerShift: null }, seed: 42 }, metrics: { hostPeakBytes: 100, devicePeakBytes: 200, loadHostPeakBytes: 90, loadDevicePeakBytes: 180, observations: { host: peakObservation(100, 'sampled-process-rss', 'case'), device: peakObservation(200, 'sampled-native-allocated', 'case'), loadHost: peakObservation(90, 'sampled-process-rss', 'loading'), loadDevice: peakObservation(180, 'sampled-native-allocated', 'loading') } }, files: ['trace.json'], av: 'synthetic-av.bin', control: { intervention: 'replace conditioning', checkpointSha256: hash, seed: 42, av: 'control-av.bin' }, reference: { revision: c.referenceRevision, sha256: hash }, numeric: { candidate: 'candidate.json', reference: 'reference.json' } };
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
  ['residency deltas mislabeled as peaks', e => e.metrics.observations.device.method = 'pre-post-residency-delta'],
  ['missing load peak', e => delete e.metrics.loadDevicePeakBytes],
  ['loading budget exceeded', e => { e.metrics.loadHostPeakBytes = 1001; e.metrics.observations.loadHost.samples[1].bytes = 1001; }],
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
  assert.match(workflow, /workflow_call:/); assert.doesNotMatch(workflow,/^  workflow_dispatch:/m,'only the existing registered terminal routes may call asset ensure'); assert.doesNotMatch(workflow, /^  (push|pull_request|schedule):/m);
  assert.match(workflow, /rw-chroma/); assert.match(workflow, /real-weights/); assert.match(workflow, /cancel-in-progress: false/);
  assert.match(workflow, /SCENEWORKS_INFERENCE_READ_TOKEN/); assert.match(workflow, /stationary_pair_authorized/);
  assert.doesNotMatch(workflow, /registration_only|pkill|flock|gpu.token/i);
});
const workflowJob = (text, name) => {
  const start=text.indexOf(`\n  ${name}:\n`); assert.ok(start>=0,`missing ${name} job`);
  const next=text.slice(start+1).search(/\n  [a-z][\w-]*:\n/); return text.slice(start,next<0?undefined:start+1+next);
};
const workflowExpression = (expression, inputs, github={}) => runInNewContext(expression.trim().replace(/^\$\{\{\s*|\s*\}\}$/g,''),{
  inputs,github:{event_name:'workflow_dispatch',run_id:'unit_fake_101',run_attempt:'1',ref:'refs/heads/unit_fake',...github},fromJSON:JSON.parse,toJSON:JSON.stringify,format:(template,...values)=>template.replace(/\{(\d+)\}/g,(_,n)=>String(values[Number(n)])),
},{timeout:100});
test('registered Candle entrypoint positively routes only explicit H3 selections to the same-ref reusable job',async()=>{
  const router=await readFile(new URL('../.github/workflows/server-candle-linux.yml',import.meta.url),'utf8');
  const callee=await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml',import.meta.url),'utf8');
  const job=workflowJob(router,'h3-terminal'),selected=/^    if: (.+)$/m.exec(job)[1],authorization=/^      stationary_pair_authorized: (.+)$/m.exec(job)[1];
  assert.match(job,/uses: \.\/.github\/workflows\/h3-finetune-terminal\.yml/);assert.doesNotMatch(job,/uses:.*@main|registration_only|activation/);assert.match(callee,/workflow_call:/);
  for(const operation of ['standard','','source','provision','readiness','campaign','diagnostic-candle-1b','h3-import-mlx','h3-import-cuda','h3-import-both','h3-admission-proof','h3-ui-proof','h3-stage-mlx','h3-stage-cuda','h3-terminal','unknown'])assert.equal(workflowExpression(selected,{starvector_terminal_operation:operation}),false,operation);
  const choices=/starvector_terminal_operation:[\s\S]*?options: \[([^\]]+)\]/.exec(router)[1].split(',').map(v=>v.trim());
  for(const operation of ['h3-terminal-mlx','h3-terminal-cuda']){
    assert.equal(choices.filter(v=>v===operation).length,1);
    const inputs={starvector_terminal_operation:operation,campaign_run_id:JSON.stringify({stationary_pair_authorized:true})};assert.equal(workflowExpression(selected,inputs),true);assert.equal(workflowExpression(authorization,inputs),true);
    for(const event of ['push','pull_request','merge_group','schedule'])assert.equal(workflowExpression(selected,inputs,{event_name:event}),false);
    const backend=/^      backend: (.+)$/m.exec(job)[1];assert.equal(workflowExpression(backend,inputs),operation.endsWith('-mlx')?'mlx':'cuda');
    for(const value of [false,null,1,'true','false'])assert.equal(workflowExpression(authorization,{...inputs,campaign_run_id:JSON.stringify({stationary_pair_authorized:value})}),false);
  }
  const native=workflowJob(callee,'terminal'),nativeIf=/^    if: (.+)$/m.exec(native)[1];
  for(const event of ['push','pull_request','merge_group','schedule'])assert.equal(workflowExpression(nativeIf,{stationary_pair_authorized:true,backend:'mlx'},{event_name:event}),false);
  assert.equal(workflowExpression(nativeIf,{stationary_pair_authorized:false,backend:'mlx'}),false);assert.equal(workflowExpression(nativeIf,{stationary_pair_authorized:true,backend:'unknown'}),false);
  for(const backend of ['mlx','cuda'])assert.equal(workflowExpression(nativeIf,{stationary_pair_authorized:true,backend}),true);
});
test('H3 bridge preserves all standing dispatch keys, old flow selection and isolated concurrency groups',async()=>{
  const router=await readFile(new URL('../.github/workflows/server-candle-linux.yml',import.meta.url),'utf8');
  const header=router.slice(router.indexOf('  workflow_dispatch:'),router.indexOf('\npermissions:'));
  const keys=[...header.matchAll(/^      ([a-z][\w_]+):$/gm)].map(m=>m[1]).sort();
  assert.deepEqual(keys,['starvector_terminal_operation','inference_revision','inference_preflight_run_id','inference_preflight_artifact','mac_app_data_source','mac_hf_home_source','windows_app_data_source','windows_hf_home_source','permanent_pin','campaign_run_id'].sort());
  const rootGroup=/^  group: (.+)$/m.exec(router)[1],cancel=/^  cancel-in-progress: (.+)$/m.exec(router)[1],build=/^    if: (.+)$/m.exec(workflowJob(router,'build-candle'))[1];
  for(const operation of ['standard','']){assert.equal(workflowExpression(build,{starvector_terminal_operation:operation}),true);assert.equal(workflowExpression(cancel,{starvector_terminal_operation:operation}),true);assert.match(workflowExpression(rootGroup,{starvector_terminal_operation:operation}),/^server-candle-linux-/);}
  for(const operation of ['source','provision','readiness','campaign','diagnostic-candle-1b']){assert.equal(workflowExpression(rootGroup,{starvector_terminal_operation:operation}),'starvector-terminal-bridge');assert.equal(workflowExpression(cancel,{starvector_terminal_operation:operation}),false);assert.equal(workflowExpression(build,{starvector_terminal_operation:operation}),false);}
  for(const operation of ['h3-terminal-mlx','h3-terminal-cuda']){const inputs={starvector_terminal_operation:operation};assert.equal(workflowExpression(cancel,inputs),false);assert.equal(workflowExpression(build,inputs),false);assert.notEqual(workflowExpression(rootGroup,inputs),workflowExpression(rootGroup,inputs,{run_id:'unit_fake_102',run_attempt:'1'}));assert.match(workflowExpression(rootGroup,inputs),/^h3-terminal-bridge-/);}
  for(const operation of ['h3-import-mlx','h3-import-cuda','h3-import-both']){const inputs={starvector_terminal_operation:operation};assert.equal(workflowExpression(cancel,inputs),false);assert.equal(workflowExpression(build,inputs),false);assert.equal(workflowExpression(rootGroup,inputs),'h3-import-native-unit_fake_101');assert.equal(workflowExpression(rootGroup,inputs,{run_id:'unit_fake_102'}),'h3-import-native-unit_fake_102');}
  const cpuInputs={starvector_terminal_operation:'h3-admission-proof'};assert.equal(workflowExpression(rootGroup,cpuInputs),'h3-admission-cpu-unit_fake_101');assert.equal(workflowExpression(rootGroup,cpuInputs,{run_id:'unit_fake_102'}),'h3-admission-cpu-unit_fake_102');assert.equal(workflowExpression(cancel,cpuInputs),false);assert.equal(workflowExpression(build,cpuInputs),false);
  for(const operation of ['h3-import-mlx','h3-import-cuda','h3-import-both','h3-admission-proof','h3-ui-proof','h3-terminal-mlx','h3-terminal-cuda'])for(const event of ['push','pull_request','merge_group','schedule'])assert.equal(workflowExpression(rootGroup,{starvector_terminal_operation:operation},{event_name:event}),'starvector-terminal-bridge');
  for(const operation of ['unknown','h3-import','h3-terminal']){const inputs={starvector_terminal_operation:operation};assert.equal(workflowExpression(build,inputs),false);assert.equal(workflowExpression(rootGroup,inputs),'starvector-terminal-bridge');}
  for(const [name,file] of [['source-closure','starvector-terminal-source.yml'],['provision','starvector-terminal-provision.yml'],['readiness','starvector-terminal-readiness.yml'],['campaign','starvector-terminal.yml']])assert.ok(workflowJob(router,`starvector-${name}`).includes(`uses: ./.github/workflows/${file}`));
});
test('H3 caller forwards only explicit request fields and the standing read token without duplicate execution logic',async()=>{
  const router=await readFile(new URL('../.github/workflows/server-candle-linux.yml',import.meta.url),'utf8'),job=workflowJob(router,'h3-terminal');
  for(const field of ['sceneworks_sha','inference_sha','expected_runner','plan_path','output_path','case_id'])assert.ok(new RegExp(field+': \\$\\{\\{ fromJSON\\(inputs\\.campaign_run_id\\)\\.'+field+'( \\|\\| \'\')? \\}\\}').test(job),field);
  assert.doesNotMatch(job,/runs-on:|run-case|cargo |permanent_pin|inference_revision|secrets: inherit/);
  assert.match(job,/SCENEWORKS_INFERENCE_READ_TOKEN: \$\{\{ secrets\.SCENEWORKS_INFERENCE_READ_TOKEN \|\| secrets\.INFERENCE_READ_TOKEN \}\}/);
  const callee=await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml',import.meta.url),'utf8');assert.match(callee,/SCENEWORKS_INFERENCE_READ_TOKEN:\n        required: false/);assert.match(callee,/group: h3-finetune-terminal-\$\{\{ github.run_id \}\}-\$\{\{ github.run_attempt \}\}/);
});
test('terminal reuses native device queues and binds manual CUDA zero without changing ordinary queues',async()=>{
  const workflow=await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml',import.meta.url),'utf8'),job=workflowJob(workflow,'terminal');
  const group=/^      group: (.+)$/m.exec(job)[1];assert.equal(workflowExpression(group,{backend:'cuda'}),'windows-candle-gpu-real-weights');assert.equal(workflowExpression(group,{backend:'mlx'}),'h3-import-proofs-nax-macos-2');assert.match(job,/cancel-in-progress: false/);
  for(const [key,value]of [['CUDA_DEVICE_ORDER','PCI_BUS_ID'],['CUDA_VISIBLE_DEVICES','0'],['SCENEWORKS_GPU_ID','0']]){const expression=new RegExp(`^          ${key}: (.+)$`,'m').exec(job)[1];assert.equal(workflowExpression(expression,{backend:'cuda'}),value);}
});
test('actual reusable preflight rejects source-ref drift before product execution without a main activation prerequisite',async t=>{
  const workflow=await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml',import.meta.url),'utf8'),code=/node --input-type=module -e '\n([\s\S]*?)\n          '/.exec(workflow)?.[1];assert.ok(code);
  assert.doesNotMatch(code,/workflow.*register|default.*branch|gh workflow|activation/);
  const dir=await mkdtemp(path.join(os.tmpdir(),'h3-dispatch-unit-fake-'));t.after(()=>rm(dir,{recursive:true,force:true}));const file=path.join(dir,'plan.json');await writeFile(file,JSON.stringify({sceneworks:sw,inference:inf,output:dir,cases:[{id:'mlx.artifact.0',backend:'mlx'}]}));
  const env={...process.env,PLAN:file,OUTPUT:dir,CASE_ID:'mlx.artifact.0',BACKEND:'mlx',EXPECTED_RUNNER:'unit_fake_runner',RUNNER_NAME:'unit_fake_runner',SW_SHA:sw,INF_SHA:inf,GITHUB_SHA:sw};
  const good=spawnSync(process.execPath,['--input-type=module','-e',code],{env,encoding:'utf8'});assert.equal(good.status,0,good.stderr);
  const bad=spawnSync(process.execPath,['--input-type=module','-e',code],{env:{...env,GITHUB_SHA:inf},encoding:'utf8'});assert.equal(bad.status,1);assert.match(bad.stderr,/caller workflow commit/);
});
test('terminal compiler selection derives only the checked-out root pin without executing Rust locally',async t=>{
  const workflow=await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml',import.meta.url),'utf8');
  const reader=workflow.slice(workflow.indexOf('      - name: Read the checked-out root Rust pin'),workflow.indexOf('      - name: Prepare the root-pinned compiler'));
  assert.match(reader,/working-directory: sceneworks/);
  const code=/node --input-type=module -e '\n([\s\S]*?)\n          '/.exec(reader)?.[1];assert.ok(code);
  const dir=await mkdtemp(path.join(os.tmpdir(),'h3-toolchain-unit-fake-'));t.after(()=>rm(dir,{recursive:true,force:true}));
  const rootPin=/^channel = "([^"]+)"$/m.exec(await readFile(new URL('../rust-toolchain.toml',import.meta.url),'utf8'))[1];
  const output=path.join(dir,'environment'),paths=path.join(dir,'paths'),cargoHome=path.join(dir,'cargo');
  const run=()=>spawnSync(process.execPath,['--input-type=module','-e',code],{cwd:dir,env:{...process.env,GITHUB_ENV:output,GITHUB_PATH:paths,CARGO_HOME:cargoHome},encoding:'utf8'});
  for(const pin of [rootPin,'9.8.7']){
    await writeFile(path.join(dir,'rust-toolchain.toml'),`[toolchain]\nchannel = "${pin}"\n`);await writeFile(output,'');await writeFile(paths,'');
    const result=run();assert.equal(result.status,0,result.stderr);assert.equal(await readFile(output,'utf8'),`RUSTUP_TOOLCHAIN=${pin}\n`);assert.equal(await readFile(paths,'utf8'),path.join(cargoHome,'bin')+'\n');
  }
  for(const source of ['[toolchain]\nchannel = "stable"\n','[toolchain]\nchannel = "1.98"\n','[toolchain]\n','channel = "9.8.7"\nchannel = "9.8.6"\n']){
    await writeFile(path.join(dir,'rust-toolchain.toml'),source);await writeFile(output,'');assert.notEqual(run().status,0);assert.equal(await readFile(output,'utf8'),'');
  }
  const prepare=workflow.slice(workflow.indexOf('      - name: Prepare the root-pinned compiler'),workflow.indexOf('      - name: Execute or reverify'));
  assert.match(prepare,/rustup toolchain install "\$RUSTUP_TOOLCHAIN" --profile minimal --no-self-update/);assert.match(prepare,/rustup run "\$RUSTUP_TOOLCHAIN" rustc --version/);assert.doesNotMatch(prepare,/rustup (?:default|override)|1\.97\.1/);
});
test('source UI proof is explicit hosted Chromium only, independent of model pin and campaign authorization',async()=>{
  const router=await readFile(new URL('../.github/workflows/server-candle-linux.yml',import.meta.url),'utf8'),job=workflowJob(router,'h3-ui-proof');
  const {PLAYWRIGHT_VERSION}=await import('../apps/web/verify/hosted-browser/guard.mjs');assert.ok(job.includes(`playwright@${PLAYWRIGHT_VERSION}`));assert.match(job,/SC26687_EXPECTED_HEAD: \$\{\{ github.sha \}\}/);
  const choices=/starvector_terminal_operation:[\s\S]*?options: \[([^\]]+)\]/.exec(router)[1].split(',').map(v=>v.trim());assert.equal(choices.filter(v=>v==='h3-ui-proof').length,1);
  const selected=/^    if: (.+)$/m.exec(job)[1];
  for(const operation of [undefined,'','unknown',...choices])for(const event of ['push','pull_request','merge_group','schedule','workflow_dispatch'])assert.equal(workflowExpression(selected,{starvector_terminal_operation:operation},{event_name:event}),event==='workflow_dispatch'&&operation==='h3-ui-proof');
  for(const name of [...router.slice(router.indexOf('\njobs:')).matchAll(/^  ([a-z][\w-]*):$/gm)].map(m=>m[1]).filter(v=>v!=='h3-ui-proof'))assert.equal(workflowExpression(/^    if: (.+)$/m.exec(workflowJob(router,name))[1],{starvector_terminal_operation:'h3-ui-proof'}),false,`${name} refuses UI proof`);
  assert.match(job,/runs-on: ubuntu-latest/);assert.match(job,/timeout-minutes: 20/);assert.match(job,/ref: \$\{\{ github.sha \}\}/);assert.match(job,/fetch-depth: 0/);assert.match(job,/persist-credentials: false/);
  for(const text of ['test "$RUNNER_ENVIRONMENT" = github-hosted','test "$GITHUB_ACTIONS" = true','SC26687_HOSTED_BROWSER_PROOF: "1"','npm ci --prefix apps/web','playwright@1.59.1','SC26687_PLAYWRIGHT_ROOT','sc26687-playwright-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}','PLAYWRIGHT_BROWSERS_PATH','resolvePlaywrightEntry','"$SC26687_PLAYWRIGHT_ROOT/node_modules/.bin/playwright" install --with-deps chromium','playwright-resolution.log','node apps/web/verify/hosted-browser/run-hosted-browser.mjs','set -euo pipefail','source-inputs.sha256','if-no-files-found: error'])assert.ok(job.includes(text),text);
  assert.doesNotMatch(job,/--prefix apps\/web playwright|npm install --no-save --package-lock=false --prefix apps\/web|npx --no-install playwright/,'Playwright must not mutate the apps/web npm graph');
  const driver=await readFile(new URL('../apps/web/verify/hosted-browser/run-hosted-browser.mjs',import.meta.url),'utf8'),guard=await readFile(new URL('../apps/web/verify/hosted-browser/guard.mjs',import.meta.url),'utf8');
  assert.match(driver,/resolvePlaywrightEntry\(\{ root: process\.env\.SC26687_PLAYWRIGHT_ROOT \}\)/);assert.match(driver,/pathToFileURL\(playwrightTools\.entry\)/);assert.doesNotMatch(driver,/await import\("playwright"\)|require\.resolve\("playwright\/package\.json"\)|NODE_PATH/);
  assert.match(guard,/export function resolvePlaywrightEntry/);assert.match(guard,/export function resolveBrowsersPath/);assert.doesNotMatch(guard,/NODE_PATH/);
  assert.doesNotMatch(job,/playwright@[~^]|permanent_pin|stationary_pair_authorized|run-case|cargo |self-hosted|CUDA_VISIBLE_DEVICES|HF_HOME|secrets: inherit/);
  assert.match(job,/if: \$\{\{ always\(\) \}\}/);assert.match(job,/tee "\$SC26687_ARTIFACT_DIR\/browser-proof.log"/);
  const rootGroup=/^  group: (.+)$/m.exec(router)[1],cancel=/^  cancel-in-progress: (.+)$/m.exec(router)[1],inputs={starvector_terminal_operation:'h3-ui-proof'};
  assert.equal(workflowExpression(rootGroup,inputs),'h3-ui-hosted-unit_fake_101-1');assert.equal(workflowExpression(rootGroup,inputs,{run_id:'unit_fake_102'}),'h3-ui-hosted-unit_fake_102-1');assert.equal(workflowExpression(rootGroup,inputs,{run_attempt:'2'}),'h3-ui-hosted-unit_fake_101-2');assert.equal(workflowExpression(cancel,inputs),false);
});
test('CPU bridge provisions its pinned private interpreter before model assets and records the observed closure',async()=>{
  const source=await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml',import.meta.url),'utf8'),start=source.indexOf('      - name: Prepare private CPU-only conditioning and source-variant tools'),assets=source.indexOf('      - name: Ensure selected assets and conditioning in this terminal route');
  assert.ok(start>0&&start<assets,'import/help failure must precede massive model downloads');const setup=source.slice(start,assets);
  for(const token of ['torch==2.9.1','https://download.pytorch.org/whl/cpu','H3_PROBE_PYTHON','cpu-tools-freeze.txt','cpu-tools.json','cpu-bridge-help.txt','--help','sys.version_info[:2]==(3,12)','"-I","-B"'])assert.ok(setup.includes(token),token);
  assert.ok(source.includes('python-version: "3.12"'));assert.doesNotMatch(setup,/--no-deps|PYTHONPATH|einops|diffusers/);assert.ok(setup.includes('numericProof:false'));
  const adapter=await readFile(new URL('./h3-finetune-terminal-artifact-probe.mjs',import.meta.url),'utf8');assert.ok(adapter.includes("process.env.H3_PROBE_PYTHON"));assert.ok(adapter.includes("const producerCommand=[python,'-I','-B'"));
});
test('asset staging is folded into the terminal delivery and runs the reduced 25/backend inventory',async()=>{
  const router=await readFile(new URL('../.github/workflows/server-candle-linux.yml',import.meta.url),'utf8'),callee=await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml',import.meta.url),'utf8');
  const choices=/starvector_terminal_operation:[\s\S]*?options: \[([^\]]+)\]/.exec(router)[1].split(',').map(v=>v.trim());
  assert.deepEqual(choices,['standard','source','provision','readiness','campaign','diagnostic-candle-1b','h3-import-mlx','h3-import-cuda','h3-import-both','h3-admission-proof','h3-ui-proof','h3-terminal-mlx','h3-terminal-cuda']);
  for(const value of ['h3-stage-mlx','h3-stage-cuda'])assert.equal(choices.filter(v=>v===value).length,0,`${value} selector removed`);
  assert.doesNotMatch(router,/\n  h3-stage:\n/);assert.doesNotMatch(router,/h3-assets-stage-/);
  const caller=workflowJob(router,'h3-terminal');assert.ok(caller.includes('campaign_request: ${{ inputs.campaign_run_id }}'));
  const terminal=workflowJob(callee,'terminal');
  const ensure=callee.indexOf('      - name: Ensure selected assets and conditioning in this terminal route');
  const resolve=callee.indexOf('      - name: Resolve the reduced operational case set for this backend');
  const validate=callee.indexOf('      - name: Validate explicit inputs before any product execution');
  const run=callee.indexOf('      - name: Execute or reverify the reduced operational cases');
  assert.ok(ensure>0&&resolve>ensure&&validate>resolve&&run>validate,'ensure -> reduced-case resolve -> validate -> run ordering');
  const ensureSlice=callee.slice(ensure,resolve);
  for(const token of ['runStage','h3-finetune-terminal-stage.mjs','prepareReferenceFixtures','receipt.identityControls','expected-assets-not-integrity-proof.json','stage-binding.json','bound-plan.json','blocked_by_allowed_roots','H3_PLAN_EFFECTIVE','license_authority','set -euo pipefail'])assert.ok(ensureSlice.includes(token),token);
  assert.doesNotMatch(ensureSlice,/run-case|acknowledged:\s*true|licenseAcknowledged:\s*true/,'staging must not synthesize license consent or execute cases');
  for(const token of ['pyarrow==19.0.1','h3-finetune-terminal-speech.py','h3-finetune-terminal-float-variants.py'])assert.ok(callee.includes(token),token);
  const resolveSlice=callee.slice(resolve,validate);
  for(const token of ['h3-finetune-terminal.mjs','executionCases','validatePlan','case-ids.json','25','18'])assert.ok(resolveSlice.includes(token),token);
  const runSlice=callee.slice(run);
  for(const token of ['H3_TERMINAL_CASES','run-case','H3_TERMINAL_SKIP_RUN','H3_PLAN_EFFECTIVE'] )assert.ok(runSlice.includes(token),token);
  assert.doesNotMatch(runSlice,/CASE_ID" "\$GITHUB_WORKSPACE\/sceneworks" "\$GITHUB_WORKSPACE\/inference"\n/, 'single-case invocation replaced by the reduced loop');
  assert.match(callee,/if: (?:\$\{\{ )?always\(\)/);
  assert.match(callee,/h3-terminal-stage-\$\{\{ inputs\.backend \}\}/);
  assert.match(callee,/if-no-files-found: warn/);
});
test('bootstrap request branches before the first plan read and skips restaging when prepared',async()=>{
  const router=await readFile(new URL('../.github/workflows/server-candle-linux.yml',import.meta.url),'utf8'),callee=await readFile(new URL('../.github/workflows/h3-finetune-terminal.yml',import.meta.url),'utf8');
  const header=callee.slice(callee.indexOf('  workflow_call:'),callee.indexOf('permissions:'));
  for(const field of ['plan_path','output_path'])assert.match(header,new RegExp(field+':\\n\\s+required: false\\n\\s+type: string'),field);
  const cpu=callee.indexOf('      - name: Prepare private CPU-only conditioning and source-variant tools');
  const bootstrap=callee.indexOf('      - name: Bootstrap the reduced terminal campaign when requested');
  const ensure=callee.indexOf('      - name: Ensure selected assets and conditioning in this terminal route');
  assert.ok(cpu>0&&bootstrap>cpu&&ensure>bootstrap,'CPU Torch preflight -> bootstrap -> ensure ordering');
  const bootSlice=callee.slice(bootstrap,ensure);
  for(const token of ['h3-finetune-terminal-bootstrap.mjs','bootstrapTerminal','request.bootstrap','sc-26691-h3-cases-v1','reduced-v1','componentSelections','legalAuthority','H3_PLAN_EFFECTIVE','H3_OUTPUT_EFFECTIVE','H3_BOOTSTRAP_PREPARED','GITHUB_OUTPUT','RUNNER_TEMP','GITHUB_RUN_ATTEMPT','H3_SCENEWORKS_SHA','H3_INFERENCE_SHA','set -euo pipefail'])assert.ok(bootSlice.includes(token),token);
  assert.match(bootSlice,/prepared:\s*true|prepared,\s*true/);assert.match(bootSlice,/proof:\s*false|proof,\s*false/);
  assert.doesNotMatch(bootSlice,/h3-finetune-terminal\.mjs"?\s+"?run-case|run-case/, 'bootstrap only prepares; it does not execute cases');
  const ensureSlice=callee.slice(ensure,callee.indexOf('      - name: Derive explicit dense source variants'));
  assert.ok(ensureSlice.indexOf('H3_BOOTSTRAP_PREPARED')>=0&&ensureSlice.indexOf('H3_BOOTSTRAP_PREPARED')<ensureSlice.indexOf('readFile('),'bootstrap branch must precede the first plan read');
  for(const token of ['external plan_path is required when request.bootstrap is absent','stageRequested && !bootstrapPrepared','binding.status="bootstrap-prepared"','H3_OUTPUT_EFFECTIVE','H3_INFERENCE_SHA'])assert.ok(ensureSlice.includes(token),token);
  assert.doesNotMatch(ensureSlice,/prepareIdentityControls\(/,'bootstrap path must not redo identity controls');
  const choices=/starvector_terminal_operation:[\s\S]*?options: \[([^\]]+)\]/.exec(router)[1].split(',').map(v=>v.trim());
  assert.equal(choices.length,13);assert.doesNotMatch(router,/\n  h3-bootstrap:\n/);
  assert.match(callee,/timeout-minutes: 1440/);assert.match(callee,/budgetSeconds:86400/);
  assert.match(callee,/H3_OUTPUT_EFFECTIVE/);
});
test('ordinary check selects the CPU-only preparation tests rather than a terminal/model command',async()=>{
  const p=JSON.parse(await readFile(new URL('../package.json',import.meta.url),'utf8'));
  for(const name of ['h3-finetune-terminal.test.mjs','h3-finetune-terminal-cases.test.mjs','h3-finetune-terminal-runtime.test.mjs'])assert.ok(p.scripts.check.includes(`scripts/${name}`));
  assert.doesNotMatch(p.scripts.check,/h3-finetune-terminal(?:-runtime)?\.mjs\s+(?:run-case|api|numeric)|gh workflow run/);
});
test('delivery records reject old CI, wrong main pin and unresolved limitations', async t => {
  const { p } = await fixture(t);
  p.delivery = { requiredCI: [{ headSha: sw, conclusion: 'success', url: 'https://example.test/ci' }], featureEndReview: { scope: 'epic-requirements-and-production-delta', sceneworks: sw, inference: inf, unresolvedFindings: 0, url: 'https://example.test/review' }, inferenceMain: { sha: inf, url: 'https://example.test/inference-main' }, sceneworksMain: { sha: 'e'.repeat(40), sourceCommit: sw, inferencePin: inf, url: 'https://example.test/sw-main' }, shortcutReadbacks: [26679,26691].map(id => ({ id, observedState: 'In Progress', readAt: '2026-10-24T00:00:00Z', url: `https://app.shortcut.com/trefry/${id}` })), limitations: [] };
  validateDelivery(p);
  for (const mutate of [d => d.requiredCI[0].headSha = 'f'.repeat(40), d => d.sceneworksMain.inferencePin = sw, d => d.limitations.push('Missing CUDA AV')]) {
    const copy = structuredClone(p); mutate(copy.delivery); assert.throws(() => validateDelivery(copy));
  }
  const prior=structuredClone(p);
  prior.delivery.featureEndReview.sceneworks='c'.repeat(40);prior.delivery.featureEndReview.inference='d'.repeat(40);
  assert.throws(()=>validateDelivery(prior),/focused late-delta/);
  prior.delivery.lateDeltaReview={scope:'late-pin-generated-repair-delta',baseSceneworks:prior.delivery.featureEndReview.sceneworks,baseInference:prior.delivery.featureEndReview.inference,sceneworks:sw,inference:inf,unresolvedFindings:0,url:'https://example.test/late-review'};
  validateDelivery(prior);
  for(const mutate of [d=>d.featureEndReview.scope='pin-only',d=>d.featureEndReview.unresolvedFindings=1,d=>d.featureEndReview.sceneworks='not-a-sha',d=>d.lateDeltaReview.baseSceneworks=sw,d=>d.lateDeltaReview.baseInference=inf,d=>d.lateDeltaReview.sceneworks='e'.repeat(40),d=>d.lateDeltaReview.inference='f'.repeat(40),d=>d.lateDeltaReview.unresolvedFindings=1,d=>d.lateDeltaReview.scope='second-whole-branch-review',d=>delete d.lateDeltaReview.url]){
    const copy=structuredClone(prior);mutate(copy.delivery);assert.throws(()=>validateDelivery(copy));
  }
});
test('honest sampled CUDA process/device-used peaks do not require native allocator highwater', async t => {
  const f = await fixture(t);
  for (const method of ['sampled-process-cuda-used', 'sampled-device-used']) {
    f.e.metrics.observations.device = peakObservation(200, method, 'case');
    f.e.metrics.observations.loadDevice = peakObservation(180, method, 'loading');
    validatePeakMetrics(f.e.metrics, { ...f.c, backend: 'cuda' });
  }
});
test('MLX cumulative allocator highwater is accepted honestly as process-global, not phase-local', async t => {
  const f = await fixture(t);
  f.e.metrics.observations.device = peakObservation(200, 'mlx-allocator-cumulative-high-water', 'case');
  f.e.metrics.observations.loadDevice = peakObservation(180, 'mlx-allocator-cumulative-high-water', 'loading');
  validatePeakMetrics(f.e.metrics, f.c);
  f.e.metrics.observations.loadDevice.attribution.includesPriorAllocations = false;
  assert.throws(() => validatePeakMetrics(f.e.metrics, f.c), /not window-local/);
});
test('sampled peak receipts reject missing cadence/attribution/blindspots, invented maxima and acquisition-only windows', async t => {
  const f = await fixture(t);
  for (const mutate of [
    o => delete o.intervalMs, o => o.attribution = {}, o => o.blindspots = [], o => o.bounds = [],
    o => o.samples = [], o => o.samples[1].bytes = 201,
    o => o.window.coverage = 'source-acquisition-through-provider-handle',
    o => o.samples[1].at = '2026-10-24T00:00:01Z',
  ]) {
    const metrics = structuredClone(f.e.metrics); mutate(metrics.observations.loadDevice);
    assert.throws(() => validatePeakMetrics(metrics, f.c));
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
test('whole-lane budget persists its original deadline across cases and dispatches',async t=>{
  const {p}=await fixture(t);const {laneBudget}=await import('./h3-finetune-terminal.mjs');const start=Date.parse('2026-10-25T00:00:00Z');
  const first=await laneBudget(p,'mlx',{now:start});assert.equal(first.remainingSeconds,86400);
  const resumed=await laneBudget(p,'mlx',{now:start+3600000,startedAt:'2026-10-26T00:00:00Z'});assert.equal(resumed.remainingSeconds,82800);assert.equal(resumed.startedAt,first.startedAt);
  await assert.rejects(laneBudget(p,'mlx',{now:start+86400001}),/original 24-hour/);
});
test('inline source binding rewrites the actual scenario and input hash together without creating consent',async t=>{
  const {p,c,dir}=await fixture(t);p.proof_kind='unit_fake';
  const {mkdir}=await import('node:fs/promises'),{bindStagedCampaignPlan}=await import('./h3-finetune-terminal-cases.mjs');
  const root=path.join(dir,'sc26691-private','101-1');await mkdir(root,{recursive:true});const source=c.inputs[0],target=path.join(root,'model.bin');await writeFile(target,await readFile(source.path));
  const scenarioPath=path.join(dir,'original-scenario.json'),scenario={caseId:c.id,steps:[{body:{sourcePath:source.path,prompt:source.path,licenseAcknowledged:false}}]};await writeFile(scenarioPath,JSON.stringify(scenario));
  c.runtime={scenario:scenarioPath,port:43210,gpuId:'unit_fake'};c.inputs.push({path:scenarioPath,sha256:await hashFile(scenarioPath),role:'scenario'});c.declaredSourceBindings=[{licenseAuthority:{kind:'unit_fake',acknowledged:false,reference:'NOT real approval'}}];
  const receiptPath=path.join(root,'h3-stage-receipt.json');await writeFile(receiptPath,JSON.stringify({proof_kind:'unit_fake',status:'completed',files:[{status:'completed',finalPath:target,expectedBytes:11,actualBytes:11,expectedSha256:source.sha256,observedSha256:source.sha256}]}));
  const result=await bindStagedCampaignPlan(p,{receiptPath,root,backend:'mlx',allowUnitFake:true}),bound=result.plan.cases.find(v=>v.id===c.id),actual=JSON.parse(await readFile(bound.runtime.scenario,'utf8'));
  assert.equal(actual.steps[0].body.sourcePath,target);assert.equal(actual.steps[0].body.prompt,source.path,'a prompt equal to a filesystem path remains literal creative text');assert.equal(actual.steps[0].body.licenseAcknowledged,false);assert.deepEqual(bound.declaredSourceBindings,c.declaredSourceBindings);
  assert.equal(bound.inputs.find(v=>v.role==='scenario').sha256,await hashFile(bound.runtime.scenario));assert.deepEqual(JSON.parse(await readFile(scenarioPath,'utf8')),scenario);assert.equal(result.plan.proof_kind,'unit_fake');
});
test('three individual source mutations are killed by executed assertions, not syntax errors', async t => {
  const { dir } = await fixture(t);
  const original = await readFile(new URL('./h3-finetune-terminal.mjs', import.meta.url), 'utf8');
  for (const file of ['h3-finetune-terminal-api.mjs', 'h3-finetune-terminal-reduced.mjs', 'h3-finetune-terminal.test.mjs']) await writeFile(path.join(dir, file), await readFile(new URL(file, import.meta.url)));
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
