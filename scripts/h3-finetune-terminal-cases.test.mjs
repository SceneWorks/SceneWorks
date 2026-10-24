import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, rm, access } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { requiredCases, ARTIFACTS, SOURCE, sensitivityDecision, compareControlMedia } from './h3-finetune-terminal.mjs';
import { caseBook, concretePlan, buildScenario, nativeCommands, referenceCommand, prepareCases, fixtureBindings } from './h3-finetune-terminal-cases.mjs';
import { runScenario } from './h3-finetune-terminal-api.mjs';
const root = path.resolve(fileURLToPath(new URL('..', import.meta.url)));
// All identities below are explicitly synthetic test data, never staged model identity.
const hash = '1'.repeat(64), revision = '2'.repeat(40);
const params = () => ({ baseUrl: 'http://127.0.0.1:43210', projectId: 'synthetic-project', seed: 314159, requestedGpu: 'synthetic-device', timeoutSeconds: 300, sensitivity: { minimumControlDelta: 0.01, maximumBaselineDelta: 0.001 },
  assets: { imageA: 'img-red', imageB: 'img-blue', clipA: 'clip-forward', clipB: 'clip-flipped', voiceA: 'public-speech', voiceB: 'reversed-public-speech' } });
function execution(partition = 'ref2va', topology = 'full') {
  return { label: 'synthetic', model: { modelId: 'imported_synthetic', sourcePath: '/synthetic-only/model.safetensors', sha256: hash, baseModel: partition === 'ref2va' ? 'minimax_h3_ref' : 'minimax_h3', partition, topology, supportedModes: ['text_to_video', 'image_to_video', 'first_last_frame', 'reference_to_video'], componentTier: 'q4', components: [{ sha256: hash, source: 'synthetic', revision, licenseAcknowledged: true }] },
    mode: partition === 'ref2va' ? 'reference_to_video' : 'text_to_video', sampling: { steps: 50, duration: 5.1667, fps: 24, width: 576, height: 320, contractSource: 'synthetic declared recipe only' } };
}
const named = (group, check, backend = 'mlx') => requiredCases().find(c => c.id === `${backend}.${group}.${check}`);
const submissions = s => s.steps.filter(x => x.path === '/api/v1/video/jobs');
function mockAPI() {
  const jobs = new Map(), calls = []; let n = 0;
  return { calls, fetchImpl: async (url, options) => {
    calls.push([url.pathname, options]);
    let value, status = 200;
    if (url.pathname === '/api/v1/models/import' || url.pathname === '/api/v1/video/jobs') {
      const payload = JSON.parse(options.body), id = `job${++n}`;
      value = { id, status: 'queued', payload };
      jobs.set(id, { id, status: 'completed', payload, result: payload.modelId ? { modelId: payload.modelId } : { assets: ['synthetic-output'] } });
      status = 201;
    } else value = jobs.get(url.pathname.split('/').at(-1));
    assert.ok(value, 'mock handler is only the existing import/video/poll route shape');
    return { status, text: async () => JSON.stringify(value) };
  } };
}
test('casebook source metadata binds all four exact frozen artifact identities', () => {
  assert.deepEqual(caseBook.source, SOURCE); assert.deepEqual(caseBook.artifacts.map(a => a.artifact), ARTIFACTS);
  for (const a of caseBook.artifacts) { assert.match(a.expectedSha256, /^[a-f0-9]{64}$/); assert.ok(a.expectedBytes > 1e9); assert.match(a.pointerBlob, /^[a-f0-9]{40}$/); }
  assert.equal(new Set(caseBook.artifacts.map(a => a.expectedSha256)).size, 4);
});
test('plan derives full inventory from existing canonical contract, including last-only', () => {
  const p = concretePlan(root, '/owned-native-source', '/owned-stage');
  assert.deepEqual(p.cases.map(c => c.id), requiredCases().map(c => c.id));
  assert.equal(p.cases.filter(c => c.artifact).length, 8);
  for (const backend of ['mlx', 'cuda']) assert.ok(p.cases.some(c => c.id === `${backend}.capability.last-frame-only`));
  for (const c of p.cases) assert.ok(c.commands.length > 0 && c.commands.every(argv => argv.every(v => typeof v === 'string')));
  assert.ok(!Object.hasOwn(p, 'sceneworks') && !Object.hasOwn(p, 'inference'), 'no invented future stationary SHA');
  assert.ok(!JSON.stringify(p).includes('produce-evidence'), 'no nonexistent producer');
});
test('native commands use actual provider package, single integration binary, exact ignored cases and unfiltered listing', () => {
  for (const backend of ['mlx', 'cuda']) for (const check of ['convrot', 'pruned-interpolation', 'w4a8']) {
    const c = nativeCommands('/native', backend, check);
    assert.ok(c.inventory.includes('integration')); assert.deepEqual(c.inventory.slice(-2), ['--', '--list']);
    for (const argv of c.commands) { assert.ok(argv.includes('--exact') && argv.includes('--ignored')); assert.ok(argv.includes(backend === 'mlx' ? 'mlx-gen-minimax-h3' : 'candle-gen-minimax-h3')); assert.equal(argv.includes('cuda'), backend === 'cuda'); }
  }
});
test('reference commands call real pinned fixture programs, outside source outputs, no upstream execution here', () => {
  for (const check of ['convrot', 'pruned-interpolation', 'w4a8']) {
    const a = referenceCommand('/native', check, '/private-reference', '/private-output');
    assert.deepEqual(a.slice(0, 2), ['python3', '-I']); assert.ok(a.includes(path.join('/native', caseBook.numeric[check].program)));
    assert.equal(a.includes('--reference-dir'), check !== 'convrot'); assert.ok(a.includes('--output'));
  }
  assert.equal(caseBook.numeric.w4a8.referenceRevision, 'd908d589fbda44595f6d78d9c47ef42ba4aa8fad');
  assert.equal(caseBook.numeric.w4a8.kernelBlobs.length, 4);
  assert.equal(caseBook.numeric['pruned-interpolation'].referenceRevision, '47b06ff920c16b4b7b36de068367a5de0e0968e9');
});
test('supporting fixture selectors resolve to actual SceneWorks source, not invented test names', async () => {
  const sourceFiles = {
    sceneworksCore: await readFile(path.join(root, 'crates/sceneworks-core/tests/minimax_h3_import.rs'), 'utf8'),
    worker: await readFile(path.join(root, 'crates/sceneworks-worker/src/video_jobs/minimax_h3_import_tests.rs'), 'utf8'),
    api: await readFile(path.join(root, 'apps/rust-api/src/models.rs'), 'utf8'),
  };
  for (const bindings of Object.values(fixtureBindings)) for (const [pkg, target, selector] of Object.values(bindings)) {
    if (pkg === 'sceneworks-core' && target !== 'minimax_h3_import') { await access(path.join(root, `crates/sceneworks-core/tests/${target}.rs`)); continue; }
    const text = sourceFiles[pkg === 'sceneworks-worker' ? 'worker' : pkg === 'sceneworks-rust-api' ? 'api' : 'sceneworksCore'];
    assert.ok(text.includes(`fn ${selector.split('::').at(-1)}(`), `missing actual selector ${selector}`);
  }
});
for (const [check, mode, partition, field] of [
  ['text', 'text_to_video', 'fl2va', 'prompt'], ['image', 'image_to_video', 'fl2va', 'sourceAssetId'],
  ['first-last-frame', 'first_last_frame', 'fl2va', 'lastFrameAssetId'], ['last-frame-only', 'image_to_video', 'fl2va', 'lastFrameAssetId'],
  ['video', 'reference_to_video', 'ref2va', 'sourceClipAssetIds'], ['audio-reference', 'reference_to_video', 'ref2va', 'referenceAudioAssetIds'],
]) test(`concrete ${check} scenario uses actual mode/field, captured imported ID and repeated baseline`, async () => {
  const e = execution(partition); e.mode = mode;
  const s = buildScenario(named('capability', check), params(), [e]);
  const bodies = submissions(s).map(s => s.body);
  assert.ok(bodies.length >= 3); assert.deepEqual(bodies[0], bodies[1]); assert.notDeepEqual(bodies[1][field], bodies[2][field]);
  assert.deepEqual(bodies[0].model, { $capture: 'synthetic_installed/result/modelId' });
  if (check === 'last-frame-only') assert.ok(!Object.hasOwn(bodies[0], 'sourceAssetId'));
  const api = mockAPI(), result = await runScenario(s, { fetchImpl: api.fetchImpl });
  assert.ok(result.transcript.filter(r => r.assertion).length > 0);
  for (const [, o] of api.calls.filter(([p]) => p === '/api/v1/video/jobs')) assert.equal(JSON.parse(o.body).model, e.model.modelId);
});
test('ordered multimodal cases reverse and drop each list independently, including audio comparisons', () => {
  const s = buildScenario(named('capability', 'ordered-multimodal'), params(), [execution()]);
  const b = submissions(s).map(s => s.body); assert.equal(b.length, 8);
  for (const field of ['referenceAssetIds', 'sourceClipAssetIds', 'referenceAudioAssetIds']) {
    assert.ok(b.some(v => JSON.stringify(v[field]) === JSON.stringify([...b[0][field]].reverse())));
    assert.ok(b.some(v => v[field].length === 0));
  }
  assert.equal(s.comparisons[0].interventions.filter(v => v.stream === 'audio').length, 2);
});
test('eight artifact scenarios refuse stock substitution and use explicit ref2va base lineage', () => {
  for (const c of requiredCases().filter(c => c.group === 'artifact')) {
    const e = execution(); e.model.sha256 = caseBook.artifacts.find(a => a.artifact === c.artifact).expectedSha256;
    const s = buildScenario(c, params(), [e]); assert.equal(submissions(s).length, 1);
    assert.equal(s.steps[0].body.baseModel, 'minimax_h3_ref'); assert.equal(s.steps[0].body.minimaxH3Partition, 'ref2va');
    e.model.sha256 = hash; assert.throws(() => buildScenario(c, params(), [e]));
  }
});
test('wrong partition/base, unsupported modes, missing components, one-step and missing assets fail preparation', () => {
  for (const mutate of [e => e.model.baseModel = 'minimax_h3', e => e.model.supportedModes = [], e => e.model.components = [], e => e.sampling.steps = 1, e => e.model.components[0].licenseAcknowledged = false]) {
    const e = execution(); mutate(e); assert.throws(() => buildScenario(named('capability', 'audio-reference'), params(), [e]));
  }
  const p = params(); delete p.assets.voiceA; assert.throws(() => buildScenario(named('capability', 'audio-reference'), p, [execution()]));
});
test('full/pruned/hybrid adapter representatives and every declared turbo recipe cannot be hand-trimmed', () => {
  for (const kind of ['lora', 'lokr', 'turbo']) assert.throws(() => buildScenario(named('capability', kind), params(), [execution()]));
  for (const kind of ['lora', 'lokr']) {
    const es = ['full', 'pruned', 'hybrid'].map(topology => {
      const e = execution(topology === 'hybrid' ? 'hybrid' : 'ref2va', topology === 'hybrid' ? 'full' : topology); e.label = topology;
      if (topology === 'hybrid') e.model.companion = { source: 'synthetic companion', sha256: hash };
      e.adapters = [{ id: `synthetic_${kind}`, kind, sha256: hash, source: 'synthetic-only', revision, compatibilitySource: 'synthetic contract', topology: e.model.topology, partition: e.model.partition, licenseAcknowledged: true }]; return e;
    });
    const s = buildScenario(named('capability', kind), params(), es), bodies = submissions(s).map(s => s.body);
    for (let i = 0; i < bodies.length; i += 3) { assert.equal(bodies[i].loras.length, 1); assert.equal(bodies[i+2].loras.length, 0); assert.deepEqual(bodies[i].advanced, bodies[i+2].advanced); }
    es[0].adapters[0].topology = 'other'; assert.throws(() => buildScenario(named('capability', kind), params(), es));
  }
});
test('hybrid plan requires explicit companion and exercises text, image and reference routes', () => {
  const es = ['text_to_video', 'image_to_video', 'reference_to_video'].map((mode, i) => {
    const e = execution('hybrid', 'full'); e.label = `route${i}`; e.mode = mode; e.model.companion = { source: 'synthetic companion', sha256: hash }; return e;
  });
  assert.ok(buildScenario(named('capability', 'hybrid'), params(), es).steps.length > 0);
  delete es[0].model.companion; assert.throws(() => buildScenario(named('capability', 'hybrid'), params(), es));
});
test('cancel/retry uses actual job handlers and never fabricates a successful native model load', () => {
  const e = { label: 'cancel', model: { modelId: 'imported_cancel', sourcePath: '/synthetic/input', sha256: hash, baseModel: 'minimax_h3_ref', partition: 'ref2va' } };
  for (const check of ['cancel', 'interruption-retry']) {
    const s = buildScenario(named('lifecycle', check), params(), [e]);
    assert.equal(submissions(s).length, 0); assert.ok(s.steps.some(s => s.expectedJobStatus === 'cancelled'));
    assert.equal(s.steps.some(s => s.path?.$concat?.includes('/retry')), check === 'interruption-retry');
  }
});
test('restart replay duplicates an actual persisted imported job after external task-owned restart', () => {
  const e = execution(); e.reuseInstalled = true; e.replayJobId = 'persisted-original-job';
  const s = buildScenario(named('lifecycle', 'restart-replay'), params(), [e]);
  assert.equal(s.steps[1].path, '/api/v1/jobs/persisted-original-job/duplicate'); assert.equal(s.steps[1].expectedStatus, 201);
  assert.ok(!s.steps.some(s => s.path === '/api/v1/models/import'));
  delete e.replayJobId; assert.throws(() => buildScenario(named('lifecycle', 'restart-replay'), params(), [e]));
});
test('failed replay/retry assertions cooperatively cancel the newly submitted actual job ID', async () => {
  for (const operation of ['duplicate','retry']) {
    const calls=[];
    const s={baseUrl:params().baseUrl,caseId:`mlx.lifecycle.${operation}`,timeoutSeconds:10,steps:[{id:'submit',method:'POST',path:{$concat:['/api/v1/jobs/','original',`/${operation}`]},body:{},expectedStatus:201,assert:[{pointer:'/payload/model',equals:'imported_expected'}]}]};
    await assert.rejects(runScenario(s,{fetchImpl:async (url,options)=>{calls.push([url.pathname,options.method]); return {status:url.pathname.endsWith('/cancel')?200:201,text:async()=>JSON.stringify({id:'new-job',payload:{model:'wrong-stock'}})};}}));
    assert.deepEqual(calls.at(-1),['/api/v1/jobs/new-job/cancel','POST']);
  }
});
test('lifecycle fixture-only cases refuse to masquerade as a generation scenario', () => {
  for (const check of ['source-drift', 'leases', 'duplicate', 'relink-rescan', 'provenance-license']) assert.throws(() => buildScenario(named('lifecycle', check), params(), [execution()]));
});
test('managed/linked equivalence and cold/warm/staged require each actual execution binding', () => {
  assert.throws(() => buildScenario(named('lifecycle', 'managed-linked-equivalence'), params(), [execution()]));
  assert.throws(() => buildScenario(named('lifecycle', 'cold-warm-staged'), params(), [execution()]));
  const a = execution(), b = execution(); b.label = 'linked'; b.ownershipMode = 'linked'; b.linkedRootId = 'owned-root'; b.linkedRelativePath = 'source.safetensors';
  const s = buildScenario(named('lifecycle', 'managed-linked-equivalence'), params(), [a,b]);
  const linked = s.steps.find(s => s.id === 'linked_import').body; assert.equal(linked.ownershipMode, 'linked'); assert.ok(!Object.hasOwn(linked, 'sourcePath'));
});
test('assets carry executable CPU ffmpeg recipes and pinned public speech provenance, no personal assets', () => {
  for (const key of ['imageA','imageB','clipA','clipB','tone']) assert.equal(caseBook.assets[key].command[0], 'ffmpeg');
  assert.match(caseBook.assets.voice.url, /5be91486e11a2d616f4ec5db8d3fd248585ac07a/);
  assert.match(caseBook.assets.voice.upstreamLicense, /CC-BY-4.0/); assert.match(caseBook.assets.tone.meaning, /NEVER voice/);
});
test('trained turbo recipes retain every variant and reject invented cheap steps/geometry', () => {
  const es = caseBook.sampling.turbo.map((r, i) => {
    const e = execution(i === 3 ? 'ref2va' : i === 2 ? 'hybrid' : 'fl2va', i === 1 ? 'pruned' : 'full'); e.label = `turbo${i}`;
    if (i === 2) e.model.companion = { source: 'synthetic companion', sha256: hash };
    Object.assign(e.sampling, { steps: r.steps, schedulerShift: r.schedulerShift, audioSchedulerShift: r.audioSchedulerShift, width: r.trainingShortEdge * 2, height: r.trainingShortEdge });
    e.adapters = [{ id: r.id, kind: 'lora', sha256: hash, source: 'synthetic', revision, topology: e.model.topology, partition: e.model.partition, compatibilitySource: 'synthetic', licenseAcknowledged: true }]; return e;
  });
  assert.ok(buildScenario(named('capability', 'turbo'), params(), es).steps.length > 0);
  es[0].sampling.steps = 1; assert.throws(() => buildScenario(named('capability', 'turbo'), params(), es));
});
test('LTX parity and independent/format generalization cannot pass with mislabeled H3/stock inputs', () => {
  for (const check of ['ltx-add-select-generate','independent-finetune','renamed','native-single','sharded-diffusers','dense-f16','dense-bf16','dense-f32']) assert.throws(() => buildScenario(named('generalization', check), params(), [execution()]));
});
test('sharded sources retain per-file content closure without a fabricated directory checksum', () => {
  const e=execution(); e.model.sourceFormat='sharded-diffusers'; e.model.sourceKind='directory'; e.model.files=[{path:'/synthetic-only/shard1.safetensors',sha256:hash},{path:'/synthetic-only/shard2.safetensors',sha256:'3'.repeat(64)}];
  const s=buildScenario(named('generalization','sharded-diffusers'),params(),[e]);
  assert.ok(!Object.hasOwn(s.steps[0].body,'expectedSha256'));
  e.model.sha256='4'.repeat(64); assert.throws(()=>buildScenario(named('generalization','sharded-diffusers'),params(),[e]));
});
test('decoded sensitivity rejects hash-only change, sub-threshold effects and nondeterministic noise', async () => {
  assert.deepEqual(sensitivityDecision(0.08,0.01,0.02,0.02), { interventionDelta:0.08, repeatedBaselineDelta:0.01, threshold:0.03 });
  for (const args of [[0,0,0.01,0.01],[0.02,0.01,0.01,0.02],[0.1,0.03,0.01,0.02],[NaN,0,0.01,0.01]]) assert.throws(() => sensitivityDecision(...args));
  const decoded = [], deltas = [0.001,0.1];
  const r = await compareControlMedia('base','repeat','changed','audio',0.01,0.005, async p => { decoded.push(p); return { synthetic:p }; }, async () => deltas.shift());
  assert.deepEqual(decoded, ['base','repeat','changed']); assert.equal(r.stream,'audio'); assert.equal(r.interventionDelta,0.1);
});
test('complete canonical preparation writes executable scenarios without executing commands or inventing receipts', async t => {
  const parent = await mkdtemp(path.join(os.tmpdir(), 'sc26691-complete-prep-')); t.after(() => rm(parent, { recursive: true, force: true }));
  const out = path.join(parent, 'candidate'), p = { ...params(), sceneworks: revision, inference: revision, swRoot: root, inferenceRoot: '/synthetic-native', bindings: {} };
  const plan = concretePlan(root, '/synthetic-native', out);
  for (const c of plan.cases.filter(c => c.scenario)) {
    const check = c.id.split('.').slice(2).join('.'); let es = [execution()];
    if (c.artifact) es[0].model.sha256 = caseBook.artifacts.find(a => a.artifact === c.artifact).expectedSha256;
    if (c.group === 'capability') {
      const mode = { text:'text_to_video', image:'image_to_video', 'last-frame-only':'image_to_video', 'first-last-frame':'first_last_frame' }[check];
      if (mode) { es = [execution('fl2va')]; es[0].mode = mode; }
      if (check === 'hybrid') es = ['text_to_video','image_to_video','reference_to_video'].map((mode,i) => { const e=execution('hybrid'); e.label=`mode${i}`; e.mode=mode; e.model.companion={source:'synthetic',sha256:hash}; return e; });
      if (['lora','lokr','turbo'].includes(check)) {
        const recipes = check === 'turbo' ? caseBook.sampling.turbo : [{}, {}, {}];
        es = recipes.map((r,i) => { const e=execution(i===2?'hybrid':i===3?'ref2va':'fl2va',i===1?'pruned':'full'); e.label=`adapter${i}`;
          if (i===2) e.model.companion={source:'synthetic',sha256:hash};
          if (check==='turbo') Object.assign(e.sampling,{steps:r.steps,schedulerShift:r.schedulerShift,audioSchedulerShift:r.audioSchedulerShift,width:r.trainingShortEdge*2,height:r.trainingShortEdge});
          e.adapters=[{id:r.id??`synthetic_${check}`,kind:check==='turbo'?'lora':check,sha256:hash,source:'synthetic',revision,compatibilitySource:'synthetic',topology:e.model.topology,partition:e.model.partition,licenseAcknowledged:true}]; return e; });
      }
    }
    if (check==='ltx-add-select-generate') { es[0].family='ltx'; es[0].model.baseModel='ltx_2_3'; }
    if (check==='independent-finetune') es[0].model.lineage={source:'synthetic independent',revision};
    if (check==='renamed') Object.assign(es[0].model,{originalPath:'/synthetic-only/original.safetensors',originalSha256:hash});
    if (['native-single','sharded-diffusers'].includes(check)) es[0].model.sourceFormat=check;
    if (check==='sharded-diffusers') Object.assign(es[0].model,{sourceKind:'directory',files:[{path:'/synthetic-only/shard.safetensors',sha256:hash}]});
    if (check.startsWith('dense-')) es[0].model.sourceFormat=check.slice(6);
    if (check==='restart-replay') { es[0].reuseInstalled=true; es[0].replayJobId='synthetic-before-restart'; }
    if (check==='managed-linked-equivalence') { const b=execution(); b.label='linked'; b.ownershipMode='linked'; b.linkedRootId='synthetic-root'; b.linkedRelativePath='model.safetensors'; es.push(b); }
    if (check==='cold-warm-staged') es=['cold','warm','staged'].map(label=>({...execution(),label,reuseInstalled:label!=='cold'}));
    p.bindings[c.id]=es;
  }
  const result=await prepareCases(p,out);
  assert.equal(result.status,'prepared-not-executed'); assert.equal(result.cases,requiredCases().length);
  assert.equal(result.scenarios,plan.cases.filter(c=>c.scenario).length);
  for (const c of plan.cases.filter(c=>c.scenario)) assert.equal(JSON.parse(await readFile(c.scenario,'utf8')).caseId,c.id);
  await assert.rejects(access(path.join(out,'evidence.json')));
  await assert.rejects(prepareCases(p,out),/EEXIST/);
});
test('prepare fails before creating output when final pair or concrete bindings are missing', async t => {
  const parent = await mkdtemp(path.join(os.tmpdir(), 'sc26691-prep-test-')); t.after(() => rm(parent, { recursive: true, force: true }));
  const out = path.join(parent, 'candidate');
  await assert.rejects(prepareCases({}, out));
  await assert.rejects(prepareCases({ ...params(), sceneworks: revision, inference: revision, swRoot: root, inferenceRoot: '/native' }, out));
  await assert.rejects(access(out));
});
