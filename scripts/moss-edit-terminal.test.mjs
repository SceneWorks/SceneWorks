import test from 'node:test';
import assert from 'node:assert/strict';
import { Api, VERSION, digest, sha, makeNpy, inspectNpy, tensorData, validateAdmission, createPlan, validatePlan, primaryRows, replaceInputs, assertRequestBinding, checkJob, validateAdvancedDigests, memorySummary, samplesBracketJob, reviewVerdict, safePath, mcpArgs, isDenoiseStep, mcpResultAssets, validateExportedWorkflow, cacheComparison, reserveAttempt, resumeUiAttempt, resolveAttempt, failAttempt, assertDispatchClear, reconcileAttempt, json, save, fileIdentity, LATENT_UPSCALE_GUARDS, latentUpscaleGuard, validateSeededRecipeSelection } from './lib/moss-edit-terminal.mjs';
import { prepare, freeze, bind, frozen, submitReplay, submitGuard, measuredCacheComparison, waitJob, readCanceledJob, retainEditorSource, validateRetainedEditor, validateRecoveryAdjudication, acceptedRecoveryFailure, retainOutputs } from './moss-edit-terminal.mjs';
import { BACKEND_FILES, treeIdentity, validateArtifactInventory, executionBinding, validateBinding, validateEditorSource, pngRgbIdentity, validateInferencePins } from './lib/moss-edit-terminal.mjs';
import { mkdtemp, mkdir, writeFile, readFile, readdir, cp, unlink, open, rm, symlink, rename, truncate, realpath } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { deflateSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { sourceIdentity, filesUnder, rgbPng } from './lib/moss-edit-terminal-actions.mjs';
import { adapterRefusalState } from './moss-edit-terminal-actions.mjs';
import { executionTooling } from './lib/moss-edit-native-reuse.mjs';
import { verifyUiRequestBinding } from './moss-edit-terminal-browser.mjs';

globalThis.fetch = async () => { throw new Error('LIVE_API_FORBIDDEN_IN_PURE_TESTS'); };
const contract = await json(fileURLToPath(new URL('./fixtures/moss-edit-product-contract-v2.json',import.meta.url)));

// These are pure protocol/negative-control tests, never integrated acceptance.
const inputs = { provenance: 'generated inputs', single_case: { references: ['B'], width: 768, height: 512, prompt: 'single fixed prompt' }, four_case: { references: ['B','L','P','G'], width: 512, height: 768, prompt: 'four fixed prompt', mandatory_preservation: ['artist','bottle left','lamp right','poster'], typography_scope: 'exact lettering excluded before render' } };
const policy = { metric: 'rgb8-mae/v1', tolerance: 0.001, implementationSha256: 'a'.repeat(64) };
const plan = () => ({ ...createPlan(inputs), cacheComparisonPolicy: policy });
test('allocation is exact, includes all eight immutable quality cells and actual supplemental executions', () => {
  const p = plan(); validatePlan(p, inputs); assert.equal(p.rows.length, 50); assert.equal(p.renderExecutions, 50);
  assert.equal(p.rows.filter(r => r.category === 'quality').length, 8);
  assert.equal(p.requiredGuardCases.length*2,58);assert.equal(p.requiredExternal.length,10);
  assert.equal(p.rows.filter(r => r.output === 'latents').length, 16);
  for (const backend of ['mlx','cuda']) for (const transport of ['rest','mcp']) assert.ok(p.rows.some(r => r.backend === backend && r.transport === transport && r.category === 'advanced'));
});
test('quality requirements or guard omission cannot be silently weakened', () => {
  for (const mutate of [p => p.rows[0].request.seed++, p => p.rows[0].anchors.pop(), p => p.requiredGuardCases.pop(), p => p.rows.pop(), p => p.cacheComparisonPolicy = null]) {
    const p = plan(); mutate(p); assert.throws(() => validatePlan(p, inputs));
  }
});
test('tensor preparation retains F32 data and distinct file/content digests', () => {
  const bytes = makeNpy([2,3], 'float32', i => i / 7 - 0.25), metadata = inspectNpy(bytes);
  assert.deepEqual(metadata.shape, [2,3]); assert.equal(metadata.sha256, sha(bytes)); assert.notEqual(metadata.contentSha256, metadata.sha256);
  assert.equal(tensorData(bytes).readFloatLE(0), -0.25);
  assert.throws(() => inspectNpy(bytes, [1,128,4,4]));
});
test('nonfinite, wrong data length, nonbinary and disguised outputs fail', () => {
  assert.throws(() => makeNpy([1], 'float32', () => NaN)); assert.throws(() => makeNpy([1], 'uint8', () => 2));
  const bytes = makeNpy([1], 'float32', () => 1); assert.throws(() => inspectNpy(bytes.subarray(0,-1)));
  const wrong = Buffer.from(bytes); wrong.writeFloatLE(Infinity, wrong.length-4); assert.throws(() => inspectNpy(wrong));
  assert.throws(() => inspectNpy(Buffer.from('PNG pretending to be tensor')));
});
test('candidate-bound GPU1 and MLX identity requires no hardware lease or owner census', () => {
  const lock = { candidate: 'one' };
  const admission = { version: VERSION, lockSha256: digest(lock), backend: 'cuda', device: { uuid: 'GPU-actual1', pciOrRegistryId: '0000:45:00.0', name: 'actual device', physicalIndex: 1, logicalIndex: 0 }, cudaVisibleDevices: 'GPU-actual1', sceneWorksGpuId: '1', postCompile: true };
  validateAdmission(admission, lock, 'cuda');
  validateAdmission({ ...admission, exclusive: false, observedAt: '1970-01-01T00:00:00Z', expiresAt: '1970-01-01T00:00:00Z' }, lock, 'cuda');
  validateAdmission({ version: VERSION, lockSha256: digest(lock), backend: 'mlx', device: { uuid: 'actual-mlx', pciOrRegistryId: 'actual-registry-id', name: 'Apple Silicon', platform: 'apple-silicon' } }, lock, 'mlx');
  for (const mutate of [a => a.device.physicalIndex = 0, a => a.device.logicalIndex = 1, a => a.cudaVisibleDevices = 'GPU-other', a => a.sceneWorksGpuId = '0', a => a.postCompile = false, a => a.device.uuid = '', a => a.lockSha256 = digest({ candidate: 'other' }), a => a.backend = 'mlx']) {
    const copy = structuredClone(admission); mutate(copy); assert.throws(() => validateAdmission(copy, lock, 'cuda'));
  }
});
test('ordered duplicates and typed tensor identities survive materialization', () => {
  const row = { refs: ['B','L','B'], request: { advanced: { mossImageAdvanced: { positive: { values: '$positive' }, noise: '$noise' } } } };
  const uploaded = { B: { id: 'b' }, L: { id: 'l' }, positive: { id: 'p', type: 'tensor', tensor: { sha256: 'c'.repeat(64) } }, noise: { id: 'n', type: 'tensor', tensor: { sha256: 'd'.repeat(64) } } };
  const body = replaceInputs(row, uploaded, 'project'); assert.deepEqual(body.referenceAssetIds,['b','l','b']); assert.deepEqual(body.advanced.mossImageAdvanced.noise, { assetId: 'n', sha256: 'd'.repeat(64) });
  assert.throws(() => replaceInputs(row, {}, 'project'));
});
test('MCP maps only supported actual tool fields, preserving absent broadcast negative', () => {
  const request = { mode: 'edit_image', prompt: '', seed: null, advanced: { guidanceScale: 4, mossImageAdvanced: { negativePrompts: ['a','b'], output: 'latents' } } };
  const args = mcpArgs(request); assert.equal(args.seed, null); assert.equal(Object.hasOwn(args,'negativePrompt'), false); assert.deepEqual(args.mossImageAdvanced.negativePrompts,['a','b']);
  assertRequestBinding(request,{...request,negativePrompt:''}); // API persistence normalizes the omitted field.
  assert.throws(()=>assertRequestBinding(request,{...request,negativePrompt:'broadcast'}),/unsupported broadcast field/);
  assert.throws(()=>assertRequestBinding(request,{...request,negativePrompt:null}),/unsupported broadcast field/);
});
test('continued UI replay uses the repaired normalized request checker',async()=>{
  const expected={advanced:{mossImageAdvanced:{negativePrompts:['a','b']}},referenceAssetIds:['owned-input']},actual={...structuredClone(expected),negativePrompt:''};
  verifyUiRequestBinding(expected,actual);
  assert.throws(()=>verifyUiRequestBinding(expected,{...actual,negativePrompt:'broadcast'}),/unsupported broadcast field/);
  const controller=await readFile(fileURLToPath(new URL('./moss-edit-terminal-actions.mjs',import.meta.url)),'utf8');
  assert.match(controller,/repairTooling\?await import\(pathToFileURL\(resolve\(toolingRoot,'scripts\/moss-edit-terminal-browser\.mjs'\)\)\.href\)/);
});
test('manual job ID and terminal status cannot substitute actual persisted request', () => {
  assert.throws(() => assertRequestBinding({ seed: 173 }, { seed: 0 }));
  assert.throws(() => assertRequestBinding({}, { advanced: { mossImageSampleIndex: 0 } }));
  const row = primaryRows(inputs)[0]; const request = { ...row.request, projectId: 'p' };
  const job = { id: 'actual', projectId: 'p', payload: request, backend: 'mlx', status: 'completed', result: { assets: [{ id: 'a' }] }, workerId: 'native', startedAt: 'start', completedAt: 'end' };
  checkJob(row, request, job); assert.throws(() => checkJob(row, request, { ...job, backend: 'cuda' }));
  assert.throws(() => checkJob(row, request, { ...job, result: { ...job.result, outputPublication: { state: 'pending' } } }));
});

// Actual H43 run40388147087/1 Editor row. Request SHA6304a53281a2f6782a3903488fd35cc2746cf82696d9e23ffc7c48338ceb6ab4;
// terminal-job SHA383fd963735d8aa5638b338f220b9664874bba006a7f7148a2f07207be573f34.
// Keep every authored payload field; omit unrelated server-added catalog/result metadata only.
test('actual Editor job binds authored GPU routing on envelope while every generation field and UI POST stay strict',()=>{
  const request={"projectId":"project_bf98fa406d6e29990994d3e42e8c70a3","projectName":"MOSS edit sc-26752-final-r19","requestedGpu":"auto","mode":"edit_image","model":"moss_image_sft","prompt":"Change only the blue bottle body and blue cap to deep red. Keep its silver collar, handle shape, water droplets, exact silhouette and position unchanged. Preserve both orange pieces, pale stone surface, lighting, shadows, background, framing and empty space on the left. Do not add text or objects.","negativePrompt":"","seed":173,"count":1,"width":768,"height":512,"referenceAssetIds":["asset_67b88aa63ad9e007f68966ab1ba37e5c"],"advanced":{"mossImage":{"acceleration":"off","max_sequence_length":4096,"training_shift":1,"uncond_skip_layers":[]},"mossImageEdit":{"reference_max_edge":512,"use_kv_cache":true},"mossImageAdvanced":{"output":"images"},"steps":30,"guidanceScale":4,"schedulerShift":5}};
  const job={"id":"job_5c5b69b99e7d7ba5358c6f9c9bec6494","projectId":"project_bf98fa406d6e29990994d3e42e8c70a3","requestedGpu":"auto","assignedGpu":"1","backend":"1","status":"completed","workerId":"moss-sc-26752-final-r19-cuda","startedAt":"2026-10-26T20:00:54Z","completedAt":"2026-10-26T20:02:59Z","payload":{"projectId":"project_bf98fa406d6e29990994d3e42e8c70a3","projectName":"MOSS edit sc-26752-final-r19","mode":"edit_image","model":"moss_image_sft","prompt":"Change only the blue bottle body and blue cap to deep red. Keep its silver collar, handle shape, water droplets, exact silhouette and position unchanged. Preserve both orange pieces, pale stone surface, lighting, shadows, background, framing and empty space on the left. Do not add text or objects.","negativePrompt":"","seed":173,"count":1,"width":768,"height":512,"referenceAssetIds":["asset_67b88aa63ad9e007f68966ab1ba37e5c"],"advanced":{"guidanceScale":4,"mossImage":{"acceleration":"off","max_sequence_length":4096,"slg":null,"training_shift":1,"uncond_skip_layers":[]},"mossImageAdvanced":{"output":"images"},"mossImageEdit":{"reference_max_edge":512,"use_kv_cache":true},"schedulerShift":5,"steps":30}},"result":{"assets":[{"id":"asset_0c794a168ece4dec9750dbf5b4d2a788"}]}};
  const row={id:'quality-cuda-sft-single',backend:'cuda',expected:'completed',outputCount:1};
  assert.equal(Object.hasOwn(job.payload,'requestedGpu'),false);assert.equal(job.requestedGpu,'auto');assert.equal(job.assignedGpu,'1');
  checkJob(row,request,job,'1');
  for(const requestedGpu of [undefined,'0','1','mlx'])assert.throws(()=>checkJob(row,request,{...job,requestedGpu},'1'),/routing field requestedGpu/);
  for(const change of [{seed:174},{prompt:'changed'},{referenceAssetIds:['foreign-reference']},{advanced:{...job.payload.advanced,steps:29}},{advanced:{...job.payload.advanced,mossImageEdit:{...job.payload.advanced.mossImageEdit,use_kv_cache:false}}}])assert.throws(()=>checkJob(row,request,{...job,payload:{...job.payload,...change}},'1'),/frozen request field/);
  assert.throws(()=>checkJob(row,request,job,'0'),/worker launch contract/);
  assert.throws(()=>checkJob(row,request,{...job,backend:'0'},'1'),/worker launch contract/);
  verifyUiRequestBinding(request,structuredClone(request));assert.throws(()=>verifyUiRequestBinding(request,job.payload),/frozen request field requestedGpu/);
  assert.throws(()=>assertRequestBinding(request,{...request,requestedGpu:'0'}),/frozen request field requestedGpu/);
});

test('cancellation cannot publish images or stale completed reports', () => {
  const row = { expected: 'canceled', outputCount: 0, backend: 'mlx' }, request = { projectId: 'p' };
  const job = { id: 'j', projectId: 'p', payload: request, backend: 'mlx', status: 'canceled', result: { assets: [] } };
  checkJob(row, request, job);
  assert.throws(() => checkJob(row, request, { ...job, result: { assets: [{ id: 'a' }] } }));
  assert.throws(() => checkJob(row, request, { ...job, result: { mossImageEditExecutionReport: {} } }));
});
test('conditioning uses exact FP32 row+mask digest and caller noise has no fabricated seed', () => {
  const tensors = { positive: makeNpy([2,3,4096], 'float32', i => i/10000), negative: makeNpy([2,3,4096], 'float32', i => -i/10000), mask: makeNpy([2,3], 'uint8', i => [1,1,0,1,1,1][i]), negativeMask: makeNpy([2,3], 'uint8', i => [1,0,1,0,1,0][i]), noise: makeNpy([4,128,4,4], 'float32', i => i/1000) };
  const conditioning = Array.from({ length: 2 }, (_, b) => Object.fromEntries(['positive','negative'].map(branch => { const values = tensorData(tensors[branch]), mask = tensorData(tensors[branch === 'positive' ? 'mask' : 'negativeMask']).subarray(b*3,b*3+3); return [branch, { source: 'embeddings', sha256: sha(Buffer.concat([values.subarray(b*49152,(b+1)*49152),mask])), sequence_length: 3, valid_tokens: [...mask].reduce((a,n) => a+n,0) }]; })));
  const samples = Array.from({length:4},(_,i) => ({ seed: null, noise_sha256: sha(tensorData(tensors.noise).subarray(i*8192,(i+1)*8192)) }));
  const base = { conditioning, samples }, request = { advanced: { mossImageAdvanced: { positive: {}, negative: {}, noise: {} } } };
  validateAdvancedDigests(base,request,tensors); const mutant = structuredClone(base); mutant.samples[0].seed = 0; assert.throws(() => validateAdvancedDigests(mutant,request,tensors));
  mutant.samples[0].seed = null; mutant.conditioning[0].positive.sha256 = sha('pre-rounded bf16'); assert.throws(() => validateAdvancedDigests(mutant,request,tensors));
  for(const conditioning of [undefined,base.conditioning.slice(0,1),[...base.conditioning,base.conditioning[0]],[base.conditioning[1],base.conditioning[0]]])assert.throws(()=>validateAdvancedDigests({...base,conditioning},request,tensors));
  assert.throws(()=>validateAdvancedDigests(base,request,{...tensors,positive:undefined}));
  const wrongNoise=structuredClone(base);wrongNoise.samples[1].noise_sha256='f'.repeat(64);assert.throws(()=>validateAdvancedDigests(wrongNoise,request,tensors));
});
// Actual H45 run40396253038/1 job_5fdb29022c0e52cf2a7f9b4c0b345768.
// Projection of requestfe557e872d…394b and asset7b4910e9df…277f9; no model output fabricated.
test('actual H45 Editor output-only metadata preserves ordinary B1 text conditioning and generated noise',()=>{
 const request={prompt:'Change only the blue bottle body and blue cap to deep red. Keep its silver collar, handle shape, water droplets, exact silhouette and position unchanged. Preserve both orange pieces, pale stone surface, lighting, shadows, background, framing and empty space on the left. Do not add text or objects.',negativePrompt:'',count:1,seed:173,width:768,height:512,advanced:{mossImageAdvanced:{output:'images'}}};
 const base={conditioning:[{positive:{sequence_length:71,sha256:'3c30c24eb4bb292a41c0645808a54b2adedae8e977d685150f1918c851ee10ca',source:'text',valid_tokens:71},negative:{sequence_length:10,sha256:'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',source:'text',valid_tokens:10}}],samples:[{noise_sha256:'02c6f1b94c7334676fc6df334f9c5b3e3c6e27f6ec3309a32912d871b821edac',seed:173}]};
 validateAdvancedDigests(base,request,new Proxy({},{get(){throw Error('output-only must not read tiny tensor fixtures');}}));
 for(const change of [x=>x.conditioning=[],x=>x.conditioning[0].positive.source='embeddings',x=>x.conditioning[0].positive.sha256=sha('different prompt'),x=>x.conditioning[0].negative.sha256=sha('different negative'),x=>x.samples[0].seed=174,x=>x.samples[0].noise_sha256='missing']){const mutant=structuredClone(base);change(mutant);assert.throws(()=>validateAdvancedDigests(mutant,request,{}));}
});
test('B2 text negatives stay ordered and noise-only B1 checks actual complete tensor shape and digest',()=>{
 const request={prompts:['one','two'],seed:173,advanced:{mossImageAdvanced:{output:'images',negativePrompts:['bad one','bad two']}}},base={conditioning:['one','two'].map((prompt,i)=>({positive:{source:'text',sha256:sha(prompt)},negative:{source:'text',sha256:sha(request.advanced.mossImageAdvanced.negativePrompts[i])}})),samples:Array.from({length:4},(_,i)=>({seed:173+i,noise_sha256:'a'.repeat(64)}))};
 validateAdvancedDigests(base,request,{});for(const branch of ['positive','negative']){const mutant=structuredClone(base);mutant.conditioning[0][branch].sha256=base.conditioning[1][branch].sha256;assert.throws(()=>validateAdvancedDigests(mutant,request,{}));}
 const noise=makeNpy([1,128,4,4],'float32',i=>i/100),ordinary={prompt:'one',negativePrompt:'',seed:null,width:64,height:64,advanced:{mossImageAdvanced:{output:'images',noise:{assetId:'caller-noise'}}}},single={conditioning:[base.conditioning[0]],samples:[{seed:null,noise_sha256:sha(tensorData(noise))}]};single.conditioning[0]=structuredClone(single.conditioning[0]);single.conditioning[0].negative.sha256=sha('');
 validateAdvancedDigests(single,ordinary,{noise});assert.throws(()=>validateAdvancedDigests(single,ordinary,{noise:makeNpy([1,128,2,8],'float32',i=>i/100)}));
 const changed=structuredClone(single);changed.samples[0].noise_sha256='b'.repeat(64);assert.throws(()=>validateAdvancedDigests(changed,ordinary,{noise}));
});
test('canonical output retention precedes job and report semantics and never reads unowned media or unused NPY',async()=>{
 const root=await temporary('retain-output'),image=rgbPng(16,16,Buffer.alloc(16*16*3)),pixelSha='a'.repeat(64);
 await save(resolve(root,'input-identities.json'),{B:{width:16,height:16,pixelsSha256:pixelSha}}); // Deliberately no five tiny NPY files.
 for(const mode of ['success','job-failure','report-failure','extra-within-bound','foreign-project','foreign-lineage','unsafe-path']){
  const dest=resolve(root,mode);await mkdir(dest);
  const row={backend:'cuda',variant:'sft',refs:['B'],expected:'completed',output:'images',outputCount:1},request={projectId:'project',mode:'edit_image',model:'moss_image_sft',prompt:'one',negativePrompt:'',seed:173,count:1,width:16,height:16,advanced:{steps:30,guidanceScale:4,schedulerShift:5,mossImage:{acceleration:'off',training_shift:1,max_sequence_length:4096},mossImageEdit:{reference_max_edge:512,use_kv_cache:true},mossImageAdvanced:{output:'images'}}};
  const report={adapter_repository:'chinoll/Moss-Image-Edit',adapter_revision:'3be9c40037e1b89b17aa85ff6f8ee805872aa558',adapter_sha256:'17a48b8c605dcffdedb92a2b887d18cf316505be90802717669fb6556098dcdf',config_sha256:'b6b471cf7bc77b0874af03be618f8a06b6997d92b35de12c47a06b292ade2669',applied_projections:252,edit_contract:'ref-prefix-only-t0__target-text-original-moss-v1',reference_max_edge:512,use_kv_cache:true,references:[{pixels_sha256:pixelSha,original_dimensions:[16,16],encoded_dimensions:[16,16],resized_pixels_sha256:pixelSha,latent_sha256:pixelSha}],base:{execution_device:'cuda',resolved:{variant:'sft',steps:30,guidance:4,shift:5,training_shift:1,max_sequence_length:4096},output_kind:'images',latent_shape:[1,128,1,1],conditioning:[{positive:{source:'text',sha256:mode==='report-failure'?sha('wrong'):sha('one')},negative:{source:'text',sha256:sha('')}}],samples:[{seed:173,noise_sha256:'b'.repeat(64)}]}};
  const asset={id:'asset',projectId:mode==='foreign-project'?'foreign':'project',lineage:{jobId:mode==='foreign-lineage'?'foreign-job':'job'},file:{path:mode==='unsafe-path'?'../foreign.png':'images/output.png'},recipe:{rawAdapterSettings:{mossImageSampleIndex:0,mossImageEditExecutionReport:report}}},job={id:'job',projectId:'project',payload:request,status:'completed',backend:mode==='job-failure'?'0':'1',workerId:'owned',startedAt:'2026-10-26T21:00:00Z',completedAt:'2026-10-26T21:01:00Z',result:{assets:[{id:'asset'}]}};
  if(mode==='extra-within-bound')job.result.assets.push({id:'extra'});
  await save(resolve(dest,'terminal-job.json'),job);let mediaReads=0;
  const api={async request(path){if(path.endsWith('/workflow'))return {retained:true};if(path.endsWith('/assets/asset'))return asset;if(path.endsWith('/assets/extra'))return {...asset,id:'extra'};if(path.includes('/files/')){mediaReads++;return image;}throw Error('unexpected inert request');}};
  const run=()=>retainOutputs(api,row,request,job,dest,root,'1','owned');
  if(mode==='success')assert.equal((await run())[0].sha256,sha(image));else await assert.rejects(run);
  assert.deepEqual(await json(resolve(dest,'terminal-job.json')),job);
  if(['success','job-failure','report-failure','extra-within-bound'].includes(mode)){assert.equal(mediaReads,mode==='extra-within-bound'?2:1);assert.deepEqual(await readFile(resolve(dest,'output-0.png')),image);assert.deepEqual(await json(resolve(dest,'asset-0.json')),asset);assert.deepEqual(await json(resolve(dest,'provider-report-0.json')),report);if(mode==='extra-within-bound')assert.deepEqual(await readFile(resolve(dest,'output-1.png')),image);}else assert.equal(mediaReads,0);
 }
});
test('canonical retention refuses malformed or noncompleted asset sets before any GET',async()=>{
 const root=await temporary('retain-bound'),request={projectId:'project'},row={backend:'cuda',expected:'completed',outputCount:1},base={id:'job',projectId:'project',payload:request,status:'completed',backend:'1',workerId:'owned',startedAt:'2026-10-26T21:00:00Z',completedAt:'2026-10-26T21:01:00Z',result:{assets:[{id:'asset'}]}};
 let gets=0;const api={async request(){gets++;throw Error('must refuse before GET');}};
 for(const assets of [Array.from({length:5},(_,i)=>({id:`asset${i}`})),[{id:'same'},{id:'same'}],'not an array',[{}],[{id:''}],[{id:17}],[{id:'../foreign'}]])await assert.rejects(()=>retainOutputs(api,row,request,{...base,result:{assets}},root,root,'1','owned'),/at most four unique valid/);
 for(const status of ['failed','canceled','interrupted','running'])await assert.rejects(()=>retainOutputs(api,row,request,{...base,status},root,root,'1','owned'));
 await assert.rejects(()=>retainOutputs(api,row,request,{...base,result:{...base.result,outputPublication:{state:'pending'}}},root,root,'1','owned'),/publication still pending/);
 assert.deepEqual(await retainOutputs(api,{...row,expected:'canceled',outputCount:0},request,{...base,status:'canceled',result:null},root,root,'1','owned'),[]);
 await assert.rejects(()=>retainOutputs(api,{...row,expected:'canceled',outputCount:0},request,{...base,status:'canceled'},root,root,'1','owned'));
 assert.equal(gets,0);
});
test('whole-device sampled peak requires baseline/active/drain and sampler identity', () => {
  const samples = { version: VERSION, jobId: 'j', samplerSha256: 'e'.repeat(64), device: { uuid: 'actual' }, samples: ['baseline','active','drain'].map((phase,i) => ({ at: new Date(i*1000).toISOString(), phase, usedBytes: [10,30,12][i] })) };
  assert.equal(memorySummary(samples,'j').sampledPeakBytes,30); assert.throws(() => memorySummary({ ...samples, samples: samples.samples.slice(1) },'j'));
  assert.throws(() => memorySummary(samples,'other-job'));
});
test('whole-second job snapshots bracket millisecond device samples without admitting a late baseline', () => {
  const job={ startedAt:'2026-10-25T14:25:02Z', completedAt:'2026-10-25T14:25:05Z' };
  const samples={ samples:[
    { phase:'baseline',at:'2026-10-25T14:25:02.284Z' },
    { phase:'active',at:'2026-10-25T14:25:03.300Z' },
    { phase:'drain',at:'2026-10-25T14:25:05.400Z' },
  ] };
  const attempt={startedAt:'2026-10-25T14:25:02.654Z'};
  assert.equal(samplesBracketJob(samples,job,attempt),true);
  assert.equal(samplesBracketJob(samples,job),false);
  assert.equal(samplesBracketJob(samples,{ ...job,startedAt:'2026-10-25T14:25:02.000Z' },attempt),false);
  assert.equal(samplesBracketJob({ samples:[{ ...samples.samples[0],at:'2026-10-25T14:25:02.800Z' },...samples.samples.slice(1)] },job,attempt),false);
  assert.equal(samplesBracketJob({ samples:[{ ...samples.samples[0],at:'2026-10-25T14:25:03.000Z' },...samples.samples.slice(1)] },job,attempt),false);
  assert.equal(samplesBracketJob({ samples:[samples.samples[0],samples.samples[2]] },job,attempt),false);
  assert.equal(samplesBracketJob({ samples:samples.samples.slice(0,2) },job,attempt),false);
});
test('quality cannot pass from PNG production, a missing anchor or an averaged failure', () => {
  const row = primaryRows(inputs)[0], outputs = [{ sha256: 'f'.repeat(64) }];
  const review = { version: VERSION, rowId: row.id, reviewer: 'human observer', reviewedAt: 'now', outputSha256: outputs.map(o => o.sha256), anchors: row.anchors.map(name => ({ name, verdict: 'PASS', observation: 'visible actual detail' })) };
  assert.equal(reviewVerdict(row,review,outputs),'PASS'); review.anchors[2].verdict = 'FAIL'; assert.equal(reviewVerdict(row,review,outputs),'FAIL'); review.anchors.pop(); assert.throws(() => reviewVerdict(row,review,outputs));
});
test('portable evidence cannot escape the candidate directory', () => {
  for (const name of ['../private','/private','a/../../private','a\\private','a//b']) assert.throws(() => safePath('/candidate',name));
});

const temporary = label => mkdtemp(resolve(tmpdir(),`moss-terminal-${label}-`));
test('R16 recovery adjudication binds original bytes, completed job and only the failed seventh attempt',async t=>{
  const root=await temporary('recovery-adjudication');t.after(()=>rm(root,{recursive:true,force:true}));
  for(let index=0;index<257;index++)await save(resolve(root,`old/${index}.json`),{index});
  const original=await filesUnder(root),sourceSha='a'.repeat(40),inferenceSha='b'.repeat(40),toolingSha='c'.repeat(40),candidateId='actual-r16';
  const manifest={version:1,actualRunId:'40198829729',attempt:1,sourceSha,inferenceSha,files:original};
  await save(resolve(root,'continuation-original-inventory.json'),manifest);
  const inventorySha256=(await fileIdentity(resolve(root,'continuation-original-inventory.json'))).sha256;
  const lock={candidateId,sources:{sceneworks:{head:sourceSha},inference:{head:inferenceSha}}};
  const state={root,lock,config:{repair:{name:'r16-checker-recovery',inventorySha256,tooling:{head:toolingSha}}}};
  const id='advanced-mlx-sft',attempt=resolve(root,'attempts/mlx'),dir=resolve(root,'rows',id),message='per-prompt negatives gained an unsupported broadcast field';
  const start={requestSha256:'d'.repeat(64)},job={id:'job_4ccba4f5dc5d78aa0991ca4df750cb65',payload:{input:1},result:{assets:[{id:'actual-latent'}]}};
  const readback={...job,status:'completed'},rawEvidence=[{file:'old/0.json',...await fileIdentity(resolve(root,'old/0.json'))}];
  const proof={state:'NO_ACTIVE_OWNED_JOB',attemptSha256:digest(start),lockSha256:digest(lock),jobs:[readback],rawEvidence};
  await save(resolve(dir,'failure.json'),{message,jobId:job.id});
  await save(resolve(attempt,`${id}.failure.json`),{message,jobId:job.id,attemptSha256:digest(start)});
  await save(resolve(attempt,`${id}.start.json`),start);
  await save(resolve(dir,'receipt.json'),{recoveredWithoutSubmission:true,jobId:job.id,jobSha256:digest(job)});
  await save(resolve(dir,'terminal-job.json'),job);
  await save(resolve(dir,'recovery-job-readback.json'),readback);
  await save(resolve(attempt,`${id}.reconciliation.json`),proof);
  await save(resolve(attempt,`${id}.resolution.json`),{state:'COORDINATOR_RECONCILED',attemptSha256:digest(start),proofSha256:digest(proof)});
  await save(resolve(root,'recovery-adjudication.json'),{version:VERSION,name:'r16-checker-recovery',rowId:id,candidateId,sourceSha,inferenceSha,toolingSha,inventorySha256,lockSha256:digest(lock),originalRunId:manifest.actualRunId,originalAttempt:manifest.attempt,jobId:job.id,originalAttemptSha256:digest(start),requestSha256:start.requestSha256,reconciliationSha256:digest(proof),rawEvidence});
  await assert.rejects(()=>validateRecoveryAdjudication(state),/original partial evidence\/candidate boundary changed/);
  assert.equal(await validateRecoveryAdjudication(state,inventorySha256),true);
  assert.equal(acceptedRecoveryFailure(true,'mlx',`${id}.failure.json`),true);
  assert.equal(acceptedRecoveryFailure(true,'mlx','later-mlx.failure.json'),false);
  assert.equal(acceptedRecoveryFailure(true,'mlx','guard-training-shift-nonfinite.failure.json'),false);
  assert.equal(acceptedRecoveryFailure(true,'mlx','guard-training-shift-nonfinite.failure.json',true),true);
  assert.equal(acceptedRecoveryFailure(true,'cuda','guard-training-shift-nonfinite.failure.json',true),false);
  assert.equal(acceptedRecoveryFailure(true,'cuda',`${id}.failure.json`),false);
  assert.equal(acceptedRecoveryFailure(false,'mlx',`${id}.failure.json`),false);
  await writeFile(resolve(dir,'recovery-job-readback.json'),JSON.stringify({...readback,id:'different-job'}));
  await assert.rejects(()=>validateRecoveryAdjudication(state,inventorySha256),/authenticated completed-job reconciliation changed/);
  await writeFile(resolve(dir,'recovery-job-readback.json'),JSON.stringify(readback)+'\n');
  await writeFile(resolve(root,'old/0.json'),'changed original evidence');
  await assert.rejects(()=>validateRecoveryAdjudication(state,inventorySha256),/original evidence bytes changed/);
});
function png(pixel, color = 2) {
  const chunk = (type,data) => {
    const body = Buffer.concat([Buffer.from(type),data]); let crc = 0xffffffff;
    for(const byte of body) {crc ^= byte;for(let i=0;i<8;i++)crc=(crc>>>1)^((crc&1)?0xedb88320:0);}
    const prefix=Buffer.alloc(4),suffix=Buffer.alloc(4);prefix.writeUInt32BE(data.length);suffix.writeUInt32BE((crc^0xffffffff)>>>0);
    return Buffer.concat([prefix,body,suffix]);
  };
  const ihdr=Buffer.alloc(13);ihdr.writeUInt32BE(1,0);ihdr.writeUInt32BE(1,4);ihdr[8]=8;ihdr[9]=color;
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',ihdr),chunk('IDAT',deflateSync(Buffer.from([0,...pixel]))),chunk('IEND',Buffer.alloc(0))]);
}
test('source contract Step observations discriminate actual nonterminal denoise from decode/preparation/final step', async () => {
  for(const snapshot of contract.stepSnapshots) assert.equal(isDenoiseStep(snapshot),true);
  for(const snapshot of contract.notSteps) assert.equal(isDenoiseStep(snapshot),false);
  const directory=await temporary('cancel'),calls=[];let reads=0;
  const api={request:async(path,options) => {calls.push({path,method:options?.method??'GET'});if(options?.method==='POST')return{id:'owned-job',status:'canceling'};return reads++===0?contract.stepSnapshots[0]:{id:'owned-job',status:'canceled',result:{}};}};
  const job=await waitJob(api,{cancelAtDenoise:true},'owned-job',directory,4000);
  assert.equal(job.status,'canceled');assert.deepEqual(calls.map(c=>c.path),['/api/v1/jobs/owned-job','/api/v1/jobs/owned-job/cancel','/api/v1/jobs/owned-job']);
  assert.equal((await json(resolve(directory,'cancel-observation.json'))).message,'Image 1/1 — step 1/4.');
});
test('terminal cancellation with synthetic publication pending is accepted only with zero outputs; completed still waits',async()=>{
  const root=await temporary('cancel-publication');
  const request={mode:'edit_image',projectId:'owned-project'},canceled={id:'owned-job',projectId:request.projectId,payload:request,backend:'mlx',status:'canceled',result:{assets:[],assetWrites:[],outputPublication:{state:'pending'}}};
  const api={request:async(path,options)=>{assert.equal(options?.method??'GET','GET');assert.equal(path,'/api/v1/jobs/owned-job');return canceled;}};
  assert.equal((await waitJob(api,{},'owned-job',root,1000)).status,'canceled');
  checkJob({backend:'mlx',expected:'canceled',outputCount:0},request,canceled);
  assert.throws(()=>checkJob({backend:'mlx',expected:'canceled',outputCount:0},request,{...canceled,result:{...canceled.result,assetWrites:[{id:'unpublished-write'}]}}));
  let reads=0;const completed={...canceled,status:'completed',result:{assets:[],outputPublication:{state:'pending'}}};
  const finished=await waitJob({request:async(path,options)=>{assert.equal(options?.method??'GET','GET');reads++;return reads===1?completed:{...completed,result:{assets:[],outputPublication:{state:'ready'}}};}}, {},'owned-job',resolve(root,'completed'),3000);
  assert.equal(reads,2);assert.equal(finished.result.outputPublication.state,'ready');
  await rm(root,{recursive:true,force:true});
});
test('known canceled-job recovery authenticates readback with GET only and rejects another job or output',async()=>{
  const request={mode:'edit_image',projectId:'owned-project'},submission={id:'owned-job',payload:request},observed={id:'owned-job',workerId:'owned-worker'},response={updatedAt:'2026-10-25T16:36:36Z'},binding={endpoint:{workerId:'owned-worker',workerGpuId:'mlx'}};
  const row={backend:'mlx',expected:'canceled',outputCount:0},job={id:'owned-job',projectId:request.projectId,payload:request,backend:'mlx',status:'canceled',workerId:'owned-worker',canceledAt:'2026-10-25T16:36:37Z',result:{assets:[],assetWrites:[],outputPublication:{state:'pending'}}};
  const calls=[],api={request:async(path,options)=>{assert.equal(options?.method??'GET','GET');calls.push(path);return path.endsWith('/metrics')?{jobId:job.id}:job;}};
  assert.equal((await readCanceledJob(api,row,request,submission,observed,response,binding)).job.id,job.id);
  assert.deepEqual(calls,['/api/v1/jobs/owned-job','/api/v1/jobs/owned-job/metrics']);
  for(const bad of [{...job,id:'another-job'},{...job,result:{...job.result,assetWrites:[{id:'unpublished'}]}},{...job,status:'running'}])await assert.rejects(()=>readCanceledJob({request:async()=>bad},row,request,submission,observed,response,binding));
});
test('actual source MCP companion shape matches canonical asset IDs regardless of block/summary order', () => {
  const tensor={dtype:'float32',shape:[1,128,4,4],byteLength:8192};
  const outputs=[{assetId:'sample-zero',sha256:'0'.repeat(64),tensor},{assetId:'sample-one',sha256:'1'.repeat(64),tensor}];
  const result=structuredClone(contract.mcpResult);
  // ContentBlock::json is an actual text JSON block in the product wire protocol.
  result.content.push({type:'text',text:JSON.stringify(result.structuredContent)});delete result.structuredContent;
  const assets=mcpResultAssets(result,'owned-job',outputs,'http://127.0.0.1:0');
  assert.equal(assets[0].assetId,'sample-zero');assert.match(assets[0].mediaUrl,/zero\.npy/);assert.match(assets[1].companionUrl,/one\.sceneworks-workflow\.json/);
  const missing=structuredClone(result);missing.content.splice(0,1);assert.throws(()=>mcpResultAssets(missing,'owned-job',outputs,'http://127.0.0.1:0'));
  assert.throws(()=>mcpResultAssets(result,'different-job',outputs,'http://127.0.0.1:0'));
  const foreign=structuredClone(result);const summary=JSON.parse(foreign.content.at(-1).text);summary.assets[0].id='foreign';foreign.content.at(-1).text=JSON.stringify(summary);assert.throws(()=>mcpResultAssets(foreign,'owned-job',outputs,'http://127.0.0.1:0'));
  const wrongShape=structuredClone(result),wrongSummary=JSON.parse(wrongShape.content.at(-1).text);wrongSummary.assets[0].tensor.shape=[1,128,2,8];wrongShape.content.at(-1).text=JSON.stringify(wrongSummary);assert.throws(()=>mcpResultAssets(wrongShape,'owned-job',outputs,'http://127.0.0.1:0'));
});
test('portable source export keeps sanitized display provenance while the new execution must scrub it', () => {
  const original=JSON.stringify(contract.exportedWorkflow);
  validateExportedWorkflow(contract.exportedWorkflow,contract.originalReport,0);
  assert.equal(JSON.stringify(contract.exportedWorkflow),original);
  assert.throws(()=>assertRequestBinding({},contract.exportedWorkflow));
  const forged=structuredClone(contract.exportedWorkflow);forged.advanced.mossImageEditExecutionReport.adapter_path='/private/leak';assert.throws(()=>validateExportedWorkflow(forged,contract.originalReport,0));
  assert.throws(()=>validateExportedWorkflow(contract.exportedWorkflow,contract.originalReport,1));
  assertRequestBinding({mode:'edit_image'},{mode:'edit_image',advanced:{mossImage:{acceleration:'off'}}});
});
test('retained H47 Euler sigma serialization compares exact F32 only and preserves every other provenance field', () => {
  // Actual advanced-reentry-cuda job_e32983c669971ffd54d0187439785fcb, run40415337753/1.
  const report=structuredClone(contract.originalReport),workflow=structuredClone(contract.exportedWorkflow);
  report.base.sampling={kind:'euler',sigmas:[1,0.90976083278656,0.7173325419425964,0.023381737992167473,0]};report.base.resolved={guidance:4};
  const actual=workflow.advanced.mossImageEditExecutionReport;actual.base.sampling={kind:'euler',sigmas:[1,0.9097608327865601,0.7173325419425964,0.023381737992167473,0]};actual.base.resolved={guidance:4};
  workflow.advanced.mossImageExecutionReport=structuredClone(actual.base);
  assert.notEqual(report.base.sampling.sigmas[1],actual.base.sampling.sigmas[1]);assert.equal(Math.fround(report.base.sampling.sigmas[1]),Math.fround(actual.base.sampling.sigmas[1]));
  const before=JSON.stringify({report,workflow});validateExportedWorkflow(workflow,report,0);assert.equal(JSON.stringify({report,workflow}),before);
  const next=Buffer.alloc(4);next.writeFloatLE(Math.fround(actual.base.sampling.sigmas[1]));next.writeUInt32LE(next.readUInt32LE()+1);const nextF32=next.readFloatLE();
  const reject=change=>{const copy=structuredClone(workflow);change(copy.advanced);assert.throws(()=>validateExportedWorkflow(copy,report,0));};
  reject(a=>{a.mossImageEditExecutionReport.base.sampling.sigmas[1]=nextF32;});reject(a=>{a.mossImageExecutionReport.sampling.sigmas[1]=nextF32;});
  reject(a=>{a.mossImageEditExecutionReport.base.resolved.guidance=4.000000029802322;});assert.equal(Math.fround(4.000000029802322),Math.fround(4));
  reject(a=>{a.mossImageEditExecutionReport.extra=true;});reject(a=>{a.mossImageEditExecutionReport.base.sampling.extra=true;});
  reject(a=>{delete a.mossImageEditExecutionReport.base.sampling.kind;});reject(a=>{a.mossImageEditExecutionReport.base.sampling.sigmas.pop();});reject(a=>{a.mossImageEditExecutionReport.base.sampling.sigmas.reverse();});
  reject(a=>{a.mossImageEditExecutionReport.adapter_revision='0'.repeat(40);});reject(a=>{a.mossImageEditExecutionReport.base.samples[0].noise_sha256='0'.repeat(64);});reject(a=>{a.mossImageSampleIndex=1;});
  for(const bad of [NaN,Infinity,-Infinity,1e100,'0.9097608327865601',null]){reject(a=>{a.mossImageEditExecutionReport.base.sampling.sigmas[1]=bad;});const invalid=structuredClone(report);invalid.base.sampling.sigmas[1]=bad;assert.throws(()=>validateExportedWorkflow(workflow,invalid,0));}
  reject(a=>{a.mossImageEditExecutionReport.base.sampling.sigmas={0:1};});
  const dmdReport=structuredClone(report),dmdWorkflow=structuredClone(workflow);dmdReport.base.sampling.kind='dmd';dmdWorkflow.advanced.mossImageEditExecutionReport.base.sampling.kind='dmd';assert.throws(()=>validateExportedWorkflow(dmdWorkflow,dmdReport,0));
});
test('actual queue GPU-id/candle label is separate from the admitted native CUDA report device', () => {
  const row={backend:'cuda',expected:'completed',outputCount:1},request={projectId:'p'},job={id:'j',projectId:'p',payload:request,status:'completed',workerId:'native-GPU1',startedAt:'start',completedAt:'end',result:{assets:[{id:'a'}]}};
  checkJob(row,request,{...job,backend:'1'},'1');checkJob(row,request,{...job,backend:'candle'},'1');
  for(const backend of ['0','cpu','cuda','mlx','other'])assert.throws(()=>checkJob(row,request,{...job,backend},'1'));
  assert.throws(()=>checkJob(row,request,{...job,backend:'1'},'0'));
});
test('advanced cancel and actual retry use identical caller tensors/noise/masks/output on both backends', () => {
  const p=plan();for(const backend of ['mlx','cuda']) {
    const cancel=p.rows.find(r=>r.id===`advanced-cancel-${backend}`),retry=p.rows.find(r=>r.id===`advanced-reentry-${backend}`);
    assert.equal(cancel.cancelAtDenoise,true);assert.equal(cancel.expected,'canceled');assert.equal(cancel.outputCount,0);
    assert.equal(retry.action,'retry');assert.equal(retry.replayOf,cancel.id);assert.equal(retry.outputCount,4);assert.equal(retry.expected,'completed');
    assert.deepEqual(retry.request,cancel.request);assert.equal(retry.request.seed,null);assert.equal(retry.request.advanced.mossImageAdvanced.output,'latents');assert.deepEqual(retry.refs,['B','L']);
  }
});
test('unexpected enqueue and ambiguous network failure retain evidence and block subsequent dispatch until explicit reconciliation', async () => {
  const root=await temporary('guard'),path=resolve(root,'guard');await mkdir(path);
  const ticket=await reserveAttempt(root,'cuda','guard-test',{lockSha256:'a'.repeat(64),requestSha256:'b'.repeat(64)});
  let failure;try{await submitGuard({request:async()=>({id:'unexpected-native-job',status:'queued'})},{expectedStatus:400,errorIncludes:['reject']},{},path);}catch(error){failure=error;}
  assert.equal(failure.jobId,'unexpected-native-job');assert.equal((await json(resolve(path,'unexpected-enqueue.json'))).id,'unexpected-native-job');
  await failAttempt(ticket,failure,failure.jobId);await assert.rejects(()=>assertDispatchClear(root,'cuda'),/unresolved/);
  const raw=resolve(root,'authoritative.json');await save(raw,{id:'unexpected-native-job',status:'canceled'});
  const proof={version:VERSION,lockSha256:'a'.repeat(64),attemptSha256:digest(ticket.start),backend:'cuda',coordinator:'root',observedAt:new Date().toISOString(),state:'NO_ACTIVE_OWNED_JOB',jobs:[{id:'unexpected-native-job',status:'canceled',result:{}}],rawEvidence:[{file:'authoritative.json',...(await fileIdentity(raw))}]};
  const active=structuredClone(proof);active.jobs[0].status='running';await assert.rejects(()=>reconcileAttempt(root,'cuda','guard-test',active,'a'.repeat(64)));
  await reconcileAttempt(root,'cuda','guard-test',proof,'a'.repeat(64));await assertDispatchClear(root,'cuda');
  assert.equal((await json(resolve(ticket.directory,'guard-test.failure.json'))).jobId,'unexpected-native-job');
  const other=await reserveAttempt(root,'cuda','guard-network',{requestSha256:'c'.repeat(64)}),networkPath=resolve(root,'network');await mkdir(networkPath);
  const original=new Error('connection lost after submit');let observed;try{await submitGuard({request:async()=>{throw original;}},{expectedStatus:400,errorIncludes:['reject']},{},networkPath);}catch(error){observed=error;}
  assert.equal(observed,original);await failAttempt(other,observed);await assert.rejects(()=>assertDispatchClear(root,'cuda'),/unresolved/);
  const unresolved=await json(resolve(other.directory,'guard-network.failure.json'));assert.equal(unresolved.jobId,null);assert.equal(unresolved.message,'connection lost after submit');
});
test('a crash reservation and concurrent commands fail closed; an expected refusal resolves only that exact attempt', async () => {
  const root=await temporary('mutex'),ticket=await reserveAttempt(root,'mlx','guard-shape',{requestSha256:'a'.repeat(64)});
  await assert.rejects(()=>reserveAttempt(root,'mlx','another',{}));await assert.rejects(()=>assertDispatchClear(root,'mlx'),/unresolved/);
  const path=resolve(root,'guard');await mkdir(path);const error=new Error('product API returned HTTP 400');error.status=400;error.responseBody='invalid tensor shape';
  await submitGuard({request:async()=>{throw error;}},{expectedStatus:400,errorIncludes:['tensor shape']},{},path);
  await resolveAttempt(ticket,'EXPECTED_REFUSAL');await assertDispatchClear(root,'mlx');assert.equal((await json(resolve(path,'response.json'))).status,400);
});
test('UI intent reserves dispatch before human submission and adoption resumes only the identical frozen request',async()=>{
  const root=await temporary('ui-intent'),fields={transport:'ui',requestSha256:'a'.repeat(64),lockSha256:'b'.repeat(64)};
  const intent=await reserveAttempt(root,'mlx','quality-mlx-base-single',fields);
  await assert.rejects(()=>reserveAttempt(root,'mlx','guard-other',{}));await assert.rejects(()=>assertDispatchClear(root,'mlx'),{code:'MOSS_AWAITING_RESOLUTION'});
  await assert.rejects(()=>resumeUiAttempt(root,'mlx',intent.id,{...fields,requestSha256:'c'.repeat(64)}));
  const adopted=await resumeUiAttempt(root,'mlx',intent.id,fields);assert.deepEqual(adopted.start,intent.start);
  await resolveAttempt(adopted,'TERMINAL_VERIFIED',{jobId:'actual-ui-job'});await assertDispatchClear(root,'mlx');
  await assert.rejects(()=>resumeUiAttempt(root,'mlx',intent.id,fields));
});
test('source-owned MAE records bind exact ordered output bytes/pixels/jobs/requests and reject copied metrics', async () => {
  const root=await temporary('metric'),bytes=[png([0,0,0]),png([255,0,0])],p=plan(),row=p.rows.find(r=>r.id==='cachefalse-mlx');
  p.cacheComparisonPolicy={metric:'rgb8-mae/v1',tolerance:0.34,implementationSha256:'a'.repeat(64)};
  const lock={files:{harnessLibrary:{sha256:'a'.repeat(64)}}},state={root,plan:p,lock};
  for(const [i,id] of [row.compareTo,row.id].entries()) {
    const directory=resolve(root,'rows',id);await mkdir(directory,{recursive:true});const request=structuredClone(p.rows.find(r=>r.id===id).request);
    await save(resolve(directory,'request.json'),request);await writeFile(resolve(directory,'output.png'),bytes[i]);
    await save(resolve(directory,'receipt.json'),{lockSha256:digest(lock),execution:'VERIFIED',jobId:`actual-${i}`,requestSha256:digest(request),outputs:[{file:'output.png',sha256:sha(bytes[i])}]});
  }
  const measured=await measuredCacheComparison(state,row);assert.equal(measured.value,1/3);assert.deepEqual(measured.pair.map(item=>item.jobId),['actual-0','actual-1']);assert.deepEqual(measured.pair.map(item=>item.outputSha256),bytes.map(sha));
  const copied=structuredClone(measured);copied.pair[0].outputSha256='x';assert.notEqual(digest(copied),digest(await measuredCacheComparison(state,row)));
  assert.throws(()=>cacheComparison(p.cacheComparisonPolicy,digest(lock),copied.pair,bytes));
  const wrong={...state,plan:{...p,cacheComparisonPolicy:{...p.cacheComparisonPolicy,implementationSha256:'b'.repeat(64)}}};await assert.rejects(()=>measuredCacheComparison(wrong,row),/implementation/);
  await writeFile(resolve(root,'rows',row.id,'output.png'),png([0,0,0]));await assert.rejects(()=>measuredCacheComparison(state,row),/output changed/);
});

// Tiny inert Git fixture data, entirely inside TMPDIR. No git commit, checkout,
// clone, production repository mutation or remote operation is executed.
async function gitFixture(directory,manifest,extraFiles = {}) {
  await mkdir(resolve(directory,'.git','objects'),{recursive:true});await mkdir(resolve(directory,'.git','refs','heads'),{recursive:true});await mkdir(resolve(directory,'.git','refs','remotes','origin'),{recursive:true});
  const object=async(type,bytes)=>{const body=Buffer.concat([Buffer.from(`${type} ${bytes.length}\0`),bytes]),hash=createHash('sha1').update(body).digest('hex');await mkdir(resolve(directory,'.git','objects',hash.slice(0,2)),{recursive:true});await writeFile(resolve(directory,'.git','objects',hash.slice(0,2),hash.slice(2)),deflateSync(body));return hash;};
  const nodes={};
  for(const [file,body] of Object.entries({'Cargo.toml':manifest,...extraFiles})) {
    const parts=file.split('/');let node=nodes;for(const part of parts.slice(0,-1))node=node[part]??={};node[parts.at(-1)]=body;
    await mkdir(resolve(directory,...parts.slice(0,-1)),{recursive:true});await writeFile(resolve(directory,file),body);
  }
  const treeFor=async node=>{const entries=[];for(const [name,value] of Object.entries(node).sort(([a,av],[b,bv])=>(a+(typeof av==='object'?'/':'')).localeCompare(b+(typeof bv==='object'?'/':''),'en'))) {
    const dir=typeof value==='object',id=dir?await treeFor(value):await object('blob',Buffer.from(value));entries.push(Buffer.from(`${dir?'40000':'100644'} ${name}\0`),Buffer.from(id,'hex'));
  }return object('tree',Buffer.concat(entries));};const tree=await treeFor(nodes);
  const head=await object('commit',Buffer.from(`tree ${tree}\nauthor Fixture <fixture@example.invalid> 1 +0000\ncommitter Fixture <fixture@example.invalid> 1 +0000\n\nInert test fixture\n`));
  await writeFile(resolve(directory,'.git','HEAD'),'ref: refs/heads/main\n');await writeFile(resolve(directory,'.git','config'),'[core]\nrepositoryformatversion = 0\nbare = false\n');
  await writeFile(resolve(directory,'.git','refs','heads','main'),`${head}\n`);await writeFile(resolve(directory,'.git','refs','remotes','origin','main'),`${head}\n`);await writeFile(resolve(directory,'Cargo.toml'),manifest);
  execFileSync('git',['-C',directory,'read-tree','HEAD'],{env:{...process.env,GIT_CONFIG_GLOBAL:'/dev/null',GIT_CONFIG_NOSYSTEM:'1'}});return head;
}
test('Actions accepts a clean stationary SceneWorks feature candidate only with exact pins and inference on main', async () => {
  const directory=await temporary('actions-source-barrier'),inference=resolve(directory,'inference'),product=resolve(directory,'product');
  const inferenceHead=await gitFixture(inference,'[package]\nname="fixture-inference"\n');
  const pinnedProduct=async revision=>{
    const manifest=`[dependencies]\ninference={git="https://github.com/SceneWorks/inference",rev="${revision}"}\n`;
    return gitFixture(product,manifest,{'crates/sceneworks-worker/Cargo.toml':manifest,'crates/sceneworks-memory-adapter/Cargo.toml':manifest,'Cargo.lock':`source = "git+https://github.com/SceneWorks/inference?rev=${revision}#${revision}"\n`});
  };
  const productMain=await pinnedProduct(inferenceHead);
  const manifest=await readFile(resolve(product,'Cargo.toml'),'utf8');
  const sourceSha=await gitFixture(product,manifest,{'crates/sceneworks-worker/Cargo.toml':manifest,'crates/sceneworks-memory-adapter/Cargo.toml':manifest,'Cargo.lock':await readFile(resolve(product,'Cargo.lock'),'utf8'),'terminal-candidate.txt':'stationary feature candidate\n'});
  await writeFile(resolve(product,'.git/refs/remotes/origin/main'),`${productMain}\n`);
  assert.throws(()=>execFileSync('git',['-C',product,'merge-base','--is-ancestor',sourceSha,'origin/main']));
  const config={sourceRoot:product,inferenceRoot:inference,sourceSha,inferenceSha:inferenceHead};
  assert.deepEqual(await sourceIdentity(config),{sceneworks:treeIdentity(product),inference:treeIdentity(inference)});
  await assert.rejects(()=>sourceIdentity({...config,sourceSha:productMain}),/exact paired dispatch sources/);
  await writeFile(resolve(product,'terminal-candidate.txt'),'changed after freeze\n');
  await assert.rejects(()=>sourceIdentity(config),/must be clean/);
  await writeFile(resolve(product,'terminal-candidate.txt'),'stationary feature candidate\n');
  const unmergedInference=await gitFixture(inference,'[package]\nname="unmerged-fixture-inference"\n');
  await writeFile(resolve(inference,'.git/refs/remotes/origin/main'),`${inferenceHead}\n`);
  await assert.rejects(()=>sourceIdentity({...config,inferenceSha:unmergedInference}),/manifest pin/);
  await unlink(resolve(product,'terminal-candidate.txt'));
  const unmergedProduct=await pinnedProduct(unmergedInference);
  await assert.rejects(()=>sourceIdentity({...config,sourceSha:unmergedProduct,inferenceSha:unmergedInference}),error=>error.code===1&&error.cmd.includes('merge-base --is-ancestor'));
});
test('bounded API-free prepare to actual clean-source freeze writes a valid lock and refuses reuse or tampered metric source', async () => {
  const directory=await temporary('prepare-freeze'),source=resolve(directory,'original.png');await writeFile(source,png([10,20,30]));
  const manifest={...structuredClone(inputs),assets:['B','L','P','G'].map(id=>({id,path:source,sha256:sha(png([10,20,30])),receipt_sha256:'e'.repeat(64)}))};
  const inputPath=resolve(directory,'inputs.json');await save(inputPath,manifest);const output=resolve(directory,'candidate');await prepare(inputPath,output);
  const p=await json(resolve(output,'cases.json'));assert.equal(p.renderExecutions,50);assert.equal(p.rows.filter(r=>r.category==='quality').length,8);assert.equal((await json(resolve(output,'input-identities.json'))).B.pixelsSha256,sha(Buffer.from([10,20,30])));
  const library=fileURLToPath(new URL('./lib/moss-edit-terminal.mjs',import.meta.url));p.cacheComparisonPolicy={metric:'rgb8-mae/v1',tolerance:0.01,implementationSha256:(await fileIdentity(library)).sha256};await writeFile(resolve(output,'cases.json'),JSON.stringify(p));
  const inference=resolve(directory,'inference'),product=resolve(directory,'product');const inferenceHead=await gitFixture(inference,'[package]\nname="fixture-inference"\n');
  const manifestPin=`[dependencies]\ninference={git="https://github.com/SceneWorks/inference",rev="${inferenceHead}"}\n`,lockPin=`source = "git+https://github.com/SceneWorks/inference?rev=${inferenceHead}#${inferenceHead}"\n`;
  await gitFixture(product,manifestPin,{'crates/sceneworks-worker/Cargo.toml':manifestPin,'crates/sceneworks-memory-adapter/Cargo.toml':manifestPin,'Cargo.lock':lockPin,'scripts/moss-edit-terminal.mjs':await readFile(fileURLToPath(new URL('./moss-edit-terminal.mjs',import.meta.url)),'utf8'),'scripts/lib/moss-edit-terminal.mjs':await readFile(library,'utf8')});
  const launcher=await json(resolve(output,'launch.example.json')),artifact=resolve(directory,'fixture-inventory.json');await save(artifact,{fixture:true});
  Object.assign(launcher,{sources:{sceneworks:product,inference},approvedBy:'PURE_FIXTURE_ONLY',approvalEvidenceSha256:'f'.repeat(64),phase18ReviewSha256:'f'.repeat(64),inferencePreMergeReviewSha256:'f'.repeat(64)});
  for(const key of ['apiBinary','workerBinary','webPackage','packages','toolchain','numericalCarryForward'])launcher.files[key]=artifact;
  const sources={sceneworks:treeIdentity(product),inference:treeIdentity(inference)},files=Object.fromEntries(await Promise.all(BACKEND_FILES.map(async name=>[name,await fileIdentity(artifact)])));
  for(const backend of ['mlx','cuda']) {launcher.artifactInventories[backend]=resolve(directory,`${backend}-inventory.json`);await save(launcher.artifactInventories[backend],{backend,sources,files});}
  for(const name of p.requiredGuardCases)launcher.guardRequests[name]=LATENT_UPSCALE_GUARDS.includes(name)?{...latentUpscaleGuard(name),noProviderLoadProbe:'inert-source-fixture'}:{request:{},expectedStatus:400,errorIncludes:['fixture rejection'],noProviderLoadProbe:'inert-source-fixture'};
  launcher.guardRequests['corrupt-adapter']={request:{},expectedStatus:201,expectedTerminal:'failed',errorIncludes:['MOSS edit frozen component identity mismatch'],noProviderLoadProbe:'inert-source-fixture'};
  const configPath=resolve(output,'launch.json'),legacy={...launcher,finalSourceReviewSha256:launcher.inferencePreMergeReviewSha256};delete legacy.inferencePreMergeReviewSha256;
  const legacyPath=resolve(output,'legacy-launch.json');await save(legacyPath,legacy);await assert.rejects(()=>freeze(legacyPath),/inference pre-merge source review identities/);
  await save(configPath,launcher);await freeze(configPath);
  const lock=await json(resolve(output,'protocol.lock.json'));assert.equal(lock.sources.inference.head,inferenceHead);assert.equal(lock.counts.renders,50);assert.equal(lock.counts.guards,58);assert.equal(lock.counts.guardCasesPerBackend,29);assert.equal(lock.files.harnessLibrary.sha256,p.cacheComparisonPolicy.implementationSha256);
  assert.equal(lock.approvals.inferencePreMergeReview,launcher.inferencePreMergeReviewSha256);assert.equal('finalSourceReview' in lock.approvals,false);
  assert.equal(JSON.stringify(lock).includes(directory),false);assert.equal('launcherSha256' in lock,false);await frozen(configPath);
  const relocated=resolve(directory,'second-host');await mkdir(relocated);await cp(product,resolve(relocated,'product'),{recursive:true});await cp(inference,resolve(relocated,'inference'),{recursive:true});
  const second=resolve(relocated,'candidate');await mkdir(second);for(const file of ['protocol.lock.json','cases.json','inputs.original.json','input-identities.json'])await cp(resolve(output,file),resolve(second,file));
  await cp(resolve(output,'inputs'),resolve(second,'inputs'),{recursive:true});
  await cp(artifact,resolve(relocated,'artifact.json'));
  const local=structuredClone(launcher);local.localBackend='cuda';local.sources={sceneworks:resolve(relocated,'product'),inference:resolve(relocated,'inference')};local.evidenceRoot=second;local.plan=resolve(second,'cases.json');
  for(const name of [...BACKEND_FILES,'numericalCarryForward'])local.files[name]=resolve(relocated,'artifact.json');
  const localPath=resolve(second,'launch.json');await save(localPath,local);await bind(localPath);const relocatedState=await frozen(localPath);
  assert.equal(digest(relocatedState.lock),digest(lock));assert.equal(relocatedState.binding.backend,'cuda');assert.notEqual(relocatedState.binding.paths.sources.sceneworks,launcher.sources.sceneworks);
  assert.equal(await readFile(resolve(second,'protocol.lock.json'),'utf8'),await readFile(resolve(output,'protocol.lock.json'),'utf8'));await assert.rejects(()=>bind(localPath),/EEXIST/);
  // Run the actual bind/frozen entrypoints from a separate clean tooling checkout.
  // Product source, lock, plan and frozen protocol bytes remain those above.
  const toolingRoot=await realpath(await (async()=>{const path=resolve(directory,'tooling');await cp(fileURLToPath(new URL('./',import.meta.url)),resolve(path,'scripts'),{recursive:true});return path;})());
  const git=(...args)=>execFileSync('git',['-C',toolingRoot,...args],{encoding:'utf8',stdio:['ignore','pipe','pipe']}).trim();
  git('init');git('add','.');git('-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-m','inert tooling');
  const tooling=await executionTooling(toolingRoot,git('rev-parse','HEAD')),protocol=await import(pathToFileURL(resolve(toolingRoot,'scripts/moss-edit-terminal.mjs')).href);
  const reused=structuredClone(local),reusedRoot=resolve(directory,'reused-cuda');await cp(second,reusedRoot,{recursive:true});await rm(resolve(reusedRoot,'bindings'),{recursive:true});
  reused.executionTooling=tooling;reused.evidenceRoot=reusedRoot;reused.plan=resolve(reusedRoot,'cases.json');reused.files.harness=resolve(reused.sources.sceneworks,'scripts/moss-edit-terminal.mjs');reused.files.harnessLibrary=resolve(reused.sources.sceneworks,'scripts/lib/moss-edit-terminal.mjs');
  const reusedPath=resolve(reusedRoot,'launch.json');await writeFile(reusedPath,JSON.stringify(reused));await protocol.bind(reusedPath);const reusedState=await protocol.frozen(reusedPath);
  assert.equal(digest(reusedState.lock),digest(lock));assert.equal(reusedState.binding.backend,'cuda');assert.equal(reusedState.config.executionTooling.head,tooling.head);assert.notEqual(tooling.head,lock.sources.sceneworks.head);
  assert.equal(await readFile(resolve(reusedRoot,'protocol.lock.json'),'utf8'),await readFile(resolve(output,'protocol.lock.json'),'utf8'));
  const detached=structuredClone(local),detachedPath=resolve(second,'detached-launch.json'),detachedArtifact=resolve(relocated,'detached-artifact.json');await cp(artifact,detachedArtifact);
  for(const name of [...BACKEND_FILES,'numericalCarryForward'])detached.files[name]=detachedArtifact;await save(detachedPath,detached);
  await assert.rejects(()=>frozen(detachedPath),/binding changed/);assert.equal(digest((await frozen(detachedPath,{retained:true})).lock),digest(lock));
  await writeFile(detachedArtifact,'changed artifact');await assert.rejects(()=>frozen(detachedPath,{retained:true}),/changed/);
  local.backends.cuda.projectId='other-project';await writeFile(localPath,JSON.stringify(local));await assert.rejects(()=>frozen(localPath),/binding changed/);local.backends.cuda.projectId=launcher.backends.cuda.projectId;
  local.guardRequests['edit-dmd'].expectedStatus=409;await writeFile(localPath,JSON.stringify(local));await assert.rejects(()=>frozen(localPath),/guard requests changed/);
  await assert.rejects(()=>prepare(inputPath,output),/EEXIST/);await assert.rejects(()=>freeze(configPath),/EEXIST/);
  p.cacheComparisonPolicy.implementationSha256='0'.repeat(64);await writeFile(resolve(output,'cases.json'),JSON.stringify(p));await assert.rejects(()=>freeze(configPath),/actual source-owned implementation/);
});

test('v3 seeded text rows cover row-two image and latent UI replay on both native backends', () => {
  const p=plan();for(const backend of ['mlx','cuda']) for(const output of ['images','latents']) {
    const row=p.rows.find(r=>r.id===`seeded-${backend}-${output}`), replay=p.rows.find(r=>r.id===`${row.id}-replay`);
    assert.equal(row.request.prompt,row.request.prompts[0]);assert.equal(row.request.count,2);assert.equal(row.outputCount,4);assert.equal(row.request.seed,173);
    assert.equal(row.request.advanced.mossImageAdvanced.noise,undefined);assert.equal(row.request.advanced.mossImageAdvanced.positive,undefined);
    assert.deepEqual(replay.request,row.request);assert.equal(replay.action,'seeded-recipe');assert.equal(replay.transport,'ui-full');assert.equal(replay.originSampleIndex,2);
    const request={...row.request,referenceAssetIds:['b','l']}, origin={jobId:'job',outputs:[{assetId:'a0'},{assetId:'a1'},{assetId:'a2'},{assetId:'a3'}]};
    const asset={id:'a2',recipe:{prompt:request.prompts[1],seed:175,rawAdapterSettings:{promptBatch:request.prompts,referenceAssetIds:request.referenceAssetIds,mossImageAdvanced:request.advanced.mossImageAdvanced,mossImageSampleIndex:2}}};
    const history={originRowId:row.id,originJobId:'job',originSampleIndex:2,selectedAssetId:'a2',selectedRecipe:asset.recipe,hydrationObserved:true,keepSeed:true};
    validateSeededRecipeSelection(replay,request,history,origin,asset);
    for(const mutate of [h=>h.originSampleIndex=0,h=>h.selectedAssetId='a0',h=>h.keepSeed=false,h=>h.selectedRecipe.prompt=request.prompts[0]]) {const invalid=structuredClone(history);mutate(invalid);assert.throws(()=>validateSeededRecipeSelection(replay,request,invalid,origin,asset));}
    assert.throws(()=>validateSeededRecipeSelection(replay,{...request,prompt:request.prompts[1]},history,origin,asset));
    assert.throws(()=>validateSeededRecipeSelection(replay,{...request,seed:175},history,origin,asset));
  }
});
test('v3 exact latent-inline-upscale refusals distinguish both supported engines', () => {
  for(const name of LATENT_UPSCALE_GUARDS) {const guard=latentUpscaleGuard(name);assert.equal(guard.request.upscale.enabled,true);assert.equal(guard.request.advanced.mossImageAdvanced.output,'latents');assert.equal(guard.expectedStatus,400);assert.ok(guard.errorIncludes.includes('disable upscale or select images output'));}
  assert.equal(new Set(LATENT_UPSCALE_GUARDS.map(name=>latentUpscaleGuard(name).request.upscale.engine)).size,2);
});

test('v4 path-independent backend inventories reject source, artifact and binding substitution', () => {
  const sources={sceneworks:{head:'1'.repeat(40),tree:'2'.repeat(40)},inference:{head:'3'.repeat(40),tree:'4'.repeat(40)}};
  const files=Object.fromEntries(BACKEND_FILES.map(name=>[name,{byteLength:12,sha256:sha(name)}]));
  const inventory={backend:'cuda',sources,files};assert.deepEqual(validateArtifactInventory(inventory,'cuda',sources),inventory);
  for(const mutate of [x=>x.backend='mlx',x=>x.sources.inference.head='5'.repeat(40),x=>delete x.files.workerBinary,x=>x.files.apiBinary.path='/not-an-identity',x=>x.files.webPackage.sha256='invalid']) {
    const altered=structuredClone(inventory);mutate(altered);assert.throws(()=>validateArtifactInventory(altered,'cuda',sources));
  }
  const lock={version:VERSION,sources,files:{harness:{byteLength:1,sha256:sha('harness')}},artifacts:{cuda:inventory,mlx:{backend:'mlx',sources,files:{...files,apiBinary:{byteLength:99,sha256:sha('other-native-binary')}}}}};
  const config={localBackend:'cuda',sources:{sceneworks:'/local/sw',inference:'/local/inference'},files:{},plan:'/local/cases.json',evidenceRoot:'/local/evidence',backends:{cuda:{apiUrl:'http://127.0.0.1:1234',projectId:'actual-project',workerId:'actual-worker',workerGpuId:'1',tokenEnv:'INERT_FIXTURE_TOKEN_ENV'}}};
  const binding=executionBinding(config,lock,{...lock.files,...files});validateBinding(binding,lock,'cuda');
  for(const mutate of [x=>x.lockSha256=sha('other-candidate'),x=>x.backend='mlx',x=>x.sources.sceneworks.tree='8'.repeat(40),x=>x.files.apiBinary=lock.artifacts.mlx.files.apiBinary,x=>x.endpoint.workerGpuId='0',x=>x.endpoint.apiUrl='http://token@example.invalid']) {
    const altered=structuredClone(binding);mutate(altered);assert.throws(()=>validateBinding(altered,lock,'cuda'));
  }
});
test('v4 checks every manifest inference revision and the resolved Cargo.lock commit rather than a matching comment', () => {
  const revision='a'.repeat(40),other='b'.repeat(40),manifest=`inference={git="https://github.com/SceneWorks/inference",rev="${revision}"}`,cargoLock=`source = "git+https://github.com/SceneWorks/inference?rev=${revision}#${revision}"`;
  validateInferencePins([manifest,manifest,manifest],cargoLock,revision);
  assert.throws(()=>validateInferencePins([manifest,manifest.replace(revision,other),manifest],cargoLock,revision));
  assert.throws(()=>validateInferencePins([`# ${manifest}\n${manifest.replace(revision,other)}`,manifest,manifest],cargoLock,revision));
  assert.throws(()=>validateInferencePins([`anything=1 # ${manifest}`,manifest,manifest],cargoLock,revision));
  assert.throws(()=>validateInferencePins([manifest,manifest,manifest],cargoLock.replace(`#${revision}`,`#${other}`),revision));
});

function editorCase() {
  const row=primaryRows(inputs).find(row=>row.id==='quality-cuda-sft-single');
  const expected={...structuredClone(row.request),projectId:'target-project',referenceAssetIds:['uploaded-original-B']};
  const uploadBytes=png([10,20,30,255],6),original=pngRgbIdentity(png([10,20,30]));
  const origin={jobId:'actual-mlx-origin',outputs:[{assetId:'actual-origin-output'}]};
  // Actual retained H27 SFT recipe schedule; H32 Windows GET differs at seven decimal spellings.
  const sigmas=[1,0.9929999709129333,0.9850000143051147,0.9779999852180481,0.968999981880188,0.9599999785423279,0.9509999752044678,0.9409999847412109,0.9300000071525574,0.9179999828338623,0.9049999713897705,0.8920000195503235,0.8769999742507935,0.8610000014305115,0.843999981880188,0.824999988079071,0.8040000200271606,0.781000018119812,0.7559999823570251,0.7279999852180481,0.6959999799728394,0.6599999666213989,0.6189999580383301,0.5719999670982361,0.5169999599456787,0.453000009059906,0.37700000405311584,0.2840000092983246,0.17000001668930054,0.024000002071261406,0];
  const recipe={prompt:'original request',seed:173,rawAdapterSettings:{steps:30,mossImageEditExecutionReport:{base:{sampling:{kind:'euler',sigmas}}}}};
  const originJob={id:origin.jobId,projectId:'origin-project',status:'completed',payload:{projectId:'origin-project',prompt:'original request',advanced:{steps:30}},result:{assets:[{id:'actual-origin-output',recipe:structuredClone(recipe)}]}};
  const originAsset={id:'actual-origin-output',projectId:'origin-project',lineage:{jobId:origin.jobId},recipe};
  const at=n=>new Date(Date.now()-10000+n*1000).toISOString();
  const asset={id:'actual-ui-scratch',projectId:'target-project',type:'image',createdAt:at(1),file:{path:'assets/scratch.png'}};
  const lockSha256=sha('inert-candidate'),bindingSha256=sha('inert-binding');
  const capture={version:VERSION,rowId:row.id,lockSha256,bindingSha256,observer:'INERT_FIXTURE_NOT_RUNTIME',capturedAt:at(4),
    sourceUpload:{inputId:'B',file:'observed-upload.png',sha256:sha(uploadBytes),observedAt:at(1),byteObserver:'authenticated-product-asset-readback',network:{method:'POST',path:'/api/v1/projects/target-project/assets',request:{observer:'playwright-request-metadata-and-product-response',fileSha256:null,postDataState:'file-bytes-unavailable-in-observer',contentType:'multipart/form-data'},response:{id:asset.id}}},
    network:{method:'POST',path:'/api/v1/image/jobs',observedAt:at(2),request:{...structuredClone(expected),referenceAssetIds:[asset.id]},response:{id:'actual-editor-job'}},
    history:{originJobId:origin.jobId,originAssetId:originAsset.id,originSampleIndex:0,hydrationObserved:true,selectedRecipe:structuredClone(originAsset.recipe)}};
  return {row,expected,capture,intent:{startedAt:at(0),lockSha256,bindingSha256},original,uploadBytes,servedBytes:uploadBytes,asset,origin,originJob,originAsset,historyJob:structuredClone(originJob),historyAsset:structuredClone(originAsset)};
}
test('v4 Editor source binding admits only the observed opaque original raster and intact real history semantics', () => {
  const fixture=editorCase(),before=JSON.stringify(fixture.capture);
  const proof=validateEditorSource(fixture);assert.equal(proof.assetId,fixture.asset.id);assert.equal(proof.semanticRequestSha256,digest(fixture.expected));assert.equal(JSON.stringify(fixture.capture),before);
  assert.notEqual(sha(fixture.uploadBytes),sha(png([10,20,30]))); // Real canvas PNG encoding may differ; RGB cannot.
  const mutateBytes=(f,bytes)=>{f.uploadBytes=bytes;f.servedBytes=bytes;f.capture.sourceUpload.sha256=sha(bytes);};
  const mutations=[f=>f.row.transport='ui-full',f=>f.row.id='quality-mlx-base-single',f=>f.capture.bindingSha256=sha('other-binding'),f=>f.capture.network.method='GET',f=>f.capture.sourceUpload.inputId='L',f=>f.asset.projectId='foreign-project',f=>f.asset.lineage={jobId:'generated-output'},
    f=>f.capture.sourceUpload.network.response.id='other-upload',f=>f.capture.sourceUpload.network.path='/api/v1/projects/foreign/assets',f=>f.capture.sourceUpload.sha256=sha('unobserved-file'),f=>f.servedBytes=png([11,20,30]),
    f=>mutateBytes(f,png([10,20,30,0],6)),f=>mutateBytes(f,png([11,20,30,255],6)),f=>f.original.width=2,
    f=>f.capture.sourceUpload.observedAt=new Date(Date.parse(f.intent.startedAt)-1000).toISOString(),f=>f.asset.createdAt=f.capture.capturedAt,f=>f.capture.network.observedAt='invalid',
    f=>f.capture.network.request.referenceAssetIds.push('extra-ref'),f=>f.capture.network.request.referenceAssetIds=['unobserved-scratch'],f=>f.capture.network.request.sourceAssetId='different-first-slot',f=>f.capture.network.request.seed++,
    f=>f.capture.history.originJobId='fabricated-history',f=>f.capture.history.originAssetId='different-sample',f=>f.capture.history.originSampleIndex=2,f=>f.capture.history.hydrationObserved=false,
    f=>f.historyJob.payload.prompt='rewritten-history',f=>f.historyJob.result.assets[0].id='rewritten-output',f=>f.historyAsset.recipe.seed++,f=>f.historyAsset.lineage.jobId='wrong-origin',f=>f.capture.history.selectedRecipe={seed:175}];
  for(const mutate of mutations){const f=editorCase();mutate(f);assert.throws(()=>validateEditorSource(f));}
  assert.throws(()=>assertRequestBinding(fixture.expected,fixture.capture.network.request)); // No ordinary/UI-wide ID alias.
});
test('Editor source binding records raw cross-host recipe drift but accepts only exact FP32 sigma bits',()=>{
  const f=editorCase(),s=value=>value.rawAdapterSettings.mossImageEditExecutionReport.base.sampling.sigmas;
  const observed=new Map([[1,0.9929999709129332],[3,0.977999985218048],[5,0.959999978542328],[7,0.9409999847412108],[9,0.9179999828338624],[10,0.9049999713897704],[29,0.02400000207126141]]);
  for(const recipe of [f.historyJob.result.assets[0].recipe,f.historyAsset.recipe,f.capture.history.selectedRecipe])for(const [index,value] of observed)s(recipe)[index]=value;
  const proof=validateEditorSource(f);
  for(const comparison of Object.values(proof.historyComparison)) {
    assert.equal(comparison.rawEqual,false);assert.equal(comparison.typedEqual,true);assert.deepEqual(comparison.sigmaDifferingIndices,[...observed.keys()]);
  }
  const mutated=change=>{const copy=structuredClone(f);change(copy);assert.throws(()=>validateEditorSource(copy));};
  const bytes=new DataView(new ArrayBuffer(4));bytes.setFloat32(0,0.9929999709129332,false);bytes.setUint32(0,bytes.getUint32(0,false)+1,false);
  for(const get of [x=>x.historyJob.result.assets[0].recipe,x=>x.historyAsset.recipe,x=>x.capture.history.selectedRecipe])mutated(x=>{s(get(x))[1]=bytes.getFloat32(0,false);});
  mutated(x=>{s(x.historyAsset.recipe).pop();});
  mutated(x=>{s(x.capture.history.selectedRecipe)[0]=-0;});
  mutated(x=>{x.historyJob.result.assets[0].id='changed-output';});
  mutated(x=>{x.historyAsset.recipe.seed=174;});
});
test('v4 Editor adoption retains authenticated product read-back and final validation rejects changed evidence', async () => {
  const f=editorCase(),root=await temporary('editor-binding'),dest=resolve(root,'rows',f.row.id);await mkdir(dest,{recursive:true});
  const sources={sceneworks:{head:'1'.repeat(40),tree:'2'.repeat(40)},inference:{head:'3'.repeat(40),tree:'4'.repeat(40)}};
  const files=Object.fromEntries(BACKEND_FILES.map(name=>[name,{byteLength:1,sha256:sha(name)}]));
  const lock={version:VERSION,sources,files:{},artifacts:{cuda:{files}},createdAt:f.intent.startedAt};
  const config={localBackend:'cuda',sources:{},files:{},plan:'inert-plan',evidenceRoot:root,backends:{cuda:{apiUrl:'http://127.0.0.1:1234',projectId:f.expected.projectId,workerId:'fixture-worker',workerGpuId:'1',tokenEnv:'INERT_FIXTURE_TOKEN_ENV'}}};
  const binding=executionBinding(config,lock,files),state={root,config,lock,binding};f.capture.lockSha256=digest(lock);f.capture.bindingSha256=digest(binding);
  f.intent.lockSha256=f.capture.lockSha256;f.intent.bindingSha256=f.capture.bindingSha256;
  await save(resolve(root,'bindings','cuda.json'),binding);await save(resolve(root,'input-identities.json'),{B:f.original});await save(resolve(root,'attempts','cuda',`${f.row.id}.start.json`),f.intent);
  const origin=resolve(root,'rows','quality-mlx-sft-single');await save(resolve(origin,'receipt.json'),f.origin);await save(resolve(origin,'terminal-job.json'),f.originJob);await save(resolve(origin,'asset-0.json'),f.originAsset);
  const capturePath=resolve(root,'capture.json');await writeFile(resolve(root,f.capture.sourceUpload.file),f.uploadBytes);
  const calls=[],api={token:'INERT_NOT_A_CREDENTIAL',base:new URL(config.backends.cuda.apiUrl),request:async(path,options)=>{calls.push([path,options]);if(path.endsWith(`/assets/${f.asset.id}`))return f.asset;if(path.endsWith('/files/assets/scratch.png'))return f.servedBytes;if(path===`/api/v1/jobs/${f.origin.jobId}`)return f.historyJob;if(path.endsWith(`/assets/${f.originAsset.id}`))return f.historyAsset;throw Error('unexpected inert request');}};
  await assert.rejects(()=>retainEditorSource(capturePath,state,f.row,f.expected,dest,f.capture,{...api,token:''}),/authentication/);
  await retainEditorSource(capturePath,state,f.row,f.expected,dest,f.capture,api);
  assert.equal(calls.length,4);assert.equal(calls[1][1].raw,true);assert.equal((await validateRetainedEditor(state,f.row,f.expected,f.capture,dest)).assetId,f.asset.id);
  const altered=structuredClone(f.capture);altered.history.hydrationObserved=false;await assert.rejects(()=>validateRetainedEditor(state,f.row,f.expected,altered,dest),/capture\/reserved intent changed/);
  await writeFile(resolve(dest,'source-served.png'),png([99,20,30]));await assert.rejects(()=>validateRetainedEditor(state,f.row,f.expected,f.capture,dest),/bytes changed/);
});

test('file identity streams exact empty, small and chunk-boundary bytes and follows HF aliases',async t=>{
  const root=await mkdtemp(resolve(tmpdir(),'moss-identity-'));t.after(()=>rm(root,{recursive:true,force:true}));
  for(const size of [0,17,1024*1024-1,1024*1024,1024*1024+19]) {
    const bytes=Buffer.alloc(size);for(let i=0;i<size;i++)bytes[i]=i%251;
    const path=resolve(root,`bytes-${size}`);await writeFile(path,bytes);
    assert.deepEqual(await fileIdentity(path),{byteLength:size,sha256:sha(bytes)});
  }
  await symlink(resolve(root,'bytes-17'),resolve(root,'hf-alias'));
  assert.deepEqual(await fileIdentity(resolve(root,'hf-alias')),await fileIdentity(resolve(root,'bytes-17')));
  await assert.rejects(()=>fileIdentity(root),/regular file/);
});
test('file identity hashes a real sparse file larger than 2 GiB with an independent incremental digest',async t=>{
  const root=await mkdtemp(resolve(tmpdir(),'moss-identity-large-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const home=resolve(root,'hf');await mkdir(home);
  const path=resolve(home,'large.safetensors'),size=2**31+19,prefix=Buffer.from('first bytes'),suffix=Buffer.from('last exact bytes');
  const file=await open(path,'wx');try{await file.truncate(size);await file.write(prefix,0,prefix.length,0);await file.write(suffix,0,suffix.length,size-suffix.length);}finally{await file.close();}
  const expected=createHash('sha256').update(prefix),zeros=Buffer.alloc(7*1024*1024);
  for(let remaining=size-prefix.length-suffix.length;remaining>0;){const n=Math.min(remaining,zeros.length);expected.update(zeros.subarray(0,n));remaining-=n;}
  expected.update(suffix);
  const identity={byteLength:size,sha256:expected.digest('hex')};assert.deepEqual(await fileIdentity(path),identity);
  const config={root,hfHome:home,candidateId:'inert-large',editAdapterFiles:['large.safetensors'],packageFiles:[{file:'large.safetensors',sha256:identity.sha256}]};
  const corrupt=Buffer.from('inert corrupt fixture'),mutation=await adapterRefusalState(config,'corrupt-adapter',{adapterFile:'large.safetensors',corruptBytesBase64:corrupt.toString('base64'),corruptSha256:sha(corrupt)});
  try{assert.deepEqual(await readFile(path),corrupt);}finally{await mutation.restore();}
  assert.equal(mutation.observation.beforeSha256,identity.sha256);
});
test('file identity rejects deterministic in-place mutation, replacement, truncation and growth',async t=>{
  const root=await mkdtemp(resolve(tmpdir(),'moss-identity-change-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const probe=await open(resolve(root,'probe'),'wx'),prototype=Object.getPrototypeOf(probe),read=prototype.read;await probe.close();
  for(const kind of ['mutation','replacement','truncation','growth']) {
    const path=resolve(root,kind),bytes=Buffer.alloc(1024*1024+19,7);await writeFile(path,bytes);let changed=false;
    const mocked=t.mock.method(prototype,'read',async function(...args){
      const result=await read.apply(this,args);
      if(!changed&&result.bytesRead){changed=true;
        if(kind==='replacement'){const other=resolve(root,'replacement-new');await writeFile(other,bytes);await rename(other,path);}
        else if(kind==='truncation')await truncate(path,3);
        else if(kind==='growth'){const handle=await open(path,'a');try{await handle.write(Buffer.from('extra'));}finally{await handle.close();}}
        else {const handle=await open(path,'r+');try{await handle.write(Buffer.from([8]),0,1,0);}finally{await handle.close();}}
      }
      return result;
    });
    try{await assert.rejects(()=>fileIdentity(path),/changed while hashing/);assert.equal(changed,true);}finally{mocked.mock.restore();}
  }
});

test('actual replay wire supplies the API required payloadChanges without changing the source payload', async t => {
  const dest = await mkdtemp(resolve(tmpdir(), 'moss-replay-wire-'));
  t.after(() => rm(dest, { recursive: true, force: true }));
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    const body = JSON.parse(options.body);
    calls.push({ path: url.pathname, method: options.method, body });
    // DuplicateJobRequest in core/contracts.rs requires this object; {} is a decode failure.
    if (!Object.hasOwn(body, 'payloadChanges') || body.payloadChanges === null || Array.isArray(body.payloadChanges) || typeof body.payloadChanges !== 'object') {
      return new Response('{"detail":"missing field `payloadChanges`"}', { status: 422 });
    }
    return Response.json({ id: 'job_replay', duplicateOfJobId: 'job_original' }, { status: 201 });
  });
  const api = new Api('http://inert.invalid', 'test-only-secret');
  await assert.rejects(api.request('/api/v1/jobs/job_original/duplicate', { method: 'POST', body: {} }), { status: 422 });
  calls.length = 0;
  for (const action of ['duplicate', 'retry']) {
    assert.equal((await submitReplay(api, 'job_original', action, dest)).id, 'job_replay');
  }
  assert.deepEqual(calls, ['duplicate', 'retry'].map(action => ({ path: `/api/v1/jobs/job_original/${action}`, method: 'POST', body: { payloadChanges: {} } })));
  assert.deepEqual(await readdir(dest), []);
});

test('failed replay retains exact bounded HTTP response without authentication or request capture', async t => {
  const dest = await mkdtemp(resolve(tmpdir(), 'moss-replay-failure-'));
  t.after(() => rm(dest, { recursive: true, force: true }));
  const body = '{"detail":[{"type":"json_invalid","ctx":{"error":"missing field `payloadChanges`"}}]}\n';
  t.mock.method(globalThis, 'fetch', async () => new Response(body, { status: 422, headers: { 'content-type': 'application/json', 'set-cookie': 'not-evidence' } }));
  await assert.rejects(submitReplay(new Api('http://inert.invalid', 'test-only-secret'), 'job_original', 'duplicate', dest), { status: 422, responseBody: body });
  const receipt = JSON.parse(await readFile(resolve(dest, 'replay-response.json'), 'utf8'));
  assert.deepEqual(receipt, { method: 'POST', path: '/api/v1/jobs/job_original/duplicate', status: 422, contentType: 'application/json', byteLength: Buffer.byteLength(body), sha256: sha(body), body });
  assert.equal(JSON.stringify(receipt).includes('test-only-secret'), false);
  assert.equal(JSON.stringify(receipt).includes('not-evidence'), false);
});

test('transport failure and oversized response cannot create a replay HTTP receipt', async t => {
  const dest = await mkdtemp(resolve(tmpdir(), 'moss-replay-transport-'));
  t.after(() => rm(dest, { recursive: true, force: true }));
  const api = new Api('http://inert.invalid');
  const fetch = t.mock.method(globalThis, 'fetch', async () => { throw new Error('inert connection failure'); });
  await assert.rejects(submitReplay(api, 'job_original', 'retry', dest), /inert connection failure/);
  fetch.mock.mockImplementation(async () => new Response('x'.repeat(64 * 1024 + 1), { status: 422 }));
  await assert.rejects(submitReplay(api, 'job_original', 'duplicate', dest), { status: 422 });
  assert.deepEqual(await readdir(dest), []);
});
