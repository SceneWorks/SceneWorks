// Pure metadata controls; no process/service/model/device launch.
import assert from 'node:assert/strict';
import test from 'node:test';
import {readFile} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {ORIGINAL,validateControllerOnlyChanges,validateStagedMatch,validateOriginalBuild,matchedRuntime,validateResidentEvents,validateCompleteControl} from './h3-matched-resident-control.mjs';
import {validatePlan,validatePoolEvidence} from './h3-cuda-memory-terminal.mjs';
import {privateEnvironment} from './h3-finetune-terminal-runtime.mjs';
const GIB=1024**3;
function plan(){return {version:1,mode:'matched-resident-control',matchedResidentControlAuthorized:true,stationary_pair_authorized:true,sceneworks:'a'.repeat(40),inference:ORIGINAL.inference,requestSha256:'b5b5ba095203fab1eddb9ba1793162959328e16ee34a0ca6e19d44e2325bce23',requestPath:'D:\\request.json',licenseAcknowledgmentsPath:ORIGINAL.root+'/services/workstation-veda/config/license-acknowledgments.json',expectedRunner:'cuda-windows',physicalGpuIndex:0,physicalGpuUuid:matchedRuntime().physicalGpuUuid,seed:42,poolBytes:92*GIB,externalReserveBytes:3*GIB,hostBudgetBytes:384*GIB,workstationDeviceBytes:95*GIB,timeoutSeconds:7200,sources:Object.fromEntries([
  ['tier','SceneWorks--minimax-h3-mlx','137ce668c55a20bc0935fd1cf2a3de8448abb7f4'],['base','MiniMaxAI--MiniMax-H3','6818f6c32d12b210915e44ad56a4228c2608f160'],['predictor','Veda-Sparse--Minimax-H3-T2VA-Veda-8NFE-600Step-Preview','9a1fd3a41b4a754a7886e64e82edbddf599fd1bd'],['turbo','lightx2v--Minimax-h3-Turbo','5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1']
].map(([role,repo,revision])=>[role,{revision,root:`E:\\huggingface\\hub\\models--${repo}\\snapshots\\${revision}`}]))};}
function build(){const command=['cargo','build','--locked','--profile','release','--manifest-path','D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks\\sceneworks\\Cargo.toml','-p','sceneworks-rust-api','--bin','sceneworks-rust-api','--features','sceneworks-rust-api/backend-candle','--features','sceneworks-rust-api/h3-memory-proof'];return {binary:ORIGINAL.binary,sha256:ORIGINAL.binarySha256,sceneworks:ORIGINAL.sceneworks,inference:ORIGINAL.inference,backend:'cuda',command,configuration:{command:[...command],proof_kind:'campaign_native',flags:{RUSTFLAGS:null,CARGO_ENCODED_RUSTFLAGS:null,CARGO_BUILD_TARGET:null,RUSTUP_TOOLCHAIN:null}}};}
function resident(){return [{event:'generation_memory_policy_applied',engine:'minimax_h3',strategy:'Resident',scope_configured:true,stage_residency:false,stream_transformer_blocks:false,tile_vae_decode:false,chunk_attention:false,window_component:'none'},{event:'h3_attention_execution',transformer_window:'None',transformer_component:'None',attention_chunk_size:'None',flash_calls:2,materialized_calls:50}];}
test('mode requires explicit one-control authority and exact original GPU0/92+3/M4/request/seed',()=>{
  assert.equal(validatePlan(plan()).mode,'matched-resident-control');
  for(const delta of [{mode:'other'},{mode:undefined},{matchedResidentControlAuthorized:false},{stationary_pair_authorized:false},{poolBytes:93*GIB},{poolBytes:21*GIB},{externalReserveBytes:2*GIB},{workstationDeviceBytes:96*GIB},{physicalGpuIndex:1},{seed:43},{inference:'b'.repeat(40)},{requestSha256:'b'.repeat(64)},{additionalSparse24:true},{retainedWorkstation:{}},{primitiveProof:{}},{licenseAcknowledgmentsPath:'D:/different.json'},{binaryPath:'D:/other.exe'}])assert.throws(()=>validatePlan({...plan(),...delta}));
  const env=privateEnvironment('D:\\fake', {...matchedRuntime(),backend:'cuda'},'test-token');
  assert.equal(env.SCENEWORKS_CUDA_VRAM_CAP_GB,'95');assert.equal(env.SCENEWORKS_CUDA_PROOF_POOL_BYTES,String(92*GIB));
  assert.equal(env.SCENEWORKS_H3_VEDA_FORCE_DENSE,undefined);assert.equal(env.SCENEWORKS_H3_ATTENTION_ROUTE,undefined);
});
test('assembled existing CLI validates explicit control and refuses local execution before asset/service work',()=>{
  const env={...process.env,SW_F329_PLAN:JSON.stringify(plan()),GITHUB_ACTIONS:'false'};
  const script=fileURLToPath(new URL('./h3-cuda-memory-terminal.mjs',import.meta.url));
  const check=spawnSync(process.execPath,[script,'check'],{env,encoding:'utf8'});assert.equal(check.status,0,check.stderr);assert.match(check.stdout,/no model executed/);
  const run=spawnSync(process.execPath,[script,'run','.','.','D:/unused-control'],{env,encoding:'utf8'});assert.notEqual(run.status,0);assert.match(run.stderr,/AssertionError/);
});
test('original executable and full original build receipt are immutable, never a new compile identity',()=>{
  const identities={receiptSha256:ORIGINAL.buildReceiptSha256,binarySha256:ORIGINAL.binarySha256};validateOriginalBuild(build(),identities);
  for(const mutate of [r=>r.binary='D:/other.exe',r=>r.sha256='f'.repeat(64),r=>r.sceneworks='a'.repeat(40),r=>r.inference='b'.repeat(40),r=>r.command.push('--no-default-features'),r=>r.configuration.flags.RUSTFLAGS='-C target-cpu=native',r=>r.configuration.proof_kind='unit_fake']){const r=build();mutate(r);assert.throws(()=>validateOriginalBuild(r,identities));}
  for(const key of ['receiptSha256','binarySha256'])assert.throws(()=>validateOriginalBuild(build(),{...identities,[key]:'f'.repeat(64)}));
});
test('only explicit controller files can differ from the actual compiled CGPU closure',()=>{
  validateControllerOnlyChanges(['scripts/h3-matched-resident-control.mjs','docs/design/minimax-h3-cuda-low-vram-plan.md']);
  for(const file of ['Cargo.lock','Cargo.toml','crates/sceneworks-worker/src/video_jobs.rs','apps/rust-api/src/main.rs','apps/web/src/index.ts','config/models.json','scripts/arbitrary.mjs'])assert.throws(()=>validateControllerOnlyChanges([file]));
});
test('frozen whole staged collection cannot lose or substitute source files, revisions or bytes',()=>{
  const sources=[{role:'tier',repo:'model',revision:'a'.repeat(40),relative:'q4/file.safetensors',bytes:123,sha256:'a'.repeat(64)}];validateStagedMatch(sources,sources.map(row=>({...row,path:'D:/fresh-stage'})));
  for(const change of [[],[...sources,...sources],[{...sources[0],revision:'b'.repeat(40)}],[{...sources[0],sha256:'f'.repeat(64)}],[{...sources[0],relative:'different.safetensors'}]])assert.throws(()=>validateStagedMatch(sources,change));
});
test('actual Resident/windowNone/zero-transfers is mandatory even when labels claim Resident',()=>{
  assert.equal(validateResidentEvents(resident()).windowTransfers,0);
  for(const mutate of [e=>e[0].strategy='BoundedTransformerResidency',e=>e[0].stream_transformer_blocks=true,e=>e[0].tile_vae_decode=true,e=>e[0].chunk_attention=true,e=>e[0].window_size=1,e=>e[1].transformer_window='Some(1)',e=>e.push({event:'h3_transformer_window_materialized'}),e=>e.pop(),e=>e.push(e[0])]){const e=resident();mutate(e);assert.throws(()=>validateResidentEvents(e));}
});
test('95GiB observation explicitly preserves 92pool+3reserve without changing default24 validation',()=>{
  const snapshot={event:'h3_cuda_proof_pool_snapshot',job_id:'job',logical_ordinal:0,cap_bytes:92*GIB,reserved_high_bytes:40*GIB,used_high_bytes:39*GIB,reserved_bytes:40*GIB,device_used_bytes:41*GIB};
  const options={jobId:'job',poolBytes:92*GIB,externalReserveBytes:3*GIB,totalEnvelopeBytes:95*GIB,metrics:{samples:[{host:1,device:41*GIB},{host:2,device:42*GIB}]}};
  assert.equal(validatePoolEvidence([snapshot],options).totalEnvelopeBytes,95*GIB);
  assert.throws(()=>validatePoolEvidence([snapshot],{...options,totalEnvelopeBytes:24*GIB}));
  assert.throws(()=>validatePoolEvidence([{...snapshot,device_used_bytes:44*GIB}],options));
  assert.throws(()=>validatePoolEvidence([snapshot],{...options,metrics:{samples:[{host:1,device:96*GIB},{host:1,device:42*GIB}]}}));
});
test('partial result, missing cleanup/source/binary/media/observations never validates a control',()=>{
  const libraries=['cudart64_12.dll','nvrtc64_120_0.dll','curand64_10.dll','cublas64_12.dll','cublasLt64_12.dll','nvcuda.dll','VCRUNTIME140.dll'].map(name=>({name,path:'D:/SIMULATED/'+name,version:'SIMULATED',bytes:1,sha256:'a'.repeat(64)}));
  const r={kind:'matched-resident-control',completed:true,ownedShutdownCompleted:true,controllerSourcePair:{sceneworks:'a'.repeat(40),inference:ORIGINAL.inference},runtimeSourcePair:{sceneworks:ORIGINAL.sceneworks,inference:ORIGINAL.inference},binarySha256:ORIGINAL.binarySha256,job:{id:'job_SIMULATED',status:'completed'},media:[{sha256:'a'.repeat(64)}],events:resident(),residency:{policy:{strategy:'Resident'},windowTransfers:0},envelope:{totalEnvelopeBytes:95*GIB,externalReserveBytes:3*GIB},content:{},avValidation:{},libraries,samples:[{},{}]};validateCompleteControl(r);
  for(const mutate of [r=>r.completed=false,r=>r.ownedShutdownCompleted=false,r=>r.error='partial',r=>r.binaryError='changed',r=>r.media=[],r=>r.samples=[],r=>r.libraries=[],r=>r.libraries[0].sha256='bad',r=>r.events=[],r=>r.runtimeSourcePair.sceneworks='a'.repeat(40),r=>r.job.status='cancelled',r=>r.residency.windowTransfers=1]){const copy=structuredClone(r);mutate(copy);assert.throws(()=>validateCompleteControl(copy));}
});
test('workflow reuses original runtime checkout and existing guarded manual lease, not an API build',async()=>{
  const source=await readFile(new URL('./h3-matched-resident-control.mjs',import.meta.url),'utf8');
  assert.match(source,/class ImmutableOriginalService extends StockService/);assert.match(source,/runtime-sceneworks/);assert.match(source,/historicalLoadedDllHashes:'not recorded/);assert.doesNotMatch(source,/await (?:runner|exec)\(['"]cargo/);
  const workflow=await readFile(new URL('../.github/workflows/h3-cuda-memory-terminal.yml',import.meta.url),'utf8');assert.match(workflow,/path: runtime-sceneworks/);assert.match(workflow,/ref: c8815c7cd46feba2f82562a315d1a44a6c88a9e8/);assert.match(workflow,/matched-resident-control\/\*\.json/);
  assert.match(workflow,/python scripts\\h3-cuda-memory-lease\.py/);assert.doesNotMatch(workflow,/node scripts\\h3-matched-resident-control\.mjs run/);
});
