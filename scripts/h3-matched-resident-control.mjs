// Conditional manual diagnostic. Never a feature/campaign acceptance or new build.
import assert from 'node:assert/strict';
import { copyFile,mkdir,readFile,realpath,stat,writeFile } from 'node:fs/promises';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileSha256 } from './lib/file-sha256.mjs';
import { StockService,stageModels,validatePlan,anchorRequest,campaignCudaEnvironment,validateAnchorDevice,validatePoolEvidence,collectDecodedMediaDiagnostics } from './h3-cuda-memory-terminal.mjs';
import { ProcessObservations,assertQueuedMatchesSubmitted,bindCompletedMedia,hardware } from './h3-finetune-terminal-runtime.mjs';
import { runScenario } from './h3-finetune-terminal-api.mjs';
import { sourcePair,verifyRenderedContent,validateAVAgainstRequest } from './h3-finetune-terminal.mjs';

const exec=promisify(execFile),GIB=1024**3;
const load=async file=>JSON.parse(await readFile(file,'utf8'));
const save=async(file,value)=>writeFile(file,JSON.stringify(value,null,2)+'\n',{flag:'wx'});
export const ORIGINAL=Object.freeze({
  sceneworks:'c8815c7cd46feba2f82562a315d1a44a6c88a9e8',inference:'2ff0353820e370d004305ae99a035ab38b93de30',
  root:'D:/Codex/work/sw-f329-acceptance-42243645213-1',
  binary:'D:/sc26691-campaign/sc26691-private/42243645213-1-target/release/sceneworks-rust-api.exe',
  binarySha256:'a5e83b704d67b999b1919d7308654d42dc9e1bc2de66347c67b0f07242f9b621',
  buildReceiptSha256:'8ac02a48dd1e1c5c1b523dd45684c6704658402808b357804ef46ca7758a3343',
  licenseSha256:'dd4544529abaf532afb3c2eda712348d190c9f4e65f0233c4cb7424e1697145a',
  stagedSha256:'a58d6f7e2d2655a5537b5f7deac02fb95694d5e0a1ec1d47161a14801223d6dd',
  anchors:Object.freeze({'workstation-veda':'f060da00e0d14b5ba58237274587228ab11ffd7549bd5d72179ca510098a6c36','24gib-veda-sparse':'45a7fdd93c48e2c0c61b97a4a811b2915f289142a6e38320d608936a827cb8f7'}),
});
const CONTROLLER_ONLY=new Set([
  'docs/design/minimax-h3-cuda-low-vram-plan.md','docs/testing/sw-f329-cuda-memory-terminal.md',
  'scripts/h3-cuda-memory-terminal.mjs','scripts/h3-cuda-memory-terminal.test.mjs',
  'scripts/h3-residency-component-evidence.mjs','scripts/h3-residency-component-evidence.test.mjs',
  'scripts/h3-matched-resident-control.mjs','scripts/h3-matched-resident-control.test.mjs',
  'scripts/h3-cuda-memory-lease.py','scripts/h3-cuda-memory-lease.test.py','.github/workflows/h3-cuda-memory-terminal.yml',
]);
export function validateControllerOnlyChanges(paths){
  assert.ok(paths.every(file=>CONTROLLER_ONLY.has(file)),'retained binary compiled-source closure changed');
  return paths;
}
export function validateStagedMatch(previous,current){
  const identities=files=>files.map(({role,repo,revision,relative,bytes,sha256})=>({role,repo,revision,relative,bytes,sha256})).sort((a,b)=>JSON.stringify(a).localeCompare(JSON.stringify(b)));
  assert.deepEqual(identities(current),identities(previous),'actual staged source collection must match original bytes');
}
export function validateOriginalBuild(receipt,{receiptSha256,binarySha256}){
  assert.equal(receiptSha256,ORIGINAL.buildReceiptSha256);assert.equal(binarySha256,ORIGINAL.binarySha256);
  assert.equal(receipt.sha256,binarySha256);assert.equal(path.win32.resolve(receipt.binary),path.win32.resolve(ORIGINAL.binary));
  assert.equal(receipt.sceneworks,ORIGINAL.sceneworks);assert.equal(receipt.inference,ORIGINAL.inference);assert.equal(receipt.backend,'cuda');
  const command=['cargo','build','--locked','--profile','release','--manifest-path','D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks\\sceneworks\\Cargo.toml','-p','sceneworks-rust-api','--bin','sceneworks-rust-api','--features','sceneworks-rust-api/backend-candle','--features','sceneworks-rust-api/h3-memory-proof'];
  assert.deepEqual(receipt.command,command);assert.deepEqual(receipt.configuration.command,command);
  assert.equal(receipt.configuration.proof_kind,'campaign_native');
  assert.deepEqual(receipt.configuration.flags,{RUSTFLAGS:null,CARGO_ENCODED_RUSTFLAGS:null,CARGO_BUILD_TARGET:null,RUSTUP_TOOLCHAIN:null});
  return receipt;
}
export function matchedRuntime(){
  return {port:0,gpuId:'0',deviceId:'GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0',physicalGpuIndex:0,physicalGpuUuid:'GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0',
    cudaProofFeatures:['sceneworks-rust-api/h3-memory-proof'],memoryCap:{kind:'cuda-vram-gb',value:95},cudaProofEnvironment:{SCENEWORKS_CUDA_PROOF_POOL_BYTES:String(92*GIB)}};
}
export function validateResidentEvents(events){
  const policies=events.filter(e=>e.event==='generation_memory_policy_applied'&&e.engine==='minimax_h3');assert.equal(policies.length,1,'one actual H3 request policy required');
  const p=policies[0];assert.equal(p.strategy,'Resident');assert.equal(p.scope_configured,true);
  for(const field of ['stage_residency','stream_transformer_blocks','tile_vae_decode','chunk_attention'])assert.equal(p[field],false);
  assert.equal(p.window_component,'none');assert.equal(p.window_size,undefined);
  assert.equal(events.filter(e=>e.event==='h3_transformer_window_materialized').length,0,'Resident control must transfer zero streamed blocks');
  const reports=events.filter(e=>e.event==='h3_attention_execution');assert.equal(reports.length,1);
  assert.equal(reports[0].transformer_window,'None');assert.equal(reports[0].transformer_component,'None');
  assert.equal(reports[0].attention_chunk_size,'None');assert.ok(reports[0].flash_calls>0&&reports[0].materialized_calls>0);
  return {policy:p,attention:reports[0],windowTransfers:0};
}
export function validateCompleteControl(record){
  assert.equal(record.kind,'matched-resident-control');assert.equal(record.completed,true);assert.equal(record.ownedShutdownCompleted,true);
  for(const key of ['error','shutdownError','observationError','binaryError'])assert.equal(record[key],undefined,'partial control is not completed evidence');
  assert.equal(record.runtimeSourcePair.sceneworks,ORIGINAL.sceneworks);assert.equal(record.runtimeSourcePair.inference,ORIGINAL.inference);
  assert.match(record.controllerSourcePair.sceneworks,/^[a-f0-9]{40}$/);assert.equal(record.controllerSourcePair.inference,ORIGINAL.inference);
  assert.equal(record.binarySha256,ORIGINAL.binarySha256);assert.equal(record.job.status,'completed');assert.equal(record.media.length,1);
  assert.match(record.job.id,/^job_/);assert.match(record.media[0].sha256,/^[a-f0-9]{64}$/);
  assert.ok(typeof record.media[0].av==='string'&&record.media[0].av.length>0);
  assert.equal(record.content.frames,345);assert.ok(Number.isFinite(record.content.spatialStd.median));
  assert.equal(record.avValidation.frameCount,345);assert.equal(record.avValidation.width,1344);assert.equal(record.avValidation.height,768);assert.equal(record.avValidation.fps,24);
  assert.equal(record.avValidation.audio.codec,'aac');assert.equal(record.avValidation.audio.sampleRate,32000);assert.equal(record.avValidation.audio.channels,2);assert.equal(record.avValidation.audio.expectedSamples,460000);
  assert.equal(record.generationMetrics.steps,8);const veda=record.generationMetrics.vedaExecution??record.generationMetrics.veda_execution;
  assert.equal(veda.requested,true);assert.ok((veda.sparseCalls??veda.sparse_calls)>0);assert.equal(veda.denseCalls??veda.dense_calls,0);
  validateResidentEvents(record.events);
  assert.equal(record.residency.windowTransfers,0);assert.equal(record.residency.policy.strategy,'Resident');
  assert.equal(record.envelope.totalEnvelopeBytes,95*GIB);assert.equal(record.envelope.externalReserveBytes,3*GIB);
  assert.ok(record.content&&record.avValidation&&record.samples?.length>=2);
  for(const name of [...CUDA_DLLS,'nvcuda.dll','VCRUNTIME140.dll'])assert.ok(record.libraries.some(row=>row.name.toLowerCase()===name.toLowerCase()));
  for(const row of record.libraries){assert.match(row.sha256,/^[a-f0-9]{64}$/);assert.ok(typeof row.version==='string'&&row.version.length>0&&typeof row.path==='string'&&row.path.length>0&&Number.isSafeInteger(row.bytes)&&row.bytes>0);}
  return record;
}
async function verifiedOriginal(relative,expected){const file=path.join(ORIGINAL.root,relative);assert.equal(await fileSha256(file),expected,`original ${relative} changed`);return file;}
async function originalBuild(){
  const file=await verifiedOriginal('workstation-veda/service-build.json',ORIGINAL.buildReceiptSha256);
  return validateOriginalBuild(await load(file),{receiptSha256:await fileSha256(file),binarySha256:await fileSha256(ORIGINAL.binary)});
}
class ImmutableOriginalService extends StockService{
  async build(){
    this.buildIdentity=await originalBuild();this.binary=ORIGINAL.binary;this.binaryHash=ORIGINAL.binarySha256;
    await copyFile(path.join(ORIGINAL.root,'workstation-veda/service-build.json'),path.join(this.dir,'service-build.json'));
    await copyFile(path.join(ORIGINAL.root,'workstation-veda/service-build.log'),path.join(this.dir,'service-build.log'));
    await save(path.join(this.dir,'prebuilt-reuse.json'),{kind:'immutable-original-build-reuse',runtimeSourcePair:{sceneworks:ORIGINAL.sceneworks,inference:ORIGINAL.inference},controllerSourcePair:this.controllerSourcePair,originalBuildReceiptSha256:ORIGINAL.buildReceiptSha256,binarySha256:this.binaryHash,meaning:'Original CGPU/M4 build reused read-only; no Cargo executed or new-head compile claimed'});
  }
}
const CUDA_DLLS=['cudart64_12.dll','nvrtc64_120_0.dll','curand64_10.dll','cublas64_12.dll','cublasLt64_12.dll'];
async function libraryEvidence(pid){
  assert.ok(Number.isSafeInteger(pid)&&pid>1);
  const script=`$ErrorActionPreference='Stop'; @(Get-Process -Id ${pid} -ErrorAction Stop | Select-Object -ExpandProperty Modules | Where-Object { $_.ModuleName -match '^(cudart|nvcuda|nvrtc|curand|cublas|vcruntime)' } | ForEach-Object { [pscustomobject]@{name=$_.ModuleName;path=$_.FileName;version=$_.FileVersionInfo.FileVersion} }) | ConvertTo-Json -Compress`;
  const {stdout}=await exec('powershell.exe',['-NoProfile','-NonInteractive','-Command',script],{timeout:30000});
  const rows=JSON.parse(stdout);assert.ok(Array.isArray(rows));
  for(const name of [...CUDA_DLLS,'nvcuda.dll','VCRUNTIME140.dll'])assert.ok(rows.some(row=>row.name.toLowerCase()===name.toLowerCase()),`actual loaded ${name} required`);
  for(const row of rows){row.path=await realpath(row.path);row.sha256=await fileSha256(row.path);row.bytes=(await stat(row.path)).size;}
  return rows;
}
function events(log){return log.split('\n').flatMap(line=>{try{const e=JSON.parse(line);return [e.fields??e];}catch{return [];}});}

export async function runMatchedResidentControl(p,controllerRoot,inferenceRoot,output){
  validatePlan(p);assert.equal(p.mode,'matched-resident-control');assert.equal(process.platform,'win32');
  assert.equal(process.env.GITHUB_ACTIONS,'true');assert.equal(process.env.GITHUB_EVENT_NAME,'workflow_dispatch');assert.equal(process.env.GITHUB_SHA,p.sceneworks);assert.equal(process.env.RUNNER_NAME,p.expectedRunner);
  campaignCudaEnvironment(p,process.env);assert.ok(path.resolve(output).toLowerCase().startsWith('d:\\codex\\work\\'));
  // The separate checkout supplies the original runtime configuration and registry.
  const runtimeRoot=path.join(path.dirname(controllerRoot),'runtime-sceneworks');
  assert.equal((await exec('git',['rev-parse','HEAD'],{cwd:controllerRoot})).stdout.trim(),p.sceneworks);
  assert.equal((await exec('git',['status','--porcelain'],{cwd:controllerRoot})).stdout.trim(),'');
  const changed=(await exec('git',['diff','--name-only','--no-renames',ORIGINAL.sceneworks,p.sceneworks],{cwd:controllerRoot})).stdout.trim().split('\n').filter(Boolean);
  validateControllerOnlyChanges(changed);
  await mkdir(output);const runtimePlan={...p,output,sceneworks:ORIGINAL.sceneworks};
  await sourcePair(runtimePlan,runtimeRoot,inferenceRoot);await originalBuild();
  const license=await verifiedOriginal('services/workstation-veda/config/license-acknowledgments.json',ORIGINAL.licenseSha256);runtimePlan.licenseAcknowledgmentsPath=license;
  assert.equal(await fileSha256(p.requestPath),p.requestSha256);const request=anchorRequest(await load(p.requestPath),42,0);
  await save(path.join(output,'source-identity.json'),{kind:'matched-resident-control',controllerSourcePair:{sceneworks:p.sceneworks,inference:p.inference},runtimeSourcePair:{sceneworks:ORIGINAL.sceneworks,inference:ORIGINAL.inference},changedControllerOnlyPaths:changed,originalBinarySha256:ORIGINAL.binarySha256,originalBuildReceiptSha256:ORIGINAL.buildReceiptSha256,historicalLoadedDllHashes:'not recorded; new loaded-module hashes below are not retroactive historical attestations'});
  const staged=await stageModels(runtimePlan,output);
  const oldStaged=await load(await verifiedOriginal('staged-models.json',ORIGINAL.stagedSha256));
  validateStagedMatch(oldStaged.files,staged.files);
  await save(path.join(output,'request.json'),{request,originalRequestSha256:p.requestSha256});
  await save(path.join(output,'hardware.json'),await hardware('cuda',matchedRuntime().deviceId,matchedRuntime()));
  const cudaBin='C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin';assert.equal(path.win32.resolve(process.env.CUDA_PATH??''),path.win32.dirname(cudaBin));
  process.env.PATH=cudaBin+';'+process.env.PATH;
  for(const key of ['SCENEWORKS_GPU_MEMORY_LIMIT_BYTES','SCENEWORKS_CUDA_VRAM_CAP_GB','SCENEWORKS_CUDA_PROOF_POOL_BYTES','SCENEWORKS_H3_VEDA_FORCE_DENSE','SCENEWORKS_H3_ATTENTION_ROUTE'])delete process.env[key];
  const dir=path.join(output,'matched-resident-control');await mkdir(dir);
  const c={id:'matched-resident-control',backend:'cuda',timeoutSeconds:p.timeoutSeconds,loadHostBudgetBytes:p.hostBudgetBytes,loadDeviceBudgetBytes:95*GIB,declaredSourceBindings:[{family:'h3',model:{modelId:'minimax_h3',baseModel:'minimax_h3'}}],runtime:matchedRuntime()};
  const service=new ImmutableOriginalService(runtimePlan,c,runtimeRoot,inferenceRoot,dir,{hardDeadlineMs:Date.now()+(p.timeoutSeconds+900)*1000});service.staged=staged;service.controllerSourcePair={sceneworks:p.sceneworks,inference:p.inference};
  const record={kind:'matched-resident-control',completed:false,controllerSourcePair:service.controllerSourcePair,runtimeSourcePair:{sceneworks:ORIGINAL.sceneworks,inference:ORIGINAL.inference},binarySha256:ORIGINAL.binarySha256,settings:request,scope:'One observational Resident/pool control; not feature/campaign/24GiB-fit or final-media numerical acceptance'};
  let observer;
  try{
    await service.initialize();record.libraries=await libraryEvidence(service.worker.pid);
    observer=new ProcessObservations(service,{intervalMs:250}).start();
    const transcript=[],scenario={caseId:c.id,baseUrl:'http://127.0.0.1:0',timeoutSeconds:p.timeoutSeconds,steps:[{id:'project',method:'POST',path:'/api/v1/projects',body:{name:'SW-F329 matched Resident control'},expectedStatus:201},{id:'submit',method:'POST',path:'/api/v1/video/jobs',body:{...request,projectId:{$capture:'project/id'}},expectedStatus:201},{id:'complete',waitJob:{$capture:'submit/id'},assert:[{pointer:'/status',equals:'completed'}]}]};
    const executed=await runScenario(scenario,{baseUrl:()=>service.baseUrl,fetchImpl:(...args)=>service.fetch(...args),beforeStep:({step,captures})=>service.checkStep(step,captures),onRecord:async row=>{transcript.push(row);if(row.method==='POST'&&row.route==='/api/v1/video/jobs'&&row.result?.id)service.jobs.add(row.result.id);await writeFile(path.join(dir,'api-transcript.json'),JSON.stringify(transcript,null,2)+'\n');}});
    record.job=executed.captures.complete;assertQueuedMatchesSubmitted({...request,projectId:executed.captures.project.id},record.job);validateAnchorDevice(record.job,0);
    record.generationMetrics=await service.request('GET',`/api/v1/jobs/${record.job.id}/metrics`);
    const veda=record.generationMetrics.vedaExecution??record.generationMetrics.veda_execution;assert.ok((veda?.sparseCalls??veda?.sparse_calls)>0);assert.equal(veda.denseCalls??veda.dense_calls,0);
    assert.equal(record.generationMetrics.steps,8);
    record.media=await bindCompletedMedia(service,record.job,dir,'video');assert.equal(record.media.length,1);
    const raw=record.media[0].asset.recipe?.rawAdapterSettings;assert.equal(raw?.effectiveSteps,8);assert.equal(raw?.minimaxH3Tier,'q4');
    record.content=await verifyRenderedContent(path.join(dir,record.media[0].av));
    record.avValidation=validateAVAgainstRequest(record.media[0].decoded,request,{model:'minimax_h3',fps:24,width:1344,height:768,audio:{required:true,sampleRate:32000,channels:2,codec:'aac'}});
    record.observations=await observer.finish();record.samples=observer.samples;observer=undefined;
    const log=service.worker.log();assert.match(log,/CUDA_PROOF_POOL installed logical_ordinal=0 cap_bytes=98784247808/);
    record.events=events(log);record.residency=validateResidentEvents(record.events);
    record.envelope=validatePoolEvidence(record.events,{jobId:record.job.id,poolBytes:92*GIB,externalReserveBytes:3*GIB,totalEnvelopeBytes:95*GIB,metrics:{samples:record.samples}});
    record.completed=true;
  }catch(error){record.error=error.stack;throw error;}
  finally{
    if(observer)try{record.observations=await observer.finish();}catch(error){record.observationError=error.message;}
    if(service.worker)try{await service.stop();record.ownedShutdownCompleted=true;}catch(error){record.shutdownError=error.stack;record.completed=false;}
    try{assert.equal(await fileSha256(ORIGINAL.binary),ORIGINAL.binarySha256);}catch(error){record.binaryError=error.stack;record.completed=false;}
    await save(path.join(dir,'result.json'),record);
  }
  validateCompleteControl(record);
  // All complete AV metrics are observations, with no invented pixel threshold.
  const comparisons=[];
  for(const [id,digest] of Object.entries(ORIGINAL.anchors)){
    const previous=await load(await verifiedOriginal(`${id}/result.json`,digest));assert.equal(previous.passed,true);assert.deepEqual(previous.sourcePair,record.runtimeSourcePair);assert.deepEqual(previous.settings,request);
    const media=previous.media[0],oldFile=path.join(ORIGINAL.root,id,media.av),newFile=path.join(dir,record.media[0].av);assert.equal(await fileSha256(oldFile),media.sha256);
    const comparisonDir=path.join(output,`observed-vs-${id}`);await mkdir(comparisonDir);
    const metrics=await collectDecodedMediaDiagnostics(oldFile,newFile,comparisonDir,{videoElements:1344*768*345*3,audioSamples:record.avValidation.audio.expectedSamples,audioPadding:record.avValidation.audio.frameSamples+1});
    comparisons.push({originalAnchor:id,originalReceiptSha256:digest,originalMediaSha256:media.sha256,newMediaSha256:record.media[0].sha256,metrics,interpretation:'Observational only; no E5/feature/media-equality acceptance threshold'});
    assert.equal(await fileSha256(oldFile),media.sha256);
  }
  await sourcePair(runtimePlan,runtimeRoot,inferenceRoot);
  await save(path.join(output,'matched-resident-control.json'),{kind:'matched-resident-control',executionValidated:true,controllerSourcePair:record.controllerSourcePair,runtimeSourcePair:record.runtimeSourcePair,jobId:record.job.id,comparisons,scope:record.scope,cleanupProof:'Separate existing physical-GPU lease receipt must additionally verify post_execution_verified'});
}
