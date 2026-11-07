// Conditional manual diagnostic. Never a feature/campaign acceptance or new build.
import assert from 'node:assert/strict';
import { copyFile,mkdir,readFile,realpath,stat,writeFile } from 'node:fs/promises';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { lockedInferenceRevision } from './select-windows-cargo-cache.mjs';
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
// This is the reviewed JS execution/import closure, NOT the compiled runtime
// tree. Cargo/Rust/UI changes at controller HEAD are never compiled or launched.
export const COMMON_CONTROLLER_FILES=Object.freeze([
  'scripts/ci/h3-import-native-proof.mjs',
  'scripts/h3-finetune-terminal-api.mjs','scripts/h3-finetune-terminal-cases.mjs',
  'scripts/h3-finetune-terminal-reduced.mjs','scripts/h3-finetune-terminal-runtime.mjs',
  'scripts/h3-finetune-terminal-stage.mjs','scripts/h3-finetune-terminal.mjs',
  'scripts/lib/file-sha256.mjs','scripts/lib/h3-campaign-build-profile.mjs',
  'scripts/lib/h3-owned-processes.mjs','scripts/lib/h3-peer-lane.mjs',
  'scripts/lib/rust-facade-source.mjs','scripts/lib/rust-source-slices.mjs',
  'scripts/lib/windows-verbatim-path.mjs','scripts/scan-inference-provenance.mjs',
  'scripts/select-windows-cargo-cache.mjs','docs/testing/sc-26691-h3-cases.json',
]);
export const MANUAL_CONTROLLER_FILES=Object.freeze([
  'scripts/h3-cuda-memory-terminal.mjs','scripts/h3-matched-resident-control.mjs',
]);
const MANUAL_IMPORTS={
  'scripts/h3-cuda-memory-terminal.mjs':[
    'node:assert/strict','node:crypto','node:fs/promises','node:path','node:url','node:util','node:child_process',
    './lib/file-sha256.mjs','./scan-inference-provenance.mjs','./lib/rust-source-slices.mjs',
    './lib/rust-facade-source.mjs','./h3-finetune-terminal-runtime.mjs','./ci/h3-import-native-proof.mjs',
    './h3-finetune-terminal-api.mjs','./h3-finetune-terminal.mjs',
  ],
  'scripts/h3-matched-resident-control.mjs':[
    'node:assert/strict','node:fs/promises','node:path','node:child_process','node:util',
    './select-windows-cargo-cache.mjs','./lib/file-sha256.mjs','./h3-cuda-memory-terminal.mjs',
    './h3-finetune-terminal-runtime.mjs','./h3-finetune-terminal-api.mjs','./h3-finetune-terminal.mjs',
  ],
};
export const RUNTIME_TREES=Object.freeze({sceneworks:'229377ce79cf9c3ec0660b2346af13807ee11664',
  inference:'75d392586953d990cf346c7b82f69f7ef246b133',config:'f70f28c989d3b5fe018d8bbd863d19ca11aa7847',
  lock:'d3dc83f19044ad80b3924f2d321f6480fd7afaa5'});

export function validateManualImports(file,source){
  // A deliberately bounded assertion for these two reviewed modules, not a
  // generic JavaScript parser. All imports must retain their literal single-line
  // declaration form; new re-exports/require/loaders need another source review.
  const lines=source.split(/\r?\n/).filter(line=>/^\s*import\b/.test(line));
  const imports=lines.map(line=>{const match=/^import .+ from ['"]([^'"]+)['"];?$/.exec(line);assert.ok(match,'literal reviewed imports required');return match[1];});
  assert.deepEqual(imports,MANUAL_IMPORTS[file],'manual controller import closure changed');
  assert.doesNotMatch(source,/^\s*export\s+.*\bfrom\s*['"]|\brequire\s*\(/m,'no unreviewed re-export/require execution');
  const dynamic=[...source.matchAll(/\bimport\s*\(([^)]*)\)/g)].map(match=>match[1]);
  assert.deepEqual(dynamic,file===MANUAL_CONTROLLER_FILES[0]?["'./h3-matched-resident-control.mjs'"]:[], 'only the explicit matched-mode dynamic import is reachable');
  return {file,staticImports:imports,dynamicImports:dynamic};
}

export function validateControllerClosure(entries,{controllerHead,expectedHead}){
  assert.equal(controllerHead,expectedHead,'controller must be the authorized current HEAD');
  assert.match(controllerHead,/^[a-f0-9]{40}$/);
  assert.deepEqual(entries.map(row=>row.path).sort(),[...COMMON_CONTROLLER_FILES,...MANUAL_CONTROLLER_FILES].sort(),'complete reviewed controller closure required');
  for(const row of entries){
    assert.equal(row.afterMode,'100644','regular Git source blobs only');
    assert.match(row.afterBlob,/^[a-f0-9]{40}$/);assert.equal(row.actualBlob,row.afterBlob,'controller working bytes differ from authorized Git source');
    assert.match(row.actualSha256,/^[a-f0-9]{64}$/);
    if(COMMON_CONTROLLER_FILES.includes(row.path)){
      assert.equal(row.beforeMode,row.afterMode);assert.equal(row.beforeBlob,row.afterBlob,'common harness/data differs from original CGPU');
    }else{
      assert.ok(MANUAL_CONTROLLER_FILES.includes(row.path));
      if(row.path===MANUAL_CONTROLLER_FILES[1]){assert.equal(row.beforeBlob,null,'new manual driver has no historical compiled claim');assert.equal(row.beforeMode,null);}
      else{assert.equal(row.beforeMode,'100644');assert.match(row.beforeBlob,/^[a-f0-9]{40}$/);}
    }
  }
  return {kind:'reviewed-js-execution-import-closure/v1',controllerHead,baseline:ORIGINAL.sceneworks,entries,
    scope:'17 unchanged common JS/data blobs plus2 exact current reviewed manual modules; NOT whole-controller compiled-source equivalence'};
}

export function validateSourceBindings(value){
  assert.equal(value.kind,'separate-controller-and-immutable-runtime/v1');
  for(const revision of Object.values(value.controllerSourcePair))assert.match(revision,/^[a-f0-9]{40}$/);
  assert.deepEqual(Object.keys(value.controllerSourcePair).sort(),['inference','sceneworks']);
  assert.equal(value.controllerCargoPin,value.controllerSourcePair.inference);
  assert.deepEqual(value.runtimeSourcePair,{sceneworks:ORIGINAL.sceneworks,inference:ORIGINAL.inference});
  assert.equal(value.runtimeCargoPin,ORIGINAL.inference);
  assert.deepEqual(value.runtimeTrees,RUNTIME_TREES);
  assert.equal(value.controllerClosure.controllerHead,value.controllerSourcePair.sceneworks);
  validateControllerClosure(value.controllerClosure.entries,{controllerHead:value.controllerClosure.controllerHead,expectedHead:value.controllerSourcePair.sceneworks});
  assert.equal(value.noCurrentRuntimeBuild,true);
  return value;
}

async function git(root,args){return (await exec('git',args,{cwd:root})).stdout.trim();}
async function treeEntry(root,revision,file,{optional=false}={}){
  const line=await git(root,['ls-tree',revision,'--',file]);
  if(optional&&!line)return {mode:null,blob:null};
  const match=/^(100644) blob ([a-f0-9]{40})\t(.+)$/.exec(line);
  assert.ok(match&&match[3]===file,'expected regular exact source path: '+file);
  return {mode:match[1],blob:match[2]};
}

export async function verifySourceBindings(p,controllerRoot,inferenceRoot,runtimeRoot,runtimeInferenceRoot,output){
  await sourcePair({...p,output},controllerRoot,inferenceRoot);
  const originalPlan={...p,output,sceneworks:ORIGINAL.sceneworks,inference:ORIGINAL.inference};
  await sourcePair(originalPlan,runtimeRoot,runtimeInferenceRoot);
  const roots=await Promise.all([controllerRoot,inferenceRoot,runtimeRoot,runtimeInferenceRoot].map(value=>realpath(value)));
  assert.equal(new Set(roots.map(value=>value.toLowerCase())).size,4,'controller and immutable runtime require four separate exact checkouts');
  const entries=[];
  for(const file of [...COMMON_CONTROLLER_FILES,...MANUAL_CONTROLLER_FILES]){
    const before=await treeEntry(controllerRoot,ORIGINAL.sceneworks,file,{optional:file===MANUAL_CONTROLLER_FILES[1]});
    const after=await treeEntry(controllerRoot,p.sceneworks,file);
    const actual=await realpath(path.join(controllerRoot,file));
    const relative=path.relative(roots[0],actual);assert.ok(relative&&!relative.startsWith('..')&&!path.isAbsolute(relative),'controller source escapes checkout');
    entries.push({path:file,beforeMode:before.mode,beforeBlob:before.blob,afterMode:after.mode,afterBlob:after.blob,
      actualBlob:await git(controllerRoot,['hash-object','--path',file,actual]),actualSha256:await fileSha256(actual)});
  }
  const imports=[];
  for(const file of MANUAL_CONTROLLER_FILES)imports.push(validateManualImports(file,await readFile(path.join(controllerRoot,file),'utf8')));
  const controllerClosure=validateControllerClosure(entries,{controllerHead:await git(controllerRoot,['rev-parse','HEAD']),expectedHead:p.sceneworks});
  controllerClosure.manualImportAssertions=imports;
  return validateSourceBindings({kind:'separate-controller-and-immutable-runtime/v1',
    controllerSourcePair:{sceneworks:p.sceneworks,inference:p.inference},
    controllerCargoPin:lockedInferenceRevision(await readFile(path.join(controllerRoot,'Cargo.lock'),'utf8')),
    runtimeSourcePair:{sceneworks:ORIGINAL.sceneworks,inference:ORIGINAL.inference},
    runtimeCargoPin:lockedInferenceRevision(await readFile(path.join(runtimeRoot,'Cargo.lock'),'utf8')),
    runtimeTrees:{sceneworks:await git(runtimeRoot,['rev-parse','HEAD^{tree}']),inference:await git(runtimeInferenceRoot,['rev-parse','HEAD^{tree}']),
      config:await git(runtimeRoot,['rev-parse','HEAD:config']),lock:await git(runtimeRoot,['rev-parse','HEAD:Cargo.lock'])},
    controllerClosure,noCurrentRuntimeBuild:true,
    checkoutRoles:{controller:roots[0],controllerLeaseEngine:roots[1],runtimeConfiguration:roots[2],runtimeEngineSource:roots[3]},
    meaning:'Current manual JS controller and current-engine lease only; original CGPU/M4 compiled EXE/config/source reused. Current Cargo/Rust/UI are neither compiled nor launched.'});
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
  assert.match(record.controllerSourcePair.sceneworks,/^[a-f0-9]{40}$/);assert.match(record.controllerSourcePair.inference,/^[a-f0-9]{40}$/);
  validateSourceBindings(record.sourceBindings);assert.deepEqual(record.sourceBindings.controllerSourcePair,record.controllerSourcePair);assert.deepEqual(record.sourceBindings.runtimeSourcePair,record.runtimeSourcePair);
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
    await save(path.join(this.dir,'prebuilt-reuse.json'),{kind:'immutable-original-build-reuse',runtimeSourcePair:{sceneworks:ORIGINAL.sceneworks,inference:ORIGINAL.inference},controllerSourcePair:this.controllerSourcePair,sourceBindings:this.sourceBindings,originalBuildReceiptSha256:ORIGINAL.buildReceiptSha256,binarySha256:this.binaryHash,meaning:'Original CGPU/M4 build reused read-only; no Cargo executed or new-head compile claimed'});
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
  assert.equal(process.env.H3_CONTROLLER_AUTHORIZED_PAIR,`${p.sceneworks}:${p.inference}`);
  assert.equal(process.env.H3_TERMINAL_AUTHORIZED_PAIR,`${ORIGINAL.sceneworks}:${ORIGINAL.inference}`);
  assert.equal(process.execArgv.length,0,'no unreviewed Node loaders/flags');
  for(const key of ['NODE_OPTIONS','NODE_PATH'])assert.ok(!process.env[key],'no external Node loaders/module search roots');
  // Fixed sibling checkouts cannot be overridden by the request or environment.
  const runtimeRoot=path.join(path.dirname(controllerRoot),'runtime-sceneworks');
  const runtimeInferenceRoot=path.join(path.dirname(controllerRoot),'runtime-inference');
  await mkdir(output);
  const bindings=await verifySourceBindings(p,controllerRoot,inferenceRoot,runtimeRoot,runtimeInferenceRoot,output);
  const runtimePlan={...p,output,sceneworks:ORIGINAL.sceneworks,inference:ORIGINAL.inference};
  await originalBuild();
  const license=await verifiedOriginal('services/workstation-veda/config/license-acknowledgments.json',ORIGINAL.licenseSha256);runtimePlan.licenseAcknowledgmentsPath=license;
  assert.equal(await fileSha256(p.requestPath),p.requestSha256);const request=anchorRequest(await load(p.requestPath),42,0);
  await save(path.join(output,'source-identity.json'),{kind:'matched-resident-control',...bindings,originalBinarySha256:ORIGINAL.binarySha256,originalBuildReceiptSha256:ORIGINAL.buildReceiptSha256,historicalLoadedDllHashes:'not recorded; new loaded-module hashes below are not retroactive historical attestations'});
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
  const service=new ImmutableOriginalService(runtimePlan,c,runtimeRoot,runtimeInferenceRoot,dir,{hardDeadlineMs:Date.now()+(p.timeoutSeconds+900)*1000});service.staged=staged;service.sourceBindings=bindings;service.controllerSourcePair={sceneworks:p.sceneworks,inference:p.inference};
  const record={kind:'matched-resident-control',completed:false,sourceBindings:bindings,controllerSourcePair:service.controllerSourcePair,runtimeSourcePair:{sceneworks:ORIGINAL.sceneworks,inference:ORIGINAL.inference},binarySha256:ORIGINAL.binarySha256,settings:request,scope:'One observational Resident/pool control; not feature/campaign/24GiB-fit or final-media numerical acceptance'};
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
  assert.deepEqual(await verifySourceBindings(p,controllerRoot,inferenceRoot,runtimeRoot,runtimeInferenceRoot,output),bindings,'source bindings changed during control');
  await save(path.join(output,'matched-resident-control.json'),{kind:'matched-resident-control',executionValidated:true,sourceBindings:bindings,controllerSourcePair:record.controllerSourcePair,runtimeSourcePair:record.runtimeSourcePair,jobId:record.job.id,comparisons,scope:record.scope,cleanupProof:'Separate existing physical-GPU lease receipt must additionally verify post_execution_verified'});
}
