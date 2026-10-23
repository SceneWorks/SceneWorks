// ONE review fix pass: DECLARED SYNTHETIC CPU fixtures. No build/helper/GPU/model execution.
import assert from 'node:assert/strict';
import test from 'node:test';
import { EventEmitter } from 'node:events';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp, mkdir, readFile, writeFile, realpath, rm, copyFile } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { loadProductTargets, targetRegistry, terminalCatalog, terminalPlan } from './flux3-image-klein-terminal-plan.mjs';
import { planHash } from './flux3-image-klein-terminal-budget.mjs';
import { snapshotPath, stageTerminalArtifacts, verifyTerminalMapping, artifactStagingPlan } from './flux3-image-klein-terminal-artifacts.mjs';
import { launchTerminalSession, guardHardware, fileHash, verifyResourceProducerHardware, assessForeignGraphics, gpuProcessMemoryScript, validateGpuProcessMemory, FOREIGN_GRAPHICS_PROCESS_LIMIT_BYTES, FOREIGN_GRAPHICS_TOTAL_LIMIT_BYTES } from './flux3-image-klein-terminal-session.mjs';
import { produceTerminalBuild, verifyTerminalBuild, verifyPreparationBuild, buildRecipe } from './flux3-image-klein-terminal-build.mjs';
import { nativeDatasetFingerprint, expectedNativeArtifacts, bindTrainingExpectation, verifyNativeContinuation, RNG_ALGORITHM } from './flux3-image-klein-terminal-training-proof.mjs';
import { assertFinishedPng, assertCheckpoint, trainingSequence, pngCrc32, sha256 } from './flux3-image-klein-terminal-driver.mjs';
import { deflateSync } from 'node:zlib';
const sources = await loadProductTargets(), catalog = terminalCatalog(sources), registry = targetRegistry(sources), source = 'a'.repeat(40), inference = 'b'.repeat(40), digest = 'd'.repeat(64);
async function temp(t) { const root = await mkdtemp(path.join(os.tmpdir(), 'sc26243-review-synthetic-')); t.after(() => rm(root, { recursive: true, force: true })); return root; }
async function inv(root, contents, repository, resolvedRevision) { await mkdir(root, { recursive: true }); const files = []; for (const [name, value] of Object.entries(contents)) { const bytes = Buffer.isBuffer(value) ? value : Buffer.from(typeof value === 'string' ? value : JSON.stringify(value)); await mkdir(path.dirname(path.join(root,name)), { recursive: true }); await writeFile(path.join(root,name),bytes); files.push({ path:name, bytes:bytes.length, sha256:sha256(bytes) }); } return { root, authority:'parent-approved-pinned-inventory', repository, resolvedRevision, files }; }
async function installReceipt(root, inventory, modelId) { const file=path.join(root, `${modelId}-genuine-declared-fixture-receipt.json`); const receipt={schemaVersion:2,modelId,repo:inventory.repository,snapshotRevision:inventory.resolvedRevision,variant:'bf16',jobId:'DECLARED_SYNTHETIC_DOWNLOAD_JOB',completedAt:'declared fixture time',manifestFiles:[],resolvedFiles:inventory.files.map(f=>f.path)}; await writeFile(file,JSON.stringify(receipt)); inventory.installationReceipt={root,path:file,sha256:await fileHash(file)}; return receipt; }
async function artifactFixture(t, backend='candle', tier='q4', modelId='flux3_klein') {
  const root=await temp(t), state=path.join(root,'state'), home=path.join(root,'hf'); await mkdir(path.join(state,'data'),{recursive:true});
  const target=(()=>{const original=registry.find(t=>t.backend===backend&&t.modelId===modelId);return {...original,model:structuredClone(original.model),downloads:structuredClone(original.downloads)};})(), primary=target.downloads.find(d=>d.variant==='bf16'), snapshot=snapshotPath(home,primary.repo,primary.revision);
  const config={quantization:null,text_encoder:{path:'text_encoder'},vae:{path:'ae.safetensors'}};
  const dense=await inv(snapshot, {'config.json':config,'manifest.json':'SYNTHETIC','model.safetensors':'SYNTHETIC DENSE','ae.safetensors':'SYNTHETIC AE','LICENSE.md':'FLUX Kommunity synthetic fixture','NOTICE.md':'synthetic notice','text_encoder/config.json':'{}','text_encoder/tokenizer.json':'{}','text_encoder/tokenizer_config.json':'{}','text_encoder/model.safetensors.index.json':{weight_map:{a:'model-00001-of-00002.safetensors',b:'model-00002-of-00002.safetensors'}},'text_encoder/model-00001-of-00002.safetensors':'SYNTHETIC TE1','text_encoder/model-00002-of-00002.safetensors':'SYNTHETIC TE2'}, primary.repo,primary.revision);
  await installReceipt(root,dense,modelId);
  const external = role => ({...dense,root:role==='text_encoder'?path.join(snapshot,'text_encoder'):snapshot,role,files:dense.files.filter(f=>role==='text_encoder'?f.path.startsWith('text_encoder/'):f.path==='ae.safetensors').map(f=>({...f,path:role==='text_encoder'?f.path.slice('text_encoder/'.length):f.path}))});
  const row=target.downloads.find(d=>d.variant===tier), marker={format:backend==='mlx'?'mlx-affine':tier==='q4'?'ggml-q4_1':'ggml-q8_0',bits:tier==='q4'?4:8,group_size:backend==='mlx'?64:32,algorithm_version:1};
  const packed=await inv(path.join(root,'external-packed'),{'config.json':{...config,quantization:marker},'model.safetensors':'DECLARED SYNTHETIC PACKED','LICENSE.md':'FLUX Kommunity synthetic fixture','NOTICE.md':'synthetic notice'},primary.repo,primary.revision);
  // Synthetic catalog pins are confined to this pure fixture; production registry is never edited.
  const weight=packed.files.find(f=>f.path==='model.safetensors'); row.localDerivation.weightsBytes=weight.bytes; row.localDerivation.weightsSha256=weight.sha256;
  const filesJson=files=>Object.fromEntries(files.map(f=>[f.path,{bytes:f.bytes,sha256:f.sha256}]));
  const receipt={format_version:1,source_repository:primary.repo,source_revision:primary.revision,source_artifacts:Object.fromEntries(dense.files.filter(f=>['config.json','model.safetensors'].includes(f.path)).map(f=>[f.path,f.sha256])),algorithm:marker.format,bits:marker.bits,group_size:marker.group_size,algorithm_version:1,package_files:filesJson(packed.files.filter(f=>f.path!=='model.safetensors')),components:[{role:'dit',placement:'owned',source_repository:primary.repo,source_revision:primary.revision,source_artifacts:{'model.safetensors':dense.files.find(f=>f.path==='model.safetensors').sha256},files:filesJson([weight])},...['text_encoder','autoencoder'].map(role=>({role,placement:'external',source_repository:primary.repo,source_revision:primary.revision,source_artifacts:Object.fromEntries(dense.files.filter(f=>role==='text_encoder'?f.path.startsWith('text_encoder/'):f.path==='ae.safetensors').map(f=>[f.path,f.sha256])),files:filesJson(dense.files.filter(f=>role==='text_encoder'?f.path.startsWith('text_encoder/'):f.path==='ae.safetensors'))}))]};
  const rfile=path.join(packed.root,'flux3-conversion-receipt.json'); await writeFile(rfile,JSON.stringify(receipt)); packed.files.push({path:'flux3-conversion-receipt.json',bytes:Buffer.byteLength(JSON.stringify(receipt)),sha256:await fileHash(rfile)}); packed.externalComponents=['text_encoder','autoencoder'].map(external);
  const cell=structuredClone(catalog.cases.find(c=>c.backend===backend&&c.modelId===modelId&&c.tier===tier)); assert(cell); const inputSpec={packages:{[`${modelId}:${backend}:bf16`]:dense,[cell.artifactKey]:packed}}, admission={backend,hfHome:home};
  return {root,state,home,target,dense,packed,cell,inputSpec,admission,identity:{state,workerId:'fixture-worker'}};
}

test('q4/q8 concrete launcher materializes once at exact worker path before any service; missing mapping fails closed', async t=>{
  for (const [backend,tier] of [['candle','q4'],['mlx','q8']]) {
    const f=await artifactFixture(t,backend,tier), a={...f.admission,outputRoot:f.root,binary:path.join(f.root,'never-executed.exe'),binarySha256:digest,sourceSha:source,inferenceMainSha:inference,timeoutSeconds:60,artifactInputs:f.inputSpec,artifactCells:[f.cell,f.cell],artifactTargets:[f.target],approvedPlan:{inputSpecSha256:{[backend]:planHash(f.inputSpec)}}}; let workerId; const children=[];
    const child=()=>{const c=new EventEmitter();Object.assign(c,{pid:700+children.length,exitCode:null,signalCode:null,stdin:{end(){c.exitCode=0;c.emit('exit',0,null);}},kill(){c.exitCode=0;c.emit('exit',0,null);}});children.push(c);return c;};
    let badWitness=false,witnessInvocations=0;
    const deps={fixtureOnly:true,skipPinReadForFixture:true,execWitness:async(binary,args,options)=>{witnessInvocations++;assert.equal(binary,a.binary);assert.equal(args[0],'--flux3-training-config-witness');assert.equal(options.env.CUDA_VISIBLE_DEVICES,'-1');assert.equal(options.env.SCENEWORKS_BACKEND_CANDLE_ENABLED,'false');assert.equal(options.env.SCENEWORKS_BACKEND_MLX_ENABLED,'false');assert.equal(options.env.SCENEWORKS_FLUX3_WITNESS_ROOT,session.identity.state);assert.equal(options.timeout,30000);const bytes=await readFile(args[1]);assert.deepEqual(JSON.parse(bytes),{declaredSynthetic:true});return{stdout:JSON.stringify({status:'ok',witness:{formatVersion:1,inputSha256:badWitness?digest:sha256(bytes)}})};},hash:async()=>digest,guard:async()=>({}),cp:async()=>{},open:async()=>({fd:42,close:async()=>{}}),readFile:async()=>'{"event":"api_listening","address":"127.0.0.1:43130"}',
      spawn:(binary,args,options)=>{assert(existsSync(path.join(options.env.SCENEWORKS_DATA_DIR,'models/derived',f.cell.modelId,tier,f.target.downloads.find(d=>d.variant===tier).localDerivation.conversion,'model.safetensors')), 'mapping must exist BEFORE the first API/worker spawn');workerId=options.env.SCENEWORKS_WORKER_ID;return child();},controller:()=>({verify:async()=>true,startWorker:async fn=>fn(),call:async()=>[{id:workerId,status:'idle',gpuId:backend==='candle'?'1':'mlx',capabilities:backend==='candle'?['gpu','candle']:['gpu']}],close(){}})};
    const session=await launchTerminalSession(a,deps);
    try {const mapped=session.artifactMapping.packages[f.cell.artifactKey], expected=path.join(session.identity.state,'data/models/derived',f.cell.modelId,tier,f.target.downloads.find(d=>d.variant===tier).localDerivation.conversion);assert.equal(await realpath(mapped.root),await realpath(expected));assert.equal(await fileHash(path.join(mapped.root,'model.safetensors')),f.packed.files.find(x=>x.path==='model.safetensors').sha256);assert.equal(session.artifactMapping.copies,f.packed.files.length);assert.equal(session.artifactMapping.copiedBytes,artifactStagingPlan([f.cell,f.cell],[f.target],f.inputSpec).totalCopyBytes); await verifyTerminalMapping(session,f.cell,f.packed);
      await assert.rejects(session.trainingWitness({declaredSynthetic:true}),/worker paused/);assert.equal(witnessInvocations,0);await session.pauseWorker();const witness=await session.trainingWitness({declaredSynthetic:true});assert.equal(witness.binarySha256,digest);badWitness=true;await assert.rejects(session.trainingWitness({declaredSynthetic:true}),/exact persisted plan bytes/);assert.equal(witnessInvocations,2);
      await rm(path.join(mapped.root,'model.safetensors')); await assert.rejects(verifyTerminalMapping(session,f.cell,f.packed));
    } finally {await session.close();}
    let spawned=0;await assert.rejects(launchTerminalSession({...a,artifactInputs:{packages:{}}},{...deps,spawn:()=>{spawned++;throw new Error('must not spawn');}}));assert.equal(spawned,0);
  }
});
test('component roots and native receipt inputs bind actual selected snapshot, not same repo/rev on another tree',async t=>{
 const f=await artifactFixture(t), mapping=await stageTerminalArtifacts({admission:f.admission,identity:f.identity,inputSpec:f.inputSpec,cells:[f.cell],registry:[f.target]},{diskFree:async()=>1e12}),s={identity:f.identity,artifactMapping:mapping}; await verifyTerminalMapping(s,f.cell,f.packed);
 const forged=structuredClone(f.packed);forged.externalComponents[0].root=path.join(f.root,'other-encoder');await mkdir(forged.externalComponents[0].root);for(const item of forged.externalComponents[0].files){await mkdir(path.dirname(path.join(forged.externalComponents[0].root,item.path)),{recursive:true});await copyFile(path.join(f.packed.externalComponents[0].root,item.path),path.join(forged.externalComponents[0].root,item.path));}
 mapping.packages[f.cell.artifactKey].inputInventorySha256=planHash(forged);await assert.rejects(verifyTerminalMapping(s,f.cell,forged),/actual worker resolver/);
 const wrong=structuredClone(f.inputSpec);wrong.packages[f.cell.artifactKey].externalComponents[1].root=path.join(f.root,'same-label-wrong-ae');await mkdir(wrong.packages[f.cell.artifactKey].externalComponents[1].root);await assert.rejects(stageTerminalArtifacts({admission:f.admission,identity:{...f.identity,state:await (async()=>{const p=path.join(f.root,'other-state');await mkdir(p);return p;})()},inputSpec:wrong,cells:[f.cell],registry:[f.target]},{diskFree:async()=>1e12}));
});

test('PUP inventory must be the exact worker snapshot and genuine receipt, not matching labels on a different tree',async t=>{
 const f=await artifactFixture(t);f.cell.options.enhancePrompt=true;const row=f.target.model.downloads.find(d=>d.componentId==='prompt_upsampler'),root=snapshotPath(f.home,row.repo,row.revision),pup=await inv(root,{'config.json':'{}','model.safetensors':'DECLARED SYNTHETIC PUP'},row.repo,row.revision);await installReceipt(f.root,pup,'flux3_image_klein_pup');f.packed.components={prompt_upsampler:pup};
 const mapping=await stageTerminalArtifacts({admission:f.admission,identity:f.identity,inputSpec:f.inputSpec,cells:[f.cell],registry:[f.target]},{diskFree:async()=>1e12}),session={identity:f.identity,artifactMapping:mapping};await verifyTerminalMapping(session,f.cell,f.packed);
 const wrong=structuredClone(f.packed);wrong.components.prompt_upsampler.root=path.join(f.root,'same-label-wrong-pup');await mkdir(wrong.components.prompt_upsampler.root);for(const item of pup.files)await copyFile(path.join(root,item.path),path.join(wrong.components.prompt_upsampler.root,item.path));mapping.packages[f.cell.artifactKey].inputInventorySha256=planHash(wrong);mapping.pup.inputInventorySha256=planHash(wrong.components.prompt_upsampler);await assert.rejects(verifyTerminalMapping(session,f.cell,wrong),/actual worker resolver/);
 mapping.packages[f.cell.artifactKey].inputInventorySha256=planHash(f.packed);mapping.pup.inputInventorySha256=planHash(pup);await writeFile(mapping.pup.install.marker,'{}');await assert.rejects(verifyTerminalMapping(session,f.cell,f.packed),/schemaVersion/);
});

test('owned build producer observes source/lock/resolved pins and stale receipt/current label cannot substitute old executable',async t=>{
 const root=await temp(t), src=path.join(root,'source'), output=path.join(root,'build-output'), targetDir=path.join(root,'warm-target');await mkdir(output); const pin=`x = { git = "https://github.com/SceneWorks/inference", rev = "${inference}" }\n`;
 for(const name of ['Cargo.toml','crates/sceneworks-worker/Cargo.toml','crates/sceneworks-memory-adapter/Cargo.toml']){await mkdir(path.dirname(path.join(src,name)),{recursive:true});await writeFile(path.join(src,name),pin);} const lock=`source = "git+https://github.com/SceneWorks/inference?rev=${inference}#${inference}"\n`;await writeFile(path.join(src,'Cargo.lock'),lock);
 const cargo=path.join(root,'cargo.exe'),rustc=path.join(root,'rustc.exe'),cl=path.join(root,'cl.exe'),nvcc=path.join(root,'nvcc.exe');await writeFile(nvcc,'DECLARED SYNTHETIC CUDA COMPILER');await writeFile(cargo,'DECLARED SYNTHETIC TOOL');await writeFile(rustc,'DECLARED SYNTHETIC COMPILER');await writeFile(cl,'DECLARED SYNTHETIC MSVC');
 const a={actualBuild:true,userBuildApprovalId:'SYNTHETIC_NOT_REAL_BUILD_APPROVAL',resourceAdmissionId:'fixture',sourceSha:source,inferenceMainSha:inference,backend:'candle',cargoExecutable:cargo,cargoExecutableSha256:await fileHash(cargo),rustcExecutable:rustc,rustcExecutableSha256:await fileHash(rustc),targetDir,buildOutputRoot:output,inferenceRepository:src,cargoLockSha256:sha256(lock),msvcVersion:'14.44',msvcCompilerExecutable:cl,msvcCompilerSha256:await fileHash(cl),cudaCompilerExecutable:nvcc,cudaCompilerSha256:await fileHash(nvcc),rustflags:'-C debuginfo=0'};
 let mixed=false,changed=false,stale=false,built=0,changedHead=false,headReads=0; const exec=async(command,args,options)=>{
  if(command==='git')return{stdout:args[0]==='rev-parse'?(changedHead&&++headReads>1?'e'.repeat(40):source):''}; if(command===rustc)return{stdout:'rustc synthetic\nhost: x86_64-pc-windows-msvc\n'}; if(command===cl)return{stdout:'Microsoft C/C++ Compiler Version 19.44.0000'};if(command===nvcc)return{stdout:'Cuda compilation tools, release 12.9, V12.9.86'}; assert.equal(options.env.CUDA_VISIBLE_DEVICES,'-1');assert.equal(options.env.CARGO_NET_OFFLINE,'true');
  if(args[0]==='metadata')return{stdout:JSON.stringify({packages:[{name:'gen-core',source:`git+https://github.com/SceneWorks/inference?rev=${inference}#${mixed?'c'.repeat(40):inference}`}]})};
  built++;const bin=args[args.indexOf('--bin')+1],binary=path.join(targetDir,`x86_64-pc-windows-msvc/release/${bin}.exe`);await mkdir(path.dirname(binary),{recursive:true});await writeFile(binary,'SYNTHETIC EXECUTABLE NEVER RUN');if(changed)await writeFile(path.join(src,'Cargo.lock'),`${lock}# changed\n`);
  return{stdout:[...(stale?[]:[JSON.stringify({reason:'compiler-artifact',target:{name:bin,kind:['bin']},executable:binary})]),JSON.stringify({reason:'build-finished',success:true})].join('\n'),stderr:''};
 };
 const produced=await produceTerminalBuild(a,{root:src,platform:'win32',exec,env:{PATH:'fixture'}}), transcript=JSON.parse(await readFile(path.join(path.dirname(produced.receiptPath),'producer-transcript.json'),'utf8'));
 const admission={sourceSha:source,inferenceMainSha:inference,backend:'candle',binary:produced.receipt.binary.path,binarySha256:produced.receipt.binary.sha256,binarySourceSha:source,binaryBuildReceipt:{root:output,path:produced.receiptPath,sha256:produced.receiptSha256},approvedPlan:{buildReceipts:{candle:{receiptSha256:produced.receiptSha256,transcriptSha256:produced.receipt.transcriptSha256,binarySha256:produced.receipt.binary.sha256,cargoLockSha256:a.cargoLockSha256,rustflagsSha256:sha256(a.rustflags),cargoExecutableSha256:a.cargoExecutableSha256,rustcExecutableSha256:a.rustcExecutableSha256,msvcCompilerSha256:a.msvcCompilerSha256,cudaCompilerSha256:a.cudaCompilerSha256}}}};
 const prepAdmission={...admission,approvedPlan:undefined,actualProvision:true,userProvisionApprovalId:'SYNTHETIC_CPU_PREP_APPROVAL',preparationBuildReceiptBinding:admission.approvedPlan.buildReceipts.candle};
 await verifyPreparationBuild(prepAdmission,{sourceRoot:src});await assert.rejects(verifyTerminalBuild(prepAdmission,{sourceRoot:src}),/producer receipt required/);
 for(const field of ['receiptSha256','transcriptSha256','binarySha256','cargoLockSha256','rustflagsSha256','cargoExecutableSha256','rustcExecutableSha256','msvcCompilerSha256','cudaCompilerSha256']){const bad=structuredClone(prepAdmission);bad.preparationBuildReceiptBinding[field]='0'.repeat(64);await assert.rejects(verifyPreparationBuild(bad,{sourceRoot:src}));}
 await verifyTerminalBuild(admission,{sourceRoot:src});const stdoutFile=path.join(path.dirname(produced.receiptPath),'build-stdout.log'),stdoutBytes=await readFile(stdoutFile);await writeFile(stdoutFile,'ALTERED DECLARED BUILD TRANSCRIPT');await assert.rejects(verifyTerminalBuild(admission,{sourceRoot:src}),/transcript bytes changed/);await writeFile(stdoutFile,stdoutBytes);await assert.rejects(verifyTerminalBuild({...admission,sourceSha:'e'.repeat(40),binarySourceSha:'e'.repeat(40)},{sourceRoot:src}));await writeFile(admission.binary,'OLD EXECUTABLE RELABELED CURRENT');await assert.rejects(verifyTerminalBuild(admission,{sourceRoot:src}));
 const resource=await produceTerminalBuild({...a,binaryRole:'resource-adapter'},{root:src,platform:'win32',exec,env:{PATH:'fixture'}}),resourceAdmission={...admission,binary:resource.receipt.binary.path,binarySha256:resource.receipt.binary.sha256,binaryBuildReceipt:{root:output,path:resource.receiptPath,sha256:resource.receiptSha256},approvedPlan:{resourceBuildReceipts:{candle:{...admission.approvedPlan.buildReceipts.candle,receiptSha256:resource.receiptSha256,transcriptSha256:resource.receipt.transcriptSha256,binarySha256:resource.receipt.binary.sha256}}}};
 await assert.rejects(verifyPreparationBuild({...resourceAdmission,approvedPlan:undefined,actualProvision:true,userProvisionApprovalId:'SYNTHETIC_CPU_PREP_APPROVAL',preparationBuildReceiptBinding:resourceAdmission.approvedPlan.resourceBuildReceipts.candle},{sourceRoot:src}),/role\/package/);
 await verifyTerminalBuild(resourceAdmission,{sourceRoot:src,role:'resource-adapter'});await assert.rejects(verifyTerminalBuild(resourceAdmission,{sourceRoot:src}),/producer receipt required/);
 await assert.rejects(verifyTerminalBuild({...resourceAdmission,binaryBuildReceipt:admission.binaryBuildReceipt,approvedPlan:{resourceBuildReceipts:admission.approvedPlan.buildReceipts}},{sourceRoot:src,role:'resource-adapter'}),/role\/package/);
 assert.equal(buildRecipe('candle','win32','resource-adapter').bin,'memory-candle-adapter');assert.deepEqual(buildRecipe('mlx','darwin','resource-adapter').features,['mlx']);assert.throws(()=>buildRecipe('candle','win32','arbitrary-command'));
 mixed=true;const before=built;await assert.rejects(produceTerminalBuild(a,{root:src,platform:'win32',exec,env:{}}),/mixed/);assert.equal(built,before);mixed=false;stale=true;await assert.rejects(produceTerminalBuild(a,{root:src,platform:'win32',exec,env:{}}),/ACTUAL executable/);stale=false;changed=true;await assert.rejects(produceTerminalBuild(a,{root:src,platform:'win32',exec,env:{}}),/changed during build/);assert.equal(transcript.recipe.jobs,2);
 await writeFile(path.join(src,'Cargo.lock'),lock);changed=false;changedHead=true;headReads=0;await assert.rejects(produceTerminalBuild(a,{root:src,platform:'win32',exec,env:{}}),/HEAD differs/);
});
test('resource native-boundary observation binds actual adapter parent PID, executable, build role, plan request and hardware (DECLARED MOCK)',async t=>{
 const root=await temp(t),binary=path.join(root,'never-executed-resource-adapter');await writeFile(binary,'DECLARED SYNTHETIC RESOURCE ADAPTER');const binarySha256=await fileHash(binary);
 for(const backend of ['candle','mlx']){
  const approval={schemaVersion:1,kind:'flux3-terminal-plan-approval',approvalId:'fixture',userPlanApprovalId:'DECLARED_SYNTHETIC_NOT_USER_APPROVAL',approvedBy:'fixture',sourceSha:source,inferenceMainSha:inference,ledgerId:'declared-resource-ledger',ledgerRoots:{mlx:path.join(root,'mlx-ledger'),candle:path.join(root,'candle-ledger')},inputSpecSha256:{mlx:digest,candle:digest},plan:terminalPlan(sources)},invocation=approval.plan.generationInvocations.find(i=>i.backend===backend&&i.kind==='resource_anchor');invocation.requestSha256=digest;
  const a={actualRun:true,backend,sourceSha:source,inferenceMainSha:inference,binary,binarySha256,binaryBuildReceipt:{sha256:'e'.repeat(64)},gpuUuid:'GPU-fixture',hostName:'nax-macos-2',runner:backend==='mlx'?'nax-macos-2':'declared-cuda-runner',approvedPlan:approval,approvedPlanSha256:planHash(approval)},context={adapterPid:300,caseId:invocation.caseId,invocationId:invocation.invocationId,requestSha256:digest,sourceSha:source,binarySha256,approvedPlanSha256:a.approvedPlanSha256};
  const producer={pid:300,ppid:100,command:binary,executable:binary,startedAt:'declared-fixed-start'},table=[producer,{pid:301,ppid:300,command:'node',startedAt:'declared-cli-start'}],proof=backend==='candle'?{physicalGpuId:'1',gpuUuid:a.gpuUuid,logicalCudaOrdinal:0,ownComputePids:[300]}:{runner:'nax-macos-2',host:'nax-macos-2.local',runtime:{platform:'darwin',arch:'arm64'},coordinatorLeaseId:'DECLARED_MOCK_LEASE',exclusivity:'coordinator-attested-and-process-monitored'};let guards=0;
  const deps={parentPid:300,controllerPid:301,validate:async()=>{},verifyBuild:async value=>{assert.equal(value.binary,binary);},processTable:async()=>structuredClone(table),guard:async(value,pids)=>{assert.deepEqual(pids,[300]);guards++;return proof;}};
  const observation=await verifyResourceProducerHardware(a,context,deps);assert.equal(guards,1);for(const key of ['adapterPid','caseId','invocationId','requestSha256','sourceSha','binarySha256','approvedPlanSha256'])assert.equal(observation[key],context[key]);assert.equal(observation.buildReceiptSha256,a.binaryBuildReceipt.sha256);assert.deepEqual(observation.proof,proof);
  for(const mutate of [c=>c.adapterPid=999,c=>c.caseId='wrong',c=>c.requestSha256='e'.repeat(64),c=>c.sourceSha='c'.repeat(40),c=>c.binarySha256=digest]){const bad=structuredClone(context);mutate(bad);await assert.rejects(verifyResourceProducerHardware(a,bad,deps));}
  await assert.rejects(verifyResourceProducerHardware(a,context,{...deps,verifyBuild:async()=>{throw new Error('wrong sealed build role');}}),/wrong sealed build role/);
  await assert.rejects(verifyResourceProducerHardware(a,context,{...deps,guard:async()=>backend==='candle'?{...proof,gpuUuid:'GPU-wrong',ownComputePids:[]}:{...proof,host:'primary.local'}}));
  await assert.rejects(verifyResourceProducerHardware(a,context,{...deps,guard:async()=>backend==='candle'?{...proof,ownComputePids:[999]}:{...proof,runtime:{platform:'darwin',arch:'x64'}}}));
  let polls=0;await assert.rejects(verifyResourceProducerHardware(a,context,{...deps,processTable:async()=>++polls===1?table:[{...producer,startedAt:'reused-pid'}]}),/changed identity/);
  await assert.rejects(verifyResourceProducerHardware(a,context,{...deps,processTable:async()=>[{...producer,executable:path.join(root,'foreign.exe')},table[1]]}));
 }
});

test('GPU1 guard: bounded measured desktop C+G passes, pure compute/over-limit/unmeasurable refuse, foreign runners refuse (DECLARED MOCK)',async()=>{
 const a={backend:'candle',gpuUuid:'GPU-abcd'},own=[21],table=[{pid:20,ppid:10,command:'node.exe'},{pid:10,ppid:0,command:'Runner.Worker.exe'}],MiB=1024**2,LUID='0xdeadbeef_0x00000001';
 const startedAt='2000-01-01T00:00:00.0000000Z';
 const processTable=()=>[...table,...procs.filter(([pid])=>!own.includes(pid)).map(([pid])=>({pid,ppid:1,command:'synthetic-desktop.exe',startedAt}))];
 let procs,memory,luid,memoryQueries;
 const exec=async(cmd,args)=>{const script=args.at(-1);
  if(cmd==='powershell.exe'&&script.includes('cuDeviceGetLuid'))return{stdout:`${a.gpuUuid}${luid?` luid=${luid}`:''}`};
  if(cmd==='powershell.exe'&&script.includes('GPU Process Memory')){memoryQueries.push(script);return{stdout:JSON.stringify({samples:[...memory].map(([pid,cookedBytes])=>({pid,luid:LUID,physicalNode:0,status:0,cookedBytes})),identities:procs.filter(([pid,type])=>type==='C+G'&&!own.includes(pid)).map(([pid])=>({pid,startedAt})),errors:[]})};}
  if(args.includes('--query-gpu=index,uuid'))return{stdout:`1,${a.gpuUuid}`};
  if(args.includes('-x'))return{stdout:`<uuid>${a.gpuUuid}</uuid><processes>${procs.map(([pid,type])=>`<process_info><pid>${pid}</pid><type>${type}</type></process_info>`).join('')}</processes>`};
  return{stdout:procs.filter(([,type])=>type!=='G').map(([pid])=>`${pid},${a.gpuUuid}`).join('\n')};};
 const deps={exec,platform:'win32',controllerPid:20,processTable:async()=>processTable()};
 const reset=(p,m,l=LUID)=>{procs=p;memory=new Map(m);luid=l;memoryQueries=[];};
 // The real desktop shape: compositor + shell apps as C+G, measured on the CUDA-reported LUID.
 reset([[99,'C+G'],[98,'C+G'],[97,'G']],[[99,900*MiB],[98,100*MiB],[12345,5000*MiB]]);
 const proof=await guardHardware(a,own,deps);
 const {counterProof,...graphics}=proof.foreignGraphics;
 assert.deepEqual(graphics,{luid:LUID,processes:2,totalBytes:1000*MiB,maxProcessBytes:900*MiB,processLimitBytes:1024*MiB,totalLimitBytes:3*1024*MiB},'measured foreign total is recorded, never hidden');
 assert.deepEqual(counterProof.requiredProcesses,[{pid:99,startedAt},{pid:98,startedAt}]);
 assert.deepEqual([...validateGpuProcessMemory(counterProof,LUID,counterProof.requiredProcesses)],[[99,900*MiB],[98,100*MiB]]);
 assert(memoryQueries.length===1&&memoryQueries[0].includes(`luid_${LUID}_phys_`),'memory comes from the GPU1 LUID CUDA itself reported');
 reset([[97,'G']],[]);assert.equal((await guardHardware(a,own,deps)).foreignGraphics.totalBytes,0);assert.equal(memoryQueries.length,0,'graphics-only needs no measurement');
 reset([[99,'C+G']],[[99,1024*MiB+1]]);await assert.rejects(guardHardware(a,own,deps),/uses 1025 MiB of GPU1 \(limit 1024 MiB\)/);
 reset([[99,'C+G'],[98,'C+G'],[96,'C+G'],[95,'C+G']],[[99,900*MiB],[98,900*MiB],[96,900*MiB],[95,900*MiB]]);await assert.rejects(guardHardware(a,own,deps),/in total \(limit 3072 MiB\)/);
 reset([[99,'C']],[[99,1]]);await assert.rejects(guardHardware(a,own,deps),/foreign pure-compute/,'pure compute refuses at any size');
 reset([[99,'C+G']],[]);await assert.rejects(guardHardware(a,own,deps),/required foreign GPU counter sample missing/);
 reset([[99,'C+G']],[[99,MiB]],null);await assert.rejects(guardHardware(a,own,deps),/no adapter LUID/);
 reset([[99,'C+G']],[[99,MiB]]);await assert.rejects(guardHardware(a,own,{...deps,processTable:async()=>[...table,{pid:11,ppid:0,command:'Runner.Worker.exe'}]}),/foreign Runner.Worker/);
 // An unidentified compute-app type on GPU1 is not presumed desktop.
 reset([[99,'C+G']],[[99,MiB]]);await assert.rejects(guardHardware(a,own,{...deps,exec:async(c,args)=>args.includes('-x')?{stdout:`<uuid>${a.gpuUuid}</uuid><processes></processes>`}:exec(c,args)}),/pure compute or unidentified type/);
});
const counterFixture=JSON.parse(await readFile(new URL('./fixtures/flux3-gpu-counter-normalized-synthetic.json',import.meta.url),'utf8'));
const counterLuid=counterFixture.luid, counterRequired=counterFixture.requiredProcesses;
const knownCounterError={category:'InvalidResult',code:'CounterApiError'};
function counterObservation(){return structuredClone(counterFixture.collections[0]);}
function unrequiredSample(observation){return observation.samples.find(s=>!counterRequired.some(p=>p.pid===s.pid));}

test('normalized counter fixture retains two independent all-valid 29-row sets, not hardware evidence',()=>{
 assert.equal(counterFixture.declaredSynthetic,true);assert.equal(counterFixture.hardwareEvidence,false);
 assert.equal(counterFixture.collections.length,2);assert.equal(counterRequired.length,24);
 assert.equal(counterFixture.collections.flatMap(c=>c.samples).length,58);
 for(const observation of counterFixture.collections){
  assert.equal(observation.samples.length,29);assert.equal(observation.identities.length,24);
  assert(observation.samples.every(s=>s.status===0&&!('alive' in s)));
  assert.deepEqual(observation.identities,counterRequired);
  const memory=validateGpuProcessMemory(observation,counterLuid,counterRequired);
  assert.equal(memory.size,24);
  for(const required of counterRequired)assert.equal(memory.get(required.pid),observation.samples.find(s=>s.pid===required.pid).cookedBytes);
 }
});
test('counter validator accepts status 0/1, sums distinct physical nodes, and excludes valid unrequired rows',()=>{
 const observation=counterObservation(),first=observation.samples[0];first.status=1;
 observation.samples.push({...first,physicalNode:1,cookedBytes:8192});
 const memory=validateGpuProcessMemory(observation,counterLuid.toUpperCase().replaceAll('0X','0x'),counterRequired);
 assert.equal(memory.get(first.pid),first.cookedBytes+8192);assert.equal(memory.size,24);
 assert(!memory.has(unrequiredSample(observation).pid));
 first.cookedBytes=0;observation.samples.pop();assert.equal(validateGpuProcessMemory(observation,counterLuid,counterRequired).get(first.pid),0);
});
test('counter partial results require recognized errors explained only by invalid unrequired instances',()=>{
 for(const status of [0x800007d1,0xc0000bba]){
  const observation=counterObservation(),unrequired=unrequiredSample(observation);
  // Whether an unrequired instance is owned or has exited is deliberately NOT an identity assertion.
  unrequired.status=status;delete unrequired.cookedBytes;observation.errors=[{...knownCounterError}];
  assert.equal(validateGpuProcessMemory(observation,counterLuid,counterRequired).size,24);
  observation.errors=[];assert.throws(()=>validateGpuProcessMemory(observation,counterLuid,counterRequired),/unexplained/);
  observation.errors=[{...knownCounterError}];observation.samples[0].status=status;
  assert.throws(()=>validateGpuProcessMemory(observation,counterLuid,counterRequired),/required foreign GPU process has invalid/);
 }
 const observation=counterObservation();observation.errors=[{...knownCounterError}];
 assert.throws(()=>validateGpuProcessMemory(observation,counterLuid,counterRequired),/unexplained/);
});
test('counter validator rejects missing, ambiguous, reused and unproved required identities',()=>{
 for(const [label,mutate] of [
  ['missing identity',o=>o.identities.shift()],['reused PID',o=>o.identities[0].startedAt='2000-01-02T00:00:00.0000000Z'],
  ['duplicate identity',o=>o.identities.push({...o.identities[0]})],['unexpected identity',o=>o.identities.push({pid:999999,startedAt:counterRequired[0].startedAt})],
  ['missing start',o=>delete o.identities[0].startedAt],['missing required sample',o=>o.samples.shift()]
 ]){const observation=counterObservation();mutate(observation);assert.throws(()=>validateGpuProcessMemory(observation,counterLuid,counterRequired),undefined,label);}
 for(const required of [[...counterRequired,counterRequired[0]],counterRequired.map((p,i)=>i?p:{...p,startedAt:undefined}),counterRequired.map((p,i)=>i?p:{...p,startedAt:'not a date'})]){
  assert.throws(()=>validateGpuProcessMemory(counterObservation(),counterLuid,required));
 }
});
test('counter validator fails closed on malformed samples and provider/permission errors',()=>{
 const mutations=[
  ['wrong LUID',s=>s.luid='0xdeadbeef_0x00000002'],['missing LUID',s=>delete s.luid],
  ['negative physical node',s=>s.physicalNode=-1],['fractional physical node',s=>s.physicalNode=0.5],['missing physical node',s=>delete s.physicalNode],
  ['invalid PID',s=>s.pid=0],['nonintegral PID',s=>s.pid=1.5],
  ['unknown status',s=>s.status=2],['negative status',s=>s.status=-1],['oversized status',s=>s.status=0x100000000],['missing status',s=>delete s.status],['string status',s=>s.status='0'],
  ['NaN bytes',s=>s.cookedBytes=NaN],['infinite bytes',s=>s.cookedBytes=Infinity],['negative bytes',s=>s.cookedBytes=-1],['missing bytes',s=>delete s.cookedBytes],['string bytes',s=>s.cookedBytes='4096']
 ];
 for(const [label,mutate] of mutations)for(const required of [true,false]){
  const observation=counterObservation();mutate(required?observation.samples[0]:unrequiredSample(observation));
  observation.errors=required?[]:[{...knownCounterError}];
  assert.throws(()=>validateGpuProcessMemory(observation,counterLuid,counterRequired),undefined,`${label}; required=${required}`);
 }
 for(const value of [0.5,Number.MAX_SAFE_INTEGER+1]){const observation=counterObservation();observation.samples[0].cookedBytes=value;assert.throws(()=>validateGpuProcessMemory(observation,counterLuid,counterRequired));}
 const duplicate=counterObservation();duplicate.samples.push({...duplicate.samples[0]});assert.throws(()=>validateGpuProcessMemory(duplicate,counterLuid,counterRequired),/duplicate/);
 for(const error of [{category:'PermissionDenied',code:'CounterApiError'},{category:'InvalidResult',code:'UNKNOWN_COUNTER_PROVIDER_ERROR'},{category:'NotSpecified',code:'CounterApiError'},null]){
  const observation=counterObservation();unrequiredSample(observation).status=0xc0000bba;observation.errors=[error];
  assert.throws(()=>validateGpuProcessMemory(observation,counterLuid,counterRequired),/provider\/permission/);
 }
 for(const field of ['samples','identities','errors']){const observation=counterObservation();delete observation[field];assert.throws(()=>validateGpuProcessMemory(observation,counterLuid,counterRequired),/structured/);}
});

// Execute the actual collector text in Windows PowerShell, but shadow BOTH hardware-facing
// cmdlets before it is evaluated. These scripted functions are test-only; no real counters,
// CIM, CUDA, GPU, API, provider or model is queried, nor is the production query rewritten.
async function scriptedCounterCollector(observation,required=counterRequired){
 const payload=Buffer.from(JSON.stringify(observation)).toString('base64');
 const prefix=`$ProgressPreference='SilentlyContinue'; $fixture=([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${payload}')) | ConvertFrom-Json);
function Get-Counter {
 [CmdletBinding()] param([string]$Counter,[int]$MaxSamples)
 if ($Counter -cne '\\GPU Process Memory(*luid_${counterLuid}_phys_*)\\Dedicated Usage' -or $MaxSamples -ne 1) { throw 'unexpected synthetic query' };
 foreach ($e in $fixture.errors) {
  $category=[System.Management.Automation.ErrorCategory]$e.category;
  $id=if ($e.code -eq 'CounterApiError') {'CounterApiError'} else {'SyntheticUnknown'};
  Write-Error -Message 'SYNTHETIC_PRIVATE_RAW_ERROR_MUST_NOT_LEAK' -Category $category -ErrorId $id;
 };
 $rows=@($fixture.samples | ForEach-Object {
  $value=if ($null -eq $_.cookedBytes) {[double]::NaN} else {[double]$_.cookedBytes};
  [pscustomobject]@{InstanceName=('pid_{0}_luid_{1}_phys_{2}' -f $_.pid,$_.luid,$_.physicalNode);Status=$_.status;CookedValue=$value}
 });
 [pscustomobject]@{CounterSamples=$rows}
}
function Get-CimInstance {
 [CmdletBinding()] param([string]$ClassName)
 if ($ClassName -cne 'Win32_Process') { throw 'unexpected synthetic identity query' };
 $fixture.identities | ForEach-Object { [pscustomobject]@{ProcessId=$_.pid;CreationDate=[datetime]::Parse($_.startedAt).ToUniversalTime()} }
}
`;
 const script=prefix+gpuProcessMemoryScript(counterLuid,required);
 assert(!script.includes('cuDeviceGetLuid'));
 const result=await promisify(execFile)('powershell.exe',['-NoProfile','-NonInteractive','-EncodedCommand',Buffer.from(script,'utf16le').toString('base64')],{timeout:30000,maxBuffer:1024*1024,env:{...process.env,CUDA_VISIBLE_DEVICES:'-1'}});
 assert(!result.stdout.includes('SYNTHETIC_PRIVATE_RAW_ERROR_MUST_NOT_LEAK'));
 assert(!result.stderr.includes('SYNTHETIC_PRIVATE_RAW_ERROR_MUST_NOT_LEAK'));
 assert.equal(result.stderr.trim(),'');return JSON.parse(result.stdout.replace(/^﻿/,''));
}
test('Windows PowerShell collector executes only scripted counter/CIM fixtures and captures sanitized errors',{skip:process.platform!=='win32'},async()=>{
 for(const source of counterFixture.collections){
  const observed=await scriptedCounterCollector(source);assert.deepEqual(observed,source);
  assert.equal(validateGpuProcessMemory(observed,counterLuid,counterRequired).size,24);
 }
 for(const status of [0x800007d1,0xc0000bba]){
  const source=counterObservation();source.samples[0].status=1;
  const row=unrequiredSample(source);row.status=status;delete row.cookedBytes;source.errors=[{...knownCounterError}];
  const observed=await scriptedCounterCollector(source);assert.deepEqual(observed,source);
  assert.deepEqual(Object.keys(observed.errors[0]).sort(),['category','code']);
  assert.equal(validateGpuProcessMemory(observed,counterLuid,counterRequired).size,24);
 }
 for(const error of [{category:'PermissionDenied',code:'CounterApiError'},{category:'InvalidResult',code:'UNKNOWN_COUNTER_PROVIDER_ERROR'}]){
  const source=counterObservation();unrequiredSample(source).status=0xc0000bba;delete unrequiredSample(source).cookedBytes;source.errors=[error];
  const observed=await scriptedCounterCollector(source);assert.deepEqual(observed.errors,[error]);
  assert.throws(()=>validateGpuProcessMemory(observed,counterLuid,counterRequired),/provider\/permission/);
 }
 const source=counterObservation();source.samples[0].cookedBytes=-1;
 const observed=await scriptedCounterCollector(source);assert(!('cookedBytes' in observed.samples[0]));
 assert.throws(()=>validateGpuProcessMemory(observed,counterLuid,counterRequired),/invalid counter data/);
 // Singleton arrays must remain arrays under PS5 as well as the real-shape multirow fixture.
 const singleton={samples:[counterObservation().samples[0]],identities:[counterRequired[0]],errors:[]};
 assert.deepEqual(await scriptedCounterCollector(singleton,[counterRequired[0]]),singleton);
});

test('foreign desktop thresholds are exactly 1 GiB per process and 3 GiB in total (mutation checks)',()=>{
 const MiB=1024**2;assert.equal(FOREIGN_GRAPHICS_PROCESS_LIMIT_BYTES,1024*MiB);assert.equal(FOREIGN_GRAPHICS_TOTAL_LIMIT_BYTES,3*1024*MiB);
 assert.equal(assessForeignGraphics([1],new Map([[1,1024*MiB]])).totalBytes,1024*MiB,'the limit itself is allowed');
 assert.throws(()=>assessForeignGraphics([1],new Map([[1,1024*MiB+1]])),/limit 1024 MiB/,'one byte over refuses');
 const three=new Map([[1,1024*MiB],[2,1024*MiB],[3,1024*MiB]]);assert.equal(assessForeignGraphics([1,2,3],three).totalBytes,3*1024*MiB);
 assert.throws(()=>assessForeignGraphics([1,2,3,4],new Map([...three,[4,1]])),/in total/);
 // Mutated (tighter or looser) thresholds change the verdict, so the constants are load-bearing.
 assert.throws(()=>assessForeignGraphics([1],new Map([[1,900*MiB]]),{processLimit:512*MiB}),/limit 512 MiB/);
 assert.throws(()=>assessForeignGraphics([1,2],new Map([[1,900*MiB],[2,900*MiB]]),{totalLimit:1024*MiB}),/in total/);
 assert.equal(assessForeignGraphics([1],new Map([[1,2047*MiB]]),{processLimit:2048*MiB}).processes,1);
 assert.throws(()=>assessForeignGraphics([1],new Map([[1,-1]])),/unmeasurable/);assert.throws(()=>assessForeignGraphics([1],new Map([[1,Number.NaN]])),/unmeasurable/);
});
test('secondary Mac cannot infer exclusivity from labels: hashed coordinator lease, ancestor runner and stable process snapshot monitored',async t=>{
 const root=await temp(t),a={backend:'mlx',runner:'nax-macos-2',runnerTarget:'secondary',runnerLabel:'rw-mage',hostName:'nax-macos-2',admissionId:'fixture',sourceSha:source,inferenceMainSha:inference,runnerAccount:'fixture-account',timeoutSeconds:60};
 const table=[{pid:20,ppid:10,command:'node',startedAt:'controller-start'},{pid:10,ppid:1,command:'Runner.Worker',startedAt:'runner-start'},{pid:1,ppid:0,command:'launchd',startedAt:'boot'},{pid:30,ppid:1,command:'WindowServer',startedAt:'desktop-start'}];
 const lease={kind:'flux3-secondary-mac-exclusive-workload-lease',coordinatorApprovalId:'DECLARED_SYNTHETIC',approvedBy:'fixture',exclusiveGpuWorkload:true,noForeignGpuWorkload:true,admissionId:a.admissionId,sourceSha:source,inferenceMainSha:inference,runnerAccount:a.runnerAccount,hostName:a.hostName,observedAt:new Date(Date.now()-1000).toISOString(),expiresAt:new Date(Date.now()+120000).toISOString(),ancestorRunnerPids:[10],baselineProcesses:table};
 const file=path.join(root,'lease.json');await writeFile(file,JSON.stringify(lease));a.macWorkloadLease={root,path:file,sha256:await fileHash(file)};
 const deps={platform:'darwin',arch:'arm64',env:{RUNNER_NAME:'nax-macos-2'},controllerPid:20,processTable:async()=>table,exec:async()=>({stdout:'nax-macos-2.local'})};
 await guardHardware(a,[21],deps);await guardHardware(a,[21],{...deps,processTable:async()=>[...table,{pid:21,ppid:20,command:'owned-worker',startedAt:'new-owned'}]});
 await assert.rejects(guardHardware(a,[21],{...deps,processTable:async()=>[...table,{pid:99,ppid:1,command:'foreign-metal-task',startedAt:'new-foreign'}]}),/foreign process invalidates/);
 await assert.rejects(guardHardware({...a,macWorkloadLease:{...a.macWorkloadLease,sha256:digest}},[],deps),/lease bytes changed/);
 await assert.rejects(guardHardware({...a,macWorkloadLease:undefined},[],deps),/attestation lease/);
});
function png(width,height,value=1) {const chunk=(name,data)=>{const size=Buffer.alloc(4),crc=Buffer.alloc(4),bytes=Buffer.concat([Buffer.from(name),data]);size.writeUInt32BE(data.length);crc.writeUInt32BE(pngCrc32(bytes));return Buffer.concat([size,bytes,crc]);};const header=Buffer.alloc(13);header.writeUInt32BE(width);header.writeUInt32BE(height,4);header[8]=8;header[9]=2;const pixels=Buffer.alloc((width*3+1)*height);pixels[1]=value;return Buffer.concat([Buffer.from('89504e470d0a1a0a','hex'),chunk('IHDR',header),chunk('IDAT',deflateSync(pixels)),chunk('IEND',Buffer.alloc(0))]);}
async function nativeProofFixture(t) {
 const f=await artifactFixture(t), target=registry.find(t=>t.backend==='candle'&&t.modelId==='flux3_klein_base'),row=target.downloads.find(d=>d.variant==='bf16'),root=snapshotPath(f.home,row.repo,row.revision),contents={};for(const item of f.dense.files)contents[item.path]=await readFile(path.join(f.dense.root,item.path));const inventory=await inv(root,contents,row.repo,row.revision),dense={root,inventory};
 const data=path.join(f.state,'dataset');await mkdir(data);const image=path.join(data,'target.png'),bytes=png(16,16);await writeFile(image,bytes);const input={assetId:'owned-input',sha256:sha256(bytes),decodedContentSha256:assertFinishedPng(bytes,16,16).decodedContentSha256};
 const targets=JSON.parse(await readFile(new URL('../tests/fixtures/rust_migration_contracts/training/target-registry.json',import.meta.url),'utf8')),trainingTarget=targets.targets.find(t=>t.id==='flux3_klein_lora'),config={...structuredClone(trainingTarget.defaults),steps:4,saveEvery:1,rank:4,alpha:4,resolution:512};config.advanced.resume=false;config.advanced.sampleEvery=0;
 const outputDir=path.join(f.state,'training');await mkdir(outputDir);const plan={schemaVersion:1,planVersion:1,jobId:'train',target:{targetId:'flux3_klein_lora',kernel:'flux3_klein_lora',family:'flux3',baseModel:'flux3_klein_base',baseModelRepo:row.repo,baseModelPath:root},dataset:{datasetId:'dataset',datasetVersion:1,rootPath:data,items:[{imagePath:image,caption:'declared synthetic caption'}]},config,output:{loraId:'trained',fileName:'lora.safetensors',outputDir,triggerWords:[]}};
 const execution={sourceSha:source,inferenceMainSha:inference,binarySha256:digest,buildReceiptSha256:'e'.repeat(64)},factors={'fixture.lora_A.weight':[4,1],'fixture.lora_B.weight':[1,4]},configuration={optimizer:'adamw',learning_rate:'0.0001',rank:'4',alpha:'4.0',scaling:'alpha/rank',precision:'bf16',backend:'candle-cuda;os=windows;arch=x86_64',scheduler:'Constant;warmup=0;cycles=0.5;power=1',accumulation:'1',accumulation_scaling:'loss/configured_N_before_backward; epoch-tail flush',rng:RNG_ALGORITHM,config_sha256:digest};
 const witness=p=>({formatVersion:1,inputSha256:planHash(p),configSha256:digest,configuration,factors,activeFactorKeys:Object.keys(factors),trainingReference:'7ecb1e1b4fe500be7fb7381a7bb79a24d7fa3941',rngAlgorithm:RNG_ALGORITHM,baseRepository:row.repo,baseRevision:row.revision,rank:4,alpha:4,seed:config.seed,accumulation:1,resolution:512,steps:4,...execution});
 const session={identity:{state:f.state,workerId:'w'},execution,artifactMapping:{snapshots:{flux3_klein_base:dense}}},items=[{target:input,caption:'declared synthetic caption'}];
 const expected=await bindTrainingExpectation(session,plan,config,items,witness(plan),assertFinishedPng);
 return {...f,target,trainingTarget,targets,dense,input,plan,session,items,witness,expected,outputDir};
}
function nativeContainer(expected,step,backend='candle') {
 const tensors={},data=[],manifest={},order=Object.keys(expected.factors).sort(),active=new Set(expected.activeFactorKeys);let offset=0;
 for(const [i,name]of order.entries())for(const key of [`factor/${name}`,...(backend==='candle'?[`optim.${i}.m`,`optim.${i}.v`]:active.has(name)?[`moment_first/${name}`,`moment_second/${name}`]:[])]){const shape=expected.factors[name],bytes=Buffer.alloc(shape.reduce((n,s)=>n*s,1)*4);bytes.writeFloatLE(key.startsWith('factor/')?(!active.has(name)?0.125:step===1?0.25:0.5):step/100);tensors[key]={dtype:'F32',shape,data_offsets:[offset,offset+bytes.length]};manifest[key]={dtype:'F32',shape,sha256:sha256(bytes)};offset+=bytes.length;data.push(bytes);}
 const seed=Buffer.alloc(8),counter=Buffer.alloc(8);seed.writeBigUInt64LE(BigInt(expected.seed));counter.writeBigUInt64LE(BigInt(step));
 const identity={schema:1,training_reference:expected.trainingReference,base_repo:expected.baseRepository,base_revision:expected.baseRevision,artifacts:expected.artifacts,dataset:expected.dataset,preprocessing:'f'.repeat(64),configuration:{...expected.configuration,recipe_sha256:'f'.repeat(64)},factors:expected.factors};
 const cursor={schema:1,micro_step:step,update_step:step,accumulation_count:0,sampler_epoch:step-1,sampler_cursor:1,rng_state:[...Buffer.concat([Buffer.from(RNG_ALGORITHM),Buffer.from([0]),seed,counter])],last_loss_bits:1065353216};
 const metadata={optimizer:'adamw',optimizer_config:'adamw;lr=0.0001;wd=0.0001;b1=0.9;b2=0.999;eps=0.00000001','optim.step':String(step),format:'flux3-native-training-resume-v1',identity:JSON.stringify(identity),cursor:JSON.stringify(cursor),policy:expected.policy,gain_bits:'3ff0000000000000',pending_keys:'[]',optimizer_order:JSON.stringify(order),tensor_manifest:JSON.stringify(manifest)};
 if(backend==='mlx'){metadata.format='flux3-mlx-training-resume-v1';delete metadata.optimizer_order;delete metadata.optimizer;delete metadata['optim.step'];metadata.optimizer_config='adamw;lr=38d1b717;wd=00000000;synthetic';metadata.optimizer_steps=JSON.stringify(Object.fromEntries([...active].map(key=>[key,step])));}
 const raw=Buffer.from(JSON.stringify({__metadata__:metadata,...tensors})),header=Buffer.alloc(Math.ceil(raw.length/8)*8,0x20);raw.copy(header);const prefix=Buffer.alloc(8);prefix.writeBigUInt64LE(BigInt(header.length));return Buffer.concat([prefix,header,...data]);
}
test('native immutable identity binds real plan bytes, staged dataset paths/content, source inventories, options/factors and exact RNG/pending policy',async t=>{
 const f=await nativeProofFixture(t), first=path.join(f.outputDir,'first.safetensors'),last=path.join(f.outputDir,'last.safetensors'),cell=catalog.cases.find(c=>c.backend==='candle'&&c.modelId==='flux3_klein_base'&&c.kind==='training');await writeFile(first,nativeContainer(f.expected,1));await writeFile(last,nativeContainer(f.expected,4));const initial=await assertCheckpoint(first,cell,1),final=await assertCheckpoint(last,cell,4);verifyNativeContinuation(initial,f.expected);verifyNativeContinuation(final,f.expected,initial);
 for(const mutate of [s=>s.identity.dataset=digest,s=>s.identity.configuration.config_sha256='e'.repeat(64),s=>s.identity.configuration.learning_rate='1.0',s=>s.identity.factors['fixture.lora_A.weight']=[8,1],s=>s.identity.artifacts['text_encoder/shard0']=digest,s=>s.cursor.rng_state[0]^=1,s=>s.cursor.accumulation_count=1,s=>s.pendingKeys=['fixture.lora_A.weight'],s=>s.policy='micro_steps=8;accumulation=1;sampler_len=1',s=>s.cursor.sampler_epoch=0,s=>s.optimizerConfig='adamw;lr=changed']){const bad=structuredClone(final);mutate(bad);assert.throws(()=>verifyNativeContinuation(bad,f.expected,initial));}
 await assert.rejects(bindTrainingExpectation(f.session,f.plan,f.plan.config,f.items,{...f.witness(f.plan),inputSha256:digest},assertFinishedPng),/persisted plan bytes/);
 await assert.rejects(bindTrainingExpectation(f.session,f.plan,f.plan.config,f.items,{...f.witness(f.plan),binarySha256:'e'.repeat(64)},assertFinishedPng),/different build/);
 const altered=structuredClone(f.plan);altered.config.advanced.flux3.caption_dropout=0.5;await assert.rejects(bindTrainingExpectation(f.session,altered,f.plan.config,f.items,f.witness(altered),assertFinishedPng),/submitted config/);
 await writeFile(f.plan.dataset.items[0].imagePath,png(16,16,2));await assert.rejects(nativeDatasetFingerprint(f.plan,f.state,f.items,assertFinishedPng));
});
test('Candle checkpoint must carry its ACTUAL native optim.step/config, never an inferred cursor alias',async t=>{
 const f=await nativeProofFixture(t),cell=catalog.cases.find(c=>c.backend==='candle'&&c.modelId==='flux3_klein_base'&&c.kind==='training'),bytes=nativeContainer(f.expected,4),length=Number(bytes.readBigUInt64LE());
 for(const mutate of [m=>delete m['optim.step'],m=>m['optim.step']='3',m=>m['optim.step']='004',m=>m.optimizer='rose',m=>m.optimizer_config='']){const header=JSON.parse(bytes.subarray(8,8+length));mutate(header.__metadata__);const raw=Buffer.from(JSON.stringify(header)),padded=Buffer.alloc(Math.ceil(raw.length/8)*8,0x20),prefix=Buffer.alloc(8);raw.copy(padded);prefix.writeBigUInt64LE(BigInt(padded.length));const file=path.join(f.outputDir,'corrupt-native-counter.safetensors');await writeFile(file,Buffer.concat([prefix,padded,bytes.subarray(8+length)]));await assert.rejects(assertCheckpoint(file,cell,4));}
});

test('MLX inactive workflow factors legitimately omit moments; active optimizer advancement and inactive immutability are required',async t=>{
 const f=await nativeProofFixture(t),expected=structuredClone(f.expected);expected.factors['content_mode_blocks.image_cond.fixture.lora_A.weight']=[4,1];
 const first=path.join(f.outputDir,'mlx-first.safetensors'),last=path.join(f.outputDir,'mlx-final.safetensors'),cell={...catalog.cases.find(c=>c.backend==='mlx'&&c.modelId==='flux3_klein_base'&&c.kind==='training')};
 await writeFile(first,nativeContainer(expected,1,'mlx'));await writeFile(last,nativeContainer(expected,4,'mlx'));const initial=await assertCheckpoint(first,cell,1),final=await assertCheckpoint(last,cell,4);verifyNativeContinuation(initial,expected);verifyNativeContinuation(final,expected,initial);
 assert(!final.optimizer['content_mode_blocks.image_cond.fixture.lora_A.weight']);
 for(const mutate of [s=>s.tensorManifest['factor/content_mode_blocks.image_cond.fixture.lora_A.weight'].sha256=digest,s=>{for(const o of Object.values(s.optimizer))for(const key of [o.first,o.second])s.tensorManifest[key].sha256=initial.tensorManifest[key].sha256;},s=>s.optimizer[expected.activeFactorKeys[0]].step=1,s=>delete s.optimizer[expected.activeFactorKeys[0]],s=>{for(const name of expected.activeFactorKeys)s.tensorManifest[`factor/${name}`].sha256=initial.tensorManifest[`factor/${name}`].sha256;}]){const bad=structuredClone(final);mutate(bad);assert.throws(()=>verifyNativeContinuation(bad,expected,initial));}
});

test('MOCK owned API lifecycle proves actual persisted plans before starts, native identity continuity, fresh resume/export/reload and retained PNG',async t=>{
 const f=await nativeProofFixture(t),cell=catalog.cases.find(c=>c.backend==='candle'&&c.modelId==='flux3_klein_base'&&c.options.trainingMode==='captioned'),project={id:'p',path:path.join(f.state,'data/projects/p')};await mkdir(project.path,{recursive:true});let paused=false,restarts=0,canceled=false,resumedPlan,initialPlan,resumePolls=0,imagePolls=0,photo,witnessCalls=0;const calls=[];
 const initialPath=path.join(f.outputDir,'lora-step000001.resume.safetensors'),finalPath=path.join(f.outputDir,'lora-step000004.resume.safetensors'),exported=path.join(f.outputDir,'lora.safetensors');
 const session={...f.session,worker:{gpuId:'1'},workerPid:100,async pauseWorker(){paused=true;},failClosed(error){throw error;},async restartWorker(){paused=false;this.workerPid++;this.identity.workerId=`w-${++restarts}`;return{pid:this.workerPid};},async trainingWitness(plan){assert(paused);witnessCalls++;return f.witness(plan);},wait:async fn=>{for(let i=0;i<10;i++){const value=await fn();if(value)return value;}throw new Error('mock exhausted');},assertDeviceProof:async()=>({kind:'DECLARED_SYNTHETIC_NOT_GPU_PROOF'}),endpoint:{
  async exchange(method,route,body){assert(paused);calls.push([method,route,body]);return{status:400,body:{error:'checkpoint identity mismatch'}};},
  async call(method,route,body){calls.push([method,route,body]);
   if(route.endsWith('/training/datasets'))return{id:'dataset'};if(route==='/api/v1/training/targets')return f.targets;
   if(route.endsWith('/training/jobs')){assert(paused);initialPlan=structuredClone(f.plan);initialPlan.config=structuredClone(body.config);return{id:'train',payload:{dryRun:false,plan:initialPlan}};}
   if(route.startsWith('/api/v1/jobs?'))return[];
   if(route==='/api/v1/jobs/train'){if(!canceled){assert(!paused);await writeFile(initialPath,nativeContainer(f.expected,1));}return{id:'train',type:'lora_train',workerId:session.identity.workerId,status:canceled?'canceled':'running',progress:0.25,result:{resumeCheckpointStep:1},payload:{plan:initialPlan}};}
   if(route.endsWith('/cancel')){canceled=true;return{};}
   if(route.endsWith('/retry')){assert(paused);assert.equal(body.resume,true);resumedPlan=structuredClone(initialPlan);resumedPlan.config.advanced.resume=true;return{id:'resumed',payload:{plan:resumedPlan}};}
   if(route==='/api/v1/jobs/resumed'){assert(!paused);const done=++resumePolls>1;if(done){await writeFile(finalPath,nativeContainer(f.expected,4));const header=Buffer.from(JSON.stringify({'fixture.lora_A.weight':{dtype:'F32',shape:[4,1],data_offsets:[0,16]}})),prefix=Buffer.alloc(8);prefix.writeBigUInt64LE(BigInt(header.length));await writeFile(exported,Buffer.concat([prefix,header,Buffer.alloc(16)]));}return{id:'resumed',type:'lora_train',workerId:session.identity.workerId,status:done?'completed':'running',progress:done?1:0.5,payload:{plan:resumedPlan},result:{resumeCheckpointStep:done?4:1,loraId:'trained',outputPath:exported}};}
   if(route.startsWith('/api/v1/loras?'))return[{id:'trained',name:'Synthetic trained adapter'}];
   if(route==='/api/v1/image/jobs'){assert.equal(restarts,3);assert(!paused);const sidecar={id:'asset',projectId:'p',type:'image',generationSetId:'set',lineage:{jobId:'image'},file:{path:'assets/images/image.png',mimeType:'image/png',width:body.width,height:body.height},recipe:{model:body.model,mode:body.mode,adapter:'candle_flux3',prompt:body.prompt,seed:body.seed,loras:body.loras.map(l=>({...l,hash:sha256(Buffer.from([]))})),normalizedSettings:{width:body.width,height:body.height,count:1,family:'flux3'},rawAdapterSettings:{...body.advanced,realModelInference:true,numInferenceSteps:body.advanced.steps,repo:cell.repository}}};sidecar.recipe.loras[0].hash=await fileHash(exported);const generationSet={schemaVersion:1,id:'set',projectId:'p',jobId:'image',mode:body.mode,model:body.model,prompt:body.prompt,negativePrompt:body.negativePrompt,count:1,createdAt:'synthetic time',recipe:structuredClone(sidecar.recipe)},asset={...structuredClone(sidecar),generationSet,url:'/api/v1/projects/p/files/assets/images/image.png',sidecarPath:'assets/images/image.sceneworks.json'},bytes=png(body.width,body.height);await mkdir(path.join(project.path,'assets/images'),{recursive:true});await writeFile(path.join(project.path,sidecar.file.path),bytes);await writeFile(path.join(project.path,asset.sidecarPath),JSON.stringify(sidecar));photo={asset,bytes,job:{id:'image',projectId:'p',type:'image_generate',payload:body,status:'completed',stage:'completed',backend:'1',workerId:session.identity.workerId,result:{model:body.model,adapter:'candle_flux3',generationSetId:'set',assetIds:['asset'],assets:[sidecar]}}};return photo.job;}
   if(route==='/api/v1/jobs/image')return ++imagePolls===1?{...photo.job,status:'running',progress:0.5}:photo.job;
   if(route.endsWith('/assets'))return[photo.asset];throw new Error(`unexpected MOCK route ${route}`);
  },served:async()=>({status:200,bytes:photo.bytes})}};
 const report={events:[],requests:[],jobs:[],outputs:[],adapters:[],observations:[],timeoutSeconds:10};await trainingSequence(session,cell,f.target,project,{'train-0':f.input},{training:{captioned:{steps:4,saveEvery:1,rank:4,items:[{binding:'train-0',caption:'declared synthetic caption'}]}}},path.join(f.state,'retained'),report);
 assert.equal(witnessCalls,2);assert.equal(report.training.oldPid,101);assert.equal(report.training.resumedPid,102);assert.equal(report.checkpoint.state.cursor.micro_step,1);assert.equal(report.finalCheckpoint.state.cursor.micro_step,4);assert.deepEqual(report.checkpoint.state.identity,report.finalCheckpoint.state.identity);assert.equal(report.outputs[0].semanticAcceptance,'PENDING');assert.equal(report.outputs[0].recipe.loras[0].hash,await fileHash(exported));assert(calls.some(c=>c[2]?.payloadChanges));
});
