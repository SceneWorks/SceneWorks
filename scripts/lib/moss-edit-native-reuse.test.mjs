import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm,cp,realpath} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve,dirname} from 'node:path';
import {execFileSync} from 'node:child_process';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {VERSION,BACKEND_FILES,digest,save,fileIdentity,treeIdentity} from './moss-edit-terminal.mjs';
import {filesUnder} from './moss-edit-terminal-actions.mjs';
import {ordinaryExecutionCapture,validateOrdinaryExecutionCaptures} from './moss-edit-quality-amendment.mjs';
import {reuseProducers,executionTooling,validateOrdinaryMlx,readOnlyStoppedPackageSource} from './moss-edit-native-reuse.mjs';
globalThis.fetch=async()=>{throw Error('LIVE_NETWORK_FORBIDDEN');};
const temp=async t=>{const root=await mkdtemp(resolve(tmpdir(),'moss-native-reuse-'));t.after(()=>rm(root,{recursive:true,force:true}));return realpath(root);};
const env={GITHUB_ACTIONS:'true',GITHUB_REPOSITORY:'SceneWorks/SceneWorks',GITHUB_SHA:'a'.repeat(40),GITHUB_RUN_ID:'21',GITHUB_RUN_ATTEMPT:'1'};
const sources={sceneworks:{head:env.GITHUB_SHA},inference:{head:'b'.repeat(40)}};
const endpoint=backend=>({apiUrl:'http://127.0.0.1:12345',projectId:'project',workerId:`worker-${backend}`,workerGpuId:backend==='cuda'?'1':'mlx',tokenEnv:'TOKEN'});
const git=(root,...args)=>execFileSync('git',['-C',root,...args],{encoding:'utf8',stdio:['ignore','pipe','pipe']}).trim();
async function repo(root){await mkdir(root,{recursive:true});git(root,'init');git(root,'config','user.email','fixture@example.invalid');git(root,'config','user.name','Fixture');await writeFile(resolve(root,'fixture'),'inert');git(root,'add','.');git(root,'commit','-m','fixture');return treeIdentity(root);}
test('ordinary Mac handoff retains every completed row, guard, producer and byte identity',async t=>{
 const root=await temp(t),evidence=resolve(root,'evidence'),rows=Array.from({length:25},(_,i)=>({id:`row-${i}`,backend:'mlx',expected:'completed'})),guards=Array.from({length:29},(_,i)=>`guard-${i}`),plan={rows,requiredGuardCases:guards};
 const lock={version:VERSION,candidateId:'fixture',sources,files:{},artifacts:{mlx:{files:{}},cuda:{files:{}}},planSha256:digest(plan)};
 const binding={version:VERSION,backend:'mlx',lockSha256:digest(lock),sources,files:{},endpoint:endpoint('mlx')};
 await save(resolve(evidence,'protocol.lock.json'),lock);await save(resolve(evidence,'cases.json'),plan);await save(resolve(evidence,'bindings/mlx.json'),binding);
 await save(resolve(root,'execution.json'),{version:VERSION,backend:'mlx',execution:'COMPLETE',lockSha256:digest(lock),renders:25,refusals:29});
 await mkdir(resolve(root,'handoff/data'),{recursive:true});await writeFile(resolve(root,'handoff/data/.retained'),'hidden bytes');
 await save(resolve(root,'handoff/handoff.json'),{version:VERSION,projectId:'project',sources,files:await filesUnder(resolve(root,'handoff/data'))});
 await save(resolve(evidence,'external/ordinary-execution-mlx.json'),ordinaryExecutionCapture(lock,binding,env));
 for(const row of rows){const dir=resolve(evidence,'rows',row.id),request={seed:173},job={id:`job-${row.id}`,status:'completed'};await save(resolve(dir,'request.json'),request);await save(resolve(dir,'terminal-job.json'),job);await writeFile(resolve(dir,'output.png'),'pixels');await save(resolve(dir,'receipt.json'),{execution:'VERIFIED',rowId:row.id,backend:'mlx',lockSha256:digest(lock),bindingSha256:digest(binding),requestSha256:digest(request),jobSha256:digest(job),jobId:job.id,outputs:[{file:'output.png',...await fileIdentity(resolve(dir,'output.png'))}]});}
 for(const name of guards){const dir=resolve(evidence,'guards',`mlx-${name}`),common={backend:'mlx',case:name,lockSha256:digest(lock),requestSha256:'c'.repeat(64)};await save(resolve(dir,'receipt.json'),{...common,bindingSha256:digest(binding)});await save(resolve(dir,'no-provider-load.json'),{...common,method:'source-bound-moss-boundaries/v1',observation:{before:1,after:1}});}
 const config={candidateId:'fixture',sourceSha:sources.sceneworks.head,inferenceSha:sources.inference.head},producer={runId:'21',attempt:'1',artifactId:'23'},validate=()=>validateOrdinaryMlx(root,config,producer);
 assert.equal((await validate()).lockSha256,digest(lock));
 for(const [file,change] of [['execution.json',v=>v.renders=24],['evidence/external/ordinary-execution-mlx.json',v=>v.runId='22'],['evidence/rows/row-24/request.json',v=>v.seed=174],['evidence/guards/mlx-guard-28/no-provider-load.json',v=>v.backend='cuda']]){const path=resolve(root,file),bytes=await readFile(path),value=JSON.parse(bytes);change(value);await writeFile(path,JSON.stringify(value));await assert.rejects(validate);await writeFile(path,bytes);}
 await writeFile(resolve(root,'handoff/data/.retained'),'changed');await assert.rejects(validate,/handoff bytes/);
});
test('per-backend execution captures preserve original Mac and actual new CUDA workflow provenance',async t=>{
 const root=await temp(t),tooling='c'.repeat(40),lock={version:VERSION,candidateId:'fixture',sources,files:{},artifacts:{mlx:{files:{}},cuda:{files:{}}}},comparison={nativeExecutions:{}},expected={backendExecutions:{mlx:{runId:'21',attempt:'1'},cuda:{runId:'31',attempt:'2',toolingSha:tooling}}};
 for(const backend of ['mlx','cuda']){const binding={version:VERSION,backend,lockSha256:digest(lock),sources,files:{},endpoint:endpoint(backend)},producer=expected.backendExecutions[backend];await save(resolve(root,'bindings',`${backend}.json`),binding);const file=`external/ordinary-execution-${backend}.json`;await save(resolve(root,file),ordinaryExecutionCapture(lock,binding,{...env,GITHUB_SHA:producer.toolingSha??env.GITHUB_SHA,GITHUB_RUN_ID:producer.runId,GITHUB_RUN_ATTEMPT:producer.attempt},producer.toolingSha));comparison.nativeExecutions[backend]={file,...await fileIdentity(resolve(root,file))};}
 await validateOrdinaryExecutionCaptures(root,lock,comparison,expected);
 const original=await readFile(resolve(root,comparison.nativeExecutions.mlx.file));
 for(const patch of [{runId:'21'},{attempt:'1'},{toolingSha:sources.sceneworks.head}])await assert.rejects(()=>validateOrdinaryExecutionCaptures(root,lock,comparison,{backendExecutions:{...expected.backendExecutions,cuda:{...expected.backendExecutions.cuda,...patch}}}));
 assert.deepEqual(await readFile(resolve(root,comparison.nativeExecutions.mlx.file)),original);
 const file=comparison.nativeExecutions.cuda.file,bytes=await readFile(resolve(root,file)),changed=JSON.parse(bytes);changed.toolingSha='d'.repeat(40);await writeFile(resolve(root,file),JSON.stringify(changed));await assert.rejects(()=>validateOrdinaryExecutionCaptures(root,lock,comparison,expected));comparison.nativeExecutions.cuda={file,...await fileIdentity(resolve(root,file))};await assert.rejects(()=>validateOrdinaryExecutionCaptures(root,lock,comparison,expected));
});
test('stopped cache source requires archived and live producer bytes; it remains read only',async t=>{
 const base=await temp(t),sourceRoot=resolve(base,'listener/SceneWorks/source'),inferenceRoot=resolve(base,'inference'),actualSources={sceneworks:await repo(sourceRoot),inference:await repo(inferenceRoot)},parent=resolve(sourceRoot,'../../moss-sc-26752'),oldRoot=resolve(parent,'moss-execute-21-1-cuda'),nativeRoot=resolve(parent,'moss-build-21-1-cuda'),archive=resolve(base,'archive');
 const config={candidateId:'fixture',sourceSha:actualSources.sceneworks.head,inferenceSha:actualSources.inference.head,sourceRoot,inferenceRoot,root:resolve(parent,'moss-execute-31-1-cuda'),buildRoot:resolve(base,'builds'),packageFiles:[{file:'weights.bin',sha256:'e'.repeat(64)}]},producers={native:{runId:'21',attempt:'1'},stopped:{runId:'21',attempt:'1',artifactId:'24'}},identity={byteLength:1,sha256:'f'.repeat(64)};
 await save(resolve(config.buildRoot,'cuda/inventory.json'),{backend:'cuda',sources:actualSources,files:Object.fromEntries(BACKEND_FILES.map(name=>[name,identity]))});
 await save(resolve(nativeRoot,'hf-owner.json'),{identity:'original owner'});
 const prior={...config,backend:'cuda',root:oldRoot,hfHome:resolve(oldRoot,'hf')},transfer={operation:'same-volume-private-HF-ownership-transfer',from:resolve(nativeRoot,'hf'),to:prior.hfHome,candidateId:config.candidateId,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,originalPackages:identity,originalOwner:await fileIdentity(resolve(nativeRoot,'hf-owner.json')),before:config.packageFiles,after:config.packageFiles};
 await save(resolve(oldRoot,'runtime.json'),prior);await save(resolve(oldRoot,'hf-transfer.json'),transfer);await cp(oldRoot,archive,{recursive:true});
 const validate=()=>readOnlyStoppedPackageSource(config,archive,producers);assert.equal(await validate(),prior.hfHome);
 for(const [file,change] of [['runtime.json',v=>v.sourceSha='0'.repeat(40)],['hf-transfer.json',v=>v.originalPackages.sha256='0'.repeat(64)],['hf-transfer.json',v=>v.before=[]]]){const path=resolve(archive,file),bytes=await readFile(path),value=JSON.parse(bytes);change(value);await writeFile(path,JSON.stringify(value));await assert.rejects(validate);await writeFile(path,bytes);}
 const live=resolve(oldRoot,'runtime.json'),bytes=await readFile(live);await writeFile(live,Buffer.concat([bytes,Buffer.from('\n')]));await assert.rejects(validate,/live stopped package owner/);await writeFile(live,bytes);assert.equal(await validate(),prior.hfHome);
 assert.deepEqual(await readFile(resolve(oldRoot,'hf-transfer.json')),await readFile(resolve(archive,'hf-transfer.json')));
});
test('tooling is an exact clean checkout and producer inputs are closed numeric IDs',async t=>{
 const root=await temp(t),paths=['moss-edit-terminal.mjs','lib/moss-edit-terminal.mjs','moss-edit-terminal-actions.mjs','lib/moss-edit-terminal-actions.mjs','moss-edit-terminal-browser.mjs','lib/moss-edit-terminal-controls.mjs','lib/moss-edit-quality-amendment.mjs','lib/moss-edit-native-reuse.mjs'];
 for(const path of paths){await mkdir(dirname(resolve(root,'scripts',path)),{recursive:true});await writeFile(resolve(root,'scripts',path),'// inert\n');}await repo(root);const head=git(root,'rev-parse','HEAD'),capture=await executionTooling(root,head);assert.equal(Object.keys(capture.identities).length,8);await writeFile(resolve(root,'scripts',paths[0]),'changed');await assert.rejects(()=>executionTooling(root,head),/clean/);
 const values={MOSS_NATIVE_RUN_ID:'21',MOSS_NATIVE_RUN_ATTEMPT:'1',MOSS_MLX_RUN_ID:'21',MOSS_MLX_RUN_ATTEMPT:'1',MOSS_MLX_ARTIFACT_ID:'23',MOSS_CUDA_HF_RUN_ID:'21',MOSS_CUDA_HF_RUN_ATTEMPT:'1',MOSS_CUDA_HF_ARTIFACT_ID:'24'};assert.equal(reuseProducers(values).mlx.artifactId,'23');for(const key of Object.keys(values))assert.throws(()=>reuseProducers({...values,[key]:'0'}));
});
test('assembled new tooling launcher accepts frozen product paths and rejects substituted protocol paths before any model operation',async t=>{
 const root=await temp(t),toolingRoot=resolve(root,'tooling'),sourceRoot=resolve(root,'product');
 await cp(fileURLToPath(new URL('../../scripts',import.meta.url)),resolve(toolingRoot,'scripts'),{recursive:true});
 await repo(toolingRoot);const head=git(toolingRoot,'rev-parse','HEAD'),tooling=await executionTooling(toolingRoot,head);
 const {frozen}=await import(pathToFileURL(resolve(toolingRoot,'scripts/moss-edit-terminal.mjs')).href),plan=resolve(root,'cases.json'),launch=resolve(root,'launch.json');await save(plan,{});
 const config={version:VERSION,evidenceRoot:resolve(root,'evidence'),plan,executionTooling:tooling,sources:{sceneworks:sourceRoot},files:{harness:resolve(sourceRoot,'scripts/moss-edit-terminal.mjs'),harnessLibrary:resolve(sourceRoot,'scripts/lib/moss-edit-terminal.mjs')}};
 await save(launch,config);await assert.rejects(()=>frozen(launch),error=>error.code==='ENOENT'&&error.path.endsWith('protocol.lock.json'));
 for(const change of [v=>v.files.harness=v.executionTooling.files.harness,v=>v.executionTooling.files.browserControls=v.executionTooling.files.browserDriver,v=>v.executionTooling.identities.actionController.sha256='0'.repeat(64),v=>delete v.executionTooling.files.qualityAmendment]){const value=structuredClone(config);change(value);await writeFile(launch,JSON.stringify(value));await assert.rejects(()=>frozen(launch),/tooling|frozen product/);}
});
