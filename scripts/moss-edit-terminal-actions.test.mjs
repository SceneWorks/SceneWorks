import test from 'node:test';
import { constants } from 'node:fs';
import { stripJsoncComments } from './lib/jsonc.mjs';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, readFile, readdir, symlink, link, unlink, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { VERSION, sha, digest, save, json, fileIdentity, Api, decodeRgbPng, createPlan, validateEditReport, reserveAttempt, resumeUiAttempt, resolveAttempt, assertDispatchClear } from './lib/moss-edit-terminal.mjs';
import { inspectJournal, noLoadComparison, OBSERVATION_METHOD, hostBudgetHeader, hostGuardSpec, deriveHostTensorBound, hostTensorInvariant } from './lib/moss-edit-terminal-observation.mjs';
import { nativeDevice, CUDA_UUID, CUDA_PCI, childEnvironment, utilityWorkerEnvironment, standaloneWorkerEnvironment, stackReadinessSnapshot, readyUtilityWorker, relocateRegistry, filesUnder, assertOriginalEvidence, assertH18CheckpointFiles, continuationRemainingRows, retryRemainingRows, submittedCanceledRemainingRows, validateFailedUiCapture, continuationRowDisposition, checkpointDatabases, sampler, rgbPng, retainQualityPairs, command, validateDispatchEnvelope, R16_ORIGINAL_INVENTORY_SHA256, H18_RETRY_INVENTORY_SHA256 } from './lib/moss-edit-terminal-actions.mjs';
import { runtimeConfig, validateBundle, createBundle, inventoryReview, validateUpload, arrangeBuilds, resolveGuardAliases, prepareGuardFixtures, adapterRefusalState, isolatePackages, projectionCopyFlags, validatePackagePlan } from './moss-edit-terminal-actions.mjs';
import { privateHfIntegrity, sealPrivateHf, transferPrivateHf, adoptOriginalPrivateHf, releasePrivateHf, materializePrivateAdapter, normalizeMossEnvironment, exportMossEnvironment, allowedPrivateParent, cargoTarget, mossArtifactRoute, MAC_MOSS_BASE } from './lib/moss-private-hf.mjs';
import { lstat, realpath, readlink } from 'node:fs/promises';
import { submitGuard, validatePortableCapture, validateLexicalNonfinite, validateWorkerRefusal } from './moss-edit-terminal.mjs';

// Pure fixtures and owned CPU subprocesses only. Any accidental network is a test failure.
globalThis.fetch=async()=>{throw Error('LIVE_NETWORK_FORBIDDEN');};
const temporary=label=>mkdtemp(resolve(tmpdir(),`moss-actions-${label}-`));
const sourceRoot=fileURLToPath(new URL('../',import.meta.url));
const coverage='moss_image_base+sft; bounded cache-context before bind; cache-dispatch; runtime load before media catalog; actual MOSS generate/edit calls after admission';
const identity={candidateId:'inert-candidate',sourceSha:'a'.repeat(40),inferenceSha:'b'.repeat(40),workerId:'actual-fixture-worker',gpuId:'1',binarySha256:'c'.repeat(64),producerSha256:'d'.repeat(64),coverage};
const base=Date.parse('2026-10-24T12:00:00Z'),at=offset=>new Date(base+offset).toISOString();
function journal(events=['startup_begin','ready_after_bounded_startup_probe','heartbeat'],custom={}) {
  let previous='0'.repeat(64),active=null,counts={allocationCapableEntries:0,providerLoads:0,providerRuns:0,boundedContextEntries:0,cacheDispatchEntries:0};
  return Buffer.from(events.map((event,index)=>{
    if(event==='job_begin')active='actual-job';
    if(['provider_load','provider_run'].includes(event))counts.allocationCapableEntries++;if(event==='provider_load')counts.providerLoads++;if(event==='provider_run')counts.providerRuns++;if(event==='cache_context')counts.boundedContextEntries++;if(event==='cache_dispatch')counts.cacheDispatchEntries++;
    const body=JSON.stringify({version:1,sequence:index+1,identity,event,engine:event.startsWith('provider')||event.startsWith('cache_')?'moss_image_sft':null,jobId:active,at:at(index+1),...counts,...custom[index]});
    const line=JSON.stringify({body,previous,sha256:sha(`${previous}\n${body}`)});previous=JSON.parse(line).sha256;if(event==='job_end')active=null;return line;
  }).join('\n')+'\n');
}
test('actual source-bound startup/ready/heartbeat prefix proves unchanged boundary counts',()=>{
  const before=journal(),after=journal(['startup_begin','ready_after_bounded_startup_probe','heartbeat','heartbeat']);
  const proof=noLoadComparison(before,after,identity,{beforeAt:at(3),afterAt:at(4)});
  assert.equal(proof.method,OBSERVATION_METHOD);assert.equal(proof.after.allocationCapableEntries,0);assert.equal(proof.after.sequence,4);
});
test('missing ready, wrong worker/binary/coverage, truncation and tampering cannot masquerade as zero',()=>{
  for(const bytes of [Buffer.alloc(0),journal(['startup_begin','heartbeat']),journal().subarray(0,-1),journal().toString().replace('startup_begin','startup_fake'),journal(undefined,{0:{identity:{...identity,coverage:'unknown'}}})])assert.throws(()=>inspectJournal(bytes,identity));
  for(const key of ['candidateId','sourceSha','inferenceSha','workerId','gpuId','binarySha256','producerSha256'])assert.throws(()=>inspectJournal(journal(),{...identity,[key]:'other'}));
});
test('covered real entries increment conservative counters; warm run and bypassed hooks cannot pass refusal',()=>{
  const entries=['startup_begin','ready_after_bounded_startup_probe','heartbeat','job_begin','cache_context','provider_load','provider_run','job_end','heartbeat'];
  const after=journal(entries);assert.deepEqual(inspectJournal(after,identity).counts,{allocationCapableEntries:2,providerLoads:1,providerRuns:1,boundedContextEntries:1,cacheDispatchEntries:0});
  assert.throws(()=>noLoadComparison(journal(),after,identity,{beforeAt:at(3),afterAt:at(9)}),/allocation-capable/);
  assert.throws(()=>inspectJournal(journal(['startup_begin','ready_after_bounded_startup_probe','provider_run','heartbeat']),identity),/binding/);
  assert.throws(()=>inspectJournal(journal(entries,{8:{providerRuns:0}}),identity),/counters/);
  const warm=journal(['startup_begin','ready_after_bounded_startup_probe','heartbeat','job_begin','provider_run','job_end','heartbeat']);assert.equal(inspectJournal(warm,identity).counts.providerRuns,1);
});
test('append replacement, stale before and absent after observations fail closed',()=>{
  const before=journal(),after=journal(['startup_begin','ready_after_bounded_startup_probe','heartbeat','heartbeat']);
  assert.throws(()=>noLoadComparison(before,before,identity,{beforeAt:at(3),afterAt:at(4)}));
  assert.throws(()=>noLoadComparison(before,after,identity,{beforeAt:at(5000),afterAt:at(4)}));
  const changed=journal(['startup_begin','ready_after_bounded_startup_probe','heartbeat','heartbeat'],{0:{at:at(0)}});assert.throws(()=>noLoadComparison(before,changed,identity,{beforeAt:at(3),afterAt:at(4)}),/truncated/);
});
test('actual GPU1 identity and UUID mapping reject GPU0 even if it reports unused memory',async()=>{
  const execute=async(command,args)=>{assert.equal(command,'nvidia-smi');assert.ok(args.includes(`--id=${CUDA_UUID}`));return {stdout:`1, ${CUDA_UUID}, ${CUDA_PCI}, RTX PRO 6000, 12\n`};};
  const actual=await nativeDevice('cuda',execute);assert.equal(actual.device.physicalIndex,1);assert.equal(actual.usedBytes,12*1024*1024);
  await assert.rejects(()=>nativeDevice('cuda',async()=>({stdout:`0, ${CUDA_UUID}, ${CUDA_PCI}, RTX PRO 6000, 0\n`})),/GPU1/);
});
test('child environment isolates owned state/cache/auth and never invents a hardware ownership gate',()=>{
  const config={stateRoot:'/private/owned/state',root:'/private/owned',backend:'cuda',port:17957,workerId:'worker',hfHome:'/private/owned/hf',candidateId:'candidate',sourceSha:identity.sourceSha,inferenceSha:identity.inferenceSha,observationJournal:'/private/owned/journal'};
  const env=childEnvironment(config,'private-inert-token');assert.equal(env.CUDA_VISIBLE_DEVICES,CUDA_UUID);assert.equal(env.SCENEWORKS_GPU_ID,'1');assert.equal(env.SCENEWORKS_ACCESS_TOKEN,'private-inert-token');assert.equal(env.HF_HOME,config.hfHome);assert.equal(env.SCENEWORKS_MOSS_CANDIDATE_ID,'candidate');
  assert.equal(Object.keys(env).some(name=>/OWNERSHIP|LEASE|AGENT_TOKEN/.test(name)),false);
});
test('owned installation stack enables a distinct CPU download claimant and requires its actual registration',async()=>{
  for(const backend of ['mlx','cuda']) {
    const config={stateRoot:'/private/owned/state',backend,port:17957,workerId:`native-${backend}`,hfHome:'/private/owned/hf'};
    const common=childEnvironment(config,'private-inert-token'),native={...common,...standaloneWorkerEnvironment()},api={...common,...utilityWorkerEnvironment(config)};
    assert.equal(Object.hasOwn(common,'SCENEWORKS_WORKER_CHILD'),false);assert.equal(Object.hasOwn(api,'SCENEWORKS_WORKER_CHILD'),false);
    assert.equal(native.SCENEWORKS_WORKER_CHILD,'1');assert.equal(native.SCENEWORKS_WORKER_ONLY,'1');
    assert.equal(api.SCENEWORKS_RUN_UTILITY_INPROCESS,'1');assert.equal(api.SCENEWORKS_RUST_WORKER_GPU_ID,'cpu');assert.equal(api.SCENEWORKS_UTILITY_WORKERS,'1');
    assert.notEqual(api.SCENEWORKS_WORKER_ID,native.SCENEWORKS_WORKER_ID);
    const gpu={id:native.SCENEWORKS_WORKER_ID,gpuId:native.SCENEWORKS_GPU_ID,status:'idle',capabilities:['image_edit']};
    const utility={id:api.SCENEWORKS_WORKER_ID,gpuId:api.SCENEWORKS_RUST_WORKER_GPU_ID,status:'idle',capabilities:['cpu','model_download']};
    assert.equal(readyUtilityWorker([gpu],config.workerId),undefined);
    assert.equal(readyUtilityWorker([gpu,utility],config.workerId),utility);
    for(const changes of [{id:gpu.id},{gpuId:gpu.gpuId},{status:'busy'},{status:'offline'},{capabilities:['cpu']}])assert.equal(readyUtilityWorker([gpu,{...utility,...changes}],config.workerId),undefined);
  }
  const source=await readFile(resolve(sourceRoot,'scripts/lib/moss-edit-terminal-actions.mjs'),'utf8');
  assert.match(source,/launch\('api',utilityWorkerEnvironment\(config\)\)/);
  assert.match(source,/utility=readyUtilityWorker\(workers,config.workerId\)/);
  assert.match(source,/utilityWorker:utility/);
  assert.match(source,/launch\('worker',standaloneWorkerEnvironment\(\)\)/);
  assert.match(source,/launch\(`worker-restart-\$\{restarts\}`,\{\.\.\.standaloneWorkerEnvironment\(\),/);
  assert.match(source,/stdio:\[name.startsWith\('worker'\)\?'pipe':'ignore'/);
  assert.match(source,/if\(!registered\)await save\(resolve\(config.root,'startup-readiness.json'\),stackReadinessSnapshot/);
});
test('startup failure diagnostics retain bounded worker state without credentials or response bodies',()=>{
  const snapshot=stackReadinessSnapshot({workerId:'native',httpStatus:200,health:{status:'ok',readiness:{status:'ready'},token:'secret'},
    workers:[{id:'native-utility',gpuId:'cpu',status:'offline',capabilities:['cpu','model_download'],accessToken:'secret',extra:{password:'secret'}}],
    children:[{pid:1,exitCode:null,signalCode:null,startError:new Error('secret')}]});
  assert.deepEqual(snapshot.health,{status:'ok',readinessStatus:'ready'});
  assert.deepEqual(snapshot.workers,[{id:'native-utility',gpuId:'cpu',status:'offline',capabilities:['cpu','model_download']}]);
  assert.equal(snapshot.children[0].startError,true);assert.doesNotMatch(JSON.stringify(snapshot),/secret|accessToken|password/);
  const bounded=stackReadinessSnapshot({workers:Array.from({length:30},()=>({id:'x'.repeat(300),capabilities:Array(100).fill('cpu')})),children:Array(10).fill({})});
  assert.equal(bounded.workers.length,16);assert.equal(bounded.workers[0].id.length,200);assert.equal(bounded.workers[0].capabilities.length,64);assert.equal(bounded.children.length,2);
});
test('stopped registry relocation preserves all identities and rejects any unrelated project path',()=>{
  const original=[{id:'same-project',path:'/old/data/projects/campaign.sceneworks',name:'actual',other:{jobId:'same-job'}}];
  const moved=relocateRegistry(original,'/old/data','/new/data');assert.deepEqual({...moved[0],path:original[0].path},original[0]);assert.equal(moved[0].path,resolve('/new/data/projects/campaign.sceneworks'));
  for(const path of ['/outside/project','/old/data/../outside','/old/database/project'])assert.throws(()=>relocateRegistry([{id:'x',path}],'/old/data','/new/data'));
});
test('SQLite handoff requires actual jobs DB, successful checkpoint and integrity, never a live/busy copy',async()=>{
  const root=await temporary('database');await mkdir(resolve(root,'cache'));await writeFile(resolve(root,'cache/jobs.db'),'inert bytes');
  class Database {prepare(sql){return {get:()=>sql.includes('checkpoint')?{busy:0,log:0,checkpointed:0}:{integrity_check:'ok'}};}close(){this.closed=true;}}
  assert.equal((await checkpointDatabases(root,Database))[0].file,'cache/jobs.db');
  class Busy extends Database {prepare(){return {get:()=>({busy:1})};}}await assert.rejects(()=>checkpointDatabases(root,Busy),/busy/);
  class Corrupt extends Database {prepare(sql){return {get:()=>sql.includes('checkpoint')?{busy:0}:{integrity_check:'corrupt'}};}}await assert.rejects(()=>checkpointDatabases(root,Corrupt),/integrity/);
});
test('immutable input bundle rejects changed, extra and incorrectly dispatched bytes',async()=>{
  const root=await temporary('bundle'),bundle={sourceSha:identity.sourceSha,inferenceSha:identity.inferenceSha,candidateId:identity.candidateId};await save(resolve(root,'campaign.json'),bundle);
  await save(resolve(root,'bundle-inventory.json'),{files:await filesUnder(root)});const hash=(await fileIdentity(resolve(root,'bundle-inventory.json'))).sha256;
  assert.deepEqual(await validateBundle(root,hash,bundle.sourceSha,bundle.inferenceSha,bundle.candidateId),bundle);
  await assert.rejects(()=>validateBundle(root,hash,bundle.sourceSha,bundle.inferenceSha,'different'),/dispatch/);await writeFile(resolve(root,'unexpected'),'extra');await assert.rejects(()=>validateBundle(root,hash,bundle.sourceSha,bundle.inferenceSha,bundle.candidateId),/files\/bytes/);
});
test('native artifact arrangement accepts one source-bound artifact per backend and rejects merged attempts',async()=>{
  const root=await temporary('builds');for(const backend of ['mlx','cuda'])await save(resolve(root,`moss-native-${backend}-123-1/inventory.json`),{backend});await arrangeBuilds(root);assert.equal((await json(resolve(root,'cuda/inventory.json'))).backend,'cuda');
  const duplicate=await temporary('duplicate');for(const attempt of [1,2])await save(resolve(duplicate,`moss-native-mlx-123-${attempt}/inventory.json`),{backend:'mlx'});await assert.rejects(()=>arrangeBuilds(duplicate),/exactly one/);
});
test('continuation preserves original bytes and dispatches only untouched rows after the recovered job',async t=>{
  const root=await temporary('continuation');t.after(()=>rm(root,{recursive:true,force:true}));
  const rows=Array.from({length:25},(_,index)=>({id:index===6?'advanced-mlx-sft':`row-${index}`,backend:'mlx'})),plan={rows};
  for(const row of rows.slice(0,6))await save(resolve(root,'rows',row.id,'receipt.json'),{execution:'VERIFIED',rowId:row.id});
  await save(resolve(root,'rows/advanced-mlx-sft/failure.json'),{rowId:'advanced-mlx-sft',message:'per-prompt negatives gained an unsupported broadcast field'});
  const manifest={files:await filesUnder(root)};assert.equal(await assertOriginalEvidence(root,manifest,{exact:true}),7);
  assert.deepEqual((await continuationRemainingRows(root,plan)).map(row=>row.id),rows.slice(7).map(row=>row.id));
  await save(resolve(root,'attempts/mlx/row-7.start.json'),{id:'row-7'});
  await assert.rejects(()=>continuationRemainingRows(root,plan),/already touched/);
  await assert.rejects(()=>assertOriginalEvidence(root,manifest,{exact:true}),/changed/);
  await rm(resolve(root,'attempts'),{recursive:true});
  await writeFile(resolve(root,'rows/row-0/receipt.json'),'changed prior receipt');
  await assert.rejects(()=>assertOriginalEvidence(root,manifest),/original evidence bytes changed/);
  await assert.rejects(()=>continuationRemainingRows(root,plan));
});
test('H18 pre-POST UI failure retries the exact unresolved intent once and keeps all seven verified rows',async t=>{
  const root=await temporary('retry'),evidence=resolve(root,'evidence');t.after(()=>rm(root,{recursive:true,force:true}));
  const rows=Array.from({length:25},(_,index)=>({id:index===6?'advanced-mlx-sft':index===7?'advanced-mlx-sft-replay':`row-${index}`,backend:'mlx'})),plan={rows},lockSha256='a'.repeat(64);
  for(const row of rows.slice(0,7))await save(resolve(evidence,'rows',row.id,'receipt.json'),{execution:'VERIFIED',rowId:row.id});
  const original=await filesUnder(evidence),fields={transport:'ui',requestSha256:'b'.repeat(64),bindingSha256:'c'.repeat(64),lockSha256};
  const ticket=await reserveAttempt(evidence,'mlx','advanced-mlx-sft-replay',fields);
  const capture={rowId:ticket.id,lockSha256,error:"locator.click: Error: strict mode violation: getByRole('link', { name: 'Download portable workflow (.json)', exact: true }) resolved to 2 elements",submissions:[],imageJobPosts:[]};
  assert.equal(validateFailedUiCapture(capture,lockSha256,{historical:'h18'}),true);
  for(const mutation of [{submissions:[{}]},{imageJobPosts:[{}]},{error:'different UI failure'}])assert.throws(()=>validateFailedUiCapture({...capture,...mutation},lockSha256,{historical:'h18'}),/pre-submission/);
  const h19={...capture,error:'observed upload lacks multipart file bytes',actions:[{action:'drop-workflow-file'}]};
  assert.equal(validateFailedUiCapture(h19,lockSha256,{historical:'h19'}),true);
  assert.throws(()=>validateFailedUiCapture({...h19,actions:[...h19.actions,{action:'click',control:'Generate'}]},lockSha256,{historical:'h19'}),/pre-submission/);
  assert.equal(validateFailedUiCapture({...capture,error:'unrecognized selector',submissionPhase:'before-generate-attempt'},lockSha256),true);
  assert.throws(()=>validateFailedUiCapture({...capture,error:'Generate click timed out',submissionPhase:'generate-attempted'},lockSha256),/pre-submission/);
  const remaining=await retryRemainingRows(evidence,plan,lockSha256);
  assert.deepEqual(remaining.map(row=>row.id),rows.slice(7).map(row=>row.id));
  assert.deepEqual(rows.map(row=>continuationRowDisposition(row,remaining,{retry:true})),rows.map((_,index)=>({run:index>=7,reuseUiIntent:index===7})));
  await assert.rejects(()=>assertDispatchClear(evidence,'mlx'),{code:'MOSS_AWAITING_RESOLUTION'});
  const adopted=await resumeUiAttempt(evidence,'mlx',ticket.id,fields);assert.deepEqual(adopted.start,ticket.start);
  await resolveAttempt(adopted,'TERMINAL_VERIFIED',{jobId:'one-retried-ui-job'});
  assert.deepEqual(await filesUnder(resolve(evidence,'rows')),original.filter(item=>item.file.startsWith('rows/')).map(item=>({...item,file:item.file.slice(5)})));
  await assertDispatchClear(evidence,'mlx');await assert.rejects(()=>retryRemainingRows(evidence,plan,lockSha256));
  assert.equal((await json(resolve(evidence,'attempts/mlx/advanced-mlx-sft-replay.resolution.json'))).jobId,'one-retried-ui-job');
});
test('known submitted canceled job resumes after eight verified rows without dispatching it again',async t=>{
  const root=await temporary('known-cancel'),evidence=resolve(root,'evidence');t.after(()=>rm(root,{recursive:true,force:true}));
  const rows=Array.from({length:25},(_,index)=>({id:index===8?'advanced-cancel-mlx':`row-${index}`,backend:'mlx',...(index===8?{expected:'canceled',transport:'rest',cancelAtDenoise:true}:{})})),plan={rows};
  for(const row of rows.slice(0,8))await save(resolve(evidence,'rows',row.id,'receipt.json'),{execution:'VERIFIED'});
  const lock={candidateId:'fixture'},lockSha256=digest(lock),bindingSha256=sha('binding'),id=rows[8].id,dir=resolve(evidence,'rows',id),attempt=resolve(evidence,'attempts/mlx'),request={mode:'edit_image',projectId:'owned-project'},jobId='owned-canceled-job';
  await save(resolve(evidence,'protocol.lock.json'),lock);await save(resolve(attempt,`${id}.start.json`),{id,backend:'mlx',lockSha256,bindingSha256,requestSha256:digest(request)});await writeFile(resolve(attempt,'dispatch.lock'),id);
  await save(resolve(dir,'request.json'),request);await save(resolve(dir,'submission.json'),{id:jobId,status:'queued',projectId:request.projectId,payload:request,createdAt:'2026-10-25T16:33:21Z'});
  await save(resolve(dir,'admission.json'),{version:VERSION,lockSha256,backend:'mlx',device:{uuid:'owned-apple',pciOrRegistryId:'owned-apple',name:'Apple Silicon',platform:'apple-silicon'}});
  await save(resolve(dir,'cancel-observation.json'),{id:jobId,status:'running',stage:'generating',progress:0.25,message:'Image 1/4 — step 1/4.',workerId:'owned-worker',updatedAt:'2026-10-25T16:33:22Z'});
  await save(resolve(dir,'cancel-response.json'),{id:jobId,cancelRequested:true,workerId:'owned-worker',projectId:request.projectId,updatedAt:'2026-10-25T16:33:23Z'});
  const sampleFile=`samples/${id}-retry-2.jsonl`,samples=[{phase:'baseline',at:'2026-10-25T16:33:20Z'},{phase:'active',at:'2026-10-25T16:33:22Z'},{phase:'drain',at:'2026-10-25T16:33:24Z'}].map(item=>({...item,usedBytes:1,device:{uuid:'owned-apple'},raw:{actual:'fixture'}}));
  await mkdir(resolve(root,'samples'));await writeFile(resolve(root,sampleFile),samples.map(item=>JSON.stringify(item)).join('\n')+'\n');
  const checkpoint={files:[{file:sampleFile,...await fileIdentity(resolve(root,sampleFile))}],appendOnlyTails:[]};
  const result=await submittedCanceledRemainingRows(evidence,plan,lockSha256,bindingSha256,checkpoint,root);
  assert.equal(result.recovery.jobId,jobId);assert.equal(result.remaining[0].id,rows[9].id);assert.equal(result.remaining.length,16);
  const oldResponse=await json(resolve(dir,'cancel-response.json'));
  await writeFile(resolve(dir,'cancel-response.json'),JSON.stringify({...oldResponse,id:'foreign-job'}));
  await assert.rejects(()=>submittedCanceledRemainingRows(evidence,plan,lockSha256,bindingSha256,checkpoint,root),/cancellation/);
  await writeFile(resolve(dir,'cancel-response.json'),JSON.stringify(oldResponse));
  await save(resolve(evidence,'rows',rows[9].id,'receipt.json'),{execution:'VERIFIED'});
  await assert.rejects(()=>submittedCanceledRemainingRows(evidence,plan,lockSha256,bindingSha256,checkpoint,root),/later row/);
});
test('H18 checkpoint permits only two sealed post-upload log prefixes and archives disappearing mutex',async t=>{
  const root=await temporary('checkpoint'),retained=await temporary('retained');t.after(()=>rm(root,{recursive:true,force:true}));t.after(()=>rm(retained,{recursive:true,force:true}));
  for(let index=0;index<300;index++)await save(resolve(root,'evidence',`old-${index}.json`),{index});
  await mkdir(resolve(root,'evidence/attempts/mlx'),{recursive:true});await writeFile(resolve(root,'evidence/attempts/mlx/dispatch.lock'),'advanced-mlx-sft-replay');
  const tails=[];for(const name of ['api-continuation.log','journal-continuation.jsonl']){await writeFile(resolve(root,name),'artifact-prefix');tails.push({file:name,stoppedByteLength:0,stoppedSha256:''});}
  const files=await filesUnder(root);for(const tail of tails){await writeFile(resolve(root,tail.file),'artifact-prefix\npost-upload-tail');const identity=await fileIdentity(resolve(root,tail.file));tail.stoppedByteLength=identity.byteLength;tail.stoppedSha256=identity.sha256;}
  const checkpoint={files,appendOnlyTails:tails};assert.equal(await assertH18CheckpointFiles(root,checkpoint),303);
  for(const item of files){const path=resolve(retained,item.file==='evidence/attempts/mlx/dispatch.lock'?'continuation-h18-dispatch.lock':item.file.startsWith('evidence/')?item.file.slice(9):`continuation-h18-root/${item.file}`);await mkdir(resolve(path,'..'),{recursive:true});await writeFile(path,await readFile(resolve(root,item.file)));}
  await unlink(resolve(root,'evidence/attempts/mlx/dispatch.lock'));
  assert.equal(await assertH18CheckpointFiles(retained,checkpoint,{retained:true}),303);
  await writeFile(resolve(retained,'continuation-h18-root/api-continuation.log'),'artifact-prefix\nchanged-tail');await assert.rejects(()=>assertH18CheckpointFiles(retained,checkpoint,{retained:true}),/stopped bytes changed/);
  const chained=await temporary('chained');t.after(()=>rm(chained,{recursive:true,force:true}));
  const next={files:files.filter(item=>!['api-continuation.log','journal-continuation.jsonl'].includes(item.file)),appendOnlyTails:[]};
  for(const item of next.files){const path=resolve(chained,item.file==='evidence/attempts/mlx/dispatch.lock'?'continuation-prior-2-dispatch.lock':item.file.startsWith('evidence/')?item.file.slice(9):`continuation-prior-2-root/${item.file}`);await mkdir(resolve(path,'..'),{recursive:true});await writeFile(path,await readFile(item.file==='evidence/attempts/mlx/dispatch.lock'?resolve(retained,'continuation-h18-dispatch.lock'):resolve(root,item.file)));}
  assert.equal(await assertH18CheckpointFiles(chained,next,{retained:true,archiveTag:'prior-2'}),301);
  await writeFile(resolve(chained,'continuation-prior-2-dispatch.lock'),'changed mutex');await assert.rejects(()=>assertH18CheckpointFiles(chained,next,{retained:true,archiveTag:'prior-2'}),/artifact bytes changed/);
});
test('fixture aliases require real upload/readback IDs and preserve actual lexical nonfinite bytes',()=>{
  const bindings={campaignProject:'actual-project',foreign:{asset:{id:'actual-foreign-id'}}};
  const resolved=resolveGuardAliases({request:{assetId:'$fixture:foreign'},rawJson:'{"projectId":"$campaignProject","shift":NaN,"assetId":"$fixture:foreign"}'},bindings);
  assert.equal(resolved.request.assetId,'actual-foreign-id');assert.equal(resolved.rawJson,'{"projectId":"actual-project","shift":NaN,"assetId":"actual-foreign-id"}');assert.throws(()=>resolveGuardAliases('$fixture:invented',bindings));
});
test('actual foreign upload setup retains second owned project and byte-bound readbacks',async()=>{
  const root=await temporary('fixtures'),bytes=Buffer.from('actual inert upload bytes');await writeFile(resolve(root,'fixture.npy'),bytes);
  const config={root,bundleRoot:root,projectId:'campaign',candidateId:'candidate',guardFixtures:[{alias:'foreign',project:'foreign',file:'fixture.npy',sha256:sha(bytes),filename:'fixture.npy',mime:'application/x-npy',type:'tensor'}],guardRequests:{foreign:{request:{assetId:'$fixture:foreign'}}}};
  let posted;const api={async request(path,options={}){if(path==='/api/v1/projects')return {id:'actual-second-owned-project'};if(options.method==='POST'){posted=Buffer.from(await options.body.get('file').arrayBuffer());return {id:'actual-upload'};}if(options.raw)return bytes;return {id:'actual-upload',projectId:'actual-second-owned-project',type:'tensor',file:{path:'media/fixture.npy'}};}};
  const result=await prepareGuardFixtures(config,api);assert.equal(result.foreign.request.assetId,'actual-upload');assert.deepEqual(posted,bytes);assert.equal((await json(resolve(root,'evidence/guard-fixtures/foreign.json'))).actualSha256,sha(bytes));
});
test('missing-input setup purges only a fresh campaign fixture and retains its actual ID and HTTP404',async()=>{
  const root=await temporary('missing-fixture'),bytes=Buffer.from('actual uploaded fixture bytes');await writeFile(resolve(root,'fixture.npy'),bytes);
  const config={root,bundleRoot:root,projectId:'campaign',candidateId:'candidate',guardFixtures:[{alias:'missing',project:'campaign',file:'fixture.npy',sha256:sha(bytes),filename:'fixture.npy',mime:'application/x-npy',type:'tensor',deleteAfterUpload:true,cases:['missing-tensor']}],guardRequests:{'missing-tensor':{request:{assetId:'$fixture:missing'}}}};
  const calls=[];let deleted=false;
  const api={async request(path,options={}){calls.push({path,method:options.method??'GET'});if(options.method==='POST')return {id:'fresh-actual-id'};if(options.method==='DELETE'){assert.equal(path,'/api/v1/projects/campaign/assets/fresh-actual-id/purge?permanent=true');deleted=true;return {id:'fresh-actual-id'};}if(deleted){const error=Error('actual missing fixture');error.status=404;throw error;}if(options.raw)return bytes;return {id:'fresh-actual-id',projectId:'campaign',type:'tensor',file:{path:'media/fixture.npy'}};}};
  const result=await prepareGuardFixtures(config,api);assert.equal(result['missing-tensor'].request.assetId,'fresh-actual-id');const binding=await json(resolve(root,'evidence/guard-fixtures/missing.json'));assert.equal(binding.deleted.readbackStatus,404);assert.equal(binding.asset.id,'fresh-actual-id');assert.deepEqual(await readFile(resolve(root,'evidence/guard-fixtures/missing-actual.bin')),bytes);assert.equal(calls.filter(call=>call.method==='DELETE').length,1);
  const wrong={...api,async request(path,options={}){if(options.method==='DELETE')throw Error('delete denied');if(options.method==='POST')return {id:'fresh-second-id'};if(options.raw)return bytes;return {id:'fresh-second-id',projectId:'campaign',type:'tensor',file:{path:'media/fixture.npy'}};}};
  const deniedRoot=await temporary('missing-denied');await assert.rejects(()=>prepareGuardFixtures({...config,root:deniedRoot},wrong),/delete denied/);
});
test('owned adapter negative states are real missing/corrupt bytes and restore exact original',async()=>{
  const root=await temporary('adapter'),file='hub/snapshot/edit.safetensors',original=Buffer.from('original adapter');await mkdir(resolve(root,'hf/hub/snapshot'),{recursive:true});await writeFile(resolve(root,'hf',file),original);
  const config={root,hfHome:resolve(root,'hf'),candidateId:'candidate',editAdapterFiles:[file],packageFiles:[{file,sha256:sha(original)}]};
  const missing=await adapterRefusalState(config,'missing-adapter',{adapterFile:file});await assert.rejects(()=>readFile(resolve(config.hfHome,file)),{code:'ENOENT'});await missing.restore();
  const corrupt=Buffer.from('actual corrupt bytes'),mutation=await adapterRefusalState(config,'corrupt-adapter',{adapterFile:file,corruptBytesBase64:corrupt.toString('base64'),corruptSha256:sha(corrupt)});assert.deepEqual(await readFile(resolve(config.hfHome,file)),corrupt);await mutation.restore();assert.deepEqual(await readFile(resolve(config.hfHome,file)),original);
  await assert.rejects(()=>adapterRefusalState(config,'missing-adapter',{adapterFile:'../shared'}));
});
test('private HF projection independently owns exact dereferenced bytes and refuses existing targets',async t=>{
  const root=await temporary('packages');t.after(()=>rm(root,{recursive:true,force:true}));const source=resolve(root,'source'),run=resolve(root,'run'),bytes=Buffer.from('frozen model bytes');await mkdir(source);await writeFile(resolve(source,'blob'),bytes);await symlink(resolve(source,'blob'),resolve(source,'weights'));
  const config={root:run,hfHome:resolve(run,'hf'),packageSource:source,packageFiles:[{file:'weights',sha256:sha(bytes)}]},target=resolve(config.hfHome,'weights');await isolatePackages(config);
  const info=await lstat(target),original=await lstat(resolve(source,'blob'));assert.equal(info.isFile(),true);assert.equal(info.isSymbolicLink(),false);assert.equal(info.nlink,1);assert.equal(info.size,bytes.length);assert.notEqual(info.ino,original.ino);assert.deepEqual(await readFile(target),bytes);assert.equal(sha(await readFile(target)),config.packageFiles[0].sha256);
  await writeFile(target,'owned negative state');assert.deepEqual(await readFile(resolve(source,'blob')),bytes);
  await assert.rejects(()=>isolatePackages(config),{code:'EEXIST'});assert.equal(await readFile(target,'utf8'),'owned negative state');assert.deepEqual(await readFile(resolve(source,'blob')),bytes);
  await writeFile(target,bytes);await writeFile(resolve(source,'blob'),'changed shared fixture');assert.deepEqual(await readFile(target),bytes);await assert.rejects(()=>isolatePackages({...config,hfHome:resolve(root,'other/hf'),root:resolve(root,'other')}),/source package file differs/);
});
test('optional APFS clone flags retain exclusive creation and ordinary copy fallback on other devices/platforms',()=>{
  assert.equal(projectionCopyFlags('darwin',1,1),constants.COPYFILE_EXCL|constants.COPYFILE_FICLONE);
  for(const [platform,source,target]of [['darwin',1,2],['linux',1,1],['win32',1,1]])assert.equal(projectionCopyFlags(platform,source,target),constants.COPYFILE_EXCL);
  assert.equal(projectionCopyFlags('darwin',1,1)&constants.COPYFILE_FICLONE_FORCE,0);
});
test('installer omits unavailable source adapter and refuses copying installed projections',async()=>{
  const root=await temporary('installed-projection'),source=resolve(root,'shared'),buildRoot=resolve(root,'build'),backbone=Buffer.from('exact backbone');
  await mkdir(source);await writeFile(resolve(source,'backbone'),backbone);
  const packageFiles=[{file:'backbone',sha256:sha(backbone)},{file:'adapter',sha256:sha('adapter')}],build={root:buildRoot,hfHome:resolve(buildRoot,'hf'),packageSource:source,packageFiles,editAdapterFiles:['adapter']};
  await isolatePackages(build,{omitAdapter:true});assert.deepEqual(await readFile(resolve(build.hfHome,'backbone')),backbone);await assert.rejects(()=>readFile(resolve(build.hfHome,'adapter')),{code:'ENOENT'});
  await assert.rejects(()=>isolatePackages(build,{installedInventory:{hfHome:source,files:packageFiles}}),/without copying/);
});
test('raw JSON API transport sends exact NaN spelling and retains actual HTTP refusal',async()=>{
  const original=globalThis.fetch,rawJson='{"training_shift":NaN}';let observed;
  globalThis.fetch=async(url,options)=>{observed=options;return new Response('actual lexical rejection',{status:422});};
  try{const root=await temporary('raw');await submitGuard(new Api('http://127.0.0.1:0','inert'),{rawJson,expectedStatus:422,errorIncludes:['lexical rejection']},rawJson,root);assert.equal(observed.body,rawJson);assert.equal(observed.headers['Content-Type'],'application/json');const refusal=await json(resolve(root,'response.json'));assert.equal(refusal.status,422);assert.equal(refusal.http.body,'actual lexical rejection');assert.equal(refusal.http.sha256,sha('actual lexical rejection'));assert.equal(refusal.http.byteLength,Buffer.byteLength('actual lexical rejection'));}
  finally{globalThis.fetch=original;}
});
test('whole-device sampler retains raw baseline/active/drain observations, never allocator claims',async()=>{
  const root=await temporary('samples');let used=10;const result=await sampler({backend:'cuda',file:resolve(root,'memory.jsonl'),intervalMs:1,observe:async()=>({device:{uuid:CUDA_UUID},usedBytes:used++,raw:'actual fixture observation'})});await new Promise(done=>setTimeout(done,10));const evidence=await result.stop('actual-job','f'.repeat(64));assert.equal(evidence.samples[0].phase,'baseline');assert.ok(evidence.samples.some(item=>item.phase==='active'));assert.equal(evidence.samples.at(-1).phase,'drain');assert.equal('deviceAllocations' in evidence,false);
});
test('quality pair artifacts contain actual unscaled source/output pixels and deterministic output crops',async()=>{
  const root=await temporary('quality'),row={id:'quality-fixture',anchors:['anchor'],output:'images',refs:['B']},directory=resolve(root,'rows',row.id);await mkdir(directory,{recursive:true});
  const original=rgbPng(2,2,Buffer.alloc(12,11)),output=rgbPng(3,3,Buffer.alloc(27,22));await mkdir(resolve(root,'inputs'));await writeFile(resolve(root,'inputs/B.png'),original);await writeFile(resolve(directory,'output-0.png'),output);await save(resolve(root,'input-identities.json'),{B:{file:'inputs/B.png',sha256:sha(original)}});await save(resolve(directory,'receipt.json'),{outputs:[{file:'output-0.png',sha256:sha(output)}]});
  await retainQualityPairs({root,lock:{candidateId:'inert'}},row);const pair=decodeRgbPng(await readFile(resolve(directory,'review-pair-B.png')));assert.equal(pair.width,3);assert.equal(pair.height,5);assert.equal(pair.pixels[0],11);assert.equal(pair.pixels[2*3*3],22);const retained=await json(resolve(directory,'visual-pairs.json'));assert.equal(retained.detailCrops.length,3);assert.equal(retained.inputPairs[0].inputSha256,sha(original));
});
test('owned command errors and timeouts retain logs and never silently succeed',async()=>{
  const root=await temporary('command');await command(process.execPath,['-e','process.stdout.write("actual CPU command")'],{log:resolve(root,'ok.log')});assert.equal(await readFile(resolve(root,'ok.log'),'utf8'),'actual CPU command');await assert.rejects(()=>command(process.execPath,['-e','process.exit(7)'],{log:resolve(root,'failed.log')}),/exit 7/);
});
test('source hooks cover both cold/warm cache routes and precede actual device/provider entry',async()=>{
  const cache=await readFile(resolve(sourceRoot,'crates/sceneworks-worker/src/generator_cache.rs'),'utf8'),runtime=await readFile(resolve(sourceRoot,'crates/sceneworks-worker/src/inference_runtime.rs'),'utf8'),worker=await readFile(resolve(sourceRoot,'crates/sceneworks-worker/src/lib.rs'),'utf8'),journalSource=await readFile(resolve(sourceRoot,'crates/sceneworks-worker/src/moss_terminal_observation.rs'),'utf8');
  assert.equal((cache.match(/entry\(engine_id, "cache_context"\)\?;\s*let _backend_context = bind_backend_load_context/g)??[]).length,2);assert.equal((cache.match(/entry\(engine_id, "cache_dispatch"\)/g)??[]).length,2);assert.match(runtime,/entry\(id, "provider_load"\)[\s\S]*media\(\)\.load\(id, spec\)/);assert.ok(worker.indexOf('moss_terminal_observation::initialize')<worker.indexOf('let mut health = cuda_startup_health'));assert.match(worker,/begin_job\(&job\)/);assert.match(journalSource,/create_new\(true\)/);assert.match(journalSource,/include_bytes!\("moss_terminal_observation.rs"\)/);const moss=await readFile(resolve(sourceRoot,'crates/sceneworks-worker/src/image_jobs/moss.rs'),'utf8');assert.equal((moss.match(/moss_terminal_observation::provider_run\(\)\?;\s*let output = generator\s*\.generate/g)??[]).length,2);assert.doesNotMatch(cache,/entry\(engine_id, "provider_run"\)/);
});
test('Actions retains exact paired sources/builds and all phases without a fake success or hardware lock',async()=>{
  const workflow=await readFile(resolve(sourceRoot,'.github/workflows/moss-edit-terminal.yml'),'utf8'),controller=await readFile(resolve(sourceRoot,'scripts/moss-edit-terminal-actions.mjs'),'utf8');
  for(const label of ['rw-starvector','real-weights','execute-mlx:','execute-cuda:','verify-retained:','artifact-ids:','bundle_manifest_sha256:','if: always()','execution_run_attempt:'])assert.ok(workflow.includes(label),label);
  const mlxBuild=workflow.match(/- name: Build exact embedded API, worker, web and native inventories \(MLX\)([\s\S]*?)(?=      - name:)/)?.[1],cudaBuild=workflow.match(/- name: Build exact embedded API, worker, web and native inventories \(CUDA\)([\s\S]*?)(?=      - name:)/)?.[1];
  assert.match(mlxBuild,/if: matrix\.backend == 'mlx'/);assert.match(mlxBuild,/env:\s+CARGO_PROFILE_DEV_STRIP: none\s+CARGO_PROFILE_TEST_STRIP: none/);assert.doesNotMatch(cudaBuild,/CARGO_PROFILE_(?:DEV|TEST)_STRIP/);
  assert.match(workflow,/needs: execute-mlx/);assert.match(workflow,/CUDA_VISIBLE_DEVICES: GPU-e4b79931-7be6-f216-460a-f5405cfafffe/);assert.doesNotMatch(workflow,/continue-on-error|pull_request:|^  push:|lease-helper|owner-token|SCENEWORKS_.*LEASE/i);assert.ok(controller.indexOf('await verifyBuild(config)')<controller.indexOf('const stack=await startStack(config,token);let stopped'));
  const config=runtimeConfig({sourceSha:identity.sourceSha,inferenceSha:identity.inferenceSha,candidateId:'candidate',backends:{cuda:{hfHome:'/source/cache',port:17957,installAdapter:true,editAdapterFiles:['weights'],packageFiles:[{file:'weights',sha256:'a'.repeat(64)}]}}},{sourceRoot,inferenceRoot:'/inference',root:'/private/run',backend:'cuda',buildRoot:'/builds',bundleRoot:'/bundle'});assert.equal(config.hfHome,resolve('/private/run/hf'));assert.equal(config.packageSource,resolve('/source/cache'));
});
test('registered MOSS dispatcher rejects ambiguous inputs before calling exact-ref native phases',async()=>{
  const execute={candidate_id:'sc-26752-final',bundle_run_id:'123',bundle_artifact_id:'456',bundle_manifest_sha256:'f'.repeat(64),phase:'execute'},validate=value=>validateDispatchEnvelope(JSON.stringify(value),identity.sourceSha,identity.inferenceSha,identity.sourceSha);
  assert.deepEqual(validate(execute),{...execute,execution_run_id:'',execution_run_attempt:'1',review_artifact_id:'',review_run_id:'',upload_path:'',upload_kind:''});
  const verify={...execute,phase:'verify',execution_run_id:'789',execution_run_attempt:'2',review_artifact_id:'987',review_run_id:'999'};assert.deepEqual(validate(verify),{...verify,upload_path:'',upload_kind:''});
  const upload={candidate_id:execute.candidate_id,bundle_manifest_sha256:execute.bundle_manifest_sha256,phase:'upload',upload_kind:'bundle',upload_path:'/Users/michael/.codex/epic-runs/sc-26752/sealed-final-inputs'};
  assert.equal(validate(upload).bundle_run_id,'');assert.equal(validate({...upload,upload_kind:'review'}).phase,'upload');
  const tooling='c'.repeat(40),continuation={...execute,phase:'continue',tooling_sha:tooling,product_sha:identity.sourceSha,execution_run_id:'789',execution_run_attempt:'1',original_inventory_path:'/Users/michael/.codex/epic-runs/sc-26752/terminal57-final-input-prep-r16/r16-original-partial-evidence-inventory.json',original_inventory_sha256:R16_ORIGINAL_INVENTORY_SHA256};
  const repaired=value=>validateDispatchEnvelope(JSON.stringify(value),tooling,identity.inferenceSha,tooling);
  assert.equal(repaired(continuation).tooling_sha,tooling);
  assert.throws(()=>repaired({...continuation,tooling_sha:identity.sourceSha}),/separately identify H17 product/);
  assert.throws(()=>repaired({...continuation,original_inventory_path:'/private/foreign'}),/sealed local original-evidence/);
  assert.throws(()=>repaired({...continuation,original_inventory_sha256:'e'.repeat(64)}),/sealed local original-evidence/);
  const retry={...continuation,retry_inventory_path:'/Users/michael/.codex/epic-runs/sc-26752/terminal57-final-input-prep-r16/h18-actual-artifact-inventory.json',retry_inventory_sha256:H18_RETRY_INVENTORY_SHA256};
  assert.equal(repaired(retry).retry_inventory_sha256,H18_RETRY_INVENTORY_SHA256);
  assert.throws(()=>repaired({...retry,retry_inventory_sha256:'e'.repeat(64)}),/sealed or downloaded failed-attempt artifact/);
  const chained={...retry,retry_inventory_path:'/Users/michael/.codex/epic-runs/sc-26752/terminal57-final-input-prep-r16/h19-actual-artifact-inventory.json',retry_inventory_sha256:'d7bcbe27f63ef072376585feaf2ae6f9bbe3bf9f16c0857254aa28a2e3bcaed8',retry_run_id:'40210686983',retry_artifact_id:'12604389896'};
  assert.equal(repaired(chained).retry_run_id,'40210686983');
  assert.throws(()=>repaired({...chained,retry_artifact_id:''}),/failed-attempt artifact/);
  assert.throws(()=>validateDispatchEnvelope(JSON.stringify({...retry,tooling_sha:'82c534d50b72c1a5984514b90548002da212d735'}),'82c534d50b72c1a5984514b90548002da212d735',identity.inferenceSha,'82c534d50b72c1a5984514b90548002da212d735'),/distinct tooling/);
  assert.throws(()=>repaired({...execute,retry_inventory_path:retry.retry_inventory_path,retry_inventory_sha256:retry.retry_inventory_sha256}),/cannot consume/);
  assert.throws(()=>repaired({...continuation,review_artifact_id:'999'}),/no review/);
  assert.equal(repaired({...verify,tooling_sha:tooling,product_sha:identity.sourceSha,native_run_id:'789',native_run_attempt:'1'}).native_run_id,'789');
  assert.throws(()=>repaired({...verify,tooling_sha:tooling,product_sha:identity.sourceSha}),/original native run/);
  for(const value of [null,[],{}, {...execute,extra:'unexpected'},{...execute,bundle_run_id:123},{...execute,bundle_artifact_id:'../other'},{...execute,candidate_id:'../other'},{...execute,bundle_manifest_sha256:'wrong'},{...execute,phase:'build'},{...execute,execution_run_id:'789'},{...verify,execution_run_id:''},{...verify,review_artifact_id:''},{...verify,execution_run_attempt:'0'}])assert.throws(()=>validate(value));
  assert.throws(()=>validateDispatchEnvelope('{',identity.sourceSha,identity.inferenceSha,identity.sourceSha));
  assert.throws(()=>validateDispatchEnvelope(JSON.stringify(execute),'feature/branch',identity.inferenceSha,identity.sourceSha));
  assert.throws(()=>validateDispatchEnvelope(JSON.stringify(execute),identity.sourceSha,'main',identity.sourceSha));
  assert.throws(()=>validateDispatchEnvelope(JSON.stringify(execute),identity.sourceSha,identity.inferenceSha,'0'.repeat(40)));
  for(const change of [{upload_path:'/Users/michael/.codex/epic-runs/sc-26752/../private'},{upload_path:'/private/foreign'},{upload_kind:'credentials'},{bundle_run_id:'123'},{execution_run_id:'789'}])assert.throws(()=>validate({...upload,...change}));
  const dispatcher=await readFile(resolve(sourceRoot,'.github/workflows/memory-catalog-campaign.yml'),'utf8'),workflow=await readFile(resolve(sourceRoot,'.github/workflows/moss-edit-terminal.yml'),'utf8');
  assert.match(dispatcher,/moss-edit-terminal:\s+if:.*inputs\.profile == 'moss-edit-terminal'[\s\S]*?needs: moss-dispatch[\s\S]*?uses: \.\/\.github\/workflows\/moss-edit-terminal\.yml/);
  assert.match(dispatcher,/validateDispatchEnvelope\(process\.env\.MOSS_DISPATCH,process\.env\.MOSS_SOURCE_SHA,process\.env\.MOSS_INFERENCE_SHA,process\.env\.GITHUB_SHA\)/);
  for(const backend of ['mlx','candle'])assert.ok(dispatcher.includes(`(inputs.profile == '' || inputs.profile == 'catalog') && inputs.backend == '${backend}'`));
  for(const profile of ['qwen-image-2-1-app-lifecycle','iris-mlx-real-weights','supra2-mlx-real-weights'])assert.ok(dispatcher.includes(`inputs.profile == '${profile}'`));
  assert.match(workflow,/workflow_call:/);assert.match(workflow,/SCENEWORKS_INFERENCE_READ_TOKEN:\s+required: false/);
  assert.match(workflow,/INFERENCE_READ_TOKEN:\s+required: false/);assert.equal((workflow.match(/secrets\.INFERENCE_READ_TOKEN \|\| secrets\.SCENEWORKS_INFERENCE_READ_TOKEN \|\| github\.token/g)??[]).length,6);
  assert.match(workflow,/test "\$RUNNER_NAME" = nax-macos/);assert.match(workflow,/run-id: \$\{\{ inputs.review_run_id \|\| inputs.bundle_run_id \}\}/);
});
test('first bundle copies only exact fixed inputs, provenance and declared fixtures, then seals every byte',async()=>{
  const root=await temporary('first-bundle'),input=resolve(root,'original.png'),receipt=resolve(root,'source.json'),numeric=resolve(root,'numeric.json'),fixture=resolve(root,'guard.npy');
  const pixels=rgbPng(1,1,Buffer.from([10,20,30]));await writeFile(input,pixels);await writeFile(receipt,'actual source receipt');await writeFile(numeric,'actual numerical receipt');await writeFile(fixture,'actual guard fixture');
  const inputs={provenance:'generated inputs; no camera-photo claim',assets:['B','L','P','G'].map(id=>({id,path:input,sha256:sha(pixels),source_receipt:receipt,receipt_sha256:sha('actual source receipt')})),single_case:{references:['B'],width:768,height:512,prompt:'fixed single prompt'},four_case:{references:['B','L','P','G'],width:512,height:768,prompt:'fixed four prompt',mandatory_preservation:['identity','bottle','lamp','poster'],typography_scope:'exact lettering excluded'},common:{seed:173,steps:30,guidance_scale:4,inference_shift:5}};
  const campaign={sourceSha:identity.sourceSha,inferenceSha:identity.inferenceSha,candidateId:identity.candidateId,approvedBy:'fixture',approvalEvidenceSha256:'e'.repeat(64),phase18ReviewSha256:'e'.repeat(64),inferencePreMergeReviewSha256:'e'.repeat(64),cacheTolerance:0.001,numericalCarryForward:{file:'numeric.json',sha256:sha('actual numerical receipt')},guardFixtures:[{file:'guard.npy',sha256:sha('actual guard fixture')}],guardRequests:Object.fromEntries(createPlan(inputs).requiredGuardCases.map(name=>[name,{request:{},expectedStatus:400}]))};
  await save(resolve(root,'inputs.json'),inputs);
  const legacy={...campaign,finalSourceReviewSha256:campaign.inferencePreMergeReviewSha256};delete legacy.inferencePreMergeReviewSha256;
  await save(resolve(root,'legacy-campaign.json'),legacy);
  await assert.rejects(()=>createBundle(resolve(root,'legacy-campaign.json'),resolve(root,'inputs.json'),resolve(root,'legacy-sealed')),/inference pre-merge source review provenance/);
  await save(resolve(root,'campaign.json'),campaign);const output=resolve(root,'sealed'),inventory=await createBundle(resolve(root,'campaign.json'),resolve(root,'inputs.json'),output);
  assert.equal((await validateUpload(output,'bundle',inventory.sha256,campaign.sourceSha,campaign.inferenceSha,campaign.candidateId)).inputManifest,'inputs.json');
  const copied=await json(resolve(output,'inputs.json'));assert.deepEqual(copied.common,inputs.common);assert.deepEqual(copied.four_case,inputs.four_case);assert.equal(copied.provenance,inputs.provenance);
  for(const item of copied.assets){assert.equal((await fileIdentity(resolve(output,item.path))).sha256,sha(pixels));assert.equal((await fileIdentity(resolve(output,item.source_receipt))).sha256,item.receipt_sha256);}
  await assert.rejects(()=>createBundle(resolve(root,'campaign.json'),resolve(root,'inputs.json'),output),/EEXIST/);
  await writeFile(resolve(output,'unexpected'),'not reviewed');await assert.rejects(()=>validateUpload(output,'bundle',inventory.sha256,campaign.sourceSha,campaign.inferenceSha,campaign.candidateId),/files\/bytes/);
  const reviews=resolve(root,'reviews');await save(resolve(reviews,'external/actual-review.json'),{observer:'root actual observer',verdict:'AWAITING'});const reviewInventory=await inventoryReview(reviews,campaign.sourceSha,campaign.inferenceSha,campaign.candidateId);
  await validateUpload(reviews,'review',reviewInventory.sha256,campaign.sourceSha,campaign.inferenceSha,campaign.candidateId);
  await assert.rejects(()=>validateUpload(reviews,'review',reviewInventory.sha256,'0'.repeat(40),campaign.inferenceSha,campaign.candidateId),/source\/candidate/);
  await writeFile(resolve(reviews,'external/actual-review.json'),'changed verdict');await assert.rejects(()=>validateUpload(reviews,'review',reviewInventory.sha256,campaign.sourceSha,campaign.inferenceSha,campaign.candidateId),/files\/bytes/);
});
test('actual PNG workflow inspector evidence preserves original exported bytes and rejects substitutions',async()=>{
  const contract=await json(resolve(sourceRoot,'scripts/fixtures/moss-edit-product-contract-v2.json')),bytes=rgbPng(1,1,Buffer.from([1,2,3]));
  const inspection={status:'workflow',workflow:contract.exportedWorkflow,resolution:{missing:[]}},startedAt=new Date(Date.now()-2000).toISOString(),observedAt=new Date(Date.now()-1000).toISOString(),workflow={file:'actual-download.png',sha256:sha(bytes),inspectUploadSha256:sha(bytes),browserFile:{startedAt,byteLength:bytes.length,sha256:sha(bytes)},inspection,inspectionNetwork:{method:'POST',path:'/api/v1/workflows/inspect',status:200,observedAt,request:{observer:'playwright-request-metadata-and-product-response',fileSha256:null,byteLength:null,postDataState:'file-bytes-unavailable-in-observer',contentType:'multipart/form-data'},response:inspection}},origin={outputs:[{sha256:sha(bytes)}]};
  validatePortableCapture(workflow,bytes,origin,contract.originalReport,0);
  for(const mutate of [value=>value.inspectUploadSha256=sha('other'),value=>value.browserFile.sha256=sha('other'),value=>value.inspectionNetwork.path='/api/v1/fake',value=>value.inspectionNetwork.response={},value=>value.inspection.workflow.advanced.mossImageEditExecutionReport.adapter_path='/private/leak']){
    const changed=structuredClone(workflow);mutate(changed);assert.throws(()=>validatePortableCapture(changed,bytes,origin,contract.originalReport,0));
  }
  assert.throws(()=>validatePortableCapture(workflow,rgbPng(1,1,Buffer.from([4,5,6])),origin,contract.originalReport,0));
});
test('lexical scalar proof targets only actual training shift and has no unrelated malformed JSON',()=>{
  for(const token of ['NaN','Infinity','-Infinity'])validateLexicalNonfinite(`{"advanced":{"mossImage":{"training_shift":${token}}}}`);
  for(const raw of ['{"prompt":"NaN","advanced":{"mossImage":{"training_shift":null}}}','{"training_shift":NaN}','{"advanced":{"mossImage":{"training_shift":NaN}},"bad":}','{"advanced":{"mossImage":{"training_shift":NaN,"other":Infinity}}}'])assert.throws(()=>validateLexicalNonfinite(raw));
});
test('current adapter locator and native report preserve the exact approved config and weight identities',async()=>{
  const manifest=JSON.parse(stripJsoncComments(await readFile(resolve(sourceRoot,'config/manifests/builtin.models.jsonc'),'utf8')));
  const revision='3be9c40037e1b89b17aa85ff6f8ee805872aa558',weight='17a48b8c605dcffdedb92a2b887d18cf316505be90802717669fb6556098dcdf',configSha='b6b471cf7bc77b0874af03be618f8a06b6997d92b35de12c47a06b292ade2669';
  for(const id of ['moss_image_base','moss_image_sft']) {
    const component=manifest.models.find(model=>model.id===id).conditionalComponents.find(component=>component.componentId==='moss-image-edit');
    assert.equal(component.revision,revision);assert.equal(component.weightsSha256,weight);assert.equal(component.license,'unspecified');
    assert.deepEqual(component.files,['adapter_config.json','adapter_model.safetensors']);
  }
  const row={backend:'cuda',variant:'base',refs:['B'],output:'images',outputCount:1};
  const request={width:16,height:16,seed:7,advanced:{steps:1,guidanceScale:1,schedulerShift:1,mossImage:{},mossImageEdit:{reference_max_edge:64,use_kv_cache:true}}};
  const report={adapter_repository:'chinoll/Moss-Image-Edit',adapter_revision:revision,adapter_sha256:weight,config_sha256:configSha,applied_projections:252,edit_contract:'ref-prefix-only-t0__target-text-original-moss-v1',references:[{}],reference_max_edge:64,use_kv_cache:true,
    base:{execution_device:'cuda',resolved:{variant:'base',steps:1,guidance:1,shift:1,training_shift:1,max_sequence_length:4096},output_kind:'images',latent_shape:[1,128,1,1],samples:[{seed:7,noise_sha256:'a'.repeat(64)}]}};
  validateEditReport(row,request,report);
  for(const changed of [{adapter_revision:'e19b6ab0fb8fd010cb6f1c7f5b9bfe152252df11'},{adapter_sha256:'0'.repeat(64)},{config_sha256:'0'.repeat(64)}])assert.throws(()=>validateEditReport(row,request,{...report,...changed}),/exact published edit adapter/);
});
test('installer projection is exactly the integrated pinned config and adapter weights on both variants',()=>{
  const component={componentId:'moss-image-edit',requiredFor:['moss_image_edit'],repo:'chinoll/Moss-Image-Edit',revision:'e'.repeat(40),files:['adapter_config.json','adapter_model.safetensors'],weightsSha256:'f'.repeat(64)},prefix=`hub/models--chinoll--Moss-Image-Edit/snapshots/${component.revision}/`,config={editAdapterFiles:component.files.map(file=>prefix+file),packageFiles:[{file:prefix+'adapter_model.safetensors',sha256:component.weightsSha256}]},manifest={models:['moss_image_base','moss_image_sft'].map(id=>({id,conditionalComponents:[structuredClone(component)]}))};
  validatePackagePlan(config,manifest);assert.throws(()=>validatePackagePlan({...config,editAdapterFiles:[config.editAdapterFiles[1]]},manifest),/both pinned/);assert.throws(()=>validatePackagePlan({...config,packageFiles:[{file:prefix+'adapter_model.safetensors',sha256:'0'.repeat(64)}]},manifest),/weight SHA/);const changed=structuredClone(manifest);changed.models[1].conditionalComponents[0].revision='a'.repeat(40);assert.throws(()=>validatePackagePlan(config,changed),/identity differs/);
});
test('bounded cold context and warm dispatch may precede a real scoped budget refusal without model allocation',()=>{
  for(const events of [['cache_context'],['cache_dispatch'],['cache_context','cache_dispatch']]) {
    const after=journal(['startup_begin','ready_after_bounded_startup_probe','heartbeat','job_begin',...events,'job_end','heartbeat']),options={beforeAt:at(3),afterAt:at(6+events.length),jobId:'actual-job'};
    const proof=noLoadComparison(journal(),after,identity,options);assert.equal(proof.after.allocationCapableEntries,0);assert.equal(proof.after.providerLoads,0);assert.equal(proof.after.providerRuns,0);assert.equal(proof.after.boundedContextEntries,events.includes('cache_context')?1:0);assert.equal(proof.after.cacheDispatchEntries,events.includes('cache_dispatch')?1:0);
    assert.throws(()=>noLoadComparison(journal(),after,identity,{...options,jobId:null}),/HTTP refusal/);assert.throws(()=>noLoadComparison(journal(),after,identity,{...options,jobId:'different-job'}),/scope/);
  }
  assert.throws(()=>noLoadComparison(journal(),journal(['startup_begin','ready_after_bounded_startup_probe','heartbeat','heartbeat']),identity,{beforeAt:at(3),afterAt:at(4),jobId:'actual-job'}),/scope/);
});
test('device-budget queued failure retains actual response/job and cannot replace strict malformed HTTP refusal',async()=>{
  const root=await temporary('worker-refusal'),request={projectId:'actual-project',model:'moss_image_base'},queued={id:'actual-budget-job',projectId:'actual-project',payload:request,status:'queued'},specification={expectedStatus:201,expectedTerminal:'failed',errorIncludes:['additional bytes']};
  const api={async request(path,options){assert.equal(path,'/api/v1/image/jobs');const body=JSON.stringify(queued);Object.assign(options.observation,{status:201,body,byteLength:Buffer.byteLength(body),sha256:sha(body)});return queued;}};
  assert.equal((await submitGuard(api,specification,request,root)).id,queued.id);assert.equal((await json(resolve(root,'queued-response.json'))).http.status,201);
  const endpoint={workerId:'actual-worker',workerGpuId:'1'},job={...queued,status:'failed',backend:'1',workerId:endpoint.workerId,startedAt:at(1),updatedAt:at(2),result:{assets:[]},error:'selected package needs additional bytes; reduce dimensions'};
  validateWorkerRefusal(specification,request,job,endpoint,'cuda');
  for(const change of [{status:'completed'},{workerId:'another'},{error:'shape error'},{result:{assets:[{id:'unexpected-output'}]}},{payload:{...request,model:'moss_image_sft'}}])assert.throws(()=>validateWorkerRefusal(specification,request,{...job,...change},endpoint,'cuda'));
  const strict=await temporary('strict-http');await assert.rejects(()=>submitGuard(api,{expectedStatus:400,errorIncludes:['malformed']},request,strict),error=>error.jobId===queued.id);assert.equal((await json(resolve(strict,'unexpected-enqueue.json'))).id,queued.id);
});
test('corrupt adapter worker preparation can fail after enqueue with no provider entry and no outputs',()=>{
  const request={projectId:'actual-project',model:'moss_image_sft'},specification={expectedStatus:201,expectedTerminal:'failed',errorIncludes:['MOSS edit frozen component identity mismatch: adapter_model.safetensors']},endpoint={workerId:'actual-worker',workerGpuId:'1'};
  const job={id:'actual-job',projectId:request.projectId,payload:request,status:'failed',backend:'1',workerId:endpoint.workerId,startedAt:at(1),updatedAt:at(2),result:{assets:[]},error:'Invalid payload: MOSS edit frozen component identity mismatch: adapter_model.safetensors'};
  validateWorkerRefusal(specification,request,job,endpoint,'cuda');
  const after=journal(['startup_begin','ready_after_bounded_startup_probe','heartbeat','job_begin','job_end','heartbeat']);assert.equal(noLoadComparison(journal(),after,identity,{beforeAt:at(3),afterAt:at(6),jobId:job.id}).after.providerLoads,0);
  assert.throws(()=>validateWorkerRefusal(specification,request,{...job,error:'some other native error'},endpoint,'cuda'));
  const provider=journal(['startup_begin','ready_after_bounded_startup_probe','heartbeat','job_begin','provider_load','job_end','heartbeat']);assert.throws(()=>noLoadComparison(journal(),provider,identity,{beforeAt:at(3),afterAt:at(7),jobId:job.id}),/allocation-capable/);
});
const hostSourceFixture=()=>({
  advanced:`pub struct EditAdvanced {\npub positive: P,\npub negative: N,\npub negative_prompts: T,\npub noise: Q,\npub output: O,\n}\npub struct EmbeddingsRef {\npub values: T,\npub mask: T,\n}\nmeta.shape[0] > 8; if !(1..=4096).contains(&context); meta.shape[2] != 4096; filter(|n| *n > 0 && *n <= 8); images,128,native.height as usize / 16,native.width as usize / 16; m.dtype != "uint8" || m.shape != meta.shape[..2]; .metadata.as_ref().expect("resolved metadata").byte_length; *n <= tensor_artifact::MAX_REQUEST_BYTES; pub fn references(options: &EditAdvanced) -> Vec<&TensorRef> {\n for e in [&options.positive, &options.negative] {refs.push(&e.values);refs.push(mask);} refs.push(noise);\n}`,
  moss:'if value > 4096',tensor:`MAX_DATA_BYTES: usize = 512 * 1024 * 1024; MAX_HEADER_BYTES: usize = 4096; MAX_REQUEST_BYTES: usize = 1280 * 1024 * 1024 + 128 * 1024; byte_length: data; "<f4" => "float32"; "|u1" | "<u1" => "uint8";\nlet (n, start) = match prefix[6] {\n1 => (size,10),\n2 | 3 => (size,12,),\n_ => return Err("unsupported NPY version"),\n    };`,
  manifest:JSON.stringify({models:['moss_image_base','moss_image_sft'].map(id=>({id,limits:{maxDimension:4096}}))})
});
test('exact source-bound host sum is unreachable, including all dtypes/roles/framing without reference bytes',()=>{
  const source=hostSourceFixture(),proof=deriveHostTensorBound(source);assert.equal(proof.maximumDataSum,1342242816);assert.equal(proof.maxRequestBytes,1342308352);assert.equal(proof.headroom,65536);assert.equal(proof.maximumFileFraming,20540);assert.equal(proof.maximumOwnedHostBytes,1879117836);assert.ok(proof.maximumDataSum+proof.maximumFileFraming<proof.maxRequestBytes);assert.ok(proof.excludes.includes('reference image pixels'));
  let maximum=0;for(let batch=1;batch<=8;batch++)for(let count=1;batch*count<=8;count++)maximum=Math.max(maximum,2*batch*4096*4096*4+2*batch*4096+(batch*count)*128*256*256*4);assert.equal(maximum,proof.maximumDataSum);
  for(const mutate of [value=>value.advanced=value.advanced.replace('meta.shape[0] > 8','meta.shape[0] > 9'),value=>value.tensor=value.tensor.replace('1280 * 1024','1279 * 1024'),value=>value.advanced=value.advanced.replace('pub output: O,','pub output: O,\npub extra_tensor: X,'),value=>value.manifest=value.manifest.replaceAll('4096','8192')]){const changed=structuredClone(source);mutate(changed);assert.throws(()=>deriveHostTensorBound(changed));}
});
test('integrated advanced tensor source recomputes the exact host envelope and rejects source drift',async()=>{
  const proof=await hostTensorInvariant(sourceRoot),text={};
  assert.equal(proof.maximumDataSum,1342242816);assert.equal(proof.headroom,65536);
  for(const [key,identity] of Object.entries(proof.sources))text[key]=await readFile(resolve(sourceRoot,identity.file),'utf8');
  assert.deepEqual(deriveHostTensorBound(text).observed,proof.observed);
  const changed={...text,advanced:text.advanced.replace('meta.shape[0] > 8','meta.shape[0] > 9')};
  assert.notEqual(changed.advanced,text.advanced);assert.throws(()=>deriveHostTensorBound(changed),/envelope changed/);
});
test('host slot submits exact small over-budget NPY declaration and retains actual import refusal/no new asset',async()=>{
  const root=await temporary('host-budget'),bytes=hostBudgetHeader(),specification=hostGuardSpec();await writeFile(resolve(root,'request.npy'),bytes);assert.equal(bytes.length,128);const header=bytes.subarray(10).toString();assert.match(header,/\(1, 134217729\)/);assert.equal(134217729*4,512*1024*1024+4);let calls=0;
  const api={async request(path,options={}){if(options.method!=='POST')return [{id:'existing-actual-asset'}];calls++;assert.equal(path,'/api/v1/projects/actual-project/assets');assert.deepEqual(Buffer.from(await options.body.get('file').arrayBuffer()),bytes);const error=Error('actual source import refusal');error.status=400;error.responseBody='tensor payload exceeds budget';throw error;}};
  await submitGuard(api,specification,{projectId:'actual-project',upload:specification.upload},root);assert.equal(calls,1);assert.equal((await json(resolve(root,'response.json'))).body,'tensor payload exceeds budget');assert.deepEqual(await json(resolve(root,'assets-before.json')),await json(resolve(root,'assets-after.json')));
  const changed=await temporary('host-created');await writeFile(resolve(changed,'request.npy'),bytes);let reads=0;const wrong={async request(path,options={}){if(options.method!=='POST')return reads++?[{id:'unexpected-created'}]:[];const error=Error('refusal');error.status=400;error.responseBody='tensor payload exceeds budget';throw error;}};await assert.rejects(()=>submitGuard(wrong,specification,{projectId:'actual-project'},changed),/created an asset/);
});

test('adapter disk backup restores through guard failure and never follows replacement links',async t=>{
  const root=await temporary('restore-links');t.after(()=>rm(root,{recursive:true,force:true}));
  const home=resolve(root,'hf'),file='hub/edit.safetensors',path=resolve(home,file),original=Buffer.from('exact owned adapter'),foreign=resolve(root,'shared-target');
  await mkdir(resolve(home,'hub'),{recursive:true});await writeFile(path,original);await writeFile(foreign,'shared bytes');
  const config={root,hfHome:home,candidateId:'inert',editAdapterFiles:[file],packageFiles:[{file,sha256:sha(original)}]},corrupt=Buffer.from('corrupt');
  const mutation=await adapterRefusalState(config,'corrupt-adapter',{adapterFile:file,corruptBytesBase64:corrupt.toString('base64'),corruptSha256:sha(corrupt)});
  try{await unlink(path);await symlink(foreign,path);throw Error('inert guard failure');}catch(error){assert.match(error.message,/guard failure/);}finally{await mutation.restore();}
  assert.deepEqual(await readFile(path),original);assert.equal(await readFile(foreign,'utf8'),'shared bytes');
  assert.deepEqual(await readdir(resolve(home,'hub')),['edit.safetensors']);await mutation.restore();
  await assert.rejects(()=>adapterRefusalState(config,'corrupt-adapter',{adapterFile:file,corruptBytesBase64:corrupt.toString('base64'),corruptSha256:'0'.repeat(64)}),/immutable corrupt/);
  assert.deepEqual(await readFile(path),original);assert.deepEqual(await readdir(resolve(home,'hub')),['edit.safetensors']);
});
test('adapter refusal rejects shared symlink, hardlink and escaped parent projections without mutation',async t=>{
  const root=await temporary('unsafe-adapter');t.after(()=>rm(root,{recursive:true,force:true}));
  const home=resolve(root,'hf'),shared=resolve(root,'shared'),bytes=Buffer.from('shared immutable adapter');await mkdir(home);await mkdir(shared);await writeFile(resolve(shared,'adapter'),bytes);
  for(const kind of ['symlink','hardlink','parent']) {
    const file=kind==='parent'?'alias/adapter':kind,path=resolve(home,file);
    if(kind==='symlink')await symlink(resolve(shared,'adapter'),path);
    else if(kind==='hardlink')await link(resolve(shared,'adapter'),path);
    else await symlink(shared,resolve(home,'alias'));
    const config={root,hfHome:home,candidateId:'inert',editAdapterFiles:[file],packageFiles:[{file,sha256:sha(bytes)}]};
    await assert.rejects(()=>adapterRefusalState(config,'missing-adapter',{adapterFile:file}),/private regular|escaped/);
    assert.deepEqual(await readFile(resolve(shared,'adapter')),bytes);
  }
});

async function privateHfFixture(t,backend='mlx') {
  const base=await realpath(await temporary('private-handoff'));t.after(()=>rm(base,{recursive:true,force:true}));
  const parent=resolve(base,'moss-sc-26752'),buildRoot=resolve(parent,`moss-build-123-1-${backend}`),root=resolve(parent,`moss-execute-123-1-${backend}`);
  await mkdir(buildRoot,{recursive:true});await mkdir(root);
  const common={sourceRoot:resolve(base,'checkout/source'),candidateId:'candidate',sourceSha:'a'.repeat(40),inferenceSha:'b'.repeat(40),backend,packageSource:resolve(base,'shared'),packageFiles:Array.from({length:43},(_,i)=>({file:`hub/model/file-${i}`,sha256:sha(`bytes-${i}`)}))};
  const build={...common,root:buildRoot,hfHome:resolve(buildRoot,'hf')},run={...common,root,hfHome:resolve(root,'hf')};
  for(const [i,item] of common.packageFiles.entries()){await mkdir(resolve(build.hfHome,'hub/model'),{recursive:true});await writeFile(resolve(build.hfHome,item.file),`bytes-${i}`);}
  await save(resolve(buildRoot,'runtime.json'),build);const owner=await sealPrivateHf(build),installed={hfHome:build.hfHome,files:owner.files};
  return {base,build,run,installed,owner};
}
test('original stopped CUDA HF is adopted into a new run only from its sealed exact owner',async t=>{
  const {base,build,run,installed}=await privateHfFixture(t,'cuda'),currentRoot=resolve(base,'moss-sc-26752/moss-execute-456-1-cuda');
  await mkdir(currentRoot);const current={...run,root:currentRoot,hfHome:resolve(currentRoot,'hf')};
  await assert.rejects(()=>adoptOriginalPrivateHf({...current,candidateId:'other'},installed,{},123,1),/another candidate\/source/);
  assert.ok((await lstat(build.hfHome)).isDirectory());
  const receipt=await adoptOriginalPrivateHf(current,installed,{sha256:sha('original native packages')},123,1);
  assert.equal(receipt.operation,'same-volume-original-stopped-CUDA-HF-adoption');assert.equal(receipt.from,build.hfHome);assert.equal(receipt.to,current.hfHome);
  assert.deepEqual(receipt.before,receipt.after);await assert.rejects(()=>lstat(build.hfHome),{code:'ENOENT'});
  await assert.rejects(()=>adoptOriginalPrivateHf(current,installed,{},123,1),/already exists/);
});
test('all 43 installed packages transfer by rename, retaining original receipts and exclusive adapter ownership',async t=>{
  const {build,run,installed}=await privateHfFixture(t),original=await readFile(resolve(build.root,'hf-owner.json')),before=await lstat(resolve(build.hfHome,installed.files[0].file));
  const receipt=await transferPrivateHf(run,installed,{sha256:sha('original packages receipt'),byteLength:12});
  assert.equal(receipt.before.length,43);assert.deepEqual(receipt.before,receipt.after);assert.equal((await lstat(resolve(run.hfHome,installed.files[0].file))).ino,before.ino);assert.equal((await lstat(resolve(run.hfHome,installed.files[0].file))).nlink,1);
  await assert.rejects(()=>lstat(build.hfHome),{code:'ENOENT'});assert.deepEqual(await readFile(resolve(build.root,'hf-owner.json')),original);assert.equal(installed.hfHome,build.hfHome);
  run.editAdapterFiles=[installed.files[0].file];const mutation=await adapterRefusalState(run,'missing-adapter',{adapterFile:installed.files[0].file});await mutation.restore();
  await releasePrivateHf(run);await assert.rejects(()=>lstat(run.hfHome),{code:'ENOENT'});assert.equal((await json(resolve(run.root,'hf-final-integrity.json'))).files.length,43);assert.deepEqual(await readFile(resolve(build.root,'hf-owner.json')),original);
});
test('HF transfer rejects existing destination, wrong run/source/pin/candidate, changed bytes and escaped paths before mutation',async t=>{
  for(const kind of ['destination','run','source','pin','candidate','bytes','escape','link','hardlink','ancestor','descendant','runtime','inventory']) {
    const {base,build,run,installed}=await privateHfFixture(t);let config=run,input=installed;
    if(kind==='destination')await mkdir(run.hfHome);
    if(kind==='run')input={...installed,hfHome:resolve(base,'moss-sc-26752/moss-build-124-1-mlx/hf')};
    if(kind==='source')config={...run,sourceSha:'c'.repeat(40)};
    if(kind==='pin')config={...run,inferenceSha:'c'.repeat(40)};
    if(kind==='candidate')config={...run,candidateId:'other'};
    if(kind==='bytes')await writeFile(resolve(build.hfHome,installed.files[0].file),'tampered');
    if(kind==='escape')config={...run,root:resolve(base,'unrelated/moss-execute-123-1-mlx'),hfHome:resolve(base,'unrelated/moss-execute-123-1-mlx/hf')};
    if(kind==='link'){await symlink(build.hfHome,resolve(run.root,'linked'));input={...installed,hfHome:resolve(run.root,'linked')};}
    if(kind==='descendant')await symlink(resolve(base,'shared'),resolve(build.hfHome,'extra-link'));
    if(kind==='ancestor'){await rm(resolve(build.hfHome,'hub/model'),{recursive:true});await mkdir(resolve(base,'external'));await symlink(resolve(base,'external'),resolve(build.hfHome,'hub/model'));}
    if(kind==='hardlink')await link(resolve(build.hfHome,installed.files[0].file),resolve(base,'shared-link'));
    if(kind==='runtime')await writeFile(resolve(build.root,'runtime.json'),JSON.stringify({...build,port:1}));
    if(kind==='inventory')input={...installed,files:installed.files.slice(1)};
    await assert.rejects(()=>transferPrivateHf(config,input,{}),undefined,kind);assert.ok(await lstat(build.hfHome));
  }
});
test('HF release refuses altered bytes, wrong ownership and escaped/shared cleanup targets',async t=>{
  const {build,run,installed}=await privateHfFixture(t);await transferPrivateHf(run,installed,{});
  for(const config of [{...run,hfHome:build.hfHome},{...run,candidateId:'other'},{...run,root:resolve(run.root,'..'),hfHome:resolve(run.root,'../hf')}])await assert.rejects(()=>releasePrivateHf(config));
  await writeFile(resolve(run.hfHome,installed.files[0].file),'corrupt');await assert.rejects(()=>releasePrivateHf(run),/hash mismatch/);assert.ok(await lstat(run.hfHome));await assert.rejects(()=>lstat(resolve(run.root,'hf-released.json')),{code:'ENOENT'});
});
test('durable MOSS storage lives outside checkout cleanup and release follows owned worker shutdown',async()=>{
  const workflow=await readFile(resolve(sourceRoot,'.github/workflows/moss-edit-terminal.yml'),'utf8'),controller=await readFile(resolve(sourceRoot,'scripts/moss-edit-terminal-actions.mjs'),'utf8');
  for(const line of workflow.split('\n').filter(line=>/^\s+MOSS_(ROOT|BUILDS|BUNDLE):/.test(line)))assert.match(line,/github.workspace.*\/\.\.\/moss-sc-26752\/moss-|format\('\{0\}\/\.\.\/moss-sc-26752', github.workspace\)|\/Volumes\/Models\/sceneworks-terminal-sc-26752\/moss-/);
  assert.match(controller,/await transferPrivateHf\(config/);assert.doesNotMatch(controller,/isolatePackages\(config,\{installedInventory/);
  assert.ok(controller.indexOf('await stack.stop();stopped=true;')<controller.indexOf('await releasePrivateHf(config)'));assert.ok(controller.indexOf("await exportState(config,resolve(config.root,'handoff'))")<controller.indexOf('await releasePrivateHf(config)'));
});

test('same-volume private handoff follows original authenticated listener path without assuming routing labels',async t=>{
  const {base,run,build,installed}=await privateHfFixture(t),oldOwner=await readFile(resolve(build.root,'hf-owner.json'));
  const current={...run,sourceRoot:resolve(base,'other-listener/checkout/source'),root:resolve(base,'other-listener/moss-sc-26752/moss-execute-123-1-mlx'),hfHome:resolve(base,'other-listener/moss-sc-26752/moss-execute-123-1-mlx/hf')};
  await mkdir(current.root,{recursive:true});const receipt=await transferPrivateHf(current,installed,{});assert.equal(receipt.from,build.hfHome);assert.equal(receipt.to,current.hfHome);assert.deepEqual(await readFile(resolve(build.root,'hf-owner.json')),oldOwner);await assert.rejects(()=>lstat(build.hfHome),{code:'ENOENT'});
});

async function installerPointerFixture(t) {
  const fixture=await privateHfFixture(t),config=fixture.build,revision='3be9c40037e1b89b17aa85ff6f8ee805872aa558',repo='hub/models--chinoll--Moss-Image-Edit',prefix=`${repo}/snapshots/${revision}`;
  config.editAdapterFiles=[`${prefix}/adapter_config.json`,`${prefix}/adapter_model.safetensors`];
  const originals=[];
  for(const [i,file] of config.editAdapterFiles.entries()) {
    const bytes=Buffer.from(`installer-owned-${i}`),blob=resolve(config.hfHome,repo,'blobs',i?'a'.repeat(64):'b'.repeat(40)),path=resolve(config.hfHome,file);
    await mkdir(resolve(config.hfHome,repo,'blobs'),{recursive:true});await mkdir(resolve(config.hfHome,prefix),{recursive:true});await writeFile(blob,bytes);await symlink(`../../blobs/${i?'a'.repeat(64):'b'.repeat(40)}`,path);
    config.packageFiles.push({file,sha256:sha(bytes)});originals.push({path,blob,bytes});
  }
  return {...fixture,config,originals};
}
test('actual installer-shaped private snapshot links materialize to exclusive regular files with unchanged blobs and receipts',async t=>{
  const {config,originals}=await installerPointerFixture(t),owner=await readFile(resolve(config.root,'hf-owner.json'));
  const operations=await materializePrivateAdapter(config);assert.equal(operations.length,2);
  for(const {path,blob,bytes} of originals){assert.equal((await lstat(path)).isSymbolicLink(),false);assert.equal((await lstat(path)).nlink,1);assert.notEqual((await lstat(path)).ino,(await lstat(blob)).ino);assert.deepEqual(await readFile(path),bytes);assert.deepEqual(await readFile(blob),bytes);}
  assert.deepEqual(await readFile(resolve(config.root,'hf-owner.json')),owner);assert.equal((await json(resolve(config.root,'adapter-materialization.json'))).operations.length,2);
});
test('adapter materialization refuses external blobs, hash mismatches, unexpected links, shared blob hardlinks and escaped ownership',async t=>{
  for(const kind of ['external','hash','unexpected','hardlink','ancestor','ownership','unexpected-pair']) {
    const {base,config,originals}=await installerPointerFixture(t),first=originals[0];let selected=config;
    if(kind==='external'){await unlink(first.path);await writeFile(resolve(base,'b'.repeat(40)),first.bytes);await symlink(resolve(base,'b'.repeat(40)),first.path);}
    if(kind==='hash')await writeFile(first.blob,'wrong bytes');
    if(kind==='unexpected')await symlink(first.blob,resolve(config.hfHome,'unexpected-link'));
    if(kind==='hardlink')await link(first.blob,resolve(base,'shared-blob'));
    if(kind==='ancestor'){const blobDirectory=resolve(first.blob,'..');await rm(blobDirectory,{recursive:true});await mkdir(resolve(base,'external-blobs'));await symlink(resolve(base,'external-blobs'),blobDirectory);}
    if(kind==='ownership')selected={...config,sourceRoot:resolve(base,'other/checkout/source')};
    if(kind==='unexpected-pair')selected={...config,editAdapterFiles:[config.editAdapterFiles[0],'../outside']};
    await assert.rejects(()=>materializePrivateAdapter(selected),undefined,kind);assert.equal((await lstat(first.path)).isSymbolicLink(),true);await assert.rejects(()=>lstat(resolve(config.root,'adapter-materialization.json')),{code:'ENOENT'});
  }
});
test('regular installed adapters remain exclusive and malformed regular bytes are refused',async t=>{
  const {config,originals}=await installerPointerFixture(t);for(const row of originals){await unlink(row.path);await writeFile(row.path,row.bytes);}
  assert.deepEqual(await materializePrivateAdapter(config),[]);await writeFile(originals[0].path,'wrong');await assert.rejects(()=>materializePrivateAdapter(config),/mismatch/);
});
test('canonical artifact environment removes dot segments and proves workspace aliases before exporting paths',async t=>{
  const base=await realpath(await temporary('canonical-env'));t.after(()=>rm(base,{recursive:true,force:true}));const workspace=resolve(base,'workspace/repository');await mkdir(workspace,{recursive:true});await symlink(workspace,resolve(base,'alias'));
  const env={GITHUB_WORKSPACE:workspace,MOSS_ROOT:resolve(workspace,'..')+'/repository/../moss-sc-26752/moss-build-123-1-mlx',MOSS_BUILDS:workspace+'/../moss-sc-26752/moss-builds-123-1-build-mlx',MOSS_BUNDLE:workspace+'/../moss-sc-26752/moss-bundle-123-1',GITHUB_ENV:resolve(base,'env')};
  const output=await exportMossEnvironment(env);assert.equal(output.MOSS_ROOT,resolve(workspace,'../moss-sc-26752/moss-build-123-1-mlx'));for(const value of Object.values(output))assert.ok(!value.split(/[\\/]/).some(part=>part==='.'||part==='..'));assert.match(await readFile(env.GITHUB_ENV,'utf8'),/MOSS_BUILDS=/);
  const alias={...env,GITHUB_WORKSPACE:resolve(base,'alias'),MOSS_ROOT:resolve(base,'alias')+'/../moss-sc-26752/moss-build-123-1-mlx',MOSS_BUILDS:undefined,MOSS_BUNDLE:undefined};assert.equal((await normalizeMossEnvironment(alias)).MOSS_ROOT,output.MOSS_ROOT);
  await assert.rejects(()=>normalizeMossEnvironment({...env,MOSS_ROOT:resolve(base,'unrelated/moss-build-123-1-mlx')}),/escaped/);
  await rm(resolve(workspace,'../moss-sc-26752'),{recursive:true});await mkdir(resolve(base,'unrelated'));await symlink(resolve(base,'unrelated'),resolve(workspace,'../moss-sc-26752'));await assert.rejects(()=>normalizeMossEnvironment(env),/linked ancestors/);
});
test('every artifact job normalizes paths before its first operation and installer materialization follows shutdown',async()=>{
  const workflow=await readFile(resolve(sourceRoot,'.github/workflows/moss-edit-terminal.yml'),'utf8'),controller=await readFile(resolve(sourceRoot,'scripts/moss-edit-terminal-actions.mjs'),'utf8');
  for(const name of ['upload-inputs','build-native','execute-mlx','execute-cuda','verify-retained']) {
    // Select by job body, independently of the display name of its first artifact step.
    const body=workflow.split(/(?=^  [a-z][a-z-]*:\n)/m).find(body=>body.startsWith(`  ${name}:`));assert.ok(body,name);
    const artifact=body.search(/uses: actions\/(?:download|upload)-artifact/),normalize=body.indexOf('moss-edit-terminal-actions.mjs normalize-env');assert.ok(normalize>=0&&normalize<artifact,name);
  }
  assert.doesNotMatch(workflow,/path: \$\{\{ inputs.upload_path/);assert.match(workflow,/path: \$\{\{ env.MOSS_UPLOAD_PATH/);
  assert.ok(controller.indexOf('}finally{await stack.stop();}')<controller.indexOf('await materializePrivateAdapter(config)'));assert.ok(controller.indexOf('await materializePrivateAdapter(config)')<controller.indexOf("await save(resolve(out,'packages.json')"));
});

async function installerHardlinkFixture(t) {
  const fixture=await installerPointerFixture(t);for(const row of fixture.originals){await unlink(row.path);await link(row.blob,row.path);}return fixture;
}
test('runtime-proven exact two private installer hardlinks become independent exclusive files',async t=>{
  const {config,originals}=await installerHardlinkFixture(t),before=await lstat(originals[0].path,{bigint:true});assert.equal(before.nlink,2n);
  const operations=await materializePrivateAdapter(config);assert.equal(operations.length,2);
  for(const row of originals){const file=await lstat(row.path,{bigint:true}),blob=await lstat(row.blob,{bigint:true});assert.equal(file.nlink,1n);assert.equal(blob.nlink,1n);assert.notEqual(file.ino,blob.ino);assert.deepEqual(await readFile(row.path),row.bytes);assert.deepEqual(await readFile(row.blob),row.bytes);}
  for(const operation of operations){assert.equal(operation.operation,'materialize-proven-private-installer-hardlink');assert.equal(operation.originalLinkIdentity.nlink,'2');assert.equal(operation.originalBlobIdentity.nlink,'2');assert.equal(operation.originalLinkIdentity.ino,operation.originalBlobIdentity.ino);assert.equal(operation.sameIdentityEntries.length,2);assert.equal(operation.afterSnapshot.nlink,'1');assert.equal(operation.afterBlob.nlink,'1');}
  await privateHfIntegrity(config);const row=originals[0],mutation=await adapterRefusalState(config,'missing-adapter',{adapterFile:config.editAdapterFiles[0]});await mutation.restore();assert.deepEqual(await readFile(row.path),row.bytes);
});
test('hardlink normalization refuses external or third aliases, wrong hashes and unrelated nonexclusive files before replacement',async t=>{
  for(const kind of ['third-external','third-internal','outside-pair','wrong-hash','unrelated','wrong-blob']) {
    const {base,config,originals}=await installerHardlinkFixture(t),first=originals[0],second=originals[1];
    if(kind==='third-external')await link(first.blob,resolve(base,'outside-third'));
    if(kind==='third-internal')await link(first.blob,resolve(config.hfHome,'third-alias'));
    if(kind==='outside-pair'){await unlink(first.blob);await link(first.path,resolve(base,'outside-second'));}
    if(kind==='wrong-hash')await writeFile(first.blob,'wrong bytes');
    if(kind==='unrelated')await link(resolve(config.hfHome,config.packageFiles[0].file),resolve(base,'unrelated-shared'));
    if(kind==='wrong-blob'){await unlink(first.blob);await link(first.path,resolve(config.hfHome,'f'.repeat(64)));}
    const before=await Promise.all(originals.map(row=>lstat(row.path,{bigint:true})));
    await assert.rejects(()=>materializePrivateAdapter(config),undefined,kind);
    for(const [index,row] of originals.entries()){const after=await lstat(row.path,{bigint:true});assert.equal(after.ino,before[index].ino,kind);assert.equal(after.nlink,before[index].nlink,kind);}
    assert.equal((await lstat(second.path)).nlink,2);assert.equal((await lstat(second.path)).ino,(await lstat(second.blob)).ino);await assert.rejects(()=>lstat(resolve(config.root,'adapter-materialization.json')),{code:'ENOENT'});
  }
});
test('exclusive-file refusals identify the exact leaf, type and unrounded link identity',async t=>{
  const {base,build}=await privateHfFixture(t),path=resolve(build.hfHome,build.packageFiles[0].file);await link(path,resolve(base,'external-alias'));const info=await lstat(path,{bigint:true});
  await assert.rejects(()=>privateHfIntegrity(build),error=>{assert.match(error.message,/private HF requires exclusive regular files/);const metadata=JSON.parse(error.message.slice(error.message.indexOf('{')));assert.equal(metadata.path,path);assert.equal(metadata.type,'file');assert.equal(metadata.dev,String(info.dev));assert.equal(metadata.ino,String(info.ino));assert.equal(metadata.nlink,'2');return true;});
});

test('exact Mac external HF/artifact storage preserves internal Cargo identity and refuses other roots/platforms',()=>{
  const sourceRoot='/runner/_work/SceneWorks/SceneWorks/source',base=resolve(sourceRoot,'../..','moss-sc-26752'),leaf='moss-build-123-1-mlx';
  for(const root of [resolve(base,leaf),resolve(MAC_MOSS_BASE,leaf)])assert.equal(cargoTarget({root,sourceRoot}),resolve(base,leaf,'target'));
  assert.equal(allowedPrivateParent(MAC_MOSS_BASE,sourceRoot,'darwin'),true);
  for(const platform of ['linux','win32'])assert.equal(allowedPrivateParent(MAC_MOSS_BASE,sourceRoot,platform),false);
  for(const parent of ['/Volumes/Models/other','/Volumes/Models/sceneworks-terminal-sc-26752/escape','/Volumes/Other/sceneworks-terminal-sc-26752'])assert.equal(allowedPrivateParent(parent,sourceRoot,'darwin'),false);
  for(const key of ['MOSS_ROOT','MOSS_BUILDS'])assert.equal(mossArtifactRoute(key,resolve(MAC_MOSS_BASE,leaf),base,'darwin').external,true);
  for(const [key,path,platform]of [['MOSS_BUNDLE',resolve(MAC_MOSS_BASE,leaf),'darwin'],['MOSS_ROOT',resolve(MAC_MOSS_BASE,leaf),'linux'],['MOSS_ROOT',resolve(MAC_MOSS_BASE,'../escape',leaf),'darwin'],['MOSS_ROOT',resolve(MAC_MOSS_BASE,'not-owned'),'darwin']])assert.throws(()=>mossArtifactRoute(key,path,base,platform),/escaped/);
  assert.equal(mossArtifactRoute('MOSS_BUNDLE',resolve(base,'moss-bundle-123-1'),base,'darwin').external,false);
});
