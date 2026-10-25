// Pure CPU/in-process fakes. Never creates a listener, child process, GPU context or model run.
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import os from 'node:os';
import { mkdtemp, mkdir, readFile, writeFile, rm, symlink, cp, access } from 'node:fs/promises';
import { executeNativeComparison,executeFixture,validateComparisonProof,OwnedService,ProcessObservations,loadIdentity,bindCompletedMedia,executeAPICase,verifyAPIBindings,privateEnvironment,authorized,assertControlledConfiguration,requestedAdapterBindings } from './h3-finetune-terminal-runtime.mjs';
import { hashFile,verifyInputs,validateEvidence,validatePeakMetrics,requiredCases,SOURCE,validateBackendReadback,executionPlanDigest } from './h3-finetune-terminal.mjs';
import { buildScenario,caseBook } from './h3-finetune-terminal-cases.mjs';
const pause=()=>new Promise(resolve=>setTimeout(resolve,1));
const sw='a'.repeat(40),inf='b'.repeat(40),semantic='c'.repeat(64);
async function fixture(t,group='capability',check='image',backend='mlx') {
  const root=await mkdtemp(path.join(os.tmpdir(),'sc26691-runtime-unit-fake-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const dir=path.join(root,'case');await mkdir(dir);const native=path.join(root,'native');await mkdir(native);
  const input=path.join(root,'synthetic-checkpoint');await writeFile(input,'unit_fake NOT MODEL WEIGHTS');const digest=await hashFile(input);
  const component=path.join(root,'synthetic-component');await writeFile(component,'unit_fake component');const componentHash=await hashFile(component);
  const scenarioPath=path.join(root,'scenario.json');
  const c={...requiredCases().find(c=>c.id===`${backend}.${group}.${check}`),timeoutSeconds:30,evidence:'evidence.json',inputs:[{path:input,sha256:digest,role:'checkpoint'},{path:component,sha256:componentHash,role:'component'}],loadHostBudgetBytes:100000,loadDeviceBudgetBytes:100000,minimumControlDelta:0.01,controlStream:'video',runtime:{port:43210,gpuId:'unit_fake_gpu',deviceId:'unit_fake_device',scenario:scenarioPath,sampleIntervalMs:1}};
  const p={sceneworks:sw,inference:inf,source:SOURCE,output:root,responsibleTask:'sc-26691'};
  const execution={label:'test',mode:'image_to_video',model:{modelId:'imported_unit_fake',sourcePath:input,sha256:digest,baseModel:'minimax_h3',partition:'fl2va',topology:'full',componentTier:'q4',supportedModes:['image_to_video'],components:[{sha256:componentHash,source:'unit_fake',revision:inf,licenseAcknowledged:true}]},sampling:{contractSource:'unit_fake recipe',steps:50,width:576,height:320,fps:24,duration:5.1667}};
  const parameters={baseUrl:'http://127.0.0.1:43210',projectId:'unit_fake_project',requestedGpu:'unit_fake_gpu',timeoutSeconds:30,seed:42,assets:{imageA:'unit_fake_red',imageB:'unit_fake_blue'},sensitivity:{minimumControlDelta:0.01,maximumBaselineDelta:0.005}};
  const scenario=(group==='capability'||(group==='lifecycle'&&check==='cancel'))?buildScenario(c,parameters,[execution]):{baseUrl:parameters.baseUrl,caseId:c.id,timeoutSeconds:30,steps:[{id:'import',method:'POST',path:'/api/v1/models/import',body:{modelId:'imported_unit_fake',sourcePath:input},expectedStatus:201,assert:[{pointer:'/payload/modelId',equals:'imported_unit_fake'}]},{id:'installed',waitJob:{$capture:'import/id'},assert:[{pointer:'/result/modelId',equals:'imported_unit_fake'}]}]};
  await writeFile(scenarioPath,JSON.stringify(scenario));c.inputs.push({path:scenarioPath,sha256:await hashFile(scenarioPath),role:'scenario'});
  if(group==='capability')c.declaredSourceBindings=[execution];
  return {root,dir,native,input,digest,component,componentHash,p,c,scenario,execution,parameters};
}
function fakeRuntime(f,{wrongListener=false,backend='mlx'}={}) {
  let pid=7000,jobNo=0,serviceDir,activeWorker,heartbeat=Date.now();const children=[],calls=[],jobs=new Map(),assets=new Map();
  const spawnOwned=(argv,env,cwd,logPath)=>{
    const kind=argv[0]===process.execPath?'guardian':env.SCENEWORKS_WORKER_ONLY==='1'?'worker':'api';
    serviceDir=path.dirname(env.SCENEWORKS_CONFIG_DIR);let code=null,logs='';let resolveExit;const done=new Promise(r=>resolveExit=r);
    const child={pid:++pid,kind,env,argv,logPath,exitCode:()=>code,log:()=>logs,append:v=>{logs+=JSON.stringify(v)+'\n';},close:()=>{if(code!==null)return;code=0;resolveExit({code:0,signal:null});},stdin:{end:()=>{calls.push(['end',kind,child.pid]);child.close();if(kind==='guardian')for(const c of children.filter(c=>c.kind==='api'&&c.env.SCENEWORKS_PARENT_PID===String(child.pid)))c.close();}},signal:s=>{assert.equal(s,'SIGTERM');assert.ok([...jobs.values()].every(j=>['completed','canceled','failed'].includes(j.status)),'no signal before cooperative completion');calls.push(['signal',kind,child.pid,s]);child.close();},exit:async()=>{const result=await done;await writeFile(logPath,logs,{flag:'wx'});return result;}};
    children.push(child);calls.push(['spawn',kind,child.pid]);if(kind==='worker')activeWorker=child;return child;
  };
  const deps={proof_kind:'unit_fake',platform:'darwin',sleep:pause,spawnOwned,requireFreePort:async port=>{calls.push(['free',port]);},listenerPid:async()=>wrongListener?99999:children.filter(c=>c.kind==='api'&&c.exitCode()===null).at(-1)?.pid,
    runProcess:async(argv,options)=>{calls.push(['build',argv]);assert.ok(argv.includes('sceneworks-rust-api'));const binary=path.join(options.env.CARGO_TARGET_DIR,'debug','sceneworks-rust-api');await mkdir(path.dirname(binary),{recursive:true});await writeFile(binary,'unit_fake binary, never executable');await writeFile(options.logPath,'unit_fake build log',{flag:'wx'});},
    hardware:async()=>({device:'unit_fake CPU fake, not native hardware',driver:'unit_fake'}),
    decode:async file=>({sha256:await hashFile(file),proof_kind:'unit_fake',streams:[{codec_type:'video',width:576,height:320},{codec_type:'audio',proof_kind:'unit_fake'}]}),
    delta:async(a,b)=> (await readFile(a,'utf8'))===(await readFile(b,'utf8'))?0:0.1,
    sample:async service=>({at:new Date().toISOString(),pid:service.worker.pid,host:100,device:200,deviceCounterAt:new Date().toISOString(),method:backend==='mlx'?'mlx-allocator-cumulative-high-water':'sampled-device-used',deviceId:'unit_fake_device'}),
    fetch:async(url,options)=>{
      calls.push(['request',url.pathname,options.method]);assert.ok(options.headers['X-SceneWorks-Token']);let value,status=200;
      if(url.pathname==='/api/v1/health')value={status:'ok',service:'sceneworks-api'};
      else if(url.pathname==='/api/v1/workers'){const active=[...jobs.values()].find(j=>!['completed','canceled','failed','interrupted'].includes(j.status));value=[{id:activeWorker.env.SCENEWORKS_WORKER_ID,status:active?'busy':'idle',currentJobId:active?.id??null,lastSeenAt:new Date(++heartbeat).toISOString()}];}
      else if(url.pathname==='/api/v1/models/import'){const body=JSON.parse(options.body);const id=`import-${++jobNo}`;value={id,status:'queued',payload:body};jobs.set(id,{id,status:'queued',payload:body,result:{modelId:body.modelId}});status=201;}
      else if(url.pathname==='/api/v1/video/jobs'){
        const body=JSON.parse(options.body),id=`video-${++jobNo}`,assetId=`asset-${jobNo}`,project=path.join(serviceDir,'data','projects','unit_fake.sceneworks');
        await mkdir(project,{recursive:true});const file=`${assetId}.mp4`;await writeFile(path.join(project,file),body.sourceAssetId??body.prompt??'unit_fake_video');
        const asset={id:assetId,projectId:body.projectId,type:'video',file:{path:file},recipe:{rawSettings:{proof_kind:'unit_fake'}}};assets.set(assetId,asset);
        const job={id,status:'completed',projectId:body.projectId,payload:{...body,generationSetId:id},result:{assetIds:[assetId]}};jobs.set(id,job);value={...job,status:'queued'};status=201;
        activeWorker.append({event:'h3_import_prepared_load',jobId:id,provider:'unit_fake_native',backend:'unit_fake_gpu',native_backend:f.c.backend,seal_validated:true,cache_state:'Cold',identity:{checkpointId:'unit_fake_source',planId:'unit_fake_plan',semanticDigest:semantic,assembly:{family:'minimax-h3',partition:'fl2va',sourceEncoding:'unit_fake-convrot',components:{}},sourceFiles:[{path:f.input,sha256:f.digest}]},prepared_sources:[{path:f.input,sha256:f.digest},{path:f.component,sha256:f.componentHash}],operation:body.mode,seed:body.seed,steps:body.advanced.steps,scheduler_shift:null});
      } else if(url.pathname.includes('/assets/'))value=assets.get(url.pathname.split('/').at(-1));
      else if(url.pathname.startsWith('/api/v1/projects/'))value={id:'unit_fake_project',path:path.join(serviceDir,'data','projects','unit_fake.sceneworks')};
      else if(url.pathname.endsWith('/duplicate')){const original=jobs.get(url.pathname.split('/').at(-2));assert.ok(original);return deps.fetch(new URL('/api/v1/video/jobs',url),{...options,body:JSON.stringify(original.payload)});}
      else if(url.pathname.endsWith('/cancel')){const id=url.pathname.split('/').at(-2),job=jobs.get(id);job.cancelRequested=true;if(job.status==='queued')job.status='canceled';value=job;}
      else if(url.pathname.startsWith('/api/v1/jobs/')){value=jobs.get(url.pathname.split('/').at(-1));if(value?.cancelRequested&&!['completed','failed'].includes(value.status))value.status='canceled';else if(value?.status==='queued')value.status='completed';}
      assert.ok(value,`no invented fake API route: ${url.pathname}`);return {status,text:async()=>JSON.stringify(value)};
    }};
  return {deps,children,calls,jobs,assets,spawnOwned,get serviceDir(){return serviceDir;}};
}
test('real service/native entry is refused locally before spawn or transport',async t=>{
  const f=await fixture(t);assert.throws(()=>authorized(f.p,f.c));
  await assert.rejects(new OwnedService(f.p,f.c,f.root,f.native,f.dir).initialize(),/Actions-only|equal/);
});
test('task input ownership is checked before opening an out-of-root path',async t=>{
  const f=await fixture(t);await verifyInputs(f.c,[f.root]);
  await assert.rejects(verifyInputs({inputs:[{path:path.join(f.root,'..','unowned-not-present'),sha256:semantic}]},[f.root]),/outside task staging/);
});
test('native adapter consumes actual named executed comparisons without inventing flat arrays',async t=>{
  const f=await fixture(t,'numeric','convrot');const fixturePath=path.join(f.native,'tests','fixtures','h3_convrot_int8.json');await mkdir(path.dirname(fixturePath),{recursive:true});
  const refs=[];for(const key of ['reference_helper','reference_cuda','reference_producer']){const file=path.join(f.root,key);await writeFile(file,`unit_fake pinned ${key}`);refs.push({path:file,sha256:await hashFile(file),role:'reference'});}
  const fx={cases:[{group:64,dtype:'bf16'},{group:256,dtype:'f32'}],...Object.fromEntries(['reference_helper','reference_cuda','reference_producer'].map((key,i)=>[key,{sha256:refs[i].sha256}]))};await writeFile(fixturePath,JSON.stringify(fx));
  f.c.inputs.push({path:fixturePath,sha256:await hashFile(fixturePath),role:'numeric-fixture'},...refs);f.c.referenceRevision=caseBook.numeric.convrot.referenceRevision;
  const name=caseBook.numeric.convrot.selector;f.c.numericProof={kind:'native-pinned-fixture',package:'mlx-gen-minimax-h3',tests:[name]};
  const commands=[];const runProcess=async(argv,options)=>{commands.push(argv);let text;
    if(argv[0]==='cargo'){const binary=path.join(options.env.CARGO_TARGET_DIR,'integration-unit-fake');await mkdir(path.dirname(binary),{recursive:true});await writeFile(binary,'unit_fake binary NOT executable');text=JSON.stringify({reason:'compiler-artifact',target:{name:'integration'},profile:{test:true},executable:binary});}
    else text=argv.includes('--list')?`${name}: test\nunrelated::case: test\n`:`running 1 test\ntest ${name} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored;\n`;
    await writeFile(options.logPath,text,{flag:'wx'});
  };
  const result=await executeNativeComparison(f.p,f.c,f.root,f.native,f.dir,{proof_kind:'unit_fake',runProcess,hardware:async()=>({device:'unit_fake',driver:'unit_fake'})});
  assert.equal(result.proof.proof_kind,'unit_fake');assert.equal(result.proof.fixtureCases.length,fx.cases.length);assert.ok(!result.evidence.numeric);assert.ok(commands.some(a=>a.includes('--no-run')));assert.ok(commands.some(a=>a.includes('--list')));
  assert.throws(()=>validateComparisonProof(f.c,result.proof,result.log),/unit_fake/);
  assert.equal(validateComparisonProof(f.c,result.proof,result.log,{allowUnitFake:true}).passed,1);
  assert.equal(validateComparisonProof(f.c,result.proof,result.log.replace(`test ${name} ... ok`,`test ${name} ... unit_fake backend diagnostic\nok`),{allowUnitFake:true}).passed,1);
  for(const mutate of [p=>p.tests=[],p=>p.runs[0].passed=0,p=>p.referenceBlobs[0].sha256=semantic]){const p=structuredClone(result.proof);mutate(p);assert.throws(()=>validateComparisonProof(f.c,p,result.log,{allowUnitFake:true}));}
  await assert.rejects(validateEvidence(f.p,f.c,f.dir,result.log),/unit_fake/);
});
test('owned API pipeline binds real-shaped jobs/assets/load events, compares media and emits only unit_fake',async t=>{
  const f=await fixture(t),fake=fakeRuntime(f);const r=await executeAPICase(f.p,f.c,f.root,f.native,f.dir,fake.deps);
  assert.equal(r.evidence.proof_kind,'unit_fake');assert.equal(r.comparisons.length,1);assert.equal(r.comparisons[0].repeatedBaselineDelta,0);assert.equal(r.comparisons[0].interventionDelta,0.1);
  assert.equal(r.evidence.runtimeIdentity.checkpointSha256,f.digest);assert.equal(r.evidence.completedJob.output,r.evidence.av);
  assert.ok(r.evidence.files.includes('service-lifecycle.json'));for(const name of r.evidence.files)await readFile(path.join(f.dir,name));
  validatePeakMetrics(r.evidence.metrics,f.c);
  assert.ok(fake.children.every(c=>c.exitCode()===0));assert.equal(fake.calls.filter(c=>c[0]==='free').length,1);
  await assert.rejects(verifyAPIBindings(f.p,f.c,f.dir,fake.deps.decode,fake.deps.delta),/unit_fake/);
  assert.equal((await verifyAPIBindings(f.p,f.c,f.dir,fake.deps.decode,fake.deps.delta,{allowUnitFake:true})).comparisons,1);
  await assert.rejects(validateEvidence(f.p,f.c,f.dir,'H3_API_ASSERTIONS_EXECUTED=1'),/unit_fake/);
});
test('portable case evidence redecodes all retained outputs without publishing or rereading model/cache bytes',async t=>{
  const f=await fixture(t),fake=fakeRuntime(f);await executeAPICase(f.p,f.c,f.root,f.native,f.dir,fake.deps);
  await assert.rejects(access(path.join(f.dir,'service')));const bundle=JSON.parse(await readFile(path.join(f.dir,'source-job-output.json'),'utf8'));
  const copied=path.join(f.root,'downloaded-case');await cp(f.dir,copied,{recursive:true});await rm(path.join(f.root,'services'),{recursive:true});await rm(bundle.build.binary);await rm(f.input);await rm(f.component);
  const r=await verifyAPIBindings(f.p,f.c,copied,fake.deps.decode,fake.deps.delta,{allowUnitFake:true,contentReadback:true});assert.equal(r.comparisons,1);
  await assert.rejects(verifyAPIBindings(f.p,f.c,copied,fake.deps.decode,fake.deps.delta,{contentReadback:true}),/unit_fake/);
});
test('backend readbacks bind both source identities, canonical cases and input hashes without a fixed population number',async t=>{
  const f=await fixture(t),p={...f.p,version:1,proof_kind:'unit_fake',nextAuthorizedAction:'Unit fake only; never dispatch models',cases:requiredCases().map(c=>({...c,command:['node','unit_fake'],timeoutSeconds:10,evidence:'evidence.json',inputs:[{path:c.backend==='cuda'?'D:\\unit_fake\\source':f.input,sha256:f.digest,role:'checkpoint',artifact:c.artifact}],loadHostBudgetBytes:1000,loadDeviceBudgetBytes:2000,minimumControlDelta:0.01,controlStream:'video',referenceRevision:inf,referenceCommand:['node','unit_fake'],referenceOutput:'reference.json',tolerance:{atol:0.01,rtol:0}}))};
  const r={kind:'backend-content-readback',proof_kind:'unit_fake',backend:'mlx',plan:p,planSha256:executionPlanDigest(p),verificationRunId:'unit_fake',verificationRunAttempt:'1',cases:p.cases.filter(c=>c.backend==='mlx').map(c=>({id:c.id,runId:'unit_fake',runAttempt:'1',planSha256:executionPlanDigest(p),executionSha256:f.digest,sourceInputs:c.inputs,verified:{files:[{path:'unit_fake.json',sha256:f.digest}]}}))};
  assert.throws(()=>validateBackendReadback(p,'mlx',r),/unit_fake/);assert.equal(validateBackendReadback(p,'mlx',r,{allowUnitFake:true}),r);
  for(const mutate of [v=>v.cases.pop(),v=>v.plan.inference=sw,v=>v.cases[0].sourceInputs[0].sha256=semantic]){const changed=structuredClone(r);mutate(changed);assert.throws(()=>validateBackendReadback(p,'mlx',changed,{allowUnitFake:true}));}
});
test('source-bound fixture adapter records actual assertions without fictional load/peak receipts',async t=>{
  const f=await fixture(t,'lifecycle','source-drift');f.c.command=['cargo','test','--manifest-path',path.join(f.root,'Cargo.toml'),'-p','sceneworks-core','--test','minimax_h3_import','--','unit_fake_assertion'];f.c.fixtureProof={kind:'source-bound-assertions'};
  const r=await executeFixture(f.p,f.c,f.root,f.native,f.dir,{proof_kind:'unit_fake',runProcess:async(argv,options)=>{await writeFile(options.logPath,'test unit_fake_assertion ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored;\n');},hardware:async()=>({device:'unit_fake',driver:'unit_fake'})});
  assert.equal(r.proof.passed,1);assert.equal(r.evidence.proof_kind,'unit_fake');assert.ok(!r.evidence.metrics&&!r.evidence.runtimeIdentity);
});
test('self-contained restart campaign scenario creates baseline, restarts owned state and replays captured job',async t=>{
  const f=await fixture(t,'lifecycle','restart-replay');const s=buildScenario(f.c,f.parameters,[f.execution]);assert.equal(s.lifecycleActions.length,1);
  await writeFile(f.c.runtime.scenario,JSON.stringify(s));f.c.inputs.find(i=>i.role==='scenario').sha256=await hashFile(f.c.runtime.scenario);f.c.declaredSourceBindings=[f.execution];
  const fake=fakeRuntime(f),r=await executeAPICase(f.p,f.c,f.root,f.native,f.dir,fake.deps),bundle=JSON.parse(await readFile(path.join(f.dir,'source-job-output.json'),'utf8'));
  assert.equal(bundle.epochs.length,2);assert.equal(bundle.bindings.length,2);assert.equal(bundle.actions[0].kind,'restart');assert.ok(fake.calls.some(v=>v[0]==='request'&&v[1].endsWith('/duplicate')));assert.ok(fake.children.every(v=>v.exitCode()===0));assert.equal(r.evidence.metrics,undefined,'multi-process restart is not a fictional single-PID peak');
});
test('foreign listener is never contacted or signalled',async t=>{
  const f=await fixture(t),fake=fakeRuntime(f,{wrongListener:true});
  await assert.rejects(new OwnedService(f.p,f.c,f.root,f.native,f.dir,fake.deps).initialize(),/not owned/);
  assert.equal(fake.calls.filter(c=>c[0]==='request').length,0);assert.equal(fake.calls.filter(c=>c[0]==='signal').length,0);
  assert.ok(fake.children.every(c=>c.exitCode()===0));
});
test('exact cached HF imports preserve real API lineage and refuse absent private snapshot bytes before transport',async t=>{
  const f=await fixture(t),repo='unit_fake/model',revision=inf,file='model.safetensors';f.c.runtime.seedFiles=[{input:f.input,destination:`hf/hub/models--unit_fake--model/snapshots/${revision}/${file}`,mode:'hardlink'}];
  const fake=fakeRuntime(f),service=await new OwnedService(f.p,f.c,f.root,f.native,f.dir,fake.deps).initialize(),step={method:'POST',path:'/api/v1/models/import',body:{source:{kind:'huggingFace',repo,revision,files:[file]},expectedSha256:f.digest}};
  await service.checkStep(step,{});assert.equal(fake.calls.filter(v=>v[0]==='request'&&v[1]==='/api/v1/models/import').length,0);
  await assert.rejects(service.checkStep({...step,body:{...step.body,source:{...step.body.source,files:['absent.safetensors']}}},{}));assert.equal(await hashFile(f.input),f.digest);await service.stop();
});
test('owner tampering and binary replacement refuse further transport',async t=>{
  const f=await fixture(t),fake=fakeRuntime(f);const service=await new OwnedService(f.p,f.c,f.root,f.native,f.dir,fake.deps).initialize();
  const original=await readFile(service.ownerPath,'utf8');await writeFile(service.ownerPath,JSON.stringify({token:'foreign',pid:process.pid}));
  await assert.rejects(service.request('GET','/api/v1/health'));await writeFile(service.ownerPath,original);
  await writeFile(service.binary,'replaced');await assert.rejects(service.request('GET','/api/v1/health'));await writeFile(service.binary,'unit_fake binary, never executable');await service.stop();
});
test('restart only closes owned processes after actual cooperative job cancellation',async t=>{
  const f=await fixture(t),fake=fakeRuntime(f);const service=await new OwnedService(f.p,f.c,f.root,f.native,f.dir,fake.deps).initialize();
  fake.jobs.set('owned-active',{id:'owned-active',status:'running',payload:{modelId:'imported_unit_fake'}});service.jobs.add('owned-active');
  await service.action({kind:'restart'},{});assert.equal(service.epochs.length,2);assert.equal(fake.jobs.get('owned-active').status,'canceled');
  const cancel=fake.calls.findIndex(v=>v[0]==='request'&&v[1].endsWith('/cancel')),signal=fake.calls.findIndex(v=>v[0]==='signal');assert.ok(cancel>=0&&signal>cancel);await service.stop();
  assert.ok(fake.children.every(c=>c.exitCode()===0));
});
test('filesystem interruption only replaces a hashed seeded private copy, preserving original and backup',async t=>{
  const f=await fixture(t),replacement=path.join(f.root,'replacement');await writeFile(replacement,'unit_fake replacement');const hash=await hashFile(replacement);f.c.inputs.push({path:replacement,sha256:hash,role:'mutation-input'});f.c.runtime.seedFiles=[{input:f.input,destination:'data/private/model.safetensors'}];
  const fake=fakeRuntime(f),service=await new OwnedService(f.p,f.c,f.root,f.native,f.dir,fake.deps).initialize();
  const r=await service.action({kind:'replace-owned-source',path:'data/private/model.safetensors',expectedSha256:f.digest,replacement,replacementSha256:hash},{});
  assert.equal(await hashFile(f.input),f.digest);assert.equal(await hashFile(r.backup),f.digest);assert.equal(await hashFile(r.target),hash);
  await assert.rejects(service.action({kind:'replace-owned-source',path:'../outside',expectedSha256:f.digest,replacement,replacementSha256:hash},{}));await service.stop();
});
test('case-specific preload import proof has no invented successful native load or zero peak',async t=>{
  const f=await fixture(t,'lifecycle','cancel'),fake=fakeRuntime(f);
  const r=await executeAPICase(f.p,f.c,f.root,f.native,f.dir,{...fake.deps,sample:async()=>{throw new Error('no native device initialized');}});
  assert.equal(r.evidence.proof_kind,'unit_fake');assert.ok(!r.evidence.runtimeIdentity);assert.equal(r.evidence.metrics,undefined);assert.equal(r.evidence.av,undefined);
});
test('media binding refuses path/symlink escape outside the owned project',async t=>{
  const f=await fixture(t),owned=path.join(f.root,'owned'),project=path.join(owned,'data','project');await mkdir(project,{recursive:true});const outside=path.join(f.root,'outside.mp4');await writeFile(outside,'unit_fake outside');await symlink(outside,path.join(project,'escape.mp4'));
  const service={runDir:owned,request:async(method,route)=>route.includes('/assets/')?{id:'asset',projectId:'project',type:'video',file:{path:'escape.mp4'}}:{id:'project',path:project}};
  await assert.rejects(bindCompletedMedia(service,{id:'job',status:'completed',projectId:'project',payload:{},result:{assetIds:['asset']}},f.dir,'escape',async()=>({proof_kind:'unit_fake'})),/owned/);
});
test('native load normalizer rejects stock checkpoint, wrong job and unverified sources',async t=>{
  const f=await fixture(t),fake=fakeRuntime(f),r=await executeAPICase(f.p,f.c,f.root,f.native,f.dir,fake.deps);
  const event=r.evidence.loadEvent,job={id:r.evidence.completedJob.id,status:'completed'};assert.equal(loadIdentity(event,job,f.c.inputs).checkpointSha256,f.digest);
  const wrong=structuredClone(event);wrong.identity.sourceFiles[0].sha256=semantic;assert.throws(()=>loadIdentity(wrong,job,f.c.inputs));assert.throws(()=>loadIdentity(event,{...job,id:'wrong'},f.c.inputs));
});
test('CUDA observed bound and MLX fresh post-case cumulative counter retain honest attribution',async t=>{
  const f=await fixture(t),service={worker:{pid:12345},c:{backend:'cuda'}};let n=0;
  const observations=new ProcessObservations(service,{intervalMs:1,sleepImpl:pause,sample:async()=>({at:new Date().toISOString(),pid:12345,host:10,device:20+ ++n,method:'sampled-device-used',deviceId:'unit_fake_device'})}).start();await pause();
  const metrics=await observations.finish();validatePeakMetrics(metrics,{backend:'cuda',loadHostBudgetBytes:1000,loadDeviceBudgetBytes:1000});assert.equal(metrics.observations.device.attribution.includesOtherProcesses,true);assert.ok(metrics.observations.loadDevice.bounds.includes('observed-device-global-bound'));
  assert.equal(metrics.loadDevicePeakBytes,metrics.devicePeakBytes);
});
test('missing device initialization preserves real-shaped host loading samples, never substitutes zero',async()=>{
  const service={worker:{pid:12345},jobs:new Set(),c:{backend:'mlx',loadHostBudgetBytes:10000,loadDeviceBudgetBytes:10000}};let n=0;
  const observations=new ProcessObservations(service,{intervalMs:1,sleepImpl:pause,sample:async()=>({at:new Date().toISOString(),pid:12345,host:++n===1?999:10,device:n===1?null:20,deviceUnavailable:n===1?'not initialized':undefined,deviceCounterAt:new Date().toISOString(),method:'mlx-allocator-cumulative-high-water',deviceId:'unit_fake_device'})}).start();await pause();const metrics=await observations.finish();
  assert.equal(metrics.hostPeakBytes,999);assert.equal(metrics.devicePeakBytes,20);assert.ok(metrics.observations.device.samples.every(s=>s.bytes===20));assert.ok(observations.gaps.length>0);
});
test('explicit campaign RSS bound requests cooperative cancellation, never a native process kill',async()=>{
  const calls=[];let status='running';const service={worker:{pid:12345},jobs:new Set(['owned']),c:{backend:'cuda',loadHostBudgetBytes:5,loadDeviceBudgetBytes:1000},request:async(method,route)=>{calls.push([method,route]);if(method==='POST')status='canceled';return {id:'owned',status};}};
  const observations=new ProcessObservations(service,{intervalMs:1,sleepImpl:pause,sample:async()=>({at:new Date().toISOString(),pid:12345,host:10,device:20,method:'sampled-device-used',deviceId:'unit_fake_device'})}).start();await pause();await assert.rejects(observations.finish(),/bound exceeded/);
  assert.equal(status,'canceled');assert.deepEqual(calls,[['GET','/api/v1/jobs/owned'],['POST','/api/v1/jobs/owned/cancel']]);assert.ok(observations.budgetExceeded);
});
test('contrasts hold complete creative configuration and bind only the adapters actually requested',()=>{
  const base={model:'imported_unit_fake',seed:42,width:576,height:320,advanced:{steps:4,schedulerShift:12},loras:[{id:'turbo'},{id:'lokr'}]};
  const changed={...structuredClone(base),loras:[{id:'turbo'}]};assertControlledConfiguration(base,changed,['loras']);
  changed.advanced.steps=1;assert.throws(()=>assertControlledConfiguration(base,changed,['loras']));assert.throws(()=>assertControlledConfiguration(base,changed,['advanced']));
  assert.deepEqual(requestedAdapterBindings([{id:'turbo',sha256:'t'},{id:'lokr',sha256:'l'}],[{id:'turbo'}]),[{id:'turbo',sha256:'t'}]);
});
test('private environment isolates cache/config/credentials while preserving device restrictions',()=>{
  const env=privateEnvironment('/unit_fake_owned',{port:43210,gpuId:'unit_fake',backend:'mlx'},'task-generated-token');
  assert.equal(env.HF_HOME,path.join('/unit_fake_owned','hf'));assert.equal(env.SCENEWORKS_RUN_UTILITY_INPROCESS,'0');assert.ok(!Object.hasOwn(env,'HF_TOKEN'));assert.equal(env.HF_HUB_OFFLINE,'1');assert.equal(env.SCENEWORKS_ACCESS_TOKEN,'task-generated-token');
});
test('single author-declared hybrid does not require a fictional companion checkpoint',()=>{
  const c=requiredCases().find(v=>v.id==='mlx.capability.hybrid');
  const model={modelId:'imported_single_hybrid',sourcePath:'/unit_fake/model',sha256:semantic,baseModel:'minimax_h3',partition:'hybrid',topology:'pruned',hybridBinding:{kind:'single-checkpoint',declarationSource:'unit_fake author declaration'},componentTier:'q4',components:[{sha256:semantic,source:'unit_fake',revision:inf,licenseAcknowledged:true}],supportedModes:['text_to_video','image_to_video','reference_to_video']};
  const executions=model.supportedModes.map((mode,i)=>({label:`hybrid${i}`,model,mode,sampling:{contractSource:'unit_fake',steps:50,width:576,height:320,duration:5.1667,fps:24}}));
  const scenario=buildScenario(c,{baseUrl:'http://127.0.0.1:43210',projectId:'unit_fake',seed:1,requestedGpu:'unit_fake',timeoutSeconds:30,assets:{imageA:'red',imageB:'blue',voiceA:'speech',voiceB:'reversed'},sensitivity:{minimumControlDelta:0.01,maximumBaselineDelta:0.001}},executions);
  assert.ok(scenario.steps.length);assert.ok(!model.companion);
});
