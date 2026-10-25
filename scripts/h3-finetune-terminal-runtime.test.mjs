// Pure CPU/in-process fakes. Never creates a listener, child process, GPU context or model run.
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import os from 'node:os';
import { mkdtemp, mkdir, readFile, writeFile, rm, symlink, cp, access, stat } from 'node:fs/promises';
import { executeNativeComparison,executeFixture,validateComparisonProof,OwnedService,ProcessObservations,loadIdentity,bindCompletedMedia,executeAPICase,verifyAPIBindings,privateEnvironment,authorized,hardware,verifyGenerationSourcePrepCancel,SOURCE_PREP_CHUNK_BYTES,verifyOwnershipEquivalence,verifyCacheCoverage,observedSourceBytes,verifyDenseSourceDtypes,assertControlledConfiguration,requestedAdapterBindings } from './h3-finetune-terminal-runtime.mjs';
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
  const p={proof_kind:'unit_fake',sceneworks:sw,inference:inf,source:SOURCE,output:root,responsibleTask:'sc-26691'};
  const execution={licenseAuthority:{kind:'unit_fake',acknowledged:true,approvedBy:'unit_fake only',reference:'synthetic consent, never real approval',approvedAt:'2026-10-25T00:00:00Z',subjects:[{role:'source',sha256:digest},{role:'component',sha256:componentHash}]},label:'test',mode:'image_to_video',model:{modelId:'imported_unit_fake',sourcePath:input,sha256:digest,baseModel:'minimax_h3',partition:'fl2va',topology:'full',componentTier:'q4',supportedModes:['image_to_video'],components:[{sha256:componentHash,source:'unit_fake',revision:inf,licenseAcknowledged:true}]},sampling:{contractSource:'unit_fake recipe',steps:50,width:576,height:320,fps:24,duration:5.1667}};
  const parameters={proof_kind:'unit_fake',baseUrl:'http://127.0.0.1:43210',projectId:'unit_fake_project',requestedGpu:'unit_fake_gpu',timeoutSeconds:30,seed:42,assets:{imageA:'unit_fake_red',imageB:'unit_fake_blue'},sensitivity:{minimumControlDelta:0.01,maximumBaselineDelta:0.005}};
  const scenario=(group==='capability'||(group==='lifecycle'&&check==='cancel'))?buildScenario(c,parameters,[execution]):{baseUrl:parameters.baseUrl,caseId:c.id,timeoutSeconds:30,steps:[{id:'import',method:'POST',path:'/api/v1/models/import',body:{modelId:'imported_unit_fake',sourcePath:input,licenseAcknowledged:true},expectedStatus:201,assert:[{pointer:'/payload/modelId',equals:'imported_unit_fake'}]},{id:'installed',waitJob:{$capture:'import/id'},assert:[{pointer:'/result/modelId',equals:'imported_unit_fake'}]}]};
  await writeFile(scenarioPath,JSON.stringify(scenario));c.inputs.push({path:scenarioPath,sha256:await hashFile(scenarioPath),role:'scenario'});
  c.declaredSourceBindings=[execution];
  return {root,dir,native,input,digest,component,componentHash,p,c,scenario,execution,parameters};
}
async function replaceScenario(f,c,executions,parameters=f.parameters){
  Object.assign(f.c,c);f.c.declaredSourceBindings=executions;f.scenario=buildScenario(f.c,parameters,executions);await writeFile(f.c.runtime.scenario,JSON.stringify(f.scenario));f.c.inputs.find(v=>v.role==='scenario').sha256=await hashFile(f.c.runtime.scenario);return f.scenario;
}
function fakeRuntime(f,{wrongListener=false,backend='mlx'}={}) {
  let pid=7000,jobNo=0,nativeLoadCount=0,serviceDir,activeWorker,heartbeat=Date.now();const children=[],calls=[],jobs=new Map(),assets=new Map();
  const preparationCancel=f.c.id.endsWith('.generation-source-prep-cancel');
  const prepSpan=jobId=>({jobId,checkpointId:'unit_fake_source',phase:'source-and-normalization'});
  const preparationSource=()=>path.join(serviceDir,'data','source-prep','model.safetensors');
  const spawnOwned=(argv,env,cwd,logPath)=>{
    const kind=argv[0]===process.execPath?'guardian':env.SCENEWORKS_WORKER_ONLY==='1'?'worker':'api';
    serviceDir=path.dirname(env.SCENEWORKS_CONFIG_DIR);let code=null,logs='';let resolveExit;const done=new Promise(r=>resolveExit=r);
    const child={pid:++pid,kind,env,argv,logPath,exitCode:()=>code,log:()=>logs,append:v=>{logs+=JSON.stringify(v)+'\n';},close:()=>{if(code!==null)return;code=0;resolveExit({code:0,signal:null});},stdin:{end:()=>{calls.push(['end',kind,child.pid]);child.close();if(kind==='guardian')for(const c of children.filter(c=>c.kind==='api'&&c.env.SCENEWORKS_PARENT_PID===String(child.pid)))c.close();}},signal:s=>{assert.equal(s,'SIGTERM');assert.ok([...jobs.values()].every(j=>['completed','canceled','failed'].includes(j.status)),'no signal before cooperative completion');calls.push(['signal',kind,child.pid,s]);child.close();},exit:async()=>{const result=await done;await writeFile(logPath,logs,{flag:'wx'});return result;}};
    children.push(child);calls.push(['spawn',kind,child.pid]);if(kind==='worker'){activeWorker=child;nativeLoadCount=0;}return child;
  };
  const deps={proof_kind:'unit_fake',platform:'darwin',sleep:pause,spawnOwned,requireFreePort:async port=>{calls.push(['free',port]);},listenerPid:async()=>wrongListener?99999:children.filter(c=>c.kind==='api'&&c.exitCode()===null).at(-1)?.pid,
    runProcess:async(argv,options)=>{calls.push(['build',argv]);assert.ok(argv.includes('sceneworks-rust-api'));const binary=path.join(options.env.CARGO_TARGET_DIR,'debug','sceneworks-rust-api');await mkdir(path.dirname(binary),{recursive:true});await writeFile(binary,'unit_fake binary, never executable');await writeFile(options.logPath,'unit_fake build log',{flag:'wx'});},
    hardware:async()=>({device:'unit_fake CPU fake, not native hardware',driver:'unit_fake'}),
    decode:async file=>({sha256:await hashFile(file),proof_kind:'unit_fake',streams:[{codec_type:'video',codec_name:'h264',width:576,height:320,avg_frame_rate:'24/1',r_frame_rate:'24/1',nb_read_frames:'124',nb_frames:'124',duration:String(124/24),start_time:'0'},{codec_type:'audio',codec_name:'aac',sample_rate:'32000',channels:2,nb_read_frames:'162',duration:String(124/24),start_time:'0',proof_kind:'unit_fake'}]}),
    delta:async(a,b)=> (await readFile(a,'utf8'))===(await readFile(b,'utf8'))?0:0.1,
    sample:async service=>({at:new Date().toISOString(),pid:service.worker.pid,host:100,device:200,deviceCounterAt:new Date().toISOString(),method:backend==='mlx'?'mlx-allocator-cumulative-high-water':'sampled-device-used',deviceId:'unit_fake_device'}),
    fetch:async(url,options)=>{
      calls.push(['request',url.pathname,options.method]);assert.ok(options.headers['X-SceneWorks-Token']);let value,status=200;
      if(url.pathname==='/api/v1/health')value={status:'ok',service:'sceneworks-api'};
      else if(url.pathname==='/api/v1/workers'){const active=[...jobs.values()].find(j=>!['completed','canceled','failed','interrupted'].includes(j.status));value=[{id:activeWorker.env.SCENEWORKS_WORKER_ID,status:active?'busy':'idle',currentJobId:active?.id??null,lastSeenAt:new Date(++heartbeat).toISOString()}];}
      else if(url.pathname==='/api/v1/models/library-roots')value={roots:[{rootId:'unit_fake_root',path:f.root}]};
      else if(url.pathname==='/api/v1/models/import'){const body=JSON.parse(options.body);const id=`import-${++jobNo}`;value={id,status:'queued',payload:body};const actual=body.expectedSha256&&body.sourcePath?await hashFile(body.sourcePath):null,mismatch=actual&&actual!==body.expectedSha256;jobs.set(id,{id,status:mismatch?'failed':'queued',payload:body,result:mismatch?{}:{modelId:body.modelId},...(mismatch?{error:`Imported model failed its integrity check (sha256 ${actual}, source declares ${body.expectedSha256})`}:{})});status=201;}
      else if(url.pathname==='/api/v1/video/jobs'&&preparationCancel){
        // Cold preparation: the generation stays in preparation and emits an actual partial
        // ColdContentRead checkpoint. It is never completed before the cancellation action.
        const body=JSON.parse(options.body),id=`video-${++jobNo}`,source=preparationSource();
        await mkdir(path.dirname(source),{recursive:true});await writeFile(source,await readFile(f.input));
        const sourceBytes=(await stat(source)).size;
        // Hash cancellation is BEFORE codec copying: no normalization copy event may be emitted.
        // The core checksum can complete before the native seal starts: these are different observers.
        activeWorker.append({jobId:id,span:prepSpan(id),event:'source_file_hash_progress',path:source,acquisition:'ColdContentRead',state:'Acquired',hashed_bytes:sourceBytes,source_bytes:sourceBytes});
        activeWorker.append({jobId:id,span:prepSpan(id),event:'artifact_seal_hash_progress',path:source,acquisition:'ColdContentRead',state:'Reading',hashed_bytes:1,source_bytes:null});
        activeWorker.append({jobId:id,span:prepSpan(id),event:'artifact_seal_hash_progress',path:source,acquisition:'ColdContentRead',state:'Reading',hashed_bytes:SOURCE_PREP_CHUNK_BYTES,source_bytes:sourceBytes});
        const modelProvenance={plan:{checkpointId:'unit_fake_source',planId:'unit_fake_plan',semanticDigest:semantic}},payload={...body,expectedImportedModelRef:modelProvenance.plan,generationSetId:id};jobs.set(id,{id,status:'preparing',payload,modelProvenance,result:{modelId:body.model}});value={id,status:'queued',payload,modelProvenance};status=201;
      }
      else if(url.pathname==='/api/v1/video/jobs'){
        const body=JSON.parse(options.body),id=`video-${++jobNo}`,assetId=`asset-${jobNo}`,project=path.join(serviceDir,'data','projects','unit_fake.sceneworks');
        await mkdir(project,{recursive:true});const file=`${assetId}.mp4`;await writeFile(path.join(project,file),body.sourceAssetId??body.prompt??'unit_fake_video');
        const asset={id:assetId,projectId:body.projectId,type:'video',file:{path:file},recipe:{rawSettings:{proof_kind:'unit_fake'}}};assets.set(assetId,asset);
        const job={id,status:'completed',projectId:body.projectId,payload:{...body,generationSetId:id},result:{assetIds:[assetId]}};jobs.set(id,job);value={...job,status:'queued'};status=201;
        activeWorker.append({event:'h3_import_prepared_load',jobId:id,provider:'unit_fake_native',backend:'unit_fake_gpu',native_backend:f.c.backend,seal_validated:true,cache_layer:'generator_metadata',loaded_spec_offload_policy:'Sequential',loaded_spec_load_shape:'unit_fake_observed_shape',retains_dit_weights:false,cache_state:nativeLoadCount++===0?'Cold':'Warm',identity:{checkpointId:'unit_fake_source',planId:'unit_fake_plan',semanticDigest:semantic,assembly:{family:'minimax-h3',partition:'fl2va',sourceEncoding:'unit_fake-convrot',components:{},componentClosure:[{revision:inf,files:[{relativePath:'component',sha256:'sha256:'+f.componentHash,sizeBytes:19}]}]},sourceFiles:[{path:f.input,sha256:f.digest}]},prepared_sources:[{path:f.input,sha256:f.digest},{path:f.component,sha256:f.componentHash}],operation:body.mode,seed:body.seed,steps:body.advanced.steps,scheduler_shift:null});
        const constrained=Boolean(activeWorker.env.SCENEWORKS_MLX_MEMORY_CAP_GB||activeWorker.env.SCENEWORKS_CUDA_VRAM_CAP_GB);activeWorker.append({event:'generation_memory_policy_applied',jobId:id,scope_configured:true,stage_residency:constrained,stream_transformer_blocks:constrained,window_size:constrained?2:null,window_component:constrained?'Dit':'none',strategy:constrained?'unit_fake_btr':'unit_fake_default'});
      } else if(url.pathname.includes('/assets/'))value=assets.get(url.pathname.split('/').at(-1));
      else if(url.pathname.startsWith('/api/v1/projects/'))value={id:'unit_fake_project',path:path.join(serviceDir,'data','projects','unit_fake.sceneworks')};
      else if(url.pathname.endsWith('/duplicate')){const original=jobs.get(url.pathname.split('/').at(-2));assert.ok(original);return deps.fetch(new URL('/api/v1/video/jobs',url),{...options,body:JSON.stringify(original.payload)});}
      else if(options.method==='DELETE'&&url.pathname.startsWith('/api/v1/models/')){
        const id=decodeURIComponent(url.pathname.split('/').at(-1));value={id,removedManifestEntry:true,permanent:new URL(url).searchParams.get('permanent')==='true'};
      }
      else if(url.pathname.endsWith('/cancel')){
        const id=url.pathname.split('/').at(-2),job=jobs.get(id);assert.ok(job,'cancel must target a submitted job');job.cancelRequested=true;
        if(preparationCancel){
          const source=preparationSource(),sourceBytes=(await stat(source)).size;
          activeWorker.append({jobId:id,span:prepSpan(id),event:'artifact_seal_hash_progress',path:source,acquisition:'ColdContentRead',state:'Canceled',hashed_bytes:SOURCE_PREP_CHUNK_BYTES,source_bytes:sourceBytes});
          job.status='canceled';job.message='h3-preparation-canceled after joined teardown';value=job;
        } else { job.status='canceled'; value=job; }
      } else if(url.pathname.startsWith('/api/v1/jobs/')){value=jobs.get(url.pathname.split('/').at(-1));if(value?.cancelRequested&&!['completed','failed','canceled'].includes(value.status))value.status='canceled';else if(value?.status==='queued')value.status='completed';}
      assert.ok(value,`no invented fake API route: ${url.pathname}`);return {status,text:async()=>JSON.stringify(value)};
    }};
  return {deps,children,calls,jobs,assets,spawnOwned,get serviceDir(){return serviceDir;}};
}
test('real service/native entry is refused locally before spawn or transport',async t=>{
  const f=await fixture(t);assert.throws(()=>authorized(f.p,f.c));
  await assert.rejects(new OwnedService(f.p,f.c,f.root,f.native,f.dir).initialize(),/unit_fake|Actions-only|equal/);
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
  const fake=fakeRuntime(f),service=await new OwnedService(f.p,f.c,f.root,f.native,f.dir,fake.deps).initialize(),step={method:'POST',path:'/api/v1/models/import',body:{modelId:f.execution.model.modelId,licenseAcknowledged:true,source:{kind:'huggingFace',repo,revision,files:[file]},expectedSha256:f.digest}};
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
  const f=await fixture(t),service={epochs:[{workerPid:12345}],worker:{pid:12345},c:{backend:'cuda'}};let n=0;
  const observations=new ProcessObservations(service,{intervalMs:1,sleepImpl:pause,sample:async()=>({at:new Date().toISOString(),pid:12345,host:10,device:20+ ++n,method:'sampled-device-used',deviceId:'unit_fake_device'})}).start();await pause();
  const metrics=await observations.finish();validatePeakMetrics(metrics,{backend:'cuda',loadHostBudgetBytes:1000,loadDeviceBudgetBytes:1000});assert.equal(metrics.observations.device.attribution.includesOtherProcesses,true);assert.ok(metrics.observations.loadDevice.bounds.includes('observed-device-global-bound'));
  assert.equal(metrics.loadDevicePeakBytes,metrics.devicePeakBytes);
});
test('missing device initialization preserves real-shaped host loading samples, never substitutes zero',async()=>{
  const service={epochs:[{workerPid:12345}],worker:{pid:12345},jobs:new Set(),c:{backend:'mlx',loadHostBudgetBytes:10000,loadDeviceBudgetBytes:10000}};let n=0;
  const observations=new ProcessObservations(service,{intervalMs:1,sleepImpl:pause,sample:async()=>({at:new Date().toISOString(),pid:12345,host:++n===1?999:10,device:n===1?null:20,deviceUnavailable:n===1?'not initialized':undefined,deviceCounterAt:new Date().toISOString(),method:'mlx-allocator-cumulative-high-water',deviceId:'unit_fake_device'})}).start();await pause();const metrics=await observations.finish();
  assert.equal(metrics.hostPeakBytes,999);assert.equal(metrics.devicePeakBytes,20);assert.ok(metrics.observations.device.samples.every(s=>s.bytes===20));assert.ok(observations.gaps.length>0);
});
test('explicit campaign RSS bound requests cooperative cancellation, never a native process kill',async()=>{
  const calls=[];let status='running';const service={epochs:[{workerPid:12345}],worker:{pid:12345},jobs:new Set(['owned']),c:{backend:'cuda',loadHostBudgetBytes:5,loadDeviceBudgetBytes:1000},request:async(method,route)=>{calls.push([method,route]);if(method==='POST')status='canceled';return {id:'owned',status};}};
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
test('root-selected compiler reaches private child environments without inheriting credentials',()=>{
  const previous=process.env.RUSTUP_TOOLCHAIN;process.env.RUSTUP_TOOLCHAIN='9.8.7';
  try{const env=privateEnvironment('/unit_fake_owned',{port:43210,gpuId:'0',backend:'cuda'},'unit_fake_token');assert.equal(env.RUSTUP_TOOLCHAIN,'9.8.7');assert.ok(!Object.hasOwn(env,'HF_TOKEN'));}
  finally{if(previous===undefined)delete process.env.RUSTUP_TOOLCHAIN;else process.env.RUSTUP_TOOLCHAIN=previous;}
});
test('CUDA receipt binds PCI-ordered device zero rather than NVML index zero using only fake inventory',async()=>{
  const csv='0, GPU-unit_fake-high, 00000000:41:00.0, unit_fake_high, unit_fake_driver\n1, GPU-unit_fake-low, 00000000:01:00.0, unit_fake_low, unit_fake_driver\n';
  const env={CUDA_DEVICE_ORDER:'PCI_BUS_ID',CUDA_VISIBLE_DEVICES:'0'},calls=[];
  const exec=async(command,args)=>{calls.push([command,args]);return{stdout:csv};};
  const value=await hardware('cuda','GPU-unit_fake-low',{env,exec});assert.equal(value.cuda.selectedPhysicalDevice.nvmlIndex,1);assert.equal(value.cuda.selectedPhysicalDevice.uuid,'GPU-unit_fake-low');assert.equal(calls[0][0],'nvidia-smi');assert.ok(calls[0][1].includes('--query-gpu=index,uuid,pci.bus_id,name,driver_version'));
  await assert.rejects(hardware('cuda','GPU-unit_fake-high',{env,exec}),/manual device zero/);
  for(const bad of [{...env,CUDA_VISIBLE_DEVICES:'1'},{...env,CUDA_DEVICE_ORDER:'FASTEST_FIRST'}])await assert.rejects(hardware('cuda','GPU-unit_fake-low',{env:bad,exec}),/manual CUDA proof/);
});
test('native unknown source size stays unknown while a real zero-byte size stays zero',()=>{assert.equal(observedSourceBytes(undefined),null);assert.equal(observedSourceBytes(null),null);assert.equal(observedSourceBytes('None'),null);assert.equal(observedSourceBytes(0),0);assert.equal(observedSourceBytes('0'),0);assert.equal(observedSourceBytes(1024),1024);});
async function coldCancelFake(t){
  const f=await fixture(t,'lifecycle','generation-source-prep-cancel');
  // An actual source larger than one cold checkpoint: the partial reading must be meaningful.
  const bytes=Buffer.alloc(3*SOURCE_PREP_CHUNK_BYTES+4093);for(let i=0;i<bytes.length;i++)bytes[i]=(i*31+7)&0xff;
  await writeFile(f.input,bytes);
  const digest=await hashFile(f.input);f.digest=digest;f.execution.model.sha256=digest;
  for(const subject of f.execution.licenseAuthority.subjects)if(subject.role==='source')subject.sha256=digest;
  f.c.inputs.find(input=>input.role==='checkpoint').sha256=digest;
  await replaceScenario(f,f.c,[f.execution]);
  const fake=fakeRuntime(f),result=await executeAPICase(f.p,f.c,f.root,f.native,f.dir,fake.deps);
  return {f,fake,result,receiptName:result.evidence.sourceCancelReceipt,dir:f.dir};
}
async function copyCase(dir,name){
  const target=`${dir}-${name}`;await cp(dir,target,{recursive:true});return target;
}
async function editReceipt(dir,mutate){
  const name=(await readdirNames(dir)).find(v=>v.startsWith('source-cancel-')&&v.endsWith('.json')&&v!=='source-cancel-actions.json');
  const file=path.join(dir,name),receipt=JSON.parse(await readFile(file,'utf8'));mutate(receipt);await writeFile(file,JSON.stringify(receipt,null,2)+'\n');return {receipt,name};
}
async function readdirNames(dir){const {readdir}=await import('node:fs/promises');return readdir(dir);}
async function editRetained(dir,predicate,mutate){
  const lifecycle=JSON.parse(await readFile(path.join(dir,'service-lifecycle.json'),'utf8'));
  for(const epoch of lifecycle.epochs){const file=path.join(dir,`worker-${epoch.epoch}.log`);const lines=(await readFile(file,'utf8')).split('\n').map(line=>{try{const value=JSON.parse(line);if(predicate(value)){mutate(value);return JSON.stringify(value);}return line;}catch{return line;}});await writeFile(file,lines.join('\n'));}
}
async function appendRetained(dir,event){
  const lifecycle=JSON.parse(await readFile(path.join(dir,'service-lifecycle.json'),'utf8'));
  const file=path.join(dir,`worker-${lifecycle.epochs.at(-1).epoch}.log`);await writeFile(file,(await readFile(file,'utf8'))+JSON.stringify(event)+'\n');
}
test('cold generation source-preparation cancellation binds retained partial bytes, joined teardown and lease release',async t=>{
  const {f,fake,result,receiptName,dir}=await coldCancelFake(t);
  assert.equal(result.evidence.proof_kind,'unit_fake');
  assert.ok(receiptName&&receiptName.startsWith('source-cancel-'));
  assert.equal(result.evidence.av,undefined);assert.equal(result.evidence.runtimeIdentity,undefined);assert.equal(result.evidence.metrics,undefined);
  const verified=await verifyGenerationSourcePrepCancel(f.p,f.c,dir,{allowUnitFake:true,receiptName});
  assert.equal(verified.checkpointSha256,f.digest);assert.equal(verified.sourceBytes,3*SOURCE_PREP_CHUNK_BYTES+4093);
  assert.equal(verified.partialBytes,SOURCE_PREP_CHUNK_BYTES);assert.ok(verified.elapsedMs<30000);
  assert.equal(result.evidence.runtimeBindings,'source-job-output.json');const sourceVerification=await verifyAPIBindings(f.p,f.c,dir,fake.deps.decode,fake.deps.delta,{allowUnitFake:true});assert.equal(sourceVerification.nativeClosures,0);assert.deepEqual(sourceVerification.jobs,[]);
  const actualReceipt=JSON.parse(await readFile(path.join(dir,receiptName),'utf8'));assert.ok(Date.parse(actualReceipt.sourceAudit.auditedAt)>=Date.parse(actualReceipt.terminal.acknowledgedAt));assert.ok(actualReceipt.generationJob.payload.generationSetId,'actual queued canonical enrichment must not be mistaken for raw request bytes');
  const wrongRole=structuredClone(f.c);for(const input of wrongRole.inputs)if(input.role==='checkpoint')input.role='component';await assert.rejects(verifyGenerationSourcePrepCancel(f.p,wrongRole,dir,{allowUnitFake:true,receiptName}),/checkpoint-role input/);
  await assert.rejects(verifyGenerationSourcePrepCancel(f.p,f.c,dir,{receiptName}),/unit_fake/);
  await assert.rejects(verifyGenerationSourcePrepCancel(f.p,f.c,dir,{allowUnitFake:true}),/receipt is required/);
  // Free label/receipt dictionaries never certify the case.
  const wrong=structuredClone(await JSON.parse(await readFile(path.join(dir,receiptName),'utf8')));wrong.observation={state:'passed'};
  await writeFile(path.join(f.root,'free-receipt.json'),JSON.stringify(wrong));
  await assert.rejects(verifyGenerationSourcePrepCancel(f.p,f.c,f.root,{allowUnitFake:true,receiptName:'free-receipt.json'}),/actual native artifact observer/);
});
test('cold generation source-preparation cancellation fails loudly on every substituted or missed field',async t=>{
  const {f,receiptName,dir}=await coldCancelFake(t);
  const reading=value=>value.event==='artifact_seal_hash_progress'&&value.state==='Reading';
  const cases=[
    ['missing observation',async d=>{await editReceipt(d,r=>delete r.observation);},/actual partial ColdContentRead observation required/],
    ['zero bytes',async d=>{await editReceipt(d,r=>{r.observation.hashed_bytes=0;});await editRetained(d,reading,v=>{v.hashed_bytes=0;});},/zero|partial/],
    ['memo acquisition',async d=>{await editReceipt(d,r=>{r.observation.acquisition='MemoryMemo';});await editRetained(d,reading,v=>{v.acquisition='MemoryMemo';});},/only a cold content read/],
    ['full source bytes',async d=>{await editReceipt(d,r=>{r.observation.hashed_bytes=4*SOURCE_PREP_CHUNK_BYTES;r.observation.source_bytes=4*SOURCE_PREP_CHUNK_BYTES;});await editRetained(d,reading,v=>{v.hashed_bytes=4*SOURCE_PREP_CHUNK_BYTES;v.source_bytes=4*SOURCE_PREP_CHUNK_BYTES;});},/partial, not the full source/],
    ['wrong job',async d=>{await editReceipt(d,r=>{r.jobId='other-job';});},/actual .*generation job/],
    ['wrong path',async d=>{await editReceipt(d,r=>{r.observation.path=path.join(f.root,'not-owned');});},/actual observed source path|same observed owned source path/],
    ['wrong source size',async d=>{await editReceipt(d,r=>{r.observation.source_bytes+=1;});},/source audit size|stat size/],
    ['wrong planned source role',async d=>{await editReceipt(d,r=>{r.sourceAudit.sha256=semantic;});},/declared checkpoint source|verified checkpoint input/],
    ['acquired source',async d=>{await appendRetained(d,{jobId:(await JSON.parse(await readFile(path.join(d,receiptName),'utf8'))).jobId,span:{jobId:'x',checkpointId:'c',phase:'source-and-normalization'},event:'artifact_seal_hash_progress',path:(await JSON.parse(await readFile(path.join(d,receiptName),'utf8'))).observation.path,acquisition:'ColdContentRead',state:'Acquired',hashed_bytes:SOURCE_PREP_CHUNK_BYTES,source_bytes:3*SOURCE_PREP_CHUNK_BYTES+4093});},/fully Acquired/],
    ['native load present',async d=>{const r=JSON.parse(await readFile(path.join(d,receiptName),'utf8'));await appendRetained(d,{event:'h3_import_prepared_load',jobId:r.jobId,seal_validated:true,cache_state:'Cold'});},/native load/],
    ['copy started before cancel',async d=>{const r=JSON.parse(await readFile(path.join(d,receiptName),'utf8'));await appendRetained(d,{jobId:r.jobId,span:{jobId:r.jobId,checkpointId:'unit_fake_source',phase:'source-and-normalization'},event:'h3_normalization_copy_progress',state:'Writing',copied_payload_bytes:SOURCE_PREP_CHUNK_BYTES,total_payload_bytes:3*SOURCE_PREP_CHUNK_BYTES+4093});},/hashing window was missed|normalization copy began/],
    ['missing joined marker',async d=>{await editReceipt(d,r=>{r.terminal.job.message='canceled';});},/joined-teardown/],
    ['hard abandon',async d=>{await editReceipt(d,r=>{r.terminal.hardAbandoned=true;});},/hard-abandoned/],
    ['over grace',async d=>{await editReceipt(d,r=>{r.terminal.elapsedMs=30000;});},/under the 30000ms/],
    ['busy teardown',async d=>{await editReceipt(d,r=>{r.teardown.idle.worker.status='busy';});},/fresh owned worker Idle/],
    ['refused removal',async d=>{await editReceipt(d,r=>{r.removal.response.removedManifestEntry=false;});},/refused model removal/],
    ['removal before idle',async d=>{await editReceipt(d,r=>{r.removal.at='2000-01-01T00:00:00.000Z';});},/removal must follow/],
  ];
  for(const [name,mutate,pattern]of cases){
    const copy=await copyCase(dir,name.replace(/[^a-z0-9]+/gi,'-'));
    try{
      await mutate(copy);
      await assert.rejects(verifyGenerationSourcePrepCancel(f.p,f.c,copy,{allowUnitFake:true,contentReadback:true,receiptName}),pattern,name);
      // A failed/missed cancellation still retains its actual receipt for forensic readback.
      assert.ok((await readdirNames(copy)).some(v=>v.startsWith('source-cancel-')&&v!=='source-cancel-actions.json'),name);
    }finally{await rm(copy,{recursive:true,force:true});}
  }
});
test('cold generation-preparation cancellation cannot be classified passed while blocked or unexecuted',async()=>{
  const pending=requiredCases().filter(c=>c.id.endsWith('.generation-source-prep-cancel'));
  assert.deepEqual(pending.map(c=>c.backend),['mlx','cuda']);
  for(const c of pending){
    await assert.rejects(validateEvidence({},c,'/unit_fake_unread','test result: ok. 1 passed; 0 failed; 0 ignored;'));
    await assert.rejects(validateEvidence({output:'/unit_fake_unread'},c,'/unit_fake_unread','test result: ok. 1 passed; 0 failed; 0 ignored;'));
  }
});
test('review repair linked runtime accepts omitted ownership and refuses fictional enum before transport',async t=>{
  const f=await fixture(t),fake=fakeRuntime(f),service=await new OwnedService(f.p,f.c,f.root,f.native,f.dir,fake.deps).initialize();
  const body={modelId:f.execution.model.modelId,linkedRootId:'unit_fake_root',linkedRelativePath:path.basename(f.input),licenseAcknowledged:true};
  try{await service.checkStep({method:'POST',path:'/api/v1/models/import',body},{});await assert.rejects(service.checkStep({method:'POST',path:'/api/v1/models/import',body:{...body,ownershipMode:'linked'}},{}),/omitted/);}finally{await service.stop();}
});
test('review repair all successful generalization and replay executions require actual load events',async t=>{
  for(const check of ['generalization.native-single','lifecycle.restart-replay']){
    const f=await fixture(t);f.execution.model.sourceFormat='native-single';await replaceScenario(f,requiredCases().find(v=>v.id===`mlx.${check}`),[f.execution]);
    const fake=fakeRuntime(f),launch=fake.deps.spawnOwned;fake.deps.spawnOwned=(...args)=>{const child=launch(...args),append=child.append;child.append=value=>{if(value.event!=='h3_import_prepared_load')append(value);};return child;};
    await assert.rejects(executeAPICase(f.p,f.c,f.root,f.native,f.dir,fake.deps),/EVERY successful imported generation/);
  }
});
test('review repair readback rejects empty or fabricated native history even with completed AV',async t=>{
  const f=await fixture(t);f.execution.model.sourceFormat='native-single';await replaceScenario(f,requiredCases().find(v=>v.id==='mlx.generalization.native-single'),[f.execution]);const fake=fakeRuntime(f);await executeAPICase(f.p,f.c,f.root,f.native,f.dir,fake.deps);
  const file=path.join(f.dir,'source-job-output.json'),bundle=JSON.parse(await readFile(file,'utf8'));bundle.bindings=[];await writeFile(file,JSON.stringify(bundle));
  await assert.rejects(verifyAPIBindings(f.p,f.c,f.dir,fake.deps.decode,fake.deps.delta,{allowUnitFake:true}),/missing an actual native closure/);
});
test('review repair copied closure metadata cannot replace authentic retained worker trace',async t=>{
  const f=await fixture(t),fake=fakeRuntime(f);await executeAPICase(f.p,f.c,f.root,f.native,f.dir,fake.deps);await writeFile(path.join(f.dir,'worker-1.log'),'');
  await assert.rejects(verifyAPIBindings(f.p,f.c,f.dir,fake.deps.decode,fake.deps.delta,{allowUnitFake:true}),/authentic worker/);
});
test('dense precision witnesses inspect actual body tensor headers rather than source-format labels',async t=>{
  const f=await fixture(t);Object.assign(f.c,requiredCases().find(c=>c.id==='mlx.generalization.dense-f32'));f.execution.model.sourceFormat='f32';
  const header={__metadata__:{format:'pt'},'transformer_blocks.0.attn.to_q.weight':{dtype:'BF16',shape:[1,1],data_offsets:[0,2]},'transformer_blocks.0.adaln_proj.weight':{dtype:'F32',shape:[1,1],data_offsets:[2,6]}};const bytes=Buffer.from(JSON.stringify(header)),prefix=Buffer.alloc(8);prefix.writeBigUInt64LE(BigInt(bytes.length));await writeFile(f.input,Buffer.concat([prefix,bytes,Buffer.alloc(6)]));const digest=await hashFile(f.input);f.execution.model.sha256=digest;f.c.inputs.find(input=>input.role==='checkpoint').sha256=digest;
  await assert.rejects(verifyDenseSourceDtypes(f.c,f.root),/actual source body precision differs/);
  f.c.id='mlx.generalization.dense-bf16';f.execution.model.sourceFormat='bf16';const facts=await verifyDenseSourceDtypes(f.c,f.root);assert.equal(facts.expectedBodyDtype,'BF16');assert.equal(facts.bodyMatrices,1);assert.equal(facts.files[0].histogram.F32,1,'source-required non-body F32 AdaLN buffers must remain honestly recorded');
});
test('review repair request-bound AV validator runs on the real output-binding path',async t=>{
  const f=await fixture(t),fake=fakeRuntime(f),decode=fake.deps.decode;fake.deps.decode=async file=>{const decoded=await decode(file);decoded.streams[0].nb_read_frames='1';return decoded;};
  await assert.rejects(executeAPICase(f.p,f.c,f.root,f.native,f.dir,fake.deps),/frames but the request resolved/);
});
test('review repair managed linked equivalence compares actual semantic assembly and both output streams',async t=>{
  const f=await fixture(t),linked=structuredClone(f.execution);linked.label='linked';linked.model.modelId='imported_linked';Object.assign(linked,{ownershipMode:'linked',linkedRootId:'unit_fake_root',linkedRelativePath:path.basename(f.input)});
  const scenario=await replaceScenario(f,requiredCases().find(v=>v.id==='mlx.lifecycle.managed-linked-equivalence'),[f.execution,linked],{...f.parameters,equivalence:{maximumBaselineDelta:0.001,maximumVideoDelta:0.001,maximumAudioDelta:0.001}});
  const fake=fakeRuntime(f);await executeAPICase(f.p,f.c,f.root,f.native,f.dir,fake.deps);const bundle=JSON.parse(await readFile(path.join(f.dir,'source-job-output.json'),'utf8')),byJob=new Map(bundle.bindings.map(v=>[v.job.id,v]));
  assert.equal((await verifyAPIBindings(f.p,f.c,f.dir,fake.deps.decode,fake.deps.delta,{allowUnitFake:true})).ownershipEquivalence.length,2);
  const changed=byJob.get(bundle.captures.linked_av0_done.id),original=structuredClone(changed.runtimeIdentity.quantization.observedAssembly);changed.runtimeIdentity.quantization.observedAssembly.components={different:'bytes'};
  await assert.rejects(verifyOwnershipEquivalence(bundle,scenario,byJob,f.dir,fake.deps.decode,fake.deps.delta),/semantic component/);changed.runtimeIdentity.quantization.observedAssembly=original;
  await writeFile(path.join(f.dir,changed.media.av),'different linked output');await assert.rejects(verifyOwnershipEquivalence(bundle,scenario,byJob,f.dir,fake.deps.decode,async(a,b)=>a.includes('output-0')&&b.includes('output-2')?0.1:0),/not equivalent/);
  await assert.rejects(verifyOwnershipEquivalence(bundle,scenario,byJob,f.dir,fake.deps.decode,async(a,b)=>a.includes('output-0')&&b.includes('output-1')||a.includes('output-2')&&b.includes('output-3')?0.1:0),/baseline noise/);
});
test('content identity uses real copied bytes under a different layout and actual API checksum refusal for different bytes',async t=>{
  const f=await fixture(t),layout=path.join(f.root,'unrelated-layout');await mkdir(layout);const renamed=path.join(layout,'opaque.safetensors'),different=path.join(layout,'different.safetensors');await cp(f.input,renamed);const bytes=await readFile(f.input);bytes[bytes.length-1]^=1;await writeFile(different,bytes);const differentSHA=await hashFile(different);
  const positive=structuredClone(f.execution);positive.model.sourcePath=renamed;positive.model.originalPath=f.input;positive.model.originalSha256=f.digest;
  const negative=structuredClone(positive);negative.label='different_bytes';negative.model.modelId='imported_different_bytes';negative.model.sourcePath=different;negative.model.sha256=differentSHA;negative.identityControl={kind:'different-bytes-negative',expectedSourceSha256:f.digest};for(const subject of negative.licenseAuthority.subjects)if(subject.role==='source')subject.sha256=differentSHA;
  const receipt=path.join(f.root,'identity-controls.json');await writeFile(receipt,JSON.stringify({kind:'h3-finetune-stage-identity-controls',status:'prepared-not-modelproof',proof_kind:'unit_fake',sourcePath:f.input,sourceExpectedSHA:f.digest,sourceActualSHA:f.digest,renamed:{path:renamed,size:bytes.length,actualSHA:f.digest},differentBytes:{path:different,size:bytes.length,actualSHA:differentSHA,expectedOriginalSHA:f.digest}}));
  f.c.inputs.push({path:renamed,sha256:f.digest,role:'checkpoint'},{path:different,sha256:differentSHA,role:'checkpoint'},{path:receipt,sha256:await hashFile(receipt),role:'identity-control-fixture'});
  await replaceScenario(f,requiredCases().find(c=>c.id==='mlx.generalization.renamed'),[positive,negative],{...f.parameters,identificationControlReceipt:receipt});const fake=fakeRuntime(f);await executeAPICase(f.p,f.c,f.root,f.native,f.dir,fake.deps);const verified=await verifyAPIBindings(f.p,f.c,f.dir,fake.deps.decode,fake.deps.delta,{allowUnitFake:true});
  assert.equal(verified.identification.sameContentRecognized.sha256,f.digest);assert.equal(verified.identification.differentContentRefused.observedSha256,differentSHA);assert.equal(await hashFile(f.input),f.digest);
  const source=JSON.parse(await readFile(path.join(f.dir,'source-job-output.json'),'utf8'));source.captures.different_bytes_refused.status='completed';await writeFile(path.join(f.dir,'source-job-output.json'),JSON.stringify(source));await assert.rejects(verifyAPIBindings(f.p,f.c,f.dir,fake.deps.decode,fake.deps.delta,{allowUnitFake:true}),/different bytes must not be accepted/);
});
test('review repair cache coverage rejects three cold runs and unchanged staged policy',async t=>{
  const f=await fixture(t),executions=['cold','warm','staged'].map(label=>({...structuredClone(f.execution),label,reuseInstalled:label!=='cold',...(label==='staged'?{memoryCap:{kind:'mlx-memory-gb',value:1}}:{})}));
  const scenario=await replaceScenario(f,requiredCases().find(v=>v.id==='mlx.lifecycle.cold-warm-staged'),executions),fake=fakeRuntime(f);const result=await executeAPICase(f.p,f.c,f.root,f.native,f.dir,fake.deps);validatePeakMetrics(result.evidence.metrics,f.c);assert.equal(result.evidence.metrics.kind,'owned-worker-epochs');const wrongEpoch=structuredClone(result.evidence.metrics);wrongEpoch.epochs[1].pid=wrongEpoch.epochs[0].pid;assert.throws(()=>validatePeakMetrics(wrongEpoch,f.c));
  const bundle=JSON.parse(await readFile(path.join(f.dir,'source-job-output.json'),'utf8')),byJob=new Map(bundle.bindings.map(v=>[v.job.id,v])),events=[];for(const epoch of bundle.epochs)for(const line of(await readFile(path.join(f.dir,`worker-${epoch.epoch}.log`),'utf8')).split('\n')){try{events.push(JSON.parse(line));}catch{}}
  const verified=await verifyAPIBindings(f.p,f.c,f.dir,fake.deps.decode,fake.deps.delta,{allowUnitFake:true});assert.equal(verified.cacheCoverage.cacheLayer,'generator-metadata-not-DiT-residency');
  const warm=byJob.get(bundle.captures.warm_av0_done.id);warm.event.cache_state='Cold';assert.throws(()=>verifyCacheCoverage(bundle,scenario,byJob,events),/actually hit/);warm.event.cache_state='Warm';
  const applied=events.find(e=>e.event==='generation_memory_policy_applied'&&e.jobId===bundle.captures.staged_av0_done.id);applied.stream_transformer_blocks=false;assert.throws(()=>verifyCacheCoverage(bundle,scenario,byJob,events),/bounded transformer/);applied.stream_transformer_blocks=true;
  const before=events.find(e=>e.event==='generation_memory_policy_applied'&&e.jobId===warm.job.id);Object.assign(before,{stream_transformer_blocks:true,window_size:2});assert.throws(()=>verifyCacheCoverage(bundle,scenario,byJob,events),/not a staged transition/);
});
test('locator normalization compares retained bytes not fresh UUIDs and preserves reference order',async()=>{
  const {normalizedControlRequest:normalize}=await import('./h3-finetune-terminal-runtime.mjs');
  const request={projectId:'project-A',model:'model-A',prompt:'literal prompt',seed:42,referenceAssetIds:['image-A','image-B'],referenceAudioAssetIds:[]};
  const identity={checkpointSha256:semantic,topology:'full',quantization:'int8',components:[]},metadata={type:'image',mimeType:'image/png',width:32,height:32};
  const context={runtimeIdentity:identity,declaredSourceBindings:[{model:{modelId:'model-A'},adapters:[]}],conditioningBindings:[{projectId:'project-A',assetId:'image-A',sha256:'a'.repeat(64),metadata},{projectId:'project-A',assetId:'image-B',sha256:'b'.repeat(64),metadata}]};
  const original=normalize(request,context),otherRequest={...request,projectId:'fresh-project',model:'fresh-model',referenceAssetIds:['fresh-image-1','fresh-image-2']},other={...context,declaredSourceBindings:[{model:{modelId:'fresh-model'},adapters:[]}],conditioningBindings:context.conditioningBindings.map((v,i)=>({...v,projectId:'fresh-project',assetId:otherRequest.referenceAssetIds[i]}))};
  assert.deepEqual(normalize(otherRequest,other),original,'same bytes with different UUIDs must compare equal');
  const changed=structuredClone(context);changed.conditioningBindings[0].sha256='c'.repeat(64);assert.notDeepEqual(normalize(request,changed),original,'different bytes with the same UUID must not compare equal');
  assert.notDeepEqual(normalize({...request,referenceAssetIds:[...request.referenceAssetIds].reverse()},context),original,'reference order is creative control, not a locator');
  assert.notDeepEqual(normalize({...request,prompt:'different literal prompt'},context),original);assert.equal(original.prompt,'literal prompt');
});
test('reduced cross-case controls consume three distinct actual jobs and reject source or seed drift',async t=>{
  const f=await fixture(t),{verifyReducedControls}=await import('./h3-finetune-terminal-runtime.mjs');
  const cases=[1,5,8].map(index=>({id:`mlx.reduced.unit-${index}`,backend:'mlx',index,kind:'full-av',sensitivity:{minimumControlDelta:0.01,maximumBaselineDelta:0.001}}));
  const base={projectId:'unit_fake',model:'imported_unit_fake',seed:1,advanced:{steps:4},width:576,height:320,referenceAudioAssetIds:['unit_a','unit_b']};
  const bundles=new Map();for(const c of cases){const dir=path.join(f.root,c.id);await mkdir(dir);const request={...base,...(c.index===8?{referenceAudioAssetIds:[]}:{}),advanced:{...base.advanced}},job={id:`unit-job-${c.index}`,status:'completed',payload:request},bundle={bindings:[{job,runtimeIdentity:{checkpointSha256:f.digest,samplingPolicy:{steps:4}},media:{av:'unit_fake.mp4'}}],requests:[{route:'/api/v1/video/jobs',request,result:{id:job.id}}]};bundles.set(c.index,bundle);await writeFile(path.join(dir,'source-job-output.json'),JSON.stringify(bundle));}
  const plan={...f.p,cases},decode=async()=>({unit_fake:true}),delta=async(_a,b)=>b.includes('unit-8')?0.1:0;
  const result=await verifyReducedControls(plan,'mlx',{decode,delta});assert.equal(result.length,1);assert.equal(result[0].baselineJobId,'unit-job-1');assert.equal(result[0].repeatedJobId,'unit-job-5');assert.equal(result[0].interventionJobId,'unit-job-8');
  const changed=bundles.get(8);changed.requests[0].request.seed=2;await writeFile(path.join(f.root,cases[2].id,'source-job-output.json'),JSON.stringify(changed));await assert.rejects(verifyReducedControls(plan,'mlx',{decode,delta}),/non-intervened creative control/);
  changed.requests[0].request.seed=1;changed.bindings[0].job.id='unit-job-1';await writeFile(path.join(f.root,cases[2].id,'source-job-output.json'),JSON.stringify(changed));await assert.rejects(verifyReducedControls(plan,'mlx',{decode,delta}),/reuse the baseline job ID/);
});
test('single author-declared hybrid does not require a fictional companion checkpoint',()=>{
  const c=requiredCases().find(v=>v.id==='mlx.capability.hybrid');
  const model={modelId:'imported_single_hybrid',sourcePath:'/unit_fake/model',sha256:semantic,baseModel:'minimax_h3',partition:'hybrid',topology:'pruned',hybridBinding:{kind:'single-checkpoint',declarationSource:'unit_fake author declaration'},componentTier:'q4',components:[{sha256:semantic,source:'unit_fake',revision:inf,licenseAcknowledged:true}],supportedModes:['text_to_video','image_to_video','reference_to_video']};
  const executions=model.supportedModes.map((mode,i)=>({licenseAuthority:{kind:'unit_fake',acknowledged:true,approvedBy:'unit fake',reference:'synthetic only',approvedAt:'2026-10-25T00:00:00Z',subjects:['source','component'].map(role=>({role,sha256:semantic}))},label:`hybrid${i}`,model,mode,sampling:{contractSource:'unit_fake',steps:50,width:576,height:320,duration:5.1667,fps:24}}));
  const scenario=buildScenario(c,{proof_kind:'unit_fake',baseUrl:'http://127.0.0.1:43210',projectId:'unit_fake',seed:1,requestedGpu:'unit_fake',timeoutSeconds:30,assets:{imageA:'red',imageB:'blue',voiceA:'speech',voiceB:'reversed'},sensitivity:{minimumControlDelta:0.01,maximumBaselineDelta:0.001}},executions);
  assert.ok(scenario.steps.length);assert.ok(!model.companion);
});
