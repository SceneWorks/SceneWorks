// Tiny structured files + injected asset transports only; never model, browser or listener work.
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import os from 'node:os';
import {createHash} from 'node:crypto';
import {mkdtemp,mkdir,readFile,writeFile,rm,stat,access,link,copyFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {bootstrapTerminal,validateBootstrapRequest,MINIMAX_H3_MAX_PIXELS} from './h3-finetune-terminal-bootstrap.mjs';
import {stripJsoncComments} from './lib/jsonc.mjs';
import {caseBook} from './h3-finetune-terminal-cases.mjs';
import {runStage,prepareReferenceFixtures,selectStageFiles,bindScopedAuthority,assertScopedBinding} from './h3-finetune-terminal-stage.mjs';
import {validatePlan,verifyInputs,hashFile,runCase} from './h3-finetune-terminal.mjs';
import {runScenario} from './h3-finetune-terminal-api.mjs';
const root=fileURLToPath(new URL('..',import.meta.url));
// Tests must never write under a real HF cache: neither the Windows campaign root nor the HF_HOME
// / ~/.cache/huggingface this process inherited. Every path a test materializes stays in its temp dir.
const REAL_HF_ROOTS=[process.env.HF_HOME,process.env.HF_HUB_CACHE,'E:\\huggingface',path.join(os.homedir(),'.cache','huggingface')].filter(Boolean).map(p=>path.resolve(p).toLowerCase());
function assertTestOwned(file,temporary){
  const resolved=path.resolve(file),lower=resolved.toLowerCase();
  assert.ok(lower.startsWith(path.resolve(temporary).toLowerCase()+path.sep),`${resolved} escapes the test temp dir ${temporary}`);
  for(const real of REAL_HF_ROOTS)assert.ok(lower!==real&&!lower.startsWith(real+path.sep),`${resolved} is under the real HF cache ${real}`);
}
const digest=value=>createHash('sha256').update(value).digest('hex');
const json=(file,value)=>writeFile(file,JSON.stringify(value,null,2)+'\n');
function sft(tensors){const width={U8:1,I8:1,F32:4,F8_E4M3:1},header={},data=[];let offset=0;for(const [name,dtype,shape,provided]of tensors){const bytes=provided??Buffer.alloc(shape.reduce((a,b)=>a*b,1)*width[dtype],1);header[name]={dtype,shape,data_offsets:[offset,offset+bytes.length]};offset+=bytes.length;data.push(bytes);}const text=Buffer.from(JSON.stringify(header)),size=Buffer.alloc(8);size.writeBigUInt64LE(BigInt(text.length));return Buffer.concat([size,text,...data]);}
function packed(){const w='blocks.0.attn.qkv_proj',c='blocks.0.mlp.fc2',wd=Buffer.from(JSON.stringify({format:'asym_w4a8_int8',group_size:16,convrot_groupsize:256})),cd=Buffer.from(JSON.stringify({format:'int8_tensorwise',convrot:true,convrot_groupsize:256}));return sft([[w+'.weight','I8',[16,128]],[w+'.weight_codebook','F32',[16]],[w+'.weight_s_rel','F8_E4M3',[16,16]],[w+'.weight_s_channel','F32',[16]],[w+'.comfy_quant','U8',[wd.length],wd],[c+'.weight','I8',[16,256]],[c+'.weight_scale','F32',[16,1]],[c+'.comfy_quant','U8',[cd.length],cd]]);}
function pruned(){return sft([['adaln_t_table','F32',[4,2]],['blocks.0.adaln_proj.linear.weight','F32',[36,2]],['blocks.0.adaln_proj.linear.bias','F32',[36]],['final_layer.adaln_proj.linear.weight','F32',[4,2]],['final_layer.adaln_proj.linear.bias','F32',[4]],['blocks.0.attn.qkv_proj.weight','I8',[6,2]],['padding','U8',[32768]]]);}
function request(){return {stationary_pair_authorized:true,bootstrap:{casebook:'sc-26691-h3-cases-v1',cohort:'reduced-v1',operator:{compatibleBase:'builtin:minimax_h3',projectName:'unit_fake proof only',seed:314159,requestedGpu:'mlx',gpuId:'mlx',componentTier:'q4',hostBudgetBytes:1024**3,deviceBudgetBytes:1024**3,minimumControlDelta:0.01,maximumBaselineDelta:0.001,width:1088,height:544,wideWidth:1344,wideHeight:768,timeoutSeconds:30,lokrAmplitude:0.05,pulpcutSampling:{steps:8,schedulerShift:12,audioSchedulerShift:3,contractSource:'unit_fake operator-selected sampling, NOT an authoritative trained-shift claim'}},componentSelections:[{modelId:'minimax_h3',tier:'q4'},{modelId:'minimax_h3_ref',tier:'q4'}],legalAuthority:{proof_kind:'unit_fake',acknowledged:true,assetSets:['singularity','hybrid','pulpcut','turbo','speech'],licenseIds:['MiniMax-H3-Community-License-Agreement','Apache-2.0','CC-BY-4.0'],inheritedLicenseIds:['MiniMax-H3-Community-License-Agreement'],includeSelectedComponents:true}}};}
async function fixture(t){
  const temporary=await mkdtemp(path.join(os.tmpdir(),'h3-bootstrap-unit-fake-'));t.after(()=>rm(temporary,{recursive:true,force:true}));
  const product=path.join(temporary,'inference'),evidenceNative=path.join(temporary,'inference-evidence'),evidenceRoot=path.join(temporary,'logs');for(const d of [product,evidenceNative,evidenceRoot])await mkdir(d);
  const env={GITHUB_ACTIONS:'true',GITHUB_EVENT_NAME:'workflow_dispatch',GITHUB_SHA:'a'.repeat(40),H3_SCENEWORKS_SHA:'a'.repeat(40),H3_INFERENCE_SHA:'b'.repeat(40),H3_NATIVE_EVIDENCE_SHA:'c'.repeat(40),H3_NATIVE_EVIDENCE_REF:'main',GITHUB_RUN_ID:'101',GITHUB_RUN_ATTEMPT:'1',RUNNER_NAME:'nax-macos',HF_HOME:path.join(temporary,'hf'),HOME:temporary,H3_TERMINAL_ROUTE:'h3-terminal-mlx'};
  const book=structuredClone(caseBook),data=new Map(),models=[sft([['padding','U8',[65536]]]),sft([['padding','U8',[60000]]]),pruned(),packed()];
  book.artifacts.forEach((v,i)=>{v.expectedBytes=models[i].length;v.expectedSha256=digest(models[i]);data.set(v.artifact,models[i]);});
  for(const [set,rows]of Object.entries(book.stagingAssets))for(const v of rows){const bytes=Buffer.from('unit_fake '+set+' '+v.file);v.expectedBytes=bytes.length;v.expectedSha256=digest(bytes);data.set(v.file,bytes);}
  book.turboArtifacts.files.forEach(v=>{v.sha256=book.stagingAssets.turbo.find(row=>row.file===v.file).expectedSha256;});
  const calls={stage:0,inputs:0};
  const stage=async options=>{calls.stage++;return runStage({...options,fetchImpl:async url=>{const filename=decodeURIComponent(String(url).split('/resolve/')[1].split('/').slice(1).join('/'));assert.ok(data.has(filename),'only injected pinned fixture bytes may be fetched');const bytes=data.get(filename);return new Response(bytes,{status:200,headers:{'content-length':String(bytes.length)}});},probe:async()=>({available:true,device:'unit_fake no GPU context'}),codeSha:env.GITHUB_SHA});};
  const referenceExecute=async(command,args,options)=>{const argv=[command,...args];if(argv[1]==='-version')return {code:0,stdout:'unit_fake tool 0'};if(argv[0].includes('ffprobe')){const image=String(argv.at(-1)).includes('image'),audio=String(argv.at(-1)).includes('tone');return {code:0,stdout:JSON.stringify({streams:[audio?{codec_type:'audio',sample_rate:'32000',channels:1,duration:'1'}:{codec_type:'video',width:32,height:32,nb_read_frames:image?'1':'24',duration:'1'}]})};}if(argv.at(-1)!=='-')await writeFile(path.join(options.cwd,argv.at(-1)),Buffer.from('unit_fake generated '+argv.at(-1)));return {code:0,stdout:''};};
  const resolveInputs=async options=>{calls.inputs++;const component=path.join(options.stageRoot,'component.bin'),bytes=Buffer.from('unit_fake actual component bytes');await writeFile(component,bytes);const legal={licenseId:'MiniMax-H3-Community-License-Agreement',inheritedLicenseId:'MiniMax-H3-Community-License-Agreement',attributionRequired:'MiniMax H3'},entry={repoType:'model',repo:'unit/component',revision:'d'.repeat(40),file:'component.bin',expectedSha256:digest(bytes),expectedBytes:bytes.length,legal};const ack=await options.acknowledgmentFor(entry,{component:true,modelId:'minimax_h3',componentId:'text_encoder',legal});const part={id:'text_encoder',source:entry.repo,revision:entry.revision,sha256:entry.expectedSha256,path:component,licenseAcknowledged:ack.acknowledged};
    const configPath=path.join(options.stageRoot,'source-config.json'),dit={hidden_size:2,num_attention_heads:1,attention_head_dim:2,num_layers:1,ffn_dim:4,time_embed_dim:4,text_dim:2,in_channels:1,audio_in_channels:1,patch_size:[1,1,1],norm_eps:1e-6,qk_norm_eps:1e-6,final_norm_eps:1e-6};await json(configPath,dit);
    const referenceDir=path.join(options.stageRoot,'reference');await mkdir(referenceDir);const reference=path.join(referenceDir,'reference.py');await writeFile(reference,'# unit_fake no code execution\n');
    return {componentsByModel:{minimax_h3:[part],minimax_h3_ref:[part]},seedFiles:[],componentRecords:[{...entry,status:'completed',actualBytes:bytes.length,observedSha256:entry.expectedSha256,finalPath:component,expectedLegal:legal,explicitAcknowledgment:ack}],inputs:[{path:component,sha256:entry.expectedSha256,role:'component'},{path:configPath,sha256:await hashFile(configPath),role:'source-config'},{path:reference,sha256:await hashFile(reference),role:'reference'}],config:{path:configPath,sha256:await hashFile(configPath),dit},referenceDir};};
  const prepareSpeech=async({stageRoot})=>{const directory=path.join(stageRoot,'conditioning','speech-1272-128104-0000');await mkdir(directory,{recursive:true});const outputs=[];for(const [name,role]of [['speech.wav','speech-source'],['reversed.wav','diagnostic-reversed-not-speaker']]){const file=path.join(directory,name);await writeFile(file,'unit_fake '+name);outputs.push({path:name,role,sha256:await hashFile(file)});}return {status:'completed',proof_kind:'unit_fake',directory,outputs};};
  return {temporary,env,book,calls,args:{request:request(),backend:'mlx',swRoot:root,inferenceRoot:product,evidenceRoot,env,outputRoot:path.join(temporary,'h3-terminal-bootstrap-101-1'),deps:{proof_kind:'unit_fake',book,stage,resolveInputs,prepareReferences:prepareReferenceFixtures,referenceExecute,tools:{ffmpeg:'/unit_fake/ffmpeg',ffprobe:'/unit_fake/ffprobe'},prepareSpeech}}};
}
function fakeAPI(){let index=0,library;const jobs=new Map(),calls=[];return {calls,fetchImpl:async(url,options)=>{const route=url.pathname;calls.push({route,options});let result,status=200;if(route==='/api/v1/projects'){status=201;result={id:'unit-project'};}else if(route.endsWith('/assets')&&options.method==='POST'){assert.ok(options.body instanceof FormData);assert.ok(options.body.get('file').size>0);status=201;result={id:'unit-asset-'+(++index),projectId:'unit-project'};}else if(route==='/api/v1/models/library-roots'){status=201;library=JSON.parse(options.body).path;result={rootId:'unit-root'};}else if(['/api/v1/models/import','/api/v1/loras/import','/api/v1/video/jobs'].includes(route)){status=201;const payload=JSON.parse(options.body),id='unit-job-'+(++index);result={id,status:'queued',payload};const source=payload.sourcePath??(payload.linkedRelativePath?path.join(library,payload.linkedRelativePath):undefined),observed=source?await hashFile(source):undefined,failed=observed&&payload.expectedSha256&&observed!==payload.expectedSha256;jobs.set(id,{id,status:failed?'failed':'completed',payload,result:{modelId:payload.modelId,loraId:payload.loraId},...(failed?{error:`integrity check expected ${payload.expectedSha256}, actual ${observed}`}:{})});}else if(route==='/api/v1/loras'){result=calls.filter(c=>c.route==='/api/v1/loras/import').map(c=>{const p=JSON.parse(c.options.body);return {id:p.loraId,baseModel:p.baseModel};});}else{result=jobs.get(route.split('/').at(-1));assert.ok(result,'mock only the existing owned API protocol');}return {status,text:async()=>JSON.stringify(result)};}};}
// Seed a case exactly as the owned runtime does before it submits the scenario: managed sources and
// adapters are imported from private copies inside the case's own data directory.
async function seedLikeRuntime(plan,c){
  const runDir=path.join(plan.output,'services',c.id);
  for(const seed of c.runtime.seedFiles){const dest=path.join(runDir,...seed.destination.split('/'));try{await access(dest);continue;}catch{}await mkdir(path.dirname(dest),{recursive:true});if(seed.mode==='hardlink')await link(seed.input,dest);else await copyFile(seed.input,dest);}
  return runDir;
}
test('fresh filesystem and one scoped input reach a real source-bound prepared cohort without prior plan/scenario files',async t=>{
  for(const backend of ['mlx','cuda']){
  const f=await fixture(t),original=structuredClone(f.args.request.bootstrap.legalAuthority);f.args.backend=backend;if(backend==='cuda')Object.assign(f.args.request.bootstrap.operator,{gpuId:'1',requestedGpu:'1'});f.env.H3_TERMINAL_ROUTE=`h3-terminal-${backend}`;await assert.rejects(access(f.args.outputRoot),/ENOENT/);
  const result=await bootstrapTerminal(f.args);assert.equal(result.prepared,true);assert.equal(result.proof,false);assert.equal(result.cases,25);assert.equal(f.calls.stage,1);assert.equal(f.calls.inputs,1);
  const plan=JSON.parse(await readFile(result.planPath,'utf8'));validatePlan(plan);assert.equal(plan.backend,backend);
  // sc-26691 user-approved scope decision (2026-10-30): the compatible-base cases (2, 4 with its
  // identity negative, 20, 25) are planned at MLX 8 / CUDA 50 with a recorded step policy; every
  // other case keeps its recipe's steps on both backends, and no shift is invented for them.
  const expectedSteps={2:{mlx:8,cuda:50},4:{mlx:8,cuda:50},20:{mlx:8,cuda:50},25:{mlx:8,cuda:50},17:8,18:8,21:8};
  for(const v of plan.cases.filter(v=>v.declaredSourceBindings))for(const e of v.declaredSourceBindings){
    const want=expectedSteps[v.index],steps=typeof want==='object'?want[backend]:want??4;assert.equal(e.sampling.steps,steps,`${v.id} ${e.label}`);
    if(typeof want==='object'){assert.deepEqual(v.backendSteps,{cuda:50,mlx:8});assert.equal(e.sampling.stepPolicy.backend,backend);assert.equal(e.sampling.stepPolicy.steps,steps);assert.equal(e.sampling.stepPolicy.decision.approvedAt,'2026-10-30');assert.equal(e.sampling.stepPolicy.decision.measuredRun,'41061562586');assert.equal(e.sampling.schedulerShift,undefined,'no shift is invented for the compatible-base recipe');assert.match(e.sampling.contractSource,new RegExp(`at ${steps} steps on ${backend}`));}
    else assert.equal(e.sampling.stepPolicy,undefined,v.id);
  }
  {const tampered=structuredClone(plan),c2=tampered.cases.find(v=>v.index===2);c2.declaredSourceBindings[0].sampling.steps=backend==='mlx'?50:8;assert.throws(()=>validatePlan(tampered),/its declared (mlx|cuda) recipe is/);}
  // sc-26691: no well-known port; every case service binds an OS-assigned port at start.
  for(const v of plan.cases.filter(v=>v.runtime?.scenario))assert.equal(v.runtime.port,0,v.id);
  // The staged private root (and everything bound from it) lives in this test's temp dir only.
  assertTestOwned(plan.assetStaging.receipt.path,f.temporary);for(const c of plan.cases)for(const input of c.inputs??[])if(!input.path.startsWith(root))assertTestOwned(input.path,f.temporary);assert.equal(plan.proof_kind,'unit_fake');assert.equal(plan.preparationState,'bound');assert.equal(plan.assetStaging.receipt.sha256,await hashFile(plan.assetStaging.receipt.path));
  for(const c of plan.cases)await verifyInputs(c);
  const c=plan.cases.find(v=>v.index===1),scenario=JSON.parse(await readFile(c.runtime.scenario,'utf8')),api=fakeAPI();assert.equal(scenario.baseUrl,'http://127.0.0.1:0','scenario names the OS-assigned owned origin, not a fixed port');
  // The API confines a local-copy import to <data>/models and a LoRA import to <data>/loras, so
  // every managed source and adapter is imported from a seeded private copy in the case's own data
  // directory — never straight from the staging root, which the API refuses. Seed exactly as the
  // owned runtime does before it submits the scenario.
  const runDir=await seedLikeRuntime(plan,c),dataDir=path.join(runDir,'data');
  const imports=scenario.steps.filter(v=>['/api/v1/models/import','/api/v1/loras/import'].includes(v.path)&&v.body?.sourcePath);
  assert.ok(imports.length>=2,'the case imports its managed source and its accelerator');
  for(const step of imports){const root=path.join(dataDir,step.path==='/api/v1/models/import'?'models':'loras');assert.ok(step.body.sourcePath.startsWith(root+path.sep),`${step.id} must import from the case's own ${path.basename(root)} directory, not the staging root`);}
  const executed=await runScenario(scenario,{baseUrl:'http://127.0.0.1:50001',fetchImpl:api.fetchImpl,sleep:async()=>{},uploadFile:async spec=>new Blob([await readFile(spec.path)],{type:spec.mimeType})});assert.ok(executed.transcript.length>0);assert.equal(api.calls.filter(v=>v.route==='/api/v1/video/jobs').length,1);assert.equal(api.calls.filter(v=>v.route==='/api/v1/loras/import').length,1,'published accelerator is actually imported, not only selected as builtin');
  assert.deepEqual(c.declaredSourceBindings[0].licenseAuthority.operatorInput,original);assertScopedBinding(c.declaredSourceBindings[0].licenseAuthority,{allowUnitFake:true});
  const identity=plan.cases.find(v=>v.index===4).declaredSourceBindings;assert.notEqual(identity[0].model.sha256,identity[1].model.sha256);assert.equal(identity[0].model.originalSha256,identity[0].model.sha256);const lokrCase=plan.cases.find(v=>v.index===3);await seedLikeRuntime(plan,lokrCase);const lokr=lokrCase.declaredSourceBindings[0].adapters.find(v=>v.kind==='lokr');assert.equal(lokr.sha256,await hashFile(lokr.importPath));
  const identityCase=plan.cases.find(v=>v.index===4);await seedLikeRuntime(plan,identityCase);const identityScenario=JSON.parse(await readFile(identityCase.runtime.scenario,'utf8')),identityAPI=fakeAPI();await runScenario(identityScenario,{baseUrl:'http://127.0.0.1:50002',fetchImpl:identityAPI.fetchImpl,sleep:async()=>{},uploadFile:async spec=>new Blob([await readFile(spec.path)],{type:spec.mimeType})});assert.equal(identityAPI.calls.filter(v=>v.route==='/api/v1/video/jobs').length,1,'negative integrity control never generates');
  await assert.rejects(runCase(plan,c.id,root,f.args.inferenceRoot),/unit_fake/);await assert.rejects(access(path.join(result.output,c.id,'execution.json')),/ENOENT/);
  }
});
test('missing or out-of-scope permission refuses before input materialization and never manufactures approval',async t=>{
  for(const mutation of [v=>delete v.legalAuthority,v=>{v.legalAuthority.acknowledged=false;},v=>{v.legalAuthority.licenseIds=['Apache-2.0'];},v=>{v.legalAuthority.includeSelectedComponents=false;}]){const f=await fixture(t);mutation(f.args.request.bootstrap);await assert.rejects(bootstrapTerminal(f.args));assert.equal(f.calls.stage,0);assert.equal(f.calls.inputs,0);await assert.rejects(access(f.args.outputRoot),/ENOENT/);}
});
test('bootstrap rejects arbitrary source/code/template fields and synthetic consent in real mode',()=>{
  for(const [key,value]of [['commands',['sh','bad']],['source','https://untrusted.invalid'],['files',{}],['template','eval()']]){const r=request();r.bootstrap[key]=value;assert.throws(()=>validateBootstrapRequest(r,{backend:'mlx',unitFake:true}),/unsupported bootstrap/);}
  assert.throws(()=>validateBootstrapRequest(request(),{backend:'mlx'}),/synthetic acknowledgment/);
  // A fixed operator port (MLX run 41044021373 collided on 43210) is refused; omitted or 0 is OS-assigned.
  for(const port of [43210,8000,-1,'0']){const r=request();r.bootstrap.operator.port=port;assert.throws(()=>validateBootstrapRequest(r,{backend:'mlx',unitFake:true}),/OS-assigned loopback port/);}
  {const r=request();r.bootstrap.operator.port=0;validateBootstrapRequest(r,{backend:'mlx',unitFake:true});}
  // The macOS lane's jobs must be routable to the product MLX worker: a request naming GPU 0 is
  // refused before anything is prepared, instead of queuing jobs no worker will ever claim.
  for(const field of ['gpuId','requestedGpu']){const r=request();r.bootstrap.operator[field]='0';assert.throws(()=>validateBootstrapRequest(r,{backend:'mlx'}),/MLX worker/);}
  // CUDA runs only on physical GPU 1: GPU 0 is the owner's display/work GPU.
  for(const [gpuId,requestedGpu] of [['0','0'],['0','1'],['1','0']]){const r=request();Object.assign(r.bootstrap.operator,{gpuId,requestedGpu});assert.throws(()=>validateBootstrapRequest(r,{backend:'cuda',unitFake:true}),/physical GPU 1/);}
  const input=request().bootstrap.legalAuthority,binding=bindScopedAuthority(input,[{role:'source',sha256:'a'.repeat(64),legal:{licenseId:'MiniMax-H3-Community-License-Agreement'},origin:{kind:'casebook-asset',assetSet:'singularity'}}],{allowUnitFake:true});binding.operatorInput.acknowledged=false;assert.throws(()=>assertScopedBinding(binding,{allowUnitFake:true}));
});

test('cohort geometry stays inside the MiniMax-H3 maxPixels the product would otherwise silently fit to', async () => {
  const manifest=JSON.parse(stripJsoncComments(await readFile(new URL('../config/manifests/builtin.models.jsonc',import.meta.url),'utf8')).replace(/,(\s*[}\]])/g,'$1'));
  for(const id of ['minimax_h3','minimax_h3_ref'])assert.equal(manifest.models.find(entry=>entry.id===id)?.limits?.maxPixels,MINIMAX_H3_MAX_PIXELS,`${id} limits.maxPixels`);
  validateBootstrapRequest(request(),{backend:'mlx',unitFake:true});
  // Campaign 40874652324's dispatch asked 1536x768 for the 768p recipe; VideoRequest::from_payload
  // fits it to 1408x704 while the stored payload still says 1536x768.
  const wide=request();Object.assign(wide.bootstrap.operator,{wideWidth:1536,wideHeight:768});
  assert.throws(()=>validateBootstrapRequest(wide,{backend:'mlx',unitFake:true}),/exceeds MiniMax-H3 limits.maxPixels/);
  for(const lane of ['cuda','mlx']){const op=JSON.parse(await readFile(new URL(`../docs/testing/sc-26691-campaign/h3-terminal-${lane}.json`,import.meta.url),'utf8'));const found=[];JSON.stringify(op,(k,v)=>{if(v&&typeof v==='object'&&'wideWidth' in v)found.push(v);return v;});assert.ok(found.length>0,lane);for(const o of found)for(const [w,h] of [[o.width,o.height],[o.wideWidth,o.wideHeight]])assert.ok(w*h<=MINIMAX_H3_MAX_PIXELS,`${lane} dispatch kit ${w}x${h}`);}
});
