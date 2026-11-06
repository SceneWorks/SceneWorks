#!/usr/bin/env node
// SW-F329: one bounded three-anchor campaign. Manual Actions only; no ordinary CI compute.
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { copyFile, link, mkdir, open, readFile, readdir, realpath, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { execFile } from 'node:child_process';
import { fileSha256 } from './lib/file-sha256.mjs';
import { OwnedService, ProcessObservations, processSample, assertQueuedMatchesSubmitted, bindCompletedMedia, hardware } from './h3-finetune-terminal-runtime.mjs';
import { runScenario } from './h3-finetune-terminal-api.mjs';
import { authorized, sourcePair, verifyRenderedContent, decodedDelta, validateAVAgainstRequest } from './h3-finetune-terminal.mjs';

const exec=promisify(execFile), GIB=1024**3;
const json=async file=>JSON.parse(await readFile(file,'utf8'));
const save=async(file,value)=>writeFile(file,JSON.stringify(value,null,2)+'\n',{flag:'wx'});
const SOURCES={
  tier:{repo:'SceneWorks/minimax-h3-mlx',revision:'137ce668c55a20bc0935fd1cf2a3de8448abb7f4',prefixes:['q4/transformer/','q4/text_encoder/']},
  base:{repo:'MiniMaxAI/MiniMax-H3',revisions:['6818f6c32d12b210915e44ad56a4228c2608f160','939557dc319dd91227e30195a763f272ba7f8765'],prefixes:['vae/','audio_vae/','FL2VA/audio_vae/','tokenizer/','processor/','audio_scheduler/','scheduler/']},
  predictor:{repo:'Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview',revision:'9a1fd3a41b4a754a7886e64e82edbddf599fd1bd',files:['minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors']},
  turbo:{repo:'lightx2v/Minimax-h3-Turbo',revision:'5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1',files:['minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors']},
};
const sha40=value=>assert.match(value??'',/^[a-f0-9]{40}$/,'exact source revision required');
const sha64=value=>assert.match(value??'',/^[a-f0-9]{64}$/,'exact input content commitment required');
const bytes=(value,name)=>assert.ok(Number.isSafeInteger(value)&&value>0,`${name}: positive integer bytes required`);
const absolute=value=>assert.ok(typeof value==='string'&&path.win32.isAbsolute(value),'absolute Windows path required');

export function validatePlan(p){
  assert.equal(p.version,1);assert.equal(p.stationary_pair_authorized,true);
  sha40(p.sceneworks);sha40(p.inference);sha64(p.requestSha256);
  assert.ok(['cuda-windows','cuda-windows-2'].includes(p.expectedRunner));
  absolute(p.requestPath);absolute(p.licenseAcknowledgmentsPath);
  assert.ok(Number.isInteger(p.seed)&&p.seed>=0&&p.seed<=0xffffffff,'explicit chosen seed required');
  bytes(p.poolBytes,'poolBytes');bytes(p.externalReserveBytes,'externalReserveBytes');
  assert.equal(p.poolBytes+p.externalReserveBytes,24*GIB,'24 GiB includes the non-pool reserve');
  assert.ok(p.externalReserveBytes>=2*GIB,'explicit context/library/foreign reserve at least2GiB required');
  bytes(p.hostBudgetBytes,'hostBudgetBytes');bytes(p.workstationDeviceBytes,'workstationDeviceBytes');
  assert.ok(p.workstationDeviceBytes<=96*GIB,'reported workstation envelope cannot be enlarged');
  assert.ok(Number.isInteger(p.timeoutSeconds)&&p.timeoutSeconds>=60&&p.timeoutSeconds<=7200);
  assert.equal(p.additionalSparse24,true,'normal sparse24GiB anchor is required for resident/window Veda equivalence');
  assert.ok(typeof p.additionalSparse24Reason==='string'&&p.additionalSparse24Reason.trim(),'focused additional sparse render requires the concrete evidence gap');
  assert.deepEqual(Object.keys(p.sources).sort(),Object.keys(SOURCES).sort());
  for(const [role,spec] of Object.entries(SOURCES)){
    const source=p.sources[role];absolute(source.root);sha40(source.revision);
    assert.ok((spec.revisions??[spec.revision]).includes(source.revision),`${role}: wrong immutable snapshot`);
    assert.ok(source.root.replaceAll('\\','/').endsWith(`/models--${spec.repo.replaceAll('/','--')}/snapshots/${source.revision}`),'source root must identify its exact HF snapshot');
  }
  absolute(p.primitiveProof.directory);assert.match(p.primitiveProof.runUrl,/^https:\/\/github\.com\/SceneWorks\/inference\/actions\/runs\/\d+$/);
  absolute(p.materializingProof.logPath);sha40(p.materializingProof.revision);
  assert.match(p.materializingProof.runUrl,/^https:\/\/github\.com\/SceneWorks\/inference\/actions\/runs\/\d+$/);
  return p;
}

export function anchorRequest(original,seed){
  assert.equal(original.model,'minimax_h3');assert.equal(original.mode,'text_to_video');
  assert.equal(original.width,1344);assert.equal(original.height,768);assert.equal(original.fps,24);assert.equal(original.duration,14.375);
  assert.equal(original.quality,'balanced');assert.equal(original.fitMode,'crop');
  assert.ok(typeof original.prompt==='string'&&original.prompt.length>0);
  const loras=original.loras;assert.equal(loras?.length,1);assert.equal(loras[0].id,'minimax_h3_turbo_8step');assert.equal(loras[0].weight,1);
  assert.deepEqual(original.advanced?.veda,{enabled:true,fullAttentionLayers:'',fullAttentionSteps:'',generated:'90%',reference:'90%'});
  for(const name of ['referenceAssetIds','sourceClipAssetIds','referenceAudioAssetIds'])assert.equal(original[name]?.length??0,0,'this exact anchor has no reference inputs');
  for(const name of ['sourceAssetId','lastFrameAssetId','referenceClipAssetId','sourceClipAssetId'])assert.ok(original[name]==null,'this exact anchor has no source conditioning');
  const advanced=structuredClone(original.advanced);
  assert.ok(advanced.steps===undefined||advanced.steps===8,'original Turbo8step recipe cannot change');
  assert.ok(advanced.mlxQuantize===undefined||advanced.mlxQuantize===4,'original q4 tier cannot change');
  advanced.mlxQuantize=4; // make the original catalog-default q4 selection explicit
  return {model:original.model,mode:original.mode,prompt:original.prompt,negativePrompt:original.negativePrompt??'',width:1344,height:768,fps:24,duration:14.375,quality:original.quality,fitMode:original.fitMode,seed,loras:[{id:loras[0].id,weight:loras[0].weight}],advanced,requestedGpu:'1'};
}

function parseEvents(raw){const out=[];for(const line of raw.split('\n'))try{const value=JSON.parse(line);out.push(value.fields??value);}catch{}return out;}
function numeric(value,name){assert.ok(Number.isSafeInteger(value)&&value>=0,`${name}: actual integer observation required`);return value;}
export function validatePoolEvidence(events,{jobId,poolBytes,externalReserveBytes,metrics}){
  const snapshots=events.filter(e=>e.event==='h3_cuda_proof_pool_snapshot'&&String(e.jobId??e.job_id)===jobId&&e.cap_bytes!==undefined);
  assert.ok(snapshots.length,'actual worker capped-pool observations required');
  for(const s of snapshots){assert.equal(s.logical_ordinal,0);assert.equal(s.cap_bytes,poolBytes);assert.ok(numeric(s.reserved_high_bytes,'reserved_high_bytes')<=poolBytes,'pool reservation exceeded enforced budget');assert.ok(numeric(s.used_high_bytes,'used_high_bytes')>0,'pool did not witness actual tensors');}
  const outside=snapshots.filter(s=>s.device_used_bytes!==undefined).map(s=>{
    const whole=numeric(s.device_used_bytes,'device_used_bytes'),reserved=numeric(s.reserved_bytes,'reserved_bytes');
    assert.ok(whole>=reserved,'outside-pool attribution is unavailable/inconsistent, never zero');
    const value=whole-reserved;assert.ok(value<=externalReserveBytes,'observed outside-pool allocations exceeded the declared reserve');return value;
  });
  assert.ok(outside.length,'actual paired whole-device/current-pool observations required for outside-pool attribution');
  assert.ok(metrics.samples?.length>=2,'whole-device/RSS observations required');
  for(const sample of metrics.samples){assert.ok(sample.host>0&&sample.device>0,'missing observations are not zero');assert.ok(sample.device<=poolBytes+externalReserveBytes,'whole device exceeded24GiB');}
  // Continuous pool high-water plus the entire explicitly reserved outside-pool allowance.
  const reservedHighBytes=Math.max(...snapshots.map(s=>s.reserved_high_bytes));
  assert.ok(reservedHighBytes+externalReserveBytes<=24*GIB);
  return {reservedHighBytes,externalReserveBytes,observedOutsidePoolPeakBytes:Math.max(...outside),totalEnvelopeBytes:24*GIB,scope:'driver-continuous pool; separately reserved context/library/foreign allocations; sampled outside-pool/whole-device/RSS blind spots retained'};
}

export function validateMaterializingProofLog(log){
  assert.match(log,/test result: ok\. 1 passed; 0 failed; 0 ignored/);
  assert.match(log,/MODEL_INFERENCE_BODY_COMPLETED veda::tests::actual_cuda_kernel_gate/,'late completed CUDA test body required');
  assert.match(log,/H3_DENSE_ATTENTION_ROUTES/);
  const materialized=Number(log.match(/materialized_calls:\s*(\d+)/)?.[1]);assert.ok(materialized>0,'forced Veda-dense alone is not materializing-route evidence');
  const receipts=[...log.matchAll(/S3_DEVICE_RECEIPT (\{[^\n]+\})/g)].map(match=>JSON.parse(match[1]));
  for(const dtype of ['BF16','F16'])assert.ok(receipts.some(row=>row.schema==='sc-25777-actual-cuda-kernel-v2'&&row.device_ordinal===0&&row.dtype===dtype&&row.sparse_calls>0&&row.forced_failure_dense===true),'actual CUDA kernel receipts must cover both output dtypes');
  return {materializedCalls:materialized,receipts};
}


// Existing CUDA H3 terminal operator maximumBaselineDelta: 0.001 (sc-26691 campaign).
// Keep that same-backend RMS ceiling fixed before observing this campaign. Full native
// frames/channels below strengthen the existing sampled decodedDelta witness; neither
// cross-backend tolerances nor forced-dense outputs are an equality oracle.
export const VEDA_EQUIVALENCE = Object.freeze({video:0.001,audio:0.001,
  basis:'sc-26691 CUDA same-backend maximumBaselineDelta=0.001; full native decoded RGB and stereo PCM RMS'});

export function validateVedaParityPair(records){
  const take=id=>{const found=records.filter(r=>r.id===id);assert.equal(found.length,1,`one actual ${id} anchor required`);return found[0];};
  const resident=take('workstation-veda'),streamed=take('24gib-veda-sparse');
  assert.notEqual(resident.job?.id,streamed.job?.id,'independent completed generations required');
  assert.deepEqual(resident.sourcePair,streamed.sourcePair,'Veda parity source pair changed');
  sha40(resident.sourcePair?.sceneworks);sha40(resident.sourcePair?.inference);
  assert.deepEqual(resident.settings,streamed.settings,'Veda parity changed creative settings');
  const witnesses=[];
  for(const [record,windowed] of [[resident,false],[streamed,true]]){
    assert.equal(record.passed,true);assert.equal(record.job?.status,'completed');
    assert.ok(typeof record.job.id==='string'&&record.job.id.length>0);
    assertQueuedMatchesSubmitted({...record.settings,projectId:record.job.payload?.projectId},record.job);
    assert.equal(record.settings.advanced?.veda?.enabled,true);
    const veda=record.generationMetrics?.vedaExecution??record.generationMetrics?.veda_execution;
    assert.equal(veda?.requested,true);assert.ok(numeric(veda.sparseCalls??veda.sparse_calls,'actual sparse calls')>0,'both parity anchors must execute sparse attention');
    assert.ok(typeof veda.kernel==='string'&&veda.kernel.length>0,'actual sparse kernel identity required');
    assert.ok(!/forced for validation/.test(veda.fallbackReason??veda.fallback_reason??''),'forced dense is not the Veda equality oracle');
    assert.equal(record.media?.length,1);const media=record.media[0];
    assert.equal(media.jobId,record.job.id);sha64(media.sha256);assert.equal(media.decoded?.sha256,media.sha256);
    assert.equal(media.av,path.basename(media.av),'bound media path must stay in its anchor directory');
    validateAVAgainstRequest(media.decoded,record.settings,{model:'minimax_h3',fps:24,width:1344,height:768,audio:{required:true,sampleRate:32000,channels:2,codec:'aac'}});
    const selections=record.events?.filter(e=>e.event==='video_memory_strategy_selected'&&e.route==='minimax_h3'&&e.request_frames===345)??[];
    const strategy=windowed?'BoundedTransformerResidency':'Resident';
    assert.ok(selections.length&&selections.every(e=>e.strategy===strategy),'actual selected residency differs from the parity route');
    const executions=record.events.filter(e=>e.event==='h3_attention_execution');
    assert.equal(executions.length,1,'one completed request execution-policy witness required');
    const execution=executions[0];
    assert.equal(execution.transformer_window,windowed?'Some(1)':'None');
    assert.equal(execution.transformer_component,windowed?'Some(Both)':'None');
    const transfers=record.events.filter(e=>e.event==='h3_transformer_window_materialized');
    if(windowed){
      for(const component of ['text_encoder','dit']){
        const rows=transfers.filter(e=>e.component===component);assert.ok(rows.length>0,`actual ${component} transfer windows required`);
        for(const row of rows){assert.equal(row.block_count,1);assert.ok(numeric(row.base_bytes,'transferred bytes')>0);assert.ok(Number.isInteger(row.first_block)&&row.first_block>=0);assert.match(row.device,/Cuda\s*\{\s*gpu_id:\s*0\s*\}/,'actual CUDA transfer device required');}
      }
    }else assert.equal(transfers.length,0,'resident anchor must not silently execute a transformer window');
    witnesses.push({anchor:record.id,jobId:record.job.id,media:media.av,sha256:media.sha256,strategy,execution,sparseCalls:veda.sparseCalls??veda.sparse_calls,kernel:veda.kernel});
  }
  const left=resident.generationMetrics.vedaExecution??resident.generationMetrics.veda_execution;
  const right=streamed.generationMetrics.vedaExecution??streamed.generationMetrics.veda_execution;
  for(const key of ['kernel','predictor','plan','planMatch','executionGrid','planGrid'])assert.deepEqual(right[key],left[key],`Veda parity changed ${key}`);
  return {resident,streamed,witnesses};
}

// Read fixed-size chunks, never materialize a long-clip RGB vector in JS memory.
export async function rawNumericDifference(a,b,{stream,expectedElements,channels}){
  assert.ok(['video','audio'].includes(stream));assert.ok(Number.isInteger(channels)&&channels>0);
  const width=stream==='video'?1:4,[as,bs]=await Promise.all([stat(a),stat(b)]);
  assert.ok(as.size>0&&as.size===bs.size&&as.size%width===0,'decoded stream length differs or is empty');
  const elements=as.size/width;assert.equal(elements%channels,0,'incomplete channel tuple');
  if(expectedElements!==undefined)assert.equal(elements,expectedElements,'full decoded frame/sample coverage required');
  const files=await Promise.all([open(a,'r'),open(b,'r')]),buffers=[Buffer.alloc(64*1024),Buffer.alloc(64*1024)];
  let squared=0,maxAbs=0;const channelSquares=Array(channels).fill(0);
  try{
    for(let offset=0;offset<as.size;offset+=buffers[0].length){
      const length=Math.min(buffers[0].length,as.size-offset);
      await Promise.all(files.map(async(file,j)=>{let n=0;while(n<length){const read=await file.read(buffers[j],n,length-n,offset+n);assert.ok(read.bytesRead>0,'decoded stream truncated during comparison');n+=read.bytesRead;}}));
      for(let i=0;i<length;i+=width){
        const x=width===1?buffers[0][i]/255:buffers[0].readFloatLE(i),y=width===1?buffers[1][i]/255:buffers[1].readFloatLE(i);
        assert.ok(Number.isFinite(x)&&Number.isFinite(y),'non-finite decoded PCM value');
        const d=Math.abs(x-y),sq=d*d;squared+=sq;maxAbs=Math.max(maxAbs,d);channelSquares[((offset+i)/width)%channels]+=sq;
      }
    }
  }finally{await Promise.all(files.map(f=>f.close()));}
  return {elements,channels,rmse:Math.sqrt(squared/elements),maxAbs,channelRmse:channelSquares.map(sum=>Math.sqrt(sum/(elements/channels))),coverage:'all native-resolution frames or all native-rate interleaved audio samples'};
}

export async function compareDecodedMedia(a,b,dir,{videoElements,audioSamples,audioPadding=0,audioChannels=2,command=exec,sampledDelta=decodedDelta}={}){
  assert.ok(Number.isSafeInteger(videoElements)&&videoElements>0&&videoElements%3===0,'native video element count required');
  assert.ok(Number.isSafeInteger(audioSamples)&&audioSamples>0&&Number.isSafeInteger(audioPadding)&&audioPadding>=0,'native audio sample coverage required');
  const report={policy:VEDA_EQUIVALENCE,streams:{}};const raw=[];
  try{
    for(const stream of ['video','audio']){
      const files=[];
      for(const [index,file] of [a,b].entries()){
        const target=path.join(dir,`veda-parity-${randomUUID()}-${index}-${stream}.raw`);raw.push(target);files.push(target);
        const args=stream==='video'?['-map','0:v:0','-an','-vsync','0','-pix_fmt','rgb24','-f','rawvideo']:['-map','0:a:0','-vn','-c:a','pcm_f32le','-f','f32le'];
        await command('ffmpeg',['-v','error','-xerror','-nostdin','-n','-i',file,...args,target],{timeout:300000,maxBuffer:1048576});
      }
      if(stream==='audio')for(const file of files){const count=(await stat(file)).size/(4*audioChannels);assert.ok(Number.isInteger(count)&&count>=audioSamples-audioPadding&&count<=audioSamples+audioPadding,'full decoded audio sample coverage required');}
      const full=await rawNumericDifference(...files,{stream,channels:stream==='video'?3:audioChannels,expectedElements:stream==='video'?videoElements:undefined});
      if(stream==='audio')assert.ok(full.elements/audioChannels>=audioSamples-audioPadding&&full.elements/audioChannels<=audioSamples+audioPadding,'full decoded audio sample coverage required');
      report.streams[stream]={...full,maximumRmse:VEDA_EQUIVALENCE[stream],sampledRmse:await sampledDelta(a,b,stream)};
      assert.ok(full.rmse<=VEDA_EQUIVALENCE[stream]&&full.channelRmse.every(v=>v<=VEDA_EQUIVALENCE[stream]),`${stream} full decoded Veda residency comparison exceeds fixed same-backend RMS ceiling`);
      assert.ok(Number.isFinite(report.streams[stream].sampledRmse)&&report.streams[stream].sampledRmse>=0&&report.streams[stream].sampledRmse<=VEDA_EQUIVALENCE[stream],`${stream} existing decodedDelta witness exceeds same-backend RMS ceiling`);
    }
    return report;
  }catch(error){error.comparison=report;throw error;}
  finally{await Promise.all(raw.map(file=>rm(file,{force:true})));}
}

export async function compareVedaAnchors(records,output,{compare=compareDecodedMedia}={}){
  let report={kind:'cuda-veda-resident-window-equivalence',passed:false,policy:VEDA_EQUIVALENCE};
  try{
    const {resident,streamed,witnesses}=validateVedaParityPair(records);
    const files=[resident,streamed].map(record=>path.join(output,record.id,record.media[0].av));
    for(const [i,file] of files.entries())assert.equal(await fileSha256(file),witnesses[i].sha256,'bound parity media changed');
    report={...report,sourcePair:resident.sourcePair,anchors:witnesses};
    const verification=validateAVAgainstRequest(resident.media[0].decoded,resident.settings,{model:'minimax_h3',fps:24,width:1344,height:768,audio:{required:true,sampleRate:32000,channels:2,codec:'aac'}});
    report.comparison=await compare(...files,output,{videoElements:1344*768*345*3,audioSamples:verification.audio.expectedSamples,audioPadding:verification.audio.frameSamples+1});
    for(const [i,file] of files.entries())assert.equal(await fileSha256(file),witnesses[i].sha256,'bound parity media changed while decoding');
    report.passed=true;return report;
  }catch(error){report.error=error.stack;if(error.comparison)report.comparison=error.comparison;throw error;}
  finally{await save(path.join(output,'veda-equivalence.json'),report);}
}

async function walk(root,prefix=''){
  const out=[];for(const entry of await readdir(path.join(root,prefix),{withFileTypes:true})){
    const relative=prefix+entry.name;if(entry.isDirectory())out.push(...await walk(root,relative+'/'));
    else out.push(relative);
  }return out.sort();
}
async function stageModels(p,output){
  const hub=path.join(output,'staged','hub'),files=[];await mkdir(hub,{recursive:true});
  for(const [role,spec] of Object.entries(SOURCES)){
    const source=p.sources[role],repoRoot=path.join(hub,`models--${spec.repo.replaceAll('/','--')}`),snapshot=path.join(repoRoot,'snapshots',source.revision);
    const selected=(await walk(source.root)).filter(file=>spec.files?.includes(file)||spec.prefixes?.some(prefix=>file.startsWith(prefix))||['model_index.json','modular_model_index.json','LICENSE'].includes(file));
    assert.ok(selected.length,`${role}: no source files selected`);
    for(const relative of selected){
      const input=path.join(source.root,relative),resolved=await realpath(input),before=await stat(input);assert.ok(before.isFile()&&before.size>0,`${role}/${relative}: empty or unavailable source`);
      // HF snapshots may link to their immutable repository-local blobs, never an arbitrary root.
      const namespace=await realpath(path.dirname(path.dirname(source.root))),rel=path.relative(namespace,resolved);assert.ok(rel&&!rel.startsWith('..')&&!path.isAbsolute(rel),'source link escaped its repository namespace');
      const destination=path.join(snapshot,relative);await mkdir(path.dirname(destination),{recursive:true});
      let method='hardlink';try{await link(resolved,destination);}catch(error){if(error.code!=='EXDEV')throw error;method='cross-volume-copy-once';await copyFile(resolved,destination);}
      const digest=await fileSha256(destination),after=await stat(input);assert.equal(after.size,before.size);assert.equal(after.mtimeMs,before.mtimeMs,'source changed during staging');
      assert.equal((await stat(destination)).size,before.size);files.push({role,repo:spec.repo,revision:source.revision,relative,source:input,path:destination,bytes:before.size,sha256:digest,method});
    }
    await mkdir(path.join(repoRoot,'refs'),{recursive:true});await writeFile(path.join(repoRoot,'refs','main'),source.revision+'\n',{flag:'wx'});
    // Every selected source index must resolve all its actual weight shards.
    for(const file of files.filter(file=>file.role===role&&file.relative.endsWith('.index.json'))){const index=await json(file.path);for(const shard of new Set(Object.values(index.weight_map??{})))assert.ok((await stat(path.join(path.dirname(file.path),shard))).size>0,'incomplete staged shard index');}
  }
  const receipt={kind:'sw-f329-staged-models',sourcePair:{sceneworks:p.sceneworks,inference:p.inference},files,meaning:'payloads staged/hashed once; later anchors hardlink these owned immutable bytes, with no shared-cache writes'};
  await save(path.join(output,'staged-models.json'),receipt);return receipt;
}

// Reuse process/build/request ownership. Stock sources are bound once above instead of going
// through the imported-checkpoint campaign's separate25case authority and staging inventory.
class StockService extends OwnedService {
  async assertOwner(){
    const owner=await json(this.ownerPath);assert.equal(owner.token,this.owner);assert.equal(owner.pid,process.pid);
    const current=await stat(this.binary);assert.equal(current.size,this.binaryStamp.size);assert.equal(current.mtimeMs,this.binaryStamp.mtimeMs,'owned binary changed');
  }
  async initialize(){
    authorized(this.p,this.c);await sourcePair(this.p,this.swRoot,this.inferenceRoot);await hardware('cuda',this.c.runtime.deviceId);
    await this.refuseLeftoverOwnedProcesses();
    this.runDir=path.join(this.p.output,'services',this.c.id);await mkdir(this.runDir,{recursive:true});this.owner=randomUUID();
    for(const folder of ['config','data','data/cache','data/models','hf/hub','cache'])await mkdir(path.join(this.runDir,folder),{recursive:true});
    this.ownerPath=path.join(this.runDir,'owner.json');await save(this.ownerPath,{token:this.owner,pid:process.pid,sceneworks:this.p.sceneworks,inference:this.p.inference,proof_kind:'campaign_native'});
    this.seeded=new Set();this.seededInputs=new Map();
    for(const file of this.staged.files){const dest=path.join(this.runDir,'hf','hub',`models--${file.repo.replaceAll('/','--')}`,'snapshots',file.revision,file.relative);await mkdir(path.dirname(dest),{recursive:true});await link(file.path,dest);}
    for(const [role,spec] of Object.entries(SOURCES)){const source=this.p.sources[role],ref=path.join(this.runDir,'hf','hub',`models--${spec.repo.replaceAll('/','--')}`,'refs','main');await mkdir(path.dirname(ref),{recursive:true});await writeFile(ref,source.revision+'\n',{flag:'wx'});}
    const acknowledgment=await json(this.p.licenseAcknowledgmentsPath);assert.equal(acknowledgment.schemaVersion,1);assert.ok(acknowledgment.acknowledgments?.minimax_h3,'existing explicit base-model license acknowledgment required');
    const accepted=Object.fromEntries(Object.entries(acknowledgment.acknowledgments).filter(([key])=>key.startsWith('minimax_h3')));await save(path.join(this.runDir,'config','license-acknowledgments.json'),{schemaVersion:1,acknowledgments:accepted});
    await this.build();this.binaryStamp=await stat(this.binary);
    try{await this.start();return this;}
    catch(error){
      // Reuse the owned guardian's parent-death drain; never contact/kill a foreign process.
      this.worker?.stdin.end();this.guardian?.stdin.end();
      const exits=[];for(const child of [this.worker,this.guardian,this.api].filter(Boolean)){
        let timer;try{exits.push({pid:child.pid,...await Promise.race([child.exit(),new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('owned startup cleanup exceeded180s; no forced kill')),180000);})])});}
        catch(failure){exits.push({pid:child.pid,error:failure.message});}finally{clearTimeout(timer);}
      }
      this.stopped=true;await save(path.join(this.dir,'startup-failure.json'),{error:error.message,exits});throw error;
    }
  }
}

async function retainedProofs(p,inferenceRoot){
  assert.equal((await readFile(path.join(p.primitiveProof.directory,'source.txt'),'utf8')).trim(),p.inference,'primitive allocator proof must name the accepted inference main commit');
  const log=await readFile(path.join(p.primitiveProof.directory,'allocator-proof.log'),'utf8');
  assert.match(log,/test result: ok\. 1 passed; 0 failed; 0 ignored/);const marker=log.match(/CUDA_PROOF_POOL_EVIDENCE (\{[^\n]+\})/);assert.ok(marker,'positive allocator enforcement evidence required');const primitive=JSON.parse(marker[1]);assert.equal(primitive.overCapRefused,true);assert.equal(primitive.bothConstructors,true);assert.equal(primitive.cublasStreamResetRebound,true);assert.ok(primitive.reservedHighBytes<=primitive.capBytes);
  const numericLog=await readFile(p.materializingProof.logPath,'utf8'),materializing=validateMaterializingProofLog(numericLog),sources=[];
  for(const selected of ['crates/media/candle-gen/candle-gen-minimax-h3','crates/media/candle-gen/candle-gen/src/attention.rs','crates/kernels/candle-quant-kernels/src']){
    const blob=async revision=>(await exec('git',['rev-parse',`${revision}:${selected}`],{cwd:inferenceRoot})).stdout.trim();
    const expected=await blob(p.materializingProof.revision),actual=await blob(p.inference);assert.equal(expected,actual,`numerical evidence source changed: ${selected}`);sources.push({path:selected,object:actual});
  }
  return {primitive:{...primitive,runUrl:p.primitiveProof.runUrl},materializing:{...materializing,revision:p.materializingProof.revision,sources,runUrl:p.materializingProof.runUrl,logSha256:await fileSha256(p.materializingProof.logPath)}};
}

export async function run(p,swRoot,inferenceRoot,output){
  validatePlan(p);assert.equal(process.platform,'win32');assert.equal(process.env.GITHUB_ACTIONS,'true');assert.equal(process.env.GITHUB_EVENT_NAME,'workflow_dispatch');assert.equal(process.env.RUNNER_NAME,p.expectedRunner);assert.equal(process.env.GITHUB_SHA,p.sceneworks);
  assert.ok(path.resolve(output).toLowerCase().startsWith('d:\\codex\\work\\'),'all generated output belongs onD');
  await mkdir(output);await sourcePair({...p,output},swRoot,inferenceRoot);assert.equal(await fileSha256(p.requestPath),p.requestSha256,'original request changed');
  const original=await json(p.requestPath),request=anchorRequest(original,p.seed);await save(path.join(output,'request.json'),{original,chosenSeed:p.seed,originalSeed:original.seed,request});
  const proofs=await retainedProofs(p,inferenceRoot);await save(path.join(output,'focused-proofs.json'),proofs);
  const inventory=await hardware('cuda',process.env.CUDA_VISIBLE_DEVICES);await save(path.join(output,'hardware.json'),inventory);
  const staged=await stageModels(p,output),results=[];
  const anchors=[['workstation-veda',false,false],['24gib-veda-dense',true,true],['24gib-veda-sparse',true,false]];
  for(const [id,capped,forceDense] of anchors){
    const dir=path.join(output,id);await mkdir(dir);const parameters={...p,output};
    const c={id,backend:'cuda',timeoutSeconds:p.timeoutSeconds,loadHostBudgetBytes:p.hostBudgetBytes,loadDeviceBudgetBytes:capped?24*GIB:p.workstationDeviceBytes,declaredSourceBindings:[{family:'h3',model:{modelId:'minimax_h3',baseModel:'minimax_h3'}}],runtime:{port:0,gpuId:'1',deviceId:process.env.CUDA_VISIBLE_DEVICES,cudaProofFeatures:['sceneworks-rust-api/h3-memory-proof'],memoryCap:capped?{kind:'cuda-vram-gb',value:24}:undefined,cudaProofEnvironment:capped?{SCENEWORKS_CUDA_PROOF_POOL_BYTES:String(p.poolBytes),...(forceDense?{SCENEWORKS_H3_VEDA_FORCE_DENSE:'1'}:{})}:{}}};
    const service=new StockService(parameters,c,swRoot,inferenceRoot,dir,{hardDeadlineMs:Date.now()+(p.timeoutSeconds+900)*1000});service.staged=staged;
    let observer,currentStage='before-submit',record={id,sourcePair:{sceneworks:p.sceneworks,inference:p.inference},settings:request,startedAt:new Date().toISOString(),passed:false};
    try{
      await service.initialize();observer=new ProcessObservations(service,{intervalMs:250,sample:async s=>({...await processSample(s),stage:currentStage})}).start();
      const transcript=[],scenario={caseId:id,baseUrl:'http://127.0.0.1:0',timeoutSeconds:p.timeoutSeconds,steps:[{id:'project',method:'POST',path:'/api/v1/projects',body:{name:`SW-F329 ${id}`},expectedStatus:201},{id:'submit',method:'POST',path:'/api/v1/video/jobs',body:{...request,projectId:{$capture:'project/id'}},expectedStatus:201},{id:'complete',waitJob:{$capture:'submit/id'},assert:[{pointer:'/status',equals:'completed'}]}]};
      const executed=await runScenario(scenario,{baseUrl:()=>service.baseUrl,fetchImpl:(...args)=>service.fetch(...args),beforeStep:({step,captures})=>service.checkStep(step,captures),onRecord:async row=>{transcript.push(row);if(row.result?.stage)currentStage=row.result.stage;if(row.method==='POST'&&row.route==='/api/v1/video/jobs'&&row.result?.id)service.jobs.add(row.result.id);await writeFile(path.join(dir,'api-transcript.json'),JSON.stringify(transcript,null,2)+'\n');}});
      const job=executed.captures.complete;record.job=job;assertQueuedMatchesSubmitted({...request,projectId:executed.captures.project.id},job);
      const metric=await service.request('GET',`/api/v1/jobs/${job.id}/metrics`);record.generationMetrics=metric;
      const veda=metric.vedaExecution??metric.veda_execution;assert.ok(veda,'actual Veda execution report required');
      if(forceDense){assert.equal(veda.sparseCalls??veda.sparse_calls,0);assert.match(veda.fallbackReason??veda.fallback_reason??'',/forced for validation/);}else assert.ok((veda.sparseCalls??veda.sparse_calls)>0,'normal Veda anchor must actually execute sparse attention');
      record.media=await bindCompletedMedia(service,job,dir,'video');assert.equal(record.media.length,1);
      record.content=await verifyRenderedContent(path.join(dir,record.media[0].av));
      const raw=record.media[0].asset.recipe?.rawAdapterSettings;assert.ok(raw,'actual resolved generation settings required');assert.equal(raw.effectiveSteps,8);assert.equal(raw.minimaxH3Tier,'q4');assert.equal(metric.steps,8);
      record.observations=await observer.finish();record.samples=observer.samples;observer=undefined;
      record.sampledStagePeaks=Object.fromEntries([...new Set(record.samples.map(sample=>sample.stage))].map(stage=>{const group=record.samples.filter(sample=>sample.stage===stage);return [stage,{hostPeakBytes:Math.max(...group.map(sample=>sample.host)),devicePeakBytes:Math.max(...group.map(sample=>sample.device??NaN)),samples:group.length,scope:'sampled against actual product job stage; may miss transients, pool watermark remains full-process continuous'}];}));
      const events=parseEvents(service.worker.log());record.events=events;
      if(capped)record.envelope=validatePoolEvidence(events,{jobId:job.id,poolBytes:p.poolBytes,externalReserveBytes:p.externalReserveBytes,metrics:{samples:record.samples}});
      record.passed=true;
    }catch(error){record.error=error.stack;throw error;}
    finally{if(observer)try{record.observations=await observer.finish();}catch(error){record.observationError=error.message;}if(service.worker)try{await service.stop();}catch(error){record.passed=false;record.shutdownError=error.message;}if(service.binary)try{assert.equal(await fileSha256(service.binary),service.binaryHash,'owned executable changed during the anchor');}catch(error){record.passed=false;record.binaryError=error.message;}record.endedAt=new Date().toISOString();await save(path.join(dir,'result.json'),record);}
    assert.equal(record.passed,true,record.shutdownError);results.push(record);
  }
  const equivalence=await compareVedaAnchors(results,output);
  await sourcePair({...p,output},swRoot,inferenceRoot);await save(path.join(output,'result.json'),{feature:'SW-F329',passed:true,equivalence,sourcePair:{sceneworks:p.sceneworks,inference:p.inference},anchors:results.map(result=>({id:result.id,jobId:result.job.id,passed:result.passed,envelope:result.envelope})),proofs});
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  try{const [mode,swRoot,inferenceRoot,output]=process.argv.slice(2),p=JSON.parse(process.env.SW_F329_PLAN??'null');validatePlan(p);if(mode==='check')console.log('SW-F329 plan shape and explicit limits checked; no model executed');else{assert.equal(mode,'run');await run(p,path.resolve(swRoot),path.resolve(inferenceRoot),path.resolve(output));}}
  catch(error){console.error(error.stack);process.exitCode=1;}
}
