#!/usr/bin/env node
// SW-F329: one bounded three-anchor campaign. Manual Actions only; no ordinary CI compute.
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { copyFile, link, mkdir, open, readFile, readdir, realpath, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { execFile } from 'node:child_process';
import { fileSha256, sha256 } from './lib/file-sha256.mjs';
import { parse as parseProvenanceCandidates, populationSha256 as provenancePopulationSha256 } from './scan-inference-provenance.mjs';
import { rustCodeMask, withoutRustTestModules } from './lib/rust-source-slices.mjs';
import { rustFunctionSlices } from './lib/rust-facade-source.mjs';
import { OwnedService, ProcessObservations, processSample, assertQueuedMatchesSubmitted, bindCompletedMedia, hardware, validateCudaRuntimeDevice } from './h3-finetune-terminal-runtime.mjs';
import { manualCudaSelection } from './ci/h3-import-native-proof.mjs';
import { runScenario } from './h3-finetune-terminal-api.mjs';
import { authorized, sourcePair, verifyRenderedContent, decodedDelta, decodeAV, validateAVAgainstRequest } from './h3-finetune-terminal.mjs';

const exec=promisify(execFile), GIB=1024**3;
const json=async file=>JSON.parse(await readFile(file,'utf8'));
const save=async(file,value)=>writeFile(file,JSON.stringify(value,null,2)+'\n',{flag:'wx'});
const SOURCES={
  tier:{repo:'SceneWorks/minimax-h3-mlx',revision:'137ce668c55a20bc0935fd1cf2a3de8448abb7f4',prefixes:['q4/transformer/','q4/text_encoder/']},
  base:{repo:'MiniMaxAI/MiniMax-H3',revisions:['6818f6c32d12b210915e44ad56a4228c2608f160','939557dc319dd91227e30195a763f272ba7f8765'],prefixes:['vae/','audio_vae/','FL2VA/audio_vae/','tokenizer/','processor/','audio_scheduler/','scheduler/']},
  predictor:{repo:'Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview',revision:'9a1fd3a41b4a754a7886e64e82edbddf599fd1bd',files:['minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors']},
  turbo:{repo:'lightx2v/Minimax-h3-Turbo',revision:'5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1',files:['minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors']},
};
// One previously accepted uncapped control, never a general anchor-resume mechanism.
export const RETAINED_WORKSTATION=Object.freeze({
  jobId:'job_b29fc9a4c1047efdabebe4eaaea1b754',
  sourcePair:Object.freeze({sceneworks:'66b5d26d94676914eba718a9af3d4c2ad3561099',inference:'073e34f21382dc25ef711e0620ebf16bc7e74b6d'}),
  runUrl:'https://github.com/SceneWorks/SceneWorks/actions/runs/42184091082',
  binarySha256:'0437977ead2b1cf7582c59fdaf581952c0b9b04001bceaeb90eaf9a8e38af03a',
  mediaSha256:'6fe29ea7d18c1288ff5b2e4e906455f989e4a16d1f20fb01d11032e08661fb6c',
  files:Object.freeze({
    'workstation-veda/result.json':'fe48b66bfaac3bd093a18188bed529af512be0fcfd57699efdd5be5606c4e5c4',
    'workstation-veda/service-build.json':'b34facb80e9fb1fee0032b01b38cac4b7cbbf6ed00ab3564d1a18e650a7cc79b',
    'workstation-veda/exits-1.json':'288878029bce877fd4318fdb4a9588c04eec2f8c4f5b62dc52754b65ea641c8e',
    'hardware.json':'6b5a14871376735772c404b00c5fbbe86e73fcbbec988d21a7f75c18e8b020e1',
    'staged-models.json':'1e4f6d59ddd436cfb032968f1e1e4ccc28d4e47861cf264ba94400740ed5f04c',
    'request.json':'3d1f60aa44a8dfb85541353374f6e8cc75e009ddd6757b423ac2ea8694a87c71',
  }),
});
const RETAINED_BOUNDED_QK_PATHS=[
  'crates/media/candle-gen/candle-gen-minimax-h3/src/dit/layers.rs',
  'crates/media/candle-gen/candle-gen-minimax-h3/src/dit/model.rs',
  'crates/contracts/gen-core/src/h3_memory/window.rs',
  'crates/contracts/gen-core-h3/src/h3_memory.rs',
];
const RETAINED_ENGINE_PATHS=[
  'crates/media/candle-gen/vendor/candle-core/src/cuda_backend/device.rs',
  'crates/media/candle-gen/vendor/candle-core/src/cuda_backend/proof_pool.rs',
  'crates/media/candle-gen/vendor/candle-core/src/cuda_backend/proof_pool/recovery.rs',
  'crates/media/candle-gen/vendor/candle-core/src/lib.rs',
  'crates/media/candle-gen/vendor/candle-core/VENDORED.md',
  'crates/media/candle-gen/candle-gen/tests/cuda_proof_pool.rs',
  'scripts/ci/cuda_memory_envelope_proof.py','scripts/tests/test_cuda_memory_envelope_proof.py',
  'scripts/ci/safe_controls.json','scripts/check_model_inference_policy.py','scripts/tests/test_test_suites.py',
  ...RETAINED_BOUNDED_QK_PATHS,
];
const RETAINED_PIN_METADATA=[
  'config/anchor-loader-closures.json','config/inference-provider-closures.json','config/inference-third-party-source.json',
  'config/inference-provenance-candidates.tsv',
];
const RETAINED_CONTROLLER_PATHS=[
  'scripts/h3-cuda-memory-terminal.mjs','scripts/h3-cuda-memory-terminal.test.mjs',
  'docs/testing/sw-f329-cuda-memory-terminal.md','.github/workflows/h3-cuda-memory-terminal.yml',
];
const RETAINED_CI_PATHS=[
  '.github/workflows/server-candle-linux.yml','scripts/select-linux-flash-cache.mjs','scripts/server-candle-linux-workflow.test.mjs',
];
const RETAINED_EQUIVALENCE=Symbol('validated SW-F329 uncapped control equivalence');
const sha40=value=>assert.match(value??'',/^[a-f0-9]{40}$/,'exact source revision required');
const sha64=value=>assert.match(value??'',/^[a-f0-9]{64}$/,'exact input content commitment required');
const bytes=(value,name)=>assert.ok(Number.isSafeInteger(value)&&value>0,`${name}: positive integer bytes required`);
const absolute=value=>assert.ok(typeof value==='string'&&path.win32.isAbsolute(value),'absolute Windows path required');
const CAMPAIGN_GPU_UUIDS=Object.freeze(['GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0','GPU-e4b79931-7be6-f216-460a-f5405cfafffe']);
export function campaignCudaSelection(p){
  const selected=manualCudaSelection(p),uuid=CAMPAIGN_GPU_UUIDS[selected.physicalGpuIndex];
  if(p.physicalGpuUuid!==undefined||selected.physicalGpuIndex===0)assert.equal(p.physicalGpuUuid,uuid,'campaign requires its canonical selected physical GPU UUID');
  if(selected.physicalGpuIndex===0)assert.equal(p.retainedWorkstation,undefined,'GPU0 requires a fresh workstation control; retained GPU1 evidence cannot be reused');
  return {physicalGpuIndex:selected.physicalGpuIndex,physicalGpuUuid:uuid};
}
export function campaignCudaEnvironment(p,env){
  const selected=campaignCudaSelection(p);assert.equal(env.SCENEWORKS_GPU_ID,String(selected.physicalGpuIndex),'worker GPU differs from selected physical index');
  assert.equal(env.CUDA_DEVICE_ORDER,'PCI_BUS_ID');assert.equal(env.CUDA_VISIBLE_DEVICES,selected.physicalGpuUuid,'campaign requires its exact selected UUID mask');return selected;
}
export function anchorRuntime(p,capped,forceDense){
  const selected=campaignCudaSelection(p);
  return {port:0,gpuId:String(selected.physicalGpuIndex),deviceId:selected.physicalGpuUuid,...selected,cudaProofFeatures:['sceneworks-rust-api/h3-memory-proof'],memoryCap:capped?{kind:'cuda-vram-gb',value:24}:undefined,cudaProofEnvironment:capped?{SCENEWORKS_CUDA_PROOF_POOL_BYTES:String(p.poolBytes),...(forceDense?{SCENEWORKS_H3_VEDA_FORCE_DENSE:'1'}:{})}:{}};
}
export function validateAnchorDevice(job,physicalGpuIndex){
  manualCudaSelection({physicalGpuIndex,physicalGpuUuid:CAMPAIGN_GPU_UUIDS[physicalGpuIndex]});
  assert.equal(String(job.requestedGpu),String(physicalGpuIndex),'anchor request differs from selected physical GPU');
  assert.equal(String(job.assignedGpu),String(physicalGpuIndex),'completed anchor assigned GPU differs from selected physical GPU');
}

export function validatePlan(p){
  campaignCudaSelection(p);
  assert.ok(p.mode===undefined||p.mode==='matched-resident-control','unknown terminal mode');
  const matched=p.mode==='matched-resident-control';
  assert.equal(p.version,1);assert.equal(p.stationary_pair_authorized,true);
  sha40(p.sceneworks);sha40(p.inference);sha64(p.requestSha256);
  assert.ok(['cuda-windows','cuda-windows-2'].includes(p.expectedRunner));
  absolute(p.requestPath);absolute(p.licenseAcknowledgmentsPath);
  assert.ok(Number.isInteger(p.seed)&&p.seed>=0&&p.seed<=0xffffffff,'explicit chosen seed required');
  bytes(p.poolBytes,'poolBytes');bytes(p.externalReserveBytes,'externalReserveBytes');
  assert.equal(p.poolBytes+p.externalReserveBytes,(matched?95:24)*GIB,'declared envelope includes the non-pool reserve');
  assert.ok(p.externalReserveBytes>=2*GIB,'explicit context/library/foreign reserve at least2GiB required');
  bytes(p.hostBudgetBytes,'hostBudgetBytes');bytes(p.workstationDeviceBytes,'workstationDeviceBytes');
  assert.ok(p.workstationDeviceBytes<=96*GIB,'reported workstation envelope cannot be enlarged');
  assert.ok(Number.isInteger(p.timeoutSeconds)&&p.timeoutSeconds>=60&&p.timeoutSeconds<=7200);
  if(matched){
    assert.equal(p.matchedResidentControlAuthorized,true,'one explicit conditional control authorization required');
    assert.equal(p.physicalGpuIndex,0);assert.equal(p.seed,42);
    assert.equal(p.requestSha256,'b5b5ba095203fab1eddb9ba1793162959328e16ee34a0ca6e19d44e2325bce23');
    assert.equal(path.win32.resolve(p.licenseAcknowledgmentsPath),path.win32.resolve('D:/Codex/work/sw-f329-acceptance-42243645213-1/services/workstation-veda/config/license-acknowledgments.json'),'control must reuse the original frozen acknowledgment');
    assert.equal(p.poolBytes,92*GIB);assert.equal(p.externalReserveBytes,3*GIB);assert.equal(p.workstationDeviceBytes,95*GIB);
    for(const key of ['retainedWorkstation','additionalSparse24','additionalSparse24Reason','primitiveProof','materializingProof','binaryPath','prebuilt','runtimeSourcePair','runtimeSceneworksRoot','runtimeInferenceRoot','controllerSourcePair'])assert.equal(p[key],undefined,'single control has fixed original runtime/build and cannot claim campaign proofs');
  }else{
    assert.equal(p.additionalSparse24,true,'normal sparse24GiB anchor is required for resident/window Veda equivalence');
    assert.ok(typeof p.additionalSparse24Reason==='string'&&p.additionalSparse24Reason.trim(),'focused additional sparse render requires the concrete evidence gap');
  }
  assert.deepEqual(Object.keys(p.sources).sort(),Object.keys(SOURCES).sort());
  for(const [role,spec] of Object.entries(SOURCES)){
    const source=p.sources[role];absolute(source.root);sha40(source.revision);
    assert.ok((spec.revisions??[spec.revision]).includes(source.revision),`${role}: wrong immutable snapshot`);
    assert.ok(source.root.replaceAll('\\','/').endsWith(`/models--${spec.repo.replaceAll('/','--')}/snapshots/${source.revision}`),'source root must identify its exact HF snapshot');
  }
  if(matched)return p;
  absolute(p.primitiveProof.directory);assert.match(p.primitiveProof.runUrl,/^https:\/\/github\.com\/SceneWorks\/inference\/actions\/runs\/\d+$/);
  if(p.primitiveProof.revision!==undefined)sha40(p.primitiveProof.revision);
  if(p.retainedWorkstation!==undefined){
    const r=p.retainedWorkstation;absolute(r.campaignDirectory);absolute(r.sourceAuditPath);sha64(r.sourceAuditSha256);
    assert.equal(r.runUrl,RETAINED_WORKSTATION.runUrl);if(r.binaryPath!==undefined)absolute(r.binaryPath);
    assert.equal(p.seed,42);assert.equal(p.workstationDeviceBytes,95*GIB);
    assert.equal(p.poolBytes,21*GIB);assert.equal(p.externalReserveBytes,3*GIB);
  }
  absolute(p.materializingProof.logPath);sha40(p.materializingProof.revision);
  assert.match(p.materializingProof.runUrl,/^https:\/\/github\.com\/SceneWorks\/inference\/actions\/runs\/\d+$/);
  return p;
}

export function anchorRequest(original,seed,physicalGpuIndex=1){
  manualCudaSelection({physicalGpuIndex,physicalGpuUuid:CAMPAIGN_GPU_UUIDS[physicalGpuIndex]});
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
  return {model:original.model,mode:original.mode,prompt:original.prompt,negativePrompt:original.negativePrompt??'',width:1344,height:768,fps:24,duration:14.375,quality:original.quality,fitMode:original.fitMode,seed,loras:[{id:loras[0].id,weight:loras[0].weight}],advanced,requestedGpu:String(physicalGpuIndex)};
}

function parseEvents(raw){const out=[];for(const line of raw.split('\n'))try{const value=JSON.parse(line);out.push(value.fields??value);}catch{}return out;}
function numeric(value,name){assert.ok(Number.isSafeInteger(value)&&value>=0,`${name}: actual integer observation required`);return value;}
export function validatePoolEvidence(events,{jobId,poolBytes,externalReserveBytes,metrics,totalEnvelopeBytes=24*GIB}){
  assert.equal(totalEnvelopeBytes,poolBytes+externalReserveBytes,'pool and reserve must fill the declared envelope');
  const snapshots=events.filter(e=>e.event==='h3_cuda_proof_pool_snapshot'&&String(e.jobId??e.job_id)===jobId);
  assert.ok(snapshots.length,'actual worker capped-pool observations required');
  for(const s of snapshots){
    assert.equal(s.logical_ordinal,0);assert.equal(s.cap_bytes,poolBytes);
    const reservedHigh=numeric(s.reserved_high_bytes,'reserved_high_bytes'),usedHigh=numeric(s.used_high_bytes,'used_high_bytes');
    assert.ok(reservedHigh<=poolBytes,'pool reservation exceeded enforced budget');
    assert.ok(usedHigh<=poolBytes,'pool usage exceeded enforced budget');
    assert.ok(usedHigh<=reservedHigh,'pool usage exceeds its reservation high-water');
  }
  assert.ok(snapshots.some(s=>s.used_high_bytes>0),'pool did not witness actual tensors for this job');
  const outside=snapshots.map(s=>{
    const whole=numeric(s.device_used_bytes,'device_used_bytes'),reserved=numeric(s.reserved_bytes,'reserved_bytes');
    assert.ok(reserved<=s.reserved_high_bytes,'current pool reservation exceeds its recorded high-water');
    assert.ok(whole>=reserved,'outside-pool attribution is unavailable/inconsistent, never zero');
    const value=whole-reserved;assert.ok(value<=externalReserveBytes,'observed outside-pool allocations exceeded the declared reserve');return value;
  });
  assert.ok(outside.length,'actual paired whole-device/current-pool observations required for outside-pool attribution');
  assert.ok(metrics.samples?.length>=2,'whole-device/RSS observations required');
  for(const sample of metrics.samples){assert.ok(numeric(sample.host,'sampled host bytes')>0&&numeric(sample.device,'sampled device bytes')>0,'missing observations are not zero');assert.ok(sample.device<=totalEnvelopeBytes,'whole device exceeded declared envelope');}
  // Continuous pool high-water plus the entire explicitly reserved outside-pool allowance.
  const reservedHighBytes=Math.max(...snapshots.map(s=>s.reserved_high_bytes));
  assert.ok(reservedHighBytes+externalReserveBytes<=totalEnvelopeBytes);
  return {reservedHighBytes,externalReserveBytes,observedOutsidePoolPeakBytes:Math.max(...outside),totalEnvelopeBytes,scope:'driver-continuous pool; separately reserved context/library/foreign allocations; sampled outside-pool/whole-device/RSS blind spots retained'};
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

export function validateVedaParityPair(records,{sourceEquivalence}={}){
  const take=id=>{const found=records.filter(r=>r.id===id);assert.equal(found.length,1,`one actual ${id} anchor required`);return found[0];};
  const resident=take('workstation-veda'),streamed=take('24gib-veda-sparse');
  assert.notEqual(resident.job?.id,streamed.job?.id,'independent completed generations required');
  if(sourceEquivalence){
    assert.equal(sourceEquivalence[RETAINED_EQUIVALENCE],true,'source equivalence must be verified against actual Git objects');
    assert.equal(resident.job?.id,RETAINED_WORKSTATION.jobId);
    assert.deepEqual(resident.sourcePair,RETAINED_WORKSTATION.sourcePair);
    assert.deepEqual(sourceEquivalence.originalSourcePair,resident.sourcePair);
    assert.deepEqual(sourceEquivalence.targetSourcePair,streamed.sourcePair);
    assert.equal(resident.media?.[0]?.sha256,RETAINED_WORKSTATION.mediaSha256);
  }else assert.deepEqual(resident.sourcePair,streamed.sourcePair,'Veda parity source pair changed');
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

export async function collectDecodedMediaDiagnostics(a,b,dir,{videoElements,audioSamples,audioPadding=0,audioChannels=2,command=exec,sampledDelta=decodedDelta}={}){
  assert.ok(Number.isSafeInteger(videoElements)&&videoElements>0&&videoElements%3===0,'native video element count required');
  assert.ok(Number.isSafeInteger(audioSamples)&&audioSamples>0&&Number.isSafeInteger(audioPadding)&&audioPadding>=0,'native audio sample coverage required');
  const report={streams:{}};const raw=[];
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
      report.streams[stream]=full;
    }
    for(const stream of ['video','audio'])report.streams[stream].sampledRmse=await sampledDelta(a,b,stream);
    for(const [stream,metric] of Object.entries(report.streams))assert.ok(Number.isFinite(metric.sampledRmse)&&metric.sampledRmse>=0,`${stream} existing decodedDelta witness requires finite nonnegative RMS`);
    return report;
  }catch(error){error.comparison=report;throw error;}
  finally{await Promise.all(raw.map(file=>rm(file,{force:true})));}
}

export async function compareDecodedMedia(a,b,dir,options={}){
  let report={policy:VEDA_EQUIVALENCE,streams:{}};
  try{
    // Collect complete native RGB and stereo PCM before applying either legacy ceiling.
    report={...report,...await collectDecodedMediaDiagnostics(a,b,dir,options)};
    for(const stream of ['video','audio'])report.streams[stream].maximumRmse=VEDA_EQUIVALENCE[stream];
    for(const stream of ['video','audio']){
      const full=report.streams[stream];
      assert.ok(full.rmse<=VEDA_EQUIVALENCE[stream]&&full.channelRmse.every(v=>v<=VEDA_EQUIVALENCE[stream]),`${stream} full decoded Veda residency comparison exceeds fixed same-backend RMS ceiling`);
      assert.ok(full.sampledRmse<=VEDA_EQUIVALENCE[stream],`${stream} existing decodedDelta witness exceeds same-backend RMS ceiling`);
    }
    return report;
  }catch(error){error.comparison={...report,...error.comparison};throw error;}
}

export async function compareVedaAnchors(records,output,{compare=compareDecodedMedia,sourceEquivalence}={}){
  let report={kind:'cuda-veda-resident-window-equivalence',passed:false,policy:VEDA_EQUIVALENCE};
  try{
    const {resident,streamed,witnesses}=validateVedaParityPair(records,{sourceEquivalence});
    const files=[resident,streamed].map(record=>path.join(output,record.id,record.media[0].av));
    for(const [i,file] of files.entries())assert.equal(await fileSha256(file),witnesses[i].sha256,'bound parity media changed');
    report={...report,sourcePair:sourceEquivalence?undefined:resident.sourcePair,sourcePairs:{resident:resident.sourcePair,streamed:streamed.sourcePair},sourceEquivalence,anchors:witnesses};
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
export async function stageModels(p,output){
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
export class StockService extends OwnedService {
  async assertOwner(){
    const owner=await json(this.ownerPath);assert.equal(owner.token,this.owner);assert.equal(owner.pid,process.pid);
    const current=await stat(this.binary);assert.equal(current.size,this.binaryStamp.size);assert.equal(current.mtimeMs,this.binaryStamp.mtimeMs,'owned binary changed');
  }
  async initialize(){
    validateCudaRuntimeDevice(this.c.runtime);
    authorized(this.p,this.c);await sourcePair(this.p,this.swRoot,this.inferenceRoot);await hardware('cuda',this.c.runtime.deviceId,this.c.runtime);
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

// The retained case constructs its own predictor, packed layout and Veda runtime. It does not
// enter model admission or transformer materialization. Compare its numerical/layout/getter
// dependencies, including shared backend/build inputs, rather than unrelated H3 pricing facts.
const MATERIALIZING_PROOF_SOURCES=[
  'Cargo.toml','Cargo.lock','rust-toolchain.toml','.cargo/config.toml',
  ...['Cargo.toml','src/lib.rs','src/veda.rs','src/veda_pointer_v1.cu','src/nn.rs','src/denoise','src/dit/mod.rs',
    'src/dit/config.rs','src/dit/positions.rs','src/audio_config.rs','src/reference.rs',
    'src/memory_strategy.rs','src/memory_strategy/execution.rs'].map(file=>`crates/media/candle-gen/candle-gen-minimax-h3/${file}`),
  'crates/llm/test-support/model_proof.rs',
  'crates/media/candle-gen/candle-gen',
  'crates/kernels/candle-quant-kernels',
  'crates/media/candle-gen/vendor/candle-core',
  'crates/media/candle-gen/vendor/candle-kernels',
  'crates/media/candle-gen/vendor/candle-flash-attn',
  ...['Cargo.toml','src/lib.rs','src/error.rs','src/runtime.rs','src/runtime','src/attention_budget.rs',
    'src/veda.rs','src/veda','src/weightsmeta.rs','src/numeric.rs','src/model_input_proof.rs',
    'src/model_input_macros.rs','src/h3_memory.rs','src/h3_reference.rs'].map(file=>`crates/contracts/gen-core/${file}`),
  ...['Cargo.toml','src/lib.rs','src/veda.rs','src/veda','src/h3_audio_config.rs','src/h3_reference.rs',
    'src/h3_schedule.rs','src/h3_memory.rs'].map(file=>`crates/contracts/gen-core-h3/${file}`),
];
// These two files are not in the raw Veda unit case's numerical execution closure:
// Core's vendoring note is documentation; candle-gen's fixture belongs to tests/main.rs,
// not the dependency library linked by the H3 unit-test binary. No other path is exempt.
const MATERIALIZING_NON_NUMERICAL_FILES=Object.freeze({
  'crates/media/candle-gen/vendor/candle-core':'VENDORED.md',
  'crates/media/candle-gen/candle-gen':'tests/cuda_proof_pool.rs',
});
async function materializingTreeMapping(inferenceRoot,reference,target,selected,measuredObject,targetObject,command){
  const excluded=`${selected}/${MATERIALIZING_NON_NUMERICAL_FILES[selected]}`;
  const changed=(await command('git',['diff','--name-only','--no-renames',reference,target,'--',selected],{cwd:inferenceRoot})).stdout.trim().split(/\r?\n/);
  assert.deepEqual(changed,[excluded],`numerical evidence source changed: ${selected}`);
  const entries=async revision=>{
    const output=(await command('git',['ls-tree','-r',revision,'--',selected],{cwd:inferenceRoot})).stdout.trim();
    assert.ok(output,'recursive numerical source tree must exist');
    const rows=output.split(/\r?\n/).map(line=>{
      const match=line.match(/^(\d{6}) blob ([a-f0-9]{40})\t(.+)$/);
      assert.ok(match,'recursive numerical source must be a Git blob');
      assert.ok(match[3].startsWith(`${selected}/`),'recursive numerical source must remain in selected tree');
      return {mode:match[1],object:match[2],path:match[3]};
    });
    assert.equal(new Set(rows.map(row=>row.path)).size,rows.length,'recursive numerical source paths must be unique');
    const omitted=rows.filter(row=>row.path===excluded),remaining=rows.filter(row=>row.path!==excluded);
    assert.equal(omitted.length,1,'exact non-numerical source must exist');assert.ok(remaining.length,'numerical source tree cannot be empty');
    assert.equal(omitted[0].mode,'100644','non-numerical mapping permits content changes only');
    return {omitted:omitted[0],remaining};
  };
  const before=await entries(reference),after=await entries(target);
  assert.deepEqual(after.remaining,before.remaining,`numerical evidence recursive source changed: ${selected}`);
  assert.notEqual(before.omitted.object,after.omitted.object,'non-numerical mapping must describe actual content movement');
  return {path:selected,object:measuredObject,targetObject,mapping:{
    kind:'exact-non-numerical-file-only',changedPaths:changed,
    measuredFile:before.omitted,targetFile:after.omitted,unchangedRecursiveBlobs:before.remaining,
  }};
}
export async function materializingProofSources(inferenceRoot,reference,target,command=exec){
  const sources=[];
  for(const selected of MATERIALIZING_PROOF_SOURCES){
    const blob=async revision=>{
      const object=(await command('git',['rev-parse','--verify',`${revision}:${selected}`],{cwd:inferenceRoot})).stdout.trim();
      assert.match(object,/^[a-f0-9]{40}$/,'numerical source object must exist');return object;
    };
    const expected=await blob(reference),actual=await blob(target);
    if(expected!==actual&&Object.hasOwn(MATERIALIZING_NON_NUMERICAL_FILES,selected)){
      sources.push(await materializingTreeMapping(inferenceRoot,reference,target,selected,expected,actual,command));
    }else{
      assert.equal(expected,actual,`numerical evidence source changed: ${selected}`);sources.push({path:selected,object:actual});
    }
  }
  return sources;
}

// A fixed source comparison for this one repair, not an interpreter or caller-supplied exemption.
// Existing Rust slicing masks literal/comment braces and removes only authored cfg(test) modules.
const residentProduction=source=>withoutRustTestModules(source).replace(/^[ \t]+/gm,'');
const residentLines=source=>source.split(/\r?\n/).map(line=>line.trim()).filter(line=>line&&!line.startsWith('//')).join('\n');
const residentTokens=source=>residentLines(source).replace(/\s/g,'');
function residentReplace(source,needle,replacement){
  assert.equal(source.split(needle).length,2,'one exact reviewed source expression required');return source.replace(needle,replacement);
}
function residentBraceEnd(source,opening){
  const code=rustCodeMask(source);assert.equal(code[opening],'{');let depth=1,end=opening+1;
  while(end<code.length&&depth){if(code[end]==='{')depth++;if(code[end]==='}')depth--;end++;}
  assert.equal(depth,0,'reviewed source block must close');return end;
}
function residentFunction(source,name){
  const found=rustFunctionSlices(source).filter(fn=>fn.name===name);assert.equal(found.length,1,`one reviewed ${name} required`);
  const fn=found[0],opening=rustCodeMask(fn.source).indexOf('{');return {...fn,signature:fn.source.slice(0,opening),body:fn.source.slice(opening+1,-1)};
}
export function validateBoundedQkResidentSources(before,after){
  const old=before.map(residentProduction),next=after.map(residentProduction);assert.equal(old.length,4);assert.equal(next.length,4);
  let layers=next[0];
  for(const name of ['precondition_projected_qk','map_qk_token_chunks']){
    const fn=residentFunction(layers,name);layers=layers.slice(0,fn.start)+layers.slice(fn.end);
  }
  const report='#[derive(Clone, Copy, Debug, Default)]\npub(super) struct QkPreconditionReport {';
  const reportAt=layers.indexOf(report);assert.ok(reportAt>=0,'exact non-Drop Q/K report required');
  const reportEnd=residentBraceEnd(layers,reportAt+report.length-1);
  assert.equal(residentTokens(layers.slice(reportAt,reportEnd)),residentTokens(report+'pub norm_chunks: usize, pub rotary_chunks: usize, pub maximum_chunk_elements: usize,}'));
  layers=layers.slice(0,reportAt)+layers.slice(reportEnd);
  const marker='let (q, k, v) =';assert.equal(old[0].split(marker).length,2);assert.equal(layers.split(marker).length,2);
  const originalAt=old[0].indexOf(marker),originalOpen=old[0].indexOf('{',originalAt),originalEnd=residentBraceEnd(old[0],originalOpen);
  const at=layers.indexOf(marker),opening=layers.indexOf('{',at),end=residentBraceEnd(layers,opening);
  assert.equal(residentTokens(layers.slice(at+marker.length,opening)),'ifrope.is_some()&&x.device().is_cuda()&&crate::block_stream::current().is_some()','bounded helper must exclude Resident and refiner');
  const elseOpen=layers.indexOf('{',end);assert.equal(layers.slice(end,elseOpen).trim(),'else');const elseEnd=residentBraceEnd(layers,elseOpen);
  assert.equal(residentLines(layers.slice(elseOpen,elseEnd)),residentLines(old[0].slice(originalOpen,originalEnd)),'Resident/refiner QKV math changed');
  layers=layers.slice(0,at)+old[0].slice(originalAt,originalEnd)+layers.slice(elseEnd);
  assert.equal(residentLines(layers),residentLines(old[0]),'code outside the guarded Q/K repair changed');

  const release='if self.stream_window.is_some() && self.device.is_cuda() {';
  assert.equal(next[1].split(release).length,2,'one explicit streamed CUDA projection release required');
  const releaseAt=next[1].indexOf(release),releaseEnd=residentBraceEnd(next[1],releaseAt+release.length-1);
  assert.equal(residentTokens(next[1].slice(releaseAt+release.length,releaseEnd-1)),'drop(video);drop(audio);');
  assert.equal(residentLines(next[1].slice(0,releaseAt)+next[1].slice(releaseEnd)),residentLines(old[1]),'Resident model/lifetimes changed outside the streamed release');

  const originalFns=rustFunctionSlices(old[2]),targetFns=rustFunctionSlices(next[2]);
  const added=['qk_precondition_token_block','windowed_qk_precondition_peak','windowed_operation_peak_with_adapter','operation_peak_for','bounded_cuda_denoise_with_veda_bytes','compose_veda_peak','bounded_cuda_denoise_peak_bytes','denoise_peak_bytes_impl'];
  assert.deepEqual(targetFns.filter(fn=>!originalFns.some(oldFn=>oldFn.name===fn.name)).map(fn=>fn.name).sort(),[...added].sort(),'unexpected allocation helper');
  const body=(source,name)=>residentTokens(residentFunction(source,name).body);
  const signature=(source,name)=>residentTokens(residentFunction(source,name).signature);
  for(const name of ['operation_peak_with_adapter','denoise_peak_bytes','denoise_with_veda_bytes'])assert.equal(signature(next[2],name),signature(old[2],name),`legacy ${name} parameter binding changed`);
  assert.equal(body(next[2],'operation_peak_with_adapter'),'self.operation_peak_for(rows,adapter_work,false)','legacy working bound must select the unchanged arm');
  assert.equal(signature(next[2],'operation_peak_for'),residentReplace(signature(old[2],'operation_peak_with_adapter'),'adapter_work:u64)','adapter_work:u64,windowed:bool)').replace('operation_peak_with_adapter','operation_peak_for'),'delegated operation parameter binding changed');
  let operation=body(next[2],'operation_peak_for');
  operation=residentReplace(operation,'letprecondition=ifwindowed{self.windowed_qk_precondition_peak(rows,inner)?}else{self.qk_norm_peak(rows,inner)?.max(self.rotary_peak(rows,inner)?)};','');
  operation=residentReplace(operation,'precondition.max(materialized)','self.qk_norm_peak(rows,inner)?.max(self.rotary_peak(rows,inner)?).max(materialized)');
  operation=residentReplace(operation,'times(ifwindowed{5}else{4},hidden)?','times(4,hidden)?');
  operation=residentReplace(operation,'letblock_norm=ifwindowed{plus(plus(times(9,hidden)?,times(8,rows)?)?,times(4,self.hidden)?,)?}else{times(8,hidden)?};','');
  operation=residentReplace(operation,'.max(block_norm)','.max(times(8,hidden)?)');
  assert.equal(operation,body(old[2],'operation_peak_with_adapter'),'legacy operation arithmetic changed');
  assert.equal(body(next[2],'denoise_peak_bytes'),'self.denoise_peak_bytes_impl(task,backend,rows,window,false)','legacy denoise must select the unchanged arm');
  let denoise=body(old[2],'denoise_peak_bytes');
  denoise=residentReplace(denoise,'.operation_peak_with_adapter(','.operation_peak_for(');
  denoise=residentReplace(denoise,'RowDomain::Packed)}),)?','RowDomain::Packed)}),bounded_qk,)?');
  denoise=residentReplace(denoise,'bounded_qk,)?,);','bounded_qk,)?);');
  assert.equal(body(next[2],'denoise_peak_bytes_impl'),denoise,'legacy denoise composition changed');
  const composeCall='self.compose_veda_peak(task,backend,rows,window,veda,facts,dense)';
  assert.equal(residentReplace(body(next[2],'denoise_with_veda_bytes'),composeCall,body(next[2],'compose_veda_peak')),body(old[2],'denoise_with_veda_bytes'),'legacy Veda composition changed');
  for(const fn of originalFns){
    if(!['operation_peak_with_adapter','denoise_peak_bytes','denoise_with_veda_bytes'].includes(fn.name))assert.equal(residentTokens(residentFunction(next[2],fn.name).source),residentTokens(fn.source),`legacy ${fn.name} changed`);
  }
  const globals=(source,functions)=>{for(const fn of [...functions].reverse()){
    const testOnly='#[cfg(test)]\n';let start=fn.start;
    if(fn.name==='windowed_operation_peak_with_adapter'&&source.slice(0,start).endsWith(testOnly))start-=testOnly.length;
    source=source.slice(0,start)+source.slice(fn.end);
  }return residentTokens(source);};
  let targetGlobals=globals(next[2],targetFns);
  targetGlobals=residentReplace(targetGlobals,'pubconstCUDA_QK_PRECONDITION_ELEMENTS:u64=crate::veda::workspace::CUDA_DENSE_SCORE_ELEMENTS;','');
  targetGlobals=residentReplace(targetGlobals,'#[allow(clippy::too_many_arguments)]','');
  assert.equal(targetGlobals,globals(old[2],originalFns),'allocation types/constants outside bounded helpers changed');

  // These exact substitutions leave the None/full/Resident calls and all other pricing code
  // byte-token-equivalent; selected Some(window) CUDA is the only new caller.
  let pricing=residentTokens(next[3]);
  const args='task,self.backend,Self::task_rows(task,rows),windowasusize,';
  pricing=residentReplace(pricing,`ifmatches!(self.backend,NativeBackend::Candle{..}){profile.bounded_cuda_denoise_peak_bytes(${args})?}else{profile.denoise_peak_bytes(${args})?}`,`profile.denoise_peak_bytes(${args})?`);
  pricing=residentReplace(pricing,'letselected_window=window;','');
  const selected='let(dense,with_veda)=ifselected_window.is_some(){(profile.bounded_cuda_denoise_peak_bytes(task,self.backend,rows,window,)?,profile.bounded_cuda_denoise_with_veda_bytes(task,self.backend,rows,window,veda,)?,)}else{(profile.denoise_peak_bytes(task,self.backend,rows,window)?,profile.denoise_with_veda_bytes(task,self.backend,rows,window,veda,)?,)};letdense=self.with_adapter_denoise(task,rows,dense);Ok(peak.max(dense.max(with_veda)))';
  const legacy='letdense=self.with_adapter_denoise(task,rows,profile.denoise_peak_bytes(task,self.backend,rows,window)?,);Ok(peak.max(dense.max(profile.denoise_with_veda_bytes(task,self.backend,rows,window,veda,)?)))';
  pricing=residentReplace(pricing,selected,legacy);
  assert.equal(pricing,residentTokens(old[3]),'Resident/None/MLX pricing or selected-window guard changed');
  return {scope:'mechanically unchanged Resident/refiner QKV and model; unchanged legacy/None pricing; bounded helpers only on selected CUDA windows',paths:[...RETAINED_BOUNDED_QK_PATHS]};
}

export function validateRetainedWindowProvenance({beforeTsv,afterTsv,beforeAudit,afterAudit,engineFrom,engineTo,beforeBlob,afterBlob}){
  const windowPath='crates/contracts/gen-core/src/h3_memory/window.rs';
  sha40(engineFrom);sha40(engineTo);
  sha40(beforeBlob);sha40(afterBlob);assert.notEqual(beforeBlob,afterBlob,'window provenance must describe an actual blob change');
  const before=parseProvenanceCandidates(beforeTsv),after=parseProvenanceCandidates(afterTsv);
  const matches=before.filter(row=>row.path===windowPath);assert.equal(matches.length,1,'one original window provenance row required');
  const row=matches[0];assert.equal(row.blob,beforeBlob,'original provenance blob must match original engine');
  assert.equal(row.markerSha256,'66a666f0a533b9e30fc274055f28f6237445ef4d4978e4c30b948ea5ce4d6620');
  assert.equal(row.area,'architecture:crates/contracts/gen-core');
  const oldLine=[windowPath,beforeBlob,row.markerSha256,row.area].join('\t');
  const newLine=[windowPath,afterBlob,row.markerSha256,row.area].join('\t');
  assert.equal(afterTsv,residentReplace(beforeTsv,oldLine,newLine),'only the reviewed window blob provenance cell may change');
  // Same payload projection as check-license-coverage's auditCanonicalDigest. Hash bytes
  // belong to the existing owner; importing/running that CLI would perform unrelated checks.
  const digest=record=>sha256(JSON.stringify(Object.fromEntries(['artifacts','prospectiveDisclosures','provenanceScan','portedSourceAreas','crateCoverage','crateDispositions','includeSites'].map(key=>[key,record[key]]))));
  assert.equal(beforeAudit.inferenceRevision,engineFrom);assert.equal(beforeAudit.provenanceScan.revision,engineFrom);assert.equal(beforeAudit.crateCoverage.revision,engineFrom);
  assert.equal(beforeAudit.provenanceScan.matchedFiles,before.length);
  assert.equal(beforeAudit.provenanceScan.populationSha256,provenancePopulationSha256(before),'original population must be derived from its exact table');
  assert.equal(beforeAudit.auditDigest,digest(beforeAudit),'original audit digest must be derived from its payload');
  const expected=structuredClone(beforeAudit);expected.inferenceRevision=engineTo;expected.provenanceScan.revision=engineTo;expected.crateCoverage.revision=engineTo;
  expected.provenanceScan.populationSha256=provenancePopulationSha256(after);expected.auditDigest=digest(expected);
  assert.deepEqual(afterAudit,expected,'provenance JSON may only restamp the pin and derived population/audit digests');
  return {path:'config/inference-provenance-candidates.tsv',windowPath,beforeBlob,afterBlob,markerSha256:row.markerSha256,classification:row.area,matchedFiles:after.length,populationSha256:expected.provenanceScan.populationSha256,auditDigest:expected.auditDigest};
}

export async function retainedWorkstationSources(swRoot,inferenceRoot,target,audit,command=exec){
  const originalSourcePair=RETAINED_WORKSTATION.sourcePair;
  sha40(target.sceneworks);sha40(target.inference);
  assert.equal(audit.kind,'sw-f329-uncapped-workstation-source-equivalence-v1');
  assert.deepEqual(audit.originalSourcePair,originalSourcePair);assert.deepEqual(audit.targetSourcePair,target);
  assert.equal(audit.uncappedAllocatorPreserved,true,'reviewed uncapped allocation equivalence required');
  assert.match(audit.reviewUrl,/^https:\/\/github\.com\/SceneWorks\/inference\/pull\/\d+$/);
  const git=async(root,args)=>(await command('git',args,{cwd:root})).stdout.trim();
  const object=async(root,revision,file)=>{
    const row=await git(root,['ls-tree',revision,'--',file]);
    if(!row)return null;const match=row.match(/^\d+ (?:blob|tree) ([a-f0-9]{40})\t/);assert.ok(match,'invalid Git object receipt');return match[1];
  };
  const audited=async(root,from,to,entries,allowed)=>{
    assert.ok(Array.isArray(entries));const paths=new Set();
    for(const entry of entries){
      assert.ok(allowed.includes(entry.path),'retention cannot exempt an arbitrary source path');assert.ok(!paths.has(entry.path));paths.add(entry.path);
      if(entry.before!==null)sha40(entry.before);sha40(entry.after);
      assert.equal(await object(root,from,entry.path),entry.before,'reviewed original source object changed');
      assert.equal(await object(root,to,entry.path),entry.after,'reviewed target source object changed');
    }
    return paths;
  };
  const changed=async(root,from,to)=>(await git(root,['diff','--name-only','--no-renames',from,to])).split('\n').filter(Boolean);
  const engineFrom=originalSourcePair.inference,engineTo=target.inference;
  await git(inferenceRoot,['merge-base','--is-ancestor',engineFrom,engineTo]);
  const enginePaths=await audited(inferenceRoot,engineFrom,engineTo,audit.engineChanges,RETAINED_ENGINE_PATHS);
  assert.deepEqual((await changed(inferenceRoot,engineFrom,engineTo)).sort(),[...enginePaths].sort(),'engine changes extend beyond reviewed allocator/streamed-QK sources');
  let boundedQk;
  if(RETAINED_BOUNDED_QK_PATHS.some(file=>enginePaths.has(file))){
    assert.ok(RETAINED_BOUNDED_QK_PATHS.every(file=>enginePaths.has(file)),'complete reviewed bounded Q/K repair required');
    for(const field of ['boundedQkResidentPreserved','boundedQkRefinerPreserved','boundedQkResidentPricingPreserved'])assert.equal(audit[field],true,`explicit reviewed ${field} required`);
    const source=async revision=>Promise.all(RETAINED_BOUNDED_QK_PATHS.map(file=>git(inferenceRoot,['show',`${revision}:${file}`])));
    boundedQk=validateBoundedQkResidentSources(await source(engineFrom),await source(engineTo));
  }
  const swFrom=originalSourcePair.sceneworks,swTo=target.sceneworks;
  await git(swRoot,['merge-base','--is-ancestor',swFrom,swTo]);
  const metadataPaths=await audited(swRoot,swFrom,swTo,audit.consumerPinMetadata,RETAINED_PIN_METADATA);
  if(boundedQk)for(const file of ['config/inference-provenance-candidates.tsv','config/inference-third-party-source.json'])assert.ok(metadataPaths.has(file),'bounded Q/K repair requires both exact generated provenance object bindings');
  const ciPaths=await audited(swRoot,swFrom,swTo,audit.consumerCiChanges??[],RETAINED_CI_PATHS);
  const consumerChanges=[];let generatedProvenance;
  for(const file of await changed(swRoot,swFrom,swTo)){
    if(file==='Cargo.lock'||file==='Cargo.toml'||/^crates\/[^/]+\/Cargo\.toml$/.test(file)){
      const before=await git(swRoot,['show',`${swFrom}:${file}`]),after=await git(swRoot,['show',`${swTo}:${file}`]);
      assert.equal(before.replaceAll(engineFrom,engineTo),after,'retained control cannot cross a dependency/configuration change');
    }else if(file==='config/inference-provenance-candidates.tsv'){
      assert.ok(boundedQk&&metadataPaths.has(file),'provenance movement requires the reviewed bounded Q/K repair and exact object audit');
      const records=await Promise.all(['config/inference-provenance-candidates.tsv','config/inference-third-party-source.json'].flatMap(selected=>[swFrom,swTo].map(revision=>git(swRoot,['show',`${revision}:${selected}`]))));
      const windowPath=RETAINED_BOUNDED_QK_PATHS[2];
      generatedProvenance=validateRetainedWindowProvenance({beforeTsv:records[0],afterTsv:records[1],beforeAudit:JSON.parse(records[2]),afterAudit:JSON.parse(records[3]),engineFrom,engineTo,beforeBlob:await object(inferenceRoot,engineFrom,windowPath),afterBlob:await object(inferenceRoot,engineTo,windowPath)});
    }else if(file==='.github/workflows/h3-cuda-memory-terminal.yml'){
      const before=await git(swRoot,['show',`${swFrom}:${file}`]),after=await git(swRoot,['show',`${swTo}:${file}`]);
      const shallow='path: sceneworks\n          fetch-depth: 1';assert.equal(before.split(shallow).length,2);
      const receipts='            ${{ env.SW_F329_OUTPUT }}/workstation-veda/*.json';
      const originals='            ${{ env.SW_F329_OUTPUT }}/workstation-veda/original/**/*.json';
      assert.equal(before.split(receipts+'\n').length,2,'one original workstation artifact anchor required');
      assert.ok(!before.includes(originals),'original workflow must not already contain the retained receipt pattern');
      const expected=before.replace(shallow,'path: sceneworks\n          fetch-depth: 128').replace(receipts+'\n',`${receipts}\n${originals}\n`);
      assert.equal(expected,after,'manual source/configuration changed beyond reference history checkout and exact retained receipt artifact path');
    }else if(ciPaths.has(file)){
      if(file==='.github/workflows/server-candle-linux.yml'){
        const outside=source=>{
          const matches=[...source.matchAll(/^  build-candle:\n/gm)];assert.equal(matches.length,1,'one standard Linux build job required');
          const start=matches[0].index,end=source.slice(start+matches[0][0].length).search(/^  [A-Za-z0-9_-]+:\n/m);
          return source.slice(0,start)+(end<0?'':source.slice(start+matches[0][0].length+end));
        };
        assert.equal(outside(await git(swRoot,['show',`${swFrom}:${file}`])),outside(await git(swRoot,['show',`${swTo}:${file}`])),'CI retention exception extends beyond standard Linux build-candle');
      }
    }else assert.ok(metadataPaths.has(file)||RETAINED_CONTROLLER_PATHS.includes(file),`retained control application source changed: ${file}`);
    consumerChanges.push({path:file,before:await object(swRoot,swFrom,file),after:await object(swRoot,swTo,file)});
  }
  assert.ok([...metadataPaths].every(file=>consumerChanges.some(row=>row.path===file)),'unused metadata exemption');
  assert.ok([...ciPaths].every(file=>consumerChanges.some(row=>row.path===file)),'unused CI exemption');
  return {kind:audit.kind,originalSourcePair,targetSourcePair:target,uncappedAllocatorPreserved:true,engineChanges:audit.engineChanges,consumerChanges,
    reviewUrl:audit.reviewUrl,boundedQk,generatedProvenance,scope:'only this original uncapped control; reviewed capped allocator and mechanically guarded streamed-only Q/K repair, exact remaining application/operator/kernel/build sources',
    [RETAINED_EQUIVALENCE]:true};
}

export function validateRetainedWorkstationRecord(record,request){
  assert.equal(record.id,'workstation-veda');assert.equal(record.passed,true);assert.equal(record.job?.id,RETAINED_WORKSTATION.jobId);
  assert.deepEqual(record.sourcePair,RETAINED_WORKSTATION.sourcePair);assert.deepEqual(record.settings,request);
  assert.equal(record.job.status,'completed');assert.equal(record.job.requestedGpu,'1');assert.equal(record.job.assignedGpu,'1');
  for(const field of ['error','shutdownError','observationError','binaryError'])assert.equal(record[field],undefined,'failed/partial control cannot be retained');
  assertQueuedMatchesSubmitted({...request,projectId:record.job.payload?.projectId},record.job);
  const metrics=record.generationMetrics;assert.equal(metrics.backend,'cuda');assert.equal(metrics.steps,8);assert.equal(metrics.seed,42);
  assert.equal(metrics.width,1344);assert.equal(metrics.height,768);assert.equal(metrics.quantBits,4);
  assert.equal(record.media?.length,1);const media=record.media[0];assert.equal(media.jobId,record.job.id);
  assert.equal(media.sha256,RETAINED_WORKSTATION.mediaSha256);assert.equal(media.decoded?.sha256,media.sha256);
  assert.equal(media.av,path.basename(media.av));
  validateAVAgainstRequest(media.decoded,request,{model:'minimax_h3',fps:24,width:1344,height:768,audio:{required:true,sampleRate:32000,channels:2,codec:'aac'}});
  assert.equal(media.asset?.recipe?.rawAdapterSettings?.effectiveSteps,8);assert.equal(media.asset.recipe.rawAdapterSettings.minimaxH3Tier,'q4');
  assert.equal(record.content?.frames,345);assert.ok(record.content.spatialStd?.min>=record.content.spatialStd?.floor&&record.content.spatialStd.floor>=0.07);
  assert.ok(record.observations?.hostPeakBytes>0&&record.observations.devicePeakBytes>0&&record.observations.devicePeakBytes<=95*GIB);
  assert.ok(record.events?.every(e=>!(e.cap_bytes>0)),'only the uncapped control can be retained');
  const veda=metrics.vedaExecution??metrics.veda_execution;assert.equal(veda?.requested,true);assert.ok(veda.sparseCalls>0);
  assert.ok(!veda.fallbackReason);assert.equal(veda.kernel,'cuda-veda-sage-int8-tiled-v2');
  const selections=record.events.filter(e=>e.event==='video_memory_strategy_selected'&&e.route==='minimax_h3');
  assert.ok(selections.length&&selections.every(e=>e.request_frames===345&&e.strategy==='Resident'));
  assert.equal(record.events.filter(e=>e.event==='h3_transformer_window_materialized').length,0);
  const executions=record.events.filter(e=>e.event==='h3_attention_execution');assert.equal(executions.length,1);
  assert.equal(executions[0].transformer_window,'None');assert.equal(executions[0].transformer_component,'None');
  return record;
}

export function retainedWorkstationWeights(reference,current,sources){
  const normalize=receipt=>{
    assert.equal(receipt.kind,'sw-f329-staged-models');assert.ok(receipt.files?.length>0);
    assert.deepEqual([...new Set(receipt.files.map(file=>file.role))].sort(),Object.keys(SOURCES).sort(),'all original source roles required');
    assert.equal(new Set(receipt.files.map(file=>`${file.role}/${file.relative}`)).size,receipt.files.length,'duplicate retained source file');
    return receipt.files.map(file=>{
      const spec=SOURCES[file.role];assert.ok(spec,'unknown retained weight role');assert.equal(file.repo,spec.repo);assert.equal(file.revision,sources[file.role].revision);
      sha64(file.sha256);bytes(file.bytes,'retained source bytes');assert.ok(typeof file.relative==='string'&&!file.relative.includes('..'));
      return {role:file.role,repo:file.repo,revision:file.revision,relative:file.relative,bytes:file.bytes,sha256:file.sha256};
    }).sort((a,b)=>`${a.role}/${a.relative}`.localeCompare(`${b.role}/${b.relative}`));
  };
  assert.deepEqual(reference.sourcePair,RETAINED_WORKSTATION.sourcePair);
  assert.deepEqual(normalize(reference),normalize(current),'actual staged source weight commitments differ');
}

export function retainedWorkstationHardware(reference,current){
  const selected=hardware=>{
    const cuda=hardware.cuda,device=cuda?.selectedPhysicalDevice;assert.equal(device?.nvmlIndex,1);
    assert.equal(device.uuid,'GPU-e4b79931-7be6-f216-460a-f5405cfafffe');assert.equal(cuda.cudaVisibleDevices,device.uuid);assert.equal(cuda.cudaDeviceOrder,'PCI_BUS_ID');
    return {device,driver:hardware.driver};
  };
  assert.deepEqual(selected(reference),selected(current),'retained control hardware/driver changed');
}

export function retainedWorkstationBuild(reference,current,target){
  assert.equal(reference.sha256,RETAINED_WORKSTATION.binarySha256);sha64(current.sha256);
  if(target){assert.equal(current.sceneworks,target.sceneworks);assert.equal(current.inference,target.inference);}
  const configuration=build=>{
    assert.equal(build.backend,'cuda');assert.equal(build.configuration?.proof_kind,'campaign_native');
    assert.deepEqual(build.configuration.command,build.command,'native build receipt commands disagree');
    const command=[...build.command],index=command.indexOf('--manifest-path');assert.ok(index>=0);command[index+1]='<owned-manifest>';
    assert.ok(command.includes('release')&&command.includes('sceneworks-rust-api/h3-memory-proof'));
    return {command,flags:build.configuration.flags};
  };
  assert.deepEqual(configuration(reference),configuration(current),'retained control native build configuration changed');
}

async function retainWorkstation(p,swRoot,inferenceRoot,output,request){
  const spec=p.retainedWorkstation,root=spec.campaignDirectory,original={};
  for(const [file,digest] of Object.entries(RETAINED_WORKSTATION.files)){
    const input=path.join(root,file);assert.equal(await fileSha256(input),digest,`accepted control receipt changed: ${file}`);original[file]=await json(input);
  }
  const record=validateRetainedWorkstationRecord(original['workstation-veda/result.json'],request);
  assert.ok(record.observations.hostPeakBytes<=p.hostBudgetBytes,'retained control exceeded the declared host observation budget');
  assert.equal(original['request.json'].originalSeed,null);assert.equal(original['request.json'].chosenSeed,42);
  assert.deepEqual(original['request.json'].request,request);assert.deepEqual(anchorRequest(original['request.json'].original,42),request);
  for(const exit of ['workerExit','apiExit']){assert.equal(original['workstation-veda/exits-1.json'][exit]?.code,0);assert.equal(original['workstation-veda/exits-1.json'][exit]?.signal,null);}
  const build=original['workstation-veda/service-build.json'];assert.equal(build.sha256,RETAINED_WORKSTATION.binarySha256);
  assert.equal(build.sceneworks,record.sourcePair.sceneworks);assert.equal(build.inference,record.sourcePair.inference);
  assert.equal(await fileSha256(spec.binaryPath??build.binary),build.sha256,'original accepted executable bytes changed or are unavailable');
  assert.equal(await fileSha256(spec.sourceAuditPath),spec.sourceAuditSha256,'source equivalence review changed');
  const sourceEquivalence=await retainedWorkstationSources(swRoot,inferenceRoot,{sceneworks:p.sceneworks,inference:p.inference},await json(spec.sourceAuditPath));
  const dir=path.join(output,'workstation-veda');await mkdir(dir);
  for(const file of Object.keys(RETAINED_WORKSTATION.files)){
    const destination=path.join(dir,'original',file);await mkdir(path.dirname(destination),{recursive:true});await copyFile(path.join(root,file),destination);
    assert.equal(await fileSha256(destination),RETAINED_WORKSTATION.files[file]);
  }
  await copyFile(spec.sourceAuditPath,path.join(dir,'source-equivalence-review.json'));
  assert.equal(await fileSha256(path.join(dir,'source-equivalence-review.json')),spec.sourceAuditSha256);
  const media=record.media[0],source=path.join(root,'workstation-veda',media.av),target=path.join(dir,media.av);
  assert.equal(await fileSha256(source),media.sha256,'accepted reference media changed');
  try{await link(source,target);}catch(error){if(error.code!=='EXDEV')throw error;await copyFile(source,target);}
  assert.equal(await fileSha256(target),media.sha256,'retained reference media copy changed');
  const decoded=await decodeAV(target);assert.equal(decoded.sha256,media.sha256);
  validateAVAgainstRequest(decoded,request,{model:'minimax_h3',fps:24,width:1344,height:768,audio:{required:true,sampleRate:32000,channels:2,codec:'aac'}});
  const content=await verifyRenderedContent(target);assert.equal(content.frames,345);
  // Preserve the original measured result/build bytes and source pair. Revalidation has its own receipt.
  await copyFile(path.join(root,'workstation-veda/result.json'),path.join(dir,'result.json'));
  await copyFile(path.join(root,'workstation-veda/service-build.json'),path.join(dir,'service-build.json'));
  const retention={kind:'retained-original-uncapped-workstation',runUrl:spec.runUrl,jobId:record.job.id,originalSourcePair:record.sourcePair,
    originalReceipts:RETAINED_WORKSTATION.files,binarySha256:build.sha256,mediaSha256:media.sha256,sourceAuditSha256:spec.sourceAuditSha256,sourceAuditFile:'source-equivalence-review.json',sourceEquivalence,
    revalidated:{decoded,content},meaning:'original95GiB execution retained; no new workstation measurement or current-source relabel'};
  await save(path.join(dir,'retention.json'),retention);
  return {record,build,hardware:original['hardware.json'],weights:original['staged-models.json'],sourceEquivalence,retention};
}

export async function primitiveProofSource(inferenceRoot,measured,target,command=exec){
  sha40(measured);sha40(target);
  await command('git',['merge-base','--is-ancestor',measured,target],{cwd:inferenceRoot});
  const tree=async revision=>{
    const object=(await command('git',['rev-parse','--verify',`${revision}^{tree}`],{cwd:inferenceRoot})).stdout.trim();
    sha40(object);return object;
  };
  const measuredTree=await tree(measured),targetTree=await tree(target);
  assert.equal(measuredTree,targetTree,'primitive proof requires identical complete source trees');
  return {measuredRevision:measured,targetRevision:target,tree:measuredTree,scope:'original execution; ancestor with identical complete source tree'};
}

export function validatePrimitiveProofLog(log){
  assert.match(log,/test result: ok\. 1 passed; 0 failed; 0 ignored/);
  const marker=log.match(/CUDA_PROOF_POOL_EVIDENCE (\{[^\n]+\})/);assert.ok(marker,'positive allocator enforcement evidence required');
  const primitive=JSON.parse(marker[1]);
  for(const field of ['overCapRefused','bothConstructors','cublasStreamResetRebound'])assert.equal(primitive[field],true,`allocator primitive must prove ${field}`);
  assert.equal(primitive.logicalOrdinal,0,'allocator primitive must use the bound logical device');
  for(const field of ['capBytes','heldBytes','usedHighBytes','reservedHighBytes'])assert.ok(Number.isSafeInteger(primitive[field])&&primitive[field]>0,`allocator primitive must report exact positive ${field}`);
  assert.equal(primitive.capBytes,512*1024**2,'allocator primitive must use the dedicated512MiB cap');
  assert.ok(primitive.capBytes/2<=primitive.heldBytes&&primitive.heldBytes<=primitive.usedHighBytes&&primitive.usedHighBytes<=primitive.reservedHighBytes,'allocator primitive must positively exercise the bounded pool');
  assert.ok(Number.isSafeInteger(primitive.successfulRecoveryCount)&&primitive.successfulRecoveryCount>=0,'allocator primitive must report an exact nonnegative recovery count');
  assert.equal(typeof primitive.fragmentationRecoveryExercised,'boolean','allocator primitive must report whether recovery occurred');
  assert.equal(primitive.fragmentationRecoveryExercised,primitive.successfulRecoveryCount>0,'allocator recovery observation must match its count');
  assert.ok(primitive.reservedHighBytes<=primitive.capBytes);return primitive;
}

async function retainedProofs(p,inferenceRoot){
  const measured=p.primitiveProof.revision??p.inference;
  assert.equal((await readFile(path.join(p.primitiveProof.directory,'source.txt'),'utf8')).trim(),measured,'primitive allocator proof must name its original measured source');
  const primitiveSource=await primitiveProofSource(inferenceRoot,measured,p.inference);
  const log=await readFile(path.join(p.primitiveProof.directory,'allocator-proof.log'),'utf8');
  const primitive=validatePrimitiveProofLog(log);
  const numericLog=await readFile(p.materializingProof.logPath,'utf8'),materializing=validateMaterializingProofLog(numericLog);
  const sources=await materializingProofSources(inferenceRoot,p.materializingProof.revision,p.inference);
  return {primitive:{...primitive,source:primitiveSource,runUrl:p.primitiveProof.runUrl},materializing:{...materializing,revision:p.materializingProof.revision,sources,runUrl:p.materializingProof.runUrl,logSha256:await fileSha256(p.materializingProof.logPath)}};
}

export async function run(p,swRoot,inferenceRoot,output){
  validatePlan(p);assert.equal(process.platform,'win32');assert.equal(process.env.GITHUB_ACTIONS,'true');assert.equal(process.env.GITHUB_EVENT_NAME,'workflow_dispatch');assert.equal(process.env.RUNNER_NAME,p.expectedRunner);assert.equal(process.env.GITHUB_SHA,p.sceneworks);
  const selected=campaignCudaEnvironment(p,process.env);
  assert.ok(path.resolve(output).toLowerCase().startsWith('d:\\codex\\work\\'),'all generated output belongs onD');
  await mkdir(output);await sourcePair({...p,output},swRoot,inferenceRoot);assert.equal(await fileSha256(p.requestPath),p.requestSha256,'original request changed');
  const original=await json(p.requestPath),request=anchorRequest(original,p.seed,selected.physicalGpuIndex);await save(path.join(output,'request.json'),{original,chosenSeed:p.seed,originalSeed:original.seed,request});
  await save(path.join(output,'device-selection.json'),{...selected,logicalCudaOrdinal:0,focusedProofScope:'Existing B9 GPU1 component/source proofs keep their original run/device identities; they are not new product-device measurements. All product anchors use the selected physical device.'});
  const proofs=await retainedProofs(p,inferenceRoot);await save(path.join(output,'focused-proofs.json'),proofs);
  const retained=p.retainedWorkstation?await retainWorkstation(p,swRoot,inferenceRoot,output,request):undefined;
  const inventory=await hardware('cuda',selected.physicalGpuUuid,selected);await save(path.join(output,'hardware.json'),inventory);
  if(retained)retainedWorkstationHardware(retained.hardware,inventory);
  const staged=await stageModels(p,output),results=[];
  if(retained)retainedWorkstationWeights(retained.weights,staged,p.sources);
  const anchors=[['workstation-veda',false,false],['24gib-veda-dense',true,true],['24gib-veda-sparse',true,false]];
  for(const [id,capped,forceDense] of anchors){
    if(id==='workstation-veda'&&retained){results.push(retained.record);continue;}
    const dir=path.join(output,id);await mkdir(dir);const parameters={...p,output};
    const c={id,backend:'cuda',timeoutSeconds:p.timeoutSeconds,loadHostBudgetBytes:p.hostBudgetBytes,loadDeviceBudgetBytes:capped?24*GIB:p.workstationDeviceBytes,declaredSourceBindings:[{family:'h3',model:{modelId:'minimax_h3',baseModel:'minimax_h3'}}],runtime:anchorRuntime(p,capped,forceDense)};
    const service=new StockService(parameters,c,swRoot,inferenceRoot,dir,{hardDeadlineMs:Date.now()+(p.timeoutSeconds+900)*1000});service.staged=staged;
    let observer,currentStage='before-submit',record={id,sourcePair:{sceneworks:p.sceneworks,inference:p.inference},settings:request,startedAt:new Date().toISOString(),passed:false};
    try{
      await service.initialize();if(retained){const build=await json(path.join(dir,'service-build.json'));assert.equal(build.sha256,service.binaryHash);retainedWorkstationBuild(retained.build,build,{sceneworks:p.sceneworks,inference:p.inference});}
      observer=new ProcessObservations(service,{intervalMs:250,sample:async s=>({...await processSample(s),stage:currentStage})}).start();
      const transcript=[],scenario={caseId:id,baseUrl:'http://127.0.0.1:0',timeoutSeconds:p.timeoutSeconds,steps:[{id:'project',method:'POST',path:'/api/v1/projects',body:{name:`SW-F329 ${id}`},expectedStatus:201},{id:'submit',method:'POST',path:'/api/v1/video/jobs',body:{...request,projectId:{$capture:'project/id'}},expectedStatus:201},{id:'complete',waitJob:{$capture:'submit/id'},assert:[{pointer:'/status',equals:'completed'}]}]};
      const executed=await runScenario(scenario,{baseUrl:()=>service.baseUrl,fetchImpl:(...args)=>service.fetch(...args),beforeStep:({step,captures})=>service.checkStep(step,captures),onRecord:async row=>{transcript.push(row);if(row.result?.stage)currentStage=row.result.stage;if(row.method==='POST'&&row.route==='/api/v1/video/jobs'&&row.result?.id)service.jobs.add(row.result.id);await writeFile(path.join(dir,'api-transcript.json'),JSON.stringify(transcript,null,2)+'\n');}});
      const job=executed.captures.complete;record.job=job;assertQueuedMatchesSubmitted({...request,projectId:executed.captures.project.id},job);validateAnchorDevice(job,selected.physicalGpuIndex);
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
  const equivalence=await compareVedaAnchors(results,output,{sourceEquivalence:retained?.sourceEquivalence});
  if(retained){
    assert.equal(await fileSha256(p.retainedWorkstation.sourceAuditPath),p.retainedWorkstation.sourceAuditSha256);
    await retainedWorkstationSources(swRoot,inferenceRoot,{sceneworks:p.sceneworks,inference:p.inference},await json(p.retainedWorkstation.sourceAuditPath));
  }
  await sourcePair({...p,output},swRoot,inferenceRoot);await save(path.join(output,'result.json'),{feature:'SW-F329',passed:true,equivalence,sourcePair:{sceneworks:p.sceneworks,inference:p.inference},anchors:results.map(result=>({id:result.id,jobId:result.job.id,passed:result.passed,sourcePair:result.sourcePair,retained:result.id==='workstation-veda'?retained?.retention:undefined,envelope:result.envelope})),proofs});
}

async function main(){
  const [mode,swRoot,inferenceRoot,output]=process.argv.slice(2),p=JSON.parse(process.env.SW_F329_PLAN??'null');validatePlan(p);
  if(mode==='check')console.log('SW-F329 plan shape and explicit limits checked; no model executed');
  else if(mode==='device')console.log(JSON.stringify(campaignCudaSelection(p)));
  else{assert.equal(mode,'run');if(p.mode==='matched-resident-control'){const {runMatchedResidentControl}=await import('./h3-matched-resident-control.mjs');await runMatchedResidentControl(p,path.resolve(swRoot),path.resolve(inferenceRoot),path.resolve(output));}else await run(p,path.resolve(swRoot),path.resolve(inferenceRoot),path.resolve(output));}
}
// Complete this module's evaluation before the optional module imports StockService.
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch(error=>{console.error(error.stack);process.exitCode=1;});
