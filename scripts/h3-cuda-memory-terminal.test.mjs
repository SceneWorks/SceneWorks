import assert from 'node:assert/strict';
import test from 'node:test';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { readFile, writeFile, mkdtemp, rm, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { anchorRequest, validatePlan, validatePoolEvidence, validateMaterializingProofLog, materializingProofSources, validateVedaParityPair, rawNumericDifference, compareDecodedMedia, compareVedaAnchors, VEDA_EQUIVALENCE } from './h3-cuda-memory-terminal.mjs';
import { privateEnvironment } from './h3-finetune-terminal-runtime.mjs';

const GIB=1024**3;
function plan(){return {version:1,stationary_pair_authorized:true,sceneworks:'a'.repeat(40),inference:'b'.repeat(40),requestSha256:'c'.repeat(64),requestPath:'D:\\fixture.json',licenseAcknowledgmentsPath:'D:\\licenses.json',expectedRunner:'cuda-windows-2',seed:42,poolBytes:22*GIB,externalReserveBytes:2*GIB,hostBudgetBytes:64*GIB,workstationDeviceBytes:95*GIB,timeoutSeconds:7200,additionalSparse24:true,additionalSparse24Reason:'E5 compares normal Veda resident and streamed outputs',sources:Object.fromEntries([
  ['tier','SceneWorks--minimax-h3-mlx','137ce668c55a20bc0935fd1cf2a3de8448abb7f4'],
  ['base','MiniMaxAI--MiniMax-H3','6818f6c32d12b210915e44ad56a4228c2608f160'],
  ['predictor','Veda-Sparse--Minimax-H3-T2VA-Veda-8NFE-600Step-Preview','9a1fd3a41b4a754a7886e64e82edbddf599fd1bd'],
  ['turbo','lightx2v--Minimax-h3-Turbo','5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1'],
].map(([role,repo,revision])=>[role,{root:`E:\\huggingface\\hub\\models--${repo}\\snapshots\\${revision}`,revision}])),primitiveProof:{directory:'D:\\primitive',runUrl:'https://github.com/SceneWorks/inference/actions/runs/1'},materializingProof:{logPath:'D:\\fixture-proof.log',revision:'b'.repeat(40),runUrl:'https://github.com/SceneWorks/inference/actions/runs/2'}};}
function original(){return {model:'minimax_h3',mode:'text_to_video',prompt:'Fixture request; no generation.',width:1344,height:768,fps:24,duration:14.375,quality:'balanced',fitMode:'crop',seed:null,loras:[{id:'minimax_h3_turbo_8step',weight:1}],advanced:{veda:{enabled:true,fullAttentionLayers:'',fullAttentionSteps:'',generated:'90%',reference:'90%'}}};}
test('exact original geometry, Turbo8step and Veda controls survive all anchors',()=>{
  const source=original(),request=anchorRequest(source,42);assert.equal(request.seed,42);assert.equal(source.seed,null);assert.equal(request.advanced.mlxQuantize,4);assert.equal(request.advanced.steps,undefined);assert.equal(request.loras[0].id,'minimax_h3_turbo_8step');assert.deepEqual(request.advanced.veda,source.advanced.veda);assert.equal(request.requestedGpu,'1');
  for(const change of [{width:768},{duration:5.17},{quality:'fast'},{loras:[]}])assert.throws(()=>anchorRequest({...source,...change},42));
});
test('private worker environment adds only explicitly allowed proof controls',()=>{
  const baseline=privateEnvironment('D:\\fixture',{port:0,gpuId:'1',backend:'cuda'},'fixture-token');assert.equal(baseline.SCENEWORKS_CUDA_PROOF_POOL_BYTES,undefined);assert.equal(baseline.SCENEWORKS_H3_VEDA_FORCE_DENSE,undefined);
  const proof=privateEnvironment('D:\\fixture',{port:0,gpuId:'1',backend:'cuda',cudaProofEnvironment:{SCENEWORKS_CUDA_PROOF_POOL_BYTES:String(22*GIB),SCENEWORKS_H3_VEDA_FORCE_DENSE:'1'}},'fixture-token');assert.equal(proof.SCENEWORKS_CUDA_PROOF_POOL_BYTES,String(22*GIB));assert.equal(proof.SCENEWORKS_H3_VEDA_FORCE_DENSE,'1');
  for(const controls of [{PATH:'D:\\bad'},{SCENEWORKS_CUDA_PROOF_POOL_BYTES:'24GiB'},{SCENEWORKS_H3_ATTENTION_ROUTE:'auto'},{SCENEWORKS_H3_VEDA_FORCE_DENSE:'0'}])assert.throws(()=>privateEnvironment('D:\\fixture',{port:0,gpuId:'1',backend:'cuda',cudaProofEnvironment:controls},'fixture-token'));
});
test('logical admission override, missing pool observations and over-envelope samples cannot prove24GiB',()=>{
  const options={jobId:'job_fixture',poolBytes:22*GIB,externalReserveBytes:2*GIB,metrics:{samples:[{host:30*GIB,device:21*GIB},{host:31*GIB,device:22*GIB}]}};
  const event={event:'h3_cuda_proof_pool_snapshot',jobId:'job_fixture',logical_ordinal:0,cap_bytes:22*GIB,used_high_bytes:20*GIB,reserved_high_bytes:21*GIB,reserved_bytes:20*GIB,device_used_bytes:21*GIB};
  assert.equal(validatePoolEvidence([event],options).totalEnvelopeBytes,24*GIB);
  assert.throws(()=>validatePoolEvidence([],options));assert.throws(()=>validatePoolEvidence([{...event,reserved_high_bytes:23*GIB}],options));assert.throws(()=>validatePoolEvidence([event],{...options,metrics:{samples:[{host:30*GIB,device:25*GIB},{host:30*GIB,device:22*GIB}]}}));assert.throws(()=>validatePoolEvidence([event],{...options,jobId:'other-job'}));
  assert.throws(()=>validatePoolEvidence([{...event,device_used_bytes:23*GIB}],options));assert.throws(()=>validatePoolEvidence([{...event,device_used_bytes:undefined}],options));
});
test('plan refuses unenforced admission-only proof, enlarged24GiB envelope and missing frozen authorization',()=>{
  validatePlan(plan());
  assert.throws(()=>validatePlan({version:1,stationary_pair_authorized:false}));
  // Shape validation precedes all I/O/device/service calls; an incomplete request cannot start work.
  assert.throws(()=>validatePlan({version:1,stationary_pair_authorized:true,sceneworks:'a'.repeat(40),inference:'b'.repeat(40),requestSha256:'c'.repeat(64),expectedRunner:'cuda-windows-2',requestPath:'D:\\fixture.json',licenseAcknowledgmentsPath:'D:\\license.json',seed:42,poolBytes:24*GIB,externalReserveBytes:2*GIB}));
  assert.throws(()=>validatePlan({...plan(),additionalSparse24Reason:''}));
  assert.throws(()=>validatePlan({...plan(),additionalSparse24:false}));
  const missing=plan();delete missing.additionalSparse24;assert.throws(()=>validatePlan(missing));
});
test('assembled CLI checks plan shape without services/files/device work and refuses local execution',()=>{
  const entry=fileURLToPath(new URL('./h3-cuda-memory-terminal.mjs',import.meta.url));
  const env={...process.env,SW_F329_PLAN:JSON.stringify(plan()),GITHUB_ACTIONS:'false'};
  const checked=spawnSync(process.execPath,[entry,'check'],{env,encoding:'utf8'});assert.equal(checked.status,0,checked.stderr);assert.match(checked.stdout,/no model executed/);
  const refused=spawnSync(process.execPath,[entry,'run','D:\\source','D:\\inference','D:\\Codex\\work\\fixture'],{env,encoding:'utf8'});assert.notEqual(refused.status,0);assert.ok(!refused.stdout.includes('CUDA_PROOF_POOL'));
});
test('dense fallback marker alone or a skipped/CPU body cannot replace actual materializing CUDA proof',()=>{
  const body='MODEL_INFERENCE_BODY_COMPLETED veda::tests::actual_cuda_kernel_gate';
  const routes='H3_DENSE_ATTENTION_ROUTES AttentionExecutionReport { materialized_calls: 2 }';
  const receipts=['BF16','F16'].map(dtype=>'S3_DEVICE_RECEIPT '+JSON.stringify({schema:'sc-25777-actual-cuda-kernel-v2',dtype,device_ordinal:0,sparse_calls:1,forced_failure_dense:true})).join('\n');
  const summary='test result: ok. 1 passed; 0 failed; 0 ignored';
  const log=[body,routes,receipts,summary].join('\n');assert.equal(validateMaterializingProofLog(log).materializedCalls,2);
  for(const invalid of [log.replace(body,''),log.replace(receipts,''),log.replace('materialized_calls: 2','materialized_calls: 0'),log.replace(summary,'test result: ok. 0 passed; 0 failed; 1 ignored')])assert.throws(()=>validateMaterializingProofLog(invalid));
});
test('retained Veda proof maps accounting-only movement and rejects numerical, getter and build source changes',async()=>{
  const root='D:\\unit_fake_inference',reference='a'.repeat(40),target='b'.repeat(40),object='c'.repeat(40);
  const h3='crates/media/candle-gen/candle-gen-minimax-h3';
  const accounting=[`${h3}/src/memory_strategy/adapters.rs`,`${h3}/src/memory_strategy/streaming.rs`,'crates/contracts/gen-core/src/h3_memory/window.rs'];
  const calls=[];
  const resolver=(changed=[])=>async(command,args,options)=>{
    assert.equal(command,'git');assert.deepEqual(args.slice(0,2),['rev-parse','--verify']);assert.equal(options.cwd,root);
    const [revision,selected]=args[2].split(':');calls.push(selected);
    return {stdout:(revision===target&&changed.includes(selected)?'d'.repeat(40):object)+'\n'};
  };
  // Accounting changes also move the old whole-H3 tree proxy, without changing the direct case.
  const sources=await materializingProofSources(root,reference,target,resolver([...accounting,h3]));
  assert.ok(sources.length>0);assert.ok(accounting.every(file=>!calls.includes(file)));assert.ok(!calls.includes(h3));
  for(const changed of [`${h3}/src/veda.rs`,`${h3}/src/veda_pointer_v1.cu`,`${h3}/src/nn.rs`,`${h3}/src/memory_strategy.rs`,`${h3}/src/memory_strategy/execution.rs`,
    'crates/media/candle-gen/candle-gen','crates/kernels/candle-quant-kernels',
    'crates/media/candle-gen/vendor/candle-core','Cargo.toml','Cargo.lock']){
    assert.ok(calls.includes(changed),`missing numerical source dependency: ${changed}`);
    await assert.rejects(materializingProofSources(root,reference,target,resolver([changed])),/numerical evidence source changed/);
  }
  await assert.rejects(materializingProofSources(root,reference,target,async()=>{throw new Error('missing Git source object');}),/missing Git source object/);
  await assert.rejects(materializingProofSources(root,reference,target,async()=>({stdout:'not-an-object'})),/numerical source object must exist/);
});
test('manual wrapper keeps CUDA memory acceptance separate from ordinary CI and runs early startup checks',async()=>{
  const caller=await readFile(new URL('../.github/workflows/server-candle-linux.yml',import.meta.url),'utf8');
  const workflow=await readFile(new URL('../.github/workflows/h3-cuda-memory-terminal.yml',import.meta.url),'utf8');
  assert.match(caller,/h3-cuda-memory:\n\s+if: \$\{\{ github\.event_name == 'workflow_dispatch' && inputs\.starvector_terminal_operation == 'h3-cuda-memory' \}\}\n\s+uses: \.\/\.github\/workflows\/h3-cuda-memory-terminal\.yml/);
  assert.match(workflow,/workflow_call:/);assert.doesNotMatch(workflow,/\n\s+(?:pull_request|push|schedule|merge_group):/);
  assert.match(workflow,/github\.event_name == 'workflow_dispatch'.*stationary_pair_authorized == true/);
  assert.match(workflow,/group: windows-candle-gpu-real-weights/);assert.match(workflow,/nvidia-smi -i 1 --query-gpu=uuid/);
  assert.ok(workflow.indexOf('Check script entry point')<workflow.indexOf('Build and execute the three'));
  assert.match(workflow,/python scripts\\h3-cuda-memory-lease\.test\.py/);
  assert.match(workflow,/python scripts\\h3-cuda-memory-lease\.py "%GITHUB_WORKSPACE%\\sceneworks" "%GITHUB_WORKSPACE%\\inference" "%SW_F329_OUTPUT%"/);
  assert.doesNotMatch(workflow,/node scripts\\h3-cuda-memory-terminal\.mjs run/);
  assert.match(workflow,/\$\{\{ env\.SW_F329_OUTPUT \}\}-lease\/\*\.json/);
});


const hash='1'.repeat(64);
function parityRecords(){
  const settings=anchorRequest(original(),42),sourcePair={sceneworks:'a'.repeat(40),inference:'b'.repeat(40)};
  return ['workstation-veda','24gib-veda-sparse'].map((id,index)=>{
    const jobId=`unit_fake_${id}`,projectId=`unit_fake_project_${index}`,windowed=index===1;
    const video={codec_type:'video',codec_name:'h264',width:1344,height:768,nb_read_frames:345,nb_frames:345,avg_frame_rate:'24/1',r_frame_rate:'24/1',duration:'14.375',start_time:'0'};
    const audio={codec_type:'audio',codec_name:'aac',sample_rate:'32000',channels:2,duration:'14.375',start_time:'0'};
    const events=[{event:'video_memory_strategy_selected',route:'minimax_h3',request_frames:345,strategy:windowed?'BoundedTransformerResidency':'Resident'},
      {event:'h3_attention_execution',transformer_window:windowed?'Some(1)':'None',transformer_component:windowed?'Some(Both)':'None'}];
    if(windowed)for(const component of ['text_encoder','dit'])events.push({event:'h3_transformer_window_materialized',component,first_block:0,block_count:1,base_bytes:100,device:'Cuda { gpu_id: 0 }'});
    return {id,passed:true,sourcePair:structuredClone(sourcePair),settings:structuredClone(settings),job:{id:jobId,status:'completed',projectId,requestedGpu:'1',payload:{...structuredClone(settings),projectId}},
      generationMetrics:{vedaExecution:{requested:true,sparseCalls:8,kernel:'unit_fake actual sparse kernel',predictor:'unit_fake predictor',plan:'unit_fake same plan',executionGrid:[345,768,1344]}},
      media:[{jobId,sha256:hash,av:'unit_fake.mp4',decoded:{sha256:hash,streams:[video,audio]}}],events};
  });
}
async function directory(t){const parent=path.resolve(tmpdir()),dir=await mkdtemp(path.join(parent,'sw-f329-parity-'));t.after(async()=>{assert.equal(path.dirname(path.resolve(dir)),parent);await rm(dir,{recursive:true,force:true});});return dir;}
function pcm(values){const buffer=Buffer.alloc(values.length*4);values.forEach((value,index)=>buffer.writeFloatLE(value,index*4));return buffer;}

test('Veda equivalence requires actual sparse execution and different selected/executed residency with identical controls',()=>{
  const records=parityRecords();assert.equal(validateVedaParityPair(records).witnesses[1].sparseCalls,8);
  const mutations=[
    r=>r.pop(),r=>r.push(structuredClone(r[1])),r=>r[1].id='24gib-veda-dense',r=>r[1].passed=false,
    r=>r[1].job.id=r[0].job.id,r=>r[1].settings.seed=43,r=>r[1].job.payload.advanced.veda.generated='80%',
    r=>r[1].sourcePair.inference='c'.repeat(40),r=>r[1].generationMetrics.vedaExecution.sparseCalls=0,
    r=>r[1].generationMetrics.vedaExecution.fallbackReason='forced for validation',r=>r[1].generationMetrics.vedaExecution.kernel='',
    r=>r[1].generationMetrics.vedaExecution.plan='different-plan',r=>r[1].media[0].jobId='wrong-job',
    r=>r[1].media[0].decoded.streams[0].nb_read_frames=124,r=>r[1].media[0].decoded.streams[1].channels=1,
    r=>r[1].events[0].strategy='Resident',r=>r[1].events[1].transformer_window='None',r=>r[1].events[1].transformer_component='Some(Dit)',
    r=>r[1].events=r[1].events.filter(e=>e.component!=='text_encoder'),r=>r[1].events[2].device='Cpu',
    r=>r[1].events[2].block_count=50,r=>r[0].events.push({...r[1].events[2]}),r=>r[1].events.pop(),
  ];
  for(const mutate of mutations){const broken=parityRecords();mutate(broken);assert.throws(()=>validateVedaParityPair(broken));}
});

test('full raw numerical comparison sees late frames, every channel, non-finite PCM and truncation',async t=>{
  const dir=await directory(t),a=path.join(dir,'reference.raw'),b=path.join(dir,'candidate.raw');
  const baseline=Buffer.alloc(65538,100);await writeFile(a,baseline);await writeFile(b,baseline);
  assert.equal((await rawNumericDifference(a,b,{stream:'video',channels:3,expectedElements:baseline.length})).rmse,0);
  const changed=Buffer.from(baseline);changed[changed.length-1]=255;await writeFile(b,changed);
  const metric=await rawNumericDifference(a,b,{stream:'video',channels:3,expectedElements:baseline.length});
  assert.ok(metric.maxAbs>0.6);assert.ok(metric.channelRmse[2]>VEDA_EQUIVALENCE.video,'last pixel after the chunk boundary must affect its native channel');
  await writeFile(b,changed.subarray(0,3));await assert.rejects(rawNumericDifference(a,b,{stream:'video',channels:3}));
  await writeFile(a,pcm([0.1,0.2,0.3,0.4]));await writeFile(b,pcm([0.1,-0.2,0.3,-0.4]));
  const audio=await rawNumericDifference(a,b,{stream:'audio',channels:2,expectedElements:4});assert.equal(audio.channelRmse[0],0);assert.ok(audio.channelRmse[1]>0.5);
  for(const value of [NaN,Infinity]){await writeFile(b,pcm([0.1,value,0.3,0.4]));await assert.rejects(rawNumericDifference(a,b,{stream:'audio',channels:2}),/non-finite/);}
});

test('assembled decoder uses all native frames/stereo samples and fixed existing H3 RMS ceiling, with bounded cleanup',async t=>{
  const dir=await directory(t),calls=[];
  const makeCommand=({videoChange=false,audioChange=false,swapChannels=false,truncateVideo=false,truncateAudio=false,truncateVideoBoth=false,truncateAudioBoth=false}={})=>async(command,args)=>{
    assert.equal(command,'ffmpeg');calls.push(args);assert.ok(args.includes('-xerror'));assert.ok(args.includes('-n'));
    assert.ok(!args.includes('-vf')&&!args.includes('-ar')&&!args.includes('-ac'),'no frame sampling, resampling or mono collapse');
    const candidate=args[args.indexOf('-i')+1]==='candidate',video=args.includes('rawvideo');
    let output=video?Buffer.alloc(36,100):pcm([0.1,0.2,0.3,0.4]);
    if(candidate&&videoChange)output[35]=255;
    if(candidate&&audioChange)output=pcm([0.1,-0.2,0.3,-0.4]);
    if(candidate&&swapChannels)output=pcm([0.2,0.1,0.4,0.3]);
    if((video&&truncateVideoBoth)||(!video&&truncateAudioBoth))output=output.subarray(0,video?3:8);
    if(candidate&&((video&&truncateVideo)||(!video&&truncateAudio)))output=output.subarray(0,video?3:8);
    await writeFile(args.at(-1),output);return {stdout:'',stderr:''};
  };
  const options={videoElements:36,audioSamples:2,audioChannels:2,sampledDelta:async()=>0};
  const good=await compareDecodedMedia('reference','candidate',dir,{...options,command:makeCommand()});
  assert.equal(good.streams.video.elements,36);assert.equal(good.streams.audio.elements,4);assert.equal(good.streams.video.maximumRmse,0.001);
  for(const mutation of [{videoChange:true},{audioChange:true},{swapChannels:true},{truncateVideo:true},{truncateAudio:true},{truncateVideoBoth:true},{truncateAudioBoth:true}])await assert.rejects(compareDecodedMedia('reference','candidate',dir,{...options,command:makeCommand(mutation)}));
  for(const sampled of [0.01,-1,NaN,undefined])await assert.rejects(compareDecodedMedia('reference','candidate',dir,{...options,command:makeCommand(),sampledDelta:async()=>sampled}),/decodedDelta/);
  assert.ok(calls.length>0);assert.deepEqual(await readdir(dir),[],'generated raw vectors never become retained giant artifacts');
});

test('equivalence failure writes a failed receipt and prevents a two-anchor-only successful campaign',async t=>{
  const dir=await directory(t);await assert.rejects(compareVedaAnchors([parityRecords()[0]],dir));
  const receipt=JSON.parse(await readFile(path.join(dir,'veda-equivalence.json'),'utf8'));assert.equal(receipt.passed,false);assert.match(receipt.error,/24gib-veda-sparse/);
  const source=await readFile(new URL('./h3-cuda-memory-terminal.mjs',import.meta.url),'utf8');
  assert.ok(source.indexOf('const equivalence=await compareVedaAnchors(results,output)')<source.indexOf("{feature:'SW-F329',passed:true,equivalence"));
});


test('fixed RMS policy stays bound to the existing same-backend CUDA H3 campaign ceiling',async()=>{
  const prior=JSON.parse(await readFile(new URL('../docs/testing/sc-26691-campaign/h3-terminal-cuda.json',import.meta.url),'utf8'));
  for(const stream of ['video','audio'])assert.equal(VEDA_EQUIVALENCE[stream],prior.bootstrap.operator.maximumBaselineDelta);
});
