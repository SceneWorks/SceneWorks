import assert from 'node:assert/strict';
import test from 'node:test';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { readFile, writeFile, mkdtemp, mkdir as mkdirForTest, rm, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { anchorRequest, validatePlan, validatePoolEvidence, validateMaterializingProofLog, materializingProofSources, primitiveProofSource, validatePrimitiveProofLog, retainedWorkstationSources, validateRetainedWorkstationRecord, retainedWorkstationWeights, retainedWorkstationHardware, retainedWorkstationBuild, RETAINED_WORKSTATION, validateVedaParityPair, rawNumericDifference, compareDecodedMedia, compareVedaAnchors, VEDA_EQUIVALENCE } from './h3-cuda-memory-terminal.mjs';
import { privateEnvironment } from './h3-finetune-terminal-runtime.mjs';
import { validateBoundedQkResidentSources, validateRetainedWindowProvenance } from './h3-cuda-memory-terminal.mjs';
import { sha256 } from './lib/file-sha256.mjs';
import { parse as parseProvenanceCandidates, populationSha256 } from './scan-inference-provenance.mjs';
import { campaignCudaSelection,campaignCudaEnvironment,anchorRuntime,validateAnchorDevice } from './h3-cuda-memory-terminal.mjs';
import { collectDecodedMediaDiagnostics } from './h3-cuda-memory-terminal.mjs';
import { validateCudaRuntimeDevice,hardware } from './h3-finetune-terminal-runtime.mjs';

const GIB=1024**3;
const GPU0='GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0',GPU1='GPU-e4b79931-7be6-f216-460a-f5405cfafffe';
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
test('explicit GPU0 preserves typed zero through all fresh anchor requests and owned runtime configuration',()=>{
  const p={...plan(),physicalGpuIndex:0,physicalGpuUuid:GPU0,poolBytes:21*GIB,externalReserveBytes:3*GIB,hostBudgetBytes:384*GIB};
  assert.equal(validatePlan(p),p);assert.deepEqual(campaignCudaSelection(plan()),{physicalGpuIndex:1,physicalGpuUuid:GPU1});
  const request=anchorRequest(original(),42,0),legacy=anchorRequest(original(),42);
  assert.deepEqual({...request,requestedGpu:'1'},legacy);assert.equal(request.requestedGpu,'0');
  validateAnchorDevice({requestedGpu:'0',assignedGpu:'0'},0);
  for(const job of [{requestedGpu:'1',assignedGpu:'0'},{requestedGpu:'0',assignedGpu:'1'},{requestedGpu:'0'}])assert.throws(()=>validateAnchorDevice(job,0));
  for(const [capped,dense] of [[false,false],[true,true],[true,false]]){
    const runtime=anchorRuntime(p,capped,dense);assert.equal(runtime.gpuId,'0');assert.equal(runtime.deviceId,GPU0);
    assert.deepEqual(validateCudaRuntimeDevice(runtime),{physicalGpuIndex:0,physicalGpuUuid:GPU0});
    assert.equal(privateEnvironment('D:\\unit_fake', {...runtime,backend:'cuda'},'unit_fake_token').SCENEWORKS_GPU_ID,'0');
    assert.deepEqual(runtime.memoryCap,capped?{kind:'cuda-vram-gb',value:24}:undefined);
    assert.equal(runtime.cudaProofEnvironment.SCENEWORKS_CUDA_PROOF_POOL_BYTES,capped?String(21*GIB):undefined);
    assert.equal(runtime.cudaProofEnvironment.SCENEWORKS_H3_VEDA_FORCE_DENSE,dense?'1':undefined);
  }
  for(const physicalGpuIndex of [-1,2,true,false,0.5,'0',null])assert.throws(()=>validatePlan({...p,physicalGpuIndex}));
  for(const physicalGpuUuid of [undefined,null,'0',GPU1])assert.throws(()=>validatePlan({...p,physicalGpuUuid}));
  assert.throws(()=>validatePlan({...p,retainedWorkstation:{}}),/fresh workstation/);
  assert.throws(()=>validateCudaRuntimeDevice({...anchorRuntime(p,false,false),gpuId:'1'}));
  assert.throws(()=>validateCudaRuntimeDevice({...anchorRuntime(p,false,false),deviceId:GPU1}));
  assert.deepEqual(validateCudaRuntimeDevice({gpuId:'1'}),{physicalGpuIndex:1,physicalGpuUuid:undefined});
});
test('explicit GPU0 environment and hardware bind UUID, PCI order and NVML index without ordinal fallback',async()=>{
  const p={physicalGpuIndex:0,physicalGpuUuid:GPU0},env={SCENEWORKS_GPU_ID:'0',CUDA_VISIBLE_DEVICES:GPU0,CUDA_DEVICE_ORDER:'PCI_BUS_ID'};
  assert.deepEqual(campaignCudaEnvironment(p,env),p);
  const csv=`0, ${GPU0}, 00000000:01:00.0, unit_fake_zero, unit_fake_driver\n1, ${GPU1}, 00000000:41:00.0, unit_fake_one, unit_fake_driver`;
  const exec=async()=>({stdout:csv});
  const result=await hardware('cuda',GPU0,{...p,env,exec});assert.equal(result.cuda.selectedPhysicalDevice.nvmlIndex,0);
  for(const mutation of [{SCENEWORKS_GPU_ID:'1'},{CUDA_VISIBLE_DEVICES:GPU1},{CUDA_VISIBLE_DEVICES:'0'},{CUDA_DEVICE_ORDER:'FASTEST_FIRST'}]){
    assert.throws(()=>campaignCudaEnvironment(p,{...env,...mutation}));
    await assert.rejects(hardware('cuda',GPU0,{...p,env:{...env,...mutation},exec}));
  }
  await assert.rejects(hardware('cuda',GPU1,{...p,env,exec}));
  await assert.rejects(hardware('cuda',GPU0,{...p,env,exec:async()=>({stdout:csv.replace('0,','2,')})}),/NVML identity/);
});
test('assembled device CLI validates default1 and explicit0 before setup without running models',()=>{
  const script=fileURLToPath(new URL('./h3-cuda-memory-terminal.mjs',import.meta.url));
  for(const [p,index,uuid] of [[plan(),1,GPU1],[{...plan(),physicalGpuIndex:0,physicalGpuUuid:GPU0},0,GPU0]]){
    const result=spawnSync(process.execPath,[script,'device'],{env:{...process.env,SW_F329_PLAN:JSON.stringify(p)},encoding:'utf8'});
    assert.equal(result.status,0,result.stderr);assert.deepEqual(JSON.parse(result.stdout),{physicalGpuIndex:index,physicalGpuUuid:uuid});
  }
  for(const mutation of [{physicalGpuIndex:false},{physicalGpuIndex:0},{physicalGpuIndex:0,physicalGpuUuid:GPU1},{physicalGpuIndex:0,physicalGpuUuid:GPU0,retainedWorkstation:{}}]){
    const result=spawnSync(process.execPath,[script,'device'],{env:{...process.env,SW_F329_PLAN:JSON.stringify({...plan(),...mutation})},encoding:'utf8'});
    assert.notEqual(result.status,0);assert.equal(result.stdout,'');
  }
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
test('capped job may start at zero but must witness positive usage within its own bounded snapshots',()=>{
  const options={jobId:'owned_job',poolBytes:22*GIB,externalReserveBytes:2*GIB,metrics:{samples:[{host:30*GIB,device:GIB},{host:31*GIB,device:21*GIB}]}};
  const zero={event:'h3_cuda_proof_pool_snapshot',job_id:'owned_job',logical_ordinal:0,cap_bytes:22*GIB,used_high_bytes:0,reserved_high_bytes:0,reserved_bytes:0,device_used_bytes:GIB};
  const positive={...zero,used_high_bytes:19*GIB,reserved_high_bytes:20*GIB,reserved_bytes:20*GIB,device_used_bytes:21*GIB};
  assert.equal(validatePoolEvidence([zero,positive],options).reservedHighBytes,20*GIB);
  assert.throws(()=>validatePoolEvidence([zero,{...zero}],options),/actual tensors for this job/);
  assert.throws(()=>validatePoolEvidence([zero,{...positive,job_id:'other_job'}],options),/actual tensors for this job/);
  for(const bad of [{reserved_high_bytes:23*GIB},{used_high_bytes:23*GIB},{used_high_bytes:NaN},{used_high_bytes:Infinity},{used_high_bytes:undefined},{cap_bytes:undefined},{device_used_bytes:undefined},{reserved_bytes:21*GIB},{device_used_bytes:23*GIB}]){
    for(const rows of [[{...zero,...bad},positive],[zero,{...positive,...bad}]])assert.throws(()=>validatePoolEvidence(rows,options));
  }
  for(const field of ['host','device'])for(const invalid of [undefined,0,NaN,Infinity])assert.throws(()=>validatePoolEvidence([zero,positive],{...options,metrics:{samples:[{host:GIB,device:GIB},{host:GIB,device:GIB,[field]:invalid}]}}));
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
    assert.equal(command,'git');assert.equal(options.cwd,root);
    if(args[0]==='diff')return {stdout:`${args.at(-1)}/src/numerical.rs\n`};
    assert.deepEqual(args.slice(0,2),['rev-parse','--verify']);
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

test('Veda tree mappings preserve measured identities and permit only the exact documentation or unreachable integration fixture',async()=>{
  const reference='a'.repeat(40),target='b'.repeat(40),oldObject='c'.repeat(40),newObject='d'.repeat(40);
  const cases=[['crates/media/candle-gen/vendor/candle-core','VENDORED.md'],['crates/media/candle-gen/candle-gen','tests/cuda_proof_pool.rs']];
  for(const [selected,file] of cases){
    const excluded=`${selected}/${file}`,code=`${selected}/src/lib.rs`;
    const command=(mutation)=>async(program,args)=>{
      assert.equal(program,'git');
      if(args[0]==='rev-parse')return {stdout:args[2]===`${target}:${selected}`?newObject:oldObject};
      if(args[0]==='diff')return {stdout:mutation==='other-path'?`${code}\n`:mutation==='extra-path'?`${excluded}\n${code}\n`:`${excluded}\n`};
      assert.equal(args[0],'ls-tree');
      const after=args[2]===target;
      if(after&&mutation==='missing-tree')return {stdout:''};
      if(after&&mutation==='malformed-tree')return {stdout:'not a tree row'};
      const doc=`${after&&mutation==='file-mode'?'100755':'100644'} blob ${after?newObject:oldObject}\t${excluded}`;
      const source=`${after&&mutation==='code-mode'?'100755':'100644'} blob ${after&&mutation==='code-object'?newObject:oldObject}\t${code}`;
      return {stdout:[...(after&&mutation==='missing-file'?[]:[doc]),...(after&&mutation==='deleted-code'?[]:[source]),...(after&&mutation==='added-code'?[`100644 blob ${oldObject}\t${selected}/src/extra.rs`]:[])].join('\n')};
    };
    const sources=await materializingProofSources('D:\\unit_fake',reference,target,command());
    assert.equal(sources.length,45);
    const mapped=sources.find(source=>source.path===selected);
    assert.equal(mapped.object,oldObject);assert.equal(mapped.targetObject,newObject);
    assert.deepEqual(mapped.mapping.changedPaths,[excluded]);
    assert.deepEqual(mapped.mapping.measuredFile,{mode:'100644',object:oldObject,path:excluded});
    assert.deepEqual(mapped.mapping.targetFile,{mode:'100644',object:newObject,path:excluded});
    assert.deepEqual(mapped.mapping.unchangedRecursiveBlobs,[{mode:'100644',object:oldObject,path:code}]);
    for(const mutation of ['other-path','extra-path','missing-tree','malformed-tree','file-mode','code-mode','code-object','missing-file','deleted-code','added-code']){
      await assert.rejects(materializingProofSources('D:\\unit_fake',reference,target,command(mutation)),undefined,`${selected}: ${mutation}`);
    }
  }
});
test('primitive proof retains its measured revision only across an ancestor with the identical whole tree',async()=>{
  const measured='a'.repeat(40),target='b'.repeat(40),tree='c'.repeat(40),calls=[];
  const command=async(program,args)=>{assert.equal(program,'git');calls.push(args);return {stdout:tree+'\n'};};
  const proof=await primitiveProofSource('D:\\unit_fake',measured,target,command);
  assert.equal(proof.measuredRevision,measured);assert.equal(proof.targetRevision,target);assert.equal(proof.tree,tree);
  assert.deepEqual(calls[0],['merge-base','--is-ancestor',measured,target]);
  await assert.rejects(primitiveProofSource('D:\\unit_fake',measured,target,async(_program,args)=>{
    if(args[0]==='merge-base')throw new Error('not an ancestor');return {stdout:tree};
  }),/not an ancestor/);
  await assert.rejects(primitiveProofSource('D:\\unit_fake',measured,target,async(_program,args)=>({stdout:args.at(-1)===`${target}^{tree}`?'d'.repeat(40):tree})),/identical complete source trees/);
  await assert.rejects(primitiveProofSource('D:\\unit_fake',measured,target,async()=>({stdout:'missing'})),/exact source revision/);
});
test('manual wrapper keeps CUDA memory acceptance separate from ordinary CI and runs early startup checks',async()=>{
  const caller=await readFile(new URL('../.github/workflows/server-candle-linux.yml',import.meta.url),'utf8');
  const workflow=await readFile(new URL('../.github/workflows/h3-cuda-memory-terminal.yml',import.meta.url),'utf8');
  assert.match(caller,/h3-cuda-memory:\n\s+if: \$\{\{ github\.event_name == 'workflow_dispatch' && inputs\.starvector_terminal_operation == 'h3-cuda-memory' \}\}\n\s+uses: \.\/\.github\/workflows\/h3-cuda-memory-terminal\.yml/);
  assert.match(workflow,/workflow_call:/);assert.doesNotMatch(workflow,/\n\s+(?:pull_request|push|schedule|merge_group):/);
  assert.match(workflow,/github\.event_name == 'workflow_dispatch'.*stationary_pair_authorized == true/);
  assert.match(workflow,/group: windows-candle-gpu-real-weights/);assert.match(workflow,/nvidia-smi -i \$selection\.physicalGpuIndex --query-gpu=uuid/);
  assert.ok(workflow.indexOf('h3-cuda-memory-terminal.mjs device')<workflow.indexOf('nvidia-smi -i'));
  assert.match(workflow,/\$uuid -cne \$selection\.physicalGpuUuid/);
  assert.match(workflow,/SCENEWORKS_GPU_ID=\$\(\$selection\.physicalGpuIndex\)/);
  assert.ok(workflow.indexOf('Check script entry point')<workflow.indexOf('Build and execute the three'));
  assert.match(workflow,/python scripts\\h3-cuda-memory-lease\.test\.py/);
  assert.match(workflow,/python scripts\\h3-cuda-memory-lease\.py "%GITHUB_WORKSPACE%\\sceneworks" "%GITHUB_WORKSPACE%\\inference" "%SW_F329_OUTPUT%"/);
  assert.doesNotMatch(workflow,/node scripts\\h3-cuda-memory-terminal\.mjs run/);
  assert.match(workflow,/\$\{\{ env\.SW_F329_OUTPUT \}\}-lease\/\*\.json/);
  assert.equal(workflow.split('            ${{ env.SW_F329_OUTPUT }}/workstation-veda/original/**/*.json\n').length,2,'all six nested original receipts must be uploaded once');
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
  const makeCommand=({videoChange=false,audioChange=false,swapChannels=false,nonfiniteAudio=false,truncateVideo=false,truncateAudio=false,truncateVideoBoth=false,truncateAudioBoth=false}={})=>async(command,args)=>{
    assert.equal(command,'ffmpeg');calls.push(args);assert.ok(args.includes('-xerror'));assert.ok(args.includes('-n'));
    assert.ok(!args.includes('-vf')&&!args.includes('-ar')&&!args.includes('-ac'),'no frame sampling, resampling or mono collapse');
    const candidate=args[args.indexOf('-i')+1]==='candidate',video=args.includes('rawvideo');
    let output=video?Buffer.alloc(36,100):pcm([0.1,0.2,0.3,0.4]);
    if(candidate&&video&&videoChange)output[35]=255;
    if(candidate&&!video&&audioChange)output=pcm([0.1,-0.2,0.3,-0.4]);
    if(candidate&&!video&&swapChannels)output=pcm([0.2,0.1,0.4,0.3]);
    if(candidate&&!video&&nonfiniteAudio)output=pcm([0.1,NaN,0.3,0.4]);
    if((video&&truncateVideoBoth)||(!video&&truncateAudioBoth))output=output.subarray(0,video?3:8);
    if(candidate&&((video&&truncateVideo)||(!video&&truncateAudio)))output=output.subarray(0,video?3:8);
    await writeFile(args.at(-1),output);return {stdout:'',stderr:''};
  };
  const options={videoElements:36,audioSamples:2,audioChannels:2,sampledDelta:async()=>0};
  const good=await compareDecodedMedia('reference','candidate',dir,{...options,command:makeCommand()});
  assert.equal(good.streams.video.elements,36);assert.equal(good.streams.audio.elements,4);assert.equal(good.streams.video.maximumRmse,0.001);
  const diagnostics=await collectDecodedMediaDiagnostics('reference','candidate',dir,{...options,command:makeCommand({videoChange:true,audioChange:true})});
  assert.ok(diagnostics.streams.video.rmse>VEDA_EQUIVALENCE.video);assert.ok(diagnostics.streams.audio.rmse>VEDA_EQUIVALENCE.audio);
  assert.equal(diagnostics.policy,undefined);assert.equal(diagnostics.streams.video.maximumRmse,undefined,'collection alone does not grade acceptance');
  assert.deepEqual(await readdir(dir),[],'successful ungraded diagnostics remove every generated vector');
  const beforeFailure=calls.length;
  await assert.rejects(compareDecodedMedia('reference','candidate',dir,{...options,command:makeCommand({videoChange:true,audioChange:true})}),error=>{
    assert.match(error.message,/video full decoded/);
    const report=error.comparison;assert.equal(report.policy,VEDA_EQUIVALENCE);
    assert.equal(report.streams.video.elements,36);assert.equal(report.streams.audio.elements,4);assert.equal(report.streams.audio.channels,2);
    assert.ok(report.streams.audio.channelRmse[1]>0.5);assert.equal(report.streams.audio.sampledRmse,0);
    assert.equal(report.streams.audio.maximumRmse,VEDA_EQUIVALENCE.audio);return true;
  });
  assert.equal(calls.length-beforeFailure,4,'video grading failure must follow all four native decodes');
  assert.deepEqual(await readdir(dir),[],'video grading failure removes both complete RGB and PCM pairs');
  for(const mutation of [{videoChange:true},{audioChange:true},{swapChannels:true},{nonfiniteAudio:true},{truncateVideo:true},{truncateAudio:true},{truncateVideoBoth:true},{truncateAudioBoth:true}]){
    await assert.rejects(compareDecodedMedia('reference','candidate',dir,{...options,command:makeCommand(mutation)}));
    assert.deepEqual(await readdir(dir),[],'each invalid stream or grading refusal cleans its own outputs');
  }
  const decode=makeCommand();
  await assert.rejects(collectDecodedMediaDiagnostics('reference','candidate',dir,{...options,command:async(program,args)=>{
    await decode(program,args);if(args.includes('f32le')&&args[args.indexOf('-i')+1]==='candidate')throw new Error('injected audio decode failure after output creation');
  }}),/injected audio decode failure/);
  assert.deepEqual(await readdir(dir),[],'partial decoder outputs are cleaned on command failure');
  const beforeSampled=calls.length;
  await assert.rejects(collectDecodedMediaDiagnostics('reference','candidate',dir,{...options,command:makeCommand(),sampledDelta:async()=>{
    assert.equal(calls.length-beforeSampled,4,'both full streams precede the optional sampled witness');throw new Error('injected sampled witness failure');
  }}),error=>{
    assert.match(error.message,/sampled witness failure/);assert.equal(error.comparison.streams.video.elements,36);assert.equal(error.comparison.streams.audio.elements,4);return true;
  });
  assert.deepEqual(await readdir(dir),[],'sampled witness failure cleans all complete native outputs');
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

function retainedRecord(){
  const record=parityRecords()[0];record.sourcePair=structuredClone(RETAINED_WORKSTATION.sourcePair);
  record.job.id=RETAINED_WORKSTATION.jobId;record.job.assignedGpu='1';
  record.media[0].jobId=record.job.id;record.media[0].sha256=RETAINED_WORKSTATION.mediaSha256;record.media[0].decoded.sha256=record.media[0].sha256;
  record.media[0].asset={recipe:{rawAdapterSettings:{effectiveSteps:8,minimaxH3Tier:'q4'}}};
  Object.assign(record.generationMetrics,{backend:'cuda',steps:8,seed:42,width:1344,height:768,quantBits:4});
  record.generationMetrics.vedaExecution.kernel='cuda-veda-sage-int8-tiled-v2';
  record.content={frames:345,spatialStd:{min:0.12,floor:0.07}};record.observations={hostPeakBytes:30*GIB,devicePeakBytes:42*GIB};return record;
}

// Source text only: deliberately small route/ownership fixtures, never compiled or executed.
function boundedQkSourceFixture(){
  const qkv='{\nlet q = rms_weighted(x, weight, eps)?;\nlet k = q.clone();\nlet v = k.clone();\n(q, k, v)\n}';
  const layers=`impl DitAttention {\nfn forward_veda() {\nlet (q, k, v) = ${qkv};\nfinish(q,k,v);\n}\n}`;
  const nextLayers=layers.replace(`= ${qkv}`,`= if rope.is_some() && x.device().is_cuda() && crate::block_stream::current().is_some() {\nprecondition_projected_qk();\n} else ${qkv}`)+`
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct QkPreconditionReport {
pub norm_chunks: usize, pub rotary_chunks: usize, pub maximum_chunk_elements: usize,
}
pub(super) fn precondition_projected_qk() {}
fn map_qk_token_chunks() {}`;
  const model='fn forward() {\nlet video = project_video();\nlet audio = project_audio();\nlet x = scatter(&video,&audio);\nbody(x);\n}';
  const nextModel=model.replace('body(x);','if self.stream_window.is_some() && self.device.is_cuda() {\ndrop(video);\ndrop(audio);\n}\nbody(x);');
  const operation='let attention = self.qk_norm_peak(rows, inner)?.max(self.rotary_peak(rows, inner)?).max(materialized); let ffn = plus(times(4, hidden)?, work)?; Ok(attention.max(ffn).max(times(8, hidden)?))';
  const denoise='work = work.max(self.cuda_working.ok_or_else(invalid)?.operation_peak_with_adapter(rows.packed,task.adapter.as_ref().map_or(0,|a|{a.forward_working_bytes_for(backend,rows,RowDomain::Packed)}),)?,); Ok(work)';
  const veda='let facts = self.cuda_working.ok_or_else(invalid)?; let dense = self.denoise_peak_bytes(task,backend,rows,window)?; let attention = dense.max(facts.bytes); Ok(attention)';
  const window=`struct Facts { bytes: u64 }\nimpl Facts {\nfn operation_peak() { self.operation_peak_with_adapter(rows,0) }\nfn operation_peak_with_adapter(self, rows: u64, adapter_work: u64) -> Result<u64> {${operation}}\nfn denoise_peak_bytes() {${denoise}}\nfn denoise_with_veda_bytes() {${veda}}\n}`;
  const operationFor='let precondition = if windowed {self.windowed_qk_precondition_peak(rows,inner)?} else {self.qk_norm_peak(rows,inner)?.max(self.rotary_peak(rows,inner)?)}; '+operation.replace('self.qk_norm_peak(rows, inner)?.max(self.rotary_peak(rows, inner)?)','precondition').replace('times(4, hidden)?','times(if windowed { 5 } else { 4 }, hidden)?').replace('Ok(attention','let block_norm = if windowed {plus(plus(times(9, hidden)?, times(8, rows)?)?, times(4, self.hidden)?,)?} else {times(8, hidden)?}; Ok(attention').replace('.max(times(8, hidden)?)','.max(block_norm)');
  const nextWindow=window.replace(`{${operation}}`,'{self.operation_peak_for(rows,adapter_work,false)}').replace(`{${denoise}}`,'{self.denoise_peak_bytes_impl(task,backend,rows,window,false)}').replace(`{${veda}}`,'{let facts = self.cuda_working.ok_or_else(invalid)?; let dense = self.denoise_peak_bytes(task,backend,rows,window)?; self.compose_veda_peak(task,backend,rows,window,veda,facts,dense)}')+`
pub const CUDA_QK_PRECONDITION_ELEMENTS: u64 = crate::veda::workspace::CUDA_DENSE_SCORE_ELEMENTS;
fn qk_precondition_token_block() {}
fn windowed_qk_precondition_peak() {}
#[cfg(test)]
fn windowed_operation_peak_with_adapter() {}
fn operation_peak_for(self, rows: u64, adapter_work: u64, windowed: bool) -> Result<u64> {${operationFor}}
fn bounded_cuda_denoise_with_veda_bytes() {}
#[allow(clippy::too_many_arguments)]
fn compose_veda_peak() {let attention = dense.max(facts.bytes); Ok(attention)}
fn bounded_cuda_denoise_peak_bytes() {}
fn denoise_peak_bytes_impl() {${denoise.replace('.operation_peak_with_adapter(','.operation_peak_for(').replace('RowDomain::Packed)}),)?,);','RowDomain::Packed)}),bounded_qk,)?);')}}`;
  const args='task,self.backend,Self::task_rows(task,rows),window as usize,';
  const legacy='let dense = self.with_adapter_denoise(task,rows,profile.denoise_peak_bytes(task,self.backend,rows,window)?,); Ok(peak.max(dense.max(profile.denoise_with_veda_bytes(task,self.backend,rows,window,veda,)?)))';
  const selected='let (dense,with_veda) = if selected_window.is_some() {(profile.bounded_cuda_denoise_peak_bytes(task,self.backend,rows,window,)?,profile.bounded_cuda_denoise_with_veda_bytes(task,self.backend,rows,window,veda,)?,)} else {(profile.denoise_peak_bytes(task,self.backend,rows,window)?,profile.denoise_with_veda_bytes(task,self.backend,rows,window,veda,)?,)}; let dense = self.with_adapter_denoise(task,rows,dense); Ok(peak.max(dense.max(with_veda)))';
  const pricing=`fn pricing() { profile.denoise_peak_bytes(${args})?; ${legacy} }`;
  const nextPricing=pricing.replace(`profile.denoise_peak_bytes(${args})?`,`if matches!(self.backend,NativeBackend::Candle {..}) {profile.bounded_cuda_denoise_peak_bytes(${args})?} else {profile.denoise_peak_bytes(${args})?}`).replace(legacy,'let selected_window = window; '+selected);
  return {before:[layers,model,window,pricing],after:[nextLayers,nextModel,nextWindow,nextPricing]};
}

function provenanceSourceFixture(){
  const beforeBlob='a'.repeat(40),afterBlob='b'.repeat(40),engineFrom=RETAINED_WORKSTATION.sourcePair.inference,engineTo='d'.repeat(40);
  const windowPath='crates/contracts/gen-core/src/h3_memory/window.rs';
  const beforeTsv=`# path\tgit_blob_sha1\tnormalized_marker_sha256\tsource_area\n${windowPath}\t${beforeBlob}\t66a666f0a533b9e30fc274055f28f6237445ef4d4978e4c30b948ea5ce4d6620\tarchitecture:crates/contracts/gen-core\nother/source.rs\t${'e'.repeat(40)}\t${'f'.repeat(64)}\tarchitecture:other`;
  const afterTsv=beforeTsv.replace(beforeBlob,afterBlob);
  const beforeAudit={inferenceRevision:engineFrom,artifacts:[],prospectiveDisclosures:[],provenanceScan:{revision:engineFrom,matchedFiles:2,populationSha256:populationSha256(parseProvenanceCandidates(beforeTsv))},portedSourceAreas:[{id:'unchanged-classification'}],crateCoverage:{revision:engineFrom,cratePrefixes:139},crateDispositions:[],includeSites:[]};
  const digest=record=>sha256(JSON.stringify(Object.fromEntries(['artifacts','prospectiveDisclosures','provenanceScan','portedSourceAreas','crateCoverage','crateDispositions','includeSites'].map(key=>[key,record[key]]))));
  beforeAudit.auditDigest=digest(beforeAudit);
  const afterAudit=structuredClone(beforeAudit);afterAudit.inferenceRevision=engineTo;afterAudit.provenanceScan.revision=engineTo;afterAudit.crateCoverage.revision=engineTo;afterAudit.provenanceScan.populationSha256=populationSha256(parseProvenanceCandidates(afterTsv));afterAudit.auditDigest=digest(afterAudit);
  return {beforeTsv,afterTsv,beforeAudit,afterAudit,engineFrom,engineTo,beforeBlob,afterBlob};
}

function sourceReviewFixture({engineExtra=[],consumerExtra=[],graphDrift=false,ciOutsideChange=false,manualOutsideChange=false,boundedQk=false,provenance=boundedQk,
  manualArtifactPattern='${{ env.SW_F329_OUTPUT }}/workstation-veda/original/**/*.json'}={}){
  const target={sceneworks:'c'.repeat(40),inference:'d'.repeat(40)},before='a'.repeat(40),after='b'.repeat(40);
  const enginePath='crates/media/candle-gen/vendor/candle-core/src/cuda_backend/device.rs';
  const metadata='config/inference-provider-closures.json',workflow='.github/workflows/server-candle-linux.yml';
  const ci=['scripts/select-linux-flash-cache.mjs','scripts/server-candle-linux-workflow.test.mjs',workflow];
  const boundedPaths=['crates/media/candle-gen/candle-gen-minimax-h3/src/dit/layers.rs','crates/media/candle-gen/candle-gen-minimax-h3/src/dit/model.rs','crates/contracts/gen-core/src/h3_memory/window.rs','crates/contracts/gen-core-h3/src/h3_memory.rs'];
  const boundedSources=boundedQkSourceFixture();
  const provenanceSources=provenanceSourceFixture(),provenancePaths=['config/inference-provenance-candidates.tsv','config/inference-third-party-source.json'];
  const audit={kind:'sw-f329-uncapped-workstation-source-equivalence-v1',originalSourcePair:structuredClone(RETAINED_WORKSTATION.sourcePair),targetSourcePair:target,
    uncappedAllocatorPreserved:true,reviewUrl:'https://github.com/SceneWorks/inference/pull/9999',
    engineChanges:[{path:enginePath,before,after}],consumerPinMetadata:[{path:metadata,before,after}],
    consumerCiChanges:ci.map(file=>({path:file,before:file===workflow?before:null,after}))};
  if(boundedQk){
    audit.engineChanges.push(...boundedPaths.map(file=>({path:file,before,after})));
    for(const field of ['boundedQkResidentPreserved','boundedQkRefinerPreserved','boundedQkResidentPricingPreserved'])audit[field]=true;
  }
  if(provenance)audit.consumerPinMetadata.push(...provenancePaths.map(file=>({path:file,before,after})));
  const command=async(program,args,{cwd})=>{
    assert.equal(program,'git');
    if(args[0]==='merge-base')return {stdout:''};
    if(args[0]==='diff')return {stdout:(cwd==='engine'?[enginePath,...(boundedQk?boundedPaths:[]),...engineExtra]:['Cargo.toml','Cargo.lock',metadata,...(provenance?provenancePaths:[]),'scripts/h3-cuda-memory-terminal.mjs','.github/workflows/h3-cuda-memory-terminal.yml',...ci,...consumerExtra]).join('\n')};
    if(args[0]==='ls-tree'){
      const file=args.at(-1),old=args[1]===RETAINED_WORKSTATION.sourcePair.inference||args[1]===RETAINED_WORKSTATION.sourcePair.sceneworks;
      return {stdout:old&&ci.includes(file)&&file!==workflow?'':`100644 blob ${old?before:after}\t${file}`};
    }
    assert.equal(args[0],'show');const [revision,file]=args[1].split(':');
    if(cwd==='engine'&&boundedPaths.includes(file))return {stdout:boundedSources[revision===RETAINED_WORKSTATION.sourcePair.inference?'before':'after'][boundedPaths.indexOf(file)]};
    const old=revision===RETAINED_WORKSTATION.sourcePair.sceneworks;
    if(provenance&&file===provenancePaths[0])return {stdout:provenanceSources[old?'beforeTsv':'afterTsv']};
    if(provenance&&file===provenancePaths[1])return {stdout:JSON.stringify(provenanceSources[old?'beforeAudit':'afterAudit'])};
    if(file===workflow)return {stdout:`name: fixture\njobs:\n  build-candle:\n    steps: ${old?'old':'new'}\n  other-job:\n    unchanged: ${!old&&ciOutsideChange?'false':'true'}\n`};
    if(file==='.github/workflows/h3-cuda-memory-terminal.yml')return {stdout:`path: sceneworks\n          fetch-depth: ${old?'1':'128'}\nCUDA_COMPUTE_CAP: ${!old&&manualOutsideChange?'80':'120'}\n            \${{ env.SW_F329_OUTPUT }}/workstation-veda/*.json\n${!old&&manualArtifactPattern!==null?'            '+manualArtifactPattern+'\n':''}          if-no-files-found: error\n`};
    return {stdout:`inference = "${old?RETAINED_WORKSTATION.sourcePair.inference:target.inference}"${!old&&graphDrift?'\nother = "changed"':''}`};
  };
  return {target,audit,command,verify:(review=audit)=>retainedWorkstationSources('consumer','engine',target,review,command)};
}

test('only the complete immutable uncapped workstation control can be retained',()=>{
  validateRetainedWorkstationRecord(retainedRecord(),anchorRequest(original(),42));
  const mutations=[r=>r.id='24gib-veda-dense',r=>r.passed=false,r=>r.job.id='other-job',r=>r.job.status='failed',r=>r.job.assignedGpu='0',
    r=>r.sourcePair.inference='f'.repeat(40),r=>r.settings.seed=43,r=>r.generationMetrics.steps=4,r=>r.generationMetrics.quantBits=8,
    r=>r.media[0].sha256='0'.repeat(64),r=>r.media[0].decoded.streams[0].nb_read_frames=124,r=>r.media[0].decoded.streams[1].channels=1,
    r=>r.media[0].asset.recipe.rawAdapterSettings.effectiveSteps=4,r=>r.content.frames=124,r=>r.content.spatialStd.min=0,
    r=>r.observations.devicePeakBytes=96*GIB,r=>r.events.push({cap_bytes:21*GIB}),r=>r.generationMetrics.vedaExecution.sparseCalls=0,
    r=>r.generationMetrics.vedaExecution.fallbackReason='forced for validation',r=>r.events[0].strategy='BoundedTransformerResidency',
    r=>r.events[1].transformer_window='Some(1)',r=>r.events.push({event:'h3_transformer_window_materialized'}),r=>r.shutdownError='partial cleanup'];
  for(const mutate of mutations){const bad=retainedRecord();mutate(bad);assert.throws(()=>validateRetainedWorkstationRecord(bad,anchorRequest(original(),42)));}
  const p=plan();p.retainedWorkstation={campaignDirectory:'D:\\accepted',sourceAuditPath:'D:\\review.json',sourceAuditSha256:hash,runUrl:RETAINED_WORKSTATION.runUrl};
  p.poolBytes=21*GIB;p.externalReserveBytes=3*GIB;validatePlan(p);
  for(const field of ['sourceAuditPath','sourceAuditSha256','runUrl']){const bad=structuredClone(p);delete bad.retainedWorkstation[field];assert.throws(()=>validatePlan(bad));}
});

test('retention source review allows only exact reviewed cap, pin and standard-build CI objects',async()=>{
  const fixture=sourceReviewFixture(),verified=await fixture.verify();assert.deepEqual(verified.targetSourcePair,fixture.target);
  const mutations=[a=>a.uncappedAllocatorPreserved=false,a=>a.targetSourcePair.sceneworks='f'.repeat(40),a=>a.reviewUrl='https://example.com/review',
    a=>a.engineChanges[0].after='f'.repeat(40),a=>a.engineChanges[0].before='f'.repeat(40),a=>a.engineChanges.push({...a.engineChanges[0]}),
    a=>a.engineChanges[0].path='crates/media/candle-gen/candle-gen-minimax-h3/src/nn.rs',a=>a.consumerPinMetadata[0].path='config/manifests/builtin.models.jsonc',
    a=>a.consumerCiChanges=[]];
  for(const mutate of mutations){const bad=structuredClone(fixture.audit);mutate(bad);await assert.rejects(fixture.verify(bad));}
  for(const options of [{engineExtra:['crates/kernels/candle-quant-kernels/src/veda_int8.cu']},
    {consumerExtra:['crates/sceneworks-worker/src/video_jobs/wan.rs']},{graphDrift:true},{ciOutsideChange:true},{manualOutsideChange:true}])await assert.rejects(sourceReviewFixture(options).verify());
});

test('bounded Q/K retention mechanically preserves Resident/refiner math, ownership and legacy pricing',()=>{
  const fixture=boundedQkSourceFixture();
  assert.equal(validateBoundedQkResidentSources(fixture.before,fixture.after).paths.length,4);
  const mutations=[
    [0,'rms_weighted(x, weight, eps)','different_norm(x, weight, eps)'],
    [0,'rope.is_some() && x.device().is_cuda()','x.device().is_cuda()'],
    [0,'crate::block_stream::current().is_some()','true'],
    [0,'finish(q,k,v);','precondition_projected_qk(q,k,v);'],
    [0,'Clone, Copy, Debug, Default','Clone, Debug, Default'],
    [1,'if self.stream_window.is_some() && self.device.is_cuda()','if self.device.is_cuda()'],
    [1,'body(x);','body(changed(x));'],
    [2,'self.operation_peak_for(rows,adapter_work,false)','self.operation_peak_for(rows,adapter_work,true)'],
    [2,'self.denoise_peak_bytes_impl(task,backend,rows,window,false)','self.denoise_peak_bytes_impl(task,backend,rows,window,true)'],
    [2,'self.operation_peak_with_adapter(rows,0)','self.operation_peak_with_adapter(rows,1)'],
    [2,'rows: u64, adapter_work: u64','adapter_work: u64, rows: u64'],
    [2,'struct Facts { bytes: u64 }','struct Facts { bytes: u32 }'],
    [2,'crate::veda::workspace::CUDA_DENSE_SCORE_ELEMENTS;','u64::MAX;'],
    [2,'dense.max(facts.bytes)','dense.min(facts.bytes)'],
    [3,'selected_window.is_some()','true'],
    [3,'profile.denoise_peak_bytes(task,self.backend,rows,window)?','profile.bounded_cuda_denoise_peak_bytes(task,self.backend,rows,window)?'],
  ];
  for(const [index,from,to] of mutations){
    const after=[...fixture.after];assert.ok(after[index].includes(from));after[index]=after[index].replace(from,to);
    assert.throws(()=>validateBoundedQkResidentSources(fixture.before,after),undefined,`${index}: ${from}`);
  }
  for(const index of [0,1,2,3]){
    const after=[...fixture.after];after[index]+='\nfn unreviewed_production_change() {}';
    assert.throws(()=>validateBoundedQkResidentSources(fixture.before,after));
  }
  const testOnly=[...fixture.after];testOnly[1]+='\n#[cfg(test)]\nmod proof { fn extra_control() {} }';
  validateBoundedQkResidentSources(fixture.before,testOnly);
});

test('bounded Q/K source audit requires all exact reviewed objects and every explicit Resident scope assertion',async()=>{
  const fixture=sourceReviewFixture({boundedQk:true}),checked=await fixture.verify();assert.equal(checked.boundedQk.paths.length,4);
  for(const field of ['boundedQkResidentPreserved','boundedQkRefinerPreserved','boundedQkResidentPricingPreserved']){
    for(const value of [undefined,false,'true']){
      const bad=structuredClone(fixture.audit);if(value===undefined)delete bad[field];else bad[field]=value;
      await assert.rejects(fixture.verify(bad),/explicit reviewed/);
    }
  }
  for(const mutate of [a=>a.engineChanges.pop(),a=>a.engineChanges[1].before='e'.repeat(40),a=>a.engineChanges[1].after='e'.repeat(40),a=>a.engineChanges.push({path:'crates/media/candle-gen/candle-gen-minimax-h3/src/nn.rs',before:'a'.repeat(40),after:'b'.repeat(40)})]){
    const bad=structuredClone(fixture.audit);mutate(bad);await assert.rejects(fixture.verify(bad));
  }
});

test('selected-window FFN/norm allowances keep exact legacy false arms and cannot escape their reviewed flag',()=>{
  const fixture=boundedQkSourceFixture();validateBoundedQkResidentSources(fixture.before,fixture.after);
  for(const [from,to] of [
    ['if windowed { 5 } else { 4 }','if windowed { 5 } else { 3 }'],
    ['if windowed { 5 } else { 4 }','if true { 5 } else { 4 }'],
    ['if windowed { 5 } else { 4 }','if windowed { 6 } else { 4 }'],
    ['let block_norm = if windowed','let block_norm = if true'],
    ['else {times(8, hidden)?}','else {times(7, hidden)?}'],
    ['times(9, hidden)?','times(8, hidden)?'],
    ['times(8, rows)?','times(4, rows)?'],
    ['times(4, self.hidden)?','times(2, self.hidden)?'],
    ['.max(block_norm)','.max(times(9, hidden)?)'],
  ]){
    const after=[...fixture.after];assert.ok(after[2].includes(from));after[2]=after[2].replace(from,to);
    assert.throws(()=>validateBoundedQkResidentSources(fixture.before,after),undefined,from);
  }
});

test('generated provenance permits only the actual window blob cell and its exact derived pin/population/audit restamp',()=>{
  const fixture=provenanceSourceFixture(),verified=validateRetainedWindowProvenance(fixture);
  assert.equal(verified.afterBlob,fixture.afterBlob);assert.equal(verified.matchedFiles,2);
  assert.equal(verified.populationSha256,fixture.afterAudit.provenanceScan.populationSha256);
  for(const mutate of [
    f=>f.afterTsv=f.afterTsv.replace('architecture:crates/contracts/gen-core','architecture:changed'),
    f=>f.afterTsv=f.afterTsv.replace('66a666f0','76a666f0'),
    f=>f.beforeTsv=f.beforeTsv.replace('66a666f0','76a666f0'),
    f=>f.afterTsv=f.afterTsv.replace('e'.repeat(40),'c'.repeat(40)),
    f=>f.afterTsv=f.afterTsv.replace(f.afterBlob,'c'.repeat(40)),
    f=>f.beforeBlob='c'.repeat(40),
    f=>f.afterBlob='c'.repeat(40),
    f=>f.afterTsv+='\n'+f.afterTsv.split('\n')[1],
    f=>f.afterAudit.provenanceScan.populationSha256='0'.repeat(64),
    f=>f.afterAudit.auditDigest='0'.repeat(64),
    f=>f.beforeAudit.auditDigest='0'.repeat(64),
    f=>f.afterAudit.provenanceScan.matchedFiles++,
    f=>f.afterAudit.crateCoverage.cratePrefixes++,
    f=>f.afterAudit.portedSourceAreas[0].id='different-classification',
    f=>f.afterAudit.inferenceRevision='c'.repeat(40),
    f=>f.afterAudit.provenanceScan.revision='c'.repeat(40),
  ]){const bad=structuredClone(fixture);mutate(bad);assert.throws(()=>validateRetainedWindowProvenance(bad));}
});

test('complete source audit binds the generated provenance objects and the reviewed bounded engine path',async()=>{
  const fixture=sourceReviewFixture({boundedQk:true,provenance:true}),verified=await fixture.verify();
  assert.equal(verified.generatedProvenance.windowPath,'crates/contracts/gen-core/src/h3_memory/window.rs');
  for(const mutate of [a=>a.consumerPinMetadata.pop(),a=>a.consumerPinMetadata[1].after='c'.repeat(40),a=>a.consumerPinMetadata[1].path='config/unreviewed.tsv']){
    const bad=structuredClone(fixture.audit);mutate(bad);await assert.rejects(fixture.verify(bad));
  }
  await assert.rejects(sourceReviewFixture({provenance:true}).verify(),/reviewed bounded Q\/K repair/);
  await assert.rejects(sourceReviewFixture({boundedQk:true,provenance:false}).verify(),/both exact generated provenance object bindings/);
});

test('manual retention mapping requires the exact nested original-receipt artifact pattern and refuses broader workflow changes',async()=>{
  await sourceReviewFixture().verify();
  for(const options of [{manualArtifactPattern:null},
    {manualArtifactPattern:'${{ env.SW_F329_OUTPUT }}/workstation-veda/original/**'},
    {manualArtifactPattern:'${{ env.SW_F329_OUTPUT }}/**/*.json'},
    {manualArtifactPattern:'${{ env.SW_F329_OUTPUT }}/workstation-veda/original/**/*.json\n            ${{ env.SW_F329_OUTPUT }}/extra/*.json'},
    {manualArtifactPattern:'${{ env.SW_F329_OUTPUT }}/workstation-veda/original/**/*.json\n            ${{ env.SW_F329_OUTPUT }}/workstation-veda/original/**/*.json'},
    {manualOutsideChange:true}]){
    await assert.rejects(sourceReviewFixture(options).verify(),/manual source\/configuration changed/);
  }
});

test('retained parity needs the verified narrow source mapping and still compares all native AV',async t=>{
  const fixture=sourceReviewFixture(),sourceEquivalence=await fixture.verify(),resident=retainedRecord(),streamed=parityRecords()[1];
  streamed.sourcePair=fixture.target;streamed.generationMetrics.vedaExecution=structuredClone(resident.generationMetrics.vedaExecution);
  assert.throws(()=>validateVedaParityPair([resident,streamed]),/source pair changed/);
  assert.throws(()=>validateVedaParityPair([resident,streamed],{sourceEquivalence:JSON.parse(JSON.stringify(sourceEquivalence))}),/actual Git objects/);
  assert.equal(validateVedaParityPair([resident,streamed],{sourceEquivalence}).resident.sourcePair.inference,RETAINED_WORKSTATION.sourcePair.inference);
  const dir=await directory(t);let calls=0;
  await mkdirForTest(path.join(dir,'workstation-veda'));await mkdirForTest(path.join(dir,'24gib-veda-sparse'));
  // The real comparator's hash guard stays active; a fabricated media payload cannot reach decoding.
  await writeFile(path.join(dir,'workstation-veda','unit_fake.mp4'),'unit_fake');await writeFile(path.join(dir,'24gib-veda-sparse','unit_fake.mp4'),'unit_fake');
  await assert.rejects(compareVedaAnchors([resident,streamed],dir,{sourceEquivalence,compare:async()=>{calls++;}}),/bound parity media changed/);
  assert.equal(calls,0);
});

test('retention independently binds all weight roles, native configuration, physical UUID and driver',()=>{
  const p=plan(),files=Object.entries(p.sources).map(([role,source])=>({role,repo:({tier:'SceneWorks/minimax-h3-mlx',base:'MiniMaxAI/MiniMax-H3',predictor:'Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview',turbo:'lightx2v/Minimax-h3-Turbo'})[role],revision:source.revision,relative:'unit_fake.safetensors',bytes:1,sha256:hash}));
  const weights={kind:'sw-f329-staged-models',sourcePair:structuredClone(RETAINED_WORKSTATION.sourcePair),files};retainedWorkstationWeights(weights,weights,p.sources);
  for(const mutate of [w=>w.files.pop(),w=>w.files.push(w.files[0]),w=>w.files[0].sha256='f'.repeat(64),w=>w.files[0].revision='f'.repeat(40)]){const bad=structuredClone(weights);mutate(bad);assert.throws(()=>retainedWorkstationWeights(weights,bad,p.sources));}
  const hardware={driver:'unit_fake_driver',cuda:{cudaDeviceOrder:'PCI_BUS_ID',cudaVisibleDevices:'GPU-e4b79931-7be6-f216-460a-f5405cfafffe',selectedPhysicalDevice:{nvmlIndex:1,uuid:'GPU-e4b79931-7be6-f216-460a-f5405cfafffe',name:'unit_fake'}}};retainedWorkstationHardware(hardware,hardware);
  for(const mutate of [h=>h.driver='different',h=>h.cuda.selectedPhysicalDevice.nvmlIndex=0,h=>h.cuda.cudaVisibleDevices='0']){const bad=structuredClone(hardware);mutate(bad);assert.throws(()=>retainedWorkstationHardware(hardware,bad));}
  const build={backend:'cuda',sha256:RETAINED_WORKSTATION.binarySha256,command:['cargo','build','--profile','release','--manifest-path','old/Cargo.toml','--features','sceneworks-rust-api/h3-memory-proof'],configuration:{proof_kind:'campaign_native',flags:{RUSTFLAGS:null}}};build.configuration.command=[...build.command];
  const current=structuredClone(build);current.command[5]='new/Cargo.toml';current.configuration.command=[...current.command];current.sha256=hash;retainedWorkstationBuild(build,current);
  for(const mutate of [b=>b.command.push('--no-default-features'),b=>b.configuration.flags.RUSTFLAGS='-C target-cpu=native',b=>b.backend='cpu',b=>b.configuration.proof_kind='unit_fake']){const bad=structuredClone(current);mutate(bad);assert.throws(()=>retainedWorkstationBuild(build,bad));}
});

test('allocator primitive requires enforcement and an honest consistent observed recovery count, including zero',()=>{
  const evidence={logicalOrdinal:0,overCapRefused:true,bothConstructors:true,cublasStreamResetRebound:true,successfulRecoveryCount:1,fragmentationRecoveryExercised:true,heldBytes:469762048,usedHighBytes:536870912,reservedHighBytes:536870912,capBytes:536870912};
  const log=value=>'CUDA_PROOF_POOL_EVIDENCE '+JSON.stringify(value)+'\ntest result: ok. 1 passed; 0 failed; 0 ignored';
  validatePrimitiveProofLog(log(evidence));
  assert.equal(validatePrimitiveProofLog(log({...evidence,successfulRecoveryCount:0,fragmentationRecoveryExercised:false})).successfulRecoveryCount,0);
  for(const field of ['overCapRefused','bothConstructors','cublasStreamResetRebound','successfulRecoveryCount','fragmentationRecoveryExercised']){const bad={...evidence};delete bad[field];assert.throws(()=>validatePrimitiveProofLog(log(bad)));}
  for(const count of [-1,0.5,true,'1',null,Number.MAX_SAFE_INTEGER+1])assert.throws(()=>validatePrimitiveProofLog(log({...evidence,successfulRecoveryCount:count})));
  for(const [successfulRecoveryCount,fragmentationRecoveryExercised] of [[0,true],[1,false],[0,0],[1,1],[0,'false']])assert.throws(()=>validatePrimitiveProofLog(log({...evidence,successfulRecoveryCount,fragmentationRecoveryExercised})));
  for(const change of [{logicalOrdinal:1},{capBytes:512},{heldBytes:1},{heldBytes:536870913},{usedHighBytes:469762047},{reservedHighBytes:536870913},{capBytes:'536870912'},{heldBytes:true},{usedHighBytes:NaN},{reservedHighBytes:null}])assert.throws(()=>validatePrimitiveProofLog(log({...evidence,...change})));
});
