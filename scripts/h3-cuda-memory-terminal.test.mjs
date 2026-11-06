import assert from 'node:assert/strict';
import test from 'node:test';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { readFile } from 'node:fs/promises';
import { anchorRequest, validatePlan, validatePoolEvidence, validateMaterializingProofLog } from './h3-cuda-memory-terminal.mjs';
import { privateEnvironment } from './h3-finetune-terminal-runtime.mjs';

const GIB=1024**3;
function plan(){return {version:1,stationary_pair_authorized:true,sceneworks:'a'.repeat(40),inference:'b'.repeat(40),requestSha256:'c'.repeat(64),requestPath:'D:\\fixture.json',licenseAcknowledgmentsPath:'D:\\licenses.json',expectedRunner:'cuda-windows-2',seed:42,poolBytes:22*GIB,externalReserveBytes:2*GIB,hostBudgetBytes:64*GIB,workstationDeviceBytes:95*GIB,timeoutSeconds:7200,sources:Object.fromEntries([
  ['tier','SceneWorks--minimax-h3-mlx','137ce668c55a20bc0935fd1cf2a3de8448abb7f4'],
  ['base','MiniMaxAI--MiniMax-H3','6818f6c32d12b210915e44ad56a4228c2608f160'],
  ['predictor','Veda-Sparse--Minimax-H3-T2VA-Veda-8NFE-600Step-Preview','9a1fd3a41b4a754a7886e64e82edbddf599fd1bd'],
  ['turbo','lightx2v--Minimax-h3-Turbo','5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1'],
].map(([role,repo,revision])=>[role,{root:`E:\\huggingface\\hub\\models--${repo}\\snapshots\\${revision}`,revision}])),primitiveProof:{directory:'D:\\primitive',runUrl:'https://github.com/SceneWorks/inference/actions/runs/1'},materializingProof:{logPath:'D:\\fixture-proof.log',revision:'b'.repeat(40),runUrl:'https://github.com/SceneWorks/inference/actions/runs/2'}};}
function original(){return {model:'minimax_h3',mode:'text_to_video',prompt:'Fixture request; no generation.',width:1344,height:768,fps:24,duration:14.375,quality:'balanced',fitMode:'crop',seed:null,loras:[{id:'minimax_h3_turbo_8step',weight:1}],advanced:{veda:{enabled:true,fullAttentionLayers:'',fullAttentionSteps:'',generated:'90%',reference:'90%'}}};}
test('exact original geometry, Turbo8step and Veda controls survive both anchors',()=>{
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
  assert.throws(()=>validatePlan({...plan(),additionalSparse24:true}));
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
test('manual wrapper keeps CUDA memory acceptance separate from ordinary CI and runs early startup checks',async()=>{
  const caller=await readFile(new URL('../.github/workflows/server-candle-linux.yml',import.meta.url),'utf8');
  const workflow=await readFile(new URL('../.github/workflows/h3-cuda-memory-terminal.yml',import.meta.url),'utf8');
  assert.match(caller,/h3-cuda-memory:\n\s+if: \$\{\{ github\.event_name == 'workflow_dispatch' && inputs\.starvector_terminal_operation == 'h3-cuda-memory' \}\}\n\s+uses: \.\/\.github\/workflows\/h3-cuda-memory-terminal\.yml/);
  assert.match(workflow,/workflow_call:/);assert.doesNotMatch(workflow,/\n\s+(?:pull_request|push|schedule|merge_group):/);
  assert.match(workflow,/github\.event_name == 'workflow_dispatch'.*stationary_pair_authorized == true/);
  assert.match(workflow,/group: windows-candle-gpu-real-weights/);assert.match(workflow,/nvidia-smi -i 1 --query-gpu=uuid/);
  assert.ok(workflow.indexOf('Check script entry point')<workflow.indexOf('Build and execute the two'));
});
