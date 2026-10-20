// DECLARED SYNTHETIC CPU fixtures: no adapter process, hardware probe, model or GPU exists.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { initializeLedger, reserveInvocation, auditLedger, planHash } from './flux3-image-klein-terminal-budget.mjs';
import { launchFlux3Resource } from './flux3-resource-launcher.mjs';
import { flux3AnchorRequestDigest, flux3CaptureDigest, verifyNativeResourceCapture } from './lib/flux3-resource-contract.mjs';
const digest='d'.repeat(64), adapterPid=12345;
async function fixture(t) {
  const root=await mkdtemp(path.join(tmpdir(),'flux3-resource-launch-DECLARED-CPU-fixture-'));t.after(()=>rm(root,{recursive:true,force:true}));
  const invocation={backend:'candle',caseId:'candle.base.resource-native.bf16',kind:'resource_anchor',invocationId:'candle.base.resource-native.bf16.generation-1',
    modelId:'flux3_klein_base',tier:'bf16',requestSha256:flux3AnchorRequestDigest('candle','flux3_klein_base','bf16'),requestTemplate:{count:1},nativeGenerationCalls:1,warmups:0,repeats:0};
  const approval={schemaVersion:1,kind:'flux3-terminal-plan-approval',approvalId:'DECLARED-SYNTHETIC',userPlanApprovalId:'NOT_USER_AUTHORIZATION',approvedBy:'CPU fixture',
    sourceSha:'a'.repeat(40),inferenceMainSha:'b'.repeat(40),ledgerId:'DECLARED-CPU-resource-ledger',ledgerRoots:{candle:path.join(root,'candle'),mlx:path.join(root,'mlx')},
    inputSpecSha256:{candle:digest,mlx:digest},plan:{kind:'flux3-installed-product-representative-plan',generationInvocations:[invocation],trainingLifecycles:[],
      resourceCases:[{backend:'candle',caseId:invocation.caseId,status:'READY',parentNativeBoundaryVerificationId:'DECLARED-SYNTHETIC',nativeBoundaryReceiptSha256:digest}],
      budget:{globalHardCeiling:48,totalGenerationInvocations:1,byBackend:{candle:1,mlx:0},unallocatedNotAuthorized:47,automaticWarmups:0,automaticRepeats:0,automaticRetries:0,trainingLifecycles:0,stepsPerLifecycle:4}}};
  await initializeLedger({approval,backend:'candle',ledgerRoot:approval.ledgerRoots.candle});
  const admission={actualRun:true,fixtureOnly:true,backend:'candle',approvedPlan:approval,approvedPlanSha256:planHash(approval),ledgerId:approval.ledgerId,ledgerRoot:approval.ledgerRoots.candle,
    sourceSha:approval.sourceSha,inferenceMainSha:approval.inferenceMainSha,binary:path.join(root,'DECLARED-NONEXISTENT-adapter.exe'),binarySha256:digest,binaryBuildReceipt:{sha256:digest},
    outputRoot:path.join(root,'outputs'),hfHome:path.join(root,'EMPTY-UNUSED-cache'),timeoutSeconds:60,gpuUuid:'GPU-declared-fixture-1'};
  const request={action:'run',planned:{backend:'candle',target:{provider:'flux3_klein_base',tier:'bf16'}}};
  const anchorFragment={status:'gated',quality:{result:'not_run'},fixtureOnly:true,observedMemory:{conditioning:{activeBytes:1,allocatorBytes:1,reclaimableBytes:0}}};
  let calls=0, generated=0, latest, guarded=false;
  const execute=async (_binary,actualRequest,options)=>{
    assert.equal(guarded,true,'hardware guard must precede owned spawn/load');
    calls++;assert.equal(actualRequest.action,'run');assert.equal(actualRequest.resourceEnvelope,true);
    assert.equal(options.env.CUDA_VISIBLE_DEVICES,'1');assert.equal(options.env.HF_HUB_OFFLINE,'1');
    assert.equal(options.env.HF_TOKEN,undefined);assert.equal(options.env.SCENEWORKS_STALE_SECRET,undefined);
    const input=JSON.parse(options.env.SCENEWORKS_FLUX3_INVOCATION);
    assert.equal(JSON.parse(await readFile(input.admissionFile,'utf8')).runDeadline,admission.runDeadline ?? 60000+1000);
    const event=await reserveInvocation({approval,...input}); // REAL atomic ledger BEFORE scripted generate.
    generated++;
    const hardwareObservation={kind:'flux3-resource-producer-hardware-observation',backend:'candle',adapterPid,
      caseId:invocation.caseId,invocationId:invocation.invocationId,requestSha256:invocation.requestSha256,sourceSha:admission.sourceSha,
      inferenceMainSha:admission.inferenceMainSha,binarySha256:digest,buildReceiptSha256:digest,approvedPlanSha256:admission.approvedPlanSha256,
      observedAt:'2026-10-20T00:00:00.000Z',adapterProcess:{pid:adapterPid,startedAt:'DECLARED-SYNTHETIC',executable:admission.binary},
      proof:{physicalGpuId:'1',gpuUuid:admission.gpuUuid,logicalCudaOrdinal:0,ownComputePids:[adapterPid]}};
    const spec={schemaVersion:1,provenanceKind:'actual_borrowed_load_spec_before_catalog_load',backend:'candle',provider:invocation.modelId,tier:'bf16',
      weights:{kind:'Dir',path:'DECLARED-SYNTHETIC'},components:{},adapters:[],precision:'Bf16',nativeDebugFormatVersion:1,nativeDebugSha256:digest};
    spec.sha256=flux3CaptureDigest('flux3_actual_load_spec_v1\0',spec);
    latest={schemaVersion:1,evidenceKind:'flux3_native_resource_capture_envelope_v1',anchorEligible:false,captureId:invocation.invocationId,caseId:invocation.caseId,
      requestSha256:invocation.requestSha256,anchorFragment:structuredClone(anchorFragment),anchorFragmentSha256:flux3CaptureDigest('flux3_canonical_anchor_fragment_v1\0',anchorFragment),
      nativeOutput:{schemaVersion:1,kind:'Images',width:1024,height:1024,format:'RGB8',representation:'interleaved_host_u8',bytes:1024*1024*3,
        sha256:digest,identityDomain:'flux3_native_output_v1',reservation:{...event,hardwareObservation},
        actualGenerationRequest:{prompt:'a portrait of a person in a sunlit studio, editorial photograph',negativePrompt:null,width:1024,height:1024,count:1,
          seed:26243,steps:50,guidance:3.5,trueCfg:null,sampler:null,scheduler:null,schedulerShift:null,enhancePrompt:false,
          layout:null,layoutBindings:[],referenceCarriers:[],conditioningCount:0}},actualLoadSpec:spec,loadedGenerator:{provider:invocation.modelId,backend:'candle'}};
    latest.captureBindingSha256=flux3CaptureDigest('flux3_native_capture_binding_v1\0',latest);
    return {adapterPid,stdout:JSON.stringify(latest),stderr:'DECLARED CPU FIXTURE ONLY',code:0,signal:null};
  };
  const dependencies={validate:async a=>assert.equal(a.fixtureOnly,true),guard:async(a,pids)=>{assert.equal(a.fixtureOnly,true);assert.deepEqual(pids,[]);guarded=true;},execute,now:()=>1000,env:{HF_TOKEN:'fixture-only-placeholder',SCENEWORKS_STALE_SECRET:'fixture-only-placeholder'}};
  return {admission,request,invocation,anchorFragment,dependencies,calls:()=>calls,generated:()=>generated,latest:()=>latest};
}
test('one explicit resource child, real inner permit, unchanged canonical fragment extracted only after full envelope verification',async t=>{
  const f=await fixture(t);const result=await launchFlux3Resource({admission:f.admission,request:f.request,invocationId:f.invocation.invocationId},f.dependencies);
  assert.equal(f.calls(),1);assert.equal(f.generated(),1);assert.equal(result.adapterPid,adapterPid);
  assert.deepEqual(JSON.parse(await readFile(result.resultFile,'utf8')),f.anchorFragment);
  await assert.rejects(launchFlux3Resource({admission:f.admission,request:f.request,invocationId:f.invocation.invocationId},f.dependencies),/consumed/);
  assert.equal(f.generated(),1,'failed second attempt cannot generate or refund');
});
test('launcher has no implicit actual admission, batch ladder, arbitrary loader environment, or automatic retry',async t=>{
  const f=await fixture(t);
  for(const config of [{admission:{...f.admission,actualRun:false},request:f.request},
    {admission:f.admission,request:{...f.request,action:'run_batch'}},
    {admission:f.admission,request:f.request,resourceEnvironment:{HF_TOKEN:'fixture-only-placeholder'}}]) {
    await assert.rejects(launchFlux3Resource({...config,invocationId:f.invocation.invocationId},f.dependencies));
  }
  assert.equal(f.calls(),0);
  await assert.rejects(launchFlux3Resource({admission:f.admission,request:f.request,invocationId:f.invocation.invocationId},
    {...f.dependencies,guard:async()=>{throw Error('DECLARED foreign GPU/host fixture');}}),/foreign GPU/);
  assert.equal(f.calls(),0,'failed hardware guard cannot spawn/load the adapter');
  await assert.rejects(launchFlux3Resource({admission:f.admission,request:f.request,invocationId:f.invocation.invocationId},
    {...f.dependencies,execute:async()=>{throw Error('declared adapter failure');}}),/adapter failure/);
  assert.equal(f.generated(),0);
});
test('capture verification rejects tampered fragment/output/request/permit/hardware PID/GPU/build and supplemental-as-anchor',async t=>{
  const f=await fixture(t);await launchFlux3Resource({admission:f.admission,request:f.request,invocationId:f.invocation.invocationId},f.dependencies);
  const ledger=await auditLedger({approval:f.admission.approvedPlan,backend:'candle',ledgerRoot:f.admission.ledgerRoot});
  const context={admission:f.admission,invocation:f.invocation,adapterPid,ledgerEvent:ledger.events[0]},original=f.latest();
  for(const mutate of [v=>v.anchorFragment.quality.result='passed',v=>v.nativeOutput.sha256='e'.repeat(64),v=>v.requestSha256=digest,
    v=>v.nativeOutput.reservation.invocationId='different',v=>v.nativeOutput.actualGenerationRequest.steps=4,v=>v.nativeOutput.reservation.hardwareObservation.adapterPid++,
    v=>v.nativeOutput.reservation.hardwareObservation.binarySha256='e'.repeat(64),v=>v.nativeOutput.reservation.hardwareObservation.proof.gpuUuid='GPU-wrong',
    v=>v.evidenceKind='flux3_supplemental_resource_observation_v1']) {
    const changed=structuredClone(original);mutate(changed);assert.throws(()=>verifyNativeResourceCapture(changed,context),/BLOCKED/);
    // Even a re-hashed mutated envelope cannot substitute another bound request/producer/permit.
    if(changed.requestSha256!==original.requestSha256 || changed.nativeOutput.actualGenerationRequest.steps!==50 || changed.nativeOutput.reservation.hardwareObservation.adapterPid!==adapterPid
      || changed.nativeOutput.reservation.hardwareObservation.binarySha256!==digest || changed.nativeOutput.reservation.hardwareObservation.proof.gpuUuid!==f.admission.gpuUuid
      || changed.nativeOutput.reservation.invocationId!==f.invocation.invocationId || changed.evidenceKind!==original.evidenceKind) {
      delete changed.captureBindingSha256;changed.captureBindingSha256=flux3CaptureDigest('flux3_native_capture_binding_v1\0',changed);
      assert.throws(()=>verifyNativeResourceCapture(changed,context),/BLOCKED/);
    }
  }
  assert.deepEqual(verifyNativeResourceCapture(original,context).anchorFragment,f.anchorFragment);
  const {validateRecord}=await import('./memory-calibration-harness.mjs');assert.throws(()=>validateRecord({...original,id:original.caseId,logicalCaseId:original.caseId}),/invalid status/i);
});
