import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {admission,proofCases,validateReceipt,requiredFixtures,validateProvisioning} from './cv-provider-validation.mjs';
const source='a'.repeat(40), inference='b'.repeat(40);
const mac={CV_SOURCE_SHA:source,CV_INFERENCE_SHA:inference,DISPATCH_SHA:source,CV_BACKEND:'mlx',CV_BUILD_ONLY:'false',CV_MAC_GO:'true',CV_OUTPUT:'/tmp/cv-proof',CARGO_TARGET_DIR:'/tmp/cv-target',CV_FIXTURE_DIR:'/tmp/fixtures',ORT_DYLIB_PATH:'/tmp/libonnxruntime.dylib'};
const cuda={...mac,CV_BACKEND:'cuda',CV_OUTPUT:'C:/cv-proof',CARGO_TARGET_DIR:'C:/cv-target',CV_FIXTURE_DIR:'C:/fixtures',ORT_DYLIB_PATH:'C:/onnxruntime.dll',SCENEWORKS_GPU_ID:'1',CUDA_VISIBLE_DEVICES:'1',NVIDIA_VISIBLE_DEVICES:'1',CUDA_DEVICE_ORDER:'PCI_BUS_ID'};
test('selected provider proofs cover both factors and preserve physical admission',()=>{
  assert.equal(admission(mac,'darwin').length,6);
  assert.equal(admission(cuda,'win32').length,4);
  for(const env of [{...mac,CV_MAC_GO:'false'},{...mac,DISPATCH_SHA:inference},{...mac,ORT_DYLIB_PATH:''}]) assert.throws(()=>admission(env,'darwin'));
  for(const key of ['SCENEWORKS_GPU_ID','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES']) assert.throws(()=>admission({...cuda,[key]:'0'},'win32'));
  assert.doesNotThrow(()=>admission({...mac,CV_BUILD_ONLY:'true',CV_MAC_GO:'false',CV_FIXTURE_DIR:'',ORT_DYLIB_PATH:''},'darwin'));
  for(const platform of ['darwin','win32']) assert.deepEqual(proofCases(platform).filter(p=>p.factor).map(p=>p.factor),[2,4]);
  assert(requiredFixtures('darwin').includes('refs_ext.safetensors'));
  assert(requiredFixtures('win32').includes('yolo11m.onnx'));
});
test('proof acceptance rejects skipped bodies, fallback, wrong case and wrong dimensions',()=>{
  const selected=proofCases('win32').find(p=>p.factor===2);
  const valid={case:selected.test.split('::').at(-1),completed:true,device:'cuda',result:{factor:2,width:1026,height:66,sourceWidth:513,sourceHeight:33,providerExecutions:[{executions:{CUDAExecutionProvider:2}}]}};
  validateReceipt(valid,selected);
  for(const changed of [{completed:false},{device:'cpu'},{case:'another'},{result:{...valid.result,width:513}},{result:{...valid.result,providerExecutions:[{executions:{CPUExecutionProvider:2}}]}}]) assert.throws(()=>validateReceipt({...valid,...changed},selected));
});
test('workflow has only manual triggers and preserves native execution guards',async()=>{
  const workflow=await readFile(new URL('../.github/workflows/cv-provider-validation.yml',import.meta.url),'utf8');
  assert.match(workflow,/workflow_dispatch:/);
  assert.doesNotMatch(workflow,/^  (push|pull_request|schedule|workflow_run):/m);
  assert.match(workflow,/real-weights/);
  assert.match(workflow,/--cuda-ordinals 0 --probe-visible 1/);
  assert.match(workflow,/cancel-in-progress: false/);
  const controller=await readFile(new URL('./cv-provider-validation.mjs',import.meta.url),'utf8');
  assert.match(controller,/enterMacMetalExecution/);
  assert.match(controller,/metalChildOptions/);
  assert.match(controller,/--no-run/);
  assert.match(controller,/ignoredTestRegistration/);
});

test('provisioning requires pinned fixtures, runtime extraction and independent oracle provenance',async()=>{
  const pins=JSON.parse(await readFile(new URL('./cv-fixture-manifest.json',import.meta.url),'utf8'));
  const selected=[...pins.models.filter(item=>item.backends.includes('mlx')),pins.photo];
  const identities=selected.map(item=>({file:item.name,bytes:item.bytes,sha256:item.sha256}));
  const outputs=Object.fromEntries(['refs.safetensors','refs_ext.safetensors','ref_people.json'].map(name=>[name,{sha256:'c'.repeat(64)}]));
  for(const [file,meta] of Object.entries(outputs)) identities.push({file,...meta});
  const runtime={path:'/tmp/ort.dylib',sha256:'d'.repeat(64)};
  const provision={status:'passed',sourceSha:source,inferenceSha:inference,
    onnx:selected.filter(item=>item.name.endsWith('.onnx')).map(item=>({file:item.name,tensors:1,externalData:false})),
    runtime:[{file:runtime.path,sha256:runtime.sha256}],
    oracle:{status:'passed',sourceSha:source,inferenceSha:inference,upstreamCommit:pins.oracle.upstreamCommit,device:'cpu',tensorCount:225,personCount:4,
      weightsSha256:pins.models.find(item=>item.name==='yolo11m_fused_mlx.safetensors').sha256,photoSha256:pins.photo.sha256,outputs}};
  validateProvisioning(provision,identities,runtime,pins,source,inference,'darwin');
  for(const mutate of [p=>p.status='failed',p=>p.oracle.tensorCount=224,p=>p.oracle.device='mlx',p=>p.oracle.sourceSha=inference,p=>p.runtime=[],p=>p.onnx[0].externalData=true,p=>delete p.oracle.outputs['refs.safetensors']]) {
    const changed=structuredClone(provision);mutate(changed);
    assert.throws(()=>validateProvisioning(changed,identities,runtime,pins,source,inference,'darwin'));
  }
  const wrong=structuredClone(identities);wrong[0].sha256='0'.repeat(64);
  assert.throws(()=>validateProvisioning(provision,wrong,runtime,pins,source,inference,'darwin'));
  const workflow=await readFile(new URL('../.github/workflows/cv-oracle-provision.yml',import.meta.url),'utf8');
  assert.match(workflow,/workflow_dispatch:/);
  assert.doesNotMatch(workflow,/^  (push|pull_request|schedule|workflow_run):/m);
});
