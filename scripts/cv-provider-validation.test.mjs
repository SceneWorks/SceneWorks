import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {admission,proofCases,validateReceipt,requiredFixtures} from './cv-provider-validation.mjs';
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
