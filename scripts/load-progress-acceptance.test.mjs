import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { admission, childEnvironment, CASES } from './load-progress-acceptance.mjs';
const env = {LOAD_PROGRESS_CASE:'image_mlx',LOAD_PROGRESS_SOURCE_SHA:'a'.repeat(40),LOAD_PROGRESS_INFERENCE_SHA:'b'.repeat(40),LOAD_PROGRESS_OUTPUT:'/tmp/load-progress-case',LOAD_PROGRESS_MAC_GO:'true'};
test('manual selector freezes one exact ignored test and rejects platform mismatch',()=>{
  assert.equal(admission(env,'darwin'),'tests::load_progress_acceptance::image_mlx');
  assert.throws(()=>admission({...env,LOAD_PROGRESS_CASE:'image_cuda'},'darwin'),/Windows/);
  assert.throws(()=>admission({...env,LOAD_PROGRESS_CASE:'all'},'darwin'),/one known/);
  assert.throws(()=>admission({...env,LOAD_PROGRESS_SOURCE_SHA:'main'},'darwin'),/exact source/);
  assert.throws(()=>admission({...env,LOAD_PROGRESS_INFERENCE_SHA:'main'},'darwin'),/exact inference/);
});
test('Mac compute requires explicit go; compilation and metadata checks do not',()=>{
  const noGo={...env,LOAD_PROGRESS_MAC_GO:'false'};
  assert.throws(()=>admission(noGo,'darwin'),/explicit.*go/);
  assert.doesNotThrow(()=>admission(noGo,'darwin',{execute:false}));
});
test('CUDA retains established physical GPU0 restrictions for all candle cases',()=>{
  const cuda={...env,LOAD_PROGRESS_CASE:'image_cuda',SCENEWORKS_GPU_ID:'0',CUDA_VISIBLE_DEVICES:'0',NVIDIA_VISIBLE_DEVICES:'0',CUDA_DEVICE_ORDER:'PCI_BUS_ID'};
  assert.equal(admission(cuda,'win32'),'tests::load_progress_acceptance::image_cuda');
  for(const key of ['SCENEWORKS_GPU_ID','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES']) assert.throws(()=>admission({...cuda,[key]:'1'},'win32'),/required/);
  for(const selected of ['audio','training']) assert.doesNotThrow(()=>admission({...cuda,LOAD_PROGRESS_CASE:selected},'win32'));
});
test('child setup keeps frozen cache and device but removes credentials and mutable app state',()=>{
  const child=childEnvironment({...env,HF_TOKEN:'private',GH_TOKEN:'private',SCENEWORKS_DATA_DIR:'/production',SCENEWORKS_GPU_ID:'0',HF_HUB_CACHE:'/frozen/hub',SCENEWORKS_METAL_EXECUTION_LOCK_FD:'5'},'/owned/state','/owned/evidence');
  assert.equal(child.HF_TOKEN,undefined);assert.equal(child.GH_TOKEN,undefined);assert.equal(child.SCENEWORKS_DATA_DIR,undefined);
  assert.equal(child.HF_HUB_CACHE,'/frozen/hub');assert.equal(child.SCENEWORKS_GPU_ID,'0');assert.equal(child.SCENEWORKS_METAL_EXECUTION_LOCK_FD,'5');
  assert.equal(child.HF_HUB_OFFLINE,'1');assert.equal(child.TRANSFORMERS_OFFLINE,'1');assert.equal(child.CARGO_BUILD_JOBS,'2');
});
test('all four real cases are ignored and the existing eight-cell campaign remains separate',async()=>{
  const source=await readFile(new URL('../apps/rust-api/src/tests/load_progress_acceptance.rs',import.meta.url),'utf8');
  for(const selected of CASES) assert.match(source,new RegExp('#\\[ignore = "sc27984[^\\n]+\\nasync fn '+selected+'\\('));
  const controller=await readFile(new URL('./load-progress-acceptance.mjs',import.meta.url),'utf8');
  assert.match(controller,/--no-run/);assert.match(controller,/--ignored/);assert.match(controller,/--exact/);
  assert.doesNotMatch(controller,/moss-training-acceptance\.mjs|memory-catalog-campaign/);
});
