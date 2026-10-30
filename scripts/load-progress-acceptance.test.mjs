import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { admission, childEnvironment, dispatchOptions, cacheInventory, ortSmokeRequired, validateOrtReceipt, ORT_FILTER, CASES } from './load-progress-acceptance.mjs';
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

test('registered dispatcher selects only bounded profile cases and rejects wrong source/options before native setup',()=>{
  const base={profile:'load-progress-acceptance',ref:'a'.repeat(40),inference_sha:'b'.repeat(40),backend:'candle',moss_options_json:JSON.stringify({case:'image_cuda',hf_hub_cache:'E:/huggingface/hub'})};
  assert.deepEqual(dispatchOptions(base,base.ref),{case:'image_cuda',backend:'cuda',hf_hub_cache:'E:/huggingface/hub',build_only:true,mac_go:false});
  assert.throws(()=>dispatchOptions({...base,profile:'catalog'},base.ref));
  assert.throws(()=>dispatchOptions(base,'c'.repeat(40)),/exact dispatched/);
  for(const options of [{case:'all',hf_hub_cache:'E:/hub'},{case:'image_mlx',hf_hub_cache:'E:/hub'},{case:'image_cuda',hf_hub_cache:'relative'},{case:'image_cuda',hf_hub_cache:'E:/hub',unknown:true},{case:'image_cuda',hf_hub_cache:'E:/hub',build_only:'false'}]) assert.throws(()=>dispatchOptions({...base,moss_options_json:JSON.stringify(options)},base.ref));
  const mlx={...base,backend:'mlx',moss_options_json:JSON.stringify({case:'image_mlx',hf_hub_cache:'/frozen/hub',build_only:false})};
  assert.throws(()=>dispatchOptions(mlx,mlx.ref),/explicit.*go/);
  assert.equal(dispatchOptions({...mlx,moss_options_json:JSON.stringify({case:'image_mlx',hf_hub_cache:'/frozen/hub',build_only:false,mac_go:true})},mlx.ref).build_only,false);
});
test('manual profile calls only dedicated acceptance and keeps catalog and MOSS campaign conditions separate',async()=>{
  const dispatcher=await readFile(new URL('../.github/workflows/memory-catalog-campaign.yml',import.meta.url),'utf8');
  assert.match(dispatcher,/if \(i.profile === 'load-progress-acceptance'\) \{ emit\('enabled','false'\)/);
  assert.match(dispatcher,/uses: \.\/\.github\/workflows\/load-progress-acceptance.yml/);
  assert.match(dispatcher,/inputs.profile == '' \|\| inputs.profile == 'catalog'/);
  const workflow=await readFile(new URL('../.github/workflows/load-progress-acceptance.yml',import.meta.url),'utf8');
  assert.match(workflow,/workflow_dispatch:/);assert.match(workflow,/workflow_call:/);assert.doesNotMatch(workflow,/^  (push|pull_request|schedule):/m);
});

test('cache preflight stats only exact frozen files and retains all missing component names',async()=>{
  const manifest=[{id:'chatterbox_tts',downloads:[{repo:'Example/Audio',revision:'a'.repeat(40),files:['model.safetensors']},{repo:'Example/Codec',revision:'b'.repeat(40),componentId:'codec',files:['weights.safetensors']}]}];
  const result=await cacheInventory('/nonexistent-sc27980-hub','audio',manifest);
  assert.equal(result.requiredFiles,2);assert.equal(result.presentFiles,0);assert.equal(result.missing.length,2);assert.equal(result.missing[1].component,'codec');
  assert.equal(result.totalPresentBytes,0);
});

test('ORT session construction is selected once before CUDA image only, never ordinary CI',()=>{
  assert.equal(ORT_FILTER,'model_io::tests::external_initializer_session_creation_acceptance');
  assert.equal(ortSmokeRequired('image_cuda','win32'),true);
  for(const selected of ['image_mlx','audio','training']) assert.equal(ortSmokeRequired(selected,'win32'),false);
  for(const selected of CASES) assert.equal(ortSmokeRequired(selected,'darwin'),false);
});

test('ORT receipt binds actual session result to bounded byte ranges and refusal coverage',()=>{
  const receipt={kind:'sc27980-ort-session-acceptance',passed:true,outputs:4,externalFileBytes:16777216,selectedExternalBytes:40,nestedExternalData:true,omittedLength:true,aliasesAndOverlaps:true,decimalOffsetPadding:true,sourceModelUnchanged:true,missingTruncatedOverflowRefusals:true};
  assert.doesNotThrow(()=>validateOrtReceipt(receipt));
  for(const [key,value] of [['selectedExternalBytes',16777216],['outputs',0],['sourceModelUnchanged',false],['missingTruncatedOverflowRefusals',false]]) assert.throws(()=>validateOrtReceipt({...receipt,[key]:value}));
});
