import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, rm, mkdir, realpath, symlink, stat } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { admission, freshOutput, outsideCheckout, workflowEnvironment, childEnvironment, dispatchOptions, cacheInventory, ortSmokeRequired, validateOrtReceipt, ignoredTestRegistration, API_TEST_TARGET, ORT_FILTER, runCudaAll, CUDA_CASES, NATIVE_CASES, CASES } from './load-progress-acceptance.mjs';
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
test('workflow preflight normalizes Windows artifact paths and preserves resource/source refusal', async()=>{
  const selected={...env,LOAD_PROGRESS_BACKEND:'cuda',LOAD_PROGRESS_CASE:'cuda_all',DISPATCH_SHA:env.LOAD_PROGRESS_SOURCE_SHA,LOAD_PROGRESS_OUTPUT:'D:\\runner\\_work\\SceneWorks\\SceneWorks/../load-progress-123-1',LOAD_PROGRESS_TARGET_DIR:'D:\\runner\\_work\\SceneWorks\\SceneWorks/../target-cuda',SCENEWORKS_GPU_ID:'0',CUDA_VISIBLE_DEVICES:'0',NVIDIA_VISIBLE_DEVICES:'0',CUDA_DEVICE_ORDER:'PCI_BUS_ID'};
  const actual=workflowEnvironment(selected,'win32');
  assert.equal(actual.output,'directory=D:/runner/_work/SceneWorks/load-progress-123-1\n');
  assert(actual.environment.includes('LOAD_PROGRESS_TARGET_DIR=D:\\runner\\_work\\SceneWorks\\target-cuda\n'));
  assert.throws(()=>workflowEnvironment({...selected,DISPATCH_SHA:'c'.repeat(40)},'win32'),/dispatch/);
  assert.throws(()=>workflowEnvironment({...selected,CUDA_VISIBLE_DEVICES:'1'},'win32'),/required/);
  assert.throws(()=>workflowEnvironment({...selected,LOAD_PROGRESS_OUTPUT:'D:/output\ninjected=yes'},'win32'),/absolute/);
  const mac={...env,LOAD_PROGRESS_BACKEND:'mlx',DISPATCH_SHA:env.LOAD_PROGRESS_SOURCE_SHA,LOAD_PROGRESS_TARGET_DIR:'/tmp/worker/../target',LOAD_PROGRESS_MAC_GO:'false'};
  assert.throws(()=>workflowEnvironment(mac,'darwin'),/explicit.*go/);
  assert.equal(workflowEnvironment({...mac,BUILD_ONLY:'true'},'darwin').output,'directory=/tmp/load-progress-case\n');
  const workflow=await readFile(new URL('../.github/workflows/load-progress-acceptance.yml',import.meta.url),'utf8');
  assert.match(workflow,/id: preflight-cuda\n        if: \$\{\{ inputs.backend == 'cuda' \|\| inputs.backend == 'cpu' \}\}\n        shell: powershell/);
  assert.match(workflow,/id: preflight-mlx\n        if: \$\{\{ inputs.backend == 'mlx' \}\}\n        shell: bash/);
  assert.match(workflow,/node scripts\/load-progress-acceptance.mjs --workflow-preflight/);
  assert.match(workflow,/steps.preflight-cuda.outputs.directory \|\| steps.preflight-mlx.outputs.directory/);
  assert.doesNotMatch(workflow,/path: \|[\s\S]*github.workspace.*\/\.\.\//);
});
test('actual CUDA acceptance owns the shared UUID lease while build-only stays outside it',async()=>{
  const workflow=await readFile(new URL('../.github/workflows/load-progress-acceptance.yml',import.meta.url),'utf8');
  assert.match(workflow,/if "%BUILD_ONLY%"=="true" \(node scripts\/load-progress-acceptance.mjs --build-only\) else \(python -I scripts\/ci\/windows_cuda_execution.py --actor load-progress-acceptance --budget-minutes 170 --cuda-ordinals 0 --probe-visible 0 -- node scripts\/load-progress-acceptance.mjs\)/);
  assert.match(workflow,/sceneworks-cuda-execution-\$\{\{ github.run_id \}\}-\$\{\{ github.run_attempt \}\}-load-progress-acceptance\//);
});
test('child setup keeps frozen cache and device but removes credentials and mutable app state',()=>{
  const child=childEnvironment({...env,HF_TOKEN:'private',GH_TOKEN:'private',SCENEWORKS_DATA_DIR:'/production',SCENEWORKS_GPU_ID:'0',HF_HUB_CACHE:'/frozen/hub',SCENEWORKS_METAL_EXECUTION_LOCK_FD:'5',SW_CI_CUDA_EXECUTION_ACTIVE:'load-progress-acceptance'},'/owned/state','/owned/evidence');
  assert.equal(child.HF_TOKEN,undefined);assert.equal(child.GH_TOKEN,undefined);assert.equal(child.SCENEWORKS_DATA_DIR,undefined);
  assert.equal(child.HF_HUB_CACHE,'/frozen/hub');assert.equal(child.SCENEWORKS_GPU_ID,'0');assert.equal(child.SCENEWORKS_METAL_EXECUTION_LOCK_FD,'5');
  assert.equal(child.HF_HUB_OFFLINE,'1');assert.equal(child.TRANSFORMERS_OFFLINE,'1');assert.equal(child.CARGO_BUILD_JOBS,'2');
  assert.equal(child.SW_CI_CUDA_EXECUTION_ACTIVE,'load-progress-acceptance');
});
test('Windows child retains toolchain search path once before adding the ORT DLL directory',()=>{
  const child=childEnvironment({...env,Path:'C:\\Rust\\bin;C:\\MSVC\\bin'},'/owned/state','/owned/evidence','win32');
  assert.equal(child.PATH,'C:\\Rust\\bin;C:\\MSVC\\bin');
  assert.deepEqual(Object.keys(child).filter(key=>key.toLowerCase()==='path'),['PATH']);
  child.PATH='C:\\ORT;'+child.PATH;
  assert.equal(child.PATH,'C:\\ORT;C:\\Rust\\bin;C:\\MSVC\\bin');
});
test('all four real cases are ignored and the existing eight-cell campaign remains separate',async()=>{
  const source=await readFile(new URL('../apps/rust-api/src/tests/load_progress_acceptance.rs',import.meta.url),'utf8');
  for(const selected of NATIVE_CASES) assert.match(source,new RegExp('#\\[ignore = "sc27984[^\\n]+\\nasync fn '+selected+'\\('));
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

test('ORT session construction is a separate CPU case, never part of any GPU case',()=>{
  assert.equal(ORT_FILTER,'model_io::tests::external_initializer_session_creation_acceptance');
  assert.equal(ortSmokeRequired('ort_cpu','win32'),true);
  for(const selected of [...NATIVE_CASES,'cuda_all']) assert.equal(ortSmokeRequired(selected,'win32'),false);
  for(const selected of CASES) assert.equal(ortSmokeRequired(selected,'darwin'),false);
});

test('ORT receipt binds actual session result to bounded byte ranges and refusal coverage',()=>{
  const receipt={kind:'sc27980-ort-session-acceptance',passed:true,executionProvider:'CPUExecutionProvider',tensorValuesVerified:true,outputs:4,externalFileBytes:16777216,selectedExternalBytes:40,nestedExternalData:true,omittedLength:true,aliasesAndOverlaps:true,decimalOffsetPadding:true,sourceModelUnchanged:true,missingTruncatedOverflowRefusals:true};
  assert.doesNotThrow(()=>validateOrtReceipt(receipt));
  for(const [key,value] of [['executionProvider','CUDAExecutionProvider'],['tensorValuesVerified',false],['selectedExternalBytes',16777216],['outputs',0],['sourceModelUnchanged',false],['missingTruncatedOverflowRefusals',false]]) assert.throws(()=>validateOrtReceipt({...receipt,[key]:value}));
});

test('CUDA aggregate selector is explicit and rejected on MLX',()=>{
  const base={profile:'load-progress-acceptance',ref:'a'.repeat(40),inference_sha:'b'.repeat(40),backend:'candle',moss_options_json:JSON.stringify({case:'cuda_all',hf_hub_cache:'E:/huggingface/hub',build_only:false})};
  assert.equal(dispatchOptions(base,base.ref).case,'cuda_all');
  assert.throws(()=>dispatchOptions({...base,backend:'mlx',moss_options_json:JSON.stringify({case:'cuda_all',hf_hub_cache:'/frozen/hub',mac_go:true})},base.ref),/Windows/);
});

test('CUDA aggregate runs the existing cases sequentially with one target and independent fresh outputs',async()=>{
  const temp=await mkdtemp(path.join(os.tmpdir(),'sc27980-aggregate-test-'));
  try {
    const selected={...env,LOAD_PROGRESS_CASE:'cuda_all',LOAD_PROGRESS_OUTPUT:path.join(temp,'attempt')};
    const calls=[];
    const result=await runCudaAll(selected,{},async (child,options)=>{
      calls.push(child);
      assert.deepEqual(options,{});
      return {case:child.LOAD_PROGRESS_CASE,status:'passed',sourceSha:child.LOAD_PROGRESS_SOURCE_SHA,inferenceSha:child.LOAD_PROGRESS_INFERENCE_SHA};
    });
    assert.equal(result.status,'passed');
    assert.deepEqual(calls.map(child=>child.LOAD_PROGRESS_CASE),CUDA_CASES);
    assert.equal(new Set(calls.map(child=>child.LOAD_PROGRESS_TARGET_DIR)).size,1);
    assert.equal(new Set(calls.map(child=>child.LOAD_PROGRESS_OUTPUT)).size,3);
    assert.deepEqual(calls.filter(child=>ortSmokeRequired(child.LOAD_PROGRESS_CASE,'win32')).map(child=>child.LOAD_PROGRESS_CASE),[]);
    const saved=JSON.parse(await readFile(path.join(selected.LOAD_PROGRESS_OUTPUT,'evidence/controller.json'),'utf8'));
    assert.deepEqual(saved.cases.map(item=>item.case),CUDA_CASES);
    assert.equal(saved.status,'passed');
  } finally { await rm(temp,{recursive:true,force:true}); }
});

for(const options of [{buildOnly:true},{checkOnly:true}]) {
  test(`CUDA aggregate ${options.buildOnly?'build-only':'check-only'} prepares once and cannot claim acceptance`,async()=>{
    const temp=await mkdtemp(path.join(os.tmpdir(),'sc27980-aggregate-mode-test-'));
    try {
      const calls=[];
      const expected=options.buildOnly?'compiled-no-model-execution':'checked-no-model-execution';
      const result=await runCudaAll({...env,LOAD_PROGRESS_CASE:'cuda_all',LOAD_PROGRESS_OUTPUT:path.join(temp,'attempt')},options,async (child,mode)=>{
        calls.push(child.LOAD_PROGRESS_CASE); assert.deepEqual(mode,options);
        return {case:child.LOAD_PROGRESS_CASE,status:expected,sourceSha:child.LOAD_PROGRESS_SOURCE_SHA,inferenceSha:child.LOAD_PROGRESS_INFERENCE_SHA};
      });
      assert.deepEqual(calls,['image_cuda']); assert.equal(result.status,expected);assert.notEqual(result.status,'passed');
    } finally { await rm(temp,{recursive:true,force:true}); }
  });
}

test('CUDA aggregate fails closed without running later cases or losing prior receipts',async()=>{
  const temp=await mkdtemp(path.join(os.tmpdir(),'sc27980-aggregate-fail-test-'));
  try {
    const calls=[],output=path.join(temp,'attempt');
    await assert.rejects(runCudaAll({...env,LOAD_PROGRESS_CASE:'cuda_all',LOAD_PROGRESS_OUTPUT:output},{},async child=>{
      calls.push(child.LOAD_PROGRESS_CASE);
      if(child.LOAD_PROGRESS_CASE==='audio') throw new Error('owned audio failed');
      return {case:child.LOAD_PROGRESS_CASE,status:'passed',sourceSha:child.LOAD_PROGRESS_SOURCE_SHA,inferenceSha:child.LOAD_PROGRESS_INFERENCE_SHA};
    }),/owned audio failed/);
    assert.deepEqual(calls,['image_cuda','audio']);
    const saved=JSON.parse(await readFile(path.join(output,'evidence/controller.json'),'utf8'));
    assert.equal(saved.status,'failed'); assert.deepEqual(saved.cases.map(item=>item.case),['image_cuda']);
    assert.deepEqual(saved.requiredCases,CUDA_CASES);
  } finally { await rm(temp,{recursive:true,force:true}); }
});

test('acceptance concurrency preserves distinct pending dispatches on both backends',async()=>{
  const workflow=await readFile(new URL('../.github/workflows/load-progress-acceptance.yml',import.meta.url),'utf8');
  assert.match(workflow,/group: \$\{\{ format\('load-progress-\{0\}-\{1\}-\{2\}', inputs.backend, github.run_id, github.run_attempt\) \}\}/);
  assert.doesNotMatch(workflow,/windows-candle-gpu-real-weights/);
  assert.match(workflow,/cancel-in-progress: false/);
  for(const selected of CUDA_CASES) assert(workflow.includes('/'+selected+'/evidence/'),'aggregate child evidence must be uploaded');
});

test('API test target owns the actual registered acceptance module rather than the CLI binary',async()=>{
  const manifest=await readFile(new URL('../apps/rust-api/Cargo.toml',import.meta.url),'utf8');
  const packageName=/^name = "([^"]+)"/m.exec(manifest)[1];
  assert.deepEqual(API_TEST_TARGET.args,['--lib']);
  assert.equal(API_TEST_TARGET.name,packageName.replaceAll('-','_'));
  const library=await readFile(new URL('../'+API_TEST_TARGET.source,import.meta.url),'utf8');
  assert.match(library,/#\[cfg\(test\)\]\s*mod tests;/);
  const modules=await readFile(new URL('../apps/rust-api/src/tests/mod.rs',import.meta.url),'utf8');
  assert.match(modules,/mod load_progress_acceptance;/);
  const cases=await readFile(new URL('../apps/rust-api/src/tests/load_progress_acceptance.rs',import.meta.url),'utf8');
  for(const selected of NATIVE_CASES) assert(cases.includes(`async fn ${selected}()`));
});

test('zero-match and unrelated libtest listings cannot reach acceptance execution',()=>{
  const selected='tests::load_progress_acceptance::image_cuda';
  assert.deepEqual(ignoredTestRegistration(`${selected}: test\r\n\r\n1 test, 0 benchmarks\r\n`,selected),{filter:selected,registered:true});
  for(const listing of ['0 tests, 0 benchmarks\n','tests::other: test\n1 test, 0 benchmarks\n',`${selected}: test\ntests::other: test\n2 tests, 0 benchmarks\n`]) assert.throws(()=>ignoredTestRegistration(listing,selected),/exact ignored test is not registered/);
});

const cpuEnv={...env,LOAD_PROGRESS_CASE:'ort_cpu',LOAD_PROGRESS_BACKEND:'cpu',SCENEWORKS_GPU_ID:'cpu',CUDA_VISIBLE_DEVICES:'',NVIDIA_VISIBLE_DEVICES:''};
test('CPU selector is independent of model cache and Mac go, with exact source and worker filter',()=>{
  const inputs={profile:'load-progress-acceptance',ref:env.LOAD_PROGRESS_SOURCE_SHA,inference_sha:env.LOAD_PROGRESS_INFERENCE_SHA,backend:'candle',moss_options_json:JSON.stringify({case:'ort_cpu',build_only:false})};
  assert.deepEqual(dispatchOptions(inputs,inputs.ref),{case:'ort_cpu',backend:'cpu',hf_hub_cache:'',build_only:false,mac_go:false});
  assert.throws(()=>dispatchOptions({...inputs,backend:'mlx'},inputs.ref),/Windows candle/);
  assert.throws(()=>dispatchOptions(inputs,'c'.repeat(40)),/exact dispatched/);
  assert.equal(admission(cpuEnv,'win32'),ORT_FILTER);
  assert.throws(()=>admission(cpuEnv,'darwin'),/Windows/);
  for(const key of ['CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES']) assert.throws(()=>admission({...cpuEnv,[key]:'0'},'win32'),/empty/);
  assert.throws(()=>admission({...cpuEnv,SCENEWORKS_GPU_ID:'0'},'win32'),/CPU device/);
  for(const selectedCase of [...NATIVE_CASES,'cuda_all']) assert.throws(()=>admission({...cpuEnv,LOAD_PROGRESS_CASE:selectedCase},'win32'),/case\/backend mismatch/);
  for(const backend of ['cuda','mlx']) assert.throws(()=>admission({...cpuEnv,LOAD_PROGRESS_BACKEND:backend},'win32'),/case\/backend mismatch/);
  const preflight={...cpuEnv,DISPATCH_SHA:env.LOAD_PROGRESS_SOURCE_SHA,LOAD_PROGRESS_TARGET_DIR:'C:/targets/ort',LOAD_PROGRESS_OUTPUT:'C:/acceptance/ort'};
  assert.doesNotThrow(()=>workflowEnvironment(preflight,'win32'));
  assert.throws(()=>workflowEnvironment({...preflight,LOAD_PROGRESS_BACKEND:undefined},'win32'),/explicit workflow backend/);
  const child=childEnvironment({...cpuEnv,CUDA_VISIBLE_DEVICES:'0',NVIDIA_VISIBLE_DEVICES:'0'},'state','evidence','win32');
  assert.equal(child.SCENEWORKS_GPU_ID,'cpu'); assert.equal(child.CUDA_VISIBLE_DEVICES,''); assert.equal(child.NVIDIA_VISIBLE_DEVICES,'');
});
test('dedicated CPU route preserves GPU guards and bypasses model compilation, inventory and device probes',async()=>{
  const controller=await readFile(new URL('./load-progress-acceptance.mjs',import.meta.url),'utf8');
  const workflow=await readFile(new URL('../.github/workflows/load-progress-acceptance.yml',import.meta.url),'utf8');
  const fixture=await readFile(new URL('../crates/sceneworks-worker/src/model_io.rs',import.meta.url),'utf8');
  assert.match(controller,/if\(!cpu\) receipt.binary=await compileTest\('sceneworks-rust-api'/);
  assert.match(controller,/if \(!buildOnly && !cpu\)/);
  assert.match(controller,/if\(cpu\) \{\s+receipt.device=\{executionProvider:'CPUExecutionProvider'/);
  assert.match(controller,/process.env.LOAD_PROGRESS_CASE!=='ort_cpu' && await enterMacMetalExecution/);
  assert.match(workflow,/inputs.backend == 'cpu' && fromJSON\('\["self-hosted","Windows","X64","cuda"\]'/);
  const gpuStep=workflow.split('      - name: Compile then execute the selected bounded CUDA jobs')[1].split('      - name:')[0];
  assert.match(gpuStep,/if: \$\{\{ inputs.backend == 'cuda' \}\}/);
  assert.match(gpuStep,/windows_cuda_execution.py/);
  const cpuStep=workflow.split('      - name: Compile then execute only the explicit CPU ORT fixture')[1].split('      - name:')[0];
  assert.match(cpuStep,/if: \$\{\{ inputs.backend == 'cpu' \}\}/);
  assert.doesNotMatch(cpuStep,/windows_cuda_execution|nvidia-smi|metal/);
  assert.match(fixture,/with_no_environment_execution_providers\(\)/);
  assert.match(fixture,/with_execution_providers\(\[ort::ep::CPU::default\(\).build\(\).error_on_failure\(\)\]\)/);
  assert.match(fixture,/"executionProvider":"CPUExecutionProvider"/);
});


test('outside-checkout boundary accepts the exact parent while retaining same-volume and containment refusals',()=>{
  const checkout='D:\\actions-runner-3\\_work\\SceneWorks\\SceneWorks';
  const parent=path.win32.dirname(checkout);
  assert.equal(path.win32.relative(checkout,parent),'..');
  assert.equal(outsideCheckout(checkout,parent,path.win32),true);
  assert.equal(outsideCheckout(checkout,path.win32.join(parent,'load-progress-41175299803-1'),path.win32),true);
  assert.equal(outsideCheckout(checkout,'E:\\outside',path.win32),false);
  assert.equal(outsideCheckout(checkout,checkout,path.win32),false);
  assert.equal(outsideCheckout(checkout,path.win32.join(checkout,'nested'),path.win32),false);
  assert.equal(outsideCheckout('/runner/SceneWorks/SceneWorks','/runner/SceneWorks',path.posix),true);
  assert.equal(outsideCheckout('/runner/SceneWorks/SceneWorks','/runner/SceneWorks/SceneWorks/nested',path.posix),false);
});
test('fresh sibling creates evidence while checkout, inside, symlink-inside and reused attempts are refused',async()=>{
  const temp=await realpath(await mkdtemp(path.join(os.tmpdir(),'sc27980-output-boundary-')));
  try {
    const checkout=path.join(temp,'SceneWorks'); await mkdir(checkout);
    const output=path.join(temp,'load-progress-attempt');
    const created=await freshOutput(output,checkout);
    assert.equal(created.output,output); assert((await stat(created.evidence)).isDirectory());
    await assert.rejects(freshOutput(output,checkout),{code:'EEXIST'});
    await assert.rejects(freshOutput(checkout,checkout),/output must be outside checkout/);
    await assert.rejects(freshOutput(path.join(checkout,'nested'),checkout),/output must be outside checkout/);
    const link=path.join(temp,'linked-checkout'); await symlink(checkout,link,'junction');
    await assert.rejects(freshOutput(path.join(link,'attempt'),checkout),/output parent must physically be outside checkout/);
  } finally { await rm(temp,{recursive:true,force:true}); }
});
