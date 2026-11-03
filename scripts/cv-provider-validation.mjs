#!/usr/bin/env node
// Explicit bounded CV provider proofs. No model execution on push or pull_request.
import assert from 'node:assert/strict';
import { spawn, execFile as execFileCallback } from 'node:child_process';
import { createReadStream } from 'node:fs';
import { mkdir, open, readFile, readdir, realpath, stat, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { promisify } from 'node:util';
import { assertPins } from './qwen-image-2-1-app-lifecycle.mjs';
import { ignoredTestRegistration } from './load-progress-acceptance.mjs';
import { enterMacMetalExecution, metalChildOptions } from './lib/apple-metal-execution-lock.mjs';
const execFile = promisify(execFileCallback);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export function proofCases(platform) {
  const person = platform === 'darwin'
    ? ['yolo11_mlx_forward_matches_reference_oracle', 'yolo11_mlx_per_block_isolation', 'yolo11_matches_ultralytics_reference_on_photo']
    : ['person_detect_candle_real_weights_finds_person'];
  return [
    ...person.map(name => ({name, test:`person_jobs::tests::${name}`, device:platform === 'darwin' ? 'mlx' : 'cuda'})),
    {name:'pose', test:'pose_jobs::tests::pose_detect_candle_real_weights_finds_person', device:platform === 'darwin' ? 'coreml' : 'cuda'},
    ...[2,4].map(factor => ({name:`upscale_x${factor}`, test:'upscale_jobs::tests::real_esrgan_candle_real_weights_upscales', device:platform === 'darwin' ? 'coreml' : 'cuda', factor})),
  ];
}
export function admission(env, platform = process.platform) {
  assert(['darwin','win32'].includes(platform), 'native macOS or Windows required');
  assert.equal(env.CV_BACKEND, platform === 'darwin' ? 'mlx' : 'cuda');
  for(const name of ['CV_SOURCE_SHA','CV_INFERENCE_SHA']) assert(/^[a-f0-9]{40}$/.test(env[name] ?? ''), `${name}: exact SHA required`);
  if(env.DISPATCH_SHA) assert.equal(env.DISPATCH_SHA,env.CV_SOURCE_SHA,'dispatch ref must equal source SHA');
  assert(['true','false'].includes(env.CV_BUILD_ONLY),'explicit build-only selection required');
  const paths = platform === 'win32' ? path.win32 : path.posix;
  for(const name of ['CV_OUTPUT','CARGO_TARGET_DIR']) assert(paths.isAbsolute(env[name] ?? '') && !/[\r\n]/.test(env[name]), `${name}: absolute path required`);
  if(env.CV_BUILD_ONLY !== 'true') {
    for(const name of ['CV_FIXTURE_DIR','ORT_DYLIB_PATH']) assert(paths.isAbsolute(env[name] ?? '') && !/[\r\n]/.test(env[name]), `${name}: existing absolute path required`);
    if(platform === 'darwin') assert.equal(env.CV_MAC_GO,'true','explicit developer Mac execution go required');
    else {
      for(const name of ['SCENEWORKS_GPU_ID','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES']) assert.equal(env[name],'1',`${name}: physical GPU1 required`);
      assert.equal(env.CUDA_DEVICE_ORDER,'PCI_BUS_ID');
    }
  }
  return proofCases(platform);
}
export function validateReceipt(receipt, selected) {
  assert.equal(receipt.case,selected.test.split('::').at(-1));
  assert.equal(receipt.completed,true,'test body must complete');
  assert.equal(receipt.device,selected.device,'required provider must execute; fallback is not hardware evidence');
  assert(receipt.result && typeof receipt.result === 'object');
  if(selected.device!=='mlx') {
    const profiles=receipt.result.providerExecutions;
    assert.equal(profiles?.length,selected.name==='pose'?2:1,'every ONNX graph needs execution evidence');
    const provider=selected.device==='cuda'?'CUDAExecutionProvider':'CoreMLExecutionProvider';
    for(const profile of profiles) assert(profile.executions?.[provider]>0,`no ${provider} node executed`);
  }
  if(selected.factor) {
    assert.equal(receipt.result.factor,selected.factor);
    assert.equal(receipt.result.width,receipt.result.sourceWidth*selected.factor);
    assert.equal(receipt.result.height,receipt.result.sourceHeight*selected.factor);
  }
}
export function validateProvisioning(provision, fixtureIdentities, runtime, pins, source, inference, platform) {
  assert.equal(provision.status,'passed','complete provisioning receipt required');
  assert.equal(provision.sourceSha,source); assert.equal(provision.inferenceSha,inference);
  const backend=platform==='darwin'?'mlx':'cuda';
  const identities=new Map(fixtureIdentities.map(item=>[item.file,item]));
  const expected=[...pins.models.filter(item=>item.backends.includes(backend)),pins.photo];
  for(const item of expected) {
    const actual=identities.get(item.name);
    assert(actual,`missing pinned fixture ${item.name}`);
    assert.equal(actual.sha256,item.sha256,`${item.name}: wrong pinned bytes`);
    assert.equal(actual.bytes,item.bytes,`${item.name}: wrong pinned length`);
  }
  const onnx=expected.filter(item=>item.name.endsWith('.onnx'));
  assert.equal(provision.onnx?.length,onnx.length);
  for(const item of onnx) assert(provision.onnx.some(graph=>graph.file===item.name&&graph.externalData===false&&graph.tensors>0));
  assert(provision.runtime?.some(item=>item.file===runtime.path&&item.sha256===runtime.sha256),'runtime must match the verified wheel extraction');
  if(platform==='darwin') {
    const oracle=provision.oracle;
    assert.equal(oracle?.status,'passed'); assert.equal(oracle.sourceSha,source); assert.equal(oracle.inferenceSha,inference);
    assert.equal(oracle.upstreamCommit,pins.oracle.upstreamCommit); assert.equal(oracle.device,'cpu');
    assert.equal(oracle.tensorCount,225); assert.equal(oracle.personCount,4);
    assert.equal(oracle.weightsSha256,pins.models.find(item=>item.name==='yolo11m_fused_mlx.safetensors').sha256);
    assert.equal(oracle.photoSha256,pins.photo.sha256);
    for(const name of ['refs.safetensors','refs_ext.safetensors','ref_people.json']) {
      const actual=identities.get(name), expected=oracle.outputs?.[name];
      assert(actual&&expected&&/^[a-f0-9]{64}$/.test(expected.sha256),`missing independent oracle ${name}`);
      assert.equal(actual.sha256,expected.sha256,`mismatched independent oracle ${name}`);
    }
  }
}
async function sha256(file) {
  const hash=createHash('sha256'); for await (const data of createReadStream(file)) hash.update(data); return hash.digest('hex');
}
export function requiredFixtures(platform) {
  return ['people.jpg','yolox_m_8xb8-300e_humanart-c2c7a14a.onnx',
    'rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx','real_esrgan_x2.onnx','real_esrgan_x4.onnx',
    ...(platform==='darwin'?['yolo11m_fused_mlx.safetensors','refs.safetensors','refs_ext.safetensors','ref_people.json']:['yolo11m.onnx'])];
}
async function fixtureIdentity(directory) {
  const result=[];
  // A dedicated, explicitly supplied fixture directory includes every external
  // ONNX data file; never infer or traverse an application/HF cache root.
  async function visit(relative='') {
    for(const item of await readdir(path.join(directory,relative),{withFileTypes:true})) {
      const name=path.join(relative,item.name); assert(!item.isSymbolicLink(),`fixture symlink not allowed: ${name}`);
      if(item.isDirectory()) await visit(name);
      else { assert(item.isFile()); const file=path.join(directory,name); result.push({file:name,bytes:(await stat(file)).size,sha256:await sha256(file)}); }
    }
  }
  await visit(); return result.sort((a,b)=>a.file.localeCompare(b.file));
}
export async function run(env=process.env) {
  const cases=admission(env), buildOnly=env.CV_BUILD_ONLY==='true';
  const git=async (...args)=>(await execFile('git',args,{cwd:root})).stdout.trim();
  assert.equal(await git('rev-parse','HEAD'),env.CV_SOURCE_SHA);
  assert.equal(await git('status','--porcelain'),'','clean paired source required');
  const manifests=await Promise.all(['Cargo.toml','crates/sceneworks-worker/Cargo.toml','crates/sceneworks-memory-adapter/Cargo.toml'].map(file=>readFile(path.join(root,file),'utf8')));
  const lock=await readFile(path.join(root,'Cargo.lock'),'utf8'); assertPins(manifests,lock,env.CV_INFERENCE_SHA);
  const output=path.resolve(env.CV_OUTPUT), checkout=await realpath(root), parent=await realpath(path.dirname(output));
  assert(path.relative(checkout,parent).startsWith(`..${path.sep}`),'evidence must be outside checkout');
  await mkdir(output); // An old receipt must never satisfy a new attempt.
  const receipt={schemaVersion:1,status:'building',sourceSha:env.CV_SOURCE_SHA,inferenceSha:env.CV_INFERENCE_SHA,
    lockSha256:createHash('sha256').update(lock).digest('hex'),platform:process.platform,runner:env.RUNNER_NAME,
    startedAt:new Date().toISOString(),cases:[]};
  const persist=()=>writeFile(path.join(output,'controller.json'),JSON.stringify(receipt,null,2)+'\n');
  const childEnv={...env,RUST_MIN_STACK:String(8*1024*1024),RUST_TEST_THREADS:'1'};
  async function execute(command,args,name,extra={}) {
    const log=await open(path.join(output,`${name}.log`),'wx');
    try {
      const child=spawn(command,args,metalChildOptions({cwd:root,env:{...childEnv,...extra},stdio:['ignore',log.fd,log.fd],windowsHide:true}));
      const code=await new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',(code,signal)=>signal?reject(new Error(`${name}: ${signal}`)):resolve(code));});
      assert.equal(code,0,`${name} failed; see retained log`);
    } finally {await log.close();}
  }
  try {
    await persist();
    if(!buildOnly) {
      const directory=await realpath(env.CV_FIXTURE_DIR);
      for(const file of requiredFixtures(process.platform)) assert((await stat(path.join(directory,file))).isFile(),`missing ${file}`);
      receipt.fixtures=await fixtureIdentity(directory);
      assert((await stat(env.ORT_DYLIB_PATH)).isFile());
      receipt.ortRuntime={path:env.ORT_DYLIB_PATH,sha256:await sha256(env.ORT_DYLIB_PATH)};
      const pinFile=path.join(root,'scripts/cv-fixture-manifest.json');
      const pins=JSON.parse(await readFile(pinFile,'utf8'));
      assert(env.CV_PROVISION_RECEIPT,'provisioning receipt required');
      receipt.provisioning=JSON.parse(await readFile(env.CV_PROVISION_RECEIPT,'utf8'));
      assert.equal(receipt.provisioning.manifestSha256,await sha256(pinFile));
      validateProvisioning(receipt.provisioning,receipt.fixtures,receipt.ortRuntime,pins,env.CV_SOURCE_SHA,env.CV_INFERENCE_SHA,process.platform);
      if(process.platform==='win32') receipt.hardware=(await execFile('nvidia-smi',['-i','1','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader'])).stdout.trim();
      else receipt.hardware=JSON.parse((await execFile('system_profiler',['SPHardwareDataType','-json'])).stdout).SPHardwareDataType?.map(({chip_type,physical_memory,machine_model})=>({chip_type,physical_memory,machine_model}));
      Object.assign(childEnv, {CV_FIXTURE_DIR:directory,
        SCENEWORKS_PERSON_DETECTOR_WEIGHTS:path.join(directory,process.platform==='darwin'?'yolo11m_fused_mlx.safetensors':'yolo11m.onnx'),
        SCENEWORKS_TEST_PERSON_IMAGE:path.join(directory,'people.jpg'),SCENEWORKS_TEST_POSE_IMAGE:path.join(directory,'people.jpg'),
        SCENEWORKS_DWPOSE_DET:path.join(directory,'yolox_m_8xb8-300e_humanart-c2c7a14a.onnx'),
        SCENEWORKS_DWPOSE_POSE:path.join(directory,'rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx'),
        SCENEWORKS_REALESRGAN_X2_ONNX:path.join(directory,'real_esrgan_x2.onnx'),SCENEWORKS_REALESRGAN_X4_ONNX:path.join(directory,'real_esrgan_x4.onnx')});
      // Use the bounded odd 513x33 fixture in the proof, irrespective of listener env.
      delete childEnv.SCENEWORKS_TEST_UPSCALE_IMAGE;
      await persist();
    }
    const build=['test','--locked','-p','sceneworks-worker','--lib','--no-run','--message-format=json-render-diagnostics','-j','1'];
    if(process.platform==='win32') build.push('--features','backend-candle');
    await execute('cargo',build,'build');
    const messages=(await readFile(path.join(output,'build.log'),'utf8')).split(/\r?\n/).flatMap(line=>{try{return [JSON.parse(line)];}catch{return [];}});
    const binaries=messages.filter(message=>message.reason==='compiler-artifact'&&message.target?.name==='sceneworks_worker'&&message.profile?.test&&message.executable);
    assert.equal(binaries.length,1,'one exact worker libtest executable required');
    const binary=binaries[0].executable; receipt.binary={path:binary,sha256:await sha256(binary)};
    for(const selected of cases) {
      await execute(binary,[selected.test,'--list','--ignored','--exact'],`${selected.name}-list`);
      ignoredTestRegistration(await readFile(path.join(output,`${selected.name}-list.log`),'utf8'),selected.test);
      if(buildOnly) continue;
      const directory=path.join(output,selected.name);await mkdir(directory);
      await execute(binary,[selected.test,'--ignored','--exact','--nocapture','--test-threads=1'],selected.name,
        {CV_PROOF_OUTPUT:directory,CV_EXPECTED_DEVICE:selected.device,...(selected.factor?{SCENEWORKS_TEST_UPSCALE_FACTOR:String(selected.factor)}:{})});
      const log=await readFile(path.join(output,`${selected.name}.log`),'utf8');
      assert.match(log,/test result: ok\. 1 passed; 0 failed; 0 ignored;/,'exactly one completed body required');
      const completed=JSON.parse(await readFile(path.join(directory,'case.json'),'utf8'));validateReceipt(completed,selected);
      const artifacts=[];
      for(const name of await readdir(directory)) artifacts.push({file:path.join(selected.name,name),sha256:await sha256(path.join(directory,name))});
      receipt.cases.push({...selected,completed,artifacts});await persist();
    }
    assert.equal(await git('status','--porcelain'),'','source changed during proof');
    receipt.status=buildOnly?'compiled-no-model-execution':'passed';
    if(!buildOnly) assert.equal(receipt.cases.length,cases.length);
    return receipt;
  } catch(error) {receipt.status='failed';receipt.error=error.message;throw error;}
  finally {receipt.finishedAt=new Date().toISOString();await persist();}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) {
  try {
    admission(process.env);
    if(process.env.CV_BUILD_ONLY==='true'||!(await enterMacMetalExecution(3600))) await run();
  } catch(error) {console.error(error);process.exitCode=1;}
}
