#!/usr/bin/env node
// sc27984: one explicitly selected real product job, separate from ordinary CI.
import assert from 'node:assert/strict';
import { spawn, execFile as execFileCb } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir, open, stat, realpath } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { stripJsoncComments } from './lib/jsonc.mjs';
import { assertPins } from './qwen-image-2-1-app-lifecycle.mjs';
import { enterMacMetalExecution, metalChildOptions } from './lib/apple-metal-execution-lock.mjs';
const execFile = promisify(execFileCb);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const CASES = ['image_mlx', 'image_cuda', 'audio', 'training'];
export function admission(env, platform = process.platform, { execute = true } = {}) {
  assert(CASES.includes(env.LOAD_PROGRESS_CASE), 'select exactly one known LOAD_PROGRESS_CASE');
  assert(['darwin', 'win32'].includes(platform), 'native macOS or Windows runner required');
  assert(/^[a-f0-9]{40}$/.test(env.LOAD_PROGRESS_SOURCE_SHA ?? ''), 'exact source SHA required');
  assert(/^[a-f0-9]{40}$/.test(env.LOAD_PROGRESS_INFERENCE_SHA ?? ''), 'exact inference SHA required');
  assert(env.LOAD_PROGRESS_OUTPUT && (platform==='win32'?path.win32:path.posix).isAbsolute(env.LOAD_PROGRESS_OUTPUT), 'absolute unique output directory required');
  if (platform === 'darwin') {
    if (execute) assert.equal(env.LOAD_PROGRESS_MAC_GO, 'true', "Michael's explicit dev Mac GPU go required");
    assert.notEqual(env.LOAD_PROGRESS_CASE, 'image_cuda', 'CUDA case needs Windows');
  } else {
    assert.notEqual(env.LOAD_PROGRESS_CASE, 'image_mlx', 'MLX case needs Mac');
    // Reuse the established real-weights workflow physical GPU0 restriction.
    for (const key of ['SCENEWORKS_GPU_ID', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES']) assert.equal(env[key], '0', `${key}=0 required`);
    assert.equal(env.CUDA_DEVICE_ORDER, 'PCI_BUS_ID');
  }
  return `tests::load_progress_acceptance::${env.LOAD_PROGRESS_CASE}`;
}
// Pure registered-dispatcher selector. It cannot start or load anything.
export function dispatchOptions(inputs, dispatchedSha) {
  assert.equal(inputs.profile, 'load-progress-acceptance');
  assert.equal(inputs.ref, dispatchedSha, 'ref must equal exact dispatched source SHA');
  assert(['mlx','candle'].includes(inputs.backend), 'backend must be mlx or candle');
  const options=JSON.parse(inputs.moss_options_json || '{}' );
  assert(options && typeof options==='object' && !Array.isArray(options), 'load progress options must be an object');
  assert(Object.keys(options).every(key=>['case','hf_hub_cache','build_only','mac_go'].includes(key)), 'unknown load progress option');
  for(const key of ['build_only','mac_go']) assert(options[key]===undefined || typeof options[key]==='boolean', `${key} must be boolean`);
  const backend=inputs.backend==='mlx'?'mlx':'cuda', platform=backend==='mlx'?'darwin':'win32';
  assert(typeof options.hf_hub_cache==='string' && (platform==='win32'?path.win32:path.posix).isAbsolute(options.hf_hub_cache), 'absolute existing HF hub cache required');
  assert(!/[\r\n]/.test(options.hf_hub_cache), 'invalid cache path');
  const selected={case:options.case,backend,hf_hub_cache:options.hf_hub_cache,build_only:options.build_only??true,mac_go:options.mac_go??false};
  admission({LOAD_PROGRESS_CASE:selected.case,LOAD_PROGRESS_SOURCE_SHA:inputs.ref,LOAD_PROGRESS_INFERENCE_SHA:inputs.inference_sha,LOAD_PROGRESS_OUTPUT:platform==='win32'?'C:/load-progress':'/tmp/load-progress',LOAD_PROGRESS_MAC_GO:String(selected.mac_go),SCENEWORKS_GPU_ID:'0',CUDA_VISIBLE_DEVICES:'0',NVIDIA_VISIBLE_DEVICES:'0',CUDA_DEVICE_ORDER:'PCI_BUS_ID'},platform,{execute:!selected.build_only});
  return selected;
}
// Exact manifest filenames only: metadata/stat, no recursive scans, hashes or model reads.
export async function cacheInventory(hub, selectedCase, manifest = null) {
  assert(hub && path.isAbsolute(hub), 'explicit absolute HF_HUB_CACHE required');
  const models=manifest ?? JSON.parse(stripJsoncComments(await readFile(path.join(root,'config/manifests/builtin.models.jsonc'),'utf8'))).models;
  const model=selectedCase==='audio'?'chatterbox_tts':'moss_image_sft';
  const entry=models.find(m=>m.id===model); assert(entry, 'shipped selected model missing');
  const rows=new Map();
  for(const source of entry.downloads) {
    assert(/^[a-f0-9]{40}$/.test(source.revision), 'immutable snapshot required');
    for(const file of source.files) {
      assert(typeof file==='string' && !file.includes('*') && !file.includes('..') && !path.isAbsolute(file),'exact manifest file required');
      const target=path.join(hub,`models--${source.repo.replaceAll('/','--')}`,'snapshots',source.revision,file);
      const row={repo:source.repo,revision:source.revision,file,path:target,component:source.componentId??null};
      try { const metadata=await stat(target); assert(metadata.isFile()&&metadata.size>0,'empty/non-file artifact'); Object.assign(row,{present:true,bytes:metadata.size}); }
      catch(error) { Object.assign(row,{present:false,error:error.code??error.message}); }
      rows.set(`${source.repo}@${source.revision}/${file}`,row);
    }
  }
  const files=[...rows.values()];
  return {model,hub,requiredFiles:files.length,presentFiles:files.filter(f=>f.present).length,totalPresentBytes:files.reduce((sum,f)=>sum+(f.bytes??0),0),missing:files.filter(f=>!f.present),files};
}
export function childEnvironment(env, state, evidence) {
  const child = { ...env };
  for (const key of Object.keys(child)) if (/TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY/i.test(key) || (key.startsWith('SCENEWORKS_') && !['SCENEWORKS_GPU_ID', 'SCENEWORKS_METAL_EXECUTION_LOCK_FD'].includes(key))) delete child[key];
  Object.assign(child, { HF_HUB_OFFLINE:'1', TRANSFORMERS_OFFLINE:'1', HF_DATASETS_OFFLINE:'1', LOAD_PROGRESS_STATE_ROOT:state, LOAD_PROGRESS_OUTPUT:evidence, CARGO_BUILD_JOBS:'2', CARGO_INCREMENTAL:'0', RUST_TEST_THREADS:'1' });
  return child;
}
export async function run(env = process.env, { checkOnly = false, buildOnly = false } = {}) {
  const filter = admission(env, process.platform, { execute: !checkOnly && !buildOnly });
  const output = path.resolve(env.LOAD_PROGRESS_OUTPUT);
  const relative = path.relative(root, output);
  assert(relative.startsWith(`..${path.sep}`), 'output must be outside checkout');
  await mkdir(path.dirname(output), { recursive:true });
  // Resolve the existing parent before creating output; a symlink must not hide the checkout.
  const parent = await realpath(path.dirname(output));
  assert(path.relative(root,parent).startsWith(`..${path.sep}`), 'output parent must physically be outside checkout');
  await mkdir(output); // EEXIST is a fail-closed fresh-attempt requirement.
  const evidence = path.join(output,'evidence'); await mkdir(evidence);
  const receipt = { schemaVersion:1, case:env.LOAD_PROGRESS_CASE, status:'preflight', sourceSha:env.LOAD_PROGRESS_SOURCE_SHA, inferenceSha:env.LOAD_PROGRESS_INFERENCE_SHA, startedAt:new Date().toISOString(), platform:process.platform, runner:env.RUNNER_NAME ?? null, test:filter };
  const persist = () => writeFile(path.join(evidence,'controller.json'), JSON.stringify(receipt,null,2)+'\n');
  const childEnv = childEnvironment(env,path.join(output,'state'),path.join(evidence,'case'));
  childEnv.CARGO_TARGET_DIR = path.resolve(env.LOAD_PROGRESS_TARGET_DIR ?? path.join(path.dirname(output), `sc27980-target-${env.LOAD_PROGRESS_SOURCE_SHA}-${process.platform}`));
  assert(path.relative(root,childEnv.CARGO_TARGET_DIR).startsWith(`..${path.sep}`),'build target must be outside checkout');
  const git = async (...args) => (await execFile('git',args,{cwd:root})).stdout.trim();
  try {
    assert.equal(await git('rev-parse','HEAD'),env.LOAD_PROGRESS_SOURCE_SHA);
    assert.equal(await git('status','--porcelain'),'','clean frozen source required');
    const manifests = await Promise.all(['Cargo.toml','crates/sceneworks-worker/Cargo.toml','crates/sceneworks-memory-adapter/Cargo.toml'].map(file=>readFile(path.join(root,file),'utf8')));
    const lock = await readFile(path.join(root,'Cargo.lock'),'utf8'); assertPins(manifests,lock,env.LOAD_PROGRESS_INFERENCE_SHA);
    const provenance = JSON.parse(await readFile(path.join(root,'config/inference-third-party-source.json'),'utf8'));
    assert.equal(provenance.inferenceRevision,env.LOAD_PROGRESS_INFERENCE_SHA,'source provenance must match final pin');
    receipt.lockSha256 = createHash('sha256').update(lock).digest('hex');
    // Record resolved mlx-rs source identity without requiring a particular pre-terminal pin.
    receipt.mlxSources = [...new Set([...lock.matchAll(/source = "(git\+https:\/\/github.com\/michaeltrefry\/mlx-rs[^"\n]+)"/g)].map(m=>m[1]))];
    if (process.platform === 'darwin') {
      const hardware = JSON.parse((await execFile('system_profiler',['SPHardwareDataType','-json'],{timeout:30000})).stdout);
      receipt.hardware = hardware.SPHardwareDataType?.map(({chip_type,physical_memory,machine_model})=>({chip_type,physical_memory,machine_model}));
    } else {
      receipt.hardware = (await execFile('nvidia-smi',['-i','0','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader'],{timeout:30000})).stdout.trim();
      receipt.device = {gpuId:'0',cudaVisibleDevices:'0',nvidiaVisibleDevices:'0',cudaDeviceOrder:'PCI_BUS_ID'};
    }
    await persist();
    if (checkOnly) { receipt.status='checked-no-model-execution'; return receipt; }
    if (!buildOnly) {
      receipt.cache=await cacheInventory(env.HF_HUB_CACHE,env.LOAD_PROGRESS_CASE); await persist();
      assert.equal(receipt.cache.missing.length,0,'provision missing frozen model files before acceptance; see controller.json');
    }
    const execute = async (command,args,name) => {
      const log = await open(path.join(evidence,`${name}.log`),'wx');
      try {
        const child = spawn(command,args,metalChildOptions({cwd:root,env:childEnv,stdio:['ignore',log.fd,log.fd],windowsHide:true}));
        const result = await new Promise((resolve,reject)=>{child.once('error',reject);child.once('exit',(code,signal)=>resolve({code,signal}));});
        assert.equal(result.code,0,`${name} failed (${JSON.stringify(result)}); inspect retained log`);
      } finally { await log.close(); }
    };
    const args = ['test','--locked','-p','sceneworks-rust-api','--bin','sceneworks-rust-api','--no-run','--message-format=json-render-diagnostics','-j','2'];
    if (process.platform === 'win32') args.push('--features','backend-candle');
    receipt.status='building'; await persist(); await execute('cargo',args,'build');
    const messages = (await readFile(path.join(evidence,'build.log'),'utf8')).split(/\r?\n/).flatMap(line=>{try{return [JSON.parse(line)];}catch{return [];}});
    const artifacts = messages.filter(v=>v.reason==='compiler-artifact' && v.target?.name==='sceneworks-rust-api' && v.profile?.test && v.executable);
    assert.equal(artifacts.length,1,'exact compiled test executable required');
    const executable = artifacts[0].executable; const metadata = await stat(executable);
    assert(metadata.isFile());
    receipt.binary = {path:executable,sha256:createHash('sha256').update(await readFile(executable)).digest('hex')};
    if (buildOnly) { receipt.status='compiled-no-model-execution'; return receipt; }
    receipt.status='running'; await persist();
    // One fresh libtest process, one exact ignored test; no cargo test zero-match success.
    await execute(executable,[filter,'--ignored','--exact','--nocapture','--test-threads=1'],'native');
    const result = JSON.parse(await readFile(path.join(evidence,'case','receipt.json'),'utf8'));
    assert.equal(result.case,env.LOAD_PROGRESS_CASE); assert.equal(result.status,'passed');
    receipt.status='passed'; return receipt;
  } catch(error) { receipt.status='failed'; receipt.error=error.message; throw error; }
  finally { receipt.finishedAt=new Date().toISOString(); await persist(); }
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  if (process.argv.includes('--help')) {
    process.stdout.write('Set LOAD_PROGRESS_CASE=image_mlx|image_cuda|audio|training, LOAD_PROGRESS_SOURCE_SHA, LOAD_PROGRESS_INFERENCE_SHA, LOAD_PROGRESS_OUTPUT. Mac requires LOAD_PROGRESS_MAC_GO=true; Windows uses established physical GPU0 visibility. --check-only validates source/device metadata; --build-only compiles ignored tests without model execution. Run each case once at the final clean pin pair.\n');
  } else {
    try {
      const checkOnly=process.argv.includes('--check-only'), buildOnly=process.argv.includes('--build-only');
      admission(process.env,process.platform,{execute:!checkOnly&&!buildOnly});
      if (!checkOnly && !buildOnly && await enterMacMetalExecution(3600)) { /* existing lease helper owns child */ }
      else await run(process.env,{checkOnly,buildOnly});
    } catch(error) { process.stderr.write(`${error.message}\n`); process.exitCode=1; }
  }
}
