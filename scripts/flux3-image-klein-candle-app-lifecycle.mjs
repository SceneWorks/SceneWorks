// sc-26235: admitted Windows ordinary API job -> native Candle worker -> saved asset.
// No downloads or model-library mutations. Reuses the shared lifecycle assertions unchanged.
import assert from 'node:assert/strict';
import { spawn, execFile as execFileCb } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, cp, readFile, writeFile, copyFile, open, realpath, stat } from 'node:fs/promises';
import { createServer } from 'node:net';
import path from 'node:path';
import os from 'node:os';
import { pathToFileURL, fileURLToPath } from 'node:url';
const execFile = promisify(execFileCb);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { imageBody, assertModel, assertAccepted, assertSavedAsset, assertServedPng, assertOwnedHealth, ownedApiController, boundApiUrl, bindOwnedEnv, attemptIdentity, redactSensitiveText, MODEL_REVISION, MODEL_REPO } = await import(pathToFileURL(path.join(root, 'scripts/flux3-image-klein-app-lifecycle.mjs')));
const { assertPins } = await import(pathToFileURL(path.join(root, 'scripts/qwen-image-2-1-app-lifecycle.mjs')));
export function candleChildSpawnOptions(env, logFd) {
  return { cwd: root, env, windowsHide: true, stdio: [Object.hasOwn(env, 'SCENEWORKS_WORKER_CHILD') ? 'pipe' : 'ignore', logFd, logFd] };
}

// JobSnapshot.backend has carried backend_label(settings.gpu_id) since June 2026.
// The device identity differs from the tensor backend; Candle is proven by its adapter.
export function candleSavedAssetContext(worker, { projectId, workerId }) {
  assert.equal(worker.id, workerId, 'owned worker identity mismatch');
  assert.equal(worker.gpuId, '1', 'this admission is physical GPU1 only');
  assert(worker.capabilities.includes('candle') && worker.capabilities.includes('gpu'));
  return { projectId, workerId, backend: worker.gpuId, adapter: 'candle_flux3' };
}

export async function runCandleAppLifecycle(env = process.env) {
assert.equal(process.platform, 'win32', 'Windows Candle controller required');
const source = env.FLUX3_APP_SOURCE_SHA;
assert(/^[a-f0-9]{40}$/.test(source ?? ''), 'exact admitted source SHA required');
const inference = env.FLUX3_APP_INFERENCE_SHA;
assert(/^[a-f0-9]{40}$/.test(inference ?? ''), 'exact admitted inference SHA required');
const binarySource = env.FLUX3_CANDLE_BINARY_SOURCE_SHA;
assert(/^[a-f0-9]{40}$/.test(binarySource ?? ''), 'exact compiled binary source SHA required');
const binaryHash = env.FLUX3_CANDLE_BINARY_SHA256;
assert(/^[a-f0-9]{64}$/.test(binaryHash ?? ''), 'exact compiled binary hash required');
const binary = path.resolve(env.FLUX3_CANDLE_APP_BINARY || path.join(root, '.local/sc26235-app-target/release/sceneworks-rust-api.exe'));
const snapshot = `E:/huggingface/hub/models--black-forest-labs--flux-3-image-klein/snapshots/${MODEL_REVISION}`;
const required = ['config.json', 'manifest.json', 'model.safetensors', 'ae.safetensors', 'LICENSE.md', 'NOTICE.md', 'text_encoder/config.json', 'text_encoder/tokenizer.json', 'text_encoder/tokenizer_config.json', 'text_encoder/model.safetensors.index.json', 'text_encoder/model-00001-of-00002.safetensors', 'text_encoder/model-00002-of-00002.safetensors'];
const output = path.join(os.tmpdir(), `SceneWorks-sc26235-cuda-app-${new Date().toISOString().replaceAll(/[:.]/g,'-')}-${process.pid}`);
const evidence = path.join(output, 'evidence'), privateLogs = path.join(output, 'private-logs');
await mkdir(evidence, { recursive: true }); await mkdir(privateLogs);
console.log(`EVIDENCE_ROOT=${evidence}`);
const identity = attemptIdentity(output), { state, workerId } = identity;
const receipt = { schema_version: 1, kind: 'flux3_image_klein_candle_app_lifecycle', status: 'running', phase: 'lifecycle', started_at: new Date().toISOString(), identity: { source_sha: source, inference_sha: inference, model_repo: MODEL_REPO, model_revision: MODEL_REVISION, worker_id: workerId, visibility: '1', physical_gpu_id: '1', logical_cuda_ordinal: 0 }, cache_policy: 'existing cache read-only; no relocate/download/install/reverify endpoints', events: [], timings_seconds: {}, cleanup: [] };
const secrets = Object.entries(process.env).filter(([k,v]) => /TOKEN|SECRET|PASSWORD|API_KEY/i.test(k) && v?.length >= 8).map(([,v])=>v);
const save = () => writeFile(path.join(evidence, 'receipt.json'), redactSensitiveText(JSON.stringify(receipt, null, 2)+'\n', secrets));
const hash = async file => { const h=createHash('sha256'); for await (const chunk of createReadStream(file)) { alive(); h.update(chunk); } return h.digest('hex'); };
const children=[], handles=[];
const controller=new AbortController(); let monitor, endpoint;
const deadline=setTimeout(()=>controller.abort(new Error('admitted runtime hard deadline (25 minutes)')),25*60_000);
const alive=()=> { if(controller.signal.aborted) throw controller.signal.reason; for(const c of children) assert(c.exitCode===null && c.signalCode===null,'owned service exited'); };
const sleep=async ms=>{ alive(); await new Promise(r=>setTimeout(r,ms)); alive(); };
const wait=async(fn,seconds,label)=>{const end=Date.now()+seconds*1000;while(Date.now()<end){alive();const x=await fn();if(x)return x;await sleep(1000);}throw new Error(`${label} deadline exceeded`);};
const smi=async args=>(await execFile('nvidia-smi',args,{timeout:5000,windowsHide:true})).stdout.trim();
const runners=async()=> /^"Runner\.Worker\.exe"/im.test((await execFile('tasklist.exe',['/FI','IMAGENAME eq Runner.Worker.exe','/FO','CSV','/NH'],{timeout:5000,windowsHide:true})).stdout);
const apps=async()=> (await smi(['--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'])).split(/\r?\n/).filter(Boolean).map(line=>{const [pid,uuid]=line.split(',').map(x=>x.trim());return {pid:Number(pid),uuid};}).filter(x=>Number.isFinite(x.pid));
const sample=async()=>{const text=await smi(['-i','1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits']);const [memory,util]=text.split(',').map(x=>Number(x.trim()));assert(Number.isFinite(memory)&&Number.isFinite(util),'GPU1 telemetry unavailable');receipt.gpu1_peak_device_used_mib=Math.max(receipt.gpu1_peak_device_used_mib??0,memory);return {memory,util};};
const launch=async(name,env)=>{alive();const log=await open(path.join(privateLogs,`${name}.log`),'wx');handles.push(log);const child=spawn(binary,[],candleChildSpawnOptions(env,log.fd));child.logName=name;child.done=new Promise((resolve,reject)=>{child.once('error',reject);child.once('exit',(code,signal)=>resolve({code,signal}));});child.done.catch(()=>controller.abort(new Error('owned service launch failed')));children.push(child);return child;};
try {
  const head=(await execFile('git',['rev-parse','HEAD'],{cwd:root})).stdout.trim();assert.equal(head,source);
  await execFile('git',['diff','--exit-code','HEAD','--'],{cwd:root});
  assertPins(await Promise.all(['Cargo.toml','crates/sceneworks-worker/Cargo.toml','crates/sceneworks-memory-adapter/Cargo.toml'].map(f=>readFile(path.join(root,f),'utf8'))),await readFile(path.join(root,'Cargo.lock'),'utf8'),inference);
  receipt.components=[];
  for(const file of required){const full=path.join(snapshot,file);const s=await stat(full);assert(s.isFile()&&s.size>0,`missing local required component ${file}`);receipt.components.push({file,bytes:s.size,sha256:await hash(full)});}
  assert(/FLUX.*Kommunity/i.test(await readFile(path.join(snapshot,'LICENSE.md'),'utf8')),'cached license mismatch');
  const inputDelta=(await execFile('git',['diff','--name-only',binarySource,source,'--','*.rs','Cargo.toml','Cargo.lock','**/Cargo.toml'],{cwd:root})).stdout.split(String.fromCharCode(10)).map(f=>f.trim()).filter(Boolean);assert(inputDelta.every(f=>f.includes('/tests/')&&f.endsWith('.rs')),'binary reuse refused: non-test Rust/Cargo input changed');receipt.identity.binary_input_delta=inputDelta;
  receipt.identity.binary_sha256=await hash(binary);assert.equal(receipt.identity.binary_sha256,binaryHash);receipt.identity.binary_compiled_source_sha=binarySource;receipt.identity.binary_reuse_reason='Verified merge delta contains only Rust test and scripts/fixtures; no non-test Rust/Cargo input changed';
  await mkdir(path.join(state,'data','projects'),{recursive:true});await mkdir(path.join(state,'credentials'));await cp(path.join(root,'config'),path.join(state,'config'),{recursive:true});
  const base={...process.env};for(const key of Object.keys(base))if(key.startsWith('SCENEWORKS_')||/TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY/i.test(key)||['HF_ENDPOINT','HF_XET_CACHE','TRANSFORMERS_CACHE','CARGO_TARGET_DIR'].includes(key))delete base[key];
  Object.assign(base,{HF_HOME:'E:/huggingface',HF_HUB_CACHE:'E:/huggingface/hub',HUGGINGFACE_HUB_CACHE:'E:/huggingface/hub',HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1',CUDA_VISIBLE_DEVICES:'1',CUDA_DEVICE_ORDER:'PCI_BUS_ID',SCENEWORKS_GPU_ID:'1',SCENEWORKS_BACKEND_CANDLE_ENABLED:'true',SCENEWORKS_BACKEND_MLX_ENABLED:'false',SCENEWORKS_CANDLE_REQUIRED:'1',SCENEWORKS_RESOLVED_CACHE_ENABLED:'false',SCENEWORKS_HUGGINGFACE_BASE_URL:'http://127.0.0.1:9',SCENEWORKS_WORKER_CHILD:'1',SCENEWORKS_POLL_SECONDS:'1',SCENEWORKS_HEARTBEAT_SECONDS:'2'});
  base.CUDA_PATH='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9';const inheritedPath=process.env.PATH;for(const key of Object.keys(base))if(key.toUpperCase()==='PATH')delete base[key];base.PATH=base.CUDA_PATH+'/bin;'+inheritedPath;
  const socket=createServer();await new Promise((resolve,reject)=>{socket.once('error',reject);socket.listen({host:'127.0.0.1',port:17935,exclusive:true},resolve);});await new Promise(resolve=>socket.close(resolve));
  assert(!await runners(),'Runner.Worker appeared: admission paused');
  const gpu=(await smi(['-i','1','--query-gpu=index,uuid,name','--format=csv,noheader'])).split(',').map(x=>x.trim());assert.equal(gpu[0],'1');receipt.identity.physical_gpu_uuid=gpu[1];receipt.identity.gpu_name=gpu.slice(2).join(',');
  const driverProbe = `Add-Type -TypeDefinition 'using System.Runtime.InteropServices; public static class CudaIdentity { [DllImport("nvcuda.dll")] public static extern int cuInit(uint flags); [DllImport("nvcuda.dll")] public static extern int cuDeviceGetCount(out int count); [DllImport("nvcuda.dll")] public static extern int cuDeviceGet(out int device, int ordinal); [DllImport("nvcuda.dll")] public static extern int cuDeviceGetUuid_v2([Out] byte[] uuid, int device); }'; if([CudaIdentity]::cuInit(0) -ne 0){throw "CUDA identity init failed"}; $n=0; if([CudaIdentity]::cuDeviceGetCount([ref]$n) -ne 0 -or $n -ne 1){throw "CUDA visibility identity mismatch"}; $d=0; if([CudaIdentity]::cuDeviceGet([ref]$d,0) -ne 0){throw "CUDA ordinal identity unavailable"}; $u=New-Object byte[] 16; if([CudaIdentity]::cuDeviceGetUuid_v2($u,$d) -ne 0){throw "CUDA UUID unavailable"}; $h=([BitConverter]::ToString($u)).Replace("-","").ToLowerInvariant(); Write-Output ("GPU-"+$h.Substring(0,8)+"-"+$h.Substring(8,4)+"-"+$h.Substring(12,4)+"-"+$h.Substring(16,4)+"-"+$h.Substring(20,12))`;
  const resolved = (await execFile('powershell.exe',['-NoProfile','-Command',driverProbe],{timeout:30000,windowsHide:true,env:{...process.env,CUDA_VISIBLE_DEVICES:'1',CUDA_DEVICE_ORDER:'PCI_BUS_ID'}})).stdout.trim();
  assert.equal(resolved.toLowerCase(),gpu[1].toLowerCase(),'masked logical CUDA0 UUID differs from physicalGPU1 before context creation');receipt.identity.masked_cuda0_uuid=resolved;
  const baseline=await sample();receipt.gpu1_baseline_device_used_mib=baseline.memory;assert(baseline.util<=5&&baseline.memory<=2048,'GPU1 admission utilization/memory bound exceeded');
  const gpuXml=await smi(['-i','1','-q','-x']);assert(gpuXml.includes('<processes>')&&gpuXml.includes(gpu[1]),'GPU1 process-type identity unavailable');
  const processTypes=[...gpuXml.matchAll(/<process_info>([\s\S]*?)<\/process_info>/g)].map(([,block])=>{const pid=Number(/<pid>([^<]+)<\/pid>/.exec(block)?.[1]);const type=/<type>([^<]+)<\/type>/.exec(block)?.[1]?.trim();assert(Number.isFinite(pid)&&pid>0&&['C','G','C+G'].includes(type),'GPU1 process-type record unavailable');return {pid,type};});
  const ownPids=new Set([process.pid,...children.map(c=>c.pid)]);assert(!processTypes.some(p=>p.type==='C'&&!ownPids.has(p.pid)),'GPU1 has a foreign pure-compute process');
  receipt.admission_guard={version:2,approved_correction:'WDDM graphics/C+G clients allowed; blanket compute-apps emptiness removed',gpu1_max_utilization_percent:5,gpu1_max_device_used_mib:2048,observed_utilization_percent:baseline.util,observed_device_used_mib:baseline.memory,foreign_pure_compute_count:0,runner_worker_count:0,observed_type_counts:Object.fromEntries(['C','G','C+G'].map(t=>[t,processTypes.filter(p=>p.type===t).length]))};
  await save();
  const started=Date.now();
  const apiEnv={...bindOwnedEnv(base,identity),SCENEWORKS_API_PORT:'17935',SCENEWORKS_RUST_WORKER_GPU_ID:'cpu'};
  assert(!await runners(),'Runner.Worker appeared immediately before API launch: admission paused');receipt.admission_guard.checked_at=new Date().toISOString();
  const api=await launch('api',apiEnv);receipt.identity.api_pid=api.pid;
  const url=await wait(async()=>boundApiUrl(await readFile(path.join(privateLogs,'api.log'),'utf8')),180,'owned API listener');assert.equal(new URL(url).port,'17935');
  endpoint=ownedApiController({url,state,api,controller});receipt.health=await wait(async()=>{try{return await endpoint.verify();}catch{alive();return false;}},180,'owned API readiness');await assertOwnedHealth(receipt.health,state);
  assert(!await runners(),'Runner.Worker appeared before worker launch: admission paused');
  const worker=await endpoint.startWorker(()=>launch('worker',{...bindOwnedEnv(base,identity,url),SCENEWORKS_WORKER_ONLY:'1'}));receipt.identity.worker_pid=worker.pid;
  let monitoring=false;monitor=setInterval(async()=>{if(monitoring)return;monitoring=true;try{if(await runners())throw new Error('Runner.Worker appeared: admission paused');await sample();const own=(await apps()).filter(a=>a.pid===worker.pid);if(own.length){assert(own.every(a=>a.uuid===gpu[1]),'owned worker bound to unexpected physical GPU');receipt.identity.actual_worker_gpu_uuid=own[0].uuid;}}catch(e){controller.abort(e);}finally{monitoring=false;}},2000);
  receipt.worker=await wait(async()=>{const list=await endpoint.call('GET','/api/v1/workers');const w=list.find(w=>w.id===workerId);if(!w||w.status!=='idle')return false;assert.equal(w.gpuId,'1');assert(w.capabilities.includes('candle')&&w.capabilities.includes('gpu'),'native Candle worker capabilities missing');return w;},180,'owned Candle worker');
  receipt.timings_seconds.startup=(Date.now()-started)/1000;
  const models=await endpoint.call('GET','/api/v1/models');const model=models.find(m=>m.id==='flux3_klein');assertModel(model);assert.equal(model.installState,'installed','complete local model not installed: STOP, no downloads');assert.equal(await realpath(model.installedPath),await realpath(snapshot));receipt.catalog_model=model;
  const project=await endpoint.call('POST','/api/v1/projects',{name:'sc-26235 admitted native Candle ordinary saved-image smoke'});assert((await realpath(project.path)).startsWith(await realpath(state)),'project escaped isolated state');
  const body=imageBody(project.id);receipt.request=body;const t=Date.now();const accepted=await endpoint.call('POST','/api/v1/image/jobs',body);assertAccepted(accepted,body);receipt.accepted_job=accepted;await save();
  const job=await wait(async()=>{const j=await endpoint.call('GET',`/api/v1/jobs/${accepted.id}`);const event={status:j.status,stage:j.stage,backend:j.backend,worker_id:j.workerId};if(JSON.stringify(event)!==JSON.stringify(receipt.events.at(-1))){receipt.events.push(event);await save();}assert(!['failed','canceled'].includes(j.status),'ordinary image job failed/canceled; inspect owned logs');return j.status==='completed'&&j;},600,'ordinary saved image job');receipt.timings_seconds.job=(Date.now()-t)/1000;
  if(!receipt.identity.actual_worker_gpu_uuid){const own=(await apps()).filter(a=>a.pid===worker.pid);assert(own.length&&own.every(a=>a.uuid===gpu[1]),'worker PID GPU UUID proof unavailable');receipt.identity.actual_worker_gpu_uuid=own[0].uuid;}
  assert.equal(receipt.identity.actual_worker_gpu_uuid,gpu[1]);
  const assets=await endpoint.call('GET',`/api/v1/projects/${project.id}/assets`);const asset=assets.find(a=>a.id===job.result?.assetIds?.[0]);assert(asset,'completed asset absent from library');
  const media=path.join(project.path,asset.file.path),sidecarFile=media.replace(/\.png$/,'.sceneworks.json');const sidecar=JSON.parse(await readFile(sidecarFile,'utf8'));
  receipt.completed_job=job;receipt.library_asset=asset;
  receipt.saved_asset=assertSavedAsset(job,asset,sidecar,candleSavedAssetContext(receipt.worker,{projectId:project.id,workerId}));
  const served=await endpoint.served(`/api/v1/projects/${project.id}/files/${asset.file.path}`);const saved=await readFile(media);assertServedPng(served.bytes,saved,body.width,body.height);
  await copyFile(media,path.join(evidence,'saved-asset.png'));await copyFile(sidecarFile,path.join(evidence,'saved-asset.sceneworks.json'));
  receipt.completed_job=job;receipt.library_asset=asset;receipt.saved_asset={...receipt.saved_asset,sha256:createHash('sha256').update(saved).digest('hex'),bytes:saved.length,width:body.width,height:body.height,api_status:served.status,saved_path:media};receipt.timings_seconds.total=(Date.now()-started)/1000;receipt.status='passed';
} catch(e) {receipt.status='failed';receipt.error=e.message;process.exitCode=1;} finally {
  clearTimeout(deadline);if(monitor)clearInterval(monitor);endpoint?.close();
  for(const child of children.reverse()){
    try{let forced=false;if(child.exitCode===null&&child.signalCode===null){child.stdin?.end();let timer;const exited=await Promise.race([child.done.then(()=>true),new Promise(resolve=>{timer=setTimeout(()=>resolve(false),20000);})]);clearTimeout(timer);if(!exited){forced=true;child.kill('SIGKILL');await child.done;}}receipt.cleanup.push({pid:child.pid,service:child.logName,exited:child.exitCode!==null||child.signalCode!==null,exit_code:child.exitCode,signal:child.signalCode,forced});}catch{receipt.status='failed';receipt.cleanup.push({pid:child.pid,service:child.logName,exited:false,exit_code:child.exitCode,signal:child.signalCode});process.exitCode=1;}
  }
  await Promise.all(handles.map(h=>h.close()));
  for(const name of ['api','worker'])try{await writeFile(path.join(evidence,`${name}.log`),redactSensitiveText(await readFile(path.join(privateLogs,`${name}.log`),'utf8'),secrets));}catch{}
  receipt.finished_at=new Date().toISOString();await save();
}
console.log(JSON.stringify({status:receipt.status,evidence,asset:receipt.saved_asset,timings_seconds:receipt.timings_seconds,gpu1_peak_device_used_mib:receipt.gpu1_peak_device_used_mib,error:receipt.error}));

return receipt;
}

// Resume evidence collection ONLY: this mode has no worker launch, POST or generation path.
export async function readbackCandleAppLifecycle(receiptFile, env = process.env) {
  assert.equal(process.platform, 'win32');
  const evidence = path.dirname(path.resolve(receiptFile));
  const original = JSON.parse(await readFile(receiptFile, 'utf8'));
  const state = path.dirname(original.health.directories.data);
  const source = env.FLUX3_APP_SOURCE_SHA;
  assert.equal((await execFile('git', ['rev-parse', 'HEAD'], { cwd: root })).stdout.trim(), source);
  await execFile('git', ['diff', '--exit-code', 'HEAD', '--'], { cwd: root });
  const delta = (await execFile('git', ['diff', '--name-only', original.identity.binary_compiled_source_sha, source, '--', '*.rs', 'Cargo.toml', 'Cargo.lock', '**/Cargo.toml'], { cwd: root })).stdout.split(String.fromCharCode(10)).map(f => f.trim()).filter(Boolean);
  assert(delta.every(f => f.includes('/tests/') && f.endsWith('.rs')), 'readback binary inputs changed');
  const binary = path.resolve(env.FLUX3_CANDLE_APP_BINARY || path.join(root, '.local/sc26235-app-target/release/sceneworks-rust-api.exe'));
  assert.equal(await hashReadbackFile(binary), original.identity.binary_sha256);
  assert.equal(original.identity.actual_worker_gpu_uuid, original.identity.physical_gpu_uuid);
  assert.equal(original.identity.physical_gpu_id, '1');
  const receipt = structuredClone(original);
  receipt.original_controller_error = receipt.error; delete receipt.error;
  receipt.status = 'running';
  receipt.readback = { source_sha: source, started_at: new Date().toISOString(), api_only: true, new_generation: false, binary_input_delta: delta, requests: [] };
  const secrets = Object.entries(env).filter(([k,v]) => /TOKEN|SECRET|PASSWORD|API_KEY/i.test(k) && v?.length >= 8).map(([,v]) => v);
  const save = () => writeFile(path.join(evidence, 'receipt.final.json'), redactSensitiveText(JSON.stringify(receipt, null, 2) + '\n', secrets));
  const childEnv = { ...env };
  for (const key of Object.keys(childEnv)) if (key.startsWith('SCENEWORKS_') || /TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY/i.test(key)) delete childEnv[key];
  Object.assign(childEnv, { HF_HOME: 'E:/huggingface', HF_HUB_CACHE: 'E:/huggingface/hub', HUGGINGFACE_HUB_CACHE: 'E:/huggingface/hub', HF_HUB_OFFLINE: '1', CUDA_VISIBLE_DEVICES: '-1', SCENEWORKS_GPU_ID: 'cpu', SCENEWORKS_BACKEND_CANDLE_ENABLED: 'false', SCENEWORKS_BACKEND_MLX_ENABLED: 'false', SCENEWORKS_CANDLE_REQUIRED: '0' });
  Object.assign(childEnv, bindOwnedEnv(childEnv, { state, workerId: original.identity.worker_id }), { SCENEWORKS_API_PORT: '17935' });
  const privateLog = path.join(path.dirname(evidence), 'private-logs', 'readback-api.log');
  const log = await open(privateLog, 'wx');
  const controller = new AbortController();
  const deadline = setTimeout(() => controller.abort(new Error('readback deadline exceeded')), 5 * 60_000);
  let api, endpoint;
  const started = Date.now();
  try {
    const socket = createServer(); await new Promise((resolve,reject) => { socket.once('error',reject); socket.listen({ host:'127.0.0.1', port:17935, exclusive:true },resolve); }); await new Promise(resolve => socket.close(resolve));
    api = spawn(binary, [], candleChildSpawnOptions(childEnv, log.fd));
    const done = new Promise((resolve,reject) => { api.once('error',reject); api.once('exit',(code,signal) => resolve({code,signal})); }); done.catch(() => {});
    const alive = () => { if (controller.signal.aborted) throw controller.signal.reason; assert(api.exitCode === null && api.signalCode === null, 'owned readback API exited'); };
    const wait = async (fn) => { const end = Date.now() + 180_000; while (Date.now() < end) { alive(); const value = await fn(); if (value) return value; await new Promise(resolve => setTimeout(resolve,500)); } throw new Error('readback readiness timed out'); };
    const url = await wait(async () => boundApiUrl(await readFile(privateLog,'utf8')));
    endpoint = ownedApiController({ url, state, api, controller });
    receipt.readback.health = await wait(async () => { try { return await endpoint.verify(); } catch { alive(); return false; } });
    const get = async route => { receipt.readback.requests.push({ method:'GET', route }); return endpoint.call('GET',route); };
    const job = await get(`/api/v1/jobs/${original.accepted_job.id}`);
    assert.equal(job.id,original.accepted_job.id); assert.equal(job.projectId,original.request.projectId);
    const project = await get(`/api/v1/projects/${job.projectId}`);
    assert(path.relative(await realpath(state),await realpath(project.path)).startsWith('data' + path.sep), 'readback project escaped original state');
    const assets = await get(`/api/v1/projects/${project.id}/assets`);
    const asset = assets.find(a => a.id === job.result?.assetIds?.[0]); assert(asset,'original job asset missing');
    const media = path.join(project.path,asset.file.path), sidecarFile = media.replace(/\.png$/,'.sceneworks.json');
    const sidecar = JSON.parse(await readFile(sidecarFile,'utf8'));
    receipt.saved_asset = assertSavedAsset(job,asset,sidecar,candleSavedAssetContext(original.worker,{projectId:project.id,workerId:original.identity.worker_id}));
    const route = `/api/v1/projects/${project.id}/files/${asset.file.path}`;
    receipt.readback.requests.push({method:'GET',route});
    const served = await endpoint.served(route), saved = await readFile(media);
    assertServedPng(served.bytes,saved,original.request.width,original.request.height);
    await copyFile(media,path.join(evidence,'saved-asset.png')); await copyFile(sidecarFile,path.join(evidence,'saved-asset.sceneworks.json'));
    receipt.saved_asset = { ...receipt.saved_asset, tensor_backend:'candle', sha256:createHash('sha256').update(saved).digest('hex'), bytes:saved.length, width:original.request.width, height:original.request.height, api_status:served.status, saved_path:media };
    receipt.completed_job = job; receipt.library_asset = asset; receipt.status = 'passed';
  } catch (error) { receipt.status = 'failed'; receipt.error = error.message; process.exitCode = 1; }
  finally {
    clearTimeout(deadline); endpoint?.close();
    if (api) { let forced=false; if(api.exitCode===null&&api.signalCode===null) { api.stdin?.end(); let timer; const exited=await Promise.race([onceReadbackExit(api).then(()=>true),new Promise(resolve=>{timer=setTimeout(()=>resolve(false),20000);})]);clearTimeout(timer);if(!exited){forced=true;api.kill('SIGKILL');await onceReadbackExit(api);} } receipt.readback.cleanup = { pid:api.pid, exited:api.exitCode!==null||api.signalCode!==null, exit_code:api.exitCode, signal:api.signalCode, forced, owned_api_only:true }; }
    await log.close();
    await writeFile(path.join(evidence,'readback-api.log'),redactSensitiveText(await readFile(privateLog,'utf8'),secrets));
    receipt.readback.finished_at = new Date().toISOString(); receipt.readback.elapsed_seconds = (Date.now()-started)/1000;
    await save();
  }
  console.log(JSON.stringify({ status:receipt.status, receipt:path.join(evidence,'receipt.final.json'), asset:receipt.saved_asset, error:receipt.error }));
  return receipt;
}
async function hashReadbackFile(file) { const h=createHash('sha256'); for await(const chunk of createReadStream(file)) h.update(chunk); return h.digest('hex'); }
async function onceReadbackExit(child) { if(child.exitCode!==null||child.signalCode!==null)return; await new Promise(resolve => child.once('exit',resolve)); }

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  const task = args.length === 0 ? runCandleAppLifecycle() : args.length === 2 && args[0] === '--readback' ? readbackCandleAppLifecycle(args[1]) : Promise.reject(new Error('expected no args or --readback <original receipt>'));
  task.catch(error => { console.error(error.message); process.exitCode = 1; });
}
