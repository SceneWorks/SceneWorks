// Purpose-specific native process, sampling and stopped-state transport for sc-26757.
// Importing this module does not start a process, device, browser or API.
import { spawn, execFile as callback } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdir, readFile, writeFile, readdir, lstat, cp, open } from 'node:fs/promises';
import { resolve, relative, dirname, basename, isAbsolute } from 'node:path';
import { deflateSync } from 'node:zlib';
import { VERSION, sha, digest, canonical, requireThat, save, json, safePath, decodeRgbPng, fileIdentity, treeIdentity, validateInferencePins, validateAdmission, assertRequestBinding, isDenoiseStep, Api } from './moss-edit-terminal.mjs';

const execFile = promisify(callback);
const optionalJson = path => json(path).catch(error => { if(error.code==='ENOENT')return null;throw error; });
export const R16_ORIGINAL_INVENTORY_SHA256 = '6a42624ebf84ffd06d54b7e4f845a5214bb485c12c9cf6211e475963b9488a0e';
export const H18_RETRY_TOOLING_SHA = '82c534d50b72c1a5984514b90548002da212d735';
export const H18_RETRY_RUN_ID = 40208210890;
export const H18_RETRY_ARTIFACT_ID = 12604596938;
export const H18_RETRY_INVENTORY_SHA256 = 'f849b52f0e7724e13b1f8d5a8c7f2797038e22d0d8dfb1ab8e6e2cfd9af03668';
export function validateDispatchEnvelope(raw,sourceSha,inferenceSha,dispatchSha) {
  requireThat(/^[a-f0-9]{40}$/.test(sourceSha)&&/^[a-f0-9]{40}$/.test(dispatchSha)&&/^[a-f0-9]{40}$/.test(inferenceSha),'MOSS dispatch requires exact product/tooling/inference commits');
  const value=JSON.parse(raw),required=['candidate_id','bundle_manifest_sha256','phase'],optional=['bundle_run_id','bundle_artifact_id','execution_run_id','execution_run_attempt','review_artifact_id','review_run_id','upload_path','upload_kind','tooling_sha','product_sha','original_inventory_path','original_inventory_sha256','native_run_id','native_run_attempt','retry_inventory_path','retry_inventory_sha256','retry_run_id','retry_artifact_id'];
  requireThat(value&&typeof value==='object'&&!Array.isArray(value)&&required.every(key=>typeof value[key]==='string'&&value[key])&&Object.keys(value).every(key=>[...required,...optional].includes(key)&&typeof value[key]==='string'),'MOSS dispatch JSON has missing, unknown or non-string fields');
  requireThat(/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(value.candidate_id)&&/^[a-f0-9]{64}$/.test(value.bundle_manifest_sha256),'MOSS dispatch candidate or manifest SHA is invalid');
  requireThat(['upload','execute','continue','verify'].includes(value.phase),'MOSS dispatch phase must be upload, execute, continue or verify');
  const empty=keys=>keys.every(key=>value[key]===undefined||value[key]==='');
  const repair=value.phase==='continue'||(value.phase==='verify'&&value.tooling_sha);
  requireThat(sourceSha===dispatchSha&&(!repair?empty(['tooling_sha','product_sha']):value.tooling_sha===dispatchSha&&/^[a-f0-9]{40}$/.test(value.product_sha??'')&&value.product_sha!==dispatchSha),'MOSS workflow commit must equal the exact product source or separately identify H17 product and H18 repair tooling');
  if(value.phase==='upload') {
    requireThat(['bundle','review'].includes(value.upload_kind)&&typeof value.upload_path==='string'&&value.upload_path.startsWith('/Users/michael/.codex/epic-runs/sc-26752/')&&resolve(value.upload_path)===value.upload_path,'MOSS upload requires a canonical directory inside the owned epic run');
    requireThat(empty(['bundle_run_id','bundle_artifact_id','execution_run_id','execution_run_attempt','review_artifact_id','review_run_id','tooling_sha','product_sha','original_inventory_path','original_inventory_sha256','native_run_id','native_run_attempt','retry_inventory_path','retry_inventory_sha256','retry_run_id','retry_artifact_id']),'MOSS upload cannot consume execution or prior artifact IDs');
  } else {
    requireThat([value.bundle_run_id,value.bundle_artifact_id].every(id=>/^[1-9][0-9]*$/.test(id??'')),'MOSS execution/verify requires exact bundle IDs');
    requireThat(empty(['upload_path','upload_kind']),'MOSS execution/verify cannot upload host files');
    if(value.phase==='verify') {
      requireThat([value.execution_run_id,value.review_artifact_id,value.review_run_id??value.bundle_run_id,value.execution_run_attempt??'1'].every(id=>/^[1-9][0-9]*$/.test(id??'')),'MOSS verify requires exact execution run, attempt and review artifact/run IDs');
      requireThat(empty(['original_inventory_path','original_inventory_sha256','retry_inventory_path','retry_inventory_sha256','retry_run_id','retry_artifact_id'])&&(!repair?empty(['native_run_id','native_run_attempt']):[value.native_run_id,value.native_run_attempt].every(id=>/^[1-9][0-9]*$/.test(id??''))),'MOSS repaired verify requires original native run/attempt and no host inventory path');
    } else if(value.phase==='continue') {
      requireThat([value.execution_run_id,value.execution_run_attempt??'1'].every(id=>/^[1-9][0-9]*$/.test(id??''))&&empty(['review_artifact_id','review_run_id','native_run_id','native_run_attempt']),'MOSS continuation requires one exact original execution attempt and no review/native override');
      requireThat(typeof value.original_inventory_path==='string'&&value.original_inventory_path.startsWith('/Users/michael/.codex/epic-runs/sc-26752/')&&resolve(value.original_inventory_path)===value.original_inventory_path&&value.original_inventory_sha256===R16_ORIGINAL_INVENTORY_SHA256,'MOSS continuation requires the exact sealed local original-evidence boundary');
      const retry=value.retry_inventory_path!==undefined||value.retry_inventory_sha256!==undefined;
      if(retry)requireThat(value.tooling_sha!==H18_RETRY_TOOLING_SHA&&value.retry_inventory_path?.startsWith('/Users/michael/.codex/epic-runs/sc-26752/')&&resolve(value.retry_inventory_path)===value.retry_inventory_path&&/^[a-f0-9]{64}$/.test(value.retry_inventory_sha256??'')&&(value.retry_inventory_sha256===H18_RETRY_INVENTORY_SHA256?empty(['retry_run_id','retry_artifact_id']):[value.retry_run_id,value.retry_artifact_id].every(id=>/^[1-9][0-9]*$/.test(id??''))),'MOSS retry needs a distinct tooling commit and exact sealed or downloaded failed-attempt artifact');
      else requireThat(empty(['retry_run_id','retry_artifact_id']),'MOSS continuation cannot name an artifact without a retry inventory');
    } else requireThat(empty(['execution_run_id','execution_run_attempt','review_artifact_id','review_run_id','tooling_sha','product_sha','original_inventory_path','original_inventory_sha256','native_run_id','native_run_attempt','retry_inventory_path','retry_inventory_sha256','retry_run_id','retry_artifact_id']),'MOSS execute cannot consume prior execution or review evidence');
  }
  const legacy=optional.slice(0,8),repairFields=optional.slice(8);
  return Object.fromEntries([...required,...legacy,...repairFields.filter(key=>value[key]!==undefined)].map(key=>[key,value[key]??(key==='execution_run_attempt'?'1':key==='review_run_id'&&value.phase==='verify'?value.bundle_run_id:'')]));
}
export const CUDA_UUID = 'GPU-e4b79931-7be6-f216-460a-f5405cfafffe';
export const CUDA_PCI = '00000000:C1:00.0';
export const delay = ms => new Promise(done=>setTimeout(done,ms));
export function contained(root,path) {
  const part=relative(resolve(root),resolve(path));
  requireThat(part && part!=='..' && !part.startsWith(`..${process.platform==='win32'?'\\':'/'}`) && !isAbsolute(part),'campaign path escaped its owned directory');
  return resolve(path);
}
export async function filesUnder(root) {
  const files=[];
  const visit=async dir=>{for(const name of (await readdir(dir)).sort()) {
    const path=resolve(dir,name),info=await lstat(path);requireThat(!info.isSymbolicLink(),'campaign transport refuses symlinks/junctions');
    if(info.isDirectory())await visit(path);else {requireThat(info.isFile(),'campaign transport accepts regular files only');files.push({file:relative(root,path).split('\\').join('/'),...await fileIdentity(path)});}
  }};await visit(resolve(root));return files;
}
export async function assertOriginalEvidence(root,manifest,{exact=false}={}) {
  requireThat(Array.isArray(manifest?.files)&&manifest.files.length>0&&new Set(manifest.files.map(item=>item.file)).size===manifest.files.length,'original evidence inventory has missing or duplicate paths');
  if(exact)requireThat(canonical(await filesUnder(root))===canonical(manifest.files),'original evidence bytes/files changed before continuation');
  else for(const item of manifest.files) {
    const path=safePath(root,item.file),info=await lstat(path);
    requireThat(info.isFile()&&!info.isSymbolicLink()&&canonical(await fileIdentity(path))===canonical({byteLength:item.byteLength,sha256:item.sha256}),`original evidence bytes changed: ${item.file}`);
  }
  return manifest.files.length;
}
export async function assertH18CheckpointFiles(root,checkpoint,{retained=false,archiveTag='h18'}={}) {
  const tails=checkpoint.appendOnlyTails;
  requireThat(Array.isArray(checkpoint.files)&&checkpoint.files.length>=300&&new Set(checkpoint.files.map(item=>item.file)).size===checkpoint.files.length&&Array.isArray(tails)&&new Set(tails.map(item=>item.file)).size===tails.length&&tails.length<=3&&tails.every(item=>archiveTag==='h18'?['api-continuation.log','journal-continuation.jsonl'].includes(item.file):/^(?:api-continuation-retry(?:-[2-9][0-9]*)?\.log|journal-continuation-retry(?:-[2-9][0-9]*)?\.jsonl|samples\/[a-z0-9-]+-retry-[2-9][0-9]*\.jsonl)$/.test(item.file))&&(archiveTag!=='h18'||canonical(tails.map(item=>item.file).sort())===canonical(['api-continuation.log','journal-continuation.jsonl'])),'prior artifact inventory or exact post-upload append-only logs changed');
  for(const item of checkpoint.files) {
    const tail=tails.find(entry=>entry.file===item.file);
    const relativeFile=retained?(item.file==='evidence/attempts/mlx/dispatch.lock'?`continuation-${archiveTag}-dispatch.lock`:item.file.startsWith('evidence/')?item.file.slice('evidence/'.length):`continuation-${archiveTag}-root/${item.file}`):item.file;
    const path=safePath(root,relativeFile),info=await lstat(path);
    requireThat(info.isFile()&&!info.isSymbolicLink(),`H18 checkpoint file is not regular: ${item.file}`);
    if(tail) {
      const bytes=await readFile(path);
      requireThat(tail.stoppedByteLength===bytes.length&&tail.stoppedSha256===sha(bytes)&&item.byteLength<=bytes.length&&item.sha256===sha(bytes.subarray(0,item.byteLength)),`H18 append-only artifact prefix or stopped bytes changed: ${item.file}`);
    } else requireThat(canonical(await fileIdentity(path))===canonical({byteLength:item.byteLength,sha256:item.sha256}),`H18 artifact bytes changed: ${item.file}`);
  }
  return checkpoint.files.length;
}
export async function continuationRemainingRows(evidenceRoot,plan) {
  const rows=plan.rows.filter(row=>row.backend==='mlx');
  requireThat(rows.length===25&&rows[6]?.id==='advanced-mlx-sft','original MLX frozen row order changed');
  for(const row of rows.slice(0,6))requireThat((await json(resolve(evidenceRoot,'rows',row.id,'receipt.json'))).execution==='VERIFIED',`original verified predecessor ${row.id} missing`);
  const failed=await json(resolve(evidenceRoot,'rows/advanced-mlx-sft/failure.json'));
  requireThat(failed.rowId==='advanced-mlx-sft'&&failed.message==='per-prompt negatives gained an unsupported broadcast field','original failed seventh row is not the reviewed checker defect');
  for(const row of rows.slice(7)) {
    for(const file of [`rows/${row.id}/receipt.json`,`rows/${row.id}/failure.json`,`attempts/mlx/${row.id}.start.json`]) {
      try{await lstat(safePath(evidenceRoot,file));requireThat(false,`later row ${row.id} was already touched`);}catch(error){if(error.code!=='ENOENT')throw error;}
    }
  }
  return rows.slice(7);
}
export async function retryRemainingRows(evidenceRoot,plan,lockSha256) {
  const rows=plan.rows.filter(row=>row.backend==='mlx'),id='advanced-mlx-sft-replay';
  requireThat(rows.length===25&&rows[6]?.id==='advanced-mlx-sft'&&rows[7]?.id===id,'H18 retry frozen MLX row order changed');
  for(const row of rows.slice(0,7))requireThat((await json(resolve(evidenceRoot,'rows',row.id,'receipt.json'))).execution==='VERIFIED',`verified predecessor ${row.id} missing`);
  const start=await json(resolve(evidenceRoot,'attempts/mlx',`${id}.start.json`));
  requireThat(start.id===id&&start.backend==='mlx'&&start.transport==='ui'&&start.lockSha256===lockSha256&&await readFile(resolve(evidenceRoot,'attempts/mlx/dispatch.lock'),'utf8')===id,'exact unresolved H18 UI intent/dispatch lock missing');
  for(const row of rows.slice(7))for(const file of [`rows/${row.id}/receipt.json`,`rows/${row.id}/failure.json`,`attempts/mlx/${row.id}.failure.json`,`attempts/mlx/${row.id}.resolution.json`,...(row.id===id?[]:[`attempts/mlx/${row.id}.start.json`])]) {
    try{await lstat(safePath(evidenceRoot,file));requireThat(false,`unsubmitted row ${row.id} already changed`);}catch(error){if(error.code!=='ENOENT')throw error;}
  }
  return rows.slice(7);
}
export async function submittedCanceledRemainingRows(evidenceRoot,plan,lockSha256,bindingSha256,checkpoint,root) {
  const rows=plan.rows.filter(row=>row.backend==='mlx');
  requireThat(rows.length===25,'frozen MLX row count changed');
  let index=0;
  while(index<rows.length&&await optionalJson(resolve(evidenceRoot,'rows',rows[index].id,'receipt.json'))) {
    requireThat((await json(resolve(evidenceRoot,'rows',rows[index].id,'receipt.json'))).execution==='VERIFIED',`prior row ${rows[index].id} lacks a verified receipt`);
    index++;
  }
  const row=rows[index];
  requireThat(index>=8&&row?.expected==='canceled'&&row.transport==='rest'&&row.cancelAtDenoise===true,'only a known submitted terminal cancellation may be adopted');
  const dir=resolve(evidenceRoot,'rows',row.id),attempt=resolve(evidenceRoot,'attempts/mlx'),start=await json(resolve(attempt,`${row.id}.start.json`));
  const [request,submitted,admitted,observed,response]=await Promise.all(['request.json','submission.json','admission.json','cancel-observation.json','cancel-response.json'].map(name=>json(resolve(dir,name))));
  requireThat(start.id===row.id&&start.backend==='mlx'&&start.lockSha256===lockSha256&&start.bindingSha256===bindingSha256&&start.requestSha256===digest(request)&&await readFile(resolve(attempt,'dispatch.lock'),'utf8')===row.id&&submitted?.id&&submitted.status==='queued'&&submitted.projectId===request.projectId,'canceled attempt lacks the exact known submitted REST request/lock');
  assertRequestBinding(request,submitted.payload);
  validateAdmission(admitted,await json(resolve(evidenceRoot,'protocol.lock.json')),'mlx');
  requireThat(observed.id===submitted.id&&observed.status==='running'&&isDenoiseStep(observed)&&response.id===submitted.id&&response.cancelRequested===true&&response.projectId===request.projectId&&response.workerId===observed.workerId&&Date.parse(response.updatedAt)>=Date.parse(observed.updatedAt),'original denoise cancellation and native acknowledgement differ');
  for(const file of [`rows/${row.id}/receipt.json`,`rows/${row.id}/failure.json`,`attempts/mlx/${row.id}.failure.json`,`attempts/mlx/${row.id}.resolution.json`])requireThat(!await optionalJson(resolve(evidenceRoot,file)),`submitted cancellation ${row.id} was already resolved/failed`);
  for(const later of rows.slice(index+1))for(const file of [`rows/${later.id}/receipt.json`,`rows/${later.id}/failure.json`,`attempts/mlx/${later.id}.start.json`])requireThat(!await optionalJson(resolve(evidenceRoot,file)),`later row ${later.id} was already touched`);
  const samples=checkpoint.files.filter(item=>new RegExp(`^samples/${row.id}-retry-[2-9][0-9]*\\.jsonl$`).test(item.file));
  requireThat(samples.length===1,'original submitted cancellation raw sample is not sealed in the failed artifact');
  const sampleFile=samples[0].file;
  const lines=(await readFile(safePath(root,sampleFile),'utf8')).trimEnd().split('\n').map(line=>JSON.parse(line));
  requireThat(lines.length>=3&&lines[0].phase==='baseline'&&lines.at(-1).phase==='drain'&&lines.some(item=>item.phase==='active'&&Date.parse(item.at)>=Date.parse(submitted.startedAt??submitted.createdAt)&&Date.parse(item.at)<=Date.parse(response.updatedAt))&&lines.every(item=>item.device?.uuid===admitted.device.uuid&&item.raw),'original whole-device canceled-job trace is incomplete or changed');
  return {remaining:rows.slice(index+1),recovery:{rowId:row.id,jobId:submitted.id,requestSha256:digest(request),submissionSha256:digest(submitted),attemptSha256:digest(start),sampleFile}};
}
export function validateFailedUiCapture(capture,lockSha256,{historical=null}={}) {
  const oldSelector=historical==='h18'&&capture.error?.includes('strict mode violation')&&capture.error.includes('Download portable workflow (.json)');
  const oldObserver=historical==='h19'&&capture.error==='observed upload lacks multipart file bytes'&&capture.actions?.some(item=>item.action==='drop-workflow-file')&&!capture.actions.some(item=>item.action==='click'&&/^Generate(?: edit| [0-9]+ images?)?$/.test(item.control));
  requireThat(capture.rowId==='advanced-mlx-sft-replay'&&capture.lockSha256===lockSha256&&typeof capture.error==='string'&&capture.error.length>0&&(oldSelector||oldObserver||(!historical&&capture.submissionPhase==='before-generate-attempt'))&&Array.isArray(capture.submissions)&&capture.submissions.length===0&&Array.isArray(capture.imageJobPosts)&&capture.imageJobPosts.length===0,'failed UI capture was not a proven pre-submission attempt');
  return true;
}
export function continuationRowDisposition(row,remaining,{retry=false}={}) {
  const run=remaining.some(item=>item.id===row.id);
  return {run,reuseUiIntent:run&&retry&&row.id==='advanced-mlx-sft-replay'};
}
export async function sourceIdentity(config) {
  const sources={sceneworks:treeIdentity(config.sourceRoot),inference:treeIdentity(config.inferenceRoot)};
  requireThat(sources.sceneworks.head===config.sourceSha && sources.inference.head===config.inferenceSha,'Actions checkout differs from the exact paired dispatch sources');
  const manifests=await Promise.all(['Cargo.toml','crates/sceneworks-worker/Cargo.toml','crates/sceneworks-memory-adapter/Cargo.toml'].map(path=>readFile(resolve(config.sourceRoot,path),'utf8')));
  validateInferencePins(manifests,await readFile(resolve(config.sourceRoot,'Cargo.lock'),'utf8'),config.inferenceSha);
  // SceneWorks is the stationary permanent-pin feature candidate: its campaign
  // must pass before that feature merges to main. Only inference has landed.
  await execFile('git',['-C',config.inferenceRoot,'merge-base','--is-ancestor',config.inferenceSha,'origin/main']);
  return sources;
}
export function childEnvironment(config,token) {
  const env={...process.env};
  for(const name of Object.keys(env))if(name.startsWith('SCENEWORKS_')||/TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY/.test(name))delete env[name];
  const state=resolve(config.stateRoot),cuda=config.backend==='cuda';
  Object.assign(env,{SCENEWORKS_DATA_DIR:resolve(state,'data'),SCENEWORKS_CONFIG_DIR:resolve(state,'config'),SCENEWORKS_CREDENTIALS_DIR:resolve(state,'credentials'),
    SCENEWORKS_JOBS_DB_PATH:resolve(state,'data','cache','jobs.db'),SCENEWORKS_ACCESS_TOKEN:token,
    SCENEWORKS_API_HOST:'127.0.0.1',SCENEWORKS_API_PORT:String(config.port),SCENEWORKS_API_URL:`http://127.0.0.1:${config.port}`,
    SCENEWORKS_WORKER_ID:config.workerId,SCENEWORKS_GPU_ID:cuda?'1':'mlx',
    SCENEWORKS_BACKEND_MLX_ENABLED:String(!cuda),SCENEWORKS_BACKEND_CANDLE_ENABLED:String(cuda),SCENEWORKS_POLL_SECONDS:'1',SCENEWORKS_HEARTBEAT_SECONDS:'2',
    HF_HOME:config.hfHome,HF_HUB_CACHE:resolve(config.hfHome,'hub'),HUGGINGFACE_HUB_CACHE:resolve(config.hfHome,'hub'),HF_XET_CACHE:resolve(config.hfHome,'xet'),HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1'});
  if(cuda)Object.assign(env,{CUDA_DEVICE_ORDER:'PCI_BUS_ID',CUDA_VISIBLE_DEVICES:CUDA_UUID});else delete env.CUDA_VISIBLE_DEVICES;
  // This is an observation journal, never hardware admission/ownership authority.
  if(config.observationJournal)env.SCENEWORKS_MOSS_OBSERVATION_JOURNAL=config.observationJournal;
  if(config.candidateId)env.SCENEWORKS_MOSS_CANDIDATE_ID=config.candidateId;
  env.SCENEWORKS_MOSS_SOURCE_SHA=config.sourceSha;env.SCENEWORKS_MOSS_INFERENCE_SHA=config.inferenceSha;
  return env;
}
// The API owns and drains this CPU loop; its identity must not replace the GPU registration.
export function utilityWorkerEnvironment(config) {
  return {SCENEWORKS_RUN_UTILITY_INPROCESS:'1',SCENEWORKS_RUST_WORKER_GPU_ID:'cpu',SCENEWORKS_WORKER_ID:`${config.workerId}-utility`,SCENEWORKS_UTILITY_WORKERS:'1'};
}
// Only standalone worker processes own the Windows stdin-EOF shutdown contract.
export function standaloneWorkerEnvironment() {
  return {SCENEWORKS_WORKER_ONLY:'1',SCENEWORKS_WORKER_CHILD:'1'};
}
// Whitelist bounded readiness fields; never persist tokens, environment or response bodies.
export function stackReadinessSnapshot({health,workers=[],children=[],workerId,httpStatus=null}) {
  const text=value=>typeof value==='string'?value.slice(0,200):null;
  return {version:VERSION,at:new Date().toISOString(),expectedWorkerId:workerId,httpStatus,
    health:{status:text(health?.status),readinessStatus:text(health?.readiness?.status)},
    workers:workers.slice(0,16).map(worker=>({id:text(worker.id),gpuId:text(worker.gpuId),status:text(worker.status),
      capabilities:Array.isArray(worker.capabilities)?worker.capabilities.slice(0,64).map(text):[]})),
    children:children.slice(0,2).map((child,index)=>({role:index===0?'api':'worker',pid:child.pid??null,exitCode:child.exitCode??null,signalCode:child.signalCode??null,startError:Boolean(child.startError)}))};
}
export function readyUtilityWorker(workers,workerId) {
  return workers.find(worker=>worker.id===`${workerId}-utility`&&worker.gpuId==='cpu'&&worker.status==='idle'&&worker.capabilities?.includes('model_download'));
}
export async function command(command,args,{cwd,env=process.env,log,timeoutMs=120*60*1000}={}) {
  await mkdir(dirname(log),{recursive:true,mode:0o700});const stream=await open(log,'wx',0o600);
  const child=spawn(command,args,{cwd,env,stdio:['ignore',stream.fd,stream.fd],windowsHide:true});
  let timer;try {
    const code=await new Promise((done,fail)=>{child.once('error',fail);child.once('exit',(code,signal)=>signal?fail(Error(`owned command exited by ${signal}`)):done(code));timer=setTimeout(()=>{child.kill();fail(Error('owned command deadline exceeded'));},timeoutMs);});
    requireThat(code===0,`owned command failed (${basename(command)}, exit ${code}); inspect its retained log`);
  } finally {clearTimeout(timer);await stream.close();}
}
export async function nativeDevice(backend,execute=execFile) {
  if(backend==='cuda') {
    const {stdout}=await execute('nvidia-smi',[`--id=${CUDA_UUID}`,'--query-gpu=index,uuid,pci.bus_id,name,memory.used','--format=csv,noheader,nounits'],{timeout:10000,windowsHide:true});
    const rows=stdout.trim().split(/\r?\n/);requireThat(rows.length===1,'CUDA UUID probe returned another device count');
    const [index,uuid,pci,name,memory]=rows[0].split(',').map(item=>item.trim());
    requireThat(index==='1'&&uuid===CUDA_UUID&&pci.toUpperCase()===CUDA_PCI,'terminal campaign is restricted to physical GPU1 UUID/PCI');
    const usedBytes=Number(memory)*1024*1024;requireThat(Number.isFinite(usedBytes)&&usedBytes>=0,'invalid whole-device CUDA memory observation');
    return {device:{uuid,pciOrRegistryId:pci,name,physicalIndex:1,logicalIndex:0},usedBytes,raw:stdout};
  }
  requireThat(backend==='mlx','unknown native backend');
  const arm=(await execute('sysctl',['-n','hw.optional.arm64'],{timeout:10000})).stdout.trim();requireThat(arm==='1','native MLX requires Apple silicon');
  const registry=(await execute('ioreg',['-rd1','-c','IOPlatformExpertDevice'],{timeout:10000})).stdout;
  const uuid=registry.match(/"IOPlatformUUID"\s*=\s*"([^"]+)"/)?.[1];requireThat(uuid,'actual Apple platform registry identity missing');
  const vm=(await execute('vm_stat',[],{timeout:10000})).stdout,total=Number((await execute('sysctl',['-n','hw.memsize'],{timeout:10000})).stdout);
  const page=Number(vm.match(/page size of (\d+) bytes/)?.[1]);
  const pages=['free','inactive','speculative'].map(name=>Number(vm.match(new RegExp(`Pages ${name}:\\s+(\\d+)\\.`))?.[1]));
  const usedBytes=total-pages.reduce((sum,n)=>sum+n,0)*page;
  requireThat(Number.isSafeInteger(total)&&total>0&&Number.isSafeInteger(usedBytes)&&usedBytes>=0&&usedBytes<=total,'invalid sampled Apple unified-memory usage');
  return {device:{uuid,pciOrRegistryId:uuid,name:'Apple Silicon unified memory',platform:'apple-silicon'},usedBytes,raw:{registry,vm,total,page,method:'whole-host unified physical memory: total minus free/inactive/speculative; includes other processes'}};
}
export async function sampler({backend,file,observe=()=>nativeDevice(backend),intervalMs=250}) {
  const samples=[];let stopped=false,pending=Promise.resolve(),timer,error;
  await mkdir(dirname(file),{recursive:true});const handle=await open(file,'wx',0o600);
  const capture=async phase=>{const actual=await observe(),sample={at:new Date().toISOString(),phase,usedBytes:actual.usedBytes};samples.push(sample);await handle.write(`${JSON.stringify({...sample,device:actual.device,raw:actual.raw})}\n`);return actual.device;};
  const device=await capture('baseline');
  const schedule=()=>{if(!stopped)timer=setTimeout(()=>{pending=capture('active').catch(failure=>{error=failure;stopped=true;}).finally(schedule);},intervalMs);};schedule();
  return {async stop(jobId,samplerSha256){stopped=true;clearTimeout(timer);await pending;try{if(error)throw error;const last=await capture('drain');requireThat(last.uuid===device.uuid,'sampled another physical device');return {version:VERSION,jobId,samplerSha256,device,samples};}finally{await handle.close();}}};
}
export async function startStack(config,token) {
  requireThat(token&&config.workerId&&Number.isInteger(config.port),'native stack requires actual token/worker/port');
  await sourceIdentity(config);await nativeDevice(config.backend);await mkdir(resolve(config.stateRoot,'data'),{recursive:true,mode:0o700});await mkdir(resolve(config.stateRoot,'credentials'),{recursive:true,mode:0o700});
  if(config.continuation) {
    requireThat(config.backend==='mlx'&&config.continuation.name==='r16-checker-recovery'&&/^[a-f0-9]{40}$/.test(config.continuation.toolingSha??''),'only the reviewed stopped MLX continuation may reopen state');
    requireThat((await filesUnder(resolve(config.stateRoot,'config'))).length>0,'original stopped stack configuration missing');
  } else await cp(resolve(config.sourceRoot,'config'),resolve(config.stateRoot,'config'),{recursive:true,errorOnExist:true,force:false});
  const env=childEnvironment(config,token),children=[],handles=[];
  const continuationSuffix=config.continuation?(config.continuation.retry?`-continuation-retry${config.continuation.retry.ordinal>1?`-${config.continuation.retry.ordinal}`:''}`:'-continuation'):'';
  const launch=async(name,extra)=>{const handle=await open(resolve(config.root,`${name}${continuationSuffix}.log`),'wx',0o600);handles.push(handle);const child=spawn(config.apiBinary,[],{cwd:config.sourceRoot,env:{...env,...extra},stdio:[name.startsWith('worker')?'pipe':'ignore',handle.fd,handle.fd],windowsHide:true});child.once('error',error=>{child.startError=error;});children.push(child);return child;};
  const api=await launch('api',utilityWorkerEnvironment(config)),client=new Api(env.SCENEWORKS_API_URL,token);let worker=await launch('worker',standaloneWorkerEnvironment()),restarts=0;
  const stopOne=async(child,isWorker)=>{if(child.exitCode!==null||child.signalCode!==null)return;let timer;try {
    await new Promise((done,fail)=>{child.once('exit',done);timer=setTimeout(()=>fail(Error('owned native process did not stop; state transfer forbidden')),30000);if(process.platform==='win32'&&isWorker)child.stdin.end();else child.kill('SIGTERM');});
  }finally{clearTimeout(timer);}};
  const stop=async()=>{await stopOne(worker,true);await stopOne(api,false);for(const handle of handles)await handle.close();};
  const restartIdleWorker=async reason=>{
    const deadline=Date.now()+120000;let idle=false;
    while(Date.now()<deadline){const current=(await client.request('/api/v1/workers')).find(item=>item.id===config.workerId);if(current?.status==='idle'&&!current.currentJobId){idle=true;break;}await delay(250);}
    requireThat(idle,'owned worker never drained to idle; refuse cache restart');
    requireThat(worker.exitCode===null&&worker.signalCode===null&&!worker.startError,'owned worker exited unexpectedly; do not silently restart a failed campaign');
    const oldPid=worker.pid;await stopOne(worker,true);requireThat(worker.exitCode!==null||worker.signalCode!==null,'owned worker cache process still running');
    const oldExit={code:worker.exitCode,signal:worker.signalCode};
    await sourceIdentity(config);await nativeDevice(config.backend);config.observationJournal=resolve(config.root,`journal-${config.continuation?(config.continuation.retry?`continuation-retry${config.continuation.retry.ordinal>1?`-${config.continuation.retry.ordinal}`:''}-`:'continuation-'):''}${++restarts}.jsonl`);
    worker=await launch(`worker-restart-${restarts}`,{...standaloneWorkerEnvironment(),SCENEWORKS_MOSS_OBSERVATION_JOURNAL:config.observationJournal});
    return {reason,oldPid,oldExit,newPid:worker.pid,journal:config.observationJournal,source:await sourceIdentity(config),workerId:config.workerId,at:new Date().toISOString(),meaning:'actual idle owned worker process exited; its native resident caches cannot be reused by reentry'};
  };
  try {
    const deadline=Date.now()+180000;let ready,utility,lastHealth,lastWorkers=[],lastHttpStatus=null;
    while(Date.now()<deadline) {
      for(const child of children)requireThat(!child.startError&&child.exitCode===null&&child.signalCode===null,'source-built API/worker exited before readiness');
      try {const health=await client.request('/api/v1/health'),workers=await client.request('/api/v1/workers');lastHealth=health;lastWorkers=workers;lastHttpStatus=null;ready=workers.find(item=>item.id===config.workerId);utility=readyUtilityWorker(workers,config.workerId);
        if(health.status==='ok'&&health.readiness?.status==='ready'&&ready?.gpuId===(config.backend==='cuda'?'1':'mlx')&&ready.status==='idle'&&utility)break;
      }catch(error){lastHttpStatus=Number.isInteger(error.status)?error.status:null;}await delay(500);
    }
    const registered=ready?.gpuId===(config.backend==='cuda'?'1':'mlx')&&ready.status==='idle'&&utility;
    if(!registered)await save(resolve(config.root,'startup-readiness.json'),stackReadinessSnapshot({health:lastHealth,workers:lastWorkers,children,workerId:config.workerId,httpStatus:lastHttpStatus}));
    requireThat(registered,'actual native/CPU model_download workers never registered on their bound identities; inspect startup-readiness.json');
    await save(resolve(config.root,config.continuation?`stack${continuationSuffix}.json`:'stack.json'),{version:VERSION,backend:config.backend,source:await sourceIdentity(config),binary:await fileIdentity(config.apiBinary),apiUrl:client.base.href,worker:ready,utilityWorker:utility,pids:{api:api.pid,worker:worker.pid},at:new Date().toISOString(),...(config.continuation?{tooling:config.continuation.toolingSha}:{})});
    return {api:client,children,stop,restartIdleWorker,async restartWorker(canceledJobId){
      const job=await client.request(`/api/v1/jobs/${encodeURIComponent(canceledJobId)}`);
    requireThat(job.status==='canceled'&&job.workerId===config.workerId&&!(job.result?.assets?.length)&&!(job.result?.assetWrites?.length),'owned cancellation is not drained; do not restart');
      return {canceledJobId,...await restartIdleWorker('canceled-job-reentry')};
    }};
  }catch(error){await stop();throw error;}
}
export async function checkpointDatabases(dataRoot,Database) {
  if(!Database)({DatabaseSync:Database}=await import('node:sqlite'));
  const checks=[];
  for(const entry of await filesUnder(dataRoot))if(entry.file.endsWith('.db')) {
    const db=new Database(resolve(dataRoot,entry.file));try {
      const checkpoint=db.prepare('PRAGMA wal_checkpoint(TRUNCATE)').get();requireThat(checkpoint.busy===0,'owned SQLite database is still busy; never copy live state');
      const integrity=db.prepare('PRAGMA integrity_check').get();requireThat(Object.values(integrity)[0]==='ok','owned SQLite integrity failed');checks.push({file:entry.file,checkpoint,integrity});
    }finally{db.close();}
  }requireThat(checks.some(item=>item.file==='cache/jobs.db'),'actual jobs database missing from stopped handoff');return checks;
}
export function relocateRegistry(registry,oldDataRoot,newDataRoot) {
  requireThat(Array.isArray(registry),'actual recent-projects.json must be an array');
  const oldPrefix=oldDataRoot.replace(/\\/g,'/').replace(/\/$/,'')+'/';
  return registry.map(item=>{requireThat(typeof item.id==='string'&&typeof item.path==='string','owned project registry entry lacks its ID/path');const path=item.path.replace(/\\/g,'/');requireThat(path.startsWith(oldPrefix)&&!path.slice(oldPrefix.length).split('/').includes('..'),'handoff includes a project outside owned state');return {...item,path:resolve(newDataRoot,...path.slice(oldPrefix.length).split('/'))};});
}
export async function exportState(config,out,Database) {
  const dataRoot=resolve(config.stateRoot,'data'),checks=await checkpointDatabases(dataRoot,Database);
  await mkdir(out,{mode:0o700});await cp(dataRoot,resolve(out,'data'),{recursive:true,errorOnExist:true,force:false});
  const files=await filesUnder(resolve(out,'data'));
  requireThat(!files.some(item=>item.file.endsWith('-wal')&&item.byteLength>0),'checkpoint left a nonempty SQLite WAL');
  await save(resolve(out,'handoff.json'),{version:VERSION,projectId:config.projectId,sourceDataRoot:dataRoot,sources:await sourceIdentity(config),checks,files});
}
export async function importState(config,from) {
  const manifest=await json(resolve(from,'handoff.json'));requireThat(manifest.version===VERSION&&manifest.projectId===config.projectId&&canonical(manifest.sources)===canonical(await sourceIdentity(config)),'cross-backend state differs from the frozen paired sources/project');
  requireThat(canonical(await filesUnder(resolve(from,'data')))===canonical(manifest.files),'stopped-state handoff bytes changed');
  const target=resolve(config.stateRoot,'data');await cp(resolve(from,'data'),target,{recursive:true,errorOnExist:true,force:false});
  const path=resolve(target,'recent-projects.json'),before=await json(path),after=relocateRegistry(before,manifest.sourceDataRoot,target);
  requireThat(after.some(item=>item.id===config.projectId),'original campaign project missing from transferred registry');
  await writeFile(path,`${JSON.stringify(after,null,2)}\n`,{mode:0o600});
  await save(resolve(config.root,'state-relocation.json'),{version:VERSION,manifestSha256:digest(manifest),before,after,changed:'owned registry location fields only; project/job/asset/payload/recipe bytes preserved'});
}
// Review material only: actual decoded pixels, no generated comparison image or verdict.
export function rgbPng(width,height,pixels) {
  requireThat(Number.isSafeInteger(width)&&width>0&&Number.isSafeInteger(height)&&height>0&&width*height<=64*1024*1024&&pixels.length===width*height*3,'bounded RGB review raster required');
  const chunk=(type,data)=>{const body=Buffer.concat([Buffer.from(type),data]);let crc=0xffffffff;
    for(const byte of body){crc^=byte;for(let i=0;i<8;i++)crc=(crc>>>1)^((crc&1)?0xedb88320:0);}
    const length=Buffer.alloc(4),tail=Buffer.alloc(4);length.writeUInt32BE(data.length);tail.writeUInt32BE((crc^0xffffffff)>>>0);return Buffer.concat([length,body,tail]);};
  const header=Buffer.alloc(13);header.writeUInt32BE(width);header.writeUInt32BE(height,4);header[8]=8;header[9]=2;
  const rows=Buffer.alloc((width*3+1)*height);for(let y=0;y<height;y++)pixels.copy(rows,y*(width*3+1)+1,y*width*3,(y+1)*width*3);
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',header),chunk('IDAT',deflateSync(rows)),chunk('IEND',Buffer.alloc(0))]);
}
export async function retainQualityPairs(state,row) {
  requireThat(row.anchors&&row.output==='images','visual review pairs belong to the eight image anchors');
  const directory=resolve(state.root,'rows',row.id),receipt=await json(resolve(directory,'receipt.json')),inputs=await json(resolve(state.root,'input-identities.json'));
  const outputBytes=await readFile(safePath(directory,receipt.outputs[0].file));requireThat(sha(outputBytes)===receipt.outputs[0].sha256,'actual output changed before review material');
  const output=decodeRgbPng(outputBytes),inputPairs=[],detailCrops=[];
  for(const id of row.refs) {
    const sourceBytes=await readFile(safePath(state.root,inputs[id].file));requireThat(sha(sourceBytes)===inputs[id].sha256,'original input changed before review material');
    const input=decodeRgbPng(sourceBytes),width=Math.max(input.width,output.width),height=input.height+output.height,pixels=Buffer.alloc(width*height*3);
    for(const [image,top] of [[input,0],[output,input.height]])for(let y=0;y<image.height;y++)image.pixels.copy(pixels,((top+y)*width)*3,y*image.width*3,(y+1)*image.width*3);
    const file=`review-pair-${id}.png`,bytes=rgbPng(width,height,pixels);await writeFile(resolve(directory,file),bytes,{flag:'wx',mode:0o600});
    inputPairs.push({inputId:id,inputSha256:inputs[id].sha256,outputSha256:receipt.outputs[0].sha256,file,sha256:sha(bytes),layout:'unscaled original above actual output',inputGeometry:{width:input.width,height:input.height},outputGeometry:{width:output.width,height:output.height}});
  }
  const width=Math.max(1,Math.floor(output.width/3)),height=Math.max(1,Math.floor(output.height/3)),top=Math.floor((output.height-height)/2);
  for(let i=0;i<3;i++) {
    const left=Math.floor((output.width-width)*i/2),pixels=Buffer.alloc(width*height*3);
    for(let y=0;y<height;y++)output.pixels.copy(pixels,y*width*3,((top+y)*output.width+left)*3,((top+y)*output.width+left+width)*3);
    const file=`review-detail-${i}.png`,bytes=rgbPng(width,height,pixels);await writeFile(resolve(directory,file),bytes,{flag:'wx',mode:0o600});detailCrops.push({file,sha256:sha(bytes),outputSha256:receipt.outputs[0].sha256,rectangle:{left,top,width,height}});
  }
  await save(resolve(directory,'visual-pairs.json'),{version:VERSION,lockSha256:digest(state.lock),rowId:row.id,inputPairs,detailCrops});
}
