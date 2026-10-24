// Purpose-specific native process, sampling and stopped-state transport for sc-26757.
// Importing this module does not start a process, device, browser or API.
import { spawn, execFile as callback } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdir, readFile, writeFile, readdir, lstat, cp, open } from 'node:fs/promises';
import { resolve, relative, dirname, basename, isAbsolute } from 'node:path';
import { deflateSync } from 'node:zlib';
import { VERSION, sha, digest, canonical, requireThat, save, json, safePath, decodeRgbPng, fileIdentity, treeIdentity, validateInferencePins, Api } from './moss-edit-terminal.mjs';

const execFile = promisify(callback);
export function validateDispatchEnvelope(raw,sourceSha,inferenceSha,dispatchSha) {
  requireThat(/^[a-f0-9]{40}$/.test(sourceSha)&&sourceSha===dispatchSha&&/^[a-f0-9]{40}$/.test(inferenceSha),'MOSS dispatch requires the exact workflow source SHA and inference-main pin');
  const value=JSON.parse(raw),required=['candidate_id','bundle_manifest_sha256','phase'],optional=['bundle_run_id','bundle_artifact_id','execution_run_id','execution_run_attempt','review_artifact_id','review_run_id','upload_path','upload_kind'];
  requireThat(value&&typeof value==='object'&&!Array.isArray(value)&&required.every(key=>typeof value[key]==='string'&&value[key])&&Object.keys(value).every(key=>[...required,...optional].includes(key)&&typeof value[key]==='string'),'MOSS dispatch JSON has missing, unknown or non-string fields');
  requireThat(/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(value.candidate_id)&&/^[a-f0-9]{64}$/.test(value.bundle_manifest_sha256),'MOSS dispatch candidate or manifest SHA is invalid');
  requireThat(['upload','execute','verify'].includes(value.phase),'MOSS dispatch phase must be upload, execute or verify');
  const empty=keys=>keys.every(key=>value[key]===undefined||value[key]==='');
  if(value.phase==='upload') {
    requireThat(['bundle','review'].includes(value.upload_kind)&&typeof value.upload_path==='string'&&value.upload_path.startsWith('/Users/michael/.codex/epic-runs/sc-26752/')&&resolve(value.upload_path)===value.upload_path,'MOSS upload requires a canonical directory inside the owned epic run');
    requireThat(empty(['bundle_run_id','bundle_artifact_id','execution_run_id','execution_run_attempt','review_artifact_id','review_run_id']),'MOSS upload cannot consume execution or prior artifact IDs');
  } else {
    requireThat([value.bundle_run_id,value.bundle_artifact_id].every(id=>/^[1-9][0-9]*$/.test(id??'')),'MOSS execution/verify requires exact bundle IDs');
    requireThat(empty(['upload_path','upload_kind']),'MOSS execution/verify cannot upload host files');
    if(value.phase==='verify')requireThat([value.execution_run_id,value.review_artifact_id,value.review_run_id??value.bundle_run_id,value.execution_run_attempt??'1'].every(id=>/^[1-9][0-9]*$/.test(id??'')),'MOSS verify requires exact execution run, attempt and review artifact/run IDs');
    else requireThat(empty(['execution_run_id','execution_run_attempt','review_artifact_id','review_run_id']),'MOSS execute cannot consume prior execution or review evidence');
  }
  return Object.fromEntries([...required,...optional].map(key=>[key,value[key]??(key==='execution_run_attempt'?'1':key==='review_run_id'&&value.phase==='verify'?value.bundle_run_id:'')]));
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
    SCENEWORKS_WORKER_ID:config.workerId,SCENEWORKS_WORKER_CHILD:'1',SCENEWORKS_GPU_ID:cuda?'1':'mlx',
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
  await cp(resolve(config.sourceRoot,'config'),resolve(config.stateRoot,'config'),{recursive:true,errorOnExist:true,force:false});
  const env=childEnvironment(config,token),children=[],handles=[];
  const launch=async(name,extra)=>{const handle=await open(resolve(config.root,`${name}.log`),'wx',0o600);handles.push(handle);const child=spawn(config.apiBinary,[],{cwd:config.sourceRoot,env:{...env,...extra},stdio:[name.startsWith('worker')?'pipe':'ignore',handle.fd,handle.fd],windowsHide:true});child.once('error',error=>{child.startError=error;});children.push(child);return child;};
  const api=await launch('api',utilityWorkerEnvironment(config)),client=new Api(env.SCENEWORKS_API_URL,token);let worker=await launch('worker',{SCENEWORKS_WORKER_ONLY:'1'}),restarts=0;
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
    await sourceIdentity(config);await nativeDevice(config.backend);config.observationJournal=resolve(config.root,`journal-${++restarts}.jsonl`);
    worker=await launch(`worker-restart-${restarts}`,{SCENEWORKS_WORKER_ONLY:'1',SCENEWORKS_MOSS_OBSERVATION_JOURNAL:config.observationJournal});
    return {reason,oldPid,oldExit,newPid:worker.pid,journal:config.observationJournal,source:await sourceIdentity(config),workerId:config.workerId,at:new Date().toISOString(),meaning:'actual idle owned worker process exited; its native resident caches cannot be reused by reentry'};
  };
  try {
    const deadline=Date.now()+180000;let ready,utility;
    while(Date.now()<deadline) {
      for(const child of children)requireThat(!child.startError&&child.exitCode===null&&child.signalCode===null,'source-built API/worker exited before readiness');
      try {const health=await client.request('/api/v1/health'),workers=await client.request('/api/v1/workers');ready=workers.find(item=>item.id===config.workerId);utility=readyUtilityWorker(workers,config.workerId);
        if(health.status==='ok'&&health.readiness?.status==='ready'&&ready?.gpuId===(config.backend==='cuda'?'1':'mlx')&&ready.status==='idle'&&utility)break;
      }catch{}await delay(500);
    }
    requireThat(ready?.gpuId===(config.backend==='cuda'?'1':'mlx')&&ready.status==='idle'&&utility,'actual native/CPU model_download workers never registered on their bound identities');
    await save(resolve(config.root,'stack.json'),{version:VERSION,backend:config.backend,source:await sourceIdentity(config),binary:await fileIdentity(config.apiBinary),apiUrl:client.base.href,worker:ready,utilityWorker:utility,pids:{api:api.pid,worker:worker.pid},at:new Date().toISOString()});
    return {api:client,children,stop,restartIdleWorker,async restartWorker(canceledJobId){
      const job=await client.request(`/api/v1/jobs/${encodeURIComponent(canceledJobId)}`);
      requireThat(job.status==='canceled'&&job.workerId===config.workerId&&!(job.result?.assets?.length)&&job.result?.outputPublication?.state!=='pending','owned cancellation/publication is not drained; do not restart');
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
