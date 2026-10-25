// sc-26691 adapters for the existing harness. Production entry is explicit Actions-only.
// No model implementation/importer/registry: own processes, existing API, native tests and receipts.
import assert from 'node:assert/strict';
import path from 'node:path';
import net from 'node:net';
import { randomUUID } from 'node:crypto';
import { createWriteStream } from 'node:fs';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdir, readFile, writeFile, realpath, stat, copyFile, rename, link } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { hashFile, sourcePair, verifyInputs, runProcess, executedTests, normalizeLoadEvent, decodeAV, compareControlMedia } from './h3-finetune-terminal.mjs';
import { runScenario, expand } from './h3-finetune-terminal-api.mjs';
import { caseBook } from './h3-finetune-terminal-cases.mjs';
import { cudaDeviceEvidence } from './ci/h3-import-native-proof.mjs';
const exec = promisify(execFile);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const json = async p => JSON.parse(await readFile(p, 'utf8'));
const save = (p, v) => writeFile(p, JSON.stringify(v, null, 2) + '\n', { flag: 'wx' });
const sha = v => assert.match(v ?? '', /^[a-f0-9]{64}$/);
const terminal = v => ['completed','failed','cancelled','canceled','interrupted'].includes(v);
const creativePayload = payload => {
  const copy=structuredClone(payload);
  for(const key of ['generationSetId','generationSet','jobId','createdAt','outputDir'])delete copy[key];
  return copy;
};
const conditionKeys=['prompt','sourceAssetId','lastFrameAssetId','sourceClipAssetId','sourceClipAssetIds','referenceAssetIds','referenceClipAssetId','referenceAudioAssetIds','loras'];
const submittedPayload=(requests,job)=>requests.find(r=>r.result?.id===job.id&&r.route==='/api/v1/video/jobs')?.request??job.payload;
export function assertControlledConfiguration(baseline,intervention,allowed=conditionKeys) {
  assert.ok(allowed.length && allowed.every(k=>conditionKeys.includes(k)),'only declared conditioning/adapter fields may change');
  const a=creativePayload(baseline),b=creativePayload(intervention);for(const key of allowed){delete a[key];delete b[key];}assert.deepEqual(a,b,'contrast changed source/seed/geometry/sampling or another non-intervened creative control');
}
export function requestedAdapterBindings(adapters, requested) {
  const ids=new Set(requested.flatMap(a=>typeof a==='string'?[a]:[a.id,a.loraId,a.name,a.path].filter(Boolean)));
  return adapters.filter(a=>[a.id,a.name,a.path].some(v=>v&&ids.has(v)));
}
const inside = (root, value) => {
  const p = path.resolve(root, value), rel = path.relative(root, p);
  assert.ok(rel && !rel.startsWith('..') && !path.isAbsolute(rel), 'path must remain below owned directory'); return p;
};
async function confined(root, value) {
  const logical=path.resolve(root),r=await realpath(root),candidate=path.resolve(value);
  const within=(base,p)=>{const rel=path.relative(base,p);return rel && !rel.startsWith('..') && !path.isAbsolute(rel);};
  assert.ok(within(logical,candidate)||within(r,candidate),'path must remain below owned directory');
  const p=await realpath(candidate);inside(r,p);return p;
}
export function authorized(p, c) {
  assert.notEqual(p.proof_kind,'unit_fake','unit_fake is not campaign authorization');
  assert.equal(process.env.GITHUB_ACTIONS, 'true', 'real services/native commands are explicit Actions-only');
  assert.equal(process.env.H3_TERMINAL_AUTHORIZED_PAIR, `${p.sceneworks}:${p.inference}`);
  if (c.backend === 'mlx') { assert.equal(process.platform,'darwin'); assert.equal(process.env.RUNNER_NAME,'nax-macos-2'); }
  else assert.equal(process.platform,'win32');
}
export async function hardware(backend, deviceId, deps={}) {
  const execute=deps.exec??exec;
  if (backend === 'mlx') {
    const gpu = JSON.parse((await execute('system_profiler',['SPDisplaysDataType','-json'])).stdout).SPDisplaysDataType;
    assert.ok(gpu?.length); return {device:gpu[0].sppci_model ?? gpu[0]._name,driver:`Apple Metal driver bundled with macOS ${(await execute('sw_vers',['-productVersion'])).stdout.trim()}`};
  }
  const inventory=(await execute('nvidia-smi',['--query-gpu=index,uuid,pci.bus_id,name,driver_version','--format=csv,noheader'])).stdout;
  const cuda=cudaDeviceEvidence(deps.env??process.env,inventory),row=cuda.selectedPhysicalDevice;
  assert.equal(row.uuid,deviceId,'requested CUDA UUID must be the actual PCI-ordered manual device zero');
  return {device:row.uuid+' '+row.name,driver:row.driverVersion,cuda};
}

// A native test already performs the numerical comparison. Do not invent another flat-vector gate.
export function validateComparisonProof(c, proof, log, { allowUnitFake = false } = {}) {
  assert.equal(proof.kind,'native-pinned-fixture');
  assert.ok(proof.proof_kind==='campaign_native' || (allowUnitFake && proof.proof_kind==='unit_fake'), 'unit_fake is never campaign evidence');
  assert.equal(proof.backend,c.backend); assert.equal(proof.comparisonKind,'native-assertions-against-precomputed-pinned-cpu-reference');
  assert.deepEqual(proof.tests,c.numericProof.tests); sha(proof.binary.sha256);
  assert.ok(proof.binary.buildCommand.length && proof.binary.configuration && proof.inventoryLog);
  assert.ok(proof.fixtures.length && proof.referenceBlobs.length);
  for(const input of [...proof.fixtures,...proof.referenceBlobs]) {
    assert.ok(c.inputs.some(i=>i.path===input.path && i.sha256===input.sha256),'numerical fixture/reference not independently input-bound'); sha(input.sha256);
  }
  const check=c.numericProof.mechanism??c.id.split('.').at(-1), expected=caseBook.numeric[check];
  if (expected?.fixture) {
    const names=expected.selectors ?? [expected.selector];
    for(const name of names) assert.ok(proof.tests.includes(name),'missing contracted native comparison');
    assert.equal(proof.referenceRevision,expected.referenceRevision);
  }
  if(c.id.endsWith('.tensor-mapping')) {
    assert.ok(c.numericProof.supportingCommands?.length,'tensor mapping also requires its existing source-layout/provider fixture assertions');
    assert.deepEqual(proof.supporting?.map(v=>v.command),c.numericProof.supportingCommands);
    for(const item of proof.supporting)assert.ok(item.passed>0);
  }
  const escaped=s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  for(const name of proof.tests) assert.match(log,new RegExp(`(?:^|\\n)test ${escaped(name)} \\.\\.\\.`),'named native test was not selected/executed');
  assert.equal(proof.runs.length,proof.tests.length);
  assert.equal(new Set(proof.runs.map(r=>r.test)).size,proof.tests.length);
  for(const r of proof.runs) { assert.ok(proof.tests.includes(r.test)); assert.deepEqual(r.command,[proof.binary.path,r.test,'--exact','--ignored','--nocapture','--test-threads=1']); assert.equal(r.passed,1); assert.equal(r.failed,0); assert.equal(r.ignored,0); }
  assert.equal(executedTests(log),proof.tests.length);
  assert.ok(!/\bSKIP\b/.test(log),'guard-return/skip is not native numerical proof');
  return {kind:proof.kind,comparisonKind:proof.comparisonKind,tests:proof.tests,passed:proof.tests.length,fixtureCases:proof.fixtureCases};
}
export async function executeNativeComparison(p,c,swRoot,inferenceRoot,dir,deps={}) {
  const fake=deps.proof_kind==='unit_fake';
  if(!fake) { authorized(p,c); await sourcePair(p,swRoot,inferenceRoot); await verifyInputs(c,[p.output,swRoot,inferenceRoot]); }
  const runner=deps.runProcess ?? runProcess, n=c.numericProof; assert.equal(n.kind,'native-pinned-fixture');
  assert.ok(n.tests?.length && new Set(n.tests).size===n.tests.length);
  const fixtureInputs=c.inputs.filter(i=>i.role==='numeric-fixture'), refs=c.inputs.filter(i=>i.role==='reference');
  assert.ok(fixtureInputs.length && refs.length);
  for(const i of [...fixtureInputs,...refs]) assert.equal(await hashFile(i.path),i.sha256);
  const authority=caseBook.numeric[n.mechanism??c.id.split('.').at(-1)];assert.ok(authority?.fixture,'native fixture mechanism required');
  assert.ok(fixtureInputs.some(i=>path.resolve(i.path)===path.resolve(inferenceRoot,authority.fixture)),'bind the actual compile-time fixture, not a parallel file');
  const pinnedFixture=await json(path.join(inferenceRoot,authority.fixture));
  const sourceHashes=[...(authority.kernelBlobs?.map(v=>v.sha256) ?? Object.values(authority.referenceSha256??{})),...(pinnedFixture.reference?.kernelBlobs??[]).map(v=>v.sha256)];
  if((n.mechanism??c.id.split('.').at(-1))==='convrot') for(const key of ['reference_helper','reference_cuda','reference_producer'])sourceHashes.push(pinnedFixture[key].sha256);
  for(const hash of sourceHashes)assert.ok(refs.some(v=>v.sha256===hash),'missing exact independently pinned external arithmetic blob');
  const h=await (deps.hardware??hardware)(c.backend,c.runtime?.deviceId);
  const base=['cargo','test','--locked','--manifest-path',path.join(inferenceRoot,'Cargo.toml'),'-p',n.package,'--test','integration'];
  assert.equal(n.package,c.backend==='mlx'?'mlx-gen-minimax-h3':'candle-gen-minimax-h3');
  if(c.backend==='cuda') base.push('--features','cuda');
  const buildCommand=[...base,'--no-run','--message-format=json'];
  const opts={cwd:inferenceRoot,env:{...process.env,...(c.backend==='cuda'?{RUSTC_WRAPPER:''}:{}),CARGO_TARGET_DIR:path.join(p.output,`native-target-${p.inference}-${c.backend}`),CARGO_PROFILE_TEST_DEBUG:'0',CARGO_PROFILE_DEV_DEBUG:'0'},timeoutSeconds:c.timeoutSeconds};
  await runner(buildCommand,{...opts,logPath:path.join(dir,'native-build.log')});
  const records=(await readFile(path.join(dir,'native-build.log'),'utf8')).split('\n').flatMap(l=>{try{return [JSON.parse(l)];}catch{return [];}});
  const artifact=records.find(v=>v.reason==='compiler-artifact' && v.target?.name==='integration' && v.profile?.test && v.executable);
  assert.ok(artifact,'actual cargo test executable record required');
  const binary=await confined(p.output,artifact.executable), binaryHash=await hashFile(binary);
  await runner([binary,'--list'],{...opts,logPath:path.join(dir,'native-inventory.log')});
  const inventory=await readFile(path.join(dir,'native-inventory.log'),'utf8');
  const runs=[],logs=[];
  for(const [index,test] of n.tests.entries()) {
    assert.ok(inventory.split('\n').includes(`${test}: test`),'test absent from unfiltered actual binary inventory');
    const logName=`native-test-${index}.log`,command=[binary,test,'--exact','--ignored','--nocapture','--test-threads=1'];
    await runner(command,{...opts,logPath:path.join(dir,logName)});
    const log=await readFile(path.join(dir,logName),'utf8'); assert.equal(executedTests(log),1); logs.push(log);
    runs.push({test,command,log:logName,passed:1,failed:0,ignored:0});
  }
  const supporting=[];
  for(const [index,command] of (n.supportingCommands??[]).entries()) {
    assert.equal(command[0],'cargo');assert.ok(command.includes('test')&&command.includes(path.join(swRoot,'Cargo.toml')),'supporting assertions must use this paired SceneWorks source');
    const log=`mapping-support-${index}.log`;await runner(command,{...opts,cwd:swRoot,logPath:path.join(dir,log)});
    const text=await readFile(path.join(dir,log),'utf8');supporting.push({command,log,passed:executedTests(text)});
  }
  const fixtureCases=[];
  for(const input of fixtureInputs) { const f=await json(input.path); for(const [i,v] of (f.cases??[]).entries()) fixtureCases.push({fixtureSha256:input.sha256,id:v.id??{index:i,group:v.group,dtype:v.dtype}}); }
  const proof={kind:n.kind,proof_kind:fake?'unit_fake':'campaign_native',backend:c.backend,sceneworks:p.sceneworks,inference:p.inference,comparisonKind:'native-assertions-against-precomputed-pinned-cpu-reference',referenceRevision:c.referenceRevision,tests:n.tests,fixtureCases,fixtures:fixtureInputs,referenceBlobs:refs,binary:{path:binary,sha256:binaryHash,buildCommand,configuration:{backend:c.backend,package:n.package,testTarget:'integration',features:c.backend==='cuda'?['cuda']:[],locked:true,cargoArtifact:{target:artifact.target,profile:artifact.profile,features:artifact.features??[]},buildEnvironment:Object.fromEntries(['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_TARGET','RUSTUP_TOOLCHAIN'].map(k=>[k,process.env[k]??null]))}},inventoryLog:'native-inventory.log',runs,supporting};
  validateComparisonProof(c,proof,logs.join('\n'),{allowUnitFake:fake});
  const files=['native-build.log','native-inventory.log',...runs.map(r=>r.log),...supporting.map(r=>r.log)];
  await save(path.join(dir,'native-comparison.json'),proof);
  const e={status:'executed',proof_kind:proof.proof_kind,caseId:c.id,backend:c.backend,sceneworks:p.sceneworks,inference:p.inference,hardware:h,testInputSha256:fixtureInputs[0].sha256,files:[...files,'native-comparison.json'],nativeComparison:'native-comparison.json'};
  await save(path.join(dir,c.evidence),e);
  if(!fake) {await sourcePair(p,swRoot,inferenceRoot);await verifyInputs(c,[p.output,swRoot,inferenceRoot]);}
  return {evidence:e,proof,log:logs.join('\n')};
}

export async function executeFixture(p,c,swRoot,inferenceRoot,dir,deps={}) {
  const fake=deps.proof_kind==='unit_fake';if(!fake){authorized(p,c);await sourcePair(p,swRoot,inferenceRoot);await verifyInputs(c,[p.output,swRoot,inferenceRoot]);}
  assert.equal(c.fixtureProof.kind,'source-bound-assertions');assert.equal(c.command[0],'cargo');assert.ok(c.command.includes('test')&&c.command.includes(path.join(swRoot,'Cargo.toml')));
  assert.ok(c.inputs.length);for(const input of c.inputs)assert.equal(await hashFile(input.path),input.sha256);
  const state=path.join(dir,'fixture-state');await mkdir(state);
  const env={...privateEnvironment(state,{port:43210,gpuId:'cpu',backend:c.backend},randomUUID()),CARGO_TARGET_DIR:path.join(p.output,`fixture-target-${p.sceneworks}-${p.inference}-${c.backend}`),CARGO_PROFILE_TEST_DEBUG:'0',CARGO_PROFILE_DEV_DEBUG:'0'};
  await (deps.runProcess??runProcess)(c.command,{cwd:swRoot,env,logPath:path.join(dir,'assertions.log'),timeoutSeconds:c.timeoutSeconds});
  const log=await readFile(path.join(dir,'assertions.log'),'utf8'),passed=executedTests(log);assert.ok(!/\bSKIP\b/.test(log));
  const proof={kind:'source-bound-assertions',proof_kind:fake?'unit_fake':'campaign_fixture',sceneworks:p.sceneworks,inference:p.inference,command:c.command,passed,inputs:c.inputs,meaning:'Executed source-bound synthetic contract assertions; no real-model/native load or measured peak claimed'};
  await save(path.join(dir,'assertion-proof.json'),proof);
  const e={status:'executed',proof_kind:proof.proof_kind,caseId:c.id,backend:c.backend,sceneworks:p.sceneworks,inference:p.inference,hardware:await(deps.hardware??hardware)(c.backend,c.fixtureProof.deviceId),testInputSha256:c.inputs[0].sha256,files:['assertions.log','assertion-proof.json']};
  await save(path.join(dir,c.evidence),e);if(!fake){await sourcePair(p,swRoot,inferenceRoot);await verifyInputs(c,[p.output,swRoot,inferenceRoot]);}return {evidence:e,proof,log};
}
function spawnOwned(argv,env,cwd,logPath) {
  const child=spawn(argv[0],argv.slice(1),{env,cwd,stdio:['pipe','pipe','pipe'],shell:false});
  const chunks=[]; let failure;
  const output=createWriteStream(logPath,{flags:'wx'});output.on('error',e=>{failure=e;});
  child.stdout.pipe(output,{end:false});child.stderr.pipe(output,{end:false});
  child.on('error',e=>{failure=e;});
  const exited=new Promise(resolve=>child.on('close',(code,signal)=>resolve({code,signal})));
  for(const stream of [child.stdout,child.stderr]) stream.on('data',b=>chunks.push(Buffer.from(b)));
  return {pid:child.pid,exitCode:()=>child.exitCode,stdin:child.stdin,signal:s=>child.kill(s),exit:async()=>{const result=await exited;await new Promise(resolve=>output.end(resolve));if(failure)throw failure;return result;},log:()=>Buffer.concat(chunks).toString('utf8')};
}
async function requireFreePort(port) {
  const socket=net.createServer();
  await new Promise((resolve,reject)=>{socket.once('error',reject);socket.listen(port,'127.0.0.1',resolve);});
  await new Promise((resolve,reject)=>socket.close(e=>e?reject(e):resolve()));
}
async function listenerPid(port) {
  if(process.platform==='win32') {
    const s=await exec('powershell.exe',['-NoProfile','-NonInteractive','-Command',`(Get-NetTCPConnection -State Listen -LocalPort ${port} -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique) -join ','`]);
    const ids=s.stdout.trim().split(',').map(Number);assert.equal(ids.length,1);return ids[0];
  }
  const s=await exec('lsof',['-nP',`-iTCP:${port}`,'-sTCP:LISTEN','-Fp']);
  const ids=[...new Set(s.stdout.split('\n').filter(l=>l.startsWith('p')).map(l=>Number(l.slice(1))))];assert.equal(ids.length,1);return ids[0];
}
export function privateEnvironment(runDir,runtime,token) {
  const env={}; // No inherited user HF settings/tokens, credentials, proxies or application roots.
  for(const key of ['PATH','RUSTUP_TOOLCHAIN','SystemRoot','WINDIR','COMSPEC','PATHEXT','TMPDIR','TEMP','TMP','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','MLX_METAL_DEVICE_INDEX','SCENEWORKS_GPU_MEMORY_LIMIT_BYTES','LD_LIBRARY_PATH','DYLD_LIBRARY_PATH']) if(process.env[key]!==undefined)env[key]=process.env[key];
  Object.assign(env,{SCENEWORKS_API_HOST:'127.0.0.1',SCENEWORKS_API_PORT:String(runtime.port),SCENEWORKS_API_URL:`http://127.0.0.1:${runtime.port}`,SCENEWORKS_DATA_DIR:path.join(runDir,'data'),SCENEWORKS_CONFIG_DIR:path.join(runDir,'config'),SCENEWORKS_JOBS_DB_PATH:path.join(runDir,'data','cache','jobs.db'),SCENEWORKS_ACCESS_TOKEN:token,SCENEWORKS_RUN_UTILITY_INPROCESS:'0',SCENEWORKS_UTILITY_WORKERS:'1',SCENEWORKS_WORKER_ID:`sc26691-${path.basename(runDir)}`,SCENEWORKS_GPU_ID:runtime.gpuId,SCENEWORKS_BACKEND_MLX_ENABLED:runtime.backend==='mlx'?'1':'0',SCENEWORKS_BACKEND_CANDLE_ENABLED:runtime.backend==='cuda'?'1':'0',SCENEWORKS_LOG_FORMAT:'json',SCENEWORKS_WORKER_SHUTDOWN_TIMEOUT_SECONDS:'120',RUST_LOG:'info,sceneworks=debug',HF_HOME:path.join(runDir,'hf'),HF_HUB_CACHE:path.join(runDir,'hf','hub'),HUGGINGFACE_HUB_CACHE:path.join(runDir,'hf','hub'),TRANSFORMERS_CACHE:path.join(runDir,'hf','transformers'),XDG_CACHE_HOME:path.join(runDir,'cache'),HF_HUB_OFFLINE:'1',HF_DATASETS_OFFLINE:'1'});
  return env;
}
export class OwnedService {
  constructor(p,c,swRoot,inferenceRoot,dir,deps={}) {
    Object.assign(this,{p,c,swRoot,inferenceRoot,dir,deps});this.fake=deps.proof_kind==='unit_fake';this.jobs=new Set();this.epochs=[];this.drainRecords=[];this.quiescence=[];this.token=randomUUID();this.generation=0;this.startTime=undefined;this.stopped=false;
  }
  async initialize() {
    if(!this.fake){authorized(this.p,this.c);await sourcePair(this.p,this.swRoot,this.inferenceRoot);await verifyInputs(this.c,[this.p.output,this.swRoot,this.inferenceRoot]);}
    const r=this.c.runtime;
    assert.ok(this.c.inputs.some(i=>i.path===r?.scenario && i.role==='scenario'),'bind the actual request/action scenario as an input');
    assert.ok(r && Number.isInteger(r.port) && r.port>1024 && r.port<=65535 && r.gpuId);
    if(!this.fake && this.c.backend==='cuda') {
      assert.equal(String(r.gpuId),'0','manual campaign must retain product GPU zero');
      await hardware('cuda',r.deviceId); // Confirm UUID/PCI mapping before any build, listener or model load.
    }
    const services=path.join(this.p.output,'services');await mkdir(services,{recursive:true});
    this.runDir=path.join(services,this.c.id);await mkdir(this.runDir);this.owner=randomUUID();
    for(const d of ['config','data','data/cache','hf','hf/hub','cache']) await mkdir(path.join(this.runDir,d),{recursive:true});
    this.ownerPath=path.join(this.runDir,'owner.json');await save(this.ownerPath,{token:this.owner,pid:process.pid,sceneworks:this.p.sceneworks,inference:this.p.inference,proof_kind:this.fake?'unit_fake':'campaign_native'});
    this.seeded=new Set();this.seededInputs=new Map();
    for(const seed of r.seedFiles??[]) {
      const input=this.c.inputs.find(i=>i.path===seed.input);assert.ok(input,'seed must be an independently bound input');
      const canonicalInput=await confined(input.role==='source-config'?this.swRoot:this.p.output,input.path); // No user/shared-cache seeds; only explicitly staged task bytes or source config.
      assert.equal(await hashFile(input.path),input.sha256);
      assert.ok(/^(config|data|hf)\//.test(seed.destination) && !/credentials|gpu_telemetry|jobs\.db/.test(seed.destination),'only explicit clean platform state, never credentials/stale counters/database seeds');
      const dest=inside(this.runDir,seed.destination);await mkdir(path.dirname(dest),{recursive:true});
      if(seed.mode==='hardlink'){assert.notEqual(input.role,'source-config');await link(canonicalInput,dest);}else{assert.ok(seed.mode===undefined||seed.mode==='copy');await copyFile(input.path,dest);}
      const canonical=await realpath(dest);this.seeded.add(canonical);this.seededInputs.set(canonical,input);
    }
    await this.build();
    try {await this.start();return this;} catch(error) {
      // No scenario was submitted. Close only our guardian/worker handles; never touch the foreign listener.
      if(this.worker){if((this.deps.platform??process.platform)==='win32')this.worker.stdin.end();else this.worker.signal('SIGTERM');}
      this.guardian?.stdin.end();
      const exits=[];
      for(const proc of [this.worker,this.guardian,this.api].filter(Boolean)){let timer;try{exits.push({pid:proc.pid,...await Promise.race([proc.exit(),new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('owned startup cleanup timeout; no forced kill')),120000);})])});}catch(cleanup){exits.push({pid:proc.pid,error:cleanup.message});}finally{clearTimeout(timer);}}
      await save(path.join(this.dir,'startup-failure.json'),{error:error.message,ownedApiPid:this.api?.pid,ownedWorkerPid:this.worker?.pid,guardianPid:this.guardian?.pid,exits,meaning:'No scenario submitted; only owned graceful shutdown requested, no force kill'});
      throw error;
    }
  }
  async build() {
    const runner=this.deps.runProcess??runProcess;
    const command=['cargo','build','--locked','--manifest-path',path.join(this.swRoot,'Cargo.toml'),'-p','sceneworks-rust-api','--bin','sceneworks-rust-api'];
    if(this.c.backend==='cuda')command.push('--features','sceneworks-rust-api/backend-candle');
    const buildRoot=path.join(this.p.output,`service-build-${this.p.sceneworks}-${this.p.inference}-${this.c.backend}`),target=path.join(buildRoot,'target'),receipt=path.join(buildRoot,'build.json');
    const configuration={command,flags:Object.fromEntries(['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_TARGET','RUSTUP_TOOLCHAIN'].map(k=>[k,process.env[k]??null])),proof_kind:this.fake?'unit_fake':'campaign_native'};
    let cached;
    try{await mkdir(buildRoot);}catch(error){if(error.code!=='EEXIST')throw error;cached=await json(receipt);assert.deepEqual(cached.configuration,configuration,'source/build configuration drift; preserve prior candidate');assert.equal(await hashFile(cached.binary),cached.sha256);}
    if(cached){this.buildIdentity=cached;this.binary=cached.binary;this.binaryHash=cached.sha256;await copyFile(path.join(buildRoot,'build.log'),path.join(this.dir,'service-build.log'));}
    else {
      await runner(command,{cwd:this.swRoot,env:{...process.env,...(this.c.backend==='cuda'?{RUSTC_WRAPPER:''}:{}),CARGO_TARGET_DIR:target,CARGO_PROFILE_DEV_DEBUG:'0'},logPath:path.join(buildRoot,'build.log'),timeoutSeconds:this.c.timeoutSeconds});
      this.binary=path.join(target,'debug',`sceneworks-rust-api${(this.deps.platform??process.platform)==='win32'?'.exe':''}`);
      this.binaryHash=await hashFile(this.binary);this.buildIdentity={command,binary:this.binary,sha256:this.binaryHash,sceneworks:this.p.sceneworks,inference:this.p.inference,backend:this.c.backend,configuration};await save(receipt,this.buildIdentity);await copyFile(path.join(buildRoot,'build.log'),path.join(this.dir,'service-build.log'));
    }
    await save(path.join(this.dir,'service-build.json'),this.buildIdentity);
  }
  async assertOwner() {const record=await json(this.ownerPath);assert.equal(record.token,this.owner);assert.equal(record.pid,process.pid);assert.equal(await hashFile(this.binary),this.binaryHash);}
  async start() {
    await this.assertOwner();assert.ok(!this.api || this.stopped,'refuse second live service');
    await (this.deps.requireFreePort??requireFreePort)(this.c.runtime.port); // No HTTP probe of a pre-existing/user API.
    const launch=this.deps.spawnOwned??spawnOwned, epoch=++this.generation;
    const env=privateEnvironment(this.runDir,{...this.c.runtime,backend:this.c.backend},this.token);
    this.guardian=launch([process.execPath,'-e',"process.stdin.resume();process.stdin.on('end',()=>process.exit(0));"],env,this.swRoot,path.join(this.dir,`guardian-${epoch}.log`));
    assert.ok(this.guardian.pid>1);env.SCENEWORKS_PARENT_PID=String(this.guardian.pid);
    this.api=launch([this.binary],env,this.swRoot,path.join(this.dir,`api-${epoch}.log`));assert.ok(this.api.pid>1);
    this.startTime=new Date().toISOString();this.stopped=false;
    const deadline=Date.now()+60000;
    for(;;) {
      assert.equal(this.api.exitCode(),null,'new API exited before ownership/readiness');
      let pid;try{pid=await (this.deps.listenerPid??listenerPid)(this.c.runtime.port);}catch{}
      if(pid!==undefined){assert.equal(pid,this.api.pid,'refuse contacting a listener not owned by this child');break;}
      assert.ok(Date.now()<deadline,'owned API listener timeout');await (this.deps.sleep??sleep)(100);
    }
    this.baseUrl=`http://127.0.0.1:${this.c.runtime.port}`;
    const health=await this.request('GET','/api/v1/health');assert.equal(health.service,'sceneworks-api');assert.equal(health.status,'ok');
    // Preserve, rather than reuse, a previous owned worker's cumulative counter across restart.
    try{await rename(path.join(this.runDir,'config','gpu_telemetry.json'),path.join(this.runDir,'config',`gpu_telemetry.before-${epoch}.json`));}catch(error){if(error.code!=='ENOENT')throw error;}
    this.workerId=env.SCENEWORKS_WORKER_ID;
    this.worker=launch([this.binary],{...env,SCENEWORKS_WORKER_ONLY:'1',SCENEWORKS_WORKER_CHILD:'1'},this.swRoot,path.join(this.dir,`worker-${epoch}.log`));assert.ok(this.worker.pid>1);
    const record={epoch,started:this.startTime,apiPid:this.api.pid,workerPid:this.worker.pid,guardianPid:this.guardian.pid,binarySha256:this.binaryHash,listenerPid:this.api.pid,runDirectory:this.runDir};this.epochs.push(record);await save(path.join(this.dir,`ownership-${epoch}.json`),record);
  }
  async fetch(url,options={}) {
    await this.assertOwner();assert.ok(!this.stopped);assert.equal(new URL(url).origin,this.baseUrl);assert.equal(await (this.deps.listenerPid??listenerPid)(this.c.runtime.port),this.api.pid,'listener ownership changed');
    return (this.deps.fetch??fetch)(url,{...options,headers:{...options.headers,'X-SceneWorks-Token':this.token}});
  }
  async request(method,route,body) {
    const response=await this.fetch(new URL(route,this.baseUrl),{method,headers:{'content-type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),signal:AbortSignal.timeout(30000)});
    const value=JSON.parse(await response.text());assert.ok(response.status>=200 && response.status<300,`${route}: ${JSON.stringify(value)}`);return value;
  }
  async waitIdle() {
    const deadline=Date.now()+120000;let seen;
    for(;;){
      if(this.worker.exitCode()!==null)return {epoch:this.generation,workerPid:this.worker.pid,status:'exited',exitCode:this.worker.exitCode()};
      const workers=await this.request('GET','/api/v1/workers'),worker=workers.find(v=>v.id===this.workerId);
      if(worker){if(seen!==undefined && worker.lastSeenAt!==seen && worker.status==='idle'&&!worker.currentJobId)return {epoch:this.generation,workerPid:this.worker.pid,worker,meaning:'Fresh owned-worker Idle heartbeat after terminal job acknowledgement and cleanup'};seen=worker.lastSeenAt;}
      assert.ok(Date.now()<deadline,'owned worker has not acknowledged quiescence; never signal an active/unknown Metal buffer');await(this.deps.sleep??sleep)(250);
    }
  }
  async drain() {
    const records=[];
    for(const id of this.jobs) {
      let job=await this.request('GET',`/api/v1/jobs/${encodeURIComponent(id)}`);
      if(!terminal(job.status))await this.request('POST',`/api/v1/jobs/${encodeURIComponent(id)}/cancel`);
      const deadline=Date.now()+120000;
      while(!terminal(job.status)){assert.ok(Date.now()<deadline,'cooperative cancel did not complete; do not signal native worker');await (this.deps.sleep??sleep)(250);job=await this.request('GET',`/api/v1/jobs/${encodeURIComponent(id)}`);}
      records.push(job);
    }
    this.quiescence.push(await this.waitIdle());
    const log=`drained-${this.generation}-${this.drainRecords.length}.json`;
    await save(path.join(this.dir,log),records);this.drainRecords.push(log);this.jobs.clear();return records;
  }
  async stop() {
    if(this.stopped)return;await this.assertOwner();await this.drain();
    // No signal until every owned submitted job is terminal. Never signal arbitrary PIDs.
    if((this.deps.platform??process.platform)==='win32')this.worker.stdin.end();else this.worker.signal('SIGTERM');
    const bounded=async proc=>{let timer;try{return await Promise.race([proc.exit(),new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('owned graceful exit timeout; no forced kill')),120000);})]);}finally{clearTimeout(timer);}};
    const workerExit=await bounded(this.worker);
    this.guardian.stdin.end();await bounded(this.guardian); // Existing API parent-death watchdog -> graceful drain, on BOTH OSes.
    const apiExit=await bounded(this.api);this.stopped=true;
    await save(path.join(this.dir,`exits-${this.generation}.json`),{workerExit,apiExit,ended:new Date().toISOString()});
    assert.equal(workerExit.code,0,'owned worker did not exit gracefully');assert.equal(apiExit.code,0,'owned API did not exit gracefully');
  }
  async checkStep(step,captures) {
    if(step.waitJob)return;
    const route=expand(step.path,captures),body=expand(step.body,captures);
    if(step.method==='POST'&&route==='/api/v1/models/import') {
      assert.ok(!body.repo && !body.sourceUrl && body.source?.kind!=='url','use an explicit staged local/linked source or exact cached Hugging Face source, never an implicit URL download');
      if(body.source?.kind==='huggingFace') {
        const {repo,revision,files}=body.source;assert.match(repo??'',/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/);assert.match(revision??'',/^[a-f0-9]{40}$/);assert.ok(Array.isArray(files)&&files.length&&files.every(f=>typeof f==='string'),'exact staged HF file inventory required');
        const snapshot=inside(this.runDir,`hf/hub/models--${repo.replaceAll('/','--')}/snapshots/${revision}`);
        for(const file of files){const target=await confined(snapshot,inside(snapshot,file)),seed=this.seededInputs.get(target);assert.ok(seed,'HF import must reuse an actually seeded private snapshot, not download missing weights');if(files.length===1&&body.expectedSha256)assert.equal(seed.sha256,body.expectedSha256);}
        return; // The EXISTING importer/registry still owns provenance, licensing, interpretation and ingestion.
      }
      const source=body.sourcePath??body.source?.path??body.source?.stagedPath;if(source)await confined(this.p.output,source);
      else {
        assert.ok(body.ownershipMode==='linked'&&body.linkedRootId&&body.linkedRelativePath,'explicit staged local/linked source required');
        const roots=await this.request('GET','/api/v1/models/library-roots'),root=roots.roots.find(r=>r.rootId===body.linkedRootId);assert.ok(root,'owned approved linked root required');
        await confined(this.p.output,root.path);await confined(root.path,path.resolve(root.path,body.linkedRelativePath));
      }
    }
    if(route.startsWith('/api/v1/models/library-roots')&&body?.path)await confined(this.p.output,body.path);
  }
  async action(action,captures) {
    await this.assertOwner();
    if(action.kind==='restart' || action.kind==='interrupt-import') {
      if(action.jobCapture){const job=action.jobCapture.split('/').reduce((v,k)=>v[k],captures);assert.ok(typeof job==='string');this.jobs.add(job);}
      if(action.kind==='interrupt-import'){
        assert.ok(action.jobCapture,'interrupted import must identify the actual submitted import');
        const id=action.jobCapture.split('/').reduce((v,k)=>v[k],captures),deadline=Date.now()+60000;
        for(;;){const job=await this.request('GET',`/api/v1/jobs/${encodeURIComponent(id)}`);assert.ok(!terminal(job.status),'missed active import interruption; do not claim queued/completed work was interrupted');if(['preparing','downloading','loading_model','running','saving'].includes(job.status))break;assert.ok(Date.now()<deadline,'owned import did not enter an active transfer');await(this.deps.sleep??sleep)(100);}
      }
      await this.stop();
      if(action.kind==='interrupt-import') {
        const id=action.jobCapture.split('/').reduce((v,k)=>v[k],captures),records=await json(path.join(this.dir,this.drainRecords.at(-1))),job=records.find(v=>v.id===id);
        assert.ok(job?.payload?.modelId && !job.payload.model,'interruption target must be this owned import, not a native render');assert.ok(['cancelled','canceled'].includes(job.status),'missed import interruption is unresolved, not successful proof');
      }
      await this.start();return {kind:action.kind,epochs:this.epochs};
    }
    assert.equal(action.kind,'replace-owned-source');await this.drain();
    const target=await confined(this.runDir,path.resolve(this.runDir,action.path));assert.ok(this.seeded.has(target),'mutation only of an explicitly seeded private source copy');assert.equal(await hashFile(target),action.expectedSha256);
    const replacement=this.c.inputs.find(i=>i.path===action.replacement && i.sha256===action.replacementSha256);assert.ok(replacement);assert.equal(await hashFile(replacement.path),replacement.sha256);
    const backup=target+`.before-${this.owner}`;await rename(target,backup);await copyFile(replacement.path,target);
    return {kind:action.kind,target,backup,beforeSha256:action.expectedSha256,afterSha256:await hashFile(target)};
  }
}

export async function processSample(service) {
  const pid=service.worker.pid,at=new Date().toISOString();let host;
  if(process.platform==='win32')host=Number((await exec('powershell.exe',['-NoProfile','-NonInteractive','-Command',`(Get-Process -Id ${pid} -ErrorAction Stop).WorkingSet64`])).stdout.trim());
  else host=Number((await exec('/bin/ps',['-o','rss=','-p',String(pid)])).stdout.trim())*1024;
  assert.ok(Number.isFinite(host)&&host>0,'RSS unavailable; missing is not zero');
  const base={at,pid,host,method:service.c.backend==='mlx'?'mlx-allocator-cumulative-high-water':'sampled-device-used',deviceId:service.c.runtime.deviceId??'Apple Metal unified device'};
  try {
    if(service.c.backend==='mlx') {
      const file=inside(service.runDir,'config/gpu_telemetry.json'),s=await stat(file),v=await json(file);assert.ok(s.isFile(),'actual owned-worker telemetry file required');assert.ok(Number.isFinite(v.peakBytes)&&v.peakBytes>=0,'MLX peak unavailable, not zero');
      return {...base,at:new Date().toISOString(),probeStartedAt:at,device:v.peakBytes,deviceCounterAt:s.mtime.toISOString()};
    }
    const rows=(await exec('nvidia-smi',['--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'])).stdout.trim().split('\n').map(l=>l.split(',').map(v=>v.trim()));
    const row=rows.find(r=>r[0]===service.c.runtime.deviceId);assert.ok(row && row[1]!=='' && Number.isFinite(Number(row[1])) && Number(row[1])>=0,'device used memory unavailable, not zero');
    return {...base,at:new Date().toISOString(),probeStartedAt:at,device:Number(row[1])*1024*1024};
  } catch(error) {return {...base,at:new Date().toISOString(),probeStartedAt:at,device:null,deviceUnavailable:error.message};} // Preserve actual host loading observations even before device initialization.
}
function observedInterval(samples,device,nominal) {
  let previous,maximum=nominal;
  for(const sample of samples)if(Number.isFinite(device?sample.device:sample.host)){const at=Date.parse(sample.at);if(previous!==undefined)maximum=Math.max(maximum,at-previous);previous=at;}
  return maximum;
}
export class ProcessObservations {
  constructor(service,{sample=processSample,intervalMs=250,sleepImpl=sleep}={}){Object.assign(this,{service,sample,intervalMs,sleepImpl});this.samples=[];this.gaps=[];this.done=false;}
  async checkBudget(sample) {
    const c=this.service.c;
    if(!this.budgetExceeded && (sample.host>c.loadHostBudgetBytes || (Number.isFinite(sample.device)&&sample.device>c.loadDeviceBudgetBytes))) {
      this.budgetExceeded={at:sample.at,host:sample.host,device:sample.device,deviceMethod:sample.method,meaning:'Observed explicit campaign bound exceeded; cooperative cancellation only, never a force kill or runtime-admission change'};
      for(const id of this.service.jobs??[]){const job=await this.service.request('GET',`/api/v1/jobs/${encodeURIComponent(id)}`);if(!terminal(job.status))await this.service.request('POST',`/api/v1/jobs/${encodeURIComponent(id)}/cancel`);}
    }
  }
  start(){this.started=new Date().toISOString();this.loop=(async()=>{while(!this.done){try{const sample=await this.sample(this.service);this.samples.push(sample);await this.checkBudget(sample);if(!Number.isFinite(sample.device))this.gaps.push({at:sample.at,error:sample.deviceUnavailable??'device counter unavailable; host observation retained'});}catch(error){this.gaps.push({at:new Date().toISOString(),error:error.message});}if(!this.done)await this.sleepImpl(this.intervalMs);}})();return this;}
  async finish(){
    this.completed=new Date().toISOString();this.done=true;await this.loop;
    const deadline=Date.now()+15000;let last,counterBarrier;
    do {try{last=await this.sample(this.service);if(Number.isFinite(last.device)){if(this.service.c.backend!=='mlx')break;if(counterBarrier===undefined)counterBarrier=last.deviceCounterAt;else if(last.deviceCounterAt!==counterBarrier)break;}}catch(error){this.gaps.push({at:new Date().toISOString(),error:error.message});}assert.ok(Date.now()<deadline,'no fresh post-completion native counter; cannot bound lazy loading/decode');await this.sleepImpl(250);}while(true);
    this.samples.push(last);await this.checkBudget(last);assert.ok(!this.budgetExceeded,'observed campaign bound exceeded; cooperative cancellation requested');this.ended=new Date().toISOString();assert.ok(this.samples.length>=2,'actual observation series required');
    const pid=this.service.worker.pid, deviceId=last.deviceId, cumulative=last.method==='mlx-allocator-cumulative-high-water';
    assert.ok(this.samples.every(s=>s.pid===pid && s.method===last.method && s.deviceId===deviceId),'do not misattribute different worker lifetimes/devices to a single peak window');
    const make=(device,phase)=>({method:device?last.method:'sampled-process-rss',sampled:device&&cumulative?false:true,intervalMs:observedInterval(this.samples,device,this.intervalMs),targetPauseMs:this.intervalMs,samples:this.samples.filter(s=>Number.isFinite(device?s.device:s.host)).map(s=>({at:s.at,probeStartedAt:s.probeStartedAt,bytes:device?s.device:s.host,...(device&&s.deviceCounterAt?{counterWrittenAt:s.deviceCounterAt}:{})})),blindspots:[...this.gaps.map(g=>`${g.at}: ${g.error}`),'intervalMs is the maximum observed valid-sample gap including probe latency; targetPauseMs is only the configured delay. Host/device reads occur within the retained probeStartedAt..at window. Process sampling may miss transients; process-lifetime allocator counters include prior allocations and allocator-only attribution.'],window:{start:this.started,end:this.ended,phase,coverage:phase==='loading'?'full-case-load-inclusive':'full-case'},bounds:[...(phase==='loading'?['full-case-load-inclusive-upper-bound']:[]),...(device&&!cumulative?['observed-device-global-bound']:[])],attribution:{scope:device?(cumulative?'process-lifetime':'device'):'process',pid,deviceId:device?deviceId:undefined,includesPriorAllocations:device&&cumulative,includesOtherProcesses:device&&!cumulative,meaning:device?(cumulative?'Existing owned worker gpu_telemetry.json peakBytes, fresh after completion; process-lifetime MLX allocator highwater, not phase-local or residency delta':'Observed device-global used-memory samples; includes other processes, not model-attributable allocation'):'RSS of the owned single worker process, API/other processes excluded'}});
    const hostPeakBytes=Math.max(...this.samples.map(s=>s.host)),devicePeakBytes=Math.max(...this.samples.filter(s=>Number.isFinite(s.device)).map(s=>s.device));assert.ok(hostPeakBytes>0&&devicePeakBytes>0);
    return {hostPeakBytes,devicePeakBytes,loadHostPeakBytes:hostPeakBytes,loadDevicePeakBytes:devicePeakBytes,observations:{host:make(false,'case'),device:make(true,'case'),loadHost:make(false,'loading'),loadDevice:make(true,'loading')}};
  }
}
export function loadIdentity(event,job,inputs) {
  const closure=normalizeLoadEvent(event);assert.equal(closure.jobId,job.id);assert.equal(job.status,'completed');
  if(job.payload?.mode!==undefined)assert.equal(closure.operation,job.payload.mode,'observed native operation differs from actual queued job');
  if(job.payload?.seed!==undefined)assert.equal(closure.seed,job.payload.seed,'observed native seed differs from actual queued job');
  if(job.payload?.advanced?.steps!==undefined)assert.equal(closure.steps,Number(job.payload.advanced.steps),'observed native NFE differs from actual queued recipe');
  if(job.payload?.advanced?.schedulerShift!==undefined)assert.equal(closure.schedulerShift,Number(job.payload.advanced.schedulerShift),'observed native video shift differs from actual queued recipe');
  const checkpoint=inputs.find(i=>i.role==='checkpoint' && closure.identity.sourceFiles.some(s=>s.sha256===i.sha256));assert.ok(checkpoint,'actual observed source closure must contain verified imported checkpoint');
  for(const f of closure.identity.sourceFiles)assert.ok(inputs.some(i=>i.sha256===f.sha256),'observed source not in verified input closure');
  const assembly=closure.identity.assembly;assert.ok(assembly && typeof assembly==='object');
  const pins=role=>closure.prepared_sources.filter(s=>inputs.some(i=>i.role===role&&i.sha256===s.sha256));
  return {checkpointSha256:checkpoint.sha256,planId:closure.identity.planId,semanticDigest:closure.identity.semanticDigest,topology:assembly.adalnPruning??{kind:'no-pruning-binding-in-observed-assembly'},quantization:{sourceEncoding:assembly.sourceEncoding??null,observedAssembly:assembly},components:pins('component'),adapters:pins('adapter'),operation:closure.operation,samplingPolicy:{steps:closure.steps,schedulerShift:closure.schedulerShift},seed:closure.seed};
}
export async function bindCompletedMedia(service,job,dir,name,decode=decodeAV) {
  assert.equal(job.status,'completed');const ids=job.result?.assetIds??job.result?.assets?.map(a=>a.id);assert.ok(ids?.length,'completed actual result asset IDs required');
  const bound=[];
  for(const id of ids){
    const asset=await service.request('GET',`/api/v1/projects/${encodeURIComponent(job.projectId??job.payload.projectId)}/assets/${encodeURIComponent(id)}`);
    assert.equal(asset.id,id);if(asset.type!=='video')continue;
    const project=await service.request('GET',`/api/v1/projects/${encodeURIComponent(asset.projectId)}`);assert.equal(project.id,asset.projectId);
    const projectRoot=await confined(service.runDir,project.path),source=await confined(projectRoot,path.resolve(projectRoot,asset.file.path));
    const av=`${name}-${bound.length}${path.extname(source)}`,dest=inside(dir,av);await copyFile(source,dest);const decoded=await decode(dest);
    const video=decoded.streams?.find(s=>s.codec_type==='video');assert.ok(video,'actual decoded video metadata required');
    for(const axis of ['width','height'])if(job.payload?.[axis]!==undefined)assert.equal(video[axis],job.payload[axis],'decoded canvas differs from the resolved queued geometry');
    bound.push({jobId:job.id,assetId:id,projectId:asset.projectId,source,av,sha256:await hashFile(dest),decoded,asset});
  }
  assert.ok(bound.length,'no actual synchronized video asset in the completed job');return bound;
}
export async function verifyAPIBindings(p,c,dir,decode=decodeAV,delta, {allowUnitFake=false,contentReadback=false}={}) {
  const bundle=await json(inside(dir,'source-job-output.json'));
  assert.ok(bundle.proof_kind==='campaign_native'||(allowUnitFake&&bundle.proof_kind==='unit_fake'),'unit_fake is never campaign proof');
  assert.equal(bundle.build.sceneworks,p.sceneworks);assert.equal(bundle.build.inference,p.inference);
  if(!contentReadback){await confined(p.output,bundle.build.binary);assert.equal(await hashFile(bundle.build.binary),bundle.build.sha256,'owned service binary drift');}
  const lifecycle=await json(inside(dir,'service-lifecycle.json'));for(const record of lifecycle.drains)assert.equal(await hashFile(inside(dir,record.path)),record.sha256,'owned cancellation/restart record drift');
  if(c.id.endsWith('.restart-replay'))assert.ok(bundle.epochs.length>=2 && bundle.actions.some(v=>v.kind==='restart'),'restart/replay requires actual owned restart records');
  if(c.id.endsWith('.interruption-retry'))assert.ok(bundle.epochs.length>=2 && bundle.actions.some(v=>v.kind==='interrupt-import'),'import interruption/retry requires actual owned cooperative cancellation/restart records');
  const byJob=new Map();
  for(const binding of bundle.bindings) {
    const identity=loadIdentity(binding.event,binding.job,c.inputs);assert.deepEqual(identity,binding.runtimeIdentity);
    const selected=c.declaredSourceBindings?.find(v=>v.model.modelId===binding.job.payload.model);
    if(c.declaredSourceBindings){
      assert.ok(selected,'completed job selected an undeclared/stock model');assert.equal(identity.checkpointSha256,selected.model.sha256);
      const request=bundle.requests?.find(r=>r.result.id===binding.job.id&&r.route==='/api/v1/video/jobs')?.request;
      const requested=request?.loras??binding.job.payload.loras??[];
      const adapters=requestedAdapterBindings(selected.adapters??[],requested),prepared=normalizeLoadEvent(binding.event).prepared_sources;
      for(const input of [...(selected.model.components??[]),...adapters])assert.ok(prepared.some(s=>s.sha256===input.sha256),'requested component/adapter missing from actual prepared closure');
      for(const input of identity.adapters)assert.ok(adapters.some(a=>a.sha256===input.sha256),'unrequested adapter leaked into native cache/load');
    }
    assert.equal(normalizeLoadEvent(binding.event).backend,c.backend);assert.equal(binding.media.jobId,binding.job.id);
    assert.equal(await hashFile(inside(dir,binding.media.av)),binding.media.sha256);await decode(inside(dir,binding.media.av));byJob.set(binding.job.id,binding);
  }
  for(const media of bundle.media) {
    assert.equal(await hashFile(inside(dir,media.av)),media.sha256);const decoded=await decode(inside(dir,media.av)),job=bundle.jobs.find(v=>v.id===media.jobId);assert.ok(job?.status==='completed');
    const video=decoded.streams?.find(s=>s.codec_type==='video');assert.ok(video);for(const axis of ['width','height'])if(job.payload?.[axis]!==undefined)assert.equal(video[axis],job.payload[axis]);
  }
  const comparisons=await json(inside(dir,'comparisons.json'));
  for(const comparison of comparisons) {
    const b=byJob.get(comparison.baselineJobId),r=byJob.get(comparison.repeatedJobId),i=byJob.get(comparison.interventionJobId);assert.ok(b&&r&&i);
    assert.deepEqual(creativePayload(submittedPayload(bundle.requests??[],b.job)),creativePayload(submittedPayload(bundle.requests??[],r.job)));
    assertControlledConfiguration(submittedPayload(bundle.requests??[],b.job),submittedPayload(bundle.requests??[],i.job),comparison.allowedChanges);
    for(const other of [r,i]){assert.equal(other.runtimeIdentity.checkpointSha256,b.runtimeIdentity.checkpointSha256);assert.equal(other.runtimeIdentity.seed,b.runtimeIdentity.seed);assert.deepEqual(other.runtimeIdentity.samplingPolicy,b.runtimeIdentity.samplingPolicy);}
    const actual=await compareControlMedia(inside(dir,b.media.av),inside(dir,r.media.av),inside(dir,i.media.av),comparison.stream,comparison.minimumDelta,comparison.maximumBaselineDelta,decode,delta);
    if(!contentReadback)for(const key of ['interventionDelta','repeatedBaselineDelta','threshold'])assert.equal(actual[key],comparison[key],'retained sensitivity differs from actual decoded media'); // Portable decoding rechecks declared bounds, not a host-specific ffmpeg scalar golden.
  }
  return {jobs:bundle.media.map(v=>v.jobId),nativeClosures:bundle.bindings.length,comparisons:comparisons.length};
}
export async function executeAPICase(p,c,swRoot,inferenceRoot,dir,deps={}) {
  const service=await new OwnedService(p,c,swRoot,inferenceRoot,dir,deps).initialize();
  const observations=new ProcessObservations(service,{sample:deps.sample??processSample,sleepImpl:deps.sleep??sleep,intervalMs:c.runtime.sampleIntervalMs??250}).start();
  const records=[],actions=[],bound=[],traceEvents=[];let result,metrics,error;
  try {
    const scenario=await json(c.runtime.scenario);assert.equal(scenario.caseId,c.id);assert.equal(scenario.baseUrl,service.baseUrl,'scenario must target newly owned service');
    result=await runScenario(scenario,{fetchImpl:(url,options)=>service.fetch(url,options),sleep:deps.sleep??sleep,beforeStep:async ({step,captures})=>{assert.ok(!observations.budgetExceeded,'campaign observation bound exceeded; no further job submission');await service.checkStep(step,captures);for(const action of [...(scenario.lifecycleActions??[]),...(c.runtime.actions??[])])if(action.beforeStep===step.id)actions.push(await service.action(action,captures));},onRecord:async r=>{
      records.push(r);
      if(r.result?.id && r.method==='POST' && (r.route==='/api/v1/models/import'||r.route==='/api/v1/video/jobs'||/\/(retry|duplicate)$/.test(r.route)))service.jobs.add(r.result.id);
      await writeFile(path.join(dir,'api-transcript.json'),JSON.stringify({caseId:c.id,proof_kind:service.fake?'unit_fake':'campaign_native',transcript:records},null,2));
    }});
    if(c.id.endsWith('.cancel')) {
      const imported=records.filter(r=>r.method==='POST'&&r.route==='/api/v1/models/import').map(r=>r.result.id);
      assert.ok(imported.some(id=>records.some(r=>r.method==='POST'&&r.route===`/api/v1/jobs/${encodeURIComponent(id)}/cancel`)&&Object.values(result.captures).some(j=>j?.id===id&&j.status==='canceled')),'cancel case requires an actually cancelled submitted import, not import-only success');
    }
    const jobs=[...new Map(Object.values(result.captures).filter(v=>v?.id&&v.status==='completed'&&v.payload?.model).map(j=>[j.id,j])).values()];
    const measured=['artifact','capability'].includes(c.group)||/\.(bounded-loading|cold-warm-staged)$/.test(c.id);
    if(measured)metrics=await observations.finish();
    else {observations.done=true;await observations.loop;assert.ok(!observations.budgetExceeded,'observed campaign bound exceeded');}
    await save(path.join(dir,'raw-observations.json'),{samples:observations.samples,gaps:observations.gaps,epochs:service.epochs,meaning:measured?'Whole case/load-inclusive observations':'Observed only; no successful model load or peak is invented for a refusal/cancel or multi-process restart'});
    for(const [i,job] of jobs.entries())bound.push(...await bindCompletedMedia(service,job,dir,`output-${i}`,deps.decode??decodeAV));
    // Collect actual JSON tracing after the completed jobs; no synthetic load event is created.
    const traceLogs=[service.api.log(),service.worker.log()];
    for(const epoch of service.epochs.slice(0,-1))for(const kind of ['api','worker'])traceLogs.push(await readFile(path.join(dir,`${kind}-${epoch.epoch}.log`),'utf8'));
    for(const text of traceLogs)for(const line of text.split('\n')){try{const event=JSON.parse(line);if((event.fields??event).event==='h3_import_prepared_load')traceEvents.push(event);}catch{}}
    const loads=bound.map(b=>({media:b,job:jobs.find(j=>j.id===b.jobId),event:traceEvents.find(t=>(t.fields??t).jobId===b.jobId)}));
    const successful=['artifact','capability'].includes(c.group)||/\.(bounded-loading|cold-warm-staged)$/.test(c.id);
    const matched=loads.filter(v=>v.event);if(successful)assert.equal(matched.length,loads.length,'each actual generation needs observed native load closure');
    if(successful)assert.ok(matched.length>0,'no actual successful model execution');
    const proofs=matched.map(v=>({...v,runtimeIdentity:loadIdentity(v.event,v.job,c.inputs)}));
    await save(path.join(dir,'source-job-output.json'),{proof_kind:service.fake?'unit_fake':'campaign_native',build:service.buildIdentity,epochs:service.epochs,bindings:proofs,jobs,requests:records.filter(r=>r.method==='POST'&&r.result?.id),media:bound,actions,metrics});
    const comparisons=[];
    for(const spec of scenario.comparisons??[]) {
      const find=name=>{const job=result.captures[name];const value=proofs.find(p=>p.job.id===job?.id);assert.ok(value,'actual completed comparison job/load/output binding missing');return value;};
      const baseline=find(spec.baseline),repeat=find(spec.repeatedBaseline);
      for(const contrast of spec.interventions){const changed=find(typeof contrast==='string'?contrast:contrast.job);assert.equal(changed.runtimeIdentity.checkpointSha256,baseline.runtimeIdentity.checkpointSha256);assert.equal(repeat.runtimeIdentity.checkpointSha256,baseline.runtimeIdentity.checkpointSha256);assert.equal(changed.runtimeIdentity.seed,baseline.runtimeIdentity.seed);assert.deepEqual(changed.runtimeIdentity.samplingPolicy,baseline.runtimeIdentity.samplingPolicy);assert.deepEqual(creativePayload(submittedPayload(records,repeat.job)),creativePayload(submittedPayload(records,baseline.job)),'unchanged repeated baseline creative request required');assertControlledConfiguration(submittedPayload(records,baseline.job),submittedPayload(records,changed.job),contrast.allowedChanges);
        comparisons.push({baselineJobId:baseline.job.id,repeatedJobId:repeat.job.id,interventionJobId:changed.job.id,allowedChanges:contrast.allowedChanges??conditionKeys,minimumDelta:spec.thresholds.minimumControlDelta,maximumBaselineDelta:spec.thresholds.maximumBaselineDelta,...await compareControlMedia(inside(dir,baseline.media.av),inside(dir,repeat.media.av),inside(dir,changed.media.av),contrast.stream??spec.stream,spec.thresholds.minimumControlDelta,spec.thresholds.maximumBaselineDelta,deps.decode??decodeAV,deps.delta)});
      }
    }
    await save(path.join(dir,'comparisons.json'),comparisons);
    const primary=proofs[0];const h=await (deps.hardware??hardware)(c.backend,c.runtime.deviceId);
    const e={status:'executed',proof_kind:service.fake?'unit_fake':'campaign_native',sceneworks:p.sceneworks,inference:p.inference,caseId:c.id,backend:c.backend,hardware:h,runtimeBindings:'source-job-output.json',files:[...bound.map(v=>v.av),'api-transcript.json','source-job-output.json','comparisons.json','service-build.json','service-build.log','raw-observations.json','service-lifecycle.json',...service.epochs.flatMap(v=>[`api-${v.epoch}.log`,`worker-${v.epoch}.log`,`guardian-${v.epoch}.log`,`ownership-${v.epoch}.json`,`exits-${v.epoch}.json`])],apiTranscript:'api-transcript.json',testInputSha256:c.inputs[0].sha256,metrics};
    if(primary){Object.assign(e,{runtimeIdentity:primary.runtimeIdentity,loadEvent:primary.event,completedJob:{id:primary.job.id,status:primary.job.status,providerId:normalizeLoadEvent(primary.event).providerId,output:primary.media.av},av:primary.media.av});}
    if(c.artifact)Object.assign(e,{source:p.source,artifact:c.artifact});
    if(c.group==='capability') {assert.ok(comparisons.length);const control=proofs.find(v=>v.job.id===comparisons[0].interventionJobId);e.control={av:control.media.av,checkpointSha256:control.runtimeIdentity.checkpointSha256,seed:control.runtimeIdentity.seed,intervention:'Actual scenario comparison manifest; see all source-bound contrasts',comparisons:'comparisons.json',bindings:'source-job-output.json'};}
    await save(path.join(dir,c.evidence),e);return {evidence:e,result,comparisons};
  } catch(e){error=e;throw e;} finally {
    observations.done=true;await observations.loop;
    let observationWriteError;
    try{await save(path.join(dir,'sampler-exit.json'),{samples:observations.samples,gaps:observations.gaps,budgetExceeded:observations.budgetExceeded,proof_kind:service.fake?'unit_fake':'campaign_native'});}catch(e){observationWriteError=e;}
    try{await service.stop();await save(path.join(dir,'service-lifecycle.json'),{epochs:service.epochs,quiescence:service.quiescence,drains:await Promise.all(service.drainRecords.map(async name=>({path:name,sha256:await hashFile(path.join(dir,name))}))),observationWriteError:observationWriteError?.message,proof_kind:service.fake?'unit_fake':'campaign_native'});if(observationWriteError&&!error)throw observationWriteError;}catch(stopError){await save(path.join(dir,'shutdown-failure.json'),{error:stopError.message,originalError:error?.message,epochs:service.epochs});if(!error)throw stopError;}
  }
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  try{const [mode,planPath,id,swRoot,inferenceRoot]=process.argv.slice(2);const p=await json(planPath),c=p.cases.find(v=>v.id===id);assert.ok(c);authorized(p,c);const dir=process.env.H3_CASE_DIR;assert.equal(process.env.H3_CASE_ID,id);assert.ok(dir);await confined(p.output,dir);
    if(mode==='numeric'){const r=await executeNativeComparison(p,c,swRoot,inferenceRoot,dir);console.log(r.log);}
    else if(mode==='fixture'){const r=await executeFixture(p,c,swRoot,inferenceRoot,dir);console.log(r.log);}
    else{assert.equal(mode,'api');const r=await executeAPICase(p,c,swRoot,inferenceRoot,dir);console.log(`H3_API_ASSERTIONS_EXECUTED=${r.result.transcript.filter(v=>v.assertion).length}`);}
  }catch(error){console.error(error.stack);process.exitCode=1;}
}
