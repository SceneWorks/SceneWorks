// sc-26691 adapters for the existing harness. Production entry is explicit Actions-only.
// No model implementation/importer/registry: own processes, existing API, native tests and receipts.
import assert from 'node:assert/strict';
import { campaignProfileArgs, CAMPAIGN_PROFILE_DIR } from './lib/h3-campaign-build-profile.mjs';
import path from 'node:path';
import net from 'node:net';
import { randomUUID,createHash } from 'node:crypto';
import { createWriteStream } from 'node:fs';
import { spawn, execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdir, open, readFile, writeFile, realpath, stat, copyFile, rename, link } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { hashFile, sourcePair, verifyInputs, runProcess, executedTests, normalizeLoadEvent, decodeAV, decodedDelta, validateAVAgainstRequest, compareControlMedia, isSourceCancelCase, allowedInputRoots, verifiedStagingRoot, authorized, executionCases } from './h3-finetune-terminal.mjs';
import { runScenario, expand } from './h3-finetune-terminal-api.mjs';
import { caseBook,assertLicenseAuthority } from './h3-finetune-terminal-cases.mjs';
import { cudaDeviceEvidence, MANUAL_CUDA_PHYSICAL_GPU } from './ci/h3-import-native-proof.mjs';
import { privateRunRoot } from './h3-finetune-terminal-stage.mjs';
// Windows: the bootstrap output root is ~95 characters deep and cargo/nvcc nest candle's kernel
// build ~50 more below CARGO_TARGET_DIR, so a target under the output root overruns MAX_PATH.
// There the target is a short sibling of this run's private root (`<run>-<attempt>-<kind>`).
export function cargoTarget(c,kind,longPath,platform=process.platform){
  return platform==='win32'?`${privateRunRoot({backend:c.backend,platform,env:process.env,runId:process.env.GITHUB_RUN_ID,attempt:process.env.GITHUB_RUN_ATTEMPT})}-${kind}`:longPath;
}
// A unit_fake run never resolves the real Windows private root (E:\huggingface\sc26691-private):
// unless a test names a platform explicitly, it takes the POSIX layout under its own output root.
export const UNIT_FAKE_PLATFORM = 'linux';
export function runPlatform(deps={}){return deps.platform??(deps.proof_kind==='unit_fake'?UNIT_FAKE_PLATFORM:process.platform);}
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
// ---------------------------------------------------------------------------
// Cold generation source-preparation cancellation (lifecycle.generation-source-prep-cancel).
//
// This binds the REAL SceneWorks contract, not a label or callback count:
//   * span `h3_import_preparation` (jobId, checkpointId, phase="source-and-normalization")
//     is entered INSIDE the detached spawn_blocking preparation task;
//   * `artifact_seal_hash_progress` events carry path, acquisition ("ColdContentRead"),
//     state ("Reading"|"Acquired"|"Canceled"|"Failed"), hashed_bytes, source_bytes.
//     Native cold reads check the cancel flag every CHUNK bytes, so a cooperative cancel is
//     acknowledged at the NEXT checkpoint — source-derived, not an invented timeout;
//   * `h3_normalization_copy_progress` carries Writing|Canceled|Failed plus
//     copied_payload_bytes/total_payload_bytes — but a cold HASH cancel happens BEFORE
//     codec copying, so a valid hash-only cancel emits NONE of these events;
//   * `h3_import_source_prepared` / `h3_import_prepared_load` must be ABSENT for a
//     canceled preparation;
//   * the root job `.message` marker is posted only after `CancelJoinGuard::cancel_and_join`
//     has awaited the detached task's exit. Its real bound is CANCEL_JOIN_GRACE (30s) plus
//     CANCEL_JOIN_ABANDON (5s) in `crates/sceneworks-worker/src/progress.rs`; a hard-abandoned
//     task cannot prove lease release, so acknowledgement must arrive UNDER the 30s grace.
// ---------------------------------------------------------------------------
export const SOURCE_PREP_CHUNK_BYTES = 8 * 1024 * 1024; // native ArtifactSeal hash chunk; not a callback-count proxy
export const PREP_CANCEL_ACK_MS = 30000;            // CancelJoinGuard::CANCEL_JOIN_GRACE
export const PREP_CANCEL_ABANDON_MS = 5000;         // CancelJoinGuard::CANCEL_JOIN_ABANDON
export const PREP_SPAN_PHASE = 'source-and-normalization';
// The owned service's shutdown grace (SCENEWORKS_WORKER_SHUTDOWN_TIMEOUT_SECONDS). The API spends up
// to its connection drain (`API_CONNECTION_DRAIN_TIMEOUT`, 5 s) plus this grace stopping its
// in-process utility pool, and the GPU worker up to this grace winding a job down, so the harness's
// graceful-exit bound must be STRICTLY larger than both or it times out on a shutdown the service
// is still entitled to (sc-26691: a 120 s bound against a 120 s grace).
export const SERVICE_SHUTDOWN_GRACE_SECONDS = 120;
export const API_CONNECTION_DRAIN_SECONDS = 5;
export const SHUTDOWN_BOUND_HEADROOM_SECONDS = 60;
/** Milliseconds the harness waits for an owned API or worker to exit after its shutdown request. */
export function ownedExitBoundMs(graceSeconds = SERVICE_SHUTDOWN_GRACE_SECONDS) {
  assert.ok(Number.isInteger(graceSeconds) && graceSeconds > 0);
  return (graceSeconds + API_CONNECTION_DRAIN_SECONDS + SHUTDOWN_BOUND_HEADROOM_SECONDS) * 1000;
}
export const PREP_CANCELED_MARKER = 'h3-preparation-canceled';
const receiptNameFor = jobId => `source-cancel-${encodeURIComponent(jobId)}.json`;
const traceFields = event => event?.fields ?? event ?? {};
export function observedSourceBytes(value){if(value===undefined||value===null||value==='None')return null;const number=typeof value==='number'?value:/^\d+$/.test(String(value))?Number(value):NaN;return Number.isSafeInteger(number)&&number>=0?number:null;}
const traceSpan = event => [event?.span,...(Array.isArray(event?.spans)?[...event.spans].reverse():[]),event].find(span=>span&&span.phase!==undefined)??{};
// Only actual cold source-hash progress inside the preparation span is admissible.
function coldSourceProgress(event) {
  const fields = traceFields(event);
  if (fields.event !== 'artifact_seal_hash_progress') return null;
  const span = traceSpan(event);
  return span.phase === PREP_SPAN_PHASE ? { fields, span } : null;
}
function ownedBelow(root, value) {
  const paths=path.win32.isAbsolute(root)?path.win32:path.posix;
  if(typeof value!=='string'||!paths.isAbsolute(value))return undefined;
  const resolved=paths.resolve(value),relative=paths.relative(paths.resolve(root),resolved);
  return relative&&!relative.startsWith('..')&&!paths.isAbsolute(relative)?resolved:undefined;
}
export {authorized};
export async function hardware(backend, deviceId, deps={}) {
  const execute=deps.exec??exec;
  if (backend === 'mlx') {
    const gpu = JSON.parse((await execute('system_profiler',['SPDisplaysDataType','-json'])).stdout).SPDisplaysDataType;
    assert.ok(gpu?.length); return {device:gpu[0].sppci_model ?? gpu[0]._name,driver:`Apple Metal driver bundled with macOS ${(await execute('sw_vers',['-productVersion'])).stdout.trim()}`};
  }
  const inventory=(await execute('nvidia-smi',['--query-gpu=index,uuid,pci.bus_id,name,driver_version','--format=csv,noheader'])).stdout;
  const cuda=cudaDeviceEvidence(deps.env??process.env,inventory),row=cuda.selectedPhysicalDevice;
  assert.equal(row.uuid,deviceId,'requested CUDA UUID must be the actual PCI-ordered manual physical device 1');
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
    for(const item of proof.supporting){assert.equal(item.passed,1);assert.ok(item.command.includes('--exact')&&!item.command.includes('--ignored'),'portable value-bearing mapping assertion must execute exactly, not be ignored');assert.ok(item.command.includes('minimax_h3_codec::tests::actual_native_bytes_normalize_through_persisted_plan_and_shared_cache_in_all_precisions'),'QKV/FFN exact-byte codec coverage is mandatory on both backends');}
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
  if(!fake) { authorized(p,c); await sourcePair(p,swRoot,inferenceRoot); await verifyInputs(c,await allowedInputRoots(p,swRoot,inferenceRoot)); }
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
  const base=['cargo','test','--locked',...campaignProfileArgs(),'--manifest-path',path.join(inferenceRoot,'Cargo.toml'),'-p',n.package,'--test','integration'];
  assert.equal(n.package,c.backend==='mlx'?'mlx-gen-minimax-h3':'candle-gen-minimax-h3');
  if(c.backend==='cuda') base.push('--features','cuda');
  const buildCommand=[...base,'--no-run','--message-format=json'];
  const opts={cwd:inferenceRoot,env:{...process.env,...(c.backend==='cuda'?{RUSTC_WRAPPER:''}:{}),CARGO_TARGET_DIR:cargoTarget(c,'native',path.join(p.output,`native-target-${p.inference}-${c.backend}`),runPlatform(deps)),CARGO_PROFILE_TEST_DEBUG:'0',CARGO_PROFILE_DEV_DEBUG:'0'},timeoutSeconds:c.timeoutSeconds};
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
    const text=await readFile(path.join(dir,log),'utf8');assert.equal(executedTests(text),1);assert.ok(!/\bSKIP\b/.test(text));assert.ok(text.includes('test minimax_h3_codec::tests::actual_native_bytes_normalize_through_persisted_plan_and_shared_cache_in_all_precisions ...'),'exact portable QKV/FFN value assertion did not execute');supporting.push({command,log,passed:1});
  }
  const fixtureCases=[];
  for(const input of fixtureInputs) { const f=await json(input.path); for(const [i,v] of (f.cases??[]).entries()) fixtureCases.push({fixtureSha256:input.sha256,id:v.id??{index:i,group:v.group,dtype:v.dtype}}); }
  const proof={kind:n.kind,proof_kind:fake?'unit_fake':'campaign_native',backend:c.backend,sceneworks:p.sceneworks,inference:p.inference,comparisonKind:'native-assertions-against-precomputed-pinned-cpu-reference',referenceRevision:c.referenceRevision,tests:n.tests,fixtureCases,fixtures:fixtureInputs,referenceBlobs:refs,binary:{path:binary,sha256:binaryHash,buildCommand,configuration:{backend:c.backend,package:n.package,testTarget:'integration',features:c.backend==='cuda'?['cuda']:[],locked:true,cargoArtifact:{target:artifact.target,profile:artifact.profile,features:artifact.features??[]},buildEnvironment:Object.fromEntries(['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_TARGET','RUSTUP_TOOLCHAIN'].map(k=>[k,process.env[k]??null]))}},inventoryLog:'native-inventory.log',runs,supporting};
  validateComparisonProof(c,proof,logs.join('\n'),{allowUnitFake:fake});
  const files=['native-build.log','native-inventory.log',...runs.map(r=>r.log),...supporting.map(r=>r.log)];
  await save(path.join(dir,'native-comparison.json'),proof);
  const e={status:'executed',proof_kind:proof.proof_kind,caseId:c.id,backend:c.backend,sceneworks:p.sceneworks,inference:p.inference,hardware:h,testInputSha256:fixtureInputs[0].sha256,files:[...files,'native-comparison.json'],nativeComparison:'native-comparison.json'};
  await save(path.join(dir,c.evidence),e);
  if(!fake) {await sourcePair(p,swRoot,inferenceRoot);await verifyInputs(c,await allowedInputRoots(p,swRoot,inferenceRoot));}
  return {evidence:e,proof,log:logs.join('\n')};
}

export async function executeFixture(p,c,swRoot,inferenceRoot,dir,deps={}) {
  const fake=deps.proof_kind==='unit_fake';if(!fake){authorized(p,c);await sourcePair(p,swRoot,inferenceRoot);await verifyInputs(c,await allowedInputRoots(p,swRoot,inferenceRoot));}
  assert.equal(c.fixtureProof.kind,'source-bound-assertions');assert.equal(c.command[0],'cargo');assert.ok(c.command.includes('test')&&c.command.includes(path.join(swRoot,'Cargo.toml')));
  assert.ok(c.inputs.length);for(const input of c.inputs)assert.equal(await hashFile(input.path),input.sha256);
  const state=path.join(dir,'fixture-state');await mkdir(state);
  const env={...privateEnvironment(state,{port:43210,gpuId:'cpu',backend:c.backend},randomUUID()),CARGO_TARGET_DIR:cargoTarget(c,'fixture',path.join(p.output,`fixture-target-${p.sceneworks}-${p.inference}-${c.backend}`),runPlatform(deps)),CARGO_PROFILE_TEST_DEBUG:'0',CARGO_PROFILE_DEV_DEBUG:'0'};
  await (deps.runProcess??runProcess)(c.command,{cwd:swRoot,env,logPath:path.join(dir,'assertions.log'),timeoutSeconds:c.timeoutSeconds});
  const log=await readFile(path.join(dir,'assertions.log'),'utf8'),passed=executedTests(log);assert.ok(!/\bSKIP\b/.test(log));
  const proof={kind:'source-bound-assertions',proof_kind:fake?'unit_fake':'campaign_fixture',sceneworks:p.sceneworks,inference:p.inference,command:c.command,passed,inputs:c.inputs,meaning:'Executed source-bound synthetic contract assertions; no real-model/native load or measured peak claimed'};
  await save(path.join(dir,'assertion-proof.json'),proof);
  const e={status:'executed',proof_kind:proof.proof_kind,caseId:c.id,backend:c.backend,sceneworks:p.sceneworks,inference:p.inference,hardware:await(deps.hardware??hardware)(c.backend,c.fixtureProof.deviceId),testInputSha256:c.inputs[0].sha256,files:['assertions.log','assertion-proof.json']};
  await save(path.join(dir,c.evidence),e);if(!fake){await sourcePair(p,swRoot,inferenceRoot);await verifyInputs(c,await allowedInputRoots(p,swRoot,inferenceRoot));}return {evidence:e,proof,log};
}
export function spawnOwned(argv,env,cwd,logPath) {
  const child=spawn(argv[0],argv.slice(1),{env,cwd,stdio:['pipe','pipe','pipe'],shell:false});
  const chunks=[]; let failure;
  const output=createWriteStream(logPath,{flags:'wx'});output.on('error',e=>{failure=e;});
  child.stdout.pipe(output,{end:false});child.stderr.pipe(output,{end:false});
  child.on('error',e=>{failure=e;});
  const exited=new Promise(resolve=>child.on('close',(code,signal)=>resolve({code,signal})));
  for(const stream of [child.stdout,child.stderr]) stream.on('data',b=>chunks.push(Buffer.from(b)));
  return {pid:child.pid,exitCode:()=>child.exitCode??child.signalCode, // A signal death (SIGABRT on a stack overflow) has a null exitCode.
    stdin:child.stdin,signal:s=>child.kill(s),exit:async()=>{const result=await exited;await new Promise(resolve=>output.end(resolve));if(failure)throw failure;return result;},log:()=>Buffer.concat(chunks).toString('utf8')};
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
  for(const key of ['PATH','RUSTUP_TOOLCHAIN','SystemRoot','WINDIR','COMSPEC','PATHEXT','TMPDIR','TEMP','TMP','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','MLX_METAL_DEVICE_INDEX','SCENEWORKS_GPU_MEMORY_LIMIT_BYTES','SCENEWORKS_CUDA_VRAM_CAP_GB','SCENEWORKS_MLX_MEMORY_CAP_GB','LD_LIBRARY_PATH','DYLD_LIBRARY_PATH']) if(process.env[key]!==undefined)env[key]=process.env[key];
  Object.assign(env,{SCENEWORKS_API_HOST:'127.0.0.1',SCENEWORKS_API_PORT:String(runtime.port),SCENEWORKS_API_URL:`http://127.0.0.1:${runtime.port}`,SCENEWORKS_DATA_DIR:path.join(runDir,'data'),SCENEWORKS_CONFIG_DIR:path.join(runDir,'config'),SCENEWORKS_JOBS_DB_PATH:path.join(runDir,'data','cache','jobs.db'),SCENEWORKS_ACCESS_TOKEN:token,SCENEWORKS_RUN_UTILITY_INPROCESS:'1',SCENEWORKS_UTILITY_WORKERS:'1',SCENEWORKS_WORKER_ID:`sc26691-${path.basename(runDir)}`,SCENEWORKS_GPU_ID:runtime.gpuId,SCENEWORKS_BACKEND_MLX_ENABLED:runtime.backend==='mlx'?'1':'0',SCENEWORKS_BACKEND_CANDLE_ENABLED:runtime.backend==='cuda'?'1':'0',SCENEWORKS_LOG_FORMAT:'json',SCENEWORKS_WORKER_SHUTDOWN_TIMEOUT_SECONDS:String(SERVICE_SHUTDOWN_GRACE_SECONDS),RUST_LOG:'info,sceneworks=debug',HF_HOME:path.join(runDir,'hf'),HF_HUB_CACHE:path.join(runDir,'hf','hub'),HUGGINGFACE_HUB_CACHE:path.join(runDir,'hf','hub'),TRANSFORMERS_CACHE:path.join(runDir,'hf','transformers'),XDG_CACHE_HOME:path.join(runDir,'cache'),HF_HUB_OFFLINE:'1',HF_DATASETS_OFFLINE:'1'});
  // Mirror the shipped desktop (apps/desktop/src/setup.rs). The API runs its in-process utility
  // pool, which is the ONLY worker that serves model_import/model_download/model_convert/lora_import
  // — with it off, nothing could ever claim an import and the first real case waited out its whole
  // deadline in `queued`. macOS requires MLX; off-Mac requires Candle and enforces unsupported shapes.
  if(runtime.backend==='mlx')env.SCENEWORKS_MLX_REQUIRED='1';
  else Object.assign(env,{SCENEWORKS_CANDLE_REQUIRED:'1',SCENEWORKS_CANDLE_UNSUPPORTED_MODE:'enforce'});
  if(runtime.memoryCap){assert.equal(runtime.memoryCap.kind,runtime.backend==='cuda'?'cuda-vram-gb':'mlx-memory-gb');assert.ok(Number.isFinite(runtime.memoryCap.value)&&runtime.memoryCap.value>0);env[runtime.backend==='cuda'?'SCENEWORKS_CUDA_VRAM_CAP_GB':'SCENEWORKS_MLX_MEMORY_CAP_GB']=String(runtime.memoryCap.value);}
  return env;
}
async function confinedTaskSource(p,value,{allowRoot=false}={}) {
  for(const root of [p.output,...(p.assetStaging?[await verifiedStagingRoot(p)]:[])]){const rel=path.relative(path.resolve(root),path.resolve(value));if(allowRoot&&rel==='')return realpath(root);if(rel&&!rel.startsWith('..')&&!path.isAbsolute(rel))return confined(root,value);}
  throw new Error('source outside task output or verified private staging root');
}
export class OwnedService {
  constructor(p,c,swRoot,inferenceRoot,dir,deps={}) {
    Object.assign(this,{p,c,swRoot,inferenceRoot,dir,deps});this.fake=deps.proof_kind==='unit_fake';this.jobs=new Set();this.epochs=[];this.drainRecords=[];this.quiescence=[];this.token=randomUUID();this.generation=0;this.startTime=undefined;this.stopped=false;
  }
  async initialize() {
    if(!this.fake){authorized(this.p,this.c);await sourcePair(this.p,this.swRoot,this.inferenceRoot);await verifyInputs(this.c,await allowedInputRoots(this.p,this.swRoot,this.inferenceRoot));}
    assert.ok(this.c.declaredSourceBindings?.length,'every API case requires declared source and legal authority inputs');
    for(const source of this.c.declaredSourceBindings){assertLicenseAuthority(source,{allowUnitFake:this.fake});const role=source.family==='ltx'?'adapter':'checkpoint',sha256=source.family==='ltx'?source.ltxAdapter.sha256:source.model.sha256;assert.ok(this.c.inputs.some(input=>input.role===role&&input.sha256===sha256),'explicit source authority must bind a verified source-role input before any import');for(const component of source.model.components)assert.ok(this.c.inputs.some(input=>input.role==='component'&&input.sha256===component.sha256),'inherited component authority must bind actual staged input bytes');for(const adapter of source.adapters??[])assert.ok(this.c.inputs.some(input=>input.role==='adapter'&&input.sha256===adapter.sha256),'adapter authority must bind actual staged adapter-role bytes');}
    this.sourceDtypeFacts=await verifyDenseSourceDtypes(this.c,this.p.output);
    const r=this.c.runtime;
    assert.ok(this.c.inputs.some(i=>i.path===r?.scenario && i.role==='scenario'),'bind the actual request/action scenario as an input');
    assert.ok(r && Number.isInteger(r.port) && r.port>1024 && r.port<=65535 && r.gpuId);
    if(!this.fake && this.c.backend==='cuda') {
      assert.equal(String(r.gpuId),MANUAL_CUDA_PHYSICAL_GPU,'manual campaign runs on physical GPU 1 (GPU 0 is the owner display GPU)');
      await hardware('cuda',r.deviceId); // Confirm UUID/PCI mapping before any build, listener or model load.
    }
    const services=path.join(this.p.output,'services');await mkdir(services,{recursive:true});
    this.runDir=path.join(services,this.c.id);await mkdir(this.runDir);this.owner=randomUUID();
    for(const d of ['config','data','data/cache','hf','hf/hub','cache']) await mkdir(path.join(this.runDir,d),{recursive:true});
    this.ownerPath=path.join(this.runDir,'owner.json');await save(this.ownerPath,{token:this.owner,pid:process.pid,sceneworks:this.p.sceneworks,inference:this.p.inference,proof_kind:this.fake?'unit_fake':'campaign_native'});
    this.seeded=new Set();this.seededInputs=new Map();
    for(const seed of r.seedFiles??[]) {
      const input=this.c.inputs.find(i=>i.path===seed.input);assert.ok(input,'seed must be an independently bound input');
      const canonicalInput=await (input.role==='source-config'?confined(this.swRoot,input.path):confinedTaskSource(this.p,input.path)); // No user/shared-cache seeds; only explicitly staged task bytes or source config.
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
      for(const proc of [this.worker,this.guardian,this.api].filter(Boolean)){let timer;try{exits.push({pid:proc.pid,...await Promise.race([proc.exit(),new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('owned startup cleanup timeout; no forced kill')),ownedExitBoundMs());})])});}catch(cleanup){exits.push({pid:proc.pid,error:cleanup.message});}finally{clearTimeout(timer);}}
      await save(path.join(this.dir,'startup-failure.json'),{error:error.message,ownedApiPid:this.api?.pid,ownedWorkerPid:this.worker?.pid,guardianPid:this.guardian?.pid,exits,meaning:'No scenario submitted; only owned graceful shutdown requested, no force kill'});
      throw error;
    }
  }
  async build() {
    const runner=this.deps.runProcess??runProcess;
    const command=['cargo','build','--locked',...campaignProfileArgs(),'--manifest-path',path.join(this.swRoot,'Cargo.toml'),'-p','sceneworks-rust-api','--bin','sceneworks-rust-api'];
    if(this.c.backend==='cuda')command.push('--features','sceneworks-rust-api/backend-candle');
    const platform=runPlatform(this.deps),buildRoot=path.join(this.p.output,`service-build-${this.p.sceneworks}-${this.p.inference}-${this.c.backend}`),receipt=path.join(buildRoot,'build.json');
    const target=cargoTarget(this.c,'target',path.join(buildRoot,'target'),platform);
    const configuration={command,flags:Object.fromEntries(['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','CARGO_BUILD_TARGET','RUSTUP_TOOLCHAIN'].map(k=>[k,process.env[k]??null])),proof_kind:this.fake?'unit_fake':'campaign_native'};
    let cached;
    try{await mkdir(buildRoot);}catch(error){if(error.code!=='EEXIST')throw error;cached=await json(receipt);assert.deepEqual(cached.configuration,configuration,'source/build configuration drift; preserve prior candidate');assert.equal(await hashFile(cached.binary),cached.sha256);}
    if(cached){this.buildIdentity=cached;this.binary=cached.binary;this.binaryHash=cached.sha256;await copyFile(path.join(buildRoot,'build.log'),path.join(this.dir,'service-build.log'));}
    else {
      // A failed build keeps its log in the case directory, which is what the evidence upload carries.
      try{await runner(command,{cwd:this.swRoot,env:{...process.env,...(this.c.backend==='cuda'?{RUSTC_WRAPPER:''}:{}),CARGO_TARGET_DIR:target,CARGO_PROFILE_DEV_DEBUG:'0'},logPath:path.join(buildRoot,'build.log'),timeoutSeconds:this.c.timeoutSeconds});}
      catch(error){await copyFile(path.join(buildRoot,'build.log'),path.join(this.dir,'service-build.log')).catch(()=>{});throw error;}
      this.binary=path.join(target,CAMPAIGN_PROFILE_DIR,`sceneworks-rust-api${platform==='win32'?'.exe':''}`);
      this.binaryHash=await hashFile(this.binary);this.buildIdentity={command,binary:this.binary,sha256:this.binaryHash,sceneworks:this.p.sceneworks,inference:this.p.inference,backend:this.c.backend,configuration};await save(receipt,this.buildIdentity);await copyFile(path.join(buildRoot,'build.log'),path.join(this.dir,'service-build.log'));
    }
    await save(path.join(this.dir,'service-build.json'),this.buildIdentity);
  }
  async assertOwner() {const record=await json(this.ownerPath);assert.equal(record.token,this.owner);assert.equal(record.pid,process.pid);assert.equal(await hashFile(this.binary),this.binaryHash);}
  async start() {
    await this.assertOwner();assert.ok(!this.api || this.stopped,'refuse second live service');
    await (this.deps.requireFreePort??requireFreePort)(this.c.runtime.port); // No HTTP probe of a pre-existing/user API.
    const launch=this.deps.spawnOwned??spawnOwned, epoch=++this.generation;
    const env=privateEnvironment(this.runDir,{...this.c.runtime,memoryCap:this.memoryCap??this.c.runtime.memoryCap,backend:this.c.backend},this.token);
    this.guardian=launch([process.execPath,'-e',"process.stdin.resume();process.stdin.on('end',()=>process.exit(0));"],env,this.swRoot,path.join(this.dir,`guardian-${epoch}.log`));
    assert.ok(this.guardian.pid>1);env.SCENEWORKS_PARENT_PID=String(this.guardian.pid);
    // The API's in-process utility worker registers under the API's SCENEWORKS_WORKER_ID unchanged.
    // Sharing the GPU worker's id let the GPU worker's registration replace it, so nothing advertised
    // model_import and every import stayed queued. The desktop likewise gives them distinct ids.
    this.api=launch([this.binary],{...env,SCENEWORKS_WORKER_ID:`${env.SCENEWORKS_WORKER_ID}-utility`},this.swRoot,path.join(this.dir,`api-${epoch}.log`));assert.ok(this.api.pid>1);
    this.startTime=new Date().toISOString();this.stopped=false;
    const deadline=Date.now()+60000;
    for(;;) {
      assert.equal(this.api.exitCode(),null,'new API exited before ownership/readiness');
      let pid;try{pid=await (this.deps.listenerPid??listenerPid)(this.c.runtime.port);}catch{}
      if(pid!==undefined){assert.equal(pid,this.api.pid,'refuse contacting a listener not owned by this child');break;}
      assert.ok(Date.now()<deadline,'owned API listener timeout');await (this.deps.sleep??sleep)(100);
    }
    this.baseUrl=`http://127.0.0.1:${this.c.runtime.port}`;
    // The API binds a bootstrap health surface first and reports `starting` until its
    // readiness-critical startup completes, then the ready app answers `ok`. Wait for readiness
    // within the same deadline; only `starting` is an acceptable in-between state, and the owned
    // API exiting is still an immediate failure.
    let health;
    for(;;){
      assert.equal(this.api.exitCode(),null,'new API exited before readiness');
      health=await this.request('GET','/api/v1/health');assert.equal(health.service,'sceneworks-api');
      if(health.status==='ok')break;
      assert.equal(health.status,'starting','API health must be starting or ok');
      assert.ok(Date.now()<deadline,'owned API readiness timeout');await (this.deps.sleep??sleep)(100);
    }
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
    if(this.stopped)return;await this.assertOwner();
    // No signal until every owned submitted job is terminal. Never signal arbitrary PIDs. A dead
    // owned API cannot drain or cancel anything: its jobs end with it, so the owned worker and
    // guardian still get their graceful shutdown request. Without it they outlived the case and
    // held the lane open until the job timeout.
    let drainError;try{await this.drain();}catch(error){if(this.api.exitCode()===null)throw error;drainError=error;}
    if((this.deps.platform??process.platform)==='win32')this.worker.stdin.end();else this.worker.signal('SIGTERM');
    const bounded=async proc=>{let timer;try{return await Promise.race([proc.exit(),new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('owned graceful exit timeout; no forced kill')),ownedExitBoundMs());})]);}finally{clearTimeout(timer);}};
    const workerExit=await bounded(this.worker);
    this.guardian.stdin.end();await bounded(this.guardian); // Existing API parent-death watchdog -> graceful drain, on BOTH OSes.
    const apiExit=await bounded(this.api);this.stopped=true;
    await save(path.join(this.dir,`exits-${this.generation}.json`),{workerExit,apiExit,ended:new Date().toISOString(),...(drainError?{drainError:drainError.message}:{})});
    if(drainError)throw drainError;
    assert.equal(workerExit.code,0,'owned worker did not exit gracefully');assert.equal(apiExit.code,0,'owned API did not exit gracefully');
  }
  async checkStep(step,captures) {
    if(step.waitJob)return;
    const route=expand(step.path,captures),body=expand(step.body,captures);
    if(step.method==='POST'&&route==='/api/v1/jobs'&&['model_import','lora_import','model_download','lora_download'].includes(body?.type))throw new Error('generic job submission cannot bypass source-bound import/license authority checks');
    if(step.method==='POST'&&/^\/api\/v1\/models\/[^/]+\/(?:download|license)$/.test(route)){
      const modelId=decodeURIComponent(route.split('/')[4]),source=this.c.declaredSourceBindings.find(v=>v.model.modelId===modelId||v.model.baseModel===modelId);assert.ok(source,'component acquisition/acknowledgment requires declared source authority');assertLicenseAuthority(source,{allowUnitFake:this.fake});if(body?.componentId)assert.ok(source.model.components.some(component=>component.id===body.componentId),'undeclared inherited component acquisition');
    }
    if(step.method==='POST'&&route==='/api/v1/loras/import'){
      const h3=this.c.declaredSourceBindings.find(v=>v.family!=='ltx'&&v.model.modelId===body.baseModel&&v.adapters?.some(a=>a.id===body.loraId&&a.importPath));if(h3){assertLicenseAuthority(h3,{allowUnitFake:this.fake});const a=h3.adapters.find(a=>a.id===body.loraId);assert.equal(body.sourcePath,a.importPath);assert.equal(body.expectedSha256,a.sha256);assert.equal(body.family,'minimax-h3');assert.equal(body.licenseAcknowledged,h3.licenseAuthority.acknowledged);assert.ok(!body.repo&&!body.sourceUrl);await confinedTaskSource(this.p,body.sourcePath);return;}

      const selected=this.c.declaredSourceBindings.find(v=>v.family==='ltx'&&v.ltxAdapter.id===body.loraId);assert.ok(selected,'undeclared LTX adapter import');assertLicenseAuthority(selected,{allowUnitFake:this.fake});
      assert.equal(body.licenseAcknowledged,selected.licenseAuthority.acknowledged);assert.equal(body.expectedSha256,selected.ltxAdapter.sha256);assert.equal(body.baseModel,selected.model.modelId);assert.ok(!body.repo&&!body.sourceUrl,'LTX baseline uses its actual staged adapter, not an implicit download');await confinedTaskSource(this.p,body.sourcePath);
    }
    if(step.method==='POST'&&route==='/api/v1/models/import') {
      assert.ok(body.ownershipMode===undefined||body.ownershipMode==='managed','linked ownership must be omitted; the only ownership enum is managed');
      const selected=this.c.declaredSourceBindings.find(v=>v.model.modelId===body.modelId);assert.ok(selected,'undeclared imported source');assertLicenseAuthority(selected,{allowUnitFake:this.fake});assert.equal(body.licenseAcknowledged,selected.licenseAuthority.acknowledged);
      if(body.linkedRootId||body.linkedRelativePath)assert.ok(body.ownershipMode===undefined&&!body.sourcePath&&!body.source,'linked root/path requires omitted ownershipMode and no managed locator');
      assert.ok(!body.repo && !body.sourceUrl && body.source?.kind!=='url','use an explicit staged local/linked source or exact cached Hugging Face source, never an implicit URL download');
      if(body.source?.kind==='huggingFace') {
        const {repo,revision,files}=body.source;assert.match(repo??'',/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/);assert.match(revision??'',/^[a-f0-9]{40}$/);assert.ok(Array.isArray(files)&&files.length&&files.every(f=>typeof f==='string'),'exact staged HF file inventory required');
        const snapshot=inside(this.runDir,`hf/hub/models--${repo.replaceAll('/','--')}/snapshots/${revision}`);
        for(const file of files){const target=await confined(snapshot,inside(snapshot,file)),seed=this.seededInputs.get(target);assert.ok(seed,'HF import must reuse an actually seeded private snapshot, not download missing weights');if(files.length===1&&body.expectedSha256)assert.equal(seed.sha256,body.expectedSha256);}
        return; // The EXISTING importer/registry still owns provenance, licensing, interpretation and ingestion.
      }
      const source=body.sourcePath??body.source?.path??body.source?.stagedPath;if(source)await confinedTaskSource(this.p,source);
      else {
        assert.ok(body.ownershipMode===undefined&&body.linkedRootId&&body.linkedRelativePath,'actual linked DTO requires root/path with ownership omitted');
        const roots=await this.request('GET','/api/v1/models/library-roots'),root=roots.roots.find(r=>r.rootId===body.linkedRootId);assert.ok(root,'owned approved linked root required');
        await confinedTaskSource(this.p,root.path,{allowRoot:true});await confined(root.path,path.resolve(root.path,body.linkedRelativePath));
      }
    }
    if(route.startsWith('/api/v1/models/library-roots')&&body?.path)await confinedTaskSource(this.p,body.path,{allowRoot:true});
  }
  // Retained JSON events from the live owned worker/API buffers plus every prior epoch log.
  async preparationTrace() {
    const chunks=[];
    for(const proc of [this.api,this.worker]) if(typeof proc?.log==='function') chunks.push(proc.log());
    for(const epoch of this.epochs.slice(0,-1)) for(const kind of ['api','worker']) {
      try{ chunks.push(await readFile(path.join(this.dir,`${kind}-${epoch.epoch}.log`),'utf8')); }
      catch(error){ if(error.code!=='ENOENT') throw error; }
    }
    const events=[];
    for(const text of chunks) for(const line of text.split('\n')) { try{ const value=JSON.parse(line); if(traceFields(value).event) events.push(value); }catch{} }
    return events;
  }
  // Wait for an ACTUAL partial cold checkpoint on this job's preparation span; never a memo hit,
  // a zero-byte label or the full source. A complete/fast read is a loud failure, not a pass.
  async waitColdPartialReading(jobId,deadline) {
    for(;;) {
      const events=await this.preparationTrace();
      const rows=events.filter(event=>{
        const progress=coldSourceProgress(event), fields=progress?.fields;
        if(!progress||String(tracedJobId(event))!==String(jobId))return false;
        const hashed=Number(fields.hashed_bytes),source=observedSourceBytes(fields.source_bytes);
        return fields.acquisition==='ColdContentRead'&&fields.state==='Reading'&&Number.isSafeInteger(hashed)&&hashed>0&&Number.isSafeInteger(source)&&hashed<source;
      });
      if(rows.length)return rows.at(-1);
      const job=await this.request('GET',`/api/v1/jobs/${encodeURIComponent(jobId)}`);
      assert.ok(!terminal(job.status),'cold source preparation finished before a partial cold checkpoint was observed; no cancellation proof exists');
      assert.ok(Date.now()<deadline,'cold source preparation never emitted an actual partial ColdContentRead checkpoint');
      await (this.deps.sleep??sleep)(50);
    }
  }
  // The root marker is posted only AFTER CancelJoinGuard::cancel_and_join awaited the detached task.
  // Acknowledgement must arrive UNDER the 30s grace; a 30s+ wait is a hard-abandon, not proof.
  async waitPreparationCanceled(jobId,requestedAt) {
    for(;;) {
      const job=await this.request('GET',`/api/v1/jobs/${encodeURIComponent(jobId)}`);
      if(job.status==='canceled') {
        assert.equal(typeof job.message,'string','canceled preparation must carry a root message');
        assert.ok(job.message.includes(PREP_CANCELED_MARKER),'canceled preparation lacks the joined-teardown root marker');
        const elapsedMs=Date.now()-requestedAt;
        assert.ok(elapsedMs>=0&&elapsedMs<PREP_CANCEL_ACK_MS,`cooperative cancellation was not acknowledged under the ${PREP_CANCEL_ACK_MS}ms joined grace; the guard would have hard-abandoned the detached task`);
        return {job,observedAt:new Date(requestedAt+elapsedMs).toISOString(),acknowledgedAt:new Date(requestedAt+elapsedMs).toISOString(),elapsedMs,hardAbandoned:false};
      }
      assert.ok(!['completed','failed','interrupted','cancelled'].includes(job.status),`job reached non-canceled terminal state ${job.status}; source-preparation cancellation was not observed`);
      assert.ok(Date.now()-requestedAt<PREP_CANCEL_ACK_MS,'cooperative cancellation was not acknowledged under the joined grace; hard-abandon evidence is not lease-release proof');
      await (this.deps.sleep??sleep)(25);
    }
  }
  // Managed install -> fresh owned Idle restart (cold memo) -> generation -> observed partial cold
  // checkpoint -> cooperative cancel -> joined-teardown marker -> actual trace/request/job binding ->
  // fresh Idle -> owned model removal as lease-release witness. Originals are never touched.
  async sourceCancel(action,captures) {
    await this.assertOwner();
    assert.ok(action.generationCapture&&action.modelCapture,'source cancellation must bind the actual submitted generation and its owned imported model');
    const jobId=action.generationCapture.split('/').reduce((v,k)=>v[k],captures);
    const modelId=action.modelCapture.split('/').reduce((v,k)=>v[k],captures);
    assert.ok(typeof jobId==='string'&&typeof modelId==='string','actual generation/model captures required');
    assert.ok(this.jobs.has(jobId),'source cancellation target must be this owned submitted generation');
    const receipt={actionKind:'source-cancel',kind:'generation-source-prep-cancel-receipt',version:1,proof_kind:this.fake?'unit_fake':'campaign_native',caseId:this.c.id,backend:this.c.backend,jobId,modelId,chunkBytes:SOURCE_PREP_CHUNK_BYTES,spanPhase:PREP_SPAN_PHASE};
    const receiptPath=path.join(this.dir,receiptNameFor(jobId));
    try {
      const job=await this.request('GET',`/api/v1/jobs/${encodeURIComponent(jobId)}`);
      assert.equal(job.payload?.model,modelId,'generation must reference the actually installed imported model');
      receipt.generationJob=job; // Actual queued canonical snapshot; this is planned provenance, NOT a completed native seal.
      const planned=job.modelProvenance?.plan??job.payload?.modelProvenance?.plan??job.payload?.expectedImportedModelRef;
      assert.ok(planned?.checkpointId&&planned.semanticDigest,'canceled preparation must bind the actual queued planned import commitment');
      receipt.plannedImportRef=structuredClone(planned);
      const deadline=Math.min(Date.now()+this.c.timeoutSeconds*1000,this.fake?Infinity:Number(process.env.H3_CASE_DEADLINE_MS??Infinity));
      const reading=await this.waitColdPartialReading(jobId,deadline);
      const fields=traceFields(reading),sourceBytes=observedSourceBytes(fields.source_bytes),partialBytes=Number(fields.hashed_bytes);
      assert.equal(fields.acquisition,'ColdContentRead');assert.equal(fields.state,'Reading');
      assert.ok(Number.isSafeInteger(sourceBytes)&&sourceBytes>0,'actual source_bytes required');
      assert.ok(Number.isSafeInteger(partialBytes)&&partialBytes>0&&partialBytes<sourceBytes,'a partial cold checkpoint between zero and the full source is required');
      const sourcePath=await confined(this.p.output,fields.path); // Canonical confinement BEFORE stat/hash, including symlink escape refusal.
      assert.equal(traceSpan(reading).checkpointId,planned.checkpointId,'cold source trace differs from the actual planned checkpoint commitment');
      assert.equal((await stat(sourcePath)).size,sourceBytes,'source_bytes differs from the actual owned source size');
      receipt.observation=reading;
      const requestedAt=Date.now();
      const response=await this.request('POST',`/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`);
      receipt.cancelRequest={method:'POST',route:`/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`,requestedAt:new Date(requestedAt).toISOString(),response};
      receipt.terminal=await this.waitPreparationCanceled(jobId,requestedAt);
      // Audit only AFTER cancellation: a full independent read before POST /cancel would
      // miss the actual partial-hash window while falsely labeling the audit post-cancel.
      receipt.sourceAudit={path:fields.path,canonicalPath:sourcePath,ownedRoot:await realpath(this.p.output),sizeBytes:sourceBytes,sha256:await hashFile(sourcePath),auditedAt:new Date().toISOString(),meaning:'Independent post-cancel Node content audit; NOT a native acquired seal/memo or model load'};
      assert.ok(this.c.inputs.some(input=>input.role==='checkpoint'&&input.sha256===receipt.sourceAudit.sha256),'observed private source must match an actual verified checkpoint-role input');
      const events=await this.preparationTrace(),retained=event=>events.some(value=>JSON.stringify(value)===JSON.stringify(event));
      assert.ok(retained(reading),'observed cold partial reading is absent from the retained worker/API trace');
      const canceled=events.find(event=>{const progress=coldSourceProgress(event),value=progress?.fields;return progress&&String(tracedJobId(event))===String(jobId)&&value.path===fields.path&&value.acquisition==='ColdContentRead'&&value.state==='Canceled';});
      assert.ok(canceled,'retained trace lacks the actual Canceled source progress event');
      const canceledFields=traceFields(canceled),canceledBytes=Number(canceledFields.hashed_bytes);
      assert.ok(Number.isSafeInteger(canceledBytes)&&canceledBytes>=partialBytes&&canceledBytes<sourceBytes,'Canceled must report the exact partial bytes actually reached, below the full source');
      receipt.canceled=canceled;
      assert.ok(!events.some(event=>{const progress=coldSourceProgress(event),value=progress?.fields;return progress&&String(tracedJobId(event))===String(jobId)&&value.path===fields.path&&value.state==='Acquired';}),'the canceled source path was fully Acquired; no cold-preparation cancellation occurred');
      // Cold source HASH cancellation happens BEFORE codec copying (confirmed against aa3):
      // a valid hash-only cancel has NO copy event. A Writing/copy event means the hashing
      // window was missed and cancellation did not actually occur during source hashing.
      assert.ok(!events.some(event=>traceFields(event).event==='h3_normalization_copy_progress'&&String(tracedJobId(event))===String(jobId)),'a normalization copy event shows the hashing window was missed; the source was already copied when canceled');
      assert.ok(!events.some(event=>traceFields(event).event==='h3_import_source_prepared'&&String(tracedJobId(event))===String(jobId)),'canceled preparation published h3_import_source_prepared; the source was prepared, not canceled');
      assert.ok(!events.some(event=>traceFields(event).event==='h3_import_prepared_load'&&String(tracedJobId(event))===String(jobId)),'canceled preparation reached a native load closure');
      const idle=await this.waitIdle();
      assert.ok(idle.status!=='exited'&&idle.worker?.status==='idle'&&!idle.worker.currentJobId,'fresh owned worker must acknowledge Idle after joined teardown');
      receipt.teardown={guardGraceMs:PREP_CANCEL_ACK_MS,guardAbandonMs:PREP_CANCEL_ABANDON_MS,idle};
      const removalAt=new Date().toISOString();
      const removal=await this.request('DELETE',`/api/v1/models/${encodeURIComponent(modelId)}?permanent=true`);
      assert.equal(removal.id,modelId,'removal must target the actually installed imported model');
      assert.equal(removal.removedManifestEntry,true,'owned model removal refused; the source lease was not released');
      receipt.removal={method:'DELETE',route:`/api/v1/models/${encodeURIComponent(modelId)}?permanent=true`,at:removalAt,response:removal};
      receipt.sourceRole={modelId,checkpointSha256:receipt.sourceAudit.sha256};
      return receipt;
    } catch(error) { receipt.error=error.message; throw error; }
    finally { await writeFile(receiptPath,JSON.stringify(receipt,null,2)+'\n'); }
  }
  async action(action,captures) {
    await this.assertOwner();
    if(action.kind==='source-cancel') return this.sourceCancel(action,captures);
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
      if(action.memoryCap)this.memoryCap=structuredClone(action.memoryCap);
      await this.start();return {kind:action.kind,epochs:this.epochs,...(action.memoryCap?{memoryCap:structuredClone(action.memoryCap)}:{})};
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
    const summarize=samples=>{
    const last=samples.at(-1),pid=samples[0].pid, deviceId=last.deviceId, cumulative=last.method==='mlx-allocator-cumulative-high-water';
    assert.ok(samples.every(s=>s.pid===pid && s.method===last.method && s.deviceId===deviceId),'do not misattribute different worker lifetimes/devices to a single peak window');
    const make=(device,phase)=>({method:device?last.method:'sampled-process-rss',sampled:device&&cumulative?false:true,intervalMs:observedInterval(samples,device,this.intervalMs),targetPauseMs:this.intervalMs,samples:samples.filter(s=>Number.isFinite(device?s.device:s.host)).map(s=>({at:s.at,pid:s.pid,probeStartedAt:s.probeStartedAt,bytes:device?s.device:s.host,...(device&&s.deviceCounterAt?{counterWrittenAt:s.deviceCounterAt}:{})})),blindspots:[...this.gaps.map(g=>`${g.at}: ${g.error}`),'intervalMs is the maximum observed valid-sample gap including probe latency; targetPauseMs is only the configured delay. Host/device reads occur within the retained probeStartedAt..at window. Process sampling may miss transients; process-lifetime allocator counters include prior allocations and allocator-only attribution.'],window:{start:this.started,end:this.ended,phase,coverage:phase==='loading'?'full-case-load-inclusive':'full-case'},bounds:[...(phase==='loading'?['full-case-load-inclusive-upper-bound']:[]),...(device&&!cumulative?['observed-device-global-bound']:[])],attribution:{scope:device?(cumulative?'process-lifetime':'device'):'process',pid,deviceId:device?deviceId:undefined,includesPriorAllocations:device&&cumulative,includesOtherProcesses:device&&!cumulative,meaning:device?(cumulative?'Existing owned worker gpu_telemetry.json peakBytes, fresh after completion; process-lifetime MLX allocator highwater, not phase-local or residency delta':'Observed device-global used-memory samples; includes other processes, not model-attributable allocation'):'RSS of the owned single worker process, API/other processes excluded'}});
    const hostPeakBytes=Math.max(...samples.map(s=>s.host)),devicePeakBytes=Math.max(...samples.filter(s=>Number.isFinite(s.device)).map(s=>s.device));assert.ok(hostPeakBytes>0&&devicePeakBytes>0);
    return {hostPeakBytes,devicePeakBytes,loadHostPeakBytes:hostPeakBytes,loadDevicePeakBytes:devicePeakBytes,observations:{host:make(false,'case'),device:make(true,'case'),loadHost:make(false,'loading'),loadDevice:make(true,'loading')}};
    };
    const groups=[...new Set(this.samples.map(sample=>sample.pid))].map(pid=>{assert.ok(this.service.epochs.some(epoch=>epoch.workerPid===pid),"observations must belong to actual owned worker epochs");return {pid,metrics:summarize(this.samples.filter(sample=>sample.pid===pid))};});
    if(groups.length===1)return groups[0].metrics;
    const aggregate={kind:"owned-worker-epochs",epochs:groups,meaning:"Maximum observed upper bound across non-overlapping owned worker epochs; each process lifetime/counter remains separately attributed, never summed or relabeled as one PID."};
    for(const key of ["hostPeakBytes","devicePeakBytes","loadHostPeakBytes","loadDevicePeakBytes"])aggregate[key]=Math.max(...groups.map(group=>group.metrics[key]));
    return aggregate;
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
  assert.ok(assembly.componentClosure?.length,'actual immutable inherited component closure is required');
  for(const component of assembly.componentClosure){assert.match(component.revision??'',/^[a-f0-9]{40}$/);assert.ok(component.files?.length,'component content closure cannot be empty');for(const file of component.files){assert.match(file.sha256??'',/^sha256:[a-f0-9]{64}$/,'schema2 component digest must be an actual typed SHA256 commitment');assert.ok(inputs.some(i=>i.role==='component'&&i.sha256===file.sha256.slice(7)),'every inherited component file/profile must match verified staged bytes');}}
  const pins=role=>closure.prepared_sources.filter(s=>inputs.some(i=>i.role===role&&i.sha256===s.sha256));
  return {checkpointSha256:checkpoint.sha256,planId:closure.identity.planId,semanticDigest:closure.identity.semanticDigest,topology:assembly.adalnPruning??{kind:'no-pruning-binding-in-observed-assembly'},quantization:{sourceEncoding:assembly.sourceEncoding??null,observedAssembly:assembly},components:pins('component'),adapters:pins('adapter'),operation:closure.operation,samplingPolicy:{steps:closure.steps,schedulerShift:closure.schedulerShift},seed:closure.seed};
}
export async function verifyDenseSourceDtypes(c,outputRoot){
  const format=c.id.split('.').at(-1),expected={'dense-f16':'F16','dense-bf16':'BF16','dense-f32':'F32'}[format];if(!expected)return undefined;
  const facts=[];let matrices=0;
  for(const source of c.declaredSourceBindings){const files=source.model.sourceKind==='directory'?source.model.files.filter(file=>file.path.toLowerCase().endsWith('.safetensors')):[{path:source.model.sourcePath,sha256:source.model.sha256}];
    for(const file of files){assert.ok(c.inputs.some(input=>input.role==='checkpoint'&&input.path===file.path&&input.sha256===file.sha256),'dense format must bind the actual selected source file');const canonical=await confined(outputRoot,file.path),handle=await open(canonical,'r');
      try{const read=async(buffer,position)=>{let done=0;while(done<buffer.length){const result=await handle.read(buffer,done,buffer.length-done,position+done);assert.ok(result.bytesRead>0,'truncated source metadata');done+=result.bytesRead;}};const prefix=Buffer.alloc(8);await read(prefix,0);const length=prefix.readBigUInt64LE();assert.ok(length>0n&&length<=64n*1024n*1024n&&length+8n<=BigInt((await handle.stat()).size),'bounded actual safetensors metadata required');const bytes=Buffer.alloc(Number(length));await read(bytes,8);const header=JSON.parse(bytes.toString('utf8'));
        const histogram={};for(const [name,tensor]of Object.entries(header)){if(name==='__metadata__')continue;histogram[tensor.dtype]=(histogram[tensor.dtype]??0)+1;if(tensor.shape?.length>=2&&/\.(?:attn\.(?:to_[qkv]|to_out\.0|qkv_proj|out_proj)|mlp\.(?:fc1|fc2)|ff\.net\.(?:0\.proj|2))\.weight$/.test(name)){assert.equal(tensor.dtype,expected,'actual source body precision differs from its format label');matrices++;}assert.ok(!name.includes('comfy_quant'),'dense source case cannot substitute a quantized dialect');}facts.push({path:file.path,sha256:file.sha256,histogram});
      }finally{await handle.close();}
    }
  }assert.ok(matrices>0,'no actual H3 body matrices in declared dense source');return{expectedBodyDtype:expected,bodyMatrices:matrices,files:facts,meaning:'actual bounded source header facts, not a scenario precision label'};
}
export function resolvedMediaContract(c,job){
  const selected=c.declaredSourceBindings?.find(v=>v.model.modelId===job.payload.model);assert.ok(selected,'actual media requires its declared source/base contract');
  const model=selected.family==='ltx'?selected.model.modelId:selected.model.baseModel;
  const audio=selected.family==='ltx'?{required:true,sampleRate:selected.model.outputAudioSampleRate,channels:2,codec:'aac'}:{required:true,sampleRate:32000,channels:2,codec:'aac'};
  if(selected.family==='ltx')assert.ok([24000,48000].includes(audio.sampleRate),'LTX core/BWE output sample rate must be source-resolved, not guessed');
  return {model,fps:job.payload.fps,width:job.payload.width,height:job.payload.height,audio};
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
    validateAVAgainstRequest(decoded,job.payload,resolvedMediaContract(service.c,job));
    const video=decoded.streams?.find(s=>s.codec_type==='video');assert.ok(video,'actual decoded video metadata required');
    for(const axis of ['width','height'])if(job.payload?.[axis]!==undefined)assert.equal(video[axis],job.payload[axis],'decoded canvas differs from the resolved queued geometry');
    bound.push({jobId:job.id,assetId:id,projectId:asset.projectId,source,av,sha256:await hashFile(dest),decoded,asset});
  }
  assert.ok(bound.length,'no actual synchronized video asset in the completed job');return bound;
}
export function semanticAssembly(value){
  if(Array.isArray(value))return value.map(semanticAssembly);
  if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).filter(([key,v])=>!['checkpointId','planId','rootId','linkedRootId','ownershipMode'].includes(key)&&!(typeof v==='string'&&(path.isAbsolute(v)||path.win32.isAbsolute(v))&&/path$/i.test(key))).map(([key,v])=>[key,semanticAssembly(v)]));
  return value;
}
export async function verifyOwnershipEquivalence(bundle,scenario,byJob,dir,decode=decodeAV,delta){
  const spec=scenario.ownershipEquivalence;assert.ok(spec,'actual controlled ownership comparison is required');
  const find=(label,index)=>{const captured=bundle.captures?.[`${label}_av${index}_done`],binding=byJob.get(captured?.id);assert.ok(binding,'ownership comparison lacks actual repeated job/native/output binding');assert.deepEqual(captured,binding.job);return binding;};
  const managed=[find(spec.managed,0),find(spec.managed,1)],linked=[find(spec.linked,0),find(spec.linked,1)];
  const base=managed[0];
  for(const binding of [...managed,...linked]){
    assert.equal(binding.runtimeIdentity.checkpointSha256,base.runtimeIdentity.checkpointSha256);
    assert.deepEqual(semanticAssembly(binding.runtimeIdentity.quantization.observedAssembly),semanticAssembly(base.runtimeIdentity.quantization.observedAssembly),'ownership changed resolved semantic component/conditioning assembly');
    const actual=creativePayload(submittedPayload(bundle.requests??[],binding.job)),expected=creativePayload(submittedPayload(bundle.requests??[],base.job));delete actual.model;delete expected.model;assert.deepEqual(actual,expected,'ownership comparison changed creative controls');
  }
  const results=[];
  for(const stream of ['video','audio']){
    const maximum=spec.thresholds?.[stream==='video'?'maximumVideoDelta':'maximumAudioDelta'],noiseLimit=spec.thresholds?.maximumBaselineDelta;
    assert.ok([maximum,noiseLimit].every(v=>Number.isFinite(v)&&v>=0),'predeclared ownership/noise bounds required');
    const difference=async(a,b)=>(delta??decodedDelta)(inside(dir,a.media.av),inside(dir,b.media.av),stream);
    for(const pair of [managed,linked])assert.ok(await difference(...pair)<=noiseLimit,'ownership repeated-baseline noise exceeds predeclared bound');
    const differenceAcrossOwnership=await difference(managed[0],linked[0]);assert.ok(differenceAcrossOwnership<=maximum,'managed/linked decoded output is not equivalent');results.push({stream,difference:differenceAcrossOwnership,maximum});
  }
  return results;
}
export async function verifyIdentificationCoverage(c,dir,bundle,scenario,byJob,{allowUnitFake=false}={}){
  const spec=scenario.identificationControl;assert.ok(spec,'content identification requires actual renamed and different-bytes fixtures');
  const input=c.inputs.find(value=>value.role==='identity-control-fixture'&&value.path===spec.receipt);assert.ok(input,'actual staged control receipt must be independently input-bound');
  const copied=inside(dir,'identification-input.json');assert.equal(await hashFile(copied),input.sha256);const control=await json(copied);
  assert.equal(control.kind,'h3-finetune-stage-identity-controls');assert.equal(control.status,'prepared-not-modelproof');assert.ok(control.proof_kind==='campaign_stage_derived'||allowUnitFake&&control.proof_kind==='unit_fake','synthetic identity fixture is not real staged content');
  assert.equal(control.sourcePath,spec.originalPath);assert.equal(control.sourceExpectedSHA,spec.expectedSha256);assert.equal(control.sourceActualSHA,spec.expectedSha256);assert.equal(control.renamed.actualSHA,spec.expectedSha256);assert.notEqual(control.renamed.path,control.sourcePath);assert.notEqual(control.differentBytes.actualSHA,spec.expectedSha256);assert.equal(control.differentBytes.expectedOriginalSHA,spec.expectedSha256);
  for(const [file,sha256]of [[control.sourcePath,spec.expectedSha256],[control.renamed.path,spec.expectedSha256],[control.differentBytes.path,control.differentBytes.actualSHA]])assert.ok(c.inputs.some(value=>value.role==='checkpoint'&&value.path===file&&value.sha256===sha256),'all original/copied/different source bytes must be actual verified input roles');
  const positive=bundle.captures?.[`${spec.positive}_av0_done`],binding=byJob.get(positive?.id);assert.ok(binding,'renamed source must actually generate with a native closure/output');assert.equal(binding.runtimeIdentity.checkpointSha256,spec.expectedSha256);
  const selected=c.declaredSourceBindings.find(value=>value.model.modelId===positive.payload.model);assert.equal(selected?.model.sourcePath,control.renamed.path,'the actual renamed copy must be the selected import source');assert.ok(bundle.requests.some(record=>record.route==='/api/v1/models/import'&&record.request?.modelId===selected.model.modelId&&(selected.ownershipMode==='linked'?record.request.linkedRootId===expand(selected.linkedRootId,bundle.captures??{})&&record.request.linkedRelativePath===selected.linkedRelativePath:record.request.sourcePath===control.renamed.path)&&record.request.expectedSha256===spec.expectedSha256),'renamed source must be actually submitted, not substituted by an original/builtin');
  const negative=bundle.captures?.[`${spec.negative}_refused`];assert.equal(negative?.status,'failed','different bytes must not be accepted as the original source');assert.equal(negative.payload.expectedSha256,spec.expectedSha256);assert.ok(negative.error?.includes('integrity check')&&negative.error.includes(control.differentBytes.actualSHA)&&negative.error.includes(spec.expectedSha256),'actual API checksum refusal must identify both observed and expected content');
  const request=bundle.requests.find(record=>record.route==='/api/v1/models/import'&&record.result?.id===negative.id);assert.ok(request&&request.request.sourcePath===control.differentBytes.path&&request.request.expectedSha256===spec.expectedSha256,'negative refusal must bind the actually submitted different-byte fixture');
  return {sameContentRecognized:{original:control.sourcePath,relocated:control.renamed.path,sha256:spec.expectedSha256,nativeJobId:positive.id},differentContentRefused:{jobId:negative.id,observedSha256:control.differentBytes.actualSHA,expectedSha256:spec.expectedSha256},meaning:'content identity independent of name/path/repository layout; no new training claimed'};
}
export const tracedJobId=event=>(event.fields??event).jobId??event.span?.jobId??event.spans?.find(span=>span.jobId)?.jobId;
export function verifyCacheCoverage(bundle,scenario,byJob,observed){
  const spec=scenario.cacheCoverage;assert.ok(spec,'cache coverage requires source-bound execution contract');
  const find=name=>{const job=bundle.captures?.[name],binding=byJob.get(job?.id);assert.ok(binding,'cache phase lacks actual completed job/source/output');assert.deepEqual(binding.job,job);return binding;};
  const [cold,warm,staged]=[find(spec.cold),find(spec.warm),find(spec.staged)];
  assert.equal(normalizeLoadEvent(cold.event).cacheState,'Cold','cold must actually miss generator metadata cache');assert.equal(normalizeLoadEvent(warm.event).cacheState,'Warm','warm must actually hit generator metadata cache');
  for(const binding of [cold,warm,staged]){assert.equal((binding.event.fields??binding.event).cache_layer,'generator_metadata','actual cache layer must be generator metadata');for(const key of ['loaded_spec_offload_policy','loaded_spec_load_shape'])assert.ok(typeof(binding.event.fields??binding.event)[key]==='string'&&(binding.event.fields??binding.event)[key].length,'observed loaded-spec policy/shape required');assert.equal((binding.event.fields??binding.event).retains_dit_weights,false,'metadata warmth must not masquerade as retained DiT weights');assert.equal(binding.runtimeIdentity.checkpointSha256,cold.runtimeIdentity.checkpointSha256);assert.equal(binding.runtimeIdentity.semanticDigest,cold.runtimeIdentity.semanticDigest);assert.deepEqual(creativePayload(submittedPayload(bundle.requests??[],binding.job)),creativePayload(submittedPayload(bundle.requests??[],cold.job)),'cache/staging changed creative request');}
  const policy=binding=>{const record=observed.find(event=>(event.fields??event).event==='generation_memory_policy_applied'&&tracedJobId(event)===binding.job.id);assert.ok(record,'actual configured per-job memory policy event required');const fields=record.fields??record;assert.equal(fields.scope_configured,true);assert.ok(fields.strategy);assert.equal(typeof fields.stage_residency,'boolean','actual applied stage-residency policy must be observed, not a scenario label');return fields;};
  const before=policy(warm),after=policy(staged);policy(cold);
  assert.equal(after.stream_transformer_blocks,true,'staged run must actually configure bounded transformer residency');assert.ok(Number.isInteger(after.window_size)&&after.window_size>0&&['Dit','TextEncoder','Both'].includes(after.window_component),'actual configured window/component required');
  assert.ok(before.stream_transformer_blocks!==true||!Number.isFinite(before.window_size)||after.window_size<before.window_size,'unchanged sequential/resident policy is not a staged transition');
  assert.ok(bundle.actions.some(action=>action.kind==='restart'&&action.memoryCap),'constrained staging must be bound to actual owned-worker restart/cap input');
  return {cacheLayer:'generator-metadata-not-DiT-residency',coldJobId:cold.job.id,warmJobId:warm.job.id,stagedJobId:staged.job.id,policyBefore:before,policyAfter:after};
}
// Re-verify a cold source-preparation cancellation from retained bytes only. The receipt is a
// correlation index, never the proof: every event must reappear verbatim in the retained
// worker/API epoch logs, and the terminal snapshot must reappear in the actual API transcript.
export async function verifyGenerationSourcePrepCancel(p,c,dir,{allowUnitFake=false,receiptName,contentReadback=false}={}) {
  assert.ok(typeof receiptName==='string'&&receiptName.length>0,'actual retained source-cancel receipt is required; labels or invented receipts are not proof');
  const receipt=await json(inside(dir,receiptName));
  assert.equal(receipt.kind,'generation-source-prep-cancel-receipt');assert.equal(receipt.version,1);
  assert.ok(receipt.proof_kind==='campaign_native'||(allowUnitFake&&receipt.proof_kind==='unit_fake'),'unit_fake receipt is never campaign evidence');
  assert.equal(receipt.caseId,c.id);assert.equal(receipt.backend,c.backend);
  assert.ok(c.declaredSourceBindings?.length,'planned source/legal authority cannot be absent during readback');
  const jobId=receipt.jobId,modelId=receipt.modelId;assert.ok(jobId&&modelId,'actual generation/model identity required');assert.equal(receipt.generationJob?.id,jobId,'planned snapshot must be the actual canceled generation job');
  const planned=c.declaredSourceBindings.filter(source=>source.model?.modelId===modelId);assert.ok(planned.length,'canceled generation model is not a declared planned source');
  const declaredShas=new Set(planned.flatMap(source=>[source.model.sha256,...(source.model.files??[]).map(file=>file.sha256)]));
  assert.equal(receipt.spanPhase,PREP_SPAN_PHASE);
  assert.equal(receipt.chunkBytes,SOURCE_PREP_CHUNK_BYTES);
  const observation=receipt.observation;assert.ok(observation,'actual partial ColdContentRead observation required');
  const observed=traceFields(observation);assert.equal(observed.event,'artifact_seal_hash_progress','actual native artifact observer required; a core checksum callback is not the native pin');assert.equal(tracedJobId(observation),jobId,'cold progress must belong to the canceled generation');
  assert.equal(observed.acquisition,'ColdContentRead','only a cold content read is admissible; memo/other acquisitions are not cancellation evidence');
  assert.equal(observed.state,'Reading');
  assert.equal(traceSpan(observation).phase,PREP_SPAN_PHASE,'observation must belong to the source-and-normalization preparation span');
  const partialBytes=Number(observed.hashed_bytes),sourceBytes=observedSourceBytes(observed.source_bytes);
  assert.ok(Number.isSafeInteger(partialBytes)&&partialBytes>0,'zero/tiny partial bytes are not a cold checkpoint');
  assert.ok(Number.isSafeInteger(sourceBytes)&&partialBytes<sourceBytes,'hashed_bytes must be partial, not the full source');
  const audit=receipt.sourceAudit;assert.ok(audit,'independent post-cancel source audit required');
  sha(audit.sha256);
  assert.ok(declaredShas.has(audit.sha256),'observed owned source identity differs from the declared checkpoint source');
  assert.ok(c.inputs?.some(input=>input.role==='checkpoint'&&input.sha256===audit.sha256),'observed owned source is not a verified checkpoint-role input');
  assert.equal(audit.path,observed.path,'post-cancel audit must bind the actual observed source path');
  assert.equal(audit.sizeBytes,sourceBytes,'source audit size differs from the observed source_bytes');
  // The owned install may be removed by the lease-release witness after the audit, so the
  // retained proof is the audit's declared-input sha (rehashed by runCase on the original
  // checkpoint input before/after) plus the owned-root path, never a later re-stat.
  const ownedRoot=contentReadback?audit.ownedRoot:await realpath(p.output);assert.equal(audit.ownedRoot,ownedRoot,'owned source audit root differs from actual backend output root');
  assert.ok(ownedBelow(ownedRoot,audit.canonicalPath)&&(ownedBelow(ownedRoot,audit.path)||ownedBelow(p.output,audit.path)),'observed source escaped the owned output/service roots');
  const canceled=receipt.canceled;assert.ok(canceled,'actual Canceled source progress event required');
  const canceledFields=traceFields(canceled),canceledBytes=Number(canceledFields.hashed_bytes);assert.equal(canceledFields.event,'artifact_seal_hash_progress');assert.equal(tracedJobId(canceled),jobId);assert.equal(traceSpan(canceled).phase,PREP_SPAN_PHASE);const canceledSourceBytes=observedSourceBytes(canceledFields.source_bytes);assert.ok(canceledSourceBytes===null||canceledSourceBytes===sourceBytes,'source size changed during canceled hashing');
  assert.equal(canceledFields.path,observed.path,'Canceled source progress must refer to the same observed owned source path');assert.equal(canceledFields.state,'Canceled');assert.equal(canceledFields.acquisition,'ColdContentRead');
  assert.ok(Number.isSafeInteger(canceledBytes)&&canceledBytes>=partialBytes&&canceledBytes<sourceBytes,'Canceled must report exact partial bytes actually reached, below the full source');
  // Cold source HASH cancellation is BEFORE codec copying; a valid hash-only cancel has NO
  // h3_normalization_copy_progress event. A Writing/copy event means the hashing window was missed.
  assert.ok(!receipt.normalization,'a normalization copy event shows the hashing window was missed, not a cold hash cancellation');
  const terminal=receipt.terminal;assert.ok(terminal?.job,'terminal canceled job snapshot required');
  assert.equal(terminal.job.id,jobId);assert.equal(terminal.job.status,'canceled','job must terminate canceled');
  assert.ok(typeof terminal.job.message==='string'&&terminal.job.message.includes(PREP_CANCELED_MARKER),'joined-teardown canceled root message marker required');
  assert.equal(terminal.hardAbandoned,false,'hard-abandoned cancellation cannot prove lease release');
  assert.ok(Date.parse(audit.auditedAt)>=Date.parse(terminal.acknowledgedAt),'independent audit must occur AFTER joined cancellation, not postpone POST /cancel');
  const plannedRef=receipt.generationJob.modelProvenance?.plan??receipt.generationJob.payload?.modelProvenance?.plan??receipt.generationJob.payload?.expectedImportedModelRef;
  assert.ok(plannedRef?.checkpointId&&plannedRef.semanticDigest,'actual queued planned source commitment required, not a fabricated completed seal');assert.deepEqual(receipt.plannedImportRef,plannedRef);assert.equal(traceSpan(observation).checkpointId,plannedRef.checkpointId,'partial trace differs from planned checkpoint');
  assert.ok(Number.isInteger(terminal.elapsedMs)&&terminal.elapsedMs>=0&&terminal.elapsedMs<PREP_CANCEL_ACK_MS,`cooperative cancellation must be acknowledged under the ${PREP_CANCEL_ACK_MS}ms joined grace`);
  const requestedAt=Date.parse(receipt.cancelRequest?.requestedAt),acknowledgedAt=Date.parse(terminal.acknowledgedAt);assert.ok(Number.isFinite(requestedAt)&&Number.isFinite(acknowledgedAt));assert.equal(terminal.elapsedMs,acknowledgedAt-requestedAt,'cancel grace must derive from actual request/ack timestamps, not a free elapsed label');
  assert.equal(receipt.cancelRequest?.method,'POST');assert.equal(receipt.cancelRequest?.route,`/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`,'cooperative cancel request must target the actual generation job');
  const teardown=receipt.teardown;assert.equal(teardown?.guardGraceMs,PREP_CANCEL_ACK_MS);assert.equal(teardown?.guardAbandonMs,PREP_CANCEL_ABANDON_MS);
  const idle=teardown?.idle;assert.ok(idle?.status!=='exited'&&idle?.worker?.status==='idle'&&!idle.worker.currentJobId,'fresh owned worker Idle after joined teardown required');
  const removal=receipt.removal;assert.equal(removal?.method,'DELETE');assert.equal(removal?.route,`/api/v1/models/${encodeURIComponent(modelId)}?permanent=true`);
  assert.equal(removal?.response?.id,modelId,'removal must target the actually installed imported model');
  assert.equal(removal?.response?.removedManifestEntry,true,'refused model removal cannot prove source lease release');
  assert.ok(Date.parse(removal.at)>=Date.parse(idle.worker.lastSeenAt),'owned model removal must follow the fresh Idle acknowledgement');
  assert.deepEqual(receipt.generationJob?.payload?.model,modelId);
  // Raw retained trace binding (never the receipt dictionary alone).
  const lifecycle=await json(inside(dir,'service-lifecycle.json'));
  let raw='';
  for(const epoch of lifecycle.epochs??[]) for(const kind of ['api','worker']) raw+=await readFile(inside(dir,`${kind}-${epoch.epoch}.log`),'utf8')+'\n';
  const events=[];for(const line of raw.split('\n')){try{const value=JSON.parse(line);if(traceFields(value).event)events.push(value);}catch{}}
  const retained=event=>events.some(value=>JSON.stringify(value)===JSON.stringify(event));
  assert.ok(retained(observation),'partial cold observation is absent from retained worker/API trace');
  assert.ok(retained(canceled),'Canceled source progress is absent from retained trace');
  assert.ok(!events.some(event=>traceFields(event).event==='h3_normalization_copy_progress'&&String(tracedJobId(event))===String(jobId)),'retained trace shows normalization copy began; the cold hash cancellation window was missed');
  assert.ok(!events.some(event=>{const progress=coldSourceProgress(event),value=progress?.fields;return progress&&String(tracedJobId(event))===String(jobId)&&value.path===observed.path&&value.state==='Acquired';}),'retained trace shows the source was fully Acquired');
  assert.ok(!events.some(event=>traceFields(event).event==='h3_import_source_prepared'&&String(tracedJobId(event))===String(jobId)),'retained trace published h3_import_source_prepared for the canceled job');
  assert.ok(!events.some(event=>traceFields(event).event==='h3_import_prepared_load'&&String(tracedJobId(event))===String(jobId)),'retained trace contains a native load for the canceled job');
  // Actual request/job snapshots from the API transcript.
  const transcript=await json(inside(dir,'api-transcript.json'));
  const jobPath=`/api/v1/jobs/${encodeURIComponent(jobId)}`;
  assert.ok(transcript.transcript.some(row=>row.method==='GET'&&row.route===jobPath&&row.result?.id===jobId&&row.result?.status==='canceled'&&typeof row.result?.message==='string'&&row.result.message.includes(PREP_CANCELED_MARKER)),'retained API transcript lacks the actual canceled job snapshot with root marker');
  const submit=transcript.transcript.find(row=>row.method==='POST'&&row.route==='/api/v1/video/jobs'&&row.result?.id===jobId);
  assert.ok(submit,'retained API transcript lacks the actual generation submission');
  assert.equal(submit.request?.model,modelId,'queued generation did not reference the planned imported model');
  for(const [key,value]of Object.entries(submit.request))assert.deepEqual(receipt.generationJob?.payload?.[key],value,`queued generation differs from submitted ${key}`);
  const queuedRef=submit.result?.modelProvenance?.plan??submit.result?.payload?.modelProvenance?.plan??submit.result?.payload?.expectedImportedModelRef;
  assert.deepEqual(receipt.plannedImportRef,queuedRef,'planned commitment differs from actual API submission receipt');
  return {jobId,modelId,partialBytes,sourceBytes,elapsedMs:terminal.elapsedMs,checkpointSha256:audit.sha256,removalId:modelId};
}
export async function verifyAPIBindings(p,c,dir,decode=decodeAV,delta, {allowUnitFake=false,contentReadback=false}={}) {
  const bundle=await json(inside(dir,'source-job-output.json'));
  if(c.kind==='full-av'){assert.equal(bundle.bindings.length,1,'one actual native binding per reduced AV case');assert.equal(bundle.media.length,1,'one actual AV output per reduced AV case');assert.equal(bundle.requests.filter(r=>r.route==='/api/v1/video/jobs').length,1,'no hidden render fan-out');}
  if(isSourceCancelCase(c)){assert.equal(bundle.bindings.length,0,'cold cancellation completes no load');assert.equal(bundle.media.length,0,'cold cancellation produces no AV');assert.equal(bundle.metrics,undefined,'cold cancellation claims no completed-load peak');}
  assert.ok(bundle.proof_kind==='campaign_native'||(allowUnitFake&&bundle.proof_kind==='unit_fake'),'unit_fake is never campaign proof');
  assert.ok(c.declaredSourceBindings?.length,'declared input authority cannot be absent during readback');for(const source of c.declaredSourceBindings)assertLicenseAuthority(source,{allowUnitFake});
  if(c.kind==='full-av')for(const source of bundle.conditioningBindings??[]){assert.ok(c.inputs.some(i=>i.sha256===source.sha256),'undeclared actual conditioning source');assert.equal(await hashFile(await confined(dir,inside(dir,source.file))),source.sha256,'retained conditioning bytes changed');}
  const scenarioInput=c.inputs.find(i=>i.role==='scenario'&&i.path===c.runtime.scenario);assert.ok(scenarioInput&&typeof bundle.scenarioText==='string');assert.equal(createHash('sha256').update(bundle.scenarioText).digest('hex'),scenarioInput.sha256,'actual scenario bytes differ from bound input');const scenario=JSON.parse(bundle.scenarioText);assert.equal(scenario.caseId,c.id);
  assert.equal(bundle.build.sceneworks,p.sceneworks);assert.equal(bundle.build.inference,p.inference);
  if(!contentReadback){await confined(p.output,bundle.build.binary);assert.equal(await hashFile(bundle.build.binary),bundle.build.sha256,'owned service binary drift');}
  if(/\.dense-(f16|bf16|f32)$/.test(c.id)){const expected={'dense-f16':'F16','dense-bf16':'BF16','dense-f32':'F32'}[c.id.split('.').at(-1)];assert.equal(bundle.sourceDtypeFacts?.expectedBodyDtype,expected);assert.ok(bundle.sourceDtypeFacts.bodyMatrices>0);for(const file of bundle.sourceDtypeFacts.files)assert.ok(c.inputs.some(input=>input.role==='checkpoint'&&input.path===file.path&&input.sha256===file.sha256));if(!contentReadback)assert.deepEqual(bundle.sourceDtypeFacts,await verifyDenseSourceDtypes(c,p.output));}
  const lifecycle=await json(inside(dir,'service-lifecycle.json'));for(const record of lifecycle.drains)assert.equal(await hashFile(inside(dir,record.path)),record.sha256,'owned cancellation/restart record drift');
  if(c.id.endsWith('.restart-replay'))assert.ok(bundle.epochs.length>=2 && bundle.actions.some(v=>v.kind==='restart'),'restart/replay requires actual owned restart records');
  if(c.id.endsWith('.interruption-retry'))assert.ok(bundle.epochs.length>=2 && bundle.actions.some(v=>v.kind==='interrupt-import'),'import interruption/retry requires actual owned cooperative cancellation/restart records');
  const observed=[];
  for(const epoch of bundle.epochs)for(const kind of ['api','worker'])for(const line of (await readFile(inside(dir,`${kind}-${epoch.epoch}.log`),'utf8')).split('\n')){try{const event=JSON.parse(line);if((event.fields??event).event)observed.push(event);}catch{}}
  const byJob=new Map();
  for(const binding of bundle.bindings) {
    assert.ok(observed.some(event=>JSON.stringify(event)===JSON.stringify(binding.event)),'native closure must occur in retained authentic worker/API trace, not copied/fabricated binding metadata');
    const identity=loadIdentity(binding.event,binding.job,c.inputs);assert.deepEqual(identity,binding.runtimeIdentity);
    const selected=c.declaredSourceBindings?.find(v=>v.model.modelId===binding.job.payload.model);
    if(c.declaredSourceBindings){
      assert.ok(selected,'completed job selected an undeclared/stock model');assert.equal(identity.checkpointSha256,selected.model.sha256);assert.equal(normalizeLoadEvent(binding.event).identity.assembly.partition,selected.model.partition,'actual native partition differs from declared import');if(selected.model.topology==='pruned')assert.ok(normalizeLoadEvent(binding.event).identity.assembly.adalnPruning,'pruned declaration requires actual pruned assembly');
      const request=bundle.requests?.find(r=>r.result.id===binding.job.id&&r.route==='/api/v1/video/jobs')?.request;
      if(c.kind==='full-av')normalizedControlRequest(request,{conditioningBindings:bundle.conditioningBindings,runtimeIdentity:identity,declaredSourceBindings:c.declaredSourceBindings},{allowUnitFake});
      const requested=request?.loras??binding.job.payload.loras??[];
      const adapters=requestedAdapterBindings(selected.adapters??[],requested),prepared=normalizeLoadEvent(binding.event).prepared_sources;
      for(const input of [...(selected.model.components??[]),...adapters])assert.ok(prepared.some(s=>s.sha256===input.sha256),'requested component/adapter missing from actual prepared closure');
      for(const input of identity.adapters)assert.ok(adapters.some(a=>a.sha256===input.sha256),'unrequested adapter leaked into native cache/load');
    }
    assert.equal(normalizeLoadEvent(binding.event).backend,c.backend);assert.equal(binding.media.jobId,binding.job.id);
    assert.equal(await hashFile(inside(dir,binding.media.av)),binding.media.sha256);validateAVAgainstRequest(await decode(inside(dir,binding.media.av)),binding.job.payload,resolvedMediaContract(c,binding.job));byJob.set(binding.job.id,binding);
  }
  for(const job of bundle.jobs.filter(j=>j.status==='completed'&&j.payload?.model?.startsWith('imported_')))assert.ok(byJob.has(job.id)&&bundle.media.some(m=>m.jobId===job.id),'completed imported generation/replay is missing an actual native closure/output binding');
  for(const media of bundle.media) {
    if(bundle.jobs.find(j=>j.id===media.jobId)?.payload?.model?.startsWith('imported_'))assert.ok(byJob.has(media.jobId),'imported output cannot pass with zero load events');
    assert.equal(await hashFile(inside(dir,media.av)),media.sha256);const decoded=await decode(inside(dir,media.av)),job=bundle.jobs.find(v=>v.id===media.jobId);assert.ok(job?.status==='completed');
    validateAVAgainstRequest(decoded,job.payload,resolvedMediaContract(c,job));
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
  const identification=(c.sourceVariant==='opaque-renamed-linked'||c.id.endsWith('.renamed'))?await verifyIdentificationCoverage(c,dir,bundle,scenario,byJob,{allowUnitFake}):undefined;
  const ownership=c.id.endsWith('.managed-linked-equivalence')?await verifyOwnershipEquivalence(bundle,scenario,byJob,dir,decode,delta):undefined;
  const cache=c.id.endsWith('.cold-warm-staged')?verifyCacheCoverage(bundle,scenario,byJob,observed):undefined;
  return {jobs:bundle.media.map(v=>v.jobId),nativeClosures:bundle.bindings.length,comparisons:comparisons.length,...(identification?{identification}:{}),...(ownership?{ownershipEquivalence:ownership}:{}),...(cache?{cacheCoverage:cache}:{})};
}
export async function executeAPICase(p,c,swRoot,inferenceRoot,dir,deps={}) {
  const service=await new OwnedService(p,c,swRoot,inferenceRoot,dir,deps).initialize();
  const observations=new ProcessObservations(service,{sample:deps.sample??processSample,sleepImpl:deps.sleep??sleep,intervalMs:c.runtime.sampleIntervalMs??250}).start();
  const records=[],actions=[],bound=[],traceEvents=[];let result,metrics,error;
  try {
    const scenarioText=await readFile(c.runtime.scenario,'utf8'),scenario=JSON.parse(scenarioText);assert.equal(scenario.caseId,c.id);assert.equal(scenario.baseUrl,service.baseUrl,'scenario must target newly owned service');
    if(scenario.identificationControl){const input=c.inputs.find(value=>value.role==='identity-control-fixture'&&value.path===scenario.identificationControl.receipt);assert.ok(input,'staged identity-control metadata input required');assert.ok((await stat(input.path)).size<=1024*1024,'identity control receipt must be bounded metadata, never a weight payload');assert.equal(await hashFile(input.path),input.sha256);await copyFile(input.path,inside(dir,'identification-input.json'));}
    result=await runScenario(scenario,{uploadFile:async spec=>{assert.ok(typeof spec.fileName==='string'&&!/[\\/]/.test(spec.fileName),'bounded uploaded filename required');const input=c.inputs.find(i=>i.path===spec.path&&i.sha256===spec.sha256&&i.role.startsWith('conditioning-'));assert.ok(input,'upload must bind an actual declared conditioning input');const file=await confinedTaskSource(p,spec.path),info=await stat(file);assert.ok(info.size>0&&info.size<=256*1024*1024);const data=await readFile(file);assert.equal(createHash('sha256').update(data).digest('hex'),spec.sha256);return new Blob([data],{type:spec.mimeType});},deadlineMs:deps.deadlineMs??(service.fake?Infinity:Number(process.env.H3_CASE_DEADLINE_MS??Infinity)),fetchImpl:(url,options)=>service.fetch(url,options),sleep:deps.sleep??sleep,beforeStep:async ({step,captures})=>{assert.ok(!observations.budgetExceeded,'campaign observation bound exceeded; no further job submission');await service.checkStep(step,captures);for(const action of [...(scenario.lifecycleActions??[]),...(c.runtime.actions??[])])if(action.beforeStep===step.id)actions.push(await service.action(action,captures));},onRecord:async r=>{
      records.push(r);
      if(r.result?.id && r.method==='POST' && (r.route==='/api/v1/models/import'||r.route==='/api/v1/loras/import'||r.route==='/api/v1/video/jobs'||/\/(retry|duplicate)$/.test(r.route)))service.jobs.add(r.result.id);
      await writeFile(path.join(dir,'api-transcript.json'),JSON.stringify({caseId:c.id,proof_kind:service.fake?'unit_fake':'campaign_native',transcript:records},null,2));
    }});
    if(c.id.endsWith('.cancel')) {
      const imported=records.filter(r=>r.method==='POST'&&r.route==='/api/v1/models/import').map(r=>r.result.id);
      assert.ok(imported.some(id=>records.some(r=>r.method==='POST'&&r.route===`/api/v1/jobs/${encodeURIComponent(id)}/cancel`)&&Object.values(result.captures).some(j=>j?.id===id&&j.status==='canceled')),'cancel case requires an actually cancelled submitted import, not import-only success');
    }
    const jobs=[...new Map(Object.values(result.captures).filter(v=>v?.id&&v.status==='completed'&&v.payload?.model).map(j=>[j.id,j])).values()];
    if(c.kind==='full-av')assert.equal(jobs.length,1,'each reduced full-AV cell must contain exactly one actual generation, not hidden fan-out');
    const conditioningBindings=c.kind==='full-av'?await bindConditioningSources(service,c,jobs,records,dir):[];
    const measured=c.kind==='full-av'||['artifact','capability'].includes(c.group)||/\.(bounded-loading|cold-warm-staged)$/.test(c.id);
    if(measured)metrics=await observations.finish();
    else {observations.done=true;await observations.loop;assert.ok(!observations.budgetExceeded,'observed campaign bound exceeded');}
    await save(path.join(dir,'raw-observations.json'),{samples:observations.samples,gaps:observations.gaps,epochs:service.epochs,meaning:measured?'Whole case/load-inclusive observations':'Observed only; no successful model load or peak is invented for a refusal/cancel or multi-process restart'});
    for(const [i,job] of jobs.entries())bound.push(...await bindCompletedMedia(service,job,dir,`output-${i}`,deps.decode??decodeAV));
    // Collect actual JSON tracing after the completed jobs; no synthetic load event is created.
    const traceLogs=[service.api.log(),service.worker.log()];
    for(const epoch of service.epochs.slice(0,-1))for(const kind of ['api','worker'])traceLogs.push(await readFile(path.join(dir,`${kind}-${epoch.epoch}.log`),'utf8'));
    for(const text of traceLogs)for(const line of text.split('\n')){try{const event=JSON.parse(line);if((event.fields??event).event==='h3_import_prepared_load')traceEvents.push(event);}catch{}}
    const loads=bound.map(b=>({media:b,job:jobs.find(j=>j.id===b.jobId),event:traceEvents.find(t=>(t.fields??t).jobId===b.jobId)}));
    const imported=loads.filter(v=>v.job.payload.model?.startsWith('imported_'));
    const matched=imported.filter(v=>v.event);assert.equal(matched.length,imported.length,'EVERY successful imported generation/replay needs its actual native load closure');
    const sourceCancel=actions.find(action=>action.actionKind==='source-cancel');
    // A canceled cold preparation has NO successful load, AV or peak requirement; only a
    // successful generation path requires one.
    if(!isSourceCancelCase(c)&&scenario.steps.some(s=>s.path==='/api/v1/video/jobs'||typeof s.path==='string'&&s.path.endsWith('/duplicate')))assert.ok(loads.length>0,'no actual successful model execution');
    if(isSourceCancelCase(c))assert.ok(sourceCancel,'cold source-preparation cancellation case did not produce an actual action receipt');
    const proofs=matched.map(v=>({...v,runtimeIdentity:loadIdentity(v.event,v.job,c.inputs)}));
    await save(path.join(dir,'source-job-output.json'),{proof_kind:service.fake?'unit_fake':'campaign_native',build:service.buildIdentity,epochs:service.epochs,...(service.sourceDtypeFacts?{sourceDtypeFacts:service.sourceDtypeFacts}:{}),scenarioText,conditioningBindings,captures:result.captures,bindings:proofs,jobs,requests:records.filter(r=>r.method==='POST'&&r.result?.id),media:bound,actions,metrics});
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
    const sourceCancelReceipt=sourceCancel?receiptNameFor(sourceCancel.jobId):undefined;
    const e={status:'executed',proof_kind:service.fake?'unit_fake':'campaign_native',sceneworks:p.sceneworks,inference:p.inference,caseId:c.id,backend:c.backend,hardware:h,runtimeBindings:'source-job-output.json',files:[...bound.map(v=>v.av),...conditioningBindings.map(v=>v.file),...(scenario.identificationControl?['identification-input.json']:[]),'api-transcript.json','source-job-output.json','comparisons.json','service-build.json','service-build.log','raw-observations.json','service-lifecycle.json','source-cancel-actions.json',...(sourceCancelReceipt?[sourceCancelReceipt]:[]),...service.epochs.flatMap(v=>[`api-${v.epoch}.log`,`worker-${v.epoch}.log`,`guardian-${v.epoch}.log`,`ownership-${v.epoch}.json`,`exits-${v.epoch}.json`])],apiTranscript:'api-transcript.json',testInputSha256:c.inputs[0].sha256,metrics,...(sourceCancelReceipt?{sourceCancelReceipt}:{})};
    if(primary){Object.assign(e,{runtimeIdentity:primary.runtimeIdentity,loadEvent:primary.event,completedJob:{id:primary.job.id,status:primary.job.status,providerId:normalizeLoadEvent(primary.event).providerId,output:primary.media.av},av:primary.media.av});}
    if(c.artifact)Object.assign(e,{source:p.source,artifact:c.artifact});
    if(c.group==='capability') {assert.ok(comparisons.length);const control=proofs.find(v=>v.job.id===comparisons[0].interventionJobId);e.control={av:control.media.av,checkpointSha256:control.runtimeIdentity.checkpointSha256,seed:control.runtimeIdentity.seed,intervention:'Actual scenario comparison manifest; see all source-bound contrasts',comparisons:'comparisons.json',bindings:'source-job-output.json'};}
    await save(path.join(dir,c.evidence),e);return {evidence:e,result,comparisons};
  } catch(e){error=e;throw e;} finally {
    observations.done=true;await observations.loop;
    let observationWriteError;
    try{await save(path.join(dir,'sampler-exit.json'),{samples:observations.samples,gaps:observations.gaps,budgetExceeded:observations.budgetExceeded,proof_kind:service.fake?'unit_fake':'campaign_native'});}catch(e){observationWriteError=e;}
    // Always persist the actual action receipts, including a failed/missed cancellation.
    try{await writeFile(path.join(dir,'source-cancel-actions.json'),JSON.stringify({proof_kind:service.fake?'unit_fake':'campaign_native',actions:actions.filter(action=>action.actionKind==='source-cancel')},null,2)+'\n');}catch{}
    try{await service.stop();await save(path.join(dir,'service-lifecycle.json'),{epochs:service.epochs,quiescence:service.quiescence,drains:await Promise.all(service.drainRecords.map(async name=>({path:name,sha256:await hashFile(path.join(dir,name))}))),observationWriteError:observationWriteError?.message,proof_kind:service.fake?'unit_fake':'campaign_native'});if(observationWriteError&&!error)throw observationWriteError;}catch(stopError){await save(path.join(dir,'shutdown-failure.json'),{error:stopError.message,originalError:error?.message,epochs:service.epochs});if(!error)throw stopError;}
  }
}
const conditioningFields={sourceAssetId:'image',lastFrameAssetId:'image',referenceAssetIds:'image',sourceClipAssetId:'video',sourceClipAssetIds:'video',referenceClipAssetId:'video',referenceAudioAssetIds:'audio'};
async function bindConditioningSources(service,c,jobs,requests,dir){
  const bindings=[];
  for(const job of jobs){const request=submittedPayload(requests,job),projectId=request.projectId;assert.ok(typeof projectId==='string'&&projectId);
    for(const [field,type]of Object.entries(conditioningFields))for(const id of (Array.isArray(request[field])?request[field]:request[field]?[request[field]]:[])){
      if(bindings.some(v=>v.projectId===projectId&&v.assetId===id))continue;
      const asset=await service.request('GET',`/api/v1/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(id)}`),project=await service.request('GET',`/api/v1/projects/${encodeURIComponent(projectId)}`);
      assert.equal(asset.id,id);assert.equal(asset.projectId,projectId);assert.equal(asset.type,type);assert.equal(project.id,projectId);
      const root=await confined(service.runDir,project.path),source=await confined(root,path.resolve(root,asset.file.path)),before=await stat(source);assert.ok(before.size>0&&before.size<=256*1024*1024,'conditioning evidence must be bounded media, never model weights');
      const sha256=await hashFile(source);assert.ok(c.inputs.some(i=>i.sha256===sha256),'actual owned conditioning bytes must match a declared input SHA');
      const extension=path.extname(source),file=`conditioning-${bindings.length}${/^\.[a-z0-9]{1,10}$/i.test(extension)?extension:'.bin'}`;await copyFile(source,inside(dir,file));assert.equal(await hashFile(inside(dir,file)),sha256);
      const after=await stat(source);for(const key of ['dev','ino','size','mtimeMs','ctimeMs'])assert.equal(after[key],before[key],'conditioning changed during source binding');
      bindings.push({projectId,assetId:id,source,file,sha256,metadata:{type:asset.type,...Object.fromEntries(['mimeType','width','height','duration','fps'].map(k=>[k,asset.file[k]??null]))}});
    }
  }
  return bindings;
}
export function normalizedControlRequest(request,{conditioningBindings,runtimeIdentity,declaredSourceBindings}, {allowUnitFake=false}={}){
  const result=creativePayload(request),source=declaredSourceBindings?.find(v=>v.model.modelId===request.model);
  assert.ok(runtimeIdentity?.checkpointSha256,'actual resolved checkpoint identity required');
  if(!allowUnitFake)assert.ok(Array.isArray(conditioningBindings)&&source,'actual conditioning bindings and declared imported source required');
  result.projectId='verified-task-owned-project';
  result.model={checkpointSha256:runtimeIdentity.checkpointSha256,topology:runtimeIdentity.topology??null,quantization:runtimeIdentity.quantization??null,components:(runtimeIdentity.components??[]).map(v=>v.sha256)};
  for(const field of Object.keys(conditioningFields))if(result[field]!==undefined){
    const resolve=id=>{const binding=conditioningBindings?.find(v=>v.projectId===request.projectId&&v.assetId===id);if(!binding){assert.ok(allowUnitFake,'reference asset has no actual retained source-byte binding');return id;}assert.equal(binding.metadata.type,conditioningFields[field]);return {sha256:binding.sha256,metadata:binding.metadata};};
    result[field]=Array.isArray(result[field])?result[field].map(resolve):resolve(result[field]);
  }
  if(source&&result.loras)result.loras=result.loras.map(value=>{const selected=source.adapters?.find(a=>a.id===value.id);assert.ok(selected,'requested adapter lacks declared source identity');const {id,...controls}=value;return {...controls,sha256:selected.sha256};});
  return result;
}
// Shared baselines refer to actual retained case jobs. No extra generation or copied execution.
export async function verifyReducedControls(p,backend,{decode=decodeAV,delta=decodedDelta,caseRoot=p.output}={}) {
  const rows=new Map();
  for(const c of executionCases(p).filter(c=>c.backend===backend&&c.kind==='full-av')){
    const dir=inside(caseRoot,c.id),bundle=await json(inside(dir,'source-job-output.json'));
    assert.equal(bundle.bindings.length,1,'reduced AV receipt must identify one actual native job');
    const binding=bundle.bindings[0];assert.equal(binding.job.status,'completed');
    for(const source of bundle.conditioningBindings??[]){assert.equal(await hashFile(await confined(dir,inside(dir,source.file))),source.sha256);assert.ok(c.inputs?.some(i=>i.sha256===source.sha256),'retained conditioning must match a declared source input');}
    const rawRequest=submittedPayload(bundle.requests??[],binding.job),request=normalizedControlRequest(rawRequest,{conditioningBindings:bundle.conditioningBindings,runtimeIdentity:binding.runtimeIdentity,declaredSourceBindings:c.declaredSourceBindings},{allowUnitFake:p.proof_kind==='unit_fake'});
    rows.set(c.index,{c,dir,binding,request,rawRequest});
  }
  const results=[];
  for(const [repeatIndex,baselineIndex] of [[5,1],[13,3],[22,15]]){
    const repeat=rows.get(repeatIndex);if(!repeat)continue;const baseline=rows.get(baselineIndex);assert.ok(baseline,'actual repeated baseline source missing');
    assert.notEqual(repeat.binding.job.id,baseline.binding.job.id,'a repeat must actually execute, not duplicate a job receipt');
    assert.deepEqual(creativePayload(repeat.request),creativePayload(baseline.request),'unchanged baseline source/seed/recipe/conditioning required');
    assert.equal(repeat.binding.runtimeIdentity.checkpointSha256,baseline.binding.runtimeIdentity.checkpointSha256);
    for(const stream of ['video','audio']){const measured=await delta(inside(baseline.dir,baseline.binding.media.av),inside(repeat.dir,repeat.binding.media.av),stream);assert.ok(Number.isFinite(measured)&&measured<=repeat.c.sensitivity.maximumBaselineDelta,'unstable repeated baseline is UNPROVEN, not a causal pass');}
  }
  const controls={6:[1,5,'referenceAssetIds','video'],7:[1,5,'sourceClipAssetIds','video'],8:[1,5,'referenceAudioAssetIds','audio'],9:[1,5,'referenceAssetIds','video'],10:[1,5,'sourceClipAssetIds','video'],11:[1,5,'referenceAudioAssetIds','audio'],12:[1,5,'loras','video'],14:[3,13,'loras','video'],16:[15,22,'prompt','video']};
  for(const [index,[baseIndex,repeatIndex,field,stream]]of Object.entries(controls)){
    const changed=rows.get(Number(index));if(!changed)continue;
    const baseline=rows.get(baseIndex),repeat=rows.get(repeatIndex);assert.ok(baseline&&repeat,'actual baseline and independently executed repeat required');
    assert.equal(new Set([baseline.binding.job.id,repeat.binding.job.id,changed.binding.job.id]).size,3,'intervention and repeat cannot reuse the baseline job ID');
    for(const other of [repeat,changed]){assert.equal(other.binding.runtimeIdentity.checkpointSha256,baseline.binding.runtimeIdentity.checkpointSha256);assert.deepEqual(other.binding.runtimeIdentity.samplingPolicy,baseline.binding.runtimeIdentity.samplingPolicy);}
    assertControlledConfiguration(baseline.request,changed.request,[field]);assert.notDeepEqual(changed.request[field],baseline.request[field],'declared intervention did not change its actual request');
    const thresholds=changed.c.sensitivity;assert.ok(thresholds&&Number.isFinite(thresholds.minimumControlDelta)&&thresholds.minimumControlDelta>0&&Number.isFinite(thresholds.maximumBaselineDelta)&&thresholds.maximumBaselineDelta>=0,'predeclared sensitivity thresholds required');
    const measured=await compareControlMedia(inside(baseline.dir,baseline.binding.media.av),inside(repeat.dir,repeat.binding.media.av),inside(changed.dir,changed.binding.media.av),stream,thresholds.minimumControlDelta,thresholds.maximumBaselineDelta,decode,delta);
    results.push({caseId:changed.c.id,baselineCaseId:baseline.c.id,repeatedCaseId:repeat.c.id,baselineJobId:baseline.binding.job.id,repeatedJobId:repeat.binding.job.id,interventionJobId:changed.binding.job.id,allowedChanges:[field],stream,locatorNormalization:{fields:['projectId','model',...Object.keys(conditioningFields).filter(k=>baseline.rawRequest[k]!==undefined),...(baseline.rawRequest.loras?['loras']:[])],referenceOrderPreserved:true,comparedLiterally:Object.keys(baseline.rawRequest).filter(k=>!['projectId','model','loras',...Object.keys(conditioningFields)].includes(k)),meaning:'Only organizational locators map to actual retained input bytes and resolved source identities; creative values and reference ordering remain literal.'},...measured});
  }
  return results;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  try{const [mode,planPath,id,swRoot,inferenceRoot]=process.argv.slice(2);const p=await json(planPath),c=p.cases.find(v=>v.id===id);assert.ok(c);authorized(p,c);const dir=process.env.H3_CASE_DIR;assert.equal(process.env.H3_CASE_ID,id);assert.ok(dir);await confined(p.output,dir);
    if(mode==='generation-source-prep-cancel'){
      assert.ok(isSourceCancelCase(c),'this dispatcher is only for the canonical cold source-preparation cancellation case');
      const r=await executeAPICase(p,c,swRoot,inferenceRoot,dir);
      console.log(`H3_API_ASSERTIONS_EXECUTED=${r.result.transcript.filter(v=>v.assertion).length}`);
    }
    else if(mode==='actual-artifact-probe'){const {executeActualArtifactProbe}=await import('./h3-finetune-terminal-artifact-probe.mjs');const r=await executeActualArtifactProbe(p,c,swRoot,p.nativeEvidence?.root??inferenceRoot,dir);console.log(JSON.stringify(r.proofSummary));}
    else if(mode==='numeric'){const r=await executeNativeComparison(p,c,swRoot,inferenceRoot,dir);console.log(r.log);}
    else if(mode==='fixture'){const r=await executeFixture(p,c,swRoot,inferenceRoot,dir);console.log(r.log);}
    else{assert.equal(mode,'api');const r=await executeAPICase(p,c,swRoot,inferenceRoot,dir);console.log(`H3_API_ASSERTIONS_EXECUTED=${r.result.transcript.filter(v=>v.assertion).length}`);}
  }catch(error){console.error(error.stack);process.exitCode=1;}
}
