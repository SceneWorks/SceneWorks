#!/usr/bin/env node
// Dispatch-only, offline terminal campaign. This is NOT the bounded CPU fixture driver.
// Native code owns training/evaluation observations; this driver never manufactures evidence.
import { spawn, spawnSync } from 'node:child_process';
import { closeSync, copyFileSync, existsSync, lstatSync, mkdirSync, openSync, readFileSync, readSync, readdirSync, writeFileSync, writeSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { inflateSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { createInterface } from 'node:readline';
import { digest, parseTestResult } from './moss-consumer-assertions.mjs';
import { defaultInspect, parseTestList, selectLibTestExecutable, executePreparation, isPreparationCommand, PRE_MODEL_CPU_TEST } from './moss-native-api-validation.mjs';
import { stageLatestMossArtifacts } from './moss-training-artifacts.mjs';
import { inventoryMossTrainingCache } from './moss-training-cache-inventory.mjs';
import { lockedInferenceRevision } from './select-windows-cargo-cache.mjs';

export const MANIFEST_SHA256 = 'c02b05e3c40e6c9666554580ff0636d5853691ffe5f5db5d586d12468c82a0fd';
export const BOUNDED_PRODUCT = 'tests::moss_training_acceleration::moss_bounded_product_acceptance';
export const BOUNDED_GPU_UUID = 'GPU-e4b79931-7be6-f216-460a-f5405cfafffe';
export const BOUNDED_GPU_CAP_BYTES = 80 * 1024 ** 3;
export const BOUNDED_GPU_RESERVE_BYTES = 2 * 1024 ** 3;
export const PREFIX = 'tests::moss_training_acceptance::';
export const CELLS = ['base/lora', 'base/lokr', 'sft/lora', 'sft/lokr'];
export const testsFor = (phase, executionBackend='mlx') => {
  const sourceBackend=phase==='cross-load'?(executionBackend==='mlx'?'cuda':'mlx'):executionBackend;
  return CELLS.map(cell => `${PREFIX}moss_real_weights_${sourceBackend}_${cell.replace('/', '_')}_${phase === 'cross-load' ? 'cross_backend' : 'round_trip'}`);
};
export const PREFLIGHT = `${PREFIX}moss_real_weights_device_preflight`;
// Match sceneworks_worker::WORKER_ENTRY_STACK_BYTES and the native API worker
// harness. The unoptimized shared body poll frame alone exceeds libtest's 2 MiB
// default on macOS; this applies to exact native tests, never preparation tools.
export const NATIVE_TEST_STACK_BYTES = 8 * 1024 * 1024;
export const BOUNDS = Object.freeze({ build: 7200_000, case: 3600_000, crossLoad: 3600_000, list: 300_000, release: 30_000 });
const fail = message => { throw new Error(`MOSS acceptance: ${message}`); };
const assert = (value, message) => { if (!value) fail(message); };
const json = file => JSON.parse(readFileSync(file, 'utf8'));
const store = (file, value) => { mkdirSync(path.dirname(file), { recursive: true }); writeFileSync(file, JSON.stringify(value, null, 2) + '\n'); };
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

export function parseArgs(argv) {
  const o = {};
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i]?.slice(2);
    assert(argv[i]?.startsWith('--') && argv[i + 1] && ['phase','root','output','backend','manifest','peer-artifacts','capture-cell'].includes(key) && !Object.hasOwn(o, key), 'expected unique known --option value pairs');
    o[key] = argv[i + 1];
  }
  for (const key of ['phase','root','output','backend','manifest']) assert(o[key], `--${key} is required`);
  assert(['capture','cross-load','finalize','bounded-product'].includes(o.phase), 'invalid phase');
  assert(['mlx','cuda'].includes(o.backend), 'backend must be mlx or cuda');
  for (const key of ['root','output','manifest','peer-artifacts']) if (o[key]) o[key] = path.resolve(o[key]);
  assert(!inside(o.root, o.output), 'output must be outside source checkout');
  if (['cross-load','finalize'].includes(o.phase)) assert(o['peer-artifacts'], '--peer-artifacts required for cross-load/finalize');
  if (o['capture-cell']) assert(o.phase === 'capture' && CELLS.includes(o['capture-cell']), 'capture-cell must select an exact capture cell');
  if (o.phase === 'bounded-product') assert(o.backend === 'cuda' && !o['peer-artifacts'] && !o['capture-cell'], 'bounded product requires CUDA and no campaign selectors');
  return o;
}
function inside(root, file) { const r = path.relative(root, file); return r === '' || (!r.startsWith(`..${path.sep}`) && r !== '..' && !path.isAbsolute(r)); }

// Stream large safetensors/exports: hashing must not allocate a model-sized buffer.
export function fileDigest(file) {
  const fd=openSync(file,'r'), hash=createHash('sha256'), buffer=Buffer.allocUnsafe(1024*1024);
  try { let n; while((n=readSync(fd,buffer,0,buffer.length,null))>0) hash.update(buffer.subarray(0,n)); return hash.digest('hex'); }
  finally { closeSync(fd); }
}
// Yield between bounded reads so owned signal cancellation is observable during full auth.
export async function fileDigestChecked(file, check, onChunk = () => {}) {
  check();
  const fd=openSync(file,'r'), hash=createHash('sha256'), buffer=Buffer.allocUnsafe(1024*1024);
  try {
    while(true) {
      check();const n=readSync(fd,buffer,0,buffer.length,null);
      if(n===0)break;
      hash.update(buffer.subarray(0,n));onChunk(n);
      await new Promise(resolve=>setImmediate(resolve));check();
    }
    check();return hash.digest('hex');
  } finally {closeSync(fd);}
}
// Every artifact member is bound to bytes, never just metadata or a filename glob.
export function artifactMembers(root) {
  const members = [];
  function walk(dir) {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const file = path.join(dir, entry.name);
      assert(!lstatSync(file).isSymbolicLink(), `artifact symlink forbidden: ${file}`);
      if (entry.isDirectory()) walk(file);
      else { assert(entry.isFile(), `nonregular artifact: ${file}`); members.push({ path: path.relative(root, file).split(path.sep).join('/'), sha256: fileDigest(file) }); }
    }
  }
  walk(root);
  return members.sort((a,b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
}
export function verifyBundle(dir, source, phase, backend) {
  const receipt = json(path.join(dir, 'driver.json'));
  assert(receipt.ok && receipt.phase === phase && receipt.backend === backend, `incomplete ${backend}/${phase} bundle`);
  for (const key of ['workflowSource','inferencePin','manifestSha256','cargoLockSha256','driverSha256','workflowSha256']) assert(receipt.source[key] === source[key], `peer source mismatch: ${key}`);
  assert(receipt.cases?.length === CELLS.length && receipt.cases.every((c,i) => c.executionPassed && c.caseId === `${phase === 'cross-load' ? (backend === 'mlx' ? 'cuda' : 'mlx') : backend}/${CELLS[i]}`), 'all four exact completed cells required');
  const actual = artifactMembers(dir).filter(m => m.path !== 'driver.json');
  assert(JSON.stringify(actual) === JSON.stringify(receipt.members), 'artifact members changed, missing or added');
  return receipt;
}

export function sourceFacts(root, manifest, env, inspect = defaultInspect) {
  assert(env.GITHUB_EVENT_NAME === 'workflow_dispatch', 'requires explicit workflow_dispatch');
  assert(/^[a-f0-9]{40}$/.test(env.MOSS_WORKFLOW_SOURCE ?? '') && env.MOSS_WORKFLOW_SOURCE === env.GITHUB_SHA, 'workflowSource must be exact dispatched GitHub SHA');
  const snapshot = inspect(root);
  assert(snapshot.clean && snapshot.head === env.GITHUB_SHA, 'requires clean exact dispatched source');
  assert(digest(readFileSync(manifest)) === MANIFEST_SHA256, 'sealed campaign manifest hash mismatch');
  const lock = readFileSync(path.join(root, 'Cargo.lock'));
  const pin = lockedInferenceRevision(lock.toString());
  assert(/^[a-f0-9]{40}$/.test(pin) && (!env.MOSS_INFERENCE_PIN || pin === env.MOSS_INFERENCE_PIN), 'permanent Cargo.lock inference pin must equal explicit input when supplied');
  // Terminal execution may precede feature -> main, but the selected source
  // must already be merged into the owning epic integration branch or main.
  const bounded = env.MOSS_ACCEPTANCE_PHASE === 'bounded-product';
  const mergedRefs=[bounded ? 'refs/remotes/origin/feature/sc-26850-cuda-training-acceleration' : 'refs/remotes/origin/feature/sc-26584-moss-image-training','refs/remotes/origin/main'];
  const merged=mergedRefs.some(ref=>spawnSync('git',['merge-base','--is-ancestor',snapshot.head,ref],{cwd:root}).status===0);
  assert(merged, 'workflowSource is not proven merged into the owning epic integration branch or main');
  const inferenceRoot=env.MOSS_INFERENCE_SOURCE_ROOT;
  assert(inferenceRoot && existsSync(inferenceRoot), 'missing-source-access: provide a read-only inference/main checkout using existing SCENEWORKS_INFERENCE_READ_TOKEN; Cargo.lock alone is not ancestry proof');
  const ancestry=spawnSync('git',['merge-base','--is-ancestor',pin,'HEAD'],{cwd:inferenceRoot});
  const branch=spawnSync('git',['symbolic-ref','--short','HEAD'],{cwd:inferenceRoot,encoding:'utf8'});
  const remote=spawnSync('git',['remote','get-url','origin'],{cwd:inferenceRoot,encoding:'utf8'});
  assert(ancestry.status===0 && branch.status===0 && branch.stdout.trim()==='main' && remote.status===0 && /github\.com[:/]SceneWorks\/inference(?:\.git)?\s*$/.test(remote.stdout), 'permanent inference pin must be proven reachable from the actual inference/main source checkout');
  for (const name of ['Cargo.toml','crates/sceneworks-worker/Cargo.toml','crates/sceneworks-memory-adapter/Cargo.toml']) {
    const text = readFileSync(path.join(root,name),'utf8');
    assert(!/\[patch\.[^\]]*inference[^\]]*\]/i.test(text), 'inference dependency overrides forbidden');
  }
  for (const key of Object.keys(env)) assert(!/^CARGO_(?:PATCH|SOURCE)_/.test(key), 'Cargo source/patch overrides forbidden');
  return { ...(bounded ? {boundedRustSha256:fileDigest(path.join(root,'apps/rust-api/src/tests/moss_training_acceleration.rs')),deviceGuardSha256:fileDigest(path.join(root,'scripts/ci/moss-training-acceptance/bounded_product_claim.py'))} : {}), workflowSource: snapshot.head, inferencePin: pin, manifestSha256: MANIFEST_SHA256, finalMergedSource:true, clean:snapshot.clean, dependencyOverridesUsed:false, cargoLockSha256: digest(lock), driverSha256: digest(readFileSync(path.join(root,'scripts/moss-training-acceptance.mjs'))), workflowSha256: digest(readFileSync(path.join(root,'.github/workflows/moss-training-acceptance.yml'))) };
}

export function checkDiscovery(all, ignored, backend='mlx') {
  const expected = [...testsFor('capture',backend), ...testsFor('cross-load',backend), PREFLIGHT].sort();
  for (const [label, entries] of [['unfiltered',all],['ignored',ignored]]) {
    const names = entries.filter(e => e.kind === 'test' && e.name.startsWith(PREFIX)).map(e => e.name).sort();
    assert(JSON.stringify(names) === JSON.stringify(expected), `${label} discovery must contain exactly the acceptance FQNs (including preflight)`);
  }
  return expected;
}

function validatePreflightFields(p, backend, source, caseId, phase = 'capture') {
  assert(p.phase === phase, 'preflight phase mismatch');
  assert(p.caseId === caseId && p.verdict === 'admitted' && p.refusalReason === null, 'exact cell must have admitted native preweight preflight');
  assert(p.modelRevisions && Object.keys(p.modelRevisions).length > 0 && Object.values(p.modelRevisions).every(r => /^[0-9a-f]{40}$/.test(r)), 'expected component revisions required');
  for (const key of ['resolvedCapProvenance','headerDemandProvenance']) assert(typeof p[key] === 'string' && p[key].length > 0, `native ${key} required`);
  for (const key of ['workflowSource','inferencePin','manifestSha256']) assert(p[key] === source[key], `preflight ${key} mismatch`);
  assert(p.backend === backend && p.offline === true && p.weightsLoaded === false, 'preflight must be offline and before weights load');
  const observations=p.observationSchema;
  if (phase === 'capture') assert(observations?.gradientObservations===true && observations?.fixedEvaluationPair===true && observations?.numericFactorDelta===true && observations?.preparedTensorSnapshots===true, 'native pin lacks required measured evaluation/gradient/factor/prepared-tensor diagnostic interfaces; refuse before weight loading');
  assert(typeof p.deviceIdentity === 'string' && p.deviceIdentity.length > 0, 'actual claimed device identity required');
  const fraction = backend === 'mlx' ? 0.60 : 0.85;
  assert(p.gpuMemoryLimitFraction === fraction, 'wrong product global fraction');
  for (const key of ['totalCapacityBytes','resolvedCapBytes','headerDerivedDemandBytes']) assert(Number.isSafeInteger(p[key]) && p[key] > 0, `positive actual ${key} required`);
  // The product transports the user's fraction as f32. Preserve that existing
  // representation rather than rejecting a valid native cap using f64 0.60/0.85.
  assert(p.resolvedCapBytes <= Math.floor(p.totalCapacityBytes * Math.fround(fraction)), 'resolved cap exceeds selected device fraction');
  assert(p.headerDerivedDemandBytes <= p.resolvedCapBytes, 'header-derived selected-phase demand exceeds admitted cap');
  assert(Array.isArray(p.weights) && p.weights.length > 0, 'cached weight headers and files required');
  for (const weight of p.weights) {
    assert(weight.present === true && path.isAbsolute(weight.path) && /^[0-9a-f]{64}$/.test(weight.sha256), 'present resolved cached weight path/hash required');
  }
  return p.resolvedCapBytes;
}

export function validatePreflight(p, backend, source, caseId, phase = 'capture') {
  const cap=validatePreflightFields(p,backend,source,caseId,phase);
  for(const weight of p.weights) assert(existsSync(weight.path) && fileDigest(weight.path)===weight.sha256,`offline cached weight missing or changed: ${weight.path}`);
  return cap;
}
export async function validatePreflightChecked(p, backend, source, caseId, phase, check) {
  const cap=validatePreflightFields(p,backend,source,caseId,phase);
  for(const weight of p.weights) assert(await fileDigestChecked(weight.path,check)===weight.sha256,`offline cached weight missing or changed: ${weight.path}`);
  check();return cap;
}

// The Rust API, native training and generation share this one launched process.
// Native samples enforce its unchanged cap; the parent owns exit/stdio proof.
// No descendant identity, metadata-helper absence or kernel peak claim is made.
export function validateNativeRoot(record, state, capBytes) {
  const raw=record?.raw, measured=raw?.rootProcess;
  assert(measured?.scope==='native-api-worker-root' && measured.pid===state.rootProcess?.pid,'native root PID/scope mismatch');
  assert(raw.serverJoined===true && Number.isSafeInteger(raw.nativeBodyReturnedAtUnixMs),'native/API joins missing');
  assert(measured.metadataHelperAbsenceClaim===false && measured.measurement==='native-current-process-rss-sampled-maximum','unsupported native RSS/release claim');
  assert(measured.firstFailure===null && measured.missingMeasurementCount===0,'native root RSS/cancellation failed');
  const ceiling=capBytes??measured.resolvedCeilingBytes;
  assert(Number.isSafeInteger(ceiling)&&ceiling>0&&measured.resolvedCeilingBytes===ceiling,'native root cap differs from admitted ceiling');
  assert(Array.isArray(measured.samples)&&measured.samples.length>=2,'positive native root RSS observations missing');
  let peak=0,last=-Infinity;
  for(const sample of measured.samples){
    const at=Date.parse(sample.at);
    assert(Number.isFinite(at)&&at>=last&&Number.isSafeInteger(sample.bytes)&&sample.bytes>0&&sample.bytes<=ceiling,'missing/zero/over-cap native root RSS');
    last=at;peak=Math.max(peak,sample.bytes);
  }
  assert(measured.sampledPeakBytes===peak,'native root sampled peak mismatch');
  return measured;
}

export async function executeSupervised(root, command, log, limitMs, state, {env=process.env,capBytes=null,cancelFile=`${log}.cancel`,releaseMs=BOUNDS.release,spawnChild=spawn,executableDigest=fileDigest,guardAlive=()=>true,onSpawn=()=>{}}={}) {
  const binarySha256=executableDigest(command[0]);
  assert(/^[a-f0-9]{64}$/.test(binarySha256),'exact native executable SHA required');
  const fd=openSync(log,'w');let child,closed=false,code=null,error=null,logOpen=true;
  const deadline=Date.now()+limitMs;let cancelAt=null,reason=null;
  try {
    child=spawnChild(command[0],command.slice(1),{cwd:root,env:{...env,MOSS_ACCEPTANCE_CANCEL_FILE:cancelFile},stdio:['ignore','pipe','pipe'],windowsHide:true});
    const completed=new Promise(resolve=>{
      child.on('error',e=>{error=e;});
      child.on('close',c=>{closed=true;code=c;resolve();});
    });
    assert(Number.isSafeInteger(child.pid)&&child.pid>0,'owned native root PID unavailable');
    state.rootProcess={scope:'native-api-worker-root',pid:child.pid,executable:command[0],binarySha256};
    onSpawn(state.rootProcess);
    for(const stream of [child.stdout,child.stderr]){
      stream.on('data',bytes=>{if(logOpen)try{writeSync(fd,bytes);}catch(e){error??=e;}});
      stream.on('error',e=>{error??=e;});
    }
    while(!closed){
      if(state.cancelled||error||!guardAlive()||Date.now()>=deadline){
        if(cancelAt===null){cancelAt=Date.now();reason??=error?.message??(!guardAlive()?'physical-device claim helper exited':'owned cancellation/deadline');writeFileSync(cancelFile,JSON.stringify({cancel:true,reason}));}
        if(Date.now()>=cancelAt+releaseMs)fail('native root did not cooperatively close; refusing next case or forced GPU kill');
      }
      await Promise.race([completed,pause(25)]);
    }
    state.cleanup={scope:'native-api-worker-root',released:true,stdioClosed:true,metadataHelperAbsenceClaim:false};
    if(state.cancelled||cancelAt!==null||Date.now()>=deadline)fail(reason??'owned cancellation/deadline');
    if(error)throw error;
    if(code===0&&env.MOSS_ACCEPTANCE_OUTPUT){
      const file=path.join(env.MOSS_ACCEPTANCE_OUTPUT,...(env.MOSS_ACCEPTANCE_PREFLIGHT_ONLY==='1'?['preflight']:[]),'evidence.json');
      validateNativeRoot(json(file),state,capBytes);
    }
    return {exit:code,output:readFileSync(log,'utf8')};
  } finally {
    state.cleanup??={scope:'native-api-worker-root',released:closed,stdioClosed:closed,metadataHelperAbsenceClaim:false};
    state.unsafe=!closed;
    logOpen=false;closeSync(fd);
  }
}

// Decode the actual bounded native PNG payload, not a filename or header-only proof.
export function decodePreviewPng(bytes) {
  assert(bytes.subarray(0,8).equals(Buffer.from('89504e470d0a1a0a','hex')), 'preview is not PNG');
  let offset=8, channels=null, ended=false; const data=[];
  const crc32=b=>{let c=0xffffffff;for(const v of b){c^=v;for(let i=0;i<8;i++)c=(c>>>1)^((c&1)?0xedb88320:0);}return (c^0xffffffff)>>>0;};
  while(offset<bytes.length) {
    assert(offset+12<=bytes.length,'truncated PNG chunk');
    const n=bytes.readUInt32BE(offset),end=offset+12+n;
    assert(end<=bytes.length,'truncated PNG data');
    const type=bytes.toString('ascii',offset+4,offset+8),chunk=bytes.subarray(offset+8,end-4);
    assert(crc32(bytes.subarray(offset+4,end-4))===bytes.readUInt32BE(end-4),'PNG CRC mismatch');
    if(offset===8) {
      assert(type==='IHDR' && n===13 && chunk.readUInt32BE(0)===256 && chunk.readUInt32BE(4)===256 && chunk[8]===8 && chunk[10]===0 && chunk[11]===0 && chunk[12]===0,'preview must be decoded 256x256 noninterlaced 8-bit PNG');
      channels=({0:1,2:3,4:2,6:4})[chunk[9]];assert(channels,'unsupported PNG color type');
    } else if(type==='IDAT') data.push(chunk);
    else if(type==='IEND') {assert(n===0 && end===bytes.length,'invalid PNG end');ended=true;break;}
    else assert(type!=='IHDR' && (type.charCodeAt(0)&32 || type==='PLTE'),'unsupported critical PNG chunk');
    offset=end;
  }
  assert(ended && data.length,'incomplete PNG');
  const stride=256*channels,raw=inflateSync(Buffer.concat(data),{maxOutputLength:256*(stride+1)});
  assert(raw.length===256*(stride+1),'decoded PNG dimensions mismatch');
  const pixels=Buffer.alloc(256*stride);
  for(let y=0;y<256;y++) {
    const filter=raw[y*(stride+1)];assert(filter<=4,'invalid PNG filter');
    for(let x=0;x<stride;x++) {
      const a=x>=channels?pixels[y*stride+x-channels]:0,b=y?pixels[(y-1)*stride+x]:0,c=y&&x>=channels?pixels[(y-1)*stride+x-channels]:0;
      const p=a+b-c,pa=Math.abs(p-a),pb=Math.abs(p-b),pc=Math.abs(p-c);
      const prediction=[0,a,b,Math.floor((a+b)/2),pa<=pb&&pa<=pc?a:pb<=pc?b:c][filter];
      pixels[y*stride+x]=(raw[y*(stride+1)+1+x]+prediction)&255;
    }
  }
  return {width:256,height:256,pixelSha256:digest(pixels)};
}

export function publishPreview(output,receipt,entry,cell) {
  remainingCaseMs(entry.caseStartedAt);
  const caseRoot=path.join(output,...cell.split('/')),raw=json(path.join(caseRoot,'evidence.json'));
  assert(raw.caseId===entry.caseId,'preview native case mismatch');
  const directory=path.join(output,'previews',cell.replace('/','-'));mkdirSync(directory,{recursive:true});
  const images=[];
  for(const role of ['before-untrained','after-trained','disable-control']) {
    const reference=raw.raw?.images?.[role]?.artifact;
    assert(reference && typeof reference.path==='string' && !reference.path.includes('\\') && !reference.path.includes(':') && !reference.path.split('/').includes('..') && !path.isAbsolute(reference.path),'unsafe preview reference');
    const file=path.resolve(caseRoot,reference.path);
    assert(inside(caseRoot,file) && fileDigest(file)===reference.sha256,'preview bytes differ from native hash');
    const decoded=decodePreviewPng(readFileSync(file)),name=`${role}.png`;
    copyFileSync(file,path.join(directory,name),2); // COPYFILE_EXCL: never replace a published preview.
    images.push({role,file:name,decoded,reviewedArtifact:{path:`capture-${receipt.backend}/${cell}/${reference.path}`,sha256:reference.sha256}});
  }
  const index={manifestSha256:MANIFEST_SHA256,source:receipt.source,workflow:receipt.workflow,caseId:entry.caseId,caseStartedAt:entry.caseStartedAt,caseDeadlineAt:entry.caseDeadlineAt,publishedAt:new Date().toISOString(),images};
  remainingCaseMs(entry.caseStartedAt);
  store(path.join(directory,'index.json'),index);
}

export function remainingCaseMs(startedAt, now = Date.now()) {
  const start=Date.parse(startedAt);
  assert(Number.isFinite(start) && start <= now, 'invalid/future original caseStartedAt');
  const remaining=start+BOUNDS.case-now;
  assert(remaining>0,'original compute phase deadline expired');
  return remaining;
}

// CPU preparation proves only root/stdio completion. Native cases retain their
// separate authenticated descendant/RSS/release contract.
export async function executeCommand(root, args, log, limitMs, state, supervision) {
  return isPreparationCommand(root,args) ? executePreparation(root,args,log,limitMs,state,supervision)
    : executeSupervised(root,args,log,limitMs,state,supervision);
}

const commandComplete=(root,args,state)=>state && !state.unsafe&&!state.cancelled && (isPreparationCommand(root,args)
  ? state.completion?.scope==='root-and-stdio' && state.completion.completed===true
  : state.cleanup?.scope==='native-api-worker-root' && state.cleanup.released===true && state.cleanup.stdioClosed===true);

export function checkBoundedDiscovery(all, ignored) {
  for (const [label, entries] of [['unfiltered',all],['ignored',ignored]]) {
    const names=entries.filter(e=>e.kind==='test' && (label==='ignored'?e.name.startsWith('tests::moss_training_acceleration::'):e.name===BOUNDED_PRODUCT)).map(e=>e.name);
    assert(JSON.stringify(names)===JSON.stringify([BOUNDED_PRODUCT]),`${label} bounded product discovery must contain exactly its ignored FQN`);
  }
  return [BOUNDED_PRODUCT];
}

// The helper owns only a host-global UUID lease and context-free admission. The
// existing Node supervisor continues to own/authenticate the native child PID.
export async function claimBoundedDevice({root,output,env,source,deadline}) {
  mkdirSync(output,{recursive:true});
  const claimFile=path.join(output,'physical-claim.json');
  const helper=path.join(root,'scripts/ci/moss-training-acceptance/bounded_product_claim.py');
  const child=spawn('python',['-I',helper,'--inference-root',env.MOSS_INFERENCE_SOURCE_ROOT,'--output',output,'--selector',BOUNDED_PRODUCT,'--deadline',deadline],{cwd:root,env,stdio:['pipe','pipe','pipe'],windowsHide:true});
  const fd=openSync(path.join(output,'device-claim.log'),'w');
  let closed=false,exit=null,problem=null,readyResolve,readyReject;
  const ready=new Promise((resolve,reject)=>{readyResolve=resolve;readyReject=reject;});
  const complete=new Promise(resolve=>child.on('close',code=>{closed=true;exit=code;readyReject(new Error(`device claim helper exited ${code}`));closeSync(fd);resolve();}));
  child.on('error',error=>{problem=error;readyReject(error);});
  child.stdin.on('error',error=>{problem=error;readyReject(error);});
  child.stderr.on('data',bytes=>writeSync(fd,bytes));
  const lines=createInterface({input:child.stdout});
  lines.on('line',line=>{writeSync(fd,line+'\n');try {const value=JSON.parse(line);if(value.state==='admitted')readyResolve(value);} catch(error){problem=error;readyReject(error);}});
  let timer;
  try {
    const admission=await Promise.race([ready,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('device admission exceeded case deadline')),Math.max(1,Date.parse(deadline)-Date.now()));})]);
    validateBoundedClaim(admission,source);
    return {admission,claimFile,closed:()=>closed,alive:()=>!closed&&!problem,bindRoot(root){child.stdin.write(JSON.stringify({root})+'\n');},async release() {
      if(!closed)child.stdin.end(JSON.stringify({release:true})+'\n');
      await Promise.race([complete,new Promise((_,reject)=>{const timeout=setTimeout(()=>reject(new Error('device claim helper did not close')),BOUNDS.release);complete.finally(()=>clearTimeout(timeout));})]);
      assert(exit===0&&!problem,'device claim helper release failed');
      const final=json(claimFile);validateBoundedClaim(final,source);
      assert(final.state==='released'&&final.postExecutionVerified===true&&final.leaseReleased===true,'physical UUID release not verified');
      return final;
    }};
  } catch(error) {child.stdin.end();await Promise.race([complete,pause(BOUNDS.release)]);error.deviceClaimUnsafe=!closed;throw error;}
  finally {clearTimeout(timer);}
}

export function validateBoundedClaim(value,source) {
  assert(value.selector===BOUNDED_PRODUCT && value.deviceUuid===BOUNDED_GPU_UUID && value.physicalIndex===1,'bounded source selector/physical GPU claim mismatch');
  assert(value.activityProfile==='non-isolated-correctness-headroom' && value.childGpuCapBytes===BOUNDED_GPU_CAP_BYTES && value.gpuFreeReserveBytes===BOUNDED_GPU_RESERVE_BYTES,'bounded source cap/reserve/activity policy mismatch');
  assert(value.workflowSource===source.workflowSource && value.inferencePin===source.inferencePin,'device claim source mismatch');
  assert(value.visibility?.CUDA_DEVICE_ORDER==='PCI_BUS_ID'&&value.visibility?.CUDA_VISIBLE_DEVICES==='1'&&value.visibility?.NVIDIA_VISIBLE_DEVICES==='1','exact child visibility required');
  assert(value.ordinalBinding?.mapping?.length===1 && value.ordinalBinding.mapping[0].cuda_ordinal===0 && value.ordinalBinding.mapping[0].nvml_index===1 && value.ordinalBinding.mapping[0].uuid===BOUNDED_GPU_UUID,'actual CUDA ordinal-to-UUID binding mismatch');
}

export function boundedReaderBindings(stdout, weights) {
  assert(Array.isArray(weights)&&weights.length>0,'bounded actual weight inventory missing');
  const normalize=value=>{const normal=String(value).replaceAll(String.fromCharCode(92),'/').toLowerCase();return normal.startsWith('//?/')?normal.slice(4):normal;};
  const inventory=new Map();
  for(const weight of weights) {
    assert(typeof weight.canonicalPath==='string' && path.isAbsolute(weight.canonicalPath) && /^[a-f0-9]{64}$/.test(weight.sha256) && weight.expectedSha256===weight.sha256 && weight.trustedHashMatches===true,'native canonical published weight path/hash missing');
    inventory.set(normalize(weight.canonicalPath),weight);
  }
  const tokenizerInventory=[...inventory.entries()].filter(([,weight])=>normalize(weight.relativePath)==='tokenizer/tokenizer.json');
  assert(tokenizerInventory.length===1,'exact selected tokenizer/tokenizer.json inventory entry required');
  const tokenizerIdentity=tokenizerInventory[0][0];
  const bound=new Map(),tokenizers=new Map(),readers=[],otherReaders=[];
  for(const line of stdout.split(/\r?\n/)) {
    if(!line.startsWith('MODEL_INFERENCE_INPUT_USED '))continue;
    const match=line.match(/^MODEL_INFERENCE_INPUT_USED (\S+) (.+)$/);
    assert(match&&match[1]===BOUNDED_PRODUCT,'reader belongs to another selector');
    const packet=JSON.parse(match[2]);
    const tokenizer=packet.reader==='candle-gen-moss-image::text_encoder::tokenizer';
    if(!tokenizer && packet.reader!=='candle-gen-moss-image::loader::load_weights_cancellable') {
      otherReaders.push({scope:match[1],...packet,inventoryJoins:(packet.items??[]).map(item=>{const weight=inventory.get(normalize(item.absolute_lexical_path?.display));return weight?{canonicalPath:weight.canonicalPath,sha256:weight.sha256}:null;})});
      continue;
    }
    assert(packet.schema_version===1 && packet.role===(tokenizer?'tokenizer':'model-weights') && packet.complete===true && Array.isArray(packet.items)&&packet.items.length>0,'incomplete source-owned MOSS reader');
    for(const item of packet.items) {
      assert(item.provenance?.kind==='successful-reader-binding'&&item.provenance.content_integrity==='not_attested','reader grade changed');
      const actual=item.absolute_lexical_path?.display;
      const weight=inventory.get(normalize(actual));
      assert(weight,`reader path absent from native prehashed inventory: ${actual}`);
      if(tokenizer)assert(normalize(actual)===tokenizerIdentity,'tokenizer reader does not bind selected tokenizer/tokenizer.json');
      (tokenizer?tokenizers:bound).set(normalize(actual),{canonicalPath:weight.canonicalPath,sha256:weight.sha256,reader:packet.reader,readerIntegrity:'not_attested',hashProvenance:'native pre-load file SHA256, joined to actual successful source reader'});
    }
    readers.push(packet);
  }
  // HF snapshot symlinks resolve to extensionless blob targets. The selected
  // source member classifies the file; canonical identity only joins the reader.
  const expected=[...inventory.entries()].filter(([,weight])=>String(weight.relativePath??weight.path??weight.canonicalPath).toLowerCase().endsWith('.safetensors')).map(([identity])=>identity);
  assert(expected.length>0&&expected.every(value=>bound.has(value)),'not every selected safetensors file has an actual successful MOSS reader receipt');
  assert(tokenizers.has(tokenizerIdentity),'selected tokenizer/tokenizer.json has no actual successful MOSS tokenizer reader receipt');
  return {readers,otherReaders,consumedWeights:[...bound.values()],consumedTokenizers:[...tokenizers.values()]};
}

export function validateBoundedSample(sample,caseOutput) {
  assert(typeof sample?.relativePath==='string'&&!path.isAbsolute(sample.relativePath),'sole sample must name its relative native artifact');
  const file=path.resolve(caseOutput,sample.relativePath);
  assert(inside(caseOutput,file)&&/^[a-f0-9]{64}$/.test(sample.sha256),'sample path/hash invalid');
  const bytes=readFileSync(file);assert(digest(bytes)===sample.sha256,'served sample bytes changed');
  return {...sample,decoded:decodePreviewPng(bytes)};
}

async function runBoundedProduct({root,output,env,source,receipt,executable,baseEnv,command,deviceClaim=claimBoundedDevice}) {
  assert(receipt.backend==='cuda','bounded product is CUDA only');
  const caseStartedAt=new Date().toISOString(),caseDeadlineAt=new Date(Date.now()+BOUNDS.case).toISOString();
  const caseOutput=path.join(output,'base','lora');mkdirSync(caseOutput,{recursive:true});
  const entry={caseId:'cuda/base/lora',executionBackend:'cuda',test:BOUNDED_PRODUCT,caseStartedAt,caseDeadlineAt,rawEvidence:'base/lora/evidence.json'};
  receipt.cases.push(entry);receipt.recordKind='bounded-product-execution';receipt.isCampaignEvidence=false;
  const cellEnv={...env,...baseEnv,CUDA_DEVICE_ORDER:'PCI_BUS_ID',CUDA_VISIBLE_DEVICES:'1',NVIDIA_VISIBLE_DEVICES:'1',MODEL_INFERENCE_CASE_FQN:BOUNDED_PRODUCT,MOSS_ACCEPTANCE_PHASE:'bounded-product',MOSS_ACCEPTANCE_CASE_ID:entry.caseId,MOSS_ACCEPTANCE_SOURCE_CASE_ID:entry.caseId,MOSS_ACCEPTANCE_EXECUTION_BACKEND:'cuda',MOSS_ACCEPTANCE_OUTPUT:caseOutput,MOSS_ACCEPTANCE_CASE_STARTED_AT:caseStartedAt,MOSS_ACCEPTANCE_CASE_DEADLINE_AT:caseDeadlineAt,MOSS_ACCEPTANCE_RESOLVED_CAP_BYTES:String(BOUNDED_GPU_CAP_BYTES),MOSS_ACCEPTANCE_CUDA_UUID:BOUNDED_GPU_UUID};
  const lease=await deviceClaim({root,output:path.join(output,'device-claim'),env:cellEnv,source,deadline:caseDeadlineAt});
  let nativeState;
  try {
    receipt.deviceClaim=lease.admission;
    const {result,state}=await command([executable,BOUNDED_PRODUCT,'--ignored','--exact','--nocapture','--test-threads=1'],'bounded-product',remainingCaseMs(caseStartedAt),{capBytes:BOUNDED_GPU_CAP_BYTES,guardAlive:lease.alive,onSpawn:lease.bindRoot,env:{...cellEnv,MOSS_ACCEPTANCE_DEVICE_CLAIM:lease.claimFile}});
    nativeState=state;
    entry.counts=parseTestResult(result.output,result.exit,BOUNDED_PRODUCT,'green');
    const evidence=json(path.join(caseOutput,'evidence.json'));
    assert(evidence.recordKind==='bounded-product-acceptance-evidence' && evidence.isCampaignEvidence===false && evidence.acceptanceDisposition==='accepted','bounded native product evidence missing; campaign or counts cannot replace it');
    assert(evidence.selector===BOUNDED_PRODUCT && evidence.phase==='bounded-product' && evidence.caseId===entry.caseId && evidence.source?.workflowSource===source.workflowSource && evidence.source?.inferencePin===source.inferencePin && evidence.source?.runId===receipt.workflow.runId && String(evidence.source?.runAttempt)===String(receipt.workflow.attempt),'bounded native source/run/selector mismatch');
    assert(state.rootProcess?.binarySha256===receipt.executable.sha256,'launched bounded binary SHA mismatch');
    validateNativeRoot(evidence,state,BOUNDED_GPU_CAP_BYTES);
    receipt.inputConsumption=boundedReaderBindings(result.output,evidence.raw?.weightFiles);
    store(path.join(output,'input-consumption.json'),receipt.inputConsumption);
    receipt.sample=validateBoundedSample(evidence.raw?.sample,caseOutput);
    remainingCaseMs(caseStartedAt);entry.executionPassed=true;
  } finally {
    // The command wrapper retains state even when native execution throws.
    nativeState??=receipt.commands.at(-1)?.supervision;
    if(nativeState?.unsafe) {
      receipt.failures.push('native child unclosed; physical claim must not be certified released');
      // Do not send a release command while the native writer may remain alive.
      receipt.deviceClaimUnsafe=true;
    } else {
      try {
        receipt.deviceClaim=await lease.release();
        if(nativeState?.rootProcess) assert(receipt.deviceClaim.nativeProcessExited===true&&receipt.deviceClaim.nativeRoot?.pid===nativeState.rootProcess.pid&&receipt.deviceClaim.nativeRoot?.binarySha256===receipt.executable.sha256,'UUID release did not authenticate the owned native PID/binary');
      } catch(error) {receipt.deviceClaimUnsafe=lease.closed?!lease.closed():false;throw error;}
    }
  }
  remainingCaseMs(caseStartedAt);
  receipt.cellComplete=true;receipt.ok=true;
}

export async function run(options) {
  const { root,output,backend,phase,manifest,env=process.env,execute=executeCommand,inspect=defaultInspect,preflightValidator=validatePreflightChecked,sourceProvider=sourceFacts,cacheInventory=inventoryMossTrainingCache }=options;
  const cellIndex=options['capture-cell'] === undefined ? null : CELLS.indexOf(options['capture-cell']);
  assert(cellIndex === null || (phase === 'capture' && cellIndex >= 0), 'invalid capture-cell selector');
  const continuing=cellIndex !== null && cellIndex > 0;
  let receipt;
  if (continuing) {
    receipt=json(path.join(output,'driver.json'));
    assert(receipt.phase===phase && receipt.backend===backend && !receipt.ok && receipt.cellComplete===true && receipt.failures.length===0, 'prior capture not cleanly closed');
    assert(receipt.cases.length===cellIndex && receipt.cases.every((c,i)=>c.executionPassed && c.caseId===`${backend}/${CELLS[i]}` && c.test===testsFor('capture',backend)[i] && c.rawEvidence===`${CELLS[i]}/evidence.json`), 'continuation requires exact closed case prefix');
    assert(receipt.commands.length===6+2*cellIndex && receipt.commands.every(c=>c.exitCode===0 && commandComplete(root,c.command,c.supervision)), 'prior capture commands not cleanly closed');
    assert(JSON.stringify(artifactMembers(output).filter(m=>m.path!=='driver.json'))===JSON.stringify(receipt.members), 'prior capture members changed, missing or added');
  } else {
    assert(!existsSync(output) || readdirSync(output).length===0,'output must be new/empty; never carry prior campaign evidence');
    mkdirSync(output,{recursive:true});
    receipt={ phase,backend,ok:false,source:null,commands:[],cases:[],failures:[] };
  }
  const priorMembers=continuing?receipt.members:[];
  receipt.cellComplete=false;
  let active=null, interrupted=false;
  const cancel=()=>{interrupted=true;if(active)active.cancelled=true;};
  process.on('SIGINT',cancel);process.on('SIGTERM',cancel);
  const command=async(args,name,limitMs,extra={})=>{
    assert(!interrupted,'interrupted; refusing next command');
    const state={cancelled:false}; active=state;
    const entry={command:args,log:`${name}.log`,startedAt:new Date().toISOString(),boundMs:limitMs};receipt.commands.push(entry);
    try { const r=await execute(root,args,path.join(output,entry.log),limitMs,state,{env:{...env,...extra.env},capBytes:extra.capBytes??null,guardAlive:extra.guardAlive,onSpawn:extra.onSpawn});entry.exitCode=r.exit;assert(r.exit===0,`${name} exited ${r.exit}`);assert(commandComplete(root,args,state),`${name} completion/cleanup not proven`);return {result:r,state}; }
    finally {entry.finishedAt=new Date().toISOString();entry.supervision=state;active=null;}
  };
  try {
    const currentSource=sourceProvider(root,manifest,{...env,MOSS_ACCEPTANCE_PHASE:phase},inspect);
    if(continuing) assert(JSON.stringify(receipt.source)===JSON.stringify(currentSource), 'continuation source proof changed');
    receipt.source=currentSource;
    if(phase!=='finalize') {
      receipt.cacheInventory=cacheInventory({root,phase:phase==='bounded-product'?'bounded-product':phase,env});
      assert(receipt.cacheInventory.ready,'required cached MOSS snapshots/files missing before compilation: '+receipt.cacheInventory.missing.map(x=>x.path).join(', '));
    }
    const source=receipt.source;
    let peers=options['peer-artifacts'];
    if(['cross-load','finalize'].includes(phase)) {
      const selectedRoot=`${output}-selected-artifacts`;
      receipt.artifactSelection=stageLatestMossArtifacts(peers,selectedRoot,{phases:phase==='cross-load'?['capture']:['capture','cross-load'],source,runId:phase==='cross-load'?env.GITHUB_RUN_ID:env.MOSS_CAPTURE_RUN_ID});
      store(path.join(output,'selected-artifacts.json'),receipt.artifactSelection);
      if(phase==='finalize'&&!env.MOSS_INSPECTION_JSON&&existsSync(path.join(peers,'inspections.json'))) copyFileSync(path.join(peers,'inspections.json'),path.join(selectedRoot,'inspections.json'),2);
      peers=selectedRoot;
    }
    if(phase==='finalize') {
      for(const b of ['mlx','cuda']) for(const p of ['capture','cross-load']) verifyBundle(path.join(peers,`${p}-${b}`),source,p,b);
      // Only the semantic validator may call an acceptance record accepted. It must
      // inspect native receipts, loss values, tensors, artifacts and inspections.
      await command(['python3','-I',path.join(root,'scripts/ci/moss-training-acceptance/verify_evidence.py'),'--root',peers,'--manifest',manifest,'--selection',path.join(output,'selected-artifacts.json'),'--output',path.join(output,'verification.json')],'finalize',BOUNDS.list);
      const verified=json(path.join(output,'verification.json'));
      assert(verified.accepted===true && verified.acceptedCaseCount===8,'final validator did not accept all eight native evidence records');
      assert(Array.isArray(verified.records) && verified.records.length===8 && new Set(verified.records).size===8,'validator must name eight complete accepted record files');
      const seen=new Set();
      for(let i=0;i<verified.records.length;i++) {
        const name=verified.records[i];
        assert(typeof name==='string' && !path.isAbsolute(name),'accepted record must be relative to finalization output');
        const file=path.resolve(output,name);assert(inside(output,file),'accepted record escaped finalization output');
        const record=json(file);
        assert(record.recordKind==='campaign-acceptance-evidence' && record.isCampaignEvidence===true && record.acceptanceDisposition==='accepted','schema fixtures/partial records cannot be accepted');
        assert(!seen.has(record.caseId) && ['mlx','cuda'].some(b=>CELLS.some(c=>record.caseId===`${b}/${c}`)),'unexpected/duplicate accepted cell');seen.add(record.caseId);
        assert(record.manifestSha256===source.manifestSha256 && record.identity?.source?.sceneWorksSourceSha===source.workflowSource && record.identity?.source?.inferencePin===source.inferencePin,'accepted source binding mismatch');
        const start=Date.parse(record.cleanup?.caseStartedAt),end=Date.parse(record.cleanup?.caseEndedAt);
        assert(Number.isFinite(start)&&Number.isFinite(end)&&end>=start&&end-start<=BOUNDS.case,'accepted case wall deadline invalid');
        assert(Array.isArray(record.inspections)&&record.inspections.length>=3&&record.inspections.every(x=>x.reviewer&&x.findings?.length&&Date.parse(x.completedAt)>=start&&Date.parse(x.completedAt)<=Date.now()),'missing or future named preview inspection');
        await command(['python3','-I','-m','jsonschema','-i',file,path.join(root,'scripts/ci/moss-training-acceptance/evidence.schema.json')],`schema-${i}`,BOUNDS.list);
      }
      receipt.ok=true;
    } else {
      assert((backend==='mlx'&&process.platform==='darwin')||(backend==='cuda'&&process.platform==='win32')||options.testPlatform===true,'backend/platform mismatch');
      const baseEnv={RUST_MIN_STACK:String(NATIVE_TEST_STACK_BYTES),MOSS_REAL_WEIGHTS_ACCEPTANCE:'1',MOSS_ACCEPTANCE_MANIFEST:manifest,MOSS_ACCEPTANCE_WORKFLOW_SOURCE:source.workflowSource,MOSS_ACCEPTANCE_INFERENCE_PIN:source.inferencePin,HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1',HF_DATASETS_OFFLINE:'1'};
      const sourceBackend=phase==='cross-load'?(backend==='mlx'?'cuda':'mlx'):backend;
      if(phase==='cross-load') {
        for(const b of ['mlx','cuda']) {
          const capture=verifyBundle(path.join(peers,`capture-${b}`),source,'capture',b);
          assert(capture.workflow?.runId===env.GITHUB_RUN_ID,'cross-load capture belongs to another workflow run');
        }
        receipt.sourceCapture={backend:sourceBackend,driverSha256:fileDigest(path.join(peers,`capture-${sourceBackend}`,'driver.json'))};
        baseEnv.MOSS_ACCEPTANCE_PEER_ARTIFACTS=peers;
      }
      const buildStart=continuing?Date.parse(receipt.preparationStartedAt):phase==='cross-load'?Date.parse(env.MOSS_CROSS_LOAD_JOB_STARTED_AT):Date.now();
      assert(Number.isFinite(buildStart) && buildStart<=Date.now(), 'invalid original preparation clock');
      receipt.preparationStartedAt=new Date(buildStart).toISOString();
      const workflow={ref:env.GITHUB_REF,sha:env.GITHUB_SHA,fileSha256:source.workflowSha256,runId:env.GITHUB_RUN_ID,attempt:env.GITHUB_RUN_ATTEMPT,jobId:env.GITHUB_JOB};
      for(const [key,value] of Object.entries(workflow)) assert(typeof value==='string' && value.length>0,`missing actual workflow provenance: ${key}`);
      assert(/^[1-9][0-9]*$/.test(workflow.attempt),'invalid actual workflow attempt');
      workflow.attempt=Number(workflow.attempt);
      assert(Number.isSafeInteger(workflow.attempt),'workflow attempt exceeds integer range');
      if(continuing) assert(JSON.stringify(receipt.workflow)===JSON.stringify(workflow), 'continuation run/attempt/job changed');
      receipt.workflow=workflow;
      receipt.runnerName=env.RUNNER_NAME??null;
      let executable;
      if(continuing) {
        executable=receipt.executable.path;
        assert((options.executableDigest?options.executableDigest(executable):fileDigest(executable))===receipt.executable.sha256, 'continuation binary changed');
        assert(JSON.stringify(receipt.discovery)===JSON.stringify([...testsFor('capture',backend),...testsFor('cross-load',backend),PREFLIGHT].sort()), 'continuation discovery changed');
        const cpu=receipt.preModelCpu;
        const prior=receipt.commands.filter(c=>c.log==='pre-model-cpu.log');
        assert(cpu?.test===PRE_MODEL_CPU_TEST && cpu.log==='pre-model-cpu.log' && prior.length===1
          && JSON.stringify(prior[0].command)===JSON.stringify([executable,PRE_MODEL_CPU_TEST,'--exact','--nocapture','--test-threads=1']), 'continuation CPU preparation identity missing or changed');
        assert(JSON.stringify(cpu.counts)===JSON.stringify(parseTestResult(readFileSync(path.join(output,cpu.log),'utf8'),prior[0].exitCode,PRE_MODEL_CPU_TEST,'green')), 'continuation CPU preparation result changed');
        const preparedAt=Date.parse(receipt.preparationFinishedAt);
        assert(preparedAt>=buildStart && preparedAt<=Date.now() && preparedAt-buildStart<=BOUNDS.build, 'original preparation deadline exceeded');
      } else {
      const preparationRemaining=()=>{const remaining=BOUNDS.build-(Date.now()-buildStart);assert(remaining>0,'build preparation deadline exceeded');return remaining;};
      const rustc=await command(['rustc','-Vv'],'rustc-version',Math.min(BOUNDS.list,preparationRemaining()));
      const cargo=await command(['cargo','-V'],'cargo-version',Math.min(BOUNDS.list,preparationRemaining()));
      const host=rustc.result.output.match(/^host: (\S+)$/m)?.[1];
      assert(host && /^rustc \S+/m.test(rustc.result.output) && /^cargo \S+/m.test(cargo.result.output),'actual Rust/Cargo toolchain observations missing');
      receipt.toolchain={rustc:rustc.result.output.trim(),cargo:cargo.result.output.trim(),target:env.CARGO_BUILD_TARGET||host,nativeRuntime:`inference-source/${source.inferencePin}`,libtestStackBytes:NATIVE_TEST_STACK_BYTES};
      const compile=await command(['cargo','test','--locked','-p','sceneworks-rust-api','--lib',...(backend==='cuda'?['--features',phase==='bounded-product'?'backend-candle,model-inference-proof':'backend-candle']:[]),'--no-run','--message-format=json'],'compile',preparationRemaining());
      executable=selectLibTestExecutable(compile.result.output,options.statExecutable);
      receipt.executable={path:executable,sha256:options.executableDigest?options.executableDigest(executable):fileDigest(executable)};
      receipt.toolchain.binarySha256=receipt.executable.sha256;
      const all=await command([executable,'--list'],'list',Math.min(BOUNDS.list,preparationRemaining()));
      const ignored=await command([executable,'--ignored','--list'],'ignored-list',Math.min(BOUNDS.list,preparationRemaining()));
      const allTests=parseTestList(all.result.output), ignoredTests=parseTestList(ignored.result.output);
      receipt.discovery=phase==='bounded-product'?checkBoundedDiscovery(allTests,ignoredTests):checkDiscovery(allTests,ignoredTests,backend);
      if(phase!=='bounded-product') {
      assert(allTests.filter(e=>e.kind==='test'&&e.name===PRE_MODEL_CPU_TEST).length===1
        && !ignoredTests.some(e=>e.name===PRE_MODEL_CPU_TEST), 'required unignored CPU preparation test missing or ignored');
      // CI_IMPACT: one CPU-only library/dispatch/future-layout entry in the already-built
      // API binary, once per preparation. This is not native learning/case acceptance evidence.
      const cpu=await command([executable,PRE_MODEL_CPU_TEST,'--exact','--nocapture','--test-threads=1'],'pre-model-cpu',Math.min(BOUNDS.list,preparationRemaining()));
      receipt.preModelCpu={test:PRE_MODEL_CPU_TEST,log:'pre-model-cpu.log',counts:parseTestResult(cpu.result.output,cpu.result.exit,PRE_MODEL_CPU_TEST,'green')};
      }
      assert(Date.now()-buildStart<=BOUNDS.build,'build preparation deadline exceeded');
      receipt.preparationFinishedAt=new Date().toISOString();
      }
      if(phase==='bounded-product') {
        await runBoundedProduct({root,output,env,source,receipt,executable,baseEnv,command,deviceClaim:options.deviceClaim});
      } else {
      for(let i=cellIndex??0;i<(cellIndex===null?CELLS.length:cellIndex+1);i++) {
        const cell=CELLS[i], test=testsFor(phase,backend)[i];
        const caseStartedAt=new Date().toISOString();
        const caseOutput=path.join(output,...cell.split('/'));mkdirSync(caseOutput,{recursive:true});
        const entry={caseId:`${sourceBackend}/${cell}`,sourceCaseId:`${sourceBackend}/${cell}`,executionBackend:backend,test,caseStartedAt,caseDeadlineAt:new Date(Date.parse(caseStartedAt)+BOUNDS.case).toISOString(),rawEvidence:path.relative(output,path.join(caseOutput,'evidence.json')).split(path.sep).join('/')};receipt.cases.push(entry);
        const preflightFile=path.join(caseOutput,'devicepreflight.json');
        const cellEnv={...baseEnv,MOSS_ACCEPTANCE_CASE_ID:entry.caseId,MOSS_ACCEPTANCE_SOURCE_CASE_ID:entry.sourceCaseId,MOSS_ACCEPTANCE_EXECUTION_BACKEND:backend,MOSS_ACCEPTANCE_PHASE:phase,MOSS_ACCEPTANCE_CROSS_LOAD_STARTED_AT:phase==='cross-load'?caseStartedAt:'',MOSS_ACCEPTANCE_OUTPUT:caseOutput,MOSS_ACCEPTANCE_DEVICE_PREFLIGHT:preflightFile,MOSS_ACCEPTANCE_CASE_STARTED_AT:caseStartedAt,MOSS_ACCEPTANCE_CASE_DEADLINE_AT:entry.caseDeadlineAt};
        const preflight=await command([executable,PREFLIGHT,'--ignored','--exact','--nocapture','--test-threads=1'],`preflight-${cell.replace('/','-')}`,remainingCaseMs(caseStartedAt),{env:{...cellEnv,MOSS_ACCEPTANCE_PREFLIGHT_ONLY:'1'}});
        parseTestResult(preflight.result.output,preflight.result.exit,PREFLIGHT,'green');
        const capBytes=await preflightValidator(json(preflightFile),backend,source,entry.caseId,phase,()=>{assert(!interrupted,'interrupted during native weight verification');remainingCaseMs(caseStartedAt);});
        const {result,state}=await command([executable,test,'--ignored','--exact','--nocapture','--test-threads=1'],`case-${cell.replace('/','-')}`,remainingCaseMs(caseStartedAt),{capBytes,env:{...cellEnv,MOSS_ACCEPTANCE_RESOLVED_CAP_BYTES:String(capBytes)}});
        entry.counts=parseTestResult(result.output,result.exit,test,'green');
        assert(existsSync(path.join(caseOutput,'evidence.json')),'native case evidence missing (test counts are not learning evidence)');
        assert(state.rootProcess?.binarySha256===receipt.executable.sha256,'launched native binary SHA mismatch');
        validateNativeRoot(json(path.join(caseOutput,'evidence.json')),state,capBytes);
        remainingCaseMs(caseStartedAt);entry.executionPassed=true;
        if(cellIndex!==null) publishPreview(output,receipt,entry,cell);
      }
      receipt.cellComplete=true;
      receipt.ok=receipt.cases.length===CELLS.length; // Capture/cross-load execution only; NEVER an acceptance attestation.
      }
    }
  } catch(e) {receipt.failures.push(e.message);if(e.deviceClaimUnsafe)receipt.deviceClaimUnsafe=true;}
  finally {
    process.off('SIGINT',cancel);process.off('SIGTERM',cancel);
    if(interrupted){receipt.ok=false;receipt.failures.push('signal interruption');}
    // An unclosed native writer can still mutate evidence after this function returns.
    // Retain raw diagnostics, but do not claim an immutable member seal in that state.
    const released=!receipt.deviceClaimUnsafe && receipt.commands.every(c=>isPreparationCommand(root,c.command)
      ? c.supervision?.completion?.scope==='root-and-stdio' && c.supervision?.completion?.completed===true && c.supervision?.completion?.stdioClosed!==false && c.supervision?.unsafe!==true
      : c.supervision?.cleanup?.released===true && c.supervision?.unsafe!==true);
    if(!released) {
      receipt.ok=false;receipt.cellComplete=false;receipt.members=[];
      receipt.failures.push('artifact members unsealed: command release/completion unproven');
    } else {
      for(const member of priorMembers) if(!existsSync(path.join(output,member.path)) || fileDigest(path.join(output,member.path))!==member.sha256) {receipt.ok=false;receipt.cellComplete=false;receipt.failures.push('immutable prior raw file changed');}
      receipt.members=artifactMembers(output).filter(m=>m.path!=='driver.json');
    }
    store(path.join(output,'driver.json'),receipt);
  }
  return receipt;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  try {const receipt=await run(parseArgs(process.argv.slice(2)));const success=receipt.ok || (receipt.cellComplete && receipt.failures.length===0);if(!success)console.error(receipt.failures.join('\n'));process.exitCode=success?0:1;}catch(e){console.error(e.message);process.exitCode=1;}
}
