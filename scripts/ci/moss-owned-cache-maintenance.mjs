// One stopped sc-26752 recovery only. No configurable deletion paths or model execution.
import { createHash } from 'node:crypto';
import { readFile, rm, cp, mkdir, writeFile, lstat, readdir, readlink } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { win32 as path, resolve as fsResolve, relative as fsRelative } from 'node:path';
const execute = promisify(execFile);
const check = (value, message) => { if (!value) throw Error(message); };
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const normal = value => path.normalize(value).toLowerCase();
const native = '857f7a1996e3678a0773c3509ac0087522c19ce6';

// One archived failed execution, not a new maintenance mode. Product and Actions heads differ.
export const H47_OWNER = Object.freeze({
  run:'40415337753',attempt:1,candidate:'sc-26752-final-r19',
  source:'3aa58facf61fb9a2e3b6e139ce4a0a331c9f7685',actionsSource:'d41f376bdd1dfe7c36f4cfbb5fb9c402b11efc50',inference:'78100568d8e5b9fb5b199872dc57cb28f5b87340',
  workspace:'D:\\actions-runner\\_work\\SceneWorks\\SceneWorks',root:'D:\\actions-runner\\_work\\SceneWorks\\moss-sc-26752\\moss-execute-40415337753-1-cuda',
  runtimeSha:'c39c03be757a782ae5bff5093b4ed96fa25591a0b01b75f8e785bbd6f5348c63',metadataRequired:true,
});
const H47_BASE=path.dirname(H47_OWNER.root),H47_BUILD=path.join(H47_BASE,'moss-build-40415337753-1-cuda');
const H47_SOURCE=path.join(H47_BASE,'moss-execute-40349954885-1-cuda','hf');
const H47_BINARY=path.join(H47_BASE,'moss-builds-40415337753-1-execute-cuda','cuda','sceneworks-rust-api.exe');
const H47_PROCESS_ROOTS=[H47_OWNER.root,H47_BUILD,path.dirname(path.dirname(H47_BINARY))];
const H47_RECORDS=Object.freeze({'hf-transfer.json':'7ec1b9bcf39e86b69da20ce0430baadd651cce98ad4bfe359a142a2355bd1d04','cuda-package-preparation.json':'d44b6efd6f887505fbb7af3df4f588b7c410f30eb1ecc0ca0b360512e1c981fc'});
// These are identities, never kill targets. A reused PID needs positive later-start evidence.
const H47_PIDS=[35884,46352,45652],H47_LAST_OBSERVED=Date.parse('2026-10-26T23:51:33.348Z');
// Derived from original native403499/1 packages.json (SHA below), not campaign hints.
export const R19_REUSE_CAPACITY=Object.freeze({packagesSha256:'b4b2c1992a282220ddc5f51e86b2dad95a40723baf22f6324e9da60ecd2e62c3',packagesByteLength:11907,packageCount:43,projectionBytes:59241637664,adapterBlobBytes:297435928,planningReserveBytes:4*1024**3,requiredFreeBytes:63834040888});
export function reuseCapacityBudget(bytes) {
  check(bytes.length===R19_REUSE_CAPACITY.packagesByteLength&&sha(bytes)===R19_REUSE_CAPACITY.packagesSha256,'original R19 package metadata changed');
  const packages=JSON.parse(bytes),files=packages.files;
  check(files?.length===43&&new Set(files.map(item=>item.file)).size===43&&files.every(item=>Number.isSafeInteger(item.byteLength)&&item.byteLength>0),'exact43 authenticated package sizes required');
  const adapters=files.filter(item=>/^hub\/models--chinoll--Moss-Image-Edit\/snapshots\/3be9c40037e1b89b17aa85ff6f8ee805872aa558\/adapter_(config\.json|model\.safetensors)$/.test(item.file));
  check(adapters.length===2,'exact two adapter sizes required');
  const projectionBytes=files.reduce((n,item)=>n+item.byteLength,0),adapterBlobBytes=adapters.reduce((n,item)=>n+item.byteLength,0),requiredFreeBytes=projectionBytes+adapterBlobBytes+R19_REUSE_CAPACITY.planningReserveBytes;
  check(projectionBytes===R19_REUSE_CAPACITY.projectionBytes&&adapterBlobBytes===R19_REUSE_CAPACITY.adapterBlobBytes&&requiredFreeBytes===R19_REUSE_CAPACITY.requiredFreeBytes,'R19 staging arithmetic changed');
  return {...R19_REUSE_CAPACITY,projectionBytes,adapterBlobBytes,requiredFreeBytes,adapterFiles:adapters.map(item=>item.file),files};
}
// Historical metadata authenticates immutable package lengths, never a fresh native producer.
export const FRESH_PACKAGE_PROJECTION_SHA256='2128eed007c64a1d1400db0f13ce67d2971e9bd99a19578f191a7b9473127922';
export function freshCapacityBudget(files,editAdapterFiles) {
  check(Array.isArray(files)&&files.length===43&&files.every(item=>item&&Object.keys(item).sort().join(',')==='byteLength,file,sha256'&&typeof item.file==='string'&&/^[a-f0-9]{64}$/.test(item.sha256??'')&&Number.isSafeInteger(item.byteLength)&&item.byteLength>0),'fresh CUDA requires exact43 sized sealed package entries');
  const projectionSha256=sha(JSON.stringify(files.map(({file,sha256,byteLength})=>({byteLength,file,sha256}))));
  check(projectionSha256===FRESH_PACKAGE_PROJECTION_SHA256,'fresh CUDA immutable package projection changed');
  const adapters=files.filter(item=>/^hub\/models--chinoll--Moss-Image-Edit\/snapshots\/3be9c40037e1b89b17aa85ff6f8ee805872aa558\/adapter_(config\.json|model\.safetensors)$/.test(item.file));
  check(Array.isArray(editAdapterFiles)&&JSON.stringify(editAdapterFiles.slice().sort())===JSON.stringify(adapters.map(item=>item.file).sort())&&adapters.length===2,'fresh CUDA requires exact two sealed adapters');
  const projectionBytes=files.reduce((sum,item)=>sum+item.byteLength,0),adapterBlobBytes=adapters.reduce((sum,item)=>sum+item.byteLength,0),planningReserveBytes=R19_REUSE_CAPACITY.planningReserveBytes,requiredFreeBytes=projectionBytes+adapterBlobBytes+planningReserveBytes;
  check(projectionBytes===R19_REUSE_CAPACITY.projectionBytes&&adapterBlobBytes===R19_REUSE_CAPACITY.adapterBlobBytes&&requiredFreeBytes===R19_REUSE_CAPACITY.requiredFreeBytes,'fresh CUDA staging arithmetic changed');
  return {projectionSha256,packageCount:files.length,projectionBytes,adapterBlobBytes,planningReserveBytes,requiredFreeBytes};
}
export function hasReuseCapacity(observation,budget=R19_REUSE_CAPACITY) {
  check(observation?.drive==='D'&&['free','size'].every(key=>Number.isSafeInteger(observation[key])&&observation[key]>=0)&&observation.free<=observation.size,'actual D volume capacity required');
  return observation.free>=budget.requiredFreeBytes;
}
export function validateH47Processes(processes) {
  check(Array.isArray(processes)&&processes.length<=128,'bounded H47 process observation required');
  const reused=[];
  for(const item of processes) {
    check(Number.isSafeInteger(item.pid)&&item.pid>0&&typeof item.executable==='string'&&path.isAbsolute(item.executable)&&typeof item.commandLine==='string'&&item.commandLine.length>0&&item.commandLine.length<=32768&&Number.isFinite(Date.parse(item.startedAt)),'unreadable H47 process identity');
    const executable=normal(item.executable),command=item.commandLine.replaceAll('/','\\').toLowerCase();
    check(!H47_PROCESS_ROOTS.some(root=>executable===normal(root)||executable.startsWith(normal(root)+'\\')||command.includes(normal(root))),'owned H47 process is still live');
    if(H47_PIDS.includes(item.pid)){check(Date.parse(item.startedAt)>H47_LAST_OBSERVED,'H47 PID reuse is not proved');reused.push({pid:item.pid,startedAt:item.startedAt});}
  }
  return {observedProcesses:processes.length,reusedPids:reused,ownedProcessesAbsent:true};
}
export const H47_PROCESS_FILTER = ["Name='sceneworks-rust-api.exe'",...H47_PIDS.map(pid=>`ProcessId=${pid}`),...H47_PROCESS_ROOTS.flatMap(root=>[root,root.replaceAll('\\','/')].flatMap(value=>['CommandLine','ExecutablePath'].map(field=>`${field} LIKE '%${value.replaceAll('\\','\\\\')}%'`)))].join(' OR ');
export async function h47Processes() {
  // WMI scopes the observation to fixed owned roots, executable family and recorded PIDs.
  const filter=Buffer.from(H47_PROCESS_FILTER).toString('base64');
  return powershell(`$ErrorActionPreference='Stop';$filter=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${filter}'));$items=@(Get-CimInstance Win32_Process -Filter $filter | ForEach-Object {@{pid=[int]$_.ProcessId;startedAt=if($_.CreationDate){$_.CreationDate.ToUniversalTime().ToString('o')}else{$null};executable=$_.ExecutablePath;commandLine=$_.CommandLine}});ConvertTo-Json -InputObject $items -Depth 4 -Compress`);
}
export function validateH47Projection(observation,runtime,transfer) {
  const proof=validateHfObservation(observation,path.join(H47_OWNER.root,'hf'),runtime);
  check(proof.completeProjection&&runtime.packageFiles.length===43&&transfer.after.length===43,'complete H47 projection required');
  const expected=new Map(transfer.after.map(item=>[item.file,item])),extras=new Set();
  for(const entry of proof.entries){
    check(entry.type==='file'&&entry.nlink==='1','H47 HF must contain exclusive regular files');
    check(['mtimeNs','ctimeNs'].every(key=>/^[0-9]+$/.test(entry[key]??'')),'H47 authenticated timestamps required');
    const item=expected.get(entry.path);
    if(item)check(entry.sha256===item.sha256&&entry.byteLength===String(item.byteLength),'H47 projection bytes changed');
    else {
      const adapter=transfer.after.find(item=>runtime.editAdapterFiles.includes(item.file)&&item.sha256===entry.sha256&&String(item.byteLength)===entry.byteLength);
      const blobPath=/^hub\/models--chinoll--Moss-Image-Edit\/blobs\/[a-f0-9]{40}(?:[a-f0-9]{24})?$/.test(entry.path),duplicateAdapter=Boolean(adapter&&extras.has(adapter.file));
      if(!(blobPath&&adapter&&!duplicateAdapter)) {
        const error=Error('unexpected H47 HF extra file');
        // Metadata already observed above; never read an unknown file to explain a refusal.
        error.h47Extra={path:entry.path.slice(0,512).replace(/[\x00-\x1f\x7f]/g,'?'),pathTruncated:entry.path.length>512,byteLength:/^[0-9]{1,20}$/.test(entry.byteLength)?entry.byteLength:null,sha256:/^[a-f0-9]{64}$/.test(entry.sha256??'')?entry.sha256:null,validBlobPath:blobPath,adapterMatched:Boolean(adapter),duplicateAdapter};
        throw error;
      }
      extras.add(adapter.file);
    }
  }
  return proof; // Exactly43 projected files plus at most the two materialized private adapter blobs.
}
export async function h47SourceEntries(root) {
  const entries=await hfEntries(root);
  for(const entry of entries){const info=await lstat(fsResolve(root,entry.path),{bigint:true});check(JSON.stringify(entryIdentity(info))===JSON.stringify(Object.fromEntries(Object.keys(entryIdentity(info)).map(key=>[key,entry[key]]))),'protected source changed during observation');entry.mtimeNs=String(info.mtimeNs);entry.ctimeNs=String(info.ctimeNs);}
  return entries;
}
// Only the already authenticated original R19 native/MLX/stopped-HF producers.
export function validateReuseMaintenance(env,inputs) {
  const dispatch=inputs.moss_terminal?JSON.parse(inputs.moss_terminal):inputs;
  const expected={phase:'execute',candidate_id:H47_OWNER.candidate,bundle_run_id:'40349390662',bundle_artifact_id:'12663171392',bundle_manifest_sha256:'4f4adcd82b46bc9c1e57373bff835b59f05bc8846b4b71871bdf9f8f09f0daa5',native_run_id:'40349954885',native_run_attempt:'1',mlx_run_id:'40349954885',mlx_run_attempt:'1',mlx_artifact_id:'12676636780',mlx_tooling_sha:H47_OWNER.source,cuda_hf_run_id:'40349954885',cuda_hf_run_attempt:'1',cuda_hf_artifact_id:'12677781687'};
  check(env.MOSS_MAINTENANCE_PHASE==='execute'&&env.GITHUB_ACTIONS==='true'&&env.GITHUB_JOB==='windows-maintenance'&&env.MOSS_SOURCE_SHA===H47_OWNER.source&&env.MOSS_INFERENCE_SHA===H47_OWNER.inference&&env.MOSS_CANDIDATE===H47_OWNER.candidate,'native reuse maintenance requires exact original R19 pair');
  check(/^[a-f0-9]{40}$/.test(env.MOSS_TOOLING_SHA??'')&&env.MOSS_TOOLING_SHA===env.GITHUB_SHA&&env.MOSS_TOOLING_SHA!==env.MOSS_SOURCE_SHA&&dispatch.tooling_sha===env.MOSS_TOOLING_SHA,'native reuse maintenance requires actual separate tooling source');
  check((dispatch.product_sha??dispatch.source_sha)===env.MOSS_SOURCE_SHA&&Object.entries(expected).every(([key,value])=>dispatch[key]===value),'native reuse maintenance producer/bundle identity changed');
  for(const key of ['NATIVE_RUN_ID','NATIVE_RUN_ATTEMPT','MLX_RUN_ID','MLX_RUN_ATTEMPT','MLX_ARTIFACT_ID','CUDA_HF_RUN_ID','CUDA_HF_RUN_ATTEMPT','CUDA_HF_ARTIFACT_ID'])check(env[`MOSS_${key}`]===expected[key.toLowerCase()],'native reuse maintenance environment differs from dispatch');
  if(inputs.moss_terminal){
    check(Object.keys(dispatch).sort().join(',')===[...Object.keys(expected),'product_sha','tooling_sha'].sort().join(','),'closed native reuse maintenance dispatch required');
    check(inputs.ref===env.GITHUB_SHA&&inputs.inference_sha===env.MOSS_INFERENCE_SHA,'native reuse maintenance outer source changed');
  }
  return {mode:'original-r19-native-reuse',productSha:env.MOSS_SOURCE_SHA,inferenceSha:env.MOSS_INFERENCE_SHA,toolingSha:env.MOSS_TOOLING_SHA,nativeRunId:expected.native_run_id,mlxArtifactId:expected.mlx_artifact_id,stoppedHfArtifactId:expected.cuda_hf_artifact_id};
}
async function reuseInvocation(env,io) {
  const bytes=await io.read(env.GITHUB_EVENT_PATH);check(bytes.length<=65536,'bounded maintenance dispatch required');
  return validateReuseMaintenance(env,JSON.parse(bytes).inputs??{});
}
async function validateH47Invocation(env,io) {
  if(env.MOSS_MAINTENANCE_PHASE==='execute'&&env.MOSS_NATIVE_RUN_ID)return reuseInvocation(env,io);
  check(env.GITHUB_ACTIONS==='true'&&env.GITHUB_JOB==='windows-maintenance'&&env.MOSS_SOURCE_SHA===env.GITHUB_SHA&&env.MOSS_NATIVE_RUN_ID==='','H47 cleanup requires normal new-pair maintenance');
  const event=JSON.parse(await io.read(env.GITHUB_EVENT_PATH)),inputs=event.inputs??{},dispatch=inputs.moss_terminal?JSON.parse(inputs.moss_terminal):inputs;
  check(dispatch.phase==='execute'&&!(dispatch.native_run_id),'H47 cleanup requires ordinary fresh execute');
}
async function validateFreshInvocation(env,io) {
  check(env.GITHUB_ACTIONS==='true'&&env.GITHUB_JOB==='windows-maintenance'&&env.MOSS_SOURCE_SHA===env.GITHUB_SHA&&env.MOSS_NATIVE_RUN_ID==='','fresh maintenance requires actual same-source Actions job without reuse');
  const bytes=await io.read(env.GITHUB_EVENT_PATH);check(bytes.length<=65536,'bounded fresh maintenance dispatch required');
  const inputs=JSON.parse(bytes).inputs??{},dispatch=inputs.moss_terminal?JSON.parse(inputs.moss_terminal):inputs;
  const prior=['tooling_sha','product_sha','native_run_id','native_run_attempt','execution_run_id','review_run_id','review_artifact_id','mlx_run_id','mlx_run_attempt','mlx_artifact_id','mlx_tooling_sha','cuda_hf_run_id','cuda_hf_run_attempt','cuda_hf_artifact_id','original_inventory_path','original_inventory_sha256','retry_inventory_path','retry_inventory_sha256','retry_run_id','retry_artifact_id','mlx_candidate_id','mlx_source_sha','mlx_inference_sha'];
  check(dispatch.phase==='execute'&&prior.every(key=>!dispatch[key])&&(!dispatch.execution_run_attempt||dispatch.execution_run_attempt==='1')&&dispatch.candidate_id===env.MOSS_CANDIDATE&&dispatch.bundle_manifest_sha256===env.MOSS_BUNDLE_SHA&&/^[a-f0-9]{64}$/.test(dispatch.bundle_manifest_sha256??'')&&[dispatch.bundle_run_id,dispatch.bundle_artifact_id].every(id=>/^[1-9][0-9]*$/.test(id??'')),'fresh maintenance requires ordinary sealed execute without prior evidence');
  check(!env.MOSS_TOOLING_SHA&&['NATIVE_RUN_ATTEMPT','EXECUTION_RUN_ID','MLX_RUN_ID','MLX_ARTIFACT_ID','CUDA_HF_RUN_ID','CUDA_HF_ARTIFACT_ID'].every(key=>!env[`MOSS_${key}`]),'fresh maintenance environment contains prior evidence');
  check((inputs.moss_terminal?inputs.ref:inputs.source_sha)===env.GITHUB_SHA&&inputs.inference_sha===env.MOSS_INFERENCE_SHA,'fresh maintenance dispatch source changed');
}

export async function observeH47Owner(io,runtime,metadata) {
  const runtimePath=path.join(H47_OWNER.root,'runtime.json');validateTree(await io.inspect(runtimePath),runtimePath);
  check(JSON.stringify(validateRuntime(H47_OWNER,await io.read(runtimePath)))===JSON.stringify(runtime),'H47 runtime changed during observation');
  const records={};
  for(const [file,expected] of Object.entries(H47_RECORDS)){const recordPath=path.join(H47_OWNER.root,file);validateTree(await io.inspect(recordPath),recordPath);const bytes=await io.read(recordPath);check(sha(bytes)===expected,'H47 archived ownership record changed');records[file]=JSON.parse(bytes);}
  const transfer=records['hf-transfer.json'],preparation=records['cuda-package-preparation.json'];
  check(normal(runtime.root)===normal(H47_OWNER.root)&&normal(runtime.hfHome)===normal(path.join(H47_OWNER.root,'hf'))&&normal(transfer.from)===normal(path.join(H47_BUILD,'hf'))&&normal(transfer.to)===normal(runtime.hfHome)&&normal(preparation.readOnlySource)===normal(H47_SOURCE),'H47 ownership/source confinement changed');
  check(transfer.operation==='same-volume-private-HF-ownership-transfer'&&preparation.mode==='rehydrated-exact-packages'&&preparation.originalRunId==='40349954885'&&preparation.originalAttempt==='1'&&JSON.stringify(transfer.before)===JSON.stringify(transfer.after),'H47 private transfer changed');
  validateTree(await io.inspect(path.join(H47_BUILD,'hf-owner.json')),path.join(H47_BUILD,'hf-owner.json'));
  check(await io.hash(path.join(H47_BUILD,'hf-owner.json'))===transfer.originalOwner.sha256,'H47 original private build owner changed');
  validateH47Projection(metadata,runtime,transfer);
  validateTree(await io.inspect(H47_SOURCE),H47_SOURCE);
  const sourceEntries=await io.entries(H47_SOURCE);
  check(Array.isArray(sourceEntries)&&sourceEntries.length>=43&&sourceEntries.length<=10000&&sourceEntries.every(item=>item.type==='file'&&item.nlink==='1'&&/^[0-9]+$/.test(item.mtimeNs??'')&&/^[0-9]+$/.test(item.ctimeNs??'')),'protected original source ownership changed');
  const processes=validateH47Processes(await io.processes());
  return {records:H47_RECORDS,protectedSource:H47_SOURCE,protectedSourceMetadataSha256:sha(JSON.stringify(sourceEntries)),processes};
}

export const OWNED_RUNS = Object.freeze([
  { run: '40145054879', candidate: 'sc-26752-final-r4', source: '37d5ad93f858025d3202a2ddcab465c67174e0c7', inference: '878f26aba50bfb6c371437da9ca7ead81052c25a', workspace: 'D:\\actions-runner\\_work\\SceneWorks\\SceneWorks', runtimeSha: '1b8948de0a9f7ede92019896359683605fe2c4e7577971520da881dda6aca710' },
  { run: '40152001538', candidate: 'sc-26752-final-r5', source: '9ac78275e2f5f536541c1790fc840232fd5bdfdd', inference: native, workspace: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', runtimeSha: '7ac52e00e6e712012b907a740e96e5c9d39c72483499205f637916e286e70b43' },
  { run: '40155160639', candidate: 'sc-26752-final-r6', source: '2490b2eed9bf986967c389ac69171d03b04835d7', inference: native, workspace: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', runtimeSha: 'cfb0491496d4e29aafc466404526592fe2d2950a12f1566c74fcab9e728f007e' },
  { run: '40163530825', candidate: 'sc-26752-final-r10', source: 'c17fd74650f2464427b30cea7669c7a284ea4a73', inference: native, workspace: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', root: 'D:\\actions-runner-2\\_work\\SceneWorks\\moss-sc-26752\\moss-build-40163530825-1-cuda', runtimeSha: '546808654e2a0d44d26a5d594b9e7c20961b2b9d947e900effdefdf9f8c2d625', cacheBinarySha256: '93dbe1db05a946517484539199b69927ebc4eb9bae3bab508ba2636529b81b97', metadataRequired: true },
  { run: '40167292776', candidate: 'sc-26752-final-r12', source: 'e6475176bf1cf07a46530adbe659aa0d6492c71a', inference: native, workspace: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', root: 'D:\\actions-runner-2\\_work\\SceneWorks\\moss-sc-26752\\moss-build-40167292776-1-cuda', runtimeSha: '47a25951ea0e1d6fd076802c2ca68421e36d90a296fa1813fc344e6d0a461355', cacheBinarySha256: '07cb4ef31caba2f875d6750c461715f880c79288201af88af11997e31809d69d', metadataRequired: true },
  { run: '40169818205', candidate: 'sc-26752-final-r13', source: 'ba6bdd824823f01f8c4dc435f9fad50f45f97893', inference: native, workspace: 'D:\\actions-runner\\_work\\SceneWorks\\SceneWorks', root: 'D:\\actions-runner\\_work\\SceneWorks\\moss-sc-26752\\moss-build-40169818205-1-cuda', runtimeSha: '76554eb462c580cd6afa7b5102aaeaa434de85049fe1b6701fe7ba2347f6d61f', cacheBinarySha256: '09088a12b26968177adf5a6ead76b088be2f791e5756224497db4fb7d20cf8fd', metadataRequired: true },
  { run: '40174802721', candidate: 'sc-26752-final-r14', source: '12e1b364360b43b28e864879be4c68bb28a696bc', inference: native, workspace: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', root: 'D:\\actions-runner-2\\_work\\SceneWorks\\moss-sc-26752\\moss-build-40174802721-1-cuda', runtimeSha: '4082f9c28ae31aee7dda8e850d04b197d7741a01edbc3856aa53ddac6ba8a532', cacheBinarySha256: 'ab36062090356912f9131654fa5c3ec87ed808f5e0fc0cbd32e388773edb75c2', metadataRequired: true },
  { run: '40181253824', candidate: 'sc-26752-final-r15', source: 'a0692191b09813fab859659bf382bd7c91a60140', inference: native, workspace: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', root: 'D:\\actions-runner-2\\_work\\SceneWorks\\moss-sc-26752\\moss-build-40181253824-1-cuda', runtimeSha: 'a4c17012e7143f8775809decbd72216a7c3114209bf91c3f7f0d9608f0585cc4', metadataRequired: true },
  H47_OWNER,
].map(item => Object.freeze({ ...item, attempt: 1, root: item.root ?? path.join(item.workspace, `moss-build-${item.run}-1-cuda`) })));
// Bigint identities are serialized as decimal strings; never round Windows file IDs.
export function entryIdentity(info) {
  return {type:info.isSymbolicLink()?'symlink':info.isDirectory()?'directory':info.isFile()?'file':'other',dev:String(info.dev),ino:String(info.ino),nlink:String(info.nlink),byteLength:String(info.size)};
}
export async function hfEntries(root,{timestamps=false}={}) {
  if(process.platform==='win32')await inspectTree(root); // No Windows reparse traversal, including non-symlink tags.
  const rootInfo=await lstat(root,{bigint:true});check(rootInfo.isDirectory()&&!rootInfo.isSymbolicLink(),'HF metadata root must be a real directory');
  const entries=[];let visited=0;
  const visit=async dir=>{for(const name of (await readdir(dir)).sort()) {
    check(++visited<=10000,'HF metadata inventory bound exceeded');const file=fsResolve(dir,name),info=await lstat(file,{bigint:true}),entry={path:fsRelative(root,file).replaceAll('\\','/'),...entryIdentity(info),...(timestamps?{mtimeNs:String(info.mtimeNs),ctimeNs:String(info.ctimeNs)}:{})};
    if(info.isDirectory())await visit(file);else {if(info.isSymbolicLink())entry.link=await readlink(file);entries.push(entry);}
  }};await visit(root);return entries;
}
export async function hashMetadataFile(file,expected) {
  const identity=info=>({...entryIdentity(info),...(expected.mtimeNs!==undefined?{mtimeNs:String(info.mtimeNs),ctimeNs:String(info.ctimeNs)}:{})});
  const before=identity(await lstat(file,{bigint:true}));check(before.type==='file'&&JSON.stringify(before)===JSON.stringify(Object.fromEntries(Object.keys(before).map(key=>[key,expected[key]]))),'HF metadata file changed before hash');
  const hash=createHash('sha256');for await(const chunk of createReadStream(file))hash.update(chunk);
  check(JSON.stringify(identity(await lstat(file,{bigint:true})))===JSON.stringify(before),'HF metadata file changed during hash');return hash.digest('hex');
}
export async function observeHf(runtime) {
  const entries=await hfEntries(runtime.hfHome,{timestamps:normal(runtime.hfHome)===normal(path.join(H47_OWNER.root,'hf'))}),adapters=new Set(runtime.editAdapterFiles),packages=new Set(runtime.packageFiles?.map(item=>item.file)??[]),blobPrefix='hub/models--chinoll--Moss-Image-Edit/blobs/';
  for(const entry of entries)if(entry.type==='file'&&(entry.nlink!=='1'||packages.has(entry.path)||adapters.has(entry.path)||entry.path.startsWith(blobPrefix)))entry.sha256=await hashMetadataFile(fsResolve(runtime.hfHome,entry.path),entry);
  return {root:runtime.hfHome,entries};
}
export function validateHfObservation(observation,hf,runtime) {
  check(normal(observation.root??'')===normal(hf)&&Array.isArray(observation.entries)&&observation.entries.length<=10000,'owned HF metadata escaped fixed root');
  for(const entry of observation.entries) {
    check(typeof entry.path==='string'&&!path.isAbsolute(entry.path)&&!entry.path.split(/[\\/]/).some(part=>part==='..'||part==='.')&&!entry.path.includes(':')&&normal(path.join(hf,entry.path)).startsWith(normal(hf)+'\\'),'HF metadata path escaped');
    check(['file','symlink','other'].includes(entry.type)&&['dev','ino','nlink','byteLength'].every(key=>/^[0-9]+$/.test(entry[key]??'')),'invalid HF metadata identity');
    const hashRequired=entry.type==='file'&&(entry.nlink!=='1'||/^hub\/models--chinoll--Moss-Image-Edit\/(?:blobs\/|snapshots\/[a-f0-9]{40}\/adapter_(?:config\.json|model\.safetensors)$)/.test(entry.path));
    check((!hashRequired&&!entry.sha256)||/^[a-f0-9]{64}$/.test(entry.sha256??''),'missing/invalid HF metadata hash');
  }
  const entries=new Map(observation.entries.map(entry=>[entry.path,entry]));check(entries.size===observation.entries.length,'duplicate HF metadata path');
  const completeProjection=runtime?.packageFiles?.length===43&&runtime.packageFiles.every(item=>entries.get(item.file)?.type==='file'&&entries.get(item.file)?.sha256===item.sha256);
  const projectedBytes=completeProjection?runtime.packageFiles.reduce((total,item)=>total+BigInt(entries.get(item.file).byteLength),0n):0n;check(projectedBytes<=BigInt(Number.MAX_SAFE_INTEGER),'HF projection byte count exceeds exact range');
  return {...observation,completeProjection,projectedBytes:Number(projectedBytes)};
}
// Observation only: never part of OWNED_RUNS and never a deletion/cache-copy candidate.
export const R9_DIAGNOSTIC = Object.freeze({run:'40161906148',attempt:1,candidate:'sc-26752-final-r9',source:'97ab1f826528d62df99a645af69688d295fb1b53',inference:native,workspace:'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks',root:'D:\\actions-runner-2\\_work\\SceneWorks\\moss-sc-26752\\moss-build-40161906148-1-cuda'});
export async function diagnoseR9(io) {
  const binding=R9_DIAGNOSTIC,stop=await io.stopped(binding),hf=path.join(binding.root,'hf'),file=path.join(binding.root,'runtime.json');
  const inventory=validateTree(await io.inspect(hf),hf);validateTree(await io.inspect(file),file);
  let bytes;try{bytes=await io.read(file);}catch(error){if(error.code==='ENOENT'){check(!inventory.exists,'R9 HF exists without runtime ownership proof');return {readOnly:true,stop,inventory,runtime:null};}throw error;}
  check(bytes.length<=131072,'unexpected R9 runtime size');const runtime=JSON.parse(bytes);
  check(runtime.candidateId===binding.candidate&&runtime.sourceSha===binding.source&&runtime.inferenceSha===binding.inference&&runtime.backend==='cuda','R9 observed runtime identity mismatch');
  for(const [field,expected] of Object.entries({root:binding.root,hfHome:hf,sourceRoot:path.join(binding.workspace,'source'),packageSource:'E:\\sceneworks-evaluation\\moss\\sc-26322-reference-v9'}))check(normal(runtime[field]??'')===normal(expected),`R9 runtime ${field} mismatch`);
  return {readOnly:true,stop,inventory,runtime:{sha256:sha(bytes),byteLength:bytes.length,utf8:bytes.toString('utf8')}};
}
export function destination(env) {
  check(env.GITHUB_REPOSITORY === 'SceneWorks/SceneWorks', 'wrong repository');
  check(/^\d+$/.test(env.GITHUB_RUN_ID ?? '') && /^\d+$/.test(env.GITHUB_RUN_ATTEMPT ?? ''), 'missing current Actions identity');
  check(!OWNED_RUNS.some(item => item.run === env.GITHUB_RUN_ID), 'cannot reclaim the executing run');
  check(/^[a-f0-9]{40}$/.test(env.MOSS_SOURCE_SHA ?? '') && /^[a-f0-9]{40}$/.test(env.MOSS_INFERENCE_SHA ?? ''), 'missing current source identity');
  check(/^D:\\actions-runner(?:-[234])?\\_work\\SceneWorks\\SceneWorks$/i.test(path.normalize(env.GITHUB_WORKSPACE ?? '')), 'unexpected Windows workspace');
  const root = path.join(env.GITHUB_WORKSPACE, '..', 'moss-sc-26752', `moss-build-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}-cuda`);
  check(normal(env.MOSS_ROOT ?? '') === normal(root), 'current run root mismatch');
  return root;
}
export function validateStopped(binding, latest, attempt, jobs) {
  for (const run of [latest, attempt]) check(String(run.id) === binding.run && run.run_attempt === 1 && run.head_sha === (binding.run===H47_OWNER.run?H47_OWNER.actionsSource:binding.source) && run.status === 'completed' && run.conclusion, 'old run identity/attempt is not stopped');
  check(jobs.length > 0 && jobs.every(job => job.status === 'completed' && job.conclusion && Number(job.run_id) === Number(binding.run)), 'old run still has nonterminal jobs');
}
export function validateRuntime(binding, bytes) {
  check(sha(bytes) === binding.runtimeSha, 'old runtime bytes changed');
  const runtime = JSON.parse(bytes);
  check(runtime.candidateId === binding.candidate && runtime.sourceSha === binding.source && runtime.inferenceSha === binding.inference && runtime.backend === 'cuda', 'old runtime identity mismatch');
  for (const [field, expected] of Object.entries({ root: binding.root, hfHome: path.join(binding.root, 'hf'), sourceRoot: path.join(binding.workspace, 'source'), packageSource: binding.run===H47_OWNER.run?path.join(H47_BASE,'moss-execute-40331712229-1-cuda','hf'):'E:\\sceneworks-evaluation\\moss\\sc-26322-reference-v9' })) check(normal(runtime[field] ?? '') === normal(expected), `old runtime ${field} mismatch`);
  return runtime;
}
export function validateTree(snapshot, expected) {
  check(normal(snapshot.root ?? '') === normal(expected) && normal(path.parse(expected).root) === 'd:\\', 'tree escaped exact D: path');
  check(typeof snapshot.exists === 'boolean', 'missing existence observation');
  check(snapshot.safe === true && snapshot.reparsePoints === 0, 'reparse/ancestry confinement failed');
  check(Number.isSafeInteger(snapshot.bytes) && snapshot.bytes >= 0 && Number.isSafeInteger(snapshot.fileCount) && snapshot.fileCount >= 0, 'invalid byte inventory');
  return snapshot;
}
// PowerShell is used only for Windows file attributes and volume telemetry. Traversal refuses
// every reparse tag BEFORE descending, including junctions and non-symlink reparse points.
export async function powershell(script) {
  const encoded = Buffer.from(script, 'utf16le').toString('base64');
  const { stdout } = await execute('powershell.exe', ['-NoProfile', '-NonInteractive', '-EncodedCommand', encoded], { maxBuffer: 16 * 1024 * 1024, windowsHide: true });
  return JSON.parse(stdout.replace(/^\uFEFF/, '').trim());
}
export async function inspectTree(root) {
  const encoded = Buffer.from(root).toString('base64');
  return powershell(`$ErrorActionPreference='Stop';[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$p=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${encoded}'))
$cursor=[IO.Path]::GetFullPath($p)
while($cursor){if([IO.File]::Exists($cursor) -or [IO.Directory]::Exists($cursor)){if(([IO.File]::GetAttributes($cursor) -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw 'Reparse ancestor'}};$next=[IO.Path]::GetDirectoryName($cursor);if($next -eq $cursor){break};$cursor=$next}
$exists=[IO.Directory]::Exists($p);$bytes=[long]0;$count=0;$visited=0;$files=@();$stack=New-Object 'Collections.Generic.Stack[string]';if($exists){$stack.Push($p)}
while($stack.Count){$dir=$stack.Pop();foreach($entry in [IO.Directory]::EnumerateFileSystemEntries($dir)){$visited++;if($visited -gt 250000){throw 'Inventory bound exceeded'};$a=[IO.File]::GetAttributes($entry);if(($a -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw 'Reparse descendant'};if(($a -band [IO.FileAttributes]::Directory) -ne 0){$stack.Push($entry)}else{$size=([IO.FileInfo]::new($entry)).Length;$bytes+=$size;$count++;if($p.EndsWith('\\hf')){$files+=@{path=$entry.Substring($p.Length+1);bytes=$size};if($files.Count -gt 10000){throw 'HF inventory bound exceeded'}}}}}
@{root=$p;exists=$exists;safe=$true;reparsePoints=0;bytes=$bytes;fileCount=$count;files=@($files)}|ConvertTo-Json -Depth 5 -Compress`);
}
export async function volume() {
  return powershell("$ErrorActionPreference='Stop';$v=Get-Volume -DriveLetter D;@{drive='D';size=[long]$v.Size;free=[long]$v.SizeRemaining}|ConvertTo-Json -Compress");
}
export async function actionsApi(token, route) {
  check(token, 'Actions read token unavailable');
  const response = await fetch(`https://api.github.com/repos/SceneWorks/SceneWorks/${route}`, { headers: { Authorization: `Bearer ${token}`, Accept: 'application/vnd.github+json' }, signal: AbortSignal.timeout(30000) });
  check(response.ok, `Actions identity query failed HTTP${response.status}`);
  return response.json();
}
export async function stopped(binding, api) {
  const latest = await api(`actions/runs/${binding.run}`), attempt = await api(`actions/runs/${binding.run}/attempts/1`);
  const jobs = [];
  for (let page = 1; page <= 10; page++) {
    const result = await api(`actions/runs/${binding.run}/attempts/1/jobs?per_page=100&page=${page}`);
    check(Array.isArray(result.jobs) && Number.isInteger(result.total_count), 'incomplete Actions jobs response');
    jobs.push(...result.jobs);
    if (jobs.length === result.total_count) { validateStopped(binding, latest, attempt, jobs); return { run: binding.run, attempt: 1, head: latest.head_sha, status: latest.status, jobs: jobs.map(({ id, status, conclusion }) => ({ id, status, conclusion })) }; }
    check(result.jobs.length === 100, 'truncated Actions jobs response');
  }
  throw Error('Actions jobs page bound exceeded');
}
// The Node24 cp implementation creates an absent directory with exclusive mkdir.
// Do not pre-create it: errorOnExist deliberately refuses an existing directory.
export async function copyCacheExclusive(from,to) {
  let existing;try{existing=await lstat(to);}catch(error){if(error.code!=='ENOENT')throw error;}
  check(!existing,'cache destination already exists');
  await cp(from,to,{recursive:true,dereference:false,force:false,errorOnExist:true});
}
export async function maintain(env, io, {copyCache=true}={}) {
  const root = destination(env), target = path.join(root, 'target');
  if(copyCache&&env.MOSS_MAINTENANCE_PHASE==='execute'&&!env.MOSS_NATIVE_RUN_ID) {
    await validateFreshInvocation(env,io);
    const receipt={version:1,kind:'sc-26752-fresh-maintenance-noop',currentRun:env.GITHUB_RUN_ID,currentSource:env.MOSS_SOURCE_SHA,currentInference:env.MOSS_INFERENCE_SHA,root,operations:[],cleanup:'not-requested-old-caches-retained',cache:'not-copied',capacity:'DEFERRED-to-fresh-CUDA-build-before-first-HF-copy'};
    io.report({...receipt,stage:'fresh-noop'});return receipt; // No volume, old tree, process, prune or copy.
  }
  const reuse=env.MOSS_NATIVE_RUN_ID&&env.MOSS_MAINTENANCE_PHASE!=='cuda-rebase'?await reuseInvocation(env,io):null;
  check(!reuse||copyCache,'native reuse cannot use reference-only maintenance');
  const receipt = { version: 1, kind: 'sc-26752-stopped-run-disk-recovery', currentRun: env.GITHUB_RUN_ID, currentSource: env.MOSS_SOURCE_SHA, ...(reuse?{reuse}:{}), before: await io.volume(), candidates: [], operations: [] };
  if(reuse&&hasReuseCapacity(receipt.before)) {
    receipt.capacity={...R19_REUSE_CAPACITY,mode:'capacity-only-no-pruning',planningReserveMeaning:'Planning reserve only; concurrent disk use can still cause ENOSPC'};
    receipt.cache={state:'capacity-only-original-targets-retained-no-copy'};receipt.after=receipt.before;
    io.report({...receipt,stage:'completed'});return receipt; // No old tree, process, copy or mutation.
  }
  validateTree(await io.inspect(root), root);
  receipt.r9Observation = await diagnoseR9(io);
  for (const binding of OWNED_RUNS.filter(item=>copyCache||item.run!==H47_OWNER.run)) {
    const stop = await io.stopped(binding), hf = path.join(binding.root, 'hf');
    const inventory = validateTree(await io.inspect(hf), hf);
    let cache = null, cacheProbeError = null;
    if (binding.run === '40155160639' || binding.run === '40163530825' || binding.run === '40167292776' || binding.run === '40169818205' || binding.run === '40174802721') {
      try { cache = validateTree(await io.inspect(path.join(binding.root, 'target')), path.join(binding.root, 'target')); }
      catch (error) { cacheProbeError = String(error.message).slice(0, 200); } // Optional seed stays untouched.
    }
    const runtime = inventory.exists || cache?.exists ? validateRuntime(binding, await io.read(path.join(binding.root, 'runtime.json'))) : null;
    const metadata = binding.metadataRequired && inventory.exists ? validateHfObservation(await io.metadata(runtime), hf, runtime) : null;
    if(metadata?.completeProjection)check(metadata.projectedBytes<=inventory.bytes,'HF projection exceeds measured inventory');
    if(binding.run===H47_OWNER.run&&inventory.exists)await validateH47Invocation(env,io);
    let owner=null;
    if(binding.run===H47_OWNER.run&&inventory.exists)try {owner=await observeH47Owner(io,runtime,metadata);}catch(error) {
      if(error.h47Extra)io.report({stage:'h47-extra-file-refusal',readOnly:true,run:binding.run,entry:error.h47Extra,volumeBefore:{drive:'D',...Object.fromEntries(['free','size'].map(key=>[key,Number.isSafeInteger(receipt.before?.[key])&&receipt.before[key]>=0?receipt.before[key]:null]))}});
      throw error;
    }
    receipt.candidates.push({ ...binding, stop, inventory, metadata, cache, cacheProbeError, ...(owner?{owner}:{}) });
  }
  // The full disk may prevent any local pre-write. Emit the bounded before inventory into the
  // retained Actions log first; the same receipt is saved outside HF once space is reclaimed.
  io.report({ ...receipt, stage: 'validated-before-mutation' });
  for (const candidate of receipt.candidates) {
    if (!candidate.inventory.exists) continue;
    await io.stopped(candidate);
    const runtime = validateRuntime(candidate, await io.read(path.join(candidate.root, 'runtime.json')));
    const leaf = path.join(candidate.root, 'hf');
    validateTree(await io.inspect(leaf), leaf);
    let removalMetadata;
    if(candidate.metadataRequired) {
      const metadata = validateHfObservation(await io.metadata(runtime), leaf, runtime);
      check(JSON.stringify(metadata)===JSON.stringify(candidate.metadata),'owned HF metadata changed before cleanup');removalMetadata=metadata;
      io.report({stage:candidate.run===H47_OWNER.run?'fixed-h47-readonly-metadata-before-hf-cleanup':candidate.run==='40181253824'?'fixed-r15-readonly-metadata-before-hf-cleanup':candidate.run==='40174802721'?'fixed-r14-readonly-metadata-before-hf-cleanup':candidate.run==='40169818205'?'fixed-r13-readonly-metadata-before-hf-cleanup':candidate.run==='40167292776'?'fixed-r12-readonly-metadata-before-hf-cleanup':'fixed-r10-readonly-metadata-before-hf-cleanup',run:candidate.run,source:candidate.source,runtimeSha256:candidate.runtimeSha,metadata});
    }
    if(candidate.run===H47_OWNER.run) {
      const owner=await observeH47Owner(io,runtime,removalMetadata);
      check(owner.protectedSourceMetadataSha256===candidate.owner.protectedSourceMetadataSha256,'protected original HF changed before removal');
      await io.stopped(candidate);validateTree(await io.inspect(leaf),leaf);
      const identity=entries=>entries.map(item=>Object.fromEntries(['path','type','dev','ino','nlink','byteLength','mtimeNs','ctimeNs'].map(key=>[key,item[key]])));
      check(JSON.stringify(identity(await io.entries(leaf)))===JSON.stringify(identity(removalMetadata.entries)),'H47 HF identity changed at removal boundary');
      validateH47Processes(await io.processes()); // Last live check: never stop or kill anything.
    }
    await io.remove(leaf); // Exact fixed HF leaf only; no root/cache/model-source wildcard.
    receipt.operations.push({ kind: 'remove-owned-hf', path: leaf, after: await io.volume() });
    io.report({ stage: 'hf-reclaimed', operation: receipt.operations.at(-1) });
  }
  if(copyCache&&!reuse) {
  const original = receipt.candidates.find(item => item.run === '40174802721' && item.cache?.exists) ?? receipt.candidates.find(item => item.run === '40169818205' && item.cache?.exists) ?? receipt.candidates.find(item => item.run === '40167292776' && item.cache?.exists) ?? receipt.candidates.find(item => item.run === '40163530825' && item.cache?.exists) ?? receipt.candidates.find(item => item.run === '40155160639'), from = path.join(original.root, 'target');
  if (original.cache?.exists) {
    await io.stopped(original);
    validateRuntime(original, await io.read(path.join(original.root, 'runtime.json')));
    const cacheInventory = validateTree(await io.inspect(from), from);
    const dest = validateTree(await io.inspect(target), target);
    const freeBeforeCopy = await io.volume();
    // Completeness is a property of the projection, never of which target cache was selected.
    const completeProjections = receipt.candidates.filter(item => item.inventory.exists && (['40145054879', '40152001538'].includes(item.run) || (['40163530825', '40167292776', '40169818205', '40174802721', '40181253824'].includes(item.run) && item.metadata?.completeProjection === true)));
    const projectionReserveBytes = Math.max(0, ...completeProjections.map(item => item.inventory.bytes));
    const rebuildReserveBytes = 4 * 1024 ** 3;
    const requiredFreeBytes = cacheInventory.bytes + projectionReserveBytes + rebuildReserveBytes;
    const budget = { cacheBytes: cacheInventory.bytes, freeBeforeCopy, projectionReserveBytes, rebuildReserveBytes, requiredFreeBytes };
    if (dest.exists) receipt.cache = { state: 'destination-exists-original-cache-retained', from, to: target, ...budget };
    else if (!completeProjections.length || !Number.isSafeInteger(freeBeforeCopy.free) || freeBeforeCopy.free < requiredFreeBytes) {
      receipt.cache = { state: 'insufficient-proved-copy-headroom-original-cache-retained', from, to: target, ...budget };
    } else {
      const binaryHash = file => io.hash ? io.hash(file) : io.read(file).then(sha);
      const originalBinarySha256 = await binaryHash(path.join(from, 'release', 'sceneworks-rust-api.exe'));
      if(original.cacheBinarySha256)check(originalBinarySha256===original.cacheBinarySha256,'fixed source cache binary differs from actual retained artifact');
      await io.ensure(root);
      validateTree(await io.inspect(root), root);
      check(!(await io.inspect(target)).exists, 'destination appeared before cache copy');
      await io.copy(from, target);
      const copied = validateTree(await io.inspect(target), target);
      check(copied.bytes === cacheInventory.bytes && copied.fileCount === cacheInventory.fileCount, 'copied cache inventory differs');
      check(await binaryHash(path.join(target, 'release', 'sceneworks-rust-api.exe')) === originalBinarySha256, 'copied cache binary differs');
      receipt.cache = { kind: 'copied-build-cache-not-new-source-certification', from, to: target, ...budget, afterCopy: await io.volume(), originalRun: original.run, originalSource: original.source, originalInference: original.inference, originalRuntimeSha256: original.runtimeSha, originalBinarySha256, originalCacheRetained: true, requiredNextStep: 'normal cargo build at the current ref and actual binary inventory; never relabel this cache receipt' };
    }
  }
  if (original.cacheProbeError) receipt.cache = { state: 'unproved-cache-retained-without-copy', error: original.cacheProbeError };
  } else receipt.cache={state:reuse?'native-reuse-original-targets-retained-no-copy':'hf-only-recovery-original-targets-retained-no-copy'};
  receipt.after = await io.volume();
  io.report({ ...receipt, stage: 'completed' });
  await io.ensure(root);
  await io.write(path.join(root, 'windows-maintenance.json'), `${JSON.stringify(receipt, null, 2)}\n`);
  return receipt;
}
export function validateHfOnlyEnvelope(env) {
  const value=JSON.parse(env.MOSS_REFERENCE);
  check(env.GITHUB_ACTIONS==='true'&&env.MOSS_SOURCE_SHA===env.GITHUB_SHA&&/^[a-f0-9]{40}$/.test(env.GITHUB_SHA)&&env.MOSS_REFERENCE_BACKEND==='mlx','fixed reference Actions source required');
  check(value&&Object.keys(value).sort().join(',')==='artifact_id,artifact_run_id,phase'&&['upload','execute'].includes(value.phase),'closed reference cleanup envelope required');
  for(const key of ['artifact_id','artifact_run_id'])check(value.phase==='upload'?value[key]===null:typeof value[key]==='string'&&/^[1-9][0-9]{5,19}$/.test(value[key]),'exact reference artifact identity required');
  return value;
}
if (['--moss-disk-maintenance','--moss-disk-hf-only'].includes(process.argv[2])) {
  check(process.platform === 'win32', 'disk maintenance is Windows-only');
  const token = process.env.MOSS_MAINTENANCE_TOKEN;
  try {
    if(process.argv[2]==='--moss-disk-hf-only')validateHfOnlyEnvelope(process.env);
    await maintain(process.env, { volume, inspect: inspectTree, metadata: observeHf, entries: h47SourceEntries, processes: h47Processes, hash: async file => hashMetadataFile(file,entryIdentity(await lstat(file,{bigint:true}))), stopped: binding => stopped(binding, route => actionsApi(token, route)), read: readFile,
      remove: leaf => rm(leaf, { recursive: true, force: false }), copy: copyCacheExclusive, ensure: root => mkdir(root, { recursive: true }),
      write: (file, bytes) => writeFile(file, bytes, { flag: 'wx' }), report: value => console.log(JSON.stringify(value)) },{copyCache:process.argv[2]!=='--moss-disk-hf-only'});
  } catch (error) { console.error(JSON.stringify({ status: 'STOP', stage: 'owned-disk-maintenance', message: error.message })); process.exitCode = 1; }
}
