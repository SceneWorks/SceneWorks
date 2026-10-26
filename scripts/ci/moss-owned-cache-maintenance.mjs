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
export const OWNED_RUNS = Object.freeze([
  { run: '40145054879', candidate: 'sc-26752-final-r4', source: '37d5ad93f858025d3202a2ddcab465c67174e0c7', inference: '878f26aba50bfb6c371437da9ca7ead81052c25a', workspace: 'D:\\actions-runner\\_work\\SceneWorks\\SceneWorks', runtimeSha: '1b8948de0a9f7ede92019896359683605fe2c4e7577971520da881dda6aca710' },
  { run: '40152001538', candidate: 'sc-26752-final-r5', source: '9ac78275e2f5f536541c1790fc840232fd5bdfdd', inference: native, workspace: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', runtimeSha: '7ac52e00e6e712012b907a740e96e5c9d39c72483499205f637916e286e70b43' },
  { run: '40155160639', candidate: 'sc-26752-final-r6', source: '2490b2eed9bf986967c389ac69171d03b04835d7', inference: native, workspace: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', runtimeSha: 'cfb0491496d4e29aafc466404526592fe2d2950a12f1566c74fcab9e728f007e' },
  { run: '40163530825', candidate: 'sc-26752-final-r10', source: 'c17fd74650f2464427b30cea7669c7a284ea4a73', inference: native, workspace: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', root: 'D:\\actions-runner-2\\_work\\SceneWorks\\moss-sc-26752\\moss-build-40163530825-1-cuda', runtimeSha: '546808654e2a0d44d26a5d594b9e7c20961b2b9d947e900effdefdf9f8c2d625', cacheBinarySha256: '93dbe1db05a946517484539199b69927ebc4eb9bae3bab508ba2636529b81b97', metadataRequired: true },
  { run: '40167292776', candidate: 'sc-26752-final-r12', source: 'e6475176bf1cf07a46530adbe659aa0d6492c71a', inference: native, workspace: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', root: 'D:\\actions-runner-2\\_work\\SceneWorks\\moss-sc-26752\\moss-build-40167292776-1-cuda', runtimeSha: '47a25951ea0e1d6fd076802c2ca68421e36d90a296fa1813fc344e6d0a461355', cacheBinarySha256: '07cb4ef31caba2f875d6750c461715f880c79288201af88af11997e31809d69d', metadataRequired: true },
  { run: '40169818205', candidate: 'sc-26752-final-r13', source: 'ba6bdd824823f01f8c4dc435f9fad50f45f97893', inference: native, workspace: 'D:\\actions-runner\\_work\\SceneWorks\\SceneWorks', root: 'D:\\actions-runner\\_work\\SceneWorks\\moss-sc-26752\\moss-build-40169818205-1-cuda', runtimeSha: '76554eb462c580cd6afa7b5102aaeaa434de85049fe1b6701fe7ba2347f6d61f', cacheBinarySha256: '09088a12b26968177adf5a6ead76b088be2f791e5756224497db4fb7d20cf8fd', metadataRequired: true },
  { run: '40174802721', candidate: 'sc-26752-final-r14', source: '12e1b364360b43b28e864879be4c68bb28a696bc', inference: native, workspace: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', root: 'D:\\actions-runner-2\\_work\\SceneWorks\\moss-sc-26752\\moss-build-40174802721-1-cuda', runtimeSha: '4082f9c28ae31aee7dda8e850d04b197d7741a01edbc3856aa53ddac6ba8a532', cacheBinarySha256: 'ab36062090356912f9131654fa5c3ec87ed808f5e0fc0cbd32e388773edb75c2', metadataRequired: true },
  { run: '40181253824', candidate: 'sc-26752-final-r15', source: 'a0692191b09813fab859659bf382bd7c91a60140', inference: native, workspace: 'D:\\actions-runner-2\\_work\\SceneWorks\\SceneWorks', root: 'D:\\actions-runner-2\\_work\\SceneWorks\\moss-sc-26752\\moss-build-40181253824-1-cuda', runtimeSha: 'a4c17012e7143f8775809decbd72216a7c3114209bf91c3f7f0d9608f0585cc4', metadataRequired: true },
].map(item => Object.freeze({ ...item, attempt: 1, root: item.root ?? path.join(item.workspace, `moss-build-${item.run}-1-cuda`) })));
// Bigint identities are serialized as decimal strings; never round Windows file IDs.
export function entryIdentity(info) {
  return {type:info.isSymbolicLink()?'symlink':info.isDirectory()?'directory':info.isFile()?'file':'other',dev:String(info.dev),ino:String(info.ino),nlink:String(info.nlink),byteLength:String(info.size)};
}
export async function hfEntries(root) {
  if(process.platform==='win32')await inspectTree(root); // No Windows reparse traversal, including non-symlink tags.
  const rootInfo=await lstat(root,{bigint:true});check(rootInfo.isDirectory()&&!rootInfo.isSymbolicLink(),'HF metadata root must be a real directory');
  const entries=[];let visited=0;
  const visit=async dir=>{for(const name of (await readdir(dir)).sort()) {
    check(++visited<=10000,'HF metadata inventory bound exceeded');const file=fsResolve(dir,name),info=await lstat(file,{bigint:true}),entry={path:fsRelative(root,file).replaceAll('\\','/'),...entryIdentity(info)};
    if(info.isDirectory())await visit(file);else {if(info.isSymbolicLink())entry.link=await readlink(file);entries.push(entry);}
  }};await visit(root);return entries;
}
export async function hashMetadataFile(file,expected) {
  const before=entryIdentity(await lstat(file,{bigint:true}));check(before.type==='file'&&JSON.stringify(before)===JSON.stringify(Object.fromEntries(Object.keys(before).map(key=>[key,expected[key]]))),'HF metadata file changed before hash');
  const hash=createHash('sha256');for await(const chunk of createReadStream(file))hash.update(chunk);
  check(JSON.stringify(entryIdentity(await lstat(file,{bigint:true})))===JSON.stringify(before),'HF metadata file changed during hash');return hash.digest('hex');
}
export async function observeHf(runtime) {
  const entries=await hfEntries(runtime.hfHome),adapters=new Set(runtime.editAdapterFiles),packages=new Set(runtime.packageFiles?.map(item=>item.file)??[]),blobPrefix='hub/models--chinoll--Moss-Image-Edit/blobs/';
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
  for (const run of [latest, attempt]) check(String(run.id) === binding.run && run.run_attempt === 1 && run.head_sha === binding.source && run.status === 'completed' && run.conclusion, 'old run identity/attempt is not stopped');
  check(jobs.length > 0 && jobs.every(job => job.status === 'completed' && job.conclusion && Number(job.run_id) === Number(binding.run)), 'old run still has nonterminal jobs');
}
export function validateRuntime(binding, bytes) {
  check(sha(bytes) === binding.runtimeSha, 'old runtime bytes changed');
  const runtime = JSON.parse(bytes);
  check(runtime.candidateId === binding.candidate && runtime.sourceSha === binding.source && runtime.inferenceSha === binding.inference && runtime.backend === 'cuda', 'old runtime identity mismatch');
  for (const [field, expected] of Object.entries({ root: binding.root, hfHome: path.join(binding.root, 'hf'), sourceRoot: path.join(binding.workspace, 'source'), packageSource: 'E:\\sceneworks-evaluation\\moss\\sc-26322-reference-v9' })) check(normal(runtime[field] ?? '') === normal(expected), `old runtime ${field} mismatch`);
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
  const receipt = { version: 1, kind: 'sc-26752-stopped-run-disk-recovery', currentRun: env.GITHUB_RUN_ID, currentSource: env.MOSS_SOURCE_SHA, before: await io.volume(), candidates: [], operations: [] };
  validateTree(await io.inspect(root), root);
  receipt.r9Observation = await diagnoseR9(io);
  for (const binding of OWNED_RUNS) {
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
    receipt.candidates.push({ ...binding, stop, inventory, metadata, cache, cacheProbeError });
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
    if(candidate.metadataRequired) {
      const metadata = validateHfObservation(await io.metadata(runtime), leaf, runtime);
      check(JSON.stringify(metadata)===JSON.stringify(candidate.metadata),'owned HF metadata changed before cleanup');
      io.report({stage:candidate.run==='40181253824'?'fixed-r15-readonly-metadata-before-hf-cleanup':candidate.run==='40174802721'?'fixed-r14-readonly-metadata-before-hf-cleanup':candidate.run==='40169818205'?'fixed-r13-readonly-metadata-before-hf-cleanup':candidate.run==='40167292776'?'fixed-r12-readonly-metadata-before-hf-cleanup':'fixed-r10-readonly-metadata-before-hf-cleanup',run:candidate.run,source:candidate.source,runtimeSha256:candidate.runtimeSha,metadata});
    }
    await io.remove(leaf); // Exact fixed HF leaf only; no root/cache/model-source wildcard.
    receipt.operations.push({ kind: 'remove-owned-hf', path: leaf, after: await io.volume() });
    io.report({ stage: 'hf-reclaimed', operation: receipt.operations.at(-1) });
  }
  if(copyCache) {
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
  } else receipt.cache={state:'hf-only-recovery-original-targets-retained-no-copy'};
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
// Manual observation/reclamation of three exact, archived owners. R19 is observation-only.
// This separate route never calls maintain(), copies a cache, or removes a run/target root.
export const SELECTED_HF = Object.freeze([
  {key:'r19-protected',run:'40349954885',source:'3aa58facf61fb9a2e3b6e139ce4a0a331c9f7685',product:'3aa58facf61fb9a2e3b6e139ce4a0a331c9f7685',inference:'78100568d8e5b9fb5b199872dc57cb28f5b87340',candidate:'sc-26752-final-r19',listener:'actions-runner',runtimeSha:'7356a1e46e2e8145f84c5a9144bad378bda0d7ec829df1a94191d4cce538b756',artifact:'12677781687',artifactBytes:76528916,artifactSha:'f4c84bfedb071ef6b3c8dbdba775e18077acde7a679bfab36349b0d0a865f8f8',transferSha:'c193cda586b18fc81513ac15069326a79215e8f328430b3fdf22250864fac94d',pids:[49752,54004]},
  {key:'h39',run:'40331712229',source:'63874fe81e45750a8b98a5bf1b821383af40ae72',product:'2f20745fa728b9715628f3a8d8fd016512712267',inference:'7275f7c672eecabeba9342e99897ef484e992e14',candidate:'sc-26752-final-r18',listener:'actions-runner',runtimeSha:'5d756823a2ead65e9da094eb353a6b530ea6798b7a6796a28fbdcc1494548fd6',artifact:'12655937992',artifactBytes:75943567,artifactSha:'1a06c4b1eacadc24c11ca3cdae97342db9766eab0bdcd85cae47bb3ac1e1cb62',transferSha:'a487d16ca326b53b1b27d0000a1f6e50cae7a01ba16f001d56afb9c3c94fc34f',pids:[29800,8692]},
  {key:'h42-partial',run:'40383291274',source:'d2b5ff1a45b36a6b316974c4e8a080d31be0f407',product:'3aa58facf61fb9a2e3b6e139ce4a0a331c9f7685',inference:'78100568d8e5b9fb5b199872dc57cb28f5b87340',candidate:'sc-26752-final-r19',listener:'actions-runner-2',runtimeSha:'543e91c95f855d519240abe7e2ac41e195ad1e571e5cb7933c1dafa3070bdd60',artifact:'12681360045',artifactBytes:8534,artifactSha:'0923b0ce57d5fe60d39b27a711951d424bdbe4a4d2ec430303d224d93bbbd562',derivedRuntimeSha:'449d09a6d6f2dddb5c3ef9293fe3385f3b6d659da82f48b531710b50d9b95c21',pids:[]},
].map(item=>{const workspace=`D:\\${item.listener}\\_work\\SceneWorks\\SceneWorks`,parent=path.join(workspace,'..','moss-sc-26752'),root=path.join(parent,`moss-execute-${item.run}-1-cuda`);return Object.freeze({...item,attempt:1,workspace,root,leaf:path.join(item.key==='h42-partial'?path.join(parent,`moss-build-${item.run}-1-cuda`):root,'hf'),pids:Object.freeze(item.pids)});}));
export function selectedHfEnvelope(env) {
  const value=JSON.parse(env.MOSS_REFERENCE??'{}');
  check(env.GITHUB_ACTIONS==='true'&&env.GITHUB_REPOSITORY==='SceneWorks/SceneWorks'&&env.GITHUB_JOB==='moss-owned-hf-maintenance'&&env.MOSS_REFERENCE_BACKEND==='candle','registered Windows HF maintenance job required');
  check(/^[a-f0-9]{40}$/.test(env.GITHUB_SHA??'')&&env.MOSS_SOURCE_SHA===env.GITHUB_SHA&&/^[1-9][0-9]*$/.test(env.GITHUB_RUN_ID??'')&&/^[1-9][0-9]*$/.test(env.GITHUB_RUN_ATTEMPT??'')&&!SELECTED_HF.some(item=>item.run===env.GITHUB_RUN_ID),'exact maintenance source/current run required');
  check(/^D:\\actions-runner(?:-[234])?\\_work\\SceneWorks\\SceneWorks$/i.test(path.normalize(env.GITHUB_WORKSPACE??'')),'fixed Windows workspace required');
  check(value&&Object.keys(value).sort().join(',')==='candidate,phase'&&['inventory','cleanup'].includes(value.phase),'closed selected HF envelope required');
  check(value.phase==='inventory'?value.candidate===null:['h39','h42-partial'].includes(value.candidate),'exact unprotected HF cleanup candidate required');
  return value;
}
export function selectedArtifact(binding,artifact) {
  check(String(artifact.id)===binding.artifact&&artifact.name===`moss-execution-cuda-${binding.run}-1`&&artifact.size_in_bytes===binding.artifactBytes&&artifact.digest===`sha256:${binding.artifactSha}`&&artifact.expired===false&&String(artifact.workflow_run?.id)===binding.run&&artifact.workflow_run?.head_sha===binding.source,'archived HF artifact identity changed');return artifact;
}
export function selectedRuntime(binding,bytes) {
  check(bytes.length<=131072&&sha(bytes)===binding.runtimeSha,'archived execution runtime bytes changed');const runtime=JSON.parse(bytes);
  check(runtime.candidateId===binding.candidate&&runtime.sourceSha===binding.product&&runtime.inferenceSha===binding.inference&&runtime.backend==='cuda'&&runtime.root===binding.root&&runtime.hfHome===path.join(binding.root,'hf')&&runtime.sourceRoot===path.join(binding.workspace,'source'),'selected runtime producer/path mismatch');
  check(runtime.packageFiles?.length===43&&sha(JSON.stringify(runtime.packageFiles))==='8d774bf01dffe857fe255e5c9fdb45dcdf7071ae1239f35e1555bf6ba39cf464','selected runtime package closure changed');
  return runtime;
}
export function partialBuildRuntime(runtime) {
  const root=path.dirname(SELECTED_HF[2].leaf),result={...runtime};
  // Exact runtimeConfig(configure) projection plus execute-reused-cuda's packageSource override.
  Object.assign(result,{root,hfHome:path.join(root,'hf'),stateRoot:path.join(root,'state'),observationJournal:path.join(root,'journal-0.jsonl'),playwrightModule:path.join(root,'playwright/node_modules/playwright/index.mjs'),handoffRoot:path.join(root,'prior/handoff'),priorEvidenceRoot:path.join(root,'prior/evidence'),reviewRoot:path.join(root,'reviews'),packageSource:SELECTED_HF[0].leaf});
  const bytes=Buffer.from(JSON.stringify(result,null,2)+'\n');check(sha(bytes)===SELECTED_HF[2].derivedRuntimeSha,'derived partial build runtime changed');return bytes;
}
export function selectedEntries(binding,entries,packages) {
  check(Array.isArray(entries)&&entries.length<=43,'unexpected HF file inventory');const allowed=new Map(packages.map(item=>[item.file,item])),seen=new Set(),ids=new Set();let bytes=0n;
  for(const entry of entries) {
    const item=allowed.get(entry.path);check(item&&!seen.has(entry.path)&&entry.type==='file'&&entry.nlink==='1'&&['dev','ino','byteLength'].every(key=>/^[0-9]+$/.test(entry[key]??'')),'HF requires fixed exclusive regular package files');
    const id=entry.dev+':'+entry.ino;check(!ids.has(id)&&BigInt(entry.ino)>0n,'ambiguous HF file ownership');ids.add(id);seen.add(entry.path);
    check(Number.isSafeInteger(item.byteLength)&&item.byteLength>=0&&BigInt(entry.byteLength)<=BigInt(item.byteLength),'HF file exceeds archived package extent');bytes+=BigInt(entry.byteLength);
    if(binding.key!=='h42-partial')check(BigInt(entry.byteLength)===BigInt(item.byteLength),'complete HF package extent changed');
  }
  if(binding.key!=='h42-partial')check(entries.length===43,'complete HF inventory missing package');
  check(bytes<=BigInt(Number.MAX_SAFE_INTEGER),'HF byte sum exceeds exact range');return {entries,bytes:Number(bytes),sha256:sha(JSON.stringify(entries))};
}
export function selectedProcesses(observation) {
  check(observation?.complete===true&&Array.isArray(observation.matches)&&observation.matches.length===0,'live/ambiguous owned process: no cleanup permitted');return observation;
}
export async function selectedLiveProcesses(binding) {
  // Targeted WMI filter only. A recycled historical PID is ambiguous and blocks cleanup too.
  const roots=[path.basename(binding.root),path.basename(path.dirname(binding.leaf)),`moss-builds-${binding.run}-1-execute-cuda`],filters=[...binding.pids.map(pid=>`ProcessId=${pid}`),...new Set(roots)].map(value=>typeof value==='string'&&value.startsWith('ProcessId=')?value:`CommandLine LIKE '%${value}%'`);
  const binary=path.join(path.dirname(binding.root),`moss-builds-${binding.run}-1-execute-cuda`,'cuda','sceneworks-rust-api.exe');filters.push(`ExecutablePath='${binary.replaceAll('\\','\\\\')}'`);
  const encoded=Buffer.from(filters.join(' OR ')).toString('base64');
  return powershell(`$ErrorActionPreference='Stop';[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$filter=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${encoded}'))
$matches=@(Get-CimInstance Win32_Process -Filter $filter | ForEach-Object {@{pid=[long]$_.ProcessId;name=$_.Name;executable=$_.ExecutablePath}})
@{complete=$true;matches=$matches}|ConvertTo-Json -Depth 5 -Compress`);
}
export async function maintainSelectedHf(env,io) {
  const request=selectedHfEnvelope(env); // Reject before telemetry, API, process query or filesystem access.
  const receipt={kind:'sc-26752-selected-hf-maintenance/v1',request,currentRun:env.GITHUB_RUN_ID,currentAttempt:env.GITHUB_RUN_ATTEMPT,toolingSha:env.GITHUB_SHA,before:await io.volume(),candidates:[],operations:[]};
  const read=async file=>{validateTree(await io.inspect(file),file);const bytes=await io.read(file);check(bytes.length<=131072,'bounded ownership record required');return bytes;};
  // Extents come from an archived, exact-hash transfer; learned bytes are never scanned.
  const protectedRuntime=selectedRuntime(SELECTED_HF[0],await read(path.join(SELECTED_HF[0].root,'runtime.json'))),transferBytes=await read(path.join(SELECTED_HF[0].root,'hf-transfer.json'));
  check(sha(transferBytes)===SELECTED_HF[0].transferSha,'protected archived transfer bytes changed');const packages=JSON.parse(transferBytes).after;
  check(packages?.length===43&&JSON.stringify(packages.map(({file,sha256})=>({file,sha256})))===JSON.stringify(protectedRuntime.packageFiles),'archived package extents changed');
  async function observe(binding) {
    const stop=await io.stopped(binding),artifact=selectedArtifact(binding,await io.artifact(binding)),runtimeBytes=await read(path.join(binding.root,'runtime.json')),runtime=selectedRuntime(binding,runtimeBytes);
    const inventory=validateTree(await io.inspect(binding.leaf),binding.leaf);let metadata=null,derivedRuntime=null;
    if(binding.key==='h42-partial'&&inventory.exists){const bytes=await read(path.join(path.dirname(binding.leaf),'runtime.json')),expected=partialBuildRuntime(runtime);check(bytes.equals(expected),'live partial build runtime differs from exact derived projection');derivedRuntime={sha256:sha(bytes),byteLength:bytes.length,provenance:'derived from archived execution runtime plus reviewed configure transform; not an archived build runtime'};}
    if(binding.transferSha){const bytes=await read(path.join(binding.root,'hf-transfer.json'));check(sha(bytes)===binding.transferSha,'archived transfer bytes changed');}
    if(inventory.exists){metadata=selectedEntries(binding,await io.entries(binding.leaf),packages);check(metadata.bytes===inventory.bytes&&metadata.entries.length===inventory.fileCount,'HF telemetry/metadata mismatch');}
    return {key:binding.key,protected:binding.key==='r19-protected',leaf:binding.leaf,run:binding.run,attempt:1,actionsSource:binding.source,productSource:binding.product,native:binding.inference,runtimeSha256:sha(runtimeBytes),artifact:{id:artifact.id,digest:artifact.digest},stop,inventory,metadata,derivedRuntime};
  }
  for(const binding of SELECTED_HF)receipt.candidates.push(await observe(binding));
  const report=stage=>io.report({stage,receipt,receiptSha256:sha(JSON.stringify(receipt))});report('validated-readonly-inventory');
  if(request.phase==='cleanup') {
    const binding=SELECTED_HF.find(item=>item.key===request.candidate),original=receipt.candidates.find(item=>item.key===binding.key);check(binding.key!=='r19-protected','protected R19 is never a cleanup candidate');
    selectedProcesses(await io.processes(binding));const current=await observe(binding);
    check(JSON.stringify(current.inventory)===JSON.stringify(original.inventory)&&JSON.stringify(current.metadata)===JSON.stringify(original.metadata),'HF identity changed before cleanup');
    selectedProcesses(await io.processes(binding));
    report('validated-before-single-leaf-cleanup');
    if(current.inventory.exists){await io.remove(binding.leaf);const after=validateTree(await io.inspect(binding.leaf),binding.leaf);check(!after.exists,'HF cleanup did not remove exact leaf');receipt.operations.push({kind:'remove-single-owned-hf',path:binding.leaf,observedBytesBefore:current.inventory.bytes,after,volume:await io.volume()});}
  }
  receipt.after=await io.volume();report('completed');return receipt;
}
if(process.argv[2]==='--moss-selected-hf') {
  try {selectedHfEnvelope(process.env);check(process.platform==='win32','selected HF maintenance is Windows-only');const token=process.env.MOSS_MAINTENANCE_TOKEN;
    await maintainSelectedHf(process.env,{volume,inspect:inspectTree,entries:hfEntries,read:readFile,stopped:binding=>stopped(binding,route=>actionsApi(token,route)),artifact:binding=>actionsApi(token,`actions/artifacts/${binding.artifact}`),processes:selectedLiveProcesses,remove:leaf=>rm(leaf,{recursive:true,force:false}),report:value=>console.log(JSON.stringify(value))});
  }catch(error){console.error(JSON.stringify({status:'STOP',stage:'selected-hf-maintenance',message:error.message}));process.exitCode=1;}
}
if (['--moss-disk-maintenance','--moss-disk-hf-only'].includes(process.argv[2])) {
  check(process.platform === 'win32', 'disk maintenance is Windows-only');
  const token = process.env.MOSS_MAINTENANCE_TOKEN;
  try {
    if(process.argv[2]==='--moss-disk-hf-only')validateHfOnlyEnvelope(process.env);
    await maintain(process.env, { volume, inspect: inspectTree, metadata: observeHf, hash: async file => hashMetadataFile(file,entryIdentity(await lstat(file,{bigint:true}))), stopped: binding => stopped(binding, route => actionsApi(token, route)), read: readFile,
      remove: leaf => rm(leaf, { recursive: true, force: false }), copy: copyCacheExclusive, ensure: root => mkdir(root, { recursive: true }),
      write: (file, bytes) => writeFile(file, bytes, { flag: 'wx' }), report: value => console.log(JSON.stringify(value)) },{copyCache:process.argv[2]!=='--moss-disk-hf-only'});
  } catch (error) { console.error(JSON.stringify({ status: 'STOP', stage: 'owned-disk-maintenance', message: error.message })); process.exitCode = 1; }
}
