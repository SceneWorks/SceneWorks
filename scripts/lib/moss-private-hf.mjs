// Private installed model ownership only; shared HF inputs are never moved or removed.
import { lstat, readdir, rename, rm, writeFile, realpath, readlink, copyFile, mkdtemp, mkdir, appendFile } from 'node:fs/promises';
import { constants } from 'node:fs';
import { resolve, dirname, basename } from 'node:path';
import { canonical, requireThat, safePath, fileIdentity, json, digest } from './moss-edit-terminal.mjs';
import { inspectTree, powershell, hfEntries, entryIdentity } from '../ci/moss-owned-cache-maintenance.mjs';

const refuseFile=(message,path,info)=>requireThat(false,`${message}: ${JSON.stringify({path,...entryIdentity(info)})}`);
const absent=async path=>{try{await lstat(path);return false;}catch(error){if(error.code==='ENOENT')return true;throw error;}};
const identity=config=>({candidateId:config.candidateId,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,backend:config.backend,packagesSha256:digest(config.packageFiles)});
function owned(config,phase) {
  const match=new RegExp(`^moss-${phase}-([0-9]+)-([0-9]+)-(mlx|cuda)$`).exec(basename(config.root));
  requireThat(match&&match[3]===config.backend&&dirname(config.root)===resolve(config.sourceRoot,'../..','moss-sc-26752')&&config.hfHome===resolve(config.root,'hf')&&config.hfHome!==config.packageSource,'exact current private HF ownership required');
  return match;
}
async function confined(path,{descendants=true}={}) {
  for(let cursor=resolve(path);;cursor=dirname(cursor)) {
    const info=await lstat(cursor);requireThat(!info.isSymbolicLink(),'private HF refuses linked ancestors');
    if(dirname(cursor)===cursor)break;
  }
  if(process.platform==='win32') {
    if(descendants)await inspectTree(path); // Reject every Windows reparse tag before traversal.
    else await powershell(`$ErrorActionPreference='Stop';$p=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${Buffer.from(resolve(path)).toString('base64')}'));while($p){if(([IO.File]::GetAttributes($p) -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw 'Reparse ancestor'};$p=[IO.Path]::GetDirectoryName($p)};@{safe=$true}|ConvertTo-Json -Compress`);
  }
}
export async function privateHfIntegrity(config) {
  await confined(config.hfHome);
  const visit=async dir=>{for(const name of await readdir(dir)) {
    const path=resolve(dir,name),info=await lstat(path,{bigint:true});
    if(info.isSymbolicLink())refuseFile('private HF refuses symlinks/junctions',path,info);
    if(info.isDirectory())await visit(path);else if(!info.isFile()||info.nlink!==1n)refuseFile('private HF requires exclusive regular files',path,info);
  }};await visit(config.hfHome);
  const files=[];for(const item of config.packageFiles){const actual=await fileIdentity(safePath(config.hfHome,item.file));requireThat(actual.sha256===item.sha256,'private HF package hash mismatch');files.push({file:item.file,...actual});}return files;
}
export async function sealPrivateHf(config) {
  owned(config,'build');
  const receipt={...identity(config),root:config.root,hfHome:config.hfHome,runtime:await fileIdentity(resolve(config.root,'runtime.json')),files:await privateHfIntegrity(config)};
  await writeFile(resolve(config.root,'hf-owner.json'),JSON.stringify(receipt,null,2)+'\n',{flag:'wx',mode:0o600});return receipt;
}
export async function transferPrivateHf(config,installedInventory,inventoryIdentity) {
  const match=owned(config,'execute'),from=installedInventory.hfHome,fromRoot=dirname(from);
  requireThat(basename(fromRoot)===`moss-build-${match[1]}-${match[2]}-${config.backend}`&&basename(from)==='hf','installed HF source must name this exact run/backend private build');
  requireThat(from===resolve(fromRoot,'hf')&&from!==config.packageSource,'installed HF source must be this exact run/backend private build');
  requireThat(canonical(installedInventory.files.map(({file,sha256})=>({file,sha256})))===canonical(config.packageFiles.map(({file,sha256})=>({file,sha256}))),'installed projection must match every frozen package identity');
  await confined(config.root);requireThat(await absent(config.hfHome),'execution HF destination already exists');
  requireThat((await lstat(from)).dev===(await lstat(config.root)).dev,'private HF transfer requires the same volume');
  const owner=await json(resolve(fromRoot,'hf-owner.json')),runtime=await json(resolve(fromRoot,'runtime.json'));
  owned(runtime,'build');
  requireThat(canonical(identity(runtime))===canonical(identity(config))&&runtime.root===fromRoot&&runtime.hfHome===from&&owner.root===fromRoot&&owner.hfHome===from&&canonical(identity(runtime))===canonical(Object.fromEntries(Object.keys(identity(runtime)).map(key=>[key,owner[key]]))),'private HF build candidate/source/pin ownership mismatch');
  requireThat(canonical(await fileIdentity(resolve(fromRoot,'runtime.json')))===canonical(owner.runtime),'private HF runtime identity changed');
  const before=await privateHfIntegrity({...config,hfHome:from});requireThat(canonical(before)===canonical(owner.files)&&canonical(before)===canonical(installedInventory.files.map(({file,byteLength,sha256})=>({file,byteLength,sha256}))),'private HF installed inventory mismatch');
  requireThat(await absent(config.hfHome),'execution HF destination appeared before transfer');
  await rename(from,config.hfHome);
  const after=await privateHfIntegrity(config);requireThat(canonical(before)===canonical(after),'private HF transfer changed package bytes');
  const receipt={...identity(config),operation:'same-volume-private-HF-ownership-transfer',from,to:config.hfHome,originalPackages:inventoryIdentity,originalOwner:await fileIdentity(resolve(fromRoot,'hf-owner.json')),before,after};
  await writeFile(resolve(config.root,'hf-transfer.json'),JSON.stringify(receipt,null,2)+'\n',{flag:'wx',mode:0o600});return receipt;
}
export async function releasePrivateHf(config) {
  owned(config,'execute');const receipt=await json(resolve(config.root,'hf-transfer.json'));
  requireThat(receipt.to===config.hfHome&&canonical(identity(config))===canonical(Object.fromEntries(Object.keys(identity(config)).map(key=>[key,receipt[key]]))),'private HF cleanup ownership mismatch');
  const files=await privateHfIntegrity(config);requireThat(canonical(files)===canonical(receipt.after),'private HF final integrity mismatch');
  // Caller has stopped its workers and exported project state. Retain evidence outside the deleted leaf.
  await writeFile(resolve(config.root,'hf-final-integrity.json'),JSON.stringify({transfer:await fileIdentity(resolve(config.root,'hf-transfer.json')),files,workersStopped:true},null,2)+'\n',{flag:'wx',mode:0o600});
  await rm(config.hfHome,{recursive:true,force:false});
  await writeFile(resolve(config.root,'hf-released.json'),JSON.stringify({hfHome:config.hfHome,operation:'release-current-private-HF-after-worker-stop'})+'\n',{flag:'wx',mode:0o600});
}

// hf-hub installs snapshot pointers to blobs. Normalize only the two approved adapter
// pointers after the installer stops; execution's blanket link refusal remains intact.
export async function materializePrivateAdapter(config) {
  owned(config,'build');await confined(config.hfHome,{descendants:false});
  const allowed=config.editAdapterFiles??[],prefix=/^hub\/models--chinoll--Moss-Image-Edit\/snapshots\/[a-f0-9]{40}\//;
  requireThat(allowed.length===2&&allowed.every(file=>prefix.test(file))&&new Set(allowed.map(file=>file.slice(0,file.lastIndexOf('/')))).size===1&&canonical(allowed.map(file=>basename(file)).sort())===canonical(['adapter_config.json','adapter_model.safetensors']),'exact approved adapter snapshot pair required');
  const paths=new Set(allowed.map(file=>safePath(config.hfHome,file))),entries=await hfEntries(config.hfHome),plans=[],pairedPaths=new Set();
  for(const entry of entries)if(entry.type==='symlink'&&!paths.has(safePath(config.hfHome,entry.path)))refuseFile('unexpected private HF symlink',safePath(config.hfHome,entry.path),await lstat(safePath(config.hfHome,entry.path),{bigint:true}));
  for(const file of allowed) {
    const path=safePath(config.hfHome,file),info=await lstat(path,{bigint:true}),expected=config.packageFiles.find(item=>item.file===file);
    requireThat(expected,'adapter snapshot is absent from approved package inventory');await confined(dirname(path),{descendants:false});
    if(info.isFile()&&info.nlink===1n){requireThat((await fileIdentity(path)).sha256===expected.sha256,'regular adapter bytes/ownership mismatch');continue;}
    const blobRoot=resolve(dirname(path),'../../blobs'),sameIdentity=entries.filter(entry=>entry.type==='file'&&entry.dev===String(info.dev)&&entry.ino===String(info.ino));
    let link=null,blob;
    if(info.isSymbolicLink()){link=await readlink(path);blob=await realpath(path);}
    else {
      if(!info.isFile()||info.nlink!==2n||info.ino===0n||sameIdentity.length!==2)refuseFile('adapter hardlink requires exactly two proven private entries',path,info);
      const other=sameIdentity.find(entry=>safePath(config.hfHome,entry.path)!==path);requireThat(other,'private adapter blob alias missing');blob=safePath(config.hfHome,other.path);
    }
    requireThat(dirname(blob)===blobRoot&&/^[a-f0-9]{40}(?:[a-f0-9]{24})?$/.test(basename(blob)),'adapter link escapes exact private repository blobs');
    await confined(blob,{descendants:false});const blobInfo=await lstat(blob,{bigint:true}),before=await fileIdentity(blob),expectedLinks=link===null?2n:1n;
    requireThat(blobInfo.isFile()&&!blobInfo.isSymbolicLink()&&blobInfo.nlink===expectedLinks&&before.sha256===expected.sha256,'adapter blob identity/hash mismatch');
    if(link===null)requireThat(info.dev===blobInfo.dev&&info.ino===blobInfo.ino,'adapter snapshot/blob hardlink identity mismatch');
    pairedPaths.add(path);pairedPaths.add(blob);plans.push({file,path,info,expected,link,blob,blobInfo,before,expectedLinks,sameIdentity});
  }
  // Expected pairs are proved before any replacement; every unrelated entry stays exclusive.
  for(const entry of entries) {
    const path=safePath(config.hfHome,entry.path);
    if(entry.type==='symlink'&&paths.has(path))continue;
    if(entry.type!=='file'||(entry.nlink!=='1'&&!pairedPaths.has(path)))refuseFile('private HF requires exclusive regular files',path,await lstat(path,{bigint:true}));
  }
  const operations=[];
  for(const {file,path,info,link,blob,blobInfo,before,expectedLinks,sameIdentity} of plans) {
    const temporary=await mkdtemp(resolve(dirname(path),'.moss-adapter-'));
    try {
      const replacement=resolve(temporary,'payload');await copyFile(blob,replacement,constants.COPYFILE_EXCL);
      requireThat(canonical(await fileIdentity(replacement))===canonical(before)&&(await lstat(replacement,{bigint:true})).nlink===1n,'materialized adapter bytes/ownership mismatch');
      const current=await lstat(path,{bigint:true}),currentBlob=await lstat(blob,{bigint:true});
      requireThat(current.ino===info.ino&&current.dev===info.dev&&current.nlink===info.nlink&&(link===null?current.isFile():current.isSymbolicLink()&&await readlink(path)===link)&&currentBlob.ino===blobInfo.ino&&currentBlob.dev===blobInfo.dev&&currentBlob.nlink===expectedLinks&&canonical(await fileIdentity(blob))===canonical(before),'adapter link/blob changed before replacement');
      await rename(replacement,path);const after=await lstat(path,{bigint:true}),afterBlob=await lstat(blob,{bigint:true});
      requireThat(after.isFile()&&!after.isSymbolicLink()&&after.nlink===1n&&afterBlob.nlink===1n&&after.ino!==afterBlob.ino&&canonical(await fileIdentity(path))===canonical(before)&&canonical(await fileIdentity(blob))===canonical(before),'adapter materialization did not produce exclusive frozen bytes');
      operations.push({file,originalLink:link,originalBlob:blob,originalLinkIdentity:entryIdentity(info),originalBlobIdentity:entryIdentity(blobInfo),sameIdentityEntries:sameIdentity,afterSnapshot:entryIdentity(after),afterBlob:entryIdentity(afterBlob),identity:before,operation:link===null?'materialize-proven-private-installer-hardlink':'materialize-approved-private-installer-pointer'});
    }finally{await rm(temporary,{recursive:true,force:false});}
  }
  await privateHfIntegrity(config);
  await writeFile(resolve(config.root,'adapter-materialization.json'),JSON.stringify({candidateId:config.candidateId,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,operations},null,2)+'\n',{flag:'wx',mode:0o600});return operations;
}
export async function normalizeMossEnvironment(env) {
  const workspace=await realpath(env.GITHUB_WORKSPACE),base=resolve(workspace,'..','moss-sc-26752'),rawBase=resolve(env.GITHUB_WORKSPACE,'..','moss-sc-26752'),result={};
  await mkdir(base,{recursive:true});await confined(base,{descendants:false});
  for(const key of ['MOSS_ROOT','MOSS_BUILDS','MOSS_BUNDLE'])if(env[key]) {
    requireThat(dirname(resolve(env[key]))===rawBase&&/^moss-(?:build|builds|execute|review|bundle)-[0-9]+-[0-9]+(?:-(?:build|execute|review|mlx|cuda))*$/.test(basename(env[key])),'MOSS artifact path escaped durable namespace');
    result[key]=resolve(base,basename(env[key]));
  }
  result.MOSS_SOURCE=resolve(workspace,'source');result.MOSS_INFERENCE=resolve(workspace,'inference');
  if(env.MOSS_UPLOAD_PATH)result.MOSS_UPLOAD_PATH=await realpath(env.MOSS_UPLOAD_PATH);
  requireThat(Object.values(result).every(value=>!/[\r\n]/.test(value)),'invalid multiline Actions path');
  return result;
}
export async function exportMossEnvironment(env) {
  const result=await normalizeMossEnvironment(env);await appendFile(env.GITHUB_ENV,Object.entries(result).map(([key,value])=>`${key}=${value}\n`).join(''));return result;
}
