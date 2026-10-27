// Private installed model ownership only; shared HF inputs are never moved or removed.
import { lstat, readdir, rename, rm, writeFile, realpath, readlink, copyFile, mkdtemp, mkdir, appendFile } from 'node:fs/promises';
import { constants } from 'node:fs';
import { resolve, dirname, basename } from 'node:path';
import { canonical, requireThat, safePath, fileIdentity, json, digest } from './moss-edit-terminal.mjs';
import { inspectTree, powershell, hfEntries, entryIdentity } from '../ci/moss-owned-cache-maintenance.mjs';

const refuseFile=(message,path,info)=>requireThat(false,`${message}: ${JSON.stringify({path,...entryIdentity(info)})}`);
const absent=async path=>{try{await lstat(path);return false;}catch(error){if(error.code==='ENOENT')return true;throw error;}};
const identity=config=>({candidateId:config.candidateId,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,backend:config.backend,packagesSha256:digest(config.packageFiles)});
export const MAC_MOSS_BASE='/Volumes/Models/sceneworks-terminal-sc-26752';
export function allowedPrivateParent(parent,sourceRoot,platform=process.platform) {return parent===resolve(sourceRoot,'../..','moss-sc-26752')||(platform==='darwin'&&parent===MAC_MOSS_BASE);}
export function cargoTarget(config) {return resolve(config.sourceRoot,'../..','moss-sc-26752',basename(config.root),'target');}
function owned(config,phase) {
  const match=new RegExp(`^moss-${phase}-([0-9]+)-([0-9]+)-(mlx|cuda)$`).exec(basename(config.root));
  requireThat(match&&match[3]===config.backend&&allowedPrivateParent(dirname(config.root),config.sourceRoot)&&config.hfHome===resolve(config.root,'hf')&&config.hfHome!==config.packageSource,'exact current private HF ownership required');
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
// The corrected CUDA binary uses the same frozen 43 installed packages. Move
// the last stopped private owner into the new build namespace on this volume;
// never fall back to a second projection of the shared model store.
export async function adoptStoppedCudaBuildPrivateHf(config,artifactRoot,old,continuity,artifactIdentity) {
  owned(config,'build');
  requireThat(config.backend==='cuda'&&old?.runId>0&&old?.attempt>0&&old?.artifactId>0&&old?.originalRunId>0&&old?.originalAttempt>0&&old.runId!==old.originalRunId&&continuity?.oldSourceSha===old.sourceSha&&continuity?.oldInferenceSha===old.inferenceSha&&continuity?.sourceSha===config.sourceSha&&continuity?.inferenceSha===config.inferenceSha,'cross-candidate CUDA package owner or reviewed source pair missing');
  const prior=await json(resolve(artifactRoot,'runtime.json')),transfer=await json(resolve(artifactRoot,'hf-transfer.json')),preparation=await json(resolve(artifactRoot,'cuda-package-preparation.json'));
  const fromRoot=prior.root,from=prior.hfHome;
  requireThat(basename(fromRoot)===`moss-execute-${old.runId}-${old.attempt}-cuda`&&from===resolve(fromRoot,'hf')&&prior.backend==='cuda'&&prior.candidateId===old.candidateId&&prior.sourceSha===old.sourceSha&&prior.inferenceSha===old.inferenceSha,'stopped CUDA artifact names another execution owner');
  owned(prior,'execute');
  requireThat(transfer.operation==='same-volume-stopped-CUDA-HF-adoption'&&transfer.to===from&&transfer.priorRunId>0&&transfer.priorAttempt>0&&transfer.priorArtifactId>0&&transfer.originalRunId===old.originalRunId&&transfer.originalAttempt===old.originalAttempt&&transfer.candidateId===old.candidateId&&transfer.sourceSha===old.sourceSha&&transfer.inferenceSha===old.inferenceSha&&canonical(transfer.before)===canonical(transfer.after),'stopped CUDA owner lacks its authenticated original transfer');
  const parentPath=resolve(artifactRoot,'stopped-cuda-hf-transfer.json'),parent=await json(parentPath);
  requireThat(canonical(await fileIdentity(parentPath))===canonical(transfer.priorTransfer)&&parent.to===transfer.from&&parent.originalRunId===old.originalRunId&&parent.originalAttempt===old.originalAttempt&&canonical(parent.after)===canonical(transfer.before)&&canonical(parent.before)===canonical(parent.after)&&canonical(parent.origin??{originalBuildHf:parent.from,originalOwner:parent.originalOwner,initialTransfer:transfer.priorTransfer})===canonical(transfer.origin),'stopped CUDA transfer chain changed');
  requireThat(preparation.mode==='stopped-execution-HF'&&preparation.candidateId===old.candidateId&&preparation.sourceSha===old.sourceSha&&preparation.inferenceSha===old.inferenceSha&&preparation.stoppedArtifact?.runId===transfer.priorRunId&&preparation.stoppedArtifact?.attempt===transfer.priorAttempt&&preparation.stoppedArtifact?.artifactId===transfer.priorArtifactId&&canonical(preparation.transfer)===canonical(await fileIdentity(resolve(artifactRoot,'hf-transfer.json'))),'stopped CUDA package preparation changed');
  for(const name of ['runtime.json','hf-transfer.json','cuda-package-preparation.json'])requireThat(canonical(await fileIdentity(resolve(artifactRoot,name)))===canonical(await fileIdentity(resolve(fromRoot,name))),`stopped CUDA ${name} differs from archived bytes`);
  requireThat(canonical(prior.packageFiles.map(({file,sha256})=>({file,sha256})))===canonical(config.packageFiles.map(({file,sha256})=>({file,sha256})))&&canonical(transfer.after.map(({file,sha256})=>({file,sha256})))===canonical(config.packageFiles.map(({file,sha256})=>({file,sha256})))&&config.packageFiles.length===43,'corrected CUDA package identities changed');
  await confined(config.root);requireThat(await absent(config.hfHome),'new build private HF destination already exists');
  requireThat((await lstat(from)).dev===(await lstat(config.root)).dev,'stopped CUDA private HF is not on the new build volume');
  const before=await privateHfIntegrity({...prior,hfHome:from});
  requireThat(canonical(before)===canonical(transfer.after),'stopped CUDA private package bytes changed');
  await rename(from,config.hfHome);
  const after=await privateHfIntegrity(config);requireThat(canonical(after)===canonical(before),'cross-candidate package ownership transfer changed bytes');
  const receipt={operation:'same-volume-stopped-CUDA-to-corrected-build',old:{...old,artifactIdentity},current:identity(config),sourceContinuity:continuity,from,to:config.hfHome,priorRuntime:await fileIdentity(resolve(artifactRoot,'runtime.json')),priorTransfer:await fileIdentity(resolve(artifactRoot,'hf-transfer.json')),priorPreparation:await fileIdentity(resolve(artifactRoot,'cuda-package-preparation.json')),parentTransfer:await fileIdentity(parentPath),before,after};
  await writeFile(resolve(config.root,'cuda-rebase-hf-transfer.json'),JSON.stringify(receipt,null,2)+'\n',{flag:'wx',mode:0o600});return receipt;
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
export async function adoptOriginalPrivateHf(config,installedInventory,inventoryIdentity,originalRunId,originalAttempt) {
  owned(config,'execute');
  requireThat(config.backend==='cuda'&&Number.isSafeInteger(originalRunId)&&originalRunId>0&&Number.isSafeInteger(originalAttempt)&&originalAttempt>0,'only the original stopped CUDA build HF may be adopted');
  const from=resolve(installedInventory.hfHome),fromRoot=dirname(from);
  requireThat(from===resolve(fromRoot,'hf')&&basename(fromRoot)===`moss-build-${originalRunId}-${originalAttempt}-cuda`,'original private CUDA build root escaped the approved durable namespace');
  requireThat(canonical(installedInventory.files.map(({file,byteLength,sha256})=>({file,byteLength,sha256})))===canonical(config.packageFiles.map(({file,byteLength,sha256})=>({file,byteLength,sha256}))) || canonical(installedInventory.files.map(({file,sha256})=>({file,sha256})))===canonical(config.packageFiles.map(({file,sha256})=>({file,sha256}))),'original installed CUDA package inventory differs from campaign');
  await confined(config.root);requireThat(await absent(config.hfHome),'new private CUDA execution HF destination already exists');
  const owner=await json(resolve(fromRoot,'hf-owner.json')),runtime=await json(resolve(fromRoot,'runtime.json'));
  owned(runtime,'build');requireThat(allowedPrivateParent(dirname(fromRoot),runtime.sourceRoot)&&runtime.root===fromRoot&&runtime.hfHome===from&&owner.root===fromRoot&&owner.hfHome===from&&canonical(identity(runtime))===canonical(identity(config))&&canonical(Object.fromEntries(Object.keys(identity(config)).map(key=>[key,owner[key]])))===canonical(identity(config)),'original sealed CUDA HF belongs to another candidate/source');
  requireThat(canonical(await fileIdentity(resolve(fromRoot,'runtime.json')))===canonical(owner.runtime),'original sealed CUDA runtime bytes changed');
  requireThat((await lstat(from)).dev===(await lstat(config.root)).dev,'original private CUDA HF is not on the execution volume');
  const before=await privateHfIntegrity({...config,hfHome:from});
  requireThat(canonical(before)===canonical(owner.files)&&canonical(before)===canonical(installedInventory.files.map(({file,byteLength,sha256})=>({file,byteLength,sha256}))),'original sealed CUDA package bytes changed');
  await rename(from,config.hfHome);
  const after=await privateHfIntegrity(config);requireThat(canonical(after)===canonical(before),'same-volume CUDA ownership transfer changed package bytes');
  const receipt={...identity(config),operation:'same-volume-original-stopped-CUDA-HF-adoption',from,to:config.hfHome,originalRunId,originalAttempt,originalPackages:inventoryIdentity,originalOwner:await fileIdentity(resolve(fromRoot,'hf-owner.json')),before,after};
  await writeFile(resolve(config.root,'hf-transfer.json'),JSON.stringify(receipt,null,2)+'\n',{flag:'wx',mode:0o600});return receipt;
}
export async function adoptStoppedExecutionPrivateHf(config,artifactRoot,inventoryIdentity,originalRunId,originalAttempt,runId,attempt,artifactId) {
  owned(config,'execute');
  requireThat(config.backend==='cuda'&&[originalRunId,originalAttempt,runId,attempt,artifactId].every(value=>Number.isSafeInteger(value)&&value>0)&&runId!==originalRunId,'stopped CUDA adoption requires distinct exact run and artifact identities');
  const prior=await json(resolve(artifactRoot,'runtime.json'));
  const fromRoot=prior.root,from=prior.hfHome;
  requireThat(basename(fromRoot)===`moss-execute-${runId}-${attempt}-cuda`&&from===resolve(fromRoot,'hf')&&prior.backend==='cuda'&&prior.candidateId===config.candidateId&&prior.sourceSha===config.sourceSha&&prior.inferenceSha===config.inferenceSha,'prior CUDA artifact names another private execution');
  owned(prior,'execute');
  const recorded=await json(resolve(artifactRoot,'hf-transfer.json')),prepared=await json(resolve(artifactRoot,'cuda-package-preparation.json'));
  const initial=recorded.operation==='same-volume-original-stopped-CUDA-HF-adoption',chained=recorded.operation==='same-volume-stopped-CUDA-HF-adoption';
  requireThat((initial||chained)&&recorded.to===from&&recorded.originalRunId===originalRunId&&recorded.originalAttempt===originalAttempt&&canonical(recorded.before)===canonical(recorded.after)&&canonical(recorded.after.map(({file,sha256})=>({file,sha256})))===canonical(config.packageFiles.map(({file,sha256})=>({file,sha256})))&&canonical(recorded.originalPackages)===canonical(inventoryIdentity)&&canonical(Object.fromEntries(Object.keys(identity(config)).map(key=>[key,recorded[key]])))===canonical(identity(config)),'prior CUDA artifact does not prove original sealed package transfer');
  let origin;
  if(initial) {
    requireThat(recorded.from===resolve(dirname(recorded.from),'hf')&&basename(dirname(recorded.from))===`moss-build-${originalRunId}-${originalAttempt}-cuda`,'prior CUDA original build owner path changed');
    origin={originalBuildHf:recorded.from,originalOwner:recorded.originalOwner,initialTransfer:await fileIdentity(resolve(artifactRoot,'hf-transfer.json'))};
  } else {
    requireThat([recorded.priorRunId,recorded.priorAttempt,recorded.priorArtifactId].every(value=>Number.isSafeInteger(value)&&value>0)&&recorded.from===resolve(dirname(recorded.from),'hf')&&basename(dirname(recorded.from))===`moss-execute-${recorded.priorRunId}-${recorded.priorAttempt}-cuda`,'prior CUDA execution owner path changed');
    const top=resolve(artifactRoot,'stopped-cuda-hf-transfer.json'),legacy=resolve(artifactRoot,'evidence/external/stopped-cuda-hf-transfer.json');
    const hasTop=!await absent(top),hasLegacy=!await absent(legacy);
    requireThat(hasTop!==hasLegacy,'stopped CUDA predecessor has missing or ambiguous archived transfer');
    const parentPath=hasTop?top:legacy,parent=await json(parentPath);
    requireThat(canonical(recorded.priorTransfer)===canonical(await fileIdentity(parentPath))&&['same-volume-original-stopped-CUDA-HF-adoption','same-volume-stopped-CUDA-HF-adoption'].includes(parent.operation)&&parent.to===recorded.from&&parent.originalRunId===originalRunId&&parent.originalAttempt===originalAttempt&&canonical(parent.originalPackages)===canonical(inventoryIdentity)&&canonical(parent.before)===canonical(parent.after)&&canonical(parent.after)===canonical(recorded.before)&&canonical(Object.fromEntries(Object.keys(identity(config)).map(key=>[key,parent[key]])))===canonical(identity(config)),'stopped CUDA predecessor transfer is missing or changed');
    const parentOrigin=parent.origin??{originalBuildHf:parent.from,originalOwner:parent.originalOwner,initialTransfer:recorded.priorTransfer};
    requireThat(canonical(recorded.origin)===canonical(parentOrigin),'stopped CUDA origin chain changed');origin=recorded.origin;
  }
  requireThat(origin.originalBuildHf===resolve(dirname(origin.originalBuildHf),'hf')&&basename(dirname(origin.originalBuildHf))===`moss-build-${originalRunId}-${originalAttempt}-cuda`&&canonical(origin.originalOwner)===canonical(await fileIdentity(resolve(dirname(origin.originalBuildHf),'hf-owner.json'))),'original sealed owner receipt changed after CUDA transfer');
  requireThat(prepared.mode===(initial?'original-sealed-HF':'stopped-execution-HF')&&prepared.originalRunId===originalRunId&&prepared.originalAttempt===originalAttempt&&prepared.candidateId===config.candidateId&&prepared.sourceSha===config.sourceSha&&prepared.inferenceSha===config.inferenceSha&&canonical(prepared.originalNativePackages)===canonical(inventoryIdentity)&&canonical(prepared.transfer)===canonical(await fileIdentity(resolve(artifactRoot,'hf-transfer.json')))&&(!chained||prepared.stoppedArtifact?.runId===recorded.priorRunId&&prepared.stoppedArtifact?.attempt===recorded.priorAttempt&&prepared.stoppedArtifact?.artifactId===recorded.priorArtifactId),'prior CUDA package preparation differs from its sealed transfer');
  for(const name of ['runtime.json','hf-transfer.json','cuda-package-preparation.json'])requireThat(canonical(await fileIdentity(resolve(artifactRoot,name)))===canonical(await fileIdentity(resolve(fromRoot,name))),`stopped CUDA ${name} differs from downloaded artifact`);
  await confined(config.root);requireThat(await absent(config.hfHome),'new private CUDA destination already exists');
  requireThat((await lstat(from)).dev===(await lstat(config.root)).dev,'stopped CUDA private HF is not on the execution volume');
  const before=await privateHfIntegrity({...config,hfHome:from});requireThat(canonical(before)===canonical(recorded.after),'stopped CUDA package bytes changed');
  requireThat(await absent(config.hfHome),'new private CUDA destination appeared before transfer');
  await rename(from,config.hfHome);
  const after=await privateHfIntegrity(config);requireThat(canonical(after)===canonical(before),'stopped CUDA ownership transfer changed package bytes');
  const receipt={...identity(config),operation:'same-volume-stopped-CUDA-HF-adoption',from,to:config.hfHome,originalRunId,originalAttempt,priorRunId:runId,priorAttempt:attempt,priorArtifactId:artifactId,priorTransfer:await fileIdentity(resolve(artifactRoot,'hf-transfer.json')),priorRuntime:await fileIdentity(resolve(artifactRoot,'runtime.json')),origin,originalPackages:inventoryIdentity,before,after};
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
export function mossArtifactRoute(key,path,rawBase,platform=process.platform) {
  const supplement=key==='MOSS_PRIOR_MLX'||key==='MOSS_SUPPLEMENT';
  const external=!supplement&&key!=='MOSS_BUNDLE'&&platform==='darwin'&&dirname(resolve(path))===MAC_MOSS_BASE;
  const leaf=basename(path),valid=supplement?new RegExp(`^moss-h27-${key==='MOSS_PRIOR_MLX'?'prior':'hidden'}-[0-9]+-[0-9]+$`).test(leaf):/^moss-(?:build|builds|execute|review|bundle)-[0-9]+-[0-9]+(?:-(?:build|execute|review|mlx|cuda))*$/.test(leaf);
  requireThat((dirname(resolve(path))===rawBase||external)&&valid,'MOSS artifact path escaped durable namespace');
  return {external,leaf:basename(path)};
}
export async function normalizeMossEnvironment(env) {
  const workspace=await realpath(env.GITHUB_WORKSPACE),base=resolve(workspace,'..','moss-sc-26752'),rawBase=resolve(env.GITHUB_WORKSPACE,'..','moss-sc-26752'),result={};
  await mkdir(base,{recursive:true});await confined(base,{descendants:false});
  for(const key of ['MOSS_ROOT','MOSS_BUILDS','MOSS_OLD_BUILDS','MOSS_BUNDLE','MOSS_PRIOR_MLX','MOSS_SUPPLEMENT','MOSS_STOPPED_CUDA'])if(env[key]) {
    const {external}=mossArtifactRoute(key,env[key],rawBase);
    if(external) {
      const mount=resolve(MAC_MOSS_BASE,'..');await confined(mount,{descendants:false});
      requireThat((await lstat(mount)).isDirectory()&&(await lstat(mount)).dev!==(await lstat(dirname(mount))).dev,'MOSS external storage must be a mounted separate volume');
      await mkdir(MAC_MOSS_BASE,{recursive:true,mode:0o700});await confined(MAC_MOSS_BASE,{descendants:false});
    }
    result[key]=resolve(external?MAC_MOSS_BASE:base,basename(env[key]));
  }
  result.MOSS_SOURCE=resolve(workspace,'source');result.MOSS_INFERENCE=resolve(workspace,'inference');
  if(env.MOSS_UPLOAD_PATH)result.MOSS_UPLOAD_PATH=await realpath(env.MOSS_UPLOAD_PATH);
  requireThat(Object.values(result).every(value=>!/[\r\n]/.test(value)),'invalid multiline Actions path');
  return result;
}
export async function exportMossEnvironment(env) {
  const result=await normalizeMossEnvironment(env);await appendFile(env.GITHUB_ENV,Object.entries(result).map(([key,value])=>`${key}=${value}\n`).join(''));return result;
}
