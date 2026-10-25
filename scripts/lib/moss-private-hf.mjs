// Private installed model ownership only; shared HF inputs are never moved or removed.
import { lstat, readdir, rename, rm, writeFile } from 'node:fs/promises';
import { resolve, dirname, basename } from 'node:path';
import { canonical, requireThat, safePath, fileIdentity, json, digest } from './moss-edit-terminal.mjs';
import { inspectTree } from '../ci/moss-owned-cache-maintenance.mjs';

const absent=async path=>{try{await lstat(path);return false;}catch(error){if(error.code==='ENOENT')return true;throw error;}};
const identity=config=>({candidateId:config.candidateId,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,backend:config.backend,packagesSha256:digest(config.packageFiles)});
function owned(config,phase) {
  const match=new RegExp(`^moss-${phase}-([0-9]+)-([0-9]+)-(mlx|cuda)$`).exec(basename(config.root));
  requireThat(match&&match[3]===config.backend&&dirname(config.root)===resolve(config.sourceRoot,'../..','moss-sc-26752')&&config.hfHome===resolve(config.root,'hf')&&config.hfHome!==config.packageSource,'exact current private HF ownership required');
  return match;
}
async function confined(path) {
  for(let cursor=resolve(path);;cursor=dirname(cursor)) {
    const info=await lstat(cursor);requireThat(!info.isSymbolicLink(),'private HF refuses linked ancestors');
    if(dirname(cursor)===cursor)break;
  }
  if(process.platform==='win32')await inspectTree(path); // Reject every Windows reparse tag before traversal.
}
export async function privateHfIntegrity(config) {
  await confined(config.hfHome);
  const visit=async dir=>{for(const name of await readdir(dir)) {
    const path=resolve(dir,name),info=await lstat(path);
    requireThat(!info.isSymbolicLink(),'private HF refuses symlinks/junctions');
    if(info.isDirectory())await visit(path);else requireThat(info.isFile()&&info.nlink===1,'private HF requires exclusive regular files');
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
