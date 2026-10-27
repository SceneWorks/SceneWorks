// Shared read-only validation for the original frozen reference payload.
// This module has no CLI entrypoint, so case-suite dispatch cannot cycle through
// the reference controller's top-level await.
import {readFile,lstat,readdir} from 'node:fs/promises';
import {createReadStream} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve,dirname,sep} from 'node:path';

export const PAYLOAD_SHA='269f9dc9e9d7852d131457aaa6a1fe9c558a9dffee71b9fd57a2026f5663b9df';
export const STAGED='/Users/michael/.codex/epic-runs/sc-26752/upstream-reference-actions-r15/payload';
const PREP='/Users/michael/.codex/epic-runs/sc-26752/upstream-reference-prep-r15';
const check=(value,message)=>{if(!value)throw Error(message);};
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
async function fileHash(path){const digest=createHash('sha256');for await(const chunk of createReadStream(path))digest.update(chunk);return digest.digest('hex');}
export function safe(root,file){check(typeof file==='string'&&file.length&&!file.includes('\\')&&!file.split('/').some(part=>!part||part==='.'||part==='..')&&!file.startsWith('/'),'unsafe payload path');const path=resolve(root,file);check(path.startsWith(resolve(root)+sep),'payload escape');return path;}
async function noLinks(root){for(let path=resolve(root);;path=dirname(path)){check(!(await lstat(path)).isSymbolicLink(),'linked ancestor refused');if(dirname(path)===path)break;}}
export async function verifyPayload(root,expected=PAYLOAD_SHA){
  await noLinks(root);const bytes=await readFile(resolve(root,'payload-manifest.json'));check(hash(bytes)===expected,'payload manifest SHA mismatch');
  const manifest=JSON.parse(bytes);check(manifest.kind==='sc-26752-frozen-upstream-base-single-reference/v1'&&manifest.actualNativeSource==='a0692191b09813fab859659bf382bd7c91a60140'&&manifest.nativePin==='857f7a1996e3678a0773c3509ac0087522c19ce6','reference provenance mismatch');
  check(Array.isArray(manifest.files)&&manifest.files.length===45,'fixed payload closure required');const wanted=new Set(['payload-manifest.json']);
  for(const item of manifest.files){const path=safe(root,item.file);check(!wanted.has(item.file),'duplicate payload file');wanted.add(item.file);await noLinks(path);const info=await lstat(path);check(info.isFile()&&info.size===item.byteLength&&await fileHash(path)===item.sha256,'payload bytes mismatch');}
  const walk=async(dir,prefix='')=>{for(const entry of await readdir(dir,{withFileTypes:true})){const file=prefix+entry.name;check(!entry.isSymbolicLink(),'payload link refused');if(entry.isDirectory())await walk(resolve(dir,entry.name),file+'/');else check(wanted.has(file),'unlisted payload file');}};await walk(root);
  return manifest;
}
export function validateRuntime(runtime){
  check(runtime.launcher===PREP+'/venv/bin/python'&&runtime.versions?.torch==='2.9.1'&&runtime.versions?.transformers==='5.5.3'&&runtime.versions?.diffusers==='0.38.0','fixed prepared runtime required');
  check(runtime.files.length>1000&&runtime.files.length<25000,'bounded runtime inventory required');
  for(const file of runtime.files)check(['/opt/homebrew/',PREP+'/venv/','/Users/michael/Library/Python/3.14/'].some(base=>file.path.startsWith(base))&&resolve(file.path)===file.path&&/^[a-f0-9]{64}$/.test(file.sha256)&&Number.isSafeInteger(file.byteLength)&&file.byteLength>=0,'runtime file escaped authenticated host environment');
}
