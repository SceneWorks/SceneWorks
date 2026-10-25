// One authenticated frozen-upstream Base/single diagnostic. No native evidence is modified.
import {readFile,writeFile,appendFile,mkdir,lstat,realpath,readdir,cp,symlink} from 'node:fs/promises';
import {createReadStream,createWriteStream} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve,dirname,sep} from 'node:path';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
export const PAYLOAD_SHA='269f9dc9e9d7852d131457aaa6a1fe9c558a9dffee71b9fd57a2026f5663b9df';
export const STAGED='/Users/michael/.codex/epic-runs/sc-26752/upstream-reference-actions-r15/payload';
const PREP='/Users/michael/.codex/epic-runs/sc-26752/upstream-reference-prep-r15';
const BASE='/Volumes/Models/sceneworks-terminal-sc-26752',ADAPTER=BASE+'/moss-upstream-reference-r15-retained-adapter';
const check=(value,message)=>{if(!value)throw Error(message);};
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
async function fileHash(path){const digest=createHash('sha256');for await(const chunk of createReadStream(path))digest.update(chunk);return digest.digest('hex');}
export function safe(root,file){check(typeof file==='string'&&file.length&&!file.includes('\\')&&!file.split('/').some(part=>!part||part==='.'||part==='..')&&!file.startsWith('/'),'unsafe payload path');const path=resolve(root,file);check(path.startsWith(resolve(root)+sep),'payload escape');return path;}
export function envelope(raw,ref,sha,backend){
  const value=JSON.parse(raw);check(value&&Object.keys(value).sort().join(',')==='artifact_id,artifact_run_id,phase','closed reference envelope required');
  check(/^[a-f0-9]{40}$/.test(ref)&&ref===sha&&backend==='mlx','exact dispatch source and mlx required');
  check(['upload','execute'].includes(value.phase),'unknown reference phase');
  for(const key of ['artifact_id','artifact_run_id'])check(value.phase==='upload'?value[key]===null:typeof value[key]==='string'&&/^[1-9][0-9]{5,19}$/.test(value[key]),'exact artifact identity required');
  return value;
}
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
async function execute(root,payload){
  const manifest=await verifyPayload(payload),runtime=JSON.parse(await readFile(resolve(payload,'runtime-files.json')));validateRuntime(runtime);
  check(await realpath(runtime.launcher)===runtime.executable,'prepared Python launcher changed');
  for(const item of runtime.files){const info=await lstat(item.path);check(info.isFile()&&info.size===item.byteLength&&await fileHash(item.path)===item.sha256,'prepared runtime file changed');}
  const plan=JSON.parse(await readFile(resolve(payload,'snapshot-mapping-plan.json')));check(plan.actualSourceSha===manifest.actualNativeSource&&plan.nativeSha===manifest.nativePin&&plan.plannedMappings.length===23,'fixed model mapping required');
  const view=resolve(root,'snapshot-view');await mkdir(view,{mode:0o700});const identities=[];
  for(const item of plan.plannedMappings){
    check(item.source===(item.plannedComponent==='snapshot'?safe('/Volumes/Models/huggingface',item.file):safe(ADAPTER,item.plannedRelativePath)),'model source escaped fixed original/retained roots');
    const info=await lstat(await realpath(item.source));check(info.isFile()&&info.size===item.byteLength&&await fileHash(item.source)===item.sha256,'frozen model identity mismatch');identities.push({file:item.file,byteLength:item.byteLength,sha256:item.sha256,source:item.source});
    if(item.plannedComponent==='snapshot'){const target=safe(view,item.plannedRelativePath);await mkdir(dirname(target),{recursive:true,mode:0o700});await symlink(item.source,target);}
    else check(item.plannedComponent==='adapter'&&info.nlink===1,'retained adapter must be independently owned');
  }
  await writeFile(resolve(root,'authentication.json'),JSON.stringify({payloadManifestSha256:PAYLOAD_SHA,actualNativeSource:manifest.actualNativeSource,nativePin:manifest.nativePin,wrapperSha256:manifest.wrapperSha256,runtimeFiles:runtime.files.length,versions:runtime.versions,models:identities,scope:'reference diagnostic only; native evidence unchanged'},null,2)+'\n',{flag:'wx'});
  const output=resolve(root,'upstream-base-single.png'),log=createWriteStream(resolve(root,'execution.log'),{flags:'wx'});
  await new Promise((done,fail)=>{const child=spawn(runtime.launcher,['-B',resolve(payload,'reference_diagnostic.py'),'--snapshot-view',view,'--device','mps','--output',output,'--execute-reviewed'],{cwd:payload,env:{...process.env,PYTHONPATH:'',PYTHONDONTWRITEBYTECODE:'1',PYTHONPYCACHEPREFIX:resolve(root,'pycache'),HF_HOME:resolve(root,'hf'),HF_MODULES_CACHE:resolve(root,'modules'),TORCH_HOME:resolve(root,'torch'),XDG_CACHE_HOME:resolve(root,'cache'),HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1',PYTORCH_ENABLE_MPS_FALLBACK:'0',CUDA_VISIBLE_DEVICES:'',OMP_NUM_THREADS:'1'},stdio:['ignore','pipe','pipe']});child.stdout.pipe(log,{end:false});child.stderr.pipe(log,{end:false});child.on('error',fail);child.on('close',code=>{log.end();code===0?done():fail(Error(`fixed reference exited ${code}`));});});
}
async function main(){
  check(process.env.GITHUB_ACTIONS==='true'&&process.env.GITHUB_REPOSITORY==='SceneWorks/SceneWorks'&&process.platform==='darwin','registered Mac Actions route only');
  const value=envelope(process.env.MOSS_REFERENCE,process.env.SOURCE_REF,process.env.GITHUB_SHA,process.env.INPUT_BACKEND);
  const root=resolve(BASE,`moss-upstream-reference-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}`);check(/^[1-9][0-9]+$/.test(process.env.GITHUB_RUN_ID)&&/^[1-9][0-9]*$/.test(process.env.GITHUB_RUN_ATTEMPT),'actual Actions identity required');
  if(process.argv[2]==='validate'){
    if(value.phase==='upload')await verifyPayload(STAGED);
    await appendFile(process.env.GITHUB_OUTPUT,Object.entries({...value,root,payload:STAGED}).map(([key,value])=>`${key}=${value??''}\n`).join(''));return;
  }
  check(process.argv[2]==='execute'&&value.phase==='execute','fixed execute phase required');
  await noLinks(BASE);check((await lstat('/Volumes/Models')).dev!==(await lstat('/Volumes')).dev,'mounted Models volume required');await mkdir(root,{mode:0o700});
  const download=resolve(process.env.RUNNER_TEMP,`moss-upstream-reference-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}`);await verifyPayload(download);await cp(download,resolve(root,'payload'),{recursive:true,errorOnExist:true,force:false});await execute(root,resolve(root,'payload'));
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))await main();
