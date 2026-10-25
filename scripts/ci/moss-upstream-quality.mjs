// Fixed three-cell upstream quality suite. Base/single remains its original run.
import {readFile,writeFile,appendFile,mkdir,lstat,stat,realpath,readdir,cp,symlink} from 'node:fs/promises';
import {createReadStream,createWriteStream} from 'node:fs';
import {createHash} from 'node:crypto';
import {resolve,dirname} from 'node:path';
import {spawn} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {safe,verifyPayload as verifyPriorPayload,validateRuntime} from './moss-upstream-reference-common.mjs';

export const PRIOR_PAYLOAD='/Users/michael/.codex/epic-runs/sc-26752/upstream-reference-actions-r15/payload';
export const CASES_STAGED='/Users/michael/.codex/epic-runs/sc-26752/h22-upstream-comparison-prep/payload';
export const CASES_MANIFEST_SHA256='af8ff343a598772bbe114164651b80225fc6f62b66ac4da26e5cb2071570ca57';
const BASE='/Volumes/Models/sceneworks-terminal-sc-26752';
const INPUTS='/Users/michael/.codex/epic-runs/sc-26752/quality-inputs-v1/inputs.json';
const INPUTS_SHA='5fba2bfa980357e5e94e5bfb37523b120cec90ccbdc7e8836d553a20002fa458';
const PRIOR_SHA='269f9dc9e9d7852d131457aaa6a1fe9c558a9dffee71b9fd57a2026f5663b9df';
const BASE_PLAN_SHA='d76c205e2c59676cea6b7d22b55c153a2888d76a5f43577578cc08d61cd51b22';
const CONFIG_SHA='896fa02e2ada5d939617c63785abcb94125ee1d423692aafbec0399894278a7a';
const NOISE_SHA='92e4addfaab4f8f5738c6a29ab2da1a56e2b204726b83250721edfec09608d82';
const AMENDMENT_SHA='596ad6038ae5e1ab211e646255b9a1f67be1cdd3a79ea07a741fce506d414fe9';
const IDS=['base-four','sft-single','sft-four'];
const SHA=/^[a-f0-9]{64}$/;
const check=(ok,message)=>{if(!ok)throw Error(message);};
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
async function fileHash(path){const digest=createHash('sha256');for await(const chunk of createReadStream(path))digest.update(chunk);return digest.digest('hex');}
async function noLinks(root){for(let path=resolve(root);;path=dirname(path)){check(!(await lstat(path)).isSymbolicLink(),'linked ancestor refused');if(dirname(path)===path)break;}}
export function envelope(raw,ref,sha,backend){
  const value=JSON.parse(raw);
  check(value&&Object.keys(value).sort().join(',')==='artifact_id,artifact_run_id,phase'&&['upload-cases','execute-cases'].includes(value.phase),'closed quality suite envelope required');
  check(/^[a-f0-9]{40}$/.test(ref)&&ref===sha&&backend==='mlx','exact quality suite source and mlx required');
  if(value.phase==='upload-cases')check(value.artifact_id===null&&value.artifact_run_id===null,'upload cannot consume artifact IDs');
  else check([value.artifact_id,value.artifact_run_id].every(id=>typeof id==='string'&&/^[1-9][0-9]{5,19}$/.test(id)),'execute requires exact uploaded artifact identity');
  return value;
}
export async function verifyCasePayload(root,expected=CASES_MANIFEST_SHA256){
  await noLinks(root);
  const bytes=await readFile(resolve(root,'payload-manifest.json'));
  check(hash(bytes)===expected,'quality payload manifest SHA mismatch');
  const manifest=JSON.parse(bytes);
  check(manifest.kind==='sc-26752-upstream-quality-cases-payload/v1'&&manifest.priorPayloadManifestSha256===PRIOR_SHA&&Array.isArray(manifest.files)&&manifest.files.length>=12,'fixed quality payload closure required');
  const wanted=new Set(['payload-manifest.json']);
  for(const item of manifest.files){
    const path=safe(root,item.file);check(!wanted.has(item.file)&&SHA.test(item.sha256)&&Number.isSafeInteger(item.byteLength)&&item.byteLength>0,'duplicate or malformed quality payload file');wanted.add(item.file);
    await noLinks(path);const info=await lstat(path);check(info.isFile()&&info.size===item.byteLength&&await fileHash(path)===item.sha256,'quality payload bytes mismatch');
  }
  const walk=async(dir,prefix='')=>{for(const entry of await readdir(dir,{withFileTypes:true})){const file=prefix+entry.name;check(!entry.isSymbolicLink(),'quality payload link refused');if(entry.isDirectory())await walk(resolve(dir,entry.name),file+'/');else check(wanted.has(file),'unlisted quality payload file');}};await walk(root);
  const inputsBytes=await readFile(INPUTS);check(hash(inputsBytes)===INPUTS_SHA,'frozen quality input manifest changed');
  const inputs=JSON.parse(inputsBytes),assets=Object.fromEntries(inputs.assets.map(item=>[item.id,item]));
  const cases=JSON.parse(await readFile(resolve(root,'cases.json'))),models=JSON.parse(await readFile(resolve(root,'model-mappings.json')));
  check(cases.kind==='sc-26752-upstream-quality-cases/v1'&&cases.priorPayloadManifestSha256===PRIOR_SHA&&Array.isArray(cases.cases)&&cases.cases.length===3&&cases.cases.every((item,index)=>item.id===IDS[index]),'exact three ordered quality cases required');
  for(const item of cases.cases){
    const four=item.id.endsWith('four'),variant=item.id.split('-')[0],task=inputs[four?'four_case':'single_case'];
    check(item.variant===variant&&item.rowId===`quality-mlx-${item.id}`&&item.prompt===task.prompt&&item.negativePrompt===''&&item.width===task.width&&item.height===task.height&&item.seed===173&&item.steps===30&&item.cfg===4&&item.inferenceShift===5&&item.referenceMaxEdge===512&&item.useKvCache===true,'case request/settings differ from frozen native cell');
    check(Array.isArray(item.refs)&&item.refs.length===task.references.length&&item.refs.every((ref,index)=>ref.id===task.references[index]&&ref.file===`inputs/${ref.id}.png`&&ref.sha256===assets[ref.id]?.sha256),'ordered frozen input identities changed');
    check(item.noise?.file==='inputs/native-seed173-f32le.bin'&&item.noise.sha256===NOISE_SHA&&item.noise.byteLength===786432&&JSON.stringify(item.noise.shape)===JSON.stringify(four?[1,128,48,32]:[1,128,32,48]),'case noise must be exact C-order seed173 shape');
  }
  const oldPlan=JSON.parse(await readFile(resolve(PRIOR_PAYLOAD,'snapshot-mapping-plan.json')));
  check(models.kind==='sc-26752-upstream-quality-model-mappings/v1'&&models.priorPayloadManifestSha256===PRIOR_SHA&&models.base?.planSha256===BASE_PLAN_SHA&&JSON.stringify(models.base.plannedMappings)===JSON.stringify(oldPlan.plannedMappings)&&models.sft?.length===23&&models.changedShards?.length===3&&models.sharedMappings?.length===20,'model mapping is not the fixed Base/SFT source closure');
  const byPath=new Map(oldPlan.plannedMappings.map(item=>[item.plannedRelativePath,item]));
  const changed=new Set(models.changedShards.map(item=>typeof item==='string'?item:item.plannedRelativePath));
  const shared=new Set(models.sharedMappings.map(item=>typeof item==='string'?item:item.plannedRelativePath));
  check(changed.size===3&&shared.size===20&&[...changed].every(path=>/transformer.*\.safetensors$/.test(path))&&![...changed].some(path=>shared.has(path))&&new Set(models.sft.map(item=>item.plannedRelativePath)).size===23&&models.sft.every(item=>{
    const prior=byPath.get(item.plannedRelativePath);
    return prior&&((changed.has(item.plannedRelativePath)&&item.file.startsWith('hub/models--OpenMOSS-Team--MOSS-Image-SFT/snapshots/fc5de0d56fc910cbca6f6625a70f5f582b26deb1/')&&item.file.endsWith('/'+item.plannedRelativePath)&&item.sha256!==prior.sha256)||(shared.has(item.plannedRelativePath)&&JSON.stringify(item)===JSON.stringify(prior)));
  }),'SFT model mapping changed more than its three transformer shards');
  check(hash(await readFile(resolve(root,'reference-config.plan.json')))===CONFIG_SHA,'frozen reference configuration changed');
  check(hash(await readFile(resolve(root,'quality-amendment.md')))===AMENDMENT_SHA,'approved quality amendment changed in case payload');
  return {manifest,cases,models};
}
const boundStat=info=>Object.fromEntries(['dev','ino','nlink','size','mtimeNs','ctimeNs'].map(key=>[key,String(info[key])]));
export async function authenticateModels(root,payload,plans){
  const priorFile=resolve(BASE,'moss-upstream-reference-40191507328-1/authentication.json');
  const prior=JSON.parse(await readFile(resolve(payload,'prior-authentication-binding.json'))),newSft=JSON.parse(await readFile(resolve(payload,'new-sft-shards-authentication.json')));
  check(prior.kind==='sc-26752-prior-reference-stat-binding/v1'&&prior.priorRunId==='40191507328'&&prior.priorAttempt===1&&prior.authenticationSha256==='a0e0c279a6e661e1703c5c4f7423d9f7ff940d5bfdcee850f152e9a8850e08f3'&&await fileHash(priorFile)===prior.authenticationSha256&&prior.files?.length===23,'original Base/adapter authentication binding changed');
  const priorAuth=JSON.parse(await readFile(priorFile));
  check(priorAuth.payloadManifestSha256===PRIOR_SHA&&priorAuth.models?.length===23&&newSft.kind==='sc-26752-new-sft-original-shard-authentication/v1'&&newSft.files?.length===3,'original and SFT shard authentication records differ');
  const original=new Map(prior.files.map(item=>[item.file,item])),additional=new Map(newSft.files.map(item=>[item.file,item]));
  const identity=new Map(),views={},observed=[];
  for(const [variant,plan] of Object.entries(plans)){
    const view=resolve(root,`snapshot-${variant}`);await mkdir(view,{mode:0o700});views[variant]=view;
    for(const item of plan){
      const record=original.get(item.file)??additional.get(item.file),source=await realpath(item.source),info=await stat(source,{bigint:true});
      check(record&&info.isFile()&&source===record.realPath&&record.source===item.source&&record.file===item.file&&record.byteLength===item.byteLength&&record.sha256===item.sha256&&JSON.stringify(boundStat(info))===JSON.stringify(record.stat),'frozen model source/stat identity differs from prior authenticated or one-time SFT hash');
      if(additional.has(item.file))check(record.streamCount===1&&record.authenticatedAt&&record.source.includes('/models--OpenMOSS-Team--MOSS-Image-SFT/'),'new SFT shard lacks one-time streaming proof');
      else check(priorAuth.models.some(auth=>auth.source===item.source&&auth.file===item.file&&auth.sha256===item.sha256&&auth.byteLength===item.byteLength)&&BigInt(record.stat.ctimeNs)<BigInt(prior.authenticationBirthtimeNs),'original model source differs from Base/single authenticated bytes');
      const key=`${source}:${item.sha256}`;
      if(!identity.has(key)){identity.set(key,{source,file:item.file,byteLength:item.byteLength,sha256:item.sha256,method:additional.has(item.file)?'sealed-once-streamed-sha256-plus-stable-stat':'prior-authenticated-sha256-plus-stable-stat'});observed.push({source,stat:record.stat});}
      if(item.plannedComponent==='snapshot'){
        const target=safe(view,item.plannedRelativePath);await mkdir(dirname(target),{recursive:true,mode:0o700});await symlink(item.source,target);
      }else check(item.plannedComponent==='adapter'&&source.startsWith(BASE+'/moss-upstream-reference-r15-retained-adapter/')&&info.nlink===1n,'retained adapter identity/ownership changed');
    }
  }
  return {views,identities:[...identity.values()],assertStable:async()=>{for(const item of observed)check(JSON.stringify(boundStat(await stat(item.source,{bigint:true})))===JSON.stringify(item.stat),'frozen model source changed during quality suite');}};
}
async function execute(root,payload){
  const {cases,models}=await verifyCasePayload(payload);await verifyPriorPayload(PRIOR_PAYLOAD);
  const runtime=JSON.parse(await readFile(resolve(PRIOR_PAYLOAD,'runtime-files.json')));validateRuntime(runtime);
  check(await realpath(runtime.launcher)===runtime.executable,'prepared Python launcher changed');
  for(const item of runtime.files){const info=await lstat(item.path);check(info.isFile()&&info.size===item.byteLength&&await fileHash(item.path)===item.sha256,'prepared runtime file changed');}
  const {views,identities,assertStable}=await authenticateModels(root,payload,{base:models.base.plannedMappings,sft:models.sft});
  const wrapper=fileURLToPath(new URL('./moss-upstream-quality.py',import.meta.url)),wrapperSha256=await fileHash(wrapper);
  await writeFile(resolve(root,'authentication.json'),JSON.stringify({qualityCasesManifestSha256:CASES_MANIFEST_SHA256,priorPayloadManifestSha256:PRIOR_SHA,toolingSha:process.env.GITHUB_SHA,wrapperSha256,nativeProductSha:'42a8aff9852d01e0506df8afc466c0024b6eed8c',nativeInferenceSha:'857f7a1996e3678a0773c3509ac0087522c19ce6',runtimeVersions:runtime.versions,models:identities,cases:IDS,scope:'upstream quality comparison only; H21 native evidence unchanged'},null,2)+'\n',{flag:'wx'});
  const outputs=[];
  for(const item of cases.cases){
    const output=resolve(root,`upstream-${item.id}.png`),log=createWriteStream(resolve(root,`upstream-${item.id}.log`),{flags:'wx'});
    const args=['-B',wrapper,'--prior-payload',PRIOR_PAYLOAD,'--case-payload',payload,'--case',item.id,'--snapshot-view',views[item.variant],'--device','mps','--output',output,'--execute-reviewed'];
    await new Promise((done,fail)=>{const child=spawn(runtime.launcher,args,{cwd:payload,env:{...process.env,PYTHONPATH:'',PYTHONDONTWRITEBYTECODE:'1',PYTHONPYCACHEPREFIX:resolve(root,'pycache'),HF_HOME:resolve(root,'hf'),HF_MODULES_CACHE:resolve(root,'modules'),TORCH_HOME:resolve(root,'torch'),XDG_CACHE_HOME:resolve(root,'cache'),HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1',PYTORCH_ENABLE_MPS_FALLBACK:'0',CUDA_VISIBLE_DEVICES:'',OMP_NUM_THREADS:'1'},stdio:['ignore','pipe','pipe']});child.stdout.pipe(log,{end:false});child.stderr.pipe(log,{end:false});child.on('error',fail);child.on('close',code=>{log.end();code===0?done():fail(Error(`upstream ${item.id} exited ${code}`));});});
    outputs.push({caseId:item.id,imageSha256:await fileHash(output),diagnosticSha256:await fileHash(output.replace(/\.png$/,'.diagnostic.json'))});
    await assertStable();
  }
  await writeFile(resolve(root,'suite-summary.json'),JSON.stringify({toolingSha:process.env.GITHUB_SHA,qualityCasesManifestSha256:CASES_MANIFEST_SHA256,outputs},null,2)+'\n',{flag:'wx'});
}
export async function main(){
  check(process.env.GITHUB_ACTIONS==='true'&&process.env.GITHUB_REPOSITORY==='SceneWorks/SceneWorks'&&process.platform==='darwin','registered Mac Actions route only');
  const value=envelope(process.env.MOSS_REFERENCE,process.env.SOURCE_REF,process.env.GITHUB_SHA,process.env.INPUT_BACKEND);
  const root=resolve(BASE,`moss-upstream-quality-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}`);
  check(/^[1-9][0-9]+$/.test(process.env.GITHUB_RUN_ID)&&/^[1-9][0-9]*$/.test(process.env.GITHUB_RUN_ATTEMPT),'actual Actions identity required');
  if(process.argv[2]==='validate'){
    if(value.phase==='upload-cases')await verifyCasePayload(CASES_STAGED);
    await appendFile(process.env.GITHUB_OUTPUT,Object.entries({...value,root,payload:CASES_STAGED}).map(([key,value])=>`${key}=${value??''}\n`).join(''));return;
  }
  check(process.argv[2]==='execute'&&value.phase==='execute-cases','fixed execute-cases phase required');
  await noLinks(BASE);check((await lstat('/Volumes/Models')).dev!==(await lstat('/Volumes')).dev,'mounted Models volume required');await mkdir(root,{mode:0o700});
  const download=resolve(process.env.RUNNER_TEMP,`moss-upstream-reference-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}`);
  await verifyCasePayload(download);await cp(download,resolve(root,'payload'),{recursive:true,errorOnExist:true,force:false});await execute(root,resolve(root,'payload'));
}
