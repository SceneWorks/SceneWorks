// Four supplemental product-API measurements. Never changes the frozen product plan.
// The caller supplies its already-running API and existing protocol helpers.
import {readFile,writeFile,mkdir,copyFile} from 'node:fs/promises';
import {resolve,dirname,basename} from 'node:path';
import {fileURLToPath} from 'node:url';

export const QUALITY_NOISE_SHA='92e4addfaab4f8f5738c6a29ab2da1a56e2b204726b83250721edfec09608d82';
const requireThat=(value,message)=>{if(!value)throw Error(message);};
const json=async path=>JSON.parse(await readFile(path));
const save=async(path,value)=>{await mkdir(dirname(path),{recursive:true});await writeFile(path,JSON.stringify(value,null,2)+'\n',{flag:'wx',mode:0o600});};
const encode=encodeURIComponent;
export function controlledQualityRows(plan){
  const rows=plan.rows.filter(row=>row.backend==='cuda'&&row.category==='quality');
  requireThat(rows.length===4&&new Set(rows.map(row=>row.id)).size===4,'exact four CUDA quality rows required');
  return rows.map(original=>{
    const row=structuredClone(original),four=row.refs.length===4;
    requireThat(row.id===`quality-cuda-${row.variant}-${four?'four':'single'}`&&['base','sft'].includes(row.variant)&&JSON.stringify(row.refs)===JSON.stringify(four?['B','L','P','G']:['B']),'frozen CUDA quality row identity/order changed');
    requireThat(row.output==='images'&&row.outputCount===1&&row.request.count===1&&row.request.seed===173&&!row.request.advanced.mossImageAdvanced,'quality controls need original seeded text/image rows');
    row.transport='rest';row.request.seed=null;
    row.request.advanced.mossImageAdvanced={output:'images',noise:four?'$qualityNoisePortrait':'$qualityNoiseLandscape'};
    return row;
  });
}
export function qualityNoiseNpy(raw,shape,h){
  requireThat(raw.length===786432&&h.sha(raw)===QUALITY_NOISE_SHA,'frozen quality raw noise changed');
  requireThat(JSON.stringify(shape)==='[1,128,32,48]'||JSON.stringify(shape)==='[1,128,48,32]','quality noise shape changed');
  const bytes=h.makeNpy(shape,'float32',i=>{const value=raw.readFloatLE(i*4);requireThat(Number.isFinite(value),'nonfinite quality noise');return value;});
  h.inspectNpy(bytes,shape);
  requireThat(h.tensorData(bytes).equals(raw),'quality NPY transposed or changed raw noise');
  return bytes;
}
export function validateControlledReport(row,request,report,inputs,h){
  const base=h.validateEditReport(row,request,report,inputs);
  requireThat(base.samples.length===1&&base.samples[0].seed===null&&base.samples[0].noise_sha256===QUALITY_NOISE_SHA,'controlled sample did not consume exact caller noise');
  const conditioning=base.conditioning;
  requireThat(conditioning?.length===1&&conditioning[0].positive?.source==='text'&&conditioning[0].positive.sha256===h.sha(request.prompt)&&conditioning[0].negative?.source==='text'&&conditioning[0].negative.sha256===h.sha(''),'controlled quality bypassed normal positive/negative text');
  const tokens=row.refs.length===4?145:71;
  requireThat(conditioning[0].positive.sequence_length===tokens&&conditioning[0].positive.valid_tokens===tokens&&conditioning[0].negative.sequence_length===10&&conditioning[0].negative.valid_tokens===10,'controlled quality text/mask shape changed');
  return base;
}
export async function captureControlledCudaQuality({api,config,state,outputRoot,toolingSha,helpers:h,env=process.env,timeoutMs=120*60*1000}){
  requireThat(env.GITHUB_ACTIONS==='true'&&env.GITHUB_REPOSITORY==='SceneWorks/SceneWorks'&&env.GITHUB_SHA===toolingSha&&/^[a-f0-9]{40}$/.test(toolingSha??'')&&/^[1-9][0-9]*$/.test(env.GITHUB_RUN_ID??'')&&/^[1-9][0-9]*$/.test(env.GITHUB_RUN_ATTEMPT??''),'controlled quality requires actual registered Actions producer');
  requireThat(config.backend==='cuda'&&state.binding.backend==='cuda'&&state.binding.endpoint.workerGpuId==='1','controlled quality is restricted to physical GPU1');
  requireThat(resolve(outputRoot)===resolve(state.root,'controlled-quality'),'supplemental evidence must use its separate fresh directory');
  await mkdir(outputRoot,{mode:0o700});
  const completed=[],observation={};let current=null,currentJobId=null;
  try{
    const sources=await h.sourceIdentity(config);
    requireThat(h.canonical(sources)===h.canonical(state.lock.sources),'controlled quality native source pair changed');
    const binary=await h.fileIdentity(config.apiBinary);
    requireThat(h.canonical(binary)===h.canonical(state.lock.artifacts.cuda.files.workerBinary)&&h.canonical(binary)===h.canonical(state.lock.artifacts.cuda.files.apiBinary),'controlled quality native binary changed');
    const rows=controlledQualityRows(state.plan),project=state.binding.endpoint.projectId;
    const bundleInventory=await json(resolve(config.bundleRoot,'bundle-inventory.json'));
    requireThat(h.canonical(await h.fileIdentity(resolve(config.bundleRoot,'bundle-inventory.json')))===h.canonical(state.lock.files.qualityBundleInventory),'controlled quality bundle inventory changed');
    const rawFile='historical/injected-first-step/upstream/noise.f32le',entry=bundleInventory.files.find(item=>item.file===rawFile);
    requireThat(entry?.sha256===QUALITY_NOISE_SHA&&entry.byteLength===786432,'sealed upstream noise absent');
    const raw=await readFile(resolve(config.bundleRoot,rawFile));requireThat(h.sha(raw)===entry.sha256,'sealed upstream noise changed');
    const uploaded=await json(resolve(state.root,'uploaded-cuda.json')),inputs=await json(resolve(state.root,'input-identities.json'));
    await mkdir(resolve(outputRoot,'inputs'));
    const imageInputs={};
    for(const id of ['B','L','P','G']){
      const input=inputs[id],source=resolve(state.root,input.file),readback=await api.request(`/api/v1/projects/${encode(project)}/assets/${encode(uploaded[id]?.id)}`);
      requireThat(input.kind==='image'&&h.sha(await readFile(source))===input.sha256&&readback.id===uploaded[id].id&&readback.projectId===project,'controlled reference/project identity changed');
      const file=`inputs/${id}.png`;await copyFile(source,resolve(outputRoot,file));imageInputs[id]={...input,file};
      await save(resolve(outputRoot,'inputs',`${id}.asset.json`),readback);
    }
    await save(resolve(outputRoot,'input-identities.json'),imageInputs);
    await save(resolve(outputRoot,'measurement-inputs.json'),{originalLockSha256:h.digest(state.lock),originalBindingSha256:h.digest(state.binding),sources,toolingSha,helper:await h.fileIdentity(fileURLToPath(import.meta.url)),runId:env.GITHUB_RUN_ID,attempt:env.GITHUB_RUN_ATTEMPT,binary,rawNoise:entry,scope:'Four additional REST controls only; original50/58 plan/receipts unchanged; visual acceptance pending.'});
    const noises={};
    for(const [id,shape] of [['qualityNoiseLandscape',[1,128,32,48]],['qualityNoisePortrait',[1,128,48,32]]]){
      const bytes=qualityNoiseNpy(raw,shape,h),file=`inputs/${id}.npy`;await writeFile(resolve(outputRoot,file),bytes,{flag:'wx',mode:0o600});
      const form=new FormData();form.append('file',new Blob([bytes],{type:'application/x-npy'}),basename(file));
      const observed={},asset=await api.request(`/api/v1/projects/${encode(project)}/assets`,{method:'POST',body:form,observation:observed});
      await save(resolve(outputRoot,'inputs',`${id}.upload.json`),{file,sha256:h.sha(bytes),shape,response:observed,asset});
      const readback=await api.request(`/api/v1/projects/${encode(project)}/assets/${encode(asset.id)}`);
      requireThat(asset.id===readback.id&&readback.projectId===project&&readback.type==='tensor'&&readback.tensor?.sha256===h.sha(bytes)&&h.canonical(readback.tensor.shape)===h.canonical(shape),'controlled NPY upload/readback changed');
      noises[id]=readback;
    }
    for(const row of rows){
      current=row.id;currentJobId=null;const directory=resolve(outputRoot,'rows',row.id);await mkdir(directory,{recursive:true,mode:0o700});
      const admission=await h.nativeDevice('cuda');await save(resolve(directory,'admission.json'),admission);
      const request=h.replaceInputs(row,{...uploaded,...noises},project);await save(resolve(directory,'request.json'),request);
      await save(resolve(directory,'attempt.json'),{originalRowId:row.id,originalLockSha256:h.digest(state.lock),requestSha256:h.digest(request),phase:'POSTING_ONCE',at:new Date().toISOString()});
      for(const key of Object.keys(observation))delete observation[key];
      const queued=await api.request('/api/v1/image/jobs',{method:'POST',body:request,observation});
      currentJobId=queued.id??null;
      await save(resolve(directory,'submission.json'),{observed:observation,job:queued});requireThat(queued.id,'controlled POST returned no job identity');
      const job=await h.waitJob(api,row,queued.id,directory,timeoutMs,[queued]);await save(resolve(directory,'terminal-job.json'),job);
      h.checkJob(row,request,job,'1');requireThat(job.status==='completed'&&job.workerId===state.binding.endpoint.workerId&&job.result?.assets?.length===1,'controlled job failed or changed worker/output count');
      const summary=job.result.assets[0],asset=await api.request(`/api/v1/projects/${encode(project)}/assets/${encode(summary.id)}`);
      requireThat(asset.id===summary.id&&asset.projectId===project&&asset.lineage?.jobId===job.id&&asset.recipe?.rawAdapterSettings?.mossImageSampleIndex===0,'controlled output asset/job/recipe ownership changed');
      await save(resolve(directory,'asset-0.json'),asset);
      await save(resolve(directory,'workflow-0.json'),await api.request(`/api/v1/projects/${encode(project)}/assets/${encode(asset.id)}/workflow`));
      const path=asset.file?.path??asset.file?.relativePath??summary.mediaPath;requireThat(typeof path==='string'&&!path.startsWith('/')&&!path.split('/').includes('..'),'controlled output path escaped project');
      const bytes=await api.request(`/api/v1/projects/${encode(project)}/files/${path.split('/').map(encode).join('/')}`,{raw:true,max:32*1024*1024});
      requireThat(bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10]))&&bytes.readUInt32BE(16)===request.width&&bytes.readUInt32BE(20)===request.height,'controlled output is not actual full-size PNG');
      await writeFile(resolve(directory,'output-0.png'),bytes,{flag:'wx',mode:0o600});
      const report=h.findEditReport(asset,job);validateControlledReport(row,request,report,inputs,h);await save(resolve(directory,'provider-report-0.json'),report);
      const receipt={version:state.lock.version,rowId:row.id,originalLockSha256:h.digest(state.lock),originalBindingSha256:h.digest(state.binding),jobId:job.id,backend:'cuda',execution:'VERIFIED',acceptance:'AWAITING_FRESH_VISUAL_COMPARISON',requestSha256:h.digest(request),jobSha256:h.digest(job),providerReportSha256:(await h.fileIdentity(resolve(directory,'provider-report-0.json'))).sha256,outputs:[{assetId:asset.id,file:'output-0.png',sha256:h.sha(bytes)}],rawNoiseSha256:QUALITY_NOISE_SHA,sources,toolingSha,runId:env.GITHUB_RUN_ID,attempt:env.GITHUB_RUN_ATTEMPT,binary:state.lock.artifacts.cuda.files.workerBinary};
      await save(resolve(directory,'receipt.json'),receipt);await h.retainQualityPairs({root:outputRoot,lock:state.lock},row);
      completed.push({rowId:row.id,jobId:job.id,outputSha256:h.sha(bytes),receipt:await h.fileIdentity(resolve(directory,'receipt.json'))});
    }
    const result={execution:'COMPLETE',renders:4,acceptance:'AWAITING_FRESH_VISUAL_COMPARISON',rows:completed};await save(resolve(outputRoot,'result.json'),result);return result;
  }catch(error){
    // Preserve uncertain submissions and stop this supplemental batch. The
    // caller still records the independent original25+29 completion and drains
    // its owned worker normally. Never retry or proceed to another POST here.
    const result={execution:'FAILED',renders:completed.length,rowId:current,acceptedJobId:currentJobId,message:String(error.message).slice(0,4096),lastHttpObservation:observation,rows:completed};await save(resolve(outputRoot,'failure.json'),result);return result;
  }
}
