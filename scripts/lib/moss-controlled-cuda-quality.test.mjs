import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {tmpdir} from 'node:os';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {deflateSync} from 'node:zlib';
import {captureControlledCudaQuality,controlledQualityRows,qualityNoiseNpy,QUALITY_NOISE_SHA} from './moss-controlled-cuda-quality.mjs';
const source=process.env.MOSS_TEST_SOURCE_ROOT??fileURLToPath(new URL('../../',import.meta.url));
const h=await import(pathToFileURL(resolve(source,'scripts/lib/moss-edit-terminal.mjs'))),actions=await import(pathToFileURL(resolve(source,'scripts/lib/moss-edit-terminal-actions.mjs')));
const inputs={provenance:'synthetic fixture',single_case:{references:['B'],width:768,height:512,prompt:'fixed single prompt'},four_case:{references:['B','L','P','G'],width:512,height:768,prompt:'fixed four prompt',mandatory_preservation:['person','bottle','lamp','poster'],typography_scope:'no OCR'}};
const plan=h.createPlan(inputs),before=JSON.stringify(plan);
function frozenRaw(){
  let state=173n;const mask=(1n<<64n)-1n,uniform=()=>{state=(state+0x9e3779b97f4a7c15n)&mask;let z=state;z=((z^(z>>30n))*0xbf58476d1ce4e5b9n)&mask;z=((z^(z>>27n))*0x94d049bb133111ebn)&mask;return Number((z^(z>>31n))>>11n)/9007199254740992;};
  const data=Buffer.alloc(786432);for(let i=0;i<196608;i+=2){const r=Math.sqrt(-2*Math.log(Math.max(uniform(),1e-12))),t=Math.PI*2*uniform();data.writeFloatLE(r*Math.cos(t),i*4);data.writeFloatLE(r*Math.sin(t),(i+1)*4);}return data;
}
const raw=frozenRaw();
test('four REST controls preserve original plan and change only noise plus seed',()=>{
  const rows=controlledQualityRows(plan);assert.equal(rows.length,4);assert.equal(JSON.stringify(plan),before);
  for(const row of rows){const original=plan.rows.find(x=>x.id===row.id),restored=structuredClone(row);restored.transport=original.transport;restored.request.seed=173;delete restored.request.advanced.mossImageAdvanced;assert.deepEqual(restored,original);}
  const changed=structuredClone(plan);changed.rows.find(r=>r.id==='quality-cuda-base-single').request.seed=174;assert.throws(()=>controlledQualityRows(changed),/original seeded/);
});
test('NPY wrappers preserve exact frozen flat bytes in both aspect shapes; reject changed raw/shape',()=>{
  assert.equal(h.sha(raw),QUALITY_NOISE_SHA);
  const landscape=qualityNoiseNpy(raw,[1,128,32,48],h),portrait=qualityNoiseNpy(raw,[1,128,48,32],h);
  assert.deepEqual(h.tensorData(landscape),raw);assert.deepEqual(h.tensorData(portrait),raw);assert.notEqual(h.sha(landscape),h.sha(portrait));
  const changed=Buffer.from(raw);changed[123]^=1;assert.throws(()=>qualityNoiseNpy(changed,[1,128,32,48],h),/raw noise changed/);assert.throws(()=>qualityNoiseNpy(raw,[1,128,64,24],h),/shape changed/);
});
function png(width,height){
  const chunk=(type,data)=>{const name=Buffer.from(type),len=Buffer.alloc(4);len.writeUInt32BE(data.length);return Buffer.concat([len,name,data,Buffer.alloc(4)]);};
  const ihdr=Buffer.alloc(13);ihdr.writeUInt32BE(width);ihdr.writeUInt32BE(height,4);ihdr[8]=8;ihdr[9]=2;
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',ihdr),chunk('IDAT',deflateSync(Buffer.alloc((width*3+1)*height))),chunk('IEND',Buffer.alloc(0))]);
}
async function fixture(mutateReport=()=>{},failPost=false){
  const root=await mkdtemp(resolve(tmpdir(),'moss-controlled-quality-')),bundleRoot=resolve(root,'bundle'),stateRoot=resolve(root,'evidence'),sources={sceneworks:{head:'a'.repeat(40)},inference:{head:'b'.repeat(40)}};
  await mkdir(resolve(bundleRoot,'historical/injected-first-step/upstream'),{recursive:true});await mkdir(resolve(stateRoot,'inputs'),{recursive:true});
  const noiseFile='historical/injected-first-step/upstream/noise.f32le';await writeFile(resolve(bundleRoot,noiseFile),raw);
  await writeFile(resolve(bundleRoot,'bundle-inventory.json'),JSON.stringify({files:[{file:noiseFile,byteLength:raw.length,sha256:h.sha(raw)}]}));
  const apiBinary=resolve(root,'native.exe');await writeFile(apiBinary,'fixture binary');const binary=await h.fileIdentity(apiBinary),identities={},uploaded={},assets={},jobs={};let posts=0,uploads=0;
  for(const id of ['B','L','P','G']){const file=`inputs/${id}.png`,bytes=png(16,16);await writeFile(resolve(stateRoot,file),bytes);identities[id]={file,kind:'image',sha256:h.sha(bytes),width:16,height:16,pixelsSha256:h.sha(Buffer.alloc(16*16*3))};uploaded[id]={id:`input-${id}`,projectId:'project',type:'image'};assets[uploaded[id].id]=uploaded[id];}
  await writeFile(resolve(stateRoot,'input-identities.json'),JSON.stringify(identities));await writeFile(resolve(stateRoot,'uploaded-cuda.json'),JSON.stringify(uploaded));
  const lock={version:h.VERSION,candidateId:'actual-candidate',sources,files:{qualityBundleInventory:await h.fileIdentity(resolve(bundleRoot,'bundle-inventory.json'))},artifacts:{cuda:{files:{apiBinary:binary,workerBinary:binary}}}},state={root:stateRoot,lock,plan,binding:{backend:'cuda',endpoint:{projectId:'project',workerId:'owned-worker',workerGpuId:'1'}}};
  const env={GITHUB_ACTIONS:'true',GITHUB_REPOSITORY:'SceneWorks/SceneWorks',GITHUB_SHA:'c'.repeat(40),GITHUB_RUN_ID:'123456',GITHUB_RUN_ATTEMPT:'1'},config={backend:'cuda',bundleRoot,apiBinary};
  const api={async request(path,options={}){
    if(path.endsWith('/assets')&&options.method==='POST'){
      uploads++;const bytes=Buffer.from(await options.body.get('file').arrayBuffer()),tensor=h.inspectNpy(bytes),asset={id:`noise-${uploads}`,projectId:'project',type:'tensor',tensor:{sha256:h.sha(bytes),shape:tensor.shape}};assets[asset.id]=asset;Object.assign(options.observation,{status:201,body:JSON.stringify(asset)});return asset;
    }
    if(path==='/api/v1/image/jobs'){
      posts++;if(failPost){Object.assign(options.observation,{status:503,body:'fixture unavailable'});throw Error('fixture POST unavailable');}
      const request=options.body,id=`job-${posts}`,assetId=`output-${posts}`,row=controlledQualityRows(plan).find(r=>r.request.model===request.model&&r.request.width===request.width),references=row.refs.map(ref=>({pixels_sha256:identities[ref].pixelsSha256,original_dimensions:[16,16],encoded_dimensions:[16,16],resized_pixels_sha256:'d'.repeat(64),latent_sha256:'e'.repeat(64)}));
      const report={adapter_repository:'chinoll/Moss-Image-Edit',adapter_revision:'3be9c40037e1b89b17aa85ff6f8ee805872aa558',adapter_sha256:'17a48b8c605dcffdedb92a2b887d18cf316505be90802717669fb6556098dcdf',config_sha256:'b6b471cf7bc77b0874af03be618f8a06b6997d92b35de12c47a06b292ade2669',applied_projections:252,edit_contract:'ref-prefix-only-t0__target-text-original-moss-v1',reference_max_edge:512,use_kv_cache:true,references,base:{execution_device:'cuda',resolved:{variant:row.variant,steps:30,guidance:4,shift:5,training_shift:1,max_sequence_length:4096},output_kind:'images',latent_shape:[1,128,request.height/16,request.width/16],samples:[{seed:null,noise_sha256:QUALITY_NOISE_SHA}],conditioning:[{positive:{source:'text',sha256:h.sha(request.prompt),sequence_length:row.refs.length===4?145:71,valid_tokens:row.refs.length===4?145:71},negative:{source:'text',sha256:h.sha(''),sequence_length:10,valid_tokens:10}}]}};mutateReport(report);
      jobs[id]={id,projectId:'project',backend:'candle',payload:request,status:'completed',workerId:'owned-worker',startedAt:'2026-10-26T01:00:00Z',completedAt:'2026-10-26T01:01:00Z',result:{assets:[{id:assetId}]}};assets[assetId]={id:assetId,projectId:'project',lineage:{jobId:id},file:{path:assetId+'.png'},recipe:{rawAdapterSettings:{mossImageSampleIndex:0,mossImageEditExecutionReport:report}}};Object.assign(options.observation,{status:201,body:JSON.stringify({id})});return {id,status:'queued'};
    }
    const assetId=path.split('/')[6];if(path.endsWith('/workflow'))return {fixture:'actual workflow response'};
    if(path.includes('/assets/'))return assets[assetId];
    if(path.includes('/files/')){const id=path.split('/').at(-1).replace('.png',''),job=Object.values(jobs).find(j=>j.result.assets[0].id===id);return png(job.payload.width,job.payload.height);}
    throw Error('unexpected fixture API route '+path);
  }};
  const helpers={...h,sourceIdentity:async()=>sources,nativeDevice:async()=>({device:{uuid:actions.CUDA_UUID,physicalIndex:1,logicalIndex:0,pciOrRegistryId:'00000000:C1:00.0'}}),waitJob:async(_api,_row,id)=>jobs[id],retainQualityPairs:actions.retainQualityPairs};
  return {root,args:{api,config,state,outputRoot:resolve(stateRoot,'controlled-quality'),toolingSha:env.GITHUB_SHA,helpers,env},posts:()=>posts,uploads:()=>uploads};
}
test('four product API captures retain real helper-validated reports, pairs and source/binary/noise identities',async()=>{
  const f=await fixture();try{const result=await captureControlledCudaQuality(f.args);assert.equal(result.execution,'COMPLETE');assert.equal(result.renders,4);assert.equal(f.posts(),4);assert.equal(f.uploads(),2);for(const row of result.rows){const dir=resolve(f.args.outputRoot,'rows',row.rowId),receipt=JSON.parse(await readFile(resolve(dir,'receipt.json'))),pairs=JSON.parse(await readFile(resolve(dir,'visual-pairs.json')));assert.equal(receipt.rawNoiseSha256,QUALITY_NOISE_SHA);assert.equal(receipt.acceptance,'AWAITING_FRESH_VISUAL_COMPARISON');assert.equal(pairs.detailCrops.length,3);}assert.equal(JSON.stringify(plan),before);}finally{await rm(f.root,{recursive:true,force:true});}
});
test('wrong consumed raw noise aborts supplement after one POST and retains accepted job without retry',async()=>{
  const f=await fixture(report=>{report.base.samples[0].noise_sha256='f'.repeat(64);});try{const result=await captureControlledCudaQuality(f.args);assert.equal(result.execution,'FAILED');assert.equal(f.posts(),1);assert.equal(result.acceptedJobId,'job-1');assert.match(result.message,/exact caller noise/);}finally{await rm(f.root,{recursive:true,force:true});}
});
test('failed POST retains observation and never submits another control',async()=>{
  const f=await fixture(undefined,true);try{const result=await captureControlledCudaQuality(f.args);assert.equal(result.execution,'FAILED');assert.equal(f.posts(),1);assert.equal(result.acceptedJobId,null);assert.equal(result.lastHttpObservation.status,503);}finally{await rm(f.root,{recursive:true,force:true});}
});
