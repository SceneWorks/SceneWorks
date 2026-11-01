// Actual selected-checkpoint operands -> pinned CPU bridge -> ignored native consumer.
// Dispatch-only adapter; no math oracle, model forward, download or license acknowledgment here.
import assert from 'node:assert/strict';
import { campaignProfileArgs } from './lib/h3-campaign-build-profile.mjs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { open, readFile, writeFile, mkdir, realpath } from 'node:fs/promises';
import { authorized, runProcess, executedTests, hashFile } from './h3-finetune-terminal.mjs';
import { stripVerbatim } from './lib/windows-verbatim-path.mjs';
/** This checkout's root: the repository the verify-time staging binding must stay outside of. */
const SCENEWORKS_ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
/**
 * The receipt file the native probe writes per case — `receipt_file_name` in candle-gen-minimax-h3
 * tests/support/artifact_probe_support.rs: `{backend}.{safe id}.{sha256(id)[..8]}.json`, where the
 * safe id keeps ASCII alphanumerics and `._-` and maps everything else to `_` (sc-26691: the
 * harness read `{backend}.{id}.json`, a name the probe never writes).
 */
export function nativeReceiptName(backend,caseId){return `${backend}.${[...caseId].map(ch=>/^[A-Za-z0-9._-]$/.test(ch)?ch:'_').join('')}.${createHash('sha256').update(caseId,'utf8').digest('hex').slice(0,8)}.json`;}
/** The probe reports `manifest_path` canonicalized (`\\?\E:\...` on Windows); compare the file, not the spelling. */
async function sameFile(a,b){const canonical=async v=>{const value=stripVerbatim(v);try{return await realpath(value);}catch{return path.resolve(value);}};return (await canonical(a))===(await canonical(b));}

export const ACTUAL_SELECTOR='artifact_probe::actual_checkpoint_projections_match_pinned_cpu_oracle';
export const FROZEN_PROVENANCE=[
  {script:'scripts/ci/generate_minimax_h3_w4a8_fixture.py',sha256:'14c05c1dea46dfec0447d97b575a0c6f525e2d41bcc703ad16e1610828ea9676'},
  {script:'scripts/h3_convrot_int8_oracle.py',sha256:'25682137344a8d94bc134585603dd51e22a2edac7a148d5363c48ce0a38dbd82'},
  {script:'scripts/h3_pruned_adaln_oracle.py',sha256:'f9de6f19c29e9ce4858f74b930f0e9a641fe106333608587d7ce4d8630b16b5b'},
];
export const FROZEN_PRODUCERS=FROZEN_PROVENANCE.map(v=>v.script);
const tolerances={w4a8_projection:0,convrot_projection:0,mixed_w4a8_convrot:0,pruned_timestep_interpolation:2e-6,pruned_reduced_projection:2e-5};
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
const sha=value=>assert.match(value??'',/^[a-f0-9]{64}$/,'sha256 required');
const positive=value=>assert.ok(Number.isSafeInteger(value)&&value>0,'positive safe byte/count bound required');
const save=(file,value)=>writeFile(file,JSON.stringify(value,null,2)+'\n',{flag:'wx'});
const below=(root,file)=>{const rel=path.relative(root,file);return rel!==''&&!rel.startsWith('..')&&!path.isAbsolute(rel);};
async function outside(file,swRoot,inferenceRoot,{root,missing=false}={}) {
  assert.ok(typeof file==='string'&&path.isAbsolute(file)&&!file.split(/[\\/]/).includes('..'),'absolute outside-repository path required');
  const canonical=missing?path.join(await realpath(path.dirname(file)),path.basename(file)):await realpath(file);
  for(const repo of [swRoot,inferenceRoot]){const r=await realpath(repo);assert.ok(canonical!==r&&!below(r,canonical),'probe paths must be outside both repositories');}
  if(root){const r=await realpath(root);assert.ok(below(r,canonical),'probe path must be below campaign private output');}
  return canonical;
}
async function bounded(file,cap) {
  const handle=await open(file,'r');
  try{const info=await handle.stat();assert.ok(info.isFile()&&info.size<=cap,'bounded metadata file required');const bytes=Buffer.alloc(info.size);let done=0;while(done<bytes.length){const r=await handle.read(bytes,done,bytes.length-done,done);assert.ok(r.bytesRead,'truncated metadata');done+=r.bytesRead;}assert.equal((await handle.stat()).size,info.size,'metadata changed during read');return bytes;}finally{await handle.close();}
}
const json=async(file,cap=8*1024*1024)=>JSON.parse((await bounded(file,cap)).toString('utf8'));
async function boundJSON(file,expected,cap){sha(expected);const bytes=await bounded(file,cap);assert.equal(digest(bytes),expected,'metadata sha256 mismatch');return JSON.parse(bytes.toString('utf8'));}
export function validateProbeManifest(m,backend) {
  assert.equal(m.schema,'sceneworks.h3.actual_checkpoint_probe');assert.equal(m.schema_version,1);assert.equal(m.backend,backend==='cuda'?'candle':backend);
  assert.ok(path.isAbsolute(m.artifact?.path??'')||path.win32.isAbsolute(m.artifact?.path??''));sha(m.artifact.sha256);positive(m.artifact.bytes);
  assert.ok(m.staging?.receipt_path);sha(m.staging.receipt_sha256);assert.equal(m.staging.verified_artifact_sha256,m.artifact.sha256);
  const ceilings={max_cases:64,max_selected_tensors:8192,max_selected_bytes:512*1024*1024};
  for(const [key,cap]of Object.entries(ceilings)){const n=m.limits?.[key]??cap;positive(n);assert.ok(n<=cap,'manifest raises hard selection bound');}
  assert.ok(Array.isArray(m.cases)&&m.cases.length>0&&m.cases.length<=(m.limits?.max_cases??64));
  const ids=new Set(),names=new Set();
  for(const c of m.cases){assert.match(c.id??'',/^[A-Za-z0-9_.-]{1,128}$/);assert.ok(!ids.has(c.id),'duplicate probe case');ids.add(c.id);assert.ok(Object.hasOwn(tolerances,c.kind),'unsupported probe kind');assert.equal(c.tolerance?.max_abs,tolerances[c.kind],'canonical tolerance required');assert.ok(c.select&&c.input,'native selector and input required');
    assert.ok(Array.isArray(c.selected_tensors)&&c.selected_tensors.length>0,'explicit raw selected_tensors required');assert.equal(new Set(c.selected_tensors).size,c.selected_tensors.length,'duplicate selected tensor');
    for(const name of c.selected_tensors){assert.ok(typeof name==='string'&&name.length>0&&name.length<=512&&!/[\\/]/.test(name),'invalid raw tensor name');names.add(name);}
  }
  assert.ok(names.size<=(m.limits?.max_selected_tensors??8192),'selected tensor bound exceeded');return [...names].sort();
}
async function stagingBinding(m,p,c,roots) {
  const receiptPath=await outside(m.staging.receipt_path,...roots,{root:p.output});
  assert.ok(c.inputs.some(i=>i.role==='staging-projection'&&i.path===m.staging.receipt_path&&i.sha256===m.staging.receipt_sha256),'staging projection must be input-bound');
  const s=await boundJSON(receiptPath,m.staging.receipt_sha256,4*1024*1024);
  assert.equal(s.schema,'sceneworks.h3.staging_receipt');assert.equal(s.schema_version,1);assert.ok(typeof s.producer==='string'&&s.producer.trim());assert.equal(s.full_file_digest_verified,true);
  assert.equal(s.artifact_path,m.artifact.path);assert.equal(s.artifact_sha256,m.artifact.sha256);assert.equal(s.artifact_bytes,m.artifact.bytes);
  const parentPath=await outside(s.parent_receipt_path,...roots,{root:p.assetStaging?.root??p.output});
  assert.ok(c.inputs.some(i=>i.role==='staging-receipt'&&i.path===s.parent_receipt_path&&i.sha256===s.parent_receipt_sha256),'full staging parent must be input-bound');
  const parent=await boundJSON(parentPath,s.parent_receipt_sha256,64*1024*1024);
  assert.equal(parent.kind,'h3-finetune-stage-receipt');assert.equal(parent.version,1);assert.equal(parent.status,'completed');assert.equal(parent.proof_kind,'frozen_stage_content');assert.equal(parent.producer?.codeSha,p.sceneworks);assert.equal(await realpath(parent.settings.privateRoot),await realpath(p.assetStaging?.root??p.output));
  const f=parent.files?.find(v=>v.finalPath===m.artifact.path);assert.ok(f,'artifact absent from staging parent');assert.equal(f.status,'completed');assert.equal(f.expectedBytes,m.artifact.bytes);assert.equal(f.actualBytes,m.artifact.bytes);assert.equal(f.expectedSha256,m.artifact.sha256);assert.equal(f.observedSha256,m.artifact.sha256);
  // Legal inputs are consumed as staged, never filled or inferred by this adapter.
  assert.ok(f.explicitAcknowledgment,'staging operator acknowledgment missing');return s;
}
// Header + declared selected payload ONLY. Full-file identity is trusted from the SHA-bound
// staging chain, never recomputed. Output mini preserves raw names/dtypes/shapes and exact bytes.
async function readArtifactIndex(handle,artifactBytes,deadline=Infinity){
    const before=await handle.stat({bigint:true});assert.equal(before.size,BigInt(artifactBytes));assert.ok(before.isFile());
    const read=async(n,position)=>{assert.ok(Date.now()<deadline,'case deadline expired during selected reads');const b=Buffer.alloc(n);let done=0;while(done<n){const r=await handle.read(b,done,n-done,position+done);assert.ok(r.bytesRead,'truncated selected range');done+=r.bytesRead;}return b;};
    const prefix=await read(8,0),length=prefix.readBigUInt64LE();assert.ok(length>0n&&length<=100000000n&&length+8n<=before.size,'invalid bounded safetensors header');
    const start=8+Number(length),header=JSON.parse((await read(Number(length),8)).toString('utf8'));
    const widths={BOOL:1,U8:1,I8:1,F8_E4M3:1,F8_E5M2:1,I16:2,U16:2,F16:2,BF16:2,I32:4,U32:4,F32:4,I64:8,U64:8,F64:8};
    const spans=[];
    for(const [name,t]of Object.entries(header)){if(name==='__metadata__')continue;assert.ok(t&&Array.isArray(t.shape)&&t.shape.length<=8&&t.shape.every(v=>Number.isSafeInteger(v)&&v>=0),'invalid tensor shape');assert.ok(Object.hasOwn(widths,t.dtype),'unsupported tensor dtype');assert.ok(Array.isArray(t.data_offsets)&&t.data_offsets.length===2);const [a,b]=t.data_offsets;assert.ok(Number.isSafeInteger(a)&&Number.isSafeInteger(b)&&a>=0&&b>=a&&b<=artifactBytes-start,'selected range exceeds source bounds');const elements=t.shape.reduce((a,b)=>a*BigInt(b),1n);assert.equal(elements*BigInt(widths[t.dtype]),BigInt(b-a),'tensor size differs from dtype/shape');spans.push([a,b]);}
    spans.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);let end=0;for(const [a,b]of spans){assert.equal(a,end,'noncontiguous/overlapping source ranges');end=b;}assert.equal(end,artifactBytes-start);
    return {before,start,header,read};
}
export async function inspectActualArtifactHeader(artifact){const handle=await open(artifact.path,'r');try{const value=await readArtifactIndex(handle,artifact.bytes);return {header:value.header,headerBytesRead:value.start,weightPayloadBytesRead:0};}finally{await handle.close();}}
export async function extractSelectedRanges(m,names,output,{deadline=Infinity}={}) {
  const handle=await open(m.artifact.path,'r');
  try {
    const {before,start,header,read}=await readArtifactIndex(handle,m.artifact.bytes,deadline);
    let bytes=0;const mini={},selected=[];
    for(const name of names){assert.ok(Object.hasOwn(header,name)&&name!=='__metadata__','selected tensor missing');const t=header[name],size=t.data_offsets[1]-t.data_offsets[0];bytes+=size;assert.ok(bytes<=(m.limits?.max_selected_bytes??512*1024*1024),'selected byte bound exceeded');mini[name]={dtype:t.dtype,shape:t.shape,data_offsets:[bytes-size,bytes]};}
    if(header.__metadata__)mini.__metadata__=header.__metadata__;
    const encoded=Buffer.from(JSON.stringify(mini)),padding=(8-encoded.length%8)%8,miniHeader=Buffer.concat([encoded,Buffer.alloc(padding,32)]),miniPrefix=Buffer.alloc(8);miniPrefix.writeBigUInt64LE(BigInt(miniHeader.length));
    const dest=output?await open(output,'wx'):undefined;
    try {if(dest){await dest.writeFile(miniPrefix);await dest.writeFile(miniHeader);}for(const name of names){const t=header[name],[a,b]=t.data_offsets,hash=createHash('sha256');for(let at=a;at<b;){const chunk=await read(Math.min(64*1024,b-at),start+at);hash.update(chunk);if(dest)await dest.writeFile(chunk);at+=chunk.length;}selected.push({name,dtype:t.dtype,shape:t.shape,bytes:b-a,sha256:hash.digest('hex')});}}finally{await dest?.close();}
    const after=await handle.stat({bigint:true});for(const key of ['dev','ino','size','mtimeNs','ctimeNs'])assert.equal(after[key],before[key],'source changed during selected reads');
    return {selected,selected_bytes:bytes,whole_file_digest_recomputed:false};
  }finally{await handle.close();}
}
// Metadata bootstrap for the SAME native manifest. This prepares inputs, never proof.
// Raw AdaLN aliases mirror minimax_h3_codec::{strip_prefix,native_outputs}; packed bases
// stay raw because a projection probe must NOT imply the codec's QKV split/FFN reorder.
function prunedCanonical(raw){let name=raw.replace(/^(?:model\.diffusion_model\.|diffusion_model\.)/,'');if(name.startsWith('blocks.'))name='transformer_blocks.'+name.slice(7);if(name.startsWith('final_layer.adaln_proj.linear.'))name='norm_out.linear.'+name.slice('final_layer.adaln_proj.linear.'.length);return name;}
const preparedInput=(path,sha256,role)=>({path,sha256,role});
export async function prepareActualProbeManifest({backend,probe,artifact,stagingReceipt,outputDir,dit,ditBinding,nativeEvidence,proofKind='campaign_prepared_metadata'}){
  assert.ok(['mlx','cuda','candle'].includes(backend),'explicit backend required');const nativeBackend=backend==='cuda'?'candle':backend,stageBackend=nativeBackend==='candle'?'cuda':'mlx';
  assert.ok(['pruned-artifact','w4a8-packed-convrot'].includes(probe),'unsupported actual artifact probe');assert.ok(['campaign_prepared_metadata','unit_fake'].includes(proofKind));
  assert.ok(nativeEvidence&&typeof nativeEvidence.ref==='string'&&nativeEvidence.ref.length,'dispatch native evidence ref required');for(const value of [nativeEvidence.sha,nativeEvidence.productPin])assert.match(value??'',/^[a-f0-9]{40}$/);assert.notEqual(nativeEvidence.sha,nativeEvidence.productPin,'native evidence identity must stay distinct from product pin');
  const swRoot=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..'),evidenceRoot=await realpath(nativeEvidence.root);await outside(outputDir,swRoot,evidenceRoot);
  const sourcePath=await outside(artifact?.path,swRoot,evidenceRoot);sha(artifact.sha256);positive(artifact.bytes);const parentPath=await outside(stagingReceipt?.path,swRoot,evidenceRoot),parent=await boundJSON(parentPath,stagingReceipt.sha256,64*1024*1024);
  assert.equal(parent.kind,'h3-finetune-stage-receipt');assert.equal(parent.version,1);assert.equal(parent.status,'completed');assert.ok(parent.proof_kind==='frozen_stage_content'||proofKind==='unit_fake'&&parent.proof_kind==='unit_fake','unit_fake stage cannot prepare campaign metadata');assert.match(parent.producer?.codeSha??'',/^[a-f0-9]{40}$/);const privateRoot=await realpath(parent.settings.privateRoot);assert.ok(below(privateRoot,sourcePath)&&below(privateRoot,parentPath),'source and staging receipt must remain below actual private root');
  const owner=await json(path.join(privateRoot,'sc26691-owner.json'));assert.equal(owner.kind,'h3-finetune-stage-owner');assert.equal(await realpath(owner.root),privateRoot);assert.equal(owner.backend,stageBackend);assert.equal(parent.settings.backend,stageBackend);assert.equal(owner.codeSha,parent.producer.codeSha);assert.ok(parent.producer.runId&&parent.producer.runAttempt,'staging producer run identity missing');assert.equal(String(owner.runId),String(parent.producer.runId));assert.equal(String(owner.runAttempt),String(parent.producer.runAttempt));
  const staged=parent.files?.find(f=>f.finalPath===artifact.path);assert.ok(staged,'artifact absent from actual staging receipt');assert.equal(staged.status,'completed');for(const key of ['expectedSha256','observedSha256'])assert.equal(staged[key],artifact.sha256);for(const key of ['expectedBytes','actualBytes'])assert.equal(staged[key],artifact.bytes);assert.ok(staged.explicitAcknowledgment,'actual operator acknowledgment must already be staged');
  const handle=await open(sourcePath,'r');let manifest,selectedTensorNames,selectedBytes,headerFacts;
  try{
    const {before,start,header,read}=await readArtifactIndex(handle,artifact.bytes),names=Object.keys(header).filter(n=>n!=='__metadata__');assert.ok(names.length<=1000000,'header tensor count bound exceeded');const aliases={},cases=[];
    const requireTensor=(name,dtypes,shape)=>{const t=header[name];assert.ok(t,`missing actual tensor ${name}`);assert.ok(dtypes.includes(t.dtype),`unsupported actual dtype ${name}`);if(shape)assert.deepEqual(t.shape,shape,`actual tensor shape mismatch ${name}`);return t;};
    if(probe==='pruned-artifact'){
      assert.ok(ditBinding?.path,'pruned probe needs an explicit SHA-bound source config, not default geometry');await outside(ditBinding.path,swRoot,evidenceRoot);const config=await boundJSON(ditBinding.path,ditBinding.sha256,8*1024*1024);if(dit!==undefined)assert.deepEqual(dit,config,'supplied dit differs from SHA-bound source/operator config');dit=config;
      for(const key of ['hidden_size','num_attention_heads','attention_head_dim','ffn_dim','time_embed_dim','text_dim','in_channels','audio_in_channels','num_layers'])positive(dit[key]);assert.ok(dit.num_layers<=4096,'pruned layer count bound exceeded');for(const key of ['norm_eps','qk_norm_eps','final_norm_eps'])assert.ok(Number.isFinite(dit[key])&&dit[key]>0,`missing source config ${key}`);assert.ok(Array.isArray(dit.patch_size)&&dit.patch_size.length===3&&dit.patch_size.every(v=>Number.isSafeInteger(v)&&v>0),'source patch_size must be explicit [t,h,w]');
      const rawByCanonical=new Map();for(const raw of names){const canonical=prunedCanonical(raw);if(canonical==='adaln_t_table'||/^transformer_blocks\.\d+\.adaln_proj\.linear\.(?:weight|bias)$/.test(canonical)||/^norm_out\.linear\.(?:weight|bias)$/.test(canonical)){assert.ok(!rawByCanonical.has(canonical),`ambiguous raw aliases for ${canonical}`);rawByCanonical.set(canonical,raw);if(raw!==canonical)aliases[raw]=canonical;}}
      const raw=canonical=>{assert.ok(rawByCanonical.has(canonical),`missing actual pruned tensor ${canonical}`);return rawByCanonical.get(canonical);};const tableName=raw('adaln_t_table'),table=requireTensor(tableName,['F32']);assert.ok(table.shape.length===2&&table.shape[0]>=2&&table.shape[1]>0,'actual table must be F32 [grid>=2,rank>0]');const [grid,rank]=table.shape;assert.ok(rank<=dit.time_embed_dim,'pruned rank exceeds declared source timestep width');assert.ok(rank<=2**20&&18*dit.hidden_size<=2**22,'pruned input/output element bound exceeded');
      for(const name of names)assert.ok(!/^time_embedder\.(?:linear_[12]|proj_(?:in|out))\./.test(prunedCanonical(name)),'direct table conflicts with a full timestep MLP');
      const canonicalBasis=['adaln_t_table'];for(let layer=0;layer<dit.num_layers;layer++)for(const suffix of ['weight','bias'])canonicalBasis.push(`transformer_blocks.${layer}.adaln_proj.linear.${suffix}`);for(const suffix of ['weight','bias'])canonicalBasis.push(`norm_out.linear.${suffix}`);
      // Validate every actual projection against bound source geometry; Candle selects ALL of it.
      for(const canonical of canonicalBasis.slice(1)){const out=canonical.startsWith('norm_out.')?2*dit.hidden_size:18*dit.hidden_size;requireTensor(raw(canonical),['F16','BF16','F32'],canonical.endsWith('.weight')?[out,rank]:[out]);}
      const basis=canonicalBasis.map(raw),prefix='transformer_blocks.0.adaln_proj.linear',reduced=[raw(`${prefix}.weight`),raw(`${prefix}.bias`)];
      cases.push({id:'pruned-timestep-interpolation',kind:'pruned_timestep_interpolation',select:{table:'adaln_t_table',grid,rank,topology:nativeBackend==='candle',aliases},input:{dtype:'F32',timesteps:[0,0.1235,0.77725,1]}/*sc-26691: t is the [0,1] sigma — interpolation_position clamps it with no factor of 1000, so 123.5/777.25/1000 all collapsed onto the last row*/,selected_tensors:nativeBackend==='candle'?basis:[tableName],tolerance:{max_abs:tolerances.pruned_timestep_interpolation}});
      cases.push({id:'pruned-reduced-projection',kind:'pruned_reduced_projection',select:{prefix,weight:`${prefix}.weight`,bias_tensor:`${prefix}.bias`,rank,out_features:18*dit.hidden_size,topology:nativeBackend==='candle',aliases},input:{dtype:'F32',shape:[1,rank],values:Array.from({length:rank},(_,i)=>(i%7-3)/8)},selected_tensors:nativeBackend==='candle'?basis:reduced,tolerance:{max_abs:tolerances.pruned_reduced_projection}});
      headerFacts={grid,rank,hidden_size:dit.hidden_size,num_layers:dit.num_layers,reducedProjection:{rawWeight:reduced[0],rawBias:reduced[1],out:18*dit.hidden_size,in:rank},configBinding:ditBinding};
    }else{
      const w4=[],conv=[],descriptorNames=names.filter(n=>n.endsWith('.comfy_quant')).sort();assert.ok(descriptorNames.length<=8192,'packed descriptor count bound exceeded');for(const descriptor of descriptorNames){
        const base=descriptor.slice(0,-'.comfy_quant'.length),relative=header[`${base}.weight_s_rel`],scale=header[`${base}.weight_scale`];assert.ok(!(relative&&scale),`contradictory packed families at ${base}`);if(!relative&&!scale)continue;
        for(const forbidden of ['scales','biases'])assert.ok(!header[`${base}.${forbidden}`],`foreign affine quantizer operand at ${base}`);
        const weight=requireTensor(`${base}.weight`,['I8']);assert.ok(weight.shape.length===2&&weight.shape.every(v=>v>0),'packed weight must be nonempty [out,stored_in]');const [n,stored]=weight.shape,k=relative?stored*2:stored;assert.ok(k%256===0&&k<=2**22&&n%16===0,'packed probe requires a shared 256-aligned input and 16-aligned output');
        const suffixes=relative?['weight','weight_codebook','weight_s_rel','weight_s_channel','comfy_quant']:['weight','weight_scale','comfy_quant'];if(relative){requireTensor(`${base}.weight_codebook`,['F32'],[16]);requireTensor(`${base}.weight_s_rel`,['F8_E4M3'],[n,k/16]);requireTensor(`${base}.weight_s_channel`,['F32'],[n]);}else{requireTensor(`${base}.weight_scale`,['F32'],[n,1]);assert.ok(!header[`${base}.weight_codebook`]&&!header[`${base}.weight_s_channel`],'ConvRot base must not carry W4 codebook companions');}
        const d=requireTensor(descriptor,['U8']);assert.ok(d.shape.length===1&&d.shape[0]>0&&d.shape[0]<=4096,'bounded actual comfy_quant descriptor required');const selected=suffixes.map(s=>`${base}.${s}`),candidate={base,n,k,selected,bytes:selected.reduce((sum,name)=>sum+header[name].data_offsets[1]-header[name].data_offsets[0],0)};(relative?w4:conv).push(candidate);
      }
      // Keep only the cheapest candidate at each K; do not materialize an N*M pair grid.
      const cheapest=items=>{const map=new Map();for(const item of items){const old=map.get(item.k);if(!old||item.bytes<old.bytes||item.bytes===old.bytes&&item.base.localeCompare(old.base)<0)map.set(item.k,item);}return map;},primary=cheapest(w4),secondary=cheapest(conv),pairs=[...primary.values()].filter(a=>secondary.has(a.k)).map(a=>({a,b:secondary.get(a.k),bytes:a.bytes+secondary.get(a.k).bytes})).sort((a,b)=>a.bytes-b.bytes||a.a.base.localeCompare(b.a.base)||a.b.base.localeCompare(b.b.base));assert.ok(pairs.length,'no actual compatible W4A8 + ConvRot operand pair');const {a,b}=pairs[0];assert.ok(pairs[0].bytes<=256*1024*1024,'selected payload exceeds 256 MiB bootstrap bound');assert.ok(a.n+b.n<=2**22,'mixed output element bound exceeded');
      const descriptors=[];for(const [candidate,format]of [[a,'asym_w4a8_int8'],[b,'int8_tensorwise']]){const name=`${candidate.base}.comfy_quant`,t=header[name],bytes=await read(t.data_offsets[1]-t.data_offsets[0],start+t.data_offsets[0]),descriptor=JSON.parse(bytes.toString('utf8'));assert.equal(descriptor.format,format);assert.equal(descriptor.convrot_groupsize,256,'pinned BF16 replay requires rotation group 256');if(format==='asym_w4a8_int8'){assert.equal(descriptor.group_size,16);assert.deepEqual(Object.keys(descriptor).sort(),['convrot_groupsize','format','group_size']);}else{assert.equal(descriptor.convrot,true);assert.deepEqual(Object.keys(descriptor).sort(),['convrot','convrot_groupsize','format']);}descriptors.push({name,sha256:digest(bytes),bytes:bytes.length});}
      cases.push({id:'mixed-w4a8-convrot',kind:'mixed_w4a8_convrot',select:{base:a.base,convrot_base:b.base,w4a8_bias:false,convrot_bias:false,expect_route:'w4a8',aliases:{}},input:{dtype:'BF16',shape:[1,a.k],values:Array.from({length:a.k},(_,i)=>(i%7-3)/8)},selected_tensors:[...a.selected,...b.selected],tolerance:{max_abs:0}});headerFacts={w4a8:{rawBase:a.base,out:a.n,in:a.k},convrot:{rawBase:b.base,out:b.n,in:b.k},descriptors};
    }
    selectedTensorNames=[...new Set(cases.flatMap(c=>c.selected_tensors))].sort();assert.ok(selectedTensorNames.length<=4096,'selected tensor count exceeds bootstrap bound');selectedBytes=selectedTensorNames.reduce((sum,name)=>sum+header[name].data_offsets[1]-header[name].data_offsets[0],0);assert.ok(selectedBytes>0&&selectedBytes<=256*1024*1024,'selected payload exceeds 256 MiB bootstrap bound');
    const after=await handle.stat({bigint:true});for(const key of ['dev','ino','size','mtimeNs','ctimeNs'])assert.equal(after[key],before[key],'source changed during metadata bootstrap');
    manifest={schema:'sceneworks.h3.actual_checkpoint_probe',schema_version:1,backend:nativeBackend,artifact:{...artifact},staging:{},oracle:{producer:'h3-actual-checkpoint-oracle-bridge/1',producer_version:1,artifact_sha256:artifact.sha256,producer_provenance:FROZEN_PROVENANCE},limits:{max_cases:32,max_selected_tensors:4096,max_selected_bytes:256*1024*1024},...(probe==='pruned-artifact'?{dit}:{}),cases};
  }finally{await handle.close();}
  // Only real observed parent fields are projected. No operator consent is created here.
  const projectionPath=path.join(outputDir,'staging-projection.json'),specPath=path.join(outputDir,'actual-probe-spec.json'),receiptPath=path.join(outputDir,'actual-probe-prepared.json');await save(projectionPath,{schema:'sceneworks.h3.staging_receipt',schema_version:1,producer:'scripts/h3-finetune-terminal-artifact-probe.mjs metadata bootstrap',full_file_digest_verified:true,artifact_path:artifact.path,artifact_sha256:artifact.sha256,artifact_bytes:artifact.bytes,parent_receipt_path:stagingReceipt.path,parent_receipt_sha256:stagingReceipt.sha256});manifest.staging={receipt_path:projectionPath,receipt_sha256:digest(await bounded(projectionPath,4*1024*1024)),verified_artifact_sha256:artifact.sha256};validateProbeManifest(manifest,stageBackend);await save(specPath,manifest);const spec={path:specPath,sha256:digest(await bounded(specPath,8*1024*1024))};
  const inputs=[preparedInput(projectionPath,manifest.staging.receipt_sha256,'staging-projection'),preparedInput(stagingReceipt.path,stagingReceipt.sha256,'staging-receipt'),preparedInput(spec.path,spec.sha256,'actual-artifact-probe-spec'),...(ditBinding?[preparedInput(ditBinding.path,ditBinding.sha256,'source-config')]:[])];
  await save(receiptPath,{kind:'actual-artifact-probe-prepared',version:1,status:'prepared-not-numerical-proof',proof_kind:proofKind,artifact,stagingReceipt,spec,selectedTensorNames,selectedBytes,headerFacts,nativeEvidence,productPin:nativeEvidence.productPin,whole_file_digest_recomputed:false,weight_payload_bytes_read:0,oracle_values_produced:false,native_values_produced:false,device_proof:false,qkv_ffn_tensor_mapping_evidence:false});
  return {spec,inputs,receipt:{path:receiptPath,sha256:digest(await bounded(receiptPath,8*1024*1024))},selectedTensorNames,selectedBytes,headerFacts};
}
function provenance(actual,expected){
  // Manifest/bridge use records (with optional upstream pins); Rust serializes its validated
  // BTreeMap as script -> SHA. These are two transports of the SAME frozen provenance set.
  const records=Array.isArray(actual)?actual:actual&&typeof actual==='object'?Object.entries(actual).map(([script,sha256])=>({script,sha256})):[];
  for(const item of records)if(item.source_revision_sha256!==undefined)assert.equal(item.source_revision_sha256,item.sha256);
  const core=list=>list.map(({script,sha256})=>({script,sha256})).sort((a,b)=>a.script.localeCompare(b.script));assert.equal(records.length,expected.length);assert.deepEqual(core(records),core(expected),'frozen producer provenance mismatch');
}
// Device proof per backend (sc-26691). The native receipt says device_proof, but the harness owns the
// rule and states its basis, so evidence claims exactly what was proven:
// - cuda: the receipt's candle CUDA device (kind "cuda"), no CPU fallback. The lane pins physical GPU 1
//   (CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=1, asserted before the probe; the receipt
//   records visible_devices verbatim).
// - mlx: MLX exposes the default device's type and index, never a hardware UUID. Proof is the receipt
//   reporting the MLX default device as a GPU (kind "mlx-default-gpu", label "mlx device type Gpu index
//   Some(<index>)"), no CPU fallback, AND the probe environment not disabling the GPU default
//   (MLX_USE_GPU, recorded on the receipt as deviceEnvironment). No hardware identity is claimed.
export const DEVICE_PROOF_BASIS=Object.freeze({cuda:'cuda-device-no-fallback',mlx:'mlx-default-gpu-no-fallback'});
const MLX_GPU_DISABLED=/^\s*(0|false|no|off)\s*$/i;
export const mlxDeviceEnvironment=env=>({MLX_USE_GPU:typeof env?.MLX_USE_GPU==='string'?env.MLX_USE_GPU:null});
export function deviceProof(backend,native,deviceEnvironment){
  const device=native?.device;if(native?.device_proof!==true||device?.cpu_fallback!==false)return {proof:false,basis:null};
  if(backend==='cuda')return device.kind==='cuda'?{proof:true,basis:DEVICE_PROOF_BASIS.cuda}:{proof:false,basis:null};
  assert.equal(backend,'mlx');
  const label=/^mlx device type Gpu index Some\((\d+)\)$/.exec(device.label??'');
  const gpu=device.kind==='mlx-default-gpu'&&label!==null&&Number.isInteger(device.index)&&device.index===Number(label[1]);
  // The environment must be recorded (null = unset); an unrecorded or GPU-disabling MLX_USE_GPU is no proof.
  const recorded=deviceEnvironment!==null&&typeof deviceEnvironment==='object'&&Object.hasOwn(deviceEnvironment,'MLX_USE_GPU')&&(deviceEnvironment.MLX_USE_GPU===null||typeof deviceEnvironment.MLX_USE_GPU==='string');
  return gpu&&recorded&&!MLX_GPU_DISABLED.test(deviceEnvironment.MLX_USE_GPU??'')?{proof:true,basis:DEVICE_PROOF_BASIS.mlx}:{proof:false,basis:null};
}
const selectedCore=items=>items.map(({name,dtype,shape,bytes,sha256})=>({name,dtype,shape,bytes,sha256})).sort((a,b)=>a.name.localeCompare(b.name));
export function mergeProbeReceipts(m,producer,oracle,native,selection,expectedProvenance,{deviceEnvironment}={}) {
  assert.equal(producer.schema,'sceneworks.h3.actual_checkpoint_oracle_producer_receipt');assert.equal(producer.schema_version,1);assert.equal(producer.status,'ok');assert.equal(producer.device,'cpu');assert.equal(producer.gpu_used,false);assert.equal(producer.network_used,false);provenance(producer.producer_provenance,expectedProvenance);
  assert.equal(oracle.schema,'sceneworks.h3.actual_checkpoint_oracle');assert.equal(oracle.schema_version,1);assert.equal(oracle.producer,'h3-actual-checkpoint-oracle-bridge/1');assert.equal(oracle.producer_version,1);assert.equal(oracle.artifact_sha256,m.artifact.sha256);assert.equal(oracle.device,'cpu');provenance(oracle.producer_provenance,expectedProvenance);
  for(const key of ['path','sha256','bytes'])assert.equal(producer.artifact?.[key],m.artifact[key]);assert.equal(producer.artifact.whole_file_digest_recomputed,false);
  assert.equal(native.length,m.cases.length);assert.equal(oracle.cases?.length,m.cases.length);assert.equal(producer.cases?.length,m.cases.length);
  const result=[];
  for(const c of m.cases){const matches=native.filter(r=>r.case_id===c.id),o=oracle.cases.filter(r=>r.id===c.id),pr=producer.cases.filter(r=>r.id===c.id);assert.equal(matches.length,1,'missing/duplicate native receipt');assert.equal(o.length,1);assert.equal(pr.length,1);assert.equal(pr[0].status,'ok');const n=matches[0];
    assert.equal(n.schema,'sceneworks.h3.actual_checkpoint_probe_receipt');assert.equal(n.schema_version,1);assert.equal(n.backend,m.backend);assert.equal(n.kind,c.kind);assert.equal(n.artifact?.path,m.artifact.path);assert.equal(n.artifact.sha256,m.artifact.sha256);assert.equal(n.artifact.bytes,m.artifact.bytes);assert.equal(n.artifact.whole_file_digest_recomputed,false);
    const selected=selectedCore(selection.selected.filter(s=>c.selected_tensors.includes(s.name)));assert.equal(selected.length,c.selected_tensors.length);assert.deepEqual(selectedCore(n.selected??[]),selected);assert.deepEqual(selectedCore(o[0].selected??[]),selected);assert.deepEqual(selectedCore(pr[0].selected??[]),selected);
    assert.equal(n.oracle_sha256,m.oracle.sha256);assert.equal(n.oracle_path,m.oracle.path);assert.equal(n.artifact.staging_receipt_path,m.staging.receipt_path);assert.equal(n.artifact.staging_receipt_sha256,m.staging.receipt_sha256);provenance(n.oracle_producer_provenance,expectedProvenance);
    assert.deepEqual(n.native?.shape,o[0].output?.shape);assert.deepEqual(n.oracle?.values,o[0].output?.values);assert.deepEqual(n.oracle?.shape,o[0].output?.shape);assert.ok(Number.isFinite(n.tolerance?.max_abs));assert.equal(Math.fround(n.tolerance.max_abs),Math.fround(c.tolerance.max_abs),'native F32 tolerance differs from canonical bound');assert.equal(o[0].tolerance?.max_abs,c.tolerance.max_abs);
    const values=n.native?.values;assert.ok(Array.isArray(values)&&values.length>0&&values.every(Number.isFinite),'native values missing');assert.ok(n.oracle.values.length===values.length&&n.oracle.values.every(Number.isFinite));assert.ok(Number.isFinite(n.comparison?.max_abs)&&n.comparison.max_abs>=0,'native comparison missing');
    assert.ok(Number.isFinite(n.comparison.tolerance),'comparison tolerance missing');assert.equal(Math.fround(n.comparison.tolerance),Math.fround(c.tolerance.max_abs),'comparison used a different tolerance');
    // Receipt coherence check only, not a second arithmetic oracle: compare the two already
    // produced F32 vectors exactly as the native receipt did. Never generate expected values.
    const observed=values.reduce((maximum,value,index)=>Math.max(maximum,Math.abs(Math.fround(value-n.oracle.values[index]))),0);assert.equal(n.comparison.max_abs,observed,'reported max_abs differs from retained native/oracle values');
    for(const output of [n.native,n.oracle]){const bytes=Buffer.alloc(output.values.length*4);output.values.forEach((value,index)=>bytes.writeFloatLE(value,index*4));assert.equal(output.values_sha256,digest(bytes),'output vector sha256 mismatch');}
    const device=n.native.device;assert.ok(device&&typeof device.label==='string'&&device.label.length,'device identity missing');const {proof,basis}=deviceProof(m.backend==='mlx'?'mlx':'cuda',n.native,deviceEnvironment); // manifest backend: mlx | candle (CUDA)
    const pass=n.status==='pass'&&n.pass===true&&proof&&n.comparison.max_abs<=Math.fround(c.tolerance.max_abs);
    result.push({case_id:c.id,kind:c.kind,status:pass?'PASS':'FAIL',pass,artifact:m.artifact,selected,native:n.native,oracle:n.oracle,max_abs:n.comparison.max_abs,tolerance:n.tolerance,device,cpu_fallback:device.cpu_fallback,device_proof:proof,deviceProofBasis:basis,standalone_dense_fidelity_claimed:false,qkv_ffn_tensor_mapping_evidence:false,scope:'Selected packed-projection/pruned-curve comparison only; never full QKV/FFN tensor mapping or standalone dense fidelity'});
  }
  return result;
}
async function readOracleAuthority(root,{snapshotDir,retained=false,unitFake=false}={}){
  const sources=[];for(const [index,producer]of FROZEN_PROVENANCE.entries()){const bytes=await bounded(path.join(root,retained?`frozen-producer-${index}.py`:producer.script),4*1024*1024);if(!unitFake)assert.equal(digest(bytes),producer.sha256,'frozen script source changed');if(snapshotDir)await writeFile(path.join(snapshotDir,`frozen-producer-${index}.py`),bytes,{flag:'wx'});sources.push(bytes.toString('utf8'));}
  const constant=(source,name)=>{const match=new RegExp(`^${name} = ["']([a-f0-9]{40})["']$`,'m').exec(source);assert.ok(match,`frozen producer missing ${name}`);return match[1];};
  return {packed:constant(sources[0],'KITCHEN_MERGED_TREE'),format:constant(sources[1],'KITCHEN'),pruned:constant(sources[2],'REVISION')};
}
function campaignCase(p,value,authority){
  const pruned=value.kind?.startsWith('pruned_'),revision=pruned?authority.pruned:authority.packed;
  assert.notEqual(revision,p.nativeEvidence?.sha,'native-vs-native is not independent CPU oracle evidence');
  return {...value,productPin:p.inference,nativeEvidence:structuredClone(p.nativeEvidence),oracle_authority:{revision,source:pruned?'pinned-pruned-reference-source':'comfy-kitchen',producer_provenance:FROZEN_PROVENANCE,format_reference_revision:pruned?revision:authority.format,meaning:'Independent pinned CPU arithmetic, not production Rust output; bridge wrapper checkout is recorded separately'},native_authority:{revision:p.nativeEvidence?.sha,ref:p.nativeEvidence?.ref},bridge_execution_revision:p.nativeEvidence?.sha};
}
async function expectedProvenance(inferenceRoot){const actual=await Promise.all(FROZEN_PRODUCERS.map(async script=>({script,sha256:digest(await bounded(path.join(inferenceRoot,script),4*1024*1024))})));provenance(actual,FROZEN_PROVENANCE);return actual;}
async function verifyEvidenceSource(p,swRoot,inferenceRoot){
  const n=p.nativeEvidence;assert.ok(n&&typeof n.ref==='string'&&n.ref.length,'explicit native evidence ref required');assert.match(n.sha??'',/^[a-f0-9]{40}$/);assert.equal(n.productPin,p.inference);assert.equal(await realpath(n.root),await realpath(inferenceRoot));
  const exec=promisify(execFile);for(const [cwd,revision]of [[swRoot,p.sceneworks],[inferenceRoot,n.sha]]){assert.equal((await exec('git',['rev-parse','HEAD'],{cwd})).stdout.trim(),revision,'evidence source revision drift');assert.equal((await exec('git',['status','--porcelain'],{cwd})).stdout.trim(),'','evidence source must stay clean');}
  // Product pin stays on main; evidence is allowed to be the coordinator's exact feature SHA.
  const cargo=await bounded(path.join(swRoot,'crates/sceneworks-worker/Cargo.toml'),1024*1024);assert.ok(cargo.toString('utf8').includes(`rev = "${p.inference}"`),'product inference pin drift');
}
const receiptFiles=(m,name)=>[...FROZEN_PRODUCERS.map((_,index)=>`frozen-producer-${index}.py`),name,'producer-spec.json','probe-manifest.json','selected-ranges.json','producer-receipt.json','cpu-oracle.json','producer.log','native.log',...m.cases.map(k=>`native-receipts/${nativeReceiptName(m.backend,k.id)}`)];
export async function executeActualArtifactProbe(p,c,swRoot,inferenceRoot,dir,deps={}) {
  const fake=deps.proof_kind==='unit_fake',runner=deps.runProcess??runProcess,contract=c.actualArtifactProbe,receiptName=contract?.receipt??'artifact-probe-receipt.json';assert.equal(path.basename(receiptName),receiptName);
  let manifest,receipt={kind:'actual-artifact-probe',version:1,proof_kind:fake?'unit_fake':'campaign_native',caseId:c.id,backend:c.backend,sceneworks:p.sceneworks,inference:p.inference,status:'FAIL',pass:false,cases:[],productPin:p.inference,nativeEvidence:structuredClone(p.nativeEvidence),whole_file_digest_recomputed:false};
  try {
    const deadline=process.env.H3_CASE_DEADLINE_MS===undefined?Date.now()+c.timeoutSeconds*1000:Number(process.env.H3_CASE_DEADLINE_MS);
    const remaining=()=>{assert.ok(Number.isFinite(deadline)&&deadline>Date.now(),'case deadline expired');return Math.min(c.timeoutSeconds,(deadline-Date.now())/1000);};remaining();
    const python=fake?'python3':process.env.H3_PROBE_PYTHON;
    if(!fake){authorized(p,c);assert.equal(process.env.GITHUB_EVENT_NAME,'workflow_dispatch','actual probe is dispatch-only');if(c.backend==='cuda'){assert.equal(process.env.CUDA_VISIBLE_DEVICES,'1','manual probe runs on physical GPU 1 (GPU 0 is the owner display GPU)');assert.equal(process.env.CUDA_DEVICE_ORDER,'PCI_BUS_ID');}assert.ok(python&&path.isAbsolute(python),'explicit preflighted private Python 3.12 venv required');await verifyEvidenceSource(p,swRoot,inferenceRoot);}
    await outside(dir,swRoot,inferenceRoot,{root:p.output});assert.equal(contract?.kind,'actual-artifact-probe');assert.ok(contract.noAV===true);
    const spec=await outside(contract.spec.path,swRoot,inferenceRoot,{root:p.output});assert.ok(c.inputs.some(i=>i.role==='actual-artifact-probe-spec'&&i.path===contract.spec.path&&i.sha256===contract.spec.sha256));
    manifest=await boundJSON(spec,contract.spec.sha256,8*1024*1024);if(!fake){const book=JSON.parse(await readFile(path.join(swRoot,'docs/testing/sc-26691-h3-cases.json'),'utf8')),expected=book.artifacts[c.probe==='pruned-artifact'?2:3];assert.equal(manifest.artifact.sha256,expected.expectedSha256,'probe must consume the declared frozen artifact');assert.equal(manifest.artifact.bytes,expected.expectedBytes);}const names=validateProbeManifest(manifest,c.backend);receipt.artifact=manifest.artifact;receipt.staging=manifest.staging;
    await outside(manifest.artifact.path,swRoot,inferenceRoot,{root:p.assetStaging?.root??p.output});assert.ok(c.inputs.some(i=>i.role==='checkpoint'&&i.path===manifest.artifact.path&&i.sha256===manifest.artifact.sha256),'actual source must be a bound checkpoint');
    await stagingBinding(manifest,p,c,[swRoot,inferenceRoot]);
    receipt.nativeEvidence=structuredClone(p.nativeEvidence);receipt.productPin=p.inference;
    const expected=fake?FROZEN_PROVENANCE:await expectedProvenance(inferenceRoot);if(manifest.oracle?.producer_provenance)provenance(manifest.oracle.producer_provenance,expected);
    const authority=await readOracleAuthority(inferenceRoot,{snapshotDir:dir,unitFake:fake});for(const k of manifest.cases)campaignCase(p,k,authority);
    const mini=path.join(dir,'selected-tensors.safetensors'),selection=await extractSelectedRanges(manifest,names,mini,{deadline});await save(path.join(dir,'selected-ranges.json'),selection);
    const working=path.join(dir,'probe-manifest.json'),oraclePath=path.join(dir,'cpu-oracle.json'),producerPath=path.join(dir,'producer-receipt.json'),receiptDir=path.join(dir,'native-receipts');await mkdir(receiptDir);
    const specBytes=await bounded(spec,8*1024*1024);assert.equal(digest(specBytes),contract.spec.sha256);await writeFile(working,specBytes,{flag:'wx'});await writeFile(path.join(dir,'producer-spec.json'),specBytes,{flag:'wx'});
    const producerCommand=[python,'-I','-B',path.join(inferenceRoot,'scripts/h3_actual_checkpoint_oracle.py'),'--spec',working,'--selected-tensors',mini,'--output',oraclePath,'--receipt',producerPath];
    if(c.actualArtifactProbe.referenceDir){const referenceRoot=await outside(contract.referenceDir,swRoot,inferenceRoot);const references=c.inputs.filter(i=>i.role==='reference'&&below(referenceRoot,path.resolve(i.path)));assert.ok(references.length,'local pinned reference sources must be input-bound');for(const input of references){const file=await outside(input.path,swRoot,inferenceRoot,{root:referenceRoot});sha(input.sha256);assert.equal(digest(await bounded(file,8*1024*1024)),input.sha256,'local pinned reference source digest mismatch');}producerCommand.push('--reference-dir',contract.referenceDir);}
    const env={...process.env,CARGO_TARGET_DIR:path.join(p.output,`native-target-${p.nativeEvidence.sha}-${c.backend}`),CARGO_PROFILE_TEST_DEBUG:'0',CARGO_PROFILE_DEV_DEBUG:'0',...(c.backend==='cuda'?{RUSTC_WRAPPER:''}:{})};
    await runner(producerCommand,{cwd:inferenceRoot,env,logPath:path.join(dir,'producer.log'),timeoutSeconds:remaining()});
    const producer=await json(producerPath),oracleBytes=await bounded(oraclePath,256*1024*1024),oracle=JSON.parse(oracleBytes.toString('utf8'));assert.equal(producer.spec_sha256,digest(await bounded(working,8*1024*1024)));provenance(oracle.producer_provenance,expected);
    assert.equal(producer.status,'ok','producer did not succeed');provenance(producer.producer_provenance,expected);assert.equal(producer.selected_tensors_sha256,await hashFile(mini),'producer did not bind exact selected mini');
    manifest.oracle={path:oraclePath,sha256:digest(oracleBytes),producer:oracle.producer,producer_version:oracle.producer_version,artifact_sha256:oracle.artifact_sha256,producer_provenance:oracle.producer_provenance};await writeFile(working,JSON.stringify(manifest,null,2)+'\n');
    const command=['cargo','test','--locked',...campaignProfileArgs(),'--manifest-path',path.join(inferenceRoot,'Cargo.toml'),'-p',c.backend==='mlx'?'mlx-gen-minimax-h3':'candle-gen-minimax-h3',...(c.backend==='cuda'?['--features','cuda']:[]),'--test','integration',ACTUAL_SELECTOR,'--','--exact','--ignored','--nocapture','--test-threads=1'];
    // The native probe inherits this env; record what it saw (MLX_USE_GPU decides the MLX default device).
    const deviceEnvironment=c.backend==='mlx'?mlxDeviceEnvironment(env):undefined;
    let nativeError;try{await runner(command,{cwd:inferenceRoot,env:{...env,H3_ACTUAL_PROBE_MANIFEST:working,H3_ACTUAL_PROBE_RECEIPT_DIR:receiptDir},logPath:path.join(dir,'native.log'),timeoutSeconds:remaining()});}catch(error){nativeError=error;}
    const native=await Promise.all(manifest.cases.map(k=>json(path.join(receiptDir,nativeReceiptName(manifest.backend,k.id)),256*1024*1024)));
    for(const n of native){assert.ok(await sameFile(n.manifest_path,working),'native probe read a different manifest');assert.equal(n.manifest_sha256,digest(await bounded(working,8*1024*1024)));}
    const cases=mergeProbeReceipts(manifest,producer,oracle,native,selection,expected,{deviceEnvironment}).map(value=>campaignCase(p,value,authority));receipt={...receipt,...(deviceEnvironment?{deviceEnvironment}:{}),manifest,manifestPath:working,producerCommand,nativeCommand:command,cases,status:cases.every(r=>r.pass)&&!nativeError?'PASS':'FAIL',pass:cases.every(r=>r.pass)&&!nativeError};
    const log=await readFile(path.join(dir,'native.log'),'utf8');assert.equal(executedTests(log),1,'ignored native selector must actually pass exactly once');assert.ok(!/\bSKIP\b/.test(log));if(nativeError)throw nativeError;assert.equal(receipt.pass,true,'actual probe failed or CPU fallback is not device proof');remaining();
    if(!fake)await verifyEvidenceSource(p,swRoot,inferenceRoot);
  }catch(error){receipt.status='FAIL';receipt.pass=false;receipt.refusal={code:'actual_artifact_probe_refused',detail:error.message};if(!receipt.cases.length)receipt.cases=(manifest?.cases??[{id:c.id}]).map(k=>({case_id:k.id,kind:k.kind,status:'FAIL',pass:false,device_proof:false,productPin:p.inference,nativeEvidence:structuredClone(p.nativeEvidence),refusal:receipt.refusal}));await save(path.join(dir,receiptName),receipt);throw error;}
  await save(path.join(dir,receiptName),receipt);
  const files=receiptFiles(manifest,receiptName);
  await save(path.join(dir,c.evidence),{status:'executed',proof_kind:receipt.proof_kind,caseId:c.id,backend:c.backend,sceneworks:p.sceneworks,inference:p.inference,actualArtifactProbe:receiptName,files});
  return {receipt,proofSummary:{cases:receipt.cases.length,pass:true,device_proof:true,qkv_ffn_tensor_mapping_evidence:false}};
}
export async function verifyActualArtifactProbe(p,c,dir,{allowUnitFake=false,contentReadback=false}={}) {
  const e=await json(path.join(dir,c.evidence));assert.equal(e.status,'executed');assert.ok(e.proof_kind==='campaign_native'||allowUnitFake&&e.proof_kind==='unit_fake');for(const [key,value]of Object.entries({caseId:c.id,backend:c.backend,sceneworks:p.sceneworks,inference:p.inference}))assert.equal(e[key],value);
  assert.equal(e.actualArtifactProbe,c.actualArtifactProbe.receipt);const receipt=await json(path.join(dir,e.actualArtifactProbe),256*1024*1024);assert.equal(receipt.pass,true);assert.equal(receipt.status,'PASS');assert.equal(receipt.proof_kind,e.proof_kind);
  const manifest=await json(path.join(dir,'probe-manifest.json'));validateProbeManifest(manifest,c.backend);assert.deepEqual(receipt.manifest,manifest);
  const original=await boundJSON(path.join(dir,'producer-spec.json'),c.actualArtifactProbe.spec.sha256,8*1024*1024);assert.deepEqual({...manifest,oracle:original.oracle},original,'only oracle transport may change after production');
  if(contentReadback){assert.deepEqual(await boundJSON(c.actualArtifactProbe.spec.path,c.actualArtifactProbe.spec.sha256,8*1024*1024),original);await stagingBinding(manifest,p,c,[SCENEWORKS_ROOT,p.nativeEvidence.root]);}
  const producer=await json(path.join(dir,'producer-receipt.json'));assert.equal(producer.spec_sha256,digest(await bounded(path.join(dir,'producer-spec.json'),8*1024*1024)));
  const oracle=await boundJSON(path.join(dir,'cpu-oracle.json'),manifest.oracle.sha256,256*1024*1024),selection=await json(path.join(dir,'selected-ranges.json')),native=await Promise.all(manifest.cases.map(k=>json(path.join(dir,'native-receipts',nativeReceiptName(manifest.backend,k.id)),256*1024*1024)));
  assert.deepEqual(receipt.nativeEvidence,p.nativeEvidence);assert.equal(receipt.productPin,p.inference);provenance(manifest.oracle.producer_provenance,FROZEN_PROVENANCE);
  for(const n of native){assert.ok(await sameFile(n.manifest_path,receipt.manifestPath),'native probe read a different manifest');assert.equal(n.manifest_sha256,digest(await bounded(path.join(dir,'probe-manifest.json'),8*1024*1024)));}
  if(contentReadback)assert.deepEqual(await extractSelectedRanges(manifest,validateProbeManifest(manifest,c.backend)),selection,'selected source bytes changed since native proof');
  const authority=await readOracleAuthority(dir,{retained:true,unitFake:e.proof_kind==='unit_fake'&&allowUnitFake});
  if(c.backend==='mlx')assert.ok(receipt.deviceEnvironment&&Object.hasOwn(receipt.deviceEnvironment,'MLX_USE_GPU'),'MLX device proof needs the recorded probe environment');else assert.equal(receipt.deviceEnvironment,undefined);
  const merged=mergeProbeReceipts(manifest,producer,oracle,native,selection,FROZEN_PROVENANCE,{deviceEnvironment:receipt.deviceEnvironment}).map(value=>campaignCase(p,value,authority));assert.ok(merged.every(r=>r.pass),'not all native probes passed on device');
  // CUDA receipts retained before deviceProofBasis existed (runs at 4c3625898 and the CUDA subset runs) carry
  // no basis; their rule is unchanged, so the recomputed basis is compared only when the receipt records one.
  const legacy=c.backend==='cuda'&&receipt.cases.every(r=>!Object.hasOwn(r,'deviceProofBasis'));if(legacy)for(const r of merged)delete r.deviceProofBasis;
  assert.deepEqual(receipt.cases,JSON.parse(JSON.stringify(merged)),'retained receipt differs from the recomputed merge'); // The receipt went through JSON, which writes -0 as 0; strict deepEqual would refuse a native -0.0.
  assert.deepEqual(e.files,receiptFiles(manifest,c.actualArtifactProbe.receipt),'missing or extra proof files');
  const files=[];for(const name of e.files){assert.ok(typeof name==='string'&&!path.isAbsolute(name)&&below(path.resolve(dir),path.resolve(dir,name)),'receipt file escapes case');const canonical=await realpath(path.join(dir,name));assert.ok(below(await realpath(dir),canonical),'receipt symlink escapes case');files.push({path:name,sha256:digest(await bounded(canonical,256*1024*1024))});}
  assert.equal(executedTests(await readFile(path.join(dir,'native.log'),'utf8')),1);
  return {evidenceSha256:digest(await bounded(path.join(dir,c.evidence),8*1024*1024)),processLogSha256:digest(await bounded(path.join(dir,'process.log'),64*1024*1024)),files,numeric:{kind:'actual-artifact-probe',cases:merged.length,passed:merged.length,device_proof:true,qkv_ffn_tensor_mapping_evidence:false}};
}
