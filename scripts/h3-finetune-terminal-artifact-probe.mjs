// Actual selected-checkpoint operands -> pinned CPU bridge -> ignored native consumer.
// Dispatch-only adapter; no math oracle, model forward, download or license acknowledgment here.
import assert from 'node:assert/strict';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { open, readFile, writeFile, mkdir, realpath } from 'node:fs/promises';
import { authorized } from './h3-finetune-terminal-runtime.mjs';
import { runProcess, executedTests, hashFile } from './h3-finetune-terminal.mjs';

export const ACTUAL_SELECTOR='artifact_probe::actual_checkpoint_projections_match_pinned_cpu_oracle';
export const FROZEN_PROVENANCE=[
  {script:'scripts/ci/generate_minimax_h3_w4a8_fixture.py',sha256:'14c05c1dea46dfec0447d97b575a0c6f525e2d41bcc703ad16e1610828ea9676'},
  {script:'scripts/h3_convrot_int8_oracle.py',sha256:'6ec5a42f680b439b1df347363c80a06ee38693adef9b441d45d3e025f2247100'},
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
  assert.ok(path.isAbsolute(m.artifact?.path));sha(m.artifact.sha256);positive(m.artifact.bytes);
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
  const parentPath=await outside(s.parent_receipt_path,...roots,{root:p.output});
  assert.ok(c.inputs.some(i=>i.role==='staging-receipt'&&i.path===s.parent_receipt_path&&i.sha256===s.parent_receipt_sha256),'full staging parent must be input-bound');
  const parent=await boundJSON(parentPath,s.parent_receipt_sha256,64*1024*1024);
  assert.equal(parent.kind,'h3-finetune-stage-receipt');assert.equal(parent.version,1);assert.equal(parent.status,'completed');assert.equal(parent.proof_kind,'frozen_stage_content');assert.equal(parent.producer?.codeSha,p.sceneworks);assert.equal(await realpath(parent.settings.privateRoot),await realpath(p.output));
  const f=parent.files?.find(v=>v.finalPath===m.artifact.path);assert.ok(f,'artifact absent from staging parent');assert.equal(f.status,'completed');assert.equal(f.expectedBytes,m.artifact.bytes);assert.equal(f.actualBytes,m.artifact.bytes);assert.equal(f.expectedSha256,m.artifact.sha256);assert.equal(f.observedSha256,m.artifact.sha256);
  // Legal inputs are consumed as staged, never filled or inferred by this adapter.
  assert.ok(f.explicitAcknowledgment,'staging operator acknowledgment missing');return s;
}
// Header + declared selected payload ONLY. Full-file identity is trusted from the SHA-bound
// staging chain, never recomputed. Output mini preserves raw names/dtypes/shapes and exact bytes.
export async function extractSelectedRanges(m,names,output,{deadline=Infinity}={}) {
  const handle=await open(m.artifact.path,'r');
  try {
    const before=await handle.stat({bigint:true});assert.equal(before.size,BigInt(m.artifact.bytes));assert.ok(before.isFile());
    const read=async(n,position)=>{assert.ok(Date.now()<deadline,'case deadline expired during selected reads');const b=Buffer.alloc(n);let done=0;while(done<n){const r=await handle.read(b,done,n-done,position+done);assert.ok(r.bytesRead,'truncated selected range');done+=r.bytesRead;}return b;};
    const prefix=await read(8,0),length=prefix.readBigUInt64LE();assert.ok(length>0n&&length<=100000000n&&length+8n<=before.size,'invalid bounded safetensors header');
    const start=8+Number(length),header=JSON.parse((await read(Number(length),8)).toString('utf8'));
    const widths={BOOL:1,U8:1,I8:1,F8_E4M3:1,F8_E5M2:1,I16:2,U16:2,F16:2,BF16:2,I32:4,U32:4,F32:4,I64:8,U64:8,F64:8};
    const spans=[];
    for(const [name,t]of Object.entries(header)){if(name==='__metadata__')continue;assert.ok(t&&Array.isArray(t.shape)&&t.shape.length<=8&&t.shape.every(v=>Number.isSafeInteger(v)&&v>=0),'invalid tensor shape');assert.ok(Object.hasOwn(widths,t.dtype),'unsupported tensor dtype');assert.ok(Array.isArray(t.data_offsets)&&t.data_offsets.length===2);const [a,b]=t.data_offsets;assert.ok(Number.isSafeInteger(a)&&Number.isSafeInteger(b)&&a>=0&&b>=a&&b<=m.artifact.bytes-start,'selected range exceeds source bounds');const elements=t.shape.reduce((a,b)=>a*BigInt(b),1n);assert.equal(elements*BigInt(widths[t.dtype]),BigInt(b-a),'tensor size differs from dtype/shape');spans.push([a,b]);}
    spans.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);let end=0;for(const [a,b]of spans){assert.equal(a,end,'noncontiguous/overlapping source ranges');end=b;}assert.equal(end,m.artifact.bytes-start);
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
function provenance(actual,expected){
  // Manifest/bridge use records (with optional upstream pins); Rust serializes its validated
  // BTreeMap as script -> SHA. These are two transports of the SAME frozen provenance set.
  const records=Array.isArray(actual)?actual:actual&&typeof actual==='object'?Object.entries(actual).map(([script,sha256])=>({script,sha256})):[];
  for(const item of records)if(item.source_revision_sha256!==undefined)assert.equal(item.source_revision_sha256,item.sha256);
  const core=list=>list.map(({script,sha256})=>({script,sha256})).sort((a,b)=>a.script.localeCompare(b.script));assert.equal(records.length,expected.length);assert.deepEqual(core(records),core(expected),'frozen producer provenance mismatch');
}
const selectedCore=items=>items.map(({name,dtype,shape,bytes,sha256})=>({name,dtype,shape,bytes,sha256})).sort((a,b)=>a.name.localeCompare(b.name));
export function mergeProbeReceipts(m,producer,oracle,native,selection,expectedProvenance) {
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
    const device=n.native.device;assert.ok(device&&typeof device.label==='string'&&device.label.length,'device identity missing');const deviceProof=n.native.device_proof===true&&device.cpu_fallback===false&&device.kind===(m.backend==='mlx'?'metal':'cuda');
    const pass=n.status==='pass'&&n.pass===true&&deviceProof&&n.comparison.max_abs<=Math.fround(c.tolerance.max_abs);
    result.push({case_id:c.id,kind:c.kind,status:pass?'PASS':'FAIL',pass,artifact:m.artifact,selected,native:n.native,oracle:n.oracle,max_abs:n.comparison.max_abs,tolerance:n.tolerance,device,cpu_fallback:device.cpu_fallback,device_proof:deviceProof,standalone_dense_fidelity_claimed:false,qkv_ffn_tensor_mapping_evidence:false,scope:'Selected packed-projection/pruned-curve comparison only; never full QKV/FFN tensor mapping or standalone dense fidelity'});
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
  return {...value,productPin:p.inference,nativeEvidence:structuredClone(p.nativeEvidence),oracle_authority:{revision,repository:pruned?'deepbeepmeep/Wan2GP':'comfyanonymous/comfy-kitchen',producer_provenance:FROZEN_PROVENANCE,format_reference_revision:pruned?revision:authority.format,meaning:'Independent pinned CPU arithmetic, not production Rust output; bridge wrapper checkout is recorded separately'},native_authority:{revision:p.nativeEvidence?.sha,ref:p.nativeEvidence?.ref},bridge_execution_revision:p.nativeEvidence?.sha};
}
async function expectedProvenance(inferenceRoot){const actual=await Promise.all(FROZEN_PRODUCERS.map(async script=>({script,sha256:digest(await bounded(path.join(inferenceRoot,script),4*1024*1024))})));provenance(actual,FROZEN_PROVENANCE);return actual;}
async function verifyEvidenceSource(p,swRoot,inferenceRoot){
  const n=p.nativeEvidence;assert.ok(n&&typeof n.ref==='string'&&n.ref.length,'explicit native evidence ref required');assert.match(n.sha??'',/^[a-f0-9]{40}$/);assert.equal(n.productPin,p.inference);assert.equal(await realpath(n.root),await realpath(inferenceRoot));
  const exec=promisify(execFile);for(const [cwd,revision]of [[swRoot,p.sceneworks],[inferenceRoot,n.sha]]){assert.equal((await exec('git',['rev-parse','HEAD'],{cwd})).stdout.trim(),revision,'evidence source revision drift');assert.equal((await exec('git',['status','--porcelain'],{cwd})).stdout.trim(),'','evidence source must stay clean');}
  // Product pin stays on main; evidence is allowed to be the coordinator's exact feature SHA.
  const cargo=await bounded(path.join(swRoot,'crates/sceneworks-worker/Cargo.toml'),1024*1024);assert.ok(cargo.toString('utf8').includes(`rev = "${p.inference}"`),'product inference pin drift');
}
const receiptFiles=(m,name)=>[...FROZEN_PRODUCERS.map((_,index)=>`frozen-producer-${index}.py`),name,'producer-spec.json','probe-manifest.json','selected-ranges.json','producer-receipt.json','cpu-oracle.json','producer.log','native.log',...m.cases.map(k=>`native-receipts/${m.backend}.${k.id}.json`)];
export async function executeActualArtifactProbe(p,c,swRoot,inferenceRoot,dir,deps={}) {
  const fake=deps.proof_kind==='unit_fake',runner=deps.runProcess??runProcess,contract=c.actualArtifactProbe,receiptName=contract?.receipt??'artifact-probe-receipt.json';assert.equal(path.basename(receiptName),receiptName);
  let manifest,receipt={kind:'actual-artifact-probe',version:1,proof_kind:fake?'unit_fake':'campaign_native',caseId:c.id,backend:c.backend,sceneworks:p.sceneworks,inference:p.inference,status:'FAIL',pass:false,cases:[],productPin:p.inference,nativeEvidence:structuredClone(p.nativeEvidence),whole_file_digest_recomputed:false};
  try {
    const deadline=process.env.H3_CASE_DEADLINE_MS===undefined?Date.now()+c.timeoutSeconds*1000:Number(process.env.H3_CASE_DEADLINE_MS);
    const remaining=()=>{assert.ok(Number.isFinite(deadline)&&deadline>Date.now(),'case deadline expired');return Math.min(c.timeoutSeconds,(deadline-Date.now())/1000);};remaining();
    if(!fake){authorized(p,c);assert.equal(process.env.GITHUB_EVENT_NAME,'workflow_dispatch','actual probe is dispatch-only');await verifyEvidenceSource(p,swRoot,inferenceRoot);}
    await outside(dir,swRoot,inferenceRoot,{root:p.output});assert.equal(contract?.kind,'actual-artifact-probe');assert.ok(contract.noAV===true);
    const spec=await outside(contract.spec.path,swRoot,inferenceRoot,{root:p.output});assert.ok(c.inputs.some(i=>i.role==='actual-artifact-probe-spec'&&i.path===contract.spec.path&&i.sha256===contract.spec.sha256));
    manifest=await boundJSON(spec,contract.spec.sha256,8*1024*1024);const names=validateProbeManifest(manifest,c.backend);receipt.artifact=manifest.artifact;receipt.staging=manifest.staging;
    await outside(manifest.artifact.path,swRoot,inferenceRoot,{root:p.output});assert.ok(c.inputs.some(i=>i.role==='checkpoint'&&i.path===manifest.artifact.path&&i.sha256===manifest.artifact.sha256),'actual source must be a bound checkpoint');
    await stagingBinding(manifest,p,c,[swRoot,inferenceRoot]);
    receipt.nativeEvidence=structuredClone(p.nativeEvidence);receipt.productPin=p.inference;
    const expected=fake?FROZEN_PROVENANCE:await expectedProvenance(inferenceRoot);if(manifest.oracle?.producer_provenance)provenance(manifest.oracle.producer_provenance,expected);
    const authority=await readOracleAuthority(inferenceRoot,{snapshotDir:dir,unitFake:fake});for(const k of manifest.cases)campaignCase(p,k,authority);
    const mini=path.join(dir,'selected-tensors.safetensors'),selection=await extractSelectedRanges(manifest,names,mini,{deadline});await save(path.join(dir,'selected-ranges.json'),selection);
    const working=path.join(dir,'probe-manifest.json'),oraclePath=path.join(dir,'cpu-oracle.json'),producerPath=path.join(dir,'producer-receipt.json'),receiptDir=path.join(dir,'native-receipts');await mkdir(receiptDir);
    const specBytes=await bounded(spec,8*1024*1024);assert.equal(digest(specBytes),contract.spec.sha256);await writeFile(working,specBytes,{flag:'wx'});await writeFile(path.join(dir,'producer-spec.json'),specBytes,{flag:'wx'});
    const producerCommand=['python3','-I',path.join(inferenceRoot,'scripts/h3_actual_checkpoint_oracle.py'),'--spec',working,'--selected-tensors',mini,'--output',oraclePath,'--receipt',producerPath];
    if(c.actualArtifactProbe.referenceDir){const referenceRoot=await outside(contract.referenceDir,swRoot,inferenceRoot);const references=c.inputs.filter(i=>i.role==='reference'&&below(referenceRoot,path.resolve(i.path)));assert.ok(references.length,'local pinned reference sources must be input-bound');for(const input of references){const file=await outside(input.path,swRoot,inferenceRoot,{root:referenceRoot});sha(input.sha256);assert.equal(digest(await bounded(file,8*1024*1024)),input.sha256,'local pinned reference source digest mismatch');}producerCommand.push('--reference-dir',contract.referenceDir);}
    const env={...process.env,CARGO_TARGET_DIR:path.join(p.output,`native-target-${p.inference}-${c.backend}`),CARGO_PROFILE_TEST_DEBUG:'0',CARGO_PROFILE_DEV_DEBUG:'0',...(c.backend==='cuda'?{RUSTC_WRAPPER:''}:{})};
    await runner(producerCommand,{cwd:inferenceRoot,env,logPath:path.join(dir,'producer.log'),timeoutSeconds:remaining()});
    const producer=await json(producerPath),oracleBytes=await bounded(oraclePath,256*1024*1024),oracle=JSON.parse(oracleBytes.toString('utf8'));assert.equal(producer.spec_sha256,digest(await bounded(working,8*1024*1024)));provenance(oracle.producer_provenance,expected);
    assert.equal(producer.status,'ok','producer did not succeed');provenance(producer.producer_provenance,expected);assert.equal(producer.selected_tensors_sha256,await hashFile(mini),'producer did not bind exact selected mini');
    manifest.oracle={path:oraclePath,sha256:digest(oracleBytes),producer:oracle.producer,producer_version:oracle.producer_version,artifact_sha256:oracle.artifact_sha256,producer_provenance:oracle.producer_provenance};await writeFile(working,JSON.stringify(manifest,null,2)+'\n');
    const command=['cargo','test','--locked','--manifest-path',path.join(inferenceRoot,'Cargo.toml'),'-p',c.backend==='mlx'?'mlx-gen-minimax-h3':'candle-gen-minimax-h3',...(c.backend==='cuda'?['--features','cuda']:[]),'--test','integration',ACTUAL_SELECTOR,'--','--exact','--ignored','--nocapture','--test-threads=1'];
    let nativeError;try{await runner(command,{cwd:inferenceRoot,env:{...env,H3_ACTUAL_PROBE_MANIFEST:working,H3_ACTUAL_PROBE_RECEIPT_DIR:receiptDir},logPath:path.join(dir,'native.log'),timeoutSeconds:remaining()});}catch(error){nativeError=error;}
    const native=await Promise.all(manifest.cases.map(k=>json(path.join(receiptDir,`${manifest.backend}.${k.id}.json`),256*1024*1024)));
    for(const n of native){assert.equal(n.manifest_path,working);assert.equal(n.manifest_sha256,digest(await bounded(working,8*1024*1024)));}
    const cases=mergeProbeReceipts(manifest,producer,oracle,native,selection,expected).map(value=>campaignCase(p,value,authority));receipt={...receipt,manifest,producerCommand,nativeCommand:command,cases,status:cases.every(r=>r.pass)&&!nativeError?'PASS':'FAIL',pass:cases.every(r=>r.pass)&&!nativeError};
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
  if(contentReadback){assert.deepEqual(await boundJSON(c.actualArtifactProbe.spec.path,c.actualArtifactProbe.spec.sha256,8*1024*1024),original);await stagingBinding(manifest,p,c,[dir,p.nativeEvidence.root]);}
  const producer=await json(path.join(dir,'producer-receipt.json'));assert.equal(producer.spec_sha256,digest(await bounded(path.join(dir,'producer-spec.json'),8*1024*1024)));
  const oracle=await boundJSON(path.join(dir,'cpu-oracle.json'),manifest.oracle.sha256,256*1024*1024),selection=await json(path.join(dir,'selected-ranges.json')),native=await Promise.all(manifest.cases.map(k=>json(path.join(dir,`native-receipts/${manifest.backend}.${k.id}.json`),256*1024*1024)));
  assert.deepEqual(receipt.nativeEvidence,p.nativeEvidence);assert.equal(receipt.productPin,p.inference);provenance(manifest.oracle.producer_provenance,FROZEN_PROVENANCE);
  for(const n of native){assert.equal(n.manifest_path,path.join(dir,'probe-manifest.json'));assert.equal(n.manifest_sha256,digest(await bounded(path.join(dir,'probe-manifest.json'),8*1024*1024)));}
  if(contentReadback)assert.deepEqual(await extractSelectedRanges(manifest,validateProbeManifest(manifest,c.backend)),selection,'selected source bytes changed since native proof');
  const authority=await readOracleAuthority(dir,{retained:true,unitFake:e.proof_kind==='unit_fake'&&allowUnitFake});
  const merged=mergeProbeReceipts(manifest,producer,oracle,native,selection,FROZEN_PROVENANCE).map(value=>campaignCase(p,value,authority));assert.ok(merged.every(r=>r.pass),'not all native probes passed on device');assert.deepEqual(receipt.cases,merged);
  assert.deepEqual(e.files,receiptFiles(manifest,c.actualArtifactProbe.receipt),'missing or extra proof files');
  const files=[];for(const name of e.files){assert.ok(typeof name==='string'&&!path.isAbsolute(name)&&below(path.resolve(dir),path.resolve(dir,name)),'receipt file escapes case');const canonical=await realpath(path.join(dir,name));assert.ok(below(await realpath(dir),canonical),'receipt symlink escapes case');files.push({path:name,sha256:digest(await bounded(canonical,256*1024*1024))});}
  assert.equal(executedTests(await readFile(path.join(dir,'native.log'),'utf8')),1);
  return {evidenceSha256:digest(await bounded(path.join(dir,c.evidence),8*1024*1024)),processLogSha256:digest(await bounded(path.join(dir,'process.log'),64*1024*1024)),files,numeric:{kind:'actual-artifact-probe',cases:merged.length,passed:merged.length,device_proof:true,qkv_ffn_tensor_mapping_evidence:false}};
}
