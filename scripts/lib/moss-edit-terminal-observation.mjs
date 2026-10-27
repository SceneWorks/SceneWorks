import { sha, canonical, requireThat } from './moss-edit-terminal.mjs';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { stripJsoncComments } from './jsonc.mjs';

export const OBSERVATION_METHOD='source-bound-moss-boundaries/v1';
export function hostBudgetHeader() {
  // Deliberately unsupported declaration, not a complete otherwise-valid tensor.
  let header="{'descr': '<f4', 'fortran_order': False, 'shape': (1, 134217729), }";
  header+=' '.repeat((64-((10+header.length+1)%64))%64)+'\n';
  const prefix=Buffer.from([0x93,78,85,77,80,89,1,0,0,0]);prefix.writeUInt16LE(header.length,8);return Buffer.concat([prefix,Buffer.from(header)]);
}
export function hostGuardSpec() {
  return {upload:{file:'guard-fixtures/host-declared-over-budget.npy',sha256:sha(hostBudgetHeader()),filename:'host-declared-over-budget.npy',mime:'application/x-npy'},expectedStatus:400,errorIncludes:['tensor payload exceeds budget'],noProviderLoadProbe:OBSERVATION_METHOD,invariant:'host-tensor-bound.json',coverage:'unreachable aggregate invariant plus actual declared per-artifact import budget refusal'};
}
export function deriveHostTensorBound(text) {
  const {advanced,moss,tensor,manifest}=text;
  const number=(source,pattern)=>Number(source.match(pattern)?.[1]);
  const observed={batch:number(advanced,/meta\.shape\[0\]\s*>\s*(\d+)/),context:number(advanced,/if !\(1\.\.=(\d+)\)\.contains\(&context\)/),embeddingWidth:number(advanced,/meta\.shape\[2\]\s*!=\s*(\d+)/),
    images:number(advanced,/filter\(\|n\|\s*\*n > 0 && \*n <= (\d+)\)/),geometry:number(moss,/if value > (\d+)/),
    heightStride:number(advanced,/native\.height as usize \/ (\d+)/),widthStride:number(advanced,/native\.width as usize \/ (\d+)/),
    channels:number(advanced,/images,\s*(\d+),\s*native\.height/),artifactMiB:number(tensor,/MAX_DATA_BYTES: usize = (\d+) \* 1024 \* 1024/),header:number(tensor,/MAX_HEADER_BYTES: usize = (\d+)/)};
  requireThat(canonical(observed)===canonical({batch:8,context:4096,embeddingWidth:4096,images:8,geometry:4096,heightStride:16,widthStride:16,channels:128,artifactMiB:512,header:4096}),'integrated source tensor envelope changed; reconcile this source-owned invariant before freeze');
  requireThat(/MAX_REQUEST_BYTES: usize = 1280 \* 1024 \* 1024 \+ 128 \* 1024/.test(tensor)&&/byte_length: data/.test(tensor)&&/"<f4" => "float32"/.test(tensor)&&/"\|u1" \| "<u1" => "uint8"/.test(tensor),'integrated NPY dtype/data/request-bound semantics changed');
  requireThat(/m\.dtype != "uint8" \|\| m\.shape != meta\.shape\[\.\.2\]/.test(advanced)&&/\.metadata\.as_ref\(\)\.expect\("resolved metadata"\)\.byte_length/.test(advanced)&&/\*n <= tensor_artifact::MAX_REQUEST_BYTES/.test(advanced),'integrated mask dtype or aggregate operand changed');
  const fields=advanced.match(/pub struct EditAdvanced \{([\s\S]*?)\n\}/)?.[1].match(/pub [a-z_]+:/g)?.map(field=>field.slice(4,-1)).sort();
  requireThat(canonical(fields)===canonical(['positive','negative','negative_prompts','noise','output'].sort()),'advanced tensor carrier changed; do not certify a stale five-field sum');
  const embeddingsFields=advanced.match(/pub struct EmbeddingsRef \{([\s\S]*?)\n\}/)?.[1].match(/pub [a-z_]+:/g)?.map(field=>field.slice(4,-1)).sort();
  const references=advanced.match(/pub fn references\([\s\S]*?\n\}/)?.[0],versions=tensor.match(/let \(n, start\) = match prefix\[6\] \{([\s\S]*?)\n    \};/)?.[1];
  requireThat(canonical(embeddingsFields)===canonical(['mask','values'])&&(references?.match(/refs\.push\(/g)??[]).length===3&&references.includes('[&options.positive, &options.negative]')&&references.includes('refs.push(noise)'),'five-field tensor references changed');
  requireThat((versions?.match(/=>/g)??[]).length===3&&versions.includes('2 | 3 =>')&&versions.includes('unsupported NPY version')&&/12,\s*\)/.test(versions),'permitted NPY framing/version bound changed');
  const models=JSON.parse(stripJsoncComments(manifest)).models;requireThat(['moss_image_base','moss_image_sft'].every(id=>models.find(model=>model.id===id)?.limits?.maxDimension===4096),'integrated model geometry envelope changed');
  const embeddings=8*4096*4096*4,mask=8*4096,noise=8*128*256*256*4,sum=2*embeddings+2*mask+noise,cap=1280*1024*1024+128*1024;
  return {kind:'source-bound-unreachable-aggregate-invariant/v1',observed,fields:5,maxima:{positive:embeddings,negative:embeddings,positiveMask:mask,negativeMask:mask,noise},maximumDataSum:sum,maxRequestBytes:cap,headroom:cap-sum,
    maximumFileFraming:5*(4096+12),maximumOwnedHostBytes:sum+embeddings+4096+12,
    excludes:['reference image pixels','prompt and negative strings','NPY framing'],
    conclusion:'aggregate sum cannot exceed MAX_REQUEST_BYTES through otherwise-valid public tensor inputs; no aggregate refusal is claimed',
    actualRefusal:{layer:'NPY declared per-artifact data budget before payload-length validation',dtype:'float32',shape:[1,134217729],declaredDataBytes:512*1024*1024+4,perArtifactBytes:512*1024*1024,submittedBytes:hostBudgetHeader().length,sha256:sha(hostBudgetHeader()),error:'tensor payload exceeds budget',completeTensor:false}};
}
export async function hostTensorInvariant(sourceRoot) {
  const paths={advanced:'crates/sceneworks-core/src/moss_image_advanced.rs',moss:'crates/sceneworks-core/src/moss_image.rs',tensor:'crates/sceneworks-core/src/tensor_artifact.rs',manifest:'config/manifests/builtin.models.jsonc'},text={},sources={};
  for(const [key,file] of Object.entries(paths)){const bytes=await readFile(resolve(sourceRoot,file));text[key]=bytes.toString('utf8');sources[key]={file,byteLength:bytes.length,sha256:sha(bytes)};}
  return {...deriveHostTensorBound(text),sources};
}
export function inspectJournal(bytes,expected) {
  const text=Buffer.isBuffer(bytes)?bytes.toString('utf8'):bytes;
  requireThat(text.endsWith('\n')&&Buffer.byteLength(text)<=128*1024*1024,'missing/truncated/oversized MOSS observation journal');
  let previous='0'.repeat(64),sequence=0,identity,ready=false,active=null,last;
  const counts={allocationCapableEntries:0,providerLoads:0,providerRuns:0,boundedContextEntries:0,cacheDispatchEntries:0},jobs=[];
  for(const line of text.trimEnd().split('\n')) {
    const envelope=JSON.parse(line),body=JSON.parse(envelope.body);
    requireThat(envelope.previous===previous&&envelope.sha256===sha(`${previous}\n${envelope.body}`)&&body.sequence===++sequence&&body.version===1,'MOSS journal append integrity/sequence failed');
    requireThat(Number.isFinite(Date.parse(body.at)),'MOSS journal timestamp missing');
    identity??=body.identity;requireThat(canonical(body.identity)===canonical(identity),'MOSS journal producer changed');
    for(const [key,value] of Object.entries(expected))requireThat(identity[key]===value,`MOSS journal ${key} differs from frozen source/worker/binary`);
    requireThat(identity.coverage==='moss_image_base+sft; bounded cache-context before bind; cache-dispatch; runtime load before media catalog; actual MOSS generate/edit calls after admission','unknown MOSS instrumentation coverage');
    if(sequence===1)requireThat(body.event==='startup_begin'&&body.jobId===null,'actual worker startup marker missing');
    else if(body.event==='ready_after_bounded_startup_probe'){requireThat(!ready&&active===null,'duplicate/late worker ready marker');ready=true;}
    else if(body.event==='heartbeat')requireThat(ready&&body.jobId===active,'missing ready marker or journal heartbeat lost job binding');
    else if(body.event==='job_begin'){requireThat(ready&&active===null&&body.jobId,'MOSS job bypassed ready/scope binding');active=body.jobId;jobs.push({jobId:active,startSequence:sequence});}
    else if(body.event==='job_end'){requireThat(active&&body.jobId===active,'MOSS job-end binding differs');jobs.at(-1).endSequence=sequence;active=null;}
    else {
      requireThat(ready&&active&&body.jobId===active&&['moss_image_base','moss_image_sft'].includes(body.engine),'MOSS boundary lacks actual worker/job binding');
      requireThat(['cache_context','cache_dispatch','provider_load','provider_run'].includes(body.event),'unknown MOSS observation event');
      if(['provider_load','provider_run'].includes(body.event))counts.allocationCapableEntries++;
      if(body.event==='cache_context')counts.boundedContextEntries++;
      if(body.event==='cache_dispatch')counts.cacheDispatchEntries++;
      if(body.event==='provider_load')counts.providerLoads++;
      if(body.event==='provider_run')counts.providerRuns++;
    }
    requireThat(Object.entries(counts).every(([key,value])=>body[key]===value),'MOSS boundary counters were omitted, reset or invented');
    previous=envelope.sha256;last=body;
  }
  requireThat(ready&&last?.event==='heartbeat'&&active===null,'live instrumented idle worker heartbeat missing');
  return {identity,counts,jobs,sequence,head:previous,lastAt:last.at,bytes:Buffer.byteLength(text)};
}
export function noLoadComparison(beforeBytes,afterBytes,expected,{beforeAt,afterAt,jobId=null}) {
  const before=inspectJournal(beforeBytes,expected),after=inspectJournal(afterBytes,expected);
  requireThat(Buffer.from(afterBytes).subarray(0,Buffer.byteLength(beforeBytes)).equals(Buffer.from(beforeBytes)),'MOSS journal was truncated/replaced between refusal observations');
  requireThat(after.sequence>before.sequence&&Date.parse(before.lastAt)<=Date.parse(beforeAt)&&Date.parse(beforeAt)-Date.parse(before.lastAt)<=3000&&Date.parse(after.lastAt)>=Date.parse(afterAt),'refusal lacks source-bound live before/after worker observations');
  requireThat(['allocationCapableEntries','providerLoads','providerRuns'].every(key=>before.counts[key]===after.counts[key]),'refusal entered an allocation-capable MOSS/provider path');
  const jobs=after.jobs.filter(job=>job.startSequence>before.sequence);
  if(jobId)requireThat(jobs.length===1&&jobs[0].jobId===jobId&&jobs[0].endSequence>jobs[0].startSequence&&!before.jobs.some(job=>job.jobId===jobId),'actual refused worker job scope missing/different/incomplete');
  else requireThat(jobs.length===0&&canonical(before.counts)===canonical(after.counts),'HTTP refusal unexpectedly entered a worker/cache path');
  return {method:OBSERVATION_METHOD,identity:before.identity,jobId,before:{...before.counts,sequence:before.sequence,head:before.head,at:before.lastAt},after:{...after.counts,sequence:after.sequence,head:after.head,at:after.lastAt}};
}
