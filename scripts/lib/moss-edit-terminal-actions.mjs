// Purpose-specific native process, sampling and stopped-state transport for sc-26757.
// Importing this module does not start a process, device, browser or API.
import { spawn, execFile as callback } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdir, readFile, writeFile, readdir, lstat, cp, open } from 'node:fs/promises';
import { resolve, relative, dirname, basename, isAbsolute } from 'node:path';
import { deflateSync } from 'node:zlib';
import { VERSION, sha, digest, canonical, requireThat, save, json, safePath, decodeRgbPng, fileIdentity, treeIdentity, validateInferencePins, validateAdmission, assertRequestBinding, replaceInputs, isDenoiseStep, Api } from './moss-edit-terminal.mjs';

const execFile = promisify(callback);
const optionalJson = path => json(path).catch(error => { if(error.code==='ENOENT')return null;throw error; });
export const R16_ORIGINAL_INVENTORY_SHA256 = '6a42624ebf84ffd06d54b7e4f845a5214bb485c12c9cf6211e475963b9488a0e';
export const H18_RETRY_TOOLING_SHA = '82c534d50b72c1a5984514b90548002da212d735';
export const H18_RETRY_RUN_ID = 40208210890;
export const H18_RETRY_ARTIFACT_ID = 12604596938;
export const H18_RETRY_INVENTORY_SHA256 = 'f849b52f0e7724e13b1f8d5a8c7f2797038e22d0d8dfb1ab8e6e2cfd9af03668';
export function validateDispatchEnvelope(raw,sourceSha,inferenceSha,dispatchSha) {
  requireThat(/^[a-f0-9]{40}$/.test(sourceSha)&&/^[a-f0-9]{40}$/.test(dispatchSha)&&/^[a-f0-9]{40}$/.test(inferenceSha),'MOSS dispatch requires exact product/tooling/inference commits');
  const value=JSON.parse(raw),required=['candidate_id','bundle_manifest_sha256','phase'],optional=['bundle_run_id','bundle_artifact_id','execution_run_id','execution_run_attempt','review_artifact_id','review_run_id','upload_path','upload_kind','tooling_sha','product_sha','original_inventory_path','original_inventory_sha256','native_run_id','native_run_attempt','retry_inventory_path','retry_inventory_sha256','retry_run_id','retry_artifact_id','mlx_run_id','mlx_run_attempt','mlx_artifact_id','mlx_tooling_sha','cuda_hf_run_id','cuda_hf_run_attempt','cuda_hf_artifact_id'];
  requireThat(value&&typeof value==='object'&&!Array.isArray(value)&&required.every(key=>typeof value[key]==='string'&&value[key])&&Object.keys(value).every(key=>[...required,...optional].includes(key)&&typeof value[key]==='string'),'MOSS dispatch JSON has missing, unknown or non-string fields');
  requireThat(/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(value.candidate_id)&&/^[a-f0-9]{64}$/.test(value.bundle_manifest_sha256),'MOSS dispatch candidate or manifest SHA is invalid');
  requireThat(['upload','execute','continue','continue-cuda','verify'].includes(value.phase),'MOSS dispatch phase must be upload, execute, continue, continue-cuda or verify');
  const empty=keys=>keys.every(key=>value[key]===undefined||value[key]==='');
  const repair=['continue','continue-cuda'].includes(value.phase)||(value.phase==='verify'&&value.tooling_sha);
  requireThat(sourceSha===dispatchSha&&(!repair?empty(['tooling_sha','product_sha']):value.tooling_sha===dispatchSha&&/^[a-f0-9]{40}$/.test(value.product_sha??'')&&value.product_sha!==dispatchSha),'MOSS workflow commit must equal the exact product source or separately identify H17 product and H18 repair tooling');
  if(value.phase==='upload') {
    requireThat(['bundle','review'].includes(value.upload_kind)&&typeof value.upload_path==='string'&&value.upload_path.startsWith('/Users/michael/.codex/epic-runs/sc-26752/')&&resolve(value.upload_path)===value.upload_path,'MOSS upload requires a canonical directory inside the owned epic run');
    requireThat(empty(['bundle_run_id','bundle_artifact_id','execution_run_id','execution_run_attempt','review_artifact_id','review_run_id','tooling_sha','product_sha','original_inventory_path','original_inventory_sha256','native_run_id','native_run_attempt','retry_inventory_path','retry_inventory_sha256','retry_run_id','retry_artifact_id','mlx_run_id','mlx_run_attempt','mlx_artifact_id','mlx_tooling_sha','cuda_hf_run_id','cuda_hf_run_attempt','cuda_hf_artifact_id']),'MOSS upload cannot consume execution or prior artifact IDs');
  } else {
    requireThat([value.bundle_run_id,value.bundle_artifact_id].every(id=>/^[1-9][0-9]*$/.test(id??'')),'MOSS execution/verify requires exact bundle IDs');
    requireThat(empty(['upload_path','upload_kind']),'MOSS execution/verify cannot upload host files');
    if(value.phase==='verify') {
      requireThat([value.execution_run_id,value.review_artifact_id,value.review_run_id??value.bundle_run_id,value.execution_run_attempt??'1'].every(id=>/^[1-9][0-9]*$/.test(id??'')),'MOSS verify requires exact execution run, attempt and review artifact/run IDs');
      requireThat(empty(['original_inventory_path','original_inventory_sha256','retry_inventory_path','retry_inventory_sha256','retry_run_id','retry_artifact_id','mlx_run_id','mlx_run_attempt','mlx_artifact_id','mlx_tooling_sha','cuda_hf_run_id','cuda_hf_run_attempt','cuda_hf_artifact_id'])&&(!repair?empty(['native_run_id','native_run_attempt']):[value.native_run_id,value.native_run_attempt].every(id=>/^[1-9][0-9]*$/.test(id??''))),'MOSS repaired verify requires original native run/attempt and no host inventory path');
    } else if(value.phase==='continue') {
      requireThat([value.execution_run_id,value.execution_run_attempt??'1'].every(id=>/^[1-9][0-9]*$/.test(id??''))&&empty(['review_artifact_id','review_run_id','native_run_id','native_run_attempt','mlx_run_id','mlx_run_attempt','mlx_artifact_id','mlx_tooling_sha','cuda_hf_run_id','cuda_hf_run_attempt','cuda_hf_artifact_id']),'MOSS continuation requires one exact original execution attempt and no review/native override');
      requireThat(typeof value.original_inventory_path==='string'&&value.original_inventory_path.startsWith('/Users/michael/.codex/epic-runs/sc-26752/')&&resolve(value.original_inventory_path)===value.original_inventory_path&&value.original_inventory_sha256===R16_ORIGINAL_INVENTORY_SHA256,'MOSS continuation requires the exact sealed local original-evidence boundary');
      const retry=value.retry_inventory_path!==undefined||value.retry_inventory_sha256!==undefined;
      if(retry)requireThat(value.tooling_sha!==H18_RETRY_TOOLING_SHA&&value.retry_inventory_path?.startsWith('/Users/michael/.codex/epic-runs/sc-26752/')&&resolve(value.retry_inventory_path)===value.retry_inventory_path&&/^[a-f0-9]{64}$/.test(value.retry_inventory_sha256??'')&&(value.retry_inventory_sha256===H18_RETRY_INVENTORY_SHA256?empty(['retry_run_id','retry_artifact_id']):[value.retry_run_id,value.retry_artifact_id].every(id=>/^[1-9][0-9]*$/.test(id??''))),'MOSS retry needs a distinct tooling commit and exact sealed or downloaded failed-attempt artifact');
      else requireThat(empty(['retry_run_id','retry_artifact_id']),'MOSS continuation cannot name an artifact without a retry inventory');
    } else if(value.phase==='continue-cuda') {
      requireThat([value.execution_run_id,value.execution_run_attempt??'1',value.mlx_run_id,value.mlx_run_attempt,value.mlx_artifact_id,value.cuda_hf_run_id,value.cuda_hf_run_attempt,value.cuda_hf_artifact_id].every(id=>/^[1-9][0-9]*$/.test(id??''))&&/^[a-f0-9]{40}$/.test(value.mlx_tooling_sha??'')&&value.mlx_tooling_sha!==value.tooling_sha&&value.mlx_run_id!==value.execution_run_id&&value.cuda_hf_run_id!==value.execution_run_id&&value.cuda_hf_run_id!==value.mlx_run_id&&empty(['original_inventory_path','original_inventory_sha256','retry_inventory_path','retry_inventory_sha256','retry_run_id','retry_artifact_id','review_artifact_id','review_run_id','native_run_id','native_run_attempt']),'MOSS CUDA-only continuation requires distinct completed MLX and stopped CUDA HF artifacts');
    } else requireThat(empty(['execution_run_id','execution_run_attempt','review_artifact_id','review_run_id','tooling_sha','product_sha','original_inventory_path','original_inventory_sha256','native_run_id','native_run_attempt','retry_inventory_path','retry_inventory_sha256','retry_run_id','retry_artifact_id','mlx_run_id','mlx_run_attempt','mlx_artifact_id','mlx_tooling_sha','cuda_hf_run_id','cuda_hf_run_attempt','cuda_hf_artifact_id']),'MOSS execute cannot consume prior execution or review evidence');
  }
  const legacy=optional.slice(0,8),repairFields=optional.slice(8);
  return Object.fromEntries([...required,...legacy,...repairFields.filter(key=>value[key]!==undefined)].map(key=>[key,value[key]??(key==='execution_run_attempt'?'1':key==='review_run_id'&&value.phase==='verify'?value.bundle_run_id:'')]));
}
export const CUDA_UUID = 'GPU-e4b79931-7be6-f216-460a-f5405cfafffe';
export const CUDA_PCI = '00000000:C1:00.0';
export const delay = ms => new Promise(done=>setTimeout(done,ms));
export function contained(root,path) {
  const part=relative(resolve(root),resolve(path));
  requireThat(part && part!=='..' && !part.startsWith(`..${process.platform==='win32'?'\\':'/'}`) && !isAbsolute(part),'campaign path escaped its owned directory');
  return resolve(path);
}
export async function filesUnder(root) {
  const files=[];
  const visit=async dir=>{for(const name of (await readdir(dir)).sort()) {
    const path=resolve(dir,name),info=await lstat(path);requireThat(!info.isSymbolicLink(),'campaign transport refuses symlinks/junctions');
    if(info.isDirectory())await visit(path);else {requireThat(info.isFile(),'campaign transport accepts regular files only');files.push({file:relative(root,path).split('\\').join('/'),...await fileIdentity(path)});}
  }};await visit(resolve(root));return files;
}
export async function assertOriginalEvidence(root,manifest,{exact=false}={}) {
  requireThat(Array.isArray(manifest?.files)&&manifest.files.length>0&&new Set(manifest.files.map(item=>item.file)).size===manifest.files.length,'original evidence inventory has missing or duplicate paths');
  if(exact)requireThat(canonical(await filesUnder(root))===canonical(manifest.files),'original evidence bytes/files changed before continuation');
  else for(const item of manifest.files) {
    const path=safePath(root,item.file),info=await lstat(path);
    requireThat(info.isFile()&&!info.isSymbolicLink()&&canonical(await fileIdentity(path))===canonical({byteLength:item.byteLength,sha256:item.sha256}),`original evidence bytes changed: ${item.file}`);
  }
  return manifest.files.length;
}
export async function assertH18CheckpointFiles(root,checkpoint,{retained=false,archiveTag='h18'}={}) {
  const tails=checkpoint.appendOnlyTails;
  requireThat(Array.isArray(checkpoint.files)&&checkpoint.files.length>=300&&new Set(checkpoint.files.map(item=>item.file)).size===checkpoint.files.length&&Array.isArray(tails)&&new Set(tails.map(item=>item.file)).size===tails.length&&tails.length<=3&&tails.every(item=>archiveTag==='h18'?['api-continuation.log','journal-continuation.jsonl'].includes(item.file):/^(?:api-continuation-retry(?:-[2-9][0-9]*)?\.log|journal-continuation-retry(?:-[2-9][0-9]*)?\.jsonl|samples\/[a-z0-9-]+-retry-[2-9][0-9]*\.jsonl)$/.test(item.file))&&(archiveTag!=='h18'||canonical(tails.map(item=>item.file).sort())===canonical(['api-continuation.log','journal-continuation.jsonl'])),'prior artifact inventory or exact post-upload append-only logs changed');
  for(const item of checkpoint.files) {
    const tail=tails.find(entry=>entry.file===item.file);
    const relativeFile=retained?(item.file==='evidence/attempts/mlx/dispatch.lock'?`continuation-${archiveTag}-dispatch.lock`:item.file.startsWith('evidence/')?item.file.slice('evidence/'.length):`continuation-${archiveTag}-root/${item.file}`):item.file;
    const path=safePath(root,relativeFile),info=await lstat(path);
    requireThat(info.isFile()&&!info.isSymbolicLink(),`H18 checkpoint file is not regular: ${item.file}`);
    if(tail) {
      const bytes=await readFile(path);
      requireThat(tail.stoppedByteLength===bytes.length&&tail.stoppedSha256===sha(bytes)&&item.byteLength<=bytes.length&&item.sha256===sha(bytes.subarray(0,item.byteLength)),`H18 append-only artifact prefix or stopped bytes changed: ${item.file}`);
    } else requireThat(canonical(await fileIdentity(path))===canonical({byteLength:item.byteLength,sha256:item.sha256}),`H18 artifact bytes changed: ${item.file}`);
  }
  return checkpoint.files.length;
}
export async function continuationRemainingRows(evidenceRoot,plan) {
  const rows=plan.rows.filter(row=>row.backend==='mlx');
  requireThat(rows.length===25&&rows[6]?.id==='advanced-mlx-sft','original MLX frozen row order changed');
  for(const row of rows.slice(0,6))requireThat((await json(resolve(evidenceRoot,'rows',row.id,'receipt.json'))).execution==='VERIFIED',`original verified predecessor ${row.id} missing`);
  const failed=await json(resolve(evidenceRoot,'rows/advanced-mlx-sft/failure.json'));
  requireThat(failed.rowId==='advanced-mlx-sft'&&failed.message==='per-prompt negatives gained an unsupported broadcast field','original failed seventh row is not the reviewed checker defect');
  for(const row of rows.slice(7)) {
    for(const file of [`rows/${row.id}/receipt.json`,`rows/${row.id}/failure.json`,`attempts/mlx/${row.id}.start.json`]) {
      try{await lstat(safePath(evidenceRoot,file));requireThat(false,`later row ${row.id} was already touched`);}catch(error){if(error.code!=='ENOENT')throw error;}
    }
  }
  return rows.slice(7);
}
export async function retryRemainingRows(evidenceRoot,plan,lockSha256,bindingSha256) {
  const rows=plan.rows.filter(row=>row.backend==='mlx');
  requireThat(rows.length===25&&rows[6]?.id==='advanced-mlx-sft'&&rows[7]?.id==='advanced-mlx-sft-replay','frozen MLX retry row order changed');
  let index=0;
  while(index<rows.length&&await optionalJson(resolve(evidenceRoot,'rows',rows[index].id,'receipt.json'))) {
    requireThat((await json(resolve(evidenceRoot,'rows',rows[index].id,'receipt.json'))).execution==='VERIFIED',`verified predecessor ${rows[index].id} missing`);
    index++;
  }
  const row=rows[index],id=row?.id;
  requireThat(index>=7&&row?.transport?.startsWith('ui-'),'retry frontier must be one unsubmitted UI row after verified predecessors');
  const start=await json(resolve(evidenceRoot,'attempts/mlx',`${id}.start.json`));
  requireThat(start.id===id&&start.backend==='mlx'&&start.transport==='ui'&&start.lockSha256===lockSha256&&(!bindingSha256||start.bindingSha256===bindingSha256)&&await readFile(resolve(evidenceRoot,'attempts/mlx/dispatch.lock'),'utf8')===id,'exact unresolved UI intent/dispatch lock missing');
  for(const later of rows.slice(index))for(const file of [`rows/${later.id}/receipt.json`,`rows/${later.id}/failure.json`,`attempts/mlx/${later.id}.failure.json`,`attempts/mlx/${later.id}.resolution.json`,...(later.id===id?[]:[`attempts/mlx/${later.id}.start.json`])]) {
    try{await lstat(safePath(evidenceRoot,file));requireThat(false,`unsubmitted row ${later.id} already changed`);}catch(error){if(error.code!=='ENOENT')throw error;}
  }
  return rows.slice(index);
}
export function lexicalParserRefusal(specification,response) {
  requireThat(specification?.expectedStatus===400&&typeof specification.rawJson==='string'&&canonical(specification.errorIncludes)===canonical(['Failed to parse the request body as JSON']),'only the frozen lexical training-shift refusal may be adjudicated');
  let body;try{body=JSON.parse(response?.body);}catch{}
  const detail=body?.detail;
  requireThat(response?.status===422&&response.transportError===null&&response.http?.status===422&&response.http.body===response.body&&response.http.sha256===sha(response.body)&&response.http.byteLength===Buffer.byteLength(response.body)&&Array.isArray(detail)&&detail.length===1&&detail[0]?.type==='json_invalid'&&canonical(detail[0].loc)===canonical(['body',0])&&detail[0].msg==='JSON decode error'&&detail[0].ctx?.error?.includes('Failed to parse the request body as JSON: advanced.mossImage.training_shift:'),'actual lexical parser refusal was not the source-owned HTTP 422 JSON error');
  return true;
}
export async function postRowsGuardFrontier(evidenceRoot,plan,lockSha256,bindingSha256,specification,{allowLater=false}={}) {
  const rows=plan.rows.filter(row=>row.backend==='mlx'),guards=plan.requiredGuardCases;
  requireThat(rows.length===25&&guards.length===29&&guards[18]==='training-shift-nonfinite','post-row guard frontier differs from the frozen plan');
  for(const row of rows) {
    const receipt=await json(resolve(evidenceRoot,'rows',row.id,'receipt.json'));
    requireThat(receipt.rowId===row.id&&receipt.backend==='mlx'&&receipt.execution==='VERIFIED'&&receipt.lockSha256===lockSha256&&receipt.bindingSha256===bindingSha256,`verified MLX predecessor ${row.id} changed`);
  }
  for(const name of guards.slice(0,18)) {
    const dir=resolve(evidenceRoot,'guards',`mlx-${name}`),receipt=await json(resolve(dir,'receipt.json')),proof=await json(resolve(dir,'no-provider-load.json'));
    requireThat(receipt.case===name&&receipt.backend==='mlx'&&receipt.lockSha256===lockSha256&&receipt.bindingSha256===bindingSha256&&proof.case===name&&proof.backend==='mlx'&&proof.lockSha256===lockSha256&&proof.requestSha256===receipt.requestSha256&&proof.method==='source-bound-moss-boundaries/v1',`completed MLX guard ${name} changed`);
  }
  const name=guards[18],dir=resolve(evidenceRoot,'guards',`mlx-${name}`),attempt=resolve(evidenceRoot,'attempts/mlx');
  const [raw,response,admission,start,failure]=await Promise.all([
    readFile(resolve(dir,'request.raw.json'),'utf8'),json(resolve(dir,'response.json')),json(resolve(dir,'admission.json')),
    json(resolve(attempt,`guard-${name}.start.json`)),json(resolve(attempt,`guard-${name}.failure.json`))]);
  lexicalParserRefusal(specification,response);
  requireThat(raw===specification.rawJson&&start.id===`guard-${name}`&&start.backend==='mlx'&&start.lockSha256===lockSha256&&start.bindingSha256===bindingSha256&&start.requestSha256===sha(raw)&&admission.lockSha256===lockSha256&&failure.attemptSha256===digest(start)&&failure.state==='UNRESOLVED'&&failure.jobId===null&&failure.httpStatus===422&&failure.message==='product API returned HTTP 422','original lexical guard request/refusal/failed attempt changed');
  for(const file of [`guards/mlx-${name}/receipt.json`,`guards/mlx-${name}/no-provider-load.json`,`attempts/mlx/dispatch.lock`])requireThat(!await optionalJson(resolve(evidenceRoot,file)),`unresolved lexical guard already resolved: ${file}`);
  const resolution=await optionalJson(resolve(attempt,`guard-${name}.resolution.json`)),proof=await optionalJson(resolve(attempt,`guard-${name}.reconciliation.json`));
  if(allowLater)requireThat(resolution?.state==='SOURCE_PARSER_REFUSAL'&&resolution.attemptSha256===digest(start)&&proof?.version===VERSION&&proof.kind==='source-parser-refusal-before-dispatch'&&proof.backend==='mlx'&&proof.case===name&&resolution.proofSha256===digest(proof)&&proof.attemptSha256===digest(start)&&proof.lockSha256===lockSha256&&proof.bindingSha256===bindingSha256&&proof.jobId===null&&proof.responseSha256===(await fileIdentity(resolve(dir,'response.json'))).sha256&&proof.requestSha256===sha(raw)&&proof.failureSha256===digest(failure),'historical parser refusal was not explicitly reconciled');
  else requireThat(!resolution&&!proof,'unresolved lexical guard already resolved');
  if(!allowLater)for(const later of guards.slice(19))for(const file of [`guards/mlx-${later}/receipt.json`,`attempts/mlx/guard-${later}.start.json`])requireThat(!await optionalJson(resolve(evidenceRoot,file)),`later guard ${later} already touched`);
  return {name,guardStartIndex:18,attemptSha256:digest(start),failureSha256:digest(failure),responseSha256:(await fileIdentity(resolve(dir,'response.json'))).sha256,requestSha256:sha(raw)};
}
export async function reconcileParserRefusal(evidenceRoot,plan,lockSha256,bindingSha256,specification) {
  const frontier=await postRowsGuardFrontier(evidenceRoot,plan,lockSha256,bindingSha256,specification),name=frontier.name,attempt=resolve(evidenceRoot,'attempts/mlx'),proof={version:VERSION,kind:'source-parser-refusal-before-dispatch',backend:'mlx',case:name,lockSha256,bindingSha256,attemptSha256:frontier.attemptSha256,failureSha256:frontier.failureSha256,requestSha256:frontier.requestSha256,responseSha256:frontier.responseSha256,jobId:null};
  await save(resolve(attempt,`guard-${name}.reconciliation.json`),proof);
  await save(resolve(attempt,`guard-${name}.resolution.json`),{version:VERSION,state:'SOURCE_PARSER_REFUSAL',attemptSha256:frontier.attemptSha256,proofSha256:digest(proof)});
  return frontier;
}
export async function correctedDmdRequest(evidenceRoot,backend,specification,lockSha256,bindingSha256) {
  requireThat(['mlx','cuda'].includes(backend)&&specification.expectedStatus===400&&canonical(specification.errorIncludes)===canonical(['MOSS edit requires ordinary Base/SFT, full-layer CFG, and no SLG'])&&canonical(specification.refs)===canonical(['B']),'only the frozen edit+DMD refusal may use the proven ordinary DMD configuration');
  const rowId=`ordinary-${backend}-unmerged-bf16`,dir=resolve(evidenceRoot,'rows',rowId);
  const [source,receipt,job,uploaded]=await Promise.all([json(resolve(dir,'request.json')),json(resolve(dir,'receipt.json')),json(resolve(dir,'terminal-job.json')),json(resolve(evidenceRoot,`uploaded-${backend}.json`))]);
  requireThat(receipt.rowId===rowId&&receipt.backend===backend&&receipt.execution==='VERIFIED'&&receipt.lockSha256===lockSha256&&receipt.bindingSha256===bindingSha256&&receipt.requestSha256===digest(source)&&receipt.jobSha256===digest(job)&&receipt.outputs?.length===1&&job.id===receipt.jobId&&job.status==='completed'&&source.mode==='text_to_image'&&source.model==='moss_image_sft'&&canonical(source.advanced)===canonical({steps:2,mossImage:{acceleration:'unmerged_bf16'}})&&source.projectId===uploaded.B?.projectId&&uploaded.B?.id,'ordinary native DMD positive control or ordered B asset changed');
  const frozen=replaceInputs({request:specification.request,refs:specification.refs},uploaded,source.projectId);
  requireThat(frozen.mode==='edit_image'&&frozen.model===source.model&&frozen.referenceAssetIds?.length===1&&frozen.referenceAssetIds[0]===uploaded.B.id&&frozen.advanced?.mossImageEdit,'frozen edit+DMD target/request changed');
  const request={...source,mode:'edit_image',referenceAssetIds:[uploaded.B.id],advanced:{...source.advanced,mossImageEdit:frozen.advanced.mossImageEdit}};
  return {request,correction:{version:VERSION,name:'edit-dmd-positive-control-correction',backend,frozenGuardSha256:digest(specification),frozenRequestSha256:digest(frozen),positiveRowId:rowId,positiveRequestSha256:digest(source),positiveReceiptSha256:digest(receipt),positiveJobSha256:digest(job),referenceAssetId:uploaded.B.id,changedFields:['mode','referenceAssetIds','advanced.mossImageEdit'],correctedRequestSha256:digest(request)}};
}
export async function postDmdGuardFrontier(evidenceRoot,plan,lockSha256,bindingSha256,guards,{allowLater=false}={}) {
  const lexical=await postRowsGuardFrontier(evidenceRoot,plan,lockSha256,bindingSha256,guards['training-shift-nonfinite'],{allowLater:true});
  const names=plan.requiredGuardCases,name='edit-dmd';requireThat(names.length===29&&names[21]===name&&lexical.guardStartIndex===18,'frozen DMD guard frontier changed');
  for(const prior of names.slice(18,21)) {
    const dir=resolve(evidenceRoot,'guards',`mlx-${prior}${prior==='training-shift-nonfinite'?'-retry-7':''}`),receipt=await json(resolve(dir,'receipt.json')),proof=await json(resolve(dir,'no-provider-load.json'));
    requireThat(receipt.case===prior&&receipt.backend==='mlx'&&receipt.lockSha256===lockSha256&&receipt.bindingSha256===bindingSha256&&proof.case===prior&&proof.backend==='mlx'&&proof.lockSha256===lockSha256&&proof.requestSha256===receipt.requestSha256&&proof.method==='source-bound-moss-boundaries/v1',`completed MLX guard ${prior} changed`);
  }
  const corrected=await correctedDmdRequest(evidenceRoot,'mlx',guards[name],lockSha256,bindingSha256),dir=resolve(evidenceRoot,'guards',`mlx-${name}`),attempt=resolve(evidenceRoot,'attempts/mlx');
  const [request,response,admission,start,failure]=await Promise.all([json(resolve(dir,'request.json')),json(resolve(dir,'response.json')),json(resolve(dir,'admission.json')),json(resolve(attempt,`guard-${name}.start.json`)),json(resolve(attempt,`guard-${name}.failure.json`))]);
  const detail=JSON.parse(response.body)?.detail;
  requireThat(canonical(request)===canonical(replaceInputs({request:guards[name].request,refs:guards[name].refs},await json(resolve(evidenceRoot,'uploaded-mlx.json')),corrected.request.projectId))&&response.status===400&&response.transportError===null&&response.http?.status===400&&response.http.body===response.body&&response.http.sha256===sha(response.body)&&response.http.byteLength===Buffer.byteLength(response.body)&&detail==='advanced.mossImage: unsupported: MOSS-Image: DMD uses fixed step codes, no Euler, layer skipping or SLG'&&start.id===`guard-${name}`&&start.backend==='mlx'&&start.lockSha256===lockSha256&&start.bindingSha256===bindingSha256&&start.requestSha256===digest(request)&&admission.lockSha256===lockSha256&&failure.attemptSha256===digest(start)&&failure.state==='UNRESOLVED'&&failure.jobId===null&&failure.httpStatus===400&&failure.message==='product API returned HTTP 400','original edit+DMD early general refusal or failed attempt changed');
  for(const file of [`guards/mlx-${name}/receipt.json`,`guards/mlx-${name}/no-provider-load.json`,`attempts/mlx/dispatch.lock`])requireThat(!await optionalJson(resolve(evidenceRoot,file)),`failed edit+DMD guard already produced an acceptance receipt: ${file}`);
  const resolution=await optionalJson(resolve(attempt,`guard-${name}.resolution.json`)),proof=await optionalJson(resolve(attempt,`guard-${name}.reconciliation.json`));
  if(allowLater)requireThat(resolution?.state==='SOURCE_GUARD_INPUT_CORRECTION'&&resolution.attemptSha256===digest(start)&&proof?.version===VERSION&&proof.kind==='source-guard-input-correction-before-dispatch'&&proof.backend==='mlx'&&proof.case===name&&proof.lockSha256===lockSha256&&proof.bindingSha256===bindingSha256&&proof.attemptSha256===digest(start)&&proof.failureSha256===digest(failure)&&proof.requestSha256===digest(request)&&proof.responseSha256===(await fileIdentity(resolve(dir,'response.json'))).sha256&&proof.correctedRequestSha256===digest(corrected.request)&&proof.jobId===null&&resolution.proofSha256===digest(proof),'historical edit+DMD guard was not explicitly reconciled');
  else requireThat(!resolution&&!proof,'failed edit+DMD guard was already reconciled');
  if(!allowLater)for(const later of names.slice(22))requireThat(!await optionalJson(resolve(attempt,`guard-${later}.start.json`)),`later guard ${later} was already touched`);
  return {name,guardStartIndex:21,attemptSha256:digest(start),failureSha256:digest(failure),responseSha256:(await fileIdentity(resolve(dir,'response.json'))).sha256,requestSha256:digest(request),correctedRequestSha256:digest(corrected.request),positiveRowId:corrected.correction.positiveRowId,positiveReceiptSha256:corrected.correction.positiveReceiptSha256};
}
export async function reconcileDmdInputCorrection(evidenceRoot,plan,lockSha256,bindingSha256,guards) {
  const frontier=await postDmdGuardFrontier(evidenceRoot,plan,lockSha256,bindingSha256,guards),attempt=resolve(evidenceRoot,'attempts/mlx'),name=frontier.name,proof={version:VERSION,kind:'source-guard-input-correction-before-dispatch',backend:'mlx',case:name,lockSha256,bindingSha256,attemptSha256:frontier.attemptSha256,failureSha256:frontier.failureSha256,requestSha256:frontier.requestSha256,responseSha256:frontier.responseSha256,correctedRequestSha256:frontier.correctedRequestSha256,jobId:null};
  await save(resolve(attempt,`guard-${name}.reconciliation.json`),proof);await save(resolve(attempt,`guard-${name}.resolution.json`),{version:VERSION,state:'SOURCE_GUARD_INPUT_CORRECTION',attemptSha256:frontier.attemptSha256,proofSha256:digest(proof)});return frontier;
}
export async function submittedCanceledRemainingRows(evidenceRoot,plan,lockSha256,bindingSha256,checkpoint,root) {
  const rows=plan.rows.filter(row=>row.backend==='mlx');
  requireThat(rows.length===25,'frozen MLX row count changed');
  let index=0;
  while(index<rows.length&&await optionalJson(resolve(evidenceRoot,'rows',rows[index].id,'receipt.json'))) {
    requireThat((await json(resolve(evidenceRoot,'rows',rows[index].id,'receipt.json'))).execution==='VERIFIED',`prior row ${rows[index].id} lacks a verified receipt`);
    index++;
  }
  const row=rows[index];
  requireThat(index>=8&&row?.expected==='canceled'&&row.transport==='rest'&&row.cancelAtDenoise===true,'only a known submitted terminal cancellation may be adopted');
  const dir=resolve(evidenceRoot,'rows',row.id),attempt=resolve(evidenceRoot,'attempts/mlx'),start=await json(resolve(attempt,`${row.id}.start.json`));
  const [request,submitted,admitted,observed,response]=await Promise.all(['request.json','submission.json','admission.json','cancel-observation.json','cancel-response.json'].map(name=>json(resolve(dir,name))));
  requireThat(start.id===row.id&&start.backend==='mlx'&&start.lockSha256===lockSha256&&start.bindingSha256===bindingSha256&&start.requestSha256===digest(request)&&await readFile(resolve(attempt,'dispatch.lock'),'utf8')===row.id&&submitted?.id&&submitted.status==='queued'&&submitted.projectId===request.projectId,'canceled attempt lacks the exact known submitted REST request/lock');
  assertRequestBinding(request,submitted.payload);
  validateAdmission(admitted,await json(resolve(evidenceRoot,'protocol.lock.json')),'mlx');
  requireThat(observed.id===submitted.id&&observed.status==='running'&&isDenoiseStep(observed)&&response.id===submitted.id&&response.cancelRequested===true&&response.projectId===request.projectId&&response.workerId===observed.workerId&&Date.parse(response.updatedAt)>=Date.parse(observed.updatedAt),'original denoise cancellation and native acknowledgement differ');
  for(const file of [`rows/${row.id}/receipt.json`,`rows/${row.id}/failure.json`,`attempts/mlx/${row.id}.failure.json`,`attempts/mlx/${row.id}.resolution.json`])requireThat(!await optionalJson(resolve(evidenceRoot,file)),`submitted cancellation ${row.id} was already resolved/failed`);
  for(const later of rows.slice(index+1))for(const file of [`rows/${later.id}/receipt.json`,`rows/${later.id}/failure.json`,`attempts/mlx/${later.id}.start.json`])requireThat(!await optionalJson(resolve(evidenceRoot,file)),`later row ${later.id} was already touched`);
  const samples=checkpoint.files.filter(item=>new RegExp(`^samples/${row.id}-retry-[2-9][0-9]*\\.jsonl$`).test(item.file));
  requireThat(samples.length===1,'original submitted cancellation raw sample is not sealed in the failed artifact');
  const sampleFile=samples[0].file;
  const lines=(await readFile(safePath(root,sampleFile),'utf8')).trimEnd().split('\n').map(line=>JSON.parse(line));
  requireThat(lines.length>=3&&lines[0].phase==='baseline'&&lines.at(-1).phase==='drain'&&lines.some(item=>item.phase==='active'&&Date.parse(item.at)>=Date.parse(submitted.startedAt??submitted.createdAt)&&Date.parse(item.at)<=Date.parse(response.updatedAt))&&lines.every(item=>item.device?.uuid===admitted.device.uuid&&item.raw),'original whole-device canceled-job trace is incomplete or changed');
  return {remaining:rows.slice(index+1),recovery:{rowId:row.id,jobId:submitted.id,requestSha256:digest(request),submissionSha256:digest(submitted),attemptSha256:digest(start),sampleFile}};
}
export function validateFailedUiCapture(capture,lockSha256,{historical=null,rowId='advanced-mlx-sft-replay'}={}) {
  const oldSelector=historical==='h18'&&capture.error?.includes('strict mode violation')&&capture.error.includes('Download portable workflow (.json)');
  const oldObserver=historical==='h19'&&capture.error==='observed upload lacks multipart file bytes'&&capture.actions?.some(item=>item.action==='drop-workflow-file')&&!capture.actions.some(item=>item.action==='click'&&/^Generate(?: edit| [0-9]+ images?)?$/.test(item.control));
  requireThat(capture.rowId===rowId&&capture.lockSha256===lockSha256&&typeof capture.error==='string'&&capture.error.length>0&&(oldSelector||oldObserver||(!historical&&capture.submissionPhase==='before-generate-attempt'))&&Array.isArray(capture.submissions)&&capture.submissions.length===0&&Array.isArray(capture.imageJobPosts)&&capture.imageJobPosts.length===0&&!(capture.actions??[]).some(item=>item.action==='click'&&/^Generate(?: edit| [0-9]+ images?)?$/.test(item.control)),'failed UI capture was not a proven pre-submission attempt');
  return true;
}
export function continuationRowDisposition(row,remaining,{retry=false}={}) {
  const run=remaining.some(item=>item.id===row.id);
  return {run,reuseUiIntent:run&&retry&&row.id===remaining[0]?.id&&row.transport.startsWith('ui-')};
}
export async function sourceIdentity(config) {
  const sources={sceneworks:treeIdentity(config.sourceRoot),inference:treeIdentity(config.inferenceRoot)};
  requireThat(sources.sceneworks.head===config.sourceSha && sources.inference.head===config.inferenceSha,'Actions checkout differs from the exact paired dispatch sources');
  const manifests=await Promise.all(['Cargo.toml','crates/sceneworks-worker/Cargo.toml','crates/sceneworks-memory-adapter/Cargo.toml'].map(path=>readFile(resolve(config.sourceRoot,path),'utf8')));
  validateInferencePins(manifests,await readFile(resolve(config.sourceRoot,'Cargo.lock'),'utf8'),config.inferenceSha);
  // SceneWorks is the stationary permanent-pin feature candidate: its campaign
  // must pass before that feature merges to main. Only inference has landed.
  await execFile('git',['-C',config.inferenceRoot,'merge-base','--is-ancestor',config.inferenceSha,'origin/main']);
  return sources;
}
export function childEnvironment(config,token) {
  const env={...process.env};
  for(const name of Object.keys(env))if(name.startsWith('SCENEWORKS_')||/TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY/.test(name))delete env[name];
  const state=resolve(config.stateRoot),cuda=config.backend==='cuda';
  Object.assign(env,{SCENEWORKS_DATA_DIR:resolve(state,'data'),SCENEWORKS_CONFIG_DIR:resolve(state,'config'),SCENEWORKS_CREDENTIALS_DIR:resolve(state,'credentials'),
    SCENEWORKS_JOBS_DB_PATH:resolve(state,'data','cache','jobs.db'),SCENEWORKS_ACCESS_TOKEN:token,
    SCENEWORKS_API_HOST:'127.0.0.1',SCENEWORKS_API_PORT:String(config.port),SCENEWORKS_API_URL:`http://127.0.0.1:${config.port}`,
    SCENEWORKS_WORKER_ID:config.workerId,SCENEWORKS_GPU_ID:cuda?'1':'mlx',
    SCENEWORKS_BACKEND_MLX_ENABLED:String(!cuda),SCENEWORKS_BACKEND_CANDLE_ENABLED:String(cuda),SCENEWORKS_POLL_SECONDS:'1',SCENEWORKS_HEARTBEAT_SECONDS:'2',
    HF_HOME:config.hfHome,HF_HUB_CACHE:resolve(config.hfHome,'hub'),HUGGINGFACE_HUB_CACHE:resolve(config.hfHome,'hub'),HF_XET_CACHE:resolve(config.hfHome,'xet'),HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1'});
  if(cuda)Object.assign(env,{CUDA_DEVICE_ORDER:'PCI_BUS_ID',CUDA_VISIBLE_DEVICES:CUDA_UUID});else delete env.CUDA_VISIBLE_DEVICES;
  // This is an observation journal, never hardware admission/ownership authority.
  if(config.observationJournal)env.SCENEWORKS_MOSS_OBSERVATION_JOURNAL=config.observationJournal;
  if(config.candidateId)env.SCENEWORKS_MOSS_CANDIDATE_ID=config.candidateId;
  env.SCENEWORKS_MOSS_SOURCE_SHA=config.sourceSha;env.SCENEWORKS_MOSS_INFERENCE_SHA=config.inferenceSha;
  return env;
}
// The API owns and drains this CPU loop; its identity must not replace the GPU registration.
export function utilityWorkerEnvironment(config) {
  return {SCENEWORKS_RUN_UTILITY_INPROCESS:'1',SCENEWORKS_RUST_WORKER_GPU_ID:'cpu',SCENEWORKS_WORKER_ID:`${config.workerId}-utility`,SCENEWORKS_UTILITY_WORKERS:'1'};
}
// Only standalone worker processes own the Windows stdin-EOF shutdown contract.
export function standaloneWorkerEnvironment() {
  return {SCENEWORKS_WORKER_ONLY:'1',SCENEWORKS_WORKER_CHILD:'1'};
}
// Whitelist bounded readiness fields; never persist tokens, environment or response bodies.
export function stackReadinessSnapshot({health,workers=[],children=[],workerId,httpStatus=null}) {
  const text=value=>typeof value==='string'?value.slice(0,200):null;
  return {version:VERSION,at:new Date().toISOString(),expectedWorkerId:workerId,httpStatus,
    health:{status:text(health?.status),readinessStatus:text(health?.readiness?.status)},
    workers:workers.slice(0,16).map(worker=>({id:text(worker.id),gpuId:text(worker.gpuId),status:text(worker.status),
      capabilities:Array.isArray(worker.capabilities)?worker.capabilities.slice(0,64).map(text):[]})),
    children:children.slice(0,2).map((child,index)=>({role:index===0?'api':'worker',pid:child.pid??null,exitCode:child.exitCode??null,signalCode:child.signalCode??null,startError:Boolean(child.startError)}))};
}
export function readyUtilityWorker(workers,workerId) {
  return workers.find(worker=>worker.id===`${workerId}-utility`&&worker.gpuId==='cpu'&&worker.status==='idle'&&worker.capabilities?.includes('model_download'));
}
export async function command(command,args,{cwd,env=process.env,log,timeoutMs=120*60*1000}={}) {
  await mkdir(dirname(log),{recursive:true,mode:0o700});const stream=await open(log,'wx',0o600);
  const child=spawn(command,args,{cwd,env,stdio:['ignore',stream.fd,stream.fd],windowsHide:true});
  let timer;try {
    const code=await new Promise((done,fail)=>{child.once('error',fail);child.once('exit',(code,signal)=>signal?fail(Error(`owned command exited by ${signal}`)):done(code));timer=setTimeout(()=>{child.kill();fail(Error('owned command deadline exceeded'));},timeoutMs);});
    requireThat(code===0,`owned command failed (${basename(command)}, exit ${code}); inspect its retained log`);
  } finally {clearTimeout(timer);await stream.close();}
}
export async function nativeDevice(backend,execute=execFile) {
  if(backend==='cuda') {
    const {stdout}=await execute('nvidia-smi',[`--id=${CUDA_UUID}`,'--query-gpu=index,uuid,pci.bus_id,name,memory.used','--format=csv,noheader,nounits'],{timeout:10000,windowsHide:true});
    const rows=stdout.trim().split(/\r?\n/);requireThat(rows.length===1,'CUDA UUID probe returned another device count');
    const [index,uuid,pci,name,memory]=rows[0].split(',').map(item=>item.trim());
    requireThat(index==='1'&&uuid===CUDA_UUID&&pci.toUpperCase()===CUDA_PCI,'terminal campaign is restricted to physical GPU1 UUID/PCI');
    const usedBytes=Number(memory)*1024*1024;requireThat(Number.isFinite(usedBytes)&&usedBytes>=0,'invalid whole-device CUDA memory observation');
    return {device:{uuid,pciOrRegistryId:pci,name,physicalIndex:1,logicalIndex:0},usedBytes,raw:stdout};
  }
  requireThat(backend==='mlx','unknown native backend');
  const arm=(await execute('sysctl',['-n','hw.optional.arm64'],{timeout:10000})).stdout.trim();requireThat(arm==='1','native MLX requires Apple silicon');
  const registry=(await execute('ioreg',['-rd1','-c','IOPlatformExpertDevice'],{timeout:10000})).stdout;
  const uuid=registry.match(/"IOPlatformUUID"\s*=\s*"([^"]+)"/)?.[1];requireThat(uuid,'actual Apple platform registry identity missing');
  const vm=(await execute('vm_stat',[],{timeout:10000})).stdout,total=Number((await execute('sysctl',['-n','hw.memsize'],{timeout:10000})).stdout);
  const page=Number(vm.match(/page size of (\d+) bytes/)?.[1]);
  const pages=['free','inactive','speculative'].map(name=>Number(vm.match(new RegExp(`Pages ${name}:\\s+(\\d+)\\.`))?.[1]));
  const usedBytes=total-pages.reduce((sum,n)=>sum+n,0)*page;
  requireThat(Number.isSafeInteger(total)&&total>0&&Number.isSafeInteger(usedBytes)&&usedBytes>=0&&usedBytes<=total,'invalid sampled Apple unified-memory usage');
  return {device:{uuid,pciOrRegistryId:uuid,name:'Apple Silicon unified memory',platform:'apple-silicon'},usedBytes,raw:{registry,vm,total,page,method:'whole-host unified physical memory: total minus free/inactive/speculative; includes other processes'}};
}
export async function sampler({backend,file,observe=()=>nativeDevice(backend),intervalMs=250}) {
  const samples=[];let stopped=false,pending=Promise.resolve(),timer,error;
  await mkdir(dirname(file),{recursive:true});const handle=await open(file,'wx',0o600);
  const capture=async phase=>{const actual=await observe(),sample={at:new Date().toISOString(),phase,usedBytes:actual.usedBytes};samples.push(sample);await handle.write(`${JSON.stringify({...sample,device:actual.device,raw:actual.raw})}\n`);return actual.device;};
  const device=await capture('baseline');
  const schedule=()=>{if(!stopped)timer=setTimeout(()=>{pending=capture('active').catch(failure=>{error=failure;stopped=true;}).finally(schedule);},intervalMs);};schedule();
  return {async stop(jobId,samplerSha256){stopped=true;clearTimeout(timer);await pending;try{if(error)throw error;const last=await capture('drain');requireThat(last.uuid===device.uuid,'sampled another physical device');return {version:VERSION,jobId,samplerSha256,device,samples};}finally{await handle.close();}}};
}
export async function startStack(config,token) {
  requireThat(token&&config.workerId&&Number.isInteger(config.port),'native stack requires actual token/worker/port');
  await sourceIdentity(config);await nativeDevice(config.backend);await mkdir(resolve(config.stateRoot,'data'),{recursive:true,mode:0o700});await mkdir(resolve(config.stateRoot,'credentials'),{recursive:true,mode:0o700});
  if(config.continuation) {
    requireThat(config.backend==='mlx'&&config.continuation.name==='r16-checker-recovery'&&/^[a-f0-9]{40}$/.test(config.continuation.toolingSha??''),'only the reviewed stopped MLX continuation may reopen state');
    requireThat((await filesUnder(resolve(config.stateRoot,'config'))).length>0,'original stopped stack configuration missing');
  } else await cp(resolve(config.sourceRoot,'config'),resolve(config.stateRoot,'config'),{recursive:true,errorOnExist:true,force:false});
  const env=childEnvironment(config,token),children=[],handles=[];
  const continuationSuffix=config.continuation?(config.continuation.retry?`-continuation-retry${config.continuation.retry.ordinal>1?`-${config.continuation.retry.ordinal}`:''}`:'-continuation'):'';
  const launch=async(name,extra)=>{const handle=await open(resolve(config.root,`${name}${continuationSuffix}.log`),'wx',0o600);handles.push(handle);const child=spawn(config.apiBinary,[],{cwd:config.sourceRoot,env:{...env,...extra},stdio:[name.startsWith('worker')?'pipe':'ignore',handle.fd,handle.fd],windowsHide:true});child.once('error',error=>{child.startError=error;});children.push(child);return child;};
  const api=await launch('api',utilityWorkerEnvironment(config)),client=new Api(env.SCENEWORKS_API_URL,token);let worker=await launch('worker',standaloneWorkerEnvironment()),restarts=0;
  const stopOne=async(child,isWorker)=>{if(child.exitCode!==null||child.signalCode!==null)return;let timer;try {
    await new Promise((done,fail)=>{child.once('exit',done);timer=setTimeout(()=>fail(Error('owned native process did not stop; state transfer forbidden')),30000);if(process.platform==='win32'&&isWorker)child.stdin.end();else child.kill('SIGTERM');});
  }finally{clearTimeout(timer);}};
  const stop=async()=>{await stopOne(worker,true);await stopOne(api,false);for(const handle of handles)await handle.close();};
  const restartIdleWorker=async reason=>{
    const deadline=Date.now()+120000;let idle=false;
    while(Date.now()<deadline){const current=(await client.request('/api/v1/workers')).find(item=>item.id===config.workerId);if(current?.status==='idle'&&!current.currentJobId){idle=true;break;}await delay(250);}
    requireThat(idle,'owned worker never drained to idle; refuse cache restart');
    requireThat(worker.exitCode===null&&worker.signalCode===null&&!worker.startError,'owned worker exited unexpectedly; do not silently restart a failed campaign');
    const oldPid=worker.pid;await stopOne(worker,true);requireThat(worker.exitCode!==null||worker.signalCode!==null,'owned worker cache process still running');
    const oldExit={code:worker.exitCode,signal:worker.signalCode};
    await sourceIdentity(config);await nativeDevice(config.backend);config.observationJournal=resolve(config.root,`journal-${config.continuation?(config.continuation.retry?`continuation-retry${config.continuation.retry.ordinal>1?`-${config.continuation.retry.ordinal}`:''}-`:'continuation-'):''}${++restarts}.jsonl`);
    worker=await launch(`worker-restart-${restarts}`,{...standaloneWorkerEnvironment(),SCENEWORKS_MOSS_OBSERVATION_JOURNAL:config.observationJournal});
    return {reason,oldPid,oldExit,newPid:worker.pid,journal:config.observationJournal,source:await sourceIdentity(config),workerId:config.workerId,at:new Date().toISOString(),meaning:'actual idle owned worker process exited; its native resident caches cannot be reused by reentry'};
  };
  try {
    const deadline=Date.now()+180000;let ready,utility,lastHealth,lastWorkers=[],lastHttpStatus=null;
    while(Date.now()<deadline) {
      for(const child of children)requireThat(!child.startError&&child.exitCode===null&&child.signalCode===null,'source-built API/worker exited before readiness');
      try {const health=await client.request('/api/v1/health'),workers=await client.request('/api/v1/workers');lastHealth=health;lastWorkers=workers;lastHttpStatus=null;ready=workers.find(item=>item.id===config.workerId);utility=readyUtilityWorker(workers,config.workerId);
        if(health.status==='ok'&&health.readiness?.status==='ready'&&ready?.gpuId===(config.backend==='cuda'?'1':'mlx')&&ready.status==='idle'&&utility)break;
      }catch(error){lastHttpStatus=Number.isInteger(error.status)?error.status:null;}await delay(500);
    }
    const registered=ready?.gpuId===(config.backend==='cuda'?'1':'mlx')&&ready.status==='idle'&&utility;
    if(!registered)await save(resolve(config.root,'startup-readiness.json'),stackReadinessSnapshot({health:lastHealth,workers:lastWorkers,children,workerId:config.workerId,httpStatus:lastHttpStatus}));
    requireThat(registered,'actual native/CPU model_download workers never registered on their bound identities; inspect startup-readiness.json');
    await save(resolve(config.root,config.continuation?`stack${continuationSuffix}.json`:'stack.json'),{version:VERSION,backend:config.backend,source:await sourceIdentity(config),binary:await fileIdentity(config.apiBinary),apiUrl:client.base.href,worker:ready,utilityWorker:utility,pids:{api:api.pid,worker:worker.pid},at:new Date().toISOString(),...(config.continuation?{tooling:config.continuation.toolingSha}:{})});
    return {api:client,children,stop,restartIdleWorker,async restartWorker(canceledJobId){
      const job=await client.request(`/api/v1/jobs/${encodeURIComponent(canceledJobId)}`);
    requireThat(job.status==='canceled'&&job.workerId===config.workerId&&!(job.result?.assets?.length)&&!(job.result?.assetWrites?.length),'owned cancellation is not drained; do not restart');
      return {canceledJobId,...await restartIdleWorker('canceled-job-reentry')};
    }};
  }catch(error){await stop();throw error;}
}
export async function checkpointDatabases(dataRoot,Database) {
  if(!Database)({DatabaseSync:Database}=await import('node:sqlite'));
  const checks=[];
  for(const entry of await filesUnder(dataRoot))if(entry.file.endsWith('.db')) {
    const db=new Database(resolve(dataRoot,entry.file));try {
      const checkpoint=db.prepare('PRAGMA wal_checkpoint(TRUNCATE)').get();requireThat(checkpoint.busy===0,'owned SQLite database is still busy; never copy live state');
      const integrity=db.prepare('PRAGMA integrity_check').get();requireThat(Object.values(integrity)[0]==='ok','owned SQLite integrity failed');checks.push({file:entry.file,checkpoint,integrity});
    }finally{db.close();}
  }requireThat(checks.some(item=>item.file==='cache/jobs.db'),'actual jobs database missing from stopped handoff');return checks;
}
export function relocateRegistry(registry,oldDataRoot,newDataRoot) {
  requireThat(Array.isArray(registry),'actual recent-projects.json must be an array');
  const oldPrefix=oldDataRoot.replace(/\\/g,'/').replace(/\/$/,'')+'/';
  return registry.map(item=>{requireThat(typeof item.id==='string'&&typeof item.path==='string','owned project registry entry lacks its ID/path');const path=item.path.replace(/\\/g,'/');requireThat(path.startsWith(oldPrefix)&&!path.slice(oldPrefix.length).split('/').includes('..'),'handoff includes a project outside owned state');return {...item,path:resolve(newDataRoot,...path.slice(oldPrefix.length).split('/'))};});
}
export async function exportState(config,out,Database) {
  const dataRoot=resolve(config.stateRoot,'data'),checks=await checkpointDatabases(dataRoot,Database);
  await mkdir(out,{mode:0o700});await cp(dataRoot,resolve(out,'data'),{recursive:true,errorOnExist:true,force:false});
  const files=await filesUnder(resolve(out,'data'));
  requireThat(!files.some(item=>item.file.endsWith('-wal')&&item.byteLength>0),'checkpoint left a nonempty SQLite WAL');
  await save(resolve(out,'handoff.json'),{version:VERSION,projectId:config.projectId,sourceDataRoot:dataRoot,sources:await sourceIdentity(config),checks,files});
}
export async function importState(config,from) {
  const manifest=await json(resolve(from,'handoff.json'));requireThat(manifest.version===VERSION&&manifest.projectId===config.projectId&&canonical(manifest.sources)===canonical(await sourceIdentity(config)),'cross-backend state differs from the frozen paired sources/project');
  requireThat(canonical(await filesUnder(resolve(from,'data')))===canonical(manifest.files),'stopped-state handoff bytes changed');
  const target=resolve(config.stateRoot,'data');await cp(resolve(from,'data'),target,{recursive:true,errorOnExist:true,force:false});
  const path=resolve(target,'recent-projects.json'),before=await json(path),after=relocateRegistry(before,manifest.sourceDataRoot,target);
  requireThat(after.some(item=>item.id===config.projectId),'original campaign project missing from transferred registry');
  await writeFile(path,`${JSON.stringify(after,null,2)}\n`,{mode:0o600});
  await save(resolve(config.root,'state-relocation.json'),{version:VERSION,manifestSha256:digest(manifest),before,after,changed:'owned registry location fields only; project/job/asset/payload/recipe bytes preserved'});
}
export async function validateCompletedMlxArtifact(prior,{candidateId,sourceSha,inferenceSha,originalRunId,originalAttempt,toolingSha,runId,attempt,artifactId}) {
  requireThat([originalRunId,originalAttempt,runId,attempt,artifactId].every(id=>Number.isSafeInteger(id)&&id>0)&&/^[a-f0-9]{40}$/.test(toolingSha),'completed MLX handoff requires exact run, artifact and producer identities');
  const evidence=resolve(prior,'evidence'),[marker,retry,lock,plan,binding,handoff]=await Promise.all([
    json(resolve(prior,'execution-continuation-retry-8.json')),json(resolve(evidence,'continuation-retry-8.json')),
    json(resolve(evidence,'protocol.lock.json')),json(resolve(evidence,'cases.json')),
    json(resolve(evidence,'bindings/mlx.json')),json(resolve(prior,'handoff/handoff.json'))]);
  const lockSha256=digest(lock),bindingSha256=digest(binding),rows=plan.rows.filter(row=>row.backend==='mlx'),guards=plan.requiredGuardCases;
  requireThat(marker.version===VERSION&&marker.execution==='COMPLETE'&&marker.backend==='mlx'&&marker.renders===25&&marker.refusals===29&&marker.lockSha256===lockSha256&&marker.toolingSha===toolingSha&&marker.originalRunId===originalRunId&&retry.ordinal===8&&retry.toolingSha===toolingSha&&retry.originalRunId===originalRunId&&retry.originalAttempt===originalAttempt&&retry.candidateId===candidateId&&retry.sourceSha===sourceSha&&retry.inferenceSha===inferenceSha&&retry.lockSha256===lockSha256&&lock.candidateId===candidateId&&lock.sources.sceneworks.head===sourceSha&&lock.sources.inference.head===inferenceSha&&rows.length===25&&guards.length===29&&binding.endpoint.projectId===handoff.projectId&&handoff.version===VERSION&&handoff.sources.sceneworks.head===sourceSha&&handoff.sources.inference.head===inferenceSha,'completed MLX producer, original product, frozen plan or handoff changed');
  for(const row of rows) {
    const dir=resolve(evidence,'rows',row.id),[receipt,request,job]=await Promise.all(['receipt.json','request.json','terminal-job.json'].map(name=>json(resolve(dir,name))));
    requireThat(receipt.rowId===row.id&&receipt.backend==='mlx'&&receipt.execution==='VERIFIED'&&receipt.lockSha256===lockSha256&&receipt.bindingSha256===bindingSha256&&receipt.requestSha256===digest(request)&&receipt.jobSha256===digest(job)&&receipt.jobId===job.id&&Array.isArray(receipt.outputs)&&(row.expected==='canceled'?receipt.outputs.length===0&&job.status==='canceled':receipt.outputs.length>0&&job.status==='completed'),`completed MLX row ${row.id} changed`);
    for(const output of receipt.outputs)requireThat((await fileIdentity(safePath(dir,output.file))).sha256===output.sha256,`completed MLX output ${row.id} changed`);
  }
  for(const name of guards) {
    const suffix=name==='training-shift-nonfinite'?'-retry-7':name==='edit-dmd'?'-retry-8':'',dir=resolve(evidence,'guards',`mlx-${name}${suffix}`),[receipt,proof]=await Promise.all(['receipt.json','no-provider-load.json'].map(file=>json(resolve(dir,file))));
    requireThat(receipt.case===name&&receipt.backend==='mlx'&&receipt.lockSha256===lockSha256&&receipt.bindingSha256===bindingSha256&&proof.case===name&&proof.backend==='mlx'&&proof.lockSha256===lockSha256&&proof.requestSha256===receipt.requestSha256&&proof.method==='source-bound-moss-boundaries/v1'&&proof.observation,`completed MLX refusal ${name} changed`);
  }
  requireThat(canonical(await filesUnder(resolve(prior,'handoff/data')))===canonical(handoff.files),'completed MLX stopped-state handoff bytes changed');
  return {version:VERSION,name:'completed-mlx-artifact-for-cuda',runId,attempt,artifactId,toolingSha,originalRunId,originalAttempt,candidateId,sourceSha,inferenceSha,lockSha256,bindingSha256,projectId:handoff.projectId,completionSha256:(await fileIdentity(resolve(prior,'execution-continuation-retry-8.json'))).sha256,retrySha256:(await fileIdentity(resolve(evidence,'continuation-retry-8.json'))).sha256,handoffSha256:(await fileIdentity(resolve(prior,'handoff/handoff.json'))).sha256,verifiedRows:rows.length,verifiedGuards:guards.length};
}
export const H27_HIDDEN_HANDOFF_FILES = [
  'models/.sceneworks-external-library-closures.json',
  'models/.sceneworks-external-library-state.json',
  'models/.sceneworks-external-library-state.lock',
  'models/OpenMOSS-Team__MOSS-Image-Base/.sceneworks-download-complete.json',
  'models/OpenMOSS-Team__MOSS-Image-Base/.sceneworks-download-receipt.lock',
  'models/OpenMOSS-Team__MOSS-Image-SFT/.sceneworks-download-complete.json',
  'models/OpenMOSS-Team__MOSS-Image-SFT/.sceneworks-download-receipt.lock'
];
export async function stageCompletedMlxHandoffSupplement(prior,stopped,stage,identity) {
  const local=await validateCompletedMlxArtifact(stopped,identity);
  const [artifactManifest,artifactHandoff,artifactMarker,artifactRetry]=await Promise.all([
    json(resolve(prior,'handoff/handoff.json')),
    fileIdentity(resolve(prior,'handoff/handoff.json')),
    fileIdentity(resolve(prior,'execution-continuation-retry-8.json')),
    fileIdentity(resolve(prior,'evidence/continuation-retry-8.json'))
  ]);
  requireThat(artifactHandoff.sha256===local.handoffSha256&&artifactMarker.sha256===local.completionSha256&&artifactRetry.sha256===local.retrySha256,'downloaded H27 artifact differs from completed stopped Mac producer');
  const present=await filesUnder(resolve(prior,'handoff/data'));
  const expected=artifactManifest.files,byPath=new Map(expected.map(item=>[item.file,item]));
  requireThat(expected.length===311&&present.length===304&&new Set(expected.map(item=>item.file)).size===expected.length&&present.every(item=>canonical(item)===canonical(byPath.get(item.file))),'downloaded H27 handoff has changed or extra files');
  const missing=expected.filter(item=>!present.some(found=>found.file===item.file));
  requireThat(canonical(missing.map(item=>item.file).sort())===canonical([...H27_HIDDEN_HANDOFF_FILES].sort()),'downloaded H27 artifact does not omit exactly the known hidden metadata');
  requireThat((await filesUnder(resolve(stopped,'handoff/data'))).every((item,index)=>canonical(item)===canonical(expected[index])),'stopped Mac handoff differs from original completed manifest');
  for(const item of missing) {
    const source=safePath(resolve(stopped,'handoff/data'),item.file),target=safePath(resolve(stage,'handoff/data'),item.file);
    requireThat(canonical(await fileIdentity(source))===canonical({byteLength:item.byteLength,sha256:item.sha256}),'original hidden handoff metadata changed');
    await mkdir(dirname(target),{recursive:true});await cp(source,target,{errorOnExist:true,force:false});
  }
  requireThat(/^[a-f0-9]{40}$/.test(identity.stagingToolingSha)&&identity.stagingToolingSha!==identity.toolingSha,'supplement must identify its separate staging tooling');
  const record={version:VERSION,name:'completed-mlx-hidden-handoff-supplement',runId:identity.runId,attempt:identity.attempt,artifactId:identity.artifactId,toolingSha:identity.toolingSha,stagingToolingSha:identity.stagingToolingSha,completionSha256:local.completionSha256,retrySha256:local.retrySha256,handoffSha256:local.handoffSha256,artifactFiles:present,addedFiles:missing};
  await save(resolve(stage,'supplement.json'),record);
  return record;
}
export async function validateCompletedMlxHandoffSupplement(prior,identity) {
  const supplement=await json(resolve(prior,'supplement.json'));
  requireThat(supplement.version===VERSION&&supplement.name==='completed-mlx-hidden-handoff-supplement'&&supplement.runId===identity.runId&&supplement.attempt===identity.attempt&&supplement.artifactId===identity.artifactId&&supplement.toolingSha===identity.toolingSha&&supplement.stagingToolingSha===identity.cudaToolingSha&&supplement.completionSha256===identity.completionSha256&&supplement.retrySha256===identity.retrySha256&&supplement.handoffSha256===identity.handoffSha256,'merged hidden metadata has wrong original producer');
  const expected=(await json(resolve(prior,'handoff/handoff.json'))).files;
  const partition=[...supplement.artifactFiles,...supplement.addedFiles].sort((a,b)=>a.file.localeCompare(b.file));
  requireThat(expected.length===311&&supplement.artifactFiles.length===304&&supplement.addedFiles.length===7&&canonical(supplement.addedFiles.map(item=>item.file).sort())===canonical([...H27_HIDDEN_HANDOFF_FILES].sort())&&canonical(partition)===canonical([...expected].sort((a,b)=>a.file.localeCompare(b.file))),'merged H27 handoff partition changed');
  requireThat(canonical(await filesUnder(resolve(prior,'handoff/data')))===canonical(expected),'merged H27 handoff differs from completed manifest');
  return supplement;
}
// Review material only: actual decoded pixels, no generated comparison image or verdict.
export function rgbPng(width,height,pixels) {
  requireThat(Number.isSafeInteger(width)&&width>0&&Number.isSafeInteger(height)&&height>0&&width*height<=64*1024*1024&&pixels.length===width*height*3,'bounded RGB review raster required');
  const chunk=(type,data)=>{const body=Buffer.concat([Buffer.from(type),data]);let crc=0xffffffff;
    for(const byte of body){crc^=byte;for(let i=0;i<8;i++)crc=(crc>>>1)^((crc&1)?0xedb88320:0);}
    const length=Buffer.alloc(4),tail=Buffer.alloc(4);length.writeUInt32BE(data.length);tail.writeUInt32BE((crc^0xffffffff)>>>0);return Buffer.concat([length,body,tail]);};
  const header=Buffer.alloc(13);header.writeUInt32BE(width);header.writeUInt32BE(height,4);header[8]=8;header[9]=2;
  const rows=Buffer.alloc((width*3+1)*height);for(let y=0;y<height;y++)pixels.copy(rows,y*(width*3+1)+1,y*width*3,(y+1)*width*3);
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',header),chunk('IDAT',deflateSync(rows)),chunk('IEND',Buffer.alloc(0))]);
}
export async function retainQualityPairs(state,row) {
  requireThat(row.anchors&&row.output==='images','visual review pairs belong to the eight image anchors');
  const directory=resolve(state.root,'rows',row.id),receipt=await json(resolve(directory,'receipt.json')),inputs=await json(resolve(state.root,'input-identities.json'));
  const outputBytes=await readFile(safePath(directory,receipt.outputs[0].file));requireThat(sha(outputBytes)===receipt.outputs[0].sha256,'actual output changed before review material');
  const output=decodeRgbPng(outputBytes),inputPairs=[],detailCrops=[];
  for(const id of row.refs) {
    const sourceBytes=await readFile(safePath(state.root,inputs[id].file));requireThat(sha(sourceBytes)===inputs[id].sha256,'original input changed before review material');
    const input=decodeRgbPng(sourceBytes),width=Math.max(input.width,output.width),height=input.height+output.height,pixels=Buffer.alloc(width*height*3);
    for(const [image,top] of [[input,0],[output,input.height]])for(let y=0;y<image.height;y++)image.pixels.copy(pixels,((top+y)*width)*3,y*image.width*3,(y+1)*image.width*3);
    const file=`review-pair-${id}.png`,bytes=rgbPng(width,height,pixels);await writeFile(resolve(directory,file),bytes,{flag:'wx',mode:0o600});
    inputPairs.push({inputId:id,inputSha256:inputs[id].sha256,outputSha256:receipt.outputs[0].sha256,file,sha256:sha(bytes),layout:'unscaled original above actual output',inputGeometry:{width:input.width,height:input.height},outputGeometry:{width:output.width,height:output.height}});
  }
  const width=Math.max(1,Math.floor(output.width/3)),height=Math.max(1,Math.floor(output.height/3)),top=Math.floor((output.height-height)/2);
  for(let i=0;i<3;i++) {
    const left=Math.floor((output.width-width)*i/2),pixels=Buffer.alloc(width*height*3);
    for(let y=0;y<height;y++)output.pixels.copy(pixels,y*width*3,((top+y)*output.width+left)*3,((top+y)*output.width+left+width)*3);
    const file=`review-detail-${i}.png`,bytes=rgbPng(width,height,pixels);await writeFile(resolve(directory,file),bytes,{flag:'wx',mode:0o600});detailCrops.push({file,sha256:sha(bytes),outputSha256:receipt.outputs[0].sha256,rectangle:{left,top,width,height}});
  }
  await save(resolve(directory,'visual-pairs.json'),{version:VERSION,lockSha256:digest(state.lock),rowId:row.id,inputPairs,detailCrops});
}
