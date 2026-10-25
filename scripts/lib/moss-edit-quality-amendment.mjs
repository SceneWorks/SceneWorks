// The approved quality amendment changes only visual acceptance. Native execution
// evidence and the original strict-anchor reviews remain immutable.
import {readFile} from 'node:fs/promises';
import {resolve,posix} from 'node:path';
import {execFileSync} from 'node:child_process';
import {canonical,digest,fileIdentity,json,requireThat,safePath,sha,reviewVerdict} from './moss-edit-terminal.mjs';
import {CASES_MANIFEST_SHA256} from '../ci/moss-upstream-quality.mjs';

export const AMENDMENT_SHA256='596ad6038ae5e1ab211e646255b9a1f67be1cdd3a79ea07a741fce506d414fe9';
export const FINAL_EXECUTION_TOOLING_SHA='7dfbb2205034529ff3abfe16772b2726e153c069';
const SHA=/^[a-f0-9]{64}$/;
const CELLS=['base-single','base-four','sft-single','sft-four'];
const NOISE_SHA='92e4addfaab4f8f5738c6a29ab2da1a56e2b204726b83250721edfec09608d82';
const classifications={
  'PASS/PASS':'both-pass',
  'PASS/FAIL':'native-exceeds-upstream',
  'FAIL/FAIL':'shared-upstream-limitation',
  'FAIL/PASS':'native-only-regression',
};
const identity=async(root,record)=>{
  requireThat(record&&typeof record.file==='string'&&SHA.test(record.sha256??'')&&Number.isSafeInteger(record.byteLength)&&record.byteLength>0,'upstream evidence needs file identity');
  const actual=await fileIdentity(safePath(root,record.file));
  requireThat(canonical(actual)===canonical({sha256:record.sha256,byteLength:record.byteLength}),'upstream evidence bytes changed');
  return safePath(root,record.file);
};
export function validateExecutionToolingCapture(capture,executionToolingSha,verificationSourceRoot,ordinal=3){
  const tooling=capture?.repair?.tooling;
  requireThat(capture?.repair?.retryOrdinal===ordinal&&tooling?.head===executionToolingSha&&tooling.source?.head===executionToolingSha&&tooling.sourceRoot&&tooling.files&&tooling.identities,'captured execution tooling differs from retained retry producer');
  const git=(...args)=>execFileSync('git',['-C',verificationSourceRoot,...args]);
  requireThat(tooling.source.tree===git('rev-parse',`${executionToolingSha}^{tree}`).toString().trim(),'H21 captured source tree changed');
  const expected={harness:'scripts/moss-edit-terminal.mjs',harnessLibrary:'scripts/lib/moss-edit-terminal.mjs',actionController:'scripts/moss-edit-terminal-actions.mjs',actionLibrary:'scripts/lib/moss-edit-terminal-actions.mjs',browserDriver:'scripts/moss-edit-terminal-browser.mjs',browserControls:'scripts/lib/moss-edit-terminal-controls.mjs'};
  requireThat(canonical(Object.keys(tooling.files).sort())===canonical(Object.keys(expected).sort()),'H21 captured tooling file closure changed');
  for(const [name,path] of Object.entries(expected)){
    requireThat(posix.relative(tooling.sourceRoot,tooling.files[name])===path,'H21 captured tooling path changed');
    const bytes=git('show',`${executionToolingSha}:${path}`);
    requireThat(canonical(tooling.identities[name])===canonical({byteLength:bytes.length,sha256:sha(bytes)}),'H21 captured tooling bytes differ from its Git object');
  }
  return true;
}

export async function loadQualityAmendment(root,lock,rows,inputs,{historicalExecutionToolingSha,executionToolingSha,verificationToolingSha,verificationSourceRoot}={}){
  const file=resolve(root,'quality-upstream-comparison.json');
  const comparison=await json(file).catch(error=>{if(error.code==='ENOENT')return null;throw error;});
  if(!comparison)return null;
  requireThat(sha(await readFile(resolve(root,'quality-acceptance-amendment.md')))===AMENDMENT_SHA256,'approved quality amendment bytes changed');
  requireThat(comparison.version==='sc-26757-upstream-comparison/v1'&&comparison.amendmentSha256===AMENDMENT_SHA256&&comparison.lockSha256===digest(lock),'quality comparison not bound to approved policy and frozen lock');
  const captureEntry=comparison.executionToolingCapture;
  requireThat(captureEntry?.file==='execution-tooling-capture.json'&&String(captureEntry.runId)==='40214209083'&&String(captureEntry.artifactId)==='12607298574'&&captureEntry.attempt===1&&historicalExecutionToolingSha==='2cd0363d59cb3eb0989bbb708000b6ffc907932a'&&verificationSourceRoot,'amended review requires actual H21 execution-tooling capture');
  const capture=await json(await identity(root,captureEntry));
  validateExecutionToolingCapture(capture,historicalExecutionToolingSha,verificationSourceRoot);
  const finalEntry=comparison.finalExecutionToolingCapture;
  requireThat(finalEntry?.file==='final-execution-tooling-capture.json'&&/^[1-9][0-9]*$/.test(String(finalEntry.runId))&&/^[1-9][0-9]*$/.test(String(finalEntry.artifactId))&&String(finalEntry.runId)!==String(captureEntry.runId)&&String(finalEntry.artifactId)!==String(captureEntry.artifactId)&&finalEntry.attempt===1&&executionToolingSha===FINAL_EXECUTION_TOOLING_SHA&&/^[a-f0-9]{40}$/.test(verificationToolingSha??'')&&verificationToolingSha!==historicalExecutionToolingSha,'amended review requires distinct actual final execution-tooling capture');
  execFileSync('git',['-C',verificationSourceRoot,'merge-base','--is-ancestor',executionToolingSha,verificationToolingSha]);
  const finalCapture=await json(await identity(root,finalEntry));
  validateExecutionToolingCapture(finalCapture,executionToolingSha,verificationSourceRoot,4);
  requireThat(canonical(Object.keys(comparison.upstream??{}).sort())===canonical([...CELLS].sort())&&Array.isArray(comparison.rows)&&comparison.rows.length===8,'all four upstream cells and eight native comparisons required');
  const qualityRows=rows.filter(row=>row.anchors);
  requireThat(qualityRows.length===8&&canonical(comparison.rows.map(r=>r.rowId).sort())===canonical(qualityRows.map(r=>r.id).sort()),'quality comparison omitted or duplicated native cell');
  let suite=null;
  for(const cell of CELLS){
    const entry=comparison.upstream[cell];
    requireThat(entry?.cell===cell&&/^[1-9][0-9]*$/.test(String(entry.runId))&&/^[1-9][0-9]*$/.test(String(entry.artifactId))&&Number.isInteger(entry.attempt)&&entry.attempt>0&&/^[a-f0-9]{40}$/.test(entry.toolingSha??'')&&entry.reviewer&&entry.reviewedAt&&Number.isFinite(Date.parse(entry.reviewedAt)),'upstream cell missing actual run/artifact and visual reviewer identity');
    if(cell==='base-single')requireThat(String(entry.runId)==='40191507328'&&entry.attempt===1&&String(entry.artifactId)==='12598132624'&&entry.toolingSha==='d6af3f2623fff31cf000cfd5d79f23069b385d39'&&entry.image?.sha256==='6b089d1d74affa136878896ac987556e20aa962657aa30a1a8ebaae95a5a30ad'&&entry.diagnostic?.sha256==='db86973abdbb53b4d48a98f93aded598b168583dc61bb37b3264e476160d0964'&&entry.authentication?.sha256==='a0e0c279a6e661e1703c5c4f7423d9f7ff940d5bfdcee850f152e9a8850e08f3','original Base/single result substituted');
    else {
      const current={runId:String(entry.runId),artifactId:String(entry.artifactId),attempt:entry.attempt,toolingSha:entry.toolingSha};
      if(suite)requireThat(canonical(current)===canonical(suite),'three upstream cases must come from one authenticated MPS suite');
      else suite=current;
    }
    const [image,diagnostic,authentication]=await Promise.all(['image','diagnostic','authentication'].map(key=>identity(root,entry[key])));
    const result=await json(diagnostic),auth=await json(authentication);
    requireThat(result?.result?.width===(cell.endsWith('four')?512:768)&&result.result.height===(cell.endsWith('four')?768:512)&&result.result.seed===173&&result.result.actual_steps===30&&result.result.cfg===4&&result.result.inference_shift===5&&result.result.kv_cache===true&&result.result.dit_quantization==='none'&&result.result.text_quantization==='none'&&result.noiseAllocations===1,'upstream diagnostic settings differ from frozen quality case');
    requireThat(result.noiseSha256===NOISE_SHA&&entry.noiseSha256===NOISE_SHA&&auth&&typeof auth==='object','upstream noise/authentication evidence missing');
    if(cell==='base-single')requireThat(auth.payloadManifestSha256==='269f9dc9e9d7852d131457aaa6a1fe9c558a9dffee71b9fd57a2026f5663b9df'&&result.wrapperSha256==='ffe0e18dbf021ca5f8f62cf7cacb54f4b0c2a40c4d6142c862bd4df9d42f248d'&&result.frozenFilesVerifiedAfterRun===24,'original Base/single frozen learned source changed');
    else requireThat(auth.qualityCasesManifestSha256===CASES_MANIFEST_SHA256&&auth.priorPayloadManifestSha256==='269f9dc9e9d7852d131457aaa6a1fe9c558a9dffee71b9fd57a2026f5663b9df'&&auth.toolingSha===entry.toolingSha&&entry.toolingSha===FINAL_EXECUTION_TOOLING_SHA&&auth.nativeProductSha===lock.sources.sceneworks.head&&auth.nativeInferenceSha===lock.sources.inference.head&&auth.wrapperSha256===result.wrapperSha256&&result.caseId===cell&&result.variant===cell.split('-')[0]&&result.frozenFilesVerifiedAfterRun===24&&canonical(result.noiseShape)===canonical(cell.endsWith('four')?[1,128,48,32]:[1,128,32,48]),'three-case upstream producer/source/shape identity changed');
    const expectedRefs=cell.endsWith('four')?['B','L','P','G']:['B'];
    requireThat(canonical(entry.inputs?.map(item=>item.id))===canonical(expectedRefs)&&entry.inputs.every(item=>item.sha256===inputs[item.id]?.sha256),'upstream reference inputs differ from frozen ordered images');
    requireThat(result.result.references?.length===expectedRefs.length&&result.result.references.every((path,index)=>path.endsWith(`/${expectedRefs[index]}.png`))&&result.result.prompt===qualityRows.find(row=>row.variant===cell.split('-')[0]&&row.refs.length===expectedRefs.length).request.prompt&&result.result.negative_prompt==='','upstream prompt/reference order differs from frozen quality case');
    requireThat(image&&entry.observation&&entry.observation.trim(),'actual upstream image requires case-specific observed outcome');
  }
  const summaryEntry=comparison.suiteSummary;
  requireThat(suite&&summaryEntry?.file==='upstream-suite-summary.json'&&String(summaryEntry.runId)===suite.runId&&String(summaryEntry.artifactId)===suite.artifactId&&summaryEntry.attempt===suite.attempt,'three-case suite summary missing actual artifact identity');
  const summary=await json(await identity(root,summaryEntry));
  requireThat(summary.toolingSha===suite.toolingSha&&summary.qualityCasesManifestSha256===CASES_MANIFEST_SHA256&&Array.isArray(summary.outputs)&&summary.outputs.length===3,'three-case suite result provenance changed');
  validateSuiteOutputs(summary,comparison.upstream);
  return comparison;
}

export function validateSuiteOutputs(summary,upstream){
  requireThat(Array.isArray(summary?.outputs)&&summary.outputs.length===3,'three-case suite output count changed');
  for(const cell of CELLS.filter(name=>name!=='base-single')){
    const output=summary.outputs.find(item=>item.caseId===cell),entry=upstream?.[cell];
    requireThat(summary.outputs.filter(item=>item.caseId===cell).length===1&&output.imageSha256===entry?.image?.sha256&&output.diagnosticSha256===entry?.diagnostic?.sha256,'upstream image/diagnostic differs from actual three-case suite output');
  }
  return true;
}

export function compareQualityRow(row,nativeReview,nativeReviewSha256,outputs,comparison){
  // This checks the historical review's identities/anchor names even if its
  // verdict was FAIL under the original stricter-than-upstream rule.
  reviewVerdict(row,nativeReview,outputs);
  const record=comparison.rows.find(item=>item.rowId===row.id),cell=`${row.variant}-${row.refs.length===4?'four':'single'}`,upstream=comparison.upstream[cell];
  requireThat(record&&record.upstreamCell===cell&&record.nativeReviewSha256===nativeReviewSha256&&canonical(record.nativeOutputSha256)===canonical(outputs.map(item=>item.sha256))&&record.upstreamImageSha256===upstream.image.sha256&&record.reviewer&&record.reviewedAt&&Number.isFinite(Date.parse(record.reviewedAt)),'native/upstream comparison not bound to actual reviewed outputs');
  requireThat(record.instructionFollowing?.native?.trim()&&record.instructionFollowing?.upstream?.trim()&&record.referencePreservation?.native?.trim()&&record.referencePreservation?.upstream?.trim(),'instruction and reference outcomes require two observed narratives');
  requireThat(Array.isArray(record.anchors)&&canonical(record.anchors.map(a=>a.name))===canonical(row.anchors),'comparison must cover every original visual anchor');
  const shared=[];let regression=false;
  for(let i=0;i<row.anchors.length;i++){
    const anchor=record.anchors[i],historical=nativeReview.anchors[i];
    requireThat(anchor.nativeVerdict===historical.verdict&&['PASS','FAIL'].includes(anchor.upstreamVerdict)&&anchor.nativeObservation?.trim()&&anchor.upstreamObservation?.trim()&&anchor.comparativeObservation?.trim()&&typeof anchor.materialNativeOnlyRegression==='boolean','paired feature observations or explicit regression judgment missing');
    requireThat(anchor.materialNativeOnlyRegression||!(anchor.nativeVerdict==='FAIL'&&anchor.upstreamVerdict==='PASS'),'native-only failed anchor cannot be waived');
    const classification=anchor.materialNativeOnlyRegression?'native-only-regression':classifications[`${anchor.nativeVerdict}/${anchor.upstreamVerdict}`];
    requireThat(anchor.classification===classification,'native/upstream difference was misclassified');
    if(classification==='shared-upstream-limitation')shared.push(anchor.name);
    if(classification==='native-only-regression')regression=true;
  }
  requireThat(canonical(record.sharedLimitations??[])===canonical(shared)&&record.sharedLimitations.every(name=>record.anchors.find(anchor=>anchor.name===name)?.upstreamObservation?.trim()),'shared upstream limitations not documented from corresponding cells');
  requireThat(record.verdict===(regression?'FAIL':'PASS'),'native-only regression cannot pass amended quality acceptance');
  return record.verdict;
}
