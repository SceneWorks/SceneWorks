import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {copyFile,mkdtemp,readFile,rm,writeFile} from 'node:fs/promises';
import {existsSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {VERSION,reviewVerdict,sha,digest} from './moss-edit-terminal.mjs';
import {compareQualityRow,validateExecutionToolingCapture,loadQualityAmendment,AMENDMENT_SHA256} from './moss-edit-quality-amendment.mjs';

const hash='a'.repeat(64),reviewHash='b'.repeat(64),upstreamHash='c'.repeat(64);
const row={id:'quality-mlx-base-single',variant:'base',refs:['B'],anchors:['Bottle color','Bottle and orange placement']};
const outputs=[{sha256:hash}];
const review={version:VERSION,rowId:row.id,reviewer:'actual reviewer',reviewedAt:'2026-10-25T00:00:00Z',outputSha256:[hash],anchors:[
  {name:row.anchors[0],verdict:'PASS',observation:'The bottle is red.'},
  {name:row.anchors[1],verdict:'FAIL',observation:'The native orange is absent.'},
]};
const comparison=()=>({upstream:{'base-single':{image:{sha256:upstreamHash}}},rows:[{
  rowId:row.id,upstreamCell:'base-single',nativeReviewSha256:reviewHash,nativeOutputSha256:[hash],upstreamImageSha256:upstreamHash,
  reviewer:'actual comparison reviewer',reviewedAt:'2026-10-25T01:00:00Z',
  instructionFollowing:{native:'Native bottle is red.',upstream:'Upstream bottle is red.'},
  referencePreservation:{native:'Native misses an orange.',upstream:'Upstream misses an orange.'},
  anchors:[
    {name:row.anchors[0],nativeVerdict:'PASS',nativeObservation:'Red bottle observed.',upstreamVerdict:'PASS',upstreamObservation:'Red bottle observed.',comparativeObservation:'Both recolor bottle.',materialNativeOnlyRegression:false,classification:'both-pass'},
    {name:row.anchors[1],nativeVerdict:'FAIL',nativeObservation:'Orange missing.',upstreamVerdict:'FAIL',upstreamObservation:'Orange missing.',comparativeObservation:'Both omit the orange in this fixed case.',materialNativeOnlyRegression:false,classification:'shared-upstream-limitation'},
  ],sharedLimitations:[row.anchors[1]],verdict:'PASS',
}]});

test('approved amendment is pinned and a truthful shared limitation can pass while strict historical FAIL remains',()=>{
  assert.equal(AMENDMENT_SHA256,'596ad6038ae5e1ab211e646255b9a1f67be1cdd3a79ea07a741fce506d414fe9');
  assert.equal(reviewVerdict(row,review,outputs),'FAIL');
  assert.equal(compareQualityRow(row,review,reviewHash,outputs,comparison()),'PASS');
});
test('compound FAIL/FAIL still blocks when the reviewer observes a material native-only regression',()=>{
  const value=comparison(),anchor=value.rows[0].anchors[1];
  anchor.comparativeObservation='Both fail placement, but upstream retains the orange while native removes it.';
  anchor.materialNativeOnlyRegression=true;anchor.classification='native-only-regression';
  value.rows[0].sharedLimitations=[];value.rows[0].verdict='FAIL';
  assert.equal(compareQualityRow(row,review,reviewHash,outputs,value),'FAIL');
  value.rows[0].verdict='PASS';assert.throws(()=>compareQualityRow(row,review,reviewHash,outputs,value),/native-only regression/);
});
test('upstream-pass/native-fail and changed historical review or output cannot be waived',()=>{
  const value=comparison(),anchor=value.rows[0].anchors[1];
  anchor.upstreamVerdict='PASS';anchor.classification='native-only-regression';value.rows[0].sharedLimitations=[];
  assert.throws(()=>compareQualityRow(row,review,reviewHash,outputs,value),/cannot be waived/);
  assert.throws(()=>compareQualityRow(row,review,'d'.repeat(64),outputs,comparison()),/bound to actual reviewed outputs/);
  const changed=comparison();changed.rows[0].nativeOutputSha256=['d'.repeat(64)];
  assert.throws(()=>compareQualityRow(row,review,reviewHash,outputs,changed),/bound to actual reviewed outputs/);
});
test('actual paired visual narratives and every original anchor remain mandatory',()=>{
  const missing=comparison();missing.rows[0].anchors[1].comparativeObservation='';
  assert.throws(()=>compareQualityRow(row,review,reviewHash,outputs,missing),/paired feature observations/);
  const omitted=comparison();omitted.rows[0].anchors.pop();
  assert.throws(()=>compareQualityRow(row,review,reviewHash,outputs,omitted),/every original visual anchor/);
});
test('captured execution closure must match committed Git objects, independently of verifier HEAD',()=>{
  const root=fileURLToPath(new URL('../../',import.meta.url)),head=execFileSync('git',['-C',root,'rev-parse','HEAD'],{encoding:'utf8'}).trim();
  const tree=execFileSync('git',['-C',root,'rev-parse',`${head}^{tree}`],{encoding:'utf8'}).trim();
  const original='/Users/michael/actions-runner-sceneworks/_work/SceneWorks/SceneWorks/tooling';
  const paths={harness:'scripts/moss-edit-terminal.mjs',harnessLibrary:'scripts/lib/moss-edit-terminal.mjs',actionController:'scripts/moss-edit-terminal-actions.mjs',actionLibrary:'scripts/lib/moss-edit-terminal-actions.mjs',browserDriver:'scripts/moss-edit-terminal-browser.mjs',browserControls:'scripts/lib/moss-edit-terminal-controls.mjs'};
  const files=Object.fromEntries(Object.entries(paths).map(([name,path])=>[name,`${original}/${path}`]));
  const identities=Object.fromEntries(Object.entries(paths).map(([name,path])=>{const bytes=execFileSync('git',['-C',root,'show',`${head}:${path}`]);return[name,{byteLength:bytes.length,sha256:sha(bytes)}];}));
  const capture={repair:{retryOrdinal:3,tooling:{head,source:{head,tree},sourceRoot:original,files,identities}}};
  assert.equal(validateExecutionToolingCapture(capture,head,root),true);
  const changed=structuredClone(capture);changed.repair.tooling.identities.harness.sha256='d'.repeat(64);
  assert.throws(()=>validateExecutionToolingCapture(changed,head,root),/Git object/);
  const wrongPath=structuredClone(capture);wrongPath.repair.tooling.files.harness=resolve('/tmp','moss-edit-terminal.mjs');
  assert.throws(()=>validateExecutionToolingCapture(wrongPath,head,root),/path changed/);
});

const originalResult='/Volumes/Models/sceneworks-terminal-sc-26752/moss-upstream-reference-40191507328-1';
test('complete four-cell overlay loads and actual suite summary rejects swapped same-shape images',{skip:!existsSync(resolve(originalResult,'upstream-base-single.png'))},async()=>{
  const sourceRoot=fileURLToPath(new URL('../../',import.meta.url)),h21='2cd0363d59cb3eb0989bbb708000b6ffc907932a';
  const verificationSha=execFileSync('git',['-C',sourceRoot,'rev-parse','HEAD'],{encoding:'utf8'}).trim();
  const root=await mkdtemp(resolve(tmpdir(),'moss-h22-overlay-'));
  const put=async(file,bytes)=>{const path=resolve(root,file);await writeFile(path,bytes);return {file,byteLength:Buffer.byteLength(bytes),sha256:sha(bytes)};};
  const record=async(file)=>{const bytes=await readFile(resolve(root,file));return {file,byteLength:bytes.length,sha256:sha(bytes)};};
  try {
    await copyFile('/Users/michael/.codex/epic-runs/sc-26752/QUALITY-ACCEPTANCE-AMENDMENT-2026-10-25.md',resolve(root,'quality-acceptance-amendment.md'));
    const paths={harness:'scripts/moss-edit-terminal.mjs',harnessLibrary:'scripts/lib/moss-edit-terminal.mjs',actionController:'scripts/moss-edit-terminal-actions.mjs',actionLibrary:'scripts/lib/moss-edit-terminal-actions.mjs',browserDriver:'scripts/moss-edit-terminal-browser.mjs',browserControls:'scripts/lib/moss-edit-terminal-controls.mjs'};
    const toolingRoot='/Users/michael/actions-runner-sceneworks/_work/SceneWorks/SceneWorks/tooling';
    const files=Object.fromEntries(Object.entries(paths).map(([name,path])=>[name,`${toolingRoot}/${path}`]));
    const identities=Object.fromEntries(Object.entries(paths).map(([name,path])=>{const bytes=execFileSync('git',['-C',sourceRoot,'show',`${h21}:${path}`]);return[name,{byteLength:bytes.length,sha256:sha(bytes)}];}));
    const capture={repair:{retryOrdinal:3,tooling:{head:h21,source:{head:h21,tree:execFileSync('git',['-C',sourceRoot,'rev-parse',`${h21}^{tree}`],{encoding:'utf8'}).trim()},sourceRoot:toolingRoot,files,identities}}};
    const executionToolingCapture=await put('execution-tooling-capture.json',JSON.stringify(capture));
    Object.assign(executionToolingCapture,{runId:'40214209083',artifactId:'12607298574',attempt:1});
    const finalPaths=Object.fromEntries(Object.entries(paths).map(([name,path])=>[name,`${toolingRoot}/${path}`]));
    const finalIdentities=Object.fromEntries(Object.entries(paths).map(([name,path])=>{const bytes=execFileSync('git',['-C',sourceRoot,'show',`${verificationSha}:${path}`]);return[name,{byteLength:bytes.length,sha256:sha(bytes)}];}));
    const finalCapture=await put('final-execution-tooling-capture.json',JSON.stringify({repair:{retryOrdinal:4,tooling:{head:verificationSha,source:{head:verificationSha,tree:execFileSync('git',['-C',sourceRoot,'rev-parse',`${verificationSha}^{tree}`],{encoding:'utf8'}).trim()},sourceRoot:toolingRoot,files:finalPaths,identities:finalIdentities}}}));
    Object.assign(finalCapture,{runId:'40200000003',artifactId:'12600000003',attempt:1});
    const frozen=JSON.parse(await readFile('/Users/michael/.codex/epic-runs/sc-26752/quality-inputs-v1/inputs.json'));
    const inputs=Object.fromEntries(frozen.assets.map(item=>[item.id,item]));
    const lock={sources:{sceneworks:{head:'42a8aff9852d01e0506df8afc466c0024b6eed8c'},inference:{head:'857f7a1996e3678a0773c3509ac0087522c19ce6'}}};
    const cells=['base-single','base-four','sft-single','sft-four'];
    const rows=['mlx','cuda'].flatMap(backend=>cells.map(cell=>({id:`quality-${backend}-${cell}`,variant:cell.split('-')[0],refs:cell.endsWith('four')?['B','L','P','G']:['B'],anchors:['actual anchor'],request:{prompt:frozen[cell.endsWith('four')?'four_case':'single_case'].prompt}})));
    const upstream={},summary={toolingSha:verificationSha,qualityCasesManifestSha256:'af8ff343a598772bbe114164651b80225fc6f62b66ac4da26e5cb2071570ca57',outputs:[]};
    for(const cell of cells){
      const base=cell==='base-single',refs=cell.endsWith('four')?['B','L','P','G']:['B'];
      let image,diagnostic,authentication;
      if(base){
        for(const name of ['upstream-base-single.png','upstream-base-single.diagnostic.json','authentication.json'])await copyFile(resolve(originalResult,name),resolve(root,base&&name==='authentication.json'?'base-single-authentication.json':name));
        image=await record('upstream-base-single.png');diagnostic=await record('upstream-base-single.diagnostic.json');authentication=await record('base-single-authentication.json');
      } else {
        image=await put(`upstream-${cell}.png`,Buffer.from(`synthetic ${cell} image fixture`));
        const result={caseId:cell,variant:cell.split('-')[0],result:{width:cell.endsWith('four')?512:768,height:cell.endsWith('four')?768:512,seed:173,actual_steps:30,cfg:4,inference_shift:5,kv_cache:true,dit_quantization:'none',text_quantization:'none',prompt:frozen[cell.endsWith('four')?'four_case':'single_case'].prompt,negative_prompt:'',references:refs.map(id=>`/fixture/${id}.png`)},noiseSha256:'92e4addfaab4f8f5738c6a29ab2da1a56e2b204726b83250721edfec09608d82',noiseShape:cell.endsWith('four')?[1,128,48,32]:[1,128,32,48],noiseAllocations:1,wrapperSha256:hash,frozenFilesVerifiedAfterRun:24};
        diagnostic=await put(`upstream-${cell}.diagnostic.json`,JSON.stringify(result));
        authentication=await put('suite-authentication.json',JSON.stringify({qualityCasesManifestSha256:summary.qualityCasesManifestSha256,priorPayloadManifestSha256:'269f9dc9e9d7852d131457aaa6a1fe9c558a9dffee71b9fd57a2026f5663b9df',toolingSha:verificationSha,nativeProductSha:lock.sources.sceneworks.head,nativeInferenceSha:lock.sources.inference.head,wrapperSha256:hash}));
        summary.outputs.push({caseId:cell,imageSha256:image.sha256,diagnosticSha256:diagnostic.sha256});
      }
      upstream[cell]={cell,runId:base?'40191507328':'40200000001',artifactId:base?'12598132624':'12600000002',attempt:1,toolingSha:base?'d6af3f2623fff31cf000cfd5d79f23069b385d39':verificationSha,reviewer:'fixture reviewer',reviewedAt:'2026-10-25T00:00:00Z',image,diagnostic,authentication,noiseSha256:'92e4addfaab4f8f5738c6a29ab2da1a56e2b204726b83250721edfec09608d82',inputs:refs.map(id=>({id,sha256:inputs[id].sha256})),observation:`Observed ${cell} fixture`};
    }
    const suiteSummary=await put('upstream-suite-summary.json',JSON.stringify(summary));Object.assign(suiteSummary,{runId:'40200000001',artifactId:'12600000002',attempt:1});
    const comparison={version:'sc-26757-upstream-comparison/v1',amendmentSha256:AMENDMENT_SHA256,lockSha256:digest(lock),executionToolingCapture,finalExecutionToolingCapture:finalCapture,upstream,rows:rows.map(row=>({rowId:row.id})),suiteSummary};
    const path=resolve(root,'quality-upstream-comparison.json');await writeFile(path,JSON.stringify(comparison));
    assert.deepEqual(await loadQualityAmendment(root,lock,rows,inputs,{historicalExecutionToolingSha:h21,executionToolingSha:verificationSha,verificationToolingSha:verificationSha,verificationSourceRoot:sourceRoot}),comparison);
    [upstream['base-four'].image,upstream['sft-four'].image]=[upstream['sft-four'].image,upstream['base-four'].image];
    await writeFile(path,JSON.stringify(comparison));
    await assert.rejects(()=>loadQualityAmendment(root,lock,rows,inputs,{historicalExecutionToolingSha:h21,executionToolingSha:verificationSha,verificationToolingSha:verificationSha,verificationSourceRoot:sourceRoot}),/actual three-case suite output/);
  } finally {await rm(root,{recursive:true,force:true});}
});
