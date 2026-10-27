import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {copyFile,cp,mkdir,mkdtemp,readFile,rm,writeFile} from 'node:fs/promises';
import {existsSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {VERSION,reviewVerdict,sha,digest,fileIdentity} from './moss-edit-terminal.mjs';
import {loadCandidateQualityAmendment} from '../moss-edit-terminal.mjs';
import {compareQualityRow,ordinaryExecutionCapture,validateOrdinaryExecutionCaptures,validateExecutionToolingCapture,validateSuiteOutputs,loadQualityAmendment,loadReferenceCorrection,originalReferenceContextSha256,AMENDMENT_SHA256,UPSTREAM_SUITE_TOOLING_SHA} from './moss-edit-quality-amendment.mjs';

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
test('three-case suite rejects swapped same-shape images without local model evidence',()=>{
  const upstream=Object.fromEntries(['base-four','sft-single','sft-four'].map((cell,index)=>[cell,{image:{sha256:String(index+1).repeat(64)},diagnostic:{sha256:String(index+4).repeat(64)}}]));
  const summary={outputs:Object.entries(upstream).map(([caseId,entry])=>({caseId,imageSha256:entry.image.sha256,diagnosticSha256:entry.diagnostic.sha256}))};
  assert.equal(validateSuiteOutputs(summary,upstream),true);
  const swapped=structuredClone(upstream);
  [swapped['base-four'].image,swapped['sft-four'].image]=[swapped['sft-four'].image,swapped['base-four'].image];
  assert.throws(()=>validateSuiteOutputs(summary,swapped),/actual three-case suite output/);
  assert.throws(()=>validateSuiteOutputs({...summary,outputs:summary.outputs.slice(0,2)},upstream),/suite output count/);
});

const originalResult='/Volumes/Models/sceneworks-terminal-sc-26752/moss-upstream-reference-40191507328-1';
test('ordinary Actions captures bind both backends to the same real artifact run and selected policy never falls back',{timeout:10000},async()=>{
  const root=await mkdtemp(resolve(tmpdir(),'moss-ordinary-captures-'));
  const lock={version:VERSION,candidateId:'ordinary-fixture',sources:{sceneworks:{head:'e'.repeat(40)},inference:{head:'f'.repeat(40)}},files:{},artifacts:{mlx:{files:{}},cuda:{files:{}}},qualityPolicy:{amendmentSha256:AMENDMENT_SHA256}};
  const env={GITHUB_ACTIONS:'true',GITHUB_REPOSITORY:'SceneWorks/SceneWorks',GITHUB_SHA:lock.sources.sceneworks.head,GITHUB_RUN_ID:'40200000011',GITHUB_RUN_ATTEMPT:'1'},value={nativeExecutions:{}};
  const expected={executionRunId:env.GITHUB_RUN_ID,executionRunAttempt:env.GITHUB_RUN_ATTEMPT};
  try{
    await mkdir(resolve(root,'bindings'));await mkdir(resolve(root,'external'));
    for(const backend of ['mlx','cuda']){
      const binding={version:VERSION,lockSha256:digest(lock),backend,sources:lock.sources,files:{},endpoint:{apiUrl:'http://127.0.0.1:12345',projectId:'project',workerId:`worker-${backend}`,tokenEnv:'TOKEN',workerGpuId:backend==='cuda'?'1':'mlx'}};
      await writeFile(resolve(root,'bindings',`${backend}.json`),JSON.stringify(binding));
      const capture=ordinaryExecutionCapture(lock,binding,env),file=`external/ordinary-execution-${backend}.json`;
      await writeFile(resolve(root,file),JSON.stringify(capture));value.nativeExecutions[backend]={file,...await fileIdentity(resolve(root,file))};
      assert.throws(()=>ordinaryExecutionCapture(lock,binding,{...env,GITHUB_ACTIONS:'false'}),/actual paired Actions/);
    }
    await validateOrdinaryExecutionCaptures(root,lock,value,expected);
    await assert.rejects(()=>loadQualityAmendment(root,lock,[],{}),/selected approved quality comparison missing/);
    const missing=structuredClone(value);delete missing.nativeExecutions.mlx;await assert.rejects(()=>validateOrdinaryExecutionCaptures(root,lock,missing,expected),/both actual native captures/);
    const file=value.nativeExecutions.cuda.file,bytes=await readFile(resolve(root,file));
    for(const mutate of [c=>{c.runId='40200000012';},c=>{c.attempt='2';},c=>{c.bindingSha256=hash;},c=>{c.lockSha256=hash;},c=>{c.sources.inference.head='d'.repeat(40);},c=>{c.artifactName='unrelated-artifact';}]){
      const changed=JSON.parse(bytes);mutate(changed);await writeFile(resolve(root,file),JSON.stringify(changed));value.nativeExecutions.cuda={file,...await fileIdentity(resolve(root,file))};
      await assert.rejects(()=>validateOrdinaryExecutionCaptures(root,lock,value,expected),/actual execution artifact, lock or binding/);
    }
  }finally{await rm(root,{recursive:true,force:true});}
});
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
    const finalCapture=await put('final-execution-tooling-capture.json',JSON.stringify({repair:{retryOrdinal:5,tooling:{head:verificationSha,source:{head:verificationSha,tree:execFileSync('git',['-C',sourceRoot,'rev-parse',`${verificationSha}^{tree}`],{encoding:'utf8'}).trim()},sourceRoot:toolingRoot,files:finalPaths,identities:finalIdentities}}}));
    Object.assign(finalCapture,{runId:'40200000003',artifactId:'12600000003',attempt:1});
    const frozen=JSON.parse(await readFile('/Users/michael/.codex/epic-runs/sc-26752/quality-inputs-v1/inputs.json'));
    const inputs=Object.fromEntries(frozen.assets.map(item=>[item.id,item]));
    const lock={sources:{sceneworks:{head:'42a8aff9852d01e0506df8afc466c0024b6eed8c'},inference:{head:'857f7a1996e3678a0773c3509ac0087522c19ce6'}}};
    const cells=['base-single','base-four','sft-single','sft-four'];
    const rows=['mlx','cuda'].flatMap(backend=>cells.map(cell=>({id:`quality-${backend}-${cell}`,variant:cell.split('-')[0],refs:cell.endsWith('four')?['B','L','P','G']:['B'],anchors:['actual anchor'],request:{prompt:frozen[cell.endsWith('four')?'four_case':'single_case'].prompt}})));
    const upstream={},summary={toolingSha:UPSTREAM_SUITE_TOOLING_SHA,qualityCasesManifestSha256:'af8ff343a598772bbe114164651b80225fc6f62b66ac4da26e5cb2071570ca57',outputs:[]};
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
        authentication=await put('suite-authentication.json',JSON.stringify({qualityCasesManifestSha256:summary.qualityCasesManifestSha256,priorPayloadManifestSha256:'269f9dc9e9d7852d131457aaa6a1fe9c558a9dffee71b9fd57a2026f5663b9df',toolingSha:UPSTREAM_SUITE_TOOLING_SHA,nativeProductSha:lock.sources.sceneworks.head,nativeInferenceSha:lock.sources.inference.head,wrapperSha256:hash}));
        summary.outputs.push({caseId:cell,imageSha256:image.sha256,diagnosticSha256:diagnostic.sha256});
      }
      upstream[cell]={cell,runId:base?'40191507328':'40200000001',artifactId:base?'12598132624':'12600000002',attempt:1,toolingSha:base?'d6af3f2623fff31cf000cfd5d79f23069b385d39':UPSTREAM_SUITE_TOOLING_SHA,reviewer:'fixture reviewer',reviewedAt:'2026-10-25T00:00:00Z',image,diagnostic,authentication,noiseSha256:'92e4addfaab4f8f5738c6a29ab2da1a56e2b204726b83250721edfec09608d82',inputs:refs.map(id=>({id,sha256:inputs[id].sha256})),observation:`Observed ${cell} fixture`};
    }
    const suiteSummary=await put('upstream-suite-summary.json',JSON.stringify(summary));Object.assign(suiteSummary,{runId:'40200000001',artifactId:'12600000002',attempt:1});
    const comparison={version:'sc-26757-upstream-comparison/v1',amendmentSha256:AMENDMENT_SHA256,lockSha256:digest(lock),executionToolingCapture,finalExecutionToolingCapture:finalCapture,upstream,rows:rows.map(row=>({rowId:row.id})),suiteSummary};
    const path=resolve(root,'quality-upstream-comparison.json');await writeFile(path,JSON.stringify(comparison));
    assert.deepEqual(await loadQualityAmendment(root,lock,rows,inputs,{historicalExecutionToolingSha:h21,executionToolingSha:verificationSha,finalExecutionOrdinal:5,verificationToolingSha:verificationSha,verificationSourceRoot:sourceRoot}),comparison);
    await assert.rejects(()=>loadQualityAmendment(root,lock,rows,inputs,{historicalExecutionToolingSha:h21,executionToolingSha:UPSTREAM_SUITE_TOOLING_SHA,finalExecutionOrdinal:5,verificationToolingSha:verificationSha,verificationSourceRoot:sourceRoot}),/distinct actual final execution-tooling capture/);
    await assert.rejects(()=>loadQualityAmendment(root,lock,rows,inputs,{historicalExecutionToolingSha:h21,executionToolingSha:verificationSha,finalExecutionOrdinal:4,verificationToolingSha:verificationSha,verificationSourceRoot:sourceRoot}),/distinct actual final execution-tooling capture/);
    // Assemble the ordinary finalizer's actual no-repair launch shape. Upstream
    // keeps its old producer while new native captures bind the new paired lock.
    const bundleRoot=resolve(root,'bundle');await mkdir(bundleRoot);
    const frozenFiles=[],frozenPut=async(file,bytes)=>{await writeFile(resolve(bundleRoot,file),bytes);frozenFiles.push({file,byteLength:Buffer.byteLength(bytes),sha256:sha(bytes)});};
    const references=[];
    for(const [cell,entry] of Object.entries(upstream))for(const role of ['image','diagnostic','authentication']){
      const item=entry[role];if(!frozenFiles.some(record=>record.file===item.file))await frozenPut(item.file,await readFile(resolve(root,item.file)));
      references.push({cell,role,file:item.file,sha256:item.sha256,runId:entry.runId,artifactId:entry.artifactId,toolingSha:entry.toolingSha});
    }
    await mkdir(resolve(bundleRoot,'historical'));
    await frozenPut('historical/original-quality-upstream-comparison.json',JSON.stringify(comparison));
    await frozenPut('historical/upstream-four-cell-map.json',JSON.stringify({references}));
    await writeFile(resolve(bundleRoot,'bundle-inventory.json'),JSON.stringify({files:frozenFiles}));
    const ordinaryLock={version:VERSION,candidateId:'ordinary-fixture',sources:{sceneworks:{head:'e'.repeat(40)},inference:{head:'f'.repeat(40)}},files:{qualityBundleInventory:await fileIdentity(resolve(bundleRoot,'bundle-inventory.json'))},artifacts:{mlx:{files:{}},cuda:{files:{}}},qualityPolicy:{amendmentSha256:AMENDMENT_SHA256}};
    await mkdir(resolve(root,'bindings'));await mkdir(resolve(root,'external'));
    const env={GITHUB_ACTIONS:'true',GITHUB_REPOSITORY:'SceneWorks/SceneWorks',GITHUB_SHA:ordinaryLock.sources.sceneworks.head,GITHUB_RUN_ID:'40200000011',GITHUB_RUN_ATTEMPT:'1'};
    const nativeExecutions={};
    for(const backend of ['mlx','cuda']){
      const binding={version:VERSION,lockSha256:digest(ordinaryLock),backend,sources:ordinaryLock.sources,files:ordinaryLock.files,endpoint:{apiUrl:'http://127.0.0.1:12345',projectId:'actual-project',workerId:`worker-${backend}`,tokenEnv:'TOKEN',workerGpuId:backend==='cuda'?'1':'mlx'}};
      await put(`bindings/${backend}.json`,JSON.stringify(binding));
      nativeExecutions[backend]=await put(`external/ordinary-execution-${backend}.json`,JSON.stringify(ordinaryExecutionCapture(ordinaryLock,binding,env)));
      assert.throws(()=>ordinaryExecutionCapture(ordinaryLock,binding,{...env,GITHUB_SHA:lock.sources.sceneworks.head}),/actual paired Actions/);
    }
    await put('input-identities.json',JSON.stringify(inputs));
    const ordinary={...comparison,lockSha256:digest(ordinaryLock),nativeExecutions};
    delete ordinary.executionToolingCapture;delete ordinary.finalExecutionToolingCapture;
    const state={root,lock:ordinaryLock,plan:{rows},config:{qualityBundleRoot:bundleRoot,qualityExecution:{executionRunId:env.GITHUB_RUN_ID,executionRunAttempt:'1'}}};
    const loadOrdinary=()=>loadCandidateQualityAmendment(state);
    await writeFile(path,JSON.stringify(ordinary));assert.deepEqual(await loadOrdinary(),ordinary);
    await rm(path);await assert.rejects(loadOrdinary,/selected approved quality comparison missing/);
    const mutations=[
      [value=>{value.lockSha256=hash;},/both frozen locks/],
      [value=>{delete value.nativeExecutions.mlx;},/both actual native captures/],
      [value=>{value.nativeExecutions.cuda.sha256=hash;},/bytes changed/],
      [value=>{value.upstream['base-four'].runId='40200000099';},/producer substituted/],
      [value=>{value.upstream['base-four'].authentication.sha256=hash;},/producer substituted/],
      [value=>{value.rows.pop();},/eight native comparisons/],
    ];
    for(const [mutate,pattern] of mutations){const changed=structuredClone(ordinary);mutate(changed);await writeFile(path,JSON.stringify(changed));await assert.rejects(loadOrdinary,pattern);}
    await writeFile(path,JSON.stringify(ordinary));state.config.qualityExecution.executionRunId='40200000012';await assert.rejects(loadOrdinary,/actual execution artifact/);state.config.qualityExecution.executionRunId=env.GITHUB_RUN_ID;
    const diagnosticFile=upstream['base-four'].diagnostic.file,diagnosticBytes=await readFile(resolve(root,diagnosticFile)),changedDiagnostic=JSON.parse(diagnosticBytes);changedDiagnostic.result.cfg=5;
    await writeFile(resolve(root,diagnosticFile),JSON.stringify(changedDiagnostic));await assert.rejects(loadOrdinary,/bytes changed/);await writeFile(resolve(root,diagnosticFile),diagnosticBytes);
    // Even after loading the ordinary policy, fresh native output/review hashes
    // and material native-only differences retain the same strict row checks.
    const actual=comparisonForOrdinaryRow(rows[0]);
    assert.equal(compareQualityRow(actual.row,actual.review,reviewHash,outputs,actual.comparison),'PASS');
    assert.throws(()=>compareQualityRow(actual.row,actual.review,reviewHash,[{sha256:upstreamHash}],actual.comparison),/review.*output|reviewed outputs/);
    await writeFile(path,JSON.stringify(comparison));
    [upstream['base-four'].image,upstream['sft-four'].image]=[upstream['sft-four'].image,upstream['base-four'].image];
    await writeFile(path,JSON.stringify(comparison));
    await assert.rejects(()=>loadQualityAmendment(root,lock,rows,inputs,{historicalExecutionToolingSha:h21,executionToolingSha:verificationSha,finalExecutionOrdinal:5,verificationToolingSha:verificationSha,verificationSourceRoot:sourceRoot}),/actual three-case suite output/);
  } finally {await rm(root,{recursive:true,force:true});}
});

function comparisonForOrdinaryRow(){
  return {row,review,comparison:comparison()};
}

// CI_IMPACT: CPU evidence hashing/JSON fixtures only; retained actual archives are optional local fixtures.
const correctionStage='/Users/michael/.codex/epic-runs/sc-26752/resume-20261026/quality-reference-correction-staging';
test('causal reference correction keeps policy and all original primary reference context mandatory',async()=>{
  const original={upstream:{cell:{image:{sha256:hash}}},suiteSummary:{sha256:reviewHash}},changed=structuredClone(original);changed.upstream.cell.image.sha256=upstreamHash;
  assert.notEqual(originalReferenceContextSha256(original),originalReferenceContextSha256(changed));
  await assert.rejects(()=>loadReferenceCorrection('/unused',{referenceCorrection:{file:'arbitrary.json'}},{},[],{}),/sidecar path/);
});
test('actual three-call proof and corrected four references authenticate separately without replacing primary rows',{skip:!existsSync(resolve(correctionStage,'quality-reference-correction'))},async()=>{
  const root=await mkdtemp(resolve(tmpdir(),'moss-reference-correction-'));
  try{
    await cp(resolve(correctionStage,'quality-reference-correction'),resolve(root,'quality-reference-correction'),{recursive:true});
    const frozen=JSON.parse(await readFile('/Users/michael/.codex/epic-runs/sc-26752/quality-inputs-v1/inputs.json')),inputs=Object.fromEntries(frozen.assets.map(x=>[x.id,x])),cells=['base-single','base-four','sft-single','sft-four'];
    const rows=['mlx','cuda'].flatMap(backend=>cells.map(cell=>({id:`quality-${backend}-${cell}`,backend,variant:cell.split('-')[0],refs:cell.endsWith('four')?['B','L','P','G']:['B'],anchors:['actual anchor'],request:{prompt:frozen[cell.endsWith('four')?'four_case':'single_case'].prompt}})));
    const record=async file=>({file,...await fileIdentity(resolve(root,file))}),original={upstream:{historical:'immutable original references'},suiteSummary:{historical:'immutable old summary'}},lock={sources:{sceneworks:{head:'a'.repeat(40)},inference:{head:'b'.repeat(40)}}};
    const correction={version:'sc-26757-mps-causal-reference-correction/v1',amendmentSha256:AMENDMENT_SHA256,lockSha256:digest(lock),originalContextSha256:originalReferenceContextSha256(original),proof:{root:'quality-reference-correction/proof',archiveAuthentication:await record('quality-reference-correction/proof-archive-authentication.json')},corrected:{root:'quality-reference-correction/corrected',archiveAuthentication:await record('quality-reference-correction/corrected-archive-authentication.json')},upstream:{},rows:rows.map(x=>({rowId:x.id})),controlledCudaComparisons:rows.filter(x=>x.backend==='cuda').map(x=>({rowId:x.id}))};
    for(const cell of cells){const prefix='quality-reference-correction/corrected';correction.upstream[cell]={cell,runId:'40438860208',attempt:1,artifactId:'12708569178',toolingSha:'6444adf801a4a3e84b1bfbc5e92d02ddfe0a955c',reviewer:'CPU fixture only',reviewedAt:'2026-10-27T00:00:00Z',observation:'Fixture verifies evidence, not a visual PASS',image:await record(`${prefix}/upstream-${cell}.png`),diagnostic:await record(`${prefix}/upstream-${cell}.diagnostic.json`),authentication:await record(`${prefix}/authentication.json`),noiseSha256:'92e4addfaab4f8f5738c6a29ab2da1a56e2b204726b83250721edfec09608d82',inputs:(cell.endsWith('four')?['B','L','P','G']:['B']).map(id=>({id,sha256:inputs[id].sha256}))};}
    const file='quality-reference-correction.json',load=async value=>{await writeFile(resolve(root,file),JSON.stringify(value));return loadReferenceCorrection(root,{...original,referenceCorrection:await record(file)},lock,rows,inputs);};
    const before=JSON.stringify(original),result=await load(correction);assert.equal(result.referenceCorrectionApplied,true);assert.equal(result.rows.length,8);assert.equal(result.controlledCudaComparisons.length,4);assert.equal(JSON.stringify(original),before);
    for(const mutate of [x=>{x.amendmentSha256=hash;},x=>{x.originalContextSha256=hash;},x=>{x.lockSha256=hash;},x=>{x.proof.archiveAuthentication.sha256=hash;},x=>{x.rows.pop();},x=>{x.controlledCudaComparisons.pop();},x=>{x.upstream['base-single'].image=x.upstream['sft-single'].image;},x=>{x.upstream['base-four'].runId='40438860209';},x=>{x.upstream['sft-four'].inputs.reverse();},x=>{x.upstream['sft-single'].noiseSha256=hash;}]){const bad=structuredClone(correction);mutate(bad);await assert.rejects(()=>load(bad));}
    const output=resolve(root,'quality-reference-correction/proof/upstream/mps-bf16-explicit-bool-causal.f32le');await writeFile(output,'changed');await assert.rejects(()=>load(correction),/bytes changed/);
  }finally{await rm(root,{recursive:true,force:true});}
});
