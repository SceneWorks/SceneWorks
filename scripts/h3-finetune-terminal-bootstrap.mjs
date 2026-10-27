// First-run orchestration for the existing H3 terminal routes. No alternate importer or oracle.
import assert from 'node:assert/strict';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {mkdir,readFile,writeFile,realpath,stat} from 'node:fs/promises';
import {caseBook,prepareCases,finalizeCampaignPlan,prepareControlledLoKr} from './h3-finetune-terminal-cases.mjs';
import {SOURCE,hashFile,sourcePair,validatePlan} from './h3-finetune-terminal.mjs';
import {operationalCases} from './h3-finetune-terminal-reduced.mjs';
import {assertCliEnvironment,privateRunRoot,selectStageFiles,runStage,prepareReferenceFixtures,bindScopedAuthority} from './h3-finetune-terminal-stage.mjs';
import {cudaDeviceEvidence} from './ci/h3-import-native-proof.mjs';
import {prepareBootstrapInputs} from './h3-finetune-terminal-bootstrap-inputs.mjs';
import {prepareActualProbeManifest,inspectActualArtifactHeader} from './h3-finetune-terminal-artifact-probe.mjs';
const execute=promisify(execFile),clone=structuredClone;
const REQUIRED_SETS=['singularity','hybrid','pulpcut','turbo','speech'];
const BASE_LICENSE='MiniMax-H3-Community-License-Agreement';
const sha40=value=>assert.match(value??'',/^[a-f0-9]{40}$/,'actual dispatch revision required');
const sha64=value=>assert.match(value??'',/^[a-f0-9]{64}$/,'actual file identity required');
const positive=value=>assert.ok(Number.isFinite(value)&&value>0,'explicit positive operator bound required');
const save=(file,value)=>writeFile(file,JSON.stringify(value,null,2)+'\n');
const canonicalSha=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const capture=name=>({$capture:name});
function keys(value,allowed,label){assert.ok(value&&typeof value==='object'&&!Array.isArray(value),`${label} object required`);for(const key of Object.keys(value))assert.ok(allowed.includes(key),`unsupported ${label} field: ${key}`);}
export function validateBootstrapRequest(request,{backend,unitFake=false}={}){
  assert.ok(['mlx','cuda'].includes(backend));assert.equal(request.stationary_pair_authorized,true,'explicit stationary pair authorization required');
  assert.ok(request.inspect_only!==true&&request.inspectOnly!==true,'inline bootstrap prepares actual inputs; inspect-only cannot produce a ready cohort');
  assert.ok(request.prepare_identity_controls!==false&&request.prepare_reference_fixtures!==false,'selected reduced cohort requires its identity and reference controls');
  assert.ok(request.prepare_float_variants!==true,'optional dense conversion is not part of the reduced bootstrap cohort');
  const outerSets=request.asset_sets??request.assetSets;if(outerSets!==undefined)assert.deepEqual([...outerSets].sort(),[...REQUIRED_SETS].sort(),'bootstrap asset sources come only from the pinned reduced cohort');
  const selected=request.case_ids??(request.case_id?[request.case_id]:[]);assert.ok(Array.isArray(selected));for(const id of selected)assert.ok(operationalCases(backend).some(c=>c.id===id),'unknown or wrong-backend bootstrap case');
  const b=request.bootstrap;keys(b,['casebook','cohort','operator','componentSelections','components','legalAuthority'],'bootstrap');
  assert.equal(b.casebook,'sc-26691-h3-cases-v1');assert.equal(b.cohort,'reduced-v1');
  assert.ok(request.files===undefined&&request.urls===undefined&&request.commands===undefined,'bootstrap cannot override sources or execute supplied commands');
  const op=b.operator;keys(op,['compatibleBase','projectName','seed','requestedGpu','gpuId','port','componentTier','hostBudgetBytes','deviceBudgetBytes','minimumControlDelta','maximumBaselineDelta','width','height','wideWidth','wideHeight','timeoutSeconds','lokrAmplitude','pulpcutSampling'],'operator');
  assert.equal(op.compatibleBase,'builtin:minimax_h3','explicit compatible base binding required');
  assert.ok(Number.isSafeInteger(op.seed)&&op.seed>=0&&op.seed<=0xffffffff);assert.ok(typeof op.requestedGpu==='string'&&op.requestedGpu.length);assert.ok(typeof op.gpuId==='string'&&op.gpuId.length);
  if(backend==='cuda')assert.equal(op.gpuId,'0','manual bootstrap retains product GPU zero');
  assert.ok(Number.isInteger(op.port)&&op.port>1024&&op.port<=65535);assert.ok(['q4','q8','bf16'].includes(op.componentTier));
  for(const field of ['hostBudgetBytes','deviceBudgetBytes','minimumControlDelta','lokrAmplitude'])positive(op[field]);assert.ok(Number.isFinite(op.maximumBaselineDelta)&&op.maximumBaselineDelta>=0);
  for(const field of ['width','height','wideWidth','wideHeight'])assert.ok(Number.isInteger(op[field])&&op[field]>0&&op[field]%32===0,'explicit stride-aligned cohort geometry required');
  assert.equal(Math.min(op.width,op.height),544,'544-short-edge recipes require their declared geometry');assert.equal(Math.min(op.wideWidth,op.wideHeight),768,'768 recipe requires its declared geometry');
  assert.ok(op.timeoutSeconds===undefined||Number.isInteger(op.timeoutSeconds)&&op.timeoutSeconds>0&&op.timeoutSeconds<=7200);
  keys(op.pulpcutSampling,['steps','schedulerShift','audioSchedulerShift','contractSource'],'PulpCut sampling');assert.equal(op.pulpcutSampling.steps,8,'published PulpCut merge declares eight steps');for(const field of ['schedulerShift','audioSchedulerShift'])positive(op.pulpcutSampling[field]);assert.ok(typeof op.pulpcutSampling.contractSource==='string'&&op.pulpcutSampling.contractSource.length>10,'record the actual operator/source sampling authority; do not invent a trained shift');
  const selections=b.componentSelections??b.components;assert.ok(Array.isArray(selections)&&selections.length===2,'select the two existing compatible bases explicitly');assert.ok(!(b.componentSelections&&b.components),'choose one component selection field');
  for(const id of ['minimax_h3','minimax_h3_ref'])assert.equal(selections.filter(v=>v.modelId===id&&v.tier===op.componentTier).length,1,'same explicit component tier required for both base partitions');
  for(const selection of selections)keys(selection,['modelId','tier'],'component selection');
  const legal=b.legalAuthority;assert.ok(legal&&typeof legal==='object');assert.equal(legal.acknowledged,true,'explicit scoped operator acknowledgment required; none is automatic');
  assert.ok(unitFake||legal.proof_kind!=='unit_fake','synthetic acknowledgment is not real approval');
  for(const field of ['assetSets','licenseIds','inheritedLicenseIds'])assert.ok(Array.isArray(legal[field])&&legal[field].every(v=>typeof v==='string'&&v.length),`explicit legal ${field} scope required`);
  for(const set of REQUIRED_SETS)assert.ok(legal.assetSets.includes(set),`asset set ${set} is outside approved scope`);
  for(const id of [BASE_LICENSE,'Apache-2.0','CC-BY-4.0'])assert.ok(legal.licenseIds.includes(id),`license ${id} is outside approved scope`);
  assert.ok(legal.inheritedLicenseIds.includes(BASE_LICENSE),'inherited MiniMax terms are outside approved scope');assert.equal(legal.includeSelectedComponents,true,'selected component scope must be explicit');
  return {bootstrap:b,operator:op,selections,legal};
}
function scopedFact(role,sha256,legal,origin){sha64(sha256);return {role,sha256,legal,origin};}
function perFileAcknowledgment(input,entry,origin){
  const legal=entry.legal;assert.ok(legal?.licenseId,'authoritative license facts required');assert.ok(input.licenseIds.includes(legal.licenseId),'component license outside approved scope');if(legal.inheritedLicenseId)assert.ok(input.inheritedLicenseIds.includes(legal.inheritedLicenseId),'inherited component license outside approved scope');
  return {sha256:entry.expectedSha256,acknowledged:input.acknowledged,licenseId:legal.licenseId,...(legal.inheritedLicenseId?{inheritedLicenseId:legal.inheritedLicenseId,inheritedAcknowledged:input.acknowledged}:{}),...(legal.attributionRequired?{attribution:legal.attributionRequired}:{}),origin};
}
async function prepareSpeech({stageRoot,stagePath,swRoot,env}){
  assert.ok(env.H3_SPEECH_PYTHON&&path.isAbsolute(env.H3_SPEECH_PYTHON),'private preflighted CPU interpreter required for speech');
  const directory=path.join(stageRoot,'conditioning','speech-1272-128104-0000');
  await execute(env.H3_SPEECH_PYTHON,['-I','-B',path.join(swRoot,'scripts/h3-finetune-terminal-speech.py'),'--stage-receipt',stagePath,'--stage-root',stageRoot,'--output-dir',directory],{cwd:swRoot,env,timeout:120000,maxBuffer:1024*1024});
  const receipt=JSON.parse(await readFile(path.join(directory,'speech-receipt.json'),'utf8'));assert.equal(receipt.status,'completed');assert.equal(receipt.proof_kind,'campaign_native');
  return {...receipt,directory};
}
function sampling(op,recipe,book){
  const geometry=recipe?.trainingShortEdge===768?{width:op.wideWidth,height:op.wideHeight}:{width:op.width,height:op.height};
  return {...geometry,fps:24,duration:124/24,steps:recipe?.steps??50,...(recipe?{schedulerShift:recipe.schedulerShift,audioSchedulerShift:recipe.audioSchedulerShift}:{}),contractSource:recipe?`Frozen recipe ${recipe.id}; ${book.turboArtifacts.repo}@${book.turboArtifacts.revision}`:'Explicit compatible-base unaccelerated 50-step cohort recipe; no teacher is staged'};
}
function recipeFor(c,book){const index=({'fl-turbo-4v1.0-768':0,'fl-turbo-8':1,'fl-turbo-4v01':2})[c.adapterRecipe];return index!==undefined?book.sampling.turbo[index]:[1,3,5,6,7,8,9,10,11,12,13,14].includes(c.index)?book.sampling.turbo[3]:null;}
function primaryFor(c,book){const match=/^artifact\.(\d)-/.exec(c.modelKey);return match?book.artifacts[Number(match[1])]:book.stagingAssets[c.assetSet==='independent'?'pulpcut':'hybrid'][0];}
function requestedMode(c){return ({text:'text_to_video',image:'image_to_video','last-frame-only':'image_to_video','first-last-frame':'first_last_frame',reference:'reference_to_video','ordered-mixed':'reference_to_video'})[c.conditioning]??'reference_to_video';}
async function checkedInput(file,role){return {path:file,sha256:await hashFile(file),role};}
export async function bootstrapTerminal({request,backend,swRoot,inferenceRoot,evidenceRoot,env=process.env,outputRoot,deps={}}){
  const fake=deps.proof_kind==='unit_fake';assert.ok(fake||Object.keys(deps).length===0,'bootstrap dependency injection is unit_fake only');
  const {operator:op,selections,legal}=validateBootstrapRequest(request,{backend,unitFake:fake});
  if(!fake)assertCliEnvironment({env,platform:process.platform,backend});
  const sceneworks=env.H3_SCENEWORKS_SHA??env.GITHUB_SHA,inference=env.H3_INFERENCE_SHA;sha40(sceneworks);sha40(inference);assert.equal(sceneworks,env.GITHUB_SHA);
  const nativeEvidence={ref:env.H3_NATIVE_EVIDENCE_REF,sha:env.H3_NATIVE_EVIDENCE_SHA,root:path.join(path.dirname(inferenceRoot),'inference-evidence'),productPin:inference};sha40(nativeEvidence.sha);assert.equal(nativeEvidence.ref,'feature/sc-26679-h3-finetune-import');
  assert.ok(path.isAbsolute(outputRoot)&&path.isAbsolute(evidenceRoot));for(const source of [swRoot,inferenceRoot,nativeEvidence.root]){const relative=path.relative(path.resolve(source),path.resolve(outputRoot));assert.ok(relative&&(relative.startsWith('..')||path.isAbsolute(relative)),'bootstrap output must be outside source checkouts');}
  const physicalOutput=path.join(await realpath(path.dirname(outputRoot)),path.basename(outputRoot));for(const source of [swRoot,inferenceRoot,nativeEvidence.root]){const relative=path.relative(await realpath(source),physicalOutput);assert.ok(relative&&(relative.startsWith('..')||path.isAbsolute(relative)),'physical bootstrap output cannot enter source checkouts');}
  await mkdir(outputRoot);await mkdir(evidenceRoot,{recursive:true});
  const marker={kind:'h3-terminal-bootstrap-owner',version:1,backend,sceneworks,inference,nativeEvidence,runId:env.GITHUB_RUN_ID,runAttempt:env.GITHUB_RUN_ATTEMPT};const startedAt=env.H3_TERMINAL_JOB_STARTED_AT??new Date().toISOString();assert.ok(Number.isFinite(Date.parse(startedAt)));await save(path.join(outputRoot,`lane-start-${backend}.json`),{sceneworks,inference,startedAt,budgetSeconds:86400});await save(path.join(outputRoot,'bootstrap-owner.json'),marker);
  const statePath=path.join(outputRoot,'bootstrap.json'),writeState=async value=>{await save(statePath,value);await save(path.join(evidenceRoot,'bootstrap.json'),value);};await writeState({...marker,status:'preparing',prepared:false,proof:false});
  try{
    if(!fake)await sourcePair({sceneworks,inference,output:outputRoot},swRoot,inferenceRoot);
    let deviceId=op.gpuId;if(backend==='cuda'){if(fake)deviceId='unit_fake_gpu';else{const csv=(await execute('nvidia-smi',['--query-gpu=index,uuid,pci.bus_id,name,driver_version','--format=csv,noheader'],{env,timeout:30000})).stdout;deviceId=cudaDeviceEvidence({...env,CUDA_DEVICE_ORDER:'PCI_BUS_ID',CUDA_VISIBLE_DEVICES:'0'},csv).selectedPhysicalDevice.uuid;}}
    const book=fake&&deps.book?deps.book:caseBook;assert.deepEqual(book.source,SOURCE);
    const authorityPath=path.join(outputRoot,'operator-acknowledgment.json');await save(authorityPath,legal);const authorityInput=await checkedInput(authorityPath,'operator-acknowledgment');
    const facts=[];for(const set of REQUIRED_SETS)for(const file of selectStageFiles(book,[set]))facts.push(scopedFact(set==='turbo'?'adapter':'source',file.expectedSha256,file.legal,{kind:'casebook-asset',assetSet:set,repo:file.repo,revision:file.revision,file:file.file}));
    let authority=bindScopedAuthority(legal,facts,{allowUnitFake:fake});
    const stage=await(deps.stage??runStage)({request:{asset_sets:REQUIRED_SETS,prepare_identity_controls:true,license_authority:authority},backend,env,platform:process.platform,evidenceDir:evidenceRoot,...(fake?{book,allowUnitFake:true}: {})});
    assert.equal(stage.kind,'h3-finetune-stage-receipt');assert.equal(stage.status,'completed');assert.ok(fake?stage.proof_kind==='unit_fake':stage.proof_kind==='frozen_stage_content');
    const stageRoot=stage.settings.privateRoot,stagePath=path.join(stageRoot,'h3-stage-receipt.json');assert.equal(await realpath(stageRoot),await realpath(privateRunRoot({backend,platform:process.platform,env,runId:env.GITHUB_RUN_ID,attempt:env.GITHUB_RUN_ATTEMPT})));const stageOwner=JSON.parse(await readFile(path.join(stageRoot,'sc26691-owner.json'),'utf8'));assert.equal(stageOwner.codeSha,sceneworks);assert.equal(stageOwner.backend,backend);assert.equal(String(stageOwner.runId),String(env.GITHUB_RUN_ID));assert.equal(String(stageOwner.runAttempt),String(env.GITHUB_RUN_ATTEMPT));assert.ok(stage.identityControls?.differentBytes?.actualSHA,'actual identity controls must exist before building their source bindings');
    const acknowledgmentFor=async(entry,context)=>{
      if(context?.reference)return {scope:'existing-pinned-producer-source',acknowledged:legal.acknowledged};
      assert.equal(context?.component,true);assert.ok(selections.some(v=>v.modelId===context.modelId),'component source outside selected catalog bases');assert.equal(legal.includeSelectedComponents,true);
      const origin={kind:'catalog-component',modelId:context.modelId,componentId:context.componentId};const ack=perFileAcknowledgment(legal,entry,origin);
      if(!context.scopeOnly){sha64(entry.expectedSha256);facts.push(scopedFact('component',entry.expectedSha256,entry.legal,origin));}
      return ack;
    };
    const resolved=await(deps.resolveInputs??prepareBootstrapInputs)({swRoot,stageRoot,evidenceRoot,backend,componentSelections:selections,nativeEvidence,compatibleBase:op.compatibleBase,acknowledgmentFor,env,...(fake?{proofKind:'unit_fake',deps:deps.inputDeps??{}}:{})});
    for(const id of ['minimax_h3','minimax_h3_ref'])assert.ok(Array.isArray(resolved.componentsByModel?.[id])&&resolved.componentsByModel[id].length,`required actual component closure missing: ${id}`);
    stage.files.push(...resolved.componentRecords);authority=bindScopedAuthority(legal,facts,{allowUnitFake:fake});stage.licenseAuthority=authority;await save(stagePath,stage);
    const references=await(deps.prepareReferences??prepareReferenceFixtures)({root:stageRoot,evidenceDir:evidenceRoot,backend,env,...(fake?{allowUnitFake:true,book,execute:deps.referenceExecute,tools:deps.tools}: {})});assert.equal(references.status,'prepared-not-modelproof');
    const speech=await(deps.prepareSpeech??prepareSpeech)({stageRoot,stagePath,swRoot,env});assert.equal(speech.status,'completed');
    const media=[];for(const file of references.fixtures){assert.equal(await hashFile(file.outputPath),file.sha256);media.push({key:file.assetKey,path:file.outputPath,sha256:file.sha256,role:`conditioning-${file.kind}`,mimeType:file.kind==='image'?'image/png':file.kind==='video'?'video/mp4':'audio/wav'});}
    for(const [role,key]of [['speech-source','voiceA'],['diagnostic-reversed-not-speaker','voiceB']]){const file=speech.outputs.find(v=>v.role===role);assert.ok(file);const filename=path.join(speech.directory,file.path);assert.equal(await hashFile(filename),file.sha256);media.push({key,path:filename,sha256:file.sha256,role:'conditioning-audio',mimeType:'audio/wav'});}
    for(const key of ['imageA','imageB','clipA','clipB','voiceA','voiceB'])assert.ok(media.some(v=>v.key===key),`required generated conditioning ${key} missing`);
    const pruned=stage.files.find(v=>v.file===book.artifacts[2].artifact);assert.ok(pruned);const cfg=resolved.config.dit;
    const inspected=await inspectActualArtifactHeader({path:pruned.finalPath,bytes:pruned.actualBytes});const q=cfg.num_attention_heads*cfg.attention_head_dim,h=cfg.hidden_size;
    const normalizedHeader=Object.fromEntries(Object.entries(inspected.header).map(([name,value])=>[name.replace(/^(?:model\.diffusion_model\.|diffusion_model\.)/,''),value]));const nativeQ=normalizedHeader['transformer_blocks.0.attn.to_q.weight'],fused=normalizedHeader['blocks.0.attn.qkv_proj.weight']??normalizedHeader['transformer_blocks.0.attn.qkv_proj.weight'];assert.ok(nativeQ||fused,'actual source has no supported canonical/body Q projection');assert.deepEqual((nativeQ??fused).shape,nativeQ?[q,h]:[3*q,h],'LoKr dimensions must derive from the actual source and explicit compatible config');
    const lokr=await prepareControlledLoKr({target:'transformer_blocks.0.attn.to_q',shape:[q,h],seed:op.seed,amplitude:op.lokrAmplitude,sourceCheckpointSha256:pruned.observedSha256,sourceContractSha256:resolved.config.sha256,sourceDeclaration:`Explicit ${op.compatibleBase}, actual source ${pruned.observedSha256}; source Q shape verified`,...(fake?{proof_kind:'unit_fake'}:{})},path.join(outputRoot,'generated-lokr'));
    facts.push(scopedFact('adapter',lokr.provenance.sha256,{licenseId:'CC0-1.0',inheritedLicenseId:BASE_LICENSE,attributionRequired:'Generated compatibility factors; not trained'}, {kind:'generated-control',assetSet:'singularity',generator:'prepareControlledLoKr',sourceSha256:pruned.observedSha256}));
    const identity=stage.identityControls;facts.push(scopedFact('source',identity.differentBytes.actualSHA,book.stagingLicenses.singularity,{kind:'generated-control',assetSet:'singularity',generator:'single-weight-byte-identity-control',sourceSha256:identity.sourceActualSHA}));
    authority=bindScopedAuthority(legal,facts,{allowUnitFake:fake});stage.licenseAuthority=authority;stage.bootstrap={kind:'source-bound-preparation',operatorInputSha256:authority.inputSha256,derivedInputs:[...media.map(v=>({path:v.path,sha256:v.sha256})),{path:lokr.file,sha256:lokr.provenance.sha256}],proof:false};await save(stagePath,stage);
    // Freeze the full parent receipt before either probe projection binds its hash.
    const stageReceipt={path:stagePath,sha256:await hashFile(stagePath)},actualArtifactProbes={},probeInputs={};
    for(const c of operationalCases(backend).filter(v=>v.kind==='actual-artifact-probe')){const index=c.index===23?2:3,row=stage.files.find(v=>v.file===book.artifacts[index].artifact);assert.ok(row);const directory=path.join(outputRoot,c.id);await mkdir(directory);const prepared=await prepareActualProbeManifest({backend,probe:c.probe,artifact:{path:row.finalPath,sha256:row.observedSha256,bytes:row.actualBytes},stagingReceipt:stageReceipt,outputDir:directory,dit:cfg,ditBinding:{path:resolved.config.path,sha256:resolved.config.sha256},nativeEvidence,proofKind:fake?'unit_fake':'campaign_prepared_metadata'});actualArtifactProbes[c.id]={kind:'actual-artifact-probe',probe:c.probe,spec:prepared.spec,sourceRole:'checkpoint',receipt:'artifact-probe-receipt.json',noAV:true,referenceDir:resolved.referenceDir};probeInputs[c.id]=prepared.inputs;}
    const project=capture('project/id'),setupSteps=[{id:'project',method:'POST',path:'/api/v1/projects',body:{name:op.projectName??'H3 reduced terminal proof'},expectedStatus:201}],assets={};
    for(const file of media.filter(v=>['imageA','imageB','clipA','clipB','voiceA','voiceB'].includes(v.key))){const id=`asset_${file.key}`;setupSteps.push({id,method:'POST',path:{$concat:['/api/v1/projects/',project,'/assets']},upload:{path:file.path,sha256:file.sha256,mimeType:file.mimeType,fileName:path.basename(file.path)},expectedStatus:201});assets[file.key]=capture(`${id}/id`);}
    const bindings={},caseInputs={},caseRuntime={},timeoutSeconds=op.timeoutSeconds??7200;
    const commonInputs=[authorityInput,...resolved.inputs,...media.map(({path:filename,sha256,role})=>({path:filename,sha256,role})),await checkedInput(lokr.file,'adapter')];
    const rowFor=source=>stage.files.find(v=>v.observedSha256===(source.expectedSha256??source.sha256));
    for(const c of operationalCases(backend)){
      const source=primaryFor(c,book),row=rowFor(source);assert.ok(row,`actual staged source missing for ${c.id}`);caseInputs[c.id]=c.kind==='actual-artifact-probe'?[authorityInput,...resolved.inputs.filter(v=>['reference','source-config'].includes(v.role))]:clone(commonInputs);caseInputs[c.id].push({path:row.finalPath,sha256:row.observedSha256,role:'checkpoint',...(c.group==='artifact'?{artifact:c.index<=4?book.artifacts[c.index-1].artifact:undefined}:{})});
      if(c.kind==='actual-artifact-probe'){caseInputs[c.id].push(...probeInputs[c.id]);continue;}
      const partition=c.assetSet==='hybrid'?'hybrid':c.assetSet==='independent'?'fl2va':'ref2va',topology=[1,2,5,6,7,8,9,10,11,12].includes(c.index)?'full':'pruned',modelId=`imported_sc26691_${c.modelKey.replace(/[^a-zA-Z0-9]/g,'_')}`;
      const model={modelId,sourcePath:row.finalPath,sha256:row.observedSha256,baseModel:partition==='ref2va'?'minimax_h3_ref':'minimax_h3',partition,topology,sourceKind:'file',sourceFormat:'native-single',componentTier:op.componentTier,components:clone(resolved.componentsByModel[partition==='ref2va'?'minimax_h3_ref':'minimax_h3']),supportedModes:partition==='ref2va'?['reference_to_video']:partition==='hybrid'?['text_to_video','image_to_video','first_last_frame','reference_to_video']:['text_to_video','image_to_video','first_last_frame']};
      if(partition==='hybrid')model.hybridBinding={kind:'single-checkpoint',declarationSource:`${source.repo}@${source.revision}/${source.file}: author-declared hybrid selection merge; same imported bytes permitted by the declared contract`};if(c.index===21)model.lineage={source:source.repo,revision:source.revision,kind:'trained-turbo-delta-merge-requant-not-independent-full-training'};
      const recipe=recipeFor(c,book),adapters=[];if(recipe&&c.index!==12){const index=book.sampling.turbo.findIndex(v=>v.id===recipe.id),file=book.turboArtifacts.files[index],staged=stage.files.find(v=>v.observedSha256===file.sha256);assert.ok(staged);adapters.push({id:`sc26691_${recipe.id.replace('minimax_h3_','')}_${canonicalSha(modelId).slice(0,12)}`,recipeId:recipe.id,kind:'lora',role:'accelerator',sha256:staged.observedSha256,source:book.turboArtifacts.repo,revision:book.turboArtifacts.revision,compatibilitySource:`Frozen ${recipe.id}; explicit ${partition}/${topology} import`,topology,partition,licenseAcknowledged:legal.acknowledged,importPath:staged.finalPath});caseInputs[c.id].push({path:staged.finalPath,sha256:staged.observedSha256,role:'adapter'});}
      if([3,13].includes(c.index))adapters.push({id:`sc26691_lokr_${modelId}`,kind:'lokr',sha256:lokr.provenance.sha256,source:'controlled-generated-body-only-not-trained',revision:sceneworks,compatibilitySource:`Actual bound source ${pruned.observedSha256} Q projection`,topology,partition,licenseAcknowledged:legal.acknowledged,importPath:lokr.file});
      const e={label:'source',licenseAuthority:clone(authority),model,mode:requestedMode(c),sampling:sampling(op,recipe,book),adapters};if(c.index===21)e.sampling={...sampling(op,null,book),...clone(op.pulpcutSampling)};
      if(c.index===4){model.originalPath=identity.sourcePath;model.originalSha256=identity.sourceActualSHA;model.sourcePath=identity.renamed.path;Object.assign(e,{ownershipMode:'linked',linkedRootId:capture('library/rootId'),linkedRelativePath:path.relative(stageRoot,identity.renamed.path).split(path.sep).join('/')});const negative=clone(e);negative.label='different_bytes';negative.model.modelId='imported_sc26691_identity_negative';negative.model.sourcePath=identity.differentBytes.path;negative.model.sha256=identity.differentBytes.actualSHA;negative.identityControl={kind:'different-bytes-negative',expectedSourceSha256:identity.sourceActualSHA};delete negative.ownershipMode;delete negative.linkedRootId;delete negative.linkedRelativePath;bindings[c.id]=[e,negative];caseInputs[c.id].push({path:identity.renamed.path,sha256:identity.renamed.actualSHA,role:'checkpoint'},{path:identity.differentBytes.path,sha256:identity.differentBytes.actualSHA,role:'checkpoint'});}
      else bindings[c.id]=[e];
    }
    // Every MANAGED source and every adapter is imported from a private copy inside the case's own
    // app data directory, never straight from the staging root: the API confines a local-copy
    // import to <data>/models and a LoRA import to <data>/loras (its LAN-exposure rule), so a
    // staging-root path is refused. The copy is a hardlink — the staging root and the case
    // directories share a volume — so a 34 GB checkpoint still costs nothing per case, and a
    // source-drift action renames the seeded entry rather than writing through it, so the staged
    // original is never modified. Linked and Hugging Face sources keep their own locators.
    const seedInto=(caseId,input,sub)=>{
      const destination=`data/${sub}/${path.basename(input)}`,runtime=(caseRuntime[caseId]??={seedFiles:[...resolved.seedFiles]});
      if(!runtime.seedFiles.some(v=>v.destination===destination))runtime.seedFiles.push({input,destination,mode:'hardlink'});
      return path.join(outputRoot,'services',caseId,...destination.split('/'));
    };
    for(const [caseId,entries] of Object.entries(bindings))for(const e of entries){
      if(e.model?.sourcePath&&e.ownershipMode!=='linked'&&!e.importSource)e.model.sourcePath=seedInto(caseId,e.model.sourcePath,`models/sc26691-${e.label}`);
      for(const a of e.adapters??[])if(a.importPath)a.importPath=seedInto(caseId,a.importPath,'loras/sc26691');
    }
    const parameters={backend,sceneworks,inference,swRoot,inferenceRoot,output:outputRoot,nativeEvidence,projectId:project,seed:op.seed,requestedGpu:op.requestedGpu,baseUrl:`http://127.0.0.1:${op.port}`,timeoutSeconds,assets,setupSteps:[...setupSteps,{id:'library',method:'POST',path:'/api/v1/models/library-roots',body:{path:stageRoot,label:'sc26691 owned staged library'},expectedStatus:201}],bindings,actualArtifactProbes,caseInputs,caseRuntime,runtime:{port:op.port,gpuId:op.gpuId,deviceId,seedFiles:resolved.seedFiles},budgets:{[backend]:{hostBytes:op.hostBudgetBytes,deviceBytes:op.deviceBudgetBytes}},sensitivity:{minimumControlDelta:op.minimumControlDelta,maximumBaselineDelta:op.maximumBaselineDelta},identificationControlReceipt:path.join(stageRoot,'fixtures','identity-controls.json'),nextAuthorizedAction:'Run only the explicit stationary-pair reduced terminal cases in this owned manual lane; no implicit approval or publication',...(fake?{proof_kind:'unit_fake',unitCasebook:book}:{})};
    const preparedRoot=path.join(outputRoot,'prepared');await prepareCases(parameters,preparedRoot);const bound=await finalizeCampaignPlan(parameters,preparedRoot);const plan=JSON.parse(await readFile(bound.plan,'utf8'));plan.assetStaging={root:stageRoot,backend,receipt:stageReceipt};plan.bootstrap={inputSha256:canonicalSha(request.bootstrap),authorityInput,prepared:true,proof:false};validatePlan(plan);const readyRoots=await Promise.all([outputRoot,stageRoot,swRoot,inferenceRoot,nativeEvidence.root].map(root=>realpath(root)));for(const file of new Set(plan.cases.flatMap(c=>c.inputs.map(i=>i.path)))){const resolvedPath=await realpath(file),info=await stat(resolvedPath);assert.ok(info.isFile()&&info.size>0,`required prepared input missing/empty: ${file}`);assert.ok(readyRoots.some(root=>{const relative=path.relative(root,resolvedPath);return relative&&!relative.startsWith('..')&&!path.isAbsolute(relative);}),'prepared input escapes the owned task/source roots');}await save(bound.plan,plan);
    const result={...marker,status:'prepared-not-executed',planPath:bound.plan,output:outputRoot,prepared:true,proof:false,cases:plan.cases.length,operatorInputSha256:authority.inputSha256,stageReceipt};await writeState(result);return result;
  }catch(error){await writeState({...marker,status:'failed',prepared:false,proof:false,error:error.message});throw error;}
}
