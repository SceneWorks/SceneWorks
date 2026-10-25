#!/usr/bin/env node
// sc-26757 native Actions controller. Imports are inert; only explicit CLI phases execute.
import { constants } from 'node:fs';
import { randomBytes } from 'node:crypto';
import { mkdir, readFile, writeFile, cp, appendFile, chmod, rename, readdir, lstat, realpath, mkdtemp, unlink, rmdir, stat, copyFile } from 'node:fs/promises';
import { execFile as callback } from 'node:child_process';
import { promisify } from 'node:util';
import { resolve, dirname, basename } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { VERSION, BACKEND_FILES, sha, digest, canonical, requireThat, json, save, safePath, fileIdentity, treeIdentity, createPlan, validatePlan, validateArtifactInventory, validateAdmission, assertDispatchClear, reconcileAttempt, compareTypedMossEditHistory } from './lib/moss-edit-terminal.mjs';
import { prepare, freeze, bind, frozen } from './moss-edit-terminal.mjs';
import { command, sourceIdentity, filesUnder, assertOriginalEvidence, assertH18CheckpointFiles, continuationRemainingRows, retryRemainingRows, postRowsGuardFrontier, reconcileParserRefusal, postDmdGuardFrontier, reconcileDmdInputCorrection, validateCompletedMlxArtifact, stageCompletedMlxHandoffSupplement, validateCompletedMlxHandoffSupplement, submittedCanceledRemainingRows, validateFailedUiCapture, continuationRowDisposition, contained, nativeDevice, sampler, startStack, exportState, importState, retainQualityPairs, delay, CUDA_UUID, R16_ORIGINAL_INVENTORY_SHA256, H18_RETRY_TOOLING_SHA, H18_RETRY_RUN_ID, H18_RETRY_ARTIFACT_ID, H18_RETRY_INVENTORY_SHA256 } from './lib/moss-edit-terminal-actions.mjs';
import { inspectJournal, noLoadComparison, OBSERVATION_METHOD, hostBudgetHeader, hostGuardSpec, hostTensorInvariant } from './lib/moss-edit-terminal-observation.mjs';
import { stripJsoncComments } from './lib/jsonc.mjs';
import { cudaSourceContinuity, validateCudaRebaseRecord } from './lib/moss-edit-cuda-rebase.mjs';

import { sealPrivateHf, transferPrivateHf, adoptOriginalPrivateHf, adoptStoppedExecutionPrivateHf, adoptStoppedCudaBuildPrivateHf, releasePrivateHf, privateHfIntegrity, materializePrivateAdapter, exportMossEnvironment, cargoTarget, MAC_MOSS_BASE } from './lib/moss-private-hf.mjs';

const execFile=promisify(callback);
const optional=async path=>json(path).catch(error=>{if(error.code==='ENOENT')return null;throw error;});
async function latestRetry(evidenceRoot) {
  let retry=await optional(resolve(evidenceRoot,'continuation-retry.json'));
  for(let ordinal=2;retry;ordinal++) {
    const next=await optional(resolve(evidenceRoot,`continuation-retry-${ordinal}.json`));
    if(!next)return retry;
    requireThat(next.ordinal===ordinal&&next.previousToolingSha===retry.toolingSha,'retained retry chain changed');
    retry=next;
  }
  return null;
}
async function guardRepairFields(evidenceRoot,retry) {
  if(!retry)return{};
  if(retry.guardFrontier)retry={...retry,[retry.guardFrontier.name==='edit-dmd'?'postGuardDmd':'postRowsGuard']:retry.guardFrontier};
  if(retry.postRowsGuard)return {postRowsGuard:retry.postRowsGuard,postRowsGuardOrdinal:retry.ordinal};
  if(retry.postGuardDmd) {
    const earlier=await json(resolve(evidenceRoot,'continuation-retry-7.json'));
    requireThat(earlier.ordinal===7&&earlier.postRowsGuard?.name==='training-shift-nonfinite'&&retry.ordinal===8&&retry.previousToolingSha===earlier.toolingSha,'DMD retry lost the sealed earlier lexical refusal');
    return {postRowsGuard:earlier.postRowsGuard,postRowsGuardOrdinal:7,postGuardDmd:retry.postGuardDmd};
  }
  return{};
}
async function npmCommand(args,options) {
  if(process.platform!=='win32')return command('npm',args,options);
  // .cmd is not an executable under shell:false. Use the installed Node npm
  // entrypoint directly, retaining argument-array semantics without cmd quoting.
  const cli=resolve(dirname(process.execPath),'node_modules/npm/bin/npm-cli.js');await fileIdentity(cli);
  return command(process.execPath,[cli,...args],options);
}
export function runtimeConfig(bundle,{sourceRoot,inferenceRoot,root,backend,buildRoot,bundleRoot}) {
  requireThat(['mlx','cuda'].includes(backend)&&bundle.backends?.[backend],'native backend configuration missing');
  requireThat(/^[a-f0-9]{40}$/.test(bundle.sourceSha)&&/^[a-f0-9]{40}$/.test(bundle.inferenceSha)&&/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(bundle.candidateId),'exact reviewed paired source/candidate identity required');
  const platform=bundle.backends[backend];requireThat(platform.hfHome&&Number.isInteger(platform.port)&&platform.port>=1024&&platform.port<=65535&&Array.isArray(platform.packageFiles)&&platform.packageFiles.length,'actual native package files/HF home/unprivileged loopback port required');
  requireThat(platform.installAdapter===true&&platform.editAdapterFiles?.length,'each native build must retain the real missing-adapter install/select UI phase before freezing its complete package inventory');
  return {...bundle,...platform,sourceRoot:resolve(sourceRoot),inferenceRoot:resolve(inferenceRoot),root:resolve(root),stateRoot:resolve(root,'state'),backend,buildRoot:resolve(buildRoot),bundleRoot:resolve(bundleRoot),
    packageSource:resolve(platform.hfHome),hfHome:resolve(root,'hf'),
    workerId:`moss-${bundle.candidateId}-${backend}`,observationJournal:resolve(root,'journal-0.jsonl'),playwrightModule:resolve(root,'playwright/node_modules/playwright/index.mjs'),
    handoffRoot:resolve(root,'prior/handoff'),priorEvidenceRoot:resolve(root,'prior/evidence'),reviewRoot:resolve(root,'reviews')};
}
export const projectionCopyFlags=(platform,sourceDev,targetDev)=>constants.COPYFILE_EXCL|(platform==='darwin'&&sourceDev===targetDev?constants.COPYFILE_FICLONE:0);
export async function isolatePackages(config,{omitAdapter=false,installedInventory=null}={}) {
  requireThat(config.hfHome===resolve(config.root,'hf')&&config.packageSource!==config.hfHome,'native jobs require their own HF projection; shared caches are read-only');
  requireThat(!installedInventory,'installed private HF must transfer ownership without copying');
  for(const item of config.packageFiles) {
    if(omitAdapter&&(config.editAdapterFiles??[]).includes(item.file))continue;
    const source=safePath(config.packageSource,item.file);requireThat((await fileIdentity(source)).sha256===item.sha256,'source package file differs from immutable input inventory');
    const target=safePath(config.hfHome,item.file);await mkdir(dirname(target),{recursive:true,mode:0o700});requireThat((await stat(source)).isFile(),'package projection source must be a regular file');await copyFile(source,target,projectionCopyFlags(process.platform,(await stat(source)).dev,(await stat(dirname(target))).dev));
  }
}
export function validatePackagePlan(config,manifest) {
  const components=['moss_image_base','moss_image_sft'].map(model=>manifest.models?.find(item=>item.id===model)?.conditionalComponents?.find(item=>item.componentId==='moss-image-edit'&&item.requiredFor?.includes('moss_image_edit')));
  requireThat(components.every(item=>item?.repo==='chinoll/Moss-Image-Edit'&&/^[a-f0-9]{40}$/.test(item.revision)&&canonical(item.files?.slice().sort())===canonical(['adapter_config.json','adapter_model.safetensors'])),'integrated source must expose the exact complete MOSS edit package on both Base/SFT');
  const component=components[0];requireThat(components.every(item=>item.repo===component.repo&&item.revision===component.revision&&item.weightsSha256===component.weightsSha256),'Base/SFT edit package source identity differs');
  const prefix=`hub/models--${component.repo.replaceAll('/','--')}/snapshots/${component.revision}/`,expected=component.files.map(file=>prefix+file).sort();
  requireThat(canonical(config.editAdapterFiles.slice().sort())===canonical(expected),'installer must omit exactly both pinned adapter projection files, preserving the complete base package');
  requireThat(config.packageFiles.find(item=>item.file===prefix+'adapter_model.safetensors')?.sha256===component.weightsSha256,'package inventory adapter weight SHA differs from integrated source');
}
export async function validateBundle(root,expectedSha,sourceSha,inferenceSha,candidateId) {
  requireThat(/^[a-f0-9]{64}$/.test(expectedSha),'immutable bundle inventory SHA256 required');
  const path=resolve(root,'bundle-inventory.json');requireThat((await fileIdentity(path)).sha256===expectedSha,'downloaded campaign bundle inventory differs from dispatch');
  const inventory=await json(path),actual=(await filesUnder(root)).filter(item=>item.file!=='bundle-inventory.json');
  requireThat(canonical(actual)===canonical(inventory.files),'campaign bundle files/bytes differ from immutable inventory');
  const bundle=await json(resolve(root,'campaign.json'));requireThat(bundle.sourceSha===sourceSha&&bundle.inferenceSha===inferenceSha&&bundle.candidateId===candidateId,'bundle differs from exact paired sources/candidate dispatch');
  return bundle;
}
export async function createBundle(campaignPath,inputManifest,out) {
  const campaign=await json(campaignPath),inputs=await json(inputManifest),base=dirname(resolve(campaignPath));
  requireThat(/^[a-f0-9]{40}$/.test(campaign.sourceSha)&&/^[a-f0-9]{40}$/.test(campaign.inferenceSha)&&/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(campaign.candidateId),'bundle needs exact reviewed source/candidate identities');
  const plan=createPlan(inputs);requireThat(canonical(Object.keys(campaign.guardRequests??{}).sort())===canonical(plan.requiredGuardCases.slice().sort()),'bundle must declare every source-owned refusal case');
  requireThat(campaign.approvedBy&&[campaign.approvalEvidenceSha256,campaign.phase18ReviewSha256,campaign.inferencePreMergeReviewSha256].every(value=>/^[a-f0-9]{64}$/.test(value??'')),'bundle needs actual approval, integrated requirements review and inference pre-merge source review provenance');
  requireThat(Number.isFinite(campaign.cacheTolerance)&&campaign.cacheTolerance>=0&&campaign.cacheTolerance<=1,'bundle needs its predeclared cache comparison tolerance');
  await mkdir(out,{mode:0o700});
  const copy=async(source,file,hash)=>{requireThat((await fileIdentity(source)).sha256===hash,'bundle source bytes differ from declared SHA256');const target=safePath(out,file);await mkdir(dirname(target),{recursive:true,mode:0o700});await cp(source,target,{errorOnExist:true,force:false,dereference:true});await chmod(target,0o600);};
  for(const item of inputs.assets) {
    await copy(resolve(dirname(resolve(inputManifest)),item.path),`inputs/${item.id}.png`,item.sha256);item.path=`inputs/${item.id}.png`;
    if(item.source_receipt){await copy(resolve(dirname(resolve(inputManifest)),item.source_receipt),`inputs/${item.id}.source-receipt.json`,item.receipt_sha256);item.source_receipt=`inputs/${item.id}.source-receipt.json`;}
  }
  const copied=new Map();
  for(const item of [campaign.numericalCarryForward,campaign.historicalNumericalCarryForward,campaign.sourceContinuityAudit,campaign.stoppedCuda?.inventory,...(campaign.guardFixtures??[]).flatMap(fixture=>[fixture,fixture.corruptWith].filter(Boolean))].filter(Boolean)) {
    requireThat(item?.file&&/^[a-f0-9]{64}$/.test(item.sha256??''),'bundle requires declared numerical/fixture bytes and SHA256');
    if(copied.has(item.file)){requireThat(copied.get(item.file)===item.sha256,'conflicting fixture SHA256');continue;}
    await copy(safePath(base,item.file),item.file,item.sha256);copied.set(item.file,item.sha256);
  }
  campaign.inputManifest='inputs.json';await save(resolve(out,'inputs.json'),inputs);await save(resolve(out,'campaign.json'),campaign);
  await save(resolve(out,'bundle-inventory.json'),{files:await filesUnder(out)});
  const inventory=await fileIdentity(resolve(out,'bundle-inventory.json'));await validateBundle(out,inventory.sha256,campaign.sourceSha,campaign.inferenceSha,campaign.candidateId);return inventory;
}
export async function inventoryReview(root,sourceSha,inferenceSha,candidateId) {
  requireThat(/^[a-f0-9]{40}$/.test(sourceSha)&&/^[a-f0-9]{40}$/.test(inferenceSha)&&/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(candidateId),'review upload needs exact source/candidate identities');
  const files=await filesUnder(root);requireThat(files.length>0,'review artifact cannot be empty');
  await save(resolve(root,'review-inventory.json'),{sourceSha,inferenceSha,candidateId,files});return fileIdentity(resolve(root,'review-inventory.json'));
}
export async function validateUpload(root,kind,expectedSha,sourceSha,inferenceSha,candidateId) {
  if(kind==='bundle')return validateBundle(root,expectedSha,sourceSha,inferenceSha,candidateId);
  requireThat(kind==='review','unknown MOSS upload kind');
  const file=resolve(root,'review-inventory.json');requireThat((await fileIdentity(file)).sha256===expectedSha,'review inventory differs from dispatch');
  const inventory=await json(file);requireThat(inventory.sourceSha===sourceSha&&inventory.inferenceSha===inferenceSha&&inventory.candidateId===candidateId,'review upload source/candidate differs');
  requireThat(canonical(inventory.files)===canonical((await filesUnder(root)).filter(item=>item.file!=='review-inventory.json')),'review upload files/bytes differ from inventory');return inventory;
}
export async function arrangeBuilds(root,onlyBackend=null) {
  requireThat(onlyBackend===null||['cuda','mlx'].includes(onlyBackend),'single-backend native artifact route is invalid');
  const entries=await readdir(root,{withFileTypes:true});
  for(const backend of (onlyBackend?[onlyBackend]:['mlx','cuda'])) {
    const names=entries.filter(item=>item.isDirectory()&&new RegExp(`^moss-native-${backend}-[0-9]+-[0-9]+$`).test(item.name));
    requireThat(names.length===1,`require exactly one immutable ${backend} build artifact, never merge attempts`);
    const from=resolve(root,names[0].name);requireThat((await json(resolve(from,'inventory.json'))).backend===backend,'downloaded native artifact routing mismatch');
    await rename(from,resolve(root,backend));
  }
}
export async function configure(bundlePath,sourceRoot,inferenceRoot,root,backend,buildRoot) {
  const config=runtimeConfig(await json(bundlePath),{sourceRoot,inferenceRoot,root,backend,buildRoot,bundleRoot:dirname(bundlePath)});
  await mkdir(root,{recursive:true,mode:0o700});await save(resolve(root,'runtime.json'),config);return resolve(root,'runtime.json');
}
export async function browserIdentity(config) {
  process.env.PLAYWRIGHT_BROWSERS_PATH=resolve(config.root,'browser-cache');
  const {chromium}=await import(pathToFileURL(config.playwrightModule).href),packageRoot=dirname(config.playwrightModule);
  const pkg=await json(resolve(packageRoot,'package.json'));requireThat(pkg.version==='1.56.1','terminal browser runtime must use exact Playwright 1.56.1');
  return {version:pkg.version,package:await fileIdentity(resolve(packageRoot,'package.json')),browserManifest:await fileIdentity(resolve(packageRoot,'../playwright-core/browsers.json')),chromium:await fileIdentity(chromium.executablePath())};
}
export async function provisionBrowser(config) {
  // Deliberately outside the source checkout; Chromium runs headless/CPU only.
  const prefix=resolve(config.root,'playwright'),env={...process.env,PLAYWRIGHT_BROWSERS_PATH:resolve(config.root,'browser-cache')};
  await npmCommand(['install','--prefix',prefix,'--no-save','--ignore-scripts','--package-lock=false','playwright@1.56.1'],{cwd:config.sourceRoot,env,log:resolve(config.root,'browser-install.log')});
  await command(process.execPath,[resolve(prefix,'node_modules/playwright/cli.js'),'install','chromium'],{cwd:config.sourceRoot,env,log:resolve(config.root,'chromium-install.log')});
  await save(resolve(config.root,'browser-provision.json'),await browserIdentity(config));
}
export async function verifyBuild(config) {
  const sources=await sourceIdentity(config),paths=buildPaths(config);
  const inventory=validateArtifactInventory(await json(resolve(config.buildRoot,config.backend,'inventory.json')),config.backend,sources);
  for(const name of BACKEND_FILES)requireThat(canonical(await fileIdentity(paths[name]))===canonical(inventory.files[name]),`native ${name} changed before process startup`);
  const toolchain=await json(paths.toolchain);requireThat(canonical(toolchain.browser)===canonical(await browserIdentity(config)),'actual headless browser differs from the source-bound native build inventory');
  if(toolchain.reusedPrivateHf)requireThat(config.backend==='cuda'&&canonical(await fileIdentity(resolve(dirname(paths.toolchain),'private-hf-reuse.json')))===canonical(toolchain.reusedPrivateHf),'source-paired private HF build ownership proof changed');
  const packages=await json(paths.packages);for(const item of packages.files)requireThat(canonical(await fileIdentity(safePath(config.hfHome,item.file)))===canonical({byteLength:item.byteLength,sha256:item.sha256}),'executing owned HF projection differs from frozen native package bytes');
  if(process.platform!=='win32')await chmod(paths.apiBinary,0o700);
  // The binary inventory must be verified before a single native child executes.
  await nativeDevice(config.backend);return inventory;
}
export function resolveGuardAliases(value,bindings) {
  if(typeof value==='string') {
    if(value.startsWith('$fixture:')){const alias=value.slice(9);requireThat(bindings[alias]?.asset?.id,`actual fixture alias ${alias} missing`);return bindings[alias].asset.id;}
    if(value.startsWith('$fixture-sha:')){const alias=value.slice(13);requireThat(bindings[alias]?.asset?.tensor?.sha256,`actual tensor metadata alias ${alias} missing`);return bindings[alias].asset.tensor.sha256;}
    if(value==='$campaignProject')return bindings.campaignProject;
    return value;
  }
  if(Array.isArray(value))return value.map(item=>resolveGuardAliases(item,bindings));
  if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).map(([key,item])=>[key,key==='rawJson'?item.replace(/"\$(fixture:[A-Za-z0-9_-]+|campaignProject)"/g,(_,token)=>JSON.stringify(resolveGuardAliases(`$${token}`,bindings))):resolveGuardAliases(item,bindings)]));
  return value;
}
export async function prepareGuardFixtures(config,api) {
  const root=resolve(config.root,'evidence','guard-fixtures'),bindings={campaignProject:config.projectId};
  for(const fixture of config.guardFixtures??[]) {
    requireThat(/^[A-Za-z0-9_-]+$/.test(fixture.alias)&&!bindings[fixture.alias]&&['campaign','foreign'].includes(fixture.project),'invalid/duplicate owned guard fixture alias');
    let projectId=config.projectId;
    if(fixture.project==='foreign') {
      if(!bindings.foreignProject){const project=await api.request('/api/v1/projects',{method:'POST',body:{name:`MOSS foreign-input refusal ${config.candidateId}`}});requireThat(project.id&&project.id!==config.projectId,'actual second owned project required');bindings.foreignProject=project;}
      projectId=bindings.foreignProject.id;
    }
    const bytes=await readFile(safePath(config.bundleRoot,fixture.file));requireThat(sha(bytes)===fixture.sha256,'guard fixture bytes changed before actual upload');
    const form=new FormData();form.append('file',new Blob([bytes],{type:fixture.mime}),fixture.filename);
    const uploaded=await api.request(`/api/v1/projects/${encodeURIComponent(projectId)}/assets`,{method:'POST',body:form});
    const route=`/api/v1/projects/${encodeURIComponent(projectId)}/assets/${encodeURIComponent(uploaded.id)}`;
    const asset=await api.request(route);requireThat(asset.id===uploaded.id&&asset.projectId===projectId&&asset.type===fixture.type&&asset.file?.path,'actual uploaded fixture/project/type readback mismatch');
    const media=`/api/v1/projects/${encodeURIComponent(projectId)}/files/${asset.file.path.split('/').map(encodeURIComponent).join('/')}`;
    requireThat(sha(await api.request(media,{raw:true}))===fixture.sha256,'actual guard upload/readback bytes changed');
    let actualSha256=fixture.sha256,deleted=null;
    if(fixture.deleteAfterUpload) {
      requireThat(fixture.project==='campaign'&&!fixture.corruptWith&&fixture.cases?.length&&fixture.cases.every(name=>['missing-reference','missing-tensor'].includes(name)),'only fresh owned missing-input fixtures may be deleted');
      const result=await api.request(`${route}/purge?permanent=true`,{method:'DELETE'});
      let absent=false;try{await api.request(route);}catch(error){requireThat(error.status===404,'deleted fixture readback must be an actual HTTP404');absent=true;}
      requireThat(absent,'fresh fixture still exists after permanent deletion');
      deleted={route:`${route}/purge?permanent=true`,result,readbackStatus:404};
    }
    if(fixture.corruptWith) {
      requireThat(fixture.type==='tensor'&&fixture.project==='campaign','only an explicitly owned campaign tensor may be corrupted');
      const corrupt=await readFile(safePath(config.bundleRoot,fixture.corruptWith.file));requireThat(sha(corrupt)===fixture.corruptWith.sha256&&sha(corrupt)!==fixture.sha256,'actual corrupt tensor bytes/SHA required');
      const registry=await json(resolve(config.stateRoot,'data','recent-projects.json')),project=registry.find(item=>item.id===projectId);
      requireThat(project,'owned actual project registry missing');const owned=contained(resolve(config.stateRoot,'data'),project.path);
      await writeFile(safePath(owned,asset.file.path),corrupt,{mode:0o600});actualSha256=sha(corrupt);
      requireThat(sha(await api.request(media,{raw:true}))===actualSha256,'corrupt owned tensor bytes not actually served');
    }
    const readback=deleted?asset:await api.request(route);requireThat(readback.id===asset.id&&readback.projectId===projectId,'fixture identity changed after setup');
    const uploadedFile=`guard-fixtures/${fixture.alias}-uploaded.bin`,actualFile=`guard-fixtures/${fixture.alias}-actual.bin`;
    await mkdir(root,{recursive:true,mode:0o700});await writeFile(resolve(root,`${fixture.alias}-uploaded.bin`),bytes,{flag:'wx',mode:0o600});
    await writeFile(resolve(root,`${fixture.alias}-actual.bin`),deleted?bytes:await api.request(media,{raw:true}),{flag:'wx',mode:0o600});
    bindings[fixture.alias]={projectId,asset:readback,uploadedSha256:fixture.sha256,actualSha256,uploadedFile,actualFile,cases:fixture.cases??[],...(deleted?{deleted}:{} )};
    await save(resolve(root,`${fixture.alias}.json`),bindings[fixture.alias]);
  }
  await save(resolve(root,'bindings.json'),bindings);
  const resolved=resolveGuardAliases(config.guardRequests,bindings);
  for(const name of ['missing-reference','missing-tensor','foreign-reference','foreign-tensor','corrupt-tensor','embeddings-nonfinite','mask-nonbinary'])if(Object.hasOwn(resolved,name)) {
    const fixture=(config.guardFixtures??[]).find(item=>item.cases?.includes(name)),binding=fixture&&bindings[fixture.alias];
    requireThat(binding&&canonical(resolved[name].request).includes(JSON.stringify(binding.asset.id)),`${name} requires its actual uploaded/read-back fixture ID`);
    requireThat(fixture.project===(name.startsWith('foreign-')?'foreign':'campaign')&&fixture.type===(name.endsWith('-reference')?'image':'tensor'),`${name} fixture has wrong real project/type`);
    if(name.startsWith('missing-'))requireThat(binding.deleted?.readbackStatus===404,`${name} requires actual deletion and HTTP404 readback`);
    else if(!name.startsWith('foreign-'))requireThat(fixture.corruptWith&&binding.actualSha256!==binding.uploadedSha256,`${name} requires actual owned corrupt/nonfinite/nonbinary bytes`);
    if(name.endsWith('-reference'))requireThat(!resolved[name].refs?.length,'missing/foreign reference guard cannot overwrite its actual fixture ID with ordinary references');
  }
  await save(resolve(config.root,'evidence','resolved-guards.json'),resolved);return resolved;
}
export async function prepareHostGuard(config) {
  const root=resolve(config.root,'evidence'),specification=hostGuardSpec();
  await save(resolve(root,specification.invariant),await hostTensorInvariant(config.sourceRoot));
  await mkdir(resolve(root,'guard-fixtures'),{recursive:true,mode:0o700});await writeFile(safePath(root,specification.upload.file),hostBudgetHeader(),{flag:'wx',mode:0o600});
  config.guardRequests={...config.guardRequests,'host-budget-refusal':specification};
}
export async function adapterRefusalState(config,name,specification) {
  if(!['missing-adapter','corrupt-adapter'].includes(name))return {observation:null,restore:async()=>{}};
  requireThat(config.editAdapterFiles?.includes(specification.adapterFile),'adapter refusal must name an exact owned edit-package file');
  let path=safePath(config.hfHome,specification.adapterFile);
  const expected=config.packageFiles.find(item=>item.file===specification.adapterFile);
  requireThat(config.hfHome===resolve(config.root,'hf'),'adapter refusal requires the owned HF projection');
  const home=await realpath(config.hfHome),parent=await realpath(dirname(path));
  requireThat(home===resolve(await realpath(config.root),'hf'),'adapter refusal requires a private HF directory');
  if(parent!==home)contained(home,parent);
  path=resolve(parent,basename(path));
  const original=await lstat(path);
  requireThat(original.isFile()&&!original.isSymbolicLink()&&original.nlink===1,'adapter refusal requires a private regular file');
  requireThat(expected?.sha256===(await fileIdentity(path)).sha256,'adapter refusal initial bytes differ from frozen package inventory');
  const bytes=name==='corrupt-adapter'?Buffer.from(specification.corruptBytesBase64??'','base64'):null;
  if(bytes)requireThat(bytes.length>0&&sha(bytes)===specification.corruptSha256&&sha(bytes)!==expected.sha256,'actual immutable corrupt adapter bytes required');
  const backup=await mkdtemp(resolve(parent,'.moss-terminal-original-')),saved=resolve(backup,'original');
  let moved=false,restored=false;
  const verifySaved=async()=>{
    const info=await lstat(saved);
    requireThat(info.isFile()&&info.nlink===1&&info.dev===original.dev&&info.ino===original.ino,'owned adapter backup was replaced');
    requireThat((await fileIdentity(saved)).sha256===expected.sha256,'owned adapter backup changed');
  };
  const restore=async()=>{
    if(restored)return;
    await verifySaved();
    // Unlink the negative fixture itself; never follow a replacement link.
    await unlink(path).catch(error=>{if(error.code!=='ENOENT')throw error;});
    await rename(saved,path);moved=false;
    requireThat((await fileIdentity(path)).sha256===expected.sha256,'owned adapter fixture failed to restore exact bytes');
    await rmdir(backup);restored=true;
  };
  try {
    await rename(path,saved);moved=true;
    await verifySaved();
    if(bytes)await writeFile(path,bytes,{flag:'wx',mode:0o600});
  }catch(error){if(moved)await restore();else await rmdir(backup);throw error;}
  return {observation:{candidateId:config.candidateId,file:specification.adapterFile,beforeSha256:expected.sha256,state:name,actualSha256:bytes?sha(bytes):null},restore};
}
export async function buildBackend(config) {
  const sources=await sourceIdentity(config),out=resolve(config.buildRoot,config.backend);await mkdir(out,{recursive:true,mode:0o700});
  validatePackagePlan(config,JSON.parse(stripJsoncComments(await readFile(resolve(config.sourceRoot,'config/manifests/builtin.models.jsonc'),'utf8'))));
  await provisionBrowser(config);
  const env={...process.env,CARGO_TARGET_DIR:cargoTarget(config),VITE_API_BASE_URL:'',CARGO_PROFILE_RELEASE_STRIP:'none'};
  const logRoot=resolve(out,'logs');
  await npmCommand(['--prefix','apps/web','ci'],{cwd:config.sourceRoot,env,log:resolve(logRoot,'web-ci.log')});
  await npmCommand(['--prefix','apps/web','run','build'],{cwd:config.sourceRoot,env,log:resolve(logRoot,'web-build.log')});
  const features=config.backend==='cuda'?'embed-web,backend-candle':'embed-web';
  await command('cargo',['build','--release','--locked','-p','sceneworks-rust-api','--bin','sceneworks-rust-api','--features',features],{cwd:config.sourceRoot,env,log:resolve(logRoot,'native-build.log')});
  requireThat(canonical(await sourceIdentity(config))===canonical(sources),'native build changed the frozen source checkouts');
  const binary=resolve(out,config.backend==='cuda'?'sceneworks-rust-api.exe':'sceneworks-rust-api');
  await cp(resolve(env.CARGO_TARGET_DIR,'release',config.backend==='cuda'?'sceneworks-rust-api.exe':'sceneworks-rust-api'),binary,{errorOnExist:true,force:false});
  await save(resolve(out,'web.json'),{files:await filesUnder(resolve(config.sourceRoot,'apps/web/dist'))});
  if(config.reusedPrivateHf) {
    const {artifactRoot,old,continuity,artifactIdentity}=config.reusedPrivateHf;
    await adoptStoppedCudaBuildPrivateHf(config,artifactRoot,old,continuity,artifactIdentity);
    const receipt=await json(resolve(config.root,'cuda-rebase-hf-transfer.json'));
    requireThat(config.backend==='cuda'&&receipt.to===config.hfHome&&canonical(receipt.after.map(({file,sha256})=>({file,sha256})))===canonical(config.packageFiles.map(({file,sha256})=>({file,sha256}))),'corrected CUDA build lost authenticated installed package ownership');
  } else await isolatePackages(config,{omitAdapter:config.installAdapter===true});
  let installation=null;
  if(config.reusedPrivateHf) {
    const from=resolve(config.reusedPrivateHf.artifactRoot,'evidence/external/raw-install-cuda'),to=resolve(out,'installation');
    await cp(from,to,{recursive:true,errorOnExist:true,force:false});
    installation={mode:'authenticated-stopped-CUDA-install',oldArtifact:config.reusedPrivateHf.artifactIdentity,capture:await fileIdentity(resolve(from,'capture.json')),transfer:await fileIdentity(resolve(config.root,'cuda-rebase-hf-transfer.json')),files:await filesUnder(to)};
    requireThat(installation.files.length>0&&canonical(installation.files)===canonical(await filesUnder(from)),'stopped source-bound installer evidence changed during new build');
  }
  if(config.installAdapter===true&&!config.reusedPrivateHf) {
    requireThat(config.editAdapterFiles?.length&&config.editAdapterFiles.every(file=>config.packageFiles.some(item=>item.file===file)),'actual isolated adapter-install projection required');
    config.apiBinary=binary;const token=randomBytes(32).toString('hex'),stack=await startStack(config,token);
    try {
      const project=await stack.api.request('/api/v1/projects',{method:'POST',body:{name:`MOSS installer ${config.candidateId}`}});
      const {capturePackageInstallUi}=await import('./moss-edit-terminal-browser.mjs');
      const capture=await capturePackageInstallUi({apiUrl:`http://127.0.0.1:${config.port}`,projectId:project.id,captureRoot:resolve(out,'installation'),playwrightModule:config.playwrightModule,token,model:'moss_image_base',utilityWorkerId:`${config.workerId}-utility`});
      installation={capture:await fileIdentity(capture),files:await filesUnder(resolve(out,'installation'))};
    }finally{await stack.stop();}
  }
  if(!config.reusedPrivateHf)await materializePrivateAdapter(config);
  const packages=[];
  const reused=config.reusedPrivateHf?(await json(resolve(config.root,'cuda-rebase-hf-transfer.json'))).after:null;
  for(const [index,item] of config.packageFiles.entries()) {
    const identity=reused?reused[index]:await fileIdentity(contained(config.hfHome,resolve(config.hfHome,item.file)));
    requireThat(identity.file===undefined||identity.file===item.file,'reused private package order changed');
    requireThat(identity.sha256===item.sha256,'native package bytes differ from frozen model/package identities');packages.push({...item,byteLength:identity.byteLength,sha256:identity.sha256});
  }
  await save(resolve(out,'packages.json'),{hfHome:config.hfHome,files:packages});
  await sealPrivateHf(config);
  if(config.reusedPrivateHf)await cp(resolve(config.root,'cuda-rebase-hf-transfer.json'),resolve(out,'private-hf-reuse.json'),{errorOnExist:true,force:false});
  const toolchain={node:(await execFile(process.execPath,['--version'])).stdout,rust:(await execFile('rustc',['-Vv'])).stdout,
    observationProducerSha256:(await fileIdentity(resolve(config.sourceRoot,'crates/sceneworks-worker/src/moss_terminal_observation.rs'))).sha256,
    installation,...(config.reusedPrivateHf?{reusedPrivateHf:await fileIdentity(resolve(out,'private-hf-reuse.json'))}:{}),browser:await browserIdentity(config),features,sources,buildCommand:['cargo','build','--release','--locked','-p','sceneworks-rust-api','--features',features]};
  await save(resolve(out,'toolchain.json'),toolchain);
  const paths={apiBinary:binary,workerBinary:binary,webPackage:resolve(out,'web.json'),packages:resolve(out,'packages.json'),toolchain:resolve(out,'toolchain.json')},files={};
  for(const name of BACKEND_FILES)files[name]=await fileIdentity(paths[name]);
  await save(resolve(out,'inventory.json'),{backend:config.backend,sources,files});
  await save(resolve(out,'build.json'),{version:VERSION,backend:config.backend,sources,files,paths,device:await nativeDevice(config.backend)});
}
export async function buildCudaRebase(config,artifactRoot,runId,attempt,artifactId) {
  const old=config.stoppedCuda;
  requireThat(config.backend==='cuda'&&old?.candidateId&&old?.sourceSha&&old?.inferenceSha&&old?.runId===runId&&old?.attempt===attempt&&old?.artifactId===artifactId&&old?.originalRunId>0&&old?.originalAttempt>0&&old.inventory?.file&&/^[a-f0-9]{64}$/.test(old.inventory.sha256??''),'corrected CUDA build requires exact stopped owner and archived artifact inventory');
  const manifestPath=safePath(config.bundleRoot,old.inventory.file);
  requireThat((await fileIdentity(manifestPath)).sha256===old.inventory.sha256,'stopped CUDA artifact inventory changed');
  const manifest=await json(manifestPath);
  requireThat(manifest.version===1&&manifest.actualRunId===old.runId&&manifest.attempt===old.attempt&&manifest.artifactId===old.artifactId&&manifest.appendOnlyTails?.length===0&&new Set(manifest.files.map(item=>item.file)).size===manifest.files.length,'stopped CUDA artifact ID or exact file inventory changed');
  const actual=await filesUnder(artifactRoot),byFile=items=>items.slice().sort((a,b)=>a.file.localeCompare(b.file));
  requireThat(canonical(byFile(actual))===canonical(byFile(manifest.files)),'stopped CUDA artifact files/bytes changed');
  const continuity=cudaSourceContinuity({sceneworksRoot:config.sourceRoot,inferenceRoot:config.inferenceRoot,oldSourceSha:old.sourceSha,oldInferenceSha:old.inferenceSha,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,auditPath:safePath(config.bundleRoot,config.sourceContinuityAudit.file),auditSha256:config.sourceContinuityAudit.sha256});
  config.reusedPrivateHf={artifactRoot,old,continuity,artifactIdentity:{sha256:old.inventory.sha256,runId:old.runId,attempt:old.attempt,artifactId:old.artifactId}};
  return buildBackend(config);
}
export async function rehydrateCudaPackages(config) {
  requireThat(config.backend==='cuda'&&config.root===resolve(config.sourceRoot,'../..','moss-sc-26752',basename(config.root))&&/^moss-build-[0-9]+-[0-9]+-cuda$/.test(basename(config.root)),'CUDA rehydration requires a fresh exact owned build root');
  const sources=await sourceIdentity(config),paths=buildPaths(config),inventory=validateArtifactInventory(await json(resolve(config.buildRoot,'cuda','inventory.json')),'cuda',sources);
  for(const name of BACKEND_FILES)requireThat(canonical(await fileIdentity(paths[name]))===canonical(inventory.files[name]),`original H17 CUDA ${name} artifact changed`);
  validatePackagePlan(config,JSON.parse(stripJsoncComments(await readFile(resolve(config.sourceRoot,'config/manifests/builtin.models.jsonc'),'utf8'))));
  await provisionBrowser(config);
  await isolatePackages(config,{omitAdapter:true});
  config.apiBinary=paths.apiBinary;const token=randomBytes(32).toString('hex'),stack=await startStack(config,token);
  try {
    const project=await stack.api.request('/api/v1/projects',{method:'POST',body:{name:`MOSS continuation package install ${config.candidateId}`}});
    const {capturePackageInstallUi}=await import(pathToFileURL(resolve(config.sourceRoot,'scripts/moss-edit-terminal-browser.mjs')).href);
    await capturePackageInstallUi({apiUrl:`http://127.0.0.1:${config.port}`,projectId:project.id,captureRoot:resolve(config.root,'installation'),playwrightModule:config.playwrightModule,token,model:'moss_image_base',utilityWorkerId:`${config.workerId}-utility`});
  }finally{await stack.stop();}
  await materializePrivateAdapter(config);
  const packages=[];for(const item of config.packageFiles) {
    const actual=await fileIdentity(safePath(config.hfHome,item.file));requireThat(actual.sha256===item.sha256,'rehydrated package differs from frozen H17 weights');packages.push({...item,...actual});
  }
  await save(resolve(config.root,'packages.json'),{hfHome:config.hfHome,files:packages});
  await sealPrivateHf(config);
  await save(resolve(config.root,'rehydration.json'),{version:VERSION,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,candidateId:config.candidateId,originalNativeInventory:inventory,installation:await filesUnder(resolve(config.root,'installation')),packages:await fileIdentity(resolve(config.root,'packages.json'))});
}
export async function prepareContinuationCudaPackages(config,rehydrateConfig,originalRunId,originalAttempt,stopped=null) {
  requireThat(config.backend==='cuda'&&rehydrateConfig.backend==='cuda'&&config.candidateId===rehydrateConfig.candidateId&&config.sourceSha===rehydrateConfig.sourceSha&&config.inferenceSha===rehydrateConfig.inferenceSha&&config.buildRoot===rehydrateConfig.buildRoot,'CUDA package preparation differs from the exact reviewed candidate/artifact');
  const inventory=validateArtifactInventory(await json(resolve(config.buildRoot,'cuda','inventory.json')),'cuda',await sourceIdentity(config));
  const installedPath=resolve(config.buildRoot,'cuda','packages.json');requireThat(canonical(await fileIdentity(installedPath))===canonical(inventory.files.packages),'original native CUDA installed-package inventory changed');
  const installed=await json(installedPath),from=installed.hfHome;
  let mode;
  if(stopped)mode='stopped-execution-HF';
  else try {await lstat(from);mode='original-sealed-HF';}
  catch(error){if(error.code!=='ENOENT')throw error;mode='rehydrated-exact-packages';}
  if(mode==='stopped-execution-HF') {
    await adoptStoppedExecutionPrivateHf(config,stopped.artifactRoot,inventory.files.packages,originalRunId,originalAttempt,stopped.runId,stopped.attempt,stopped.artifactId);
    await retainStoppedCudaSource(config,stopped.artifactRoot);
  } else if(mode==='original-sealed-HF')await adoptOriginalPrivateHf(config,installed,inventory.files.packages,originalRunId,originalAttempt);
  else {
    await rehydrateCudaPackages(rehydrateConfig);
    await transferPrivateHf(config,await json(resolve(rehydrateConfig.root,'packages.json')),inventory.files.packages);
    await mkdir(resolve(config.root,'rehydration'),{mode:0o700});
    await cp(resolve(rehydrateConfig.root,'rehydration.json'),resolve(config.root,'rehydration/rehydration.json'),{errorOnExist:true,force:false});
    await cp(resolve(rehydrateConfig.root,'installation'),resolve(config.root,'rehydration/installation'),{recursive:true,errorOnExist:true,force:false});
  }
  await save(resolve(config.root,'cuda-package-preparation.json'),{version:VERSION,mode,originalRunId,originalAttempt,candidateId:config.candidateId,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,originalNativePackages:inventory.files.packages,transfer:await fileIdentity(resolve(config.root,'hf-transfer.json')),...(stopped?{stoppedArtifact:{runId:stopped.runId,attempt:stopped.attempt,artifactId:stopped.artifactId}}:{})});
}
export async function retainStoppedCudaSource(config,artifactRoot) {
  // Evidence belongs to the create-only Mac handoff import later in executeBackend.
  for(const name of ['runtime.json','hf-transfer.json','cuda-package-preparation.json'])await cp(resolve(artifactRoot,name),resolve(config.root,`stopped-cuda-${name}`),{errorOnExist:true,force:false});
}
export async function importCompletedMlxEvidence(config) {
  await cp(config.priorEvidenceRoot,resolve(config.root,'evidence'),{recursive:true,errorOnExist:true,force:false});
}
export async function completedMlxHandoff(config,originalRunId,originalAttempt,runId,attempt,artifactId,mlxToolingSha,cudaToolingSha,toolingRoot) {
  requireThat(config.backend==='cuda'&&treeIdentity(toolingRoot).head===cudaToolingSha&&cudaToolingSha!==mlxToolingSha,'completed Mac/CUDA tooling identities must be separate clean commits');
  await execFile('git',['-C',toolingRoot,'merge-base','--is-ancestor',mlxToolingSha,cudaToolingSha]);
  await sourceIdentity(config);
  const record=await validateCompletedMlxArtifact(resolve(config.root,'prior'),{candidateId:config.candidateId,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,originalRunId,originalAttempt,toolingSha:mlxToolingSha,runId,attempt,artifactId});
  await validateCompletedMlxHandoffSupplement(resolve(config.root,'prior'),{...record,cudaToolingSha});
  await save(resolve(config.root,'completed-mlx-artifact.json'),{...record,cudaToolingSha,supplementSha256:(await fileIdentity(resolve(config.root,'prior/supplement.json'))).sha256});
  return record;
}
export async function completedCudaRebaseHandoff(config,old,toolingRoot,toolingSha) {
  requireThat(config.backend==='cuda'&&treeIdentity(toolingRoot).head===toolingSha&&toolingSha===config.sourceSha,'corrected CUDA controller must be the new stationary product source');
  requireThat(config.sourceContinuityAudit?.file&&/^[a-f0-9]{64}$/.test(config.sourceContinuityAudit.sha256??''),'new paired bundle lacks the reviewed Mac source closure audit');
  const auditPath=safePath(config.bundleRoot,config.sourceContinuityAudit.file),auditSha256=config.sourceContinuityAudit.sha256;
  const continuity=cudaSourceContinuity({sceneworksRoot:config.sourceRoot,inferenceRoot:config.inferenceRoot,oldSourceSha:old.sourceSha,oldInferenceSha:old.inferenceSha,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,auditPath,auditSha256});
  const mac=await validateCompletedMlxArtifact(resolve(config.root,'prior'),{candidateId:old.candidateId,sourceSha:old.sourceSha,inferenceSha:old.inferenceSha,originalRunId:old.originalRunId,originalAttempt:old.originalAttempt,toolingSha:old.toolingSha,runId:old.runId,attempt:old.attempt,artifactId:old.artifactId});
  await validateCompletedMlxHandoffSupplement(resolve(config.root,'prior'),{...mac,cudaToolingSha:toolingSha});
  const handoff=await json(resolve(config.handoffRoot,'handoff.json'));
  const record={name:'sc-26757-corrected-cuda-candidate/v1',old:{...mac,sources:handoff.sources},current:{candidateId:config.candidateId,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,toolingSha},continuity};
  validateCudaRebaseRecord(record,mac,config);
  await save(resolve(config.root,'cuda-rebase.json'),record);
  return record;
}
export async function historicalMlxLaunchForCudaRebase(config,macRoot,oldSourceRoot,oldInferenceRoot,oldMlxBuildRoot,toolingRoot) {
  const record=await json(resolve(config.root,'cuda-rebase.json')),old=record.old;
  requireThat(treeIdentity(toolingRoot).head===config.sourceSha&&treeIdentity(oldSourceRoot).head===old.sourceSha&&treeIdentity(oldInferenceRoot).head===old.inferenceSha,'historical verification source checkouts changed');
  const authenticated=await validateCompletedMlxArtifact(macRoot,{candidateId:old.candidateId,sourceSha:old.sourceSha,inferenceSha:old.inferenceSha,originalRunId:old.originalRunId,originalAttempt:old.originalAttempt,toolingSha:old.toolingSha,runId:old.runId,attempt:old.attempt,artifactId:old.artifactId});
  requireThat(canonical(old)===canonical({...authenticated,sources:(await json(resolve(macRoot,'handoff/handoff.json'))).sources}),'historical Mac artifact changed before full verification');
  await validateCompletedMlxHandoffSupplement(macRoot,{...authenticated,cudaToolingSha:config.sourceSha});
  const launch=await json(resolve(macRoot,'continuation-retry-8-launch.json')),evidence=resolve(macRoot,'evidence');
  requireThat(launch.candidateId===old.candidateId&&launch.localBackend==='mlx'&&launch.repair?.retryOrdinal===8&&launch.repair.postGuardDmd&&launch.repair.postRowsGuard,'historical Mac launch is not the completed H27 producer');
  launch.historicalEvidenceOnly=true;launch.plan=resolve(evidence,'cases.json');launch.evidenceRoot=evidence;
  launch.historicalQualityReviewRoot=resolve(config.root,'evidence');
  launch.sources={sceneworks:oldSourceRoot,inference:oldInferenceRoot};
  const names={harness:'moss-edit-terminal.mjs',harnessLibrary:'lib/moss-edit-terminal.mjs',actionController:'moss-edit-terminal-actions.mjs',actionLibrary:'lib/moss-edit-terminal-actions.mjs',observationValidator:'lib/moss-edit-terminal-observation.mjs',browserDriver:'moss-edit-terminal-browser.mjs',browserControls:'lib/moss-edit-terminal-controls.mjs'};
  for(const [name,path] of Object.entries(names))launch.files[name]=resolve(oldSourceRoot,'scripts',path);
  for(const name of ['apiBinary','workerBinary','webPackage','packages','toolchain'])launch.files[name]=resolve(oldMlxBuildRoot,basename(launch.files[name]));
  launch.files.fixtureBindings=resolve(evidence,'guard-fixtures/bindings.json');launch.files.hostTensorBound=resolve(evidence,'host-tensor-bound.json');
  requireThat(config.historicalNumericalCarryForward?.sha256==='bcd3309fab103cb93063adf565aafe6858695e5845e27000fb7c6c1ed26718cf','historical numerical source proof missing from new sealed bundle');
  launch.files.numericalCarryForward=safePath(config.bundleRoot,config.historicalNumericalCarryForward.file);
  const currentNames={harness:'moss-edit-terminal.mjs',harnessLibrary:'lib/moss-edit-terminal.mjs',actionController:'moss-edit-terminal-actions.mjs',actionLibrary:'lib/moss-edit-terminal-actions.mjs',browserDriver:'moss-edit-terminal-browser.mjs',browserControls:'lib/moss-edit-terminal-controls.mjs',qualityAmendment:'lib/moss-edit-quality-amendment.mjs'},files=Object.fromEntries(Object.entries(currentNames).map(([name,path])=>[name,resolve(toolingRoot,'scripts',path)]));
  launch.repair={...launch.repair,originalRoot:macRoot,tooling:{sourceRoot:toolingRoot,head:config.sourceSha,source:treeIdentity(toolingRoot),files,identities:Object.fromEntries(await Promise.all(Object.entries(files).map(async([name,path])=>[name,await fileIdentity(path)])))},historicalExecutionToolingSha:'2cd0363d59cb3eb0989bbb708000b6ffc907932a',executionToolingSha:old.toolingSha,verificationToolingSha:config.sourceSha};
  const path=resolve(config.root,'historical-mlx-launch.json');await save(path,launch);return path;
}
async function harness(config,args,label) {
  const prefix=config.continuation?.retry?`continuation-retry${config.continuation.retry.ordinal>1?`-${config.continuation.retry.ordinal}`:''}-`:config.toolingRoot||config.continuation?'continuation-':'';
  await command(process.execPath,[resolve(config.toolingRoot??config.continuation?.toolingRoot??config.sourceRoot,'scripts/moss-edit-terminal.mjs'),...args],{cwd:config.sourceRoot,env:process.env,log:resolve(config.root,'commands',`${prefix}${label}.log`)});
}
export function compareCrossBackendHistoryJob(origin,actual) {
  const jobIdentityEqual=actual.id===origin.id&&actual.projectId===origin.projectId&&actual.type===origin.type&&actual.status===origin.status;
  const payloadEqual=canonical(actual.payload)===canonical(origin.payload);
  const comparison=compareTypedMossEditHistory(origin.result,actual.result,{kind:'result',steps:origin.payload?.advanced?.steps});
  return {jobIdentityEqual,payloadEqual,resultEqual:comparison.rawEqual,sigmaF32Equivalent:comparison.sigmaF32Equivalent,sigmaDifferingIndices:comparison.sigmaDifferingIndices,normalizedResultEqual:comparison.normalizedEqual,typedResultEqual:comparison.typedEqual};
}
export async function authenticateCrossBackendHistory(api,evidenceRoot,readbackRoot=evidenceRoot) {
  const comparisons=[];
  for(const rowId of ['quality-mlx-base-four','quality-mlx-sft-single']) {
    const path=resolve(evidenceRoot,'rows',rowId,'terminal-job.json'),origin=await json(path);
    const actual=await api.request(`/api/v1/jobs/${encodeURIComponent(origin.id)}`);
    comparisons.push({rowId,originFile:await fileIdentity(path),actual,...compareCrossBackendHistoryJob(origin,actual)});
  }
  await save(resolve(readbackRoot,'external/cross-backend-history-readback.json'),{version:VERSION,name:'cross-backend-history-before-first-cuda-dispatch',comparisons});
  requireThat(comparisons.every(item=>item.jobIdentityEqual&&item.payloadEqual&&item.typedResultEqual),'cross-backend history handoff changed original authored job/result identities');
  return comparisons;
}
export async function relocateImportedCudaModelLibrary(api,config) {
  requireThat(config.backend==='cuda'&&config.completedMlx&&config.hfHome===resolve(config.root,'hf'),'only the authenticated cross-backend CUDA handoff may rebind its imported model library');
  const relative='models/.sceneworks-external-library-state.json';
  const imported=resolve(config.handoffRoot,'data',relative),live=resolve(config.stateRoot,'data',relative),hub=resolve(config.hfHome,'hub');
  const [oldIdentity,liveIdentity,oldBinding]=await Promise.all([fileIdentity(imported),fileIdentity(live),json(imported)]);
  requireThat(canonical(oldIdentity)===canonical(liveIdentity)&&oldBinding.configuredPath===oldBinding.canonicalPath&&oldBinding.configuredPath!==hub&&oldBinding.physicalIdentity,'imported Mac model-library binding or original handoff bytes changed');
  const before=await api.request('/api/v1/model-library');
  requireThat(before.available===false&&before.configuredLibraryPath===hub&&canonical(before.expectedLibrary)===canonical(oldBinding),'transferred API did not expose the exact unavailable imported Mac library and new private CUDA cache');
  const dryRun=await api.request('/api/v1/model-library/relocate',{method:'POST',body:{path:config.hfHome,dryRun:true}});
  requireThat(dryRun.adopted===false&&dryRun.hfHome===config.hfHome&&dryRun.libraryRoot===hub&&canonical(await fileIdentity(live))===canonical(liveIdentity),'product relocation dry run changed state or resolved another package root');
  const adoption=await api.request('/api/v1/model-library/relocate',{method:'POST',body:{path:config.hfHome}});
  const [after,newBinding,newIdentity]=await Promise.all([api.request('/api/v1/model-library'),json(live),fileIdentity(live)]);
  requireThat(adoption.adopted===true&&adoption.hfHome===config.hfHome&&adoption.libraryRoot===hub&&after.available===true&&after.probeStatus==='available'&&after.configuredLibraryPath===hub&&canonical(after.expectedLibrary)===canonical(newBinding)&&newBinding.configuredPath===hub&&newBinding.physicalIdentity&&canonical(newIdentity)!==canonical(oldIdentity),'product did not prove the same complete private CUDA model library is available after adoption');
  const receipt={version:VERSION,name:'cross-backend-model-library-adoption-before-first-cuda-dispatch',importedFile:oldIdentity,oldBinding,configuredHfHome:config.hfHome,before,dryRun,adoption,after,adoptedFile:newIdentity,newBinding};
  await save(resolve(config.root,'evidence/external/cuda-model-library-relocation.json'),receipt);
  return receipt;
}
function buildPaths(config) {
  const out=resolve(config.buildRoot,config.backend),binary=resolve(out,config.backend==='cuda'?'sceneworks-rust-api.exe':'sceneworks-rust-api');
  return {apiBinary:binary,workerBinary:binary,webPackage:resolve(out,'web.json'),packages:resolve(out,'packages.json'),toolchain:resolve(out,'toolchain.json')};
}
async function launcher(config,projectId) {
  const root=resolve(config.root,'evidence'),paths=buildPaths(config),files={...paths};
  for(const [name,path] of Object.entries({harness:'moss-edit-terminal.mjs',harnessLibrary:'lib/moss-edit-terminal.mjs',actionController:'moss-edit-terminal-actions.mjs',actionLibrary:'lib/moss-edit-terminal-actions.mjs',observationValidator:'lib/moss-edit-terminal-observation.mjs',browserDriver:'moss-edit-terminal-browser.mjs',browserControls:'lib/moss-edit-terminal-controls.mjs'}))files[name]=resolve(config.sourceRoot,'scripts',path);
  files.numericalCarryForward=safePath(config.bundleRoot,config.numericalCarryForward.file);
  files.fixtureBindings=resolve(root,'guard-fixtures/bindings.json');
  files.hostTensorBound=resolve(root,'host-tensor-bound.json');
  requireThat((await fileIdentity(files.numericalCarryForward)).sha256===config.numericalCarryForward.sha256,'numerical source carry-forward artifact changed');
  return {version:VERSION,candidateId:config.candidateId,localBackend:config.backend,plan:resolve(root,'cases.json'),evidenceRoot:root,
    sources:{sceneworks:config.sourceRoot,inference:config.inferenceRoot},inferenceMainRef:'origin/main',files,
    artifactInventories:Object.fromEntries(['mlx','cuda'].map(backend=>[backend,resolve(config.buildRoot,backend,'inventory.json')])),
    backends:{[config.backend]:{apiUrl:`http://127.0.0.1:${config.port}`,projectId,workerId:config.workerId,workerGpuId:config.backend==='cuda'?'1':'mlx',tokenEnv:'MOSS_EDIT_ACTIONS_TOKEN'}},
    guardRequests:config.guardRequests,approvedBy:config.approvedBy,approvalEvidenceSha256:config.approvalEvidenceSha256,phase18ReviewSha256:config.phase18ReviewSha256,inferencePreMergeReviewSha256:config.inferencePreMergeReviewSha256};
}
async function actualJournal(config,expected,after) {
  const deadline=Date.now()+10000;let lastError;
  while(Date.now()<deadline) {
    try {const bytes=await readFile(config.observationJournal),snapshot=inspectJournal(bytes,expected);
      if((!after||Date.parse(snapshot.lastAt)>=Date.parse(after))&&Date.now()-Date.parse(snapshot.lastAt)<=3000)return bytes;
    }catch(error){lastError=error;}await delay(250);
  }throw Error(`actual instrumented worker journal unavailable/stale: ${lastError?.message??'no fresh idle heartbeat'}`);
}
async function deviceReceipt(config,state,label) {
  const actual=await nativeDevice(config.backend),receipt={version:VERSION,lockSha256:digest(state.lock),backend:config.backend,device:actual.device,
    ...(config.backend==='cuda'?{cudaVisibleDevices:CUDA_UUID,sceneWorksGpuId:'1',postCompile:true}:{})};
  validateAdmission(receipt,state.lock,config.backend);const path=resolve(config.root,'devices',`${label}.json`);await save(path,receipt);return path;
}
export async function validateContinuation(config) {
  const c=config.continuation;
  requireThat(c?.name==='r16-checker-recovery'&&config.backend==='mlx'&&Number.isSafeInteger(c.originalRunId)&&c.originalRunId>0&&Number.isSafeInteger(c.originalAttempt)&&c.originalAttempt>0,'exact stopped MLX attempt required for checker recovery');
  requireThat(config.root===resolve(MAC_MOSS_BASE,`moss-execute-${c.originalRunId}-${c.originalAttempt}-mlx`)&&config.hfHome===resolve(config.root,'hf')&&config.stateRoot===resolve(config.root,'state'),'continuation may only reopen its exact private original Mac execution root');
  requireThat(c.toolingRoot&&/^[a-f0-9]{40}$/.test(c.toolingSha??'')&&treeIdentity(c.toolingRoot).head===c.toolingSha&&c.toolingSha!==config.sourceSha,'repair tooling must be a separate clean committed checkout');
  const manifestIdentity=await fileIdentity(c.inventoryPath);requireThat(manifestIdentity.sha256===c.inventorySha256&&c.inventorySha256===R16_ORIGINAL_INVENTORY_SHA256,'original evidence boundary inventory differs from exact R16 dispatch');
  const manifest=await json(c.inventoryPath),identity=await sourceIdentity(config),lock=await json(resolve(config.root,'evidence/protocol.lock.json'));
  requireThat(manifest.version===1&&manifest.actualRunId===c.originalRunId&&manifest.attempt===c.originalAttempt&&manifest.sourceSha===config.sourceSha&&manifest.inferenceSha===config.inferenceSha&&manifest.actualEvidenceRoot===resolve(config.root,'evidence')&&Array.isArray(manifest.files)&&manifest.files.length===257,'original R16 evidence boundary/candidate differs');
  requireThat(lock.sources.sceneworks.head===config.sourceSha&&lock.sources.inference.head===config.inferenceSha&&lock.candidateId===config.candidateId&&canonical(lock.sources)===canonical(identity),'original frozen candidate/product pair changed');
  const launch=await json(resolve(config.root,'launch.json')),binding=await json(resolve(config.root,'evidence/bindings/mlx.json')),plan=await json(resolve(config.root,'evidence/cases.json'));
  requireThat(launch.candidateId===config.candidateId&&launch.evidenceRoot===resolve(config.root,'evidence')&&launch.localBackend==='mlx'&&binding.endpoint.projectId&&binding.endpoint.workerId===config.workerId&&binding.endpoint.apiUrl===`http://127.0.0.1:${config.port}`,'original launch/backend endpoint differs');
  const rows=plan.rows.filter(row=>row.backend==='mlx');let remaining,retry=null;
  if(c.retry) {
    const chained=Boolean(c.retry.artifactRoot);
    requireThat(c.toolingSha!==H18_RETRY_TOOLING_SHA&&c.retry.inventoryPath&&/^[a-f0-9]{64}$/.test(c.retry.inventorySha256??'')&&(chained||c.retry.inventorySha256===H18_RETRY_INVENTORY_SHA256),'retry requires a distinct exact tooling revision and sealed prior checkpoint');
    const checkpointIdentity=await fileIdentity(c.retry.inventoryPath),checkpoint=await json(c.retry.inventoryPath);
    requireThat(checkpointIdentity.sha256===c.retry.inventorySha256&&checkpoint.version===1&&checkpoint.productSha===config.sourceSha&&checkpoint.originalRunId===c.originalRunId&&checkpoint.originalAttempt===c.originalAttempt&&checkpoint.actualRoot===config.root&&checkpoint.files?.length>=300,'failed-continuation checkpoint identity changed');
    let ordinal=1,previousToolingSha=H18_RETRY_TOOLING_SHA,capturePath='captures/advanced-mlx-sft-replay/failure.json';
    if(chained) {
      while(await optional(resolve(config.root,'evidence',`continuation-retry-${ordinal+1}.json`)))ordinal++;
      ordinal++;
      const previous=await json(resolve(config.root,'evidence',ordinal===2?'continuation-retry.json':`continuation-retry-${ordinal-1}.json`));
      previousToolingSha=previous.toolingSha;
      requireThat(checkpoint.actualRunId===c.retry.runId&&checkpoint.attempt===1&&checkpoint.artifactId===c.retry.artifactId&&checkpoint.toolingSha===previousToolingSha&&previous.originalInventorySha256===R16_ORIGINAL_INVENTORY_SHA256&&previous.lockSha256===digest(lock)&&previous.sourceSha===config.sourceSha&&previous.inferenceSha===config.inferenceSha&&previous.candidateId===config.candidateId&&previous.checkpointInventorySha256===(ordinal===2?H18_RETRY_INVENTORY_SHA256:(await fileIdentity(resolve(config.root,'evidence',`continuation-prior-${ordinal-1}-inventory.json`))).sha256)&&c.toolingSha!==previousToolingSha,'chained retry does not descend from the exact prior stopped producer');
      await execFile('git',['-C',c.toolingRoot,'merge-base','--is-ancestor',previousToolingSha,c.toolingSha]);
      await assertOriginalEvidence(c.retry.artifactRoot,checkpoint,{exact:true});
      await assertH18CheckpointFiles(config.root,checkpoint,{archiveTag:`prior-${ordinal}`});
    } else {
      requireThat(checkpoint.actualRunId===H18_RETRY_RUN_ID&&checkpoint.attempt===1&&checkpoint.artifactId===H18_RETRY_ARTIFACT_ID&&checkpoint.toolingSha===H18_RETRY_TOOLING_SHA,'H18 failed-continuation artifact identity changed');
      await assertH18CheckpointFiles(config.root,checkpoint);
    }
    await assertOriginalEvidence(resolve(config.root,'evidence'),manifest);
    const historical=await json(resolve(config.root,'continuation-start.json')),adjudication=await json(resolve(config.root,'evidence/recovery-adjudication.json'));
    requireThat(historical.toolingSha===H18_RETRY_TOOLING_SHA&&historical.inventorySha256===R16_ORIGINAL_INVENTORY_SHA256&&historical.lockSha256===digest(lock)&&adjudication.toolingSha===H18_RETRY_TOOLING_SHA&&adjudication.jobId==='job_4ccba4f5dc5d78aa0991ca4df750cb65'&&adjudication.lockSha256===digest(lock),'H18 completed-job recovery/adjudication changed');
    let pending;for(const row of rows)if(!await optional(resolve(config.root,'evidence/rows',row.id,'receipt.json'))){pending=row;break;}
    let capture=null,recovery=null,guardFrontier=null;
    if(chained&&pending?.expected==='canceled'&&await optional(resolve(config.root,'evidence/rows',pending.id,'submission.json'))) {
      const submitted=await submittedCanceledRemainingRows(resolve(config.root,'evidence'),plan,digest(lock),digest(binding),checkpoint,config.root);
      remaining=submitted.remaining;recovery=submitted.recovery;
    } else if(chained&&!pending) {
      const guards=await json(resolve(config.root,'evidence/resolved-guards.json'));
      guardFrontier=checkpoint.files.some(item=>item.file==='evidence/attempts/mlx/guard-edit-dmd.failure.json')
        ?await postDmdGuardFrontier(resolve(config.root,'evidence'),plan,digest(lock),digest(binding),guards)
        :await postRowsGuardFrontier(resolve(config.root,'evidence'),plan,digest(lock),digest(binding),guards['training-shift-nonfinite']);
      requireThat(!checkpoint.files.some(item=>item.file==='evidence/attempts/mlx/dispatch.lock'),'post-row failed guard retained an active submission mutex');
      remaining=[];
    } else {
      if(chained)capturePath=`captures/${pending?.id}${ordinal===2?'-retry':`-retry-${ordinal-1}`}/failure.json`;
      capture=await json(resolve(config.root,capturePath));
      validateFailedUiCapture(capture,digest(lock),{historical:!chained?'h18':ordinal===2?'h19':null,rowId:pending?.id});
      remaining=await retryRemainingRows(resolve(config.root,'evidence'),plan,digest(lock),digest(binding));
    }
    requireThat(!await optional(resolve(config.root,ordinal===1?'continuation-retry-start.json':`continuation-retry-${ordinal}-start.json`))&&!await optional(resolve(config.root,'execution-continuation.json')),'continuation retry already started or completed');
    retry={checkpoint,checkpointIdentity,historical,adjudication,capture,recovery,guardFrontier,ordinal,previousToolingSha,pendingRowId:capture?.rowId};
  } else {
    await assertOriginalEvidence(resolve(config.root,'evidence'),manifest,{exact:true});
    remaining=await continuationRemainingRows(resolve(config.root,'evidence'),plan);
    requireThat(!await optional(resolve(config.root,'continuation-start.json'))&&!await optional(resolve(config.root,'execution.json')),'original attempt already continued or completed');
  }
  const transfer=await json(resolve(config.root,'hf-transfer.json'));
  requireThat(transfer.to===config.hfHome&&transfer.candidateId===config.candidateId&&transfer.sourceSha===config.sourceSha&&transfer.inferenceSha===config.inferenceSha&&canonical(await privateHfIntegrity(config))===canonical(transfer.after),'stopped private HF ownership or package bytes changed');
  return {manifest,manifestIdentity,lock,binding,launch,rows,remaining,retry};
}
export async function executeBackend(config) {
  const continuing=Boolean(config.continuation);
  const repairTooling=Boolean(config.toolingRoot||continuing);
  const original=continuing?await validateContinuation(config):null;
  const cudaRebase=config.cudaRebase?validateCudaRebaseRecord(config.cudaRebase,config.cudaRebase.old,config):null;
  if(cudaRebase){
    requireThat(config.backend==='cuda'&&!continuing&&!config.completedMlx&&!config.toolingRoot&&canonical(cudaRebase)===canonical(await json(resolve(config.root,'cuda-rebase.json'))),'corrected CUDA requires its separately authenticated source-paired Mac handoff');
    requireThat(canonical(cudaRebase.continuity)===canonical(cudaSourceContinuity({sceneworksRoot:config.sourceRoot,inferenceRoot:config.inferenceRoot,oldSourceSha:cudaRebase.old.sourceSha,oldInferenceSha:cudaRebase.old.inferenceSha,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,auditPath:safePath(config.bundleRoot,config.sourceContinuityAudit.file),auditSha256:config.sourceContinuityAudit.sha256})),'corrected CUDA source difference changed before execution');
    const old=cudaRebase.old,authenticated=await validateCompletedMlxArtifact(resolve(config.root,'prior'),{candidateId:old.candidateId,sourceSha:old.sourceSha,inferenceSha:old.inferenceSha,originalRunId:old.originalRunId,originalAttempt:old.originalAttempt,toolingSha:old.toolingSha,runId:old.runId,attempt:old.attempt,artifactId:old.artifactId});
    requireThat(canonical(old)===canonical({...authenticated,sources:(await json(resolve(config.handoffRoot,'handoff.json'))).sources}),'completed Mac producer changed after corrected CUDA handoff authentication');
    await validateCompletedMlxHandoffSupplement(resolve(config.root,'prior'),{...authenticated,cudaToolingSha:config.sourceSha});
  }
  if(config.completedMlx) {
    const record=await json(resolve(config.root,'completed-mlx-artifact.json')),prior=resolve(config.root,'prior');
    requireThat(config.backend==='cuda'&&canonical(record)===canonical(config.completedMlx)&&record.candidateId===config.candidateId&&record.sourceSha===config.sourceSha&&record.inferenceSha===config.inferenceSha&&record.cudaToolingSha===config.toolingSha&&record.toolingSha===(await json(resolve(prior,'evidence/continuation-retry-8.json'))).toolingSha&&record.completionSha256===(await fileIdentity(resolve(prior,'execution-continuation-retry-8.json'))).sha256&&record.retrySha256===(await fileIdentity(resolve(prior,'evidence/continuation-retry-8.json'))).sha256&&record.handoffSha256===(await fileIdentity(resolve(prior,'handoff/handoff.json'))).sha256&&record.supplementSha256===(await fileIdentity(resolve(prior,'supplement.json'))).sha256,'CUDA-only launch lost its authenticated completed Mac artifact');
  }
  if(original?.retry)config.continuation.retry.ordinal=original.retry.ordinal;
  const retrySuffix=original?.retry?(original.retry.ordinal>1?`retry-${original.retry.ordinal}`:'retry'):null;
  const inventory=validateArtifactInventory(await json(resolve(config.buildRoot,config.backend,'inventory.json')),config.backend,await sourceIdentity(config));
  const installedPath=resolve(config.buildRoot,config.backend,'packages.json');requireThat(canonical(await fileIdentity(installedPath))===canonical(inventory.files.packages),'installed package inventory differs from source-bound native build');
  if(!continuing&&config.preparedCudaPackages) {
    const prepared=await json(resolve(config.root,'cuda-package-preparation.json')),transfer=await json(resolve(config.root,'hf-transfer.json'));
    requireThat(config.backend==='cuda'&&prepared.candidateId===config.candidateId&&prepared.sourceSha===config.sourceSha&&prepared.inferenceSha===config.inferenceSha&&canonical(prepared.originalNativePackages)===canonical(inventory.files.packages)&&canonical(prepared.transfer)===canonical(await fileIdentity(resolve(config.root,'hf-transfer.json')))&&transfer.to===config.hfHome&&canonical(await privateHfIntegrity(config))===canonical(transfer.after),'prepared private CUDA packages changed before native execution');
  } else if(!continuing)await transferPrivateHf(config,await json(installedPath),inventory.files.packages);
  await verifyBuild(config);const paths=buildPaths(config);config.apiBinary=paths.apiBinary;
  // Each execution phase is fresh. An interrupted phase retains immutable attempts;
  // do not start another stack and replay submissions under the same candidate.
  if(continuing) {
    config.projectId=original.binding.endpoint.projectId;
    config.observationJournal=resolve(config.root,config.continuation.retry?`journal-continuation-${retrySuffix}.jsonl`:'journal-continuation.jsonl');
    if(config.continuation.retry) {
      const tag=original.retry.ordinal===1?'h18':`prior-${original.retry.ordinal}`;
      await copyFile(config.continuation.retry.inventoryPath,resolve(config.root,`evidence/continuation-${tag}-inventory.json`),constants.COPYFILE_EXCL);
      if(!original.retry.guardFrontier)await copyFile(resolve(config.root,'evidence/attempts/mlx/dispatch.lock'),resolve(config.root,`evidence/continuation-${tag}-dispatch.lock`),constants.COPYFILE_EXCL);
      for(const item of original.retry.checkpoint.files.filter(item=>!item.file.startsWith('evidence/'))) {
        const destination=safePath(resolve(config.root,'evidence',`continuation-${tag}-root`),item.file);
        await mkdir(dirname(destination),{recursive:true,mode:0o700});
        await copyFile(safePath(config.root,item.file),destination,constants.COPYFILE_EXCL);
      }
      const retryStart={version:VERSION,name:'r16-checker-recovery-selector-retry',ordinal:original.retry.ordinal,originalRunId:config.continuation.originalRunId,originalAttempt:config.continuation.originalAttempt,failedRunId:original.retry.checkpoint.actualRunId,failedAttempt:original.retry.checkpoint.attempt,failedArtifactId:original.retry.checkpoint.artifactId,candidateId:config.candidateId,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,previousToolingSha:original.retry.previousToolingSha,toolingSha:config.continuation.toolingSha,originalInventorySha256:original.manifestIdentity.sha256,checkpointInventorySha256:original.retry.checkpointIdentity.sha256,...(original.retry.guardFrontier?{[original.retry.guardFrontier.name==='edit-dmd'?'postGuardDmd':'postRowsGuard']:original.retry.guardFrontier}:{dispatchLockSha256:(await fileIdentity(resolve(config.root,'evidence',`continuation-${tag}-dispatch.lock`))).sha256}),uiAttemptSha256:digest(await json(resolve(config.root,'evidence/attempts/mlx/advanced-mlx-sft-replay.start.json'))),lockSha256:digest(original.lock),...(original.retry.recovery?{submittedRecovery:original.retry.recovery}:{}),...(original.retry.pendingRowId&&original.retry.ordinal>=4?{pendingUiAttemptRowId:original.retry.pendingRowId,pendingUiAttemptSha256:digest(await json(resolve(config.root,'evidence/attempts/mlx',`${original.retry.pendingRowId}.start.json`)))}:{}),at:new Date().toISOString()};
      const file=original.retry.ordinal===1?'continuation-retry.json':`continuation-${retrySuffix}.json`;
      await save(resolve(config.root,original.retry.ordinal===1?'continuation-retry-start.json':`continuation-${retrySuffix}-start.json`),retryStart);
      await save(resolve(config.root,'evidence',file),retryStart);
    } else {
      await save(resolve(config.root,'continuation-start.json'),{version:VERSION,name:config.continuation.name,originalRunId:config.continuation.originalRunId,originalAttempt:config.continuation.originalAttempt,candidateId:config.candidateId,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,toolingSha:config.continuation.toolingSha,inventorySha256:original.manifestIdentity.sha256,lockSha256:digest(original.lock),at:new Date().toISOString()});
      await copyFile(config.continuation.inventoryPath,resolve(config.root,'evidence/continuation-original-inventory.json'),constants.COPYFILE_EXCL);
    }
  } else {
    requireThat(!await optional(resolve(config.root,'execution-start.json')),'execution already started; authoritative owned-job reconciliation is required before any continuation');
    await save(resolve(config.root,'execution-start.json'),{version:VERSION,candidateId:config.candidateId,backend:config.backend,sources:await sourceIdentity(config),at:new Date().toISOString()});
  }
  const token=randomBytes(32).toString('hex');process.env.MOSS_EDIT_ACTIONS_TOKEN=token;
  if(config.backend==='cuda') {
    const handoff=await json(resolve(config.handoffRoot,'handoff.json'));config.projectId=handoff.projectId;
    await importState(config,config.handoffRoot,cudaRebase?{authenticatedMac:cudaRebase.old}:{});
    if(!cudaRebase)await importCompletedMlxEvidence(config);
    if(config.completedMlx) {
      await save(resolve(config.root,'evidence/external/completed-mlx-artifact-for-cuda.json'),config.completedMlx);
      await cp(resolve(config.root,'prior/supplement.json'),resolve(config.root,'evidence/external/completed-mlx-handoff-supplement.json'),{errorOnExist:true,force:false});
    }
  }
  const stack=await startStack(config,token);let stopped=false;
  try {
    if(continuing) {
      config.guardRequests=await json(resolve(config.root,'evidence/resolved-guards.json'));
      const project=await stack.api.request(`/api/v1/projects/${encodeURIComponent(config.projectId)}`);
      requireThat(project.id===config.projectId,'original owned project did not reopen from stopped state');
    } else if(config.backend==='mlx') {
      const project=await stack.api.request('/api/v1/projects',{method:'POST',body:{name:`MOSS edit ${config.candidateId}`}});config.projectId=project.id;
      const original=await json(safePath(config.bundleRoot,config.inputManifest));
      for(const item of original.assets)item.path=safePath(config.bundleRoot,item.path);
      const originalPath=resolve(config.root,'original-inputs.json');await save(originalPath,original);await prepare(originalPath,resolve(config.root,'evidence'));
      const planPath=resolve(config.root,'evidence','cases.json'),plan=await json(planPath);
      plan.cacheComparisonPolicy={metric:'rgb8-mae/v1',tolerance:config.cacheTolerance,implementationSha256:(await fileIdentity(resolve(config.sourceRoot,'scripts/lib/moss-edit-terminal.mjs'))).sha256};
      validatePlan(plan,original);await writeFile(planPath,`${JSON.stringify(plan,null,2)}\n`,{mode:0o600});
      await prepareHostGuard(config);config.guardRequests=await prepareGuardFixtures(config,stack.api);
    } else if(cudaRebase) {
      const project=await stack.api.request(`/api/v1/projects/${encodeURIComponent(config.projectId)}`);requireThat(project.id===config.projectId,'corrected CUDA did not reopen the authenticated Mac project');
      const originalInputs=await json(safePath(config.bundleRoot,config.inputManifest));
      for(const item of originalInputs.assets)item.path=safePath(config.bundleRoot,item.path);
      const originalPath=resolve(config.root,'original-inputs.json');await save(originalPath,originalInputs);await prepare(originalPath,resolve(config.root,'evidence'));
      const planPath=resolve(config.root,'evidence/cases.json'),plan=await json(planPath);
      plan.cacheComparisonPolicy={metric:'rgb8-mae/v1',tolerance:config.cacheTolerance,implementationSha256:(await fileIdentity(resolve(config.sourceRoot,'scripts/lib/moss-edit-terminal.mjs'))).sha256};
      validatePlan(plan,originalInputs);await writeFile(planPath,`${JSON.stringify(plan,null,2)}\n`,{mode:0o600});
      const priorPlan=await json(resolve(config.priorEvidenceRoot,'cases.json'));
      requireThat(canonical(plan.rows)===canonical(priorPlan.rows)&&canonical(plan.requiredGuardCases)===canonical(priorPlan.requiredGuardCases)&&canonical(await json(resolve(config.root,'evidence/input-identities.json')))===canonical(await json(resolve(config.priorEvidenceRoot,'input-identities.json'))),'corrected CUDA changed frozen row, refusal or original input semantics');
      await prepareHostGuard(config);config.guardRequests=await prepareGuardFixtures(config,stack.api);
      for(const row of plan.rows.filter(row=>row.backend==='mlx'))await cp(resolve(config.priorEvidenceRoot,'rows',row.id),resolve(config.root,'evidence/rows',row.id),{recursive:true,errorOnExist:true,force:false});
      await authenticateCrossBackendHistory(stack.api,config.priorEvidenceRoot,resolve(config.root,'evidence'));
      await save(resolve(config.root,'evidence/external/completed-mlx-artifact-for-cuda.json'),cudaRebase.old);
      await relocateImportedCudaModelLibrary(stack.api,{...config,completedMlx:cudaRebase.old});
    } else {
      config.guardRequests=await json(resolve(config.root,'evidence','resolved-guards.json'));
      const project=await stack.api.request(`/api/v1/projects/${encodeURIComponent(config.projectId)}`);requireThat(project.id===config.projectId,'target API did not reopen the original transferred project identity');
      await authenticateCrossBackendHistory(stack.api,resolve(config.root,'evidence'));
      if(config.completedMlx)await relocateImportedCudaModelLibrary(stack.api,config);
    }
    if((config.backend==='mlx'||cudaRebase)&&!continuing)for(const backend of (cudaRebase?['cuda']:['mlx','cuda'])) {
      const toolchain=await json(resolve(config.buildRoot,backend,'toolchain.json'));requireThat(toolchain.installation?.files?.length,'actual installer capture missing from native artifact');
      for(const item of toolchain.installation.files)requireThat((await fileIdentity(safePath(resolve(config.buildRoot,backend,'installation'),item.file))).sha256===item.sha256,'native installer evidence bytes changed');
      await cp(resolve(config.buildRoot,backend,'installation'),resolve(config.root,'evidence/external',`raw-install-${backend}`),{recursive:true,errorOnExist:true,force:false});
      await cp(resolve(config.buildRoot,backend,'toolchain.json'),resolve(config.root,'evidence/external',`toolchain-${backend}.json`),{errorOnExist:true,force:false});
    }
    const toolingRoot=config.toolingRoot??config.continuation?.toolingRoot,toolingFiles=repairTooling?Object.fromEntries(Object.entries({harness:'moss-edit-terminal.mjs',harnessLibrary:'lib/moss-edit-terminal.mjs',actionController:'moss-edit-terminal-actions.mjs',actionLibrary:'lib/moss-edit-terminal-actions.mjs',browserDriver:'moss-edit-terminal-browser.mjs',browserControls:'lib/moss-edit-terminal-controls.mjs'}).map(([name,path])=>[name,resolve(toolingRoot,'scripts',path)])):null;
    const tooling=repairTooling?{sourceRoot:toolingRoot,head:config.toolingSha??config.continuation.toolingSha,source:treeIdentity(toolingRoot),files:toolingFiles,identities:Object.fromEntries(await Promise.all(Object.entries(toolingFiles).map(async([name,path])=>[name,await fileIdentity(path)])))}:null;
    const retry=continuing?original.retry:await latestRetry(resolve(config.root,'evidence'));
    const launch=continuing?{...original.launch,guardRequests:config.guardRequests,repair:{name:'r16-checker-recovery',originalRoot:config.root,inventorySha256:original.manifestIdentity.sha256,tooling,...(retry?{previousToolingSha:H18_RETRY_TOOLING_SHA,checkpointInventorySha256:H18_RETRY_INVENTORY_SHA256,retryOrdinal:retry.ordinal,...(await guardRepairFields(resolve(config.root,'evidence'),retry))}:{})}}:await launcher(config,config.projectId);
    if(repairTooling&&!continuing)launch.repair={name:'r16-checker-recovery',inventorySha256:(await fileIdentity(resolve(config.root,'evidence/continuation-original-inventory.json'))).sha256,tooling,...(retry?{previousToolingSha:H18_RETRY_TOOLING_SHA,checkpointInventorySha256:H18_RETRY_INVENTORY_SHA256,retryOrdinal:retry.ordinal??1,executionToolingSha:retry.toolingSha,...(await guardRepairFields(resolve(config.root,'evidence'),retry))}:{})};
    if(config.completedMlx)launch.repair.completedMlx=config.completedMlx;
    if(cudaRebase){launch.cudaRebase={...cudaRebase,auditPath:safePath(config.bundleRoot,config.sourceContinuityAudit.file)};launch.stoppedCuda=config.stoppedCuda;}
    const launchPath=resolve(config.root,continuing?(config.continuation.retry?`continuation-${retrySuffix}-launch.json`:'continuation-launch.json'):'launch.json');await save(launchPath,launch);
    if(config.backend==='cuda'&&repairTooling)await cp(launchPath,resolve(config.root,'evidence/external/cuda-execution-tooling-capture.json'),{errorOnExist:true,force:false});
    if(!continuing) {if(config.backend==='mlx'||cudaRebase)await freeze(launchPath);else await bind(launchPath);}
    const state=await frozen(launchPath),toolchain=await json(paths.toolchain),expectedJournal={candidateId:config.candidateId,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,workerId:config.workerId,
      gpuId:state.binding.endpoint.workerGpuId,
      binarySha256:state.lock.artifacts[config.backend].files.workerBinary.sha256,producerSha256:toolchain.observationProducerSha256};
    await actualJournal(config,expectedJournal);
    if(!continuing)await harness(config,['upload',launchPath,config.backend,await deviceReceipt(config,state,'upload')],'upload');
    const samplerSha256=(await fileIdentity(resolve(config.sourceRoot,'scripts/lib/moss-edit-terminal-actions.mjs'))).sha256;
    const originalActions=repairTooling?await import(pathToFileURL(resolve(config.sourceRoot,'scripts/lib/moss-edit-terminal-actions.mjs')).href):null;
    if(continuing&&!original.retry) {
      await harness(config,['recover-completed',launchPath,'advanced-mlx-sft'],'recover-completed');
      const id='advanced-mlx-sft',root=state.root,job=await json(resolve(root,'rows',id,'recovery-job-readback.json'));
      const start=await json(resolve(root,'attempts/mlx',`${id}.start.json`));
      const rawEvidence=await Promise.all([
        `rows/${id}/failure.json`,`rows/${id}/terminal-job.json`,`rows/${id}/recovery-job-readback.json`,
        `rows/${id}/receipt.json`,`attempts/mlx/${id}.failure.json`
      ].map(async file=>({file,...await fileIdentity(safePath(root,file))})));
      const proof={version:VERSION,lockSha256:digest(state.lock),attemptSha256:digest(start),coordinator:'source-bound continuation controller',backend:'mlx',observedAt:new Date().toISOString(),state:'NO_ACTIVE_OWNED_JOB',jobs:[job],rawEvidence};
      await reconcileAttempt(root,'mlx',id,proof,digest(state.lock));
      await save(resolve(root,'recovery-adjudication.json'),{version:VERSION,name:'r16-checker-recovery',rowId:id,originalRunId:config.continuation.originalRunId,originalAttempt:config.continuation.originalAttempt,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,candidateId:config.candidateId,toolingSha:config.continuation.toolingSha,inventorySha256:original.manifestIdentity.sha256,lockSha256:digest(state.lock),jobId:job.id,requestSha256:start.requestSha256,originalAttemptSha256:digest(start),reconciliationSha256:digest(proof),rawEvidence});
    }
    if(original?.retry?.recovery) {
      const id=original.retry.recovery.rowId;
      await harness(config,['recover-canceled',launchPath,id],`${id}-recover`);
      const receipt=await json(resolve(state.root,'rows',id,'receipt.json'));
      requireThat(receipt.jobId===original.retry.recovery.jobId&&receipt.recoveredWithoutSubmission===true,'known canceled job was not recovered without submission');
      const restarted=await stack.restartWorker(receipt.jobId);await actualJournal(config,expectedJournal);
      const proofFile=resolve(config.root,'cleanup',`${id}.json`);await save(proofFile,restarted);
      const target=state.plan.rows.find(item=>item.action==='retry'&&item.replayOf===id);
      requireThat(target,'recovered cancellation has no allocated retry row');
      const saved=resolve(state.root,'external',`cleanup-${id}.json`);await cp(proofFile,saved,{errorOnExist:true,force:false});
      await save(resolve(state.root,'external',`cleanup-before-${target.id}.json`),{version:VERSION,lockSha256:digest(state.lock),canceledJobId:receipt.jobId,verdict:'PASS',observer:'source-bound owned-worker process-exit/restart',files:[{file:`external/cleanup-${id}.json`,...await fileIdentity(saved)}]});
    }
    for(const row of state.plan.rows.filter(row=>row.backend===config.backend)) {
      const disposition=continuing?continuationRowDisposition(row,original.remaining,{retry:Boolean(original.retry)}):{run:true,reuseUiIntent:false};
      if(!disposition.run)continue;
      const reusingUiIntent=disposition.reuseUiIntent;
      if(!reusingUiIntent)await assertDispatchClear(state.root,config.backend);
      const suffix=retrySuffix?`-${retrySuffix}`:'';
      await appendFile(resolve(config.root,continuing?(original.retry?`progress-continuation-${retrySuffix}.jsonl`:'progress-continuation.jsonl'):'progress.jsonl'),`${JSON.stringify({rowId:row.id,phase:'before',at:new Date().toISOString()})}\n`,{mode:0o600});
      const measurement=await (originalActions?.sampler??sampler)({backend:config.backend,file:resolve(config.root,'samples',`${row.id}${suffix}.jsonl`)});
      let capture;
      try {
        const admission=await deviceReceipt(config,state,`${row.id}${suffix}`);
        if(row.transport.startsWith('ui-')) {
          if(!reusingUiIntent)await harness(config,['ui-task',launchPath,row.id],`${row.id}-intent`);
          const {captureUi}=repairTooling?await import(pathToFileURL(resolve(toolingRoot,'scripts/moss-edit-terminal-browser.mjs')).href):await import('./moss-edit-terminal-browser.mjs');
          capture=await captureUi({launchPath,rowId:row.id,captureRoot:resolve(config.root,'captures',`${row.id}${suffix}`),playwrightModule:config.playwrightModule,token});
        }
        await harness(config,['run',launchPath,row.id,admission,...(capture?[capture]:[])],`${row.id}-run`);
        const receipt=await json(resolve(state.root,'rows',row.id,'receipt.json'));
        if(row.expected==='canceled') {
          const restarted=await stack.restartWorker(receipt.jobId);await actualJournal(config,expectedJournal);
          const proofFile=resolve(config.root,'cleanup',`${row.id}.json`);await save(proofFile,restarted);
          const target=state.plan.rows.find(item=>item.action==='retry'&&item.replayOf===row.id);
          const saved=resolve(state.root,'external',`cleanup-${row.id}.json`);await cp(proofFile,saved,{errorOnExist:true,force:false});
          await save(resolve(state.root,'external',`cleanup-before-${target.id}.json`),{version:VERSION,lockSha256:digest(state.lock),canceledJobId:receipt.jobId,verdict:'PASS',observer:'source-bound owned-worker process-exit/restart',files:[{file:`external/cleanup-${row.id}.json`,...await fileIdentity(saved)}]});
        }
        const memory=await measurement.stop(receipt.jobId,samplerSha256),rawPath=resolve(state.root,'rows',row.id,'memory-raw.jsonl');
        await cp(resolve(config.root,'samples',`${row.id}${suffix}.jsonl`),rawPath,{errorOnExist:true,force:false});memory.raw={file:'memory-raw.jsonl',...await fileIdentity(rawPath)};
        await save(resolve(state.root,'rows',row.id,'memory.json'),memory);
        if(row.anchors)await (originalActions?.retainQualityPairs??retainQualityPairs)(state,row);
        if(row.compareTo)await harness(config,['compare-cache',launchPath,row.id],`${row.id}-cache`);
        await appendFile(resolve(config.root,continuing?(original.retry?`progress-continuation-${retrySuffix}.jsonl`:'progress-continuation.jsonl'):'progress.jsonl'),`${JSON.stringify({rowId:row.id,phase:'verified',jobId:receipt.jobId,at:new Date().toISOString()})}\n`);
      }catch(error){await measurement.stop('UNRESOLVED',samplerSha256).catch(()=>{});throw error;}
    }
    if(original?.retry?.guardFrontier) {
      if(original.retry.guardFrontier.name==='edit-dmd')await reconcileDmdInputCorrection(state.root,state.plan,digest(state.lock),digest(state.binding),launch.guardRequests);
      else await reconcileParserRefusal(state.root,state.plan,digest(state.lock),digest(state.binding),launch.guardRequests[original.retry.guardFrontier.name]);
    }
    for(const [guardIndex,name] of state.plan.requiredGuardCases.entries()) {
      if(original?.retry?.guardFrontier&&guardIndex<original.retry.guardFrontier.guardStartIndex)continue;
      const retryGuard=original?.retry?.guardFrontier?.name===name;
      requireThat(launch.guardRequests[name].noProviderLoadProbe===OBSERVATION_METHOD,'every refusal must freeze the actual source-bound MOSS journal probe');
      let coldRestart=null;
      if(['device-budget-before-load','corrupt-adapter'].includes(name)&&launch.guardRequests[name].expectedTerminal==='failed') {
        coldRestart=await stack.restartIdleWorker(`allocated-${name}-cold-refusal`);await actualJournal(config,expectedJournal);
      }
      const mutation=await adapterRefusalState(config,name,launch.guardRequests[name]);
      try {
      const before=await actualJournal(config,expectedJournal),beforeAt=new Date().toISOString();
      await harness(config,[retryGuard?'guard-retry':'guard',launchPath,name,config.backend,await deviceReceipt(config,state,`guard-${name}${retryGuard?`-${retrySuffix}`:''}`),...(retryGuard?[String(original.retry.ordinal)]:[])],`guard-${name}`);
      const afterAt=new Date().toISOString(),after=await actualJournal(config,expectedJournal,afterAt),directory=resolve(state.root,'guards',`${config.backend}-${name}${retryGuard?`-retry-${original.retry.ordinal}`:''}`);
      await writeFile(resolve(directory,'journal-before.jsonl'),before,{flag:'wx',mode:0o600});await writeFile(resolve(directory,'journal-after.jsonl'),after,{flag:'wx',mode:0o600});await cp(paths.toolchain,resolve(directory,'toolchain.json'),{errorOnExist:true,force:false});
      const receipt=await json(resolve(directory,'receipt.json')),rawEvidence=[];
      if(coldRestart)await save(resolve(directory,'cold-worker-restart.json'),coldRestart);
      if(mutation.observation)await save(resolve(directory,'adapter-state.json'),mutation.observation);
      for(const file of ['journal-before.jsonl','journal-after.jsonl','toolchain.json'])rawEvidence.push({file,...await fileIdentity(resolve(directory,file))});
      if(mutation.observation)rawEvidence.push({file:'adapter-state.json',...await fileIdentity(resolve(directory,'adapter-state.json'))});
      if(coldRestart)rawEvidence.push({file:'cold-worker-restart.json',...await fileIdentity(resolve(directory,'cold-worker-restart.json'))});
      await save(resolve(directory,'no-provider-load.json'),{version:VERSION,lockSha256:digest(state.lock),case:name,backend:config.backend,requestSha256:receipt.requestSha256,method:OBSERVATION_METHOD,
        observer:'actual source-bound instrumented native worker',beforeAt,afterAt,jobId:receipt.jobId??null,beforeFile:'journal-before.jsonl',afterFile:'journal-after.jsonl',toolchainFile:'toolchain.json',rawEvidence,observation:noLoadComparison(before,after,expectedJournal,{beforeAt,afterAt,jobId:receipt.jobId??null})});
      if(retryGuard&&name==='training-shift-nonfinite') {
        const prefix=`guards/mlx-${name}-retry-${original.retry.ordinal}`;
        await save(resolve(state.root,'external','lexical-parser-status-recovery.json'),{version:VERSION,name:'lexical-parser-status-recovery',ordinal:original.retry.ordinal,case:name,backend:'mlx',lockSha256:digest(state.lock),original:original.retry.guardFrontier,
          retryRequestSha256:(await fileIdentity(resolve(directory,'request.raw.json'))).sha256,retryResponseSha256:(await fileIdentity(resolve(directory,'response.json'))).sha256,retryReceiptSha256:(await fileIdentity(resolve(directory,'receipt.json'))).sha256,retryNoLoadSha256:(await fileIdentity(resolve(directory,'no-provider-load.json'))).sha256,retryResolutionSha256:(await fileIdentity(resolve(state.root,'attempts/mlx',`guard-${name}-retry-${original.retry.ordinal}.resolution.json`))).sha256,path:prefix});
      } else if(retryGuard&&name==='edit-dmd') {
        const prefix=`guards/mlx-${name}-retry-${original.retry.ordinal}`;
        await save(resolve(state.root,'external','edit-dmd-input-correction.json'),{version:VERSION,name:'edit-dmd-input-correction',ordinal:original.retry.ordinal,case:name,backend:'mlx',lockSha256:digest(state.lock),original:original.retry.guardFrontier,path:prefix,
          correctedRequestSha256:receipt.requestSha256,requestCorrectionSha256:(await fileIdentity(resolve(directory,'request-correction.json'))).sha256,retryResponseSha256:(await fileIdentity(resolve(directory,'response.json'))).sha256,retryReceiptSha256:(await fileIdentity(resolve(directory,'receipt.json'))).sha256,retryNoLoadSha256:(await fileIdentity(resolve(directory,'no-provider-load.json'))).sha256,retryResolutionSha256:(await fileIdentity(resolve(state.root,'attempts/mlx',`guard-${name}-retry-${original.retry.ordinal}.resolution.json`))).sha256});
      }
      }finally{await mutation.restore();}
    }
    // Stop only the processes this controller spawned. No global census/ownership gate.
    await stack.stop();stopped=true;
    if(config.backend==='mlx')await exportState(config,resolve(config.root,'handoff'));
    if(!continuing)await releasePrivateHf(config);
    await save(resolve(config.root,continuing?(original.retry?`execution-continuation-${retrySuffix}.json`:'execution-continuation.json'):'execution.json'),{version:VERSION,lockSha256:digest(state.lock),backend:config.backend,renders:25,refusals:29,execution:'COMPLETE',acceptance:'AWAITING_ALL_EXTERNAL_EVIDENCE_AND_VISUAL_REVIEW',...(continuing?{toolingSha:config.continuation.toolingSha,originalRunId:config.continuation.originalRunId}:{} )});
  }finally{if(!stopped)await stack.stop();delete process.env.MOSS_EDIT_ACTIONS_TOKEN;}
}
export async function finalize(config) {
  // Root-owned actual reviews/observations arrive as immutable evidence files,
  // not synthetic PASS receipts generated by this controller.
  requireThat(config.reviewRoot,'actual external/visual review artifact root required');
  if(config.cudaRebaseReview) {
    const priorCuda=resolve(config.root,'prior-cuda'),record=await json(resolve(priorCuda,'cuda-rebase.json'));
    requireThat(record.name==='sc-26757-corrected-cuda-candidate/v1'&&record.current.candidateId===config.candidateId&&record.current.sourceSha===config.sourceSha&&record.current.inferenceSha===config.inferenceSha,'retained corrected CUDA execution changed source pair');
    config.cudaRebase={...record,auditPath:safePath(config.bundleRoot,config.sourceContinuityAudit.file)};
    config.priorEvidenceRoot=resolve(priorCuda,'evidence');
    await cp(resolve(priorCuda,'cuda-rebase.json'),resolve(config.root,'cuda-rebase.json'),{errorOnExist:true,force:false});
  }
  await cp(config.priorEvidenceRoot,resolve(config.root,'evidence'),{recursive:true,errorOnExist:true,force:false});
  const root=resolve(config.root,'evidence'),binding=await json(resolve(root,'bindings',`${config.backend}.json`));
  config.guardRequests=await json(resolve(root,'resolved-guards.json'));
  const launch=await launcher(config,binding.endpoint.projectId),launchPath=resolve(config.root,'review-launch.json');
  if(config.cudaRebaseReview) {
    const {oldSourceRoot,oldInferenceRoot,oldMlxBuildRoot}=config.cudaRebaseReview;
    launch.cudaRebase={...config.cudaRebase,historicalLaunchPath:await historicalMlxLaunchForCudaRebase(config,resolve(config.root,'prior'),oldSourceRoot,oldInferenceRoot,oldMlxBuildRoot,config.sourceRoot)};launch.stoppedCuda=config.stoppedCuda;
  }
  if(config.toolingRoot) {
    const files=Object.fromEntries(Object.entries({harness:'moss-edit-terminal.mjs',harnessLibrary:'lib/moss-edit-terminal.mjs',actionController:'moss-edit-terminal-actions.mjs',actionLibrary:'lib/moss-edit-terminal-actions.mjs',browserDriver:'moss-edit-terminal-browser.mjs',browserControls:'lib/moss-edit-terminal-controls.mjs',qualityAmendment:'lib/moss-edit-quality-amendment.mjs'}).map(([name,path])=>[name,resolve(config.toolingRoot,'scripts',path)]));
    const tooling={sourceRoot:config.toolingRoot,head:config.toolingSha,source:treeIdentity(config.toolingRoot),files,identities:Object.fromEntries(await Promise.all(Object.entries(files).map(async([name,path])=>[name,await fileIdentity(path)])))};
    const retry=await latestRetry(root);
    launch.repair={name:'r16-checker-recovery',inventorySha256:(await fileIdentity(resolve(root,'continuation-original-inventory.json'))).sha256,tooling,historicalExecutionToolingSha:'2cd0363d59cb3eb0989bbb708000b6ffc907932a',executionToolingSha:retry?.toolingSha??H18_RETRY_TOOLING_SHA,verificationToolingSha:config.toolingSha,...(retry?{previousToolingSha:H18_RETRY_TOOLING_SHA,checkpointInventorySha256:H18_RETRY_INVENTORY_SHA256,retryOrdinal:retry.ordinal??1,...(await guardRepairFields(root,retry))}:{})};
    const completed=await optional(resolve(root,'external/completed-mlx-artifact-for-cuda.json'));
    if(completed) {
      const cudaCapture=await json(resolve(root,'external/cuda-execution-tooling-capture.json'));
      requireThat(cudaCapture.repair?.completedMlx&&canonical(cudaCapture.repair.completedMlx)===canonical(completed),'CUDA-only launch lost its separate completed-Mac producer');
      launch.repair.completedMlx=completed;launch.repair.cudaExecutionToolingSha=cudaCapture.repair.tooling.head;
    }
  }
  await save(launchPath,launch);
  for(const entry of await filesUnder(config.reviewRoot)) {
    const destination=safePath(root,entry.file);await mkdir(dirname(destination),{recursive:true});await cp(safePath(config.reviewRoot,entry.file),destination,{errorOnExist:true,force:false});
  }
  await harness(config,['verify-retained',launchPath],'final-verify');
}
export async function main(args=process.argv.slice(2)) {
  const [phase,...rest]=args;
  if(phase==='normalize-env'&&rest.length===0)return exportMossEnvironment(process.env);
  if(phase==='bundle'&&rest.length===3){const result=await createBundle(...rest);process.stdout.write(`${JSON.stringify(result)}\n`);return result;}
  if(phase==='inventory-review'&&rest.length===4){const result=await inventoryReview(...rest);process.stdout.write(`${JSON.stringify(result)}\n`);return result;}
  if(phase==='validate-upload'&&rest.length===6)return validateUpload(...rest);
  if(phase==='arrange-builds'&&(rest.length===1||rest.length===2))return arrangeBuilds(rest[0],rest[1]??null);
  if(phase==='validate-bundle'&&rest.length===5)return validateBundle(...rest);
  if(phase==='configure-env'&&rest.length===0) {
    const env=process.env;await validateBundle(env.MOSS_BUNDLE,env.MOSS_BUNDLE_SHA,env.MOSS_SOURCE_SHA,env.MOSS_INFERENCE_SHA,env.MOSS_CANDIDATE);
    return configure(resolve(env.MOSS_BUNDLE,'campaign.json'),env.MOSS_SOURCE,env.MOSS_INFERENCE,env.MOSS_ROOT,env.MOSS_BACKEND,env.MOSS_BUILDS);
  }
  if(phase==='configure'&&rest.length===6)return configure(...rest);
  if(phase==='continue-mlx'&&rest.length===12) {
    const [runtime,toolingRoot,toolingSha,runId,attempt,inventoryPath,inventorySha256,retryPath,retrySha,retryRun,retryArtifact,retryArtifactRoot]=rest,config=await json(runtime);
    requireThat(Boolean(retryPath)===Boolean(retrySha),'H18 retry inventory path and SHA must be supplied together');
    requireThat(Boolean(retryRun)===Boolean(retryArtifact)&&(retryRun?Boolean(retryPath):true),'chained retry run/artifact must be supplied together with its inventory');
    config.continuation={name:'r16-checker-recovery',toolingRoot:resolve(toolingRoot),toolingSha,originalRunId:Number(runId),originalAttempt:Number(attempt),inventoryPath:resolve(inventoryPath),inventorySha256,...(retryPath?{retry:{inventoryPath:resolve(retryPath),inventorySha256:retrySha,...(retryRun?{runId:Number(retryRun),artifactId:Number(retryArtifact),artifactRoot:resolve(retryArtifactRoot)}:{})}}:{})};
    return executeBackend(config);
  }
  if(phase==='rehydrate-cuda'&&rest.length===1)return rehydrateCudaPackages(await json(rest[0]));
  if(phase==='prepare-cuda-packages'&&(rest.length===4||rest.length===8)) {
    const [runtime,rehydrateRuntime,originalRunId,originalAttempt,priorRunId,priorAttempt,priorArtifactId,priorArtifactRoot]=rest;
    const stopped=rest.length===8?{runId:Number(priorRunId),attempt:Number(priorAttempt),artifactId:Number(priorArtifactId),artifactRoot:priorArtifactRoot}:null;
    return prepareContinuationCudaPackages(await json(runtime),await json(rehydrateRuntime),Number(originalRunId),Number(originalAttempt),stopped);
  }
  if(phase==='completed-mlx-handoff'&&rest.length===8) {
    const [runtime,originalRunId,originalAttempt,runId,attempt,artifactId,mlxToolingSha,cudaToolingSha]=rest;
    const toolingRoot=resolve(dirname(fileURLToPath(import.meta.url)),'..');
    return completedMlxHandoff(await json(runtime),Number(originalRunId),Number(originalAttempt),Number(runId),Number(attempt),Number(artifactId),mlxToolingSha,cudaToolingSha,toolingRoot);
  }
  if(phase==='completed-cuda-rebase-handoff'&&rest.length===10) {
    const [runtime,candidateId,sourceSha,inferenceSha,originalRunId,originalAttempt,runId,attempt,artifactId,toolingSha]=rest,config=await json(runtime);
    const old={candidateId,sourceSha,inferenceSha,originalRunId:Number(originalRunId),originalAttempt:Number(originalAttempt),runId:Number(runId),attempt:Number(attempt),artifactId:Number(artifactId),toolingSha};
    return completedCudaRebaseHandoff(config,old,config.sourceRoot,config.sourceSha);
  }
  if(phase==='stage-completed-mlx-handoff'&&rest.length===13) {
    const [prior,stopped,stage,candidateId,sourceSha,inferenceSha,originalRunId,originalAttempt,runId,attempt,artifactId,mlxToolingSha,cudaToolingSha]=rest;
    const toolingRoot=resolve(dirname(fileURLToPath(import.meta.url)),'..');
    requireThat(treeIdentity(toolingRoot).head===cudaToolingSha&&cudaToolingSha!==mlxToolingSha,'hidden handoff staging requires exact separate clean CUDA tooling');
    await execFile('git',['-C',toolingRoot,'merge-base','--is-ancestor',mlxToolingSha,cudaToolingSha]);
    return stageCompletedMlxHandoffSupplement(prior,stopped,stage,{candidateId,sourceSha,inferenceSha,originalRunId:Number(originalRunId),originalAttempt:Number(originalAttempt),runId:Number(runId),attempt:Number(attempt),artifactId:Number(artifactId),toolingSha:mlxToolingSha,stagingToolingSha:cudaToolingSha});
  }
  if(phase==='continue-cuda'&&rest.length===3) {
    const [runtime,toolingRoot,toolingSha]=rest,config=await json(runtime);
    requireThat(treeIdentity(toolingRoot).head===toolingSha&&toolingSha!==config.sourceSha,'CUDA continuation tooling differs from exact clean repair commit');
    config.toolingRoot=resolve(toolingRoot);config.toolingSha=toolingSha;config.preparedCudaPackages=true;
    return executeBackend(config);
  }
  if(phase==='continue-cuda-only'&&rest.length===4) {
    const [runtime,toolingRoot,toolingSha,recordPath]=rest,config=await json(runtime),record=await json(recordPath);
    requireThat(resolve(recordPath)===resolve(config.root,'completed-mlx-artifact.json')&&record.cudaToolingSha===toolingSha&&treeIdentity(toolingRoot).head===toolingSha&&toolingSha!==config.sourceSha,'CUDA-only continuation must use its authenticated completed-Mac handoff and clean new tooling');
    config.toolingRoot=resolve(toolingRoot);config.toolingSha=toolingSha;config.preparedCudaPackages=true;config.completedMlx=record;
    return executeBackend(config);
  }
  if(phase==='execute-cuda-rebase'&&rest.length===2) {
    const [runtime,recordPath]=rest,config=await json(runtime),record=await json(recordPath);
    requireThat(resolve(recordPath)===resolve(config.root,'cuda-rebase.json'),'corrected CUDA must use its owned authenticated handoff record');
    config.cudaRebase=record;
    return executeBackend(config);
  }
  if(phase==='build-cuda-rebase'&&rest.length===5)return buildCudaRebase(await json(rest[0]),resolve(rest[1]),Number(rest[2]),Number(rest[3]),Number(rest[4]));
  if(phase==='finalize-cuda-rebase'&&rest.length===4) {
    const [runtime,oldSourceRoot,oldInferenceRoot,oldMlxBuildRoot]=rest,config=await json(runtime);
    config.cudaRebaseReview={oldSourceRoot:resolve(oldSourceRoot),oldInferenceRoot:resolve(oldInferenceRoot),oldMlxBuildRoot:resolve(oldMlxBuildRoot)};
    return finalize(config);
  }
  if(phase==='finalize-repair'&&rest.length===3) {
    const [runtime,toolingRoot,toolingSha]=rest,config=await json(runtime);
    requireThat(treeIdentity(toolingRoot).head===toolingSha&&toolingSha!==config.sourceSha,'retained verifier differs from exact clean repair commit');
    config.toolingRoot=resolve(toolingRoot);config.toolingSha=toolingSha;return finalize(config);
  }
  requireThat(rest.length===1,'Usage: bundle CAMPAIGN INPUT_MANIFEST FRESH_ROOT | inventory-review ROOT SOURCE INFERENCE CANDIDATE | validate-upload ROOT KIND SHA SOURCE INFERENCE CANDIDATE | validate-bundle ROOT SHA SOURCE INFERENCE CANDIDATE | configure BUNDLE SOURCE INFERENCE ROOT BACKEND BUILDS | configure-env | arrange-builds DOWNLOAD_ROOT | build CONFIG | provision-browser CONFIG | execute CONFIG | finalize CONFIG');
  const config=await json(rest[0]);
  if(phase==='build')return buildBackend(config);
  if(phase==='provision-browser')return provisionBrowser(config);
  if(phase==='execute')return executeBackend(config);
  if(phase==='finalize')return finalize(config);
  throw Error('unknown MOSS terminal Actions phase');
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch(error=>{process.stderr.write(`${JSON.stringify({status:'STOP',message:error.message})}\n`);process.exitCode=1;});
