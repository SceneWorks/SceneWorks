#!/usr/bin/env node
// sc-26757 native Actions controller. Imports are inert; only explicit CLI phases execute.
import { randomBytes } from 'node:crypto';
import { mkdir, readFile, writeFile, cp, appendFile, chmod, rename, readdir } from 'node:fs/promises';
import { execFile as callback } from 'node:child_process';
import { promisify } from 'node:util';
import { resolve, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { VERSION, BACKEND_FILES, sha, digest, canonical, requireThat, json, save, safePath, fileIdentity, validatePlan, validateArtifactInventory, validateAdmission, assertDispatchClear } from './lib/moss-edit-terminal.mjs';
import { prepare, freeze, bind, frozen } from './moss-edit-terminal.mjs';
import { command, sourceIdentity, filesUnder, contained, nativeDevice, sampler, startStack, exportState, importState, retainQualityPairs, delay, CUDA_UUID } from './lib/moss-edit-terminal-actions.mjs';
import { inspectJournal, noLoadComparison, OBSERVATION_METHOD, hostBudgetHeader, hostGuardSpec, hostTensorInvariant } from './lib/moss-edit-terminal-observation.mjs';
import { stripJsoncComments } from './lib/jsonc.mjs';

const execFile=promisify(callback);
const optional=async path=>json(path).catch(error=>{if(error.code==='ENOENT')return null;throw error;});
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
export async function isolatePackages(config,{omitAdapter=false}={}) {
  requireThat(config.hfHome===resolve(config.root,'hf')&&config.packageSource!==config.hfHome,'native jobs require their own HF projection; shared caches are read-only');
  for(const item of config.packageFiles) {
    const source=safePath(config.packageSource,item.file);requireThat((await fileIdentity(source)).sha256===item.sha256,'source package file differs from immutable input inventory');
    if(omitAdapter&&(config.editAdapterFiles??[]).includes(item.file))continue;
    const target=safePath(config.hfHome,item.file);await mkdir(dirname(target),{recursive:true,mode:0o700});await cp(source,target,{dereference:true,errorOnExist:true,force:false});
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
export async function arrangeBuilds(root) {
  const entries=await readdir(root,{withFileTypes:true});
  for(const backend of ['mlx','cuda']) {
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
    let actualSha256=fixture.sha256;
    if(fixture.corruptWith) {
      requireThat(fixture.type==='tensor'&&fixture.project==='campaign','only an explicitly owned campaign tensor may be corrupted');
      const corrupt=await readFile(safePath(config.bundleRoot,fixture.corruptWith.file));requireThat(sha(corrupt)===fixture.corruptWith.sha256&&sha(corrupt)!==fixture.sha256,'actual corrupt tensor bytes/SHA required');
      const registry=await json(resolve(config.stateRoot,'data','recent-projects.json')),project=registry.find(item=>item.id===projectId);
      requireThat(project,'owned actual project registry missing');const owned=contained(resolve(config.stateRoot,'data'),project.path);
      await writeFile(safePath(owned,asset.file.path),corrupt,{mode:0o600});actualSha256=sha(corrupt);
      requireThat(sha(await api.request(media,{raw:true}))===actualSha256,'corrupt owned tensor bytes not actually served');
    }
    const readback=await api.request(route);requireThat(readback.id===asset.id&&readback.projectId===projectId,'fixture identity changed after setup');
    const uploadedFile=`guard-fixtures/${fixture.alias}-uploaded.bin`,actualFile=`guard-fixtures/${fixture.alias}-actual.bin`;
    await mkdir(root,{recursive:true,mode:0o700});await writeFile(resolve(root,`${fixture.alias}-uploaded.bin`),bytes,{flag:'wx',mode:0o600});
    await writeFile(resolve(root,`${fixture.alias}-actual.bin`),await api.request(media,{raw:true}),{flag:'wx',mode:0o600});
    bindings[fixture.alias]={projectId,asset:readback,uploadedSha256:fixture.sha256,actualSha256,uploadedFile,actualFile,cases:fixture.cases??[]};
    await save(resolve(root,`${fixture.alias}.json`),bindings[fixture.alias]);
  }
  await save(resolve(root,'bindings.json'),bindings);
  const resolved=resolveGuardAliases(config.guardRequests,bindings);
  for(const name of ['foreign-reference','foreign-tensor','corrupt-tensor','embeddings-nonfinite','mask-nonbinary'])if(Object.hasOwn(resolved,name)) {
    const fixture=(config.guardFixtures??[]).find(item=>item.cases?.includes(name)),binding=fixture&&bindings[fixture.alias];
    requireThat(binding&&canonical(resolved[name].request).includes(JSON.stringify(binding.asset.id)),`${name} requires its actual uploaded/read-back fixture ID`);
    requireThat(fixture.project===(name.startsWith('foreign-')?'foreign':'campaign')&&fixture.type===(name==='foreign-reference'?'image':'tensor'),`${name} fixture has wrong real project/type`);
    if(!name.startsWith('foreign-'))requireThat(fixture.corruptWith&&binding.actualSha256!==binding.uploadedSha256,`${name} requires actual owned corrupt/nonfinite/nonbinary bytes`);
    if(name==='foreign-reference')requireThat(!resolved[name].refs?.length,'foreign reference guard cannot overwrite its actual foreign ID with ordinary references');
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
  const path=safePath(config.hfHome,specification.adapterFile),original=await readFile(path),expected=config.packageFiles.find(item=>item.file===specification.adapterFile);
  requireThat(expected?.sha256===sha(original),'adapter refusal initial bytes differ from frozen package inventory');
  let corruptSha256=null;
  if(name==='missing-adapter')await rename(path,`${path}.moss-terminal-missing`);
  else {
    const bytes=Buffer.from(specification.corruptBytesBase64??'','base64');requireThat(bytes.length>0&&sha(bytes)===specification.corruptSha256&&sha(bytes)!==expected.sha256,'actual immutable corrupt adapter bytes required');
    await writeFile(path,bytes);corruptSha256=sha(bytes);
  }
  return {observation:{candidateId:config.candidateId,file:specification.adapterFile,beforeSha256:expected.sha256,state:name,actualSha256:corruptSha256},async restore(){
    if(name==='missing-adapter')await rename(`${path}.moss-terminal-missing`,path);else await writeFile(path,original);
    requireThat((await fileIdentity(path)).sha256===expected.sha256,'owned adapter fixture failed to restore exact bytes');
  }};
}
export async function buildBackend(config) {
  const sources=await sourceIdentity(config),out=resolve(config.buildRoot,config.backend);await mkdir(out,{recursive:true,mode:0o700});
  validatePackagePlan(config,JSON.parse(stripJsoncComments(await readFile(resolve(config.sourceRoot,'config/manifests/builtin.models.jsonc'),'utf8'))));
  await provisionBrowser(config);
  const env={...process.env,CARGO_TARGET_DIR:resolve(config.root,'target'),VITE_API_BASE_URL:'',CARGO_PROFILE_RELEASE_STRIP:'none'};
  const logRoot=resolve(out,'logs');
  await npmCommand(['--prefix','apps/web','ci'],{cwd:config.sourceRoot,env,log:resolve(logRoot,'web-ci.log')});
  await npmCommand(['--prefix','apps/web','run','build'],{cwd:config.sourceRoot,env,log:resolve(logRoot,'web-build.log')});
  const features=config.backend==='cuda'?'embed-web,backend-candle':'embed-web';
  await command('cargo',['build','--release','--locked','-p','sceneworks-rust-api','--bin','sceneworks-rust-api','--features',features],{cwd:config.sourceRoot,env,log:resolve(logRoot,'native-build.log')});
  requireThat(canonical(await sourceIdentity(config))===canonical(sources),'native build changed the frozen source checkouts');
  const binary=resolve(out,config.backend==='cuda'?'sceneworks-rust-api.exe':'sceneworks-rust-api');
  await cp(resolve(env.CARGO_TARGET_DIR,'release',config.backend==='cuda'?'sceneworks-rust-api.exe':'sceneworks-rust-api'),binary,{errorOnExist:true,force:false});
  await save(resolve(out,'web.json'),{files:await filesUnder(resolve(config.sourceRoot,'apps/web/dist'))});
  await isolatePackages(config,{omitAdapter:config.installAdapter===true});
  let installation=null;
  if(config.installAdapter===true) {
    requireThat(config.editAdapterFiles?.length&&config.editAdapterFiles.every(file=>config.packageFiles.some(item=>item.file===file)),'actual isolated adapter-install projection required');
    config.apiBinary=binary;const token=randomBytes(32).toString('hex'),stack=await startStack(config,token);
    try {
      const project=await stack.api.request('/api/v1/projects',{method:'POST',body:{name:`MOSS installer ${config.candidateId}`}});
      const {capturePackageInstallUi}=await import('./moss-edit-terminal-browser.mjs');
      const capture=await capturePackageInstallUi({apiUrl:`http://127.0.0.1:${config.port}`,projectId:project.id,captureRoot:resolve(out,'installation'),playwrightModule:config.playwrightModule,token,model:'moss_image_base'});
      installation={capture:await fileIdentity(capture),files:await filesUnder(resolve(out,'installation'))};
    }finally{await stack.stop();}
  }
  const packages=[];
  for(const item of config.packageFiles) {
    const path=contained(config.hfHome,resolve(config.hfHome,item.file)),identity=await fileIdentity(path);
    requireThat(identity.sha256===item.sha256,'native package bytes differ from frozen model/package identities');packages.push({...item,...identity});
  }
  await save(resolve(out,'packages.json'),{hfHome:config.hfHome,files:packages});
  const toolchain={node:(await execFile(process.execPath,['--version'])).stdout,rust:(await execFile('rustc',['-Vv'])).stdout,
    observationProducerSha256:(await fileIdentity(resolve(config.sourceRoot,'crates/sceneworks-worker/src/moss_terminal_observation.rs'))).sha256,
    installation,browser:await browserIdentity(config),features,sources,buildCommand:['cargo','build','--release','--locked','-p','sceneworks-rust-api','--features',features]};
  await save(resolve(out,'toolchain.json'),toolchain);
  const paths={apiBinary:binary,workerBinary:binary,webPackage:resolve(out,'web.json'),packages:resolve(out,'packages.json'),toolchain:resolve(out,'toolchain.json')},files={};
  for(const name of BACKEND_FILES)files[name]=await fileIdentity(paths[name]);
  await save(resolve(out,'inventory.json'),{backend:config.backend,sources,files});
  await save(resolve(out,'build.json'),{version:VERSION,backend:config.backend,sources,files,paths,device:await nativeDevice(config.backend)});
}
async function harness(config,args,label) {
  await command(process.execPath,[resolve(config.sourceRoot,'scripts/moss-edit-terminal.mjs'),...args],{cwd:config.sourceRoot,env:process.env,log:resolve(config.root,'commands',`${label}.log`)});
}
function buildPaths(config) {
  const out=resolve(config.buildRoot,config.backend),binary=resolve(out,config.backend==='cuda'?'sceneworks-rust-api.exe':'sceneworks-rust-api');
  return {apiBinary:binary,workerBinary:binary,webPackage:resolve(out,'web.json'),packages:resolve(out,'packages.json'),toolchain:resolve(out,'toolchain.json')};
}
async function launcher(config,projectId) {
  const root=resolve(config.root,'evidence'),paths=buildPaths(config),files={...paths};
  for(const [name,path] of Object.entries({harness:'moss-edit-terminal.mjs',harnessLibrary:'lib/moss-edit-terminal.mjs',actionController:'moss-edit-terminal-actions.mjs',actionLibrary:'lib/moss-edit-terminal-actions.mjs',observationValidator:'lib/moss-edit-terminal-observation.mjs',browserDriver:'moss-edit-terminal-browser.mjs'}))files[name]=resolve(config.sourceRoot,'scripts',path);
  files.numericalCarryForward=safePath(config.bundleRoot,config.numericalCarryForward.file);
  files.fixtureBindings=resolve(root,'guard-fixtures/bindings.json');
  files.hostTensorBound=resolve(root,'host-tensor-bound.json');
  requireThat((await fileIdentity(files.numericalCarryForward)).sha256===config.numericalCarryForward.sha256,'numerical source carry-forward artifact changed');
  return {version:VERSION,candidateId:config.candidateId,localBackend:config.backend,plan:resolve(root,'cases.json'),evidenceRoot:root,
    sources:{sceneworks:config.sourceRoot,inference:config.inferenceRoot},inferenceMainRef:'origin/main',files,
    artifactInventories:Object.fromEntries(['mlx','cuda'].map(backend=>[backend,resolve(config.buildRoot,backend,'inventory.json')])),
    backends:{[config.backend]:{apiUrl:`http://127.0.0.1:${config.port}`,projectId,workerId:config.workerId,workerGpuId:config.backend==='cuda'?'1':'mlx',tokenEnv:'MOSS_EDIT_ACTIONS_TOKEN'}},
    guardRequests:config.guardRequests,approvedBy:config.approvedBy,approvalEvidenceSha256:config.approvalEvidenceSha256,phase18ReviewSha256:config.phase18ReviewSha256,finalSourceReviewSha256:config.finalSourceReviewSha256};
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
export async function executeBackend(config) {
  await isolatePackages(config);
  await verifyBuild(config);const paths=buildPaths(config);config.apiBinary=paths.apiBinary;
  // Each execution phase is fresh. An interrupted phase retains immutable attempts;
  // do not start another stack and replay submissions under the same candidate.
  requireThat(!await optional(resolve(config.root,'execution-start.json')),'execution already started; authoritative owned-job reconciliation is required before any continuation');
  await save(resolve(config.root,'execution-start.json'),{version:VERSION,candidateId:config.candidateId,backend:config.backend,sources:await sourceIdentity(config),at:new Date().toISOString()});
  const token=randomBytes(32).toString('hex');process.env.MOSS_EDIT_ACTIONS_TOKEN=token;
  if(config.backend==='cuda') {
    const handoff=await json(resolve(config.handoffRoot,'handoff.json'));config.projectId=handoff.projectId;await importState(config,config.handoffRoot);
    await cp(config.priorEvidenceRoot,resolve(config.root,'evidence'),{recursive:true,errorOnExist:true,force:false});
  }
  const stack=await startStack(config,token);let stopped=false;
  try {
    if(config.backend==='mlx') {
      const project=await stack.api.request('/api/v1/projects',{method:'POST',body:{name:`MOSS edit ${config.candidateId}`}});config.projectId=project.id;
      const original=await json(safePath(config.bundleRoot,config.inputManifest));
      for(const item of original.assets)item.path=safePath(config.bundleRoot,item.path);
      const originalPath=resolve(config.root,'original-inputs.json');await save(originalPath,original);await prepare(originalPath,resolve(config.root,'evidence'));
      const planPath=resolve(config.root,'evidence','cases.json'),plan=await json(planPath);
      plan.cacheComparisonPolicy={metric:'rgb8-mae/v1',tolerance:config.cacheTolerance,implementationSha256:(await fileIdentity(resolve(config.sourceRoot,'scripts/lib/moss-edit-terminal.mjs'))).sha256};
      validatePlan(plan,original);await writeFile(planPath,`${JSON.stringify(plan,null,2)}\n`,{mode:0o600});
      await prepareHostGuard(config);config.guardRequests=await prepareGuardFixtures(config,stack.api);
    } else {
      config.guardRequests=await json(resolve(config.root,'evidence','resolved-guards.json'));
      const project=await stack.api.request(`/api/v1/projects/${encodeURIComponent(config.projectId)}`);requireThat(project.id===config.projectId,'target API did not reopen the original transferred project identity');
      for(const id of ['quality-mlx-base-four','quality-mlx-sft-single']) {
        const origin=await json(resolve(config.root,'evidence','rows',id,'terminal-job.json')),actual=await stack.api.request(`/api/v1/jobs/${encodeURIComponent(origin.id)}`);
        requireThat(canonical(actual.payload)===canonical(origin.payload)&&canonical(actual.result)===canonical(origin.result),'cross-backend history handoff changed original authored job/result identities');
      }
    }
    if(config.backend==='mlx')for(const backend of ['mlx','cuda']) {
      const toolchain=await json(resolve(config.buildRoot,backend,'toolchain.json'));requireThat(toolchain.installation?.files?.length,'actual installer capture missing from native artifact');
      for(const item of toolchain.installation.files)requireThat((await fileIdentity(safePath(resolve(config.buildRoot,backend,'installation'),item.file))).sha256===item.sha256,'native installer evidence bytes changed');
      await cp(resolve(config.buildRoot,backend,'installation'),resolve(config.root,'evidence/external',`raw-install-${backend}`),{recursive:true,errorOnExist:true,force:false});
      await cp(resolve(config.buildRoot,backend,'toolchain.json'),resolve(config.root,'evidence/external',`toolchain-${backend}.json`),{errorOnExist:true,force:false});
    }
    const launch=await launcher(config,config.projectId),launchPath=resolve(config.root,'launch.json');await save(launchPath,launch);
    if(config.backend==='mlx')await freeze(launchPath);else await bind(launchPath);
    const state=await frozen(launchPath),toolchain=await json(paths.toolchain),expectedJournal={candidateId:config.candidateId,sourceSha:config.sourceSha,inferenceSha:config.inferenceSha,workerId:config.workerId,
      gpuId:state.binding.endpoint.workerGpuId,
      binarySha256:state.lock.artifacts[config.backend].files.workerBinary.sha256,producerSha256:toolchain.observationProducerSha256};
    await actualJournal(config,expectedJournal);
    await harness(config,['upload',launchPath,config.backend,await deviceReceipt(config,state,'upload')],'upload');
    const samplerSha256=(await fileIdentity(resolve(config.sourceRoot,'scripts/lib/moss-edit-terminal-actions.mjs'))).sha256;
    for(const row of state.plan.rows.filter(row=>row.backend===config.backend)) {
      await assertDispatchClear(state.root,config.backend);
      await appendFile(resolve(config.root,'progress.jsonl'),`${JSON.stringify({rowId:row.id,phase:'before',at:new Date().toISOString()})}\n`,{mode:0o600});
      const measurement=await sampler({backend:config.backend,file:resolve(config.root,'samples',`${row.id}.jsonl`)});
      let capture;
      try {
        const admission=await deviceReceipt(config,state,row.id);
        if(row.transport.startsWith('ui-')) {
          await harness(config,['ui-task',launchPath,row.id],`${row.id}-intent`);
          const {captureUi}=await import('./moss-edit-terminal-browser.mjs');
          capture=await captureUi({launchPath,rowId:row.id,captureRoot:resolve(config.root,'captures',row.id),playwrightModule:config.playwrightModule,token});
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
        await cp(resolve(config.root,'samples',`${row.id}.jsonl`),rawPath,{errorOnExist:true,force:false});memory.raw={file:'memory-raw.jsonl',...await fileIdentity(rawPath)};
        await save(resolve(state.root,'rows',row.id,'memory.json'),memory);
        if(row.anchors)await retainQualityPairs(state,row);
        if(row.compareTo)await harness(config,['compare-cache',launchPath,row.id],`${row.id}-cache`);
        await appendFile(resolve(config.root,'progress.jsonl'),`${JSON.stringify({rowId:row.id,phase:'verified',jobId:receipt.jobId,at:new Date().toISOString()})}\n`);
      }catch(error){await measurement.stop('UNRESOLVED',samplerSha256).catch(()=>{});throw error;}
    }
    for(const name of state.plan.requiredGuardCases) {
      requireThat(launch.guardRequests[name].noProviderLoadProbe===OBSERVATION_METHOD,'every refusal must freeze the actual source-bound MOSS journal probe');
      let coldRestart=null;
      if(name==='device-budget-before-load'&&launch.guardRequests[name].expectedTerminal==='failed') {
        coldRestart=await stack.restartIdleWorker('allocated-device-budget-cold-refusal');await actualJournal(config,expectedJournal);
      }
      const mutation=await adapterRefusalState(config,name,launch.guardRequests[name]);
      try {
      const before=await actualJournal(config,expectedJournal),beforeAt=new Date().toISOString();
      await harness(config,['guard',launchPath,name,config.backend,await deviceReceipt(config,state,`guard-${name}`)],`guard-${name}`);
      const afterAt=new Date().toISOString(),after=await actualJournal(config,expectedJournal,afterAt),directory=resolve(state.root,'guards',`${config.backend}-${name}`);
      await writeFile(resolve(directory,'journal-before.jsonl'),before,{flag:'wx',mode:0o600});await writeFile(resolve(directory,'journal-after.jsonl'),after,{flag:'wx',mode:0o600});await cp(paths.toolchain,resolve(directory,'toolchain.json'),{errorOnExist:true,force:false});
      const receipt=await json(resolve(directory,'receipt.json')),rawEvidence=[];
      if(coldRestart)await save(resolve(directory,'cold-worker-restart.json'),coldRestart);
      if(mutation.observation)await save(resolve(directory,'adapter-state.json'),mutation.observation);
      for(const file of ['journal-before.jsonl','journal-after.jsonl','toolchain.json'])rawEvidence.push({file,...await fileIdentity(resolve(directory,file))});
      if(mutation.observation)rawEvidence.push({file:'adapter-state.json',...await fileIdentity(resolve(directory,'adapter-state.json'))});
      if(coldRestart)rawEvidence.push({file:'cold-worker-restart.json',...await fileIdentity(resolve(directory,'cold-worker-restart.json'))});
      await save(resolve(directory,'no-provider-load.json'),{version:VERSION,lockSha256:digest(state.lock),case:name,backend:config.backend,requestSha256:receipt.requestSha256,method:OBSERVATION_METHOD,
        observer:'actual source-bound instrumented native worker',beforeAt,afterAt,jobId:receipt.jobId??null,beforeFile:'journal-before.jsonl',afterFile:'journal-after.jsonl',toolchainFile:'toolchain.json',rawEvidence,observation:noLoadComparison(before,after,expectedJournal,{beforeAt,afterAt,jobId:receipt.jobId??null})});
      }finally{await mutation.restore();}
    }
    // Stop only the processes this controller spawned. No global census/ownership gate.
    await stack.stop();stopped=true;
    if(config.backend==='mlx')await exportState(config,resolve(config.root,'handoff'));
    await save(resolve(config.root,'execution.json'),{version:VERSION,lockSha256:digest(state.lock),backend:config.backend,renders:25,refusals:29,execution:'COMPLETE',acceptance:'AWAITING_ALL_EXTERNAL_EVIDENCE_AND_VISUAL_REVIEW'});
  }finally{if(!stopped)await stack.stop();delete process.env.MOSS_EDIT_ACTIONS_TOKEN;}
}
export async function finalize(config) {
  // Root-owned actual reviews/observations arrive as immutable evidence files,
  // not synthetic PASS receipts generated by this controller.
  requireThat(config.reviewRoot,'actual external/visual review artifact root required');
  await cp(config.priorEvidenceRoot,resolve(config.root,'evidence'),{recursive:true,errorOnExist:true,force:false});
  const root=resolve(config.root,'evidence'),binding=await json(resolve(root,'bindings',`${config.backend}.json`));
  config.guardRequests=await json(resolve(root,'resolved-guards.json'));
  const launch=await launcher(config,binding.endpoint.projectId),launchPath=resolve(config.root,'review-launch.json');await save(launchPath,launch);
  for(const entry of await filesUnder(config.reviewRoot)) {
    const destination=safePath(root,entry.file);await mkdir(dirname(destination),{recursive:true});await cp(safePath(config.reviewRoot,entry.file),destination,{errorOnExist:true,force:false});
  }
  await harness(config,['verify-retained',launchPath],'final-verify');
}
export async function main(args=process.argv.slice(2)) {
  const [phase,...rest]=args;
  if(phase==='arrange-builds'&&rest.length===1)return arrangeBuilds(rest[0]);
  if(phase==='validate-bundle'&&rest.length===5)return validateBundle(...rest);
  if(phase==='configure-env'&&rest.length===0) {
    const env=process.env;await validateBundle(env.MOSS_BUNDLE,env.MOSS_BUNDLE_SHA,env.MOSS_SOURCE_SHA,env.MOSS_INFERENCE_SHA,env.MOSS_CANDIDATE);
    return configure(resolve(env.MOSS_BUNDLE,'campaign.json'),env.MOSS_SOURCE,env.MOSS_INFERENCE,env.MOSS_ROOT,env.MOSS_BACKEND,env.MOSS_BUILDS);
  }
  if(phase==='configure'&&rest.length===6)return configure(...rest);
  requireThat(rest.length===1,'Usage: validate-bundle ROOT SHA SOURCE INFERENCE CANDIDATE | configure BUNDLE SOURCE INFERENCE ROOT BACKEND BUILDS | configure-env | arrange-builds DOWNLOAD_ROOT | build CONFIG | provision-browser CONFIG | execute CONFIG | finalize CONFIG');
  const config=await json(rest[0]);
  if(phase==='build')return buildBackend(config);
  if(phase==='provision-browser')return provisionBrowser(config);
  if(phase==='execute')return executeBackend(config);
  if(phase==='finalize')return finalize(config);
  throw Error('unknown MOSS terminal Actions phase');
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch(error=>{process.stderr.write(`${JSON.stringify({status:'STOP',message:error.message})}\n`);process.exitCode=1;});
