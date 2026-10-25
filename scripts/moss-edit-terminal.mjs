#!/usr/bin/env node
// Dedicated edit protocol. Never invoke the historical sc-26325/GPU0 harness.
import { constants } from 'node:fs';
import { readFile, mkdir, writeFile, copyFile, chmod, readdir } from 'node:fs/promises';
import { resolve, dirname, basename } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { VERSION, sha, digest, canonical, requireThat, json, save, safePath, fileIdentity, treeIdentity, verifyFrozenSources, validateAdmission, makeNpy, inspectNpy, pngRgbIdentity, validateAdvancedDigests, createPlan, validatePlan, Api, Mcp, mcpArgs, mcpJson, replaceInputs, assertRequestBinding, checkJob, findEditReport, validateEditReport, memorySummary, reviewVerdict, isDenoiseStep, validateExportedWorkflow, mcpResultAssets, cacheComparison, assertDispatchClear, reserveAttempt, resumeUiAttempt, resolveAttempt, failAttempt, reconcileAttempt, LATENT_UPSCALE_GUARDS, latentUpscaleGuard, validateSeededRecipeSelection } from './lib/moss-edit-terminal.mjs';
import { BACKEND_FILES, validateArtifactInventory, executionBinding, validateBinding, validateEditorSource } from './lib/moss-edit-terminal.mjs';
import { OBSERVATION_METHOD, noLoadComparison, hostTensorInvariant, hostGuardSpec, hostBudgetHeader } from './lib/moss-edit-terminal-observation.mjs';
import { assertOriginalEvidence, assertH18CheckpointFiles, validateFailedUiCapture, R16_ORIGINAL_INVENTORY_SHA256, H18_RETRY_TOOLING_SHA, H18_RETRY_RUN_ID, H18_RETRY_ARTIFACT_ID, H18_RETRY_INVENTORY_SHA256 } from './lib/moss-edit-terminal-actions.mjs';

const sleep = ms => new Promise(resolveSleep => setTimeout(resolveSleep, ms));
const encode = value => encodeURIComponent(value);
const jobPath = id => `/api/v1/jobs/${encode(id)}`;
const assetPath = (project, asset) => `/api/v1/projects/${encode(project)}/assets/${encode(asset)}`;
const mediaPath = (project, path) => `/api/v1/projects/${encode(project)}/files/${path.split('/').map(encode).join('/')}`;
function noReceipts(value) {
  if (!value || typeof value !== 'object') return;
  for (const [key, child] of Object.entries(value)) {
    requireThat(!/ExecutionReport$/.test(key) && key !== 'mossImageSampleIndex', 'replay imported server-only report/sample index');
    noReceipts(child);
  }
}
// Preserve raw local evidence in a restricted directory. Console/portable summary contains IDs,
// hashes and verdicts only; MCP media tickets and bearer credentials are never persisted.
function scrubMcp(value) {
  if (typeof value === 'string') {
    if (value.trimStart().startsWith('{') || value.trimStart().startsWith('[')) { try { return JSON.stringify(scrubMcp(JSON.parse(value))); } catch {} }
    return value.replace(/https?:\/\/[^\s"<>]+/g, item => { try { const url = new URL(item); url.search = ''; return url.href; } catch { return '[redacted media URL]'; } });
  }
  if (Array.isArray(value)) return value.map(scrubMcp);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).filter(([key]) => !['data', 'blob'].includes(key)).map(([key, child]) => [key, scrubMcp(child)]));
  return value;
}
export async function prepare(inputPath, out) {
  const inputs = await json(inputPath);
  await mkdir(out, { mode: 0o700 });
  await mkdir(resolve(out, 'inputs'), { mode: 0o700 });
  const identities = {};
  for (const item of inputs.assets) {
    requireThat((await fileIdentity(item.path)).sha256 === item.sha256, `original ${item.id} hash mismatch`);
    const name = `inputs/${item.id}.png`; await copyFile(item.path, resolve(out, name)); await chmod(resolve(out,name),0o600);
    identities[item.id] = { file: name, sha256: item.sha256, kind: 'image', sourceReceiptSha256: item.receipt_sha256, ...pngRgbIdentity(await readFile(item.path)) };
  }
  const tensors = {
    positive: makeNpy([2, 3, 4096], 'float32', i => ((i * 17 % 127) - 63) / 1024),
    negative: makeNpy([2, 3, 4096], 'float32', i => ((i * 29 % 113) - 56) / 2048),
    mask: makeNpy([2, 3], 'uint8', i => [1, 1, 0, 1, 1, 1][i]),
    negativeMask: makeNpy([2, 3], 'uint8', i => [1, 0, 1, 0, 1, 0][i]),
    noise: makeNpy([4, 128, 4, 4], 'float32', i => ((i * 73 % 1021) - 510) / 257),
  };
  for (const [id, bytes] of Object.entries(tensors)) {
    const file = `inputs/${id}.npy`; await writeFile(resolve(out, file), bytes, { flag: 'wx', mode: 0o600 });
    identities[id] = { file, kind: 'tensor', ...inspectNpy(bytes) };
  }
  await save(resolve(out, 'inputs.original.json'), inputs);
  await save(resolve(out, 'input-identities.json'), identities);
  const plan = createPlan(inputs);
  // Causal execution order within each backend: quality -> advanced/replays -> cancel ->
  // reentry -> reference transitions -> cache comparison -> ordinary/DMD -> return edit.
  await save(resolve(out, 'cases.json'), plan);
  await save(resolve(out, 'launch.example.json'), {
    version: VERSION, candidateId:'ACTUAL-COORDINATOR-CANDIDATE', localBackend: 'mlx', plan: resolve(out, 'cases.json'), evidenceRoot: out,
    artifactInventories: { mlx: '/ABSOLUTE/SOURCE-BOUND/MLX-BUILD-INVENTORY.json', cuda: '/ABSOLUTE/SOURCE-BOUND/CUDA-BUILD-INVENTORY.json' },
    sources: { sceneworks: '/ABSOLUTE/CLEAN/FINAL/SCENEWORKS', inference: '/ABSOLUTE/CLEAN/FINAL/INFERENCE' },
    inferenceMainRef: 'origin/main',
    files: { harness: fileURLToPath(import.meta.url), harnessLibrary: fileURLToPath(new URL('./lib/moss-edit-terminal.mjs', import.meta.url)), apiBinary: '/ABSOLUTE/BUILT/API', workerBinary: '/ABSOLUTE/BUILT/WORKER', webPackage: '/ABSOLUTE/WEB/BUILD/INVENTORY.json', packages: '/ABSOLUTE/VERIFIED/PACKAGE/INVENTORY.json', toolchain: '/ABSOLUTE/ACTUAL/TOOLCHAIN.json', numericalCarryForward: '/ABSOLUTE/SOURCE/BOUND/NUMERICAL/RECEIPT.json' },
    backends: { mlx: { apiUrl: 'http://127.0.0.1:8000', projectId: 'ACTUAL-ISOLATED-PROJECT', workerId:'ACTUAL-OWNED-WORKER', workerGpuId:'mlx', tokenEnv: 'MOSS_TERMINAL_API_TOKEN' }, cuda: { apiUrl: 'http://ADMITTED-WINDOWS-HOST:8000', projectId: 'ACTUAL-ISOLATED-PROJECT', workerId:'ACTUAL-OWNED-WORKER', workerGpuId:'1', tokenEnv: 'MOSS_TERMINAL_API_TOKEN' } },
    guardRequests: Object.fromEntries(LATENT_UPSCALE_GUARDS.map(name => [name, latentUpscaleGuard(name)])), approvedBy: '', approvalEvidenceSha256: '', phase18ReviewSha256: '', inferencePreMergeReviewSha256: '',
  });
  console.log(JSON.stringify({ status: 'DRAFT', version: VERSION, renderExecutions: plan.renderExecutions, guards: plan.requiredGuardCases.length, next: 'Complete launcher, cache metric, source reviews, package/toolchain inventories and coordinator freeze before any execution.' }));
}
async function load(configPath) {
  const config = await json(configPath), root = resolve(config.evidenceRoot);
  requireThat(config.version === VERSION, 'wrong launcher version');
  if (config.repair) {
    requireThat(config.repair.name==='r16-checker-recovery'&&/^[a-f0-9]{64}$/.test(config.repair.inventorySha256??''),'only the original bounded R16 checker repair may relax relocated execution paths');
    const tooling = config.repair.tooling;
    requireThat(tooling?.sourceRoot && /^[a-f0-9]{40}$/.test(tooling.head ?? '') && canonical(treeIdentity(tooling.sourceRoot)) === canonical(tooling.source) && tooling.head === tooling.source.head, 'continuation tooling checkout changed');
    requireThat(resolve(tooling.files?.harness) === fileURLToPath(import.meta.url) && resolve(tooling.files?.harnessLibrary) === fileURLToPath(new URL('./lib/moss-edit-terminal.mjs',import.meta.url)) && resolve(tooling.files?.browserDriver) === fileURLToPath(new URL('./moss-edit-terminal-browser.mjs',import.meta.url)) && resolve(tooling.files?.browserControls) === fileURLToPath(new URL('./lib/moss-edit-terminal-controls.mjs',import.meta.url)), 'continuation did not execute its declared tooling');
    for (const [name, path] of Object.entries(tooling.files)) requireThat(canonical(await fileIdentity(path)) === canonical(tooling.identities[name]), `continuation tooling ${name} changed`);
    requireThat(resolve(config.files.harness) === resolve(config.sources.sceneworks,'scripts/moss-edit-terminal.mjs') && resolve(config.files.harnessLibrary) === resolve(config.sources.sceneworks,'scripts/lib/moss-edit-terminal.mjs'), 'continuation displaced frozen production protocol files');
  } else requireThat(resolve(config.files.harness) === fileURLToPath(import.meta.url) && resolve(config.files.harnessLibrary) === fileURLToPath(new URL('./lib/moss-edit-terminal.mjs',import.meta.url)), 'local binding must identify the protocol source actually executing');
  return { config, root, plan: await json(config.plan) };
}
export function validateLexicalNonfinite(rawJson) {
  requireThat(typeof rawJson==='string'&&Buffer.byteLength(rawJson)<=1024*1024,'bounded exact lexical JSON required');
  const matches=[...rawJson.matchAll(/"training_shift"\s*:\s*(?:NaN|[+-]?Infinity)(?=\s*[,}])/g)];
  requireThat(matches.length===1,'exact training_shift nonfinite lexical token required');
  const match=matches[0],valid=JSON.parse(rawJson.slice(0,match.index)+'"training_shift":1'+rawJson.slice(match.index+match[0].length));
  requireThat(valid.advanced?.mossImage?.training_shift===1,'lexical refusal must target actual advanced.mossImage.training_shift');
}
export async function freeze(configPath) {
  const { config, root, plan } = await load(configPath);
  requireThat(typeof config.candidateId === 'string' && /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$/.test(config.candidateId),'actual stable campaign candidate ID required');
  validatePlan(plan, await json(resolve(root, 'inputs.original.json')));
  requireThat(config.approvedBy && [config.approvalEvidenceSha256, config.phase18ReviewSha256, config.inferencePreMergeReviewSha256].every(v => /^[a-f0-9]{64}$/.test(v ?? '')), 'coordinator approval, integrated requirements review and inference pre-merge source review identities required');
  for (const name of plan.requiredGuardCases) {
    const guard = config.guardRequests[name];
    if (LATENT_UPSCALE_GUARDS.includes(name)) {
      const exact = latentUpscaleGuard(name);
      requireThat(canonical(guard?.request) === canonical(exact.request) && canonical(guard.refs) === canonical(exact.refs) && guard.expectedStatus === exact.expectedStatus && canonical(guard.errorIncludes) === canonical(exact.errorIncludes), 'freeze the exact valid latent+enabled inline-upscale request/actionable refusal');
    }
    const worker=guard?.expectedTerminal!==undefined;
    requireThat(!worker||(['device-budget-before-load','corrupt-adapter'].includes(name)&&guard.expectedTerminal==='failed'&&guard.expectedStatus===201&&guard.rawJson===undefined),'only the allocated device-budget/corrupt-adapter guards may observe real queued worker refusals');
    if(guard?.upload) {
      requireThat(name==='host-budget-refusal'&&canonical(guard)===canonical(hostGuardSpec()),'only the exact source-owned host slot may use the per-artifact import budget route');
      requireThat(resolve(config.files.hostTensorBound)===safePath(root,guard.invariant),'shared candidate must freeze the actual source-derived host invariant receipt');
      requireThat(canonical(await json(safePath(root,guard.invariant)))===canonical(await hostTensorInvariant(config.sources.sceneworks)),'recompute the host aggregate invariant against integrated source before freeze');
      requireThat((await readFile(safePath(root,guard.upload.file))).equals(hostBudgetHeader()),'retain the exact unsupported oversize declaration header');
    }
    requireThat(guard && (guard.request || typeof guard.rawJson === 'string'||guard.upload) && (worker||[400, 403, 404, 409, 413, 422, 503].includes(guard.expectedStatus)) && guard.errorIncludes?.length && guard.noProviderLoadProbe, `freeze actual product refusal request/error/no-load probe for ${name}`);
    if (guard.rawJson !== undefined) {requireThat(name==='training-shift-nonfinite'&&!guard.request,'only the allocated lexical scalar refusal may use raw JSON; tensor nonfinite must use actual NPY');validateLexicalNonfinite(guard.rawJson);}
  }
  const files = {}, sources = { sceneworks: treeIdentity(config.sources.sceneworks), inference: treeIdentity(config.sources.inference) }, artifacts = {};
  for (const name of ['harness', 'harnessLibrary', 'apiBinary', 'workerBinary', 'webPackage', 'packages', 'toolchain', 'numericalCarryForward']) requireThat(config.files[name], `missing ${name}`);
  requireThat(resolve(config.files.harness) === fileURLToPath(import.meta.url) && resolve(config.files.harnessLibrary) === fileURLToPath(new URL('./lib/moss-edit-terminal.mjs',import.meta.url)), 'freeze the source actually executing this protocol');
  for (const [name, path] of Object.entries(config.files)) files[name] = await fileIdentity(path);
  for (const backend of ['mlx','cuda']) artifacts[backend] = validateArtifactInventory(await json(config.artifactInventories[backend]),backend,sources);
  requireThat(plan.cacheComparisonPolicy.implementationSha256 === files.harnessLibrary.sha256,'cache metric must freeze the actual source-owned implementation');
  const sharedFiles = Object.fromEntries(Object.entries(files).filter(([name]) => !BACKEND_FILES.includes(name)));
  const lock = { version: VERSION, candidateId:config.candidateId, createdAt: new Date().toISOString(), planSha256: digest(plan), guardRequestsSha256: digest(config.guardRequests), inputIdentitiesSha256: digest(await json(resolve(root, 'input-identities.json'))), sources, files:sharedFiles, artifacts, approvals: { by: config.approvedBy, evidence: config.approvalEvidenceSha256, phase18Review: config.phase18ReviewSha256, inferencePreMergeReview: config.inferencePreMergeReviewSha256 }, counts: { renders: plan.renderExecutions, guards: plan.requiredGuardCases.length*2, guardCasesPerBackend:plan.requiredGuardCases.length } };
  await verifyFrozenSources(config, lock);
  executionBinding(config,lock,files); // Validate the actual local endpoint before writing the immutable lock.
  await save(resolve(root, 'protocol.lock.json'), lock);
  await bind(configPath);
  console.log(JSON.stringify({ status: 'FROZEN_NOT_EXECUTED', lockSha256: digest(lock), counts: lock.counts }));
}
export async function bind(configPath) {
  const {config,root} = await load(configPath), lock = await json(resolve(root,'protocol.lock.json'));
  await verifyFrozenSources(config,lock);
  requireThat(digest(await json(resolve(root,'input-identities.json'))) === lock.inputIdentitiesSha256, 'input identity manifest changed');
  for (const input of Object.values(await json(resolve(root,'input-identities.json')))) requireThat((await fileIdentity(safePath(root,input.file))).sha256 === input.sha256,'frozen original input bytes changed');
  const files = {};
  for (const name of Object.keys({...lock.files,...lock.artifacts[config.localBackend].files})) files[name] = await fileIdentity(config.files[name]);
  const binding = executionBinding(config,lock,files); validateBinding(binding,lock,config.localBackend);
  await save(resolve(root,'bindings',`${config.localBackend}.json`),binding);
  return binding;
}
export async function frozen(configPath, { retained = false } = {}) {
  const state = await load(configPath), lock = await json(resolve(state.root, 'protocol.lock.json'));
  requireThat(digest(await json(resolve(state.root, 'input-identities.json'))) === lock.inputIdentitiesSha256, 'input identity manifest changed');
  for (const input of Object.values(await json(resolve(state.root,'input-identities.json')))) requireThat((await fileIdentity(safePath(state.root,input.file))).sha256 === input.sha256,'frozen original input bytes changed');
  validatePlan(state.plan, await json(resolve(state.root, 'inputs.original.json')));
  await verifyFrozenSources(state.config, lock);
  if(state.config.files.fixtureBindings)for(const fixture of Object.values(await json(state.config.files.fixtureBindings)))if(fixture?.asset) {
    requireThat((await fileIdentity(safePath(state.root,fixture.uploadedFile))).sha256===fixture.uploadedSha256&&(await fileIdentity(safePath(state.root,fixture.actualFile))).sha256===fixture.actualSha256,'retained actual guard fixture upload/readback bytes changed');
  }
  const binding = validateBinding(await json(resolve(state.root,'bindings',`${state.config.localBackend}.json`)),lock,state.config.localBackend);
  if (retained || state.config.repair) requireThat(canonical(state.config.backends[state.config.localBackend])===canonical(binding.endpoint),'retained verification cannot substitute original project/worker/API identities');
  if (!retained && !state.config.repair) requireThat(canonical(executionBinding(state.config,lock,binding.files)) === canonical(binding),'local execution binding changed; never overwrite a binding');
  // Read-only artifact verification may relocate retained bytes. Source/artifact
  // hashes and the original immutable execution bindings remain mandatory.
  return { ...state, lock, binding };
}
function client(config, backend) { const b = config.backends[backend]; requireThat(b?.projectId && b.apiUrl, 'missing actual backend/project'); return new Api(b.apiUrl, process.env[b.tokenEnv] ?? ''); }
async function admission(path, state, backend) {
  requireThat(backend === state.config.localBackend, 'submission must use the verified local backend binding');
  const value = await json(path); validateAdmission(value, state.lock, backend);
  return value;
}
async function upload(configPath, backend, admissionPath) {
  const state = await frozen(configPath), admitted = await admission(admissionPath, state, backend);
  const { config, root } = state, api = client(config, backend), project = config.backends[backend].projectId;
  const target = resolve(root, `uploaded-${backend}.json`);
  const ticket = await reserveAttempt(root,backend,'upload',{lockSha256:digest(state.lock),bindingSha256:digest(state.binding),requestSha256:state.lock.inputIdentitiesSha256});
  const attempt = resolve(root, `upload-${backend}`);
  try {
  await mkdir(attempt, { mode: 0o700 });
  await save(resolve(attempt, 'admission.json'), admitted);
  const assets = {};
  for (const [id, item] of Object.entries(await json(resolve(root, 'input-identities.json')))) {
    await admission(admissionPath, state, backend);
    const bytes = await readFile(safePath(root, item.file)); requireThat(sha(bytes) === item.sha256, `frozen input ${id} changed`);
    const form = new FormData(); form.append('file', new Blob([bytes], { type: item.kind === 'tensor' ? 'application/x-npy' : 'image/png' }), basename(item.file));
    const asset = await api.request(`/api/v1/projects/${encode(project)}/assets`, { method: 'POST', body: form });
    requireThat(asset.id && asset.projectId === project, 'upload not bound to actual project');
    if (item.kind === 'tensor') requireThat(asset.type === 'tensor' && asset.tensor.sha256 === item.sha256 && canonical(asset.tensor.shape) === canonical(item.shape), 'import changed original tensor identity');
    assets[id] = asset;
    await save(resolve(attempt, `${id}.json`), asset);
  }
  await verifyFrozenSources(config, state.lock); await save(target, assets);
  await resolveAttempt(ticket,'UPLOAD_COMPLETE');
  console.log(JSON.stringify({ status: 'UPLOADED', backend, assets: Object.fromEntries(Object.entries(assets).map(([id, a]) => [id, a.id])) }));
  } catch(error) { await failAttempt(ticket,error); throw error; }
}
async function mcpClient(config, backend, transcript) {
  const b = config.backends[backend];
  const mcp = new Mcp(b.apiUrl, process.env[b.tokenEnv] ?? '', message => transcript.push({ at: new Date().toISOString(), message: scrubMcp(message) }));
  await mcp.initialize(); return mcp;
}
async function expectedRequest(state, row) {
  const binding = validateBinding(await json(resolve(state.root,'bindings',`${row.backend}.json`)),state.lock,row.backend);
  return replaceInputs(row, await json(resolve(state.root, `uploaded-${row.backend}.json`)), binding.endpoint.projectId);
}
async function uiTask(configPath, id) {
  const state = await frozen(configPath), row = state.plan.rows.find(r => r.id === id);
  requireThat(row?.transport.startsWith('ui-'), 'not an allocated UI row');
  requireThat(row.backend === state.config.localBackend, 'UI task must use its actual local execution binding');
  const previous=state.plan.rows.filter(r=>r.backend===row.backend).slice(0,state.plan.rows.filter(r=>r.backend===row.backend).findIndex(r=>r.id===id));
  for(const earlier of previous)requireThat((await json(resolve(state.root,'rows',earlier.id,'receipt.json'))).execution==='VERIFIED',`complete/reconcile predecessor ${earlier.id} before UI submission`);
  const request=await expectedRequest(state,row);
  await reserveAttempt(state.root,row.backend,row.id,{transport:'ui',lockSha256:digest(state.lock),bindingSha256:digest(state.binding),requestSha256:digest(request)});
  console.log(JSON.stringify({ status: 'AWAITING_REAL_UI', rowId: id, transport: row.transport, bindingSha256:digest(state.binding), expectedRequest: request, requiredCapture: ['before.png showing selected model/adapter and ordered original references/tensors/controls', 'after.png showing the actual submitted job/result', 'capture.json: version,rowId,lockSha256,bindingSha256,observer,capturedAt,network:{method,path,observedAt,request,response:{id}},screenshots:[{phase,file,sha256}],progress:[actual snapshots],actions:[actual observed actions]', ...(row.transport === 'ui-editor' ? ['sourceUpload:{inputId:B,file,sha256,observedAt,network:{method:POST,path:/api/v1/projects/PROJECT/assets,request:{fileSha256},response:{id}}}; actual unchanged opaque original scratch PNG; history:{originJobId,originAssetId,originSampleIndex:0,hydrationObserved:true,selectedRecipe}; authenticated target API read-back must preserve original job/asset/recipe identity'] : []), ...(row.action === 'seeded-recipe' ? ['Open origin sample index2 (conditioning row two); retain selectedAssetId/selectedRecipe; actual viewer Use this recipe with Keep seed; history:{originRowId,originJobId,originSampleIndex:2,selectedAssetId,selectedRecipe,hydrationObserved:true,keepSeed:true}; restore prompt0/whole B2/count2/base seed173'] : []), ...(row.action === 'workflow' || id === 'quality-cuda-base-four' ? ['workflow export .sceneworks-workflow.json, inspect/resolution response and actual ordered reacquisition observations'] : [])] }, null, 2));
}
export async function validateRetainedEditor(state,row,expected,capture,dest) {
  const proof = await json(resolve(dest,'source-binding.json'));
  const binding = validateBinding(await json(resolve(state.root,'bindings',`${row.backend}.json`)),state.lock,row.backend);
  requireThat(proof.version === VERSION && proof.lockSha256 === digest(state.lock) && proof.bindingSha256 === digest(binding) && proof.rowId === row.id && proof.readback.apiUrl === new URL(binding.endpoint.apiUrl).href && proof.readback.authenticated === true && Date.parse(proof.readback.at) >= Date.parse(capture.network.observedAt), 'Editor source proof lacks candidate-bound authenticated read-back');
  const required = ['source-upload.png','source-served.png','source-asset.json','history-job.json','history-asset.json'];
  requireThat(canonical(proof.files.map(item=>item.file).sort()) === canonical([...required].sort()), 'Editor source proof omitted actual retained upload/history evidence');
  for (const item of proof.files) requireThat((await fileIdentity(safePath(dest,item.file))).sha256 === item.sha256,'Editor source proof bytes changed');
  const origin = await json(resolve(state.root,'rows','quality-mlx-sft-single','receipt.json')), intent = await json(resolve(state.root,'attempts',row.backend,`${row.id}.start.json`));
  requireThat(proof.captureSha256 === digest(capture) && proof.intentSha256 === digest(intent),'Editor capture/reserved intent changed after authenticated read-back');
  const result = validateEditorSource({ row,expected,capture,intent, original:(await json(resolve(state.root,'input-identities.json'))).B,
    uploadBytes:await readFile(resolve(dest,'source-upload.png')),servedBytes:await readFile(resolve(dest,'source-served.png')),asset:await json(resolve(dest,'source-asset.json')),
    origin,originJob:await json(resolve(state.root,'rows','quality-mlx-sft-single','terminal-job.json')),originAsset:await json(resolve(state.root,'rows','quality-mlx-sft-single','asset-0.json')),
    historyJob:await json(resolve(dest,'history-job.json')),historyAsset:await json(resolve(dest,'history-asset.json')) });
  requireThat(canonical(result) === canonical(proof.source), 'Editor source binding changed after authenticated adoption');
  return result;
}
export async function retainEditorSource(capturePath,state,row,expected,dest,capture,api = client(state.config,row.backend)) {
  requireThat(api.token, 'Editor source/history read-back requires actual product authentication');
  const upload = capture.sourceUpload; requireThat(upload?.file && upload.network?.response?.id,'actual Editor scratch upload capture required');
  const asset = await api.request(assetPath(expected.projectId,upload.network.response.id));
  safePath(state.root,asset.file?.path);
  const bytes = await api.request(mediaPath(expected.projectId,asset.file.path),{raw:true,max:128*1024*1024});
  const origin = await json(resolve(state.root,'rows','quality-mlx-sft-single','receipt.json'));
  const originAsset = await json(resolve(state.root,'rows','quality-mlx-sft-single','asset-0.json'));
  const historyJob = await api.request(jobPath(origin.jobId));
  const historyAsset = await api.request(assetPath(originAsset.projectId,originAsset.id));
  await writeFile(resolve(dest,'source-upload.png'),await readFile(safePath(dirname(capturePath),upload.file)),{flag:'wx',mode:0o600});
  await writeFile(resolve(dest,'source-served.png'),bytes,{flag:'wx',mode:0o600});
  await save(resolve(dest,'source-asset.json'),asset); await save(resolve(dest,'history-job.json'),historyJob); await save(resolve(dest,'history-asset.json'),historyAsset);
  const args = { row,expected,capture,intent:await json(resolve(state.root,'attempts',row.backend,`${row.id}.start.json`)),original:(await json(resolve(state.root,'input-identities.json'))).B,
    uploadBytes:await readFile(resolve(dest,'source-upload.png')),servedBytes:bytes,asset,origin,originJob:await json(resolve(state.root,'rows','quality-mlx-sft-single','terminal-job.json')),originAsset,historyJob,historyAsset };
  const files = [];
  for (const file of ['source-upload.png','source-served.png','source-asset.json','history-job.json','history-asset.json']) files.push({file,...await fileIdentity(resolve(dest,file))});
  await save(resolve(dest,'source-binding.json'),{version:VERSION,lockSha256:digest(state.lock),bindingSha256:digest(state.binding),rowId:row.id,captureSha256:digest(capture),intentSha256:digest(args.intent),readback:{apiUrl:api.base.href,authenticated:true,at:new Date().toISOString()},files,source:validateEditorSource(args)});
  await validateRetainedEditor(state,row,expected,capture,dest);
}
export function validatePortableCapture(workflow,bytes,origin,report,sampleIndex) {
  const inspection=workflow.inspection,network=workflow.inspectionNetwork;
  const browser=workflow.browserFile,wire=network?.request,exactWire=wire?.observer==='playwright-request-post-data'&&wire.fileSha256===sha(bytes)&&wire.byteLength===bytes.length;
  const omittedWire=wire?.observer==='playwright-request-metadata-and-product-response'&&wire.fileSha256===null&&wire.byteLength===null&&wire.postDataState==='file-bytes-unavailable-in-observer'&&wire.contentType==='multipart/form-data';
  requireThat(inspection?.status==='workflow'&&inspection.workflow&&network?.method==='POST'&&network.path==='/api/v1/workflows/inspect'&&network.status===200&&canonical(network.response)===canonical(inspection)&&browser?.sha256===sha(bytes)&&browser.byteLength===bytes.length&&workflow.inspectUploadSha256===browser.sha256&&Date.parse(network.observedAt)>=Date.parse(browser.startedAt)&&(exactWire||omittedWire),'portable replay must bind the real browser File and successful product inspection to the unchanged downloaded bytes');
  if(workflow.file.endsWith('.png'))requireThat(bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10]))&&sha(bytes)===origin.outputs[sampleIndex].sha256,'portable PNG is not the actual original output with its embedded workflow');
  else requireThat(workflow.file.endsWith('.sceneworks-workflow.json')&&canonical(JSON.parse(bytes))===canonical(inspection.workflow),'JSON export differs from actual product inspection');
  validateExportedWorkflow(inspection.workflow,report,sampleIndex);
}
function uiGuards(row) {
  return [...((row.action==='workflow'||row.id==='quality-cuda-base-four')?['workflow-unresolved']:[]),...(row.id===`advanced-${row.backend}-sft-replay`?LATENT_UPSCALE_GUARDS:[])];
}
function uiGuard(capture,name) {
  const observation=capture.actions.find(item=>item.action==='no-post-guard'&&item.name===name);
  requireThat(observation?.disabled===true&&Number.isSafeInteger(observation.postCountBefore)&&observation.postCountBefore===observation.postCountAfter&&Array.isArray(observation.posts)&&observation.posts.length===0&&Date.parse(observation.startedAt)<=Date.parse(observation.endedAt)&&Date.parse(observation.endedAt)<=Date.parse(capture.network.observedAt),'actual disabled UI guard/no-POST observation missing or unbound');
  return observation;
}
export async function adoptUi(capturePath, state, row, request, dest) {
  const capture = await json(capturePath);
  requireThat(capture.version === VERSION && capture.rowId === row.id && capture.lockSha256 === digest(state.lock) && capture.observer && Date.parse(capture.capturedAt) >= Date.parse(state.lock.createdAt), 'UI capture not bound to frozen candidate');
  requireThat(capture.network?.method === 'POST' && capture.network.path === '/api/v1/image/jobs' && capture.network.response?.id && capture.actions?.length, 'real observed UI submission request/response/action capture required; a job ID alone cannot pass');
  requireThat(capture.bindingSha256 === digest(state.binding), 'UI capture belongs to another backend execution binding');
  if (row.transport === 'ui-editor') await retainEditorSource(capturePath,state,row,request,dest,capture);
  else { requireThat(!capture.sourceUpload,'scratch aliases are forbidden outside the allocated Editor row'); assertRequestBinding(request,capture.network.request); }
  noReceipts(capture.network.request);
  for (const phase of ['before', 'after']) {
    const shot = capture.screenshots?.find(s => s.phase === phase); requireThat(shot, `missing UI ${phase} screenshot`);
    const bytes = await readFile(safePath(dirname(capturePath), shot.file)); requireThat(sha(bytes) === shot.sha256 && bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])), 'invalid UI screenshot binding');
    await writeFile(resolve(dest, `ui-${phase}.png`), bytes, { flag: 'wx', mode: 0o600 });
  }
  for(const name of uiGuards(row)) {
    const observation=uiGuard(capture,name),bytes=await readFile(safePath(dirname(capturePath),observation.screenshot.file));requireThat(sha(bytes)===observation.screenshot.sha256,'actual UI guard screenshot changed');
    await writeFile(resolve(dest,`ui-guard-${name}.png`),bytes,{flag:'wx',mode:0o600});
  }
  if (row.action === 'seeded-recipe') {
    const origin = await json(resolve(state.root, 'rows', row.replayOf, 'receipt.json'));
    const selected = await json(resolve(state.root, 'rows', row.replayOf, 'asset-2.json'));
    validateSeededRecipeSelection(row, request, capture.history, origin, selected);
  }
  if (row.action === 'workflow' || row.id === 'quality-cuda-base-four') {
    requireThat((capture.workflow?.file?.endsWith('.sceneworks-workflow.json') || capture.workflow?.file?.endsWith('.png')) && capture.workflow.inspection && capture.workflow.reacquired?.length, 'actual portable PNG/JSON export/inspection/ordered reacquisition required');
    const bytes = await readFile(safePath(dirname(capturePath), capture.workflow.file)); requireThat(sha(bytes) === capture.workflow.sha256, 'workflow bytes changed');
    const originId = row.replayOf ?? 'quality-mlx-base-four', origin = await json(resolve(state.root,'rows',originId,'receipt.json'));
    requireThat(capture.workflow.originRowId === originId && capture.workflow.originJobId === origin.jobId, 'portable replay is not bound to the allocated original job');
    const sampleIndex = capture.workflow.originSampleIndex ?? 0;
    requireThat(Number.isSafeInteger(sampleIndex) && sampleIndex >= 0 && sampleIndex < origin.outputs.length,'invalid exported original sample selection');
    const inspection=capture.workflow.inspection;
    validatePortableCapture(capture.workflow,bytes,origin,await json(resolve(state.root,'rows',originId,`provider-report-${sampleIndex}.json`)),sampleIndex);
    const png=capture.workflow.file.endsWith('.png');
    await writeFile(resolve(dest, png?'replay.png':'replay.sceneworks-workflow.json'), bytes, { flag: 'wx', mode: 0o600 });
    await save(resolve(dest,'workflow-inspection.json'),inspection);
    requireThat(capture.workflow.reacquired.length >= row.refs.length && row.refs.every((ref,index) => capture.workflow.reacquired.some(item => item.ordinal === index && item.inputId === ref && item.assetId === request.referenceAssetIds[index])), 'portable replay did not reacquire each original reference ordinal');
  }
  if (row.id === 'quality-cuda-sft-single') {
    const origin = await json(resolve(state.root,'rows','quality-mlx-sft-single','receipt.json'));
    requireThat(capture.history?.originJobId === origin.jobId && capture.history.hydrationObserved === true, 'Editor primary requires actual allocated history hydration');
  }
  requireThat(capture.progress?.some(j => j.id === capture.network.response.id && j.status === 'running'), 'UI needs observed actual running job, not only terminal lookup');
  await save(resolve(dest, 'ui-capture.json'), capture);
  return capture.network.response.id;
}
export async function waitJob(api, row, id, dest, timeoutMs, initial = []) {
  const progress = [...initial]; let canceled = false; const started = Date.now();
  try {
    while (Date.now() - started < timeoutMs) {
      const job = await api.request(jobPath(id)); progress.push({ observedAt: new Date().toISOString(), ...job });
      // Cancellation is admitted only after the actual product reports denoising.
      if (row.cancelAtDenoise && !canceled && isDenoiseStep(job)) {
        await save(resolve(dest, 'cancel-observation.json'), job);
        await save(resolve(dest, 'cancel-response.json'), await api.request(`${jobPath(id)}/cancel`, { method: 'POST', body: {} })); canceled = true;
      }
      if (['completed', 'failed', 'canceled', 'interrupted'].includes(job.status) && (job.status !== 'completed' || job.result?.outputPublication?.state !== 'pending')) {
        if (row.cancelAtDenoise) requireThat(canceled, 'job ended before observed-denoise cancellation');
        return job;
      }
      await sleep(1000);
    }
    throw new Error('AWAITING: job deadline; it may still be running. Reconcile before any next submission.');
  } finally { await save(resolve(dest, 'progress.json'), progress); }
}
async function retainOutputs(api, row, request, job, dest, root) {
  const outputs = [];
  const identities = await json(resolve(root, 'input-identities.json')), tensors = {};
  if (request.advanced.mossImageAdvanced) for (const name of ['positive','negative','mask','negativeMask','noise']) tensors[name] = await readFile(safePath(root, identities[name].file));
  for (const [index, summary] of (job.result.assets ?? []).entries()) {
    const asset = await api.request(assetPath(request.projectId, summary.id));
    requireThat(asset.id === summary.id && asset.projectId === request.projectId && asset.lineage?.jobId === job.id, 'asset ownership/job lineage mismatch');
    await save(resolve(dest, `asset-${index}.json`), asset);
    requireThat(asset.recipe, 'persisted recipe missing');
    if (request.mode === 'edit_image') requireThat(asset.recipe.rawAdapterSettings?.mossImageSampleIndex === index, 'output artifact is not bound to the matching native sample index');
    const path = asset.file?.path ?? asset.file?.relativePath ?? summary.mediaPath;
    requireThat(typeof path === 'string' && !path.startsWith('/') && !path.includes('..'), 'missing safe actual project asset path');
    const bytes = await api.request(mediaPath(request.projectId, path), { raw: true, max: 512 * 1024 * 1024 + 4108 });
    let tensor;
    if (row.output === 'latents') {
      requireThat(asset.type === 'tensor' && asset.tensor?.dtype === 'float32', 'latent output replaced by image/placeholder');
      tensor = inspectNpy(bytes, [1, 128, request.height / 16, request.width / 16]);
      requireThat(tensor.sha256 === asset.tensor.sha256 && tensor.byteLength === asset.tensor.byteLength, 'tensor artifact bytes disagree with actual published metadata');
    } else requireThat(bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) && bytes.readUInt32BE(16) === request.width && bytes.readUInt32BE(20) === request.height, 'expected full-size actual PNG output');
    if (request.mode === 'edit_image') { const report = findEditReport(asset, job); const base = validateEditReport(row, request, report, identities); validateAdvancedDigests(base, request, tensors); await save(resolve(dest, `provider-report-${index}.json`), report); }
    else {
      const report = asset.recipe.rawAdapterSettings?.mossImageExecutionReport ?? job.result.mossImageExecutionReport;
      requireThat(report?.execution_device === row.backend && report.resolved?.variant === row.variant && canonical(report.resolved.acceleration) === canonical(request.advanced.mossImage.acceleration) && report.samples?.length === 1 && report.samples[0].seed === 173 && report.output_kind === 'images' && report.assets?.length, 'ordinary/DMD regression lacks actual matching native execution report');
      await save(resolve(dest, `provider-report-${index}.json`), report);
    }
    const file = `output-${index}.${row.output === 'latents' ? 'npy' : 'png'}`;
    await writeFile(resolve(dest, file), bytes, { flag: 'wx', mode: 0o600 });
    const workflow = await api.request(`${assetPath(request.projectId, asset.id)}/workflow`); await save(resolve(dest, `workflow-${index}.json`), workflow);
    outputs.push({ assetId: asset.id, file, sha256: sha(bytes), ...(tensor ? { tensor } : {}) });
  }
  return outputs;
}
export async function retainMcpResults(mcp, job, outputs, dest, api) {
  const result = await mcp.call('tools/call', { name: 'get_job_result', arguments: { jobId: job.id, includeWorkflow: true } });
  requireThat(!result.isError, 'actual MCP result retrieval failed');
  const assets = mcpResultAssets(result,job.id,outputs,api.base.origin);
  const identities = [];
  const download = async url => {
    const response = await fetch(url, { redirect: 'error', signal: AbortSignal.timeout(60_000) }); requireThat(response.ok, 'MCP ticketed result download failed');
    const chunks = []; let total = 0; for await (const chunk of response.body) { total += chunk.length; requireThat(total <= 512 * 1024 * 1024 + 4108, 'MCP result exceeded bound'); chunks.push(chunk); }
    return Buffer.concat(chunks);
  };
  for (const [index, asset] of assets.entries()) {
    const bytes = await download(asset.mediaUrl); requireThat(sha(bytes) === outputs[index].sha256,'REST/MCP result bytes disagree');
    const companion = await download(asset.companionUrl); requireThat(companion.length <= 2*1024*1024 && sha(companion) === asset.companionSha256,'MCP workflow companion bytes/hash mismatch');
    validateExportedWorkflow(JSON.parse(companion),await json(resolve(dest,`provider-report-${index}.json`)),index);
    const companionFile = `output-${index}.sceneworks-workflow.json`;
    await writeFile(resolve(dest,companionFile),companion,{flag:'wx',mode:0o600});
    identities.push({ assetId:asset.assetId,sha256:sha(bytes),mimeType:asset.mimeType,byteLength:bytes.length,workflowCompanion:{file:companionFile,sha256:sha(companion)} });
  }
  await save(resolve(dest, 'mcp-result-identities.json'), identities);
}
export async function submitReplay(api, originJobId, action, dest) {
  requireThat(['duplicate', 'retry'].includes(action), 'unsupported replay action');
  const path = `${jobPath(originJobId)}/${action}`, observation = {};
  try {
    return await api.request(path, { method: 'POST', body: { payloadChanges: {} }, observation });
  } catch (error) {
    // Retain only the bounded failed replay response; never request/auth headers or successful jobs.
    if (observation.status >= 400 && observation.byteLength <= 64 * 1024) await save(resolve(dest, 'replay-response.json'), { method: 'POST', path, ...observation });
    throw error;
  }
}
async function run(configPath, id, admissionPath, uiCapture) {
  const state = await frozen(configPath), { config, root, plan, lock } = state;
  const row = plan.rows.find(r => r.id === id); requireThat(row, 'unknown frozen row');
  const admitted = await admission(admissionPath, state, row.backend), api = client(config, row.backend);
  const expected = await expectedRequest(state, row), dest = resolve(root, 'rows', id); let request = expected;
  const previous = plan.rows.filter(r => r.backend === row.backend).slice(0, plan.rows.filter(r => r.backend === row.backend).findIndex(r => r.id === id));
  for (const earlier of previous) requireThat((await json(resolve(root, 'rows', earlier.id, 'receipt.json'))).execution === 'VERIFIED', `complete/reconcile predecessor ${earlier.id} before next execution`);
  if (row.transport.startsWith('ui-')) requireThat(uiCapture, 'AWAITING_REAL_UI: use ui-task then provide the actual capture');
  const fields={lockSha256:digest(lock),bindingSha256:digest(state.binding),requestSha256:digest(expected)};
  const ticket = row.transport.startsWith('ui-') ? await resumeUiAttempt(root,row.backend,row.id,fields) : await reserveAttempt(root,row.backend,row.id,fields);
  const transcript = []; let mcp, jobId, submitted, job, submissionSaved=false; const started = new Date().toISOString();
  try {
  await mkdir(dest, { recursive: false, mode: 0o700 }).catch(async error => { if (error.code === 'ENOENT') { await mkdir(dirname(dest), { recursive: true, mode: 0o700 }); await mkdir(dest, { mode: 0o700 }); } else throw error; });
  await save(resolve(dest, 'admission.json'), admitted);
  if (!row.transport.startsWith('ui-')) await save(resolve(dest,'request.json'),request);
    await admission(admissionPath, state, row.backend);
    if (row.transport.startsWith('ui-')) {
      const observed=await json(uiCapture);jobId=observed.network?.response?.id??null;
      await save(resolve(dest,'ui-observed-submission.json'),observed);
      await save(resolve(dest,'request.json'),row.transport === 'ui-editor' ? observed.network.request : request);
      jobId = await adoptUi(uiCapture, state, row, request, dest);
      if (row.transport === 'ui-editor') request = observed.network.request;
    }
    else if (['duplicate','retry'].includes(row.action)) {
      const origin = await json(resolve(root, 'rows', row.replayOf, 'receipt.json'));
      if (row.action === 'retry') {
        const cleanup = await json(resolve(root,'external',`cleanup-before-${row.id}.json`));
        requireThat(cleanup.lockSha256 === digest(lock) && cleanup.canceledJobId === origin.jobId && cleanup.verdict === 'PASS' && cleanup.observer && cleanup.files?.length, 'actual owned cleanup proof required before retry/reentry');
        for (const entry of cleanup.files) requireThat((await fileIdentity(safePath(root,entry.file))).sha256 === entry.sha256, 'cleanup proof changed');
      }
      submitted = await submitReplay(api, origin.jobId, row.action, dest); jobId = submitted.id;
      await save(resolve(dest,'submission.json'),submitted);submissionSaved=true;
      requireThat(submitted.duplicateOfJobId === origin.jobId || submitted.sourceJobId === origin.jobId, 'duplicate not linked to actual original job');
    } else if (row.transport === 'mcp') {
      mcp = await mcpClient(config, row.backend, transcript);
      const result = await mcp.call('tools/call', { name: 'generate_image', arguments: mcpArgs(request), _meta: { progressToken: row.id } }, 60 * 60 * 1000);
      await save(resolve(dest, 'mcp-submission.json'), scrubMcp(result));
      const summary = mcpJson(result); requireThat(summary?.jobId, 'MCP actual generation returned no canonical job ID'); jobId = summary.jobId;
      requireThat(transcript.some(t => t.message.method === 'notifications/progress'), 'actual MCP progress not observed');
    } else { submitted = await api.request('/api/v1/image/jobs', { method: 'POST', body: request }); jobId = submitted.id;await save(resolve(dest,'submission.json'),submitted);submissionSaved=true; }
    requireThat(jobId, 'actual submission returned no job ID');
    if(!submissionSaved)await save(resolve(dest, 'submission.json'), submitted ?? { id: jobId, transport: row.transport });
    job = await waitJob(api, row, jobId, dest, 60 * 60 * 1000, submitted ? [submitted] : []);
    await save(resolve(dest, 'terminal-job.json'), job); checkJob(row, request, job,config.backends[row.backend].workerGpuId);
    requireThat(job.workerId === state.binding.endpoint.workerId,'job executed on a different bound worker');
    const outputs = await retainOutputs(api, row, request, job, dest, root);
    if (row.output === 'latents' && job.status === 'completed') { mcp ??= await mcpClient(config, row.backend, transcript); await retainMcpResults(mcp, job, outputs, dest, api); }
    await save(resolve(dest, 'metrics.json'), await api.request(`${jobPath(jobId)}/metrics`));
    await verifyFrozenSources(config, lock);
    await save(resolve(dest, 'receipt.json'), { version: VERSION, lockSha256: digest(lock), bindingSha256:digest(state.binding), rowId: id, jobId, backend: row.backend, device: admitted.device, execution: 'VERIFIED', acceptance: 'AWAITING_EXTERNAL_EVIDENCE_AND_VISUAL_REVIEW', started, ended: new Date().toISOString(), outputs, semanticRequestSha256:digest(expected), requestSha256: digest(request), jobSha256: digest(job), admissionSha256: digest(admitted) });
    await resolveAttempt(ticket,'TERMINAL_VERIFIED',{jobId});
    console.log(JSON.stringify({ rowId: id, jobId, execution: 'VERIFIED', acceptance: 'AWAITING', outputs: outputs.map(o => ({ assetId: o.assetId, sha256: o.sha256 })) }));
  } catch (error) {
    await save(resolve(dest, 'failure.json'), { version: VERSION, rowId: id, jobId: jobId ?? null, status: 'FAIL_OR_AWAITING_RECONCILIATION', message: error.message, observedAt: new Date().toISOString() });
    await failAttempt(ticket,error,jobId);
    throw error;
  } finally { if (transcript.length) await save(resolve(dest, 'mcp-transcript.json'), transcript); mcp?.close(); }
}
export async function recoverCompletedRow(configPath, id = 'advanced-mlx-sft') {
  const state=await frozen(configPath),{config,root,lock}=state;
  requireThat(config.repair?.name==='r16-checker-recovery'&&id==='advanced-mlx-sft'&&config.localBackend==='mlx','only the reviewed R16 completed SFT job may be recovered');
  const row=state.plan.rows.find(item=>item.id===id),dest=resolve(root,'rows',id),attemptDir=resolve(root,'attempts','mlx');
  requireThat(row?.transport==='mcp'&&row.output==='latents'&&row.backend==='mlx','recovery row contract changed');
  const [failure,attemptFailure,start,request,retainedJob,admitted]=await Promise.all([
    json(resolve(dest,'failure.json')),json(resolve(attemptDir,`${id}.failure.json`)),json(resolve(attemptDir,`${id}.start.json`)),
    json(resolve(dest,'request.json')),json(resolve(dest,'terminal-job.json')),json(resolve(dest,'admission.json'))]);
  const expected=await expectedRequest(state,row),message='per-prompt negatives gained an unsupported broadcast field';
  requireThat(failure.version===VERSION&&failure.rowId===id&&failure.jobId===retainedJob.id&&failure.message===message&&attemptFailure.state==='UNRESOLVED'&&attemptFailure.jobId===retainedJob.id&&attemptFailure.message===message&&attemptFailure.attemptSha256===digest(start),'original checker failure or attempt identity changed');
  requireThat(start.lockSha256===digest(lock)&&start.bindingSha256===digest(state.binding)&&start.requestSha256===digest(expected)&&digest(request)===digest(expected),'original failed attempt differs from frozen request/candidate');
  requireThat(!(await readdir(dest)).includes('receipt.json'),'failed row already recovered; never resubmit or overwrite');
  validateAdmission(admitted,lock,'mlx');assertRequestBinding(expected,request);
  checkJob(row,request,retainedJob,state.binding.endpoint.workerGpuId);
  requireThat(retainedJob.status==='completed'&&retainedJob.workerId===state.binding.endpoint.workerId,'failed checker did not leave a completed owned job');
  const api=client(config,'mlx'),actual=await api.request(jobPath(retainedJob.id));
  requireThat(actual.id===retainedJob.id&&actual.status==='completed'&&actual.workerId===retainedJob.workerId&&actual.projectId===retainedJob.projectId&&canonical(actual.payload)===canonical(retainedJob.payload)&&canonical(actual.result)===canonical(retainedJob.result),'authenticated completed job readback differs from original native result');
  await save(resolve(dest,'recovery-job-readback.json'),actual);
  const outputs=await retainOutputs(api,row,request,actual,dest,root);
  const transcript=[],mcp=await mcpClient(config,'mlx',transcript);
  try {await retainMcpResults(mcp,actual,outputs,dest,api);}finally{mcp.close();if(transcript.length)await save(resolve(dest,'mcp-recovery-transcript.json'),transcript);}
  await save(resolve(dest,'metrics.json'),await api.request(`${jobPath(actual.id)}/metrics`));
  const originalRaw=resolve(config.repair.originalRoot,'samples',`${id}.jsonl`),raw=(await readFile(originalRaw,'utf8')).trimEnd().split('\n').map(line=>JSON.parse(line));
  requireThat(raw.length>=3&&raw[0].phase==='baseline'&&raw.at(-1).phase==='drain'&&raw.some(item=>item.phase==='active')&&raw.every(item=>item.device?.uuid===admitted.device.uuid&&item.raw),'original whole-device sampler trace incomplete or changed');
  const rawPath=resolve(dest,'memory-raw.jsonl');await copyFile(originalRaw,rawPath,constants.COPYFILE_EXCL);
  const samplerSha256=lock.files.actionLibrary.sha256,memory={version:VERSION,jobId:actual.id,samplerSha256,device:raw[0].device,samples:raw.map(({at,phase,usedBytes})=>({at,phase,usedBytes})),raw:{file:'memory-raw.jsonl',...await fileIdentity(rawPath)}};
  memorySummary(memory,actual.id);await save(resolve(dest,'memory.json'),memory);
  await verifyFrozenSources(config,lock);
  const receipt={version:VERSION,lockSha256:digest(lock),bindingSha256:digest(state.binding),rowId:id,jobId:actual.id,backend:'mlx',device:admitted.device,execution:'VERIFIED',acceptance:'AWAITING_EXTERNAL_EVIDENCE_AND_VISUAL_REVIEW',started:start.startedAt,ended:new Date().toISOString(),outputs,semanticRequestSha256:digest(expected),requestSha256:digest(request),jobSha256:digest(retainedJob),admissionSha256:digest(admitted),recoveredWithoutSubmission:true};
  await save(resolve(dest,'receipt.json'),receipt);
  console.log(JSON.stringify({rowId:id,jobId:actual.id,execution:'VERIFIED',recoveredWithoutSubmission:true,outputs:outputs.map(item=>item.sha256)}));
}
export async function readCanceledJob(api,row,request,submission,observed,response,binding) {
  const job=await api.request(jobPath(submission.id));
  requireThat(job.id===submission.id&&job.projectId===request.projectId&&job.workerId===binding.endpoint.workerId&&job.status==='canceled'&&Date.parse(job.canceledAt)>=Date.parse(response.updatedAt)&&canonical(job.payload)===canonical(submission.payload),'authenticated terminal canceled job differs from the known submitted job');
  checkJob(row,request,job,binding.endpoint.workerGpuId);
  requireThat((job.result?.assetWrites??[]).length===0&&(job.result?.assets??[]).length===0,'canceled native job wrote an output asset');
  return {job,metrics:await api.request(`${jobPath(job.id)}/metrics`)};
}
export async function recoverCanceledRow(configPath,id) {
  const state=await frozen(configPath),{config,root,lock}=state,ordinal=config.repair?.retryOrdinal;
  requireThat(Number.isSafeInteger(ordinal)&&ordinal>=2&&config.localBackend==='mlx','only a chained source-bound MLX cancellation may be recovered');
  const record=await json(resolve(root,`continuation-retry-${ordinal}.json`)),recovery=record.submittedRecovery;
  requireThat(recovery?.rowId===id&&record.toolingSha===config.repair.tooling.head,'known submitted cancellation is not bound to current repair tooling');
  const row=state.plan.rows.find(item=>item.id===id),dest=resolve(root,'rows',id),attempt=resolve(root,'attempts/mlx'),start=await json(resolve(attempt,`${id}.start.json`));
  requireThat(row?.transport==='rest'&&row.expected==='canceled'&&row.cancelAtDenoise===true&&await readFile(resolve(attempt,'dispatch.lock'),'utf8')===id,'only the original unresolved REST cancellation may be recovered');
  const [request,submission,admitted,observed,response]=await Promise.all(['request.json','submission.json','admission.json','cancel-observation.json','cancel-response.json'].map(name=>json(resolve(dest,name))));
  const expected=await expectedRequest(state,row);
  requireThat(recovery.jobId===submission.id&&recovery.requestSha256===digest(request)&&recovery.submissionSha256===digest(submission)&&recovery.attemptSha256===digest(start)&&start.requestSha256===digest(expected)&&start.bindingSha256===digest(state.binding)&&start.lockSha256===digest(lock)&&digest(request)===digest(expected),'retained canceled submission/request/attempt changed');
  validateAdmission(admitted,lock,'mlx');assertRequestBinding(expected,submission.payload);
  requireThat(observed.id===submission.id&&isDenoiseStep(observed)&&response.id===submission.id&&response.cancelRequested===true&&response.workerId===observed.workerId&&Date.parse(response.updatedAt)>=Date.parse(observed.updatedAt),'actual denoise cancellation record changed');
  const {job,metrics}=await readCanceledJob(client(config,'mlx'),row,request,submission,observed,response,state.binding);
  const samplePath=safePath(config.repair.originalRoot,recovery.sampleFile),checkpoint=await json(resolve(root,`continuation-prior-${ordinal}-inventory.json`)),tail=checkpoint.appendOnlyTails?.find(item=>item.file===recovery.sampleFile),sampleBytes=await readFile(samplePath);
  const sealed=checkpoint.files.find(item=>item.file===recovery.sampleFile);
  requireThat((tail?tail.stoppedByteLength===sampleBytes.length&&tail.stoppedSha256===sha(sampleBytes):sealed?.byteLength===sampleBytes.length&&sealed.sha256===sha(sampleBytes)),'original canceled whole-device trace changed');
  const raw=sampleBytes.toString('utf8').trimEnd().split('\n').map(line=>JSON.parse(line));
  const memory={version:VERSION,jobId:job.id,samplerSha256:lock.files.actionLibrary.sha256,device:raw[0]?.device,samples:raw.map(({at,phase,usedBytes})=>({at,phase,usedBytes}))};
  memorySummary(memory,job.id);
  requireThat(raw.every(item=>item.device?.uuid===admitted.device.uuid&&item.raw)&&Date.parse(raw[0].at)<=Date.parse(job.startedAt)&&Date.parse(raw.at(-1).at)>=Date.parse(job.canceledAt)&&raw.some(item=>item.phase==='active'&&Date.parse(item.at)>=Date.parse(job.startedAt)&&Date.parse(item.at)<=Date.parse(job.canceledAt)),'original whole-device samples do not bracket the canceled native execution');
  for(const file of ['terminal-job.json','receipt.json','failure.json','memory.json','memory-raw.jsonl'])requireThat(!(await readdir(dest)).includes(file),'canceled row already materialized; never resubmit or overwrite');
  await verifyFrozenSources(config,lock);
  const rawPath=resolve(dest,'memory-raw.jsonl');await copyFile(samplePath,rawPath,constants.COPYFILE_EXCL);memory.raw={file:'memory-raw.jsonl',...await fileIdentity(rawPath)};
  await save(resolve(dest,'recovery-job-readback.json'),job);await save(resolve(dest,'terminal-job.json'),job);await save(resolve(dest,'metrics.json'),metrics);await save(resolve(dest,'memory.json'),memory);
  const proof={version:VERSION,rowId:id,jobId:job.id,lockSha256:digest(lock),attemptSha256:digest(start),requestSha256:digest(request),submissionSha256:digest(submission),admissionSha256:digest(admitted),cancelObservationSha256:digest(observed),cancelResponseSha256:digest(response),checkpointInventorySha256:record.checkpointInventorySha256,jobSha256:digest(job),rawSample:memory.raw,source:'authenticated original native terminal cancellation; no new submission or cancel request'};
  await save(resolve(dest,'canceled-recovery.json'),proof);
  await save(resolve(dest,'receipt.json'),{version:VERSION,lockSha256:digest(lock),bindingSha256:digest(state.binding),rowId:id,jobId:job.id,backend:'mlx',device:admitted.device,execution:'VERIFIED',acceptance:'AWAITING_EXTERNAL_EVIDENCE_AND_VISUAL_REVIEW',started:start.startedAt,ended:new Date().toISOString(),outputs:[],semanticRequestSha256:digest(expected),requestSha256:digest(request),jobSha256:digest(job),admissionSha256:digest(admitted),recoveredWithoutSubmission:true,recoveryProofSha256:digest(proof)});
  await resolveAttempt({directory:attempt,mutex:resolve(attempt,'dispatch.lock'),id,start},'TERMINAL_VERIFIED',{jobId:job.id,recoveredWithoutSubmission:true,proofSha256:digest(proof)});
  return proof;
}
export async function submitGuard(api,specification,request,path) {
  let submitted;const observation={};
  const upload=specification.upload,route=upload?`/api/v1/projects/${encode(request.projectId)}/assets`:'/api/v1/image/jobs';let body=request,before;
  if(upload) {
    const bytes=await readFile(resolve(path,'request.npy'));requireThat(sha(bytes)===upload.sha256&&bytes.equals(hostBudgetHeader()),'submitted host-budget header bytes changed');
    body=new FormData();body.append('file',new Blob([bytes],{type:upload.mime}),upload.filename);
    before=await api.request(`${route}?includeRejected=true&includeTrashed=true`);await save(resolve(path,'assets-before.json'),before);
  }
  try { submitted = await api.request(route,{method:'POST',observation,...(specification.rawJson === undefined ? {body} : {rawJson:specification.rawJson})}); }
  catch(error) {
    await save(resolve(path,'response.json'),{status:error.status??null,body:error.responseBody??null,http:observation,transportError:error.status?null:error.message});
    if(upload) {
      const after=await api.request(`${route}?includeRejected=true&includeTrashed=true`);await save(resolve(path,'assets-after.json'),after);
      requireThat(Array.isArray(before)&&Array.isArray(after)&&canonical(before.map(asset=>asset.id).sort())===canonical(after.map(asset=>asset.id).sort()),'host import refusal created an asset/output');
    }
    if (error.status !== specification.expectedStatus || !specification.errorIncludes.every(text => error.responseBody?.includes(text))) throw error;
    return;
  }
  if(upload){await save(resolve(path,'unexpected-asset.json'),submitted);const error=Error('host import budget refusal unexpectedly created an asset');error.assetId=submitted.id??null;throw error;}
  if(specification.expectedTerminal==='failed') {
    requireThat(observation.status===201&&submitted.id,'actual queued HTTP 201 worker-refusal response required');
    await save(resolve(path,'queued-response.json'),{http:observation,job:submitted});return submitted;
  }
  await save(resolve(path,'unexpected-enqueue.json'),submitted);
  const error = new Error('refusal unexpectedly enqueued an actual job; authoritative reconciliation required');
  error.jobId = submitted.id ?? null; throw error;
}
export function validateWorkerRefusal(specification,request,job,endpoint,backend) {
  requireThat(specification.expectedTerminal==='failed'&&specification.expectedStatus===201,'worker-refusal specification missing');
  assertRequestBinding(request,job.payload);checkJob({backend,expected:'failed',outputCount:0},request,job,endpoint.workerGpuId);
  requireThat(job.workerId===endpoint.workerId&&job.startedAt&&job.updatedAt&&specification.errorIncludes.every(text=>job.error?.includes(text)),'actual budget failure/worker/timing/no-output evidence missing or different');
}
async function guard(configPath, name, backend, admissionPath) {
  const state = await frozen(configPath); requireThat(state.plan.requiredGuardCases.includes(name), 'unallocated refusal');
  const specification = state.config.guardRequests[name], admitted = await admission(admissionPath, state, backend);
  const path = resolve(state.root, 'guards', `${backend}-${name}`);
  const request = specification.upload ? {projectId:state.binding.endpoint.projectId,upload:specification.upload} : specification.rawJson === undefined ? replaceInputs({ request: specification.request, refs: specification.refs ?? [] }, await json(resolve(state.root, `uploaded-${backend}.json`)), state.config.backends[backend].projectId) : specification.rawJson;
  const requestSha256 = typeof request === 'string' ? sha(Buffer.from(request)) : digest(request);
  const ticket = await reserveAttempt(state.root,backend,`guard-${name}`,{lockSha256:digest(state.lock),bindingSha256:digest(state.binding),requestSha256});
  try {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 }); await mkdir(path, { mode: 0o700 });
  if (typeof request === 'string') await writeFile(resolve(path,'request.raw.json'),request,{flag:'wx',mode:0o600});
  else await save(resolve(path, 'request.json'), request);
  if(specification.upload)await writeFile(resolve(path,'request.npy'),await readFile(safePath(state.root,specification.upload.file)),{flag:'wx',mode:0o600});
  await save(resolve(path, 'admission.json'), admitted);
  const api=client(state.config,backend),queued=await submitGuard(api,specification,request,path);let workerJob=null;
  if(queued) {
    requireThat(['device-budget-before-load','corrupt-adapter'].includes(name),'worker refusal is forbidden for other malformed inputs');
    try {
      workerJob=await waitJob(api,{expected:'failed'},queued.id,path,specification.timeoutMs??120000);
      validateWorkerRefusal(specification,request,workerJob,state.binding.endpoint,backend);await save(resolve(path,'terminal-job.json'),workerJob);
    }catch(error){error.jobId=queued.id;throw error;}
  }
  const responseFile=workerJob?'queued-response.json':'response.json',responseSha256=(await fileIdentity(resolve(path,responseFile))).sha256;
  await save(resolve(path, 'receipt.json'), { version: VERSION, lockSha256: digest(state.lock), bindingSha256:digest(state.binding), case: name, backend, route:specification.upload?`/api/v1/projects/${encode(request.projectId)}/assets`:'/api/v1/image/jobs', contentType:specification.upload?'multipart/form-data':'application/json', rawJson:typeof request==='string', status: 'AWAITING_NO_PROVIDER_LOAD_PROOF', requestSha256,responseFile,responseSha256,assetSnapshotSha256:specification.upload?{before:(await fileIdentity(resolve(path,'assets-before.json'))).sha256,after:(await fileIdentity(resolve(path,'assets-after.json'))).sha256}:null, jobId:workerJob?.id??null,jobSha256:workerJob?digest(workerJob):null, probe: specification.noProviderLoadProbe });
  await resolveAttempt(ticket,'EXPECTED_REFUSAL');
  } catch(error) { await failAttempt(ticket,error,error.jobId??null); throw error; }
}
export async function measuredCacheComparison(state,row) {
  const ids = [row.compareTo,row.id]; requireThat(ids[0], 'not a cache comparison row');
  const pair = [],buffers = [],requests = [];
  for (const id of ids) {
    const directory = resolve(state.root,'rows',id), receipt = await json(resolve(directory,'receipt.json')),request = await json(resolve(directory,'request.json'));
    requireThat(receipt.lockSha256 === digest(state.lock) && receipt.execution === 'VERIFIED' && receipt.outputs.length === 1 && receipt.requestSha256 === digest(request),'cache comparison requires exact verified row request/output');
    const bytes = await readFile(safePath(directory,receipt.outputs[0].file)); requireThat(sha(bytes) === receipt.outputs[0].sha256,'cache comparison output changed');
    pair.push({rowId:id,jobId:receipt.jobId,requestSha256:receipt.requestSha256,outputSha256:receipt.outputs[0].sha256}); buffers.push(bytes);
    if (id === 'quality-cuda-sft-single') {
      const cachedRow = state.plan.rows.find(item=>item.id===id), expected = await expectedRequest(state,cachedRow);
      await validateRetainedEditor(state,cachedRow,expected,await json(resolve(directory,'ui-capture.json')),directory);
      requests.push(expected); // Same frozen RGB source, after proving the real Editor scratch upload.
    } else requests.push(request);
  }
  requests[1].advanced.mossImageEdit.use_kv_cache = true;
  requireThat(canonical(requests[0]) === canonical(requests[1]),'cache comparison differs in authored inputs beyond cache policy');
  requireThat(state.plan.cacheComparisonPolicy.implementationSha256 === state.lock.files.harnessLibrary.sha256,'cache implementation differs from frozen executing library');
  const comparison=state.config?.repair?(await import(pathToFileURL(state.config.files.harnessLibrary).href)).cacheComparison:cacheComparison;
  return comparison(state.plan.cacheComparisonPolicy,digest(state.lock),pair,buffers);
}
async function compareCache(configPath,rowId) {
  const state = await frozen(configPath),row = state.plan.rows.find(r => r.id === rowId);
  requireThat(row?.compareTo,'unknown cache comparison row');
  const comparison = await measuredCacheComparison(state,row);
  await save(resolve(state.root,'rows',rowId,'cache-comparison.json'),comparison);
  console.log(JSON.stringify({rowId,metric:comparison.metric,value:comparison.value,verdict:comparison.value <= state.plan.cacheComparisonPolicy.tolerance ? 'PASS':'FAIL'}));
}
async function reconcile(configPath,backend,id,proofPath) {
  const state = await frozen(configPath);
  await reconcileAttempt(state.root,backend,id,await json(proofPath),digest(state.lock));
  console.log(JSON.stringify({backend,attempt:id,state:'RECONCILED_FOR_DISPATCH_ONLY',candidateAcceptance:'FAIL_RECORDED_ATTEMPT_PRESERVED'}));
}
export async function validateRecoveryAdjudication(state,expectedInventorySha256=R16_ORIGINAL_INVENTORY_SHA256) {
    const repair=state.config.repair,manifestPath=resolve(state.root,'continuation-original-inventory.json'),manifest=await json(manifestPath),adjudication=await json(resolve(state.root,'recovery-adjudication.json'));
    requireThat(repair.name==='r16-checker-recovery'&&repair.inventorySha256===expectedInventorySha256&&repair.inventorySha256===(await fileIdentity(manifestPath)).sha256&&manifest.version===1&&manifest.actualRunId===adjudication.originalRunId&&manifest.attempt===adjudication.originalAttempt&&manifest.sourceSha===state.lock.sources.sceneworks.head&&manifest.inferenceSha===state.lock.sources.inference.head&&manifest.files?.length===257,'original partial evidence/candidate boundary changed');
    await assertOriginalEvidence(state.root,manifest);
    if(repair.previousToolingSha) {
      const retry=await json(resolve(state.root,'continuation-retry.json')),checkpointPath=resolve(state.root,'continuation-h18-inventory.json'),checkpoint=await json(checkpointPath);
      requireThat(repair.previousToolingSha===H18_RETRY_TOOLING_SHA&&retry.previousToolingSha===H18_RETRY_TOOLING_SHA&&(repair.retryOrdinal>1||retry.toolingSha===repair.tooling.head)&&retry.name==='r16-checker-recovery-selector-retry'&&retry.failedRunId===H18_RETRY_RUN_ID&&retry.failedAttempt===1&&retry.failedArtifactId===H18_RETRY_ARTIFACT_ID&&retry.candidateId===state.lock.candidateId&&retry.sourceSha===state.lock.sources.sceneworks.head&&retry.inferenceSha===state.lock.sources.inference.head&&retry.lockSha256===digest(state.lock)&&retry.originalInventorySha256===repair.inventorySha256&&retry.checkpointInventorySha256===H18_RETRY_INVENTORY_SHA256&&repair.checkpointInventorySha256===H18_RETRY_INVENTORY_SHA256&&retry.checkpointInventorySha256===(await fileIdentity(checkpointPath)).sha256,'H18 retry checkpoint identity changed');
      requireThat(checkpoint.version===1&&checkpoint.actualRunId===H18_RETRY_RUN_ID&&checkpoint.attempt===1&&checkpoint.artifactId===H18_RETRY_ARTIFACT_ID&&checkpoint.toolingSha===H18_RETRY_TOOLING_SHA&&checkpoint.productSha===state.lock.sources.sceneworks.head&&checkpoint.originalRunId===retry.originalRunId&&checkpoint.originalAttempt===retry.originalAttempt&&checkpoint.files?.length>=300,'H18 artifact inventory was substituted');
      const lockEntry=checkpoint.files.find(item=>item.file==='evidence/attempts/mlx/dispatch.lock');
      requireThat(lockEntry&&lockEntry.sha256===retry.dispatchLockSha256,'original H18 dispatch lock identity changed');
      await assertH18CheckpointFiles(state.root,checkpoint,{retained:true});
      const start=await json(resolve(state.root,'attempts/mlx/advanced-mlx-sft-replay.start.json')),resolution=await json(resolve(state.root,'attempts/mlx/advanced-mlx-sft-replay.resolution.json'));
      requireThat(retry.uiAttemptSha256===digest(start)&&resolution.attemptSha256===digest(start)&&resolution.state==='TERMINAL_VERIFIED','only the exact unresolved H18 UI intent may consume its archived dispatch lock');
      let previous=retry;
      for(let ordinal=2;ordinal<=(repair.retryOrdinal??1);ordinal++) {
        const current=await json(resolve(state.root,`continuation-retry-${ordinal}.json`)),priorPath=resolve(state.root,`continuation-prior-${ordinal}-inventory.json`),prior=await json(priorPath);
        requireThat(current.ordinal===ordinal&&current.name===retry.name&&current.previousToolingSha===previous.toolingSha&&current.toolingSha!==(previous.toolingSha)&&current.originalRunId===retry.originalRunId&&current.originalAttempt===retry.originalAttempt&&current.candidateId===retry.candidateId&&current.sourceSha===retry.sourceSha&&current.inferenceSha===retry.inferenceSha&&current.originalInventorySha256===repair.inventorySha256&&current.lockSha256===digest(state.lock)&&current.uiAttemptSha256===digest(start)&&current.checkpointInventorySha256===(await fileIdentity(priorPath)).sha256&&prior.version===1&&prior.actualRunId===current.failedRunId&&prior.attempt===current.failedAttempt&&prior.artifactId===current.failedArtifactId&&prior.toolingSha===previous.toolingSha&&prior.productSha===retry.sourceSha&&prior.originalRunId===retry.originalRunId&&prior.originalAttempt===retry.originalAttempt&&prior.files?.length>=300,'chained failed-attempt artifact or tooling identity changed');
        const heldLock=await fileIdentity(resolve(state.root,`continuation-prior-${ordinal}-dispatch.lock`));
        requireThat(heldLock.sha256===current.dispatchLockSha256&&prior.files.find(item=>item.file==='evidence/attempts/mlx/dispatch.lock')?.sha256===heldLock.sha256,'chained original submission mutex changed');
        await assertH18CheckpointFiles(state.root,prior,{retained:true,archiveTag:`prior-${ordinal}`});
        if(current.submittedRecovery) {
          const known=current.submittedRecovery,id=known.rowId,row=state.plan.rows.find(item=>item.id===id),dir=resolve(state.root,'rows',id),attempt=resolve(state.root,'attempts/mlx');
          requireThat(row?.expected==='canceled'&&row.transport==='rest'&&row.cancelAtDenoise===true&&prior.files.some(item=>item.file===`evidence/rows/${id}/submission.json`)&&!prior.files.some(item=>[`evidence/rows/${id}/receipt.json`,`evidence/rows/${id}/terminal-job.json`,`evidence/attempts/mlx/${id}.resolution.json`].includes(item.file)),'recovered cancellation was not a single unresolved submitted job');
          const [submitted,request,admitted,observed,response,origin,job,readback,receipt,proof,resolution]=await Promise.all([
            json(resolve(dir,'submission.json')),json(resolve(dir,'request.json')),json(resolve(dir,'admission.json')),json(resolve(dir,'cancel-observation.json')),json(resolve(dir,'cancel-response.json')),json(resolve(attempt,`${id}.start.json`)),json(resolve(dir,'terminal-job.json')),json(resolve(dir,'recovery-job-readback.json')),json(resolve(dir,'receipt.json')),json(resolve(dir,'canceled-recovery.json')),json(resolve(attempt,`${id}.resolution.json`))]);
          requireThat(known.jobId===submitted.id&&known.requestSha256===digest(request)&&known.submissionSha256===digest(submitted)&&known.attemptSha256===digest(origin)&&origin.requestSha256===digest(request)&&origin.lockSha256===digest(state.lock)&&observed.id===submitted.id&&isDenoiseStep(observed)&&response.id===submitted.id&&response.cancelRequested===true&&job.id===submitted.id&&job.status==='canceled'&&canonical(job)===canonical(readback)&&job.workerId===observed.workerId&&!(job.result?.assets?.length)&&!(job.result?.assetWrites?.length)&&receipt.recoveredWithoutSubmission===true&&receipt.jobId===job.id&&receipt.jobSha256===digest(job)&&receipt.admissionSha256===digest(admitted)&&receipt.recoveryProofSha256===digest(proof)&&proof.jobSha256===digest(job)&&proof.attemptSha256===digest(origin)&&proof.submissionSha256===digest(submitted)&&proof.cancelObservationSha256===digest(observed)&&proof.cancelResponseSha256===digest(response)&&proof.checkpointInventorySha256===current.checkpointInventorySha256&&resolution.state==='TERMINAL_VERIFIED'&&resolution.attemptSha256===digest(origin)&&resolution.jobId===job.id&&resolution.recoveredWithoutSubmission===true&&resolution.proofSha256===digest(proof),'authenticated canceled-job recovery/adjudication changed');
          const sample=prior.appendOnlyTails?.find(item=>item.file===known.sampleFile),sealed=prior.files.find(item=>item.file===known.sampleFile),memory=await json(resolve(dir,'memory.json'));
          requireThat((sample?sample.stoppedSha256===memory.raw?.sha256&&sample.stoppedByteLength===memory.raw.byteLength:sealed?.sha256===memory.raw?.sha256&&sealed.byteLength===memory.raw.byteLength)&&(await fileIdentity(resolve(dir,'memory-raw.jsonl'))).sha256===memory.raw.sha256,'original canceled-job whole-device sample changed');
        } else {
          const oldCapture=await json(resolve(state.root,`continuation-prior-${ordinal}-root/captures/advanced-mlx-sft-replay${ordinal===2?'-retry':`-retry-${ordinal-1}`}/failure.json`));
          validateFailedUiCapture(oldCapture,digest(state.lock),{historical:ordinal===2?'h19':null});
        }
        previous=current;
      }
      requireThat(previous.toolingSha===repair.tooling.head&&!(await readdir(state.root)).some(name=>/^continuation-retry-[0-9]+\.json$/.test(name)&&Number(name.slice('continuation-retry-'.length,-'.json'.length))>(repair.retryOrdinal??1)),'final tooling or chained retry count changed');
    }
    requireThat(adjudication.version===VERSION&&adjudication.name==='r16-checker-recovery'&&adjudication.rowId==='advanced-mlx-sft'&&adjudication.candidateId===state.lock.candidateId&&adjudication.sourceSha===state.lock.sources.sceneworks.head&&adjudication.inferenceSha===state.lock.sources.inference.head&&adjudication.toolingSha===(repair.previousToolingSha??repair.tooling.head)&&adjudication.inventorySha256===repair.inventorySha256&&adjudication.lockSha256===digest(state.lock),'checker adjudication substituted candidate/source/tooling');
    const id='advanced-mlx-sft',dir=resolve(state.root,'rows',id),attempt=resolve(state.root,'attempts/mlx'),failure=await json(resolve(dir,'failure.json')),attemptFailure=await json(resolve(attempt,`${id}.failure.json`)),start=await json(resolve(attempt,`${id}.start.json`)),receipt=await json(resolve(dir,'receipt.json')),job=await json(resolve(dir,'terminal-job.json')),readback=await json(resolve(dir,'recovery-job-readback.json')),proof=await json(resolve(attempt,`${id}.reconciliation.json`)),resolution=await json(resolve(attempt,`${id}.resolution.json`));
    requireThat(failure.message==='per-prompt negatives gained an unsupported broadcast field'&&attemptFailure.message===failure.message&&failure.jobId===job.id&&attemptFailure.jobId===job.id&&attemptFailure.attemptSha256===digest(start)&&receipt.recoveredWithoutSubmission===true&&receipt.jobId===job.id&&receipt.jobSha256===digest(job)&&adjudication.jobId===job.id&&adjudication.originalAttemptSha256===digest(start)&&adjudication.requestSha256===start.requestSha256,'recovery does not identify the exact original failed completed job');
    requireThat(readback.id===job.id&&readback.status==='completed'&&canonical(readback.payload)===canonical(job.payload)&&canonical(readback.result)===canonical(job.result)&&proof.state==='NO_ACTIVE_OWNED_JOB'&&proof.attemptSha256===digest(start)&&proof.lockSha256===digest(state.lock)&&proof.jobs?.length===1&&canonical(proof.jobs[0])===canonical(readback)&&adjudication.reconciliationSha256===digest(proof)&&resolution.state==='COORDINATOR_RECONCILED'&&resolution.attemptSha256===digest(start)&&resolution.proofSha256===digest(proof),'authenticated completed-job reconciliation changed');
    requireThat(proof.rawEvidence?.length>0&&canonical(adjudication.rawEvidence)===canonical(proof.rawEvidence),'adjudication raw evidence differs from reconciliation');
    for(const entry of proof.rawEvidence)requireThat(canonical(await fileIdentity(safePath(state.root,entry.file)))===canonical({byteLength:entry.byteLength,sha256:entry.sha256}),'recovery raw evidence changed');
    return true;
}
export function acceptedRecoveryFailure(acceptedRecovery,backend,name) {
  return acceptedRecovery&&backend==='mlx'&&name==='advanced-mlx-sft.failure.json';
}
async function verify(configPath, retained = false) {
  const state = await frozen(configPath,{retained}), findings = [], summaries = [];
  const requireEvidence = async (name, check) => { try { await check(); } catch (error) { findings.push({ item: name, state: ['ENOENT','MOSS_AWAITING_RESOLUTION'].includes(error.code) ? 'AWAITING' : 'FAIL', reason: error.code === 'ENOENT' ? 'required evidence missing' : error.message }); } };
  let acceptedRecovery=false;
  if(state.config.repair)await requireEvidence('r16-original-checker-recovery',async()=>{acceptedRecovery=await validateRecoveryAdjudication(state);});
  for (const backend of ['mlx','cuda']) await requireEvidence(`dispatch-${backend}`,async () => {
    const binding = validateBinding(await json(resolve(state.root,'bindings',`${backend}.json`)),state.lock,backend);
    const attempts = await readdir(resolve(state.root,'attempts',backend)).catch(error => {if(error.code === 'ENOENT')return[];throw error;});
    for (const name of attempts.filter(name=>name.endsWith('.start.json'))) {
      const start = await json(resolve(state.root,'attempts',backend,name));
      requireThat(start.lockSha256 === digest(state.lock) && start.bindingSha256 === digest(binding),'attempt belongs to another frozen candidate/backend binding');
    }
    requireThat(!attempts.some(name => name.endsWith('.failure.json') && !acceptedRecoveryFailure(acceptedRecovery,backend,name)),'recorded failed submission attempt remains failed even after safe reconciliation');
    await assertDispatchClear(state.root,backend);
  });
  for (const row of state.plan.rows) await requireEvidence(row.id, async () => {
    const path = resolve(state.root, 'rows', row.id);
    const failure = await json(resolve(path,'failure.json')).catch(error => { if (error.code === 'ENOENT') return null; throw error; });
    requireThat(!failure || (acceptedRecovery&&row.id==='advanced-mlx-sft'), 'recorded execution attempt failed or requires authoritative reconciliation; cannot accept this candidate');
    const receipt = await json(resolve(path, 'receipt.json'));
    const binding = validateBinding(await json(resolve(state.root,'bindings',`${row.backend}.json`)),state.lock,row.backend);
    requireThat(receipt.version === VERSION && receipt.lockSha256 === digest(state.lock) && receipt.bindingSha256 === digest(binding) && receipt.execution === 'VERIFIED', 'row receipt not bound/verified');
    const request = await json(resolve(path, 'request.json')), job = await json(resolve(path, 'terminal-job.json'));
    requireThat(digest(request) === receipt.requestSha256 && digest(job) === receipt.jobSha256, 'retained request/job changed');
    const expected = await expectedRequest(state,row);
    requireThat(receipt.semanticRequestSha256 === digest(expected),'frozen semantic request changed');
    if (row.transport === 'ui-editor') {
      const capture = await json(resolve(path,'ui-capture.json'));
      requireThat(capture.network.response.id === job.id && digest(capture.network.request) === digest(request) && capture.bindingSha256 === digest(binding),'Editor POST/job/backend identity changed');
      await validateRetainedEditor(state,row,expected,capture,path);
      requireThat(capture.actions?.length && capture.progress?.some(snapshot=>snapshot.id === job.id && snapshot.status === 'running'),'Editor capture lost its actual actions/running observation');
      for (const phase of ['before','after']) {
        const shot = capture.screenshots?.find(shot=>shot.phase === phase);
        requireThat(shot && (await fileIdentity(resolve(path,`ui-${phase}.png`))).sha256 === shot.sha256,'Editor before/after screenshot bytes changed or are missing');
      }
    } else assertRequestBinding(expected,request);
    if(row.transport.startsWith('ui-')) {
      const capture=await json(resolve(path,'ui-capture.json'));
      requireThat(capture.version===VERSION&&capture.rowId===row.id&&capture.lockSha256===digest(state.lock)&&capture.bindingSha256===digest(binding)&&capture.network.response.id===job.id&&digest(capture.network.request)===digest(request)&&capture.actions?.length&&capture.progress?.some(item=>item.id===job.id&&item.status==='running'),'retained actual product UI POST/actions/running evidence changed');
      for(const phase of ['before','after'])requireThat((await fileIdentity(resolve(path,`ui-${phase}.png`))).sha256===capture.screenshots?.find(item=>item.phase===phase)?.sha256,'retained actual UI screenshot changed');
      for(const name of uiGuards(row)){const observation=uiGuard(capture,name);requireThat((await fileIdentity(resolve(path,`ui-guard-${name}.png`))).sha256===observation.screenshot.sha256,'retained UI refusal screenshot changed');}
      if(row.action==='seeded-recipe')validateSeededRecipeSelection(row,request,capture.history,await json(resolve(state.root,'rows',row.replayOf,'receipt.json')),await json(resolve(state.root,'rows',row.replayOf,'asset-2.json')));
      if(row.action==='workflow'||row.id==='quality-cuda-base-four') {
        const workflow=capture.workflow,originId=row.replayOf??'quality-mlx-base-four',origin=await json(resolve(state.root,'rows',originId,'receipt.json')),index=workflow.originSampleIndex??0;
        requireThat(workflow.originRowId===originId&&workflow.originJobId===origin.jobId&&Number.isSafeInteger(index)&&index>=0&&index<origin.outputs.length,'retained portable export origin changed');
        const bytes=await readFile(resolve(path,workflow.file.endsWith('.png')?'replay.png':'replay.sceneworks-workflow.json'));
        requireThat(sha(bytes)===workflow.sha256&&canonical(await json(resolve(path,'workflow-inspection.json')))===canonical(workflow.inspection),'retained actual export/inspection bytes changed');
        validatePortableCapture(workflow,bytes,origin,await json(resolve(state.root,'rows',originId,`provider-report-${index}.json`)),index);
        requireThat(row.refs.every((ref,ordinal)=>workflow.reacquired.some(item=>item.ordinal===ordinal&&item.inputId===ref&&item.assetId===request.referenceAssetIds[ordinal])),'retained ordered workflow reacquisition changed');
      }
    }
    checkJob(row, request, job,binding.endpoint.workerGpuId);
    requireThat(job.workerId === binding.endpoint.workerId,'job executed on a different bound worker');
    for (const output of receipt.outputs) requireThat((await fileIdentity(safePath(path, output.file))).sha256 === output.sha256, 'output bytes changed');
    const samples = await json(resolve(path, 'memory.json')); requireThat(samples.device.uuid === receipt.device.uuid, 'memory sampling observed another device');
    requireThat(samples.samplerSha256===state.lock.files.actionLibrary?.sha256&&samples.raw&&(await fileIdentity(safePath(path,samples.raw.file))).sha256===samples.raw.sha256,'actual frozen sampler/raw observation bytes required');
    const raw=(await readFile(safePath(path,samples.raw.file),'utf8')).trimEnd().split('\n').map(line=>JSON.parse(line));
    requireThat(raw.length===samples.samples.length&&raw.every((item,index)=>item.device.uuid===receipt.device.uuid&&item.raw&&canonical({at:item.at,phase:item.phase,usedBytes:item.usedBytes})===canonical(samples.samples[index])),'raw whole-device observations differ from reported samples');
    const memory = memorySummary(samples, job.id);
    requireThat(Date.parse(samples.samples[0].at) <= Date.parse(job.startedAt) && Date.parse(samples.samples.at(-1).at) >= Date.parse(job.completedAt ?? job.canceledAt) && samples.samples.some(s => s.phase === 'active' && Date.parse(s.at) >= Date.parse(job.startedAt) && Date.parse(s.at) <= Date.parse(job.completedAt ?? job.canceledAt)), 'whole-device samples did not bracket actual worker execution');
    if (row.anchors) requireThat(reviewVerdict(row, await json(resolve(path, 'quality-review.json')), receipt.outputs) === 'PASS', 'required visual anchor failed');
    if (row.anchors) {
      const pairs = await json(resolve(path, 'visual-pairs.json'));
      requireThat(pairs.lockSha256 === digest(state.lock) && pairs.rowId === row.id && pairs.inputPairs?.length === row.refs.length && pairs.detailCrops?.length >= 3, 'retain full original/output pairs plus at least three detail crops');
      const inputs = await json(resolve(state.root,'input-identities.json'));
      for (const [i, entry] of pairs.inputPairs.entries()) requireThat(entry.inputId === row.refs[i] && entry.inputSha256 === inputs[row.refs[i]].sha256 && entry.outputSha256 === receipt.outputs[0].sha256, 'visual pairs are not bound to ordered original inputs/actual output');
      for (const entry of [...pairs.inputPairs, ...pairs.detailCrops]) requireThat((await fileIdentity(safePath(path, entry.file))).sha256 === entry.sha256, 'visual pair/crop bytes changed');
    }
    if (row.compareTo) { const comparison = await json(resolve(path, 'cache-comparison.json')), actual = await measuredCacheComparison(state,row); requireThat(canonical(comparison) === canonical(actual) && comparison.value <= state.plan.cacheComparisonPolicy.tolerance,'cached/uncached actual metric provenance/value mismatch or failed tolerance'); }
    summaries.push({ rowId: row.id, jobId: job.id, backend: row.backend, verdict: 'PASS', outputs: receipt.outputs.map(o => o.sha256), memory });
  });
  for (const backend of ['mlx', 'cuda']) for (const name of state.plan.requiredGuardCases) await requireEvidence(`${backend}/${name}`, async () => {
    const path = resolve(state.root, 'guards', `${backend}-${name}`), receipt = await json(resolve(path, 'receipt.json')), proof = await json(resolve(path, 'no-provider-load.json'));
    const binding = validateBinding(await json(resolve(state.root,'bindings',`${backend}.json`)),state.lock,backend);
    requireThat(receipt.bindingSha256 === digest(binding),'refusal belongs to another backend execution binding');
    const request = await readFile(resolve(path,receipt.rawJson?'request.raw.json':'request.json'));
    const specification=state.config.guardRequests[name];
    requireThat((await fileIdentity(safePath(path,receipt.responseFile))).sha256===receipt.responseSha256,'actual refusal HTTP response bytes changed');
    requireThat((receipt.rawJson?sha(request):digest(JSON.parse(request)))===receipt.requestSha256&&receipt.contentType===(specification.upload?'multipart/form-data':'application/json'),'retained exact refusal request bytes/content type changed');
    if(specification.upload) {
      requireThat(name==='host-budget-refusal'&&canonical(specification)===canonical(hostGuardSpec())&&receipt.route===`/api/v1/projects/${encode(binding.endpoint.projectId)}/assets`,'per-artifact route substituted for another refusal');
      requireThat((await readFile(resolve(path,'request.npy'))).equals(hostBudgetHeader())&&canonical(await json(safePath(state.root,specification.invariant)))===canonical(await hostTensorInvariant(state.config.sources.sceneworks)),'host header/integrated source invariant changed');
      const before=await json(resolve(path,'assets-before.json')),after=await json(resolve(path,'assets-after.json'));requireThat(Array.isArray(before)&&Array.isArray(after)&&canonical(before.map(asset=>asset.id).sort())===canonical(after.map(asset=>asset.id).sort()),'host refusal created asset/output');
      requireThat((await fileIdentity(resolve(path,'assets-before.json'))).sha256===receipt.assetSnapshotSha256.before&&(await fileIdentity(resolve(path,'assets-after.json'))).sha256===receipt.assetSnapshotSha256.after,'actual host asset-list snapshots changed');
    } else requireThat(receipt.route==='/api/v1/image/jobs','refusal left the exact image-job route');
    requireThat((proof.jobId??null)===(receipt.jobId??null),'refusal journal belongs to a different actual worker job');
    if(specification.expectedTerminal==='failed') {
      requireThat(['device-budget-before-load','corrupt-adapter'].includes(name),'worker refusal substituted for malformed HTTP refusal');
      const job=await json(resolve(path,'terminal-job.json')),queued=await json(resolve(path,'queued-response.json'));
      requireThat(job.id===receipt.jobId&&queued.job.id===job.id&&queued.http.status===201&&sha(queued.http.body)===queued.http.sha256&&Buffer.byteLength(queued.http.body)===queued.http.byteLength&&canonical(JSON.parse(queued.http.body))===canonical(queued.job)&&digest(job)===receipt.jobSha256,'queued/terminal worker-refusal identities changed');
      validateWorkerRefusal(specification,JSON.parse(request),job,binding.endpoint,backend);
    } else {
      const response=await json(resolve(path,'response.json'));requireThat(!receipt.jobId&&response.status===specification.expectedStatus&&specification.errorIncludes.every(text=>response.body?.includes(text)),'actual strict HTTP refusal differs from frozen case');
      requireThat(response.http?.status===response.status&&response.http.body===response.body&&sha(response.http.body)===response.http.sha256&&Buffer.byteLength(response.http.body)===response.http.byteLength,'actual refusal HTTP response body/status/hash were not retained');
    }
    requireThat(receipt.lockSha256 === digest(state.lock) && proof.version === VERSION && proof.lockSha256 === digest(state.lock) && proof.case === name && proof.backend === backend && proof.requestSha256 === receipt.requestSha256 && proof.method === OBSERVATION_METHOD && receipt.probe === OBSERVATION_METHOD && proof.observer && proof.rawEvidence?.length, 'missing source-bound refusal-before-MOSS-boundary evidence');
    for (const entry of proof.rawEvidence) requireThat((await fileIdentity(safePath(path, entry.file))).sha256 === entry.sha256, 'no-load probe evidence changed');
    const toolchain = await json(safePath(path,proof.toolchainFile));
    requireThat(canonical(await fileIdentity(safePath(path,proof.toolchainFile))) === canonical(state.lock.artifacts[backend].files.toolchain),'refusal probe toolchain/producer evidence differs from frozen native build');
    const actual = noLoadComparison(await readFile(safePath(path,proof.beforeFile)),await readFile(safePath(path,proof.afterFile)),{
      candidateId:state.lock.candidateId,sourceSha:state.lock.sources.sceneworks.head,inferenceSha:state.lock.sources.inference.head,workerId:binding.endpoint.workerId,
      gpuId:binding.endpoint.workerGpuId,
      binarySha256:state.lock.artifacts[backend].files.workerBinary.sha256,producerSha256:toolchain.observationProducerSha256
    },proof);
    requireThat(canonical(actual)===canonical(proof.observation),'refusal boundary observation was changed or fabricated');
  });
  for (const name of state.plan.requiredExternal) await requireEvidence(name, async () => {
    const evidence = await json(resolve(state.root, 'external', `${name}.json`));
    requireThat(evidence.version === VERSION && evidence.lockSha256 === digest(state.lock) && evidence.verdict === 'PASS' && evidence.observer && evidence.observations?.length && evidence.files?.length && evidence.rowIds?.length && evidence.rowIds.every(id => state.plan.rows.some(r => r.id === id)), 'required external evidence absent/unbound/failed');
    for (const entry of evidence.files) requireThat((await fileIdentity(safePath(state.root, entry.file))).sha256 === entry.sha256, 'external evidence changed');
    if(name==='package-install-select-ui')for(const backend of ['mlx','cuda']) {
      const toolchainPath=resolve(state.root,'external',`toolchain-${backend}.json`),toolchain=await json(toolchainPath);
      requireThat(canonical(await fileIdentity(toolchainPath))===canonical(state.lock.artifacts[backend].files.toolchain)&&toolchain.installation?.files?.length,'actual source-bound native installer capture required');
      for(const item of toolchain.installation.files)requireThat((await fileIdentity(safePath(resolve(state.root,'external',`raw-install-${backend}`),item.file))).sha256===item.sha256,'source-bound real installer capture bytes changed');
    }
  });
  const verdict = findings.some(f => f.state === 'FAIL') ? 'FAIL' : findings.length ? 'AWAITING' : 'PASS';
  // stdout is portable: raw product objects, local source paths and media tickets are excluded.
  console.log(JSON.stringify({ version: VERSION, lockSha256: digest(state.lock), verdict, counts: state.lock.counts, verifiedRows: summaries.length, findings, rows: summaries }, null, 2));
  if (verdict !== 'PASS') process.exitCode = 2;
}
export async function main(argv = process.argv.slice(2)) {
  const [command, ...args] = argv;
  if (command === 'prepare' && args.length === 2) return prepare(...args);
  if (command === 'freeze' && args.length === 1) return freeze(...args);
  if (command === 'bind' && args.length === 1) return bind(...args);
  if (command === 'upload' && args.length === 3) return upload(...args);
  if (command === 'ui-task' && args.length === 2) return uiTask(...args);
  if (command === 'run' && [3, 4].includes(args.length)) return run(...args);
  if (command === 'recover-completed' && args.length === 2) return recoverCompletedRow(...args);
  if (command === 'recover-canceled' && args.length === 2) return recoverCanceledRow(...args);
  if (command === 'guard' && args.length === 4) return guard(...args);
  if (command === 'compare-cache' && args.length === 2) return compareCache(...args);
  if (command === 'reconcile' && args.length === 4) return reconcile(...args);
  if (command === 'verify' && args.length === 1) return verify(...args);
  if (command === 'verify-retained' && args.length === 1) return verify(args[0],true);
  throw new Error('Usage: moss-edit-terminal.mjs prepare INPUTS_JSON FRESH_OUTPUT_DIR | freeze LAUNCH_JSON | bind LAUNCH_JSON | upload LAUNCH_JSON BACKEND ADMISSION_JSON | ui-task LAUNCH_JSON ROW | run LAUNCH_JSON ROW ADMISSION_JSON [UI_CAPTURE_JSON] | guard LAUNCH_JSON CASE BACKEND ADMISSION_JSON | compare-cache LAUNCH_JSON ROW | reconcile LAUNCH_JSON BACKEND ATTEMPT PROOF_JSON | verify LAUNCH_JSON | verify-retained LAUNCH_JSON');
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch(error => { console.error(JSON.stringify({ status: 'STOP', message: error.message })); process.exitCode = 1; });
