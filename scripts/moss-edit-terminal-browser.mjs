#!/usr/bin/env node
// Real React interactions only. Importing this module cannot start a browser or job.
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, isAbsolute, basename, dirname } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { acknowledgeRecipeChoice, advanced, click, fill, select, checked, modelPicker, labelledSelect, configure, verifyReplayForm } from './lib/moss-edit-terminal-controls.mjs';
export { labelledSelect, configure, verifyReplayForm } from './lib/moss-edit-terminal-controls.mjs';
import { frozen } from './moss-edit-terminal.mjs';
import { VERSION, sha, digest, requireThat, json, save, replaceInputs, assertRequestBinding, resumeUiAttempt, LATENT_UPSCALE_GUARDS } from './lib/moss-edit-terminal.mjs';

// Parents may be new on the first row; each capture leaf must remain exclusive.
export async function createFreshCaptureDirectory(captureRoot) {
  requireThat(typeof captureRoot==='string'&&isAbsolute(captureRoot),'capture directory must be absolute');
  await mkdir(dirname(captureRoot),{recursive:true,mode:0o700});
  await mkdir(captureRoot,{mode:0o700});
}

const tensorPicker=labelledSelect;
const now = () => new Date().toISOString();
const endpoint = value => new URL(value).pathname;

// Extract only the actual file part, never multipart credentials or a reconstructed raster.
export function multipartFile(bytes, contentType) {
  const boundary = /boundary=(?:"([^"]+)"|([^;\s]+))/i.exec(contentType ?? '');
  requireThat(Buffer.isBuffer(bytes) && boundary, 'observed upload lacks multipart file bytes');
  const marker = Buffer.from(`\r\n--${boundary[1] ?? boundary[2]}`);
  let start = bytes.indexOf(Buffer.from(`--${boundary[1] ?? boundary[2]}\r\n`));
  requireThat(start === 0, 'invalid multipart boundary');
  while (start >= 0) {
    const headerStart = bytes.indexOf(Buffer.from('\r\n'), start) + 2;
    const cut = bytes.indexOf(Buffer.from('\r\n\r\n'), headerStart);
    requireThat(cut >= headerStart && cut - headerStart < 16384, 'invalid multipart headers');
    const end = bytes.indexOf(marker, cut + 4);
    requireThat(end >= 0, 'unterminated observed multipart upload');
    const headers = bytes.subarray(headerStart, cut).toString('utf8');
    if (/content-disposition:[^\r\n]*\bname="file"[^\r\n]*\bfilename=/i.test(headers)) return bytes.subarray(cut + 4, end);
    start = bytes.subarray(end + marker.length, end + marker.length + 2).toString() === '--' ? -1 : end + 2;
  }
  throw Error('observed multipart upload has no file part');
}

export function jobSnapshots(value) {
  if (Array.isArray(value)) return value.flatMap(jobSnapshots);
  if (!value || typeof value !== 'object') return [];
  if (typeof value.id === 'string' && typeof value.status === 'string') return [value];
  return ['jobs','items','data','job'].flatMap(key => value[key] ? jobSnapshots(value[key]) : []);
}

export function eventSnapshots(type, data) {
  requireThat(['job.updated','jobs.snapshot'].includes(type), 'not a job SSE event');
  return jobSnapshots(JSON.parse(data));
}

export function describeUiRow(row) {
  requireThat(row && /^ui-(full|simple|editor)$/.test(row.transport), 'row is not an allocated UI route');
  requireThat(!row.cancelAtDenoise, 'UI cancellation is not allocated by this protocol');
  const workflow = row.action === 'workflow' || row.id === 'quality-cuda-base-four';
  const originRowId = workflow ? row.replayOf ?? 'quality-mlx-base-four' : row.transport === 'ui-editor' ? 'quality-mlx-sft-single' : row.replayOf;
  return { shell: row.transport.slice(3), workflow, originRowId, sampleIndex: row.action === 'seeded-recipe' ? 2 : 0,
    history: row.action === 'seeded-recipe' || row.transport === 'ui-editor', latentGuards: row.id === `advanced-${row.backend}-sft-replay` };
}

// Response observation is passive: no routing, fulfillment, synthetic responses or fetches.
function observe(page, apiOrigin) {
  const pending = new Set(), errors = [], requests = new WeakMap();
  const result = { submissions: [], installations: [], uploads: [], inspections: [], progress: [], posts: [] };
  page.on('request', request => {
    if (new URL(request.url()).origin !== apiOrigin) return;
    const observedAt = now(); requests.set(request, observedAt);
    if (request.method() === 'POST' && endpoint(request.url()) === '/api/v1/image/jobs') result.posts.push({ observedAt, path: endpoint(request.url()) });
  });
  page.on('response', response => {
    const task = (async () => {
      const request = response.request(), url = new URL(response.url());
      if (url.origin !== apiOrigin || !url.pathname.startsWith('/api/v1/')) return;
      const path = url.pathname, method = request.method(), observedAt = requests.get(request);
      if (!observedAt) return;
      const installation = /^\/api\/v1\/models\/moss_image_(?:base|sft)\/conditional-components\/moss_image_edit\/download$/.test(path);
      if (method === 'POST' && (installation || path === '/api/v1/image/jobs' || path === '/api/v1/workflows/inspect' || /^\/api\/v1\/projects\/[^/]+\/assets$/.test(path))) {
        const body = await response.json();
        const network = { method, path, observedAt, responseObservedAt: now(), status: response.status(), response: body };
        if (installation || path === '/api/v1/image/jobs') { network.request = request.postDataJSON(); (installation ? result.installations : result.submissions).push(network); }
        else {
          const bytes = multipartFile(request.postDataBuffer(), request.headers()['content-type']);
          network.request = { fileSha256: sha(bytes), byteLength: bytes.length };
          (path === '/api/v1/workflows/inspect' ? result.inspections : result.uploads).push({ network, bytes });
        }
      } else if (/^\/api\/v1\/(?:jobs(?:\/|$)|image\/jobs(?:\/|$))/.test(path) && response.ok() && (response.headers()['content-type'] ?? '').includes('json')) {
        for (const job of jobSnapshots(await response.json())) result.progress.push({ ...job, observedAt: now(), observationPath: path });
      }
    })().catch(error => errors.push(error));
    pending.add(task); task.finally(() => pending.delete(task));
  });
  return { ...result, async flush() { while (pending.size) await Promise.all([...pending]); if (errors.length) throw errors[0]; } };
}

async function observeSse(context, page, base, observation, actions) {
  const cdp = await context.newCDPSession(page), eventSources = new Set();
  await cdp.send('Network.enable');
  cdp.on('Network.requestWillBeSent', event => { const url = new URL(event.request.url); if (url.origin === base.origin && url.pathname === '/api/v1/jobs/events') eventSources.add(event.requestId); });
  cdp.on('Network.eventSourceMessageReceived', event => {
    if (!eventSources.has(event.requestId) || !['job.updated','jobs.snapshot'].includes(event.eventName)) return;
    try { for (const job of eventSnapshots(event.eventName,event.data)) observation.progress.push({ ...job, observedAt: now(), observationPath: '/api/v1/jobs/events', eventName: event.eventName, eventId: event.eventId }); }
    catch (error) { actions.push({ at: now(), action: 'invalid-actual-sse-frame', error: error.message }); }
  });
  return cdp;
}

async function waitFor(page, observation, predicate, message, timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) { await observation.flush(); const value = await predicate(); if (value) return value; await page.waitForTimeout(100); }
  throw Error(message);
}

async function waitValue(page, locator, value) {
  await locator.waitFor({ state: 'visible' });
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) { if (await locator.inputValue() === String(value)) return; await page.waitForTimeout(100); }
  throw Error('actual UI did not restore the expected control value before the deadline');
}

async function screenshot(page, root, phase) { const file = `${phase}.png`; await page.screenshot({ path: resolve(root, file), fullPage: true, animations: 'disabled' }); return { phase, file, sha256: sha(await readFile(resolve(root, file))) }; }
async function readonly(context, base, path, token) {
  requireThat(path.startsWith('/api/v1/'), 'readback must use a product API path');
  const response = await context.request.get(new URL(path, base).href, { headers: { 'X-SceneWorks-Token': token }, maxRedirects: 0 });
  requireThat(response.ok(), `authenticated UI identity readback failed: ${path} HTTP ${response.status()}`); return response.json();
}

async function shell(page, kind, actions) {
  await click(page.getByRole('radio', { name: kind === 'simple' ? 'Simple' : 'Advanced', exact: true }), actions, `${kind} interface`);
  await click(page.locator('nav').getByRole('button', { name: 'Image', exact: true }), actions, 'Image navigation');
}

export async function openAsset(page, asset, actions) {
  requireThat(asset?.id && typeof asset.displayName === 'string' && asset.displayName, 'original asset lacks its UI identity');
  await click(page.locator('nav').getByRole('button', { name: 'Assets', exact: true }), actions, 'Assets navigation');
  await fill(page.getByRole('searchbox', { name: 'Search assets', exact: true }),asset.displayName,actions,'Search assets');
  await select(page.getByRole('combobox', { name: 'Asset type', exact: true }),'all',actions,'Asset type');
  // The accessible tile is source-owned in AssetGrid. Do not index a gallery by position.
  const tile = page.locator(`.asset-tile[data-asset-id=${JSON.stringify(asset.id)}]`);
  await tile.waitFor({ state: 'visible' });
  requireThat(await tile.count() === 1, `original asset tile is ambiguous: ${asset.id}`);
  await tile.dblclick(); actions.push({ at: now(), action: 'open-asset', assetId: asset.id, displayName: asset.displayName });
  await page.getByRole('button', { name: 'Save As…', exact: true }).waitFor({ state: 'visible' });
}

async function originFor(state, rowId, sampleIndex, read) {
  const receipt = await json(resolve(state.root, 'rows', rowId, 'receipt.json'));
  requireThat(receipt.execution === 'VERIFIED' && receipt.outputs[sampleIndex]?.assetId, 'original row/sample is not verified');
  const assetId = receipt.outputs[sampleIndex].assetId, project = state.binding.endpoint.projectId;
  const asset = await read(`/api/v1/projects/${encodeURIComponent(project)}/assets/${encodeURIComponent(assetId)}`);
  const job = await read(`/api/v1/jobs/${encodeURIComponent(receipt.jobId)}`);
  requireThat(asset.id === assetId && asset.lineage?.jobId === receipt.jobId && job.id === receipt.jobId && job.status === 'completed', 'original UI history identity mismatch');
  return { receipt, asset, job };
}

async function historyReplay(page, row, route, origin, actions) {
  await openAsset(page, origin.asset, actions);
  await checked(page.getByLabel('Keep seed', { exact: true }), true, actions, 'Keep seed');
  await click(page.getByRole('button', { name: 'Use this recipe', exact: true }), actions, 'Use this recipe');
  await waitValue(page,page.getByLabel('Prompt', { exact: true }),row.request.prompt);
  await advanced(page, actions);
  // Read the form before any changes. A recipe copied from disk alone is not hydration proof.
  const prompt = await page.getByLabel('Prompt', { exact: true }).inputValue();
  requireThat(prompt === row.request.prompt, 'history hydration changed original prompt0');
  if (row.action === 'seeded-recipe') {
    requireThat(await page.getByLabel('One positive prompt per conditioning row', { exact: true }).inputValue() === row.request.prompts.join('\n'), 'history lost full conditioning batch');
    requireThat(Number(await page.getByLabel('Variations', { exact: true }).inputValue()) === row.request.count && Number(await page.getByLabel('Seed', { exact: true }).inputValue()) === row.request.seed, 'Keep seed lost count/base seed');
  }
  return { originRowId: route.originRowId, originJobId: origin.receipt.jobId, originAssetId: origin.asset.id, originSampleIndex: route.sampleIndex,
    selectedAssetId: origin.asset.id, selectedRecipe: origin.asset.recipe, hydrationObserved: true, keepSeed: true, observedAt: now(), form: { prompt } };
}

async function dropFile(page, path, actions) {
  const bytes = await readFile(path);
  // Real HTML file drag/drop, dispatched at the global unclaimed-drop target. No React state injection.
  const transfer = await page.evaluateHandle(({ data, name, type }) => { const transfer = new DataTransfer(); transfer.items.add(new File([Uint8Array.from(data)], name, { type })); return transfer; }, { data: [...bytes], name: basename(path), type: path.endsWith('.png') ? 'image/png' : 'application/json' });
  try { await page.locator('body').dispatchEvent('dragover', { dataTransfer: transfer }); await page.locator('body').dispatchEvent('drop', { dataTransfer: transfer }); }
  finally { await transfer.dispose(); }
  actions.push({ at: now(), action: 'drop-workflow-file', file: basename(path), sha256: sha(bytes) });
}

async function workflowReplay(page, state, row, origin, expected, uploaded, observation, actions, root) {
  await openAsset(page, origin.asset, actions);
  const tensor = origin.asset.type === 'tensor';
  const downloadEvent = page.waitForEvent('download');
  await click(tensor ? page.getByRole('link', { name: 'Download portable workflow (.json)', exact: true }) : page.getByRole('button', { name: 'Save As…', exact: true }), actions, tensor ? 'Download portable workflow (.json)' : 'Save As…');
  const download = await downloadEvent, file = tensor ? 'original.sceneworks-workflow.json' : 'original.png';
  requireThat(!await download.failure(), 'actual browser workflow download failed'); await download.saveAs(resolve(root, file));
  const bytes = await readFile(resolve(root, file));
  if (!tensor) requireThat(sha(bytes) === origin.receipt.outputs[0].sha256, 'Save As downloaded different original image bytes');
  actions.push({ at: now(), action: 'download-workflow', file, sha256: sha(bytes), originAssetId: origin.asset.id });
  await page.keyboard.press('Escape');
  const inspectionsBefore = observation.inspections.length;
  await dropFile(page, resolve(root, file), actions);
  const inspection = await waitFor(page, observation, () => observation.inspections[inspectionsBefore], 'real workflow inspection did not occur');
  requireThat(inspection.network.status === 200 && inspection.network.response.status === 'workflow' && inspection.network.response.workflow && inspection.network.request.fileSha256 === sha(bytes), 'actual export is not a portable workflow');
  const dialog = page.getByRole('dialog'), use = dialog.getByRole('button', { name: 'Use this workflow', exact: true });
  await use.waitFor({ state: 'visible' });
  await noPostAttempt(page, observation, use, actions, root, 'workflow-unresolved', { resolution: inspection.network.response.resolution, visibleText: await dialog.innerText() });
  const reacquired = [];
  for (const [index, inputId] of row.refs.entries()) {
    const label = row.refs.length === 1 ? 'Reference image' : `Reference image ${index + 1} of ${row.refs.length}`;
    await select(dialog.getByLabel(label, { exact: true }), expected.referenceAssetIds[index], actions, label);
    reacquired.push({ ordinal: index, inputId, assetId: expected.referenceAssetIds[index], observedAt: now() });
  }
  const tensors = inspection.network.response.resolution?.tensors ?? [];
  for (const tensor of tensors) {
    const match = Object.entries(uploaded).find(([,asset]) => asset.type === 'tensor' && asset.tensor?.sha256 === tensor.sha256);
    requireThat(match, `no original tensor for portable role ${tensor.role}`);
    await select(tensorPicker(dialog,`Exact ${tensor.role} tensor`), match[1].id, actions, `Exact ${tensor.role} tensor`);
    reacquired.push({ role: tensor.role, inputId: match[0], assetId: match[1].id, observedAt: now() });
  }
  requireThat(await use.isEnabled(), 'workflow remained unresolved after exact input reacquisition');
  await click(use, actions, 'Use this workflow');
  await dialog.waitFor({ state: 'hidden' });
  await waitValue(page,page.getByLabel('Prompt', { exact: true }),row.request.prompt);
  return { file, sha256: sha(bytes), inspectUploadSha256: inspection.network.request.fileSha256, inspection: inspection.network.response, inspectionNetwork: inspection.network,
    originRowId: row.replayOf ?? 'quality-mlx-base-four', originJobId: origin.receipt.jobId, originSampleIndex: 0, reacquired };
}

async function noPostAttempt(page, observation, button, actions, root, name, detail) {
  requireThat(await button.isDisabled(), `${name} did not disable submission`);
  const startedAt = now(), before = observation.posts.length;
  // A real pointer press on a disabled HTML button is deliberately ignored by the browser.
  await button.click({ force: true, timeout: 2000 });
  await page.waitForTimeout(1000); await observation.flush();
  const posts = observation.posts.slice(before);
  requireThat(posts.length === 0, `${name} dispatched an image job`);
  const shot = await screenshot(page, root, name);
  actions.push({ action: 'no-post-guard', name, startedAt, endedAt: now(), disabled: true, postCountBefore: before, postCountAfter: observation.posts.length, posts, screenshot: shot, ...detail });
}

export function completedUiJob(progress, id, outputCount) {
  const job = progress.filter(item => item.id === id).at(-1);
  requireThat(!job || !['failed','canceled','interrupted'].includes(job.status), 'allocated UI generation did not complete successfully');
  return job?.status === 'completed' && job.result?.outputPublication?.state !== 'pending' && job.result?.assets?.length === outputCount ? job : null;
}

async function waitResultUi(page, observation, route, row, job, read, actions) {
  const assetIds = job.result.assets.map(asset => asset.id);
  requireThat(assetIds.every(id => typeof id === 'string' && id), 'completed UI job lacks actual canonical output IDs');
  const asset = await read(`/api/v1/projects/${encodeURIComponent(job.projectId)}/assets/${encodeURIComponent(assetIds[0])}`);
  requireThat(asset.id === assetIds[0] && asset.lineage?.jobId === job.id, 'actual completed output readback belongs to another job');
  if (route.shell === 'editor') {
    const outputs = page.getByRole('region',{name:'Retained MOSS edit outputs',exact:true});
    await outputs.waitFor({state:'visible'});
    const link = outputs.getByRole('link',{name:'Output 1: image',exact:true});
    await link.waitFor({state:'visible'});
    const path = new URL(await link.getAttribute('href'),new URL(page.url()).origin).pathname;
    requireThat(decodeURIComponent(path).endsWith(`/${asset.file.path}`), 'Editor result link did not bind the completed output');
    await page.locator('.ie-unsaved-badge').waitFor({state:'visible'});
    await page.locator('.ie-canvas .konvajs-content canvas').first().waitFor({state:'visible'});
    requireThat(await page.locator('.ie-busy').count() === 0, 'Editor still shows its active operation');
  } else if (row.output === 'latents') {
    const card = page.locator('.studio-results .worker-progress-card.completed').filter({has:page.getByTitle(job.id,{exact:true})});
    await card.waitFor({state:'visible'});
    await card.getByText(asset.displayName,{exact:true}).waitFor({state:'visible'});
    requireThat(await card.locator('.worker-progress-card__thumb-cell').count() === row.outputCount && (await card.innerText()).includes('FP32 tensor'), 'actual native latent results have not rendered in the allocated Studio');
  } else {
    const images = page.locator(route.shell === 'simple' ? '.su-results-grid .su-result-open img' : '.studio-results img');
    await waitFor(page,observation,async () => images.evaluateAll((elements,path) => elements.some(image => image.complete && image.naturalWidth > 0 && decodeURIComponent(new URL(image.src).pathname).endsWith(`/${path}`)),asset.file.path), 'actual completed output image did not render in its allocated Studio',30000);
  }
  const evidence = {at:now(),action:'observe-rendered-completed-result',jobId:job.id,assetIds,firstAssetId:asset.id,shell:route.shell,output:row.output};
  actions.push(evidence); return evidence;
}

export function installerComponent(catalog, model) {
  requireThat(['moss_image_base','moss_image_sft'].includes(model), 'installer requires an exact MOSS Base/SFT catalog ID');
  const entry = (Array.isArray(catalog) ? catalog : catalog?.models)?.find(item => item.id === model);
  requireThat(entry?.installState === 'installed', 'install preflight requires the complete base model in its owned projection');
  const purpose = entry.conditionalPurposes?.moss_image_edit;
  requireThat(purpose && !purpose.blocked && ['missing','incomplete'].includes(purpose.installState), 'install proof requires a genuinely missing/incomplete unblocked edit adapter; preinstalled is not installation evidence');
  const components = entry.conditionalComponents?.filter(item => item.requiredFor?.includes('moss_image_edit'));
  requireThat(components?.length === 1, 'MOSS edit installer must identify its exact conditional component');
  const component = components[0];
  requireThat(component.componentId === 'moss-image-edit' && component.repo === 'chinoll/Moss-Image-Edit' && /^[a-f0-9]{40}$/.test(component.revision ?? '') && !component.blocked, 'installer catalog does not identify the pinned MOSS edit adapter');
  return { entry, component };
}

export function installedDownloadJobs({ model, component, network, progress, utilityWorkerId }) {
  requireThat(typeof utilityWorkerId === 'string' && utilityWorkerId, 'installer requires its actual owned CPU utility worker identity');
  requireThat(network?.method === 'POST' && network.path === `/api/v1/models/${model}/conditional-components/moss_image_edit/download` && network.status === 201 && network.request?.licenseAcknowledged === true && network.response?.purpose === 'moss_image_edit', 'missing actual acknowledged edit-adapter download POST');
  const jobs = network.response.jobs;
  requireThat(Array.isArray(jobs) && jobs.length === 1 && new Set(jobs.map(job => job.id)).size === jobs.length && jobs.every(job => typeof job.id === 'string' && job.id && job.type === 'model_download'), 'actual installer must queue its one native model_download job; already-installed/no-job response cannot pass');
  const queued = network.response.components;
  requireThat(Array.isArray(queued) && queued.length === 1 && queued[0].componentId === component.componentId && queued[0].repo === component.repo && queued[0].revision === component.revision && queued[0].status === 'queued' && queued[0].jobId === jobs[0].id, 'download response changed pinned component/job identity');
  for (const job of jobs) {
    const observed = progress.filter(item => item.id === job.id);
    const downloading = observed.slice(0,-1).filter(item => item.status === 'downloading'), completed = observed.at(-1);
    requireThat(downloading.length && completed?.status === 'completed', 'installer needs actual downloading and completed native download observations');
    requireThat(observed.every(item => !item.error && !['failed','canceled','interrupted'].includes(item.status)), 'installer observed an unsuccessful native download');
    requireThat([...downloading,completed].every(item => item.workerId === utilityWorkerId && item.assignedGpu === 'cpu' && item.stage === item.status), 'native download observations belong to another worker/device or stage');
    for (const item of [job,...observed]) requireThat(item.type === 'model_download' && item.payload?.modelId === model && item.payload?.repo === component.repo && item.payload?.revision === component.revision, 'observed download progress belongs to another model/repo/revision');
  }
  return jobs.map(job => job.id);
}

// Separate pre-freeze proof. The controller owns the isolated missing-adapter
// projection and final package hashing; this function never edits any HF cache.
export async function capturePackageInstallUi({ apiUrl, projectId, captureRoot, playwrightModule, token, model, utilityWorkerId }) {
  requireThat(typeof captureRoot === 'string' && isAbsolute(captureRoot) && typeof playwrightModule === 'string' && isAbsolute(playwrightModule), 'installer requires absolute fresh capture and Playwright module paths');
  requireThat(typeof utilityWorkerId === 'string' && utilityWorkerId, 'installer requires its actual owned CPU utility worker identity');
  const base = new URL(apiUrl);
  requireThat(['http:','https:'].includes(base.protocol) && !base.username && !base.password && !base.search && !base.hash && projectId && token, 'installer requires the authenticated isolated API/project');
  requireThat(['moss_image_base','moss_image_sft'].includes(model), 'installer requires an exact MOSS Base/SFT catalog ID');
  await createFreshCaptureDirectory(captureRoot);
  const imported = await import(pathToFileURL(playwrightModule).href), chromium = imported.chromium ?? imported.default?.chromium;
  requireThat(chromium?.launch, 'explicit module did not export Playwright chromium');
  const browser = await chromium.launch({headless:true,args:['--disable-gpu','--disable-webgl','--disable-dev-shm-usage']});
  let context, page, observation;
  const actions = [], screenshots = [], startedAt = now();
  try {
    context = await browser.newContext({viewport:{width:1600,height:1200},acceptDownloads:true,serviceWorkers:'block'});
    page = await context.newPage(); page.setDefaultTimeout(30000); observation = observe(page,base.origin);
    await observeSse(context,page,base,observation,actions);
    const read = path => readonly(context,base,path,token), catalogBefore = await read('/api/v1/models');
    const {entry,component} = installerComponent(catalogBefore,model);
    const projects = await read('/api/v1/projects'), project = (Array.isArray(projects) ? projects : projects.projects)?.find(item => item.id === projectId);
    requireThat(project?.name, 'installer original owned project is absent');
    await page.goto(base.href,{waitUntil:'domcontentloaded'});
    await page.getByRole('radio',{name:'Advanced',exact:true}).or(page.getByLabel('Password',{exact:true})).waitFor({state:'visible'});
    if (await page.getByLabel('Password',{exact:true}).isVisible()) { await page.getByLabel('Password',{exact:true}).fill(token); await click(page.getByRole('button',{name:'Unlock',exact:true}),actions,'Unlock isolated install preflight'); }
    await shell(page,'full',actions);
    await click(page.locator('.project-pill'),actions,'Workspace picker');
    await click(page.getByRole('option').filter({hasText:project.name}),actions,`Workspace ${project.name}`);
    await click(page.getByRole('tab',{name:'Edit',exact:true}),actions,'Image Edit');
    await select(modelPicker(page),model,actions,'Model');
    const controls = page.getByRole('group',{name:'MOSS Image Edit',exact:true}), status = controls.getByRole('status'), install = controls.getByRole('button',{name:'Install MOSS edit adapter',exact:true});
    await status.waitFor({state:'visible'});
    const statusBefore = await status.innerText();
    requireThat(/^Edit adapter: (missing|incomplete)\./.test(statusBefore.trim()), 'actual UI adapter was already installed or blocked; cannot claim an install');
    requireThat(await install.isDisabled(), 'installer did not require explicit source/license acknowledgment');
    screenshots.push(await screenshot(page,captureRoot,'install-before'));
    await checked(controls.getByLabel('I acknowledge the adapter source and unspecified weight license.',{exact:true}),true,actions,'Acknowledge adapter source and unspecified weight license');
    requireThat(await install.isEnabled(), 'actual missing-adapter install remains blocked after acknowledgment');
    await click(install,actions,'Install MOSS edit adapter');
    const network = await waitFor(page,observation,() => observation.installations[0],'actual UI did not submit the edit-adapter installer');
    requireThat(network.path === `/api/v1/models/${model}/conditional-components/moss_image_edit/download` && network.status === 201 && network.response?.jobs?.length === 1,'actual installer did not queue one native adapter download');
    const ids = network.response.jobs.map(job => job.id);
    await waitFor(page,observation,() => {
      for (const id of ids) {
        const states = observation.progress.filter(job => job.id === id), last = states.at(-1);
        requireThat(!last || !['failed','canceled','interrupted'].includes(last.status), `actual installer job ${id} ended ${last?.status}`);
      }
      return ids.every(id => observation.progress.filter(job => job.id === id).at(-1)?.status === 'completed');
    },'actual native adapter download did not complete',60*60*1000);
    const progress = observation.progress.filter(job => ids.includes(job.id));
    installedDownloadJobs({model,component,network,progress,utilityWorkerId});
    // The application's actual SSE completion refreshes its model catalog; do
    // not reload/override state to conceal a broken installed-state transition.
    await controls.getByText(/^Edit adapter: installed\./).waitFor({state:'visible'});
    const statusAfter = await status.innerText(), catalogAfter = await read('/api/v1/models');
    const after = (Array.isArray(catalogAfter) ? catalogAfter : catalogAfter.models)?.find(item => item.id === model);
    requireThat(after?.conditionalPurposes?.moss_image_edit?.installState === 'installed' && await install.count() === 0 && observation.posts.length === 0 && observation.installations.length === 1, 'installer did not finish exactly once without a generation dispatch');
    screenshots.push(await screenshot(page,captureRoot,'install-after'));
    const capture = {version:VERSION,phase:'pre-freeze-package-install',observer:'actual-playwright-react-ui/v1',apiUrl:base.href,projectId,model,utilityWorkerId,startedAt,capturedAt:now(),network,jobIds:ids,progress,actions,screenshots,statusBefore,statusAfter,catalogBefore:entry,catalogAfter:after,component,imageJobPosts:observation.posts};
    const path = resolve(captureRoot,'capture.json'); await save(path,capture); return path;
  } catch (error) {
    if (page) await screenshot(page,captureRoot,'install-failure').catch(() => {});
    await save(resolve(captureRoot,'failure.json'),{version:VERSION,phase:'pre-freeze-package-install',apiUrl:base.href,projectId,model,at:now(),error:error.message,actions,screenshots,installations:observation?.installations ?? [],progress:observation?.progress ?? [],imageJobPosts:observation?.posts ?? []}).catch(() => {});
    throw error;
  } finally { try { await context?.close(); } finally { await browser.close(); } }
}

export async function captureUi({ launchPath, rowId, captureRoot, playwrightModule, token }) {
  requireThat([launchPath,captureRoot,playwrightModule].every(isAbsolute), 'browser driver requires absolute launcher/capture/module paths');
  const state = await frozen(launchPath), row = state.plan.rows.find(item => item.id === rowId), route = describeUiRow(row);
  requireThat(row.backend === state.config.localBackend && state.binding.backend === row.backend, 'browser row belongs to another execution binding');
  const expected = replaceInputs(row, await json(resolve(state.root, `uploaded-${row.backend}.json`)), state.binding.endpoint.projectId);
  const intent = await json(resolve(state.root, 'attempts', row.backend, `${row.id}.start.json`));
  requireThat(intent.transport === 'ui' && intent.lockSha256 === digest(state.lock) && intent.bindingSha256 === digest(state.binding) && intent.requestSha256 === digest(expected), 'controller must reserve this exact UI attempt once before capture');
  await resumeUiAttempt(state.root,row.backend,row.id,{lockSha256:digest(state.lock),bindingSha256:digest(state.binding),requestSha256:digest(expected)});
  token ??= process.env[state.binding.endpoint.tokenEnv]; requireThat(typeof token === 'string' && token, 'bound launcher token is missing');
  await createFreshCaptureDirectory(captureRoot);
  const imported = await import(pathToFileURL(playwrightModule).href), chromium = imported.chromium ?? imported.default?.chromium;
  requireThat(chromium?.launch, 'explicit module did not export Playwright chromium');
  const browser = await chromium.launch({ headless: true, args: ['--disable-gpu','--disable-webgl','--disable-dev-shm-usage'] });
  let context;
  const setupFailure = async error => { try { await context?.close(); } finally { await browser.close(); } throw error; };
  context = await browser.newContext({ viewport: { width: 1600, height: 1200 }, acceptDownloads: true, serviceWorkers: 'block' }).catch(setupFailure);
  const page = await context.newPage().catch(setupFailure); page.setDefaultTimeout(30000);
  const base = new URL(state.binding.endpoint.apiUrl), observation = observe(page, base.origin), actions = [], shots = [];
  // Chromium reports the browser's actual EventSource network frames. No injected listener,
  // replacement EventSource, direct polling or synthetic progress is used as UI evidence.
  await observeSse(context,page,base,observation,actions).catch(setupFailure);
  let capture;
  try {
    await page.goto(base.href, { waitUntil: 'domcontentloaded' });
    await page.getByRole('radio', { name: 'Advanced', exact: true }).or(page.getByLabel('Password', { exact: true })).waitFor({ state: 'visible' });
    if (await page.getByLabel('Password', { exact: true }).isVisible()) { await page.getByLabel('Password', { exact: true }).fill(token); await click(page.getByRole('button', { name: 'Unlock', exact: true }), actions, 'Unlock bound isolated API'); }
    await shell(page, 'full', actions);
    const read = path => readonly(context, base, path, token), projects = await read('/api/v1/projects');
    const list = Array.isArray(projects) ? projects : projects.projects;
    const project = list?.find(item => item.id === state.binding.endpoint.projectId); requireThat(project?.name, 'bound original project missing from target API');
    await click(page.locator('.project-pill'), actions, 'Workspace picker');
    await click(page.getByRole('option').filter({ hasText: project.name }), actions, `Workspace ${project.name}`);
    const uploaded = await json(resolve(state.root, `uploaded-${row.backend}.json`));
    const origin = route.originRowId ? await originFor(state, route.originRowId, route.sampleIndex, read) : null;
    let history, workflow;
    if (route.history) history = await historyReplay(page, row, route, origin, actions);
    if (route.workflow) workflow = await workflowReplay(page, state, row, origin, expected, uploaded, observation, actions, captureRoot);
    if (route.shell === 'editor') {
      await openAsset(page, uploaded.B, actions); await click(page.getByRole('button', { name: 'Edit', exact: true }), actions, 'Edit original B in Image Editor');
      await click(page.locator('.ie-acc-head').filter({ hasText: /^AI Edit$/ }), actions, 'Editor AI Edit tool');
    } else if (route.shell === 'simple') {
      await shell(page, 'simple', actions); await click(page.getByRole('tab', { name: 'Edit', exact: true }), actions, 'Simple Edit');
    } else if (!route.workflow && !route.history) await click(page.getByRole('button', { name: 'Edit', exact: true }).or(page.getByRole('tab', { name: 'Edit', exact: true })), actions, 'Image Edit');
    if (route.shell === 'full' && (route.workflow || route.history)) await verifyReplayForm(page, expected, actions);
    else await configure(page, expected, route.shell, actions);
    if (route.latentGuards) {
      for (const guard of LATENT_UPSCALE_GUARDS) {
        const engine = guard.slice('latent-inline-upscale-'.length);
        await checked(page.getByLabel('Upscale', { exact: true }), true, actions, 'Upscale');
        await select(labelledSelect(page,'Engine'), engine, actions, 'Engine');
        const error = page.getByText('MOSS latent output cannot use inline upscale', { exact: false });
        await error.first().waitFor({ state: 'visible' });
        await noPostAttempt(page, observation, page.getByRole('button', { name: 'Generate', exact: true }), actions, captureRoot, guard, { engine, output: 'latents', error: await error.first().innerText() });
        await checked(page.getByLabel('Upscale', { exact: true }), false, actions, 'Upscale');
      }
    }
    shots.push(await screenshot(page, captureRoot, 'before'));
    requireThat(observation.posts.length === 0, 'UI interaction dispatched an unallocated image job before Generate');
    const generateName = route.shell === 'editor' ? 'Generate edit' : route.shell === 'simple' ? `Generate ${expected.count} image${expected.count > 1 ? 's' : ''}` : 'Generate';
    const generate = page.getByRole('button', { name: generateName, exact: true });
    requireThat(await generate.isEnabled(), 'frozen UI request is not runnable; inspect the retained screen');
    await click(generate, actions, generateName);
    const network = await waitFor(page, observation, async () => { if(observation.submissions[0])return observation.submissions[0];await acknowledgeRecipeChoice(page,actions);return observation.submissions[0]; }, 'actual UI did not submit the allocated image job');
    requireThat(network.status >= 200 && network.status < 300 && network.response?.id && observation.posts.length === 1, 'UI created other than one successful allocated job');
    let sourceUpload;
    if (route.shell === 'editor') {
      const actualId = network.request.referenceAssetIds?.[0], upload = observation.uploads.find(item => item.network.response?.id === actualId);
      requireThat(upload, 'Editor scratch upload was not observed');
      const bound = structuredClone(expected); bound.referenceAssetIds = [actualId]; assertRequestBinding(bound, network.request);
      const file = 'source-upload.png'; await writeFile(resolve(captureRoot,file), upload.bytes, { flag: 'wx', mode: 0o600 });
      sourceUpload = { inputId: 'B', file, sha256: sha(upload.bytes), observedAt: upload.network.observedAt, network: upload.network };
    } else assertRequestBinding(expected, network.request);
    await waitFor(page, observation, () => {
      const jobs = observation.progress.filter(job => job.id === network.response.id);
      if (jobs.some(job => job.status === 'running')) return true;
      requireThat(!jobs.some(job => ['completed','failed','canceled','interrupted'].includes(job.status)), 'actual UI job ended without an observed running state');
      return false;
    }, 'UI never observed the actual running job', 30 * 60 * 1000);
    const completed = await waitFor(page,observation,() => completedUiJob(observation.progress,network.response.id,row.outputCount), 'actual UI job/result did not reach completed publication',60*60*1000);
    const resultUi = await waitResultUi(page,observation,route,row,completed,read,actions);
    shots.push(await screenshot(page, captureRoot, 'after'));
    await observation.flush();
    requireThat(observation.posts.length === 1, 'UI dispatched an extra unallocated job');
    capture = { version: VERSION, rowId, lockSha256: digest(state.lock), bindingSha256: digest(state.binding), observer: 'actual-playwright-react-ui/v1', capturedAt: now(),
      network, screenshots: shots, progress: observation.progress.filter(job => job.id === network.response.id), actions, resultUi,
      ...(sourceUpload ? { sourceUpload } : {}), ...(history ? { history } : {}), ...(workflow ? { workflow } : {}) };
    const path = resolve(captureRoot,'capture.json'); await save(path,capture); return path;
  } catch (error) {
    await screenshot(page,captureRoot,'failure').catch(() => {});
    await save(resolve(captureRoot,'failure.json'), { version: VERSION, rowId, lockSha256: digest(state.lock), bindingSha256: digest(state.binding), at: now(), error: error.message, actions, screenshots: shots,
      submissions: observation.submissions, progress: observation.progress, imageJobPosts: observation.posts }).catch(() => {});
    throw error;
  } finally { try { await context.close(); } finally { await browser.close(); } }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [launchPath,rowId,captureRoot,playwrightModule,...extra] = process.argv.slice(2);
  if (extra.length || !playwrightModule) throw Error('Usage: moss-edit-terminal-browser.mjs LAUNCH_JSON ROW_ID FRESH_CAPTURE_DIR ABSOLUTE_PLAYWRIGHT_MODULE');
  captureUi({launchPath,rowId,captureRoot,playwrightModule}).then(path => console.log(path)).catch(error => { console.error(error.message); process.exitCode = 1; });
}
