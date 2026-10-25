// CPU-only inert protocol tests. Never import Playwright, open a port or launch a browser.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, writeFile, lstat, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { createFreshCaptureDirectory, captureUi, capturePackageInstallUi, openAsset, installerComponent, installedDownloadJobs, completedUiJob, describeUiRow, multipartFile, jobSnapshots, eventSnapshots } from './moss-edit-terminal-browser.mjs';
import { createPlan, LATENT_UPSCALE_GUARDS } from './lib/moss-edit-terminal.mjs';

const inputs = { provenance: 'inert browser route fixture', single_case: { references: ['B'], width: 768, height: 512, prompt: 'single' },
  four_case: { references: ['B','L','P','G'], width: 512, height: 768, prompt: 'four', mandatory_preservation: ['left','right'], typography_scope: 'excluded' } };

test('all fixed UI allocations retain their actual route, origin, selected sample and guards', () => {
  const plan = createPlan(inputs), ui = plan.rows.filter(row => row.transport.startsWith('ui-'));
  assert.equal(plan.rows.length, 50); assert.equal(plan.requiredGuardCases.length * 2, 58);
  assert.equal(ui.length, 10);
  assert.deepEqual(ui.map(row => row.id), ['quality-mlx-base-single','quality-mlx-sft-single','quality-cuda-base-four','quality-cuda-sft-single','advanced-mlx-sft-replay','advanced-cuda-sft-replay','seeded-mlx-images-replay','seeded-mlx-latents-replay','seeded-cuda-images-replay','seeded-cuda-latents-replay']);
  for (const row of ui) {
    const route = describeUiRow(row);
    assert.equal(route.shell, row.transport.slice(3));
    if (row.action === 'seeded-recipe') { assert.equal(route.history,true); assert.equal(route.originRowId,row.replayOf); assert.equal(route.sampleIndex,2); }
    if (row.action === 'workflow') { assert.equal(route.workflow,true); assert.equal(route.originRowId,row.replayOf); assert.equal(route.latentGuards,true); }
  }
  assert.deepEqual(describeUiRow(ui.find(row => row.id === 'quality-cuda-base-four')), {shell:'full',workflow:true,originRowId:'quality-mlx-base-four',sampleIndex:0,history:false,latentGuards:false});
  assert.deepEqual(describeUiRow(ui.find(row => row.id === 'quality-cuda-sft-single')), {shell:'editor',workflow:false,originRowId:'quality-mlx-sft-single',sampleIndex:0,history:true,latentGuards:false});
  assert.deepEqual(LATENT_UPSCALE_GUARDS,['latent-inline-upscale-real-esrgan','latent-inline-upscale-seedvr2']);
  assert.equal(plan.rows.filter(row => row.category === 'quality').length,8);
});

test('UI driver rejects REST/MCP or cancellation allocation rather than dispatching a fallback', () => {
  assert.throws(() => describeUiRow({transport:'rest'}), /allocated UI/);
  assert.throws(() => describeUiRow({transport:'mcp'}), /allocated UI/);
  assert.throws(() => describeUiRow({transport:'ui-full',cancelAtDenoise:true}), /not allocated/);
});

test('binary multipart bytes survive exact extraction including NUL and CRLF, with quoted boundary', () => {
  const source = Buffer.from([137,80,78,71,13,10,26,10,0,255,13,10,33]);
  const raw = Buffer.concat([Buffer.from('--unique\r\nContent-Disposition: form-data; name="projectId"\r\n\r\nproject\r\n--unique\r\nContent-Disposition: form-data; name="file"; filename="scratch.png"\r\nContent-Type: image/png\r\n\r\n'),source,Buffer.from('\r\n--unique--\r\n')]);
  assert.deepEqual(multipartFile(raw,'multipart/form-data; boundary="unique"'),source);
  assert.deepEqual(multipartFile(raw,'multipart/form-data; boundary=unique'),source);
  assert.throws(() => multipartFile(raw.subarray(0,raw.length-20),'multipart/form-data; boundary=unique'), /unterminated/);
  assert.throws(() => multipartFile(raw,'multipart/form-data; boundary=wrong'), /boundary/);
  assert.throws(() => multipartFile(raw,'application/json'), /multipart/);
});

test('a text field called file is never accepted as an observed raster upload', () => {
  const raw = Buffer.from('--b\r\nContent-Disposition: form-data; name="file"\r\n\r\nnot a file\r\n--b--\r\n');
  assert.throws(() => multipartFile(raw,'multipart/form-data; boundary=b'), /no file part/);
});

test('actual JSON and SSE job shapes preserve running IDs and event payloads', () => {
  const running = {id:'actual-job',status:'running',stage:'generating',progress:0.1,message:'step 1/4'};
  const completed = {id:'finished-job',status:'completed'};
  assert.deepEqual(jobSnapshots([running,completed]),[running,completed]);
  assert.deepEqual(jobSnapshots({jobs:[running]}),[running]);
  assert.deepEqual(jobSnapshots({data:{job:running}}),[running]);
  assert.deepEqual(eventSnapshots('job.updated',JSON.stringify(running)),[running]);
  assert.deepEqual(eventSnapshots('jobs.snapshot',JSON.stringify({jobs:[running],clearedJobIds:[]})),[running]);
  assert.deepEqual(jobSnapshots({payload:{id:'nested-not-a-job',status:'running'}}),[]);
  assert.throws(() => eventSnapshots('worker.updated',JSON.stringify(running)), /not a job/);
  assert.throws(() => eventSnapshots('job.updated','broken'), SyntaxError);
});

test('import is inert and capture rejects relative paths before any source or module work', async () => {
  await assert.rejects(captureUi({launchPath:'relative.json',rowId:'quality-mlx-base-single',captureRoot:'/tmp/no-created-output',playwrightModule:'/not-installed/playwright/index.mjs'}), /absolute/);
  await assert.rejects(capturePackageInstallUi({apiUrl:'http://127.0.0.1:1234',projectId:'inert',token:'inert',model:'moss_image_base',captureRoot:'relative',playwrightModule:'/not-installed/playwright/index.mjs'}), /absolute/);
});

function installerFixture() {
  const model = 'moss_image_base', component = {componentId:'moss-image-edit',repo:'chinoll/Moss-Image-Edit',revision:'a'.repeat(40),requiredFor:['moss_image_edit']};
  const entry = {id:model,installState:'installed',conditionalPurposes:{moss_image_edit:{installState:'missing',blocked:false}},conditionalComponents:[component]};
  const job = {id:'inert-download-job',type:'model_download',status:'queued',payload:{modelId:model,repo:component.repo,revision:component.revision}};
  const network = {method:'POST',path:`/api/v1/models/${model}/conditional-components/moss_image_edit/download`,status:201,request:{licenseAcknowledged:true},response:{purpose:'moss_image_edit',jobs:[job],components:[{...component,status:'queued',jobId:job.id}]}};
  const utilityWorkerId = 'owned-native-utility', owned = {workerId:utilityWorkerId,assignedGpu:'cpu',error:null};
  // R5's actual seven SSE observations: claim, size estimate, transfer, verification, completion.
  const progress = [
    {...job,stage:'queued',progress:0,workerId:null,assignedGpu:null,error:null},
    {...job,...owned,status:'preparing',stage:'preparing',progress:0},
    {...job,...owned,status:'downloading',stage:'downloading',progress:0.1},
    {...job,...owned,status:'downloading',stage:'downloading',progress:0.1},
    {...job,...owned,status:'downloading',stage:'downloading',progress:0.7957257856892123},
    {...job,...owned,status:'downloading',stage:'downloading',progress:0.95},
    {...job,...owned,status:'completed',stage:'completed',progress:1},
  ];
  return {model,component,entry,network,progress,utilityWorkerId};
}

test('installer selects the pinned missing component and rejects preinstalled, blocked or incomplete base state', () => {
  const fixture = installerFixture();
  assert.deepEqual(installerComponent([fixture.entry],fixture.model),{entry:fixture.entry,component:fixture.component});
  assert.deepEqual(installerComponent({models:[fixture.entry]},fixture.model),{entry:fixture.entry,component:fixture.component});
  for (const state of ['installed','blocked','unknown']) {
    const entry = structuredClone(fixture.entry); entry.conditionalPurposes.moss_image_edit.installState = state;
    assert.throws(() => installerComponent([entry],fixture.model), /genuinely missing/);
  }
  const baseMissing = structuredClone(fixture.entry); baseMissing.installState = 'missing';
  assert.throws(() => installerComponent([baseMissing],fixture.model), /complete base model/);
  const unpinned = structuredClone(fixture.entry); unpinned.conditionalComponents[0].revision = 'main';
  assert.throws(() => installerComponent([unpinned],fixture.model), /pinned/);
  const blocked = structuredClone(fixture.entry); blocked.conditionalPurposes.moss_image_edit.blocked = true;
  assert.throws(() => installerComponent([blocked],fixture.model), /genuinely missing/);
  assert.throws(() => installerComponent([fixture.entry],'moss_image_other'), /exact MOSS/);
});

test('installer requires acknowledged pinned native downloading-to-completed progress from its owned CPU worker', () => {
  const fixture = installerFixture();
  assert.deepEqual(installedDownloadJobs(fixture),['inert-download-job']);
  const mutations = [
    item => {item.network.request.licenseAcknowledged=false;},
    item => {item.network.response.jobs=[];},
    item => {item.network.response.components[0].status='installed';},
    item => {item.network.response.components[0].revision='b'.repeat(40);},
    item => {item.progress=item.progress.filter(job => job.status !== 'downloading');},
    item => {item.progress=[item.progress.at(-1)];},
    item => {item.progress=item.progress.slice(0,2);},
    item => {item.progress.pop();},
    item => {for (const job of item.progress.filter(job => job.status === 'downloading')) job.status=job.stage='running';},
    item => {item.progress[2].payload={...item.progress[2].payload,repo:'other/repo'};},
    item => {item.progress[2].payload={...item.progress[2].payload,modelId:'moss_image_sft'};},
    item => {item.progress[2].payload={...item.progress[2].payload,revision:'b'.repeat(40)};},
    item => {item.progress[2].type='image_generate';},
    item => {item.progress.at(-1).id='other-job';},
    item => {item.utilityWorkerId='';},
    item => {item.utilityWorkerId='foreign-worker';},
  ];
  for (const index of [2,-1]) for (const [key,value] of [['workerId','foreign-worker'],['workerId',null],['assignedGpu','mlx'],['stage','running'],['error','download failed']]) mutations.push(item => {item.progress.at(index)[key]=value;});
  for (const status of ['failed','canceled','interrupted']) {
    mutations.push(item => {item.progress.at(-1).status=status;});
    mutations.push(item => {item.progress.splice(2,0,{...item.progress[2],status});});
  }
  for (const mutate of mutations) { const item=structuredClone(fixture); mutate(item); assert.throws(() => installedDownloadJobs(item)); }
});

test('after-capture gate requires completed canonical publication and every allocated output', () => {
  const running={id:'ui-job',status:'running'}, pending={id:'ui-job',status:'completed',result:{outputPublication:{state:'pending'},assets:[{id:'a'},{id:'b'}]}}, complete={id:'ui-job',status:'completed',result:{assets:[{id:'a'},{id:'b'}]}};
  assert.equal(completedUiJob([running],'ui-job',2),null);
  assert.equal(completedUiJob([running,pending],'ui-job',2),null);
  assert.equal(completedUiJob([running,complete],'ui-job',4),null);
  assert.equal(completedUiJob([running,complete],'different-job',2),null);
  assert.equal(completedUiJob([running,pending,complete],'ui-job',2),complete);
  for(const status of ['failed','canceled','interrupted']) assert.throws(()=>completedUiJob([{id:'ui-job',status}],'ui-job',2),/did not complete/);
});

test('driver has no direct mutation transport or network interception fallback', async () => {
  const source = await readFile(new URL('./moss-edit-terminal-browser.mjs',import.meta.url),'utf8');
  assert.doesNotMatch(source,/\bfetch\s*\(|\.request\.(?:post|put|patch|delete)\s*\(|\.route\s*\(|\.fulfill\s*\(|\.addInitScript\s*\(/);
  assert.doesNotMatch(source,/reserveAttempt\s*\(/);
  assert.match(source,/await import\(pathToFileURL\(playwrightModule\)\.href\)/);
  assert.match(source,/Network\.eventSourceMessageReceived/);
  assert.match(source,/await verifyReplayForm\(page, expected, actions\)/);
});

// Inert locator protocol only: no browser, API, window or generated image is started.
function assetLibrary(assets) {
  const opened=[];let search='',type='';
  const control={async click(){},async fill(value){search=value;},async inputValue(){return search;},async selectOption(value){type=value;}};
  const page={getByRole(role){return role==='combobox'?{...control,async inputValue(){return type;}}:{...control,async waitFor(){}};},locator(selector){
    if(selector==='nav')return {getByRole(){return control;}};
    const match=/^\.asset-tile\[data-asset-id=(".*")\]$/.exec(selector);assert.ok(match,`selection must use exact asset identity: ${selector}`);
    const matches=assets.filter(asset=>asset.id===JSON.parse(match[1]));
    return {async waitFor(){},async count(){return matches.length;},async dblclick(){opened.push(matches[0]);}};
  }};return {page,opened};
}
test('replay opens the exact asset when retained outputs share the same display name',async()=>{
  const assets=['base-advanced','base-duplicate','sft-advanced'].map(id=>({id,displayName:'MOSS sampled latents 1'}));
  const library=assetLibrary(assets),actions=[];
  await openAsset(library.page,assets[2],actions);
  assert.deepEqual(library.opened,[assets[2]]);assert.equal(actions.at(-1).assetId,'sft-advanced');
});
test('replay refuses missing or ambiguous asset IDs even when their display names match',async()=>{
  const asset={id:'same-id',displayName:'MOSS sampled latents 1'};
  for(const assets of [[],[asset,{...asset}]]){
    const library=assetLibrary(assets);
    await assert.rejects(openAsset(library.page,asset,[]),/ambiguous/);assert.deepEqual(library.opened,[]);
  }
});

test('production capture directory creates fresh nested parents and refuses stale leaves before runtime',async t=>{
  const root=await mkdtemp(resolve(tmpdir(),'moss-fresh-capture-'));t.after(()=>rm(root,{recursive:true,force:true}));const capture=resolve(root,'fresh-session/captures/quality-mlx-base-single');
  await createFreshCaptureDirectory(capture);assert.equal((await lstat(capture)).isDirectory(),true);assert.deepEqual(await readdir(capture),[]);
  if(process.platform!=='win32'){assert.equal((await lstat(capture)).mode&0o777,0o700);assert.equal((await lstat(resolve(capture,'..'))).mode&0o777,0o700);}
  await writeFile(resolve(capture,'sentinel'),'retained');await assert.rejects(()=>createFreshCaptureDirectory(capture),{code:'EEXIST'});assert.equal(await readFile(resolve(capture,'sentinel'),'utf8'),'retained');await assert.rejects(()=>createFreshCaptureDirectory('relative/capture'),/absolute/);
});
test('both browser entrypoints use the fresh parent/leaf helper before importing browser runtime',async t=>{
  const source=await readFile(new URL('./moss-edit-terminal-browser.mjs',import.meta.url),'utf8');assert.equal((source.match(/await createFreshCaptureDirectory\(captureRoot\)/g)??[]).length,2);
  for(const name of ['capturePackageInstallUi','captureUi']){const body=source.slice(source.indexOf(`export async function ${name}(`));assert.ok(body.indexOf('await createFreshCaptureDirectory(captureRoot)')<body.indexOf('await import(pathToFileURL(playwrightModule).href)'));}
  const root=await mkdtemp(resolve(tmpdir(),'moss-capture-bootstrap-'));t.after(()=>rm(root,{recursive:true,force:true}));const captureRoot=resolve(root,'fresh/install/capture');
  await assert.rejects(()=>capturePackageInstallUi({apiUrl:'http://127.0.0.1:17957/',projectId:'inert-owned-project',captureRoot,playwrightModule:resolve(root,'runtime-must-not-load.mjs'),token:'inert-token',model:'moss_image_base',utilityWorkerId:'inert-utility'}),{code:'ERR_MODULE_NOT_FOUND'});assert.equal((await lstat(captureRoot)).isDirectory(),true);
});

test('production control extraction remains frozen and latent guards use the same tested labelled Engine control',async()=>{
  const driver=await readFile(new URL('./moss-edit-terminal-browser.mjs',import.meta.url),'utf8'),controller=await readFile(new URL('./moss-edit-terminal-actions.mjs',import.meta.url),'utf8'),controls=await readFile(new URL('./lib/moss-edit-terminal-controls.mjs',import.meta.url),'utf8');
  assert.match(driver,/import \{[^\n]*configure, verifyReplayForm \} from '\.\/lib\/moss-edit-terminal-controls.mjs'/);
  assert.match(controller,/browserControls:'lib\/moss-edit-terminal-controls.mjs'/);
  assert.match(driver,/await select\(labelledSelect\(page,'Engine'\), engine/);
  assert.doesNotMatch(controls,/\bfetch\s*\(|\.request\.|\.route\s*\(|node:|process\./);
});

test('submission polling acknowledges only the real positive recipe choice before awaiting the actual POST',async()=>{
  const source=await readFile(new URL('./moss-edit-terminal-browser.mjs',import.meta.url),'utf8');
  assert.match(source,/await acknowledgeRecipeChoice\(page,actions\);return observation.submissions\[0\]/);
});
