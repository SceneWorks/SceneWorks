import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync} from 'node:fs';
import {verifyReplayForm,waitForModelCatalog,waitForReplayForm,modelPicker} from './moss-edit-terminal-controls.mjs';

let chromium;
try {({chromium}=await import('playwright'));} catch(error) {if(error.code!=='ERR_MODULE_NOT_FOUND')throw error;}

const expected={model:'moss_image_sft',prompt:'Make the bottle deep red.',count:2,width:64,height:64,seed:173,referenceAssetIds:['asset_B'],advanced:{steps:2,guidanceScale:4,schedulerShift:5,mossImage:{training_shift:2.5,max_sequence_length:128},mossImageEdit:{reference_max_edge:32,use_kv_cache:true},mossImageAdvanced:{output:'latents'}}};
const input=(label,value)=>`<label>${label}<input value="${value}"></label>`;
const picker=(label,option)=>`<label>${label}<select><option value="${option}">${option}</option></select></label>`;
const html=`<label class="settings-field-model">Model<select><option value="moss_image_base">Base</option><option value="moss_image_sft">SFT</option></select></label>
${Object.entries({'Prompt':expected.prompt,'Variations':2,'Width override':64,'Height override':64,'Steps':2,'Guidance':4,'Schedule shift':5,'Training shift':2.5,'Context limit':128,'Reference maximum edge':32,'Seed':173}).map(([name,value])=>input(name,value)).join('')}
<div role="group" aria-label="Ordered MOSS references"><label>Image 1<select><option value="asset_B">B</option></select></label></div>
<label>Cache reference K/V<input type="checkbox" checked></label>
${[['Native output','latents'],['Positive embeddings',''],['Positive attention mask',''],['Negative embeddings',''],['Negative attention mask',''],['Explicit initial noise','']].map(([name,value])=>picker(name,value)).join('')}
<label>Native text prompt batch<input type="checkbox"></label><label>Per-prompt negative strings<input type="checkbox"></label><label>Upscale<input type="checkbox"></label>`;

test('real browser waits for complete recipe hydration after Prompt is already restored, but never selects a wrong Model',{skip:!chromium||!existsSync(chromium.executablePath())?'Playwright Chromium is unavailable on this host':false},async()=>{
  const browser=await chromium.launch({headless:true,args:['--disable-gpu']});
  try {
    const page=await browser.newPage(),actions=[];await page.setContent(html);
    assert.equal(await page.getByLabel('Prompt',{exact:true}).inputValue(),expected.prompt);
    await assert.rejects(()=>verifyReplayForm(page,expected,actions),/actual replay did not restore Model/);
    await page.evaluate(()=>setTimeout(()=>{document.querySelector('label.settings-field-model select').value='moss_image_sft';},250));
    await waitForReplayForm(page,expected,actions,2000);
    assert.equal(await modelPicker(page).inputValue(),expected.model);
    assert.equal(actions.filter(item=>item.action==='observe-complete-replay-hydration').length,1);
    await modelPicker(page).selectOption('moss_image_base');
    await assert.rejects(()=>waitForReplayForm(page,expected,actions,300),/Model expected moss_image_sft, observed moss_image_base/);
    assert.equal(await modelPicker(page).inputValue(),'moss_image_base');
  } finally {await browser.close();}
});

test('cold catalog gate must settle into the real model picker before recipe comparison',{skip:!chromium||!existsSync(chromium.executablePath())?'Playwright Chromium is unavailable on this host':false},async()=>{
  const browser=await chromium.launch({headless:true,args:['--disable-gpu']});
  try {
    const page=await browser.newPage(),actions=[];
    const gate='<section class="model-availability-gate"><div role="progressbar" aria-label="Initializing models" style="display:block;width:100px;height:10px"></div><p>Checking which models are installed and available on this machine.</p></section>';
    await page.setContent(gate);
    await page.evaluate(content=>setTimeout(()=>{document.body.innerHTML=content;},10250),html);
    await waitForModelCatalog(page,actions,15000);
    assert.ok(actions.at(-1).elapsedMs>=10000,'catalog wait did not cover the prior ten-second form timeout');
    assert.equal(actions.at(-1).action,'observe-model-catalog-ready');
    assert.equal(await modelPicker(page).inputValue(),'moss_image_base');
    await page.setContent(gate);
    await assert.rejects(()=>waitForModelCatalog(page,actions,250),/initializing=true; gate=.*Checking which models/);
    assert.equal(await page.getByRole('progressbar',{name:'Initializing models'}).isVisible(),true);
    await page.setContent('<section class="model-availability-gate"><h2>No supported model installed</h2></section>');
    await assert.rejects(()=>waitForModelCatalog(page,actions,3000),/model catalog settled without an available Image Studio model/);
  } finally {await browser.close();}
});
