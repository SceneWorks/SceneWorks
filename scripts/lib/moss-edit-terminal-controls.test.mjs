import test from 'node:test';
import assert from 'node:assert/strict';
import {existsSync} from 'node:fs';
import {verifyReplayForm,waitForModelCatalog,waitForReplayForm,modelPicker,configure,controlScope,click} from './moss-edit-terminal-controls.mjs';

let chromium;
try {({chromium}=await import('playwright'));} catch(error) {if(error.code!=='ERR_MODULE_NOT_FOUND')throw error;}

const expected={model:'moss_image_sft',prompt:'Make the bottle deep red.',count:2,width:64,height:64,seed:173,referenceAssetIds:['asset_B'],advanced:{steps:2,guidanceScale:4,schedulerShift:5,mossImage:{training_shift:2.5,max_sequence_length:128},mossImageEdit:{reference_max_edge:32,use_kv_cache:true},mossImageAdvanced:{output:'latents'}}};
const input=(label,value)=>`<label>${label}<input value="${value}"></label>`;
const picker=(label,option)=>`<label>${label}<select><option value="${option}">${option}</option></select></label>`;
const html=`<label class="settings-field-model">Model<select><option value="moss_image_base">Base</option><option value="moss_image_sft">SFT</option></select></label>
${Object.entries({'Prompt':expected.prompt,'Variations':2,'Width override':64,'Height override':64,'Steps':2,'Guidance':4,'Schedule shift':5,'Training shift':2.5,'Context limit':128,'Reference maximum edge':32,'Seed':173}).map(([name,value])=>input(name,value)).join('')}
<div role="group" aria-label="Ordered MOSS references"><div class="asset-picker-field"><div class="asset-preview-chip"><small title="asset_B">B</small></div></div></div>
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

test('catalog transition between locator reads accepts the mounted Image Studio form',{skip:!chromium||!existsSync(chromium.executablePath())?'Playwright Chromium is unavailable on this host':false},async()=>{
  const browser=await chromium.launch({headless:true,args:['--disable-gpu']});
  try {
    const page=await browser.newPage(),actions=[];
    await page.setContent('<section class="model-availability-gate"><div role="progressbar" aria-label="Initializing models">Loading</div></section>');
    let switched=false;
    const racingPage={
      locator:(...args)=>page.locator(...args),
      on:(...args)=>page.on(...args),off:(...args)=>page.off(...args),
      waitForTimeout:(...args)=>page.waitForTimeout(...args),
      getByRole:(role,options)=>{
        const locator=page.getByRole(role,options);
        if(role!=='progressbar')return locator;
        return {isVisible:async()=>{
          const visible=await locator.isVisible();
          if(visible&&!switched){switched=true;await page.setContent(html);}
          return visible;
        }};
      },
    };
    await waitForModelCatalog(racingPage,actions,1000);
    assert.equal(switched,true);
    assert.equal(await modelPicker(page).inputValue(),'moss_image_base');
    assert.equal(actions.at(-1).action,'observe-model-catalog-ready');
  } finally {await browser.close();}
});

test('Editor configuration and Generate stay in the active inspector despite retained Studio duplicates',{skip:!chromium||!existsSync(chromium.executablePath())?'Playwright Chromium is unavailable on this host':false},async()=>{
 const browser=await chromium.launch({headless:true,args:['--disable-gpu']});
 try{
  const page=await browser.newPage();page.setDefaultTimeout(1500);
  const scalarNames=['Reference maximum edge','Steps','Guidance','Schedule shift','Training shift','Context limit','Output width','Output height','Negative prompt'];
  const tensorNames=['Positive embeddings','Positive attention mask','Negative embeddings','Negative attention mask','Explicit initial noise'];
  for(const supplied of [false,true]){
   const native=supplied?{output:'latents',positive:{values:{assetId:'asset_pos'},mask:{assetId:'asset_pmask'}},negative:{values:{assetId:'asset_neg'},mask:{assetId:'asset_nmask'}},noise:{assetId:'asset_noise'},negativePrompts:['first negative','second negative']}:{output:'images'};
   const request={...structuredClone(expected),count:1,width:768,height:512,negativePrompt:'preserve original',advanced:{...structuredClone(expected.advanced),mossImageAdvanced:native}};
   const form=id=>`<form id="${id}" onsubmit="event.preventDefault();window.submissions.push({form:this.id,values:Object.fromEntries(new FormData(this))})">
    <div class="ie-section"><div class="ie-sec-title">Model</div><select class="ie-select" name="Model"><option value="moss_image_base">Base</option><option value="moss_image_sft">SFT</option></select>
    <div class="ie-field"><span>Instruction</span><textarea class="ie-textarea" name="Instruction">old instruction</textarea></div></div>
    ${scalarNames.map(name=>`<label>${name}<input name="${name}" value="9"></label>`).join('')}
    <label>Cache reference K/V<input name="Cache reference K/V" type="checkbox"></label>
    <label>Native output<select name="Native output"><option value="images">images</option><option value="latents">latents</option></select></label>
    ${tensorNames.map(name=>`<label>${name}<select name="${name}">${['','asset_pos','asset_pmask','asset_neg','asset_nmask','asset_noise'].map(v=>`<option value="${v}">${v}</option>`).join('')}</select></label>`).join('')}
    <label>Per-prompt negative strings<input name="Per-prompt negative strings" type="checkbox"></label><label>One negative string per conditioning row<textarea name="One negative string per conditioning row"></textarea></label>
    <div class="ie-field"><span>Seed</span><input name="Seed" value="9" ${supplied?'disabled':''}></div>
    <p>Edit adapter: installed.</p><button type="submit">Generate edit</button></form>`;
   await page.setContent(`<script>window.submissions=[]</script><div id="retained-studio">${form('studio')}</div><section class="image-editor-surface"><div class="ie-insp-body">${form('editor')}</div></section>`);
   const snapshot=()=>page.locator('#studio [name]').evaluateAll(elements=>elements.map(x=>({name:x.name,value:x.value,checked:x.checked,disabled:x.disabled})));
   const before=await snapshot();assert.equal(await page.getByLabel('Reference maximum edge',{exact:true}).count(),2);
   await assert.rejects(()=>page.getByLabel('Reference maximum edge',{exact:true}).fill('32'),/strict mode violation/);
   const actions=[];await configure(page,request,'editor',actions);const editor=controlScope(page,'editor');
   assert.equal(await editor.getByLabel('Reference maximum edge',{exact:true}).inputValue(),'32');assert.equal(await editor.getByLabel('Output width',{exact:true}).inputValue(),'768');
   for(const [name,value] of [['Steps',2],['Guidance',4],['Schedule shift',5],['Training shift',2.5],['Context limit',128],['Output height',512],['Negative prompt','preserve original']])assert.equal(await editor.getByLabel(name,{exact:true}).inputValue(),String(value));
   assert.equal(await editor.getByLabel('Cache reference K/V',{exact:true}).isChecked(),true);
   assert.equal(await editor.locator('[name="Native output"]').inputValue(),native.output);
   for(const [name,value] of tensorNames.map((name,i)=>[name,supplied?['asset_pos','asset_pmask','asset_neg','asset_nmask','asset_noise'][i]:'']))assert.equal(await editor.locator(`[name="${name}"]`).inputValue(),value);
   assert.equal(await editor.locator('[name="Seed"]').inputValue(),supplied?'9':'173');assert.equal(await editor.getByLabel('Per-prompt negative strings',{exact:true}).isChecked(),supplied);
   if(supplied)assert.equal(await editor.getByLabel('One negative string per conditioning row',{exact:true}).inputValue(),native.negativePrompts.join('\n'));
   await click(editor.getByRole('button',{name:'Generate edit',exact:true}),actions,'Generate edit');const submissions=await page.evaluate(()=>window.submissions);
   assert.equal(submissions.length,1);assert.equal(submissions[0].form,'editor');assert.equal(submissions[0].values.Model,'moss_image_sft');assert.equal(submissions[0].values.Instruction,request.prompt);assert.equal(submissions[0].values['Output width'],'768');
   assert.deepEqual(await snapshot(),before);assert.equal(controlScope(page,'full'),page);assert.equal(controlScope(page,'simple'),page);
  }
  const driver=await (await import('node:fs/promises')).readFile(new URL('../moss-edit-terminal-browser.mjs',import.meta.url),'utf8');assert.match(driver,/controlScope\(page,route\.shell\)\.getByRole\('button', \{ name: generateName/);
 }finally{await browser.close();}
});
