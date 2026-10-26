// Pure DOM interactions shared by the terminal browser and rendered-UI tests.
const now=()=>new Date().toISOString();
const requireThat=(condition,message)=>{if(!condition)throw Error(message);};
export async function click(locator, actions, label) { await locator.click(); actions.push({ at: now(), action: 'click', control: label }); }
export async function fill(locator, value, actions, label) { await locator.fill(String(value)); requireThat(await locator.inputValue() === String(value), `UI did not retain ${label}`); actions.push({ at: now(), action: 'fill', control: label, value }); }
export async function select(locator, value, actions, label) { await locator.selectOption(String(value)); requireThat(await locator.inputValue() === String(value), `UI did not select ${label}`); actions.push({ at: now(), action: 'select', control: label, value }); }
export async function checked(locator, value, actions, label) { await locator.setChecked(value); requireThat(await locator.isChecked() === value, `UI did not retain ${label}`); actions.push({ at: now(), action: 'check', control: label, value }); }
const escapeRegExp = value => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
// Wrapped selects include option text, and tensor pickers append measured SHA256.
// Match the source-owned label prefix and its control without losing identity.
export const labelledSelect = (scope, label) => scope.locator('label').filter({ hasText: new RegExp(`^${escapeRegExp(label)}`) }).locator('select');
const tensorPicker=labelledSelect;
export const modelPicker = page => page.locator('label.settings-field-model select');
// KeepAlive retains Studio forms while Image Editor is active. Never resolve its
// shared control labels against the whole page or choose an arbitrary first match.
export const controlScope = (page,kind) => kind==='editor' ? page.locator('.image-editor-surface:visible .ie-insp-body:visible') : page;
export async function waitForModelCatalog(page,actions,timeoutMs=120000) {
  const started=Date.now(),deadline=started+timeoutMs,responses=[];
  const onResponse=response=>{try {if(new URL(response.url()).pathname==='/api/v1/models')responses.push({status:response.status(),at:now()});}catch{}};
  page.on('response',onResponse);
  try {
    let terminalSince=null;
    while(Date.now()<deadline){
      // React can replace the gate with the form between two Playwright reads.
      // Observe the model first, then the gate; an overlapping transition is
      // retried instead of treating a stale gate observation as a failure.
      const modelVisible=await modelPicker(page).isVisible();
      const initializing=await page.getByRole('progressbar',{name:'Initializing models'}).isVisible();
      if(modelVisible&&!initializing){
        actions.push({at:now(),action:'observe-model-catalog-ready',elapsedMs:Date.now()-started,modelResponses:responses});
        return;
      }
      const terminalGate=await page.locator('.model-availability-gate').isVisible()&&!initializing;
      terminalSince=terminalGate?(terminalSince??Date.now()):null;
      if(terminalSince!==null&&Date.now()-terminalSince>=1000)throw Error('model catalog settled without an available Image Studio model');
      await page.waitForTimeout(Math.min(250,deadline-Date.now()));
    }
    throw Error('model catalog remained unsettled before the deadline');
  } catch(error) {
    const gate=page.getByRole('progressbar',{name:'Initializing models'});
    const initializing=await gate.isVisible().catch(()=>false);
    const gateText=initializing?await page.locator('.model-availability-gate').first().innerText().catch(()=>null):null;
    throw Error(`image model catalog/form unavailable after ${Date.now()-started} ms; initializing=${initializing}; gate=${JSON.stringify(gateText)}; modelResponses=${JSON.stringify(responses)}; ${error.message}`);
  } finally {page.off('response',onResponse);}
}
export async function advanced(page, actions) {
  const simple = page.locator('details.su-advanced');
  if (await simple.count()) { if (!(await simple.getAttribute('open') !== null)) await click(simple.locator('summary'), actions, 'Advanced disclosure'); return; }
  const button = page.locator('.advanced-section').getByRole('button', { name: /^Advanced/ });
  if (await button.count()) { if (await button.getAttribute('aria-expanded') !== 'true') await click(button, actions, 'Advanced disclosure'); return; }
  const fallback = page.getByRole('button', { name: /^Advanced/ });
  if (await fallback.count() === 1 && await fallback.getAttribute('aria-expanded') !== 'true') await click(fallback, actions, 'Advanced disclosure');
  await page.getByLabel('Steps', { exact: true }).waitFor({ state: 'visible' });
}

// Wrapped select labels include option text in their accessible label. Scope by the
// source-owned visible label prefix, then target its select; preserve repeated slots.
export async function configureReferences(page,ids,actions) {
  requireThat(Array.isArray(ids)&&ids.length>=1&&ids.length<=4,'one through four ordered references required');
  const picks=page.getByRole('group',{name:/^Ordered MOSS references/});await picks.waitFor({state:'visible'});
  const old=await picks.locator('select').count();
  for(let index=old;index>ids.length;index--)await click(picks.getByRole('button',{name:`Remove image ${index}`,exact:true}),actions,`Remove image ${index}`);
  for(let index=Math.min(old,ids.length);index<ids.length;index++)await click(picks.getByRole('button',{name:'Add reference image',exact:true}),actions,'Add reference image');
  for(const [index,id]of ids.entries())await select(labelledSelect(picks,`Image ${index+1}`),id,actions,`Image ${index+1}`);
}

// Verify restoration before changing any controls. A driver must not repair a broken replay.
export async function verifyReplayForm(page, expected, actions) {
  await advanced(page, actions);
  const actual = {}, scalars = { 'Prompt': expected.prompt, 'Variations': expected.count, 'Width override': expected.width, 'Height override': expected.height,
    'Steps': expected.advanced.steps, 'Guidance': expected.advanced.guidanceScale, 'Schedule shift': expected.advanced.schedulerShift,
    'Training shift': expected.advanced.mossImage.training_shift ?? 1, 'Context limit': expected.advanced.mossImage.max_sequence_length ?? 4096,
    'Reference maximum edge': expected.advanced.mossImageEdit.reference_max_edge, 'Model': expected.model };
  for (const [label,value] of Object.entries(scalars)) {
    actual[label] = await (label === 'Model' ? modelPicker(page) : page.getByLabel(label, { exact: true })).inputValue();
    requireThat(actual[label] === String(value), `actual replay did not restore ${label}`);
  }
  const group = page.getByRole('group', { name: /^Ordered MOSS references/ });
  requireThat(await group.locator('select').count() === expected.referenceAssetIds.length, 'replay changed reference slot count');
  for (const [index,id] of expected.referenceAssetIds.entries()) requireThat(await labelledSelect(group,`Image ${index + 1}`).inputValue() === id, 'replay changed original reference order');
  requireThat(await page.getByLabel('Cache reference K/V', { exact: true }).isChecked() === expected.advanced.mossImageEdit.use_kv_cache, 'replay changed K/V cache setting');
  const native = expected.advanced.mossImageAdvanced ?? { output: 'images' };
  for (const [label,value] of [['Native output',native.output],['Positive embeddings',native.positive?.values?.assetId ?? ''],['Positive attention mask',native.positive?.mask?.assetId ?? ''],['Negative embeddings',native.negative?.values?.assetId ?? ''],['Negative attention mask',native.negative?.mask?.assetId ?? ''],['Explicit initial noise',native.noise?.assetId ?? '']]) {
    requireThat(await (labelledSelect(page,label)).inputValue() === value, `replay changed ${label}`);
  }
  if (native.noise) requireThat(await page.getByLabel('Seed', { exact: true }).isDisabled(), 'supplied-noise replay left seed editable');
  else requireThat(await page.getByLabel('Seed', { exact: true }).inputValue() === String(expected.seed), 'replay changed original base seed');
  requireThat(await page.getByLabel('Native text prompt batch', { exact: true }).isChecked() === Boolean(expected.prompts), 'replay changed conditioning batch mode');
  if (expected.prompts) requireThat(await page.getByLabel('One positive prompt per conditioning row', { exact: true }).inputValue() === expected.prompts.join('\n'), 'replay changed conditioning row order');
  requireThat(await page.getByLabel('Per-prompt negative strings', { exact: true }).isChecked() === Boolean(native.negativePrompts), 'replay changed negative conditioning source');
  if (native.negativePrompts) requireThat(await page.getByLabel('One negative string per conditioning row', { exact: true }).inputValue() === native.negativePrompts.join('\n'), 'replay changed per-row negative strings');
  requireThat(!await page.getByLabel('Upscale', { exact: true }).isChecked(), 'replay unexpectedly enabled inline upscale');
  actions.push({ at: now(), action: 'verify-restored-form', values: actual, referenceAssetIds: expected.referenceAssetIds, nativeOutput: native.output });
}

export async function waitForReplayForm(page, expected, actions, timeoutMs=10000) {
  const started=Date.now(),deadline=started+timeoutMs;
  let lastError;
  try {
    do {
      page.setDefaultTimeout(Math.max(1,Math.min(1000,deadline-Date.now())));
      try {
        await verifyReplayForm(page,expected,actions);
        actions.push({at:now(),action:'observe-complete-replay-hydration',expectedModel:expected.model,elapsedMs:Date.now()-started});
        return;
      } catch(error) { lastError=error; }
      if(Date.now()>=deadline)break;
      await page.waitForTimeout(Math.min(100,deadline-Date.now()));
    } while(true);
    const model=await modelPicker(page).inputValue({timeout:1000}).catch(error=>`unavailable: ${error.message}`);
    throw Error(`actual replay form did not stabilize in ${Date.now()-started} ms: ${lastError?.message}; Model expected ${expected.model}, observed ${model}`);
  } finally {page.setDefaultTimeout(30000);}
}

export async function configure(page, expected, kind, actions) {
  const form=controlScope(page,kind);
  if (kind === 'editor') {
    await form.waitFor({state:'visible'});
    await select(form.locator('.ie-section').filter({has:page.getByText('Model',{exact:true})}).locator(':scope > select.ie-select'), expected.model, actions, 'Editor model');
    await fill(form.locator('.ie-field').filter({has:page.getByText('Instruction',{exact:true})}).locator('textarea.ie-textarea'), expected.prompt, actions, 'Editor instruction');
  } else {
    if (kind === 'simple') {
      const field = page.locator('.su-settings-bar .su-field').filter({ has: page.getByText('Model', { exact: true }) });
      await click(field.locator('button.su-select'), actions, 'Simple model');
      await page.locator('.su-sheet').waitFor({ state: 'visible' });
      // Select the actual catalog label carrying this exact ID, read from the UI's catalog.
      const models = await page.locator('.su-sheet').getByRole('button').allTextContents();
      const variant = expected.model.endsWith('_base') ? /MOSS.*Base/i : /MOSS.*SFT/i;
      const names = models.map(name => name.trim()).filter(name => variant.test(name));
      requireThat(names.length === 1, 'Simple model picker did not offer one exact MOSS variant');
      await click(page.locator('.su-sheet').getByRole('button', { name: names[0], exact: true }), actions, names[0]);
    } else await select(modelPicker(page), expected.model, actions, 'Model');
    await fill(page.getByLabel('Prompt', { exact: true }), expected.prompt, actions, 'Prompt');
    await advanced(page, actions);
    await configureReferences(page,expected.referenceAssetIds,actions);
  }
  const controls = { 'Reference maximum edge': expected.advanced.mossImageEdit.reference_max_edge, 'Steps': expected.advanced.steps,
    'Guidance': expected.advanced.guidanceScale, 'Schedule shift': expected.advanced.schedulerShift, 'Training shift': expected.advanced.mossImage.training_shift ?? 1,
    'Context limit': expected.advanced.mossImage.max_sequence_length ?? 4096,
    [kind === 'editor' ? 'Output width' : kind === 'simple' ? 'Custom width' : 'Width override']: expected.width,
    [kind === 'editor' ? 'Output height' : kind === 'simple' ? 'Custom height' : 'Height override']: expected.height };
  for (const [label,value] of Object.entries(controls)) await fill(form.getByLabel(label, { exact: true }), value, actions, label);
  await checked(form.getByLabel('Cache reference K/V', { exact: true }), expected.advanced.mossImageEdit.use_kv_cache, actions, 'Cache reference K/V');
  const native = expected.advanced.mossImageAdvanced ?? { output: 'images' };
  await select(labelledSelect(form,'Native output'), native.output, actions, 'Native output');
  for (const [label,ref] of [['Positive embeddings',native.positive?.values],['Positive attention mask',native.positive?.mask],['Negative embeddings',native.negative?.values],['Negative attention mask',native.negative?.mask],['Explicit initial noise',native.noise]]) {
    const picker = tensorPicker(form,label);
    if (ref || await picker.isEnabled()) await select(picker, ref?.assetId ?? '', actions, label);
    else requireThat(await picker.inputValue() === '', `disabled tensor role ${label} retained a stale selection`);
  }
  await checked(form.getByLabel('Per-prompt negative strings', { exact: true }), Boolean(native.negativePrompts), actions, 'Per-prompt negative strings');
  if (native.negativePrompts) await fill(form.getByLabel('One negative string per conditioning row', { exact: true }), native.negativePrompts.join('\n'), actions, 'One negative string per conditioning row');
  if (kind !== 'editor') {
    const batch = page.getByLabel('Native text prompt batch', { exact: true });
    if (await batch.isEnabled()) await checked(batch, Boolean(expected.prompts), actions, 'Native text prompt batch');
    if (expected.prompts) await fill(page.getByLabel('One positive prompt per conditioning row', { exact: true }), expected.prompts.join('\n'), actions, 'One positive prompt per conditioning row');
    if (kind === 'full') await fill(page.getByLabel('Variations', { exact: true }), expected.count, actions, 'Variations');
  }
  if (!native.noise) {
    const seed = kind === 'editor' ? form.locator('.ie-field').filter({ has: page.getByText('Seed', { exact: true }) }).locator('input') : page.getByLabel('Seed', { exact: true });
    await fill(seed, expected.seed ?? '', actions, 'Seed');
  }
  const negative = form.getByLabel('Negative prompt', { exact: true });
  if (await negative.count() && await negative.isEnabled()) await fill(negative, expected.negativePrompt ?? '', actions, 'Negative prompt');
  if (kind === 'full') await checked(page.getByLabel('Upscale', { exact: true }), false, actions, 'Upscale');
  const adapter = form.getByText(/^Edit adapter: installed\./);
  await adapter.waitFor({ state: 'visible' }); actions.push({ at: now(), action: 'observe-installed-adapter', text: await adapter.innerText(), model: expected.model });
}

// This frozen campaign requires embedded portable recipes. Acknowledge only the
// normal first-use product choice, through its actual positive UI action.
export async function acknowledgeRecipeChoice(page,actions) {
  const dialog=page.getByRole('dialog',{name:'Include the recipe in generated images?',exact:true});
  if(!await dialog.count())return false;
  requireThat(await dialog.count()===1,'ambiguous first-use recipe choice');
  const error=dialog.getByRole('alert');
  requireThat(!await error.count(),`recipe choice persistence failed${await error.count()?': '+await error.innerText():''}`);
  const accept=dialog.getByRole('button',{name:'Ok, got it',exact:true});
  if(!await accept.count()||!await accept.isEnabled())return false;
  await click(accept,actions,'Include recipe in generated images: Ok, got it');return true;
}
