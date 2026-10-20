// sc-26243: pure, weights-free terminal matrix. NOT an execution or measurement receipt.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { stripJsoncComments } from './lib/jsonc.mjs';
export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const digestPattern = /^[a-f0-9]{64}$/;
export const shaPattern = /^[a-f0-9]{40}$/;
export const PHASES = ['prerequisites', 'ordinary', 'references-layout', 'upsampler', 'adaptation', 'native-tiers', 'limits', 'failures'];
export async function loadProductTargets(root = ROOT) {
  const manifest = JSON.parse(stripJsoncComments(await readFile(path.join(root, 'config/manifests/builtin.models.jsonc'), 'utf8')));
  const capabilities = Object.fromEntries(await Promise.all(['mlx', 'candle'].map(async backend => [backend, JSON.parse(await readFile(path.join(root, `config/engine-capabilities/runtime/capabilities.${backend}.json`), 'utf8'))])));
  const allModels = Array.isArray(manifest) ? manifest : manifest.models;
  const models = allModels.filter(m => ['flux3_klein', 'flux3_klein_base'].includes(m.id));
  assert.equal(models.length, 2);
  const upsampler = allModels.find(m => m.id === 'flux3_image_klein_pup'); assert(upsampler?.componentOnly);
  return { models, capabilities, upsampler };
}
function generators(value) {
  if (Array.isArray(value?.generator_capabilities)) return value.generator_capabilities;
  for (const child of Object.values(value ?? {})) if (child && typeof child === 'object' && !Array.isArray(child)) { const found = generators(child); if (found.length) return found; }
  return [];
}
export function targetRegistry({ models, capabilities }) {
  return ['mlx', 'candle'].flatMap(backend => models.map(model => {
    const platform = backend === 'mlx' ? 'macos' : 'windows';
    const engine = generators(capabilities[backend]).find(e => e.id === model.id);
    assert(engine, `missing shipped ${backend}/${model.id} capability`);
    const downloads = model.downloads.filter(d => !d.coRequisite && (!d.platforms || d.platforms.includes(platform)));
    for (const d of model.downloads) assert.deepEqual(d.files, [], 'whole pinned repository only');
    return { backend, modelId: model.id, checkpoint: model.id.endsWith('_base') ? 'base' : 'distilled', model,
      engine, downloads, tiers: ['bf16', ...engine.supported_quants], artifactKey: tier => `${model.id}:${backend}:${tier}` };
  }));
}
const placement = { version: 1, boxes: [{ id: 'teapot', desc: 'teal ceramic teapot', bbox: [0.125, 0.25, 0.75, 0.875], bbox_space: 'normalized' }] };
const edits = { version: 1, boxes: [
  { id: 'move', desc: 'move the teapot', from: 'ref_image_0', reference_id: 'slot-0', src_bbox: [0.125, 0.25, 0.75, 0.875], src_space: 'normalized', tgt_bbox: [0.25, 0.125, 0.875, 0.75], tgt_space: 'normalized' },
  { id: 'remove', desc: 'remove the cup', from: 'ref_image_0', reference_id: 'slot-0', src_bbox: [0.01, 0.02, 0.11, 0.12], src_space: 'normalized', tgt_bbox: null },
] };
export function terminalCatalog(sources) {
  const targets = targetRegistry(sources), cases = [];
  for (const t of targets) {
    const add = (scenario, phase, options = {}, tier = 'bf16', kind = 'image') => {
      const productionRow = t.downloads.find(d => d.variant === tier);
      const required = { phase, backend: t.backend, modelId: t.modelId, tier };
      cases.push({ schemaVersion: 1, caseId: `${t.backend}.${t.checkpoint}.${scenario}.${tier}`, ...required, kind,
        artifactKey: t.artifactKey(tier), repository: t.downloads[0].repo, resolvedRevision: t.downloads[0].revision,
        inputBindings: Array.from({ length: options.references ?? 0 }, (_, index) => ({ role: 'reference', index, binding: `reference-${index}`, instanceId: `slot-${index}` })), options,
        expectedObservable: kind === 'image' ? ['owned ordinary API job completed', 'progress observed', 'GET library + raw sidecar agree', 'served PNG hash/dimensions match retained PNG', 'saved settings + input hashes', 'human/oracle visual inspection required'] : [scenario],
        status: productionRow ? 'NOT_RUN' : 'BLOCKED', blocker: productionRow ? null : 'No production catalog row/trusted real conversion hashes. Endpoint must refuse this tier; do not turn this cell into unsupported or PASS.',
      });
    };
    add('install-identity', 'prerequisites', {}, 'bf16', 'installation');
    add('install-provision', 'prerequisites', { provisionModelId: t.modelId }, 'bf16', 'provision');
    if (t.checkpoint === 'distilled') {
      add('install-provision-pup', 'prerequisites', { provisionModelId: sources.upsampler.id }, 'bf16', 'provision');
      const pup = cases.at(-1), download = sources.upsampler.downloads[0]; assert.deepEqual(download.files, []);
      pup.repository = download.repo; pup.resolvedRevision = download.revision; pup.artifactKey = `${sources.upsampler.id}:${t.backend}:bf16`;
    }
    add('t2i', 'ordinary'); add('cancel', 'ordinary', {}, 'bf16', 'cancel');
    add('saved-settings-repro', 'ordinary', {}, 'bf16', 'replay');
    add('refs-1', 'references-layout', { references: 1 }); add('refs-10', 'references-layout', { references: 10 });
    add('layout-placement', 'references-layout', { layout: placement });
    add('box-move-remove', 'references-layout', { references: 1, layout: edits });
    for (const [pupMode, references, layout] of [['t2i', 0, null], ['i2i', 1, null], ['t2i_bbox', 0, placement], ['i2i_bbox', 1, edits]]) {
      add(`pup-${pupMode}`, 'upsampler', { enhancePrompt: true, pupMode, references, ...(layout ? { layout } : {}) });
    }
    for (const scale of [0, 0.75]) add(`lora-${scale === 0 ? 'zero' : 'on'}`, 'adaptation', { loraScale: scale });
    if (t.checkpoint === 'base') for (const trainingMode of ['captioned', 'edit']) add(`train-interrupt-resume-export-${trainingMode}`, 'adaptation', { trainingMode }, 'bf16', 'training');
    for (const tier of t.tiers.filter(q => q !== 'bf16')) add('native-tier', 'native-tiers', {}, tier);
    add('portrait', 'limits', { width: 768, height: 1024 }); add('landscape', 'limits', { width: 1024, height: 768 });
    add('max-resolution', 'limits', { width: 4096, height: 4096 });
    for (const invalid of ['bad-reference', 'reference-count', 'bad-box', 'checkpoint-compatibility']) add(invalid, 'failures', { invalid }, 'bf16', 'reject');
    for (const failure of ['install-missing', 'authentication-denied']) add(failure, 'failures', { failure }, 'bf16', 'installation-failure');
  }
  for (const c of cases) {
    const visual = ['image', 'training', 'replay'].includes(c.kind);
    c.semanticAcceptance = visual ? 'PENDING' : 'NOT_APPLICABLE';
    c.visualObligations = visual ? ['Inspect exact retained input/output hashes: nonblank coherent image, prompt fidelity, quality and artifacts; validity/status/hash difference alone is not semantic acceptance.'] : [];
    if (c.inputBindings.length) c.visualObligations.push('Compare all ordered retained reference views with output for requested source identity and reference-order semantics.');
    if (c.options.layout) c.visualObligations.push('Verify placement against authored boxes; source-to-target moves must change location and explicit removals must remove the named source object without unintended duplicates.');
    if (Object.hasOwn(c.options, 'loraScale')) c.visualObligations.push('Inspect scale-zero/on pair for intended adapter effect and acceptable quality; different pixels alone do not pass.');
    if (c.kind === 'training') c.visualObligations.push('Compare trained-adapter reload with retained dataset views/captions for intended learned effect and acceptable ordinary generation quality.');
    if (c.options.enhancePrompt) c.expectedObservable.push(`native PUP mode=${c.options.pupMode}, outcome=enhanced, original/effective prompts, no fallbackReason`, ...(c.options.layout ? ['exact locked annotation bytes and authored boxes retained'] : []));
    if (Object.hasOwn(c.options, 'loraScale')) c.expectedObservable.push(`provisioned adapter hash/rank/identity; API weight=${c.options.loraScale}`, 'scale-on image differs from matching scale-zero image');
    if (c.kind === 'training') c.expectedObservable.push('dryRun:false; real factors/optimizer/RNG checkpoint before completion', 'cancel; fresh PID/worker ID; compatible retry resume:true', 'checkpoint cursor and learned factors advance; exported adapter registered', 'fresh process ordinary generation reload + retained PNG');
    if (c.kind === 'replay') c.expectedObservable.push('reconstruct request from persisted recipe; fresh-process same-seed PNG hash agrees');
    if (c.kind === 'provision') c.expectedObservable.push('explicit per-repository user download/disk/account-gate approval', 'authenticated fixed pinned config response, no credential output', 'owned ordinary model_download job + matching schema-v2 job-bound receipt', 'whole repository inventory hashes + installed catalog identity');
    if (c.kind === 'installation-failure') c.expectedObservable.push('separate explicit empty-cache fault admission; no upstream requests/credentials; actionable rejection', 'injected failure is NOT evidence of successful runner authentication');
  }
  return { schemaVersion: 1, kind: 'flux3-installed-product-diagnostic-catalog', diagnosticOnly: true, status: 'NOT_RUN', phases: PHASES, cases,
    policy: { downloads: 'never automatic; explicit per-repository approved provision cells only', captures: false, resourceAnchorsAreImageProof: false, visualProof: 'human/oracle inspection; HTTP completion is insufficient', training: 'dryRun:false, real bounded job only' } };
}
// Small representative CODE default, not permission to execute. Full catalog is diagnostic only.
export function terminalPlan(sources) {
  const catalog = terminalCatalog(sources), cases = [], resources = [], trainingLifecycles = [];
  const pick = (backend, checkpoint, scenario, { tier = 'bf16', options = {}, name = scenario } = {}) => {
    const source = catalog.cases.find(c => c.caseId === `${backend}.${checkpoint}.${scenario}.bf16`); assert(source);
    const cell = structuredClone(source); cell.caseId = `${backend}.${checkpoint}.${name}.${tier}`; cell.tier = tier; cell.artifactKey = `${cell.modelId}:${backend}:${tier}`; cell.options = { ...cell.options, ...options };
    cell.inputBindings = Array.from({ length: cell.options.references ?? 0 }, (_, index) => ({ role: 'reference', index, binding: `reference-${index}`, instanceId: `slot-${index}` }));
    cases.push(cell); return cell;
  };
  for (const backend of ['mlx', 'candle']) {
    pick(backend, 'distilled', 'saved-settings-repro', { name: 'representative-repro' });
    pick(backend, 'distilled', 'pup-t2i', { tier: 'q4', name: 'representative-pup-text', options: { width: 768, height: 1024 } });
    pick(backend, 'distilled', 'pup-i2i', { name: 'representative-ref-1', options: { width: 1024, height: 768 } });
    pick(backend, 'distilled', 'pup-t2i_bbox', { tier: 'q8', name: 'representative-layout-max', options: { width: 4096, height: 4096 } });
    pick(backend, 'base', 'pup-i2i_bbox', { name: 'representative-move-remove-10', options: { references: 10 } });
    pick(backend, 'base', 'refs-1', { name: 'representative-base-ref-1', options: { width: 768, height: 1024 } });
    pick(backend, 'distilled', 'refs-10', { name: 'representative-distilled-ref-10', options: { width: 1024, height: 768 } });
    pick(backend, 'distilled', 'lora-zero'); pick(backend, 'distilled', 'lora-on');
    const train = pick(backend, 'base', `train-interrupt-resume-export-${backend === 'mlx' ? 'edit' : 'captioned'}`);
    trainingLifecycles.push({ lifecycleId: train.caseId, caseId: train.caseId, backend, steps: 4, saveEvery: 1, rank: 4, resolution: 512, startId: `${train.caseId}.training-start`, resumeId: `${train.caseId}.training-resume`, generationReloads: 1 });
    pick(backend, 'base', 'cancel');
    if (backend === 'candle') for (const [checkpoint, tier] of [['base', 'fp8r'], ['distilled', 'nvfp4']]) pick(backend, checkpoint, 't2i', { tier, name: 'representative-native' });
    // API-only checks never imply additional GPU generations. Actual probes pause only our worker.
    for (const scenario of ['bad-reference', 'reference-count', 'bad-box', 'checkpoint-compatibility', 'install-missing', 'authentication-denied']) pick(backend, 'base', scenario);
    for (const checkpoint of ['distilled', 'base']) pick(backend, checkpoint, 'install-provision');
    const pup = structuredClone(catalog.cases.find(c => c.caseId === `${backend}.distilled.install-provision-pup.bf16`)); cases.push(pup);
    const native = backend === 'mlx' ? [['distilled', 'bf16'], ['base', 'bf16'], ['distilled', 'q4'], ['distilled', 'q8']] : [['distilled', 'bf16'], ['base', 'bf16'], ['distilled', 'q4'], ['distilled', 'q8'], ['base', 'fp8r'], ['distilled', 'nvfp4']];
    for (const [checkpoint, tier] of native) {
      const base = catalog.cases.find(c => c.caseId === `${backend}.${checkpoint}.t2i.bf16`);
      resources.push({ caseId: `${backend}.${checkpoint}.resource-native.${tier}`, backend, modelId: base.modelId, tier, kind: 'resource_anchor', resourcePhase: 'ordinary', options: {}, artifactKey: `${base.modelId}:${backend}:${tier}`, nativeGenerationCalls: 1, warmups: 0, repeats: 0, status: 'BLOCKED', blocker: 'Actual native inner-Generator.generate single-use permit boundary must be verified by the authorized resource owner; a launcher flag is not execution proof.' });
    }
    resources.push({ caseId: `${backend}.base.resource-pup-edit-10.bf16`, backend, modelId: 'flux3_klein_base', tier: 'bf16', kind: 'resource_supplemental', resourcePhase: 'prompt-upsampling', options: { references: 10, enhancePrompt: true, pupMode: 'i2i_bbox', layout: structuredClone(edits) }, artifactKey: `flux3_klein_base:${backend}:bf16`, nativeGenerationCalls: 1, warmups: 0, repeats: 0, status: 'BLOCKED', blocker: 'Native single-call permit boundary unverified; supplemental PUP observations are not anchors or installed-product image proof.' });
  }
  const generationInvocations = [];
  for (const c of cases.filter(c => ['image', 'replay', 'cancel', 'training'].includes(c.kind))) for (let index = 0; index < (c.kind === 'replay' ? 2 : 1); index++) generationInvocations.push({ invocationId: `${c.caseId}.generation-${index + 1}`, caseId: c.caseId, backend: c.backend, kind: 'product_generation', purpose: c.kind === 'training' ? 'training-reload' : c.kind === 'replay' ? `repro-${index + 1}` : c.kind, modelId: c.modelId, tier: c.tier, options: structuredClone(c.options), nativeGenerationCalls: 1, warmups: 0, repeats: 0 });
  for (const c of resources) generationInvocations.push({ invocationId: `${c.caseId}.generation-1`, caseId: c.caseId, backend: c.backend, kind: c.kind, purpose: c.resourcePhase, modelId: c.modelId, tier: c.tier, options: structuredClone(c.options), nativeGenerationCalls: 1, warmups: 0, repeats: 0 });
  for (const invocation of generationInvocations) {
    const source = cases.find(c => c.caseId === invocation.caseId) ?? resources.find(c => c.caseId === invocation.caseId), bindings = Array.from({ length: source.options.references ?? 0 }, (_, index) => ({ role: 'reference', index, binding: `reference-${index}`, instanceId: `slot-${index}` }));
    const image = { ...source, inputBindings: bindings, options: source.kind === 'training' ? { loraScale: 0.75 } : source.options };
    invocation.requestTemplate = imageRequest(image, targetRegistry(sources).find(t => t.backend === source.backend && t.modelId === source.modelId), '$OWNED_PROJECT', Object.fromEntries(bindings.map(b => [b.binding, { assetId: `$${b.binding}` }])), { id: '$BOUND_ADAPTER', name: '$BOUND_ADAPTER' });
  }
  assert.equal(generationInvocations.filter(i => i.kind === 'product_generation').length, 26); assert.equal(resources.length, 12);
  return { schemaVersion: 1, kind: 'flux3-installed-product-representative-plan', status: 'NOT_RUN', phases: PHASES, cases, resourceCases: resources, trainingLifecycles, generationInvocations,
    budget: { globalHardCeiling: 48, productGenerations: 26, resourceGenerations: 12, totalGenerationInvocations: 38, byBackend: { mlx: 17, candle: 21 }, unallocatedNotAuthorized: 10, automaticWarmups: 0, automaticRepeats: 0, automaticRetries: 0, trainingLifecycles: 2, stepsPerLifecycle: 4 },
    uncovered: [{ requirement: 'Base q4/q8 production artifact identities', status: 'BLOCKED', reason: 'Base HF403 and absent genuine converted production pins remain blockers, even though these diagnostic cells receive no GPU permits.' }, { requirement: 'Semantic image quality/reference/layout/adaptation acceptance', status: 'PENDING', reason: 'Requires human/oracle evidence bound to retained input/output hashes.' }, { requirement: 'Native resource invocation enforcement', status: 'BLOCKED', reason: 'Authorized resource owner must verify exactly one inner Generator.generate per permit; hidden warmups/rungs/seeds must expand counts and reduce the plan before approval.' }],
    policy: { ...catalog.policy, fullInventory: 'catalog command only; never automatically executed', execution: 'exact expanded plan/hash and explicit user approval required; disjoint per-backend single-use permits, local atomic ledgers, coordinator aggregates same global plan; no allocation of spare ten' } };
}
export function imageRequest(cell, target, projectId, inputs, lora) {
  const options = cell.options;
  const body = { projectId, mode: options.references ? 'edit_image' : 'text_to_image', model: cell.modelId,
    prompt: 'A small teal ceramic teapot and a white cup on a pale yellow table, soft daylight, studio photograph', negativePrompt: '', seed: 26243,
    width: options.width ?? 1024, height: options.height ?? 1024, count: 1,
    advanced: { steps: target.model.defaults.steps, sampler: target.model.defaults.sampler, guidanceScale: target.model.defaults.guidanceScale ?? 0, enhancePrompt: options.enhancePrompt === true, quantTier: cell.tier }, loras: [] };
  if (options.references) {
    body.referenceAssetIds = cell.inputBindings.map(b => { assert(inputs[b.binding], `missing ${b.binding}`); return inputs[b.binding].assetId; });
    body.advanced.flux3_reference_ids = cell.inputBindings.map(b => b.instanceId);
  }
  if (options.layout) body.advanced.flux3_layout = structuredClone(options.layout);
  if (Object.hasOwn(options, 'loraScale')) { assert(lora?.id, 'real provisioned compatible LoRA required'); body.loras = [{ id: lora.id, name: lora.name ?? lora.id, weight: options.loraScale }]; }
  return body;
}
export function requestedDiskBudget(plan, cells, { present = {}, conversionOutputBytes = {}, freeBytes, buildBytes = 32 * 1024 ** 3, stateBytes = 16 * 1024 ** 3 }) {
  const repos = new Map();
  for (const c of cells) {
    const t = plan.find(t => t.backend === c.backend && t.modelId === c.modelId); assert(t);
    for (const d of t.model.downloads.filter(d => c.kind === 'provision' ? d.repo === c.repository && d.revision === c.resolvedRevision : !d.coRequisite || c.options.enhancePrompt)) repos.set(`${d.repo}@${d.revision}`, d.estimatedSizeBytes);
  }
  const repositoryBytes = [...repos].reduce((n, [key, bytes]) => { assert(Number.isSafeInteger(bytes) && bytes > 0); const existing = present[key] ?? 0; assert(Number.isSafeInteger(existing) && existing >= 0 && existing <= bytes, 'present repository bytes must be bounded verified inventory observations'); return n + bytes - existing; }, 0);
  const derived = new Map();
  for (const c of cells) { const row = plan.find(t => t.backend === c.backend && t.modelId === c.modelId).downloads.find(d => d.variant === c.tier && d.localDerivation); if (row) derived.set(c.artifactKey, Math.max(row.localDerivation.weightsBytes, row.footprint.diskSizeBytes - row.estimatedSizeBytes)); }
  let conversionBytes = 0;
  for (const [key, minimum] of derived) { assert(Number.isSafeInteger(conversionOutputBytes[key]) && conversionOutputBytes[key] >= minimum, `explicit conversion/output disk allowance required (at least ${minimum} bytes): ${key}`); conversionBytes += conversionOutputBytes[key]; }
  const requiredBytes = repositoryBytes + conversionBytes + buildBytes + stateBytes;
  assert(Number.isSafeInteger(freeBytes) && freeBytes >= requiredBytes, `disk admission needs ${requiredBytes} bytes; repository + conversion + build/state margins (NOT measurements)`);
  return { repositoryBytes, conversionBytes, buildBytes, stateBytes, requiredBytes, freeBytes };
}
