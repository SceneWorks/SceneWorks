// sc-26691 reduced operational inventory: pure manifest DATA + validation + coverage accounting.
// This is part of the existing terminal harness, not a new framework. It never executes anything,
// imports no runtime/case module and invents no paths, SHAs, metrics or run IDs.
import assert from 'node:assert/strict';
//
// The parent's 84-cell regression catalog (42/backend) stays untouched: callers pass it in, or the
// helper reads this source-only ID list. Every catalog cell must map to exactly one category:
// real-weight, executed-proof or unproven.

export const BACKENDS = Object.freeze(['mlx', 'cuda']);

// 42 catalog cells per backend, copied from the parent requiredCases() at the ff base.
export const CATALOG_SUFFIXES = Object.freeze([
  'artifact.0', 'artifact.1', 'artifact.2', 'artifact.3',
  'numeric.tensor-mapping', 'numeric.convrot', 'numeric.pruned-interpolation', 'numeric.w4a8',
  'capability.text', 'capability.image', 'capability.first-last-frame', 'capability.last-frame-only',
  'capability.video', 'capability.ordered-multimodal', 'capability.audio-reference', 'capability.hybrid',
  'capability.turbo', 'capability.lora', 'capability.lokr',
  'generalization.renamed', 'generalization.independent-finetune', 'generalization.native-single',
  'generalization.sharded-diffusers', 'generalization.dense-f16', 'generalization.dense-bf16',
  'generalization.dense-f32', 'generalization.existing-families', 'generalization.ltx-add-select-generate',
  'generalization.compatible-base',
  'lifecycle.managed-linked-equivalence', 'lifecycle.restart-replay', 'lifecycle.source-drift',
  'lifecycle.interruption-retry', 'lifecycle.removal', 'lifecycle.cancel',
  'lifecycle.generation-source-prep-cancel', 'lifecycle.relink-rescan', 'lifecycle.leases',
  'lifecycle.duplicate', 'lifecycle.bounded-loading', 'lifecycle.cold-warm-staged',
  'lifecycle.provenance-license',
]);
export const CATALOG_CELLS_PER_BACKEND = CATALOG_SUFFIXES.length; // 42
export const CATALOG_CELLS = CATALOG_CELLS_PER_BACKEND * BACKENDS.length; // 84

export function defaultCatalog() {
  return BACKENDS.flatMap(backend => CATALOG_SUFFIXES.map(suffix => ({ id: `${backend}.${suffix}`, backend, group: suffix.split('.')[0] })));
}
export function catalogIds(backend) {
  if (backend === undefined) return defaultCatalog().map(cell => cell.id);
  if (!BACKENDS.includes(backend)) throw new Error(`unknown backend ${backend}`);
  return CATALOG_SUFFIXES.map(suffix => `${backend}.${suffix}`);
}

// The seven broad matrix cases may run only on the measured-cheaper backend once cheap causal
// witnesses have passed on the displaced backend. Case 8 (drop-audio) is the one actual mixed
// causal check that runs on both.
export const BROAD_MATRIX_CASES = Object.freeze([
  'real-drop-images', 'real-drop-video', 'real-reverse-images', 'real-reverse-video',
  'real-reverse-audio', 'real-hybrid-prompt-change', 'real-hybrid-text-repeat',
]);
export const BOTH_BACKEND_CAUSAL_CASE = 'real-drop-audio';

const caseDef = (index, slug, kind, group, details = {}) => ({ index, slug, kind, group, ...details });

// sc-26691 user-approved scope decision, 2026-10-30. The cases with no trained accelerator run the
// compatible-base unaccelerated recipe: 50 denoise steps on CUDA, but 8 on MLX. At the tiled MLX
// kernel one H3 step measured ~5.4-6.6 min (run 41061562586), so a 50-step MLX render (~5 h) fits
// neither the 7200 s case bound nor the 24 h lane, while 8 steps (~50 min) does. Only the step
// count differs: inputs, geometry, adapters, seed and scheduler shift are identical on both
// backends. The per-backend count is DECLARED on the case (`backendSteps`), bound into the plan's
// sampling and checked by every validator; it is never a silent runtime override. Because the two
// backends then render different recipes, no validation may treat an MLX and a CUDA output of these
// cases as equivalent: they are neither matrix anchors, baselines/repeats nor broad-matrix cases,
// and comparisons stay within one backend.
export const COMPATIBLE_BASE_STEPS = Object.freeze({ cuda: 50, mlx: 8 });
export const BACKEND_STEP_DECISION = Object.freeze({
  story: 'sc-26691', approvedBy: 'user', approvedAt: '2026-10-30', measuredRun: '41061562586',
  measurement: 'MLX H3 denoise at the tiled kernel ~5.4-6.6 min/step; 50 steps ~5 h exceeds the 7200 s case bound and the 24 h lane, 8 steps ~50 min fits',
  scope: 'MLX runs the compatible-base 50-step cases at 8 steps; CUDA keeps 50; every other input and parameter identical',
});
const backendSteps = () => ({ backendSteps: { ...COMPATIBLE_BASE_STEPS } });
/** The case's declared per-backend denoise step count, or null when its recipe fixes the steps. */
export function declaredBackendSteps(c, backend = c?.backend) {
  const steps = c?.backendSteps?.[backend];
  return steps === undefined ? null : steps;
}
/** The plan's step-policy record for a backend-specific case (bound beside its sampling). */
export function backendStepPolicy(c, backend = c?.backend) {
  const steps = declaredBackendSteps(c, backend);
  assert.ok(Number.isInteger(steps) && steps >= 2, `${c?.id ?? c?.slug} declares no ${backend} step count`);
  return { kind: 'backend-specific-steps', backend, steps, byBackend: { ...c.backendSteps }, decision: { ...BACKEND_STEP_DECISION } };
}
/**
 * A planned sampling must request exactly the case's declared step count on ITS backend, and carry
 * the matching step-policy record; a case without backend-specific steps may not carry one (no
 * undeclared per-backend override).
 */
export function assertDeclaredBackendSteps(c, sampling, backend = c?.backend) {
  const steps = declaredBackendSteps(c, backend);
  if (steps === null) { assert.equal(sampling?.stepPolicy, undefined, `${c?.id ?? c?.slug} declares no backend-specific steps; an undeclared per-backend step override is refused`); return; }
  assert.equal(sampling?.steps, steps, `${c.id ?? c.slug} requests ${sampling?.steps} steps; its declared ${backend} recipe is ${steps}`);
  assert.deepEqual(sampling.stepPolicy, backendStepPolicy(c, backend), `${c.id ?? c.slug} step policy differs from the declared ${backend} decision`);
}
// The inline anchors whose timings choose the matrix backend; compared across backends, so their
// recipe (steps included) must be identical on both.
export const MATRIX_ANCHOR_ROLES = Object.freeze({ reference: 'artifact-0-ordered-mixed-baseline-ref-turbo', prompt: 'real-hybrid-text' });

// 25 operational cases per backend. max = 22 full-av + 2 actual-artifact probes + 1 source-cancel.
export const OPERATIONAL_CASE_DEFS = Object.freeze([
  caseDef(1, 'artifact-0-ordered-mixed-baseline-ref-turbo', 'full-av', 'artifact', {
    modelKey: 'artifact.0-ref2va-v2-int8', assetSet: 'singularity', conditioning: 'ordered-mixed', adapterRecipe: 'ref-turbo-baseline',
    acceptance: {
      ac: ['job.status=completed', 'job.payload.model=artifact.0-ref2va-v2-int8', 'ordered image+video+audio references reach the request', 'ref-turbo recipe steps/shift/audioShift bound', 'decoded AV matches resolved lattice'],
      e: ['jobSnapshot', 'apiTranscript', 'nativeLoadClosure', 'outputAV', 'samplingRecipe'],
    },
  }),
  caseDef(2, 'artifact-1-reference', 'full-av', 'artifact', {
    modelKey: 'artifact.1-ref2va-v1.3-int8', assetSet: 'singularity', conditioning: 'reference', adapterRecipe: 'none', ...backendSteps(),
    acceptance: { ac: ['job.status=completed', 'job.payload.model=artifact.1-ref2va-v1.3-int8', 'reference_to_video operation', 'decoded AV matches resolved lattice'], e: ['jobSnapshot', 'nativeLoadClosure', 'outputAV'] },
  }),
  caseDef(3, 'artifact-2-lokr-ref-turbo-baseline', 'full-av', 'artifact', {
    modelKey: 'artifact.2-ref2va-pruned-v1.3-int8', assetSet: 'singularity', conditioning: 'reference', adapterRecipe: 'lokr+ref-turbo-baseline',
    acceptance: { ac: ['job.status=completed', 'LoKr adapter + ref-turbo recipe bound', 'partition pruned', 'decoded AV matches resolved lattice'], e: ['jobSnapshot', 'nativeLoadClosure', 'outputAV', 'adapterClosure', 'samplingRecipe'] },
  }),
  caseDef(4, 'artifact-3-opaque-renamed-linked-source', 'full-av', 'artifact', {
    modelKey: 'artifact.3-ref2va-pruned-v1.3-w4a8', assetSet: 'singularity', conditioning: 'reference', adapterRecipe: 'none', sourceVariant: 'opaque-renamed-linked', ...backendSteps(),
    acceptance: { ac: ['job.status=completed', 'opaque renamed linked source resolves to a declared import', 'native-single source format preserved', 'decoded AV matches resolved lattice'], e: ['jobSnapshot', 'nativeLoadClosure', 'outputAV', 'sourceResolution'] },
  }),
  caseDef(5, 'real-mixed-repeat', 'full-av', 'real', {
    modelKey: 'artifact.0-ref2va-v2-int8', assetSet: 'singularity', conditioning: 'ordered-mixed', adapterRecipe: 'ref-turbo-baseline', baselineSlug: 'artifact-0-ordered-mixed-baseline-ref-turbo',
    acceptance: { ac: ['job.status=completed', 'unchanged repeated baseline of case 1', 'same source/seed/recipe/geometry', 'decoded AV is a stable repeated baseline'], e: ['jobSnapshot', 'apiTranscript', 'outputAV', 'baselineComparison'] },
  }),
  caseDef(6, 'real-drop-images', 'full-av', 'real', {
    modelKey: 'artifact.0-ref2va-v2-int8', assetSet: 'singularity', conditioning: 'ordered-mixed', adapterRecipe: 'ref-turbo-baseline', baselineSlug: 'artifact-0-ordered-mixed-baseline-ref-turbo', broad: true,
    acceptance: { ac: ['job.status=completed', 'only image references dropped vs case 1', 'causal decoded-media intervention remains observable'], e: ['jobSnapshot', 'outputAV', 'baselineComparison'] },
  }),
  caseDef(7, 'real-drop-video', 'full-av', 'real', {
    modelKey: 'artifact.0-ref2va-v2-int8', assetSet: 'singularity', conditioning: 'ordered-mixed', adapterRecipe: 'ref-turbo-baseline', baselineSlug: 'artifact-0-ordered-mixed-baseline-ref-turbo', broad: true,
    acceptance: { ac: ['job.status=completed', 'only video references dropped vs case 1', 'causal decoded-media intervention remains observable'], e: ['jobSnapshot', 'outputAV', 'baselineComparison'] },
  }),
  caseDef(8, 'real-drop-audio', 'full-av', 'real', {
    modelKey: 'artifact.0-ref2va-v2-int8', assetSet: 'singularity', conditioning: 'ordered-mixed', adapterRecipe: 'ref-turbo-baseline', baselineSlug: 'artifact-0-ordered-mixed-baseline-ref-turbo', bothBackendsCausal: true,
    acceptance: { ac: ['job.status=completed', 'only audio references dropped vs case 1', 'actual mixed causal check runs on both backends'], e: ['jobSnapshot', 'outputAV', 'baselineComparison'] },
  }),
  caseDef(9, 'real-reverse-images', 'full-av', 'real', {
    modelKey: 'artifact.0-ref2va-v2-int8', assetSet: 'singularity', conditioning: 'ordered-mixed', adapterRecipe: 'ref-turbo-baseline', baselineSlug: 'artifact-0-ordered-mixed-baseline-ref-turbo', broad: true,
    acceptance: { ac: ['job.status=completed', 'only image reference order reversed vs case 1', 'causal decoded-media intervention remains observable'], e: ['jobSnapshot', 'outputAV', 'baselineComparison'] },
  }),
  caseDef(10, 'real-reverse-video', 'full-av', 'real', {
    modelKey: 'artifact.0-ref2va-v2-int8', assetSet: 'singularity', conditioning: 'ordered-mixed', adapterRecipe: 'ref-turbo-baseline', baselineSlug: 'artifact-0-ordered-mixed-baseline-ref-turbo', broad: true,
    acceptance: { ac: ['job.status=completed', 'only video reference order reversed vs case 1', 'causal decoded-media intervention remains observable'], e: ['jobSnapshot', 'outputAV', 'baselineComparison'] },
  }),
  caseDef(11, 'real-reverse-audio', 'full-av', 'real', {
    modelKey: 'artifact.0-ref2va-v2-int8', assetSet: 'singularity', conditioning: 'ordered-mixed', adapterRecipe: 'ref-turbo-baseline', baselineSlug: 'artifact-0-ordered-mixed-baseline-ref-turbo', broad: true,
    acceptance: { ac: ['job.status=completed', 'only audio reference order reversed vs case 1', 'causal decoded-media intervention remains observable'], e: ['jobSnapshot', 'outputAV', 'baselineComparison'] },
  }),
  caseDef(12, 'real-drop-lora', 'full-av', 'real', {
    modelKey: 'artifact.0-ref2va-v2-int8', assetSet: 'singularity', conditioning: 'ordered-mixed', adapterRecipe: 'drop-lora', baselineSlug: 'artifact-0-ordered-mixed-baseline-ref-turbo',
    acceptance: { ac: ['job.status=completed', 'only LoRA removed vs case 1', 'source/seed/geometry/ref-turbo recipe retained'], e: ['jobSnapshot', 'outputAV', 'baselineComparison'] },
  }),
  caseDef(13, 'real-lokr-repeat', 'full-av', 'real', {
    modelKey: 'artifact.2-ref2va-pruned-v1.3-int8', assetSet: 'singularity', conditioning: 'reference', adapterRecipe: 'lokr-repeat', baselineSlug: 'artifact-2-lokr-ref-turbo-baseline',
    acceptance: { ac: ['job.status=completed', 'unchanged repeated baseline of case 3', 'same LoKr/ref-turbo recipe'], e: ['jobSnapshot', 'outputAV', 'baselineComparison'] },
  }),
  caseDef(14, 'real-drop-lokr', 'full-av', 'real', {
    modelKey: 'artifact.2-ref2va-pruned-v1.3-int8', assetSet: 'singularity', conditioning: 'reference', adapterRecipe: 'drop-lokr', baselineSlug: 'artifact-2-lokr-ref-turbo-baseline',
    acceptance: { ac: ['job.status=completed', 'only LoKr removed vs case 3', 'ref-turbo recipe retained'], e: ['jobSnapshot', 'outputAV', 'baselineComparison'] },
  }),
  caseDef(15, 'real-hybrid-text', 'full-av', 'real', {
    modelKey: 'hybrid-single-checkpoint', assetSet: 'hybrid', conditioning: 'text', adapterRecipe: 'fl-turbo-4v01',
    acceptance: { ac: ['job.status=completed', 'hybrid text_to_video operation', 'declared hybrid source and fl-turbo-4v01 recipe bound'], e: ['jobSnapshot', 'nativeLoadClosure', 'outputAV', 'samplingRecipe'] },
  }),
  caseDef(16, 'real-hybrid-prompt-change', 'full-av', 'real', {
    modelKey: 'hybrid-single-checkpoint', assetSet: 'hybrid', conditioning: 'text', adapterRecipe: 'fl-turbo-4v01', baselineSlug: 'real-hybrid-text', broad: true,
    acceptance: { ac: ['job.status=completed', 'only prompt changed vs case 15', 'decoded-media change attributable to prompt'], e: ['jobSnapshot', 'outputAV', 'baselineComparison'] },
  }),
  caseDef(17, 'real-hybrid-image', 'full-av', 'real', {
    modelKey: 'hybrid-single-checkpoint', assetSet: 'hybrid', conditioning: 'image', adapterRecipe: 'fl-turbo-8',
    acceptance: { ac: ['job.status=completed', 'hybrid image_to_video operation', 'fl-turbo-8 recipe bound'], e: ['jobSnapshot', 'nativeLoadClosure', 'outputAV', 'samplingRecipe'] },
  }),
  caseDef(18, 'real-hybrid-last-only', 'full-av', 'real', {
    modelKey: 'hybrid-single-checkpoint', assetSet: 'hybrid', conditioning: 'last-frame-only', adapterRecipe: 'fl-turbo-8',
    acceptance: { ac: ['job.status=completed', 'last-frame-only image_to_video operation', 'no sourceAssetId', 'fl-turbo-8 recipe bound'], e: ['jobSnapshot', 'outputAV', 'samplingRecipe'] },
  }),
  caseDef(19, 'real-hybrid-first-last', 'full-av', 'real', {
    modelKey: 'hybrid-single-checkpoint', assetSet: 'hybrid', conditioning: 'first-last-frame', adapterRecipe: 'fl-turbo-4v1.0-768',
    acceptance: { ac: ['job.status=completed', 'first_last_frame operation with both endpoints', 'fl-turbo-4v1.0_768 recipe bound'], e: ['jobSnapshot', 'outputAV', 'samplingRecipe'] },
  }),
  caseDef(20, 'real-hybrid-reference', 'full-av', 'real', {
    modelKey: 'hybrid-single-checkpoint', assetSet: 'hybrid', conditioning: 'reference', adapterRecipe: 'compatible-declared-sampling', ...backendSteps(),
    acceptance: { ac: ['job.status=completed', 'hybrid reference_to_video operation', 'compatible declared source and sampling only'], e: ['jobSnapshot', 'outputAV', 'samplingRecipe', 'sourceResolution'] },
  }),
  caseDef(21, 'real-independent-fl2va', 'full-av', 'real', {
    modelKey: 'independent-pulpcut-merge-fl2va', assetSet: 'independent', conditioning: 'text', adapterRecipe: 'none', sourceVariant: 'independent-finetune-pulpcut-merge',
    acceptance: { ac: ['job.status=completed', 'independently identified PulpCut MERGE source', 'fl2va operation', 'decoded AV matches resolved lattice'], e: ['jobSnapshot', 'nativeLoadClosure', 'outputAV', 'sourceResolution'] },
  }),
  caseDef(22, 'real-hybrid-text-repeat', 'full-av', 'real', {
    modelKey: 'hybrid-single-checkpoint', assetSet: 'hybrid', conditioning: 'text', adapterRecipe: 'fl-turbo-4v01', baselineSlug: 'real-hybrid-text', broad: true,
    acceptance: { ac: ['job.status=completed', 'unchanged repeated baseline of case 15', 'same declared hybrid source/recipe'], e: ['jobSnapshot', 'outputAV', 'baselineComparison'] },
  }),
  caseDef(23, 'numeric-pruned-artifact', 'actual-artifact-probe', 'numeric', {
    modelKey: 'artifact.2-ref2va-pruned-v1.3-int8', assetSet: 'singularity', probe: 'pruned-artifact',
    acceptance: { ac: ['actual bounded real tensor probe', 'no AV output claimed', 'pruned interpolation artifact matches the declared fixture'], e: ['artifactProbeReceipt', 'artifactSha256'] },
  }),
  caseDef(24, 'numeric-w4a8-artifact', 'actual-artifact-probe', 'numeric', {
    modelKey: 'artifact.3-ref2va-pruned-v1.3-w4a8', assetSet: 'singularity', probe: 'w4a8-packed-convrot',
    acceptance: { ac: ['actual mixed packed/ConvRot tensor probe', 'no AV output claimed'], e: ['artifactProbeReceipt', 'artifactSha256'] },
  }),
  caseDef(25, 'lifecycle-generation-source-prep-cancel', 'source-cancel', 'lifecycle', {
    // sc-26691: a ref2va generation must name a visual reference or the API refuses it at submit
    // ("Reference to Video requires at least one reference image or video clip") before any cold
    // source preparation could start.
    modelKey: 'artifact.3-ref2va-pruned-v1.3-w4a8', assetSet: 'singularity', conditioning: 'reference', probe: 'native-source-prep-cancel', ...backendSteps(),
    acceptance: { ac: ['actual native cold source-preparation observer', 'no AV output claimed', 'no Acquired seal / source_prepared / native load for the canceled job'], e: ['cancelReceipt', 'partialByteEvidence', 'joinedTeardownMarker'] },
  }),
]);
export const OPERATIONAL_CASES_PER_BACKEND = OPERATIONAL_CASE_DEFS.length; // 25
export const MAX_OPERATIONAL_EXECUTIONS = OPERATIONAL_CASES_PER_BACKEND * BACKENDS.length; // 50

// Which operational cases provide real-weight evidence for each catalog cell (by slug).
export const REAL_CELL_MAP = Object.freeze({
  'artifact.0': ['artifact-0-ordered-mixed-baseline-ref-turbo'],
  'artifact.1': ['artifact-1-reference'],
  'artifact.2': ['artifact-2-lokr-ref-turbo-baseline'],
  'artifact.3': ['artifact-3-opaque-renamed-linked-source'],
  'numeric.convrot': ['numeric-w4a8-artifact'],
  'numeric.pruned-interpolation': ['numeric-pruned-artifact'],
  'numeric.w4a8': ['numeric-w4a8-artifact'],
  'capability.text': ['real-hybrid-text', 'real-hybrid-prompt-change', 'real-hybrid-text-repeat'],
  'capability.image': ['real-hybrid-image'],
  'capability.first-last-frame': ['real-hybrid-first-last'],
  'capability.last-frame-only': ['real-hybrid-last-only'],
  'capability.video': ['artifact-0-ordered-mixed-baseline-ref-turbo', 'artifact-3-opaque-renamed-linked-source', 'real-drop-video', 'real-reverse-video'],
  'capability.ordered-multimodal': ['artifact-0-ordered-mixed-baseline-ref-turbo', 'real-mixed-repeat', 'real-drop-images', 'real-drop-video', 'real-drop-audio', 'real-reverse-images', 'real-reverse-video', 'real-reverse-audio'],
  'capability.audio-reference': ['artifact-0-ordered-mixed-baseline-ref-turbo', 'real-drop-audio', 'real-reverse-audio'],
  'capability.hybrid': ['real-hybrid-text', 'real-hybrid-image', 'real-hybrid-reference'],
  'capability.turbo': ['artifact-0-ordered-mixed-baseline-ref-turbo', 'real-hybrid-text', 'real-hybrid-image', 'real-hybrid-first-last'],
  'capability.lora': ['artifact-0-ordered-mixed-baseline-ref-turbo', 'real-mixed-repeat', 'real-drop-lora'],
  'capability.lokr': ['artifact-2-lokr-ref-turbo-baseline', 'real-lokr-repeat', 'real-drop-lokr'],
  'generalization.renamed': ['artifact-3-opaque-renamed-linked-source'],
  'generalization.independent-finetune': ['real-independent-fl2va'],
  'generalization.native-single': ['artifact-3-opaque-renamed-linked-source'],
  'lifecycle.generation-source-prep-cancel': ['lifecycle-generation-source-prep-cancel'],
});
// Cells whose catalog guarantee can also be met by an explicit executed proof when the optional
// real case(s) are absent (e.g. native fixtures plus an actual artifact probe).
export const PROOF_FALLBACK_CELLS = Object.freeze(new Set([
  'numeric.tensor-mapping', 'numeric.convrot', 'generalization.renamed', 'capability.lora', 'capability.lokr',
]));

const slugById = (backend, slug) => `${backend}.reduced.${slug}`;
const defBySlug = new Map(OPERATIONAL_CASE_DEFS.map(def => [def.slug, def]));
const HEX40 = /^[a-f0-9]{40}$/;
const HEX64 = /^[a-f0-9]{64}$/;

export function operationalCases(backend) {
  const backends = backend === undefined ? BACKENDS : [backend];
  for (const value of backends) if (!BACKENDS.includes(value)) throw new Error(`unknown backend ${value}`);
  return backends.flatMap(value => OPERATIONAL_CASE_DEFS.map(def => ({
    ...structuredClone(def),
    id: slugById(value, def.slug),
    backend: value,
    baseline: def.baselineSlug ? slugById(value, def.baselineSlug) : null,
    repeatsOf: def.repeatsSlug ? slugById(value, def.repeatsSlug) : null,
  })));
}
export function operationalCaseById(id) {
  return operationalCases().find(value => value.id === id) ?? null;
}
function assertNoInventedIdentity(value, path = 'case') {
  if (typeof value === 'string') {
    if (HEX40.test(value) || HEX64.test(value)) throw new Error(`${path}: invented SHA-like identity is not allowed in the manifest`);
    if (value.startsWith('/') || /^[A-Za-z]:[\\/]/.test(value)) throw new Error(`${path}: invented absolute path is not allowed in the manifest`);
    return;
  }
  if (Array.isArray(value)) { value.forEach((item, index) => assertNoInventedIdentity(item, `${path}[${index}]`)); return; }
  if (value && typeof value === 'object') for (const [key, item] of Object.entries(value)) assertNoInventedIdentity(item, `${path}.${key}`);
}
// Validate the 25/backend operational inventory, stable IDs, dependencies, catalog membership and
// the absence of guessed paths/SHAs/metrics.
export function validateOperationalCaseInventory(plan = {}) {
  if (Array.isArray(plan)) plan = { cases: plan };
  const cases = plan.cases ?? operationalCases();
  const catalog = normalizeCatalog(plan.catalog ?? defaultCatalog());
  const catalogSet = new Set(catalog.map(cell => cell.id));
  for (const item of cases) assertNoInventedIdentity(item, item.id ?? 'case');
  assert.equal(cases.length, MAX_OPERATIONAL_EXECUTIONS, 'exactly 25 operational cases per backend');
  assert.equal(new Set(cases.map(value => value.id)).size, cases.length, 'operational case ids must be unique');
  const byBackend = { mlx: [], cuda: [] };
  for (const item of cases) {
    assert.ok(BACKENDS.includes(item.backend), `unknown backend on ${item.id}`);
    assert.equal(item.id, slugById(item.backend, item.slug), `unstable operational id ${item.id}`);
    assert.ok(['full-av', 'actual-artifact-probe', 'source-cancel'].includes(item.kind), `invalid kind on ${item.id}`);
    assert.ok(Array.isArray(item.acceptance?.ac) && item.acceptance.ac.length > 0, `missing AC on ${item.id}`);
    assert.ok(Array.isArray(item.acceptance?.e) && item.acceptance.e.length > 0, `missing evidence pointers on ${item.id}`);
    assert.ok(item.modelKey && item.assetSet, `missing model/asset identity on ${item.id}`);
    if (item.kind === 'full-av') assert.ok(item.conditioning && item.adapterRecipe, `full-av case ${item.id} needs conditioning/adapter recipe`);
    byBackend[item.backend].push(item);
  }
  for (const backend of BACKENDS) {
    const selected = byBackend[backend];
    assert.equal(selected.length, OPERATIONAL_CASES_PER_BACKEND, `${backend} must have 25 operational cases`);
    const kinds = selected.reduce((counts, item) => ({ ...counts, [item.kind]: (counts[item.kind] ?? 0) + 1 }), {});
    assert.deepEqual(kinds, { 'full-av': 22, 'actual-artifact-probe': 2, 'source-cancel': 1 }, `${backend} kind counts must be 22/2/1`);
    const byId = new Map(selected.map(item => [item.id, item]));
    for (const item of selected) {
      if (item.baseline) {
        const baseline = byId.get(item.baseline);
        assert.ok(baseline, `${item.id} baseline ${item.baseline} is not in the same backend`);
        assert.ok(baseline.index < item.index, `${item.id} baseline must precede the case`);
        assert.equal(baseline.kind, 'full-av', `${item.id} baseline must be full-av`);
      }
      if (item.repeatsOf) assert.ok(byId.has(item.repeatsOf), `${item.id} repeatsOf target missing`);
    }
    for (const slug of [...BROAD_MATRIX_CASES, BOTH_BACKEND_CAUSAL_CASE]) assert.ok(defBySlug.has(slug), `matrix case ${slug} missing from the inventory`);
    // Backend-specific step counts (sc-26691, 2026-10-30) are the declared compatible-base pair,
    // and a case carrying them is never compared across backends or against a repeat.
    const comparedSlugs = new Set([...Object.values(MATRIX_ANCHOR_ROLES), ...BROAD_MATRIX_CASES, BOTH_BACKEND_CAUSAL_CASE, ...selected.flatMap(value => value.baselineSlug ? [value.slug, value.baselineSlug] : [])]);
    for (const item of selected) if (item.backendSteps !== undefined) {
      assert.deepEqual(item.backendSteps, { ...COMPATIBLE_BASE_STEPS }, `${item.id} backend step counts must be the declared compatible-base pair`);
      assert.ok(!comparedSlugs.has(item.slug), `${item.id} has backend-specific steps and cannot be an anchor, baseline, repeat or matrix case`);
    }
  }
  for (const suffix of Object.keys(REAL_CELL_MAP)) assert.ok(CATALOG_SUFFIXES.includes(suffix), `real mapping references unknown catalog cell ${suffix}`);
  for (const slug of Object.values(REAL_CELL_MAP).flat()) assert.ok(defBySlug.has(slug), `real cell map references unknown operational case ${slug}`);
  assert.ok(catalogSet.size === CATALOG_CELLS, 'catalog must hold exactly 84 cells');
  return { ok: true, cases, backends: { mlx: byBackend.mlx.length, cuda: byBackend.cuda.length }, catalog };
}

function normalizeCatalog(catalog) {
  assert.ok(Array.isArray(catalog), 'catalog array required');
  const cells = catalog.map(entry => {
    const id = typeof entry === 'string' ? entry : entry?.id;
    assert.ok(typeof id === 'string', 'catalog cell id required');
    const [backend, ...rest] = id.split('.');
    assert.ok(BACKENDS.includes(backend), `catalog cell ${id} has an unknown backend`);
    const suffix = rest.join('.');
    assert.ok(CATALOG_SUFFIXES.includes(suffix), `catalog cell ${id} is not a known regression cell`);
    if (typeof entry === 'object' && entry.backend !== undefined) assert.equal(entry.backend, backend, `catalog cell ${id} backend mismatch`);
    return { id, backend, suffix, group: suffix.split('.')[0] };
  });
  assert.equal(cells.length, CATALOG_CELLS, 'catalog must hold exactly 84 cells');
  assert.equal(new Set(cells.map(cell => cell.id)).size, CATALOG_CELLS, 'catalog cell ids must be unique');
  for (const backend of BACKENDS) assert.equal(cells.filter(cell => cell.backend === backend).length, CATALOG_CELLS_PER_BACKEND, `${backend} must hold 42 catalog cells`);
  return cells;
}

// Exact proof references only: provided ids/run refs, never invented.
function normalizeProofIndex(proofIndex) {
  assert.ok(Array.isArray(proofIndex), 'proofIndex array required');
  const known = new Set(defaultCatalog().map(cell => cell.id));
  return proofIndex.map((entry, index) => {
    assert.ok(entry && typeof entry === 'object', `proofIndex[${index}] must be an object`);
    assert.ok(typeof entry.id === 'string' && entry.id.length > 0, `proofIndex[${index}] needs an explicit id`);
    assert.equal(entry.status, 'passed', `proofIndex ${entry.id} is not passed`);
    assert.ok(Array.isArray(entry.catalogIds) && entry.catalogIds.length > 0, `proofIndex ${entry.id} needs explicit catalog memberships`);
    for (const id of entry.catalogIds) assert.ok(known.has(id), `proofIndex ${entry.id} references unknown catalog cell ${id}`);
    assert.match(entry.sourceSha ?? '', HEX40, `proofIndex ${entry.id} needs the executed source SHA`);
    assert.ok(entry.config && typeof entry.config === 'object' && Object.keys(entry.config).length > 0, `proofIndex ${entry.id} needs the executed configuration`);
    assert.ok(typeof entry.runRef === 'string' && entry.runRef.length > 0, `proofIndex ${entry.id} needs the parent-provided run reference`);
    assert.ok(Array.isArray(entry.testSelectors) && entry.testSelectors.length > 0 && entry.testSelectors.every(value => typeof value === 'string' && value.length > 0), `proofIndex ${entry.id} needs exact test selectors`);
    return {
      id: entry.id, kind: entry.kind ?? 'executed-proof', status: entry.status, catalogIds: [...entry.catalogIds],
      sourceSha: entry.sourceSha, config: structuredClone(entry.config), runRef: entry.runRef,
      testSelectors: [...entry.testSelectors], backend: entry.backend ?? null, caseId: entry.caseId ?? null, guarantees: entry.guarantees ?? [],
    };
  });
}
function normalizeReceipts(realReceipts) {
  assert.ok(Array.isArray(realReceipts), 'realReceipts array required');
  const cases = new Map(operationalCases().map(item => [item.id, item]));
  const receipts = new Map();
  for (const entry of realReceipts) {
    const item = cases.get(entry?.caseId);
    assert.ok(item, `real receipt references unknown operational case ${entry?.caseId}`);
    assert.equal(entry.backend, item.backend, `real receipt ${entry.caseId} backend mismatch`);
    assert.equal(entry.status, 'passed', `real receipt ${entry.caseId} is not passed`);
    assert.ok(typeof entry.runRef === 'string' && entry.runRef.length > 0, `real receipt ${entry.caseId} needs the actual run reference`);
    assert.ok(entry.evidence && typeof entry.evidence === 'object', `real receipt ${entry.caseId} needs actual evidence`);
    if (entry.sourceSha !== undefined) assert.match(entry.sourceSha, HEX40, `real receipt ${entry.caseId} has an invalid source SHA`);
    if (item.kind === 'full-av') assert.ok(Array.isArray(entry.evidence.jobIds) && entry.evidence.jobIds.length > 0, `full-av case ${entry.caseId} needs actual job ids`);
    if (item.kind === 'actual-artifact-probe') assert.ok(entry.evidence.artifactSha256 && HEX64.test(entry.evidence.artifactSha256) || typeof entry.evidence.probeReceipt === 'string', `probe case ${entry.caseId} needs actual artifact/probe evidence`);
    if (item.kind === 'source-cancel') assert.ok(typeof entry.evidence.cancelReceipt === 'string' && entry.evidence.cancelReceipt.length > 0, `source-cancel case ${entry.caseId} needs its actual cancel receipt`);
    const known = receipts.get(entry.caseId);
    if (!known) receipts.set(entry.caseId, { ...entry, backend: item.backend, kind: item.kind });
  }
  return receipts;
}

function cellProofs(cell, proofs) {
  return proofs.filter(proof => proof.kind !== 'cheap-causal-witness' && proof.status === 'passed' && proof.catalogIds.includes(cell.id));
}
function triadFor({ backend, selectedBackend, slug, receipts, proofs }) {
  const caseId = slugById(backend, slug);
  const witness = proofs.find(proof => proof.kind === 'cheap-causal-witness' && proof.backend === backend && proof.status === 'passed'
    && (proof.config?.encodedSeamOnly===true || proof.caseId === caseId || (Array.isArray(proof.guarantees) && proof.guarantees.includes(slug)))
    && proof.catalogIds.some(id => id.startsWith(`${backend}.`)));
  const ownMixedCaseId = slugById(backend, BOTH_BACKEND_CAUSAL_CASE);
  const peerCaseId = slugById(selectedBackend, slug);
  const ownMixed = receipts.get(ownMixedCaseId);
  const peer = receipts.get(peerCaseId);
  return {
    caseId,
    witnessId: witness?.id ?? null, witnessScope:'encoded-native-surface-only', endToEndAVProven:false,
    ownMixedCaseId: ownMixed ? ownMixedCaseId : null,
    peerCaseId: peer ? peerCaseId : null,
    ok: Boolean(witness && ownMixed && peer),
  };
}
function isOmitted(slug, backend, matrixDecision) {
  return matrixDecision?.status === 'selected' && backend === matrixDecision.displacedBackend && BROAD_MATRIX_CASES.includes(slug);
}

// Map every one of the 84 catalog cells to exactly one category. Missing receipts/proofs are
// UNPROVEN reasons, never default success.
export function validateMatrixDecision(matrixDecision) {
  if (matrixDecision === undefined) return;
  assert.ok(matrixDecision && typeof matrixDecision === 'object', 'matrixDecision must be an object');
  assert.ok(['selected', 'unresolved'].includes(matrixDecision.status), 'matrixDecision status must be selected or unresolved');
  if (matrixDecision.status === 'selected') {
    assert.ok(BACKENDS.includes(matrixDecision.selectedBackend), 'matrixDecision selectedBackend invalid');
    assert.ok(BACKENDS.includes(matrixDecision.displacedBackend), 'matrixDecision displacedBackend invalid');
    assert.notEqual(matrixDecision.selectedBackend, matrixDecision.displacedBackend, 'matrixDecision backends must differ');
  }
  return matrixDecision;
}
export function buildCatalogCoverage(catalogOrOptions = {}, realReceiptsArg, proofIndexArg, matrixDecisionArg) {
  const options = Array.isArray(catalogOrOptions) ? { catalog: catalogOrOptions, realReceipts: realReceiptsArg, proofIndex: proofIndexArg, matrixDecision: matrixDecisionArg } : catalogOrOptions ?? {};
  const { catalog = defaultCatalog(), realReceipts = [], proofIndex = [], matrixDecision } = options;
  validateMatrixDecision(matrixDecision);
  const inventory = validateOperationalCaseInventory({ catalog });
  const cells = inventory.catalog;
  const receipts = normalizeReceipts(realReceipts);
  const proofs = normalizeProofIndex(proofIndex);
  const coverage = [], unproven = [], omittedCoverage = new Map();
  for (const cell of cells) {
    const mappedSlugs = REAL_CELL_MAP[cell.suffix] ?? [];
    const mappedIds = mappedSlugs.map(slug => slugById(cell.backend, slug));
    const omittedIds = mappedIds.filter(id => isOmitted(defBySlug.get(id.split('.reduced.')[1])?.slug, cell.backend, matrixDecision));
    const requiredIds = mappedIds.filter(id => !omittedIds.includes(id));
    const missingIds = requiredIds.filter(id => !receipts.has(id));
    const triads = omittedIds.map(id => {
      const slug = id.split('.reduced.')[1];
      const triad = triadFor({ backend: cell.backend, selectedBackend: matrixDecision?.selectedBackend, slug, receipts, proofs });
      omittedCoverage.set(triad.caseId, triad);
      return triad;
    });
    const failedTriads = triads.filter(triad => !triad.ok);
    const matchingProofs = cellProofs(cell, proofs);
    if (mappedIds.length > 0) {
      if (omittedIds.length === 0 && failedTriads.length === 0 && missingIds.length === 0) {
        coverage.push({ cellId: cell.id, backend: cell.backend, group: cell.group, category: 'real-weight', reducedCaseIds: requiredIds, omittedCaseIds: omittedIds, witness: triads.length ? triads : undefined, proofs: [], reason: null });
      } else if (omittedIds.length === 0 && PROOF_FALLBACK_CELLS.has(cell.suffix) && failedTriads.length === 0 && matchingProofs.length > 0) {
        coverage.push({ cellId: cell.id, backend: cell.backend, group: cell.group, category: 'executed-proof', reducedCaseIds: requiredIds.filter(id => receipts.has(id)), omittedCaseIds: omittedIds, missingRealCaseIds: missingIds, proofs: matchingProofs.map(proofRef), reason: null });
      } else {
        const reason = failedTriads.length
          ? `omitted broad case coverage is missing a cheap witness / own case8 / peer matrix receipt: ${failedTriads.map(triad => triad.caseId).join(', ')}`
          : omittedIds.length ? 'Surface exercised via passed native encoded witnesses and an actual mixed render where retained; this omitted end-to-end render cell is NOT proven on this backend.' : `missing passed real receipt(s): ${missingIds.join(', ')}`;
        coverage.push({ cellId: cell.id, backend: cell.backend, group: cell.group, category: 'unproven', reducedCaseIds: requiredIds, omittedCaseIds: omittedIds, surfaceWitnesses:triads, proofs: [], reason });
        unproven.push({ cellId: cell.id, reason });
      }
    } else if (matchingProofs.length > 0) {
      coverage.push({ cellId: cell.id, backend: cell.backend, group: cell.group, category: 'executed-proof', reducedCaseIds: [], omittedCaseIds: [], proofs: matchingProofs.map(proofRef), reason: null });
    } else {
      const reason = 'no real-weight case and no explicit executed-proof receipt provided';
      coverage.push({ cellId: cell.id, backend: cell.backend, group: cell.group, category: 'unproven', reducedCaseIds: [], omittedCaseIds: [], proofs: [], reason });
      unproven.push({ cellId: cell.id, reason });
    }
  }
  assert.equal(coverage.length, CATALOG_CELLS, 'every catalog cell must be categorized');
  for (const entry of coverage) assert.ok(['real-weight', 'executed-proof', 'unproven'].includes(entry.category), `invalid category on ${entry.cellId}`);
  return { coverage, unproven, omittedCoverage: [...omittedCoverage.values()], receipts, proofs, matrixDecision: matrixDecision ?? null };
}
function proofRef(proof) {
  return { id: proof.id, sourceSha: proof.sourceSha, config: structuredClone(proof.config), status: proof.status, testSelectors: [...proof.testSelectors], runRef: proof.runRef };
}

// Count shared actual jobs once; conflicting identities for one job id invalidate the closure.
export function summarizeUniqueExecutions(executions = []) {
  assert.ok(Array.isArray(executions), 'executions array required');
  const byJob = new Map(), shared = [], conflicts = [], unidentified = [];
  for (const row of executions) {
    const caseId = row?.caseId, backend = row?.backend, jobId = row?.jobId;
    assert.ok(typeof caseId === 'string' && BACKENDS.includes(backend), 'execution row needs a case id and a known backend');
    if (!jobId) { unidentified.push({ caseId, backend, role: row.role ?? null }); continue; }
    assert.ok(typeof jobId === 'string' && jobId.length > 0, 'job id must be a non-empty string');
    const identity = JSON.stringify({ backend, sourceSha: typeof row.sourceSha === 'string' ? row.sourceSha : null, modelKey: typeof row.modelKey === 'string' ? row.modelKey : null });
    const entry = byJob.get(jobId) ?? { jobId, caseIds: new Set(), identities: new Set() };
    entry.caseIds.add(caseId); entry.identities.add(identity); byJob.set(jobId, entry);
  }
  for (const entry of byJob.values()) {
    if (entry.identities.size > 1) conflicts.push({ jobId: entry.jobId, identities: [...entry.identities], caseIds: [...entry.caseIds] });
    if (entry.caseIds.size > 1) shared.push({ jobId: entry.jobId, caseIds: [...entry.caseIds] });
  }
  return { uniqueExecutions: byJob.size, shared, conflicts, unidentified };
}

// Choose the matrix backend only from finite positive INLINE anchor timings on BOTH backends with
// matching recipe/geometry/source/pin and recorded hardware. Pending/missing or tied anchors leave
// the decision unresolved (unselected); no backend is assumed faster.
export function chooseMatrixBackend(anchorsOrOptions = {}) {
  const options = (Array.isArray(anchorsOrOptions) || (anchorsOrOptions && !('anchors' in anchorsOrOptions) && ('mlx' in anchorsOrOptions || 'cuda' in anchorsOrOptions)))
    ? { anchors: anchorsOrOptions } : anchorsOrOptions ?? {};
  const { anchors, weights = { reference: 5, prompt: 2 } } = options;
  assert.ok(weights && typeof weights === 'object', 'weights object required');
  assert.ok(Number.isFinite(weights.reference) && weights.reference > 0, 'reference weight must be positive');
  assert.ok(Number.isFinite(weights.prompt) && weights.prompt > 0, 'prompt weight must be positive');
  const rows = Array.isArray(anchors) ? anchors : BACKENDS.flatMap(backend => Array.isArray(anchors?.[backend]) ? anchors[backend].map(row => ({ ...row, backend: row.backend ?? backend })) : []);
  assert.ok(rows.length > 0, 'inline anchor timings required');
  const roles = MATRIX_ANCHOR_ROLES;
  // A missing or not-yet-measured (pending) anchor means UNSELECTED: no credit and no skip.
  const pending = [];
  for (const [role, slug] of Object.entries(roles)) for (const backend of BACKENDS) {
    const measured = rows.filter(row => row.role === role && row.backend === backend);
    if (measured.length === 0 || measured.some(row => row.wallMs === undefined || row.wallMs === null)) pending.push(`${role}/${backend} ${slugById(backend, slug)}`);
  }
  if (pending.length > 0) {
    return { status: 'unresolved', reason: `pending inline anchors: ${pending.join(', ')}; no matrix backend selected`, anchorIds: null, numbers: null, weights: { ...weights } };
  }
  const normalized = {};
  for (const [role, slug] of Object.entries(roles)) {
    const selected = rows.filter(row => row.role === role);
    assert.equal(selected.length, BACKENDS.length, `anchor role ${role} requires exactly one measurement per backend`);
    for (const backend of BACKENDS) assert.equal(selected.filter(row => row.backend === backend).length, 1, `anchor role ${role} missing on ${backend}`);
    for (const row of selected) {
      assert.equal(row.caseId ?? slugById(row.backend, slug), slugById(row.backend, slug), `anchor ${role}/${row.backend} is not the required case`);
      assert.ok(Number.isFinite(row.wallMs) && row.wallMs > 0, `anchor ${role}/${row.backend} needs finite positive observed wall time`);
      assert.ok(row.recipe && row.geometry && row.sourceSha && row.pinSha && row.hardware, `anchor ${role}/${row.backend} needs recipe/geometry/source/pin/hardware`);
    }
    for (const field of ['recipe', 'geometry', 'sourceSha', 'pinSha']) {
      const mlx = selected.find(row => row.backend === 'mlx')?.[field], cuda = selected.find(row => row.backend === 'cuda')?.[field];
      assert.ok(mlx !== undefined && cuda !== undefined, `anchor ${role} lacks ${field}`);
      assert.deepEqual(mlx, cuda, `anchor ${role} ${field} differs across backends; no valid matrix comparison`);
    }
    for (const backend of BACKENDS) {
      const row = selected.find(value => value.backend === backend);
      normalized[backend] = normalized[backend] ?? {};
      normalized[backend][role] = { caseId: slugById(backend, slug), role, wallMs: row.wallMs, hardware: String(row.hardware), recipe: row.recipe, geometry: row.geometry, sourceSha: row.sourceSha, pinSha: row.pinSha };
    }
  }
  for (const backend of BACKENDS) {
    const pair = normalized[backend];
    assert.equal(pair.reference.sourceSha, pair.prompt.sourceSha, `${backend} anchors must share the staged source SHA`);
    assert.equal(pair.reference.pinSha, pair.prompt.pinSha, `${backend} anchors must share the source pin`);
    assert.equal(pair.reference.hardware, pair.prompt.hardware, `${backend} anchors must share the hardware identity`);
  }
  const numbers = Object.fromEntries(BACKENDS.map(backend => [backend, {
    referenceWallMs: normalized[backend].reference.wallMs,
    promptWallMs: normalized[backend].prompt.wallMs,
    weightedCost: normalized[backend].reference.wallMs * weights.reference + normalized[backend].prompt.wallMs * weights.prompt,
  }]));
  const anchorIds = { reference: Object.fromEntries(BACKENDS.map(backend => [backend, normalized[backend].reference.caseId])), prompt: Object.fromEntries(BACKENDS.map(backend => [backend, normalized[backend].prompt.caseId])) };
  if (numbers.mlx.weightedCost === numbers.cuda.weightedCost) {
    return { status: 'unresolved', reason: 'measured incremental cost is equal; no matrix backend is selected', anchorIds, numbers, weights: { ...weights } };
  }
  const selectedBackend = numbers.mlx.weightedCost < numbers.cuda.weightedCost ? 'mlx' : 'cuda';
  const displacedBackend = selectedBackend === 'mlx' ? 'cuda' : 'mlx';
  return { status: 'selected', selectedBackend, displacedBackend, anchorIds, numbers, weights: { ...weights }, reason: `${selectedBackend} has the lower measured incremental cost (${numbers[selectedBackend].weightedCost} < ${numbers[displacedBackend].weightedCost})` };
}

// Full closure validation: inventory, 84-cell categorization, conditional execution counts and the
// asymmetric-matrix witness triad. Returns unproven reasons rather than default success.
export function validateClosure({ catalog = defaultCatalog(), realReceipts = [], proofIndex = [], matrixDecision, limitations = [], executionRows } = {}) {
  const inventory = validateOperationalCaseInventory({ catalog });
  const built = buildCatalogCoverage({ catalog, realReceipts, proofIndex, matrixDecision });
  const receipts = built.receipts;
  const executedCases = [...receipts.keys()].sort();
  const byCase = new Map(operationalCases().map(item => [item.id, item]));
  const selected = matrixDecision?.status === 'selected' ? matrixDecision.selectedBackend : null;
  const displaced = matrixDecision?.status === 'selected' ? matrixDecision.displacedBackend : null;
  const expectedCases = [];
  if (selected) {
    for (const backend of BACKENDS) {
      const omit = backend === displaced ? new Set(BROAD_MATRIX_CASES) : new Set();
      for (const item of operationalCases(backend)) if (!omit.has(item.slug)) expectedCases.push(item.id);
    }
  } else for (const backend of BACKENDS) for (const item of operationalCases(backend)) expectedCases.push(item.id);
  const expectedSet = new Set(expectedCases), actualSet = new Set(executedCases);
  const missing = expectedCases.filter(id => !actualSet.has(id));
  const unexpected = executedCases.filter(id => !expectedSet.has(id));
  const countsBy = list => list.reduce((counts, id) => { const kind = byCase.get(id)?.kind; return kind ? { ...counts, [kind]: (counts[kind] ?? 0) + 1 } : counts; }, { 'full-av': 0, 'actual-artifact-probe': 0, 'source-cancel': 0 });
  const counts = countsBy(executedCases);
  const expectedCounts = selected ? { 'full-av': 37, 'actual-artifact-probe': 4, 'source-cancel': 2 } : { 'full-av': 44, 'actual-artifact-probe': 4, 'source-cancel': 2 };
  const countMismatches = [];
  for (const kind of Object.keys(expectedCounts)) if ((counts[kind] ?? 0) !== expectedCounts[kind]) countMismatches.push(`${kind}: ${counts[kind] ?? 0} != ${expectedCounts[kind]}`);
  const rows = executionRows ?? [...receipts.values()].flatMap(receipt => {
    if (receipt.kind !== 'full-av') return [];
    return receipt.evidence.jobIds.map(jobId => {
      const item = byCase.get(receipt.caseId);
      return { caseId: receipt.caseId, backend: receipt.backend, jobId, sourceSha: receipt.sourceSha ?? null, modelKey: item?.modelKey ?? null };
    });
  });
  const executions = summarizeUniqueExecutions(rows);
  const unproven = [...built.unproven];
  for (const id of missing) unproven.push({ cellId: id, reason: `required operational case was not actually executed: ${id}` });
  if (unexpected.length) unproven.push({ cellId: unexpected.join(','), reason: `unexpected/fake real rows on an asymmetric matrix: ${unexpected.join(', ')}` });
  if (countMismatches.length) unproven.push({ cellId: 'execution-counts', reason: `conditional counts do not match: ${countMismatches.join('; ')}` });
  if (executions.conflicts.length) unproven.push({ cellId: 'unique-executions', reason: `conflicting duplicate job identity: ${executions.conflicts.map(conflict => conflict.jobId).join(', ')}` });
  const executionInventoryComplete=missing.length===0&&unexpected.length===0&&countMismatches.length===0&&executions.conflicts.length===0;
  const omissionPrerequisitesComplete=built.omittedCoverage.every(value=>value.ok);
  const requiredReducedComplete=executionInventoryComplete&&omissionPrerequisitesComplete;
  const catalogComplete=built.unproven.length===0;
  const defaultLimitations = [
    selected?'Only the measured-cheaper matrix backend runs the seven broad causal cases; the displaced backend exercises the encoded surface via its own cheap witness + case8 + peer matrix receipts, but omitted end-to-end AV contrasts remain explicitly unproven on that backend.':'Both backends run the maximum reduced inventory; no asymmetric matrix omission is credited.',
    'Hardware/driver/runner combinations beyond the recorded inline anchors are not exhausted.',
    'Production combinations beyond the 25-case operational inventory remain limited; no metrics or peaks are inferred from counts.',
  ];
  return {
    executionInventoryComplete, requiredReducedComplete, catalogComplete, catalogUnproven:built.unproven, ok:requiredReducedComplete, status:requiredReducedComplete?(catalogComplete?'reduced-complete':'reduced-complete-with-catalog-disclosures'):'required-reduced-evidence-incomplete',
    inventory, coverage: built.coverage, omittedCoverage: built.omittedCoverage, unproven,
    counts: { ...counts, total: executedCases.length }, expectedCounts: { ...expectedCounts, total: expectedCounts['full-av'] + expectedCounts['actual-artifact-probe'] + expectedCounts['source-cancel'] },
    missing, unexpected, matrixDecision: matrixDecision ?? null, executions,
    limitations: [...defaultLimitations, ...limitations],
  };
}

// These are encoded native DiT seams, never raw TE/VAE or end-to-end AV evidence.
export const CONDITIONING_WITNESS_SELECTOR='pruned_adaln::manual_unit_patch_and_mixed_reference_full_forward_matches_resident_control';
export const ADAPTER_WITNESS_SELECTOR='adapter_topology::a_genuine_lokr_installs_on_the_pruned_curve_with_the_same_delta';
export const CONDITIONING_WITNESS_PROPERTIES=Object.freeze([
  'context_values','context_vs_zero_same_shape','first_only_values','first_only_vs_zero_same_shape','last_only_values','first_vs_last_placement','first_only_vs_drop','last_only_vs_drop','first_last_swap_values','first_last_drop_first','first_last_drop_last','reference_image_values','fl2va_vs_ref2va_positioning','ordered_reference_values','reference_video_only_values','reference_audio_only_CONDITIONING_ROWS','reference_audio_values_visual_fixed','hybrid_checkpoint_FL_portion_vs_wrong_REF_portion','hybrid_checkpoint_REF_portion_vs_wrong_FL_portion',
  'reference_image_drop_same_layout','reference_video_drop_same_layout','reference_audio_only_drop_same_layout_CONDITIONING_ROWS','ordered_video_references_AB_vs_BA','ordered_audio_references_AB_vs_BA_visual_fixed','mixed_image_video_soundtrack_audio_reference_order','mixed_image_drop_same_layout','mixed_video_carrier_drop_same_layout','mixed_standalone_audio_drop_same_layout','mixed_video_soundtrack_drop_same_layout',
]);
export function validateNativeWitnessMarkers(log,{backend,testSelectors,adapters=false}={}){
  assert.ok(BACKENDS.includes(backend));assert.ok(testSelectors?.includes(CONDITIONING_WITNESS_SELECTOR),'passed conditioning selector required');
  const nativeBackend=backend==='mlx'?'metal':'cuda',rows=[];
  for(const line of log.split(/\r?\n/)){
    const m=/\b(H3_CONDITIONING_CAUSAL|H3_ADAPTER_STACK_CAUSAL)\s+(.+)/.exec(line);if(!m)continue;
    const fields=Object.fromEntries([...m[2].matchAll(/([a-z_]+)=([^\s]+)/g)].map(v=>[v[1],v[2]]));if(fields.backend!==nativeBackend)continue;
    rows.push({marker:m[1],...fields});
  }
  const finite=(value,label)=>{const n=Number(value);assert.ok(value!==undefined&&Number.isFinite(n),`missing finite ${label}`);return n;};
  const effect=row=>{const delta=finite(row.max_abs,'max_abs'),floor=finite(row.floor,'floor');assert.ok(floor>=0&&delta>floor,'causal effect must exceed its actual floor');};
  for(const property of CONDITIONING_WITNESS_PROPERTIES){const matches=rows.filter(row=>row.marker==='H3_CONDITIONING_CAUSAL'&&row.property===property);assert.equal(matches.length,1,`missing or duplicate executed native witness: ${property}`);assert.equal(matches[0].generated_only,'true');effect(matches[0]);}
  if(adapters){assert.ok(testSelectors.includes(ADAPTER_WITNESS_SELECTOR),'passed adapter selector required');for(const topology of ['full','pruned']){
    for(const property of ['lora_vs_base','lokr_vs_base','stack_vs_base','stack_drop_lora','stack_drop_lokr']){const matches=rows.filter(row=>row.marker==='H3_ADAPTER_STACK_CAUSAL'&&row.topology===topology&&row.property===property);assert.equal(matches.length,1,`missing adapter witness ${topology}/${property}`);assert.equal(matches[0].seam,'adaptable_projection');effect(matches[0]);}
    const matches=rows.filter(row=>row.marker==='H3_ADAPTER_STACK_CAUSAL'&&row.topology===topology&&row.property==='component_differentials');assert.equal(matches.length,1);const bound=finite(matches[0].bound,'bound');assert.ok(bound>=0);for(const key of ['additive_max_abs','order_max_abs']){const value=finite(matches[0][key],key);assert.ok(value>=0&&value<=bound,'adapter component differential exceeds bound');}
  }}
  return {backend,conditioning:CONDITIONING_WITNESS_PROPERTIES.map(property=>({property,status:'passed',scope:'encoded-generated-rows-only'})),adapterStack:adapters,rawPreprocessing:false,endToEndAV:false,omittedAVContrasts:'NOT proven by encoded witnesses'};
}

// Convenience helper for the parent wiring.
export function catalogCoverageCounts(coverage) { return coverage.reduce((counts, entry) => ({ ...counts, [entry.category]: (counts[entry.category] ?? 0) + 1 }), { 'real-weight': 0, 'executed-proof': 0, unproven: 0 }); }
