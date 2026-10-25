// Pure in-process data/validation tests. No downloads, network, models, ports, GPU or Rust runs.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BACKENDS, BOTH_BACKEND_CAUSAL_CASE, BROAD_MATRIX_CASES, CATALOG_CELLS, CATALOG_SUFFIXES,
  MAX_OPERATIONAL_EXECUTIONS, OPERATIONAL_CASE_DEFS, REAL_CELL_MAP, buildCatalogCoverage,
  catalogCoverageCounts, chooseMatrixBackend, defaultCatalog, operationalCases, summarizeUniqueExecutions,
  validateClosure, validateOperationalCaseInventory,
} from './h3-finetune-terminal-reduced.mjs';
import { requiredCases } from './h3-finetune-terminal.mjs';

const sha40 = value => value.repeat(40);
const sha64 = value => value.repeat(64);
const anchorsFor = ({ mlxReference = 100, mlxPrompt = 50, cudaReference = 120, cudaPrompt = 60, override = {} } = {}) => {
  const base = { recipe: 'same-recipe', geometry: { width: 576, height: 320 }, sourceSha: sha40('a'), pinSha: sha40('b'), hardware: 'same-hardware-label' };
  const row = (backend, role, wallMs, hardware = `${backend}-hardware`) => ({ backend, role, ...base, ...override, wallMs, hardware });
  return { mlx: [row('mlx', 'reference', mlxReference), row('mlx', 'prompt', mlxPrompt)], cuda: [row('cuda', 'reference', cudaReference), row('cuda', 'prompt', cudaPrompt)] };
};
const receiptFor = item => ({
  caseId: item.id, backend: item.backend, status: 'passed', runRef: `actual-run:${item.id}`, sourceSha: sha40('c'),
  evidence: item.kind === 'full-av' ? { jobIds: [`job:${item.id}`] } : item.kind === 'actual-artifact-probe' ? { artifactSha256: sha64('d') } : { cancelReceipt: `cancel:${item.id}` },
});
const CELL_FOR_SLUG = Object.fromEntries(Object.entries(REAL_CELL_MAP).flatMap(([suffix, slugs]) => slugs.map(slug => [slug, suffix])));
function fullPlan({ selected = 'mlx', asymmetric = true } = {}) {
  const matrixDecision = asymmetric ? chooseMatrixBackend({ anchors: anchorsFor() }) : undefined;
  assert.equal(asymmetric, Boolean(matrixDecision?.status === 'selected'));
  const executed = operationalCases().filter(item => !(asymmetric && item.backend === matrixDecision.displacedBackend && BROAD_MATRIX_CASES.includes(item.slug)));
  const realReceipts = executed.map(receiptFor);
  const proofIndex = [];
  const realSuffixes = new Set(Object.keys(REAL_CELL_MAP));
  for (const cell of defaultCatalog()) {
    if (realSuffixes.has(cell.suffix)) continue;
    proofIndex.push({ id: `proof:${cell.id}`, status: 'passed', catalogIds: [cell.id], sourceSha: sha40('e'), config: { kind: 'source-proof', cell: cell.suffix }, runRef: `proof-run:${cell.id}`, testSelectors: [`selector:${cell.suffix}`] });
  }
  if (asymmetric) for (const slug of BROAD_MATRIX_CASES) {
    const backend = matrixDecision.displacedBackend, cellId = `${backend}.${CELL_FOR_SLUG[slug]}`;
    proofIndex.push({ id: `witness:${backend}:${slug}`, kind: 'cheap-causal-witness', backend, caseId: `${backend}.reduced.${slug}`, status: 'passed', catalogIds: [cellId], sourceSha: sha40('f'), config: { kind: 'cheap-causal-witness' }, runRef: `witness-run:${backend}:${slug}`, testSelectors: [`witness:${slug}`], guarantees: [slug] });
  }
  return { realReceipts, proofIndex, matrixDecision };
}
const assertEveryCellCategorizedOnce = coverage => {
  assert.equal(coverage.length, CATALOG_CELLS);
  assert.equal(new Set(coverage.map(entry => entry.cellId)).size, CATALOG_CELLS);
  for (const entry of coverage) assert.ok(['real-weight', 'executed-proof', 'unproven'].includes(entry.category));
  const counts = catalogCoverageCounts(coverage);
  assert.equal(counts['real-weight'] + counts['executed-proof'] + counts.unproven, CATALOG_CELLS);
};

test('inventory matches the parent 84-cell catalog and the 25/backend operational contract', () => {
  assert.deepEqual(defaultCatalog().map(cell => cell.id).sort(), requiredCases().map(cell => cell.id).sort());
  assert.equal(defaultCatalog().length, 84);
  assert.equal(OPERATIONAL_CASE_DEFS.length, 25);
  assert.equal(operationalCases().length, MAX_OPERATIONAL_EXECUTIONS);
  for (const backend of BACKENDS) {
    const cases = operationalCases(backend);
    assert.equal(cases.length, 25);
    const counts = cases.reduce((value, item) => ({ ...value, [item.kind]: (value[item.kind] ?? 0) + 1 }), {});
    assert.deepEqual(counts, { 'full-av': 22, 'actual-artifact-probe': 2, 'source-cancel': 1 });
    assert.equal(new Set(cases.map(item => item.id)).size, 25);
    assert.ok(cases.every(item => item.acceptance.ac.length > 0 && item.acceptance.e.length > 0));
  }
  const validated = validateOperationalCaseInventory();
  assert.equal(validated.ok, true);
  assert.deepEqual(validated.backends, { mlx: 25, cuda: 25 });
  assert.ok(BROAD_MATRIX_CASES.every(slug => OPERATIONAL_CASE_DEFS.some(item => item.slug === slug)));
  assert.ok(OPERATIONAL_CASE_DEFS.some(item => item.slug === BOTH_BACKEND_CAUSAL_CASE));
  for (const mutate of [
    cases => cases.pop(),
    cases => { cases[0] = { ...cases[0], kind: 'unknown' }; },
    cases => { cases[1] = { ...cases[1], baseline: `${cases[1].backend}.reduced.not-a-case` }; },
    cases => { cases[0] = { ...cases[0], id: 'mlx.unstable.id' }; },
    cases => { cases[0] = { ...cases[0], acceptance: { ac: [], e: [] } }; },
  ]) { const cases = operationalCases(); mutate(cases); assert.throws(() => validateOperationalCaseInventory({ cases })); }
});
test('catalog coverage never counts unexecuted work as passed and lists explicit real case ids', () => {
  const empty = buildCatalogCoverage({ realReceipts: [], proofIndex: [] });
  assertEveryCellCategorizedOnce(empty.coverage);
  assert.deepEqual(catalogCoverageCounts(empty.coverage), { 'real-weight': 0, 'executed-proof': 0, unproven: 84 });
  const item = operationalCases('mlx').find(value => value.slug === 'artifact-0-ordered-mixed-baseline-ref-turbo');
  const covered = buildCatalogCoverage({ realReceipts: [receiptFor(item)], proofIndex: [] });
  const cell = covered.coverage.find(entry => entry.cellId === 'mlx.artifact.0');
  assert.equal(cell.category, 'real-weight'); assert.deepEqual(cell.reducedCaseIds, [item.id]);
  assert.equal(covered.coverage.filter(entry => entry.category === 'unproven').length, 83);
});
test('executed proof must be explicit, passed and complete; otherwise the cell stays unproven', () => {
  const cellId = 'mlx.generalization.sharded-diffusers', proof = { id: 'proof:shards', status: 'passed', catalogIds: [cellId], sourceSha: sha40('a'), config: { kind: 'source-assertions' }, runRef: 'provided-run-ref', testSelectors: ['selector:shards'] };
  const covered = buildCatalogCoverage({ proofIndex: [proof] });
  const cell = covered.coverage.find(entry => entry.cellId === cellId);
  assert.equal(cell.category, 'executed-proof');
  assert.deepEqual(cell.proofs[0], { id: 'proof:shards', sourceSha: sha40('a'), config: { kind: 'source-assertions' }, status: 'passed', testSelectors: ['selector:shards'], runRef: 'provided-run-ref' });
  for (const mutate of [value => { value.status = 'failed'; }, value => { delete value.sourceSha; }, value => { value.testSelectors = []; }, value => { value.runRef = ''; }, value => { value.catalogIds = ['mlx.unknown.cell']; }]) {
    const bad = structuredClone(proof); mutate(bad); assert.throws(() => buildCatalogCoverage({ proofIndex: [bad] }));
  }
  const unproven = buildCatalogCoverage({ proofIndex: [] }).coverage.find(entry => entry.cellId === cellId);
  assert.equal(unproven.category, 'unproven'); assert.match(unproven.reason, /no real-weight case and no explicit executed-proof/);
});
test('matrix backend is chosen only from matching finite anchors and ties/pending stay unresolved', () => {
  const selected = chooseMatrixBackend({ anchors: anchorsFor() });
  assert.equal(selected.status, 'selected'); assert.equal(selected.selectedBackend, 'mlx'); assert.equal(selected.displacedBackend, 'cuda');
  assert.equal(selected.numbers.mlx.weightedCost, 600); assert.equal(selected.numbers.cuda.weightedCost, 720);
  assert.match(selected.reason, /lower measured incremental cost/);
  assert.equal(selected.anchorIds.reference.mlx, 'mlx.reduced.artifact-0-ordered-mixed-baseline-ref-turbo');
  assert.equal(selected.anchorIds.prompt.cuda, 'cuda.reduced.real-hybrid-text');
  const tie = chooseMatrixBackend({ anchors: anchorsFor({ cudaReference: 100, cudaPrompt: 50 }) });
  assert.equal(tie.status, 'unresolved'); assert.equal(tie.selectedBackend, undefined);
  const pending = chooseMatrixBackend({ anchors: { mlx: anchorsFor().mlx, cuda: anchorsFor().cuda.map(row => row.role === 'prompt' ? { ...row, wallMs: null } : row) } });
  assert.equal(pending.status, 'unresolved'); assert.match(pending.reason, /pending inline anchors/); assert.equal(pending.selectedBackend, undefined);
  const deleted = anchorsFor(); delete deleted.cuda[1].wallMs;
  assert.equal(chooseMatrixBackend({ anchors: deleted }).status, 'unresolved');
  for (const mutate of [value => { value.mlx[0].wallMs = 0; }, value => { value.mlx[0].wallMs = -5; }, value => { value.cuda[1].wallMs = Number.NaN; }, value => { delete value.cuda[0].recipe; }]) {
    const bad = structuredClone(anchorsFor()); mutate(bad); assert.throws(() => chooseMatrixBackend({ anchors: bad }));
  }
  const mismatch = anchorsFor(); mismatch.cuda[0].pinSha = sha40('9');
  assert.throws(() => chooseMatrixBackend({ anchors: mismatch }), /differs across backends/);
  const hardwareSplit = anchorsFor(); hardwareSplit.cuda[1].hardware = 'other-hardware';
  assert.throws(() => chooseMatrixBackend({ anchors: hardwareSplit }), /share the hardware/);
});
test('shared actual job ids count once and conflicting duplicate identities fail', () => {
  const summary = summarizeUniqueExecutions([
    { caseId: 'mlx.reduced.real-mixed-repeat', backend: 'mlx', jobId: 'job-shared', sourceSha: sha40('a') },
    { caseId: 'mlx.reduced.real-drop-images', backend: 'mlx', jobId: 'job-shared', sourceSha: sha40('a') },
    { caseId: 'mlx.reduced.real-drop-audio', backend: 'mlx', jobId: 'job-own', sourceSha: sha40('a') },
  ]);
  assert.equal(summary.uniqueExecutions, 2);
  assert.deepEqual(summary.shared, [{ jobId: 'job-shared', caseIds: ['mlx.reduced.real-mixed-repeat', 'mlx.reduced.real-drop-images'] }]);
  assert.equal(summary.conflicts.length, 0);
  const conflict = summarizeUniqueExecutions([
    { caseId: 'mlx.reduced.real-mixed-repeat', backend: 'mlx', jobId: 'job-shared', sourceSha: sha40('a') },
    { caseId: 'cuda.reduced.real-mixed-repeat', backend: 'cuda', jobId: 'job-shared', sourceSha: sha40('a') },
  ]);
  assert.equal(conflict.conflicts.length, 1); assert.equal(conflict.uniqueExecutions, 1);
  assert.throws(() => summarizeUniqueExecutions([{ caseId: 3, backend: 'mlx' }]));
});
test('conditional matrix accounts for 37 AV + 4 probes + 2 cancel without claiming omitted AV contrasts', () => {
  const plan = fullPlan();
  const closure = validateClosure(plan);
  assert.equal(closure.executionInventoryComplete,true);
  assert.equal(closure.ok,true);assert.equal(closure.requiredReducedComplete,true);assert.equal(closure.catalogComplete,false);
  assert.equal(closure.status,'reduced-complete-with-catalog-disclosures');
  assert.deepEqual(closure.counts, { 'full-av': 37, 'actual-artifact-probe': 4, 'source-cancel': 2, total: 43 });
  assert.deepEqual(closure.expectedCounts, { 'full-av': 37, 'actual-artifact-probe': 4, 'source-cancel': 2, total: 43 });
  assertEveryCellCategorizedOnce(closure.coverage);
  assert.ok(closure.unproven.length>0);
  assert.ok(closure.coverage.some(entry=>entry.omittedCaseIds.length&&entry.category==='unproven'&&/NOT proven/.test(entry.reason)));
  assert.equal(closure.omittedCoverage.length, BROAD_MATRIX_CASES.length);
  assert.ok(closure.omittedCoverage.every(triad => triad.ok));
  assert.ok(closure.limitations.some(value => /displaced backend/.test(value)));
  assert.equal(closure.executions.conflicts.length, 0);
});
test('a missing required real case is unproven and blocks closure', () => {
  const plan = fullPlan(), dropped = plan.realReceipts.find(receipt => receipt.caseId.endsWith('real-hybrid-image'));
  plan.realReceipts = plan.realReceipts.filter(receipt => receipt !== dropped);
  const closure = validateClosure(plan);
  assert.equal(closure.ok, false);
  assert.ok(closure.missing.includes(dropped.caseId));
  assert.ok(closure.coverage.some(entry => entry.cellId === dropped.backend + '.capability.image' && entry.category === 'unproven'));
});
test('omitted broad cases need all three witnesses or the cell and closure stay unproven', () => {
  for (const drop of ['witness', 'ownMixed', 'peer']) {
    const plan = fullPlan(), displaced = plan.matrixDecision.displacedBackend, selected = plan.matrixDecision.selectedBackend;
    if (drop === 'witness') plan.proofIndex = plan.proofIndex.filter(proof => proof.kind !== 'cheap-causal-witness');
    if (drop === 'ownMixed') plan.realReceipts = plan.realReceipts.filter(receipt => receipt.caseId !== `${displaced}.reduced.${BOTH_BACKEND_CAUSAL_CASE}`);
    if (drop === 'peer') plan.realReceipts = plan.realReceipts.filter(receipt => receipt.caseId !== `${selected}.reduced.${BROAD_MATRIX_CASES[0]}`);
    const closure = validateClosure(plan);
    assert.equal(closure.ok, false, drop);
    assert.ok(closure.unproven.some(entry => /cheap witness|own case8|peer matrix/.test(entry.reason) || closure.missing.length > 0), drop);
    if (drop === 'witness') assert.equal(plan.matrixDecision.status, 'selected');
  }
});
test('a fake passed real row on the displaced backend is unexpected and never accepted', () => {
  const plan = fullPlan(), fakeCase = `${plan.matrixDecision.displacedBackend}.reduced.${BROAD_MATRIX_CASES[0]}`;
  plan.realReceipts = [...plan.realReceipts, { caseId: fakeCase, backend: plan.matrixDecision.displacedBackend, status: 'passed', runRef: 'fake-run', sourceSha: sha40('a'), evidence: { jobIds: ['fake-job'] } }];
  const closure = validateClosure(plan);
  assert.equal(closure.ok, false);
  assert.ok(closure.unexpected.includes(fakeCase));
  assert.ok(closure.unproven.some(entry => /unexpected\/fake real rows/.test(entry.reason)));
});
test('without a selected matrix both backends must execute all 25 (50), not the conditional 43', () => {
  const conditional = fullPlan({ asymmetric: false });
  conditional.realReceipts = conditional.realReceipts.filter(receipt => !receipt.caseId.endsWith(`reduced.${BROAD_MATRIX_CASES[0]}`) || receipt.backend !== 'cuda');
  const failed = validateClosure({ ...conditional, matrixDecision: undefined });
  assert.equal(failed.ok, false);
  assert.ok(failed.unproven.some(entry => /required operational case was not actually executed/.test(entry.reason)));
});
test('native encoded markers require all passed causal discriminators, not counts',async()=>{
  const {CONDITIONING_WITNESS_PROPERTIES:properties,CONDITIONING_WITNESS_SELECTOR:selector,validateNativeWitnessMarkers:validate}=await import('./h3-finetune-terminal-reduced.mjs');
  const lines=properties.map(property=>`H3_CONDITIONING_CAUSAL backend=metal property=${property} max_abs=1e-2 floor=1e-5 generated_only=true`);
  const proof=validate(lines.join('\n'),{backend:'mlx',testSelectors:[selector]});assert.equal(proof.conditioning.length,29);assert.equal(proof.rawPreprocessing,false);assert.equal(proof.endToEndAV,false);
  assert.throws(()=>validate(lines.slice(1).join('\n'),{backend:'mlx',testSelectors:[selector]}),/missing or duplicate/);
  assert.throws(()=>validate(lines.join('\n'),{backend:'mlx',testSelectors:[]}),/selector/);
  assert.throws(()=>validate(lines.join('\n').replace('max_abs=1e-2','max_abs=0'),{backend:'mlx',testSelectors:[selector]}),/exceed/);
});
test('required reduced completeness does not gate on optional catalog proof completeness',()=>{
  const complete=fullPlan({asymmetric:false});complete.proofIndex=[];const result=validateClosure(complete);
  assert.equal(result.ok,true);assert.equal(result.requiredReducedComplete,true);assert.equal(result.catalogComplete,false);assert.ok(result.catalogUnproven.length>0);
  complete.realReceipts.pop();const missing=validateClosure(complete);assert.equal(missing.ok,false);assert.equal(missing.requiredReducedComplete,false);
});
test('cold cancellation is not a bounded loaded peak and W4 projection is not full mapping',()=>{
  assert.equal(REAL_CELL_MAP['lifecycle.bounded-loading'],undefined);assert.equal(REAL_CELL_MAP['numeric.tensor-mapping'],undefined);
  const mapping=buildCatalogCoverage({realReceipts:operationalCases().map(receiptFor)});
  for(const backend of ['mlx','cuda'])for(const suffix of ['numeric.tensor-mapping','lifecycle.bounded-loading'])assert.equal(mapping.coverage.find(c=>c.cellId===`${backend}.${suffix}`).category,'unproven');
});
test('all-25-on-both-backends closure proves the 44 AV + 4 probe + 2 cancel maximum', () => {
  const plan = fullPlan({ asymmetric: false });
  const closure = validateClosure(plan);
  assert.equal(closure.ok, true, JSON.stringify(closure.unproven));
  assert.deepEqual(closure.counts, { 'full-av': 44, 'actual-artifact-probe': 4, 'source-cancel': 2, total: 50 });
  assert.equal(closure.matrixDecision, null);
  assertEveryCellCategorizedOnce(closure.coverage);
  assert.ok(closure.coverage.every(entry => entry.category !== 'unproven'));
});
