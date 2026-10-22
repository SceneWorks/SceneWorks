// DECLARED SYNTHETIC CPU fixtures only: no sealed binary, weights, device, provider or campaign permit.
import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
import { loadProductTargets, terminalPlan, usesPersonalInputs } from './flux3-image-klein-terminal-plan.mjs';
import { validateApprovedPlan, validateInputSpecAddendum, boundInputSpecSha256, initializeLedger, reserveInvocation, auditLedger, aggregateLedgerReceipts, verifyLedgerReceipt, planHash, admittedHost } from './flux3-image-klein-terminal-budget.mjs';
import { validateNativeBoundaryReceipt, nativeBoundaryReceiptSha256, buildSupplementalResourceCase, NATIVE_BOUNDARY_RECEIPT_KIND } from './lib/flux3-native-boundary.mjs';
import { selfTestEnvironment, runNativeBoundarySelfTest, productionResourceRequest } from './flux3-native-boundary-self-test.mjs';
import { launchFlux3Resource } from './flux3-resource-launcher.mjs';

const sources = await loadProductTargets();
const d = c => c.repeat(64), source = 'a'.repeat(40), inference = 'b'.repeat(40);
const candleBinary = d('1'), candleReceipt = d('2'), mlxBinary = d('3'), mlxReceipt = d('4');
const pupInventory = ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'preprocessor_config.json']
  .map((p, i) => ({ path: p, bytes: i + 1, sha256: d(String(i % 10)) }));
const binding = backend => buildSupplementalResourceCase({ backend, prompt: 'The same subject, unchanged identity, outdoors',
  references: Array.from({ length: 10 }, (_, i) => ({ path: `references/ref-${i}.png`, sha256: d(String(i)) })),
  promptUpsampler: { root: 'prompt_upsampler', repository: 'black-forest-labs/flux-3-image-klein-pup', resolvedRevision: '8b069751eb5dcab23829d9c5eccb470eb8b8a4ed', inventory: pupInventory } });
const resourceBindings = { candle: binding('candle'), mlx: binding('mlx') };
// A synthetic receipt shaped exactly like the sealed adapter's self-test output.
function receiptFor(plan, backend, binary, buildReceipt, mutate = () => {}) {
  const receipt = { schemaVersion: 1, kind: NATIVE_BOUNDARY_RECEIPT_KIND, mode: 'DECLARED SYNTHETIC', lane: backend, binarySha256: binary, buildReceiptSha256: buildReceipt,
    sourceSha: source, inferenceMainSha: inference, campaignPermitsConsumed: 0,
    cases: plan.generationInvocations.filter(i => i.backend === backend && i.kind.startsWith('resource_')).map(i => ({ invocationId: i.invocationId, caseId: i.caseId, kind: i.kind,
      requestSha256: i.requestSha256, valid: { reserveCalls: 1, generateCalls: 1, order: ['reserve', 'generate'] },
      reused: { refused: true, generateCalls: 0 }, missing: { refused: true, generateCalls: 0 }, mismatchedDigest: { refused: true, generateCalls: 0 }, mismatchedCase: { refused: true, generateCalls: 0 } })) };
  mutate(receipt); return receipt;
}
function readyPlan() {
  const unproved = terminalPlan(sources, { resourceBindings });
  const receipt = receiptFor(unproved, 'candle', candleBinary, candleReceipt);
  return { plan: terminalPlan(sources, { resourceBindings, addendumBackends: ['mlx'], nativeBoundary: { candle: { receipt, parentNativeBoundaryVerificationId: 'DECLARED-SYNTHETIC-candle-verification', binarySha256: candleBinary, buildReceiptSha256: candleReceipt, sourceSha: source, inferenceMainSha: inference } } }), candleSelfTest: receipt };
}
async function approvalFixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'flux3-native-boundary-DECLARED-SYNTHETIC-')); t.after(() => rm(root, { recursive: true, force: true }));
  const { plan } = readyPlan();
  const approval = { schemaVersion: 1, kind: 'flux3-terminal-plan-approval', approvalId: 'DECLARED-SYNTHETIC', userPlanApprovalId: 'NOT_USER_AUTHORIZATION', approvedBy: 'CPU fixture',
    sourceSha: source, inferenceMainSha: inference, ledgerId: 'declared-boundary-ledger', ledgerRoots: { candle: path.join(root, 'candle'), mlx: path.join(root, 'mlx') },
    inputSpecSha256: { candle: d('c') }, deferredInputSpecBackends: ['mlx'], resourceBuildReceipts: { candle: { receiptSha256: candleReceipt, binarySha256: candleBinary } }, plan };
  const addendum = { schemaVersion: 1, kind: 'flux3-terminal-plan-input-spec-addendum', approvedPlanSha256: planHash(approval), ledgerId: approval.ledgerId, backend: 'mlx',
    inputSpecSha256: d('e'), buildReceipt: { receiptSha256: d('5'), binarySha256: d('6') }, resourceBuildReceipt: { receiptSha256: mlxReceipt, binarySha256: mlxBinary },
    nativeBoundaryReceipt: receiptFor(plan, 'mlx', mlxBinary, mlxReceipt), parentNativeBoundaryVerificationId: 'DECLARED-SYNTHETIC-mlx-verification', addendumId: 'DECLARED-SYNTHETIC-addendum', userApprovalId: 'NOT_USER_AUTHORIZATION', approvedBy: 'CPU fixture' };
  return { root, approval, addendum };
}

test('the planner fixes every resource request digest and is READY only with a matching sealed-binary receipt', () => {
  const blocked = terminalPlan(sources);
  assert(blocked.resourceCases.every(r => r.status === 'BLOCKED'), 'shipped default stays BLOCKED');
  assert(blocked.generationInvocations.filter(i => i.kind === 'resource_anchor').every(i => /^[a-f0-9]{64}$/.test(i.requestSha256)));
  const { plan } = readyPlan();
  assert(plan.resourceCases.filter(r => r.backend === 'candle').every(r => r.status === 'READY' && r.nativeBoundaryBinarySha256 === candleBinary));
  assert(plan.resourceCases.filter(r => r.backend === 'mlx').every(r => r.status === 'READY_ON_ADDENDUM'));
  assert.equal(plan.generationInvocations.length, 38, 'counts unchanged');
  const unproved = terminalPlan(sources, { resourceBindings });
  for (const [label, mutate, binary] of [
    ['wrong binary', () => {}, d('9')],
    ['wrong digest', r => { r.cases[0].requestSha256 = d('0'); }, candleBinary],
    ['two generates', r => { r.cases[0].valid.generateCalls = 2; }, candleBinary],
    ['generate before reserve', r => { r.cases[0].valid.order = ['generate', 'reserve']; }, candleBinary],
    ['reused permit generated', r => { r.cases[0].reused = { refused: true, generateCalls: 1 }; }, candleBinary],
    ['missing case', r => { r.cases.pop(); }, candleBinary],
    ['spent campaign permits', r => { r.campaignPermitsConsumed = 1; }, candleBinary],
  ]) {
    const receipt = receiptFor(unproved, 'candle', candleBinary, candleReceipt, mutate);
    assert.throws(() => terminalPlan(sources, { resourceBindings, nativeBoundary: { candle: { receipt, parentNativeBoundaryVerificationId: 'x', binarySha256: binary, buildReceiptSha256: candleReceipt, sourceSha: source, inferenceMainSha: inference } } }), /BLOCKED: native-boundary receipt/, label);
  }
});
test('approval refuses a READY row proved by a binary other than the approved sealed resource build', async t => {
  const { approval } = await approvalFixture(t); validateApprovedPlan(approval);
  const changed = structuredClone(approval); changed.resourceBuildReceipts.candle.binarySha256 = d('8');
  assert.throws(() => validateApprovedPlan(changed), /binary other than the approved sealed resource build/);
  const readyMlx = structuredClone(approval); readyMlx.plan.resourceCases.find(r => r.backend === 'mlx').status = 'READY';
  assert.throws(() => validateApprovedPlan(readyMlx), /deferred backend is READY only by addendum|READY requires/);
});
test('the addendum binds only the deferred backend and can never change the plan, the bound spec, the ledger or counts', async t => {
  const { approval, addendum } = await approvalFixture(t);
  assert.equal(validateInputSpecAddendum(approval, addendum).inputSpecSha256, addendum.inputSpecSha256);
  assert.equal(boundInputSpecSha256(approval, 'candle'), approval.inputSpecSha256.candle);
  assert.throws(() => boundInputSpecSha256(approval, 'candle', addendum), /already bound by the approval/);
  for (const [label, mutate, pattern] of [
    ['wrong plan', a => { a.approvedPlanSha256 = d('f'); }, /different approved plan/],
    ['wrong ledger', a => { a.ledgerId = 'another-ledger'; }, /ledger identity/],
    ['bound backend', a => { a.backend = 'candle'; }, /only a deferred backend/],
    ['plan smuggled', a => { a.plan = approval.plan; }, /exactly its binding fields/],
    ['counts smuggled', a => { a.budget = { totalGenerationInvocations: 48 }; }, /exactly its binding fields/],
    ['candle spec smuggled', a => { a.candleInputSpecSha256 = d('9'); }, /exactly its binding fields/],
    ['wrong self-test binary', a => { a.nativeBoundaryReceipt.binarySha256 = d('9'); }, /different binary/],
    ['wrong self-test digest', a => { a.nativeBoundaryReceipt.cases[0].requestSha256 = d('0'); }, /request digest differs/],
    ['self-test for another build', a => { a.nativeBoundaryReceipt.buildReceiptSha256 = d('7'); }, /different sealed build receipt/],
    ['no self-test', a => { a.nativeBoundaryReceipt = null; }, /native-boundary receipt/],
  ]) {
    const changed = structuredClone(addendum); mutate(changed);
    assert.throws(() => validateInputSpecAddendum(approval, changed), pattern, label);
  }
  const withMlxBuild = structuredClone(approval); withMlxBuild.resourceBuildReceipts.mlx = { receiptSha256: d('5'), binarySha256: d('6') };
  assert.throws(() => validateInputSpecAddendum(withMlxBuild, { ...addendum, approvedPlanSha256: planHash(withMlxBuild) }), /cannot replace build bindings/);
});
test('deferred-backend permits, including READY_ON_ADDENDUM resource rows, are never reservable without the addendum', async t => {
  const { approval, addendum } = await approvalFixture(t);
  await initializeLedger({ approval, backend: 'mlx', ledgerRoot: approval.ledgerRoots.mlx });
  const product = approval.plan.generationInvocations.find(i => i.backend === 'mlx' && i.kind === 'product_generation');
  const resource = approval.plan.generationInvocations.find(i => i.backend === 'mlx' && i.kind === 'resource_anchor');
  const reserve = (invocation, inputSpecAddendum) => reserveInvocation({ approval, ledgerRoot: approval.ledgerRoots.mlx, ledgerId: approval.ledgerId, approvedPlanSha256: planHash(approval),
    invocationId: invocation.invocationId, backend: 'mlx', caseId: invocation.caseId, kind: invocation.kind, requestSha256: invocation.requestSha256 ?? d('a'), inputSpecAddendum });
  await assert.rejects(reserve(product), /deferred; its permits stay unusable/);
  await assert.rejects(reserve(resource), /deferred; its permits stay unusable/);
  const wrong = structuredClone(addendum); wrong.nativeBoundaryReceipt.binarySha256 = d('9');
  await assert.rejects(reserve(resource, wrong), /different binary/);
  const event = await reserve(resource, addendum);
  assert.equal(event.invocationId, resource.invocationId, 'only a validated addendum makes the row reservable');
});
test('the self-test is unreachable from production: the launcher never passes the guard and always sets the invocation', async () => {
  const env = selfTestEnvironment({ PATH: 'x', SCENEWORKS_FLUX3_INVOCATION: '{}', SCENEWORKS_FLUX3_BUDGET_NODE: 'node', SCENEWORKS_FLUX3_KLEIN_ROOT: 'r', HF_TOKEN: 'DECLARED-SYNTHETIC-NOT-A-CREDENTIAL', SCENEWORKS_GPU_ID: '1' }, 'inputs');
  assert.deepEqual(Object.keys(env).sort(), ['CUDA_VISIBLE_DEVICES', 'PATH', 'SCENEWORKS_FLUX3_BOUNDARY_SELF_TEST', 'SCENEWORKS_FLUX3_RESOURCE_INPUT_ROOT'].sort());
  assert.equal(env.CUDA_VISIBLE_DEVICES, '-1');
  const launcher = await (await import('node:fs/promises')).readFile(new URL('./flux3-resource-launcher.mjs', import.meta.url), 'utf8');
  assert.doesNotMatch(launcher, /SCENEWORKS_FLUX3_BOUNDARY_SELF_TEST|flux3_native_boundary_self_test/, 'production launcher has no path to the hidden mode');
  assert.match(launcher, /k => !k\.startsWith\('SCENEWORKS_'\)/, 'parent SCENEWORKS_* (including a stray guard) never reaches the production child');
  assert.match(launcher, /env\.SCENEWORKS_FLUX3_INVOCATION=JSON\.stringify/, 'production always sets the invocation the self-test guard refuses');
  assert.equal(typeof launchFlux3Resource, 'function');
});
test('the producer refuses a sealed binary whose hash differs from its build receipt, before running it', async t => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'flux3-self-test-producer-DECLARED-SYNTHETIC-')); t.after(() => rm(root, { recursive: true, force: true }));
  const binary = path.join(root, 'memory-candle-adapter.exe'); await writeFile(binary, 'DECLARED SYNTHETIC NOT A BINARY');
  const build = { kind: 'flux3-owned-build-producer-receipt', backend: 'candle', recipe: { bin: 'memory-candle-adapter' }, sourceSha: source, inferenceMainSha: inference, binary: { sha256: d('9') } };
  const receiptPath = path.join(root, 'producer-receipt.json'), bytes = JSON.stringify(build) + '\n'; await writeFile(receiptPath, bytes);
  const planFile = path.join(root, 'plan.json'); await writeFile(planFile, JSON.stringify(terminalPlan(sources, { resourceBindings })));
  let ran = false;
  await assert.rejects(runNativeBoundarySelfTest({ lane: 'candle', binary, buildReceipt: { path: receiptPath, root, sha256: createHash('sha256').update(bytes).digest('hex') }, sourceSha: source, inferenceMainSha: inference, planFile },
    { run: async () => { ran = true; } }), /binary differs from its sealed build receipt/);
  assert.equal(ran, false, 'no execution of an unapproved binary');
});
test('each self-test case carries the exact production run request (canonical anchor, bound supplemental)', async () => {
  const calibrationPlan = JSON.parse(await (await import('node:fs/promises')).readFile(new URL('../config/memory-calibration-plan.json', import.meta.url), 'utf8'));
  const plan = terminalPlan(sources, { resourceBindings });
  for (const invocation of plan.generationInvocations.filter(i => i.kind.startsWith('resource_'))) {
    const request = productionResourceRequest(calibrationPlan, plan, invocation);
    assert.equal(request.action, 'run'); assert.equal(request.planned.backend, invocation.backend); assert.equal(request.planned.target.tier, invocation.tier);
    assert.equal(Boolean(request.resourceCase), invocation.kind === 'resource_supplemental');
  }
  assert.equal(nativeBoundaryReceiptSha256({ a: 1 }).length, 64);
  assert.throws(() => validateNativeBoundaryReceipt({ kind: 'x' }, {}), /kind\/schema/);
});
test('reservation refuses a wrong ledger identity, a different approved plan or a malformed request digest', async t => {
  const { approval } = await approvalFixture(t);
  await initializeLedger({ approval, backend: 'candle', ledgerRoot: approval.ledgerRoots.candle });
  const product = approval.plan.generationInvocations.find(i => i.backend === 'candle' && i.kind === 'product_generation');
  const base = { approval, ledgerRoot: approval.ledgerRoots.candle, ledgerId: approval.ledgerId, approvedPlanSha256: planHash(approval),
    invocationId: product.invocationId, backend: 'candle', caseId: product.caseId, kind: product.kind, requestSha256: d('a') };
  await assert.rejects(reserveInvocation({ ...base, ledgerId: 'another-ledger' }), /ledger identity differs/);
  await assert.rejects(reserveInvocation({ ...base, approvedPlanSha256: d('f') }), /different approved plan/);
  await assert.rejects(reserveInvocation({ ...base, requestSha256: 'not-a-digest' }), /actual request digest required/);
  assert.equal((await reserveInvocation(base)).invocationId, product.invocationId, 'the correct identity still reserves');
});
test('a deferred ledger is bound to ONE addendum: a different valid addendum is refused after a spend', async t => {
  const { approval, addendum } = await approvalFixture(t);
  await initializeLedger({ approval, backend: 'mlx', ledgerRoot: approval.ledgerRoots.mlx });
  await initializeLedger({ approval, backend: 'candle', ledgerRoot: approval.ledgerRoots.candle });
  const other = { ...structuredClone(addendum), inputSpecSha256: d('7'), addendumId: 'DECLARED-SYNTHETIC-other-addendum' };
  assert.equal(validateInputSpecAddendum(approval, other).inputSpecSha256, d('7'), 'the other addendum is valid on its own');
  const product = approval.plan.generationInvocations.filter(i => i.backend === 'mlx' && i.kind === 'product_generation');
  const reserve = (invocation, inputSpecAddendum) => reserveInvocation({ approval, ledgerRoot: approval.ledgerRoots.mlx, ledgerId: approval.ledgerId, approvedPlanSha256: planHash(approval),
    invocationId: invocation.invocationId, backend: 'mlx', caseId: invocation.caseId, kind: invocation.kind, requestSha256: d('a'), inputSpecAddendum });
  await reserve(product[0], addendum);
  await assert.rejects(reserve(product[1], other), /different addendum than the one recorded/);
  await assert.rejects(auditLedger({ approval, backend: 'mlx', ledgerRoot: approval.ledgerRoots.mlx, inputSpecAddendum: other }), /different addendum than the one recorded/);
  await assert.rejects(auditLedger({ approval, backend: 'mlx', ledgerRoot: approval.ledgerRoots.mlx }), /present that same addendum/);
  const mlx = await auditLedger({ approval, backend: 'mlx', ledgerRoot: approval.ledgerRoots.mlx, inputSpecAddendum: addendum });
  assert.equal(mlx.inputSpecAddendumSha256, planHash(addendum), 'the ledger receipt carries its recorded addendum binding');
  const stripped = { ...mlx }; delete stripped.inputSpecAddendumSha256;
  assert.throws(() => verifyLedgerReceipt(approval, stripped, 'mlx', addendum), /spends but no recorded addendum binding/);
  const candle = await auditLedger({ approval, backend: 'candle', ledgerRoot: approval.ledgerRoots.candle });
  const closure = { kind: 'flux3-terminal-finalized-ledger-heads', coordinatorClosureId: 'DECLARED-SYNTHETIC', producersStopped: true, collectionManifestSha256: d('c'),
    ledgerId: approval.ledgerId, approvedPlanSha256: planHash(approval), backends: Object.fromEntries([mlx, candle].map(r => [r.manifest.backend, { sequence: r.state.sequence, headSha256: r.state.headSha256 }])) };
  assert.throws(() => aggregateLedgerReceipts(approval, [mlx, candle], closure, { mlx: other }), /different addendum than the one recorded/);
  assert.equal(aggregateLedgerReceipts(approval, [mlx, candle], closure, { mlx: addendum }).caseCompletionProven, false);
  await reserve(product[1], addendum); // the SAME addendum keeps working
});
test('a forged READY row (label/digest without its full re-validated receipt) is refused at approval', async t => {
  const { approval } = await approvalFixture(t); validateApprovedPlan(approval);
  for (const [label, mutate, pattern] of [
    ['no carried receipt', a => { delete a.plan.nativeBoundaryReceipts; }, /native-boundary receipt/],
    ['receipt digest swapped', a => { a.plan.resourceCases.find(r => r.backend === 'candle').nativeBoundaryReceiptSha256 = d('5'); }, /READY digest is not the carried receipt/],
    ['receipt tampered', a => { a.plan.nativeBoundaryReceipts.candle.cases[0].valid.generateCalls = 2; }, /one reservation before exactly one generate/],
    ['receipt for other request', a => { a.plan.nativeBoundaryReceipts.candle.cases[0].requestSha256 = d('0'); }, /request digest differs/],
    ['partial readiness', a => { Object.assign(a.plan.resourceCases.find(r => r.backend === 'candle'), { status: 'BLOCKED' }); }, /every resource row of its backend or none/],
    ['other source', a => { a.sourceSha = 'c'.repeat(40); }, /source identity differs/],
  ]) {
    const changed = structuredClone(approval); mutate(changed);
    assert.throws(() => validateApprovedPlan(changed), pattern, label);
  }
});

// --- Host lanes (sc-26243): one backend's permits split across two named hosts, one ledger each. ---
const DATASET_HOST = 'nax-macos', GENERAL_HOST = 'nax-macos-2';
function lanedPlan() {
  const unproved = terminalPlan(sources, { resourceBindings });
  const receipt = receiptFor(unproved, 'candle', candleBinary, candleReceipt);
  return terminalPlan(sources, { resourceBindings, addendumBackends: ['mlx'], hostLanes: { mlx: { datasetHost: DATASET_HOST, generalHost: GENERAL_HOST } },
    nativeBoundary: { candle: { receipt, parentNativeBoundaryVerificationId: 'DECLARED-SYNTHETIC-candle-verification', binarySha256: candleBinary, buildReceiptSha256: candleReceipt, sourceSha: source, inferenceMainSha: inference } } });
}
async function lanedFixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'flux3-host-lanes-DECLARED-SYNTHETIC-')); t.after(() => rm(root, { recursive: true, force: true }));
  const plan = lanedPlan();
  const approval = { schemaVersion: 1, kind: 'flux3-terminal-plan-approval', approvalId: 'DECLARED-SYNTHETIC', userPlanApprovalId: 'NOT_USER_AUTHORIZATION', approvedBy: 'CPU fixture',
    sourceSha: source, inferenceMainSha: inference, ledgerId: 'declared-lanes-ledger',
    ledgerRoots: { candle: path.join(root, 'candle'), [`mlx@${DATASET_HOST}`]: path.join(root, 'mlx-dataset'), [`mlx@${GENERAL_HOST}`]: path.join(root, 'mlx-general') },
    inputSpecSha256: { candle: d('c') }, deferredInputSpecBackends: [`mlx@${DATASET_HOST}`, `mlx@${GENERAL_HOST}`], resourceBuildReceipts: { candle: { receiptSha256: candleReceipt, binarySha256: candleBinary } }, plan };
  const laneResources = host => plan.generationInvocations.filter(i => i.backend === 'mlx' && i.kind.startsWith('resource_') && plan.hostLanes.mlx[host].includes(i.invocationId));
  const addendumFor = (host, binary, build) => ({ schemaVersion: 1, kind: 'flux3-terminal-plan-input-spec-addendum', approvedPlanSha256: planHash(approval), ledgerId: approval.ledgerId, backend: 'mlx', lane: host,
    inputSpecSha256: d(host === DATASET_HOST ? 'e' : 'f'), buildReceipt: { receiptSha256: d('5'), binarySha256: d('6') }, resourceBuildReceipt: { receiptSha256: build, binarySha256: binary },
    nativeBoundaryReceipt: receiptFor({ generationInvocations: laneResources(host) }, 'mlx', binary, build), parentNativeBoundaryVerificationId: `DECLARED-SYNTHETIC-${host}-verification`,
    addendumId: `DECLARED-SYNTHETIC-${host}-addendum`, userApprovalId: 'NOT_USER_AUTHORIZATION', approvedBy: 'CPU fixture' });
  return { root, approval, dataset: addendumFor(DATASET_HOST, mlxBinary, mlxReceipt), general: addendumFor(GENERAL_HOST, d('7'), d('8')) };
}
const laneReserve = (approval, host, invocation, inputSpecAddendum) => reserveInvocation({ approval, ledgerRoot: approval.ledgerRoots[`mlx@${host}`], ledgerId: approval.ledgerId, approvedPlanSha256: planHash(approval),
  invocationId: invocation.invocationId, backend: 'mlx', caseId: invocation.caseId, kind: invocation.kind, requestSha256: invocation.requestSha256 ?? d('a'), inputSpecAddendum, host });

test('the planner splits MLX into disjoint covering host lanes: personal inputs on the dataset host, the rest on the general host', () => {
  const plan = lanedPlan(), lanes = plan.hostLanes.mlx, mlx = plan.generationInvocations.filter(i => i.backend === 'mlx');
  assert.equal(plan.generationInvocations.length, 38, 'counts unchanged'); assert.equal(mlx.length, 17);
  assert.deepEqual(Object.keys(plan.hostLanes), ['mlx'], 'Candle stays one lane');
  const ids = id => mlx.find(i => i.invocationId === id);
  const datasetGenerations = lanes[DATASET_HOST].filter(ids), general = lanes[GENERAL_HOST];
  assert.equal(datasetGenerations.length, 8); assert.equal(general.length, 9);
  assert(datasetGenerations.every(id => usesPersonalInputs(ids(id))) && general.every(id => !usesPersonalInputs(ids(id))), 'every personal-input permit, and only those, is on the dataset host');
  const training = plan.trainingLifecycles.find(t => t.backend === 'mlx');
  assert(lanes[DATASET_HOST].includes(training.startId) && lanes[DATASET_HOST].includes(training.resumeId), 'training runs where the dataset is');
  assert(lanes[DATASET_HOST].includes('mlx.base.pup_edit.bf16.generation-1') && general.filter(id => ids(id).kind === 'resource_anchor').length === 4);
  assert.throws(() => terminalPlan(sources, { hostLanes: { mlx: { datasetHost: 'same', generalHost: 'same' } } }), /two distinct named hosts/);
});
test('approval validates host lanes: disjoint, covering, one ledger root per lane, lifecycle+adapter cases on one host', async t => {
  const { approval } = await lanedFixture(t); validateApprovedPlan(approval);
  const mlxProduct = id => approval.plan.generationInvocations.find(i => i.invocationId === id);
  for (const [label, mutate, pattern] of [
    ['overlap', a => { a.plan.hostLanes.mlx[GENERAL_HOST].push(a.plan.hostLanes.mlx[DATASET_HOST][0]); }, /disjoint/],
    ['uncovered', a => { a.plan.hostLanes.mlx[GENERAL_HOST].pop(); }, /cover exactly/],
    ['foreign id', a => { a.plan.hostLanes.mlx[GENERAL_HOST].push('candle.base.cancel.bf16.generation-1'); }, /disjoint|cover exactly/],
    ['adapter split from its lifecycle', a => { const lanes = a.plan.hostLanes.mlx, id = lanes[DATASET_HOST].find(i => mlxProduct(i)?.adapterBinding); lanes[DATASET_HOST] = lanes[DATASET_HOST].filter(i => i !== id); lanes[GENERAL_HOST].push(id); }, /adapter case runs on the host holding/],
    ['training split', a => { const lanes = a.plan.hostLanes.mlx, tr = a.plan.trainingLifecycles.find(t => t.backend === 'mlx'); lanes[DATASET_HOST] = lanes[DATASET_HOST].filter(i => i !== tr.resumeId); lanes[GENERAL_HOST].push(tr.resumeId); }, /one host/],
    ['backend-keyed mlx root', a => { a.ledgerRoots.mlx = a.ledgerRoots[`mlx@${DATASET_HOST}`]; }, /one fixed ledger root per ledger lane/],
    ['shared lane root', a => { a.ledgerRoots[`mlx@${GENERAL_HOST}`] = a.ledgerRoots[`mlx@${DATASET_HOST}`]; }, /distinct fixed roots/],
    ['unknown deferred lane', a => { a.deferredInputSpecBackends = ['mlx']; }, /known ledger lanes/],
    ['READY on a split backend', a => { a.plan.resourceCases.find(r => r.backend === 'mlx').status = 'READY'; }, /proved per host by addendum/],
  ]) { const changed = structuredClone(approval); mutate(changed); assert.throws(() => validateApprovedPlan(changed), pattern, label); }
});
test('no permit is reservable on the wrong host, and the refusal writes nothing to the ledger', async t => {
  const { approval, dataset, general } = await lanedFixture(t), lanes = approval.plan.hostLanes.mlx;
  for (const host of [DATASET_HOST, GENERAL_HOST]) await initializeLedger({ approval, backend: 'mlx', host, ledgerRoot: approval.ledgerRoots[`mlx@${host}`] });
  const of = host => approval.plan.generationInvocations.filter(i => lanes[host].includes(i.invocationId));
  const generalProduct = of(GENERAL_HOST).find(i => i.kind === 'product_generation'), generalAnchor = of(GENERAL_HOST).find(i => i.kind === 'resource_anchor');
  const datasetProduct = of(DATASET_HOST).find(i => i.kind === 'product_generation');
  // A general-host permit presented on the dataset host's ledger (with that host's own valid addendum).
  await assert.rejects(laneReserve(approval, DATASET_HOST, generalProduct, dataset), /assigned to another mlx host lane.*wrong host/);
  await assert.rejects(laneReserve(approval, DATASET_HOST, generalAnchor, dataset), /wrong host/);
  await assert.rejects(laneReserve(approval, GENERAL_HOST, datasetProduct, general), /wrong host/);
  // An unknown host or the backend-keyed root never resolves to a lane.
  await assert.rejects(reserveInvocation({ approval, ledgerRoot: approval.ledgerRoots[`mlx@${DATASET_HOST}`], ledgerId: approval.ledgerId, approvedPlanSha256: planHash(approval), invocationId: datasetProduct.invocationId, backend: 'mlx', caseId: datasetProduct.caseId, kind: datasetProduct.kind, requestSha256: d('a'), inputSpecAddendum: dataset }), /split by host/);
  await assert.rejects(laneReserve(approval, 'nax-macos-3', datasetProduct, dataset), /not an approved mlx lane/);
  // The other host's addendum never unlocks this host's permits.
  await assert.rejects(laneReserve(approval, DATASET_HOST, datasetProduct, general), /different host lane/);
  for (const host of [DATASET_HOST, GENERAL_HOST]) {
    const receipt = await auditLedger({ approval, backend: 'mlx', host, ledgerRoot: approval.ledgerRoots[`mlx@${host}`] });
    assert.equal(receipt.events.length, 0, 'refusals spent nothing'); assert.equal(receipt.inputSpecAddendumSha256, undefined, 'refusals pinned no addendum');
    assert.deepEqual(receipt.manifest.permitIds, of(host).map(i => i.invocationId), 'each lane ledger allocates only its own permits'); assert.equal(receipt.manifest.host, host);
  }
  // Each lane spends its own permits under its own pinned addendum.
  assert.equal((await laneReserve(approval, DATASET_HOST, datasetProduct, dataset)).invocationId, datasetProduct.invocationId);
  assert.equal((await laneReserve(approval, GENERAL_HOST, generalAnchor, general)).invocationId, generalAnchor.invocationId);
  await assert.rejects(laneReserve(approval, DATASET_HOST, of(DATASET_HOST).find(i => i.kind === 'resource_supplemental'), { ...dataset, addendumId: 'DECLARED-SYNTHETIC-other' }), /different addendum than the one recorded/);
});
test('a host addendum proves exactly its own lane: its receipt covers that lane\'s resource rows only', async t => {
  const { approval, dataset, general } = await lanedFixture(t);
  assert.equal(validateInputSpecAddendum(approval, dataset).ledgerLane, `mlx@${DATASET_HOST}`); assert.equal(validateInputSpecAddendum(approval, general).ledgerLane, `mlx@${GENERAL_HOST}`);
  assert.equal(boundInputSpecSha256(approval, 'mlx', general, GENERAL_HOST), d('f'));
  assert.throws(() => boundInputSpecSha256(approval, 'mlx', general, DATASET_HOST), /different host lane/);
  assert.throws(() => boundInputSpecSha256(approval, 'mlx', undefined, GENERAL_HOST), /mlx@nax-macos-2 input specification is deferred/);
  const allMlx = { ...general, nativeBoundaryReceipt: receiptFor(approval.plan, 'mlx', d('7'), d('8')) };
  assert.throws(() => validateInputSpecAddendum(approval, allMlx), /exactly the plan resource invocations/);
  const { lane, ...noLane } = general; assert.throws(() => validateInputSpecAddendum(approval, noLane), /exactly its binding fields/);
  assert.throws(() => validateInputSpecAddendum(approval, { ...general, lane: 'nax-macos-3' }), /not an approved mlx lane/);
});
test('aggregation spans every lane ledger: candle plus both MLX hosts, never a partial set', async t => {
  const { approval, dataset, general } = await lanedFixture(t), lanes = approval.plan.hostLanes.mlx;
  await initializeLedger({ approval, backend: 'candle', ledgerRoot: approval.ledgerRoots.candle });
  for (const host of [DATASET_HOST, GENERAL_HOST]) await initializeLedger({ approval, backend: 'mlx', host, ledgerRoot: approval.ledgerRoots[`mlx@${host}`] });
  await laneReserve(approval, GENERAL_HOST, approval.plan.generationInvocations.find(i => lanes[GENERAL_HOST].includes(i.invocationId) && i.kind === 'product_generation'), general);
  const receipts = [await auditLedger({ approval, backend: 'candle', ledgerRoot: approval.ledgerRoots.candle }),
    ...await Promise.all([DATASET_HOST, GENERAL_HOST].map(host => auditLedger({ approval, backend: 'mlx', host, ledgerRoot: approval.ledgerRoots[`mlx@${host}`], inputSpecAddendum: host === GENERAL_HOST ? general : undefined })))];
  const lane = r => r.manifest.host ? `${r.manifest.backend}@${r.manifest.host}` : r.manifest.backend;
  const closure = { kind: 'flux3-terminal-finalized-ledger-heads', coordinatorClosureId: 'DECLARED_SYNTHETIC_TRUSTED_CLOSURE', producersStopped: true, collectionManifestSha256: d('9'), ledgerId: approval.ledgerId, approvedPlanSha256: planHash(approval),
    backends: Object.fromEntries(receipts.map(r => [lane(r), { sequence: r.state.sequence, headSha256: r.state.headSha256 }])) };
  const addenda = { [`mlx@${DATASET_HOST}`]: dataset, [`mlx@${GENERAL_HOST}`]: general };
  assert.equal(aggregateLedgerReceipts(approval, receipts, closure, addenda).generationCount, 1);
  assert.throws(() => aggregateLedgerReceipts(approval, receipts.slice(0, 2), closure, addenda), /every complete latest lane receipt/);
  const { [`mlx@${DATASET_HOST}`]: _omitted, ...partialHeads } = closure.backends;
  assert.throws(() => aggregateLedgerReceipts(approval, receipts, { ...closure, backends: partialHeads }, addenda), /EVERY ledger lane/);
});
test('admission: a split backend admits only its lane hosts; an unsplit MLX backend keeps nax-macos-2 only', async t => {
  const { approval } = await lanedFixture(t);
  assert.equal(admittedHost(approval, { backend: 'mlx', runner: DATASET_HOST }), DATASET_HOST);
  assert.throws(() => admittedHost(approval, { backend: 'mlx', runner: 'nax-macos-3' }), /not an approved mlx lane/);
  assert.equal(admittedHost(approval, { backend: 'candle', runner: 'declared-cuda-runner' }), undefined);
  const { approval: unsplit } = await approvalFixture(t);
  assert.throws(() => admittedHost(unsplit, { backend: 'mlx', runner: DATASET_HOST }), /only secondary nax-macos-2/);
  assert.equal(admittedHost(unsplit, { backend: 'mlx', runner: GENERAL_HOST }), undefined);
});
