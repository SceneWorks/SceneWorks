// DECLARED SYNTHETIC local ledger fixtures only. ZERO physical GPU/model invocations.
import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtemp, readFile, writeFile, rm, mkdir } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { loadProductTargets, terminalPlan, terminalCatalog } from './flux3-image-klein-terminal-plan.mjs';
import { validateApprovedPlan, initializeLedger, reserveInvocation, auditLedger, aggregateLedgerReceipts, planHash } from './flux3-image-klein-terminal-budget.mjs';
const sources = await loadProductTargets(), profile = terminalPlan(sources), digest = 'd'.repeat(64);
async function fixture(t) {
  const root = await mkdtemp(path.join(os.tmpdir(), 'sc26243-synthetic-ledger-only-')); t.after(() => rm(root, { recursive: true, force: true }));
  const approval = { schemaVersion: 1, kind: 'flux3-terminal-plan-approval', approvalId: 'declared-fixture-approval', userPlanApprovalId: 'DECLARED_SYNTHETIC_NOT_USER_AUTHORIZATION', approvedBy: 'synthetic CPU test', sourceSha: 'a'.repeat(40), inferenceMainSha: 'b'.repeat(40), ledgerId: 'declared-fixture-ledger', ledgerRoots: { mlx: path.join(root, 'mlx-ledger'), candle: path.join(root, 'candle-ledger') }, inputSpecSha256: { mlx: digest, candle: digest }, plan: structuredClone(profile) };
  return { root, approval };
}
// Declared fixture coordinator observations, separate from candidate receipts under mutation.
const heads = (approval, receipts) => ({kind:'flux3-terminal-finalized-ledger-heads',coordinatorClosureId:'DECLARED_SYNTHETIC_TRUSTED_CLOSURE',producersStopped:true,collectionManifestSha256:digest,ledgerId:approval.ledgerId,approvedPlanSha256:planHash(approval),backends:Object.fromEntries(receipts.map(r => [r.manifest.backend, { sequence:r.state.sequence,headSha256:r.state.headSha256 }]))});
const reservation = (approval, backend, invocation = approval.plan.generationInvocations.find(i => i.backend === backend && i.kind === 'product_generation')) => ({ approval, backend, ledgerRoot: approval.ledgerRoots[backend], ledgerId: approval.ledgerId, approvedPlanSha256: planHash(approval), invocationId: invocation.invocationId, caseId: invocation.caseId, kind: invocation.kind, requestSha256: digest });
test('representative default expands EXACTLY 26 product +12 resource =38 globally, no automatic repeats/spares', () => {
  assert.equal(profile.generationInvocations.length, 38); assert.equal(profile.generationInvocations.filter(i => i.kind === 'product_generation').length, 26); assert.equal(profile.resourceCases.length, 12);
  assert.deepEqual(profile.budget.byBackend, { mlx: 17, candle: 21 }); assert.equal(profile.budget.globalHardCeiling, 48); assert.equal(profile.budget.unallocatedNotAuthorized, 10);
  assert(profile.generationInvocations.every(i => i.nativeGenerationCalls === 1 && i.warmups === 0 && i.repeats === 0 && i.requestTemplate.count === 1));
  assert.equal(profile.trainingLifecycles.length, 2); assert(profile.trainingLifecycles.every(t => t.steps === 4 && t.saveEvery === 1 && t.generationReloads === 1));
  assert(profile.resourceCases.every(c => c.status === 'BLOCKED')); assert(profile.uncovered.some(c => /Base q4\/q8/.test(c.requirement) && c.status === 'NOT_RUN')); assert(!profile.uncovered.some(c => /HF403/.test(c.reason))); assert(profile.uncovered.every(c => c.status !== 'PASS'));
  for (const backend of ['mlx', 'candle']) {
    assert.deepEqual(new Set(profile.cases.filter(c => c.backend === backend && c.options.enhancePrompt).map(c => c.options.pupMode)), new Set(['t2i', 'i2i', 't2i_bbox', 'i2i_bbox']));
    for (const modelId of ['flux3_klein', 'flux3_klein_base']) for (const count of [1, 10]) assert(profile.cases.some(c => c.backend === backend && c.modelId === modelId && c.options.references === count));
  }
  assert.equal(terminalCatalog(sources).diagnosticOnly, true); assert(profile.cases.length < terminalCatalog(sources).cases.length);
});
test('approval rejects diagnostic catalog, >48 inner calls, implicit multiplication, duplicate permits and missing user approval', async t => {
  const { approval } = await fixture(t); assert.equal(validateApprovedPlan(approval).generationCount, 38);
  for (const mutate of [a => delete a.userPlanApprovalId, a => a.plan = terminalCatalog(sources), a => a.plan.generationInvocations[0].nativeGenerationCalls = 2, a => a.plan.generationInvocations[0].warmups = 1, a => a.plan.generationInvocations[0].repeats = 2, a => a.plan.generationInvocations[1].invocationId = a.plan.generationInvocations[0].invocationId, a => a.plan.trainingLifecycles[0].steps = 20, a => a.plan.trainingLifecycles.push(a.plan.trainingLifecycles[0]), a => { while (a.plan.generationInvocations.length < 49) a.plan.generationInvocations.push({ ...a.plan.generationInvocations[0], invocationId: `unapproved-extra-${a.plan.generationInvocations.length}` }); }]) { const bad = structuredClone(approval); mutate(bad); assert.throws(() => validateApprovedPlan(bad)); }
});
test('disjoint backend ledgers enforce one GLOBAL immutable quota; coordinator aggregates unique receipts', async t => {
  const { approval } = await fixture(t);
  for (const backend of ['mlx', 'candle']) { await initializeLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] }); await reserveInvocation(reservation(approval, backend)); }
  const receipts = await Promise.all(['mlx', 'candle'].map(backend => auditLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] })));
  assert.equal(aggregateLedgerReceipts(approval, receipts, heads(approval, receipts)).generationCount, 2); assert.equal(aggregateLedgerReceipts(approval, receipts, heads(approval, receipts)).approvedTotal, 38);
  await assert.rejects(reserveInvocation({ ...reservation(approval, 'mlx'), ...{ invocationId: approval.plan.generationInvocations.find(i => i.backend === 'candle').invocationId } }), /absent|consumed/);
  assert.throws(() => aggregateLedgerReceipts(approval, [receipts[0], receipts[0]], heads(approval, receipts)), /duplicate/);
  const corrupt = structuredClone(receipts[0]); corrupt.events[0].requestSha256 = 'e'.repeat(64); assert.throws(() => aggregateLedgerReceipts(approval, [corrupt], heads(approval, receipts)));
});
test('failed/canceled reservations stay spent across resumed sessions; no automatic retry or spare-ten allocation', async t => {
  const { approval } = await fixture(t), backend = 'candle'; await initializeLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] });
  const first = reservation(approval, backend); await reserveInvocation(first);
  await assert.rejects(reserveInvocation(first), /already consumed/);
  await assert.rejects(reserveInvocation({ ...first, invocationId: 'unallocated-spare-ten', kind: 'retry' }), /absent/);
  const receipt = await auditLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] }); assert.equal(receipt.state.generationCount, 1); assert.equal(receipt.events.length, 1);
});
test('atomic local reservation lock prevents concurrent double consumption', async t => {
  const { approval } = await fixture(t), backend = 'mlx'; await initializeLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] });
  const attempts = await Promise.allSettled([reserveInvocation(reservation(approval, backend)), reserveInvocation(reservation(approval, backend))]);
  assert.equal(attempts.filter(a => a.status === 'fulfilled').length, 1); assert.equal((await auditLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] })).state.generationCount, 1);
});
test('journal/state crash mismatch blocks all further starts rather than refunding/recovering a permit', async t => {
  const { approval } = await fixture(t), backend = 'candle', ledgerRoot = approval.ledgerRoots[backend]; await initializeLedger({ approval, backend, ledgerRoot }); await reserveInvocation(reservation(approval, backend));
  const file = path.join(ledgerRoot, 'state.json'), state = JSON.parse(await readFile(file, 'utf8')); state.generationCount = 0; await writeFile(file, JSON.stringify(state));
  await assert.rejects(auditLedger({ approval, backend, ledgerRoot }), /crash\/tamper/); await assert.rejects(reserveInvocation(reservation(approval, backend)), /crash\/tamper/);
});
test('runtime never initializes missing ledger; durable sibling claim forbids reset after directory deletion', async t => {
  const { approval } = await fixture(t), backend = 'mlx', ledgerRoot = approval.ledgerRoots[backend];
  await assert.rejects(reserveInvocation(reservation(approval, backend)), /missing\/busy/);
  await initializeLedger({ approval, backend, ledgerRoot }); await rm(ledgerRoot, { recursive: true });
  await assert.rejects(initializeLedger({ approval, backend, ledgerRoot }), /EEXIST/); await assert.rejects(reserveInvocation(reservation(approval, backend)), /missing\/busy/);
  await assert.rejects(reserveInvocation({ ...reservation(approval, backend), ledgerRoot: path.join(path.dirname(ledgerRoot), 'alternate-reset') }), /alternate ledger/);
});
test('only two fixed training lifecycles, one start and one planned resume each; reload still consumes generation permit', async t => {
  const { approval } = await fixture(t), backend = 'candle', training = approval.plan.trainingLifecycles.find(t => t.backend === backend); await initializeLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] });
  const base = { ...reservation(approval, backend), caseId: training.caseId };
  await assert.rejects(reserveInvocation({ ...base, invocationId: training.resumeId, kind: 'training_resume' }), /initial training start/);
  await reserveInvocation({ ...base, invocationId: training.startId, kind: 'training_start' }); await reserveInvocation({ ...base, invocationId: training.resumeId, kind: 'training_resume' });
  assert.equal((await auditLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] })).state.generationCount, 0);
  const reload = approval.plan.generationInvocations.find(i => i.caseId === training.caseId); await reserveInvocation(reservation(approval, backend, reload));
  assert.equal((await auditLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] })).state.generationCount, 1);
  await assert.rejects(reserveInvocation({ ...base, invocationId: training.resumeId, kind: 'training_resume' }), /already consumed/);
});
test('resource permits remain BLOCKED without real owner-verified single-call boundary; exact normalized request required', async t => {
  const { approval } = await fixture(t), backend = 'mlx', resource = approval.plan.generationInvocations.find(i => i.backend === backend && i.kind === 'resource_anchor'); await initializeLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] });
  await assert.rejects(reserveInvocation(reservation(approval, backend, resource)), /boundary is unverified/);
  assert.equal((await auditLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] })).state.generationCount, 0);
  // Separate DECLARED SYNTHETIC approved fixture, never a claim that native enforcement exists.
  const second = await fixture(t), a = second.approval, r = a.plan.resourceCases.find(c => c.caseId === resource.caseId); r.status = 'READY'; r.parentNativeBoundaryVerificationId = 'synthetic-contract-only'; r.nativeBoundaryReceiptSha256 = digest;
  const invocation = a.plan.generationInvocations.find(i => i.caseId === r.caseId); invocation.requestSha256 = digest; await initializeLedger({ approval: a, backend, ledgerRoot: a.ledgerRoots[backend] });
  await assert.rejects(reserveInvocation({ ...reservation(a, backend, invocation), requestSha256: 'e'.repeat(64) }), /exact normalized/);
  await reserveInvocation(reservation(a, backend, invocation)); assert.equal((await auditLedger({ approval: a, backend, ledgerRoot: a.ledgerRoots[backend] })).state.generationCount, 1);
});
test('aggregation validates complete ledger manifests, contiguous chains and final state; forged/truncated receipts never become zero-spend proof', async t => {
  const { approval } = await fixture(t), backend = 'candle', ledgerRoot = approval.ledgerRoots[backend]; await initializeLedger({ approval, backend, ledgerRoot }); await reserveInvocation(reservation(approval, backend));
  const receipt = await auditLedger({ approval, backend, ledgerRoot });
  await initializeLedger({approval,backend:'mlx',ledgerRoot:approval.ledgerRoots.mlx});const other=await auditLedger({approval,backend:'mlx',ledgerRoot:approval.ledgerRoots.mlx}),closure=heads(approval,[receipt,other]);
  assert.equal(aggregateLedgerReceipts(approval,[receipt,other],closure).caseCompletionProven,false);
  assert.throws(()=>aggregateLedgerReceipts(approval,[receipt],closure),/partial backend/);
  assert.throws(()=>aggregateLedgerReceipts(approval,[receipt,other]),/trusted coordinator closure/);
  const rehash = r => { const {eventSha256,...event}=r.events[0];r.events[0].eventSha256=planHash(event);r.state.headSha256=r.events[0].eventSha256; };
  for (const mutate of [r=>r.events=[],r=>r.state.generationCount=38,r=>r.manifest.ledgerId='different-ledger',r=>r.manifest.backend='mlx',r=>r.manifest.trainingIds=[],r=>r.manifest.permitIds=[],r=>{r.events[0].ledgerId='different';rehash(r);},r=>{r.events[0].sequence=7;rehash(r);},r=>{r.events[0].previousSha256=digest;rehash(r);},r=>{r.events[0].approvedPlanSha256=digest;rehash(r);},r=>r.state.consumedIds=[]]) { const bad=structuredClone(receipt);mutate(bad);assert.throws(()=>aggregateLedgerReceipts(approval,[bad,other],closure)); }
  assert.throws(()=>aggregateLedgerReceipts(approval,[receipt,receipt],closure),/duplicate ledger\/backend/);
  assert.throws(()=>aggregateLedgerReceipts(approval,[{state:{generationCount:38},events:[]},other],closure),/manifest/);
  const next=approval.plan.generationInvocations.filter(i=>i.backend===backend&&i.kind==='product_generation')[1];await reserveInvocation(reservation(approval,backend,next));const latest=await auditLedger({approval,backend,ledgerRoot}),latestClosure=heads(approval,[latest,other]);
  assert.throws(()=>aggregateLedgerReceipts(approval,[receipt,other],latestClosure),/replayed\/stale/);assert.equal(aggregateLedgerReceipts(approval,[latest,other],latestClosure).generationCount,2);
  for(const mutate of [c=>c.producersStopped=false,c=>c.ledgerId='wrong',c=>c.approvedPlanSha256=digest,c=>delete c.backends.mlx]){const bad=structuredClone(latestClosure);mutate(bad);assert.throws(()=>aggregateLedgerReceipts(approval,[latest,other],bad));}
});
