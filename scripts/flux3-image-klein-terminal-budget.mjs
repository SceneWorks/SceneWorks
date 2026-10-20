// No service/distributed allocation. One immutable GLOBAL plan assigns disjoint backend permits.
// Product/resource/resume sessions consume their backend's SAME stable local atomic ledger.
import assert from 'node:assert/strict';
import { createHash, randomBytes } from 'node:crypto';
import { mkdir, open, readFile, readdir, rename, unlink } from 'node:fs/promises';
import path from 'node:path';
export const GLOBAL_GENERATION_CEILING = 48;
export const planHash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const sha = /^[a-f0-9]{64}$/, revision = /^[a-f0-9]{40}$/;
const kinds = ['product_generation', 'resource_anchor', 'resource_supplemental', 'warmup', 'repeat', 'retry'];
export function validateApprovedPlan(approval) {
  assert(approval?.schemaVersion === 1 && approval.kind === 'flux3-terminal-plan-approval', 'explicit immutable terminal plan approval required');
  assert(approval.approvalId && approval.userPlanApprovalId && approval.approvedBy, 'parent-recorded user plan approval identity required');
  assert(revision.test(approval.sourceSha ?? '') && revision.test(approval.inferenceMainSha ?? ''), 'approved final source/MAIN identities required');
  assert(/^[a-zA-Z0-9_-]{12,100}$/.test(approval.ledgerId ?? ''), 'stable non-resettable ledger identity required');
  for (const backend of ['mlx', 'candle']) { assert(/^(?:[a-zA-Z]:[\\/]|\/|\\\\)/.test(approval.ledgerRoots?.[backend] ?? ''), 'fixed absolute per-backend ledger roots required'); assert(sha.test(approval.inputSpecSha256?.[backend] ?? ''), 'both backend input specifications must be bound before execution'); }
  assert.notEqual(approval.ledgerRoots.mlx, approval.ledgerRoots.candle, 'backend ledgers have distinct fixed roots, never independent duplicate quotas');
  const plan = approval.plan; assert(plan?.kind === 'flux3-installed-product-representative-plan' && plan.diagnosticOnly !== true, 'diagnostic catalog is never an execution plan');
  const invocations = plan.generationInvocations; assert(Array.isArray(invocations) && invocations.length > 0 && invocations.length <= GLOBAL_GENERATION_CEILING, 'global expanded physical generation count must be 1..48');
  assert.equal(new Set(invocations.map(i => i.invocationId)).size, invocations.length, 'single-use invocation IDs must be globally unique');
  for (const i of invocations) { assert(typeof i.invocationId === 'string' && /^[a-zA-Z0-9_.-]+$/.test(i.invocationId) && typeof i.caseId === 'string' && ['mlx', 'candle'].includes(i.backend) && kinds.includes(i.kind)); assert.equal(i.requestTemplate?.count, 1, 'one permit cannot hide a batched request'); assert.equal(i.nativeGenerationCalls, 1, 'one permit is ONE actual inner Generator.generate, never a command containing hidden loops'); assert.equal(i.warmups, 0, 'warmups require separate named permits'); assert.equal(i.repeats, 0, 'repeats require separate named permits'); if (i.kind === 'retry') assert(i.parentRetryApprovalId, 'retry needs separate explicit parent approval and a preallocated remaining-budget permit'); }
  assert.equal(plan.budget.globalHardCeiling, GLOBAL_GENERATION_CEILING); assert.equal(plan.budget.totalGenerationInvocations, invocations.length);
  for (const backend of ['mlx', 'candle']) assert.equal(plan.budget.byBackend[backend], invocations.filter(i => i.backend === backend).length);
  assert.equal(plan.budget.unallocatedNotAuthorized, GLOBAL_GENERATION_CEILING - invocations.length);
  assert.equal(plan.budget.automaticWarmups, 0); assert.equal(plan.budget.automaticRepeats, 0); assert.equal(plan.budget.automaticRetries, 0);
  const training = plan.trainingLifecycles ?? []; assert(training.length <= 2, 'at most TWO short training lifecycles globally'); assert.equal(plan.budget.trainingLifecycles, training.length); assert.equal(plan.budget.stepsPerLifecycle, 4);
  const trainingIds = training.flatMap(t => { assert(['mlx', 'candle'].includes(t.backend) && t.caseId && t.lifecycleId); assert.equal(t.steps, 4); assert.equal(t.saveEvery, 1); assert.equal(t.rank, 4); assert.equal(t.resolution, 512); assert.equal(t.generationReloads, 1); assert([t.startId, t.resumeId].every(id => typeof id === 'string' && /^[a-zA-Z0-9_.-]+$/.test(id))); return [t.startId, t.resumeId]; });
  assert.equal(new Set([...trainingIds, ...invocations.map(i => i.invocationId)]).size, trainingIds.length + invocations.length);
  return { approvedPlanSha256: planHash(approval), generationCount: invocations.length, backendQuotas: plan.budget.byBackend, trainingLifecycles: training.length };
}
function location(approval, backend, ledgerRoot) {
  validateApprovedPlan(approval); assert(['mlx', 'candle'].includes(backend));
  assert.equal(path.resolve(ledgerRoot), path.resolve(approval.ledgerRoots[backend]), 'alternate ledger root/reset is forbidden'); return path.resolve(ledgerRoot);
}
function expectedManifest(approval, backend) {
  return { schemaVersion: 1, ledgerId: approval.ledgerId, backend, approvedPlanSha256: planHash(approval), approvalId: approval.approvalId, userPlanApprovalId: approval.userPlanApprovalId, permitIds: approval.plan.generationInvocations.filter(i => i.backend === backend).map(i => i.invocationId), trainingIds: approval.plan.trainingLifecycles.filter(t => t.backend === backend).flatMap(t => [t.startId, t.resumeId]) };
}
async function durableNew(file, value) { const h = await open(file, 'wx'); try { await h.writeFile(JSON.stringify(value) + '\n'); await h.sync(); } finally { await h.close(); } }
// Explicit, separately invoked CPU setup ONLY. Runtime never initializes or resets a ledger.
// The durable sibling initialization claim survives deletion of the ledger directory.
export async function initializeLedger({ approval, backend, ledgerRoot }) {
  const root = location(approval, backend, ledgerRoot), hash = planHash(approval);
  await durableNew(path.join(path.dirname(root), `.flux3-${approval.ledgerId}-${backend}.initialized`), { ledgerId: approval.ledgerId, backend, root, approvedPlanSha256: hash });
  await mkdir(root); await mkdir(path.join(root, 'events'));
  const manifest = expectedManifest(approval, backend);
  await durableNew(path.join(root, 'manifest.json'), manifest);
  await durableNew(path.join(root, 'state.json'), { sequence: 0, generationCount: 0, consumedIds: [], headSha256: planHash(manifest) });
  return { ledgerId: approval.ledgerId, backend, approvedPlanSha256: hash, allocated: manifest.permitIds.length };
}
async function lock(root, callback) {
  let handle; try { handle = await open(path.join(root, 'consume.lock'), 'wx'); } catch { throw new Error('BLOCKED: existing stable ledger missing/busy; no initialization/reset/stale-lock recovery is permitted at runtime'); }
  try { return await callback(); } finally { await handle.close(); await unlink(path.join(root, 'consume.lock')); }
}
function approvedEvent(approval, event) {
  if (event.kind === 'training_start' || event.kind === 'training_resume') {
    const training = approval.plan.trainingLifecycles.find(t => t.backend === event.backend && t.caseId === event.caseId && (event.kind === 'training_start' ? t.startId : t.resumeId) === event.invocationId); assert(training, 'unplanned training start/resume'); return false;
  }
  const invocation = approval.plan.generationInvocations.find(i => i.invocationId === event.invocationId);
  assert(invocation && invocation.backend === event.backend && invocation.caseId === event.caseId && invocation.kind === event.kind, 'invocation absent from this immutable backend quota; no spare allocation or automatic retry');
  if (invocation.kind.startsWith('resource_')) {
    const resource = approval.plan.resourceCases.find(c => c.caseId === invocation.caseId && c.backend === invocation.backend);
    assert(resource?.status === 'READY' && resource.parentNativeBoundaryVerificationId && sha.test(resource.nativeBoundaryReceiptSha256 ?? ''), 'BLOCKED: native inner-generate boundary is unverified; a contract flag is not proof. Authorized owner/coordinator must bind the real single-call verification receipt before approving this immutable plan.');
    assert(sha.test(invocation.requestSha256 ?? ''), 'resource request must be normalized/hash-bound in the approved plan');
  }
  if (invocation.requestSha256) assert.equal(event.requestSha256, invocation.requestSha256, 'resource permit must bind exact normalized approved request');
  return true;
}
export function verifyLedgerReceipt(approval, receipt, expectedBackend = receipt?.manifest?.backend) {
  validateApprovedPlan(approval); assert(['mlx', 'candle'].includes(expectedBackend), 'complete ledger manifest/backend required');
  const { manifest, state, events } = receipt ?? {}; assert.deepEqual(manifest, expectedManifest(approval, expectedBackend), 'complete ledger manifest/plan/backend identity mismatch'); assert(Array.isArray(events));
  let head = planHash(manifest), generations = 0; const ids = [];
  for (const [index, event] of events.entries()) {
    const { eventSha256, ...record } = event;
    assert.deepEqual(Object.keys(record).sort(), ['sequence','previousSha256','approvedPlanSha256','ledgerId','backend','caseId','invocationId','kind','requestSha256','reservedAt','controllerPid'].sort());
    assert.equal(eventSha256, planHash(record)); assert.equal(record.sequence, index + 1); assert.equal(record.previousSha256, head); assert.equal(record.ledgerId, approval.ledgerId); assert.equal(record.approvedPlanSha256, planHash(approval)); assert.equal(record.backend, expectedBackend); assert(sha.test(record.requestSha256)); assert(Number.isInteger(record.controllerPid) && record.controllerPid > 0 && Number.isFinite(Date.parse(record.reservedAt)));
    if (record.kind === 'training_resume') { const training = approval.plan.trainingLifecycles.find(t => t.resumeId === record.invocationId); assert(training && ids.includes(training.startId), 'resume precedes its initial training start'); }
    if (approvedEvent(approval, record)) generations++; assert(!ids.includes(record.invocationId), 'single-use permit reused'); ids.push(record.invocationId); head = eventSha256;
  }
  assert(generations <= manifest.permitIds.length && generations <= GLOBAL_GENERATION_CEILING);
  assert.deepEqual(state, { sequence: events.length, generationCount: generations, consumedIds: ids, headSha256: head }, 'ledger crash/tamper mismatch; coordinator must reconcile, never reuse a reservation');
  return { manifest, state, events };
}
async function audited(approval, backend, root) {
  const manifest = JSON.parse(await readFile(path.join(root, 'manifest.json'), 'utf8')), state = JSON.parse(await readFile(path.join(root, 'state.json'), 'utf8')), files = (await readdir(path.join(root, 'events'))).sort(), events = [];
  for (const [index, file] of files.entries()) { assert.equal(file, `${String(index + 1).padStart(6, '0')}.json`, 'ledger journal missing/foreign event'); events.push(JSON.parse(await readFile(path.join(root, 'events', file), 'utf8'))); }
  return verifyLedgerReceipt(approval, { manifest, state, events }, backend);
}
export async function auditLedger({ approval, backend, ledgerRoot }) { const root = location(approval, backend, ledgerRoot); return lock(root, () => audited(approval, backend, root)); }
export async function reserveInvocation({ approval, ledgerRoot, ledgerId, approvedPlanSha256, invocationId, backend, caseId, kind, requestSha256 }) {
  const root = location(approval, backend, ledgerRoot); assert.equal(ledgerId, approval.ledgerId); assert.equal(approvedPlanSha256, planHash(approval)); assert(sha.test(requestSha256 ?? ''), 'actual request digest required before each inner generation start');
  return lock(root, async () => {
    const { state } = await audited(approval, backend, root); assert(!state.consumedIds.includes(invocationId), 'permit already consumed; failed/canceled attempts count and cannot auto-resubmit');
    if (kind === 'training_resume') { const training = approval.plan.trainingLifecycles.find(t => t.resumeId === invocationId); assert(training && state.consumedIds.includes(training.startId), 'planned resume requires the already-consumed initial training start'); }
    const record = { sequence: state.sequence + 1, previousSha256: state.headSha256, approvedPlanSha256, ledgerId, backend, caseId, invocationId, kind, requestSha256, reservedAt: new Date().toISOString(), controllerPid: process.pid };
    const generation = approvedEvent(approval, record), event = { ...record, eventSha256: planHash(record) };
    // Journal first. A crash before state publication is a fail-closed reconciliation blocker.
    await durableNew(path.join(root, 'events', `${String(record.sequence).padStart(6, '0')}.json`), event);
    const next = { sequence: record.sequence, generationCount: state.generationCount + Number(generation), consumedIds: [...state.consumedIds, invocationId], headSha256: event.eventSha256 };
    const temporary = path.join(root, `state-${randomBytes(16).toString('hex')}.tmp`); await durableNew(temporary, next); await rename(temporary, path.join(root, 'state.json'));
    return event;
  });
}
export function aggregateLedgerReceipts(approval, receipts, closure) {
  validateApprovedPlan(approval); const ids = new Set(), ledgers = new Set(); let generationCount = 0;
  assert(closure?.kind === 'flux3-terminal-finalized-ledger-heads' && closure.coordinatorClosureId && closure.producersStopped === true && sha.test(closure.collectionManifestSha256 ?? ''), 'trusted coordinator closure from stopped producers/sealed collected artifacts required; receipts cannot attest their own freshness');
  assert.equal(closure.ledgerId, approval.ledgerId); assert.equal(closure.approvedPlanSha256, planHash(approval));
  const expectedHeads = closure.backends;
  assert(expectedHeads && Object.keys(expectedHeads).sort().join(',') === 'candle,mlx', 'trusted latest coordinator-audited heads for BOTH backend ledgers required');
  for (const candidate of receipts) {
    const receipt = verifyLedgerReceipt(approval, candidate), key = `${receipt.manifest.ledgerId}:${receipt.manifest.backend}`; assert(!ledgers.has(key), 'duplicate ledger/backend receipt; use one complete latest receipt per backend'); ledgers.add(key);
    assert.deepEqual({ sequence: receipt.state.sequence, headSha256: receipt.state.headSha256 }, expectedHeads[receipt.manifest.backend], 'replayed/stale receipt differs from trusted latest coordinator-audited head');
    for (const event of receipt.events) { assert(!ids.has(event.invocationId), 'duplicate receipt/permit across backend sessions'); ids.add(event.invocationId); }
    generationCount += receipt.state.generationCount;
  }
  assert.equal(ledgers.size, 2, 'partial backend aggregation is not a global campaign count; both complete latest receipts required');
  assert(generationCount <= approval.plan.generationInvocations.length && generationCount <= GLOBAL_GENERATION_CEILING);
  return { proofKind: 'reservation-accounting-only', caseCompletionProven: false, coordinatorClosureId: closure.coordinatorClosureId, collectionManifestSha256: closure.collectionManifestSha256, ledgerId: approval.ledgerId, approvedPlanSha256: planHash(approval), generationCount, approvedTotal: approval.plan.generationInvocations.length, globalHardCeiling: GLOBAL_GENERATION_CEILING, unallocatedNotAuthorized: GLOBAL_GENERATION_CEILING - approval.plan.generationInvocations.length };
}
