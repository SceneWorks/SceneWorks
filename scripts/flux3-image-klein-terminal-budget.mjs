// No service/distributed allocation. One immutable GLOBAL plan assigns disjoint backend permits.
// Product/resource/resume sessions consume their backend's SAME stable local atomic ledger.
import assert from 'node:assert/strict';
import { createHash, randomBytes } from 'node:crypto';
import { mkdir, open, readFile, readdir, rename, unlink } from 'node:fs/promises';
import path from 'node:path';
import { validateNativeBoundaryReceipt } from './lib/flux3-native-boundary.mjs';
export const GLOBAL_GENERATION_CEILING = 48;
export const planHash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const sha = /^[a-f0-9]{64}$/, revision = /^[a-f0-9]{40}$/;
const kinds = ['product_generation', 'resource_anchor', 'resource_supplemental', 'warmup', 'repeat', 'retry'];
export function validateApprovedPlan(approval) {
  assert(approval?.schemaVersion === 1 && approval.kind === 'flux3-terminal-plan-approval', 'explicit immutable terminal plan approval required');
  assert(approval.approvalId && approval.userPlanApprovalId && approval.approvedBy, 'parent-recorded user plan approval identity required');
  assert(revision.test(approval.sourceSha ?? '') && revision.test(approval.inferenceMainSha ?? ''), 'approved final source/MAIN identities required');
  assert(/^[a-zA-Z0-9_-]{12,100}$/.test(approval.ledgerId ?? ''), 'stable non-resettable ledger identity required');
  // A backend whose inputs cannot exist yet (e.g. a second host still to be provisioned) may be
  // DEFERRED: its spec is absent here and only a parent-recorded addendum chained to this exact
  // approval can bind it later. At least one backend is bound at approval time.
  const deferred = approval.deferredInputSpecBackends ?? [];
  assert(Array.isArray(deferred) && deferred.every(b => ['mlx', 'candle'].includes(b)) && new Set(deferred).size === deferred.length && deferred.length < 2, 'deferred input specifications must name at most one known backend');
  for (const backend of ['mlx', 'candle']) {
    assert(/^(?:[a-zA-Z]:[\\/]|\/|\\\\)/.test(approval.ledgerRoots?.[backend] ?? ''), 'fixed absolute per-backend ledger roots required');
    if (deferred.includes(backend)) assert(!Object.hasOwn(approval.inputSpecSha256 ?? {}, backend), `deferred ${backend} input specification cannot also be bound in the approval`);
    else assert(sha.test(approval.inputSpecSha256?.[backend] ?? ''), 'every non-deferred backend input specification must be bound before execution');
  }
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
  // Adapter cases use ONLY the adapter their own backend's lifecycle exports in this plan: same
  // backend, the lifecycle's trained (base) checkpoint, and strictly after its reload generation.
  invocations.forEach((i, index) => {
    if (i.purpose === 'training-reload' || !Object.hasOwn(i.options ?? {}, 'loraScale')) return;
    const binding = i.adapterBinding;
    assert(binding?.source === 'training-export', `adapter invocation ${i.invocationId} must bind the plan's own exported LoRA`);
    const lifecycle = training.find(t => t.lifecycleId === binding.lifecycleId);
    assert(lifecycle && lifecycle.backend === i.backend && binding.backend === i.backend, `adapter invocation ${i.invocationId} must bind its own backend's lifecycle`);
    assert(binding.modelId === i.modelId, `adapter invocation ${i.invocationId} targets a different checkpoint than its LoRA was trained on`);
    const reload = invocations.findIndex(j => j.caseId === lifecycle.caseId && j.purpose === 'training-reload');
    assert(reload >= 0 && reload < index, `adapter invocation ${i.invocationId} must run strictly after its LoRA's training lifecycle`);
    assert.equal(invocations[reload].modelId, i.modelId, `adapter invocation ${i.invocationId} checkpoint differs from the trained LoRA's`);
  });
  // Resource rows: BLOCKED (unreservable), READY (proved in this plan against the approval's own
  // sealed resource build) or READY_ON_ADDENDUM (a deferred backend proved only by its addendum).
  for (const row of plan.resourceCases ?? []) {
    assert(['BLOCKED', 'READY', 'READY_ON_ADDENDUM'].includes(row.status), `unknown resource row status ${row.status}`);
    if (row.status === 'BLOCKED') continue;
    const rowInvocations = invocations.filter(i => i.caseId === row.caseId && i.backend === row.backend);
    assert(rowInvocations.length && rowInvocations.every(i => sha.test(i.requestSha256 ?? '')), `${row.caseId} resource request must be hash-bound before it can be READY`);
    if (row.status === 'READY') {
      assert(!deferred.includes(row.backend), `${row.caseId}: a deferred backend is READY only by addendum`);
      assert(row.parentNativeBoundaryVerificationId && sha.test(row.nativeBoundaryReceiptSha256 ?? ''), `${row.caseId}: READY requires the native-boundary verification receipt`);
      assert.equal(row.nativeBoundaryBinarySha256, approval.resourceBuildReceipts?.[row.backend]?.binarySha256, `${row.caseId}: native-boundary receipt came from a binary other than the approved sealed resource build`);
    } else assert(deferred.includes(row.backend), `${row.caseId}: READY_ON_ADDENDUM is only for a deferred backend`);
  }
  // Never trust a READY label or digest alone: re-validate the FULL receipt the plan carries against
  // the approval's own sealed resource build, source, inference and every resource invocation.
  for (const backend of ['mlx', 'candle']) {
    const ready = (plan.resourceCases ?? []).filter(r => r.backend === backend && r.status === 'READY');
    if (!ready.length) continue;
    const build = approval.resourceBuildReceipts?.[backend];
    const receiptSha256 = validateNativeBoundaryReceipt(plan.nativeBoundaryReceipts?.[backend], { backend,
      invocations: invocations.filter(i => i.backend === backend && i.kind.startsWith('resource_')),
      binarySha256: build?.binarySha256, buildReceiptSha256: build?.receiptSha256, sourceSha: approval.sourceSha, inferenceMainSha: approval.inferenceMainSha });
    for (const row of ready) assert.equal(row.nativeBoundaryReceiptSha256, receiptSha256, `${row.caseId}: READY digest is not the carried receipt's`);
    assert.equal(ready.length, (plan.resourceCases ?? []).filter(r => r.backend === backend).length, `${backend}: a receipt proves every resource row of its backend or none`);
  }
  return { approvedPlanSha256: planHash(approval), generationCount: invocations.length, backendQuotas: plan.budget.byBackend, trainingLifecycles: training.length };
}
const ADDENDUM_KEYS = ['schemaVersion', 'kind', 'approvedPlanSha256', 'ledgerId', 'backend', 'inputSpecSha256', 'buildReceipt', 'resourceBuildReceipt', 'nativeBoundaryReceipt', 'parentNativeBoundaryVerificationId', 'addendumId', 'userApprovalId', 'approvedBy'].sort();
/** Validate a parent-recorded input-spec addendum. It can ONLY bind a deferred backend's input
 *  specification for this exact approval: it never changes the plan, any bound spec, the ledger
 *  identity or the counts, and an addendum for any other approval is refused. */
export function validateInputSpecAddendum(approval, addendum) {
  const base = validateApprovedPlan(approval);
  assert(addendum && typeof addendum === 'object' && JSON.stringify(Object.keys(addendum).sort()) === JSON.stringify(ADDENDUM_KEYS), 'addendum carries exactly its binding fields, nothing that could amend the plan, ledger or counts');
  assert(addendum.schemaVersion === 1 && addendum.kind === 'flux3-terminal-plan-input-spec-addendum', 'explicit input-spec addendum required');
  assert.equal(addendum.approvedPlanSha256, base.approvedPlanSha256, 'addendum is chained to a different approved plan');
  assert.equal(addendum.ledgerId, approval.ledgerId, 'addendum ledger identity differs from the approval');
  assert((approval.deferredInputSpecBackends ?? []).includes(addendum.backend), 'only a deferred backend input specification can be bound by addendum');
  assert(sha.test(addendum.inputSpecSha256 ?? ''), 'addendum must bind an input specification digest');
  for (const key of ['addendumId', 'userApprovalId', 'approvedBy', 'parentNativeBoundaryVerificationId']) assert(typeof addendum[key] === 'string' && addendum[key].length > 0, `addendum ${key} required`);
  // The deferred host's own sealed builds, bound like the approval's bindings for a non-deferred
  // backend; never a replacement for build bindings the approval already carries.
  for (const key of ['buildReceipt', 'resourceBuildReceipt']) assert(addendum[key] && sha.test(addendum[key].receiptSha256 ?? '') && sha.test(addendum[key].binarySha256 ?? ''), `addendum ${key} must bind a sealed build receipt and binary`);
  assert(!approval.buildReceipts?.[addendum.backend] && !approval.resourceBuildReceipts?.[addendum.backend], 'an addendum cannot replace build bindings the approval already carries');
  // Its READY_ON_ADDENDUM resource rows become reservable only with a valid self-test receipt from
  // exactly this sealed resource binary, proving exactly the plan's fixed request digests.
  const invocations = approval.plan.generationInvocations.filter(i => i.backend === addendum.backend && i.kind.startsWith('resource_')
    && approval.plan.resourceCases.find(r => r.caseId === i.caseId && r.backend === i.backend)?.status === 'READY_ON_ADDENDUM');
  const nativeBoundaryReceiptSha256 = invocations.length ? validateNativeBoundaryReceipt(addendum.nativeBoundaryReceipt, { backend: addendum.backend, invocations,
    binarySha256: addendum.resourceBuildReceipt.binarySha256, buildReceiptSha256: addendum.resourceBuildReceipt.receiptSha256,
    sourceSha: approval.sourceSha, inferenceMainSha: approval.inferenceMainSha }) : null;
  return { backend: addendum.backend, inputSpecSha256: addendum.inputSpecSha256, approvedPlanSha256: base.approvedPlanSha256, nativeBoundaryReceiptSha256,
    buildReceipt: addendum.buildReceipt, resourceBuildReceipt: addendum.resourceBuildReceipt };
}
/** The input-spec digest a backend may run with. A deferred backend stays unusable (BLOCKED)
 *  until its addendum exists; the approval's own hash and ledgers are unchanged by it. */
export function boundInputSpecSha256(approval, backend, addendum) {
  assert(['mlx', 'candle'].includes(backend));
  // Callers validate the approval itself; the addendum path re-validates it in full below.
  if (!(approval.deferredInputSpecBackends ?? []).includes(backend)) {
    assert(addendum === undefined || addendum === null, `${backend} input specification is already bound by the approval; no addendum may replace it`);
    return approval.inputSpecSha256[backend];
  }
  if (!addendum) throw new Error(`BLOCKED: ${backend} input specification is deferred; its permits stay unusable until a parent-recorded addendum binds it`);
  const bound = validateInputSpecAddendum(approval, addendum); assert.equal(bound.backend, backend, 'addendum binds a different backend');
  return bound.inputSpecSha256;
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
function approvedEvent(approval, event, addendum) {
  if (event.kind === 'training_start' || event.kind === 'training_resume') {
    const training = approval.plan.trainingLifecycles.find(t => t.backend === event.backend && t.caseId === event.caseId && (event.kind === 'training_start' ? t.startId : t.resumeId) === event.invocationId); assert(training, 'unplanned training start/resume'); return false;
  }
  const invocation = approval.plan.generationInvocations.find(i => i.invocationId === event.invocationId);
  assert(invocation && invocation.backend === event.backend && invocation.caseId === event.caseId && invocation.kind === event.kind, 'invocation absent from this immutable backend quota; no spare allocation or automatic retry');
  if (invocation.kind.startsWith('resource_')) {
    const resource = approval.plan.resourceCases.find(c => c.caseId === invocation.caseId && c.backend === invocation.backend);
    if (resource?.status === 'READY_ON_ADDENDUM') {
      assert(addendum, 'BLOCKED: this resource row is reservable only after its backend addendum binds a validated native-boundary self-test receipt');
      assert.equal(validateInputSpecAddendum(approval, addendum).backend, invocation.backend, 'BLOCKED: addendum binds a different backend');
    } else assert(resource?.status === 'READY' && resource.parentNativeBoundaryVerificationId && sha.test(resource.nativeBoundaryReceiptSha256 ?? ''), 'BLOCKED: native inner-generate boundary is unverified; a contract flag is not proof. Authorized owner/coordinator must bind the real single-call verification receipt before approving this immutable plan.');
    assert(sha.test(invocation.requestSha256 ?? ''), 'resource request must be normalized/hash-bound in the approved plan');
  }
  if (invocation.requestSha256) assert.equal(event.requestSha256, invocation.requestSha256, 'resource permit must bind exact normalized approved request');
  return true;
}
export function verifyLedgerReceipt(approval, receipt, expectedBackend = receipt?.manifest?.backend, inputSpecAddendum) {
  validateApprovedPlan(approval); assert(['mlx', 'candle'].includes(expectedBackend), 'complete ledger manifest/backend required');
  const { manifest, state, events, inputSpecAddendumSha256 } = receipt ?? {};
  // A deferred backend's ledger is bound to exactly ONE addendum: recorded before its first spend
  // and required (identical) for every later reservation, audit and aggregation.
  if ((approval.deferredInputSpecBackends ?? []).includes(expectedBackend)) {
    if ((events?.length ?? 0) > 0) assert(sha.test(inputSpecAddendumSha256 ?? ''), 'BLOCKED: deferred backend ledger has spends but no recorded addendum binding');
    if (inputSpecAddendumSha256 !== undefined) {
      assert(inputSpecAddendum, 'BLOCKED: this ledger is bound to its recorded addendum; present that same addendum');
      assert.equal(planHash(inputSpecAddendum), inputSpecAddendumSha256, 'BLOCKED: a different addendum than the one recorded in this ledger');
    }
  } else assert.equal(inputSpecAddendumSha256, undefined, 'a non-deferred backend ledger carries no addendum binding'); assert.deepEqual(manifest, expectedManifest(approval, expectedBackend), 'complete ledger manifest/plan/backend identity mismatch'); assert(Array.isArray(events));
  let head = planHash(manifest), generations = 0; const ids = [];
  for (const [index, event] of events.entries()) {
    const { eventSha256, ...record } = event;
    assert.deepEqual(Object.keys(record).sort(), ['sequence','previousSha256','approvedPlanSha256','ledgerId','backend','caseId','invocationId','kind','requestSha256','reservedAt','controllerPid'].sort());
    assert.equal(eventSha256, planHash(record)); assert.equal(record.sequence, index + 1); assert.equal(record.previousSha256, head); assert.equal(record.ledgerId, approval.ledgerId); assert.equal(record.approvedPlanSha256, planHash(approval)); assert.equal(record.backend, expectedBackend); assert(sha.test(record.requestSha256)); assert(Number.isInteger(record.controllerPid) && record.controllerPid > 0 && Number.isFinite(Date.parse(record.reservedAt)));
    if (record.kind === 'training_resume') { const training = approval.plan.trainingLifecycles.find(t => t.resumeId === record.invocationId); assert(training && ids.includes(training.startId), 'resume precedes its initial training start'); }
    if (approvedEvent(approval, record, inputSpecAddendum)) generations++; assert(!ids.includes(record.invocationId), 'single-use permit reused'); ids.push(record.invocationId); head = eventSha256;
  }
  assert(generations <= manifest.permitIds.length && generations <= GLOBAL_GENERATION_CEILING);
  assert.deepEqual(state, { sequence: events.length, generationCount: generations, consumedIds: ids, headSha256: head }, 'ledger crash/tamper mismatch; coordinator must reconcile, never reuse a reservation');
  return { manifest, state, events, ...(inputSpecAddendumSha256 !== undefined ? { inputSpecAddendumSha256 } : {}) };
}
const ADDENDUM_BINDING = 'input-spec-addendum.json';
async function boundAddendum(root) {
  try { return JSON.parse(await readFile(path.join(root, ADDENDUM_BINDING), 'utf8')); } catch (error) { if (error.code === 'ENOENT') return undefined; throw error; }
}
async function audited(approval, backend, root, inputSpecAddendum) {
  const manifest = JSON.parse(await readFile(path.join(root, 'manifest.json'), 'utf8')), state = JSON.parse(await readFile(path.join(root, 'state.json'), 'utf8')), files = (await readdir(path.join(root, 'events'))).sort(), events = [];
  for (const [index, file] of files.entries()) { assert.equal(file, `${String(index + 1).padStart(6, '0')}.json`, 'ledger journal missing/foreign event'); events.push(JSON.parse(await readFile(path.join(root, 'events', file), 'utf8'))); }
  const binding = await boundAddendum(root);
  return verifyLedgerReceipt(approval, { manifest, state, events, ...(binding ? { inputSpecAddendumSha256: binding.addendumSha256 } : {}) }, backend, inputSpecAddendum);
}
export async function auditLedger({ approval, backend, ledgerRoot, inputSpecAddendum }) { const root = location(approval, backend, ledgerRoot); return lock(root, () => audited(approval, backend, root, inputSpecAddendum)); }
export async function reserveInvocation({ approval, ledgerRoot, ledgerId, approvedPlanSha256, invocationId, backend, caseId, kind, requestSha256, inputSpecAddendum }) {
  const root = location(approval, backend, ledgerRoot);
  // A deferred backend has no permit before its addendum (and only ever the SAME addendum, below).
  boundInputSpecSha256(approval, backend, inputSpecAddendum);
  assert.equal(ledgerId, approval.ledgerId, 'reservation ledger identity differs from the approval');
  assert.equal(approvedPlanSha256, planHash(approval), 'reservation is bound to a different approved plan');
  assert(sha.test(requestSha256 ?? ''), 'actual request digest required before each inner generation start');
  return lock(root, async () => {
    const { state } = await audited(approval, backend, root, inputSpecAddendum);
    if ((approval.deferredInputSpecBackends ?? []).includes(backend) && !(await boundAddendum(root))) {
      // First spend: record WHICH addendum this ledger is bound to, exclusively and durably.
      await durableNew(path.join(root, ADDENDUM_BINDING), { ledgerId: approval.ledgerId, backend, approvedPlanSha256: planHash(approval), addendumSha256: planHash(inputSpecAddendum) });
    } assert(!state.consumedIds.includes(invocationId), 'permit already consumed; failed/canceled attempts count and cannot auto-resubmit');
    if (kind === 'training_resume') { const training = approval.plan.trainingLifecycles.find(t => t.resumeId === invocationId); assert(training && state.consumedIds.includes(training.startId), 'planned resume requires the already-consumed initial training start'); }
    const record = { sequence: state.sequence + 1, previousSha256: state.headSha256, approvedPlanSha256, ledgerId, backend, caseId, invocationId, kind, requestSha256, reservedAt: new Date().toISOString(), controllerPid: process.pid };
    const generation = approvedEvent(approval, record, inputSpecAddendum), event = { ...record, eventSha256: planHash(record) };
    // Journal first. A crash before state publication is a fail-closed reconciliation blocker.
    await durableNew(path.join(root, 'events', `${String(record.sequence).padStart(6, '0')}.json`), event);
    const next = { sequence: record.sequence, generationCount: state.generationCount + Number(generation), consumedIds: [...state.consumedIds, invocationId], headSha256: event.eventSha256 };
    const temporary = path.join(root, `state-${randomBytes(16).toString('hex')}.tmp`); await durableNew(temporary, next); await rename(temporary, path.join(root, 'state.json'));
    return event;
  });
}
export function aggregateLedgerReceipts(approval, receipts, closure, inputSpecAddenda = {}) {
  validateApprovedPlan(approval); const ids = new Set(), ledgers = new Set(); let generationCount = 0;
  assert(closure?.kind === 'flux3-terminal-finalized-ledger-heads' && closure.coordinatorClosureId && closure.producersStopped === true && sha.test(closure.collectionManifestSha256 ?? ''), 'trusted coordinator closure from stopped producers/sealed collected artifacts required; receipts cannot attest their own freshness');
  assert.equal(closure.ledgerId, approval.ledgerId); assert.equal(closure.approvedPlanSha256, planHash(approval));
  const expectedHeads = closure.backends;
  assert(expectedHeads && Object.keys(expectedHeads).sort().join(',') === 'candle,mlx', 'trusted latest coordinator-audited heads for BOTH backend ledgers required');
  for (const candidate of receipts) {
    const receipt = verifyLedgerReceipt(approval, candidate, candidate?.manifest?.backend, inputSpecAddenda[candidate?.manifest?.backend]), key = `${receipt.manifest.ledgerId}:${receipt.manifest.backend}`; assert(!ledgers.has(key), 'duplicate ledger/backend receipt; use one complete latest receipt per backend'); ledgers.add(key);
    assert.deepEqual({ sequence: receipt.state.sequence, headSha256: receipt.state.headSha256 }, expectedHeads[receipt.manifest.backend], 'replayed/stale receipt differs from trusted latest coordinator-audited head');
    for (const event of receipt.events) { assert(!ids.has(event.invocationId), 'duplicate receipt/permit across backend sessions'); ids.add(event.invocationId); }
    generationCount += receipt.state.generationCount;
  }
  assert.equal(ledgers.size, 2, 'partial backend aggregation is not a global campaign count; both complete latest receipts required');
  assert(generationCount <= approval.plan.generationInvocations.length && generationCount <= GLOBAL_GENERATION_CEILING);
  return { proofKind: 'reservation-accounting-only', caseCompletionProven: false, coordinatorClosureId: closure.coordinatorClosureId, collectionManifestSha256: closure.collectionManifestSha256, ledgerId: approval.ledgerId, approvedPlanSha256: planHash(approval), generationCount, approvedTotal: approval.plan.generationInvocations.length, globalHardCeiling: GLOBAL_GENERATION_CEILING, unallocatedNotAuthorized: GLOBAL_GENERATION_CEILING - approval.plan.generationInvocations.length };
}
