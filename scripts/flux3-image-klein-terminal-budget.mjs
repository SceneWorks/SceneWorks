// No service/distributed allocation. One immutable GLOBAL plan assigns disjoint backend permits.
// Product/resource/resume sessions consume their backend's SAME stable local atomic ledger.
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import { createHash, randomBytes } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, open, readFile, readdir, rename, unlink, realpath, stat } from 'node:fs/promises';
import path from 'node:path';
import { validateNativeBoundaryReceipt } from './lib/flux3-native-boundary.mjs';
export const GLOBAL_GENERATION_CEILING = 48;
export const planHash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const sha = /^[a-f0-9]{64}$/, revision = /^[a-f0-9]{40}$/;
const kinds = ['product_generation', 'resource_anchor', 'resource_supplemental', 'warmup', 'repeat', 'retry'];
// Host lanes: the immutable plan MAY split one backend's permits across named hosts
// (plan.hostLanes[backend][host] = disjoint, covering permit/training IDs). Every lane is its own
// ledger (root, quota, input spec or addendum) under the SAME approval hash. An unsplit backend is
// one lane keyed by the backend itself, so its ledger is unchanged.
const hostPattern = /^[a-zA-Z0-9_.-]{1,64}$/;
export function ledgerLane(approval, backend, host) {
  assert(['mlx', 'candle'].includes(backend));
  const lanes = approval?.plan?.hostLanes?.[backend];
  if (!lanes) return backend;
  if (typeof host !== 'string' || !Object.hasOwn(lanes, host)) throw new Error(`BLOCKED: ${backend} permits are split by host; ${host ?? 'an unnamed host'} is not an approved ${backend} lane`);
  return `${backend}@${host}`;
}
export function ledgerLaneKeys(approval) {
  return ['mlx', 'candle'].flatMap(backend => { const lanes = approval?.plan?.hostLanes?.[backend]; return lanes ? Object.keys(lanes).sort().map(host => `${backend}@${host}`) : [backend]; });
}
/** True when `id` (a generation permit or training start/resume ID) belongs to the host's lane. */
export function inHostLane(plan, backend, host, id) {
  const lanes = plan?.hostLanes?.[backend];
  return !lanes || (Object.hasOwn(lanes, host ?? '') && lanes[host].includes(id));
}
function validateHostLanes(plan) {
  const hostLanes = plan.hostLanes; if (hostLanes === undefined) return;
  assert(hostLanes && typeof hostLanes === 'object' && !Array.isArray(hostLanes) && Object.keys(hostLanes).every(b => ['mlx', 'candle'].includes(b)), 'host lanes split only known backends');
  for (const [backend, lanes] of Object.entries(hostLanes)) {
    const hosts = Object.keys(lanes ?? {}); assert(hosts.length >= 1 && hosts.every(h => hostPattern.test(h) && Array.isArray(lanes[h])), `${backend} host lanes must name hosts with permit lists`);
    const expected = [...plan.generationInvocations.filter(i => i.backend === backend).map(i => i.invocationId), ...(plan.trainingLifecycles ?? []).filter(t => t.backend === backend).flatMap(t => [t.startId, t.resumeId])];
    const assigned = hosts.flatMap(h => lanes[h]);
    assert.equal(new Set(assigned).size, assigned.length, `${backend} host lanes must be disjoint: one permit is reservable on exactly one host`);
    assert.deepEqual([...assigned].sort(), [...expected].sort(), `${backend} host lanes must cover exactly that backend's permits`);
    const laneOf = id => hosts.find(h => lanes[h].includes(id));
    for (const t of (plan.trainingLifecycles ?? []).filter(t => t.backend === backend)) {
      const lane = laneOf(t.startId), reloads = plan.generationInvocations.filter(i => i.caseId === t.caseId);
      assert(laneOf(t.resumeId) === lane && reloads.every(i => laneOf(i.invocationId) === lane), `${t.lifecycleId}: a training lifecycle runs start, resume and reload on one host`);
      for (const i of plan.generationInvocations.filter(i => i.adapterBinding?.lifecycleId === t.lifecycleId)) assert.equal(laneOf(i.invocationId), lane, `${i.invocationId}: an adapter case runs on the host holding its lifecycle's export`);
    }
    for (const i of plan.generationInvocations.filter(i => i.backend === backend)) assert(plan.generationInvocations.filter(j => j.caseId === i.caseId).every(j => laneOf(j.invocationId) === laneOf(i.invocationId)), `${i.caseId}: one case runs on one host`);
  }
}
export function validateApprovedPlan(approval) {
  assert(approval?.schemaVersion === 1 && approval.kind === 'flux3-terminal-plan-approval', 'explicit immutable terminal plan approval required');
  assert(approval.approvalId && approval.userPlanApprovalId && approval.approvedBy, 'parent-recorded user plan approval identity required');
  if (Object.hasOwn(approval, 'candleSourceTransitionSha256')) assert(sha.test(approval.candleSourceTransitionSha256 ?? ''), 'valid root Candle transition authorization digest required');
  validateCandleDeviceTransitionBinding(approval);
  validateCandleMeasurementTransitionBinding(approval);
  assert(revision.test(approval.sourceSha ?? '') && revision.test(approval.inferenceMainSha ?? ''), 'approved final source/MAIN identities required');
  assert(/^[a-zA-Z0-9_-]{12,100}$/.test(approval.ledgerId ?? ''), 'stable non-resettable ledger identity required');
  // A backend whose inputs cannot exist yet (e.g. a second host still to be provisioned) may be
  // DEFERRED: its spec is absent here and only a parent-recorded addendum chained to this exact
  // approval can bind it later. At least one backend is bound at approval time.
  const plan = approval.plan; assert(plan?.kind === 'flux3-installed-product-representative-plan' && plan.diagnosticOnly !== true, 'diagnostic catalog is never an execution plan');
  const invocations = plan.generationInvocations; assert(Array.isArray(invocations) && invocations.length > 0 && invocations.length <= GLOBAL_GENERATION_CEILING, 'global expanded physical generation count must be 1..48');
  validateHostLanes(plan);
  // Every ledger lane (a backend, or a backend@host when the plan splits it) has its own fixed
  // root and either an approval-bound input spec or a deferred one bound later by addendum.
  const laneKeys = ledgerLaneKeys(approval), deferred = approval.deferredInputSpecBackends ?? [];
  assert(Array.isArray(deferred) && deferred.every(b => laneKeys.includes(b)) && new Set(deferred).size === deferred.length && deferred.length < laneKeys.length, 'deferred input specifications must name known ledger lanes, leaving at least one bound');
  assert.deepEqual(Object.keys(approval.ledgerRoots ?? {}).sort(), [...laneKeys].sort(), 'exactly one fixed ledger root per ledger lane');
  for (const lane of laneKeys) {
    assert(/^(?:[a-zA-Z]:[\\/]|\/|\\\\)/.test(approval.ledgerRoots[lane] ?? ''), 'fixed absolute per-backend ledger roots required');
    if (deferred.includes(lane)) assert(!Object.hasOwn(approval.inputSpecSha256 ?? {}, lane), `deferred ${lane} input specification cannot also be bound in the approval`);
    else assert(sha.test(approval.inputSpecSha256?.[lane] ?? ''), 'every non-deferred backend input specification must be bound before execution');
  }
  assert.equal(new Set(laneKeys.map(lane => approval.ledgerRoots[lane])).size, laneKeys.length, 'backend ledgers have distinct fixed roots, never independent duplicate quotas');
  const laneKeyOf = i => ledgerLane(approval, i.backend, Object.keys(plan.hostLanes?.[i.backend] ?? {}).find(h => plan.hostLanes[i.backend][h].includes(i.invocationId)));
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
      assert(!plan.hostLanes?.[row.backend], `${row.caseId}: a host-split backend's resource rows are proved per host by addendum`);
      assert(!deferred.includes(row.backend), `${row.caseId}: a deferred backend is READY only by addendum`);
      assert(row.parentNativeBoundaryVerificationId && sha.test(row.nativeBoundaryReceiptSha256 ?? ''), `${row.caseId}: READY requires the native-boundary verification receipt`);
      assert.equal(row.nativeBoundaryBinarySha256, approval.resourceBuildReceipts?.[row.backend]?.binarySha256, `${row.caseId}: native-boundary receipt came from a binary other than the approved sealed resource build`);
    } else assert(rowInvocations.every(i => deferred.includes(laneKeyOf(i))), `${row.caseId}: READY_ON_ADDENDUM is only for a deferred backend`);
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
// This is one fixed continuation, not a general source migration or quota-reset mechanism.
const CANDLE_PREVIOUS_SOURCE = '7090f742a7d2973dc3cc1c144aa8a272cdc293c9';
const CANDLE_REPLAY_CASE = 'candle.distilled.representative-repro.bf16';
const CANDLE_CONSUMED = `${CANDLE_REPLAY_CASE}.generation-1`, CANDLE_NEXT = `${CANDLE_REPLAY_CASE}.generation-2`;
const rawHash = bytes => createHash('sha256').update(bytes).digest('hex');
const canonical = value => JSON.stringify(canonicalValue(value));
function canonicalValue(value) {
  if (Array.isArray(value)) return Array.from(value, canonicalValue);
  if (value && typeof value === 'object') {
    assert.equal(Object.getPrototypeOf(value), Object.prototype, 'approval must contain plain JSON objects only');
    return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonicalValue(value[key])]));
  }
  assert(value === null || typeof value === 'string' || typeof value === 'boolean' || (typeof value === 'number' && Number.isFinite(value)), 'approval contains a non-JSON value');
  return value;
}
function exactKeys(value, keys, label) {
  assert(value && typeof value === 'object' && !Array.isArray(value), `${label} object required`);
  assert.deepEqual(Object.keys(value).sort(), [...keys].sort(), `${label} has unknown/missing fields`);
}
const DEVICE_TRANSITION_KEYS = ['schemaVersion','previousSourceSha','nextSourceSha','previousApprovedPlanSha256','previousHeadSha256','previousSequence','previousGenerationCount','previousConsumedIds','previousClaimSha256','previousDevice','nextDevice'];
const PREVIOUS_CANDLE_DEVICE = {physicalGpuId:'1',gpuUuid:'GPU-e4b79931-7be6-f216-460a-f5405cfafffe',pciBusId:'00000000:C1:00.0'};
const NEXT_CANDLE_DEVICE = {physicalGpuId:'0',gpuUuid:'GPU-b1a31911-c7b4-2901-3d8b-9a62e228bfc0',pciBusId:'00000000:21:00.0'};
/** Strict fixed device/source/prefix binding; absence preserves the legacy same-device format.
 * Does not itself authenticate the authorization: carried-prefix/append validation does that. */
export function validateCandleDeviceTransitionBinding(approval) {
  if (!Object.hasOwn(approval, 'candleDeviceTransition')) return undefined;
  const binding = approval.candleDeviceTransition;
  exactKeys(binding, DEVICE_TRANSITION_KEYS, 'Candle device transition');
  assert.equal(binding.schemaVersion, 1);
  assert(sha.test(approval.candleSourceTransitionSha256 ?? ''), 'device transition requires root source-transition authorization');
  assert.equal(binding.previousSourceSha, CANDLE_PREVIOUS_SOURCE, 'only the fixed original Candle source may transition devices');
  assert(revision.test(approval.sourceSha ?? '') && approval.sourceSha !== CANDLE_PREVIOUS_SOURCE, 'device transition needs a different approved source');
  assert.equal(binding.nextSourceSha, approval.sourceSha, 'device destination source differs from approval.sourceSha');
  for (const field of ['previousApprovedPlanSha256','previousHeadSha256','previousClaimSha256']) assert(sha.test(binding[field] ?? ''), `device transition ${field} required`);
  assert.equal(binding.previousSequence, 1); assert.equal(binding.previousGenerationCount, 1);
  assert.deepEqual(binding.previousConsumedIds, [CANDLE_CONSUMED]);
  for (const [field, expected] of [['previousDevice',PREVIOUS_CANDLE_DEVICE],['nextDevice',NEXT_CANDLE_DEVICE]]) {
    exactKeys(binding[field], ['physicalGpuId','gpuUuid','pciBusId'], `Candle ${field}`);
    assert.deepEqual(binding[field], expected, `Candle ${field} is not the fixed authorized physical device`);
  }
  return binding;
}
const MEASUREMENT_KIND = 'windows-owned-process-accounting-v1';
const MEASUREMENT_REASON = 'measurement-only owned-process producer/consumer accounting; renderer and inference unchanged';
const SECURITY_REASON = 'measurement owned-process producer/consumer accounting and RUSTSEC-2026-0386 Tokio security update; renderer source and inference pin unchanged';
const PREVIOUS_LOCK_SHA256 = '96feb71e685fa37faa01509108824463d09e88ed057ac7079ba0f11ea2b60e92';
const NEXT_LOCK_SHA256 = 'c33fba4fef58de889073666728c920a67c04687f7b24f4d4aaf5657cccd9a0c7';
const PREVIOUS_LOCK_GIT_BLOB = '93b2c6e0634b1493ae8621f85e5154fc25ca2aab';
const SECURITY_KEYS = ['schemaVersion','advisory','package','previousVersion','nextVersion','previousSourceSha','nextSourceSha','previousCargoLockSha256','nextCargoLockSha256'];
export function validateCandleDependencySecurityUpdate(approval) {
  const measurement = approval.candleMeasurementTransition;
  if (!measurement || !Object.hasOwn(measurement, 'dependencySecurityUpdate')) return undefined;
  const update = measurement.dependencySecurityUpdate;
  exactKeys(update, SECURITY_KEYS, 'fixed Candle dependency security update');
  assert.deepEqual(update, {schemaVersion:1,advisory:'RUSTSEC-2026-0386',package:'tokio',previousVersion:'1.53.1',nextVersion:'1.53.3',
    previousSourceSha:CANDLE_PREVIOUS_SOURCE,nextSourceSha:approval.sourceSha,previousCargoLockSha256:PREVIOUS_LOCK_SHA256,nextCargoLockSha256:NEXT_LOCK_SHA256}, 'only the fixed approved Tokio security update is permitted');
  for (const role of ['api','resource']) {
    const approved = approval[role === 'api' ? 'buildReceipts' : 'resourceBuildReceipts']?.candle;
    assert.equal(approved?.cargoLockSha256, NEXT_LOCK_SHA256, 'security destination build lock differs from fixed current lock');
  }
  return update;
}
/** Pure exact-byte/package validation: no other package/source/dependency/checksum can change. */
export function validateCandleDependencySecurityLocks(previousBytes, nextBytes) {
  const previous = Buffer.from(previousBytes), next = Buffer.from(nextBytes);
  assert.equal(rawHash(previous), PREVIOUS_LOCK_SHA256, 'historical lock is not the fixed original source lock');
  assert.equal(rawHash(next), NEXT_LOCK_SHA256, 'current lock is not the fixed Tokio security lock');
  const split = bytes => bytes.toString('utf8').split(/(?=^\[\[package\]\])/m);
  const before = split(previous), after = split(next);
  assert.equal(before.length, after.length, 'security update changed package count');
  let changed = 0;
  for (const [index, block] of before.entries()) {
    if (!/^name = "tokio"$/m.test(block)) { assert.equal(after[index], block, 'security update changed another package or lock metadata'); continue; }
    changed++;
    assert(/^version = "1\.53\.1"$/m.test(block));
    assert(/^checksum = "202caea871b69668250d242070849eb495be178ed697a3e98aebce5bc81a0bed"$/m.test(block));
    const expected = block.replace(/^version = "1\.53\.1"$/m, 'version = "1.53.3"')
      .replace(/^checksum = "202caea871b69668250d242070849eb495be178ed697a3e98aebce5bc81a0bed"$/m, 'checksum = "ce3335fa71841cda333a58d7615b03901380ecf09d59b3296d21f8bbac0dde4e"');
    assert.equal(after[index], expected, 'Tokio security update changed source/dependencies or an unapproved field');
  }
  assert.equal(changed, 1, 'exactly one fixed Tokio package update required');
  return true;
}
const OLD_CANDLE_BINARIES = {api:'d264bf21ba5e5a19012388041da92bc54764d2b3144c3d10065505b1a383e52d',resource:'4795a3e8fd5d0ee2d60c6586a72153759133209403d6b15120af7d50fcc48df5'};
/** Approval-bound fixed measurement producer/consumer replacement, never a generic binary remap.
 * Authentication and the previous approval's bindings are checked by compatibility/append. */
export function validateCandleMeasurementTransitionBinding(approval) {
  if (!Object.hasOwn(approval, 'candleMeasurementTransition')) return undefined;
  const binding = approval.candleMeasurementTransition, device = validateCandleDeviceTransitionBinding(approval);
  assert(device, 'measurement transition requires the fixed device/source transition');
  const security = validateCandleDependencySecurityUpdate(approval);
  exactKeys(binding, ['schemaVersion','previousSourceSha','nextSourceSha','measurementKind','reason','previousBuilds','nextBuilds', ...(security ? ['dependencySecurityUpdate'] : [])], 'Candle measurement transition');
  assert.equal(binding.schemaVersion, 1); assert.equal(binding.measurementKind, MEASUREMENT_KIND); assert.equal(binding.reason, security ? SECURITY_REASON : MEASUREMENT_REASON);
  assert.equal(binding.previousSourceSha, CANDLE_PREVIOUS_SOURCE); assert.equal(binding.nextSourceSha, approval.sourceSha);
  for (const side of ['previousBuilds','nextBuilds']) {
    exactKeys(binding[side], ['api','resource'], `measurement ${side}`);
    for (const role of ['api','resource']) {
      const build = binding[side][role]; exactKeys(build, ['binarySha256','receiptSha256'], `measurement ${side} ${role}`);
      assert(sha.test(build.binarySha256 ?? '') && sha.test(build.receiptSha256 ?? ''), 'measurement binary and producer receipt digests required');
      if (side === 'previousBuilds') assert.equal(build.binarySha256, OLD_CANDLE_BINARIES[role], 'only the fixed original Candle binaries may transition');
      else {
        const approved = approval[role === 'api' ? 'buildReceipts' : 'resourceBuildReceipts']?.candle;
        assert.equal(build.binarySha256, approved?.binarySha256, 'measurement destination binary differs from approved build');
        assert.equal(build.receiptSha256, approved?.receiptSha256, 'measurement destination producer receipt differs from approved build');
      }
    }
  }
  assert.notEqual(binding.nextBuilds.resource.binarySha256, binding.previousBuilds.resource.binarySha256, 'measurement transition must replace the resource measurement binary');
  return binding;
}
/** Hash ALL approval fields canonically, omitting ONLY the root authorization digest. */
export function candleNextApprovalBaseSha256(approval) {
  const { candleSourceTransitionSha256, ...base } = approval;
  return rawHash(`flux3-candle-next-approval-base-v1\n${canonical(base)}`);
}
export function validateCandleSourceCompatibility(previousApproval, approval, authorization, previousReceipt) {
  validateApprovedPlan(previousApproval); validateApprovedPlan(approval);
  assert.equal(previousApproval.sourceSha, CANDLE_PREVIOUS_SOURCE, 'only the fixed original Candle source may transition');
  assert(!Object.hasOwn(previousApproval, 'candleSourceTransitionSha256'), 'a second Candle source transition is forbidden');
  assert(sha.test(approval.candleSourceTransitionSha256 ?? ''), 'parent authorization raw SHA required');
  assert.notEqual(approval.sourceSha, previousApproval.sourceSha, 'new source must actually differ');
  assert(!previousApproval.plan.hostLanes?.candle && !approval.plan.hostLanes?.candle, 'fixed Candle prefix cannot migrate host lanes');
  for (const a of [previousApproval, approval]) {
    assert(!(a.deferredInputSpecBackends ?? []).includes('candle'), 'fixed Candle source prefix must have approval-bound inputs');
    validateNativeBoundaryReceipt(a.plan.nativeBoundaryReceipts?.candle, {backend:'candle',invocations:a.plan.generationInvocations.filter(i => i.backend === 'candle' && i.kind.startsWith('resource_')),
      binarySha256:a.resourceBuildReceipts?.candle?.binarySha256,buildReceiptSha256:a.resourceBuildReceipts?.candle?.receiptSha256,sourceSha:a.sourceSha,inferenceMainSha:a.inferenceMainSha});
  }
  const deviceBinding = validateCandleDeviceTransitionBinding(approval), measurementBinding = validateCandleMeasurementTransitionBinding(approval);
  const security = validateCandleDependencySecurityUpdate(approval);
  if (security) for (const key of ['buildReceipts','resourceBuildReceipts']) assert.equal(previousApproval[key]?.candle?.cargoLockSha256, PREVIOUS_LOCK_SHA256, 'security original build lock differs from fixed old source lock');
  if (measurementBinding) for (const role of ['api','resource']) {
    const previous = previousApproval[role === 'api' ? 'buildReceipts' : 'resourceBuildReceipts']?.candle;
    assert.equal(measurementBinding.previousBuilds[role].binarySha256, previous?.binarySha256, 'measurement original binary differs from old approval');
    assert.equal(measurementBinding.previousBuilds[role].receiptSha256, previous?.receiptSha256, 'measurement original producer receipt differs from old approval');
  }
  if (deviceBinding) {
    // Compatibility may omit the new field ONLY after proving its exact authorization and
    // authentic old prefix. The append path additionally verifies the immutable claim bytes.
    verifyLedgerReceipt(previousApproval, previousReceipt, 'candle');
    validateTransitionAuthorization(authorization, previousApproval, approval, previousReceipt);
  }
  const scrub = (a, next) => {
    const result = structuredClone(a); delete result.approvalId; delete result.sourceSha;
    if (next) {
      delete result.candleSourceTransitionSha256;
      if (deviceBinding) delete result.candleDeviceTransition;
      if (measurementBinding) delete result.candleMeasurementTransition;
    }
    for (const key of ['buildReceipts', 'resourceBuildReceipts']) {
      assert(result[key]?.candle, 'both sealed Candle builds required');
      delete result[key].candle.receiptSha256; delete result[key].candle.transcriptSha256;
      if (measurementBinding) delete result[key].candle.binarySha256;
      if (security) delete result[key].candle.cargoLockSha256;
    }
    if (result.plan.nativeBoundaryReceipts?.candle) {
      delete result.plan.nativeBoundaryReceipts.candle.sourceSha;
      delete result.plan.nativeBoundaryReceipts.candle.buildReceiptSha256;
      if (measurementBinding) delete result.plan.nativeBoundaryReceipts.candle.binarySha256;
    }
    for (const row of result.plan.resourceCases ?? []) if (row.backend === 'candle') {
      delete row.nativeBoundaryReceiptSha256; delete row.parentNativeBoundaryVerificationId;
      if (measurementBinding) delete row.nativeBoundaryBinarySha256;
    }
    return result;
  };
  assert.deepEqual(scrub(approval, true), scrub(previousApproval, false), 'source transition changed immutable plan/input/quota/build/root fields or introduced an unknown field');
  return true;
}
const AUTH_KEYS = ['schemaVersion','kind','authorizationId','approvedBy','producersStopped','previousSourceSha','previousApprovedPlanSha256','previousHeadSha256','previousSequence','previousGenerationCount','previousConsumedIds','previousClaimSha256','previousApproval','previousReceipt','previousBuilds','nextBuilds','nextSourceSha','nextApprovalBaseSha256','baseline'];
function transitionAuthorizationKeys(approval) {
  return [...AUTH_KEYS, ...(validateCandleDeviceTransitionBinding(approval) ? ['deviceTransition'] : []), ...(validateCandleMeasurementTransitionBinding(approval) ? ['measurementTransition'] : []), ...(validateCandleDependencySecurityUpdate(approval) ? ['previousCargoLock'] : [])];
}
const BASELINE_KEYS = ['caseId','consumedInvocationId','nextInvocationId','png','sidecar','report','readback','firstRequestSha256','secondRecordedRequestSha256','oldJobId','oldWorkerId','oldWorkerPid','allowedLineageRemap'];
function artifactBinding(binding) {
  exactKeys(binding, ['path','sha256'], 'artifact binding');
  assert(typeof binding.path === 'string' && path.isAbsolute(binding.path) && sha.test(binding.sha256), 'absolute owned artifact and raw digest required');
}
function validateTransitionAuthorization(authorization, previousApproval, approval, previousReceipt) {
  const deviceBinding = validateCandleDeviceTransitionBinding(approval);
  exactKeys(authorization, transitionAuthorizationKeys(approval), 'source transition authorization');
  const measurementBinding = validateCandleMeasurementTransitionBinding(approval);
  if (validateCandleDependencySecurityUpdate(approval)) {
    artifactBinding(authorization.previousCargoLock);
    assert.equal(authorization.previousCargoLock.sha256, PREVIOUS_LOCK_SHA256, 'historical lock binding differs from fixed Git source');
    assert.equal(path.basename(authorization.previousCargoLock.path), 'Cargo.lock', 'historical metadata root must contain exact Cargo.lock');
  }
  if (measurementBinding) {
    assert.deepEqual(authorization.measurementTransition, measurementBinding, 'authorization measurement transition differs from approval');
    for (const [side, builds] of [['previousBuilds', authorization.previousBuilds], ['nextBuilds', authorization.nextBuilds]]) for (const role of ['api','resource']) {
      assert.equal(measurementBinding[side][role].receiptSha256, builds?.[role]?.sha256, 'measurement producer receipt differs from authorized artifact');
    }
  }
  if (deviceBinding) {
    assert.deepEqual(authorization.deviceTransition, deviceBinding, 'authorization device transition differs from approval');
    for (const field of ['previousSourceSha','nextSourceSha','previousApprovedPlanSha256','previousHeadSha256','previousSequence','previousGenerationCount','previousConsumedIds','previousClaimSha256']) {
      assert.deepEqual(deviceBinding[field], authorization[field], `device transition ${field} differs from authorized old prefix`);
    }
  }
  assert.equal(authorization.schemaVersion, 1); assert.equal(authorization.kind, 'flux3-candle-source-transition-authorization');
  for (const field of ['authorizationId','approvedBy']) assert(typeof authorization[field] === 'string' && authorization[field].length);
  assert.equal(authorization.producersStopped, true, 'stopped-producer parent authorization required');
  assert.equal(authorization.previousSourceSha, CANDLE_PREVIOUS_SOURCE);
  assert.equal(authorization.previousSourceSha, previousApproval.sourceSha);
  assert.equal(authorization.nextSourceSha, approval.sourceSha);
  assert.equal(authorization.previousApprovedPlanSha256, planHash(previousApproval));
  assert.equal(authorization.nextApprovalBaseSha256, candleNextApprovalBaseSha256(approval), 'next approval base digest mismatch');
  assert.equal(authorization.previousSequence, 1); assert.equal(authorization.previousGenerationCount, 1);
  assert.deepEqual(authorization.previousConsumedIds, [CANDLE_CONSUMED]);
  assert.deepEqual(previousReceipt.state, {sequence:1,generationCount:1,consumedIds:[CANDLE_CONSUMED],headSha256:authorization.previousHeadSha256}, 'authorization does not name exact fixed old prefix');
  for (const field of ['previousHeadSha256','previousClaimSha256']) assert(sha.test(authorization[field] ?? ''));
  for (const field of ['previousApproval','previousReceipt']) artifactBinding(authorization[field]);
  for (const field of ['previousBuilds','nextBuilds']) { exactKeys(authorization[field], ['api','resource'], field); for (const binding of Object.values(authorization[field])) artifactBinding(binding); }
  const baseline = authorization.baseline; exactKeys(baseline, BASELINE_KEYS, 'baseline');
  assert.equal(baseline.caseId, CANDLE_REPLAY_CASE); assert.equal(baseline.consumedInvocationId, CANDLE_CONSUMED); assert.equal(baseline.nextInvocationId, CANDLE_NEXT);
  assert.deepEqual(baseline.allowedLineageRemap, ['projectId']);
  for (const field of ['png','sidecar','report','readback']) artifactBinding(baseline[field]);
  for (const field of ['firstRequestSha256','secondRecordedRequestSha256']) assert(sha.test(baseline[field] ?? ''));
  for (const field of ['oldJobId','oldWorkerId']) assert(typeof baseline[field] === 'string' && baseline[field].length);
  assert(Number.isSafeInteger(baseline.oldWorkerPid) && baseline.oldWorkerPid > 0);
  for (const id of [CANDLE_CONSUMED, CANDLE_NEXT]) assert(approval.plan.generationInvocations.some(i => i.invocationId === id && i.caseId === CANDLE_REPLAY_CASE && i.backend === 'candle' && i.kind === 'product_generation'), 'fixed replay pair missing');
}
const TRANSITION_KEYS = ['sequence','previousSha256','approvedPlanSha256','ledgerId','backend','kind','previousApprovalJson','previousReceiptJson','authorizationJson','authorizationSha256'];
function transitionPrefix(approval, event) {
  const { eventSha256, ...record } = event; exactKeys(record, TRANSITION_KEYS, 'Candle transition event');
  assert.equal(eventSha256, planHash(record)); assert.equal(record.kind, 'candle_source_transition'); assert.equal(record.backend, 'candle'); assert.equal(record.sequence, 2);
  assert.equal(record.ledgerId, approval.ledgerId); assert.equal(record.approvedPlanSha256, planHash(approval));
  assert(sha.test(approval.candleSourceTransitionSha256 ?? ''), 'unknown/unapproved Candle source transition');
  for (const field of ['previousApprovalJson','previousReceiptJson','authorizationJson']) assert(typeof record[field] === 'string');
  assert.equal(rawHash(record.authorizationJson), approval.candleSourceTransitionSha256, 'authorization raw bytes differ from approval');
  assert.equal(record.authorizationSha256, approval.candleSourceTransitionSha256);
  const authorization = JSON.parse(record.authorizationJson), previousApproval = JSON.parse(record.previousApprovalJson), previousReceipt = JSON.parse(record.previousReceiptJson);
  validateCandleSourceCompatibility(previousApproval, approval, authorization, previousReceipt);
  assert.equal(rawHash(record.previousApprovalJson), authorization.previousApproval.sha256, 'old approval raw bytes mismatch');
  assert.equal(rawHash(record.previousReceiptJson), authorization.previousReceipt.sha256, 'old final receipt raw bytes mismatch');
  verifyLedgerReceipt(previousApproval, previousReceipt, 'candle');
  validateTransitionAuthorization(authorization, previousApproval, approval, previousReceipt);
  assert.equal(record.previousSha256, previousReceipt.state.headSha256);
  return {authorization, previousApproval, previousReceipt, authorizationSha256:record.authorizationSha256, consumedInvocationId:CANDLE_CONSUMED, nextInvocationId:CANDLE_NEXT};
}
/** Pure authenticated carried-prefix lookup. No generic skip-consumed authority. */
export function candleReplayPrefix(receipt, approval) {
  verifyLedgerReceipt(approval, receipt, 'candle');
  const event = receipt.events.find(e => e.kind === 'candle_source_transition');
  assert(event, 'authenticated Candle source transition required for replay prefix');
  return transitionPrefix(approval, event);
}
async function ownedArtifact(binding, ownedRoot, text = true) {
  artifactBinding(binding);
  // Reject foreign/UNC paths BEFORE any filesystem access to the supplied path.
  const lexicalRelative = path.relative(path.resolve(ownedRoot), path.resolve(binding.path));
  assert(lexicalRelative && lexicalRelative !== '..' && !lexicalRelative.startsWith(`..${path.sep}`) && !path.isAbsolute(lexicalRelative), 'artifact escaped lexical owned root');
  const root = await realpath(ownedRoot), file = await realpath(binding.path), relative = path.relative(root, file);
  assert(relative && relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative), 'artifact escaped owned root');
  assert((await stat(file)).isFile(), 'owned regular artifact required');
  if (!text) {
    const hash = createHash('sha256'); for await (const chunk of createReadStream(file)) hash.update(chunk);
    assert.equal(hash.digest('hex'), binding.sha256, 'owned artifact raw bytes mismatch'); return;
  }
  const bytes = await readFile(file); assert.equal(rawHash(bytes), binding.sha256, 'owned artifact raw bytes mismatch'); return bytes.toString('utf8');
}
export async function verifyCandleDependencySecurityArchive(approval, authorization, ownedRoot) {
  if (!validateCandleDependencySecurityUpdate(approval)) return undefined;
  const archived = Buffer.from(await ownedArtifact(authorization.previousCargoLock, ownedRoot), 'utf8');
  const sourceRoot = fileURLToPath(new URL('..', import.meta.url));
  // Fixed command/commit/path: caller artifacts cannot select a Git epoch or repository.
  const {stdout} = await promisify(execFile)('git', ['-C', sourceRoot, 'show', `${CANDLE_PREVIOUS_SOURCE}:Cargo.lock`], {encoding:'buffer',maxBuffer:16*1024*1024});
  assert.equal(createHash('sha1').update(`blob ${stdout.length}\0`).update(stdout).digest('hex'), PREVIOUS_LOCK_GIT_BLOB, 'original source lock Git blob changed');
  assert.deepEqual(archived, stdout, 'archived lock differs from immutable original source Git blob');
  validateCandleDependencySecurityLocks(archived, await readFile(new URL('../Cargo.lock', import.meta.url)));
  const file = await realpath(authorization.previousCargoLock.path);
  assert.equal(path.basename(file), 'Cargo.lock', 'historical metadata must resolve to Cargo.lock');
  return path.dirname(file);
}
// previousBuilds/nextBuilds api/resource bindings name the RAW producer-receipt.json:
// use produceTerminalBuild's receiptPath + receiptSha256, never an outer *-build-result.json.
async function verifyTransitionBuilds(approval, bindings, ownedRoot, historicalSourceRoot) {
  const { verifyTerminalBuild } = await import('./flux3-image-klein-terminal-build.mjs');
  const results = {};
  for (const [key, role] of [['api','api-worker'],['resource','resource-adapter']]) {
    const json = await ownedArtifact(bindings[key], ownedRoot), receipt = JSON.parse(json);
    // Check the binary is also owned before delegating the FULL real verifier (default checked code root).
    await ownedArtifact({path:receipt.binary.path,sha256:receipt.binary.sha256}, ownedRoot, false);
    const verified = await verifyTerminalBuild({approvedPlan:approval,backend:'candle',sourceSha:approval.sourceSha,inferenceMainSha:approval.inferenceMainSha,
      binaryBuildReceipt:{...bindings[key],root:ownedRoot},binary:receipt.binary.path,binarySha256:receipt.binary.sha256}, {role,...(historicalSourceRoot ? {sourceRoot:historicalSourceRoot} : {})});
    const transcript = JSON.parse(await ownedArtifact({path:path.join(path.dirname(bindings[key].path), 'producer-transcript.json'),sha256:verified.transcriptSha256}, ownedRoot));
    results[key] = {receipt:verified,receiptSha256:rawHash(json),transcript};
  }
  return results;
}
export async function appendCandleSourceTransition({approval, ledgerRoot, continuation}) {
  const root = location(approval, 'candle', ledgerRoot), ownedRoot = path.resolve(root, '../..');
  assert(sha.test(approval.candleSourceTransitionSha256 ?? ''), 'new approval must pin parent authorization');
  assert.equal(continuation?.sha256, approval.candleSourceTransitionSha256, 'authorization differs from approved raw SHA');
  const authorizationJson = await ownedArtifact(continuation, ownedRoot), authorization = JSON.parse(authorizationJson);
  // Validate structure before following any artifact paths.
  exactKeys(authorization, transitionAuthorizationKeys(approval), 'source transition authorization');
  const previousApprovalJson = await ownedArtifact(authorization.previousApproval, ownedRoot), previousApproval = JSON.parse(previousApprovalJson);
  const previousReceiptJson = await ownedArtifact(authorization.previousReceipt, ownedRoot), previousReceipt = JSON.parse(previousReceiptJson);
  validateCandleSourceCompatibility(previousApproval, approval, authorization, previousReceipt); verifyLedgerReceipt(previousApproval, previousReceipt, 'candle');
  validateTransitionAuthorization(authorization, previousApproval, approval, previousReceipt);
  const claimFile = path.join(path.dirname(root), `.flux3-${approval.ledgerId}-candle.initialized`);
  const claimJson = await ownedArtifact({path:claimFile,sha256:authorization.previousClaimSha256}, ownedRoot);
  assert.deepEqual(JSON.parse(claimJson), {ledgerId:previousApproval.ledgerId,backend:'candle',root,approvedPlanSha256:planHash(previousApproval)}, 'old initialization claim mismatch');
  const historicalSourceRoot = await verifyCandleDependencySecurityArchive(approval, authorization, ownedRoot);
  const beforeBuilds = await verifyTransitionBuilds(previousApproval, authorization.previousBuilds, ownedRoot, historicalSourceRoot), afterBuilds = await verifyTransitionBuilds(approval, authorization.nextBuilds, ownedRoot);
  for (const key of ['api','resource']) {
    const before = beforeBuilds[key], after = afterBuilds[key];
    assert.deepEqual(before.receipt.recipe, after.receipt.recipe, 'build recipe changed');
    const measurement = validateCandleMeasurementTransitionBinding(approval);
    if (measurement) {
      for (const [side, build] of [['previousBuilds',before],['nextBuilds',after]]) {
        assert.equal(build.receipt.binary.sha256, measurement[side][key].binarySha256, 'actual producer binary differs from measurement transition');
        assert.equal(build.receiptSha256, measurement[side][key].receiptSha256, 'actual producer receipt differs from measurement transition');
      }
    } else assert.equal(before.receipt.binary.sha256, after.receipt.binary.sha256, 'sealed binary changed');
    for (const field of ['recipe','cargoExecutableSha256','rustcExecutableSha256','rustcSha256','rustflags','msvcVersion','msvcObserved','cudaObserved']) assert.deepEqual(before.transcript[field], after.transcript[field], `build toolchain ${field} changed`);
    if (historicalSourceRoot) {
      for (const [build, expected] of [[before,PREVIOUS_LOCK_SHA256],[after,NEXT_LOCK_SHA256]]) {
        assert.equal(build.receipt.cargoLockSha256, expected, 'actual producer security lock mismatch');
        assert.equal(build.transcript.sourceBefore.cargoLockSha256, expected, 'actual transcript security lock mismatch');
      }
    } else assert.equal(before.transcript.sourceBefore.cargoLockSha256, after.transcript.sourceBefore.cargoLockSha256, 'build cargoLockSha256 changed');
    assert.deepEqual(before.transcript.sourceBefore.manifestSha256, after.transcript.sourceBefore.manifestSha256, 'build manifestSha256 changed');
  }
  for (const binding of Object.values(authorization.baseline).filter(v => v && typeof v === 'object' && !Array.isArray(v))) await ownedArtifact(binding, ownedRoot);
  const { validateReplayPrefixBaseline } = await import('./flux3-image-klein-terminal-driver.mjs');
  const baseline = await validateReplayPrefixBaseline({previousApproval,previousReceipt,authorization,ownedRoot});
  return lock(root, async () => {
    // The claim is immutable too: proof validation may take time, so re-read its exact bytes
    // under the SAME consume lock before either append or idempotent acceptance.
    const lockedClaimJson = await ownedArtifact({path:claimFile,sha256:authorization.previousClaimSha256}, ownedRoot);
    assert.equal(lockedClaimJson, claimJson, 'old initialization claim changed during proof validation');
    // Re-read and verify under consume.lock; never reconcile partial journal/state.
    const files = (await readdir(path.join(root, 'events'))).sort();
    if (files.length === 2) {
      const receipt = await audited(approval, 'candle', root), prefix = candleReplayPrefix(receipt, approval);
      assert.equal(receipt.state.sequence, 2); assert.equal(receipt.state.generationCount, 1);
      assert.deepEqual(receipt.state.consumedIds, [CANDLE_CONSUMED]);
      assert.equal(prefix.authorizationSha256, continuation.sha256);
      return {receipt,authorization,previousApproval,previousReceipt,baseline,eventSha256:receipt.state.headSha256,alreadyApplied:true};
    }
    const current = await audited(previousApproval, 'candle', root);
    assert.deepEqual(current, previousReceipt, 'old latest ledger differs from authorized FINAL receipt');
    assert.equal(current.state.sequence, 1, 'transition only at fixed one-spend prefix');
    const record = {sequence:2,previousSha256:current.state.headSha256,approvedPlanSha256:planHash(approval),ledgerId:approval.ledgerId,backend:'candle',kind:'candle_source_transition',previousApprovalJson,previousReceiptJson,authorizationJson,authorizationSha256:continuation.sha256};
    const event = {...record,eventSha256:planHash(record)}, next = {...current.state,sequence:2,headSha256:event.eventSha256};
    const receipt = verifyLedgerReceipt(approval, {...current,state:next,events:[...current.events,event]}, 'candle');
    await durableNew(path.join(root, 'events', '000002.json'), event);
    const temporary = path.join(root, `state-${randomBytes(16).toString('hex')}.tmp`); await durableNew(temporary, next); await rename(temporary, path.join(root, 'state.json'));
    return {receipt,authorization,previousApproval,previousReceipt,baseline,eventSha256:event.eventSha256,alreadyApplied:false};
  });
}
const ADDENDUM_KEYS = ['schemaVersion', 'kind', 'approvedPlanSha256', 'ledgerId', 'backend', 'inputSpecSha256', 'buildReceipt', 'resourceBuildReceipt', 'nativeBoundaryReceipt', 'parentNativeBoundaryVerificationId', 'addendumId', 'userApprovalId', 'approvedBy'].sort();
/** Validate a parent-recorded input-spec addendum. It can ONLY bind a deferred backend's input
 *  specification for this exact approval: it never changes the plan, any bound spec, the ledger
 *  identity or the counts, and an addendum for any other approval is refused. */
export function validateInputSpecAddendum(approval, addendum) {
  const base = validateApprovedPlan(approval);
  // A host-split backend's addendum binds exactly ONE host lane, named by its `lane` (host) field.
  const laned = Boolean(approval.plan.hostLanes?.[addendum?.backend]), keys = laned ? [...ADDENDUM_KEYS, 'lane'].sort() : ADDENDUM_KEYS;
  assert(addendum && typeof addendum === 'object' && JSON.stringify(Object.keys(addendum).sort()) === JSON.stringify(keys), 'addendum carries exactly its binding fields, nothing that could amend the plan, ledger or counts');
  assert(addendum.schemaVersion === 1 && addendum.kind === 'flux3-terminal-plan-input-spec-addendum', 'explicit input-spec addendum required');
  assert.equal(addendum.approvedPlanSha256, base.approvedPlanSha256, 'addendum is chained to a different approved plan');
  assert.equal(addendum.ledgerId, approval.ledgerId, 'addendum ledger identity differs from the approval');
  assert(['mlx', 'candle'].includes(addendum.backend), 'addendum must name a known backend');
  const lane = ledgerLane(approval, addendum.backend, laned ? addendum.lane : undefined);
  assert((approval.deferredInputSpecBackends ?? []).includes(lane), 'only a deferred backend input specification can be bound by addendum');
  assert(sha.test(addendum.inputSpecSha256 ?? ''), 'addendum must bind an input specification digest');
  for (const key of ['addendumId', 'userApprovalId', 'approvedBy', 'parentNativeBoundaryVerificationId']) assert(typeof addendum[key] === 'string' && addendum[key].length > 0, `addendum ${key} required`);
  // The deferred host's own sealed builds, bound like the approval's bindings for a non-deferred
  // backend; never a replacement for build bindings the approval already carries.
  for (const key of ['buildReceipt', 'resourceBuildReceipt']) assert(addendum[key] && sha.test(addendum[key].receiptSha256 ?? '') && sha.test(addendum[key].binarySha256 ?? ''), `addendum ${key} must bind a sealed build receipt and binary`);
  assert(!approval.buildReceipts?.[addendum.backend] && !approval.resourceBuildReceipts?.[addendum.backend], 'an addendum cannot replace build bindings the approval already carries');
  // Its READY_ON_ADDENDUM resource rows become reservable only with a valid self-test receipt from
  // exactly this sealed resource binary, proving exactly the plan's fixed request digests.
  const invocations = approval.plan.generationInvocations.filter(i => i.backend === addendum.backend && i.kind.startsWith('resource_')
    && inHostLane(approval.plan, i.backend, addendum.lane, i.invocationId)
    && approval.plan.resourceCases.find(r => r.caseId === i.caseId && r.backend === i.backend)?.status === 'READY_ON_ADDENDUM');
  const nativeBoundaryReceiptSha256 = invocations.length ? validateNativeBoundaryReceipt(addendum.nativeBoundaryReceipt, { backend: addendum.backend, invocations,
    binarySha256: addendum.resourceBuildReceipt.binarySha256, buildReceiptSha256: addendum.resourceBuildReceipt.receiptSha256,
    sourceSha: approval.sourceSha, inferenceMainSha: approval.inferenceMainSha }) : null;
  return { backend: addendum.backend, ...(laned ? { lane: addendum.lane } : {}), ledgerLane: lane, inputSpecSha256: addendum.inputSpecSha256, approvedPlanSha256: base.approvedPlanSha256, nativeBoundaryReceiptSha256,
    buildReceipt: addendum.buildReceipt, resourceBuildReceipt: addendum.resourceBuildReceipt };
}
/** The input-spec digest a backend (on `host`, when the plan splits it by host) may run with. A
 *  deferred lane stays unusable (BLOCKED) until its addendum exists; the approval's own hash and
 *  ledgers are unchanged by it. */
export function boundInputSpecSha256(approval, backend, addendum, host) {
  const lane = ledgerLane(approval, backend, host);
  // Callers validate the approval itself; the addendum path re-validates it in full below.
  if (!(approval.deferredInputSpecBackends ?? []).includes(lane)) {
    assert(addendum === undefined || addendum === null, `${backend} input specification is already bound by the approval; no addendum may replace it`);
    return approval.inputSpecSha256[lane];
  }
  if (!addendum) throw new Error(`BLOCKED: ${lane} input specification is deferred; its permits stay unusable until a parent-recorded addendum binds it`);
  const bound = validateInputSpecAddendum(approval, addendum); assert.equal(bound.backend, backend, 'addendum binds a different backend');
  assert.equal(bound.ledgerLane, lane, 'BLOCKED: addendum binds a different host lane');
  return bound.inputSpecSha256;
}
/** The host an admission runs its backend's permits on. A host-split backend admits only its plan
 *  lanes; an unsplit MLX backend keeps its single admitted secondary Mac. */
export function admittedHost(approval, admission) {
  const lanes = approval.plan.hostLanes?.[admission.backend];
  if (lanes) { ledgerLane(approval, admission.backend, admission.runner); return admission.runner; }
  if (admission.backend === 'mlx') assert.equal(admission.runner, 'nax-macos-2', 'only secondary nax-macos-2 admitted');
  return undefined;
}
function location(approval, backend, ledgerRoot, host) {
  validateApprovedPlan(approval); const lane = ledgerLane(approval, backend, host);
  assert.equal(path.resolve(ledgerRoot), path.resolve(approval.ledgerRoots[lane]), 'alternate ledger root/reset is forbidden'); return path.resolve(ledgerRoot);
}
function expectedManifest(approval, backend, host) {
  const laned = Boolean(approval.plan.hostLanes?.[backend]), own = id => inHostLane(approval.plan, backend, host, id);
  return { schemaVersion: 1, ledgerId: approval.ledgerId, backend, ...(laned ? { host } : {}), approvedPlanSha256: planHash(approval), approvalId: approval.approvalId, userPlanApprovalId: approval.userPlanApprovalId, permitIds: approval.plan.generationInvocations.filter(i => i.backend === backend && own(i.invocationId)).map(i => i.invocationId), trainingIds: approval.plan.trainingLifecycles.filter(t => t.backend === backend).flatMap(t => [t.startId, t.resumeId]).filter(own) };
}
async function durableNew(file, value) { const h = await open(file, 'wx'); try { await h.writeFile(JSON.stringify(value) + '\n'); await h.sync(); } finally { await h.close(); } }
// Explicit, separately invoked CPU setup ONLY. Runtime never initializes or resets a ledger.
// The durable sibling initialization claim survives deletion of the ledger directory.
export async function initializeLedger({ approval, backend, ledgerRoot, host }) {
  const root = location(approval, backend, ledgerRoot, host), hash = planHash(approval), lane = ledgerLane(approval, backend, host);
  assert(!(backend === 'candle' && Object.hasOwn(approval, 'candleSourceTransitionSha256')), 'Candle transition approval cannot initialize a fresh ledger');
  await durableNew(path.join(path.dirname(root), `.flux3-${approval.ledgerId}-${lane}.initialized`), { ledgerId: approval.ledgerId, backend, ...(lane !== backend ? { host } : {}), root, approvedPlanSha256: hash });
  await mkdir(root); await mkdir(path.join(root, 'events'));
  const manifest = expectedManifest(approval, backend, host);
  await durableNew(path.join(root, 'manifest.json'), manifest);
  await durableNew(path.join(root, 'state.json'), { sequence: 0, generationCount: 0, consumedIds: [], headSha256: planHash(manifest) });
  return { ledgerId: approval.ledgerId, backend, ...(lane !== backend ? { host } : {}), approvedPlanSha256: hash, allocated: manifest.permitIds.length };
}
async function lock(root, callback) {
  let handle; try { handle = await open(path.join(root, 'consume.lock'), 'wx'); } catch { throw new Error('BLOCKED: existing stable ledger missing/busy; no initialization/reset/stale-lock recovery is permitted at runtime'); }
  try { return await callback(); } finally { await handle.close(); await unlink(path.join(root, 'consume.lock')); }
}
function approvedEvent(approval, event, addendum, host) {
  // Wrong-host refusal: a permit assigned to another host lane is never reservable here.
  if (!inHostLane(approval.plan, event.backend, host, event.invocationId)) throw new Error(`BLOCKED: ${event.invocationId} is assigned to another ${event.backend} host lane, not ${host}; no permit is reservable on the wrong host`);
  if (event.kind === 'training_start' || event.kind === 'training_resume') {
    const training = approval.plan.trainingLifecycles.find(t => t.backend === event.backend && t.caseId === event.caseId && (event.kind === 'training_start' ? t.startId : t.resumeId) === event.invocationId); assert(training, 'unplanned training start/resume'); return false;
  }
  const invocation = approval.plan.generationInvocations.find(i => i.invocationId === event.invocationId);
  assert(invocation && invocation.backend === event.backend && invocation.caseId === event.caseId && invocation.kind === event.kind, 'invocation absent from this immutable backend quota; no spare allocation or automatic retry');
  if (invocation.kind.startsWith('resource_')) {
    const resource = approval.plan.resourceCases.find(c => c.caseId === invocation.caseId && c.backend === invocation.backend);
    if (resource?.status === 'READY_ON_ADDENDUM') {
      assert(addendum, 'BLOCKED: this resource row is reservable only after its backend addendum binds a validated native-boundary self-test receipt');
      const bound = validateInputSpecAddendum(approval, addendum); assert.equal(bound.backend, invocation.backend, 'BLOCKED: addendum binds a different backend');
      assert.equal(bound.ledgerLane, ledgerLane(approval, invocation.backend, host), 'BLOCKED: addendum binds a different host lane');
    } else assert(resource?.status === 'READY' && resource.parentNativeBoundaryVerificationId && sha.test(resource.nativeBoundaryReceiptSha256 ?? ''), 'BLOCKED: native inner-generate boundary is unverified; a contract flag is not proof. Authorized owner/coordinator must bind the real single-call verification receipt before approving this immutable plan.');
    assert(sha.test(invocation.requestSha256 ?? ''), 'resource request must be normalized/hash-bound in the approved plan');
  }
  if (invocation.requestSha256) assert.equal(event.requestSha256, invocation.requestSha256, 'resource permit must bind exact normalized approved request');
  return true;
}
export function verifyLedgerReceipt(approval, receipt, expectedBackend = receipt?.manifest?.backend, inputSpecAddendum, host = receipt?.manifest?.host) {
  validateApprovedPlan(approval); assert(['mlx', 'candle'].includes(expectedBackend), 'complete ledger manifest/backend required');
  const { manifest, state, events, inputSpecAddendumSha256 } = receipt ?? {}, lane = ledgerLane(approval, expectedBackend, host);
  // A deferred backend's ledger is bound to exactly ONE addendum: recorded before its first spend
  // and required (identical) for every later reservation, audit and aggregation.
  if ((approval.deferredInputSpecBackends ?? []).includes(lane)) {
    if ((events?.length ?? 0) > 0) assert(sha.test(inputSpecAddendumSha256 ?? ''), 'BLOCKED: deferred backend ledger has spends but no recorded addendum binding');
    if (inputSpecAddendumSha256 !== undefined) {
      assert(inputSpecAddendum, 'BLOCKED: this ledger is bound to its recorded addendum; present that same addendum');
      assert.equal(planHash(inputSpecAddendum), inputSpecAddendumSha256, 'BLOCKED: a different addendum than the one recorded in this ledger');
    }
  } else assert.equal(inputSpecAddendumSha256, undefined, 'a non-deferred backend ledger carries no addendum binding');
  assert(Array.isArray(events));
  const transitions = events.filter(e => e.kind === 'candle_source_transition');
  const requiresTransition = expectedBackend === 'candle' && Object.hasOwn(approval, 'candleSourceTransitionSha256');
  let prefix;
  if (requiresTransition) {
    assert.equal(transitions.length, 1, 'authenticated Candle source transition required, exactly once');
    assert.equal(events[1], transitions[0], 'source transition is only the fixed second journal event');
    prefix = transitionPrefix(approval, transitions[0]);
    assert.deepEqual(manifest, prefix.previousReceipt.manifest, 'immutable old manifest differs from carried prefix');
    assert.deepEqual(events.slice(0, 1), prefix.previousReceipt.events, 'immutable old event differs from carried prefix');
    assert.equal(inputSpecAddendumSha256, undefined, 'fixed Candle transition has no addendum');
  } else {
    assert.equal(transitions.length, 0, 'unknown/unapproved source transition');
    assert.deepEqual(manifest, expectedManifest(approval, expectedBackend, host), 'complete ledger manifest/plan/backend identity mismatch');
  }
  let head = prefix ? transitions[0].eventSha256 : planHash(manifest), generations = prefix ? 1 : 0; const ids = prefix ? [CANDLE_CONSUMED] : [];
  for (const [index, event] of events.entries()) {
    if (prefix && index < 2) continue;
    const { eventSha256, ...record } = event;
    assert.deepEqual(Object.keys(record).sort(), ['sequence','previousSha256','approvedPlanSha256','ledgerId','backend','caseId','invocationId','kind','requestSha256','reservedAt','controllerPid'].sort());
    assert.equal(eventSha256, planHash(record)); assert.equal(record.sequence, index + 1); assert.equal(record.previousSha256, head); assert.equal(record.ledgerId, approval.ledgerId); assert.equal(record.approvedPlanSha256, planHash(approval)); assert.equal(record.backend, expectedBackend); assert(sha.test(record.requestSha256)); assert(Number.isInteger(record.controllerPid) && record.controllerPid > 0 && Number.isFinite(Date.parse(record.reservedAt)));
    if (record.kind === 'training_resume') { const training = approval.plan.trainingLifecycles.find(t => t.resumeId === record.invocationId); assert(training && ids.includes(training.startId), 'resume precedes its initial training start'); }
    if (approvedEvent(approval, record, inputSpecAddendum, host)) generations++; assert(!ids.includes(record.invocationId), 'single-use permit reused'); ids.push(record.invocationId); head = eventSha256;
  }
  assert(generations <= manifest.permitIds.length && generations <= GLOBAL_GENERATION_CEILING);
  assert.deepEqual(state, { sequence: events.length, generationCount: generations, consumedIds: ids, headSha256: head }, 'ledger crash/tamper mismatch; coordinator must reconcile, never reuse a reservation');
  return { manifest, state, events, ...(inputSpecAddendumSha256 !== undefined ? { inputSpecAddendumSha256 } : {}) };
}
const ADDENDUM_BINDING = 'input-spec-addendum.json';
async function boundAddendum(root) {
  try { return JSON.parse(await readFile(path.join(root, ADDENDUM_BINDING), 'utf8')); } catch (error) { if (error.code === 'ENOENT') return undefined; throw error; }
}
async function audited(approval, backend, root, inputSpecAddendum, host) {
  assert(!(await readdir(root)).some(file => /^state-.*\.tmp$/.test(file)), 'ledger partial state publication; no automatic repair/reset');
  const manifest = JSON.parse(await readFile(path.join(root, 'manifest.json'), 'utf8')), state = JSON.parse(await readFile(path.join(root, 'state.json'), 'utf8')), files = (await readdir(path.join(root, 'events'))).sort(), events = [];
  for (const [index, file] of files.entries()) { assert.equal(file, `${String(index + 1).padStart(6, '0')}.json`, 'ledger journal missing/foreign event'); events.push(JSON.parse(await readFile(path.join(root, 'events', file), 'utf8'))); }
  const binding = await boundAddendum(root);
  return verifyLedgerReceipt(approval, { manifest, state, events, ...(binding ? { inputSpecAddendumSha256: binding.addendumSha256 } : {}) }, backend, inputSpecAddendum, host);
}
export async function auditLedger({ approval, backend, ledgerRoot, inputSpecAddendum, host }) { const root = location(approval, backend, ledgerRoot, host); return lock(root, () => audited(approval, backend, root, inputSpecAddendum, host)); }
export async function reserveInvocation({ approval, ledgerRoot, ledgerId, approvedPlanSha256, invocationId, backend, caseId, kind, requestSha256, inputSpecAddendum, host }) {
  const root = location(approval, backend, ledgerRoot, host), lane = ledgerLane(approval, backend, host);
  // Wrong-host refusal BEFORE the lock or any write: another lane's permit is never spendable here.
  if (!inHostLane(approval.plan, backend, host, invocationId)) throw new Error(`BLOCKED: ${invocationId} is assigned to another ${backend} host lane, not ${host}; no permit is reservable on the wrong host`);
  // A deferred backend has no permit before its addendum (and only ever the SAME addendum, below).
  boundInputSpecSha256(approval, backend, inputSpecAddendum, host);
  assert.equal(ledgerId, approval.ledgerId, 'reservation ledger identity differs from the approval');
  assert.equal(approvedPlanSha256, planHash(approval), 'reservation is bound to a different approved plan');
  assert(sha.test(requestSha256 ?? ''), 'actual request digest required before each inner generation start');
  return lock(root, async () => {
    const { state } = await audited(approval, backend, root, inputSpecAddendum, host);
    assert(!state.consumedIds.includes(invocationId), 'permit already consumed; failed/canceled attempts count and cannot auto-resubmit');
    if (kind === 'training_resume') { const training = approval.plan.trainingLifecycles.find(t => t.resumeId === invocationId); assert(training && state.consumedIds.includes(training.startId), 'planned resume requires the already-consumed initial training start'); }
    const record = { sequence: state.sequence + 1, previousSha256: state.headSha256, approvedPlanSha256, ledgerId, backend, caseId, invocationId, kind, requestSha256, reservedAt: new Date().toISOString(), controllerPid: process.pid };
    // Validate the permit fully BEFORE any durable write (addendum binding or journal).
    const generation = approvedEvent(approval, record, inputSpecAddendum, host), event = { ...record, eventSha256: planHash(record) };
    if ((approval.deferredInputSpecBackends ?? []).includes(lane) && !(await boundAddendum(root))) {
      // First spend: record WHICH addendum this ledger is bound to, exclusively and durably.
      await durableNew(path.join(root, ADDENDUM_BINDING), { ledgerId: approval.ledgerId, backend, ...(lane !== backend ? { host } : {}), approvedPlanSha256: planHash(approval), addendumSha256: planHash(inputSpecAddendum) });
    }
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
  // Heads/addenda are keyed by ledger lane: a backend, or backend@host when the plan splits it.
  const expectedHeads = closure.backends, laneKeys = ledgerLaneKeys(approval);
  assert(expectedHeads && JSON.stringify(Object.keys(expectedHeads).sort()) === JSON.stringify([...laneKeys].sort()), 'trusted latest coordinator-audited heads for EVERY ledger lane required');
  for (const candidate of receipts) {
    assert(['mlx', 'candle'].includes(candidate?.manifest?.backend), 'complete ledger manifest/backend required');
    const lane = ledgerLane(approval, candidate.manifest.backend, candidate.manifest.host);
    const receipt = verifyLedgerReceipt(approval, candidate, candidate?.manifest?.backend, inputSpecAddenda[lane], candidate?.manifest?.host), key = `${receipt.manifest.ledgerId}:${lane}`; assert(!ledgers.has(key), 'duplicate ledger/backend receipt; use one complete latest receipt per backend'); ledgers.add(key);
    assert.deepEqual({ sequence: receipt.state.sequence, headSha256: receipt.state.headSha256 }, expectedHeads[lane], 'replayed/stale receipt differs from trusted latest coordinator-audited head');
    for (const event of receipt.events) { if (event.kind === 'candle_source_transition') continue; assert(!ids.has(event.invocationId), 'duplicate receipt/permit across backend sessions'); ids.add(event.invocationId); }
    generationCount += receipt.state.generationCount;
  }
  assert.equal(ledgers.size, laneKeys.length, 'partial backend aggregation is not a global campaign count; every complete latest lane receipt required');
  assert(generationCount <= approval.plan.generationInvocations.length && generationCount <= GLOBAL_GENERATION_CEILING);
  return { proofKind: 'reservation-accounting-only', caseCompletionProven: false, coordinatorClosureId: closure.coordinatorClosureId, collectionManifestSha256: closure.collectionManifestSha256, ledgerId: approval.ledgerId, approvedPlanSha256: planHash(approval), generationCount, approvedTotal: approval.plan.generationInvocations.length, globalHardCeiling: GLOBAL_GENERATION_CEILING, unallocatedNotAuthorized: GLOBAL_GENERATION_CEILING - approval.plan.generationInvocations.length };
}
