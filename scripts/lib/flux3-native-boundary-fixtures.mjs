// DECLARED SYNTHETIC CPU TEST FIXTURES ONLY. Never evidence: builds a self-consistent READY backend
// (full receipt, sealed resource-build binding, READY rows) so tests exercise the real validators.
import { NATIVE_BOUNDARY_RECEIPT_KIND, nativeBoundaryReceiptSha256 } from './flux3-native-boundary.mjs';

export function declaredSyntheticReceipt(invocations, { backend, binarySha256, buildReceiptSha256, sourceSha, inferenceMainSha }) {
  const refused = { refused: true, generateCalls: 0 };
  return { schemaVersion: 1, kind: NATIVE_BOUNDARY_RECEIPT_KIND, mode: 'DECLARED SYNTHETIC CPU TEST FIXTURE', lane: backend, binarySha256, buildReceiptSha256,
    sourceSha, inferenceMainSha, campaignPermitsConsumed: 0,
    cases: invocations.filter(i => i.backend === backend && i.kind.startsWith('resource_')).map(i => ({ invocationId: i.invocationId, caseId: i.caseId, kind: i.kind,
      requestSha256: i.requestSha256, valid: { reserveCalls: 1, generateCalls: 1, order: ['reserve', 'generate'] },
      reused: { ...refused }, missing: { ...refused }, mismatchedDigest: { ...refused }, mismatchedCase: { ...refused } })) };
}

/** Mark every resource row of `backend` READY under a synthetic sealed resource build. Resource
 *  invocations without a request digest get `requestSha256` (default: a fixed synthetic digest). */
export function declaredSyntheticReadyBackend(approval, backend, { binarySha256 = '7'.repeat(64), buildReceiptSha256 = '8'.repeat(64), requestSha256 = '9'.repeat(64) } = {}) {
  const plan = approval.plan;
  for (const i of plan.generationInvocations.filter(i => i.backend === backend && i.kind.startsWith('resource_'))) i.requestSha256 ??= requestSha256;
  const receipt = declaredSyntheticReceipt(plan.generationInvocations, { backend, binarySha256, buildReceiptSha256, sourceSha: approval.sourceSha, inferenceMainSha: approval.inferenceMainSha });
  approval.resourceBuildReceipts = { ...(approval.resourceBuildReceipts ?? {}), [backend]: { receiptSha256: buildReceiptSha256, binarySha256 } };
  plan.nativeBoundaryReceipts = { ...(plan.nativeBoundaryReceipts ?? {}), [backend]: receipt };
  const digest = nativeBoundaryReceiptSha256(receipt);
  for (const row of plan.resourceCases.filter(r => r.backend === backend)) Object.assign(row, { status: 'READY', blocker: null, parentNativeBoundaryVerificationId: 'DECLARED-SYNTHETIC', nativeBoundaryReceiptSha256: digest, nativeBoundaryBinarySha256: binarySha256 });
  return receipt;
}
