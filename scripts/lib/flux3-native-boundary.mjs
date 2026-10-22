// FLUX3 native resource-boundary verification (sc-26243): the receipt the SEALED resource adapter's
// hidden zero-budget CPU self-test emits, and the rule that turns it into READY resource rows.
// No weights, devices, providers or campaign permits are involved anywhere in this module.
import { createHash } from 'node:crypto';
import { normalizeResourceCase } from './flux3-resource-contract.mjs';

export const NATIVE_BOUNDARY_RECEIPT_KIND = 'flux3-native-boundary-self-test-receipt';
const sha = /^[a-f0-9]{64}$/, revision = /^[a-f0-9]{40}$/;
const REFUSALS = ['reused', 'missing', 'mismatchedDigest', 'mismatchedCase'];
const fail = message => { throw new Error(`BLOCKED: native-boundary receipt ${message}`); };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/** Digest that READY rows / addenda bind. */
export function nativeBoundaryReceiptSha256(receipt) {
  return createHash('sha256').update('sceneworks-flux3-native-boundary-receipt-v1\0').update(JSON.stringify(receipt)).digest('hex');
}

/** Validate a self-test receipt against the EXACT sealed resource binary and the plan's own resource
 *  invocations for one backend: one proof per invocation, digests equal to the plan's fixed
 *  requestSha256, one reservation strictly before exactly one generate, and every reused/missing/
 *  mismatched permit refused with zero generates. Returns the receipt digest. */
export function validateNativeBoundaryReceipt(receipt, { backend, binarySha256, buildReceiptSha256, sourceSha, inferenceMainSha, invocations }) {
  if (!receipt || receipt.schemaVersion !== 1 || receipt.kind !== NATIVE_BOUNDARY_RECEIPT_KIND) fail('kind/schema required');
  if (receipt.lane !== backend) fail(`lane ${receipt.lane} is not ${backend}`);
  if (!sha.test(binarySha256 ?? '') || receipt.binarySha256 !== binarySha256) fail('was produced by a different binary than the approved sealed resource build');
  if (!sha.test(buildReceiptSha256 ?? '') || receipt.buildReceiptSha256 !== buildReceiptSha256) fail('binds a different sealed build receipt');
  if (!revision.test(sourceSha ?? '') || receipt.sourceSha !== sourceSha) fail('source identity differs from the approval');
  if (!revision.test(inferenceMainSha ?? '') || receipt.inferenceMainSha !== inferenceMainSha) fail('inference identity differs from the approval');
  if (receipt.campaignPermitsConsumed !== 0) fail('must not consume campaign permits');
  const expected = (invocations ?? []).filter(i => i.backend === backend);
  if (!expected.length || !Array.isArray(receipt.cases) || receipt.cases.length !== expected.length) fail('must prove exactly the plan resource invocations of its backend');
  for (const invocation of expected) {
    const proof = receipt.cases.filter(c => c.invocationId === invocation.invocationId);
    if (proof.length !== 1) fail(`missing or duplicate proof for ${invocation.invocationId}`);
    const [p] = proof;
    if (p.caseId !== invocation.caseId || p.kind !== invocation.kind) fail(`case identity differs for ${invocation.invocationId}`);
    if (!sha.test(invocation.requestSha256 ?? '') || p.requestSha256 !== invocation.requestSha256) fail(`request digest differs from the plan for ${invocation.invocationId}`);
    if (!same(p.valid, { reserveCalls: 1, generateCalls: 1, order: ['reserve', 'generate'] })) fail(`does not prove one reservation before exactly one generate for ${invocation.invocationId}`);
    for (const refusal of REFUSALS) if (!same(p[refusal], { refused: true, generateCalls: 0 })) fail(`${refusal} permit was not refused before generate for ${invocation.invocationId}`);
  }
  return nativeBoundaryReceiptSha256(receipt);
}

/** The bound PUP-edit supplemental resource case for one backend, from hash-only input bindings
 *  (paths relative to the resource input root, so the digest is host-independent). */
export function buildSupplementalResourceCase({ backend, prompt, references, promptUpsampler }) {
  if (!Array.isArray(references) || references.length !== 10) throw new Error('BLOCKED: the PUP edit resource case binds exactly ten ordered references');
  const value = {
    schemaVersion: 1, caseId: `${backend}.base.pup_edit.bf16`, backend, modelId: 'flux3_klein_base', tier: 'bf16', scenario: 'pup_edit',
    generation: { prompt, width: 1024, height: 1024, seed: 26243, steps: 50, guidance: 3.5, pup: true },
    references: references.map((r, index) => ({ instanceId: `slot-${index}`, assetId: `reference-${index}`, index, path: r.path, sha256: r.sha256 })),
    adapters: [],
    components: { prompt_upsampler: { root: promptUpsampler.root, repository: promptUpsampler.repository, resolvedRevision: promptUpsampler.resolvedRevision, inventory: promptUpsampler.inventory.map(({ path, bytes, sha256 }) => ({ path, bytes, sha256 })) } },
  };
  normalizeResourceCase(value);
  return value;
}
