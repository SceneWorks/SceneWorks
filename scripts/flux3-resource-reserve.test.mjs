// Explicit synthetic CPU-only ledger/CLI fixtures. No native registry, model, GPU or capture.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { initializeLedger, auditLedger, planHash } from './flux3-image-klein-terminal-budget.mjs';
import { flux3AnchorRequestDigest } from './lib/flux3-resource-contract.mjs';
const script = fileURLToPath(new URL('./flux3-resource-reserve.mjs', import.meta.url));
async function fixture(t) {
  const root = await mkdtemp(path.join(tmpdir(), 'flux3-resource-DECLARED-CPU-fixture-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const backend = 'candle', caseId = 'candle.base.resource-native.bf16', kind = 'resource_anchor';
  const digest = flux3AnchorRequestDigest(backend, 'flux3_klein_base', 'bf16');
  const invocation = { backend, caseId, kind, invocationId: `${caseId}.generation-1`, requestSha256: digest,
    requestTemplate: { count: 1 }, nativeGenerationCalls: 1, warmups: 0, repeats: 0 };
  const approval = { schemaVersion: 1, kind: 'flux3-terminal-plan-approval', approvalId: 'CPU-fixture',
    userPlanApprovalId: 'DECLARED_SYNTHETIC_NOT_USER_AUTHORIZATION', approvedBy: 'CPU-fixture',
    sourceSha: 'a'.repeat(40), inferenceMainSha: 'b'.repeat(40), ledgerId: 'DECLARED-CPU-fixture-ledger',
    ledgerRoots: { candle: path.join(root, 'candle'), mlx: path.join(root, 'mlx') },
    inputSpecSha256: { candle: digest, mlx: digest },
    plan: { kind: 'flux3-installed-product-representative-plan', generationInvocations: [invocation], trainingLifecycles: [],
      resourceCases: [{ backend, caseId, status: 'READY', parentNativeBoundaryVerificationId: 'DECLARED-SYNTHETIC', nativeBoundaryReceiptSha256: digest }],
      budget: { globalHardCeiling: 48, totalGenerationInvocations: 1, byBackend: { candle: 1, mlx: 0 },
        unallocatedNotAuthorized: 47, automaticWarmups: 0, automaticRepeats: 0, automaticRetries: 0, trainingLifecycles: 0, stepsPerLifecycle: 4 } } };
  const approvalFile = path.join(root, 'approval.json');
  await writeFile(approvalFile, JSON.stringify(approval));
  await initializeLedger({ approval, backend, ledgerRoot: approval.ledgerRoots[backend] });
  const admissionFile = path.join(root, 'admission.json');
  await writeFile(admissionFile, JSON.stringify({ actualRun: false, fixtureOnly: true, approvedPlan: approval, backend, ledgerId: approval.ledgerId,
    ledgerRoot: approval.ledgerRoots[backend], approvedPlanSha256: planHash(approval) }));
  return { approval, input: { approvalFile, admissionFile, adapterPid: 12345, sourceSha: approval.sourceSha, binarySha256: digest,
    ledgerRoot: approval.ledgerRoots[backend], ledgerId: approval.ledgerId,
    approvedPlanSha256: planHash(approval), invocationId: invocation.invocationId, backend, caseId, kind, requestSha256: digest } };
}
// A separate Node process exercises the REAL atomic ledger; ONLY physical hardware is an
// explicitly injected synthetic CPU observation. Production CLI has no injection flag/module.
const fixtureProgram = `import {reserveResourceInvocation} from ${JSON.stringify(new URL('./flux3-resource-reserve.mjs', import.meta.url).href)};
let text='';for await(const chunk of process.stdin)text+=chunk;
try { const result=await reserveResourceInvocation(JSON.parse(text),{verifyHardware:async(_a,context)=>({...context,kind:'DECLARED_SYNTHETIC_CPU_HARDWARE'})});console.log(JSON.stringify(result)); }
catch(error){console.error('BLOCKED: '+error.message);process.exitCode=1;}`;
const run = input => spawnSync(process.execPath, ['--input-type=module', '-e', fixtureProgram], { input: JSON.stringify(input), encoding: 'utf8', env: { ...process.env, CUDA_VISIBLE_DEVICES: '-1' } });
test('real resource CLI atomically spends before call; repeat/fail/cancel cannot refund or reuse', async t => {
  const { approval, input } = await fixture(t);
  let scriptedCalls = 0;
  for (let attempt = 0; attempt < 3; attempt++) {
    const result = run(input);
    if (result.status === 0) {
      const receipt = JSON.parse(result.stdout);
      assert.equal(receipt.invocationId, input.invocationId);
      assert.equal(receipt.requestSha256, input.requestSha256);
      scriptedCalls++; // A thrown/canceled scripted call would still leave this reservation spent.
    } else assert.match(result.stderr, /already consumed/);
  }
  assert.equal(scriptedCalls, 1);
  const receipt = await auditLedger({ approval, backend: 'candle', ledgerRoot: input.ledgerRoot });
  assert.equal(receipt.state.generationCount, 1);
});
test('production CLI has no synthetic hardware bypass and refuses fixture admissions before spend', async t => {
  const { approval, input } = await fixture(t);
  const result = spawnSync(process.execPath,[script],{input:JSON.stringify({...input,fixtureOnly:true}),encoding:'utf8',env:{...process.env,CUDA_VISIBLE_DEVICES:'-1'}});
  assert.notEqual(result.status,0);assert.match(result.stderr,/BLOCKED/);
  assert.equal((await auditLedger({approval,backend:'candle',ledgerRoot:input.ledgerRoot})).state.generationCount,0);
});
test('resource CLI rejects unplanned IDs, mutated request, alternate root and product permit before spend', async t => {
  const { approval, input } = await fixture(t);
  for (const changed of [{ ...input, invocationId: 'unallocated-spare' }, { ...input, requestSha256: 'e'.repeat(64) },
    { ...input, ledgerRoot: path.join(input.ledgerRoot, 'reset') }, { ...input, kind: 'product_generation' }]) {
    const result = run(changed); assert.notEqual(result.status, 0); assert.match(result.stderr, /BLOCKED/);
  }
  assert.equal((await auditLedger({ approval, backend: 'candle', ledgerRoot: input.ledgerRoot })).state.generationCount, 0);
});
