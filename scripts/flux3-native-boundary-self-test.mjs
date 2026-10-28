// Zero-budget native-boundary self-test PRODUCER (sc-26243). Runs the SEALED resource-adapter
// binary's hidden CPU self-test (counting generator, scripted single-use ledger, no weights, no
// device, no provider, no campaign permit) over EVERY resource invocation of one backend in the
// plan, and validates the receipt it emits. The receipt is what makes a resource row READY.
//
//   node scripts/flux3-native-boundary-self-test.mjs SELF_TEST.json
//   SELF_TEST.json: { lane, binary, buildReceipt: { path, root, sha256 }, sourceSha,
//                     inferenceMainSha, planFile, resourceInputRoot, outputFile }
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { contained, fileHash } from './flux3-image-klein-terminal-session.mjs';
import { planAnchor } from './memory-calibration-harness.mjs';
import { validateNativeBoundaryReceipt } from './lib/flux3-native-boundary.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const SELF_TEST_ACTION = 'flux3_native_boundary_self_test';

/** The exact production `run` request a resource invocation sends (canonical anchor planned case,
 *  plus the bound supplemental resource case for PUP-edit). */
export function productionResourceRequest(calibrationPlan, plan, invocation) {
  const planned = planAnchor(calibrationPlan, `${invocation.modelId}:${invocation.tier}:${invocation.backend}`);
  const row = plan.resourceCases.find(r => r.caseId === invocation.caseId && r.backend === invocation.backend);
  if (invocation.kind === 'resource_supplemental') {
    assert(row?.resourceCase, `${invocation.caseId}: supplemental resource case must be bound in the plan`);
    return { action: 'run', planned, resourceCase: row.resourceCase };
  }
  return { action: 'run', planned };
}

/** Only the self-test guard, the hidden device and the OS essentials: no production invocation,
 *  ledger transport, model root, credential or provider selection can reach the child. */
export function selfTestEnvironment(env, resourceInputRoot) {
  const keep = ['SystemRoot', 'SYSTEMROOT', 'WINDIR', 'PATH', 'Path', 'TEMP', 'TMP', 'HOME', 'USERPROFILE'];
  const child = Object.fromEntries(keep.filter(k => env[k] !== undefined).map(k => [k, env[k]]));
  return { ...child, SCENEWORKS_FLUX3_BOUNDARY_SELF_TEST: '1', CUDA_VISIBLE_DEVICES: '-1',
    ...(resourceInputRoot ? { SCENEWORKS_FLUX3_RESOURCE_INPUT_ROOT: resourceInputRoot } : {}) };
}

async function execute(binary, request, env) {
  const child = spawn(binary, [], { env, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
  let stdout = '', stderr = '';
  child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');
  child.stdout.on('data', chunk => { stdout += chunk; }); child.stderr.on('data', chunk => { stderr += chunk; });
  const exit = new Promise((resolve, reject) => { child.once('error', reject); child.once('close', code => resolve(code)); });
  child.stdin.end(JSON.stringify(request) + '\n');
  return { code: await exit, stdout, stderr };
}

export async function runNativeBoundarySelfTest(config, { run = execute, env = process.env } = {}) {
  const { lane, host, binary, buildReceipt, sourceSha, inferenceMainSha, planFile, resourceInputRoot } = config;
  assert(['candle', 'mlx'].includes(lane), 'lane must be candle or mlx');
  // The sealed binary under test IS the approved resource-adapter build.
  const receiptFile = await contained(buildReceipt.path, buildReceipt.root), receiptBytes = await readFile(receiptFile);
  assert.equal(await fileHash(receiptFile), buildReceipt.sha256, 'sealed build receipt digest mismatch');
  const build = JSON.parse(receiptBytes);
  assert.equal(build.kind, 'flux3-owned-build-producer-receipt'); assert.equal(build.backend, lane);
  assert.equal(build.recipe?.bin, `memory-${lane}-adapter`, 'binary is not the sealed resource-adapter role');
  assert.equal(build.sourceSha, sourceSha); assert.equal(build.inferenceMainSha, inferenceMainSha);
  assert.equal(await fileHash(binary), build.binary.sha256, 'binary differs from its sealed build receipt');
  const plan = JSON.parse(await readFile(planFile, 'utf8'));
  const calibrationPlan = JSON.parse(await readFile(path.join(ROOT, 'config/memory-calibration-plan.json'), 'utf8'));
  // A host-split backend proves only the resource invocations of the named host's lane.
  const lanes = plan.hostLanes?.[lane]; if (lanes) assert(Object.hasOwn(lanes, host ?? ''), `${lane} is split by host; name an approved host lane`); else assert.equal(host, undefined, `${lane} is not split by host`);
  const invocations = plan.generationInvocations.filter(i => i.backend === lane && i.kind.startsWith('resource_') && (!lanes || lanes[host].includes(i.invocationId)));
  assert(invocations.length && invocations.every(i => /^[a-f0-9]{64}$/.test(i.requestSha256 ?? '')), 'every resource request digest must be fixed in the plan first');
  const request = {
    action: SELF_TEST_ACTION, lane, expectedBinarySha256: build.binary.sha256, buildReceiptSha256: buildReceipt.sha256,
    buildReceiptBinarySha256: build.binary.sha256, sourceSha, inferenceMainSha,
    cases: invocations.map(i => ({ invocationId: i.invocationId, caseId: i.caseId, kind: i.kind, requestSha256: i.requestSha256,
      request: productionResourceRequest(calibrationPlan, plan, i) })),
  };
  const output = await run(binary, request, selfTestEnvironment(env, resourceInputRoot));
  assert.equal(output.code, 0, `sealed self-test refused: ${output.stdout.slice(0, 2000)}${output.stderr.slice(0, 2000)}`);
  const receipt = JSON.parse(output.stdout);
  const receiptSha256 = validateNativeBoundaryReceipt(receipt, { backend: lane, invocations, binarySha256: build.binary.sha256,
    buildReceiptSha256: buildReceipt.sha256, sourceSha, inferenceMainSha });
  return { receipt, receiptSha256, binarySha256: build.binary.sha256, buildReceiptSha256: buildReceipt.sha256 };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    assert.equal(process.argv.length, 3, 'one explicit self-test JSON file required');
    const config = JSON.parse(await readFile(process.argv[2], 'utf8'));
    const result = await runNativeBoundarySelfTest(config);
    await writeFile(config.outputFile, JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
    console.log(JSON.stringify({ receiptSha256: result.receiptSha256, binarySha256: result.binarySha256, cases: result.receipt.cases.length }));
  } catch (error) { console.error(`BLOCKED: ${error.message}`); process.exitCode = 1; }
}
