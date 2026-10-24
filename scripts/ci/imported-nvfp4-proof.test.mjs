// CPU-only proof-policy tests. Fixtures never execute Cargo, a model, or CUDA.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { CASE, DEVICE_PROOF, discoverFqn, observeExecution, validateExecution, validateEvidence, redact, regularOutput, stageArtifact } from './imported-nvfp4-proof.mjs';

const fqn = `imported_nvfp4_gpu_smoke::${CASE.filter}`;
function packet() {
  return {
    schema_version: 1, status: 'completed', test_fqn: fqn, actual_devices: ['candle-cuda'],
    device_provenance: { ...DEVICE_PROOF, anchors: ['immutable producer source anchor'] },
    checkpoint_id: 'fixture-import-plan', output: 'generated-image',
    configuration: { route: 'candle_checkpoint_plan', provider: 'krea_2_turbo', geometry: [256, 256], seed: 42, steps: 1 },
    runtime_facts: { schemaVersion: 1, materialization: { status: 'reported', rows: [
      { codecId: 'nvfp4-v1', representation: 'native-packed', tensorCount: 1 },
    ] } },
  };
}
function log() {
  return `RESULT status=pass route=candle_checkpoint_plan provider=krea_2_turbo geometry=256x256\n`
    + `MODEL_INFERENCE_DEVICE ${fqn} candle-cuda\nMODEL_INFERENCE_BODY_COMPLETED ${fqn}\n`
    + 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.0s\n';
}

test('fixed ignored discovery requires exactly one actual FQN', () => {
  assert.equal(discoverFqn(`${fqn}: test\n1 test, 0 benchmarks\n`), fqn);
  assert.throws(() => discoverFqn('0 tests, 0 benchmarks\n'));
  assert.throws(() => discoverFqn(`${fqn}: test\nother::${CASE.filter}: test\n`));
});
test('complete exact-one-case evidence has explicit actual-device provenance', () => {
  assert.deepEqual(validateExecution(log(), 0, fqn, packet()), {
    passed: 1, failed: 0, ignored: 0, body_completed: 1, actual_devices: ['candle-cuda'],
  });
});
for (const [name, mutate] of [
  ['zero matches', (text) => text.replace('1 passed', '0 passed')],
  ['still ignored', (text) => text.replace('0 ignored', '1 ignored')],
  ['missing body/early return', (text) => text.replace(`MODEL_INFERENCE_BODY_COMPLETED ${fqn}\n`, '')],
  ['duplicate body', (text) => text + `MODEL_INFERENCE_BODY_COMPLETED ${fqn}\n`],
  ['wrong body identity', (text) => text.replace(`BODY_COMPLETED ${fqn}`, 'BODY_COMPLETED other')],
  ['missing actual device', (text) => text.replace(`MODEL_INFERENCE_DEVICE ${fqn} candle-cuda\n`, '')],
  ['CPU/default-context substitution', (text) => text.replace('candle-cuda', 'cpu')],
  ['missing real generated result', (text) => text.replace(/^RESULT[^\n]*\n/, '')],
  ['multiple test summaries', (text) => text + 'test result: ok. 0 passed; 0 failed; 0 ignored;\n'],
]) {
  test(`reject ${name}`, () => assert.throws(() => validateExecution(mutate(log()), 0, fqn, packet())));
}
test('reject process failure even if a stale pass marker exists', () => {
  assert.throws(() => validateExecution(log(), 1, fqn, packet()));
});
test('capability alone and unavailable/dense receipts cannot seal CUDA proof', () => {
  for (const representation of ['dense-fallback', 'unavailable']) {
    const value = packet();
    value.runtime_facts.materialization.rows[0].representation = representation;
    assert.throws(() => validateExecution(log(), 0, fqn, value));
  }
  const missing = packet(); missing.runtime_facts.materialization = { status: 'unavailable' };
  assert.throws(() => validateExecution(log(), 0, fqn, missing));
  const zero = packet(); zero.runtime_facts.materialization.rows[0].tensorCount = 0;
  assert.throws(() => validateExecution(log(), 0, fqn, zero));
});
test('reject stale producer provenance and incorrect model configuration', () => {
  const stale = packet(); stale.device_provenance.inference_revision = '0'.repeat(40);
  assert.throws(() => validateExecution(log(), 0, fqn, stale));
  const changed = packet(); changed.configuration.geometry = [512, 512];
  assert.throws(() => validateExecution(log(), 0, fqn, changed));
});
test('failure observation records real zero/ignored counts, never converts them to pass', () => {
  assert.deepEqual(observeExecution('test result: ok. 0 passed; 0 failed; 1 ignored;\n'), {
    summaries: [{ status: 'ok', passed: 0, failed: 0, ignored: 1 }], body_completed: 0, actual_devices: [],
  });
});
test('fixed direct-case JSON is compatible with common-v1 source/device/body/output evidence', () => {
  const item = (relative_path) => ({ relative_path, bytes: 3, sha256: 'a'.repeat(64) });
  const value = {
    schema_version: 1, accepted: true,
    source: { revision: '1'.repeat(40), tree: '2'.repeat(40) },
    inference_source: { revision: DEVICE_PROOF.inference_revision },
    case: { ...CASE, target: 'sceneworks_worker', fqn },
    host: { platform: 'win32', machine: 'x64', runner: 'owned-fixture' },
    actual_devices: ['candle-cuda'], device_provenance: DEVICE_PROOF,
    fixture_identity: { sha256: '3'.repeat(64), bytes: 3, checkpoint_id: 'fixture-plan' },
    configuration: packet().configuration,
    test_counts: validateExecution(log(), 0, fqn, packet()), body_completed: 1,
    outputs: CASE.output_contract.map(item), logs: ['discovery.log', 'execution.log'].map(item),
    duration_seconds: 1,
  };
  validateEvidence(value);
  for (const mutate of [
    (copy) => { copy.source.tree = ''; },
    (copy) => { copy.actual_devices = ['cpu']; },
    (copy) => { copy.test_counts.passed = 0; },
    (copy) => { copy.body_completed = 0; },
    (copy) => { copy.case.execution_role = 'unregistered'; },
    (copy) => { copy.outputs[0].relative_path = '../checkpoint'; },
  ]) {
    const copy = structuredClone(value); mutate(copy);
    assert.throws(() => validateEvidence(copy));
  }
});
test('bounded log redaction removes credentials and signed URL queries', () => {
  const safe = redact('TOKEN=secret https://user:password@example.test/asset?sig=secret&key=secret\x1b[31m');
  assert(!safe.includes('secret') && !safe.includes('password') && !safe.includes('user:'));
  assert(safe.includes('REDACTED'));
});
test('completed body output remains unaccepted until external physical-resource sealing', async () => {
  const directory = await fs.mkdtemp(path.join(path.dirname(fileURLToPath(import.meta.url)), 'proof-unit-'));
  try {
    for (const name of ['discovery.log', 'execution.log', 'completion.json', 'generated-image']) {
      await fs.writeFile(path.join(directory, name), 'owned CPU fixture bytes');
    }
    const evidence = { accepted: false, body_proof_accepted: true, resource_verification_pending: true };
    const staging = await stageArtifact(directory, evidence);
    const recorded = JSON.parse(await fs.readFile(path.join(staging, 'evidence.json')));
    assert.equal(recorded.accepted, false);
    assert.equal(recorded.body_proof_accepted, true);
    assert.equal(recorded.resource_verification_pending, true);
    assert((await fs.readdir(staging)).includes('generated-image'));
  } finally {
    await fs.rm(directory, { recursive: true, force: true });
  }
});
test('artifact staging rejects unknown/hardlinked files and never uploads unsafe originals', async () => {
  const directory = await fs.mkdtemp(path.join(path.dirname(fileURLToPath(import.meta.url)), 'proof-unit-'));
  try {
    await fs.writeFile(path.join(directory, 'execution.log'), 'redacted fixture log\n');
    await fs.writeFile(path.join(directory, 'outside-owned-fixture'), 'not model evidence');
    await fs.link(path.join(directory, 'outside-owned-fixture'), path.join(directory, 'completion.json'));
    await assert.rejects(regularOutput(directory, 'completion.json'));
    await assert.rejects(regularOutput(directory, '../outside-owned-fixture'));
    await assert.rejects(regularOutput(directory, 'outside-owned-fixture'));
    const evidence = { accepted: false, error: 'fixture failure' };
    const staging = await stageArtifact(directory, evidence);
    assert.deepEqual((await fs.readdir(staging)).sort(), ['evidence.json', 'execution.log']);
    assert.equal(JSON.parse(await fs.readFile(path.join(staging, 'evidence.json'))).accepted, false);
  } finally {
    await fs.rm(directory, { recursive: true, force: true });
  }
});
