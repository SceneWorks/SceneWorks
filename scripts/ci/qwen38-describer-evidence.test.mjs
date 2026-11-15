// CPU-only tests of the Qwen3.8 27B describer evidence validator. No Cargo, model or CUDA runs here.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { FIELDS, GUARD_ACTOR, REPO, TEST_NAME, main, stage, summary, validateReceipt, validateResult } from './qwen38-describer-evidence.mjs';

function result(tier = 'q4') {
  return {
    schema_version: 1,
    test_fqn: `qwen38_describer_gpu_smoke::${TEST_NAME}`,
    repo: REPO, revision: 'a'.repeat(40), tier,
    weights_dir: 'E:\\huggingface\\hub\\fixture',
    image: { kind: 'synthetic-scene', width: 512, height: 384 },
    fields: Object.fromEntries(FIELDS.map((name) => [name, `${name} text | with a pipe`])),
    wall_seconds: 42.5, wall_time_scope: 'fixture',
    gpu_memory: { gpu_index: '1', baseline_mib: 500, peak_mib: 21000, peak_delta_mib: 20500, method: 'fixture' },
  };
}
// Shaped as windows_cuda_execution.py writes them for each policy.
function receipt(completed = true) {
  return { kind: 'cuda-execution-resource-lease-v1', actor: GUARD_ACTOR, resource_completed: completed,
    admission_policy: 'strict-idle', post_child_idle_verified: completed };
}
function sharedReceipt(completed = true) {
  return { kind: 'cuda-execution-resource-lease-v1', actor: GUARD_ACTOR, resource_completed: completed,
    admission_policy: 'shared-display', shared_device: true, isolated_device: false,
    post_child_idle_verified: false, observation_note: 'shared display GPU: fixture',
    shared_admission: { 'GPU-two': { samples: 10, baseline_used_mib: 14259, min_used_mib: 14200,
      free_mib: 83628, min_free_mib: 28000, max_utilization_percent: 12 } } };
}

test('a complete result is accepted for its own tier', () => {
  const accepted = validateResult(result('q8'), 'q8');
  assert.equal(accepted.tier, 'q8');
  assert.equal(accepted.gpu_memory.peak_mib, 21000);
  assert.deepEqual(Object.keys(accepted.fields), [...FIELDS]);
});
test('the validator lists the worker contract fields in order', async () => {
  const source = await fs.readFile(fileURLToPath(new URL('../../crates/sceneworks-core/src/dataset_batch_describe.rs', import.meta.url)), 'utf8');
  const block = source.match(/pub const DESCRIBE_FIELDS: \[&str; \d+\] = \[([^\]]*)\];/);
  assert(block, 'DESCRIBE_FIELDS is declared in the shared contract');
  assert.deepEqual([...block[1].matchAll(/"([a-z_]+)"/g)].map((match) => match[1]), [...FIELDS]);
});
test('empty pose and camera fields are accepted, as the output contract allows', () => {
  const copy = result();
  copy.fields.camera = '';
  copy.fields.pose = '';
  const accepted = validateResult(copy, 'q4');
  assert.equal(accepted.fields.camera, '');
  assert.equal(accepted.fields.pose, '');
});
for (const name of FIELDS) {
  test(`reject an empty ${name} field`, () => {
    for (const value of ['pose', 'camera'].includes(name) ? [null, undefined] : ['', '   ', null, undefined]) {
      const copy = result();
      copy.fields[name] = value;
      assert.throws(() => validateResult(copy, 'q4'), new RegExp(name));
    }
  });
}
for (const [name, mutate, tier] of [
  ['a different tier than requested', (copy) => { copy.tier = 'q8'; }, 'q4'],
  ['an unsupported tier', () => {}, 'bf16'],
  ['another test', (copy) => { copy.test_fqn = 'other::test'; }, 'q4'],
  ['another repository', (copy) => { copy.repo = 'someone/else'; }, 'q4'],
  ['an unpinned revision', (copy) => { copy.revision = 'main'; }, 'q4'],
  ['missing wall time', (copy) => { delete copy.wall_seconds; }, 'q4'],
  ['missing peak memory', (copy) => { delete copy.gpu_memory.peak_mib; }, 'q4'],
  ['a peak that was never sampled', (copy) => { copy.gpu_memory.peak_mib = 0; }, 'q4'],
  ['an unknown schema', (copy) => { copy.schema_version = 2; }, 'q4'],
  ['a missing image size', (copy) => { delete copy.image; }, 'q4'],
]) {
  test(`reject ${name}`, () => {
    const copy = result();
    mutate(copy);
    assert.throws(() => validateResult(copy, tier));
  });
}
test('the guard receipt must come from this actor and complete', () => {
  validateReceipt(receipt());
  assert.throws(() => validateReceipt(receipt(false)));
  assert.throws(() => validateReceipt({ ...receipt(), actor: 'iris-worker' }));
  assert.throws(() => validateReceipt({ ...receipt(), kind: 'other' }));
});
test('each admission needs a receipt of its own guard policy', () => {
  validateReceipt(sharedReceipt(), 'shared');
  // An idle run cannot pass on a shared-display receipt, nor a shared run on a strict one.
  assert.throws(() => validateReceipt(sharedReceipt(), 'idle'), /strict-idle/);
  assert.throws(() => validateReceipt(receipt(), 'shared'), /shared-display/);
  assert.throws(() => validateReceipt(sharedReceipt(false), 'shared'), /did not complete/);
  assert.throws(() => validateReceipt({ ...sharedReceipt(), shared_device: false }, 'shared'), /shared card/);
  assert.throws(() => validateReceipt({ ...sharedReceipt(), admission_policy: 'functional-coexistence' }, 'shared'),
    /shared-display/);
  assert.throws(() => validateReceipt({ ...sharedReceipt(), post_child_idle_verified: true }, 'shared'), /idle after/);
  assert.throws(() => validateReceipt({ ...sharedReceipt(), shared_admission: {} }, 'shared'), /admission window/);
  assert.throws(() => validateReceipt({ ...sharedReceipt(), shared_admission: { 'GPU-two': {
    free_mib: 20000, min_free_mib: 28000 } } }, 'shared'), /free memory/);
  assert.throws(() => validateReceipt({ ...receipt(), post_child_idle_verified: false }, 'idle'), /idle after/);
  assert.throws(() => validateReceipt({ ...sharedReceipt(), actor: 'load-progress-acceptance' }, 'shared'));
});
test('the summary escapes table pipes and names the peak, the image size and the admission', () => {
  const text = summary({ accepted: true, gpu_admission: 'idle', result: validateResult(result(), 'q4') });
  assert.match(text, /GPU admission: idle \(strict-idle guard/);
  assert.match(summary({ accepted: true, gpu_admission: 'shared' }),
    /GPU admission: shared \(shared-display guard.*measured on a shared card/);
  assert.match(text, /Peak GPU memory: 21000 MiB/);
  assert.match(text, /Image: synthetic-scene, 512x384/);
  assert.match(text, /description text \\\| with a pipe/);
});
test('a 1536 run must have described a 1536x1536 scene', () => {
  const large = result();
  large.image = { kind: 'synthetic-scene', width: 1536, height: 1536 };
  assert.match(summary({ accepted: true, result: validateResult(large, 'q4', '1536') }), /Image: synthetic-scene, 1536x1536/);
  assert.throws(() => validateResult(result(), 'q4', '1536'), /image size/);
  assert.throws(() => validateResult(large, 'q4', 'default'), /image size/);
  assert.throws(() => validateResult(large, 'q4', '2048'), /image side/);
});

async function scratch(fn) {
  const directory = await fs.mkdtemp(path.join(path.dirname(fileURLToPath(import.meta.url)), 'qwen38-evidence-unit-'));
  try {
    await fn(directory);
  } finally {
    await fs.rm(directory, { recursive: true, force: true });
  }
}
test('staging accepts a complete run and copies only the validated files', () => scratch(async (directory) => {
  const output = path.join(directory, 'proof');
  await fs.mkdir(output);
  await fs.writeFile(path.join(output, 'describer-result.json'), JSON.stringify(result()));
  await fs.writeFile(path.join(directory, 'receipt.json'), JSON.stringify(receipt()));
  const evidence = await stage(output, path.join(directory, 'receipt.json'), 'q4');
  assert.equal(evidence.accepted, true, evidence.error);
  assert.deepEqual((await fs.readdir(path.join(output, 'artifact'))).sort(), ['evidence.json', 'resource-receipt.json']);
  const recorded = JSON.parse(await fs.readFile(path.join(output, 'artifact', 'evidence.json'), 'utf8'));
  assert.equal(recorded.result.fields.lighting, 'lighting text | with a pipe');
}));
test('a shared-card run is accepted on a completed shared-display receipt and labels its peak', () => scratch(async (directory) => {
  const output = path.join(directory, 'proof');
  await fs.mkdir(output);
  await fs.writeFile(path.join(output, 'describer-result.json'), JSON.stringify(result()));
  await fs.writeFile(path.join(directory, 'receipt.json'), JSON.stringify(sharedReceipt()));
  const evidence = await stage(output, path.join(directory, 'receipt.json'), 'q4', 'default', 'shared');
  assert.equal(evidence.accepted, true, evidence.error);
  assert.equal(evidence.gpu_admission, 'shared');
  assert.equal(evidence.resource_receipt.admission_policy, 'shared-display');
  assert.equal(evidence.resource_receipt.shared_admission['GPU-two'].baseline_used_mib, 14259);
  assert.equal(evidence.result.gpu_memory.measured_on, 'shared-card');
  assert.equal(evidence.result.gpu_memory.peak_mib, 21000);
  assert.deepEqual((await fs.readdir(path.join(output, 'artifact'))).sort(), ['evidence.json', 'resource-receipt.json']);
}));
test('a shared-card run without a receipt, or with a strict one, is a failure', () => scratch(async (directory) => {
  const output = path.join(directory, 'proof');
  await fs.mkdir(output);
  await fs.writeFile(path.join(output, 'describer-result.json'), JSON.stringify(result()));
  const missing = await stage(output, path.join(directory, 'no-receipt.json'), 'q4', 'default', 'shared');
  assert.equal(missing.accepted, false);
  assert.match(missing.error, /no-receipt\.json/);
  const strict = path.join(directory, 'strict');
  await fs.mkdir(strict);
  await fs.writeFile(path.join(strict, 'describer-result.json'), JSON.stringify(result()));
  await fs.writeFile(path.join(directory, 'receipt.json'), JSON.stringify(receipt()));
  const wrong = await stage(strict, path.join(directory, 'receipt.json'), 'q4', 'default', 'shared');
  assert.equal(wrong.accepted, false);
  assert.match(wrong.error, /shared-display/);
}));
test('an unknown admission is refused', () => scratch(async (directory) => {
  await assert.rejects(stage(path.join(directory, 'proof'), path.join(directory, 'no-receipt.json'), 'q4', 'default', 'anything'), /admission/);
}));
test('staging records a failure when the smoke wrote no result, keeping the receipt', () => scratch(async (directory) => {
  const output = path.join(directory, 'proof');
  await fs.mkdir(output);
  await fs.writeFile(path.join(directory, 'receipt.json'), JSON.stringify(receipt(false)));
  const evidence = await stage(output, path.join(directory, 'receipt.json'), 'q4');
  assert.equal(evidence.accepted, false);
  assert.match(evidence.error, /describer-result\.json/);
  assert.equal(evidence.resource_receipt.resource_completed, false);
}));
test('staging rejects a completed result when the guard did not complete', () => scratch(async (directory) => {
  const output = path.join(directory, 'proof');
  await fs.mkdir(output);
  await fs.writeFile(path.join(output, 'describer-result.json'), JSON.stringify(result()));
  await fs.writeFile(path.join(directory, 'receipt.json'), JSON.stringify(receipt(false)));
  const evidence = await stage(output, path.join(directory, 'receipt.json'), 'q4');
  assert.equal(evidence.accepted, false);
  assert.match(evidence.error, /CUDA guard did not complete/);
}));
test('main reads the guard receipt for this run and sets a failing exit code', () => scratch(async (directory) => {
  const output = path.join(directory, 'proof');
  await fs.mkdir(output);
  const receiptDir = path.join(directory, `sceneworks-cuda-execution-7-1-${GUARD_ACTOR}`);
  await fs.mkdir(receiptDir);
  await fs.writeFile(path.join(receiptDir, 'resource-receipt.json'), JSON.stringify(receipt()));
  const previous = process.exitCode;
  try {
    await main({ MODEL_PROOF_OUTPUT_DIR: output, SCENEWORKS_QWEN38_DESCRIBER_TIER: 'q4',
      RUNNER_TEMP: directory, GITHUB_RUN_ID: '7', GITHUB_RUN_ATTEMPT: '1' });
    assert.equal(process.exitCode, 1, 'a run without a result must fail');
  } finally {
    process.exitCode = previous;
  }
  const evidence = JSON.parse(await fs.readFile(path.join(output, 'artifact', 'evidence.json'), 'utf8'));
  assert.equal(evidence.resource_receipt.resource_completed, true);
  assert.equal(evidence.accepted, false);
}));
