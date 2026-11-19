// CPU-only tests of the Qwen3.8 27B describer evidence validator. No Cargo, model or CUDA runs here.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BLOCK_DRAFT_CHECKS, validateBlockDraft, EXPANDER_REPO, EXPANDER_TEST_NAME, FIELDS, GUARD_ACTOR, REHOST_REPO, TEST_NAME, UPSTREAM_REPO, expanderModelSnapshots, expanderPinnedSnapshots,
  main, pinnedSnapshots, stage, summary, validateExpanderResult, validateReceipt, validateResult } from './qwen38-describer-evidence.mjs';

const PINNED = pinnedSnapshots();

function result(tier = 'q4') {
  return {
    schema_version: 1,
    test_fqn: `qwen38_describer_gpu_smoke::${TEST_NAME}`,
    repo: PINNED[tier].repo, revision: PINNED[tier].revision, tier,
    weights_dir: `E:\\huggingface\\hub\\${PINNED[tier].relativeDir.replaceAll('/', '\\')}`,
    image: { kind: 'synthetic-scene', width: 512, height: 384 },
    fields: Object.fromEntries(FIELDS.map((name) => [name, `${name} text | with a pipe`])),
    wall_seconds: 42.5, wall_time_scope: 'fixture',
    gpu_memory: { gpu_index: '1', baseline_mib: 500, peak_mib: 21000, peak_delta_mib: 20500, method: 'fixture' },
  };
}
// Shaped as windows_cuda_execution.py writes it under its one admission rule.
function receipt(completed = true) {
  return { kind: 'cuda-execution-resource-lease-v1', actor: GUARD_ACTOR, resource_completed: completed,
    admission_policy: 'shared-free-memory', shared_device: true, isolated_device: false,
    post_child_idle_verified: false };
}
// A receipt from a retired policy: never accepted.
function retiredReceipt(policy) {
  return { ...receipt(), admission_policy: policy };
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
  ['an unsupported tier', () => {}, 'q2'],
  ['another test', (copy) => { copy.test_fqn = 'other::test'; }, 'q4'],
  ['another repository', (copy) => { copy.repo = 'someone/else'; }, 'q4'],
  ['an unpinned revision', (copy) => { copy.revision = 'main'; }, 'q4'],
  ['another cached revision', (copy) => { copy.revision = 'b'.repeat(40); }, 'q4'],
  ['a quantised tier from the upstream repository', (copy) => { copy.repo = UPSTREAM_REPO; }, 'q4'],
  ['a weights folder outside the pinned snapshot', (copy) => { copy.weights_dir = 'E:\\huggingface\\hub\\fixture'; }, 'q4'],
  ['a quantised tier at the snapshot root', (copy) => { copy.weights_dir = copy.weights_dir.replace(/\\q4-full$/, ''); }, 'q4'],
  ['a quantised tier in its older plain folder', (copy) => { copy.weights_dir = copy.weights_dir.replace(/\\q4-full$/, '\\q4'); }, 'q4'],
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
test('each tier pins its own repository, revision and weights folder from the catalog', () => {
  for (const tier of ['q4', 'q8']) {
    assert.equal(PINNED[tier].repo, REHOST_REPO);
    assert.equal(PINNED[tier].relativeDir,
      `models--SceneWorks--qwen3.8-27b-abliterated-candle/snapshots/${PINNED[tier].revision}/${tier}-full`);
  }
  assert.equal(PINNED.bf16.repo, UPSTREAM_REPO);
  assert.equal(PINNED.bf16.relativeDir, `models--huihui-ai--Huihui-Qwen3.8-27B-abliterated/snapshots/${PINNED.bf16.revision}`);
  assert.notEqual(PINNED.bf16.revision, PINNED.q4.revision);
  // A bf16 result is accepted at the upstream root and refused inside a tier folder or from the re-host.
  validateResult(result('bf16'), 'bf16');
  assert.throws(() => validateResult({ ...result('bf16'), weights_dir: `${result('bf16').weights_dir}\\bf16` }, 'bf16'), /weights_dir/);
  assert.throws(() => validateResult({ ...result('bf16'), repo: REHOST_REPO }, 'bf16'), /must load from/);
  // A catalog without the row is refused rather than guessed.
  assert.throws(() => pinnedSnapshots({ vision_describe_qwen3_8_27b: { product: { downloads: [] } } }), /one SceneWorks/);
});
test('the guard receipt must come from this actor and complete', () => {
  validateReceipt(receipt());
  assert.throws(() => validateReceipt(receipt(false)));
  assert.throws(() => validateReceipt({ ...receipt(), actor: 'iris-worker' }));
  assert.throws(() => validateReceipt({ ...receipt(), kind: 'other' }));
});
test('a run needs a receipt of the one shared guard policy', () => {
  validateReceipt(receipt(), 'shared');
  for (const policy of ['strict-idle', 'functional-coexistence', 'shared-display']) {
    assert.throws(() => validateReceipt(retiredReceipt(policy)), /shared-free-memory/);
  }
  assert.throws(() => validateReceipt(receipt(false)), /did not complete/);
  assert.throws(() => validateReceipt({ ...receipt(), shared_device: false }), /shared card/);
  assert.throws(() => validateReceipt({ ...receipt(), isolated_device: true }), /isolated card/);
  assert.throws(() => validateReceipt({ ...receipt(), actor: 'load-progress-acceptance' }));
});
test('the summary escapes table pipes and names the peak, the image size and the admission', () => {
  const text = summary({ accepted: true, gpu_admission: 'shared', result: validateResult(result(), 'q4') });
  assert.match(text, /GPU admission: shared \(shared-free-memory guard.*measured on a shared card/);
  assert.doesNotMatch(text, /strict-idle|idle before and after/);
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
test('a shared-card run is accepted on a completed shared receipt and labels its peak', () => scratch(async (directory) => {
  const output = path.join(directory, 'proof');
  await fs.mkdir(output);
  await fs.writeFile(path.join(output, 'describer-result.json'), JSON.stringify(result()));
  await fs.writeFile(path.join(directory, 'receipt.json'), JSON.stringify(receipt()));
  const evidence = await stage(output, path.join(directory, 'receipt.json'), 'q4', 'default', 'shared');
  assert.equal(evidence.accepted, true, evidence.error);
  assert.equal(evidence.gpu_admission, 'shared');
  assert.equal(evidence.resource_receipt.admission_policy, 'shared-free-memory');
  assert.equal(evidence.result.gpu_memory.measured_on, 'shared-card');
  assert.equal(evidence.result.gpu_memory.peak_mib, 21000);
  assert.deepEqual((await fs.readdir(path.join(output, 'artifact'))).sort(), ['evidence.json', 'resource-receipt.json']);
}));
test('a shared-card run without a receipt, or with a retired strict one, is a failure', () => scratch(async (directory) => {
  const output = path.join(directory, 'proof');
  await fs.mkdir(output);
  await fs.writeFile(path.join(output, 'describer-result.json'), JSON.stringify(result()));
  const missing = await stage(output, path.join(directory, 'no-receipt.json'), 'q4', 'default', 'shared');
  assert.equal(missing.accepted, false);
  assert.match(missing.error, /no-receipt\.json/);
  const strict = path.join(directory, 'strict');
  await fs.mkdir(strict);
  await fs.writeFile(path.join(strict, 'describer-result.json'), JSON.stringify(result()));
  await fs.writeFile(path.join(directory, 'receipt.json'), JSON.stringify(retiredReceipt('strict-idle')));
  const wrong = await stage(strict, path.join(directory, 'receipt.json'), 'q4', 'default', 'shared');
  assert.equal(wrong.accepted, false);
  assert.match(wrong.error, /shared-free-memory/);
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

// ---- the MiniMax-H3-Image prompt expander (job=expander, SW-7202) ------------------------------
const EXPANDER = expanderPinnedSnapshots();

function expanderResult(tier = 'q4') {
  return {
    schema_version: 1, job: 'expander',
    test_fqn: `h3_prompt_expander_gpu_smoke::${EXPANDER_TEST_NAME}`,
    backend: 'candle', repo: EXPANDER[tier].repo, revision: EXPANDER[tier].revision, tier, suggestion_tier: tier,
    weights_dir: `E:\\huggingface\\hub\\${EXPANDER[tier].relativeDir.replaceAll('/', '\\')}`,
    other_tiers_present: [], tier_isolated: true,
    prompt_files: { repo: EXPANDER.prompt.repo, revision: EXPANDER.prompt.revision },
    prompt: 'a lighthouse', expanded_prompt: 'A weathered lighthouse | on a basalt headland at dusk.',
    wh_ratio: '9:16', workflow: 't2i',
    // The recipe's own budget (`request_params.json` max_tokens at the pinned revision).
    generation: { finishReason: 'stop', usage: { promptTokens: 900, generatedTokens: 1200 }, maxNewTokens: 4096 },
    wall_seconds: 61.5, wall_time_scope: 'fixture',
    memory: { gpu_index: '1', baseline_mib: 500, peak_mib: 21000, peak_delta_mib: 20500, method: 'fixture' },
  };
}

test('the expander tiers pin their Candle folders and the upstream prompt snapshot from the catalog', () => {
  for (const tier of ['q4', 'q8']) {
    assert.equal(EXPANDER[tier].repo, EXPANDER_REPO);
    assert.equal(EXPANDER[tier].relativeDir,
      `models--SceneWorks--minimax-h3-image-pe/snapshots/${EXPANDER[tier].revision}/candle-${tier}-full`);
  }
  assert.equal(EXPANDER.prompt.repo, 'MiniMaxAI/MiniMax-H3-Image-PE');
  assert.match(EXPANDER.prompt.revision, /^[a-f0-9]{40}$/);
  assert.throws(() => expanderPinnedSnapshots({ minimax_h3_image_pe: { product: { downloads: [] } } }), /one Candle/);
});
test('a complete expander result is accepted for its own tier', () => {
  const accepted = validateExpanderResult(expanderResult('q8'), 'q8');
  assert.equal(accepted.tier, 'q8');
  assert.equal(accepted.wh_ratio, '9:16');
  assert.equal(accepted.gpu_memory.peak_mib, 21000);
});
for (const [name, mutate, tier] of [
  ['an unpublished tier', () => {}, 'bf16'],
  ['a different tier than requested', (copy) => { copy.tier = 'q8'; }, 'q4'],
  ['a suggestion block naming another tier', (copy) => { copy.suggestion_tier = 'q8'; }, 'q4'],
  ['a tier_isolated flag that hides another tier', (copy) => { copy.other_tiers_present = ['candle-q8-full']; }, 'q4'],
  ['a tier_isolated flag that is missing', (copy) => { delete copy.tier_isolated; }, 'q4'],
  ['a describer result', (copy) => { copy.job = 'describer'; }, 'q4'],
  ['another test', (copy) => { copy.test_fqn = 'other::test'; }, 'q4'],
  ['the MLX backend', (copy) => { copy.backend = 'mlx'; }, 'q4'],
  ['another cached revision', (copy) => { copy.revision = 'b'.repeat(40); }, 'q4'],
  ['the other tier folder', (copy) => { copy.weights_dir = copy.weights_dir.replace(/candle-q4-full$/, 'candle-q8-full'); }, 'q4'],
  ['the MLX layout folder', (copy) => { copy.weights_dir = copy.weights_dir.replace(/candle-q4-full$/, 'q4-full'); }, 'q4'],
  ['the older plain folder', (copy) => { copy.weights_dir = copy.weights_dir.replace(/candle-q4-full$/, 'candle-q4'); }, 'q4'],
  ['prompt files from another revision', (copy) => { copy.prompt_files.revision = 'c'.repeat(40); }, 'q4'],
  ['an empty expansion', (copy) => { copy.expanded_prompt = '  '; }, 'q4'],
  ['an unexpanded prompt', (copy) => { copy.expanded_prompt = 'a lighthouse'; }, 'q4'],
  ['a leaked reasoning block', (copy) => { copy.expanded_prompt += ' </think>'; }, 'q4'],
  ['a malformed ratio', (copy) => { copy.wh_ratio = 'tall'; }, 'q4'],
  ['the edit workflow', (copy) => { copy.workflow = 'edit'; }, 'q4'],
  ['a reply cut off at its budget', (copy) => { copy.generation.finishReason = 'length'; }, 'q4'],
  ['a reply that used the whole budget', (copy) => { copy.generation.usage.generatedTokens = 4096; }, 'q4'],
  ['an unrecorded tier listing', (copy) => { delete copy.other_tiers_present; }, 'q4'],
  ['missing peak memory', (copy) => { delete copy.memory.peak_mib; }, 'q4'],
]) {
  test(`reject an expander result with ${name}`, () => {
    const copy = expanderResult();
    mutate(copy);
    assert.throws(() => validateExpanderResult(copy, tier));
  });
}
test('a run on a cache holding another tier is accepted but recorded as not isolated', () => {
  const shared = validateExpanderResult({ ...expanderResult(), other_tiers_present: ['candle-q8-full'],
    tier_isolated: false }, 'q4');
  assert.equal(shared.tier_isolated, false);
  assert.equal(validateExpanderResult(expanderResult(), 'q4').tier_isolated, true);
});
test('the expander summary names the loaded folder, the isolation, the ratio and the expansion', () => {
  const text = summary({ job: 'expander', accepted: true, gpu_admission: 'shared',
    result: validateExpanderResult({ ...expanderResult(), other_tiers_present: ['candle-q8-full'],
      tier_isolated: false }, 'q4') });
  assert.match(text, /## MiniMax-H3-Image prompt expander CUDA validation/);
  assert.match(text, /Tier isolated: \*\*false\*\* \(other tier folders present: candle-q8-full\)/);
  assert.match(text, /Loaded weights: `E:\\huggingface\\hub\\models--SceneWorks--minimax-h3-image-pe/);
  assert.match(summary({ job: 'expander', accepted: true, result: validateExpanderResult(expanderResult(), 'q4') }),
    /Tier isolated: \*\*true\*\* \(other tier folders present: none\)/);
  assert.match(text, /proposed aspect ratio 9:16/);
  assert.match(text, /> A weathered lighthouse/);
  assert.match(text, /Peak GPU memory: 21000 MiB/);
});
test('staging an expander run reads its own result and keeps the receipt', () => scratch(async (directory) => {
  const output = path.join(directory, 'proof');
  await fs.mkdir(output);
  await fs.writeFile(path.join(output, 'expander-result.json'), JSON.stringify(expanderResult()));
  await fs.writeFile(path.join(directory, 'receipt.json'), JSON.stringify(receipt()));
  const evidence = await stage(output, path.join(directory, 'receipt.json'), 'q4', 'default', 'shared', 'expander');
  assert.equal(evidence.accepted, true, evidence.error);
  assert.equal(evidence.job, 'expander');
  assert.equal(evidence.result.gpu_memory.measured_on, 'shared-card');
  assert.deepEqual((await fs.readdir(path.join(output, 'artifact'))).sort(), ['evidence.json', 'resource-receipt.json']);
  // A describer result does not satisfy an expander run.
  const crossed = path.join(directory, 'crossed');
  await fs.mkdir(crossed);
  await fs.writeFile(path.join(crossed, 'describer-result.json'), JSON.stringify(result()));
  const wrong = await stage(crossed, path.join(directory, 'receipt.json'), 'q4', 'default', 'shared', 'expander');
  assert.equal(wrong.accepted, false);
  assert.match(wrong.error, /expander-result\.json/);
  await assert.rejects(stage(output, path.join(directory, 'receipt.json'), 'q4', 'default', 'shared', 'captioner'), /job/);
}));
test('main validates the expander when the workflow selects job=expander', () => scratch(async (directory) => {
  const output = path.join(directory, 'proof');
  await fs.mkdir(output);
  await fs.writeFile(path.join(output, 'expander-result.json'), JSON.stringify(expanderResult('q8')));
  const receiptDir = path.join(directory, `sceneworks-cuda-execution-7-1-${GUARD_ACTOR}`);
  await fs.mkdir(receiptDir);
  await fs.writeFile(path.join(receiptDir, 'resource-receipt.json'), JSON.stringify(receipt()));
  const previous = process.exitCode;
  try {
    await main({ MODEL_PROOF_OUTPUT_DIR: output, SCENEWORKS_QWEN38_DESCRIBER_TIER: 'q8',
      SCENEWORKS_QWEN38_VALIDATION_JOB: 'expander', RUNNER_TEMP: directory, GITHUB_RUN_ID: '7', GITHUB_RUN_ATTEMPT: '1' });
    assert.notEqual(process.exitCode, 1);
  } finally {
    process.exitCode = previous;
  }
  const evidence = JSON.parse(await fs.readFile(path.join(output, 'artifact', 'evidence.json'), 'utf8'));
  assert.equal(evidence.accepted, true, evidence.error);
  assert.equal(evidence.result.tier, 'q8');
}));

// ---- the abliterated describer as the expander's model (SW-7208) --------------------------------
function abliteratedResult(tier = 'q4') {
  const pinned = expanderModelSnapshots('abliterated');
  return {
    ...expanderResult(tier),
    expander_model: 'abliterated', catalog_id: 'vision_describe_qwen3_8_27b',
    rewriter_model_id: 'vision_describe_qwen3_8_27b',
    repo: pinned[tier].repo, revision: pinned[tier].revision,
    weights_dir: `E:\\huggingface\\hub\\${pinned[tier].relativeDir.replaceAll('/', '\\')}`,
  };
}

test('the abliterated expander loads the describer tier folders with the expander prompt files', () => {
  const pinned = expanderModelSnapshots('abliterated');
  for (const tier of ['q4', 'q8']) {
    assert.equal(pinned[tier].repo, REHOST_REPO);
    assert.match(pinned[tier].relativeDir, new RegExp(`/${tier}-full$`));
  }
  assert.deepEqual(pinned.prompt, EXPANDER.prompt);
  const accepted = validateExpanderResult(abliteratedResult('q8'), 'q8', pinned, 'abliterated');
  assert.equal(accepted.catalog_id, 'vision_describe_qwen3_8_27b');
  assert.equal(accepted.expander_model, 'abliterated');
});
test('an expander result is accepted only for the model that ran', () => {
  const pinned = expanderModelSnapshots('abliterated');
  // MiniMax-H3-Image-PE's result does not satisfy an abliterated request, nor the reverse.
  assert.throws(() => validateExpanderResult(expanderResult(), 'q4', pinned, 'abliterated'), /another expander model/);
  assert.throws(() => validateExpanderResult(abliteratedResult(), 'q4', EXPANDER, 'minimax'), /another expander model/);
  // The job record must name the model that ran.
  assert.throws(() => validateExpanderResult({ ...abliteratedResult(), rewriter_model_id: 'minimax_h3_image_pe' }, 'q4',
    pinned, 'abliterated'), /names another model/);
  // The describer's folder under the expander's name is refused.
  assert.throws(() => validateExpanderResult({ ...abliteratedResult(), expander_model: 'minimax',
    catalog_id: 'minimax_h3_image_pe', rewriter_model_id: 'minimax_h3_image_pe' }, 'q4', EXPANDER, 'minimax'));
  assert.throws(() => validateExpanderResult(abliteratedResult(), 'q4', pinned, 'captioner'), /expander model/);
});
test('main validates the abliterated expander when the workflow selects it', () => scratch(async (directory) => {
  const output = path.join(directory, 'proof');
  await fs.mkdir(output);
  await fs.writeFile(path.join(output, 'expander-result.json'), JSON.stringify(abliteratedResult('q4')));
  const receiptDir = path.join(directory, `sceneworks-cuda-execution-9-1-${GUARD_ACTOR}`);
  await fs.mkdir(receiptDir);
  await fs.writeFile(path.join(receiptDir, 'resource-receipt.json'), JSON.stringify(receipt()));
  const previous = process.exitCode;
  try {
    await main({ MODEL_PROOF_OUTPUT_DIR: output, SCENEWORKS_QWEN38_DESCRIBER_TIER: 'q4',
      SCENEWORKS_QWEN38_VALIDATION_JOB: 'expander', SCENEWORKS_H3_EXPANDER_MODEL: 'abliterated',
      RUNNER_TEMP: directory, GITHUB_RUN_ID: '9', GITHUB_RUN_ATTEMPT: '1' });
    assert.notEqual(process.exitCode, 1);
  } finally {
    process.exitCode = previous;
  }
  const evidence = JSON.parse(await fs.readFile(path.join(output, 'artifact', 'evidence.json'), 'utf8'));
  assert.equal(evidence.accepted, true, evidence.error);
  assert.equal(evidence.requested_expander_model, 'abliterated');
  assert.equal(evidence.result.catalog_id, 'vision_describe_qwen3_8_27b');
  assert.match(summary(evidence), /Model: `vision_describe_qwen3_8_27b`/);
}));

// SW-7204: the block-draft record the smoke wrote, re-checked per the run's `block_draft` input.
function blockDraft(check, overrides = {}) {
  return {
    check,
    decode: {
      proposer: 'block_draft', verify: 'tree', targetPasses: 40, rowsCheckedPerPass: 9.5,
      acceptedTokensPerPass: 2.4, draftFallback: null,
      draft: { repo: 'SceneWorks/qwen3.8-27b-dflash2', revision: '9030fabfb5e3ca0c5c180f84927af86aa9f9a2e9',
        format: 'dflash2', precision: 'q4' },
      ...overrides,
    },
  };
}

test('the block-draft checks are record, on and no_fit, and record asserts nothing', () => {
  assert.deepEqual(BLOCK_DRAFT_CHECKS, ['record', 'on', 'no_fit']);
  assert.equal(validateBlockDraft(undefined), null);
  assert.throws(() => validateBlockDraft(blockDraft('on'), 'sometimes'), /must be one of/);
});

test('an on run needs the draft, its precision, a tree and more than one token per pass', () => {
  assert.deepEqual(validateBlockDraft(blockDraft('on'), 'on'), blockDraft('on'));
  assert.throws(() => validateBlockDraft(blockDraft('record'), 'on'), /ran the record block-draft check/);
  assert.throws(() => validateBlockDraft(blockDraft('on', { proposer: 'mtp' }), 'on'), /did not run/);
  assert.throws(() => validateBlockDraft(blockDraft('on', { verify: 'linear' }), 'on'), /tree/);
  assert.throws(() => validateBlockDraft(blockDraft('on', { acceptedTokensPerPass: 1 }), 'on'), /above 1.0/);
  assert.throws(() => validateBlockDraft(blockDraft('on', { draftFallback: 'x' }), 'on'), /fell back/);
});

test('a no_fit run must decode without the draft and name why', () => {
  const fellBack = blockDraft('no_fit', {
    proposer: 'mtp', draft: null,
    draftFallback: "draft model: its load needs an estimated 2 bytes beside the target's 9, but only 10 bytes are available",
  });
  assert.deepEqual(validateBlockDraft(fellBack, 'no_fit'), fellBack);
  assert.throws(() => validateBlockDraft(blockDraft('no_fit'), 'no_fit'), /could not fit/);
  assert.throws(() => validateBlockDraft(blockDraft('no_fit', { proposer: 'mtp', draft: null }), 'no_fit'),
    /did not fit/);
});
