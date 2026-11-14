// Validates and stages the bounded evidence of the dispatch-only Qwen3.8 27B describer CUDA run
// (SW-7120). Runs no model: it reads the result the guarded worker smoke wrote and the CUDA guard's
// resource receipt, and fails unless the described fields are filled and the guard completed. `camera`
// may be an empty string: the output contract allows it when the image shows no camera character.
import assert from 'node:assert/strict';
import * as fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

export const FIELDS = Object.freeze(['description', 'style', 'camera', 'composition', 'lighting', 'environment']);
export const TIERS = Object.freeze(['q4', 'q8']);
export const REPO = 'SceneWorks/qwen3.8-27b-abliterated-candle';
export const TEST_NAME = 'qwen38_describer_describes_a_synthetic_scene_on_cuda';
export const GUARD_ACTOR = 'qwen38-describer';
// The workflow's `image_side` input and the scene size the smoke draws for it.
export const IMAGE_SIZES = Object.freeze({ default: [512, 384], 1536: [1536, 1536] });
// The worker caps one field at 16,000 characters; six of them, JSON-escaped, stay far below this.
const MAX_INPUT_BYTES = 1024 * 1024;

export function validateResult(result, tier, imageSide = 'default') {
  assert(TIERS.includes(tier), `requested tier must be one of ${TIERS.join(', ')}`);
  assert(Object.hasOwn(IMAGE_SIZES, imageSide), `requested image side must be one of ${Object.keys(IMAGE_SIZES).join(', ')}`);
  assert.equal(result?.schema_version, 1, 'unknown describer result schema');
  assert.equal(typeof result.test_fqn, 'string');
  assert.equal(result.test_fqn.split('::').at(-1), TEST_NAME, 'result is not from the describer smoke');
  assert.equal(result.repo, REPO);
  assert.match(String(result.revision), /^[a-f0-9]{40}$/, 'result must name the pinned revision');
  assert.equal(result.tier, tier, 'result tier differs from the requested tier');
  assert.deepEqual([result.image?.width, result.image?.height], IMAGE_SIZES[imageSide],
    'result image size differs from the requested image side');
  const empty = FIELDS.filter((name) => typeof result.fields?.[name] !== 'string'
    || (name !== 'camera' && !result.fields[name].trim()));
  assert.deepEqual(empty, [], `described fields are empty: ${empty.join(', ')}`);
  assert(Number.isFinite(result.wall_seconds) && result.wall_seconds > 0, 'wall time is missing');
  const memory = result.gpu_memory;
  for (const key of ['baseline_mib', 'peak_mib', 'peak_delta_mib']) {
    assert(Number.isInteger(memory?.[key]) && memory[key] >= 0, `gpu_memory.${key} is missing`);
  }
  assert(memory.peak_mib > 0, 'peak GPU memory was never sampled');
  return {
    tier: result.tier,
    revision: result.revision,
    fields: Object.fromEntries(FIELDS.map((name) => [name, result.fields[name]])),
    wall_seconds: result.wall_seconds,
    wall_time_scope: result.wall_time_scope,
    gpu_memory: memory,
    image: result.image,
  };
}

export function validateReceipt(receipt) {
  assert.equal(receipt?.kind, 'cuda-execution-resource-lease-v1', 'not a CUDA guard receipt');
  assert.equal(receipt.actor, GUARD_ACTOR);
  assert.equal(receipt.resource_completed, true, `the CUDA guard did not complete: ${receipt.error ?? 'no error recorded'}`);
}

async function readBoundedJson(file) {
  const info = await fs.lstat(file);
  assert(info.isFile() && !info.isSymbolicLink(), `not a regular file: ${file}`);
  assert(info.size > 0 && info.size <= MAX_INPUT_BYTES, `unbounded input: ${file}`);
  return JSON.parse(await fs.readFile(file, 'utf8'));
}

export function summary(evidence) {
  const lines = [`## Qwen3.8 27B describer CUDA validation`, '', `Accepted: **${evidence.accepted}**`, ''];
  if (evidence.error) lines.push(`Error: ${evidence.error}`, '');
  const result = evidence.result;
  if (result) {
    const mib = (value) => `${value} MiB (${(value / 1024).toFixed(2)} GiB)`;
    lines.push(`Tier: \`${result.tier}\` at \`${result.revision}\``, '',
      `Image: ${result.image.kind}, ${result.image.width}x${result.image.height}`, '',
      `Wall time: ${result.wall_seconds.toFixed(1)} s (${result.wall_time_scope})`, '',
      `Peak GPU memory: ${mib(result.gpu_memory.peak_mib)}; baseline ${mib(result.gpu_memory.baseline_mib)}; `
        + `delta ${mib(result.gpu_memory.peak_delta_mib)}`, '',
      '| Field | Text |', '| --- | --- |',
      ...FIELDS.map((name) => `| ${name} | ${result.fields[name].replace(/\s+/g, ' ').replace(/\|/g, '\\|')} |`), '');
  }
  return `${lines.join('\n')}\n`;
}

// Stage only what was validated: evidence.json, plus a byte copy of the guard receipt.
export async function stage(outputDir, receiptFile, tier, imageSide = 'default') {
  const evidence = { schema_version: 1, accepted: false, requested_tier: tier, requested_image_side: imageSide };
  const artifact = path.join(outputDir, 'artifact');
  await fs.mkdir(artifact, { recursive: true });
  let failure = null;
  try {
    evidence.result = validateResult(
      await readBoundedJson(path.join(outputDir, 'describer-result.json')), tier, imageSide);
  } catch (error) {
    failure = error;
  }
  // The receipt is staged even when the result is missing: it says why the guarded run failed.
  try {
    const receipt = await readBoundedJson(receiptFile);
    await fs.writeFile(path.join(artifact, 'resource-receipt.json'), `${JSON.stringify(receipt, null, 2)}\n`, { flag: 'wx' });
    evidence.resource_receipt = { actor: receipt.actor, resource_completed: receipt.resource_completed };
    validateReceipt(receipt);
  } catch (error) {
    failure ??= error;
  }
  evidence.accepted = failure === null;
  if (failure) evidence.error = String(failure?.message ?? failure);
  await fs.writeFile(path.join(artifact, 'evidence.json'), `${JSON.stringify(evidence, null, 2)}\n`, { flag: 'wx' });
  return evidence;
}

export async function main(env = process.env) {
  const outputDir = env.MODEL_PROOF_OUTPUT_DIR;
  assert(outputDir && path.isAbsolute(outputDir), 'MODEL_PROOF_OUTPUT_DIR must be an absolute directory');
  const tier = env.SCENEWORKS_QWEN38_DESCRIBER_TIER;
  assert(env.RUNNER_TEMP && /^\d+$/.test(env.GITHUB_RUN_ID ?? '') && /^\d+$/.test(env.GITHUB_RUN_ATTEMPT ?? ''),
    'the workflow run identity is required to find the guard receipt');
  // windows_cuda_execution.py writes its receipt here for this actor.
  const receiptFile = path.join(env.RUNNER_TEMP,
    `sceneworks-cuda-execution-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}-${GUARD_ACTOR}`, 'resource-receipt.json');
  const imageSide = env.SCENEWORKS_QWEN38_DESCRIBER_IMAGE_SIDE?.trim() || 'default';
  const evidence = await stage(outputDir, receiptFile, tier, imageSide);
  if (env.GITHUB_STEP_SUMMARY) await fs.appendFile(env.GITHUB_STEP_SUMMARY, summary(evidence));
  console.log(`Qwen3.8 27B describer validation accepted=${evidence.accepted}${evidence.error ? `: ${evidence.error}` : ''}`);
  if (!evidence.accepted) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) await main();
