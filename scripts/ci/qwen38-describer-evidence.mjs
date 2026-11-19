// Validates and stages the bounded evidence of the dispatch-only Qwen3.8 27B describer CUDA run
// (SW-7120). Runs no model: it reads the result the guarded worker smoke wrote and the CUDA guard's
// resource receipt, and fails unless the described fields are filled and the guard completed. `pose` and
// `camera` may be empty strings: the output contract allows them when the image shows no people or
// no camera character.
//
// With `SCENEWORKS_QWEN38_VALIDATION_JOB=expander` it validates the MiniMax-H3-Image prompt expander
// smoke's `expander-result.json` instead (SW-7202): the selected q4/q8 Candle tier at its pinned
// folder (the folder the job LOADED, its result's `executionIdentity.weightsDir`), the upstream
// prompt files at their pinned revision and a complete well-formed expansion. `tier_isolated` is
// true when no other expander tier folder was in the snapshot: a fresh-hub run that proves the job
// needed only the selected tier.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import * as fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const FIELDS = Object.freeze(['description', 'pose', 'style', 'camera', 'composition', 'lighting', 'environment']);
// Fields the output contract allows to be empty.
export const MAY_BE_EMPTY = Object.freeze(['pose', 'camera']);
export const TIERS = Object.freeze(['q4', 'q8', 'bf16']);
// The Candle lane's quantised tiers come from this re-host; full precision from the upstream repository.
export const REHOST_REPO = 'SceneWorks/qwen3.8-27b-abliterated-candle';
export const UPSTREAM_REPO = 'huihui-ai/Huihui-Qwen3.8-27B-abliterated';
export const CATALOG_ID = 'vision_describe_qwen3_8_27b';
const SPECS_FILE = fileURLToPath(new URL('../../config/manifests/registry/specs.json', import.meta.url));

// Each tier's pinned snapshot on the Candle lane, read from the compiled catalog (the source the
// worker's catalog reads): the repository, its pinned revision, and the weights folder relative to
// the hub cache (`models--<repo>/snapshots/<revision>`, then for a quantised tier the folder its
// row's files name, e.g. `q4-full/`, the folder the worker loads).
export function pinnedSnapshots(specs = JSON.parse(readFileSync(SPECS_FILE, 'utf8'))) {
  const rows = specs?.[CATALOG_ID]?.product?.downloads ?? [];
  return Object.freeze(Object.fromEntries(TIERS.map((tier) => {
    const repo = tier === 'bf16' ? UPSTREAM_REPO : REHOST_REPO;
    const matches = rows.filter((row) => row.repo === repo && row.variant === tier);
    assert.equal(matches.length, 1, `the catalog has one ${repo} ${tier} row`);
    const { revision, files } = matches[0];
    assert.match(String(revision), /^[a-f0-9]{40}$/, `the catalog pins ${repo} ${tier} to an exact revision`);
    const folders = new Set((files ?? []).map((file) => String(file).split('/').slice(0, -1).join('/')));
    assert.equal(folders.size, 1, `the catalog's ${repo} ${tier} row names one folder`);
    const [folder] = folders;
    assert.equal(folder === '', tier === 'bf16', `only full precision loads from the ${repo} snapshot root`);
    const relativeDir = [`models--${repo.replace('/', '--')}`, 'snapshots', revision, ...(folder ? [folder] : [])].join('/');
    return [tier, Object.freeze({ repo, revision, relativeDir })];
  })));
}

export const TEST_NAME = 'qwen38_describer_describes_a_synthetic_scene_on_cuda';
export const GUARD_ACTOR = 'qwen38-describer';
// The workflow's `image_side` input and the scene size the smoke draws for it.
export const IMAGE_SIZES = Object.freeze({ default: [512, 384], 1536: [1536, 1536] });
// The worker caps one field at 16,000 characters; seven of them, JSON-escaped, stay far below this.
const MAX_INPUT_BYTES = 1024 * 1024;

export function validateResult(result, tier, imageSide = 'default', pinned = pinnedSnapshots()) {
  assert(TIERS.includes(tier), `requested tier must be one of ${TIERS.join(', ')}`);
  assert(Object.hasOwn(IMAGE_SIZES, imageSide), `requested image side must be one of ${Object.keys(IMAGE_SIZES).join(', ')}`);
  assert.equal(result?.schema_version, 1, 'unknown describer result schema');
  assert.equal(typeof result.test_fqn, 'string');
  assert.equal(result.test_fqn.split('::').at(-1), TEST_NAME, 'result is not from the describer smoke');
  assert.equal(result.tier, tier, 'result tier differs from the requested tier');
  const expected = pinned[tier];
  assert.equal(result.repo, expected.repo, `the ${tier} tier must load from ${expected.repo}`);
  assert.equal(result.revision, expected.revision, `the ${tier} tier must load the pinned revision ${expected.revision}`);
  assert(typeof result.weights_dir === 'string'
    && result.weights_dir.replaceAll('\\', '/').replace(/\/+$/, '').endsWith(`/${expected.relativeDir}`),
  `weights_dir must be the pinned snapshot folder .../${expected.relativeDir}`);
  assert.deepEqual([result.image?.width, result.image?.height], IMAGE_SIZES[imageSide],
    'result image size differs from the requested image side');
  const empty = FIELDS.filter((name) => typeof result.fields?.[name] !== 'string'
    || (!MAY_BE_EMPTY.includes(name) && !result.fields[name].trim()));
  assert.deepEqual(empty, [], `described fields are empty: ${empty.join(', ')}`);
  assert(Number.isFinite(result.wall_seconds) && result.wall_seconds > 0, 'wall time is missing');
  const memory = validateGpuMemory(result.gpu_memory);
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

// The prompt expander (SW-7202): its catalog id, the Candle tiers' re-host and the published tiers.
export const JOBS = Object.freeze(['describer', 'expander']);
export const EXPANDER_CATALOG_ID = 'minimax_h3_image_pe';
export const EXPANDER_REPO = 'SceneWorks/minimax-h3-image-pe';
export const EXPANDER_TIERS = Object.freeze(['q4', 'q8']);
export const EXPANDER_TEST_NAME = 'h3_prompt_expander_expands_a_prompt_on_cuda';

// Each published expander tier's pinned Candle folder (the Windows/Linux row's), and the `prompt`
// co-requisite's upstream snapshot, from the compiled catalog.
export function expanderPinnedSnapshots(specs = JSON.parse(readFileSync(SPECS_FILE, 'utf8'))) {
  const rows = specs?.[EXPANDER_CATALOG_ID]?.product?.downloads ?? [];
  const tiers = Object.fromEntries(EXPANDER_TIERS.map((tier) => {
    const matches = rows.filter((row) => row.repo === EXPANDER_REPO && row.variant === tier
      && (row.platforms ?? []).includes('windows'));
    assert.equal(matches.length, 1, `the catalog has one Candle ${EXPANDER_REPO} ${tier} row`);
    const { revision, files } = matches[0];
    assert.match(String(revision), /^[a-f0-9]{40}$/, `the catalog pins ${EXPANDER_REPO} ${tier} to an exact revision`);
    assert.equal(files?.length, 1, `the catalog's ${tier} row names one folder`);
    const [folder, rest] = String(files[0]).split('/');
    assert.equal(rest, '*', `the catalog's ${tier} row names its whole folder`);
    const relativeDir = [`models--${EXPANDER_REPO.replace('/', '--')}`, 'snapshots', revision, folder].join('/');
    return [tier, Object.freeze({ repo: EXPANDER_REPO, revision, relativeDir })];
  }));
  const prompts = rows.filter((row) => row.coRequisite === true && row.componentId === 'prompt');
  assert.equal(prompts.length, 1, 'the catalog has one prompt co-requisite');
  return Object.freeze({ ...tiers, prompt: Object.freeze({ repo: prompts[0].repo, revision: prompts[0].revision }) });
}

function validateGpuMemory(memory) {
  for (const key of ['baseline_mib', 'peak_mib', 'peak_delta_mib']) {
    assert(Number.isInteger(memory?.[key]) && memory[key] >= 0, `gpu_memory.${key} is missing`);
  }
  assert(memory.peak_mib > 0, 'peak GPU memory was never sampled');
  return memory;
}

export function validateExpanderResult(result, tier, pinned = expanderPinnedSnapshots()) {
  assert(EXPANDER_TIERS.includes(tier), `requested expander tier must be one of ${EXPANDER_TIERS.join(', ')}`);
  assert.equal(result?.schema_version, 1, 'unknown expander result schema');
  assert.equal(result.job, 'expander', 'result is not from the expander smoke');
  assert.equal(typeof result.test_fqn, 'string');
  assert.equal(result.test_fqn.split('::').at(-1), EXPANDER_TEST_NAME, 'result is not from the expander smoke');
  assert.equal(result.backend, 'candle', 'the CUDA run must be the Candle backend');
  assert.equal(result.tier, tier, 'result tier differs from the requested tier');
  assert.equal(result.suggestion_tier, tier, 'the suggestion block names another tier than requested');
  const expected = pinned[tier];
  assert.equal(result.repo, expected.repo, `the ${tier} tier must load from ${expected.repo}`);
  assert.equal(result.revision, expected.revision, `the ${tier} tier must load the pinned revision ${expected.revision}`);
  assert(typeof result.weights_dir === 'string'
    && result.weights_dir.replaceAll('\\', '/').replace(/\/+$/, '').endsWith(`/${expected.relativeDir}`),
  `the job must have loaded the pinned tier folder .../${expected.relativeDir}`);
  assert.deepEqual(result.prompt_files, { repo: pinned.prompt.repo, revision: pinned.prompt.revision },
    'the prompt files must come from the pinned upstream snapshot');
  assert(Array.isArray(result.other_tiers_present)
    && result.other_tiers_present.every((folder) => typeof folder === 'string'), 'other_tiers_present is missing');
  const tierIsolated = result.other_tiers_present.length === 0;
  assert.equal(result.tier_isolated, tierIsolated, 'tier_isolated disagrees with other_tiers_present');
  assert.equal(typeof result.prompt, 'string', 'the input prompt is missing');
  const expanded = result.expanded_prompt;
  assert(typeof expanded === 'string' && expanded.trim(), 'the expanded prompt is empty');
  assert.notEqual(expanded.trim(), result.prompt.trim(), 'the prompt came back unexpanded');
  assert(!/<\/?think>/.test(expanded), 'the reasoning block leaked into the expansion');
  assert.match(String(result.wh_ratio), /^[1-9]\d*:[1-9]\d*$/, 'the proposed aspect ratio is not W:H');
  assert.equal(result.workflow, 't2i');
  const generation = result.generation;
  assert.equal(generation?.finishReason, 'stop', 'the reply was not completed');
  const generated = generation?.usage?.generatedTokens;
  assert(Number.isInteger(generated) && generated > 0 && generated < generation.maxNewTokens,
    'the reply ran to its budget');
  assert(Number.isFinite(result.wall_seconds) && result.wall_seconds > 0, 'wall time is missing');
  return {
    tier: result.tier,
    revision: result.revision,
    prompt: result.prompt,
    expanded_prompt: expanded,
    wh_ratio: result.wh_ratio,
    generation,
    weights_dir: result.weights_dir,
    other_tiers_present: result.other_tiers_present,
    tier_isolated: tierIsolated,
    wall_seconds: result.wall_seconds,
    wall_time_scope: result.wall_time_scope,
    gpu_memory: validateGpuMemory(result.memory),
  };
}

// The guard policy each GPU admission must have run under. The wrapper has one admission rule for
// every actor (free memory and a stable window, never an idle card), so a run is always `shared`.
export const ADMISSION_POLICIES = Object.freeze({ shared: 'shared-free-memory' });

export function validateReceipt(receipt, admission = 'shared') {
  assert.equal(receipt?.kind, 'cuda-execution-resource-lease-v1', 'not a CUDA guard receipt');
  assert.equal(receipt.actor, GUARD_ACTOR);
  assert.equal(receipt.resource_completed, true, `the CUDA guard did not complete: ${receipt.error ?? 'no error recorded'}`);
  assert.equal(receipt.admission_policy, ADMISSION_POLICIES[admission],
    `a ${admission} run must hold the card under the ${ADMISSION_POLICIES[admission]} policy`);
  assert.equal(receipt.shared_device, true, 'a shared run receipt must record the shared card');
  assert.equal(receipt.isolated_device, false, 'a shared run receipt cannot claim an isolated card');
}

async function readBoundedJson(file) {
  const info = await fs.lstat(file);
  assert(info.isFile() && !info.isSymbolicLink(), `not a regular file: ${file}`);
  assert(info.size > 0 && info.size <= MAX_INPUT_BYTES, `unbounded input: ${file}`);
  return JSON.parse(await fs.readFile(file, 'utf8'));
}

export function summary(evidence) {
  const expander = evidence.job === 'expander';
  const title = expander ? 'MiniMax-H3-Image prompt expander CUDA validation' : 'Qwen3.8 27B describer CUDA validation';
  const lines = [`## ${title}`, '', `Accepted: **${evidence.accepted}**`, ''];
  if (evidence.error) lines.push(`Error: ${evidence.error}`, '');
  lines.push('GPU admission: shared (shared-free-memory guard: per-GPU lock and owned child, admitted on free '
    + 'memory and a stable window, no idle check; the card may be shared with the desktop). The peak below was '
    + 'measured on a shared card: card-wide, above the smoke\'s own baseline.', '');
  const result = evidence.result;
  if (result && expander) {
    const mib = (value) => `${value} MiB (${(value / 1024).toFixed(2)} GiB)`;
    lines.push(`Tier: \`${result.tier}\` at \`${result.revision}\``, '',
      `Loaded weights: \`${result.weights_dir}\``, '',
      `Tier isolated: **${result.tier_isolated}** (other tier folders present: `
        + `${result.other_tiers_present.length ? result.other_tiers_present.join(', ') : 'none'})`, '',
      `Wall time: ${result.wall_seconds.toFixed(1)} s (${result.wall_time_scope})`, '',
      `Generated tokens: ${result.generation.usage.generatedTokens} of ${result.generation.maxNewTokens}; `
        + `proposed aspect ratio ${result.wh_ratio}`, '',
      `Peak GPU memory: ${mib(result.gpu_memory.peak_mib)}; baseline ${mib(result.gpu_memory.baseline_mib)}; `
        + `delta ${mib(result.gpu_memory.peak_delta_mib)}`, '',
      `Prompt: ${result.prompt}`, '', `> ${result.expanded_prompt.replace(/\s+/g, ' ')}`, '');
  } else if (result) {
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
// `admission` is how the run took the card: always `shared`, under the guard's one shared-free-memory
// rule (the per-GPU lock and owned child, no idle check), and it needs a completed receipt of that
// policy. The peak is measured above the smoke's own baseline, on a card other programs may also
// use, so it is labelled as such.
export async function stage(outputDir, receiptFile, tier, imageSide = 'default', admission = 'shared',
  job = 'describer') {
  assert(Object.hasOwn(ADMISSION_POLICIES, admission), `unsupported GPU admission ${admission}`);
  assert(JOBS.includes(job), `unsupported job ${job}`);
  const evidence = job === 'expander'
    ? { schema_version: 1, job, accepted: false, requested_tier: tier, gpu_admission: admission }
    : { schema_version: 1, accepted: false, requested_tier: tier, requested_image_side: imageSide,
      gpu_admission: admission };
  const artifact = path.join(outputDir, 'artifact');
  await fs.mkdir(artifact, { recursive: true });
  let failure = null;
  try {
    evidence.result = job === 'expander'
      ? validateExpanderResult(await readBoundedJson(path.join(outputDir, 'expander-result.json')), tier)
      : validateResult(await readBoundedJson(path.join(outputDir, 'describer-result.json')), tier, imageSide);
    evidence.result.gpu_memory = { ...evidence.result.gpu_memory, measured_on: 'shared-card' };
  } catch (error) {
    failure = error;
  }
  // The receipt is staged even when the result is missing: it says why the guarded run failed.
  try {
    const receipt = await readBoundedJson(receiptFile);
    await fs.writeFile(path.join(artifact, 'resource-receipt.json'), `${JSON.stringify(receipt, null, 2)}\n`, { flag: 'wx' });
    evidence.resource_receipt = { actor: receipt.actor, resource_completed: receipt.resource_completed,
      admission_policy: receipt.admission_policy };
    validateReceipt(receipt, admission);
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
  const job = env.SCENEWORKS_QWEN38_VALIDATION_JOB?.trim() || 'describer';
  const evidence = await stage(outputDir, receiptFile, tier, imageSide, 'shared', job);
  if (env.GITHUB_STEP_SUMMARY) await fs.appendFile(env.GITHUB_STEP_SUMMARY, summary(evidence));
  const name = job === 'expander' ? 'MiniMax-H3-Image prompt expander' : 'Qwen3.8 27B describer';
  console.log(`${name} validation accepted=${evidence.accepted}${evidence.error ? `: ${evidence.error}` : ''}`);
  if (!evidence.accepted) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) await main();
