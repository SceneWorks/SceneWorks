// Policy, provenance and controller startup only. Never executes learned code or Cargo.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync, spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { discoverRootTests } from '../run-root-tests.mjs';
import { ROOT, CASES, FEATURE, body, checkRouting, admission, lockedPin, discover, validateExecution } from './aesthetic-cpu-validation.mjs';
const read = p => readFileSync(path.join(ROOT, p), 'utf8');
const manifest = checkRouting();
const pin = lockedPin(read('Cargo.lock'));
const sha = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: ROOT, encoding: 'utf8' }).trim();
const valid = { ...process.env, GITHUB_EVENT_NAME: 'workflow_dispatch', GITHUB_REPOSITORY: 'SceneWorks/SceneWorks', GITHUB_WORKFLOW_REF: 'SceneWorks/SceneWorks/.github/workflows/nax-kernel-validation.yml@refs/heads/fixture', AESTHETIC_SUITE: 'aesthetic-cpu', AESTHETIC_SOURCE_SHA: sha, AESTHETIC_DISPATCH_SHA: sha, AESTHETIC_INFERENCE_SHA: pin };

test('every aesthetic learned call in nearby tests has explicit proof feature and ignored routing', () => {
  assert.equal(CASES.filter(c => c.package === 'sceneworks-core').length, 2);
  for (const packageName of ['sceneworks-core', 'sceneworks-image-quality']) {
    const source = read(`crates/${packageName}/src/${packageName === 'sceneworks-core' ? 'dataset_quality.rs' : 'lib.rs'}`);
    const tests = source.slice(source.indexOf('#[cfg(test)]'));
    const callBodies = [...tests.matchAll(/^    fn (\w+)\([^\n]*\) (?:-> [^{]+ )?\{[\s\S]*?^    \}/gm)].filter(m => /\.predict\(|evaluate_aesthetic\(/.test(m[0]));
    const expected = CASES.filter(c => c.package === packageName).map(c => c.fqn.split('::').at(-1));
    if (packageName === 'sceneworks-core') expected.push('aesthetic_predictor_parses_and_refuses_invalid_embeddings', 'aesthetic_non_style_refuses_before_model_execution');
    assert.deepEqual(callBodies.map(m => m[1]).sort(), expected.sort(), 'new learned/wrapper call must be classified explicitly');
    for (const c of CASES.filter(c => c.package === packageName)) {
      const proof = body(source, c.fqn), at = source.indexOf(proof);
      const attrs = source.slice(source.lastIndexOf('\n    }', at) + 6, at);
      assert.ok(attrs.includes(`#[cfg(feature = "${FEATURE}")]`));
      assert.match(attrs, /#\[ignore = "learned CPU MLP execution; explicit aesthetic-cpu manual validation only"\]/);
    }
  }
  const core = read(CASES[0].source);
  const refusal = body(core, 'aesthetic_predictor_parses_and_refuses_invalid_embeddings');
  assert.deepEqual([...refusal.matchAll(/predict\(&\[(.*?)\]\)/g)].map(m => m[1]), ['0.0, 0.0', '1.0, 2.0, 3.0']);
  assert.match(refusal, /assert_eq!\(predictor.input_dim\(\), 2\)/);
  const nonStyle = body(core, 'aesthetic_non_style_refuses_before_model_execution');
  assert.match(nonStyle, /layers: vec!\[\]/);
  assert.match(nonStyle, /for kind in \[DatasetKind::Person, DatasetKind::Object\]/);
  assert.match(nonStyle, /evaluate_aesthetic\(&\[\],/);
  assert.doesNotMatch(body(read(CASES[2].source), 'bundled_aesthetic_predictor_loads_without_execution'), /\.predict\(/);
});

test('manual dispatcher adds one hosted CPU suite while preserving NAX default and source-only ordinary checks', () => {
  const workflow = read('.github/workflows/nax-kernel-validation.yml');
  const triggers = workflow.slice(workflow.indexOf('\non:'), workflow.indexOf('\npermissions:'));
  assert.doesNotMatch(triggers, /\n  (?:push|pull_request|pull_request_target|schedule|merge_group|workflow_run):/);
  assert.match(triggers, /default: nax/);
  assert.match(triggers, /options: \[nax, aesthetic-cpu, cv\]/);
  // workflow_call has no suite input: the existing guarded NAX caller retains its default.
  assert.doesNotMatch(triggers.split('  workflow_call:')[1], /suite:|inference_sha:/);
  const cpu = workflow.slice(workflow.indexOf('\n  aesthetic-cpu:'));
  assert.match(cpu, /github.event_name == 'workflow_dispatch'.*inputs.suite == 'aesthetic-cpu'/);
  assert.match(cpu, /runs-on: ubuntu-latest/);
  assert.doesNotMatch(cpu, /self-hosted|metal_execution|cuda_execution|\$\{\{ inputs\.[^}]+\}\}[^\n]*\n\s*run:/);
  assert.match(cpu, /run: node scripts\/ci\/aesthetic-cpu-validation.mjs --preflight/);
  assert.match(cpu, /run: node scripts\/ci\/aesthetic-cpu-validation.mjs --run/);
  for (const block of [...cpu.matchAll(/run: ([^\n]*(?:\n(?:          .*|\s*))*)/g)]) assert.doesNotMatch(block[0], /\$\{\{\s*inputs\./, 'dispatch values enter commands through env only');
  const normalCaller = read('.github/workflows/macos-mlx.yml');
  const caller = normalCaller.slice(normalCaller.indexOf('\n  nax-kernel-validation:'), normalCaller.indexOf('\n  manual-verification:'));
  assert.match(caller, /github.event_name == 'workflow_dispatch'/);
  assert.doesNotMatch(caller, /suite:|aesthetic-cpu/);
  const dispatchers = readdirSync(path.join(ROOT, '.github/workflows')).filter(name => /\.ya?ml$/.test(name) && read(`.github/workflows/${name}`).includes('aesthetic-cpu-validation.mjs --run'));
  assert.deepEqual(dispatchers, ['nax-kernel-validation.yml'], 'learned execution entrypoint belongs only to its explicit manual dispatcher');
  const pkg = JSON.parse(read('package.json'));
  assert.equal(pkg.scripts.check, 'node scripts/run-root-tests.mjs');
  assert.ok(discoverRootTests().includes('scripts/ci/aesthetic-cpu-validation.test.mjs'));
  assert.ok(discoverRootTests().includes('scripts/lib/script-helpers.test.mjs'));
  assert.ok(discoverRootTests().includes('scripts/historical-artifacts.test.mjs'));
});

test('dispatch admission rejects normal CI, wrong workflows, moved source and pin substitution', () => {
  admission(valid, sha, pin);
  for (const patch of [{ GITHUB_EVENT_NAME: 'push' }, { GITHUB_EVENT_NAME: 'pull_request' }, { GITHUB_EVENT_NAME: 'workflow_call' }, { GITHUB_REPOSITORY: 'fork/SceneWorks' }, { GITHUB_WORKFLOW_REF: 'SceneWorks/SceneWorks/.github/workflows/check.yml@refs/heads/main' }, { AESTHETIC_SUITE: 'nax' }, { AESTHETIC_SOURCE_SHA: 'main; echo bad' }, { AESTHETIC_DISPATCH_SHA: '0'.repeat(40) }, { AESTHETIC_INFERENCE_SHA: '0'.repeat(40) }]) assert.throws(() => admission({ ...valid, ...patch }, sha, pin));
  assert.throws(() => lockedPin('no inference dependency'));
  assert.throws(() => lockedPin(`git+https://github.com/SceneWorks/inference?rev=${pin}#${'0'.repeat(40)}`));
});

test('exact two core ignored cases and one bundled case reject missing/duplicate/additional discovery', () => {
  const expected = CASES.slice(0, 2).map(c => c.fqn);
  const text = `${expected.map(fqn => `${fqn}: test`).join('\n')}\n2 tests, 0 benchmarks\n`;
  assert.deepEqual(discover(text, expected), [...expected].sort());
  for (const bad of ['', text.replace(expected[0], 'other'), `${text}${expected[0]}: test\n`, text.replace('2 tests', '0 tests')]) assert.throws(() => discover(bad, expected));
  const bundled = CASES[2].fqn;
  assert.deepEqual(discover(`${bundled}: test\n1 test, 0 benchmarks\n`, [bundled]), [bundled]);
});

function log(c, fixture) {
  const packet = { case: c.fqn, backend: 'rust-vec-f64', device: 'cpu', score: fixture.expected_score ?? 4.5, model_sha256: fixture.model_sha256, inputs: fixture.inputs };
  return `\nAESTHETIC_BODY_COMPLETED ${JSON.stringify(packet)}\ntest result: ok. 1 passed; 0 failed; 0 ignored;\n`;
}
test('actual body identity, CPU backend, synthetic model/input/result and unignored execution are all required', () => {
  for (const c of CASES) {
    const fixture = manifest.fixtures[c.fixture], text = log(c, fixture);
    assert.equal(validateExecution(text, 0, c, fixture).body_completed, 1);
    for (const bad of [text.replace('1 passed', '0 passed'), text.replace('0 ignored', '1 ignored'), text.replace('0 failed', '1 failed'), text.replace(c.fqn, 'other'), text.replace('"cpu"', '"gpu"'), text.replace('rust-vec-f64', 'unknown'), text + text, text.replace(/^AESTHETIC_BODY_COMPLETED .*\n/m, '')]) assert.throws(() => validateExecution(bad, 0, c, fixture));
    assert.throws(() => validateExecution(text, 1, c, fixture));
    if (fixture.kind === 'synthetic') {
      assert.throws(() => validateExecution(text.replace(fixture.model_sha256, '0'.repeat(64)), 0, c, fixture));
      assert.throws(() => validateExecution(text.replace('"inputs":', '"wrong_inputs":'), 0, c, fixture));
      assert.throws(() => validateExecution(text.replace(`"score":${fixture.expected_score}`, '"score":99'), 0, c, fixture));
    }
  }
});

test('assembled entrypoint --check starts without dispatch or models; --preflight refuses ordinary CI', () => {
  const script = path.join(ROOT, 'scripts/ci/aesthetic-cpu-validation.mjs');
  const checked = spawnSync(process.execPath, [script, '--check'], { cwd: ROOT, encoding: 'utf8', timeout: 15_000 });
  assert.equal(checked.status, 0, checked.stderr);
  assert.match(checked.stdout, /no model executed/);
  const directory = mkdtempSync(path.join(tmpdir(), 'aesthetic-routing-'));
  try {
    const refused = spawnSync(process.execPath, [script, '--preflight'], { cwd: ROOT, env: { ...valid, GITHUB_EVENT_NAME: 'push', AESTHETIC_OUTPUT: directory }, encoding: 'utf8', timeout: 15_000 });
    assert.notEqual(refused.status, 0);
    assert.match(refused.stderr, /explicit dedicated manual dispatch required/);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});


test('API readiness and caption coverage retain ordinary refusal fixtures; only ignored wrapper enables learned dimension', () => {
  const api = read('apps/rust-api/src/tests/training.rs');
  assert.match(body(api, 'training_dataset_readiness_reports_and_persists_tier0_cache'), /training_dataset_readiness_fixture\(false\)/);
  const c = CASES.find(c => c.package === 'sceneworks-rust-api');
  const proof = body(api, c.fqn), at = api.indexOf(proof);
  const attrs = api.slice(api.lastIndexOf('\n\n', at) + 2, at);
  assert.match(attrs, /#\[cfg\(feature = "aesthetic-inference-proof"\)\]/);
  assert.match(attrs, /#\[ignore = "learned CPU MLP execution; explicit aesthetic-cpu manual validation only"\]/);
  assert.match(proof, /training_dataset_readiness_fixture\(true\)/);
  assert.equal([...api.matchAll(/training_dataset_readiness_fixture\(true\)/g)].length, 1);
  assert.match(api, /"textEmbedding": \[1.0\]/, 'the wrong-length refusal remains wrong for both 2-d and 768-d embeddings');
  assert.match(api, /vec!\[0.0; if execute_aesthetic \{ 768 \} else \{ 2 \}\]/);
  assert.match(body(api, 'training_dataset_caption_alignment_requires_current_text_embedding_coverage'), /let embedding: Vec<f64> = vec!\[1.0, 0.0\];/);
  // Other endpoint calls use only face sidecars (Person path or empty image sidecar), never MLP.
  assert.doesNotMatch(api, /vec!\[0\.0;\s*768\]/, 'a new full-dimensional sidecar fixture requires learned-routing review');
});


// GitHub's context-availability table deliberately excludes runner/job/steps/env from
// jobs.<job_id>.env. They are available after allocation, in steps or runtime variables.
// https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#context-availability
function assertJobEnvContexts(job) {
  const env = job.match(/^    env:\n((?:      [^\n]*\n)*)/m)?.[1];
  assert.ok(env, 'job environment is present');
  const allowed = new Set(['github', 'needs', 'strategy', 'matrix', 'vars', 'secrets', 'inputs']);
  for (const expression of env.matchAll(/\$\{\{([\s\S]*?)\}\}/g)) {
    for (const root of expression[1].matchAll(/\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\./g)) {
      assert.ok(allowed.has(root[1]), `context ${root[1]} is unavailable in jobs.<job_id>.env`);
    }
  }
}

test('job environment rejects unavailable runner contexts and the actual initialization step exports shell-safe paths', () => {
  const workflow = read('.github/workflows/nax-kernel-validation.yml');
  const cpu = workflow.slice(workflow.indexOf('\n  aesthetic-cpu:'));
  assertJobEnvContexts(cpu);
  for (const context of ['runner', 'job', 'steps', 'env']) {
    const broken = cpu.replace('    env:\n', `    env:\n      INVALID: \${{ ${context}.temp }}/evidence\n`);
    assert.throws(() => assertJobEnvContexts(broken), /unavailable in jobs/);
  }
  const step = cpu.match(/- name: Initialize CPU evidence paths on the assigned runner\n        shell: bash\n        run: \|\n([\s\S]*?)(?=\n      - )/);
  assert.ok(step, 'actual runner initialization entrypoint exists');
  assert.ok(cpu.indexOf(step[0]) < cpu.indexOf('node scripts/ci/aesthetic-cpu-validation.mjs --preflight'));
  const directory = mkdtempSync(path.join(tmpdir(), 'aesthetic runner paths '));
  try {
    const envFile = path.join(directory, 'github env');
    const result = spawnSync('bash', ['-euo', 'pipefail', '-c', step[1].replace(/^          /gm, '')], {
      env: { ...process.env, RUNNER_TEMP: directory, GITHUB_ENV: envFile }, encoding: 'utf8', timeout: 15_000,
    });
    assert.equal(result.status, 0, result.stderr);
    assert.equal(readFileSync(envFile, 'utf8'), `AESTHETIC_OUTPUT=${directory}/aesthetic-cpu-evidence\nCARGO_TARGET_DIR=${directory}/aesthetic-cpu-target\n`);
    assert.match(cpu, /path: \$\{\{ runner.temp \}\}\/aesthetic-cpu-evidence\//, 'artifact step retains valid step-level runner context');
  } finally { rmSync(directory, { recursive: true, force: true }); }
});
