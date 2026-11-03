// Explicit manual-only learned validation. Importing, --check and --preflight execute no model.
import { createHash } from 'node:crypto';
import { execFileSync, spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { hostname, cpus, platform, arch } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = path.resolve(fileURLToPath(new URL('../../', import.meta.url)));
export const FEATURE = 'model-inference-proof';
export const MANIFEST = 'scripts/ci/fixtures/aesthetic-cpu.json';
export const CASES = Object.freeze([
  { package: 'sceneworks-core', source: 'crates/sceneworks-core/src/dataset_quality.rs', fqn: 'dataset_quality::tests::aesthetic_predictor_parses_and_forwards', fixture: 'synthetic-chain', feature: FEATURE },
  { package: 'sceneworks-core', source: 'crates/sceneworks-core/src/dataset_quality.rs', fqn: 'dataset_quality::tests::aesthetic_is_style_only_and_advisory', fixture: 'synthetic-constant', feature: FEATURE },
  { package: 'sceneworks-image-quality', source: 'crates/sceneworks-image-quality/src/lib.rs', fqn: 'tests::bundled_aesthetic_predictor_loads_and_scores', fixture: 'bundled-laion-head', feature: FEATURE },
  { package: 'sceneworks-rust-api', source: 'apps/rust-api/src/tests/training.rs', fqn: 'tests::training::training_dataset_aesthetic_readiness_executes_bundled_head', fixture: 'api-bundled-laion-head', feature: 'aesthetic-inference-proof' },
]);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const read = name => readFileSync(path.join(ROOT, name));
const git = (...args) => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();

export function body(source, fqn) {
  const name = fqn.split('::').at(-1);
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const matches = [...source.matchAll(new RegExp(`^( *)(?:async )?fn ${escaped}\\(\\) \\{[\\s\\S]*?^\\1\\}`, 'gm'))];
  if (matches.length !== 1) throw new Error(`exactly one source body required: ${fqn}`);
  return matches[0][0];
}

export function checkRouting() {
  const manifest = JSON.parse(read(MANIFEST));
  if (manifest.schema_version !== 1 || manifest.backend !== 'rust-vec-f64' || manifest.device !== 'cpu') throw new Error('fixture backend/device mismatch');
  for (const c of CASES) {
    const source = read(c.source).toString();
    const proof = body(source, c.fqn);
    const at = source.indexOf(proof);
    const annotations = source.slice(source.lastIndexOf('\n\n', at) + 2, at);
    if (!annotations.includes(`#[cfg(feature = "${c.feature}")]`) || !/#\[(?:tokio::)?test\]/.test(annotations) || !annotations.includes('#[ignore = "learned CPU MLP execution; explicit aesthetic-cpu manual validation only"]')) throw new Error(`missing feature/ignore route: ${c.fqn}`);
    if (!proof.includes('AESTHETIC_BODY_COMPLETED')) throw new Error(`missing terminal body marker: ${c.fqn}`);
    if (!manifest.fixtures[c.fixture]) throw new Error(`missing fixture identity: ${c.fixture}`);
  }
  const asset = manifest.fixtures['bundled-laion-head'];
  if (hash(read(asset.path)) !== asset.model_sha256) throw new Error('bundled model SHA mismatch');
  return manifest;
}

export function discover(text, expected) {
  const found = [...text.matchAll(/^(.+): test$/gm)].map(m => m[1]).sort();
  if (JSON.stringify(found) !== JSON.stringify([...expected].sort())) throw new Error(`exact ignored discovery mismatch: ${JSON.stringify(found)}`);
  if (!new RegExp(`^${expected.length} tests?, 0 benchmarks$`, 'm').test(text)) throw new Error('missing exact discovery count');
  return found;
}

export function validateExecution(text, status, c, fixture) {
  if (status !== 0) throw new Error(`test process failed: ${status}`);
  const counts = [...text.matchAll(/^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;/gm)];
  if (counts.length !== 1 || counts[0][1] !== '1' || counts[0][2] !== '0' || counts[0][3] !== '0') throw new Error('expected exactly one executed, unignored passing case');
  const packets = [...text.matchAll(/^AESTHETIC_BODY_COMPLETED (.+)$/gm)].map(m => JSON.parse(m[1]));
  if (packets.length !== 1) throw new Error('expected exactly one completed test body');
  const packet = packets[0];
  if (packet.case !== c.fqn || packet.backend !== 'rust-vec-f64' || packet.device !== 'cpu' || !Number.isFinite(packet.score)) throw new Error('body identity/backend/device/result mismatch');
  if (fixture.kind === 'synthetic') {
    if (packet.model_sha256 !== fixture.model_sha256 || JSON.stringify(packet.inputs) !== JSON.stringify(fixture.inputs) || Math.abs(packet.score - fixture.expected_score) > 1e-9) throw new Error('synthetic fixture/input/result mismatch');
  }
  return { ...packet, model_sha256: fixture.model_sha256, passed: 1, failed: 0, ignored: 0, body_completed: 1 };
}

export function admission(env, head, pin) {
  if (env.GITHUB_EVENT_NAME !== 'workflow_dispatch' || env.GITHUB_REPOSITORY !== 'SceneWorks/SceneWorks' || env.AESTHETIC_SUITE !== 'aesthetic-cpu' || !/^SceneWorks\/SceneWorks\/\.github\/workflows\/nax-kernel-validation\.yml@/.test(env.GITHUB_WORKFLOW_REF ?? '')) throw new Error('explicit dedicated manual dispatch required');
  if (!/^[a-f0-9]{40}$/.test(env.AESTHETIC_SOURCE_SHA ?? '') || head !== env.AESTHETIC_SOURCE_SHA || head !== env.AESTHETIC_DISPATCH_SHA) throw new Error('exact selected dispatch/source SHA required');
  if (!/^[a-f0-9]{40}$/.test(env.AESTHETIC_INFERENCE_SHA ?? '') || pin !== env.AESTHETIC_INFERENCE_SHA) throw new Error('selected inference identity differs from locked source');
}

export function lockedPin(lock) {
  const pins = [...lock.matchAll(/git\+https:\/\/github.com\/SceneWorks\/inference\?rev=([a-f0-9]{40})#([a-f0-9]{40})/g)];
  if (!pins.length || pins.some(m => m[1] !== m[2] || m[1] !== pins[0][1])) throw new Error('ambiguous locked inference identity');
  return pins[0][1];
}

function preflight(env) {
  const manifest = checkRouting();
  const head = git('rev-parse', 'HEAD');
  const inference = lockedPin(read('Cargo.lock').toString());
  admission(env, head, inference);
  if (git('status', '--porcelain', '--untracked-files=no')) throw new Error('tracked source differs from selected SHA');
  const directory = path.resolve(env.AESTHETIC_OUTPUT ?? '');
  if (!env.AESTHETIC_OUTPUT || directory === ROOT || directory.startsWith(`${ROOT}${path.sep}`)) throw new Error('evidence directory must be outside checkout');
  mkdirSync(directory, { recursive: true });
  const evidence = {
    schema_version: 1, status: 'not_executed', source: { revision: head, tree: git('rev-parse', 'HEAD^{tree}'), cargo_lock_sha256: hash(read('Cargo.lock')) },
    inference_revision: inference, fixture_manifest_sha256: hash(read(MANIFEST)),
    host: { platform: platform(), arch: arch(), hostname: hostname(), runner: env.RUNNER_NAME ?? null, cpu: cpus()[0]?.model ?? 'unknown' },
    backend: 'rust-vec-f64', device: 'cpu', run_url: `${env.GITHUB_SERVER_URL}/${env.GITHUB_REPOSITORY}/actions/runs/${env.GITHUB_RUN_ID}`,
    fixtures: manifest.fixtures, cases: CASES.map(c => ({ ...c, source_sha256: hash(read(c.source)), test_body_sha256: hash(body(read(c.source).toString(), c.fqn)), status: 'not_executed' })),
  };
  writeFileSync(path.join(directory, 'result.json'), `${JSON.stringify(evidence, null, 2)}\n`);
  return { directory, evidence, manifest };
}

function run(env) {
  const { directory, evidence, manifest } = preflight(env);
  const record = () => writeFileSync(path.join(directory, 'result.json'), `${JSON.stringify(evidence, null, 2)}\n`);
  function command(executable, args, log, timeout) {
    const result = spawnSync(executable, args, { cwd: ROOT, env: { ...env, CARGO_BUILD_JOBS: '1', CARGO_INCREMENTAL: '0', HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1' }, encoding: 'utf8', timeout, maxBuffer: 32 * 1024 * 1024 });
    writeFileSync(path.join(directory, log), `${result.stdout ?? ''}\n${result.stderr ?? ''}`);
    evidence.commands ??= []; evidence.commands.push({ executable, args, log, status: result.status, error: result.error?.message ?? null }); record();
    if (result.error || result.status !== 0) throw new Error(`command failed: ${log}`);
    return result.stdout;
  }
  try {
    evidence.rustc = command('rustc', ['-Vv'], 'rustc.log', 30_000).trim();
    for (const pkg of [...new Set(CASES.map(c => c.package))]) {
      const build = command('cargo', ['test', '--locked', '-p', pkg, '--features', CASES.find(c => c.package === pkg).feature, '--lib', '--no-run', '--message-format=json'], `${pkg}-build.log`, 30 * 60_000);
      const artifacts = build.split('\n').filter(Boolean).map(line => JSON.parse(line)).filter(row => row.reason === 'compiler-artifact' && row.target?.name === pkg.replaceAll('-', '_') && row.profile?.test && row.executable);
      if (artifacts.length !== 1) throw new Error(`exactly one test executable required: ${pkg}`);
      const binary = artifacts[0].executable;
      const cases = CASES.filter(c => c.package === pkg);
      for (const c of cases) evidence.cases.find(row => row.fqn === c.fqn).test_binary_sha256 = hash(readFileSync(binary));
      record();
      discover(command(binary, ['aesthetic', '--ignored', '--list'], `${pkg}-discovery.log`, 30_000), cases.map(c => c.fqn));
      for (const c of cases) {
        const text = command(binary, [c.fqn, '--exact', '--ignored', '--nocapture', '--test-threads=1'], `${c.fixture}-execution.log`, 60_000);
        const result = validateExecution(text, 0, c, manifest.fixtures[c.fixture]);
        Object.assign(evidence.cases.find(row => row.fqn === c.fqn), { status: 'passed', result }); record();
      }
    }
    evidence.status = 'passed'; record();
  } catch (error) {
    evidence.status = 'failed'; evidence.error = error.message; record(); throw error;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const mode = process.argv[2];
    if (mode === '--check') { checkRouting(); console.log('aesthetic CPU routing checked; no model executed'); }
    else if (mode === '--preflight') { preflight(process.env); console.log('manual dispatch admitted; no model executed'); }
    else if (mode === '--run') run(process.env);
    else throw new Error('expected --check, --preflight or --run');
  } catch (error) { console.error(error.message); process.exitCode = 1; }
}
