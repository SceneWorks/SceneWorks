// Manual-only, fixed worker case. No user-controlled Cargo selectors or remote helper imports.
import assert from 'node:assert/strict';
import { spawn, execFileSync } from 'node:child_process';
import { createReadStream } from 'node:fs';
import * as fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { SECURITY_BACKPORT_MANIFEST_SHA256, verifySecuritySources } from './zip9-source-overrides.mjs';

export const CASE = Object.freeze({
  package: 'sceneworks-worker', target_kind: 'lib',
  source: 'crates/sceneworks-worker/src/imported_nvfp4_gpu_smoke.rs',
  filter: 'imported_nvfp4_worker_gpu_smoke', execution_role: 'direct',
  backend: 'candle-cuda', cargo_features: ['backend-candle'],
  configuration_source: 'imported_nvfp4_worker_gpu_smoke image_generate payload',
  output_contract: ['generated-image', 'completion.json'],
});
export const DEVICE_PROOF = Object.freeze({
  kind: 'source-enforced-postload-receipt',
  audited_app_revision: 'f71f3ab4235494356ab03da2ad99aff5527ce0ff',
  inference_revision: '91ffb4a819aa2990c652db2d881ba0b9b10126ce',
  security_backport_manifest_sha256: SECURITY_BACKPORT_MANIFEST_SHA256,
});
const FILE_LIMITS = Object.freeze({
  'discovery.log': 8 * 1024 * 1024, 'execution.log': 8 * 1024 * 1024,
  'completion.json': 1024 * 1024, 'generated-image': 64 * 1024 * 1024,
  'evidence.json': 2 * 1024 * 1024,
});

export function redact(text) {
  return text.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '')
    .replace(/https?:\/\/[^\s"'<>]+/g, (raw) => {
      try {
        const url = new URL(raw);
        url.username = ''; url.password = ''; url.hash = '';
        if (url.search) url.search = '?REDACTED';
        return url.toString();
      } catch { return '[REDACTED_URL]'; }
    })
    .replace(/((?:token|password|secret|api[_-]?key|access[_-]?key)\s*[:=]\s*)\S+/gi, '$1[REDACTED]');
}

export function discoverFqn(log) {
  const names = [...log.matchAll(/^([A-Za-z_][A-Za-z0-9_:]*): test\r?$/gm)]
    .map((m) => m[1]).filter((name) => name.split('::').at(-1) === CASE.filter);
  assert.equal(names.length, 1, 'fixed ignored case must discover exactly one actual libtest FQN');
  return names[0];
}

export function observeExecution(log) {
  const lines = log.split(/\r?\n/);
  return {
    summaries: lines.flatMap((line) => {
      const match = line.match(/^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;/);
      return match ? [{ status: match[1], passed: Number(match[2]), failed: Number(match[3]), ignored: Number(match[4]) }] : [];
    }),
    body_completed: lines.filter((line) => line.startsWith('MODEL_INFERENCE_BODY_COMPLETED ')).length,
    actual_devices: [...new Set(lines.flatMap((line) => line.match(/^MODEL_INFERENCE_DEVICE \S+ (\S+)$/)?.slice(1) ?? []))],
  };
}

export function validateExecution(log, exitCode, fqn, packet) {
  assert.equal(exitCode, 0, 'real worker test exited unsuccessfully');
  const lines = log.split(/\r?\n/);
  const summaries = lines.map((line) => line.match(/^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;/)).filter(Boolean);
  assert.equal(summaries.length, 1, 'direct case requires exactly one libtest result');
  assert.deepEqual(summaries[0].slice(1).map(Number), [1, 0, 0], 'zero matches or ignored cases are not acceptance');
  const body = lines.filter((line) => line.startsWith('MODEL_INFERENCE_BODY_COMPLETED '));
  assert.deepEqual(body, [`MODEL_INFERENCE_BODY_COMPLETED ${fqn}`], 'real model body must complete exactly once');
  const devices = lines.filter((line) => line.startsWith('MODEL_INFERENCE_DEVICE '));
  assert.deepEqual(devices, [`MODEL_INFERENCE_DEVICE ${fqn} candle-cuda`], 'actual request-scoped device proof is required');
  assert(lines.some((line) => /RESULT status=pass route=candle_checkpoint_plan provider=krea_2_turbo /.test(line)), 'post-generation worker result is missing');
  assert.equal(packet.schema_version, 1);
  assert.equal(packet.test_fqn, fqn);
  assert.equal(packet.status, 'completed');
  assert.deepEqual(packet.actual_devices, ['candle-cuda']);
  for (const [key, value] of Object.entries(DEVICE_PROOF)) assert.equal(packet.device_provenance?.[key], value);
  assert(Array.isArray(packet.device_provenance.anchors) && packet.device_provenance.anchors.length > 0);
  assert.equal(packet.configuration.route, 'candle_checkpoint_plan');
  assert.equal(packet.configuration.provider, 'krea_2_turbo');
  assert.deepEqual(packet.configuration.geometry, [256, 256]);
  assert.equal(packet.configuration.seed, 42);
  assert.equal(packet.configuration.steps, 1);
  assert.equal(packet.output, 'generated-image');
  assert.equal(typeof packet.checkpoint_id, 'string');
  assert(packet.checkpoint_id.length > 0);
  assert.equal(packet.runtime_facts?.schemaVersion, 1);
  assert.equal(packet.runtime_facts.materialization?.status, 'reported');
  assert(packet.runtime_facts.materialization.rows.some((row) => row.codecId === 'nvfp4-v1'
    && row.representation === 'native-packed' && row.tensorCount > 0), 'same-model measured native NVFP4 row is required, not capability alone');
  return { passed: 1, failed: 0, ignored: 0, body_completed: 1, actual_devices: ['candle-cuda'] };
}

export function validateEvidence(evidence) {
  assert.equal(evidence.schema_version, 1);
  assert.equal(evidence.accepted, true);
  assert(/^[a-f0-9]{40}$/.test(evidence.source?.revision));
  assert(/^[a-f0-9]{40}$/.test(evidence.source?.tree));
  assert.equal(evidence.inference_source?.revision, DEVICE_PROOF.inference_revision);
  assert.equal(evidence.inference_source.security_overrides?.kind, 'exact-pinned-leaf-security-backport');
  assert.equal(evidence.inference_source.security_overrides.manifest_sha256, SECURITY_BACKPORT_MANIFEST_SHA256);
  assert(/^[a-f0-9]{64}$/.test(evidence.inference_source.security_overrides.cargo_source_graph_sha256));
  assert.equal(evidence.case?.package, CASE.package);
  assert.equal(evidence.case?.target_kind, 'lib');
  assert.equal(evidence.case?.execution_role, 'direct');
  assert.equal(evidence.case?.backend, CASE.backend);
  assert.equal(evidence.case?.source, CASE.source);
  assert.equal(typeof evidence.case.target, 'string');
  assert(evidence.case.target.length > 0);
  assert.equal(evidence.case.fqn.split('::').at(-1), CASE.filter);
  assert.deepEqual(evidence.case.cargo_features, ['backend-candle']);
  assert(evidence.host?.platform && evidence.host?.machine && evidence.host?.runner);
  assert.deepEqual(evidence.actual_devices, ['candle-cuda']);
  assert.equal(evidence.device_provenance?.kind, DEVICE_PROOF.kind);
  assert.deepEqual(evidence.test_counts, {
    passed: 1, failed: 0, ignored: 0, body_completed: 1, actual_devices: ['candle-cuda'],
  });
  assert.equal(evidence.body_completed, 1);
  assert(/^[a-f0-9]{64}$/.test(evidence.fixture_identity?.sha256));
  assert(evidence.fixture_identity.bytes > 0 && evidence.fixture_identity.checkpoint_id);
  assert.equal(evidence.configuration?.route, 'candle_checkpoint_plan');
  assert.deepEqual(evidence.outputs.map((item) => item.relative_path).sort(), [...CASE.output_contract].sort());
  assert.deepEqual(evidence.logs.map((item) => item.relative_path).sort(), ['discovery.log', 'execution.log']);
  for (const item of [...evidence.outputs, ...evidence.logs]) {
    assert(Object.hasOwn(FILE_LIMITS, item.relative_path) && item.bytes > 0 && item.bytes <= FILE_LIMITS[item.relative_path]);
    assert(/^[a-f0-9]{64}$/.test(item.sha256));
  }
  assert(Number.isFinite(evidence.duration_seconds) && evidence.duration_seconds >= 0);
}

export async function regularOutput(directory, name) {
  assert(Object.hasOwn(FILE_LIMITS, name), 'output is not in the fixed artifact allowlist');
  const file = path.join(directory, name);
  const info = await fs.lstat(file);
  assert(info.isFile() && !info.isSymbolicLink() && info.nlink === 1, `not an owned regular output: ${name}`);
  assert(info.size > 0 && info.size <= FILE_LIMITS[name], `invalid bounded output size: ${name}`);
  const hash = createHash('sha256');
  for await (const data of createReadStream(file)) hash.update(data);
  return { relative_path: name, bytes: info.size, sha256: hash.digest('hex') };
}

export async function stageArtifact(directory, evidence) {
  const staging = path.join(directory, 'artifact');
  await fs.mkdir(staging, { recursive: false });
  const names = ['discovery.log', 'execution.log', ...((evidence.accepted || evidence.body_proof_accepted) ? CASE.output_contract : [])];
  evidence.artifact_files = [];
  for (const name of names) {
    try {
      const verified = await regularOutput(directory, name);
      const bytes = await fs.readFile(path.join(directory, name));
      assert.equal(bytes.length, verified.bytes);
      assert.equal(createHash('sha256').update(bytes).digest('hex'), verified.sha256);
      await fs.writeFile(path.join(staging, name), bytes, { flag: 'wx', mode: 0o600 });
      evidence.artifact_files.push(verified);
    } catch (error) {
      // Failed runs may have no logs yet. Never upload an unsafe original path,
      // even on failure: only verified byte copies enter the artifact directory.
      if (evidence.accepted || evidence.body_proof_accepted) {
        evidence.accepted = false;
        evidence.body_proof_accepted = false;
        evidence.error = redact(String(error));
        process.exitCode = 1;
      }
    }
  }
  const manifest = `${JSON.stringify(evidence, null, 2)}\n`;
  assert(Buffer.byteLength(manifest) <= FILE_LIMITS['evidence.json']);
  await fs.writeFile(path.join(directory, 'evidence.json'), manifest, { flag: 'wx' });
  await fs.writeFile(path.join(staging, 'evidence.json'), manifest, { flag: 'wx', mode: 0o600 });
  return staging;
}

async function runCargo(args, directory, filename) {
  const cargo = process.platform === 'win32' ? 'cargo.exe' : 'cargo';
  const started = performance.now();
  const chunks = [];
  let bytes = 0;
  let truncated = false;
  const carry = { stdout: '', stderr: '' };
  function consume(channel, data, final = false) {
    carry[channel] += data;
    const lines = carry[channel].split(/\r?\n/);
    carry[channel] = final ? '' : lines.pop();
    for (const line of lines) {
      const safe = `${redact(line)}\n`;
      const length = Buffer.byteLength(safe);
      if (bytes + length <= FILE_LIMITS[filename]) {
        chunks.push(safe); bytes += length; process.stdout.write(safe);
      } else truncated = true;
    }
    if (carry[channel].length > 65536) { carry[channel] = ''; truncated = true; }
  }
  const child = spawn(cargo, args, { stdio: ['ignore', 'pipe', 'pipe'], shell: false });
  child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');
  child.stdout.on('data', (data) => consume('stdout', data));
  child.stderr.on('data', (data) => consume('stderr', data));
  const code = await new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('close', (status) => resolve(status));
  });
  consume('stdout', '\n', true);
  consume('stderr', '\n', true);
  const log = chunks.join('');
  await fs.writeFile(path.join(directory, filename), log, { flag: 'wx' });
  // Never kill a running GPU process to enforce log bounds. Refuse evidence only
  // after the child has actually completed, rather than truncate into a false pass.
  assert(!truncated, 'bounded raw log was truncated; this is not complete proof');
  return { code, log, duration_seconds: (performance.now() - started) / 1000 };
}

function git(args) {
  return execFileSync('git', args, { encoding: 'utf8', maxBuffer: 1024 * 1024 }).trim();
}

async function checkpointIdentity(file) {
  const actual = await fs.realpath(file);
  const before = await fs.stat(actual);
  assert(before.isFile() && before.size > 0, 'required imported checkpoint is missing');
  const hash = createHash('sha256');
  for await (const bytes of createReadStream(actual)) hash.update(bytes);
  const after = await fs.stat(actual);
  assert.equal(after.size, before.size); assert.equal(after.mtimeMs, before.mtimeMs);
  return { path: file, resolved_path: actual, bytes: before.size, sha256: hash.digest('hex') };
}

export async function main() {
  const started = performance.now();
  const directory = process.env.MODEL_PROOF_OUTPUT_DIR;
  assert(directory && path.isAbsolute(directory), 'MODEL_PROOF_OUTPUT_DIR must be an absolute fresh owned directory');
  await fs.mkdir(directory, { recursive: false });
  const evidence = {
    schema_version: 1, accepted: false, case: CASE,
    host: { platform: process.platform, machine: os.arch(), runner: process.env.RUNNER_NAME ?? os.hostname() },
    actual_devices: [], configuration: {}, outputs: [], logs: [],
  };
  try {
    evidence.source = { revision: git(['rev-parse', 'HEAD']), tree: git(['rev-parse', 'HEAD^{tree}']) };
    assert(/^[a-f0-9]{40}$/.test(evidence.source.revision) && /^[a-f0-9]{40}$/.test(evidence.source.tree));
    assert.equal(git(['status', '--porcelain', '--untracked-files=no']), '', 'proof must identify the actual clean source tree');
    const lock = await fs.readFile('Cargo.lock', 'utf8');
    const pins = [...lock.matchAll(/source = "git\+https:\/\/github\.com\/SceneWorks\/inference\?rev=([a-f0-9]{40})#([a-f0-9]{40})"/g)];
    assert(pins.length > 0 && pins.every((pin) => pin[1] === pin[2]));
    const revisions = new Set(pins.map((pin) => pin[2]));
    assert.equal(revisions.size, 1, 'all locked inference packages must agree');
    evidence.inference_source = { revision: [...revisions][0], lock_sha256: createHash('sha256').update(lock).digest('hex') };
    assert.equal(evidence.inference_source.revision, DEVICE_PROOF.inference_revision,
      'actual-device provenance must be revalidated for a changed inference producer, before running the model');
    const checkpoint = process.env.SCENEWORKS_IMPORTED_NVFP4_CHECKPOINT?.trim();
    assert(checkpoint, 'SCENEWORKS_IMPORTED_NVFP4_CHECKPOINT is required');
    evidence.fixture_identity = await checkpointIdentity(checkpoint);
    const cargo = process.platform === 'win32' ? 'cargo.exe' : 'cargo';
    const metadata = JSON.parse(execFileSync(cargo, ['metadata', '--locked', '--offline', '--features', 'sceneworks-worker/backend-candle', '--format-version', '1'], { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 }));
    evidence.inference_source.security_overrides = await verifySecuritySources(metadata);
    const worker = metadata.packages.find((pkg) => pkg.name === CASE.package);
    const libraries = worker?.targets.filter((target) => target.kind.includes('lib')) ?? [];
    assert.equal(libraries.length, 1, 'fixed worker case must belong to an existing library test target');
    const common = ['test', '--locked', '-p', CASE.package, '--features', 'backend-candle', '--release', '--lib'];
    const discovery = await runCargo([...common, CASE.filter, '--', '--ignored', '--list'], directory, 'discovery.log');
    assert.equal(discovery.code, 0, 'ignored case discovery failed');
    const fqn = discoverFqn(discovery.log);
    evidence.case = { ...CASE, target: libraries[0].name, fqn };
    const execution = await runCargo([...common, fqn, '--', '--exact', '--ignored', '--nocapture', '--test-threads=1'], directory, 'execution.log');
    evidence.execution_duration_seconds = execution.duration_seconds;
    evidence.test_counts = observeExecution(execution.log);
    const completionFile = await regularOutput(directory, 'completion.json');
    const packet = JSON.parse(await fs.readFile(path.join(directory, 'completion.json'), 'utf8'));
    evidence.test_counts = validateExecution(execution.log, execution.code, fqn, packet);
    const afterMetadata = JSON.parse(execFileSync(cargo, ['metadata', '--locked', '--offline', '--features', 'sceneworks-worker/backend-candle', '--format-version', '1'], { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 }));
    assert.deepEqual(await verifySecuritySources(afterMetadata), evidence.inference_source.security_overrides,
      'effective security source graph changed during the model proof');
    assert.equal(git(['rev-parse', 'HEAD']), evidence.source.revision);
    assert.equal(git(['status', '--porcelain', '--untracked-files=no']), '', 'proof source changed during execution');
    evidence.actual_devices = evidence.test_counts.actual_devices;
    evidence.body_completed = evidence.test_counts.body_completed;
    evidence.configuration = packet.configuration;
    evidence.device_provenance = packet.device_provenance;
    assert.equal(path.resolve(packet.checkpoint_path).toLowerCase(), path.resolve(checkpoint).toLowerCase());
    const afterCheckpoint = await checkpointIdentity(checkpoint);
    assert.equal(afterCheckpoint.resolved_path, evidence.fixture_identity.resolved_path);
    assert.equal(afterCheckpoint.sha256, evidence.fixture_identity.sha256, 'the checkpoint changed during the proof');
    assert.equal(afterCheckpoint.bytes, evidence.fixture_identity.bytes);
    assert.equal(packet.runtime_facts.sourceBinding?.sizeBytes, afterCheckpoint.bytes);
    assert.equal(packet.runtime_facts.sourceBinding?.stableToken, `${path.basename(afterCheckpoint.resolved_path)}@${afterCheckpoint.bytes}`);
    evidence.fixture_identity.checkpoint_id = packet.checkpoint_id;
    evidence.inference_source.runtime_facts = packet.runtime_facts;
    evidence.outputs = [completionFile, await regularOutput(directory, 'generated-image')];
    evidence.logs = [await regularOutput(directory, 'discovery.log'), await regularOutput(directory, 'execution.log')];
    const names = await fs.readdir(directory);
    assert(names.every((name) => Object.hasOwn(FILE_LIMITS, name)), 'unexpected output is not uploadable');
    evidence.accepted = true;
    evidence.duration_seconds = (performance.now() - started) / 1000;
    validateEvidence(evidence);
    // A completed body is not whole-run acceptance until the external UUID/Job
    // controller has drained the child tree and verified the post-child idle state.
    evidence.body_proof_accepted = true;
    evidence.resource_verification_pending = true;
    evidence.accepted = false;
  } catch (error) {
    evidence.accepted = false;
    evidence.body_proof_accepted = false;
    evidence.error = redact(String(error?.stack ?? error));
    process.exitCode = 1;
  } finally {
    evidence.duration_seconds = (performance.now() - started) / 1000;
    await stageArtifact(directory, evidence);
    if (process.env.GITHUB_STEP_SUMMARY) {
      await fs.appendFile(process.env.GITHUB_STEP_SUMMARY,
        `## Imported NVFP4 model validation\n\nAccepted: **${evidence.accepted}**\n\nSource: \`${evidence.source?.revision ?? 'unverified'}\`\n\nDuration: ${evidence.duration_seconds.toFixed(3)} seconds\n\nRaw logs, identities and generated output are in the fixed proof artifact.\n`);
    }
    console.log(`Imported NVFP4 model proof accepted=${evidence.accepted}; evidence=${path.join(directory, 'evidence.json')}`);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) await main();
