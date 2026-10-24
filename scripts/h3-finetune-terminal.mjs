#!/usr/bin/env node
// sc-26691: explicit, source-bound terminal execution. No ordinary CI/model execution.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { mkdir, readFile, realpath, rename, stat, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { promisify } from 'node:util';
import { execFile } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const exec = promisify(execFile);
export const SOURCE = { repo: 'WarmBloodAban/Minimax-h3_Singularity', revision: 'a56850c870825d6a8ae501a595c4dbbbc77bbc7e' };
export const ARTIFACTS = [
  'Minimax-h3_Singularity_ref2va_v2_int8.safetensors',
  'Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors',
  'Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors',
  'Minimax-h3_Singularity_ref2va_v1.3_Pruned_w4a8.safetensors',
];
// Mechanism coverage, not a claim that every checkpoint supports every operation.
export const SURFACE = {
  numeric: ['tensor-mapping', 'convrot', 'pruned-interpolation', 'w4a8'],
  capability: ['text', 'image', 'first-last-frame', 'video', 'ordered-multimodal', 'audio-reference', 'hybrid', 'turbo', 'lora', 'lokr'],
  generalization: ['renamed', 'independent-finetune', 'native-single', 'sharded-diffusers', 'dense-f16', 'dense-bf16', 'dense-f32', 'existing-families', 'ltx-add-select-generate', 'compatible-base'],
  lifecycle: ['managed-linked-equivalence', 'restart-replay', 'source-drift', 'interruption-retry', 'removal', 'cancel', 'relink-rescan', 'leases', 'duplicate', 'bounded-loading', 'cold-warm-staged', 'provenance-license'],
};
export function requiredCases() {
  return ['mlx', 'cuda'].flatMap(backend => [
    ...ARTIFACTS.map((artifact, i) => ({ id: `${backend}.artifact.${i}`, backend, group: 'artifact', artifact })),
    ...Object.entries(SURFACE).flatMap(([group, checks]) => checks.map(check => ({ id: `${backend}.${group}.${check}`, backend, group }))),
  ]);
}
const sha40 = v => assert.match(v ?? '', /^[a-f0-9]{40}$/);
const sha64 = v => assert.match(v ?? '', /^[a-f0-9]{64}$/);
const positive = v => assert.ok(Number.isFinite(v) && v > 0, 'positive measured value required');
const json = async p => JSON.parse(await readFile(p, 'utf8'));
// Final delivery links arrive after hardware execution; they are not execution inputs.
const digest = ({ delivery: _delivery, ...inputs }) => createHash('sha256').update(JSON.stringify(inputs)).digest('hex');
export async function hashFile(p) {
  const h = createHash('sha256');
  for await (const chunk of createReadStream(p)) h.update(chunk);
  return h.digest('hex');
}
async function save(p, value) {
  await writeFile(`${p}.tmp`, JSON.stringify(value, null, 2) + '\n', { flag: 'w' });
  await rename(`${p}.tmp`, p);
}
export function validatePlan(p) {
  assert.equal(p.version, 1);
  sha40(p.sceneworks); sha40(p.inference);
  assert.deepEqual(p.source, SOURCE);
  assert.ok(path.isAbsolute(p.output), 'external absolute output required');
  assert.ok(p.responsibleTask === 'sc-26691');
  assert.ok(typeof p.nextAuthorizedAction === 'string' && p.nextAuthorizedAction.length > 10);
  assert.equal(p.cases.length, requiredCases().length, 'complete acceptance inventory required');
  assert.equal(new Set(p.cases.map(c => c.id)).size, p.cases.length);
  for (const expected of requiredCases()) {
    const c = p.cases.find(c => c.id === expected.id);
    for (const [k, v] of Object.entries(expected)) assert.equal(c?.[k], v, `missing/mismatched ${expected.id}.${k}`);
    assert.ok(Number.isInteger(c.timeoutSeconds) && c.timeoutSeconds > 0 && c.timeoutSeconds <= 7200);
    assert.ok(Array.isArray(c.command) && c.command.length > 0 && c.command.every(s => typeof s === 'string'));
    assert.ok(Array.isArray(c.inputs) && c.inputs.length > 0);
    for (const input of c.inputs) { sha64(input.sha256); assert.ok(path.isAbsolute(input.path)); }
    assert.ok(typeof c.evidence === 'string' && c.evidence.length > 0);
    positive(c.loadHostBudgetBytes); positive(c.loadDeviceBudgetBytes);
    if (c.group === 'numeric') {
      sha40(c.referenceRevision); compareNumeric([0], [0], c.tolerance);
      assert.ok(Array.isArray(c.referenceCommand) && c.referenceCommand.length > 0 && c.referenceCommand.every(s => typeof s === 'string'));
      assert.ok(typeof c.referenceOutput === 'string' && c.referenceOutput.length > 0);
    }
    if (c.group === 'capability') { positive(c.minimumControlDelta); assert.ok(['video', 'audio'].includes(c.controlStream)); }
  }
  return p;
}
export async function sourcePair(p, root, inferenceRoot) {
  for (const [dir, revision] of [[root, p.sceneworks], [inferenceRoot, p.inference]]) {
    assert.equal((await exec('git', ['rev-parse', 'HEAD'], { cwd: dir })).stdout.trim(), revision, 'source revision drift');
    assert.equal((await exec('git', ['status', '--porcelain'], { cwd: dir })).stdout.trim(), '', 'source must stay clean');
  }
  const cargo = await readFile(path.join(root, 'crates/sceneworks-worker/Cargo.toml'), 'utf8');
  assert.ok(cargo.includes(`rev = "${p.inference}"`), 'permanent paired inference pin required');
  await exec('git', ['merge-base', '--is-ancestor', p.inference, 'origin/main'], { cwd: inferenceRoot });
  const out = await realpath(p.output);
  for (const dir of [root, inferenceRoot]) {
    const rel = path.relative(await realpath(dir), out);
    assert.ok(rel.startsWith(`..${path.sep}`) || path.isAbsolute(rel), 'outputs must be outside source trees');
  }
}
export async function verifyInputs(c) {
  for (const input of c.inputs) assert.equal(await hashFile(input.path), input.sha256, `content changed: ${input.path}`);
}
export function verifyFrozenArtifact(c, catalog) {
  assert.equal(catalog.sha, SOURCE.revision, 'wrong frozen HF revision');
  const file = catalog.siblings?.find(s => s.rfilename === c.artifact);
  sha64(file?.lfs?.sha256);
  const input = c.inputs.find(i => i.role === 'checkpoint' && i.artifact === c.artifact);
  assert.equal(input?.sha256, file.lfs.sha256, 'checkpoint is not the frozen source artifact');
  return file;
}
async function fetchFrozenCatalog(c) {
  const url = `https://huggingface.co/api/models/${SOURCE.repo}/revision/${SOURCE.revision}?blobs=true`;
  const r = await fetch(url, { signal: AbortSignal.timeout(30000) });
  assert.ok(r.ok, `frozen source metadata HTTP ${r.status}; no substitution permitted`);
  const catalog = await r.json(); verifyFrozenArtifact(c, catalog); return catalog;
}
export function compareNumeric(candidate, reference, tolerance) {
  assert.ok(Array.isArray(candidate) && candidate.length > 0 && candidate.length === reference.length);
  assert.ok(Number.isFinite(tolerance.atol) && tolerance.atol >= 0 && Number.isFinite(tolerance.rtol) && tolerance.rtol >= 0);
  let maxAbs = 0;
  candidate.forEach((v, i) => {
    const r = reference[i];
    assert.ok(Number.isFinite(v) && Number.isFinite(r));
    maxAbs = Math.max(maxAbs, Math.abs(v - r));
    assert.ok(Math.abs(v - r) <= tolerance.atol + tolerance.rtol * Math.abs(r), `numeric mismatch at ${i}`);
  });
  return { elements: candidate.length, maxAbs };
}
export function executedTests(log) {
  const matches = [...log.matchAll(/test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;/g)];
  assert.ok(matches.length > 0, 'missing actual test execution');
  let passed = 0;
  for (const m of matches) { assert.equal(+m[2], 0); assert.equal(+m[3], 0, 'ignored tests are not evidence'); passed += +m[1]; }
  assert.ok(passed > 0, 'zero executed tests');
  return passed;
}
// ffprobe metadata is not decoding. Decode both streams completely; corrupt/missing AV fails.
export async function decodeAV(file, command = exec) {
  positive((await stat(file)).size);
  const { stdout } = await command('ffprobe', ['-v', 'error', '-count_frames', '-show_streams', '-of', 'json', file]);
  const streams = JSON.parse(stdout).streams;
  for (const type of ['video', 'audio']) {
    const stream = streams.find(s => s.codec_type === type);
    assert.ok(stream, `missing decoded ${type}`);
    positive(Number(stream.nb_read_frames)); positive(Number(stream.duration));
  }
  await command('ffmpeg', ['-v', 'error', '-xerror', '-i', file, '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-'], { timeout: 300000, maxBuffer: 1048576 });
  return { sha256: await hashFile(file), streams };
}
export async function decodedDelta(a, b, stream, command = exec) {
  const samples = async file => {
    const args = stream === 'video'
      ? ['-an', '-vf', 'fps=1,scale=16:16', '-pix_fmt', 'rgb24', '-f', 'rawvideo']
      : ['-vn', '-ac', '1', '-ar', '8000', '-f', 'f32le'];
    const { stdout } = await command('ffmpeg', ['-v', 'error', '-xerror', '-i', file, '-t', '8', ...args, 'pipe:1'], { encoding: 'buffer', timeout: 300000, maxBuffer: 1048576 });
    if (stream === 'video') return [...stdout].map(v => v / 255);
    assert.equal(stdout.length % 4, 0);
    return Array.from({ length: stdout.length / 4 }, (_, i) => stdout.readFloatLE(i * 4));
  };
  const x = await samples(a), y = await samples(b);
  assert.ok(x.length > 0 && x.length === y.length, 'control geometry/duration differs');
  let squared = 0;
  x.forEach((v, i) => { assert.ok(Number.isFinite(v) && Number.isFinite(y[i])); squared += (v - y[i]) ** 2; });
  return Math.sqrt(squared / x.length);
}
const contained = (base, name) => {
  const p = path.resolve(base, name), rel = path.relative(base, p);
  assert.ok(rel !== '' && !rel.startsWith('..') && !path.isAbsolute(rel), 'evidence path escapes case directory');
  return p;
};
export function validatePeakMetrics(metrics, c) {
  const hostMethods = ['sampled-process-rss', 'os-rss-high-water'];
  const deviceMethods = ['sampled-process-cuda-used', 'sampled-device-used', 'sampled-native-allocated', 'native-allocated-high-water', 'mlx-allocator-cumulative-high-water'];
  for (const [name, bytesKey] of [['host', 'hostPeakBytes'], ['device', 'devicePeakBytes'], ['loadHost', 'loadHostPeakBytes'], ['loadDevice', 'loadDevicePeakBytes']]) {
    positive(metrics?.[bytesKey]);
    const o = metrics.observations?.[name], device = name.toLowerCase().includes('device');
    assert.ok((device ? deviceMethods : hostMethods).includes(o?.method), 'honest peak observation method required');
    assert.equal(o.sampled, o.method.startsWith('sampled-'));
    if (o.method === 'sampled-process-cuda-used') assert.equal(c.backend, 'cuda');
    if (o.method === 'mlx-allocator-cumulative-high-water') assert.equal(c.backend, 'mlx');
    const start = Date.parse(o.window?.start), end = Date.parse(o.window?.end);
    assert.ok(Number.isFinite(start) && Number.isFinite(end) && end >= start, 'observed peak window required');
    assert.equal(o.window.phase, name.startsWith('load') ? 'loading' : 'case');
    assert.ok((name.startsWith('load') ? ['full-case-load-inclusive', 'native-loading-windows'] : ['full-case']).includes(o.window.coverage), 'source acquisition alone does not cover deferred native loading');
    if (o.window.coverage === 'full-case-load-inclusive') assert.ok(o.bounds?.includes('full-case-load-inclusive-upper-bound'), 'whole-case bound is not an exact loading-phase peak');
    assert.ok(o.attribution?.scope && o.attribution?.meaning, 'peak attribution and exact meaning required');
    if (device) assert.ok(o.attribution.deviceId, 'device attribution required');
    if (o.method === 'sampled-device-used') {
      assert.equal(o.attribution.scope, 'device'); assert.equal(o.attribution.includesOtherProcesses, true);
      assert.ok(o.bounds?.includes('observed-device-global-bound'), 'device-global bound is not model/process allocation evidence');
    } else assert.ok(Number.isSafeInteger(o.attribution.pid) && o.attribution.pid > 0, 'process attribution required');
    assert.ok(Array.isArray(o.blindspots) && o.blindspots.every(v => typeof v === 'string' && v.length > 0));
    assert.ok(Array.isArray(o.samples) && o.samples.length > 0, 'actual peak observations required');
    if (o.sampled) { positive(o.intervalMs); assert.ok(o.samples.length >= 2 && o.blindspots.length > 0, 'sampled peaks must disclose sampling blindspots'); }
    if (o.method === 'mlx-allocator-cumulative-high-water' || o.method === 'os-rss-high-water') {
      assert.equal(o.attribution.scope, 'process-lifetime');
      assert.equal(o.attribution.includesPriorAllocations, true, 'cumulative highwater is not window-local');
    }
    for (const sample of o.samples) {
      const at = Date.parse(sample.at);
      assert.ok(at >= start && at <= end && Number.isFinite(sample.bytes) && sample.bytes >= 0, 'sample outside observation window or invalid bytes');
    }
    assert.equal(o.samples.reduce((peak, sample) => Math.max(peak, sample.bytes), 0), metrics[bytesKey], 'peak differs from retained observations');
  }
  assert.ok(metrics.loadHostPeakBytes <= c.loadHostBudgetBytes && metrics.loadDevicePeakBytes <= c.loadDeviceBudgetBytes, 'loading exceeds declared budgets');
}
export function normalizeLoadEvent(raw) {
  const fields = raw?.fields ?? raw;
  assert.equal(fields?.event, 'h3_import_prepared_load');
  assert.equal(fields.seal_validated, true);
  assert.ok(['Cold', 'Warm'].includes(fields.cache_state));
  const parse = value => typeof value === 'string' ? JSON.parse(value) : value;
  const optionNumber = value => {
    if (value === null || value === 'None') return null;
    const number = typeof value === 'number' ? value : Number(/^Some\(([^)]+)\)$/.exec(value ?? '')?.[1]);
    assert.ok(Number.isFinite(number), 'missing/invalid observed sampling field'); return number;
  };
  assert.ok(fields.backend && fields.backend !== 'cpu', 'CPU fallback cannot prove native H3 acceptance');
  return { kind: 'prepared-native-load-closure', jobId: fields.jobId, providerId: fields.provider, backend: fields.native_backend, deviceLabel: fields.backend, identity: parse(fields.identity), prepared_sources: parse(fields.prepared_sources), seed: fields.seed, operation: fields.operation, steps: optionNumber(fields.steps), schedulerShift: optionNumber(fields.scheduler_shift), cacheState: fields.cache_state };
}
export async function validateEvidence(p, c, dir, log, decode = decodeAV, delta = decodedDelta) {
  const canonical = requiredCases().find(item => item.id === c.id);
  assert.ok(canonical, 'unknown acceptance case');
  for (const [key, value] of Object.entries(canonical)) assert.equal(c[key], value, 'case reclassification cannot weaken acceptance');
  const e = await json(contained(dir, c.evidence));
  assert.equal(e.status, 'executed');
  assert.equal(e.sceneworks, p.sceneworks); assert.equal(e.inference, p.inference);
  assert.equal(e.caseId, c.id); assert.equal(e.backend, c.backend);
  assert.ok(e.hardware?.device && e.hardware?.driver, 'physical hardware identity required');
  const modelExecution = ['artifact', 'capability'].includes(c.group) || /\.(bounded-loading|cold-warm-staged)$/.test(c.id);
  if (modelExecution) {
    // Honest post-acquisition sealed closure, not a per-tensor read census or intended plan.
    assert.ok(e.runtimeIdentity?.checkpointSha256, 'actual loaded checkpoint content identity required');
    assert.ok(c.inputs.some(i => i.role === 'checkpoint' && i.sha256 === e.runtimeIdentity.checkpointSha256), 'wrong/stock checkpoint loaded');
    for (const key of ['topology', 'quantization', 'components', 'adapters', 'operation', 'samplingPolicy', 'planId']) assert.ok(e.runtimeIdentity[key] != null, `missing runtime ${key}`);
    for (const component of e.runtimeIdentity.components) assert.ok(c.inputs.some(i => i.sha256 === component.sha256), 'unverified runtime component');
    sha64(e.runtimeIdentity.semanticDigest);
    const closure = normalizeLoadEvent(e.loadEvent);
    assert.equal(closure.kind, 'prepared-native-load-closure');
    assert.equal(closure.jobId, e.completedJob?.id); assert.ok(closure.jobId);
    assert.equal(e.completedJob.status, 'completed'); assert.equal(closure.backend, c.backend);
    assert.ok(closure.providerId && closure.providerId === e.completedJob.providerId);
    assert.equal(closure.operation, e.runtimeIdentity.operation); assert.equal(closure.seed, e.runtimeIdentity.seed);
    assert.equal(closure.steps, e.runtimeIdentity.samplingPolicy.steps);
    assert.equal(closure.schedulerShift, e.runtimeIdentity.samplingPolicy.schedulerShift);
    assert.equal(closure.identity.planId, e.runtimeIdentity.planId);
    assert.equal(closure.identity.semanticDigest, e.runtimeIdentity.semanticDigest);
    assert.ok(closure.identity.sourceFiles.some(i => i.sha256 === e.runtimeIdentity.checkpointSha256), 'missing sealed checkpoint identity');
    assert.ok(closure.prepared_sources.length > 0, 'missing native prepared closure');
    for (const input of closure.identity.sourceFiles) assert.ok(c.inputs.some(i => i.sha256 === input.sha256), 'native source closure differs from verified inputs');
    for (const input of closure.prepared_sources) {
      // Imports may create native derivatives during this case; don't invent their hashes in
      // the pre-run plan. Verify their actual bytes after acquisition, confined to task state.
      const resolved = await realpath(input.path), relative = path.relative(await realpath(p.output), resolved);
      const declared = c.inputs.some(i => path.resolve(i.path) === path.resolve(input.path) && i.sha256 === input.sha256);
      assert.ok(declared || (relative !== '' && !relative.startsWith('..') && !path.isAbsolute(relative)), 'undeclared prepared source outside task state');
      assert.equal(await hashFile(input.path), input.sha256, 'prepared source content changed');
    }
    validatePeakMetrics(e.metrics, c);
  } else assert.ok(c.inputs.some(i => i.sha256 === e.testInputSha256), 'executed test input identity required');
  const files = [];
  for (const name of e.files ?? []) { const f = contained(dir, name); files.push({ path: name, sha256: await hashFile(f) }); }
  assert.ok(files.length > 0, 'missing supporting evidence');
  let av, numeric;
  if (['artifact', 'capability'].includes(c.group)) {
    assert.equal(e.completedJob.output, e.av, 'successful output not bound to native job');
    av = await decode(contained(dir, e.av));
    if (c.group === 'artifact') {
      assert.deepEqual(e.source, SOURCE); assert.equal(e.artifact, c.artifact);
      const catalogPath = path.join(dir, 'source-catalog.json');
      verifyFrozenArtifact(c, await json(catalogPath));
      files.push({ path: 'source-catalog.json', sha256: await hashFile(catalogPath) });
      assert.equal(e.runtimeIdentity.checkpointSha256, c.inputs.find(i => i.artifact === c.artifact)?.sha256);
    } else {
      // Paired intervention: same checkpoint/seed, changed conditioning/routing/adapter.
      assert.ok(e.control?.intervention, 'causal conditioning control required');
      assert.equal(e.control.checkpointSha256, e.runtimeIdentity.checkpointSha256);
      assert.ok(Number.isSafeInteger(e.runtimeIdentity.seed));
      assert.equal(e.control.seed, e.runtimeIdentity.seed);
      const controlAV = await decode(contained(dir, e.control.av));
      assert.notEqual(controlAV.sha256, av.sha256, 'conditioning/adapters ignored');
      // Compute a scale-sensitive delta from decoded media, not a producer-claimed scalar/hash.
      const measuredDelta = await delta(contained(dir, e.av), contained(dir, e.control.av), c.controlStream);
      assert.ok(measuredDelta >= c.minimumControlDelta, 'conditioning/adapters ignored');
      files.push({ path: e.control.av, sha256: await hashFile(contained(dir, e.control.av)), measuredDelta, stream: c.controlStream });
    }
  } else {
    if (c.group !== 'numeric' && e.apiTranscript) {
      const transcript = await json(contained(dir, e.apiTranscript));
      assert.equal(transcript.caseId, c.id);
      const assertions = transcript.transcript.filter(r => r.assertion);
      assert.ok(assertions.length > 0, 'no API assertions executed');
      for (const a of assertions) assert.deepEqual(a.actual, a.expected);
      assert.ok(log.includes(`H3_API_ASSERTIONS_EXECUTED=${assertions.length}`), 'API driver did not complete');
      files.push({ path: e.apiTranscript, sha256: await hashFile(contained(dir, e.apiTranscript)) });
    } else executedTests(log);
    if (c.group === 'numeric') {
      sha40(e.reference?.revision); sha64(e.reference?.sha256);
      assert.equal(e.reference.revision, c.referenceRevision, 'reference revision drift');
      assert.ok(c.inputs.some(i => i.role === 'reference' && i.sha256 === e.reference.sha256), 'unpinned executable reference');
      assert.equal(e.numeric.reference, c.referenceOutput);
      const candidatePath = contained(dir, e.numeric.candidate), referencePath = contained(dir, e.numeric.reference);
      const referenceRun = await json(path.join(dir, 'reference-execution.json'));
      assert.deepEqual(referenceRun.command, c.referenceCommand);
      assert.equal(referenceRun.revision, c.referenceRevision);
      assert.equal(referenceRun.outputSha256, await hashFile(referencePath), 'executable reference output was replaced');
      numeric = compareNumeric(await json(candidatePath), await json(referencePath), c.tolerance);
      files.push({ path: e.numeric.candidate, sha256: await hashFile(candidatePath) }, { path: e.numeric.reference, sha256: await hashFile(referencePath) });
    }
  }
  return { evidenceSha256: await hashFile(contained(dir, c.evidence)), processLogSha256: createHash('sha256').update(log).digest('hex'), files, av, numeric };
}
// No shell interpolation. The process must implement cooperative timeout/cancel (native GPU
// execution must not be killed mid-command-buffer); Actions is the final job timeout boundary.
export async function runProcess(argv, { cwd, env, logPath, timeoutSeconds }) {
  const output = createWriteStream(logPath, { flags: 'wx' });
  await new Promise((resolve, reject) => { output.once('open', resolve); output.once('error', reject); });
  const child = spawn(argv[0], argv.slice(1), { cwd, env, stdio: ['ignore', 'pipe', 'pipe'], shell: false });
  child.stdout.pipe(output, { end: false }); child.stderr.pipe(output, { end: false });
  let expired = false;
  const timer = setTimeout(() => { expired = true; }, timeoutSeconds * 1000);
  try {
    const code = await new Promise((resolve, reject) => { child.on('error', reject); child.on('close', resolve); });
    assert.ok(!expired, 'case exceeded timeout; failed, never accepted on resume');
    assert.equal(code, 0, 'case process failed');
  } finally { clearTimeout(timer); await new Promise(resolve => output.end(resolve)); }
}
export async function runCase(p, id, root, inferenceRoot) {
  assert.equal(process.env.GITHUB_ACTIONS, 'true', 'native campaign execution is Actions-only');
  assert.equal(process.env.H3_TERMINAL_AUTHORIZED_PAIR, `${p.sceneworks}:${p.inference}`, 'explicit final pair authorization required');
  validatePlan(p); await sourcePair(p, root, inferenceRoot);
  const c = p.cases.find(c => c.id === id); assert.ok(c, 'unknown case');
  if (c.backend === 'mlx') { assert.equal(process.platform, 'darwin'); assert.equal(process.env.RUNNER_NAME, 'nax-macos-2'); }
  else assert.equal(process.platform, 'win32');
  await verifyInputs(c);
  const dir = contained(p.output, c.id); await mkdir(dir, { recursive: true });
  const identity = { planSha256: digest(p), caseId: id, command: c.command, sceneworks: p.sceneworks, inference: p.inference, runId: process.env.GITHUB_RUN_ID, runAttempt: process.env.GITHUB_RUN_ATTEMPT, responsibleTask: p.responsibleTask, timeoutSeconds: c.timeoutSeconds, nextAuthorizedAction: p.nextAuthorizedAction };
  const statePath = path.join(dir, 'execution.json');
  let previous;
  try { previous = await json(statePath); } catch (e) { if (e.code !== 'ENOENT') throw e; }
  if (previous) {
    assert.equal(previous.planSha256, identity.planSha256, 'resume plan drift');
    assert.equal(previous.status, 'passed', 'interrupted/failed case requires a new output directory, not silent overwrite');
    const verified = await validateEvidence(p, c, dir, await readFile(path.join(dir, 'process.log'), 'utf8'));
    assert.deepEqual(verified, previous.verified, 'resume evidence drift');
    return previous;
  }
  await save(statePath, { ...identity, status: 'running', started: new Date().toISOString() });
  try {
    if (c.group === 'artifact') await save(path.join(dir, 'source-catalog.json'), await fetchFrozenCatalog(c));
    const options = { cwd: root, env: { ...process.env, H3_CASE_DIR: dir, H3_CASE_ID: id, H3_INFERENCE_ROOT: inferenceRoot, H3_TIMEOUT_SECONDS: String(c.timeoutSeconds) }, timeoutSeconds: c.timeoutSeconds };
    if (c.group === 'numeric') {
      await runProcess(c.referenceCommand, { ...options, logPath: path.join(dir, 'reference.log') });
      await save(path.join(dir, 'reference-execution.json'), { command: c.referenceCommand, revision: c.referenceRevision, outputSha256: await hashFile(contained(dir, c.referenceOutput)) });
    }
    await runProcess(c.command, { ...options, logPath: path.join(dir, 'process.log') });
    await sourcePair(p, root, inferenceRoot); await verifyInputs(c);
    const verified = await validateEvidence(p, c, dir, await readFile(path.join(dir, 'process.log'), 'utf8'));
    const receipt = { ...identity, status: 'passed', completed: new Date().toISOString(), verified };
    await save(statePath, receipt); return receipt;
  } catch (error) { await save(statePath, { ...identity, status: 'failed', error: error.message }); throw error; }
}
export function validateDelivery(p) {
  const d = p.delivery;
  assert.ok(d, 'missing integrated delivery records');
  const link = v => assert.match(v ?? '', /^https:\/\//);
  assert.ok(Array.isArray(d.requiredCI) && d.requiredCI.length > 0);
  for (const check of d.requiredCI) { assert.ok([p.sceneworks, p.inference].includes(check.headSha), 'CI source mismatch'); assert.equal(check.conclusion, 'success'); link(check.url); }
  assert.equal(d.featureEndReview?.sceneworks, p.sceneworks); assert.equal(d.featureEndReview?.inference, p.inference);
  assert.equal(d.featureEndReview.unresolvedFindings, 0); link(d.featureEndReview.url);
  assert.equal(d.inferenceMain?.sha, p.inference); link(d.inferenceMain.url);
  sha40(d.sceneworksMain?.sha); assert.equal(d.sceneworksMain.sourceCommit, p.sceneworks); assert.equal(d.sceneworksMain.inferencePin, p.inference); link(d.sceneworksMain.url);
  assert.ok(Array.isArray(d.shortcutReadbacks) && d.shortcutReadbacks.some(r => r.id === 26679) && d.shortcutReadbacks.some(r => r.id === 26691));
  for (const r of d.shortcutReadbacks) { assert.ok(r.observedState && r.readAt); link(r.url); }
  assert.ok(Array.isArray(d.limitations) && d.limitations.length === 0, 'unresolved limitations are not a pass');
  // These are recorded readbacks, not an authorization to merge or mutate the tracker.
}
export async function verifyCampaign(p) {
  validatePlan(p);
  for (const c of p.cases) {
    await verifyInputs(c);
    const dir = contained(p.output, c.id), r = await json(path.join(dir, 'execution.json'));
    assert.equal(r.status, 'passed', `not executed: ${c.id}`); assert.equal(r.planSha256, digest(p));
    assert.ok(r.runId && r.runAttempt, 'missing Actions execution identity');
    assert.deepEqual(await validateEvidence(p, c, dir, await readFile(path.join(dir, 'process.log'), 'utf8')), r.verified);
  }
  // Hardware evidence alone is not integrated delivery / tracker completion.
  validateDelivery(p);
  return { status: 'hardware-evidence-complete-delivery-recorded', sceneworks: p.sceneworks, inference: p.inference, cases: p.cases.length };
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [mode, planFile, ...args] = process.argv.slice(2);
  try {
    if (mode === 'list') console.log(JSON.stringify({ source: SOURCE, cases: requiredCases() }, null, 2));
    else {
      const p = validatePlan(await json(planFile));
      if (mode === 'run-case') console.log(JSON.stringify(await runCase(p, args[0], args[1], args[2])));
      else if (mode === 'verify') console.log(JSON.stringify(await verifyCampaign(p)));
      else if (mode === 'check-plan') console.log(`complete inventory: ${p.cases.length} cases; no execution claimed`);
      else throw new Error('usage: h3-finetune-terminal.mjs list | check-plan PLAN | run-case PLAN CASE SW_ROOT INFERENCE_ROOT | verify PLAN');
    }
  } catch (e) { console.error(e.message); process.exitCode = 1; }
}
