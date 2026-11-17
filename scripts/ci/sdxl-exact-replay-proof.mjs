// SW-7148. Manual-only production API/worker witness; importing this module never runs a model.
import assert from 'node:assert/strict';
import { createHash, createHmac, randomBytes } from 'node:crypto';
import { createServer } from 'node:http';
import { spawn, execFileSync } from 'node:child_process';
import { readFile, writeFile, mkdir, realpath, stat, open, rm } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { fileSha256, sha256 } from '../lib/file-sha256.mjs';
import { inferenceRevision } from '../lib/inference-provenance.mjs';
import { failureDiagnostic, readDiagnosticJson, safeErrorClass, safePostRenderCheck } from './sdxl-exact-replay-diagnostic.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const HEX = /^[a-f0-9]{64}$/;
const MEMBER_SHA = /^sha256:[a-f0-9]{64}$/;
const SHA = /^[a-f0-9]{40}$/;
export const SCOPE = 'same_device_pinned_runtime';
export const REQUEST = Object.freeze({ model: 'sdxl', mode: 'text_to_image',
  prompt: 'a photorealistic red apple on a wooden table, studio lighting',
  negativePrompt: '', width: 1024, height: 1024, count: 1, seed: 16402, loras: [],
  advanced: { mlxQuantize: 4, sampler: 'ddim', scheduler: 'normal', steps: 20, guidanceScale: 7.5, enhancePrompt: false } });
export function canonical(value) {
  return JSON.stringify(value, (_key, entry) => entry && typeof entry === 'object' && !Array.isArray(entry)
    ? Object.fromEntries(Object.entries(entry).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) : entry);
}
export const digest = (value) => createHash('sha256').update(canonical(value)).digest('hex');
const same = (a, b, message) => assert.equal(digest(a), digest(b), message);
const fact = (value) => { assert.equal(value?.state, 'resolved', 'missing completed fact'); return value.value; };

// Source members are private. Only this digest enters the public evidence. The separate accepted
// payload digest retains request binding; authoredOptionsSha256 includes the task-owned project ID.
export function sourceIdentity(route) {
  assert.equal(route?.backend, 'candle', 'backend fallback');
  assert.equal(route.models?.length, 1, 'unexpected model/adapter slots');
  const model = route.models[0];
  assert.equal(model.modelId, 'sdxl', 'model substitution');
  assert.equal(model.backend, 'candle', 'model backend substitution');
  assert.equal(model.slot, 'modelManifestEntry', 'model slot substitution');
  same(route.expectedProvider, { modelSlot: 'modelManifestEntry', modelId: 'sdxl', operation: 'image_generate', providerId: 'sdxl' }, 'provider substitution');
  assert(model.members?.length > 0, 'source members missing');
  for (const member of model.members) {
    assert(['primary', 'required_component', 'co_requisite'].includes(member.role), 'adapter/optional source outside witness');
    assert(member.files?.length > 0, 'member bytes missing');
    for (const file of member.files) assert(MEMBER_SHA.test(file.sha256) && Number.isSafeInteger(file.sizeBytes) && file.sizeBytes > 0, 'invalid member digest');
  }
  const { authoredOptionsSha256: _request, ...source } = model;
  return { backend: route.backend, platform: route.platform, expectedProvider: route.expectedProvider, models: [source] };
}
export function validateDispatch(input) {
  same(Object.keys(input).sort(), ['deviceUuid', 'ownerGo', 'phase', 'planPath', 'proof'], 'unknown dispatch fields');
  assert.equal(input.proof, 'sdxl-exact-replay', 'unknown proof');
  assert.equal(input.ownerGo, true, 'owner go window not confirmed');
  assert(['qualify', 'replay'].includes(input.phase), 'unknown phase');
  assert(typeof input.planPath === 'string' && path.win32.isAbsolute(input.planPath), 'absolute private Windows plan required');
  assert(/^GPU-[a-f0-9-]{36}$/i.test(input.deviceUuid), 'physical UUID missing');
}
export function parseDispatch(raw) {
  assert(typeof raw === 'string' && raw.startsWith('{"proof":"sdxl-exact-replay",'), 'canonical SDXL dispatch prefix required');
  const input = JSON.parse(raw);
  validateDispatch(input);
  return input;
}
function dispatchInput() {
  const input = parseDispatch(process.env.SDXL_REPLAY_DISPATCH);
  assert.equal(input.planPath, process.env.SDXL_REPLAY_PLAN, 'dispatch plan changed');
  assert.equal(input.phase, process.env.SDXL_REPLAY_PHASE, 'dispatch phase changed');
  assert.equal(input.deviceUuid, process.env.CUDA_VISIBLE_DEVICES, 'dispatch UUID changed');
}
export function validatePlan(plan, sourceSha, pin, phase, now = Date.now()) {
  assert.equal(plan.version, 1, 'unsupported proof plan');
  assert(['qualify', 'replay'].includes(phase), 'unknown phase');
  assert(SHA.test(sourceSha) && SHA.test(pin), 'unfrozen source');
  assert.equal(plan.sourceSha, sourceSha, 'app source changed');
  assert.equal(plan.inferenceRevision, pin, 'inference pin changed');
  assert(HEX.test(plan.sourceIdentitySha256) && HEX.test(plan.scheduleSha256), 'source/schedule preflight missing');
  assert.equal(plan.requestSha256, digest(REQUEST), 'authored request changed');
  assert(/^GPU-[a-f0-9-]{36}$/i.test(plan.deviceUuid), 'physical device UUID missing');
  assert.equal(plan.physicalIndex, 1, 'only approved physical index one');
  const starts = Date.parse(plan.goStartsAt), expires = Date.parse(plan.goExpiresAt);
  assert(Number.isFinite(starts) && Number.isFinite(expires) && starts <= now && now < expires
    && expires - starts <= 120 * 60 * 1000, 'owner go window unavailable');
  assert.equal(plan.ownerGo, true, 'owner go window not confirmed');
  assert(Array.isArray(plan.runtimeFiles) && plan.runtimeFiles.length > 0 && plan.runtimeFiles.every((file) => HEX.test(file.sha256)), 'pinned runtime library hashes missing');
  assert(Array.isArray(plan.files) && plan.files.length > 0, 'selected member map missing');
  if (phase === 'replay') {
    assert(HEX.test(plan.qualificationSha256), 'reviewed qualification missing');
    assert.equal(plan.declaration?.scope, SCOPE, 'reviewed declaration scope missing');
    assert.equal(plan.declaration?.candidateInferenceRevision, pin, 'declaration candidate changed');
    assert.equal(plan.declaration?.candidateSourceSha, sourceSha, 'declaration app candidate changed');
    assert(/^https:\/\/github\.com\/SceneWorks\/inference\/pull\/\d+(?:#.*)?$/.test(plan.declaration?.reviewUrl ?? ''), 'reviewed provider declaration missing');
  }
}
export async function verifyMembers(route, plan) {
  assert.equal(digest(sourceIdentity(route)), plan.sourceIdentitySha256, 'selected source changed');
  const root = await realpath(plan.hub);
  const selected = route.models.flatMap((model) => model.members.flatMap((member, memberIndex) => member.files.map((file) => ({ ...file, slot: model.slot, memberIndex }))));
  assert.equal(plan.files.length, selected.length, 'member map cardinality changed');
  const used = new Set();
  for (const file of selected) {
    const matches = plan.files.filter((entry) => entry.slot === file.slot && entry.memberIndex === file.memberIndex && entry.relativePath === file.relativePath);
    assert.equal(matches.length, 1, 'member map missing or duplicated');
    const actual = await realpath(matches[0].path);
    const relative = path.relative(root, actual);
    assert(relative && !relative.startsWith('..') && !path.isAbsolute(relative), 'member escapes provisioned cache');
    assert(!used.has(actual), 'duplicate physical member'); used.add(actual);
    const info = await stat(actual);
    assert(info.isFile() && info.size === file.sizeBytes, 'selected member size changed');
    assert.equal(`sha256:${await fileSha256(actual)}`, file.sha256, 'selected member bytes changed');
  }
}

export function completedWitness({ job, receipt, route, png, workerId, device, pin, scheduleSha256, payloadSha256 }) {
  assert.equal(job.status, 'completed', 'job did not complete');
  assert.equal(job.workerId, workerId, 'worker changed');
  assert.equal(job.assignedGpu, device, 'device changed');
  assert.equal(job.result?.adapter, 'candle_sdxl', 'provider fallback');
  assert.equal(receipt.version, 1, 'receipt version changed');
  assert.equal(receipt.jobId, job.id, 'receipt belongs to another job');
  assert.equal(receipt.workerId, workerId, 'private worker receipt mismatch');
  assert.equal(receipt.acceptedBackend, 'candle', 'accepted backend changed');
  assert.equal(receipt.acceptedDevice, device, 'accepted device changed');
  assert.equal(receipt.inferenceRevision, pin, 'compiled runtime changed');
  same(receipt.acceptedSourceRoute, route, 'accepted source route changed');
  assert(HEX.test(receipt.attemptId) && HEX.test(receipt.acceptedBindingSha256), 'private claim proof missing');
  assert.equal(receipt.outputs?.length, 1, 'output cardinality changed');
  const output = receipt.outputs[0];
  assert.equal(output.outputIndex, 0, 'output index changed');
  assert.equal(output.artifactIds?.length, 1, 'artifact cardinality changed');
  assert.equal(output.artifactSha256?.length, 1, 'artifact digest missing');
  assert.equal(output.stages?.length ?? 0, 0, 'composite output not qualified');
  assert.equal(receipt.stages?.length ?? 0, 0, 'composite receipt not qualified');
  assert.equal(job.result.assets?.length, 1, 'persisted output missing');
  assert.equal(job.result.assets[0].id, output.artifactIds[0], 'artifact association changed');
  assert(png.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10])), 'final output is not PNG');
  const pngSha256 = sha256(png);
  assert.equal(output.artifactSha256[0], pngSha256, 'final PNG differs from trusted receipt');
  const execution = output.execution ?? receipt.execution;
  assert.equal(execution?.state, 'resolved', 'execution unavailable');
  const identity = execution.identity;
  assert.equal(identity?.version, 1, 'execution identity version changed');
  assert(['non_deterministic', 'deterministic'].includes(identity.deterministic_replay?.kind), 'replay support missing');
  assert.equal(identity.provider_id, 'sdxl', 'executed provider changed');
  assert.equal(identity.backend, 'candle', 'executed backend changed');
  assert.equal(identity.runtime_id, 'candle-execution-v1', 'runtime identity changed');
  assert.equal(identity.output_units, 1, 'provider output count changed');
  same(fact(identity.sampling), ['ddim'], 'sampler fallback');
  const schedule = fact(identity.schedule);
  assert.equal(schedule.length, 1, 'schedule count changed');
  assert.equal(schedule[0].sampler, 'ddim', 'schedule sampler fallback');
  assert.equal(schedule[0].schedule?.count, 21, 'provider steps changed');
  assert.equal(schedule[0].schedule?.element, 'f32_le_bits', 'schedule precision changed');
  assert.equal(digest(schedule), scheduleSha256, 'effective provider schedule changed');
  for (const key of ['estimator', 'geometry', 'quality_policy', 'output_policy']) fact(identity[key]);
  same(fact(identity.geometry), [{ width: 1024, height: 1024, channels: 3, format: 'rgb8' }], 'output geometry changed');
  assert(fact(identity.quality_policy).compute?.some((entry) => /^Cuda/.test(entry.device)), 'no executed CUDA compute');
  const operations = fact(identity.quality_policy).operations;
  const representations = ['sdxl/clip_l/representation', 'sdxl/clip_bigg/representation', 'sdxl/unet/representation']
    .map((key) => operations?.[key]?.length === 1 ? operations[key][0]?.value : undefined);
  same(representations, ['packed_or_imported', 'packed_or_imported', 'packed'], 'q4 representation changed');
  assert.equal(identity.randomness?.kind, 'draws', 'draw identity missing');
  assert.equal(identity.randomness.scheme, 'candle-ordered-provider-streams-v1:rand-0.9-StdRng-rand_distr-StandardNormal-f32', 'RNG scheme changed');
  const draws = identity.randomness.draws;
  assert(draws?.some((draw) => draw.output_index === 0 && draw.purpose === 'sdxl/pipeline' && draw.seed === REQUEST.seed)
    && draws.every((draw) => draw.output_index === 0 && (draw.seed === REQUEST.seed || (draw.seed == null && HEX.test(draw.noise_sha256)))),
  'seed/draw binding changed');
  // Only hashes and selected nonprivate identities are published, never source routes/results.
  return { jobId: job.id, attemptId: receipt.attemptId, executionSha256: digest(execution),
    executionWithoutSupportSha256: digest({ ...identity, deterministic_replay: undefined }),
    support: { kind: identity.deterministic_replay?.kind, ...(identity.deterministic_replay?.scope ? { scope: identity.deterministic_replay.scope } : {}) }, pngSha256, payloadSha256,
    acceptedBindingSha256: receipt.acceptedBindingSha256, sourceIdentitySha256: digest(sourceIdentity(route)),
    scheduleSha256, qualitySha256: digest(identity.quality_policy), representations, providerId: identity.provider_id, runtimeId: identity.runtime_id };
}
export function verifyQueuedPayload(payload) {
  for (const [key, value] of Object.entries(REQUEST)) {
    if (key === 'advanced') {
      for (const [option, expected] of Object.entries(value)) same(payload.advanced?.[option], expected, 'accepted option changed');
    } else same(payload[key], value, 'accepted request changed');
  }
}
export function compareFresh(first, second) {
  assert.notEqual(first.jobId, second.jobId, 'not two fresh jobs');
  assert.notEqual(first.attemptId, second.attemptId, 'reused completed attempt');
  for (const field of ['executionSha256', 'pngSha256', 'payloadSha256', 'acceptedBindingSha256', 'sourceIdentitySha256', 'scheduleSha256', 'qualitySha256'])
    assert.equal(first[field], second[field], `fresh ${field} differs`);
}
export function verifyReplayCandidate(candidate, qualification) {
  same(candidate.support, { kind: 'deterministic', scope: SCOPE }, 'provider has no qualified deterministic declaration');
  for (const key of ['executionWithoutSupportSha256', 'pngSha256', 'sourceIdentitySha256', 'scheduleSha256', 'qualitySha256'])
    assert.equal(candidate[key], qualification[key], 'reviewed provider behavior changed');
}
export function requireReplayQualification(plan, qualification) {
  assert.equal(digest(qualification), plan.qualificationSha256, 'qualification evidence changed');
  assert.equal(qualification.accepted, true, 'qualification was not physically completed');
  assert.equal(qualification.phase, 'qualify', 'not qualification evidence');
  assert.equal(qualification.deviceUuid, plan.deviceUuid, 'qualification physical device changed');
  assert.equal(qualification.requestSha256, plan.requestSha256, 'qualification request changed');
  assert.equal(qualification.sourceIdentitySha256, plan.sourceIdentitySha256, 'qualification source changed');
  assert.equal(qualification.scheduleSha256, plan.scheduleSha256, 'qualification schedule changed');
  assert(HEX.test(qualification.loadedRuntimeLibrariesSha256), 'qualified loaded runtime libraries missing');
  assert.equal(qualification.jobs?.length, 2, 'qualification cardinality invalid');
  compareFresh(qualification.jobs[0], qualification.jobs[1]);
}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const git = (...args) => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' }).trim();
async function listen(server) { await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve)); return server.address().port; }
async function runProof() {
  dispatchInput();
  assert.equal(process.platform, 'win32', 'Windows proof only');
  assert.equal(process.env.GITHUB_EVENT_NAME, 'workflow_dispatch', 'manual dispatch only');
  assert(process.env.SW_CI_CUDA_EXECUTION_ACTIVE, 'physical UUID guard missing');
  assert.equal(git('status', '--porcelain'), '', 'dirty source cannot qualify');
  const plan = JSON.parse(await readFile(process.env.SDXL_REPLAY_PLAN, 'utf8'));
  const phase = process.env.SDXL_REPLAY_PHASE;
  const sourceSha = git('rev-parse', 'HEAD'), pin = await inferenceRevision(ROOT);
  validatePlan(plan, sourceSha, pin, phase);
  assert.equal(process.env.CUDA_VISIBLE_DEVICES, plan.deviceUuid, 'guard visible UUID differs');
  const qualification = phase === 'replay' ? JSON.parse(await readFile(plan.qualificationFile, 'utf8')) : null;
  if (qualification) requireReplayQualification(plan, qualification);
  const output = process.env.SDXL_REPLAY_OUTPUT;
  assert(path.isAbsolute(output), 'absolute owned output required');
  await mkdir(output, { recursive: false });
  const state = path.join(output, 'private-state'); await mkdir(state);
  const evidence = { version: 1, accepted: false, bodyAccepted: false, phase, sourceSha, inferenceRevision: pin,
    deviceUuid: plan.deviceUuid, physicalIndex: plan.physicalIndex, requestSha256: digest(REQUEST), sourceIdentitySha256: plan.sourceIdentitySha256,
    scheduleSha256: plan.scheduleSha256, jobs: [] };
  const binary = path.resolve(process.env.SDXL_REPLAY_BINARY);
  evidence.binarySha256 = await fileSha256(binary);
  const runtimeFiles = async () => {
    const records = [];
    for (const file of plan.runtimeFiles) { assert.equal(await fileSha256(file.path), file.sha256, 'pinned runtime library changed'); records.push({ name: path.basename(file.path).toLowerCase(), sha256: file.sha256 }); }
    return records.sort((a, b) => a.name.localeCompare(b.name));
  };
  const libraries = await runtimeFiles();
  assert.equal(new Set(libraries.map((file) => file.name)).size, libraries.length, 'duplicate runtime libraries');
  evidence.runtimeLibrariesSha256 = digest(libraries);
  evidence.osIdentitySha256 = digest({ platform: process.platform, arch: process.arch, release: os.release(), version: os.version() });
  const runtime = () => execFileSync('nvidia-smi', ['-i', plan.deviceUuid, '--query-gpu=uuid,driver_version,name,pci.bus_id', '--format=csv,noheader'], { encoding: 'utf8' }).trim();
  // Production Candle loads CUDA ordinal zero. The registration's numeric gpuId is an NVML
  // label, not a CUDA ordinal. Verify both map to the exact UUID locked by the existing guard.
  const metadata = JSON.parse(execFileSync('python', ['-I', path.join(ROOT, 'scripts/ci/windows_cuda_metadata.py')], { encoding: 'utf8' }));
  const physicalIndex = Number(execFileSync('nvidia-smi', ['-i', plan.deviceUuid, '--query-gpu=index', '--format=csv,noheader'], { encoding: 'utf8' }).trim());
  verifyDeviceSelection(metadata, physicalIndex, plan);
  const baselineRuntime = runtime(); assert(baselineRuntime.startsWith(plan.deviceUuid), 'physical UUID unavailable');
  evidence.driverDeviceSha256 = sha256(baselineRuntime);
  if (qualification) for (const key of ['driverDeviceSha256', 'runtimeLibrariesSha256', 'osIdentitySha256']) assert.equal(evidence[key], qualification[key], 'qualified runtime/device changed');
  const workerId = `sdxl-exact-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}`;
  const master = randomBytes(32).toString('hex');
  const workerKey = createHmac('sha256', Buffer.from(master, 'utf8')).update(workerId).digest('hex');
  const env = Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith('SCENEWORKS_') && !/TOKEN|SECRET|PASSWORD|CREDENTIAL|API_KEY/.test(key)));
  Object.assign(env, { SCENEWORKS_DATA_DIR: path.join(state, 'data'), SCENEWORKS_CONFIG_DIR: path.join(state, 'config'),
    SCENEWORKS_JOBS_DB_PATH: path.join(state, 'jobs.db'), SCENEWORKS_CREDENTIALS_DIR: path.join(state, 'credentials'),
    SCENEWORKS_WORKER_PROOF_MASTER: master, SCENEWORKS_WORKER_CHILD: '1', SCENEWORKS_PARENT_PID: String(process.pid),
    SCENEWORKS_TRUST_LOOPBACK: '1', SCENEWORKS_API_HOST: '127.0.0.1', SCENEWORKS_WORKER_ID: workerId,
    SCENEWORKS_GPU_ID: String(plan.physicalIndex), SCENEWORKS_RUST_WORKER_GPU_ID: String(plan.physicalIndex),
    SCENEWORKS_BACKEND_CANDLE_ENABLED: 'true', SCENEWORKS_BACKEND_MLX_ENABLED: 'false',
    SCENEWORKS_POLL_SECONDS: '1', SCENEWORKS_HEARTBEAT_SECONDS: '1', SCENEWORKS_LOG_FORMAT: 'json',
    HF_HUB_CACHE: plan.hub, HUGGINGFACE_HUB_CACHE: plan.hub, HF_HOME: path.dirname(plan.hub), HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1' });
  const children = [], handles = []; let db, failure, claimJob = null, claimRoute = null;
  const reservation = createServer(); const port = await listen(reservation); await new Promise((resolve) => reservation.close(resolve));
  const base = `http://127.0.0.1:${port}`;
  const api = async (method, route, body) => {
    const response = await fetch(base + route, { method, ...(body ? { headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) } : {}), signal: AbortSignal.timeout(10000) });
    assert(response.ok, `HTTP ${method} refusal ${response.status}`); return response.json();
  };
  const alive = () => { if (failure) throw failure; assert(children.every((child) => child.exitCode === null && child.signalCode === null), 'owned runtime exited'); validatePlan(plan, sourceSha, pin, phase); };
  const wait = async (probe, seconds) => { const end = Date.now() + seconds * 1000; while (Date.now() < end) { alive(); const found = await probe(); if (found) return found; await delay(500); } throw new Error('proof deadline exceeded'); };
  const launch = async (name, overrides) => {
    const log = await open(path.join(state, `${name}.log`), 'wx'); handles.push(log);
    // WORKER_CHILD treats stdin EOF as shutdown on Windows; keep the child pipe open.
    const child = spawn(binary, [], { cwd: ROOT, env: { ...env, ...overrides }, stdio: ['pipe', log.fd, log.fd] });
    child.on('error', () => { failure = new Error('owned runtime failed to start'); });
    child.done = new Promise((resolve) => { child.once('exit', resolve); child.once('error', resolve); }); children.push(child);
  };
  // Hold claims until read-only source/payload/proof checks complete. Forward every admitted claim
  // to the real API; never manufacture a capability, route, completed receipt or provider report.
  const proxy = createServer(async (req, res) => {
    try {
      if (req.url === '/api/v1/jobs/claim' && !claimJob) { res.writeHead(200, { 'content-type': 'application/json' }); res.end('{"job":null}'); return; }
      const chunks = []; for await (const chunk of req) chunks.push(chunk);
      const response = await fetch(base + req.url, { method: req.method, headers: { 'content-type': 'application/json' },
        ...(['GET', 'HEAD'].includes(req.method) ? {} : { body: Buffer.concat(chunks) }), signal: AbortSignal.timeout(10000) });
      const bytes = Buffer.from(await response.arrayBuffer());
      if (req.url === '/api/v1/jobs/claim' && response.ok) {
        const claimed = JSON.parse(bytes);
        if (claimed.job) {
          assert.equal(claimed.job.id, claimJob, 'unexpected claimed job');
          same(claimed.portableModelBinding, claimRoute, 'claim route changed');
          assert(HEX.test(claimed.executionClaimCapability), 'private claim capability missing');
          claimJob = null;
        }
      }
      res.writeHead(response.status, { 'content-type': response.headers.get('content-type') ?? 'application/json' }); res.end(bytes);
    } catch { failure = new Error('claim transport/preflight refused'); res.writeHead(503); res.end(); }
  });
  let stage = 'start';
  try {
    stage = 'api_start';
    const proxyPort = await listen(proxy);
    await launch('api', { SCENEWORKS_API_PORT: String(port), SCENEWORKS_RUN_UTILITY_INPROCESS: '0' });
    await wait(async () => { try { return await api('GET', '/api/v1/health'); } catch { return false; } }, 120);
    const { DatabaseSync } = await import('node:sqlite');
    db = new DatabaseSync(env.SCENEWORKS_JOBS_DB_PATH, { readOnly: true });
    stage = 'worker_start';
    await launch('worker', { SCENEWORKS_WORKER_ONLY: '1', SCENEWORKS_API_URL: `http://127.0.0.1:${proxyPort}`, SCENEWORKS_WORKER_PROOF_KEY: workerKey });
    await wait(async () => {
      const workers = await api('GET', '/api/v1/workers');
      const owned = workers.find((worker) => worker.id === workerId);
      return owned?.status === 'idle' && owned.gpuId === String(plan.physicalIndex)
        && db.prepare('select source_proof_authenticated from workers where id=?').get(workerId)?.source_proof_authenticated === 1;
    }, 180);
    const workerProcess = children.at(-1);
    const checkLoadedRuntime = async () => {
      const command = `(Get-Process -Id ${workerProcess.pid}).Modules | Select-Object -ExpandProperty FileName | ConvertTo-Json -Compress`;
      const names = [JSON.parse(execFileSync('powershell', ['-NoProfile', '-NonInteractive', '-Command', command], { encoding: 'utf8' }))].flat();
      const selected = names.filter((name) => /^(?:nv|cuda|cublas|cudnn|cufft|curand|cusolver|cusparse|vcruntime|msvcp).*\.dll$/i.test(path.basename(name)));
      assert(selected.some((name) => /^nvcuda/i.test(path.basename(name))), 'CUDA driver not loaded by worker');
      const actual = await Promise.all(selected.map(async (file) => ({ name: path.basename(file).toLowerCase(), sha256: await fileSha256(file) })));
      assert.equal(new Set(actual.map((file) => file.name)).size, actual.length, 'duplicate loaded runtime libraries');
      const byName = new Map(actual.map((file) => [file.name, file.sha256]));
      for (const file of libraries) assert.equal(byName.get(file.name), file.sha256, 'censused runtime library changed');
      const loadedDigest = digest(actual.sort((a, b) => a.name.localeCompare(b.name)));
      if (qualification) assert.equal(loadedDigest, qualification.loadedRuntimeLibrariesSha256, 'qualified loaded runtime libraries changed');
      if (evidence.loadedRuntimeLibrariesSha256) assert.equal(loadedDigest, evidence.loadedRuntimeLibrariesSha256, 'loaded runtime libraries changed between renders');
      evidence.loadedRuntimeLibrariesSha256 = loadedDigest;
    };
    stage = 'project_create';
    const project = await api('POST', '/api/v1/projects', { name: 'SDXL exact replay proof' });
    const runJob = async (sourceJob) => {
      stage = sourceJob ? 'exact_action' : 'fresh_action';
      alive(); assert.equal(runtime(), baselineRuntime, 'device/runtime changed before render');
      assert.equal(await fileSha256(binary), evidence.binarySha256, 'runtime binary changed');
      const accepted = sourceJob ? await api('POST', `/api/v1/jobs/${sourceJob}/exact-replay`, {})
        : await api('POST', '/api/v1/image/jobs', { ...REQUEST, projectId: project.id, requestedGpu: String(plan.physicalIndex) });
      stage = 'queued_payload';
      const queued = db.prepare('select status, payload_json, portable_model_binding_json from jobs where id=?').get(accepted.id);
      assert.equal(queued?.status, 'queued', 'job bypassed claim preflight');
      const envelope = JSON.parse(queued.portable_model_binding_json);
      const routes = Object.values(envelope.routes).filter((route) => route.backend === 'candle' && route.platform === 'windows');
      assert.equal(routes.length, 1, 'ambiguous accepted Candle route');
      const route = routes[0]; stage = 'selected_source'; await verifyMembers(route, plan);
      const payload = JSON.parse(queued.payload_json); verifyQueuedPayload(payload);
      const payloadSha256 = digest(payload);
      claimRoute = route; claimJob = accepted.id;
      stage = 'render';
      const job = await wait(async () => { const job = await api('GET', `/api/v1/jobs/${accepted.id}`); assert(!['failed', 'canceled', 'interrupted'].includes(job.status), 'render failed/canceled'); return job.status === 'completed' && job; }, 3600);
      stage = 'completed_receipt';
      const publicReceipt = await wait(async () => { try { return await api('GET', `/api/v1/jobs/${job.id}/execution-receipt`); } catch { return false; } }, 30);
      const row = db.prepare('select execution_receipt_json from jobs where id=? and status=\'completed\' and progress_side_effects_pending=0').get(job.id);
      const receipt = JSON.parse(row.execution_receipt_json);
      assert.equal(publicReceipt.acceptedBindingSha256, receipt.acceptedBindingSha256, 'public/private receipt mismatch');
      const asset = job.result.assets?.[0]; assert(asset?.file?.path, 'persisted PNG path missing');
      const projectRoot = await realpath(project.path), file = await realpath(path.resolve(projectRoot, asset.file.path));
      const relative = path.relative(projectRoot, file); assert(relative && !relative.startsWith('..') && !path.isAbsolute(relative), 'PNG escaped owned project');
      const png = await readFile(file);
      stage = 'completed_identity';
      const witness = completedWitness({ job, receipt, route, png, workerId, device: String(plan.physicalIndex), pin, scheduleSha256: plan.scheduleSha256, payloadSha256 });
      stage = 'post_render_environment';
      await verifyMembers(route, plan);
      await runtimeFiles(); await checkLoadedRuntime();
      assert.equal(runtime(), baselineRuntime, 'device/runtime changed after render');
      return witness;
    };
    evidence.jobs.push(await runJob(), await runJob()); stage = 'compare_fresh'; compareFresh(...evidence.jobs);
    if (phase === 'replay') {
      verifyReplayCandidate(evidence.jobs[0], qualification.jobs[0]);
      evidence.jobs.push(await runJob(evidence.jobs[0].jobId));
      compareFresh(evidence.jobs[0], evidence.jobs[2]);
    }
    evidence.bodyAccepted = true;
  } catch (error) {
    // The public artifact must never contain private cache paths, request bodies or runtime logs.
    evidence.failureStage = stage;
    evidence.failureErrorClass = safeErrorClass(error?.name);
    if (stage === 'post_render_environment') evidence.failureCheck = safePostRenderCheck(error?.message);
    throw error;
  } finally {
    claimJob = null;
    let drained = true;
    for (const child of children.reverse()) {
      child.kill(); await Promise.race([child.done, delay(10000)]);
      drained &&= child.pid === undefined || child.exitCode !== null || child.signalCode !== null;
    }
    evidence.bodyAccepted &&= drained;
    proxy.closeAllConnections(); await new Promise((resolve) => proxy.close(resolve));
    db?.close(); for (const handle of handles) await handle.close();
    await writeFile(path.join(output, 'evidence.json'), JSON.stringify(evidence, null, 2) + '\n', { flag: 'wx' });
    await rm(state, { recursive: true, force: true });
    assert(drained, 'owned child did not drain');
  }
}
export function verifyDeviceSelection(metadata, physicalIndex, plan) {
  assert.equal(metadata.context_created, false, 'unexpected context in device metadata');
  same(metadata.devices, [{ ordinal: 0, uuid: plan.deviceUuid }], 'visible CUDA ordinal is not the approved UUID');
  assert.equal(physicalIndex, plan.physicalIndex, 'worker physical device label differs');
}
export function finalizeEvidence(evidence, resource) {
  evidence.accepted = evidence.bodyAccepted === true && resource.resource_completed === true
    && resource.binding?.kind === 'context-free-cuda-ordinal-uuid'
    && resource.binding?.devices?.length === 1 && resource.binding.devices[0].uuid === evidence.deviceUuid
    && resource.binding.devices[0].index === evidence.physicalIndex
    && digest(resource.binding.mapping) === digest([{ cuda_ordinal: 0, uuid: evidence.deviceUuid, nvml_index: evidence.physicalIndex }]);
  evidence.resourceReceiptSha256 = digest(resource);
  assert.equal(evidence.accepted, true, 'physical guard did not complete');
  return evidence;
}
export async function finalizeArtifacts(output, resourceFile) {
  assert(typeof output === 'string' && path.isAbsolute(output), 'absolute owned output required');
  const file = path.join(output, 'evidence.json');
  const evidence = await readDiagnosticJson(file);
  const resource = await readDiagnosticJson(resourceFile);
  try {
    assert(evidence && resource, 'proof or resource evidence unavailable');
    finalizeEvidence(evidence, resource);
    await writeFile(file, JSON.stringify(evidence, null, 2) + '\n');
    return evidence;
  } catch {
    // The guard may have killed the child before its finally block wrote evidence. Preserve a
    // bounded failure artifact in the existing uploader's path, never acceptance or private state.
    await mkdir(output, { recursive: true });
    await writeFile(file, JSON.stringify(failureDiagnostic(evidence, resource), null, 2) + '\n');
    throw new Error('SDXL proof or physical guard did not complete');
  }
}
async function main() {
  if (process.argv[2] === 'preflight') {
    dispatchInput();
    const plan = JSON.parse(await readFile(process.env.SDXL_REPLAY_PLAN, 'utf8'));
    validatePlan(plan, git('rev-parse', 'HEAD'), await inferenceRevision(ROOT), process.env.SDXL_REPLAY_PHASE);
    assert.equal(process.env.CUDA_VISIBLE_DEVICES, plan.deviceUuid, 'approved UUID differs');
    const { DatabaseSync } = await import('node:sqlite');
    assert.equal(typeof DatabaseSync, 'function', 'Node SQLite support missing');
  } else if (process.argv[2] === 'finalize') {
    await finalizeArtifacts(process.env.SDXL_REPLAY_OUTPUT, process.env.SDXL_REPLAY_RESOURCE_RECEIPT);
  } else { assert.equal(process.argv.length, 2, 'unexpected proof arguments'); await runProof(); }
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(() => { console.error('SDXL proof refused; no deterministic declaration was made.'); process.exitCode = 1; });
}
