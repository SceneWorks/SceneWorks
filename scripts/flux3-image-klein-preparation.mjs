// Explicit CPU-only source preparation. Import/plan never reads environment, credentials or cache.
import assert from 'node:assert/strict';
import { spawn, execFile as execCb } from 'node:child_process';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';
import { cp, mkdir, open, readFile, readdir, realpath, stat, statfs, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { hostname, userInfo } from 'node:os';
import { StringDecoder } from 'node:string_decoder';
import { ROOT, shaPattern, digestPattern } from './flux3-image-klein-terminal-plan.mjs';
import { attemptIdentity, bindOwnedEnv, boundApiUrl, ownedApiController, utilityWorkerId, runnerHfCredential, authorizeFlux3Repository, redactSensitiveText, assertClaimedWithin } from './flux3-image-klein-app-lifecycle.mjs';
import { contained, fileHash, safeEnvironment } from './flux3-image-klein-terminal-session.mjs';
import { assertPins } from './qwen-image-2-1-app-lifecycle.mjs';
import { verifyPreparationBuild } from './flux3-image-klein-terminal-build.mjs';
const exec = promisify(execCb);
const hash = value => createHash('sha256').update(typeof value === 'string' || Buffer.isBuffer(value) ? value : JSON.stringify(value)).digest('hex');
const fixed = {
  flux3_klein: ['black-forest-labs/flux-3-image-klein', '8454017f9ebaada30c998fccd9f67a80c7580f68'],
  flux3_klein_base: ['black-forest-labs/flux-3-image-klein-base', '1469d27cfb9bb1bd81d3a93f5a023b39dbf136e4'],
  flux3_image_klein_pup: ['black-forest-labs/flux-3-image-klein-pup', '8b069751eb5dcab23829d9c5eccb470eb8b8a4ed'],
};
const floor = 30 * 1024 ** 3;
const id = value => { assert(/^[a-zA-Z0-9_.-]+$/.test(value ?? ''), 'invalid owned identity'); return value; };
const text = value => typeof value === 'string' && value.length > 0;
export function preparationPlan(request, sources) {
  assert.deepEqual(Object.keys(request).sort(), ['backend', 'inferenceMainSha', 'modelIds', 'sourceSha']);
  assert(['candle', 'mlx'].includes(request.backend));
  assert(shaPattern.test(request.sourceSha) && shaPattern.test(request.inferenceMainSha));
  assert(Array.isArray(request.modelIds) && request.modelIds.length && new Set(request.modelIds).size === request.modelIds.length);
  const models = [...sources.models, sources.upsampler], repositories = [], visited = new Set(), visiting = new Set();
  const platform = request.backend === 'candle' ? 'windows' : 'macos';
  const applicable = d => !d.platforms || d.platforms.includes(platform);
  const visit = modelId => {
    assert(Object.hasOwn(fixed, modelId), 'only fixed FLUX3 source repositories may be prepared');
    if (visited.has(modelId)) return;
    assert(!visiting.has(modelId), 'cyclic preparation co-requisite closure'); visiting.add(modelId);
    const model = models.find(m => m.id === modelId), row = model?.downloads.find(d => !d.coRequisite && applicable(d) && (!d.variant || d.variant === 'bf16'));
    const validateRow = (d, target) => {
      assert(d && d.provider === 'huggingface'); assert.deepEqual([d.repo, d.revision], fixed[target]); assert.deepEqual(d.files, []);
      assert(!d.localDerivation && Number.isSafeInteger(d.estimatedSizeBytes) && d.estimatedSizeBytes > 0);
    };
    validateRow(row, modelId);
    const dependencies = [];
    // The ordinary API queues soft AND hard co-requisites before the primary.
    // Resolve them from source, never ambient cache presence or caller labels.
    for (const d of model.downloads.filter(d => d.coRequisite && applicable(d) && (!d.variant || d.variant === row.variant))) {
      const dependency = Object.keys(fixed).find(key => fixed[key][0] === d.repo);
      assert(dependency && !d.choiceGroup && !d.choiceOption, 'unapproved/ambiguous preparation co-requisite');
      validateRow(d, dependency); visit(dependency);
      const resolved = repositories.find(r => r.modelId === dependency); assert.equal(resolved.downloadBytes, d.estimatedSizeBytes);
      dependencies.push(dependency, ...resolved.dependencies);
    }
    repositories.push({ modelId, repository: row.repo, resolvedRevision: row.revision, files: [], downloadBytes: row.estimatedSizeBytes, dependencies: [...new Set(dependencies)], license: model.license ?? null, licenseUrl: model.licenseUrl ?? null, gated: model.gated === true, ...(row.variant ? { variant: row.variant } : {}) });
    visiting.delete(modelId); visited.add(modelId);
  };
  request.modelIds.forEach(visit);
  const value = { schemaVersion: 1, kind: 'flux3-cpu-preparation-plan', request: structuredClone(request), repositories, downloadBytes: repositories.reduce((n, r) => n + r.downloadBytes, 0), generationAttempts: 0, conversions: false, candidateAuthority: 'OBSERVED_NOT_GPU_APPROVED' };
  return { ...value, planSha256: hash(value) };
}
export function validatePreparationApproval(a, plan, sources) {
  assert.deepEqual(plan, preparationPlan(plan.request, sources), 'preparation plan/catalog changed');
  assert.equal(a.actualProvision, true, 'explicit actualProvision approval required before any side effect');
  for (const key of ['userProvisionApprovalId', 'admissionId', 'approvedBy', 'runner', 'runnerAccount']) assert(text(a[key]), `missing ${key}`);
  id(a.userProvisionApprovalId);
  assert.equal(a.preparationPlanSha256, plan.planSha256); assert.equal(a.sourceSha, plan.request.sourceSha); assert.equal(a.inferenceMainSha, plan.request.inferenceMainSha); assert.equal(a.backend, plan.request.backend);
  assert(shaPattern.test(a.inferenceMainTipSha ?? '') && path.isAbsolute(a.inferenceRepository ?? ''));
  for (const key of ['outputRoot', 'hfHome', 'binary']) assert(path.isAbsolute(a[key] ?? ''), `absolute ${key} required`);
  assert(digestPattern.test(a.binarySha256 ?? '') && a.binarySourceSha === a.sourceSha && a.binaryBuildReceipt && a.preparationBuildReceiptBinding, 'independently approved sealed API build required');
  assert(Number.isInteger(a.timeoutSeconds) && a.timeoutSeconds >= 60 && a.timeoutSeconds <= 10800);
  assert(Number.isSafeInteger(a.diskBudgetBytes) && a.diskBudgetBytes >= plan.downloadBytes + floor);
  assert.equal(a.downloadApprovals?.length, plan.repositories.length);
  for (const row of plan.repositories) {
    const matches = a.downloadApprovals.filter(p => p.repository === row.repository && p.resolvedRevision === row.resolvedRevision); assert.equal(matches.length, 1);
    const p = matches[0]; assert(text(p.userDownloadApprovalId));
    for (const key of ['wholeRepository', 'licenseAcknowledged', 'accountGateAcknowledged', 'diskAcknowledged']) assert.equal(p[key], true, `explicit per-repository ${key} required`);
    assert.equal(p.downloadBytes, row.downloadBytes); assert.equal(p.runnerAccount, a.runnerAccount); assert.equal(p.hfHome, a.hfHome);
  }
  assert.equal(a.generationAttempts ?? 0, 0); assert(!a.approvedPlan && !a.ledgerRoot && !a.extraEnv, 'preparation cannot carry generation authority or environment overrides');
  return plan;
}
export function redactPreparation(value, secrets = []) {
  return redactSensitiveText(String(value), secrets)
    .replace(/https?:\/\/[^\s"'<>]+/gi, raw => { try { const u = new URL(raw); u.username = ''; u.password = ''; if (u.search) u.search = '?REDACTED'; u.hash = ''; return u.href; } catch { return '[REDACTED_URL]'; } })
    .replace(/(authorization\s*["']?\s*[:=]\s*["']?)(?:Bearer\s+)?[^\s"',}]+/gi, '$1[REDACTED]')
    .replace(/\bhf_[A-Za-z0-9]+\b/g, '[REDACTED]');
}
export function preparationEnvironment(env, a, identity, credential) {
  const allowed = new Set(['PATH', 'Path', 'SystemRoot', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT', 'TEMP', 'TMP', 'TMPDIR', 'HOME', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'LANG', 'LC_ALL']);
  // Filter BEFORE value reads: no unrelated ambient credential, proxy, backend,
  // provider or endpoint selection survives into the approved download process.
  const base = safeEnvironment(Object.fromEntries(Object.keys(env).filter(k => allowed.has(k)).map(k => [k, env[k]])));
  return bindOwnedEnv({ ...base, CUDA_VISIBLE_DEVICES: '-1', SCENEWORKS_GPU_ID: 'cpu', SCENEWORKS_BACKEND_CANDLE_ENABLED: 'false', SCENEWORKS_BACKEND_MLX_ENABLED: 'false', SCENEWORKS_CANDLE_REQUIRED: '0', SCENEWORKS_UTILITY_JOBS: 'model_download', SCENEWORKS_UTILITY_WORKERS: '1', HF_HOME: a.hfHome, HF_HUB_CACHE: path.join(a.hfHome, 'hub'), HF_ENDPOINT: 'https://huggingface.co', SCENEWORKS_HUGGINGFACE_BASE_URL: 'https://huggingface.co', ...(credential ? { HF_TOKEN: credential } : {}) }, identity);
}
export function assertPreparationWorkers(workers, workerId, { idle = false } = {}) {
  const base = utilityWorkerId(workerId);
  assert(Array.isArray(workers) && workers.length === 1, 'exactly one owned CPU preparation worker required');
  const w = workers[0]; assert(w.id === base || w.id?.startsWith(`${base}-`));
  assert.equal(w.gpuId, 'cpu'); assert.deepEqual([...w.capabilities].sort(), ['cpu', 'model_download']);
  assert(idle ? w.status === 'idle' : ['idle', 'busy'].includes(w.status), 'owned preparation worker not ready');
  return w;
}
export function preparationTransport(endpoint, repositories, { workerId, now = Date.now } = {}) {
  const downloads = new Map(repositories.map(r => [`/api/v1/models/${id(r.modelId)}/download`, r])), jobs = new Map();
  return {
    async call(method, route, body) {
      const match = /^\/api\/v1\/jobs\/([a-zA-Z0-9_.-]+)(\/cancel)?$/.exec(route);
      const row = downloads.get(route);
      assert((method === 'GET' && body === undefined && ['/api/v1/health', '/api/v1/workers', '/api/v1/models'].includes(route)) || (match && jobs.has(match[1]) && ((method === 'GET' && !match[2] && body === undefined) || (method === 'POST' && match[2] && body === undefined))) || (method === 'POST' && row), 'CPU preparation transport refuses non-download/generation/training/conversion routes');
      if (row) assert.deepEqual(body, { licenseAcknowledged: true, ...(row.variant ? { variant: row.variant } : {}) });
      const requestedAt = now();
      const result = await endpoint.call(method, route, body);
      if (method === 'POST' && row) {
        assert.equal(result.type, 'model_download'); assert.equal(result.payload?.modelId, row.modelId); assert.equal(result.payload?.repo, row.repository); assert.equal(result.payload?.revision, row.resolvedRevision); assert.deepEqual(result.payload?.files, []);
        assert(!Object.hasOwn(result.payload, 'localDerivation') && !Object.hasOwn(result.payload, 'conversion'), 'preparation must never auto-convert'); jobs.set(id(result.id), { row, createdAt: result.createdAt, requestedAt });
      }
      return result;
    },
    async dependencyFor(worker, primaryId, now) {
      assert(workerId, 'owned worker identity required for a dependency lookup');
      assertPreparationWorkers([worker], workerId);
      const parent = jobs.get(primaryId); assert(parent && !parent.dependency, 'owned primary required');
      assert.equal(worker.status, 'busy'); const dependencyId = id(worker.currentJobId); assert.notEqual(dependencyId, primaryId);
      assert(Number.isFinite(Date.parse(parent.createdAt)), 'actual primary creation timestamp required');
      // One read of THIS verified CPU worker's current job, never a raw queue
      // listing or caller-selected job. Admit it only after the complete proof.
      const job = await endpoint.call('GET', `/api/v1/jobs/${dependencyId}`);
      assert.equal(job.id, dependencyId); assert.equal(job.type, 'model_download'); assert.equal(job.workerId, worker.id);
      assert(Number.isFinite(job.progress) && job.progress >= 0 && job.progress <= 1, 'dependency progress counter invalid');
      const row = repositories.find(r => parent.row.dependencies.includes(r.modelId) && r.repository === job.payload?.repo && r.resolvedRevision === job.payload?.revision);
      assert(row, 'CPU worker dependency is outside the explicitly approved closure');
      assert.equal(job.payload?.modelId, parent.row.modelId); assert.deepEqual(job.payload?.files, []);
      assert(!Object.hasOwn(job.payload, 'localDerivation') && !Object.hasOwn(job.payload, 'conversion'), 'dependency must never auto-convert');
      // The API creates dependencies BEFORE the returned primary, possibly on
      // opposite sides of a second boundary. Bind to the observed POST start,
      // allowing only the API timestamp's one-second serialization precision.
      const created = Date.parse(job.createdAt); assert(created >= parent.requestedAt - 1000 && created <= Date.parse(parent.createdAt) && created <= now + 5000, 'dependency is not owned by this download request');
      assert(!['failed', 'canceled', 'interrupted'].includes(job.status), 'owned dependency download failed');
      jobs.set(dependencyId, { row, dependency: true, createdAt: job.createdAt }); return job;
    },
    ownsJob: jobId => jobs.has(jobId),
  };
}
export const PREPARATION_PROGRESS_LEASE_MS = 60_000;
export function advancePreparationClaimClock(state, primary, worker, dependency, now) {
  assert(Number.isFinite(now) && now >= state.lastNow, 'monotonic preparation clock required');
  let activeMs = 0, activeUntil = 0, activeId = null;
  if (!primary.workerId && dependency) {
    assert.equal(worker.status, 'busy'); assert.equal(worker.currentJobId, dependency.id);
    assert.equal(dependency.workerId, worker.id); assert.equal(dependency.type, 'model_download');
    assert(!['failed', 'canceled', 'interrupted'].includes(dependency.status), 'owned dependency download failed');
    const progress = dependency.progress, heartbeat = Date.parse(dependency.lastHeartbeatAt);
    const previous = state.dependencies.get(dependency.id);
    if (!previous) state.dependencies.set(dependency.id, { progress: Number.isFinite(progress) ? progress : -Infinity, heartbeat: Number.isFinite(heartbeat) ? heartbeat : -Infinity, advancedAt: null });
    else {
      // Heartbeat-only keepalives cannot manufacture progress; changing a
      // percentage without an advancing heartbeat cannot renew the lease either.
      if (Number.isFinite(progress) && progress > previous.progress && Number.isFinite(heartbeat) && heartbeat > previous.heartbeat) {
        previous.progress = progress; previous.heartbeat = heartbeat; previous.advancedAt = now;
      }
      const fresh = Number.isFinite(heartbeat) && now - heartbeat >= -5000 && now - heartbeat <= PREPARATION_PROGRESS_LEASE_MS;
      if (['running', 'downloading'].includes(dependency.status) && fresh && previous.advancedAt !== null && now - previous.advancedAt <= PREPARATION_PROGRESS_LEASE_MS) {
        activeId = dependency.id; activeUntil = previous.advancedAt + PREPARATION_PROGRESS_LEASE_MS;
        // Never retroactively exempt an unobserved gap beyond the prior lease.
        if (state.activeId === activeId) activeMs = Math.max(0, Math.min(now, state.activeUntil) - state.lastNow);
      }
    }
  }
  state.suspendedMs += activeMs;
  state.lastNow = now; state.activeId = activeId; state.activeUntil = activeUntil;
  assertClaimedWithin(primary, state.firstSeen + state.suspendedMs, now);
}
export async function waitPreparationDownload({ transport, accepted, workerId, timeoutSeconds, check, disk, now = Date.now, sleep }) {
  const firstSeen = now(), end = firstSeen + timeoutSeconds * 1000;
  const clock = { firstSeen, lastNow: firstSeen, suspendedMs: 0, activeId: null, activeUntil: 0, dependencies: new Map() };
  while (now() < end) {
    check(); await disk(); const job = await transport.call('GET', `/api/v1/jobs/${accepted.id}`);
    assert.equal(job.id, accepted.id); assert.equal(job.type, 'model_download');
    assert(!['failed', 'canceled', 'interrupted'].includes(job.status), 'owned preparation download did not complete');
    const worker = assertPreparationWorkers(await transport.call('GET', '/api/v1/workers'), workerId);
    if (job.workerId) assert.equal(job.workerId, worker.id);
    if (job.status === 'completed') { assert.equal(job.workerId, worker.id); return job; }
    const dependency = !job.workerId && worker.status === 'busy' && worker.currentJobId !== job.id ? await transport.dependencyFor(worker, job.id, now()) : null;
    advancePreparationClaimClock(clock, job, worker, dependency, now());
    await sleep(1000);
  }
  throw new Error('preparation download deadline exceeded');
}
async function insideDirectory(directory, root) {
  const value = await realpath(directory), relative = path.relative(await realpath(root), value);
  assert(relative && !relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative), 'directory escaped approved root'); assert((await stat(value)).isDirectory()); return value;
}
// Byte-identical ports of sceneworks-core `safe_download_dir` (the managed
// `data/models/<X>` receipt directory, keyed by REPOSITORY, never model ID) and
// `safe_repo_dir_name` (the HF hub `models--<X>` directory). Both are pinned by
// tests/fixtures/rust_migration_contracts/repo_slugs.json, which the Rust core,
// API and worker tests assert too, so drift fails on both sides.
export function safeDownloadDir(value) {
  let output = '', replacing = false;
  for (const character of value) {
    if (/^[A-Za-z0-9_.-]$/.test(character)) { output += character; replacing = false; } else if (!replacing) { output += '__'; replacing = true; }
  }
  output = output.replace(/^_+|_+$/g, ''); return output || 'download';
}
export function safeRepoDirName(repo) {
  const output = [...repo].map(character => (/^[A-Za-z0-9._-]$/.test(character) ? character : '--')).join('').replace(/^-+|-+$/g, '');
  return output || null;
}
// Top-level `license:` of a README model-card front matter, else null.
export function readmeCardLicense(text) {
  const match = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(text); if (!match) return null;
  const line = /^license:[ \t]*(.*?)[ \t]*$/m.exec(match[1]); if (!line) return null;
  const value = line[1].replace(/^(['"])(.*)\1$/, '$2').trim(); return value || null;
}
export async function discoverPreparationInventory({ state, hfHome, row, job, authorization, installed }) {
  assert.equal(job.status, 'completed'); assert.equal(job.type, 'model_download');
  assert.equal(job.payload?.modelId, row.modelId); assert.equal(job.payload?.repo, row.repository); assert.equal(job.payload?.revision, row.resolvedRevision); assert.deepEqual(job.payload?.files, []);
  const marker = await contained(path.join(state, 'data/models', safeDownloadDir(row.repository), '.sceneworks-download-complete.json'), state), markerBytes = await readFile(marker), envelope = JSON.parse(markerBytes);
  const matches = (envelope.receipts ?? [envelope]).filter(r => r.jobId === job.id && r.modelId === row.modelId && r.repo === row.repository && r.snapshotRevision === row.resolvedRevision); assert.equal(matches.length, 1);
  const receipt = matches[0]; assert.equal(receipt.schemaVersion, 2); assert(/^\d{4}-\d{2}-\d{2}T[0-9:.+-]+Z?$/.test(receipt.completedAt) && Number.isFinite(Date.parse(receipt.completedAt))); assert.deepEqual(receipt.manifestFiles, []); assert(Array.isArray(receipt.resolvedFiles) && receipt.resolvedFiles.length);
  assert.equal(installed?.id, row.modelId); assert.equal(installed.installState, 'installed');
  // The exact pinned snapshot is derived from the approved coordinates and must
  // equal the completed job's own reported HF-cache path. The catalog reports
  // either that snapshot (FLUX 3 runtimes) or its repository cache root (other
  // models, e.g. PUP); nothing else is accepted.
  const repoDir = safeRepoDirName(row.repository); assert(repoDir, 'approved repository has no HF cache directory');
  const snapshot = await insideDirectory(path.join(hfHome, 'hub', `models--${repoDir}`, 'snapshots', row.resolvedRevision), hfHome);
  assert.equal(job.result?.storage, 'huggingface_cache'); assert.equal(await realpath(job.result.path), snapshot, 'completed job must report the exact pinned snapshot');
  const listed = await realpath(installed.installedPath); assert(listed === snapshot || listed === path.dirname(path.dirname(snapshot)), 'catalog installed path is neither the pinned snapshot nor its repository cache root');
  const files = [];
  async function walk(directory, prefix = '') {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const name = prefix + entry.name; assert(/^[A-Za-z0-9_./-]+$/.test(name) && !name.split('/').includes('..'), 'invalid snapshot entry');
      if (entry.isDirectory()) await walk(path.join(directory, entry.name), `${name}/`);
      else { const file = await contained(path.join(snapshot, name), hfHome); const bytes = (await stat(file)).size; assert(Number.isSafeInteger(bytes) && bytes >= 0, 'invalid snapshot artifact size'); files.push({ path: name, bytes, sha256: await fileHash(file) }); }
    }
  }
  await walk(snapshot); files.sort((a, b) => a.path.localeCompare(b.path));
  assert.deepEqual([...receipt.resolvedFiles].sort(), files.map(f => f.path).sort(), 'actual receipt must cover complete observed snapshot, not a supplied inventory');
  const totalBytes = files.reduce((n, f) => n + f.bytes, 0); assert.equal(totalBytes, row.downloadBytes, 'actual pinned repository size differs from approved whole-repository bytes');
  assert.equal(await fileHash(await contained(path.join(snapshot, 'config.json'), hfHome)), authorization.configSha256);
  // Prefer a shipped LICENSE file. Only when none exists may the README card's
  // own `license:` stand in, and only if it equals the source catalog license.
  const licenseFile = files.find(f => /^LICENSE(?:\.md|\.txt)?$/i.test(f.path)); let license;
  if (licenseFile) license = { source: 'file', path: licenseFile.path, sha256: licenseFile.sha256 };
  else {
    const readme = files.find(f => f.path === 'README.md'); assert(readme && typeof row.license === 'string' && row.license, 'actual upstream license file or catalog-matching README card license required');
    const declared = readmeCardLicense(await readFile(await contained(path.join(snapshot, 'README.md'), hfHome), 'utf8'));
    assert(declared && declared.toLowerCase() === row.license.toLowerCase(), 'README card license must equal the source catalog license');
    license = { source: 'readme-card', path: readme.path, sha256: readme.sha256, license: declared };
  }
  return { authority: 'OBSERVED_NOT_GPU_APPROVED', gpuExecutionApproved: false, modelId: row.modelId, repository: row.repository, resolvedRevision: row.resolvedRevision, root: snapshot, containmentRoot: await realpath(hfHome), totalBytes, files, jobId: job.id, workerId: job.workerId, receiptSha256: hash(markerBytes), receipt: { schemaVersion: receipt.schemaVersion, jobId: receipt.jobId, modelId: receipt.modelId, repo: receipt.repo, snapshotRevision: receipt.snapshotRevision, completedAt: receipt.completedAt, manifestFiles: receipt.manifestFiles, resolvedFiles: receipt.resolvedFiles }, authorization: { httpStatus: authorization.httpStatus, configSha256: authorization.configSha256 }, license };
}
// Keep complete SANITIZED lines: first 64 KiB plus a rolling tail, <=2 MiB
// total. Each pipe's incomplete raw line is separately bounded to 64 KiB;
// oversized lines are discarded whole, never cut into a token or signed URL.
export function boundedPreparationLog(secrets) {
  const cap = 2 * 1024 ** 2, firstCap = 64 * 1024, lineCap = 64 * 1024;
  const marker = '[older log lines omitted]\n', tailCap = cap - firstCap - Buffer.byteLength(marker);
  const streams = new Map(), tail = []; let first = '', firstClosed = false, tailBytes = 0, offset = 0, omitted = false;
  const retain = raw => {
    let line = redactPreparation(raw, secrets), bytes = Buffer.byteLength(line);
    if (bytes > tailCap) { line = '[oversized sanitized line omitted]\n'; bytes = Buffer.byteLength(line); }
    if (!firstClosed && Buffer.byteLength(first) + bytes <= firstCap) { first += line; return; }
    firstClosed = true; tail.push({ line, bytes }); tailBytes += bytes;
    while (tailBytes > tailCap) { tailBytes -= tail[offset].bytes; tail[offset++] = null; omitted = true; }
    if (offset > 1024 && offset * 2 > tail.length) { tail.splice(0, offset); offset = 0; }
  };
  const consume = (state, text) => {
    let start = 0;
    while (start < text.length) {
      const newline = text.indexOf('\n', start), end = newline < 0 ? text.length : newline;
      if (!state.discard) {
        const part = text.slice(start, end);
        if (Buffer.byteLength(state.pending) + Buffer.byteLength(part) > lineCap) {
          state.pending = ''; state.discard = true; retain('[oversized log line omitted]\n');
        } else state.pending += part;
      }
      if (newline < 0) break;
      if (!state.discard) retain(state.pending + '\n');
      state.pending = ''; state.discard = false; start = newline + 1;
    }
  };
  const stateFor = key => {
    if (!streams.has(key)) { assert(streams.size < 2, 'only owned stdout/stderr may be retained'); streams.set(key, { decoder: new StringDecoder('utf8'), pending: '', discard: false, ended: false }); }
    return streams.get(key);
  };
  const end = key => { const state = stateFor(key); if (state.ended) return; consume(state, state.decoder.end()); if (state.pending && !state.discard) retain(state.pending + '\n'); state.pending = ''; state.ended = true; };
  return {
    push(key, bytes) { const state = stateFor(key); if (!state.ended) consume(state, state.decoder.write(bytes)); },
    end,
    finish() { for (const key of streams.keys()) end(key); },
    text() { return first + (omitted ? marker : '') + tail.slice(offset).map(entry => entry.line).join(''); },
  };
}
// Dedicated factory: deliberately not launchTerminalSession with weakened GPU/ledger checks.
async function cpuSession(a, io, credential) {
  const identity = attemptIdentity(a.outputRoot), controller = new AbortController(), secrets = [credential];
  await mkdir(identity.state); await mkdir(path.join(identity.state, 'data/projects'), { recursive: true }); await mkdir(path.join(identity.state, 'credentials')); await cp(path.join(io.root, 'config'), path.join(identity.state, 'config'), { recursive: true });
  const log = await open(path.join(a.outputRoot, `${identity.nonce}-cpu-api.log`), 'wx'), retained = boundedPreparationLog(secrets); let writes = Promise.resolve(), logError;
  let child;
  try { child = io.spawn(a.binary, [], { cwd: io.root, env: preparationEnvironment(io.env, a, identity, credential), windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] }); }
  catch { await log.close(); throw new Error('owned CPU API launch refused'); }
  const done = new Promise(resolve => { child.once('close', resolve); child.once('error', () => controller.abort(new Error('owned CPU API launch failed'))); });
  for (const stream of [child.stdout, child.stderr]) {
    // Always drain both pipes, including shutdown output after cancellation.
    // Retention volume cannot abort the session or backpressure its child.
    stream.on('data', bytes => retained.push(stream, bytes));
    stream.on('end', () => retained.end(stream));
  }
  const deadline = setTimeout(() => controller.abort(new Error('preparation deadline exceeded')), a.timeoutSeconds * 1000);
  let endpoint;
  const close = async () => {
    clearTimeout(deadline); endpoint?.close(); controller.abort();
    if (child.exitCode === null && child.signalCode === null) { child.kill('SIGTERM'); let timer; await Promise.race([done, new Promise(resolve => { timer = setTimeout(resolve, 20_000); })]); clearTimeout(timer); if (child.exitCode === null && child.signalCode === null) { child.kill('SIGKILL'); await done; } }
    await done; retained.finish();
    writes = writes.then(() => log.write(retained.text())); await writes.catch(() => { logError = true; }); await log.close(); if (logError) throw new Error('owned preparation log retention failed');
  };
  const check = () => { if (controller.signal.aborted) throw new Error('owned preparation session stopped'); assert(child.exitCode === null && child.signalCode === null, 'owned CPU API exited'); };
  const wait = async fn => { for (let n = 0; n < 360; n++) { check(); const value = await fn(); if (value) return value; await io.sleep(500); } throw new Error('owned CPU utility readiness timeout'); };
  try {
    const url = await wait(() => boundApiUrl(retained.text())); endpoint = ownedApiController({ url, state: identity.state, api: child, controller, fetchImpl: io.fetch });
    await wait(() => endpoint.verify());
    await wait(async () => { const workers = await endpoint.call('GET', '/api/v1/workers'); if (!workers.length) return false; return assertPreparationWorkers(workers, identity.workerId, { idle: true }); });
    return { identity, endpoint, close, check };
  } catch { await close(); throw new Error('owned CPU preparation startup failed'); }
}
export async function runPreparation(a, plan, sources, dependencies = {}) {
  validatePreparationApproval(a, plan, sources); // ALL approvals before environment/credentials/spawn/network.
  const io = { root: ROOT, env: process.env, platform: process.platform, account: () => userInfo().username, host: hostname, git: async (root, ...args) => (await exec('git', args, { cwd: root, timeout: 30_000 })).stdout.trim(), verifyBuild: verifyPreparationBuild, statfs, readCredential: runnerHfCredential, authorize: authorizeFlux3Repository, spawn, fetch, now: Date.now, sleep: ms => new Promise(resolve => setTimeout(resolve, ms)), ...dependencies };
  assert.equal(io.env.CUDA_VISIBLE_DEVICES, '-1', 'CPU preparation requires explicit parent CUDA_VISIBLE_DEVICES=-1');
  assert.equal(a.runnerAccount, io.account()); assert.equal(a.runner, io.host().split('.')[0]);
  assert.equal(io.platform, a.backend === 'candle' ? 'win32' : 'darwin'); if (io.platform === 'darwin') assert(['nax-macos', 'nax-macos-2'].includes(a.runner), 'only a known physical Mac runner prepares MLX');
  assert.equal(await io.git(io.root, 'rev-parse', 'HEAD'), a.sourceSha); assert.equal(await io.git(io.root, 'status', '--porcelain'), '');
  assert.equal(await io.git(a.inferenceRepository, 'rev-parse', 'refs/remotes/origin/main'), a.inferenceMainTipSha); await io.git(a.inferenceRepository, 'merge-base', '--is-ancestor', a.inferenceMainSha, 'refs/remotes/origin/main');
  assertPins(await Promise.all(['Cargo.toml', 'crates/sceneworks-worker/Cargo.toml', 'crates/sceneworks-memory-adapter/Cargo.toml'].map(f => readFile(path.join(io.root, f), 'utf8'))), await readFile(path.join(io.root, 'Cargo.lock'), 'utf8'), a.inferenceMainSha);
  await io.verifyBuild(a, { sourceRoot: io.root });
  const checkout = await realpath(io.root); for (const directory of [a.outputRoot, a.hfHome]) { const actual = await realpath(directory), relative = path.relative(checkout, actual); assert(relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative), 'preparation storage must be outside checkout'); }
  const disk = async () => { const cache = await io.statfs(a.hfHome), output = await io.statfs(a.outputRoot); assert(cache.bavail * cache.bsize >= floor && output.bavail * output.bsize >= floor, 'preparation disk safety floor'); };
  const initial = await io.statfs(a.hfHome); assert(initial.bavail * initial.bsize >= a.diskBudgetBytes); await disk();
  let credential, session; const inventories = [], authorizations = [];
  try {
    if (plan.repositories.some(r => r.modelId !== 'flux3_image_klein_pup')) credential = await io.readCredential({ HF_HOME: a.hfHome, HOME: io.env.HOME });
    for (const row of plan.repositories) { const proof = await io.authorize(row.repository, row.resolvedRevision, credential, { fetchImpl: io.fetch, signal: AbortSignal.timeout(30_000) }); assert.equal(proof.httpStatus, 200); assert(digestPattern.test(proof.configSha256)); authorizations.push(proof); }
    session = await cpuSession(a, io, credential); const transport = preparationTransport(session.endpoint, plan.repositories, { workerId: session.identity.workerId, now: io.now });
    for (const [index, row] of plan.repositories.entries()) {
      session.check(); await disk();
      let ready = false;
      for (let n = 0; n < 300; n++) {
        session.check(); const worker = assertPreparationWorkers(await transport.call('GET', '/api/v1/workers'), session.identity.workerId);
        if (worker.status === 'idle') { ready = true; break; }
        await io.sleep(1000);
      }
      assert(ready, 'owned CPU download worker did not return idle');
      const accepted = await transport.call('POST', `/api/v1/models/${row.modelId}/download`, { licenseAcknowledged: true, ...(row.variant ? { variant: row.variant } : {}) });
      let finished;
      try {
        finished = await waitPreparationDownload({ transport, accepted, workerId: session.identity.workerId, timeoutSeconds: a.timeoutSeconds, check: session.check, disk, now: io.now, sleep: io.sleep });
      } catch { await transport.call('POST', `/api/v1/jobs/${accepted.id}/cancel`).catch(() => {}); throw new Error('owned preparation download failed'); }
      const installed = (await transport.call('GET', '/api/v1/models')).find(m => m.id === row.modelId);
      inventories.push(await discoverPreparationInventory({ state: session.identity.state, hfHome: a.hfHome, row, job: finished, authorization: authorizations[index], installed }));
    }
    const report = { schemaVersion: 1, kind: 'flux3-cpu-preparation-observed-inventories', status: 'OBSERVED', gpuExecutionApproved: false, generationAttempts: 0, sourceSha: a.sourceSha, inferenceMainSha: a.inferenceMainSha, runner: a.runner, runnerAccount: a.runnerAccount, binarySha256: a.binarySha256, buildReceiptSha256: a.binaryBuildReceipt.sha256, preparationPlanSha256: plan.planSha256, userProvisionApprovalId: a.userProvisionApprovalId, inventories };
    const file = path.join(a.outputRoot, `${session.identity.nonce}-observed-inventories.json`); await writeFile(file, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' }); return { file, status: report.status };
  } catch { throw new Error('CPU preparation refused or incomplete; no GPU evidence or execution approval produced'); }
  finally { credential = undefined; if (session) await session.close(); }
}
