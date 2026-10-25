#!/usr/bin/env node
// sc-26691 frozen weight staging. Production entry is explicit workflow_dispatch Actions ONLY.
// No model execution, no GPU initialization, no downloaded-code execution, no shared-cache mutation.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { constants as fsConstants, createReadStream } from 'node:fs';
import { access, appendFile, copyFile, lstat, mkdir, open, readFile, readdir, realpath, rename, rm, stat, writeFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const exec = promisify(execFile);
export const CASEBOOK_RELATIVE = 'docs/testing/sc-26691-h3-cases.json';
export const CASEBOOK_URL = new URL('../docs/testing/sc-26691-h3-cases.json', import.meta.url);
export const SINGULARITY_SET = 'singularity';
export const HF_HOST = 'https://huggingface.co';
export const FILE_TIMEOUT_MS = 60 * 60 * 1000;   // per-file request bound
export const JOB_BOUND_MS = 240 * 60 * 1000;     // workflow job bound (set by the parent workflow)
export const STAGE_BACKENDS = Object.freeze(['mlx', 'cuda']);
export const MLX_RUNNER = 'nax-macos-2';
export const CUDA_RUNNERS = Object.freeze(['cuda-windows', 'cuda-windows-2']);
export const CLEANUP_ACKNOWLEDGEMENT = 'archived-verified-av-and-backend-readbacks; all owned services stopped';
const SINGULARITY_REPO = 'WarmBloodAban/Minimax-h3_Singularity';
const MARKER_FILE = 'sc26691-owner.json';
const RECEIPT_FILE = 'h3-stage-receipt.json';
const sha64 = v => assert.match(v ?? '', /^[a-f0-9]{64}$/);
const sha40 = v => assert.match(v ?? '', /^[a-f0-9]{40}$/);

function stageError(code, message) {
  const error = new Error(`${code}: ${message}`);
  error.code = code;
  return error;
}
export async function hashFileSha256(filePath, { signal, deadline, now = Date.now } = {}) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(filePath)) {
    if (signal?.aborted) throw stageError('H3_STAGE_INTERRUPTED', 'staging interrupted during content hash');
    if (deadline !== undefined && now() > deadline) throw stageError('H3_STAGE_INTERRUPTED', 'job deadline exceeded during content hash');
    hash.update(chunk);
  }
  return hash.digest('hex');
}
// A single FileHandle.write may write fewer bytes than requested. Count only actual written bytes
// and never report progress for unwritten bytes.
export async function writeAllBytes(handle, chunk, { signal, deadline, now = Date.now, onWritten } = {}) {
  let written = 0;
  while (written < chunk.length) {
    if (signal?.aborted) throw stageError('H3_STAGE_INTERRUPTED', 'staging interrupted during file write');
    if (deadline !== undefined && now() > deadline) throw stageError('H3_STAGE_INTERRUPTED', 'job deadline exceeded during file write');
    const { bytesWritten } = await handle.write(chunk, written, chunk.length - written);
    assert.ok(Number.isInteger(bytesWritten) && bytesWritten > 0, 'file write made no forward progress');
    written += bytesWritten;
    await onWritten?.(bytesWritten);
  }
  return written;
}
export async function loadCasebook() { return JSON.parse(await readFile(CASEBOOK_URL, 'utf8')); }
export async function casebookSha256() { return hashFileSha256(CASEBOOK_URL); }
// HF-relative paths may be nested (clean/file.wav, transformer/shard-00001.safetensors), but must
// stay a safe POSIX relative path: no absolute, drive/colon, backslash, null, '.' or '..' segment.
function assertSafeFileName(file) {
  assert.ok(typeof file === 'string' && file.length > 0, 'exact source relative path required');
  assert.ok(!file.includes('\0') && !file.includes('\\') && !file.includes(':'), 'unsafe source relative path characters');
  assert.ok(!path.posix.isAbsolute(file) && !path.win32.isAbsolute(file), 'source path must be relative, never absolute');
  const segments = file.split('/');
  assert.ok(segments.length > 0 && segments.every(segment => segment.length > 0 && segment !== '.' && segment !== '..'), 'unsafe source relative path segment');
  return segments;
}
const fileSegments = file => assertSafeFileName(file);
function assertRepo(repo) { assert.match(repo ?? '', /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/, 'exact Hugging Face repo required'); }
function validateLegal(legal) {
  assert.ok(legal && typeof legal === 'object', 'expected legal metadata must be an object');
  assert.ok(typeof legal.licenseId === 'string' && legal.licenseId.length > 0, 'expected legal.licenseId required');
  const known = { licenseId: legal.licenseId };
  if (legal.inheritedLicenseId !== undefined) { assert.ok(typeof legal.inheritedLicenseId === 'string' && legal.inheritedLicenseId.length > 0, 'expected inheritedLicenseId must be non-empty'); known.inheritedLicenseId = legal.inheritedLicenseId; }
  if (legal.attributionRequired !== undefined) { assert.ok(typeof legal.attributionRequired === 'string' && legal.attributionRequired.length > 0, 'expected attributionRequired must be non-empty'); known.attributionRequired = legal.attributionRequired; }
  return known;
}
function validateFileEntry(entry) {
  const repoType = entry.repoType ?? 'model';
  assert.ok(repoType === 'model' || repoType === 'dataset', 'repoType must be model or dataset');
  assertRepo(entry.repo);
  sha40(entry.revision);
  assertSafeFileName(entry.file);
  assert.ok(Number.isSafeInteger(entry.expectedBytes) && entry.expectedBytes > 0, 'positive frozen expected size required');
  sha64(entry.expectedSha256);
  return { repoType, repo: entry.repo, revision: entry.revision, file: entry.file, expectedBytes: entry.expectedBytes, expectedSha256: entry.expectedSha256, ...(entry.legal !== undefined ? { legal: validateLegal(entry.legal) } : {}) };
}
export const assetKey = entry => `${entry.repoType}:${entry.repo}@${entry.revision}/${entry.file}`;
// The frozen Singularity set is ALWAYS the canonical four; caller hashes can never override it.
export function canonicalSingularityFiles(book) {
  assert.equal(book?.source?.repo, SINGULARITY_REPO, 'wrong frozen Singularity repo');
  sha40(book.source.revision);
  assert.ok(Array.isArray(book.artifacts) && book.artifacts.length === 4, 'frozen Singularity set must be exactly four artifacts');
  const legal = book.stagingLicenses?.singularity;
  return book.artifacts.map(artifact => validateFileEntry({
    repo: book.source.repo, revision: book.source.revision, file: artifact.artifact,
    expectedBytes: artifact.expectedBytes, expectedSha256: artifact.expectedSha256,
    ...(legal !== undefined ? { legal } : {}),
  }));
}
// Additional sets are opt-in book metadata only (book.stagingAssets[setId]); request/file overrides
// are refused. Unknown, empty or duplicate selected paths reject.
export function selectStageFiles(book, setIds = [SINGULARITY_SET]) {
  assert.ok(Array.isArray(setIds) && setIds.length > 0, 'asset_sets must be a non-empty array');
  const files = [], seen = new Set();
  for (const setId of setIds) {
    assert.ok(typeof setId === 'string' && setId.length > 0, 'staging set id required');
    let selected;
    if (setId === SINGULARITY_SET) selected = canonicalSingularityFiles(book);
    else {
      const rows = book.stagingAssets?.[setId];
      assert.ok(Array.isArray(rows) && rows.length > 0, `unknown or empty staging asset set ${setId}`);
      selected = rows.map(validateFileEntry);
    }
    for (const file of selected) {
      const key = assetKey(file);
      assert.ok(!seen.has(key), `duplicate selected staging path ${key}`);
      seen.add(key); files.push(file);
    }
  }
  return files;
}
// Known HF hub cache roots only. Read-only exact-file probes; never a cache tree scan.
export function cacheHub({ platform = process.platform, env = process.env } = {}) {
  if (platform === 'win32') {
    const hub = env.HF_HUB_CACHE ?? path.win32.join('E:\\', 'huggingface', 'hub');
    assert.ok(path.win32.isAbsolute(hub), 'known Windows HF hub cache must be absolute');
    return hub;
  }
  const hub = env.HF_HUB_CACHE ?? path.join(env.HF_HOME ?? path.join(env.HOME ?? os.homedir(), '.cache', 'huggingface'), 'hub');
  assert.ok(path.isAbsolute(hub), 'known HF hub cache must be absolute');
  return hub;
}
// Probe the exact snapshot path for a selected file. A snapshot symlink is followed only after the
// resolved bytes are confined to the known hub. Cache absence is reported, never assumed present.
export async function probeCacheFile(entry, hub, { signal } = {}) {
  if (signal?.aborted) throw stageError('H3_STAGE_INTERRUPTED', 'cache probe interrupted');
  const candidate = path.join(hub, cacheRepoDir(entry), 'snapshots', entry.revision, ...fileSegments(entry.file));
  let info;
  try { info = await lstat(candidate); } catch (error) { if (error.code === 'ENOENT') return { state: 'absent', hub, candidate }; throw error; }
  const hubReal = await realpath(hub);
  const candidateReal = await realpath(candidate);
  assert.ok(within(hubReal, candidateReal), 'cache snapshot file escapes the known HF hub; refusing to follow');
  const fileInfo = await stat(candidateReal);
  assert.ok(fileInfo.isFile(), 'cached snapshot must resolve to a regular file');
  return { state: 'present', hub, hubReal, candidate, sourcePath: candidateReal, size: fileInfo.size };
}
export function parseStageRequest(raw) {
  const request = typeof raw === 'string' ? JSON.parse(raw) : raw;
  assert.ok(request && typeof request === 'object', 'H3_STAGE_REQUEST call request JSON required');
  assert.ok(request.files === undefined && request.stagingAssets === undefined && request.assetSetFiles === undefined, 'caller cannot override frozen staging metadata');
  const assetSets = request.asset_sets ?? request.assetSets ?? [SINGULARITY_SET];
  assert.ok(Array.isArray(assetSets) && assetSets.length > 0 && assetSets.every(v => typeof v === 'string'), 'asset_sets must be a non-empty string array');
  return { ...request, assetSets };
}
// Explicit real legal authority BEFORE any download. Never accept terms/login automatically.
export function assertStageAuthority(authority, files, { allowUnitFake = false } = {}) {
  assert.ok(authority && typeof authority === 'object', 'explicit license authority required before any download');
  assert.equal(authority.acknowledged, true, 'license must be explicitly acknowledged');
  assert.ok(authority.kind === 'user-approved' || (allowUnitFake && authority.kind === 'unit_fake'), 'synthetic consent is not real legal approval');
  assert.ok(typeof authority.approvedBy === 'string' && authority.approvedBy.length > 0, 'approving authority required');
  assert.ok(typeof authority.reference === 'string' && authority.reference.length > 0, 'approval reference required');
  assert.ok(Number.isFinite(Date.parse(authority.approvedAt)), 'approval timestamp required');
  fileAcknowledgments(authority, files);
  return authority;
}
// Copy ONLY known explicit acknowledgment fields from a caller subject; never arbitrary/secret fields.
const KNOWN_SUBJECT_FIELDS = ['sha256', 'acknowledged', 'licenseId', 'inheritedLicenseId', 'inheritedAcknowledged', 'attribution', 'role'];
export function knownSubject(subject) {
  return Object.fromEntries(KNOWN_SUBJECT_FIELDS.filter(key => subject?.[key] !== undefined).map(key => [key, subject[key]]));
}
// The receipt's explicitAcknowledgment comes only from actual caller subjects, never book defaults.
export function fileAcknowledgments(authority, files) {
  const subjects = Array.isArray(authority?.subjects) ? authority.subjects : [];
  const acknowledgments = new Map();
  for (const file of files) {
    const legal = file.legal;
    assert.ok(legal && typeof legal.licenseId === 'string' && legal.licenseId.length > 0, `real staging requires explicit legal.licenseId for selected file ${file.file}`);
    const candidates = subjects.filter(subject => subject?.sha256 === file.expectedSha256);
    assert.ok(candidates.length > 0, `missing explicit license authority for selected file ${file.file}`);
    const match = candidates.find(subject =>
      subject.acknowledged === true &&
      subject.licenseId === legal.licenseId &&
      (legal.inheritedLicenseId === undefined || (subject.inheritedLicenseId === legal.inheritedLicenseId && subject.inheritedAcknowledged === true)) &&
      (legal.attributionRequired === undefined || (typeof subject.attribution === 'string' && subject.attribution.includes(legal.attributionRequired)))
    );
    assert.ok(match, `selected file ${file.file} lacks an explicit acknowledged licenseId${legal.inheritedLicenseId !== undefined ? '/inherited' : ''}${legal.attributionRequired !== undefined ? '/attribution' : ''} match`);
    acknowledgments.set(assetKey(file), knownSubject(match));
  }
  return acknowledgments;
}
export function privateRunRoot({ backend, platform, env = process.env, runId, attempt }) {
  assert.ok(STAGE_BACKENDS.includes(backend), 'backend must be mlx or cuda');
  assert.match(String(runId ?? ''), /^\d+$/, 'GITHUB_RUN_ID required for a per-run private root');
  assert.match(String(attempt ?? ''), /^\d+$/, 'GITHUB_RUN_ATTEMPT required for a per-run private root');
  const leaf = `${runId}-${attempt}`;
  if (platform === 'win32') return path.win32.join('E:\\', 'huggingface', 'sc26691-private', leaf);
  const hfHome = env.HF_HOME ?? path.join(env.HOME ?? os.homedir(), '.cache', 'huggingface');
  assert.ok(path.isAbsolute(hfHome), 'HF_HOME must be absolute');
  return path.join(hfHome, 'sc26691-private', leaf);
}
// Production CLI guards, checked BEFORE any filesystem/network action.
export function assertCliEnvironment({ env = process.env, platform = process.platform, backend } = {}) {
  assert.equal(env.GITHUB_ACTIONS, 'true', 'frozen staging is explicit Actions-only');
  assert.equal(env.GITHUB_EVENT_NAME, 'workflow_dispatch', 'manual workflow_dispatch trigger required');
  assert.match(String(env.GITHUB_RUN_ID ?? ''), /^\d+$/, 'GITHUB_RUN_ID required');
  assert.match(String(env.GITHUB_RUN_ATTEMPT ?? ''), /^\d+$/, 'GITHUB_RUN_ATTEMPT required');
  sha40(env.GITHUB_SHA);
  if (env.H3_STAGE_SOURCE_SHA !== undefined) assert.equal(env.H3_STAGE_SOURCE_SHA, env.GITHUB_SHA, 'caller must check out the exact GITHUB_SHA source');
  const selected = backend ?? env.H3_STAGE_BACKEND;
  assert.ok(STAGE_BACKENDS.includes(selected), 'backend must be mlx or cuda');
  if (selected === 'mlx') { assert.equal(platform, 'darwin', 'mlx staging is macOS-only'); assert.equal(env.RUNNER_NAME, MLX_RUNNER, `mlx staging requires ${MLX_RUNNER}`); }
  else { assert.equal(platform, 'win32', 'cuda staging is Windows-only'); assert.ok(CUDA_RUNNERS.includes(env.RUNNER_NAME), 'approved full-weights CUDA listener required'); }
  return { backend: selected, codeSha: env.GITHUB_SHA };
}
// HF hub repo directory prefix: model repos use models--, dataset repos datasets-- (same hub root).
function cacheRepoDir(entry) {
  return `${entry.repoType === 'dataset' ? 'datasets--' : 'models--'}${entry.repo.replaceAll('/', '--')}`;
}
function snapshotDir(root, entry) { return path.join(root, 'hf', 'hub', cacheRepoDir(entry), 'snapshots', entry.revision); }
export function sourceUrl(entry) {
  const repoPath = entry.repoType === 'dataset' ? `datasets/${entry.repo}` : entry.repo;
  return `${HF_HOST}/${repoPath}/resolve/${entry.revision}/${fileSegments(entry.file).map(segment => encodeURIComponent(segment)).join('/')}`;
}
function within(root, candidate) {
  const relative = path.relative(path.resolve(root), path.resolve(candidate));
  return relative.length > 0 && !relative.startsWith('..') && !path.isAbsolute(relative);
}
async function lstatOrUndefined(target) { try { return await lstat(target); } catch (error) { if (error.code === 'ENOENT') return undefined; throw error; } }
// Refuse symlinks anywhere below the owned root; never write through a user/shared link.
async function ensureOwnedDir(root, directory) {
  assert.ok(within(root, directory) || path.resolve(root) === path.resolve(directory), 'staging directory must remain below the private root');
  const rootInfo = await lstat(root);
  assert.ok(rootInfo.isDirectory() && !rootInfo.isSymbolicLink(), 'private root must be a real directory');
  const relative = path.relative(root, directory);
  let current = root;
  for (const part of relative ? relative.split(path.sep) : []) {
    current = path.join(current, part);
    const info = await lstatOrUndefined(current);
    if (!info) await mkdir(current);
    else assert.ok(info.isDirectory() && !info.isSymbolicLink(), 'staging path component must not be a symlink or non-directory');
  }
}
async function ensureOwnedFileTarget(target) {
  const info = await lstatOrUndefined(target);
  if (!info) return 'absent';
  assert.ok(!info.isSymbolicLink(), 'refusing to write through a symlink');
  assert.ok(info.isFile(), 'staging target must be a regular file');
  return 'regular';
}
async function assertOwnedEvidence(evidenceDir, root) {
  assert.ok(path.isAbsolute(evidenceDir), 'H3_STAGE_EVIDENCE must be an absolute runner-temporary path');
  assert.ok(!within(root, evidenceDir) && !within(evidenceDir, root), 'evidence directory must be separate from the weight root');
  await mkdir(evidenceDir, { recursive: true });
}
async function persistReceipt(root, evidenceDir, receipt) {
  const body = JSON.stringify(receipt, null, 2) + '\n';
  const target = path.join(root, RECEIPT_FILE);
  await writeFile(`${target}.tmp`, body);
  await rename(`${target}.tmp`, target);
  if (evidenceDir) {
    const evidenceTarget = path.join(evidenceDir, `stage-receipt-${receipt.producer.runId}-${receipt.producer.runAttempt}.json`);
    await writeFile(`${evidenceTarget}.tmp`, body);
    await rename(`${evidenceTarget}.tmp`, evidenceTarget);
  }
}
async function appendEvidenceLog(evidenceDir, line) {
  if (!evidenceDir) return;
  await appendFile(path.join(evidenceDir, 'stage.log'), `${line}\n`);
}
// Real staging must never buffer a whole (e.g. 34 GB) response. Only a streaming async iterable is
// accepted; a non-streaming body is refused rather than materialized through arrayBuffer().
async function* bodyChunks(response) {
  const body = response?.body;
  assert.ok(body && typeof body[Symbol.asyncIterator] === 'function', 'source response must expose a streaming async-iterable body; refusing to buffer a full payload');
  for await (const chunk of body) yield Buffer.from(chunk);
}
const header = (response, name) => (typeof response?.headers?.get === 'function' ? response.headers.get(name) : response?.headers?.[name]) ?? '';
async function sleepUnlessAborted(sleep, ms, signal) {
  if (signal?.aborted) throw stageError('H3_STAGE_INTERRUPTED', 'staging interrupted before completion');
  await sleep(ms);
  if (signal?.aborted) throw stageError('H3_STAGE_INTERRUPTED', 'staging interrupted before completion');
}
// Stream into the same per-run .part only. Range is validated; a 200 response never appends.
async function downloadToPart({ entry, partPath, fetchImpl, fileTimeoutMs, maxRetries, sleep, signal, onChunk, deadline, now = Date.now }) {
  const url = sourceUrl(entry);
  for (let attempt = 0; ; attempt += 1) {
    if (signal?.aborted) throw stageError('H3_STAGE_INTERRUPTED', 'staging interrupted before completion');
    const existingInfo = await lstatOrUndefined(partPath);
    if (existingInfo) assert.ok(existingInfo.isFile() && !existingInfo.isSymbolicLink(), 'partial staging file must be a regular file');
    const existing = existingInfo ? existingInfo.size : 0;
    assert.ok(existing <= entry.expectedBytes, 'staged partial already exceeds the frozen expected size');
    // Abort promptly on the external SIGTERM/SIGINT controller, the per-file timeout, and the job
    // deadline, so a hanging header/body lets the interruption receipt flush.
    const remaining = deadline === undefined ? fileTimeoutMs : Math.max(1, Math.min(fileTimeoutMs, deadline - now()));
    const timeoutSignal = AbortSignal.timeout(Math.max(1, remaining));
    const combined = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;
    let response;
    try {
      response = await fetchImpl(url, { headers: existing > 0 ? { Range: `bytes=${existing}-` } : {}, redirect: 'follow', signal: combined });
    } catch (error) {
      if (signal?.aborted) throw stageError('H3_STAGE_INTERRUPTED', 'staging interrupted while awaiting the source response');
      if (combined.aborted) throw stageError('H3_STAGE_TIMEOUT', `${url}: source request timed out after ${fileTimeoutMs}ms${deadline !== undefined ? ' or the job bound' : ''}`);
      if (attempt < maxRetries) { await sleepUnlessAborted(sleep, Math.min(1000 * 2 ** attempt, 15000), signal); continue; }
      throw stageError('H3_STAGE_TRANSPORT', `${url}: ${error.message}`);
    }
    const status = response.status;
    if (status === 401 || status === 403) throw stageError(`H3_STAGE_HTTP_${status}`, `${url}: gated/unauthorized source; explicit blocker, no fallback weights`);
    let append = existing > 0;
    if (status === 206) {
      const match = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(String(header(response, 'content-range')));
      assert.ok(match, 'resumed range response requires a valid Content-Range');
      assert.equal(Number(match[1]), existing, 'Content-Range start must match the retained partial bytes');
      assert.equal(Number(match[3]), entry.expectedBytes, 'Content-Range total must equal the frozen expected size');
    } else if (status === 200) {
      // Range ignored: never append bytes from a full-body response onto a partial file.
      if (existing > 0) { append = false; }
    } else if (status === 429 || status >= 500) {
      if (attempt < maxRetries) { await sleepUnlessAborted(sleep, Math.min(1000 * 2 ** attempt, 15000), signal); continue; }
      throw stageError(`H3_STAGE_HTTP_${status}`, `${url}: transient source failure`);
    } else throw stageError(`H3_STAGE_HTTP_${status}`, `${url}: unexpected source status`);

    let total = append ? existing : 0;
    const handle = await open(partPath, append ? 'a' : 'w');
    try {
      for await (const chunk of bodyChunks(response)) {
        if (signal?.aborted) throw stageError('H3_STAGE_INTERRUPTED', 'staging interrupted before completion');
        if (total + chunk.length > entry.expectedBytes) throw stageError('H3_STAGE_OVERSIZE', `${url}: stream exceeded the frozen expected size; aborted`);
        const written = await writeAllBytes(handle, chunk, { signal, onWritten: async bytesWritten => {
          total += bytesWritten;
          if (total > entry.expectedBytes) throw stageError('H3_STAGE_OVERSIZE', `${url}: stream exceeded the frozen expected size; aborted`);
          await onChunk?.(total);
        } });
        assert.equal(written, chunk.length, 'chunk was not fully written');
      }
      await handle.sync();
    } finally { await handle.close(); }
    if (total === 0) throw stageError('H3_STAGE_ZERO_BYTES', `${url}: zero-byte transfer is not completion`);
    const actualBytes = (await stat(partPath)).size;
    if (actualBytes !== total) throw stageError('H3_STAGE_SIZE', `${url}: staged size changed during transfer`);
    if (actualBytes !== entry.expectedBytes) {
      if (attempt < maxRetries && actualBytes < entry.expectedBytes) { await sleepUnlessAborted(sleep, Math.min(1000 * 2 ** attempt, 15000), signal); continue; }
      throw stageError('H3_STAGE_SIZE', `${url}: staged ${actualBytes} bytes, expected ${entry.expectedBytes}`);
    }
    return { url, actualBytes };
  }
}
export async function stageOneFile({ entry, root, acknowledgment = null, fetchImpl = globalThis.fetch, env = process.env, platform = process.platform, hub, fileTimeoutMs = FILE_TIMEOUT_MS, maxRetries = 3, sleep = ms => new Promise(resolve => setTimeout(resolve, ms)), signal, onProgress, now = Date.now, jobDeadline }) {
  const directory = snapshotDir(root, entry), finalPath = path.join(directory, ...fileSegments(entry.file)), partPath = `${finalPath}.part`;
  assert.ok(within(root, finalPath), 'staging target escaped the private root');
  await ensureOwnedDir(root, path.dirname(finalPath));
  const startedAt = new Date(now()).toISOString();
  const record = { repoType: entry.repoType, repo: entry.repo, revision: entry.revision, file: entry.file, url: sourceUrl(entry), expectedBytes: entry.expectedBytes, expectedSha256: entry.expectedSha256, status: 'pending', startedAt, endedAt: null, durationMs: null, actualBytes: 0, observedSha256: null, finalPath, partPath, expectedLegal: entry.legal ?? null, explicitAcknowledgment: acknowledgment, cache: { hub: null, state: 'unknown', mode: null, sourcePath: null } };
  try {
    const existing = await lstatOrUndefined(finalPath);
    if (existing) {
      assert.ok(!existing.isSymbolicLink() && existing.isFile(), 'verified staging target must be a regular file');
      const actualBytes = existing.size;
      const observedSha256 = await hashFileSha256(finalPath, { signal, deadline: jobDeadline, now });
      assert.ok(actualBytes === entry.expectedBytes, `existing staged file has ${actualBytes} bytes, expected ${entry.expectedBytes}`);
      assert.equal(observedSha256, entry.expectedSha256, 'existing staged file content differs from the frozen full SHA256');
      record.cache.state = 'already-staged';
      return { ...record, status: 'completed', actualBytes, observedSha256, endedAt: new Date(now()).toISOString(), durationMs: now() - Date.parse(startedAt), reused: true };
    }
    // Prefer an existing builtin cache: exact-file probe only, full size+SHA verification, then a
    // private COPY (never a hardlink into shared HF blobs). A present-but-wrong cache is a blocker.
    const hubPath = hub ?? cacheHub({ platform, env });
    const probe = await probeCacheFile(entry, hubPath, { signal });
    if (probe.state === 'present') {
      record.cache = { hub: hubPath, state: 'present', mode: 'cache-copy', sourcePath: probe.sourcePath };
      if (probe.size !== entry.expectedBytes) { record.cache.state = 'mismatch'; record.cache.observedBytes = probe.size; throw stageError('H3_STAGE_CACHE_MISMATCH', `${probe.sourcePath}: cached size ${probe.size}, expected ${entry.expectedBytes}; explicit blocker, no fresh fallback`); }
      const cacheSha256 = await hashFileSha256(probe.sourcePath, { signal, deadline: jobDeadline, now });
      if (cacheSha256 !== entry.expectedSha256) { record.cache.state = 'mismatch'; record.cache.observedSha256 = cacheSha256; throw stageError('H3_STAGE_CACHE_MISMATCH', `${probe.sourcePath}: cached ${cacheSha256}, expected ${entry.expectedSha256}; explicit blocker, no fresh fallback`); }
      await ensureOwnedFileTarget(partPath);
      await copyFile(probe.sourcePath, partPath);
      const actualBytes = (await stat(partPath)).size;
      const observedSha256 = await hashFileSha256(partPath, { signal, deadline: jobDeadline, now });
      record.actualBytes = actualBytes; record.observedSha256 = observedSha256;
      if (actualBytes !== entry.expectedBytes || observedSha256 !== entry.expectedSha256) { record.cache.state = 'mismatch'; throw stageError('H3_STAGE_CACHE_COPY_MISMATCH', `${probe.sourcePath}: private copy ${actualBytes}/${observedSha256} differs from the frozen expected bytes/SHA`); }
      await rename(partPath, finalPath);
      const info = await lstatOrUndefined(finalPath);
      assert.ok(info && info.isFile() && !info.isSymbolicLink() && info.size === entry.expectedBytes, 'verified staged copy is not the canonical regular file');
      return { ...record, status: 'completed', actualBytes, observedSha256, endedAt: new Date(now()).toISOString(), durationMs: now() - Date.parse(startedAt) };
    }
    record.cache = { hub: hubPath, state: 'absent', mode: 'fresh-download', sourcePath: null };
    await ensureOwnedFileTarget(partPath);
    const { url, actualBytes } = await downloadToPart({ entry, partPath, fetchImpl, fileTimeoutMs, maxRetries, sleep, signal, deadline: jobDeadline, now, onChunk: async total => {
      record.actualBytes = total;
      if (jobDeadline !== undefined && now() > jobDeadline) throw stageError('H3_STAGE_INTERRUPTED', 'job deadline exceeded during transfer');
      await onProgress?.({ entry, actualBytes: total, jobDeadline });
    } });
    if (jobDeadline !== undefined) assert.ok(now() <= jobDeadline, 'staging job bound exceeded');
    // Independent post-download content identity: metadata is not proof.
    const observedSha256 = await hashFileSha256(partPath, { signal, deadline: jobDeadline, now });
    record.actualBytes = actualBytes;
    record.observedSha256 = observedSha256;
    if (observedSha256 !== entry.expectedSha256) throw stageError('H3_STAGE_SHA256', `${url}: observed ${observedSha256}, expected ${entry.expectedSha256}; partial retained`);
    await rename(partPath, finalPath);
    const info = await lstatOrUndefined(finalPath);
    assert.ok(info && info.isFile() && !info.isSymbolicLink() && info.size === entry.expectedBytes, 'verified staged file is not the canonical regular file');
    return { ...record, status: 'completed', actualBytes, observedSha256, endedAt: new Date(now()).toISOString(), durationMs: now() - Date.parse(startedAt) };
  } catch (error) {
    // Last durable metadata for a hard timeout/interruption: the retained partial's actual bytes
    // and partial content hash, never a fabricated completion.
    try {
      const partial = await lstatOrUndefined(partPath);
      if (partial?.isFile() && !partial.isSymbolicLink()) { record.actualBytes = partial.size; record.partialSha256 = await hashFileSha256(partPath); }
    } catch { /* a broken partial must not hide the original failure */ }
    record.status = (error.code === 'H3_STAGE_INTERRUPTED' || signal?.aborted) ? 'interrupted' : 'failed';
    record.error = { code: error.code ?? 'H3_STAGE_FILE', message: error.message };
    record.endedAt = new Date(now()).toISOString();
    record.durationMs = now() - Date.parse(startedAt);
    throw Object.assign(error, { record });
  }
}
// Read-only inspection of the known cache for the selected manifest. Never downloads and never
// manufactures legal approval.
export async function inspectManifest({ files, env = process.env, platform = process.platform, signal, now = Date.now }) {
  const hub = cacheHub({ platform, env }), rows = [];
  for (const entry of files) {
    if (signal?.aborted) throw stageError('H3_STAGE_INTERRUPTED', 'inspection interrupted');
    const probe = await probeCacheFile(entry, hub, { signal });
    if (probe.state === 'absent') { rows.push({ repoType: entry.repoType, repo: entry.repo, revision: entry.revision, file: entry.file, expectedBytes: entry.expectedBytes, expectedSha256: entry.expectedSha256, presence: 'missing', cache: { state: 'absent', hub }, actualBytes: null, observedSha256: null, matches: false, blocker: null, expectedLegal: entry.legal ?? null }); continue; }
    const observedSha256 = await hashFileSha256(probe.sourcePath, { signal, now });
    const matches = probe.size === entry.expectedBytes && observedSha256 === entry.expectedSha256;
    rows.push({ repoType: entry.repoType, repo: entry.repo, revision: entry.revision, file: entry.file, expectedBytes: entry.expectedBytes, expectedSha256: entry.expectedSha256, presence: 'present', cache: { state: 'present', hub, sourcePath: probe.sourcePath }, actualBytes: probe.size, observedSha256, matches, blocker: matches ? null : 'H3_STAGE_CACHE_MISMATCH', expectedLegal: entry.legal ?? null });
  }
  return { hub, files: rows, present: rows.filter(row => row.presence === 'present').length, missing: rows.filter(row => row.presence === 'missing').length, mismatched: rows.filter(row => row.blocker).length };
}
// Metadata-only device identity. No CUDA/Metal context is initialized; absence is recorded, never
// promoted into a staging gate.
export async function defaultProbe(backend, execute = exec) {
  if (backend === 'mlx') {
    const { stdout } = await execute('system_profiler', ['SPDisplaysDataType', '-json'], { timeout: 30000 });
    let device = null;
    try { const rows = JSON.parse(stdout).SPDisplaysDataType; device = rows?.[0]?.sppci_model ?? rows?.[0]?._name ?? null; } catch { /* keep null */ }
    const { stdout: version } = await execute('sw_vers', ['-productVersion'], { timeout: 30000 });
    return { kind: 'system_profiler-metadata-only', device, os: { productVersion: version.trim() }, stdoutSha256: createHash('sha256').update(stdout).digest('hex'), meaning: 'Metadata probe only; actual GPU model/OS recorded, no GPU context initialized' };
  }
  const { stdout } = await execute('nvidia-smi', ['--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total', '--format=csv,noheader'], { timeout: 30000 });
  return { kind: 'nvidia-smi-metadata-only', rows: stdout.trim().split('\n').filter(Boolean), meaning: 'Metadata probe only with UUID/PCI identity; no CUDA context initialized' };
}
export async function runStage({ request, backend, book, env = process.env, platform = process.platform, fetchImpl, probe, evidenceDir, codeSha, allowUnitFake = false, now = Date.now, sleep = ms => new Promise(resolve => setTimeout(resolve, ms)), signal, fileTimeoutMs = FILE_TIMEOUT_MS, jobBoundMs = JOB_BOUND_MS, maxRetries = 3 }) {
  const parsed = parseStageRequest(request);
  const selectedBackend = backend ?? parsed.backend;
  assert.ok(STAGE_BACKENDS.includes(selectedBackend), 'backend must be mlx or cuda');
  // Real function entry enforces the hosted manual guard itself, BEFORE any filesystem/network
  // action. unit_fake is only reachable through an explicitly injected transport.
  if (!allowUnitFake) assertCliEnvironment({ env, platform, backend: selectedBackend });
  else assert.equal(typeof fetchImpl, 'function', 'unit_fake staging requires an injected fake transport');
  assert.ok(allowUnitFake || book === undefined, 'real staging cannot inject book data; the receipt casebook SHA is the source casebook');
  const bookValue = book ?? await loadCasebook();
  const files = selectStageFiles(bookValue, parsed.assetSets);
  assert.ok(files.length > 0, 'at least one frozen file must be selected');
  const inspectOnly = parsed.inspect_only === true || parsed.inspectOnly === true;
  const authority = parsed.license_authority ?? parsed.licenseAuthority;
  if (!inspectOnly) assertStageAuthority(authority, files, { allowUnitFake });
  const acknowledgments = inspectOnly ? new Map() : fileAcknowledgments(authority, files);
  const transport = fetchImpl ?? fetch;
  const producerCodeSha = codeSha ?? env.GITHUB_SHA;
  sha40(producerCodeSha);
  const runId = env.GITHUB_RUN_ID, runAttempt = env.GITHUB_RUN_ATTEMPT;
  assert.ok(path.isAbsolute(evidenceDir), 'H3_STAGE_EVIDENCE absolute runner-temporary path required');
  if (inspectOnly) {
    // Read-only known-cache inspection: no downloads, no private root, no fabricated approval.
    await mkdir(evidenceDir, { recursive: true });
    const startedAt = new Date(now()).toISOString();
    const report = await inspectManifest({ files, env, platform, signal, now });
    const receipt = {
      version: 1, kind: 'h3-finetune-stage-inspection-receipt', proof_kind: allowUnitFake ? 'unit_fake' : 'inspect_only',
      campaign_proof: false, stagingComplete: false,
      producer: { codeSha: producerCodeSha, script: 'scripts/h3-finetune-terminal-stage.mjs', runId, runAttempt, workflow: env.GITHUB_WORKFLOW ?? null },
      source: { casebook: CASEBOOK_RELATIVE, casebookSha256: await casebookSha256(), hfHost: HF_HOST },
      set: { ids: parsed.assetSets, files: files.map(entry => ({ repoType: entry.repoType, repo: entry.repo, revision: entry.revision, file: entry.file, expectedBytes: entry.expectedBytes, expectedSha256: entry.expectedSha256, legal: entry.legal ?? null })) },
      settings: { backend: selectedBackend, platform, cacheRoot: report.hub },
      runner: { runnerName: env.RUNNER_NAME ?? null, arch: process.arch, platform, githubActions: env.GITHUB_ACTIONS === 'true' },
      startedAt, endedAt: new Date(now()).toISOString(), status: 'inspected',
      files: report.files, summary: { present: report.present, missing: report.missing, mismatched: report.mismatched },
    };
    receipt.durationMs = now() - Date.parse(startedAt);
    const target = path.join(evidenceDir, `inspect-receipt-${runId}-${runAttempt}.json`);
    await writeFile(target, JSON.stringify(receipt, null, 2) + '\n');
    await appendEvidenceLog(evidenceDir, `${receipt.endedAt} inspected present=${report.present} missing=${report.missing} mismatched=${report.mismatched}`);
    if (env.GITHUB_OUTPUT) await appendFile(env.GITHUB_OUTPUT, `status=inspected\npresent=${report.present}\nmissing=${report.missing}\nmismatched=${report.mismatched}\n`);
    return receipt;
  }
  const cacheOnly=(bookValue.stagingPlan?.cacheProbeFirst??[]).filter(id=>parsed.assetSets.includes(id)).flatMap(id=>selectStageFiles(bookValue,[id]));
  if(cacheOnly.length){const observed=await inspectManifest({files:cacheOnly,env,platform,signal,now});await mkdir(evidenceDir,{recursive:true});await writeFile(path.join(evidenceDir,'large-builtin-cache-preflight.json'),JSON.stringify(observed,null,2)+'\n');assert.equal(observed.mismatched,0,'H3_STAGE_CACHE_MISMATCH: selected raw builtin cache differs from pinned bytes; do not silently repair');assert.ok(observed.missing===0||parsed.allow_fresh_builtin_download===true,'H3_STAGE_CACHE_REQUIRED: raw builtin source absent; inspect first, then explicitly authorize a fresh large download rather than assuming it');}
  const root = privateRunRoot({ backend: selectedBackend, platform, env, runId, attempt: runAttempt });
  assert.ok(path.isAbsolute(root), 'private staging root must be absolute');
  await assertOwnedEvidence(evidenceDir, root);

  const existingRoot = await lstatOrUndefined(root);
  assert.equal(existingRoot, undefined, 'refusing to reuse an existing private staging root; use the run/attempt root');
  await mkdir(root, { recursive: true });
  const rootInfo = await lstat(root);
  assert.ok(rootInfo.isDirectory() && !rootInfo.isSymbolicLink(), 'private staging root must be a real directory');
  const owner = { kind: 'h3-finetune-stage-owner', root: path.resolve(root), runId, runAttempt, backend: selectedBackend, codeSha: producerCodeSha, createdAt: new Date(now()).toISOString() };
  await writeFile(path.join(root, MARKER_FILE), JSON.stringify(owner, null, 2) + '\n', { flag: 'wx' });

  let device;
  try { device = await (probe ?? (() => defaultProbe(selectedBackend)))(selectedBackend); }
  catch (error) { device = { available: false, error: error.message, meaning: 'Metadata probe unavailable; no GPU context initialized' }; }

  const jobDeadline = now() + jobBoundMs;
  const receipt = {
    version: 1,
    kind: 'h3-finetune-stage-receipt',
    proof_kind: allowUnitFake ? 'unit_fake' : 'frozen_stage_content',
    campaign_proof: false,
    producer: { codeSha: producerCodeSha, script: 'scripts/h3-finetune-terminal-stage.mjs', runId, runAttempt, workflow: env.GITHUB_WORKFLOW ?? null },
    source: { casebook: CASEBOOK_RELATIVE, casebookSha256: await casebookSha256(), hfHost: HF_HOST },
    pin: { consumerPin: env.H3_STAGE_CONSUMER_PIN ?? null, meaning: 'Informational current-consumer pin only; this is not a final native source pair' },
    set: { ids: parsed.assetSets, files: files.map(entry => ({ repoType: entry.repoType, repo: entry.repo, revision: entry.revision, file: entry.file, expectedBytes: entry.expectedBytes, expectedSha256: entry.expectedSha256, legal: entry.legal ?? null })) },
    settings: { backend: selectedBackend, platform, privateRoot: root, cacheRoot: cacheHub({ platform, env }), fileTimeoutMs, jobBoundMs, maxRetries },
    runner: { runnerName: env.RUNNER_NAME ?? null, arch: process.arch, platform, githubActions: env.GITHUB_ACTIONS === 'true' },
    device,
    licenseAuthority: { kind: authority.kind, approvedBy: authority.approvedBy, approvedAt: authority.approvedAt, reference: authority.reference, subjects: (authority.subjects ?? []).map(knownSubject), acknowledged: authority.acknowledged },
    startedAt: new Date(now()).toISOString(),
    endedAt: null,
    durationMs: null,
    status: 'running',
    files: files.map(entry => ({ repo: entry.repo, revision: entry.revision, file: entry.file, url: sourceUrl(entry), expectedBytes: entry.expectedBytes, expectedSha256: entry.expectedSha256, status: 'pending', startedAt: null, endedAt: null, durationMs: null, actualBytes: 0, observedSha256: null, finalPath: null, partPath: null })),
  };
  // Metadata receipt exists BEFORE any file transfer.
  await persistReceipt(root, evidenceDir, receipt);
  await appendEvidenceLog(evidenceDir, `${receipt.startedAt} start backend=${selectedBackend} files=${files.length}`);

  let failure;
  try {
    for (const [index, entry] of files.entries()) {
      if (signal?.aborted) throw stageError('H3_STAGE_INTERRUPTED', 'staging interrupted before completion');
      assert.ok(now() <= jobDeadline, 'staging job bound exceeded');
      const pending = receipt.files[index];
      pending.status = 'running'; pending.startedAt = new Date(now()).toISOString();
      await persistReceipt(root, evidenceDir, receipt);
      let progressWrite = 0;
      try {
        const completed = await stageOneFile({
          entry, root, acknowledgment: acknowledgments.get(assetKey(entry)) ?? null, fetchImpl: transport, env, platform, hub: receipt.settings.cacheRoot, fileTimeoutMs, maxRetries, sleep, signal, now, jobDeadline,
          onProgress: async ({ actualBytes }) => {
            pending.actualBytes = actualBytes;
            // Throttled durable per-file progress so a hard timeout keeps meaningful partial metadata.
            if (actualBytes - progressWrite >= 16 * 1024 * 1024) { progressWrite = actualBytes; await persistReceipt(root, evidenceDir, receipt); }
          },
        });
        Object.assign(pending, completed);
      } catch (error) {
        Object.assign(pending, error.record ?? {});
        throw error;
      }
      await persistReceipt(root, evidenceDir, receipt);
      await appendEvidenceLog(evidenceDir, `${pending.endedAt} completed ${entry.repo}@${entry.revision}/${entry.file} bytes=${pending.actualBytes}`);
    }
    for (const row of receipt.files) assert.ok(row.status === 'completed' && row.actualBytes === row.expectedBytes && row.observedSha256 === row.expectedSha256, 'every selected file requires positive validated completion');
    if (parsed.prepare_identity_controls === true) {
      // Opt-in derived fixtures from the already-staged smallest canonical source; no extra fetch.
      try {
        receipt.identityControls = await prepareIdentityControls({ root, entry: identityControlSourceEntry(bookValue), authority, evidenceDir, runId, runAttempt, allowUnitFake, signal, now });
        receipt.identityControlsSummary = { hfFiles: receipt.files.length, derivedFixtures: receipt.identityControls.derivedFixtures };
      } catch (error) {
        receipt.identityControls = error.identityControls ?? null;
        throw error;
      }
    }
    receipt.status = 'completed';
    receipt.endedAt = new Date(now()).toISOString();
    receipt.durationMs = now() - Date.parse(receipt.startedAt);
    await persistReceipt(root, evidenceDir, receipt);
    await appendEvidenceLog(evidenceDir, `${receipt.endedAt} completed all ${files.length} files`);
    if (env.GITHUB_OUTPUT) await appendFile(env.GITHUB_OUTPUT, `root=${root}\nstatus=completed\nproducer_code_sha=${producerCodeSha}\n`);
    return receipt;
  } catch (error) {
    failure = error;
    receipt.status = error.code === 'H3_STAGE_INTERRUPTED' || signal?.aborted ? 'interrupted' : 'failed';
    receipt.endedAt = new Date(now()).toISOString();
    receipt.durationMs = now() - Date.parse(receipt.startedAt);
    receipt.error = { code: error.code ?? 'H3_STAGE', message: error.message };
    await persistReceipt(root, evidenceDir, receipt);
    await appendEvidenceLog(evidenceDir, `${receipt.endedAt} ${receipt.status} ${receipt.error.code}: ${receipt.error.message}`);
    if (env.GITHUB_OUTPUT) await appendFile(env.GITHUB_OUTPUT, `root=${root}\nstatus=${receipt.status}\nproducer_code_sha=${producerCodeSha}\n`);
    throw failure;
  }
}
// Opt-in request.prepare_identity_controls===true: REAL same-byte copied fixture under a different
// name/path plus a one-bit-payload negative control, derived from the SMALLEST canonical
// Singularity file (W4A8) to avoid copying 34GB. COPYFILE_FICLONE attempts a CoW clone and falls
// back to a real copy; the shared source is never modified. This is prepared-not-modelproof data,
// not a new training run and not a new consent.
export function identityControlSourceEntry(book) {
  return [...canonicalSingularityFiles(book)].sort((a, b) => a.expectedBytes - b.expectedBytes)[0];
}
const runIdValue = value => (value === undefined || value === null ? '' : String(value));
const runAttemptValue = value => (value === undefined || value === null ? '' : String(value));
async function writeControlsReceipt({ root, evidenceDir, receipt }) {
  const body = JSON.stringify(receipt, null, 2) + '\n';
  if (evidenceDir) {
    await mkdir(evidenceDir, { recursive: true });
    const evidenceTarget = path.join(evidenceDir, 'identity-controls.json');
    await writeFile(`${evidenceTarget}.tmp`, body);
    await rename(`${evidenceTarget}.tmp`, evidenceTarget);
  }
  const rootTarget = path.join(root, 'fixtures', 'identity-controls.json');
  await mkdir(path.dirname(rootTarget), { recursive: true });
  await writeFile(`${rootTarget}.tmp`, body);
  await rename(`${rootTarget}.tmp`, rootTarget);
}
// Bounded safetensors header parse: first 8 bytes are the little-endian header length, then JSON
// metadata only. Choose a byte INSIDE an I8/U8 `.weight` tensor payload, never `.comfy_quant`,
// scale/config metadata, or a trailing descriptor, so the header/shape stays valid while the full
// SHA changes. No model/tensor execution is performed.
export const SAFETENSORS_HEADER_LIMIT = 100 * 1024 * 1024;
export async function quantizedWeightTarget(filePath, fileSize, { signal } = {}) {
  const handle = await open(filePath, 'r');
  try {
    const lengthBytes = Buffer.alloc(8);
    assert.equal((await handle.read(lengthBytes, 0, 8, 0)).bytesRead, 8, 'safetensors header length is unreadable');
    const headerLength = Number(lengthBytes.readBigUInt64LE(0));
    assert.ok(Number.isSafeInteger(headerLength) && headerLength > 0 && headerLength <= SAFETENSORS_HEADER_LIMIT, `safetensors header length ${headerLength} is outside the bounded limit`);
    assert.ok(8 + headerLength <= fileSize, 'safetensors header extends past the file');
    const headerBytes = Buffer.alloc(headerLength);
    assert.equal((await handle.read(headerBytes, 0, headerLength, 8)).bytesRead, headerLength, 'safetensors header is unreadable');
    const header = JSON.parse(headerBytes.toString('utf8'));
    assert.ok(header && typeof header === 'object' && !Array.isArray(header), 'safetensors header must be a JSON object');
    const excluded = name => name === '__metadata__' || /comfy_quant|scales|config/i.test(name);
    const candidates = Object.entries(header).filter(([name, meta]) => {
      if (excluded(name) || !name.endsWith('.weight')) return false;
      if (!meta || !['I8', 'U8'].includes(meta.dtype)) return false;
      const offsets = meta.data_offsets;
      return Array.isArray(offsets) && offsets.length === 2 && Number.isInteger(offsets[0]) && Number.isInteger(offsets[1]) && offsets[0] >= 0 && offsets[0] < offsets[1] && offsets[1] <= fileSize - 8 - headerLength;
    });
    assert.ok(candidates.length > 0, 'no quantized .weight I8/U8 tensor found in the bounded safetensors header');
    candidates.sort(([left], [right]) => left.localeCompare(right));
    const [name, meta] = candidates[0], [begin, end] = meta.data_offsets;
    const relativeIndex = begin + Math.floor((end - begin) / 2);
    return { kind: 'quantized-weight-byte', tensor: name, dtype: meta.dtype, data_offsets: [begin, end], relativeIndex, offset: 8 + headerLength + relativeIndex, headerBytes: 8 + headerLength };
  } finally { await handle.close(); }
}
export async function prepareIdentityControls({ root, entry, authority, evidenceDir, runId, runAttempt, allowUnitFake = false, fixtureOffset, signal, now = Date.now, nowIso = () => new Date(now()).toISOString() }) {
  const renamedPath = path.join(root, 'fixtures', 'independent-layout', 'unrelated-name.safetensors');
  const differentBytesPath = path.join(root, 'fixtures', 'independent-layout', 'different-bytes.safetensors');
  const sourcePath = path.join(root, 'hf', 'hub', cacheRepoDir(entry), 'snapshots', entry.revision, ...fileSegments(entry.file));
  const legal = entry.legal ?? null;
  const receipt = {
    version: 1, kind: 'h3-finetune-stage-identity-controls', status: 'failed',
    proof_kind: allowUnitFake ? 'unit_fake' : 'campaign_stage_derived',
    meaning: 'prepared-not-modelproof derived identity controls; no model execution and no new approval',
    runId: runIdValue(runId), runAttempt: runAttemptValue(runAttempt),
    sourcePath, sourceExpectedSHA: entry.expectedSha256, sourceSizeBytes: entry.expectedBytes,
    source: { repoType: entry.repoType, repo: entry.repo, revision: entry.revision, file: entry.file },
    lineage: 'same-byte renamed copy plus one-payload-byte negative control derived from the smallest frozen Singularity source; no new training',
    renamed: null, differentBytes: null, derivedFixtures: 2,
    derivedLicense: 'the same original explicit legal acknowledgment covers derived data; no new consent is synthesized for the negative hash',
    campaignPlanRequirement: 'a later campaign import still needs actual explicit input subjects for every imported file',
    originalExplicitAckReference: { reference: authority?.reference ?? null, approvedBy: authority?.approvedBy ?? null, approvedAt: authority?.approvedAt ?? null, licenseId: legal?.licenseId ?? null, inheritedLicenseId: legal?.inheritedLicenseId ?? null, attributionRequired: legal?.attributionRequired ?? null },
    startedAt: nowIso(), endedAt: null,
  };
  try {
    assert.ok(!(fixtureOffset !== undefined && !allowUnitFake), 'an injected fixtureOffset is unit_fake only; production never uses an arbitrary offset');
    assert.ok(within(root, sourcePath), 'identity-control source must stay below the private root');
    const sourceInfo = await lstatOrUndefined(sourcePath);
    assert.ok(sourceInfo?.isFile() && !sourceInfo.isSymbolicLink(), `missing verified private source for identity controls: ${sourcePath}`);
    assert.equal(sourceInfo.size, entry.expectedBytes, 'identity-control source size differs from the frozen expected size');
    assert.equal(await hashFileSha256(sourcePath, { signal, now }), entry.expectedSha256, 'identity-control source differs from the frozen full SHA256');
    await ensureOwnedDir(root, path.join(root, 'fixtures', 'independent-layout'));
    for (const target of [renamedPath, differentBytesPath]) assert.equal(await ensureOwnedFileTarget(target), 'absent', `refusing to overwrite an existing identity-control target: ${target}`);
    await copyFile(sourcePath, renamedPath, fsConstants.COPYFILE_FICLONE | fsConstants.COPYFILE_EXCL);
    await ensureOwnedFileTarget(renamedPath);
    const renamed = { path: renamedPath, size: (await stat(renamedPath)).size };
    assert.equal(renamed.size, entry.expectedBytes, 'renamed identity-control copy size differs from the source');
    renamed.actualSHA = await hashFileSha256(renamedPath, { signal, now });
    assert.equal(renamed.actualSHA, entry.expectedSha256, 'renamed identity-control copy content differs from the source');
    receipt.renamed = renamed;
    await copyFile(sourcePath, differentBytesPath, fsConstants.COPYFILE_FICLONE | fsConstants.COPYFILE_EXCL);
    const negativeSize = (await stat(differentBytesPath)).size;
    assert.equal(negativeSize, entry.expectedBytes, 'negative identity-control copy size differs from the source');
    let target;
    if (fixtureOffset !== undefined) {
      assert.ok(allowUnitFake, 'an injected fixtureOffset is unit_fake only; production never uses an arbitrary offset');
      assert.ok(Number.isSafeInteger(fixtureOffset) && fixtureOffset >= 0 && fixtureOffset < entry.expectedBytes, 'fixtureOffset lies outside the source bytes');
      target = { kind: 'unit_fake-injected-offset', tensor: null, dtype: null, relativeIndex: null, headerBytes: 0, offset: fixtureOffset };
    } else {
      target = await quantizedWeightTarget(sourcePath, entry.expectedBytes, { signal });
    }
    const offset = target.offset;
    const handle = await open(differentBytesPath, 'r+');
    let before, after;
    try {
      const buffer = Buffer.alloc(1);
      assert.equal((await handle.read(buffer, 0, 1, offset)).bytesRead, 1, 'negative identity-control payload byte is unreadable');
      before = buffer[0]; after = before ^ 1;
      assert.equal((await handle.write(Buffer.from([after]), 0, 1, offset)).bytesWritten, 1, 'negative identity-control payload byte is unwritable');
      await handle.sync();
    } finally { await handle.close(); }
    if (target.kind === 'quantized-weight-byte') {
      // Same header/descriptor/shape must remain byte-identical; only the weight payload byte flips.
      const headerLength = target.headerBytes, sourceHeader = Buffer.alloc(headerLength), mutatedHeader = Buffer.alloc(headerLength);
      const sourceHandle = await open(sourcePath, 'r'), negativeHandle = await open(differentBytesPath, 'r');
      try {
        assert.equal((await sourceHandle.read(sourceHeader, 0, headerLength, 0)).bytesRead, headerLength, 'source header re-read failed');
        assert.equal((await negativeHandle.read(mutatedHeader, 0, headerLength, 0)).bytesRead, headerLength, 'negative header read failed');
      } finally { await sourceHandle.close(); await negativeHandle.close(); }
      assert.deepEqual(mutatedHeader, sourceHeader, 'negative control changed the safetensors header/descriptor');
    }
    const negativeSha = await hashFileSha256(differentBytesPath, { signal, now });
    assert.notEqual(negativeSha, entry.expectedSha256, 'negative identity control must differ from the frozen source SHA256');
    receipt.differentBytes = { path: differentBytesPath, size: negativeSize, actualSHA: negativeSha, expectedOriginalSHA: entry.expectedSha256, mutation: { ...target, offset, before, after } };
    const rehash = await hashFileSha256(sourcePath, { signal, now });
    assert.equal(rehash, entry.expectedSha256, 'shared identity-control source changed during preparation');
    receipt.sourceActualSHA = rehash;
    receipt.status = 'prepared-not-modelproof';
    receipt.endedAt = nowIso();
    await writeControlsReceipt({ root, evidenceDir, receipt });
    return receipt;
  } catch (error) {
    receipt.status = signal?.aborted || error.code === 'H3_STAGE_INTERRUPTED' ? 'interrupted' : 'failed';
    receipt.error = { code: error.code ?? 'H3_STAGE_IDENTITY_CONTROLS', message: error.message };
    receipt.endedAt = nowIso();
    try { await writeControlsReceipt({ root, evidenceDir, receipt }); } catch { /* keep the original failure */ }
    throw Object.assign(error, { identityControls: receipt });
  }
}
// ---------------------------------------------------------------------------
// CC0 conditioning reference fixtures: run ONLY the published casebook ffmpeg recipes
// (imageA/imageB/clipA/clipB/tone) into root/conditioning/cc0. This is authored ffmpeg math, not
// model output. FFmpeg/ffprobe must resolve from the trusted host PATH outside the HF stage root;
// cwd is always the new output directory. Speech is a separate helper.
// ---------------------------------------------------------------------------
export const REFERENCE_FIXTURE_RECIPES = Object.freeze([
  { id: 'image_a', assetKey: 'imageA', kind: 'image', role: 'image' },
  { id: 'image_b', assetKey: 'imageB', kind: 'image', role: 'image' },
  { id: 'clip_a', assetKey: 'clipA', kind: 'video', role: 'clip' },
  { id: 'clip_b', assetKey: 'clipB', kind: 'video', role: 'clip' },
  { id: 'tone', assetKey: 'tone', kind: 'audio', role: 'tone' },
]);
async function resolveHostTool(name, { env, platform, root }) {
  const pathValue = platform === 'win32' ? (env.PATH ?? env.Path ?? '') : (env.PATH ?? '');
  const suffix = platform === 'win32' ? '.exe' : '';
  for (const directory of String(pathValue).split(path.delimiter)) {
    if (!directory) continue;
    const candidate = path.resolve(directory, `${name}${suffix}`);
    if (within(root, candidate)) continue; // never execute a tool from the weight stage root
    try { await access(candidate, fsConstants.X_OK); } catch { continue; }
    const real = await realpath(candidate);
    assert.ok(!within(root, real), `${name} must resolve outside the HF stage root; refusing a staged tool`);
    return real;
  }
  throw stageError(`H3_STAGE_${name.toUpperCase()}_MISSING`, `${name} was not found on the trusted host PATH`);
}
function cleanToolEnv(env) {
  return Object.fromEntries(['PATH', 'Path', 'SystemRoot', 'WINDIR', 'COMSPEC', 'PATHEXT', 'TMPDIR', 'TEMP', 'TMP'].filter(key => env[key] !== undefined).map(key => [key, env[key]]));
}
async function runTool(argv, { execute, cwd, env, timeoutMs }) {
  try {
    const result = await execute(argv[0], argv.slice(1), { cwd, env, timeout: timeoutMs });
    return { code: typeof result?.code === 'number' ? result.code : 0, stdout: result?.stdout ?? '', stderr: result?.stderr ?? '' };
  } catch (error) {
    return { code: typeof error?.code === 'number' ? error.code : 1, stdout: error?.stdout ?? '', stderr: error?.stderr ?? error?.message ?? '' };
  }
}
function probeArgv(ffprobe, kind, outputName) {
  const entries = kind === 'image'
    ? 'stream=codec_name,width,height,nb_read_frames'
    : kind === 'video' ? 'stream=codec_name,width,height,nb_read_frames,duration,avg_frame_rate,r_frame_rate' : 'stream=codec_name,sample_rate,channels,duration,nb_read_frames';
  return [ffprobe, '-v', 'error', '-select_streams', kind === 'audio' ? 'a:0' : 'v:0', '-count_frames', '-show_entries', entries, '-of', 'json', outputName];
}
function decodeArgv(ffmpeg, kind, outputName) {
  return kind === 'video'
    ? [ffmpeg, '-nostdin', '-v', 'error', '-xerror', '-i', outputName, '-map', '0:v:0', '-map', '0:a:0', '-f', 'null', '-']
    : [ffmpeg, '-nostdin', '-v', 'error', '-xerror', '-i', outputName, '-f', 'null', '-'];
}
async function writeReferenceReceipt({ root, evidenceDir, receipt }) {
  const body = JSON.stringify(receipt, null, 2) + '\n';
  if (evidenceDir) { await mkdir(evidenceDir, { recursive: true }); const target = path.join(evidenceDir, 'reference-fixtures.json'); await writeFile(`${target}.tmp`, body); await rename(`${target}.tmp`, target); }
  const rootTarget = path.join(root, 'conditioning', 'reference-fixtures.json');
  await mkdir(path.dirname(rootTarget), { recursive: true });
  await writeFile(`${rootTarget}.tmp`, body); await rename(`${rootTarget}.tmp`, rootTarget);
}
export async function prepareReferenceFixtures({ root, evidenceDir, book, backend, execute, tools, allowUnitFake = false, env = process.env, platform = process.platform, now = Date.now, signal, timeoutMs = 120000 }) {
  if (!allowUnitFake) assertCliEnvironment({ env, platform, backend: backend ?? env.H3_STAGE_BACKEND });
  assert.ok(allowUnitFake || book === undefined, 'real reference-fixture preparation cannot inject book data; use the published casebook');
  assert.ok(allowUnitFake || (execute === undefined && tools === undefined), 'injected execute/tools are unit_fake only; production resolves FFmpeg from the trusted host PATH');
  if (allowUnitFake) assert.equal(typeof execute, 'function', 'unit_fake reference fixtures require an injected execute');
  const casebook = book ?? await loadCasebook();
  if (allowUnitFake) assert.ok(casebook.assets && typeof casebook.assets === 'object', 'unit_fake book must carry the published asset recipes');
  assert.ok(typeof root === 'string' && path.isAbsolute(root), 'absolute private stage root required');
  assert.ok(typeof evidenceDir === 'string' && path.isAbsolute(evidenceDir), 'absolute evidence directory required');
  await assertOwnedEvidence(evidenceDir, root);
  const rootInfo = await lstatOrUndefined(root);
  assert.ok(rootInfo?.isDirectory() && !rootInfo.isSymbolicLink(), 'reference fixtures require the actual owned stage root');
  const owner = JSON.parse(await readFile(path.join(root, MARKER_FILE), 'utf8'));
  assert.equal(owner.kind, 'h3-finetune-stage-owner'); assert.equal(owner.root, path.resolve(root), 'owner marker does not name this canonical root');
  const stageReceipt = JSON.parse(await readFile(path.join(root, RECEIPT_FILE), 'utf8'));
  assert.equal(stageReceipt.kind, 'h3-finetune-stage-receipt');
  assert.equal(stageReceipt.status, 'completed', 'reference fixtures require an actually completed stage receipt');
  const outputDir = path.join(root, 'conditioning', 'cc0');
  assert.equal(await lstatOrUndefined(outputDir), undefined, 'refusing to overwrite an existing conditioning/cc0 output directory');
  await ensureOwnedDir(root, outputDir);
  const toolPaths = allowUnitFake
    ? { ffmpeg: tools?.ffmpeg, ffprobe: tools?.ffprobe }
    : { ffmpeg: await resolveHostTool('ffmpeg', { env, platform, root }), ffprobe: await resolveHostTool('ffprobe', { env, platform, root }) };
  assert.ok(toolPaths.ffmpeg && toolPaths.ffprobe, 'trusted/injected ffmpeg and ffprobe paths required');
  const run = execute ?? promisify(execFile);
  const toolEnv = cleanToolEnv(env), startedAt = new Date(now()).toISOString();
  const receipt = {
    version: 1, kind: 'conditioning-reference-fixtures', proof_kind: allowUnitFake ? 'unit_fake' : 'campaign_reference_fixtures',
    campaign_proof: false, meaning: 'CC0 authored ffmpeg conditioning math; not model output and not model proof',
    sourceCasebook: { path: CASEBOOK_RELATIVE, sha256: await casebookSha256() },
    runner: { backend: backend ?? env.H3_STAGE_BACKEND ?? null, platform, runnerName: env.RUNNER_NAME ?? null },
    tools: { ffmpeg: toolPaths.ffmpeg, ffprobe: toolPaths.ffprobe, ffmpegVersion: null, ffprobeVersion: null },
    outputRoot: outputDir, startedAt, endedAt: null, status: 'failed', fixtureCount: 0, expectedFixtureCount: REFERENCE_FIXTURE_RECIPES.length,
    fixtures: [],
  };
  try {
    for (const [name, argv] of [['ffmpegVersion', [toolPaths.ffmpeg, '-version']], ['ffprobeVersion', [toolPaths.ffprobe, '-version']]]) {
      const version = await runTool(argv, { execute: run, cwd: outputDir, env: toolEnv, timeoutMs });
      assert.equal(version.code, 0, `${name} invocation failed`);
      receipt.tools[name] = String(version.stdout).split('\n')[0].trim();
    }
    for (const recipe of REFERENCE_FIXTURE_RECIPES) {
      if (signal?.aborted) throw stageError('H3_STAGE_INTERRUPTED', 'reference fixture preparation interrupted');
      const asset = casebook.assets?.[recipe.assetKey];
      assert.ok(asset && Array.isArray(asset.command) && asset.command.length > 1 && asset.command.every(value => typeof value === 'string'), `published recipe ${recipe.assetKey} required`);
      assert.equal(path.basename(asset.command[0]), 'ffmpeg', `recipe ${recipe.assetKey} must call ffmpeg, not a shell or arbitrary binary`);
      const outputName = asset.command.at(-1);
      assertSafeFileName(outputName);
      assert.equal(outputName.includes('/'), false, `recipe ${recipe.assetKey} output must stay in the cc0 directory`);
      const argv = [toolPaths.ffmpeg, ...asset.command.slice(1)];
      const execution = await runTool(argv, { execute: run, cwd: outputDir, env: toolEnv, timeoutMs });
      assert.equal(execution.code, 0, `recipe ${recipe.assetKey} exited ${execution.code}: ${execution.stderr}`);
      const outputPath = path.join(outputDir, outputName);
      assert.ok(within(outputDir, outputPath), `recipe ${recipe.assetKey} output escaped the cc0 directory`);
      const outputInfo = await lstatOrUndefined(outputPath);
      assert.ok(outputInfo?.isFile() && !outputInfo.isSymbolicLink() && outputInfo.size > 0, `recipe ${recipe.assetKey} produced no positive regular output`);
      const probe = await runTool(probeArgv(toolPaths.ffprobe, recipe.kind, outputName), { execute: run, cwd: outputDir, env: toolEnv, timeoutMs });
      assert.equal(probe.code, 0, `ffprobe ${recipe.assetKey} failed`);
      const streams = JSON.parse(String(probe.stdout)).streams;
      assert.ok(Array.isArray(streams) && streams.length > 0, `ffprobe ${recipe.assetKey} reported no streams`);
      const expectedType = recipe.kind === 'audio' ? 'audio' : 'video';
      const stream = streams.find(value => value.codec_type === expectedType);
      assert.ok(stream, `ffprobe ${recipe.assetKey} missing actual ${expectedType} stream`);
      if (recipe.kind === 'audio') {
        assert.ok(Number(stream.sample_rate) > 0 && Number(stream.channels) > 0, `ffprobe ${recipe.assetKey} audio sample rate/channels missing`);
        assert.ok(Number(stream.duration) > 0, `ffprobe ${recipe.assetKey} duration missing`);
      } else {
        assert.ok(Number(stream.width) > 0 && Number(stream.height) > 0, `ffprobe ${recipe.assetKey} geometry missing`);
        assert.ok(Number(stream.nb_read_frames) > 0, `ffprobe ${recipe.assetKey} frame count missing`);
        if (recipe.kind === 'video') assert.ok(Number(stream.duration) > 0, `ffprobe ${recipe.assetKey} duration missing`);
      }
      let decode = null;
      if (recipe.kind === 'video') {
        const decoded = await runTool(decodeArgv(toolPaths.ffmpeg, recipe.kind, outputName), { execute: run, cwd: outputDir, env: toolEnv, timeoutMs });
        assert.equal(decoded.code, 0, `full CPU decode failed for ${recipe.assetKey}`);
        decode = { command: decodeArgv(toolPaths.ffmpeg, recipe.kind, outputName), code: decoded.code };
      }
      receipt.fixtures.push({
        id: recipe.id, assetKey: recipe.assetKey, role: recipe.role, kind: recipe.kind,
        command: argv, outputName, outputPath, sizeBytes: outputInfo.size, sha256: await hashFileSha256(outputPath, { signal, now }),
        exit: { code: execution.code }, probe: { command: probeArgv(toolPaths.ffprobe, recipe.kind, outputName), code: probe.code, streams }, decode,
        provenance: 'CC0-1.0 authored ffmpeg math; no model output, download, or license acceptance',
      });
    }
    receipt.fixtureCount = receipt.fixtures.length;
    assert.equal(receipt.fixtureCount, REFERENCE_FIXTURE_RECIPES.length, 'every required conditioning fixture must be positively prepared');
    receipt.provenance = 'All fixtures are CC0 authored ffmpeg math from the published casebook; no model output is claimed';
    receipt.status = 'prepared-not-modelproof'; receipt.endedAt = new Date(now()).toISOString();
    await writeReferenceReceipt({ root, evidenceDir, receipt });
    return receipt;
  } catch (error) {
    receipt.status = signal?.aborted || error.code === 'H3_STAGE_INTERRUPTED' ? 'interrupted' : 'failed';
    receipt.fixtureCount = receipt.fixtures.length;
    receipt.error = { code: error.code ?? 'H3_STAGE_REFERENCE_FIXTURES', message: error.message };
    receipt.endedAt = new Date(now()).toISOString();
    try { await writeReferenceReceipt({ root, evidenceDir, receipt }); } catch { /* keep the original failure */ }
    throw Object.assign(error, { referenceFixtures: receipt });
  }
}
// Explicit-only cleanup manifest. Never automatic, never a global cache sweeper.
export function cleanupPolicy(receipt) {
  assert.equal(receipt?.kind, 'h3-finetune-stage-receipt');
  return {
    kind: 'h3-finetune-stage-cleanup-policy',
    automatic: false,
    privateRoot: receipt.settings.privateRoot,
    marker: path.join(receipt.settings.privateRoot, MARKER_FILE),
    requires: [CLEANUP_ACKNOWLEDGEMENT, 'root derived from backend/platform/run/attempt as sc26691-private/<run>-<attempt>', 'canonical owner marker names this exact root', 'stage receipt is terminal (not running)', 'any services/ dir has actual owned lifecycle + clean exit receipts for every epoch', 'no symlink path component'],
    never: ['on failure or interruption automatically', 'global HF cache', 'unmarked or unowned path', 'an acknowledgement string alone is not lease proof'],
  };
}
async function assertCleanupPath(root) {
  const resolved = path.resolve(root);
  assert.equal(path.basename(path.dirname(resolved)), 'sc26691-private', 'cleanup only the sc26691-private run root');
  for (let current = resolved; ; current = path.dirname(current)) {
    const info = await lstatOrUndefined(current);
    assert.ok(info, `cleanup path component missing: ${current}`);
    assert.ok(info.isDirectory() && !info.isSymbolicLink(), `refusing cleanup through a symlink/non-directory path: ${current}`);
    if (path.basename(current) === 'sc26691-private') break;
    assert.notEqual(current, path.dirname(current), 'cleanup path has no sc26691-private root');
  }
  const base = path.dirname(path.resolve(root, '..'));
  const baseInfo = await lstatOrUndefined(base);
  assert.ok(baseInfo?.isDirectory() && !baseInfo.isSymbolicLink(), 'refusing cleanup through a symlinked hub parent');
  return resolved;
}
// Any owned service state must carry actual exit and lifecycle receipts. An acknowledgement string
// alone is never lease proof.
async function assertNoLiveOwnedServices(root) {
  const services = path.join(root, 'services');
  const info = await lstatOrUndefined(services);
  if (!info) return { checked: false, cases: [] };
  assert.ok(info.isDirectory() && !info.isSymbolicLink(), 'services path must be a real directory');
  const cases = (await readdir(services, { withFileTypes: true })).filter(entry => entry.isDirectory()).map(entry => entry.name);
  assert.ok(cases.length > 0, 'services directory without owned lifecycle receipts cannot be cleaned');
  for (const caseId of cases) {
    const dir = path.join(services, caseId);
    const lifecyclePath = path.join(dir, 'service-lifecycle.json');
    const lifecycleInfo = await lstatOrUndefined(lifecyclePath);
    assert.ok(lifecycleInfo?.isFile(), `missing owned service-lifecycle receipt for ${caseId}; refusing cleanup`);
    const lifecycle = JSON.parse(await readFile(lifecyclePath, 'utf8'));
    assert.ok(Array.isArray(lifecycle.epochs) && lifecycle.epochs.length > 0, `service lifecycle has no epochs for ${caseId}; refusing cleanup`);
    for (const epoch of lifecycle.epochs) {
      const exitsPath = path.join(dir, `exits-${epoch.epoch}.json`);
      const exitsInfo = await lstatOrUndefined(exitsPath);
      assert.ok(exitsInfo?.isFile(), `missing owned exits-${epoch.epoch}.json for ${caseId}; refusing cleanup`);
      const exits = JSON.parse(await readFile(exitsPath, 'utf8'));
      assert.ok(exits.workerExit?.code === 0 && exits.apiExit?.code === 0, `owned services for ${caseId} epoch ${epoch.epoch} lack clean exit receipts; refusing cleanup`);
    }
  }
  return { checked: true, cases };
}
export async function cleanupPrivateRoot({ root, acknowledgement, expectedRunId, expectedRunAttempt, backend, env = process.env, platform = process.platform }) {
  assert.equal(acknowledgement, CLEANUP_ACKNOWLEDGEMENT, 'explicit post-validation cleanup acknowledgement required');
  assert.ok(typeof root === 'string' && path.isAbsolute(root), 'explicit absolute private root required');
  const expected = privateRunRoot({ backend, platform, env, runId: expectedRunId, attempt: expectedRunAttempt });
  assert.equal(path.resolve(root), path.resolve(expected), 'cleanup only the derived approved sc26691-private run root');
  const resolved = await assertCleanupPath(root);
  const marker = JSON.parse(await readFile(path.join(root, MARKER_FILE), 'utf8'));
  assert.equal(marker.kind, 'h3-finetune-stage-owner');
  assert.equal(marker.root, resolved, 'owner marker does not name this canonical root');
  assert.equal(marker.runId, String(expectedRunId)); assert.equal(marker.runAttempt, String(expectedRunAttempt));
  assert.equal(marker.backend, backend, 'owner marker backend differs from the derived root');
  const receiptPath = path.join(root, RECEIPT_FILE), receiptInfo = await lstatOrUndefined(receiptPath);
  assert.ok(receiptInfo?.isFile(), 'missing actual stage receipt; refusing cleanup');
  const receipt = JSON.parse(await readFile(receiptPath, 'utf8'));
  assert.equal(receipt.kind, 'h3-finetune-stage-receipt');
  assert.notEqual(receipt.status, 'running', 'refusing cleanup while staging is still running');
  const services = await assertNoLiveOwnedServices(root);
  await rm(root, { recursive: true, force: false });
  return { removed: root, marker, status: receipt.status, services };
}
async function main() {
  try {
    const backend = process.argv[2] ?? process.env.H3_STAGE_BACKEND;
    const guard = assertCliEnvironment({ env: process.env, platform: process.platform, backend });
    assert.ok(process.env.H3_STAGE_REQUEST, 'H3_STAGE_REQUEST call request JSON is required');
    assert.ok(process.env.H3_STAGE_EVIDENCE, 'H3_STAGE_EVIDENCE runner-temporary evidence directory is required');
    const controller = new AbortController();
    for (const signalName of ['SIGINT', 'SIGTERM']) process.on(signalName, () => controller.abort());
    const receipt = await runStage({ request: process.env.H3_STAGE_REQUEST, backend: guard.backend, env: process.env, platform: process.platform, codeSha: guard.codeSha, evidenceDir: process.env.H3_STAGE_EVIDENCE, signal: controller.signal });
    console.log(JSON.stringify({ status: receipt.status, root: receipt.settings.privateRoot, files: receipt.files.length, proof_kind: receipt.proof_kind }));
  } catch (error) { console.error(error.code ? `${error.code}: ${error.message}` : error.message); process.exitCode = 1; }
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
