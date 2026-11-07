// Pure CPU/in-process fakes. No listeners, downloads, GPU init, model execution or shared-cache access.
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm, symlink, lstat, readdir, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import os from 'node:os';
import {
  CLEANUP_ACKNOWLEDGEMENT, CUDA_RUNNERS, FILE_TIMEOUT_MS, HF_HOST, SINGULARITY_SET,
  assertCliEnvironment, assertStageAuthority, assertWritePathOffWeights, expandHomePath, canonicalSingularityFiles, casebookSha256, cacheHub, cleanupPolicy, cleanupPrivateRoot,
  defaultProbe, hashFileSha256, identityControlSourceEntry, inspectManifest, loadCasebook, parseStageRequest, prepareIdentityControls, prepareReferenceFixtures, privateRunRoot, probeCacheFile, runStage, selectStageFiles, sourceUrl, stageOneFile, writeAllBytes,
} from './h3-finetune-terminal-stage.mjs';

const sha256 = buffer => createHash('sha256').update(buffer).digest('hex');
const sourceSha = 'a'.repeat(40);
async function tempDir(t, label = 'sc26691-stage-unit-') {
  const root = await mkdtemp(path.join(os.tmpdir(), label));
  t.after(() => rm(root, { recursive: true, force: true }));
  t.after(() => rm(workOf(root), { recursive: true, force: true }));
  return root;
}
// The campaign work root is a SIBLING of the unit HF home: writes never land under the weights root.
const workOf = hfHome => `${hfHome}-work`;
function tinyBook(bytes, file = 'tiny.bin', repo = 'unit/fake', revision = 'b'.repeat(40)) {
  return {
    source: { repo: 'WarmBloodAban/Minimax-h3_Singularity', revision: 'a56850c870825d6a8ae501a595c4dbbbc77bbc7e' },
    artifacts: [],
    stagingAssets: {
      tiny: [{ repo, revision, file, expectedBytes: bytes.length, expectedSha256: sha256(bytes), legal: { licenseId: 'unit_fake-license' } }],
      second: [{ repo, revision, file: 'second.bin', expectedBytes: 3, expectedSha256: sha256(Buffer.from('abc')), legal: { licenseId: 'unit_fake-license' } }],
    },
  };
}
function authorityFor(files, kind = 'unit_fake') {
  return { kind, acknowledged: true, approvedBy: 'unit_fake test only', reference: 'synthetic consent, never real approval', approvedAt: '2026-10-25T00:00:00Z', subjects: files.map(file => ({ role: 'source', sha256: file.expectedSha256, acknowledged: true, licenseId: file.legal?.licenseId })) };
}
function stageEnv(hfHome, overrides = {}) {
  return { GITHUB_ACTIONS: 'true', GITHUB_EVENT_NAME: 'workflow_dispatch', GITHUB_SHA: sourceSha, GITHUB_RUN_ID: '101', GITHUB_RUN_ATTEMPT: '1', HF_HOME: hfHome, HOME: hfHome, H3_WORK_ROOT: workOf(hfHome), RUNNER_NAME: 'nax-macos', H3_EXPECTED_RUNNER: overrides.RUNNER_NAME ?? 'nax-macos', H3_TERMINAL_ROUTE: overrides.RUNNER_NAME?.startsWith('cuda-')?'h3-terminal-cuda':'h3-terminal-mlx', ...overrides };
}
function streamResponse(chunks, { status = 200, headers = {}, onChunk } = {}) {
  return {
    status,
    headers: { get: name => headers[String(name).toLowerCase()] ?? null },
    body: (async function* () { for (const chunk of chunks) { await onChunk?.(chunk); yield chunk; } })(),
  };
}
function fakeTransport(handler) {
  const calls = [];
  const impl = async (url, options) => { calls.push({ url, options }); return handler(url, options); };
  impl.calls = calls;
  return impl;
}
function partPathFor(root, repo, revision, file) {
  return path.join(root, 'hf', 'hub', `models--${repo.replaceAll('/', '--')}`, 'snapshots', revision, `${file}.part`);
}
function finalPathFor(root, repo, revision, file) {
  return path.join(root, 'hf', 'hub', `models--${repo.replaceAll('/', '--')}`, 'snapshots', revision, file);
}
async function readReceiptAt(evidence) {
  const names = (await readdir(evidence)).filter(name => name.startsWith('stage-receipt-'));
  assert.equal(names.length, 1);
  return JSON.parse(await readFile(path.join(evidence, names[0]), 'utf8'));
}

test('canonical Singularity selection is exactly the four frozen files and is not caller-overridable', async () => {
  const book = await loadCasebook();
  const files = selectStageFiles(book);
  assert.deepEqual(files.map(file => file.file), book.artifacts.map(artifact => artifact.artifact));
  assert.deepEqual(files.map(file => [file.expectedBytes, file.expectedSha256]), book.artifacts.map(artifact => [artifact.expectedBytes, artifact.expectedSha256]));
  assert.equal(files.reduce((total, file) => total + file.expectedBytes, 0), 100744593468);
  assert.equal(files.length, 4);
  assert.deepEqual(canonicalSingularityFiles(book), files);
  for (const file of files) { assert.equal(file.repo, book.source.repo); assert.equal(file.revision, book.source.revision); assert.match(file.expectedSha256, /^[a-f0-9]{64}$/); }
  assert.throws(() => selectStageFiles(book, []), /non-empty/);
  assert.throws(() => selectStageFiles(book, ['unknown-set']), /unknown or empty/);
  assert.throws(() => selectStageFiles(book, [SINGULARITY_SET, SINGULARITY_SET]), /duplicate/);
  const duplicate = { ...book, stagingAssets: { dup: [{ repo: 'unit/fake', revision: 'b'.repeat(40), file: 'x.bin', expectedBytes: 1, expectedSha256: sha256(Buffer.from('x')) }, { repo: 'unit/fake', revision: 'b'.repeat(40), file: 'x.bin', expectedBytes: 1, expectedSha256: sha256(Buffer.from('x')) }] } };
  assert.throws(() => selectStageFiles(duplicate, ['dup']), /duplicate/);
  for (const file of ['../evil.bin', '..\\evil.bin', '', 'a\\b', 'a:b', '/absolute', 'a/./b']) {
    const bad = { ...book, stagingAssets: { bad: [{ repo: 'unit/fake', revision: 'b'.repeat(40), file, expectedBytes: 1, expectedSha256: sha256(Buffer.from('x')) }] } };
    assert.throws(() => selectStageFiles(bad, ['bad']), /relative|unsafe|path/);
  }
});
test('optional large raw snapshot is cache-only unless a fresh download is explicitly authorized',async t=>{
  const home=await tempDir(t),bytes=Buffer.from('tiny raw snapshot stand-in'),book=tinyBook(bytes),files=selectStageFiles(book,['tiny']);book.stagingPlan={cacheProbeFirst:['tiny']};const transport=fakeTransport(()=>streamResponse([bytes]));
  const options={request:{asset_sets:['tiny'],license_authority:authorityFor(files)},backend:'mlx',book,env:stageEnv(home),platform:'darwin',fetchImpl:transport,evidenceDir:path.join(home,'evidence'),allowUnitFake:true,probe:async()=>({kind:'unit_fake'})};
  await assert.rejects(runStage(options),/H3_STAGE_CACHE_REQUIRED/);assert.equal(transport.calls.length,0);assert.equal((await readFile(path.join(home,'evidence','large-builtin-cache-preflight.json'),'utf8')).includes('missing'),true);
  const completed=await runStage({...options,request:{...options.request,allow_fresh_builtin_download:true}});assert.equal(completed.status,'completed');assert.equal(transport.calls.length,1);
});
test('call request cannot override frozen file metadata and defaults to the Singularity set', () => {
  assert.deepEqual(parseStageRequest('{}').assetSets, [SINGULARITY_SET]);
  assert.deepEqual(parseStageRequest({ asset_sets: ['tiny'] }).assetSets, ['tiny']);
  assert.throws(() => parseStageRequest({ asset_sets: [] }), /non-empty/);
  for (const key of ['files', 'stagingAssets', 'assetSetFiles']) assert.throws(() => parseStageRequest({ [key]: [] }), /override frozen staging metadata/);
});
test('real authority must be explicit, acknowledged and cover every selected full SHA', async () => {
  const book = tinyBook(Buffer.from('authority-bytes')), files = selectStageFiles(book, ['tiny']);
  const real = authorityFor(files, 'user-approved');
  assert.equal(assertStageAuthority(real, files).kind, 'user-approved');
  const fake = authorityFor(files, 'unit_fake');
  assert.throws(() => assertStageAuthority(fake, files), /synthetic consent/);
  assert.equal(assertStageAuthority(fake, files, { allowUnitFake: true }).kind, 'unit_fake');
  for (const mutate of [v => { v.acknowledged = false; }, v => { v.kind = 'inferred'; }, v => { v.approvedBy = ''; }, v => { v.approvedAt = 'not-a-date'; }, v => { v.reference = ''; }, v => { v.subjects = v.subjects.slice(0, -1); }, v => { v.subjects = v.subjects.map(s => ({ ...s, sha256: 'f'.repeat(64) })); }]) {
    const changed = structuredClone(real); mutate(changed); assert.throws(() => assertStageAuthority(changed, files));
  }
  assert.throws(() => assertStageAuthority({ ...real, subjects: [{ role: 'component', sha256: files[0].expectedSha256 }] }, files), /missing explicit license authority|lacks an explicit/);
});
test('CLI guards reject local/wrong-runner execution before any filesystem or network action', async t => {
  const local = stageEnv(await tempDir(t), { GITHUB_ACTIONS: undefined });
  assert.throws(() => assertCliEnvironment({ env: local, platform: 'darwin', backend: 'mlx' }), /Actions-only/);
  assert.throws(() => assertCliEnvironment({ env: stageEnv('/tmp/hf', { GITHUB_EVENT_NAME: 'push' }), platform: 'darwin', backend: 'mlx' }), /workflow_dispatch/);
  assert.throws(() => assertCliEnvironment({ env: stageEnv('/tmp/hf', { GITHUB_EVENT_NAME: undefined }), platform: 'darwin', backend: 'mlx' }), /workflow_dispatch/);
  assert.throws(() => assertCliEnvironment({ env: stageEnv('/tmp/hf', { GITHUB_RUN_ID: undefined }), platform: 'darwin', backend: 'mlx' }), /GITHUB_RUN_ID/);
  assert.throws(() => assertCliEnvironment({ env: { ...stageEnv('/tmp/hf'), GITHUB_SHA: 'short' }, platform: 'darwin', backend: 'mlx' }), /match/);
  assert.throws(() => assertCliEnvironment({ env: stageEnv('/tmp/hf'), platform: 'linux', backend: 'mlx' }), /macOS-only/);
  assert.throws(() => assertCliEnvironment({ env: stageEnv('/tmp/hf', { RUNNER_NAME: 'mac-studio' }), platform: 'darwin', backend: 'mlx' }), /nax-macos/);
  // sc-26691: either approved Mac may run the MLX lane, but only the one the request names.
  assert.equal(assertCliEnvironment({ env: stageEnv('/tmp/hf', { RUNNER_NAME: 'nax-macos-2' }), platform: 'darwin', backend: 'mlx' }).backend, 'mlx');
  assert.throws(() => assertCliEnvironment({ env: stageEnv('/tmp/hf', { RUNNER_NAME: 'nax-macos-2', H3_EXPECTED_RUNNER: 'nax-macos' }), platform: 'darwin', backend: 'mlx' }), /expected_runner/);
  assert.throws(() => assertCliEnvironment({ env: stageEnv('/tmp/hf', { H3_EXPECTED_RUNNER: undefined }), platform: 'darwin', backend: 'mlx' }), /expected_runner/);
  assert.throws(() => assertCliEnvironment({ env: stageEnv('/tmp/hf', { RUNNER_NAME: CUDA_RUNNERS[1], H3_EXPECTED_RUNNER: CUDA_RUNNERS[0] }), platform: 'win32', backend: 'cuda' }), /expected_runner/);
  assert.throws(() => assertCliEnvironment({ env: stageEnv('/tmp/hf'), platform: 'win32', backend: 'cuda' }), /full-weights/);
  assert.throws(() => assertCliEnvironment({ env: stageEnv('/tmp/hf', { RUNNER_NAME: 'cuda-windows-3' }), platform: 'win32', backend: 'cuda' }), /full-weights/);
  assert.equal(assertCliEnvironment({ env: stageEnv('/tmp/hf'), platform: 'darwin', backend: 'mlx' }).codeSha, sourceSha);
  assert.equal(assertCliEnvironment({ env: stageEnv('/tmp/hf', { RUNNER_NAME: CUDA_RUNNERS[1] }), platform: 'win32', backend: 'cuda' }).backend, 'cuda');
  assert.throws(() => assertCliEnvironment({ env: { ...stageEnv('/tmp/hf'), H3_STAGE_SOURCE_SHA: 'c'.repeat(40) }, platform: 'darwin', backend: 'mlx' }), /exact GITHUB_SHA/);
  // Wrong backend/platform is refused before any root/evidence path exists.
  const hfHome = await tempDir(t), evidence = path.join(hfHome, 'evidence');
  await assert.rejects(runStage({ request: { asset_sets: [SINGULARITY_SET], license_authority: authorityFor([{ file: 'x', expectedSha256: 'e'.repeat(64) }], 'user-approved') }, backend: 'mlx', env: stageEnv(hfHome, { HF_HOME: hfHome }), platform: 'darwin', evidenceDir: evidence }), /explicit legal.licenseId|missing explicit license authority/);
  await assert.rejects(lstat(path.join(workOf(hfHome), 'sc26691-private')), /ENOENT/);
  // sc-26691: run roots live under the WORK root (default D:\sc26691-campaign / ~/sc26691-campaign),
  // never under the weights root, and a work root on the weights volume is refused loudly.
  assert.equal(privateRunRoot({ backend: 'cuda', platform: 'win32', env: {}, runId: '42', attempt: '2' }), 'D:\\sc26691-campaign\\sc26691-private\\42-2');
  assert.equal(privateRunRoot({ backend: 'cuda', platform: 'win32', env: { H3_WORK_ROOT: 'F:\\w' }, runId: '42', attempt: '2' }), 'F:\\w\\sc26691-private\\42-2');
  const home = path.resolve(os.tmpdir(), 'unit-home'), hf = path.resolve(os.tmpdir(), 'unit-models', 'huggingface');
  assert.equal(privateRunRoot({ backend: 'mlx', platform: 'darwin', env: { HF_HOME: hf, HOME: home }, runId: '42', attempt: '2' }), path.join(home, 'sc26691-campaign', 'sc26691-private', '42-2'));
  for (const workRoot of ['E:\\scratch', 'e:\\huggingface\\sc26691-private', '\\\\?\\E:\\x']) assert.throws(() => privateRunRoot({ backend: 'cuda', platform: 'win32', env: { H3_WORK_ROOT: workRoot }, runId: '42', attempt: '2' }), /H3_WRITE_ON_WEIGHTS_VOLUME/);
  assert.throws(() => privateRunRoot({ backend: 'mlx', platform: 'darwin', env: { HF_HOME: hf, HOME: home, H3_WORK_ROOT: path.join(hf, 'campaign') }, runId: '42', attempt: '2' }), /H3_WRITE_UNDER_WEIGHTS_ROOT/);
  assert.throws(() => privateRunRoot({ backend: 'mlx', platform: 'darwin', env: { HF_HOME: hf, HOME: hf }, runId: '42', attempt: '2' }), /H3_WRITE_UNDER_WEIGHTS_ROOT/, 'a default ~/ work root under the HF home is refused too');
  assert.equal(expandHomePath('~/sc26691-hf', { platform: 'darwin', env: { HOME: home } }), path.join(home, 'sc26691-hf'));
  assert.throws(() => expandHomePath('relative/hf', { platform: 'darwin', env: { HOME: home } }), /absolute path or ~\/ path required/);
  assert.equal(assertWritePathOffWeights('D:\\sc26691-campaign\\x', { platform: 'win32', env: {} }), 'D:\\sc26691-campaign\\x');
});
test('injected byte streams stage four-file-set members with actual size/full SHA and keep weights out of evidence', async t => {
  const bytes = Buffer.from('tiny frozen stream bytes with a deterministic digest');
  const book = tinyBook(bytes), files = selectStageFiles(book, ['tiny']);
  const hfHome = await tempDir(t), evidence = path.join(hfHome, 'evidence'), githubOutput = path.join(hfHome, 'output.txt');
  const transport = fakeTransport(() => streamResponse([bytes.subarray(0, 7), bytes.subarray(7)]));
  const receipt = await runStage({ request: { asset_sets: ['tiny'], license_authority: authorityFor(files) }, backend: 'mlx', book, env: stageEnv(hfHome, { GITHUB_OUTPUT: githubOutput }), platform: 'darwin', fetchImpl: transport, evidenceDir: evidence, allowUnitFake: true, probe: async () => ({ kind: 'unit_fake metadata only' }) });
  assert.equal(receipt.status, 'completed'); assert.equal(receipt.proof_kind, 'unit_fake'); assert.equal(receipt.campaign_proof, false);
  assert.equal(receipt.files.length, 1);
  const staged = receipt.files[0];
  assert.equal(staged.status, 'completed'); assert.equal(staged.actualBytes, bytes.length); assert.equal(staged.observedSha256, sha256(bytes));
  assert.equal(staged.repo, 'unit/fake'); assert.equal(staged.repoType, 'model');
  assert.deepEqual(staged.expectedLegal, { licenseId: 'unit_fake-license' });
  assert.equal(staged.explicitAcknowledgment.licenseId, 'unit_fake-license'); assert.equal(staged.explicitAcknowledgment.acknowledged, true); assert.match(staged.url, new RegExp(`^${HF_HOST}/unit/fake/resolve/b+`));
  assert.equal(await readFile(staged.finalPath, 'utf8'), bytes.toString('utf8'));
  await assert.rejects(lstat(staged.partPath), /ENOENT/);
  assert.ok((await lstat(path.join(receipt.settings.privateRoot, 'sc26691-owner.json'))).isFile());
  assert.ok((await readFile(githubOutput, 'utf8')).includes(`root=${receipt.settings.privateRoot}`));
  assert.equal(receipt.producer.codeSha, sourceSha);
  const evidenceFiles = await readdir(evidence);
  assert.ok(evidenceFiles.every(name => name.endsWith('.json') || name.endsWith('.log')), 'evidence holds only receipts/logs, never weights');
  const text = await readFile(path.join(evidence, evidenceFiles.find(name => name.startsWith('stage-receipt-'))), 'utf8');
  assert.ok(text.length < 100000); assert.ok(!text.includes(bytes.toString('base64')));
  assert.equal((await readReceiptAt(evidence)).status, 'completed');
  assert.equal(cleanupPolicy(receipt).automatic, false);
  const cleanupArgs = { root: receipt.settings.privateRoot, acknowledgement: CLEANUP_ACKNOWLEDGEMENT, expectedRunId: '101', expectedRunAttempt: '1', backend: 'mlx', env: stageEnv(hfHome), platform: 'darwin' };
  await assert.rejects(cleanupPrivateRoot({ ...cleanupArgs, acknowledgement: 'yes' }), /acknowledgement/);
  await assert.rejects(cleanupPrivateRoot({ ...cleanupArgs, expectedRunId: '999' }), /derived approved|strictly equal|Expected/);
  assert.ok((await cleanupPrivateRoot(cleanupArgs)).removed);
});
test('wrong size, zero bytes, oversize and wrong full SHA fail without false completion and retain the partial', async t => {
  const bytes = Buffer.from('0123456789abcdef');
  const book = tinyBook(bytes), files = selectStageFiles(book, ['tiny']);
  const setup = async label => { const hfHome = await tempDir(t, `sc26691-${label}-`); return { hfHome, evidence: path.join(hfHome, 'evidence'), env: stageEnv(hfHome) }; };
  const run = (context, transport) => runStage({ request: { asset_sets: ['tiny'], license_authority: authorityFor(files) }, backend: 'mlx', book, env: context.env, platform: 'darwin', fetchImpl: transport, evidenceDir: context.evidence, allowUnitFake: true, maxRetries: 0, probe: async () => ({ kind: 'unit_fake probe' }) });
  for (const [label, chunks, code] of [
    ['short', [bytes.subarray(0, 5)], /H3_STAGE_SIZE|expected 16/],
    ['zero', [], /H3_STAGE_ZERO_BYTES|zero-byte/],
    ['oversize', [Buffer.concat([bytes, Buffer.from('x')])], /H3_STAGE_OVERSIZE|exceeded/],
    ['hash', [Buffer.from('fedcba9876543210')], /H3_STAGE_SHA256|observed/],
  ]) {
    const context = await setup(label);
    await assert.rejects(run(context, fakeTransport(() => streamResponse(chunks))), code, label);
    const receipt = await readReceiptAt(context.evidence);
    assert.notEqual(receipt.status, 'completed', label);
    assert.equal(receipt.files[0].status, 'failed', label);
    assert.ok(receipt.files[0].error?.code, label);
    const part = receipt.files[0].partPath; assert.ok(part, label);
    assert.ok((await lstat(part)).isFile(), `${label}: partial retained for diagnosis`);
  }
});
test('HTTP 401/403 are explicit blockers and never fall back to other weights', async t => {
  const bytes = Buffer.from('blocked'), book = tinyBook(bytes), files = selectStageFiles(book, ['tiny']);
  for (const status of [401, 403]) {
    const hfHome = await tempDir(t, `sc26691-http-${status}-`), env = stageEnv(hfHome);
    const transport = fakeTransport(() => streamResponse([bytes], { status }));
    await assert.rejects(runStage({ request: { asset_sets: ['tiny'], license_authority: authorityFor(files) }, backend: 'mlx', book, env, platform: 'darwin', fetchImpl: transport, evidenceDir: path.join(hfHome, 'evidence'), allowUnitFake: true, maxRetries: 0, probe: async () => ({ kind: 'unit_fake probe' }) }), new RegExp(`H3_STAGE_HTTP_${status}`));
    assert.equal(transport.calls.length, 1, 'no fallback request was attempted');
    assert.equal((await readReceiptAt(path.join(hfHome, 'evidence'))).status, 'failed');
  }
});
test('range resume appends only validated partial bytes; an ignored Range restarts instead of appending', async t => {
  const bytes = Buffer.from('0123456789abcdefghijkl'), half = bytes.subarray(0, 10);
  const book = tinyBook(bytes), files = selectStageFiles(book, ['tiny']), entry = files[0];
  const resumeRoot = await tempDir(t, 'sc26691-resume-'); await mkdir(path.join(resumeRoot, 'hf', 'hub', `models--unit--fake`, 'snapshots', entry.revision), { recursive: true });
  await writeFile(partPathFor(resumeRoot, entry.repo, entry.revision, entry.file), half);
  const seen = [];
  const resume = fakeTransport((url, options) => { seen.push(options.headers.Range); return streamResponse([bytes.subarray(10)], { status: 206, headers: { 'content-range': `bytes 10-21/22` } }); });
  const completed = await stageOneFile({ entry, root: resumeRoot, fetchImpl: resume, maxRetries: 0, env: { HF_HUB_CACHE: path.join(resumeRoot, 'no-hub') } });
  assert.equal(completed.status, 'completed'); assert.equal(seen[0], 'bytes=10-');
  assert.equal(await readFile(completed.finalPath, 'utf8'), bytes.toString('utf8'));
  const reused = await stageOneFile({ entry, root: resumeRoot, fetchImpl: fakeTransport(() => { throw new Error('no refetch of a verified file'); }), maxRetries: 0, env: { HF_HUB_CACHE: path.join(resumeRoot, 'no-hub') } });
  assert.equal(reused.reused, true); assert.equal(reused.observedSha256, sha256(bytes));

  const restartRoot = await tempDir(t, 'sc26691-restart-'); await mkdir(path.join(restartRoot, 'hf', 'hub', `models--unit--fake`, 'snapshots', entry.revision), { recursive: true });
  await writeFile(partPathFor(restartRoot, entry.repo, entry.revision, entry.file), half);
  const restarted = await stageOneFile({ entry, root: restartRoot, fetchImpl: fakeTransport(() => streamResponse([bytes], { status: 200 })), maxRetries: 0, env: { HF_HUB_CACHE: path.join(restartRoot, 'no-hub') } });
  assert.equal(await readFile(restarted.finalPath, 'utf8'), bytes.toString('utf8'), 'Range-ignored body replaced the partial, never appended');

  const badRoot = await tempDir(t, 'sc26691-badrange-'); await mkdir(path.join(badRoot, 'hf', 'hub', `models--unit--fake`, 'snapshots', entry.revision), { recursive: true });
  await writeFile(partPathFor(badRoot, entry.repo, entry.revision, entry.file), half);
  await assert.rejects(stageOneFile({ entry, root: badRoot, fetchImpl: fakeTransport(() => streamResponse([bytes.subarray(10)], { status: 206, headers: { 'content-range': 'bytes 0-11/22' } })), maxRetries: 0, env: { HF_HUB_CACHE: path.join(badRoot, 'no-hub') } }), /Content-Range start/);
});
test('real runStage entry enforces the hosted manual guard before any filesystem or network action', async t => {
  const bytes = Buffer.from('guard'), hfHome = await tempDir(t, 'sc26691-realguard-'), evidence = path.join(hfHome, 'evidence');
  const book = tinyBook(bytes), files = selectStageFiles(book, ['tiny']);
  let calls = 0;
  const transport = async () => { calls += 1; throw new Error('network must not be touched'); };
  const request = { asset_sets: ['tiny'], license_authority: authorityFor(files, 'user-approved') };
  await assert.rejects(runStage({ request, backend: 'mlx', book, env: { ...stageEnv(hfHome), GITHUB_ACTIONS: undefined }, platform: 'darwin', fetchImpl: transport, evidenceDir: evidence }), /Actions-only/);
  await assert.rejects(runStage({ request, backend: 'mlx', book, env: stageEnv(hfHome, { GITHUB_EVENT_NAME: 'push' }), platform: 'darwin', fetchImpl: transport, evidenceDir: evidence }), /workflow_dispatch/);
  await assert.rejects(runStage({ request, backend: 'cuda', book, env: stageEnv(hfHome), platform: 'darwin', fetchImpl: transport, evidenceDir: evidence }), /Windows-only/);
  assert.equal(calls, 0);
  await assert.rejects(lstat(evidence), /ENOENT/);
  await assert.rejects(lstat(path.join(workOf(hfHome), 'sc26691-private')), /ENOENT/);
});
test('nested HF-relative paths stage with per-segment URL encoding and nested owned directories', async t => {
  const bytes = Buffer.from('nested bytes'), relative = 'teacher/model shard-0001.safetensors';
  const book = { source: { repo: 'WarmBloodAban/Minimax-h3_Singularity', revision: 'a56850c870825d6a8ae501a595c4dbbbc77bbc7e' }, artifacts: [], stagingAssets: { nested: [{ repo: 'unit/fake', revision: 'b'.repeat(40), file: relative, expectedBytes: bytes.length, expectedSha256: sha256(bytes), legal: { licenseId: 'unit_fake-nested' } }] } };
  const files = selectStageFiles(book, ['nested']);
  assert.equal(files[0].file, relative);
  for (const bad of ['a/../b', 'a//b', '/absolute', 'a\\b', 'C:/drive', 'a\0b']) assert.throws(() => selectStageFiles({ ...book, stagingAssets: { nested: [{ ...files[0], file: bad }] } }, ['nested']), /relative|unsafe/, bad);
  const hfHome = await tempDir(t, 'sc26691-nested-');
  let seen = null;
  const receipt = await runStage({ request: { asset_sets: ['nested'], license_authority: authorityFor(files) }, backend: 'mlx', book, env: stageEnv(hfHome, { HF_HUB_CACHE: path.join(hfHome, 'no-hub') }), platform: 'darwin', fetchImpl: fakeTransport(url => { seen = url; return streamResponse([bytes]); }), evidenceDir: path.join(hfHome, 'evidence'), allowUnitFake: true, probe: async () => ({ kind: 'unit_fake probe' }) });
  assert.match(seen, /teacher\/model%20shard-0001\.safetensors$/);
  assert.equal(await readFile(path.join(receipt.settings.privateRoot, 'hf', 'hub', 'models--unit--fake', 'snapshots', files[0].revision, 'teacher', 'model shard-0001.safetensors'), 'utf8'), bytes.toString('utf8'));
  assert.equal(receipt.files[0].observedSha256, sha256(bytes));
});
async function seedHub(hub, entry, bytes, { symlinkBlob = false } = {}) {
  const blob = path.join(hub, 'blobs', sha256(bytes));
  await mkdir(path.dirname(blob), { recursive: true }); await writeFile(blob, bytes);
  const snapshot = path.join(hub, `models--${entry.repo.replaceAll('/', '--')}`, 'snapshots', entry.revision, ...entry.file.split('/'));
  await mkdir(path.dirname(snapshot), { recursive: true });
  if (symlinkBlob) await symlink(path.relative(path.dirname(snapshot), blob), snapshot); else await writeFile(snapshot, bytes);
  return snapshot;
}
function cacheRunOptions(hfHome, book, files, hub, transport, extra = {}) {
  const setId = book.stagingAssets ? Object.keys(book.stagingAssets)[0] : SINGULARITY_SET;
  return { request: { asset_sets: [setId], license_authority: authorityFor(files) }, backend: 'mlx', book, env: stageEnv(hfHome, { HF_HUB_CACHE: hub }), platform: 'darwin', fetchImpl: transport, evidenceDir: path.join(hfHome, 'evidence'), allowUnitFake: true, probe: async () => ({ kind: 'unit_fake probe' }), ...extra };
}
test('known builtin cache is probed per exact file, copied (never hardlinked), and a mismatch blocks fresh fallback', async t => {
  const bytes = Buffer.from('cache-verified-bytes'), book = tinyBook(bytes), entry = selectStageFiles(book, ['tiny'])[0];
  const hfHome = await tempDir(t, 'sc26691-cache-'), hub = path.join(hfHome, 'hub');
  const snapshot = await seedHub(hub, entry, bytes, { symlinkBlob: true });
  assert.ok((await lstat(snapshot)).isSymbolicLink(), 'HF snapshot symlink fixture');
  let fetches = 0;
  const receipt = await runStage(cacheRunOptions(hfHome, book, [entry], hub, async () => { fetches += 1; throw new Error('cache hit must not fetch'); }));
  assert.equal(fetches, 0, 'existing cache preferred over a fresh transfer');
  assert.equal(receipt.files[0].cache.state, 'present'); assert.equal(receipt.files[0].cache.mode, 'cache-copy');
  assert.equal(receipt.files[0].observedSha256, sha256(bytes));
  assert.equal(await readFile(receipt.files[0].finalPath, 'utf8'), bytes.toString('utf8'));
  const wrong = Buffer.alloc(bytes.length, 9), hfWrong = await tempDir(t, 'sc26691-cache-wrong-'), hubWrong = path.join(hfWrong, 'hub');
  await seedHub(hubWrong, entry, wrong);
  await assert.rejects(runStage(cacheRunOptions(hfWrong, book, [entry], hubWrong, async () => { fetches += 1; return streamResponse([bytes]); })), /H3_STAGE_CACHE_MISMATCH/);
  assert.equal(fetches, 0, 'a present-but-wrong cache is a blocker, not a silent fresh fallback');
  const outside = path.join(await tempDir(t, 'sc26691-cache-outside-'), 'foreign.bin'); await writeFile(outside, bytes);
  const hfEscape = await tempDir(t, 'sc26691-cache-escape-'), hubEscape = path.join(hfEscape, 'hub');
  const escapeSnapshot = path.join(hubEscape, 'models--unit--fake', 'snapshots', entry.revision, entry.file);
  await mkdir(path.dirname(escapeSnapshot), { recursive: true }); await symlink(outside, escapeSnapshot);
  await assert.rejects(probeCacheFile(entry, hubEscape), /escapes the known HF hub/);
});
const SPEECH_DATASET = { repoType: 'dataset', repo: 'hf-internal-testing/librispeech_asr_dummy', revision: '5be91486e11a2d616f4ec5db8d3fd248585ac07a', file: 'clean/validation-00000-of-00001.parquet', expectedBytes: 9192059, expectedSha256: '4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603' };
test('dataset assets keep repoType identity, datasets-- cache prefix and the datasets/ URL shape', async t => {
  const book = { source: { repo: 'WarmBloodAban/Minimax-h3_Singularity', revision: 'a56850c870825d6a8ae501a595c4dbbbc77bbc7e' }, artifacts: [], stagingAssets: { speech: [SPEECH_DATASET] } };
  const files = selectStageFiles(book, ['speech']);
  assert.equal(files[0].repoType, 'dataset');
  assert.equal(sourceUrl(files[0]), 'https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/5be91486e11a2d616f4ec5db8d3fd248585ac07a/clean/validation-00000-of-00001.parquet');
  const hfHome = await tempDir(t, 'sc26691-dataset-'), hub = path.join(hfHome, 'hub');
  const probe = await probeCacheFile(files[0], hub);
  assert.equal(probe.state, 'absent');
  assert.ok(probe.candidate.endsWith(path.join('datasets--hf-internal-testing--librispeech_asr_dummy', 'snapshots', SPEECH_DATASET.revision, 'clean', 'validation-00000-of-00001.parquet')), probe.candidate);
  const mixed = { ...book, stagingAssets: { speech: [SPEECH_DATASET], model: [{ ...SPEECH_DATASET, repoType: 'model' }] } };
  assert.equal(selectStageFiles(mixed, ['speech', 'model']).length, 2, 'model and dataset identities never collide');
  const tiny = Buffer.from('parquet-unit-fake');
  const smallBook = { ...book, stagingAssets: { small: [{ repoType: 'dataset', repo: 'unit/fake-ds', revision: 'b'.repeat(40), file: 'clean/part-0.parquet', expectedBytes: tiny.length, expectedSha256: sha256(tiny), legal: { licenseId: 'unit_fake-dataset' } }] } };
  const smallFiles = selectStageFiles(smallBook, ['small']);
  let seen = null;
  const receipt = await runStage({ request: { asset_sets: ['small'], license_authority: authorityFor(smallFiles) }, backend: 'mlx', book: smallBook, env: stageEnv(hfHome, { HF_HUB_CACHE: path.join(hfHome, 'no-hub') }), platform: 'darwin', fetchImpl: fakeTransport(url => { seen = url; return streamResponse([tiny]); }), evidenceDir: path.join(hfHome, 'evidence'), allowUnitFake: true, probe: async () => ({ kind: 'unit_fake probe' }) });
  assert.match(seen, /^https:\/\/huggingface\.co\/datasets\/unit\/fake-ds\/resolve\/b+\/clean\/part-0\.parquet$/);
  assert.equal(receipt.files[0].repoType, 'dataset');
  assert.equal(receipt.files[0].observedSha256, sha256(tiny));
  assert.equal(await readFile(receipt.files[0].finalPath, 'utf8'), tiny.toString('utf8'));
});
test('real staging cannot inject book data while receipts hash the source casebook', async t => {
  const hfHome = await tempDir(t, 'sc26691-book-');
  await assert.rejects(runStage({ request: { asset_sets: ['tiny'], license_authority: authorityFor([{ file: 'x', expectedSha256: '0'.repeat(64) }], 'user-approved') }, backend: 'mlx', book: tinyBook(Buffer.from('x')), env: stageEnv(hfHome), platform: 'darwin', evidenceDir: path.join(hfHome, 'evidence') }), /inject book data/);
  await assert.rejects(lstat(path.join(hfHome, 'evidence')), /ENOENT/);
});
test('inspect_only reports actual per-file cache presence/hash without downloading or manufacturing approval', async t => {
  const first = Buffer.from('present-bytes'), second = Buffer.from('missing-bytes');
  const book = { source: { repo: 'WarmBloodAban/Minimax-h3_Singularity', revision: 'a56850c870825d6a8ae501a595c4dbbbc77bbc7e' }, artifacts: [], stagingAssets: { pair: [{ repo: 'unit/fake', revision: 'b'.repeat(40), file: 'present.bin', expectedBytes: first.length, expectedSha256: sha256(first), legal: { licenseId: 'unit_fake-pair' } }, { repo: 'unit/fake', revision: 'b'.repeat(40), file: 'missing.bin', expectedBytes: second.length, expectedSha256: sha256(second), legal: { licenseId: 'unit_fake-pair' } }] } };
  const files = selectStageFiles(book, ['pair']), hfHome = await tempDir(t, 'sc26691-inspect-'), hub = path.join(hfHome, 'hub');
  await seedHub(hub, files[0], first);
  let calls = 0;
  const receipt = await runStage({ request: { asset_sets: ['pair'], inspect_only: true }, backend: 'mlx', book, env: stageEnv(hfHome, { HF_HUB_CACHE: hub }), platform: 'darwin', fetchImpl: async () => { calls += 1; }, evidenceDir: path.join(hfHome, 'evidence'), allowUnitFake: true });
  assert.equal(calls, 0);
  assert.equal(receipt.kind, 'h3-finetune-stage-inspection-receipt'); assert.equal(receipt.status, 'inspected');
  assert.equal(receipt.campaign_proof, false); assert.equal(receipt.stagingComplete, false); assert.equal(receipt.licenseAuthority, undefined);
  assert.deepEqual([receipt.summary.present, receipt.summary.missing, receipt.summary.mismatched], [1, 1, 0]);
  const present = receipt.files.find(row => row.file === 'present.bin'), missing = receipt.files.find(row => row.file === 'missing.bin');
  assert.equal(present.matches, true); assert.equal(present.observedSha256, sha256(first));
  assert.deepEqual(present.expectedLegal, { licenseId: 'unit_fake-pair' });
  assert.equal(present.explicitAcknowledgment, undefined);
  assert.equal(missing.presence, 'missing'); assert.equal(missing.matches, false);
  await assert.rejects(lstat(path.join(workOf(hfHome), 'sc26691-private')), /ENOENT/);
  assert.equal((await readdir(path.join(hfHome, 'evidence'))).filter(name => name.startsWith('inspect-receipt-')).length, 1);
});
test('partial handle writes and metadata probes are counted/identified truthfully', async () => {
  const chunk = Buffer.from('abcdefg'), seen = [], writes = [];
  const handle = { async write(_chunk, offset, length) { const bytesWritten = Math.min(2, length); writes.push([offset, length, bytesWritten]); return { bytesWritten }; } };
  assert.equal(await writeAllBytes(handle, chunk, { onWritten: bytes => seen.push(bytes) }), chunk.length);
  assert.deepEqual(seen, [2, 2, 2, 1]); assert.deepEqual(writes.map(row => row[2]), [2, 2, 2, 1]);
  await assert.rejects(writeAllBytes({ write: async () => ({ bytesWritten: 0 }) }, chunk), /no forward progress/);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(writeAllBytes(handle, chunk, { signal: controller.signal }), /interrupted/);
  await assert.rejects(writeAllBytes(handle, chunk, { deadline: 0, now: () => 1 }), /job deadline/);
  const calls = [];
  const execute = async (command, args) => {
    calls.push([command, args]);
    if (command === 'nvidia-smi') return { stdout: '0, GPU-unit_fake-abc, 00000000:41:00.0, unit_fake GPU, 555.1, 32768\n' };
    if (command === 'system_profiler') return { stdout: JSON.stringify({ SPDisplaysDataType: [{ sppci_model: 'unit_fake Apple GPU' }] }) };
    if (command === 'sw_vers') return { stdout: '15.6\n' };
    throw new Error(command);
  };
  const cuda = await defaultProbe('cuda', execute);
  assert.deepEqual(cuda.rows, ['0, GPU-unit_fake-abc, 00000000:41:00.0, unit_fake GPU, 555.1, 32768']);
  assert.ok(calls[0][1].includes('--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total'));
  const mac = await defaultProbe('mlx', execute);
  assert.equal(mac.device, 'unit_fake Apple GPU'); assert.equal(mac.os.productVersion, '15.6'); assert.match(mac.stdoutSha256, /^[a-f0-9]{64}$/);
});
test('interruption aborts cooperatively, retains the partial receipt and never reports completion', async t => {
  const bytes = Buffer.alloc(64, 7), book = tinyBook(bytes), files = selectStageFiles(book, ['tiny']);
  const hfHome = await tempDir(t, 'sc26691-abort-'), controller = new AbortController(), env = stageEnv(hfHome);
  const transport = fakeTransport(() => ({ status: 200, headers: { get: () => null }, body: (async function* () { yield bytes.subarray(0, 16); controller.abort(); yield bytes.subarray(16); })() }));
  await assert.rejects(runStage({ request: { asset_sets: ['tiny'], license_authority: authorityFor(files) }, backend: 'mlx', book, env, platform: 'darwin', fetchImpl: transport, evidenceDir: path.join(hfHome, 'evidence'), allowUnitFake: true, signal: controller.signal, maxRetries: 0, probe: async () => ({ kind: 'unit_fake probe' }) }), /H3_STAGE_INTERRUPTED|interrupted/);
  const receipt = await readReceiptAt(path.join(hfHome, 'evidence'));
  assert.equal(receipt.status, 'interrupted');
  assert.equal(receipt.files[0].status, 'interrupted');
  assert.ok(receipt.files[0].actualBytes > 0 && receipt.files[0].actualBytes < bytes.length);
  assert.match(receipt.files[0].partialSha256, /^[a-f0-9]{64}$/, 'interrupted partial content hash retained');
  assert.ok((await lstat(receipt.files[0].partPath)).isFile());
  assert.equal(receipt.endedAt !== null, true);
});
test('symlinked staging components or targets are refused', async t => {
  const bytes = Buffer.from('safe'), book = tinyBook(bytes), entry = selectStageFiles(book, ['tiny'])[0];
  const root = await tempDir(t, 'sc26691-symlink-'), outside = await tempDir(t, 'sc26691-outside-');
  await symlink(outside, path.join(root, 'hf'));
  await assert.rejects(stageOneFile({ entry, root, fetchImpl: fakeTransport(() => streamResponse([bytes])), maxRetries: 0, env: { HF_HUB_CACHE: path.join(root, 'no-hub') } }), /symlink|component/);
  const cleanRoot = await tempDir(t, 'sc26691-symlink-target-');
  const dir = path.join(cleanRoot, 'hf', 'hub', `models--unit--fake`, 'snapshots', entry.revision);
  await mkdir(dir, { recursive: true });
  const outsideFile = path.join(outside, 'outside.bin'); await writeFile(outsideFile, 'outside');
  await symlink(outsideFile, `${path.join(dir, entry.file)}.part`);
  await assert.rejects(stageOneFile({ entry, root: cleanRoot, fetchImpl: fakeTransport(() => streamResponse([bytes])), maxRetries: 0, env: { HF_HUB_CACHE: path.join(cleanRoot, 'no-hub') } }), /symlink/);
  await assert.equal(await readFile(outsideFile, 'utf8'), 'outside', 'foreign bytes untouched');
});
test('private roots are never reused and receipts record producer code SHA, pins and device metadata only', async t => {
  const bytes = Buffer.from('reuse'), book = tinyBook(bytes), files = selectStageFiles(book, ['tiny']);
  const hfHome = await tempDir(t, 'sc26691-reuse-'), env = stageEnv(hfHome), evidence = path.join(hfHome, 'evidence');
  const options = { request: { asset_sets: ['tiny'], license_authority: authorityFor(files) }, backend: 'mlx', book, env, platform: 'darwin', fetchImpl: fakeTransport(() => streamResponse([bytes])), evidenceDir: evidence, allowUnitFake: true, probe: async () => ({ kind: 'unit_fake probe', gpuInitialized: false }) };
  const receipt = await runStage(options);
  assert.equal(receipt.pin.consumerPin, null); assert.match(receipt.pin.meaning, /Informational/);
  assert.equal(receipt.device.gpuInitialized, false); assert.match(receipt.source.casebookSha256, /^[a-f0-9]{64}$/);
  assert.match(receipt.source.hfHost, /huggingface/);
  assert.equal(receipt.settings.fileTimeoutMs, FILE_TIMEOUT_MS);
  await assert.rejects(runStage(options), /reuse an existing private staging root/);
});
async function ownedRoot(hfHome, { runId = '101', attempt = '1', status = 'completed', backend = 'mlx', services = false, symlinkParent = false } = {}) {
  const base = path.join(workOf(hfHome), 'sc26691-private'), root = path.join(base, `${runId}-${attempt}`);
  await mkdir(path.dirname(base), { recursive: true });
  if (symlinkParent) await symlink(hfHome, base); else await mkdir(base, { recursive: true });
  await mkdir(root, { recursive: true });
  await writeFile(path.join(root, 'sc26691-owner.json'), JSON.stringify({ kind: 'h3-finetune-stage-owner', root: path.resolve(root), runId, runAttempt: attempt, backend }));
  await writeFile(path.join(root, 'h3-stage-receipt.json'), JSON.stringify({ kind: 'h3-finetune-stage-receipt', status, settings: { privateRoot: root } }));
  if (services) await mkdir(path.join(root, 'services', 'mlx.capability.text'), { recursive: true });
  return root;
}
function serviceReceipts(root, { clean = true } = {}) {
  const dir = path.join(root, 'services', 'mlx.capability.text');
  return (async () => {
    await writeFile(path.join(dir, 'service-lifecycle.json'), JSON.stringify({ epochs: [{ epoch: 1 }] }));
    await writeFile(path.join(dir, 'exits-1.json'), JSON.stringify({ workerExit: { code: clean ? 0 : 1 }, apiExit: { code: clean ? 0 : 1 } }));
  })();
}
test('cleanup is gated on the derived run root, terminal receipt, real exit receipts and symlink-free parents', async t => {
  const hfHome = await tempDir(t, 'sc26691-cleanup-'), base = { acknowledgement: CLEANUP_ACKNOWLEDGEMENT, expectedRunId: '101', expectedRunAttempt: '1', backend: 'mlx', env: stageEnv(hfHome), platform: 'darwin' };
  const running = await ownedRoot(hfHome, { status: 'running' });
  await assert.rejects(cleanupPrivateRoot({ ...base, root: running }), /still running/);
  const stray = path.join(workOf(hfHome), 'sc26691-private', '555-1');
  await ownedRoot(hfHome, { runId: '555' });
  await assert.rejects(cleanupPrivateRoot({ ...base, root: stray, expectedRunId: '101' }), /derived approved/);
  const servicesMissing = await ownedRoot(hfHome, { runId: '102', services: true });
  await assert.rejects(cleanupPrivateRoot({ ...base, root: servicesMissing, expectedRunId: '102', expectedRunAttempt: '1' }), /missing owned service-lifecycle/);
  await serviceReceipts(servicesMissing, { clean: false });
  await assert.rejects(cleanupPrivateRoot({ ...base, root: servicesMissing, expectedRunId: '102', expectedRunAttempt: '1' }), /lack clean exit receipts/);
  await serviceReceipts(servicesMissing, { clean: true });
  assert.equal((await cleanupPrivateRoot({ ...base, root: servicesMissing, expectedRunId: '102', expectedRunAttempt: '1' })).removed, servicesMissing);
  const outside = await tempDir(t, 'sc26691-cleanup-outside-'), linkedHome = await tempDir(t, 'sc26691-cleanup-home-');
  await mkdir(path.join(outside, '103-1'), { recursive: true });
  await mkdir(workOf(linkedHome), { recursive: true });
  await symlink(outside, path.join(workOf(linkedHome), 'sc26691-private'));
  await assert.rejects(cleanupPrivateRoot({ ...base, root: path.join(workOf(linkedHome), 'sc26691-private', '103-1'), expectedRunId: '103', expectedRunAttempt: '1', env: stageEnv(linkedHome) }), /symlink/);
  assert.ok((await lstat(path.join(outside, '103-1'))).isDirectory(), 'foreign root untouched');
  assert.equal((await lstat(path.join(workOf(linkedHome), 'sc26691-private'))).isSymbolicLink(), true);
});

test('unavailable device metadata is recorded truthfully and is never a staging gate', async t => {
  const bytes = Buffer.from('probe-unavailable'), book = tinyBook(bytes), files = selectStageFiles(book, ['tiny']);
  const hfHome = await tempDir(t, 'sc26691-probe-');
  const receipt = await runStage({ request: { asset_sets: ['tiny'], license_authority: authorityFor(files) }, backend: 'mlx', book, env: stageEnv(hfHome, { HF_HUB_CACHE: path.join(hfHome, 'no-hub') }), platform: 'darwin', fetchImpl: fakeTransport(() => streamResponse([bytes])), evidenceDir: path.join(hfHome, 'evidence'), allowUnitFake: true, probe: async () => { throw new Error('unit_fake probe unavailable'); } });
  assert.equal(receipt.device.available, false);
  assert.match(receipt.device.error, /unit_fake probe unavailable/);
  assert.match(receipt.device.meaning, /Metadata probe unavailable/);
  assert.equal(receipt.status, 'completed');
});

test('canonical Singularity attaches book stagingLicenses.singularity as expected legal metadata', async () => {
  const book = await loadCasebook();
  const legal = { licenseId: 'unit_fake-canonical', inheritedLicenseId: 'unit_fake-base', attributionRequired: 'unit_fake credit' };
  const files = canonicalSingularityFiles({ ...book, stagingLicenses: { singularity: legal } });
  assert.equal(files.length, 4);
  for (const file of files) assert.deepEqual(file.legal, legal);
  assert.deepEqual(canonicalSingularityFiles(book)[0].legal,book.stagingLicenses.singularity);
  const missing=structuredClone(book);delete missing.stagingLicenses;assert.equal(canonicalSingularityFiles(missing)[0].legal,undefined,'book without stagingLicenses must not invent legal metadata');
});
test('explicit legal and per-file acknowledgment are preserved, matched and never synthesized or leaked', async t => {
  const bytes = Buffer.from('legal-bytes'), legal = { licenseId: 'unit_fake-L1', inheritedLicenseId: 'unit_fake-BASE', attributionRequired: 'credit: unit_fake source' };
  const book = { source: { repo: 'WarmBloodAban/Minimax-h3_Singularity', revision: 'a56850c870825d6a8ae501a595c4dbbbc77bbc7e' }, artifacts: [], stagingAssets: { legalSet: [{ repo: 'unit/fake', revision: 'b'.repeat(40), file: 'legal.bin', expectedBytes: bytes.length, expectedSha256: sha256(bytes), legal }] } };
  const files = selectStageFiles(book, ['legalSet']);
  assert.deepEqual(files[0].legal, legal);
  const authority = { kind: 'unit_fake', acknowledged: true, approvedBy: 'unit_fake only', reference: 'synthetic consent', approvedAt: '2026-10-25T00:00:00Z', subjects: [{ role: 'source', sha256: sha256(bytes), acknowledged: true, licenseId: 'unit_fake-L1', inheritedLicenseId: 'unit_fake-BASE', inheritedAcknowledged: true, attribution: 'Video credits: credit: unit_fake source; speaker unit_fake', secretToken: 'must-not-copy' }] };
  const hfHome = await tempDir(t, 'sc26691-legal-');
  const receipt = await runStage({ request: { asset_sets: ['legalSet'], license_authority: authority }, backend: 'mlx', book, env: stageEnv(hfHome, { HF_HUB_CACHE: path.join(hfHome, 'no-hub') }), platform: 'darwin', fetchImpl: fakeTransport(() => streamResponse([bytes])), evidenceDir: path.join(hfHome, 'evidence'), allowUnitFake: true, probe: async () => ({ kind: 'unit_fake probe' }) });
  const row = receipt.files[0];
  assert.deepEqual(row.expectedLegal, legal);
  assert.deepEqual(row.explicitAcknowledgment, { role: 'source', sha256: sha256(bytes), acknowledged: true, licenseId: 'unit_fake-L1', inheritedLicenseId: 'unit_fake-BASE', inheritedAcknowledged: true, attribution: 'Video credits: credit: unit_fake source; speaker unit_fake' });
  assert.equal(row.explicitAcknowledgment.secretToken, undefined, 'arbitrary subject secrets are never copied');
  assert.equal(receipt.licenseAuthority.subjects[0].secretToken, undefined);
  for (const mutate of [v => { delete v.subjects[0].acknowledged; }, v => { v.subjects[0].acknowledged = false; }, v => { v.subjects[0].licenseId = 'wrong'; }, v => { delete v.subjects[0].inheritedLicenseId; }, v => { v.subjects[0].inheritedAcknowledged = false; }, v => { v.subjects[0].attribution = 'no credit here'; }]) {
    const bad = structuredClone(authority); mutate(bad);
    assert.throws(() => assertStageAuthority(bad, files, { allowUnitFake: true }), /lacks an explicit|missing explicit/);
  }
  assert.throws(() => assertStageAuthority(authority, [{ ...files[0], legal: undefined }], { allowUnitFake: true }), /explicit legal.licenseId/);
  assert.throws(() => assertStageAuthority(authority, files), /synthetic consent/);
});

test('external abort aborts a hanging source request promptly and flushes an interrupted receipt', async t => {
  const bytes = Buffer.from('pending-fetch'), book = tinyBook(bytes), files = selectStageFiles(book, ['tiny']);
  const hfHome = await tempDir(t, 'sc26691-pending-'), controller = new AbortController();
  let requestSignal = null;
  const transport = async (url, options) => {
    requestSignal = options.signal;
    setTimeout(() => controller.abort(), 5);
    return new Promise((resolve, reject) => {
      if (options.signal.aborted) return reject(new Error('aborted'));
      options.signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true });
    });
  };
  await assert.rejects(runStage({ request: { asset_sets: ['tiny'], license_authority: authorityFor(files) }, backend: 'mlx', book, env: stageEnv(hfHome, { HF_HUB_CACHE: path.join(hfHome, 'no-hub') }), platform: 'darwin', fetchImpl: transport, evidenceDir: path.join(hfHome, 'evidence'), allowUnitFake: true, signal: controller.signal, maxRetries: 0, probe: async () => ({ kind: 'unit_fake probe' }) }), /H3_STAGE_INTERRUPTED|interrupted/);
  assert.ok(requestSignal && requestSignal.aborted, 'external controller is combined into the fetch signal');
  const receipt = await readReceiptAt(path.join(hfHome, 'evidence'));
  assert.equal(receipt.status, 'interrupted'); assert.equal(receipt.files[0].status, 'interrupted');
  await assert.rejects(lstat(receipt.files[0].finalPath), /ENOENT/);
});
test('non-streaming source bodies are refused instead of buffered into memory', async t => {
  const bytes = Buffer.from('buffered'), book = tinyBook(bytes), files = selectStageFiles(book, ['tiny']);
  const hfHome = await tempDir(t, 'sc26691-buffer-');
  await assert.rejects(runStage({ request: { asset_sets: ['tiny'], license_authority: authorityFor(files) }, backend: 'mlx', book, env: stageEnv(hfHome, { HF_HUB_CACHE: path.join(hfHome, 'no-hub') }), platform: 'darwin', fetchImpl: async () => ({ status: 200, headers: { get: () => null }, arrayBuffer: async () => bytes }), evidenceDir: path.join(hfHome, 'evidence'), allowUnitFake: true, maxRetries: 0, probe: async () => ({ kind: 'unit_fake probe' }) }), /streaming async-iterable body/);
  const receipt = await readReceiptAt(path.join(hfHome, 'evidence'));
  assert.equal(receipt.status, 'failed'); assert.equal(receipt.files[0].status, 'failed');
});
test('cache-copy and hash interruption retain durable bytes and never report completion', async t => {
  const bytes = Buffer.from('cache interrupted bytes'), book = tinyBook(bytes), entry = selectStageFiles(book, ['tiny'])[0];
  const hfHome = await tempDir(t, 'sc26691-cacheabort-'), hub = path.join(hfHome, 'hub'), root = path.join(hfHome, 'root');
  await seedHub(hub, entry, bytes);
  await mkdir(root, { recursive: true });
  const controller = new AbortController(); controller.abort();
  let error; try { await stageOneFile({ entry, root, env: { HF_HUB_CACHE: hub }, signal: controller.signal, maxRetries: 0 }); } catch (caught) { error = caught; }
  assert.ok(error, 'aborted cache-copy must fail');
  assert.equal(error.code, 'H3_STAGE_INTERRUPTED');
  assert.equal(error.record.status, 'interrupted');
  assert.equal(error.record.observedSha256, null);
  await assert.rejects(lstat(error.record.finalPath), /ENOENT/, 'no fake completed target');
  if (await lstat(error.record.partPath).then(() => true).catch(() => false)) assert.equal(error.record.actualBytes, 0);
  const file = path.join(hfHome, 'hash.bin'); await writeFile(file, bytes);
  await assert.rejects(hashFileSha256(file, { signal: controller.signal }), /interrupted/);
});
async function identitySource(hfHome, entry, bytes) {
  const sourcePath = path.join(hfHome, 'root', 'hf', 'hub', `models--${entry.repo.replaceAll('/', '--')}`, 'snapshots', entry.revision, ...entry.file.split('/'));
  await mkdir(path.dirname(sourcePath), { recursive: true });
  await writeFile(sourcePath, bytes);
  return sourcePath;
}
const identityAuthority = (reference = 'unit_fake explicit reference') => ({ kind: 'unit_fake', acknowledged: true, approvedBy: 'unit_fake only', reference, approvedAt: '2026-10-25T00:00:00Z', subjects: [] });
test('identity controls pick the smallest canonical file and emit real same-byte/negative copies', async t => {
  const book = await loadCasebook(), smallest = identityControlSourceEntry(book);
  assert.equal(smallest.file, 'Minimax-h3_Singularity_ref2va_v1.3_Pruned_w4a8.safetensors');
  assert.equal(smallest.expectedBytes, 11767930768, 'smallest canonical source avoids a 34GB copy');
  const hfHome = await tempDir(t, 'sc26691-identity-'), bytes = Buffer.from('identity-control-payload-bytes');
  const entry = { repoType: 'model', repo: 'unit/fake', revision: 'b'.repeat(40), file: 'source.bin', expectedBytes: bytes.length, expectedSha256: sha256(bytes), legal: { licenseId: 'unit_fake-L1' } };
  const sourcePath = await identitySource(hfHome, entry, bytes);
  const root = path.join(hfHome, 'root'), evidence = path.join(hfHome, 'evidence');
  const receipt = await prepareIdentityControls({ root, entry, authority: identityAuthority(), evidenceDir: evidence, runId: '1', runAttempt: '1', allowUnitFake: true, fixtureOffset: bytes.length - 1, now: () => 1000 });
  assert.equal(receipt.differentBytes.mutation.kind, 'unit_fake-injected-offset');
  assert.equal(receipt.status, 'prepared-not-modelproof'); assert.equal(receipt.proof_kind, 'unit_fake');
  assert.equal(receipt.sourceSizeBytes, bytes.length); assert.equal(receipt.sourceActualSHA, sha256(bytes)); assert.equal(receipt.sourcePath, sourcePath);
  assert.equal(receipt.renamed.size, bytes.length); assert.equal(receipt.renamed.actualSHA, sha256(bytes));
  assert.equal(receipt.differentBytes.size, bytes.length); assert.notEqual(receipt.differentBytes.actualSHA, sha256(bytes));
  assert.equal(receipt.differentBytes.expectedOriginalSHA, sha256(bytes));
  assert.equal(receipt.differentBytes.mutation.offset, bytes.length - 1);
  assert.equal(receipt.differentBytes.mutation.after, receipt.differentBytes.mutation.before ^ 1);
  assert.ok(receipt.renamed.path.endsWith(path.join('fixtures', 'independent-layout', 'unrelated-name.safetensors')));
  assert.equal(receipt.differentBytes.path, path.join(root, 'fixtures', 'independent-layout', 'different-bytes.safetensors'));
  assert.equal(receipt.originalExplicitAckReference.reference, 'unit_fake explicit reference');
  assert.equal(receipt.originalExplicitAckReference.licenseId, 'unit_fake-L1');
  assert.equal(receipt.derivedFixtures, 2); assert.match(receipt.derivedLicense, /no new consent/);
  assert.deepEqual(await readFile(sourcePath), bytes, 'shared source bytes unchanged');
  const [srcInfo, renamedInfo, negativeInfo] = await Promise.all([stat(sourcePath), stat(receipt.renamed.path), stat(receipt.differentBytes.path)]);
  if (srcInfo.ino > 0 && renamedInfo.ino > 0 && negativeInfo.ino > 0) {
    assert.notEqual(srcInfo.ino, renamedInfo.ino, 'same-byte fixture is a real separate file, not a hardlink');
    assert.notEqual(srcInfo.ino, negativeInfo.ino);
  }
  assert.equal(JSON.parse(await readFile(path.join(evidence, 'identity-controls.json'), 'utf8')).status, 'prepared-not-modelproof');
  assert.match(await readFile(path.join(root, 'fixtures', 'identity-controls.json'), 'utf8'), /prepared-not-modelproof/);
});
test('identity controls refuse a missing source, existing target or symlinked target and retain a failed receipt', async t => {
  const entry = { repoType: 'model', repo: 'unit/fake', revision: 'b'.repeat(40), file: 'source.bin', expectedBytes: 9, expectedSha256: sha256(Buffer.from('123456789')), legal: { licenseId: 'unit_fake-L1' } };
  const missingHome = await tempDir(t, 'sc26691-identity-missing-'), missingRoot = path.join(missingHome, 'root');
  await mkdir(missingRoot, { recursive: true });
  await assert.rejects(prepareIdentityControls({ root: missingRoot, entry, authority: identityAuthority(), evidenceDir: path.join(missingHome, 'evidence'), runId: '2', runAttempt: '1', allowUnitFake: true }), /missing verified private source/);
  assert.equal(JSON.parse(await readFile(path.join(missingHome, 'evidence', 'identity-controls.json'), 'utf8')).status, 'failed');
  const existingHome = await tempDir(t, 'sc26691-identity-existing-'), bytes = Buffer.from('123456789');
  const existingRoot = path.join(existingHome, 'root');
  await identitySource(existingHome, entry, bytes);
  const existingTarget = path.join(existingRoot, 'fixtures', 'independent-layout', 'unrelated-name.safetensors');
  await mkdir(path.dirname(existingTarget), { recursive: true }); await writeFile(existingTarget, 'existing bytes');
  await assert.rejects(prepareIdentityControls({ root: existingRoot, entry, authority: identityAuthority(), evidenceDir: path.join(existingHome, 'evidence'), runId: '3', runAttempt: '1', allowUnitFake: true }), /refusing to overwrite an existing identity-control target/);
  assert.equal(await readFile(existingTarget, 'utf8'), 'existing bytes');
  assert.equal(JSON.parse(await readFile(path.join(existingHome, 'evidence', 'identity-controls.json'), 'utf8')).status, 'failed');
  const symlinkHome = await tempDir(t, 'sc26691-identity-symlink-'), outside = path.join(await tempDir(t, 'sc26691-identity-outside-'), 'foreign.bin');
  await writeFile(outside, 'foreign bytes');
  const symlinkRoot = path.join(symlinkHome, 'root');
  await identitySource(symlinkHome, entry, bytes);
  const symlinkTarget = path.join(symlinkRoot, 'fixtures', 'independent-layout', 'different-bytes.safetensors');
  await mkdir(path.dirname(symlinkTarget), { recursive: true }); await symlink(outside, symlinkTarget);
  await assert.rejects(prepareIdentityControls({ root: symlinkRoot, entry, authority: identityAuthority(), evidenceDir: path.join(symlinkHome, 'evidence'), runId: '4', runAttempt: '1', allowUnitFake: true }), /symlink/);
  assert.equal(await readFile(outside, 'utf8'), 'foreign bytes', 'symlinked foreign bytes untouched');
  assert.equal(JSON.parse(await readFile(path.join(symlinkHome, 'evidence', 'identity-controls.json'), 'utf8')).status, 'failed');
});

test('failed requested identity controls leave the stage non-complete with a failed controls receipt', async t => {
  const bytes = Buffer.from('controls-missing-source'), base = await loadCasebook();
  const book = { ...base, stagingAssets: { tiny: [{ repo: 'unit/fake', revision: 'b'.repeat(40), file: 'tiny.bin', expectedBytes: bytes.length, expectedSha256: sha256(bytes), legal: { licenseId: 'unit_fake-license' } }] } };
  const files = selectStageFiles(book, ['tiny']);
  const hfHome = await tempDir(t, 'sc26691-controls-fail-'), evidence = path.join(hfHome, 'evidence');
  await assert.rejects(runStage({ request: { asset_sets: ['tiny'], prepare_identity_controls: true, license_authority: authorityFor(files) }, backend: 'mlx', book, env: stageEnv(hfHome, { HF_HUB_CACHE: path.join(hfHome, 'no-hub') }), platform: 'darwin', fetchImpl: fakeTransport(() => streamResponse([bytes])), evidenceDir: evidence, allowUnitFake: true, probe: async () => ({ kind: 'unit_fake probe' }) }), /missing verified private source/);
  const receipt = await readReceiptAt(path.join(hfHome, 'evidence'));
  assert.equal(receipt.status, 'failed');
  assert.equal(receipt.files.length, 1, 'per-HF-file completions stay separate from derived fixtures');
  assert.equal(receipt.identityControls?.status, 'failed');
  const controls = JSON.parse(await readFile(path.join(hfHome, 'evidence', 'identity-controls.json'), 'utf8'));
  assert.equal(controls.status, 'failed'); assert.equal(controls.derivedFixtures, 2);
});

function tinySafetensors(tensors) {
  const header = { __metadata__: { format: 'pt' } };
  const chunks = []; let offset = 0;
  for (const tensor of tensors) {
    header[tensor.name] = { dtype: tensor.dtype, shape: tensor.shape, data_offsets: [offset, offset + tensor.data.length] };
    chunks.push(tensor.data); offset += tensor.data.length;
  }
  const text = Buffer.from(JSON.stringify(header));
  const padding = (8 - (text.length % 8)) % 8;
  const padded = Buffer.concat([text, Buffer.alloc(padding, 0x20)]);
  const length = Buffer.alloc(8); length.writeBigUInt64LE(BigInt(padded.length));
  return { buffer: Buffer.concat([length, padded, ...chunks]), headerLength: padded.length, header: JSON.parse(padded.toString('utf8')) };
}
test('negative control mutates a quantized I8 .weight byte while header and descriptor stay valid', async t => {
  const weight = Buffer.from([1, 2, 3, 4, 5, 6]), descriptor = Buffer.from([9, 9]), model = tinySafetensors([
    { name: 'model.layers.0.weight', dtype: 'I8', shape: [6], data: weight },
    { name: 'model.comfy_quant', dtype: 'U8', shape: [2], data: descriptor },
    { name: 'model.scales', dtype: 'U8', shape: [1], data: Buffer.from([7]) },
  ]);
  const entry = { repoType: 'model', repo: 'unit/fake', revision: 'b'.repeat(40), file: 'w4a8.safetensors', expectedBytes: model.buffer.length, expectedSha256: sha256(model.buffer), legal: { licenseId: 'unit_fake-L1' } };
  const hfHome = await tempDir(t, 'sc26691-safetensors-'), sourcePath = await identitySource(hfHome, entry, model.buffer);
  const root = path.join(hfHome, 'root');
  const receipt = await prepareIdentityControls({ root, entry, authority: identityAuthority(), evidenceDir: path.join(hfHome, 'evidence'), runId: '6', runAttempt: '1', allowUnitFake: true, now: () => 2000 });
  const mutation = receipt.differentBytes.mutation;
  assert.equal(mutation.kind, 'quantized-weight-byte');
  assert.equal(mutation.tensor, 'model.layers.0.weight'); assert.equal(mutation.dtype, 'I8');
  assert.equal(mutation.offset, 8 + model.headerLength + mutation.data_offsets[0] + mutation.relativeIndex);
  const sourceBytes = await readFile(sourcePath), negativeBytes = await readFile(receipt.differentBytes.path);
  assert.equal(negativeBytes.length, sourceBytes.length);
  assert.notDeepEqual(negativeBytes, sourceBytes);
  assert.deepEqual(negativeBytes.subarray(0, 8 + model.headerLength), sourceBytes.subarray(0, 8 + model.headerLength), 'header/descriptor unchanged');
  const descriptorStart = 8 + model.headerLength + model.header['model.comfy_quant'].data_offsets[0];
  assert.deepEqual(negativeBytes.subarray(descriptorStart, descriptorStart + descriptor.length), descriptor, 'trailing U8 descriptor unchanged');
  assert.equal(negativeBytes[mutation.offset], sourceBytes[mutation.offset] ^ 1, 'only the quantized weight byte flipped');
  assert.notEqual(sha256(negativeBytes), sha256(sourceBytes), 'full SHA differs for the API expectedSHA refusal');
  assert.equal(sha256(sourceBytes), sha256(model.buffer), 'shared source unchanged');
});
test('negative control never uses a non-weight, label, or excluded tensor byte', async t => {
  const model = tinySafetensors([
    { name: 'model.comfy_quant', dtype: 'U8', shape: [2], data: Buffer.from([1, 2]) },
    { name: 'model.scales', dtype: 'I8', shape: [2], data: Buffer.from([3, 4]) },
    { name: 'config', dtype: 'U8', shape: [2], data: Buffer.from([5, 6]) },
    { name: 'model.layers.0.bias', dtype: 'I8', shape: [2], data: Buffer.from([7, 8]) },
  ]);
  const entry = { repoType: 'model', repo: 'unit/fake', revision: 'b'.repeat(40), file: 'no-weight.safetensors', expectedBytes: model.buffer.length, expectedSha256: sha256(model.buffer), legal: { licenseId: 'unit_fake-L1' } };
  const hfHome = await tempDir(t, 'sc26691-noweight-');
  await identitySource(hfHome, entry, model.buffer);
  const root = path.join(hfHome, 'root');
  await assert.rejects(prepareIdentityControls({ root, entry, authority: identityAuthority(), evidenceDir: path.join(hfHome, 'evidence'), runId: '7', runAttempt: '1', allowUnitFake: true }), /no quantized \.weight I8\/U8 tensor/);
  await assert.rejects(prepareIdentityControls({ root, entry, authority: identityAuthority(), evidenceDir: path.join(hfHome, 'evidence'), runId: '7', runAttempt: '1', allowUnitFake: false, fixtureOffset: 0 }), /unit_fake only/);
});
async function stagedRootFixture(t, label = 'ref-fixtures') {
  const home = await tempDir(t, `sc26691-${label}-`), root = path.join(home, 'root');
  await mkdir(root, { recursive: true });
  await writeFile(path.join(root, 'sc26691-owner.json'), JSON.stringify({ kind: 'h3-finetune-stage-owner', root: path.resolve(root), runId: '101', runAttempt: '1', backend: 'mlx' }));
  await writeFile(path.join(root, 'h3-stage-receipt.json'), JSON.stringify({ kind: 'h3-finetune-stage-receipt', status: 'completed' }));
  return { home, root, evidence: path.join(home, 'evidence') };
}
function referenceExecute({ failAsset = null } = {}) {
  const calls = [];
  const execute = async (file, args, options) => {
    calls.push({ file, args, cwd: options.cwd });
    if (args.includes('-version')) return { code: 0, stdout: `unit_fake ${path.basename(file)} 1.0\n`, stderr: '' };
    if (path.basename(file) === 'ffprobe') {
      const audio = args.includes('a:0');
      // The mock answers with EXACTLY the fields the probe asked for, the way the real ffprobe
      // does: a field the probe did not request is absent from the output. A mock that invents
      // an extra field hides a production check that reads it.
      const askedIndex = args.indexOf('-show_entries');
      const asked = askedIndex >= 0 ? String(args[askedIndex + 1] ?? '').replace(/^stream=/, '') : '';
      const values = {
        codec_type: kindFor(args),
        codec_name: 'unit_fake',
        width: '576',
        height: '320',
        nb_read_frames: '1',
        duration: '0.1',
        avg_frame_rate: '24/1',
        r_frame_rate: '24/1',
        sample_rate: '32000',
        channels: 1,
      };
      const stream = {};
      for (const key of asked.split(',').filter(Boolean)) if (key in values) stream[key] = values[key];
      return { code: 0, stdout: JSON.stringify({ streams: [stream] }), stderr: '' };
    }
    const outputName = args.at(-1);
    if (outputName && !args.includes('null') && outputName !== '-') {
      if (failAsset && args.join(' ').includes(failAsset)) return { code: 2, stdout: '', stderr: 'unit_fake recipe failure' };
      await writeFile(path.join(options.cwd, outputName), Buffer.from(`unit_fake fixture ${outputName}`));
    }
    return { code: 0, stdout: '', stderr: '' };
  };
  execute.calls = calls;
  return execute;
}
function kindFor(args) { return args.some(value => value.startsWith('a:')) ? 'audio' : 'video'; }
test('reference fixtures run only published ffmpeg recipes into cc0 with real hashes/probes and no model claim', async t => {
  const { root, evidence } = await stagedRootFixture(t), execute = referenceExecute();
  const tools = { ffmpeg: '/unit_fake/bin/ffmpeg', ffprobe: '/unit_fake/bin/ffprobe' };
  const env = { PATH: '/unit_fake/bin', RUNNER_NAME: 'nax-macos', H3_STAGE_BACKEND: 'mlx' };
  const receipt = await prepareReferenceFixtures({ root, evidenceDir: evidence, book: await loadCasebook(), execute, tools, allowUnitFake: true, env, platform: 'darwin', now: () => 0 });
  assert.equal(receipt.kind, 'conditioning-reference-fixtures'); assert.equal(receipt.status, 'prepared-not-modelproof');
  assert.equal(receipt.proof_kind, 'unit_fake'); assert.equal(receipt.campaign_proof, false); assert.match(receipt.meaning, /not model output/);
  assert.equal(receipt.fixtureCount, 5); assert.equal(receipt.expectedFixtureCount, 5);
  assert.equal(receipt.sourceCasebook.sha256, await casebookSha256());
  assert.equal(receipt.outputRoot, path.join(root, 'conditioning', 'cc0'));
  assert.equal(receipt.tools.ffmpegVersion, 'unit_fake ffmpeg 1.0');
  const expected = ['image_a', 'image_b', 'clip_a', 'clip_b', 'tone'];
  assert.deepEqual(receipt.fixtures.map(f => f.id), expected);
  for (const fixture of receipt.fixtures) {
    assert.equal(fixture.command[0], '/unit_fake/bin/ffmpeg');
    assert.equal(path.dirname(fixture.outputPath), path.join(root, 'conditioning', 'cc0'));
    assert.ok(outputInfoIsPositive(receipt, fixture));
    assert.equal(await hashFileSha256(fixture.outputPath), fixture.sha256);
    assert.ok(fixture.probe.streams.length > 0); assert.equal(fixture.exit.code, 0);
    assert.match(fixture.provenance, /CC0-1\.0.*no model output/);
  }
  assert.ok(receipt.fixtures.filter(f => f.kind === 'video').every(f => f.decode?.code === 0), 'clips are fully CPU-decoded');
  const recipeCalls = execute.calls.filter(call => call.args.includes('lavfi'));
  assert.equal(recipeCalls.length, 5);
  assert.ok(recipeCalls.every(call => call.cwd === receipt.outputRoot), 'cwd is the new cc0 output dir');
  assert.ok(execute.calls.some(call => call.args.includes('-version')));
  const evidenceNames = await readdir(evidence);
  assert.deepEqual(evidenceNames.sort(), ['reference-fixtures.json']);
  const evidenceReceipt = JSON.parse(await readFile(path.join(evidence, 'reference-fixtures.json'), 'utf8'));
  assert.equal(evidenceReceipt.status, 'prepared-not-modelproof'); assert.equal(evidenceReceipt.campaign_proof, false);
  assert.match(await readFile(path.join(root, 'conditioning', 'reference-fixtures.json'), 'utf8'), /prepared-not-modelproof/);
});
function outputInfoIsPositive(receipt, fixture) {
  return receipt.fixtures.includes(fixture) && receipt.fixtureCount === 5 && receipt.fixtures.every(f => f.sizeBytes > 0 && /^[a-f0-9]{64}$/.test(f.sha256));
}
test('reference fixtures refuse real injected execution, incomplete stage ownership and existing outputs', async t => {
  const { root, evidence } = await stagedRootFixture(t, 'ref-guard-'), book = await loadCasebook();
  const tools = { ffmpeg: '/unit_fake/ffmpeg', ffprobe: '/unit_fake/ffprobe' };
  const env = { PATH: '/unit_fake/bin', RUNNER_NAME: 'nax-macos', H3_EXPECTED_RUNNER: 'nax-macos', H3_STAGE_BACKEND: 'mlx', H3_TERMINAL_ROUTE:'h3-terminal-mlx', GITHUB_ACTIONS: 'true', GITHUB_EVENT_NAME: 'workflow_dispatch', GITHUB_SHA: sourceSha, GITHUB_RUN_ID: '101', GITHUB_RUN_ATTEMPT: '1' };
  await assert.rejects(prepareReferenceFixtures({ root, evidenceDir: evidence, execute: referenceExecute(), tools, allowUnitFake: false, env, platform: 'darwin' }), /unit_fake only/);
  await assert.rejects(prepareReferenceFixtures({ root, evidenceDir: evidence, book, allowUnitFake: false, env, platform: 'darwin' }), /cannot inject book data/);
  const running = await stagedRootFixture(t, 'ref-running-');
  await writeFile(path.join(running.root, 'h3-stage-receipt.json'), JSON.stringify({ kind: 'h3-finetune-stage-receipt', status: 'running' }));
  await assert.rejects(prepareReferenceFixtures({ root: running.root, evidenceDir: running.evidence, book, execute: referenceExecute(), tools, allowUnitFake: true, env, platform: 'darwin' }), /completed stage receipt/);
  await mkdir(path.join(root, 'conditioning', 'cc0'), { recursive: true });
  await assert.rejects(prepareReferenceFixtures({ root, evidenceDir: evidence, book, execute: referenceExecute(), tools, allowUnitFake: true, env, platform: 'darwin' }), /existing conditioning\/cc0/);
});
test('reference fixture failure retains failure metadata in evidence and never claims model proof', async t => {
  const { root, evidence } = await stagedRootFixture(t, 'ref-fail-'), tools = { ffmpeg: '/unit_fake/ffmpeg', ffprobe: '/unit_fake/ffprobe' };
  const env = { PATH: '/unit_fake/bin', RUNNER_NAME: 'nax-macos', H3_STAGE_BACKEND: 'mlx' };
  await assert.rejects(prepareReferenceFixtures({ root, evidenceDir: evidence, book: await loadCasebook(), execute: referenceExecute({ failAsset: 'testsrc2=size=576x320:rate=24:duration=2' }), tools, allowUnitFake: true, env, platform: 'darwin', now: () => 0 }), /exited 2|failed/);
  const receipt = JSON.parse(await readFile(path.join(evidence, 'reference-fixtures.json'), 'utf8'));
  assert.equal(receipt.status, 'failed'); assert.equal(receipt.campaign_proof, false); assert.equal(receipt.proof_kind, 'unit_fake');
  assert.ok(receipt.fixtureCount < 5); assert.ok(receipt.error?.message);
});
test('the weights guard refuses device, UNC and verbatim aliases of E: and links onto a weights root', async t => {
  const win = { platform: 'win32', env: {}, resolveLinks: false };
  // Text: every other namespace that can name E: is refused; a plain or \\?\X:\ drive path is normalised.
  for (const value of ['\\\\.\\E:\\x', '\\\\localhost\\E$\\x', '\\\\?\\UNC\\host\\E$\\x', '//localhost/E$/x', '//./E:/x', '\\\\?\\Volume{00000000-0000-0000-0000-000000000000}\\x', '\\\\??\\E:\\x', 'E:x', 'relative\\x']) {
    assert.throws(() => assertWritePathOffWeights(value, win), /H3_WRITE_PATH_NOT_PLAIN/, value);
  }
  for (const value of ['//?/E:/x', 'e:/x', '\\\\?\\E:\\x', 'E:/huggingface/sc26691-private']) assert.throws(() => assertWritePathOffWeights(value, win), /H3_WRITE_ON_WEIGHTS_VOLUME/, value);
  assert.equal(assertWritePathOffWeights('\\\\?\\D:\\w', win), 'D:\\w');
  assert.equal(assertWritePathOffWeights('D:/w/x', win), 'D:\\w\\x');
  // Links, on the host's own path semantics: a junction/symlink in OS temp onto a temp "weights" root
  // (never the real E:). Nothing is written through the link.
  const root = await mkdtemp(path.join(os.tmpdir(), 'sc26691-weights-guard-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const weights = path.join(root, 'weights'), link = path.join(root, 'innocent');
  await mkdir(path.join(weights, 'hub'), { recursive: true });
  await symlink(weights, link, process.platform === 'win32' ? 'junction' : 'dir');
  const envFor = hfRoot => (process.platform === 'win32' ? { HF_HUB_CACHE: path.join(hfRoot, 'hub') } : { HF_HOME: hfRoot, HOME: root });
  assert.throws(() => assertWritePathOffWeights(path.join(link, 'campaign', 'not-yet-created'), { env: envFor(weights) }), /H3_WRITE_UNDER_WEIGHTS_ROOT.*resolves to/, 'nearest existing ancestor is resolved through the link');
  assert.throws(() => assertWritePathOffWeights(link, { env: envFor(weights) }), /H3_WRITE_UNDER_WEIGHTS_ROOT/);
  assert.throws(() => assertWritePathOffWeights(path.join(weights, 'x'), { env: envFor(link) }), /H3_WRITE_UNDER_WEIGHTS_ROOT/, 'a weights root named through a link is compared as resolved');
  assert.equal(assertWritePathOffWeights(path.join(root, 'work', 'x'), { env: envFor(weights) }), path.join(root, 'work', 'x'));
  assert.doesNotThrow(() => assertWritePathOffWeights(path.join(link, 'x'), { env: envFor(weights), resolveLinks: false }), 'text alone cannot see the link: resolution is what catches it');
});
