import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { mkdtemp, mkdir, readFile, writeFile, rm, readdir, rename } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { preparationPlan, validatePreparationApproval, preparationEnvironment, preparationTransport, assertPreparationWorkers, redactPreparation, discoverPreparationInventory, runPreparation, advancePreparationClaimClock, waitPreparationDownload, PREPARATION_PROGRESS_LEASE_MS, boundedPreparationLog, safeDownloadDir, safeRepoDirName, readmeCardLicense, validatePreparationRuntimeIdentity, PreparationFailure, safePreparationFailure, parseSafePreparationFailure } from './flux3-image-klein-preparation.mjs';
import { loadProductTargets, ROOT } from './flux3-image-klein-terminal-plan.mjs';
import { verifyPreparationBuild, verifyTerminalBuild } from './flux3-image-klein-terminal-build.mjs';
import { assertClaimedWithin } from './flux3-image-klein-app-lifecycle.mjs';
const sha = value => createHash('sha256').update(value).digest('hex');
const sourceSha = 'a'.repeat(40), inferenceMainSha = /git = "https:\/\/github\.com\/SceneWorks\/inference", rev = "([a-f0-9]{40})"/.exec(await readFile(path.join(ROOT, 'Cargo.toml'), 'utf8'))[1];
const request = { backend: 'candle', sourceSha, inferenceMainSha, modelIds: ['flux3_klein_base'] };
const sources = await loadProductTargets();
const plan = preparationPlan(request, sources);
function admission(p = plan, root = path.resolve(tmpdir())) {
  return { actualProvision: true, userProvisionApprovalId: 'fixture-provision-approval', admissionId: 'fixture-cpu-admission', approvedBy: 'fixture-user', runner: 'fixture-host', runnerAccount: 'fixture-account', preparationPlanSha256: p.planSha256, sourceSha, inferenceMainSha, inferenceMainTipSha: inferenceMainSha, inferenceRepository: ROOT, backend: 'candle', outputRoot: root, hfHome: path.join(root, 'hf'), binary: path.join(root, 'sealed-api.exe'), binarySourceSha: sourceSha, binarySha256: 'b'.repeat(64), binaryBuildReceipt: { path: path.join(root, 'producer-receipt.json'), root, sha256: 'b'.repeat(64) }, preparationBuildReceiptBinding: {}, timeoutSeconds: 60, diskBudgetBytes: p.downloadBytes + 30 * 1024 ** 3, downloadApprovals: p.repositories.map(r => ({ repository: r.repository, resolvedRevision: r.resolvedRevision, userDownloadApprovalId: 'fixture-download', wholeRepository: true, licenseAcknowledged: true, accountGateAcknowledged: true, diskAcknowledged: true, downloadBytes: r.downloadBytes, runnerAccount: 'fixture-account', hfHome: path.join(root, 'hf') })) };
}
test('preparation plan is source-only, fixed, zero-generation and immutable', () => {
  assert.equal(plan.generationAttempts, 0); assert.equal(plan.conversions, false); assert.equal(plan.candidateAuthority, 'OBSERVED_NOT_GPU_APPROVED');
  assert.equal(plan.downloadBytes, 62068840528); assert.deepEqual(plan.repositories[0].files, []);
  assert.deepEqual(plan.repositories.map(r => r.modelId), ['flux3_image_klein_pup', 'flux3_klein_base']);
  assert.deepEqual(plan.repositories[1].dependencies, ['flux3_image_klein_pup']);
  assert.deepEqual(JSON.parse(JSON.stringify(plan)), preparationPlan(JSON.parse(JSON.stringify(request)), sources), 'CLI plan JSON round trip retains exact immutable catalog identity');
  for (const mutation of [r => r.modelIds.push(r.modelIds[0]), r => r.modelIds.push('foreign-model'), r => r.token = 'forbidden', r => r.backend = 'cpu']) { const r = structuredClone(request); mutation(r); assert.throws(() => preparationPlan(r, sources)); }
  const changed = structuredClone(sources); changed.models.find(m => m.id === request.modelIds[0]).downloads[0].revision = 'f'.repeat(40); assert.throws(() => preparationPlan(request, changed));
  const altered = structuredClone(plan); altered.repositories[0].downloadBytes--; assert.throws(() => validatePreparationApproval(admission(), altered, sources));
});
test('every approval mutation refuses before environment, credential, build, spawn or HTTP reads', async () => {
  const mutations = [a => a.actualProvision = false, a => delete a.userProvisionApprovalId, a => a.preparationPlanSha256 = '0'.repeat(64), a => a.sourceSha = 'c'.repeat(40), a => a.diskBudgetBytes = 1, a => a.downloadApprovals.push(a.downloadApprovals[0]), a => a.preparationBuildReceiptBinding = null, a => a.approvedPlan = {}, a => a.extraEnv = {}];
  for (const field of ['wholeRepository', 'licenseAcknowledged', 'accountGateAcknowledged', 'diskAcknowledged']) mutations.push(a => a.downloadApprovals[0][field] = false);
  mutations.push(a => a.downloadApprovals[0].downloadBytes--, a => a.downloadApprovals[0].runnerAccount = 'foreign', a => a.downloadApprovals[0].hfHome = 'foreign');
  for (const mutate of mutations) { const a = admission(); mutate(a); let touches = 0; const deps = new Proxy({}, { ownKeys() { touches++; return []; } }); await assert.rejects(runPreparation(a, plan, sources, deps)); assert.equal(touches, 0, 'approval refusal must precede any execution dependency access'); }
});
test('preparation environment has only owned CPU/download authority and no ambient GPU/provider overrides', () => {
  const env = new Proxy({ PATH: 'fixture-path', CUDA_VISIBLE_DEVICES: '0,1', SCENEWORKS_GPU_ID: 'mlx', SCENEWORKS_UTILITY_JOBS: '1', HF_TOKEN: 'ambient-secret' }, { get(target, key) { assert.notEqual(key, 'HF_TOKEN'); return target[key]; } });
  const a = admission(), identity = { state: path.join(a.outputRoot, 'state-owned'), workerId: 'owned' }, child = preparationEnvironment(env, a, identity, 'fixture-token');
  assert.equal(child.CUDA_VISIBLE_DEVICES, '-1'); assert.equal(child.SCENEWORKS_GPU_ID, 'cpu'); assert.equal(child.SCENEWORKS_RUST_WORKER_GPU_ID, 'cpu'); assert.equal(child.SCENEWORKS_BACKEND_CANDLE_ENABLED, 'false'); assert.equal(child.SCENEWORKS_BACKEND_MLX_ENABLED, 'false'); assert.equal(child.SCENEWORKS_UTILITY_JOBS, 'model_download'); assert.equal(child.SCENEWORKS_RUN_UTILITY_INPROCESS, '1'); assert.equal(child.HF_TOKEN, 'fixture-token'); assert(!child.SCENEWORKS_WORKER_CHILD);
});
test('preparation allowlist refuses all generation/training/raw-job/conversion and foreign-job requests before HTTP', async () => {
  let calls = 0; const transport = preparationTransport({ call() { calls++; throw Error('must not call'); } }, plan.repositories);
  for (const route of ['/api/v1/images/generate', '/api/v1/training/jobs', '/api/v1/jobs', '/api/v1/models/flux3_klein_base/convert', '/api/v1/jobs/foreign/cancel', '/api/v1/models/foreign/download', '/api/v1/models/flux3_klein_base/download?variant=q4']) await assert.rejects(transport.call('POST', route, {}));
  await assert.rejects(transport.call('GET', '/api/v1/jobs/foreign')); assert.equal(calls, 0);
  await assert.rejects(transport.call('POST', '/api/v1/models/flux3_klein_base/download', { licenseAcknowledged: true, variant: 'q4' })); assert.equal(calls, 0);
});
test('worker registration refuses foreign, GPU, extra capabilities and missing CPU scope', () => {
  const worker = { id: 'owned-utility-0', gpuId: 'cpu', status: 'idle', capabilities: ['cpu', 'model_download'] }; assertPreparationWorkers([worker], 'owned', { idle: true });
  for (const mutation of [w => w.id = 'foreign', w => w.gpuId = '0', w => w.capabilities.push('image_generate'), w => w.capabilities = ['model_download'], w => w.capabilities.push('placeholder'), w => w.status = 'unhealthy']) { const w = structuredClone(worker); mutation(w); assert.throws(() => assertPreparationWorkers([w], 'owned')); }
  assert.throws(() => assertPreparationWorkers([worker, worker], 'owned'));
});
test('redaction covers encoded credentials, authorization and signed-storage URL query secrets', () => {
  const secret = 'fixture/token+private'; const result = redactPreparation(`token=${secret} encoded=${encodeURIComponent(secret)} Authorization: Bearer auth-private https://user:pass@storage.example/file?X-Amz-Signature=signed-private&token=query-private hf_fixture123`, [secret]);
  for (const value of [secret, encodeURIComponent(secret), 'auth-private', 'signed-private', 'query-private', 'user:pass', 'hf_fixture123']) assert(!result.includes(value));
});
test('preparation build authority cannot replace generation approved-plan binding, and resource receipts are refused', async () => {
  await assert.rejects(verifyTerminalBuild(admission()), /parent-approved concrete/);
  const a = admission(); a.preparationBuildReceiptBinding = null; await assert.rejects(verifyPreparationBuild(a));
});
async function fixture(t, { wrongWorker = false, corruptReceipt = false, publicPup = false, logFlood = false } = {}) {
  const root = await mkdtemp(path.join(tmpdir(), 'flux3-prep-fixture-')); t.after(() => rm(root, { recursive: true, force: true }));
  // Mirrors the real layouts: base/distilled ship LICENSE.md; the actual PUP
  // repository ships none and declares its license in the README card.
  const data = { 'config.json': Buffer.from('{"fixture":true}'), ...(publicPup ? { 'README.md': Buffer.from('---\nlicense: apache-2.0\nbase_model:\n- fixture\n---\n# Fixture card\n') } : { 'LICENSE.md': Buffer.from('FLUX Kommunity fixture license') }), 'model.safetensors': Buffer.from('fixture-weights-not-a-model') };
  const tinySources = structuredClone(sources), model = publicPup ? tinySources.upsampler : tinySources.models.find(m => m.id === 'flux3_klein_base');
  // This fixture isolates the one-primary factory/inventory contract. The actual
  // source closure and >300s API-created co-requisites are tested separately below.
  model.downloads = model.downloads.filter(d => !d.coRequisite);
  model.downloads[0].estimatedSizeBytes = Object.values(data).reduce((n, b) => n + b.length, 0);
  const p = preparationPlan({ ...request, modelIds: [model.id] }, tinySources), a = admission(p, root); await mkdir(a.hfHome); const row = p.repositories[0];
  const snapshot = path.join(a.hfHome, 'hub', `models--${safeRepoDirName(row.repository)}`, 'snapshots', row.resolvedRevision);
  const observed = { spawns: [], requests: [], credentialReads: 0, buildChecks: 0, kills: [], state: undefined, worker: undefined, snapshot, data, row };
  const job = { id: 'fixture-download-job', type: 'model_download', status: 'completed', payload: { modelId: row.modelId, repo: row.repository, revision: row.resolvedRevision, files: [] }, result: { modelId: row.modelId, repo: row.repository, path: snapshot, storage: 'huggingface_cache' } };
  const deps = {
    root: ROOT, env: { HOME: root, CUDA_VISIBLE_DEVICES: '-1' }, platform: 'win32', account: () => a.runnerAccount, host: () => a.runner,
    git: async (_, ...args) => args[0] === 'status' || args[0] === 'merge-base' ? '' : args.includes('HEAD') ? sourceSha : inferenceMainSha,
    verifyBuild: async () => { observed.buildChecks++; }, statfs: async () => ({ bavail: 1024 ** 3, bsize: 1024 }),
    readCredential: async () => { observed.credentialReads++; return 'fixture-private-token'; },
    authorize: async (repo, revision, token) => { assert.equal(token, publicPup ? undefined : 'fixture-private-token'); assert.equal(repo, row.repository); assert.equal(revision, row.resolvedRevision); return { httpStatus: 200, configSha256: sha(data['config.json']) }; }, sleep: async () => {},
    spawn: (_binary, args, options) => {
      assert.deepEqual(args, []); assert.equal(options.env.CUDA_VISIBLE_DEVICES, '-1'); assert.equal(options.env.SCENEWORKS_RUST_WORKER_GPU_ID, 'cpu'); assert.equal(options.env.SCENEWORKS_UTILITY_JOBS, 'model_download'); observed.spawns.push(options);
      const child = new EventEmitter(); Object.assign(child, { exitCode: null, signalCode: null, pid: 12345, stdout: new PassThrough(), stderr: new PassThrough() });
      observed.state = path.dirname(options.env.SCENEWORKS_DATA_DIR); observed.worker = { id: `${options.env.SCENEWORKS_WORKER_ID}-0`, gpuId: wrongWorker ? '0' : 'cpu', status: 'idle', capabilities: ['cpu', 'model_download'] }; job.workerId = observed.worker.id;
      child.kill = signal => {
        observed.kills.push(signal); child.signalCode = signal; child.emit('exit');
        child.stderr.write('fixture-shutdown-tail Authorization: Bearer fixture-private-token https://storage.example/final?sig=tail-fixture-secret\n');
        child.stdout.end(); child.stderr.end(); queueMicrotask(() => child.emit('close')); return true;
      };
      queueMicrotask(() => {
        child.stdout.write(JSON.stringify({ event: 'api_listening', address: '127.0.0.1:43123' }) + '\n');
        child.stderr.write('Authorization: Bearer fixture-private-token\nhttps://storage.example/file?X-Amz-Sign');
        child.stderr.write('ature=signed-fixture-secret&X-Amz-Security-Token=query-fixture-secret\n');
        if (logFlood) {
          const line = JSON.stringify({ event: 'fixture-info', message: 'x'.repeat(128) }) + '\n';
          for (let i = 0; i < 40000; i++) child.stdout.write(line);
        }
      });
      return child;
    },
    fetch: async (url, options) => {
      const route = new URL(url).pathname; observed.requests.push([options.method, route]); let value;
      if (route === '/api/v1/health') {
        const state = observed.state; await mkdir(path.join(state, 'data/cache'), { recursive: true }); await writeFile(path.join(state, 'data/cache/jobs.db'), 'fixture-owned-db');
        value = { status: 'ok', service: 'sceneworks-api', runtime: 'rust', readiness: { status: 'ready' }, authRequired: false, directories: { data: path.join(state, 'data'), config: path.join(state, 'config'), projects: path.join(state, 'data/projects'), jobsDb: path.join(state, 'data/cache/jobs.db') } };
      } else if (route === '/api/v1/workers') value = [observed.worker];
      else if (route === `/api/v1/models/${row.modelId}/download`) {
        assert.equal(options.method, 'POST'); assert.equal(observed.spawns.length, 1); assert.equal(observed.credentialReads, publicPup ? 0 : 1);
        await mkdir(snapshot, { recursive: true }); for (const [name, bytes] of Object.entries(data)) await writeFile(path.join(snapshot, name), bytes);
        // The native API/worker key the managed receipt directory by REPOSITORY.
        const markerRoot = path.join(observed.state, 'data/models', safeDownloadDir(row.repository)); await mkdir(markerRoot, { recursive: true });
        const receipt = { schemaVersion: 2, jobId: corruptReceipt ? 'foreign-job' : job.id, modelId: row.modelId, repo: row.repository, snapshotRevision: row.resolvedRevision, completedAt: '2026-10-20T00:00:00Z', manifestFiles: [], resolvedFiles: Object.keys(data) };
        await writeFile(path.join(markerRoot, '.sceneworks-download-complete.json'), JSON.stringify(receipt)); value = { ...job, status: 'queued' };
      } else if (route === `/api/v1/jobs/${job.id}`) value = job;
      // Real catalog shape: FLUX 3 runtimes list the snapshot; other models (PUP) the repo cache root.
      else if (route === '/api/v1/models') value = [{ id: row.modelId, installState: 'installed', installedPath: publicPup ? path.dirname(path.dirname(snapshot)) : snapshot }];
      else throw Error('unexpected fixture request');
      return { ok: true, status: 200, json: async () => value };
    },
  };
  return { root, a, p, tinySources, deps, observed, job };
}
test('actual CPU factory uses fake registered API, downloads without preinstalled packages and exports observed-only job-bound inventory', async t => {
  const f = await fixture(t); await assert.rejects(readFile(path.join(f.observed.snapshot, 'config.json')), /ENOENT/);
  const result = await runPreparation(f.a, f.p, f.tinySources, f.deps); const report = JSON.parse(await readFile(result.file, 'utf8'));
  assert.equal(f.observed.spawns.length, 1); assert.equal(f.observed.buildChecks, 1); assert.equal(f.observed.credentialReads, 1); assert.deepEqual(f.observed.kills, ['SIGTERM']);
  assert.deepEqual(f.observed.requests.filter(([method]) => method === 'POST'), [['POST', '/api/v1/models/flux3_klein_base/download']]);
  assert.equal(report.status, 'OBSERVED'); assert.equal(report.generationAttempts, 0); assert.equal(report.gpuExecutionApproved, false); assert.equal(report.inventories[0].authority, 'OBSERVED_NOT_GPU_APPROVED'); assert.equal(report.inventories[0].files.length, 3); assert.equal(report.inventories[0].jobId, f.job.id);
  const retained = await readdir(f.root); assert(!retained.some(name => /ledger|generation|training/.test(name)));
  for (const file of retained.filter(name => name.endsWith('.json') || name.endsWith('.log'))) {
    const evidence = await readFile(path.join(f.root, file), 'utf8');
    for (const secret of ['fixture-private-token', 'signed-fixture-secret', 'query-fixture-secret']) assert(!evidence.includes(secret), 'split-stream credential/signed URL must not be retained');
  }
});
test('wrong CPU registration blocks download and actual foreign-job receipt blocks candidate export', async t => {
  for (const options of [{ wrongWorker: true }, { corruptReceipt: true }]) {
    const f = await fixture(t, options); await assert.rejects(runPreparation(f.a, f.p, f.tinySources, f.deps));
    if (options.wrongWorker) assert(!f.observed.requests.some(([method]) => method === 'POST'));
    assert.deepEqual(f.observed.kills, ['SIGTERM']); assert(!(await readdir(f.root)).some(name => name.endsWith('-observed-inventories.json')));
  }
});
test('discovery never accepts a pre-existing receipt without an actual completed download', async t => {
  const f = await fixture(t); await assert.rejects(discoverPreparationInventory({ state: f.root, hfHome: f.a.hfHome, row: f.observed.row, job: { ...f.job, status: 'queued' } }));
});
test('observed inventory rejects receipt identity/schema/coverage, config, total-byte and containment mutations', async t => {
  const f = await fixture(t); await runPreparation(f.a, f.p, f.tinySources, f.deps);
  const marker = path.join(f.observed.state, 'data/models', safeDownloadDir(f.observed.row.repository), '.sceneworks-download-complete.json'), original = await readFile(marker, 'utf8');
  const args = { state: f.observed.state, hfHome: f.a.hfHome, row: f.observed.row, job: f.job, authorization: { configSha256: sha(f.observed.data['config.json']), httpStatus: 200 }, installed: { id: f.observed.row.modelId, installState: 'installed', installedPath: f.observed.snapshot } };
  for (const mutate of [r => r.schemaVersion = 1, r => r.jobId = 'foreign', r => r.repo = 'foreign/repo', r => r.snapshotRevision = '0'.repeat(40), r => r.manifestFiles = ['config.json'], r => r.resolvedFiles.pop(), r => r.resolvedFiles.push('../outside')]) {
    const changed = JSON.parse(original); mutate(changed); await writeFile(marker, JSON.stringify(changed)); await assert.rejects(discoverPreparationInventory(args));
  }
  await writeFile(marker, original);
  await assert.rejects(discoverPreparationInventory({ ...args, row: { ...args.row, downloadBytes: args.row.downloadBytes + 1 } }));
  await assert.rejects(discoverPreparationInventory({ ...args, authorization: { configSha256: '0'.repeat(64) } }));
  await assert.rejects(discoverPreparationInventory({ ...args, installed: { ...args.installed, installedPath: f.root } }));
  await writeFile(path.join(f.observed.snapshot, 'unexpected-extra-file'), 'unexpected'); await assert.rejects(discoverPreparationInventory(args));
});
const inventoryArgs = f => ({ state: f.observed.state, hfHome: f.a.hfHome, row: f.observed.row, job: f.job, authorization: { configSha256: sha(f.observed.data['config.json']), httpStatus: 200 }, installed: { id: f.observed.row.modelId, installState: 'installed', installedPath: f.observed.snapshot } });
const markerOf = f => path.join(f.observed.state, 'data/models', safeDownloadDir(f.observed.row.repository), '.sceneworks-download-complete.json');
// Add a genuinely downloaded extra file: on disk, in the job's receipt, and in the approved bytes.
async function addDownloadedFile(f, args, name, bytes) {
  await writeFile(path.join(f.observed.snapshot, name), bytes);
  const envelope = JSON.parse(await readFile(markerOf(f), 'utf8')); envelope.resolvedFiles.push(name); await writeFile(markerOf(f), JSON.stringify(envelope));
  return { ...args, row: { ...args.row, downloadBytes: args.row.downloadBytes + Buffer.byteLength(bytes) } };
}
test('repository directory ports match the shared cross-language slug contract that the Rust core, API and worker also assert', async () => {
  const contract = JSON.parse(await readFile(path.join(ROOT, 'tests/fixtures/rust_migration_contracts/repo_slugs.json'), 'utf8'));
  assert(contract.cases.length >= 9);
  for (const c of contract.cases) { assert.equal(safeDownloadDir(c.repo), c.safeDownloadDir, `safeDownloadDir drift for ${c.repo}`); assert.equal(safeRepoDirName(c.repo), c.safeRepoDirName, `safeRepoDirName drift for ${c.repo}`); }
  const all = preparationPlan({ ...request, modelIds: ['flux3_klein_base', 'flux3_klein', 'flux3_image_klein_pup'] }, sources);
  for (const r of all.repositories) assert(contract.cases.some(c => c.repo === r.repository), `${r.repository} must be pinned by the shared contract`);
  assert.notEqual(safeDownloadDir('black-forest-labs/flux-3-image-klein-pup'), 'flux3_image_klein_pup');
});
test('a receipt under the model-ID directory instead of the native repository directory is never discovered', async t => {
  const f = await fixture(t); await runPreparation(f.a, f.p, f.tinySources, f.deps); const args = inventoryArgs(f);
  await discoverPreparationInventory(args);
  const wrong = path.join(f.observed.state, 'data/models', f.observed.row.modelId); assert.notEqual(wrong, path.dirname(markerOf(f)));
  await mkdir(wrong, { recursive: true }); await rename(markerOf(f), path.join(wrong, '.sceneworks-download-complete.json'));
  await assert.rejects(discoverPreparationInventory(args), /ENOENT/);
});
test('snapshot authority is the derived pinned snapshot, equal to the completed job path, listed as snapshot or repo cache root only', async t => {
  const f = await fixture(t); await runPreparation(f.a, f.p, f.tinySources, f.deps); const args = inventoryArgs(f), repoRoot = path.dirname(path.dirname(f.observed.snapshot));
  assert.equal((await discoverPreparationInventory({ ...args, installed: { ...args.installed, installedPath: repoRoot } })).totalBytes, args.row.downloadBytes);
  const other = path.join(repoRoot, 'snapshots', '0'.repeat(40)); await mkdir(other, { recursive: true });
  for (const job of [{ ...f.job, result: { ...f.job.result, path: repoRoot } }, { ...f.job, result: { ...f.job.result, path: other } }, { ...f.job, result: { ...f.job.result, storage: 'mlx_local' } }, { ...f.job, result: {} }, { ...f.job, result: undefined }]) await assert.rejects(discoverPreparationInventory({ ...args, job }));
  for (const installedPath of [other, path.join(f.a.hfHome, 'hub'), path.dirname(repoRoot), f.a.hfHome]) await assert.rejects(discoverPreparationInventory({ ...args, installed: { ...args.installed, installedPath } }));
});
test('PUP real layout: no LICENSE file, README card license equal to the catalog license is recorded with source and digest', async t => {
  const f = await fixture(t, { publicPup: true }); const result = await runPreparation(f.a, f.p, f.tinySources, f.deps); const report = JSON.parse(await readFile(result.file, 'utf8'));
  assert.equal(f.observed.row.license, 'Apache-2.0');
  assert.deepEqual(report.inventories[0].license, { source: 'readme-card', path: 'README.md', sha256: sha(f.observed.data['README.md']), license: 'apache-2.0' });
  const args = { ...inventoryArgs(f), installed: { id: f.observed.row.modelId, installState: 'installed', installedPath: path.dirname(path.dirname(f.observed.snapshot)) } };
  await assert.rejects(discoverPreparationInventory({ ...args, row: { ...args.row, license: 'MIT' } }), /README card license/);
  await assert.rejects(discoverPreparationInventory({ ...args, row: { ...args.row, license: null } }), /license file or catalog-matching README/);
  const readme = path.join(f.observed.snapshot, 'README.md'), original = await readFile(readme);
  for (const text of ['---\nlicense: mit\n---\n', '# no front matter\nlicense: apache-2.0\n', '---\nmeta:\n  license: apache-2.0\n---\n', '---\nlicense:\n---\n']) {
    await writeFile(readme, text); await assert.rejects(discoverPreparationInventory({ ...args, row: { ...args.row, downloadBytes: args.row.downloadBytes - original.length + Buffer.byteLength(text) } }), /README card license/);
  }
  await writeFile(readme, original);
  const withFile = await addDownloadedFile(f, args, 'LICENSE', 'Apache License fixture');
  const preferred = await discoverPreparationInventory(withFile); assert.equal(preferred.license.source, 'file'); assert.equal(preferred.license.path, 'LICENSE');
});
test('README card license parser reads only the top-level front-matter key', () => {
  assert.equal(readmeCardLicense('---\nlicense: apache-2.0\nbase_model:\n- x\n---\n# Card'), 'apache-2.0');
  assert.equal(readmeCardLicense('---\r\nlicense: "Apache-2.0"\r\n---\r\n'), 'Apache-2.0');
  for (const text of ['license: apache-2.0', '# Title\n---\nlicense: mit\n---\n', '---\n  license: mit\n---\n', '---\nlicense:   \n---\n', '---\nlicense: mit\n']) assert.equal(readmeCardLicense(text), null, text);
});
test('auth failures never expose signed URLs or credential values and never launch', async t => {
  const f = await fixture(t); f.deps.authorize = async () => { throw Error('Authorization: Bearer fixture-private-token https://storage.example/file?signature=signed-fixture-secret'); };
  await assert.rejects(runPreparation(f.a, f.p, f.tinySources, f.deps), error => !/private-token|signed-fixture-secret|signature=/.test(error.message)); assert.equal(f.observed.spawns.length, 0);
});
test('public PUP-only preparation never reads a credential and still uses the dedicated CPU factory', async t => {
  const f = await fixture(t, { publicPup: true }); f.deps.readCredential = async () => { throw Error('public preparation read a credential'); };
  const result = await runPreparation(f.a, f.p, f.tinySources, f.deps); assert.equal(result.status, 'OBSERVED'); assert.equal(f.observed.credentialReads, 0); assert.equal(f.observed.spawns.length, 1); assert.equal(f.observed.spawns[0].env.HF_TOKEN, undefined);
});
test('source closure deduplicates PUP and requires every dependency approval before any I/O', async () => {
  const all = preparationPlan({ ...request, modelIds: ['flux3_klein_base', 'flux3_klein', 'flux3_image_klein_pup'] }, sources);
  assert.deepEqual(all.repositories.map(r => r.modelId), ['flux3_image_klein_pup', 'flux3_klein_base', 'flux3_klein']);
  assert.equal(all.downloadBytes, 105386490130);
  for (const backend of ['mlx', 'candle']) assert.deepEqual(preparationPlan({ ...request, backend }, sources).repositories.map(r => r.modelId), ['flux3_image_klein_pup', 'flux3_klein_base']);
  let touches = 0;
  const noIO = new Proxy({}, { ownKeys() { touches++; return []; } });
  const missing = admission(); missing.downloadApprovals = missing.downloadApprovals.filter(r => !r.repository.endsWith('-pup'));
  await assert.rejects(runPreparation(missing, plan, sources, noIO));
  const bad = structuredClone(sources); bad.models.find(m => m.id === 'flux3_klein_base').downloads.find(d => d.coRequisite).repo = 'unapproved/source';
  await assert.rejects(runPreparation(admission(), plan, bad, noIO));
  // A self-rehashed plan/approval cannot omit an API-created dependency.
  const forged = structuredClone(plan); forged.repositories = forged.repositories.filter(r => r.modelId !== 'flux3_image_klein_pup'); forged.repositories[0].dependencies = []; forged.downloadBytes = forged.repositories[0].downloadBytes;
  const body = { ...forged }; delete body.planSha256; forged.planSha256 = sha(JSON.stringify(body));
  await assert.rejects(runPreparation(admission(forged), forged, sources, noIO));
  assert.equal(touches, 0, 'unapproved dependency acceptance must be killed BEFORE any execution dependency/environment access');
});
async function prerequisiteFixture({ dependencyMutation, workerMutation, idle = false, frozenProgress = false, frozenHeartbeat = false, timeoutSeconds = 900, boundary = false, completeAfterMs = 400000 } = {}) {
  const start = Date.parse('2026-10-21T00:00:00Z'); let time = start, depReads = 0;
  const row = plan.repositories.find(r => r.modelId === 'flux3_klein_base'), pup = plan.repositories.find(r => r.modelId === 'flux3_image_klein_pup');
  const worker = { id: 'owned-utility-0', gpuId: 'cpu', capabilities: ['cpu', 'model_download'], status: 'busy', currentJobId: 'owned-dependency' };
  const primary = { id: 'owned-primary', type: 'model_download', status: 'queued', progress: 0, workerId: null, createdAt: new Date(start + (boundary ? 1000 : 0)).toISOString(), payload: { modelId: row.modelId, repo: row.repository, revision: row.resolvedRevision, files: [] } };
  const requests = [];
  const endpoint = { async call(method, route) {
    requests.push([method, route]);
    if (method === 'POST' && route === `/api/v1/models/${row.modelId}/download`) return structuredClone(primary);
    if (method === 'GET' && route === '/api/v1/workers') { const w = structuredClone(worker); if (idle) { w.status = 'idle'; w.currentJobId = null; } if (workerMutation) workerMutation(w); return [w]; }
    if (method === 'GET' && route === '/api/v1/jobs/owned-primary') return time - start >= completeAfterMs ? { ...primary, status: 'completed', progress: 1, workerId: worker.id } : structuredClone(primary);
    if (method === 'GET' && route === '/api/v1/jobs/owned-dependency') {
      depReads++; const job = { id: 'owned-dependency', type: 'model_download', status: 'downloading', workerId: worker.id, createdAt: new Date(start).toISOString(), progress: frozenProgress ? 0.2 : (time - start) / 1000000, lastHeartbeatAt: new Date(frozenHeartbeat ? start : time).toISOString(), payload: { modelId: row.modelId, repo: pup.repository, revision: pup.resolvedRevision, files: [] } };
      if (dependencyMutation) dependencyMutation(job); return job;
    }
    assert.fail('unexpected fake API request');
  } };
  const transport = preparationTransport(endpoint, plan.repositories, { workerId: 'owned', now: () => time });
  const accepted = await transport.call('POST', `/api/v1/models/${row.modelId}/download`, { licenseAcknowledged: true, variant: 'bf16' });
  return { transport, requests, depReads: () => depReads, wait: () => waitPreparationDownload({ transport, accepted, workerId: 'owned', timeoutSeconds, check() {}, disk: async () => {}, now: () => time, sleep: async ms => { time += ms; } }) };
}
test('actual preparation wait handles an approved prerequisite beyond 300 seconds with no additional POSTs', async () => {
  for (const boundary of [false, true]) {
    const f = await prerequisiteFixture({ boundary }); const job = await f.wait(); assert.equal(job.status, 'completed'); assert(f.depReads() > 300);
    assert.deepEqual(f.requests.filter(([method]) => method === 'POST'), [['POST', '/api/v1/models/flux3_klein_base/download']]); assert(f.transport.ownsJob('owned-dependency'));
  }
});
test('dependency lookup rejects wrong repository/revision/owner/type/time/files/conversion and GPU selection', async () => {
  for (const dependencyMutation of [j => j.payload.repo = 'unapproved/source', j => j.payload.revision = '0'.repeat(40), j => j.workerId = 'foreign', j => j.type = 'image_generate', j => j.createdAt = '2020-01-01T00:00:00Z', j => j.payload.files = ['one-file'], j => j.payload.localDerivation = {}, j => j.payload.localDerivation = null, j => j.payload.conversion = '', j => j.progress = 2, j => j.payload.modelId = 'foreign', j => j.status = 'failed']) {
    const f = await prerequisiteFixture({ dependencyMutation }); await assert.rejects(f.wait()); assert(!f.transport.ownsJob('owned-dependency'));
  }
  const f = await prerequisiteFixture({ workerMutation: w => w.gpuId = '0' }); await assert.rejects(f.wait()); assert.equal(f.depReads(), 0);
});
test('idle, heartbeat-only and percentage-only stalls retain the 300-second guard and global deadline', async () => {
  for (const options of [{ idle: true }, { frozenProgress: true }, { frozenHeartbeat: true }]) { const f = await prerequisiteFixture(options); await assert.rejects(f.wait(), /unclaimed after 300s/); }
  const stalled = await prerequisiteFixture({ completeAfterMs: 900000, dependencyMutation: j => { if (Date.parse(j.lastHeartbeatAt) >= Date.parse('2026-10-21T00:01:40Z')) j.progress = 0.1; } });
  await assert.rejects(stalled.wait(), /unclaimed after 300s/, 'an established progress lease must expire after a stall');
  const f = await prerequisiteFixture({ timeoutSeconds: 120 }); await assert.rejects(f.wait(), /deadline exceeded/);
});
test('removing actual claim-clock suspension kills the same long-prerequisite trace before any real I/O', () => {
  const run = advance => {
    const start = Date.parse('2026-10-21T00:00:00Z'), state = { firstSeen: start, lastNow: start, suspendedMs: 0, activeId: null, activeUntil: 0, dependencies: new Map() };
    const primary = { id: 'p', type: 'model_download', status: 'queued', workerId: null }, worker = { id: 'cpu', status: 'busy', currentJobId: 'd' };
    for (let seconds = 0; seconds <= 400; seconds++) advance(state, primary, worker, { id: 'd', type: 'model_download', status: 'downloading', workerId: 'cpu', progress: seconds / 1000, lastHeartbeatAt: new Date(start + seconds * 1000).toISOString() }, start + seconds * 1000);
  };
  assert.doesNotThrow(() => run(advancePreparationClaimClock));
  const actual = advancePreparationClaimClock.toString(), changed = actual.replace('state.suspendedMs += activeMs;', 'state.suspendedMs += 0;'); assert.notEqual(changed, actual);
  const mutant = new Function('assert', 'assertClaimedWithin', 'PREPARATION_PROGRESS_LEASE_MS', `return (${changed});`)(assert, assertClaimedWithin, PREPARATION_PROGRESS_LEASE_MS);
  assert.throws(() => run(mutant), /unclaimed after 300s/);
});
test('existing partial cache never substitutes for an ordinary downloader job and fresh observed receipt', async t => {
  const f = await fixture(t), blobs = path.join(f.a.hfHome, 'hub', `models--${safeRepoDirName(f.observed.row.repository)}`, 'blobs');
  await mkdir(blobs, { recursive: true }); const partial = path.join(blobs, 'fixture-owned.incomplete'); await writeFile(partial, 'partial-not-approved');
  const result = await runPreparation(f.a, f.p, f.tinySources, f.deps);
  assert.equal(f.observed.requests.filter(([method, route]) => method === 'POST' && route.endsWith('/download')).length, 1);
  const report = JSON.parse(await readFile(result.file, 'utf8')); assert.equal(report.inventories[0].jobId, f.job.id);
  assert.equal(await readFile(partial, 'utf8'), 'partial-not-approved', 'driver must not delete or promote partial blobs; only normal downloader owns verification/resume');
  assert(!report.inventories[0].files.some(file => file.path.endsWith('.incomplete')));
});
test('more than the log cap never aborts the real CPU factory path and retains first context plus redacted shutdown tail', async t => {
  const f = await fixture(t, { logFlood: true }), result = await runPreparation(f.a, f.p, f.tinySources, f.deps);
  assert.equal(result.status, 'OBSERVED'); assert.equal(f.observed.spawns.length, 1);
  const file = (await readdir(f.root)).find(name => name.endsWith('-cpu-api.log')), bytes = await readFile(path.join(f.root, file));
  assert(bytes.length <= 2 * 1024 ** 2); const log = bytes.toString();
  assert(log.includes('api_listening')); assert(log.includes('older log lines omitted')); assert(log.includes('fixture-shutdown-tail'));
  for (const secret of ['fixture-private-token', 'signed-fixture-secret', 'query-fixture-secret', 'tail-fixture-secret']) assert(!log.includes(secret));
});
test('bounded per-pipe lines redact split tokens/URLs and omit oversize lines without losing the final tail', () => {
  const log = boundedPreparationLog(['hf_dummySecretOnly']), stdout = {}, stderr = {};
  log.push(stdout, Buffer.from('Authorization: Bearer hf_dummy'));
  log.push(stderr, Buffer.from('independent-stream\n'));
  log.push(stdout, Buffer.from('SecretOnly https://storage.example/file?sig=dummy-query'));
  log.push(stdout, Buffer.from('-secret\n'));
  log.push(stdout, Buffer.from('x'.repeat(3 * 1024 ** 2)));
  log.push(stdout, Buffer.from('\nfinal-tail\n')); log.finish();
  const retained = log.text(); assert(Buffer.byteLength(retained) <= 2 * 1024 ** 2);
  assert(retained.includes('oversized log line omitted')); assert(retained.includes('final-tail'));
  assert(!retained.includes('hf_dummySecretOnly')); assert(!retained.includes('dummy-query-secret'));
});
test('wrong parent CUDA selection refuses before credentials, build verification, launch or network', async t => {
  const f = await fixture(t); f.deps.env.CUDA_VISIBLE_DEVICES = '1'; await assert.rejects(runPreparation(f.a, f.p, f.tinySources, f.deps), /PREP_RUNNER_IDENTITY_MISMATCH/); assert.equal(f.observed.credentialReads, 0); assert.equal(f.observed.buildChecks, 0); assert.equal(f.observed.spawns.length, 0); assert.equal(f.observed.requests.length, 0);
});

test('Mac named-runner alias can differ from OS hostname while actual runner/target/label are bound before side effects', async t => {
  const f = await fixture(t), p = preparationPlan({ ...f.p.request, backend: 'mlx' }, f.tinySources);
  Object.assign(f.a, { backend: 'mlx', preparationPlanSha256: p.planSha256, runner: 'nax-macos-2', runnerTarget: 'secondary', runnerLabel: 'rw-mage' });
  f.deps.platform = 'darwin'; f.deps.env.RUNNER_NAME = 'nax-macos-2'; f.deps.host = () => 'Michael-MBP.local';
  const result = await runPreparation(f.a, p, f.tinySources, f.deps);
  assert.equal(result.status, 'OBSERVED'); assert.equal(f.observed.credentialReads, 1); assert.equal(f.observed.spawns.length, 1);
  const report = JSON.parse(await readFile(result.file, 'utf8')); assert.equal(report.runner, 'nax-macos-2');
});
test('Mac wrong/missing actual Runner identity, target, label or alias refuse before build/credentials/network/spawn', async t => {
  for (const mutation of [
    (a,d) => { d.env.RUNNER_NAME = 'nax-macos'; }, (a,d) => { delete d.env.RUNNER_NAME; },
    a => { a.runnerTarget = 'primary'; }, a => { delete a.runnerTarget; }, a => { a.runnerLabel = 'nax'; },
    a => { a.runner = 'Michael-MBP'; }, a => { a.physicalHostName = 'a-different-approved-host'; },
  ]) {
    const f = await fixture(t), p = preparationPlan({ ...f.p.request, backend: 'mlx' }, f.tinySources);
    Object.assign(f.a, { backend: 'mlx', preparationPlanSha256: p.planSha256, runner: 'nax-macos-2', runnerTarget: 'secondary', runnerLabel: 'rw-mage' });
    Object.assign(f.deps, { platform: 'darwin', host: () => 'Michael-MBP.local' }); f.deps.env.RUNNER_NAME = 'nax-macos-2'; mutation(f.a,f.deps);
    await assert.rejects(runPreparation(f.a,p,f.tinySources,f.deps), /PREP_RUNNER_IDENTITY_MISMATCH/);
    assert.equal(f.observed.credentialReads,0); assert.equal(f.observed.buildChecks,0); assert.equal(f.observed.spawns.length,0); assert.equal(f.observed.requests.length,0);
  }
});
test('Windows preparation still binds the local OS hostname rather than GitHub Runner name', () => {
  const a = { backend:'candle',runner:'local-windows-host',runnerAccount:'account' }, io = { platform:'win32',env:{RUNNER_NAME:'different-ci-label'},account:()=> 'account',host:()=> 'local-windows-host' };
  validatePreparationRuntimeIdentity(a,io); assert.throws(()=>validatePreparationRuntimeIdentity({...a,runner:'different-ci-label'},io),/PREP_RUNNER_IDENTITY_MISMATCH/);
});
test('safe preparation errors expose only allowlisted code/phase; hostile/extra/raw/oversized log text never echoes', () => {
  const secret = 'DECLARED-PRIVATE-TEXT-DO-NOT-ECHO', good = safePreparationFailure(new PreparationFailure('runtime_identity'));
  assert.deepEqual(good,{schemaVersion:1,kind:'flux3-preparation-safe-error',phase:'runtime_identity',code:'PREP_RUNNER_IDENTITY_MISMATCH'});
  assert.deepEqual(parseSafePreparationFailure(JSON.stringify(good)),good);
  for(const text of [secret,JSON.stringify({...good,message:secret}),JSON.stringify({...good,code:secret}),JSON.stringify({...good,phase:secret}),JSON.stringify({...good,schemaVersion:'1'}),'x'.repeat(4097)]) {
    const safe = parseSafePreparationFailure(text); assert.equal(safe.code,'PREP_UNKNOWN_FAILURE'); assert(!JSON.stringify(safe).includes(secret)); assert.deepEqual(Object.keys(safe).sort(),['code','kind','phase','schemaVersion']);
  }
  assert(!JSON.stringify(safePreparationFailure(new Error(secret),'authorization')).includes(secret));
  const file = readFile(new URL('./flux3-terminal-host.mjs',import.meta.url),'utf8');
  return file.then(text=>{assert.match(text,/Math\.min\(size, 4096\)/);assert.match(text,/say\(parseSafePreparationFailure/);assert.match(text,/runnerTarget: dispatch\.runnerTarget, runnerLabel: dispatch\.runnerLabel/);});
});

test('prepare CLI emits only the safe phase/code envelope on malformed private input, never exception text or paths', async t => {
  const dir=await mkdtemp(path.join(tmpdir(),'flux3-safe-cli-'));t.after(()=>rm(dir,{recursive:true,force:true}));
  const input=path.join(dir,'private-admission.json'), marker='DECLARED_PRIVATE_ERROR_TEXT_NOT_LOGGED';await writeFile(input,marker);
  const {execFile}=await import('node:child_process'), {promisify}=await import('node:util');
  await assert.rejects(promisify(execFile)(process.execPath,[path.join(ROOT,'scripts/flux3-image-klein-terminal-driver.mjs'),'prepare','run',input,path.join(dir,'plan.json')],{env:{CUDA_VISIBLE_DEVICES:'-1',PATH:process.env.PATH,SystemRoot:process.env.SystemRoot},timeout:30000}), error=>{
    const value=JSON.parse(error.stdout.trim());assert.deepEqual(value,{schemaVersion:1,kind:'flux3-preparation-safe-error',phase:'admission',code:'PREP_UNKNOWN_FAILURE'});
    assert(!error.stdout.includes(marker)&&!error.stdout.includes(input));assert(!error.stderr.includes(marker)&&!error.stderr.includes(input));return error.code===1;
  });
});
