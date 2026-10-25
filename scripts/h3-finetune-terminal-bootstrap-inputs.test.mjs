// Pure injected tiny streams/private fixtures only: no network, global caches, listeners or models.
import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, writeFile, readdir, realpath, rm, stat, symlink } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { stripJsoncComments } from './lib/jsonc.mjs';
import { prepareBootstrapInputs, publishedH3Config, readPinnedHfTree, selectComponentRows, verifyPrunedReferenceDeclaration } from './h3-finetune-terminal-bootstrap-inputs.mjs';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const blob = bytes => createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
const sourceSha = 'a'.repeat(40);
const catalog = JSON.parse(stripJsoncComments(await readFile(new URL('../config/manifests/builtin.models.jsonc', import.meta.url), 'utf8')));
const casebook = JSON.parse(await readFile(new URL('../docs/testing/sc-26691-h3-cases.json', import.meta.url), 'utf8'));
const codec = await readFile(new URL('../crates/sceneworks-core/src/minimax_h3_codec.rs', import.meta.url), 'utf8');
const licenseManifest = JSON.parse(await readFile(new URL('../apps/desktop/licenses/manifest.json', import.meta.url), 'utf8'));
const save = async (file, bytes) => { await mkdir(path.dirname(file), { recursive: true }); await writeFile(file, bytes); };
function response(bytes, headers = {}, status = 200) {
  return { status, headers: { get: name => headers[name.toLowerCase()] ?? null }, body: (async function* () {
    // Multiple tiny pieces exercise stream consumption, never response.json/arrayBuffer.
    const b = Buffer.from(bytes); for (let n = 0; n < b.length; n += 17) yield b.subarray(n, n + 17);
  })(), arrayBuffer() { throw new Error('whole response buffering forbidden'); }, json() { throw new Error('unbounded response JSON forbidden'); } };
}
async function snapshot(root) {
  const result = {};
  for (const item of await readdir(root, { withFileTypes: true })) {
    const file = path.join(root, item.name); result[item.name] = item.isDirectory() ? await snapshot(file) : hash(await readFile(file));
  }
  return result;
}
async function fixture(t, backend = 'mlx') {
  const temp = await realpath(await mkdtemp(path.join(os.tmpdir(), 'h3-bootstrap-inputs-unit-')));
  t.after(() => rm(temp, { recursive: true, force: true }));
  const swRoot = path.join(temp, 'source'), stageRoot = path.join(temp, 'sc26691-private/101-1'), nativeRoot = path.join(temp, 'native-source'), evidenceRoot = path.join(temp, 'evidence'), cacheRoot = path.join(temp, 'private-fixture-hub');
  await Promise.all([stageRoot, nativeRoot, evidenceRoot, cacheRoot].map(p => mkdir(p, { recursive: true })));
  const localCatalog = structuredClone({ models: catalog.models.filter(m => ['minimax_h3', 'minimax_h3_ref'].includes(m.id)) });
  await save(path.join(swRoot, 'config/manifests/builtin.models.jsonc'), JSON.stringify(localCatalog));
  await save(path.join(swRoot, 'apps/desktop/licenses/manifest.json'), JSON.stringify({ components: licenseManifest.components.filter(c => c.id === 'minimax-h3') }));
  const book = structuredClone(casebook), refBodies = new Map();
  for (const file of Object.keys(book.numeric['pruned-interpolation'].referenceSha256)) {
    const bytes = Buffer.from(`# unit_fake pinned source fixture: ${file}\n`); refBodies.set(file, bytes); book.numeric['pruned-interpolation'].referenceSha256[file] = hash(bytes);
  }
  await save(path.join(swRoot, 'docs/testing/sc-26691-h3-cases.json'), JSON.stringify(book));
  const producer = Buffer.from(`REVISION = "${book.numeric['pruned-interpolation'].referenceRevision}"\nSOURCES = ${JSON.stringify(book.numeric['pruned-interpolation'].referenceSha256, null, 4)}\n`);
  await save(path.join(nativeRoot, 'scripts/h3_pruned_adaln_oracle.py'), producer);
  await save(path.join(swRoot, 'crates/sceneworks-core/src/minimax_h3_codec.rs'), codec);
  await save(path.join(stageRoot, 'sc26691-owner.json'), JSON.stringify({ kind: 'h3-finetune-stage-owner', root: stageRoot, runId: '101', runAttempt: '1', backend, codeSha: sourceSha }));
  const trees = new Map(), bodies = new Map();
  const add = (repo, revision, file, lfs = false) => {
    const key = `${repo}@${revision}`, bytes = Buffer.from(lfs ? `unit_fake weight bytes ${file}` : JSON.stringify({ unit_fake: file }));
    const row = { type: 'file', path: file, size: bytes.length, oid: blob(bytes), ...(lfs ? { lfs: { oid: hash(bytes), size: bytes.length } } : {}) };
    if (!trees.has(key)) trees.set(key, []);
    if (!trees.get(key).some(r => r.path === file)) trees.get(key).push(row);
    bodies.set(`${repo}/${revision}/${file}`, bytes);
  };
  for (const { rows } of selectComponentRows(localCatalog, [{ modelId: 'minimax_h3', tier: 'q4' }], backend)) for (const row of rows) for (const pattern of row.files) {
    const file = pattern.replace(/\*$/, 'config.json'); add(row.repo, row.revision, file);
    if (pattern.endsWith('/*')) add(row.repo, row.revision, pattern.replace(/\*$/, 'model.safetensors'), true);
  }
  const calls = [], acks = [];
  const fetchImpl = async (raw, options) => {
    const url = new URL(raw); calls.push({ url: raw, options });
    assert.ok(options.signal, 'all requests must be bounded');
    if (url.hostname === 'raw.githubusercontent.com') {
      const prefix = book.numeric['pruned-interpolation'].referenceUrlPrefix; assert.ok(raw.startsWith(prefix));
      const bytes = refBodies.get(raw.slice(prefix.length)); assert.ok(bytes, 'no arbitrary source URL'); return response(bytes);
    }
    assert.equal(url.hostname, 'huggingface.co');
    const identity = /^\/api\/models\/([^/]+\/[^/]+)\/revision\/([a-f0-9]{40})$/.exec(url.pathname);
    if (identity) return response(JSON.stringify({ id: identity[1], sha: identity[2] }));
    const tree = /^\/api\/models\/([^/]+\/[^/]+)\/tree\/([a-f0-9]{40})$/.exec(url.pathname);
    if (tree) { const rows = trees.get(`${tree[1]}@${tree[2]}`); assert.ok(rows, 'unknown catalog tree'); return response(JSON.stringify(rows)); }
    const content = /^\/([^/]+\/[^/]+)\/resolve\/([a-f0-9]{40})\/(.+)$/.exec(url.pathname); assert.ok(content, 'unknown metadata URL');
    const bytes = bodies.get(`${content[1]}/${content[2]}/${content[3]}`); assert.ok(bytes, 'unknown pinned content'); return response(bytes);
  };
  // Explicit synthetic fixture authority. Nothing under test invents acknowledgment.
  const rawAck = { acknowledged: true, includeSelectedComponents: true, unit_fake: 'test input only' };
  const acknowledgmentFor = async (entry, context) => {
    acks.push({ entry: structuredClone(entry), context: structuredClone(context) });
    if (context.reference) return { scope: 'existing-pinned-producer-source', acknowledged: rawAck.acknowledged };
    assert.equal(context.component, true); assert.ok(['minimax_h3', 'minimax_h3_ref'].includes(context.modelId));
    assert.ok(trees.has(`${entry.repo}@${entry.revision}`), 'source outside fixture-approved exact sources');
    return { sha256: entry.expectedSha256, acknowledged: rawAck.acknowledged, licenseId: entry.legal.licenseId, inheritedLicenseId: entry.legal.inheritedLicenseId, inheritedAcknowledged: rawAck.acknowledged, attribution: entry.legal.attributionRequired, origin: { kind: 'catalog-component', modelId: context.modelId, componentId: context.componentId } };
  };
  const args = { swRoot, stageRoot, evidenceRoot, backend, componentSelections: [{ modelId: 'minimax_h3', tier: 'q4' }, { modelId: 'minimax_h3_ref', tier: 'q4' }], compatibleBase: 'builtin:minimax_h3', nativeEvidence: { root: nativeRoot, sha: 'b'.repeat(40), productPin: 'c'.repeat(40) }, acknowledgmentFor, proofKind: 'unit_fake', env: { HF_HUB_CACHE: cacheRoot }, deps: { fetchImpl, maxRetries: 0, frozenProducer: { sha256: hash(producer) } } };
  return { ...args, args, temp, book, catalog: localCatalog, trees, bodies, calls, acks, rawAck, refBodies, cacheRoot };
}

test('fresh private root stages measured components/references and source-bound published config, deduplicating both bases', async t => {
  const f = await fixture(t), sourceBefore = await snapshot(f.swRoot), evidenceBefore = await snapshot(f.evidenceRoot), rawBefore = structuredClone(f.rawAck);
  const r = await prepareBootstrapInputs(f.args);
  assert.ok(r.componentRecords.length > 0); assert.equal(r.seedFiles.filter(s => s.mode === 'hardlink').length, r.componentRecords.length);
  for (const row of r.componentRecords) { assert.equal(row.actualBytes, (await stat(row.finalPath)).size); assert.equal(row.observedSha256, hash(await readFile(row.finalPath))); assert.ok(row.finalPath.startsWith(f.stageRoot + path.sep)); assert.equal(row.cache.hub, f.cacheRoot); }
  for (const model of ['minimax_h3', 'minimax_h3_ref']) {
    assert.ok(r.componentsByModel[model].some(c => c.id === 'text_encoder'));
    for (const c of r.componentsByModel[model]) { assert.equal(c.licenseAcknowledged, f.rawAck.acknowledged); assert.equal(c.sha256, hash(await readFile(c.path))); assert.ok(!/transformer(_ref)?\//.test(c.file)); }
  }
  assert.deepEqual(r.componentsByModel.minimax_h3.map(c => c.path), r.componentsByModel.minimax_h3_ref.map(c => c.path));
  assert.equal(r.inputs.filter(i => i.role === 'reference').length, 3);
  for (const [file, bytes] of f.refBodies) assert.deepEqual(await readFile(path.join(r.referenceDir, file)), bytes);
  assert.equal(r.config.sha256, hash(await readFile(r.config.path))); assert.deepEqual(r.config.dit, publishedH3Config(codec));
  assert.equal(r.config.provenance.sha256, hash(Buffer.from(codec))); assert.equal(r.config.provenance.codeSha, sourceSha);
  assert.notEqual(r.config.dit.num_attention_heads * r.config.dit.attention_head_dim, r.config.dit.hidden_size, 'published rectangular projections remain legal');
  for (const seed of r.seedFiles) {
    assert.ok(seed.input.startsWith(f.stageRoot + path.sep));
    if (seed.mode === 'hardlink') assert.match(seed.destination, /^hf\/hub\/models--[^/]+\/snapshots\/[a-f0-9]{40}\//);
    else { assert.equal(seed.mode, 'copy'); assert.match(seed.destination, /^hf\/hub\/models--[^/]+\/refs\/[a-f0-9]{40}$/); }
  }
  assert.deepEqual(await snapshot(f.swRoot), sourceBefore); assert.deepEqual(await snapshot(f.evidenceRoot), evidenceBefore); assert.deepEqual(f.rawAck, rawBefore);
  const firstBody = f.calls.findIndex(c => c.url.includes('/resolve/'));
  assert.ok(firstBody >= 0); assert.ok(f.acks.filter(a => a.context.scopeOnly).length > 0);
});

test('private immutable cache refs contain exact revision bytes, deduplicate seeds and never become model components', async t => {
  const f = await fixture(t), cacheBefore = await snapshot(f.cacheRoot), r = await prepareBootstrapInputs(f.args);
  const refs = r.inputs.filter(i => i.role === 'cache-ref');
  assert.equal(refs.length, new Set(r.componentRecords.map(c => `${c.repo}@${c.revision}`)).size);
  assert.equal(new Set(r.seedFiles.map(s => s.destination)).size, r.seedFiles.length);
  for (const input of refs) {
    const revision = path.basename(input.path), bytes = await readFile(input.path);
    assert.equal(bytes.toString('ascii'), revision); assert.equal(bytes.length, 40); assert.equal(input.sha256, hash(bytes));
    const seed = r.seedFiles.find(s => s.input === input.path); assert.equal(seed.mode, 'copy');
    assert.ok(seed.destination.endsWith(`/refs/${revision}`));
    assert.ok(!r.componentRecords.some(c => c.finalPath === input.path));
    assert.ok(Object.values(r.componentsByModel).every(cs => cs.every(c => c.path !== input.path)));
  }
  assert.ok(!r.seedFiles.some(s => s.destination.endsWith('/refs/main')));
  assert.ok(!r.seedFiles.some(s => s.input === r.config.path), 'generated architecture stays input-only');
  assert.deepEqual(await snapshot(f.cacheRoot), cacheBefore);
});

test('private cache ref metadata cannot write through a symlink into a foreign cache', async t => {
  const f = await fixture(t), row = [...f.trees.keys()][0].split('@')[0];
  const repoRoot = path.join(f.stageRoot, 'hf/hub', `models--${row.replaceAll('/', '--')}`);
  await mkdir(repoRoot, { recursive: true }); await symlink(f.cacheRoot, path.join(repoRoot, 'refs'));
  const cacheBefore = await snapshot(f.cacheRoot);
  await assert.rejects(prepareBootstrapInputs(f.args), /private directory cannot be a symlink/);
  assert.deepEqual(await snapshot(f.cacheRoot), cacheBefore);
});

test('real catalog rows obey tier/platform and omit all soft predictors and stock DiT siblings', () => {
  for (const backend of ['mlx', 'cuda']) for (const tier of ['q4', 'q8', 'bf16']) for (const modelId of ['minimax_h3', 'minimax_h3_ref']) {
    const [{ rows }] = selectComponentRows(catalog, [{ modelId, tier }], backend);
    assert.ok(rows.some(r => r.componentId === 'text_encoder')); assert.ok(rows.some(r => r.componentId === 'video_vae')); assert.ok(rows.some(r => r.componentId === 'audio_vae'));
    assert.ok(rows.every(r => r.coRequisite && r.required !== 'soft' && r.componentId && (!r.variant || r.variant === tier)));
    assert.ok(rows.every(r => !r.files.some(f => /(^|\/)transformer(_ref)?\//.test(f))));
  }
});

test('missing/false authority and source/license scope mismatch refuse before any metadata or model work', async t => {
  for (const mode of ['missing', 'false', 'license', 'inherited', 'source', 'reference']) await t.test(mode, async t => {
    const f = await fixture(t), real = f.acknowledgmentFor;
    if (mode === 'missing') f.args.acknowledgmentFor = undefined;
    else f.args.acknowledgmentFor = async (entry, context) => {
      if (mode === 'source') throw new Error('source outside approved scope');
      const ack = await real(entry, context);
      if (mode === 'false') ack.acknowledged = false;
      if (mode === 'license') ack.licenseId = 'unknown-new-license';
      if (mode === 'inherited') ack.inheritedAcknowledged = false;
      if (mode === 'reference' && context.reference) ack.scope = 'unknown-source';
      return ack;
    };
    await assert.rejects(prepareBootstrapInputs(f.args)); assert.equal(f.calls.length, 0); assert.equal((await readdir(f.stageRoot)).length, 1);
  });
});

test('unknown base/tier, unpinned revision, catalog traversal and unknown provider fail before network', async t => {
  for (const mode of ['model', 'tier', 'revision', 'path', 'provider', 'duplicate']) await t.test(mode, async t => {
    const f = await fixture(t);
    if (mode === 'model') f.args.componentSelections[0].modelId = 'guessed_h3';
    if (mode === 'tier') f.args.componentSelections[0].tier = 'fp8';
    if (mode === 'duplicate') f.args.componentSelections.push(f.args.componentSelections[0]);
    const row = f.catalog.models[0].downloads.find(r => r.componentId === 'text_encoder' && r.variant === 'q4');
    if (mode === 'revision') row.revision = 'main';
    if (mode === 'path') row.files = ['../outside/*'];
    if (mode === 'provider') row.provider = 'unknown';
    await save(path.join(f.swRoot, 'config/manifests/builtin.models.jsonc'), JSON.stringify(f.catalog));
    await assert.rejects(prepareBootstrapInputs(f.args)); assert.equal(f.calls.length, 0);
  });
});

test('Git blob SHA1 is verified separately and LFS SHA256 is the actual file identity', async t => {
  const f = await fixture(t), r = await prepareBootstrapInputs(f.args);
  const config = r.componentRecords.find(c => c.file.endsWith('config.json')), weight = r.componentRecords.find(c => c.file.endsWith('.safetensors'));
  const bytes = await readFile(config.finalPath); assert.notEqual(config.observedSha256, blob(bytes)); assert.equal(config.observedSha256, hash(bytes));
  assert.equal(weight.observedSha256, hash(await readFile(weight.finalPath)));
});

test('GitSHA/LFSSHA confusion, missing actual sizes/content hashes and giant nonLFS weights never download model bodies', async t => {
  for (const mode of ['lfs-sha1', 'no-size', 'no-content', 'git-sha256', 'giant', 'blob-mismatch', 'missing-oid', 'lfs-size']) await t.test(mode, async t => {
    const f = await fixture(t), rows = [...f.trees.values()][0], weight = rows.find(r => r.lfs), config = rows.find(r => !r.lfs);
    if (mode === 'lfs-sha1') weight.lfs.oid = weight.oid;
    if (mode === 'no-size') delete weight.size;
    if (mode === 'no-content') delete weight.lfs;
    if (mode === 'git-sha256') config.oid = 'e'.repeat(64);
    if (mode === 'giant') { delete weight.lfs; weight.size = 34_000_000_000; }
    if (mode === 'blob-mismatch') config.oid = 'e'.repeat(40);
    if (mode === 'missing-oid') delete config.oid;
    if (mode === 'lfs-size') weight.lfs.size++;
    await assert.rejects(prepareBootstrapInputs(f.args)); assert.ok(!f.calls.some(c => c.url.endsWith('.safetensors')), 'never request whole weight without LFS identity');
    assert.equal((await readdir(f.stageRoot)).length, 1);
  });
});

test('HF source/revision mismatch, empty/missing pattern, traversal and unavailable metadata reject explicitly', async t => {
  for (const mode of ['source', 'revision', 'empty', 'pattern', 'traversal', 'unavailable']) await t.test(mode, async t => {
    const f = await fixture(t), real = f.args.deps.fetchImpl;
    f.args.deps.fetchImpl = async (url, options) => {
      if (mode === 'source' && url.includes('/revision/')) return response(JSON.stringify({ id: 'different/source', sha: /revision\/([a-f0-9]{40})/.exec(url)[1] }));
      if (mode === 'revision' && url.includes('/revision/')) return response(JSON.stringify({ id: /models\/([^/]+\/[^/]+)/.exec(url)[1], sha: 'e'.repeat(40) }));
      if (mode === 'unavailable') return response('denied', {}, 403);
      return real(url, options);
    };
    if (mode === 'empty') for (const rows of f.trees.values()) rows.splice(0);
    if (mode === 'pattern') for (const rows of f.trees.values()) rows.splice(0, rows.length, { type: 'file', path: 'irrelevant.json', size: 1, oid: 'a'.repeat(40) });
    if (mode === 'traversal') [...f.trees.values()][0][0].path = '../escape';
    await assert.rejects(prepareBootstrapInputs(f.args)); assert.ok(!f.calls.some(c => c.url.endsWith('.safetensors')));
  });
});

test('paginated expanded HF tree records every file and refuses off-source next links or duplicate pages', async t => {
  const revision = 'b'.repeat(40), repo = 'unit/metadata', base = `https://huggingface.co/api/models/${repo}/tree/${revision}`;
  for (const mode of ['ok', 'foreign', 'duplicate']) await t.test(mode, async () => {
    const calls = [];
    const fetchImpl = async url => { calls.push(url); if (url.includes('/revision/')) return response(JSON.stringify({ id: repo, sha: revision }));
      if (!url.includes('cursor=')) return response(JSON.stringify([{ type: 'file', path: 'a.json', size: 2, oid: 'a'.repeat(40) }]), { link: `<${mode === 'foreign' ? 'https://example.org/escape' : base + '?recursive=1&expand=1&cursor=next'}>; rel="next"` });
      return response(JSON.stringify([{ type: 'file', path: mode === 'duplicate' ? 'a.json' : 'b.json', size: 2, oid: 'b'.repeat(40) }]));
    };
    if (mode === 'ok') assert.deepEqual((await readPinnedHfTree(repo, revision, { fetchImpl })).map(r => r.path), ['a.json', 'b.json']);
    else await assert.rejects(readPinnedHfTree(repo, revision, { fetchImpl }));
    assert.ok(calls.every(u => u.startsWith('https://huggingface.co/')));
  });
});

test('known exact cache is probed read-only and private staged bytes are copied, never hardlinked', async t => {
  const f = await fixture(t), [key, rows] = [...f.trees.entries()][0], [repo, revision] = key.split('@'), row = rows.find(r => r.lfs);
  const shared = path.join(f.cacheRoot, `models--${repo.replaceAll('/', '--')}`, 'snapshots', revision, row.path), bytes = f.bodies.get(`${repo}/${revision}/${row.path}`);
  await save(shared, bytes); const before = await snapshot(f.cacheRoot), r = await prepareBootstrapInputs(f.args), privateFile = r.componentRecords.find(c => c.repo === repo && c.file === row.path);
  assert.equal(privateFile.cache.mode, 'cache-copy'); assert.notEqual((await stat(shared)).ino, (await stat(privateFile.finalPath)).ino);
  assert.deepEqual(await snapshot(f.cacheRoot), before); assert.ok(!f.calls.some(c => c.url.endsWith(`/resolve/${revision}/${row.path}`)));
});

test('parent receipt stays parent-owned, no hidden receipt edits or whole-artifact rehash by resolver', async t => {
  const f = await fixture(t), parent = path.join(f.stageRoot, 'h3-stage-receipt.json'), original = '{"unit_fake":"parent owns final persistence"}\n'; await writeFile(parent, original);
  let staged = 0; const { stageOneFile } = await import('./h3-finetune-terminal-stage.mjs');
  f.args.deps.stageOneFile = async args => { staged++; const r = await stageOneFile(args); return { ...r, unit_fake_stage_token: staged }; };
  const r = await prepareBootstrapInputs(f.args); assert.equal(r.componentRecords.length, staged); assert.equal(await readFile(parent, 'utf8'), original);
  assert.ok(r.componentRecords.every(record => record.unit_fake_stage_token));
  const source = await readFile(new URL('./h3-finetune-terminal-bootstrap-inputs.mjs', import.meta.url), 'utf8');
  assert.ok(!source.includes('hashFileSha256(')); assert.ok(!source.includes('execFile(')); assert.ok(!source.includes('createReadStream('));
});

test('source roots, owner mismatches, symlinked private directories and absent explicit base refuse', async t => {
  for (const mode of ['source-root', 'native-root', 'owner', 'symlink', 'base']) await t.test(mode, async t => {
    const f = await fixture(t);
    if (mode === 'source-root') f.args.stageRoot = f.swRoot;
    if (mode === 'native-root') f.args.stageRoot = f.nativeEvidence.root;
    if (mode === 'owner') { const file = path.join(f.stageRoot, 'sc26691-owner.json'), owner = JSON.parse(await readFile(file)); owner.root = f.swRoot; await writeFile(file, JSON.stringify(owner)); }
    if (mode === 'symlink') await symlink(f.swRoot, path.join(f.stageRoot, 'bootstrap-inputs'));
    if (mode === 'base') delete f.args.compatibleBase;
    await assert.rejects(prepareBootstrapInputs(f.args)); assert.ok(!f.calls.some(c => c.url.endsWith('.safetensors')));
  });
});

test('fixed source reference mismatch/foreign prefix refuses; downloaded text is never executed', async t => {
  for (const mode of ['hash', 'url', 'files']) await t.test(mode, async t => {
    const f = await fixture(t), ref = f.book.numeric['pruned-interpolation'];
    if (mode === 'hash') f.refBodies.set('transformer.py', Buffer.from('raise RuntimeError("never execute downloaded code")\n'));
    if (mode === 'url') ref.referenceUrlPrefix = 'https://example.org/arbitrary-code/';
    if (mode === 'files') ref.referenceSha256['../escape.py'] = 'f'.repeat(64);
    await save(path.join(f.swRoot, 'docs/testing/sc-26691-h3-cases.json'), JSON.stringify(f.book));
    await assert.rejects(prepareBootstrapInputs(f.args));
    if (mode !== 'hash') assert.equal(f.calls.length, 0);
  });
});

test('bounded sidecar mismatch refuses and no-content/oversized metadata never becomes proof', async t => {
  for (const mode of ['empty', 'oversized', 'incomplete']) await t.test(mode, async t => {
    const f = await fixture(t), real = f.args.deps.fetchImpl;
    f.args.deps.fetchImpl = async (url, options) => {
      if (mode === 'empty') return response('');
      if (mode === 'oversized') return { status: 200, body: (async function* () { yield Buffer.alloc(8 * 1024 * 1024 + 1); })() };
      if (url.includes('/resolve/') && url.endsWith('config.json')) return response('{}');
      return real(url, options);
    };
    await assert.rejects(prepareBootstrapInputs(f.args)); assert.ok(!f.calls.some(c => c.url.endsWith('.safetensors')));
  });
});

test('published configuration is parsed only from current Rust declaration, never guessed dimensions', () => {
  assert.equal(publishedH3Config(codec.replace('"num_layers": 50', '"num_layers": 49')).num_layers, 49);
  assert.throws(() => publishedH3Config('not the source declaration'));
  assert.throws(() => publishedH3Config('pub fn published() -> Self { Self(json!({"_class_name":"Other"})) }'));
});
